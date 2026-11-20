"""The AhmedML surface surrogate: one point network over the boundary centroids, two heads, area-weighted fit.

A case is 1.2 M polygons and the two fields the surface metrics score are the pressure (a scalar in m²/s²) and the wall
shear (a three-component kinematic traction whose typical magnitude is 1.5·10⁻³ m²/s²). Those are 500× apart, so the
two live on separate heads normalised by their own scale — the same lesson the AirfRANS wall readout cost three runs to
learn: when the small quantity shares a head with the large one, the large one sets the step size and the small one does
not train.

The design vector is the eight geometric parameters the release publishes per case (`body-length`, `slant-angle-degrees`
and six more). A point network over coordinates alone cannot tell two bodies apart at the same location; the vector is
what turns the model from a shape-agnostic regressor into a surrogate over the Ahmed family.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = ["CaseSurface", "SurfaceFieldNet", "normalise_case", "fit_surface_model", "predict_case"]


@dataclass
class CaseSurface:
    """One case as the fit and the prediction need it: geometry, the published weights and the design vector."""

    case_id: str
    centroids: np.ndarray         # (C, 3) polygon centres, release units
    normals: np.ndarray           # (C, 3) outward unit normals
    area: np.ndarray              # (C,) published polygon areas, m²
    parameters: np.ndarray        # (8,) the release's geometric parameters
    pressure: np.ndarray | None = None        # (C,) m²/s² — never read at prediction time
    wall_shear: np.ndarray | None = None      # (C, 3) m²/s²


def normalise_case(centroids, normals, parameters) -> tuple[np.ndarray, dict[str, Any]]:
    """Coordinates on the body's bounding box, normals as they are, parameters divided by their own scale.

    Geometry only — nothing here touches the truth, so the same function runs at fit time and at prediction time and the
    two cannot disagree about the frame."""
    centroids = np.asarray(centroids, float)
    centre = 0.5 * (centroids.min(axis=0) + centroids.max(axis=0))
    scale = float(np.ptp(centroids, axis=0).max()) or 1.0
    coordinates = (centroids - centre) / scale
    vector = np.atleast_1d(np.asarray(parameters, float))
    return coordinates, {"centre": centre, "scale": scale, "parameter_scale": float(np.abs(vector).max() or 1.0)}


def _torch():
    import torch

    return torch


class SurfaceFieldNet:
    """A point MLP with two heads — the pressure and the wall-shear traction — over geometry and design vector."""

    def __init__(self, parameters: int = 8, width: int = 256, depth: int = 3, bands: int = 4):
        torch = _torch()
        import torch.nn as nn

        self.bands, self.parameters = bands, parameters
        inputs = 3 + 3 * 2 * bands + 3 + max(parameters, 1)
        layers, last = [], inputs
        for _ in range(depth):
            layers += [nn.Linear(last, width), nn.SiLU()]
            last = width
        self.trunk = nn.Sequential(*layers)
        self.head_pressure = nn.Linear(last, 1)
        self.head_shear = nn.Linear(last, 3)
        self.width, self.depth = width, depth

    def weights(self):
        """Every weight of the trunk and of both heads, in one list — what the optimiser and the clip act on."""
        return (list(self.trunk.parameters()) + list(self.head_pressure.parameters())
                + list(self.head_shear.parameters()))

    def to(self, device: str):
        self.trunk.to(device)
        self.head_pressure.to(device)
        self.head_shear.to(device)
        self.device = device
        return self

    def features(self, coordinates, normals, parameters):
        torch = _torch()
        bands = []
        for k in range(1, self.bands + 1):
            for axis in range(3):
                angle = (2.0 * np.pi * k) * coordinates[:, axis:axis + 1]
                bands += [torch.sin(angle), torch.cos(angle)]
        return torch.cat([coordinates] + bands + [normals, parameters], dim=1)

    def __call__(self, coordinates, normals, parameters):
        hidden = self.trunk(self.features(coordinates, normals, parameters))
        return self.head_pressure(hidden)[:, 0], self.head_shear(hidden)


def _conditioning(case: CaseSurface, transform: dict[str, Any]):
    vector = np.atleast_1d(np.asarray(case.parameters, float))
    return vector / transform["parameter_scale"]


def fit_surface_model(cases: Sequence[CaseSurface], *, epochs: int = 40, batch: int = 32768, lr: float = 2e-3,
                      width: int = 256, depth: int = 3, bands: int = 4, seed: int = 0,
                      points_per_case: int = 200_000, shear_weight: float = 1.0,
                      device: str | None = None) -> dict[str, Any]:
    """Fit on the given cases, weighting every polygon by its published area.

    The metric is area-weighted, so an unweighted fit spends its capacity on the dense flat panels and buys nothing on
    the score. ``shear_weight`` balances the two heads' normalised losses; 1.0 fits them as equals once each is divided
    by its own scale."""
    torch = _torch()
    device = device or ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(seed)

    prepared, shear_scale = [], []
    for case in cases:
        if case.pressure is None or case.wall_shear is None:
            raise ValueError(f"{case.case_id}: fitting needs the truth of both surface fields")
        shear_scale.append(np.linalg.norm(case.wall_shear, axis=1))
    shear_scale = float(np.median(np.concatenate(shear_scale))) or 1.0

    for case in cases:
        coordinates, transform = normalise_case(case.centroids, case.normals, case.parameters)
        index = np.arange(len(coordinates))
        if points_per_case and len(index) > points_per_case:
            index = np.random.default_rng(seed).choice(index, points_per_case, replace=False)
        prepared.append({
            "case_id": case.case_id,
            "x": torch.from_numpy(coordinates[index]).float(),
            "n": torch.from_numpy(np.asarray(case.normals, float)[index]).float(),
            "c": torch.from_numpy(_conditioning(case, transform)).float(),
            "y": torch.from_numpy(np.asarray(case.pressure, float)[index] / 0.5).float(),
            "t": torch.from_numpy(np.asarray(case.wall_shear, float)[index] / shear_scale).float(),
            "w": torch.from_numpy(np.asarray(case.area, float)[index]).float(),
        })

    model = SurfaceFieldNet(parameters=len(np.atleast_1d(cases[0].parameters)), width=width, depth=depth,
                            bands=bands).to(device)
    parameters = model.weights()
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-5)

    history = []
    for epoch in range(1, epochs + 1):
        losses, seen = 0.0, 0
        for case in prepared:
            take = min(batch, len(case["x"]))
            index = torch.randint(0, len(case["x"]), (take,))
            x, n = case["x"][index].to(device), case["n"][index].to(device)
            conditioning = case["c"].to(device)
            y, t, w = case["y"][index].to(device), case["t"][index].to(device), case["w"][index].to(device)
            w = w / w.sum().clamp_min(1e-12)
            pressure, shear = model(x, n, conditioning.expand(take, -1))
            loss = (w * (pressure - y) ** 2).sum() + shear_weight * (w[:, None] * (shear - t) ** 2).sum(axis=1).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses += float(loss.detach()) * take
            seen += take
        history.append(losses / max(seen, 1))
    return {"model": model, "history": history, "shear_scale": shear_scale,
            "config": {"epochs": epochs, "batch": batch, "lr": lr, "width": width, "depth": depth, "bands": bands,
                       "points_per_case": points_per_case, "shear_weight": shear_weight, "device": device,
                       "parameters": int(len(np.atleast_1d(cases[0].parameters))), "shear_scale": shear_scale}}


def predict_case(model: SurfaceFieldNet, case: CaseSurface, *, shear_scale: float, dynamic_head: float = 0.5,
                 batch: int = 262144) -> tuple[np.ndarray, np.ndarray]:
    """The case's pressure and wall shear in the release's own units and native entity order."""
    torch = _torch()
    device = getattr(model, "device", "cpu")
    coordinates, transform = normalise_case(case.centroids, case.normals, case.parameters)
    conditioning = torch.from_numpy(_conditioning(case, transform)).float().to(device)
    normals = np.asarray(case.normals, float)
    pressure_parts, shear_parts = [], []
    with torch.no_grad():
        for start in range(0, len(coordinates), batch):
            stop = start + batch
            x = torch.from_numpy(coordinates[start:stop]).float().to(device)
            n = torch.from_numpy(normals[start:stop]).float().to(device)
            p, s = model(x, n, conditioning.expand(len(x), -1))
            pressure_parts.append(p.cpu().numpy())
            shear_parts.append(s.cpu().numpy())
    return np.concatenate(pressure_parts) * dynamic_head, np.concatenate(shear_parts) * shear_scale
