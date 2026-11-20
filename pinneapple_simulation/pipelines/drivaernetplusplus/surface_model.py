"""The surface-pressure surrogate the DrivAerNet++ contract asks for, and the loop that fits it.

Nothing here needs the 212 GB release: a case is a point cloud with one scalar per point, so the model is the same kind
of object as the AirfRANS coordinate network — a point-wise MLP over normalised coordinates plus a per-case
conditioning vector — and the loop is a data term weighted by the same dual areas the evaluator weights by, so a point
of high curvature counts for what it is worth on the surface rather than one point one vote.

The normalisation is the part that has to be right for a 3-D geometry: every case is translated to its bounding-box
centre, scaled by its longest side and divided by the dynamic head, because the raw coordinates run to metres and the
pressure runs to thousands of square metres per second squared, and a network that has to discover both scales from
the data is a network that spends its capacity on them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = ["CaseGeometry", "SurfacePressureNet", "normalise_case", "fit_surface_model", "predict_case"]


@dataclass
class CaseGeometry:
    """One case as the contract hands it over: the surface points, the truth and the area weights."""

    case_id: str
    points: np.ndarray            # (N, 3) metres
    pressure: np.ndarray          # (N,) m²/s²
    dual_area: np.ndarray         # (N,) m²
    parameters: np.ndarray | None = None    # the case's design vector, if the release gives one
    dynamic_head: float | None = None      # m²/s², from the case record — never from the truth at prediction time


def normalise_case(points: np.ndarray, pressure: np.ndarray, dynamic_head: float | None = None):
    """Centre and scale the geometry, and put the pressure in dynamic heads — the two scales a 3-D case carries.

    The scale is the longest side of the bounding box, which for these bodies is the car's length, so every case is
    expressed in body lengths and the network sees a comparable object whatever the variant's proportions. The dynamic
    head comes from the case record (`450.0 m²/s²` for the verified one); when it is not given, the pressure's own scale
    is used and the caller is told so by the return value."""
    points = np.asarray(points, float)
    pressure = np.asarray(pressure, float)
    centre = points.mean(axis=0)
    scale = float(np.ptp(points, axis=0).max()) or 1.0
    if dynamic_head is None:
        dynamic_head = float(np.sqrt((pressure ** 2).mean())) or 1.0
    coordinates = (points - centre) / scale
    return coordinates, pressure / dynamic_head, {"centre": centre, "scale": scale, "dynamic_head": dynamic_head}


def _torch():
    import torch

    return torch


class SurfacePressureNet:
    """A point-wise MLP over normalised coordinates with a per-case conditioning vector appended.

    ``conditioning`` is the design vector divided by its own scale; a case without one passes a single zero, which the
    first layer learns to ignore — the model then reduces to a shape-agnostic point regressor, which is the right
    fallback when the release does not carry the parameters."""

    def __init__(self, conditioning: int = 0, width: int = 256, depth: int = 3, bands: int = 4):
        torch = _torch()
        import torch.nn as nn

        self.bands, self.conditioning = bands, conditioning
        inputs = 3 + 3 * 2 * bands + max(conditioning, 1)
        layers, last = [], inputs
        for _ in range(depth):
            layers += [nn.Linear(last, width), nn.SiLU()]
            last = width
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)
        self.width, self.depth = width, depth

    def features(self, coordinates, conditioning=None):
        torch = _torch()
        bands = []
        for k in range(1, self.bands + 1):
            for axis in range(3):
                angle = (2.0 * np.pi * k) * coordinates[:, axis:axis + 1]
                bands += [torch.sin(angle), torch.cos(angle)]
        if conditioning is None:
            conditioning = torch.zeros(len(coordinates), max(self.conditioning, 1))
        return torch.cat([coordinates] + bands + [conditioning], dim=1)

    def __call__(self, coordinates, conditioning=None):
        return self.net(self.features(coordinates, conditioning))[:, 0]


def fit_surface_model(cases: Sequence[CaseGeometry], *, epochs: int = 50, batch: int = 65536, lr: float = 2e-3,
                      width: int = 256, depth: int = 3, bands: int = 4, seed: int = 0,
                      device: str | None = None) -> dict[str, Any]:
    """Fit the surrogate on the given cases, weighting every point by its dual area.

    The weighting is not decoration: the evalued metric is area-weighted, so an unweighted fit spends its capacity on
    the dense flat regions of the body and buys nothing on the metric."""
    torch = _torch()
    import torch.nn.functional as F

    torch.manual_seed(seed)
    device = device or ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    conditioning_size = 0 if cases[0].parameters is None else len(np.atleast_1d(cases[0].parameters))
    model = SurfacePressureNet(conditioning=conditioning_size, width=width, depth=depth, bands=bands)
    parameters = list(model.net.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-5)

    prepared = []
    for case in cases:
        coordinates, target, transform = normalise_case(case.points, case.pressure, case.dynamic_head)
        conditioning = None
        if case.parameters is not None:
            vector = np.atleast_1d(np.asarray(case.parameters, float))
            conditioning = vector / (np.abs(vector).max() or 1.0)
        prepared.append({"case_id": case.case_id,
                         "x": torch.from_numpy(coordinates).float(),
                         "y": torch.from_numpy(target).float(),
                         "w": torch.from_numpy(np.asarray(case.dual_area, float)).float(),
                         "c": None if conditioning is None else torch.from_numpy(conditioning).float(),
                         "transform": transform})

    history = []
    for epoch in range(1, epochs + 1):
        losses = []
        for case in prepared:
            index = torch.randint(0, len(case["x"]), (min(batch, len(case["x"])),))
            x = case["x"][index].to(device)
            y = case["y"][index].to(device)
            w = case["w"][index].to(device)
            c = None if case["c"] is None else case["c"].expand(len(index), -1).to(device)
            prediction = model(x, c)
            loss = (w * (prediction - y) ** 2).sum() / w.sum().clamp_min(1e-12)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return {"model": model, "history": history, "prepared": prepared,
            "config": {"epochs": epochs, "batch": batch, "lr": lr, "width": width, "depth": depth, "bands": bands,
                       "conditioning": conditioning_size, "device": device}}


def predict_case(model: SurfacePressureNet, case: CaseGeometry, *, batch: int = 262144) -> np.ndarray:
    """The case's pressure in the release's own units and native point order — what a submission carries."""
    torch = _torch()
    if case.dynamic_head is None:
        raise ValueError(f"{case.case_id}: predicting needs the case's dynamic head, not the truth's scale")
    coordinates, _, transform = normalise_case(case.points, case.pressure, case.dynamic_head)
    conditioning = None
    if case.parameters is not None:
        vector = np.atleast_1d(np.asarray(case.parameters, float))
        conditioning = torch.from_numpy(vector / (np.abs(vector).max() or 1.0)).float()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(coordinates), batch):
            block = torch.from_numpy(coordinates[start:start + batch]).float()
            c = None if conditioning is None else conditioning.expand(len(block), -1)
            outputs.append(model(block, c).cpu().numpy())
    return np.concatenate(outputs) * transform["dynamic_head"]
