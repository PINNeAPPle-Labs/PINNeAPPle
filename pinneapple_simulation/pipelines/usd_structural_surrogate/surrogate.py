"""The operator: a PhysicsNeMo FNO on the grid, trained on the CalculiX samples, with its own validity envelope.

The map is ``(occupancy, signed distance) -> (u_x, u_y, u_z, von Mises)`` on the fixed grid of
:mod:`.campaign`. Geometry enters as a field, not as a parameter, so a design the campaign never solved is answered
by the same forward pass as one it did -- that is what makes the dashboard's sliders instant.

Two things are deliberate:

* **The fields are kept whole, not masked.** :func:`grid.fill_outside` gives the cells outside the part the value of
  the nearest cell inside it, so the network sees one continuous field and the ``occupancy`` channel still says
  where the part is. Masking instead puts a cliff at the surface that the spectral layers smear back over it.
* **The model carries an envelope.** ``Surrogate.valid(design)`` says whether a design lies inside the box the
  campaign explored; :class:`SurrogatePredictor` exposes that to ``virtual_ccx``, which refuses to answer outside it
  instead of extrapolating silently.

Training runs under a :class:`pinneapple_analysis.cost.equal_budget.ComputeBudget`, so two operators trained here
are comparable on compute and not only on epochs -- the same accounting the library uses elsewhere.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .design import DesignSpace
from .grid import GridSpec, sample_field, signed_distance
from .grid import occupancy as grid_occupancy

__all__ = ["FIELDS", "features", "GeometryFNO", "Surrogate", "SurrogatePredictor"]

FIELDS = ("ux", "uy", "uz", "vm")            # the four output channels, in this order
_IN = 2                                      # occupancy, signed distance (PhysicsNeMo adds the coordinates)


def features(occ: np.ndarray, sdf: np.ndarray, grid: GridSpec) -> np.ndarray:
    """(2, nx, ny, nz) float32 input of the operator: occupancy and signed distance scaled to the box."""
    d = float(np.abs(sdf).max()) or 1.0
    return np.stack([np.asarray(occ, np.float32), (np.asarray(sdf, np.float32) / d).astype(np.float32)])


class GeometryFNO(nn.Module):
    """PhysicsNeMo's 3-D ``FNO``. It predicts the fields in normalised units; :class:`Surrogate` carries the scale."""

    def __init__(self, n: Sequence[int], *, latent_channels: int = 16, num_fno_layers: int = 3,
                 num_fno_modes: int = 8, padding: int = 4, seed: int = 0):
        super().__init__()
        from physicsnemo.models.fno import FNO
        torch.manual_seed(seed)
        self.config = {"n": list(n), "in_channels": _IN, "out_channels": len(FIELDS),
                       "latent_channels": latent_channels, "num_fno_layers": num_fno_layers,
                       "num_fno_modes": num_fno_modes, "padding": padding, "seed": seed}
        self.net = FNO(in_channels=_IN, out_channels=len(FIELDS), dimension=3, coord_features=True,
                       latent_channels=latent_channels, num_fno_layers=num_fno_layers,
                       num_fno_modes=num_fno_modes, padding=padding)
        self.register_buffer("out_scale", torch.ones(len(FIELDS)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised fields. The physical scale is applied by :meth:`Surrogate.predict_fields`, so the network is
        trained against targets of order one and not against pascals."""
        return self.net(x)


@dataclass
class Surrogate:
    """A trained operator plus everything needed to use it: the grid, the design space, the field scales."""
    net: GeometryFNO
    grid: GridSpec
    space: DesignSpace
    out_scale: np.ndarray
    history: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ features for one sample
    def features_of(self, sample: dict[str, np.ndarray]) -> np.ndarray:
        occ = np.asarray(sample["occupancy"]).astype(bool)
        return features(occ, signed_distance(occ, self.grid), self.grid)

    def valid(self, design: dict[str, float], tol: float = 1e-9) -> bool:
        """Is ``design`` inside the box the training samples were drawn from?"""
        return self.space.contains(design, tol)

    # ------------------------------------------------------------------ prediction
    @torch.no_grad()
    def predict_fields(self, feats: np.ndarray) -> dict[str, np.ndarray]:
        """(2,nx,ny,nz) -> the four grid fields, in physical units."""
        self.net.eval()
        x = torch.from_numpy(np.asarray(feats, np.float32))[None]
        y = self.net(x)[0].cpu().numpy()
        scale = np.asarray(self.out_scale, float).reshape((-1,) + (1,) * (y.ndim - 1))
        return {name: (y * scale)[i] for i, name in enumerate(FIELDS)}

    @torch.no_grad()
    def predict_batch(self, feats: list[np.ndarray]) -> list[dict[str, np.ndarray]]:
        """Several designs in one forward pass -- one batched call instead of ``len(feats)`` calls.

        The dashboard's optimiser evaluates the design and its four finite-difference perturbations together; five
        separate forwards cost about three times what the same five in a batch do.
        """
        self.net.eval()
        x = torch.from_numpy(np.stack([np.asarray(f, np.float32) for f in feats]))
        y = self.net(x).cpu().numpy()
        scale = np.asarray(self.out_scale, float).reshape(1, -1, 1, 1, 1)
        y = y * scale
        return [{name: y[b, i] for i, name in enumerate(FIELDS)} for b in range(y.shape[0])]

    def predict_design(self, design: dict[str, float], geometry: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        """Predict a design. ``geometry`` (a solved sample's ``nodes``/``elements``/``element_type``) is used to
        rasterise the exact part; without it the occupancy is gone and the call refuses rather than guessing."""
        if not self.valid(design):
            raise ValueError(f"design {design} is outside the envelope {self.space.bounds}")
        if geometry is None:
            raise ValueError("predict_design needs geometry={'nodes','elements','element_type'} to build the "
                             "occupancy channel; use predict_fields(features) when you already have it")
        occ = grid_occupancy(geometry["nodes"], geometry["elements"], self.grid,
                             geometry.get("element_type", "C3D4"))
        return self.predict_fields(features(occ, signed_distance(occ, self.grid), self.grid))

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.net.state_dict(), "config": self.net.config}, p.with_suffix(".pt"))
        meta = {"grid": self.grid.as_dict(), "space": self.space.as_dict(), "fields": list(FIELDS),
                "out_scale": np.asarray(self.out_scale).tolist(), "metrics": self.metrics,
                "provenance": self.provenance, "config": self.net.config, "history": self.history[-200:]}
        p.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        return p.with_suffix(".pt")

    @classmethod
    def load(cls, path: str | Path) -> Surrogate:
        p = Path(path)
        meta = json.loads(p.with_suffix(".json").read_text())
        net = GeometryFNO(meta["config"]["n"], latent_channels=meta["config"]["latent_channels"],
                          num_fno_layers=meta["config"]["num_fno_layers"],
                          num_fno_modes=meta["config"]["num_fno_modes"],
                          padding=meta["config"]["padding"], seed=meta["config"]["seed"])
        blob = torch.load(p.with_suffix(".pt"), map_location="cpu", weights_only=True)
        net.load_state_dict(blob["state"])
        net.out_scale = torch.tensor(meta["out_scale"], dtype=torch.float32)
        net.eval()
        space = DesignSpace(plate_diameter=meta["space"]["plate_diameter_m"],
                            bounds={k: tuple(v) for k, v in meta["space"]["bounds"].items()},
                            reference_load=meta["space"]["reference_load_n"], asset=meta["space"].get("asset"))
        return cls(net=net, grid=GridSpec.from_dict(meta["grid"]), space=space,
                   out_scale=np.asarray(meta["out_scale"], float), history=meta.get("history", []),
                   metrics=meta.get("metrics", {}), provenance=meta.get("provenance", {}))


# ---------------------------------------------------------------------- training
def _relative_l2(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return (torch.linalg.vector_norm((pred - true).flatten(1), dim=1)
            / torch.linalg.vector_norm(true.flatten(1), dim=1).clamp_min(1e-12)).mean()


def train(samples: dict[str, np.ndarray], grid: GridSpec, space: DesignSpace, *, budget,
          val_fraction: float = 0.12, batch_size: int = 4, lr: float = 2e-3, weight_decay: float = 1e-5,
          seed: int = 0, threads: int = 8, model_kwargs: dict[str, Any] | None = None,
          log: Any | None = None, log_every: int = 0, init: Surrogate | None = None) -> Surrogate:
    """Train the operator on a campaign's arrays (``campaign.load_samples``) under ``budget`` (a ComputeBudget)."""
    from pinneapple_analysis.cost.equal_budget import (
        cosine_lr,
        set_lr,
        train_within_budget,
    )

    if threads:
        torch.set_num_threads(int(threads))
    n = samples["occupancy"].shape[1:]
    occ = samples["occupancy"].astype(bool)
    X = np.stack([features(occ[i], signed_distance(occ[i], grid), grid) for i in range(len(occ))]).astype(np.float32)
    Y = np.stack([np.stack([samples["u"][i][..., k] for k in range(3)] + [samples["vm"][i]]) for i in range(len(occ))])
    Y = Y.astype(np.float32)

    # one global scale per field (a per-sample scale would leak the target into the input side of the split).
    # The reduction runs over the *spatial* axes: a reshape to (-1, n_fields) would interleave them, because the
    # channel axis is not the last one.
    gscale = np.maximum(np.abs(Y).max(axis=tuple(range(2, Y.ndim))).max(axis=0), 1e-30).astype(np.float32)
    Yn = (Y / gscale[None, :, None, None, None]).astype(np.float32)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    n_val = max(1, int(round(val_fraction * len(X)))) if val_fraction > 0 else 0
    val_idx, tr_idx = order[:n_val], order[n_val:]

    if init is not None:                                       # continue a run instead of starting over
        net = init.net
        net.train()
        gscale = np.asarray(init.out_scale, np.float32)
        Yn = (Y / gscale[None, :, None, None, None]).astype(np.float32)
    else:
        net = GeometryFNO(n, seed=seed, **(model_kwargs or {}))
    net.out_scale = torch.tensor(gscale, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    Xt, Yt = torch.from_numpy(X[tr_idx]), torch.from_numpy(Yn[tr_idx])
    Xv, Yv = (torch.from_numpy(X[val_idx]), torch.from_numpy(Yn[val_idx])) if n_val else (None, None)
    state = {"order": rng.permutation(len(Xt)), "pos": 0, "seen": 0}

    def step(progress: float) -> float:
        net.train()
        state["n"] = state.get("n", 0) + 1
        if state["pos"] + batch_size > len(Xt):
            state["order"], state["pos"] = rng.permutation(len(Xt)), 0
        idx = torch.from_numpy(state["order"][state["pos"]:state["pos"] + batch_size])
        state["pos"] += batch_size
        state["seen"] += len(idx)
        set_lr(opt, cosine_lr(progress, lr))
        loss = _relative_l2(net(Xt[idx]), Yt[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log_every and state["n"] % log_every == 0:
            msg = f"step {state['n']}: loss {float(loss.detach()):.4f}"
            print(msg, flush=True) if log is None else log(msg)
        return float(loss.detach())

    @torch.no_grad()
    def evaluate() -> dict[str, float]:
        if Xv is None:
            return {}
        net.eval()
        return {"val_rel_l2": float(_relative_l2(net(Xv), Yv))}

    run = train_within_budget(step, budget, count_first_step=False, evaluate=evaluate)
    net.eval()
    losses = [h["loss"] for h in run.history if h.get("loss") is not None]
    sur = Surrogate(net=net, grid=grid, space=space, out_scale=gscale,
                    history=list(run.history),
                    metrics={"steps_total": int((init.metrics.get("steps_total", 0) if init is not None else 0)
                                                + int(getattr(budget, "steps", 0) or 0)),
                             "train_rel_l2": float(losses[-1]) if losses else float("nan"),
                             "val_rel_l2": evaluate().get("val_rel_l2", float("nan")),
                             "n_train": int(len(tr_idx)), "n_val": int(n_val), "n_samples": int(len(X)),
                             "budget": budget.to_dict(), "seen_samples": int(state["seen"])},
                    provenance={"sampling_seed": seed, "fields": list(FIELDS),
                                "split": {"train": tr_idx.tolist()[:50], "val": val_idx.tolist()}})
    if log is not None:
        log(f"trained: {sur.metrics}")
    return sur


# ---------------------------------------------------------------------- virtual_ccx
class SurrogatePredictor:
    """The trained operator behind ``virtual_ccx``'s predictor interface, so it can stand where ``ccx`` stands.

        pinneapple-ccx -i job --predictor pinneapple_simulation.pipelines.usd_structural_surrogate.surrogate:load_default

    The deck's own mesh builds the occupancy channel, so a pre-processor that exports any design of the family is
    answered without being told which design it is. Outside the trained box it raises ``OutOfEnvelope`` and the
    runner either falls back to the real ``ccx`` or fails loudly -- never a silent extrapolation.
    """

    def __init__(self, surrogate: Surrogate, path: str | Path = "surrogate"):
        self.s = surrogate
        self.path = Path(path)
        self.__name__ = "pinneapple_structural_surrogate"
        b = surrogate.space.bounds
        self.envelope = {"bbox": [[-surrogate.grid.size[0], -surrogate.grid.size[1], 0.0],
                                  [surrogate.grid.size[0], surrogate.grid.size[1], float(b["h"][1]) * 1.001]]}

    def __call__(self, model: Any, coords: np.ndarray) -> dict[str, np.ndarray]:
        from pinneapple_simulation.external_solvers.calculix.virtual_ccx import OutOfEnvelope
        coords = np.asarray(coords, float)
        if not np.all(self.s.grid.contains(coords)):
            raise OutOfEnvelope(f"deck nodes outside the grid box {self.s.grid.as_dict()}")
        ids, X = model.node_array() if hasattr(model, "node_array") else (None, coords)
        pos = {int(i): k for k, i in enumerate(ids)} if ids is not None else None
        conn = np.array([[pos[int(n)] for n in c] for c in model.elements.values()], int) if pos else None
        occ = grid_occupancy(X, conn, self.s.grid, model.element_type) if conn is not None else None
        if occ is None or occ.sum() == 0:
            raise OutOfEnvelope("the deck's mesh could not be voxelised onto the grid")
        fields = self.s.predict_fields(features(occ, signed_distance(occ, self.s.grid), self.s.grid))
        u = np.stack([sample_field(fields[c], self.s.grid, coords) for c in ("ux", "uy", "uz")], axis=-1)
        return {"DISP": u}


def load_default(spec: str = "") -> SurrogatePredictor:
    """``virtual_ccx.load_predictor`` entry point; ``spec`` is a model path or ``$PINNEAPPLE_SURROGATE``."""
    import os
    path = spec or os.environ.get("PINNEAPPLE_SURROGATE", "surrogate")
    return SurrogatePredictor(Surrogate.load(path), path)
