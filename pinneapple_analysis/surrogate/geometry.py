"""Geometry-agnostic field surrogates: one interface for the models that learn on the geometry itself (#293).

The families that dominate the recent benchmarks — Transolver, GINO, the PointNet/RegDGCNN baselines, MeshGraphNet —
all take a discretised shape and return a field on it, and every one of them was written with its own input
convention. A study that wants to compare them ends up comparing the plumbing. This module fixes the plumbing once:

    sample  = GeometrySample(points, normals, conditions, fields, integrals)   # one simulation, reduced to its surface
    model   = build_geometry_surrogate("transolver")                          # one family, chosen by name
    history = model.fit(samples)                                              # train, whatever the family
    pred    = model.predict(points, normals, conditions)                      # -> fields + integrals
    report  = model.evaluate(held_out)                                        # -> relative L2, integral error, R²

Three decisions carry the interface:

- **A sample is a point set, not a grid.** The points may be any number, in any order, with any connectivity: the
  field is defined *at the points handed in*, and the same weights answer for a different count. That is what makes
  the model geometry-agnostic rather than a fixed-topology grid model.
- **The integrals are integrated from the predicted field**, by an ``integrate`` callable the case supplies, not by a
  second head. So a predicted drag can never disagree with the pressure field it came from; if the field is wrong, the
  integral is wrong in the same direction, which is the honest coupling.
- **A family that is not here says so, with the reason.** :data:`FAMILIES` lists every family the issue names:
  the implemented ones, and the open ones with the paper, the licence and what it would take — a caller reading the
  list never mistakes a stand-in for a model that is not there.

What is implemented and measured: ``transolver`` (Wu et al., ICML 2024 — physics-attention over learned slices, the
strong family on the car benchmarks) and ``pointnet`` (Qi et al., CVPR 2017 — shared MLP plus a global max pool, the
cheap baseline that is hard to beat on integrated quantities). ``gino`` and the ``meshgraphnet`` baseline are named
below, with what each would take.

The measured number, on the panel-method family of ``benchmarks/geometry_surrogate_interface.py``, is in
``benchmarks/_out/geometry_surrogate_interface.json`` and on ``docs/guides/geometry_surrogate_families.md``.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

__all__ = [
    "FAMILIES",
    "GeometryEvaluation",
    "GeometryPrediction",
    "GeometrySample",
    "GeometrySurrogate",
    "GeometryFamily",
    "available_families",
    "build_geometry_surrogate",
    "samples_from_panel",
]


# ----------------------------------------------------------------------------------------------- the data
@dataclass
class GeometrySample:
    """One simulation reduced to its surface: points, the field on them, and the integrated quantities.

    ``points`` is ``(N, 3)`` and the fields are per point (``(N,)`` or ``(N, C)``), keyed by name. ``normals`` and
    ``elements`` are optional per-point extras the model and the integrator may use (a surface normal, a panel length,
    a thickness); ``conditions`` is the global state of the run (angle of attack, Reynolds number) as scalars, given to
    every point. ``integrals`` holds the quantities the simulation reported for the whole body — the target of the
    integrals the model reconstructs from its own field.
    """

    points: np.ndarray
    fields: dict[str, np.ndarray] = field(default_factory=dict)
    normals: np.ndarray | None = None
    conditions: dict[str, float] = field(default_factory=dict)
    integrals: dict[str, float] = field(default_factory=dict)
    elements: dict[str, np.ndarray] = field(default_factory=dict)
    name: str = ""

    def __post_init__(self) -> None:
        self.points = np.ascontiguousarray(self.points, dtype=np.float64).reshape(-1, 3)
        if not len(self.points):
            raise ValueError("GeometrySample: at least one point is required")
        if self.normals is not None:
            self.normals = np.ascontiguousarray(self.normals, dtype=np.float64).reshape(-1, 3)
            if len(self.normals) != len(self.points):
                raise ValueError("GeometrySample: one normal per point")
        clean: dict[str, np.ndarray] = {}
        for k, v in self.fields.items():
            a = np.ascontiguousarray(v, dtype=float)
            a = a.reshape(-1) if a.ndim == 1 else a.reshape(len(self.points), -1)
            if len(a) != len(self.points):
                raise ValueError(f"GeometrySample: field {k!r} has {len(a)} values for {len(self.points)} points")
            clean[k] = a
        self.fields = clean
        self.conditions = {k: float(v) for k, v in self.conditions.items()}
        self.integrals = {k: float(v) for k, v in self.integrals.items()}

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def condition_keys(self) -> list[str]:
        return sorted(self.conditions)

    def subset(self, index: np.ndarray) -> GeometrySample:
        """The same sample on a subset of its points (subsampling for a batch, or a decimated surface)."""
        return GeometrySample(
            points=self.points[index],
            fields={k: v[index] for k, v in self.fields.items()},
            normals=None if self.normals is None else self.normals[index],
            conditions=dict(self.conditions), integrals=dict(self.integrals),
            elements={k: v[index] for k, v in self.elements.items()}, name=self.name)


@dataclass
class GeometryPrediction:
    """What a surrogate answers for one geometry: the field at every point given, and the integrals from it."""
    points: np.ndarray
    fields: dict[str, np.ndarray]
    integrals: dict[str, float] = field(default_factory=dict)


@dataclass
class GeometryEvaluation:
    """The measured error of a surrogate on held-out samples, per sample and summarised.

    ``relative_l2`` is the field error against the simulation (the benchmark convention), ``integral_error`` the
    absolute error of each integral, and ``rank_r2`` how well the model preserves the ordering of the designs by the
    first integral — a field error that does not reorder the designs costs nothing at the decision level.
    """
    per_sample: list[dict[str, Any]] = field(default_factory=list)

    @property
    def relative_l2(self) -> np.ndarray:
        return np.array([s["relative_l2"] for s in self.per_sample], dtype=float)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"samples": len(self.per_sample)}
        if not self.per_sample:
            return out
        l2 = self.relative_l2
        out["relative_l2_mean"] = float(np.mean(l2))
        out["relative_l2_worst"] = float(np.max(l2))
        integral_keys = [k for k in self.per_sample[0] if k.startswith("integral_error:")]
        for k in integral_keys:
            err = np.array([s[k] for s in self.per_sample], dtype=float)
            out[k] = {"mean": float(np.mean(err)), "worst": float(np.max(err))}
        if "rank_r2" in self.per_sample[0]:
            out["rank_r2"] = float(np.mean([s["rank_r2"] for s in self.per_sample]))
        return out

    def __repr__(self) -> str:
        s = self.summary()
        if not s.get("samples"):
            return "GeometryEvaluation(empty)"
        return (f"GeometryEvaluation({s['samples']} samples, relative L2 mean {s['relative_l2_mean']:.3f}, "
                f"worst {s['relative_l2_worst']:.3f})")


# ----------------------------------------------------------------------------------------------- the families
@dataclass(frozen=True)
class GeometryFamily:
    """One family in the catalogue: what it is, whether it can be built here, and what it would take if not."""
    name: str
    implemented: bool
    source: str
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "implemented": self.implemented, "source": self.source, "note": self.note}


FAMILIES: dict[str, GeometryFamily] = {
    "transolver": GeometryFamily(
        "transolver", True,
        "Wu et al., 'Transolver: A Fast Transformer Solver for PDEs on General Geometries', ICML 2024 (arXiv:2402.02366)",
        "physics-attention: every point is softly assigned to learned slices, attention runs on the slices, the result "
        "is scattered back. Cost O(N M + M^2), so it scales to the tens of thousands of surface points of a car. The "
        "library's implementation is pinneapple_neural.architectures.neural_operators.transolver and is wrapped, not "
        "reimplemented."),
    "pointnet": GeometryFamily(
        "pointnet", True,
        "Qi et al., 'PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation', CVPR 2017 "
        "(arXiv:1612.00593)",
        "shared per-point MLP, a global max pool, and the pooled code concatenated back to every point. The cheap "
        "baseline: no geometric message passing, so it is the row that shows what the attention buys."),
    "gino": GeometryFamily(
        "gino", False,
        "Li et al., 'Geometry-Informed Neural Operator for Large-Scale 3D PDEs' (GINO), NeurIPS 2023 (arXiv:2309.00583)",
        "OPEN. A graph encoder, a Fourier neural operator on a latent regular grid and a graph decoder. It needs a "
        "regridding step (the FNO part sees a grid, the geometry arrives as a graph) and the reference implementation "
        "pulls in torch-scatter and torch-geometric, which the library does not depend on. The interface here already "
        "carries what it needs — arbitrary points, per-point normals, a variable count — so a GINO backend implements "
        "the same three methods and nothing else changes."),
    "meshgraphnet": GeometryFamily(
        "meshgraphnet", False,
        "Pfaff et al., 'Learning Mesh-Based Simulation with Graph Networks', ICLR 2021 (arXiv:2010.03409)",
        "ALREADY IN THE LIBRARY, not wrapped here: pinneapple_neural.architectures.graphnn.MeshGraphNet (with #236's "
        "world-space edges) is the baseline the aero app runs. It is a fixed-topology model — it learns on the edges of "
        "one mesh — so putting it behind this interface would mean either re-triangulating every shape to a common "
        "connectivity (which is a different promise: 'geometry-agnostic' would become 'one topology') or rebuilding "
        "the graph per shape. Both are reasonable; neither is done, and wrapping it badly would make the comparison "
        "dishonest. It stays the reference row of the benchmark, trained on the shared mesh."),
}


def available_families(*, implemented_only: bool = False) -> dict[str, dict[str, Any]]:
    """The catalogue as data: every family the issue names, with whether it is here and why not."""
    return {k: v.as_dict() for k, v in FAMILIES.items() if v.implemented or not implemented_only}


def build_geometry_surrogate(family: str = "transolver", **kwargs: Any) -> GeometrySurrogate:
    """The family by name, or ``ValueError`` naming the ones that exist and the reason the requested one is open."""
    key = str(family).lower()
    if key not in FAMILIES:
        raise ValueError(f"unknown geometry-surrogate family {family!r}; known: {', '.join(sorted(FAMILIES))}")
    if not FAMILIES[key].implemented:
        raise NotImplementedError(
            f"the {key!r} family is not implemented here: {FAMILIES[key].note}")
    return {"transolver": TransolverGeometry, "pointnet": PointNetGeometry}[key](**kwargs)


# ----------------------------------------------------------------------------------------------- the interface
class GeometrySurrogate:
    """The interface every geometry-agnostic surrogate implements: ``fit``, ``predict``, ``evaluate``.

    A subclass provides ``_module`` (an ``nn.Module`` mapping ``(B, N, in_dim) -> (B, N, out_dim)``) and inherits the
    rest: feature construction from points/normals/conditions, normalisation, subsampling for the batch, the training
    loop, prediction at any point count, and the evaluation against held-out simulations.
    """

    family: str = "base"

    def __init__(self, *, fields: Sequence[str] | None = None, hidden: int = 128, depth: int = 4,
                 device: str | None = None, integrate: Callable[..., dict[str, float]] | None = None,
                 use_normals: bool = True, seed: int = 0):
        self.fields = tuple(fields) if fields else None            # set by fit when not given
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.device = device
        self.integrate = integrate
        self.use_normals = bool(use_normals)
        self.seed = int(seed)
        self.module: Any = None
        self.condition_keys: list[str] = []
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._y_mean: dict[str, float] = {}
        self._y_std: dict[str, float] = {}
        self.history: list[dict[str, float]] = []
        self.n_parameters = 0

    # ------------------------------------------------------------------------------ features
    @property
    def in_dim(self) -> int:
        """Columns the model sees per point: the coordinates, the normals when used, the conditions broadcast."""
        return 3 + (3 if self.use_normals else 0) + len(self.condition_keys)

    def features(self, sample: GeometrySample) -> np.ndarray:
        """The model's input for one sample: coordinates, optional normals, and the conditions broadcast per point."""
        parts = [sample.points]
        if self.use_normals:
            parts.append(sample.normals if sample.normals is not None else np.zeros_like(sample.points))
        if self.condition_keys:
            missing = [k for k in self.condition_keys if k not in sample.conditions]
            if missing:
                raise ValueError(f"{self.family}: the sample has no condition {missing} the model was fitted on")
            parts.append(np.repeat(np.array([[sample.conditions[k] for k in self.condition_keys]]),
                                   sample.n_points, axis=0))
        X = np.concatenate(parts, axis=1)
        if self._x_mean is not None:
            X = (X - self._x_mean) / self._x_std
        return np.ascontiguousarray(X, dtype=np.float32)

    def target(self, sample: GeometrySample) -> np.ndarray:
        Y = np.concatenate([np.asarray(sample.fields[f], dtype=float).reshape(sample.n_points, -1)
                            for f in self.fields], axis=1)
        mean = np.array([self._y_mean[f] for f in self.fields])
        std = np.array([self._y_std[f] for f in self.fields])
        return np.ascontiguousarray((Y - mean) / std, dtype=np.float32)

    # ------------------------------------------------------------------------------ the module
    def _module(self, in_dim: int, out_dim: int) -> Any:      # pragma: no cover - overridden
        raise NotImplementedError

    def _arch(self) -> dict[str, Any]:
        """The capacity knobs of this family, beyond ``hidden``/``depth``, so a checkpoint rebuilds the same module."""
        return {}

    def _load_arch(self, arch: dict[str, Any]) -> None:
        """Apply what :meth:`_arch` saved before the module is rebuilt."""
        del arch

    def _torch(self) -> Any:
        import torch
        return torch

    # ------------------------------------------------------------------------------ fit
    def fit(self, samples: Iterable[GeometrySample], *, epochs: int = 60, batch: int = 8, lr: float = 2e-3,
            n_points: int | None = None, weight_decay: float = 1e-4, seed: int | None = None,
            verbose: bool = False) -> GeometrySurrogate:
        """Train on simulations with different geometries, point counts and conditions.

        ``n_points`` (default: the smallest sample) subsamples every sample for the batch, and a fresh subsample is
        drawn every epoch — which is both how a variable count is handled and the augmentation that keeps the model
        from memorising one discretisation. Returns the model, so a fitted one is written in one expression; what the
        training did is in ``fit_report``.
        """
        samples = list(samples)
        if not samples:
            raise ValueError(f"{self.family}.fit: no samples")
        torch = self._torch()
        seed = self.seed if seed is None else int(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        if self.fields is None:
            keys = set(samples[0].fields)
            for s in samples[1:]:
                keys &= set(s.fields)
            if not keys:
                raise ValueError(f"{self.family}.fit: the samples share no field to learn")
            self.fields = tuple(sorted(keys))
        if not self.condition_keys:
            self.condition_keys = samples[0].condition_keys
        for s in samples:
            for f in self.fields:
                if f not in s.fields:
                    raise ValueError(f"{self.family}.fit: a sample has no field {f!r}")

        # normalisation from the training set, over every point of every sample
        X = np.concatenate([self._raw_features(s) for s in samples], axis=0)
        self._x_mean, self._x_std = X.mean(0), X.std(0) + 1e-8
        for f in self.fields:
            v = np.concatenate([np.asarray(s.fields[f], dtype=float).reshape(-1) for s in samples])
            self._y_mean[f], self._y_std[f] = float(v.mean()), float(v.std() + 1e-8)

        out_dim = sum(np.asarray(samples[0].fields[f]).reshape(samples[0].n_points, -1).shape[1]
                      for f in self.fields)
        self._out_dim = int(out_dim)
        self.module = self._module(self.in_dim, out_dim)
        n_points = int(n_points or min(s.n_points for s in samples))
        self.n_points = n_points
        if self.device:
            self.module.to(self.device)
        opt = torch.optim.AdamW(self.module.parameters(), lr=lr, weight_decay=weight_decay)
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        self.history = []
        for epoch in range(int(epochs)):
            order = rng.permutation(len(samples))
            total, seen = 0.0, 0
            for start in range(0, len(order), batch):
                chunk = [samples[i] for i in order[start:start + batch]]
                Xb, Yb = self._batch(chunk, n_points, rng)
                xb = torch.as_tensor(Xb, device=self.device)
                yb = torch.as_tensor(Yb, device=self.device)
                opt.zero_grad()
                out = self.module(xb)
                pred = getattr(out, "y", out)
                loss = torch.mean((pred - yb) ** 2)
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(chunk)
                seen += len(chunk)
            self.history.append({"epoch": epoch, "loss": total / max(seen, 1)})
            if verbose and (epoch % max(epochs // 10, 1) == 0 or epoch == epochs - 1):
                print(f"  epoch {epoch:4d}  loss {total / max(seen, 1):.4e}")
        self.n_parameters = int(sum(p.numel() for p in self.module.parameters()))
        self.fit_seconds = time.perf_counter() - t0
        self.fit_report = {"epochs": int(epochs), "samples": len(samples), "n_points": n_points,
                           "parameters": self.n_parameters, "seconds": self.fit_seconds,
                           "loss_first": self.history[0]["loss"] if self.history else None,
                           "loss_last": self.history[-1]["loss"] if self.history else None}
        return self

    def _raw_features(self, sample: GeometrySample) -> np.ndarray:
        parts = [sample.points]
        if self.use_normals:
            parts.append(sample.normals if sample.normals is not None else np.zeros_like(sample.points))
        if self.condition_keys:
            parts.append(np.repeat(np.array([[sample.conditions[k] for k in self.condition_keys]]),
                                   sample.n_points, axis=0))
        return np.concatenate(parts, axis=1)

    def _batch(self, chunk: list[GeometrySample], n_points: int, rng: np.random.Generator):
        X, Y = [], []
        for s in chunk:
            n = min(n_points, s.n_points)
            idx = np.sort(rng.choice(s.n_points, size=n, replace=False)) if s.n_points > n else np.arange(n)
            sub = s.subset(idx) if s.n_points > n else s
            X.append(self.features(sub))
            Y.append(self.target(sub))
        return np.stack(X), np.stack(Y)

    # ------------------------------------------------------------------------------ predict
    def predict(self, points: Any, normals: Any = None, conditions: dict[str, float] | None = None,
                elements: dict[str, Any] | None = None, *, chunk: int = 200_000) -> GeometryPrediction:
        """The field at the points handed in — any number of them, in any order — and the integrals from it.

        These operators **pool over the point set**: the slices of Transolver are weighted means of the points in the
        set and the PointNet baseline max-pools it, so the field at a point is a function of the *whole* set it arrives
        with. Two consequences worth knowing before trusting a number: predicting on a decimated surface is a
        different query, not a cheaper one; and ``chunk`` splits a set too large for memory into independent sets, so
        it changes the answer. For a surface of tens of thousands of points — a car — do not chunk, and predict on the
        discretisation the model was trained on."""
        if self.module is None:
            raise RuntimeError(f"{self.family}: predict before fit")
        sample = GeometrySample(points=points, normals=normals, conditions=conditions or {},
                                elements=elements or {})
        torch = self._torch()
        self.module.eval()
        outs = []
        with torch.no_grad():
            for start in range(0, sample.n_points, max(int(chunk), 1)):
                part = sample.subset(np.arange(start, min(start + chunk, sample.n_points)))
                xb = torch.as_tensor(self.features(part)[None, ...], device=self.device)
                out = self.module(xb)
                outs.append(getattr(out, "y", out).detach().cpu().numpy()[0])
        raw = np.concatenate(outs, axis=0) if outs else np.zeros((0, 0))
        fields: dict[str, np.ndarray] = {}
        col = 0
        for f in self.fields:
            width = np.asarray(sample.fields[f]).reshape(sample.n_points, -1).shape[1] if f in sample.fields else 1
            block = raw[:, col:col + width]
            block = block * self._y_std[f] + self._y_mean[f]
            fields[f] = block[:, 0] if width == 1 else block
            col += width
        pred = GeometryPrediction(points=np.asarray(sample.points), fields=fields)
        if self.integrate is not None:
            pred.integrals = dict(self.integrate(fields=fields, points=sample.points, normals=sample.normals,
                                                 conditions=sample.conditions, elements=sample.elements))
        return pred

    # ------------------------------------------------------------------------------ evaluate
    def evaluate(self, samples: Iterable[GeometrySample]) -> GeometryEvaluation:
        """Relative L2 of the field, the integral error and the ranking R², on simulations the model never saw."""
        samples = list(samples)
        report = GeometryEvaluation()
        truth_integral: list[float] = []
        pred_integral: list[float] = []
        # the ranking is measured on the first integral every sample reports (the lift of the family, here)
        key = next(iter(sorted(samples[0].integrals)), None) if samples else None
        for s in samples:
            pred = self.predict(s.points, s.normals, s.conditions, s.elements)
            row: dict[str, Any] = {"name": s.name, "points": s.n_points}
            num, den = 0.0, 0.0
            for f in self.fields:
                if f not in s.fields:
                    continue
                t = np.asarray(s.fields[f], dtype=float).reshape(s.n_points, -1)
                p = np.asarray(pred.fields[f], dtype=float).reshape(s.n_points, -1)
                num += float(np.sum((p - t) ** 2))
                den += float(np.sum(t ** 2))
                row[f"relative_l2:{f}"] = float(np.linalg.norm(p - t) / max(np.linalg.norm(t), 1e-300))
            row["relative_l2"] = math.sqrt(num / max(den, 1e-300))
            for k, v in s.integrals.items():
                if k in pred.integrals:
                    row[f"integral_error:{k}"] = abs(float(pred.integrals[k]) - float(v))
                    row[f"integral:{k}"] = float(pred.integrals[k])
                    row[f"truth:{k}"] = float(v)
                    if k == key:
                        truth_integral.append(float(v))
                        pred_integral.append(float(pred.integrals[k]))
            report.per_sample.append(row)
        if key is not None and len(truth_integral) >= 3:
            from scipy.stats import spearmanr

            rho = spearmanr(truth_integral, pred_integral).statistic
            for row in report.per_sample:
                row["rank_r2"] = float(rho) ** 2 if np.isfinite(rho) else 0.0
        return report

    # ------------------------------------------------------------------------------ persistence
    def save(self, directory: str) -> str:
        """Weights plus everything the input convention needs, so a reload predicts the same numbers."""
        import torch

        os.makedirs(directory, exist_ok=True)
        torch.save(self.module.state_dict(), os.path.join(directory, "weights.pt"))
        config = {"family": self.family, "fields": list(self.fields or ()), "condition_keys": self.condition_keys,
                  "hidden": self.hidden, "depth": self.depth, "use_normals": self.use_normals,
                  "arch": self._arch(),
                  "x_mean": None if self._x_mean is None else self._x_mean.tolist(),
                  "x_std": None if self._x_std is None else self._x_std.tolist(),
                  "y_mean": self._y_mean, "y_std": self._y_std, "n_points": getattr(self, "n_points", None),
                  "n_parameters": self.n_parameters, "out_dim": self._out_dim, "history": self.history}
        with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=1)
        return directory

    def load(self, directory: str) -> GeometrySurrogate:
        import torch

        with open(os.path.join(directory, "config.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
        if cfg["family"] != self.family:
            raise ValueError(f"the checkpoint is a {cfg['family']} model, this is {self.family}")
        self.fields = tuple(cfg["fields"])
        self.condition_keys = list(cfg["condition_keys"])
        self.use_normals = bool(cfg["use_normals"])
        self.hidden, self.depth = int(cfg["hidden"]), int(cfg["depth"])
        self._load_arch(dict(cfg.get("arch") or {}))
        self._x_mean = None if cfg["x_mean"] is None else np.asarray(cfg["x_mean"], dtype=float)
        self._x_std = None if cfg["x_std"] is None else np.asarray(cfg["x_std"], dtype=float)
        self._y_mean = dict(cfg["y_mean"])
        self._y_std = dict(cfg["y_std"])
        self.history = cfg.get("history", [])
        self.n_parameters = int(cfg.get("n_parameters", 0))
        self.n_points = cfg.get("n_points")
        self.module = self._module(self.in_dim, int(cfg["out_dim"]))
        self.module.load_state_dict(torch.load(os.path.join(directory, "weights.pt"), map_location=self.device))
        self.module.eval()
        self.n_parameters = int(sum(p.numel() for p in self.module.parameters()))
        return self

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(family={self.family!r}, fields={list(self.fields or ())}, "
                f"parameters={self.n_parameters})")


class TransolverGeometry(GeometrySurrogate):
    """Transolver behind the interface (Wu et al., ICML 2024): physics-attention over learned slices.

    The slices are what make it affordable on a car: attention runs among ``slices`` tokens, not among the points, so
    the cost is linear in the point count. ``hidden``/``depth``/``slices`` are the capacity knobs."""

    family = "transolver"

    def __init__(self, *, slices: int = 32, heads: int = 8, **kwargs: Any):
        super().__init__(**kwargs)
        self.slices = int(slices)
        self.heads = int(heads)

    def _module(self, in_dim: int, out_dim: int) -> Any:
        from pinneapple_neural.architectures.neural_operators.transolver import Transolver

        return Transolver(in_dim=in_dim, out_dim=out_dim, dim=self.hidden, depth=self.depth,
                          heads=self.heads, slices=self.slices)

    def _arch(self) -> dict[str, Any]:
        return {"slices": self.slices, "heads": self.heads}

    def _load_arch(self, arch: dict[str, Any]) -> None:
        self.slices = int(arch.get("slices", self.slices))
        self.heads = int(arch.get("heads", self.heads))


class PointNetGeometry(GeometrySurrogate):
    """PointNet behind the interface (Qi et al., CVPR 2017): the baseline without geometric attention."""

    family = "pointnet"

    def _module(self, in_dim: int, out_dim: int) -> Any:
        from pinneapple_neural.architectures.point_surrogates import PointNetSurrogate

        return PointNetSurrogate(in_dim=in_dim, out_dim=out_dim, dim=self.hidden, residual=False)


# ----------------------------------------------------------------------------------------------- panel adapter
def samples_from_panel(shapes: Sequence[Any], *, alphas: Sequence[float] = (4.0,), n_per_side: int = 60,
                       field: str = "cp") -> list[GeometrySample]:
    """The panel-method family as ``GeometrySample``s: a 2-D body lifted to a surface point set and labelled by the
    Hess-Smith solver the library already uses as its inviscid reference.

    Each sample carries the point set (``x, y, 0``), the outward normal of every panel, the panel length in
    ``elements`` (the integrator needs it), the angle of attack as a condition, and ``cp`` on every point plus ``cl``
    and ``cd`` as integrals. This is the small real case the interface is measured on — a real solver, real bodies,
    and a point count that is not the same for every shape once ``n_per_side`` or the geometry changes."""
    from pinneapple_design.aero.panel import panel_solve

    out: list[GeometrySample] = []
    for k, shape in enumerate(shapes):
        for alpha in alphas:
            res = panel_solve(np.asarray(shape, dtype=float), float(alpha), n_per_side=n_per_side, close_te=True)
            cp = np.asarray(res.cp, dtype=float)
            nodes = np.asarray(res.nodes, dtype=float)
            pts = np.column_stack([0.5 * (nodes[:-1, 0] + nodes[1:, 0]),
                                   0.5 * (nodes[:-1, 1] + nodes[1:, 1]),
                                   np.zeros(len(cp))])
            length = np.hypot(np.diff(nodes[:, 0]), np.diff(nodes[:, 1]))
            tangent = np.column_stack([(nodes[1:, 0] - nodes[:-1, 0]) / length,
                                       (nodes[1:, 1] - nodes[:-1, 1]) / length])
            normals = np.column_stack([-tangent[:, 1], tangent[:, 0], np.zeros(len(cp))])
            out.append(GeometrySample(
                points=pts, normals=normals, fields={field: cp},
                conditions={"alpha_deg": float(alpha)},
                integrals={"cl": float(res.cl_pressure), "cd": float(res.cd_pressure)},
                elements={"length": length}, name=f"body{k:03d}_a{alpha:g}"))
    return out


def panel_integrals(*, fields: dict[str, np.ndarray], normals: np.ndarray, elements: dict[str, np.ndarray],
                    conditions: dict[str, float], **_: Any) -> dict[str, float]:
    """``integrate`` for the panel family: the solver's own pressure integration, applied to the predicted field.

    Force per unit dynamic pressure on a panel is ``-(Cp n L)``; resolved onto the free-stream axes it gives the lift
    and drag coefficients. The same formula ``benchmarks/geometry_conditioned_surrogate.lift_from_cp`` uses, so a
    predicted Cl is the integral of the predicted Cp and the two can never disagree."""
    cp = np.asarray(fields["cp"], dtype=float).reshape(-1)
    n, L = np.asarray(normals, dtype=float), np.asarray(elements["length"], dtype=float)
    fx = -(cp * n[:, 0] * L).sum()
    fy = -(cp * n[:, 1] * L).sum()
    a = math.radians(float(conditions.get("alpha_deg", 0.0)))
    return {"cl": float(-fx * math.sin(a) + fy * math.cos(a)),
            "cd": float(fx * math.cos(a) + fy * math.sin(a))}
