"""The physics-informed surrogate of a parametric study (#136): a PINN fitted to the fields a study measured.

A ``pinn`` study is a surrogate study whose model is informed by the equation the field obeys. What it needs beyond
the scalar case, and what this module provides:

**per-sample fields.** The evaluator returns the field of every design it measures — ``"__fields__": {"u": values}``
— together with the mesh those values live on, ``"__coords__": {"x": xs, "y": ys}``. :func:`collect` stacks them
into ``(n_designs, n_points)`` and refuses the two cases that would silently compare different things: a design that
returned no field, and designs measured on different meshes.

**domain sampling driven by the mesh.** The collocation points the PDE residual is evaluated at are drawn inside the
bounding box of the mesh's own coordinates (``pinneapple_analysis.surrogate.methods._fit_pinn``), so the physics is
imposed where the field exists, not on an abstract unit square. The mesh is also required to be a tensor-product
lattice (a structured mesh), because that is what lets the residual of the *reference* be measured by finite
differences as the floor the prediction is compared against (:meth:`pinneapple_analysis.surrogate.pde.
PDEDeclaration.fd_residual`).

**the error measured against the reference.** The fit is cross-validated **by design**: every fold trains on the
other designs' fields and predicts the held-out one, so the reported relative L2 is of designs the PINN never saw,
and it is measured against the evaluator's fields, not against the PINN's own training loss. The PDE declaration
comes from ``DesignStudy.model.pde``, already validated by the schema.

What this module does *not* do: it does not train a final model for shipping (the study's answer is the report and
the validated domain), and it does not optimise — a ``pinn`` study measures, fits and reports, exactly like a
``surrogate`` study, with the field as the thing the model has to get right.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["FieldSamples", "collect", "fit_field"]

#: Wall-clock seconds per fold of PINN training, unless the study's evaluator settings override it
#: (``settings={"pinn_train_s": 2.0}``). Equal budget per fold is what makes the folds comparable.
DEFAULT_TRAIN_S = 1.0


class FieldSamples:
    """The measured fields, as arrays the fit can use: designs ``(n, d)``, values ``(n, n_points)``, mesh
    ``(n_points, ndim)`` with its coordinate names, and the lattice shape of the mesh when it is structured."""

    def __init__(self, params: np.ndarray, values: np.ndarray, coords: np.ndarray, coord_names: Sequence[str],
                 field: str, grid_shape: tuple[int, ...] | None):
        self.params = np.asarray(params, float)
        self.values = np.asarray(values, float)
        self.coords = np.asarray(coords, float)
        self.coord_names = list(coord_names)
        self.field = str(field)
        self.grid_shape = grid_shape

    def __len__(self) -> int:
        return int(len(self.params))

    def summary(self) -> dict[str, Any]:
        return {"field": self.field, "n_samples": len(self), "n_points": int(self.coords.shape[0]),
                "coords": list(self.coord_names), "grid_shape": list(self.grid_shape) if self.grid_shape else None,
                "n_variables": int(self.params.shape[1])}


def collect(samples: Sequence[Mapping[str, Any]], names: Sequence[str]) -> FieldSamples:
    """Stack the fields the measured designs returned, or refuse with the reason.

    ``samples`` are the evaluator's answers: ``{"params": {...}, "fields": {"u": values}, "coords": {"x": xs, ...}}``
    for every design that returned both, in the order of ``names`` (the study's variables). The refusals are the
    point: one field per study (a PINN here predicts one scalar field), the same mesh for every design, and at least
    two designs (a fit cross-validated by design needs something to hold out).
    """
    if not samples:
        raise ValueError("a pinn study needs per-sample fields: the evaluator returned none. Return "
                         '"__fields__": {"u": values} together with "__coords__": {"x": ..., "y": ...} for every '
                         "design, or run the study as kind 'surrogate' if there is no field to inform")
    first = samples[0]
    field = next(iter(first["fields"]))
    coord_names = list(first.get("coords") or {})
    coords0 = _coords(first)
    values, params = [], []
    for i, s in enumerate(samples):
        fields = s.get("fields") or {}
        if sorted(fields) != sorted(first["fields"]):
            raise ValueError(f"design {i} returned the fields {sorted(fields)}, the first returned "
                             f"{sorted(first['fields'])}: a study fits one field on the same mesh for every design")
        if len(fields) > 1:
            raise ValueError(f"the evaluator returned {sorted(fields)}: a pinn study fits one scalar field per "
                             "study (return the one the equation is written for, and study the others as separate "
                             "studies)")
        v = np.asarray(fields[field], float).ravel()
        if v.shape[0] != coords0.shape[0]:
            raise ValueError(f"design {i} returned {v.shape[0]} value(s) on a mesh of {coords0.shape[0]} point(s)")
        if list(s.get("coords") or {}) != coord_names or not np.allclose(_coords(s), coords0, rtol=0, atol=1e-12):
            raise ValueError(f"design {i} was measured on a different mesh than the first: a PINN of a parametric "
                             "study predicts one field on one mesh, so the mesh has to be the same for every design")
        values.append(v)
        params.append([float(s["params"][k]) for k in names])
    from pinneapple_analysis.surrogate.data import lattice_order

    lat = lattice_order(coords0)
    return FieldSamples(np.array(params, float), np.array(values, float), coords0, coord_names, field,
                        None if lat is None else lat[0])


def _coords(s: Mapping[str, Any]):
    c = s.get("coords") or {}
    if not c:
        raise ValueError('the evaluator returned a field without its mesh: return "__coords__": {"x": ..., "y": ...} '
                         "with the coordinates of every point of the field")
    return np.stack([np.asarray(v, float).ravel() for v in c.values()], 1)


def fit_field(spec, fs: FieldSamples, *, folds: int = 5, wall_s: float = DEFAULT_TRAIN_S, seed: int = 0,
              options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Cross-validate a PINN over the measured designs and report the error against the reference fields.

    Every fold trains on the other designs (same wall-clock budget each) and predicts the held-out ones, so the
    ``rel_l2`` reported for a design is of a model that never saw it. ``rel_l2`` per design is
    ``||pred - true|| / ||true||`` over the mesh, which is scale-free and comparable across designs; its mean and
    worst are what the trust check reads. The PDE residual of the held-out predictions, and of the reference fields
    themselves, are reported with the ratio between them: the reference's residual is the discretisation floor of
    the comparison, not zero.
    """
    from pinneapple_analysis.cost.equal_budget import ComputeBudget
    from pinneapple_analysis.surrogate.methods import TrainContext, fit_candidate
    from pinneapple_analysis.surrogate.pde import parse_pde

    pde = parse_pde(spec.model.pde, fs.coord_names, list(spec.base.names))
    if pde is None:
        raise ValueError("a pinn study needs model.pde: without the equation the model is an ordinary regressor")
    n = len(fs)
    k = max(2, min(int(folds), n))
    budget = ComputeBudget(wall_s=float(wall_s))
    held: list[np.ndarray] = []
    per_sample: list[dict[str, Any]] = []
    train_s: list[float] = []
    order = np.arange(n)
    for f in range(k):
        test = order[f::k]                       # interleaved folds: every fold spans the whole design space
        train = np.setdiff1d(order, test)
        ctx = TrainContext(fs.params[train], fs.values[train], np.zeros((0, fs.params.shape[1])),
                           np.zeros((0, fs.values.shape[1])), budget, seed + f,
                           grid_shape=fs.grid_shape or (fs.coords.shape[0],), coords=fs.coords, pde=pde,
                           options=dict(options or {}))
        fitted, wall = fit_candidate("pinn", ctx)
        train_s.append(float(fitted.info.get("train_wall_s", wall)))
        pred = np.asarray(fitted.predict(fs.params[test]), float)
        true = fs.values[test]
        for j, i in enumerate(test):
            den = float(np.linalg.norm(true[j]))
            rel = float(np.linalg.norm(pred[j] - true[j]) / max(den, 1e-12))
            per_sample.append({"design": int(i), "fold": f, "rel_l2": rel, "fold_train_s": round(train_s[-1], 4)})
        held.append((test, pred))
    idx = np.concatenate([t for t, _ in held])
    P = np.concatenate([p for _, p in held])
    T = fs.values[idx]
    rel = [d["rel_l2"] for d in per_sample]
    out: dict[str, Any] = {
        "method": "pinn", "family": "physics-informed", "n_samples": n, "folds": k, "n_heldout": int(len(idx)),
        "pde": pde.to_dict(), "error": {"rel_l2_mean": float(np.mean(rel)), "rel_l2_max": float(np.max(rel)),
                                        "rel_l2_median": float(np.median(rel))},
        "train_s_mean": float(np.mean(train_s)), "train_wall_s_budget": float(wall_s),
        "per_sample": sorted(per_sample, key=lambda d: d["design"]),
        "mesh": fs.summary(), "residual": None}
    if fs.grid_shape is not None and len(fs.grid_shape) >= 2:
        shape = (len(idx), *fs.grid_shape)
        r_pred = pde.fd_residual(P.reshape(shape), fs.coords, fs.params[idx])
        r_true = pde.fd_residual(T.reshape(shape), fs.coords, fs.params[idx])
        out["residual"] = {"pred_mean": float(r_pred.mean()), "data_mean": float(r_true.mean()),
                           "ratio": float(r_pred.mean() / max(r_true.mean(), 1e-30)),
                           "unit": "the field's units per unit of coordinate, squared"}
    return out
