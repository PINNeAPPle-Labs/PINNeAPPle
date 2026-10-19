"""Held-out errors of a field surrogate trained on snapshots (``data.from_snapshots``), in the terms of the snapshots:
per field over all held-out samples, per region of the body, and on the scalar quantities an engineer reads off the
field (the largest displacement at a loaded face, the largest stress).

    from pinneapple_analysis.surrogate.heldout import heldout_report
    rep = heldout_report(snaps, data, model, st.splits.test)
    rep["fields"]["von_mises"]["rel_l2"], rep["fields"]["von_mises"]["regions"], rep["qois"]

Every number compares the model's prediction at a held-out sample's parameters with the stored full-order result.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .data import SurrogateData, snapshot_values

__all__ = ["heldout_report", "region_rel_l2"]


def region_rel_l2(pred, true, regions: dict[str, Sequence[int]]) -> dict[str, float]:
    """Relative L2 error restricted to each region's points, over all samples: ``pred``/``true`` (n, m) or
    (n, m, components) with the points on axis 1."""
    p, t = np.asarray(pred, float), np.asarray(true, float)
    out = {}
    for name, idx in regions.items():
        idx = np.asarray(idx, np.int64)
        den = float(np.linalg.norm(t[:, idx]))
        out[name] = float(np.linalg.norm(p[:, idx] - t[:, idx]) / den) if den > 0 else float("nan")
    return out


def _rel(a, b) -> float:
    d = float(np.linalg.norm(b))
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / d) if d > 0 else float("nan")


def heldout_report(snaps, data: SurrogateData, model, idx, *, grow: float = 0.1) -> dict[str, Any]:
    """Errors of ``model`` (a ``SurrogateModel`` or anything with ``predict_array``) on the samples ``idx``:
    ``fields`` (relative L2 over all of them, per region: the dataset's node sets grown by ``grow`` times the body,
    see ``rom.views.zones``; ``per_sample``: the mean and median of the per-sample errors and the share of the pooled
    weight, the squared field, that the ``heaviest`` samples hold), ``qois`` (per sample, true and predicted: the largest magnitude of each vector field on
    each node set, the largest value of each scalar field) and ``samples`` (relative L2 per sample and field). Fields
    declared non-negative (``data.meta["nonnegative"]``, e.g. a von Mises stress) also count the predicted values
    below zero, which are physically invalid: ``fields[name]["negative_values"]`` and per sample
    ``negative_<name>``."""
    from pinneapple_neural.architectures.rom.views import zones
    idx = np.asarray(idx, np.int64)
    if not len(idx):
        return {"n": 0, "fields": {}, "qois": [], "samples": []}
    pred = snapshot_values(snaps, data, model.predict_array(data.X[idx]))
    true = snaps.X[idx]
    mesh = snaps.mesh(int(idx[0]))
    P = mesh.points if mesh is not None else snaps.locations["point"]
    regs = zones(snaps, P, grow)                                  # node indices: the same nodes in every design
    sets = {k: np.asarray(v["nodes"], np.int64) for k, v in (snaps.meta.get("node_sets") or {}).items()}
    units = snaps.meta.get("units") or {}
    nonneg = set(data.meta.get("nonnegative") or [])
    out: dict[str, Any] = {"n": int(len(idx)), "indices": idx.tolist(), "fields": {}, "qois": [], "samples": []}
    for f in snaps.fields:
        if f["where"] != "point":
            continue
        n = f["name"]
        tv, pv = snaps.field_values(n, true), snaps.field_values(n, pred)
        per = np.array([_rel(pv[k], tv[k]) for k in range(len(tv))])
        w = np.array([float(np.sum(np.square(tv[k]))) for k in range(len(tv))])
        heavy = np.argsort(-w)[:min(3, len(w))]                   # the designs with the largest fields
        out["fields"][n] = {"unit": units.get(n, ""), "rel_l2": _rel(pv, tv),
                            "regions": region_rel_l2(pv, tv, regs) if regs else {},
                            "max_abs_error": float(np.abs(pv - tv).max()),
                            # the pooled number weighs each design by its squared field: say how much
                            "per_sample": {"mean": float(np.nanmean(per)), "median": float(np.nanmedian(per)),
                                           "heaviest": len(heavy),
                                           "heaviest_weight": float(w[heavy].sum() / w.sum()) if w.sum() > 0 else 0.0,
                                           "heaviest_max": float(np.nanmax(per[heavy])) if len(heavy) else 0.0}}
        if n in nonneg:
            out["fields"][n].update(nonnegative=True, negative_values=int((pv < 0).sum()),
                                    min_predicted=float(pv.min()))
        mag_t = np.linalg.norm(tv, axis=-1) if tv.ndim == 3 else tv
        mag_p = np.linalg.norm(pv, axis=-1) if pv.ndim == 3 else pv
        if tv.ndim == 3:                                          # vectors: the largest magnitude on each set
            targets = {f"max |{n}| on {k}": nodes for k, nodes in sets.items()
                       if mag_t[:, nodes].max() > 0}                 # (a fixed set has nothing to compare)
        else:                                                     # scalars: the largest value on the body
            targets = {f"max {n}": np.arange(mag_t.shape[1])}
        for q, nodes in targets.items():
            qt, qp = mag_t[:, nodes].max(1), mag_p[:, nodes].max(1)
            rel = np.abs(qp - qt) / np.maximum(np.abs(qt), 1e-300)
            out["qois"].append({"name": q, "unit": units.get(n, ""), "true": qt.tolist(), "pred": qp.tolist(),
                                "rel_error_mean": float(rel.mean()), "rel_error_max": float(rel.max())})
    for k, i in enumerate(idx):
        out["samples"].append({"index": int(i), "params": data.X[i].tolist(),
                               **{n: _rel(snaps.field_values(n, pred[k:k + 1]), snaps.field_values(n, true[k:k + 1]))
                                  for n in out["fields"]},
                               **{f"negative_{n}": int((snaps.field_values(n, pred[k:k + 1]) < 0).sum())
                                  for n in out["fields"] if n in nonneg}})
    return out
