"""Where a surrogate may be used: validity bounds, an out-of-distribution gate, and where its errors concentrate.

``OODGate`` is the batched form of ``pinneapple_physics.physics_audit.Envelope`` (same rule, vectorised): a query is out of distribution when an input is
outside the training range, or when its nearest training sample is farther than ``knn_limit`` times the typical
training spacing (standardised inputs, log10 for positive inputs), i.e. inside the box but in a hole of the data.

``limitations`` measures, on held-out samples only: the error inside and outside the gate, the error per input
quartile (where the error concentrates), the worst samples, the extrapolation ratio (shift error / test error) and,
for fields, where on the grid the error is largest.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["OODGate", "limitations", "per_sample_error"]


class OODGate:
    def __init__(self, X_train: np.ndarray, names: Sequence[str], knn_limit: float = 3.0):
        X = np.asarray(X_train, float)
        self.names, self.knn_limit = list(names), float(knn_limit)
        self.lo, self.hi = X.min(0), X.max(0)
        self._log = np.all(X > 0, axis=0)
        T = self._t(X)
        self.mu, self.sd = T.mean(0), T.std(0) + 1e-12
        self.Z = (T - self.mu) / self.sd
        nn = self._nn(self.Z, exclude_self=True)
        self.spacing = float(np.median(nn)) or 1.0

    def _t(self, X):
        X = np.array(X, float, copy=True)
        X[:, self._log] = np.log10(np.maximum(X[:, self._log], 1e-300))
        return X

    def _nn(self, Q, exclude_self=False):
        out = np.empty(len(Q))
        for i in range(0, len(Q), 512):
            d = np.sqrt(((Q[i:i + 512, None, :] - self.Z[None, :, :]) ** 2).sum(-1))
            if exclude_self:
                d[np.arange(len(d)), np.arange(i, i + len(d))] = np.inf
            out[i:i + 512] = d.min(1)
        return out

    def bounds(self) -> dict[str, list[float]]:
        return {k: [float(a), float(b)] for k, a, b in zip(self.names, self.lo, self.hi, strict=True)}

    def assess(self, X: np.ndarray) -> list[dict[str, Any]]:
        """Per row: ``ood``, inputs ``outside`` the range, ``knn_distance`` (x typical spacing), ``reasons``."""
        X = np.atleast_2d(np.asarray(X, float))
        with np.errstate(invalid="ignore"):
            Q = (self._t(np.where(np.isfinite(X), X, 0.0)) - self.mu) / self.sd
        knn = self._nn(Q) / self.spacing
        rows = []
        for r, k in zip(X, knn, strict=True):
            outside = [n for n, v, a, b in zip(self.names, r, self.lo, self.hi, strict=True)
                       if not np.isfinite(v) or v < a or v > b]
            reasons = [f"{n} = {v:.4g} outside the training range {a:.4g} .. {b:.4g}"
                       for n, v, a, b in zip(self.names, r, self.lo, self.hi, strict=True) if n in outside]
            if k > self.knn_limit:
                reasons.append(f"nearest training sample at {k:.1f} x the typical spacing (limit {self.knn_limit:g})")
            rows.append({"ood": bool(outside or k > self.knn_limit), "outside": outside,
                         "knn_distance": float(k), "reasons": reasons})
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {"rule": f"out of distribution when an input is outside its training range or the nearest training "
                        f"sample is farther than {self.knn_limit:g} x the typical training spacing "
                        "(standardised inputs, log10 for positive inputs)",
                "bounds": self.bounds(), "knn_limit": self.knn_limit, "spacing": self.spacing}


def per_sample_error(pred: np.ndarray, true: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Per sample: RMS over outputs of (pred - true) / scale (``scale``: per-output std of the training targets; for
    fields the per-sample relative L2 is used instead, see ``study``)."""
    e = (np.asarray(pred) - np.asarray(true)) / scale
    return np.sqrt((e ** 2).mean(1))


def _fmt(v: float) -> str:
    return f"{v:.4g}"


def limitations(gate: OODGate, X_eval: np.ndarray, err: np.ndarray, sets: Mapping[str, np.ndarray],
                X_train: np.ndarray, *, field_err: np.ndarray | None = None, coords: np.ndarray | None = None,
                coord_names: Sequence[str] = (), metric: str = "error") -> dict[str, Any]:
    """``X_eval`` / ``err``: held-out samples (test and shift) and their per-sample error; ``sets``: name -> boolean
    mask over them. ``field_err``: (n_eval, n_points) pointwise absolute error, for the error map."""
    X_eval, err = np.asarray(X_eval, float), np.asarray(err, float)
    flags = gate.assess(X_eval)
    ood = np.array([f["ood"] for f in flags])
    out: dict[str, Any] = {"metric": metric, "validity_bounds": gate.bounds(), "ood_gate": gate.to_dict()}
    by_set = {}
    for name, m in sets.items():
        m = np.asarray(m, bool)
        if m.any():
            by_set[name] = {"n": int(m.sum()), "flagged_ood": int((ood & m).sum()),
                            "mean_error": float(err[m].mean())}
    out["by_set"] = by_set
    ins, outs = err[~ood], err[ood]
    out["gate_effect"] = {"n_inside": int(len(ins)), "n_flagged": int(len(outs)),
                          "mean_error_inside": float(ins.mean()) if len(ins) else None,
                          "mean_error_flagged": float(outs.mean()) if len(outs) else None}
    if len(ins) and len(outs) and ins.mean() > 0:
        out["gate_effect"]["ratio"] = float(outs.mean() / ins.mean())
    # error by input quartile (bins from the training distribution; outside-range samples in their own bins)
    conc = []
    for j, name in enumerate(gate.names):
        q = np.quantile(X_train[:, j], [0, .25, .5, .75, 1])
        if np.unique(q).size < 3:
            continue
        b = np.clip(np.searchsorted(q[1:-1], X_eval[:, j], side="right"), 0, 3)
        b = np.where(X_eval[:, j] < q[0], -1, np.where(X_eval[:, j] > q[-1], 4, b))
        bins = []
        labels = {-1: f"< {_fmt(q[0])} (below training)", 4: f"> {_fmt(q[-1])} (above training)"}
        for k in sorted(set(b.tolist())):
            m = b == k
            lab = labels[k] if k in labels else f"{_fmt(q[k])} .. {_fmt(q[k + 1])}"
            bins.append({"bin": lab, "n": int(m.sum()), "mean_error": float(err[m].mean())})
        valid = [x["mean_error"] for x in bins if x["n"] >= 3]
        spread = (max(valid) / max(min(valid), 1e-30)) if len(valid) >= 2 else 1.0
        conc.append({"input": name, "bins": bins, "max_over_min": float(spread)})
    conc.sort(key=lambda c: -c["max_over_min"])
    out["error_by_input"] = conc
    if conc and conc[0]["max_over_min"] > 1.5:
        c = conc[0]
        worst = max((x for x in c["bins"] if x["n"] >= 3), key=lambda x: x["mean_error"])
        out["concentration"] = (f"errors concentrate in {c['input']} {worst['bin']}: mean {metric} "
                                f"{worst['mean_error']:.3g}, {c['max_over_min']:.1f} x the best quartile")
    else:
        out["concentration"] = "no input quartile has more than 1.5 x the error of another"
    order = np.argsort(err)[::-1][:5]
    out["worst_samples"] = [{"inputs": dict(zip(gate.names, map(float, X_eval[i]), strict=True)),
                             "error": float(err[i]), "ood": bool(ood[i]), "reasons": flags[i]["reasons"]}
                            for i in order]
    if "test" in by_set and "shift" in by_set and by_set["test"]["mean_error"] > 0:
        out["extrapolation_ratio"] = by_set["shift"]["mean_error"] / by_set["test"]["mean_error"]
    if field_err is not None and coords is not None and len(field_err):
        emap = np.asarray(field_err).mean(0)
        k = int(np.argmax(emap))
        names = list(coord_names) or [f"s{i}" for i in range(coords.shape[1])]
        out["field_error"] = {"map": emap.tolist(), "max_at": dict(zip(names, map(float, coords[k]), strict=True)),
                              "max": float(emap[k]), "mean": float(emap.mean())}
    return out
