"""Where is a model's prediction reliable? Applicability gates on the inputs and a routing rule that sends the cases
the model cannot answer to the high-fidelity solver.

:class:`ApplicabilityDomain` is fitted on the model's reference data (its training or calibration inputs) and tells,
for every query point:

- ``outside``: an input lies outside the validity range (the declared bounds, or the range of the data);
- ``sparse``: inside the range, but the nearest reference points are farther than for ``level`` of the reference
  points themselves (k-nearest-neighbour distance in standardised inputs, leave-one-out on the reference): the model
  interpolates across a hole in its data;
- ``inside``: in range and as close to the data as the data are to each other.

The Mahalanobis p-value of :class:`pinneapple_analysis.trust.TrustGate` is reported as well (it flags points far from
the bulk of a roughly Gaussian cloud; the k-NN test also catches holes inside the range).

:func:`route_cases` combines the gate with the prediction interval: a case goes to the high-fidelity solver when it is
outside or sparse, when its interval is wider than the accuracy the decision needs, or when the interval straddles a
decision limit (the surrogate cannot tell on which side the case falls).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class ApplicabilityDomain:
    """k-NN density gate plus range gate on standardised inputs. Build with :meth:`fit`."""
    names: list[str]
    lo: np.ndarray
    hi: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    k: int
    level: float
    threshold: float
    n_ref: int
    bounds_source: str
    _tree: cKDTree = field(repr=False, default=None)
    _gate: object = field(repr=False, default=None)

    @classmethod
    def fit(cls, X_ref, names: Sequence[str] | None = None, bounds: Mapping[str, tuple[float, float]] | None = None,
            k: int = 5, level: float = 0.99) -> ApplicabilityDomain:
        """Fit on reference inputs ``X_ref`` (n, d). ``bounds`` (``{name: (lo, hi)}``) overrides the data range for
        the variables it names (e.g. a model card's validity range)."""
        X = np.asarray(X_ref, dtype=float)
        if X.ndim != 2 or X.shape[0] < k + 2:
            raise ValueError(f"need a 2-D reference sample with more than {k + 1} rows; got {X.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("the reference inputs contain NaN or infinite values")
        d = X.shape[1]
        names = list(names) if names is not None else [f"x{i}" for i in range(d)]
        if len(names) != d:
            raise ValueError(f"{len(names)} names for {d} columns")
        lo, hi = X.min(axis=0), X.max(axis=0)
        src = "range of the reference data"
        if bounds:
            used = [n for n in names if n in bounds]
            for n in used:
                i = names.index(n)
                lo[i], hi[i] = float(bounds[n][0]), float(bounds[n][1])
            if used:
                src = "declared validity range" + ("" if len(used) == d else " (data range for the other inputs)")
        center = X.mean(axis=0)
        scale = X.std(axis=0)
        scale = np.where(scale > 0, scale, np.where(hi > lo, (hi - lo) / np.sqrt(12.0), 1.0))
        Z = (X - center) / scale
        tree = cKDTree(Z)
        dist, _ = tree.query(Z, k=k + 1)                     # first neighbour is the point itself
        ref_knn = dist[:, -1]
        threshold = float(np.quantile(ref_knn, level))
        import torch

        from pinneapple_analysis.trust.trust_gate import TrustGate
        gate = TrustGate(ood_level=1.0 - level).fit(torch.as_tensor(Z))
        return cls(names=names, lo=lo, hi=hi, center=center, scale=scale, k=int(k), level=float(level),
                   threshold=threshold, n_ref=X.shape[0], bounds_source=src, _tree=tree, _gate=gate)

    def assess(self, X) -> dict:
        """Per query row: ``status`` (inside / sparse / outside), ``in_range``, ``out_of_range`` (names),
        ``knn_distance`` (standardised), ``density_ratio`` (threshold / distance: >= 1 is as dense as the data),
        ``neighbours_within`` (reference points within the threshold radius) and ``mahalanobis_p``."""
        Xq = np.atleast_2d(np.asarray(X, dtype=float))
        if Xq.shape[1] != len(self.names):
            raise ValueError(f"query has {Xq.shape[1]} columns, the domain {len(self.names)}")
        span = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        tol = 1e-9 * span
        below, above = Xq < self.lo - tol, Xq > self.hi + tol
        bad = below | above | ~np.isfinite(Xq)
        Z = (np.nan_to_num(Xq) - self.center) / self.scale
        dist, _ = self._tree.query(Z, k=self.k)
        knn = dist[:, -1] if dist.ndim == 2 else dist
        within = np.array([len(v) for v in self._tree.query_ball_point(Z, r=self.threshold)])
        p = self._gate.ood_p_values(Z)
        in_range = ~bad.any(axis=1)
        status = np.where(~in_range, "outside", np.where(knn > self.threshold, "sparse", "inside"))
        return {"status": status, "in_range": in_range,
                "out_of_range": [[self.names[j] for j in np.nonzero(r)[0]] for r in bad],
                "knn_distance": knn, "density_ratio": self.threshold / np.maximum(knn, 1e-300),
                "neighbours_within": within, "mahalanobis_p": p}

    def to_dict(self) -> dict:
        return {"names": self.names, "lo": self.lo.tolist(), "hi": self.hi.tolist(), "k": self.k,
                "level": self.level, "threshold": self.threshold, "n_ref": self.n_ref,
                "bounds_source": self.bounds_source}


def route_cases(status, pred, lower, upper, tolerance: float | None = None,
                limits: tuple[float | None, float | None] | None = None) -> list[dict]:
    """Routing rule, one entry per case: ``{"route": "surrogate" | "high_fidelity", "reasons": [...]}``.

    A case goes to the high-fidelity solver when (any of) its inputs are ``outside`` the validity range, its
    neighbourhood is ``sparse``, its interval half-width exceeds ``tolerance`` (the accuracy the decision needs, in
    output units), or the interval contains a limit of ``limits = (lo, hi)`` (the decision is not determined).
    """
    status = np.asarray(status)
    pred, lower, upper = (np.asarray(a, dtype=float) for a in (pred, lower, upper))
    lo_lim, hi_lim = limits if limits is not None else (None, None)
    out = []
    for i in range(status.size):
        codes, reasons = [], []
        if status[i] == "outside":
            codes.append("outside_range")
            reasons.append("outside the validity range")
        elif status[i] == "sparse":
            codes.append("sparse_data")
            reasons.append("few reference data nearby (k-NN distance above the reference level)")
        half = (upper[i] - lower[i]) / 2.0
        if tolerance is not None and half > tolerance:
            codes.append("too_uncertain")
            reasons.append(f"interval half-width {half:.4g} > tolerance {tolerance:.4g}")
        for lim, side in ((lo_lim, "lower"), (hi_lim, "upper")):
            if lim is not None and lower[i] <= lim <= upper[i]:
                codes.append("straddles_limit")
                reasons.append(f"interval contains the {side} limit {lim:.4g}: pass/fail undetermined")
        verdict = None
        if not reasons and (lo_lim is not None or hi_lim is not None):
            ok = (lo_lim is None or lower[i] > lo_lim) and (hi_lim is None or upper[i] < hi_lim)
            verdict = "meets limits" if ok else "violates limits"
        out.append({"route": "high_fidelity" if reasons else "surrogate", "codes": codes, "reasons": reasons,
                    "verdict": verdict})
    return out


ROUTE_CODES = {"outside_range": "outside the validity range", "sparse_data": "sparse reference data",
               "too_uncertain": "interval wider than the tolerance", "straddles_limit": "interval contains a limit"}


def routing_summary(routes: Sequence[dict]) -> dict:
    """Counts of each route and of each reason code (a case can have several)."""
    n = len(routes)
    hf = sum(r["route"] == "high_fidelity" for r in routes)
    why = {c: sum(c in r["codes"] for r in routes) for c in ROUTE_CODES}
    return {"n": n, "high_fidelity": hf, "surrogate": n - hf, "share_high_fidelity": hf / n if n else 0.0,
            "reasons": {c: k for c, k in why.items() if k}}


__all__ = ["ApplicabilityDomain", "route_cases", "routing_summary", "ROUTE_CODES"]
