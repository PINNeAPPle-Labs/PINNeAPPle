"""Split-conformal prediction intervals for any model's predictions (NumPy), with the coverage *measured* on a
held-out test split that the calibration never saw.

:class:`~pinneapple_analysis.uncertainty.conformal.ConformalPredictor` wraps a torch model; this module works on
predictions alone, so it serves a zoo model, an ONNX graph, a solver-trained surrogate or a column of predictions in
a CSV file alike::

    study = conformal_study(y_cal, pred_cal, y_test, pred_test, alpha=0.1)
    study.coverage, study.coverage_ci        # fraction of held-out targets inside their interval, Wilson interval
    lo, hi = conformal_interval(pred_new, study.q)

Scores are absolute residuals ``|y - pred|`` (constant width) or, when a per-point scale is given (an ensemble
standard deviation, a predicted sigma), normalised residuals ``|y - pred| / scale`` (width proportional to the
scale: locally adaptive). The guarantee is marginal coverage of at least ``1 - alpha`` for exchangeable data
(Vovk et al. 2005; Angelopoulos & Bates 2022, arXiv:2107.07511); it says nothing about a region the calibration
data do not cover, which is why the coverage is also reported per group.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from pinneapple_analysis.uncertainty.conformal import _conformal_quantile
from pinneapple_analysis.uncertainty.tolerance import wilson_interval


def _col(a, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim == 2 and a.shape[1] == 1:
        a = a[:, 0]
    if a.ndim != 1:
        raise ValueError(f"{name} must be one value per point (n,); got shape {a.shape}")
    return a


def conformal_scores(y, pred, scale=None) -> np.ndarray:
    """Non-conformity scores: ``|y - pred|``, divided by ``scale`` when given."""
    r = np.abs(_col(y, "y") - _col(pred, "pred"))
    if scale is not None:
        s = _col(scale, "scale")
        if np.any(~np.isfinite(s)) or np.any(s <= 0):
            raise ValueError("scale must be positive and finite at every point")
        r = r / s
    return r


def conformal_quantile(scores, alpha: float = 0.1) -> float:
    """The finite-sample conformal quantile: the ``ceil((n + 1)(1 - alpha))``-th smallest score (the largest when
    that exceeds ``n``; with ``n < 1 / alpha - 1`` the guarantee needs more calibration points)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    s = np.asarray(scores, dtype=float).ravel()
    s = s[np.isfinite(s)]
    return _conformal_quantile(torch.as_tensor(s, dtype=torch.float64), alpha)


def conformal_interval(pred, q: float, scale=None) -> tuple[np.ndarray, np.ndarray]:
    """``(lower, upper)`` = ``pred -+ q`` (times ``scale`` for normalised scores)."""
    p = np.asarray(pred, dtype=float)
    w = q if scale is None else q * np.asarray(scale, dtype=float)
    return p - w, p + w


def min_calibration_size(alpha: float) -> int:
    """Smallest calibration set for which the conformal quantile is finite at level ``1 - alpha``."""
    return int(np.ceil(1.0 / alpha - 1.0))


@dataclass
class ConformalStudy:
    """Split-conformal calibration and its held-out evaluation.

    ``coverage`` is the fraction of the *test* targets inside their interval; ``coverage_ci`` its Wilson interval
    at ``level``; ``reliability`` the measured test coverage at other nominal levels (same calibration data), so a
    reader sees whether the guarantee holds across levels, not only at the one chosen.
    """
    alpha: float
    q: float
    normalized: bool
    n_cal: int
    n_test: int
    coverage: float
    coverage_ci: tuple[float, float]
    mean_width: float
    median_width: float
    level: float = 0.95
    reliability: list[dict] = field(default_factory=list)
    groups: list[dict] = field(default_factory=list)
    covered: np.ndarray = field(default=None, repr=False)
    width: np.ndarray = field(default=None, repr=False)

    @property
    def target(self) -> float:
        return 1.0 - self.alpha

    @property
    def meets_target(self) -> bool:
        """The measured coverage is consistent with the guarantee: the target lies at or below the upper end of
        the Wilson interval (a shortfall the test size cannot resolve is not called a failure)."""
        return self.coverage_ci[1] >= self.target

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "target_coverage": self.target, "q": self.q, "normalized": self.normalized,
                "n_cal": self.n_cal, "n_test": self.n_test, "coverage": self.coverage,
                "coverage_ci": list(self.coverage_ci), "ci_level": self.level, "meets_target": self.meets_target,
                "mean_width": self.mean_width, "median_width": self.median_width,
                "reliability": self.reliability, "groups": self.groups}


def coverage_by_group(covered, groups, labels: Sequence[str] | None = None, level: float = 0.95,
                      width=None) -> list[dict]:
    """Measured coverage per group (e.g. bins of an input or of the prediction): where the marginal guarantee
    hides under-coverage. ``groups`` holds a group index per point."""
    c = np.asarray(covered, dtype=bool)
    g = np.asarray(groups)
    out = []
    for i, key in enumerate(np.unique(g)):
        m = g == key
        k, n = int(c[m].sum()), int(m.sum())
        row = {"group": str(labels[i] if labels is not None and i < len(labels) else key), "n": n,
               "coverage": k / n if n else float("nan"), "coverage_ci": list(wilson_interval(k, n, level))}
        if width is not None:
            row["mean_width"] = float(np.mean(np.asarray(width)[m]))
        out.append(row)
    return out


def quantile_bins(x, n_bins: int = 5) -> tuple[np.ndarray, list[str]]:
    """Equal-count bins of ``x``: a bin index per point and a label ``[lo, hi]`` per bin."""
    x = np.asarray(x, dtype=float)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, max(len(edges) - 2, 0))
    labels = [f"[{edges[i]:.4g}, {edges[i + 1]:.4g}]" for i in range(len(edges) - 1)] or [f"{x[0]:.4g}"]
    return idx, labels


def conformal_study(y_cal, pred_cal, y_test, pred_test, alpha: float = 0.1, scale_cal=None, scale_test=None,
                    level: float = 0.95, group_by=None, n_groups: int = 5, group_name: str = "",
                    reliability_alphas: Sequence[float] = (0.5, 0.3, 0.2, 0.1, 0.05)) -> ConformalStudy:
    """Calibrate on ``(y_cal, pred_cal)`` and measure the coverage on ``(y_test, pred_test)``.

    Parameters
    ----------
    alpha : miscoverage; the intervals target ``1 - alpha`` coverage.
    scale_cal, scale_test : optional per-point scales (normalised, locally adaptive intervals); both or neither.
    level : confidence level of the Wilson interval on the measured coverage.
    group_by : optional per-test-point values (an input column, the prediction) to report coverage in
        ``n_groups`` equal-count bins.
    """
    if (scale_cal is None) != (scale_test is None):
        raise ValueError("give scale_cal and scale_test together, or neither")
    s_cal = conformal_scores(y_cal, pred_cal, scale_cal)
    if s_cal.size < min_calibration_size(alpha):
        raise ValueError(f"{s_cal.size} calibration points are too few for alpha = {alpha}: "
                         f"need at least {min_calibration_size(alpha)}")
    q = conformal_quantile(s_cal, alpha)
    yt, pt = _col(y_test, "y_test"), _col(pred_test, "pred_test")
    lo, hi = conformal_interval(pt, q, scale_test)
    covered = (yt >= lo) & (yt <= hi)
    width = hi - lo
    k, n = int(covered.sum()), int(covered.size)
    s_test = conformal_scores(yt, pt, scale_test)
    rel = []
    for a in sorted({float(a) for a in reliability_alphas} | {float(alpha)}, reverse=True):
        if s_cal.size >= min_calibration_size(a):
            qa = conformal_quantile(s_cal, a)
            ka = int((s_test <= qa).sum())
            rel.append({"nominal": 1 - a, "coverage": ka / n, "coverage_ci": list(wilson_interval(ka, n, level))})
    groups = []
    if group_by is not None:
        idx, labels = quantile_bins(group_by, n_groups)
        groups = coverage_by_group(covered, idx, labels, level, width)
        for g in groups:
            g["by"] = group_name
    return ConformalStudy(alpha=float(alpha), q=float(q), normalized=scale_cal is not None, n_cal=int(s_cal.size),
                          n_test=n, coverage=k / n, coverage_ci=wilson_interval(k, n, level),
                          mean_width=float(width.mean()), median_width=float(np.median(width)), level=level,
                          reliability=rel, groups=groups, covered=covered, width=width)


def split_indices(n: int, fractions: Sequence[float] = (0.5, 0.5), seed: int = 0) -> list[np.ndarray]:
    """Random disjoint index sets with the given fractions of ``range(n)`` (the last takes the remainder)."""
    if n < len(fractions):
        raise ValueError(f"cannot split {n} rows into {len(fractions)} parts")
    perm = np.random.default_rng(seed).permutation(n)
    cuts = np.cumsum([int(round(f * n)) for f in fractions[:-1]])
    return [np.sort(p) for p in np.split(perm, cuts)]


__all__ = ["ConformalStudy", "conformal_study", "conformal_scores", "conformal_quantile", "conformal_interval",
           "coverage_by_group", "quantile_bins", "min_calibration_size", "split_indices"]
