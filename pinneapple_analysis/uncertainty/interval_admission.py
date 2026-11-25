"""Admission of prediction intervals on independent evaluation cases."""
from dataclasses import dataclass
from math import isfinite

import numpy as np
from scipy.stats import beta


@dataclass(frozen=True)
class IntervalAdmission:
    approved: bool
    coverage: float
    coverage_lower_bound: float
    mean_width: float
    sample_count: int
    reasons: tuple[str, ...]


def assess_intervals(target, lower, upper, *, min_coverage: float,
                     max_mean_width: float, min_samples: int,
                     confidence: float = 0.95) -> IntervalAdmission:
    """Assess one QoI or physical regime using independent held-out cases.

    Uses a one-sided exact binomial lower confidence bound. The caller
    must ensure independent cases and no calibration/test leakage.
    Width is in the QoI's units. Evaluate each required regime separately.
    """
    if not 0 < min_coverage < 1 or not 0 < confidence < 1:
        raise ValueError("Coverage and confidence must lie in (0, 1)")
    if not isfinite(max_mean_width) or max_mean_width < 0 or min_samples < 1:
        raise ValueError("Invalid width or sample requirement")
    y, lo, hi = (np.asarray(a, dtype=float) for a in (target, lower, upper))
    if y.ndim != 1 or not y.size or lo.shape != y.shape or hi.shape != y.shape:
        raise ValueError("Expected matching nonempty one-dimensional case arrays")
    if not all(np.isfinite(a).all() for a in (y, lo, hi)) or np.any(lo > hi):
        raise ValueError("Intervals and targets must be finite and ordered")
    n = y.size
    hits = int(np.count_nonzero((lo <= y) & (y <= hi)))
    bound = float(beta.ppf(1 - confidence, hits, n - hits + 1)) if hits else 0.0
    width = float(np.mean(hi - lo))
    reasons = []
    if n < min_samples:
        reasons.append("insufficient independent evaluation cases")
    if bound < min_coverage:
        reasons.append("coverage lower confidence bound below requirement")
    if not isfinite(width) or width > max_mean_width:
        reasons.append("interval width exceeds tolerance")
    return IntervalAdmission(not reasons, hits / n, bound, width, n, tuple(reasons))
