from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np


def pinball_loss(y_pred_q: np.ndarray, y_true: np.ndarray, q: float) -> float:
    y_pred_q = np.asarray(y_pred_q, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    q = float(q)
    e = y_true - y_pred_q
    return float(np.nanmean(np.maximum(q * e, (q - 1.0) * e)))


def coverage(y_lo: np.ndarray, y_hi: np.ndarray, y_true: np.ndarray) -> float:
    y_lo = np.asarray(y_lo, dtype=float)
    y_hi = np.asarray(y_hi, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    return float(np.nanmean((y_true >= y_lo) & (y_true <= y_hi)))


def mean_interval_width(y_lo: np.ndarray, y_hi: np.ndarray) -> float:
    y_lo = np.asarray(y_lo, dtype=float)
    y_hi = np.asarray(y_hi, dtype=float)
    return float(np.nanmean(y_hi - y_lo))


@dataclass
class ProbabilisticMetrics:
    quantiles: Sequence[float] = (0.1, 0.5, 0.9)

    def compute(self, *, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """Compute pinball loss per quantile and basic interval metrics."""
        yp = np.asarray(y_pred, dtype=float)
        yt = np.asarray(y_true, dtype=float)

        if yp.ndim == 2:
            yp = yp[:, None, :]
        if yt.ndim == 1:
            yt = yt[:, None]
        if yp.ndim != 3:
            raise ValueError(f"Expected y_pred with 3 dims after normalization, got {yp.shape}")

        out: Dict[str, float] = {}
        qs = list(self.quantiles)

        for qi, q in enumerate(qs):
            out[f"pinball_q{q:g}"] = pinball_loss(yp[:, :, qi], yt, q)

        if len(qs) >= 2:
            lo_i = int(np.argmin(qs))
            hi_i = int(np.argmax(qs))
            out["coverage"] = coverage(yp[:, :, lo_i], yp[:, :, hi_i], yt)
            out["mean_interval_width"] = mean_interval_width(yp[:, :, lo_i], yp[:, :, hi_i])

        return out

# ---------------------------------------------------------------------------
# Probability integral transform (PIT) and central-interval coverage
# ---------------------------------------------------------------------------
#
# For a calibrated predictive distribution F, the PIT u = F(y) is uniform on [0, 1] (Dawid 1984, JRSS A 147:278;
# Gneiting, Balabdaoui & Raftery 2007, JRSS B 69:243). A U-shaped histogram means intervals too narrow, a hump
# means too wide, a slope means bias. Forecast errors at h > 1 are serially correlated, so the deviation from
# uniformity is reported as a distance, not as a chi-square p-value.

def pit_gaussian(mu, sigma, y) -> np.ndarray:
    from scipy.stats import norm
    mu, sigma, y = (np.asarray(a, dtype=float) for a in (mu, sigma, y))
    return norm.cdf((y - mu) / np.maximum(sigma, 1e-12))


def pit_ensemble(samples, y, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Randomised PIT of an ensemble (members on axis 0): ties and the discreteness of the empirical CDF are
    broken uniformly at random, so a calibrated ensemble gives exactly uniform values."""
    rng = rng or np.random.default_rng(0)
    x = np.asarray(samples, dtype=float)
    y = np.asarray(y, dtype=float)
    m = x.shape[0]
    below = (x < y[None]).sum(axis=0)
    equal = (x == y[None]).sum(axis=0)
    return (below + rng.random(below.shape) * (equal + 1)) / (m + 1)


def pit_histogram(pit, n_bins: int = 10) -> Dict[str, object]:
    """Relative frequencies of the PIT in ``n_bins`` equal bins and their mean absolute deviation from 1 / n_bins
    (0 for a perfectly flat histogram), divided by 1 / n_bins."""
    u = np.asarray(pit, dtype=float).ravel()
    u = u[np.isfinite(u)]
    counts, _ = np.histogram(u, bins=n_bins, range=(0.0, 1.0))
    freq = counts / max(len(u), 1)
    return {"freq": [float(f) for f in freq],
            "deviation": float(np.mean(np.abs(freq - 1.0 / n_bins)) * n_bins), "n": int(len(u))}


def gaussian_interval_coverage(mu, sigma, y, level: float = 0.9) -> float:
    from scipy.stats import norm
    z = norm.ppf(0.5 + level / 2)
    mu, sigma, y = (np.asarray(a, dtype=float) for a in (mu, sigma, y))
    return float(np.nanmean(np.abs(y - mu) <= z * sigma))


def ensemble_interval_coverage(samples, y, level: float = 0.9) -> float:
    x = np.asarray(samples, dtype=float)
    lo, hi = np.quantile(x, [0.5 - level / 2, 0.5 + level / 2], axis=0)
    return coverage(lo, hi, y)
