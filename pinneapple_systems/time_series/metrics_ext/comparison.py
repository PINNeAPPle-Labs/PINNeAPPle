"""Comparing forecasts: skill scores, the Diebold-Mariano test and multiple-comparison control.

* :func:`skill_score` -- 1 - score / score_reference (positive: better than the reference).
* :func:`diebold_mariano` -- equal predictive accuracy of two forecasts of the same targets (Diebold & Mariano
  1995, J. Bus. Econ. Stat. 13:253) with the small-sample correction of Harvey, Leybourne & Newbold (1997, Int. J.
  Forecast. 13:281). An h-step forecast error is serially correlated up to lag h - 1, so the variance of the mean
  loss difference uses the autocovariances up to that lag.
* :func:`holm` (Holm 1979, Scand. J. Stat. 6:65, family-wise error) and :func:`benjamini_hochberg` (Benjamini &
  Hochberg 1995, JRSS B 57:289, false discovery rate) adjust a family of p-values.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["skill_score", "DMResult", "diebold_mariano", "holm", "benjamini_hochberg"]


def skill_score(score, reference) -> np.ndarray:
    """1 - score / reference, elementwise (for errors and other negatively oriented scores)."""
    score = np.asarray(score, float)
    reference = np.asarray(reference, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 - score / reference


@dataclass
class DMResult:
    statistic: float
    p_value: float           # two-sided
    p_value_better: float    # one-sided: forecast 1 has lower expected loss than forecast 2
    mean_loss_difference: float
    n: int
    max_lag: int

    def to_dict(self) -> dict:
        return asdict(self)


def diebold_mariano(e1, e2, *, h: int = 1, loss: str = "se", max_lag: int | None = None,
                    hln: bool = True) -> DMResult:
    """Test E[L(e1)] = E[L(e2)] for two error series on the same targets (in time order).

    ``loss`` is ``"se"`` (squared error), ``"ae"`` (absolute error) or ``"raw"`` (``e1``, ``e2`` are already losses,
    e.g. CRPS). ``max_lag`` defaults to h - 1. When the truncated (rectangular) long-run variance is not positive the
    Bartlett (Newey-West) weights are used instead. With ``hln`` the statistic is scaled by the Harvey-Leybourne-
    Newbold factor and referred to Student's t with n - 1 degrees of freedom; otherwise to the normal.
    """
    from scipy import stats
    e1 = np.asarray(e1, float).ravel()
    e2 = np.asarray(e2, float).ravel()
    if loss == "se":
        d = e1 ** 2 - e2 ** 2
    elif loss == "ae":
        d = np.abs(e1) - np.abs(e2)
    elif loss == "raw":
        d = e1 - e2
    else:
        raise ValueError("loss must be 'se', 'ae' or 'raw'")
    ok = np.isfinite(d)
    d = d[ok]
    n = len(d)
    lag = max(int(h) - 1, 0) if max_lag is None else int(max_lag)
    lag = min(lag, n - 1)
    dbar = float(d.mean())
    dc = d - dbar
    gam = np.array([float(dc[k:] @ dc[:n - k]) / n for k in range(lag + 1)])
    lrv = gam[0] + 2 * gam[1:].sum()
    if lrv <= 0:
        w = 1 - np.arange(1, lag + 1) / (lag + 1)
        lrv = gam[0] + 2 * (w * gam[1:]).sum()
    if lrv <= 0 or n < 2:
        return DMResult(float("nan"), float("nan"), float("nan"), dbar, n, lag)
    stat = dbar / np.sqrt(lrv / n)
    if hln:
        hh = lag + 1
        stat *= np.sqrt(max((n + 1 - 2 * hh + hh * (hh - 1) / n) / n, 1e-12))
        cdf = stats.t(df=n - 1).cdf
    else:
        cdf = stats.norm.cdf
    p2 = 2 * (1 - cdf(abs(stat)))
    return DMResult(float(stat), float(p2), float(cdf(stat)), dbar, n, lag)


def holm(p_values: dict[str, float] | Sequence[float]):
    """Holm step-down adjusted p-values (same keys / order as given); NaNs are left out of the family."""
    keys, p = _unpack(p_values)
    out = np.full(len(p), np.nan)
    idx = np.nonzero(np.isfinite(p))[0]
    m = len(idx)
    order = idx[np.argsort(p[idx])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return _pack(keys, out)


def benjamini_hochberg(p_values: dict[str, float] | Sequence[float]):
    """Benjamini-Hochberg adjusted p-values (q-values); NaNs are left out of the family."""
    keys, p = _unpack(p_values)
    out = np.full(len(p), np.nan)
    idx = np.nonzero(np.isfinite(p))[0]
    m = len(idx)
    order = idx[np.argsort(p[idx])[::-1]]
    running = 1.0
    for k, i in enumerate(order):
        rank = m - k
        running = min(running, p[i] * m / rank)
        out[i] = running
    return _pack(keys, out)


def _unpack(p_values):
    if isinstance(p_values, dict):
        keys = list(p_values)
        return keys, np.asarray([p_values[k] for k in keys], float)
    return None, np.asarray(p_values, float)


def _pack(keys, arr):
    if keys is None:
        return arr
    return {k: float(v) for k, v in zip(keys, arr, strict=False)}
