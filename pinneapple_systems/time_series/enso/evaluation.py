"""ENSO-specific scores on top of the forecast_lab ones: anomaly correlation per lead, skill stratified by target
calendar month (the spring predictability barrier), categorical El Nino / neutral / La Nina verification, and a
blocked cross-validated hindcast that gives every year an out-of-sample forecast for cheap models.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..forecast_lab.base import Forecaster, SeriesData, targets_at

__all__ = ["acc", "by_target_month", "categorical", "blocked_hindcast", "category"]

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def acc(mean: np.ndarray, y: np.ndarray) -> list[float]:
    """Anomaly correlation (Pearson, over origins) per horizon column."""
    return [float(np.corrcoef(mean[:, j], y[:, j])[0, 1]) for j in range(y.shape[1])]


def by_target_month(mean: np.ndarray, y: np.ndarray, origins: np.ndarray, month_of_year: np.ndarray,
                    horizons: Sequence[int], min_n: int = 6) -> dict[str, Any]:
    """ACC and RMSE for each (target calendar month, horizon); NaN where fewer than ``min_n`` forecasts."""
    H = np.asarray(horizons)
    tm = (np.asarray(month_of_year)[np.asarray(origins)][:, None] + H[None]) % 12      # target month (0 = Jan)
    A = np.full((12, len(H)), np.nan)
    R = np.full((12, len(H)), np.nan)
    N = np.zeros((12, len(H)), int)
    for m in range(12):
        for j in range(len(H)):
            k = tm[:, j] == m
            N[m, j] = int(k.sum())
            if k.sum() >= min_n:
                A[m, j] = np.corrcoef(mean[k, j], y[k, j])[0, 1]
                R[m, j] = np.sqrt(np.mean((mean[k, j] - y[k, j]) ** 2))
    return {"acc": A.tolist(), "rmse": R.tolist(), "n": N.tolist(), "months": list(MONTHS),
            "horizons": list(map(int, H))}


def category(x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """-1 La Nina (x <= -threshold), 0 neutral, +1 El Nino (x >= threshold)."""
    x = np.asarray(x, float)
    return np.where(x >= threshold, 1, np.where(x <= -threshold, -1, 0))


def categorical(mean: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Per horizon: 3 x 3 contingency table (rows observed La Nina / neutral / El Nino, columns forecast), hit rate
    (probability of detection) of each category, false-alarm ratio of the warm and cold categories, accuracy, and
    the Heidke skill score (accuracy relative to random forecasts with the same marginals)."""
    out = []
    for j in range(y.shape[1]):
        o, f = category(y[:, j], threshold), category(mean[:, j], threshold)
        C = np.array([[np.sum((o == a) & (f == b)) for b in (-1, 0, 1)] for a in (-1, 0, 1)])
        n = C.sum()
        pc = np.trace(C) / n
        pe = np.sum(C.sum(0) * C.sum(1)) / n ** 2
        hit = {k: (float(C[i, i] / C[i].sum()) if C[i].sum() else None)
               for i, k in enumerate(("la_nina", "neutral", "el_nino"))}
        far = {k: (float(1 - C[i, i] / C[:, i].sum()) if C[:, i].sum() else None)
               for i, k in ((0, "la_nina"), (2, "el_nino"))}
        out.append({"table": C.tolist(), "hit_rate": hit, "false_alarm_ratio": far, "accuracy": float(pc),
                    "heidke": float((pc - pe) / (1 - pe)) if pe < 1 else None})
    return {"threshold": threshold, "per_horizon": out}


def blocked_hindcast(make: Callable[[], Forecaster], data: SeriesData, horizons: Sequence[int], *,
                     block: int = 48, first: int = 36, last: int | None = None, val_frac: float = 0.15
                     ) -> dict[str, Any]:
    """Out-of-sample forecasts for every origin in [first, last]: the origins are cut into blocks of ``block``
    months; each block is forecast by a model fitted on all other origins whose targets end more than max(h)
    before the block or start after it (purged on both sides), with its last ``val_frac`` (in time) as the
    validation block. Returns origins, mean, std and targets (n, nh). Data after ``last + max(h)`` are never used.
    Meant for cheap models (one fit per block) and for the stratified skill, not as a substitute for the
    untouched test period."""
    H = np.asarray(horizons, int)
    Hm = int(H.max())
    last = (len(data) - Hm - 1) if last is None else int(last)
    origins = np.arange(first, last + 1)
    means, stds = [], []
    for b0 in range(first, last + 1, block):
        blk = origins[(origins >= b0) & (origins < b0 + block)]
        lo, hi = blk[0] - Hm, blk[-1] + Hm                           # purge: no label inside, no input overlap
        pool = origins[(origins + Hm < lo) | (origins > hi)]
        k = max(1, int(round(val_frac * len(pool))))
        val, tr = pool[-k:], pool[:-k]
        tr = tr[(tr + Hm < val[0]) | (tr > val[-1] + Hm)]
        m = make().fit(data, tr, val, H)
        f = m.predict(data, blk)
        means.append(f.mean)
        stds.append(f.std if f.std is not None else np.full_like(f.mean, np.nan))
    mean, std = np.concatenate(means), np.concatenate(stds)
    return {"origins": origins, "mean": mean, "std": std, "y": targets_at(data, origins, H), "horizons": H}
