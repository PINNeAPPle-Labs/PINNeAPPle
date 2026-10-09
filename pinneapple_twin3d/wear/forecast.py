"""Remaining-life forecast per cell: the model of the reference twins, plus confidence.

For each cell, residual thickness ``e0 - wear`` is fitted by least squares against ``x`` over the
trailing window (``win_x`` units of x, widened to at least ``min_pts`` readings when readings are sparse)
and the fit stops at a *repair* (a drop of wear larger than ``repair_drop`` (8 %) of the usable thickness).
``remaining = (residual - emin) / -slope``; cells already at/below the limit have ``remaining = 0``
(the rate is still estimated); flat or growing thickness gives ``inf``.

It is a **linear extrapolation**: it cannot anticipate a change of regime. ``confidence`` only grades the
quality of the fit (points and R^2), it is not a probability. ``remaining_lo/hi`` is a 90 % Student-t
interval on the slope (n >= 3), propagated to the remaining life: a *fit* uncertainty, not a prediction interval
for regime changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .model import WearDataset

DEFAULT_WIN_X = 30.0
CONF = ("n/a", "low", "medium", "high")
# two-sided 90 % Student-t quantile by degrees of freedom (n - 2); linear interpolation between rows
_T90 = ((1, 6.314), (2, 2.920), (3, 2.353), (4, 2.132), (5, 2.015), (6, 1.943), (8, 1.860), (10, 1.812),
        (15, 1.753), (20, 1.725), (30, 1.697), (1000, 1.646))


def t90(df):
    """Student-t 0.95 quantile (90 % two-sided interval) for ``df`` degrees of freedom (array ok)."""
    d = np.asarray(df, dtype=float)
    xs, ys = zip(*_T90)
    return np.interp(d, xs, ys)


@dataclass
class CellFit:
    remaining: np.ndarray  # (N, M) in x units
    remaining_lo: np.ndarray  # 90 % interval (steeper slope -> earlier); equals remaining when n < 3
    remaining_hi: np.ndarray  # (flatter slope -> later; inf when the slope interval reaches zero)
    slope: np.ndarray  # residual thickness change per x (negative = wearing)
    r2: np.ndarray
    n: np.ndarray
    confidence: np.ndarray  # (N, M) int index into CONF


def confidence_index(n, r2):
    n = np.asarray(n)
    r2 = np.asarray(r2, dtype=float)
    out = np.full(n.shape, 1)  # low
    ok = np.isfinite(r2)
    out = np.where(ok & (n >= 5) & (r2 >= 0.7), 2, out)
    out = np.where(ok & (n >= 8) & (r2 >= 0.9), 3, out)
    return np.where(n < 2, 0, out)


def fit_zone(ds: WearDataset, zone: str, s: int, *, win_x: float = DEFAULT_WIN_X, min_pts: int = 8,
             repair_drop: float = 0.08) -> CellFit:
    z = ds.spec.zone(zone)
    W = ds.wear[zone]
    x = ds.x
    w_s = W[s]
    have = np.isfinite(w_s)
    res_s = z.e0 - w_s
    over = have & (res_s <= z.emin)
    drop_min = repair_drop * z.usable
    active = have.copy()
    prev = np.where(have, w_s, 0.0)
    n = np.zeros(z.shape)
    sx = np.zeros(z.shape); sy = np.zeros(z.shape); sxx = np.zeros(z.shape); sxy = np.zeros(z.shape); syy = np.zeros(z.shape)
    for q in range(s, -1, -1):
        v = W[q]
        valid = np.isfinite(v)
        if q < s:
            outside = ~(x[q] > x[s] - win_x) & (n >= min_pts)
            active &= ~outside
            active &= ~(valid & ((v - prev) > drop_min))  # earlier wear much larger => repair happened after q
        take = active & valid
        xq = x[q]
        y = z.e0 - v
        n += take
        sx += np.where(take, xq, 0.0); sy += np.where(take, y, 0.0)
        sxx += np.where(take, xq * xq, 0.0); sxy += np.where(take, xq * y, 0.0); syy += np.where(take, y * y, 0.0)
        prev = np.where(take, v, prev)
    den = n * sxx - sx * sx
    denY = n * syy - sy * sy
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where((n >= 2) & (np.abs(den) > 1e-9), (n * sxy - sx * sy) / den, np.nan)
        r2 = np.where((n >= 2) & (np.abs(den) > 1e-9) & (denY > 1e-12),
                      np.minimum(1.0, (n * sxy - sx * sy) ** 2 / (den * denY)), np.nan)
        rem = np.where(slope < -1e-6, (res_s - z.emin) / -slope, np.inf)
    # 90 % interval on the slope from the regression's standard error (needs n >= 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        sxx_c = sxx - sx * sx / n
        syy_c = syy - sy * sy / n
        sxy_c = sxy - sx * sy / n
        s2 = np.maximum(syy_c - slope * sxy_c, 0.0) / (n - 2)
        se = np.where((n >= 3) & (sxx_c > 1e-12), np.sqrt(s2 / sxx_c), np.nan)
        tq = t90(np.maximum(n - 2, 1))
        steep, flat = slope - tq * se, slope + tq * se
        margin = res_s - z.emin
        lo = np.where(steep < -1e-6, margin / -steep, np.inf)
        hi = np.where(flat < -1e-6, margin / -flat, np.inf)
    lo = np.where(np.isfinite(se), lo, rem)
    hi = np.where(np.isfinite(se), hi, rem)
    rem = np.where(over, 0.0, rem)
    lo = np.where(over, 0.0, lo)
    hi = np.where(over, 0.0, hi)
    keep = lambda a: np.where(have, a, np.nan)
    return CellFit(keep(rem), keep(lo), keep(hi), keep(slope), r2, n.astype(int), confidence_index(n, r2))


@dataclass
class ZoneForecast:
    zone: str
    worst_fraction: float  # consumed fraction of usable thickness, worst cell
    n_over: int  # cells at/below the limit
    min_remaining: float  # in x units; inf = no wear trend
    rate: float  # wear rate [mm per x] of the critical cell
    usable: float
    critical_cell: Optional[tuple]
    confidence: str
    r2: float
    remaining_lo: float = float("nan")  # 90 % interval of the critical cell's remaining life
    remaining_hi: float = float("nan")


def forecast(ds: WearDataset, s: Optional[int] = None, **kw) -> Dict[str, ZoneForecast]:
    s = len(ds.x) - 1 if s is None else s
    out: Dict[str, ZoneForecast] = {}
    for name in ds.zone_names:
        z = ds.spec.zone(name)
        fit = fit_zone(ds, name, s, **kw)
        w = ds.wear[name][s]
        frac = w / z.usable
        rem = fit.remaining
        rate_all = np.where(np.isfinite(fit.slope) & (fit.slope < 0), -fit.slope, 0.0)
        if np.all(np.isnan(rem)):
            out[name] = ZoneForecast(name, float("nan"), 0, float("inf"), 0.0, z.usable, None, "n/a", float("nan"))
            continue
        m = np.nanmin(rem)
        cand = np.argwhere(np.isfinite(rem) & (rem == m) | (np.isinf(rem) & np.isinf(m)))
        best = max(cand, key=lambda ij: rate_all[tuple(ij)])  # ties (many cells at the limit) -> fastest wear
        ij = tuple(int(k) for k in best)
        out[name] = ZoneForecast(name, float(np.nanmax(frac)), int(np.sum(frac >= 1.0)), float(m),
                                 float(rate_all[ij]), z.usable, ij, CONF[int(fit.confidence[ij])],
                                 float(fit.r2[ij]) if np.isfinite(fit.r2[ij]) else float("nan"),
                                 float(fit.remaining_lo[ij]), float(fit.remaining_hi[ij]))
    return out


def forecast_history(ds: WearDataset, **kw) -> List[dict]:
    """Minimum predicted remaining life recomputed at every reading (how the forecast evolved)."""
    hist = []
    for s in range(len(ds.x)):
        if s < 1:
            hist.append({"s": s, "x": float(ds.x[s]), "min_remaining": float("nan"), "predicted_x": float("nan")})
            continue
        m = min(f.min_remaining for f in forecast(ds, s, **kw).values())
        hist.append({"s": s, "x": float(ds.x[s]), "min_remaining": m, "predicted_x": float(ds.x[s]) + m})
    return hist


def backtest(ds: WearDataset, **kw) -> dict:
    """Compare past predictions of 'x when the first cell hits the limit' with what happened."""
    actual = None
    for s in range(len(ds.x)):
        for n in ds.zone_names:
            z = ds.spec.zone(n)
            w = ds.wear[n][s]
            if np.any(np.isfinite(w) & (w >= z.usable)):
                actual = float(ds.x[s]); break
        if actual is not None:
            break
    if actual is None:
        return {"actual_x": None, "points": [], "note": "no cell reached the limit in the history"}
    pts = [{"x": h["x"], "predicted_x": h["predicted_x"], "error": h["predicted_x"] - actual}
           for h in forecast_history(ds, **kw) if h["x"] < actual and np.isfinite(h["predicted_x"])]
    return {"actual_x": actual, "points": pts, "note": ""}


def status(fraction: float) -> str:
    return "critical" if fraction >= 1 else "alert" if fraction >= 0.85 else "attention" if fraction >= 0.6 else "normal"
