r"""Critical slowing down and early-warning indicators.

Near a local bifurcation (saddle-node, transcritical, pitchfork, Hopf) the dominant eigenvalue :math:`\lambda` of the
linearised dynamics approaches zero, so perturbations decay more slowly. For a noisy system linearised around its
equilibrium, :math:`dx = \lambda x\,dt + \sigma\,dW` with :math:`\lambda < 0`, the stationary variance is
:math:`\sigma^2 / (2|\lambda|)` and the lag-:math:`\Delta t` autocorrelation is :math:`e^{\lambda \Delta t}`: both rise
as :math:`\lambda \to 0^-` (Scheffer et al. 2009, Nature 461:53; Dakos et al. 2012, PLoS ONE 7:e41010). This module
estimates those signatures on a time series.

**An indicator is not a prediction.** A rising variance or autocorrelation is *consistent with* critical slowing down;
it does not show that a transition will happen, nor when. The same rise follows from a trend left in the data, from
noise whose amplitude grows, from a change of sampling, or from a slow forcing far from any bifurcation; and many
transitions (noise-induced, global bifurcations, rate-induced tipping) give no warning at all. Every result says this
in its assumptions and in :class:`TransitionDiagnostics`.

Pipeline (:class:`EarlyWarningAnalyzer`):

1. **Detrending** (:func:`detrend`): local linear (LOESS-type, default), Gaussian kernel smoothing or a linear fit,
   with the bandwidth as a fraction of the record; indicators are computed on the residuals. Without detrending, a trend in
   the mean inflates both indicators (the classic false positive; benchmarked).
2. **Rolling statistics** (:class:`RollingStatistics`): variance, standard deviation, coefficient of variation,
   skewness, kurtosis in windows (a fraction of the record or a time length; irregular sampling and missing values
   allowed, with a minimum count per window).
3. **Autocorrelation** (:class:`AutocorrelationAnalyzer`): lag-:math:`\Delta t` autocorrelation per window; on an
   irregular grid either by resampling to a regular grid (marked as interpolated) or from pairs of samples whose
   spacing is within a tolerance of the lag.
4. **Recovery rate** (:class:`RecoveryRateEstimator`): :math:`\hat\lambda = \ln\hat\phi / \Delta t` from the AR(1)
   coefficient per window, or from an exponential fit to a recovery after a pulse.
5. **Spectral indicators**: the fraction of power below a frequency and the low-frequency spectral exponent per
   window (Welch periodogram; spectral reddening, Kleinen et al. 2003).
6. **Trend and significance**: Kendall's :math:`\tau` of each indicator against time; its p-value against
   surrogates of the residuals that keep their autocorrelation but are stationary: an AR(1) fitted to the residuals
   (default, Dakos et al. 2012), or phase-randomised / IAAFT surrogates from
   :mod:`pinneapple_systems.time_series.predictability`.
7. **Sensitivity** (:meth:`EarlyWarningAnalyzer.sensitivity`): :math:`\tau` over a grid of windows and bandwidths, and
   over sampling intervals; a signal that holds only for one choice is fragile.
8. **Non-stationarity checks**: a Kendall trend test on the residual mean (detrending incomplete), and the ADF / KPSS
   tests of :mod:`pinneapple_systems.time_series.audit.tests` when statsmodels is installed.

Benchmarks (:func:`ramp_series`): stochastic normal forms of the saddle-node, transcritical and pitchfork bifurcations
and an Ornstein-Uhlenbeck process with a decreasing restoring rate, each driven slowly towards (but not past) the
bifurcation; and negative controls with constant stability: plain, with a trend in the mean, and with growing noise.
"""
from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats

from ._contract import Assumption, Diagnostic, DynamicsResult, MethodCard, make_rng, reproducibility
from .trajectories import Trajectory

__all__ = [
    "detrend", "RollingStatistics", "AutocorrelationAnalyzer", "RecoveryRateEstimator", "StabilityIndicator",
    "CriticalSlowingDown", "EarlyWarningAnalyzer", "TransitionDiagnostics", "ramp_series", "INTERPRETATION",
    "CARD_EWS",
]

INTERPRETATION = ("An early-warning indicator is not a forecast. A significant rise is consistent with critical "
                  "slowing down (a dominant eigenvalue approaching zero); it does not establish that a transition "
                  "will occur, nor when. Trends left in the data, growing noise, changes of sampling and slow forcing "
                  "can produce the same rise, and many transitions give no warning.")

_A_LOCAL = Assumption("local bifurcation", "the transition, if any, is a local bifurcation reached slowly (the "
                      "dominant eigenvalue goes to zero); noise-induced or rate-induced tipping gives no warning")
_A_SLOW = Assumption("time-scale separation", "the control parameter changes slowly compared with the recovery "
                     "time, so each window is close to stationary")
_A_NOISE = Assumption("stationary noise", "the noise amplitude and the sampling do not change over the record "
                      "(otherwise variance can rise without any slowing down)")
_A_DETREND = Assumption("detrending adequate", "the detrending removes the slow trajectory without removing the "
                        "fluctuations (bandwidth well above the recovery time, well below the record)")


# =========================================================================== detrending
def _gaussian_smooth(t: np.ndarray, x: np.ndarray, bw: float) -> np.ndarray:
    out = np.empty_like(x)
    for i, ti in enumerate(t):
        w = np.exp(-0.5 * ((t - ti) / bw) ** 2)
        out[i] = np.sum(w * x) / np.sum(w)
    return out


def _local_linear(t: np.ndarray, x: np.ndarray, bw: float) -> np.ndarray:
    out = np.empty_like(x)
    for i, ti in enumerate(t):
        u = np.abs(t - ti) / bw
        w = np.where(u < 1, (1 - u ** 3) ** 3, 0.0)
        sw = w.sum()
        if sw <= 0 or np.count_nonzero(w) < 3:
            out[i] = x[i]
            continue
        tm = np.sum(w * t) / sw
        xm = np.sum(w * x) / sw
        var = np.sum(w * (t - tm) ** 2)
        b = np.sum(w * (t - tm) * (x - xm)) / var if var > 0 else 0.0
        out[i] = xm + b * (ti - tm)
    return out


def detrend(t, x, method: Literal["gaussian", "loess", "linear", "none"] = "loess", bandwidth: float = 0.1
            ) -> tuple[np.ndarray, np.ndarray]:
    """``(residuals, trend)``. ``bandwidth`` is a fraction of the record length (Gaussian: kernel std; loess:
    tricube half-width). Missing values (NaN) stay NaN in the residuals and are ignored by the fit.

    ``loess`` (local linear, the default) is unbiased for a locally linear trend up to the ends of the record. The
    Gaussian kernel smoother (Dakos et al. 2012) is biased near the ends by about ``slope x bandwidth``: with a
    strong trend this edge bias enters the residuals and can dominate the windows that touch the ends."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = ~np.isnan(x)
    trend = np.full_like(x, np.nan)
    span = float(t[-1] - t[0]) if t.size > 1 else 1.0
    if method == "none":
        trend[ok] = 0.0
    elif method == "linear":
        p = np.polyfit(t[ok], x[ok], 1)
        trend[ok] = np.polyval(p, t[ok])
    elif method == "gaussian":
        trend[ok] = _gaussian_smooth(t[ok], x[ok], bandwidth * span)
    elif method == "loess":
        trend[ok] = _local_linear(t[ok], x[ok], bandwidth * span)
    else:
        raise ValueError("method must be gaussian, loess, linear or none")
    return x - trend, trend


# =========================================================================== windowed statistics
def _windows(t: np.ndarray, window: float, by: str) -> list[np.ndarray]:
    """Index arrays of trailing windows ending at each sample (``by="fraction"``: window = fraction of the record
    in time; ``"time"``: a duration). Windows that do not fit in the record are skipped."""
    span = float(t[-1] - t[0])
    w = window * span if by == "fraction" else float(window)
    if not 0 < w <= span:
        raise ValueError(f"window {window} ({by}) gives a length of {w:g}, outside (0, {span:g}]")
    starts = np.searchsorted(t, t - w, side="left")
    return [np.arange(s, i + 1) for i, s in enumerate(starts) if t[i] - t[0] >= w * (1 - 1e-9)]


@dataclass
class StabilityIndicator:
    """One indicator through time: values at the end time of each window, the Kendall trend and its significance.

    ``p_value`` is from surrogates (``p_kind``) when computed, else ``None``; ``kendall_p`` is the textbook p-value of
    Kendall's tau assuming independent points, which windowed (overlapping) indicators violate: report the surrogate
    p-value, not this one."""
    name: str
    t: np.ndarray
    values: np.ndarray
    tau: float
    kendall_p: float
    p_value: float | None = None
    p_kind: str = ""
    n_surrogates: int = 0
    counts: np.ndarray | None = None

    @property
    def rising(self) -> bool:
        return self.tau > 0

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value is not None and self.p_value <= alpha and self.tau > 0

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "tau": self.tau, "p_value": self.p_value, "p_kind": self.p_kind,
                "n_surrogates": self.n_surrogates, "n_windows": int(self.t.size),
                "first": float(self.values[0]) if self.values.size else None,
                "last": float(self.values[-1]) if self.values.size else None}


def _kendall(t: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return math.nan, math.nan
    r = stats.kendalltau(t[ok], v[ok])
    return float(r.statistic), float(r.pvalue)


def _fast_view(x: np.ndarray, idx: list[np.ndarray]) -> np.ndarray | None:
    """``(n_windows, w)`` view when every window has the same length and nothing is missing (regular grid), so the
    statistics can be computed in one vectorised pass; ``None`` otherwise."""
    if not idx or np.isnan(x).any():
        return None
    w = idx[0].size
    if any(i.size != w for i in idx):
        return None
    view = np.lib.stride_tricks.sliding_window_view(x, w)
    return view[idx[0][0]: idx[0][0] + len(idx)]


class RollingStatistics:
    """Variance, std, coefficient of variation, skewness and kurtosis in trailing windows; NaN where a window has
    fewer than ``min_count`` values."""

    STATS = ("variance", "std", "cv", "skewness", "kurtosis")

    def __init__(self, window: float = 0.5, *, by: Literal["fraction", "time"] = "fraction", min_count: int = 10):
        self.window, self.by, self.min_count = window, by, min_count

    def compute(self, t, x, which: Sequence[str] = ("variance",), raw: np.ndarray | None = None
                ) -> dict[str, StabilityIndicator]:
        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float)
        idx = _windows(t, self.window, self.by)
        te = np.array([t[i[-1]] for i in idx])
        out = {}
        counts = np.array([int(np.sum(~np.isnan(x[i]))) for i in idx])
        fast = _fast_view(x, idx) if raw is None or not np.isnan(raw).any() else None
        for name in which:
            if name not in self.STATS:
                raise ValueError(f"unknown statistic {name!r}; one of {self.STATS}")
            if fast is not None and fast.shape[1] >= self.min_count:
                if name == "variance":
                    vals = fast.var(axis=1, ddof=1)
                elif name == "std":
                    vals = fast.std(axis=1, ddof=1)
                elif name == "cv":
                    m = (_fast_view(raw, idx) if raw is not None else fast).mean(axis=1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        vals = np.where(m != 0, fast.std(axis=1, ddof=1) / np.abs(m), np.nan)
                elif name == "skewness":
                    vals = stats.skew(fast, axis=1)
                else:
                    vals = stats.kurtosis(fast, axis=1)
                tau, kp = _kendall(te, vals)
                out[name] = StabilityIndicator(name, te, vals, tau, kp, counts=counts)
                continue
            vals = np.full(len(idx), np.nan)
            for k, i in enumerate(idx):
                v = x[i][~np.isnan(x[i])]
                if v.size < self.min_count:
                    continue
                if name == "variance":
                    vals[k] = v.var(ddof=1)
                elif name == "std":
                    vals[k] = v.std(ddof=1)
                elif name == "cv":
                    ref = raw[i][~np.isnan(raw[i])] if raw is not None else v
                    m = ref.mean()
                    vals[k] = v.std(ddof=1) / abs(m) if m != 0 else np.nan
                elif name == "skewness":
                    vals[k] = stats.skew(v)
                elif name == "kurtosis":
                    vals[k] = stats.kurtosis(v)
            tau, kp = _kendall(te, vals)
            out[name] = StabilityIndicator(name, te, vals, tau, kp, counts=counts)
        return out


def _ac_regular(v: np.ndarray, lag: int) -> float:
    a, b = v[:-lag], v[lag:]
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3:
        return np.nan
    a, b = a[ok], b[ok]
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan


def _ac_pairs(t: np.ndarray, v: np.ndarray, lag: float, tol: float) -> tuple[float, int]:
    ok = ~np.isnan(v)
    t, v = t[ok], v[ok]
    j = np.searchsorted(t, t + lag - tol, side="left")
    a, b = [], []
    for i in range(t.size):
        k = j[i]
        if k < t.size and abs(t[k] - t[i] - lag) <= tol:
            a.append(v[i])
            b.append(v[k])
    if len(a) < 3:
        return np.nan, len(a)
    a, b = np.array(a), np.array(b)
    return (float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan), len(a)


class AutocorrelationAnalyzer:
    """Lag autocorrelation per window. ``lag`` in samples on a regular grid; on an irregular grid set ``lag_time``
    and ``irregular="pairs"`` (pairs whose spacing is within ``tol`` of the lag) or ``"resample"`` (the caller resamples
    the trajectory first: interpolation smooths and biases the autocorrelation upwards)."""

    def __init__(self, window: float = 0.5, *, by: Literal["fraction", "time"] = "fraction", lag: int = 1,
                 lag_time: float | None = None, tol: float | None = None, min_count: int = 10):
        self.window, self.by, self.lag, self.lag_time, self.tol, self.min_count = window, by, lag, lag_time, tol, \
            min_count

    def compute(self, t, x) -> StabilityIndicator:
        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float)
        idx = _windows(t, self.window, self.by)
        te = np.array([t[i[-1]] for i in idx])
        vals = np.full(len(idx), np.nan)
        counts = np.zeros(len(idx), dtype=int)
        fast = _fast_view(x, idx) if self.lag_time is None else None
        if fast is not None and fast.shape[1] >= max(self.min_count, self.lag + 3):
            a, b = fast[:, :-self.lag], fast[:, self.lag:]
            a = a - a.mean(axis=1, keepdims=True)
            b = b - b.mean(axis=1, keepdims=True)
            den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
            with np.errstate(divide="ignore", invalid="ignore"):
                vals = np.where(den > 0, (a * b).sum(axis=1) / den, np.nan)
            counts[:] = fast.shape[1]
            tau, kp = _kendall(te, vals)
            return StabilityIndicator(f"ac_lag{self.lag}", te, vals, tau, kp, counts=counts)
        for k, i in enumerate(idx):
            if np.sum(~np.isnan(x[i])) < self.min_count:
                continue
            if self.lag_time is None:
                vals[k] = _ac_regular(x[i], self.lag)
                counts[k] = int(np.sum(~np.isnan(x[i])))
            else:
                tol = self.tol if self.tol is not None else 0.25 * self.lag_time
                vals[k], counts[k] = _ac_pairs(t[i], x[i], self.lag_time, tol)
        tau, kp = _kendall(te, vals)
        name = f"ac_lag{self.lag}" if self.lag_time is None else f"ac_lag_time{self.lag_time:g}"
        return StabilityIndicator(name, te, vals, tau, kp, counts=counts)


class RecoveryRateEstimator:
    """Recovery rate: from the AR(1) coefficient per window, ``lambda = ln(phi) / dt`` (negative for a stable state;
    approaching 0 near a bifurcation), or from a perturbation experiment, an exponential fit of ``x - x_eq``."""

    @staticmethod
    def from_ar1(indicator: StabilityIndicator, dt: float) -> StabilityIndicator:
        phi = indicator.values
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = np.where(phi > 0, np.log(phi) / dt, np.nan)
        tau, kp = _kendall(indicator.t, lam)
        return StabilityIndicator("recovery_rate", indicator.t, lam, tau, kp, counts=indicator.counts)

    @staticmethod
    def from_pulse(t, x, x_eq: float | None = None) -> dict[str, float]:
        """Fit ``x(t) = x_eq + A exp(-r t)`` to a relaxation after a perturbation (``x_eq`` estimated when not
        given). Returns ``rate`` (r > 0), its standard error, and the fitted ``x_eq`` and ``A``."""
        from scipy.optimize import curve_fit

        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float)
        ok = ~np.isnan(x)
        t, x = t[ok] - t[ok][0], x[ok]
        r0 = 1.0 / max(t[-1] / 3, 1e-12)
        if x_eq is None:
            f = lambda tt, a, r, e: e + a * np.exp(-r * tt)              # noqa: E731
            p0 = (x[0] - x[-1], r0, x[-1])
        else:
            f = lambda tt, a, r: x_eq + a * np.exp(-r * tt)                # noqa: E731
            p0 = (x[0] - x_eq, r0)
        popt, pcov = curve_fit(f, t, x, p0=p0, maxfev=20000)
        se = np.sqrt(np.diag(pcov))
        out = {"rate": float(popt[1]), "rate_se": float(se[1]), "A": float(popt[0])}
        out["x_eq"] = float(popt[2]) if x_eq is None else float(x_eq)
        return out


def _spectral(t: np.ndarray, x: np.ndarray, window: float, by: str, f_low: float, min_count: int
              ) -> tuple[StabilityIndicator, StabilityIndicator]:
    from scipy import signal

    idx = _windows(t, window, by)
    te = np.array([t[i[-1]] for i in idx])
    dt = float(np.median(np.diff(t)))
    ratio = np.full(len(idx), np.nan)
    expo = np.full(len(idx), np.nan)
    for k, i in enumerate(idx):
        v = x[i]
        if np.any(np.isnan(v)) or v.size < max(min_count, 16):
            continue
        f, p = signal.welch(v - v.mean(), fs=1.0 / dt, nperseg=min(v.size, max(16, v.size // 4)))
        f, p = f[1:], p[1:]
        if f.size < 4:
            continue
        ratio[k] = float(p[f <= f_low].sum() / p.sum()) if np.any(f <= f_low) else np.nan
        low = f <= np.quantile(f, 0.5)
        if low.sum() >= 3 and np.all(p[low] > 0):
            expo[k] = -float(np.polyfit(np.log(f[low]), np.log(p[low]), 1)[0])
    t1, p1 = _kendall(te, ratio)
    t2, p2 = _kendall(te, expo)
    return (StabilityIndicator("low_frequency_power", te, ratio, t1, p1),
            StabilityIndicator("spectral_exponent", te, expo, t2, p2))


# =========================================================================== surrogates
def _ar1_fit(x: np.ndarray) -> tuple[float, float, float]:
    v = x[~np.isnan(x)]
    a, b = v[:-1], v[1:]
    phi = float(np.dot(a - a.mean(), b - b.mean()) / np.dot(a - a.mean(), a - a.mean()))
    phi = float(np.clip(phi, -0.999, 0.999))
    res = (b - b.mean()) - phi * (a - a.mean())
    return phi, float(res.std(ddof=1)), float(v.mean())


def _surrogate(x: np.ndarray, kind: str, rng: np.random.Generator, ar1: tuple[float, float, float]) -> np.ndarray:
    nan = np.isnan(x)
    if kind == "ar1":
        phi, s, m = ar1
        y = np.empty(x.size)
        y[0] = rng.normal(0, s / math.sqrt(1 - phi ** 2))
        e = rng.normal(0, s, x.size)
        for i in range(1, x.size):
            y[i] = phi * y[i - 1] + e[i]
        y += m
    else:
        from pinneapple_systems.time_series.predictability import (
            iaaft_surrogate,
            phase_randomized_surrogate,
        )

        filled = np.where(nan, np.nanmean(x), x)
        y = phase_randomized_surrogate(filled, rng) if kind == "phase" else iaaft_surrogate(filled, rng)
    y[nan] = np.nan
    return y


# =========================================================================== analyzer
CARD_EWS = MethodCard(
    "early_warning", "validation",
    inputs={"trajectory": "one component of a Trajectory: a scalar time series, regular or irregular, NaN allowed"},
    assumptions=[_A_LOCAL, _A_SLOW, _A_NOISE, _A_DETREND],
    config={"detrend": "loess (default) / gaussian / linear / none", "bandwidth": "fraction of the record",
            "window": "fraction of the record (or a time length)", "surrogate": "ar1 / phase / iaaft",
            "n_surrogates": "size of the null distribution"},
    outputs={"indicators": "StabilityIndicator per statistic: values per window, Kendall tau, surrogate p-value",
             "diagnostics": "non-stationarity of the residuals, sampling, window counts",
             "interpretation": "what a significant rise does and does not mean"},
    failure_modes=["too few samples per window (fewer than min_count)", "a trend left in the data (no or too wide "
                   "detrending) inflates variance and autocorrelation", "growing noise or a change of sampling "
                   "raises variance without slowing down", "transitions that are not local bifurcations give no "
                   "warning", "overlapping windows make Kendall's own p-value invalid: use the surrogate p-value"],
    references=["Scheffer et al. 2009, Nature 461:53", "Dakos et al. 2012, PLoS ONE 7:e41010",
                "Kleinen, Held & Petschel-Held 2003, Ocean Dynamics 53:53 (spectral reddening)",
                "Boettiger & Hastings 2012, J. R. Soc. Interface 9:2527 (false positives and the need for "
                "controls)"])


class EarlyWarningAnalyzer:
    """Early-warning indicators of one series; see the module docstring.

    ``indicators`` chooses among ``variance``, ``std``, ``cv``, ``skewness``, ``kurtosis``, ``ac1`` (lag-1
    autocorrelation, or lag-``lag_time`` on an irregular grid), ``recovery_rate``, ``low_frequency_power``,
    ``spectral_exponent``."""

    card = CARD_EWS
    DEFAULT = ("variance", "ac1")

    def __init__(self, *, detrend: str = "loess", bandwidth: float = 0.1, window: float = 0.5,
                 by: Literal["fraction", "time"] = "fraction", indicators: Sequence[str] = DEFAULT,
                 surrogate: str = "ar1", n_surrogates: int = 199, lag_time: float | None = None,
                 min_count: int = 10, f_low: float | None = None, alpha: float = 0.05):
        if surrogate not in ("ar1", "phase", "iaaft", "none"):
            raise ValueError("surrogate must be ar1, phase, iaaft or none")
        self.detrend, self.bandwidth, self.window, self.by = detrend, bandwidth, window, by
        self.indicators = tuple(indicators)
        self.surrogate, self.n_surrogates, self.lag_time = surrogate, n_surrogates, lag_time
        self.min_count, self.f_low, self.alpha = min_count, f_low, alpha

    # ------------------------------------------------------------------ core
    def _indicators(self, t: np.ndarray, resid: np.ndarray, raw: np.ndarray) -> dict[str, StabilityIndicator]:
        out: dict[str, StabilityIndicator] = {}
        rs = [s for s in self.indicators if s in RollingStatistics.STATS]
        if rs:
            out.update(RollingStatistics(self.window, by=self.by, min_count=self.min_count).compute(t, resid, rs, raw))
        need_ac = "ac1" in self.indicators or "recovery_rate" in self.indicators
        if need_ac:
            ac = AutocorrelationAnalyzer(self.window, by=self.by, lag_time=self.lag_time,
                                         min_count=self.min_count).compute(t, resid)
            if "ac1" in self.indicators:
                ac.name = "ac1"
                out["ac1"] = ac
            if "recovery_rate" in self.indicators:
                dt = self.lag_time if self.lag_time is not None else float(np.median(np.diff(t)))
                out["recovery_rate"] = RecoveryRateEstimator.from_ar1(ac, dt)
        if {"low_frequency_power", "spectral_exponent"} & set(self.indicators):
            dt = float(np.median(np.diff(t)))
            fl = self.f_low if self.f_low is not None else 0.1 / dt / 2
            lf, se = _spectral(t, resid, self.window, self.by, fl, self.min_count)
            if "low_frequency_power" in self.indicators:
                out["low_frequency_power"] = lf
            if "spectral_exponent" in self.indicators:
                out["spectral_exponent"] = se
        return out

    def analyze(self, trajectory: Trajectory | tuple, component: str | int = 0, *, seed: int | None = 0
                ) -> DynamicsResult:
        """Indicators, their trends and surrogate significance, with the non-stationarity and sampling checks."""
        t0 = time.perf_counter()
        if isinstance(trajectory, Trajectory):
            t, x = trajectory.t, trajectory.component(component)
            src = dict(trajectory.source)
        else:
            t, x = (np.asarray(a, dtype=float) for a in trajectory)
            src = {"kind": "array"}
        failures, diags = [], []
        if np.sum(~np.isnan(x)) < 3 * self.min_count:
            failures.append(f"{int(np.sum(~np.isnan(x)))} values: at least {3 * self.min_count} are needed")
        regular = t.size < 3 or np.allclose(np.diff(t), np.median(np.diff(t)), rtol=1e-6)
        if not regular and self.lag_time is None and ({"ac1", "recovery_rate"} & set(self.indicators)):
            failures.append("irregular sampling: set lag_time (pair-based autocorrelation) or resample the "
                            "trajectory first")
        if not regular and {"low_frequency_power", "spectral_exponent"} & set(self.indicators):
            failures.append("spectral indicators need a regular grid")
        if failures:
            return DynamicsResult(CARD_EWS.name, {}, applicable=False, failures=failures,
                                  assumptions=list(CARD_EWS.assumptions), limits=[INTERPRETATION])
        resid, trend = detrend(t, x, self.detrend, self.bandwidth)
        ind = self._indicators(t, resid, x)
        rng, seed = make_rng(seed)
        if self.surrogate != "none" and self.n_surrogates > 0:
            ar1 = _ar1_fit(resid)
            null = {k: [] for k in ind}
            for _ in range(self.n_surrogates):
                s = _surrogate(resid, self.surrogate, rng, ar1)
                si = self._indicators(t, s, s + np.nan_to_num(trend))
                for k in ind:
                    null[k].append(si[k].tau)
            for k, v in ind.items():
                arr = np.array(null[k], dtype=float)
                arr = arr[np.isfinite(arr)]
                v.p_value = float((1 + np.sum(arr >= v.tau)) / (1 + arr.size)) if np.isfinite(v.tau) else None
                v.p_kind = f"{self.surrogate} surrogates of the residuals (one-sided, tau >= observed)"
                v.n_surrogates = int(arr.size)
        # checks
        miss = float(np.mean(np.isnan(x)))
        diags.append(Diagnostic("missing_fraction", miss, "ok" if miss < 0.2 else "warn", threshold=0.2))
        diags.append(Diagnostic("regular_sampling", bool(regular), "info"))
        tm, pm = _kendall(t[~np.isnan(resid)], resid[~np.isnan(resid)])
        trend_left = bool(np.isfinite(pm) and pm < 0.01 and abs(tm) > 0.1)
        diags.append(Diagnostic("residual_trend_tau", tm, "warn" if trend_left else "ok",
                                "a trend remains in the residuals: indicators may rise because of it"
                                if trend_left else "", threshold=0.1))
        if self.detrend == "none":
            diags.append(Diagnostic("no_detrending", True, "warn", "indicators on raw data: a trend in the mean "
                                    "inflates variance and autocorrelation"))
        stationarity = _stationarity(resid)
        if stationarity:
            diags.append(Diagnostic("adf_pvalue", stationarity.get("adf"), "info",
                                    "ADF on the residuals (small p: no unit root)"))
            diags.append(Diagnostic("kpss_pvalue", stationarity.get("kpss"), "info",
                                    "KPSS on the residuals (small p: not level-stationary)"))
        else:
            diags.append(Diagnostic("adf_kpss", "statsmodels not installed", "info"))
        est = {k: v.to_dict() for k, v in ind.items()}
        n_sig = sum(v.significant(self.alpha) for v in ind.values())
        est["n_significant_rising"] = n_sig
        est["interpretation"] = INTERPRETATION
        cfg = {"detrend": self.detrend, "bandwidth": self.bandwidth, "window": self.window, "by": self.by,
               "indicators": list(self.indicators), "surrogate": self.surrogate, "n_surrogates": self.n_surrogates,
               "lag_time": self.lag_time, "min_count": self.min_count, "alpha": self.alpha}
        rep = reproducibility(CARD_EWS.name, cfg, [t, x], seed=seed, wall_time_s=time.perf_counter() - t0)
        return DynamicsResult(CARD_EWS.name, est, {k: {"p_value": v.p_value, "kind": v.p_kind} for k, v in ind.items()},
                              list(CARD_EWS.assumptions), cfg, diags, {},
                              limits=[INTERPRETATION, "Kendall's own p-value is invalid for overlapping windows; the "
                                      "surrogate p-value is reported instead"],
                              reproducibility=rep,
                              details={"indicators": ind, "residuals": resid, "trend": trend, "t": t, "source": src})

    # ------------------------------------------------------------------ sensitivity
    def sensitivity(self, trajectory: Trajectory | tuple, *, windows: Sequence[float] = (0.25, 0.375, 0.5, 0.625, 0.75),
                    bandwidths: Sequence[float] = (0.05, 0.1, 0.15, 0.2), component: str | int = 0) -> dict[str, Any]:
        """Kendall tau of each indicator over a grid of windows x bandwidths (Dakos et al. 2012, Fig. 4). Returns the
        tau grids and the fraction of settings with tau > 0. No surrogates (the grid is a robustness check)."""
        if isinstance(trajectory, Trajectory):
            t, x = trajectory.t, trajectory.component(component)
        else:
            t, x = (np.asarray(a, dtype=float) for a in trajectory)
        grids = {k: np.full((len(windows), len(bandwidths)), np.nan) for k in self.indicators}
        for i, w in enumerate(windows):
            for j, b in enumerate(bandwidths):
                a = EarlyWarningAnalyzer(detrend=self.detrend, bandwidth=b, window=w, by=self.by,
                                         indicators=self.indicators, surrogate="none", lag_time=self.lag_time,
                                         min_count=self.min_count, f_low=self.f_low)
                resid, _ = detrend(t, x, self.detrend, b)
                ind = a._indicators(t, resid, x)
                for k in self.indicators:
                    grids[k][i, j] = ind[k].tau
        return {"windows": list(windows), "bandwidths": list(bandwidths), "tau": grids,
                "fraction_positive": {k: float(np.mean(g[np.isfinite(g)] > 0)) if np.isfinite(g).any() else math.nan
                                      for k, g in grids.items()}}

    def sampling_sensitivity(self, trajectory: Trajectory | tuple, factors: Sequence[int] = (1, 2, 4, 8), *,
                             component: str | int = 0, seed: int = 0) -> dict[str, Any]:
        """Indicators after keeping every ``k``-th sample: does the signal survive coarser sampling?"""
        if isinstance(trajectory, Trajectory):
            t, x = trajectory.t, trajectory.component(component)
        else:
            t, x = (np.asarray(a, dtype=float) for a in trajectory)
        rows = {}
        for k in factors:
            r = self.analyze((t[::k], x[::k]), seed=seed)
            rows[int(k)] = {n: (r.estimate[n]["tau"], r.estimate[n]["p_value"]) for n in self.indicators} \
                if r.applicable else {"failures": r.failures}
        return rows


def _stationarity(resid: np.ndarray) -> dict[str, float] | None:
    try:
        import statsmodels  # noqa: F401
    except ImportError:
        return None
    from pinneapple_systems.time_series.audit.tests import adf_test, kpss_test

    v = resid[~np.isnan(resid)]
    a, k = adf_test(v), kpss_test(v)
    return {"adf": a.get("pvalue"), "kpss": k.get("pvalue")}


class CriticalSlowingDown(EarlyWarningAnalyzer):
    """The critical-slowing-down signature: variance, lag-1 autocorrelation and the AR(1) recovery rate, which should
    rise, rise and approach zero from below together."""

    def __init__(self, **kwargs):
        kwargs.setdefault("indicators", ("variance", "ac1", "recovery_rate"))
        super().__init__(**kwargs)


class TransitionDiagnostics:
    """Summary of an early-warning analysis that keeps indicator and prediction apart.

    ``verdict`` is one of ``"no signal"``, ``"indicator signal (not a prediction)"`` or ``"inconclusive"``; it states
    which indicators rise significantly, which checks raise doubts (trend in the residuals, raw data, missing values,
    sensitivity to the window), and the fixed interpretation text."""

    def __init__(self, result: DynamicsResult, sensitivity: Mapping[str, Any] | None = None, alpha: float = 0.05):
        self.result, self.sensitivity, self.alpha = result, sensitivity, alpha

    def report(self) -> dict[str, Any]:
        r = self.result
        if not r.applicable:
            return {"verdict": "inconclusive", "reasons": list(r.failures), "is_prediction": False,
                    "interpretation": INTERPRETATION}
        ind: dict[str, StabilityIndicator] = r.details["indicators"]
        rising = [k for k, v in ind.items() if v.significant(self.alpha)]
        doubts = [f"{d.name}: {d.message}" for d in r.diagnostics if d.status == "warn"]
        if self.sensitivity is not None:
            for k, frac in self.sensitivity["fraction_positive"].items():
                if k in rising and frac < 0.8:
                    doubts.append(f"{k}: tau > 0 in only {frac:.0%} of the window x bandwidth settings")
        if not rising:
            verdict = "no signal"
        elif doubts:
            verdict = "inconclusive"
        else:
            verdict = "indicator signal (not a prediction)"
        return {"verdict": verdict, "rising_significantly": rising, "doubts": doubts, "is_prediction": False,
                "interpretation": INTERPRETATION,
                "indicators": {k: v.to_dict() for k, v in ind.items()}}


# =========================================================================== benchmark series
_KINDS = ("saddle_node", "transcritical", "pitchfork", "ou_decreasing", "stable", "stable_trend", "stable_growing_noise")


def ramp_series(kind: str, *, n: int = 1000, t_end: float = 500.0, sigma: float = 0.05, substeps: int = 20,
                seed: int | None = 0, start: float = 1.0, end: float = 0.05, trend: float = 5.0,
                noise_growth: float = 3.0) -> dict[str, Any]:
    """A series sampled at ``n`` points on ``[0, t_end]`` from a stochastic normal form whose stability decreases
    (``kind`` in ``saddle_node``, ``transcritical``, ``pitchfork``, ``ou_decreasing``) or stays constant (controls:
    ``stable``, ``stable_trend`` with an accelerating (quadratic) drift of the mean by ``trend`` stationary standard deviations, ``stable_growing_noise`` with the noise
    amplitude multiplied by ``noise_growth``). The decay rate of perturbations goes linearly from ``start`` to ``end``
    (> 0: the system never crosses the bifurcation). Euler-Maruyama with ``substeps`` steps per sample.

    Normal forms and their equilibria (decay rate ``r`` = |eigenvalue|):

    * saddle-node: ``dx = (mu - x^2) dt``, equilibrium ``sqrt(mu)``, ``r = 2 sqrt(mu)``;
    * transcritical: ``dx = (-r x - x^2) dt``, equilibrium 0 (``mu = -r``);
    * pitchfork: ``dx = (-r x - x^3) dt``, equilibrium 0;
    * ou_decreasing: ``dx = -r x dt``.

    Returns ``t``, ``x``, the true rate ``r(t)``, the theoretical lag-1 autocorrelation ``exp(-r dt)`` and variance
    ``sigma^2 / (2 r)`` of the linearisation. The saddle-node and transcritical basins shrink with ``r`` (width of
    order ``r``): keep ``sigma / sqrt(2 r_end)`` well below it, or the path leaves the basin (``tipped_at`` is set and
    the rest of ``x`` is NaN)."""
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}")
    rng, seed = make_rng(seed)
    t = np.linspace(0, t_end, n)
    dt = t[1] - t[0]
    h = dt / substeps
    ramp = kind in ("saddle_node", "transcritical", "pitchfork", "ou_decreasing")
    r_of = (lambda s: start + (end - start) * s / t_end) if ramp else (lambda s: start)
    sig_of = (lambda s: sigma * (1 + (noise_growth - 1) * s / t_end)) if kind == "stable_growing_noise" else \
        (lambda s: sigma)
    x = np.empty(n)
    if kind == "saddle_node":
        state = start / 2                                           # sqrt(mu) with r = 2 sqrt(mu)
    else:
        state = 0.0
    x[0] = state
    s = 0.0
    tipped = None
    for i in range(1, n):
        for _ in range(substeps):
            r = r_of(s)
            if kind == "saddle_node":
                mu = (r / 2) ** 2
                drift = mu - state ** 2
            elif kind == "transcritical":
                drift = -r * state - state ** 2
            elif kind == "pitchfork":
                drift = -r * state - state ** 3
            else:
                drift = -r * state
            state = state + drift * h + sig_of(s) * math.sqrt(h) * rng.standard_normal()
            s += h
        if not abs(state) < 1e3:                                   # left the basin: diverging (tipped)
            tipped = float(t[i])
            x[i:] = np.nan
            break
        x[i] = state
    if kind == "stable_trend":
        x = x + trend * (sigma / math.sqrt(2 * start)) * (t / t_end) ** 2   # accelerating drift, `trend` stds
    rate = np.array([r_of(v) for v in t])
    return {"t": t, "x": x, "rate": rate, "ac1_theory": np.exp(-rate * dt),
            "variance_theory": np.array([sig_of(v) ** 2 for v in t]) / (2 * rate), "dt": dt, "seed": seed,
            "kind": kind, "approaching_bifurcation": ramp, "tipped_at": tipped}

