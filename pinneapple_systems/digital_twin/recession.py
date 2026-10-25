"""A river digital twin from baseflow recession: calibrate on the recent past of a gauge, forecast the present, flag
a sustained departure.

Baseflow recession (the exponential "master recession curve" of hydrograph analysis; Linsley, Kohler & Paulhus,
*Hydrology for Engineers*, 3rd ed., ch. 7; Tallaksen 1995, J. Hydrol. 165, 349-370)::

    Q(t) = Q0 exp(-k (t - t0))

``fit_recession`` calibrates Q0 and k by least squares on ln Q = ln Q0 - k t (``numpy.polyfit``) and returns R^2
and the residual mean and standard deviation in flow units. ``detect_divergence`` flags the first run of readings
whose residual (actual - forecast) stays more than ``n_sigma`` calibration-residual standard deviations above or
below the calibration-residual mean for at least ``sustained_minutes`` of reading time; a single noisy reading is
not flagged.

``run_twin`` is the whole pipeline on a series (timestamps, values): split into a calibration window (the
``calibration_hours`` before the last ``comparison_hours``) and a comparison window (the last ``comparison_hours``),
calibrate, forecast, compare, and give a verdict:

* ``DIVERGENCE_DETECTED`` - a sustained departure in the comparison window;
* ``MATCHES_FORECAST`` - the calibration window was receding (k > epsilon) and the present follows the forecast;
* ``NO_CLEAN_RECESSION`` - the calibration window was not receding, so the model does not apply (said, not forced).

Scope: one exponential reservoir; no rainfall input, no routing, no rating-curve change. A poor calibration R^2 means
the gauge is not in a recession regime and the forecast should not be trusted; the result carries R^2 for that.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

__all__ = ["recession", "fit_recession", "forecast", "detect_divergence", "run_twin", "synthetic_recession"]

CALIBRATION_HOURS = 72.0
COMPARISON_HOURS = 24.0
MIN_CALIBRATION_HOURS = 24.0
MIN_CALIBRATION_POINTS = 6
N_SIGMA = 3.0
SUSTAINED_MINUTES = 60.0
K_EPSILON_PER_HOUR = 1e-4


def recession(q0: float, k_per_hour: float, hours: float | np.ndarray) -> float | np.ndarray:
    """Q0 exp(-k t), t in hours since t0."""
    return q0 * np.exp(-k_per_hour * np.asarray(hours, dtype=float))


def fit_recession(hours: Sequence[float], values: Sequence[float]) -> dict[str, float]:
    """Least-squares Q0, k on the log-linearised recession; ``hours`` is re-based to its first sample (``t0_hours``)."""
    h = np.asarray(hours, dtype=float)
    q = np.asarray(values, dtype=float)
    if h.size < 2:
        raise ValueError("need at least 2 readings to fit a recession curve")
    if np.any(q <= 0):
        raise ValueError("readings must be positive to fit a log-linear recession")
    t = h - h[0]
    slope, intercept = np.polyfit(t, np.log(q), 1)
    q0, k = float(np.exp(intercept)), float(-slope)
    res = q - recession(q0, k, t)
    ss_tot = float(np.sum((q - q.mean()) ** 2))
    return {"t0_hours": float(h[0]), "k_per_hour": k, "q0": q0,
            "r_squared": 1.0 - float(np.sum(res**2)) / ss_tot if ss_tot > 0 else 0.0,
            "n_points": int(h.size), "residual_mean": float(res.mean()),
            "residual_std": float(res.std(ddof=1)) if res.size > 1 else 0.0}


def forecast(calibration: dict[str, float], hours: Sequence[float]) -> np.ndarray:
    """The calibrated curve at ``hours`` (same axis as the fit)."""
    return recession(calibration["q0"], calibration["k_per_hour"], np.asarray(hours, float) - calibration["t0_hours"])


def detect_divergence(hours: Sequence[float], actual: Sequence[float], predicted: Sequence[float],
                      residual_mean: float, residual_std: float, n_sigma: float = N_SIGMA,
                      sustained_minutes: float = SUSTAINED_MINUTES) -> dict[str, Any]:
    """First sustained excursion of ``actual - predicted`` beyond ``residual_mean +- n_sigma * residual_std``."""
    h = np.asarray(hours, dtype=float)
    res = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    std = max(float(residual_std), 1e-9)
    upper, lower = residual_mean + n_sigma * std, residual_mean - n_sigma * std
    need = sustained_minutes / 60.0

    def first_run(flags: np.ndarray) -> int | None:
        start = None
        for i, f in enumerate(flags):
            if not f:
                start = None
                continue
            start = i if start is None else start
            if h[i] - h[start] >= need:
                return start
        return None

    up, lo = first_run(res > upper), first_run(res < lower)
    idx, direction = None, None
    if up is not None and (lo is None or up <= lo):
        idx, direction = up, "rising_above_forecast"
    elif lo is not None:
        idx, direction = lo, "falling_below_forecast"
    z = (res - residual_mean) / std
    return {"residuals": res, "n_sigma_series": z, "upper_threshold": upper, "lower_threshold": lower,
            "detected": idx is not None, "detect_index": idx,
            "detect_hour": float(h[idx]) if idx is not None else None, "direction": direction,
            "max_n_sigma": float(z.max()) if z.size else 0.0, "min_n_sigma": float(z.min()) if z.size else 0.0}


def _hours(timestamps: Sequence[datetime], t0: datetime) -> np.ndarray:
    return np.array([(ts - t0).total_seconds() / 3600.0 for ts in timestamps])


def run_twin(timestamps: Sequence[datetime], values: Sequence[float], *, stage_timestamps=None, stage_values=None,
             calibration_hours: float = CALIBRATION_HOURS, comparison_hours: float = COMPARISON_HOURS,
             n_sigma: float = N_SIGMA, sustained_minutes: float = SUSTAINED_MINUTES) -> dict[str, Any]:
    """Calibrate, forecast and compare one discharge series (and, optionally, fit the gage height the same way for
    the water level now vs forecast). Raises ``ValueError`` when the series is too short to calibrate.

    Returns the series, calibration (``q0``, ``k_per_hour``, ``r_squared``, residual statistics, window), forecast,
    comparison window, detection, verdict, the half-life ln 2 / k and ``water_level`` (``None`` without a usable
    gage-height series)."""
    ts = list(timestamps)
    q = np.asarray(values, dtype=float)
    ok = q > 0
    if ok.sum() < MIN_CALIBRATION_POINTS + 2:
        raise ValueError(f"too few positive readings to calibrate ({int(ok.sum())} of {q.size})")
    ts = [t for t, keep in zip(ts, ok, strict=False) if keep]
    q = q[ok]
    t0 = ts[0]
    h = _hours(ts, t0)
    cmp_start = max(float(h[-1]) - comparison_hours, 0.0)
    cal_start = max(cmp_start - calibration_hours, 0.0)
    cal = (h >= cal_start) & (h <= cmp_start)
    cmp = h > cmp_start
    if cal.sum() < MIN_CALIBRATION_POINTS or cmp_start - cal_start < MIN_CALIBRATION_HOURS - 1e-6:
        raise ValueError(f"only {int(cal.sum())} readings over {cmp_start - cal_start:.1f} h in the calibration "
                         "window: not enough recent history to calibrate")
    fit = fit_recession(h[cal], q[cal])
    regime = "RECESSION" if fit["k_per_hour"] > K_EPSILON_PER_HOUR else "RISING_OR_FLAT"
    pred = forecast(fit, h)
    det = detect_divergence(h[cmp], q[cmp], pred[cmp], fit["residual_mean"], fit["residual_std"], n_sigma,
                            sustained_minutes) if cmp.any() else detect_divergence([], [], [], 0.0, 1.0)
    verdict = ("DIVERGENCE_DETECTED" if det["detected"] else
               "MATCHES_FORECAST" if regime == "RECESSION" else "NO_CLEAN_RECESSION")

    water = None
    if stage_timestamps is not None and stage_values is not None and len(stage_values):
        sh = _hours(stage_timestamps, t0)
        sv = np.asarray(stage_values, dtype=float)
        good = sv > 0
        sh, sv = sh[good], sv[good]
        scal = (sh >= cal_start) & (sh <= cmp_start)
        if scal.sum() >= MIN_CALIBRATION_POINTS:
            sfit = fit_recession(sh[scal], sv[scal])
            water = {"current_ft": float(sv[-1]),
                     "current_time": [t for t, g in zip(stage_timestamps, good, strict=False) if g][-1].isoformat(),
                     "forecast_now_ft": float(forecast(sfit, [sh[-1]])[0]), "r_squared": sfit["r_squared"]}

    def at(hours: float) -> str:
        return (t0 + timedelta(hours=hours)).isoformat()

    k = fit["k_per_hour"]
    return {
        "timeseries": {"hours": h.tolist(), "timestamps": [t.isoformat() for t in ts], "values": q.tolist()},
        "calibration": {**fit, "start_hour": cal_start, "end_hour": cmp_start,
                        "start_time": ts[int(np.argmax(cal))].isoformat(), "end_time": at(cmp_start),
                        "regime": regime, "half_life_hours": float(np.log(2) / k) if k > 0 else None},
        "forecast": {"hours": h.tolist(), "values": pred.tolist()},
        "comparison": {"start_hour": cmp_start, "start_time": at(cmp_start), "hours": h[cmp].tolist(),
                       "actual": q[cmp].tolist(), "predicted": pred[cmp].tolist(),
                       "residuals": det["residuals"].tolist(), "n_sigma_series": det["n_sigma_series"].tolist()},
        "detection": {"n_sigma_rule": n_sigma, "sustained_minutes_required": sustained_minutes,
                      **{k_: det[k_] for k_ in ("upper_threshold", "lower_threshold", "detected", "detect_hour",
                                                "direction", "max_n_sigma", "min_n_sigma")},
                      "detect_time": at(det["detect_hour"]) if det["detect_hour"] is not None else None},
        "verdict": verdict,
        "water_level": water,
        "latest": {"time": ts[-1].isoformat(), "value": float(q[-1])},
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


def synthetic_recession(hours: float = 96.0, step_minutes: float = 15.0, q0: float = 2000.0,
                        k_per_hour: float = 0.004, noise: float = 0.005, bump_at_hour: float | None = None,
                        bump: float = 0.0, seed: int = 0) -> dict[str, list]:
    """A synthetic gauge record for tests and offline demos: an exponential recession with multiplicative noise and an
    optional step increase ``bump`` (fraction) from ``bump_at_hour`` on. Timestamps start 2026-01-01 UTC."""
    rng = np.random.default_rng(seed)
    h = np.arange(0.0, hours + 1e-9, step_minutes / 60.0)
    q = q0 * np.exp(-k_per_hour * h) * (1.0 + noise * rng.standard_normal(h.size))
    if bump_at_hour is not None:
        q = q * np.where(h >= bump_at_hour, 1.0 + bump, 1.0)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return {"timestamps": [start + timedelta(hours=float(x)) for x in h], "values": q.tolist()}

