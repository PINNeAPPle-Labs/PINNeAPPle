"""Observing conditions for tonight from open numerical weather prediction (Open-Meteo, CC BY 4.0).

The ensemble API returns every member of an ensemble forecast (e.g. ICON-EPS / GFS-ENS). Per hour this module turns
the members into: the probability of a clear sky (total cloud < 30 %), the median and 10-90 % range of the cloud
cover, a transparency index from humidity and cloud, and a seeing proxy from the upper-level (250 hPa) wind, the jet
stream being the main source of high-altitude turbulence.

What it is and is not: the probabilities are the ensemble's own spread, not yet recalibrated against observations
at the site (that needs a history of forecasts and observations, see ``calibrate``); the seeing proxy is a ranking
(good / average / poor), not arc-seconds.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
import warnings

import numpy as np

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
VARS = ["cloud_cover", "relative_humidity_2m", "wind_speed_250hPa"]


def fetch_ensemble(lat: float, lon: float, model: str = "icon_seamless", hours: int = 48, timeout: float = 15.0):
    q = {"latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "hourly": ",".join(VARS), "models": model,
         "forecast_hours": hours, "timezone": "UTC"}
    url = ENSEMBLE_URL + "?" + urllib.parse.urlencode(q)
    req = urllib.request.Request(url, headers={"User-Agent": "PINNeAPPle-observatory/1.0"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())


def members(resp: dict, var: str) -> np.ndarray:
    """(n_members, n_hours) array for ``var`` (the control run and every ``var_memberNN`` column)."""
    h = resp["hourly"]
    cols = [k for k in h if k == var or k.startswith(var + "_member")]
    return np.array([[np.nan if v is None else v for v in h[k]] for k in sorted(cols)], dtype=float)


def _json(a: np.ndarray) -> list:
    """Hours where the ensemble has no value (missing members or variables) become None, not NaN (invalid JSON)."""
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(a, float)]


def summarize(resp: dict, clear_threshold: float = 30.0) -> dict:
    cc = members(resp, "cloud_cover")
    rh = members(resp, "relative_humidity_2m")
    jet = members(resp, "wind_speed_250hPa")
    p_clear = np.where(np.isnan(cc), np.nan, cc < clear_threshold)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN hours are reported as None
        return _summarize(resp, cc, rh, jet, np.nanmean(p_clear, axis=0))


def _summarize(resp, cc, rh, jet, p_clear):
    q10, q50, q90 = np.nanpercentile(cc, [10, 50, 90], axis=0)
    rh50 = np.nanmedian(rh, axis=0) if rh.size else np.full(cc.shape[1], np.nan)
    jet50 = np.nanmedian(jet, axis=0) if jet.size else np.full(cc.shape[1], np.nan)
    transparency = np.clip(1 - 0.6 * q50 / 100 - 0.4 * np.clip((rh50 - 50) / 50, 0, 1), 0, 1)
    seeing = np.where(np.isnan(jet50), "unknown",
                      np.where(jet50 < 90, "good", np.where(jet50 < 160, "average", "poor")))   # km/h at 250 hPa
    return {"time": resp["hourly"]["time"], "n_members": int(cc.shape[0]), "p_clear": _json(p_clear),
            "cloud_q10": _json(q10), "cloud_q50": _json(q50), "cloud_q90": _json(q90),
            "humidity_median": _json(rh50), "jet_wind_kmh": _json(jet50), "transparency": _json(transparency),
            "seeing_proxy": seeing.tolist(), "source": "Open-Meteo ensemble API (CC BY 4.0)",
            "calibrated": False}


def calibrate(p_forecast, observed_clear, bins: int = 10) -> dict:
    """Reliability of past clear-sky probabilities against what was observed, and an isotonic-style correction map:
    for each probability bin, the observed frequency (needs a history of forecasts and observations)."""
    p = np.asarray(p_forecast, float)
    o = np.asarray(observed_clear, float)
    edges = np.linspace(0, 1, bins + 1)
    k = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    freq = [float(o[k == b].mean()) if (k == b).any() else None for b in range(bins)]
    filled = np.array([f if f is not None else (edges[b] + edges[b + 1]) / 2 for b, f in enumerate(freq)])
    corrected = np.maximum.accumulate(filled)                      # monotone
    brier = float(np.mean((p - o) ** 2))
    return {"bin_edges": edges.tolist(), "observed_freq": freq, "map": corrected.tolist(), "brier": brier}


# --------------------------------------------------------------------------------- deterministic cloud forecast
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CLOUDY_PCT = 75.0          # at or above this total cloud cover the view through a telescope is lost


def fetch_cloud_forecast(lat: float, lon: float, days: int = 3, timeout: float = 8.0) -> dict:
    """Hourly total cloud cover (%) for the next ``days`` from Open-Meteo's forecast API (no key; CC BY 4.0):
    {"time": [ISO UTC], "cloud_cover": [%], "source": ...}. Raises on network errors (callers fall back)."""
    q = {"latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "hourly": "cloud_cover", "forecast_days": days,
         "timezone": "UTC"}
    req = urllib.request.Request(FORECAST_URL + "?" + urllib.parse.urlencode(q),
                                 headers={"User-Agent": "PINNeAPPle-observatory/1.0"})
    d = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    h = d["hourly"]
    return {"time": [t if t.endswith("Z") else t + "Z" for t in h["time"]], "cloud_cover": h["cloud_cover"],
            "source": "Open-Meteo forecast API (CC BY 4.0)"}


def cloud_at(forecast: dict, iso_utc: str) -> float | None:
    """Cloud cover (%) at a time, linearly interpolated in the hourly forecast; None outside it."""
    from datetime import datetime
    ts = [datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp() for t in forecast["time"]]
    x = datetime.fromisoformat(iso_utc.replace("Z", "+00:00")).timestamp()
    cc = [np.nan if c is None else float(c) for c in forecast["cloud_cover"]]
    if not ts or x < ts[0] or x > ts[-1]:
        return None
    v = float(np.interp(x, ts, cc))
    return None if np.isnan(v) else v


def sky_view(mode: str, forecast: dict | None, iso_utc: str) -> dict:
    """What the sky lets through for an observing session. ``mode`` "perfect" removes the weather (any target, any
    night); "real" uses the cloud-cover forecast and hides the view at or above ``CLOUDY_PCT``. Without a forecast
    "real" falls back to "perfect" and says so."""
    if mode == "perfect":
        return {"mode": "perfect", "cloud_cover": 0.0, "hidden": False, "transmission": 1.0, "fallback": False}
    if forecast is None:
        return {"mode": "perfect", "cloud_cover": 0.0, "hidden": False, "transmission": 1.0, "fallback": True}
    cc = cloud_at(forecast, iso_utc)
    if cc is None:
        return {"mode": "perfect", "cloud_cover": 0.0, "hidden": False, "transmission": 1.0, "fallback": True}
    return {"mode": "real", "cloud_cover": cc, "hidden": cc >= CLOUDY_PCT, "transmission": max(0.0, 1.0 - cc / 100.0),
            "fallback": False}
