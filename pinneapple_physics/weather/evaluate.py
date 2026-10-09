"""Skill of global forecasts by lead time, and how far ahead each forecast can be trusted.

Scores (area-weighted, as in WeatherBench2): RMSE and anomaly correlation (ACC, anomalies from the 1990-2019
climatology). Reference forecasts on the same cases: persistence, climatology, and the published forecasts of
IFS HRES, Pangu-Weather, Keisler's GNN and NeuralGCM (WeatherBench2, 64 x 32 grid).

Horizons:

* ``useful``: first lead at which the ACC falls below 0.6, the usual limit of a useful synoptic forecast;
* ``no_skill``: first lead at which the RMSE reaches that of climatology (the forecast no longer adds anything);
* ``per forecast``: an ensemble (perturbed initial states) gives the spread at every lead; the spread at which,
  on average, the ACC crosses 0.6 is measured on validation forecasts, and each new forecast reports the lead at
  which its own spread reaches it: a flow-dependent horizon, known before the truth arrives.
"""
from __future__ import annotations

import numpy as np

from .data import BASELINES_64x32, Era5Store, climatology_at

SCORED = ("z500", "t850", "t2m", "msl", "u10")
WB2_NAMES = {"z500": ("geopotential", 500), "t850": ("temperature", 850), "t2m": ("2m_temperature", None),
             "msl": ("mean_sea_level_pressure", None), "u10": ("10m_u_component_of_wind", None),
             "tp6": ("total_precipitation_6hr", None)}
UNITS = {"z500": "m²/s²", "t850": "K", "t2m": "K", "msl": "Pa", "u10": "m/s", "tp6": "m"}


def rmse(f: np.ndarray, o: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Area-weighted RMSE over the last two axes (lat, lon); ``w`` (lat,) normalised to mean 1."""
    return np.sqrt(np.mean((f - o) ** 2 * w[:, None], axis=(-2, -1)))


def acc(f: np.ndarray, o: np.ndarray, clim: np.ndarray, w: np.ndarray) -> np.ndarray:
    fa, oa = f - clim, o - clim
    fa = fa - np.mean(fa * w[:, None], axis=(-2, -1), keepdims=True)
    oa = oa - np.mean(oa * w[:, None], axis=(-2, -1), keepdims=True)
    num = np.sum(fa * oa * w[:, None], axis=(-2, -1))
    den = np.sqrt(np.sum(fa ** 2 * w[:, None], axis=(-2, -1)) * np.sum(oa ** 2 * w[:, None], axis=(-2, -1)))
    return num / np.maximum(den, 1e-30)


def first_crossing(leads_h: np.ndarray, values: np.ndarray, threshold: float, below: bool = True) -> float:
    """Lead (hours, linearly interpolated) at which ``values`` first goes below (or above) ``threshold``;
    inf if it never does within the leads given."""
    v = np.asarray(values, float)
    bad = v < threshold if below else v >= threshold
    if not bad.any():
        return float("inf")
    i = int(np.argmax(bad))
    if i == 0:
        return float(leads_h[0])
    x0, x1, y0, y1 = leads_h[i - 1], leads_h[i], v[i - 1], v[i]
    return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x1)


def horizons(leads_h, acc_curve, rmse_curve, clim_rmse) -> dict:
    return {"useful_hours": first_crossing(leads_h, acc_curve, 0.6, below=True),
            "no_skill_hours": first_crossing(leads_h, np.asarray(rmse_curve) / np.asarray(clim_rmse), 1.0,
                                             below=False)}


def score_forecasts(store: Era5Store, clim: np.ndarray, forecasts: dict, init_idx, steps: int,
                    variables=SCORED) -> dict:
    """``forecasts``: {name: callable(i0, steps) -> (steps+1, channel, lat, lon) physical units or None}.
    Returns {name: {var: {"rmse": (n_init, steps+1), "acc": ...}}} on the given initial indices."""
    w = store.lat_weights()
    ch = {v: store.channels.index(v) for v in variables}
    out = {n: {v: {"rmse": [], "acc": []} for v in variables} for n in forecasts}
    for i0 in init_idx:
        truth = store.denorm(np.asarray(store.state[i0:i0 + steps + 1], np.float32))
        cl = np.stack([store.denorm(climatology_at(clim, store.times[i0 + s])) for s in range(steps + 1)])
        for name, fn in forecasts.items():
            f = fn(i0, steps)
            for v, c in ch.items():
                if f is None or f[:, c] is None or np.isnan(f[:, c]).all():
                    out[name][v]["rmse"].append(np.full(steps + 1, np.nan))
                    out[name][v]["acc"].append(np.full(steps + 1, np.nan))
                    continue
                out[name][v]["rmse"].append(rmse(f[:, c], truth[:, c], w))
                out[name][v]["acc"].append(acc(f[:, c], truth[:, c], cl[:, c], w))
    for n in out:
        for v in out[n]:
            for k in out[n][v]:
                out[n][v][k] = np.array(out[n][v][k])
    return out


class PublishedForecasts:
    """Published forecasts on the 64 x 32 grid (WeatherBench2), returned in the store's channel layout (other
    channels NaN). Only the 00 and 12 UTC initialisations exist."""

    def __init__(self, store: Era5Store, names=None):
        import xarray as xr

        self.store = store
        self.ds = {}
        for n, url in BASELINES_64x32.items():
            if names is not None and n not in names:
                continue
            try:
                self.ds[n] = xr.open_zarr(url, storage_options={"token": "anon"})
            except Exception:  # noqa: BLE001 - a missing baseline is reported as absent
                pass

    def __call__(self, name: str):
        ds = self.ds.get(name)
        store = self.store

        def fn(i0, steps):
            if ds is None:
                return None
            t0 = store.times[i0]
            if t0 not in ds.time.values:
                return None
            f = np.full((steps + 1, len(store.channels), len(store.lat), len(store.lon)), np.nan, np.float32)
            sub = ds.sel(time=t0)
            pt = sub.prediction_timedelta.values
            hours = (pt / np.timedelta64(1, "h")) if np.issubdtype(pt.dtype, np.timedelta64) else pt.astype(float)
            pos_of = {int(round(h)): j for j, h in enumerate(hours)}
            pick = [(s_, pos_of[6 * s_]) for s_ in range(1, steps + 1) if 6 * s_ in pos_of]
            if not pick:
                return None
            for v, (wb, lev) in WB2_NAMES.items():
                if v not in store.channels or wb not in sub:
                    continue
                da = sub[wb]
                if lev is not None:
                    if "level" not in da.dims or lev not in da.level.values:
                        continue
                    da = da.sel(level=lev)
                vals = da.isel(prediction_timedelta=[j for _, j in pick]).transpose(
                    "prediction_timedelta", "latitude", "longitude").values[:, ::-1]
                f[[s_ for s_, _ in pick], store.channels.index(v)] = vals
            f[0] = np.nan                                   # lead 0 is the analysis, not a forecast
            return f

        return fn


def persistence(store: Era5Store):
    def fn(i0, steps):
        x = store.denorm(np.asarray(store.state[i0], np.float32))
        return np.repeat(x[None], steps + 1, 0)
    return fn


def climatology_forecast(store: Era5Store, clim):
    def fn(i0, steps):
        return np.stack([store.denorm(climatology_at(clim, store.times[i0 + s])) for s in range(steps + 1)])
    return fn
