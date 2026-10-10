"""ERA5 from WeatherBench2 (public Google Cloud bucket, no account) as a local, normalised array for training and
evaluating global forecast models, plus the climatology needed for anomaly scores.

The default grid is 64 x 32 (5.625 degrees), the coarsest WeatherBench2 grid: a model trains on a laptop CPU, and the
published forecasts of IFS HRES, GraphCast, Pangu-Weather, Keisler and NeuralGCM exist on the same grid for 2020, so
the skill of a new model can be put next to theirs case by case.

``download(dest, years)`` writes ``dest/state.npy`` (time, channel, lat, lon) in float16 after normalising every
channel by its mean and standard deviation (kept in ``dest/meta.json``), ``dest/constants.npy`` (land-sea mask and
surface geopotential) and ``dest/times.npy``. :class:`Era5Store` reads it back lazily (memory map).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WB2 = "gs://weatherbench2/datasets"
ERA5_64x32 = f"{WB2}/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
CLIMATOLOGY_64x32 = f"{WB2}/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr"
BASELINES_64x32 = {          # published forecasts on the same grid (initialised 00/12 UTC)
    "IFS HRES": f"{WB2}/hres/2016-2022-0012-64x32_equiangular_conservative.zarr",
    "Pangu-Weather": f"{WB2}/pangu/2018-2022_0012_64x32_equiangular_conservative.zarr",
    "Keisler (GNN)": f"{WB2}/keisler/2020-64x32_equiangular_conservative.zarr",
    "NeuralGCM": f"{WB2}/neuralgcm_deterministic/2020-64x32_equiangular_conservative.zarr",
}

LEVELS = (250, 500, 700, 850)
UPPER = {"geopotential": "z", "temperature": "t", "u_component_of_wind": "u", "v_component_of_wind": "v",
         "specific_humidity": "q"}
SURFACE = {"2m_temperature": "t2m", "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10",
           "mean_sea_level_pressure": "msl", "total_precipitation_6hr": "tp6"}
CONSTANTS = {"land_sea_mask": "lsm", "geopotential_at_surface": "orog"}


def channel_names(levels=LEVELS) -> list[str]:
    return [f"{s}{lev}" for s in UPPER.values() for lev in levels] + list(SURFACE.values())


def _open(url: str):
    import xarray as xr

    return xr.open_zarr(url, storage_options={"token": "anon"})


def _to_array(ds, t0, t1, levels) -> np.ndarray:
    """(time, channel, lat, lon) float32 for one time window, latitude from north to south."""
    sub = ds.sel(time=slice(t0, t1))
    parts = [sub[v].sel(level=list(levels)).transpose("time", "level", "latitude", "longitude").values
             for v in UPPER]
    parts = [p.reshape(p.shape[0], -1, *p.shape[2:]) for p in parts]
    parts += [sub[v].transpose("time", "latitude", "longitude").values[:, None] for v in SURFACE]
    x = np.concatenate(parts, axis=1).astype(np.float32)
    return x[:, :, ::-1]                                       # WB2 latitude ascends; maps read north up


def download(dest: str | Path, years=(1990, 2022), levels=LEVELS, url: str = ERA5_64x32, log=print,
             normalization: str | Path | dict | None = None) -> Path:
    """``normalization``: mean and std per channel to use (a ``meta.json`` of another store, or the file shipped
    next to a checkpoint) instead of computing them; a trained model only works with the normalisation it was
    trained with."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    ds = _open(url)
    times = ds.time.sel(time=slice(f"{years[0]}-01-01", f"{years[1]}-12-31")).values
    names = channel_names(levels)
    nlon = ds.sizes["longitude"]
    if normalization is not None:
        norm = normalization if isinstance(normalization, dict) else json.loads(Path(normalization).read_text())
        if list(norm["channels"]) != names:
            raise ValueError("the normalisation file is for other channels")
        mean, std = np.array(norm["mean"], np.float32), np.array(norm["std"], np.float32)
    else:
        mean, std = _normalisation(ds, years, levels, nlon)
    _write_store(dest, ds, times, names, mean, std, years, levels, url, log)
    return dest


def _normalisation(ds, years, levels, nlon):
    """Mean and std per channel (area-weighted) from the first and the middle year."""
    ref = _to_array(ds, f"{years[0]}-01-01", f"{years[0]}-12-31", levels)
    if years[1] > years[0]:
        ref = np.concatenate([ref, _to_array(ds, f"{(years[0] + years[1]) // 2}-01-01",
                                             f"{(years[0] + years[1]) // 2}-12-31", levels)])
    w = np.cos(np.deg2rad(ds.latitude.values[::-1]))[None, None, :, None]
    mean = (ref * w).sum((0, 2, 3)) / (w.sum() * ref.shape[0] * nlon)
    std = np.sqrt((((ref - mean[None, :, None, None]) ** 2) * w).sum((0, 2, 3)) / (w.sum() * ref.shape[0] * nlon))
    return mean, np.maximum(std, 1e-12)


def _write_store(dest, ds, times, names, mean, std, years, levels, url, log):
    nlat, nlon = ds.sizes["latitude"], ds.sizes["longitude"]
    out = np.lib.format.open_memmap(dest / "state.npy", mode="w+", dtype=np.float16,
                                    shape=(len(times), len(names), nlat, nlon))
    k = 0
    for y in range(years[0], years[1] + 1):
        x = _to_array(ds, f"{y}-01-01", f"{y}-12-31", levels)
        out[k:k + len(x)] = ((x - mean[None, :, None, None]) / std[None, :, None, None]).astype(np.float16)
        k += len(x)
        log(f"ERA5 {y}: {len(x)} times")
    out.flush()
    const = np.stack([ds[v].transpose("latitude", "longitude").values[::-1] for v in CONSTANTS]).astype(np.float32)
    const = (const - const.mean((1, 2), keepdims=True)) / (const.std((1, 2), keepdims=True) + 1e-12)
    np.save(dest / "constants.npy", const)
    np.save(dest / "times.npy", times[:k])
    lat = ds.latitude.values[::-1].astype(float)
    (dest / "meta.json").write_text(json.dumps({
        "channels": names, "mean": mean.tolist(), "std": std.tolist(), "levels": list(levels),
        "lat": lat.tolist(), "lon": ds.longitude.values.astype(float).tolist(), "source": url,
        "step_hours": 6}))


@dataclass
class Era5Store:
    """Normalised ERA5 written by :func:`download` (memory mapped)."""
    path: Path

    def __post_init__(self):
        self.path = Path(self.path)
        self.state = np.load(self.path / "state.npy", mmap_mode="r")
        self.constants = np.load(self.path / "constants.npy")
        self.times = np.load(self.path / "times.npy")
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.channels = self.meta["channels"]
        self.mean = np.array(self.meta["mean"], np.float32)
        self.std = np.array(self.meta["std"], np.float32)
        self.lat = np.array(self.meta["lat"])
        self.lon = np.array(self.meta["lon"])

    def index(self, time) -> int:
        i = int(np.searchsorted(self.times, np.datetime64(time, "ns")))
        if i >= len(self.times) or self.times[i] != np.datetime64(time, "ns"):
            raise KeyError(f"{time} not in the store")
        return i

    def years(self, y0: int, y1: int) -> np.ndarray:
        """Indices of the times in years [y0, y1]."""
        yr = self.times.astype("datetime64[Y]").astype(int) + 1970
        return np.nonzero((yr >= y0) & (yr <= y1))[0]

    def denorm(self, x: np.ndarray, channel: str | None = None) -> np.ndarray:
        if channel is None:
            return x * self.std[:, None, None] + self.mean[:, None, None]
        c = self.channels.index(channel)
        return x * self.std[c] + self.mean[c]

    def lat_weights(self) -> np.ndarray:
        w = np.cos(np.deg2rad(self.lat))
        return w / w.mean()


def climatology(store: Era5Store, url: str = CLIMATOLOGY_64x32) -> np.ndarray:
    """Normalised 1990-2019 climatology (dayofyear 1..366, hour 0/6/12/18, channel, lat, lon) in the store's units,
    cached next to the store."""
    f = store.path / "climatology.npy"
    if f.exists():
        return np.load(f, mmap_mode="r")
    ds = _open(url)
    lev = store.meta["levels"]
    parts = [ds[v].sel(level=lev).transpose("dayofyear", "hour", "level", "latitude", "longitude").values
             for v in UPPER]
    parts = [p.reshape(p.shape[0], p.shape[1], -1, *p.shape[3:]) for p in parts]
    parts += [ds[v].transpose("dayofyear", "hour", "latitude", "longitude").values[:, :, None] for v in SURFACE]
    c = np.concatenate(parts, axis=2)[:, :, :, ::-1].astype(np.float32)
    c = (c - store.mean[None, None, :, None, None]) / store.std[None, None, :, None, None]
    np.save(f, c.astype(np.float32))
    return np.load(f, mmap_mode="r")


def climatology_at(clim: np.ndarray, time) -> np.ndarray:
    t = np.datetime64(time, "ns").astype("datetime64[h]").astype(object)
    doy = t.timetuple().tm_yday
    return np.asarray(clim[doy - 1, t.hour // 6])
