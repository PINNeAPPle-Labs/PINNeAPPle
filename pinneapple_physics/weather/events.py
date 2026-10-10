"""Extreme-weather case studies and the ERA5 states around them.

Events up to 10 January 2023 come from the WeatherBench2 store used for training (out of the training years).
Later events (e.g. the Rio Grande do Sul floods, 2024) are read from the full-resolution ERA5 of the ARCO bucket
(Google public data, no account) and averaged onto the model grid: every 0.25-degree point falls in one model
cell and the cell value is the area-weighted mean of its points, close to the conservative regridding WeatherBench2
uses (checked against it on a date both cover).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import SURFACE, UPPER, Era5Store

ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


@dataclass
class Event:
    key: str
    title: str
    init: str                       # initialisation (UTC)
    peak: str                       # valid time of the peak of the event
    var: str                        # field to show
    region: tuple                   # (lat_s, lat_n, lon_w, lon_e) for the regional view and the event score
    center: tuple                   # globe centre (lat, lon)
    what: str                       # one line on what happened
    score: str = "mean"             # how the event is summarised in its region: mean, max, min, sum
    vmin: float | None = None
    vmax: float | None = None
    extra_inits: list = field(default_factory=list)


EVENTS = [
    Event("brazil_bomb_cyclone_2020", "Bomb cyclone, southern Brazil (30 Jun - 1 Jul 2020)", "2020-06-26T00",
          "2020-06-30T18", "msl", (-45, -15, -70, -30), (-30.0, -50.0),
          "an explosive extratropical cyclone off Santa Catarina; gusts above 100 km/h and deaths on land",
          score="min", vmin=975, vmax=1035),
    Event("texas_freeze_2021", "Winter storm Uri, Texas (15-16 Feb 2021)", "2021-02-10T00", "2021-02-16T12",
          "t2m", (22, 50, 245, 285), (35.0, -100.0),
          "Arctic air reached the Gulf coast; the Texas grid failed and millions lost power", score="mean",
          vmin=-35, vmax=30),
    Event("pnw_heat_dome_2021", "Heat dome, Pacific Northwest (28-29 Jun 2021)", "2021-06-23T00", "2021-06-29T00",
          "t2m", (40, 62, 225, 255), (48.0, -120.0),
          "Lytton (Canada) reached 49.6 °C; a blocking high parked over the region", score="mean",
          vmin=-5, vmax=40),
    Event("brazil_frost_2021", "Cold wave and frost, south and south-east Brazil (29-30 Jul 2021)",
          "2021-07-25T00", "2021-07-30T06", "t2m", (-35, -10, -65, -35), (-22.0, -50.0),
          "polar air brought frost to coffee and sugar-cane areas of São Paulo and Minas Gerais", score="mean",
          vmin=-5, vmax=35),
    Event("europe_heat_2022", "Heat wave, western Europe (18-19 Jul 2022)", "2022-07-13T00", "2022-07-19T12",
          "t2m", (36, 60, 350, 20), (48.0, 5.0),
          "the UK passed 40 °C for the first time; fires in France, Spain and Portugal", score="mean",
          vmin=-5, vmax=40),
    Event("hurricane_ian_2022", "Hurricane Ian, Florida (28 Sep 2022)", "2022-09-24T00", "2022-09-28T18", "msl",
          (10, 40, 260, 300), (25.0, -80.0),
          "category 4 landfall in south-west Florida; too small for a 5.6-degree grid, kept as a hard case",
          score="min", vmin=990, vmax=1025),
    Event("rs_floods_2024", "Floods, Rio Grande do Sul (27 Apr - 2 May 2024)", "2024-04-26T00", "2024-05-01T00",
          "tp6", (-36, -24, -60, -46), (-28.0, -52.0),
          "a stalled front poured more than 500 mm in a week; the worst flood in the state's history",
          score="sum", vmin=0, vmax=25),
]


def by_key(key: str) -> Event:
    return next(e for e in EVENTS if e.key == key)


def _cell_index(src_lat, src_lon, lat, lon):
    """Model cell of every source latitude and longitude (equiangular grids)."""
    dlat, dlon = abs(lat[0] - lat[1]), lon[1] - lon[0]
    iy = np.clip(np.floor((lat[0] + dlat / 2 - src_lat) / dlat).astype(int), 0, len(lat) - 1)
    ix = np.floor(np.mod(src_lon - lon[0] + dlon / 2, 360.0) / dlon).astype(int) % len(lon)
    return iy, ix


def coarsen(field: np.ndarray, src_lat, src_lon, lat, lon) -> np.ndarray:
    """Area-weighted mean of a (..., src_lat, src_lon) field in each (lat, lon) model cell."""
    iy, ix = _cell_index(np.asarray(src_lat), np.asarray(src_lon), lat, lon)
    w = np.cos(np.deg2rad(src_lat))
    lead = field.shape[:-2]
    out = np.zeros(lead + (len(lat), len(lon)))
    wsum = np.zeros((len(lat), len(lon)))
    flat = field.reshape(-1, field.shape[-2], field.shape[-1])
    o = out.reshape(-1, len(lat), len(lon))
    ry = np.repeat(iy, len(src_lon))
    rx = np.tile(ix, len(src_lat))
    rw = np.repeat(w, len(src_lon))
    np.add.at(wsum, (ry, rx), rw)
    for k in range(flat.shape[0]):
        acc = np.zeros((len(lat), len(lon)))
        np.add.at(acc, (ry, rx), (flat[k] * w[:, None]).ravel())
        o[k] = acc / np.maximum(wsum, 1e-12)
    return out


def arco_states(store: Era5Store, times, log=print) -> np.ndarray:
    """Normalised states (time, channel, lat, lon) on the store's grid at ``times`` from ARCO ERA5. The 6-hour
    precipitation is the sum of the six hourly accumulations ending at each time."""
    import xarray as xr

    ds = xr.open_zarr(ARCO, storage_options={"token": "anon"}, chunks=None)
    lev = store.meta["levels"]
    out = []
    for t in times:
        t = np.datetime64(t, "ns")
        parts = []
        for v in UPPER:
            a = ds[v].sel(time=t, level=lev).values
            parts.append(coarsen(a, ds.latitude.values, ds.longitude.values, store.lat, store.lon))
        for v in SURFACE:
            if v == "total_precipitation_6hr":
                hrs = [t - np.timedelta64(h, "h") for h in range(6)]
                a = ds["total_precipitation"].sel(time=hrs).values.sum(0)
            else:
                a = ds[v].sel(time=t).values
            parts.append(coarsen(a, ds.latitude.values, ds.longitude.values, store.lat, store.lon)[None])
        x = np.concatenate(parts).astype(np.float32)
        out.append((x - store.mean[:, None, None]) / store.std[:, None, None])
        log(f"ARCO {np.datetime_as_string(t, unit='h')}")
    return np.stack(out)


def event_states(store: Era5Store, event: Event, leads: int, cache_dir=None, log=print) -> tuple[np.ndarray, np.ndarray]:
    """(times, normalised states) from init - 6 h to init + leads x 6 h, from the store or from ARCO."""
    t0 = np.datetime64(event.init, "ns")
    times = t0 + np.arange(-1, leads + 1) * np.timedelta64(6, "h")
    try:
        i = store.index(times[0])
        if store.index(times[-1]) - i == leads + 1:
            return times, np.asarray(store.state[i:i + leads + 2], np.float32)
    except KeyError:
        pass
    if cache_dir is not None:
        from pathlib import Path

        f = Path(cache_dir) / f"{event.key}_{leads}.npy"
        if f.exists():
            return times, np.load(f).astype(np.float32)
        x = arco_states(store, times, log=log)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, x)
        return times, x
    return times, arco_states(store, times, log=log)


def region_mask(lat, lon, box) -> np.ndarray:
    la = (lat >= box[0]) & (lat <= box[1])
    lo = np.mod(lon, 360.0)
    w, e = box[2] % 360, box[3] % 360
    lm = (lo >= w) & (lo <= e) if w <= e else (lo >= w) | (lo <= e)
    return la[:, None] & lm[None, :]
