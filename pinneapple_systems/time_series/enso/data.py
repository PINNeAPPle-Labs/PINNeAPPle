"""Openly licensed ENSO observations, downloaded once, reduced to monthly tropical-Pacific products and cached with
the SHA-256 of every source file.

Sources (all without credentials; NOAA data are public domain, ERA5 is under the Copernicus licence):

* **SST** -- NOAA OISST v2.1 (AVHRR-only, daily, 0.25 deg; Huang et al. 2021, J. Climate 34:2923; Reynolds et al.
  2007, J. Climate 20:5473) from the NOAA Open Data Dissemination bucket on AWS
  (``noaa-cdr-sea-surface-temp-optimum-interpolation-pds``). :func:`oisst_monthly` averages every final (not
  preliminary) daily file of a month into 1 deg cells over the tropical Pacific. The Nino indices are cos-latitude
  weighted box means of that field (:data:`NINO_BOXES`).
* **Thermocline** -- 20 deg C isotherm depth (D20) from the moored-buoy (TAO/TRITON) and, optionally, XBT profiles
  of the World Ocean Database (Boyer et al. 2018, NOAA Atlas NESDIS 87; AWS bucket ``noaa-wod-pds``).
  :func:`wod_d20_casts` interpolates D20 in every quality-controlled cast; :func:`d20_monthly` bins them into
  monthly longitude bins. The warm-water-volume proxy is the zonal mean D20 anomaly over 5S-5N, 120E-80W (the
  region of Meinen & McPhaden 2000, J. Climate 13:3551).
* **Atmosphere** -- ERA5 (Hersbach et al. 2020, QJRMS 146:1999) 6-hourly 10 m wind, mean sea-level pressure and SST
  on the 64 x 32 (5.625 deg) grid of WeatherBench 2 (Rasp et al. 2024, JAMES 16:e2023MS004019; bucket
  ``gs://weatherbench2``), 1959-01 to 2023-01: :func:`era5_monthly`. Zonal wind stress from the bulk formula, and a
  Tahiti-minus-Darwin pressure index (:func:`soi_from_mslp`) at the nearest grid cells.
* **Cross-checks** -- NCEI 0-700 m ocean heat content, 3-month means 1955-2020 (Levitus et al. 2012, GRL
  39:L10603; bucket ``noaa-cdr-ocean-heat-content-pds``): :func:`ncei_heat_content`. The statsmodels El Nino data
  set (Nino 1+2 SST 1950-2010, NOAA CPC): :func:`statsmodels_nino12`.

Every downloaded file is hashed and recorded in a manifest (``<cache>/enso/manifest_<product>.json``: URL ->
SHA-256 and size). Passing ``manifest=`` (a manifest written by an earlier run, e.g. the one committed with the
study) makes a download that does not reproduce a recorded hash raise instead of returning different data. Large
raw files are deleted once reduced; their hashes stay in the manifest.
"""
from __future__ import annotations

import calendar
import gzip
import hashlib
import json
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["NINO_BOXES", "SOURCES", "Manifest", "manifests", "Gridded", "fetch", "month_range", "oisst_monthly", "box_mean",
           "nino_indices", "wod_d20_casts", "d20_from_profile", "fetch_to_file", "d20_climatology", "d20_monthly", "era5_monthly", "wind_stress",
           "soi_from_mslp", "ncei_heat_content", "statsmodels_nino12", "anomalies", "coarsen"]

#: (lat_min, lat_max, lon_min, lon_max) in degrees north / east (0-360)
NINO_BOXES = {"nino34": (-5.0, 5.0, 190.0, 240.0), "nino3": (-5.0, 5.0, 210.0, 270.0),
              "nino4": (-5.0, 5.0, 160.0, 210.0), "nino12": (-10.0, 0.0, 270.0, 280.0)}

OISST_BUCKET = "https://noaa-cdr-sea-surface-temp-optimum-interpolation-pds.s3.amazonaws.com"
WOD_BUCKET = "https://noaa-wod-pds.s3.amazonaws.com"
OHC_BUCKET = "https://noaa-cdr-ocean-heat-content-pds.s3.amazonaws.com"
WB2_ERA5 = ("https://storage.googleapis.com/weatherbench2/datasets/era5/"
            "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr")

SOURCES: dict[str, dict[str, str]] = {
    "oisst": {"name": "NOAA OISST v2.1 AVHRR-only daily 0.25 deg", "license": "public domain (NOAA)",
              "url": OISST_BUCKET + "/data/v2.1/avhrr/",
              "doi": "10.25921/RE9P-PT57",
              "citation": "Huang, B. et al. (2021) Improvements of the Daily Optimum Interpolation Sea Surface "
                          "Temperature (DOISST) Version 2.1, J. Climate 34, 2923-2939."},
    "wod": {"name": "World Ocean Database (moored buoys MRB, XBT)", "license": "public domain (NOAA NCEI)",
            "url": WOD_BUCKET + "/", "citation": "Boyer, T. P. et al. (2018) World Ocean Database 2018, NOAA "
                                                 "Atlas NESDIS 87."},
    "era5": {"name": "ERA5 6-hourly on the WeatherBench 2 64x32 grid", "license": "Copernicus licence "
             "(free use and redistribution with attribution; licence text in gs://weatherbench2/datasets/era5/"
             "LICENSE)", "url": WB2_ERA5,
             "citation": "Hersbach, H. et al. (2020) The ERA5 global reanalysis, Q. J. R. Meteorol. Soc. 146, "
                         "1999-2049; Rasp, S. et al. (2024) WeatherBench 2, JAMES 16, e2023MS004019. Contains "
                         "modified Copernicus Climate Change Service information."},
    "ncei_ohc": {"name": "NCEI 0-700 m ocean heat content anomaly, 3-month means", "license": "public domain "
                 "(NOAA NCEI)", "url": OHC_BUCKET + "/data/derived/heat_content_anomaly_0-700_seasonal.nc",
                 "citation": "Levitus, S. et al. (2012) World ocean heat content and thermosteric sea level change "
                             "(0-2000 m), 1955-2010, Geophys. Res. Lett. 39, L10603."},
    "statsmodels_elnino": {"name": "statsmodels El Nino data set (Nino 1+2 SST 1950-2010, NOAA CPC)",
                           "license": "public domain (NOAA), BSD (statsmodels)",
                           "url": "statsmodels.datasets.elnino", "citation": "Seabold, S. & Perktold, J. (2010) "
                                                                            "statsmodels, Proc. SciPy."},
}


def _cache(cache_dir: str | None) -> Path:
    from ..datasets.public import _cache_dir
    d = _cache_dir(cache_dir) / "enso"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class Manifest:
    """URL -> {sha256, bytes} of every source file of a product. ``expected`` (an earlier manifest) pins hashes."""

    def __init__(self, path: Path, expected: str | Path | dict | None = None):
        self.path = Path(path)
        self.files: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            self.files = json.loads(self.path.read_text()).get("files", {})
        if isinstance(expected, (str, Path)):
            raw = Path(expected).read_bytes()
            expected = json.loads(gzip.decompress(raw) if str(expected).endswith(".gz") else raw)
        self.expected = (expected or {}).get("files", {})

    def check(self, url: str, data: bytes) -> str:
        h = _sha256(data)
        for ref in (self.expected.get(url), self.files.get(url)):
            if ref is not None and ref["sha256"] != h:
                raise ValueError(f"{url}: checksum mismatch (recorded {ref['sha256']}, got {h})")
        self.files[url] = {"sha256": h, "bytes": len(data)}
        return h

    def record(self, url: str, sha256: str, nbytes: int) -> None:
        ref = self.expected.get(url)
        if ref is not None and ref["sha256"] != sha256:
            raise ValueError(f"{url}: checksum mismatch (recorded {ref['sha256']}, got {sha256})")
        self.files[url] = {"sha256": sha256, "bytes": int(nbytes)}

    def save(self, meta: dict | None = None) -> str:
        body = {"meta": meta or {}, "files": dict(sorted(self.files.items()))}
        self.path.write_text(json.dumps(body, indent=0))
        return str(self.path)


def manifests(cache_dir: str | None = None) -> dict[str, Path]:
    """The manifests written so far in the cache: product -> path of ``manifest_<product>.json``."""
    root = _cache(cache_dir)
    return {p.stem[len("manifest_"):]: p for p in sorted(root.glob("manifest_*.json"))}


def fetch(url: str, *, timeout: int = 300, retries: int = 4) -> bytes:
    """Bytes at ``url`` (through the environment's proxy settings), retried with exponential backoff."""
    import time
    err = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PINNeAPPle-enso"})
            with urllib.request.urlopen(req, timeout=timeout) as r:      # noqa: S310 -- fixed https hosts
                return r.read()
        except Exception as e:  # noqa: BLE001 -- network errors are retried, then re-raised
            err = e
            time.sleep(2 ** (k + 1))
    raise err


def _s3_keys(bucket: str, prefix: str) -> list[str]:
    keys, token = [], None
    while True:
        q = f"{bucket}/?list-type=2&prefix={prefix}" + (f"&continuation-token={urllib.parse.quote(token)}"
                                                         if token else "")
        x = fetch(q).decode()
        keys += re.findall(r"<Key>([^<]+)</Key>", x)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", x)
        if not m:
            return keys
        token = m.group(1)



def month_range(start: str, end: str) -> list[str]:
    """'YYYY-MM' strings from ``start`` to ``end`` inclusive."""
    y, m = map(int, start.split("-"))
    y1, m1 = map(int, end.split("-"))
    out = []
    while (y, m) <= (y1, m1):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


@dataclass
class Gridded:
    """Monthly fields (nt, ny, nx) on a regular lat/lon grid (cell centres, lon in 0-360)."""
    time: list[str]
    lat: np.ndarray
    lon: np.ndarray
    values: dict[str, np.ndarray]
    source: dict[str, Any] = field(default_factory=dict)

    def sel(self, start: str | None = None, end: str | None = None) -> Gridded:
        i0 = self.time.index(start) if start else 0
        i1 = self.time.index(end) + 1 if end else len(self.time)
        return Gridded(self.time[i0:i1], self.lat, self.lon, {k: v[i0:i1] for k, v in self.values.items()},
                       self.source)


# ---------------------------------------------------------------------------
# OISST
# ---------------------------------------------------------------------------

_NC_LOCK = threading.Lock()   # the HDF5 library behind netCDF4 is not thread-safe


def _oisst_reduce(data: bytes, lat_sl, lon_sl, k: int) -> np.ndarray:
    import netCDF4
    with _NC_LOCK, netCDF4.Dataset("mem", memory=data) as d:
        v = d["sst"]
        v.set_auto_maskandscale(True)
        a = np.ma.filled(v[0, 0, lat_sl, lon_sl].astype(np.float64), np.nan)
    ny, nx = a.shape[0] // k, a.shape[1] // k
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a.reshape(ny, k, nx, k), axis=(1, 3))


def oisst_monthly(start: str = "1981-09", end: str | None = None, *, lat: tuple[float, float] = (-30.0, 30.0),
                  lon: tuple[float, float] = (100.0, 300.0), res: float = 1.0, cache_dir: str | None = None,
                  workers: int = 16, manifest: str | Path | None = None,
                  log: Callable[[str], None] = print) -> Gridded:
    """Monthly mean OISST v2.1 SST [deg C] in ``res``-degree cells (a multiple of 0.25) over ``lat`` x ``lon``.

    A month is used only when every day has its final (not preliminary) file. ``end`` defaults to the last such
    month. Each month is cached as ``<cache>/enso/oisst_<res>deg/YYYY-MM.npy``; the daily files are hashed into
    the manifest and not kept. A cell with ice or land in any 0.25 deg sub-cell keeps the mean of the rest; cells
    without ocean are NaN.
    """
    k = int(round(res / 0.25))
    if abs(k * 0.25 - res) > 1e-9:
        raise ValueError("res must be a multiple of 0.25 deg")
    root = _cache(cache_dir)
    out_dir = root / f"oisst_{res:g}deg_{lat[0]:g}_{lat[1]:g}_{lon[0]:g}_{lon[1]:g}"
    out_dir.mkdir(exist_ok=True)
    man = Manifest(root / "manifest_oisst.json", manifest)
    lat_c = -89.875 + 0.25 * np.arange(720)
    lon_c = 0.125 + 0.25 * np.arange(1440)
    lat_sl = slice(int(np.searchsorted(lat_c, lat[0])), int(np.searchsorted(lat_c, lat[1])))
    lon_sl = slice(int(np.searchsorted(lon_c, lon[0])), int(np.searchsorted(lon_c, lon[1])))
    if end is None:
        import datetime
        today = datetime.date.today()
        end = f"{today.year:04d}-{today.month:02d}"
    months = month_range(start, end)
    fields, used = [], []
    with ThreadPoolExecutor(workers) as ex:
        for mo in months:
            path = out_dir / f"{mo}.npy"
            if path.exists():
                fields.append(np.load(path))
                used.append(mo)
                continue
            y, m = map(int, mo.split("-"))
            nd = calendar.monthrange(y, m)[1]
            keys = [kk for kk in _s3_keys(OISST_BUCKET, f"data/v2.1/avhrr/{y:04d}{m:02d}/")
                    if kk.endswith(".nc") and "preliminary" not in kk]
            if len(keys) < nd:
                log(f"  OISST {mo}: {len(keys)} of {nd} final daily files, stopping")
                break
            urls = [f"{OISST_BUCKET}/{kk}" for kk in sorted(keys)]

            def one(u):
                b = fetch(u)
                return u, b, _oisst_reduce(b, lat_sl, lon_sl, k)

            res_ = list(ex.map(one, urls))
            for u, b, _ in res_:
                man.check(u, b)
            with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
                __import__("warnings").simplefilter("ignore", RuntimeWarning)
                f = np.nanmean(np.stack([r[2] for r in res_]), axis=0).astype(np.float32)
            np.save(path, f)
            fields.append(f)
            used.append(mo)
            if m == 12 or mo == months[-1]:
                man.save({"product": "oisst", "res_deg": res, "lat": lat, "lon": lon})
                log(f"  OISST through {mo}")
    man.save({"product": "oisst", "res_deg": res, "lat": lat, "lon": lon})
    la = lat_c[lat_sl].reshape(-1, k).mean(1)
    lo = lon_c[lon_sl].reshape(-1, k).mean(1)
    return Gridded(used, la, lo, {"sst": np.stack(fields)},
                   {**SOURCES["oisst"], "manifest": str(man.path)})


def box_mean(g: Gridded, var: str, box: tuple[float, float, float, float]) -> np.ndarray:
    """cos(latitude)-weighted mean of ``g.values[var]`` over cells whose centres lie in ``box``."""
    la0, la1, lo0, lo1 = box
    iy = (g.lat > la0) & (g.lat < la1)
    ix = (g.lon % 360 > lo0) & (g.lon % 360 < lo1)
    a = g.values[var][:, iy][:, :, ix]
    w = np.cos(np.deg2rad(g.lat[iy]))[:, None] * np.ones(ix.sum())[None, :]
    w = np.where(np.isfinite(a), w[None], 0.0)
    return np.nansum(np.nan_to_num(a) * w, axis=(1, 2)) / np.maximum(w.sum(axis=(1, 2)), 1e-12)


def nino_indices(g: Gridded, var: str = "sst") -> dict[str, np.ndarray]:
    """Nino 3.4, 3, 4 and 1+2 box means (not anomalies) of ``g``."""
    return {k: box_mean(g, var, b) for k, b in NINO_BOXES.items()}


def coarsen(g: Gridded, ky: int, kx: int) -> Gridded:
    """Block means of ky x kx cells (NaN-aware; trailing cells that do not fill a block are dropped)."""
    ny, nx = (len(g.lat) // ky) * ky, (len(g.lon) // kx) * kx
    vals = {}
    for k, v in g.values.items():
        a = v[:, :ny, :nx].reshape(len(v), ny // ky, ky, nx // kx, kx)
        with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            vals[k] = np.nanmean(a, axis=(2, 4))
    return Gridded(g.time, g.lat[:ny].reshape(-1, ky).mean(1), g.lon[:nx].reshape(-1, kx).mean(1), vals, g.source)


def anomalies(x: np.ndarray, time: Sequence[str], base: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Anomalies from the calendar-month climatology of the period ``base`` = ('YYYY-MM', 'YYYY-MM') inclusive.

    Returns (anomalies, climatology (12, ...)). Use a base inside the training period so that no information of
    the test period enters the anomalies.
    """
    x = np.asarray(x, float)
    mon = np.array([int(t[5:7]) for t in time])
    tt = np.array(time)
    inb = (tt >= base[0]) & (tt <= base[1])
    clim = np.stack([np.nanmean(x[inb & (mon == m)], axis=0) for m in range(1, 13)])
    return x - clim[mon - 1], clim


# ---------------------------------------------------------------------------
# World Ocean Database: 20 deg C isotherm depth
# ---------------------------------------------------------------------------

def d20_from_profile(z: np.ndarray, T: np.ndarray, *, iso: float = 20.0, max_gap: float = 60.0) -> float:
    """Depth [m] where the profile first crosses ``iso`` going down (linear interpolation between the bracketing
    levels), or NaN when the profile does not start above ``iso`` within the top 30 m, does not reach below it, or
    the bracketing levels are more than ``max_gap`` m apart."""
    z, T = np.asarray(z, float), np.asarray(T, float)
    ok = np.isfinite(z) & np.isfinite(T)
    z, T = z[ok], T[ok]
    if len(z) < 2:
        return float("nan")
    o = np.argsort(z)
    z, T = z[o], T[o]
    if z[0] > 30.0 or T[0] <= iso:
        return float("nan")
    below = np.nonzero(T < iso)[0]
    if len(below) == 0:
        return float("nan")
    j = below[0]
    if z[j] - z[j - 1] > max_gap:
        return float("nan")
    return float(z[j - 1] + (T[j - 1] - iso) * (z[j] - z[j - 1]) / (T[j - 1] - T[j]))


def fetch_to_file(url: str, path: Path, *, timeout: int = 1800, retries: int = 4, chunk: int = 1 << 22) -> str:
    """Stream ``url`` to ``path`` and return its SHA-256 (large files never sit in memory)."""
    import time
    err = None
    for k in range(retries):
        try:
            h = hashlib.sha256()
            req = urllib.request.Request(url, headers={"User-Agent": "PINNeAPPle-enso"})
            with urllib.request.urlopen(req, timeout=timeout) as r, open(path, "wb") as f:  # noqa: S310
                while True:
                    b = r.read(chunk)
                    if not b:
                        break
                    h.update(b)
                    f.write(b)
            return h.hexdigest()
        except urllib.error.HTTPError:
            raise
        except Exception as e:  # noqa: BLE001 -- network errors are retried, then re-raised
            err = e
            time.sleep(2 ** (k + 1))
    raise err


def _wod_reduce(path: str | Path, lat: tuple[float, float], lon: tuple[float, float]) -> dict[str, np.ndarray]:
    """D20 of the quality-controlled casts of one WOD ragged-array file inside the box (reads only those casts)."""
    import netCDF4
    empty = {"time": np.zeros(0), "lat": np.zeros(0), "lon": np.zeros(0), "d20": np.zeros(0)}
    with _NC_LOCK, netCDF4.Dataset(str(path)) as d:
        if "Temperature" not in d.variables:
            return empty
        la = np.ma.filled(d["lat"][:], np.nan).astype(float)
        lo = np.ma.filled(d["lon"][:], np.nan).astype(float) % 360
        sel = (la >= lat[0]) & (la <= lat[1]) & (lo >= lon[0]) & (lo <= lon[1])
        if not sel.any():
            return empty
        t = np.ma.filled(d["time"][:], np.nan).astype(float)          # days since 1770-01-01
        zr = np.ma.filled(d["z_row_size"][:], 0).astype(np.int64)
        tr = np.ma.filled(d["Temperature_row_size"][:], 0).astype(np.int64)
        pflag = np.ma.filled(d["Temperature_WODprofileflag"][:], 9) if "Temperature_WODprofileflag" in \
            d.variables else np.zeros(len(la), int)
        zo = np.concatenate([[0], np.cumsum(zr)])
        to = np.concatenate([[0], np.cumsum(tr)])
        idx = np.nonzero(sel & (pflag == 0) & (tr > 1) & (tr == zr))[0]
        out = np.full(len(idx), np.nan)
        vz, vzf, vT, vTf = d["z"], d["z_WODflag"], d["Temperature"], d["Temperature_WODflag"]
        for v in (vz, vzf, vT, vTf):
            v.set_auto_mask(False)
        # read contiguous runs of selected casts in blocks to limit the number of reads
        for b0 in range(0, len(idx), 512):
            blk = idx[b0:b0 + 512]
            zs, ze, ts_, te = zo[blk[0]], zo[blk[-1] + 1], to[blk[0]], to[blk[-1] + 1]
            if ze - zs > 5_000_000:                                    # sparse selection: per cast
                parts = [(vz[zo[i]:zo[i + 1]], vzf[zo[i]:zo[i + 1]], vT[to[i]:to[i + 1]], vTf[to[i]:to[i + 1]])
                         for i in blk]
            else:
                Z, ZF, T, TF = vz[zs:ze], vzf[zs:ze], vT[ts_:te], vTf[ts_:te]
                parts = [(Z[zo[i] - zs:zo[i + 1] - zs], ZF[zo[i] - zs:zo[i + 1] - zs],
                          T[to[i] - ts_:to[i + 1] - ts_], TF[to[i] - ts_:to[i + 1] - ts_]) for i in blk]
            for n, (zz, zf, TT, Tf) in enumerate(parts):
                good = (zf == 0) & (Tf == 0) & (np.abs(TT) < 50) & (zz > -1)
                out[b0 + n] = d20_from_profile(zz[good], TT[good])
    keep = np.isfinite(out)
    return {"time": t[idx][keep], "lat": la[idx][keep], "lon": lo[idx][keep], "d20": out[keep]}


def wod_d20_casts(years: Sequence[int], kinds: Sequence[str] = ("mrb",), *, lat: tuple[float, float] = (-8.0, 8.0),
                  lon: tuple[float, float] = (120.0, 280.0), cache_dir: str | None = None,
                  manifest: str | Path | None = None, log: Callable[[str], None] = print) -> dict[str, np.ndarray]:
    """D20 of every quality-controlled WOD cast (flags 0) in the box, from the yearly files ``wod_<kind>_<year>.nc``
    (``kind`` in mrb = moored buoys, xbt, pfl = profiling floats, ctd, ...).

    Returns arrays ``time`` (days since 1770-01-01), ``lat``, ``lon``, ``d20`` [m] and ``kind``. Each year and kind
    is cached as a small ``.npz``; the raw file (up to a few GB) is streamed to disk, hashed into the manifest and
    deleted.
    """
    root = _cache(cache_dir)
    d = root / "wod_d20"
    d.mkdir(exist_ok=True)
    man = Manifest(root / "manifest_wod.json", manifest)
    parts = []
    for kind in kinds:
        for y in years:
            path = d / f"{kind}_{y}_{lat[0]:g}_{lat[1]:g}_{lon[0]:g}_{lon[1]:g}.npz"
            if not path.exists():
                url = f"{WOD_BUCKET}/{y}/wod_{kind}_{y}.nc"
                tmp = d / f"_tmp_{kind}_{y}.nc"
                try:
                    sha = fetch_to_file(url, tmp)
                except urllib.error.HTTPError as e:
                    if e.code in (403, 404):                           # no file for that instrument and year
                        np.savez(path, time=np.zeros(0), lat=np.zeros(0), lon=np.zeros(0), d20=np.zeros(0))
                        continue
                    raise
                man.record(url, sha, tmp.stat().st_size)
                r = _wod_reduce(tmp, lat, lon)
                tmp.unlink()
                np.savez(path, **r)
                man.save({"product": "wod_d20", "lat": lat, "lon": lon})
                log(f"  WOD {kind} {y}: {len(r['d20'])} casts with D20")
            z = np.load(path)
            parts.append({**{k: z[k] for k in ("time", "lat", "lon", "d20")},
                          "kind": np.full(len(z["d20"]), kind)})
    man.save({"product": "wod_d20", "lat": lat, "lon": lon})
    return {k: np.concatenate([p[k] for p in parts]) for k in ("time", "lat", "lon", "d20", "kind")}


def d20_monthly(casts: dict[str, np.ndarray], months: Sequence[str], *, lat: tuple[float, float] = (-5.0, 5.0),
                lon_edges: Sequence[float] = tuple(range(130, 281, 10)), min_casts: int = 1,
                daily_per_platform: bool = True) -> np.ndarray:
    """Monthly mean D20 [m] per longitude bin (``len(lon_edges) - 1`` bins) for casts with latitude in ``lat``;
    NaN where a bin has fewer than ``min_casts`` casts. Each cast counts once (a mooring reporting daily weighs as
    much as 30 XBT drops); ``daily_per_platform`` is kept for the record (WOD moored data are already daily)."""
    import datetime
    base = datetime.date(1770, 1, 1)
    ym = np.array([(base + datetime.timedelta(days=float(t))).strftime("%Y-%m") for t in casts["time"]]) \
        if len(casts["time"]) else np.zeros(0, str)
    edges = np.asarray(lon_edges, float)
    sel = (casts["lat"] >= lat[0]) & (casts["lat"] <= lat[1])
    out = np.full((len(months), len(edges) - 1), np.nan)
    pos = {m: i for i, m in enumerate(months)}
    ib = np.searchsorted(edges, casts["lon"], side="right") - 1
    S = np.zeros_like(out)
    N = np.zeros_like(out)
    for m, b, v, s in zip(ym, ib, casts["d20"], sel, strict=False):
        if s and m in pos and 0 <= b < len(edges) - 1:
            S[pos[m], b] += v
            N[pos[m], b] += 1
    ok = N >= min_casts
    out[ok] = S[ok] / N[ok]
    return out


def d20_climatology(casts: dict[str, np.ndarray], lon_edges: Sequence[float], lat_edges: Sequence[float],
                    base: tuple[str, str]) -> np.ndarray:
    """Mean D20 [m] of the casts in each (lat, lon) bin over the months of ``base`` (each calendar month weighted
    equally); NaN where a bin has no cast. Shape (len(lat_edges) - 1, len(lon_edges) - 1)."""
    import datetime
    t0 = datetime.date(1770, 1, 1)
    ym = np.array([(t0 + datetime.timedelta(days=float(t))).strftime("%Y-%m") for t in casts["time"]])
    sel = (ym >= base[0]) & (ym <= base[1])
    mon = np.array([int(m[5:7]) for m in ym])
    iy = np.searchsorted(np.asarray(lat_edges, float), casts["lat"], side="right") - 1
    ix = np.searchsorted(np.asarray(lon_edges, float), casts["lon"], side="right") - 1
    ny, nx = len(lat_edges) - 1, len(lon_edges) - 1
    S = np.zeros((12, ny, nx))
    N = np.zeros((12, ny, nx))
    ok = sel & (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
    np.add.at(S, (mon[ok] - 1, iy[ok], ix[ok]), casts["d20"][ok])
    np.add.at(N, (mon[ok] - 1, iy[ok], ix[ok]), 1)
    with np.errstate(all="ignore"):
        M = S / N
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        return np.nanmean(M, axis=0)


# ---------------------------------------------------------------------------
# ERA5 (WeatherBench 2, 64 x 32)
# ---------------------------------------------------------------------------

def era5_monthly(variables: Sequence[str] = ("10m_u_component_of_wind", "10m_v_component_of_wind",
                                             "mean_sea_level_pressure", "sea_surface_temperature"),
                 *, cache_dir: str | None = None, manifest: str | Path | None = None, workers: int = 8,
                 derived: bool = True, log: Callable[[str], None] = print) -> Gridded:
    """Monthly means of 6-hourly ERA5 fields on the 64 x 32 WeatherBench 2 grid, 1959-01 to 2022-12, global.

    With ``derived`` the monthly mean of the bulk zonal and meridional wind stress (:func:`wind_stress` of each
    6-hourly wind) is added as ``taux``/``tauy`` [N m-2]. Every zarr chunk is hashed into the manifest; the monthly
    fields are cached in ``<cache>/enso/era5_wb2_64x32_monthly.npz``.
    """
    import numcodecs
    root = _cache(cache_dir)
    path = root / "era5_wb2_64x32_monthly.npz"
    meta_src = {**SOURCES["era5"]}
    if path.exists():
        z = np.load(path, allow_pickle=False)
        vals = {k[2:]: z[k] for k in z.files if k.startswith("v_")}
        return Gridded([str(t) for t in z["time"]], z["lat"], z["lon"], vals, meta_src)
    man = Manifest(root / "manifest_era5.json", manifest)
    mb = fetch(WB2_ERA5 + "/.zmetadata")
    man.check(WB2_ERA5 + "/.zmetadata", mb)
    meta = json.loads(mb)["metadata"]

    def arr(name):
        a = meta[f"{name}/.zarray"]
        codec = numcodecs.get_codec(a["compressor"]) if a["compressor"] else None
        return a, codec

    def chunk(name, idx):
        a, codec = arr(name)
        key = ".".join(map(str, idx))
        url = f"{WB2_ERA5}/{name}/{key}"
        b = fetch(url)
        raw = codec.decode(b) if codec else b
        shape = [min(c, s - i * c) for c, s, i in zip(a["chunks"], a["shape"], idx, strict=False)]
        full = np.frombuffer(raw, dtype=np.dtype(a["dtype"])).reshape(a["chunks"])
        return url, b, full[tuple(slice(0, s) for s in shape)]

    nt = arr("time")[0]["shape"][0]
    tparts = [chunk("time", (i,)) for i in range(int(np.ceil(nt / arr("time")[0]["chunks"][0])))]
    for url, b, _ in tparts:
        man.check(url, b)
    hours = np.concatenate([p[2] for p in tparts])
    lat_url, lat_b, lat = chunk("latitude", (0,))
    lon_url, lon_b, lon = chunk("longitude", (0,))
    man.check(lat_url, lat_b)
    man.check(lon_url, lon_b)
    lat, lon = lat.astype(float), lon.astype(float)
    import datetime
    t0 = datetime.datetime(1959, 1, 1)
    ym = np.array([(t0 + datetime.timedelta(hours=float(h))).strftime("%Y-%m") for h in hours])
    months = month_range("1959-01", "2022-12")
    pos = {m: i for i, m in enumerate(months)}
    mi = np.array([pos.get(m, -1) for m in ym])
    a0 = arr(variables[0])[0]
    ct, nchunks = a0["chunks"][0], int(np.ceil(a0["shape"][0] / a0["chunks"][0]))
    names = list(variables)
    stress = derived and "10m_u_component_of_wind" in names and "10m_v_component_of_wind" in names
    S = {n: np.zeros((len(months), len(lon), len(lat))) for n in names + (["taux", "tauy"] if stress else [])}
    N = np.zeros(len(months))

    def block(i):
        return i, [chunk(n, (i, 0, 0)) for n in names]

    with ThreadPoolExecutor(workers) as ex:
        for i, parts in ex.map(block, range(nchunks)):
            m = mi[i * ct: i * ct + parts[0][2].shape[0]]
            ok = m >= 0
            fields = {}
            for n, (url, b, blk) in zip(names, parts, strict=False):
                man.check(url, b)
                fields[n] = blk
            if stress:
                fields["taux"], fields["tauy"] = wind_stress(fields["10m_u_component_of_wind"],
                                                             fields["10m_v_component_of_wind"])
            for n, blk in fields.items():
                np.add.at(S[n], m[ok], blk[ok])
            np.add.at(N, m[ok], 1)
            if i % 100 == 0:
                log(f"  ERA5 chunk {i}/{nchunks}")
                man.save({"product": "era5_wb2_64x32"})
    out = {n: (v / N[:, None, None]).transpose(0, 2, 1).astype(np.float32) for n, v in S.items()}  # (t, lat, lon)
    o = np.argsort(lat)
    out = {k: v[:, o] for k, v in out.items()}
    lat = lat[o]
    np.savez_compressed(path, time=np.array(months), lat=lat, lon=lon, **{f"v_{k}": v for k, v in out.items()})
    man.save({"product": "era5_wb2_64x32", "derived_sha256": _sha256(path.read_bytes())})
    return Gridded(months, lat, lon, out, meta_src)


def wind_stress(u, v, rho_air: float = 1.2, cd: float = 1.3e-3):
    """Bulk wind stress tau = rho_air C_D |U| U [N m-2] from 10 m wind components [m/s] (constant C_D; Large &
    Pond 1981 give 1.1e-3 to 1.5e-3 for 4-11 m/s)."""
    s = np.hypot(u, v)
    return rho_air * cd * s * u, rho_air * cd * s * v


def soi_from_mslp(g: Gridded, var: str = "mean_sea_level_pressure", base: tuple[str, str] | None = None
                  ) -> np.ndarray:
    """Tahiti (17.5S, 149.6W) minus Darwin (12.4S, 130.9E) sea-level pressure, each standardised per calendar month
    over ``base`` (default: the whole record), the difference standardised again (the CPC/BoM construction) --
    taken at the nearest cells of ``g``, so on a coarse grid it is a proxy of the station index."""
    def nearest(la, lo):
        return int(np.argmin(np.abs(g.lat - la))), int(np.argmin(np.abs(g.lon - lo % 360)))

    base = base or (g.time[0], g.time[-1])
    p = g.values[var]
    out = []
    for la, lo in ((-17.5, -149.6), (-12.4, 130.9)):
        i, j = nearest(la, lo)
        a, _ = anomalies(p[:, i, j], g.time, base)
        mon = np.array([int(t[5:7]) for t in g.time])
        tt = np.array(g.time)
        inb = (tt >= base[0]) & (tt <= base[1])
        sd = np.array([a[inb & (mon == m)].std() for m in range(1, 13)])
        out.append(a / sd[mon - 1])
    d = out[0] - out[1]
    tt = np.array(g.time)
    inb = (tt >= base[0]) & (tt <= base[1])
    return d / d[inb].std()


# ---------------------------------------------------------------------------
# Cross-checks
# ---------------------------------------------------------------------------

def ncei_heat_content(box: tuple[float, float, float, float] = (-5.0, 5.0, 120.0, 280.0), *,
                      cache_dir: str | None = None, manifest: str | Path | None = None) -> dict[str, Any]:
    """Sum of the NCEI 0-700 m heat-content anomaly [1e18 J] over the 1-deg cells in ``box``, 3-month means
    (JFM, AMJ, JAS, OND) 1955-2020; ``mid_month`` is the centre month 'YYYY-MM' of each season."""
    import netCDF4
    root = _cache(cache_dir)
    url = SOURCES["ncei_ohc"]["url"]
    path = root / "ncei_hc_0-700_seasonal.nc"
    man = Manifest(root / "manifest_ncei_ohc.json", manifest)
    b = path.read_bytes() if path.exists() else fetch(url, timeout=1800)
    man.check(url, b)
    path.write_bytes(b)
    man.save({"product": "ncei_ohc"})
    with netCDF4.Dataset("mem", memory=b) as d:
        la, lo = d["lat"][:], d["lon"][:] % 360
        t = d["time"][:]
        iy = (la > box[0]) & (la < box[1])
        ix = (lo > box[2]) & (lo < box[3])
        hc = np.ma.filled(d["h18_hc"][:, 0][:, iy][:, :, ix].astype(float), np.nan)
    mids = []
    for v in t:
        k = int(np.floor(v))
        mids.append(f"{1955 + k // 12:04d}-{k % 12 + 1:02d}")
    return {"mid_month": mids, "hc": np.nansum(hc, axis=(1, 2)), "units": "1e18 J", "box": box,
            "source": SOURCES["ncei_ohc"]}


def statsmodels_nino12() -> dict[str, Any]:
    """Monthly Nino 1+2 SST [deg C] 1950-2010 as shipped with statsmodels (NOAA CPC); time as 'YYYY-MM'."""
    from statsmodels.datasets.elnino import load_pandas
    df = load_pandas().data
    months, vals = [], []
    cols = [c for c in df.columns if c != "YEAR"]
    for _, row in df.iterrows():
        for k, c in enumerate(cols):
            months.append(f"{int(row['YEAR']):04d}-{k + 1:02d}")
            vals.append(float(row[c]))
    return {"time": months, "sst": np.asarray(vals), "source": SOURCES["statsmodels_elnino"]}

