"""Download a regional subset of NOAA OISST v2.1 daily SST (AVHRR satellite-based,
0.25 deg) from the NOAA PSL THREDDS OPeNDAP server.

Region: Brazil-Malvinas Confluence, South Atlantic
    lat  -47.875 .. -32.125  (64 cells, index 168..231)
    lon  296.125 .. 311.875  (64 cells, index 1184..1247)  == 63.875W .. 48.125W
Years: 1982..2024 (full calendar years available in the archive).

Output: data/oisst_bmc_1982_2024.npz with sst[time, lat, lon] (deg C, NaN over land),
        time (datetime64[D]), lat, lon.
"""
from __future__ import annotations

import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

BASE = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.{year}.nc"
LAT0, LAT1 = 168, 231
LON0, LON1 = 1184, 1247
OUT = Path(__file__).resolve().parents[1] / "data"
CACHE = OUT / "yearly"


def _fetch(url: str, tries: int = 5) -> str:
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return r.read().decode()
        except Exception as e:  # network hiccup: back off and retry
            print(f"  retry {k + 1}: {e}", flush=True)
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"failed: {url}")


def _parse_ascii_grid(txt: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse the OPeNDAP ASCII response of sst[t][lat][lon] (Grid) -> (sst, time)."""
    body = txt.split("---------------------------------------------", 1)[1]
    m = re.search(r"sst\.sst\[(\d+)\]\[(\d+)\]\[(\d+)\]", body)
    nt, ny, nx = map(int, m.groups())
    rows = re.findall(r"^\[(\d+)\]\[(\d+)\], (.*)$", body, flags=re.M)
    arr = np.empty((nt, ny, nx), dtype=np.float32)
    for ti, yi, vals in rows[: nt * ny]:
        arr[int(ti), int(yi)] = np.array(vals.split(","), dtype=np.float32)
    tm = re.search(r"sst\.time\[\d+\]\n(.*?)\n", body)
    t = np.array(tm.group(1).split(","), dtype=np.float64)
    return arr, t


def download_year(year: int) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"sst_{year}.npz"
    if f.exists():
        return f
    dds = _fetch(BASE.format(year=year) + ".dds")
    nt = int(re.search(r"time = (\d+)", dds).group(1))
    chunks, times = [], []
    for a in range(0, nt, 32):  # ~monthly requests; a whole year trips a 502 on the proxy
        b = min(a + 31, nt - 1)
        q = f"sst[{a}:1:{b}][{LAT0}:1:{LAT1}][{LON0}:1:{LON1}]"
        s, t = _parse_ascii_grid(_fetch(BASE.format(year=year) + ".ascii?" + q))
        chunks.append(s)
        times.append(t)
    sst, t = np.concatenate(chunks), np.concatenate(times)
    # time units: days since 1800-01-01
    days = (np.datetime64("1800-01-01") + t.astype("timedelta64[D]")).astype("datetime64[D]")
    sst[(sst < -5) | (sst > 45)] = np.nan  # fill value (-9.96921e36) -> NaN (land/ice)
    np.savez_compressed(f, sst=sst, time=days)
    return f


def main(y0: int = 1982, y1: int = 2024) -> None:
    from concurrent.futures import ThreadPoolExecutor

    def job(y: int) -> None:
        t0 = time.time()
        download_year(y)
        print(f"{y} ok ({time.time() - t0:.0f}s)", flush=True)

    with ThreadPoolExecutor(4) as ex:  # modest concurrency to stay polite with the server
        list(ex.map(job, range(y0, y1 + 1)))
    parts = [np.load(CACHE / f"sst_{y}.npz") for y in range(y0, y1 + 1)]
    sst = np.concatenate([p["sst"] for p in parts])
    tt = np.concatenate([p["time"] for p in parts])
    lat = -47.875 + 0.25 * np.arange(64)
    lon = 296.125 + 0.25 * np.arange(64) - 360.0
    np.savez_compressed(OUT / f"oisst_bmc_{y0}_{y1}.npz", sst=sst, time=tt, lat=lat, lon=lon)
    print("saved", sst.shape, tt[0], tt[-1])


if __name__ == "__main__":
    main(*map(int, sys.argv[1:3])) if len(sys.argv) > 2 else main()
