"""The NOAA National Data Buoy Center files, read into arrays.

Three of NDBC's public files describe one station, and this module reads all three without a client library:

* ``<station>.txt`` -- the standard meteorological record, ten-minute values of ``WVHT`` (significant wave height, m),
  ``DPD`` (dominant -- peak -- period, s), ``APD`` (average period), ``WSPD`` (wind speed at 10 m, m/s), ``MWD``
  (mean wave direction) and the atmospheric fields;
* ``<station>.data_spec`` -- the measured wave spectral density, every thirty minutes, on NDBC's 46-band frequency
  grid, in m^2/Hz. This is the one that makes a spectral measurement possible: every other wave number in the files
  is an integral of it;
* ``<station>.spec`` -- NDBC's own separation of the same spectrum into swell and wind sea (``SwH``/``SwP`` and
  ``WWH``/``WWP``), which is the measurement the package's own split is compared against.

Missing values are ``MM`` in the text and become ``NaN`` here. Nothing is imputed: a row with a missing wave height is
a row without one, and every consumer is expected to drop it.

``to_npz`` writes the slice an example ships with -- the measured spectra, the hourly sea state and the NDBC split --
so the example runs offline and reproduces what was fetched, with the station, the window and the URLs recorded in the
file. ``load`` reads it back.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["BASE", "STATIONS", "BuoyRecord", "read_standard_met", "read_spectral_density", "read_spectral_summary",
           "load", "to_npz", "hourly"]

BASE = "https://www.ndbc.noaa.gov/data/realtime2"

#: The stations the example uses, with what each one is (NDBC station pages, 2026).
STATIONS = {
    "41010": "Canaveral, 20 NM east of Cape Canaveral, Florida, 3 m discus, water depth 30 m",
    "46002": "Oregon offshore, 275 NM west of Coos Bay, 3 m discus, water depth 3300 m",
}

_PAIRS = re.compile(r"([-\d.]+|MM|NaN)\s*\(([\d.]+)\)")


def _number(text: str) -> float:
    """NDBC's missing-value marker (``MM``) becomes ``NaN``."""
    if text in ("MM", "NaN"):
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _times(rows: list[list[str]]) -> np.ndarray:
    stamps = [f"{r[0]}-{int(r[1]):02d}-{int(r[2]):02d}T{int(r[3]):02d}:{int(r[4]):02d}:00" for r in rows]
    return np.array(stamps, dtype="datetime64[s]")


def _ascending(time: np.ndarray, **fields: np.ndarray) -> dict[str, np.ndarray]:
    """The files are written newest first; everything here is ascending in time, which a lead time needs."""
    order = np.argsort(time, kind="stable")
    return {"time": time[order], **{k: v[order] for k, v in fields.items()}}



def _rows(text: str) -> list[list[str]]:
    return [line.split() for line in text.splitlines() if line.strip() and not line.startswith("#")]


@dataclass(frozen=True)
class BuoyRecord:
    """One station, as the files describe it. Times are UTC; the spectral rows are a subset of the met rows' times."""

    station: str
    time: np.ndarray                     # (n,) datetime64[s], standard meteorological record
    hs: np.ndarray                       # (n,) WVHT, m
    tp: np.ndarray                       # (n,) DPD, s (NDBC's dominant period)
    apd: np.ndarray                      # (n,) s
    wspd: np.ndarray                     # (n,) m/s at 10 m
    mwd: np.ndarray                      # (n,) deg
    spec_time: np.ndarray                # (m,) datetime64[s]
    spec_freq: np.ndarray                # (k,) Hz, band edges from NDBC's grid
    spec_density: np.ndarray             # (m, k) m^2/Hz
    swell_hs: np.ndarray                 # (p,) SwH, m -- NDBC's own sea/swell split
    swell_tp: np.ndarray
    windsea_hs: np.ndarray               # (p,) WWH, m
    windsea_tp: np.ndarray
    split_time: np.ndarray               # (p,)
    source: str = ""

    def __len__(self) -> int:
        return int(self.time.size)

    def valid(self) -> np.ndarray:
        """Rows with a wave height *and* a peak period: everything the sea-state code can use."""
        return np.isfinite(self.hs) & np.isfinite(self.tp) & (self.hs > 0) & (self.tp > 0)

    def spectra(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(time, freq, density)`` for the spectral rows that carry a non-empty spectrum."""
        good = np.isfinite(self.spec_density).all(axis=1) & (np.nan_to_num(self.spec_density).sum(axis=1) > 0)
        return self.spec_time[good], self.spec_freq, self.spec_density[good]

    def wind_at(self, time: np.ndarray) -> np.ndarray:
        """Wind speed interpolated to ``time``, ``NaN`` outside the record."""
        ok = np.isfinite(self.wspd)
        return np.interp(time.astype("datetime64[s]").astype(np.int64),
                         self.time[ok].astype(np.int64), self.wspd[ok], left=np.nan, right=np.nan)


def read_standard_met(text: str) -> dict[str, np.ndarray]:
    """``<station>.txt`` -- the columns are fixed by NDBC, so they are read by position."""
    rows = _rows(text)
    return _ascending(
        _times(rows),
        WDIR=np.array([_number(r[5]) for r in rows]),
        wspd=np.array([_number(r[6]) for r in rows]),
        gst=np.array([_number(r[7]) for r in rows]),
        hs=np.array([_number(r[8]) for r in rows]),
        tp=np.array([_number(r[9]) for r in rows]),
        apd=np.array([_number(r[10]) for r in rows]),
        mwd=np.array([_number(r[11]) for r in rows]),
        pres=np.array([_number(r[12]) for r in rows]),
    )


def read_spectral_density(text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``<station>.data_spec`` -- ``(time, freq, density)``, density in m^2/Hz on NDBC's band grid.

    The bands are not evenly spaced (0.005 Hz up to 0.10 Hz, 0.01 Hz to 0.35, 0.02 Hz above that), so the frequency
    axis is kept as the band centres and any integral uses the band widths.
    """
    rows = _rows(text)
    freq: np.ndarray | None = None
    density, times = [], []
    for row in rows:
        pairs = _PAIRS.findall(" ".join(row[5:]))
        if len(pairs) < 3:
            continue
        values = np.array([_number(a) for a, _ in pairs])
        bands = np.array([float(b) for _, b in pairs])
        if freq is None:
            freq = bands
        elif bands.size != freq.size:
            continue
        density.append(values)
        times.append(f"{row[0]}-{int(row[1]):02d}-{int(row[2]):02d}T{int(row[3]):02d}:{int(row[4]):02d}:00")
    time = np.array(times, dtype="datetime64[s]")
    order = np.argsort(time, kind="stable")
    return time[order], np.asarray(freq, dtype=float), np.asarray(density, dtype=float)[order]


def read_spectral_summary(text: str) -> dict[str, np.ndarray]:
    """``<station>.spec`` -- NDBC's split of the spectrum into swell (``SwH``/``SwP``) and wind sea (``WWH``/``WWP``)."""
    rows = _rows(text)
    return _ascending(
        _times(rows),
        hs=np.array([_number(r[5]) for r in rows]),
        swell_hs=np.array([_number(r[6]) for r in rows]),
        swell_tp=np.array([_number(r[7]) for r in rows]),
        windsea_hs=np.array([_number(r[8]) for r in rows]),
        windsea_tp=np.array([_number(r[9]) for r in rows]),
    )


def hourly(record: BuoyRecord) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The record resampled to whole UTC hours, as the mean of the ten-minute values the hour carries.

    A wave record is one measurement every half hour and the ten-minute rows in between are blank, so picking one row
    per hour would pick a blank two times out of three; the mean over the hour is the measurement, and an hour with no
    measurement stays ``NaN``. ``count`` says how many rows each hour averaged.
    """
    hours = record.time.astype("datetime64[h]")
    uniq, inverse = np.unique(hours, return_inverse=True)
    fields = {"hs": record.hs, "tp": record.tp, "apd": record.apd, "wspd": record.wspd, "mwd": record.mwd}
    out: dict[str, np.ndarray] = {}
    for name, values in fields.items():
        sums = np.zeros(uniq.size)
        counts = np.zeros(uniq.size)
        np.add.at(sums, inverse, np.nan_to_num(values, nan=0.0))
        np.add.at(counts, inverse, np.isfinite(values).astype(float))
        out[name] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    out["count"] = counts
    return uniq, out


def to_npz(record: BuoyRecord, dest: str | Path) -> Path:
    """Write the record (spectra included) as one compressed ``.npz``, with where it came from."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest, station=record.station, time=record.time.astype(np.int64), hs=record.hs, tp=record.tp,
        apd=record.apd, wspd=record.wspd, mwd=record.mwd, spec_time=record.spec_time.astype(np.int64),
        spec_freq=record.spec_freq, spec_density=record.spec_density, swell_hs=record.swell_hs,
        swell_tp=record.swell_tp, windsea_hs=record.windsea_hs, windsea_tp=record.windsea_tp,
        split_time=record.split_time.astype(np.int64), source=record.source,
    )
    return dest


def load(path: str | Path) -> BuoyRecord:
    """Read what :func:`to_npz` wrote."""
    with np.load(Path(path), allow_pickle=False) as data:
        return BuoyRecord(
            station=str(data["station"]), time=data["time"].astype("datetime64[s]"),
            hs=data["hs"].astype(float), tp=data["tp"].astype(float), apd=data["apd"].astype(float),
            wspd=data["wspd"].astype(float), mwd=data["mwd"].astype(float),
            spec_time=data["spec_time"].astype("datetime64[s]"), spec_freq=data["spec_freq"].astype(float),
            spec_density=data["spec_density"].astype(float),
            swell_hs=data["swell_hs"].astype(float), swell_tp=data["swell_tp"].astype(float),
            windsea_hs=data["windsea_hs"].astype(float), windsea_tp=data["windsea_tp"].astype(float),
            split_time=data["split_time"].astype("datetime64[s]"), source=str(data["source"]),
        )


def fetch(station: str, *, timeout: float = 30.0) -> BuoyRecord:
    """Download the three files for a station and assemble the record.

    The network is only used here: everything downstream reads the arrays. NDBC publishes these files as US
    Government work, with no account and no key.
    """
    from urllib.request import urlopen

    def get(name: str) -> str:
        with urlopen(f"{BASE}/{name}", timeout=timeout) as response:   # noqa: S310 -- a fixed https URL
            return io.TextIOWrapper(response, encoding="utf-8", errors="replace").read()

    met = read_standard_met(get(f"{station}.txt"))
    spec_time, spec_freq, spec_density = read_spectral_density(get(f"{station}.data_spec"))
    summary = read_spectral_summary(get(f"{station}.spec"))
    return BuoyRecord(
        station=station, time=met["time"], hs=met["hs"], tp=met["tp"], apd=met["apd"], wspd=met["wspd"],
        mwd=met["mwd"], spec_time=spec_time, spec_freq=spec_freq, spec_density=spec_density,
        swell_hs=summary["swell_hs"], swell_tp=summary["swell_tp"], windsea_hs=summary["windsea_hs"],
        windsea_tp=summary["windsea_tp"], split_time=summary["time"],
        source=f"{BASE}/{station}.txt, {BASE}/{station}.data_spec, {BASE}/{station}.spec",
    )
