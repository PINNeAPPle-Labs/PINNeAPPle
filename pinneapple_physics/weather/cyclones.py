"""A tropical cyclone end to end on real data: the official best track, the vortex at the centre of a gridded
pressure field, and the three numbers a hurricane showcase has to quote — track error in kilometres, intensity error
in knots, and the lead at which the model stops being able to follow the storm.

Two kinds of data, both public and both cited where they are used:

* **Best track** — the NHC North Atlantic Hurricane Database (HURDAT2), the official 6-hourly position, maximum wind
  and minimum central pressure of every storm, reanalysed after the season (Landsea & Franklin, 2013). US Government
  work, public domain, downloaded from the NHC at run time and never stored in the repository.
* **Gridded fields** — any ``(n, lat, lon)`` array of sea-level pressure in hPa: an analysis (ERA5 through the ARCO
  bucket), a published forecast (IFS HRES, Pangu-Weather through WeatherBench2), or a model of this library.

The centre of the storm in a gridded field is the deepest pressure minimum within a search radius of the previous
position — the continuity tracker used in operational verification — refined to the sub-grid minimum by a parabola
through the three cells along each axis. A minimum shallower than ``deepest_hpa`` is reported as *not found*: on a
grid too coarse for the core, that is the honest outcome, and it is counted rather than hidden.

Intensity is compared three ways, because a grid minimum and a best track do not measure the same thing: the central
pressure directly (hPa), and the maximum wind through the empirical wind-pressure relation of Atkinson & Holliday
(1977), ``W = 6.7 (1010 - p)^0.644`` knots, which turns a grid minimum into an *equivalent* wind. The relation is
named wherever it is used; it is not a model wind.

The scale is the story. ERA5 and the 0.25-degree forecast archives place the centre of a hurricane to a few tens of
kilometres and are still 20-40 hPa too shallow; a 5.625-degree grid (WeatherBench2's 64 x 32) has cells wider than a
hurricane and cannot hold one at all. The same code measures both, which is why a showcase can say *where* the chain
has to hand over to a finer model instead of asserting it.
"""
from __future__ import annotations

import hashlib
import textwrap
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EARTH_RADIUS_KM = 6371.0
HURDAT2_URL = ("https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-092326.txt"
               )                       # the file the NHC published on 2026-09-23; pinned so a run can be reproduced
HURDAT2_CITATION = ("NOAA/NHC North Atlantic Hurricane Database (HURDAT2), Landsea & Franklin (2013), "
                    "https://www.nhc.noaa.gov/data/hurdat/ — US Government work, public domain")
AH_REFERENCE = "Atkinson & Holliday (1977): W = 6.7 (1010 - p)^0.644 kt"
STATUS_NAMES = {"TD": "tropical depression", "TS": "tropical storm", "HU": "hurricane", "EX": "extratropical",
                "LO": "low", "DB": "disturbance", "WV": "tropical wave", "SD": "subtropical depression",
                "SS": "subtropical storm", "PT": "post-tropical"}


# ----------------------------------------------------------------------------------------------- best track

@dataclass
class BestTrack:
    """One storm of HURDAT2: its 6-hourly (plus landfall) records, in the units the file publishes.

    Attributes
    ----------
    storm_id : str
        ATCF identifier: basin letter, storm number, year (``AL092022`` is the Atlantic's ninth storm of 2022).
    name : str
    times : np.ndarray
        ``datetime64[h]`` of every record.
    lat, lon : np.ndarray [degrees]
        Centre position; ``lon`` in [-180, 180], as the file publishes it.
    wind_kt : np.ndarray [kt]
        Maximum sustained wind (1-minute mean) of each record.
    pressure_hpa : np.ndarray [hPa]
        Minimum central pressure (NaN where the file has none).
    status : np.ndarray [str]
        Two-letter code of each record: TD, TS, HU (hurricane), EX, LO, ...
    record : np.ndarray [str]
        Record identifier: ``L`` landfall, ``P`` minimum pressure, empty for a routine synoptic record.
    source : dict
        Where the records came from: the file, its SHA-256 and the citation, filled in by
        :func:`load_best_track` so that a figure or a JSON can quote it.
    """

    storm_id: str
    name: str
    times: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    wind_kt: np.ndarray
    pressure_hpa: np.ndarray
    status: np.ndarray
    record: np.ndarray
    source: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.times)

    @property
    def basin(self) -> str:
        return {"AL": "North Atlantic", "EP": "eastern North Pacific", "CP": "central North Pacific"}.get(
            self.storm_id[:2], self.storm_id[:2])

    def index(self, time) -> int | None:
        """Index of an exact record, or None."""
        hit = np.nonzero(self.times == np.datetime64(time, "h"))[0]
        return int(hit[0]) if len(hit) else None

    def interpolate(self, times) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(lat, lon, wind_kt, pressure_hpa) linearly interpolated to ``times``; NaN outside the storm's life.

        Verification needs the official position at the model's valid times, which are not the records' own times
        (HURDAT2 adds landfall and pressure-minimum records at odd hours). A straight line between two records six
        hours apart is within a few kilometres of the curve; longitude is unwrapped first so a storm crossing 180
        degrees does not jump.
        """
        q = np.asarray(times, dtype="datetime64[h]").astype(np.int64).astype(float)
        x = self.times.astype("datetime64[h]").astype(np.int64).astype(float)
        inside = (q >= x[0]) & (q <= x[-1])
        lon = np.rad2deg(np.unwrap(np.deg2rad(self.lon)))
        out = [np.where(inside, np.interp(q, x, y), np.nan) for y in (self.lat, lon, self.wind_kt, self.pressure_hpa)]
        return out[0], out[1], out[2], out[3]


def parse_hurdat2(text: str) -> dict[str, BestTrack]:
    """Every storm in a HURDAT2 file, keyed by its ATCF identifier.

    The format is two kinds of line: a header (``AL092022, IAN, 40,`` — identifier, name, number of records) and the
    records it announces (``20220928, 1200, , HU, 26.0N, 82.7W, 140, 937, ...``); missing values are negative.

    Parameters
    ----------
    text : str
        The whole file.

    Returns
    -------
    storms : dict[str, BestTrack]
        One entry per header, in file order.
    """
    lines = text.splitlines()
    out: dict[str, BestTrack] = {}
    i = 0
    while i < len(lines):
        head = lines[i].strip()
        i += 1
        if not head:
            continue
        parts = [p.strip() for p in head.split(",")]
        if len(parts) < 3 or not parts[2].strip().isdigit():
            raise ValueError(f"not a HURDAT2 header: {head!r}")
        storm_id, name, count = parts[0], parts[1], int(parts[2])
        rows = []
        for line in lines[i:i + count]:
            p = [c.strip() for c in line.split(",")]
            date, clock = p[0], p[1].zfill(4)
            lat = float(p[4][:-1]) * (1.0 if p[4][-1] == "N" else -1.0)
            lon = float(p[5][:-1]) * (1.0 if p[5][-1] == "E" else -1.0)
            rows.append((np.datetime64(f"{date[:4]}-{date[4:6]}-{date[6:]}T{clock[:2]}:{clock[2:]}", "h"),
                         lat, lon, float(p[6]), float(p[7]), p[3], p[2]))
        i += count
        t, la, lo, w, pr, st, rec = zip(*rows, strict=True)
        wind = np.array(w, float)
        pres = np.array(pr, float)
        wind[wind < 0] = np.nan                     # the file's missing flag
        pres[pres < 0] = np.nan
        out[storm_id] = BestTrack(storm_id, name, np.array(t), np.array(la, float), np.array(lo, float), wind, pres,
                                  np.array(st), np.array(rec))
    return out


def load_best_track(storm_id: str, source: str | Path = HURDAT2_URL, cache: str | Path | None = None,
                    log=print) -> BestTrack:
    """One storm of HURDAT2: from the NHC (downloaded once, kept in ``cache``) or from a file already downloaded.

    Parameters
    ----------
    storm_id : str
        ATCF identifier, e.g. ``AL092022`` for Hurricane Ian 2022.
    source : str or Path
        The HURDAT2 URL, or the path of a downloaded file.
    cache : str or Path, optional
        Directory for the download (default ``~/.cache/pinneapple/hurricane``), so a second run is offline.

    Returns
    -------
    track : BestTrack
        With ``source`` = {url, file, sha256, citation} for whatever quotes it.
    """
    src = str(source)
    if src.startswith(("http://", "https://")):
        path = Path(cache or Path.home() / ".cache" / "pinneapple" / "hurricane") / src.rsplit("/", 1)[-1]
        if path.exists():
            raw = path.read_bytes()
        else:
            log(f"HURDAT2 {path.name}")
            with urllib.request.urlopen(src, timeout=120) as r:            # noqa: S310 - the NHC's own URL
                raw = r.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        url = src
    else:
        path, raw, url = Path(src), Path(src).read_bytes(), ""
    storms = parse_hurdat2(raw.decode("utf-8", errors="replace"))
    if storm_id not in storms:
        raise KeyError(f"{storm_id} not in {path.name}: it has {len(storms)} storms, {min(storms)} to {max(storms)}")
    track = storms[storm_id]
    track.source = {"url": url, "file": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
                    "citation": HURDAT2_CITATION}
    return track


# ---------------------------------------------------------------------------------------- geometry, intensity

def great_circle_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance on a sphere of radius 6371 km (haversine), broadcasting over the four arguments."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = p2 - p1
    dlam = np.deg2rad(np.asarray(lon2, float) - np.asarray(lon1, float))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def atkinson_holliday_wind(pressure_hpa, environment_hpa: float = 1010.0) -> np.ndarray:
    """Maximum wind equivalent to a central pressure, in knots: ``6.7 (environment - p)^0.644``.

    Atkinson & Holliday (1977) fitted this to observed hurricane central pressures and maximum winds; here it turns a
    *grid* pressure minimum into an intensity that can be compared with a best track, which is why an intensity error
    is quoted both in hPa and in kt. Zero where the pressure is not below the environment.
    """
    d = np.maximum(environment_hpa - np.asarray(pressure_hpa, float), 0.0)
    return 6.7 * d ** 0.644


# ---------------------------------------------------------------------------------------------- vortex tracking

@dataclass(frozen=True)
class Centre:
    """A pressure minimum found in a grid: position (degrees) and depth (hPa)."""
    lat: float
    lon: float
    pressure_hpa: float


def vortex_centre(pressure_hpa, lat, lon, guess, radius_km: float = 400.0,
                  deepest_hpa: float | None = None) -> Centre | None:
    """Deepest sea-level-pressure minimum within ``radius_km`` of ``guess``, refined to the sub-grid minimum.

    Parameters
    ----------
    pressure_hpa : np.ndarray (lat, lon) [hPa]
        The field. NaNs are ignored (a mask, a region not fetched).
    lat, lon : np.ndarray [degrees]
        Cell centres; ``lat`` may ascend or descend, ``lon`` periodic with a uniform step.
    guess : (float, float) [degrees]
        Previous centre, or an independent first guess such as the analysed position.
    radius_km : float [km]
        Search radius around ``guess``. It decides what the tracker may follow: too small and a 5-degree grid loses
        the storm between cells, too large and it jumps to the next low.
    deepest_hpa : float, optional
        A minimum shallower than this is not a storm, and None is returned.

    Returns
    -------
    centre : Centre or None
        None when no cell inside the radius is valid, or the minimum is shallower than ``deepest_hpa``.

    Notes
    -----
    The sub-grid position is the vertex of a parabola through the three cells along each axis, clipped to half a
    cell: the minimum of a smooth field is resolved better than its grid, and on a coarse grid that is most of the
    position.
    """
    la, lo, p = np.asarray(lat, float), np.asarray(lon, float), np.asarray(pressure_hpa, float)
    if p.shape != (len(la), len(lo)):
        raise ValueError(f"pressure {p.shape} does not match lat {len(la)} x lon {len(lo)}")
    d = great_circle_km(guess[0], guess[1], la[:, None], lo[None, :])
    inside = np.isfinite(p) & (d <= radius_km)
    if not inside.any():
        return None
    f = np.where(inside, p, np.inf)
    iy, ix = np.unravel_index(np.argmin(f), f.shape)
    depth = float(f[iy, ix])
    if deepest_hpa is not None and depth > deepest_hpa:
        return None

    def offset(a, b, c) -> float:
        if not (np.isfinite(a) and np.isfinite(c)):
            return 0.0
        den = a - 2.0 * b + c
        return 0.0 if den == 0 else float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))

    dy = offset(f[iy - 1, ix], depth, f[iy + 1, ix]) if 0 < iy < len(la) - 1 else 0.0
    dx = offset(f[iy, ix - 1], depth, f[iy, ix + 1]) if 0 < ix < len(lo) - 1 else 0.0
    return Centre(float(la[iy] + dy * (la[1] - la[0])), float(np.mod(lo[ix] + dx * (lo[1] - lo[0]), 360.0)), depth)


@dataclass
class StormTrack:
    """A tracked vortex: one position per valid time, NaN where the storm was not found.

    Attributes
    ----------
    times : np.ndarray
        Valid time (or lead time) of each field.
    lat, lon : np.ndarray [degrees]
        ``lon`` in [0, 360).
    pressure_hpa : np.ndarray [hPa]
    found : np.ndarray [bool]
        False where the search found no minimum deeper than the threshold: the grid lost the storm.
    """

    times: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    pressure_hpa: np.ndarray
    found: np.ndarray

    def __len__(self) -> int:
        return len(self.times)


def track_storm(pressure_hpa, lat, lon, times, start, radius_km: float = 400.0,
                deepest_hpa: float | None = 1008.0) -> StormTrack:
    """Follow the vortex from a known position through a series of pressure fields.

    Parameters
    ----------
    pressure_hpa : np.ndarray (n, lat, lon) [hPa]
        The fields, in time order.
    lat, lon : np.ndarray [degrees]
        Cell centres of the fields.
    times : np.ndarray (n,)
        Valid time of each field, carried into the result.
    start : (float, float) [degrees]
        Centre at the first field: for a forecast the analysed position it started from, for an analysis the first
        best-track position.
    radius_km : float [km]
        Search radius around the previous position (see :func:`vortex_centre`).
    deepest_hpa : float, optional
        A minimum shallower than this is not the storm; the previous position stays the guess for the next field.

    Returns
    -------
    track : StormTrack
        Positions in [0, 360) longitude, ``found`` marking the times the storm was there to be found.
    """
    p = np.asarray(pressure_hpa, float)
    n = len(p)
    out_lat, out_lon, out_p = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    found = np.zeros(n, bool)
    guess = (float(start[0]), float(np.mod(start[1], 360.0)))
    for k in range(n):
        c = vortex_centre(p[k], lat, lon, guess, radius_km, deepest_hpa)
        if c is None:
            continue
        out_lat[k], out_lon[k], out_p[k] = c.lat, c.lon, c.pressure_hpa
        found[k] = True
        guess = (c.lat, c.lon)
    return StormTrack(np.asarray(times), out_lat, out_lon, out_p, found)


# ------------------------------------------------------------------------------------------------- verification

@dataclass
class StormErrors:
    """One forecast verified against the best track: at every valid time, how far and how wrong it was.

    Attributes
    ----------
    name : str
        What was tracked (``"ERA5 0.25 deg"``, ``"IFS HRES"``, ...).
    init : np.datetime64
        The time the leads are counted from: a forecast's initialization, or the first valid time of an analysis.
    times : np.ndarray
    leads_h : np.ndarray [h]
        Hours from ``init``.
    track_km : np.ndarray [km]
        Great-circle distance between the tracked centre and the official one; NaN where the storm was not found, or
        where the best track has no position at that time (a lead past the end of the storm's record).
    pressure_hpa : np.ndarray [hPa]
        Tracked minus official central pressure: positive means the model's storm is shallower than the real one.
    wind_kt : np.ndarray [kt]
        Equivalent wind of the tracked pressure minus the official maximum wind, through
        :func:`atkinson_holliday_wind`. Both sides use the same relation, so the sign is a comparison of intensities
        and not of two different wind definitions.
    found : np.ndarray [bool]
        The times there is something to score: the vortex was tracked **and** the best track has a position. See
        :func:`verify_storm`.
    lat, lon : np.ndarray [degrees]
        Where the storm was tracked to, so the path can be drawn on the globe beside the official one.
    """

    name: str
    init: np.datetime64
    times: np.ndarray
    leads_h: np.ndarray
    track_km: np.ndarray
    pressure_hpa: np.ndarray
    wind_kt: np.ndarray
    found: np.ndarray
    lat: np.ndarray | None = None
    lon: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.times)

    def path(self) -> tuple[np.ndarray, np.ndarray]:
        """(lat, lon) of the tracked centres, NaN where the storm was not found."""
        if self.lat is None or self.lon is None:
            return np.full(len(self.times), np.nan), np.full(len(self.times), np.nan)
        return np.asarray(self.lat, float), np.asarray(self.lon, float)

    @property
    def tracked(self) -> int:
        """How many valid times held a vortex."""
        return int(np.asarray(self.found, bool).sum())


def verify_storm(track: StormTrack, best: BestTrack, name: str = "", environment_hpa: float = 1010.0,
                 init=None) -> StormErrors:
    """Errors of a tracked vortex against the best track at the same valid times.

    Parameters
    ----------
    track : StormTrack
        From :func:`track_storm`.
    best : BestTrack
        The official record, interpolated to the tracked times (:meth:`BestTrack.interpolate`).
    name : str
        Label carried into the result and the figures.
    environment_hpa : float [hPa]
        Environment pressure of the wind relation.
    init : np.datetime64, optional
        The time the leads are counted from: a forecast's own initialization, which is not its first valid time
        (a 6-hourly archive starts at T+6). Default: the first valid time of ``track``, which is what an analysis
        can mean by lead 0. Leaving it out on a forecast shifts every lead by its first step.

    Returns
    -------
    errors : StormErrors
        ``leads_h`` counted from ``init``. ``found`` is the times the vortex was tracked *and* the best track has a
        position to compare it with: a lead past the end of the storm's record has nothing to score, and counting it
        as a hit would make the skill median NaN and the "lost" horizon never arrive.
    """
    t = np.asarray(track.times, "datetime64[h]")
    t0 = t[0] if init is None else np.datetime64(init, "h")
    ola, olo, owind, opres = best.interpolate(t)
    obs_wind = np.where(np.isfinite(opres), atkinson_holliday_wind(opres, environment_hpa), owind)
    dist = great_circle_km(ola, np.mod(olo, 360.0), track.lat, track.lon)
    keep = np.asarray(track.found, bool) & np.isfinite(dist)
    return StormErrors(name or best.storm_id, t0, t, (t - t0).astype("timedelta64[h]").astype(float),
                       np.where(keep, dist, np.nan),
                       np.where(keep, track.pressure_hpa - opres, np.nan),
                       np.where(keep, atkinson_holliday_wind(track.pressure_hpa, environment_hpa) - obs_wind, np.nan),
                       keep, np.asarray(track.lat, float), np.asarray(track.lon, float))


def skill_by_lead(errors: list[StormErrors]) -> dict:
    """Median track, pressure and wind error by lead over several forecasts, and how often the storm was held.

    Parameters
    ----------
    errors : list[StormErrors]
        One per initialisation.

    Returns
    -------
    skill : dict
        ``leads_h``; ``track_km``, ``pressure_hpa``, ``wind_kt`` (medians over the forecasts with something to score);
        ``found_fraction`` (share of the forecasts that had a vortex and an official position at that lead, 0..1);
        ``n`` (how many did).
    """
    if not errors:
        raise ValueError("no forecasts to score")
    leads = np.asarray(errors[0].leads_h, float)
    out = {"leads_h": leads, "track_km": np.full(len(leads), np.nan), "pressure_hpa": np.full(len(leads), np.nan),
           "wind_kt": np.full(len(leads), np.nan), "found_fraction": np.zeros(len(leads)), "n": np.zeros(len(leads), int)}
    for k, lead in enumerate(leads):
        rows = [(e, int(np.argmin(np.abs(e.leads_h - lead)))) for e in errors]
        rows = [(e, j) for e, j in rows if abs(e.leads_h[j] - lead) <= 1e-6]
        if not rows:
            continue
        out["found_fraction"][k] = sum(bool(e.found[j]) for e, j in rows) / len(rows)
        hit = [(e, j) for e, j in rows if e.found[j]]
        out["n"][k] = len(hit)
        if hit:
            out["track_km"][k] = float(np.median([e.track_km[j] for e, j in hit]))
            out["pressure_hpa"][k] = float(np.median([e.pressure_hpa[j] for e, j in hit]))
            out["wind_kt"][k] = float(np.median([e.wind_kt[j] for e, j in hit]))
    return out


def storm_horizons(skill: dict, track_km: float = 500.0, wind_kt: float = 20.0, found_fraction: float = 0.5) -> dict:
    """Where the forecast stops being usable, on three measures the caller states.

    * ``track_hours``: first lead at which the median track error reaches ``track_km``;
    * ``intensity_hours``: first lead at which the median equivalent-wind error reaches ``wind_kt``;
    * ``lost_hours``: first lead at which fewer than ``found_fraction`` of the forecasts still have a storm to score —
      either the grid lost the vortex, or the official record has ended — which is not a track error but the absence
      of two positions to compare.

    500 km is the width of one 5.625-degree cell, 20 kt is about the spread of official 48-hour intensity forecasts,
    and half the forecasts is a coin toss: the thresholds are the caller's, and they are quoted with the number.

    Returns
    -------
    horizons : dict
        Hours (inf where the measure never crosses) and the ``thresholds`` as given.
    """
    from .evaluate import first_crossing

    leads = np.asarray(skill["leads_h"], float)
    lost = leads[np.asarray(skill["found_fraction"], float) < found_fraction]
    return {"track_hours": first_crossing(leads, skill["track_km"], track_km, below=False),
            "intensity_hours": first_crossing(leads, np.abs(skill["wind_kt"]), wind_kt, below=False),
            "lost_hours": float(lost[0]) if len(lost) else float("inf"),
            "thresholds": {"track_km": track_km, "wind_kt": wind_kt, "found_fraction": found_fraction}}


# --------------------------------------------------------------------------------------------------- the scene

def _project(la, lo, centre, size: int):
    """Orthographic projection to the pixel coordinates of the globe drawn in a +/-1.12 window; NaN when hidden."""
    la0, lo0 = np.deg2rad(centre[0]), np.deg2rad(centre[1])
    la_r, lo_r = np.deg2rad(np.asarray(la, float)), np.deg2rad(np.asarray(lo, float))
    cosc = np.sin(la0) * np.sin(la_r) + np.cos(la0) * np.cos(la_r) * np.cos(lo_r - lo0)
    x = np.cos(la_r) * np.sin(lo_r - lo0)
    y = np.cos(la0) * np.sin(la_r) - np.sin(la0) * np.cos(la_r) * np.cos(lo_r - lo0)
    px = (x / 1.12 + 1.0) / 2.0 * (size - 1)
    py = (1.0 - y / 1.12) / 2.0 * (size - 1)
    return np.where(cosc > 0.0, px, np.nan), np.where(cosc > 0.0, py, np.nan)


def track_globe(path, best: BestTrack, tracks: dict, *, pressure_hpa=None, lat=None, lon=None, times=None,
                centre=None, vmin: float = 985.0, vmax: float = 1015.0, size: int = 460, title: str = "",
                note: str = "", dpi: int = 110):
    """The storm on an Earth-2 style globe: the pressure field, the official track, and the tracks a model produced.

    Parameters
    ----------
    path : str or Path
        File written; a ``.gif`` suffix with a ``(n, lat, lon)`` field writes the animation of the storm crossing
        the globe, anything else writes a single frame (the first time).
    best : BestTrack
        The official track: white, a marker every 24 h, a star at the first landfall.
    tracks : dict
        ``{name: StormTrack or (lat, lon)}``, drawn in that order in the palette colours.
    pressure_hpa : np.ndarray, optional
        (lat, lon) or (n, lat, lon) sea-level pressure in hPa drawn on the globe; without it the globe is dark and
        only the tracks and the coastlines show.
    lat, lon : np.ndarray, optional
        Coordinates of ``pressure_hpa``.
    times : np.ndarray, optional
        Valid times of the frames of an animation, printed in the corner.
    centre : (float, float), optional
        Globe centre (lat, lon); default: the position of the strongest record of the best track.
    vmin, vmax : float [hPa]
        Colour limits of the field.
    title, note : str
        Headline, and the honest note that goes under it.
    size : int
        Globe diameter in pixels.
    dpi : int
        Output resolution.

    Returns
    -------
    path : Path
        The file written.

    Notes
    -----
    The globe, the lighting, the graticule and the coastlines are :func:`pinneapple_physics.weather.viz._globe_rgb`
    with the same window, so the tracks land on the coastlines that are drawn into that image.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    from .viz import BG, DIM, FG, _globe_rgb, coastlines

    if centre is None:
        j = int(np.nanargmin(np.asarray(best.pressure_hpa, float)))
        centre = (float(best.lat[j]), float(best.lon[j]))
    shore = coastlines()
    cmap = plt.get_cmap("cividis").with_extremes(bad=BG)
    norm = Normalize(vmin, vmax)
    series = None if pressure_hpa is None else np.asarray(pressure_hpa, float)
    frames = 1 if series is None or series.ndim == 2 else len(series)

    fig = plt.figure(figsize=(10.6, 6.4), facecolor=BG)
    ax = fig.add_axes([0.02, 0.10, 0.60, 0.80])                 # the floor is left for the colour bar and its label
    ax.set_facecolor(BG)
    ax.axis("off")
    fig.text(0.02, 0.955, title, color=FG, fontsize=14, weight="bold")
    fig.text(0.02, 0.925, "\n".join(textwrap.wrap(note, 108)) if note else "", color=DIM, fontsize=9, va="top",
             linespacing=1.45)                                  # wrapped: a note that runs off the page is not a note
    image = ax.imshow(np.zeros((size, size, 4)))
    ax.set_xlim(-0.02 * size, size)
    ax.set_ylim(size, -0.02 * size)

    if series is None:                                     # dark globe with the coastlines projected straight in
        for ln in shore:
            px, py = _project(ln[:, 1], ln[:, 0], centre, size)
            ax.plot(px, py, color="#3a4757", lw=0.6)
    obs_px, obs_py = _project(best.lat, np.mod(best.lon, 360.0), centre, size)
    palette = ["#5fb3ff", "#ff9f43", "#c792ea", "#3ddc97", "#ff5d5d", "#ffd166", "#9aa5b1"]
    for k, (name, tr) in enumerate(tracks.items()):
        if hasattr(tr, "lat"):                                          # a StormTrack (or anything with .lat/.lon)
            tla, tlo = np.asarray(tr.lat, float), np.asarray(tr.lon, float)
        else:                                                           # a bare (lat, lon) pair
            tla, tlo = np.asarray(tr[0], float), np.asarray(tr[1], float)
        ok = np.isfinite(tla)
        px, py = _project(tla[ok], np.mod(tlo[ok], 360.0), centre, size)
        ax.plot(px, py, color=palette[k % len(palette)], lw=1.8, label=name, solid_capstyle="round")
        ax.plot(px[::4], py[::4], "o", color=palette[k % len(palette)], ms=3.0, mfc=BG, mew=1.0)
    ax.plot(obs_px, obs_py, color="#ffffff", lw=2.6, label=f"HURDAT2 {best.name} (official)")
    for j in range(0, len(best), 4):                                        # a marker every 24 h
        ax.plot(obs_px[j], obs_py[j], "o", color=BG, ms=5.0, mec="#ffffff", mew=1.2)
    land = np.nonzero(np.asarray(best.record) == "L")[0]
    if len(land):
        j = int(land[0])
        ax.plot(obs_px[j], obs_py[j], "*", color="#ff5d5d", ms=15)
        ax.annotate("landfall", (obs_px[j], obs_py[j]), xytext=(obs_px[j] + 24, obs_py[j] - 30), color="#ff5d5d",
                    fontsize=9, arrowprops={"arrowstyle": "-", "color": "#ff5d5d", "lw": 0.9})
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=FG)
    if series is not None:
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="horizontal",
                          fraction=0.034, pad=0.015, aspect=40)
        cb.set_label("sea-level pressure (hPa)", color=DIM, fontsize=9)
        cb.ax.tick_params(colors=DIM, labelsize=8)
        cb.outline.set_edgecolor(DIM)
    stamp = fig.text(0.62, 0.855, "", color=FG, fontsize=11, weight="bold")

    def draw(k: int):
        if series is not None:
            f = series if series.ndim == 2 else series[k]
            image.set_data(_globe_rgb(f, np.asarray(lat, float), np.asarray(lon, float), centre, size, cmap, norm,
                                      shore))
        if times is not None:
            t = np.asarray(times)[k if (series is not None and series.ndim == 3) else 0]
            stamp.set_text(np.datetime_as_string(np.datetime64(t, "h"), unit="h").replace("T", " ") + " UTC")

    path = Path(path)
    if path.suffix.lower() == ".gif" and frames > 1:
        from matplotlib.animation import FuncAnimation, PillowWriter

        anim = FuncAnimation(fig, lambda k: (draw(k), image, stamp)[1:], frames=frames, blit=False)
        anim.save(path, writer=PillowWriter(fps=5), dpi=dpi, savefig_kwargs={"facecolor": BG})
    else:
        draw(0)
        fig.savefig(path, dpi=dpi, facecolor=BG)
    plt.close(fig)
    return path


def track_skill_figure(path, skill: dict, horizons: dict, *, title: str = "", dpi: int = 110):
    """Two panels of the same forecast: track error by lead (km) and intensity error by lead (kt).

    The band under each curve is the share of the forecasts that still held a vortex, drawn from the bottom: where it
    drops, the model has not merely drifted, it has lost the storm — and the error curve stops there because there is
    nothing left to measure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .viz import BG, DIM, FADING, FG, GOOD, POOR

    leads = np.asarray(skill["leads_h"], float) / 24.0
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.0), facecolor=BG, sharex=True)
    spec = (("track_km", "track error (km)", horizons["thresholds"]["track_km"], horizons["track_hours"],
             "median track error is 500 km at"),
            ("wind_kt", "intensity error (kt, Atkinson-Holliday)", horizons["thresholds"]["wind_kt"],
             horizons["intensity_hours"], "median intensity error reaches 20 kt at"))
    for ax, (key, label, thr, hour, sentence) in zip(axes, spec, strict=True):
        ax.set_facecolor(BG)
        y = np.asarray(skill[key], float)
        ax.plot(leads, y, color=FG, lw=2.0)
        ax.axhline(thr, color=DIM, ls=":", lw=1)
        ax.axvspan(0, leads[-1], ymin=0, ymax=1, color=GOOD, alpha=0.04)
        ax.fill_between(leads, 0, np.asarray(skill["found_fraction"], float) * np.nanmax(y) * 0.25,
                        color=POOR, alpha=0.25, step=None)
        if np.isfinite(hour):
            ax.axvline(hour / 24.0, color=FADING, lw=1.2)
            ax.annotate(sentence + f" {hour / 24:.1f} d", (hour / 24.0, np.nanmax(y) * 0.85), color=FADING,
                        fontsize=8, ha="left", xytext=(4, 0), textcoords="offset points")
        ax.set_ylabel(label, color=DIM, fontsize=9)
        ax.tick_params(colors=DIM, labelsize=8)
        for s in ax.spines.values():
            s.set_color("#253040")
    axes[-1].set_xlabel("lead time (days)", color=DIM, fontsize=9)
    lost = horizons["lost_hours"]
    axes[0].set_title(title + (f" · the storm is gone from half the forecasts at {lost:.0f} h"
                               if np.isfinite(lost) else " · the storm is held at every lead"),
                      color=FG, fontsize=10, loc="left")
    axes[-1].text(0.005, -0.30, "bar at the bottom: share of forecasts that still held a vortex; none, none",
                  color=DIM, fontsize=7, transform=axes[-1].transAxes)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=dpi, facecolor=BG)
    plt.close(fig)
    return path
