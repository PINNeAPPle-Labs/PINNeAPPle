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
             "median track error is", False),
            ("wind_kt", "intensity error (kt, Atkinson-Holliday)", horizons["thresholds"]["wind_kt"],
             horizons["intensity_hours"], "median |intensity error| reaches", True))
    for ax, (key, label, thr, hour, sentence, signed) in zip(axes, spec, strict=True):
        ax.set_facecolor(BG)
        y = np.asarray(skill[key], float)
        ax.plot(leads, y, color=FG, lw=2.0)
        for t in ((-thr, thr) if signed else (thr,)):       # the intensity error is signed and its horizon uses |.|:
            ax.axhline(t, color=DIM, ls=":", lw=1)          # both edges of the band are where the horizon is set
        ax.axvspan(0, leads[-1], ymin=0, ymax=1, color=GOOD, alpha=0.04)
        ax.fill_between(leads, 0, np.asarray(skill["found_fraction"], float) * np.nanmax(y) * 0.25,
                        color=POOR, alpha=0.25, step=None)
        if np.isfinite(hour):
            ax.axvline(hour / 24.0, color=FADING, lw=1.2)
            unit = "kt" if signed else "km"
            ax.annotate(f"{sentence} {thr:.0f} {unit} at {hour / 24:.1f} d", (hour / 24.0, np.nanmax(y) * 0.85),
                        color=FADING, fontsize=8, ha="left", xytext=(4, 0), textcoords="offset points")
        ax.set_ylabel(label, color=DIM, fontsize=9)
        ax.tick_params(colors=DIM, labelsize=8)
        for s in ax.spines.values():
            s.set_color("#253040")
    axes[-1].set_xlabel("lead time (days)", color=DIM, fontsize=9)
    lost = horizons["lost_hours"]
    axes[0].set_title(title + (f" · the storm is gone from half the forecasts at {lost:.0f} h"
                               if np.isfinite(lost) else " · the storm is held at every lead"),
                      color=FG, fontsize=10, loc="left")
    axes[-1].text(0.005, -0.30, "shaded from the bottom: share of the forecasts (0 to 1, scaled) that still had a "
                                "storm and an official position to compare at that lead",
                  color=DIM, fontsize=7, transform=axes[-1].transAxes)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=dpi, facecolor=BG)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------------------------- the showcase

# The sentence the showcase cannot leave out, in the words the issue asks for: the stages of the chain exist because
# of the scale, not because of a preference.
CORE_NOTE = ("a tropical cyclone's core is smaller than a global model's grid: at 5.625 degrees one cell is 625 km "
             "across, wider than the storm itself — which is why the chain has finer stages to hand over to, and why "
             "the coarse grid loses the core entirely rather than merely smoothing it")


def _to_pixel(lat, lon, extent, width: int, height: int):
    """Pixel coordinates of (lat, lon) in an equirectangular image of ``extent``, the way :func:`viz.regional` draws."""
    lat_s, lat_n, lon_w, lon_e = extent
    lo = np.mod(np.asarray(lon, float) - lon_w, 360.0) + lon_w      # the wrap that keeps a track in the panel
    px = (lo - lon_w) / (lon_e - lon_w) * (width - 1)
    py = (lat_n - np.asarray(lat, float)) / (lat_n - lat_s) * (height - 1)
    return px, py


def _extent_of(best, tracks, pad: float = 2.5):
    """The map the storm crossed: the official track and every followed one, padded, in [0, 360) longitude."""
    lats, lons = [np.asarray(best.lat, float)], [np.mod(np.asarray(best.lon, float), 360.0)]
    for tr in tracks.values():
        la = np.asarray(tr.lat if hasattr(tr, "lat") else tr[0], float)
        lo = np.asarray(tr.lon if hasattr(tr, "lon") else tr[1], float)
        ok = np.isfinite(la)
        lats.append(la[ok])
        lons.append(np.mod(lo[ok], 360.0))
    la, lo = np.concatenate(lats), np.concatenate(lons)
    return (float(np.min(la) - pad), float(np.max(la) + pad), float(np.min(lo) - pad), float(np.max(lo) + pad))


def _hms(deg: float, kind: str) -> str:
    """A latitude or longitude the way a map label writes it: 20N, 82W -- not 280E."""
    if kind == "lat":
        return f"{abs(deg):.0f}{'N' if deg >= 0 else 'S'}"
    lo = ((deg + 180.0) % 360.0) - 180.0
    return f"{abs(lo):.0f}{'E' if lo >= 0 else 'W'}"


def _graticule(ax, extent, width: int, height: int, colour: str = "#18222f") -> None:
    """Meridians and parallels every 5 degrees, faint, with a label every 10, so a track reads against a place."""
    lat_s, lat_n, lon_w, lon_e = extent
    for la in np.arange(np.ceil(lat_s / 5) * 5, lat_n, 5):
        y = (lat_n - la) / (lat_n - lat_s) * (height - 1)
        ax.axhline(y, color=colour, lw=0.6, zorder=0)
    for lo in np.arange(np.ceil(lon_w / 5) * 5, lon_e, 5):
        x = (lo - lon_w) / (lon_e - lon_w) * (width - 1)
        ax.axvline(x, color=colour, lw=0.6, zorder=0)
    for la in np.arange(np.ceil(lat_s / 10) * 10, lat_n, 10):
        for lo in np.arange(np.ceil(lon_w / 10) * 10, lon_e, 10):
            ax.text((lo - lon_w) / (lon_e - lon_w) * (width - 1), (lat_n - la) / (lat_n - lat_s) * (height - 1),
                    f"{_hms(la, 'lat')} {_hms(lo, 'lon')}", color="#3d4b5c", fontsize=6.5, ha="left", va="bottom",
                    zorder=0)


def track_vs_observed(path, best: BestTrack, tracks: dict, *, captions: dict | None = None, extent=None,
                      title: str = "", note: str = "", dpi: int = 110, map_width: float = 7.2):
    """The official track against the ones the models followed, on the map the storm crossed.

    Parameters
    ----------
    path : str or Path
        File written.
    best : BestTrack
        The official 6-hourly positions: white, a marker every 24 h, a star at the first landfall.
    tracks : dict
        ``{name: StormTrack or (lat, lon)}``, drawn in that order in the palette of :func:`track_globe`.
    captions : dict, optional
        ``{name: text}`` added to the legend: the number the reader needs to judge the line, e.g. the median track
        error at 48 h.
    extent : (float, float, float, float), optional
        ``(lat_s, lat_n, lon_w, lon_e)`` of the map; default: what the official and the followed tracks span, padded.

    Returns
    -------
    path : Path
        The file written.

    Notes
    -----
    The map is drawn here rather than in a mapping library, like the rest of this module: coastlines from the Natural
    Earth 1:110m set that ships with :mod:`pinneapple_physics.weather.viz`, an equirectangular projection, and the
    same pixel arithmetic as :func:`viz.regional`, so a track and a coastline land on the same pixel. The axes are
    sized to the map's own aspect, so a degree of latitude and a degree of longitude keep their ratio and the storm's
    shape is not stretched by the layout.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .viz import BG, DIM, FG, coastlines, regional_lines

    extent = _extent_of(best, tracks) if extent is None else extent
    lat_s, lat_n, lon_w, lon_e = extent
    aspect = (lon_e - lon_w) / (lat_n - lat_s)
    # the map sets the figure: its width is the caller's, its height follows the region's own aspect, and the two
    # bands (title and note above, legend and footnote below) are sized in inches rather than in fractions
    box_w = float(map_width)
    box_h = box_w / aspect
    side, below, above = 0.35, 1.20, 1.05
    figw, figh = box_w + 2 * side, box_h + below + above
    width, height = int(100 * box_w), int(100 * box_h)
    fig = plt.figure(figsize=(figw, figh), facecolor=BG)
    ax = fig.add_axes([side / figw, below / figh, box_w / figw, box_h / figh])
    ax.set_facecolor("#070c13")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#1d2634")
    _graticule(ax, extent, width, height)
    for ln in coastlines():
        px, py = regional_lines([ln], extent, (width, height))[0]
        ax.plot(px, py, color="#3a4757", lw=0.8, zorder=1)

    palette = ["#5fb3ff", "#ff9f43", "#c792ea", "#3ddc97", "#ff5d5d", "#ffd166", "#9aa5b1"]
    obs_px, obs_py = _to_pixel(best.lat, np.mod(best.lon, 360.0), extent, width, height)
    handles, labels = [], []
    for k, (name, tr) in enumerate(tracks.items()):
        tla = np.asarray(tr.lat if hasattr(tr, "lat") else tr[0], float)
        tlo = np.asarray(tr.lon if hasattr(tr, "lon") else tr[1], float)
        ok = np.isfinite(tla)
        px, py = _to_pixel(tla[ok], np.mod(tlo[ok], 360.0), extent, width, height)
        handles.append(ax.plot(px, py, color=palette[k % len(palette)], lw=2.0, zorder=3)[0])
        labels.append(name if not captions or name not in captions else f"{name}  —  {captions[name]}")
        ax.plot(px[::4], py[::4], "o", color=palette[k % len(palette)], ms=3.2, mfc="#070c13", mew=1.1, zorder=4)
    handles.append(ax.plot(obs_px, obs_py, color="#ffffff", lw=2.8, zorder=5)[0])
    labels.append(f"HURDAT2 {best.name} (official)")
    ax.plot(obs_px[::4], obs_py[::4], "o", color="#070c13", ms=5.4, mec="#ffffff", mew=1.3, zorder=6)
    land = np.nonzero(np.asarray(best.record) == "L")[0]
    if len(land):
        j = int(land[0])
        ax.plot(obs_px[j], obs_py[j], "*", color="#ff5d5d", ms=16, zorder=7)
        ax.annotate("landfall", (obs_px[j], obs_py[j]), xytext=(obs_px[j] + 16, obs_py[j] - 12), color="#ff5d5d",
                    fontsize=8.5, ha="left", va="top", arrowprops={"arrowstyle": "-", "color": "#ff5d5d", "lw": 0.8},
                    zorder=7)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(side / figw, (below - 0.62) / figh), frameon=False,
               fontsize=8.8, labelcolor=FG, ncol=2 if len(labels) > 3 else 1, columnspacing=2.0,
               title="the lines, and the number each one earns", title_fontsize=8.5)
    fig.text(side / figw, 1 - 0.18 / figh, title, color=FG, fontsize=14, weight="bold")
    fig.text(side / figw, 1 - 0.48 / figh, "\n".join(textwrap.wrap(note, 96)) if note else "", color=DIM,
             fontsize=8.8, va="top", linespacing=1.45)
    fig.text(side / figw, 0.12 / figh, "markers every 24 h on every line; the map keeps the aspect of its own "
                                       "degrees; coastline: Natural Earth 1:110m (public domain)",
             color="#4a5869", fontsize=7.4)
    path = Path(path)
    fig.savefig(path, dpi=dpi, facecolor=BG)
    plt.close(fig)
    return path


def field_vs_grid(path, panels, *, best: BestTrack | None = None, extent=None, title: str = "", note: str = "",
                  vmin: float = 985.0, vmax: float = 1015.0, dpi: int = 110, size=(10.4, 4.9)):
    """The same instant of the same storm on two grids side by side, with the official position marked on both.

    Parameters
    ----------
    path : str or Path
        File written.
    panels : sequence of dict
        One per panel, drawn left to right: ``label`` (the panel's headline), ``field`` (a ``(lat, lon)`` array of
        sea-level pressure in hPa), ``lat``, ``lon``, and ``cell_km`` (the panel's cell size, printed under the label).
    best : BestTrack, optional
        If given, its position at ``when`` (the first record inside the panels' extent in time) is marked.
    extent : (float, float, float, float), optional
        ``(lat_s, lat_n, lon_w, lon_e)`` of the map; default: the field's own box.

    Returns
    -------
    path : Path
        The file written.

    Notes
    -----
    The point of two panels instead of one is the cell, not the colour: the caption of each panel states how wide one
    cell of that grid is, and the reader compares it with the storm drawn inside it. The coarse field is drawn
    smoothly interpolated for legibility -- the figure says so, because a smoothed 5.625-degree field looks like a
    field with a storm in it, which is exactly what the number next to it denies.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    from .viz import BG, DIM, FG, coastlines, regional, regional_lines

    panels = list(panels)
    width, height = 420, 300
    if extent is None:
        lat0 = np.asarray(panels[0]["lat"], float)
        lon0 = np.mod(np.asarray(panels[0]["lon"], float), 360.0)
        extent = (float(lat0.min()), float(lat0.max()), float(lon0.min()), float(lon0.max()))
    cmap = plt.get_cmap("cividis").with_extremes(bad=BG)
    norm = Normalize(vmin, vmax)
    figw, figh = size
    fig = plt.figure(figsize=(figw, figh), facecolor=BG)
    n = len(panels)
    left, right, gap = 0.035, 0.975, 0.035
    w = (right - left - gap * (n - 1)) / n
    axes = []
    for k, panel in enumerate(panels):
        ax = fig.add_axes([left + k * (w + gap), 0.20, w, 0.50])
        ax.set_facecolor(BG)
        ax.axis("off")
        img = regional(np.asarray(panel["field"], float), np.asarray(panel["lat"], float),
                       np.mod(np.asarray(panel["lon"], float), 360.0), extent, (width, height))
        ax.imshow(img, cmap=cmap, norm=norm, interpolation="bilinear")
        # isobars at round pressures: a closed ring is what a resolved core looks like, and its absence on the coarse
        # panel is the measurement, not an impression of the colour
        levels = [lv for lv in (990.0, 1000.0, 1010.0) if vmin < lv < vmax]
        if levels and np.isfinite(img).any():
            cs = ax.contour(np.linspace(0, width - 1, img.shape[1]), np.linspace(0, height - 1, img.shape[0]), img,
                            levels=levels, colors="#e8eef5", linewidths=0.7, alpha=0.55)
            ax.clabel(cs, fmt=lambda v: f"{v:.0f}", fontsize=6, colors="#e8eef5")
        for ln in coastlines():
            px, py = regional_lines([ln], extent, (width, height))[0]
            ax.plot(px, py, color="#8fa0b4", lw=0.7)
        if best is not None and panel.get("position") is not None:
            la, lo = panel["position"]
            px, py = _to_pixel(la, np.mod(lo, 360.0), extent, width, height)
            if 0 <= px <= width and 0 <= py <= height:
                ax.plot(px, py, "*", color="#ff5d5d", ms=15, mec="#ffffff", mew=0.7, zorder=6)
                ax.annotate("official position", (px, py), xytext=(px + 8, py + 16), color="#ffffff", fontsize=7.5,
                            arrowprops={"arrowstyle": "-", "color": "#ffffff", "lw": 0.7}, zorder=6)
        ax.set_title(f"{panel['label']}  ·  one cell = {panel['cell_km']:.0f} km", color=FG, fontsize=10.5,
                     loc="left", pad=6)
        axes.append(ax)
    cax = fig.add_axes([0.33, 0.155, 0.34, 0.014])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label("sea-level pressure (hPa)", color=DIM, fontsize=8.5)
    cb.ax.tick_params(colors=DIM, labelsize=7.5)
    cb.outline.set_edgecolor(DIM)
    fig.text(0.035, 0.948, title, color=FG, fontsize=13.5, weight="bold")
    fig.text(0.035, 0.918, "\n".join(textwrap.wrap(note, 112)) if note else "", color=DIM, fontsize=8.4, va="top",
             linespacing=1.4)
    fig.text(0.035, 0.022, "the fields are drawn smoothly interpolated for legibility: a smoothed coarse field looks "
                           "like a field that holds a storm, which is what the cell size denies; the isobars are at "
                           "the round pressures the shared scale covers",
             color="#4a5869", fontsize=7.2)
    path = Path(path)
    fig.savefig(path, dpi=dpi, facecolor=BG)
    plt.close(fig)
    return path


def confidence_by_lead(path, skills: dict, horizons: dict, *, title: str = "", note: str = "", dpi: int = 110,
                       size=(8.6, 5.4)):
    """The confidence bar along the lead, one row per model: green while the forecast is still inside both
    thresholds, amber when one of them is crossed, red when the grid has lost the storm.

    Parameters
    ----------
    path : str or Path
        File written.
    skills : dict
        ``{name: skill}`` exactly as :func:`skill_by_lead` returns it, one entry per model.
    horizons : dict
        ``{name: horizons}`` as :func:`storm_horizons` returns it: the thresholds are read from it, never assumed.

    Returns
    -------
    path : Path
        The file written.

    Notes
    -----
    The bar is the picture of the same three numbers the table quotes: the track threshold in km, the intensity
    threshold in knots, and the share of the forecasts that still had a vortex. A lead is green only when all three
    hold, which is why a bar can turn amber where the *error curve* of one panel still looks flat.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .viz import BG, DIM, FADING, FG, GOOD, POOR

    fig = plt.figure(figsize=size, facecolor=BG)
    ax = fig.add_axes([0.105, 0.46, 0.86, 0.36])
    axb = fig.add_axes([0.135, 0.25, 0.83, 0.13])
    for a in (ax, axb):
        a.set_facecolor(BG)
        for s in a.spines.values():
            s.set_color("#253040")
    names = list(skills)
    palette = ["#5fb3ff", "#ff9f43", "#c792ea", "#3ddc97", "#ff5d5d"]
    leads = np.asarray(skills[names[0]]["leads_h"], float)
    days = leads / 24.0
    for k, name in enumerate(names):
        skill = skills[name]
        thr = horizons[name]["thresholds"]
        colour = palette[k % len(palette)]
        ax.plot(days, np.asarray(skill["track_km"], float), color=colour, lw=2.0, label=name)
        track_h = horizons[name]["track_hours"]
        if np.isfinite(track_h):
            ax.axvline(track_h / 24.0, color=colour, ls="--", lw=1.0, alpha=0.7)
            ax.annotate(f"{name}: {thr['track_km']:.0f} km at {track_h / 24:.1f} d",
                        (track_h / 24.0, ax.get_ylim()[1] * 0.02 + 0.02 * k), color=colour, fontsize=8, ha="left",
                        xytext=(4, 0), textcoords="offset points")
        for row, lead in enumerate(leads):
            hit = (row < len(skill["track_km"]) and np.isfinite(skill["track_km"][row])) or False
            found = float(skill["found_fraction"][row]) >= thr["found_fraction"]
            if not found:
                state, colour_row = "the grid lost the storm", POOR
            elif not hit:
                state, colour_row = "no vortex to compare", POOR
            elif (float(skill["track_km"][row]) > thr["track_km"]
                  or abs(float(skill["wind_kt"][row])) > thr["wind_kt"]):
                state, colour_row = "past one of its thresholds", FADING
            else:
                state, colour_row = "inside both thresholds", GOOD
            axb.barh(k, 6.0 / 24.0, left=lead / 24.0 - 3.0 / 24.0, height=0.55, color=colour_row,
                     alpha=0.9 if state == "inside both thresholds" else 0.75, zorder=2)
    ax.axhline(horizons[names[0]]["thresholds"]["track_km"], color=DIM, ls=":", lw=1.0)
    ax.text(float(days[-1]), horizons[names[0]]["thresholds"]["track_km"] * 1.03,
            f"the track threshold ({horizons[names[0]]['thresholds']['track_km']:.0f} km)", color=DIM, fontsize=8,
            ha="right")
    ax.set_ylabel("median track error (km)", color=DIM, fontsize=9)
    ax.tick_params(colors=DIM, labelsize=8)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=FG)
    ax.set_xlim(0, float(days[-1]))
    axb.set_xlim(0, float(days[-1]))
    axb.set_yticks(range(len(names)))
    axb.set_yticklabels(names, color=FG, fontsize=8.5)
    axb.set_ylim(-0.6, len(names) - 0.4)
    axb.tick_params(colors=DIM, labelsize=8)
    axb.set_xlabel("lead time (days)", color=DIM, fontsize=8.5)
    axb.set_title("confidence bar: green = inside both thresholds and the storm still found  ·  amber = past one of "
                  "them  ·  red = the grid lost it", color=DIM, fontsize=7.8, loc="left")
    for row, name in enumerate(names):
        h = horizons[name]
        bits = [f"track {h['track_hours'] / 24:.1f} d" if np.isfinite(h["track_hours"]) else "track held to the end",
                f"intensity {h['intensity_hours'] / 24:.1f} d" if np.isfinite(h["intensity_hours"])
                else "intensity held to the end",
                f"storm lost at {h['lost_hours']:.0f} h" if np.isfinite(h["lost_hours"]) else "storm found at every lead"]
        fig.text(0.035, 0.125 - 0.042 * row, f"{name}: " + " · ".join(bits), color=palette[row % len(palette)],
                 fontsize=8.2)
    fig.text(0.035, 0.955, title, color=FG, fontsize=13.5, weight="bold")
    fig.text(0.035, 0.922, "\n".join(textwrap.wrap(note, 112)) if note else "", color=DIM, fontsize=8.4, va="top",
             linespacing=1.4)
    path = Path(path)
    fig.savefig(path, dpi=dpi, facecolor=BG)
    plt.close(fig)
    return path


def _stage_status(skill: dict, thresholds: dict, k: int) -> tuple[str, str]:
    """The confidence word of one lead, from the same three numbers :func:`confidence_by_lead` colours."""
    from .viz import DIM, FADING, GOOD, POOR

    if k >= len(skill["track_km"]):
        return "no longer measured", POOR
    if float(skill["found_fraction"][k]) < thresholds["found_fraction"]:
        return "the grid lost the storm", POOR
    track, wind = float(skill["track_km"][k]), abs(float(skill["wind_kt"][k]))
    if not np.isfinite(track):
        return "no vortex to compare", POOR
    if track > thresholds["track_km"] or wind > thresholds["wind_kt"]:
        return "past one of its thresholds", FADING
    if k == 0:
        return "inside both thresholds", DIM
    return "inside both thresholds", GOOD


def _showcase_frame(ctx: dict, stage: str, k: int):
    """One frame of :func:`stages_video`: the map on the left, the number it is earning on the right."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    from .viz import (
        BG,
        DIM,
        FADING,
        FG,
        GOOD,
        POOR,
        _globe_rgb,
        coastlines,
        regional,
        regional_lines,
    )

    figw, figh = ctx["size"]
    fig = plt.figure(figsize=(figw, figh), facecolor=BG)
    ax_map = fig.add_axes([0.02, 0.34, 0.52, 0.50])
    ax_num = fig.add_axes([0.605, 0.34, 0.365, 0.50])
    for ax in (ax_map, ax_num):
        ax.set_facecolor(BG)
        for s in ax.spines.values():
            s.set_color("#1d2634")
        ax.tick_params(colors=DIM, labelsize=7.5)
    best = ctx["best"]
    palette = ["#5fb3ff", "#ff9f43", "#c792ea", "#3ddc97", "#ff5d5d"]
    extent, width, height = ctx["extent"], ctx["width"], ctx["height"]
    cmap = plt.get_cmap("cividis").with_extremes(bad=BG)
    norm = Normalize(ctx["vmin"], ctx["vmax"])
    rows: list[tuple[str, str, str | None]] = []
    stamp = ""

    if stage == "analysis":
        times = np.asarray(ctx["analysis"]["times"])
        step = int(np.clip(k, 0, len(times) - 1))
        stage_text = "STAGE 1 — NOW: the reanalysis, and the vortex inside it"
        ax_map.axis("off")
        img = _globe_rgb(ctx["analysis"]["fields"][step], ctx["analysis"]["lat"],
                         np.mod(ctx["analysis"]["lon"], 360.0), ctx["centre"], ctx["globe_size"], cmap, norm,
                         coastlines())
        ax_map.imshow(img, interpolation="nearest")     # the box's edge is a data edge, not a gradient to smear
        ax_map.set_xlim(-2, ctx["globe_size"] + 2)
        ax_map.set_ylim(ctx["globe_size"] + 2, -2)
        stamp = np.datetime_as_string(np.datetime64(times[step], "h"), unit="h").replace("T", " ") + " UTC"
        err = ctx["analysis"]["errors"]
        j = int(np.clip(step, 0, len(err.track_km) - 1))
        colour = palette[0]
        ax_num.plot(np.arange(len(err.track_km)), np.asarray(err.track_km, float), color=colour, lw=1.0, alpha=0.3)
        ax_num.plot(np.arange(j + 1), np.asarray(err.track_km, float)[:j + 1], color=colour, lw=2.2)
        ax_num.plot(j, float(err.track_km[j]), "o", color=colour, ms=7, mec=BG, mew=1.2)
        ax_num.axvline(j, color=FG, lw=0.9, alpha=0.5)
        ax_num.set_ylabel("the analysed centre, km from the official one", color=DIM, fontsize=8)
        ax_num.set_xlabel("analysis time, 6 h apart", color=DIM, fontsize=8.5)
        ax_num.set_ylim(0, max(60.0, float(np.nanmax(err.track_km)) * 1.2))
        ax_num.set_title(f"at this time: {float(err.track_km[j]):.0f} km, "
                         f"{float(err.pressure_hpa[j]):+.0f} hPa", color=FG, fontsize=9.5, loc="left")
        rows = [("the analysed centre is", f"{float(err.track_km[j]):.0f} km from the official position", FG),
                ("its central pressure is", f"{float(err.pressure_hpa[j]):+.0f} hPa off the official one", FG),
                ("stage 1 answers", f"does the reanalysis hold the storm at all -- it holds it at "
                                    f"{int(np.sum(err.found))} of its {len(err)} times", GOOD)]
    else:
        model = ctx["current_model"]
        colour = palette[ctx["models"].index(model) % len(palette)]
        if stage == "grid":
            source = ctx["coarse"][model]
            stage_text = f"STAGE 3 — THE GRID: {model} on the 5.625 degree archive"
            fields = np.asarray(source["fields"], float)
            step = int(np.clip(k, 0, len(fields) - 1))
            field = fields[step]
            track = source["tracks"][model]
        else:
            source = ctx
            stage_text = (f"STAGE 2 — THE NEXT DAYS: {model}, {ctx['n_inits']} initialisations"
                          if ctx["n_inits"] > 1 else f"STAGE 2 — THE NEXT DAYS: {model}")
            step = int(np.clip(k, 0, len(ctx["leads"]) - 1))
            field = ctx["fields"][step]
            track = ctx["tracks"][model]
        leads = np.asarray(source["leads"], float)
        step = int(np.clip(step, 0, len(leads) - 1))
        img = regional(field, np.asarray(source["lat"], float), np.mod(np.asarray(source["lon"], float), 360.0),
                       extent, (width, height))
        ax_map.imshow(img, cmap=cmap, norm=norm, interpolation="bilinear")
        ax_map.set_xlim(0, width)
        ax_map.set_ylim(height, 0)
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        for ln in coastlines():
            px, py = regional_lines([ln], extent, (width, height))[0]
            ax_map.plot(px, py, color="#8fa0b4", lw=0.7)
        obs_px, obs_py = _to_pixel(best.lat, np.mod(best.lon, 360.0), extent, width, height)
        ax_map.plot(obs_px, obs_py, color="#ffffff", lw=1.6, alpha=0.85)
        ax_map.plot(obs_px[::4], obs_py[::4], "o", color=BG, ms=3.0, mec="#ffffff", mew=0.9)
        tla = np.asarray(track.lat, float)[:step + 1]
        tlo = np.asarray(track.lon, float)[:step + 1]
        ok = np.isfinite(tla)
        px, py = _to_pixel(tla[ok], np.mod(tlo[ok], 360.0), extent, width, height)
        ax_map.plot(px, py, color=colour, lw=2.2, zorder=5)
        if len(px):
            ax_map.plot(px[-1], py[-1], "o", color=colour, ms=8, mec=BG, mew=1.2, zorder=6)
        stamp = f"lead +{leads[step]:.0f} h  ·  valid " + np.datetime_as_string(
            np.datetime64(source["times"][step], "h"), unit="h").replace("T", " ") + " UTC"
        days = leads / 24.0
        skill = source["skill"]
        thresholds = source["thresholds"]
        if stage == "grid":
            ax_num.plot(days, np.asarray(skill["found_fraction"], float), color=colour, lw=1.0, alpha=0.3)
            ax_num.plot(days[:step + 1], np.asarray(skill["found_fraction"], float)[:step + 1], color=colour, lw=2.2)
            ax_num.set_ylim(-0.05, 1.08)
            ax_num.set_ylabel("share of the forecasts whose grid held the storm", color=DIM, fontsize=8)
            ax_num.axvline(float(days[step]), color=FG, lw=0.9, alpha=0.5)
            ax_num.set_xlim(0, float(days[-1]))
            share = float(skill["found_fraction"][step])
            ax_num.set_title(f"at +{leads[step]:.0f} h: found on this grid in {share:.0%} of the forecasts",
                             color=FG, fontsize=9.5, loc="left")
            rows = [("one cell is", "625 km across, wider than the core", FADING),
                    ("at this lead the grid", "still holds it" if share > 0 else "cannot hold it at all",
                     GOOD if share > 0 else POOR),
                    ("over the whole stream", f"{100 * float(source['found_fraction_overall']):.0f}% of the leads "
                                              f"find it, and {float(source['median_pressure_hpa']):+.0f} hPa too "
                                              f"shallow where they do", FADING)]
        else:
            y = np.asarray(skill["track_km"], float)
            ax_num.plot(days, y, color=colour, lw=1.0, alpha=0.3)
            ax_num.plot(days[:step + 1], y[:step + 1], color=colour, lw=2.2)
            ax_num.axhline(thresholds["track_km"], color=DIM, ls=":", lw=1.0)
            ax_num.text(0.98, 0.90, f"the track threshold, {thresholds['track_km']:.0f} km", color=DIM, fontsize=7.2,
                        ha="right", va="bottom", transform=ax_num.transAxes)
            ax_num.set_ylabel("median track error (km)", color=DIM, fontsize=8)
            ax_num.axvline(float(days[step]), color=FG, lw=0.9, alpha=0.5)
            ax_num.set_xlim(0, float(days[-1]))
            ax_num.set_ylim(0, max(float(np.nanmax(y)) * 1.15, thresholds["track_km"] * 1.1))
            word, word_colour = _stage_status(skill, thresholds, step)
            ax_num.set_title(f"at +{leads[step]:.0f} h: {y[step]:.0f} km, {float(skill['wind_kt'][step]):+.0f} kt",
                             color=FG, fontsize=9.5, loc="left")
            h = source["horizons"]
            rows = [("this lead is", word, word_colour),
                    ("its intensity horizon is", f"{h['intensity_hours']:.0f} h" if np.isfinite(h["intensity_hours"])
                     else "the whole stream", FG),
                    ("its track horizon is", f"{h['track_hours']:.0f} h" if np.isfinite(h["track_hours"])
                     else "not reached inside the leads the archive publishes", FG)]
        ax_num.set_xlabel("lead time (days)", color=DIM, fontsize=8.5)

    fig.text(0.02, 0.965, ctx["title"], color="#9fb0c4", fontsize=8.4)
    fig.text(0.02, 0.915, stage_text, color=FG, fontsize=12.5, weight="bold")
    fig.text(0.02, 0.882, stamp, color="#9fb0c4", fontsize=9.5)
    for r, (head, value, colour) in enumerate(rows):
        y = 0.275 - 0.058 * r
        fig.text(0.02, y, head, color=DIM, fontsize=7.8)
        fig.text(0.02, y - 0.032, value, color=colour or FG, fontsize=8.2, weight="bold")
    fig.text(0.02, 0.018, "\n".join(textwrap.wrap(ctx["note"], 138)), color="#6c7b8d", fontsize=6.8,
             linespacing=1.3)
    if stage == "analysis":
        fig.text(0.02, 0.318, "drawn: the storm's box the reanalysis was read on; the rest of the globe carries no "
                              "data here, not a calm", color="#4a5869", fontsize=6.6)
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def stages_video(path, *, best: BestTrack, analysis: dict, tracks: dict, skills: dict, horizons: dict,
                 coarse: dict | None = None, title: str = "", note: str = CORE_NOTE, fps: float = 12.0,
                 hold: int = 8, size=(8.0, 4.5), dpi: int = 100, globe_size: int = 300,
                 vmin: float | None = None, vmax: float | None = None, centre=None, extent=None,
                 n_inits: int = 0):
    """The staged video of the hurricane chain: the analysis, the forecasts by lead, and the coarse grid that loses
    the core -- one frame per step, each captioned with the instant and the stage, the honest note always on screen.

    Parameters
    ----------
    path : str or Path
        File written (``.mp4``: H.264 through :func:`pinneapple_design.bricks.visuals.write_mp4`).
    best : BestTrack
        The official track, drawn on every map.
    analysis : dict
        The reanalysis stage: ``fields`` (``(n, lat, lon)`` hPa), ``times``, ``lat``, ``lon``, and ``errors`` (a
        :class:`StormErrors` against the official track).
    tracks : dict
        ``{model: StormTrack}`` -- the track each forecast actually followed, one per model, drawn lead by lead.
    skills, horizons : dict
        ``{model: skill_by_lead(...)}`` and ``{model: storm_horizons(...)}``: the curves and the thresholds the right
        panel quotes at every lead.
    coarse : dict, optional
        ``{model: {"fields", "lat", "lon", "times", "leads", "skill", "thresholds", "horizons", "tracks",
        "found_fraction_overall", "median_pressure_hpa"}}``: the same forecasts on the 5.625-degree archive, which is
        the stage that says why the chain hands over.
    n_inits : int
        How many initialisations the medians in ``skills`` are over, printed on the frame: a median of one forecast
        and a median of ten are not the same claim.
    hold : int
        Frames each step is held for. The steps of the data are far fewer than the frames of a 60-to-90-second video,
        and holding is how the two are reconciled: ``fps * hold`` sets the reading time per step.

    Returns
    -------
    path : Path
        The file written.

    Notes
    -----
    Nothing here is a new measurement: every number on the frame is the one :func:`verify_storm`,
    :func:`skill_by_lead` and :func:`storm_horizons` already produced, and the picture is written from the same arrays
    the figures and the JSON are. The stages are the ones this repository can draw -- the satellite, the hydrology
    chain and the street of issue #366 belong to other issues and are not in this data, which the note on the screen
    says by not claiming them.
    """
    import matplotlib

    matplotlib.use("Agg")

    from pinneapple_design.bricks.visuals import write_mp4

    from .viz import BG, coastlines  # noqa: F401  (coastlines are drawn inside _showcase_frame)

    analysis = dict(analysis)
    analysis["lat"] = np.asarray(analysis["lat"], float)
    analysis["lon"] = np.asarray(analysis["lon"], float)
    analysis["fields"] = np.asarray(analysis["fields"], float)
    if vmin is None or vmax is None:                  # the display scale of the data on screen, not of the planet
        finite = analysis["fields"][np.isfinite(analysis["fields"])]
        vmin = float(np.percentile(finite, 0.5)) if vmin is None else vmin
        vmax = float(np.percentile(finite, 97.0)) if vmax is None else vmax
    models = list(skills)
    if centre is None:
        j = int(np.nanargmin(np.asarray(best.pressure_hpa, float)))
        centre = (float(best.lat[j]), float(np.mod(best.lon[j], 360.0)))
    extent = _extent_of(best, tracks) if extent is None else extent
    ctx = {"best": best, "analysis": analysis, "tracks": tracks, "coarse": coarse or {}, "models": models,
           "size": size, "dpi": dpi, "globe_size": globe_size, "vmin": vmin, "vmax": vmax, "centre": centre,
           "extent": extent, "width": 300, "height": 210, "title": title, "note": note, "n_inits": int(n_inits)}
    frames = []
    for k in range(len(np.asarray(analysis["times"]))):
        frames += [_showcase_frame(ctx, "analysis", k)] * hold
    for model in models:
        ctx["current_model"] = model
        step = {"leads": np.asarray(skills[model]["leads_h"], float), "skill": skills[model],
                "thresholds": horizons[model]["thresholds"], "horizons": horizons[model],
                "fields": analysis["fields"], "lat": analysis["lat"], "lon": analysis["lon"],
                "times": _valid_times(tracks[model])}
        ctx.update(step)
        for k in range(len(step["leads"])):
            frames += [_showcase_frame(ctx, "forecast", k)] * hold
    if coarse:
        for model in models:
            if model not in coarse:
                continue
            ctx.update({"current_model": model})
            for k in range(len(np.asarray(coarse[model]["leads"], float))):
                frames += [_showcase_frame(ctx, "grid", k)] * hold
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return Path(write_mp4(str(path), frames, fps=fps))


def _valid_times(track: StormTrack) -> np.ndarray:
    """The valid times of a tracked series, as datetime64, whatever the caller stored."""
    times = np.asarray(track.times)
    if times.dtype.kind == "M":
        return times.astype("datetime64[h]")
    return np.array([np.datetime64(str(t), "h") for t in times], dtype="datetime64[h]")
