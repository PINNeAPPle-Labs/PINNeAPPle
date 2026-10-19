"""Accuracy checks of the observatory's methods against independent references, as data.

Each check returns a row ``{id, method, reference, metric, value, threshold, unit, status, n}``; the test suite asserts
that every row passes and the observatory app shows the same table. References (``data/reference/``):

* USNO NOVAS C 3.1 apparent places (geocentric, true equator and equinox of date), 4 epochs x 10 bodies
* JPL Horizons apparent ecliptic positions and a topocentric Moon
* USNO 2018 almanac: sunrise/sunset, Moon phases, equinoxes/solstices, Moon illumination
* NASA eclipse and transit predictions (instant of greatest eclipse)
* ISS passes from an independent pass finder (skyfield) on the same TLE
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np

from . import frames
from .astrometry import Observer, apparent
from .timescales import DAY_S, Time, jd_from_datetime

REF = os.path.join(os.path.dirname(__file__), "data", "reference")


def _load(name):
    with open(os.path.join(REF, name), encoding="utf-8") as fh:
        return json.load(fh)


def _jd(iso):
    return jd_from_datetime(datetime.fromisoformat(iso.replace("Z", "+00:00")).replace(tzinfo=timezone.utc))


def _row(key, method, reference, metric, value, threshold, unit, n, warn_only=False):
    ok = value <= threshold
    return {"id": key, "method": method, "reference": reference, "metric": metric, "value": float(value),
            "threshold": float(threshold), "unit": unit, "n": int(n),
            "status": "pass" if ok else ("warn" if warn_only else "fail")}


def check_novas():
    d = _load("novas_apparent.json")
    errs, errs_moon = [], []
    for c in d["cases"]:
        p = apparent(c["body"], Time.from_tt(c["jd_tt"]))
        dra = (p.ra[0] - c["ra_date_deg"] + 180) % 360 - 180
        e = np.hypot(dra * np.cos(np.radians(c["dec_date_deg"])), p.dec[0] - c["dec_date_deg"]) * 3600
        (errs_moon if c["body"] == "moon" else errs).append(e)
    return [_row("novas_planets", "Apparent places, Sun and planets (own pipeline + DE421)", "USNO NOVAS C 3.1 / DE405",
                 "max angular error", max(errs), 1.0, "arcsec", len(errs)),
            _row("novas_moon", "Apparent place, Moon", "USNO NOVAS C 3.1 / DE405", "max angular error", max(errs_moon), 2.0,
                 "arcsec", len(errs_moon))]


def check_horizons():
    d = _load("horizons.json")
    rows = []
    errs = []
    for c in d["cases"]:
        if "lon_deg" in c:
            t = Time(_jd(c["utc"]))
            p = apparent(c["body"], t)
            v = frames.ecliptic_matrix(float(t.tt[()] if np.ndim(t.tt) == 0 else t.tt[0])) @ p.unit_gcrs[0]
            lon = np.degrees(np.arctan2(v[1], v[0])) % 360
            lat = np.degrees(np.arcsin(v[2]))
            e = np.hypot(((lon - c["lon_deg"] + 180) % 360 - 180) * np.cos(np.radians(lat)), lat - c["lat_deg"]) * 3600
        else:
            tt = c["tdb_jd"]                                         # TDB = TT within 2 ms
            dut1 = (32.184 + 35.0) - c["tt_minus_ut1_s"]               # UT1-UTC; TAI-UTC was 35 s in March 2015
            t = Time(tt - (32.184 + 35.0) / DAY_S, dut1)
            o = c["observer"]
            p = apparent(c["body"], t, Observer(o["lat"], o["lon"], o["elevation_m"]))
            e = frames.angular_separation(p.ra_astrometric[0], p.dec_astrometric[0], c["ra_deg"], c["dec_deg"]) * 3600
        errs.append(e)
    rows.append(_row("horizons", "Apparent / astrometric places (Sun, Mars, topocentric Moon)", "JPL Horizons",
                     "max angular error", max(errs), 1.0, "arcsec", len(errs)))
    return rows


def check_usno_almanac():
    from .phenomena import illuminated_fraction, lunar_phase_events, rise_transit_set, seasons
    d = _load("usno_almanac.json")
    dts = []
    for c in d["sunrise_sunset"]:
        r = rise_transit_set("sun", Observer(c["lat"], c["lon"], 0.0), _jd(c["date_utc_start"]))
        dts += [abs(r["rise"][0] - _jd(c["rise"] + ":00Z")) * DAY_S, abs(r["set"][0] - _jd(c["set"] + ":00Z")) * DAY_S]
    rows = [_row("sunrise", "Sunrise / sunset", "USNO tables (to the minute)", "max |difference|", max(dts), 60.0, "s",
                 len(dts))]
    ph = lunar_phase_events(_jd(d["moon_phases"][0]["utc"] + ":00Z") - 2, _jd(d["moon_phases"][-1]["utc"] + ":00Z") + 2)
    dt = [min(abs(e["jd"] - _jd(c["utc"] + ":00Z")) for e in ph if e["event"] == c["event"]) * DAY_S
          for c in d["moon_phases"]]
    se = seasons(_jd(d["seasons"][0]["utc"] + ":00Z") - 3, _jd(d["seasons"][-1]["utc"] + ":00Z") + 3)
    dt2 = [min(abs(e["jd"] - _jd(c["utc"] + ":00Z")) for e in se if e["event"] == c["event"]) * DAY_S
           for c in d["seasons"]]
    rows.append(_row("moon_phases", "Moon phases", "USNO tables (to the minute)", "max |difference|", max(dt), 60.0, "s", len(dt)))
    rows.append(_row("seasons", "Equinoxes / solstices", "USNO tables (to the minute)", "max |difference|", max(dt2), 60.0, "s",
                     len(dt2)))
    ill = d["moon_illumination_0500utc"]
    k = illuminated_fraction("moon", Time(np.array([_jd(x + "T05:00:00Z") for x in ill["dates"]])))
    rows.append(_row("moon_fraction", "Moon illuminated fraction", "USNO tables (2 decimals)", "max |difference|",
                     float(np.max(np.abs(np.round(k, 2) - ill["fraction"]))), 0.01, "-", len(k)))
    return rows


def check_eclipses():
    from .phenomena import lunar_eclipses, planetary_transits, solar_eclipses
    d = _load("nasa_eclipses.json")
    dts = []
    for c in d["solar_eclipses"]:
        ev = solar_eclipses(_jd(c["utc"]) - 3, _jd(c["utc"]) + 3)
        dts.append(min(abs(e["jd"] - _jd(c["utc"])) for e in ev) * DAY_S)
    for c in d["lunar_eclipses"]:
        ev = lunar_eclipses(_jd(c["utc"]) - 3, _jd(c["utc"]) + 3)
        dts.append(min(abs(e["jd"] - _jd(c["utc"])) for e in ev) * DAY_S)
    tr = []
    for c in d["transits"]:
        ev = planetary_transits(_jd(c["utc"]) - 5, _jd(c["utc"]) + 5)
        tr.append(min(abs(e["jd"] - _jd(c["utc"])) for e in ev) * DAY_S)
    return [_row("eclipses", "Solar and lunar eclipses (greatest eclipse)", "NASA / Espenak predictions", "max |difference|",
                 max(dts), d["tolerance_s"], "s", len(dts)),
            _row("transits", "Transits of Mercury and Venus (greatest transit)", "NASA predictions", "max |difference|", max(tr),
                 d["tolerance_s"], "s", len(tr))]


def check_iss_passes():
    from .satellites import Satellite, find_passes
    d = _load("iss_passes_reference.json")
    sat = Satellite(*d["tle"])
    dts = []
    for o in d["observers"]:
        passes = find_passes(sat, Observer(o["lat"], o["lon"], o["elevation_m"]), Time(o["jd_utc_start"]), o["days"],
                             min_alt=0.0)
        mine = {"rise": [p["rise_jd"] for p in passes if p["rise_jd"]],
                "culmination": [p["culmination_jd"] for p in passes],
                "set": [p["set_jd"] for p in passes if p["set_jd"]]}
        for e in o["events"]:
            if mine[e["event"]]:
                dts.append(min(abs(np.array(mine[e["event"]]) - e["jd_utc"])) * DAY_S)
    return [_row("iss_passes", "ISS pass times (rise / culmination / set)", "independent pass finder (skyfield), same TLE",
                 "max |difference|", max(dts), 2.0, "s", len(dts))]


def check_skyfield_live(n: int = 24, seed: int = 1):
    """Random epochs 1972-2049 (the UTC era) and observers: topocentric apparent alt/az of all bodies vs skyfield (optional)."""
    try:
        from skyfield.api import load, wgs84

        from .ephemeris import find_kernel
    except ImportError:
        return []
    ts = load.timescale()
    eph = load(find_kernel())
    names = {"sun": "sun", "moon": "moon", "mercury": "mercury", "venus": "venus", "mars": "mars",
             "jupiter": "jupiter barycenter", "saturn": "saturn barycenter", "uranus": "uranus barycenter",
             "neptune": "neptune barycenter"}
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n):
        jd = rng.uniform(2441317.5, 2469807.5)
        lat, lon = rng.uniform(-70, 70), rng.uniform(-180, 180)
        t = ts.ut1_jd(jd)
        mine_t = Time(jd - t.dut1 / DAY_S, float(t.dut1))
        top = eph["earth"] + wgs84.latlon(lat, lon, 0)
        obs = Observer(lat, lon, 0.0)
        for b, sb in names.items():
            p = apparent(b, mine_t, obs, refraction=False)
            alt, az, _ = top.at(t).observe(eph[sb]).apparent().altaz()
            errs.append(frames.angular_separation(p.az[0], p.alt[0], az.degrees, alt.degrees) * 3600)
    return [_row("skyfield_altaz", "Topocentric alt/az, all bodies (random epochs 1972-2049, random sites)", "skyfield 1.x",
                 "max angular error", max(errs), 0.5, "arcsec", len(errs))]


def check_integrators():
    from .ephemeris import default_ephemeris
    from .nbody import initial_state, integrate
    r0, v0, mu = initial_state(2451545.0)
    rows = []
    out = integrate(r0, v0, mu, 365.25 * 86400 * 2, 6 * 3600, "yoshida4", n_out=40)
    rows.append(_row("nbody_energy", "N-body Yoshida-4 (dt 6 h, 2 yr): energy conservation", "exact invariant",
                     "max |dE/E|", float(np.abs(out["rel_energy_error"]).max()), 1e-8, "relative", 40))
    rows.append(_row("nbody_angmom", "N-body Yoshida-4: angular momentum conservation", "exact invariant", "max |dL|/|L|",
                     float(out["rel_angmom_error"].max()), 1e-12, "relative", 40))
    p, _ = default_ephemeris().barycentric("mars", np.array([2451545.0 + 730.5]))
    err = np.linalg.norm(out["r"][-1, 5] - p[0])
    rows.append(_row("nbody_vs_de421", "N-body Newtonian point masses vs DE421 (Mars, 2 yr)", "JPL DE421", "position difference", err,
                     500.0, "km", 1))
    return rows


def run_all(include_skyfield: bool = True) -> list[dict]:
    rows = check_novas() + check_horizons() + check_usno_almanac() + check_eclipses() + check_iss_passes() + \
        check_integrators()
    if include_skyfield:
        rows += check_skyfield_live()
    return rows
