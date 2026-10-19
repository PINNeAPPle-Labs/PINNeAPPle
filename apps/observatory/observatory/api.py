"""PINNeAPPle Observatory: web API + single-page 3-D UI.

The sky and the solar system as they are now from the visitor's location, computed by pinneapple_physics.astro
(JPL DE421, IAU 2006 precession / nutation, apparent places, refraction), with satellites from CelesTrak TLEs,
astrophysics analyses, and the PINNeAPPle engine (a physics-informed orbit surrogate checked against the classical
integrator) shown with its trust checks and the validation of every method against independent references.

Run:  uvicorn observatory.api:app --port 8094   (from apps/observatory)

Environment (all optional):
  OBS_USER / OBS_PASSWORD   HTTP Basic login on everything except /health
  OBS_MAX_HEAVY             concurrent heavy computations per worker (default 2)
  OBS_TLE_GROUPS            CelesTrak groups to load (default "stations,visual")
  OBS_TLE_FILE              a local TLE file used instead of CelesTrak (offline demos, tests)
  OBS_CACHE_DIR             TLE cache directory (default ~/.cache/pinneapple/tle)
  OBS_PRECOMPUTE            "0" to skip the background warm-up (validation table, default surrogate)
"""
from __future__ import annotations

import collections
import io
import json
import math
import os
import sys
import threading
import time
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from pinneapple_physics.astro import frames
from pinneapple_physics.astro.astrometry import Observer, airmass, apparent, extinction_mag
from pinneapple_physics.astro.ephemeris import AU_KM, PLANETS, RADIUS_KM, default_ephemeris
from pinneapple_physics.astro.phenomena import (
    lunar_eclipses,
    lunar_phase_events,
    magnitude,
    moon_phase,
    phase_angle,
    planet_events,
    planetary_transits,
    rise_transit_set,
    seasons,
    sky_state,
    solar_eclipses,
    twilight_events,
)
from pinneapple_physics.astro.timescales import Time, gast_rad

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import BusyLimiter, install  # noqa: E402

VERSION = "1.0.0"
HERE = os.path.dirname(__file__)
STATIC = os.path.join(HERE, "static")
BODIES = ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"]
SYSTEM = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]

app = FastAPI(title="PINNeAPPle Observatory", version=VERSION,
              description="The sky and the solar system, now, from where you stand: JPL ephemerides, satellites, "
                          "astrophysics analyses and a physics-informed orbit surrogate with its trust checks.")
app.add_middleware(GZipMiddleware, minimum_size=1500)
install(app, prefix="OBS")
_HEAVY = BusyLimiter("OBS_MAX_HEAVY")
EPH = default_ephemeris()


def _error(status: int, code: str, message: str, **args) -> HTTPException:
    """An HTTP error whose message the page shows in the visitor's language: ``code`` names the message in the
    interface catalogues (static/i18n), ``args`` fill its placeholders; ``detail`` stays English for API users."""
    return HTTPException(status, message, headers={"X-Error-Code": code, "X-Error-Args": json.dumps(args)})


def _time(t: str | None) -> Time:
    if not t or t == "now":
        return Time.now()
    try:
        tt = Time.from_iso(t)
    except ValueError as exc:
        raise _error(400, "bad_time", f"bad time {t!r}: use ISO 8601, e.g. 2026-10-13T21:00:00Z", value=t) from exc
    lo, hi = EPH.jd_range
    if not lo + 2 < float(tt.tdb) < hi - 2:
        raise _error(400, "time_span", "time outside the DE421 span (1900-2050)")
    return tt


def _obs(lat: float, lon: float, elev: float) -> Observer:
    if not (-90 <= lat <= 90 and -180 <= lon <= 360):
        raise _error(400, "bad_location", "latitude must be in [-90, 90] and longitude in [-180, 360]")
    return Observer(lat, ((lon + 180) % 360) - 180, elev)


def _iso(jd):
    return None if jd is None else Time(jd).iso()


def _r(x, n=5):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), n)


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION, "ephemeris": os.path.basename(EPH.path)}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ------------------------------------------------------------------------------------------------- catalogue
@lru_cache(maxsize=1)
def _catalog_payload():
    from pinneapple_physics.astro.stars import load_bright_stars, load_constellations
    c = load_bright_stars()
    return {
        "source": "Yale Bright Star Catalogue 5th rev. ed. (Hoffleit & Warren 1991); constellation figures "
                  "d3-celestial (BSD-3, O. Frohn)",
        "hr": c.hr.tolist(), "ra": np.round(c.ra, 4).tolist(), "dec": np.round(c.dec, 4).tolist(),
        "pmra": np.nan_to_num(c.pmra).tolist(), "pmdec": np.nan_to_num(c.pmdec).tolist(),
        "v": c.v.tolist(), "bv": [None if np.isnan(x) else round(float(x), 3) for x in c.bv],
        "name": [c.label(i) if (c.name[i] or c.v[i] < 2.6) else "" for i in range(len(c))],
        "proper": [bool(c.name[i]) for i in range(len(c))],
        "constellations": load_constellations(),
    }


@app.get("/api/catalog")
def catalog():
    return JSONResponse(_catalog_payload(), headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/milkyway.jpg")
def milkyway():
    import pinneapple_physics.astro as astro
    return FileResponse(os.path.join(os.path.dirname(astro.__file__), "data", "milkyway.jpg"),
                        headers={"Cache-Control": "public, max-age=86400"})


# ----------------------------------------------------------------------------------------------- learn (facts)
def _lang(lang: str) -> str:
    if lang not in ("en", "pt"):
        raise _error(400, "bad_lang", "lang must be 'en' or 'pt'")
    return lang


@app.get("/api/facts/card")
def facts_card(id: str, lang: str = "en", lat: float | None = None, lon: float | None = None, elev: float = 0.0,
               t: str | None = None):
    """The info card of an object (pinneapple_physics.astro.facts): key numbers with a source each and everyday
    comparisons, curiosities with sources, history, missions, a quiz; with lat/lon, how to see it tonight."""
    from pinneapple_physics.astro import facts
    try:
        card = facts.card(id, _lang(lang))
    except KeyError as exc:
        raise _error(404, "no_card", str(exc), value=id) from exc
    if lat is not None and lon is not None:
        tt, obs = _time(t), _obs(lat, lon, elev)
        if card["kind"] == "satellite":
            from pinneapple_physics.astro.satellites import find_passes
            sats, _ = _satellites()
            sat = next((s for s in sats if s.norad == card["norad"] and abs(float(tt.utc) - s.epoch_jd) <= 30), None)
            card["passes"] = find_passes(sat, obs, tt, 3.0, eph=EPH)[:4] if sat else []
        else:
            card["tonight"] = facts.tonight(card["id"], obs, tt, eph=EPH)
    return card


@app.get("/api/facts/search")
def facts_search(q: str, lang: str = "en", limit: int = Query(12, ge=1, le=50)):
    from pinneapple_physics.astro import facts
    return {"q": q, "results": facts.search(q, _lang(lang), limit)}


@app.get("/api/facts/list")
def facts_list(kind: str | None = None, lang: str = "en"):
    from pinneapple_physics.astro import facts
    return {"kind": kind, "cards": facts.list_cards(kind, _lang(lang))}


@app.get("/api/facts/tip")
def facts_tip(lang: str = "en", seed: int | None = None):
    from pinneapple_physics.astro import facts
    return facts.did_you_know(_lang(lang), seed)


# ------------------------------------------------------------------------------------------- observing session
_SURVEY_DOWN = {"t": 0.0}


def _observe_target(target: str):
    """(spec for pinneapple_physics.astro.observing, display info) for a body, star:HR, M<n>, con-.. or facts id."""
    from pinneapple_physics.astro import facts
    key = target.strip().lower()
    if key in BODIES:
        return key, {"id": key, "kind": "body"}
    try:
        oid = facts.resolve(target)
    except KeyError as exc:
        raise _error(404, "unknown_target", str(exc), value=target) from exc
    pos = facts.position(oid)
    if pos is None:
        raise _error(404, "unknown_target", f"{target!r} has no fixed position", value=target)
    info = {"id": oid, "kind": facts.ids()[oid], "ra_deg": pos[0], "dec_deg": pos[1]}
    size = facts.numbers(oid).get("size_arcmin")
    if size:
        info["size_arcmin"] = size["value"]
    if oid.startswith("star-"):
        info["hr"] = int(oid[5:])
    return pos, info


@app.get("/api/observe/presets")
def observe_presets():
    from dataclasses import asdict

    from pinneapple_physics.astro import observing as ob
    return {"telescopes": {k: asdict(v) for k, v in ob.TELESCOPES.items()},
            "eyepieces": {k: asdict(v) for k, v in ob.EYEPIECES.items()},
            "cameras": {k: asdict(v) for k, v in ob.CAMERAS.items()}, "survey_credit": ob.SURVEY_CREDIT}


@app.get("/api/observe/plan")
def observe_plan(target: str = "saturn", lat: float = Query(-23.5505), lon: float = Query(-46.6333), elev: float = 0.0,
                 t: str | None = None, hours: float = Query(24.0, gt=1, le=48), step_min: float = Query(5.0, ge=1, le=30),
                 min_alt: float = Query(30.0, ge=0, le=80)):
    """The night for a target: altitude / air mass / extinction / sky brightness / field rotation curves, rise,
    transit and set, twilight, the Moon, the best window (pinneapple_physics.astro.observing.visibility)."""
    from pinneapple_physics.astro import observing as ob
    spec, info = _observe_target(target)
    tt, obs = _time(t), _obs(lat, lon, elev)
    with _HEAVY:
        v = ob.visibility(spec, obs, tt, hours, step_min, min_alt, eph=EPH)
    return {**v, "target": info, "t": tt.iso()}


@app.get("/api/observe/optics")
def observe_optics(telescope: str = "newton200", eyepiece: str | None = "plossl25", camera: str | None = None,
                   aperture_mm: float | None = Query(None, gt=5, le=2000), focal_mm: float | None = Query(None, gt=10, le=40000)):
    from pinneapple_physics.astro import observing as ob
    if aperture_mm and focal_mm:
        tel = ob.Telescope(f"{aperture_mm:g} mm f/{focal_mm / aperture_mm:.1f}", aperture_mm, focal_mm, "custom")
    elif telescope in ob.TELESCOPES:
        tel = ob.TELESCOPES[telescope]
    else:
        raise _error(404, "unknown_preset", f"unknown telescope {telescope!r}", value=telescope)
    ep = ob.EYEPIECES.get(eyepiece) if eyepiece else None
    cam = ob.CAMERAS.get(camera) if camera else None
    return ob.optics(tel, ep, cam)


@app.get("/api/observe/slew")
def observe_slew(from_alt: float, from_az: float, to_alt: float, to_az: float, lat: float = -23.5505,
                 mount: str = Query("altaz", pattern="^(altaz|equatorial)$"), rate: float = Query(4.0, gt=0.1, le=20)):
    from pinneapple_physics.astro import observing as ob
    return ob.slew((from_alt, from_az), (to_alt, to_az), lat, mount, rate_deg_s=rate)


@app.get("/api/observe/weather")
def observe_weather(lat: float = Query(-23.5505), lon: float = Query(-46.6333), t: str | None = None,
                    mode: str = Query("real", pattern="^(real|perfect)$"),
                    force: str | None = Query(None, pattern="^(clear|cloudy)$")):
    """Cloud cover from the Open-Meteo forecast and what it leaves of the view ("real sky"), or no weather at all
    ("perfect sky"). Without a forecast "real" falls back to "perfect" and says so. ``force`` replaces the
    forecast by a clear or overcast one (demos, tests); OBS_WEATHER_FILE reads a saved forecast (offline)."""
    from pinneapple_physics.astro import conditions as cd
    tt = _time(t)
    fc, err = None, None
    if force:
        jd0 = math.floor(float(tt.utc) * 24) / 24
        fc = {"time": [Time(jd0 + h / 24).iso() for h in range(-24, 73)], "cloud_cover": [100 if force == "cloudy" else 0] * 97,
              "source": f"forced {force} forecast"}
    elif os.environ.get("OBS_WEATHER_FILE"):
        fc = json.load(open(os.environ["OBS_WEATHER_FILE"]))
    elif mode == "real":
        try:
            fc = _forecast(round(lat, 2), round(lon, 2), int(time.time() // 1800))
        except Exception as exc:  # noqa: BLE001
            err = type(exc).__name__
    view = cd.sky_view(mode, fc, tt.iso())
    return {**view, "forecast": fc, "error": err, "threshold": cd.CLOUDY_PCT}


@lru_cache(maxsize=64)
def _forecast(lat, lon, half_hour):
    from pinneapple_physics.astro import conditions as cd
    return cd.fetch_cloud_forecast(lat, lon)


@app.get("/api/observe/field.jpg")
def observe_field(ra: float = Query(..., ge=0, le=360), dec: float = Query(..., ge=-90, le=90),
                  fov: float = Query(1.0, gt=0.01, le=10), size: int = Query(512, ge=64, le=1024)):
    """A survey cut-out (DSS2 colour from CDS hips2fits) around a position, cached on the server. 503 when the
    survey cannot be reached (the page then draws the field from the star catalogue)."""
    from pinneapple_physics.astro import observing as ob
    if os.environ.get("OBS_SURVEY", "1") == "0" or time.time() - _SURVEY_DOWN["t"] < 600:
        raise _error(503, "no_survey", "survey images unavailable")
    cache = os.path.join(os.environ.get("OBS_CACHE_DIR") or os.path.expanduser("~/.cache/pinneapple"), "hips")
    try:
        data = ob.survey_cutout(ra, dec, fov, size, size, cache_dir=cache, timeout=6.0)
    except Exception as exc:  # noqa: BLE001
        _SURVEY_DOWN["t"] = time.time()
        raise _error(503, "no_survey", f"survey images unavailable ({type(exc).__name__})") from exc
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=604800",
                                                           "X-Credit": ob.SURVEY_CREDIT.encode("ascii", "replace").decode()})


# ----------------------------------------------------------------------------------------------- space journey
@app.get("/api/journey/plan")
def journey_plan(target: str = "mars", lat: float = Query(-23.5505), lon: float = Query(-46.6333), elev: float = 0.0,
                 t: str | None = None):
    """The trip from the observer to a body or a star (pinneapple_physics.astro.journey.plan): start, end and arrival
    points (heliocentric ICRS km), distance, light time, travel times at real speeds, scale markers, the target's
    magnitude and size versus the distance left; plus the Earth's orientation for drawing."""
    from pinneapple_physics.astro import journey
    tt, obs = _time(t), _obs(lat, lon, elev)
    try:
        j = journey.plan(obs, target, tt, eph=EPH)
    except (KeyError, ValueError) as exc:
        raise _error(404, "no_journey", str(exc), value=target) from exc
    return {**j, "gast_deg": float(np.degrees(gast_rad(tt))), "npb": frames.npb_matrix(float(tt.tt)).tolist()}


@lru_cache(maxsize=8)
def _journey_scene(day_jd: float):
    from pinneapple_physics.astro import journey
    return journey.scene(Time(day_jd), eph=EPH)


@app.get("/api/journey/scene")
def journey_scene(t: str | None = None):
    """Planets, Sun and Moon (heliocentric ICRS km) for the date, and the bright stars with a usable parallax."""
    tt = _time(t)
    return JSONResponse(_journey_scene(round(float(tt.utc) * 96) / 96), headers={"Cache-Control": "public, max-age=900"})


# ------------------------------------------------------------------------------------------------------- sky
def _body_now(b: str, t: Time, obs: Observer) -> dict:
    p = apparent(b, t, obs)
    g = apparent(b, t, None)
    d = float(p.distance_km[0])
    out = {"name": b, "ra": _r(p.ra[0]), "dec": _r(p.dec[0]), "alt": _r(p.alt[0]), "az": _r(p.az[0]),
           "alt_apparent": _r(p.alt_apparent[0]), "hour_angle": _r(p.hour_angle[0]),
           "ra_j2000": _r(p.ra_astrometric[0]), "dec_j2000": _r(p.dec_astrometric[0]),
           "distance_km": _r(d, 1), "distance_au": _r(d / AU_KM, 7), "light_time_s": _r(p.light_time_s[0], 3),
           "range_rate_km_s": _r(p.range_rate_km_s[0], 4),
           "angular_diameter_arcsec": _r(2 * np.degrees(np.arcsin(RADIUS_KM[b] / d)) * 3600, 2),
           "magnitude": _r(np.ravel(magnitude(b, t, EPH))[0], 2)}
    if p.alt_apparent[0] > -0.5:
        X = float(airmass(p.alt_apparent[0]))
        out.update(airmass=_r(X, 3), extinction_mag=_r(float(extinction_mag(p.alt_apparent[0])), 3))
    if b != "sun":
        i, r, psi = phase_angle(b, t, EPH)
        out.update(phase_angle=_r(i[0], 3), illuminated=_r((1 + np.cos(np.radians(i[0]))) / 2, 4),
                   elongation=_r(psi[0], 3), sun_distance_au=_r(r[0] / AU_KM, 6))
    out["geocentric"] = {"ra": _r(g.ra[0]), "dec": _r(g.dec[0]), "distance_km": _r(g.distance_km[0], 1)}
    return out


@lru_cache(maxsize=64)
def _almanac(lat, lon, elev, day_jd):
    obs = Observer(lat, lon, elev)
    rts = {b: {k: [_iso(x) for x in v] for k, v in rise_transit_set(b, obs, day_jd, 1.0, EPH).items()}
           for b in BODIES}
    tw = twilight_events(obs, day_jd, 1.0, EPH)
    tw = {k: {kk: [_iso(x) for x in vv] for kk, vv in v.items()} for k, v in tw.items()}
    return {"window_start": _iso(day_jd), "rise_transit_set": rts, "twilight": tw}


@app.get("/api/sky")
def sky(lat: float = Query(-23.5505), lon: float = Query(-46.6333), elev: float = Query(0.0, ge=-500, le=9000),
        t: str | None = None, span_h: float = Query(6.0, gt=0, le=48), step_min: float = Query(5.0, ge=0.5, le=60)):
    """Everything the sky view needs: per-body samples (topocentric apparent alt/az) over +-span_h/2 for smooth
    animation, the state now, the GCRS -> true-of-date matrix and sidereal time for the stars, the Moon phase,
    rise/transit/set and twilight for the observer's day."""
    tt = _time(t)
    obs = _obs(lat, lon, elev)
    jd0 = float(tt.utc)
    n = int(round(span_h * 60 / step_min)) + 1
    jds = jd0 + (np.arange(n) - (n - 1) / 2) * step_min / 1440.0
    T = Time(jds)
    samples = {}
    for b in BODIES:
        p = apparent(b, T, obs)
        samples[b] = {"alt": np.round(p.alt_apparent, 4).tolist(), "az": np.round(p.az, 4).tolist(),
                      "ra": np.round(p.ra, 5).tolist(), "dec": np.round(p.dec, 5).tolist(),
                      "dist_km": np.round(p.distance_km, 1).tolist()}
    now = {b: _body_now(b, tt, obs) for b in BODIES}
    # local "day": from local noon before t to the next, using the longitude for the time zone
    local_noon = math.floor(jd0 + lon / 360.0 - 0.0) + 0.0 - lon / 360.0
    if local_noon > jd0:
        local_noon -= 1
    alm = _almanac(round(obs.lat, 3), round(obs.lon, 3), round(obs.elevation_m), round(local_noon, 6))
    return {
        "t": tt.iso(), "jd_utc": jd0, "jd_tt": float(tt.tt), "dut1": float(tt.dut1),
        "observer": {"lat": obs.lat, "lon": obs.lon, "elevation_m": obs.elevation_m,
                     "pressure_mbar": round(obs.pressure_mbar, 1), "temperature_c": obs.temperature_c},
        "gast_deg": float(np.degrees(gast_rad(tt))), "sidereal_rate_deg_per_day": 360.98564736629,
        "lst_hours": float((np.degrees(gast_rad(tt)) + obs.lon) % 360 / 15),
        "npb": frames.npb_matrix(float(tt.tt)).tolist(),
        "sky_state": sky_state(now["sun"]["alt"]), "sun_alt": now["sun"]["alt"],
        "samples": {"jd": jds.tolist(), "bodies": samples}, "now": now,
        "moon": moon_phase(tt, EPH), "almanac": alm,
        "method": "pinneapple_physics.astro: JPL DE421, light time, solar deflection, aberration, IAU 2006/1980 "
                  "precession-nutation, Saemundsson refraction",
    }


# ---------------------------------------------------------------------------------------------- solar system
@lru_cache(maxsize=32)
def _system(day_jd: float, days: int):
    from pinneapple_physics.astro.orbits import heliocentric_state, orbit_polyline, planet_elements
    jds = day_jd + np.arange(-days, days + 1, 1.0)
    out = {"jd": jds.tolist(), "planets": {}, "orbits": {}, "elements": {}}
    for b in SYSTEM:
        p, _ = heliocentric_state(b, jds, eph=EPH)
        out["planets"][b] = np.round(p / AU_KM, 6).tolist()
        out["orbits"][b] = np.round(orbit_polyline(b, day_jd, 360, EPH), 5).tolist()
        out["elements"][b] = {k: (round(v, 6) if isinstance(v, float) else v)
                              for k, v in planet_elements(b, day_jd, EPH).items()}
    # Moon, geocentric, every 3 h over the window (it moves 13 deg/day)
    jm = day_jd + np.arange(-days, days + 0.125, 0.125)
    pm, _ = EPH.relative("moon", "earth", jm)
    E = frames.ecliptic_matrix(None)
    out["moon"] = {"jd": jm.tolist(), "pos_km": np.round(pm @ E.T, 1).tolist()}
    out["elements"]["moon"] = {k: (round(v, 6) if isinstance(v, float) else v)
                               for k, v in planet_elements("moon", day_jd, EPH).items()}
    return out


@app.get("/api/system")
def system(t: str | None = None, days: int = Query(60, ge=5, le=400)):
    """Heliocentric J2000-ecliptic positions (AU) of the planets once a day over +-days, osculating orbits and
    elements, the geocentric Moon, and the Earth's orientation (GAST, obliquity) at t."""
    tt = _time(t)
    day = math.floor(float(tt.tdb) - 0.5) + 0.5
    d = _system(day, days)
    return {**d, "t": tt.iso(), "jd_tdb": float(tt.tdb), "jd_utc": float(tt.utc), "gast_deg": float(np.degrees(gast_rad(tt))),
            "obliquity_deg": 23.4392911, "npb": frames.npb_matrix(float(tt.tt)).tolist(),
            "ecliptic_j2000": frames.ecliptic_matrix(None).tolist()}


# ------------------------------------------------------------------------------------------------ satellites
_TLE = {"t": 0.0, "sats": [], "info": {}}
_TLE_LOCK = threading.Lock()


def _satellites():
    from pinneapple_physics.astro.satellites import fetch_tles, parse_tle
    with _TLE_LOCK:
        age = time.time() - _TLE["t"]
        if (_TLE["sats"] and age < 3600) or (not _TLE["sats"] and _TLE["t"] and age < 600):
            return _TLE["sats"], _TLE["info"]            # failures are retried every 10 min, not per request
        path = os.environ.get("OBS_TLE_FILE")
        if path:
            sats, info = parse_tle(open(path).read()), {"source": os.path.basename(path), "offline_file": True}
        else:
            sats, info, seen = [], {"groups": {}}, set()
            for g in (os.environ.get("OBS_TLE_GROUPS") or "stations,visual").split(","):
                s, i = fetch_tles(g.strip(), os.environ.get("OBS_CACHE_DIR"))
                info["groups"][g] = i
                for x in s:
                    if x.norad not in seen:
                        seen.add(x.norad)
                        sats.append(x)
        _TLE.update(t=time.time(), sats=sats, info=info)
        return sats, info


@app.get("/api/satellites")
def satellites(lat: float = Query(-23.5505), lon: float = Query(-46.6333), elev: float = 0.0,
               t: str | None = None, minutes: float = Query(20.0, ge=1, le=120), step_s: float = Query(10.0, ge=2),
               limit: int = Query(60, ge=1, le=300)):
    """Look angles of the loaded satellites over +-minutes/2 around t (for animation), with sunlit flags and
    magnitudes. Satellites with stale TLEs (> 30 days from t) are dropped."""
    from pinneapple_physics.astro.satellites import look
    tt = _time(t)
    obs = _obs(lat, lon, elev)
    sats, info = _satellites()
    jd0 = float(tt.utc)
    n = int(minutes * 60 / step_s) + 1
    jds = jd0 + (np.arange(n) - (n - 1) / 2) * step_s / 86400.0
    T = Time(jds)
    out = []
    for s in sats[:limit * 3]:
        if abs(jd0 - s.epoch_jd) > 30:
            continue
        lk = look(s, obs, T, EPH)
        if np.any(lk["error"] != 0):
            continue
        out.append({"name": s.name, "norad": s.norad, "tle_age_days": round(jd0 - s.epoch_jd, 2),
                    "alt": np.round(lk["alt"], 3).tolist(), "az": np.round(lk["az"], 3).tolist(),
                    "range_km": np.round(lk["range_km"], 1).tolist(),
                    "sunlit": np.round(lk["sunlit"], 2).tolist(),
                    "mag": [None if not np.isfinite(m) else round(float(m), 2) for m in lk["magnitude"]],
                    "lat": np.round(lk["lat"], 3).tolist(), "lon": np.round(lk["lon"], 3).tolist(),
                    "height_km": np.round(lk["height_km"], 1).tolist()})
        if len(out) >= limit:
            break
    return {"t": tt.iso(), "jd": jds.tolist(), "satellites": out, "source": info,
            "count_loaded": len(sats),
            "note": None if sats else "No TLEs available (CelesTrak unreachable and no cache)."}


@app.get("/api/passes")
def passes(lat: float = Query(-23.5505), lon: float = Query(-46.6333), elev: float = 0.0, t: str | None = None,
           days: float = Query(3.0, gt=0, le=10), norad: str = Query("25544,48274,20580"),
           visible_only: bool = False):
    """Next passes (rise / culmination / set, max altitude, visibility, brightest magnitude) of selected
    satellites; defaults: ISS, Tiangong (CSS), Hubble."""
    from pinneapple_physics.astro.satellites import find_passes
    tt = _time(t)
    obs = _obs(lat, lon, elev)
    sats, info = _satellites()
    want = {int(x) for x in norad.split(",") if x.strip().isdigit()}
    out = []
    with _HEAVY:
        for s in sats:
            if s.norad in want and abs(float(tt.utc) - s.epoch_jd) <= 30:
                out += find_passes(s, obs, tt, days, eph=EPH)
    out.sort(key=lambda p: p["culmination_jd"])
    if visible_only:
        out = [p for p in out if p["visible"]]
    return {"t": tt.iso(), "passes": out, "source": info,
            "method": "SGP4 (sgp4 package, Vallado 2006) -> TEME -> ITRS (GMST82); conical Earth shadow; "
                      "diffuse-sphere magnitude from the standard magnitude at 1000 km, 90 deg phase"}


# -------------------------------------------------------------------------------------------------- analyses
def _target_vec(name: str, t: Time, obs: Observer | None):
    """(ra, dec, alt, az) of a body name or 'HR <n>' star at t."""
    from pinneapple_physics.astro.astrometry import star_apparent
    from pinneapple_physics.astro.stars import load_bright_stars
    key = name.strip().lower()
    if key in BODIES:
        p = apparent(key, t, obs)
        return (float(p.ra[0]), float(p.dec[0]), None if obs is None else float(p.alt_apparent[0]),
                None if obs is None else float(p.az[0]))
    if key.startswith("hr"):
        c = load_bright_stars()
        i = c.index_of(int(key[2:].strip()))
        r = star_apparent(c.ra[i:i + 1], c.dec[i:i + 1], t, obs, pm_ra_mas=c.pmra[i:i + 1],
                          pm_dec_mas=c.pmdec[i:i + 1], parallax_mas=np.nan_to_num(c.plx[i:i + 1]))
        return (float(r[0][0]), float(r[1][0]), None if obs is None else float(r[4][0]),
                None if obs is None else float(r[3][0]))
    raise _error(404, "unknown_target", f"unknown target {name!r} (use a body name or 'HR <number>')", value=name)


@app.get("/api/analysis/separation")
def separation(a: str, b: str, t: str | None = None, lat: float = -23.5505, lon: float = -46.6333):
    tt = _time(t)
    obs = _obs(lat, lon, 0)
    A, B = _target_vec(a, tt, obs), _target_vec(b, tt, obs)
    sep = float(frames.angular_separation(A[0], A[1], B[0], B[1]))
    pa = float(frames.position_angle(A[0], A[1], B[0], B[1]))
    return {"a": a, "b": b, "separation_deg": sep, "separation_dms": _dms(sep), "position_angle_deg": pa,
            "a_radec": A[:2], "b_radec": B[:2],
            "method": "Vincenty great-circle formula on apparent places of date (topocentric)"}


def _dms(d):
    s = abs(d) * 3600
    return f"{int(s // 3600)}° {int(s % 3600 // 60):02d}′ {s % 60:04.1f}″"


@app.get("/api/analysis/target")
def target(name: str = "jupiter", lat: float = -23.5505, lon: float = -46.6333, elev: float = 0.0,
           t: str | None = None, min_alt: float = 30.0):
    """Altitude / airmass curve over the next 24 h, the dark-sky window (Sun < -18 deg) and the best observing
    window tonight: when the target is highest while the sky is dark, with the Moon's separation and altitude."""
    tt = _time(t)
    obs = _obs(lat, lon, elev)
    jd0 = float(tt.utc)
    jds = jd0 + np.arange(0, 24 * 60 + 1, 10) / 1440.0
    T = Time(jds)
    key = name.strip().lower()
    if key in BODIES:
        p = apparent(key, T, obs)
        alt, ra, dec = p.alt_apparent, p.ra, p.dec
    else:
        res = [_target_vec(name, Time(j), obs) for j in jds]
        ra, dec, alt = (np.array([r[k] for r in res]) for k in range(3))
    sun = apparent("sun", T, obs).alt
    moon = apparent("moon", T, obs)
    msep = frames.angular_separation(ra, dec, moon.ra, moon.dec)
    X = airmass(alt)
    dark = sun < -18
    ok = dark & (alt >= min_alt)
    win = None
    if ok.any():
        idx = np.nonzero(ok)[0]
        runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
        best = max(runs, key=lambda r: len(r))
        k = best[np.argmax(alt[best])]
        win = {"start": _iso(jds[best[0]]), "end": _iso(jds[best[-1]]), "best": _iso(jds[k]),
               "max_alt": float(alt[k]), "airmass": float(X[k]), "moon_separation": float(msep[k]),
               "moon_alt": float(moon.alt_apparent[k]), "hours": len(best) / 6.0}
    return {"name": name, "t": tt.iso(), "jd": jds.tolist(), "times": [Time(j).iso() for j in jds[::6]],
            "alt": np.round(alt, 3).tolist(), "airmass": [None if not np.isfinite(x) or x > 40 else round(float(x), 3)
                                                        for x in X],
            "sun_alt": np.round(sun, 3).tolist(), "moon_alt": np.round(moon.alt_apparent, 3).tolist(),
            "moon_separation": np.round(msep, 3).tolist(), "dark": dark.tolist(), "best_window": win,
            "min_alt": min_alt}


@app.get("/api/analysis/orbit")
def orbit(body: str = "mars", t: str | None = None, lat: float = -23.5505, lon: float = -46.6333,
          line_nm: float = Query(656.281, gt=50, lt=5000)):
    """Osculating Keplerian elements, distance, light time and the Doppler shift of a spectral line."""
    from pinneapple_physics.astro.orbits import light_time_and_doppler, planet_elements
    b = body.lower()
    if b not in BODIES or b == "sun":
        raise _error(404, "not_moon_or_planet", "body must be the Moon or a planet")
    tt = _time(t)
    el = planet_elements(b, float(tt.tdb), EPH)
    p = apparent(b, tt, _obs(lat, lon, 0))
    dop = light_time_and_doppler(p.distance_km[0], p.range_rate_km_s[0], line_nm)
    return {"body": b, "t": tt.iso(), "elements": {k: (float(v) if not isinstance(v, str) else v)
                                                   for k, v in el.items()},
            "distance_km": float(p.distance_km[0]), "distance_au": float(p.distance_km[0] / AU_KM),
            "light_time_s": float(dop["light_time_s"]), "range_rate_km_s": float(p.range_rate_km_s[0]),
            "doppler": {"rest_nm": line_nm, "observed_nm": float(dop["observed_nm"]),
                        "shift_pm": float(dop["shift_nm"] * 1000), "beta": float(dop["beta"])},
            "method": "elements from the DE421 heliocentric state (J2000 ecliptic); relativistic Doppler from the "
                      "topocentric range rate"}


@app.get("/api/analysis/star")
def star(hr: int, t: str | None = None, lat: float = -23.5505, lon: float = -46.6333):
    from pinneapple_physics.astro.stars import load_bright_stars, star_properties
    c = load_bright_stars()
    try:
        i = c.index_of(hr)
    except KeyError as exc:
        raise _error(404, "unknown_star", str(exc), hr=hr) from exc
    tt = _time(t)
    ra, dec, alt, az = _target_vec(f"HR {hr}", tt, _obs(lat, lon, 0))
    props = star_properties(c, i)
    return {**props, "ra_date": ra, "dec_date": dec, "alt": alt, "az": az,
            "method": "distance from the BSC5 parallax; Teff from B-V (Ballesteros 2012); BC_V from a black-body "
                      "V-band integral zero-pointed on the Sun; L and R from Stefan-Boltzmann"}


@lru_cache(maxsize=1)
def _hr_points():
    from pinneapple_physics.astro.stars import hr_diagram, load_bright_stars
    d = hr_diagram(load_bright_stars())
    return {"hr": d["hr"].tolist(), "bv": np.round(d["bv"], 3).tolist(), "abs_mag": np.round(d["abs_mag"], 3).tolist()}


@app.get("/api/analysis/hr")
def hr_diagram():
    return JSONResponse({**_hr_points(), "note": "stars with a BSC5 trigonometric parallax >= 5 mas"},
                        headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/analysis/blackbody")
def blackbody(T: float = Query(5772.0, ge=500, le=60000)):
    from pinneapple_physics.astro.stars import blackbody_spectrum
    s = blackbody_spectrum(T)
    return {"T": T, "wavelength_nm": np.round(s["wavelength_nm"], 2).tolist(),
            "radiance": s["radiance"].tolist(), "peak_nm": s["peak_nm"], "flux_w_m2": s["flux_w_m2"], "rgb": s["rgb"]}


@lru_cache(maxsize=16)
def _events(day_jd: float, days: int, lat: float, lon: float):
    obs = Observer(lat, lon, 0.0)
    jd1 = day_jd + days
    ev = []
    for e in lunar_phase_events(day_jd, jd1, EPH):
        ev.append({**e, "kind": "moon"})
    for e in seasons(day_jd, jd1, EPH):
        ev.append({**e, "kind": "season"})
    for b in PLANETS:
        for e in planet_events(b, day_jd, jd1, EPH):
            ev.append({**e, "kind": "planet", "body": b, "type": e["event"], "event": f"{b.capitalize()} {e['event']}"})
    for e in lunar_eclipses(day_jd, jd1, EPH):
        ev.append({**e, "kind": "eclipse"})
    for e in solar_eclipses(day_jd, jd1, obs, EPH):
        ev.append({**e, "kind": "eclipse"})
    for e in planetary_transits(day_jd, jd1, EPH):
        ev.append({**e, "kind": "transit"})
    for e in ev:
        e.pop("jd", None)
        if "local" in e:
            e["local"].pop("jd", None)
    return sorted(ev, key=lambda e: e["utc"])


@app.get("/api/analysis/events")
def events(t: str | None = None, days: int = Query(365, ge=7, le=800), lat: float = -23.5505, lon: float = -46.6333):
    """Moon phases, equinoxes / solstices, oppositions, conjunctions and greatest elongations, lunar and solar
    eclipses (with local circumstances), transits of Mercury and Venus."""
    tt = _time(t)
    with _HEAVY:
        ev = _events(math.floor(float(tt.utc) - 0.5) + 0.5, days, round(lat, 1), round(lon, 1))
    return {"t": tt.iso(), "days": days, "events": ev,
            "method": "root finding on apparent geocentric ecliptic longitudes; eclipse geometry with the Danjon "
                      "shadow enlargement; local solar-eclipse circumstances from topocentric separations"}


# ----------------------------------------------------------------------------------------- lensing / black hole
_SKY = {}


@app.get("/api/analysis/lensing.png")
def lensing_png(ra: float = Query(266.4, ge=0, le=360), dec: float = Query(-29.0, ge=-90, le=90),
                fov: float = Query(60.0, ge=5, le=120), r_cam: float = Query(120.0, ge=6, le=500),
                width: int = Query(480, ge=64, le=960), disc: str = Query("auto", pattern="^(auto|on|off)$")):
    from PIL import Image

    from pinneapple_physics.astro.lensing import catalog_sky, lensing_view
    height = int(width * 9 / 16) // 2 * 2
    with _HEAVY:
        if "sky" not in _SKY:
            _SKY["sky"] = catalog_sky()
        L = lensing_view(ra, dec, fov, width, height, r_cam, sky=_SKY["sky"],
                         disc={"auto": "auto", "on": True, "off": False}[disc])
    buf = io.BytesIO()
    Image.fromarray((L["image"] * 255).astype("uint8")).save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png",
                    headers={"X-Shadow-Radius-Deg": f"{L['shadow_radius_deg']:.4f}", "X-Disc": str(int(L["disc"])),
                             "X-Bright-Stars": str(L["bright_stars"]), "X-Bright-Mag": f"{L['bright_mag']:.1f}",
                             "Cache-Control": "max-age=3600"})


@app.get("/api/analysis/lensing")
def lensing_info(elongation: float = Query(0.2666, gt=0.2, le=180)):
    from pinneapple_physics.astro.lensing import eht_comparison, sun_deflection_arcsec
    return {"eht": eht_comparison(), "sun_deflection_arcsec": float(sun_deflection_arcsec(elongation)),
            "elongation_deg": elongation,
            "method": "Schwarzschild null geodesics (RK4 on u'' = 3u^2 - u) from pinneapple_physics.blackhole; shadow "
                      "diameter 2 sqrt(27) GM/(c^2 D); weak-field deflection 4GM/(c^2 b)"}


# ---------------------------------------------------------------------------------------------------- engine
class _Bounded(collections.OrderedDict):
    """Result cache keyed on request values: the oldest entries go past ``limit`` (memory stays bounded)."""

    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self.limit:
            self.popitem(last=False)


_CACHE: dict = _Bounded(int(os.environ.get("OBS_CACHE_ENTRIES") or 256))
_CACHE_LOCK = threading.Lock()


def _surrogate(body: str, day_jd: float, days: int):
    from pinneapple_physics.astro.surrogate import train_orbit_pinn
    key = ("pinn", body, day_jd, days)
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]
    rep = train_orbit_pinn(body, day_jd, float(days), iters=800, eph=EPH)
    with _CACHE_LOCK:
        _CACHE[key] = rep
    return rep


@app.get("/api/engine/surrogate")
def engine_surrogate(body: str = "mars", t: str | None = None, days: int = Query(365, ge=30, le=730)):
    """Train (or fetch from cache) the physics-informed orbit surrogate and its trust report."""
    if body not in ("mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"):
        raise _error(404, "not_planet", "body must be a planet")
    tt = _time(t)
    day = math.floor(float(tt.tdb) - 0.5) + 0.5
    with _HEAVY:
        return _surrogate(body, day, days)


@app.get("/api/engine/integrators")
def engine_integrators(years: float = Query(2.0, gt=0, le=20), dt_hours: float = Query(6.0, ge=0.5, le=48)):
    """Leapfrog, Yoshida-4 and RK4 on the Sun-planets-Moon problem from DE421 initial conditions: energy and
    angular-momentum conservation, and the position difference to DE421."""
    from pinneapple_physics.astro.nbody import SOLAR_SYSTEM, initial_state, integrate
    key = ("int", years, dt_hours)
    if key in _CACHE:
        return _CACHE[key]
    jd0 = 2451545.0
    r0, v0, mu = initial_state(jd0)
    out = {"years": years, "dt_hours": dt_hours, "methods": {}}
    with _HEAVY:
        for m in ("leapfrog", "yoshida4", "rk4"):
            t0 = time.time()
            res = integrate(r0, v0, mu, years * 365.25 * 86400, dt_hours * 3600, m, n_out=60)
            jd = jd0 + res["t_s"] / 86400
            errs = {}
            for b in ("earth", "mars", "jupiter"):
                k = SOLAR_SYSTEM.index(b)
                p, _ = EPH.barycentric(b, jd)
                errs[b] = np.linalg.norm(res["r"][:, k] - p, axis=1).round(2).tolist()
            out["methods"][m] = {"rel_energy_error": res["rel_energy_error"].tolist(),
                                 "rel_angmom_error": res["rel_angmom_error"].tolist(), "error_vs_de421_km": errs,
                                 "force_evals": res["n_force_evals"], "seconds": time.time() - t0}
        out["t_days"] = (res["t_s"] / 86400).tolist()
    _CACHE[key] = out
    return out


# ------------------------------------------------------------------------------- orbit decay (satellites panel)
@app.get("/api/sat/decay")
def sat_decay(norad: int = 25544, days: float = Query(60.0, ge=5, le=365), f107: float = Query(150.0, ge=60, le=300),
              ap: float = Query(15.0, ge=0, le=400), B: float = Query(0.005, gt=0, le=0.2),
              h0: float | None = Query(None, ge=180, le=1000), incl: float = Query(51.6, ge=0, le=180),
              t: str | None = None):
    """Drag decay of a LEO satellite: the PINN, the RK reference with its input-uncertainty envelope, and, when a
    TLE is loaded, SGP4's own decay prediction from that TLE (an independent drag model)."""
    from pinneapple_physics.astro.decay import compare_with_sgp4, sgp4_decay, train_decay_pinn
    tt = _time(t)
    sats, _info = _satellites()
    sat = next((x for x in sats if x.norad == norad), None)
    sgp = None
    date = tt.iso()[:16]
    if sat is not None and h0 is None:
        ts, hs, info = sgp4_decay(sat.line1, sat.line2, days=1.0)
        h0, incl = float(hs[0]), info["inclination_deg"]
    h0 = 415.0 if h0 is None else h0
    key = ("decay", norad, round(h0, 1), days, f107, ap, B, round(incl, 1), date[:10])
    if key not in _CACHE:
        with _HEAVY:
            rep = train_decay_pinn(h0, B, days, f107=f107, ap=ap, date=date, inclination_deg=incl, iters=1200)
            if sat is not None:
                sgp = compare_with_sgp4(sat.line1, sat.line2, min(days, 30.0), f107=f107, ap=ap, B=B)
            rep["sgp4"] = sgp
            rep["satellite"] = None if sat is None else {"name": sat.name, "norad": sat.norad,
                                                         "tle_epoch_jd": sat.epoch_jd}
            rep["method"] = ("da/dt = -B rho(h) sqrt(mu a); rho: orbit-averaged NRLMSIS 2.0; PINN a(t) = a0 - s tau "
                             "softplus(g(tau)) trained on the ODE residual; reference: DOP853 integration; envelope: "
                             "Monte Carlo over F10.7 (+-20 %), Ap (x0.5-2) and B (+-15 %)")
            _CACHE[key] = rep
    return _CACHE[key]


# ------------------------------------------------------------------------------------------------- small bodies
@app.get("/api/smallbodies")
def smallbodies():
    from pinneapple_physics.astro.smallbodies import available_bodies, load_body
    out = []
    for k in available_bodies():
        b = load_body(k)
        refs = [c for c in b.ca_data if c["body"] == "Earth" and 1950 <= int(c["cd"][:4]) <= 2049]
        out.append({"key": k, "name": b.name, "a_au": b.a_au, "e": b.e, "i_deg": b.i_deg, "epoch_jd": b.epoch_jd,
                    "neo": b.neo, "pha": b.pha, "H": b.H, "diameter_km": b.diameter_km,
                    "close_approaches_jpl": [{"cd": c["cd"], "dist_au": float(c["dist"]),
                                              "v_rel_km_s": float(c["v_rel"])} for c in refs[:60]]})
    return {"bodies": out, "source": "JPL Small-Body Database (public domain), as bundled in the astroquery tests"}


@app.get("/api/smallbodies/track")
def smallbody_track(body: str = "apophis", t: str | None = None, days: int = Query(400, ge=30, le=1500)):
    """Heliocentric J2000-ecliptic track (AU) of a small body around t from the full propagator, and the learned
    propagator's track over its one-year validity window."""
    from pinneapple_physics.astro.smallbodies import Trajectory, load_body
    tt = _time(t)
    key = ("sbtrack", body, round(float(tt.tdb)), days)
    if key not in _CACHE:
        b = load_body(body)
        jd = float(tt.tdb) + np.linspace(-days / 2, days / 2, 241)
        with _HEAVY:
            tr = Trajectory(b, float(jd.min()), float(jd.max()), EPH)
            r, _ = tr.heliocentric(jd)
        E = frames.ecliptic_matrix(None)
        _CACHE[key] = {"body": body, "name": b.name, "jd": jd.tolist(), "pos_au": np.round(r @ E.T / AU_KM, 6).tolist()}
    return _CACHE[key]


@app.get("/api/smallbodies/approaches")
def smallbody_approaches(body: str = "apophis", start: str = "2026-01-01T00:00:00Z",
                         years: float = Query(10.0, gt=0, le=24), max_au: float = Query(0.2, gt=0, le=1)):
    from pinneapple_physics.astro.smallbodies import close_approaches, load_body
    t0 = _time(start)
    jd0 = float(t0.tdb)
    jd1 = min(jd0 + years * 365.25, EPH.jd_range[1] - 2)
    key = ("sbca", body, round(jd0), round(jd1), max_au)
    if key not in _CACHE:
        b = load_body(body)
        with _HEAVY:
            ca = close_approaches(b, jd0, jd1, max_dist_au=max_au, step_days=1.0, eph=EPH)
        refs = {c["cd"][:11]: c for c in b.ca_data if c["body"] == "Earth"}
        for c in ca:
            c["utc"] = Time.from_tt(c["jd_tdb"]).iso()
            c["lunar_distances"] = c["dist_km"] / 384400.0
            from pinneapple_physics.astro.timescales import datetime_from_jd
            d = datetime_from_jd(c["jd_tdb"]).strftime("%Y-%b-%d")
            ref = refs.get(d)
            if ref:
                c["jpl"] = {"cd": ref["cd"], "dist_au": float(ref["dist"]), "dist_min_au": float(ref["dist_min"]),
                            "dist_max_au": float(ref["dist_max"])}
        _CACHE[key] = {"body": body, "approaches": ca,
                       "method": "DOP853 test-particle integration in the DE421 field of the Sun, planets and Moon, "
                                 "with the Sun's 1PN relativistic term; no Yarkovsky or asteroid perturbers"}
    return _CACHE[key]


@app.get("/api/smallbodies/validation")
def smallbody_validation():
    from pinneapple_physics.astro import neural_propagator as npg
    from pinneapple_physics.astro.smallbodies import validation_summary
    if "sbval" not in _CACHE:
        with _HEAVY:
            val = validation_summary(EPH)
        try:
            val["neural"] = npg.trust_report()
        except OSError:
            val["neural"] = None
        _CACHE["sbval"] = val
    return _CACHE["sbval"]


@app.get("/api/smallbodies/pinn")
def smallbody_pinn(body: str = "apophis", t: str | None = None, days: int = Query(365, ge=30, le=730)):
    """The physics-informed orbit surrogate (Kepler + tau^2 network, N-body residual) trained for one small body,
    judged against the high-accuracy propagator."""
    from pinneapple_physics.astro.smallbodies import load_body
    from pinneapple_physics.astro.surrogate import train_orbit_pinn
    tt = _time(t)
    day = math.floor(float(tt.tdb) - 0.5) + 0.5
    key = ("sbpinn", body, day, days)
    if key not in _CACHE:
        with _HEAVY:
            _CACHE[key] = train_orbit_pinn(load_body(body), day, float(days), iters=800, eph=EPH)
    return _CACHE[key]


@app.get("/api/smallbodies/neural")
def smallbody_neural(body: str = "apophis"):
    """The learned propagator on a real object over one year from its SBDB epoch, against the full integrator."""
    from pinneapple_physics.astro import neural_propagator as npg
    from pinneapple_physics.astro.smallbodies import load_body
    key = ("sbnn", body)
    if key not in _CACHE:
        model = npg.NeuralPropagator.load()
        with _HEAVY:
            row = npg.evaluate(model, [load_body(body)])[0]
        _CACHE[key] = row
    return _CACHE[key]


# ----------------------------------------------------------------------------------------- stars ML / weather
@app.get("/api/stars/ml")
def stars_ml():
    from pinneapple_physics.astro.stars_ml import analyse
    if "starsml" not in _CACHE:
        _CACHE["starsml"] = analyse()
    return JSONResponse(_CACHE["starsml"], headers={"Cache-Control": "public, max-age=3600"})


_SW = {}


def _sw_models():
    from pinneapple_physics.astro.spaceweather import train_forecaster
    if not _SW:
        for h in (1, 8):
            _SW[h] = train_forecaster(h)
    return _SW


@app.get("/api/spaceweather")
def spaceweather(lat: float = Query(-23.5505), lon: float = Query(-46.6333)):
    """Kp now and forecast 3 h / 24 h ahead with calibrated intervals, and the aurora probability here."""
    from pinneapple_physics.astro import spaceweather as sw
    with _HEAVY:
        models = _sw_models()
    live, source = None, None
    try:
        times, kp = sw.fetch_live_kp()
        live = {"times": times[-56:], "kp": kp[-56:].tolist()}
        source = "NOAA SWPC planetary K index (live)"
        hist = kp
    except Exception as exc:  # noqa: BLE001
        h = sw.load_history()
        hist = h["kp"][: 330 * 8]                                   # replay: 2000-Nov-25, a stormy autumn
        source = f"offline replay of observed Kp (GFZ), 2000-Nov-25 -- live NOAA SWPC unavailable ({type(exc).__name__})"
    maglat, plat, plon = sw.geomagnetic_latitude(lat, lon)
    out = {"source": source, "live": live, "kp_now": float(hist[-1]), "maglat": maglat,
           "pole": [plat, plon], "boundary_now": float(sw.oval_boundary(hist[-1])), "forecasts": []}
    if len(hist) >= sw.RECUR + 8:
        for hzn, (model, rep) in models.items():
            lo, med, hi = sw.forecast(model, hist, 150.0)
            out["forecasts"].append({"horizon_h": hzn * 3, "lo": lo, "median": med, "hi": hi,
                                     "aurora": sw.aurora_probability(lo, med, hi, maglat),
                                     "verdict": rep["verdict"], "checks": rep["checks"],
                                     "mae_model": rep["mae_model"], "mae_persistence": rep["mae_persistence"],
                                     "coverage_80": rep["coverage_80"], "median_source": rep["median_source"]})
    out["method"] = models[8][1]["method"] + "; aurora boundary 66.5 - 2.04 Kp (geomagnetic latitude, IGRF-13 dipole)"
    return out


@app.get("/api/spaceweather/verification")
def spaceweather_verification():
    with _HEAVY:
        models = _sw_models()
    return {str(h * 3): {k: v for k, v in rep.items()} for h, (m, rep) in models.items()}


@app.get("/api/conditions")
def conditions(lat: float = Query(-23.5505), lon: float = Query(-46.6333)):
    """Cloud / transparency / seeing proxy for the next 48 h from the Open-Meteo ensemble (members' spread)."""
    from pinneapple_physics.astro.conditions import fetch_ensemble, summarize
    key = ("wx", round(lat, 2), round(lon, 2), int(time.time() // 3600))
    if key not in _CACHE:
        try:
            _CACHE[key] = summarize(fetch_ensemble(lat, lon))
        except Exception as exc:  # noqa: BLE001
            return {"error": f"Open-Meteo unavailable from the server ({type(exc).__name__}: {exc})"}
    return _CACHE[key]


@app.get("/api/blackhole/status")
def blackhole_status():
    """Show the black-hole accretion forecaster only if it is in the model zoo and passes the zoo gate."""
    try:
        import pinneapple_hub.zoo as zoo
        names = [m if isinstance(m, str) else getattr(m, "name", str(m)) for m in zoo.list()]
        cand = [n for n in names if any(k in n.lower() for k in ("black", "accretion", "riaf"))]
        for n in cand:
            st = zoo.status(n)
            if str(st.get("gate", st.get("status", ""))).lower() in ("pass", "passed", "trustworthy", "ok"):
                return {"available": True, "model": n, "status": st}
        return {"available": False, "candidates": cand,
                "note": "The black-hole accretion forecaster is not in the model zoo yet (or has not passed its "
                        "gate); its forecast and skill horizon will appear here when it does."}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "note": f"model zoo unavailable: {exc}"}


@app.get("/api/analysis/accretion.png")
def accretion_png(ra: float = Query(266.4, ge=0, le=360), dec: float = Query(-29.0, ge=-90, le=90),
                  doppler: bool = True, incl: float = Query(86.0, ge=10, le=89.5)):
    from PIL import Image

    from pinneapple_physics.astro.lensing import accretion_view
    with _HEAVY:
        if "sky_disc" not in _SKY:
            from pinneapple_physics.astro.lensing import catalog_sky
            _SKY["sky_disc"] = catalog_sky(2048, 1024)
        r = accretion_view(ra, dec, 640, 360, inclination_deg=incl, doppler=doppler, sky=_SKY["sky_disc"])
    buf = io.BytesIO()
    Image.fromarray((np.clip(r["image"], 0, 1) * 255).astype("uint8")).save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "max-age=3600"})


_VALIDATION = {"rows": None, "running": False, "seconds": None}


def _run_validation():
    from pinneapple_physics.astro import validation
    _VALIDATION["running"] = True
    t0 = time.time()
    try:
        _VALIDATION["rows"] = validation.run_all()
    except Exception as exc:  # noqa: BLE001
        _VALIDATION["error"] = str(exc)
    _VALIDATION["running"] = False
    _VALIDATION["seconds"] = time.time() - t0


@app.get("/api/validation")
def validation_table():
    """Accuracy of every method against independent references (computed once at start-up)."""
    if _VALIDATION["rows"] is None and not _VALIDATION["running"]:
        threading.Thread(target=_run_validation, daemon=True).start()
    return {"rows": _VALIDATION["rows"], "running": _VALIDATION["running"], "seconds": _VALIDATION["seconds"],
            "error": _VALIDATION.get("error")}


@app.on_event("startup")
def _warm():
    if os.environ.get("OBS_PRECOMPUTE", "1") == "0":
        return

    def go():
        _run_validation()
        try:
            tt = Time.now()
            _surrogate("mars", math.floor(float(tt.tdb) - 0.5) + 0.5, 365)
        except Exception:  # noqa: BLE001
            pass
    threading.Thread(target=go, daemon=True).start()
