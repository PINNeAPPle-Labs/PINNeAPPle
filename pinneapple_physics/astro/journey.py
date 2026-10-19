"""A journey from your place on Earth to a planet, a moon, the Sun or a star: the geometry of the trip.

Positions are heliocentric, in km, on ICRS (J2000 equatorial) axes: Solar System bodies from the JPL ephemeris at
the date of the trip, the observer from the Earth's position plus the geocentric position of the site (WGS84, Earth
rotation), stars from their J2000 direction and parallax. The module gives the distances, the light time, how long the
trip would take at real speeds, and how the target grows and brightens as you approach (apparent magnitude
m = M + 5 log10(d / 10 pc) for stars, the inverse-square law at a fixed phase for planets; angular diameter
2 asin(R / d)). Galactic coordinates use the IAU 1958 definition on J2000 axes (Hipparcos introduction, vol. 1, §1.5.3).

    from pinneapple_physics.astro import journey, Observer, Time
    j = journey.plan(Observer(-23.55, -46.63, 760), "mars", Time.from_iso("2026-10-14T00:00:00Z"))
    j["distance_km"], j["light_time_s"], j["travel_times"]
"""
from __future__ import annotations

import math

import numpy as np

from .astrometry import Observer, apparent
from .ephemeris import AU_KM, BODIES, C_KM_S, RADIUS_KM, default_ephemeris
from .stars import M_V_SUN  # absolute visual magnitude of the Sun, 4.81 (Willmer 2018)
from .timescales import Time

PC_KM = 3.0856775814913673e13
LY_KM = 9.4607304725808e12
# J2000 equatorial -> galactic (Hipparcos and Tycho Catalogues, vol. 1, eq. 1.5.11)
GALACTIC = np.array([[-0.0548755604, -0.8734370902, -0.4838350155],
                     [+0.4941094279, -0.4448296300, +0.7469822445],
                     [-0.8676661490, -0.1980763734, +0.4559837762]])

# Speeds for "how long would it take": km/s
SPEEDS = {
    "walking": 5.0 / 3600.0,                 # 5 km/h
    "airliner": 900.0 / 3600.0,              # 900 km/h cruise
    "new_horizons": 16.26,                   # fastest launch speed relative to the Earth (NASA, 2006)
    "voyager1": 17.0,                        # speed relative to the Sun today (NASA JPL Voyager mission status)
    "light": C_KM_S,
}
# Distances marked along the way: km
MARKERS = {
    "moon": 384400.0,                         # mean Earth-Moon distance (NASA Moon fact sheet)
    "au": AU_KM,
    "heliopause": 121.6 * AU_KM,             # Voyager 1 crossed it in August 2012 at 121.6 au (NASA JPL)
    "light_year": LY_KM,
    "nearest_star": 1.3020 * PC_KM,          # Proxima Centauri, Gaia DR3 parallax 768.07 mas
}


def _unit(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def galactic(v):
    """Galactic (l, b) in degrees of an ICRS (J2000) vector."""
    g = GALACTIC @ np.asarray(v, float)
    return float(np.degrees(np.arctan2(g[1], g[0])) % 360.0), float(np.degrees(np.arcsin(g[2] / np.linalg.norm(g))))


def body_position(body: str, t: Time, eph=None) -> np.ndarray:
    """Heliocentric ICRS position (km) of a Solar System body at t (geometric, no light time)."""
    eph = eph or default_ephemeris()
    jd = np.atleast_1d(t.tdb)
    p, _ = eph.barycentric(body, jd)
    s, _ = eph.barycentric("sun", jd)
    return (p - s)[0]


def observer_position(observer: Observer, t: Time, eph=None) -> np.ndarray:
    """Heliocentric ICRS position (km) of a place on Earth at t (Earth from the ephemeris + the site in GCRS)."""
    o, _ = observer.gcrs(Time(np.atleast_1d(t.utc), np.atleast_1d(t.dut1)))
    return body_position("earth", t, eph) + o[0]


def star_position(ra_deg: float, dec_deg: float, parallax_mas: float) -> np.ndarray:
    """Heliocentric ICRS position (km) of a star from its J2000 direction and parallax."""
    return _unit(ra_deg, dec_deg) * (1000.0 / parallax_mas) * PC_KM


def travel_times(distance_km: float) -> dict:
    """Seconds needed to cover a distance at the speeds in ``SPEEDS``."""
    return {k: distance_km / v for k, v in SPEEDS.items()}


def apparent_magnitude_star(abs_mag: float, distance_km):
    """m = M + 5 log10(d / 10 pc)."""
    return abs_mag + 5.0 * np.log10(np.asarray(distance_km, float) / (10.0 * PC_KM))


def angular_diameter_deg(radius_km: float, distance_km):
    d = np.asarray(distance_km, float)
    return np.degrees(2.0 * np.arcsin(np.clip(radius_km / np.maximum(d, radius_km), 0.0, 1.0)))


def _target(target, t, eph):
    """(position km, radius km, absolute magnitude or None, kind, extra) of a body name or a star (hr / facts id)."""
    if isinstance(target, str) and target.lower() in BODIES and target.lower() != "earth":
        b = target.lower()
        return body_position(b, t, eph), RADIUS_KM[b], None, "body", {"body": b}
    from . import facts
    from .stars import load_bright_stars, teff_from_bv
    oid = facts.resolve(target)
    if not oid.startswith("star-"):
        raise ValueError(f"no journey to {target!r}: choose the Sun, the Moon, a planet or a star")
    hr = int(oid[5:])
    c = load_bright_stars()
    i = c.index_of(hr)
    plx = facts._stars()[str(hr)]["parallax_mas"]["value"]
    pos = star_position(c.ra[i], c.dec[i], plx)
    d_pc = 1000.0 / plx
    M = float(c.v[i] - 5.0 * math.log10(d_pc / 10.0))
    num = facts.numbers(oid)
    R = num.get("radius_rsun", {}).get("value", 1.0) * RADIUS_KM["sun"]
    T = float(teff_from_bv(c.bv[i])) if not np.isnan(c.bv[i]) else None
    return pos, R, M, "star", {"hr": hr, "parallax_mas": plx, "v_mag": float(c.v[i]), "teff_k": T, "id": oid}


def plan(observer: Observer, target, t: Time, standoff: float | None = None, n_profile: int = 60, eph=None) -> dict:
    """The trip from ``observer`` to ``target`` (body name, 'star:<HR>', 'HR <n>' or a facts id) on date ``t``.

    Returns the start and end positions (heliocentric ICRS, km), the distance (km, au, light-years), the one-way
    light time, the travel times at real speeds, the arrival point (``standoff`` x the target's radius short of its
    centre; 4 radii for a body, 30 for a star), the scale markers passed on the way, galactic coordinates of the
    target, the Sun's apparent magnitude seen from the target, and a profile of the target's apparent magnitude and
    angular diameter versus the distance left (log-spaced)."""
    eph = eph or default_ephemeris()
    start = observer_position(observer, t, eph)
    end, R, M, kind, extra = _target(target, t, eph)
    d_vec = end - start
    D = float(np.linalg.norm(d_vec))
    standoff = standoff if standoff is not None else (4.0 if kind == "body" else 30.0)
    arrive = end - d_vec / D * R * standoff
    # profile: the target seen from distance d left
    left = np.geomspace(R * standoff, D, n_profile)
    if kind == "body":
        b = extra["body"]
        m_now = float(np.ravel(_magnitude(b, t, eph))[0]) if b != "sun" else -26.74
        mag = m_now + 5.0 * np.log10(left / D)
    else:
        mag = apparent_magnitude_star(M, left)
    prof = {"distance_left_km": left.tolist(), "magnitude": np.round(mag, 3).tolist(),
            "angular_diameter_deg": angular_diameter_deg(R, left).tolist()}
    markers = {k: v for k, v in MARKERS.items() if v < D}
    out = {"target": {"kind": kind, "radius_km": R, **extra}, "t": t.iso(),
           "start_km": start.tolist(), "end_km": end.tolist(), "arrival_km": arrive.tolist(),
           "distance_km": D, "distance_au": D / AU_KM, "distance_ly": D / LY_KM, "light_time_s": D / C_KM_S,
           "travel_times_s": travel_times(D), "markers_km": markers, "profile": prof,
           "galactic_deg": galactic(end), "standoff_radii": standoff}
    if kind == "star":
        out["sun_seen_from_target_mag"] = float(apparent_magnitude_star(M_V_SUN, np.linalg.norm(end)))
        out["target"]["abs_mag_v"] = M
    else:
        p = apparent(extra["body"], t, observer, eph, refraction=False) if extra["body"] != "sun" else None
        out["ephemeris_distance_km"] = float(p.distance_km[0]) if p is not None else None
    return out


def _magnitude(body, t, eph):
    from .phenomena import magnitude
    return magnitude(body, t, eph)


def scene(t: Time, observer: Observer | None = None, max_parallax_err: float = 0.2, eph=None) -> dict:
    """Everything a 3-D journey draws, heliocentric ICRS km: the planets, Sun and Moon at t (with radii), and the
    bright stars with a usable parallax (Hipparcos 2007 for the brightest, BSC5 otherwise) with absolute magnitude
    and B-V; stars without a parallax are listed by direction only."""
    from . import facts
    from .stars import load_bright_stars
    eph = eph or default_ephemeris()
    bodies = {b: {"pos_km": body_position(b, t, eph).tolist(), "radius_km": RADIUS_KM[b]}
              for b in ("sun", "mercury", "venus", "earth", "moon", "mars", "jupiter", "saturn", "uranus", "neptune")}
    c = load_bright_stars()
    hip = {int(k): v["parallax_mas"]["value"] for k, v in facts._stars().items()}
    stars = []
    for i in range(len(c)):
        plx = hip.get(int(c.hr[i]), c.plx[i])
        ok = plx is not None and not np.isnan(plx) and plx >= 5.0 and not c.plx_dyn[i]
        if int(c.hr[i]) in hip:
            ok = True
        if not ok:
            continue
        d_km = 1000.0 / plx * PC_KM
        stars.append([int(c.hr[i]), round(float(c.ra[i]), 4), round(float(c.dec[i]), 4), round(d_km / LY_KM, 3),
                      round(float(c.v[i] - 5 * math.log10(1000.0 / plx / 10.0)), 2),
                      None if np.isnan(c.bv[i]) else round(float(c.bv[i]), 2)])
    out = {"t": t.iso(), "bodies": bodies, "stars": stars, "star_columns": ["hr", "ra", "dec", "distance_ly",
                                                                             "abs_mag_v", "b_v"],
           "sun_abs_mag_v": M_V_SUN}
    if observer is not None:
        out["observer_km"] = observer_position(observer, t, eph).tolist()
    return out
