"""An observing session, with honest physics: plan a night, set up a telescope, slew to a target and track it.

* Visibility: altitude / azimuth / airmass curves, rise, transit and set (Solar System bodies through
  :func:`.phenomena.rise_transit_set`; stars and deep-sky objects from their apparent places, the same root finder and
  the standard 34' refraction at the horizon), astronomical twilight, the Moon's altitude, phase and separation, and the
  best window of the night.
* The atmosphere: air mass (Kasten & Young 1989, :func:`.astrometry.airmass`), extinction per band, and the sky
  brightness from the Moon (Krisciunas & Schaefer 1991) and twilight.
* Telescope optics: magnification, true field, exit pupil, resolving power (Dawes, Rayleigh), limiting magnitude,
  image scale and camera field.
* The mount: slew time with a trapezoidal velocity profile per axis, horizon and meridian limits (meridian flip of a
  German equatorial mount), and the field rotation of an alt-azimuth mount.

    from pinneapple_physics.astro import observing, Observer, Time
    v = observing.visibility("saturn", Observer(-23.55, -46.63, 760), Time.from_iso("2026-10-13T21:00:00Z"))
    o = observing.optics(observing.TELESCOPES["newton200"], eyepiece=observing.EYEPIECES["plossl25"])
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from . import frames
from .astrometry import Observer, airmass, apparent, star_apparent
from .ephemeris import BODIES, default_ephemeris
from .phenomena import STD_REFRACTION_DEG, find_roots, moon_phase, twilight_events
from .phenomena import rise_transit_set as _body_rts
from .timescales import Time

ARCSEC = 206264.806
OMEGA_EARTH_DEG_S = 360.98564736629 / 86400.0          # sidereal rotation, deg per (solar) second
OMEGA_EARTH_RAD_S = 7.2921150e-5


# ------------------------------------------------------------------------------------------------ targets
def _is_body(target) -> bool:
    return isinstance(target, str) and target.lower() in BODIES


def target_altaz(target, t: Time, observer: Observer, refraction: bool = True, eph=None):
    """Apparent (alt, az, ra_of_date, dec_of_date) in degrees of a body name or a J2000 (ra, dec) pair at the times
    ``t`` (scalar or array)."""
    jd = np.atleast_1d(t.utc)
    if _is_body(target):
        p = apparent(target.lower(), Time(jd, np.atleast_1d(t.dut1)), observer, eph, refraction=refraction)
        return (p.alt_apparent if refraction else p.alt), p.az, p.ra, p.dec
    ra0, dec0 = float(target[0]), float(target[1])
    out = np.empty((4, len(jd)))
    for i, j in enumerate(jd):
        ra, dec, alt, az, alt_r = star_apparent(np.array([ra0]), np.array([dec0]), Time(j), observer, eph=eph)
        out[:, i] = (alt_r[0] if refraction else alt[0]), az[0], ra[0], dec[0]
    return out[0], out[1], out[2], out[3]


def rise_transit_set(target, observer: Observer, jd_start: float, days: float = 1.0, eph=None) -> dict:
    """Rise, upper transit and set (UTC Julian dates) in [jd_start, jd_start + days). Bodies: the almanac definition
    (centre 34' + semi-diameter below the horizon). Stars and deep-sky objects: centre 34' below the geometric
    horizon (standard refraction), transit when the hour angle is zero."""
    if _is_body(target):
        return _body_rts(target.lower(), observer, jd_start, days, eph)

    def alt(jd):
        return target_altaz(target, Time(jd), observer, refraction=False, eph=eph)[0] + STD_REFRACTION_DEG

    def ha(jd):
        _, _, ra, _ = target_altaz(target, Time(jd), observer, refraction=False, eph=eph)
        from .timescales import gast_rad
        last = np.degrees(np.atleast_1d(gast_rad(Time(jd)))) + observer.lon
        return (last - ra + 180.0) % 360.0 - 180.0

    step = 1.0 / 24
    ev = find_roots(alt, jd_start, jd_start + days, step)
    transits = [j for j, d in find_roots(ha, jd_start, jd_start + days, step) if d > 0]
    transits = [j for j in transits if abs(ha(np.array([j]))[0]) < 1.0]
    return {"rise": [j for j, d in ev if d > 0], "transit": transits, "set": [j for j, d in ev if d < 0]}


# --------------------------------------------------------------------------------------------- atmosphere
# Extinction coefficients (mag per air mass) for a good site; Krisciunas & Schaefer (1991) adopt k_V = 0.172 for
# Mauna Kea. Blue is dimmed most, which is why low objects look yellow-red.
EXTINCTION_K = {"B": 0.30, "V": 0.17, "R": 0.11}


def extinction(alt_apparent_deg, k=None) -> dict:
    """Extinction (mag) per band at an apparent altitude: k * X with X the Kasten-Young air mass."""
    X = airmass(alt_apparent_deg)
    return {b: (k or EXTINCTION_K)[b] * X for b in (k or EXTINCTION_K)}


def nl_to_mag(b_nl):
    """Surface brightness: nanolamberts -> V mag/arcsec^2 (Krisciunas & Schaefer 1991, eq. 1 inverted)."""
    return (20.7233 - np.log(np.asarray(b_nl, float) / 34.08)) / 0.92104


def mag_to_nl(m):
    return 34.08 * np.exp(20.7233 - 0.92104 * np.asarray(m, float))


def moon_brightness_nl(moon_phase_angle_deg, moon_sep_deg, moon_alt_deg, target_alt_deg, k=0.172):
    """Sky brightness (nanolamberts, V) added by moonlight at the target, Krisciunas & Schaefer (1991, PASP 103,
    1033): B = f(rho) I*(alpha) 10^(-0.4 k X(Z_moon)) (1 - 10^(-0.4 k X(Z))), with the Moon's illuminance
    I* = 10^(-0.4 (3.84 + 0.026|alpha| + 4e-9 alpha^4)), the scattering function
    f(rho) = 10^5.36 (1.06 + cos^2 rho) + 10^(6.15 - rho/40) and the air mass X(Z) = (1 - 0.96 sin^2 Z)^-1/2.
    Zero when the Moon is below the horizon."""
    a = np.abs(np.asarray(moon_phase_angle_deg, float))
    rho = np.clip(np.asarray(moon_sep_deg, float), 0.5, 180.0)
    illum = 10 ** (-0.4 * (3.84 + 0.026 * a + 4e-9 * a**4))
    f = 10**5.36 * (1.06 + np.cos(np.radians(rho)) ** 2) + 10 ** (6.15 - rho / 40.0)
    X = lambda alt: (1 - 0.96 * np.sin(np.radians(90.0 - np.clip(alt, 0.0, 90.0))) ** 2) ** -0.5  # noqa: E731
    B = f * illum * 10 ** (-0.4 * k * X(moon_alt_deg)) * (1 - 10 ** (-0.4 * k * X(target_alt_deg)))
    return np.where(np.asarray(moon_alt_deg) > 0, B, 0.0)


def dark_sky_nl(target_alt_deg, k=0.172, b_zenith_nl=79.0):
    """The natural night sky at the target (KS91 eq. 2): B_zen 10^(-0.4 k (X - 1)) X; 79 nL ~ 21.6 V mag/arcsec^2."""
    X = (1 - 0.96 * np.sin(np.radians(90.0 - np.clip(np.asarray(target_alt_deg, float), 0.0, 90.0))) ** 2) ** -0.5
    return b_zenith_nl * 10 ** (-0.4 * k * (X - 1)) * X


def twilight_mag(sun_alt_deg):
    """Approximate zenith sky brightness in twilight, V mag/arcsec^2, from the Sun's altitude: about 11 at sunset's
    civil limit (-6 deg), 16.5 at -12 deg and the dark-sky level beyond -18 deg (a linear fit to typical twilight
    photometry; good to ~1 mag, enough to decide what can be seen)."""
    h = np.asarray(sun_alt_deg, float)
    return np.where(h > -18.0, np.clip(11.0 + (-6.0 - h) * 0.875, 3.0, 21.6), 99.0)


def sky_brightness(sun_alt_deg, moon_phase_angle_deg, moon_sep_deg, moon_alt_deg, target_alt_deg, k=0.172,
                   dark_zenith_mag=21.6):
    """Sky brightness at the target, V mag/arcsec^2: dark sky + moonlight (Krisciunas & Schaefer 1991) + twilight."""
    b = dark_sky_nl(target_alt_deg, k, float(mag_to_nl(dark_zenith_mag)))
    b = b + moon_brightness_nl(moon_phase_angle_deg, moon_sep_deg, moon_alt_deg, target_alt_deg, k)
    tw = twilight_mag(sun_alt_deg)
    b = b + np.where(tw < 90, mag_to_nl(np.minimum(tw, 90)), 0.0)
    return nl_to_mag(b)


# --------------------------------------------------------------------------------------------- visibility
def visibility(target, observer: Observer, t: Time, hours: float = 24.0, step_min: float = 5.0, min_alt: float = 30.0,
               eph=None) -> dict:
    """The night for a target from ``observer`` over ``hours`` after ``t``: samples of altitude, azimuth, air mass,
    Sun and Moon altitude, Moon separation and sky brightness; rise / transit / set; astronomical twilight; the
    Moon's phase; the best window (dark sky, Sun < -18 deg, target above ``min_alt``, highest point); and whether
    the target never rises."""
    eph = eph or default_ephemeris()
    jd0 = float(np.ravel(t.utc)[0])
    jds = jd0 + np.arange(0.0, hours * 60.0 + step_min / 2, step_min) / 1440.0
    T = Time(jds)
    alt, az, ra, dec = target_altaz(target, T, observer, eph=eph)
    sun = apparent("sun", T, observer, eph)
    moon = apparent("moon", T, observer, eph)
    sep = frames.angular_separation(ra, dec, moon.ra, moon.dec)
    mp = moon_phase(Time(jd0), eph)
    phase_angle = 180.0 - mp["phase_longitude_deg"] if mp["phase_longitude_deg"] <= 180 else mp["phase_longitude_deg"] - 180.0
    X = airmass(alt)
    sky = sky_brightness(sun.alt, phase_angle, sep, moon.alt_apparent, alt)
    rts = rise_transit_set(target, observer, jd0, hours / 24.0, eph)
    tw = twilight_events(observer, jd0, hours / 24.0, eph)
    dark = sun.alt < -18.0
    ok = dark & (alt >= min_alt)
    best = None
    if ok.any():
        idx = np.nonzero(ok)[0]
        runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
        run = max(runs, key=len)
        k = run[np.argmax(alt[run])]
        best = {"start": Time(jds[run[0]]).iso(), "end": Time(jds[run[-1]]).iso(), "best": Time(jds[k]).iso(),
                "alt": float(alt[k]), "airmass": float(X[k]), "hours": float(len(run) * step_min / 60.0),
                "moon_sep": float(sep[k]), "moon_alt": float(moon.alt_apparent[k]), "sky_mag": float(sky[k])}
    from .timescales import gast_rad
    ha = np.degrees(np.atleast_1d(gast_rad(T))) + observer.lon - ra
    q = parallactic_angle(observer.lat, ha, dec)
    ext = extinction(np.maximum(alt, -0.4))
    iso = lambda xs: [Time(x).iso() for x in xs]  # noqa: E731
    return {"jd": jds.tolist(), "alt": alt.tolist(), "az": az.tolist(), "ra_date": ra.tolist(), "dec_date": dec.tolist(),
            "parallactic_deg": q.tolist(),
            "field_rotation_deg_s": [float(x) if a > 1 and a < 89.5 else None
                                     for x, a in zip(field_rotation_rate(observer.lat, alt, az), alt, strict=True)],
            "extinction": {b: [None if not np.isfinite(x) else float(x) for x in v] for b, v in ext.items()},
            "airmass": [None if not np.isfinite(x) or x > 40 else float(x) for x in X],
            "sun_alt": sun.alt.tolist(), "moon_alt": moon.alt_apparent.tolist(), "moon_sep": sep.tolist(),
            "sky_mag": [float(x) for x in sky],
            "rise": iso(rts["rise"]), "transit": iso(rts["transit"]), "set": iso(rts["set"]),
            "astronomical_dusk": iso(tw["astronomical"]["setting"]), "astronomical_dawn": iso(tw["astronomical"]["rising"]),
            "moon_illuminated": float(mp["illuminated_fraction"]), "moon_phase_angle": float(phase_angle),
            "never_rises": bool(np.all(alt < 0)), "always_up": bool(np.all(alt > 0)), "best_window": best,
            "min_alt": min_alt}


# ------------------------------------------------------------------------------------------------ optics
@dataclass(frozen=True)
class Telescope:
    name: str
    aperture_mm: float
    focal_mm: float
    kind: str = "refractor"
    central_obstruction: float = 0.0        # diameter fraction


@dataclass(frozen=True)
class Eyepiece:
    name: str
    focal_mm: float
    afov_deg: float


@dataclass(frozen=True)
class Camera:
    name: str
    width_mm: float
    height_mm: float
    pixel_um: float


TELESCOPES = {
    "refractor80": Telescope("80 mm f/6 refractor", 80.0, 480.0, "refractor"),
    "newton200": Telescope("200 mm f/6 Newtonian", 200.0, 1200.0, "newtonian", 0.23),
    "sct8": Telescope('8" (203 mm) f/10 Schmidt-Cassegrain', 203.2, 2032.0, "sct", 0.34),
    "binoculars": Telescope("10x50 binoculars", 50.0, 250.0, "binoculars"),
}
EYEPIECES = {
    "plossl32": Eyepiece("32 mm Plössl, 50°", 32.0, 50.0),
    "plossl25": Eyepiece("25 mm Plössl, 52°", 25.0, 52.0),
    "plossl10": Eyepiece("10 mm Plössl, 52°", 10.0, 52.0),
    "wide6": Eyepiece("6 mm wide-field, 82°", 6.0, 82.0),
}
CAMERAS = {
    "apsc": Camera("APS-C DSLR (23.5 x 15.6 mm, 3.9 µm)", 23.5, 15.6, 3.9),
    "imx585": Camera("1/1.2\" CMOS (11.1 x 6.2 mm, 2.9 µm)", 11.1, 6.2, 2.9),
}


def optics(telescope: Telescope, eyepiece: Eyepiece | None = None, camera: Camera | None = None,
           naked_eye_limit: float = 6.0, wavelength_nm: float = 550.0) -> dict:
    """What a telescope shows. Standard formulas: magnification M = F / f_e; true field = AFOV / M; exit pupil D / M;
    focal ratio F / D; Dawes limit 116 / D(mm) arcsec; Rayleigh limit 1.22 lambda / D; limiting magnitude
    m = m_eye + 5 log10(D / 7 mm) (the light grasp over a 7 mm dark-adapted pupil); light grasp (D / 7)^2; image scale
    206265 / F arcsec per mm (x pixel size per pixel); camera field = sensor / F (radians)."""
    D, F = telescope.aperture_mm, telescope.focal_mm
    out = {"telescope": asdict(telescope), "focal_ratio": F / D, "dawes_arcsec": 116.0 / D,
           "rayleigh_arcsec": 1.22 * wavelength_nm * 1e-6 / D * ARCSEC,
           "limiting_mag": naked_eye_limit + 5.0 * math.log10(D / 7.0), "light_grasp": (D / 7.0) ** 2,
           "image_scale_arcsec_per_mm": ARCSEC / F, "max_useful_magnification": 2.0 * D}
    if eyepiece is not None:
        M = F / eyepiece.focal_mm
        out.update(eyepiece=asdict(eyepiece), magnification=M, true_fov_deg=eyepiece.afov_deg / M,
                   exit_pupil_mm=D / M)
    if camera is not None:
        out.update(camera=asdict(camera), image_scale_arcsec_per_px=ARCSEC / F * camera.pixel_um * 1e-3,
                   camera_fov_deg=(math.degrees(camera.width_mm / F), math.degrees(camera.height_mm / F)))
    return out


# -------------------------------------------------------------------------------------------------- mount
def axis_time(delta_deg: float, rate_deg_s: float, accel_deg_s2: float) -> float:
    """Time to move one axis by ``delta_deg`` with a trapezoidal velocity profile (accelerate, coast, decelerate);
    a triangular profile when the move is too short to reach the slew rate."""
    d = abs(delta_deg)
    d_acc = rate_deg_s**2 / accel_deg_s2                # distance spent accelerating and decelerating
    if d <= d_acc:
        return 2.0 * math.sqrt(d / accel_deg_s2)
    return 2.0 * rate_deg_s / accel_deg_s2 + (d - d_acc) / rate_deg_s


def hour_angle_dec(alt_deg, az_deg, lat_deg):
    """(hour angle, declination) in degrees from (alt, az)."""
    a, z, p = np.radians(alt_deg), np.radians(az_deg), np.radians(lat_deg)
    dec = np.arcsin(np.sin(a) * np.sin(p) + np.cos(a) * np.cos(p) * np.cos(z))
    ha = np.arctan2(-np.sin(z) * np.cos(a), np.sin(a) * np.cos(p) - np.cos(a) * np.sin(p) * np.cos(z))
    return float(np.degrees(ha)), float(np.degrees(dec))


def slew(start_altaz, target_altaz, lat_deg: float, mount: str = "altaz", rate_deg_s: float = 4.0,
         accel_deg_s2: float = 2.0, settle_s: float = 2.0, min_alt: float = 10.0, flip_limit_deg: float = 0.0) -> dict:
    """A GoTo slew: both axes move at once, each with a trapezoidal profile; the slew takes the slower axis plus a
    settling time. Refused below ``min_alt``. An equatorial (German) mount slews in hour angle and declination and
    must do a meridian flip when the target is on the other side of the meridian (more than ``flip_limit_deg`` past
    it): the tube swaps sides, a 180 deg move in hour angle."""
    alt0, az0 = start_altaz
    alt1, az1 = target_altaz
    if alt1 < min_alt:
        return {"ok": False, "reason": "below_limit", "min_alt": min_alt, "seconds": None}
    if mount == "altaz":
        d_az = (az1 - az0 + 180.0) % 360.0 - 180.0
        axes = {"azimuth": d_az, "altitude": alt1 - alt0}
        flip = False
    else:
        ha0, dec0 = hour_angle_dec(alt0, az0, lat_deg)
        ha1, dec1 = hour_angle_dec(alt1, az1, lat_deg)
        flip = (ha0 < -flip_limit_deg) != (ha1 < -flip_limit_deg) and abs(ha1 - ha0) > 1e-6
        d_ha = (ha1 - ha0 + 180.0) % 360.0 - 180.0
        if flip:
            d_ha = d_ha + (180.0 if d_ha < 0 else -180.0)
        axes = {"hour_angle": d_ha, "declination": dec1 - dec0}
    times = {k: axis_time(v, rate_deg_s, accel_deg_s2) for k, v in axes.items()}
    return {"ok": True, "mount": mount, "axes_deg": axes, "axis_seconds": times, "meridian_flip": bool(flip),
            "seconds": max(times.values()) + settle_s, "rate_deg_s": rate_deg_s, "accel_deg_s2": accel_deg_s2}


def parallactic_angle(lat_deg, ha_deg, dec_deg):
    """Parallactic angle q (deg): the angle at the target between the directions to the zenith and to the pole,
    tan q = sin H / (tan phi cos delta - sin delta cos H)."""
    p, h, d = np.radians(lat_deg), np.radians(ha_deg), np.radians(dec_deg)
    return np.degrees(np.arctan2(np.sin(h), np.tan(p) * np.cos(d) - np.sin(d) * np.cos(h)))


def field_rotation_rate(lat_deg, alt_deg, az_deg):
    """Field rotation of an alt-azimuth mount (deg per second of time): the rate of change of the parallactic angle,
    dq/dt = -omega cos(phi) cos(A) / cos(h), with A the azimuth from north through east; its size is the classical
    omega cos(phi) cos(A) / cos(h), infinite at the zenith. The image in an alt-az eyepiece turns by -q. An equatorial
    mount has no field rotation."""
    return -np.degrees(OMEGA_EARTH_RAD_S * np.cos(np.radians(lat_deg)) * np.cos(np.radians(az_deg))
                      / np.cos(np.radians(alt_deg)))


# ---------------------------------------------------------------------------------------- survey imagery
HIPS2FITS_URL = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
SURVEY = "CDS/P/DSS2/color"
SURVEY_CREDIT = "DSS2 colour (STScI Digitized Sky Survey, POSS-II / UKST) via CDS hips2fits (Aladin, Strasbourg)"


def survey_url(ra_deg: float, dec_deg: float, fov_deg: float, width: int = 512, height: int = 512,
               rotation_deg: float = 0.0, survey: str = SURVEY) -> str:
    """URL of a JPEG cut-out of a HiPS survey around (ra, dec) from CDS hips2fits (gnomonic projection, north up)."""
    import urllib.parse
    q = {"hips": survey, "width": int(width), "height": int(height), "fov": f"{fov_deg:.5f}", "ra": f"{ra_deg:.5f}",
         "dec": f"{dec_deg:.5f}", "projection": "TAN", "format": "jpg", "rotation_angle": f"{rotation_deg:.3f}"}
    return HIPS2FITS_URL + "?" + urllib.parse.urlencode(q)


def survey_cutout(ra_deg: float, dec_deg: float, fov_deg: float, width: int = 512, height: int = 512,
                  cache_dir: str | None = None, timeout: float = 8.0, survey: str = SURVEY) -> bytes:
    """JPEG bytes of a survey cut-out, cached on disk (rounded position and field, so nearby requests share a file).
    Raises OSError when the service cannot be reached and nothing is cached."""
    import hashlib
    import os
    import urllib.request
    ra, dec = round(ra_deg % 360.0, 3), round(dec_deg, 3)
    fov = float(f"{fov_deg:.3g}")
    url = survey_url(ra, dec, fov, width, height, survey=survey)
    cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "pinneapple", "hips")
    path = os.path.join(cache_dir, hashlib.sha1(url.encode()).hexdigest()[:20] + ".jpg")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return fh.read()
    req = urllib.request.Request(url, headers={"User-Agent": "PINNeAPPle-observatory/1.0"})
    data = urllib.request.urlopen(req, timeout=timeout).read()
    if not data.startswith(b"\xff\xd8"):
        raise OSError("hips2fits did not return a JPEG")
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return data
