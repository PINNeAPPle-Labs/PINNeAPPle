"""Observing session physics (#722): rise / transit / set against Skyfield, air mass against Kasten & Young (1989),
telescope optics against the standard formulas, field rotation against the analytic expression, slews and the sky
brightness model."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pinneapple_physics.astro import Observer, Time
from pinneapple_physics.astro import observing as ob

skyfield = pytest.importorskip("skyfield")

SITES = {"sao_paulo": (-23.5505, -46.6333, 760.0), "tromso": (69.6496, 18.9560, 10.0)}
STARS = {  # J2000 (Hipparcos)
    "sirius": (101.287155, -16.716116), "arcturus": (213.915300, 19.182409), "vega": (279.234735, 38.783689)}
START = "2026-10-13T12:00:00Z"


@pytest.fixture(scope="module")
def sf():
    from skyfield.api import Star, load, wgs84

    from pinneapple_physics.astro.ephemeris import find_kernel
    return {"ts": load.timescale(), "eph": load(find_kernel()), "Star": Star, "wgs84": wgs84}


@pytest.mark.parametrize("site", sorted(SITES))
@pytest.mark.parametrize("name", ["sun", "moon", *sorted(STARS)])
def test_rise_transit_set_match_skyfield(sf, site, name):
    from skyfield import almanac
    lat, lon, h = SITES[site]
    o = Observer(lat, lon, h)
    t0 = Time.from_iso(START)
    days = 2.0
    target = STARS.get(name, name)
    mine = ob.rise_transit_set(target, o, float(t0.utc), days)
    ts, eph = sf["ts"], sf["eph"]
    obs = eph["earth"] + sf["wgs84"].latlon(lat, lon, h)
    tgt = sf["Star"](ra_hours=target[0] / 15.0, dec_degrees=target[1]) if name in STARS else eph[name]
    hz = -34.0 / 60.0 if name in STARS else None
    a, b = ts.utc(2026, 10, 13, 12), ts.utc(2026, 10, 15, 12)
    rise, yr = almanac.find_risings(obs, tgt, a, b, horizon_degrees=hz)
    sett, ys = almanac.find_settings(obs, tgt, a, b, horizon_degrees=hz)
    tr = almanac.find_transits(obs, tgt, a, b)
    ref = {"rise": list(rise[yr].tt), "set": list(sett[ys].tt), "transit": list(tr.tt)}        # TT Julian dates
    for k in ("rise", "set", "transit"):
        mk = [float(Time(j).tt) for j in mine[k]]
        assert len(mk) == len(ref[k]), (k, mk, ref[k])
        for m, r in zip(sorted(mk), sorted(ref[k]), strict=True):
            assert abs(m - r) * 86400 < 60, (name, site, k, (m - r) * 86400)      # < 1 min


def test_kasten_young_air_mass():
    # Kasten & Young (1989), Appl. Opt. 28, 4735: X = 1 / (sin h + 0.50572 (h + 6.07995)^-1.6364), h in degrees
    h = np.array([90.0, 60.0, 30.0, 10.0, 5.0, 0.0])
    ref = 1.0 / (np.sin(np.radians(h)) + 0.50572 * (h + 6.07995) ** -1.6364)
    assert np.allclose(ob.airmass(h), ref, rtol=1e-12)
    assert 37 < ob.airmass(0.0) < 39                                     # finite at the horizon (sec z is not)
    assert ob.airmass(90.0) == pytest.approx(1.0, abs=1e-3)
    assert ob.airmass(30.0) == pytest.approx(1.9945, abs=1e-3)       # close to sec z = 2 well above the horizon
    e = ob.extinction(30.0)
    assert e["B"] > e["V"] > e["R"] and e["V"] == pytest.approx(0.17 * ob.airmass(30.0))


@pytest.mark.parametrize("key", sorted(ob.TELESCOPES))
def test_optics_standard_formulas(key):
    tel = ob.TELESCOPES[key]
    D, F = tel.aperture_mm, tel.focal_mm
    for ep in ob.EYEPIECES.values():
        o = ob.optics(tel, eyepiece=ep, camera=ob.CAMERAS["apsc"])
        M = F / ep.focal_mm
        assert o["magnification"] == pytest.approx(M)
        assert o["true_fov_deg"] == pytest.approx(ep.afov_deg / M)
        assert o["exit_pupil_mm"] == pytest.approx(D / M)
        assert o["dawes_arcsec"] == pytest.approx(116.0 / D)
        assert o["rayleigh_arcsec"] == pytest.approx(1.22 * 550e-9 / (D * 1e-3) * 206264.806)
        assert o["limiting_mag"] == pytest.approx(6.0 + 5 * math.log10(D / 7.0))
        assert o["image_scale_arcsec_per_px"] == pytest.approx(206264.806 * 3.9e-3 / F)
        assert o["camera_fov_deg"][0] == pytest.approx(math.degrees(23.5 / F))
    o = ob.optics(ob.TELESCOPES["newton200"], ob.EYEPIECES["plossl25"])
    assert o["magnification"] == pytest.approx(48.0) and o["true_fov_deg"] == pytest.approx(52 / 48)
    assert 13.0 < o["limiting_mag"] < 13.5 and o["dawes_arcsec"] == pytest.approx(0.58)


def test_field_rotation_matches_the_analytic_expression():
    """Numerical derivative of the parallactic angle along a star's apparent path vs omega cos(phi) cos(A)/cos(h)."""
    o = Observer(-23.5505, -46.6333, 760)
    t0 = Time.from_iso("2026-10-14T03:00:00Z")
    for target in STARS.values():
        dt = 30.0
        jd = float(t0.utc) + np.array([-dt, 0.0, dt]) / 86400
        alt, az, ra, dec = ob.target_altaz(target, Time(jd), o, refraction=False)
        if alt[1] < 5 or alt[1] > 80:
            continue
        from pinneapple_physics.astro.timescales import gast_rad
        ha = np.degrees(gast_rad(Time(jd))) + o.lon - ra
        q = ob.parallactic_angle(o.lat, ha, dec)
        rate_num = (q[2] - q[0]) / (2 * dt)
        rate = ob.field_rotation_rate(o.lat, alt[1], az[1])
        assert rate_num == pytest.approx(rate, rel=2e-3, abs=1e-7)


def test_slews():
    assert ob.axis_time(0.0, 4, 2) == 0.0
    assert ob.axis_time(8.0, 4, 2) == pytest.approx(2 + 2)                        # triangular: accelerate 4 deg, brake
    assert ob.axis_time(100.0, 4, 2) == pytest.approx(4 + (100 - 8) / 4)          # trapezoid
    s = ob.slew((30, 90), (60, 180), -23.55, "altaz")
    assert s["ok"] and s["seconds"] == pytest.approx(ob.axis_time(90, 4, 2) + 2)
    assert not ob.slew((30, 90), (5, 180), -23.55)["ok"]
    east, west = (40, 90), (40, 270)                                               # east and west of the meridian
    assert ob.slew(east, west, -23.55, "equatorial")["meridian_flip"]
    assert not ob.slew(east, (45, 100), -23.55, "equatorial")["meridian_flip"]


def test_sky_brightness():
    dark = float(ob.sky_brightness(-30, 180, 90, -10, 90))
    assert dark == pytest.approx(21.6, abs=0.05)                                   # no Moon, no twilight
    full = float(ob.sky_brightness(-30, 0, 30, 45, 60))
    quarter = float(ob.sky_brightness(-30, 90, 30, 45, 60))
    assert full < quarter < dark                                                    # moonlight brightens the sky
    assert float(ob.sky_brightness(-8, 180, 90, -10, 60)) < 14                     # bright twilight
    assert float(ob.nl_to_mag(ob.mag_to_nl(20.0))) == pytest.approx(20.0)


def test_visibility_night():
    o = Observer(-23.5505, -46.6333, 760)
    v = ob.visibility("saturn", o, Time.from_iso("2026-10-13T21:00:00Z"))
    assert v["best_window"] and 60 < v["best_window"]["alt"] < 70
    assert v["rise"] and v["set"] and v["transit"]
    polaris = ob.visibility((37.95, 89.26), o, Time.from_iso("2026-10-13T21:00:00Z"), step_min=30)
    assert polaris["never_rises"] and polaris["best_window"] is None


def test_weather_modes():
    from pinneapple_physics.astro import conditions as cd
    fc = {"time": ["2026-10-13T21:00Z", "2026-10-13T22:00Z", "2026-10-13T23:00Z"], "cloud_cover": [10, 90, 100]}
    assert cd.cloud_at(fc, "2026-10-13T21:30:00Z") == pytest.approx(50.0)
    assert cd.cloud_at(fc, "2026-10-14T05:00:00Z") is None
    assert cd.sky_view("real", fc, "2026-10-13T21:00:00Z")["hidden"] is False
    v = cd.sky_view("real", fc, "2026-10-13T22:30:00Z")
    assert v["hidden"] and v["mode"] == "real" and v["cloud_cover"] == pytest.approx(95.0)
    assert cd.sky_view("perfect", fc, "2026-10-13T22:30:00Z")["hidden"] is False
    fb = cd.sky_view("real", None, "2026-10-13T22:30:00Z")
    assert fb["mode"] == "perfect" and fb["fallback"]


def test_survey_cutout_is_cached(tmp_path, monkeypatch):
    import urllib.request
    calls = []

    class Resp:
        def read(self):
            return b"\xff\xd8 fake jpeg"

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: calls.append(req.full_url) or Resp())
    a = ob.survey_cutout(10.6847, 41.2690, 1.0, cache_dir=str(tmp_path))
    b = ob.survey_cutout(10.68471, 41.26902, 1.0, cache_dir=str(tmp_path))     # same rounded request: from disk
    assert a == b and len(calls) == 1 and "hips2fits" in calls[0] and "DSS2" in calls[0]
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: (_ for _ in ()).throw(OSError("offline")))
    with pytest.raises(OSError):
        ob.survey_cutout(83.82, -5.39, 1.0, cache_dir=str(tmp_path))
