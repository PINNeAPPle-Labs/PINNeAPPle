"""Space journey geometry (#723): arrival distances equal the ephemeris or parallax distances, light times, the Sun
seen from Alpha Centauri, galactic coordinates, travel times."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_physics.astro import Observer, Time, apparent
from pinneapple_physics.astro import journey as jn
from pinneapple_physics.astro.ephemeris import C_KM_S

SP = Observer(-23.5505, -46.6333, 760)
T = Time.from_iso("2026-10-14T00:00:00Z")


@pytest.mark.parametrize("body", ["moon", "sun", "mars", "jupiter", "neptune"])
def test_arrival_distance_matches_the_ephemeris(body):
    j = jn.plan(SP, body, T)
    ref = apparent(body, T, SP, refraction=False).distance_km[0]           # topocentric, light-time corrected
    assert j["distance_km"] == pytest.approx(ref, rel=1e-3)                   # < 0.1 %
    end, start = np.array(j["end_km"]), np.array(j["start_km"])
    arrive = np.array(j["arrival_km"])
    assert np.linalg.norm(end - arrive) == pytest.approx(4 * j["target"]["radius_km"], rel=1e-9)
    assert np.linalg.norm(arrive - start) < j["distance_km"]


@pytest.mark.parametrize("ref", ["star:2491", "star:2061", "star:5459", "star:7001"])
def test_arrival_distance_matches_the_parallax(ref):
    from pinneapple_physics.astro import facts
    j = jn.plan(SP, ref, T)
    d_plx = 1000.0 / j["target"]["parallax_mas"] * jn.PC_KM
    assert j["distance_km"] == pytest.approx(d_plx, rel=1e-3)
    assert j["distance_ly"] == pytest.approx(facts.numbers(ref.replace(":", "-"))["distance_ly"]["value"], abs=0.051)   # the card rounds to 0.1 ly


def test_sun_seen_from_alpha_centauri():
    j = jn.plan(SP, "star:5459", T)                                           # Rigil Kentaurus
    assert j["sun_seen_from_target_mag"] == pytest.approx(0.5, abs=0.1)
    assert j["sun_seen_from_target_mag"] == pytest.approx(4.81 + 5 * np.log10(1000 / 754.81 / 10), abs=1e-6)


def test_light_times():
    assert jn.MARKERS["moon"] / C_KM_S == pytest.approx(1.28, abs=0.01)
    moon = jn.plan(SP, "moon", T)
    assert moon["light_time_s"] == pytest.approx(1.28, abs=0.1)              # 1.19-1.35 s from perigee to apogee
    sun = jn.plan(SP, "sun", T)
    assert sun["light_time_s"] / 60 == pytest.approx(8.3, abs=0.15)
    assert sun["distance_au"] == pytest.approx(1.0, abs=0.02)


def test_galactic_coordinates():
    lon, b = jn.galactic(jn._unit(266.40499, -28.93617))                     # galactic centre (IAU 1958, J2000)
    assert min(lon, 360 - lon) < 0.1 and abs(b) < 0.1
    assert jn.galactic(jn._unit(192.85948, 27.12825))[1] == pytest.approx(90.0, abs=1e-3)   # north galactic pole


def test_travel_times_and_profile():
    j = jn.plan(SP, "mars", T)
    tt = j["travel_times_s"]
    assert tt["light"] == pytest.approx(j["light_time_s"])
    assert tt["walking"] > tt["airliner"] > tt["voyager1"] > tt["light"]
    assert tt["voyager1"] == pytest.approx(j["distance_km"] / 17.0)
    prof = j["profile"]
    assert prof["magnitude"][0] < prof["magnitude"][-1]                       # brighter close up
    assert prof["angular_diameter_deg"][0] > prof["angular_diameter_deg"][-1]
    assert jn.angular_diameter_deg(1737.4, 384400) == pytest.approx(0.518, abs=0.002)
    assert set(j["markers_km"]) == {"moon", "au"}                             # Mars is 1-2.5 au away


def test_scene():
    s = jn.scene(T, SP)
    assert len(s["stars"]) > 500 and set(s["bodies"]) >= {"sun", "earth", "mars", "neptune"}
    hr = {x[0]: x for x in s["stars"]}
    assert hr[2491][3] == pytest.approx(8.6, abs=0.05)                         # Sirius, Hipparcos 2007
    assert np.linalg.norm(np.array(s["observer_km"]) - np.array(s["bodies"]["earth"]["pos_km"])) < 6400
