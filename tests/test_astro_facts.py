"""Facts dataset of the observatory (#724): every card has a source per number and text in English and Portuguese, the
planet numbers are the NASA fact sheet's, and they agree with the library's independent data (DE421 masses and
orbits, IAU radii); constellation areas add up to the whole sky and match the IAU's."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pinneapple_physics.astro import facts

# NASA NSSDCA Planetary Fact Sheet - Metric (https://nssdc.gsfc.nasa.gov/planetary/factsheet/), to its precision:
# diameter km, mass 10^24 kg, density kg/m3, gravity m/s2, rotation h, day h, distance 10^6 km, period d, temp C, moons
NASA = {
    "mercury": (4879, 0.330, 5429, 3.7, 1407.6, 4222.6, 57.9, 88.0, 167, 0),
    "venus": (12104, 4.87, 5243, 8.9, -5832.5, 2802.0, 108.2, 224.7, 464, 0),
    "earth": (12756, 5.97, 5514, 9.8, 23.9, 24.0, 149.6, 365.2, 15, 1),
    "mars": (6792, 0.642, 3934, 3.7, 24.6, 24.7, 228.0, 687.0, -65, 2),
    "jupiter": (142984, 1898, 1326, 23.1, 9.9, 9.9, 778.5, 4331, -110, 95),
    "saturn": (120536, 568, 687, 9.0, 10.7, 10.7, 1432.0, 10747, -140, 274),
    "uranus": (51118, 86.8, 1270, 8.7, -17.2, 17.2, 2867.0, 30589, -195, 28),
    "neptune": (49528, 102, 1638, 11.0, 16.1, 16.1, 4515.0, 59800, -200, 16),
    "pluto": (2376, 0.0130, 1850, 0.7, -153.3, 153.3, 5906.4, 90560, -225, 5),
}
KEYS = ("diameter_km", "mass_kg", "density_kg_m3", "gravity_m_s2", "rotation_h", "day_h", "distance_sun_km",
        "orbital_period_d", "mean_temp_c", "moons")
SCALE = {"mass_kg": 1e24, "distance_sun_km": 1e6}
# IAU constellation areas, square degrees (https://www.iau.org/public/themes/constellations/)
IAU_AREAS = {"Hya": 1302.844, "Vir": 1294.428, "UMa": 1279.660, "Cet": 1231.411, "Eri": 1137.919, "Cru": 68.447,
             "Equ": 71.641, "Ori": 594.120, "UMi": 255.864}


def test_dataset_is_complete():
    assert facts.validate() == []
    kinds = facts.ids()
    assert sum(k == "star" for k in kinds.values()) >= 50
    assert sum(k == "constellation" for k in kinds.values()) == 89            # 88, Serpens in two halves
    assert sum(1 for i in kinds if i.startswith("M") and i[1:].isdigit()) == 110
    for oid in ("sun", "moon", "mercury", "neptune", "pluto", "ceres", "eris", "titan", "ganymede", "sat-iss",
                "sat-hubble", "star-2491", "M31", "con-Cru"):
        assert oid in kinds, oid


@pytest.mark.parametrize("lang", ["en", "pt"])
def test_every_card_has_sources_and_text(lang):
    src = facts.sources()
    for oid in facts.ids():
        c = facts.card(oid, lang)
        assert c["summary"] and c["history"] and c["name"], oid
        assert 3 <= len(c["curiosities"]) <= 5, oid
        for r in c["numbers"]:
            assert r["source"] in src and r["value"] is not None, (oid, r)
            assert r["value"] == facts.numbers(oid)[r["key"]]["value"]              # the card shows the dataset
        for x in c["curiosities"]:
            assert x["source"] in src and x["text"], oid
        assert set(c["sources"]) <= set(src)
        for q in c["quiz"]:
            assert 0 <= q["answer"] < len(q["options"]) and len(set(q["options"])) == len(q["options"]), oid


def test_text_is_really_in_both_languages():
    for oid in facts.ids():
        en, pt = facts.card(oid, "en"), facts.card(oid, "pt")
        assert en["summary"] != pt["summary"], oid
        assert en["history"] != pt["history"], oid
    assert facts.card("saturn", "pt")["name"] == "Saturno" and facts.card("con-Cru", "pt")["name"] == "Cruzeiro do Sul"
    with pytest.raises(ValueError):
        facts.card("saturn", "fr")


@pytest.mark.parametrize("body", sorted(NASA))
def test_planets_match_the_nasa_fact_sheet(body):
    n = facts.numbers(body)
    for key, ref in zip(KEYS, NASA[body], strict=True):
        assert n[key]["value"] / SCALE.get(key, 1) == pytest.approx(ref, abs=1e-9), (body, key)
        assert n[key]["source"] in ("nasa_planet_fs",), (body, key)


def test_planets_agree_with_the_library_ephemeris():
    """Independent check: DE421 GM values give the masses, the DE421 orbits the periods and distances, the IAU radii
    the diameters (equatorial vs mean radius: up to 3.5 % for the flattened giants)."""
    from pinneapple_physics.astro.ephemeris import GM, RADIUS_KM
    from pinneapple_physics.astro.orbits import planet_elements
    G = 6.67430e-20                                     # km^3 kg^-1 s^-2
    for b in ("mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"):
        n = facts.numbers(b)
        assert n["mass_kg"]["value"] == pytest.approx(GM[b] / G, rel=0.01), b
        assert n["diameter_km"]["value"] == pytest.approx(2 * RADIUS_KM[b], rel=0.035), b
        el = planet_elements(b, 2451545.0)
        assert n["orbital_period_d"]["value"] == pytest.approx(el["period_days"], rel=0.01), b
        assert n["distance_sun_km"]["value"] == pytest.approx(el["a_au"] * facts.AU_KM, rel=0.01), b


def test_comparisons_and_light_time():
    assert facts.comparison("diameter_km", facts.numbers("jupiter")["diameter_km"], "en").startswith("11.2 ×")
    assert "8.32 min" in facts.comparison("distance_earth_km", facts.numbers("sun")["distance_earth_km"], "en")
    assert "1,28 s" in facts.comparison("distance_earth_km", facts.numbers("moon")["distance_earth_km"], "pt")


def test_constellation_areas_cover_the_sky_and_match_the_iau():
    areas = facts._load("constellation_areas.json")["areas"]
    assert sum(areas.values()) == pytest.approx(4 * math.pi * (180 / math.pi) ** 2, rel=1e-4)
    for k, ref in IAU_AREAS.items():
        assert areas[k] == pytest.approx(ref, rel=0.005), k
    assert facts.numbers("con-Hya")["area_rank"]["value"] == 1
    assert facts.numbers("con-Cru")["area_rank"]["value"] == 88


def test_stars_and_messier():
    s = facts.numbers("star-2491")                     # Sirius
    assert s["v_mag"]["value"] == -1.46 and s["distance_ly"]["value"] == pytest.approx(8.6, abs=0.05)
    assert s["constellation"]["value"] == "CMa"
    m = facts.numbers("M31")
    assert m["constellation"]["value"] == "And" and m["distance_ly"]["unit"] == "Mly"
    ra, dec = facts.position("M31")                     # 00h 42m 44s, +41 deg 16'
    assert ra == pytest.approx(10.68, abs=0.05) and dec == pytest.approx(41.27, abs=0.05)
    assert facts.numbers("M42")["constellation"]["value"] == "Ori"
    assert facts.numbers("M16")["constellation"]["value"] == "Ser2"     # the Eagle Nebula is in Serpens Cauda


def test_search_in_both_languages():
    assert facts.search("cruzeiro")[0]["id"] == "con-Cru"
    assert facts.search("Saturno")[0]["id"] == "saturn"
    assert facts.search("andromeda galaxy")[0]["id"] == "M31"
    assert facts.search("Galáxia de Andrômeda", "pt")[0]["id"] == "M31"
    assert facts.search("betelgeuse")[0]["id"] == "star-2061"
    assert facts.search("estacao espacial internacional")[0]["id"] == "sat-iss"
    assert facts.resolve("body:saturn") == "saturn" and facts.resolve("star:2061") == "star-2061"
    assert facts.resolve("sat:25544") == "sat-iss" and facts.resolve("M 31") == "M31"


def test_tip_and_quiz():
    tip = facts.did_you_know("pt", seed=4)
    assert tip["text"] and tip["source"] in facts.sources()
    q = facts.quiz("saturn", "en")
    assert any("moons" in x["question"] for x in q)
    right = next(x for x in q if "moons" in x["question"])
    assert right["options"][right["answer"]] == "274"


def test_tonight():
    from pinneapple_physics.astro import Observer, Time
    sp = Observer(-23.5505, -46.6333, 760)
    t = Time.from_iso("2026-10-13T21:00:00Z")
    polaris = facts.tonight("star-424", sp, t)
    assert polaris["never_rises"] and polaris["rise"] is None
    sat = facts.tonight("saturn", sp, t)
    assert sat["above_now"] and 60 < sat["max_alt"] < 70 and sat["dark_best"]["alt"] > 60
    assert facts.tonight("sat-iss", sp, t) is None
    crux = facts.tonight("con-Cru", Observer(-33.87, 151.21, 0), t)       # circumpolar from Sydney
    assert crux["never_sets"]
    alts = np.array([facts.tonight("M31", sp, t)["max_alt"]])
    assert np.all((alts > 20) & (alts < 30))           # 90 - 23.55 - 41.27 = 25.2 deg
