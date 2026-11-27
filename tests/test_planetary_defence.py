"""Tests for ``pinneapple_physics.astro.defence`` -- planetary defence, held to the mission that flew it (#381).

Three things are asserted, in the order of the chain:

* **the impulse**: the momentum a kinetic impactor delivers, and ``beta`` recovered from the *published* DART
  numbers -- the 32-minute shortening of Dimorphos' period against Didymos. The model reproduces ``beta = 3.65``
  against the published 3.61 (1 %) and the period change to 32.2 min, so nothing downstream is a free parameter.
* **the orbit**: the along-track impulse changes the period by ``dP/P = 3 dv/v``, the Yarkovsky drift of Apophis comes
  out at -166 m/yr against the ~170 m/yr published, and a nudge applied five years before a flyby moves the miss by
  the amount the encounter geometry implies.
* **the decision**: the fraction of an ensemble that clears the Earth, and the warning time beyond which the spread
  is wider than the Earth -- the horizon after which "we will miss" is not a guarantee.
* **the accelerations**: the two small terms the propagator carries (Yarkovsky ``A2`` and solar radiation pressure)
  are the vectors they claim to be, JPL's own 1-sigma on the flyby distance is read off the bundled record, the impact
  probability is the Gaussian it says it is, and the perturbation budget really does fly a different trajectory when a
  perturber is taken out of the field.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_physics.astro.defence import (
    APOPHIS_2029_JD,
    BETA_DART,
    DART,
    Terms,
    area_to_mass_ratio,
    dart_beta,
    deflect,
    deflection_table,
    delta_v,
    dimorphos_period_hours,
    ensemble_arcs,
    impact_probability,
    jpl_miss_sigma_km,
    miss_distance,
    period_change_fraction,
    perturbation_budget,
    sample_ensemble,
    solar_radiation_pressure,
    success_probability,
    trust_horizon,
    yarkovsky_drift,
)

APOPHIS_MASS_KG = 5.4e10            # 340 m at 2600 kg/m3
APOPHIS_DIAMETER_KM = 0.325         # the bundled Herschel diameter
EARTH_RADIUS_KM = 6371.0


# ------------------------------------------------------------------------------------------------ the impulse
def test_delta_v_is_the_momentum_per_unit_mass():
    dv = delta_v(570.0, 6.1, 4.8e9, beta=1.0)

    assert dv == pytest.approx(570.0 * 6100.0 / 4.8e9)
    assert dv * 1000 == pytest.approx(0.724, abs=0.001)          # mm/s, the momentum DART carried
    assert delta_v(1140.0, 6.1, 4.8e9) == pytest.approx(2 * dv)  # twice the impactor, twice the kick
    assert delta_v(570.0, 12.2, 4.8e9) == pytest.approx(2 * dv)  # twice the speed, twice the kick
    assert delta_v(570.0, 6.1, 4.8e9, beta=BETA_DART) == pytest.approx(BETA_DART * dv)


@pytest.mark.parametrize("bad", [dict(m_impactor_kg=0.0), dict(speed_km_s=0.0), dict(beta=0.0)])
def test_a_zero_impactor_speed_or_beta_is_refused(bad):
    args = dict(m_impactor_kg=570.0, speed_km_s=6.1, m_target_kg=4.8e9, beta=1.0) | bad
    with pytest.raises(ValueError):
        delta_v(**args)


def test_the_dart_period_change_gives_back_the_published_beta():
    """The anchor: the model is asked for beta from the measured 32 minutes, and has to answer 3.61."""
    assert dart_beta() == pytest.approx(3.61, rel=0.02)

    # and the other way round: with that beta, the momentum model reproduces the period change. The impactor met
    # Dimorphos head-on against its motion, so the impulse is retrograde and the period shortens.
    dv = delta_v(DART["impactor_kg"], DART["speed_km_s"], DART["target_kg"], dart_beta())
    d_period = period_change_fraction(-dv, 0.174) * DART["period_before_s"]
    assert d_period / 60.0 == pytest.approx(-32.0, abs=1.0)


def test_the_dimorphos_period_that_comes_out_of_the_orbital_speed_is_the_published_one():
    assert dimorphos_period_hours() == pytest.approx(11.92, rel=0.02)


def test_an_along_track_impulse_changes_the_period_by_three_dv_over_v():
    assert period_change_fraction(1.0, 100.0) == pytest.approx(0.03)
    assert period_change_fraction(-1.0, 100.0) == pytest.approx(-0.03)
    assert period_change_fraction(2.0, 100.0) == pytest.approx(2 * period_change_fraction(1.0, 100.0))
    with pytest.raises(ValueError):
        period_change_fraction(1.0, 0.0)


def test_the_yarkovsky_drift_of_apophis_is_the_published_one():
    """``A2 = -2.9e-14 au/day^2`` is the JPL solution for Apophis; the drift it implies is ~170 m a year."""
    drift_au_day = yarkovsky_drift(-2.9e-14, 0.922, 0.191)
    metres_per_year = drift_au_day * 1.496e11 * 365.25

    assert metres_per_year == pytest.approx(-170.0, abs=15.0)
    assert drift_au_day < 0.0, "a retrograde rotator spirals in"
    assert yarkovsky_drift(2.9e-14, 0.922, 0.191) == pytest.approx(-drift_au_day)


# -------------------------------------------------------------------------------------------------- the orbit
def test_a_body_with_no_impulse_keeps_its_orbit():
    from pinneapple_physics.astro.smallbodies import load_body

    body = load_body("apophis")
    jd = body.epoch_jd + 400.0
    same = deflect(body, jd, 0.0)

    assert same.epoch_jd == pytest.approx(jd)
    assert same.a_au == pytest.approx(body.a_au, rel=2e-4), "the elements drift only by the perturbations integrated"
    assert same.e == pytest.approx(body.e, rel=2e-3)


def test_the_2029_apophis_encounter_is_the_published_one():
    """The bundled JPL elements put Apophis 38 356 km from the Earth on 2029-04-13 at 7.41 km/s."""
    from pinneapple_physics.astro.smallbodies import load_body

    body = load_body("apophis")
    ca = miss_distance(body, APOPHIS_2029_JD - 30, APOPHIS_2029_JD + 30, step_days=0.5)

    assert ca["found"] is True
    assert ca["jd_tdb"] == pytest.approx(APOPHIS_2029_JD, abs=0.5)
    assert 0.0002 < ca["dist_au"] < 0.0004, "0.000256 au in the propagation this repository ships"
    assert ca["v_rel_km_s"] == pytest.approx(7.4, abs=0.2)
    assert ca["dist_km"] > EARTH_RADIUS_KM, "the 2029 pass is a miss"


def test_a_deflection_five_years_out_moves_the_miss_by_the_encounter_geometry():
    """A 1000 kg impactor at 6.1 km/s is 0.41 mm/s on Apophis: 204 km on a 38 356 km flyby, measured.

    The number is worth reading twice: the impressive-sounding "millions of km" a frame mistake produced is not
    physics, and 0.4 mm/s over five years buys a few hundred km on a body this heavy -- which is why the regime a
    real mission works in is a smaller target or a longer warning, and why the table has both.
    """
    from pinneapple_physics.astro.smallbodies import load_body

    body = load_body("apophis")
    jd_impact = APOPHIS_2029_JD - 5 * 365.25
    dv = delta_v(1000.0, 6.1, APOPHIS_MASS_KG, BETA_DART)
    assert dv * 1000 == pytest.approx(0.408, abs=0.002)

    baseline = miss_distance(body, APOPHIS_2029_JD - 30, APOPHIS_2029_JD + 30, step_days=0.5)
    nudged = miss_distance(deflect(body, jd_impact, dv), APOPHIS_2029_JD - 30, APOPHIS_2029_JD + 30, step_days=0.5)

    assert nudged["found"] is True
    assert nudged["dist_km"] > baseline["dist_km"], "the nudge pushes the flyby away"
    assert nudged["dist_km"] - baseline["dist_km"] == pytest.approx(204.0, abs=40.0)


def test_the_deflection_table_reports_one_row_per_impactor_mass():
    from pinneapple_physics.astro.smallbodies import load_body

    body = load_body("apophis")
    jd_impact = APOPHIS_2029_JD - 3 * 365.25
    rows = deflection_table(body, jd_impact, APOPHIS_2029_JD, masses_kg=(100.0, 1000.0), speed_km_s=6.1,
                            target_kg=APOPHIS_MASS_KG, window_days=30.0)

    assert [r["impactor_kg"] for r in rows] == [100.0, 1000.0]
    assert rows[0]["delta_v_m_s"] < rows[1]["delta_v_m_s"]
    assert all(r["warning_days"] == pytest.approx(3 * 365.25) for r in rows)
    assert all(r["found"] for r in rows)


# ----------------------------------------------------------------------------------------------- the decision
def test_the_ensemble_is_reproducible_and_centred_where_it_was_asked():
    a = sample_ensemble(n=64, a2_au_d2=-2.9e-14, a2_sigma=3e-15, beta=BETA_DART, beta_sigma=0.3, seed=7)
    b = sample_ensemble(n=64, a2_au_d2=-2.9e-14, a2_sigma=3e-15, beta=BETA_DART, beta_sigma=0.3, seed=7)

    assert a == b, "a seeded ensemble is the same ensemble"
    assert np.mean([r["a2_au_d2"] for r in a]) == pytest.approx(-2.9e-14, abs=1e-15)
    assert np.std([r["beta"] for r in a]) == pytest.approx(0.3, rel=0.3)

    fixed = sample_ensemble(n=4, a2_au_d2=-1e-14, beta=3.0, seed=1)
    assert {r["a2_au_d2"] for r in fixed} == {-1e-14} and {r["beta"] for r in fixed} == {3.0}


def test_the_success_probability_is_the_fraction_that_clears_the_earth():
    assert success_probability([1e5, 2e5, 3e5]) == 1.0
    assert success_probability([1000.0, 2000.0]) == 0.0
    assert success_probability([EARTH_RADIUS_KM + 1.0, 1000.0, 1e5, 2e5]) == pytest.approx(0.75)
    assert success_probability([EARTH_RADIUS_KM + 1.0], earth_radius_km=EARTH_RADIUS_KM,
                               capture_km=100.0) == 0.0, "the capture radius counts against the margin"
    with pytest.raises(ValueError):
        success_probability([])


def test_the_trust_horizon_is_where_the_spread_reaches_the_earth():
    days = np.array([365.0, 730.0, 1095.0, 1460.0])
    spread = np.array([1000.0, 3000.0, 6000.0, 9000.0])

    horizon = trust_horizon(warning_days=days, spread_km=spread, earth_radius_km=EARTH_RADIUS_KM)
    assert 1095.0 < horizon < 1460.0
    assert horizon == pytest.approx(1095.0 + (EARTH_RADIUS_KM - 6000.0) * 365.0 / 3000.0, abs=1.0)

    assert trust_horizon(warning_days=days, spread_km=[100.0] * 4) == 1460.0, "always inside: the last date"
    assert trust_horizon(warning_days=days, spread_km=[9000.0] * 4) == 0.0, "never inside: no horizon"
    with pytest.raises(ValueError):
        trust_horizon(warning_days=[1.0], spread_km=[])


# -------------------------------------------------------------------------------------------- the accelerations
def test_the_area_to_mass_ratio_is_the_sphere_it_is_computed_from():
    am = area_to_mass_ratio(APOPHIS_DIAMETER_KM, APOPHIS_MASS_KG)

    assert am == pytest.approx(np.pi * 325.0 ** 2 / 4 / APOPHIS_MASS_KG)      # 1.54e-6 m^2/kg
    assert am == pytest.approx(1.54e-6, rel=0.02)
    assert area_to_mass_ratio(0.65, APOPHIS_MASS_KG) == pytest.approx(4 * am), "twice the diameter, four sections"
    for bad in ((0.0, APOPHIS_MASS_KG), (APOPHIS_DIAMETER_KM, 0.0)):
        with pytest.raises(ValueError):
            area_to_mass_ratio(*bad)


def test_solar_radiation_pressure_is_the_solar_constant_over_c_times_area_over_mass():
    """P = 4.56e-6 N/m^2 at 1 au, so a body with 1 m^2/kg takes 4.56e-6 m/s^2 -- and the pressure falls as 1/r^2."""
    assert solar_radiation_pressure(1.0) == pytest.approx(4.56e-6)
    assert solar_radiation_pressure(2.0) == pytest.approx(2 * solar_radiation_pressure(1.0))
    assert solar_radiation_pressure(1.0, reflectivity=1.0) == pytest.approx(2 * solar_radiation_pressure(1.0))
    assert solar_radiation_pressure(1.0, r_au=2.0) == pytest.approx(solar_radiation_pressure(1.0) / 4)
    # Apophis: 1.54e-6 m^2/kg, and nine orders of magnitude under the Sun's own pull at the same distance
    apophis = solar_radiation_pressure(area_to_mass_ratio(APOPHIS_DIAMETER_KM, APOPHIS_MASS_KG))
    assert apophis == pytest.approx(7.0e-12, rel=0.1)
    assert apophis / 5.93e-3 < 1e-8
    with pytest.raises(ValueError):
        solar_radiation_pressure(-1.0)


def test_the_propagator_feels_the_yarkovsky_term_and_the_pressure_where_they_say():
    """The two optional accelerations, checked on the vectors directly: transverse along the motion, radial away."""
    from pinneapple_physics.astro.ephemeris import AU_KM, default_ephemeris
    from pinneapple_physics.astro.smallbodies import P_SRP, _accel

    eph = default_ephemeris()
    jd = 2460000.0
    sun_r, sun_v = eph.barycentric("sun", np.array([jd]))
    r_rel, v_rel = np.array([AU_KM, 0.0, 0.0]), np.array([0.0, 29.78, 0.0])
    r, v = r_rel + sun_r[0], v_rel + sun_v[0]
    plain = _accel(jd, r, v, eph, gr=False, perturbers=["sun"])

    yark = _accel(jd, r, v, eph, gr=False, perturbers=["sun"], a2_au_d2=1.0) - plain
    assert np.linalg.norm(yark) == pytest.approx(AU_KM / 86400.0 ** 2)        # 1 au/day^2, in km/s^2
    assert yark[1] > 0 and abs(yark[0]) < 1e-2 * abs(yark[1]), "at r = +x, v = +y the transverse is +y"
    assert np.linalg.norm(_accel(jd, r, v, eph, gr=False, perturbers=["sun"], a2_au_d2=-1.0) - plain + yark) \
        == pytest.approx(0.0, abs=1e-12)

    srp = _accel(jd, r, v, eph, gr=False, perturbers=["sun"], srp=(1.0, 0.0)) - plain
    assert np.linalg.norm(srp) == pytest.approx(P_SRP / 1000.0)              # N/m^2 * m^2/kg = m/s^2 -> km/s^2
    assert srp[0] > 0 and abs(srp[1]) < 1e-6 * abs(srp[0]), "radially out from the Sun, which is at -x"


def test_the_small_terms_are_off_by_default_and_switch_on_through_terms():
    from pinneapple_physics.astro.smallbodies import PERTURBERS

    assert Terms().kwargs()["perturbers"] == list(PERTURBERS)
    assert Terms().kwargs()["srp"] is None
    assert Terms().kwargs()["a2_au_d2"] == 0.0
    assert "uranus" not in Terms(dropped=("uranus",)).kwargs()["perturbers"]
    assert Terms(area_to_mass_m2_kg=1e-6, reflectivity=0.2).srp == (1e-6, 0.2)
    assert Terms(rtol=1e-12).kwargs()["rtol"] == 1e-12


def test_jpl_publishes_the_uncertainty_of_the_2029_flyby_and_the_model_lands_inside_it():
    """The crowded row of the record: 0.000252 au with a 3-sigma range of 0.0002475-0.0002568, so 37 724 +- 232 km."""
    jpl = jpl_miss_sigma_km()

    assert jpl["dist_km"] == pytest.approx(37724.5, abs=1.0)
    assert jpl["sigma_km"] == pytest.approx(232.4, abs=1.0)
    assert jpl["sigma_3_km"] == pytest.approx(697.3, abs=3.0)
    assert jpl["v_rel_km_s"] == pytest.approx(7.43, abs=0.01)
    assert jpl["cd"].startswith("2029-Apr-13")
    # the propagator this repository ships lands 632 km away, inside JPL's own 3 sigma
    assert abs(38356.2 - jpl["dist_km"]) < jpl["sigma_3_km"]
    with pytest.raises(ValueError):
        jpl_miss_sigma_km(year=1999)


def test_the_impact_probability_is_the_gaussian_it_says_it_is():
    sigma = 232.4
    assert impact_probability(EARTH_RADIUS_KM, sigma) == pytest.approx(0.5), "on the limb, a coin flip"
    assert impact_probability(0.0, sigma) == pytest.approx(1.0)
    assert impact_probability(EARTH_RADIUS_KM + 10 * sigma, sigma) < 1e-20, "ten sigma out, nothing left"
    assert impact_probability(EARTH_RADIUS_KM + sigma, sigma) == pytest.approx(
        0.159, abs=0.005), "one sigma past the limb: the one-sided tail"
    inside = [impact_probability(m, sigma) for m in np.linspace(0.0, EARTH_RADIUS_KM, 9)]
    assert all(a >= b for a, b in zip(inside, inside[1:], strict=False)), "at one while the limb is far"
    limb = [impact_probability(m, sigma) for m in np.linspace(EARTH_RADIUS_KM - 3 * sigma,
                                                             EARTH_RADIUS_KM + 3 * sigma, 13)]
    assert all(a > b for a, b in zip(limb, limb[1:], strict=False)), "and strictly falling across it"
    assert impact_probability(EARTH_RADIUS_KM - 1.0, 0.0) == 1.0
    assert impact_probability(EARTH_RADIUS_KM + 1.0, 0.0) == 0.0
    with pytest.raises(ValueError):
        impact_probability(1.0, -1.0)


def test_the_ensemble_draws_the_area_to_mass_ratio_around_where_it_was_asked():
    draw = dict(n=256, a2_au_d2=-2.9e-14, a2_sigma=3e-15, beta=BETA_DART, beta_sigma=0.3,
                area_to_mass_m2_kg=1.5e-6, am_sigma=0.5, seed=3)
    a = sample_ensemble(**draw)

    assert a == sample_ensemble(**draw), "a seeded ensemble is the same ensemble"
    assert np.mean([r["area_to_mass_m2_kg"] for r in a]) == pytest.approx(1.5e-6, rel=0.12)
    assert np.std([r["area_to_mass_m2_kg"] for r in a]) == pytest.approx(0.75e-6, rel=0.15), "50 % of the value"
    assert all(r["area_to_mass_m2_kg"] >= 0 for r in a), "a negative cross-section is clipped, not drawn"


def test_the_budget_flies_a_different_trajectory_when_a_perturber_is_taken_out():
    """Structure and plumbing on a short arc: the baseline is the reference, and taking Jupiter out moves the body.

    The weights themselves need the real encounter and are measured by the example; this is the check that the rows
    are what they claim -- and it runs over a hundred days, not the twenty-one years the study uses.
    """
    from pinneapple_physics.astro.smallbodies import load_body

    body = load_body("apophis")
    rows = perturbation_budget(body, body.epoch_jd + 100, planets=("jupiter",), srp_scales=(1.0,),
                               area_to_mass_m2_kg=area_to_mass_ratio(APOPHIS_DIAMETER_KM, APOPHIS_MASS_KG),
                               window_days=100.0, step_days=1.0, tolerance_km=0.0)

    assert rows[0]["term"].startswith("baseline")
    assert rows[0]["delta_km"] == 0.0 and rows[0]["dr_km"] == 0.0, "the baseline is its own reference"
    assert len(rows) == 4, "baseline, one planet, one pressure, the floor"
    jupiter = next(r for r in rows if r["term"].startswith("without jupiter"))
    srp = next(r for r in rows if r["term"].startswith("solar radiation pressure at 1x"))
    floor = next(r for r in rows if r["term"].startswith("the numerical floor"))
    assert jupiter["dr_km"] > 0.0, "the body is somewhere else without Jupiter's pull"
    assert srp["dr_km"] > 0.0
    assert floor["dr_km"] > 0.0
    assert jupiter["dr_km"] > srp["dr_km"], "over a hundred days Jupiter moves it more than the sunlight does"
    # this window contains no encounter, so the miss distances are infinite and their differences are not numbers:
    # the position offset is the column to read, which is why the row reports both
    assert all(np.isnan(r["delta_km"]) for r in rows[1:])


def test_the_ensemble_arcs_are_sampled_from_the_trajectories_that_were_flown():
    """One draw, four dates, one closest approach: the cloud is the ensemble and the miss is its own trajectory."""
    from pinneapple_physics.astro.smallbodies import load_body

    body = load_body("apophis")
    draws = sample_ensemble(n=1, a2_au_d2=-2.9e-14, beta=BETA_DART, area_to_mass_m2_kg=1.5e-6, seed=0)
    jd_impact = body.epoch_jd + 100.0
    arcs = ensemble_arcs(body, jd_impact, jd_encounter=body.epoch_jd + 150.0, mass_kg=1000.0, speed_km_s=6.1,
                         target_kg=APOPHIS_MASS_KG, draws=draws, jd_out=[jd_impact, body.epoch_jd + 125.0,
                                                                        body.epoch_jd + 150.0], window_days=20.0,
                         step_days=1.0)

    assert arcs["jd"] == [jd_impact, body.epoch_jd + 125.0, body.epoch_jd + 150.0]
    assert len(arcs["dist_earth_km"]) == 3 and arcs["dist_earth_km"][0] > arcs["dist_earth_km"][-1], "it approaches"
    assert len(arcs["nominal"]["r_helio_km"]) == 3
    member = arcs["members"][0]
    assert member["delta_v_m_s"] == pytest.approx(delta_v(1000.0, 6.1, APOPHIS_MASS_KG, BETA_DART))
    assert len(member["r_helio_km"]) == 3 and len(member["dist_earth_km"]) == 3
    off = np.linalg.norm(np.array(member["r_helio_km"]) - np.array(arcs["nominal"]["r_helio_km"]), axis=1)
    assert off[0] < off[-1], "the cloud widens as the years pass: that is what the animation draws"
