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
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_physics.astro.defence import (
    APOPHIS_2029_JD,
    BETA_DART,
    DART,
    dart_beta,
    deflect,
    deflection_table,
    delta_v,
    dimorphos_period_hours,
    miss_distance,
    period_change_fraction,
    sample_ensemble,
    success_probability,
    trust_horizon,
    yarkovsky_drift,
)

APOPHIS_MASS_KG = 5.4e10            # 340 m at 2600 kg/m3
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
