"""pinneapple_physics.closed_form.rocket_nozzle against tabulated isentropic values (Anderson, Modern Compressible
Flow, Table A.1, gamma = 1.4) and identities that hold for any correct implementation (the checks of the source
app's verify_analytical.py, plus the optimum-expansion identity C_F c* = v_e)."""
from __future__ import annotations

import math

import pytest

from pinneapple_physics.closed_form import rocket_nozzle as rn


def test_area_mach_relation_matches_table_a1():
    assert rn.area_ratio(2.0, 1.4) == pytest.approx(1.6875, abs=1e-4)
    assert rn.area_ratio(1.5, 1.4) == pytest.approx(1.1762, abs=1e-4)
    assert rn.area_ratio(1.0, 1.4) == pytest.approx(1.0, abs=1e-12)


def test_inverse_recovers_both_branches():
    assert rn.mach_from_area_ratio(1.6875, 1.4) == pytest.approx(2.0, abs=1e-4)
    assert rn.mach_from_area_ratio(1.17617, 1.4) == pytest.approx(1.5, abs=1e-4)
    for M in (0.1, 0.3, 0.7, 1.3, 3.0, 6.0):
        r = rn.area_ratio(M, 1.25)
        assert rn.mach_from_area_ratio(r, 1.25, supersonic=M > 1) == pytest.approx(M, rel=1e-9)
    with pytest.raises(ValueError):
        rn.mach_from_area_ratio(0.5, 1.4)


def test_pressure_and_temperature_ratios_match_table_a1():
    assert rn.pressure_ratio(2.0, 1.4) == pytest.approx(0.1278, abs=1e-4)
    assert rn.temperature_ratio(2.0, 1.4) == pytest.approx(0.5556, abs=1e-4)


def test_c_star_is_p0_at_over_mdot_and_mdot_is_linear_in_p0():
    g, R, T0, At = 1.2, 345.0, 2800.0, math.pi * 0.05 ** 2
    m1, m2 = rn.choked_mass_flow(4e6, At, g, R, T0), rn.choked_mass_flow(8e6, At, g, R, T0)
    assert m2 / m1 == pytest.approx(2.0, rel=1e-12)
    assert 4e6 * At / m1 == pytest.approx(rn.characteristic_velocity(g, R, T0), rel=1e-12)


def test_optimum_expansion_thrust_equals_mdot_times_exit_velocity():
    """At p_e = p_a the pressure term vanishes and F = mdot v_e, i.e. C_F c* = v_e."""
    g, R, T0, p0, eps = 1.22, 330.0, 3200.0, 7e6, 12.0
    perf = rn.nozzle_performance(gamma=g, R_gas=R, T_chamber=T0, p_chamber=p0, area_ratio_exit=eps, p_atm=1.0)
    p_e = perf["p_exit_pa"]
    cf = rn.thrust_coefficient(p_e, p0, p_e, eps, g)
    assert cf * perf["c_star_m_s"] == pytest.approx(perf["v_exit_m_s"], rel=1e-10)


def test_source_app_worked_example_is_in_physical_range():
    """The source README's sample motor: rt 5 cm, re 12 cm, 4 MPa, 2800 K, gamma 1.2, R 345."""
    perf = rn.nozzle_performance(gamma=1.2, R_gas=345.0, T_chamber=2800.0, p_chamber=4e6,
                                 area_ratio_exit=(0.12 / 0.05) ** 2)
    assert 2.5 < perf["mach_exit"] < 3.5 and 1.3 < perf["thrust_coefficient"] < 1.9
    assert 200 < perf["specific_impulse_s"] < 300 and perf["p_exit_pa"] < 4e6
