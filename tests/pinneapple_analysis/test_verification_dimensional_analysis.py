"""Verify dimensional_analysis.py's formulas against known textbook
values -- not just "runs without crashing." Reference values are the
standard, widely-cited approximate figures (Incropera & DeWitt-style
property tables at ~room temperature)."""
from __future__ import annotations

import math

import pytest

from pinneapple_analysis.verification.dimensional_analysis import (
    classify_flow_regime,
    compute_dimensionless_numbers,
)


def test_reynolds_number_water_in_pipe_matches_hand_calculation():
    # rho=998 kg/m3, U=2 m/s, L=0.05 m (pipe diameter), mu=1.002e-3 Pa.s
    # Re = rho*U*L/mu = 998*2*0.05/1.002e-3 = 99600.8 (hand-computed)
    dn = compute_dimensionless_numbers(
        velocity=2.0, length=0.05, density=998.0, dynamic_viscosity=1.002e-3,
    )
    assert dn.reynolds == pytest.approx(99600.8, rel=1e-3)


def test_reynolds_number_from_kinematic_viscosity_directly():
    # air: nu=1.5e-5 m2/s, U=10 m/s, L=1 m -> Re = U*L/nu = 666666.7
    dn = compute_dimensionless_numbers(velocity=10.0, length=1.0, kinematic_viscosity=1.5e-5)
    assert dn.reynolds == pytest.approx(10.0 * 1.0 / 1.5e-5, rel=1e-9)


def test_prandtl_number_of_air_matches_known_textbook_value_of_about_0_7():
    # air at ~300K: cp=1005 J/kgK, mu=1.81e-5 Pa.s, k=0.026 W/mK -> Pr ~ 0.70
    dn = compute_dimensionless_numbers(specific_heat=1005.0, dynamic_viscosity=1.81e-5, thermal_conductivity=0.026)
    assert dn.prandtl == pytest.approx(0.70, abs=0.02)


def test_prandtl_number_of_water_matches_known_textbook_value_of_about_7():
    # water at ~300K: cp=4186 J/kgK, mu=1.002e-3 Pa.s, k=0.6 W/mK -> Pr ~ 7.0
    dn = compute_dimensionless_numbers(specific_heat=4186.0, dynamic_viscosity=1.002e-3, thermal_conductivity=0.6)
    assert dn.prandtl == pytest.approx(7.0, abs=0.3)


def test_peclet_heat_equals_reynolds_times_prandtl():
    dn = compute_dimensionless_numbers(
        velocity=2.0, length=0.05, density=998.0, dynamic_viscosity=1.002e-3,
        specific_heat=4186.0, thermal_conductivity=0.6,
    )
    assert dn.peclet_heat == pytest.approx(dn.reynolds * dn.prandtl, rel=1e-9)


def test_mach_number_half_speed_of_sound():
    dn = compute_dimensionless_numbers(velocity=170.0, speed_of_sound=340.0)
    assert dn.mach == pytest.approx(0.5)


def test_froude_number_known_formula():
    # Fr = U / sqrt(g*L); U=3 m/s, L=1 m, g=9.81 -> Fr = 3/sqrt(9.81) = 0.958
    dn = compute_dimensionless_numbers(velocity=3.0, length=1.0)
    assert dn.froude == pytest.approx(3.0 / math.sqrt(9.81), rel=1e-6)


def test_weber_number_known_formula():
    # We = rho*U^2*L/sigma; water droplet-ish: rho=1000, U=1, L=0.001, sigma=0.072
    dn = compute_dimensionless_numbers(density=1000.0, velocity=1.0, length=0.001, surface_tension=0.072)
    assert dn.weber == pytest.approx(1000.0 * 1.0 ** 2 * 0.001 / 0.072, rel=1e-9)


def test_damkohler_number_known_formula():
    dn = compute_dimensionless_numbers(reaction_rate_constant=0.5, residence_time=2.0)
    assert dn.damkohler == pytest.approx(1.0)


def test_nusselt_correlation_applies_only_within_dittus_boelter_validity_range():
    # Re=1e5, Pr=0.7 (air, turbulent pipe) -- squarely inside Dittus-Boelter's range
    dn = compute_dimensionless_numbers(
        velocity=50.0, length=0.05, kinematic_viscosity=1.5e-5,
        specific_heat=1005.0, dynamic_viscosity=1.81e-5, thermal_conductivity=0.026,
    )
    assert dn.reynolds > 1.0e4
    assert dn.nusselt_correlation is not None
    assert dn.nusselt_correlation == pytest.approx(0.023 * dn.reynolds ** 0.8 * dn.prandtl ** 0.4, rel=1e-9)
    assert "Dittus-Boelter" in dn.nusselt_correlation_name


def test_nusselt_correlation_absent_and_explained_outside_validity_range():
    # Very low Re -- laminar, Dittus-Boelter does not apply
    dn = compute_dimensionless_numbers(velocity=0.01, length=0.05, kinematic_viscosity=1.5e-5, specific_heat=1005.0, dynamic_viscosity=1.81e-5, thermal_conductivity=0.026)
    assert dn.nusselt_correlation is None
    assert "nusselt" in dn.notes


def test_unset_inputs_produce_none_not_a_fabricated_number():
    dn = compute_dimensionless_numbers(velocity=1.0)  # no length, no viscosity at all
    assert dn.reynolds is None
    assert dn.prandtl is None
    assert dn.mach is None


def test_classify_flow_regime_internal_pipe_thresholds():
    assert "laminar" in classify_flow_regime(1000.0, geometry="internal_pipe")
    assert "transitional" in classify_flow_regime(3000.0, geometry="internal_pipe")
    assert "turbulent" in classify_flow_regime(10000.0, geometry="internal_pipe")


def test_classify_flow_regime_external_bluff_body_thresholds():
    assert "Stokes" in classify_flow_regime(0.1, geometry="external_bluff_body")
    assert "laminar" in classify_flow_regime(100.0, geometry="external_bluff_body")
    assert "turbulent" in classify_flow_regime(5000.0, geometry="external_bluff_body")


def test_classify_flow_regime_unknown_geometry_raises():
    with pytest.raises(ValueError):
        classify_flow_regime(1000.0, geometry="not_a_real_geometry")


def test_classify_flow_regime_none_reynolds_is_explicit_not_a_crash():
    result = classify_flow_regime(None)
    assert "unknown" in result
