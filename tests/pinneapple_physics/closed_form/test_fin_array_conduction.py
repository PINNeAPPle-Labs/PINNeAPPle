"""Real regression tests for the generalized fin-array model -- most
importantly, that it reproduces the EXACT same numbers the two
downstream products (heatsink_design_ai, brakecool_ai) computed with
their own original, standalone formulas before this module existed.
A silent behavioral drift here would silently change what those real
products report."""
from __future__ import annotations

import math

from pinneapple_physics.closed_form import fin_array_conduction as fac


def test_matches_heatsink_design_ai_baseline_exactly():
    """heatsink_design_ai's baseline design (20 fins, 25mm tall, 1mm
    thick), 65W, standard_fan (h=35), k=205 aluminum, 80x80mm footprint,
    3mm base -- independently computed (before this module existed) as
    T_junction = 48.11806964469042 C."""
    params = fac.fin_array_profile_params(
        k_material_w_mk=205.0, h_conv_w_m2k=35.0,
        n_fins=20.0, fin_thickness_m=0.001, fin_length_m=0.025,
        transverse_extent_m=0.08, spacing_extent_m=0.08,
        base_thickness_m=0.003, power_w=65.0,
    )
    t_junction = fac.fin_array_base_temperature_c(params, t_ambient_c=25.0, power_w=65.0)
    assert abs(t_junction - 48.11806964469042) < 1e-6


def test_matches_brakecool_ai_baseline_radial_profile():
    """brakecool_ai's baseline rotor (36 vanes, 8mm gap, 4mm thick),
    k=50 cast iron, mean circumference 0.754m, radial flow length
    0.06m, 6mm friction ring -- cross-checked against a value
    independently confirmed live during brakecool_ai's own upgrade
    (800W highway h=120 -> ~112.4C nominal, ROBUST)."""
    params = fac.fin_array_profile_params(
        k_material_w_mk=50.0, h_conv_w_m2k=120.0,
        n_fins=36.0, fin_thickness_m=0.004, fin_length_m=0.008,
        transverse_extent_m=0.06, spacing_extent_m=0.754,
        base_thickness_m=0.006, power_w=800.0,
    )
    t_rotor = fac.fin_array_base_temperature_c(params, t_ambient_c=25.0, power_w=800.0)
    assert abs(t_rotor - 112.4) < 0.5  # brakecool_ai's own reported figure was rounded to 1 decimal


def test_temperature_profile_is_hottest_at_root_and_cools_toward_tip():
    params = fac.fin_array_profile_params(
        k_material_w_mk=205.0, h_conv_w_m2k=35.0,
        n_fins=20.0, fin_thickness_m=0.001, fin_length_m=0.025,
        transverse_extent_m=0.08, spacing_extent_m=0.08,
        base_thickness_m=0.003, power_w=65.0,
    )
    t_root = fac.fin_array_temperature_at_position_c(params, t_ambient_c=25.0, x_from_root_m=0.0)
    t_mid = fac.fin_array_temperature_at_position_c(params, t_ambient_c=25.0, x_from_root_m=0.0125)
    t_tip = fac.fin_array_temperature_at_position_c(params, t_ambient_c=25.0, x_from_root_m=0.025)
    assert t_root > t_mid > t_tip >= 25.0


def test_radial_profile_supports_a_different_corrected_length_than_the_resistance_network():
    """Real brakecool_ai pattern: the resistance network integrates
    lc over the axial gap direction, but the spatial coloring profile
    needs the vane's real radial extent instead -- same intrinsic `m`,
    different (explicitly passed) corrected length, not a competing
    derivation."""
    params = fac.fin_array_profile_params(
        k_material_w_mk=50.0, h_conv_w_m2k=120.0,
        n_fins=36.0, fin_thickness_m=0.004, fin_length_m=0.008,
        transverse_extent_m=0.06, spacing_extent_m=0.754,
        base_thickness_m=0.006, power_w=800.0,
    )
    radial_extent_m = 0.06
    lc_radial = radial_extent_m + 0.004 / 2.0
    t_inner = fac.fin_array_temperature_at_position_c(params, 25.0, 0.0, corrected_length_m=lc_radial)
    t_outer = fac.fin_array_temperature_at_position_c(params, 25.0, radial_extent_m, corrected_length_m=lc_radial)
    assert t_inner > t_outer  # hottest at the inner edge (nearest the hub), coolest at the outer edge
    # the boundary value at x=0 must match the scalar root temperature regardless of which
    # corrected length is used for the profile -- both real derivations share theta_fin_base_c
    t_root_default_axis = fac.fin_array_temperature_at_position_c(params, 25.0, 0.0)
    assert abs(t_inner - t_root_default_axis) < 1e-9
