"""The original test suites of the five PINNeAPPle-apps design tools (heatsink_design_ai, brakecool_ai, patchdose_ai,
ridetune_ai, soundshape_ai), ported to ``pinneapple_design.surrogate_inverse``: same assertions and seeds, imports
adapted (``train_surrogate`` returns a :class:`SurrogateFit`; tolerances are arguments instead of patched module
constants; the CAD checks read the mesh instead of a CadQuery solid).

The surrogate trainings (about 20, a few seconds each) make this suite slow: it runs with ``PINNEAPPLE_SLOW=1``.
``tests/pinneapple_design/test_surrogate_inverse.py`` holds the fast checks that run in CI.
"""
from __future__ import annotations

import math
import os
import struct
from functools import cache

import pytest

pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not os.environ.get("PINNEAPPLE_SLOW"),
                                reason="trains ~20 surrogates (set PINNEAPPLE_SLOW=1)")

import torch  # noqa: E402

from pinneapple_design.surrogate_inverse import brake_rotor as br  # noqa: E402
from pinneapple_design.surrogate_inverse import heatsink_fins as hs  # noqa: E402
from pinneapple_design.surrogate_inverse import meshes  # noqa: E402
from pinneapple_design.surrogate_inverse import reflex_port as rp  # noqa: E402
from pinneapple_design.surrogate_inverse import suspension_arm as sa  # noqa: E402
from pinneapple_design.surrogate_inverse import transdermal_patch as tp  # noqa: E402

torch.set_num_threads(1)          # tiny networks: one thread is fastest, and stable on a busy machine


@cache
def _surrogate(module, seed):
    return module.train_surrogate(seed=seed).model


def _triangle_count(stl: bytes) -> int:
    return struct.unpack("<I", stl[80:84])[0]


def _valid_binary_stl(stl: bytes) -> int:
    assert stl[:5] != b"solid"
    n = _triangle_count(stl)
    assert n > 0 and len(stl) == 80 + 4 + n * 50
    return n


# -- heatsink_design_ai ------------------------------------------------------------------------------------------

def test_hs_junction_temperature_increases_with_power():
    assert hs.junction_temperature_c(hs.BASELINE_DESIGN, 90.0, 35.0) > hs.junction_temperature_c(hs.BASELINE_DESIGN,
                                                                                                30.0, 35.0)


def test_hs_junction_temperature_decreases_with_better_cooling():
    weak = hs.junction_temperature_c(hs.BASELINE_DESIGN, 65.0, hs.COOLING_PROFILES["quiet_fan"])
    strong = hs.junction_temperature_c(hs.BASELINE_DESIGN, 65.0, hs.COOLING_PROFILES["high_static_pressure_fan"])
    assert strong < weak


def test_hs_more_fin_area_lowers_temperature():
    small = hs.HeatsinkDesign(10, 0.01, 0.001)
    big = hs.HeatsinkDesign(40, 0.03, 0.001)
    assert hs.junction_temperature_c(big, 65.0, 35.0) < hs.junction_temperature_c(small, 65.0, 35.0)


def test_hs_constraint_violations_flag_a_too_tight_design_but_not_a_valid_one():
    assert hs.constraint_violations(hs.HeatsinkDesign(55, 0.02, 0.0028))["min_spacing"] > 0
    assert all(v <= 0 for v in hs.constraint_violations(hs.HeatsinkDesign(20, 0.02, 0.001)).values())


def test_hs_optimized_design_never_violates_min_spacing():
    sur = _surrogate(hs, 3)
    for power in (35.0, 65.0, 120.0):
        for h in hs.COOLING_PROFILES.values():
            r = hs.optimize_design(sur, power, h)
            final = hs.HeatsinkDesign(round(r.design.fin_count), r.design.fin_height_m, r.design.fin_thickness_m)
            assert hs.constraint_violations(final)["min_spacing"] <= 0, (power, h, final)


def test_hs_optimizer_beats_or_matches_baseline_in_the_well_covered_regime():
    sur = _surrogate(hs, 7)
    power, h = 65.0, hs.COOLING_PROFILES["standard_fan"]
    r = hs.optimize_design(sur, power, h)
    card = hs.verify_design(r.design, power, h, r.surrogate_predicted_temp_c)
    assert card.real_temperature_c < hs.junction_temperature_c(hs.BASELINE_DESIGN, power, h)
    assert card.overall_verdict in ("PASS", "WARNING")


def test_hs_fin_temperature_profile_decreases_from_root_to_tip():
    d, power, h = hs.BASELINE_DESIGN, 65.0, hs.COOLING_PROFILES["standard_fan"]
    t_root = hs.fin_temperature_at_height_c(d, power, h, 0.0)
    t_mid = hs.fin_temperature_at_height_c(d, power, h, d.fin_height_m / 2)
    t_tip = hs.fin_temperature_at_height_c(d, power, h, d.fin_height_m)
    assert t_root > t_mid > t_tip >= hs.T_AMBIENT_C
    assert abs(t_root - (hs.T_AMBIENT_C + hs.profile_params(d, power, h)["theta_fin_base_c"])) < 0.05


def test_hs_verification_card_never_reports_confidence_above_what_actually_ran():
    sur = _surrogate(hs, 11)
    power, h = 65.0, hs.COOLING_PROFILES["standard_fan"]
    r = hs.optimize_design(sur, power, h)
    card = hs.verify_design(r.design, power, h, r.surrogate_predicted_temp_c)
    assert 0.0 < card.coverage < 1.0 and len(card.checks) >= 3


def test_hs_baseline_design_exports_a_valid_nonempty_stl():
    _valid_binary_stl(hs.export_stl_bytes(hs.BASELINE_DESIGN))


def test_hs_more_fins_means_more_triangles():
    few = _triangle_count(hs.export_stl_bytes(hs.HeatsinkDesign(5, 0.02, 0.001)))
    many = _triangle_count(hs.export_stl_bytes(hs.HeatsinkDesign(40, 0.02, 0.001)))
    assert many > few


def test_hs_cross_check_agrees_with_analytical_model_within_a_few_percent():
    """Was the dolfinx 3-D FEA check (skipped without a conda FEniCS env); now the 2-D finite-volume fin cell."""
    out = hs.fin_cell_check(hs.HeatsinkDesign(20, 0.025, 0.001), 65.0, hs.COOLING_PROFILES["standard_fan"])
    assert out["agreement"]["t_junction_relative_diff"] < 0.05
    assert out["finite_volume"]["n_cells"] > 0
    assert out["finite_volume"]["t_fin_root_c"] > out["finite_volume"]["t_fin_tip_c"]


# -- brakecool_ai ------------------------------------------------------------------------------------------------

def test_br_rotor_temp_increases_with_power():
    assert br.rotor_temperature_c(br.BASELINE_DESIGN, 1200.0, 120.0) > br.rotor_temperature_c(br.BASELINE_DESIGN,
                                                                                             400.0, 120.0)


def test_br_rotor_temp_decreases_with_better_cooling():
    city = br.rotor_temperature_c(br.BASELINE_DESIGN, 800.0, br.COOLING_PROFILES["city"])
    descent = br.rotor_temperature_c(br.BASELINE_DESIGN, 800.0, br.COOLING_PROFILES["mountain_descent"])
    assert descent < city


def test_br_more_vane_area_lowers_temperature():
    small = br.RotorDesign(16, 0.004, 0.002)
    big = br.RotorDesign(60, 0.014, 0.002)
    assert br.rotor_temperature_c(big, 800.0, 120.0) < br.rotor_temperature_c(small, 800.0, 120.0)


def test_br_optimized_design_never_violates_min_spacing():
    sur = _surrogate(br, 4)
    for power in (300.0, 800.0, 1500.0):
        for h in br.COOLING_PROFILES.values():
            r = br.optimize_design(sur, power, h)
            assert br.constraint_violations(r.design)["min_spacing"] <= 0, (power, h, r.design)


def test_br_optimizer_beats_baseline_in_the_well_covered_regime():
    sur = _surrogate(br, 6)
    power, h = 800.0, br.COOLING_PROFILES["highway"]
    r = br.optimize_design(sur, power, h)
    card = br.verify_design(r.design, power, h, r.surrogate_predicted_temp_c)
    assert card.real_temperature_c < br.rotor_temperature_c(br.BASELINE_DESIGN, power, h)
    assert card.overall_verdict in ("PASS", "WARNING")


def test_br_verification_card_coverage_is_explicit_and_partial():
    sur = _surrogate(br, 8)
    h = br.COOLING_PROFILES["highway"]
    r = br.optimize_design(sur, 800.0, h)
    card = br.verify_design(r.design, 800.0, h, r.surrogate_predicted_temp_c)
    assert 0.0 < card.coverage < 1.0 and len(card.checks) >= 3


def test_br_vane_radial_profile_matches_scalar_model_at_the_inner_edge():
    d, power, h = br.BASELINE_DESIGN, 800.0, br.COOLING_PROFILES["highway"]
    p = br.profile_params(d, power, h)
    assert abs(br.vane_temperature_at_radial_position_c(d, power, h, 0.0) - (br.T_AMBIENT_C + p["theta_vane_base_c"])) < 1e-9


def test_br_vane_radial_profile_cools_monotonically_toward_the_outer_edge():
    d, power, h = br.BASELINE_DESIGN, 800.0, br.COOLING_PROFILES["highway"]
    temps = [br.vane_temperature_at_radial_position_c(d, power, h, f) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert temps == sorted(temps, reverse=True) and temps[0] > temps[-1]


def test_br_monte_carlo_produces_a_real_varying_distribution():
    rep = br.monte_carlo_uncertainty(br.BASELINE_DESIGN, 800.0, br.COOLING_PROFILES["highway"],
                                     br.MAX_SAFE_ROTOR_TEMP_C, n_samples=200, seed=1)
    assert rep["n_samples"] == 200 and rep["std_c"] > 0.0
    assert rep["min_c"] < rep["mean_c"] < rep["max_c"] and rep["min_c"] <= rep["p95_c"] <= rep["max_c"]
    assert 0.0 <= rep["safe_fraction"] <= 1.0


def test_br_wide_safety_margin_design_is_robust():
    assert br.rotor_temperature_c(br.BASELINE_DESIGN, 800.0, br.COOLING_PROFILES["highway"]) < 150.0
    rep = br.monte_carlo_uncertainty(br.BASELINE_DESIGN, 800.0, br.COOLING_PROFILES["highway"],
                                     br.MAX_SAFE_ROTOR_TEMP_C, n_samples=200, seed=2)
    assert rep["verdict"] == "ROBUST" and rep["safe_fraction"] == 1.0 and rep["p95_c"] < br.MAX_SAFE_ROTOR_TEMP_C


def test_br_design_near_the_safety_limit_is_not_robust_and_has_more_spread_risk():
    power, h = 1480.0, br.COOLING_PROFILES["city"]
    nominal = br.rotor_temperature_c(br.BASELINE_DESIGN, power, h)
    assert 450.0 < nominal < br.MAX_SAFE_ROTOR_TEMP_C
    near = br.monte_carlo_uncertainty(br.BASELINE_DESIGN, power, h, br.MAX_SAFE_ROTOR_TEMP_C, n_samples=300, seed=3)
    robust = br.monte_carlo_uncertainty(br.BASELINE_DESIGN, 800.0, br.COOLING_PROFILES["highway"],
                                        br.MAX_SAFE_ROTOR_TEMP_C, n_samples=300, seed=3)
    assert near["verdict"] in ("MARGINAL", "FRAGILE")
    assert br.MAX_SAFE_ROTOR_TEMP_C - near["p95_c"] < br.MAX_SAFE_ROTOR_TEMP_C - robust["p95_c"]
    assert near["safe_fraction"] <= robust["safe_fraction"]


def test_br_wider_injected_tolerance_increases_real_spread():
    power, h = 1480.0, br.COOLING_PROFILES["city"]
    narrow = br.monte_carlo_uncertainty(br.BASELINE_DESIGN, power, h, br.MAX_SAFE_ROTOR_TEMP_C, n_samples=300, seed=5)
    wide = br.monte_carlo_uncertainty(br.BASELINE_DESIGN, power, h, br.MAX_SAFE_ROTOR_TEMP_C, n_samples=300, seed=5,
                                      thickness_tolerance_m=5 * br.VANE_THICKNESS_TOLERANCE_M,
                                      gap_tolerance_m=5 * br.VANE_GAP_TOLERANCE_M)
    assert wide["std_c"] > narrow["std_c"]


def test_br_vane_count_is_not_perturbed_by_manufacturing_sampling():
    """Vane count is fixed by construction: only thickness and gap are sampled (two inputs)."""
    rep = br.monte_carlo_uncertainty(br.BASELINE_DESIGN, 800.0, 120.0, 500.0, n_samples=100, seed=7)
    assert set(rep["tolerances_m"]) == {"vane_thickness", "vane_gap"}


def test_br_baseline_design_exports_a_valid_nonempty_stl():
    _valid_binary_stl(br.export_stl_bytes(br.BASELINE_DESIGN))


def test_br_more_vanes_means_more_triangles():
    few = _triangle_count(br.export_stl_bytes(br.RotorDesign(16, 0.008, 0.004)))
    many = _triangle_count(br.export_stl_bytes(br.RotorDesign(70, 0.008, 0.004)))
    assert many > few


# -- patchdose_ai ------------------------------------------------------------------------------------------------

def test_tp_thinner_membrane_gives_higher_rate():
    assert tp.release_rate_mg_day(tp.PatchDesign(20e-4, 20e-6, 60.0)) > tp.release_rate_mg_day(
        tp.PatchDesign(20e-4, 150e-6, 60.0))


def test_tp_higher_concentration_gives_higher_rate():
    assert tp.release_rate_mg_day(tp.PatchDesign(20e-4, 80e-6, 140.0)) > tp.release_rate_mg_day(
        tp.PatchDesign(20e-4, 80e-6, 10.0))


def test_tp_depletion_time_depends_only_on_membrane_thickness():
    a = tp.PatchDesign(10e-4, 50e-6, 20.0)
    b = tp.PatchDesign(35e-4, 50e-6, 130.0)
    assert abs(tp.depletion_time_days(a) - tp.depletion_time_days(b)) < 1e-9
    assert tp.depletion_time_days(tp.PatchDesign(10e-4, 150e-6, 20.0)) > tp.depletion_time_days(a)


# Target rates scaled x1000 from the source (0.05 / 0.1 / 0.12 / 0.2 "mg/day" were grams per day: the source's kg ->
# mg factor was 1e3); the designs and the physics are the same.

def test_tp_optimizer_moves_all_three_design_variables():
    r = tp.optimize_design(_surrogate(tp, 1), target_rate_mg_day=200.0, target_wear_days=1.5)
    assert abs(r.design.reservoir_conc_kg_m3 - tp.BASELINE_DESIGN.reservoir_conc_kg_m3) > 10.0


def test_tp_optimizer_hits_target_rate_and_duration_in_a_realistic_regime():
    sur = _surrogate(tp, 1)
    for rate, wear in [(50.0, 1.0), (100.0, 2.0), (120.0, 2.5)]:
        r = tp.optimize_design(sur, rate, wear)
        card = tp.verify_design(r.design, rate, wear, r.surrogate_predicted_rate_mg_day)
        assert abs(card.real_rate_mg_day - rate) / rate < 0.15
        assert card.real_depletion_days >= wear * 0.9
        assert card.overall_verdict in ("PASS", "WARNING")


def test_tp_optimized_design_stays_within_real_bounds():
    d = tp.optimize_design(_surrogate(tp, 2), 100.0, 2.0).design
    assert tp.AREA_BOUNDS_M2[0] <= d.area_m2 <= tp.AREA_BOUNDS_M2[1]
    assert tp.MEMBRANE_THICKNESS_BOUNDS_M[0] <= d.membrane_thickness_m <= tp.MEMBRANE_THICKNESS_BOUNDS_M[1]
    assert tp.RESERVOIR_CONC_BOUNDS_KG_M3[0] <= d.reservoir_conc_kg_m3 <= tp.RESERVOIR_CONC_BOUNDS_KG_M3[1]


def test_tp_constant_phase_matches_scalar_release_rate():
    d = tp.BASELINE_DESIGN
    t_dep, rate = tp.depletion_time_days(d), tp.release_rate_mg_day(d)
    pts = [p for p in tp.simulate_release_timeseries(d, t_dep * 1.5, 200) if p["day"] < t_dep * 0.98]
    assert len(pts) > 10
    for p in pts:
        assert p["rate_mg_day"] == pytest.approx(rate, rel=1e-6)
        assert p["cumulative_delivered_mg"] == pytest.approx(rate * p["day"], rel=1e-6)


def test_tp_tail_phase_strictly_decreasing():
    d = tp.BASELINE_DESIGN
    t_dep = tp.depletion_time_days(d)
    rates = [p["rate_mg_day"] for p in tp.simulate_release_timeseries(d, t_dep * 2, 300) if p["day"] > t_dep * 1.02]
    assert len(rates) > 10 and all(b < a for a, b in zip(rates, rates[1:], strict=False))


def test_tp_decay_constant_is_area_independent():
    small, big = tp.PatchDesign(5e-4, 80e-6, 60.0), tp.PatchDesign(40e-4, 80e-6, 60.0)
    assert tp.decay_rate_constant_per_day(small) == pytest.approx(tp.decay_rate_constant_per_day(big), rel=1e-9)
    assert tp.decay_rate_constant_per_day(small) == pytest.approx(1.0 / tp.depletion_time_days(small), rel=1e-9)


def test_tp_cumulative_delivered_stays_within_reasoned_bound():
    d = tp.BASELINE_DESIGN
    total = tp.simulate_release_timeseries(d, n_points=200)[-1]["cumulative_delivered_mg"]
    m = tp.reservoir_mass_mg(d)
    assert 0 < total <= 2.0 * m
    assert total == pytest.approx(1.39 * m, rel=0.05)


def test_tp_reservoir_remaining_hits_zero_at_depletion_and_stays_there():
    d = tp.BASELINE_DESIGN
    t_dep = tp.depletion_time_days(d)
    after = [p for p in tp.simulate_release_timeseries(d, t_dep * 2, 400) if p["day"] >= t_dep]
    assert len(after) > 5 and all(p["reservoir_remaining_mg"] == pytest.approx(0.0, abs=1e-6) for p in after)


def test_tp_default_duration_is_1_5x_depletion_time():
    d = tp.BASELINE_DESIGN
    assert tp.simulate_release_timeseries(d, n_points=10)[-1]["day"] == pytest.approx(1.5 * tp.depletion_time_days(d))


def test_tp_baseline_design_exports_a_valid_nonempty_stl():
    _valid_binary_stl(tp.export_stl_bytes(tp.BASELINE_DESIGN))


def test_tp_larger_area_means_larger_bounding_box():
    small, big = tp.PatchDesign(5e-4, 80e-6, 60.0), tp.PatchDesign(40e-4, 80e-6, 60.0)
    xs, ys, _ = meshes.bbox_size(tp.mesh(small))
    xb, yb, _ = meshes.bbox_size(tp.mesh(big))
    assert xb > xs and yb > ys
    assert xs == pytest.approx(2 * math.sqrt(small.area_m2 / math.pi) * 1e3, rel=1e-2)
    assert xb == pytest.approx(2 * math.sqrt(big.area_m2 / math.pi) * 1e3, rel=1e-2)


def test_tp_more_membrane_thickness_means_taller_bounding_box():
    thin, thick = tp.PatchDesign(20e-4, 10e-6, 60.0), tp.PatchDesign(20e-4, 200e-6, 60.0)
    assert meshes.bbox_size(tp.mesh(thick))[2] > meshes.bbox_size(tp.mesh(thin))[2]


# -- ridetune_ai -------------------------------------------------------------------------------------------------

def test_sa_thicker_beam_has_lower_stress():
    assert sa.bending_stress_pa(sa.ArmDesign(0.05, 0.025), 1200.0) < sa.bending_stress_pa(sa.ArmDesign(0.05, 0.01), 1200.0)


def test_sa_higher_load_gives_higher_stress_and_shorter_life():
    d = sa.BASELINE_DESIGN
    assert sa.bending_stress_pa(d, 2000.0) > sa.bending_stress_pa(d, 600.0)
    assert sa.fatigue_life_cycles(d, 2000.0) < sa.fatigue_life_cycles(d, 600.0)


def test_sa_allowable_stress_decreases_for_longer_target_life():
    assert sa.allowable_stress_for_life_pa(1e7) < sa.allowable_stress_for_life_pa(1e5)


def test_sa_optimizer_reduces_mass_while_meeting_real_target_life():
    load, target = 1200.0, 1.0e6
    r = sa.optimize_design(_surrogate(sa, 3), load, target)
    card = sa.verify_design(r.design, load, target, r.surrogate_predicted_stress_pa)
    assert r.design.mass_kg() < sa.BASELINE_DESIGN.mass_kg()
    assert card.real_life_cycles >= target * 0.7
    assert card.overall_verdict in ("PASS", "WARNING")


def test_sa_optimized_design_stays_within_real_geometric_bounds():
    sur = _surrogate(sa, 5)
    for load, target in [(600.0, 3e5), (1200.0, 1e6), (2000.0, 5e6)]:
        d = sa.optimize_design(sur, load, target).design
        assert sa.WIDTH_BOUNDS_M[0] <= d.width_m <= sa.WIDTH_BOUNDS_M[1]
        assert sa.THICKNESS_BOUNDS_M[0] <= d.thickness_m <= sa.THICKNESS_BOUNDS_M[1]


def test_sa_verification_never_fakes_meeting_an_unreachable_target():
    load, target = sa.LOAD_BOUNDS_N[1], sa.TARGET_LIFE_BOUNDS_CYCLES[1]
    r = sa.optimize_design(_surrogate(sa, 2), load, target)
    card = sa.verify_design(r.design, load, target, r.surrogate_predicted_stress_pa)
    meets = card.real_life_cycles >= target
    flagged = any(c.name == "fatigue_life_target" and c.verdict != "PASS" for c in card.checks)
    assert meets or flagged


def test_sa_stress_at_root_matches_scalar_bending_stress():
    d = sa.ArmDesign(0.05, 0.02)
    assert sa.bending_stress_at_position_pa(d, 1500.0, 0.0) == sa.bending_stress_pa(d, 1500.0)


def test_sa_stress_is_zero_at_the_free_tip_and_decreases_monotonically():
    d, load, L = sa.ArmDesign(0.05, 0.02), 1500.0, sa.ARM_LENGTH_M
    assert sa.bending_stress_at_position_pa(d, load, L) == 0.0
    s = [sa.bending_stress_at_position_pa(d, load, f * L) for f in (0.0, 0.25, 0.5)]
    assert s[0] > s[1] > s[2] > 0.0


def test_sa_fatigue_life_increases_with_both_width_and_thickness_real_derivative_sign():
    r = sa.compute_sensitivity(0.05, 0.022, 1200.0)
    assert r.d_life_d_width > 0 and r.d_life_d_thickness > 0 and r.elasticity_width > 0 and r.elasticity_thickness > 0


def test_sa_mass_derivatives_match_the_real_closed_form_linear_relation():
    w, t = 0.05, 0.022
    r = sa.compute_sensitivity(w, t, 1200.0)
    assert r.d_mass_d_width == pytest.approx(sa.STEEL_DENSITY_KG_M3 * t * sa.ARM_LENGTH_M)
    assert r.d_mass_d_thickness == pytest.approx(sa.STEEL_DENSITY_KG_M3 * w * sa.ARM_LENGTH_M)


def test_sa_thickness_elasticity_is_exactly_2x_width_elasticity_closed_form_check():
    b = sa.FATIGUE_STRENGTH_EXPONENT
    for w, t, load in [(0.05, 0.022, 1200.0), (0.02, 0.006, 600.0), (0.07, 0.03, 2400.0)]:
        r = sa.compute_sensitivity(w, t, load)
        assert r.elasticity_width == pytest.approx(-1.0 / b, rel=1e-4)
        assert r.elasticity_thickness == pytest.approx(-2.0 / b, rel=1e-4)
        assert r.elasticity_ratio == pytest.approx(2.0, rel=1e-4) and r.dominant_variable == "thickness_m"


def test_sa_dominant_variable_regression_concrete_example():
    d = sa.BASELINE_DESIGN
    r = sa.compute_sensitivity(d.width_m, d.thickness_m, 1200.0)
    assert r.dominant_variable == "thickness_m" and abs(r.elasticity_thickness) > abs(r.elasticity_width)
    assert 1.9 < r.elasticity_ratio < 2.1


def test_sa_halving_the_finite_difference_step_barely_changes_the_result():
    full = sa.compute_sensitivity(0.05, 0.022, 1200.0)
    half = sa.compute_sensitivity(0.05, 0.022, 1200.0, step_m=sa.H_M / 2)
    assert abs(full.elasticity_width - half.elasticity_width) / abs(full.elasticity_width) < 1e-4


def test_sa_result_to_dict_reports_a_correct_ratio_and_dominant_label_in_the_summary():
    r = sa.compute_sensitivity(0.05, 0.022, 1200.0)
    d = sa.sensitivity_to_dict(r)
    assert d["dominant_variable"] == "thickness_m" and "thickness" in d["summary"]
    assert f"{r.elasticity_ratio:.2f}x" in d["summary"]


def test_sa_baseline_design_exports_a_valid_nonempty_stl():
    _valid_binary_stl(sa.export_stl_bytes(sa.BASELINE_DESIGN))


def test_sa_plain_box_exports_exactly_twelve_triangles():
    assert _triangle_count(sa.export_stl_bytes(sa.BASELINE_DESIGN)) == 12


def test_sa_wider_thicker_beam_still_exports_a_valid_mesh_with_sensible_triangle_count():
    assert _valid_binary_stl(sa.export_stl_bytes(sa.ArmDesign(0.07, 0.03))) == 12


# -- soundshape_ai -----------------------------------------------------------------------------------------------

def test_rp_larger_port_area_raises_frequency():
    assert rp.resonant_frequency_hz(rp.PortDesign(0.04, 0.10), 20.0) > rp.resonant_frequency_hz(rp.PortDesign(0.015, 0.10),
                                                                                              20.0)


def test_rp_larger_cabinet_lowers_frequency():
    assert rp.resonant_frequency_hz(rp.BASELINE_DESIGN, 80.0) < rp.resonant_frequency_hz(rp.BASELINE_DESIGN, 10.0)


def test_rp_longer_port_lowers_frequency():
    assert rp.resonant_frequency_hz(rp.PortDesign(0.025, 0.25), 20.0) < rp.resonant_frequency_hz(rp.PortDesign(0.025, 0.05),
                                                                                               20.0)


def test_rp_surrogate_input_has_no_label_leakage():
    x, y = rp.generate_dataset(n_samples=10, seed=0)
    assert x.shape[1] == 3 and len(rp.SURROGATE_INPUTS) == 3


def test_rp_optimizer_hits_target_frequency_in_a_realistic_regime():
    sur = _surrogate(rp, 5)
    for vol, target in [(20.0, 35.0), (20.0, 55.0), (50.0, 30.0)]:
        r = rp.optimize_design(sur, vol, target)
        card = rp.verify_design(r.design, vol, target, r.surrogate_predicted_freq_hz)
        assert card.target_relative_error < 0.10, (vol, target, card.real_frequency_hz)
        assert card.overall_verdict in ("PASS", "WARNING")


def test_rp_optimized_design_never_violates_port_bounds():
    sur = _surrogate(rp, 9)
    for vol, target in [(20.0, 35.0), (10.0, 70.0), (50.0, 25.0)]:
        v = rp.constraint_violations(rp.optimize_design(sur, vol, target).design, vol)
        assert v["min_radius"] <= 0 and v["max_radius"] <= 0 and v["min_length"] <= 0 and v["max_length"] <= 0


def test_rp_physically_implausible_combo_is_honestly_flagged_not_faked():
    r = rp.optimize_design(_surrogate(rp, 2), 100.0, 90.0)
    card = rp.verify_design(r.design, 100.0, 90.0, r.surrogate_predicted_freq_hz)
    assert card.overall_verdict == "FAIL" and card.target_relative_error > 0.10


def test_rp_verification_card_coverage_is_explicit_and_partial():
    r = rp.optimize_design(_surrogate(rp, 13), 20.0, 40.0)
    assert 0.0 < rp.verify_design(r.design, 20.0, 40.0, r.surrogate_predicted_freq_hz).coverage < 1.0


def test_rp_pareto_frontier_spans_a_real_range_of_volumes():
    f = rp.sweep_pareto_frontier(_surrogate(rp, 4), 20.0, 40.0, n_points=8)
    vols = [p.port_volume_m3 for p in f]
    assert len(f) == 8 and max(vols) > min(vols) * 1.2


def test_rp_pareto_frontier_weights_are_swept_and_each_point_re_verified_independently():
    f = rp.sweep_pareto_frontier(_surrogate(rp, 7), 20.0, 40.0, n_points=6)
    w = [p.volume_weight for p in f]
    assert w == sorted(w) and w[0] < w[-1]
    for p in f:
        assert abs(p.real_achieved_freq_hz - rp.resonant_frequency_hz(p.design, 20.0)) < 1e-6
        assert p.port_volume_m3 == p.design.port_volume_m3()


def test_rp_pareto_frontier_low_volume_weight_points_stay_reasonably_close_to_target():
    f = rp.sweep_pareto_frontier(_surrogate(rp, 11), 20.0, 40.0, n_points=8)
    assert min(f, key=lambda p: p.volume_weight).real_freq_error_pct < 20.0


def test_rp_pareto_frontier_higher_volume_weight_yields_a_smaller_design_than_lowest_weight():
    f = rp.sweep_pareto_frontier(_surrogate(rp, 3), 20.0, 40.0, n_points=8)
    lo, hi = min(f, key=lambda p: p.volume_weight), max(f, key=lambda p: p.volume_weight)
    assert hi.port_volume_m3 < lo.port_volume_m3


def test_rp_baseline_design_exports_a_valid_nonempty_stl():
    _valid_binary_stl(rp.export_stl_bytes(rp.BASELINE_DESIGN))


def test_rp_longer_wider_port_has_a_measurably_larger_bounding_box():
    xs, ys, zs = meshes.bbox_size(rp.mesh(rp.PortDesign(0.01, 0.03)))
    xb, yb, zb = meshes.bbox_size(rp.mesh(rp.PortDesign(0.05, 0.30)))
    assert zb > zs and xb > xs and yb > ys
