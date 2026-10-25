"""pinneapple_design.surrogate_inverse: the shared machinery and the five cases, fast checks (CI).

The original suites of the five source apps, with their ~20 surrogate trainings, are in
``test_surrogate_inverse_ported.py`` (``PINNEAPPLE_SLOW=1``)."""
from __future__ import annotations

import math
import struct

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinneapple_design.geometry.bodies import _signed_volume, annulus  # noqa: E402
from pinneapple_design.surrogate_inverse import brake_rotor as br  # noqa: E402
from pinneapple_design.surrogate_inverse import core, meshes  # noqa: E402
from pinneapple_design.surrogate_inverse import heatsink_fins as hs  # noqa: E402
from pinneapple_design.surrogate_inverse import reflex_port as rp  # noqa: E402
from pinneapple_design.surrogate_inverse import suspension_arm as sa  # noqa: E402
from pinneapple_design.surrogate_inverse import transdermal_patch as tp  # noqa: E402
from pinneapple_design.thermal.fin_cell import solve_fin_cell  # noqa: E402

torch.set_num_threads(1)          # tiny networks: one thread is fastest, and stable on a busy runner


# -- core ----------------------------------------------------------------------------------------------------------

def test_fit_surrogate_learns_a_power_law_and_reports_held_out_error():
    rng = np.random.default_rng(0)
    X = torch.tensor(rng.uniform([1.0, 1.0], [10.0, 5.0], size=(2000, 2)), dtype=torch.float32)
    y = X[:, 0] ** 2 / X[:, 1]
    fit = core.fit_surrogate(X, y, [(1.0, 10.0), (1.0, 5.0)], epochs=1000, log_output=True, hidden=32, n_layers=2)
    assert fit.n_val == 200 and fit.n_train == 1800
    assert fit.val_mean_rel_err < 0.03 and fit.val_p95_rel_err >= fit.val_mean_rel_err
    with torch.no_grad():
        assert float(fit.model(torch.tensor([[4.0, 2.0]]))[0]) == pytest.approx(8.0, rel=0.05)


def test_fit_surrogate_rejects_bad_inputs():
    with pytest.raises(ValueError):
        core.fit_surrogate(torch.ones(10, 2), torch.ones(10), [(0.0, 1.0)])
    with pytest.raises(ValueError):
        core.fit_surrogate(torch.ones(10, 1), -torch.ones(10), [(0.0, 1.0)], log_output=True)


def test_descend_respects_bounds_in_both_parametrisations():
    def loss(v):
        return (v["a"] - 5.0) ** 2 + (v["b"] + 3.0) ** 2           # optimum outside the box for b

    for reparam, lr in (("clamp", 0.1), ("sigmoid", 0.1)):
        x = core.descend(loss, {"a": 1.0, "b": 0.5}, bounds={"a": (0.0, 10.0), "b": (0.0, 1.0)}, steps=600, lr=lr,
                         reparam=reparam)
        assert x["a"] == pytest.approx(5.0, abs=0.05)
        assert 0.0 <= x["b"] <= 0.01


def test_verification_card_aggregates_and_exposes_a_trust_report():
    checks = [core.banded("err", 0.03, 0.05, 0.15, pass_text="ok", warn_text="w", fail_text="f"),
              core.banded("err2", 0.10, 0.05, 0.15, pass_text="ok", warn_text="w", fail_text="f")]
    card = core.VerificationCard(checks, {"x": 1.0}, covered=["a", "b", "c"], not_covered=["d", "e"])
    assert card.overall_verdict == "WARNING" and card.trust_score == 50 and card.coverage == pytest.approx(0.6)
    assert card.x == 1.0
    tr = card.trust_report()
    assert tr.decisao == "REVIEW" and len(tr.checks) == 4
    card.checks.append(core.CheckResult("bad", core.FAIL, "broken"))
    assert card.overall_verdict == "FAIL" and card.trust_report().decisao == "REJECT"
    d = card.to_dict()
    assert d["trust"]["decision"] == "REJECT" and "Does NOT cover d, e" in d["coverage_note"]


# -- meshes --------------------------------------------------------------------------------------------------------

def test_annulus_is_closed_outward_and_has_the_ring_volume():
    V, F = annulus(1.0, 2.0, 0.5, n=256)
    vol = _signed_volume(V, F)
    assert vol == pytest.approx(math.pi * (4 - 1) * 0.5, rel=1e-3)
    edges = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2)                                       # every edge shared by two triangles: closed


def test_part_meshes_have_the_designed_dimensions():
    x, y, z = meshes.bbox_size(hs.mesh(hs.BASELINE_DESIGN))
    assert (x, y, z) == pytest.approx((80.0, 80.0, 3.0 + 25.0))
    x, y, z = meshes.bbox_size(br.mesh(br.BASELINE_DESIGN))
    assert x == pytest.approx(2 * (754.0 / (2 * math.pi) + 30.0), rel=1e-3) and z == pytest.approx(6 + 8 + 6)
    assert meshes.bbox_size(sa.mesh(sa.BASELINE_DESIGN)) == pytest.approx((50.0, 22.0, 350.0))
    x, _, z = meshes.bbox_size(tp.mesh(tp.BASELINE_DESIGN))
    assert x == pytest.approx(2 * math.sqrt(20e-4 / math.pi) * 1e3, rel=1e-3) and z == pytest.approx(1.08)
    x, _, z = meshes.bbox_size(rp.mesh(rp.BASELINE_DESIGN))
    assert x == pytest.approx(2 * (25 + 6)) and z == pytest.approx(100.0)


def test_stl_export_is_binary_and_counts_triangles():
    stl = sa.export_stl_bytes(sa.BASELINE_DESIGN)
    n = struct.unpack("<I", stl[80:84])[0]
    assert stl[:5] != b"solid" and n == 12 and len(stl) == 84 + 50 * n
    few = struct.unpack("<I", hs.export_stl_bytes(hs.HeatsinkDesign(5, 0.02, 0.001))[80:84])[0]
    many = struct.unpack("<I", hs.export_stl_bytes(hs.HeatsinkDesign(40, 0.02, 0.001))[80:84])[0]
    assert many - few == 35 * 12


# -- physics wrappers and their own analyses ---------------------------------------------------------------------

def test_heatsink_and_rotor_reproduce_the_source_baselines():
    assert hs.junction_temperature_c(hs.BASELINE_DESIGN, 65.0, 35.0) == pytest.approx(48.11806964469042, abs=1e-6)
    assert br.rotor_temperature_c(br.BASELINE_DESIGN, 800.0, 120.0) == pytest.approx(112.4, abs=0.1)
    # T - T_amb is linear in P: the surrogate's target R does not depend on the power
    r1 = hs.junction_temperature_c(hs.BASELINE_DESIGN, 10.0, 35.0) - hs.T_AMBIENT_C
    assert r1 / 10.0 == pytest.approx(hs.thermal_resistance_k_w(hs.BASELINE_DESIGN, 35.0), rel=1e-12)


def test_fin_cell_agrees_with_the_fin_formula_at_the_baseline_and_conserves_energy():
    out = hs.fin_cell_check(hs.BASELINE_DESIGN, 65.0, 35.0)
    assert out["agreement"]["t_junction_relative_diff"] < 0.01
    assert out["finite_volume"]["energy_balance_rel_error"] < 1e-9
    assert out["finite_volume"]["grid_change_rel"] < 1e-3
    assert out["finite_volume"]["t_fin_root_c"] > out["finite_volume"]["t_fin_tip_c"] > hs.T_AMBIENT_C


def test_fin_cell_one_dimensional_limit():
    """Thin copper-like fins (fin Biot number h t / 2k ~ 1e-5): 2-D conduction in the fin adds nothing, so the cell
    and the 1-D formula agree within 2 % once the cell's heat input (P / N over one pitch, i.e. N pitches) is rescaled
    to the formula's footprint (N fins + N - 1 gaps)."""
    r = solve_fin_cell(n_fins=40, fin_thickness_m=0.0005, fin_height_m=0.02, fin_spacing_m=0.0015, depth_m=0.08,
                       base_thickness_m=0.002, k_w_mk=400.0, h_w_m2k=20.0, power_w=40.0)
    from pinneapple_physics.closed_form import fin_array_conduction as fac
    w = 40 * 0.0005 + 39 * 0.0015
    p = fac.fin_array_profile_params(400.0, 20.0, 40, 0.0005, 0.02, 0.08, w, 0.002, 40.0)
    scale = (40 * 0.002) / w                    # N pitches / footprint width
    assert (r.t_junction_c - 25.0) * scale == pytest.approx(p["theta_fin_base_c"] + 40.0 * p["r_base"], rel=0.02)


def test_brake_monte_carlo_verdicts():
    robust = br.monte_carlo_uncertainty(br.BASELINE_DESIGN, 800.0, 120.0, br.MAX_SAFE_ROTOR_TEMP_C, n_samples=200, seed=1)
    near = br.monte_carlo_uncertainty(br.BASELINE_DESIGN, 1480.0, 40.0, br.MAX_SAFE_ROTOR_TEMP_C, n_samples=200, seed=1)
    assert robust["verdict"] == "ROBUST" and robust["safe_fraction"] == 1.0
    assert near["verdict"] in ("MARGINAL", "FRAGILE") and near["p95_c"] > robust["p95_c"]
    lo, hi = near["safe_fraction_ci95"]
    assert 0.0 <= lo <= near["safe_fraction"] <= hi <= 1.0


def test_patch_units_close_and_wear_time_depends_on_the_membrane_only():
    d = tp.BASELINE_DESIGN
    assert tp.release_rate_mg_day(d) == pytest.approx(64.8)                  # mg/day, by hand in the docstring test
    assert tp.release_rate_mg_day(d) * tp.depletion_time_days(d) == pytest.approx(tp.reservoir_mass_mg(d))
    assert tp.depletion_time_days(tp.PatchDesign(5e-4, 80e-6, 5.0)) == pytest.approx(tp.depletion_time_days(d))
    assert tp.depletion_time_days(tp.PatchDesign(20e-4, 200e-6, 60.0)) == pytest.approx(4.63, abs=0.01)


def test_arm_sensitivity_matches_the_power_law_exponents():
    r = sa.compute_sensitivity(0.05, 0.022, 1200.0)
    b = sa.FATIGUE_STRENGTH_EXPONENT
    assert r.elasticity_width == pytest.approx(-1 / b, rel=1e-4) and r.elasticity_thickness == pytest.approx(-2 / b, rel=1e-4)
    assert r.d_mass_d_width == pytest.approx(sa.STEEL_DENSITY_KG_M3 * 0.022 * sa.ARM_LENGTH_M)


def test_port_formula_and_rule():
    assert rp.resonant_frequency_hz(rp.PortDesign(0.025, 0.10), 20.0) == pytest.approx(
        343.0 / (2 * math.pi) * math.sqrt(math.pi * 0.025 ** 2 / (0.02 * (0.10 + 1.7 * 0.025))))
    assert rp.constraint_violations(rp.PortDesign(0.06, 0.35), 5.0)["port_volume_fraction"] > 0


# -- one surrogate per case, the default settings, through the whole loop -----------------------------------------

@pytest.fixture(scope="module")
def heatsink_fit():
    return hs.train_surrogate(n_samples=4000, epochs=1000, seed=0)


def test_heatsink_loop_beats_baseline_and_passes_on_the_rise(heatsink_fit):
    assert heatsink_fit.val_mean_rel_err < 0.05
    for power, h in ((65.0, 35.0), (120.0, 18.0)):
        r = hs.optimize_design(heatsink_fit.model, power, h)
        card = hs.verify_design(r.design, power, h, r.surrogate_predicted_temp_c)
        assert all(v <= 0 for v in hs.constraint_violations(r.design).values())
        assert card.real_temperature_c < hs.junction_temperature_c(hs.BASELINE_DESIGN, power, h) - 5.0
        assert card.relative_error < hs.REL_ERROR_WARN
        assert card.to_dict()["trust"]["decision"] in ("APPROVED", "REVIEW", "REJECT")


def test_patch_loop_hits_target_and_wear():
    fit = tp.train_surrogate(n_samples=4000, epochs=1000, seed=0)
    r = tp.optimize_design(fit.model, 100.0, 2.0)
    card = tp.verify_design(r.design, 100.0, 2.0, r.surrogate_predicted_rate_mg_day)
    assert card.target_relative_error < 0.15 and card.real_depletion_days >= 2.0 * 0.9
