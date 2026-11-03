"""recommend_architecture says when no learned model is needed, from the measured cost per solve and the study's
evaluations, and lists the checks the chosen family must pass (#1176)."""
from __future__ import annotations

import pytest

from pinneapple_analysis.verification.architecture_recommendation import recommend_architecture


def _codes(rec):
    return [c["code"] for c in rec.checks]


def test_a_closed_form_with_ten_thousand_evaluations_needs_no_model():
    rec = recommend_architecture(solve_seconds=1e-5, n_evaluations=10_000, output_kind="scalar", n_inputs=4)
    assert rec.recommended[0] == "none"
    assert "0.1 s" in rec.decision_path[0] and "no learned model" in rec.decision_path[0]
    assert _codes(rec) == ["solver.verified"] and rec.recommended[1] == "kpi_regressor"   # if the study grows


def test_fewer_evaluations_than_training_samples_needs_no_model():
    rec = recommend_architecture(solve_seconds=30.0, n_evaluations=20, output_kind="scalar", n_inputs=3,
                                 direct_budget_s=60.0)
    assert rec.recommended[0] == "none" and "no more than the ~30 solves" in rec.decision_path[0]


def test_a_few_smooth_scalars_with_tens_of_samples_go_to_a_gaussian_process():
    rec = recommend_architecture(n_high_fidelity_simulations=40, output_kind="scalar", n_inputs=3,
                                 needs_parameter_generalization=True, solve_seconds=60.0, n_evaluations=5000,
                                 accuracy_target=0.05)
    assert rec.recommended[0] == "kpi_regressor"
    assert _codes(rec) == ["surrogate.held_out_error", "surrogate.no_extrapolation",
                           "surrogate.calibrated_uncertainty"]
    assert rec.checks[0]["limit"] == 0.05


def test_a_parametric_field_on_one_grid_with_many_solves_goes_to_a_grid_operator():
    rec = recommend_architecture(n_high_fidelity_simulations=400, output_kind="grid_field",
                                 needs_parameter_generalization=True, solve_seconds=20.0, n_evaluations=10_000)
    assert rec.recommended[0] == "fno" and {"deeponet", "pino", "pod_rom"} <= set(rec.recommended)
    assert "surrogate.resolution_invariance" in _codes(rec)


def test_a_varying_mesh_goes_to_a_graph_network():
    rec = recommend_architecture(n_high_fidelity_simulations=300, output_kind="mesh_field", geometry_varies=True,
                                 needs_parameter_generalization=True)
    assert rec.recommended[0] == "mesh_graph_net" and "point_cloud_operator" in rec.recommended


def test_physics_informed_models_must_pass_a_residual_check():
    rec = recommend_architecture(n_high_fidelity_simulations=0)
    assert rec.recommended[0] == "pinn" and "surrogate.physics_residual" in _codes(rec)


def test_without_the_new_inputs_the_recommendation_is_unchanged():
    rec = recommend_architecture(n_high_fidelity_simulations=400, needs_parameter_generalization=True)
    assert rec.recommended[:3] == ["fno", "deeponet", "pino"] and "none" not in rec.recommended
    with pytest.raises(ValueError, match="output_kind"):
        recommend_architecture(output_kind="volume")
