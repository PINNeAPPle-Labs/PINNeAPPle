"""Robustness studies: real (tiny) trainings, checking structure and that
the numbers are finite and consistent -- not that a 5-epoch model is good."""
import math

import pinneapple_physics as pp
import pinneapple_neural.architectures  # noqa: F401
from pinneapple_neural.architectures.registry import ModelRegistry

from pinneapple_veriphysics import robustness as rb
from pinneapple_veriphysics.applicability import FAIL, build_applicability_map

SPEC = pp.get_preset("burgers_1d")


def _build():
    return ModelRegistry.build("vanilla_pinn", in_dim=2, out_dim=1, hidden_dim=16, n_layers=2)


def test_studies_return_finite_measured_numbers_and_feed_the_map():
    best = rb._train(SPEC, _build, epochs=5, n_collocation=32, seed=0)
    ens = rb.ensemble_study(SPEC, _build, best, n_members=1, epochs=5, n_collocation=32, n_eval=64)
    assert ens["n_members"] == 2 and math.isfinite(ens["interior_rel_spread"])
    assert set(ens["shell_rel_spread"]) == set(rb.DEFAULT_MARGINS)
    ext = rb.extrapolation_study(SPEC, best, residual_threshold=1e9, ensemble=None, n_eval=64)
    assert ext["supported_margin"] == max(rb.DEFAULT_MARGINS)  # absurdly loose threshold -> all shells ok
    tight = rb.extrapolation_study(SPEC, best, residual_threshold=0.0, n_eval=64)
    assert tight["supported_margin"] == 0.0
    ws = rb.weight_sensitivity_study(SPEC, _build, best, epochs=5, n_collocation=32, n_eval=64)
    assert len(ws["variants"]) == 2 and math.isfinite(ws["max_rel_change"])

    m = build_applicability_map(
        architecture="vanilla_pinn", coords=list(SPEC.coords), fields=list(SPEC.fields),
        domain_bounds=dict(SPEC.domain_bounds), physical_parameters=None,
        guardrail_checks=[], confidence_components=[],
        extrapolation=tight, ensemble={k: v for k, v in ens.items() if k != "members"}, weight_sensitivity=ws)
    by = {i.key: i for i in m.checklist}
    assert by["generalization"].status == FAIL and by["generalization"].tier == "VERIFIED"
    assert "UNCALIBRATED" in by["uq"].summary
    assert by["weights"].tier == "VERIFIED"


def test_shell_points_are_outside_the_box():
    x = rb._sample_shell(SPEC.domain_bounds, list(SPEC.coords), 0.25, 200)
    (xl, xh), (tl, th) = SPEC.domain_bounds["x"], SPEC.domain_bounds["t"]
    inside = (x[:, 0] >= xl) & (x[:, 0] <= xh) & (x[:, 1] >= tl) & (x[:, 1] <= th)
    assert not inside.any() and len(x) == 200


def test_condition_coverage_matches_the_installed_solver_sampler():
    cov = rb.condition_coverage(SPEC, n=20_000)
    old_solver = "uniform" in cov["sampler"]
    # Older solve_pde silently dropped burgers_1d's isclose() selectors; current main samples box faces.
    assert (set(cov["unenforced"]) == {"u_init", "bc_left", "bc_right"}) if old_solver else not cov["unenforced"]
    m = build_applicability_map(
        architecture="vanilla_pinn", coords=list(SPEC.coords), fields=list(SPEC.fields),
        domain_bounds=dict(SPEC.domain_bounds), physical_parameters=None,
        guardrail_checks=[], confidence_components=[], condition_coverage=cov)
    item = next(i for i in m.checklist if i.key == "conditions_enforced")
    assert item.status == (FAIL if old_solver else "PASS")


def test_parameter_shift_study_and_map():
    best = rb._train(SPEC, _build, epochs=5, n_collocation=32, seed=0)
    make = lambda **kw: pp.get_preset("burgers_1d", **kw)  # noqa: E731
    ps = rb.parameter_shift_study(make, {"nu": 0.01}, _build, best, residual_threshold=1e9,
                                  epochs=5, n_collocation=32, shifts=(0.1, 0.3), n_eval=64)
    assert [p["parameter"] for p in ps["parameters"]] == ["nu"]
    row = ps["parameters"][0]["shifts"][0]
    assert math.isfinite(row["rel_prediction_error"]) and row["reference_trusted"]
    m = build_applicability_map(
        architecture="vanilla_pinn", coords=list(SPEC.coords), fields=list(SPEC.fields),
        domain_bounds=dict(SPEC.domain_bounds), physical_parameters={"kinematic_viscosity": 0.01},
        guardrail_checks=[], confidence_components=[], parameter_shift=ps)
    gen = next(i for i in m.checklist if i.key == "generalization")
    assert gen.tier == "VERIFIED" and any("nu +10%" in e for e in gen.evidence)
    assert any(v.name == "nu" and v.kind == "parameter" for v in m.envelope)
    assert any("Geometry changes" in u for u in m.unsupported)
