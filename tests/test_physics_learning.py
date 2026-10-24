"""Physics learning verification: controls pass, every deliberately defective model is caught (and the usual metrics
miss the causal defects), the informative points, the sufficiency check, and architecture-vs-learned provenance."""
import numpy as np
import pytest

from pinneapple_analysis.validation.physics_learning import (BASELINE_LIMITS, baselines, fv_residual, select_points,
                                                             verify)
from pinneapple_analysis.validation.physics_learning_cases import (DEFAULT, WIDTH, Model, Section, defective_models,
                                                                   good_models)


@pytest.fixture(scope="module")
def case():
    return Section()


def test_reference_satisfies_its_own_discrete_equation(case):
    ref = good_models(case)[0]
    rows = np.arange(case.nx * sum(case.nz))
    assert fv_residual(ref, case, dict(DEFAULT), rows).max() < 1e-8
    assert abs(case.outflow(DEFAULT) / DEFAULT["P"] - 1) < 1e-9          # heat balance of the reference


def test_controls_pass(case):
    for m in good_models(case, amplitudes=(0.001, 0.003), modes=(1, 2)):
        r = verify(m, case)
        assert not r.detected, (m.name, [f.line() for f in r.failed])
        assert r.trust.decisao == "APPROVED"


def test_every_defect_is_caught_and_the_usual_metrics_miss_the_causal_ones(case):
    missed_by_metrics = []
    for m in defective_models(case):
        r = verify(m, case)
        assert r.detected and r.trust.decisao == "REJECT", m.name
        b = baselines(m, case, n_theta=6, n_points=200)
        if b["heldout_rel_l2"] <= BASELINE_LIMITS["heldout_rel_l2"] and \
                b["physics_loss"] <= BASELINE_LIMITS["physics_loss"]:
            missed_by_metrics.append(m.defect)
    assert {"out of distribution", "reversed causal response", "spurious correlation"} <= set(missed_by_metrics)


def test_the_right_check_fails(case):
    by = {m.defect: m for m in defective_models(case)}
    failed = {d: {f.check for f in verify(m, case).failed} for d, m in by.items()}
    assert "interface_flux" in failed["interface flux"]
    assert "intervention:TIM conductivity x2" in failed["reversed causal response"]
    assert "intervention:cooling x2" in failed["spurious correlation"]
    assert "energy_balance" in failed["energy balance"]
    assert "boundary:adiabatic_sides" in failed["boundary condition"]
    assert "scaling" in failed["scaling law"]
    assert "intervention:source outside the box" in failed["out of distribution"]


def test_out_of_scope_causes_and_sufficiency(case):
    ref = good_models(case)[0]                       # takes P, x_s, k_tim, h -- never R_c, sigma, k_die, k_lid
    r = verify(ref, case)
    assert [f.check for f in r.out_of_scope] == ["intervention:contact resistance added", "sufficiency"]
    assert r.insufficient and all("missing" in p for p in r.insufficient)
    full = Model("reference, every parameter", lambda X, th: case.evaluate(th, X),
                 inputs=("P", "x_s", "k_tim", "h", "R_c", "k_die", "k_lid", "sigma"))
    r2 = verify(full, case)
    assert not r2.out_of_scope and not r2.insufficient


def test_informative_points_cover_the_physics(case):
    X, why = select_points(case, dict(DEFAULT), k=24)
    assert len(X) == 24 and {"near an interface", "near the source", "near a boundary"} <= set(why)
    assert (X[:, 0] >= 0).all() and (X[:, 0] <= WIDTH).all()


def test_architecture_versus_learned_provenance(case):
    pytest.importorskip("torch")
    from pinneapple_analysis.validation.physics_learning_cases import pinn_model
    soft, _ = pinn_model(case, adam_steps=60, lbfgs_steps=0, n_collocation=400)
    hard, _ = pinn_model(case, hard_sides=True, adam_steps=60, lbfgs_steps=0, n_collocation=400)
    side = {m.name: next(f for f in verify(m, case, k=12).findings if f.check == "boundary:side_gradient")
            for m in (soft, hard)}
    assert side[hard.name].provenance == "architecture" and side[hard.name].passed
    assert side[soft.name].provenance in ("learned", "not demonstrated")
