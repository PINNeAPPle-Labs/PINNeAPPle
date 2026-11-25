"""Surrogate in the loop with a verification budget (#297).

The contract of :func:`pinneapple_analysis.surrogate_loop.loop` on cheap problems, and the measured case on the 2-D
airfoil of :func:`~pinneapple_analysis.surrogate_loop.airfoil_solver` whose optimum is known in closed form
(:func:`~pinneapple_analysis.surrogate_loop.airfoil_reference`) — so the loop is compared against the truth, not
against the best of a finer sweep.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.surrogate_loop import (
    AIRFOIL_SPACE,
    Constraint,
    EnsembleSurrogate,
    GaussianProcessSurrogate,
    airfoil_reference,
    airfoil_solver,
    benchmark_loop,
    loop,
)


# --------------------------------------------------------------------------- a cheap problem with a known answer
def bowl(X):
    """``f = (x0 - 0.2)^2 + (x1 - 0.8)^2`` under ``x1 <= 0.5``.

    The unconstrained optimum (0.2, 0.8) is outside the ceiling, so the constrained optimum is (0.2, 0.5) with
    ``f = 0.09`` and the constraint active — an answer known in closed form, which is what the loop is asked to find.
    """
    X = np.atleast_2d(np.asarray(X, float))
    f = (X[:, 0] - 0.2) ** 2 + (X[:, 1] - 0.8) ** 2
    return np.stack([f, X[:, 1]], axis=1)


BOWL_SPACE = [(0.0, 1.0), (0.0, 1.0)]
BOWL_CONS = [Constraint("x1_max", 0.5, sense="<=")]


def _small(**kw):
    cfg = dict(budget=14, n_initial=5, batch=2, n_pool=256, patience=3, seed=0)
    cfg.update(kw)
    return cfg


# --------------------------------------------------------------------------- the loop's contract
def test_the_loop_returns_a_verified_optimum_within_the_budget():
    res = loop(BOWL_SPACE, GaussianProcessSurrogate(2, 2, seed=0), bowl, constraints=BOWL_CONS, **_small())
    assert res.solver_calls <= res.budget
    assert res.best is not None and res.best.status == "verified"
    for c in BOWL_CONS:
        assert c.ok(res.best.constraints[c.name]), (c.name, res.best.constraints)
    assert res.names == ["x0", "x1"]
    # the constrained optimum sits on the ceiling at (0.2, 0.5): f = 0.09
    assert res.best.objective == pytest.approx(0.09, abs=0.03), res.best.to_dict()
    report = res.report()
    assert report["solver_calls"] == res.solver_calls <= report["budget"]
    assert report["n_verified"] + report["n_certified"] == report["n_designs"]


def test_every_design_carries_its_prediction_and_its_rejection_reason():
    res = loop(BOWL_SPACE, GaussianProcessSurrogate(2, 2, seed=0), bowl, constraints=BOWL_CONS, **_small())
    verified = res.verified
    assert len(verified) == res.solver_calls
    assert any(d.error is not None for d in verified[1:]), "a verified design after the opening one has an error"
    infeasible = [d for d in res.designs if not d.feasible]
    assert infeasible, "the opening design is not constrained, so some design must miss s >= 0.7"
    for d in infeasible:
        assert "x1_max" in d.reason and "limit" in d.reason and ">" in d.reason
    # the predictions are recorded next to the truth, so the per-round error trend is a measurement
    trend = [r.surrogate_error for r in res.rounds if r.surrogate_error is not None]
    assert trend and trend[-1] <= trend[0] + 1e-6, trend


def test_the_loop_stops_when_the_verified_optimum_stops_moving_and_says_how_much_it_spent():
    res = loop(BOWL_SPACE, GaussianProcessSurrogate(2, 2, seed=0), bowl, constraints=BOWL_CONS,
               budget=40, n_initial=5, batch=2, n_pool=256, patience=2, min_improve=1e-2, seed=0)
    assert res.stop_reason == "stabilised"
    assert res.stabilised_round is not None and res.stabilised_calls is not None
    assert res.stabilised_calls <= res.solver_calls < res.budget, "it stopped on the rule, not on the budget"
    assert res.stabilised_calls >= res.rounds[0].n_solver


def test_the_budget_is_never_exceeded_and_the_batch_is_respected():
    res = loop(BOWL_SPACE, GaussianProcessSurrogate(2, 2, seed=0), bowl, constraints=BOWL_CONS,
               budget=7, n_initial=5, batch=3, n_pool=256, seed=0)
    assert res.solver_calls <= 7
    assert res.rounds[0].n_solver == 5, "the opening design is n_initial calls, not a batch"
    assert all(r.n_solver <= 3 for r in res.rounds[1:]), [r.n_solver for r in res.rounds]


def test_maximise_flips_the_objective_and_keeps_its_own_sign():
    def hill(X):
        X = np.atleast_2d(np.asarray(X, float))
        return 1.0 - (X[:, 0] - 0.7) ** 2                         # maximise, peak at 0.7

    res = loop([(0.0, 1.0)], GaussianProcessSurrogate(1, 1, seed=0), hill, maximise=True, budget=10, n_initial=4,
               batch=2, n_pool=128, seed=0)
    assert res.best.objective > 0.95 and abs(res.best.x[0] - 0.7) < 0.06


def test_an_ensemble_surrogate_feeds_the_same_loop_from_its_own_spread():
    res = loop(BOWL_SPACE, EnsembleSurrogate(2, 2, n_members=3, hidden=16, epochs=200, seed=0), bowl,
               constraints=BOWL_CONS, **_small())
    assert res.best is not None and res.best.objective == pytest.approx(0.09, abs=0.05)


@pytest.mark.parametrize("acquisition", ["ei", "lcb"])
def test_both_acquisitions_run_and_report(acquisition):
    res = loop(BOWL_SPACE, GaussianProcessSurrogate(2, 2, seed=0), bowl, constraints=BOWL_CONS,
               acquisition=acquisition, **_small())
    assert res.acquisition == acquisition and res.report()["acquisition"] == acquisition
    assert res.best is not None


def test_certify_skips_the_solver_for_designs_the_router_can_settle_and_still_reports_a_verified_optimum():
    plain = loop(BOWL_SPACE, GaussianProcessSurrogate(2, 2, seed=0), bowl, constraints=BOWL_CONS, **_small())
    routed = loop(BOWL_SPACE, GaussianProcessSurrogate(2, 2, seed=0), bowl, constraints=BOWL_CONS,
                  certify=True, tolerance=0.05, **_small())
    certified = [d for d in routed.designs if d.status == "certified"]
    assert certified, "with a tolerance of 0.05 the bowl's designs are settled by the surrogate"
    assert all(d.reason for d in certified)
    assert any("certified by the router" in d.reason for d in certified if d.feasible)
    assert all(d.objective == d.predicted_objective for d in certified)
    assert routed.solver_calls <= plain.solver_calls
    assert routed.best is not None and routed.best.status == "verified"


def test_the_contract_refuses_what_it_cannot_do():
    class Silent:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros((len(X), 1)), np.zeros((len(X), 1))

    with pytest.raises(ValueError):
        loop(BOWL_SPACE, GaussianProcessSurrogate(2, 2), bowl, constraints=BOWL_CONS, budget=2, n_initial=2)
    with pytest.raises(ValueError):
        loop([(1.0, 0.0)], GaussianProcessSurrogate(1, 2), bowl, constraints=BOWL_CONS, budget=4, n_initial=2)
    with pytest.raises(ValueError):
        # the solver answers one output but the loop was promised two (objective + one constraint)
        loop(BOWL_SPACE, GaussianProcessSurrogate(2, 2), lambda X: np.zeros(len(np.atleast_2d(X))),
             constraints=BOWL_CONS, budget=4, n_initial=2)
    with pytest.raises(ValueError):
        # a surrogate that predicts fewer outputs than the loop reads is refused, not silently scored on zeros
        loop(BOWL_SPACE, Silent(), bowl, constraints=BOWL_CONS, budget=6, n_initial=3, batch=1)


# --------------------------------------------------------------------------- the measured case
@pytest.fixture(scope="module")
def airfoil():
    airfoil_solver(np.array([[0.02, 0.4, 0.18]]))            # warm the lazy imports out of the measurement
    return benchmark_loop(budget=16, n_initial=6, repeats=2, seed=0)


def test_airfoil_loop_reaches_the_known_optimum_and_the_fixed_design_does_not(airfoil):
    """The number the issue asks for: how many expensive evaluations before the verified optimum stopped moving."""
    reference = airfoil_reference()
    near = airfoil["loop"]
    fixed = airfoil["fixed"]
    assert near["gap_to_reference_pct"] < 1.0, near                  # measured 0.03 % (0.0087164 vs 0.0087139)
    assert fixed["gap_to_reference_pct"] > near["gap_to_reference_pct"]
    assert airfoil["gain"] >= 1.0                                   # fixed / loop: above 1 the loop wins
    assert all(c is not None and c <= airfoil["budget"] for c in near["solver_calls"])
    assert all(s is not None for s in near["solver_calls_to_stabilise"]), near["solver_calls_to_stabilise"]
    assert reference == pytest.approx(0.008714, rel=1e-4)


def test_airfoil_the_verified_optimum_is_the_corner_the_physics_expects():
    """Thinnest allowed section, and camber just enough: the constraint is active at the box's lower bound."""
    res = loop(AIRFOIL_SPACE, GaussianProcessSurrogate(3, 2, seed=0), airfoil_solver, budget=14, n_initial=6,
               batch=2, constraints=[Constraint("cl", 0.75)], n_pool=512, patience=3, seed=0, verbose=False)
    assert res.best is not None
    x = dict(zip(res.names, res.best.x, strict=True))
    assert x["thickness"] == pytest.approx(0.08, abs=0.01), x
    assert 0.7 <= res.best.constraints["cl"] <= 1.1, res.best.constraints
    assert res.best.objective <= 1.02 * airfoil_reference()
    assert res.solver_calls <= 14


def test_airfoil_a_binding_constraint_widens_the_gap_between_the_loop_and_a_fixed_design():
    """With ``Cl >= 1.1`` most of the box is infeasible; the acquisition is what keeps the loop out of it."""
    out = benchmark_loop(budget=12, n_initial=6, repeats=1, seed=1, cl_target=1.1)
    assert out["loop"]["best_cd"] < out["fixed"]["best_cd"]
    assert out["gain"] > 1.0
    assert out["loop"]["solver_calls"][0] <= 12
