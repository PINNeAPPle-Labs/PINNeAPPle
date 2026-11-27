"""Bayesian optimization has to earn its calls back (#102).

The acceptance is a comparison, not a class: *a loop that needs fewer reference simulations than random sampling,
measured in the benchmark*. These tests hold the pieces — the surrogate predicts, the acquisition points somewhere
sensible, the constrained version respects the constraint — and the comparison itself, with a fixed seed so a
regression shows up as a number that moved.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.bayesopt import (
    BayesianOptimizer,
    BOResult,
    GaussianProcess,
    benchmark_bayesopt,
    branin,
    expected_improvement,
    hartmann3,
    probability_of_improvement,
    upper_confidence_bound,
)


# ── the surrogate ──────────────────────────────────────────────────────────
def test_the_gaussian_process_interpolates_and_is_uncertain_away_from_the_data():
    X = np.linspace(0, 1, 8)[:, None]
    y = np.sin(3 * X[:, 0])
    gp = GaussianProcess().fit(X, y)
    mu, sd = gp.predict(X)
    np.testing.assert_allclose(mu, y, atol=2e-2)
    _, far = gp.predict(np.array([[5.0]]))
    assert far[0] > sd.max(), "a point far from the data must be less certain than the data themselves"


def test_the_surrogate_does_not_care_about_the_scale_of_the_output():
    """A surrogate on numbers of order 1e5 and one of order 1 get the same prior: the fit standardises y."""
    X = np.linspace(0, 1, 12)[:, None]
    f = np.sin(4 * X[:, 0])
    small, big = GaussianProcess().fit(X, f), GaussianProcess().fit(X, 1e5 * f + 3e5)
    a, _ = small.predict(np.array([[0.33]]))
    b, _ = big.predict(np.array([[0.33]]))
    assert b[0] == pytest.approx(1e5 * a[0] + 3e5, rel=1e-6)


def test_a_gaussian_process_that_was_not_fitted_says_so():
    with pytest.raises(RuntimeError, match="not fitted"):
        GaussianProcess().predict(np.zeros((1, 2)))


# ── the acquisition functions ──────────────────────────────────────────────
def test_expected_improvement_is_zero_where_nothing_can_improve_and_large_where_it_can():
    ei = expected_improvement(np.array([50.0, 0.9]), np.array([0.01, 0.2]), best=1.0)
    assert ei[0] < 1e-3, "a point far worse than the best, with no spread, is not worth a call"
    assert ei[1] > 0.05, "a point just below the best, with spread, is worth a call"


def test_the_three_acquisitions_disagree_in_the_way_they_are_meant_to():
    """Same mean, different spread: UCB is drawn to the uncertain point, PI to the one it is sure is better."""
    mu, sd = np.array([1.0, 1.0]), np.array([0.1, 3.0])
    ucb = upper_confidence_bound(mu, sd, kappa=2.0)
    assert ucb[1] > ucb[0], "UCB for a minimisation rewards the uncertain point"
    pi = probability_of_improvement(mu, sd, best=1.5)
    assert pi[0] > pi[1], "PI rewards the point the surrogate is confident about"


# ── the loop ───────────────────────────────────────────────────────────────
def test_the_loop_finds_the_branin_minimum_within_a_small_budget():
    result = BayesianOptimizer(branin, [(-5, 10), (0, 15)], n_initial=5, n_iter=19, seed=0).run()
    assert isinstance(result, BOResult) and result.n_evals == 24
    assert result.best_y < 1.0, result.best_y
    assert result.evals_to_reach(1.0) is not None
    assert all(a >= b for a, b in zip(result.best_after, result.best_after[1:], strict=False)), \
        "the running best never gets worse (the list is a running minimum)"


def test_it_maximises_when_asked():
    peak = lambda X: -((X[:, 0] - 0.3) ** 2 + (X[:, 1] + 0.2) ** 2)          # noqa: E731 - a one-line objective
    result = BayesianOptimizer(peak, [(-1, 1), (-1, 1)], n_initial=6, n_iter=20, maximize=True, seed=0).run()
    assert result.best_y > -0.01 and np.allclose(result.best_x, [0.3, -0.2], atol=0.25)
    assert result.evals_to_reach(-0.01) is not None


def test_a_batch_evaluates_distinct_points():
    seen = []
    result = BayesianOptimizer(lambda X: (seen.append(len(X)) or branin(X)), [(-5, 10), (0, 15)],
                               n_initial=4, n_iter=3, batch_size=2, seed=1).run()
    assert result.n_evals == 4 + 3 * 2
    assert seen[0] == 4 and all(n == 1 for n in seen[1:])
    points = result.X[4:]
    assert len({tuple(np.round(p, 9)) for p in points}) == len(points)


def test_a_constraint_keeps_the_answer_on_the_feasible_side():
    """The unconstrained minimum sits at x = 1; the constraint says x <= 0.2, and the loop has to obey it."""
    objective = lambda X: (X[:, 0] - 1.0) ** 2 + 0.01 * X[:, 1] ** 2         # noqa: E731
    result = BayesianOptimizer(objective, [(-1, 1), (-1, 1)], n_initial=6, n_iter=12, seed=0,
                               constraints=[(lambda x: float(x[0]), 0.2)]).run()
    feasible = [p for p, v in zip(result.X, result.y, strict=True) if p[0] <= 0.2 + 1e-9]
    assert feasible, "the loop never evaluated a feasible point"
    assert result.best_x[0] <= 0.2 + 0.15, f"best_x = {result.best_x} ignores the constraint"


@pytest.mark.parametrize("kwargs, message", [
    ({"bounds": [(1, 0)]}, "increasing"),
    ({"bounds": []}, "at least one bound"),
    ({"bounds": [(0, 1)], "acquisition": "magic"}, "unknown acquisition"),
    ({"bounds": [(0, 1)], "constraints": [lambda x: x[0]]}, "callable, limit"),
    ({"bounds": [(0, 1)], "n_initial": 0, "n_iter": 0}, "budget"),
])
def test_it_refuses_instead_of_guessing(kwargs, message):
    with pytest.raises(ValueError, match=message):
        BayesianOptimizer(lambda X: X[:, 0], **kwargs)


def test_an_objective_that_returns_the_wrong_number_of_values_is_refused():
    with pytest.raises(ValueError, match="values for"):
        BayesianOptimizer(lambda X: np.zeros(3), [(0, 1), (0, 1)], n_initial=2, n_iter=2).run()


def test_the_result_says_how_many_calls_until_a_target_and_none_when_it_never_got_there():
    result = BOResult(X=np.zeros((3, 1)), y=np.array([5.0, 3.0, 4.0]), best_x=np.zeros(1), best_y=3.0, n_initial=1)
    assert result.best_after == [5.0, 3.0, 3.0]
    assert result.evals_to_reach(3.5) == 2 and result.evals_to_reach(1.0) is None
    assert result.to_dict()["n_evals"] == 3


# ── the comparison the issue asks for ──────────────────────────────────────
def test_the_loop_needs_fewer_reference_calls_than_random_sampling():
    """The acceptance, on Branin at a budget of 24: success rate first, calls second, final value third."""
    report = benchmark_bayesopt("branin", budget=24, seeds=(0, 1, 2))
    assert report["n_seeds"] == 3 and report["budget"] == 24
    assert report["bo"]["reached_rate"] >= report["random"]["reached_rate"]
    assert report["bo"]["median_best"] <= report["random"]["median_best"]
    assert report["fewer_calls"] is True
    assert len(report["rows"]) == 3


def test_the_benchmark_refuses_a_problem_it_does_not_have():
    with pytest.raises(ValueError, match="unknown problem"):
        benchmark_bayesopt("rosenbrock")


def test_hartmann_is_the_function_it_claims_to_be():
    assert hartmann3(np.array([[0.114614, 0.555649, 0.852547]]))[0] == pytest.approx(-3.86278, abs=1e-5)
    assert branin(np.array([[np.pi, 2.275], [9.42478, 2.475]])) == pytest.approx([0.397887, 0.397887], abs=1e-4)
