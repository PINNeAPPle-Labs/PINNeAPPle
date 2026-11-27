"""The three robust formulations against their closed-form optima, and the Pareto front that carries its bands (#80).

Two kinds of check, which is what the issue asks for:

- **the optimum is known**: ``mean + k sigma`` on a trade-off is ``x* = (2/k)^(2/3)``; a chance constraint with a
  Gaussian output is ``x* = (limit + z sigma) / a`` with ``z = Phi^-1(1 - alpha)``; the reliability-based design is
  the same design as the chance constraint with ``p_f = alpha``;
- **the probability is measured, not promised**: :func:`verify_robust` counts violations of 200,000 draws at the
  optimum, and the tests decide on the binomial band of that count, not on the formula alone.

For the front, the probability that a design keeps its place is checked against an integral computed here, with
``scipy.integrate.quad``, independently of the Monte Carlo inside the module.

Measured on this machine (``2026-11-25``): ``mean + k sigma`` at k = 0.5, 1 and 2 returns 2.5198, 1.5874 and 1.0000
against exactly the same values; the chance constraint at ``alpha = 0.05`` returns 0.91121 against 0.91121, and the
Monte Carlo at that design gives a violation rate of 0.0499 (band 0.0489-0.0509) for a target of 0.05.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

pytest.importorskip("scipy")

from pinneapple_design.design_optimizer.robust import (  # noqa: E402
    RobustPareto,
    chance_constrained,
    mean_plus_k_sigma,
    pareto_with_uncertainty,
    reliability_based,
    verify_robust,
)


# ------------------------------------------------------------------------------ mean + k sigma
@pytest.mark.parametrize("k", [0.5, 1.0, 2.0])
def test_mean_plus_k_sigma_matches_the_closed_form(k):
    """mu = 1/x (bigger is cheaper) against sigma = sqrt(x) (bigger is more uncertain): x* = (2/k)^(2/3)."""
    sol = mean_plus_k_sigma(lambda x: 1.0 / float(x[0]), lambda x: float(np.sqrt(max(x[0], 1e-12))),
                            [(0.05, 20.0)], k=k, x0=[2.0])
    assert sol.formulation == "mean_plus_k_sigma"
    assert sol.x[0] == pytest.approx((2.0 / k) ** (2.0 / 3.0), rel=1e-4)
    assert sol.objective == pytest.approx(1.0 / sol.x[0] + k * np.sqrt(sol.x[0]), rel=1e-9)
    assert sol.mean == pytest.approx(1.0 / sol.x[0]) and sol.sigma == pytest.approx(np.sqrt(sol.x[0]))


def test_without_a_trade_off_the_penalty_changes_nothing():
    """The formulation is honest about its own limits: if the mean gets better everywhere, k does not move the optimum."""
    a = mean_plus_k_sigma(lambda x: float(x[0]), lambda x: 0.1, [(0.0, 5.0)], k=0.0, x0=[1.0])
    b = mean_plus_k_sigma(lambda x: float(x[0]), lambda x: 0.1 * x[0], [(0.0, 5.0)], k=3.0, x0=[1.0])
    assert a.x[0] == pytest.approx(0.0, abs=1e-6) and b.x[0] == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------------------------ chance constraint
def _chance_problem(a=2.0, s=0.5, limit=1.0, alpha=0.05):
    return dict(mean=lambda x: a * float(x[0]), sigma=lambda x: s, cost=lambda x: float(x[0]),
                alpha=alpha, limit=limit, bounds=[(0.0, 10.0)])


def test_chance_constraint_matches_the_closed_form():
    p = _chance_problem()
    sol = chance_constrained(**p)
    z = norm.ppf(1 - p["alpha"])
    exact = (p["limit"] + z * 0.5) / 2.0
    assert sol.x[0] == pytest.approx(exact, rel=1e-4)
    assert sol.margin == pytest.approx(z, rel=1e-3)                 # the constraint binds at the reliability index
    assert sol.requirement["target_probability"] == pytest.approx(0.95)
    assert sol.objective == pytest.approx(exact, rel=1e-6)


def test_the_monte_carlo_at_the_optimum_shows_the_promised_probability():
    p = _chance_problem()
    sol = chance_constrained(**p)
    v = verify_robust(sol, p["mean"], p["sigma"], n=200_000, seed=0)
    assert v["probability_of_violation"] == pytest.approx(p["alpha"], abs=0.005)
    assert v["within_target"] and v["band"][0] < p["alpha"] < v["band"][1]
    assert json.dumps(sol.to_dict())                                 # the record carries the check


def test_a_tighter_target_buys_a_larger_design():
    tight = chance_constrained(**_chance_problem(alpha=0.001))
    loose = chance_constrained(**_chance_problem(alpha=0.20))
    assert tight.x[0] > loose.x[0]
    assert tight.margin > loose.margin
    assert tight.margin == pytest.approx(norm.ppf(1 - 0.001), rel=1e-3)


def test_reliability_based_design_is_the_same_design_as_the_chance_constraint():
    """The same requirement written as a probability of failure must not change the design."""
    p = _chance_problem()
    chance = chance_constrained(**p)
    rbd = reliability_based(p["mean"], p["sigma"], target_pf=p["alpha"], cost=p["cost"], limit=p["limit"],
                            bounds=p["bounds"])
    assert rbd.formulation == "reliability_based_design"
    assert rbd.x[0] == pytest.approx(chance.x[0], abs=1e-4)
    assert rbd.requirement["beta_target"] == pytest.approx(chance.margin, rel=1e-3)
    v = verify_robust(rbd, p["mean"], p["sigma"], n=100_000, seed=1)
    assert v["probability_of_violation"] <= p["alpha"] + 1.96 * v["standard_error"] + 1e-9


def test_the_contract_rejects_what_it_cannot_do():
    with pytest.raises(ValueError, match="alpha must be"):
        chance_constrained(**_chance_problem(alpha=0.0))
    with pytest.raises(ValueError, match="give the limit"):
        chance_constrained(mean=lambda x: 1.0, sigma=lambda x: 1.0, cost=lambda x: 1.0)
    with pytest.raises(ValueError, match="k must be"):
        mean_plus_k_sigma(lambda x: 1.0, lambda x: 1.0, [(0, 1)], k=-1.0)


# ------------------------------------------------------------------------------ the front
def test_the_front_reports_the_nominal_and_the_worst_case_one():
    """Four designs, all nondominated on the means; the fourth is uncertain enough to drop out of mean + 2 sigma."""
    Y = np.array([[1.0, 4.0], [2.0, 2.5], [3.0, 1.0], [2.6, 1.4]])
    S = np.array([[0.05, 0.05], [0.05, 0.05], [0.05, 0.05], [0.4, 0.1]])
    p = pareto_with_uncertainty(Y, S, k=2.0, n_draws=20000, seed=0)
    assert isinstance(p, RobustPareto)
    assert p.nominal.tolist() == [True, True, True, True]
    assert p.robust.tolist() == [True, True, True, False]          # (3.4, 1.6) is beaten by (3.1, 1.1)
    lo, hi = p.band
    np.testing.assert_allclose(lo, Y - 2.0 * S)
    np.testing.assert_allclose(hi, Y + 2.0 * S)
    assert p.p_on_front[3] == pytest.approx(0.84, abs=0.03)        # it keeps its place most of the time anyway
    assert p.p_on_front[[0, 2]].min() > 0.995                       # the far ends of the front never move
    assert json.dumps(p.to_dict())


def test_the_band_marks_the_design_the_worst_case_drops():
    """The point of the band: a design the deterministic front keeps and the worst-case front does not."""
    Y = np.array([[1.0, 4.0], [2.0, 2.5], [2.2, 2.4], [3.0, 1.0]])
    S = np.array([[0.05, 0.05], [0.05, 0.05], [0.4, 0.4], [0.05, 0.05]])
    p = pareto_with_uncertainty(Y, S, k=1.0, n_draws=20000, seed=0)
    assert p.nominal[2] and not p.robust[2]
    assert p.p_on_front[2] == pytest.approx(0.708, abs=0.04)
    assert p.to_dict()["nominal_front"] == [0, 1, 2, 3] and p.to_dict()["robust_front"] == [0, 1, 3]


def test_a_design_off_the_front_that_joins_it_more_than_half_the_time():
    """Dominated on the means by one tenth, with a sigma that covers the gap: it is on the front 60 % of the time."""
    Y = np.array([[1.0, 4.0], [2.0, 1.95], [2.1, 2.0], [3.0, 1.0]])
    S = np.array([[0.02, 0.02], [0.02, 0.02], [0.25, 0.25], [0.02, 0.02]])
    p = pareto_with_uncertainty(Y, S, k=1.0, n_draws=20000, seed=0)
    assert not p.nominal[2]
    assert 0.55 < p.p_on_front[2] < 0.75
    assert p.could_join[2] and not p.could_leave.any()
    assert p.to_dict()["could_join"] == [2]


def test_the_probability_of_keeping_a_place_matches_the_integral():
    """One objective, three designs: P(design i is the best) has a closed form, computed here independently."""
    mu = np.array([[0.0], [0.4], [1.0]])
    sd = np.array([[1.0], [1.0], [1.5]])
    p = pareto_with_uncertainty(mu, sd, k=1.0, n_draws=50000, seed=0)

    def p_best(i):
        others = [j for j in range(3) if j != i]

        def integrand(tv):
            return norm.pdf(tv, mu[i, 0], sd[i, 0]) * np.prod(
                [1.0 - norm.cdf(tv, mu[j, 0], sd[j, 0]) for j in others])

        return quad(integrand, -12, 12, limit=400)[0]

    exact = np.array([p_best(i) for i in range(3)])
    assert exact[0] > 0.45 and exact[2] < 0.25                      # the case has a clear shape
    np.testing.assert_allclose(p.p_on_front, exact, atol=0.015)
    assert p.p_on_front[0] > p.p_on_front[1] > p.p_on_front[2]


def test_the_front_rejects_what_it_cannot_use():
    Y = np.array([[1.0, 4.0], [2.0, 2.5], [3.0, 1.0], [2.6, 1.4]])
    S = np.array([[0.05, 0.05], [0.05, 0.05], [0.05, 0.05], [0.4, 0.1]])
    with pytest.raises(ValueError, match="same shape"):
        pareto_with_uncertainty(Y, S[:, :1])
    with pytest.raises(ValueError, match=">= 0"):
        pareto_with_uncertainty(Y, -S)
    with pytest.raises(ValueError, match="k must be"):
        pareto_with_uncertainty(Y, S, k=-1.0)
