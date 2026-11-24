"""Optimization that carries the surrogate's uncertainty: bands on the Pareto front and the three robust
formulations (#80).

Two separate things, both about the same question — *what does the uncertainty do to the design I would pick?*

**The front, with bands.** A Pareto front built from the surrogate's means is a front for a model, not for the
physics. :func:`pareto_with_uncertainty` takes the mean and the standard deviation of every objective for every
design and reports, next to the nominal front:

- the **band** of each design (``mean ± k sigma`` per objective) and which designs are still nondominated under the
  *worst* case (``mean + k sigma`` for objectives that are minimized) — a front you can defend;
- ``p_on_front``: the probability that each design stays on the front when the objectives are drawn from what the
  surrogate believes about them (independent Gaussians, ``n_draws`` draws). A design with ``p_on_front = 0.5`` is
  tied with its neighbour in the ranking, and the table says so;
- ``could_leave`` / ``could_join``: the designs whose place on the front is a coin toss, which is the list an engineer
  has to resolve with another run.

**The design, with uncertainty.** The three formulations the market audit added (2026-11-03, #1263), each with an
optimum that can be written down, which is what the tests check:

- :func:`mean_plus_k_sigma` — minimize ``mu(x) + k * sigma(x)``. On the trade-off that makes the formulation worth
  having (a larger design is cheaper on the mean and more uncertain) — ``mu = 1/x``, ``sigma = sqrt(x)`` — the
  optimum is ``x* = (2 / k)^(2/3)`` exactly;
- :func:`chance_constrained` — minimize a cost subject to ``P(g(x, xi) <= 0) >= 1 - alpha``. With ``g = limit - y``
  and ``y ~ N(mu(x), sigma(x)^2)`` the constraint is ``mu(x) - z_{1-alpha} sigma(x) >= limit``, a deterministic one;
- :func:`reliability_based` — the same design written as a target probability of failure, ``P(failure) <= p_f`` with
  ``p_f = alpha``. The two must return the **same design**, and that identity is a test.

Every formulation is checked twice: against the closed-form optimum, and by Monte Carlo at that optimum — the design
the formula gives must show the probability of violation it promised, not just a value.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from pinneapple_design.design_optimizer.pareto import compute_pareto_front


# ---------------------------------------------------------------------------------------------- the three formulations
@dataclass
class RobustSolution:
    """A design chosen under uncertainty, the two ways of stating the requirement, and the check of both."""
    formulation: str
    x: np.ndarray
    objective: float
    mean: float
    sigma: float
    requirement: dict
    violation: dict = field(default_factory=dict)
    note: str = ""
    n_evals: int = 0

    @property
    def margin(self) -> float:
        """How many standard deviations the mean sits above the limit (the reliability index of this design)."""
        return self.requirement.get("reliability_index", float("nan"))

    def to_dict(self) -> dict:
        return {"formulation": self.formulation, "x": np.asarray(self.x).tolist(), "objective": self.objective,
                "mean": self.mean, "sigma": self.sigma, "requirement": self.requirement,
                "violation": self.violation, "note": self.note, "n_evals": self.n_evals}


def _minimize(fn, x0, bounds, n_restarts: int = 3) -> tuple[np.ndarray, float, int]:
    """Nelder-Mead from a few starts inside the box; a 1-D design has one minimum and this finds it."""
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    starts = [np.clip(np.asarray(x0, float), lo, hi)]
    rng = np.random.default_rng(0)
    starts += [lo + rng.random(lo.size) * (hi - lo) for _ in range(max(0, n_restarts - 1))]
    best, best_f, n = None, np.inf, 0
    for s in starts:
        r = minimize(fn, np.clip(s, lo, hi), method="Nelder-Mead", bounds=list(zip(lo, hi, strict=True)),
                     options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 4000})
        n += int(r.nfev)
        if r.fun < best_f:
            best, best_f = np.clip(r.x, lo, hi), float(r.fun)
    return best, best_f, n


def mean_plus_k_sigma(mean, sigma, bounds, k: float = 1.0, x0=None) -> RobustSolution:
    """``min mu(x) + k sigma(x)`` — the mean with a penalty of ``k`` standard deviations.

    ``mean`` and ``sigma`` are callables of a design vector (the surrogate's prediction and its standard deviation).
    The closed-form check: with ``mu = 1/x`` and ``sigma = sqrt(x)`` on ``x > 0`` the derivative
    ``-1/x^2 + k / (2 sqrt(x))`` is zero at ``x* = (2/k)^(2/3)`` — the point where another unit of design buys less
    mean than the uncertainty it adds. Without a trade-off (``mu`` increasing) the optimum is simply the best mean, and
    the penalty changes nothing, which is the honest behaviour of the formulation.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    bounds = list(bounds)
    x0 = np.array([0.5 * (b[0] + b[1]) for b in bounds]) if x0 is None else np.asarray(x0, float)

    def f(x):
        return float(mean(x) + k * sigma(x))

    best, best_f, n = _minimize(f, x0, bounds)
    return RobustSolution(formulation="mean_plus_k_sigma", x=best, objective=best_f,
                          mean=float(mean(best)), sigma=float(sigma(best)),
                          requirement={"k": float(k), "rule": "minimize mean + k sigma"}, n_evals=n)


def chance_constrained(mean, sigma, cost, alpha: float = 0.05, bounds=None, x0=None, limit=None,
                       sense: str = "above") -> RobustSolution:
    """Minimize ``cost(x)`` subject to ``P(y(x) >= limit) >= 1 - alpha`` (``sense="above"``) or
    ``P(y(x) <= limit) <= alpha`` for a failure written the other way (``sense="below"``).

    ``y ~ N(mean(x), sigma(x)^2)``, so the constraint is ``mean(x) - z sigma(x) >= limit`` with ``z = Phi^-1(1-alpha)``
    — the standard form, written out so the reliability index of the answer is visible. With ``cost = x``,
    ``mean = a x`` and ``sigma`` constant the optimum is ``x* = (limit + z sigma) / a`` exactly.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if limit is None:
        raise ValueError("give the limit the design has to respect")
    bounds = list(bounds) if bounds is not None else [(0.0, 1.0)]
    x0 = np.array([0.5 * (b[0] + b[1]) for b in bounds]) if x0 is None else np.asarray(x0, float)
    z = float(norm.ppf(1.0 - alpha))
    sign = 1.0 if sense == "above" else -1.0

    def violation(x):
        d = sign * (mean(x) - limit)                         # the slack of the constraint, in sigma
        return d, sigma(x)

    def f(x):
        d, s = violation(x)
        slack = d - z * s
        return float(cost(x) + 1e6 * max(0.0, -slack) ** 2 + (0.0 if slack >= 0 else -1e3 * slack))

    best, _, n = _minimize(f, x0, bounds)
    d, s = violation(best)
    beta = float(d / s)
    return RobustSolution(formulation="chance_constraint", x=best, objective=float(cost(best)),
                          mean=float(mean(best)), sigma=float(s),
                          requirement={"alpha": float(alpha), "limit": float(limit), "sense": sense,
                                       "target_probability": float(1.0 - alpha), "reliability_index": beta,
                                       "rule": f"P({sense} the limit) >= {1.0 - alpha:.4f}"},
                          note="the constraint is met with the reliability index reported" if d >= z * s
                          else "the constraint is not met: the cost cannot buy the margin inside the box",
                          n_evals=n)


def reliability_based(mean, sigma, target_pf: float = 0.05, **kw) -> RobustSolution:
    """The chance constraint said as a probability of failure: ``P(failure) <= target_pf``.

    Same computation as :func:`chance_constrained` with ``alpha = target_pf``; it exists because the two are how the
    requirement is written in different teams' documents, and a design must not depend on which one was used. The
    result carries both statements (``alpha`` and ``target_pf``), so the report can show the pair.
    """
    sol = chance_constrained(mean=mean, sigma=sigma, alpha=target_pf, **kw)
    sol.formulation = "reliability_based_design"
    sol.requirement["target_pf"] = float(target_pf)
    sol.requirement["beta_target"] = float(-norm.ppf(target_pf))
    return sol


def verify_robust(solution: RobustSolution, mean, sigma, n: int = 200_000, seed: int = 0,
                  limit=None, sense: str = "above") -> dict:
    """Monte Carlo at the design the formulation returned: does it show the promised probability?

    Draws ``y ~ N(mean(x*), sigma(x*)^2)`` ``n`` times and counts the violations. With independent draws the count is
    Binomial, so the 95 % band of the empirical rate is ``p +- 1.96 sqrt(p (1 - p) / n)`` — reported and used by the
    tests to decide whether the promise was kept.
    """
    lim = float(solution.requirement["limit"] if limit is None else limit)
    sense = solution.requirement.get("sense", sense)
    mu, s = float(mean(solution.x)), float(sigma(solution.x))
    y = np.random.default_rng(seed).normal(mu, max(s, 1e-300), int(n))
    bad = y < lim if sense == "above" else y > lim
    p = float(bad.mean())
    se = float(np.sqrt(max(p * (1 - p), 1e-300) / n))
    target = float(solution.requirement.get("alpha", 0.05))
    out = {"n": int(n), "probability_of_violation": p, "standard_error": se,
           "target": target, "band": [p - 1.96 * se, p + 1.96 * se],
           "within_target": bool(p <= target + 1.96 * se or (sense == "above" and p == 0.0)),
           "reliability_index_empirical": float(-norm.ppf(max(p, 1e-12))) if p > 0 else float("inf"),
           "mean": mu, "sigma": s, "limit": lim}
    solution.violation = out
    return out


# ---------------------------------------------------------------------------------------------- the front, with bands
@dataclass
class RobustPareto:
    """A Pareto front that carries the surrogate's uncertainty."""
    objectives: np.ndarray
    sigma: np.ndarray
    k: float
    nominal: np.ndarray                 # mask: nondominated on the means
    robust: np.ndarray                  # mask: nondominated on mean + k sigma
    p_on_front: np.ndarray              # per design: P(design stays nondominated)
    band: tuple[np.ndarray, np.ndarray]  # (lower, upper) per objective, per design
    n_draws: int
    params: np.ndarray | None = None

    @property
    def could_leave(self) -> np.ndarray:
        """On the nominal front, but not reliably (``p_on_front < 0.5``)."""
        return self.nominal & (self.p_on_front < 0.5)

    @property
    def could_join(self) -> np.ndarray:
        """Off the nominal front, but it would join it half the time or more."""
        return (~self.nominal) & (self.p_on_front >= 0.5)

    def table(self) -> list[dict]:
        rows = []
        for i in range(self.objectives.shape[0]):
            rows.append({"design": i,
                         "objectives": self.objectives[i].tolist(),
                         "sigma": self.sigma[i].tolist(),
                         "band": [self.band[0][i].tolist(), self.band[1][i].tolist()],
                         "nominal_front": bool(self.nominal[i]), "robust_front": bool(self.robust[i]),
                         "p_on_front": float(self.p_on_front[i])})
        return rows

    def to_dict(self) -> dict:
        return {"method": "Pareto front with surrogate bands", "k": self.k, "n_draws": self.n_draws,
                "n_designs": int(self.objectives.shape[0]),
                "nominal_front": [int(i) for i in np.nonzero(self.nominal)[0]],
                "robust_front": [int(i) for i in np.nonzero(self.robust)[0]],
                "could_leave": [int(i) for i in np.nonzero(self.could_leave)[0]],
                "could_join": [int(i) for i in np.nonzero(self.could_join)[0]],
                "rows": self.table()}


def pareto_with_uncertainty(objectives, sigma, k: float = 2.0, n_draws: int = 2000, seed: int = 0,
                            params=None) -> RobustPareto:
    """The nominal front, the frontier under the worst case at ``k`` sigma, and the probability of each design's place.

    ``objectives`` and ``sigma`` are ``(n_designs, n_objectives)``, all objectives minimized. Two fronts:

    - ``nominal``: nondominated on the means, which is what a deterministic optimization reports;
    - ``robust``: nondominated on ``mean + k sigma``, which is the front that survives the model being at the bad end
      of its own interval.

    ``p_on_front`` comes from ``n_draws`` draws of every design's objectives from ``N(mean, sigma^2)`` (independent
    across objectives and designs): the fraction of draws in which the design is nondominated. It is the number that
    turns an ordering into a statement: 1.0 is a place on the front, 0.5 is a tie with the neighbour.
    """
    Y = np.atleast_2d(np.asarray(objectives, dtype=float))
    S = np.atleast_2d(np.asarray(sigma, dtype=float))
    if Y.shape != S.shape:
        raise ValueError(f"objectives {Y.shape} and sigma {S.shape} must have the same shape")
    if np.any(S < 0):
        raise ValueError("the standard deviations must be >= 0")
    if k < 0:
        raise ValueError("k must be >= 0")
    n, m = Y.shape
    nominal = compute_pareto_front(Y)
    robust = compute_pareto_front(Y + k * S)
    rng = np.random.default_rng(seed)
    hits = np.zeros(n)
    for _ in range(int(n_draws)):
        draw = Y + S * rng.standard_normal((n, m))
        hits += compute_pareto_front(draw)
    return RobustPareto(objectives=Y, sigma=S, k=float(k), nominal=nominal, robust=robust,
                        p_on_front=hits / float(n_draws), band=(Y - k * S, Y + k * S), n_draws=int(n_draws),
                        params=None if params is None else np.asarray(params, float))


__all__ = ["RobustSolution", "RobustPareto", "mean_plus_k_sigma", "chance_constrained", "reliability_based",
           "verify_robust", "pareto_with_uncertainty"]
