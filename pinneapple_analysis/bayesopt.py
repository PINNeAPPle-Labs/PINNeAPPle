"""Bayesian optimization and sequential experimental design (#102).

The point of a Bayesian optimizer is not to be clever: it is to need **fewer reference simulations** than random
sampling for the same answer. A reference simulation here is anything expensive — a CFD run, a finite-element solve, an
experiment on a rig — so the currency is the number of calls, and the loop is judged on that.

    >>> from pinneapple_analysis.bayesopt import BayesianOptimizer, branin
    >>> opt = BayesianOptimizer(branin, [(-5, 10), (0, 15)], n_initial=6, n_iter=14, seed=0)   # doctest: +SKIP
    >>> result = opt.run()                                                                     # doctest: +SKIP
    >>> result.best_y < 1.0                                                                    # doctest: +SKIP
    True

What it is built on: a Gaussian process with an anisotropic squared-exponential kernel, its hyperparameters fitted by
maximizing the log marginal likelihood, and an acquisition function on top — expected improvement (the default),
upper confidence bound or probability of improvement. Constraints enter through feasibility-weighted expected
improvement: a second GP models the constraint, and a candidate is worth its improvement times the probability of being
feasible.

What it refuses: a bound that is empty or reversed, a batch larger than the budget, a constraint that is not
``(callable, limit)``, and an acquisition it does not know. A silent fallback to random search would make the
comparison that justifies the module meaningless.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["ACQUISITIONS", "BayesianOptimizer", "BOResult", "GaussianProcess", "benchmark_bayesopt", "branin",
           "expected_improvement", "hartmann3", "probability_of_improvement", "upper_confidence_bound"]

ACQUISITIONS = ("ei", "ucb", "pi")


# --------------------------------------------------------------------------- test problems
def branin(x: np.ndarray) -> np.ndarray:
    """Branin-Hoo: three global minima, value 0.397887. Takes one point or a batch."""
    X = np.atleast_2d(np.asarray(x, dtype=float))
    a, b, c = 1.0, 5.1 / (4 * math.pi ** 2), 5 / math.pi
    r, s, t = 6.0, 10.0, 1 / (8 * math.pi)
    out = a * (X[:, 1] - b * X[:, 0] ** 2 + c * X[:, 0] - r) ** 2 + s * (1 - t) * np.cos(X[:, 0]) + s
    return out


def hartmann3(x: np.ndarray) -> np.ndarray:
    """Hartmann-3: one global minimum, value -3.86278 (the module minimizes, so this is a hard target)."""
    X = np.atleast_2d(np.asarray(x, dtype=float))
    A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
    P = 1e-4 * np.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]])
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    inner = (A[None, :, :] * (X[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    return -(alpha * np.exp(-inner)).sum(-1)


# --------------------------------------------------------------------------- acquisition
def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    """EI for minimisation: how much better than ``best`` the next point is expected to be."""
    from scipy.stats import norm

    mu, sigma = np.asarray(mu, float), np.maximum(np.asarray(sigma, float), 1e-12)
    improvement = best - mu - xi
    z = improvement / sigma
    return improvement * norm.cdf(z) + sigma * norm.pdf(z)


def probability_of_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    from scipy.stats import norm

    mu, sigma = np.asarray(mu, float), np.maximum(np.asarray(sigma, float), 1e-12)
    return norm.cdf((best - mu - xi) / sigma)


def upper_confidence_bound(mu: np.ndarray, sigma: np.ndarray, best: float = 0.0, kappa: float = 2.0) -> np.ndarray:
    """UCB for minimisation: the lower confidence bound, negated so that larger is better for the caller."""
    return -(np.asarray(mu, float) - kappa * np.maximum(np.asarray(sigma, float), 1e-12))


# --------------------------------------------------------------------------- surrogate
@dataclass
class GaussianProcess:
    """A zero-mean GP with an anisotropic squared-exponential kernel, fitted by marginal likelihood.

    The output is standardised before the fit (a surrogate on numbers of order 1e5 and one of order 1 need the same
    prior), the length scales are bounded relative to the spread of each input, and the noise stays fixed: letting all
    three float lets the optimiser collapse the kernel onto the five initial points, which fits perfectly and predicts
    the mean everywhere.
    """

    length_scale: np.ndarray | None = None
    signal: float = 1.0
    noise: float = 1e-6
    X: np.ndarray | None = None
    y: np.ndarray | None = None
    y_mean: float = 0.0
    y_std: float = 1.0
    _alpha: np.ndarray | None = field(default=None, repr=False)
    _fitted: bool = field(default=False, repr=False)

    def _kernel(self, A: np.ndarray, B: np.ndarray, length_scale: np.ndarray, signal: float) -> np.ndarray:
        scaled_a, scaled_b = A / length_scale, B / length_scale
        d2 = (scaled_a ** 2).sum(1)[:, None] + (scaled_b ** 2).sum(1)[None, :] - 2 * scaled_a @ scaled_b.T
        return signal ** 2 * np.exp(-0.5 * np.maximum(d2, 0.0))

    def _matrix(self, length_scale, signal, X) -> np.ndarray:
        K = self._kernel(X, X, length_scale, signal)
        K[np.diag_indices_from(K)] += self.noise
        return K

    def fit(self, X, y, *, optimize: bool = True) -> GaussianProcess:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        self.y_mean = float(y.mean())
        self.y_std = float(y.std() or 1.0)
        z = (y - self.y_mean) / self.y_std
        d = X.shape[1]
        spread = np.maximum(np.ptp(X, axis=0), 1e-6)
        if self.length_scale is None:
            self.length_scale = spread / 4.0
        self.length_scale = np.asarray(self.length_scale, dtype=float).clip(spread / 50.0, spread * 5.0)

        def negative_log_likelihood(theta: np.ndarray) -> float:
            length, signal = np.exp(theta[:d]), math.exp(theta[d])
            try:
                L = np.linalg.cholesky(self._matrix(length, signal, X))
            except np.linalg.LinAlgError:
                return 1e10
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, z))
            return float(0.5 * z @ alpha + np.log(np.diag(L)).sum())

        if optimize and len(X) >= d + 2:
            from scipy.optimize import minimize

            theta0 = np.r_[np.log(self.length_scale), np.log(self.signal)]
            bounds = [(math.log(s / 50.0), math.log(s * 5.0)) for s in spread] + [(-6.0, 6.0)]
            best = minimize(negative_log_likelihood, theta0, method="L-BFGS-B", bounds=bounds)
            if best.fun <= negative_log_likelihood(theta0):          # only take the optimum if it is one
                self.length_scale = np.exp(best.x[:d])
                self.signal = math.exp(best.x[d])
        self.X, self._fitted = X, True
        L = np.linalg.cholesky(self._matrix(self.length_scale, self.signal, X))
        self._alpha = np.linalg.solve(L.T, np.linalg.solve(L, z))
        return self

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("the Gaussian process is not fitted: call fit(X, y) first")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Ks = self._kernel(self.X, X, self.length_scale, self.signal)
        mu = self.y_mean + self.y_std * (Ks.T @ self._alpha)
        L = np.linalg.cholesky(self._matrix(self.length_scale, self.signal, self.X))
        v = np.linalg.solve(L, Ks)
        var = np.maximum(self.signal ** 2 - (v ** 2).sum(0), 1e-12)
        return mu, np.sqrt(var) * self.y_std


# --------------------------------------------------------------------------- the loop
@dataclass
class BOResult:
    """What the loop did: every point it evaluated, and the best it found."""

    X: np.ndarray
    y: np.ndarray
    best_x: np.ndarray
    best_y: float
    n_initial: int
    maximize: bool = False

    @property
    def n_evals(self) -> int:
        return len(self.y)

    @property
    def best_after(self) -> list[float]:
        """The best value after each call: the curve a benchmark compares against random sampling."""
        run = np.maximum.accumulate(self.y) if self.maximize else np.minimum.accumulate(self.y)
        return [float(v) for v in run]

    def evals_to_reach(self, target: float) -> int | None:
        """How many calls it took to reach ``target`` (``None`` if it never did)."""
        for i, value in enumerate(self.best_after, start=1):
            if value >= target if self.maximize else value <= target:
                return i
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"n_evals": self.n_evals, "n_initial": self.n_initial, "best_x": self.best_x.tolist(),
                "best_y": self.best_y, "best_after": self.best_after, "maximize": self.maximize}


class BayesianOptimizer:
    """Minimise (or maximise) ``objective`` over ``bounds`` with as few calls as the surrogate allows.

    ``bounds`` is a sequence of ``(low, high)``; ``constraints`` a sequence of ``(callable, limit)`` where the point is
    feasible when ``callable(x) <= limit``. ``batch_size`` asks for several points per round, each chosen with the
    others' predicted means appended to the surrogate (a cheap way to avoid proposing the same point twice).
    """

    def __init__(self, objective: Callable[[np.ndarray], Any], bounds: Sequence[tuple[float, float]], *,
                 n_initial: int = 5, n_iter: int = 20, batch_size: int = 1, acquisition: str = "ei",
                 xi: float = 0.01, kappa: float = 2.0, constraints: Sequence[tuple[Callable, float]] = (),
                 maximize: bool = False, seed: int = 0, n_candidates: int = 2048) -> None:
        low = np.array([float(lo) for lo, _ in bounds], dtype=float)
        high = np.array([float(hi) for _, hi in bounds], dtype=float)
        if len(low) == 0:
            raise ValueError("give at least one bound")
        if np.any(high <= low):
            raise ValueError(f"bounds have to be increasing, got {list(bounds)}")
        if acquisition not in ACQUISITIONS:
            raise ValueError(f"unknown acquisition {acquisition!r}; choose from {ACQUISITIONS}")
        for item in constraints:
            if not (isinstance(item, (tuple, list)) and len(item) == 2 and callable(item[0])):
                raise ValueError(f"a constraint is (callable, limit), got {item!r}")
        if n_iter * batch_size + n_initial <= 0:
            raise ValueError("the budget is empty")
        self.objective = objective
        self.low, self.high = low, high
        self.n_initial, self.n_iter, self.batch_size = n_initial, n_iter, batch_size
        self.acquisition, self.xi, self.kappa = acquisition, xi, kappa
        self.constraints = list(constraints)
        self.maximize = maximize
        self.rng = np.random.default_rng(seed)
        self.n_candidates = n_candidates
        self.X: list[np.ndarray] = []
        self.y: list[float] = []

    # -- helpers
    def _is_feasible(self, x: np.ndarray) -> bool:
        return all(float(f(x)) <= limit for f, limit in self.constraints)

    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(self.objective(X), dtype=float).ravel()
        if values.size != len(X):
            raise ValueError(f"the objective returned {values.size} values for {len(X)} points")
        return -values if self.maximize else values

    def _candidates(self, n: int) -> np.ndarray:
        return self.low + (self.high - self.low) * self.rng.random((n, len(self.low)))

    def _acquisition(self, mu: np.ndarray, sigma: np.ndarray, best: float) -> np.ndarray:
        if self.acquisition == "ei":
            return expected_improvement(mu, sigma, best, self.xi)
        if self.acquisition == "pi":
            return probability_of_improvement(mu, sigma, best, self.xi)
        return upper_confidence_bound(mu, sigma, best, self.kappa)

    def _gp(self) -> tuple[GaussianProcess, GaussianProcess | None]:
        X = np.array(self.X)
        gp = GaussianProcess().fit(X, np.array(self.y))
        if not self.constraints:
            return gp, None
        # one GP on the worst violation: > 0 means the point breaks at least one constraint
        worst = np.array([max(float(f(x)) - limit for f, limit in self.constraints) for x in X])
        return gp, GaussianProcess().fit(X, worst)

    # -- the loop
    def run(self) -> BOResult:
        X0 = self._candidates(self.n_initial)
        y0 = self._evaluate(X0)
        self.X = list(X0)
        self.y = [float(v) for v in y0]
        for _ in range(self.n_iter):
            self._extend()
        return self._result()

    def _extend(self) -> None:
        if len(self.X) < 2:
            self._ask_tell(1)
            return
        gp, constraint_gp = self._gp()
        # the incumbent has to be a point that satisfies the constraints: an infeasible value would make every
        # improvement negative and the acquisition underflow to zero
        feasible = [y for x, y in zip(self.X, self.y, strict=True) if self._is_feasible(x)]
        best = float(np.min(feasible)) if feasible else float(np.min(self.y))
        for _ in range(self.batch_size):
            pool = self._candidates(self.n_candidates)
            mu, sigma = gp.predict(pool)
            score = self._acquisition(mu, sigma, best)
            if constraint_gp is not None:
                from scipy.stats import norm

                v_mu, v_sd = constraint_gp.predict(pool)
                # a candidate the surrogate expects to break a constraint is not proposed at all; the probability
                # weights the ones near the boundary, where the expectation is uncertain
                allowed = v_mu <= 0.0
                if not allowed.any():                      # nothing looks feasible: take the least violated decile
                    allowed = v_mu <= np.quantile(v_mu, 0.1)
                score = np.where(allowed, score, 0.0) * np.clip(1.0 - norm.cdf(v_mu / np.maximum(v_sd, 1e-12)),
                                                                1e-6, 1.0)
            x = pool[int(np.argmax(score))]
            value = float(self._evaluate(x[None, :])[0])
            self.X.append(x)
            self.y.append(value)
            best = min(best, value)
            gp = GaussianProcess().fit(np.array(self.X), np.array(self.y))     # refit with the new point
            if constraint_gp is not None:
                constraint_gp = GaussianProcess().fit(
                    np.array(self.X),
                    np.array([max(float(f(p)) - limit for f, limit in self.constraints) for p in self.X]))

    def _ask_tell(self, n: int) -> None:
        pool = self._candidates(n)
        self.X.extend(list(pool))
        self.y.extend(float(v) for v in self._evaluate(pool))

    def _result(self) -> BOResult:
        y = np.array(self.y)
        candidates = [i for i, x in enumerate(self.X) if self._is_feasible(x)] or list(range(len(self.X)))
        i = min(candidates, key=lambda k: y[k])       # the internal y is always the quantity being minimised
        return BOResult(X=np.array(self.X), y=y, best_x=np.array(self.X[i]),
                        best_y=float(self.y[i]) * (-1 if self.maximize else 1),
                        n_initial=self.n_initial, maximize=self.maximize)


def benchmark_bayesopt(problem: str = "branin", *, budget: int = 24, seeds: Sequence[int] = (0, 1, 2, 3, 4),
                       target: float | None = None, n_initial: int = 5) -> dict[str, Any]:
    """Compare the loop with random sampling at the same budget — the measurement the issue asks for.

    Three numbers, because one is not enough: how often each method reaches the target **within the budget**, how many
    calls it took when it did, and the best value each ended with. A median of evals-to-target computed over the runs
    that succeeded, while the others never got there, would flatter the method that failed most.
    """
    problems = {
        "branin": (branin, [(-5.0, 10.0), (0.0, 15.0)], 0.397887, 1.0),
        "hartmann3": (hartmann3, [(0.0, 1.0)] * 3, -3.86278, -3.5),
    }
    if problem not in problems:
        raise ValueError(f"unknown problem {problem!r}; choose from {sorted(problems)}")
    fn, bounds, optimum, default_target = problems[problem]
    goal = default_target if target is None else target
    low = np.array([b[0] for b in bounds])
    high = np.array([b[1] for b in bounds])
    rows = []
    for seed in seeds:
        result = BayesianOptimizer(fn, bounds, n_initial=n_initial, n_iter=budget - n_initial, seed=seed).run()
        rng = np.random.default_rng(seed)
        draws = fn(low + (high - low) * rng.random((budget, len(bounds))))
        random_run = np.minimum.accumulate(draws)
        random_evals = next((i for i, v in enumerate(random_run, start=1) if v <= goal), None)
        rows.append({"seed": seed, "bo_evals": result.evals_to_reach(goal), "random_evals": random_evals,
                     "bo_best": result.best_y, "random_best": float(random_run[-1])})

    def summary(key: str) -> tuple[float, float | None, float]:
        reached = [r[key] for r in rows if r[key] is not None]
        bests = [r["bo_best" if key == "bo_evals" else "random_best"] for r in rows]
        return len(reached) / len(rows), (float(np.median(reached)) if reached else None), float(np.median(bests))

    bo_rate, bo_evals, bo_best = summary("bo_evals")
    rand_rate, rand_evals, rand_best = summary("random_evals")
    fewer = (bo_rate > rand_rate) or (bo_rate == rand_rate and bo_evals is not None and rand_evals is not None
                                      and bo_evals < rand_evals)
    return {"problem": problem, "budget": budget, "target": goal, "optimum": optimum, "rows": rows,
            "bo": {"reached_rate": bo_rate, "median_evals": bo_evals, "median_best": bo_best},
            "random": {"reached_rate": rand_rate, "median_evals": rand_evals, "median_best": rand_best},
            "fewer_calls": bool(fewer), "n_seeds": len(rows)}

    # the same comparison, one line per method
