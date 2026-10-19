"""Black-box minimisers on the unit box for the red team: differential evolution (SciPy), CMA-ES and a
finite-difference L-BFGS-B (a gradient method for objectives that are smooth in the fault parameters).

CMA-ES is written from N. Hansen, "The CMA Evolution Strategy: A Tutorial" (arXiv:1604.00772, 2016): weighted
recombination, cumulative step-size adaptation, rank-one and rank-mu covariance updates with the default constants
of the tutorial's Table 1. Box constraints are handled by evaluating the projection of each sample onto the box.

Every minimiser takes an evaluation budget, a seed, and returns the best point and value and the number of
evaluations used; all are deterministic for a given seed.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = ["SearchResult", "minimize", "cmaes", "differential_evolution", "lbfgs", "METHODS"]


@dataclass
class SearchResult:
    x: np.ndarray
    fun: float
    evaluations: int
    method: str


class _Counted:
    def __init__(self, f: Callable[[np.ndarray], float], budget: int):
        self.f, self.budget, self.n = f, int(budget), 0
        self.best_x: np.ndarray | None = None
        self.best = math.inf

    def __call__(self, x: np.ndarray) -> float:
        x = np.clip(np.asarray(x, float), 0.0, 1.0)
        if self.n >= self.budget:
            return self.best + 1.0                      # out of budget: never better than the best
        self.n += 1
        v = float(self.f(x))
        if not math.isfinite(v):
            v = 1e12
        if v < self.best:
            self.best, self.best_x = v, x.copy()
        return v


def cmaes(f: Callable[[np.ndarray], float], dim: int, budget: int, seed: int = 0, sigma0: float = 0.3,
          x0: np.ndarray | None = None) -> SearchResult:
    """(mu/mu_w, lambda)-CMA-ES on [0, 1]^dim (Hansen 2016)."""
    F = _Counted(f, budget)
    rng = np.random.default_rng(seed)
    n = dim
    lam = 4 + int(3 * math.log(n))
    mu = lam // 2
    w = math.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / float((w ** 2).sum())
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chin = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    m = np.full(n, 0.5) if x0 is None else np.clip(np.asarray(x0, float), 0, 1)
    sigma = float(sigma0)
    pc, ps = np.zeros(n), np.zeros(n)
    B, D, C = np.eye(n), np.ones(n), np.eye(n)
    g = 0
    while F.n + lam <= F.budget:
        z = rng.standard_normal((lam, n))
        y = z @ (B * D).T
        x = m + sigma * y
        vals = np.array([F(xi) for xi in x])
        order = np.argsort(vals)
        y_sel = y[order[:mu]]
        y_w = w @ y_sel
        m = np.clip(m + sigma * y_w, 0.0, 1.0)
        # step size: conjugate evolution path, C^(-1/2) = B D^-1 B^T
        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (B @ ((B.T @ y_w) / D))
        g += 1
        hsig = np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * g)) / chin < 1.4 + 2 / (n + 1)
        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * y_w
        C = ((1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
             + cmu * (y_sel.T * w) @ y_sel)
        sigma *= math.exp((cs / damps) * (np.linalg.norm(ps) / chin - 1))
        sigma = min(sigma, 1.0)
        C = np.triu(C) + np.triu(C, 1).T
        ev, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(ev, 1e-20))
        if sigma * D.max() < 1e-8:
            break
    if F.best_x is None:
        F(m)
    return SearchResult(F.best_x, F.best, F.n, "cmaes")


def differential_evolution(f: Callable[[np.ndarray], float], dim: int, budget: int, seed: int = 0) -> SearchResult:
    """SciPy's differential evolution (best1bin, Latin hypercube start) within ``budget`` evaluations."""
    from scipy.optimize import differential_evolution as de
    F = _Counted(f, budget)
    popsize = max(5, min(15, budget // (dim * 4) or 5))
    maxiter = max(1, budget // (popsize * dim) - 1)
    de(F, [(0.0, 1.0)] * dim, popsize=popsize, maxiter=maxiter, seed=seed, polish=False, tol=0.0,
       init="latinhypercube", updating="immediate")
    return SearchResult(F.best_x, F.best, F.n, "de")


def lbfgs(f: Callable[[np.ndarray], float], dim: int, budget: int, seed: int = 0) -> SearchResult:
    """L-BFGS-B with finite-difference gradients from a few seeded starts (local, gradient-based)."""
    from scipy.optimize import minimize as sp_min
    F = _Counted(f, budget)
    rng = np.random.default_rng(seed)
    while F.n + 2 * (dim + 1) < F.budget:
        sp_min(F, rng.uniform(0, 1, dim), method="L-BFGS-B", bounds=[(0.0, 1.0)] * dim,
               options={"maxfun": max(dim + 2, (F.budget - F.n)), "eps": 1e-3})
    return SearchResult(F.best_x if F.best_x is not None else np.full(dim, 0.5), F.best, F.n, "lbfgs")


METHODS = {"de": differential_evolution, "cmaes": cmaes, "lbfgs": lbfgs}


def minimize(f: Callable[[np.ndarray], float], dim: int, budget: int, method: str = "de",
             seed: int = 0) -> SearchResult:
    """Minimise ``f`` on [0, 1]^dim with ``method`` (``"de"``, ``"cmaes"`` or ``"lbfgs"``)."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; use one of {sorted(METHODS)}")
    return METHODS[method](f, dim, budget, seed=seed)
