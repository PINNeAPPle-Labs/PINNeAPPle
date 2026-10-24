"""Small learning loops that train on a CPU against :class:`~pinneapple_simulation.robot_sim.sim.RobotSim`
rollouts (in the spirit of Isaac Lab, at laptop scale).

``cem(objective, mean, std, ...)``
    cross-entropy method: sample parameters from a Gaussian, keep the elite fraction, refit, repeat. Good for a
    handful of parameters (controller gains, a trajectory's timing). Every evaluation is recorded, so the learning
    curve and the best parameters are reproducible from the seed.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

__all__ = ["cem", "CEMResult"]


@dataclass
class CEMResult:
    best: np.ndarray
    best_cost: float
    history: list[dict] = field(default_factory=list)       # per iteration: mean, std, best and mean elite cost
    evaluations: int = 0


def cem(objective: Callable[[np.ndarray], float], mean, std, *, iterations: int = 6, population: int = 8,
        elite: float = 0.25, lower=None, upper=None, seed: int = 0, min_std: float = 1e-6) -> CEMResult:
    """Minimise ``objective(params)``; ``lower`` / ``upper`` clip the samples."""
    rng = np.random.default_rng(seed)
    mu, sd = np.asarray(mean, float).copy(), np.asarray(std, float).copy()
    ne = max(1, int(round(elite * population)))
    best, best_c, hist, n = mu.copy(), float(objective(mu)), [], 1
    for it in range(iterations):
        X = mu + sd * rng.standard_normal((population, mu.size))
        if lower is not None or upper is not None:
            X = np.clip(X, lower if lower is not None else -np.inf, upper if upper is not None else np.inf)
        C = np.array([objective(x) for x in X], float)
        n += population
        order = np.argsort(C)
        E = X[order[:ne]]
        if C[order[0]] < best_c:
            best, best_c = X[order[0]].copy(), float(C[order[0]])
        mu, sd = E.mean(0), np.maximum(E.std(0), min_std)
        hist.append({"iteration": it + 1, "mean": mu.tolist(), "std": sd.tolist(), "best_cost": best_c,
                     "elite_cost": float(C[order[:ne]].mean()), "population_cost": float(np.median(C))})
    return CEMResult(best, best_c, hist, n)
