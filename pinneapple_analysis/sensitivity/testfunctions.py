"""Benchmark functions with closed-form Sobol indices, to check a sensitivity estimator against the exact answer.

- Ishigami & Homma (1990): ``f = sin x1 + a sin^2 x2 + b x3^4 sin x1`` on ``[-pi, pi]^3``; indices from Sobol &
  Levitan (1999), also tabulated in Saltelli et al. (2008), *Global Sensitivity Analysis: The Primer*, sec. 4.9.
- Sobol' G function: ``f = prod (|4 x_i - 2| + a_i) / (1 + a_i)`` on ``[0, 1]^d``; partial variances
  ``V_i = 1 / (3 (1 + a_i)^2)`` (Saltelli & Sobol' 1995).
"""
from __future__ import annotations

import numpy as np


def ishigami(X, a: float = 7.0, b: float = 0.1) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.sin(X[:, 0]) + a * np.sin(X[:, 1]) ** 2 + b * X[:, 2] ** 4 * np.sin(X[:, 0])


def ishigami_space() -> list[dict]:
    return [{"name": f"x{i + 1}", "dist": "uniform", "low": -np.pi, "high": np.pi} for i in range(3)]


def ishigami_indices(a: float = 7.0, b: float = 0.1) -> dict:
    """Exact variance, first-order, total and second-order (x1, x3) indices of the Ishigami function."""
    pi = np.pi
    V1 = 0.5 * (1 + b * pi ** 4 / 5) ** 2
    V2 = a ** 2 / 8
    V13 = b ** 2 * pi ** 8 * (1 / 18 - 1 / 50)
    V = V1 + V2 + V13
    return {"V": V, "S1": np.array([V1, V2, 0.0]) / V, "ST": np.array([V1 + V13, V2, V13]) / V,
            "S2": {(0, 2): V13 / V}}


def sobol_g(X, a) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, dtype=float))
    a = np.asarray(a, dtype=float)
    return np.prod((np.abs(4 * X - 2) + a) / (1 + a), axis=1)


def sobol_g_space(d: int) -> list[dict]:
    return [{"name": f"x{i + 1}", "dist": "uniform", "low": 0.0, "high": 1.0} for i in range(d)]


def sobol_g_indices(a) -> dict:
    """Exact variance, first-order and total indices of the G function with coefficients ``a``."""
    a = np.asarray(a, dtype=float)
    Vi = 1.0 / (3.0 * (1.0 + a) ** 2)
    V = np.prod(1 + Vi) - 1
    ST = np.array([Vi[i] * np.prod(np.delete(1 + Vi, i)) for i in range(a.size)]) / V
    return {"V": V, "S1": Vi / V, "ST": ST}


__all__ = ["ishigami", "ishigami_space", "ishigami_indices", "sobol_g", "sobol_g_space", "sobol_g_indices"]
