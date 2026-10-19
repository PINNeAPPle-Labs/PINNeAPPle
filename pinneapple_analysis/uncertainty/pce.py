"""Polynomial chaos expansion by least squares: output mean, variance and Sobol indices from one regression.

Every input is written as ``x_i = F_i^-1((xi_i + 1) / 2)`` with ``xi_i`` uniform on [-1, 1] (the inverse-CDF map of
:mod:`pinneapple_analysis.uncertainty.distributions`), and the output is fitted on the orthonormal Legendre
polynomials of the ``xi`` of total degree ``<= degree``. With orthonormal terms the mean is the constant
coefficient, the variance the sum of the other squared coefficients, and the Sobol indices are sums of squared
coefficients (Sudret 2008, Reliab. Eng. Syst. Saf. 93:964). The fit is only as good as its leave-one-out error
(``loo_q2``, computed exactly from the hat matrix); a low Q2 means more samples or a higher degree, or a model that
is not smooth enough for this expansion. It is cheap for a few inputs and a low degree: the number of terms is
``C(d + degree, degree)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from math import comb

import numpy as np
from numpy.polynomial import legendre

from pinneapple_analysis.uncertainty.distributions import as_space, to_values, unit_sample


def multi_indices(d: int, degree: int) -> np.ndarray:
    """All multi-indices of ``d`` variables with total degree ``<= degree`` (constant first)."""
    out = [np.zeros(d, dtype=int)]
    for p in range(1, degree + 1):
        for combo in combinations_with_replacement(range(d), p):
            a = np.zeros(d, dtype=int)
            for i in combo:
                a[i] += 1
            out.append(a)
    return np.array(out)


def legendre_design(xi: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Design matrix of orthonormal Legendre products (w.r.t. the uniform density on [-1, 1]^d)."""
    n, d = xi.shape
    pmax = int(alphas.max()) if alphas.size else 0
    P = np.ones((d, pmax + 1, n))
    for k in range(1, pmax + 1):
        c = np.zeros(k + 1)
        c[k] = 1.0
        P[:, k, :] = (legendre.legval(xi, c) * np.sqrt(2 * k + 1)).T
    Psi = np.ones((n, len(alphas)))
    for j, a in enumerate(alphas):
        for i in np.nonzero(a)[0]:
            Psi[:, j] *= P[i, a[i]]
    return Psi


@dataclass
class PCEResult:
    """Fitted expansion. Output arrays are per output (``m``)."""
    names: list[str]
    degree: int
    n_terms: int
    n_evals: int
    mean: np.ndarray
    std: np.ndarray
    S1: np.ndarray              # (m, d)
    ST: np.ndarray              # (m, d)
    loo_q2: np.ndarray          # (m,) 1 - mean squared LOO error / variance of the samples
    alphas: np.ndarray = field(repr=False)
    coef: np.ndarray = field(repr=False)     # (n_terms, m)

    def predict_xi(self, xi: np.ndarray) -> np.ndarray:
        return legendre_design(np.atleast_2d(xi), self.alphas) @ self.coef

    def to_dict(self) -> dict:
        return {"method": "polynomial chaos (Legendre, least squares)", "names": self.names, "degree": self.degree,
                "n_terms": self.n_terms, "n_evals": self.n_evals, "mean": self.mean.tolist(),
                "std": self.std.tolist(), "S1": self.S1.tolist(), "ST": self.ST.tolist(),
                "loo_q2": self.loo_q2.tolist()}


def n_terms(d: int, degree: int) -> int:
    return comb(d + degree, degree)


def polynomial_chaos(fn, space, degree: int = 3, n: int | None = None, seed: int = 0,
                     max_terms: int = 500) -> PCEResult:
    """Fit a total-degree Legendre chaos to ``fn`` on ``n`` Latin-hypercube samples of ``space``.

    ``n`` defaults to twice the number of terms. Raises ``ValueError`` when the expansion would exceed
    ``max_terms`` terms (it is meant for a few inputs; use Sobol sampling for many).
    """
    sp = as_space(space)
    d = len(sp)
    P = n_terms(d, degree)
    if P > max_terms:
        raise ValueError(f"{P} chaos terms for {d} inputs at degree {degree} (limit {max_terms}); lower the degree")
    n = int(n or 2 * P)
    if n <= P:
        raise ValueError(f"need more samples ({n}) than chaos terms ({P})")
    U = unit_sample(d, n, "lhs", seed)
    X = to_values(sp, U)
    Y = np.asarray(fn(X), dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    xi = 2.0 * U - 1.0
    alphas = multi_indices(d, degree)
    Psi = legendre_design(xi, alphas)
    coef, *_ = np.linalg.lstsq(Psi, Y, rcond=None)
    # exact leave-one-out residuals from the hat matrix (Q R of the design)
    Q, _ = np.linalg.qr(Psi)
    h = np.clip(np.sum(Q * Q, axis=1), 0.0, 1.0 - 1e-12)
    loo = (Y - Psi @ coef) / (1.0 - h)[:, None]
    var_y = Y.var(axis=0)
    q2 = 1.0 - np.mean(loo ** 2, axis=0) / np.where(var_y > 0, var_y, 1.0)
    c2 = coef[1:] ** 2                                  # (P-1, m)
    var = c2.sum(axis=0)
    safe = np.where(var > 0, var, 1.0)
    A = alphas[1:] > 0                                   # (P-1, d)
    only = A & (A.sum(axis=1, keepdims=True) == 1)
    S1 = (only.T.astype(float) @ c2).T / safe[:, None]
    ST = (A.T.astype(float) @ c2).T / safe[:, None]
    return PCEResult(names=[p.name for p in sp], degree=degree, n_terms=P, n_evals=n, mean=coef[0].copy(),
                     std=np.sqrt(var), S1=S1, ST=ST, loo_q2=q2, alphas=alphas, coef=coef)


__all__ = ["PCEResult", "polynomial_chaos", "multi_indices", "legendre_design", "n_terms"]
