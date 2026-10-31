"""Tensor-product Lagrange bases on the unit cell [0, 1]^d (Q1 trilinear, Q2 triquadratic, any order).

Local nodes are equispaced, ordered C-style over the axes (the last axis fastest), so the local node with indices
``(a_0, ..., a_{d-1})`` has number ``ravel_multi_index(a, (p + 1,) * d)``. Derivatives are with respect to the unit
cell coordinates; divide by the cell size to get physical derivatives.
"""
from __future__ import annotations

from functools import cache
from itertools import product

import numpy as np


@cache
def _coeffs(p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Polynomial coefficients (highest power first) of the p + 1 Lagrange polynomials on equispaced nodes in [0, 1],
    with their first and second derivatives."""
    x = np.linspace(0.0, 1.0, p + 1)
    c = np.zeros((p + 1, p + 1))
    for a in range(p + 1):
        others = np.delete(x, a)
        poly = np.poly(others) / np.prod(x[a] - others)
        c[a] = poly
    d1 = np.array([np.pad(np.polyder(r), (1, 0)) for r in c])
    d2 = np.array([np.pad(np.polyder(r, 2), (2, 0)) if p >= 2 else np.zeros(p + 1) for r in c])
    return c, d1, d2


def lagrange_1d(p: int, x: np.ndarray, deriv: int = 0) -> np.ndarray:
    """Values (``deriv`` 0, 1 or 2) of the p + 1 Lagrange polynomials at ``x``: shape (len(x), p + 1)."""
    c = _coeffs(int(p))[deriv]
    x = np.asarray(x, float)
    powers = x[:, None] ** np.arange(p, -1, -1)[None, :]
    return powers @ c.T


def n_local(p: int, d: int) -> int:
    return (p + 1) ** d


def local_offsets(p: int, d: int) -> np.ndarray:
    """(nb, d) integer offsets of the local nodes inside a cell, in local-node order."""
    return np.array(list(product(range(p + 1), repeat=d)), dtype=np.int64)


def _tensor(factors: list[np.ndarray]) -> np.ndarray:
    """Outer product over the axes of per-axis (n, p + 1) tables, flattened to (n, (p + 1)^d)."""
    out = factors[0]
    for f in factors[1:]:
        out = (out[:, :, None] * f[:, None, :]).reshape(len(out), -1)
    return out


def shape(p: int, xi: np.ndarray) -> np.ndarray:
    """Basis values at unit-cell coordinates ``xi`` (n, d): shape (n, nb)."""
    xi = np.atleast_2d(xi)
    return _tensor([lagrange_1d(p, xi[:, k]) for k in range(xi.shape[1])])


def grad(p: int, xi: np.ndarray) -> np.ndarray:
    """Basis gradients in unit-cell coordinates: shape (n, nb, d)."""
    xi = np.atleast_2d(xi)
    d = xi.shape[1]
    L = [lagrange_1d(p, xi[:, k]) for k in range(d)]
    D = [lagrange_1d(p, xi[:, k], 1) for k in range(d)]
    return np.stack([_tensor([D[k] if j == k else L[j] for j in range(d)]) for k in range(d)], axis=-1)


def normal_derivative(p: int, xi: np.ndarray, axis: int, order: int) -> np.ndarray:
    """``order``-th derivative of every basis function along one axis (unit-cell coordinates): shape (n, nb)."""
    xi = np.atleast_2d(xi)
    d = xi.shape[1]
    return _tensor([lagrange_1d(p, xi[:, j], order if j == axis else 0) for j in range(d)])


@cache
def gauss_cell(n: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Tensor Gauss-Legendre rule with n points per axis on [0, 1]^d: points (n^d, d) and weights summing to 1."""
    x, w = np.polynomial.legendre.leggauss(n)
    x, w = (x + 1) / 2, w / 2
    pts = np.array(list(product(x, repeat=d)))
    wts = np.array([np.prod(c) for c in product(w, repeat=d)])
    return pts, wts
