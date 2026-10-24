"""Euler-Bernoulli beam by Hermite finite elements: static deflection and natural modes.

Two degrees of freedom per node (deflection w, rotation theta), cubic Hermite shape functions, consistent mass
matrix (Przemieniecki, *Theory of Matrix Structural Analysis*, 1968, ch. 5 and 11). Supports: "pinned"
(simply supported), "fixed" (clamped) or "free" at each end. Converges to the exact Euler-Bernoulli answers: the
simply supported first mode ``f1 = pi / (2 L^2) sqrt(EI / m)`` and mid-span deflection ``5 q L^4 / (384 EI)``.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

__all__ = ["beam_matrices", "beam_static", "beam_modes"]


def _element(EI: float, m: float, h: float):
    k = EI / h ** 3 * np.array([[12, 6 * h, -12, 6 * h], [6 * h, 4 * h * h, -6 * h, 2 * h * h],
                                [-12, -6 * h, 12, -6 * h], [6 * h, 2 * h * h, -6 * h, 4 * h * h]])
    M = m * h / 420 * np.array([[156, 22 * h, 54, -13 * h], [22 * h, 4 * h * h, 13 * h, -3 * h * h],
                                [54, 13 * h, 156, -22 * h], [-13 * h, -3 * h * h, -22 * h, 4 * h * h]])
    return k, M


def beam_matrices(L: float, EI: float, m: float, n_el: int):
    """Global stiffness, consistent mass and node positions of a uniform beam."""
    h = L / n_el
    n = 2 * (n_el + 1)
    K, M = np.zeros((n, n)), np.zeros((n, n))
    ke, me = _element(EI, m, h)
    for e in range(n_el):
        i = slice(2 * e, 2 * e + 4)
        K[i, i] += ke
        M[i, i] += me
    return K, M, np.linspace(0, L, n_el + 1)


def _free(n_nodes: int, ends) -> np.ndarray:
    fixed = []
    for node, kind in zip((0, n_nodes - 1), ends, strict=False):
        if kind == "pinned":
            fixed.append(2 * node)
        elif kind == "fixed":
            fixed += [2 * node, 2 * node + 1]
        elif kind != "free":
            raise ValueError("supports are 'pinned', 'fixed' or 'free'")
    return np.setdiff1d(np.arange(2 * n_nodes), fixed)


def beam_static(L: float, EI: float, q: float, n_el: int = 40, ends=("pinned", "pinned")) -> dict:
    """Deflection under a uniform load ``q`` (N/m, positive down) with consistent nodal loads."""
    K, _, x = beam_matrices(L, EI, 1.0, n_el)
    h = L / n_el
    f = np.zeros(len(K))
    fe = q * np.array([h / 2, h * h / 12, h / 2, -h * h / 12])
    for e in range(n_el):
        f[2 * e:2 * e + 4] += fe
    fr = _free(len(x), ends)
    u = np.zeros(len(K))
    u[fr] = np.linalg.solve(K[np.ix_(fr, fr)], f[fr])
    R = K @ u - f
    return {"x": x, "w": u[0::2], "theta": u[1::2], "w_max": float(np.abs(u[0::2]).max()),
            "reaction_sum": float(R[0::2].sum()), "load": q * L}


def beam_modes(L: float, EI: float, m: float, n_el: int = 40, n_modes: int = 4, ends=("pinned", "pinned")) -> dict:
    """Natural frequencies (Hz) and mass-normalised mode shapes (deflection at the nodes)."""
    K, M, x = beam_matrices(L, EI, m, n_el)
    fr = _free(len(x), ends)
    w2, V = eigh(K[np.ix_(fr, fr)], M[np.ix_(fr, fr)])
    w2 = np.clip(w2[:n_modes], 0, None)
    full = np.zeros((len(K), n_modes))
    full[fr] = V[:, :n_modes]
    return {"x": x, "f": np.sqrt(w2) / (2 * np.pi), "shapes": full[0::2].T}
