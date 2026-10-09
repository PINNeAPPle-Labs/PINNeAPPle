"""Airfoil shapes: CST (Kulfan) with thickness and camber weights, NACA 4-digit, derived properties.

A shape is 6 numbers: thickness weights T0..T2 and camber weights M0..M2 of a 2nd-order Bernstein CST,
upper = M + T, lower = M - T, so the thickness is positive by construction. Class function sqrt(x)(1-x) and a
fixed blunt trailing edge (0.25 % chord) so the mesh has a trailing-edge face.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Dict, Tuple

import numpy as np

PARAMS = ("T0", "T1", "T2", "M0", "M1", "M2")
BOUNDS = np.array([[0.10, 0.24], [0.08, 0.22], [0.08, 0.22],           # thickness weights
                   [-0.03, 0.08], [-0.02, 0.14], [-0.04, 0.12]])       # camber weights
LABELS = {"T0": "thickness, leading edge", "T1": "thickness, mid chord", "T2": "thickness, aft",
          "M0": "camber, front", "M1": "camber, mid chord", "M2": "camber, aft"}
TE_THICKNESS = 0.0025
N_CST = 2


def _bern(n: int, x: np.ndarray) -> np.ndarray:
    return np.stack([comb(n, i) * x ** i * (1 - x) ** (n - i) for i in range(n + 1)], -1)


def cst_surface(w: np.ndarray, x: np.ndarray, te: float) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.sqrt(x) * (1 - x) * (_bern(len(w) - 1, x) @ w) + x * te


def surfaces(shape: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Upper and lower y at chord stations x (chord 1, leading edge at 0)."""
    s = np.asarray(shape, float)
    T, M = s[:3], s[3:6]
    return cst_surface(M + T, x, TE_THICKNESS / 2), cst_surface(M - T, x, -TE_THICKNESS / 2)


def fit_cst(xu, yu, xl, yl) -> np.ndarray:
    """Least-squares CST weights for given upper/lower coordinates -> shape vector (T, M)."""
    def one(xs, ys, te):
        xs = np.clip(np.asarray(xs, float), 1e-9, 1)
        A = (np.sqrt(xs) * (1 - xs))[:, None] * _bern(N_CST, xs)
        return np.linalg.lstsq(A, np.asarray(ys) - xs * te, rcond=None)[0]
    wu, wl = one(xu, yu, TE_THICKNESS / 2), one(xl, yl, -TE_THICKNESS / 2)
    return np.concatenate([(wu - wl) / 2, (wu + wl) / 2])


def naca4(code: str, n: int = 200) -> np.ndarray:
    """CST fit of a NACA 4-digit airfoil, e.g. naca4("2412")."""
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    b = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(b))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    if m > 0:
        yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
        dy = np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    else:
        yc = dy = 0 * x
    th = np.arctan(dy)
    return fit_cst(x - yt * np.sin(th), yc + yt * np.cos(th), x + yt * np.sin(th), yc - yt * np.cos(th))


REFERENCE = {"NACA 2412": naca4("2412"), "NACA 4412": naca4("4412"), "NACA 0012": naca4("0012"),
             "NACA 4415": naca4("4415")}


def properties(shape: np.ndarray) -> Dict[str, float]:
    """Max thickness and camber (fraction of chord) and where they are, LE radius, area (fuel/spar volume proxy)."""
    x = np.linspace(0, 1, 401)
    yu, yl = surfaces(shape, x)
    t, c = yu - yl, (yu + yl) / 2
    i, k = int(np.argmax(t)), int(np.argmax(np.abs(c)))
    T0 = float(shape[0])
    return {"thickness": float(t[i]), "thickness_at": float(x[i]), "camber": float(c[k]), "camber_at": float(x[k]),
            "le_radius": float(T0 ** 2 / 2),                     # CST: r_LE = w0^2 / 2 for each surface (T0 ~ average)
            "area": float(np.trapezoid(t, x)), "min_thickness_aft": float(t[x >= 0.9].min()),
            "spar_depth": float(t[(x >= 0.2) & (x <= 0.35)].min())}


def valid(shape: np.ndarray) -> Tuple[bool, str]:
    x = np.linspace(0, 1, 401)
    yu, yl = surfaces(shape, x)
    t = yu - yl
    if (t[1:-1] <= 0).any():
        return False, "surfaces cross"
    d2 = np.gradient(np.gradient(yu, x), x)
    if (np.abs(d2[20:-5]) > 40).any():
        return False, "wavy surface"
    return True, ""


def outline(shape: np.ndarray, n: int = 80) -> Dict[str, list]:
    b = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(b))
    yu, yl = surfaces(shape, x)
    return {"x": x.round(5).tolist(), "upper": yu.round(5).tolist(), "lower": yl.round(5).tolist()}


def latin_hypercube(n: int, bounds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    d = len(bounds)
    u = (np.argsort(rng.random((d, n)), axis=1).T + rng.random((n, d))) / n
    return bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])


@dataclass
class Loop:
    """Closed wall loop for the O-grid: TE base (upper->lower), lower TE->LE, upper LE->TE."""
    xy: np.ndarray
    n_te: int


def wall_loop(shape: np.ndarray, n_surf: int = 80, n_te: int = 4) -> Loop:
    b = np.linspace(0, np.pi, n_surf + 1)
    x = 0.5 * (1 - np.cos(b))                                    # LE->TE, clustered at both ends
    yu, yl = surfaces(shape, x)
    lower = np.stack([x[::-1], yl[::-1]], 1)                     # TE -> LE
    upper = np.stack([x, yu], 1)                                 # LE -> TE
    te = np.stack([np.ones(n_te + 1), np.linspace(yu[-1], yl[-1], n_te + 1)], 1)
    xy = np.concatenate([te[:-1], lower[:-1], upper[:-1]])      # closed, no repeated point
    return Loop(xy, n_te)
