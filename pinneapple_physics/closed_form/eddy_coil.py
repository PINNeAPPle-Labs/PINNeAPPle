"""Impedance of an air-cored eddy-current probe coil above a conductive plate (Dodd & Deeds, 1968).

C. V. Dodd and W. E. Deeds, "Analytical solutions to eddy-current probe-coil problems", J. Appl. Phys. 39(6),
2829-2838, 1968. The coil has a rectangular cross-section: inner radius r1, outer radius r2, and it spans the
heights l1 to l2 above the plate (lift-off l1). It carries N turns uniformly spread. The plate is non-magnetic,
with conductivity sigma and thickness c, and has air below it. With the turn density ``n = N / ((r2 - r1)(l2 - l1))``:

    Z_air = j w pi mu0 n^2 int_0^inf I(a)^2 / a^5 [2 (l2 - l1) + (2 / a)(exp(-a (l2 - l1)) - 1)] da
    dZ    = j w pi mu0 n^2 int_0^inf I(a)^2 / a^6 (exp(-a l1) - exp(-a l2))^2 phi(a) da
    I(a)  = int_{a r1}^{a r2} x J1(x) dx = [pi x / 2 (J1(x) H0(x) - J0(x) H1(x))]_{a r1}^{a r2}   (Struve H)
    phi   = r01 (1 - e^{-2 a1 c}) / (1 - r01^2 e^{-2 a1 c}),   r01 = (a - a1)/(a + a1),   a1 = sqrt(a^2 + j w mu0 sigma)

The quadrature is done on a mapped grid (Gauss-Legendre on panels in log a). The standard depth of penetration is
``delta = 1 / sqrt(pi f mu0 sigma)``.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import j0, j1, struve

__all__ = ["coil_impedance", "skin_depth", "MU0"]

MU0 = 4e-7 * math.pi


def _I(a, r1, r2):
    def F(x):
        return math.pi * x / 2 * (j1(x) * struve(0, x) - j0(x) * struve(1, x))
    return F(a * r2) - F(a * r1)


def skin_depth(f: float, sigma: float, mu_r: float = 1.0) -> float:
    """Skin depth for ``f`` in Hz and ``sigma`` in S/m (``MU0`` is in H/m).

    Returns
    -------
    delta : float [m]
        Standard depth of penetration ``1 / sqrt(pi f mu0 mu_r sigma)``.
    """
    return 1.0 / math.sqrt(math.pi * f * MU0 * mu_r * sigma)


def _grid(scale: float, n_panels: int = 120, order: int = 16):
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.logspace(-4, 3, n_panels + 1) / scale         # a from 1e-4 / r to 1e3 / r
    a, wt = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        a.append(0.5 * (hi - lo) * x + 0.5 * (hi + lo))
        wt.append(0.5 * (hi - lo) * w)
    return np.concatenate(a), np.concatenate(wt)


def coil_impedance(f: float, *, r1: float, r2: float, l1: float, l2: float, turns: float, sigma: float,
                   thickness: float = math.inf) -> dict:
    """Air-coil impedance, its change over the plate, the normalised impedance and the skin depth."""
    w = 2 * math.pi * f
    n = turns / ((r2 - r1) * (l2 - l1))
    a, wt = _grid(r2)
    I2 = _I(a, r1, r2) ** 2
    h = l2 - l1
    L0 = math.pi * MU0 * n * n * float(np.sum(wt * I2 / a ** 5 * (2 * h + 2 / a * (np.exp(-a * h) - 1))))
    a1 = np.sqrt(a * a + 1j * w * MU0 * sigma)
    r01 = (a - a1) / (a + a1)
    if math.isinf(thickness):
        phi = r01
    else:
        e = np.exp(-2 * a1 * thickness)
        phi = r01 * (1 - e) / (1 - r01 ** 2 * e)
    dZ = 1j * w * math.pi * MU0 * n * n * complex(np.sum(wt * I2 / a ** 6 * (np.exp(-a * l1) - np.exp(-a * l2)) ** 2
                                                         * phi))
    X0 = w * L0
    return {"L0": L0, "X0": X0, "dZ": dZ, "dR": dZ.real, "dX": dZ.imag, "R_norm": dZ.real / X0,
            "X_norm": (X0 + dZ.imag) / X0, "delta": skin_depth(f, sigma)}
