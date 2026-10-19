"""Magnetic materials for 2D A_z magnetostatics: linear media, permanent magnets and nonlinear B-H curves.

A nonlinear material is described by its reluctivity as a function of ``s = |B|^2``: ``nu(s)`` with
``H = nu(s) B``, the derivative ``dnu/ds`` (for the Newton tangent ``dH/dB = nu I + 2 nu' B B^T``) and the
energy density ``w(s) = int_0^|B| H db = 0.5 int_0^s nu(t) dt`` (for the energy functional, the line
search and the co-energy torque). Both curve classes below are single-valued (no hysteresis).

``BHCurve`` interpolates measured (B, H) points with a monotone cubic (PCHIP), so H(B) stays increasing
and the energy stays convex; beyond the last point it continues with the vacuum slope ``1/mu0``.
``BrauerCurve`` is the closed form ``nu(s) = k1 exp(k2 s) + k3`` (J. R. Brauer, IEEE Trans. Magn. 11(1),
1975). ``illustrative_steel()`` is a smooth, saturating curve generated from a formula: it is NOT a
measured grade; use a supplier's data for design work.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import MU0

__all__ = ["MU0", "NU0", "LinearMaterial", "Magnet", "BHCurve", "BrauerCurve", "illustrative_steel",
           "AIR", "COPPER"]

NU0 = 1.0 / MU0


@dataclass(frozen=True)
class LinearMaterial:
    """Linear, isotropic medium: relative permeability ``mu_r`` and electrical conductivity ``sigma`` (S/m)."""

    mu_r: float = 1.0
    sigma: float = 0.0

    @property
    def nu(self) -> float:
        return NU0 / self.mu_r


@dataclass(frozen=True)
class Magnet:
    """Linear permanent magnet: ``B = mu0 mu_r H + Br``.

    ``Br`` is the remanence (T). ``direction`` is ``"radial"`` (outwards, times ``polarity``),
    ``"parallel"`` (along the pole axis given per element by the caller) or an angle in radians in the
    frame the caller supplies. ``sigma`` is used only by eddy-current post-processing.
    """

    Br: float
    mu_r: float = 1.05
    direction: str | float = "radial"
    sigma: float = 0.0

    @property
    def nu(self) -> float:
        return NU0 / self.mu_r


AIR = LinearMaterial(1.0)
COPPER = LinearMaterial(1.0, sigma=5.8e7)


class BHCurve:
    """Single-valued B-H curve from points, monotone cubic in H(B), vacuum slope above the last point."""

    def __init__(self, B, H):
        from scipy.interpolate import PchipInterpolator

        B = np.asarray(B, float)
        H = np.asarray(H, float)
        if B[0] != 0.0 or H[0] != 0.0:
            B = np.concatenate([[0.0], B])
            H = np.concatenate([[0.0], H])
        if np.any(np.diff(B) <= 0) or np.any(np.diff(H) <= 0):
            raise ValueError("B and H must be strictly increasing")
        self.B, self.H = B, H
        self._H = PchipInterpolator(B, H, extrapolate=False)
        self._dH = self._H.derivative()
        self._W = self._H.antiderivative()
        self._bmax = B[-1]
        self._Hmax = H[-1]
        self._Wmax = float(self._W(B[-1]))
        self.nu_init = float(self._dH(0.0))

    def h(self, b):
        b = np.asarray(b, float)
        out = np.where(b <= self._bmax, self._H(np.minimum(b, self._bmax)), 0.0)
        return np.where(b > self._bmax, self._Hmax + NU0 * (b - self._bmax), out)

    def dh(self, b):
        b = np.asarray(b, float)
        return np.where(b > self._bmax, NU0, self._dH(np.minimum(b, self._bmax)))

    def nu(self, s):
        b = np.sqrt(np.asarray(s, float))
        small = b < 1e-8
        bb = np.where(small, 1.0, b)
        return np.where(small, self.nu_init, self.h(bb) / bb)

    def dnu(self, s):
        """d nu / d s with s = |B|^2: (H'(b) b - H(b)) / (2 b^3)."""
        b = np.sqrt(np.asarray(s, float))
        small = b < 1e-4
        bb = np.where(small, 1.0, b)
        return np.where(small, 0.0, (self.dh(bb) * bb - self.h(bb)) / (2.0 * bb ** 3))

    def energy(self, s):
        """w = int_0^|B| H db (J/m^3)."""
        b = np.sqrt(np.asarray(s, float))
        inside = self._W(np.minimum(b, self._bmax))
        extra = self._Hmax * (b - self._bmax) + 0.5 * NU0 * (b - self._bmax) ** 2
        return np.where(b > self._bmax, self._Wmax + extra, inside)


class BrauerCurve:
    """``nu(s) = k1 exp(k2 s) + k3`` with ``s = |B|^2`` (closed-form energy and derivative)."""

    def __init__(self, k1: float, k2: float, k3: float):
        self.k1, self.k2, self.k3 = k1, k2, k3
        self.nu_init = k1 + k3

    def nu(self, s):
        return self.k1 * np.exp(self.k2 * np.asarray(s, float)) + self.k3

    def dnu(self, s):
        return self.k1 * self.k2 * np.exp(self.k2 * np.asarray(s, float))

    def energy(self, s):
        s = np.asarray(s, float)
        return 0.5 * (self.k1 / self.k2 * np.expm1(self.k2 * s) + self.k3 * s)

    def h(self, b):
        b = np.asarray(b, float)
        return self.nu(b * b) * b


def illustrative_steel(mu_r_init: float = 4000.0, Js: float = 2.0, n: int = 60) -> BHCurve:
    """Smooth saturating curve ``B = mu0 H + Js (2/pi) atan(pi (mu_r_init - 1) mu0 H / (2 Js))``.

    Initial relative permeability ``mu_r_init``, saturation polarization ``Js`` (T). Generated from a
    formula for examples and tests -- not a measured electrical-steel grade.
    """
    H = np.concatenate([[0.0], np.logspace(0, 6, n)])
    B = MU0 * H + Js * (2.0 / math.pi) * np.arctan(math.pi * (mu_r_init - 1.0) * MU0 * H / (2.0 * Js))
    return BHCurve(B, H)
