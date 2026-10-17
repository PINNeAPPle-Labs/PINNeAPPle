"""Closed-form references for verifying the 2D A_z solvers.

* :func:`wire_field` -- round conductor with uniform current (Ampere's law).
* :func:`coax_field` -- coaxial cable, inner conductor ``+I``, outer ``-I``.
* :func:`magnetized_cylinder` -- uniformly magnetized cylinder (``mu_r = 1``) inside a circular
  boundary with ``A = 0``: ``A = f(r) sin(phi - alpha)``; inside the field is uniform,
  ``B = (Br/2)(1 - a^2/R^2)`` along the magnetization (``Br/2`` as ``R -> infinity``).
* :func:`skin_effect_wire` -- current density and per-unit-length impedance of a round wire carrying
  a sinusoidal current (Bessel functions of complex argument).
* :class:`RadialLayers` -- one spatial harmonic ``cos(p phi), sin(p phi)`` of a field in concentric
  layers of linear media with radially magnetized magnets and current densities: each layer obeys
  ``f'' + f'/r - p^2 f/r^2 = rhs(r)`` and the interfaces carry continuity of ``A`` and of
  ``H_phi = -nu dA/dr``. This is the subdomain (exact 2D) solution of a slotless machine, used to check
  the air-gap field and the torque of the FEM model.

All quantities SI. Torque per metre of stack in air between radii ``r`` of a field
``A = a(r) cos(p phi) + b(r) sin(p phi)`` from the Maxwell stress tensor is
``T = (p pi r / mu0) (a b' - a' b)`` (derived in :meth:`RadialLayers.torque`).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .materials import MU0

__all__ = ["wire_field", "coax_field", "magnetized_cylinder", "magnetized_ring", "skin_effect_wire", "Layer",
           "RadialLayers", "round_conductor_eddy_loss_low_freq"]


def wire_field(r, current: float, radius: float, R_outer: float):
    """(A_z, |B|) of a round wire (uniform J) centred in a circle of radius ``R_outer`` with A=0."""
    r = np.asarray(r, float)
    k = MU0 * current / (2.0 * math.pi)
    A_out = -k * np.log(np.maximum(r, 1e-300) / R_outer)
    A_in = -k * math.log(radius / R_outer) + 0.5 * k * (1.0 - (r / radius) ** 2)
    A = np.where(r >= radius, A_out, A_in)
    B = np.where(r >= radius, k / np.maximum(r, 1e-300), k * r / radius ** 2)
    return A, B


def coax_field(r, current: float, a: float, b: float, c: float):
    """|B| of a coax: inner conductor radius ``a`` (+I), outer conductor ``b < r < c`` (-I)."""
    r = np.asarray(r, float)
    k = MU0 * current / (2.0 * math.pi)
    rr = np.maximum(r, 1e-300)
    B = np.where(r < a, k * r / a ** 2, k / rr)
    shell = (r >= b) & (r < c)
    B = np.where(shell, k / rr * (c ** 2 - r ** 2) / (c ** 2 - b ** 2), B)
    return np.where(r >= c, 0.0, B)


def magnetized_cylinder(x, y, Br: float, a: float, R: float, alpha: float = 0.0):
    """A_z of a cylinder (radius ``a``, ``mu_r = 1``, remanence ``Br`` along angle ``alpha``) in air with
    ``A = 0`` at radius ``R``. Equivalent surface current ``K = M sin(phi - alpha)``; returns (A, B_inside)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    c = 0.5 * Br * a * a  # mu0 M a^2 / 2
    f = np.where(r <= a, (c / a ** 2 - c / R ** 2) * r, c * (1.0 / np.maximum(r, 1e-300) - r / R ** 2))
    return f * np.sin(phi - alpha), 0.5 * Br * (1.0 - a * a / (R * R))


def magnetized_ring(x, y, Br: float, a: float, b: float, r0: float, R: float, alpha: float = 0.0):
    """A_z of a uniformly magnetised ring ``a < r < b`` (``mu_r = 1``, remanence ``Br`` along ``alpha``) in air
    between two circles with ``A = 0``: ``r0 < a`` and ``R > b``. Superposition of two cylinders
    (:func:`magnetized_cylinder`, radii ``b`` and ``a``) plus the harmonic ``C (1/r - r/R^2) sin(phi - alpha)``
    that cancels A on ``r0``."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = np.hypot(x, y)
    A = magnetized_cylinder(x, y, Br, b, R, alpha)[0] - magnetized_cylinder(x, y, Br, a, R, alpha)[0]
    f_r0 = 0.5 * Br * (a * a - b * b) / R ** 2 * r0  # the two cylinders' uniform inner fields, at r0
    C = -f_r0 / (1.0 / r0 - r0 / R ** 2)
    return A + C * (1.0 / np.maximum(r, 1e-300) - r / R ** 2) * np.sin(np.arctan2(y, x) - alpha)


def skin_effect_wire(r, current: complex, radius: float, sigma: float, omega: float, mu: float = MU0):
    """Phasor current density J(r) and impedance per metre Z = R + j omega L_int of a round wire.

    ``J = k I J0(k r) / (2 pi a J1(k a))`` with ``k^2 = -j omega mu sigma`` (time convention exp(+j omega t)).
    """
    from scipy.special import jv

    k = np.sqrt(-1j * omega * mu * sigma)
    J = k * current * jv(0, k * np.asarray(r)) / (2.0 * math.pi * radius * jv(1, k * radius))
    Z = k * jv(0, k * radius) / (2.0 * math.pi * radius * sigma * jv(1, k * radius))
    return J, Z


def round_conductor_eddy_loss_low_freq(B_peak: float, omega: float, sigma: float, a: float) -> float:
    """Time-averaged eddy loss per metre in a round conductor (radius ``a``, no net current) in a uniform
    transverse field ``B_peak sin(omega t)``, resistance limited (field of the eddy currents neglected):
    ``P = pi sigma omega^2 B^2 a^4 / 8`` W/m."""
    return math.pi * sigma * omega ** 2 * B_peak ** 2 * a ** 4 / 8.0


@dataclass
class Layer:
    """Annulus ``r_in < r < r_out`` with relative permeability ``mu_r``; radial remanence
    ``Br_cos cos(p phi) + Br_sin sin(p phi)`` and current density ``J_cos cos(p phi) + J_sin sin(p phi)``."""

    r_in: float
    r_out: float
    mu_r: float = 1.0
    Br_cos: float = 0.0
    Br_sin: float = 0.0
    J_cos: float = 0.0
    J_sin: float = 0.0


def _particular(p: int, kind: str, c: float):
    """Particular solution of f'' + f'/r - p^2 f / r^2 = c * r^m, kind 'm=-1' (magnet) or 'm=0' (current)."""
    if kind == "inv_r":  # rhs = c / r
        if p == 1:
            return (lambda r: 0.5 * c * r * np.log(r)), (lambda r: 0.5 * c * (np.log(r) + 1.0))
        k = c / (1.0 - p * p)
        return (lambda r: k * r), (lambda r: k + 0 * r)
    if p == 2:  # rhs = c
        return (lambda r: 0.25 * c * r * r * np.log(r)), (lambda r: 0.25 * c * (2 * r * np.log(r) + r))
    k = c / (4.0 - p * p)
    return (lambda r: k * r * r), (lambda r: 2.0 * k * r)


class RadialLayers:
    """Exact solution for one harmonic ``p >= 1`` in concentric layers; ``A = 0`` at the outer radius,
    regular at ``r = 0`` if the first layer starts there (otherwise ``dA/dr = 0``, an ideal iron wall)."""

    def __init__(self, layers: Sequence[Layer], p: int):
        if p < 1:
            raise ValueError("p >= 1")
        self.layers = list(layers)
        self.p = p
        for comp in ("cos", "sin"):
            setattr(self, f"_c_{comp}", self._solve(comp))

    def _parts(self, L: Layer, comp: str):
        """f_p, f_p' of the layer for one component. A = a cos + b sin.

        Magnet: laplacian(A) = (1/r) dBr_r/dphi; for Br_r = Bc cos + Bs sin, the cos part of the source
        is p Bs / r and the sin part -p Bc / r. Current: laplacian(A) = -mu J.
        """
        p = self.p
        mu = MU0 * L.mu_r
        cm = p * L.Br_sin if comp == "cos" else -p * L.Br_cos
        cj = -mu * (L.J_cos if comp == "cos" else L.J_sin)
        f1, d1 = _particular(p, "inv_r", cm)
        f2, d2 = _particular(p, "const", cj)
        return (lambda r: f1(r) + f2(r)), (lambda r: d1(r) + d2(r))

    def _solve(self, comp: str):
        p, Ls = self.p, self.layers
        n = len(Ls)
        M = np.zeros((2 * n, 2 * n))
        rhs = np.zeros(2 * n)
        row = 0
        parts = [self._parts(L, comp) for L in Ls]
        # inner condition
        L0 = Ls[0]
        if L0.r_in == 0.0:
            M[row, 1] = 1.0  # no r^-p term
        else:  # ideal iron wall: dA/dr = 0
            r = L0.r_in
            M[row, 0], M[row, 1] = p * r ** (p - 1), -p * r ** (-p - 1)
            rhs[row] = -parts[0][1](r)
        row += 1
        for k in range(n - 1):
            r = Ls[k].r_out
            nuk, nun = 1.0 / Ls[k].mu_r, 1.0 / Ls[k + 1].mu_r
            fk, dk = parts[k]
            fn, dn = parts[k + 1]
            # continuity of A
            M[row, 2 * k:2 * k + 2] = [r ** p, r ** -p]
            M[row, 2 * k + 2:2 * k + 4] = [-(r ** p), -(r ** -p)]
            rhs[row] = fn(r) - fk(r)
            row += 1
            # continuity of nu dA/dr
            M[row, 2 * k:2 * k + 2] = [nuk * p * r ** (p - 1), -nuk * p * r ** (-p - 1)]
            M[row, 2 * k + 2:2 * k + 4] = [-nun * p * r ** (p - 1), nun * p * r ** (-p - 1)]
            rhs[row] = nun * dn(r) - nuk * dk(r)
            row += 1
        r = Ls[-1].r_out
        M[row, 2 * n - 2:2 * n] = [r ** p, r ** -p]
        rhs[row] = -parts[-1][0](r)
        coef = np.linalg.solve(M, rhs)
        return coef, parts

    def _eval(self, r, comp):
        coef, parts = getattr(self, f"_c_{comp}")
        r = np.atleast_1d(np.asarray(r, float))
        f = np.zeros_like(r)
        d = np.zeros_like(r)
        p = self.p
        for k, L in enumerate(self.layers):
            m = (r >= L.r_in) & (r <= L.r_out)
            rr = np.where(r[m] == 0.0, 1e-300, r[m])
            c1, c2 = coef[2 * k], coef[2 * k + 1]
            fp, dp = parts[k]
            f[m] = c1 * rr ** p + c2 * rr ** (-p) + fp(rr)
            d[m] = c1 * p * rr ** (p - 1) - c2 * p * rr ** (-p - 1) + dp(rr)
        return f, d

    def A(self, r, phi):
        a, _ = self._eval(r, "cos")
        b, _ = self._eval(r, "sin")
        return a * np.cos(self.p * phi) + b * np.sin(self.p * phi)

    def B_polar(self, r, phi):
        """(B_r, B_phi) with B_r = (1/r) dA/dphi, B_phi = -dA/dr."""
        p = self.p
        a, da = self._eval(r, "cos")
        b, db = self._eval(r, "sin")
        r = np.asarray(r, float)
        Br = p / r * (-a * np.sin(p * phi) + b * np.cos(p * phi))
        Bphi = -(da * np.cos(p * phi) + db * np.sin(p * phi))
        return Br, Bphi

    def torque(self, r: float, length: float = 1.0) -> float:
        """Torque on everything inside radius ``r`` (an air radius), N m:
        ``(L/mu0) r^2 int B_r B_phi dphi = (L p pi r / mu0)(a b' - a' b)``."""
        a, da = self._eval(r, "cos")
        b, db = self._eval(r, "sin")
        return float(length * self.p * math.pi * r * (a[0] * db[0] - da[0] * b[0]) / MU0)
