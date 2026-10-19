"""Nonlinear magnetisation: the Froehlich-Kennelly saturation curve and the Jiles-Atherton hysteresis model.

:class:`FrohlichKennelly` is single-valued: ``B = mu0 H + H / (alpha + beta |H|)`` (Froehlich 1881, Kennelly
1891; the polarisation saturates at ``1/beta``, the initial relative permeability is
``1 + 1 / (mu0 alpha)``). It exposes the interface of :class:`.materials.BHCurve` (``h``, ``nu``, ``dnu``,
``energy`` as functions of ``s = |B|^2``) with every quantity in closed form, so it plugs straight into the
Newton solver of :mod:`.fem`.

:class:`JilesAtherton` is the hysteresis model of D. C. Jiles and D. L. Atherton, *Theory of ferromagnetic
hysteresis*, J. Magn. Magn. Mater. 61 (1986) 48-60. With the effective field ``He = H + alpha M``, the
anhysteretic ``Man = Ms L(He / a)`` (``L(x) = coth x - 1/x``), the split ``M = Mrev + Mirr``,
``Mrev = c (Man - Mirr)`` and the pinning law ``dMirr/dHe = delta_M (Man - Mirr) / (delta k)``
(``delta = sign(dH/dt)``; ``delta_M = 1`` when ``delta (Man - M) > 0``, else 0, which keeps the irreversible
susceptibility non-negative), the chain rule gives, with ``u = delta_M (Man - M) / (delta k)`` and
``g = dMan/dHe``::

    dM/dH = (u + c g) / (1 - alpha (u + c g))

This is derived from the definitions above rather than taken from a later paper, several of which use
inequivalent rearrangements. It is integrated along an applied ``H(t)`` path with classical RK4 on each
monotone segment; the step error is measured in the tests. ``loss_per_cycle`` is ``oint H dB``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import MU0

__all__ = ["FrohlichKennelly", "JilesAtherton", "langevin", "HysteresisLoop"]


@dataclass(frozen=True)
class FrohlichKennelly:
    """``B = mu0 H + H / (alpha + beta H)`` for ``H >= 0`` (odd extension). ``alpha`` in m/H, ``beta`` in 1/T."""

    alpha: float
    beta: float

    @classmethod
    def from_saturation(cls, mu_r_init: float, Js: float) -> FrohlichKennelly:
        """Initial relative permeability and saturation polarisation ``Js`` (T)."""
        return cls(1.0 / (MU0 * (mu_r_init - 1.0)), 1.0 / Js)

    @property
    def nu_init(self) -> float:
        return 1.0 / (MU0 + 1.0 / self.alpha)

    def b(self, h):
        h = np.asarray(h, float)
        ah = np.abs(h)
        return np.sign(h) * (MU0 * ah + ah / (self.alpha + self.beta * ah))

    def h(self, b):
        """Inverse of :meth:`b` for ``b >= 0`` (positive root of ``mu0 beta H^2 + (mu0 alpha + 1 - beta b) H
        - alpha b = 0``), written in the cancellation-free form ``2 c / (-q - sqrt(q^2 - 4 a c))``."""
        b = np.asarray(b, float)
        ab = np.abs(b)
        qa = MU0 * self.beta
        qb = MU0 * self.alpha + 1.0 - self.beta * ab
        qc = -self.alpha * ab
        disc = np.sqrt(qb * qb - 4.0 * qa * qc)
        H = np.where(qb > 0, -2.0 * qc / (qb + disc), (-qb + disc) / (2.0 * qa))
        return np.sign(b) * H

    def dbdh(self, h):
        ah = np.abs(np.asarray(h, float))
        return MU0 + self.alpha / (self.alpha + self.beta * ah) ** 2

    def dh(self, b):
        return 1.0 / self.dbdh(self.h(b))

    def nu(self, s):
        b = np.sqrt(np.asarray(s, float))
        small = b < 1e-12
        bb = np.where(small, 1.0, b)
        return np.where(small, self.nu_init, self.h(bb) / bb)

    def dnu(self, s):
        """d nu / d s, ``s = |B|^2``: ``(H'(b) b - H(b)) / (2 b^3)``, evaluated at ``b >= 1e-6`` T (the limit at
        ``b -> 0`` is finite)."""
        b = np.sqrt(np.asarray(s, float))
        bb = np.maximum(b, 1e-6)
        return (self.dh(bb) * bb - self.h(bb)) / (2.0 * bb ** 3)

    def energy(self, s):
        """``w(b) = int_0^b H db' = b H(b) - int_0^H(b) B dh`` (Legendre), closed form."""
        b = np.sqrt(np.asarray(s, float))
        H = self.h(b)
        a, be = self.alpha, self.beta
        coen = 0.5 * MU0 * H * H + H / be - a / be ** 2 * np.log1p(be * H / a)
        return b * H - coen


def langevin(x):
    """``L(x) = coth x - 1/x`` with its series ``x/3 - x^3/45`` near 0."""
    x = np.asarray(x, float)
    small = np.abs(x) < 1e-3
    xs = np.where(small, 1.0, x)
    return np.where(small, x / 3.0 - x ** 3 / 45.0, 1.0 / np.tanh(xs) - 1.0 / xs)


def _dlangevin(x):
    x = np.asarray(x, float)
    small = np.abs(x) < 1e-3
    xs = np.where(small, 1.0, x)
    return np.where(small, 1.0 / 3.0 - x * x / 15.0, 1.0 / xs ** 2 - 1.0 / np.sinh(xs) ** 2)


@dataclass
class HysteresisLoop:
    H: np.ndarray
    M: np.ndarray

    @property
    def B(self) -> np.ndarray:
        return MU0 * (self.H + self.M)

    def loss_per_cycle(self, start: int = 0) -> float:
        """``oint H dB`` (J/m^3) over the samples from ``start`` (the caller passes a closed cycle)."""
        H, B = self.H[start:], self.B[start:]
        return float(np.sum(0.5 * (H[1:] + H[:-1]) * np.diff(B)))

    def coercivity(self) -> float:
        """|H| where B crosses zero on the descending branch (linear interpolation)."""
        B, H = self.B, self.H
        dH = np.diff(H)
        for i in range(len(B) - 1):
            if dH[i] < 0 and B[i] > 0 >= B[i + 1]:
                return float(abs(H[i] - B[i] * (H[i + 1] - H[i]) / (B[i + 1] - B[i])))
        return float("nan")

    def remanence(self) -> float:
        """B where H crosses zero on the descending branch (T)."""
        B, H = self.B, self.H
        for i in range(len(H) - 1):
            if H[i] > 0 >= H[i + 1]:
                return float(B[i] - H[i] * (B[i + 1] - B[i]) / (H[i + 1] - H[i]))
        return float("nan")


@dataclass(frozen=True)
class JilesAtherton:
    """Jiles-Atherton parameters: saturation magnetisation ``Ms`` (A/m), anhysteretic shape ``a`` (A/m),
    pinning ``k`` (A/m), reversibility ``c`` (0..1), inter-domain coupling ``alpha``."""

    Ms: float
    a: float
    k: float
    c: float
    alpha: float

    def anhysteretic(self, He):
        return self.Ms * langevin(np.asarray(He, float) / self.a)

    def dMdH(self, H: float, M: float, delta: float) -> float:
        He = H + self.alpha * M
        Man = self.Ms * float(langevin(He / self.a))
        g = self.Ms / self.a * float(_dlangevin(He / self.a))
        diff = Man - M
        u = diff / (delta * self.k) if diff * delta > 0 else 0.0
        x = u + self.c * g
        return float(x / (1.0 - self.alpha * x))

    def trace(self, H_path, M0: float = 0.0, substeps: int = 4) -> HysteresisLoop:
        """Integrate ``M`` along the applied field samples ``H_path`` (RK4 with ``substeps`` steps between
        samples; the direction ``delta`` is that of each segment)."""
        H_path = np.asarray(H_path, float)
        M = np.empty_like(H_path)
        M[0] = M0
        for i in range(len(H_path) - 1):
            h0, h1 = H_path[i], H_path[i + 1]
            if h1 == h0:
                M[i + 1] = M[i]
                continue
            delta = math.copysign(1.0, h1 - h0)
            dh = (h1 - h0) / substeps
            m, h = M[i], h0
            for _ in range(substeps):
                k1 = self.dMdH(h, m, delta)
                k2 = self.dMdH(h + 0.5 * dh, m + 0.5 * dh * k1, delta)
                k3 = self.dMdH(h + 0.5 * dh, m + 0.5 * dh * k2, delta)
                k4 = self.dMdH(h + dh, m + dh * k3, delta)
                m += dh * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
                h += dh
            M[i + 1] = m
        return HysteresisLoop(H_path, M)

    def major_loop(self, H_max: float, n: int = 400, cycles: int = 2) -> tuple[HysteresisLoop, int]:
        """Initial curve then ``cycles`` full cycles ``+H_max -> -H_max -> +H_max``; returns the loop and the
        index where the last cycle starts."""
        up = np.linspace(0.0, H_max, n // 2 + 1)
        cyc = np.concatenate([np.linspace(H_max, -H_max, n + 1)[1:], np.linspace(-H_max, H_max, n + 1)[1:]])
        H = np.concatenate([up] + [cyc] * cycles)
        return self.trace(H), len(up) - 1 + (cycles - 1) * len(cyc)
