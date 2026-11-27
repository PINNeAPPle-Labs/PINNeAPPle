"""Gas PVT for underground storage: z-factor, viscosity and formation volume factor.

Units are SI throughout: pressure in Pa, temperature in K, viscosity in Pa s, density in kg/m3, and the gas
formation volume factor ``Bg`` in m3 of reservoir volume per m3 at the standard conditions the class declares
(101325 Pa and 288.71 K by default -- the ``sm3`` of a European gas storage). Every rate in this package is a
volume *at those standard conditions* per second, which is the volume an operator reports.

Correlations
------------
* pseudo-critical properties from the gas specific gravity, Sutton (1985), SPE 14265::

      T_pc [R] = 120.1 + 425.9 sg,        p_pc [psia] = 677.0 + 15.0 sg - 37.5 sg^2

* z-factor, either :func:`papay_z` (explicit, Papay 1968) or :func:`hall_yarborough_z` (implicit in the
  reduced density, Hall & Yarborough 1973, Oil & Gas J., 82; valid over 1.1 <= T_pr <= 3.0);
* viscosity, :func:`lee_gonzalez_eakin_mu` (Lee, Gonzalez & Eakin 1966, JPT 18:997).

The simulator solves for the pressure itself, fully implicit, so no pseudo-pressure is needed: it linearises
the diffusivity equation for hand analysis, not for a Newton solve. :meth:`Gas.c_g` is the isothermal gas
compressibility ``1/p - (1/z) dz/dp``, used where a total compressibility has to be quoted.

The derivatives the Newton loop needs (:meth:`Gas.dindex_dp`, :meth:`Gas.dmu_dp`) are central differences of
the correlations on a relative step of ``h``; the correlations are tabulated fits, so an analytic derivative
would carry the fit's constants but not more accuracy than the difference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "R_GAS",
    "PSIA",
    "pseudo_critical_sutton",
    "papay_z",
    "hall_yarborough_z",
    "lee_gonzalez_eakin_mu",
    "Gas",
    "SlightlyCompressible",
    "Rock",
]

R_GAS = 8.31446261815324        # J/(mol K)
PSIA = 6894.757293168           # Pa per psi
T_SC = 288.7055555555556        # 60 degF, K
P_SC = 101_325.0                # Pa


def pseudo_critical_sutton(sg: float) -> tuple[float, float]:
    """Pseudo-critical pressure [Pa] and temperature [K] of a natural gas of specific gravity ``sg`` (Sutton 1985)."""
    if not 0.5 <= sg <= 1.2:
        raise ValueError(f"specific gravity {sg} is outside the range Sutton's fit covers (0.5 to 1.2)")
    tpc = (120.1 + 425.9 * sg) * 5.0 / 9.0
    ppc = (677.0 + 15.0 * sg - 37.5 * sg * sg) * PSIA
    return ppc, tpc


def papay_z(p, T: float, ppc: float, tpc: float):
    """Papay's explicit z-factor correlation (pseudoreduced pressure and temperature)."""
    ppr = np.asarray(p, dtype=float) / ppc
    tpr = T / tpc
    return 1.0 - 3.52 * ppr / 10.0 ** (0.9813 * tpr) + 0.274 * ppr * ppr / 10.0 ** (0.8157 * tpr)


def hall_yarborough_z(p, T: float, ppc: float, tpc: float, *, tol: float = 1e-12, max_iters: int = 100):
    """Hall & Yarborough's z-factor, solved for the reduced density ``Y`` by Newton.

    ``F(Y) = -A + (Y + Y^2 + Y^3 - Y^4)/(1-Y)^3 - (14.76 t - 9.76 t^2 + 4.58 t^3) Y^2
    + (90.7 t - 242.2 t^2 + 42.4 t^3) Y^(2.18 + 2.82 t)``, ``A = 0.06125 p_pr t exp(-1.2 (1-t)^2)``,
    ``t = 1/T_pr``, and ``z = A / Y``. Raises ``ValueError`` when Newton does not converge, rather than
    returning a plausible-looking number.
    """
    ppr = np.atleast_1d(np.asarray(p, dtype=float)) / ppc
    t = tpc / T
    a = 0.06125 * ppr * t * np.exp(-1.2 * (1.0 - t) ** 2)
    y = np.clip(a * 0.2, 1e-10, 0.9)
    c2 = 14.76 * t - 9.76 * t * t + 4.58 * t ** 3
    c3 = 90.7 * t - 242.2 * t * t + 42.4 * t ** 3
    power = 2.18 + 2.82 * t
    converged = np.zeros_like(y, dtype=bool)
    for _ in range(max_iters):
        om = 1.0 - y
        f = (-a + (y + y * y + y ** 3 - y ** 4) / om ** 3 - c2 * y * y + c3 * y ** power)
        df = ((1.0 + 4.0 * y + 4.0 * y * y - 4.0 * y ** 3 + y ** 4) / om ** 4
              - 2.0 * c2 * y + power * c3 * y ** (power - 1.0))
        step = f / df
        y = np.clip(y - step, 1e-12, 0.95)
        converged |= np.abs(step) <= tol * np.maximum(y, 1e-3)
        if converged.all():
            break
    if not converged.all():
        raise ValueError(f"Hall-Yarborough did not converge at p_pr = {np.round(ppr[~converged], 3)} "
                         f"(T_pr = {T / tpc:g}); the correlation covers 1.1 <= T_pr <= 3.0")
    z = a / y
    return z.reshape(np.shape(p))


def lee_gonzalez_eakin_mu(p, T: float, mw: float, z):
    """Gas viscosity [Pa s] by Lee, Gonzalez & Eakin (1966).

    ``mu = 1e-4 K exp(X rho^Y)`` cP with ``rho`` in g/cm3, ``T`` in R and ``M`` in g/mol::

        K = (9.4 + 0.02 M) T^1.5 / (209 + 19 M + T),   X = 3.5 + 986/T + 0.01 M,   Y = 2.4 - 0.2 X
    """
    t_r = T * 1.8
    m = mw * 1000.0
    rho_gcc = np.asarray(p, dtype=float) * mw / (z * R_GAS * T) / 1000.0
    k = (9.4 + 0.02 * m) * t_r ** 1.5 / (209.0 + 19.0 * m + t_r)
    x = 3.5 + 986.0 / t_r + 0.01 * m
    y = 2.4 - 0.2 * x
    return 1e-4 * k * np.exp(x * np.power(np.maximum(rho_gcc, 1e-12), y)) * 1e-3


@dataclass(frozen=True)
class Gas:
    """A dry natural gas described by its specific gravity and a z-correlation.

    ``T`` is the reservoir temperature, held constant (no thermal model). ``index`` is ``1/Bg``, the surface
    volume that one cubic metre of reservoir pore space holds, and it is what the accumulation term of the
    material balance integrates.
    """

    T: float                       # reservoir temperature [K]
    sg: float = 0.65               # specific gravity relative to air
    mw: float = 0.0174             # molar mass [kg/mol]
    p_sc: float = P_SC             # standard pressure [Pa]
    T_sc: float = T_SC             # standard temperature [K]
    z_model: str = "papay"         # "papay" | "hall-yarborough"
    h: float = 1e-6                # relative step of the numeric derivatives

    # ------------------------------------------------------------------ correlations
    @property
    def critical(self) -> tuple[float, float]:
        """Pseudo-critical pressure [Pa] and temperature [K]."""
        return pseudo_critical_sutton(self.sg)

    def z(self, p):
        """Gas deviation factor [-]."""
        ppc, tpc = self.critical
        if self.z_model == "papay":
            return papay_z(p, self.T, ppc, tpc)
        if self.z_model in ("hall-yarborough", "hall_yarborough"):
            return hall_yarborough_z(p, self.T, ppc, tpc)
        raise ValueError(f"unknown z model {self.z_model!r}; use 'papay' or 'hall-yarborough'")

    def mu(self, p):
        """Gas viscosity [Pa s]."""
        return lee_gonzalez_eakin_mu(p, self.T, self.mw, self.z(p))

    def Bg(self, p):
        """Formation volume factor [m3 reservoir per m3 at standard conditions]."""
        p = np.asarray(p, dtype=float)
        return (self.p_sc / self.T_sc) * self.z(p) * self.T / p

    def rho(self, p):
        """Density [kg/m3]."""
        p = np.asarray(p, dtype=float)
        return p * self.mw / (self.z(p) * R_GAS * self.T)

    def index(self, p):
        """``1/Bg``: surface m3 per reservoir m3, the density the material balance integrates."""
        return 1.0 / self.Bg(p)

    def dindex_dp(self, p):
        """Derivative of :meth:`index` with respect to the pressure [1/Pa]."""
        p = np.asarray(p, dtype=float)
        h = self.h * p
        return (self.index(p + h) - self.index(p - h)) / (2.0 * h)

    def dmu_dp(self, p):
        """Derivative of :meth:`mu` with respect to the pressure [Pa s / Pa]."""
        p = np.asarray(p, dtype=float)
        h = self.h * p
        return (self.mu(p + h) - self.mu(p - h)) / (2.0 * h)

    def c_g(self, p):
        """Isothermal gas compressibility ``1/p - (1/z) dz/dp`` [1/Pa]."""
        p = np.asarray(p, dtype=float)
        h = self.h * p
        dz = (self.z(p + h) - self.z(p - h)) / (2.0 * h)
        return 1.0 / p - dz / self.z(p)


@dataclass(frozen=True)
class SlightlyCompressible:
    """The slightly compressible limit: a fluid of constant density (``Bg = 1``) and constant viscosity.

    One reservoir cubic metre is then one surface cubic metre and all the storage comes from the rock, so the
    flow equation linearises to ``dphi/dt p = eta lap(p)`` with ``eta = k / (phi mu ct)``. This is the fluid
    the line-source (Theis) verification is written for, and it is the same limit in which a reservoir
    engineer quotes a diffusivity.
    """

    viscosity: float = 2.0e-5      # Pa s

    def z(self, p):
        return np.ones_like(np.asarray(p, dtype=float))

    def mu(self, p):
        return np.full_like(np.asarray(p, dtype=float), self.viscosity)

    def Bg(self, p):
        return np.ones_like(np.asarray(p, dtype=float))

    def rho(self, p):
        return np.ones_like(np.asarray(p, dtype=float))

    def index(self, p):
        return np.ones_like(np.asarray(p, dtype=float))

    def dindex_dp(self, p):
        return np.zeros_like(np.asarray(p, dtype=float))

    def dmu_dp(self, p):
        return np.zeros_like(np.asarray(p, dtype=float))

    def c_g(self, p):
        return np.zeros_like(np.asarray(p, dtype=float))


@dataclass(frozen=True)
class Rock:
    """Pore compressibility: ``phi(p) = phi0 (1 + cr (p - p_ref))``, clipped to ``phi_min``."""

    cr: float = 0.0                # [1/Pa]
    p_ref: float = 0.0             # [Pa]
    phi_min: float = 1e-4

    def porosity(self, phi0, p):
        p = np.asarray(p, dtype=float)
        return np.clip(np.asarray(phi0, dtype=float) * (1.0 + self.cr * (p - self.p_ref)), self.phi_min, 1.0)

    def dporosity_dp(self, phi0, p):
        p = np.asarray(p, dtype=float)
        phi = np.asarray(phi0, dtype=float) * (1.0 + self.cr * (p - self.p_ref))
        inside = (phi > self.phi_min) & (phi < 1.0)
        return np.where(inside, np.asarray(phi0, dtype=float) * self.cr, 0.0)
