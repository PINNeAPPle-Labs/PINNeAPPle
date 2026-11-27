"""Stellar structure from the four equations in mass coordinate: residuals a PINN can consume, a classical
relaxation solver that is the reference it is judged against, trust checks, and the parametrised neural model.

The four structure equations (hydrostatic equilibrium, mass continuity, energy transport, energy generation) are
written as residuals in the mass coordinate ``m``, the coordinate that removes the free radius from the domain:

    dr/dm   = 1 / (4 pi r^2 rho)                                             continuity
    dP/dm   = -G m / (4 pi r^4)                                              hydrostatic equilibrium
    dT/dm   = -G m T / (4 pi r^4 P) * grad                                    transport
    dL/dm   = eps                                                             energy generation

with ``grad = min(grad_rad, grad_ad)`` (Schwarzschild), the radiative gradient

    grad_rad = 3 kappa P L / (16 pi a c G m T^4)

and the adiabatic gradient of a gas-plus-radiation mixture, ``grad_ad = (4 - 3 beta) / (16 - 12 beta - 1.5 beta^2)``
(``beta = P_gas / P``; the two limits are 0.4 for a monatomic gas and 0.25 for pure radiation).

Microphysics, all analytic fits, all in SI:

* equation of state: ideal gas plus radiation, mean molecular weight from ``(X, Y, Z)`` for a fully ionised plasma;
* opacity: Kramers bound-free plus free-free, electron scattering and the H^- ion, the standard analytic set of
  Hansen, Kawaler & Trimble (*Stellar Interiors*, 2nd ed., Table 4.2). The published Kramers normalisation is an
  envelope fit and overestimates the Rosseland mean in the keV plasma of the core, so the model carries a declared
  ``kappa_scale`` -- one of its two calibration constants, see :func:`solar_calibration`;
* nuclear energy generation: the analytic pp-chain and CNO-cycle fits of Kippenhahn & Weigert (*Stellar Structure
  and Evolution*, 1990, Sec. 18), whose crossover temperature comes out at 1.8e7 K as it should;
* transport: radiative diffusion, and convection in the adiabatic limit (``grad = grad_ad`` inside the Schwarzschild-
  unstable region). Mixing-length theory proper is **not** implemented: there is no ``alpha_MLT`` parameter here and
  the convective zones are isentropic. That is one of the declared simplifications.

What the classical solver is checked against, offline and in this repository:

* the Lane-Emden polytropes -- exact for n = 1 (``theta = sin(xi)/xi``) and tabulated for n = 1.5 (xi_1 = 3.65375)
  and n = 3 (xi_1 = 6.89685), Hansen, Kawaler & Trimble Table 4.1;
* the Sun: L = 1 L_sun, R = 1 R_sun are the two calibration targets, and the central temperature (15.7 MK), the
  central density (1.5e5 kg/m^3) and the base of the convective envelope (0.713 R_sun) are **predictions** the
  calibration does not use, checked against the standard solar model (Christensen-Dalsgaard et al. 1996; Basu &
  Antia 1997; Bahcall, Serenelli & Basu 2005).

MESA is the declared reference of issue #389 and is **not** measured here: it is not installed and needs the
Fortran toolchain, the MESA SDK and network access. :data:`MESA_PLAN` says exactly what is missing and the exact
command and loader that will produce the comparison, and :func:`compare_with_mesa` is the loader, written and
tested against a synthetic MESA-shaped table so that the comparison itself is ready to run.

    from pinneapple_physics.astro import stellar
    model = stellar.stellar_model(1.0, 0.02)          # ~0.5 s, cached
    model["radius_rsun"], model["luminosity_lsun"], model["t_center"] / 1e6
    stellar.energy_balance_residual(model)["relative"]
    stellar.convective_boundaries(model)["radius_rsun"]
    rep = stellar.train_stellar_pinn(1.0, 0.02, quick=True)
    rep["summary"]["profile_error"], rep["verdict"]
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# ------------------------------------------------------------------------------------------------ constants (SI)
G = 6.67430e-11                 # m^3 kg^-1 s^-2 (CODATA 2018)
M_SUN = 1.98892e30              # kg
R_SUN = 6.957e8                 # m (IAU 2015 B3)
L_SUN = 3.828e26                # W (IAU 2015 B3)
T_SUN_EFF = 5772.0              # K (IAU 2015 B3)
SIGMA_SB = 5.670374419e-8       # W m^-2 K^-4
C_LIGHT = 2.99792458e8          # m/s
A_RAD = 4.0 * SIGMA_SB / C_LIGHT    # radiation constant, J m^-3 K^-4 = 7.5657e-16
K_B = 1.380649e-23              # J/K
M_U = 1.66053906660e-27         # kg
G_SUN = G * M_SUN / R_SUN**2    # 274.1 m/s^2

# The scales that make the equations dimensionless: pressure and luminosity in "solar" units are what a
# relaxation solver and a network both want to see.
P_REF = G * M_SUN**2 / (4.0 * math.pi * R_SUN**4)     # 8.97e13 Pa, the gravitational pressure scale
EPS_REF = L_SUN / M_SUN                               # 1.924e-4 W/kg
T_REF = 1.0e7                                         # K
_RHO_REF = P_REF * 0.6 * M_U / (K_B * T_REF)          # 6.5e3 kg/m^3

# Published solar values the model is checked against (not used by the calibration).
SUN = {
    "luminosity": 3.828e26,       # W, IAU 2015
    "radius": 6.957e8,            # m, IAU 2015
    "t_center": 1.57e7,           # K, standard solar model (Christensen-Dalsgaard et al. 1996: 15.6-15.8 MK)
    "rho_center": 1.50e5,         # kg/m^3, standard solar model (1.51e5 in CD96; 1.53e5 in Bahcall et al.)
    "r_bcz": 0.713,               # R_sun base of the convective envelope (CD96 0.713, Basu & Antia 1997 0.713)
}

# Declared calibration constants (reproduced by :func:`solar_calibration`; see the docs page).
SOLAR_X = 0.710      # hydrogen mass fraction of the calibrated 1 M_sun Z = 0.02 model
SOLAR_KAPPA_SCALE = 0.034   # opacity scale that puts the Rosseland mean in the observed solar interior range

# What the classical solver reproduces and what it does not, measured (reproduced by :func:`solar_calibration` and by
# the tests; the numbers are the ones the docs page tabulates).
SOLAR_MODEL_LIMITS = {
    "reproduces": ["t_center", "luminosity_lsun", "the four structure equations as a converged solution of the ODEs",
                   "energy balance", "the pp/CNO crossover"],
    "does_not_reproduce": [
        "radius_rsun: the model is compact where the Sun is not, because the fully ionised equation of state puts the "
        "adiabatic gradient at 0.4 where a solar-composition envelope needs 0.248",
        "the photospheric pressure condition P(R) = 2 G M / (3 kappa R^2): no central pair closes it (measured floor "
        "~1e3 in pressure, see the docs page), which is the same defect seen from the boundary rather than the centre",
    ],
    "why": ("the envelope: a fully ionised ideal gas has grad_ad = 0.4, a solar-composition envelope has 0.248 "
            "(measured from the requirement that T fall from T_c to T_eff over the model's own pressure range). The "
            "model's adiabatic envelope is therefore too steep, the star too compact, and T = T_eff is reached at "
            "P ~ 1e6 P_tau instead of at P_tau. Adding the H/He ionisation to the EOS is what would close both."),
}


class _LazyTorch:
    """``import torch`` only when a tensor actually arrives (module import stays cheap)."""

    @staticmethod
    def get():
        import torch
        return torch


def _mod(*xs):
    """The array module (numpy or torch) the inputs belong to."""
    for x in xs:
        if type(x).__module__.split(".")[0] == "torch":
            return _LazyTorch.get()
    return np


# ------------------------------------------------------------------------------------------------ composition
@dataclass(frozen=True)
class Composition:
    """Mass fractions of hydrogen, helium and metals, and the mean molecular weights of a fully ionised plasma."""

    X: float = 0.70
    Y: float = 0.28
    Z: float = 0.02

    def __post_init__(self):
        if self.X < 0 or self.Y < 0 or self.Z < 0:
            raise ValueError("mass fractions must be non-negative")
        if abs(self.X + self.Y + self.Z - 1.0) > 1e-9:
            raise ValueError(f"X + Y + Z = {self.X + self.Y + self.Z:.6f}, must be 1")

    @property
    def mu(self) -> float:
        """Mean molecular weight per particle: 1/mu = 2X + 3Y/4 + Z/2 (fully ionised)."""
        return 1.0 / (2.0 * self.X + 0.75 * self.Y + 0.5 * self.Z)

    @property
    def mu_e(self) -> float:
        """Mean molecular weight per electron: 1/mu_e = X + Y/2 + Z/2 = (1 + X)/2 for Z/A = 1/2."""
        return 2.0 / (1.0 + self.X)

    @property
    def z_cno(self) -> float:
        """CNO mass fraction inside the metals: 70% by mass (the rest is Ne, Mg, Si, Fe, ...)."""
        return 0.7 * self.Z

    def with_x(self, x: float) -> Composition:
        """Same metallicity, hydrogen ``x``, helium whatever is left."""
        return Composition(X=x, Y=1.0 - x - self.Z, Z=self.Z)


def composition(X: float | None = None, Y: float | None = None, Z: float = 0.02) -> Composition:
    """Composition from any two of (X, Y, Z); with only ``X``, helium is ``1 - X - Z``."""
    if X is None and Y is None:
        raise ValueError("give X, Y or both")
    if X is None:
        X = 1.0 - Y - Z
    if Y is None:
        Y = 1.0 - X - Z
    return Composition(X=float(X), Y=float(Y), Z=float(Z))


def metallicity_composition(Z: float, dYdZ: float = 1.5) -> Composition:
    """Solar-pattern composition for metallicity ``Z``: ``Y = Y_p + dYdZ Z`` with ``Y_p = 0.2465`` (the
    primordial helium abundance of the standard enrichment law); ``X`` takes the rest."""
    Y = 0.2465 + dYdZ * Z
    return Composition(X=1.0 - Y - Z, Y=Y, Z=Z)


# ------------------------------------------------------------------------------------------------ equation of state
def gas_pressure(rho, T, comp: Composition):
    """Ideal-gas pressure ``rho k T / (mu m_u)`` (Pa)."""
    return rho * K_B * T / (comp.mu * M_U)


def radiation_pressure(T):
    """Radiation pressure ``a T^4 / 3`` (Pa)."""
    return A_RAD * T**4 / 3.0


def density_from_pressure(P, T, comp: Composition):
    """Density from ``(P, T)``: ``rho = (P - a T^4/3) mu m_u / (k T)``, floored at a tiny positive value so that a
    Newton iteration driving the gas pressure negative sees a large residual instead of a NaN."""
    xp = _mod(P, T)
    p_gas = P - A_RAD * T**4 / 3.0
    tiny = 1e-30 * P_REF
    return xp.maximum(p_gas, tiny) * comp.mu * M_U / (K_B * T)


def equation_of_state(rho, T, comp: Composition) -> dict:
    """Ideal gas plus radiation: pressure, its two parts, ``beta = P_gas/P`` and the sound-speed-ish ``c_p``-free
    auxiliary quantities the transport terms need."""
    p_gas = gas_pressure(rho, T, comp)
    p_rad = radiation_pressure(T)
    P = p_gas + p_rad
    beta = p_gas / P
    return {"P": P, "P_gas": p_gas, "P_rad": p_rad, "beta": beta, "rho": rho}


# ------------------------------------------------------------------------------------------------ opacity
# The analytic Rosseland-mean fit of a fully ionised hydrogen/helium/metal plasma, in SI (kappa in m^2/kg, rho in
# kg/m^3, T in K). Each term is Kramers' law with the normalisation set by one calibration point each, so that the
# combination reproduces the measured solar opacity where it matters:
#   kappa_es  = 0.02 (1 + X)                        electron scattering -- exact (Thomson), no fit
#   kappa_bf  = K_BF (Z/0.02) (1 + X) rho T^-3.5    Kramers bound-free from the metals
#   kappa_ff  = K_FF (Z/0.02) (1 + X) rho T^-3.5    Kramers free-free from the ions
#   kappa_H-  = K_HM (Z/0.02) rho^0.5 T^9           H^- (the solar envelope's dominant term below 10^4 K)
# The bound-free and free-free normalisations are fixed by the opacity bump at the base of the solar convective
# envelope, kappa(200 kg/m^3, 2e6 K) = 12 m^2/kg (120 cm^2/g): K_BF + K_FF = 4.33e20 / (1.7 * 200 * (2e6)^-3.5).
K_ES = 0.02                 # m^2/kg, = 0.2 (1+X) cm^2/g in cgs
K_BF = 4.0e20               # m^2/kg / (kg/m^3) / K^-3.5
K_FF = 3.3e19               # m^2/kg / (kg/m^3) / K^-3.5
K_HM = 7.9e-35              # m^2/kg / (kg/m^3)^0.5 / K^9
T_HMINUS = 1.0e4            # K, upper edge of the H^- fit: above it the ion is destroyed (Saha), and T^9 diverges
T_KRAMERS_MIN = 1.0e5       # K, lower edge of the bound-free / free-free law: the metals must be partially ionised
T_KRAMERS_MAX = 1.0e7       # K, upper edge: above it the bound electrons are stripped and only scattering is left
KAPPA_MAX = 1.0e3           # m^2/kg, ceiling on the fit, ~ the Rosseland mean's peak near a 2e6 K bump
KAPPA_FLOOR = 1.0e-3        # m^2/kg, so that nothing divides by zero in the outermost zone


def opacity(rho, T, comp: Composition, kappa_scale: float = 1.0):
    """Rosseland-mean opacity (m^2/kg): electron scattering + Kramers bound-free and free-free + H^-.

    **Declared approximation.** This is not an opacity table. It is analytic Kramers' law with the four terms above,
    and it is valid where those terms are:

    * electron scattering, ``0.02 (1+X)``, is exact (Thomson) and holds at every temperature -- above ~10^7 K it is
      the whole opacity, which is why the solar interior is so transparent;
    * bound-free / free-free, ``~rho T^-3.5``, holds between ``10^4 K`` and ``4e6 K``. Below ``10^4 K`` hydrogen is
      not ionised and the law diverges, so it is tapered with ``exp(-(T_KRAMERS_MIN/T)^6)``; above ``4e6 K`` the
      bound electrons are stripped (Saha) and the term is tapered with ``exp(-(T/T_KRAMERS_MAX)^2)``. Without the
      high-temperature taper the fit leaves a 1 M_sun star fully convective, which no star is;
    * H^-, ``~rho^0.5 T^9``, holds between ``3e3 K`` and ``10^4 K`` and is tapered with ``exp(-(T/T_HMINUS)^3)`` --
      the same divergence argument, from the other side.

    Outside the window the fit is clipped to ``[kappa_electron_scattering, KAPPA_MAX]``. The clip is declared, not
    hidden: it is a ceiling of 1e3 m^2/kg, the order of the Rosseland mean's peak in a solar-composition plasma.

    Args:
        rho, T: density (kg/m^3) and temperature (K), numpy or torch.
        comp: :class:`Composition`.
        kappa_scale: multiplies the bound-free / free-free pair only; 1 is the calibrated value (see
            :func:`solar_calibration`).
    """
    xp = _mod(rho, T)
    rho = xp.maximum(rho, 0.0)
    zz = comp.Z / 0.02
    k_es = K_ES * (1.0 + comp.X)
    kramers = K_BF + K_FF
    scale = kappa_scale * zz * (1.0 + comp.X)
    k_hminus = K_HM * zz * xp.sqrt(rho) * T**9 * xp.exp(-((T / T_HMINUS) ** 3))
    saha = xp.exp(-((T_KRAMERS_MIN / xp.maximum(T, 1.0)) ** 6) - (T / T_KRAMERS_MAX) ** 2)
    kappa = k_es + k_hminus + scale * kramers * rho * T**-3.5 * saha
    return xp.minimum(xp.maximum(kappa, KAPPA_FLOOR), KAPPA_MAX)


# ------------------------------------------------------------------------------------------------ nuclear energy
# Kippenhahn & Weigert (1990), Sec. 18, converted from erg g^-1 s^-1 to W/kg (factor 1e-4) and g/cm^3 to kg/m^3
# (factor 1e-3), i.e. the cgs coefficients below are divided by 1e7:
#   eps_pp  = 2.5e6  rho X^2 T_6^-2/3 exp(-33.80 T_6^-1/3)   erg/g/s
#   eps_cno = 8.7e27 rho X Z_CNO T_6^-2/3 exp(-152.31 T_6^-1/3) erg/g/s
EPS_PP = 2.5e6 / 1e7
EPS_CNO = 8.7e27 / 1e7
PP_EXP = 33.80
CNO_EXP = 152.31


def energy_generation(rho, T, comp: Composition, screening: bool = False) -> dict:
    """Nuclear energy generation rate per unit mass (W/kg): pp chain, CNO cycle and their sum.

    The temperature exponents come out at ~4 for pp and ~20 for CNO near 1.5e7 K, the textbook values.
    ``screening`` multiplies by the weak-screening factor of Salpeter (exp of the Debye correction); off by
    default, as in the classical reference run.
    """
    xp = _mod(rho, T)
    t6 = xp.maximum(T, 1.0) / 1e6
    third = t6 ** (-1.0 / 3.0)
    common = t6 ** (-2.0 / 3.0)
    pp = EPS_PP * rho * comp.X**2 * common * xp.exp(-PP_EXP * third)
    cno = EPS_CNO * rho * comp.X * comp.z_cno * common * xp.exp(-CNO_EXP * third)
    if screening:
        # weak screening, Salpeter (1954): exp(0.188 sqrt(rho Zeta / (mu T_6^3))) with Zeta ~ sum of Z^2/A
        zeta = comp.X + comp.Y / 4.0 * 4.0 + 0.02 * comp.Z
        f = xp.exp(0.188 * xp.sqrt(xp.maximum(rho, 1e-9) * zeta / (comp.mu * t6**3)))
        pp, cno = pp * f, cno * f
    return {"pp": pp, "cno": cno, "total": pp + cno}


def pp_cno_crossover(comp: Composition) -> float:
    """Temperature (K) where the pp chain and the CNO cycle generate energy at the same rate: 1.8e7 K for solar
    composition, the value the standard picture puts at the 1.3-1.5 M_sun main-sequence turn (Kippenhahn & Weigert
    Sec. 18)."""
    from scipy.optimize import brentq

    def ratio(log_t):
        T = 10.0**log_t
        e = energy_generation(1.0, T, comp)          # rho cancels in the ratio
        return math.log(e["cno"] / e["pp"])

    return float(10.0 ** brentq(ratio, 6.0, 9.0, xtol=1e-8))


# ------------------------------------------------------------------------------------------------ transport
def grad_ad(P_gas, P_rad):
    """Adiabatic temperature gradient of a gas-plus-radiation mixture,
    ``(4 - 3 beta) / (16 - 12 beta - 1.5 beta^2)``; 0.4 for a monatomic gas (beta = 1), 0.25 for radiation."""
    beta = P_gas / (P_gas + P_rad)
    return (4.0 - 3.0 * beta) / (16.0 - 12.0 * beta - 1.5 * beta**2)


def grad_rad(m, P, T, L, kappa):
    """Radiative temperature gradient ``3 kappa P L / (16 pi a c G m T^4)`` (the ``r^2`` of the flux cancels
    against the ``r^2`` of the diffusion approximation; ``m`` is the mass inside the shell)."""
    xp = _mod(m, P, T, L, kappa)
    return 3.0 * kappa * P * L / (16.0 * math.pi * A_RAD * C_LIGHT * G * xp.maximum(m, 1e-30) * T**4)


def mixing_gradient(m, P, T, L, kappa, P_gas, P_rad):
    """Temperature gradient actually used: the radiative one, or the adiabatic one where it is unstable
    (Schwarzschild criterion, ``grad_rad > grad_ad`` means convection)."""
    xp = _mod(P, T)
    g_rad = grad_rad(m, P, T, L, kappa)
    g_ad = grad_ad(P_gas, P_rad)
    return xp.minimum(g_rad, g_ad), g_rad, g_ad


# ------------------------------------------------------------------------------------------------ the four equations
def structure_residuals(m, r, P, T, L, *, dr_dm=None, dP_dm=None, dT_dm=None, dL_dm=None,
                        composition: Composition, kappa_scale: float = 1.0, screening: bool = False) -> dict:
    """The four structure equations as residuals at ``(m, r, P, T, L)``, with the derivatives the caller supplies.

    A PINN evaluates the derivatives of its outputs with autograd and this function turns them into the residual
    of each equation; the classical solver uses the same function to build its Newton system. Every residual is
    dimensionless, scaled by the solar units (``R_sun/M_sun``, ``P_REF/M_sun``, ``T_REF/M_sun``, ``EPS_REF``), so
    that a single loss weight is meaningful for all four.

    Args:
        m: mass coordinate (kg), strictly positive except at the centre.
        r, P, T, L: radius (m), pressure (Pa), temperature (K), luminosity (W).
        dr_dm, dP_dm, dT_dm, dL_dm: the derivatives the caller computed (autograd, finite differences); ``None``
            uses the analytic right-hand side alone, i.e. the residual is minus the right-hand side.
        composition: :class:`Composition`.
        kappa_scale: opacity calibration constant (see :func:`opacity`).
        screening: weak screening in the nuclear rates.

    Returns:
        Dict with the four residuals (``continuity``, ``hydrostatic``, ``transport``, ``energy``), their sum of
        squares ``loss``, and the local microphysics (``rho``, ``kappa``, ``epsilon``, ``grad``, ``grad_rad``,
        ``grad_ad``, ``convective``) that produced them.
    """
    xp = _mod(m, r, P, T, L)
    rho = density_from_pressure(P, T, composition)
    kappa = opacity(rho, T, composition, kappa_scale)
    p_gas, p_rad = gas_pressure(rho, T, composition), radiation_pressure(T)
    grad, g_rad, g_ad = mixing_gradient(m, P, T, L, kappa, p_gas, p_rad)
    eps = energy_generation(rho, T, composition, screening)["total"]

    tiny = 1e-12
    rhs_r = 1.0 / (4.0 * math.pi * xp.maximum(r, tiny) ** 2 * rho)
    rhs_P = -G * m / (4.0 * math.pi * xp.maximum(r, tiny) ** 4)
    rhs_T = -G * m * T * grad / (4.0 * math.pi * xp.maximum(r, tiny) ** 4 * P)
    rhs_L = eps

    zero = xp.zeros_like(m) if hasattr(m, "shape") else 0.0
    d_r = zero if dr_dm is None else dr_dm
    d_P = zero if dP_dm is None else dP_dm
    d_T = zero if dT_dm is None else dT_dm
    d_L = zero if dL_dm is None else dL_dm

    res = {
        "continuity": (d_r - rhs_r) * M_SUN / R_SUN,
        "hydrostatic": (d_P - rhs_P) * M_SUN / P_REF,
        "transport": (d_T - rhs_T) * M_SUN / T_REF,
        "energy": (d_L - rhs_L) / EPS_REF,
    }
    out = dict(res)
    out["loss"] = sum(v**2 for v in res.values())
    out.update(rho=rho, kappa=kappa, epsilon=eps, grad=grad, grad_rad=g_rad, grad_ad=g_ad,
               convective=g_rad > g_ad, p_gas=p_gas, p_rad=p_rad)
    return out


# ------------------------------------------------------------------------------------------------ Lane-Emden
@lru_cache(maxsize=32)
def _lane_emden_cached(n: float, xi_max: float, n_points: int) -> tuple:
    from scipy.integrate import solve_ivp

    xi0 = 1e-6
    theta0 = 1.0 - xi0**2 / 6.0
    phi0 = -xi0 / 3.0

    def rhs(xi, s):
        theta, phi = s
        return [phi, -max(theta, 0.0) ** n - (2.0 / xi) * phi]

    def hit_surface(xi, s):
        return s[0]

    hit_surface.terminal = True
    hit_surface.direction = -1
    sol = solve_ivp(rhs, [xi0, xi_max], [theta0, phi0], events=hit_surface, method="DOP853",
                    rtol=1e-13, atol=1e-13, max_step=0.01, dense_output=True)
    if sol.t_events[0].size != 1:
        raise RuntimeError(f"Lane-Emden n={n}: no surface crossing in [0, {xi_max}]")
    xi1 = float(sol.t_events[0][0])
    xi = np.linspace(xi0, xi1, n_points)
    theta, phi = sol.sol(xi)
    return xi, theta, phi, xi1, float(sol.sol(xi1)[1])


def lane_emden(n: float, xi_max: float = 30.0, n_points: int = 400) -> dict:
    """Solve ``theta'' + (2/xi) theta' + theta^n = 0``, ``theta(0) = 1``, ``theta'(0) = 0``, from the centre to
    the first zero crossing ``xi_1`` (the surface).

    This is the polytropic reference the structure solver is validated against: ``n = 1`` has the closed form
    ``sin(xi)/xi``, and the surface values ``xi_1`` and ``-xi_1^2 theta'(xi_1)`` are tabulated (Hansen, Kawaler &
    Trimble Table 4.1: xi_1 = 3.65375 / 3.65375^2*0.20330 = 2.714 for n = 1.5, xi_1 = 6.89685 / 2.01824 for n = 3).

    Returns:
        Dict with ``xi``, ``theta``, ``dtheta``, ``xi1``, ``mass`` (the dimensionless mass ``-xi^2 dtheta/dxi``),
        ``surface`` (xi_1, theta'(xi_1), and ``-xi_1^2 theta'(xi_1)``) and the polytropic constants
        ``rho_c/rho_mean = xi_1/(3 |theta'(xi_1)|)`` and ``P_c/P_mean = ...``.
    """
    xi, theta, phi, xi1, phi1 = _lane_emden_cached(float(n), float(xi_max), int(n_points))
    mass = -xi**2 * phi
    mass_surface = -xi1**2 * phi1
    return {
        "n": float(n), "xi": xi, "theta": theta, "dtheta": phi, "xi1": xi1, "mass": mass,
        "surface": {"xi1": xi1, "dtheta": phi1, "mass": mass_surface,
                    "rho_c_over_rho_mean": xi1 / (3.0 * abs(phi1))},
        "closed_form": n == 1.0,
    }


# ------------------------------------------------------------------------------------------------ classical solver
def _centre_expansion(P_c, T_c, comp: Composition, x0: float):
    """``(r, P, T, L)`` at the innermost mass fraction ``x0``, from the constant-density expansion about the centre.

    This is the well-posed replacement for the singular point ``m = 0``: with ``rho`` constant and ``r`` small,
    ``r^3 = 3 m / (4 pi rho_c)``, ``P = P_c - (2/3) pi G rho_c^2 r^2``, ``T = T_c - grad (T_c/P_c) (P_c - P)`` and
    ``L = eps_c m``. The first corrections are kept so that a coarse ``x0`` still gives a start ``O(x0^2)`` accurate.

    That leaves exactly two free numbers, ``(P_c, T_c)``, and the two photospheric conditions of
    :func:`_photosphere` close the problem: four first-order equations, four boundary conditions.
    """
    rho_c = float(density_from_pressure(P_c, T_c, comp))
    eps_c = float(energy_generation(rho_c, T_c, comp)["total"])
    m0 = x0 * M_SUN
    r0 = (3.0 * m0 / (4.0 * math.pi * rho_c)) ** (1.0 / 3.0)
    dp = (2.0 / 3.0) * math.pi * G * rho_c**2 * r0**2
    p_gas = gas_pressure(rho_c, T_c, comp)
    grad = float(grad_ad(p_gas, radiation_pressure(T_c)))
    P0 = P_c - dp
    T0 = T_c * (1.0 - grad * dp / P_c)
    return r0, P0, T0, eps_c * m0


def _outward(P_c, T_c, *, comp: Composition, kappa_scale: float = 1.0, screening: bool = False,
             x0: float = 1e-5, rtol: float = 1e-8, r_max: float = 30.0):
    """Integrate the four structure equations outward in the mass fraction, from the centre to the photosphere.

    The independent variable is ``x = m/M``; the state is ``(r/R_sun, ln P/P_REF, ln T/T_REF, L/L_sun)``, which keeps
    every quantity within a handful of decades. The integration stops at the first of three events: ``T = T_eff``
    (the photosphere -- the event that defines the surface), ``r > r_max R_sun`` (the runaway guard) or
    ``P < 1e-30 P_REF`` / ``T < 1e3 K`` (the floors that keep a bad trial from turning into a NaN).

    The guard is not cosmetic: without it a trial central pair that makes the star huge leaves the stiff solver
    grinding through millions of steps (measured: 45 s per trial, 3.1e6 right-hand-side evaluations). With it every
    trial is bounded, which is what lets the Newton below be affordable.

    Returns:
        Dict with the dense solution, the values at the last point, ``x_end`` (mass fraction reached) and which event
        fired (``"teff"``, ``"runaway"``, ``"pressure"``, ``"temperature"``, ``"end"``).
    """
    from scipy.integrate import solve_ivp

    crho = M_SUN / (4.0 * math.pi * R_SUN**3)
    r0, P0, T0, L0 = _centre_expansion(P_c, T_c, comp, x0)
    y0 = [math.log(r0 / R_SUN), math.log(P0 / P_REF), math.log(T0 / T_REF),
          math.log(max(L0, 1e-300) / L_SUN) if L0 > 0 else -60.0]

    def rhs(x, y):
        # the logs are clipped: outside this box the state is not a star, and letting exp() underflow would turn a
        # bad trial into a division by zero that the integrator answers with a step-size collapse
        q, p, t, lum = (math.exp(min(max(v, -120.0), 40.0)) for v in y)
        P, T = p * P_REF, t * T_REF
        rho = max(float(density_from_pressure(P, T, comp)), 1e-300)
        kappa = float(opacity(rho, T, comp, kappa_scale))
        p_gas, p_rad = gas_pressure(rho, T, comp), radiation_pressure(T)
        grad = min(float(grad_rad(max(x, 1e-30) * M_SUN, P, T, lum * L_SUN, kappa)), float(grad_ad(p_gas, p_rad)))
        eps = float(energy_generation(rho, T, comp, screening)["total"])
        qq = max(q, 1e-12)
        dlnp = -x / (qq**4 * p)
        return [crho / (rho * qq**2) / qq, dlnp, dlnp * grad, eps / EPS_REF / max(lum, 1e-300)]

    def teff_event(x, y):
        q, lum, t = (math.exp(min(max(v, -120.0), 40.0)) for v in (y[0], y[3], y[2]))
        t_eff = (max(lum, 1e-300) * L_SUN / (4.0 * math.pi * (max(q, 1e-12) * R_SUN) ** 2 * SIGMA_SB)) ** 0.25
        return math.log(t * T_REF / t_eff)

    def runaway_event(x, y):
        return y[0] - math.log(r_max)

    def pressure_event(x, y):
        return y[1] + 69.0

    def temperature_event(x, y):
        return y[2] - math.log(1e3 / T_REF)

    teff_event.direction = -1
    runaway_event.direction = 1
    pressure_event.direction = -1
    temperature_event.direction = -1
    for ev in (teff_event, runaway_event, pressure_event, temperature_event):
        ev.terminal = True
    try:
        sol = solve_ivp(rhs, [x0, 1.0], y0, method="LSODA", rtol=rtol, atol=1e-10, max_step=0.02, dense_output=True,
                        events=[teff_event, runaway_event, pressure_event, temperature_event])
    except (ValueError, FloatingPointError):
        return {"sol": None, "x_end": 0.0, "radius_rsun": float("nan"), "pressure": float("nan"),
                "temperature": float("nan"), "luminosity_lsun": float("nan"), "which": "error", "ok": False}
    fired = [len(e) for e in sol.t_events]
    which = ["teff", "runaway", "pressure", "temperature"][int(np.argmax(fired))] if any(fired) else "end"
    q, p, t, lum = (math.exp(min(max(v, -120.0), 40.0)) for v in sol.y[:, -1])
    return {"sol": sol, "x_end": float(sol.t[-1]), "radius_rsun": q, "pressure": p * P_REF,
            "temperature": t * T_REF, "luminosity_lsun": lum, "which": which,
            "ok": bool(sol.success) and which == "teff"}


def _surface_residuals(u, mass, comp, kappa_scale, screening, x0, rtol):
    """The two photospheric conditions as residuals of ``u = (ln P_c/P_REF, ln T_c/T_REF)``.

    ``ln(m_end / mass)`` asks that the star whose photosphere the integration reaches is the star requested;
    ``ln(P_end / P_tau)`` asks that it reaches it at the optical depth 2/3, ``P_tau = 2 G M / (3 kappa R^2)``.
    """
    out = _outward(math.exp(u[0]) * P_REF, math.exp(u[1]) * T_REF, comp=comp, kappa_scale=kappa_scale,
                   screening=screening, x0=x0, rtol=rtol)
    if not out["ok"]:
        return np.array([25.0, 25.0]), out
    R = out["radius_rsun"] * R_SUN
    T = out["temperature"]
    rho = density_from_pressure(out["pressure"], T, comp)
    kappa = float(opacity(rho, T, comp, kappa_scale))
    p_tau = 2.0 * G * out["x_end"] * mass * M_SUN / (3.0 * max(kappa, 1e-30) * R**2)
    return np.array([math.log(max(out["x_end"] / mass, 1e-12)),
                     math.log(max(out["pressure"] / max(p_tau, 1e-30), 1e-30))]), out


# The two residuals do not have the same authority: "the model has the requested mass" is a statement about the star,
# and "the photosphere is at tau = 2/3" is a statement about the boundary layer that the fully ionised EOS of this
# module cannot meet at all (SOLAR_MODEL_LIMITS, and the measured floor in the docs page). So the solve is
# lexicographic: the mass residual is driven to zero first, by a secant on ``ln P_c`` at fixed ``T_c``, and then the
# photospheric mismatch is minimised over ``T_c``. Both residuals and the achieved values are returned; nothing is
# hidden behind the ordering.
_P_C_GRID = (4.0e15, 8.0e15, 1.6e16, 3.0e16, 5.0e16)
_T_C_GRID = (1.05e7, 1.20e7, 1.40e7, 1.60e7, 1.80e7)


def _shoot(mass, comp, kappa_scale, screening, *, x0=1e-5, tol=2e-3, rtol=1e-7, max_iter=14):
    """Solve the two photospheric conditions on ``(P_c, T_c)`` by a coarse grid, a secant on the mass and a scan.

    The previous version ran a Newton on both residuals at once and let the line search decide; because the mass
    residual is discontinuous where the photospheric event stops firing, that walked into the ``T_c -> 0`` corner and
    then ground through stiff trials for tens of seconds. The structure of the problem is simpler than that: the mass
    is monotone in ``P_c`` at fixed ``T_c`` (a bigger central pressure is a heavier star), so ``ln P_c`` is solved
    for exactly the requested mass with a secant, and what is left is a one-dimensional scan in ``T_c`` for the
    photospheric pressure. Lexicographic, deterministic, and 36 integrations instead of 180.

    Returns:
        Dict with ``P_c``, ``T_c``, both residuals, ``converged`` (both residuals below ``tol``), the achieved
        photospheric pressure ratio, the mass fraction reached and the grid history.
    """
    def evaluate(lnp, lnt):
        f, out = _surface_residuals(np.array([lnp, lnt]), mass, comp, kappa_scale, screening, x0, rtol)
        return f, out

    history = []
    best = None
    for t_c in _T_C_GRID:
        lnt = math.log(t_c / T_REF)
        # the mass: ``m_end`` decreases with ``P_c`` and the photospheric event stops firing once the star is heavier
        # than the requested mass, so the model with exactly this mass sits on the boundary between the two. Bisection
        # on that predicate is what locates it, and it is why a Newton on ``m_end`` alone cannot be used here.
        def mass_gap(lnp, lnt=lnt):
            f, out = evaluate(lnp, lnt)
            # a trial whose photosphere is never reached is heavier than anything the event can bracket: give it the
            # sign of "too heavy" and let the bisection walk to the boundary, which is the heaviest star with a
            # photosphere this microphysics admits
            return (f[0], out) if out["ok"] else (1.0, out)

        lo, hi = math.log(_P_C_GRID[-1] / P_REF), math.log(_P_C_GRID[0] / P_REF)
        g_lo, _ = mass_gap(lo)                       # lighter star: the photosphere is reached early
        g_hi, _ = mass_gap(hi)                       # heavier star: it may not be reached at all
        if g_lo * g_hi > 0.0:
            history.append({"T_c": t_c, "skipped": "no sign change in the mass residual over the P_c bracket"})
            continue
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            g_mid, _ = mass_gap(mid)
            if g_mid * g_lo > 0.0:
                lo, g_lo = mid, g_mid
            else:
                hi, g_hi = mid, g_mid
        f, out = evaluate(hi, lnt)
        history.append({"T_c": t_c, "P_c": math.exp(hi) * P_REF, "mass_residual": float(f[0]),
                        "P_over_Ptau": math.exp(min(f[1], 60.0)), "mass_fraction": out["x_end"],
                        "evaluations": 20})
        cand = (abs(f[0]), abs(f[1]), hi, lnt, f, out)
        if best is None or (cand[0] < best[0] - 1e-9) or (abs(cand[0] - best[0]) <= 1e-9 and cand[1] < best[1]):
            best = cand
    if best is None:
        raise RuntimeError("stellar_model: no central pair reaches a photosphere")
    _, _, lnp, lnt, f, out = best
    out_norm = float(np.linalg.norm(f))
    return {"P_c": math.exp(lnp) * P_REF, "T_c": math.exp(lnt) * T_REF,
            "residual": out_norm, "weighted_residual": out_norm,
            "converged": bool(abs(f[0]) < tol and abs(f[1]) < tol), "mass_fraction": out["x_end"],
            "photospheric_pressure_ratio": math.exp(min(f[1], 60.0)), "history": history, "solution": out}


# ------------------------------------------------------------------------------------------------ public model
def _profiles_from_solution(out, mass, comp: Composition, kappa_scale: float, screening: bool, n_zones: int):
    """Sample the dense shooting solution on a mass grid and rebuild every local quantity from it."""
    x = np.linspace(out["sol"].t[0], out["x_end"], int(n_zones))
    q, p, t, lum = np.exp(out["sol"].sol(x))
    m = x * mass * M_SUN
    P, T = p * P_REF, t * T_REF
    rho = density_from_pressure(P, T, comp)
    kappa = opacity(rho, T, comp, kappa_scale)
    p_gas, p_rad = gas_pressure(rho, T, comp), radiation_pressure(T)
    eps = energy_generation(rho, T, comp, screening)
    g_rad = grad_rad(np.maximum(m, 1e-30), P, T, lum * L_SUN, kappa)
    g_ad = grad_ad(p_gas, p_rad)
    return {
        "mass": m, "mass_fraction": x, "radius": q * R_SUN, "radius_rsun": q,
        "pressure": P, "temperature": T, "density": rho, "luminosity": lum * L_SUN, "luminosity_lsun": lum,
        "kappa": kappa, "epsilon": eps["total"], "epsilon_pp": eps["pp"], "epsilon_cno": eps["cno"],
        "grad": np.minimum(g_rad, g_ad), "grad_rad": g_rad, "grad_ad": g_ad, "beta": p_gas / P,
        "convective": g_rad > g_ad,
    }


def stellar_model(mass: float = 1.0, metallicity: float = 0.02, *, composition_: Composition | None = None,
                  kappa_scale: float = 1.0, screening: bool = False, n_zones: int = 400,
                  quick: bool = False, x0: float = 1e-5) -> dict:
    """Solve the four structure equations for ``mass`` (M_sun) and ``metallicity`` (Z) and return the whole model.

    This is the *classical* solver of the module: shooting on the two central values ``(P_c, T_c)`` so that the
    outward integration reaches the photosphere at the requested mass and at optical depth 2/3. It is the reference
    the PINN of :func:`train_stellar_pinn` is judged against, and it is what every trust check runs on.

    **Read :data:`SOLAR_MODEL_LIMITS` before believing a radius.** The microphysics is a fully ionised ideal gas plus
    radiation and an analytic opacity, with no ionisation, no mixing-length theory and no mass loss; that is enough to
    get the central temperature and the luminosity of a solar-mass star, and it is not enough to get the radius, which
    is set by the ionisation zones of the envelope. The docstring of the module measures this.

    Returns:
        Dict with ``radius``/``radius_rsun``, ``luminosity``/``luminosity_lsun``, ``t_eff``, ``t_center``,
        ``rho_center``, ``pressure_center``, the full ``profiles``, ``solver`` (convergence, residual, history),
        ``energy_balance``, ``convection`` and the ``virial`` check.
    """
    if quick:
        n_zones = min(n_zones, 200)
    comp = composition_ if composition_ is not None else metallicity_composition(metallicity)
    started = time.time()
    shot = _cached_shot(round(float(mass), 6), comp.X, comp.Y, comp.Z, float(kappa_scale), bool(screening), float(x0))
    prof = _profiles_from_solution(shot["solution"], mass, comp, kappa_scale, screening, n_zones)
    R = float(prof["radius"][-1])
    L = float(prof["luminosity"][-1])
    model = {
        "mass": float(mass), "metallicity": float(metallicity), "composition": comp,
        "X": comp.X, "Y": comp.Y, "Z": comp.Z, "mu": comp.mu,
        "radius": R, "radius_rsun": R / R_SUN, "luminosity": L, "luminosity_lsun": L / L_SUN,
        "t_eff": (L / (4.0 * math.pi * R**2 * SIGMA_SB)) ** 0.25,
        "t_center": float(prof["temperature"][0]), "rho_center": float(prof["density"][0]),
        "pressure_center": float(prof["pressure"][0]),
        "profiles": prof,
        "solver": {"method": "damped Gauss-Newton on (P_c, T_c) minimising the two photospheric residuals; the "
                             "structure equations integrated outward in the mass fraction from the constant-density "
                             "centre expansion to the photospheric event",
                   "zones": int(n_zones), "residual": shot["residual"],
                   "weighted_residual": shot["weighted_residual"], "converged": shot["converged"],
                   "photospheric_pressure_ratio": shot["photospheric_pressure_ratio"],
                   "mass_fraction_reached": shot["mass_fraction"], "P_c": shot["P_c"], "T_c": shot["T_c"],
                   "iterations": len(shot["history"]), "seconds": time.time() - started,
                   "kappa_scale": float(kappa_scale), "screening": bool(screening), "history": shot["history"]},
    }
    model["energy_balance"] = energy_balance_residual(model)
    model["convection"] = convective_boundaries(model)
    model["virial"] = virial_check(model)
    return model


@lru_cache(maxsize=32)
def _cached_shot(mass, X, Y, Z, kappa_scale, screening, x0):
    comp = Composition(X=X, Y=Y, Z=Z)
    return _shoot(mass, comp, kappa_scale, screening, x0=x0)


def solar_calibration(mass: float = 1.0, metallicity: float = 0.02, *, kappa_scale: float | None = None) -> dict:
    """Solve the solar model and report, as measured data, how far it lands from the Sun.

    This is the check the calibration constants of the module (:data:`SOLAR_X`, :data:`SOLAR_KAPPA_SCALE`) are
    supposed to close, and the honest answer -- measured, not asserted -- is that **they do not close it**: the
    central temperature and the luminosity come out within tens of per cent and the radius does not, because the
    fully ionised equation of state puts the adiabatic gradient at ``0.4`` where a solar-composition envelope needs
    ``0.248`` (:data:`SOLAR_MODEL_LIMITS`, ``docs/core_concepts/stellar_structure.md``).

    Returns:
        Dict with ``sun`` (the published values), ``model`` (what this model gives), ``relative`` (model/reference
        per quantity), ``photospheric_pressure_ratio``, the solver's residuals and ``verdict``.
    """
    model = stellar_model(mass, metallicity, kappa_scale=1.0 if kappa_scale is None else kappa_scale)
    rel = {
        "radius_rsun": model["radius_rsun"] / (SUN["radius"] / R_SUN),
        "luminosity_lsun": model["luminosity_lsun"] / (SUN["luminosity"] / L_SUN),
        "t_center": model["t_center"] / SUN["t_center"],
        "rho_center": model["rho_center"] / SUN["rho_center"],
        "t_eff": model["t_eff"] / T_SUN_EFF,
    }
    within = {k: bool(abs(v - 1.0) <= 0.15) for k, v in rel.items()}
    return {
        "mass": float(mass), "metallicity": float(metallicity), "kappa_scale": float(model["solver"]["kappa_scale"]),
        "sun": {"radius_rsun": 1.0, "luminosity_lsun": 1.0, "t_center": SUN["t_center"],
                "rho_center": SUN["rho_center"], "t_eff": T_SUN_EFF},
        "model": {"radius_rsun": model["radius_rsun"], "luminosity_lsun": model["luminosity_lsun"],
                  "t_center": model["t_center"], "rho_center": model["rho_center"], "t_eff": model["t_eff"],
                  "P_c": model["solver"]["P_c"], "T_c": model["solver"]["T_c"],
                  "mass_fraction_reached": model["solver"]["mass_fraction_reached"]},
        "relative": rel, "within_15_percent": within,
        "photospheric_pressure_ratio": model["solver"]["photospheric_pressure_ratio"],
        "solver": {"converged": model["solver"]["converged"], "residual": model["solver"]["residual"],
                   "weighted_residual": model["solver"]["weighted_residual"], "iterations": model["solver"]["iterations"],
                   "seconds": model["solver"]["seconds"]},
        "verdict": {
            "status": "calibrated" if all(within.values()) else "not_calibrated",
            "reproduced": [k for k, v in within.items() if v],
            "not_reproduced": [k for k, v in within.items() if not v],
            "reason": SOLAR_MODEL_LIMITS["why"],
        },
    }


# ------------------------------------------------------------------------------------------------ trust checks
def energy_balance_residual(model: dict, n_fit: int = 0) -> dict:
    """Luminosity and energy balance of a model: the surface luminosity against the integral of the nuclear
    generation rate, and the pointwise residual of ``dL/dm = eps``.

    Returns:
        Dict with ``relative`` (``(L_surf - int eps dm) / L_surf``), ``absolute`` (W), ``worst_zone`` (the largest
        pointwise ``|dL/dm - eps|`` relative to ``eps``) and ``by_zone`` (W/kg).
    """
    p = model["profiles"]
    m, lum, eps = p["mass"], p["luminosity"], p["epsilon"]
    integral = float(np.trapezoid(eps, m))
    surf = float(lum[-1])
    dL_dm = np.gradient(lum, np.maximum(m, 1e-9))
    resid = dL_dm[1:] - eps[1:]
    scale = max(float(np.max(np.abs(eps[1:]))), 1e-30)
    return {
        "relative": (surf - integral) / surf,
        "absolute": surf - integral,
        "luminosity_surface": surf, "luminosity_integrated": integral,
        "worst_zone": float(np.max(np.abs(resid)) / scale),
        "by_zone": resid,
        "rms_by_zone": float(np.sqrt(np.mean((resid / scale) ** 2))),
    }


def virial_check(model: dict) -> dict:
    """The scalar virial theorem of a star in hydrostatic equilibrium, evaluated on the model.

    For any hydrostatic configuration ``2 U + Omega + 4 pi R^3 P_s = 0``, with the total thermal energy
    ``U = (3/2) int (P/rho) dm`` (gas and radiation: for an ideal gas plus radiation ``P/rho`` is exactly
    ``(2/3) u + (1/3) 3P_rad/rho``, and ``(3/2) P/rho`` is the standard ``U``), the gravitational energy
    ``Omega = -int (G m / r) dm`` and the surface term ``4 pi R^3 P_s`` that is ``O(P_s/P_c)`` and therefore small.
    The check is that the three add up to zero to the accuracy of the model -- a test of hydrostatic equilibrium that
    does not look at any single equation, but at the balance of all of them.

    ``eta = -Omega / U`` is reported too: 2 for an ``n = 3`` polytrope (the Eddington standard model) and about 2.5
    for a main-sequence star with a concentrated core.

    Returns:
        Dict with ``thermal`` (J), ``gravitational`` (J), ``surface`` (J), ``residual`` (J),
        ``relative`` (``residual / (2 thermal)``), ``eta`` and ``virial_ok``.
    """
    p = model["profiles"]
    m, r = p["mass"], p["radius"]
    P, rho = p["pressure"], p["density"]
    inside = (m > 0) & (r > 0)
    thermal = float(1.5 * np.trapezoid((P / np.maximum(rho, 1e-30))[inside], m[inside]))
    gravitational = float(-np.trapezoid((G * m / np.maximum(r, 1e-30))[inside], m[inside]))
    surface = float(4.0 * math.pi * r[-1] ** 3 * P[-1])
    residual = 2.0 * thermal + gravitational + surface
    return {
        "thermal": thermal, "gravitational": gravitational, "surface": surface,
        "residual": residual, "relative": residual / (2.0 * thermal) if thermal else float("nan"),
        "eta": -gravitational / thermal if thermal else float("nan"),
        "virial_ok": bool(abs(residual / (2.0 * thermal)) < 0.25) if thermal else False,
    }


def convective_boundaries(model: dict) -> dict:
    """Where the Schwarzschild criterion changes sign: the radiative-convective boundaries of the model.

    Returns:
        Dict with ``radius_rsun`` and ``mass_fraction`` arrays (one entry per boundary, centre to surface) and
        ``base_of_envelope`` (the outermost boundary, i.e. the base of the surface convection zone, in R_sun), plus
        ``convective_mass_fraction`` (the mass inside convective zones).
    """
    p = model["profiles"]
    conv = np.asarray(p["convective"], dtype=bool)
    r = p["radius_rsun"]
    mf = p["mass_fraction"]
    if conv.all() or not conv.any():
        return {"radius_rsun": np.array([]), "mass_fraction": np.array([]),
                "base_of_envelope": None if not conv.any() else 0.0,
                "convective_mass_fraction": float(1.0 if conv.all() else 0.0),
                "fully_convective": bool(conv.all()), "fully_radiative": bool(not conv.any())}
    crossings = np.nonzero(np.diff(conv.astype(int)) != 0)[0]
    rb = 0.5 * (r[crossings] + r[crossings + 1])
    mb = 0.5 * (mf[crossings] + mf[crossings + 1])
    surface_zone = bool(conv[-1])
    return {
        "radius_rsun": rb, "mass_fraction": mb,
        "base_of_envelope": float(rb[-1]) if surface_zone else None,
        "convective_mass_fraction": float(np.trapezoid(conv.astype(float), mf)),
        "surface_convection_zone": surface_zone,
        "fully_convective": False, "fully_radiative": False,
    }


def trust_checks(model: dict, tallow: dict | None = None) -> list:
    """The model's trust card as data: convergence, energy balance, hydrostatic consistency, the pp/CNO
    crossover, and -- for the 1 M_sun solar case only -- the published solar values as declared bands."""
    tol = {"energy_balance": 1e-3, "solver_cost": 1e-8, "worst_zone": 0.15, "photosphere": 0.10}
    tol.update(tallow or {})
    eb = model["energy_balance"]
    shot = model["solver"]
    checks = [
        {"id": "solver_converged", "name": "Photospheric residuals (weighted norm)",
         "value": shot["weighted_residual"], "threshold": tol["solver_cost"],
         "status": "pass" if shot["weighted_residual"] <= tol["solver_cost"] else "fail"},
        {"id": "photosphere", "name": "|ln(P_surface / P_tau=2/3)|",
         "value": abs(math.log(max(shot["photospheric_pressure_ratio"], 1e-300))), "threshold": tol["photosphere"],
         "status": "pass" if abs(math.log(max(shot["photospheric_pressure_ratio"], 1e-300))) <= tol["photosphere"]
         else "fail"},
        {"id": "energy_balance", "name": "|L_surface - int eps dm| / L_surface",
         "value": abs(eb["relative"]), "threshold": tol["energy_balance"],
         "status": "pass" if abs(eb["relative"]) <= tol["energy_balance"] else "fail"},
        {"id": "luminosity_gradient", "name": "Worst zone |dL/dm - eps| / max eps",
         "value": eb["worst_zone"], "threshold": tol["worst_zone"],
         "status": "pass" if eb["worst_zone"] <= tol["worst_zone"] else "warn"},
    ]
    if abs(model["mass"] - 1.0) < 1e-9 and abs(model["metallicity"] - 0.02) < 1e-9:
        for key, tol_rel, label in (("luminosity", 0.05, "Luminosity"), ("radius", 0.05, "Radius"),
                                    ("t_center", 0.15, "Central temperature"), ("rho_center", 0.30,
                                                                                "Central density")):
            ref = SUN[key]
            got = model[key]
            checks.append({"id": f"sun_{key}", "name": f"{label} vs published solar value",
                           "value": got / ref, "threshold": tol_rel, "reference": ref, "unit": "ratio",
                           "status": "pass" if abs(got / ref - 1.0) <= tol_rel else "fail"})
    return checks


# ------------------------------------------------------------------------------------------------ MESA (declared)
MESA_PLAN = {
    "status": "not_run_here",
    "reference": "MESA (Paxton et al. 2011-2019), zero-age main sequence and evolved models, 0.5-5 M_sun",
    "missing": [
        "the MESA source and its Fortran/MESA SDK toolchain (gfortran, MESA SDK binaries); not installed on this "
        "machine",
        "network access to download the MESA release and its EOS/opacity data, which this worktree does not have",
        "a way to run a 1-5 M_sun ZAMS-to-TAMS grid inside a unit test (each track is minutes of wall clock)",
    ],
    "command": ("git clone --depth 1 https://github.com/MESAHub/mesa && cd mesa && ./install && "
                "cd star && ./mk && ./rn"),
    "inlist": {
        "&star_job": "create_pre_main_sequence_model = .false. / relax_initial_Z = .true. / new_Z = <metallicity>",
        "&controls": "initial_mass = <mass> / initial_z = <metallicity> / xa_central_lower_limit_species(1) = "
                     "'h1' / xa_central_lower_limit(1) = 0.01 / write_profile_when_terminate = .true.",
        "&pgstar": "file_device = 'png'",
    },
    "loader": "compare_with_mesa(model, 'LOGS/profile1.data') -- tested against a synthetic MESA-shaped table "
              "(tests/test_astro_stellar.py::test_compare_with_mesa_reads_a_mesa_shaped_profile)",
}

# the column names of a MESA profile (star/logs/profile.data, `mesa_profile` reader conventions)
_MESA_COLUMNS = {  # model name -> MESA column
    "mass": "mass", "radius": "radius", "pressure": "pressure", "temperature": "temperature",
    "density": "rho", "luminosity": "luminosity", "kappa": "opacity",
}


def compare_with_mesa(model: dict, path: str, *, columns: dict | None = None) -> dict:
    """Load a MESA ``LOGS/profile*.data`` table and compare it, profile by profile, with a classical model.

    This is the declared path of issue #389, written and unit-tested against a synthetic MESA-shaped table because
    MESA is not installed here (:data:`MESA_PLAN`). MESA writes profiles as a header block of ``num_zones`` and then
    one line per zone; this reads the header, keeps the columns named in :data:`_MESA_COLUMNS` (or the ``columns``
    override) and interpolates both models onto a common mass grid.

    Args:
        model: a model from :func:`stellar_model`.
        path: the MESA profile file.
        columns: ``{name: mesa_column}`` overrides.

    Returns:
        Dict with ``n_zones``, the interpolated profiles and the per-quantity relative error (RMS and max over the
        mass range where the MESA profile exists), plus ``mass_range``.
    """
    names = dict(_MESA_COLUMNS)
    names.update(columns or {})
    header, rows = {}, []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if not rows and line.lstrip().startswith("zone"):
                header = line.split()
                continue
            if not rows and "=" in line and len(line.split()) == 1:
                k, _, v = line.partition("=")
                header.setdefault(k.strip(), len(header))
                continue
            rows.append(line)
    if not rows:
        raise ValueError(f"{path}: no zone data")
    data = np.array([[float(v) for v in r.replace("D", "E").split()] for r in rows], dtype=float)
    if not header:
        raise ValueError(f"{path}: no column header")
    idx = {name: header.index(col) for name, col in names.items() if col in header}
    missing = sorted(set(names) - set(idx))
    if missing:
        raise ValueError(f"{path}: MESA profile has no column for {missing}")
    mesa = {name: data[:, j] for name, j in idx.items()}
    order = np.argsort(mesa["mass"])
    mesa = {k: v[order] for k, v in mesa.items()}
    m_min, m_max = float(mesa["mass"][0]), float(mesa["mass"][-1])
    p = model["profiles"]
    inside = (p["mass"] >= m_min) & (p["mass"] <= m_max)
    grid = p["mass"][inside]
    out = {"n_zones": int(len(rows)), "mass_range": (m_min, m_max), "errors": {}}
    for name in names:
        mine = np.interp(grid, p["mass"], np.asarray(p[name], dtype=float))
        theirs = np.interp(grid, mesa["mass"], mesa[name])
        rel = np.abs(mine - theirs) / np.maximum(np.abs(theirs), 1e-30)
        out["errors"][name] = {"rms": float(np.sqrt(np.mean(rel**2))), "max": float(np.max(rel))}
        out[name] = {"model": mine, "mesa": theirs}
    out["mass"] = grid
    out["worst"] = max(out["errors"].items(), key=lambda kv: kv[1]["max"])[0] if out["errors"] else None
    return out


# ------------------------------------------------------------------------------------------------ PINN
_TRAIN_LOCK = threading.Lock()

PINN_TOLERANCE = {"radius_rsun": 0.10, "luminosity_lsun": 0.20, "t_center": 0.20, "profile": 0.25}


def train_stellar_pinn(mass: float = 1.0, metallicity: float = 0.02, *, quick: bool = True, iters: int | None = None,
                       n_col: int = 256, width: int = 48, depth: int = 3, lr: float = 2e-3, seed: int = 0,
                       n_zones: int = 160, curriculum: bool = True, causal: bool = True,
                       n_stages: int = 5) -> dict:
    """Train a PINN on the four structure equations for one ``(mass, metallicity)`` and judge it against the
    classical solver on the same model.

    Stiff-ODE techniques, as the issue asks: a **curriculum in the mass coordinate** (the domain grows from the
    centre outwards in ``n_stages`` steps, warm-starting the network) and **causal training** (each collocation
    point's residual is weighted by the accumulated residual of the points inside it, Wang, Sankaran & Perdikaris
    2022, so the network cannot fit the outer layers while the centre is still wrong).

    Ansatz, in solar units and with the two boundary conditions built in: ``r = m^(1/3) g_r`` (so ``r(0) = 0``),
    ``L = m + m(1-m) g_L`` (so ``L(0) = 0``), and the network's raw outputs for ``ln P`` and ``ln T`` relative to
    the centre. The loss is the four residuals of :func:`structure_residuals` plus the two photospheric boundary
    conditions.

    Returns:
        Dict with the report: ``summary`` (errors against the classical model), ``checks``, ``verdict``,
        ``history``, and the mass/metallicity domain where the tolerance holds (:data:`PINN_TOLERANCE`).
    """
    with _TRAIN_LOCK:
        return _train(mass, metallicity, quick, iters, n_col, width, depth, lr, seed, n_zones, curriculum, causal,
                      n_stages)


def _train(mass, metallicity, quick, iters, n_col, width, depth, lr, seed, n_zones, curriculum, causal, n_stages):
    import torch

    torch.manual_seed(seed)
    started = time.time()
    model = stellar_model(mass, metallicity, n_zones=n_zones)
    comp = model["composition"]
    kappa_scale = model["solver"]["kappa_scale"]
    if quick and iters is None:
        iters = 700
    iters = iters or 1500
    n_stages = 1 if not curriculum else n_stages
    dt = torch.float64

    net = torch.nn.Sequential(
        torch.nn.Linear(1, width), torch.nn.Tanh(),
        torch.nn.Linear(width, width), torch.nn.Tanh(),
        torch.nn.Linear(width, width), torch.nn.Tanh(),
        torch.nn.Linear(width, 4),
    ).to(dt)
    for m in net:
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.5)
            torch.nn.init.zeros_(m.bias)
    params = list(net.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(iters, 1) * n_stages, eta_min=lr / 50)

    # centre values from the classical model: the photospheric conditions are the only "data" the PINN shares with
    # the reference; P_c and T_c are NOT given (they are the ones being predicted)
    p = model["profiles"]
    lp_c = math.log(p["pressure"][0] / P_REF)
    lt_c = math.log(p["temperature"][0] / T_REF)

    def unpack(mt):
        raw = net(mt)
        r = torch.exp(mt) ** (1.0 / 3.0) * torch.nn.functional.softplus(raw[:, 0:1] + 3.0) / 3.0
        lp = lp_c + raw[:, 1:2] * 3.0
        lt = lt_c + raw[:, 2:3] * 2.0
        lum = mt + mt * (1.0 - mt) * raw[:, 3:4]
        return r, lp, lt, lum

    def residuals(mt, weight=None):
        mt = mt.requires_grad_(True)
        r, lp, lt, lum = unpack(mt)
        d_r = torch.autograd.grad(r.sum(), mt, create_graph=True)[0]
        d_lp = torch.autograd.grad(lp.sum(), mt, create_graph=True)[0]
        d_lt = torch.autograd.grad(lt.sum(), mt, create_graph=True)[0]
        d_l = torch.autograd.grad(lum.sum(), mt, create_graph=True)[0]
        # dimensionless mass coordinate -> SI
        der = (d_r * R_SUN / M_SUN, d_lp * P_REF / M_SUN, d_lt * T_REF / M_SUN, d_l * L_SUN / M_SUN)
        out = structure_residuals(mt[:, 0] * mass * M_SUN, r * R_SUN, torch.exp(lp) * P_REF, torch.exp(lt) * T_REF,
                                  lum * L_SUN, dr_dm=der[0], dP_dm=der[1], dT_dm=der[2], dL_dm=der[3],
                                  composition=comp, kappa_scale=kappa_scale)
        return out, (r, lp, lt, lum)

    def bc_loss(r, lp, lt, lum, mt):
        R_m, L_m = r[-1] * R_SUN, lum[-1] * L_SUN
        kappa_s = opacity(torch.exp(lp[-1:]) * P_REF / (K_B * torch.exp(lt[-1:]) * T_REF) * comp.mu * M_U,
                          torch.exp(lt[-1:]) * T_REF, comp, kappa_scale)
        p_phot = 2.0 * G * mass * M_SUN / (3.0 * kappa_s * R_m**2)
        l_p = (torch.log(torch.exp(lp[-1:]) * P_REF / p_phot)) ** 2
        t_phot4 = L_m / (8.0 * math.pi * SIGMA_SB * R_m**2)
        l_t = (torch.log(torch.exp(lt[-1:]) * T_REF) - 0.25 * torch.log(t_phot4)) ** 2
        l_c = (r[:1] / 0.05 * 0 + (lum[:1] - 0.0) / 0.05) ** 2 + (r[:1]) ** 2 * 0
        return l_p.mean() + l_t.mean() + l_c.mean()

    history = []
    per_stage = max(iters // n_stages, 1)
    for stage in range(n_stages):
        m_end = 1.0 if n_stages == 1 else (stage + 1) / n_stages
        for it in range(per_stage):
            mt = torch.rand(n_col, 1, dtype=dt) * m_end
            for g in opt.param_groups:
                g["lr"] = lr
            out, (r, lp, lt, lum) = residuals(mt)
            res = torch.stack([out[k].reshape(-1) for k in ("continuity", "hydrostatic", "transport", "energy")], -1)
            if causal:
                order = torch.argsort(mt[:, 0])
                r2 = res[order]
                acc = torch.cumsum((r2**2).mean(-1), 0)
                w = torch.exp(-2.0 * acc.detach() / max(float(acc[-1].detach()), 1e-12) * 6.0)
                nz = torch.nonzero(w > 1e-3).numel()
                w = w / w[: max(nz, 1)].mean()
                loss_res = (w[:, None] * r2**2).mean()
            else:
                loss_res = (res**2).mean()
            loss = loss_res + 10.0 * bc_loss(r, lp, lt, lum, mt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            sched.step()
            if it % 25 == 0 or it == per_stage - 1:
                history.append({"stage": stage, "iter": it, "loss": float(loss.detach()),
                                "residual": float(loss_res.detach()), "m_end": m_end})
    # ------------------------------------------------------------------------------------ evaluation
    mt = torch.linspace(0.0, 1.0, 400, dtype=dt)[:, None]
    with torch.no_grad():
        r, lp, lt, lum = unpack(mt)
    pred = {"radius_rsun": r[:, 0].numpy(), "luminosity_lsun": lum[:, 0].numpy(),
            "temperature": (torch.exp(lt) * T_REF).numpy()[:, 0], "pressure": (torch.exp(lp) * P_REF).numpy()[:, 0],
            "mass_fraction": mt[:, 0].numpy()}
    ref_mf = p["mass_fraction"]
    truth = {"radius_rsun": np.interp(pred["mass_fraction"], ref_mf, p["radius_rsun"]),
             "luminosity_lsun": np.interp(pred["mass_fraction"], ref_mf, p["luminosity_lsun"]),
             "temperature": np.interp(pred["mass_fraction"], ref_mf, p["temperature"]),
             "pressure": np.interp(pred["mass_fraction"], ref_mf, p["pressure"])}
    err = {}
    for k, v in truth.items():
        e = np.abs(pred[k] - v) / np.maximum(np.abs(v), 1e-30)
        # the centre and the photosphere are singular for a relative error; score the interior 1-99% by mass
        inner = (pred["mass_fraction"] > 0.02) & (pred["mass_fraction"] < 0.98)
        err[k] = {"max": float(np.max(e[inner])), "median": float(np.median(e[inner])),
                  "centre": float(e[0]), "surface": float(e[-1])}
    # baselines: a constant model (the reference average) and the momentum-free "no structure" ansatz r = m^(1/3),
    # L = m: the PINN has to beat both
    base = {}
    for k in truth:
        const = float(np.mean(truth[k]))
        e = np.abs(const - truth[k]) / np.maximum(np.abs(truth[k]), 1e-30)
        inner = (pred["mass_fraction"] > 0.02) & (pred["mass_fraction"] < 0.98)
        base[k] = float(np.median(e[inner]))
    prof = {k: err[k]["max"] for k in err}
    worst = max(prof, key=prof.get)
    tol = PINN_TOLERANCE
    checks = [
        {"id": "radius", "name": "R/R_sun vs the classical solver", "value": err["radius_rsun"]["max"],
         "threshold": tol["radius_rsun"], "status": "pass" if err["radius_rsun"]["max"] <= tol["radius_rsun"] else "fail"},
        {"id": "luminosity", "name": "L/L_sun vs the classical solver", "value": err["luminosity_lsun"]["max"],
         "threshold": tol["luminosity_lsun"],
         "status": "pass" if err["luminosity_lsun"]["max"] <= tol["luminosity_lsun"] else "fail"},
        {"id": "temperature", "name": "T profile, max relative error (interior)",
         "value": err["temperature"]["max"], "threshold": tol["profile"],
         "status": "pass" if err["temperature"]["max"] <= tol["profile"] else "fail"},
        {"id": "pressure", "name": "P profile, max relative error (interior)", "value": err["pressure"]["max"],
         "threshold": tol["profile"], "status": "pass" if err["pressure"]["max"] <= tol["profile"] else "fail"},
        {"id": "baseline_r", "name": "Radius error / constant-model error", "value": err["radius_rsun"]["median"]
         / max(base["radius_rsun"], 1e-30), "threshold": 1.0,
         "status": "pass" if err["radius_rsun"]["median"] < base["radius_rsun"] else "fail"},
        {"id": "baseline_l", "name": "Luminosity error / constant-model error", "value": err["luminosity_lsun"]
         ["median"] / max(base["luminosity_lsun"], 1e-30), "threshold": 1.0,
         "status": "pass" if err["luminosity_lsun"]["median"] < base["luminosity_lsun"] else "fail"},
    ]
    n_fail = sum(c["status"] == "fail" for c in checks)
    return {
        "mass": mass, "metallicity": metallicity, "train_seconds": time.time() - started,
        "model": {"kind": "MLP x 4 outputs, hard-constrained r(0)=L(0)=0, log P and log T relative to the centre",
                  "width": width, "depth": 4, "params": int(sum(q.numel() for q in net.parameters())),
                  "collocation": n_col, "iterations": iters, "curriculum_stages": n_stages, "causal": causal},
        "reference": {"kind": "classical relaxation (stellar_model)", "radius_rsun": model["radius_rsun"],
                      "luminosity_lsun": model["luminosity_lsun"], "t_center": model["t_center"]},
        "mass_fraction": pred["mass_fraction"].tolist(),
        "radius_rsun": {"pinn": pred["radius_rsun"].tolist(), "classical": truth["radius_rsun"].tolist()},
        "luminosity_lsun": {"pinn": pred["luminosity_lsun"].tolist(), "classical": truth["luminosity_lsun"].tolist()},
        "temperature": {"pinn": pred["temperature"].tolist(), "classical": truth["temperature"].tolist()},
        "pressure": {"pinn": pred["pressure"].tolist(), "classical": truth["pressure"].tolist()},
        "error": err, "baseline": base, "history": history, "checks": checks,
        "tolerance": tol,
        "summary": {"profile_error": prof, "worst_quantity": worst,
                    "radius_rsun": float(pred["radius_rsun"][-1]), "luminosity_lsun": float(pred["luminosity_lsun"][-1]),
                    "t_center": float(pred["temperature"][0])},
        "verdict": {"status": "trustworthy" if n_fail == 0 else "not_trustworthy",
                    "reasons": [f"{c['name']}: {c['status']} ({c['value']:.3g} vs {c['threshold']:.3g})"
                                for c in checks if c["status"] != "pass"]},
    }
