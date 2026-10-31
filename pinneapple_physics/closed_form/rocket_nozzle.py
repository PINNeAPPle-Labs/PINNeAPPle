"""Ideal rocket nozzle: 1-D isentropic flow of a calorically perfect gas through a converging-diverging nozzle.

Exit Mach number from the area ratio (inverse of the area-Mach relation, supersonic or subsonic branch, by
bisection), choked mass flow, characteristic velocity c*, thrust coefficient C_F (momentum + pressure terms) and
specific impulse. Sutton & Biblarz, *Rocket Propulsion Elements*, 9th ed. (2017), ch. 3 (eqs. 3-16, 3-23, 3-24,
3-30, 3-32, 3-33); Anderson, *Modern Compressible Flow*, 3rd ed. (2003), ch. 5 and Table A.1. The forward
area-Mach and pressure ratios come from :func:`pinneapple_physics.closed_form.compressible.isentropic_ratios`.

Assumptions: constant gamma, isentropic (no shocks, friction or heat loss), 1-D section averages, flow attached
and expanded to the given area ratio. Those are what a CFD run checks next; this is a screening estimate.

Moved here from the PINNeAPPle-apps ``rocket_nozzle_screening`` product (``isentropic.py``); the checks of its
``verify_analytical.py`` are ``tests/test_closed_form_rocket_nozzle.py``.
"""
from __future__ import annotations

import math

from .compressible import isentropic_ratios

__all__ = ["G0", "area_ratio", "mach_from_area_ratio", "pressure_ratio", "temperature_ratio",
           "choked_mass_flow", "characteristic_velocity", "thrust_coefficient", "specific_impulse",
           "nozzle_performance"]

G0 = 9.80665  # standard gravity, m/s^2 (Isp in seconds)


def area_ratio(M: float, gamma: float) -> float:
    """A/A* at Mach ``M`` (both branches).

    Returns
    -------
    ratio : float [-]
        A/A*.
    """
    if M <= 0:
        raise ValueError("Mach number must be positive")
    return float(isentropic_ratios(M, gamma)["A_Astar"])


def mach_from_area_ratio(ratio: float, gamma: float, *, supersonic: bool = True, tol: float = 1e-12,
                         max_iter: int = 300) -> float:
    """Invert A/A*(M) = ``ratio`` (>= 1) on the supersonic (default) or subsonic branch, by bisection.

    Returns
    -------
    M : float [-]
        Mach number.
    """
    if ratio < 1.0 - 1e-12:
        raise ValueError(f"area ratio must be >= 1 (got {ratio})")
    ratio = max(ratio, 1.0)
    if ratio == 1.0:
        return 1.0

    def f(M):
        return area_ratio(M, gamma) - ratio

    if supersonic:
        lo, hi = 1.0, 2.0
        while f(hi) < 0.0:
            hi *= 2.0
            if hi > 1e6:
                raise ValueError("area ratio too large to bracket a supersonic root")
    else:
        lo, hi = 1e-12, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if hi - lo < tol:
            break
        # A/A* decreases with M below 1 and increases above it
        if (f(mid) < 0.0) == supersonic:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def pressure_ratio(M: float, gamma: float) -> float:
    """Static over stagnation pressure p/p0 at Mach ``M``.

    Returns
    -------
    ratio : float [-]
        p/p0.
    """
    return float(1.0 / isentropic_ratios(M, gamma)["p0_p"])


def temperature_ratio(M: float, gamma: float) -> float:
    """Static over stagnation temperature T/T0 at Mach ``M``.

    Returns
    -------
    ratio : float [-]
        T/T0.
    """
    return float(1.0 / isentropic_ratios(M, gamma)["T0_T"])


def choked_mass_flow(p0: float, throat_area: float, gamma: float, R_gas: float, T0: float) -> float:
    """Mass flow (kg/s) through a choked throat (Sutton & Biblarz eq. 3-24).

    Returns
    -------
    mdot : float [kg/s]
        Mass flow, for SI inputs (Pa, m2, J/(kg K), K).
    """
    k = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    return p0 * throat_area * math.sqrt(gamma / (R_gas * T0)) * (2.0 / (gamma + 1.0)) ** k


def characteristic_velocity(gamma: float, R_gas: float, T0: float) -> float:
    """c* = p0 A_t / mdot (m/s): depends on the gas and chamber temperature only (eq. 3-32).

    Returns
    -------
    c_star : float [m/s]
        Characteristic velocity, for SI inputs (J/(kg K), K).
    """
    k = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    return math.sqrt(R_gas * T0 / gamma) * ((gamma + 1.0) / 2.0) ** k


def thrust_coefficient(p_exit: float, p0: float, p_atm: float, ratio: float, gamma: float) -> float:
    """C_F = momentum term + (p_e - p_a) / p0 * A_e/A_t (eq. 3-30).

    Returns
    -------
    C_F : float [-]
        Thrust coefficient.
    """
    k = (gamma + 1.0) / (gamma - 1.0)
    momentum = (2.0 * gamma ** 2 / (gamma - 1.0) * (2.0 / (gamma + 1.0)) ** k
                * (1.0 - (p_exit / p0) ** ((gamma - 1.0) / gamma)))
    return math.sqrt(max(momentum, 0.0)) + (p_exit - p_atm) / p0 * ratio


def specific_impulse(cf: float, c_star: float, g0: float = G0) -> float:
    """Isp = C_F c* / g0 (s), eq. 3-33.

    Returns
    -------
    Isp : float [s]
        Specific impulse, for ``c_star`` in m/s and ``g0`` in m/s2.
    """
    return cf * c_star / g0


def nozzle_performance(*, gamma: float, R_gas: float, T_chamber: float, p_chamber: float, area_ratio_exit: float,
                       p_atm: float = 101325.0) -> dict:
    """Exit state and performance of an ideal nozzle expanded to ``area_ratio_exit`` = A_e/A_t."""
    M = mach_from_area_ratio(area_ratio_exit, gamma, supersonic=True)
    p_e = pressure_ratio(M, gamma) * p_chamber
    T_e = temperature_ratio(M, gamma) * T_chamber
    c_star = characteristic_velocity(gamma, R_gas, T_chamber)
    cf = thrust_coefficient(p_e, p_chamber, p_atm, area_ratio_exit, gamma)
    return {"mach_exit": M, "p_exit_pa": p_e, "T_exit_k": T_e, "v_exit_m_s": M * math.sqrt(gamma * R_gas * T_e),
            "c_star_m_s": c_star, "thrust_coefficient": cf, "specific_impulse_s": specific_impulse(cf, c_star),
            "assumptions": "calorically perfect gas, isentropic, 1-D, no shocks or separation, "
                           "expanded to the given area ratio"}
