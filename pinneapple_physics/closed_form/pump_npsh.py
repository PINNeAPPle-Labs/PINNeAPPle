"""Centrifugal pump on a suction / discharge system: duty point, NPSH available against required, cavitation margin.

* Water saturation pressure: IAPWS-IF97 region 4 saturation-pressure equation (IAPWS R7-97(2012), eq. 30). It
  reproduces the release's verification values (Table 35) to 9 digits, and it is valid from 273.15 to 647.096 K.
* Liquid density: Incropera, *Fundamentals of Heat and Mass Transfer*, Table A.6 (saturated water), interpolated
  linearly.
* Atmosphere: ISO 2533 standard atmosphere in the troposphere.
* Suction and discharge losses: Darcy-Weisbach with Colebrook (:mod:`.pipe_flow`) plus minor-loss coefficients.
* Pump: the head and required-NPSH curves are quadratic fits through the catalogue points the caller gives
  (shut-off head, best-efficiency point and so on). The duty point solves ``H_pump(Q) = H_static + K_sys Q^2``.
* NPSH available: ``(p_surface - p_v) / (rho g) + z_s - h_f,suction``, where z_s is positive when the liquid level
  is above the pump (flooded suction) and negative for a suction lift (Karassik et al., *Pump Handbook*, ch. 2).
  The margin follows ANSI/HI 9.6.1-2017: NPSH_a / NPSH_3 >= 1.1 to 2.5, depending on the application.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .pipe_flow import friction_factor

__all__ = ["iapws97_psat", "water_density", "atmosphere", "pipe_loss", "fit_quadratic", "duty_point", "npsh_available"]

G = 9.80665
_N = (0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2, 0.12020824702470e5, -0.32325550322333e7,
      0.14915108613530e2, -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849, 0.65017534844798e3)

# Incropera Table A.6, saturated water: T [K], specific volume of the liquid [1e-3 m3/kg], dynamic viscosity [1e-6 Pa s]
_T = (273.15, 280, 290, 300, 310, 320, 330, 340, 350, 360, 373.15, 400, 450)
_VF = (1.000, 1.000, 1.001, 1.003, 1.007, 1.011, 1.016, 1.021, 1.027, 1.034, 1.044, 1.067, 1.123)
_MU = (1750, 1422, 1080, 855, 695, 577, 489, 420, 365, 324, 279, 217, 152)


def iapws97_psat(t_c: float) -> float:
    """Saturation pressure of water (Pa) at ``t_c`` (°C), IAPWS-IF97 eq. 30.

    Returns
    -------
    p_sat : float [Pa]
        Saturation pressure.
    """
    T = t_c + 273.15
    if not 273.15 <= T <= 647.096:
        raise ValueError("IAPWS-IF97 saturation line: 273.15 K <= T <= 647.096 K")
    n = _N
    th = T + n[8] / (T - n[9])
    A = th ** 2 + n[0] * th + n[1]
    B = n[2] * th ** 2 + n[3] * th + n[4]
    C = n[5] * th ** 2 + n[6] * th + n[7]
    return (2 * C / (-B + math.sqrt(B * B - 4 * A * C))) ** 4 * 1e6


def water_density(t_c: float) -> dict[str, float]:
    T = t_c + 273.15
    return {"rho": 1e3 / float(np.interp(T, _T, _VF)), "mu": 1e-6 * float(np.interp(T, _T, _MU))}


def atmosphere(z_m: float) -> float:
    """ISO 2533 pressure (Pa) at altitude ``z_m`` (troposphere).

    Returns
    -------
    p : float [Pa]
        Atmospheric pressure.
    """
    return 101325.0 * (1 - 2.25577e-5 * z_m) ** 5.25588


def pipe_loss(q: float, d: float, length: float, k_minor: float, rough: float, rho: float, mu: float) -> dict[str, float]:
    """Head loss (m) of a pipe with minor losses at flow ``q`` (m3/s)."""
    v = q / (math.pi * d * d / 4)
    re = rho * v * d / mu
    f = friction_factor(re, rough / d)
    return {"h": (f * length / d + k_minor) * v * v / (2 * G), "v": v, "re": re, "f": f}


def fit_quadratic(q: Sequence[float], y: Sequence[float]) -> np.ndarray:
    """Least-squares ``y = c0 + c1 q + c2 q^2`` through catalogue points; returns (c0, c1, c2)."""
    q = np.asarray(q, float)
    return np.linalg.lstsq(np.c_[np.ones_like(q), q, q * q], np.asarray(y, float), rcond=None)[0]


def duty_point(head_coef: np.ndarray, h_static: float, system_loss, q_max: float) -> float:
    """Flow where the pump head equals the system head (bisection; ``system_loss(q)`` in m)."""
    f = lambda q: (head_coef[0] + head_coef[1] * q + head_coef[2] * q * q) - h_static - system_loss(q)  # noqa: E731
    lo, hi = 1e-9, q_max
    if f(lo) < 0:
        raise ValueError("the pump cannot reach the static head (shut-off head below it)")
    if f(hi) > 0:
        raise ValueError("no duty point below q_max: the pump curve is extrapolated")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


def npsh_available(p_surface: float, t_c: float, z_suction: float, h_loss: float) -> dict[str, float]:
    """NPSH available (m) at the pump suction flange for a liquid surface at ``p_surface`` (Pa abs) and ``z_suction``
    (m, liquid level above the pump centre line: negative for a suction lift)."""
    rho = water_density(t_c)["rho"]
    pv = iapws97_psat(t_c)
    return {"npsha": (p_surface - pv) / (rho * G) + z_suction - h_loss, "pv": pv, "rho": rho,
            "h_surface": p_surface / (rho * G), "h_vap": pv / (rho * G)}
