"""Heat loss through an insulated pipe and the economic insulation thickness (ISO 12241 / VDI 2055 method).

Steady radial conduction through the insulation, whose conductivity is linear in temperature
``k(T) = k0 + k1 T``. Integrating the conductivity gives the exact Kirchhoff transform, so the mean of k over
[Ts, Ti] applies exactly:

    q' = 2 pi k_m (Ti - Ts) / ln(D_e / D_i)

The pipe wall and the inner film are neglected: steam or hot liquid is at the wall temperature, which errs on the
safe side. The outer surface coefficient follows ISO 12241:2008 sec. 4.1:

* radiation ``h_r = eps sigma (Ts^4 - Ta^4) / (Ts - Ta)`` (ISO 12241 sec. 4.1.2);
* convection on a horizontal cylinder in air (properties at the film temperature, Incropera Table A.4): natural by
  Churchill & Chu (1975, Incropera eq. 9.34), forced cross-flow by Churchill & Bernstein (1977, eq. 7.54), and
  mixed as ``Nu^3 = Nu_F^3 + Nu_N^3`` (Incropera eq. 9.66).

The surface temperature is found by bisection on the energy balance of the outer surface.

:func:`radial_fd` solves the same layer by finite volumes with k(T). Because it does not use the Kirchhoff
transform, it is an independent check. :func:`economic_thickness` minimises the annual cost, which is the lost
energy plus the annualised installed cost of the insulation (VDI 2055-1 sec. 6), over the standard thicknesses.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = ["surface_coefficient", "heat_loss", "radial_fd", "economic_thickness", "crf"]

SIGMA = 5.670374419e-8


_AIR_T = (250.0, 300.0, 350.0, 400.0, 450.0, 500.0)               # K, Incropera Table A.4 (1 atm)
_AIR_NU = (11.44e-6, 15.89e-6, 20.92e-6, 26.41e-6, 32.39e-6, 38.79e-6)
_AIR_K = (0.0223, 0.0263, 0.0300, 0.0338, 0.0373, 0.0407)
_AIR_PR = (0.720, 0.707, 0.700, 0.690, 0.686, 0.684)


def air(t_c: float) -> dict[str, float]:
    T = t_c + 273.15
    return {"nu": float(np.interp(T, _AIR_T, _AIR_NU)), "k": float(np.interp(T, _AIR_T, _AIR_K)),
            "pr": float(np.interp(T, _AIR_T, _AIR_PR)), "beta": 1 / T}


def surface_coefficient(ts: float, ta: float, d: float, eps: float, wind: float = 0.0) -> dict[str, float]:
    """Outer coefficient (W/m2K) of a horizontal cylinder of diameter d (m) in air: radiation + mixed convection."""
    dt = max(abs(ts - ta), 1e-6)
    hr = eps * SIGMA * ((ts + 273.15) ** 4 - (ta + 273.15) ** 4) / (ts - ta) if abs(ts - ta) > 1e-9 else \
        4 * eps * SIGMA * (ta + 273.15) ** 3
    a = air(0.5 * (ts + ta))
    alpha = a["nu"] / a["pr"]
    ra = 9.80665 * a["beta"] * dt * d ** 3 / (a["nu"] * alpha)
    nu_n = (0.60 + 0.387 * ra ** (1 / 6) / (1 + (0.559 / a["pr"]) ** (9 / 16)) ** (8 / 27)) ** 2
    re = wind * d / a["nu"]
    pr = a["pr"]
    nu_f = 0.3 + 0.62 * re ** 0.5 * pr ** (1 / 3) / (1 + (0.4 / pr) ** (2 / 3)) ** 0.25 * (
        1 + (re / 282000) ** 0.625) ** 0.8 if re > 0 else 0.0
    nu = (nu_f ** 3 + nu_n ** 3) ** (1 / 3)
    h_cv = nu * a["k"] / d
    return {"h_r": hr, "h_cv": h_cv, "h": hr + h_cv, "re": re, "ra": ra,
            "regime": "forced" if nu_f > nu_n else "natural"}


def _k_mean(k0: float, k1: float, a: float, b: float) -> float:
    return k0 + k1 * 0.5 * (a + b)


def heat_loss(d_pipe: float, thickness: float, t_in: float, t_amb: float, *, k0: float, k1: float = 0.0,
              eps: float = 0.9, wind: float = 0.0) -> dict[str, float]:
    """Heat loss per metre (W/m), surface temperature and coefficients of an insulated (or bare: thickness 0)
    pipe."""
    de = d_pipe + 2 * thickness

    def residual(ts):
        h = surface_coefficient(ts, t_amb, de, eps, wind)["h"]
        q_out = h * math.pi * de * (ts - t_amb)
        if thickness <= 0:
            return q_out, q_out
        q_cond = 2 * math.pi * _k_mean(k0, k1, ts, t_in) * (t_in - ts) / math.log(de / d_pipe)
        return q_cond - q_out, q_out

    if thickness <= 0:
        ts = t_in
    else:
        lo, hi = t_amb, t_in
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if residual(mid)[0] > 0 else (lo, mid)
        ts = 0.5 * (lo + hi)
    co = surface_coefficient(ts, t_amb, de, eps, wind)
    q = co["h"] * math.pi * de * (ts - t_amb)
    return {"q": q, "t_surface": ts, "d_outer": de, **co,
            "k_mean": _k_mean(k0, k1, ts, t_in) if thickness > 0 else float("nan")}


def radial_fd(d_pipe: float, thickness: float, t_in: float, t_surface: float, *, k0: float, k1: float = 0.0,
              n: int = 400) -> dict[str, np.ndarray]:
    """Temperature across the insulation by finite volumes in r with k(T), Picard iterations. The inner and outer
    temperatures are given. Returns r, T and the heat flow per metre at the inner and outer faces."""
    ri, ro = d_pipe / 2, d_pipe / 2 + thickness
    r = np.linspace(ri, ro, n + 1)
    T = t_in + (t_surface - t_in) * np.log(r / ri) / math.log(ro / ri)
    for _ in range(100):
        rf = 0.5 * (r[1:] + r[:-1])
        kf = k0 + k1 * 0.5 * (T[1:] + T[:-1])
        g = 2 * math.pi * rf * kf / np.diff(r)                  # conductance of each shell (per metre)
        A = np.zeros((n + 1, n + 1))
        b = np.zeros(n + 1)
        A[0, 0] = A[n, n] = 1.0
        b[0], b[n] = t_in, t_surface
        for i in range(1, n):
            A[i, i - 1], A[i, i + 1] = g[i - 1], g[i]
            A[i, i] = -(g[i - 1] + g[i])
        Tn = np.linalg.solve(A, b)
        if np.abs(Tn - T).max() < 1e-11:
            T = Tn
            break
        T = Tn
    kf = k0 + k1 * 0.5 * (T[1:] + T[:-1])
    flux = 2 * math.pi * 0.5 * (r[1:] + r[:-1]) * kf * -np.diff(T) / np.diff(r)
    return {"r": r, "T": T, "q_inner": float(flux[0]), "q_outer": float(flux[-1])}


def crf(rate: float, years: float) -> float:
    """Capital recovery factor ``i (1+i)^n / ((1+i)^n - 1)``."""
    return rate / (1 - (1 + rate) ** -years) if rate > 0 else 1 / years


def economic_thickness(d_pipe: float, t_in: float, t_amb: float, thicknesses: Sequence[float], *, k0: float,
                       k1: float, eps: float, wind: float, hours: float, energy_price: float, efficiency: float,
                       cost_per_m3: float, cost_per_m2: float, rate: float, years: float,
                       t_surface_max: float = float("inf")) -> dict[str, object]:
    """Annual cost per metre (energy + annualised installed insulation) for each thickness. ``energy_price`` is
    the price per kWh of fuel, which ``efficiency`` converts to heat. ``cost_per_m3`` is charged on the insulation
    volume, ``cost_per_m2`` on the cladding area. Returns the table, the economic thickness and the thinnest
    thickness that meets ``t_surface_max``."""
    rows = []
    for t in thicknesses:
        hl = heat_loss(d_pipe, t, t_in, t_amb, k0=k0, k1=k1, eps=eps, wind=wind)
        e = hl["q"] * hours / 1000 / efficiency * energy_price
        de = d_pipe + 2 * t
        capex = cost_per_m3 * math.pi / 4 * (de ** 2 - d_pipe ** 2) + (cost_per_m2 * math.pi * de if t > 0 else 0)
        rows.append({"thickness": t, "q": hl["q"], "t_surface": hl["t_surface"], "energy_cost": e,
                     "capital_cost": capex, "annual_capital": capex * crf(rate, years),
                     "total": e + capex * crf(rate, years)})
    best = min((r for r in rows if r["thickness"] > 0), key=lambda r: r["total"])
    safe = [r for r in rows if r["t_surface"] <= t_surface_max]
    return {"rows": rows, "economic": best, "thinnest_safe": safe[0] if safe else None}
