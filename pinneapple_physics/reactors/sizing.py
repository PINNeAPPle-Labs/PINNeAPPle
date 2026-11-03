"""Sizing and safety of a reactor: the volume for a conversion, the Levenspiel plot, yield and selectivity, the
number of tanks in series, the adiabatic temperature rise, Semenov's runaway number and the parametric sensitivity
of a cooled PFR.

* **Volume for a conversion**: a PFR is integrated until the key species reaches it (an event of the integration,
  not a search); a CSTR's volume is found by Brent's method on the stable steady state a start-up reaches.
* **Levenspiel plot**: ``F_A0 / (-r_A)`` against ``X`` along the PFR; its area is the PFR volume and the rectangle at
  the outlet the CSTR volume (Levenspiel, *Chemical Reaction Engineering*, 3rd ed., ch. 5).
* **Yield and selectivity** of a desired product ``D`` from the key reactant ``A``: ``Y = (F_D - F_D0) / F_A0`` and
  ``S = (F_D - F_D0) / (F_A0 - F_A)`` (Fogler, 5th ed., sec. 8.1).
* **Adiabatic rise** ``dT_ad = sum_j (-dH_j) xi_j / sum_i F_i0 cp_i`` for full conversion of the key reactant by
  each reaction.
* **Semenov number** of a cooled reactor at its inlet: ``psi = (-dH) r(c0, Tc) Ea / (Ua R Tc^2)``; a zeroth-order
  exothermic reaction runs away when ``psi > 1/e`` (Semenov 1928). For reactions of positive order it is a
  conservative bound (consumption of the reactant helps).
* **Parametric sensitivity** (Morbidelli & Varma, *AIChE J.* 34 (1988) 1863-1877): the normalized sensitivity of the
  hot spot to the coolant temperature, ``s = (Tc / T_max) dT_max/dTc``; the runaway boundary is the coolant
  temperature where ``s`` peaks. The margin is the distance from the operating ``Tc`` to it.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
from scipy.optimize import brentq

from .kinetics import R_GAS, Feed, Network, concentrations
from .models import Bed, Cooling, cstr, cstr_startup, pfr


def size_pfr(net: Network, feed: Feed, key: str, X: float, cooling: Cooling | None = None, bed: Bed | None = None, *,
             V_max: float = 1e3) -> dict[str, Any]:
    """The PFR (or packed-bed) volume at which ``key`` reaches conversion ``X``: ``{"V", "reached", "profile"}``. Not
    reached within ``V_max`` (equilibrium, a reactant running out, the bed losing its pressure): ``reached=False``
    with the conversion it got to."""
    prof = pfr(net, feed, V_max, cooling, bed, stop=(key, X))
    return {"V": prof.info["V_end"], "reached": prof.info["reached"], "X_end": float(prof.conversion(key)[-1]),
            "profile": prof}


def _cstr_state(net, feed, V, cooling, key):
    states = cstr(net, feed, V, cooling)
    if feed.phase == "liquid":
        s = cstr_startup(net, feed, V, cooling)
        return min(states, key=lambda st: abs(st.T - s["T"]))
    return next(st for st in states if st.stable)


def size_cstr(net: Network, feed: Feed, key: str, X: float, cooling: Cooling | None = None, *,
              V_bounds: tuple[float, float] = (1e-6, 1e3)) -> dict[str, Any]:
    """The CSTR volume whose start-up state (liquid) or coldest stable state (gas) reaches conversion ``X``."""
    def gap(logV):
        return _cstr_state(net, feed, math.exp(logV), cooling, key).conversion(key, feed) - X
    lo, hi = (math.log(v) for v in V_bounds)
    if gap(hi) < 0:
        return {"V": None, "reached": False, "X_end": gap(hi) + X}
    if gap(lo) > 0:
        return {"V": V_bounds[0], "reached": True, "X_end": gap(lo) + X}
    logV = brentq(gap, lo, hi, xtol=1e-10)
    V = math.exp(logV)
    return {"V": V, "reached": True, "X_end": gap(logV) + X, "state": _cstr_state(net, feed, V, cooling, key)}


def levenspiel(net: Network, feed: Feed, key: str, V: float, cooling: Cooling | None = None,
               bed: Bed | None = None, n: int = 201) -> dict[str, np.ndarray]:
    """``X`` and ``F_A0 / (-r_A)`` (m^3) along the PFR; ``area`` (trapezoid) equals the volume when the rate stays
    positive."""
    prof = pfr(net, feed, V, cooling, bed, n=n)
    X = prof.conversion(key)
    y = feed.F[key] / np.maximum(-prof.rate_of(key), 1e-300)
    return {"X": X, "FA0_over_rate": y, "area": float(np.trapezoid(y, X)), "V": float(V)}


def yield_selectivity(F_in: dict[str, float], F_out: dict[str, float], key: str, desired: str) -> dict[str, float]:
    made = F_out[desired] - F_in.get(desired, 0.0)
    used = F_in[key] - F_out[key]
    return {"yield": made / F_in[key], "selectivity": made / used if used > 0 else float("nan")}


def cstrs_for(net: Network, feed: Feed, key: str, X: float, n_max: int = 10, cooling: Cooling | None = None
              ) -> list[dict[str, Any]]:
    """Equal tanks in series: for 1..n_max tanks, the volume of each so the train reaches ``X``, and the total."""
    out = []
    for n in range(1, n_max + 1):
        def gap(logV, n=n):
            f = feed
            for _ in range(n):
                st = _cstr_state(net, f, math.exp(logV), cooling, key)
                f = Feed(dict(st.F), st.T, f.phase, f.v, f.P)
            return 1 - f.F[key] / feed.F[key] - X
        logV = brentq(gap, math.log(1e-8), math.log(1e3), xtol=1e-10)
        out.append({"n": n, "V_each": math.exp(logV), "V_total": n * math.exp(logV)})
    return out


def adiabatic_rise(net: Network, feed: Feed, key: str) -> dict[str, float]:
    """Temperature rise for full conversion of ``key`` by each reaction that consumes it, K."""
    F0 = feed.vector(net)
    heat_cap = float(F0 @ net.cp_vec)
    out = {}
    for r, dH in zip(net.reactions, net.dH_at(feed.T), strict=True):
        nu = r.stoich.get(key, 0.0)
        if nu < 0:
            out[r.name or str(r.stoich)] = float(-dH * feed.F[key] / -nu / heat_cap)
    return out


def semenov_number(net: Network, feed: Feed, cooling: Cooling, reaction: int = 0) -> dict[str, float]:
    """Semenov's ``psi`` at the inlet composition and the coolant temperature; runaway when ``psi > 1/e``."""
    if cooling.Ua <= 0:
        return {"psi": float("inf"), "critical": 1 / math.e, "runaway": True}
    r = net.reactions[reaction]
    Tc = cooling.Tc
    c, p = concentrations(feed.vector(net), Tc, feed.P, feed)
    rate = float(net.rates(c, p, Tc)[reaction])
    psi = -float(net.dH_at(Tc)[reaction]) * rate * r.Ea / (cooling.Ua * R_GAS * Tc ** 2)
    return {"psi": psi, "critical": 1 / math.e, "runaway": psi > 1 / math.e}


def parametric_sensitivity(net: Network, feed: Feed, V: float, cooling: Cooling, bed: Bed | None = None, *,
                           Tc_range: tuple[float, float] | None = None, n: int = 41, inlet_follows: bool = True
                           ) -> dict[str, Any]:
    """Hot spot and normalized sensitivity ``s = (Tc / T_max) dT_max/dTc`` over a coolant-temperature scan; the
    runaway boundary ``Tc_critical`` is where ``s`` peaks (Morbidelli & Varma 1988). ``inlet_follows``: the feed
    enters at the coolant temperature (the usual multitubular case)."""
    lo, hi = Tc_range or (cooling.Tc - 40.0, cooling.Tc + 40.0)
    step = (hi - lo) / (n - 1)
    Tcs: list[float] = []
    Tmax: list[float] = []

    def run(Tc):
        f = replace(feed, T=float(Tc)) if inlet_follows else feed
        return float(pfr(net, f, V, replace(cooling, Tc=float(Tc)), bed, n=401).T.max())
    grid = list(np.linspace(lo, hi, n))
    for _ in range(4):                       # extend the scan upward while the peak sits at its upper edge
        for Tc in grid:
            Tcs.append(float(Tc))
            Tmax.append(run(Tc))
        T_a, M_a = np.array(Tcs), np.array(Tmax)
        s = T_a / M_a * np.gradient(M_a, T_a)
        k = int(np.argmax(s))
        if k < len(Tcs) - 1:
            break
        grid = list(Tcs[-1] + step * np.arange(1, n // 2 + 1))
    interior = 0 < k < len(Tcs) - 1
    return {"Tc": T_a, "T_max": M_a, "s": s, "Tc_critical": float(T_a[k]) if interior else None,
            "margin": float(T_a[k] - cooling.Tc) if interior else None, "peak_inside_scan": interior,
            "s_at_operation": float(np.interp(cooling.Tc, T_a, s))}
