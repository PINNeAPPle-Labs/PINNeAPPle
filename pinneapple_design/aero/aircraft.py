"""From a section polar to aircraft numbers: top speed, CO2 per 100 km, stall speed, and the design constraints.

A light single-engine propeller aircraft (defaults: Cessna 172 class). The wing uses the section polar with an
elliptic-loading induced drag CL^2/(pi e AR); the rest of the airplane is a fixed CD0. Piston power lapses with
altitude (Gagg-Ferrar), fuel from the brake-specific fuel consumption, CO2 from the fuel burnt.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

G = 9.80665
CO2_PER_KG_AVGAS = 3.10                     # kg CO2 per kg of avgas burnt
KT = 0.514444


def isa_density(h: float) -> float:
    T = 288.15 - 0.0065 * h
    return 1.225 * (T / 288.15) ** 4.2559


@dataclass
class Aircraft:
    mass: float = 1100.0                    # kg
    wing_area: float = 16.2                 # m2
    aspect_ratio: float = 7.5
    oswald: float = 0.80
    cd0_rest: float = 0.020                 # fuselage, tail, gear... referred to wing area
    power_kw: float = 120.0                 # sea-level rated power
    prop_efficiency: float = 0.80
    bsfc: float = 0.30                      # kg/kWh
    cruise_speed: float = 55.0              # m/s (107 kt)
    cruise_altitude: float = 2000.0         # m
    clmax_3d_factor: float = 0.90           # wing CLmax / section clmax (clean wing)

    @property
    def weight(self) -> float:
        return self.mass * G

    @property
    def chord(self) -> float:
        return math.sqrt(self.wing_area / self.aspect_ratio)


@dataclass
class Requirements:
    max_stall_speed: float = 61 * KT        # CS-23 / Part 23 single-engine limit (here: clean wing, no flaps)
    min_thickness: float = 0.11             # spar depth / fuel volume
    max_abs_cm: float = 0.10                # trim drag and tail size
    max_cruise_cl_ratio: float = 0.70       # cruise cl <= 70 % of clmax (gust and manoeuvre margin)

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


class Polar:
    """Section polar from coefficients at a few angles: cl(alpha) and cd, cm as functions of cl on the attached part."""

    def __init__(self, alpha: np.ndarray, cl: np.ndarray, cd: np.ndarray, cm: np.ndarray):
        o = np.argsort(alpha)
        self.alpha, self.cl, self.cd, self.cm = (np.asarray(v, float)[o] for v in (alpha, cl, cd, cm))
        k = int(np.argmax(self.cl))
        self.clmax, self.alpha_clmax = float(self.cl[k]), float(self.alpha[k])
        self.at_edge = k == len(self.cl) - 1                     # clmax not reached inside the sampled range
        a = self.alpha[: k + 1]
        self._a, self._cl = a, self.cl[: k + 1]
        self._cd, self._cm = self.cd[: k + 1], self.cm[: k + 1]

    def alpha_for(self, cl: float) -> float:
        return float(np.interp(cl, self._cl, self._a, left=np.nan, right=np.nan))

    def at_cl(self, cl: float) -> Dict[str, float]:
        if cl < self._cl[0] or cl > self._cl[-1]:
            # linear extrapolation in cl, flagged
            i = (0, 1) if cl < self._cl[0] else (-2, -1)
            f = lambda y: float(y[i[0]] + (y[i[1]] - y[i[0]]) * (cl - self._cl[i[0]]) / (self._cl[i[1]] - self._cl[i[0]] + 1e-12))  # noqa: E731
            return {"alpha": f(self._a), "cd": max(f(self._cd), 1e-4), "cm": f(self._cm), "extrapolated": True}
        return {"alpha": self.alpha_for(cl), "cd": float(np.interp(cl, self._cl, self._cd)),
                "cm": float(np.interp(cl, self._cl, self._cm)), "extrapolated": False}


def performance(polar: Polar, ac: Aircraft = Aircraft(), req: Requirements = Requirements()) -> Dict[str, Any]:
    S, W, AR, e = ac.wing_area, ac.weight, ac.aspect_ratio, ac.oswald
    k_ind = 1 / (math.pi * e * AR)

    def drag(V, rho):
        q = 0.5 * rho * V * V
        CL = W / (q * S)
        s = polar.at_cl(CL)
        CD = s["cd"] + ac.cd0_rest + k_ind * CL * CL
        return q * S * CD, CL, CD, s

    # cruise: drag, fuel and CO2 per 100 km
    rho_c = isa_density(ac.cruise_altitude)
    D, CLc, CDc, sc = drag(ac.cruise_speed, rho_c)
    fuel_100km = D * 1e5 / ac.prop_efficiency * ac.bsfc / 3.6e6        # kg
    co2_100km = fuel_100km * CO2_PER_KG_AVGAS
    # top speed at full power at cruise altitude: eta P = D V
    sigma = rho_c / 1.225
    P = ac.power_kw * 1e3 * (1.132 * sigma - 0.132)
    V_s_alt = math.sqrt(2 * W / (rho_c * S * ac.clmax_3d_factor * polar.clmax))
    lo, hi = V_s_alt * 1.05, 150.0
    if drag(lo, rho_c)[0] * lo > ac.prop_efficiency * P:
        vmax = float("nan")
    else:
        for _ in range(60):
            m = 0.5 * (lo + hi)
            if drag(m, rho_c)[0] * m > ac.prop_efficiency * P:
                hi = m
            else:
                lo = m
        vmax = 0.5 * (lo + hi)
    _, CLv, CDv, sv = drag(vmax, rho_c) if vmax == vmax else (0, float("nan"), float("nan"), {"extrapolated": True})
    # stall at sea level, clean wing
    clmax_w = ac.clmax_3d_factor * polar.clmax
    v_stall = math.sqrt(2 * W / (1.225 * S * clmax_w))
    checks = [
        {"name": "Stall speed", "value": v_stall, "limit": req.max_stall_speed, "ok": v_stall <= req.max_stall_speed,
         "unit": "m/s", "why": "CS-23 limit for a single-engine aircraft (61 kt), applied to the clean wing"},
        {"name": "Trim (|Cm c/4| in cruise)", "value": abs(sc["cm"]), "limit": req.max_abs_cm,
         "ok": abs(sc["cm"]) <= req.max_abs_cm, "unit": "", "why": "a strongly nose-down section needs a bigger tail and costs trim drag"},
        {"name": "Cruise margin to stall (cl/clmax)", "value": CLc / polar.clmax, "limit": req.max_cruise_cl_ratio,
         "ok": CLc / polar.clmax <= req.max_cruise_cl_ratio, "unit": "", "why": "room for gusts and turns in cruise"},
    ]
    return {"vmax": vmax, "vmax_kt": vmax / KT, "co2_100km": co2_100km, "fuel_100km": fuel_100km,
            "fuel_l_100km": fuel_100km / 0.72, "cruise_drag": D, "cruise_cl": CLc, "cruise_cd": CDc,
            "cruise_ld": CLc / CDc, "cruise_alpha": sc["alpha"], "cruise_cm": sc["cm"], "cruise_cd_wing": sc["cd"],
            "v_stall": v_stall, "v_stall_kt": v_stall / KT, "clmax": polar.clmax, "clmax_at_edge": polar.at_edge,
            "extrapolated": bool(sc["extrapolated"] or sv.get("extrapolated")), "checks": checks,
            "vmax_cl": CLv, "power_at_altitude_kw": P / 1e3}


def evaluate(shape_props: Dict[str, float], polar: Polar, ac: Aircraft = Aircraft(),
             req: Requirements = Requirements()) -> Dict[str, Any]:
    """Performance plus the geometric requirement; feasible when every check passes."""
    perf = performance(polar, ac, req)
    t = shape_props["thickness"]
    perf["checks"].insert(0, {"name": "Thickness (spar depth, fuel)", "value": t, "limit": req.min_thickness,
                              "ok": t >= req.min_thickness, "unit": "", "why": "structure and fuel need depth"})
    perf["feasible"] = all(c["ok"] for c in perf["checks"]) and perf["vmax"] == perf["vmax"]
    perf["violations"] = [c["name"] for c in perf["checks"] if not c["ok"]]
    # normalised total violation (for constrained NSGA-II)
    pen = 0.0
    for c in perf["checks"]:
        if not c["ok"]:
            lim = abs(c["limit"]) or 1.0
            over = (c["value"] - c["limit"]) if c["name"] != "Thickness (spar depth, fuel)" else (c["limit"] - c["value"])
            pen += max(over, 0) / lim
    perf["penalty"] = pen if perf["vmax"] == perf["vmax"] else pen + 10
    return perf
