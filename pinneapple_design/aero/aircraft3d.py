"""The whole aircraft in 3D: vortex lattice (wing + tail, trimmed), section drag strip by strip from the CFD
surrogate, fuselage and tail drag by component build-up (Raymer), wing and tail weights (Raymer, general aviation),
centre of gravity, neutral point with the fuselage's destabilising moment, stall by the critical-section method.

Requirements a design must meet: stall speed, stall starting inboard (not at the tips), static margin, trim with the
tail, spar depth, span. Objectives: top speed, CO2 per 100 km, stall speed.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .aircraft import CO2_PER_KG_AVGAS, KT, Aircraft, Polar, isa_density
from .airframe import Airframe
from .geometry import BOUNDS as SHAPE_BOUNDS, REFERENCE, properties
from .vlm import solve

G0 = 9.80665
LB, FT = 0.45359237, 0.3048

# design vector: 6 airfoil weights + planform + wing position
PLAN = ("wing_area", "aspect_ratio", "taper", "sweep_le", "twist", "wing_x")
PLAN_BOUNDS = np.array([[10.0, 20.0], [6.0, 11.0], [0.40, 1.0], [-2.0, 12.0], [-5.0, 0.0], [1.95, 2.75]])
BOUNDS3D = np.vstack([SHAPE_BOUNDS, PLAN_BOUNDS])
LABELS3D = {"wing_area": "wing area (m²)", "aspect_ratio": "aspect ratio", "taper": "taper ratio",
            "sweep_le": "leading-edge sweep (°)", "twist": "tip twist (°, washout < 0)", "wing_x": "wing position (m from nose)"}
BASELINE_PLAN = {"wing_area": 16.2, "aspect_ratio": 7.5, "taper": 0.70, "sweep_le": 0.0, "twist": -3.0, "wing_x": 2.30}


@dataclass
class Requirements3D:
    max_stall_speed: float = 61 * KT          # CS-23 single-engine limit (clean wing here)
    min_thickness: float = 0.11
    min_static_margin: float = 0.05
    max_static_margin: float = 0.30
    max_tail_incidence: float = 4.0           # deg needed to trim in cruise (elevator authority)
    max_stall_station: float = 0.60           # stall must begin inboard of 60 % semi-span (aileron control)
    max_span: float = 13.5                    # hangar / taxiway
    max_cruise_cl_ratio: float = 0.70         # worst strip in cruise vs its cl max

    def as_dict(self):
        return dataclasses.asdict(self)


def airframe_from(x: np.ndarray, base: Optional[Airframe] = None) -> Airframe:
    x = np.asarray(x, float)
    kw = dict(zip(PLAN, x[6:12].tolist()))
    return dataclasses.replace(base or Airframe(), section=x[:6].copy(), **kw)


def baseline_x() -> np.ndarray:
    return np.r_[REFERENCE["NACA 2412"], [BASELINE_PLAN[k] for k in PLAN]]


# ---------------------------------------------------------------------- weights (Raymer, general aviation, lb/ft)
def wing_weight(af: Airframe, Wdg: float, q: float, tc: float, nz: float = 5.7, Wfw: float = 70 * 3.785 * 0.72) -> float:
    """kg; Wdg design gross mass (kg), q cruise dynamic pressure (Pa), Wfw fuel in the wing (kg)."""
    S = af.wing_area / FT ** 2
    lam_c4 = math.atan(math.tan(math.radians(af.sweep_le)) - (1 - af.taper) / (af.aspect_ratio * (1 + af.taper)))
    w = (0.036 * S ** 0.758 * (Wfw / LB) ** 0.0035 * (af.aspect_ratio / math.cos(lam_c4) ** 2) ** 0.6
         * (q / 47.88) ** 0.006 * af.taper ** 0.04 * (100 * tc / math.cos(lam_c4)) ** -0.3 * (nz * Wdg / LB) ** 0.49)
    return w * LB


def htail_weight(af: Airframe, Wdg: float, q: float, nz: float = 5.7) -> float:
    ht = af.htail()
    S = ht["area"] / FT ** 2
    w = 0.016 * (nz * Wdg / LB) ** 0.414 * (q / 47.88) ** 0.168 * S ** 0.896 * (100 * 0.10) ** -0.12 * (af.htail_ar) ** 0.043 * 0.7 ** -0.02
    return w * LB


def vtail_weight(af: Airframe, Wdg: float, q: float, nz: float = 5.7) -> float:
    vt = af.vtail()
    S = vt["area"] / FT ** 2
    lam = math.radians(35)
    w = 0.073 * (nz * Wdg / LB) ** 0.376 * (q / 47.88) ** 0.122 * S ** 0.873 * (100 * 0.10 / math.cos(lam)) ** -0.49 \
        * (af.vtail_ar / math.cos(lam) ** 2) ** 0.357 * 0.55 ** 0.039
    return w * LB


# ---------------------------------------------------------------------- parasite drag build-up (Raymer)
def _cf(Re: float) -> float:
    return 0.455 / (math.log10(Re) ** 2.58)                       # turbulent flat plate (no laminar run)


def fuselage_wetted(af: Airframe):
    xi = np.linspace(0, 1, 200)
    w, h = af._fus_w(xi), af._fus_h(xi)
    perim = math.pi * (3 * (w / 2 + h / 2) - np.sqrt((3 * w / 2 + h / 2) * (w / 2 + 3 * h / 2)))
    swet = np.trapezoid(perim, xi * af.length)
    d = math.sqrt(af.width * af.height)
    f = af.length / d
    return swet, 1 + 60 / f ** 3 + f / 400


def fuselage_cd0(af: Airframe, V: float, rho: float, mu: float = 1.75e-5, wet=None) -> float:
    swet, ff = wet or fuselage_wetted(af)
    return _cf(rho * V * af.length / mu) * ff * swet / af.wing_area


def vtail_cd0(af: Airframe, V: float, rho: float, mu: float = 1.75e-5, vt=None) -> float:
    vt = vt or af.vtail()
    c = vt["root_chord"] * (1 + vt["taper"]) / 2
    ff = (1 + 0.6 / 0.3 * 0.10 + 100 * 0.10 ** 4) * 1.34 * 0.9    # t/c 10 %, (x/c)m 0.3, Mach ~0.15
    return _cf(rho * V * c / mu) * ff * 2.04 * vt["area"] / af.wing_area


MISC_D_OVER_Q = 0.25       # m2: gear with fairings, struts, cooling, antennas, interference. Calibrated once so the
                           # NACA 2412 baseline (Cessna 172N class) has CD0 ~0.030 (published ~0.031) and ~124 kt at
                           # full power; the same for every design.
FUS_KF = 0.012             # per degree, fuselage Munk moment factor (wing at ~30 % of fuselage length; Raymer fig 16.14)


class Aircraft3D:
    """Evaluate one 3D design given section polars (from the CFD surrogate)."""

    def __init__(self, x: np.ndarray, wing_polar: Polar, tail_polar: Polar, ac: Aircraft = Aircraft(),
                 req: Requirements3D = Requirements3D(), base_weights: Optional[Dict[str, float]] = None,
                 nc: int = 4, ns_wing: int = 12, ns_tail: int = 5):
        self.x, self.ac, self.req = np.asarray(x, float), ac, req
        self.af = airframe_from(x)
        self.props = properties(self.x[:6])
        af = self.af
        q_c = 0.5 * isa_density(ac.cruise_altitude) * ac.cruise_speed ** 2
        Wdg = ac.mass                                              # kg; the Raymer formulas take lb (converted inside)
        self.weights = {"wing": wing_weight(af, Wdg, q_c, self.props["thickness"]),
                        "htail": htail_weight(af, Wdg, q_c), "vtail": vtail_weight(af, Wdg, q_c)}
        bw = base_weights or self.weights
        self.mass = ac.mass + sum(self.weights[k] - bw[k] for k in self.weights)
        # CG: the rest of the aircraft at 2.67 m (the Cessna's CG at 25 % MAC of the reference wing); the wing's own
        # mass moves with it (its CG at 40 % of the MAC)
        x_rest = 2.67
        x_wing = af.mac_le_x + 0.40 * af.mac
        mw = self.weights["wing"]
        base_wing_x = 2.30 + 0.40 * 1.485
        self.x_cg = x_rest + mw * (x_wing - base_wing_x) / self.mass
        self.sol = solve(af, self.x_cg, nc=nc, ns_wing=ns_wing, ns_tail=ns_tail)
        # fuselage destabilising moment: Cm_alpha += K_f W_f^2 L_f / (S c) per degree
        self.dcma_fus = FUS_KF * af.width ** 2 * af.length / (af.wing_area * af.mac) * 180 / math.pi
        self.CLa = self.sol.CL[1]
        self.Cma = self.sol.Cm[1] + self.dcma_fus
        self.x_np = self.x_cg - self.Cma / self.CLa * af.mac
        self.static_margin = (self.x_np - self.x_cg) / af.mac
        self.wp, self.tp = wing_polar, tail_polar
        L = self.sol.L
        self.w_strips = L.strip_surf == 0
        self.t_strips = L.strip_surf == 1
        self.area_w = (L.chord * np.abs(L.dy))[self.w_strips]
        self.area_t = (L.chord * np.abs(L.dy))[self.t_strips]
        self.eta = np.abs(L.yc[self.w_strips]) / (af.span / 2)
        self._wet, self._vt = fuselage_wetted(af), af.vtail()
        self._trim = self._trim_lin()

    # ------------------------------------------------------------------ one flight condition
    def _trim_lin(self):
        """strip cl as a + b*CL along the trim line, and alpha / i_t too."""
        s = self.sol
        cm_fus = np.array([0.0, self.dcma_fus, 0.0])
        Cm = s.Cm + cm_fus
        A = np.array([[s.CL[1], s.CL[2]], [Cm[1], Cm[2]]])
        Ainv = np.linalg.inv(A)
        # [alpha, it] = Ainv @ ([CL, 0] - [CL0, Cm0]) = p + r*CL
        p = Ainv @ np.array([-s.CL[0], -Cm[0]])
        r = Ainv @ np.array([1.0, 0.0])
        return p, r

    def condition(self, CL: float) -> Dict[str, Any]:
        p, r = self._trim
        a, it = p + r * CL
        u = np.array([1.0, a, it])
        cls = self.sol.cl_strip.T @ u
        cdi = float(u @ self.sol.Q @ u)
        clw, clt = cls[self.w_strips], cls[self.t_strips]
        cdw = np.interp(clw, self.wp._cl, self.wp._cd, left=np.nan, right=np.nan)
        out_w = np.isnan(cdw)
        if out_w.any():                                            # outside the attached polar: linear extrapolation
            cdw[out_w] = [self.wp.at_cl(c)["cd"] for c in clw[out_w]]
        cdt = np.interp(clt, self.tp._cl, self.tp._cd)                # tail stays well inside its polar
        cd_w = float((cdw * self.area_w).sum() / self.af.wing_area)
        cd_t = float((cdt * self.area_t).sum() / self.af.wing_area)
        # beyond the polar: above it (towards stall) is a real unknown; slightly below cl(-2 deg) is the drag bucket
        far = (clw > self.wp._cl[-1]) | (clw < self.wp._cl[0] - 0.15)
        return {"alpha": math.degrees(a), "it": math.degrees(it), "CDi": cdi, "CDp_wing": cd_w, "CDp_tail": cd_t,
                "cl_wing": clw, "extrapolated": bool(far.any())}

    def drag(self, V: float, rho: float) -> Dict[str, Any]:
        q = 0.5 * rho * V * V
        CL = self.mass * G0 / (q * self.af.wing_area)
        c = self.condition(CL)
        cd0 = (fuselage_cd0(self.af, V, rho, wet=self._wet) + vtail_cd0(self.af, V, rho, vt=self._vt)
               + MISC_D_OVER_Q / self.af.wing_area)
        CD = c["CDi"] + c["CDp_wing"] + c["CDp_tail"] + cd0
        c.update(CL=CL, CD=CD, CD0_other=cd0, D=q * self.af.wing_area * CD)
        return c

    def at_alpha(self, alpha_deg: float, it_deg: float, V: float, rho: float = 1.225, clean: bool = True) -> Dict[str, float]:
        """Lift and drag at a given angle and tail incidence (what a CFD run of the clean airframe measures):
        induced + section drag strip by strip + fuselage and fin friction; gear/struts/misc left out when clean."""
        a, it = math.radians(alpha_deg), math.radians(it_deg)
        u = np.array([1.0, a, it])
        cls = self.sol.cl_strip.T @ u
        clw, clt = cls[self.w_strips], cls[self.t_strips]
        cdw = np.array([self.wp.at_cl(c)["cd"] for c in clw])
        cdt = np.interp(clt, self.tp._cl, self.tp._cd)
        CL = float(self.sol.CL @ u)
        CD = (float(u @ self.sol.Q @ u) + float((cdw * self.area_w).sum() + (cdt * self.area_t).sum()) / self.af.wing_area
              + fuselage_cd0(self.af, V, rho, wet=self._wet) + vtail_cd0(self.af, V, rho, vt=self._vt)
              + (0.0 if clean else MISC_D_OVER_Q / self.af.wing_area))
        return {"CL": CL, "CD": CD, "CDi": float(u @ self.sol.Q @ u)}

    def stall(self) -> Dict[str, float]:
        """Critical section: the trimmed CL at which the first wing strip reaches the section cl max."""
        p, r = self._trim
        a_ = self.sol.cl_strip.T @ np.array([1.0, p[0], p[1]])
        b_ = self.sol.cl_strip.T @ np.array([0.0, r[0], r[1]])
        aw, bw = a_[self.w_strips], b_[self.w_strips]
        clmax = self.wp.clmax
        with np.errstate(divide="ignore"):
            CLs = np.where(bw > 1e-6, (clmax - aw) / bw, np.inf)
        k = int(np.argmin(CLs))
        ratio = (aw + bw * CLs[k]) / clmax                           # cl / cl max along the span at stall onset
        return {"CLmax": float(CLs[k]), "station": float(self.eta[k]), "ratio": ratio}

    # ------------------------------------------------------------------ the aircraft numbers
    def evaluate(self) -> Dict[str, Any]:
        ac, req, af = self.ac, self.req, self.af
        rho_c = isa_density(ac.cruise_altitude)
        cr = self.drag(ac.cruise_speed, rho_c)
        fuel = cr["D"] * 1e5 / ac.prop_efficiency * ac.bsfc / 3.6e6
        sigma = rho_c / 1.225
        P = ac.power_kw * 1e3 * (1.132 * sigma - 0.132)
        st = self.stall()
        W = self.mass * G0
        v_stall = math.sqrt(2 * W / (1.225 * af.wing_area * st["CLmax"])) if st["CLmax"] > 0 else float("nan")
        v_stall_alt = math.sqrt(2 * W / (rho_c * af.wing_area * st["CLmax"])) if st["CLmax"] > 0 else 30.0
        lo, hi = v_stall_alt * 1.1, 160.0
        if self.drag(lo, rho_c)["D"] * lo > ac.prop_efficiency * P:
            vmax = float("nan")
        else:
            for _ in range(40):
                m = 0.5 * (lo + hi)
                if self.drag(m, rho_c)["D"] * m > ac.prop_efficiency * P:
                    hi = m
                else:
                    lo = m
            vmax = 0.5 * (lo + hi)
        cruise_ratio = float(np.max(cr["cl_wing"]) / self.wp.clmax)
        checks = [
            {"name": "Stall speed", "value": v_stall, "limit": req.max_stall_speed, "ok": v_stall <= req.max_stall_speed,
             "unit": "m/s", "kind": "max", "why": "CS-23 limit for a single-engine aircraft (61 kt), clean wing"},
            {"name": "Stall starts inboard", "value": st["station"], "limit": req.max_stall_station,
             "ok": st["station"] <= req.max_stall_station, "unit": "η", "kind": "max",
             "why": "if the tips stall first the ailerons stop working and the aircraft can drop a wing (spin)"},
            {"name": "Static margin", "value": self.static_margin, "limit": req.min_static_margin,
             "ok": req.min_static_margin <= self.static_margin <= req.max_static_margin, "unit": "MAC", "kind": "range",
             "limit_hi": req.max_static_margin, "why": "stable in pitch without being too heavy to rotate and flare"},
            {"name": "Trim in cruise (tail incidence)", "value": abs(cr["it"]), "limit": req.max_tail_incidence,
             "ok": abs(cr["it"]) <= req.max_tail_incidence, "unit": "deg", "kind": "max",
             "why": "the tail must balance the wing with elevator to spare"},
            {"name": "Thickness (spar depth, fuel)", "value": self.props["thickness"], "limit": req.min_thickness,
             "ok": self.props["thickness"] >= req.min_thickness, "unit": "", "kind": "min", "why": "structure and fuel need depth"},
            {"name": "Span (hangar, taxiway)", "value": af.span, "limit": req.max_span, "ok": af.span <= req.max_span,
             "unit": "m", "kind": "max", "why": "fits a standard T-hangar and taxiway"},
            {"name": "Cruise margin to stall (worst section)", "value": cruise_ratio, "limit": req.max_cruise_cl_ratio,
             "ok": cruise_ratio <= req.max_cruise_cl_ratio, "unit": "", "kind": "max", "why": "room for gusts and turns"},
        ]
        pen = 0.0
        for c in checks:
            if not c["ok"]:
                if c["kind"] == "range":
                    over = max(c["limit"] - c["value"], c["value"] - c["limit_hi"]) / 0.05
                elif c["kind"] == "min":
                    over = (c["limit"] - c["value"]) / abs(c["limit"])
                else:
                    over = (c["value"] - c["limit"]) / abs(c["limit"])
                pen += max(over, 0)
        ok_v = vmax == vmax and v_stall == v_stall
        e = cr["CL"] ** 2 / (math.pi * af.aspect_ratio * cr["CDi"]) if cr["CDi"] > 0 else float("nan")
        return {"vmax": vmax, "vmax_kt": vmax / KT, "co2_100km": fuel * CO2_PER_KG_AVGAS, "fuel_100km": fuel,
                "fuel_l_100km": fuel / 0.72, "v_stall": v_stall, "v_stall_kt": v_stall / KT, "clmax_wing": st["CLmax"],
                "stall_station": st["station"], "static_margin": self.static_margin, "x_cg": self.x_cg, "x_np": self.x_np,
                "cruise_cl": cr["CL"], "cruise_cd": cr["CD"], "cruise_ld": cr["CL"] / cr["CD"], "cruise_alpha": cr["alpha"],
                "tail_incidence": cr["it"], "oswald": e,
                "drag_breakdown": {"induced": cr["CDi"], "wing_profile": cr["CDp_wing"], "tail_profile": cr["CDp_tail"],
                                   "fuselage_vtail_misc": cr["CD0_other"]},
                "mass": self.mass, "weights": self.weights, "span": af.span, "checks": checks,
                "feasible": all(c["ok"] for c in checks) and ok_v, "violations": [c["name"] for c in checks if not c["ok"]],
                "penalty": pen if ok_v else pen + 10, "extrapolated": cr["extrapolated"],
                "loading": {"eta": self.eta.tolist(), "cl": cr["cl_wing"].tolist(), "stall_ratio": st["ratio"].tolist(),
                            "clc": (cr["cl_wing"] * self.sol.L.chord[self.w_strips] / af.mac).tolist()}}
