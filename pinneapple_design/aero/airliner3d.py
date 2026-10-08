"""Airliner physics (A320 class): cruise at transonic Mach and the low-speed end, for the whole aircraft.

Cruise: vortex lattice with the Prandtl-Glauert stretch (x / beta) for compressibility, trimmed with the tail; section
drag strip by strip = friction at the real Reynolds number + the form-drag increment from the CFD section surrogate
(its drag above flat-plate friction at the training Reynolds number); wave drag per strip from the Korn equation with
Lock's rule (simple-sweep theory); fuselage, nacelles, fin by component build-up (Raymer). Mission: Breguet range with
a turbofan's specific fuel consumption, MTOW iterated with the wing weight (Raymer, transport). Low speed: wing CLmax
from the section cl max (critical section, vortex lattice) plus high-lift devices, approach speed Vref = 1.23 Vs.

Requirements: span (ICAO code C gate, 36 m), approach speed, 1.3 g buffet margin, fuel volume in the wing box, static
margin, stall starting inboard, trim, thickness. Objectives: CO2 per passenger-km, cruise Mach, approach speed.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .aircraft import KT, Polar
from .airliner import Airliner, sweep_le_from_qc
from .geometry import BOUNDS as SHAPE_BOUNDS, REFERENCE, properties
from .vlm import solve

G0 = 9.80665
LB, FT = 0.45359237, 0.3048
CO2_PER_KG_JETA1 = 3.16

PLAN = ("wing_area", "aspect_ratio", "taper", "sweep_qc", "twist", "cruise_mach", "wing_x")
PLAN_BOUNDS = np.array([[100.0, 150.0], [8.0, 12.5], [0.18, 0.40], [15.0, 35.0], [-6.0, -1.0], [0.70, 0.82], [11.0, 13.8]])
BOUNDS_AL = np.vstack([SHAPE_BOUNDS, PLAN_BOUNDS])
LABELS_AL = {"wing_area": "wing area (m²)", "aspect_ratio": "aspect ratio", "taper": "taper ratio",
             "sweep_qc": "quarter-chord sweep (°)", "twist": "tip twist (°)", "cruise_mach": "cruise Mach",
             "wing_x": "wing position (m from the nose)"}
BASELINE_PLAN = {"wing_area": 122.6, "aspect_ratio": 9.5, "taper": 0.24, "sweep_qc": 25.0, "twist": -4.0,
                 "cruise_mach": 0.78, "wing_x": 12.3}


def isa(h: float):
    T = 288.15 - 0.0065 * min(h, 11000) if h <= 11000 else 216.65
    p = 101325 * (T / 288.15) ** 5.2559 if h <= 11000 else 22632 * math.exp(-G0 * (h - 11000) / (287.05 * 216.65))
    rho = p / (287.05 * T)
    mu = 1.458e-6 * T ** 1.5 / (T + 110.4)
    return rho, math.sqrt(1.4 * 287.05 * T), mu


@dataclass
class Mission:
    passengers: int = 180
    mass_per_pax: float = 95.0          # kg incl. baggage
    range_km: float = 4800.0            # design range (~2600 nm)
    cruise_altitude: float = 10668.0    # m (FL350)
    tsfc: float = 1.61e-5               # kg/(N s) at Mach 0.78 cruise (~0.57 lb/lbf/h, CFM56 class), ~ M^0.5
    reserve: float = 0.08               # climb, descent and taxi allowance on the cruise fuel (burnt)
    oew_rest: float = 32_000.0          # kg: operating empty mass minus wing and tails (A320 class: OEW ~42 t)
    high_lift_dcl: float = 1.60         # landing CLmax increment of double-slotted / Fowler flaps on the flapped span
    flapped_fraction: float = 0.70
    slat_dcl: float = 0.45              # leading-edge slats, whole span
    landing_mass_frac: float = 0.86     # MLW / MTOW


@dataclass
class RequirementsAL:
    max_span: float = 36.0              # ICAO aerodrome reference code C (A320/737 gates)
    max_vref_kt: float = 145.0          # approach speed, category C
    buffet_margin: float = 1.30         # cruise CL x 1.3 below drag-divergence (1.3 g buffet margin)
    min_static_margin: float = 0.05
    max_static_margin: float = 0.35
    max_stall_station: float = 0.80     # clean wing; outboard slats protect the tips of real airliners
    max_tail_incidence: float = 4.0
    min_thickness: float = 0.09
    fuel_volume_margin: float = 1.0     # fuel volume in the wing box / mission fuel >= this

    def as_dict(self):
        return dataclasses.asdict(self)


def airliner_from(x: np.ndarray) -> Airliner:
    x = np.asarray(x, float)
    d = dict(zip(PLAN, x[6:13].tolist()))
    return Airliner(section=x[:6].copy(), wing_area=d["wing_area"], aspect_ratio=d["aspect_ratio"], taper=d["taper"],
                    sweep_le=sweep_le_from_qc(d["sweep_qc"], d["aspect_ratio"], d["taper"]), twist=d["twist"],
                    wing_x=d["wing_x"])


def baseline_x_al() -> np.ndarray:
    return np.r_[REFERENCE["NACA 2412"], [BASELINE_PLAN[k] for k in PLAN]]


def _cf(Re):
    return 0.455 / (math.log10(Re) ** 2.58)


def wing_weight_transport(af: Airliner, mtow: float, tc: float, nz: float = 3.75) -> float:
    S = af.wing_area / FT ** 2
    lam_c4 = math.atan(math.tan(math.radians(af.sweep_le)) - (1 - af.taper) / (af.aspect_ratio * (1 + af.taper)))
    w = 0.0051 * (mtow / LB * nz) ** 0.557 * S ** 0.649 * af.aspect_ratio ** 0.5 * tc ** -0.4 * (1 + af.taper) ** 0.1 \
        / math.cos(lam_c4) * (0.25 * S) ** 0.1
    return w * LB * WING_MASS_FACTOR


WING_MASS_FACTOR = 1.5                  # high-lift devices, spoilers, ailerons and systems carried in the wing:
                                        # calibrated so the A320-class wing is ~8.7 t (the real one ~8.8 t)
TAIL_KG_M2 = (28.0, 30.0)               # horizontal, vertical tail mass per area (transport)
MISC_CD0 = 0.0018                       # excrescences, interference, flap tracks, antennas: calibrated once so the
                                        # A320-class baseline has cruise L/D ~16.4 (MTOW 73.5 t, OEW 42.3 t)
KORN_KA = 0.91                          # Korn technology factor: 0.87 conventional ... 0.95 supercritical


class AirlinerModel:
    def __init__(self, x: np.ndarray, wing_polar: Polar, tail_polar: Polar, mission: Mission = Mission(),
                 req: RequirementsAL = RequirementsAL(), nc: int = 4, ns_wing: int = 12, ns_tail: int = 5):
        self.x, self.mi, self.req = np.asarray(x, float), mission, req
        self.af = af = airliner_from(x)
        self.mach = float(self.x[11])
        self.props = properties(self.x[:6])
        self.tc = self.props["thickness"]
        self.wp, self.tp = wing_polar, tail_polar
        rho, a, mu = isa(mission.cruise_altitude)
        self.rho, self.a, self.mu = rho, a, mu
        self.V = self.mach * a
        beta = math.sqrt(1 - self.mach ** 2)
        # CG from the balance: the wing's aerodynamic centre region (fixed for the fuselage/payload/engines, the wing
        # mass moves with the wing); the static margin then follows from where the wing is put
        x_rest = 16.9
        mw0 = 8800.0
        self.sol_lo = solve(af, x_rest, nc=nc, ns_wing=ns_wing, ns_tail=ns_tail)          # low speed
        self.sol = self._solve_comp(beta, x_rest, nc, ns_wing, ns_tail)                   # cruise
        # weights: iterate MTOW with the mission fuel
        payload = mission.passengers * mission.mass_per_pax
        ht, vt = af.htail(), af.vtail()
        self.tail_mass = TAIL_KG_M2[0] * ht["area"] + TAIL_KG_M2[1] * vt["area"]
        mtow = 73_500.0
        for _ in range(8):
            self.wing_mass = wing_weight_transport(af, mtow, self.tc)
            oew = mission.oew_rest + self.wing_mass + self.tail_mass
            W_mid = (oew + payload + 0.5 * self._fuel_guess(mtow)) * G0
            cr = self.cruise(W_mid)
            ld = cr["CL"] / cr["CD"]
            # Breguet: R = V/(g tsfc) L/D ln(W0/W1)
            tsfc = mission.tsfc * (self.mach / 0.78) ** 0.5            # high-bypass TSFC grows with Mach (Raymer)
            ratio = math.exp(mission.range_km * 1e3 * G0 * tsfc / (self.V * ld))
            zfw = oew + payload
            fuel = zfw * (ratio - 1) * (1 + mission.reserve)
            mtow_new = zfw + fuel
            self._fuel = fuel
            if abs(mtow_new - mtow) < 5:
                mtow = mtow_new
                break
            mtow = mtow_new
        self.mtow, self.oew, self.fuel, self.payload, self.cr, self.ld = mtow, oew, fuel, payload, cr, ld
        xw = af.mac_le_x + 0.40 * af.mac
        self.x_cg = x_rest + self.wing_mass * (xw - (16.0 + 0.40 * 4.04)) / mtow + 0.0 * mw0
        # neutral point (low speed, fuselage destabilising moment, Raymer K_f)
        kf = 0.025                                                          # Raymer K_fus, wing at ~45 % of the body
        dcma = kf * af.width ** 2 * af.length / (af.wing_area * af.mac) * 180 / math.pi
        CLa, Cma = self.sol_lo.CL[1], self.sol_lo.Cm[1] + dcma - self.sol_lo.CL[1] * (self.x_cg - x_rest) / af.mac
        self.dcma_fus = dcma
        self.x_np = self.x_cg - Cma / CLa * af.mac
        self.static_margin = (self.x_np - self.x_cg) / af.mac

    def _fuel_guess(self, mtow):
        return getattr(self, "_fuel", 0.25 * mtow)

    def _solve_comp(self, beta, x_ref, nc, ns_wing, ns_tail):
        """Prandtl-Glauert: solve the incompressible problem on the geometry stretched by 1/beta in x, scale by 1/beta."""
        af = self.af
        st = dataclasses.replace(af, wing_x=af.wing_x / beta, length=af.length / beta)
        # stretch: sweep tangents and chords grow by 1/beta (chords via the area and AR at fixed span)
        b = af.span
        st = dataclasses.replace(st, sweep_le=math.degrees(math.atan(math.tan(math.radians(af.sweep_le)) / beta)),
                                 wing_area=af.wing_area / beta, aspect_ratio=b * b / (af.wing_area / beta))
        sol = solve(st, x_ref / beta, nc=nc, ns_wing=ns_wing, ns_tail=ns_tail)
        # coefficients referred to the real area; lift and loading scale with 1/beta (PG), drag form unchanged
        k = (st.wing_area / af.wing_area) / beta
        sol.CL = sol.CL * k
        sol.Cm = sol.Cm * k
        sol.cl_strip = sol.cl_strip / beta
        sol.Q = sol.Q * (st.wing_area / af.wing_area) / beta ** 2
        sol.L.chord = sol.L.chord * beta
        return sol

    # ------------------------------------------------------------------ cruise
    def _trim(self, sol, CL):
        A = np.array([[sol.CL[1], sol.CL[2]], [sol.Cm[1], sol.Cm[2]]])
        a, it = np.linalg.solve(A, np.array([CL - sol.CL[0], -sol.Cm[0]]))
        return a, it

    def strips(self, sol):
        L = sol.L
        w = L.strip_surf == 0
        return w, (L.chord * np.abs(L.dy))[w], (L.chord * np.abs(L.dy))[L.strip_surf == 1]

    def wave_drag(self, cl_local: np.ndarray, M: float) -> np.ndarray:
        lam = math.radians(self.af.sweep_le) * 0.0 + math.atan(math.tan(math.radians(self.af.sweep_le))
                                                               - 2 * (1 - self.af.taper) / (self.af.aspect_ratio * (1 + self.af.taper)))
        c = math.cos(lam)                                                  # mid-chord sweep
        mdd = KORN_KA / c - self.tc / c ** 2 - np.maximum(cl_local, 0) / (10 * c ** 3)
        mcrit = mdd - (0.1 / 80) ** (1 / 3)
        return np.where(M > mcrit, 20 * (M - mcrit) ** 4, 0.0), mdd

    def cruise(self, W: float, M: Optional[float] = None) -> Dict[str, Any]:
        M = M or self.mach
        V = M * self.a
        q = 0.5 * self.rho * V * V
        af = self.af
        CL = W / (q * af.wing_area)
        a, it = self._trim(self.sol, CL)
        u = np.array([1.0, a, it])
        cls = self.sol.cl_strip.T @ u
        w, aw, at = self.strips(self.sol)
        clw, clt = cls[w], cls[~w]
        cdi = float(u @ self.sol.Q @ u)
        # section drag = friction at the real Re (both sides) * compressibility + form drag from the CFD surrogate
        chords = self.sol.L.chord[w]
        Re = self.rho * V * chords / self.mu
        cf2 = np.array([2 * _cf(r) for r in Re]) * (1 + 0.2 * M * M) ** -0.467
        lam = math.atan(math.tan(math.radians(af.sweep_le)) - (1 - af.taper) / (af.aspect_ratio * (1 + af.taper)))
        cn = clw / math.cos(lam) ** 2                                      # simple sweep: section sees cl / cos^2
        cd_sur = np.interp(cn, self.wp._cl, self.wp._cd, left=self.wp._cd[0], right=self.wp._cd[-1])
        form = np.maximum(cd_sur - 2 * _cf(4e6), 0.0)
        cdp = cf2 * (1 + 2.0 * self.tc) + form * math.cos(lam) ** 3
        cdw, mdd = self.wave_drag(clw, M)
        cd_wing = float(((cdp + cdw) * aw).sum() / af.wing_area)
        cdwave = float((cdw * aw).sum() / af.wing_area)
        cdt = float((np.interp(clt, self.tp._cl, self.tp._cd) * at).sum() / af.wing_area) * 0.75  # tail at higher Re
        cd0 = self.parasite(V)
        CD = cdi + cd_wing + cdt + cd0
        return {"CL": CL, "CD": CD, "alpha": math.degrees(a), "it": math.degrees(it), "CDi": cdi,
                "CD_wing": cd_wing - cdwave, "CD_wave": cdwave, "CD_tail": cdt, "CD0_other": cd0, "cl": clw,
                "mdd_min": float(mdd.min()), "q": q, "V": V}

    def parasite(self, V: float) -> float:
        af = self.af
        rho, mu = self.rho, self.mu
        # fuselage
        d = math.sqrt(af.width * af.height)
        f = af.length / d
        swet_f = math.pi * d * af.length * 0.80
        cdf = _cf(rho * V * af.length / mu) * (1 + 60 / f ** 3 + f / 400) * swet_f
        # nacelles (2), fin
        Dn, Ln = af.fan_diameter * 1.08, 4.3
        cdn = 2 * _cf(rho * V * Ln / mu) * (1 + 0.35 / (Ln / Dn)) * math.pi * Dn * Ln * 0.9 * 1.3
        vt = af.vtail()
        cv = vt["root_chord"] * (1 + vt["taper"]) / 2
        cdv = _cf(rho * V * cv / mu) * (1 + 1.2 * 0.10 + 100 * 0.10 ** 4) * 1.2 * 2.04 * vt["area"]
        return (cdf + cdn + cdv) / af.wing_area + MISC_CD0

    # ------------------------------------------------------------------ low speed
    def stall(self) -> Dict[str, Any]:
        sol = self.sol_lo
        A = np.array([[sol.CL[1], sol.CL[2]], [sol.Cm[1] + self.dcma_fus, sol.Cm[2]]])
        Ai = np.linalg.inv(A)
        p = Ai @ np.array([-sol.CL[0], -sol.Cm[0]])
        r = Ai @ np.array([1.0, 0.0])
        w = sol.L.strip_surf == 0
        a_ = (sol.cl_strip.T @ np.array([1.0, p[0], p[1]]))[w]
        b_ = (sol.cl_strip.T @ np.array([0.0, r[0], r[1]]))[w]
        lam = math.atan(math.tan(math.radians(self.af.sweep_le)) - (1 - self.af.taper) / (self.af.aspect_ratio * (1 + self.af.taper)))
        clmax = self.wp.clmax * math.cos(lam) ** 1.0 * 0.95                      # swept-wing section limit, 3D factor
        CLs = np.where(b_ > 1e-6, (clmax - a_) / b_, np.inf)
        k = int(np.argmin(CLs))
        eta = np.abs(sol.L.yc[w]) / (self.af.span / 2)
        ratio = (a_ + b_ * CLs[k]) / clmax
        CLmax_clean = float(CLs[k])
        CLmax_land = CLmax_clean + (self.mi.high_lift_dcl * self.mi.flapped_fraction + self.mi.slat_dcl) * math.cos(lam)
        return {"CLmax_clean": CLmax_clean, "CLmax_land": CLmax_land, "station": float(eta[k]), "eta": eta, "ratio": ratio}

    def fuel_volume(self) -> float:
        """m3 of fuel the two wing boxes hold (front spar 15 %, rear spar 65 %, 85 % usable), root to 85 % span."""
        af = self.af
        y = np.linspace(0, 0.85, 40) * af.span / 2
        c = af.root_chord * (1 - (1 - af.taper) * y / (af.span / 2))
        x = np.linspace(0.15, 0.65, 30)
        from .geometry import surfaces
        yu, yl = surfaces(self.x[:6], x)
        box_area = np.trapezoid(yu - yl, x)                                       # per unit chord^2
        return float(2 * 0.85 * np.trapezoid(box_area * c ** 2, y))

    # ------------------------------------------------------------------ the aircraft numbers
    def evaluate(self) -> Dict[str, Any]:
        af, mi, req = self.af, self.mi, self.req
        cr = self.cr
        st = self.stall()
        W_land = self.mtow * mi.landing_mass_frac * G0
        vs = math.sqrt(2 * W_land / (1.225 * af.wing_area * st["CLmax_land"])) if st["CLmax_land"] > 0 else float("nan")
        vref = 1.23 * vs
        fuel_vol = self.fuel_volume()
        fuel_ratio = fuel_vol * 800 / self.fuel
        # buffet: 1.3 g -> the drag-divergence Mach at 1.3 x cruise lift must stay above the cruise Mach
        cl13 = cr["cl"] * req.buffet_margin
        _, mdd13 = self.wave_drag(cl13, self.mach)
        buffet_ok = float(mdd13.min()) >= self.mach - 0.005
        co2_pkm = self.fuel / mi.range_km / mi.passengers * CO2_PER_KG_JETA1 * 1000      # g CO2 per passenger-km
        checks = [
            {"name": "Span (ICAO code C gate)", "value": af.span, "limit": req.max_span, "ok": af.span <= req.max_span,
             "unit": "m", "kind": "max", "why": "fits the 36 m gates of A320 / 737 airports"},
            {"name": "Approach speed Vref", "value": vref / KT, "limit": req.max_vref_kt, "ok": vref / KT <= req.max_vref_kt,
             "unit": "kt", "kind": "max", "why": "approach category C, runway length, noise"},
            {"name": "Buffet margin (1.3 g)", "value": float(mdd13.min()), "limit": self.mach, "ok": buffet_ok,
             "unit": "M", "kind": "min", "why": "a 1.3 g gust or turn at cruise must not reach shock-induced buffet"},
            {"name": "Fuel fits in the wing", "value": fuel_ratio, "limit": req.fuel_volume_margin,
             "ok": fuel_ratio >= req.fuel_volume_margin, "unit": "x", "kind": "min", "why": "the design-range fuel has to fit in the wing boxes"},
            {"name": "Static margin", "value": self.static_margin, "limit": req.min_static_margin,
             "ok": req.min_static_margin <= self.static_margin <= req.max_static_margin, "unit": "MAC", "kind": "range",
             "limit_hi": req.max_static_margin, "why": "stable in pitch, still able to rotate at take-off"},
            {"name": "Stall starts inboard", "value": st["station"], "limit": req.max_stall_station,
             "ok": st["station"] <= req.max_stall_station, "unit": "η", "kind": "max",
             "why": "a swept wing that stalls at the tips pitches up and rolls off (clean wing; outboard slats help)"},
            {"name": "Trim in cruise (tail incidence)", "value": abs(cr["it"]), "limit": req.max_tail_incidence,
             "ok": abs(cr["it"]) <= req.max_tail_incidence, "unit": "deg", "kind": "max", "why": "trimmable stabiliser range"},
            {"name": "Thickness (structure)", "value": self.tc, "limit": req.min_thickness, "ok": self.tc >= req.min_thickness,
             "unit": "", "kind": "min", "why": "wing box depth for bending and fuel"},
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
                pen += max(over, 0) * (20 if c["unit"] == "M" else 1)
        e = cr["CL"] ** 2 / (math.pi * af.aspect_ratio * cr["CDi"]) if cr["CDi"] > 0 else float("nan")
        return {"co2_pkm": co2_pkm, "mach": self.mach, "speed_kt": cr["V"] / KT, "vref_kt": vref / KT,
                "fuel_kg": self.fuel, "fuel_l_100pkm": self.fuel / 0.8 / mi.range_km / mi.passengers * 100,
                "mtow": self.mtow, "oew": self.oew, "wing_mass": self.wing_mass, "tail_mass": self.tail_mass,
                "cruise_cl": cr["CL"], "cruise_cd": cr["CD"], "cruise_ld": self.ld, "cruise_alpha": cr["alpha"],
                "tail_incidence": cr["it"], "oswald": e, "mdd": cr["mdd_min"], "clmax_clean": st["CLmax_clean"],
                "clmax_land": st["CLmax_land"], "stall_station": st["station"], "static_margin": self.static_margin,
                "x_cg": self.x_cg, "x_np": self.x_np, "span": af.span, "fuel_volume_m3": fuel_vol, "fuel_ratio": fuel_ratio,
                "drag_breakdown": {"induced": cr["CDi"], "wing_profile": cr["CD_wing"], "wave": cr["CD_wave"],
                                   "tail": cr["CD_tail"], "fuselage_nacelles_fin_misc": cr["CD0_other"]},
                "checks": checks, "feasible": all(c["ok"] for c in checks), "violations": [c["name"] for c in checks if not c["ok"]],
                "penalty": pen, "loading": {"eta": st["eta"].tolist(), "stall_ratio": st["ratio"].tolist(), "cl": cr["cl"].tolist()},
                "extrapolated": False}
