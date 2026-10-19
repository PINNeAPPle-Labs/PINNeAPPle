"""Four aircraft for flight simulation, built from the library's parametric geometry and weight / drag models.

=====================  ============================================================================================
``light_ga``           Cessna 172-class single piston (the aero optimizer's baseline ``Airframe`` and ``Aircraft``)
``regional_turboprop`` ATR 72-600-class high-wing twin turboprop (planform and gear from the ``turboprop_atr72``
                       geometry-library entry, its mesh for the visuals)
``a320``               A320-class single-aisle twin turbofan (the aero optimizer's baseline ``Airliner``)
``motor_glider``       high-aspect-ratio self-launching motor glider (Stemme S10-class span, area and mass)
=====================  ============================================================================================

Every number is either computed by the library from the geometry (spans, chords, tail sizes, lattice, wing / tail
weights by the Raymer equations the aero optimizer uses, drag build-up) or listed in ``Aircraft.info["published"]``
with its source, or in ``Aircraft.info["assumptions"]`` as an assumption. Nothing here is fitted to a flight test.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pinneapple_design.aero.aircraft3d import (
    MISC_D_OVER_Q,
    fuselage_cd0,
    htail_weight,
    vtail_cd0,
    vtail_weight,
    wing_weight,
)
from pinneapple_design.aero.airframe import Airframe
from pinneapple_design.aero.airliner import Airliner
from pinneapple_design.aero.airliner3d import MISC_CD0, TAIL_KG_M2, isa, wing_weight_transport
from pinneapple_design.aero.geometry import REFERENCE, properties

from .aero import AeroModel, ControlSurface, Fin, fuselage_moments
from .aircraft import Aircraft
from .gear import GearLeg, size_gear
from .mass import balance, cylinder, fin_plate, mass_properties, plate, point, rod
from .propulsion import Propeller, Turbofan

KT, FT, LB, HP = 0.514444, 0.3048, 0.45359237, 745.7

SOURCES = {
    "a320_tofl": ("A320-200, MTOW 78 t, sea level ISA: take-off field length about 2,090-2,100 m (certified, one engine"
                  " inoperative and 1.15 x all-engines distance, whichever is longer); e.g. "
                  "https://www.skytamer.com/Airbus_A320.html, https://aircraftinvestigation.info/airplanes/"
                  "Airbus_A320-200_CFM56.html (which also quotes about 1,825 m at 73.5 t); secondary sources. The primary "
                  "source, Airbus' A320 Aircraft Characteristics - Airport and Maintenance Planning, was not checked "
                  "(it could not be opened from the environment of this study)."),
    "cfm56_5b4": ("CFM56-5B4 take-off rating 27,000 lbf (120 kN): CFM International; EASA TCDS E.003 (CFM56-5B/-5C)."),
    "atr72_600": ("ATR 72-600: MTOW 23,000 kg (option; basic 22,800 kg), take-off field length 1,315 m at MTOW, ISA, "
                  "sea level; PW127M normal take-off power 2,475 shp, one-engine (reserve) take-off power 2,750 shp. "
                  "ATR, ATR 72-600 brochure (2023) and factsheet (2020), https://www.atr-aircraft.com/"),
    "stemme_s10": ("Stemme S10: span 23 m, wing area 18.7 m^2, loaded mass 850 kg (S10-VT: Rotax 914F, 84.5 kW). "
                   "https://en.wikipedia.org/wiki/Stemme_S10, https://planeandpilotmag.com/stemme-s10-vt/"),
    "c172": ("Cessna 172-class baseline of the aero optimizer (pinneapple_design.aero.aircraft.Aircraft: 1100 kg, "
             "120 kW, 16.2 m^2), calibrated there to a C172N (CD0 ~0.030, ~124 kt)."),
}


def profile_cd0(re: float, tc: float) -> float:
    """Turbulent flat-plate friction (0.074 Re^-0.2) x thickness form factor (1 + 2 t/c + 60 (t/c)^4), both sides,
    on the planform area (the glider-wing template's model, :mod:`pinneapple_design.intent.templates.glider_wing`)."""
    from pinneapple_design.intent.templates.glider_wing import profile_cd0 as f
    return float(f(re, tc))


def _fuselage_volume(af) -> float:
    xi = np.linspace(0, 1, 200)
    return float(np.trapezoid(math.pi / 4 * af._fus_w(xi) * af._fus_h(xi), xi * af.length))


def _wing_ac_cg(af):
    return af.mac_le_x + 0.25 * af.mac


@dataclass
class ExplicitAirframe(Airframe):
    """An :class:`Airframe` whose wing height and tails are given explicitly (for a high wing on a T-tail aircraft,
    whose tail the parametric sizing rules would put on the fuselage)."""
    wing_z: float = 0.0
    htail_spec: dict | None = None
    vtail_spec: dict | None = None

    def wing_z_root(self) -> float:
        return self.wing_z

    def htail(self):
        return dict(self.htail_spec)

    def vtail(self):
        return dict(self.vtail_spec)


def _assemble(name, af, mass_components, x_cg_target, engines, legs, controls, fin, cd0, cd_gear, cd_flap_per_rad,
              cabin, tail_point, clmax_to, clmax_clean, flap_to_deg, info, parts_fn: Callable | None = None,
              nc=4, ns_wing=14, ns_tail=6) -> Aircraft:
    comps = balance(mass_components, x_cg_target, "fuselage_systems")
    m, cg, J = mass_properties(comps)
    wing_qc = (af.wing_x + 0.25 * af.root_chord) / af.length
    cma, cnb = fuselage_moments(af.length, af.width, _fuselage_volume(af), af.wing_area, af.mac, af.span, af.height,
                                wing_qc)
    aero = AeroModel(af, cg, cd0, fin, controls, cma_fus=cma, cnb_fus=cnb, cd_gear=cd_gear,
                     cd_flap_per_rad=cd_flap_per_rad, nc=nc, ns_wing=ns_wing, ns_tail=ns_tail)
    legs = size_gear(legs, m, cg)
    info = dict(info)
    info["components"] = {c.name: {"mass": c.mass, "x": float(c.center[0]), "y": float(c.center[1]),
                                   "z": float(c.center[2])} for c in comps}
    info["cg_mac"] = float((cg[0] - af.mac_le_x) / af.mac)
    info["fuselage_cma"], info["fuselage_cnb"] = cma, cnb
    tip = np.array([af.wing_x + af.span / 2 * math.tan(math.radians(af.sweep_le)) + 0.5 * af.root_chord * af.taper,
                    af.span / 2, af.wing_z_root() + af.span / 2 * math.tan(math.radians(af.dihedral))])
    ac = Aircraft(name, af, aero, m, cg, J, engines, legs, cabin, tail_point=tail_point, wingtip=tip,
                  clmax_takeoff=clmax_to, clmax_clean=clmax_clean, flap_takeoff=math.radians(flap_to_deg), info=info)
    ac.parts_fn = parts_fn
    return ac


def _tipback_main_x(x_cg: float, comps, ground_z: float, tipback_deg: float = 15.0) -> float:
    """Main-gear station for a tip-back angle (CG to main-wheel contact, from the vertical) of ``tipback_deg``, the
    usual tricycle-gear layout rule (an assumption, used where the parametric airframe has no gear drawing data)."""
    m = sum(c.mass for c in comps)
    z_cg = sum(c.mass * float(c.center[2]) for c in comps) / m
    return x_cg + (z_cg - ground_z) * math.tan(math.radians(tipback_deg))


def _standard_controls(flap_eta=(0.05, 0.60), flap_cf=0.30, ail_eta=(0.62, 0.95), ail_cf=0.25, elev_cf=0.35):
    return {"elevator": ControlSurface("elevator", 1, 0.0, 1.01, elev_cf),
            "aileron": ControlSurface("aileron", 0, ail_eta[0], ail_eta[1], ail_cf, antisymmetric=True),
            "flap": ControlSurface("flap", 0, flap_eta[0], flap_eta[1], flap_cf)}


def _flap_drag(cf_c, flapped_fraction, delta_deg, F=0.0074):
    """Flap profile-drag increment F (cf/c)(S_flapped/S)(delta - 10 deg) (slotted flap, F = 0.0074; an assumption
    of the usual handbook form) -> per radian of deflection at that deflection."""
    d = F * cf_c * flapped_fraction * max(delta_deg - 10.0, 0.0)
    return d / max(math.radians(delta_deg), 1e-9)


def _fin_from(af, vt) -> Fin:
    c_mean = vt["root_chord"] * (1 + vt["taper"]) / 2
    # aerodynamic centre: quarter chord of the mean chord at ~40 % of the height (trapezoid)
    eta = (1 + 2 * vt["taper"]) / (3 * (1 + vt["taper"]))
    sweep = getattr(af, "vtail_sweep", 35.0)
    x_ac = vt["x_le"] + eta * vt["height"] * math.tan(math.radians(sweep)) + 0.25 * c_mean
    return Fin(vt["area"], vt["height"], np.array([x_ac, 0.0, vt["z"] + eta * vt["height"]]), sweep_deg=sweep)


# ================================================================================================= light GA
def light_ga() -> Aircraft:
    af = Airframe()
    m_to = 1100.0
    rho, _, mu = isa(0.0)
    V = 55.0
    q = 0.5 * isa(2000.0)[0] * V * V
    tc = properties(af.section)["thickness"]
    w_wing, w_ht, w_vt = wing_weight(af, m_to, q, tc), htail_weight(af, m_to, q), vtail_weight(af, m_to, q)
    ht, vt = af.htail(), af.vtail()
    zc = float(af._fus_zc(0.3))
    comps = [plate("wing", w_wing, [af.mac_le_x + 0.40 * af.mac, 0, af.wing_z_root()], af.span, af.mac, af.taper),
             plate("htail", w_ht, [ht["x_qc"], 0, ht["z"]], ht["span"], ht["root_chord"]),
             fin_plate("vtail", w_vt, [vt["x_le"] + 0.4 * vt["root_chord"], 0, vt["z"] + 0.4 * vt["height"]],
                       vt["height"], vt["root_chord"]),
             point("engine", 140.0, [0.45, 0, zc]),
             plate("fuel", 100.0, [af.mac_le_x + 0.35 * af.mac, 0, af.wing_z_root()], 0.6 * af.span, 0.5 * af.mac),
             point("occupants_front", 160.0, [2.45, 0, zc - 0.15]),
             point("occupants_rear", 80.0, [3.35, 0, zc - 0.10]),
             ]
    rest = m_to - sum(c.mass for c in comps)
    comps.append(cylinder("fuselage_systems", rest, [2.6, 0, zc], 0.75 * af.length, 0.45, shell=True))
    x_cg = _wing_ac_cg(af)
    gz = af.ground_z()
    xm = _tipback_main_x(x_cg, comps, gz)
    legs = [GearLeg("nose", np.array([0.75, 0.0, gz]), steerable=True),
            GearLeg("main_left", np.array([xm, -1.25, gz]), brakes=True),
            GearLeg("main_right", np.array([xm, 1.25, gz]), brakes=True)]
    cd0 = (fuselage_cd0(af, V, rho, mu) + vtail_cd0(af, V, rho, mu) + MISC_D_OVER_Q / af.wing_area
           + profile_cd0(rho * V * af.mac / mu, tc) + profile_cd0(rho * V * ht["root_chord"] / mu, 0.10) * ht["area"]
           / af.wing_area)
    eng = Propeller(120e3, af.prop_diameter, kind="piston", position=np.array([-0.2, 0.0, float(af._fus_zc(0.0))]))
    info = {"published": {"reference": SOURCES["c172"]},
            "assumptions": ["take-off mass 1100 kg and 120 kW (the aero optimizer's C172-class defaults)",
                            "CG at 25 % MAC", "flaps 10 deg for take-off on 5-60 % of the half-span, ailerons "
                            "62-95 %, elevator 35 % chord, rudder 30 % chord", "CLmax 1.6 with take-off flaps",
                            "engine 140 kg, fuel 100 kg, 240 kg of occupants; the fuselage and systems take the rest",
                            "fixed gear: its drag is in the calibrated miscellaneous D/q", "main gear placed for a 15 deg tip-back angle (the drawn wheel pants stay where the parametric airframe puts them)"],
            "type": "Cessna 172-class light single piston"}
    return _assemble("light_ga", af, comps, x_cg, [eng], legs, _standard_controls(elev_cf=0.40), _fin_from(af, vt),
                     cd0, 0.0, _flap_drag(0.30, 0.55, 10.0), {"pilot": np.array([2.45, -0.3, zc]),
                                                                 "rear_seat": np.array([3.35, 0.3, zc])},
                     np.array([af.length * 0.92, 0, float(af._fus_zc(0.92) - 0.5 * af._fus_h(0.92))]), 1.6, 1.4,
                     10.0, info, parts_fn=lambda: af.build("high"))


# ================================================================================================= ATR 72 class
def regional_turboprop() -> Aircraft:
    span, S = 27.05, 61.0
    ar = span * span / S
    ht = {"area": 11.7, "span": 7.31, "taper": 0.60, "x_qc": 25.2, "x_le": 24.5, "z": 7.5, "sweep_le": 8.0,
          "dihedral": 0.0}
    ht["root_chord"] = 2 * ht["area"] / (ht["span"] * (1 + ht["taper"]))
    ht["x_le"] = ht["x_qc"] - 0.25 * ht["root_chord"]
    vt = {"area": 12.5, "height": 4.5, "taper": 0.55, "x_le": 19.6, "z": 2.95}
    vt["root_chord"] = 2 * vt["area"] / (vt["height"] * (1 + vt["taper"]))
    af = ExplicitAirframe(length=27.17, width=2.77, height=2.62, wing_area=S, aspect_ratio=ar, taper=0.60,
                          sweep_le=3.1, dihedral=2.0, twist=-3.0, incidence=2.0, wing_x=11.0, high_wing=True,
                          section=REFERENCE["NACA 4415"].copy(), prop_diameter=3.93, gear=True, wing_z=3.80,
                          htail_spec=ht, vtail_spec=vt)
    af.vtail_sweep = 35.0
    ht["arm"] = ht["x_qc"] - _wing_ac_cg(af)
    m_to = 23000.0
    rho, _, mu = isa(0.0)
    V = 60.0
    tc = properties(af.section)["thickness"]
    w_wing = wing_weight_transport(af, m_to, tc, nz=3.8)
    comps = [plate("wing", w_wing, [af.mac_le_x + 0.40 * af.mac, 0, af.wing_z], af.span, af.mac, af.taper),
             plate("htail", TAIL_KG_M2[0] * ht["area"], [ht["x_qc"], 0, ht["z"]], ht["span"], ht["root_chord"]),
             fin_plate("vtail", TAIL_KG_M2[1] * vt["area"], [vt["x_le"] + 2.0, 0, vt["z"] + 1.8], vt["height"],
                       vt["root_chord"]),
             point("engine_right", 1100.0, [12.0, 4.05, 3.40]), point("engine_left", 1100.0, [12.0, -4.05, 3.40]),
             plate("fuel", 2500.0, [af.mac_le_x + 0.40 * af.mac, 0, af.wing_z], 0.70 * af.span, 0.5 * af.mac),
             rod("payload", 6800.0, [11.3, 0, 2.2], 10.5)]
    rest = m_to - sum(c.mass for c in comps)
    comps.append(cylinder("fuselage_systems", rest, [12.0, 0, 2.3], 22.0, 1.35, shell=True))
    x_cg = _wing_ac_cg(af)
    legs = [GearLeg("nose", np.array([2.78, 0.0, 0.0]), steerable=True),
            GearLeg("main_left", np.array([13.55, -2.05, 0.0]), brakes=True),
            GearLeg("main_right", np.array([13.55, 2.05, 0.0]), brakes=True)]
    cd0 = (fuselage_cd0(af, V, rho, mu) + vtail_cd0(af, V, rho, mu) + profile_cd0(rho * V * af.mac / mu, tc)
           + profile_cd0(rho * V * ht["root_chord"] / mu, 0.10) * ht["area"] / S
           + 2 * 0.0042 * math.pi * 1.0 * 7.2 * 1.3 / S + 0.0030)
    engs = [Propeller(2475 * HP, 3.93, efficiency=0.80, kind="turboprop", position=np.array([8.3, s * 4.05, 3.40]),
                      name=f"engine_{'right' if s > 0 else 'left'}") for s in (1, -1)]

    def parts():
        import pinneapple as pp
        from pinneapple_tools.visualization.studio.scene import MATERIALS
        g = pp.geometries.build("turboprop_atr72")
        for k, v in g.spec.material_table().items():                 # the entry's livery, de-icing boots, tyres
            MATERIALS.setdefault(k, v)
        return g.bodies

    info = {"published": {"atr72_600": SOURCES["atr72_600"]},
            "assumptions": ["take-off mass 23,000 kg, normal take-off power 2 x 2,475 shp (published, see sources)",
                            "planform: the turboprop_atr72 geometry entry (span, area, 3.1 deg LE sweep, 2 deg "
                            "dihedral) with one taper 0.60 for the lattice; horizontal tail 11.7 m^2 at the fin top "
                            "(T-tail), fin 12.5 m^2 (estimates)", "CG at 25 % MAC",
                            "flaps 15 deg on 5-70 % of the half-span (ATR settings 0 / 15 / 30), CLmax 2.1 with "
                            "take-off flaps", "propeller efficiency 0.80, static figure of merit 0.70, turboprop "
                            "power lapse sigma^0.75",
                            "masses: engines and nacelles 1,100 kg each, fuel 2,500 kg, payload 6,800 kg; the "
                            "fuselage and systems take the rest; wing by the Raymer transport equation",
                            "nacelle friction drag as cylinders 1.0 m dia x 7.2 m, miscellaneous CD0 0.0030, gear "
                            "drag 0.015"],
            "type": "ATR 72-600-class regional twin turboprop"}
    return _assemble("regional_turboprop", af, comps, x_cg, engs, legs,
                     _standard_controls(flap_eta=(0.05, 0.70), ail_eta=(0.72, 0.97)), _fin_from(af, vt), cd0, 0.015,
                     _flap_drag(0.30, 0.65, 15.0),
                     {"front_row": np.array([6.4, -0.9, 2.1]), "mid_cabin": np.array([11.3, -0.9, 2.1]),
                      "aft_row": np.array([16.3, -0.9, 2.1])}, np.array([22.5, 0.0, 1.45]), 2.1, 1.6, 15.0, info,
                     parts_fn=parts)


# ================================================================================================= A320 class
def a320() -> Aircraft:
    af = Airliner()
    m_to = 73_500.0
    rho, _, mu = isa(0.0)
    V = 75.0
    tc = properties(af.section)["thickness"]
    ht, vt = af.htail(), af.vtail()
    w_wing = wing_weight_transport(af, m_to, tc)
    wp = af._wing_point(af.engine_station, 0.0)
    R = af.fan_diameter / 2 * 1.08
    eng_pos = [np.array([wp[0] - 0.55 * 4.3 + 1.5, s * wp[1], wp[2] - 0.62 * R * 2 + 0.25]) for s in (1, -1)]
    zf = float(af._fus_zc(0.4))
    comps = [plate("wing", w_wing, [af.mac_le_x + 0.40 * af.mac, 0, af.wing_z_root()], af.span, af.mac, af.taper),
             plate("htail", TAIL_KG_M2[0] * ht["area"], [ht["x_qc"], 0, ht["z"]], ht["span"], ht["root_chord"]),
             fin_plate("vtail", TAIL_KG_M2[1] * vt["area"], [vt["x_le"] + 0.4 * vt["root_chord"], 0,
                                                             vt["z"] + 0.4 * vt["height"]], vt["height"],
                       vt["root_chord"]),
             point("engine_right", 3000.0, eng_pos[0]), point("engine_left", 3000.0, eng_pos[1]),
             plate("fuel", 12_000.0, [af.mac_le_x + 0.35 * af.mac, 0, af.wing_z_root()], 0.6 * af.span,
                   0.6 * af.mac),
             rod("payload", 17_100.0, [17.5, 0, zf - 0.3], 22.0)]
    rest = m_to - sum(c.mass for c in comps)
    comps.append(cylinder("fuselage_systems", rest, [17.0, 0, zf], 34.0, af.width / 2, shell=True))
    x_cg = af.mac_le_x + 0.25 * af.mac
    gz = af.ground_z()
    x_main = x_cg + 0.07 * 12.6 / 0.93
    legs = [GearLeg("nose", np.array([x_main - 12.6, 0.0, gz]), steerable=True),
            GearLeg("main_left", np.array([x_main, -3.8, gz]), brakes=True),
            GearLeg("main_right", np.array([x_main, 3.8, gz]), brakes=True)]
    # zero-lift drag: the aero optimizer's build-up (fuselage, nacelles, fin, misc) + wing and tail friction
    d = math.sqrt(af.width * af.height)
    f = af.length / d
    cdf = 0.455 / math.log10(rho * V * af.length / mu) ** 2.58 * (1 + 60 / f ** 3 + f / 400) * math.pi * d * \
        af.length * 0.80
    Dn, Ln = af.fan_diameter * 1.08, 4.3
    cdn = 2 * 0.455 / math.log10(rho * V * Ln / mu) ** 2.58 * (1 + 0.35 / (Ln / Dn)) * math.pi * Dn * Ln * 0.9 * 1.3
    cdv = vtail_cd0(af, V, rho, mu)
    cd0 = (cdf + cdn) / af.wing_area + cdv + MISC_CD0 + profile_cd0(rho * V * af.mac / mu, tc) \
        + profile_cd0(rho * V * ht["root_chord"] / mu, 0.10) * ht["area"] / af.wing_area
    engs = [Turbofan(27_000 * LB * 9.80665, position=p, name=f"engine_{n}") for p, n in zip(eng_pos, ("right", "left"), strict=True)]
    xi_t = 0.88
    info = {"published": {"cfm56_5b4": SOURCES["cfm56_5b4"], "a320_tofl": SOURCES["a320_tofl"]},
            "assumptions": ["take-off mass 73,500 kg (the aero optimizer's A320-class MTOW, below the A320ceo's "
                            "78 t), two 27,000 lbf turbofans (CFM56-5B4 rating)", "CG at 25 % MAC",
                            "CONF 1+F modelled as 10 deg of flap on 5-75 % of the half-span (slats change CLmax, "
                            "not modelled), CLmax 2.0 at take-off, gear drag 0.015",
                            "wheelbase 12.6 m and track 7.6 m chosen so the nose gear carries ~7 % of the weight; "
                            "geometry, not a published drawing", "masses: engines 3,000 kg each, fuel 12,000 kg, "
                            "payload 17,100 kg (180 x 95 kg); fuselage and systems take the rest"],
            "type": "A320-class single-aisle twin turbofan"}
    return _assemble("a320", af, comps, x_cg, engs, legs, _standard_controls(flap_eta=(0.05, 0.75),
                                                                             ail_eta=(0.75, 0.95)),
                     _fin_from(af, vt), cd0, 0.015, _flap_drag(0.30, 0.70, 10.0),
                     {"front_row": np.array([7.5, -1.5, zf - 0.3]), "mid_cabin": np.array([17.0, -1.5, zf - 0.3]),
                      "aft_row": np.array([26.0, -1.5, zf - 0.3])},
                     np.array([xi_t * af.length, 0, float(af._fus_zc(xi_t) - 0.5 * af._fus_h(xi_t))]), 2.0, 1.45,
                     10.0, info, parts_fn=lambda: af.build("high"))


# ================================================================================================= motor glider
def motor_glider() -> Aircraft:
    af = Airframe(length=8.4, width=1.15, height=1.25, wing_area=18.7, aspect_ratio=23.0 ** 2 / 18.7, taper=0.40,
                  sweep_le=1.0, dihedral=3.0, twist=-2.0, incidence=2.0, wing_x=2.9, high_wing=False,
                  section=REFERENCE["NACA 4412"].copy(), htail_volume=0.45, vtail_volume=0.025, htail_ar=5.5,
                  tail_arm_frac=0.93, prop_diameter=1.6, gear=True)
    m_to = 850.0
    rho, _, mu = isa(0.0)
    V = 28.0
    q = 0.5 * rho * V * V
    tc = properties(af.section)["thickness"]
    ht, vt = af.htail(), af.vtail()
    zc = float(af._fus_zc(0.3))
    comps = [plate("wing", 0.85 * wing_weight(af, m_to, q, tc, nz=5.3, Wfw=60.0),
                   [af.mac_le_x + 0.40 * af.mac, 0, af.wing_z_root()], af.span, af.mac, af.taper),
             plate("htail", htail_weight(af, m_to, q), [ht["x_qc"], 0, ht["z"]], ht["span"], ht["root_chord"]),
             fin_plate("vtail", vtail_weight(af, m_to, q), [vt["x_le"] + 0.4 * vt["root_chord"], 0,
                                                             vt["z"] + 0.4 * vt["height"]], vt["height"],
                       vt["root_chord"]),
             point("engine", 75.0, [3.0, 0, zc]), point("occupants", 180.0, [1.9, 0, zc - 0.1]),
             plate("fuel", 60.0, [af.mac_le_x + 0.35 * af.mac, 0, af.wing_z_root()], 0.4 * af.span, 0.5 * af.mac)]
    rest = m_to - sum(c.mass for c in comps)
    comps.append(cylinder("fuselage_systems", rest, [3.2, 0, zc], 0.8 * af.length, 0.40, shell=True))
    x_cg = af.mac_le_x + 0.30 * af.mac
    gz = af.ground_z()
    xm = _tipback_main_x(x_cg, comps, gz)
    legs = [GearLeg("nose", np.array([0.8, 0.0, gz]), steerable=True),
            GearLeg("main_left", np.array([xm, -1.0, gz]), brakes=True),
            GearLeg("main_right", np.array([xm, 1.0, gz]), brakes=True)]
    cd0 = (fuselage_cd0(af, V, rho, mu) + vtail_cd0(af, V, rho, mu) + profile_cd0(rho * V * af.mac / mu, tc)
           + profile_cd0(rho * V * ht["root_chord"] / mu, 0.10) * ht["area"] / af.wing_area + 0.0015)
    eng = Propeller(84.5e3, 1.6, efficiency=0.75, kind="piston", position=np.array([-0.2, 0.0, float(af._fus_zc(0.0))]))
    info = {"published": {"stemme_s10": SOURCES["stemme_s10"]},
            "assumptions": ["span 23 m, area 18.7 m^2, mass 850 kg, 84.5 kW (Stemme S10 / S10-VT figures); fuselage, "
                            "tails, taper 0.40, NACA 4412 camber and the nose propeller are the parametric Airframe's, "
                            "not the S10's (which has a mid-fuselage engine and a taildragger gear)",
                            "tricycle gear for the take-off model, mains at a 15 deg tip-back angle", "CG at 30 % MAC", "flaps 10 deg on 5-50 %, "
                            "CLmax 1.45 at take-off", "wing mass: Raymer general-aviation equation x 0.85 (a "
                            "composite sailplane wing, assumption); miscellaneous CD0 0.0015, gear drag 0.008"],
            "type": "high-aspect-ratio self-launching motor glider (Stemme S10-class)"}
    return _assemble("motor_glider", af, comps, x_cg, [eng], legs,
                     _standard_controls(flap_eta=(0.05, 0.50), ail_eta=(0.55, 0.95), elev_cf=0.35),
                     _fin_from(af, vt), cd0, 0.008, _flap_drag(0.25, 0.45, 10.0),
                     {"pilot": np.array([1.9, 0.0, zc - 0.1])},
                     np.array([af.length * 0.92, 0, float(af._fus_zc(0.92) - 0.5 * af._fus_h(0.92))]), 1.45, 1.3,
                     10.0, info, parts_fn=lambda: af.build("high"), ns_wing=20)


CATALOG: dict[str, Callable[[], Aircraft]] = {"light_ga": light_ga, "regional_turboprop": regional_turboprop,
                                               "a320": a320, "motor_glider": motor_glider}


def build(name: str) -> Aircraft:
    return CATALOG[name]()


def names() -> list[str]:
    return list(CATALOG)
