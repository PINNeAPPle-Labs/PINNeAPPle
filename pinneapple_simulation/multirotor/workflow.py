"""The standard multirotor workflow (#1201): rotor -> motor and battery -> airframe and sizing -> 6-DOF control ->
the designed drone flown on a mission until its battery is empty.

1. **Rotor**: BEMT (:mod:`.rotor`) checked against momentum theory (an ideally twisted rotor without losses has
   exactly the ideal induced power) and against the published static ``c_T``, ``c_P`` of the APC Slow Flyer
   10 x 4.7 (UIUC database). The assumed polar is reported as it is; the polar is then calibrated on the even rows
   and its error measured on the odd (held-out) rows, and on the high-RPM half after a fit on the low half.
2. **Motor and battery** (:mod:`.power`): ``K_v`` / ``K_t`` with copper and no-load losses (the power split closes
   ``V I``), Peukert-corrected LiPo pack with voltage sag.
3. **Sizing**: propeller diameter (the APC geometry scaled), cells in series and capacity on a grid; for each, the
   motor ``K_v`` that reaches the thrust-to-weight target at 95 % of the loaded pack voltage, the arm length from the
   propeller clearance, and the mass build-up (:func:`build_quad`, stated coefficients in ``MASS_MODEL``); the
   lightest design whose estimated hover endurance meets the target.
4. **Control** (:mod:`.quad`): LQR on the hover linearisation with position integrators; checks: every closed-loop
   pole in the left half plane, per-channel gain and phase margins with the motor lag, a 1 m step (overshoot,
   settling) and a 3 m/s wind gust (peak excursion, recovery).
5. **Re-verification**: the drone flies hover -> climb -> cruise -> return legs repeatedly on the battery until it
   is empty; the endurance is the time the flight lasted (measured), compared with the sizing estimate and with the
   momentum-theory ideal hover power over the figure of merit.

**Published endurance**: :func:`published_endurance_check` builds a quad with the DJI Phantom 4's published take-off
mass and battery (``PHANTOM4``, each value with its source) on the scaled APC propeller (the Phantom's own propeller
geometry is not public here: an assumption), hovers it until the pack is empty and compares with the published
maximum flight time within ``PHANTOM4_BAND``.
"""
from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .power import Battery, Motor
from .quad import (
    G,
    Quad,
    Segment,
    fly_mission,
    hover_power,
    loop_margins_per_channel,
    lqr_hover,
    rotor_table,
    simulate,
)
from .rotor import (
    APC_10X4_7_STATIC,
    APC_SOURCE,
    Airfoil,
    apc_10x4_7,
    bemt,
    calibrate_airfoil,
    ideal_twist_check,
    momentum_hover_power,
)

__all__ = ["MASS_MODEL", "PHANTOM4", "PHANTOM4_BAND", "rotor_verification", "build_quad", "size_quad",
           "published_endurance_check", "drone_workflow"]

#: stated mass-model and drive coefficients (assumptions, not measurements)
MASS_MODEL = {"arm_kg_per_m": 0.04, "body_kg": 0.15, "esc_kg": 0.02, "prop_kg_10in": 0.012,
              "motor_W_per_kg": 4000.0, "motor_r_ohm_ref": 0.1, "motor_i0_ref": 0.5, "motor_kg_ref": 0.05,
              "avionics_kg": 0.10, "avionics_W": 5.0, "battery_Wh_per_kg": 150.0, "clearance": 0.1,
              "cd_area_m2": 0.02, "reserve_soc": 0.15, "peukert": 1.05, "r_cell_ohm_Ah": 0.015}

#: DJI Phantom 4 as published (secondary: DJI product page and spec sheet, read through search results)
PHANTOM4 = {"mass_kg": 1.380, "capacity_Ah": 5.35, "cells": 4, "energy_Wh": 81.3, "flight_time_min": 28.0,
            "prop_diameter_in": 9.45,
            "sources": {"mass_kg": "DJI Phantom 4 specs: 1380 g, battery and propellers included",
                        "capacity_Ah": "DJI Phantom 4 Intelligent Flight Battery: 5350 mAh, 15.2 V, 81.3 Wh",
                        "flight_time_min": "DJI Phantom 4 specs: max flight time approx. 28 minutes",
                        "prop_diameter_in": "assumption: 9450-class propellers (9.45 in); the APC SF geometry is "
                                            "scaled to that diameter, the Phantom's own blade is not modelled",
                        "cells": "15.2 V nominal = 4 x 3.8 V"}}
PHANTOM4_BAND = 0.25


# ------------------------------------------------------------------ 1. rotor
def rotor_verification(airfoil: Airfoil | None = None) -> dict:
    """The momentum-theory identity, the APC 10 x 4.7 static data with the given polar (before), the polar
    calibrated on the even rows (odd rows held out) and on the low-RPM half (high half held out)."""
    d = np.asarray(APC_10X4_7_STATIC)
    rot = apc_10x4_7(airfoil)
    even = np.arange(len(d)) % 2 == 0
    low = d[:, 0] < np.median(d[:, 0])
    cal = calibrate_airfoil(rot, d[:, 0], d[:, 1], d[:, 2], train=even)
    ext = calibrate_airfoil(rot, d[:, 0], d[:, 1], d[:, 2], train=low)
    return {"ideal": ideal_twist_check(), "interpolation": cal, "extrapolation": ext, "source": APC_SOURCE}


# ------------------------------------------------------------------ 3. sizing
def build_quad(*, scale: float, cells: int, capacity_ah: float, payload_kg: float, airfoil: Airfoil,
               thrust_to_weight: float = 2.0, mass_model: Mapping[str, float] | None = None, iters: int = 6) -> Quad:
    """A quad on the APC 10 x 4.7 scaled by ``scale``, a ``cells``S ``capacity_ah`` pack, motors sized for
    ``thrust_to_weight`` (``K_v`` from the rotor speed that gives it at 95 % of the pack's nominal voltage under
    load; mass from the maximum shaft power), the arms from the propeller clearance; mass to a fixed point."""
    mm = {**MASS_MODEL, **dict(mass_model or {})}
    rot = apc_10x4_7(airfoil, scale)
    D = 2 * rot.radius
    arm = D * (1 + mm["clearance"]) / math.sqrt(2)
    bat = Battery(n_series=int(cells), capacity_ah=float(capacity_ah), r_cell=mm["r_cell_ohm_Ah"] / capacity_ah,
                  peukert=mm["peukert"], specific_energy=mm["battery_Wh_per_kg"], reserve_soc=mm["reserve_soc"])
    hover_rpm = 5000.0
    table = rotor_table(rot, hover_rpm)
    m_motor = mm["motor_kg_ref"]
    mass = payload_kg + bat.mass + 1.0
    for _ in range(iters):
        t_max = thrust_to_weight * mass * G / 4
        cT0, cP0 = float(table["cT"][0]), float(table["cP"][0])
        n_max = math.sqrt(t_max / (cT0 * RHO * D ** 4))
        w_max = 2 * math.pi * n_max
        q_max = cP0 * RHO * n_max ** 2 * D ** 5 / (2 * math.pi)
        p_max = q_max * w_max
        m_motor = max(p_max / mm["motor_W_per_kg"], 0.01)
        r_ohm = mm["motor_r_ohm_ref"] * mm["motor_kg_ref"] / m_motor
        i0 = mm["motor_i0_ref"] * m_motor / mm["motor_kg_ref"]
        v_avail = 0.95 * 3.7 * cells
        i_max = q_max / 1.0                                # first guess for K_t = 1
        kt = 0.01
        for _ in range(30):                                 # V = (Q/Kt + I0) R + Kt w  ->  solve for Kt
            i_max = q_max / kt + i0
            kt_new = (v_avail - i_max * r_ohm) / w_max
            if kt_new <= 0:
                kt_new = kt * 2
            kt = 0.5 * (kt + kt_new)
        kv = 60.0 / (2 * math.pi * kt)
        motor = Motor(kv_rpm=kv, r_ohm=r_ohm, i0=i0, mass=m_motor, max_current=i_max)
        props = 4 * mm["prop_kg_10in"] * scale ** 2
        parts = {"payload": payload_kg, "battery": bat.mass, "motors": 4 * m_motor, "esc": 4 * mm["esc_kg"],
                 "props": props, "arms": 4 * arm * mm["arm_kg_per_m"], "body": mm["body_kg"],
                 "avionics": mm["avionics_kg"]}
        new = sum(parts.values())
        hover_rpm = 60 * math.sqrt(new * G / 4 / (cT0 * RHO * D ** 4))
        if abs(new - mass) < 1e-5 * new:
            mass = new
            break
        mass = new
    table = rotor_table(rot, hover_rpm)
    # inertia: motors + props as point masses at the arm ends, the rest as a 0.1 m box at the centre
    m_tip = parts["motors"] + parts["props"] + parts["esc"]
    m_c = mass - m_tip
    ixx = m_tip / 4 * 2 * arm ** 2 + m_c * (0.1 ** 2 + 0.05 ** 2) / 12
    izz = m_tip * arm ** 2 + m_c * 2 * 0.1 ** 2 / 12
    return Quad(rot, table, motor, bat, arm, mass, np.diag([ixx, ixx, izz]), cd_area=mm["cd_area_m2"],
                avionics_W=mm["avionics_W"], parts=parts)


RHO = 1.225


def _estimate_endurance(q: Quad) -> dict:
    """Quasi-static hover endurance [s]: the hover power at mid charge, the Peukert capacity at that current down to
    the reserve."""
    hp = hover_power(q, soc=0.5)
    if not math.isfinite(hp["current_A"]):
        return {"endurance_s": 0.0, **hp}
    cap = q.battery.peukert_capacity(hp["current_A"]) * (1 - q.battery.reserve_soc)
    return {"endurance_s": cap / hp["current_A"] * 3600.0, **hp}


def size_quad(*, payload_kg: float, endurance_min: float, airfoil: Airfoil, thrust_to_weight: float = 2.0,
              scales: Sequence[float] = (0.8, 1.0, 1.2, 1.4), cells: Sequence[int] = (3, 4, 6),
              capacities: Sequence[float] = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0),
              mass_model: Mapping[str, float] | None = None) -> dict:
    """Lightest (scale, cells, capacity) whose estimated hover endurance meets ``endurance_min`` with the
    thrust-to-weight target reached at full voltage."""
    rows, best = [], None
    t0 = time.perf_counter()
    for sc in scales:
        for s in cells:
            for c in capacities:
                q = build_quad(scale=sc, cells=s, capacity_ah=c, payload_kg=payload_kg, airfoil=airfoil,
                               thrust_to_weight=thrust_to_weight, mass_model=mass_model)
                est = _estimate_endurance(q)
                ok = est["endurance_s"] >= 60 * endurance_min and est["thrust_to_weight"] >= 0.98 * thrust_to_weight
                rows.append([sc, s, c, q.mass, est["endurance_s"] / 60, est["thrust_to_weight"], ok])
                if ok and (best is None or q.mass < best[0].mass):
                    best = (q, est, {"scale": sc, "cells": s, "capacity_ah": c})
    return {"grid": rows, "best": best, "seconds": time.perf_counter() - t0}


# ------------------------------------------------------------------ published case
def published_endurance_check(airfoil: Airfoil, dt: float = 0.05) -> dict:
    """The Phantom 4's published mass and pack on the scaled APC propeller, hovered on the battery until it is empty
    (6-DOF, LQR, voltage-limited rotors): the measured hover time against the published maximum flight time."""
    p = PHANTOM4
    scale = p["prop_diameter_in"] / 10.0
    # hold the published take-off mass: the payload slot takes up what the mass build-up does not account for
    payload, q = 0.0, None
    for _ in range(8):
        q = build_quad(scale=scale, cells=p["cells"], capacity_ah=p["capacity_Ah"], payload_kg=payload,
                       airfoil=airfoil)
        if abs(q.mass - p["mass_kg"]) <= 1e-4 * p["mass_kg"]:
            break
        payload = max(payload + p["mass_kg"] - q.mass, 0.0)
    q2 = q
    ctrl = lqr_hover(q2)
    run = fly_mission(q2, ctrl, [Segment("hover", [0, 0, 0], 60.0)], dt=dt)
    t_min = run["endurance_s"] / 60
    err = t_min / p["flight_time_min"] - 1
    return {"mass_kg": q2.mass, "unaccounted_payload_kg": payload, "hover_minutes": t_min, "published_minutes":
            p["flight_time_min"], "rel_error": err, "within_band": abs(err) <= PHANTOM4_BAND,
            "mean_power_W": run["mean_power_W"], "energy_Wh": run["energy_Wh"], "sources": p["sources"]}


# ------------------------------------------------------------------ workflow
DEFAULT_MISSION = [{"name": "hover", "target": [0, 0, 0], "duration": 30.0, "speed": 1.0},
                   {"name": "climb", "target": [0, 0, 20], "duration": 30.0, "speed": 2.0},
                   {"name": "cruise", "target": [100, 0, 20], "duration": 60.0, "speed": 5.0},
                   {"name": "return", "target": [0, 0, 20], "duration": 60.0, "speed": 5.0},
                   {"name": "descend", "target": [0, 0, 0], "duration": 30.0, "speed": 1.5}]


def drone_workflow(*, payload_kg: float = 0.3, endurance_min: float = 15.0, thrust_to_weight: float = 2.0,
                   mission: Sequence[Mapping[str, Any]] | None = None, gust_m_s: float = 3.0,
                   calibrate_polar: bool = True, published: bool = True, dt: float = 0.02,
                   mass_model: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Rotor verification -> sizing -> control checks -> mission flown on the battery (see the module docstring)."""
    checks: list[dict[str, Any]] = []

    def chk(code, ok, detail, value=None, status=None):
        st = status or ("pass" if ok is True else "fail" if ok is False else "inconclusive")
        checks.append({"code": code, "status": st, "detail": detail, "value": value})

    out: dict[str, Any] = {"inputs": {"payload_kg": payload_kg, "endurance_min": endurance_min,
                                      "thrust_to_weight": thrust_to_weight}}
    # 1. rotor
    rv = rotor_verification()
    ideal = rv["ideal"]
    chk("rotor.momentum_identity", ideal["rel_error"] <= 1e-6 and ideal["inflow_spread"] <= 1e-3,
        f"ideal twist, no tip loss or profile drag: BEMT power {ideal['power_W']:.4f} W against the momentum ideal "
        f"{ideal['momentum_power_W']:.4f} W ({ideal['rel_error']:.1e}); inflow spread {ideal['inflow_spread']:.1e}",
        ideal["rel_error"])
    ci, ce = rv["interpolation"], rv["extrapolation"]
    chk("rotor.apc_assumed_polar", None, f"assumed polar on the APC SF 10x4.7 static data: c_T max error "
        f"{ci['before']['cT_max_rel']:.1%}, c_P {ci['before']['cP_max_rel']:.1%} (low-Reynolds drag and the lift's "
        "Reynolds trend are not in the assumed polar)", ci["before"], status="info")
    chk("rotor.apc_calibrated_held_out", ci["after"]["cT_max_rel"] <= 0.05 and ci["after"]["cP_max_rel"] <= 0.05,
        f"polar fitted on {ci['n_train']} rows, {ci['n_test']} held out: c_T max error {ci['after']['cT_max_rel']:.1%}, "
        f"c_P {ci['after']['cP_max_rel']:.1%} (<= 5 %)", ci["after"])
    chk("rotor.apc_extrapolated", ce["after"]["cT_max_rel"] <= 0.10 and ce["after"]["cP_max_rel"] <= 0.10,
        f"fitted below the median RPM, tested above it: c_T max error {ce['after']['cT_max_rel']:.1%}, c_P "
        f"{ce['after']['cP_max_rel']:.1%} (<= 10 %)", ce["after"])
    bounds = {"lift_slope": (3.0, 6.5), "alpha0_deg": (-8.0, 2.0), "cd0": (0.004, 0.06), "re_exponent": (-1.0, 0.0),
              "re_lift_exponent": (0.0, 0.5)}
    at_b = [k for k, v in ci["params"].items() if min(abs(v - bounds[k][0]), abs(v - bounds[k][1])) <= 1e-6 *
            (bounds[k][1] - bounds[k][0])]
    chk("rotor.polar_bounds", None, "calibrated polar " + ", ".join(f"{k} {v:.4g}" for k, v in ci["params"].items())
        + (f"; at a bound: {at_b} (an effective polar, not a measured section)" if at_b else ""), at_b,
        status="warn" if at_b else "info")
    af = ci["airfoil"] if calibrate_polar else Airfoil()
    out["rotor"] = {"ideal": ideal, "assumed_polar_errors": ci["before"], "held_out_errors": ci["after"],
                    "extrapolation_errors": ce["after"], "polar": ci["params"], "source": rv["source"]}
    # 2-3. sizing
    sz = size_quad(payload_kg=payload_kg, endurance_min=endurance_min, airfoil=af, thrust_to_weight=thrust_to_weight,
                   mass_model=mass_model)
    out["sizing_grid"] = sz["grid"]
    if sz["best"] is None:
        chk("sizing.feasible", False, f"no propeller / pack on the grid gives {endurance_min:g} min with T/W "
            f"{thrust_to_weight:g}")
        out.update(checks=checks, verdict="fail", failed=["sizing.feasible"])
        return out
    q, est, pick = sz["best"]
    n_ok = sum(1 for r in sz["grid"] if r[-1])
    chk("sizing.feasible", True, f"{n_ok} of {len(sz['grid'])} propeller / pack choices meet the endurance and T/W; "
        f"lightest {q.mass:.3f} kg")
    hp = hover_power(q, soc=0.8)
    chk("motor.power_closure", hp["motor"]["closure"] <= 1e-12, f"shaft + copper + no-load = V I to "
        f"{hp['motor']['closure']:.1e} (motor efficiency {hp['motor']['efficiency']:.1%} in hover)",
        hp["motor"]["closure"])
    ideal_p = 4 * momentum_hover_power(q.mass * G / 4, q.rotor.radius)
    b = bemt(q.rotor, hp["omega"], 0.0, n=40)
    fm = b["FM"]
    rel = abs(hp["shaft_W"] / (ideal_p / fm) - 1)
    chk("rotor.hover_power_vs_momentum", rel <= 0.02, f"hover shaft power {hp['shaft_W']:.1f} W against the "
        f"momentum ideal {ideal_p:.1f} W / FM {fm:.3f} = {ideal_p / fm:.1f} W ({rel:.2%}; FM from the BEMT at the "
        "hover speed, <= 2 %)", rel)
    out["design"] = {**pick, "mass_kg": q.mass, "parts_kg": q.parts, "prop_diameter_in": 2 * q.rotor.radius / 0.0254,
                     "arm_m": q.arm, "motor_kv": q.motor.kv_rpm, "motor_r_ohm": q.motor.r_ohm,
                     "hover_rpm": hp["rpm"], "hover_power_W": hp["power_W"], "figure_of_merit": fm,
                     "thrust_to_weight": hp["thrust_to_weight"], "estimated_endurance_min": est["endurance_s"] / 60,
                     "inertia_kg_m2": np.diag(q.J).tolist()}
    # 4. control
    ctrl = lqr_hover(q)
    re_max = float(np.max(np.real(ctrl["poles"])))
    chk("control.closed_loop_stable", re_max < 0, f"largest closed-loop real part {re_max:.3g} 1/s (LQR, 15 states)",
        re_max)
    mg = loop_margins_per_channel(ctrl, q.tau_motor)
    pm_min = min(m["phase_margin"] for m in mg)
    up_min = min(m["gain_margin_up_db"] for m in mg)
    down_min = min(m["gain_margin_down_db"] for m in mg)
    chk("control.margins", pm_min >= 45 and up_min >= 6 and down_min >= 6, "per channel, the others closed, motor "
        "lag included: " + "; ".join(f"{m['channel']} PM {m['phase_margin']:.1f} deg, GM +{m['gain_margin_up_db']:.1f}"
                                     f" / -{m['gain_margin_down_db']:.1f} dB" for m in mg) +
        " (>= 45 deg; >= 6 dB up and down)", {"phase_margin_min": pm_min, "gain_margin_up_db_min": up_min,
                                              "gain_margin_down_db_min": down_min})
    from pinneapple_systems.control.analysis import step_metrics
    st = simulate(q, ctrl, t_end=21.0, dt=0.005, reference=lambda t: np.array([1.0, 0.0, 0.0]) if t >= 1.0
                  else np.zeros(3), record_every=1)
    k = st["t"] >= 1.0
    m = step_metrics(st["t"][k] - 1.0, st["p"][k, 0], reference=1.0, y0=0.0)
    chk("control.step", m["overshoot"] <= 30.0 and m["settling_time"] <= 12.0, f"1 m step in x: overshoot "
        f"{m['overshoot']:.1f} %, 2 % settling {m['settling_time']:.2f} s (<= 30 %, <= 12 s)", m)
    gust = simulate(q, ctrl, t_end=15.0, dt=0.005, wind=lambda t: np.array([gust_m_s, 0, 0]) if 2.0 <= t < 7.0
                    else np.zeros(3), record_every=1)
    dev = np.linalg.norm(gust["p"], axis=1)
    peak = float(dev.max())
    final = float(dev[-1])
    chk("control.gust", peak <= 1.0 and final <= 0.05, f"{gust_m_s:g} m/s gust for 5 s: peak excursion {peak:.3f} m, "
        f"{final:.3f} m off at the end (<= 1 m, <= 5 cm)", {"peak_m": peak, "final_m": final})
    out["control"] = {"poles_max_real": re_max, "margins": mg, "step": m, "gust_peak_m": peak, "gust_final_m": final}
    # 5. mission
    segs = [Segment(s["name"], s["target"], float(s["duration"]), float(s.get("speed", 2.0)))
            for s in (mission or DEFAULT_MISSION)]
    t0 = time.perf_counter()
    fl = fly_mission(q, ctrl, segs, dt=dt)
    wall = time.perf_counter() - t0
    tmin = fl["endurance_s"] / 60
    chk("mission.empty_battery", fl["empty"], f"flown until the pack was empty: {tmin:.2f} min, {fl['laps']:.2f} "
        f"laps, {fl['energy_Wh']:.1f} W h drawn (wall {wall:.0f} s)", fl["endurance_s"])
    chk("mission.endurance", tmin >= endurance_min, f"measured endurance {tmin:.2f} min on the mission (target "
        f"{endurance_min:g} min; hover estimate {est['endurance_s'] / 60:.2f} min)", tmin)
    chk("mission.tracking", fl["max_tracking_error_m"] <= 2.0, f"largest distance from the moving reference "
        f"{fl['max_tracking_error_m']:.2f} m (<= 2 m)", fl["max_tracking_error_m"])
    chk("mission.no_saturation", fl["saturated_steps"] == 0, f"{fl['saturated_steps']} steps with a rotor at the "
        "voltage limit", fl["saturated_steps"], status=None if fl["saturated_steps"] == 0 else "warn")
    out["mission"] = {k: v for k, v in fl.items() if k != "history"}
    out["mission"]["wall_seconds"] = wall
    out["mission"]["edgewise_rotor"] = "not modelled: cruise power is that of the axial-flow rotor"
    if published:
        pub = published_endurance_check(af)
        out["published"] = pub
        chk("published.phantom4_hover", pub["within_band"], f"Phantom 4 mass {pub['mass_kg']:.3f} kg, 4S 5.35 A h "
            f"on a 9.45 in scaled APC propeller: hover {pub['hover_minutes']:.1f} min against the published "
            f"{pub['published_minutes']:g} min ({pub['rel_error']:+.1%}, band +-{PHANTOM4_BAND:.0%})", pub["rel_error"])
    out["mass_model"] = {**MASS_MODEL, **dict(mass_model or {})}
    out["checks"] = checks
    bad = [c["code"] for c in checks if c["status"] == "fail"]
    out["verdict"] = "fail" if bad else "pass"
    out["failed"] = bad
    return out
