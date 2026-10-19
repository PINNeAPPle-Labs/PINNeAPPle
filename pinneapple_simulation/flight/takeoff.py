"""Take-off: ground roll, rotation at V_R, lift-off, climb-out over the 35 ft screen, and a closed-form check.

:func:`simulate_takeoff` runs the 6-DOF model from brake release (on its gear at rest, full throttle) with the
:class:`~.control.TakeoffPilot` and any wind field (crosswind, gusts, turbulence); :func:`takeoff_metrics` reads off
the ground-roll distance, lift-off speed, pitch attitude and angle of attack at lift-off, the distance to 35 ft (lowest
wheel), the speed there, the climb gradient after the screen, tail and wing-tip clearances and the lateral deviation.

:func:`raymer_takeoff` is the closed-form estimate of Raymer (Aircraft Design: A Conceptual Approach, section 17.8):

    S_G  = 1 / (2 g K_A) ln((K_T + K_A V_LOF^2) / K_T),  K_T = T/W - mu,  K_A = rho / (2 W/S) (mu CL - CD0 - K CL^2)
    S_R  = t_R V_LOF                                     (rotation: ~1 s small aircraft, ~3 s large)
    S_TR = R sin(gamma_climb),  R = V_TR^2 / (0.2 g),  V_TR = 1.15 V_stall,  h_TR = R (1 - cos gamma_climb)
    S_C  = (h_obs - h_TR) / tan(gamma_climb)             (or S_TR = sqrt(R^2 - (R - h_obs)^2) if h_TR > h_obs)

with the thrust taken at 0.7 V_LOF and the ground-roll CL, CD0, K from the same aircraft model.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .aero import T_AB
from .aircraft import Aircraft, Controls
from .control import LoopDesign, TakeoffPilot, TakeoffSchedule
from .rigid_body import G0, quat_to_dcm
from .simulate import FlightHistory, simulate
from .trim import ground_equilibrium, stability_derivatives, trim

FT, KT = 0.3048, 0.514444
SCREEN = 35 * FT


def default_schedule(ac: Aircraft, kind: str = "transport") -> TakeoffSchedule:
    """V_R = 1.05 x the take-off-flap stall speed (CLmax_TO of the aircraft), climb at 1.25 V_s (between V2 = 1.13
    V_SR and V2 + 10 kt for a transport), rotation at 3 deg/s to 12 deg (transport) or 6 deg/s to 11 deg (light); an
    assumption of a typical technique, not a certified schedule."""
    Vs = ac.stall_speed()
    if kind == "transport":
        return TakeoffSchedule(V_R=1.05 * Vs, theta_rotation=math.radians(12.0), rotation_rate=math.radians(3.0),
                               V_climb=1.25 * Vs, flap=ac.flap_takeoff)
    return TakeoffSchedule(V_R=1.10 * Vs, theta_rotation=math.radians(11.0), rotation_rate=math.radians(6.0),
                           V_climb=1.30 * Vs, flap=ac.flap_takeoff)


def takeoff_derivatives(ac: Aircraft, V: float) -> dict[str, float]:
    """Control derivatives for the pilot's gain design: about a trimmed climb-out at V with the take-off flap."""
    tr = trim(ac, V, 50.0, gamma=0.0, flap=ac.flap_takeoff, gear_down=True)
    return stability_derivatives(ac, tr)


def simulate_takeoff(ac: Aircraft, schedule: TakeoffSchedule | None = None, wind=None, t_max: float = 90.0,
                     dt: float = 0.01, h_stop: float = 400 * FT, design: LoopDesign | None = None,
                     record_gamma: bool = False, record_every: int = 1, kind: str = "transport") -> FlightHistory:
    s = schedule or default_schedule(ac, kind)
    d = takeoff_derivatives(ac, s.V_climb)
    pilot = TakeoffPilot(ac, s, d, design)
    x0 = ground_equilibrium(ac, flap=s.flap, psi=s.runway_heading)

    def stop(t, x, ev):
        return -x[2] > h_stop

    hist = simulate(ac, x0, pilot, t_max, dt, wind=wind, stop=stop, record_gamma=record_gamma,
                    record_every=record_every)
    hist.events.update(pilot.events)
    hist.events["schedule"] = {"V_R": s.V_R, "theta_rotation_deg": math.degrees(s.theta_rotation),
                               "rotation_rate_deg_s": math.degrees(s.rotation_rate), "V_climb": s.V_climb,
                               "flap_deg": math.degrees(s.flap), "throttle": s.throttle}
    hist.events["derivatives"] = d
    return hist


def _point_heights(ac: Aircraft, hist: FlightHistory, pA) -> np.ndarray:
    r = T_AB @ (np.asarray(pA, float) - ac.cg)
    return np.array([-(x[2] + (quat_to_dcm(x[6:10]) @ r)[2]) for x in hist.x])


def takeoff_metrics(ac: Aircraft, hist: FlightHistory) -> dict[str, Any]:
    ev = hist.events
    x0 = hist.x[0, :2]
    hdg = 0.0
    along = (hist.x[:, :2] - x0) @ np.array([math.cos(hdg), math.sin(hdg)])
    lat = (hist.x[:, :2] - x0) @ np.array([-math.sin(hdg), math.cos(hdg)])
    wheels = np.min([_point_heights(ac, hist, g.contact) for g in ac.gear], axis=0)
    out: dict[str, Any] = {"aircraft": ac.name, "mass_kg": ac.mass, "V_R": ev.get("V_R")}
    if "t_LOF" not in ev:
        out["lifted_off"] = False
        return out
    k = int(np.searchsorted(hist.t, ev["t_LOF"]))
    out.update(lifted_off=True, ground_roll_m=float(along[k]), t_LOF=float(ev["t_LOF"]), V_LOF=float(ev["V_LOF"]),
               V_LOF_kt=float(ev["V_LOF"] / KT), theta_LOF_deg=math.degrees(ev["theta_LOF"]),
               alpha_LOF_deg=math.degrees(ev["alpha_LOF"]), V_R_kt=float(ev["V_R"] / KT),
               rotation_time_s=float(ev["t_LOF"] - ev["t_R"]), rotation_distance_m=float(along[k] - along[
                   int(np.searchsorted(hist.t, ev["t_R"]))]))
    above = np.where(wheels >= SCREEN)[0]
    if len(above):
        j = int(above[0])
        f = (SCREEN - wheels[j - 1]) / (wheels[j] - wheels[j - 1])
        out["distance_35ft_m"] = float(along[j - 1] + f * (along[j] - along[j - 1]))
        out["t_35ft"] = float(hist.t[j - 1] + f * (hist.t[j] - hist.t[j - 1]))
        out["V_35ft_kt"] = float(hist.airspeed[j] / KT)
        # climb gradient: from the screen to the end of the run (or 400 ft), as height gained over distance
        e = len(hist.t) - 1
        if along[e] - along[j] > 50:
            out["climb_gradient_pct"] = float(100 * (wheels[e] - wheels[j]) / (along[e] - along[j]))
            out["climb_end_height_m"] = float(wheels[e])
    out["theta_max_deg"] = float(np.degrees(hist.euler[:, 1].max()))
    out["alpha_max_deg"] = float(np.degrees(hist.alpha.max()))
    out["CL_max_reached"] = float(hist.CL.max())
    out["CLmax_takeoff_assumed"] = ac.clmax_takeoff
    out["stall_margin_ok"] = bool(hist.CL.max() < ac.clmax_takeoff)
    if ac.tail_point is not None:
        out["tail_clearance_min_m"] = float(_point_heights(ac, hist, ac.tail_point)[:k + 200].min())
    if ac.wingtip is not None:
        tl = ac.wingtip * np.array([1, -1, 1])
        out["wingtip_clearance_min_m"] = float(min(_point_heights(ac, hist, ac.wingtip).min(),
                                                   _point_heights(ac, hist, tl).min()))
    out["max_lateral_deviation_ground_m"] = float(np.abs(lat[:k + 1]).max())
    out["max_bank_deg"] = float(np.degrees(np.abs(hist.euler[:, 0]).max()))
    out["max_rudder_deg"] = float(np.degrees(np.abs(hist.controls["rudder"]).max()))
    out["max_aileron_deg"] = float(np.degrees(np.abs(hist.controls["aileron"]).max()))
    out["heading_at_LOF_deg"] = float(np.degrees(hist.euler[k, 2]))
    out["nz_max"] = float(hist.nz.max())
    out["peak_gear_load_kN"] = float(hist.gear_loads.max() / 1e3)
    return out


def raymer_takeoff(ac: Aircraft, V_LOF: float | None = None, mu: float = 0.02, t_rotation: float = 3.0,
                   h_obstacle: float = SCREEN, theta_ground: float = 0.0) -> dict[str, float]:
    """Raymer's closed-form take-off distance with this aircraft's thrust, lift and drag (see the module text).
    ``V_LOF`` defaults to 1.1 V_stall (take-off flap). Ground-roll lift and drag from the lattice at the ground
    attitude with the take-off flap and the gear down, ground effect on the induced drag at the wing height."""
    rho, a, _ = ac.atmosphere(np.zeros(13))
    W, S = ac.weight, ac.af.wing_area
    Vs = ac.stall_speed(rho)
    VL = V_LOF or 1.1 * Vs
    x = ground_equilibrium(ac, flap=ac.flap_takeoff)
    x[3] = 0.7 * VL
    ev = ac.evaluate(0.0, x, Controls(flap=ac.flap_takeoff, throttle=0.0))
    ae = ev["aero"]
    CL_g = ae["CL"]
    h_w = ev["h_agl"] + ac.wing_dz
    hb = max(h_w, 0.5) / ac.af.span
    k_ge = (16 * hb) ** 2 / (1 + (16 * hb) ** 2)
    K = ac.aero.k_induced * k_ge
    CD0 = ac.aero.cd0 + ac.aero.cd_gear + ac.aero.cd_flap_per_rad * ac.flap_takeoff
    T07 = sum(e.thrust(0.7 * VL, rho, a, 1.0) for e in ac.engines)
    KT_ = T07 / W - mu
    KA = rho / (2 * W / S) * (mu * CL_g - CD0 - K * CL_g ** 2)
    SG = 1 / (2 * G0 * KA) * math.log((KT_ + KA * VL ** 2) / KT_)
    SR = t_rotation * VL
    VTR = 1.15 * Vs
    R = VTR ** 2 / (0.2 * G0)
    CL_c = 2 * W / (rho * VTR ** 2 * S)
    D = 0.5 * rho * VTR ** 2 * S * (CD0 + ac.aero.k_induced * CL_c ** 2)
    T = sum(e.thrust(VTR, rho, a, 1.0) for e in ac.engines)
    gam = math.asin(max(min((T - D) / W, 0.99), 1e-4))
    hTR = R * (1 - math.cos(gam))
    if hTR > h_obstacle:
        STR, SC = math.sqrt(R ** 2 - (R - h_obstacle) ** 2), 0.0
    else:
        STR, SC = R * math.sin(gam), (h_obstacle - hTR) / math.tan(gam)
    return {"S_ground_m": SG, "S_rotation_m": SR, "S_transition_m": STR, "S_climb_m": SC,
            "total_35ft_m": SG + SR + STR + SC, "V_LOF": VL, "V_stall": Vs, "T_07VLOF_N": T07, "K_T": KT_, "K_A": KA,
            "CL_ground": CL_g, "CD0": CD0, "K_induced_ground": K, "climb_gradient_pct": 100 * math.tan(gam),
            "mu": mu, "t_rotation_s": t_rotation}


def raymer_variants(ac: Aircraft, metrics: dict[str, float], kind: str = "light") -> dict[str, dict[str, float]]:
    """The two closed-form references the take-off results quote (Raymer, Aircraft Design, sec. 17.8):

    * ``raymer_1p1Vs``: V_LOF = 1.1 V_stall and Raymer's rotation time, 1 s for light aircraft, 3 s for transports;
    * ``raymer_sim_VLOF``: the simulated lift-off speed and rotation time (``metrics`` from :func:`takeoff_metrics`),
      so only the closed-form ground roll, transition and climb differ from the simulation.

    Each carries ``sim_over_raymer``: the simulated distance to the 35 ft screen over the closed form."""
    t_rot = 3.0 if kind == "transport" else 1.0
    out = {"raymer_1p1Vs": raymer_takeoff(ac, t_rotation=t_rot),
           "raymer_sim_VLOF": raymer_takeoff(ac, V_LOF=metrics["V_LOF"], t_rotation=metrics["rotation_time_s"])}
    for v in out.values():
        v["sim_over_raymer"] = metrics["distance_35ft_m"] / v["total_35ft_m"]
    return out
