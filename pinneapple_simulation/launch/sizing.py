"""Sizing the stages: how much propellant each one needs for its share of the velocity, and how much thrust it needs
to leave the pad.

The inverse of the Tsiolkovsky relation, stage by stage from the payload down:

    R    = exp(dv / (Isp g0))                       (wet + above) / (dry + above)
    wet  = m_above (R - 1) / (1 - R eps)            m_dry = eps wet,   m_prop = wet - m_dry

with ``eps`` the structural coefficient (dry mass over wet mass of that stage) and ``m_above`` everything carried on
top of it at lift-off. A stage that is too small for its share (``exp(-dv / c) <= eps``) cannot be built: the split
is infeasible and the function says so instead of returning a negative tank.

    from pinneapple_simulation.launch.sizing import size_vehicle
    veh = size_vehicle(200.0, delta_v=(3500.0, 4100.0, 700.0))
    veh.liftoff_mass_kg, veh.delta_v_vac_m_s

The velocity split here is the one the ascent measures, not a guess: :func:`close` iterates the split against the
simulation (the gravity, drag and steering losses the flight actually pays) until the design inserts on the target.
"""
from __future__ import annotations

import math

from .vehicle import G0, LaunchVehicle, Stage

__all__ = ["DEFAULTS", "size_vehicle", "delta_v_required"]

#: the assumptions of the concept: vacuum and sea-level impulse of each stage, structural coefficient, thrust-to-weight
#: at lift-off, the diameter of the stack and the fairing mass. Every one is an engineering assumption of this study.
DEFAULTS = {
    "isp_vac": (311.0, 348.0, 292.0),        # kerolox, methalox, solid kick stage
    "isp_sl": (261.0, 306.0, 292.0),
    "eps": (0.072, 0.105, 0.16),             # dry mass over wet mass of each stage
    "tw_liftoff": 1.45,
    "diameter": (1.30, 1.30, 1.10),
    "fairing_kg": 60.0,
    "fairing_jettison_altitude_km": 105.0,
}


def _length_m(propellant_kg: float, diameter_m: float, density_kg_m3: float) -> float:
    """Tank length of a stage: its propellant volume over the area of the stack (a cylinder, no domes)."""
    volume = propellant_kg / density_kg_m3
    return round(max(volume / (math.pi * (diameter_m / 2.0) ** 2), 0.6), 1)


def delta_v_required(target_altitude_km: float, site_lat_deg: float, mu: float = 3.986004418e14,
                     radius: float = 6.371e6, omega: float = 7.2921150e-5, losses_m_s: float = 1_900.0) -> dict:
    """The velocity budget the ascent has to find: the orbital speed less the help of the Earth's rotation, plus the
    gravity, drag and steering losses (``losses_m_s`` defaults to the 1.9 km/s this model measures)."""
    r = radius + target_altitude_km * 1e3
    v_c = math.sqrt(mu / r)
    v_pad = omega * radius * math.cos(math.radians(site_lat_deg))
    gain = v_c - v_pad
    return {"circular_speed_m_s": v_c, "pad_speed_m_s": v_pad, "speed_gain_m_s": gain,
            "losses_m_s": losses_m_s, "total_m_s": gain + losses_m_s}


def size_vehicle(payload_kg: float, delta_v=(3500.0, 4100.0, 700.0), *, isp_vac=DEFAULTS["isp_vac"],
                 isp_sl=DEFAULTS["isp_sl"], eps=DEFAULTS["eps"], tw_liftoff: float = DEFAULTS["tw_liftoff"],
                 diameter=DEFAULTS["diameter"], fairing_kg: float = DEFAULTS["fairing_kg"],
                 fairing_jettison_altitude_km: float = DEFAULTS["fairing_jettison_altitude_km"],
                 name: str = "Concept-200") -> LaunchVehicle:
    """A vehicle whose three stages carry ``delta_v`` [m/s], sized from the payload up."""
    if len(delta_v) != 3:
        raise ValueError("three stages: give three velocities")
    stages = [None, None, None]
    above = payload_kg + fairing_kg
    for i in (2, 1, 0):
        c = isp_vac[i] * G0
        ratio = math.exp(delta_v[i] / c)
        if ratio * eps[i] >= 1.0:
            raise ValueError(f"stage {i + 1} cannot pay {delta_v[i]:.0f} m/s with eps {eps[i]:.2f}: "
                             f"exp(dv/c) eps = {ratio * eps[i]:.3f} >= 1")
        m0 = above * (ratio - 1.0) / (1.0 - ratio * eps[i])
        dry = eps[i] * m0
        stages[i] = Stage(f"S{i + 1}", dry_mass_kg=dry, propellant_kg=m0 - dry, isp_vac_s=isp_vac[i],
                          isp_sl_s=isp_sl[i], thrust_vac_n=1.0, diameter_m=diameter[i],
                          length_m=_length_m(m0 - dry, diameter[i], 1000.0))
        above = above + m0
    liftoff = above + fairing_kg
    thrust_sl = tw_liftoff * liftoff * G0
    stages[0] = Stage(stages[0].name, stages[0].dry_mass_kg, stages[0].propellant_kg, stages[0].isp_vac_s,
                      thrust_vac_n=thrust_sl * stages[0].isp_vac_s / stages[0].isp_sl_s,
                      diameter_m=stages[0].diameter_m, isp_sl_s=stages[0].isp_sl_s, length_m=stages[0].length_m)
    # the upper stages: thrust-to-weight of 1.1 (second) and 0.45 (kick) on their own lift-off mass
    for i, tw in ((1, 1.10), (2, 1.20)):
        wet_here = stages[i].wet_mass_kg + sum(s.wet_mass_kg for s in stages[i + 1:]) + payload_kg
        stages[i] = Stage(stages[i].name, stages[i].dry_mass_kg, stages[i].propellant_kg, stages[i].isp_vac_s,
                          thrust_vac_n=tw * wet_here * G0, diameter_m=stages[i].diameter_m,
                          isp_sl_s=stages[i].isp_sl_s, length_m=stages[i].length_m)
    veh = LaunchVehicle(stages=tuple(stages), payload_kg=payload_kg, fairing_kg=fairing_kg,
                        fairing_jettison_altitude_km=fairing_jettison_altitude_km, name=name,
                        notes={"structure": "assumption", "impulse": "assumption", "thrust": "assumption"})
    return veh


def close(site, target, *, payload_kg: float = 200.0, dt: float = 0.2, iterations: int = 6, kick_m_s: float = 950.0,
          pitch_end_altitude_km: float = 300.0, pitch_exponent: float = 0.6, verbose: bool = False):
    """Close the velocity budget on the simulation and return ``(vehicle, history, report)``.

    One knob is solved: the total ideal velocity the three stages carry. Each iteration flies the vehicle, reads the
    apogee the second stage leaves (the transfer the kick stage circularises from) and the orbit the kick stage
    reaches, and scales the budget by what the transfer was short or long. The kick stage carries a fixed ``kick_m_s``
    (it only has to circularise at apogee) and the rest is split 40 % / 60 % between the first two stages: the first
    leaves the dense air, the second builds the transfer.
    """

    from .ascent import GuidanceConfig, simulate_ascent

    budget = delta_v_required(target.altitude_km, site.lat_deg)
    total = budget["total_m_s"]
    rows, veh, hist = [], None, None
    cfg = GuidanceConfig(pitch_end_altitude_km=pitch_end_altitude_km, pitch_exponent=pitch_exponent)
    for it in range(iterations):
        split = (0.40 * (total - kick_m_s), 0.60 * (total - kick_m_s), kick_m_s)
        veh = size_vehicle(payload_kg, delta_v=split)
        hist = simulate_ascent(veh, site, target, guidance=cfg, dt=dt, t_max=3000.0)
        orb = hist.final_orbit()
        apo = float(hist.events.get("apogee_altitude_km", float("nan")))
        rows.append({"iteration": it, "total_m_s": total, "split": list(split),
                     "liftoff_mass_kg": veh.liftoff_mass_kg, "transfer_apogee_km": apo,
                     "perigee_km": orb["perigee_km"], "apogee_km": orb["apogee_km"], "e": orb["e"],
                     "reserve": float(hist.events.get("propellant_fraction_left", float("nan"))),
                     "losses_m_s": hist.losses["dv_gravity_m_s"] + hist.losses["dv_drag_m_s"]
                     + hist.losses["dv_steering_m_s"]})
        if verbose:
            print(f"  iteration {it}: {veh.liftoff_mass_kg:7.0f} kg, ideal {total:6.0f} m/s -> transfer apogee "
                  f"{apo:6.0f} km, inserted {orb['perigee_km']:6.0f} x {orb['apogee_km']:6.0f} km (e {orb['e']:.4f})")
        if abs(apo - target.altitude_km) < 10.0 and abs(orb["perigee_km"] - target.altitude_km) < 30.0:
            break
        total *= 1.0 + 0.35 * (target.altitude_km - apo) / max(target.altitude_km, 1.0)
    return veh, hist, {"rows": rows, "budget": budget, "pitch_end_altitude_km": pitch_end_altitude_km,
                       "pitch_exponent": pitch_exponent}
