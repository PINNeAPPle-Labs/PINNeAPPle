"""Landing-gear contact with a flat runway: spring-damper legs with rolling, braking and tyre side forces.

Each leg is a vertical spring-damper at a wheel contact point fixed in the airframe (its unloaded position). With
penetration d of that point below the runway (NED z = ``ground_z``) and its vertical velocity d_dot, the normal force
is N = max(0, k d + c d_dot). Along the wheel heading (the body x axis, turned by the steering angle for a steerable
leg) the tyre gives a rolling (and braking) resistance -(mu_roll + brake mu_brake) N, across it a side force
-mu_side N tanh(slip / slip_ref) (slip the tyre slip angle: a linear cornering stiffness mu_side N / slip_ref that
saturates at the friction limit). Both are regularised near zero speed so the aircraft can stand still.

Defaults (assumptions, not measured tyre data): mu_roll 0.02 (dry hard runway, brakes off), mu_brake 0.40, mu_side
0.7, slip_ref 0.10 rad, static compression 0.25 m and damping ratio 0.6 of the sprung mass (:func:`size_gear`).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .aero import T_AB


@dataclass
class GearLeg:
    name: str
    contact: np.ndarray                  # unloaded wheel contact point, airframe axes [m]
    stiffness: float = 0.0               # N/m (set by :func:`size_gear`)
    damping: float = 0.0                 # N s/m
    mu_roll: float = 0.02
    mu_brake: float = 0.40
    mu_side: float = 0.70
    slip_ref: float = 0.10
    steerable: bool = False
    brakes: bool = False
    v_reg: float = 0.05                  # m/s: regularisation of the rolling friction sign


def size_gear(legs: Sequence[GearLeg], mass: float, cg: np.ndarray, static_compression: float = 0.25,
              zeta: float = 0.6, g: float = 9.80665) -> list[GearLeg]:
    """Spring and damper of every leg so that the static compression under its share of the weight is
    ``static_compression``; the shares come from moment balance about the CG in x (nose / mains) and split equally
    between legs at the same x station (left / right)."""
    xs = np.array([float(leg.contact[0]) for leg in legs])
    stations = sorted(set(np.round(xs, 3)))
    if len(stations) != 2:
        raise ValueError("size_gear expects two gear stations in x (nose and mains, or mains and tail)")
    xa, xb = stations
    x = float(cg[0])
    fb = (x - xa) / (xb - xa)                                   # share of the aft station
    for leg in legs:
        st = xb if abs(float(leg.contact[0]) - xb) < 1e-3 else xa
        n_at = sum(abs(float(m.contact[0]) - st) < 1e-3 for m in legs)
        share = (fb if st == xb else 1 - fb) / n_at
        W = share * mass * g
        leg.stiffness = W / static_compression
        leg.damping = 2 * zeta * math.sqrt(leg.stiffness * share * mass)
    return list(legs)


def gear_forces(legs: Sequence[GearLeg], x: np.ndarray, C: np.ndarray, cg: np.ndarray, steer: float = 0.0,
                brake: float = 0.0, ground_z: float = 0.0):
    """(F_b, M_b, normal loads list) of all legs for state x (13-vector) and body->NED matrix C."""
    pos, v, w = x[0:3], x[3:6], x[10:13]
    F = np.zeros(3)
    M = np.zeros(3)
    loads = []
    for leg in legs:
        r = T_AB @ (np.asarray(leg.contact, float) - cg)
        p = pos + C @ r
        d = p[2] - ground_z
        if d <= 0:
            loads.append(0.0)
            continue
        vp = C @ (v + np.cross(w, r))
        N = max(0.0, leg.stiffness * d + leg.damping * vp[2])
        if N == 0.0:
            loads.append(0.0)
            continue
        s = steer if leg.steerable else 0.0
        dh = C @ np.array([math.cos(s), math.sin(s), 0.0])
        dh[2] = 0.0
        dh /= np.linalg.norm(dh) + 1e-12
        lh = np.array([-dh[1], dh[0], 0.0])
        vl, vs = float(vp @ dh), float(vp @ lh)
        mu = leg.mu_roll + (leg.mu_brake * float(np.clip(brake, 0, 1)) if leg.brakes else 0.0)
        F_long = -mu * N * math.tanh(vl / leg.v_reg)
        slip = math.atan2(vs, abs(vl) + 1.0)
        F_lat = -leg.mu_side * N * math.tanh(slip / leg.slip_ref)
        Fn = np.array([0.0, 0.0, -N]) + F_long * dh + F_lat * lh
        Fb = C.T @ Fn
        F += Fb
        M += np.cross(r, Fb)
        loads.append(N)
    return F, M, loads
