"""Pilot and autopilot models: pitch-attitude, altitude, speed, wings-level / heading hold, yaw damper, and the
take-off schedule (ground roll on the centreline, rotation at V_R, climb-out at a target speed).

The inner loops are designed per aircraft from its own control derivatives (:func:`.trim.stability_derivatives`) and
inertias, and scheduled with the dynamic pressure: the elevator for a pitch loop of natural frequency w and damping
zeta is

    d_e = d_e,trim + I_yy (w^2 (theta_c - theta) - 2 zeta w q + w_i int(theta_c - theta)) / (q S c Cm_de)

(roll with I_xx, Cl_da, b; yaw with I_zz, Cn_dr, b). The airframe's own stiffness and damping add to these, so the
closed loop is at least as stiff and damped as designed. Gains use a floor on the dynamic pressure so they stay finite
at low speed (the surfaces then saturate at their limits).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .aircraft import Aircraft, Controls
from .rigid_body import G0, quat_to_dcm, quat_to_euler

KT = 0.514444


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class LoopDesign:
    omega_pitch: float = 1.5
    zeta_pitch: float = 0.8
    omega_roll: float = 2.0
    zeta_roll: float = 0.8
    omega_yaw: float = 1.0
    zeta_yaw: float = 0.7
    pitch_integral: float = 0.3          # rad/s, integral gain relative to omega^2
    q_floor: float = 300.0               # Pa


class InnerLoops:
    """Attitude loops of one aircraft: :meth:`elevator`, :meth:`aileron`, :meth:`rudder` give surface increments."""

    def __init__(self, ac: Aircraft, derivs: dict[str, float], design: LoopDesign | None = None):
        self.ac, self.d, self.k = ac, derivs, design or LoopDesign()
        self.I = np.diag(ac.inertia)
        self.int_theta = 0.0

    def _q(self, ev) -> float:
        return max(ev["aero"]["q"], self.k.q_floor) if ev is not None else self.k.q_floor

    def elevator(self, ev, theta_c, theta, q, dt) -> float:
        k = self.k
        e = theta_c - theta
        self.int_theta = float(np.clip(self.int_theta + e * dt, -0.2, 0.2))
        M = self.I[1] * (k.omega_pitch ** 2 * (e + k.pitch_integral * self.int_theta) - 2 * k.zeta_pitch *
                         k.omega_pitch * q)
        return M / (self._q(ev) * self.ac.af.wing_area * self.ac.af.mac * self.d["Cm_de"])

    def aileron(self, ev, phi_c, phi, p) -> float:
        k = self.k
        L = self.I[0] * (k.omega_roll ** 2 * (phi_c - phi) - 2 * k.zeta_roll * k.omega_roll * p)
        return L / (self._q(ev) * self.ac.af.wing_area * self.ac.af.span * self.d["Cl_da"])

    def rudder(self, ev, psi_err, r) -> float:
        k = self.k
        N = self.I[2] * (k.omega_yaw ** 2 * psi_err - 2 * k.zeta_yaw * k.omega_yaw * r)
        return N / (self._q(ev) * self.ac.af.wing_area * self.ac.af.span * self.d["Cn_dr"])


class StickFixed:
    """Controls frozen at the trim values."""

    def __init__(self, controls: Controls):
        self.u = controls

    def __call__(self, t, x, last) -> Controls:
        return self.u


class Autopilot:
    """Altitude hold (through pitch attitude), wings level / heading hold (through bank), yaw damper (washed-out yaw
    rate), autothrottle on airspeed, about a trim state. ``loops`` scales the attitude bandwidths."""

    def __init__(self, ac: Aircraft, trim, derivs: dict[str, float], design: LoopDesign | None = None,
                 altitude: float | None = None, heading: float | None = None, speed: float | None = None,
                 omega_h: float = 0.25, tau_heading: float = 8.0, washout: float = 2.0, autothrottle: bool = True):
        self.ac, self.tr = ac, trim
        self.loops = InnerLoops(ac, derivs, design)
        self.h_c = trim.altitude if altitude is None else altitude
        self.psi_c = quat_to_euler(trim.x[6:10])[2] if heading is None else heading
        self.V_c = trim.V if speed is None else speed
        self.omega_h, self.tau_h, self.tw = omega_h, tau_heading, washout
        self.autothrottle = autothrottle
        self.t_last = None
        self.r_lp = 0.0
        self.int_v = 0.0
        T = sum(e.thrust(trim.V, *ac.atmosphere(trim.x)[:2], 1.0) for e in ac.engines) or 1.0
        self.k_v = 0.15 * ac.mass / T

    def __call__(self, t, x, last) -> Controls:
        dt = 0.0 if self.t_last is None else t - self.t_last
        self.t_last = t
        phi, theta, psi = quat_to_euler(x[6:10])
        p, q, r = x[10:13]
        V = last["aero"]["V"] if last is not None else self.tr.V
        vned = quat_to_dcm(x[6:10]) @ x[3:6]
        h, hdot = -x[2], -vned[2]
        gam = math.asin(np.clip(hdot / max(V, 1.0), -1, 1))
        gam_c = float(np.clip(self.omega_h * (self.h_c - h) / max(V, 1.0), -0.08, 0.08))
        theta_c = self.tr.theta + gam_c + (gam_c - gam)
        de = self.tr.elevator + self.loops.elevator(last, theta_c, theta, q, dt)
        phi_c = float(np.clip(V / (G0 * self.tau_h) * wrap(self.psi_c - psi), -math.radians(25), math.radians(25)))
        da = self.loops.aileron(last, phi_c, phi, p)
        # washed-out yaw rate: r_hp = r - low-pass(r)
        if dt > 0:
            self.r_lp += dt / self.tw * (r - self.r_lp)
        dr = self.loops.rudder(last, 0.0, r - self.r_lp)
        thr = self.tr.throttle
        if self.autothrottle:
            self.int_v = float(np.clip(self.int_v + (self.V_c - V) * dt, -50, 50))
            thr = thr + self.k_v * ((self.V_c - V) + 0.1 * self.int_v)
        return Controls(elevator=de, aileron=da, rudder=dr, throttle=float(np.clip(thr, 0, 1)),
                        flap=self.tr.controls.flap, gear_down=self.tr.controls.gear_down)


@dataclass
class TakeoffSchedule:
    V_R: float                            # rotation speed (airspeed) [m/s]
    theta_rotation: float                 # pitch attitude target of the rotation [rad]
    rotation_rate: float                  # [rad/s]
    V_climb: float                        # climb-out airspeed [m/s]
    theta_max: float = math.radians(18)
    theta_min: float = math.radians(2)
    runway_heading: float = 0.0
    flap: float = 0.0
    throttle: float = 1.0
    h_climb: float = 3.0                  # m above the runway: speed-on-pitch from here on


class TakeoffPilot:
    """Full throttle and the take-off flap from brake release; on the ground the nose wheel and the rudder hold the
    runway centreline (y = 0 in NED for a north runway, ``runway_heading`` otherwise) and the ailerons keep the wings
    level; at airspeed V_R the pitch attitude is commanded up at ``rotation_rate`` to ``theta_rotation``; once
    airborne and ``h_climb`` above the runway the pitch attitude is adjusted (at most 2 deg/s) to fly ``V_climb``;
    in the air the track is held along the runway by bank and the yaw damper keeps the turn coordinated. Records the
    events (t, x, V at V_R, lift-off) in ``events``."""

    def __init__(self, ac: Aircraft, sched: TakeoffSchedule, derivs: dict[str, float],
                 design: LoopDesign | None = None):
        self.ac, self.s = ac, sched
        self.loops = InnerLoops(ac, derivs, design)
        self.phase = "roll"
        self.events: dict[str, Any] = {}
        self.t_last = None
        self.theta_c = None
        self.t_air = None
        self.r_lp = 0.0
        self.int_y = 0.0

    def __call__(self, t, x, last) -> Controls:
        s = self.s
        dt = 0.0 if self.t_last is None else t - self.t_last
        self.t_last = t
        phi, theta, psi = quat_to_euler(x[6:10])
        p, q, r = x[10:13]
        V = last["aero"]["V"] if last is not None else 0.0
        on_ground = last is None or any(n > 0 for n in last["gear_loads"])
        c, sn = math.cos(s.runway_heading), math.sin(s.runway_heading)
        y = -sn * x[0] + c * x[1]                                   # lateral offset from the centreline (right)
        vned = quat_to_dcm(x[6:10]) @ x[3:6]
        ydot = -sn * vned[0] + c * vned[1]
        h = -x[2]
        if self.phase == "roll" and V >= s.V_R:
            self.phase = "rotate"
            self.theta_c = theta
            self.events.update(t_R=t, x_R=x.copy(), V_R=V)
        if self.phase == "rotate" and not on_ground:
            if self.t_air is None:
                self.t_air = t
                self.events.update(t_LOF=t, x_LOF=x.copy(), V_LOF=V, theta_LOF=theta,
                                   alpha_LOF=last["aero"]["alpha"])
            if h >= s.h_climb:
                self.phase = "climb"
        elif self.phase == "rotate" and on_ground and self.t_air is not None and t - self.t_air < 1.0:
            self.t_air = None                                       # bounced back: lift-off not final
            self.events.pop("t_LOF", None)
        u = Controls(throttle=s.throttle, flap=s.flap)
        # lateral: heading towards the centreline
        V_s = max(V, 5.0)
        self.int_y = float(np.clip(self.int_y + y * dt, -50.0, 50.0))
        dpsi = -float(np.clip((y + 1.5 * ydot + 0.15 * self.int_y) / (1.5 * V_s + 10.0), -0.25, 0.25))
        psi_c = s.runway_heading + dpsi
        if on_ground:
            lim = math.radians(40) / (1 + (V / 10.0) ** 2) + math.radians(2)
            u.steer = float(np.clip(2.0 * wrap(psi_c - psi) - 0.6 * r, -lim, lim))
            u.rudder = self.loops.rudder(last, wrap(psi_c - psi), r)
            u.aileron = self.loops.aileron(last, 0.0, phi, p)
        else:
            chi = math.atan2(vned[1], vned[0])
            phi_c = float(np.clip(V_s / (G0 * 6.0) * wrap(psi_c - chi), -math.radians(15), math.radians(15)))
            u.aileron = self.loops.aileron(last, phi_c, phi, p)
            if dt > 0:
                self.r_lp += dt / 2.0 * (r - self.r_lp)
            u.rudder = self.loops.rudder(last, 0.0, r - self.r_lp)
        # longitudinal
        if self.phase == "roll":
            u.elevator = 0.0
            self.loops.int_theta = 0.0
        else:
            if self.phase == "rotate":
                self.theta_c = min(self.theta_c + s.rotation_rate * dt, s.theta_rotation)
            else:
                rate = float(np.clip(0.05 * (V - s.V_climb), -1, 1)) * math.radians(2)
                self.theta_c = float(np.clip(self.theta_c + rate * dt, s.theta_min, s.theta_max))
            u.elevator = self.loops.elevator(last, self.theta_c, theta, q, dt)
        return u
