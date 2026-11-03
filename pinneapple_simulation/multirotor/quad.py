"""A quadrotor in X configuration: mass build-up, 6-DOF rigid-body dynamics with rotor lag, an LQR hover controller
with position integrators, its stability margins, and a mission flown on the battery until it is empty.

Frames: world z up (gravity ``-g e_z``), body z along the thrust. State: position, velocity (world), attitude
quaternion ``(w, x, y, z)`` body-to-world, body rates, rotor speeds. Rotor ``i`` sits at ``L (cos psi_i, sin psi_i)``
with ``psi = 45, 135, 225, 315 deg`` and spins ``s_i = +1, -1, +1, -1`` (counter-clockwise seen from above for
``+1``); its thrust ``T = c_T(J) rho n^2 D^4`` and torque ``Q = c_P(J) rho n^2 D^5 / (2 pi)`` come from a BEMT table
over the axial advance ratio ``J = V_axial / (n D)`` (:func:`.rotor.bemt`), the yaw reaction is ``-s_i Q_i``, the
rotor speed follows its command with the motor time constant ``tau_motor``. Body drag ``-1/2 rho C_d A |v - w| (v -
w)`` with the wind ``w(t)``. Edgewise (forward-flight) rotor effects are not modelled: the cruise power is that of
the axial-flow rotor (stated with the result).

LQR (:func:`pinneapple_systems.control.tuning.lqr`) on the hover linearisation ``p' = v``, ``v' = (g theta,
-g phi, dF / m)``, ``Theta' = omega``, ``omega' = J^-1 tau``, with integrators of the position error, so a steady wind
leaves no offset. Margins: each input channel of the loop is broken in turn with the others closed, the motor lag
included (:func:`loop_margins_per_channel`).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .power import Battery, Motor
from .rotor import RHO_SEA_LEVEL, Rotor, bemt

__all__ = ["Quad", "rotor_table", "lqr_hover", "loop_margins_per_channel", "simulate", "Segment", "fly_mission",
           "hover_power"]

G = 9.80665
PSI = np.radians([45.0, 135.0, 225.0, 315.0])
SPIN = np.array([1.0, -1.0, 1.0, -1.0])


def rotor_table(rotor: Rotor, rpm: float, j_max: float = 0.6, n: int = 13, rho: float = RHO_SEA_LEVEL) -> dict:
    """Propeller ``c_T`` and ``c_P`` against the axial advance ratio ``J`` at ``rpm`` (BEMT)."""
    D = 2 * rotor.radius
    nrev = rpm / 60.0
    js = np.linspace(0.0, j_max, n)
    rows = [bemt(rotor, 2 * math.pi * nrev, j * nrev * D, rho, n=24) for j in js]
    return {"J": js, "cT": np.array([r["cT_prop"] for r in rows]), "cP": np.array([r["cP_prop"] for r in rows]),
            "rpm": rpm, "D": D}


@dataclass
class Quad:
    """A quadrotor: ``rotor`` with its ``table`` (:func:`rotor_table`), ``motor``, ``battery``, arm length ``arm``
    [m], total ``mass`` [kg] and its ``parts``, inertia ``J`` [kg m^2], body drag area ``cd_area`` [m^2]."""
    rotor: Rotor
    table: Mapping[str, Any]
    motor: Motor
    battery: Battery
    arm: float
    mass: float
    J: np.ndarray
    cd_area: float = 0.02
    tau_motor: float = 0.03
    esc_efficiency: float = 0.95
    avionics_W: float = 5.0
    rho: float = RHO_SEA_LEVEL
    parts: dict = field(default_factory=dict)

    @property
    def D(self) -> float:
        return 2 * self.rotor.radius

    def coefficients(self, J):
        t = self.table
        return float(np.interp(J, t["J"], t["cT"])), float(np.interp(J, t["J"], t["cP"]))

    def thrust_torque(self, omega: float, v_axial: float) -> tuple[float, float]:
        n = max(omega, 1e-6) / (2 * math.pi)
        J = max(v_axial, 0.0) / (n * self.D)
        cT, cP = self.coefficients(J)
        T = cT * self.rho * n * n * self.D ** 4
        Q = cP * self.rho * n * n * self.D ** 5 / (2 * math.pi)
        return T, Q

    def omega_for_thrust(self, thrust: float) -> float:
        cT0 = float(self.table["cT"][0])
        return 2 * math.pi * math.sqrt(max(thrust, 0.0) / (cT0 * self.rho * self.D ** 4))

    def mixer(self) -> np.ndarray:
        """``[F, tau_x, tau_y, tau_z] = M T`` for the four rotor thrusts (hover yaw ratio ``Q / T``)."""
        x, y = self.arm * np.cos(PSI), self.arm * np.sin(PSI)
        cT0, cP0 = float(self.table["cT"][0]), float(self.table["cP"][0])
        kappa = cP0 * self.D / (2 * math.pi * cT0)
        return np.vstack([np.ones(4), y, -x, -SPIN * kappa])

    def omega_max(self, voltage: float) -> float:
        """Rotor speed the motor reaches at ``voltage`` in hover: ``(Q / K_t + I_0) R + K_t omega = V`` with
        ``Q = k_Q omega^2``."""
        cP0 = float(self.table["cP"][0])
        kq = cP0 * self.rho * self.D ** 5 / (2 * math.pi) / (2 * math.pi) ** 2
        m = self.motor
        a, b, c = kq * m.r_ohm / m.kt, m.kt, m.i0 * m.r_ohm - voltage
        return (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)


def hover_power(q: Quad, soc: float = 0.8) -> dict:
    """Electrical hover power [W] at ``soc``: rotor speed for ``m g / 4``, the motor's current and voltage, the ESC
    efficiency and the avionics; the battery current and the thrust-to-weight at full voltage."""
    w = q.omega_for_thrust(q.mass * G / 4)
    _, Q = q.thrust_torque(w, 0.0)
    op = q.motor.operating_point(Q, w)
    p = 4 * op["power_W"] / q.esc_efficiency + q.avionics_W
    i = q.battery.current_for_power(soc, p)
    vt = q.battery.terminal_voltage(soc, i) if math.isfinite(i) else float("nan")
    wmax = q.omega_max(0.95 * vt) if math.isfinite(vt) else float("nan")
    tw = 4 * q.thrust_torque(wmax, 0.0)[0] / (q.mass * G) if math.isfinite(wmax) else float("nan")
    return {"omega": w, "rpm": w * 60 / (2 * math.pi), "torque_Nm": Q, "motor": op, "power_W": p, "current_A": i,
            "voltage_V": vt, "thrust_to_weight": tw, "shaft_W": 4 * Q * w}


# ------------------------------------------------------------------ control
def lqr_hover(q: Quad, *, q_pos: float = 4.0, q_vel: float = 1.0, q_ang: float = 4.0, q_rate: float = 2.0,
              q_int: float = 0.5, r_force: float = 0.5, r_torque: float = 3000.0) -> dict:
    """LQR gains on the 15-state hover model (position, velocity, roll-pitch-yaw, body rates, position-error
    integrals) with inputs ``[dF, tau_x, tau_y, tau_z]`` (diagonal weights; the defaults were chosen on the default
    design for >= 45 deg phase margin per channel with a 30 ms motor lag and a 1 m step overshoot under 30 %)."""
    from pinneapple_systems.control.tuning import lqr
    A = np.zeros((15, 15))
    B = np.zeros((15, 4))
    A[0:3, 3:6] = np.eye(3)
    A[3, 7], A[4, 6] = G, -G                   # x'' = g theta, y'' = -g phi
    A[6:9, 9:12] = np.eye(3)
    A[12:15, 0:3] = np.eye(3)                  # integral of the position error
    B[5, 0] = 1.0 / q.mass
    Jinv = np.linalg.inv(q.J)
    B[9:12, 1:4] = Jinv
    Q = np.diag([q_pos] * 3 + [q_vel] * 3 + [q_ang] * 3 + [q_rate] * 3 + [q_int] * 3)
    R = np.diag([r_force, r_torque, r_torque, r_torque])
    k = lqr(A, B, Q, R)
    return {"K": k["K"], "A": A, "B": B, "poles": k["poles"]}


def loop_margins_per_channel(ctrl: Mapping[str, Any], tau_motor: float) -> list[dict]:
    """Gain and phase margins at each input channel, the loop broken there with the other channels closed and the
    motor lag ``1 / (tau s + 1)`` in every channel: ``L_j(s) = K_j (sI - A_j)^-1 B_j / (tau s + 1)``."""
    from pinneapple_systems.control.analysis import loop_margins
    A, B, K = ctrl["A"], ctrl["B"], ctrl["K"]
    n, m = A.shape[0], B.shape[1]
    # augment the lag of the closed channels into the plant: u_j_actual' = (u_cmd - u_actual) / tau
    out = []
    for j in range(m):
        others = [i for i in range(m) if i != j]
        Aa = np.zeros((n + m, n + m))
        Aa[:n, :n] = A
        Aa[:n, n:] = B
        Aa[n:, n:] = -np.eye(m) / tau_motor
        for i in others:                          # closed channels: u_cmd_i = -K_i x
            Aa[n + i, :n] = -K[i] / tau_motor
        Bj = np.zeros(n + m)
        Bj[n + j] = 1.0 / tau_motor
        Kj = np.r_[K[j], np.zeros(m)]

        def L(w, Aa=Aa, Bj=Bj, Kj=Kj):
            return np.array([Kj @ np.linalg.solve(1j * wk * np.eye(n + m) - Aa, Bj) for wk in np.atleast_1d(w)])
        mg = loop_margins(L, w_min=1e-3, w_max=1e3, n=1500)
        up, down = _gain_margins(L)
        out.append({"channel": ["thrust", "roll", "pitch", "yaw"][j], "phase_margin": mg["phase_margin"],
                    "gain_crossover": mg["gain_crossover"], "ms": mg["ms"], "gain_margin_up_db": up,
                    "gain_margin_down_db": down})
    return out


def _gain_margins(L, w_min: float = 1e-3, w_max: float = 1e3, n: int = 3000) -> tuple[float, float]:
    """Upward and downward gain margins [dB] from every -180 deg crossing of ``L(jw)``: a crossing with ``|L| < 1``
    bounds how much the gain may grow, one with ``|L| > 1`` (loops with integrators are conditionally stable) how much
    it may shrink. ``inf`` when there is no such crossing."""
    w = np.logspace(np.log10(w_min), np.log10(w_max), n)
    Lw = L(w)
    ph = np.unwrap(np.angle(Lw))
    k = np.floor((ph + np.pi) / (2 * np.pi))
    up, down = float("inf"), float("inf")
    for i in np.nonzero(k[1:] != k[:-1])[0]:
        target = max(k[i], k[i + 1]) * 2 * np.pi - np.pi
        f = (target - ph[i]) / (ph[i + 1] - ph[i])
        mag = float(np.exp(np.log(abs(Lw[i])) + f * (np.log(abs(Lw[i + 1])) - np.log(abs(Lw[i])))))
        db = -20 * math.log10(mag)
        if mag < 1:
            up = min(up, db)
        else:
            down = min(down, -db)
    return up, down


# ------------------------------------------------------------------ dynamics
def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _rot(qt):
    w, x, y, z = qt
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _euler(qt):
    w, x, y, z = qt
    phi = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    th = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    psi = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([phi, th, psi])


@dataclass
class Segment:
    """A mission leg: hold ``target`` (world position [m]) for ``duration`` [s], reached along a straight line at
    ``speed`` [m/s] (the reference moves, the controller tracks it)."""
    name: str
    target: Sequence[float]
    duration: float
    speed: float = 2.0


def simulate(q: Quad, ctrl: Mapping[str, Any], *, t_end: float, dt: float = 0.01,
             reference: Callable[[float], np.ndarray] | None = None,
             wind: Callable[[float], np.ndarray] | None = None, battery: bool = False, soc0: float = 1.0,
             record_every: int = 5, stop_on_empty: bool = True) -> dict:
    """Integrate the 6-DOF quadrotor (RK4 on the rigid body, exact first-order rotor lag) under the LQR. With
    ``battery`` the rotor speeds are limited by the pack's terminal voltage and the state of charge is drawn down;
    the run stops when the pack is empty (cut-off voltage or reserve). Returns the time history and the energy
    drawn."""
    K = ctrl["K"]
    Minv = np.linalg.inv(q.mixer())
    Jm, Jinv = q.J, np.linalg.inv(q.J)
    ref = reference or (lambda t: np.zeros(3))
    wnd = wind or (lambda t: np.zeros(3))
    w_h = q.omega_for_thrust(q.mass * G / 4)
    p, v, qt, om = np.zeros(3), np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(3)
    p = np.array(ref(0.0), float)
    rot_w = np.full(4, w_h)
    integ = np.zeros(3)
    soc, e_drawn = soc0, 0.0
    a_lag = math.exp(-dt / q.tau_motor)
    hist = {k: [] for k in ("t", "p", "ref", "euler", "rotor_rpm", "power_W", "soc", "voltage_V", "saturated")}
    t, k = 0.0, 0
    empty = False
    sat_count = 0

    def deriv(p_, v_, qt_, om_, thrusts, torques, t_):
        R = _rot(qt_)
        F = R @ np.array([0.0, 0.0, thrusts.sum()])
        rel = v_ - wnd(t_)
        drag = -0.5 * q.rho * q.cd_area * np.linalg.norm(rel) * rel
        acc = F / q.mass + drag / q.mass - np.array([0.0, 0.0, G])
        x, y = q.arm * np.cos(PSI), q.arm * np.sin(PSI)
        tau = np.array([np.sum(y * thrusts), -np.sum(x * thrusts), -np.sum(SPIN * torques)])
        domega = Jinv @ (tau - np.cross(om_, Jm @ om_))
        dq = 0.5 * _qmul(qt_, np.r_[0.0, om_])
        return v_, acc, dq, domega

    while t < t_end:
        r = np.asarray(ref(t), float)
        v_ref = (np.asarray(ref(t + dt), float) - r) / dt            # reference velocity (feed-forward)
        e = p - r
        ang = _euler(qt)
        x = np.r_[e, v - v_ref, ang, om, integ]
        u = -K @ x
        Fz = q.mass * G + u[0]
        cmdT = Minv @ np.array([Fz, u[1], u[2], u[3]])
        w_cmd = np.array([q.omega_for_thrust(max(c, 0.0)) for c in cmdT])
        vt = None
        if battery:
            vt_guess = q.battery.terminal_voltage(soc, 0.0)
            wmax = q.omega_max(0.95 * vt_guess)
            sat = bool(np.any(w_cmd > wmax))
            sat_count += sat
            w_cmd = np.minimum(w_cmd, wmax)
        rot_w = w_cmd + (rot_w - w_cmd) * a_lag
        v_ax = float((_rot(qt) @ np.array([0, 0, 1.0])) @ (v - wnd(t)))
        tt = np.array([q.thrust_torque(w, v_ax) for w in rot_w])
        thrusts, torques = tt[:, 0], tt[:, 1]
        # RK4 on the rigid body with the rotor forces held over the step
        s0 = (p, v, qt, om)
        k1 = deriv(*s0, thrusts, torques, t)
        s1 = [a + 0.5 * dt * b for a, b in zip(s0, k1, strict=True)]
        k2 = deriv(*s1, thrusts, torques, t + dt / 2)
        s2 = [a + 0.5 * dt * b for a, b in zip(s0, k2, strict=True)]
        k3 = deriv(*s2, thrusts, torques, t + dt / 2)
        s3 = [a + dt * b for a, b in zip(s0, k3, strict=True)]
        k4 = deriv(*s3, thrusts, torques, t + dt)
        p, v, qt, om = (a + dt / 6 * (b + 2 * c + 2 * d + f) for a, b, c, d, f in zip(s0, k1, k2, k3, k4, strict=True))
        qt = qt / np.linalg.norm(qt)
        integ = integ + dt * e
        pw = q.avionics_W + sum(q.motor.operating_point(Qi, wi)["power_W"] for Qi, wi in zip(torques, rot_w,
                                                                                          strict=True)) / q.esc_efficiency
        if battery:
            i = q.battery.current_for_power(soc, pw)
            if not math.isfinite(i) or q.battery.empty(soc, i):
                empty = True
            else:
                vt = q.battery.terminal_voltage(soc, i)
                soc = q.battery.draw(soc, i, dt)
                e_drawn += pw * dt
        if k % record_every == 0 or empty:
            hist["t"].append(t)
            hist["p"].append(p.copy())
            hist["ref"].append(r)
            hist["euler"].append(_euler(qt))
            hist["rotor_rpm"].append(rot_w * 60 / (2 * math.pi))
            hist["power_W"].append(pw)
            hist["soc"].append(soc)
            hist["voltage_V"].append(vt if vt is not None else float("nan"))
            hist["saturated"].append(sat_count)
        t += dt
        k += 1
        if empty and stop_on_empty:
            break
        if not np.all(np.isfinite(p)) or abs(p[2]) > 1e4:
            break
    out = {k: np.asarray(v) for k, v in hist.items()}
    out.update(t_final=t, empty=empty, energy_Wh=e_drawn / 3600.0, soc_final=soc, saturated_steps=sat_count)
    return out


def fly_mission(q: Quad, ctrl: Mapping[str, Any], segments: Sequence[Segment], *, repeat: bool = True,
                dt: float = 0.02, t_max: float = 7200.0, wind=None) -> dict:
    """Fly ``segments`` (repeated while ``repeat``) on the battery from full until it is empty: the endurance is the
    time the run lasted. The reference moves to each target at the segment's speed, then holds it."""
    legs, t0, start = [], 0.0, np.zeros(3)
    total = sum(s.duration for s in segments)
    for s in segments:
        tgt = np.asarray(s.target, float)
        legs.append((t0, t0 + s.duration, start.copy(), tgt, s.speed))
        t0 += s.duration
        start = tgt

    def ref(t):
        tt = t % total if repeat else min(t, total - 1e-9)
        for a, b, p0, p1, sp in legs:
            if a <= tt < b:
                d = np.linalg.norm(p1 - p0)
                f = min(1.0, (tt - a) * sp / d) if d > 0 else 1.0
                return p0 + f * (p1 - p0)
        return legs[-1][3]
    run = simulate(q, ctrl, t_end=t_max if repeat else total, dt=dt, reference=ref, wind=wind, battery=True,
                   record_every=25)
    err = np.linalg.norm(run["p"] - run["ref"], axis=1) if len(run["p"]) else np.array([np.nan])
    return {"endurance_s": float(run["t_final"]), "empty": bool(run["empty"]), "energy_Wh": run["energy_Wh"],
            "mean_power_W": float(np.mean(run["power_W"])) if len(run["power_W"]) else float("nan"),
            "max_tracking_error_m": float(np.nanmax(err)), "saturated_steps": int(run["saturated_steps"]),
            "laps": float(run["t_final"] / total), "history": run}
