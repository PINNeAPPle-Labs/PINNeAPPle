"""Water hammer in a pumped pipeline: steady operating point and the valve-closure transient by the method of
characteristics (MOC).

    line = Pipeline(length=1200, diameter=0.5, wave_speed=1100, roughness=4.5e-5)
    pump = PumpCurve(shutoff_head=60.0, k=40.0)          # H = 60 - 40 Q|Q|  [m, m^3/s]
    op = operating_point(line, pump, valve_k=...)        # pump curve = friction + valve loss
    res = moc_valve_closure(line, pump, op, closure_time=2.0, t_end=20.0)

Model (Wylie & Streeter, *Fluid Transients in Systems*, 1993, ch. 3): 1-D elastic pipe, quasi-steady Darcy-Weisbach
friction with the steady friction factor (Colebrook), pump with a fixed speed head curve at the upstream end drawing
from a sump, a valve discharging into a tank at the downstream end, ``Q = tau * Cv * sqrt(dH)`` with a closure law
``tau(t)``. Characteristics ``C+ : H_P = C_P - B Q_P``, ``C- : H_P = C_M + B Q_P`` with ``B = a / (g A)`` and
``R = f dx / (2 g D A^2)``; the grid satisfies the Courant condition ``dt = dx / a`` exactly.

Verification (``joukowsky_head_rise``, ``wave_period``): for an instantaneous closure without friction the head rise at
the valve is ``a V0 / g`` (Joukowsky) and the pressure history has the period ``4 L / a``; the MOC reproduces both to
round-off on its grid, which the tests check.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pinneapple_physics.closed_form.pipe_flow import friction_factor

G = 9.80665

__all__ = ["Pipeline", "PumpCurve", "OperatingPoint", "TransientResult", "operating_point", "moc_valve_closure",
           "joukowsky_head_rise", "wave_period", "linear_closure"]


@dataclass
class Pipeline:
    length: float               # m
    diameter: float             # m
    wave_speed: float           # m/s (water in a steel pipe: about 1000-1400)
    roughness: float = 4.5e-5   # m (commercial steel)
    z_in: float = 0.0           # elevation of the pump end [m]
    z_out: float = 0.0          # elevation of the valve end [m]
    nu: float = 1.0e-6          # kinematic viscosity [m^2/s]
    rho: float = 998.0          # kg/m^3

    @property
    def area(self) -> float:
        return np.pi * self.diameter ** 2 / 4


@dataclass
class PumpCurve:
    """``H = shutoff_head - k Q|Q|`` (fixed speed); ``sump_head``: suction-side head [m]."""
    shutoff_head: float
    k: float
    sump_head: float = 0.0

    def head(self, q):
        q = np.asarray(q, float)
        return self.sump_head + self.shutoff_head - self.k * q * np.abs(q)


@dataclass
class OperatingPoint:
    """Steady operating point of the line.

    Attributes
    ----------
    flow : float [m3/s]
    velocity : float [m/s]
    reynolds : float [-]
    friction_factor : float [-]
    pump_head : float [m]
        Head at the pump outlet.
    valve_head : float [m]
        Head upstream of the valve.
    friction_loss : float [m]
    valve_cv : float [m^2.5/s]
        ``Q = Cv sqrt(H_valve - H_tank)`` fully open.
    """
    flow: float                 # m^3/s
    velocity: float             # m/s
    reynolds: float
    friction_factor: float
    pump_head: float            # m (head at the pump outlet)
    valve_head: float           # m (head upstream of the valve)
    friction_loss: float        # m
    valve_cv: float             # m^2.5/s: Q = Cv sqrt(H_valve - H_tank) fully open
    tank_head: float


def operating_point(line: Pipeline, pump: PumpCurve, *, valve_k: float, tank_head: float = 0.0) -> OperatingPoint:
    """Flow where the pump curve meets friction (Colebrook) + the open valve's loss ``valve_k * V^2 / (2 g)``."""
    from scipy.optimize import brentq
    A, D, L = line.area, line.diameter, line.length

    def balance(q):
        v = q / A
        f = friction_factor(max(v * D / line.nu, 1.0), line.roughness / D)
        return pump.head(q) - (tank_head + (line.z_out - line.z_in) + f * L / D * v * v / (2 * G)
                               + valve_k * v * v / (2 * G))

    qmax = np.sqrt(max(pump.shutoff_head + pump.sump_head - tank_head, 1e-9) / pump.k)
    q = brentq(balance, 1e-9, qmax)
    v = q / A
    re = v * D / line.nu
    f = friction_factor(re, line.roughness / D)
    hf = f * L / D * v * v / (2 * G)
    hp = float(pump.head(q))
    hv = hp - hf
    dh = hv - tank_head
    return OperatingPoint(q, v, re, f, hp, hv, hf, q / np.sqrt(dh), tank_head)


def linear_closure(closure_time: float, start: float = 0.0):
    """``tau(t)``: 1 until ``start``, linearly to 0 over ``closure_time`` (0 = instantaneous)."""
    def tau(t):
        if closure_time <= 0:
            return 1.0 if t < start else 0.0
        return float(np.clip(1.0 - (t - start) / closure_time, 0.0, 1.0))
    return tau


def _signed_root(k: float, b: float, c: float) -> float:
    """Root of ``k Q|Q| + b Q + c = 0`` (``k >= 0``, ``b > 0``: one real root)."""
    if k <= 0:
        return -c / b
    if c <= 0:
        return (-b + np.sqrt(b * b - 4 * k * c)) / (2 * k)
    return (b - np.sqrt(b * b + 4 * k * c)) / (2 * k)


@dataclass
class TransientResult:
    """Method-of-characteristics run of a valve closure.

    Attributes
    ----------
    head : np.ndarray [m]
        Piezometric head, (nt, nx).
    flow : np.ndarray [m3/s]
        (nt, nx).
    tau : np.ndarray [-]
        Valve opening.
    courant : float [-]
    """
    t: np.ndarray               # (nt,)
    x: np.ndarray               # (nx,)
    head: np.ndarray            # (nt, nx) piezometric head [m]
    flow: np.ndarray            # (nt, nx) m^3/s
    tau: np.ndarray             # (nt,) valve opening
    dt: float
    courant: float
    op: OperatingPoint
    info: dict = field(default_factory=dict)

    def pressure(self, line: Pipeline, z=None) -> np.ndarray:
        """Gauge pressure [Pa] from the head and the pipe elevation (linear between the ends)."""
        z = np.interp(self.x, [0, line.length], [line.z_in, line.z_out]) if z is None else z
        return line.rho * G * (self.head - z[None, :])


def moc_valve_closure(line: Pipeline, pump: PumpCurve, op: OperatingPoint, *, closure_time: float, t_end: float,
                      n_reaches: int = 40, friction: bool = True, closure_start: float = 0.0,
                      save_every: int = 1) -> TransientResult:
    """Valve closure transient from the steady operating point ``op`` (pump upstream, valve + tank downstream)."""
    A, D, a = line.area, line.diameter, line.wave_speed
    N = int(n_reaches)
    dx = line.length / N
    dt = dx / a
    B = a / (G * A)
    R = (op.friction_factor if friction else 0.0) * dx / (2 * G * D * A * A)
    x = np.linspace(0.0, line.length, N + 1)
    Q = np.full(N + 1, op.flow)
    H = op.pump_head - (op.pump_head - op.valve_head) * x / line.length
    if not friction:                 # frictionless reference: flat head line at the valve head
        H = np.full(N + 1, op.valve_head)
    tau = linear_closure(closure_time, closure_start)
    nt = int(np.ceil(t_end / dt))
    Ts, Hs, Qs, taus = [0.0], [H.copy()], [Q.copy()], [1.0]
    for n in range(1, nt + 1):
        t = n * dt
        CP = H[:-1] + B * Q[:-1] - R * Q[:-1] * np.abs(Q[:-1])        # from the left neighbour
        CM = H[1:] - B * Q[1:] + R * Q[1:] * np.abs(Q[1:])            # from the right neighbour
        Hn, Qn = np.empty_like(H), np.empty_like(Q)
        Hn[1:-1] = 0.5 * (CP[:-1] + CM[1:])
        Qn[1:-1] = (CP[:-1] - CM[1:]) / (2 * B)
        # upstream: pump (frictionless reference: constant head reservoir at the valve head)
        if friction:
            Qn[0] = _signed_root(pump.k, B, CM[0] - pump.sump_head - pump.shutoff_head)
            Hn[0] = CM[0] + B * Qn[0]
        else:
            Hn[0] = op.valve_head
            Qn[0] = (Hn[0] - CM[0]) / B
        # downstream: valve into the tank, Q = tau Cv sqrt(H - H_tank)
        ta = tau(t)
        cv = ta * op.valve_cv
        if cv <= 0:
            Qn[-1] = 0.0
        else:
            Qn[-1] = _signed_root(1.0 / cv ** 2, B, op.tank_head - CP[-1])
        Hn[-1] = CP[-1] - B * Qn[-1]
        H, Q = Hn, Qn
        if n % save_every == 0:
            Ts.append(t)
            Hs.append(H.copy())
            Qs.append(Q.copy())
            taus.append(ta)
    return TransientResult(np.asarray(Ts), x, np.asarray(Hs), np.asarray(Qs), np.asarray(taus), dt * save_every,
                           a * dt / dx, op, {"B": B, "R": R, "dx": dx, "friction": friction})


def joukowsky_head_rise(wave_speed: float, dv: float) -> float:
    """``a dV / g`` [m]: head rise of an instantaneous velocity change ``dv`` (Joukowsky).

    Returns
    -------
    dH : float [m]
        Head rise.
    """
    return wave_speed * dv / G


def wave_period(length: float, wave_speed: float) -> float:
    """``4 L / a`` [s]: period of the pressure oscillation after a closure (pipe from a reservoir).

    Returns
    -------
    T : float [s]
        Period.
    """
    return 4.0 * length / wave_speed
