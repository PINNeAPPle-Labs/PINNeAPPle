"""The plant: a pumped, heated water loop with an elbow and a T junction, and its reference simulator.

    reservoir -> pump (speed n) -> heater (duty Qh) -> pipe A (with a 90° elbow) -> T junction
                                                          |-> branch B (pipe, valve uB) -> tank at z_B
                                                          '-> branch C (pipe, valve uC) -> tank at z_C

Quasi-steady hydraulics (operator moves are slow against the acoustic time 2L/a, see :func:`quasi_steady_valid`):
pump head ``H = n^2 H0 - k_p Q^2`` (affinity laws), Darcy-Weisbach pipes with the Colebrook friction factor
(``pinneapple_physics.closed_form.pipe_flow``), minor losses of the elbow, the tee and the valves (``K_v(u) = K_v0 /
u^2``); the junction head and the two branch flows are solved by Newton each time step. Energy: the heater raises the
water by ``Qh / (m cp)``; each branch loses heat to the room through ``UA`` (``T_out = T_room + (T_A - T_room)
exp(-UA / (m cp))``).

Ten sensors (pressures in bar gauge, flows in m3/h, temperatures in °C) with Gaussian noise; faults are injected on
the *truth* (a partial obstruction: extra loss coefficient in a pipe) or on the *readings* (a sensor bias drift, a
noisier sensor), so every test case has a known answer.

Water properties: White, *Fluid Mechanics* 7th ed., Table A.1 (density, viscosity, 0-100 °C).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from pinneapple_physics.closed_form.pipe_flow import friction_factor

G = 9.80665
CP_WATER = 4180.0          # J/(kg K), 20-60 °C (White Table A.1: 4.18-4.19 kJ/kg K)
_T = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], float)
_RHO = np.array([1000.0, 1000.0, 998.0, 996.0, 992.0, 988.0, 983.0, 978.0, 972.0, 965.0, 958.0])
_MU = np.array([1.788, 1.307, 1.003, 0.799, 0.657, 0.548, 0.467, 0.405, 0.355, 0.316, 0.283]) * 1e-3

SENSORS = ("P_pump", "P_tee", "P_B", "P_C", "Q_A", "Q_B", "Q_C", "T_A", "T_B", "T_C")
UNITS = {"P_pump": "bar", "P_tee": "bar", "P_B": "bar", "P_C": "bar", "Q_A": "m3/h", "Q_B": "m3/h",
         "Q_C": "m3/h", "T_A": "°C", "T_B": "°C", "T_C": "°C"}
INPUTS = ("n", "u_B", "u_C", "T_in", "Q_heat_kW")
#: operating envelope the twin's surrogate is trained on (the plant can be driven outside it)
NORMAL_BOX = {"n": (0.80, 1.00), "u_B": (0.35, 1.00), "u_C": (0.35, 1.00), "T_in": (15.0, 35.0),
              "Q_heat_kW": (300.0, 900.0)}


def water(t_c: float) -> tuple[float, float]:
    """(density kg/m3, dynamic viscosity Pa s) at ``t_c`` °C."""
    return float(np.interp(t_c, _T, _RHO)), float(np.interp(t_c, _T, _MU))


@dataclass
class Pipe:
    length: float          # m
    diameter: float        # m
    roughness: float = 4.5e-5   # commercial steel (White Table 6.1)
    k_minor: float = 0.0   # elbows, tee branch
    ua_w_k: float = 0.0    # heat-loss conductance to the room

    @property
    def area(self) -> float:
        return math.pi * self.diameter ** 2 / 4

    def k_total(self, q: float, rho: float, mu: float, k_extra: float = 0.0) -> float:
        """Loss coefficient referred to this pipe's velocity head at flow ``q``: f L/D + minor + extra."""
        v = abs(q) / self.area
        re = max(rho * v * self.diameter / mu, 1.0)
        return friction_factor(re, self.roughness / self.diameter) * self.length / self.diameter \
            + self.k_minor + k_extra

    def head_loss(self, q: float, rho: float, mu: float, k_extra: float = 0.0) -> float:
        v = q / self.area
        return self.k_total(q, rho, mu, k_extra) * v * abs(v) / (2 * G)

    def flow_from_head_loss(self, dh: float, rho: float, mu: float, k_extra: float = 0.0) -> float:
        """Inverse of :meth:`head_loss` (fixed point on the friction factor)."""
        if dh <= 0:
            return 0.0
        q = self.area * math.sqrt(2 * G * dh / max(self.k_total(1e-3, rho, mu, k_extra), 1e-9))
        for _ in range(30):
            q_new = self.area * math.sqrt(2 * G * dh / self.k_total(q, rho, mu, k_extra))
            if abs(q_new - q) < 1e-12:
                break
            q = q_new
        return q


@dataclass
class Loop:
    """Geometry and machines of the loop (SI)."""
    pipe_A: Pipe = field(default_factory=lambda: Pipe(60.0, 0.100, k_minor=0.9 + 0.3, ua_w_k=0.0))   # elbow + entry
    pipe_B: Pipe = field(default_factory=lambda: Pipe(40.0, 0.080, k_minor=1.0 + 0.3, ua_w_k=900.0))  # tee branch + elbow
    pipe_C: Pipe = field(default_factory=lambda: Pipe(35.0, 0.080, k_minor=0.4, ua_w_k=700.0))       # tee run
    h0_m: float = 32.0           # pump shut-off head at n = 1
    k_pump: float = 9000.0       # m / (m3/s)^2
    kv0_B: float = 3.0           # valve loss coefficient fully open (referred to the branch velocity)
    kv0_C: float = 3.0
    z_B: float = 6.0             # m, delivery tank levels above the pump
    z_C: float = 3.0
    t_room_c: float = 22.0
    p_atm_head_m: float = 0.0    # gauge
    wave_speed: float = 1200.0   # m/s, steel pipe full of water (for the quasi-steady validity check)

    def pump_head(self, q: float, n: float) -> float:
        return n * n * self.h0_m - self.k_pump * q * abs(q)


@dataclass
class Faults:
    """What is wrong at one instant (all zero: healthy)."""
    k_extra: dict[str, float] = field(default_factory=lambda: {"A": 0.0, "B": 0.0, "C": 0.0})   # obstruction
    bias: dict[str, float] = field(default_factory=dict)         # sensor -> additive bias (sensor unit)
    noise_gain: dict[str, float] = field(default_factory=dict)   # sensor -> noise multiplier


def _solve(loop: Loop, u: dict[str, float], ke: dict[str, float], rho: float, mu: float):
    A, B, C = loop.pipe_A, loop.pipe_B, loop.pipe_C
    kvB = loop.kv0_B / max(u["u_B"], 0.02) ** 2
    kvC = loop.kv0_C / max(u["u_C"], 0.02) ** 2

    def branch_head(pipe, q, kv, k):            # head drop tee -> tank (pipe + valve)
        return pipe.head_loss(q, rho, mu, k) + kv * (q / pipe.area) * abs(q / pipe.area) / (2 * G)

    def res(x):
        qB, qC = x
        qA = qB + qC
        h_tee = loop.pump_head(qA, u["n"]) - A.head_loss(qA, rho, mu, ke["A"])
        return np.array([h_tee - branch_head(B, qB, kvB, ke["B"]) - loop.z_B,
                         h_tee - branch_head(C, qC, kvC, ke["C"]) - loop.z_C])

    x = np.array([0.012, 0.015])
    for _ in range(60):                          # damped Newton, finite-difference Jacobian
        r = res(x)
        J = np.empty((2, 2))
        for j in range(2):
            dx = np.zeros(2)
            dx[j] = 1e-7
            J[:, j] = (res(x + dx) - r) / 1e-7
        step = np.linalg.solve(J, -r)
        lam = 1.0
        while lam > 1e-3 and np.linalg.norm(res(np.maximum(x + lam * step, 1e-6))) > np.linalg.norm(r):
            lam *= 0.5
        x = np.maximum(x + lam * step, 1e-6)
        if np.linalg.norm(step) < 1e-12:
            break
    return x, float(np.abs(res(x)).max()), kvB, kvC


def steady_state(loop: Loop, u: dict[str, float], k_extra: dict[str, float] | None = None) -> dict[str, float]:
    """True plant state for inputs ``u`` (keys of INPUTS): flows, heads, pressures and temperatures. Water
    properties at the mean of the inlet and heater-outlet temperatures (two passes)."""
    ke = {"A": 0.0, "B": 0.0, "C": 0.0, **(k_extra or {})}
    A, B, C = loop.pipe_A, loop.pipe_B, loop.pipe_C
    t_prop = u["T_in"]
    for _ in range(2):
        rho, mu = water(min(t_prop, 99.0))
        (qB, qC), resid, kvB, kvC = _solve(loop, u, ke, rho, mu)
        qA = qB + qC
        T_A = u["T_in"] + u["Q_heat_kW"] * 1e3 / (rho * qA * CP_WATER)
        t_prop = 0.5 * (u["T_in"] + T_A)
    h_pump = loop.pump_head(qA, u["n"])
    h_tee = h_pump - A.head_loss(qA, rho, mu, ke["A"])
    h_B = loop.z_B + kvB * (qB / B.area) ** 2 / (2 * G)          # upstream of valve B
    h_C = loop.z_C + kvC * (qC / C.area) ** 2 / (2 * G)

    def t_out(pipe, q):
        return loop.t_room_c + (T_A - loop.t_room_c) * math.exp(-pipe.ua_w_k / (rho * q * CP_WATER))

    bar = rho * G / 1e5
    return {"Q_A": qA, "Q_B": qB, "Q_C": qC, "H_pump": h_pump, "H_tee": h_tee, "H_B": h_B, "H_C": h_C,
            "P_pump": h_pump * bar, "P_tee": h_tee * bar, "P_B": h_B * bar, "P_C": h_C * bar,
            "T_A": T_A, "T_B": t_out(B, qB), "T_C": t_out(C, qC), "rho": rho, "mu": mu, "t_props_c": t_prop,
            "newton_residual_m": resid, "v_A": qA / A.area, "v_B": qB / B.area, "v_C": qC / C.area}


def readings(state: dict[str, float]) -> np.ndarray:
    """True sensor values in sensor units (bar, m3/h, °C), order SENSORS."""
    return np.array([state["P_pump"], state["P_tee"], state["P_B"], state["P_C"], state["Q_A"] * 3600,
                     state["Q_B"] * 3600, state["Q_C"] * 3600, state["T_A"], state["T_B"], state["T_C"]])


#: one-sigma sensor noise (sensor units): pressure transmitters 0.005 bar, flow meters 0.5 % of a 60 m3/h span
#: (0.3 m3/h), RTDs 0.05 K -- typical instrument classes, an assumption of the demo, not a measured plant
SIGMA = np.array([0.005, 0.005, 0.005, 0.005, 0.3, 0.3, 0.3, 0.05, 0.05, 0.05])


def quasi_steady_valid(loop: Loop, stroke_s: float, factor: float = 10.0) -> bool:
    """The quasi-steady model holds when a valve moves over many acoustic round trips: stroke >= factor 2L/a."""
    L = loop.pipe_A.length + max(loop.pipe_B.length, loop.pipe_C.length)
    return stroke_s >= factor * 2 * L / loop.wave_speed


# ----------------------------------------------------------------------------- scenarios
@dataclass
class Scenario:
    """A controlled test case: an operating history and (optionally) one fault with its onset and ramp."""
    name: str
    kind: str                      # "normal" | "process" | "sensor" | "domain"
    seed: int
    duration_s: float = 3600.0
    dt_s: float = 10.0
    onset_s: float = 1800.0
    ramp_s: float = 300.0
    target: str = ""               # branch (process), sensor (sensor), input (domain)
    magnitude: float = 0.0         # k_extra, bias in sensor units, noise gain, or input value reached
    mode: str = ""                 # sensor: "drift" | "noise"

    def truth(self) -> dict[str, Any]:
        return {"kind": self.kind, "target": self.target, "magnitude": self.magnitude, "mode": self.mode,
                "onset_s": self.onset_s}


def operating_history(sc: Scenario) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Slow, smooth operator moves inside NORMAL_BOX (sum of a few low-frequency sines per input, seeded); a domain
    scenario then drives its ``target`` input to ``magnitude`` (outside the box) from the onset."""
    rng = np.random.default_rng(sc.seed)
    t = np.arange(0.0, sc.duration_s + 1e-9, sc.dt_s)
    u: dict[str, np.ndarray] = {}
    for k in INPUTS:
        lo, hi = NORMAL_BOX[k]
        mid, amp = 0.5 * (lo + hi), 0.5 * (hi - lo)
        s = np.zeros_like(t)
        for _ in range(3):
            period = rng.uniform(900.0, 4000.0)
            s += rng.uniform(0.2, 0.45) * np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))
        u[k] = mid + amp * np.clip(s, -0.95, 0.95)
    if sc.kind == "domain":
        r = np.clip((t - sc.onset_s) / sc.ramp_s, 0.0, 1.0)
        u[sc.target] = u[sc.target] + r * (sc.magnitude - u[sc.target])
    return t, u


def run_plant(loop: Loop, sc: Scenario) -> dict[str, Any]:
    """Simulate one scenario: inputs, true sensor values, noisy readings and the fault applied at each step."""
    t, u = operating_history(sc)
    rng = np.random.default_rng(sc.seed + 7919)
    Y_true, Y = np.empty((len(t), len(SENSORS))), np.empty((len(t), len(SENSORS)))
    newton = 0.0
    for i, ti in enumerate(t):
        r = float(np.clip((ti - sc.onset_s) / sc.ramp_s, 0.0, 1.0)) if sc.kind in ("process", "sensor") else 0.0
        k_extra = {sc.target: r * sc.magnitude} if sc.kind == "process" else None
        st = steady_state(loop, {k: float(v[i]) for k, v in u.items()}, k_extra)
        newton = max(newton, st["newton_residual_m"])
        y = readings(st)
        Y_true[i] = y
        sig = SIGMA.copy()
        bias = np.zeros(len(SENSORS))
        if sc.kind == "sensor":
            j = SENSORS.index(sc.target)
            if sc.mode == "drift":
                bias[j] = r * sc.magnitude
            else:
                sig[j] *= 1.0 + r * (sc.magnitude - 1.0)
        Y[i] = y + bias + rng.normal(0.0, 1.0, len(SENSORS)) * sig
    return {"t": t, "u": u, "y_true": Y_true, "y": Y, "scenario": sc, "newton_max_residual_m": newton}


def with_loop(loop: Loop, **kw) -> Loop:
    return replace(loop, **kw)
