"""Interconnect delay of critical nets: a driver, a distributed RC wire as an RC ladder and a load; the Elmore delay
(first moment of the impulse response) and a SPICE-like transient (modified nodal analysis, trapezoidal rule, the
default integration of SPICE) that checks it.

Wire capacitance per length: Sakurai and Tamaru, IEEE Trans Electron Devices 30:183 (1983),
``C / eps = 1.15 (w / h) + 2.80 (t / h)^0.222`` (a line over a ground plane). Resistance ``rho(T) / (w t)`` with
``rho(T) = rho_20 (1 + alpha (T - 20))``. The Elmore delay is an upper bound of the 50 % delay of an RC tree's step
response (Gupta, Tutuianu and Pileggi, IEEE TCAD 16:95, 1997) and equals the integral of ``1 - v(t) / Vdd``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg as sla

__all__ = ["WireSpec", "wire_rc", "rc_ladder", "elmore_delay", "transient_step", "net_delay"]

EPS0 = 8.8541878128e-12


@dataclass(frozen=True)
class WireSpec:
    """A global (upper) metal line: width, thickness, height over the plane below, copper, low-k dielectric; a driver
    resistance and a load capacitance (receiver gate). Representative values, not a foundry's."""
    width_m: float = 1.0e-6
    thickness_m: float = 2.0e-6
    height_m: float = 1.0e-6
    rho_ohm_m: float = 2.2e-8
    alpha_per_k: float = 0.0039
    eps_r: float = 3.0
    r_driver_ohm: float = 100.0
    c_load_f: float = 20e-15
    segments: int = 50


def wire_rc(length_m: float, spec: WireSpec, t_c: float = 25.0) -> tuple[float, float]:
    """Total resistance (ohm) and capacitance (F) of a wire."""
    rho = spec.rho_ohm_m * (1 + spec.alpha_per_k * (t_c - 20.0))
    r = rho * length_m / (spec.width_m * spec.thickness_m)
    c_per_m = EPS0 * spec.eps_r * (1.15 * spec.width_m / spec.height_m
                                   + 2.80 * (spec.thickness_m / spec.height_m) ** 0.222)
    return r, c_per_m * length_m


def rc_ladder(r_wire: float, c_wire: float, *, r_driver: float, c_load: float, n: int = 50):
    """Series resistances and node capacitances of driver + n-segment pi-free ladder (C of each segment lumped at
    its far node) + load at the last node."""
    R = np.concatenate([[r_driver], np.full(n, r_wire / n)])
    C = np.concatenate([[0.0], np.full(n, c_wire / n)])
    C[-1] += c_load
    return R, C


def elmore_delay(R: np.ndarray, C: np.ndarray) -> float:
    """Elmore delay at the last node of a chain: ``sum_i R_i sum_{j >= i} C_j``."""
    down = np.cumsum(C[::-1])[::-1]
    return float(np.sum(R * down))


def transient_step(R: np.ndarray, C: np.ndarray, *, t_end: float, dt: float, vdd: float = 1.0) -> dict[str, Any]:
    """Step response of the chain by MNA and the trapezoidal rule. Node 0 is the driver output (behind R[0]); a node
    without capacitance is solved algebraically. Returns the far-node voltage, its 50 % crossing and the first moment
    ``int (1 - v / vdd) dt``."""
    n = R.size
    G = np.zeros((n, n))
    g = 1.0 / R
    G[0, 0] += g[0]
    for i in range(1, n):
        G[i - 1, i - 1] += g[i]
        G[i, i] += g[i]
        G[i - 1, i] -= g[i]
        G[i, i - 1] -= g[i]
    b = np.zeros(n)
    b[0] = g[0] * vdd
    Cm = np.diag(C)
    A = Cm / dt + G / 2
    B = Cm / dt - G / 2
    lu = sla.lu_factor(A)
    steps = int(np.ceil(t_end / dt))
    v = np.zeros(n)
    out = np.empty(steps + 1)
    out[0] = 0.0
    for k in range(steps):
        v = sla.lu_solve(lu, B @ v + b)
        out[k + 1] = v[-1]
    t = np.arange(steps + 1) * dt
    k50 = int(np.argmax(out >= 0.5 * vdd))
    t50 = float(np.interp(0.5 * vdd, out[k50 - 1:k50 + 1], t[k50 - 1:k50 + 1])) if k50 > 0 else float("nan")
    y = 1 - out / vdd
    m1 = float(np.sum(0.5 * (y[1:] + y[:-1])) * dt)
    return {"t": t, "v": out, "t50_s": t50, "first_moment_s": m1, "settled": float(out[-1] / vdd)}


def net_delay(length_m: float, spec: WireSpec | None = None, t_c: float = 25.0) -> dict[str, float]:
    """Elmore delay and its single-pole 50 % estimate ``ln 2 tau`` for a driver-wire-load net."""
    spec = spec or WireSpec()
    r, c = wire_rc(length_m, spec, t_c)
    R, C = rc_ladder(r, c, r_driver=spec.r_driver_ohm, c_load=spec.c_load_f, n=spec.segments)
    tau = elmore_delay(R, C)
    return {"length_mm": length_m * 1e3, "r_wire_ohm": r, "c_wire_f": c, "elmore_s": tau,
            "t50_single_pole_s": float(np.log(2) * tau)}
