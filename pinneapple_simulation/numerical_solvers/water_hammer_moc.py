"""Water hammer in a reservoir - pipe - valve system by the method of characteristics (MOC).

One-dimensional unsteady pipe flow with an elastic pipe and a slightly compressible liquid (Wylie & Streeter,
*Fluid Transients in Systems*, 1993, ch. 3): along ``dx/dt = +-a`` the compatibility equations

    C+:  H_P = C_P - B Q_P,   C_P = H_A + B Q_A - R Q_A |Q_A|
    C-:  H_P = C_M + B Q_P,   C_M = H_B - B Q_B + R Q_B |Q_B|

with ``B = a / (g A)`` and ``R = f dx / (2 g D A^2)`` (Darcy friction, steady formulation), on a grid with
``dt = dx / a`` (Courant number 1). Boundaries: a constant-head reservoir upstream; downstream a valve discharging to a
constant head ``h_downstream`` with the orifice law ``Q = tau Q_0 sqrt(dH / dH_0)`` and the closure law
``tau(t) = (1 - t / t_c)^E_m``. Reverse flow through the valve is allowed.

Limits: no column-separation (cavity) model; once the head falls to vapour pressure the computed history after that
instant is not physical, and ``min_head`` reports where it happened. Steady friction only (unsteady friction damps
the later peaks; the first peak, which sizes the pipe, is not affected).

Verification: instantaneous closure of a frictionless pipe gives the Joukowsky surge ``a V_0 / g`` exactly
(``tests/test_reason.py``).
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["valve_closure_moc", "wave_speed", "joukowsky_head", "michaud_head"]

G = 9.80665


def wave_speed(K: float, rho: float, D: float, e: float, E: float, c1: float = 1.0) -> float:
    """Pressure-wave speed in a liquid-filled elastic pipe (Korteweg; Wylie & Streeter eq. 2.29):
    ``a = sqrt((K/rho) / (1 + c1 K D / (E e)))``. ``c1 = 1 - nu^2`` for a pipe anchored against axial movement."""
    return math.sqrt((K / rho) / (1 + c1 * K * D / (E * e)))


def joukowsky_head(a: float, dv: float) -> float:
    """Head rise of an instantaneous velocity change: ``dH = a dV / g`` (Joukowsky 1898)."""
    return a * dv / G


def michaud_head(a: float, dv: float, length: float, t_close: float) -> float:
    """Michaud's (1878) estimate for a closure slower than ``2L/a``: ``dH = 2 L dV / (g t_c)``; equals Joukowsky
    for ``t_c <= 2L/a``."""
    return a * dv / G * min(1.0, 2 * length / (a * t_close))


def valve_closure_moc(*, length: float, diameter: float, wave_speed: float, friction_factor: float, q0: float,
                      h_reservoir: float, h_downstream: float = 0.0, t_close: float, closure_exponent: float = 1.0,
                      n_reaches: int = 20, t_end: float | None = None, record_every: int = 1,
                      record_field: bool = False) -> dict:
    """Transient heads and flows after the valve at the end of the pipe starts closing at t = 0.

    Returns time ``t``, the head and flow at the valve, the head envelopes ``h_max`` / ``h_min`` along ``x``, the
    steady line ``h_steady``, the overall ``max_head`` / ``min_head`` with their position and time, and the closed-form
    references (``joukowsky``, ``michaud``, ``period`` 2L/a). ``record_field``: also return ``history`` with the head
    along the pipe at every recorded step (``{"t": (T,), "h": (T, n_reaches + 1)}``)."""
    if min(length, diameter, wave_speed, q0) <= 0 or n_reaches < 2 or t_close < 0 or friction_factor < 0:
        raise ValueError("length, diameter, wave speed and flow must be positive, n_reaches >= 2, t_close >= 0")
    A = math.pi * diameter ** 2 / 4
    v0 = q0 / A
    dx = length / n_reaches
    dt = dx / wave_speed
    B = wave_speed / (G * A)
    R = friction_factor * dx / (2 * G * diameter * A ** 2)
    x = np.linspace(0.0, length, n_reaches + 1)
    H = h_reservoir - friction_factor * x / diameter * v0 ** 2 / (2 * G)
    Q = np.full(n_reaches + 1, q0)
    dh0 = H[-1] - h_downstream
    if dh0 <= 0:
        raise ValueError("the valve has no head across it in steady flow (h_reservoir - friction <= h_downstream)")
    h_steady = H.copy()
    period = 2 * length / wave_speed
    t_end = t_close + 5 * period if t_end is None else t_end
    n_steps = int(math.ceil(t_end / dt))
    hmax, hmin = H.copy(), H.copy()
    ts, hv, qv = [0.0], [H[-1]], [Q[-1]]
    hf = [H.copy()] if record_field else None
    imax = (float(H.max()), float(x[int(H.argmax())]), 0.0)
    imin = (float(H.min()), float(x[int(H.argmin())]), 0.0)
    for n in range(1, n_steps + 1):
        t = n * dt
        Hn, Qn = np.empty_like(H), np.empty_like(Q)
        cp = H[:-1] + B * Q[:-1] - R * Q[:-1] * np.abs(Q[:-1])          # from the left neighbour (C+)
        cm = H[1:] - B * Q[1:] + R * Q[1:] * np.abs(Q[1:])              # from the right neighbour (C-)
        Hn[1:-1] = 0.5 * (cp[:-1] + cm[1:])
        Qn[1:-1] = (cp[:-1] - cm[1:]) / (2 * B)
        Hn[0] = h_reservoir
        Qn[0] = (h_reservoir - cm[0]) / B
        tau = max(0.0, 1 - t / t_close) ** closure_exponent if t_close > 0 else 0.0
        Cp = cp[-1]
        if tau <= 0:
            q = 0.0
        else:
            cv = (tau * q0) ** 2 / dh0
            d = Cp - h_downstream
            if d >= 0:
                q = 0.5 * (-cv * B + math.sqrt((cv * B) ** 2 + 4 * cv * d))
            else:
                q = 0.5 * (cv * B - math.sqrt((cv * B) ** 2 - 4 * cv * d))
        Qn[-1] = q
        Hn[-1] = Cp - B * q
        H, Q = Hn, Qn
        np.maximum(hmax, H, out=hmax)
        np.minimum(hmin, H, out=hmin)
        i, j = int(H.argmax()), int(H.argmin())
        if H[i] > imax[0]:
            imax = (float(H[i]), float(x[i]), t)
        if H[j] < imin[0]:
            imin = (float(H[j]), float(x[j]), t)
        if n % record_every == 0:
            ts.append(t)
            hv.append(float(H[-1]))
            qv.append(float(Q[-1]))
            if hf is not None:
                hf.append(H.copy())
    out = {"t": np.array(ts), "h_valve": np.array(hv), "q_valve": np.array(qv), "x": x, "h_max": hmax,
            "h_min": hmin, "h_steady": h_steady, "max_head": imax[0], "max_head_at": imax[1], "max_head_time": imax[2],
            "min_head": imin[0], "min_head_at": imin[1], "min_head_time": imin[2], "v0": v0, "dt": dt,
            "period": period, "joukowsky": joukowsky_head(wave_speed, v0),
            "michaud": michaud_head(wave_speed, v0, length, t_close) if t_close > 0 else joukowsky_head(wave_speed, v0),
            "valve_head_steady": float(h_steady[-1])}
    if hf is not None:
        out["history"] = {"t": np.array(ts), "h": np.array(hf)}
    return out
