"""Time integration of an :class:`~.aircraft.Aircraft` with a controller and a wind field, and the recorded history.

    hist = simulate(ac, x0, controller, t_end=30.0, dt=0.01, wind=field)

``controller(t, x, last)`` returns :class:`~.aircraft.Controls` held over the step (zero-order hold); ``last`` is the
previous step's evaluation (or None). RK4 with quaternion renormalisation. Everything a figure, a table or a 3-D scene
needs is recorded per step (or every ``record_every`` steps): state, Euler angles, airspeed, alpha, beta, load factor
at the CG and at the cabin stations, gear loads, controls, wind at the CG and, when ``record_gamma``, the lattice
circulation (for the surface loading in the viewer).
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .aircraft import Aircraft, Controls
from .rigid_body import G0, IDX_QUAT, kinetic_energy, quat_to_euler, rk4_step


class _StepWind:
    """Wind at the aircraft's sample points at the start and (predicted) end of a step, interpolated linearly in
    time for the RK stages: second-order in the step, two field evaluations per step instead of four."""

    def __init__(self, w0: np.ndarray, w1: np.ndarray, t0: float, dt: float):
        self.w0, self.w1, self.t0, self.dt = w0, w1, t0, dt

    def velocity(self, t, points):
        f = (t - self.t0) / self.dt
        return self.w0 + f * (self.w1 - self.w0)


@dataclass
class FlightHistory:
    aircraft: str
    t: np.ndarray
    x: np.ndarray                                   # (n, 13)
    euler: np.ndarray                               # (n, 3) phi, theta, psi [rad]
    airspeed: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    CL: np.ndarray
    nz: np.ndarray                                  # vertical load factor at the CG (1 in level flight)
    ny: np.ndarray                                  # lateral load factor at the CG
    f_spec: np.ndarray                              # (n, 3) specific force at the CG, body axes [m/s^2]
    thrust: np.ndarray
    gear_loads: np.ndarray                          # (n, legs) [N]
    controls: dict[str, np.ndarray]
    wind_cg: np.ndarray                             # (n, 3) wind at the CG, body axes
    cabin_nz: dict[str, np.ndarray] = field(default_factory=dict)
    cabin_ny: dict[str, np.ndarray] = field(default_factory=dict)
    gamma: np.ndarray | None = None              # (n, panels) circulation [m^2/s]
    events: dict[str, Any] = field(default_factory=dict)

    @property
    def position(self) -> np.ndarray:
        return self.x[:, 0:3]

    @property
    def altitude(self) -> np.ndarray:
        return -self.x[:, 2]

    @property
    def ground_speed(self) -> np.ndarray:
        from .rigid_body import quat_to_dcm
        return np.array([np.linalg.norm(quat_to_dcm(q) @ v) for q, v in zip(self.x[:, 6:10], self.x[:, 3:6], strict=True)])

    def summary(self) -> dict[str, float]:
        return {"t_end": float(self.t[-1]), "nz_min": float(self.nz.min()), "nz_max": float(self.nz.max()),
                "alpha_max_deg": float(np.degrees(self.alpha.max())), "V_min": float(self.airspeed.min())}


def simulate(ac: Aircraft, x0: np.ndarray, controller: Callable[[float, np.ndarray, dict | None], Controls],
             t_end: float, dt: float = 0.01, wind=None, record_every: int = 1, record_gamma: bool = False,
             stop: Callable[[float, np.ndarray, dict], bool] | None = None, t0: float = 0.0,
             hold_wind: bool = True) -> FlightHistory:
    """``hold_wind``: sample the wind at the aircraft's sample points at the start of each step and at their
    predicted end-of-step positions, and interpolate linearly in time over the RK stages (two field evaluations per
    step instead of four; second order in the step). False samples the field at every stage."""
    x = np.asarray(x0, float).copy()
    n = int(round((t_end - t0) / dt))
    rec: dict[str, list] = {k: [] for k in ("t", "x", "euler", "V", "alpha", "beta", "CL", "f", "thrust", "gear",
                                            "wind", "gamma")}
    ctl: dict[str, list] = {k: [] for k in ("elevator", "aileron", "rudder", "flap", "throttle", "steer", "brake")}
    cab = {k: [] for k in ac.cabin_b}
    last = None
    t = t0
    for k in range(n + 1):
        u = controller(t, x, last)
        ev = ac.evaluate(t, x, u, wind)
        last = ev
        if k % record_every == 0:
            ae = ev["aero"]
            rec["t"].append(t)
            rec["x"].append(x.copy())
            rec["euler"].append(quat_to_euler(x[IDX_QUAT]))
            rec["V"].append(ae["V"])
            rec["alpha"].append(ae["alpha"])
            rec["beta"].append(ae["beta"])
            rec["CL"].append(ae["CL"])
            rec["f"].append(ev["f_spec"])
            rec["thrust"].append(ev["thrust"])
            rec["gear"].append(ev["gear_loads"])
            rec["wind"].append(np.zeros(3) if ev["wind_cg_b"] is None else ev["wind_cg_b"])
            if record_gamma:
                rec["gamma"].append(ae["gamma"].astype(np.float32))
            for c in ctl:
                ctl[c].append(getattr(ev["controls"], c))
            for s, r in ac.cabin_b.items():
                cab[s].append(ac.specific_force_at(ev, x, r))
        if stop is not None and stop(t, x, ev):
            break
        if k == n:
            break
        w_step = wind
        if hold_wind and ev.get("wind_samples") is not None:
            x_end = x + dt * ev["dx"]                               # Euler prediction of the end-of-step positions
            w_step = _StepWind(ev["wind_samples"], wind.velocity(t + dt, ac.sample_points(x_end)), t, dt)
        x = rk4_step(lambda tt, xx, u=u, w=w_step: ac.evaluate(tt, xx, u, w)["dx"], t, x, dt)
        t = t0 + (k + 1) * dt
        if not np.all(np.isfinite(x)):
            raise FloatingPointError(f"state diverged at t = {t:.3f} s")
    f = np.array(rec["f"])
    hist = FlightHistory(
        aircraft=ac.name, t=np.array(rec["t"]), x=np.array(rec["x"]), euler=np.array(rec["euler"]),
        airspeed=np.array(rec["V"]), alpha=np.array(rec["alpha"]), beta=np.array(rec["beta"]), CL=np.array(rec["CL"]),
        nz=-f[:, 2] / G0, ny=f[:, 1] / G0, f_spec=f, thrust=np.array(rec["thrust"]),
        gear_loads=np.array(rec["gear"]).reshape(len(rec["t"]), -1), controls={k: np.array(v) for k, v in ctl.items()},
        wind_cg=np.array(rec["wind"]),
        cabin_nz={k: -np.array(v)[:, 2] / G0 for k, v in cab.items()},
        cabin_ny={k: np.array(v)[:, 1] / G0 for k, v in cab.items()},
        gamma=np.array(rec["gamma"]) if record_gamma else None)
    return hist


def energy_check(ac: Aircraft, x0: np.ndarray, t_end: float = 5.0, dt: float = 0.01) -> dict[str, float]:
    """Energy and momentum consistency of the integrator with the aerodynamics, engines and gear off (a body in
    vacuum under gravity, spinning): kinetic + potential energy and the angular momentum in NED are conserved."""
    from .rigid_body import angular_momentum_ned, gravity_body, rigid_body_derivative

    def f(t, x):
        return rigid_body_derivative(x, gravity_body(x[IDX_QUAT], ac.mass), np.zeros(3), ac.mass, ac.inertia,
                                     ac.inertia_inv)

    x = np.asarray(x0, float).copy()
    E0 = kinetic_energy(x, ac.mass, ac.inertia) - ac.mass * G0 * x[2]
    H0 = angular_momentum_ned(x, ac.inertia)
    t = 0.0
    while t < t_end - 1e-12:
        x = rk4_step(f, t, x, dt)
        t += dt
    E1 = kinetic_energy(x, ac.mass, ac.inertia) - ac.mass * G0 * x[2]
    H1 = angular_momentum_ned(x, ac.inertia)
    return {"energy_rel_error": abs(E1 - E0) / abs(E0), "momentum_rel_error": float(np.linalg.norm(H1 - H0) /
                                                                                     np.linalg.norm(H0)),
            "quat_norm_error": abs(float(np.linalg.norm(x[IDX_QUAT])) - 1.0), "t_end": t_end, "dt": dt}


def rms(a) -> float:
    a = np.asarray(a, float)
    return float(math.sqrt(np.mean((a - a.mean()) ** 2)))
