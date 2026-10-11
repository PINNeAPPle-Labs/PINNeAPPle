"""Damped harmonic oscillator: integrators against the exact solution."""
from __future__ import annotations

import math

import numpy as np

from ..spec import Experiment, register


def exact(t, zeta, omega, x0=1.0, v0=0.0):
    if zeta < 1:
        wd = omega * math.sqrt(1 - zeta * zeta)
        a = x0
        b = (v0 + zeta * omega * x0) / wd
        return np.exp(-zeta * omega * t) * (a * np.cos(wd * t) + b * np.sin(wd * t))
    if zeta == 1:
        return np.exp(-omega * t) * (x0 + (v0 + omega * x0) * t)
    s = math.sqrt(zeta * zeta - 1)
    r1, r2 = -omega * (zeta - s), -omega * (zeta + s)
    c2 = (v0 - r1 * x0) / (r2 - r1)
    return (x0 - c2) * np.exp(r1 * t) + c2 * np.exp(r2 * t)


def integrate(method, zeta, omega, dt, t_end, x0=1.0, v0=0.0):
    n = int(round(t_end / dt))
    x, v = x0, v0
    xs = np.empty(n + 1)
    xs[0] = x

    def acc(x, v):
        return -2 * zeta * omega * v - omega * omega * x

    for i in range(n):
        if method == "euler":
            x, v = x + dt * v, v + dt * acc(x, v)
        elif method == "symplectic":
            v = v + dt * acc(x, v)
            x = x + dt * v
        else:  # rk4
            k1x, k1v = v, acc(x, v)
            k2x, k2v = v + 0.5 * dt * k1v, acc(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v)
            k3x, k3v = v + 0.5 * dt * k2v, acc(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v)
            k4x, k4v = v + dt * k3v, acc(x + dt * k3x, v + dt * k3v)
            x = x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
            v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        xs[i + 1] = x
    return np.linspace(0, n * dt, n + 1), xs


@register
class Oscillator(Experiment):
    name = "oscillator"
    version = "1"
    description = "Damped harmonic oscillator x'' + 2 zeta omega x' + omega^2 x = 0: integrator vs exact solution."
    tags = ["ode", "verification"]
    params = {"zeta": 0.1, "omega": 2.0, "method": "rk4", "dt": 0.01, "t_end": 20.0, "seed": 0}
    space = {"zeta": (0.0, 1.5), "omega": ("log", 0.5, 10.0), "method": ["rk4", "euler", "symplectic"],
             "dt": ("log", 0.001, 0.05)}

    def run(self, ctx):
        p = ctx.params
        ctx.input("params", p)
        with ctx.stage("integrate"):
            t, x = integrate(p["method"], p["zeta"], p["omega"], p["dt"], p["t_end"])
        ref = exact(t, p["zeta"], p["omega"])
        err = float(np.sqrt(np.mean((x - ref) ** 2)))
        ctx.output("t", t)
        ctx.output("x", x)
        ctx.metric("rmse", err)
        ctx.metric("max_abs_error", float(np.max(np.abs(x - ref))))
        stable = bool(np.all(np.isfinite(x)) and np.max(np.abs(x)) < 1e3)
        ctx.check("finite_and_bounded", stable)
        h = p["omega"] * p["dt"]
        # expected global error: O(h^4) for RK4, O(h) for the first-order schemes (with a generous constant)
        tol = {"rk4": 10 * h ** 4, "symplectic": 2 * h, "euler": 10 * h}[p["method"]]
        ctx.metric("omega_dt", h)
        ctx.check("accuracy_for_method", value=err, max=max(tol, 1e-9), detail=f"rmse <= {tol:.2g} ({p['method']}) vs the exact solution",
                  kind="reference")
        regime = "underdamped" if p["zeta"] < 1 else ("critical" if p["zeta"] == 1 else "overdamped")
        ds = ctx.dataset("trajectories", description="Oscillator trajectories x(t) with parameters and regime",
                         units={"t": "s", "x": "m"})
        ds.add(t=t.astype(np.float32), x=x.astype(np.float32), x_exact=ref.astype(np.float32),
               zeta=float(p["zeta"]), omega=float(p["omega"]), method=p["method"], regime=regime)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 2.6))
        ax.plot(t, ref, "k-", lw=1.2, label="exact")
        ax.plot(t, x, "--", color="#d95f02", lw=1.2, label=p["method"])
        ax.set_xlabel("t")
        ax.set_title(f"zeta={p['zeta']:.3g} omega={p['omega']:.3g} dt={p['dt']:.3g}  rmse={err:.2e}", fontsize=9)
        ax.legend(frameon=False, fontsize=8)
        ctx.figure("trajectory", fig)
