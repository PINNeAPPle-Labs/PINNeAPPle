"""Excitable media: FitzHugh-Nagumo (1-D) and Aliev-Panfilov (2-D) by finite differences, and activation maps.

Both are reference solvers (method of lines / explicit Euler on a uniform grid with no-flux boundaries) for the
physics-informed models that learn the same equations, with a grid-refinement check built in.

* :func:`fitzhugh_nagumo_1d` solves ``v_t = v_xx + v(1 - v)(v - a) - w + I``, ``w_t = eps (v - gamma w)`` on
  ``[0, L]`` (FitzHugh 1961, Nagumo et al. 1962, in the cubic form of the arena problem ``fitzhugh_nagumo_1d``),
  no-flux ends, with SciPy's BDF on a second-order central-difference grid.
* :func:`aliev_panfilov_2d` solves ``u_t = div(D grad u) + k u (1 - u)(u - a) - u v``,
  ``v_t = (eps0 + mu1 v / (u + mu2)) (-v - k u (u - a - 1))`` (Aliev and Panfilov, Chaos Solitons Fractals 7(3),
  1996, Eqs 1-2 and their parameter values k = 8, a = 0.15, eps0 = 0.002, mu1 = 0.2, mu2 = 0.3) on a rectangle with
  no-flux edges, a heterogeneous diffusivity map allowed (a scar: ``D = 0``), stimuli as (time window, region).
* :func:`activation_times` gives, per grid point, the first time the potential crosses a threshold upwards (linear
  interpolation between saved frames), the activation map used in cardiac mapping; :func:`conduction_velocity` its
  local speed ``1 / |grad T|``.

The model is dimensionless; Aliev and Panfilov map time units to milliseconds with ``t[ms] = 12.9 t`` and
``V[mV] = 100 u - 80``.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = ["fitzhugh_nagumo_1d", "FHNSolution", "aliev_panfilov_2d", "APSolution", "Stimulus", "activation_times",
           "conduction_velocity", "AP_PARAMETERS"]


# --------------------------------------------------------------------------- FitzHugh-Nagumo 1-D
@dataclass
class FHNSolution:
    """A FitzHugh-Nagumo run of ``v_t = v_xx + v(1 - v)(v - a) - w + I``: unit coefficients, so dimensionless.

    Attributes
    ----------
    x, t : np.ndarray [-]
        Grid and saved times.
    v, w : np.ndarray [-]
        Potential and recovery variable, ``(len(t), len(x))``.
    """
    x: np.ndarray
    t: np.ndarray
    v: np.ndarray            # (len(t), len(x))
    w: np.ndarray

    def interpolate(self, x, t) -> dict[str, np.ndarray]:
        """``v`` and ``w`` at scattered ``(x, t)`` (bicubic on the solution grid)."""
        from scipy.interpolate import RectBivariateSpline

        out = {}
        for k in ("v", "w"):
            spl = RectBivariateSpline(self.t, self.x, getattr(self, k), kx=3, ky=3)
            out[k] = spl.ev(np.asarray(t, float), np.asarray(x, float))
        return out


def fitzhugh_nagumo_1d(*, a: float = 0.1, eps: float = 0.01, gamma: float = 0.5, I: float = 0.0,  # noqa: E741  (the stimulus current's usual name)
                       L: float = 1.0, t_end: float = 2.0, nx: int = 401, nt: int = 201,
                       v0: Callable | None = None, w0: Callable | None = None,
                       rtol: float = 1e-9, atol: float = 1e-11) -> FHNSolution:
    """Method-of-lines solution on ``nx`` points (no-flux ends by mirror ghost points) saved at ``nt`` times.

    Default initial condition: the arena's front ``v0 = (1 + tanh(5 (x - L/2))) / 2`` and ``w0 = 0``."""
    from scipy.integrate import solve_ivp
    from scipy.sparse import diags

    x = np.linspace(0.0, L, nx)
    dx = x[1] - x[0]
    t = np.linspace(0.0, t_end, nt)
    v_init = v0(x) if v0 is not None else 0.5 * (1 + np.tanh(5 * (x - 0.5 * L)))
    w_init = w0(x) if w0 is not None else np.zeros(nx)
    main, off = -2.0 * np.ones(nx), np.ones(nx - 1)
    up, lo = off.copy(), off.copy()
    up[0], lo[-1] = 2.0, 2.0          # mirror ghost points: u_-1 = u_1, u_n = u_n-2
    Lap = diags([lo, main, up], [-1, 0, 1], format="csr") / dx**2

    def f(_, z):
        v, w = z[:nx], z[nx:]
        return np.concatenate([Lap @ v + v * (1 - v) * (v - a) - w + I, eps * (v - gamma * w)])

    def jac(_, z):
        from scipy.sparse import bmat
        v = z[:nx]
        dfv = diags(-3 * v**2 + 2 * (1 + a) * v - a)
        return bmat([[Lap + dfv, -diags(np.ones(nx))], [eps * diags(np.ones(nx)), -eps * gamma * diags(np.ones(nx))]],
                    format="csc")

    sol = solve_ivp(f, (0.0, t_end), np.concatenate([v_init, w_init]), method="BDF", t_eval=t, jac=jac,
                    rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"FitzHugh-Nagumo integration failed: {sol.message}")
    return FHNSolution(x, t, sol.y[:nx].T.copy(), sol.y[nx:].T.copy())


# --------------------------------------------------------------------------- Aliev-Panfilov 2-D
AP_PARAMETERS = {"k": 8.0, "a": 0.15, "eps0": 0.002, "mu1": 0.2, "mu2": 0.3}


@dataclass
class Stimulus:
    """Raise ``u`` by ``amplitude`` per unit time inside ``region(X, Y) -> bool`` during ``[t_start, t_end]``."""
    t_start: float
    t_end: float
    region: Callable[[np.ndarray, np.ndarray], np.ndarray]
    amplitude: float = 1.0


@dataclass
class APSolution:
    """An Aliev-Panfilov run. The model is dimensionless (see the module docstring for the map to ms and mV).

    Attributes
    ----------
    x, y : np.ndarray [-]
        Grid coordinates.
    t : np.ndarray [-]
        Saved times.
    u, v : np.ndarray [-]
        Excitation and recovery variables, ``(len(t), ny, nx)``.
    D : np.ndarray [-]
        Diffusivity used, ``(ny, nx)``.
    dt : float [-]
        Time step.
    """
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray                     # saved times
    u: np.ndarray                     # (len(t), ny, nx)
    v: np.ndarray
    D: np.ndarray                     # (ny, nx) diffusivity used
    params: dict = field(default_factory=dict)
    dt: float = 0.0

    def activation_times(self, threshold: float = 0.5) -> np.ndarray:
        return activation_times(self.t, self.u, threshold)


def _laplacian_noflux(u: np.ndarray, D: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """``div(D grad u)`` with face-averaged (harmonic) diffusivities and zero flux on the rectangle's edges."""
    def harm(a, b):
        s = a + b
        return np.where(s > 0, 2 * a * b / np.where(s > 0, s, 1.0), 0.0)

    fx = harm(D[:, 1:], D[:, :-1]) * (u[:, 1:] - u[:, :-1]) / dx        # flux on vertical faces
    fy = harm(D[1:, :], D[:-1, :]) * (u[1:, :] - u[:-1, :]) / dy
    out = np.zeros_like(u)
    out[:, :-1] += fx / dx
    out[:, 1:] -= fx / dx
    out[:-1, :] += fy / dy
    out[1:, :] -= fy / dy
    return out


def aliev_panfilov_2d(*, Lx: float = 100.0, Ly: float = 100.0, nx: int = 101, ny: int = 101, t_end: float = 60.0,
                      D: float | np.ndarray | Callable = 1.0, stimuli: Sequence[Stimulus] | None = None,
                      dt: float | None = None, save_every: float = 0.5, u0: np.ndarray | None = None,
                      v0: np.ndarray | None = None, **params) -> APSolution:
    """Explicit-Euler finite differences (stable step chosen from the diffusion limit when ``dt`` is None).

    ``D`` is a number, an (ny, nx) array or ``D(X, Y)``. Default stimulus: the left strip ``x < 0.05 Lx`` during
    ``t in [0, 1]``, which starts a planar wave travelling in +x."""
    p = {**AP_PARAMETERS, **params}
    x, y = np.linspace(0, Lx, nx), np.linspace(0, Ly, ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y)
    Dm = D(X, Y) if callable(D) else np.broadcast_to(np.asarray(D, float), X.shape).copy()
    dmax = float(Dm.max())
    dt_diff = 0.2 / (dmax * (1 / dx**2 + 1 / dy**2)) if dmax > 0 else 0.05
    dt = min(dt_diff, 0.02) if dt is None else float(dt)
    if stimuli is None:
        stimuli = [Stimulus(0.0, 1.0, lambda X, Y: X < 0.05 * Lx, 1.0)]
    masks = [(s, s.region(X, Y)) for s in stimuli]
    u = np.zeros_like(X) if u0 is None else np.asarray(u0, float).copy()
    v = np.zeros_like(X) if v0 is None else np.asarray(v0, float).copy()
    k, a, e0, m1, m2 = p["k"], p["a"], p["eps0"], p["mu1"], p["mu2"]
    n_steps = int(math.ceil(t_end / dt))
    every = max(1, int(round(save_every / dt)))
    ts, us, vs = [0.0], [u.copy()], [v.copy()]
    for n in range(1, n_steps + 1):
        t = n * dt
        I = np.zeros_like(u)  # noqa: E741
        for s, m in masks:
            if s.t_start <= t - dt < s.t_end:
                I[m] += s.amplitude
        du = _laplacian_noflux(u, Dm, dx, dy) + k * u * (1 - u) * (u - a) - u * v + I
        dv = (e0 + m1 * v / (u + m2)) * (-v - k * u * (u - a - 1))
        u = u + dt * du
        v = v + dt * dv
        if n % every == 0 or n == n_steps:
            ts.append(t)
            us.append(u.copy())
            vs.append(v.copy())
    if not np.all(np.isfinite(u)):
        raise RuntimeError("Aliev-Panfilov integration diverged; reduce dt")
    return APSolution(x, y, np.asarray(ts), np.asarray(us), np.asarray(vs), Dm, p, dt)


def activation_times(t: np.ndarray, u: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """First upward crossing of ``threshold`` per point (``u`` of shape (len(t), ...)), linearly interpolated between
    frames; ``nan`` where it never activates."""
    t = np.asarray(t, float)
    u = np.asarray(u, float)
    above = u >= threshold
    first = np.argmax(above, axis=0)
    never = ~above.any(axis=0)
    i1 = np.clip(first, 1, len(t) - 1)
    i0 = i1 - 1
    u0 = np.take_along_axis(u, i0[None], 0)[0]
    u1 = np.take_along_axis(u, i1[None], 0)[0]
    frac = np.clip((threshold - u0) / np.where(u1 != u0, u1 - u0, 1.0), 0.0, 1.0)
    T = t[i0] + frac * (t[i1] - t[i0])
    T = np.where(first == 0, t[0], T)
    return np.where(never, np.nan, T)


def conduction_velocity(T: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Local conduction velocity ``1 / |grad T|`` of an activation map (``nan`` where undefined)."""
    gy, gx = np.gradient(T, dy, dx)
    g = np.hypot(gx, gy)
    return np.where(g > 0, 1.0 / np.where(g > 0, g, 1.0), np.nan)
