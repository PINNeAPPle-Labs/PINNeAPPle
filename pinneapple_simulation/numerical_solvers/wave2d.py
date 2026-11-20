"""2-D scalar wave / acoustics: ``d2u/dt2 = c^2 lap(u) + f(x, y, t)`` on a rectangle, every wall held at zero.

The library's wave solvers were 1-D (``fdm``'s leapfrog) and 3-D (``fdm3d.ElasticWave3D``), with nothing in
between, so a 2-D acoustic problem had no classical solver to go to. This module fills that gap with the same
scheme the 3-D solver uses, one dimension down.

- :class:`Wave2D` -- the solver itself (numpy only): Störmer-Verlet (leapfrog) in time, second-order central
  differences in space, Dirichlet zero on the four walls.
- :class:`Wave2DFDMSolver` -- the registry face, so a ``ProblemSpec`` with ``pde.kind = "wave"`` (or
  ``"acoustic"``) and ``dim = 2`` reaches it through ``solve_from_spec``.

Quick start
-----------
>>> from pinneapple_simulation.numerical_solvers.wave2d import Wave2D, Wave2DConfig
>>> out = Wave2D(Wave2DConfig(nx=64, ny=64, nt=200, c=1.0, dt=1e-3)).solve()
>>> out.u.shape   # (nt+1, nx, ny)
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry

__all__ = ["Wave2DOutput", "Wave2DConfig", "Wave2D", "Wave2DFDMSolver"]


# ──────────────────────────────────────────────────────────────────────────────
# Output type
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Wave2DOutput:
    """Output from :class:`Wave2D`.

    Attributes
    ----------
    u : np.ndarray
        Displacement history, shape ``(nt+1, nx, ny)``; ``u[0]`` is the initial condition.
    coords : dict
        ``"x"``, ``"y"`` 1-D arrays and ``"t"`` the time array.
    meta : dict
        Solver parameters and diagnostics (wave speed, dt, CFL number).
    """
    u: np.ndarray
    coords: dict[str, np.ndarray]
    meta: dict[str, Any]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _laplacian2d(u: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Second-order central-difference Laplacian on a rectangle, Dirichlet zero at the walls.

    The wall rows themselves are set to zero after differencing: they are overwritten with zero by the caller
    anyway, and zeroing here keeps ``np.roll``'s periodic wrap from ever reaching the interior through the
    bootstrap half-step.
    """
    lap = (
        (np.roll(u, -1, axis=0) - 2.0 * u + np.roll(u, 1, axis=0)) / dx**2
        + (np.roll(u, -1, axis=1) - 2.0 * u + np.roll(u, 1, axis=1)) / dy**2
    )
    _hold_walls(lap)
    return lap


def _hold_walls(u: np.ndarray) -> np.ndarray:
    """Dirichlet zero on the four walls of the rectangle, in place."""
    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    return u


# ──────────────────────────────────────────────────────────────────────────────
# The solver
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Wave2DConfig:
    """Configuration for :class:`Wave2D`.

    Parameters
    ----------
    nx, ny : int
        Grid points along x and y.
    nt : int
        Number of time steps (``nt+1`` fields are returned).
    lx, ly : float
        Rectangle size: the domain is ``[0, lx] x [0, ly]``.
    dt : float
        Time step.  Courant stability: ``c dt sqrt(1/dx^2 + 1/dy^2) <= 1``.
    c : float
        Wave speed [m/s].
    ic_u_fn : callable, optional
        Initial displacement ``u0(X, Y) -> ndarray``.  When None a Gaussian pulse centred in the domain is
        used (the same default the 3-D solver takes).  Whatever it returns is zeroed on the walls.
    ic_du_fn : callable, optional
        Initial velocity ``du0(X, Y) -> ndarray``.  Defaults to zero.
    source_fn : callable, optional
        Source term ``f(X, Y, t) -> ndarray``, evaluated at the start of each step.
    """
    nx: int = 64
    ny: int = 64
    nt: int = 120
    lx: float = 1.0
    ly: float = 1.0
    dt: float = 1e-3
    c: float = 1.0
    ic_u_fn: Callable | None = None
    ic_du_fn: Callable | None = None
    source_fn: Callable | None = None


class Wave2D:
    """2-D scalar wave equation solver: ``d2u/dt2 = c^2 lap(u) + f(x, y, t)``.

    Discretisation: Störmer-Verlet (leapfrog) in time, second-order in ``dt`` and energy-conserving for the
    homogeneous equation; second-order central differences in space.  Both are second order, so the error falls
    as ``dx^2 + dt^2`` for a smooth solution.

    Boundary conditions: Dirichlet zero on all four walls of ``[0, lx] x [0, ly]`` (``u = 0`` on ``x = 0``,
    ``x = lx``, ``y = 0`` and ``y = ly``), enforced at every step including the first.  That is the whole
    boundary this solver has: no flux, impedance or free-surface condition is implemented.

    The first step is bootstrapped from the initial velocity, ``u^{-1} = u^0 - dt du^0 + dt^2/2 (c^2 lap u^0 +
    f^0)``, so a non-zero ``du/dt`` starts the wave moving in the right direction.
    """

    def __init__(self, cfg: Wave2DConfig | None = None):
        self.cfg = cfg or Wave2DConfig()

    def solve(self) -> Wave2DOutput:
        cfg = self.cfg
        x = np.linspace(0.0, cfg.lx, cfg.nx)
        y = np.linspace(0.0, cfg.ly, cfg.ny)
        t = np.linspace(0.0, cfg.nt * cfg.dt, cfg.nt + 1)
        dx = x[1] - x[0] if cfg.nx > 1 else cfg.lx
        dy = y[1] - y[0] if cfg.ny > 1 else cfg.ly

        X, Y = np.meshgrid(x, y, indexing="ij")

        # Courant stability check
        cfl = cfg.c * cfg.dt * np.sqrt(1.0 / dx**2 + 1.0 / dy**2)
        if cfl > 1.0:
            warnings.warn(
                f"Wave2D: CFL={cfl:.3f} > 1.0, the leapfrog scheme is unstable. Reduce dt or increase the grid "
                f"spacing.",
                stacklevel=2,
            )

        if cfg.ic_u_fn is not None:
            u = np.array(cfg.ic_u_fn(X, Y), dtype=np.float64)
        else:
            cx, cy = cfg.lx / 2.0, cfg.ly / 2.0
            sigma = min(cfg.lx, cfg.ly) / 8.0
            u = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma**2))
        _hold_walls(u)

        du = (np.array(cfg.ic_du_fn(X, Y), dtype=np.float64) if cfg.ic_du_fn is not None
              else np.zeros_like(u))

        c2 = cfg.c**2
        source0 = cfg.source_fn(X, Y, t[0]) if cfg.source_fn is not None else 0.0
        u_prev = _hold_walls(u - cfg.dt * du + 0.5 * cfg.dt**2 * (c2 * _laplacian2d(u, dx, dy) + source0))

        history = np.empty((cfg.nt + 1, cfg.nx, cfg.ny), dtype=np.float64)
        history[0] = u

        for n in range(cfg.nt):
            source = cfg.source_fn(X, Y, t[n]) if cfg.source_fn is not None else 0.0
            u_next = 2.0 * u - u_prev + cfg.dt**2 * (c2 * _laplacian2d(u, dx, dy) + source)
            _hold_walls(u_next)
            u_prev, u = u, u_next
            history[n + 1] = u

        return Wave2DOutput(
            u=history,
            coords={"x": x, "y": y, "t": t},
            meta={"c": cfg.c, "dt": float(cfg.dt), "cfl": float(cfl), "solver": "Wave2D"},
        )


# ──────────────────────────────────────────────────────────────────────────────
# The registry face: a ProblemSpec in, the 2-D wave solve out
# ──────────────────────────────────────────────────────────────────────────────

@SolverRegistry.register(
    name="wave2d_fdm",
    family="pde",
    description="2-D scalar acoustic wave equation (Stormer-Verlet) on a rectangle with every wall held at zero",
    tags=["fdm", "wave", "acoustic", "2d", "explicit"],
)
class Wave2DFDMSolver(SolverBase):
    """``solve_from_spec`` for :class:`Wave2D`: ``d2u/dt2 = c^2 lap(u) + f`` on a rectangle, every wall at zero.

    The 2-D case the 1-D (``fdm``) and 3-D (``fdm3d.ElasticWave3D``) solvers left open. It solves exactly the
    equation, the domain and the boundary ``Wave2D`` documents, and nothing else.

    Read from the spec:

    - ``spec.pde.kind`` must contain ``"wave"`` or ``"acoustic"``; ``spec.dim`` must be 2;
    - ``spec.domain_bounds`` gives the rectangle: ``x`` and ``y`` ranges (default ``(0, 1)`` each);
    - ``spec.pde.params``: ``c`` the wave speed (required and positive; ``wave_speed`` is accepted as an alias,
      the name ``fdm``'s 1-D wave solver uses), ``nx``/``ny`` grid points (default 64 each), ``nt``
      steps (default 120), ``dt`` (default 40 % of the Courant bound,
      ``0.4 / (c sqrt(1/dx^2 + 1/dy^2))``), ``u0`` a uniform initial displacement (default 0.0), ``du0`` a
      uniform initial velocity (default 0.0), ``source`` a constant source (default 0.0) and ``pulse``.

    ``pulse`` (default True): a uniform ``u0 = 0`` with ``du0 = 0`` and no source is a field that stays zero
    forever, so in that case the solver starts from the Gaussian pulse ``Wave2D`` uses by default instead of
    returning the trivial answer. Set ``pulse=False`` to get that literal zero field, or give any of ``u0``,
    ``du0`` or ``source`` to override it (then ``u0``/``du0`` are used as given).

    Refused, by name, instead of being run with a boundary it does not have: any condition that is not
    ``dirichlet`` or ``initial`` (a flux, a convection, a Robin wall), and any ``dirichlet`` condition whose
    value on the walls is not zero.
    """

    def solve_from_spec(self, spec) -> SolverOutput:
        kind = str(getattr(spec.pde, "kind", "")).lower().replace("-", "_").replace(" ", "_")
        if not any(k in kind for k in ("wave", "acoustic")):
            raise ValueError(
                f"wave2d_fdm solves wave/acoustic problems in 2-D, and the spec's kind is '{kind}'. Its "
                f"documented scope is Wave2D: d2u/dt2 = c^2 lap(u) + f on a rectangle, every wall at zero.")
        dim = int(getattr(spec, "dim", 2))
        if dim != 2:
            raise ValueError(
                f"wave2d_fdm is a 2-D solver and this spec is {dim}-D. Its documented scope is the 2-D scalar "
                f"wave equation; the 1-D and 3-D cases have their own solvers (fdm, wave3d_fdm).")

        params = dict(getattr(spec.pde, "params", {}) or {})
        c = params.get("c", params.get("wave_speed"))
        if c is None:
            raise ValueError("wave2d_fdm needs the wave speed in spec.pde.params['c']")
        c = float(c)
        if c <= 0.0:
            raise ValueError(f"wave2d_fdm needs a positive wave speed, and spec.pde.params['c'] is {c}.")

        bounds = dict(getattr(spec, "domain_bounds", {}) or {})
        lx = abs(bounds.get("x", (0.0, 1.0))[1] - bounds.get("x", (0.0, 1.0))[0])
        ly = abs(bounds.get("y", (0.0, 1.0))[1] - bounds.get("y", (0.0, 1.0))[0])
        nx = int(params.get("nx", 64))
        ny = int(params.get("ny", 64))
        _check_walls_are_cold(spec, bounds, nx, ny, solver="wave2d_fdm")

        dx, dy = lx / max(nx - 1, 1), ly / max(ny - 1, 1)
        dt = params.get("dt")
        if dt is None:
            dt = 0.4 / (c * np.sqrt(1.0 / dx**2 + 1.0 / dy**2))
        dt = float(dt)

        u0 = float(params.get("u0", 0.0))
        du0 = float(params.get("du0", 0.0))
        source = float(params.get("source", 0.0))
        pulse = bool(params.get("pulse", True)) and u0 == 0.0 and du0 == 0.0 and source == 0.0

        def _constant(value: float) -> Callable:
            return lambda X, Y: np.full_like(np.asarray(X, dtype=np.float64), value)

        cfg = Wave2DConfig(
            nx=nx, ny=ny, nt=int(params.get("nt", 120)), lx=lx, ly=ly, dt=dt, c=c,
            ic_u_fn=None if pulse else _constant(u0),
            ic_du_fn=_constant(du0),
            source_fn=(lambda X, Y, t: np.full_like(np.asarray(X, dtype=np.float64), source))
            if source else None,
        )
        out = Wave2D(cfg).solve()
        field = torch.from_numpy(np.asarray(out.u[-1], dtype=np.float32))
        energy = float(np.sum(np.asarray(out.u[-1], dtype=np.float64) ** 2))
        return SolverOutput(
            result=field,
            losses={"energy": torch.tensor(energy)},
            extras={"method": "wave2d_fdm", "grid": (nx, ny), "c": c, "dt": dt, "nt": cfg.nt,
                    "cfl": float(out.meta["cfl"]), "u0": u0, "du0": du0, "source": source, "pulse": pulse,
                    "coords": out.coords, "history_shape": tuple(np.shape(out.u))},
        )


def _check_walls_are_cold(spec, bounds, nx: int, ny: int, solver: str = "wave2d_fdm") -> None:
    """The four walls of the rectangle are held at zero and this solver has no other boundary.

    ``fdm3d``'s ``FDM3DSolver._check_walls_are_cold`` samples a 3-axis box (three-column points, a z coordinate
    invented from a default range for a spec that has no z). A 2-D spec's conditions are written against
    ``(x, y)``, so this is the same check with two-column points sampled along the four edges -- corners, the
    edge midpoints and the points in between -- instead of reusing a function that would hand a 2-D value
    function a coordinate it never declared.
    """
    conditions = tuple(getattr(spec, "conditions", ()) or ())
    others = [c for c in conditions if getattr(c, "kind", None) not in ("dirichlet", "initial")]
    if others:
        raise ValueError(
            f"{solver} holds the four walls at zero; the spec asks for "
            f"{sorted({c.kind for c in others})}, which this solver does not have.")
    from . import _bc

    (x0, x1) = bounds.get("x", (0.0, 1.0))
    (y0, y1) = bounds.get("y", (0.0, 1.0))
    x = np.linspace(x0, x1, nx)
    y = np.linspace(y0, y1, ny)
    edge = np.linspace(0.0, 1.0, 9)
    xs = x[0] + edge * (x[-1] - x[0])
    ys = y[0] + edge * (y[-1] - y[0])
    wall = np.concatenate([
        np.column_stack([np.full_like(edge, x[0]), ys]),   # x = x_min
        np.column_stack([np.full_like(edge, x[-1]), ys]),  # x = x_max
        np.column_stack([xs, np.full_like(edge, y[0])]),   # y = y_min
        np.column_stack([xs, np.full_like(edge, y[-1])]),  # y = y_max
    ]).astype(np.float32)
    pts = torch.from_numpy(wall)
    for cond in _bc.dirichlet_conditions(spec):
        values = _bc.values(cond, pts)
        if float(np.max(np.abs(np.asarray(values.detach().cpu().numpy(), dtype=np.float64)))) > 1e-12:
            raise ValueError(
                f"{solver} holds every wall at zero, and the dirichlet condition "
                f"'{getattr(cond, 'name', '')}' asks for a non-zero value on the wall.")
