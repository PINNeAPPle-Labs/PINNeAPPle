"""One-dimensional transient heat conduction, ``dT/dt = alpha d2T/dx2 + q(x, t)`` on ``[0, L]``, with both ends
held at a fixed temperature.

``Heat1D`` is the scheme: central differences in space, Crank-Nicolson in time (``theta = 1/2``, unconditionally
stable and second order), one tridiagonal solve per step (``scipy.linalg.solve_banded``). ``Heat1DSolver`` is the
registry face a ``ProblemSpec`` can drive (``name="heat1d_fdm"``).

What the scheme gets exactly, and therefore what its tests check against closed forms:

- a linear profile between the two end temperatures is a steady state, and Crank-Nicolson reproduces it to machine
  precision (the discrete Laplacian of a straight line is zero, and the boundary rows are the exact values);
- ``T(x, 0) = sin(pi x / L)`` with both ends at zero decays as ``exp(-alpha pi^2 t / L^2) sin(pi x / L)``, the
  discrete decay rate matching it up to the ``(pi dx)^2 / 12`` central-difference error;
- a constant source ``q`` with both ends at zero has the steady parabola ``T = q x (L - x) / (2 alpha)``, which the
  discrete operator satisfies exactly (a second difference is exact for a quadratic).

Usage
-----
    from pinneapple_simulation.numerical_solvers.heat1d import Heat1D, Heat1DSolver

    scheme = Heat1D(alpha=1e-4, length=0.2, nx=101, T_left=300.0, T_right=400.0, T_init=300.0)
    T_final = scheme.solve()
    history = scheme.run()          # (nt + 1, nx)

    out = Heat1DSolver().solve_from_spec(spec)     # -> SolverOutput, result is the final (nx,) field
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from scipy.linalg import solve_banded

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry

__all__ = ["Heat1D", "Heat1DSolver"]


# ─────────────────────────────────────────────────────────────────────────────
# The scheme
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Heat1D:
    """Crank-Nicolson finite differences for ``dT/dt = alpha d2T/dx2 + q(x, t)`` on ``[0, L]``.

    The two end nodes are Dirichlet data held constant in time (``T_left`` at ``x = 0``, ``T_right`` at
    ``x = L``); the interior is advanced implicitly, so there is no stability limit on ``dt``. ``alpha > 0`` is
    the thermal diffusivity [m^2/s], or 1.0 for the non-dimensional equation.

    Attributes
    ----------
    alpha : float
        Thermal diffusivity, > 0.
    length : float
        Domain length ``L``, > 0. The grid is ``x = linspace(0, L, nx)``.
    nx : int
        Grid points, at least 3 (two ends and one interior node).
    dt : float | None
        Time step; ``None`` means ``0.4 dx^2 / alpha`` (the explicit stability bound, used even though
        Crank-Nicolson is stable without it, so that the transient is actually resolved by default).
    nt : int
        Default number of steps taken by :meth:`run` and :meth:`solve`.
    T_left, T_right : float
        The temperatures held at ``x = 0`` and ``x = L``.
    T_init : float | numpy.ndarray
        Initial temperature: one number for a uniform start, or an ``(nx,)`` array for a profile. The two end
        entries of a profile are overwritten by ``T_left`` / ``T_right``.
    source : float | collections.abc.Callable
        Volumetric source ``q`` [K/s]: a constant, or ``q(x, t) -> (nx,)`` evaluated at the midpoint of each
        step (which is what makes a constant-source steady state exact under Crank-Nicolson).

    Raises
    ------
    ValueError
        ``alpha <= 0``, ``length <= 0``, ``nx < 3``, ``dt <= 0``, or an initial profile whose shape is not ``(nx,)``.
    """

    alpha: float = 1.0
    length: float = 1.0
    nx: int = 101
    dt: float | None = None
    nt: int = 200
    T_left: float = 0.0
    T_right: float = 0.0
    T_init: float | np.ndarray = 0.0
    source: float | Callable = 0.0

    def __post_init__(self):
        self.alpha = float(self.alpha)
        self.length = float(self.length)
        self.nx = int(self.nx)
        self.nt = int(self.nt)
        self.T_left = float(self.T_left)
        self.T_right = float(self.T_right)
        if not self.alpha > 0.0:
            raise ValueError(f"Heat1D needs alpha > 0 (thermal diffusivity), got {self.alpha}")
        if not self.length > 0.0:
            raise ValueError(f"Heat1D needs length > 0, got {self.length}")
        if self.nx < 3:
            raise ValueError(f"Heat1D needs nx >= 3 (two ends and an interior node), got {self.nx}")
        self.dx = self.length / (self.nx - 1)
        self.x = np.linspace(0.0, self.length, self.nx)
        if self.dt is None:
            self.dt = 0.4 * self.dx**2 / self.alpha      # the explicit bound: a resolved transient, not a limit
        self.dt = float(self.dt)
        if not self.dt > 0.0:
            raise ValueError(f"Heat1D needs dt > 0, got {self.dt}")
        if not callable(self.source):
            self.source = float(self.source)
        self.initial_state()                              # validates T_init
        self._ab = self._banded()

    # ── internals ────────────────────────────────────────────────────────────

    @property
    def r(self) -> float:
        """Half the explicit mesh ratio, ``alpha dt / (2 dx^2)``: the off-diagonal weight of the implicit side."""
        return self.alpha * self.dt / (2.0 * self.dx**2)

    def _banded(self) -> np.ndarray:
        """The constant tridiagonal matrix of the interior unknowns, in the ``(1, 1)`` banded layout."""
        r = self.r
        ab = np.zeros((3, self.nx - 2))
        ab[0, 1:] = -r
        ab[1, :] = 1.0 + 2.0 * r
        ab[2, :-1] = -r
        return ab

    # ── public API ───────────────────────────────────────────────────────────

    def source_at(self, t: float) -> np.ndarray:
        """The source ``q(x, t)`` on the grid, an ``(nx,)`` array."""
        if callable(self.source):
            return np.broadcast_to(np.asarray(self.source(self.x, t), dtype=np.float64), (self.nx,)).copy()
        return np.full(self.nx, self.source, dtype=np.float64)

    def initial_state(self) -> np.ndarray:
        """The initial temperature on the grid, an ``(nx,)`` array with the two ends already at their values."""
        if np.ndim(self.T_init) == 0:
            T = np.full(self.nx, float(self.T_init), dtype=np.float64)
        else:
            T = np.array(self.T_init, dtype=np.float64, copy=True)
            if T.shape != (self.nx,):
                raise ValueError(f"Heat1D needs an initial profile of shape ({self.nx},), got {T.shape}")
        T[0], T[-1] = self.T_left, self.T_right
        return T

    def step(self, T: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Advance the field ``T`` (``(nx,)``, at time ``t``) by one Crank-Nicolson step; returns a new array."""
        T = np.array(T, dtype=np.float64, copy=True)
        if T.shape != (self.nx,):
            raise ValueError(f"Heat1D.step needs a field of shape ({self.nx},), got {T.shape}")
        T[0], T[-1] = self.T_left, self.T_right
        r = self.r
        # Explicit half: (1 - 2r) T_i + r (T_{i-1} + T_{i+1}) + dt q, source at the middle of the step.
        rhs = (1.0 - 2.0 * r) * T[1:-1] + r * (T[:-2] + T[2:])
        rhs = rhs + self.dt * self.source_at(t + 0.5 * self.dt)[1:-1]
        # The known end values sit on the implicit side too; move them across.
        rhs[0] += r * self.T_left
        rhs[-1] += r * self.T_right
        T[1:-1] = solve_banded((1, 1), self._ab.copy(), rhs, check_finite=False)
        return T

    def run(self, nt: int | None = None) -> np.ndarray:
        """The trajectory, ``(nt + 1, nx)``, starting with the initial state."""
        steps = self.nt if nt is None else int(nt)
        T = self.initial_state()
        history = np.empty((steps + 1, self.nx), dtype=np.float64)
        history[0] = T
        for n in range(steps):
            T = self.step(T, n * self.dt)
            history[n + 1] = T
        return history

    def solve(self, nt: int | None = None) -> np.ndarray:
        """The final field, ``(nx,)``, after ``nt`` steps (``self.nt`` by default)."""
        return self.run(nt)[-1]


# ─────────────────────────────────────────────────────────────────────────────
# The registry face: a ProblemSpec in, the 1-D heat solve out
# ─────────────────────────────────────────────────────────────────────────────


@SolverRegistry.register(
    name="heat1d_fdm",
    family="pde",
    description="1-D transient heat conduction dT/dt = alpha d2T/dx2 + q on [0, L] with both ends held at a "
                "fixed temperature, by Crank-Nicolson in time (unconditionally stable, second order) with one "
                "tridiagonal solve per step",
    tags=["fdm", "heat", "1d", "transient", "crank_nicolson"],
)
class Heat1DSolver(SolverBase):
    """``solve_from_spec`` for :class:`Heat1D`: ``dT/dt = alpha d2T/dx2 + q(x, t)`` on ``[0, L]``, with the end at
    ``x = 0`` held at one temperature and the end at ``x = L`` at another, both constant in time.

    What it reads from the spec
    ---------------------------
    ``spec.pde.kind``
        Must contain ``"heat"`` or ``"diffusion"``; anything else is refused, naming the kind.
    ``spec.dim``
        Must be 1; anything else is refused.
    ``spec.domain_bounds["x"]``
        The interval, default ``(0.0, 1.0)``; its width is ``L``. ``domain_bounds["t"]`` only supplies the start
        time ``t0`` (default ``0.0``).
    ``spec.conditions``
        The two end temperatures are the values of the spec's ``dirichlet`` conditions at ``x = 0`` and ``x = L``
        (``_bc.values`` at those two points, restricted to the ends each condition's selector covers, with the
        ``"left"``/``"right"``/``"boundary"`` tags understood). An end no Dirichlet condition covers is held at
        0.0. An ``initial`` condition is read as the uniform start when it is constant (see ``T_init`` below).
    ``spec.pde.params``
        ``alpha`` (required) thermal diffusivity [m^2/s], or ``k`` together with ``rho`` and ``cp``, from which
        ``alpha = k / (rho cp)``; ``nx`` grid points (default 101, at least 3); ``dt`` (default ``0.4 dx^2 /
        alpha``, the explicit bound, even though Crank-Nicolson is stable, so the transient is resolved);
        ``nt`` steps (default 200); ``T_init`` uniform initial temperature (default 0.0, or the constant
        ``initial`` condition when the spec carries one); ``source`` constant volumetric source [K/s] (default
        0.0; a callable ``q(x, t)`` is used as it is).

    What it refuses
    ---------------
    A kind that is not heat/diffusion, a spec that is not 1-D, a missing diffusivity, an ``nx < 3``, a condition
    of a kind this solver has no boundary for (Neumann, Robin, interface, data -- it holds both ends at a fixed
    temperature and nothing else), a Dirichlet value that varies with time (the ends are constant here, and the
    error names the condition), and an initial condition that varies in space (``T_init`` is one number for the
    whole interval; the error says so and points at ``params["T_init"]``).

    Returns
    -------
    SolverOutput
        ``result`` the final temperature, an ``(nx,)`` float32 tensor; ``losses`` ``{"max_T": ...}``; ``extras``
        ``{"method": "heat1d_fdm", "alpha", "dt", "nt", "x", "T_left", "T_right"}``.
    """

    def solve_from_spec(self, spec) -> SolverOutput:
        kind = str(getattr(spec.pde, "kind", "")).lower().replace("-", "_").replace(" ", "_")
        if not any(k in kind for k in ("heat", "diffusion")):
            raise ValueError(
                f"heat1d_fdm solves heat/diffusion in 1-D, and the spec's kind is '{kind}'. Its documented scope "
                f"is 1-D transient heat conduction: dT/dt = alpha d2T/dx2 + q(x, t) on [0, L], with both ends held "
                f"at a fixed temperature.")
        if int(getattr(spec, "dim", 1)) != 1:
            raise ValueError(
                f"heat1d_fdm is a 1-D solver and this spec is {getattr(spec, 'dim', '?')}-D. Its documented scope "
                f"is the interval [0, L]; a higher-dimensional heat problem needs fdm (2-D) or fdm3d (3-D).")
        conditions = tuple(getattr(spec, "conditions", ()) or ())
        other_kinds = sorted({str(getattr(c, "kind", "?")) for c in conditions
                              if getattr(c, "kind", None) not in ("dirichlet", "initial")})
        if other_kinds:
            raise ValueError(
                f"heat1d_fdm holds both ends at a fixed temperature and has no other boundary: this spec asks for "
                f"{other_kinds}, which it would have to drop to run.")
        params = dict(getattr(spec.pde, "params", {}) or {})
        bounds = dict(getattr(spec, "domain_bounds", {}) or {})
        x_a, x_b = (float(v) for v in bounds.get("x", (0.0, 1.0)))
        low, high = min(x_a, x_b), max(x_a, x_b)
        length = high - low
        if not length > 0.0:
            raise ValueError(f"heat1d_fdm needs a non-empty interval, and domain_bounds['x'] is {bounds.get('x')}")

        alpha = self._diffusivity(params)
        nx = int(params.get("nx", 101))
        if nx < 3:
            raise ValueError(f"heat1d_fdm needs nx >= 3 (two ends and an interior node), and the spec asks for {nx}")
        dx = length / (nx - 1)
        dt = params.get("dt")
        dt = 0.4 * dx**2 / alpha if dt is None else float(dt)
        if not dt > 0.0:
            raise ValueError(f"heat1d_fdm needs dt > 0 (the spec asks for {dt})")
        nt = int(params.get("nt", 200))
        if nt < 0:
            raise ValueError(f"heat1d_fdm needs nt >= 0 (the spec asks for {nt})")

        coords = tuple(getattr(spec, "coords", ()) or ())
        t0 = float(bounds.get("t", (0.0, 1.0))[0])
        T_left, T_right = self._end_temperatures(spec, (low, high), coords, t0, t0 + nt * dt)
        T_init = self._initial_temperature(spec, params, coords, nx, high - low, t0)
        source = params.get("source", 0.0)

        scheme = Heat1D(alpha=alpha, length=length, nx=nx, dt=dt, nt=nt,
                        T_left=T_left, T_right=T_right, T_init=T_init, source=source)
        field = scheme.solve()
        return SolverOutput(
            result=torch.from_numpy(np.asarray(field, dtype=np.float32)),
            losses={"max_T": torch.tensor(float(np.max(field)))},
            extras={"method": "heat1d_fdm", "alpha": alpha, "dt": dt, "nt": nt,
                    "x": np.linspace(low, high, nx), "T_left": T_left, "T_right": T_right},
        )

    # ── spec readers ─────────────────────────────────────────────────────────

    @staticmethod
    def _diffusivity(params: dict) -> float:
        """``alpha`` from the parameters, or ``k / (rho cp)``; refused by name when neither is given."""
        alpha = params.get("alpha")
        if alpha is None and params.get("k") is not None:
            alpha = float(params["k"]) / (float(params.get("rho", 1.0)) * float(params.get("cp", 1.0)))
        if alpha is None:
            raise ValueError("heat1d_fdm needs the thermal diffusivity in spec.pde.params['alpha'] "
                             "(or k together with rho and cp)")
        alpha = float(alpha)
        if not alpha > 0.0:
            raise ValueError(f"heat1d_fdm needs a positive alpha (thermal diffusivity), and the spec gives {alpha}")
        return alpha

    @staticmethod
    def _points(coords, xs, t: float) -> torch.Tensor:
        """An ``(len(xs), len(coords))`` point matrix in the spec's own coordinate order (``x`` filled with the
        positions, ``t`` with the time, any other coordinate with zero)."""
        xs = np.asarray(xs, dtype=np.float32).reshape(-1)
        if not coords:
            return torch.from_numpy(xs.reshape(-1, 1))
        columns = []
        for name in coords:
            if name == "x":
                columns.append(xs)
            elif name == "t":
                columns.append(np.full(len(xs), float(t), dtype=np.float32))
            else:
                columns.append(np.zeros(len(xs), dtype=np.float32))
        return torch.from_numpy(np.column_stack(columns))

    @staticmethod
    def _end_temperatures(spec, ends, coords, t0: float, t_end: float) -> tuple[float, float]:
        """The two temperatures the spec's Dirichlet conditions prescribe at the ends of ``ends = (low, high)``.

        Each Dirichlet condition is evaluated at both ends and restricted with ``_bc.select`` to the ends its
        selector covers, the later condition winning if two cover the same end. An end nothing covers is 0.0.
        A value that changes between ``t0`` and ``t_end`` is refused naming its condition: the ends of this scheme
        are constant in time.
        """
        from . import _bc

        left, right = ends
        pts0 = Heat1DSolver._points(coords, [left, right], t0)
        pts1 = Heat1DSolver._points(coords, [left, right], t_end)
        edges = {"left": torch.tensor([0]), "right": torch.tensor([1])}
        boundary = torch.tensor([0, 1])
        values = [0.0, 0.0]
        for cond in _bc.dirichlet_conditions(spec):
            covered = [int(i) for i in _bc.select(cond, pts0, edges=edges, boundary=boundary)]
            if not covered:
                continue
            v0 = np.asarray(_bc.values(cond, pts0).detach().cpu().numpy(), dtype=np.float64).reshape(-1)
            v1 = np.asarray(_bc.values(cond, pts1).detach().cpu().numpy(), dtype=np.float64).reshape(-1)
            name = getattr(cond, "name", "")
            if float(np.max(np.abs(v1 - v0))) > 1e-9 * max(1.0, float(np.max(np.abs(v0)))):
                raise ValueError(
                    f"heat1d_fdm holds both ends at a temperature that does not change with time, and the "
                    f"dirichlet condition '{name}' is time-dependent (it differs between t={t0:g} and "
                    f"t={t_end:g}). Give constant end values, or use a solver whose boundary can move in time.")
            for i in covered:
                values[i] = float(v0[i])
        return values[0], values[1]

    @staticmethod
    def _initial_temperature(spec, params: dict, coords, nx: int, length: float, t0: float) -> np.ndarray:
        """The ``(nx,)`` start: ``params["T_init"]`` when given, else the spec's own initial condition when that
        is a constant, else zeros. A start that varies in space is refused by name -- this solver takes one
        uniform number as its initial temperature."""
        if params.get("T_init") is not None:
            return np.full(nx, float(params["T_init"]), dtype=np.float64)
        initial = [c for c in (getattr(spec, "conditions", ()) or ())
                   if str(getattr(c, "kind", "")).startswith("initial")]
        if not initial:
            return np.zeros(nx, dtype=np.float64)
        from . import _bc

        pts = Heat1DSolver._points(coords, np.linspace(0.0, length, nx), t0)
        values = np.asarray(_bc.values(initial[0], pts).detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        spread = float(np.max(values) - np.min(values))
        name = getattr(initial[0], "name", "")
        if spread > 1e-6 * max(1.0, float(np.max(np.abs(values)))):
            raise ValueError(
                f"heat1d_fdm starts from a uniform temperature and the initial condition '{name}' varies in space: "
                f"give that uniform start in spec.pde.params['T_init']. A profile as the initial condition needs "
                f"the PINN path, not this solver.")
        return np.full(nx, float(np.mean(values)), dtype=np.float64)
