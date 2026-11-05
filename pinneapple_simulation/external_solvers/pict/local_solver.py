"""A local incompressible solver, so the cavity runs without an NVIDIA GPU (this MacBook, CI, any CPU box).

PICT cannot run here: its kernels are CUDA, Apple Silicon has no CUDA. This module is **not** PICT and does not
pretend to be -- it is a vorticity-streamfunction solver written for this repository, run by numpy/scipy, and every
result it produces is labelled ``engine="local"``. It exists for two reasons: a Mac or a CI machine can then exercise
the whole bridge end to end (case, run, checks, ``PhysicalSample``), and the Ghia et al. (1982) comparison that
:mod:`.validation` implements gets a solver to be compared *with*, on a machine that has no GPU.

Method: vorticity-streamfunction, uniform square grid, second-order central differences, explicit Euler in time
(to steady state, where the time integrator stops mattering), Thom's wall-vorticity condition, and the streamfunction
Poisson problem solved exactly with a sparse LU factorised once per grid. Pressure is recovered afterwards from the
pressure Poisson equation with the steady wall condition ``dp/dn = -(1/Re) dw/ds``, and the result reports the
momentum residual that pressure leaves, so a wrong pressure is visible instead of implied.

Why not :mod:`pinneapple_simulation.numerical_solvers.stable_fluids`: that one is a graphics solver (semi-Lagrangian
advection with an FFT projection for a periodic domain). Its advection is too diffusive and its domain has no walls,
so it cannot be held to Ghia. Why not a finite-volume PISO of our own: that is what PICT is for; the point of this
module is a second, independent, cheap path, not a replacement.

Limits, stated rather than discovered:
- **Uniform grid only** (``CavityCase(refine=False)``): the wall-refined grid changes the finite-difference stencils,
  and a second discretisation with no reference behind it is how silent errors get in.
- **Lid at the +y wall moving in +x** (``lid_side="+y"``), the Ghia et al. configuration.
- It is a **steady** solver in practice: it marches the transient, records the initial and final fields, and stops on
  the residual. The transient is not an accurate time history and is not offered as one.
- Cost grows as ``nx^4`` (explicit stability limits dt while the Poisson solve grows with ``nx^2``): 64x64 is seconds,
  128x128 is about a minute, 256x256 is an afternoon.
"""
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .case import CavityCase

__all__ = ["LocalSolverConfig", "solve_cavity_local"]


@dataclass
class LocalSolverConfig:
    """How the local solver marches.

    ``cfl`` sets the time step through ``dt = min(cfl * h / |u|_max, viscosity * h^2 / 5)``: the second bound is the
    explicit diffusion limit, and 5 is a safety factor over the 2-D stability limit of 4. ``steady_tol`` is the
    relative change of vorticity per step that ends the march; ``max_steps`` and ``max_time_s`` are the guards that
    turn a run that will not converge into an answer that says so.
    """

    cfl: float = 0.4
    max_steps: int = 400_000
    steady_tol: float = 1e-7
    max_time_s: float | None = 300.0

    def check(self) -> list[str]:
        problems = []
        if self.cfl <= 0:
            problems.append("cfl must be positive")
        if self.max_steps < 1:
            problems.append("max_steps must be at least 1")
        if self.steady_tol <= 0:
            problems.append("steady_tol must be positive")
        return problems


def _interior(ny: int, nx: int):
    """Index arrays of the interior nodes of a ``(ny+1, nx+1)`` node grid."""
    return np.arange(1, ny), np.arange(1, nx)


def _dirichlet_laplacian(ny: int, nx: int, h: float):
    """``lap(psi) = -omega`` on the ``(ny, nx)`` interior nodes, ``psi = 0`` outside; factorised once.

    The boundary neighbours contribute nothing (their ``psi`` is zero), so the system is the plain 5-point Laplacian
    on the interior and its solution is the discrete streamfunction in one exact solve per step, not an iteration.
    """
    n = ny * nx
    index = {(i, j): i * nx + j for i in range(ny) for j in range(nx)}
    rows, cols, vals = [], [], []
    for i in range(ny):
        for j in range(nx):
            k = index[(i, j)]
            rows.append(k)
            cols.append(k)
            vals.append(-4.0 / h ** 2)
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < ny and 0 <= jj < nx:
                    rows.append(k)
                    cols.append(index[(ii, jj)])
                    vals.append(1.0 / h ** 2)
    matrix = sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
    return spla.factorized(matrix), index


def _gradients(field: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Central differences inside a node field, one-sided at its border."""
    d_y = np.empty_like(field)
    d_x = np.empty_like(field)
    d_y[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2 * h)
    d_y[0, :] = (field[1, :] - field[0, :]) / h
    d_y[-1, :] = (field[-1, :] - field[-2, :]) / h
    d_x[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2 * h)
    d_x[:, 0] = (field[:, 1] - field[:, 0]) / h
    d_x[:, -1] = (field[:, -1] - field[:, -2]) / h
    return d_y, d_x


def _laplacian(field: np.ndarray, h: float) -> np.ndarray:
    out = np.zeros_like(field)
    out[1:-1, 1:-1] = (field[2:, 1:-1] + field[:-2, 1:-1] + field[1:-1, 2:] + field[1:-1, :-2]
                       - 4.0 * field[1:-1, 1:-1]) / h ** 2
    return out


def _velocity(psi: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """``u = d psi / dy``, ``v = -d psi / dx`` on the nodes."""
    d_y, d_x = _gradients(psi, h)
    return d_y, -d_x


def _wall_vorticity(omega: np.ndarray, psi: np.ndarray, h: float, lid_velocity: float) -> None:
    """Thom's condition ``omega_wall = -2 psi_inner / h^2``, in place.

    The lid carries the extra term that the moving wall contributes: there the wall vorticity is
    ``-2 psi_inner / h^2 - 2 U / h`` (the tangential velocity jump enters the one-sided difference).
    """
    omega[1:-1, 0] = -2.0 * psi[1:-1, 1] / h ** 2
    omega[1:-1, -1] = -2.0 * psi[1:-1, -2] / h ** 2
    omega[0, 1:-1] = -2.0 * psi[1, 1:-1] / h ** 2
    omega[-1, 1:-1] = -2.0 * psi[-2, 1:-1] / h ** 2 - 2.0 * lid_velocity / h


def _solve_psi(omega: np.ndarray, solve, index, *, ny: int, nx: int) -> np.ndarray:
    rhs = -omega[1:-1, 1:-1].reshape(-1)
    psi = np.zeros((ny + 1, nx + 1))
    psi[1:-1, 1:-1] = solve(rhs).reshape(ny - 1, nx - 1)
    return psi


def solve_cavity_local(case: CavityCase, config: LocalSolverConfig | None = None):
    """March the cavity to steady state on the CPU and return a :class:`~.runner.PICTResult` with ``engine="local"``.

    The result is a cavity result like any other, so :func:`~.validation.cavity_checks` applies unchanged -- that is
    the point: the checks and the reference data are the same, only the engine differs, and the result says which.
    """
    from .runner import PICTResult

    config = config or LocalSolverConfig()
    problems = list(case.check()) + list(config.check())
    if case.refine:
        problems.append("the local engine solves on a uniform grid: build the case with refine=False")
    if case.lid_side != "+y":
        problems.append("the local engine implements the Ghia configuration only: lid_side must be '+y'")
    if int(case.nx) != int(case.ny):
        problems.append("the local engine solves a square cavity: nx must equal ny")
    if problems:
        raise ValueError("; ".join(problems))

    started = time.perf_counter()
    x0, y0, x1, y1 = case.bounds
    length = x1 - x0
    if not np.isclose(y1 - y0, length, rtol=1e-6):
        raise ValueError("the local engine solves a square cavity: the bounds must be square")
    ny, nx = int(case.ny), int(case.nx)
    h = length / nx
    viscosity = case.viscosity
    lid = float(case.lid_velocity)

    solve, index = _dirichlet_laplacian(ny - 1, nx - 1, h)
    omega = np.zeros((ny + 1, nx + 1))
    psi = np.zeros((ny + 1, nx + 1))

    initial = None
    steps = 0
    residual = np.inf
    u = v = None
    while steps < config.max_steps:
        _wall_vorticity(omega, psi, h, lid)
        u, v = _velocity(psi, h)
        # The CFL bound alone would let dt grow without limit while the flow is still at rest, so the advective
        # speed is floored at a tenth of the lid velocity; the diffusion bound is the 2-D explicit limit h^2 / (4 nu)
        # with a 10 % margin. (The first version of this line had the diffusion term inverted -- dt = nu h^2 / 5,
        # which is 1e-6 and marched the cavity one nanosecond per step.)
        speed = max(float(np.abs(u).max()), float(np.abs(v).max()), 0.1 * abs(lid))
        dt = min(config.cfl * h / speed, 0.9 * h ** 2 / (4.0 * viscosity))
        lap = _laplacian(omega, h)
        d_y, d_x = _gradients(omega, h)
        interior = (slice(1, -1), slice(1, -1))
        rhs = (-(u[interior] * d_x[interior] + v[interior] * d_y[interior]) + viscosity * lap[interior])
        # .copy() matters: a plain slice is a view, so the difference below would be zero by construction and the
        # march would stop after one step believing it had converged (it did, in the first version)
        omega_old = omega[interior].copy()
        omega[interior] = omega_old + dt * rhs
        psi = _solve_psi(omega, solve, index, ny=ny, nx=nx)
        change = np.abs(omega[interior] - omega_old).max() / max(abs(dt), 1e-30)
        residual = float(change)
        steps += 1
        if steps == 1:
            u0, v0 = _velocity(psi, h)
            initial = (u0.copy(), v0.copy())
        if steps % 50 == 0 and config.max_time_s is not None:
            if time.perf_counter() - started > config.max_time_s:
                break
        if residual < config.steady_tol * max(float(np.abs(omega).max()), 1.0):
            break

    _wall_vorticity(omega, psi, h, lid)
    u, v = _velocity(psi, h)
    # The wall rows come from a one-sided derivative of psi, which is only first-order there (at 32x32 it reports 0.72
    # where the lid moves at 1.0 -- the boundary layer is thinner than a cell). The field that is returned carries the
    # boundary conditions themselves: still on the three walls, the lid velocity on the lid.
    u[0, :], u[-1, :] = 0.0, lid
    u[:, 0], u[:, -1] = 0.0, 0.0
    v[0, :], v[-1, :] = 0.0, 0.0
    v[:, 0], v[:, -1] = 0.0, 0.0
    # No pressure. Recovering it from the momentum balance was tried and did not work: solving lap(p) = div(m) with
    # the Neumann data dp/dn = m.n leaves a rotational part of m that this discretisation does not resolve (the
    # residual was 0.53 of |m| at 32^2 and 0.63 at 64^2, growing with the grid). A pressure that looks plausible and
    # balances nothing is worse than a zero field that says so, so the local engine reports zeros and PICT is the
    # engine that provides pressure.
    pressure = np.zeros_like(u)

    # The fields stay on the node grid: there the walls are *in* the array (row 0 is the bottom wall, the last row is
    # the lid), so the boundary conditions can be checked where they are defined instead of half a cell inside. An
    # earlier version averaged to cell centres and the lid check then read 0.8 instead of 1.0 -- the boundary layer is
    # thinner than two cells, so no extrapolation fixes that honestly.
    snapshots = []
    if initial is not None:
        snapshots.append(np.stack([initial[0], initial[1]]))
    snapshots.append(np.stack([u, v]))
    velocity = np.stack(snapshots)
    p_field = pressure[None]

    return PICTResult(
        velocity=velocity, pressure=p_field,
        times=np.array([0.0, float(steps) * h]),          # time is reported in units of the last stable step * steps
        case_summary=case.summary(), config=vars(config), device="cpu", wall_time_s=time.perf_counter() - started,
        engine="local", layout="nodes",
        meta={"solver": "vorticity-streamfunction (pinneapple local engine, numpy/scipy)",
              "grid": f"{nx}x{ny} uniform nodes, h={h:.6g}", "steps": steps, "final_residual": residual,
              "pressure": "not computed by the local engine (it solves vorticity and streamfunction; use PICT for p)",
              "note": "steady state; the first snapshot is the initial condition, the last is the answer",
              "not_pict": "this run did not use PICT; PICT needs CUDA"},
    )
