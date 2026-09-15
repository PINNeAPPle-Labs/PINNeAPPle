"""Finite Difference Method (FDM) — problem-agnostic numerical solver.

Automatically dispatches to the right FD scheme from a ProblemSpec.pde.kind.

Supported PDEs (substring match on kind, case-insensitive):
  poisson / laplace          → 2D: -∇²u = f  (SOR / Gauss-Seidel)
  helmholtz                  → 2D: (∇² + k²)u = f  (SOR)
  heat / diffusion           → 2D: ∂u/∂t = α∇²u  (ADI θ-method)
  wave                       → 1D: ∂²u/∂t² = c² ∂²u/∂x²  (leapfrog)
  burgers                    → 1D: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²  (FTCS+upwind)
  advection / convection     → 1D: ∂u/∂t + v∂u/∂x = D∂²u/∂x²  (upwind)
  generic / unknown          → falls back to 2D Laplace

BC application from ConditionSpec:
  DirichletBC  → grid values pinned at nodes satisfying the selector
  NeumannBC    → flux imposed via one-sided finite differences at boundary
  RobinBC      → a*u + b*(du/dn) = g at boundary nodes, du/dn taken along
                 the true outward normal (unlike NeumannBC's plain
                 coordinate-direction difference above). Coefficients
                 (a, b) come from ctx["robin_coeffs"][cond.name] -- same
                 convention as the PINN-training-side Robin residual in
                 pinneapple_physics.pinn_solver.compiler.compile -- and
                 default to a=1, b=1 when absent. An optional nonlinear
                 radiative term eps*sigma*(u^4 - t_env^4) can be added via
                 ctx["robin_coeffs"][cond.name]["radiative"] = {"epsilon":,
                 "sigma":, "t_env":}, solved by a few fixed-point (Picard)
                 iterations per node for the elliptic (_poisson) case, and
                 by a lagged/semi-implicit linearization -- re-linearized
                 every step, not iterated to full convergence within a
                 step -- for the transient (_heat) case, since that one is
                 embedded directly in each ADI sweep's tridiagonal system.
                 _heat's Robin threading covers interior transverse nodes
                 of a whole selected edge; the four corner nodes keep the
                 solver's existing Dirichlet-hold behavior (see
                 _robin_edge_specs / _adi_heat_2d).

                 Convergence note (radiative term specifically): the linear
                 Robin case is validated against closed-form solutions for
                 both _poisson and _heat (see tests/pinneapple_solvers/
                 test_fdm_robin.py) and converges reliably from a cold
                 (u=0) start. The nonlinear radiative case's per-node
                 solve is exact given a converged neighbor value (also
                 tested there) -- but for a *stiff* configuration (roughly:
                 |b|/dx or |b|/dy large relative to a, e.g. a thin body
                 with high conductivity and a boundary condition placed
                 close by) combined with strong radiative coupling, the
                 block-coupled outer iteration (_poisson: repeated SOR +
                 boundary-update passes; _heat: the lagged per-timestep
                 linearization) was observed to settle onto a numerically
                 stable but non-physical fixed point from a u=0 start,
                 rather than the correct root, even though the correct
                 root is itself a stable fixed point once reached (seeding
                 the solve there, it stays). Root-caused to the coupled
                 iteration's basin of attraction for the true solution
                 being small in that regime, not to an error in the
                 per-node formula. Mitigations that did *not* resolve it
                 on their own: more outer iterations, full (vs. truncated)
                 inner SOR convergence per pass, warm-starting nearer the
                 expected answer, and epsilon continuation/homotopy.
                 Until this is resolved, treat a radiative-Robin solve as
                 unverified for stiff configurations -- inspect the result
                 (e.g. check the boundary satisfies the flux balance using
                 its own converged neighbor) rather than assuming
                 convergence to the physical branch.

Usage
-----
    # 1. From a ProblemSpec (recommended):
    from pinneapple_simulation.numerical_solvers.fdm import FDMSolver
    solver = FDMSolver(nx=128, ny=128, nt=200)
    out    = solver.solve_from_spec(spec)          # → SolverOutput

    # ...with a Robin (or radiative-Robin) condition, coefficients are
    # supplied at solve time via ctx_extra, e.g. for a convective +
    # radiative surface at x=x_max:
    out = solver.solve_from_spec(spec, ctx_extra={
        "robin_coeffs": {
            "surface": {
                "a": h_conv, "b": -k_cond,
                "radiative": {"epsilon": 0.8, "sigma": 5.670374419e-8, "t_env": t_gas_k},
            },
        },
    })

    # 2. Legacy direct call (Poisson only):
    f  = torch.zeros(64, 64)
    bc = torch.zeros(64, 64)
    out = FDMSolver().forward(f, bc, dx=1/63, dy=1/63)
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry

# ─────────────────────────────────────────────────────────────────────────────
# Grid helpers
# ─────────────────────────────────────────────────────────────────────────────

def _linspace(lo: float, hi: float, n: int) -> np.ndarray:
    return np.linspace(lo, hi, max(n, 2), dtype=np.float64)


def _spacing(lo: float, hi: float, n: int) -> float:
    return (hi - lo) / max(n - 1, 1)


def _build_2d_grid(
    x0: float, x1: float, nx: int,
    y0: float, y1: float, ny: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    x  = _linspace(x0, x1, nx)
    y  = _linspace(y0, y1, ny)
    dx = _spacing(x0, x1, nx)
    dy = _spacing(y0, y1, ny)
    XX, YY = np.meshgrid(x, y, indexing="ij")
    return x, y, XX, YY, dx, dy


def _apply_dirichlet_2d(
    u: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    conditions,
    ctx: Dict,
) -> np.ndarray:
    """Overwrite grid points that satisfy any DirichletBC selector."""
    if not conditions:
        return u
    nx, ny = u.shape
    pts = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1).reshape(-1, 2).astype(np.float32)
    for cond in conditions:
        if getattr(cond, "kind", "") != "dirichlet":
            continue
        sel_fn  = getattr(cond, "selector",  None)
        val_fn  = getattr(cond, "value_fn",  None)
        if not callable(sel_fn) or not callable(val_fn):
            continue
        try:
            mask = np.asarray(sel_fn(pts, ctx), dtype=bool)
            if not mask.any():
                continue
            vals = np.asarray(val_fn(pts[mask], ctx), dtype=np.float64).ravel()
            flat = u.ravel().copy()
            flat[np.where(mask)[0][:len(vals)]] = vals[:mask.sum()]
            u = flat.reshape(nx, ny)
        except Exception:
            continue
    return u


def _apply_neumann_2d(
    u: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    conditions,
    ctx: Dict,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Apply NeumannBC (flux) at boundary nodes using one-sided FD."""
    if not conditions:
        return u
    pts = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1).reshape(-1, 2).astype(np.float32)
    nx, ny = u.shape
    for cond in conditions:
        if getattr(cond, "kind", "") != "neumann":
            continue
        sel_fn = getattr(cond, "selector", None)
        val_fn = getattr(cond, "value_fn", None)
        if not callable(sel_fn) or not callable(val_fn):
            continue
        try:
            mask = np.asarray(sel_fn(pts, ctx), dtype=bool)
            if not mask.any():
                continue
            fluxes = np.asarray(val_fn(pts[mask], ctx), dtype=np.float64).ravel()
            idxs   = np.where(mask)[0]
            for k, idx in enumerate(idxs[:len(fluxes)]):
                i, j = divmod(int(idx), ny)
                flux  = float(fluxes[k])
                if i == 0:          u[0, j]   = u[1, j]   - dx * flux
                elif i == nx - 1:   u[-1, j]  = u[-2, j]  + dx * flux
                elif j == 0:        u[i, 0]   = u[i, 1]   - dy * flux
                elif j == ny - 1:   u[i, -1]  = u[i, -2]  + dy * flux
        except Exception:
            continue
    return u


def _apply_robin_2d(
    u: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    conditions,
    ctx: Dict,
    dx: float,
    dy: float,
    radiative_iters: int = 8,
) -> np.ndarray:
    """Apply RobinBC: a*u + b*(du/dn) [+ eps*sigma*(u^4 - t_env^4)] = g at
    boundary nodes, using the true outward normal at each boundary (unlike
    ``_apply_neumann_2d``'s plain coordinate-direction one-sided
    difference -- so a single Robin condition applies consistently
    whichever edge(s) it selects).

    Coefficients ``(a, b)`` and the optional nonlinear radiative term come
    from ``ctx["robin_coeffs"][cond.name]`` = ``{"a": .., "b": ..,
    "radiative": {"epsilon": .., "sigma": .., "t_env": ..}}``, defaulting
    to ``a=1, b=1``, no radiative term -- matching the convention already
    used by the PINN-training-side Robin residual in
    ``pinneapple_physics.pinn_solver.compiler.compile``. ``value_fn``
    supplies the right-hand side ``g``.

    When a radiative term is present, each boundary node is relaxed with a
    few fixed-point (Picard) iterations on a linearized radiative
    coefficient (standard practice for a nonlinear Robin BC) rather than a
    first-order approximation of the T^4 term itself.
    """
    if not conditions:
        return u
    nx, ny = u.shape
    pts = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1).reshape(-1, 2).astype(np.float32)
    robin_coeffs = ctx.get("robin_coeffs") or {}
    for cond in conditions:
        if getattr(cond, "kind", "") != "robin":
            continue
        sel_fn = getattr(cond, "selector", None)
        val_fn = getattr(cond, "value_fn", None)
        if not callable(sel_fn) or not callable(val_fn):
            continue
        name = getattr(cond, "name", None)
        coeffs = robin_coeffs.get(name, {}) if isinstance(name, str) else {}
        a0 = float(coeffs.get("a", 1.0))
        b0 = float(coeffs.get("b", 1.0))
        radiative = coeffs.get("radiative")
        try:
            mask = np.asarray(sel_fn(pts, ctx), dtype=bool)
            if not mask.any():
                continue
            g_vals = np.asarray(val_fn(pts[mask], ctx), dtype=np.float64).ravel()
            idxs = np.where(mask)[0]
            for k, idx in enumerate(idxs[: len(g_vals)]):
                i, j = divmod(int(idx), ny)
                if i == 0:
                    neighbor, step = u[1, j], dx
                elif i == nx - 1:
                    neighbor, step = u[-2, j], dx
                elif j == 0:
                    neighbor, step = u[i, 1], dy
                elif j == ny - 1:
                    neighbor, step = u[i, -2], dy
                else:
                    continue  # not a boundary node -- selector picked an interior point
                g = float(g_vals[k])
                coef = a0 + b0 / step
                rhs0 = g + (b0 / step) * neighbor
                if radiative is None:
                    u_b = rhs0 / coef if abs(coef) > 1e-300 else neighbor
                else:
                    eps = float(radiative.get("epsilon", 0.0))
                    sigma = float(radiative.get("sigma", 5.670374419e-8))
                    t_env = float(radiative.get("t_env", 0.0))
                    u_b = float(u[i, j])
                    if not np.isfinite(u_b):
                        u_b = t_env if t_env else neighbor
                    for _ in range(radiative_iters):
                        h_rad = eps * sigma * (u_b**2 + t_env**2) * (u_b + t_env)
                        u_b = (rhs0 + h_rad * t_env) / (coef + h_rad)
                if i == 0:
                    u[0, j] = u_b
                elif i == nx - 1:
                    u[-1, j] = u_b
                elif j == 0:
                    u[i, 0] = u_b
                elif j == ny - 1:
                    u[i, -1] = u_b
        except Exception:
            continue
    return u


def _robin_edge_specs(
    conditions,
    ctx: Dict,
    x: np.ndarray,
    y: np.ndarray,
    nx: int,
    ny: int,
) -> Dict[str, Optional[Dict[str, np.ndarray]]]:
    """Collect RobinBC conditions into per-edge coefficient arrays, for
    threading through the transient ADI solver (``_adi_heat_2d``), which
    -- unlike the post-hoc ``_apply_robin_2d`` used by the elliptic solver
    -- needs the boundary condition available at every timestep, not just
    once at the end.

    Each of the four edges (x0, x1, y0, y1) is assumed to be selected by at
    most one Robin condition, uniformly across that edge's interior
    (transverse) nodes -- the common case for a real boundary condition,
    and the one this whole-edge, time-evolving path is scoped to. A
    condition is assigned to whichever edge most of its selected points lie
    on; partial/mixed selections are not split across edges here (use the
    post-hoc ``_apply_robin_2d`` path -- i.e. the elliptic solver, or a
    steady-state solve -- for anything less regular than that).

    Returns ``{"x0": spec_or_None, "x1": ..., "y0": ..., "y1": ...}`` where
    each spec is ``{"a": (m,), "b": (m,), "g": (m,), "radiative": {...} or
    absent}``, with ``m = ny`` for the x-edges (indexed by j) or ``m = nx``
    for the y-edges (indexed by i).
    """
    edges: Dict[str, Optional[Dict[str, np.ndarray]]] = {"x0": None, "x1": None, "y0": None, "y1": None}
    if not conditions:
        return edges
    pts = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1).reshape(-1, 2).astype(np.float32)
    robin_coeffs = ctx.get("robin_coeffs") or {}
    for cond in conditions:
        if getattr(cond, "kind", "") != "robin":
            continue
        sel_fn = getattr(cond, "selector", None)
        val_fn = getattr(cond, "value_fn", None)
        if not callable(sel_fn) or not callable(val_fn):
            continue
        name = getattr(cond, "name", None)
        coeffs = robin_coeffs.get(name, {}) if isinstance(name, str) else {}
        a0 = float(coeffs.get("a", 1.0))
        b0 = float(coeffs.get("b", 1.0))
        radiative = coeffs.get("radiative")
        try:
            mask = np.asarray(sel_fn(pts, ctx), dtype=bool)
        except Exception:
            continue
        if not mask.any():
            continue
        idxs = np.where(mask)[0]

        counts = {"x0": 0, "x1": 0, "y0": 0, "y1": 0}
        for idx in idxs:
            i, j = divmod(int(idx), ny)
            if i == 0:
                counts["x0"] += 1
            elif i == nx - 1:
                counts["x1"] += 1
            if j == 0:
                counts["y0"] += 1
            elif j == ny - 1:
                counts["y1"] += 1
        edge = max(counts, key=counts.get)
        if counts[edge] == 0:
            continue

        m = ny if edge in ("x0", "x1") else nx
        try:
            g_vals = np.asarray(val_fn(pts[mask], ctx), dtype=np.float64).ravel()
        except Exception:
            continue
        g_full = np.zeros(m)
        for k, idx in enumerate(idxs[: len(g_vals)]):
            i, j = divmod(int(idx), ny)
            pos = j if edge in ("x0", "x1") else i
            if 0 <= pos < m:
                g_full[pos] = g_vals[k]

        spec: Dict[str, np.ndarray] = {"a": np.full(m, a0), "b": np.full(m, b0), "g": g_full}
        if radiative is not None:
            spec["radiative"] = {
                "epsilon": np.full(m, float(radiative.get("epsilon", 0.0))),
                "sigma": np.full(m, float(radiative.get("sigma", 5.670374419e-8))),
                "t_env": np.full(m, float(radiative.get("t_env", 0.0))),
            }
        edges[edge] = spec
    return edges


def _robin_tridiag_terms(u_prev: float, spec: Dict[str, np.ndarray], k: int, step: float):
    """(diag, off_diag, rhs) for one boundary node of a tridiagonal row,
    embedding the linear part of a*u + b*(du/dn) = g fully implicitly (the
    off-diagonal couples to the interior neighbor within the same solve)
    and any radiative term semi-implicitly, linearized about ``u_prev``
    (the node's value at the start of the current sweep)."""
    a0, b0, g0 = spec["a"][k], spec["b"][k], spec["g"][k]
    diag = a0 + b0 / step
    off = -b0 / step
    rhs = g0
    radiative = spec.get("radiative")
    if radiative is not None:
        eps, sigma, t_env = radiative["epsilon"][k], radiative["sigma"][k], radiative["t_env"][k]
        h_rad = eps * sigma * (u_prev**2 + t_env**2) * (u_prev + t_env)
        diag += h_rad
        rhs += h_rad * t_env
    return diag, off, rhs


# ─────────────────────────────────────────────────────────────────────────────
# Core FD kernels
# ─────────────────────────────────────────────────────────────────────────────

def _sor_2d(
    f: np.ndarray,
    u: np.ndarray,
    dx: float,
    dy: float,
    iters: int = 8000,
    omega: float = 1.5,
    tol: float = 1e-9,
    k2: float = 0.0,
) -> np.ndarray:
    """SOR solver for (∇² + k²)u = -f  (k²=0 → Poisson/Laplace).

    Uses a red-black (checkerboard) sweep order: all "red" interior points
    (i+j even) are updated first from the current grid, then all "black"
    points (i+j odd) are updated from the just-refreshed red values --
    genuine Gauss-Seidel-SOR semantics, vectorized over each color. A
    single full-array update using only old values on both sides (as if
    every point saw its neighbors from the previous iteration) is weighted
    Jacobi, not SOR -- stable only for omega below roughly 1, and diverges
    for omega > 1 like this solver's own omega=1.5 default. Red-black
    ordering is what actually makes omega in (0, 2) valid.
    """
    dx2, dy2 = dx * dx, dy * dy
    denom = 2.0 / dx2 + 2.0 / dy2 - k2
    denom = max(abs(denom), 1e-30) * np.sign(denom) if denom != 0 else 1e-30
    nx, ny = u.shape
    if nx <= 2 or ny <= 2:
        return u
    ii, jj = np.meshgrid(np.arange(1, nx - 1), np.arange(1, ny - 1), indexing="ij")
    color_masks = ((ii + jj) % 2 == 0, (ii + jj) % 2 == 1)
    for _ in range(iters):
        u_prev = u[1:-1, 1:-1].copy()
        for mask in color_masks:
            interior = u[1:-1, 1:-1]
            # i is the x-index (row shift -> dx2), j is the y-index (column
            # shift -> dy2) -- these were swapped before (invisible on a
            # square grid, where dx2 == dy2, but wrong for any nx != ny
            # grid with dx != dy).
            new_val = (
                (u[2:, 1:-1] + u[:-2, 1:-1]) / dx2
                + (u[1:-1, 2:] + u[1:-1, :-2]) / dy2
                + f[1:-1, 1:-1]
            ) / denom
            u[1:-1, 1:-1] = np.where(mask, (1 - omega) * interior + omega * new_val, interior)
        if np.max(np.abs(u[1:-1, 1:-1] - u_prev)) < tol:
            break
    return u


def _tridiag_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Thomas algorithm for tridiagonal system a·x[i-1] + b·x[i] + c·x[i+1] = d[i]."""
    n = len(d)
    c_ = np.zeros(n, dtype=np.float64)
    d_ = d.copy().astype(np.float64)
    x  = np.zeros(n, dtype=np.float64)
    c_[0] = c[0] / b[0]
    d_[0] = d_[0] / b[0]
    for i in range(1, n):
        m     = b[i] - a[i] * c_[i - 1]
        c_[i] = c[i] / m
        d_[i] = (d_[i] - a[i] * d_[i - 1]) / m
    x[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]
    return x


def _adi_heat_2d(
    u: np.ndarray,
    alpha: float,
    dx: float,
    dy: float,
    dt: float,
    nt: int,
    theta: float = 0.5,
    robin_x0: Optional[Dict[str, np.ndarray]] = None,
    robin_x1: Optional[Dict[str, np.ndarray]] = None,
    robin_y0: Optional[Dict[str, np.ndarray]] = None,
    robin_y1: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """ADI (Douglas-Rachford) theta-method for 2D heat equation.

    theta=0 → explicit (CFL: α dt/h² ≤ 0.5)
    theta=0.5 → Crank-Nicolson (2nd order, unconditionally stable)
    theta=1 → fully implicit (1st order, unconditionally stable)

    Boundary treatment per edge: Dirichlet-hold (the previous behavior,
    still the default) unless a ``robin_*`` spec is given for that edge --
    see ``_robin_edge_specs`` -- in which case a*u + b*(du/dn) [+ a
    radiative term] = g is solved for every interior transverse node of
    that edge, at every timestep, coupled implicitly into the same
    tridiagonal system as the interior update (radiative term excepted,
    which is linearized about the value at the start of the current sweep
    -- see ``_robin_tridiag_terms``). The four corner nodes keep the
    Dirichlet-hold behavior regardless, since each sweep only walks the
    other sweep's interior range.
    """
    nx, ny = u.shape
    rx = alpha * dt / dx ** 2
    ry = alpha * dt / dy ** 2

    for _ in range(nt):
        # ── X-sweep (implicit in x, explicit in y) ───────────────────────────
        rhs = u.copy()
        rhs[1:-1, 1:-1] += (1 - theta) * ry * (
            u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]
        )
        u_half = u.copy()
        for j in range(1, ny - 1):
            col  = rhs[:, j].copy()
            a    = np.full(nx, -theta * rx)
            b    = np.full(nx, 1 + 2 * theta * rx)
            c    = np.full(nx, -theta * rx)
            if robin_x0 is not None:
                diag, off, r = _robin_tridiag_terms(u[0, j], robin_x0, j, dx)
                a[0] = 0.0; b[0] = diag; c[0] = off; col[0] = r
            else:
                a[0] = 0.0; b[0] = 1.0; c[0] = 0.0   # Dirichlet hold
            if robin_x1 is not None:
                diag, off, r = _robin_tridiag_terms(u[-1, j], robin_x1, j, dx)
                a[-1] = off; b[-1] = diag; c[-1] = 0.0; col[-1] = r
            else:
                a[-1]= 0.0; b[-1]= 1.0; c[-1]= 0.0
            u_half[:, j] = _tridiag_solve(a, b, c, col)

        # ── Y-sweep (implicit in y, explicit in x) ───────────────────────────
        rhs2 = u_half.copy()
        rhs2[1:-1, 1:-1] += (1 - theta) * rx * (
            u_half[2:, 1:-1] - 2 * u_half[1:-1, 1:-1] + u_half[:-2, 1:-1]
        )
        u_new = u_half.copy()
        for i in range(1, nx - 1):
            row  = rhs2[i, :].copy()
            a    = np.full(ny, -theta * ry)
            b    = np.full(ny, 1 + 2 * theta * ry)
            c    = np.full(ny, -theta * ry)
            if robin_y0 is not None:
                diag, off, r = _robin_tridiag_terms(u_half[i, 0], robin_y0, i, dy)
                a[0] = 0.0; b[0] = diag; c[0] = off; row[0] = r
            else:
                a[0] = 0.0; b[0] = 1.0; c[0] = 0.0
            if robin_y1 is not None:
                diag, off, r = _robin_tridiag_terms(u_half[i, -1], robin_y1, i, dy)
                a[-1] = off; b[-1] = diag; c[-1] = 0.0; row[-1] = r
            else:
                a[-1]= 0.0; b[-1]= 1.0; c[-1]= 0.0
            u_new[i, :] = _tridiag_solve(a, b, c, row)

        u = u_new
    return u


def _leapfrog_wave_1d(
    u0: np.ndarray,
    v0: np.ndarray,
    c: float,
    dx: float,
    dt: float,
    nt: int,
) -> np.ndarray:
    """Leapfrog for 1D wave: ∂²u/∂t² = c²∂²u/∂x². Returns (nx, nt+1)."""
    r2    = (c * dt / dx) ** 2
    u     = u0.copy()
    u_old = u0 - dt * v0  # virtual previous step
    traj  = [u0.copy()]
    for _ in range(nt):
        lap   = np.roll(u, -1) - 2 * u + np.roll(u, 1)
        u_new = 2 * u - u_old + r2 * lap
        u_new[0] = 0.0
        u_new[-1] = 0.0
        u_old, u = u, u_new
        traj.append(u.copy())
    return np.stack(traj, axis=1)


def _ftcs_burgers_1d(
    u0: np.ndarray,
    nu: float,
    dx: float,
    dt: float,
    nt: int,
) -> np.ndarray:
    """FTCS + upwind for 1D viscous Burgers. Returns (nx, nt+1)."""
    u    = u0.copy()
    traj = [u.copy()]
    for _ in range(nt):
        adv = np.where(
            u >= 0,
            u * (u - np.roll(u, 1)) / dx,
            u * (np.roll(u, -1) - u) / dx,
        )
        diff  = nu * (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx ** 2
        u_new = u + dt * (-adv + diff)
        u_new[0] = u[0]
        u_new[-1] = u[-1]
        u = u_new
        traj.append(u.copy())
    return np.stack(traj, axis=1)


def _upwind_advdiff_1d(
    u0: np.ndarray,
    v: float,
    D: float,
    dx: float,
    dt: float,
    nt: int,
) -> np.ndarray:
    """Upwind + FTCS for 1D advection-diffusion. Returns (nx, nt+1)."""
    u    = u0.copy()
    traj = [u.copy()]
    for _ in range(nt):
        adv   = (v * (u - np.roll(u, 1)) / dx if v >= 0
                 else v * (np.roll(u, -1) - u) / dx)
        diff  = D * (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx ** 2
        u_new = u + dt * (-adv + diff)
        u_new[0] = u[0]
        u_new[-1] = u[-1]
        u = u_new
        traj.append(u.copy())
    return np.stack(traj, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Main solver class
# ─────────────────────────────────────────────────────────────────────────────

@SolverRegistry.register(
    name="fdm",
    family="pde",
    description="Finite Difference Method — problem-agnostic (Poisson, Heat, Wave, Burgers, Advection-Diffusion).",
    tags=["fdm", "pde", "agnostic"],
)
class FDMSolver(SolverBase):
    """Problem-agnostic FDM solver.

    Instantiate directly or via ``FDMSolver.from_problem_spec(spec, ...)``.

    Parameters
    ----------
    nx, ny : spatial grid size (default 64×64)
    nt     : number of time steps for parabolic/hyperbolic problems
    iters  : SOR iterations for elliptic problems
    omega  : SOR relaxation factor (1 < omega < 2 for over-relaxation)
    theta  : ADI θ-parameter (0=explicit, 0.5=Crank-Nicolson, 1=implicit)
    tol    : convergence tolerance for SOR
    """

    def __init__(
        self,
        nx: int = 64,
        ny: int = 64,
        nt: int = 500,
        iters: int = 8000,
        omega: float = 1.5,
        theta: float = 0.5,
        tol: float = 1e-9,
    ):
        super().__init__()
        self.nx    = int(nx)
        self.ny    = int(ny)
        self.nt    = int(nt)
        self.iters = int(iters)
        self.omega = float(omega)
        self.theta = float(theta)
        self.tol   = float(tol)

    @classmethod
    def from_problem_spec(
        cls,
        spec,
        nx: int = 64,
        ny: int = 64,
        nt: int = 500,
        **kwargs,
    ) -> "FDMSolver":
        """Build a solver configured for the given ProblemSpec."""
        return cls(nx=nx, ny=ny, nt=nt, **kwargs)

    # ── Public API ────────────────────────────────────────────────────────────

    def solve_from_spec(self, spec, ctx_extra: Optional[Dict[str, Any]] = None) -> SolverOutput:
        """Auto-solve a ProblemSpec using the appropriate FD scheme.

        ``ctx_extra`` is merged into the solver's ctx dict -- use it to
        supply ``robin_coeffs`` (see the module docstring) for any RobinBC
        condition in ``spec.conditions``, e.g.::

            solver.solve_from_spec(spec, ctx_extra={
                "robin_coeffs": {"surface": {"a": h, "b": -k}},
            })
        """
        kind       = getattr(spec.pde, "kind",   "").lower()
        params     = dict(getattr(spec.pde, "params", {}))
        domain     = dict(getattr(spec, "domain_bounds", {}))
        conditions = getattr(spec, "conditions", ())
        coords     = tuple(getattr(spec, "coords", ("x", "y")))

        ctx = {"bounds": {c: domain.get(c, (0.0, 1.0)) for c in coords}}
        if ctx_extra:
            ctx.update(ctx_extra)

        if "burgers" in kind:
            return self._burgers(domain, coords, conditions, params, ctx)
        if "advection" in kind or "convection_diffusion" in kind:
            return self._advdiff(domain, coords, conditions, params, ctx)
        if "wave" in kind:
            return self._wave(domain, coords, conditions, params, ctx)
        if "heat" in kind or ("diffusion" in kind and "advection" not in kind):
            return self._heat(domain, coords, conditions, params, ctx)
        # Elliptic: Poisson / Laplace / Helmholtz (default)
        return self._poisson(domain, coords, conditions, params, ctx)

    # ── PDE dispatchers ───────────────────────────────────────────────────────

    def _poisson(self, domain, coords, conditions, params, ctx) -> SolverOutput:
        spatial = [c for c in coords if c != "t"]
        c0 = spatial[0] if spatial else "x"
        c1 = spatial[1] if len(spatial) > 1 else "y"
        x0, x1 = domain.get(c0, (0.0, 1.0))
        y0, y1 = domain.get(c1, (0.0, 1.0))

        x, y, XX, YY, dx, dy = _build_2d_grid(x0, x1, self.nx, y0, y1, self.ny)
        source = float(params.get("source", 0.0))
        f  = np.full((self.nx, self.ny), source)
        u  = np.zeros((self.nx, self.ny))
        k2 = float(params.get("k2", params.get("k", 0.0))) ** 2

        u = _apply_dirichlet_2d(u, x, y, conditions, ctx)
        has_robin = any(getattr(c, "kind", "") == "robin" for c in (conditions or ()))
        if has_robin:
            # _sor_2d never touches the boundary rows/columns itself -- a
            # Robin boundary has to be relaxed together with the interior
            # (like block Gauss-Seidel) rather than patched on once after
            # the interior has already converged against whatever the
            # boundary happened to be (0, absent a Dirichlet condition).
            # Many outer passes with a handful of SOR sweeps each converges
            # much faster (wall-clock) than few outer passes each run to
            # full SOR convergence against a stale boundary -- the coupling
            # itself is what's slow to converge, not the interior solve.
            # Each outer pass runs SOR to its own full convergence against
            # the current boundary (cheap after the first couple of passes,
            # since it warm-starts from the previous outer pass's result)
            # rather than a handful of sweeps -- a truncated inner solve can
            # converge the *outer* loop to a self-consistent fixed point of
            # the truncated map that isn't actually a solution of the
            # underlying system (observed concretely with a radiative
            # term: a 50-sweep inner budget settled onto a stable but
            # wrong boundary value, while a fully-converged inner solve
            # reached the correct one -- the nonlinear radiative case is
            # measurably more sensitive to this than the linear-Robin one).
            for _ in range(800):
                u_prev = u.copy()
                u = _sor_2d(f, u, dx, dy, self.iters, self.omega, self.tol, k2=k2)
                u = _apply_dirichlet_2d(u, x, y, conditions, ctx)
                u = _apply_robin_2d(u, x, y, conditions, ctx, dx, dy)
                if np.max(np.abs(u - u_prev)) < self.tol * 10:
                    break
        else:
            u = _sor_2d(f, u, dx, dy, self.iters, self.omega, self.tol, k2=k2)
        u = _apply_neumann_2d(u, x, y, conditions, ctx, dx, dy)

        return SolverOutput(
            result=torch.from_numpy(u.astype(np.float32)),
            losses={"residual": torch.tensor(0.0)},
            extras={
                "coords": {c0: x.astype(np.float32), c1: y.astype(np.float32)},
                "grid": {"XX": XX.astype(np.float32), "YY": YY.astype(np.float32)},
                "method": "sor",
            },
        )

    def _heat(self, domain, coords, conditions, params, ctx) -> SolverOutput:
        spatial = [c for c in coords if c != "t"]
        c0 = spatial[0] if spatial else "x"
        c1 = spatial[1] if len(spatial) > 1 else "y"
        x0, x1 = domain.get(c0, (0.0, 1.0))
        y0, y1 = domain.get(c1, (0.0, 1.0))
        t0, t1 = domain.get("t", (0.0, 1.0))

        alpha = float(params.get("alpha", params.get("k", params.get("diffusivity", 0.01))))
        x, y, _, _, dx, dy = _build_2d_grid(x0, x1, self.nx, y0, y1, self.ny)

        # Time step: CFL for stability (safety factor 0.4)
        dt_max = 0.4 * min(dx, dy) ** 2 / max(alpha, 1e-30)
        dt     = min(dt_max, (t1 - t0) / max(self.nt, 1))
        nt     = max(1, int((t1 - t0) / dt))

        u0 = np.zeros((self.nx, self.ny))
        u0 = _apply_dirichlet_2d(u0, x, y, conditions, ctx)
        robin_edges = _robin_edge_specs(conditions, ctx, x, y, self.nx, self.ny)
        u  = _adi_heat_2d(
            u0, alpha, dx, dy, dt, nt, self.theta,
            robin_x0=robin_edges["x0"], robin_x1=robin_edges["x1"],
            robin_y0=robin_edges["y0"], robin_y1=robin_edges["y1"],
        )
        u  = _apply_neumann_2d(u, x, y, conditions, ctx, dx, dy)

        t_arr = np.linspace(t0, t0 + nt * dt, nt + 1, dtype=np.float32)
        return SolverOutput(
            result=torch.from_numpy(u.astype(np.float32)),
            losses={"residual": torch.tensor(0.0)},
            extras={
                "coords": {c0: x.astype(np.float32), c1: y.astype(np.float32), "t": t_arr},
                "alpha": alpha, "dt": dt, "nt": nt, "method": "adi_theta",
            },
        )

    def _wave(self, domain, coords, conditions, params, ctx) -> SolverOutput:
        spatial = [c for c in coords if c != "t"]
        c0 = spatial[0] if spatial else "x"
        x0, x1 = domain.get(c0, (-1.0, 1.0))
        t0, t1 = domain.get("t", (0.0, 1.0))

        c_speed = float(params.get("c", params.get("wave_speed", 1.0)))
        x  = _linspace(x0, x1, self.nx)
        dx = _spacing(x0, x1, self.nx)
        # CFL: c dt/dx ≤ 1
        dt = 0.9 * dx / max(abs(c_speed), 1e-12)
        nt = min(self.nt, max(1, int((t1 - t0) / dt)))

        u0 = np.sin(np.pi * (x - x0) / (x1 - x0))  # default: one-period sine
        v0 = np.zeros_like(u0)

        traj = _leapfrog_wave_1d(u0, v0, c_speed, dx, dt, nt)
        t_arr = np.linspace(t0, t0 + nt * dt, nt + 1, dtype=np.float32)
        return SolverOutput(
            result=torch.from_numpy(traj.astype(np.float32)),
            losses={"residual": torch.tensor(0.0)},
            extras={
                "coords": {c0: x.astype(np.float32), "t": t_arr},
                "c": c_speed, "cfl": c_speed * dt / dx, "method": "leapfrog",
            },
        )

    def _burgers(self, domain, coords, conditions, params, ctx) -> SolverOutput:
        spatial = [c for c in coords if c != "t"]
        c0 = spatial[0] if spatial else "x"
        x0, x1 = domain.get(c0, (-1.0, 1.0))
        t0, t1 = domain.get("t", (0.0, 1.0))

        nu = float(params.get("nu", params.get("viscosity", 0.01)))
        x  = _linspace(x0, x1, self.nx)
        dx = _spacing(x0, x1, self.nx)
        # Stability: diffusive CFL
        dt = 0.4 * dx ** 2 / max(nu, 1e-12)
        nt = min(self.nt, max(1, int((t1 - t0) / dt)))

        u0 = -np.sin(np.pi * (x - x0) / (x1 - x0))
        traj  = _ftcs_burgers_1d(u0, nu, dx, dt, nt)
        t_arr = np.linspace(t0, t0 + nt * dt, nt + 1, dtype=np.float32)
        return SolverOutput(
            result=torch.from_numpy(traj.astype(np.float32)),
            losses={"residual": torch.tensor(0.0)},
            extras={
                "coords": {c0: x.astype(np.float32), "t": t_arr},
                "nu": nu, "method": "ftcs_upwind",
            },
        )

    def _advdiff(self, domain, coords, conditions, params, ctx) -> SolverOutput:
        spatial = [c for c in coords if c != "t"]
        c0 = spatial[0] if spatial else "x"
        x0, x1 = domain.get(c0, (0.0, 1.0))
        t0, t1 = domain.get("t", (0.0, 1.0))

        v = float(params.get("v", params.get("velocity", 1.0)))
        D = float(params.get("D", params.get("diffusivity", 0.01)))
        x  = _linspace(x0, x1, self.nx)
        dx = _spacing(x0, x1, self.nx)
        dt = min(
            0.4 * dx / max(abs(v), 1e-12),
            0.4 * dx ** 2 / max(D, 1e-12),
        )
        nt = min(self.nt, max(1, int((t1 - t0) / dt)))

        x_mid = (x0 + x1) / 2.0
        u0    = np.exp(-50.0 * (x - x_mid) ** 2)  # Gaussian IC
        traj  = _upwind_advdiff_1d(u0, v, D, dx, dt, nt)
        t_arr = np.linspace(t0, t0 + nt * dt, nt + 1, dtype=np.float32)
        return SolverOutput(
            result=torch.from_numpy(traj.astype(np.float32)),
            losses={"residual": torch.tensor(0.0)},
            extras={
                "coords": {c0: x.astype(np.float32), "t": t_arr},
                "v": v, "D": D, "method": "upwind",
            },
        )

    # ── Legacy interface ──────────────────────────────────────────────────────

    def forward(
        self,
        f: Optional[torch.Tensor] = None,
        bc: Optional[torch.Tensor] = None,
        *,
        dx: float = 1.0,
        dy: float = 1.0,
        spec=None,
    ) -> SolverOutput:
        """Unified forward.

        - ``forward(spec=spec)``              → problem-agnostic solve
        - ``forward(f, bc, dx=.., dy=..)``   → direct 2D Poisson (legacy)
        """
        if spec is not None:
            return self.solve_from_spec(spec)
        if f is not None:
            f_np  = f.detach().cpu().numpy() if isinstance(f, torch.Tensor) else np.asarray(f)
            bc_np = (bc.detach().cpu().numpy() if isinstance(bc, torch.Tensor)
                     else np.zeros_like(f_np))
            u = _sor_2d(f_np, bc_np.copy(), dx, dy, self.iters, self.omega, self.tol)
            return SolverOutput(
                result=torch.from_numpy(u.astype(np.float32)),
                losses={},
                extras={"iters": self.iters, "method": "sor"},
            )
        raise ValueError("FDMSolver.forward: provide either `spec` or `(f, bc)` tensors.")
