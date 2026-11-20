"""3-D elliptic problems on a structured grid: ``lap(u) + q u = rhs`` with Dirichlet walls.

One solver covers the three kinds the scope table had no 3-D entry for: Poisson (``q = 0``, ``rhs = -f``),
Laplace (``q = 0``, ``rhs = 0``) and Helmholtz (``q = k^2``, ``rhs = f``). Seven-point central differences with
unequal spacing on each axis, red-black successive over-relaxation with the optimal relaxation factor of a cube
(``2 / (1 + sin(pi / N))``), walls held at values you give.

    from pinneapple_simulation.numerical_solvers.elliptic3d import Elliptic3D, solve_elliptic_3d
    cfg = Elliptic3D(nx=33, ny=33, nz=33, rhs=lambda X, Y, Z: 3 * np.pi**2 * np.sin(np.pi*X)*np.sin(np.pi*Y)*np.sin(np.pi*Z))
    out = solve_elliptic_3d(cfg)
    out["iterations"], out["residual"]

What it does not claim: the iteration is first order in the relaxation (the *solution* is second order in the
grid), it needs a diagonally dominant operator (``q`` below the first eigenvalue of the discrete Laplacian, which
is what keeps a Helmholtz problem away from resonance), and it has Dirichlet walls only -- a Neumann or Robin
condition is refused by :class:`Elliptic3DFDMSolver` rather than dropped.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry

__all__ = ["Elliptic3D", "solve_elliptic_3d", "Elliptic3DFDMSolver"]


@dataclass
class Elliptic3D:
    """Grid and coefficient of ``lap(u) + q u = rhs`` on a box.

    ``rhs`` is a constant or a callable ``f(X, Y, Z)`` evaluated on the grid; ``q`` is the zeroth-order coefficient
    (0 for Poisson/Laplace, ``k^2`` for Helmholtz). ``wall`` is the prescribed boundary value: a constant, a
    callable ``g(X, Y, Z)``, or a full ``(nx, ny, nz)`` array (walls that differ face to face).
    """

    nx: int = 33
    ny: int = 33
    nz: int = 33
    lx: float = 1.0
    ly: float = 1.0
    lz: float = 1.0
    q: float = 0.0
    rhs: float | Callable = 0.0
    wall: float | Callable | np.ndarray = 0.0
    tol: float = 1e-9
    max_iter: int = 200_000
    omega: float | None = None

    def grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (np.linspace(0.0, self.lx, self.nx), np.linspace(0.0, self.ly, self.ny),
                np.linspace(0.0, self.lz, self.nz))


def _as_field(value: Any, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """A constant, a callable of (X, Y, Z), or an array already on the grid."""
    if callable(value):
        return np.asarray(value(X, Y, Z), dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return np.full(X.shape, float(array))
    if array.shape != X.shape:
        raise ValueError(f"expected a field of shape {X.shape}, got {array.shape}")
    return array


def solve_elliptic_3d(cfg: Elliptic3D) -> dict:
    """Solve ``lap(u) + q u = rhs`` with the walls of ``cfg``. Returns the field, the iterations and the residual."""
    x, y, z = cfg.grid()
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    rhs = _as_field(cfg.rhs, X, Y, Z)
    u = _as_field(cfg.wall, X, Y, Z).copy()

    dx, dy, dz = (cfg.lx / max(cfg.nx - 1, 1), cfg.ly / max(cfg.ny - 1, 1), cfg.lz / max(cfg.nz - 1, 1))
    ax, ay, az = 1.0 / dx**2, 1.0 / dy**2, 1.0 / dz**2
    denom = 2.0 * (ax + ay + az) - cfg.q
    if denom <= 0:
        raise ValueError(
            f"lap(u) + q u = rhs with q = {cfg.q} is not diagonally dominant on this grid "
            f"(2 sum 1/d^2 = {2.0 * (ax + ay + az):.6g}); q has to stay below that.")
    omega = float(cfg.omega) if cfg.omega is not None else 2.0 / (1.0 + np.sin(np.pi / max(cfg.nx, cfg.ny, cfg.nz)))
    inner = u[1:-1, 1:-1, 1:-1]
    rhs_inner = rhs[1:-1, 1:-1, 1:-1]
    colour = np.indices(inner.shape).sum(axis=0) % 2
    red, black = colour == 0, colour == 1
    residual = float("inf")
    iterations = 0
    history: list[float] = []
    scale = max(1.0, float(np.max(np.abs(rhs))))
    while iterations < cfg.max_iter:
        for mask in (red, black):
            neighbours = ((u[:-2, 1:-1, 1:-1] + u[2:, 1:-1, 1:-1]) * ax
                          + (u[1:-1, :-2, 1:-1] + u[1:-1, 2:, 1:-1]) * ay
                          + (u[1:-1, 1:-1, :-2] + u[1:-1, 1:-1, 2:]) * az)
            update = (neighbours - rhs_inner) / denom
            inner[mask] = (1.0 - omega) * inner[mask] + omega * update[mask]
        iterations += 1
        if iterations % 10 == 0 or iterations == 1:
            neighbours = ((u[:-2, 1:-1, 1:-1] + u[2:, 1:-1, 1:-1]) * ax
                          + (u[1:-1, :-2, 1:-1] + u[1:-1, 2:, 1:-1]) * ay
                          + (u[1:-1, 1:-1, :-2] + u[1:-1, 1:-1, 2:]) * az)
            residual = float(np.max(np.abs(neighbours - (2.0 * (ax + ay + az) - cfg.q) * inner - rhs_inner)))
            residual /= scale
            history.append(residual)
            if residual <= cfg.tol:
                break
    return {"u": u, "iterations": iterations, "residual": residual, "omega": omega, "dx": dx, "dy": dy, "dz": dz,
            "coords": {"x": x, "y": y, "z": z}, "residual_history": history}


@SolverRegistry.register(
    name="elliptic3d_fdm",
    family="pde",
    description="3-D Poisson / Laplace / Helmholtz on a box with Dirichlet walls: seven-point finite "
                "differences, unequal spacing, red-black SOR with the optimal factor",
    tags=["fdm", "poisson", "laplace", "helmholtz", "3d", "elliptic"],
)
class Elliptic3DFDMSolver(SolverBase):
    """``solve_from_spec`` for :func:`solve_elliptic_3d`.

    The spec's parameters (``spec.pde.params``): ``nx``/``ny``/``nz`` (default 33) or ``n`` for all three, ``k``
    for Helmholtz (``q = k^2``), ``source`` -- a constant or a callable ``f(x, y, z)`` -- with the sign convention
    of each kind (Poisson ``-lap(u) = f``, Helmholtz ``lap(u) + k^2 u = f``), and ``tol``/``max_iter``/``omega``
    for the iteration. The walls take the values of the spec's Dirichlet conditions, and 0 where a wall has none.
    """

    def solve_from_spec(self, spec) -> SolverOutput:
        kind = str(getattr(spec.pde, "kind", "")).lower().replace("-", "_").replace(" ", "_")
        if not any(k in kind for k in ("poisson", "laplace", "helmholtz")):
            raise ValueError(f"elliptic3d_fdm solves poisson/laplace/helmholtz, and the spec's kind is '{kind}'.")
        if int(getattr(spec, "dim", 3)) != 3:
            raise ValueError(f"elliptic3d_fdm is a 3-D solver; the spec is {getattr(spec, 'dim', '?')}-D.")
        params = dict(getattr(spec.pde, "params", {}) or {})
        bounds = dict(getattr(spec, "domain_bounds", {}) or {})
        lx = abs(bounds.get("x", (0.0, 1.0))[1] - bounds.get("x", (0.0, 1.0))[0])
        ly = abs(bounds.get("y", (0.0, 1.0))[1] - bounds.get("y", (0.0, 1.0))[0])
        lz = abs(bounds.get("z", (0.0, 1.0))[1] - bounds.get("z", (0.0, 1.0))[0])
        n = params.get("n")
        nx, ny, nz = (int(params.get("nx", n or 33)), int(params.get("ny", n or 33)), int(params.get("nz", n or 33)))
        source = params.get("source", params.get("f", 0.0))
        if "helmholtz" in kind:
            if params.get("k") is None:
                raise ValueError("elliptic3d_fdm needs the wavenumber in spec.pde.params['k'] for a Helmholtz "
                                 "problem (it solves lap(u) + k^2 u = f).")
            q = float(params["k"]) ** 2
            rhs: Any = source
        else:
            q = 0.0
            rhs = (lambda X, Y, Z: -np.asarray(source(X, Y, Z), dtype=np.float64)) if callable(source) else -float(
                source)

        conditions = tuple(getattr(spec, "conditions", ()) or ())
        others = [c for c in conditions if getattr(c, "kind", None) not in ("dirichlet", "initial")]
        if others:
            raise ValueError("elliptic3d_fdm holds the walls at the values of the spec's dirichlet conditions and "
                             f"has no other boundary: the spec asks for {sorted({c.kind for c in others})}.")
        cfg = Elliptic3D(nx=nx, ny=ny, nz=nz, lx=lx, ly=ly, lz=lz, q=q, rhs=rhs,
                         tol=float(params.get("tol", 1e-9)), max_iter=int(params.get("max_iter", 200_000)),
                         omega=(float(params["omega"]) if params.get("omega") is not None else None))
        x, y, z = cfg.grid()
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        wall = np.zeros(X.shape)
        if conditions:
            from . import _bc
            faces = [(slice(0, 1), slice(None), slice(None)), (slice(-1, None), slice(None), slice(None)),
                     (slice(None), slice(0, 1), slice(None)), (slice(None), slice(-1, None), slice(None)),
                     (slice(None), slice(None), slice(0, 1)), (slice(None), slice(None), slice(-1, None))]
            # Every wall node once per face it lies on (edges are shared): the order the slices walk them is the
            # order the flat indices come back in, so a value lands on the node it was evaluated at.
            pts = np.concatenate([np.column_stack([X[f].ravel(), Y[f].ravel(), Z[f].ravel()]) for f in faces])
            index = np.concatenate([_face_flat_indices(f, X) for f in faces])
            tensor = torch.from_numpy(pts.astype(np.float32))
            flat = wall.ravel()
            for cond in _bc.dirichlet_conditions(spec):
                picked = _bc.select(cond, tensor, boundary=torch.arange(tensor.shape[0])).detach().cpu().numpy()
                if picked.size == 0:
                    continue
                values = _bc.values(cond, tensor[picked]).detach().cpu().numpy()
                flat[index[picked]] = values
            wall = flat.reshape(X.shape)
        cfg.wall = wall

        out = solve_elliptic_3d(cfg)
        field = torch.from_numpy(np.asarray(out["u"], dtype=np.float32))
        return SolverOutput(
            result=field,
            losses={"residual": torch.tensor(float(out["residual"]))},
            extras={"method": "elliptic3d_fdm", "kind": kind, "q": q, "grid": (nx, ny, nz),
                    "iterations": int(out["iterations"]), "residual": float(out["residual"]),
                    "omega": float(out["omega"]), "coords": out["coords"],
                    "u_min": float(out["u"].min()), "u_max": float(out["u"].max())},
        )


def _face_flat_indices(face: tuple, X: np.ndarray) -> np.ndarray:
    """Flat indices (C order) of the nodes on one face of the grid, in the order the face slices walk them."""
    grid = np.arange(X.size).reshape(X.shape)
    return grid[face].ravel()
