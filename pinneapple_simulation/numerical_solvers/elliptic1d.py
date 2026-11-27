"""1-D elliptic problems on a line: ``u'' + q u = rhs`` with the two ends held.

One direct solve covers the three kinds the scope table had no 1-D entry for: Poisson (``q = 0``,
``rhs = -f``), Laplace (``q = 0``, ``rhs = 0``) and Helmholtz (``q = k^2``, ``rhs = f``). Three-point central
differences and a tridiagonal solve (``scipy.linalg.solve_banded``): no iteration, no tolerance to choose, and
the system is the discrete operator exactly.

    from pinneapple_simulation.numerical_solvers.elliptic1d import Elliptic1D, solve_elliptic_1d
    out = solve_elliptic_1d(Elliptic1D(n=101, rhs=1.0))        # -u'' = 1, u(0) = u(1) = 0
    out["u"].max()                                             # 0.125 = f L^2 / 8, the parabola's peak

What it does not claim: the solution is second order in the grid (three-point differences), ``q`` stays below the
first discrete eigenvalue or the operator stops being definite (a Helmholtz problem at resonance has no unique
answer), and the ends are Dirichlet -- :class:`Elliptic1DFDMSolver` refuses anything else by name.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry

__all__ = ["Elliptic1D", "solve_elliptic_1d", "Elliptic1DFDMSolver"]


@dataclass
class Elliptic1D:
    """Grid and coefficient of ``u'' + q u = rhs`` on ``[0, length]`` with the ends held.

    ``rhs`` is a constant or a callable ``f(x)``; ``q`` is the zeroth-order coefficient (0 for Poisson/Laplace,
    ``k^2`` for Helmholtz); ``u_left``/``u_right`` are the two end values.
    """

    n: int = 201
    length: float = 1.0
    q: float = 0.0
    rhs: float | Callable[[np.ndarray], np.ndarray] = 0.0
    u_left: float = 0.0
    u_right: float = 0.0


def solve_elliptic_1d(cfg: Elliptic1D) -> dict:
    """Solve ``u'' + q u = rhs`` with the ends of ``cfg``. Returns the field, the grid and the residual."""
    from scipy.linalg import solve_banded

    n = int(cfg.n)
    if n < 3:
        raise ValueError(f"at least 3 points are needed (two ends and one interior node), not {n}")
    x = np.linspace(0.0, cfg.length, n)
    h = cfg.length / (n - 1) if n > 1 else cfg.length
    rhs = (np.asarray(cfg.rhs(x), dtype=np.float64) if callable(cfg.rhs)
           else np.full(n, float(cfg.rhs)))

    diag = -2.0 / h**2 + cfg.q
    off = 1.0 / h**2
    # the discrete operator with Dirichlet ends has eigenvalues -4/h^2 sin^2(j pi h / (2L)) + q: it is invertible
    # while q stays below the smallest of them, and at (or above) it the problem has no unique answer
    lam_min = 4.0 / h**2 * np.sin(np.pi * h / (2.0 * cfg.length)) ** 2
    if cfg.q >= lam_min:
        raise ValueError(
            f"u'' + q u = rhs with q = {cfg.q} is not invertible on this grid: the discrete operator's smallest "
            f"eigenvalue is {lam_min:.6g}, and q has to stay below it.")
    # the two end rows are the Dirichlet data: 1 on the diagonal, everything else zero
    ab = np.zeros((3, n))
    ab[0, 1:] = off
    ab[1, :] = diag
    ab[2, :-1] = off
    b = rhs.copy()
    b[0] = b[-1] = 0.0
    # fold the prescribed ends into the right-hand side of the neighbouring interior nodes
    b[1] -= off * cfg.u_left
    b[-2] -= off * cfg.u_right
    ab[1, 0] = ab[1, -1] = 1.0
    ab[0, 1] = 0.0
    ab[2, -2] = 0.0
    u = solve_banded((1, 1), ab, b)
    u[0], u[-1] = cfg.u_left, cfg.u_right

    interior = slice(1, -1)
    residual = float(np.max(np.abs(
        (u[:-2] - 2.0 * u[1:-1] + u[2:]) / h**2 + cfg.q * u[interior] - rhs[interior]))) if n > 2 else 0.0
    return {"u": u, "x": x, "h": h, "residual": residual}


@SolverRegistry.register(
    name="elliptic1d_fdm",
    family="pde",
    description="1-D Poisson / Laplace / Helmholtz on a line with Dirichlet ends: three-point differences "
                "and a direct tridiagonal solve",
    tags=["fdm", "poisson", "laplace", "helmholtz", "1d", "elliptic"],
)
class Elliptic1DFDMSolver(SolverBase):
    """``solve_from_spec`` for :func:`solve_elliptic_1d`.

    The spec's parameters (``spec.pde.params``): ``n`` points (default 201), ``k`` for Helmholtz (``q = k^2``) and
    ``source`` -- a constant or a callable ``f(x)`` -- with the sign convention of each kind (Poisson
    ``-u'' = f``, Helmholtz ``u'' + k^2 u = f``). ``domain_bounds["x"]`` gives the length. The two ends take the
    values of the spec's Dirichlet conditions, and 0 where an end has none.
    """

    def solve_from_spec(self, spec) -> SolverOutput:
        kind = str(getattr(spec.pde, "kind", "")).lower().replace("-", "_").replace(" ", "_")
        if not any(k in kind for k in ("poisson", "laplace", "helmholtz")):
            raise ValueError(f"elliptic1d_fdm solves poisson/laplace/helmholtz, and the spec's kind is '{kind}'.")
        if int(getattr(spec, "dim", 1)) != 1:
            raise ValueError(f"elliptic1d_fdm is a 1-D solver; the spec is {getattr(spec, 'dim', '?')}-D.")
        params = dict(getattr(spec.pde, "params", {}) or {})
        bounds = dict(getattr(spec, "domain_bounds", {}) or {})
        length = abs(bounds.get("x", (0.0, 1.0))[1] - bounds.get("x", (0.0, 1.0))[0])
        source = params.get("source", params.get("f", 0.0))
        if "helmholtz" in kind:
            if params.get("k") is None:
                raise ValueError("elliptic1d_fdm needs the wavenumber in spec.pde.params['k'] for a Helmholtz "
                                 "problem (it solves u'' + k^2 u = f).")
            q = float(params["k"]) ** 2
            rhs: Any = source
        else:
            q = 0.0
            rhs = (lambda x: -np.asarray(source(x), dtype=np.float64)) if callable(source) else -float(source)

        conditions = tuple(getattr(spec, "conditions", ()) or ())
        others = [c for c in conditions if getattr(c, "kind", None) not in ("dirichlet", "initial")]
        if others:
            raise ValueError("elliptic1d_fdm holds the two ends at the values of the spec's dirichlet conditions "
                             f"and has no other boundary: the spec asks for {sorted({c.kind for c in others})}.")
        u_left, u_right = 0.0, 0.0
        if conditions:
            from . import _bc
            ends = torch.tensor([[0.0], [length]], dtype=torch.float32)
            for cond in _bc.dirichlet_conditions(spec):
                picked = _bc.select(cond, ends, boundary=torch.arange(2))
                if picked.numel() == 0:
                    continue
                values = _bc.values(cond, ends[picked]).detach().cpu().numpy()
                for index, value in zip(picked.tolist(), values, strict=True):
                    if index == 0:
                        u_left = float(value)
                    else:
                        u_right = float(value)

        cfg = Elliptic1D(n=int(params.get("n", 201)), length=length, q=q, rhs=rhs,
                         u_left=u_left, u_right=u_right)
        out = solve_elliptic_1d(cfg)
        return SolverOutput(
            result=torch.from_numpy(np.asarray(out["u"], dtype=np.float32)),
            losses={"residual": torch.tensor(float(out["residual"]))},
            extras={"method": "elliptic1d_fdm", "kind": kind, "q": q, "n": cfg.n, "length": length,
                    "residual": float(out["residual"]), "x": out["x"],
                    "u_min": float(out["u"].min()), "u_max": float(out["u"].max()),
                    "u_left": u_left, "u_right": u_right},
        )
