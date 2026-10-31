"""Verification of the immersed heat solver: manufactured solutions on curved domains (measured L2 and H1 rates),
closed-form conduction in a sphere and a cylinder with convection, and the conditioning of small cuts.

    from pinneapple_simulation.immersed import verification as V
    rows = V.convergence(V.manufactured(2, "ball"), order=2, hs=[0.1, 0.05, 0.025])
    V.rates(rows)                                    # measured orders between successive grids
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import geometry as G
from .assembly import Discretization
from .bcs import Convection, Dirichlet, HeatFlux
from .heat import solve_heat


@dataclass
class Case:
    """A geometry, its boundary conditions and the exact temperature with its gradient."""

    name: str
    geometry: G.Geometry
    conductivity: float
    source: Callable | float
    bcs: list
    exact: Callable
    grad: Callable
    size: float                          # characteristic length (h is quoted as size / n)


def _shift(d: int) -> np.ndarray:
    # a centre off the grid symmetry planes, so the cuts are generic
    return np.array([0.0123, -0.0071, 0.0047][:d])


def manufactured(dim: int = 2, shape: str = "ball", bc: str = "dirichlet", k: float = 1.0) -> Case:
    """u = sin 2x cos 3y (sin z) + x^2 on a disk / sphere, a ring / hollow sphere, or a torus (3-D), with the
    source -k lap u and either Dirichlet data (Nitsche) or a Robin condition (h = 10, T_inf chosen so u is exact)."""
    c = _shift(dim)
    if shape == "ball":
        geom, size = G.ball(c, 0.7), 1.4
    elif shape == "annulus":
        geom, size = G.annulus(c, 0.3, 0.75), 1.5
    elif shape == "torus" and dim == 3:
        geom, size = G.torus(c, 0.5, 0.22), 1.44
    else:
        raise ValueError(f"unknown shape {shape!r} for dim {dim}")
    if dim == 2:
        u = lambda X: np.sin(2 * X[:, 0]) * np.cos(3 * X[:, 1]) + X[:, 0] ** 2               # noqa: E731
        gu = lambda X: np.stack([2 * np.cos(2 * X[:, 0]) * np.cos(3 * X[:, 1]) + 2 * X[:, 0],  # noqa: E731
                                 -3 * np.sin(2 * X[:, 0]) * np.sin(3 * X[:, 1])], 1)
        f = lambda X: k * (13 * np.sin(2 * X[:, 0]) * np.cos(3 * X[:, 1]) - 2)               # noqa: E731
    else:
        u = lambda X: np.sin(2 * X[:, 0]) * np.cos(3 * X[:, 1]) * np.sin(X[:, 2]) + X[:, 0] ** 2   # noqa: E731
        gu = lambda X: np.stack([2 * np.cos(2 * X[:, 0]) * np.cos(3 * X[:, 1]) * np.sin(X[:, 2]) + 2 * X[:, 0],  # noqa: E731
                                 -3 * np.sin(2 * X[:, 0]) * np.sin(3 * X[:, 1]) * np.sin(X[:, 2]),
                                 np.sin(2 * X[:, 0]) * np.cos(3 * X[:, 1]) * np.cos(X[:, 2])], 1)
        f = lambda X: k * (14 * np.sin(2 * X[:, 0]) * np.cos(3 * X[:, 1]) * np.sin(X[:, 2]) - 2)   # noqa: E731
    if bc == "dirichlet":
        bcs = [Dirichlet(u)]
    elif bc == "robin":
        hc = 10.0
        bcs = [Convection(hc, lambda X, n: u(X) + k * np.einsum("ij,ij->i", gu(X), n) / hc)]
    else:
        raise ValueError(bc)
    return Case(f"mms_{dim}d_{shape}_{bc}", geom, k, f, bcs, u, gu, size)


def sphere_convection(R: float = 0.5, k: float = 2.0, q: float = 100.0, h: float = 50.0, T_inf: float = 20.0
                      ) -> Case:
    """Uniform generation q in a sphere cooled by convection: T = T_inf + q R / (3 h) + q (R^2 - r^2) / (6 k)
    (Incropera et al., Fundamentals of Heat and Mass Transfer, 3.5)."""
    c = _shift(3)
    T = lambda X: T_inf + q * R / (3 * h) + q * (R ** 2 - ((X - c) ** 2).sum(1)) / (6 * k)   # noqa: E731
    gT = lambda X: -q * (X - c) / (3 * k)                                                    # noqa: E731
    return Case("sphere_convection", G.ball(c, R), k, q, [Convection(h, T_inf)], T, gT, 2 * R)


def cylinder_convection(R: float = 0.4, L: float = 1.0, k: float = 2.0, q: float = 100.0, h: float = 50.0,
                        T_inf: float = 20.0) -> Case:
    """Uniform generation in a finite cylinder, convection on the mantle, adiabatic end caps: the radial solution
    T = T_inf + q R / (2 h) + q (R^2 - r^2) / (4 k) is exact (Incropera et al., 3.5). The caps are told apart from
    the mantle by the surface normal."""
    c = _shift(3)
    T = lambda X: T_inf + q * R / (2 * h) + q * (R ** 2 - ((X[:, :2] - c[:2]) ** 2).sum(1)) / (4 * k)  # noqa: E731
    gT = lambda X: np.column_stack([-q * (X[:, :2] - c[:2]) / (2 * k), np.zeros(len(X))])            # noqa: E731
    mantle = lambda X, n: np.abs(n[:, 2]) < 0.5                                                      # noqa: E731
    bcs = [Convection(h, T_inf, where=mantle), HeatFlux(0.0)]
    return Case("cylinder_convection", G.cylinder(c, R, L), k, q, bcs, T, gT, L)


def solve_case(case: Case, n: int, order: int = 1, stabilization: str = "ghost", depth: int | None = None,
               condition: bool = False, keep_fine: bool = True, **kw) -> dict:
    """Solve a case on a grid of ``n`` cells across its characteristic size and measure everything."""
    t0 = time.perf_counter()
    disc = Discretization(case.geometry, case.size / n, order=order, depth=depth, keep_fine=keep_fine)
    sol = solve_heat(disc, case.conductivity, case.source, case.bcs, stabilization=stabilization,
                     condition=condition, **kw)
    wall = time.perf_counter() - t0
    e = sol.errors(case.exact, case.grad)
    s = disc.summary()
    row = {"case": case.name, "order": order, "n": n, "h": case.size / n, "dofs": disc.n_dofs,
           "cells_cut": s["cells_cut"], "min_cut_fraction": s["min_cut_fraction"], "stabilization": stabilization,
           "L2_rel": e["L2_rel"], "H1_rel": e.get("H1_rel"), "L2": e["L2"], "H1": e.get("H1"),
           "energy_imbalance": sol.balance["imbalance"], "quadrature_s": disc.timings["quadrature_s"],
           "assembly_s": sol.info["assembly_s"], "solve_s": sol.info["solve_seconds"],
           "solver": sol.info["solve_solver"], "wall_s": wall}
    if condition:
        row.update(cond=sol.info["cond"], cond_jacobi=sol.info["cond_jacobi"])
    return row


def convergence(case: Case, order: int = 1, ns=(8, 16, 32), **kw) -> list[dict]:
    return [solve_case(case, n, order, **kw) for n in ns]


def rates(rows: list[dict], keys=("L2", "H1")) -> dict[str, list[float]]:
    """Observed orders log(e_i / e_{i+1}) / log(h_i / h_{i+1}) between successive rows."""
    out = {}
    for k in keys:
        out[k] = [float(np.log(a[k] / b[k]) / np.log(a["h"] / b["h"])) for a, b in zip(rows[:-1], rows[1:], strict=True)
                  if a.get(k) and b.get(k)]
    return out


def conditioning(order: int = 1, offsets=(1e-1, 1e-2, 1e-3, 1e-4), stabilizations=("none", "fcm", "ghost"),
                 n: int = 10) -> list[dict]:
    """Small cuts on purpose (2-D manufactured case on a disk): the grid is placed so that the disk pokes
    ``offset`` x h past a grid line into the next column of cells, which therefore hold slivers of the part. For
    each stabilisation: the condition number (raw and Jacobi scaled), the smallest cut fraction and the error."""
    case = manufactured(2, "ball")
    h = case.size / n
    rows = []
    for off in offsets:
        for st in stabilizations:
            disc = Discretization(case.geometry, h, order=order, pad=1.0 - off, jitter=False)
            row = {"offset": off, "stabilization": st, "order": order,
                   "min_cut_fraction": disc.summary()["min_cut_fraction"]}
            try:
                sol = solve_heat(disc, case.conductivity, case.source, case.bcs, stabilization=st, condition=True)
                e = sol.errors(case.exact, case.grad, fine=False)
                row.update(cond=sol.info["cond"], cond_jacobi=sol.info["cond_jacobi"], L2_rel=e["L2_rel"])
            except Exception as exc:                                    # a singular matrix is a result too
                row["error"] = repr(exc)
            rows.append(row)
    return rows
