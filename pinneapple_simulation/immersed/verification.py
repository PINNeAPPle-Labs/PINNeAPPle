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


# ---------------------------------------------------------------- linear elasticity

@dataclass
class ElasticCase:
    name: str
    geometry: G.Geometry
    E: float
    nu: float
    body_force: Callable | None
    bcs: list
    exact: Callable                      # (n, d) displacement
    grad: Callable                       # (n, d, d) du_i / dx_j
    size: float


def elastic_manufactured(dim: int = 2, shape: str = "ball", E: float = 1.0, nu: float = 0.3) -> ElasticCase:
    """A sum of plane waves u = sum_j A_j sin(k_j . x + phi_j) on a disk / sphere (or torus in 3-D), with the body
    force f = sum_j [(lam + mu) k_j (k_j . A_j) + mu |k_j|^2 A_j] sin(...) that makes it exact, and the displacement
    prescribed on the whole surface (Nitsche)."""
    from .bcs import Displacement
    from .elasticity import lame
    lam, mu = lame(E, nu)
    c = _shift(dim)
    if dim == 2:
        K = np.array([[2.0, 1.0], [-1.0, 3.0]])
        A = np.array([[0.3, -0.2], [0.1, 0.25]])
        ph = np.array([0.3, 1.1])
    else:
        K = np.array([[2.0, 1.0, -1.0], [-1.0, 2.5, 0.5], [0.5, -1.0, 2.0]])
        A = np.array([[0.3, -0.2, 0.1], [0.1, 0.25, -0.15], [-0.2, 0.1, 0.2]])
        ph = np.array([0.3, 1.1, -0.4])
    geom, size = (G.torus(c, 0.5, 0.22), 1.44) if shape == "torus" else (G.ball(c, 0.7), 1.4)

    def u(X):
        return sum(A[j][None, :] * np.sin(X @ K[j] + ph[j])[:, None] for j in range(len(K)))

    def gu(X):
        return sum(A[j][None, :, None] * K[j][None, None, :] * np.cos(X @ K[j] + ph[j])[:, None, None]
                   for j in range(len(K)))

    def f(X):
        return sum(((lam + mu) * K[j] * (K[j] @ A[j]) + mu * (K[j] @ K[j]) * A[j])[None, :]
                   * np.sin(X @ K[j] + ph[j])[:, None] for j in range(len(K)))
    return ElasticCase(f"elastic_mms_{dim}d_{shape}", geom, E, nu, f, [Displacement(u)], u, gu, size)


def lame_sphere(a: float = 0.3, b: float = 0.6, p: float = 1.0, E: float = 1.0, nu: float = 0.3) -> ElasticCase:
    """Thick-walled sphere under internal pressure p (Lamé): u = (A + B / r^3) x with
    A = p a^3 (1 - 2 nu) / (E (b^3 - a^3)), B = p a^3 b^3 (1 + nu) / (2 E (b^3 - a^3)) (Timoshenko & Goodier,
    Theory of Elasticity, 3rd ed., section 141). Pressure on the inner surface; the exact displacement is imposed on
    the outer one (removes the rigid modes and tests Nitsche on a curved surface)."""
    from .bcs import Displacement, Pressure
    c = _shift(3)
    Ac = p * a ** 3 * (1 - 2 * nu) / (E * (b ** 3 - a ** 3))
    Bc = p * a ** 3 * b ** 3 * (1 + nu) / (2 * E * (b ** 3 - a ** 3))

    def u(X):
        Y = X - c
        r = np.linalg.norm(Y, axis=1)
        return (Ac + Bc / r ** 3)[:, None] * Y

    def gu(X):
        Y = X - c
        r = np.linalg.norm(Y, axis=1)
        return ((Ac + Bc / r ** 3)[:, None, None] * np.eye(3)
                - 3 * Bc * Y[:, :, None] * Y[:, None, :] / r[:, None, None] ** 5)
    inner = lambda X: np.linalg.norm(X - c, axis=1) < (a + b) / 2          # noqa: E731
    bcs = [Pressure(p, where=inner), Displacement(u)]
    return ElasticCase("lame_sphere", G.annulus(c, a, b), E, nu, None, bcs, u, gu, 2 * b)


def solve_elastic_case(case: ElasticCase, n: int, order: int = 1, stabilization: str = "ghost",
                       keep_fine: bool = True, **kw) -> dict:
    from .elasticity import solve_elasticity
    t0 = time.perf_counter()
    disc = Discretization(case.geometry, case.size / n, order=order, keep_fine=keep_fine)
    sol = solve_elasticity(disc, case.E, case.nu, case.body_force, case.bcs, stabilization=stabilization, **kw)
    wall = time.perf_counter() - t0
    e = sol.errors(case.exact, case.grad)
    return {"case": case.name, "order": order, "n": n, "h": case.size / n, "dofs": int(sol.info["dofs"]),
            "stabilization": stabilization, "L2_rel": e["L2_rel"], "H1_rel": e["H1_rel"], "L2": e["L2"],
            "H1": e["H1"], "force_imbalance": sol.balance["imbalance"],
            "quadrature_s": disc.timings["quadrature_s"], "assembly_s": sol.info["assembly_s"],
            "solve_s": sol.info["solve_seconds"], "solver": sol.info["solve_solver"], "wall_s": wall}


def elastic_convergence(case: ElasticCase, order: int = 1, ns=(8, 16, 32), **kw) -> list[dict]:
    return [solve_elastic_case(case, n, order, **kw) for n in ns]


# ---------------------------------------------------------------- transient heat

def sphere_cooling(R: float = 0.5, k: float = 2.0, rho_c: float = 4.0, h: float = 8.0, T_i: float = 100.0,
                   T_inf: float = 20.0, terms: int = 60):
    """A sphere at T_i cooled by convection: the series solution theta / theta_i = sum C_n exp(-zeta_n^2 Fo)
    sin(zeta_n r*) / (zeta_n r*), C_n = 4 (sin z - z cos z) / (2 z - sin 2z), 1 - zeta_n cot zeta_n = Bi
    (Incropera et al., Fundamentals of Heat and Mass Transfer, 5.6.2). Returns the geometry, the conditions and
    T(X, t)."""
    from scipy.optimize import brentq
    Bi = h * R / k
    alpha = k / rho_c
    zs = np.array([brentq(lambda z: 1 - z / np.tan(z) - Bi, (n - 1) * np.pi + 1e-9, n * np.pi - 1e-9)
                   for n in range(1, terms + 1)])
    Cn = 4 * (np.sin(zs) - zs * np.cos(zs)) / (2 * zs - np.sin(2 * zs))
    c = _shift(3)

    def T(X, t):
        rs = np.maximum(np.linalg.norm(X - c, axis=1) / R, 1e-12)
        Fo = alpha * t / R ** 2
        th = (Cn[None] * np.exp(-zs[None] ** 2 * Fo) * np.sin(zs[None] * rs[:, None])
              / (zs[None] * rs[:, None])).sum(1)
        return T_inf + (T_i - T_inf) * th
    return {"geometry": G.ball(c, R), "k": k, "rho_c": rho_c, "bcs": [Convection(h, T_inf)], "T_i": T_i,
            "exact": T, "Bi": Bi, "alpha": alpha, "R": R}


def manufactured_transient(dim: int = 2):
    """u = exp(-t) sin 2x cos 3y (sin z) + x^2 with rho c = k = 1 and Dirichlet data: f = du/dt - lap u."""
    c = _shift(dim)
    geom = G.ball(c, 0.7)
    if dim == 2:
        S = lambda X: np.sin(2 * X[:, 0]) * np.cos(3 * X[:, 1])                 # noqa: E731
        lam = 13.0
    else:
        S = lambda X: np.sin(2 * X[:, 0]) * np.cos(3 * X[:, 1]) * np.sin(X[:, 2])  # noqa: E731
        lam = 14.0
    u = lambda X, t: np.exp(-t) * S(X) + X[:, 0] ** 2                          # noqa: E731
    f = lambda X, t: (lam - 1) * np.exp(-t) * S(X) - 2                          # noqa: E731
    return {"geometry": geom, "size": 1.4, "exact": u, "source": f,
            "bcs": lambda t: [Dirichlet(lambda X: u(X, t))]}
