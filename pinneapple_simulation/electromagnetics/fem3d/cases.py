"""Closed-form 3-D cases on structured box meshes (no gmsh needed): used by the fast tests.

* :func:`manufactured_magnetostatic`: ``A = (0, 0, sin(pi x) sin(pi y))`` on the unit cube, ``J = curl curl A``.
* :func:`slab_eddy`: a conducting slab ``|z| < d`` in a uniform time-harmonic field ``B0 x`` parallel to it,
  ``B_x(z) = B0 cosh(k z) / cosh(k d)``, ``k = (1 + j) / delta``, ``delta = sqrt(2 / (omega mu0 sigma))``
  (e.g. Lammeraner and Stafl, *Eddy Currents*, Iliffe, 1966, ch. 2); with ``A* = y a(z)``,
  ``a = -B0 sinh(k z) / (k cosh(k d))`` inside and ``a = -B0 (z - d sgn z) - B0 tanh(k d) / k sgn z`` outside.
  Time-averaged loss per unit area of the slab (thickness ``2d``, both faces): ``B0^2 / (mu0^2 sigma delta)
  (sinh(2d/delta) - sin(2d/delta)) / (cosh(2d/delta) + cos(2d/delta))``, checked against the integral of
  ``sigma omega^2 |a|^2 / 2`` across the slab.
"""
from __future__ import annotations

import math

import numpy as np

from .mesh import box_mesh
from .nedelec import MU0, NU0, EdgeSpace, solve_eddy, solve_magnetostatic

__all__ = ["manufactured_magnetostatic", "slab_eddy"]


def manufactured_magnetostatic(n: int, **kw) -> dict:
    """Relative L2 error of ``B`` for the manufactured solution on an ``n``-node-per-side cube mesh."""
    g = np.linspace(0.0, 1.0, n)
    m = box_mesh(g, g, g)
    sp = EdgeSpace(m)

    def A(x):
        return np.column_stack([0 * x[:, 0], 0 * x[:, 0], np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])])

    def J(x):
        return 2 * np.pi ** 2 * A(x)

    def B(x):
        return np.column_stack([np.pi * np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1]),
                                -np.pi * np.cos(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]), 0 * x[:, 0]])

    de = m.boundary_edges
    s = solve_magnetostatic(m, 1.0, J=J, dirichlet=(de, sp.edge_values(A, de)), space=sp, **kw)
    Be = B(m.centroids)
    err = math.sqrt(np.sum(sp.vol * ((s.B - Be) ** 2).sum(1)) / np.sum(sp.vol * (Be ** 2).sum(1)))
    return {"h": 1.0 / (n - 1), "error": err, "unknowns": s.info["unknowns"], "solution": s}


def slab_eddy(nz: int, d: float = 0.01, sigma: float = 3.5e7, f: float = 200.0, B0: float = 0.01,
              air: float = 0.01, nxy: int = 3, **kw) -> dict:
    """Edge-element solution of the slab with the exact ``n x A*`` on every boundary face; ``nz`` cells across
    the slab thickness ``2 d`` (graded uniformly), ``air`` of air above and below. Returns the relative L2 error of
    ``B`` and the loss per unit area against the closed form."""
    omega = 2 * math.pi * f
    delta = math.sqrt(2.0 / (omega * MU0 * sigma))
    k = (1 + 1j) / delta
    nair = max(2, int(round(nz * air / (2 * d))))
    z = np.unique(np.r_[np.linspace(-d - air, -d, nair + 1), np.linspace(-d, d, nz + 1),
                        np.linspace(d, d + air, nair + 1)])
    L = 2 * d
    xy = np.linspace(0, L, nxy + 1)
    m = box_mesh(xy, xy, z, region=lambda c: np.where(np.abs(c[:, 2]) < d, 1, 0), names={"air": 0, "slab": 1})
    sp = EdgeSpace(m)

    def a(zz):
        zz = np.asarray(zz, complex)
        s = np.sign(zz.real)
        inside = -B0 * np.sinh(k * zz) / (k * np.cosh(k * d))
        outside = -B0 * (zz - d * s) - B0 * np.tanh(k * d) / k * s
        return np.where(np.abs(zz.real) < d, inside, outside)

    def Bx(zz):
        zz = np.asarray(zz, complex)
        return np.where(np.abs(zz.real) < d, B0 * np.cosh(k * zz) / np.cosh(k * d), B0)

    de = m.boundary_edges
    re = sp.edge_values(lambda x: np.column_stack([0 * x[:, 0], a(x[:, 2]).real, 0 * x[:, 0]]), de)
    im = sp.edge_values(lambda x: np.column_stack([0 * x[:, 0], a(x[:, 2]).imag, 0 * x[:, 0]]), de)
    sig = np.where(m.region == 1, sigma, 0.0)
    sol = solve_eddy(m, NU0, sig, omega, np.zeros((len(m.tets), 3)), dirichlet=(de, re + 1j * im), space=sp, **kw)
    from .nedelec import QUAD4, QUAD4_W

    x = m.nodes[m.tets]
    e2 = n2 = 0.0
    for q in range(4):
        zq = np.einsum("v,kv->k", QUAD4[q], x[:, :, 2])
        Be = Bx(zq)
        e2 += QUAD4_W[q] * np.sum(sp.vol * (np.abs(sol.B[:, 0] - Be) ** 2 + np.abs(sol.B[:, 1]) ** 2
                                           + np.abs(sol.B[:, 2]) ** 2))
        n2 += QUAD4_W[q] * np.sum(sp.vol * np.abs(Be) ** 2)
    t = 2 * d / delta
    p_exact = B0 ** 2 / (MU0 ** 2 * sigma * delta) * (math.sinh(t) - math.sin(t)) / (math.cosh(t) + math.cos(t))
    zz = np.linspace(-d, d, 20001)
    p_quad = float(np.trapezoid(0.5 * sigma * omega ** 2 * np.abs(a(zz)) ** 2, zz))
    p = sol.loss / L ** 2
    return {"delta": delta, "error_B": math.sqrt(e2 / n2), "loss_per_area": p, "loss_exact": p_exact,
            "loss_error": abs(p - p_exact) / p_exact,
            "loss_quadrature": p_quad, "unknowns": sol.info["unknowns"], "solution": sol}
