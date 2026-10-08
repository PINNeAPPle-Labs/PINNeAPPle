"""Differentiable P1 finite elements on simplex meshes.

Everything here is written in torch, so the solution of a PDE is differentiable with respect to the
**vertex coordinates**, which is the mesh (shape) gradient. Parametrise the geometry as
``points = f(design_parameters)`` and ``loss.backward()`` returns d(objective)/d(design_parameters).
The linear solve is :func:`torch.linalg.solve`, whose backward pass is the adjoint solve.

Matrices are dense, so this is meant for meshes up to a few thousand nodes: reference cases, shape
optimisation prototypes and tests of larger solvers.

>>> import numpy as np, torch
>>> from pinneapple_core import Mesh
>>> from pinneapple_core.fem import solve_poisson
>>> m = Mesh.interval(0.0, 1.0, 20)
>>> pts = torch.tensor(m.points, dtype=torch.float64)
>>> u = solve_poisson(pts, m.cells, 1.0, m.boundary_nodes())
>>> round(float(u.max()), 4)   # -u'' = 1 on (0,1), exact max 1/8
0.125
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Union

import numpy as np
import torch
from torch import Tensor

__all__ = ["cell_geometry", "stiffness", "mass", "solve_poisson", "integrate_p1"]


def _cells(cells) -> Tensor:
    return torch.as_tensor(np.asarray(cells), dtype=torch.long)


def cell_geometry(points: Tensor, cells) -> tuple:
    """Volumes ``(M,)`` and barycentric gradients ``(M, d+1, d)`` of every cell, differentiable in ``points``."""
    c = _cells(cells)
    d = points.shape[1]
    cp = points[c]  # (M, d+1, d)
    t = cp[:, 1:, :] - cp[:, :1, :]
    vol = torch.linalg.det(t).abs() / math.factorial(d)
    g = torch.linalg.inv(t).transpose(1, 2)  # row k is grad of lambda_{k+1}
    g = torch.cat([-g.sum(dim=1, keepdim=True), g], dim=1)
    return vol, g


def _assemble(n: int, cells, local: Tensor) -> Tensor:
    c = _cells(cells)
    k = c.shape[1]
    rows = c[:, :, None].expand(-1, k, k).reshape(-1)
    cols = c[:, None, :].expand(-1, k, k).reshape(-1)
    out = torch.zeros(n, n, dtype=local.dtype, device=local.device)
    return out.index_put((rows, cols), local.reshape(-1), accumulate=True)


def stiffness(points: Tensor, cells, kappa: Union[float, Tensor] = 1.0) -> Tensor:
    """Dense ``(N, N)`` stiffness matrix ``K_ij = integral kappa grad(phi_i) . grad(phi_j)``."""
    vol, g = cell_geometry(points, cells)
    kap = torch.as_tensor(kappa, dtype=points.dtype, device=points.device)
    # kappa is a number or one value per cell
    local = torch.einsum("mid,mjd->mij", g, g) * (vol * kap).reshape(-1, 1, 1)
    return _assemble(points.shape[0], cells, local)


def mass(points: Tensor, cells) -> Tensor:
    """Dense ``(N, N)`` consistent mass matrix ``M_ij = integral phi_i phi_j``."""
    vol, _ = cell_geometry(points, cells)
    d = points.shape[1]
    base = (torch.ones(d + 1, d + 1, dtype=points.dtype, device=points.device)
            + torch.eye(d + 1, dtype=points.dtype, device=points.device)) / ((d + 1) * (d + 2))
    return _assemble(points.shape[0], cells, vol.reshape(-1, 1, 1) * base)


def integrate_p1(points: Tensor, cells, values: Tensor) -> Tensor:
    """Integral of the P1 interpolant of nodal ``values``, differentiable in ``points`` and ``values``."""
    vol, _ = cell_geometry(points, cells)
    return (values[_cells(cells)].mean(dim=1) * vol).sum()


def solve_poisson(
    points: Tensor,
    cells,
    source: Union[float, Tensor, Callable[[Tensor], Tensor]],
    dirichlet_nodes,
    dirichlet_values: Union[float, Tensor] = 0.0,
    kappa: Union[float, Tensor] = 1.0,
) -> Tensor:
    """Solve ``-div(kappa grad u) = source`` with ``u = g`` on ``dirichlet_nodes``; returns nodal ``u``.

    ``source`` is a number, nodal values ``(N,)`` or a callable of the node coordinates (it may depend
    on the geometry). The result is differentiable with respect to ``points``, ``kappa``, ``source``
    and ``dirichlet_values``.
    """
    n = points.shape[0]
    f = source(points) if callable(source) else torch.as_tensor(source, dtype=points.dtype, device=points.device)
    f = f.expand(n) if f.ndim == 0 else f
    k = stiffness(points, cells, kappa)
    b = mass(points, cells) @ f
    dn = torch.as_tensor(np.asarray(dirichlet_nodes), dtype=torch.long)
    free = torch.ones(n, dtype=torch.bool)
    free[dn] = False
    g = torch.as_tensor(dirichlet_values, dtype=points.dtype, device=points.device)
    g = g.expand(dn.numel()) if g.ndim == 0 else g
    u = torch.zeros(n, dtype=points.dtype, device=points.device)
    u = u.index_put((dn,), g)
    rhs = b[free] - k[free][:, dn] @ g
    u_free = torch.linalg.solve(k[free][:, free], rhs)
    return u.index_put((torch.nonzero(free).squeeze(1),), u_free)
