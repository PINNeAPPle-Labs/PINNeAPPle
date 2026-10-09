"""Quantitative counterparts of the qualitative models, for ``QualitativePreview.quantify``.

``voxel_fem_cantilever``: 3-D linear elasticity on the voxelized geometry (any shape: holes, flanges, tapers),
clamped at one end with a load spread over the other. Hexahedral elements with Wilson's incompatible modes
(Wilson, Taylor, Doherty & Ghaboussi 1973), whose three extra bubble modes per direction let a single element bend
without the shear locking of the plain trilinear brick; the bubbles are condensed out. It captures what beam theory
leaves out (shear deformation of short beams, stress concentration at section changes, flanges that do not work), so
it is the check for ``Cantilever`` previews.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .descriptors import as_mesh, section_profile

__all__ = ["hex8_incompatible_stiffness", "voxel_fem_cantilever"]

_CORNERS = np.array([[i, j, k] for k in (0, 1) for j in (0, 1) for i in (0, 1)], float) * 2 - 1   # xi_i of node l


def _elasticity(E: float, nu: float) -> np.ndarray:
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2 * mu
    D[np.arange(3, 6), np.arange(3, 6)] = mu
    return D


def _strain_matrix(dN: np.ndarray) -> np.ndarray:
    """B (6 x 3n) from shape-function derivatives dN (n, 3); strain order xx, yy, zz, xy, yz, zx."""
    n = len(dN)
    B = np.zeros((6, 3 * n))
    for i in range(n):
        x, y, z = dN[i]
        B[:, 3 * i: 3 * i + 3] = [[x, 0, 0], [0, y, 0], [0, 0, z], [y, x, 0], [0, z, y], [z, 0, x]]
    return B


def hex8_incompatible_stiffness(size, E: float, nu: float) -> np.ndarray:
    """24 x 24 stiffness of a rectangular brick ``size = (hx, hy, hz)`` with Wilson's incompatible modes condensed."""
    hx, hy, hz = size
    jinv = np.array([2 / hx, 2 / hy, 2 / hz])
    detJ = hx * hy * hz / 8
    D = _elasticity(E, nu)
    g = 1 / np.sqrt(3)
    K = np.zeros((33, 33))
    for xi in (-g, g):
        for eta in (-g, g):
            for zeta in (-g, g):
                p = np.array([xi, eta, zeta])
                dN = np.empty((8, 3))
                for m, c in enumerate(_CORNERS):
                    f = 1 + c * p
                    dN[m] = [c[0] * f[1] * f[2], f[0] * c[1] * f[2], f[0] * f[1] * c[2]]
                dN = dN / 8 * jinv
                # bubble modes 1 - xi^2, 1 - eta^2, 1 - zeta^2: derivative only in their own direction
                dP = np.diag(-2 * p * jinv)
                B = np.hstack([_strain_matrix(dN), _strain_matrix(dP)])
                K += B.T @ D @ B * detJ
    Kcc, Kci, Kii = K[:24, :24], K[:24, 24:], K[24:, 24:]
    return Kcc - Kci @ np.linalg.solve(Kii, Kci.T)


def voxel_fem_cantilever(geom: Any, E: float = 70e9, nu: float = 0.3, load: float = 1000.0, axis: int | None = None,
                         load_axis: int = 2, clamp: str = "min", n_slices: int = 40, cells_across: int = 8,
                         rho: float = 2700.0) -> dict[str, float]:
    """Tip deflection, stiffness, maximum von Mises stress and mass of a body clamped at one end of ``axis`` (default
    the longest) with ``load`` (N) along ``load_axis`` spread over the other end face. ``cells_across``: voxels along
    the larger section dimension (the other is scaled to keep the voxels near cubic)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    V, F = as_mesh(geom)
    ext = np.ptp(V, axis=0)
    ax = int(np.argmax(ext)) if axis is None else int(axis)
    oth = [k for k in range(3) if k != ax]
    big = max(ext[oth])
    na = max(2, int(round(cells_across * ext[oth[0]] / big)))
    nb = max(2, int(round(cells_across * ext[oth[1]] / big)))
    prof = section_profile((V, F), axis=ax, n_slices=n_slices, n_grid=(na, nb))
    occ = prof["occupancy"]
    hx, ha, hb = prof["cell"]
    Ke = hex8_incompatible_stiffness((hx, ha, hb), E, nu)
    nx = occ.shape[0]
    shape = (nx + 1, na + 1, nb + 1)
    node = np.arange(np.prod(shape)).reshape(shape)
    el = np.argwhere(occ)
    offsets = ((_CORNERS + 1) / 2).astype(int)                                       # corner l -> (i, j, k)
    conn = np.stack([node[el[:, 0] + i, el[:, 1] + j, el[:, 2] + k] for i, j, k in offsets], axis=1)   # (ne, 8)
    dofs = (3 * conn[:, :, None] + np.arange(3)).reshape(len(el), 24)
    rows = np.repeat(dofs, 24, axis=1).ravel()
    cols = np.tile(dofs, (1, 24)).ravel()
    ndof = 3 * node.size
    K = coo_matrix((np.tile(Ke.ravel(), len(el)), (rows, cols)), shape=(ndof, ndof)).tocsr()
    used = np.zeros(node.size, bool)
    used[conn.ravel()] = True
    end_clamp, end_load = (0, nx) if clamp == "min" else (nx, 0)
    fixed_nodes = node[end_clamp][used[node[end_clamp]]]
    comp = 1 + oth.index(load_axis) if load_axis in oth else None
    if comp is None:
        raise ValueError("the load must be transverse to the beam axis")
    f = np.zeros(ndof)
    last = nx - 1 if end_load == nx else 0
    faces = np.argwhere(occ[last])
    for ia, ib in faces:
        for da in (0, 1):
            for db in (0, 1):
                f[3 * node[end_load, ia + da, ib + db] + comp] += load / (4 * len(faces))
    fixed = np.zeros(ndof, bool)
    fixed[(3 * fixed_nodes[:, None] + np.arange(3)).ravel()] = True
    free = np.repeat(used, 3) & ~fixed
    u = np.zeros(ndof)
    u[free] = spsolve(K[free][:, free].tocsc(), f[free])
    tip_nodes = np.unique(np.concatenate([[node[end_load, ia + da, ib + db] for da in (0, 1) for db in (0, 1)]
                                          for ia, ib in faces]))
    delta = float(np.mean(u[3 * tip_nodes + comp]))
    # von Mises at the element centres (the bubble modes have zero gradient there)
    dN0 = _CORNERS / 8 * np.array([2 / hx, 2 / ha, 2 / hb])
    B0 = _strain_matrix(dN0)
    eps = (B0 @ u[dofs].T).T
    sig = eps @ _elasticity(E, nu).T
    s = sig
    vm = np.sqrt(0.5 * ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2)
                 + 3 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))
    vol = len(el) * hx * ha * hb
    return {"tip_deflection": delta, "stiffness": load / delta, "max_von_mises": float(vm.max()),
            "mass": rho * vol, "stiffness_to_mass": load / delta / (rho * vol), "n_elements": int(len(el))}
