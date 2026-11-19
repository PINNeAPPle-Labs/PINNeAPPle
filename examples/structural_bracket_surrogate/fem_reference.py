"""Small-strain isotropic 3D tetrahedral FEM reference for bracket benchmarks.

Mesh input: nodes [N,3] in metres, tetrahedra [T,4], fixed node indices,
and nodal forces [N,3] in newtons. Stress output in pascals.
This is a linear-elastic baseline; point loads and sharp corners may create
mesh-dependent stress singularities. Use distributed traction for production.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def constitutive_matrix(young_pa: float, poisson: float) -> np.ndarray:
    if young_pa <= 0 or not (-1.0 < poisson < 0.5):
        raise ValueError("Require E > 0 and -1 < nu < 0.5")
    lam = young_pa * poisson / ((1 + poisson) * (1 - 2 * poisson))
    mu = young_pa / (2 * (1 + poisson))
    d = np.zeros((6, 6))
    d[:3, :3] = lam
    np.fill_diagonal(d[:3, :3], lam + 2 * mu)
    d[3:, 3:] = mu
    return d


def tetra_b_matrix(x: np.ndarray) -> tuple[np.ndarray, float]:
    if x.shape != (4, 3):
        raise ValueError("tetra coordinates must be [4,3]")
    a = np.column_stack((np.ones(4), x))
    determinant = np.linalg.det(a)
    volume = abs(determinant) / 6
    if volume <= 1e-18:
        raise ValueError("Degenerate tetrahedron")
    gradients = np.linalg.inv(a)[1:, :].T
    b = np.zeros((6, 12))
    for i, (gx, gy, gz) in enumerate(gradients):
        j = 3 * i
        b[0, j] = gx
        b[1, j + 1] = gy
        b[2, j + 2] = gz
        b[3, j] = gy; b[3, j + 1] = gx
        b[4, j + 1] = gz; b[4, j + 2] = gy
        b[5, j] = gz; b[5, j + 2] = gx
    return b, volume


def solve_elasticity(nodes: np.ndarray, tetrahedra: np.ndarray, fixed_nodes: np.ndarray,
                     nodal_forces: np.ndarray, young_pa: float, poisson: float) -> dict:
    nodes = np.asarray(nodes, dtype=float)
    tetrahedra = np.asarray(tetrahedra, dtype=int)
    fixed_nodes = np.asarray(fixed_nodes, dtype=int)
    forces = np.asarray(nodal_forces, dtype=float)
    n = len(nodes)
    if nodes.shape != (n, 3) or tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
        raise ValueError("Expected nodes [N,3], tetrahedra [T,4]")
    if forces.shape != (n, 3) or len(fixed_nodes) == 0:
        raise ValueError("Require nodal forces [N,3] and fixed nodes")
    if np.any(tetrahedra < 0) or np.any(tetrahedra >= n):
        raise ValueError("Tetrahedral connectivity out of bounds")
    if np.any(fixed_nodes < 0) or np.any(fixed_nodes >= n):
        raise ValueError("Fixed node index out of bounds")
    d = constitutive_matrix(young_pa, poisson)
    rows, cols, values, operators = [], [], [], []
    for tet in tetrahedra:
        b, volume = tetra_b_matrix(nodes[tet])
        ke = volume * b.T @ d @ b
        dofs = (3 * tet[:, None] + np.arange(3)).reshape(-1)
        rows.extend(np.repeat(dofs, 12))
        cols.extend(np.tile(dofs, 12))
        values.extend(ke.ravel())
        operators.append((b, dofs))
    k = coo_matrix((values, (rows, cols)), shape=(3*n, 3*n)).tocsr()
    constrained = (3 * np.unique(fixed_nodes)[:, None] + np.arange(3)).reshape(-1)
    free = np.setdiff1d(np.arange(3*n), constrained)
    if len(free) == 0:
        raise ValueError("No free degrees of freedom")
    u = np.zeros(3*n)
    u[free] = spsolve(k[free][:, free], forces.reshape(-1)[free])
    if not np.all(np.isfinite(u)):
        raise ValueError("Nonfinite displacement; check rigid-body constraints")
    stresses = np.stack([d @ b @ u[dofs] for b, dofs in operators])
    sx, sy, sz, txy, tyz, txz = stresses.T
    vm = np.sqrt(np.maximum(0, 0.5*((sx-sy)**2+(sy-sz)**2+(sz-sx)**2)
                            + 3*(txy*txy+tyz*tyz+txz*txz)))
    reactions = (k @ u - forces.reshape(-1)).reshape(n, 3)
    return {"displacement_m": u.reshape(n, 3), "stress_pa": stresses,
            "von_mises_pa": vm, "reaction_n": reactions,
            "force_balance_n": np.linalg.norm(reactions.sum(axis=0) + forces.sum(axis=0))}
