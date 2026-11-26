"""Discrete differential geometry on triangle surface meshes (NumPy/SciPy).

Real curvature for the geometry pipeline (``features.compute_curvature_proxy``
is only a normal-variation proxy): useful as a CAD/mesh quality feature, as a
geometry descriptor for surrogate models (curvature drives boundary-layer
behaviour and stress concentration) and for OOD checks on new shapes.

Operators -- Meyer, Desbrun, Schroeder & Barr (2003), "Discrete
differential-geometry operators for triangulated 2-manifolds":

- :func:`mixed_areas`          Voronoi areas, with the mixed-area rule for
                               obtuse triangles (Sec. 3.3, Fig. 4)
- :func:`cotan_laplacian`      cotangent stiffness matrix ``C`` (Eq. 8);
                               Laplace-Beltrami ``Delta f = M^{-1} C f``
- :func:`gaussian_curvature`   angle defect / area (Eq. 9); boundary vertices
                               use ``pi`` as the full angle (geodesic-curvature
                               term) or are reported as NaN
- :func:`mean_curvature`       ``Delta x = -2 H n`` (Eq. 7); sign from the
                               area-weighted vertex normal
- :func:`principal_curvatures` ``k1,2 = H +- sqrt(max(H^2 - K, 0))``

Harvest note: the operator choice (mixed Voronoi areas, angle defect, the
discrete Gauss-Bonnet identity as the acceptance test) follows NVIDIA
PhysicsNeMo's ``physicsnemo.mesh.curvature`` (Apache-2.0), studied for this
module; this is an independent NumPy implementation, not a copy of that
PyTorch code. See ``docs/dev/rea_harvest/ledger.json``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

__all__ = [
    "mixed_areas", "cotan_laplacian", "laplace_beltrami", "boundary_vertices",
    "gaussian_curvature", "mean_curvature", "principal_curvatures", "euler_characteristic",
]


def _vf(mesh_or_vertices: Any, faces: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if faces is None:  # MeshData-like
        V, F = mesh_or_vertices.vertices, mesh_or_vertices.faces
    else:
        V, F = mesh_or_vertices, faces
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=np.int64)
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("faces must be (n_faces, 3) triangles")
    return V, F


def _corner_angles(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Interior angle at each corner, shape (n_faces, 3)."""
    out = np.empty(F.shape)
    for c in range(3):
        a = V[F[:, (c + 1) % 3]] - V[F[:, c]]
        b = V[F[:, (c + 2) % 3]] - V[F[:, c]]
        cos = np.einsum("ij,ij->i", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        out[:, c] = np.arccos(np.clip(cos, -1.0, 1.0))
    return out


def _cot(theta: np.ndarray) -> np.ndarray:
    return np.cos(theta) / np.sin(theta)


def boundary_vertices(mesh_or_vertices: Any, faces: np.ndarray | None = None) -> np.ndarray:
    """Boolean mask of vertices on a boundary edge (edge used by one face)."""
    V, F = _vf(mesh_or_vertices, faces)
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    mask = np.zeros(len(V), bool)
    mask[uniq[counts == 1].ravel()] = True
    return mask


def euler_characteristic(mesh_or_vertices: Any, faces: np.ndarray | None = None) -> int:
    V, F = _vf(mesh_or_vertices, faces)
    e = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    used = np.unique(F).size
    return int(used - len(e) + len(F))


def mixed_areas(mesh_or_vertices: Any, faces: np.ndarray | None = None) -> np.ndarray:
    """Per-vertex mixed Voronoi area; the areas sum to the surface area."""
    V, F = _vf(mesh_or_vertices, faces)
    ang = _corner_angles(V, F)
    A = np.zeros(len(V))
    tri_area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    obtuse = ang > np.pi / 2
    any_obtuse = obtuse.any(axis=1)
    for c in range(3):
        i, j, k = F[:, c], F[:, (c + 1) % 3], F[:, (c + 2) % 3]
        # Voronoi part (non-obtuse triangles): 1/8 (|e_ij|^2 cot(angle at k) + |e_ik|^2 cot(angle at j))
        eij2 = np.sum((V[j] - V[i]) ** 2, axis=1)
        eik2 = np.sum((V[k] - V[i]) ** 2, axis=1)
        vor = (eij2 * _cot(ang[:, (c + 2) % 3]) + eik2 * _cot(ang[:, (c + 1) % 3])) / 8.0
        val = np.where(~any_obtuse, vor,
                       np.where(obtuse[:, c], tri_area / 2.0, tri_area / 4.0))
        np.add.at(A, i, val)
    return A


def cotan_laplacian(mesh_or_vertices: Any, faces: np.ndarray | None = None) -> sparse.csr_matrix:
    """Symmetric cotangent matrix ``C`` with ``C_ij = (cot a_ij + cot b_ij)/2``
    off the diagonal and ``C_ii = -sum_j C_ij``; ``(C f)_i`` is the Laplacian
    of ``f`` integrated over vertex i's mixed area (Eq. 8), so the pointwise
    Laplace-Beltrami is ``(C f)_i / A_i``."""
    V, F = _vf(mesh_or_vertices, faces)
    ang = _corner_angles(V, F)
    rows, cols, vals = [], [], []
    for c in range(3):
        i, j = F[:, (c + 1) % 3], F[:, (c + 2) % 3]    # edge opposite corner c
        w = 0.5 * _cot(ang[:, c])
        rows += [i, j]
        cols += [j, i]
        vals += [w, w]
    n = len(V)
    W = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return (W - sparse.diags(np.asarray(W.sum(axis=1)).ravel())).tocsr()


def laplace_beltrami(f: np.ndarray, mesh_or_vertices: Any, faces: np.ndarray | None = None) -> np.ndarray:
    """Pointwise Laplace-Beltrami of a vertex field ``f`` (n,) or (n, d)."""
    V, F = _vf(mesh_or_vertices, faces)
    A = mixed_areas(V, F)
    Cf = cotan_laplacian(V, F) @ np.asarray(f, dtype=float)
    return Cf / (A if Cf.ndim == 1 else A[:, None])


def gaussian_curvature(mesh_or_vertices: Any, faces: np.ndarray | None = None, *,
                       boundary: str = "geodesic") -> np.ndarray:
    """Angle-defect Gaussian curvature per vertex.

    ``boundary="geodesic"`` uses ``pi - sum(angles)`` at boundary vertices,
    so ``sum(K * area)`` obeys discrete Gauss-Bonnet ``2 pi chi`` for surfaces
    with boundary too; ``boundary="nan"`` marks them NaN instead.
    """
    V, F = _vf(mesh_or_vertices, faces)
    ang = _corner_angles(V, F)
    s = np.zeros(len(V))
    for c in range(3):
        np.add.at(s, F[:, c], ang[:, c])
    full = np.full(len(V), 2 * np.pi)
    bnd = boundary_vertices(V, F)
    full[bnd] = np.pi
    A = mixed_areas(V, F)
    K = np.divide(full - s, A, out=np.full(len(V), np.nan), where=A > 0)
    if boundary == "nan":
        K[bnd] = np.nan
    elif boundary != "geodesic":
        raise ValueError("boundary must be 'geodesic' or 'nan'")
    return K


def _vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])   # area-weighted
    n = np.zeros_like(V)
    for c in range(3):
        np.add.at(n, F[:, c], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)


def mean_curvature(mesh_or_vertices: Any, faces: np.ndarray | None = None, *,
                   signed: bool = True, boundary: str = "nan") -> np.ndarray:
    """Mean curvature ``H`` per vertex from ``Delta x = -2 H n``.

    Signed with respect to the outward vertex normal implied by the face
    winding (a counter-clockwise-outward sphere has ``H = +1/R``). Boundary
    vertices are NaN by default (the operator is one-sided there)."""
    V, F = _vf(mesh_or_vertices, faces)
    Hn = -0.5 * laplace_beltrami(V, V, F)
    H = np.linalg.norm(Hn, axis=1)
    if signed:
        H = H * np.sign(np.einsum("ij,ij->i", Hn, _vertex_normals(V, F)) + 0.0)
    if boundary == "nan":
        H[boundary_vertices(V, F)] = np.nan
    return H


def principal_curvatures(mesh_or_vertices: Any, faces: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    V, F = _vf(mesh_or_vertices, faces)
    H = mean_curvature(V, F)
    K = gaussian_curvature(V, F, boundary="nan")
    d = np.sqrt(np.maximum(H * H - K, 0.0))
    return H + d, H - d
