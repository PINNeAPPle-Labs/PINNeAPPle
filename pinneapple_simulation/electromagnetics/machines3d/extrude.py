"""Extrusion of a 2-D triangle mesh into tetrahedra: each triangle times each z layer is a prism cut into three
tetrahedra by the global node order, so neighbouring prisms cut their shared quadrilateral faces along the same
diagonal and the mesh is conforming."""
from __future__ import annotations

import numpy as np

from ..fem3d.mesh import TetMesh

__all__ = ["extrude"]


def extrude(points: np.ndarray, tris: np.ndarray, z: np.ndarray, rank: np.ndarray | None = None
            ) -> tuple[TetMesh, np.ndarray, np.ndarray]:
    """Tetrahedral mesh of ``tris`` extruded through the planes ``z`` (increasing). Node ``i`` of plane ``k`` has
    index ``k * N + i``. ``rank`` (a permutation of the nodes, default the node index) is the order that decides
    the cuts: a rank that alternates along a ring of nodes alternates the lean of the cuts, so the mesh has no
    handedness (with ranks growing with the angle every cut leans the same way round the axis, a helical bias
    that shows up as a spurious torque). Returns the mesh (one region, tag 0), and for each tetrahedron the index
    of its triangle and of its z layer."""
    points = np.asarray(points, float)
    tris = np.asarray(tris, np.int64)
    rk = np.arange(len(points)) if rank is None else np.asarray(rank)
    tris = np.take_along_axis(tris, np.argsort(rk[tris], axis=1), axis=1)   # rank(a) < rank(b) < rank(c)
    z = np.asarray(z, float)
    N, T, L = len(points), len(tris), len(z) - 1
    nodes = np.column_stack([np.tile(points, (L + 1, 1)), np.repeat(z, N)])
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    out, tri_of, layer_of = [], [], []
    for k in range(L):
        lo, hi = k * N, (k + 1) * N
        # prism (a, b, c | a', b', c') with rank(a) < rank(b) < rank(c): the tetrahedra (a, b, c, a'), (b, c, a', b'), (c, a', b', c')
        # cut every side face (p q q' p'), rank(p) < rank(q), along q-p'. The cut depends only on the order of the face's own
        # two vertices, so the prism on the other side of the face cuts it the same way
        t = np.concatenate([np.column_stack([a + lo, b + lo, c + lo, a + hi]),
                            np.column_stack([b + lo, c + lo, a + hi, b + hi]),
                            np.column_stack([c + lo, a + hi, b + hi, c + hi])])
        out.append(t)
        tri_of.append(np.tile(np.arange(T), 3))
        layer_of.append(np.full(3 * T, k))
    mesh = TetMesh(nodes, np.concatenate(out), np.zeros(3 * T * L, np.int64), {"domain": 0})
    return mesh, np.concatenate(tri_of), np.concatenate(layer_of)
