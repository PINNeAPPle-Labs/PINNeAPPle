"""Tetrahedral meshes for the 3-D edge-element solvers: topology (edges, faces, boundary), structured box meshes
and gmsh input.

A :class:`TetMesh` holds node coordinates, tetrahedra (four node indices each), a region tag per tetrahedron and
the names of the tags. Edges are oriented from the lower to the higher global node index; a tetrahedron's local
edge ``(a, b)`` carries the sign ``+1`` when ``tets[:, a] < tets[:, b]``, so the tangential continuity of the
Whitney (first-order Nedelec) functions holds across elements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

__all__ = ["TetMesh", "LOCAL_EDGES", "LOCAL_FACES", "box_mesh", "read_msh22", "from_gmsh"]

LOCAL_EDGES = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
LOCAL_FACES = np.array([(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)])


@dataclass
class TetMesh:
    nodes: np.ndarray                      # (N, 3) coordinates in m
    tets: np.ndarray                       # (M, 4) node indices
    region: np.ndarray                     # (M,) integer tag per tetrahedron
    names: dict[str, int] = field(default_factory=dict)   # region name -> tag

    def __post_init__(self):
        self.nodes = np.ascontiguousarray(self.nodes, float)
        self.tets = np.ascontiguousarray(self.tets, np.int64)
        self.region = np.asarray(self.region, np.int64)
        J = self.jacobians
        det = np.linalg.det(J)
        flip = det < 0                     # positive orientation: swap two nodes
        if np.any(flip):
            self.tets[flip] = self.tets[flip][:, [0, 2, 1, 3]]
            for k in ("jacobians",):
                self.__dict__.pop(k, None)

    # ---------------------------------------------------------------------------------------- geometry
    @cached_property
    def jacobians(self) -> np.ndarray:
        x = self.nodes[self.tets]
        return np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]], axis=1)   # rows

    @cached_property
    def volumes(self) -> np.ndarray:
        return np.abs(np.linalg.det(self.jacobians)) / 6.0

    @cached_property
    def grad_lambda(self) -> np.ndarray:
        """(M, 4, 3) gradients of the barycentric coordinates."""
        G = np.linalg.inv(self.jacobians)            # columns: gradients of lambda_1..3
        g = np.transpose(G, (0, 2, 1))
        return np.concatenate([-g.sum(axis=1, keepdims=True), g], axis=1)

    @cached_property
    def centroids(self) -> np.ndarray:
        return self.nodes[self.tets].mean(axis=1)

    @cached_property
    def h(self) -> np.ndarray:
        """Longest edge per tetrahedron."""
        x = self.nodes[self.tets]
        d = x[:, LOCAL_EDGES[:, 1]] - x[:, LOCAL_EDGES[:, 0]]
        return np.linalg.norm(d, axis=2).max(axis=1)

    # ---------------------------------------------------------------------------------------- topology
    @cached_property
    def _edge_topology(self):
        a = self.tets[:, LOCAL_EDGES[:, 0]]
        b = self.tets[:, LOCAL_EDGES[:, 1]]
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        key = np.stack([lo.ravel(), hi.ravel()], axis=1)
        edges, inv = np.unique(key, axis=0, return_inverse=True)
        return edges, inv.reshape(-1, 6), np.where(a < b, 1.0, -1.0)

    @property
    def edges(self) -> np.ndarray:
        return self._edge_topology[0]

    @property
    def tet_edges(self) -> np.ndarray:
        return self._edge_topology[1]

    @property
    def tet_edge_sign(self) -> np.ndarray:
        return self._edge_topology[2]

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @cached_property
    def _face_topology(self):
        f = np.sort(self.tets[:, LOCAL_FACES], axis=2).reshape(-1, 3)
        faces, inv, cnt = np.unique(f, axis=0, return_inverse=True, return_counts=True)
        return faces, inv.reshape(-1, 4), cnt

    @property
    def faces(self) -> np.ndarray:
        return self._face_topology[0]

    @property
    def tet_faces(self) -> np.ndarray:
        return self._face_topology[1]

    @cached_property
    def boundary_faces(self) -> np.ndarray:
        """Indices (into :attr:`faces`) of the faces of one tetrahedron only."""
        return np.flatnonzero(self._face_topology[2] == 1)

    @cached_property
    def interface_faces(self) -> dict[tuple[int, int], np.ndarray]:
        """Faces between two different region tags, keyed by the sorted tag pair."""
        tf = self.tet_faces.ravel()
        tag = np.repeat(self.region, 4)
        order = np.argsort(tf, kind="stable")
        tf, tag = tf[order], tag[order]
        same = np.flatnonzero(tf[1:] == tf[:-1])
        out: dict[tuple[int, int], list[int]] = {}
        for i in same:
            a, b = tag[i], tag[i + 1]
            if a != b:
                out.setdefault((min(a, b), max(a, b)), []).append(tf[i])
        return {k: np.array(v) for k, v in out.items()}

    @cached_property
    def boundary_nodes(self) -> np.ndarray:
        return np.unique(self.faces[self.boundary_faces])

    @cached_property
    def boundary_edges(self) -> np.ndarray:
        """Edges lying on the outer boundary (all three nodes of a boundary face span its three edges)."""
        f = self.faces[self.boundary_faces]
        pairs = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]])
        return self.edge_index(pairs)

    def edge_index(self, pairs: np.ndarray) -> np.ndarray:
        """Global edge indices of node pairs (any order); raises if a pair is not an edge."""
        pairs = np.sort(np.asarray(pairs, np.int64), axis=1)
        E = self.edges
        key = E[:, 0] * (len(self.nodes) + 1) + E[:, 1]
        q = pairs[:, 0] * (len(self.nodes) + 1) + pairs[:, 1]
        i = np.searchsorted(key, q)
        if np.any(i >= len(key)) or np.any(key[np.minimum(i, len(key) - 1)] != q):
            raise KeyError("node pair is not an edge of the mesh")
        return i

    # ---------------------------------------------------------------------------------------- regions
    def mask(self, *names: str) -> np.ndarray:
        tags = [self.names[n] for n in names]
        return np.isin(self.region, tags)

    def edges_in(self, mask: np.ndarray) -> np.ndarray:
        """Edges of the tetrahedra selected by ``mask`` (unique, sorted)."""
        return np.unique(self.tet_edges[mask])

    def nodes_in(self, mask: np.ndarray) -> np.ndarray:
        return np.unique(self.tets[mask])

    @cached_property
    def _centroid_tree(self):
        from scipy.spatial import cKDTree

        return cKDTree(self.centroids)

    @cached_property
    def _jacobians_inv(self) -> np.ndarray:
        return np.linalg.inv(self.jacobians)

    def locate(self, points: np.ndarray, tol: float = 1e-10) -> np.ndarray:
        """Index of a tetrahedron containing each point (-1 outside), by barycentric coordinates; a k-d tree on
        the centroids narrows the candidates."""
        points = np.atleast_2d(np.asarray(points, float))
        out = np.full(len(points), -1, np.int64)
        k = min(64, len(self.tets))
        _, cand = self._centroid_tree.query(points, k=k)
        cand = np.atleast_2d(cand)
        Jinv = self._jacobians_inv
        x0 = self.nodes[self.tets[:, 0]]
        for p in range(len(points)):
            c = cand[p]
            xi = np.einsum("kij,kj->ki", np.transpose(Jinv[c], (0, 2, 1)), points[p] - x0[c])
            lam = np.column_stack([1 - xi.sum(1), xi])
            ok = np.flatnonzero(lam.min(axis=1) >= -tol)
            if len(ok):
                out[p] = c[ok[0]]
            else:                                         # far from the nearest centroids: full scan
                xi = np.einsum("kij,kj->ki", np.transpose(Jinv, (0, 2, 1)), points[p] - x0)
                lam = np.column_stack([1 - xi.sum(1), xi])
                ok = np.flatnonzero(lam.min(axis=1) >= -tol)
                if len(ok):
                    out[p] = ok[0]
        return out

    def summary(self) -> dict:
        return {"nodes": int(len(self.nodes)), "tets": int(len(self.tets)), "edges": int(self.n_edges),
                "regions": {n: int((self.region == t).sum()) for n, t in self.names.items()}}


# -------------------------------------------------------------------------------------------- generators
def box_mesh(x, y, z, region=None, names: dict[str, int] | None = None) -> TetMesh:
    """Structured tetrahedral mesh of a box from tensor grids ``x``, ``y``, ``z``: each hexahedron split into six
    tetrahedra along its main diagonal (Kuhn), so neighbouring cells share their face diagonals (conforming).
    ``region(centroids) -> tags`` sets the region of each tetrahedron (default 0)."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    nx, ny, nz = len(x), len(y), len(z)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    def nid(i, j, k):
        return (i * ny + j) * nz + k

    i, j, k = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij")
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    v = [nid(i + (c >> 0 & 1), j + (c >> 1 & 1), k + (c >> 2 & 1)) for c in range(8)]   # corner c = (dx,dy,dz) bits
    # six tetrahedra sharing the diagonal v0-v7, one per permutation of the axes
    paths = [(1, 3), (1, 5), (2, 3), (2, 6), (4, 5), (4, 6)]
    tets = np.concatenate([np.column_stack([v[0], v[a], v[b], v[7]]) for a, b in paths])
    m = TetMesh(nodes, tets, np.zeros(len(tets), np.int64), dict(names or {"domain": 0}))
    if region is not None:
        m.region = np.asarray(region(m.centroids), np.int64)
    return m


def read_msh22(path) -> TetMesh:
    """ASCII gmsh 2.2 file: nodes, 4-node tetrahedra (type 4) and their physical tags; physical names if present."""
    text = Path(path).read_text()
    names: dict[str, int] = {}
    if "$PhysicalNames" in text:
        block = text.split("$PhysicalNames\n")[1].split("$EndPhysicalNames")[0].splitlines()[1:]
        for ln in block:
            dim, tag, name = ln.split(maxsplit=2)
            if int(dim) == 3:
                names[name.strip('"')] = int(tag)
    nb = text.split("$Nodes\n")[1].split("$EndNodes")[0].splitlines()[1:]
    raw = np.array([ln.split() for ln in nb], float)
    ids = raw[:, 0].astype(np.int64)
    lookup = np.full(ids.max() + 1, -1, np.int64)
    lookup[ids] = np.arange(len(ids))
    eb = text.split("$Elements\n")[1].split("$EndElements")[0].splitlines()[1:]
    tets, tags = [], []
    for ln in eb:
        p = ln.split()
        if p[1] == "4":
            nt = int(p[2])
            tags.append(int(p[3]))
            tets.append([int(v) for v in p[3 + nt:3 + nt + 4]])
    tets = lookup[np.array(tets, np.int64)]
    used = np.unique(tets)
    remap = np.full(len(ids), -1, np.int64)
    remap[used] = np.arange(len(used))
    if not names:
        names = {f"region{t}": t for t in sorted(set(tags))}
    return TetMesh(raw[used, 1:4], remap[tets], np.array(tags), names)


def from_gmsh(model=None) -> TetMesh:
    """The 3-D mesh of the current gmsh model (python API) with its physical volumes as regions."""
    import gmsh

    tags, coords, _ = gmsh.model.mesh.getNodes()
    lookup = np.full(int(tags.max()) + 1, -1, np.int64)
    lookup[tags.astype(np.int64)] = np.arange(len(tags))
    nodes = coords.reshape(-1, 3)
    tets, reg, names = [], [], {}
    for dim, ptag in gmsh.model.getPhysicalGroups(3):
        name = gmsh.model.getPhysicalName(dim, ptag) or f"region{ptag}"
        names[name] = ptag
        for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, ptag):
            types, _, elnodes = gmsh.model.mesh.getElements(3, ent)
            for t, en in zip(types, elnodes, strict=True):
                if t == 4:
                    e = lookup[en.astype(np.int64)].reshape(-1, 4)
                    tets.append(e)
                    reg.append(np.full(len(e), ptag))
    tets = np.concatenate(tets)
    used = np.unique(tets)
    remap = np.full(len(nodes), -1, np.int64)
    remap[used] = np.arange(len(used))
    return TetMesh(nodes[used], remap[tets], np.concatenate(reg), names)
