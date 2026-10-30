"""Conforming tetrahedral finite elements (P1 and isoparametric P2) on gmsh meshes: the reference that the immersed
solver is compared against, at equal accuracy, on the same parts.

The geometry is built in gmsh's OpenCASCADE kernel (or read from an STL and turned into a volume) and meshed
in-process, so the meshing time is measured; P2 mid-edge nodes are placed on the curved CAD surfaces. Steady heat
(source, prescribed temperatures as strong nodal conditions, heat flux and convection on boundary faces) and linear
elasticity use the boundary-condition classes of :mod:`pinneapple_simulation.immersed.bcs`; the same mesh can be
handed to CalculiX (:meth:`TetMesh.to_calculix`).

    from pinneapple_simulation.numerical_solvers import tet_fem as T
    mesh = T.gmsh_mesh(T.occ_sphere((0, 0, 0), 0.5), size=0.05, order=2)
    sol = T.solve_heat(mesh, conductivity=2.0, source=100.0, bcs=[Convection(50.0, 20.0)])
    sol(points), sol.errors(exact, grad), mesh.meshing_s
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse

# local corner indices of the faces (face i is opposite vertex i) and the gmsh tet10 edge order
_FACES = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
_EDGES = [(0, 1), (1, 2), (2, 0), (3, 0), (3, 2), (3, 1)]


def _edge_index(i: int, j: int) -> int:
    for k, (a, b) in enumerate(_EDGES):
        if {a, b} == {i, j}:
            return 4 + k
    raise KeyError((i, j))


# ---------------------------------------------------------------- shape functions (barycentric)

def shape(order: int, lam: np.ndarray) -> np.ndarray:
    """Values (q, nb) at barycentric points (q, 4)."""
    if order == 1:
        return lam.copy()
    N = np.empty((len(lam), 10))
    N[:, :4] = lam * (2 * lam - 1)
    for k, (a, b) in enumerate(_EDGES):
        N[:, 4 + k] = 4 * lam[:, a] * lam[:, b]
    return N


def dshape(order: int, lam: np.ndarray) -> np.ndarray:
    """Derivatives (q, nb, 3) with respect to the reference coordinates (lam_1, lam_2, lam_3); lam_0 = 1 - sum."""
    dl = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)            # d lam_i / d xi
    if order == 1:
        return np.broadcast_to(dl, (len(lam), 4, 3)).copy()
    D = np.empty((len(lam), 10, 3))
    for i in range(4):
        D[:, i] = (4 * lam[:, i] - 1)[:, None] * dl[i]
    for k, (a, b) in enumerate(_EDGES):
        D[:, 4 + k] = 4 * (lam[:, a][:, None] * dl[b] + lam[:, b][:, None] * dl[a])
    return D


def _rule(dim: int, n: int):
    from pinneapple_simulation.immersed.quadrature import simplex_rule
    return simplex_rule(dim, n)


# ---------------------------------------------------------------- meshes

@dataclass
class TetMesh:
    nodes: np.ndarray                  # (n, 3)
    tets: np.ndarray                   # (m, 4) or (m, 10), gmsh order
    order: int
    meshing_s: float = 0.0
    info: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def boundary_faces(self) -> tuple[np.ndarray, np.ndarray]:
        """(tet index, local face) of every boundary face."""
        F = self.tets[:, _FACES].reshape(-1, 3)
        key = np.sort(F, axis=1)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        b = np.nonzero(cnt[inv.ravel()] == 1)[0]
        return b // 4, b % 4

    def face_nodes(self, tet: np.ndarray, face: np.ndarray) -> np.ndarray:
        """Nodes of boundary faces: 3 corners, then (P2) the mid nodes of edges (c0 c1), (c1 c2), (c2 c0)."""
        loc = _FACES[face]
        cols = [loc[:, 0], loc[:, 1], loc[:, 2]]
        if self.order == 2:
            cols += [np.array([_edge_index(a, b) for a, b in zip(loc[:, i], loc[:, (i + 1) % 3], strict=True)])
                     for i in range(3)]
        return np.stack([self.tets[tet, c] for c in cols], axis=1)

    def volume(self) -> float:
        lam, w = _rule(3, 3)
        J = self._jacobians(lam)
        return float(np.sum(np.abs(np.linalg.det(J)) * w[None] / 6))

    def _jacobians(self, lam):
        D = dshape(self.order, lam)                                     # (q, nb, 3)
        X = self.nodes[self.tets]                                       # (m, nb, 3)
        return np.einsum("mbi,qbj->mqij", X, D)                         # dX_i / dxi_j

    def to_calculix(self):
        """The same mesh as a CalculiX model (C3D4 / C3D10 node order)."""
        from pinneapple_simulation.external_solvers.calculix.study import FEModel
        if self.order == 1:
            return FEModel(self.nodes.copy(), self.tets.copy(), "C3D4")
        perm = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]                            # gmsh tet10 -> CalculiX C3D10
        return FEModel(self.nodes.copy(), self.tets[:, perm].copy(), "C3D10")


def occ_sphere(center, radius: float) -> Callable:
    return lambda g: g.model.occ.addSphere(*center, radius)


def occ_cylinder(center, radius: float, length: float, axis: int = 2) -> Callable:
    def build(g):
        c = np.asarray(center, float)
        d = np.zeros(3)
        d[axis] = length
        return g.model.occ.addCylinder(*(c - d / 2), *d, radius)
    return build


def occ_torus(center, R: float, r: float) -> Callable:
    return lambda g: g.model.occ.addTorus(*center, R, r)


def occ_stl(path: str, angle_deg: float = 40.0) -> Callable:
    """A closed STL as a gmsh volume (its facets are the surface mesh to start from: no CAD reconstruction)."""
    def build(g):
        g.merge(path)
        g.model.mesh.classifySurfaces(np.radians(angle_deg), True, True, np.pi)
        g.model.mesh.createGeometry()
        s = g.model.getEntities(2)
        loop = g.model.geo.addSurfaceLoop([e[1] for e in s])
        v = g.model.geo.addVolume([loop])
        g.model.geo.synchronize()
        return v
    return build


def gmsh_mesh(build: Callable, size: float, order: int = 2, optimize: bool = True, threads: int = 1) -> TetMesh:
    """Mesh the volume made by ``build(gmsh)`` with tetrahedra of size ``size`` (P2 nodes on the CAD surfaces).
    ``meshing_s`` is the wall time of gmsh's generation (and high-order placement / optimisation)."""
    import gmsh
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", threads)
        gmsh.model.add("part")
        build(gmsh)
        try:
            gmsh.model.occ.synchronize()
        except Exception:
            pass
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size / 4)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        t0 = time.perf_counter()
        gmsh.model.mesh.generate(3)
        if optimize:
            gmsh.model.mesh.optimize("Netgen")
        if order == 2:
            gmsh.model.mesh.setOrder(2)
            if optimize:
                gmsh.model.mesh.optimize("HighOrder")
        dt = time.perf_counter() - t0
        tags, coords, _ = gmsh.model.mesh.getNodes()
        X = coords.reshape(-1, 3)
        etype = 4 if order == 1 else 11
        _, enodes = gmsh.model.mesh.getElementsByType(etype)
        nb = 4 if order == 1 else 10
        pos = np.zeros(int(tags.max()) + 1, np.int64)
        pos[tags.astype(np.int64)] = np.arange(len(tags))
        T = pos[enodes.astype(np.int64)].reshape(-1, nb)
    finally:
        gmsh.finalize()
    used, inv = np.unique(T, return_inverse=True)
    mesh = TetMesh(X[used], inv.reshape(T.shape), order, dt, {"elements": len(T), "size": size})
    # make every element positively oriented
    lam = np.full((1, 4), 0.25)
    neg = np.linalg.det(mesh._jacobians(lam)[:, 0]) < 0
    if neg.any():
        swap = [0, 2, 1, 3] + ([] if order == 1 else [_edge_index(0, 2), _edge_index(2, 1), _edge_index(1, 0),
                                                       _edge_index(3, 0), _edge_index(3, 1), _edge_index(3, 2)])
        mesh.tets[neg] = mesh.tets[neg][:, swap]
    return mesh


# ---------------------------------------------------------------- assembly helpers

def _scatter(conn: np.ndarray, mats: np.ndarray, n: int) -> sparse.csr_matrix:
    nb = conn.shape[1]
    r = np.repeat(conn, nb, axis=1).ravel()
    c = np.tile(conn, (1, nb)).ravel()
    return sparse.csr_matrix((mats.ravel(), (r, c)), shape=(n, n))


@dataclass
class _Volume:
    N: np.ndarray                      # (q, nb)
    G: np.ndarray                      # (m, q, nb, 3) physical gradients
    w: np.ndarray                      # (m, q) physical weights
    X: np.ndarray                      # (m, q, 3)


def _volume(mesh: TetMesh, n: int | None = None) -> _Volume:
    lam, w = _rule(3, n or (2 if mesh.order == 1 else 3))
    D = dshape(mesh.order, lam)
    J = mesh._jacobians(lam)                                            # (m, q, 3, 3)
    det = np.linalg.det(J)
    Jinv = np.linalg.inv(J)
    G = np.einsum("qbj,mqji->mqbi", D, Jinv)
    N = shape(mesh.order, lam)
    X = np.einsum("qb,mbi->mqi", N, mesh.nodes[mesh.tets])
    return _Volume(N, G, np.abs(det) * w[None] / 6, X)


@dataclass
class _Faces:
    tet: np.ndarray
    nodes: np.ndarray                  # (f, nf)
    N: np.ndarray                      # (q, nf)
    w: np.ndarray                      # (f, q) area weights
    X: np.ndarray                      # (f, q, 3)
    n: np.ndarray                      # (f, q, 3) outward normals


def _faces(mesh: TetMesh, tet=None, face=None, nq: int = 3) -> _Faces:
    if tet is None:
        tet, face = mesh.boundary_faces()
    fn = mesh.face_nodes(tet, face)
    lam, w = _rule(2, nq)                                               # (q, 3)
    if mesh.order == 1:
        N = lam.copy()
        dN = np.broadcast_to(np.array([[-1, -1], [1, 0], [0, 1]], float), (len(lam), 3, 2))
    else:
        dl = np.array([[-1, -1], [1, 0], [0, 1]], float)
        N = np.concatenate([lam * (2 * lam - 1), 4 * np.stack([lam[:, 0] * lam[:, 1], lam[:, 1] * lam[:, 2],
                                                                lam[:, 2] * lam[:, 0]], 1)], 1)
        dN = np.empty((len(lam), 6, 2))
        for i in range(3):
            dN[:, i] = (4 * lam[:, i] - 1)[:, None] * dl[i]
        for k, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
            dN[:, 3 + k] = 4 * (lam[:, a][:, None] * dl[b] + lam[:, b][:, None] * dl[a])
    P = mesh.nodes[fn]                                                  # (f, nf, 3)
    X = np.einsum("qb,fbi->fqi", N, P)
    T = np.einsum("qbj,fbi->fqij", dN, P)                              # tangents
    nrm = np.cross(T[..., 0], T[..., 1])
    area = np.linalg.norm(nrm, axis=-1)
    nrm = nrm / np.maximum(area, 1e-300)[..., None]
    opp = mesh.nodes[mesh.tets[tet, face]]                              # the vertex opposite the face
    sgn = np.sign(np.einsum("fqi,fqi->fq", X - opp[:, None], nrm))
    nrm = nrm * sgn[..., None]
    return _Faces(tet, fn, N, area * w[None] / 2, X, nrm)


def _owner(bcs: Sequence, F: _Faces) -> np.ndarray:
    """Condition owning each boundary face (decided at its centroid point, with its normal)."""
    from pinneapple_simulation.immersed.bcs import assign
    c = F.X.mean(1)
    nn = F.n.mean(1)
    nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-300)
    return assign(bcs, c, nn) if bcs else np.full(len(c), -1)


# ---------------------------------------------------------------- heat

@dataclass
class TetHeatSolution:
    mesh: TetMesh
    T: np.ndarray
    info: dict = field(default_factory=dict)
    balance: dict = field(default_factory=dict)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return evaluate(self.mesh, self.T, X)

    def errors(self, exact, grad=None) -> dict:
        return field_errors(self.mesh, self.T, exact, grad)


def assemble_heat(mesh: TetMesh, conductivity: float = 1.0, source: Any = 0.0, bcs: Sequence = ()):
    from pinneapple_simulation.immersed.bcs import evaluate as ev
    t0 = time.perf_counter()
    n = mesh.n_nodes
    k = float(conductivity)
    vol = _volume(mesh)
    Ke = k * np.einsum("mq,mqik,mqjk->mij", vol.w, vol.G, vol.G)
    A = _scatter(mesh.tets, Ke, n)
    fq = ev(source, vol.X.reshape(-1, 3)).reshape(vol.w.shape)
    b = np.bincount(mesh.tets.ravel(), np.einsum("mq,qi->mi", vol.w * fq, vol.N).ravel(), minlength=n)
    F = _faces(mesh)
    owner = _owner(bcs, F)
    fixed = {}
    for i, bc in enumerate(bcs):
        sel = np.nonzero(owner == i)[0]
        if not len(sel):
            continue
        X, nn, w, fnod = F.X[sel], F.n[sel], F.w[sel], F.nodes[sel]
        flat = X.reshape(-1, 3), nn.reshape(-1, 3)
        if bc.kind == "neumann":
            q = ev(bc.flux, *flat).reshape(w.shape)
            b += np.bincount(fnod.ravel(), np.einsum("fq,qi->fi", w * q, F.N).ravel(), minlength=n)
        elif bc.kind == "robin":
            hc = ev(bc.h, *flat).reshape(w.shape)
            Ti = ev(bc.T_inf, *flat).reshape(w.shape)
            A = A + _scatter(fnod, np.einsum("fq,qi,qj->fij", w * hc, F.N, F.N), n)
            b += np.bincount(fnod.ravel(), np.einsum("fq,qi->fi", w * hc * Ti, F.N).ravel(), minlength=n)
        else:
            nodes = np.unique(fnod)
            # nodal normal: mean of the face normals around the node, for callables of (X, n)
            nsum = np.zeros((n, 3))
            np.add.at(nsum, fnod[:, :3].ravel(), np.repeat(nn.mean(1), 3, axis=0))
            np.add.at(nsum, fnod.ravel(), np.zeros((fnod.size, 3)))
            nrm = nsum[nodes] / np.maximum(np.linalg.norm(nsum[nodes], axis=1, keepdims=True), 1e-300)
            g = ev(bc.value, mesh.nodes[nodes], nrm)
            for a, v in zip(nodes, g, strict=True):
                fixed.setdefault(int(a), float(v))
    info = {"owner": owner, "fixed": fixed}
    info["assembly_s"] = time.perf_counter() - t0
    return A.tocsr(), b, info


def _apply_dirichlet(A, b, fixed: dict):
    n = A.shape[0]
    if not fixed:
        return A, b, np.arange(n), np.zeros(0, int), np.zeros(0)
    idx = np.fromiter(fixed.keys(), int)
    val = np.fromiter(fixed.values(), float)
    free = np.setdiff1d(np.arange(n), idx)
    b2 = b - A[:, idx] @ val
    return A[free][:, free], b2[free], free, idx, val


def solve_heat(mesh: TetMesh, conductivity: float = 1.0, source: Any = 0.0, bcs: Sequence = (),
               solver: str = "auto") -> TetHeatSolution:
    from pinneapple_simulation.immersed.assembly import solve_linear
    A, b, info = assemble_heat(mesh, conductivity, source, bcs)
    Af, bf, free, idx, val = _apply_dirichlet(A, b, info.pop("fixed"))
    x, sinfo = solve_linear(Af, bf, solver)
    T = np.zeros(mesh.n_nodes)
    T[free], T[idx] = x, val
    owner = info.pop("owner")
    info.update({"solve_" + k: v for k, v in sinfo.items()}, dofs=len(free), meshing_s=mesh.meshing_s)
    sol = TetHeatSolution(mesh, T, info)
    sol.balance = _heat_balance(sol, A, b, conductivity, source, bcs, owner)
    return sol


def _heat_balance(sol, A, b, k, source, bcs, owner) -> dict:
    """Heat generated against heat leaving: on fixed-temperature nodes the consistent (reaction) flux."""
    from pinneapple_simulation.immersed.bcs import evaluate as ev
    mesh = sol.mesh
    vol = _volume(mesh)
    gen = float(np.sum(vol.w * ev(source, vol.X.reshape(-1, 3)).reshape(vol.w.shape)))
    F = _faces(mesh)
    out = {"generated_W": gen}
    total = scale = 0.0
    r = A @ sol.T - b                                                   # residual = reactions at fixed nodes
    for i, bc in enumerate(bcs):
        sel = owner == i
        if not sel.any():
            continue
        X, nn, w = F.X[sel].reshape(-1, 3), F.n[sel].reshape(-1, 3), F.w[sel].ravel()
        if bc.kind == "neumann":
            q = -float(np.sum(w * ev(bc.flux, X, nn)))
        elif bc.kind == "robin":
            Th = evaluate(mesh, sol.T, X, tet=np.repeat(F.tet[sel], F.w.shape[1]))
            q = float(np.sum(w * ev(bc.h, X, nn) * (Th - ev(bc.T_inf, X, nn))))
        else:
            q = float(-r[np.unique(F.nodes[sel])].sum())
        out[f"{type(bc).__name__}[{i}]_out_W"] = q
        total += q
        scale += abs(q)
    out["out_W"] = total
    out["imbalance"] = (total - gen) / max(abs(gen), scale, 1e-300)
    return out


# ---------------------------------------------------------------- evaluation and errors

def _locate(mesh: TetMesh, X: np.ndarray):
    """Containing element and barycentric coordinates (straight-sided map; P2 corrected by Newton steps)."""
    from scipy.spatial import cKDTree
    P = mesh.nodes[mesh.tets[:, :4]]
    cen = P.mean(1)
    tree = cKDTree(cen)
    _, cand = tree.query(X, k=min(16, len(cen)))
    cand = np.atleast_2d(cand)
    best = np.full(len(X), -1)
    lam_best = np.zeros((len(X), 4))
    score = np.full(len(X), -np.inf)
    for j in range(cand.shape[1]):
        e = cand[:, j]
        M = (P[e, 1:] - P[e, :1]).transpose(0, 2, 1)
        r = np.linalg.solve(M, (X - P[e, 0])[..., None])[..., 0]
        lam = np.column_stack([1 - r.sum(1), r])
        s = lam.min(1)
        better = s > score
        best[better], lam_best[better], score[better] = e[better], lam[better], s[better]
    return best, lam_best


def evaluate(mesh: TetMesh, u: np.ndarray, X: np.ndarray, tet: np.ndarray | None = None) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, float))
    if tet is None:
        tet, lam = _locate(mesh, X)
    else:
        P = mesh.nodes[mesh.tets[tet, :4]]
        M = (P[:, 1:] - P[:, :1]).transpose(0, 2, 1)
        r = np.linalg.solve(M, (X - P[:, 0])[..., None])[..., 0]
        lam = np.column_stack([1 - r.sum(1), r])
    if mesh.order == 2:                                                 # Newton on the curved map
        for _ in range(4):
            N = shape(2, lam)
            Xm = np.einsum("nb,nbi->ni", N, mesh.nodes[mesh.tets[tet]])
            D = dshape(2, lam)
            J = np.einsum("nbi,nbj->nij", mesh.nodes[mesh.tets[tet]], D)
            dl = np.linalg.solve(J, (X - Xm)[..., None])[..., 0]
            r = lam[:, 1:] + dl
            lam = np.column_stack([1 - r.sum(1), r])
    U = np.asarray(u)
    return np.einsum("nb,nb...->n...", shape(mesh.order, lam), U[mesh.tets[tet]])


def field_errors(mesh: TetMesh, u: np.ndarray, exact, grad=None, ncomp: int = 1) -> dict:
    vol = _volume(mesh, 4)
    U = np.asarray(u).reshape(mesh.n_nodes, ncomp)
    uh = np.einsum("qb,mbc->mqc", vol.N, U[mesh.tets])
    X = vol.X.reshape(-1, 3)
    ue = np.asarray(exact(X), float).reshape(uh.shape)
    out = {"L2": float(np.sqrt(np.sum(vol.w[..., None] * (uh - ue) ** 2))),
           "L2_norm": float(np.sqrt(np.sum(vol.w[..., None] * ue ** 2)))}
    out["L2_rel"] = out["L2"] / max(out["L2_norm"], 1e-300)
    if grad is not None:
        gh = np.einsum("mqbk,mbc->mqck", vol.G, U[mesh.tets])
        ge = np.asarray(grad(X), float).reshape(gh.shape)
        out["H1"] = float(np.sqrt(np.sum(vol.w[..., None, None] * (gh - ge) ** 2)))
        out["H1_norm"] = float(np.sqrt(np.sum(vol.w[..., None, None] * ge ** 2)))
        out["H1_rel"] = out["H1"] / max(out["H1_norm"], 1e-300)
    return out
