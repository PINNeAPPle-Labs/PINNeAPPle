"""First-order Nedelec (Whitney edge) elements for 3-D magnetostatics and time-harmonic eddy currents.

Unknowns are the line integrals of the magnetic vector potential ``A`` along the mesh edges. On a tetrahedron the
edge function of edge ``(a, b)`` is ``W = lambda_a grad(lambda_b) - lambda_b grad(lambda_a)``, its curl the constant
``2 grad(lambda_a) x grad(lambda_b)``; ``B = curl A`` is therefore constant per tetrahedron and its normal
component is continuous across faces (``div B = 0`` exactly).

Magnetostatics: ``curl(nu curl A) = J + curl(nu Br)``, weakly

    sum_K int nu curl A . curl W = int J . W + int nu Br . curl W,

with ``n x A`` prescribed (Dirichlet edges, default zero: ``B . n = 0``) on the outer boundary. Nonlinear iron
by Newton on the energy functional, the tangent ``nu + 2 nu'(|B|^2) B B^T`` and a backtracking line search.

Eddy currents (modified vector potential ``A*``, ``E = -j omega A*`` in conductors, ``exp(+j omega t)``):
``curl(nu curl A) + j omega sigma A = J_s``.

Gauge: the curl-curl operator is singular on gradients. ``gauge="tree"`` removes the edges of a spanning tree of
the nodes that carry a gradient degree of freedom (nodes that are neither on a Dirichlet boundary nor in a
conductor), so the reduced matrix is non-singular and can be factorised. ``gauge="none"`` keeps every edge and
solves the consistent singular system with conjugate gradients (magnetostatics only). In both cases the source
vector is first made discretely divergence-free, ``G^T b = 0`` for the gradient ``G`` of every free node, by
subtracting its L2 projection on gradients (a P1 Poisson solve): a current density sampled on a mesh is not
discretely solenoidal, and a tree-gauged system turns its divergence into a spurious field.

References: A. Bossavit, *Computational Electromagnetism*, Academic Press, 1998 (Whitney forms, gauging by trees);
J.-M. Jin, *The Finite Element Method in Electromagnetics*, 3rd ed., Wiley, 2014 (edge elements);
R. Albanese and G. Rubinacci, "Solution of three dimensional eddy current problems by integral and differential
methods", IEEE Trans. Magn. 24 (1988) 98-101 (tree gauge and source consistency).
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .mesh import LOCAL_EDGES, TetMesh

__all__ = ["MU0", "NU0", "EdgeSpace", "MagnetostaticSolution3D", "EddySolution3D", "solve_magnetostatic",
           "solve_eddy", "tree_gauge", "cocg"]

MU0 = 4e-7 * math.pi
NU0 = 1.0 / MU0

# 4-point Gauss rule on the tetrahedron (degree 2): barycentric coordinates and weights (sum 1)
_A, _B = 0.5854101966249685, 0.1381966011250105
QUAD4 = np.array([[_A, _B, _B, _B], [_B, _A, _B, _B], [_B, _B, _A, _B], [_B, _B, _B, _A]])
QUAD4_W = np.full(4, 0.25)
GAUSS3 = (np.array([0.5 - math.sqrt(15) / 10, 0.5, 0.5 + math.sqrt(15) / 10]), np.array([5, 8, 5]) / 18.0)


class EdgeSpace:
    """Whitney edge functions on a :class:`TetMesh`: element matrices, assembly, sources and evaluation."""

    def __init__(self, mesh: TetMesh):
        self.mesh = mesh
        gl = mesh.grad_lambda                                       # (M, 4, 3)
        s = mesh.tet_edge_sign                                      # (M, 6)
        ga, gb = gl[:, LOCAL_EDGES[:, 0]], gl[:, LOCAL_EDGES[:, 1]]  # (M, 6, 3)
        self.ga, self.gb, self.sign = ga, gb, s
        self.curls = 2.0 * np.cross(ga, gb) * s[:, :, None]         # (M, 6, 3) signed curls
        self.vol = mesh.volumes
        self.dofs = mesh.tet_edges
        rows = np.repeat(self.dofs, 6, axis=1).ravel()
        cols = np.tile(self.dofs, (1, 6)).ravel()
        self._rc = (rows, cols)
        n = mesh.n_edges
        self.n = n

    # ------------------------------------------------------------------------------------ matrices
    def _assemble(self, local: np.ndarray) -> sp.csr_matrix:
        return sp.csr_matrix((local.ravel(), self._rc), shape=(self.n, self.n))

    def stiffness(self, nu) -> sp.csr_matrix:
        """``int nu curl W_i . curl W_j`` with ``nu`` per tetrahedron (scalar)."""
        nu = np.broadcast_to(np.asarray(nu, float), self.vol.shape)
        loc = np.einsum("kid,kjd->kij", self.curls, self.curls) * (nu * self.vol)[:, None, None]
        return self._assemble(loc)

    def tangent(self, nu, dnu, B) -> sp.csr_matrix:
        """Newton tangent ``int (nu curl W_i . curl W_j + 2 nu' (B . curl W_i)(B . curl W_j))``."""
        cb = np.einsum("kid,kd->ki", self.curls, B)
        loc = np.einsum("kid,kjd->kij", self.curls, self.curls) * (nu * self.vol)[:, None, None]
        loc += np.einsum("ki,kj->kij", cb, cb) * (2.0 * dnu * self.vol)[:, None, None]
        return self._assemble(loc)

    def mass(self, coef) -> sp.csr_matrix:
        """``int coef W_i . W_j`` (exact for constant ``coef`` per tetrahedron)."""
        coef = np.broadcast_to(np.asarray(coef, float), self.vol.shape)
        ga, gb = self.ga, self.gb
        ia, ib = LOCAL_EDGES[:, 0], LOCAL_EDGES[:, 1]
        # int lambda_p lambda_q = vol (1 + delta_pq) / 20
        L = (np.ones((4, 4)) + np.eye(4)) / 20.0
        dd = lambda u, v: np.einsum("kid,kjd->kij", u, v)            # noqa: E731
        loc = (dd(gb, gb) * L[ia][:, ia][None] - dd(gb, ga) * L[ia][:, ib][None]
               - dd(ga, gb) * L[ib][:, ia][None] + dd(ga, ga) * L[ib][:, ib][None])
        loc *= np.einsum("ki,kj->kij", self.sign, self.sign) * (coef * self.vol)[:, None, None]
        return self._assemble(loc)

    def gradient(self) -> sp.csr_matrix:
        """Discrete gradient (edges x nodes): the edge circulation of a nodal P1 field."""
        E = self.mesh.edges
        n = len(E)
        return sp.csr_matrix((np.r_[-np.ones(n), np.ones(n)], (np.r_[np.arange(n), np.arange(n)],
                                                                 np.r_[E[:, 0], E[:, 1]])),
                             shape=(n, len(self.mesh.nodes)))

    # ------------------------------------------------------------------------------------ sources
    def basis_at(self, k: np.ndarray, lam: np.ndarray) -> np.ndarray:
        """Signed edge functions (n, 6, 3) at barycentric points ``lam`` (n, 4) of tetrahedra ``k``."""
        ia, ib = LOCAL_EDGES[:, 0], LOCAL_EDGES[:, 1]
        la, lb = lam[:, ia], lam[:, ib]
        W = la[:, :, None] * self.gb[k] - lb[:, :, None] * self.ga[k]
        return W * self.sign[k][:, :, None]

    def current_load(self, J, mask: np.ndarray | None = None) -> np.ndarray:
        """``int J . W_i`` for a current density ``J``: an (M, 3) array constant per tetrahedron, or a callable
        ``J(points) -> (n, 3)`` integrated with a 4-point rule over the tetrahedra in ``mask`` (default all)."""
        M = len(self.vol)
        sel = np.arange(M) if mask is None else np.flatnonzero(mask)
        b = np.zeros(self.n)
        if len(sel) == 0:
            return b
        if callable(J):
            x = self.mesh.nodes[self.mesh.tets[sel]]                  # (m, 4, 3)
            for q in range(4):
                lam = np.broadcast_to(QUAD4[q], (len(sel), 4))
                pts = np.einsum("kv,kvd->kd", lam, x)
                Jq = np.asarray(J(pts), float)
                W = self.basis_at(sel, lam)
                val = np.einsum("kid,kd->ki", W, Jq) * (QUAD4_W[q] * self.vol[sel])[:, None]
                np.add.at(b, self.dofs[sel], val)
        else:
            J = np.asarray(J, float)[sel]
            # int W = vol (grad lambda_b - grad lambda_a) / 4
            Wint = (self.gb[sel] - self.ga[sel]) * self.sign[sel][:, :, None] / 4.0
            val = np.einsum("kid,kd->ki", Wint, J) * self.vol[sel][:, None]
            np.add.at(b, self.dofs[sel], val)
        return b

    def magnet_load(self, nu, Br) -> np.ndarray:
        """``int nu Br . curl W_i`` (remanence ``Br`` (M, 3) per tetrahedron)."""
        nu = np.broadcast_to(np.asarray(nu, float), self.vol.shape)
        val = np.einsum("kid,kd->ki", self.curls, Br) * (nu * self.vol)[:, None]
        b = np.zeros(self.n)
        np.add.at(b, self.dofs, val)
        return b

    def edge_values(self, A: Callable[[np.ndarray], np.ndarray], edges: np.ndarray | None = None) -> np.ndarray:
        """Degrees of freedom of a vector field: line integrals of ``A`` along the edges (3-point Gauss)."""
        E = self.mesh.edges if edges is None else self.mesh.edges[edges]
        x0, x1 = self.mesh.nodes[E[:, 0]], self.mesh.nodes[E[:, 1]]
        t = x1 - x0
        out = np.zeros(len(E))
        for s, w in zip(*GAUSS3, strict=True):
            out = out + w * np.einsum("kd,kd->k", np.asarray(A(x0 + s * t), float), t)
        return out

    # ------------------------------------------------------------------------------------ evaluation
    def curl(self, a: np.ndarray) -> np.ndarray:
        """``B = curl A`` per tetrahedron (M, 3)."""
        return np.einsum("ki,kid->kd", a[self.dofs], self.curls)

    def evaluate(self, a: np.ndarray, points: np.ndarray, cells: np.ndarray | None = None) -> np.ndarray:
        """The vector field ``A`` at points (NaN outside the mesh)."""
        points = np.atleast_2d(points)
        k = self.mesh.locate(points) if cells is None else cells
        out = np.full((len(points), 3), np.nan, dtype=a.dtype)
        ok = k >= 0
        kk = k[ok]
        Jinv = np.linalg.inv(self.mesh.jacobians[kk])
        xi = np.einsum("kji,kj->ki", Jinv, points[ok] - self.mesh.nodes[self.mesh.tets[kk, 0]])
        lam = np.column_stack([1 - xi.sum(1), xi])
        W = self.basis_at(kk, lam)
        out[ok] = np.einsum("ki,kid->kd", a[self.dofs[kk]], W)
        return out


# ---------------------------------------------------------------------------------------------- gauge
def tree_gauge(mesh: TetMesh, fixed_edges: np.ndarray, root_nodes: np.ndarray) -> np.ndarray:
    """Edges of a spanning tree that reaches every node not in ``root_nodes`` (breadth first from all roots at
    once) using only edges not in ``fixed_edges``. Removing them leaves exactly one unknown per independent
    curl: the gradient of every free node is spanned by tree edges."""
    from scipy.sparse.csgraph import breadth_first_order

    N = len(mesh.nodes)
    E = mesh.edges
    allowed = np.ones(len(E), bool)
    allowed[fixed_edges] = False
    roots = np.zeros(N, bool)
    roots[root_nodes] = True
    if not roots.any():
        roots[0] = True
    eid = np.flatnonzero(allowed)
    a, b = E[eid, 0], E[eid, 1]
    V = N                                                      # virtual node joined to every root
    r = np.flatnonzero(roots)
    rows = np.r_[a, b, np.full(len(r), V), r]
    cols = np.r_[b, a, r, np.full(len(r), V)]
    data = np.r_[eid + 1, eid + 1, np.zeros(len(r)), np.zeros(len(r))].astype(float)
    data[data == 0] = -1                                       # virtual links
    G = sp.csr_matrix((data, (rows, cols)), shape=(N + 1, N + 1))
    _, pred = breadth_first_order(G, V, directed=False, return_predecessors=True)
    free = np.flatnonzero(~roots)
    unreached = free[pred[free] < 0]
    if len(unreached):                                         # components without a root: one root each
        extra = []
        seen = np.zeros(N, bool)
        for n0 in unreached:
            if seen[n0]:
                continue
            order, _ = breadth_first_order(G, n0, directed=False, return_predecessors=True)
            seen[order[order < N]] = True
            extra.append(n0)
        return tree_gauge(mesh, fixed_edges, np.r_[root_nodes, extra].astype(np.int64))
    p = pred[free]
    child = free[p != V]
    parent = p[p != V]
    return mesh.edge_index(np.column_stack([child, parent]))


def _project_source(space: EdgeSpace, b: np.ndarray, free_nodes: np.ndarray) -> np.ndarray:
    """``b - M G phi`` with ``(G^T M G) phi = G^T b`` on the free nodes: afterwards ``G^T b = 0`` there."""
    G = space.gradient()[:, free_nodes]
    gb = G.T @ b
    if np.linalg.norm(gb) <= 1e-14 * max(np.linalg.norm(b), 1e-300):
        return b
    M = space.mass(1.0)
    S = (G.T @ M @ G).tocsc()
    phi = spla.spsolve(S, gb)
    return b - M @ (G @ phi)


def _factorise(Kr):
    return spla.splu(Kr.tocsc(), permc_spec="COLAMD")


def cocg(A, b, M_inv, rtol: float = 1e-9, maxiter: int = 20000):
    """Conjugate orthogonal conjugate gradients for complex symmetric ``A`` (van der Vorst and Melissen, IEEE
    Trans. Magn. 26 (1990) 706-708): CG with the unconjugated product ``x^T y``, diagonal preconditioner
    ``M_inv`` (an array: diagonal) or symmetric preconditioner (a callable). Returns the solution and the number
    of iterations; raises if the relative residual stays above ``rtol``."""
    prec = M_inv if callable(M_inv) else (lambda v: M_inv * v)
    x = np.zeros_like(b)
    r = b.copy()
    z = prec(r)
    p = z.copy()
    rho = r @ z
    nb = np.linalg.norm(b)
    if nb == 0:
        return x, 0
    for k in range(1, maxiter + 1):
        q = A @ p
        alpha = rho / (p @ q)
        x += alpha * p
        r -= alpha * q
        if np.linalg.norm(r) <= rtol * nb:
            return x, k
        z = prec(r)
        rho_new = r @ z
        p = z + (rho_new / rho) * p
        rho = rho_new
    raise RuntimeError(f"COCG did not converge in {maxiter} iterations (residual {np.linalg.norm(r) / nb:.2e})")


# ---------------------------------------------------------------------------------------------- magnetostatics
@dataclass
class MagnetostaticSolution3D:
    mesh: TetMesh
    a: np.ndarray                  # edge degrees of freedom (Wb)
    B: np.ndarray                  # (M, 3) flux density per tetrahedron (T)
    H: np.ndarray                  # (M, 3) field strength (A/m)
    energy: float                  # stored magnetic energy int w(B) dV (J)
    coenergy: float                # int (H . B - w) dV (J)
    iterations: int
    residual: float
    info: dict = field(default_factory=dict)

    def B_at(self, points) -> np.ndarray:
        k = self.mesh.locate(points)
        out = np.full((len(k), 3), np.nan)
        out[k >= 0] = self.B[k[k >= 0]]
        return out

    def source_work(self) -> float:
        """``int J . A`` with the (discretely solenoidal) current load of the solve. For a source ``J = i j1``
        it is ``i psi``, ``psi`` the flux linkage of the winding ``j1``, and ``d W' / d i = psi`` (co-energy
        derivative tree: :mod:`pinneapple_physics.potential`, issue #928)."""
        return float(self.info["current_load"] @ self.a)


def _energy_density(B2, BdotBr, nu_lin, nonlinear_masks, Br2):
    """Energy and co-energy densities. Linear (magnet) material with ``H = nu (B - Br)``:
    ``w = nu |B - Br|^2 / 2`` and ``w' = H . B - w``; nonlinear: ``w = int H db``."""
    w = 0.5 * nu_lin * (B2 - 2 * BdotBr + Br2)
    for mask, curve in nonlinear_masks:
        w[mask] = curve.energy(B2[mask])
    return w


def solve_magnetostatic(mesh: TetMesh, nu, J=None, J_mask=None, Br=None, nonlinear: Mapping | None = None,
                        dirichlet: tuple[np.ndarray, np.ndarray] | None = None, gauge: str = "none",
                        tol: float = 1e-9, max_iter: int = 50, space: EdgeSpace | None = None,
                        a0: np.ndarray | None = None, cg_tol: float = 1e-10, verbose: bool = False
                        ) -> MagnetostaticSolution3D:
    """3-D magnetostatics on edge elements.

    nu: reluctivity per tetrahedron (M,) or scalar (``1 / (mu0 mu_r)``); for nonlinear regions it is the initial
    guess only. J: (M, 3) current density per tetrahedron or a callable of points (evaluated on ``J_mask``).
    Br: (M, 3) remanence. nonlinear: ``{region_tag: curve}`` with ``curve.nu(s)``, ``curve.dnu(s)``,
    ``curve.energy(s)`` of ``s = |B|^2`` (e.g. :class:`~pinneapple_simulation.electromagnetics.materials.BHCurve`).
    dirichlet: ``(edges, values)`` of prescribed edge circulations; default: every outer-boundary edge zero.
    gauge: ``"none"`` (default: conjugate gradients with a Jacobi preconditioner on the singular, consistent
    system) or ``"tree"`` (sparse LU of the tree-gauged system; exact but slow beyond ~1e5 unknowns).
    """
    t0 = time.perf_counter()
    sp_ = space or EdgeSpace(mesh)
    M = len(mesh.tets)
    nu_lin = np.broadcast_to(np.asarray(nu, float), (M,)).copy()
    Br = np.zeros((M, 3)) if Br is None else np.asarray(Br, float)
    nl = [(mesh.region == tag, curve) for tag, curve in (nonlinear or {}).items()]
    if dirichlet is None:
        d_edges = mesh.boundary_edges
        d_vals = np.zeros(len(d_edges))
    else:
        d_edges, d_vals = (np.asarray(v) for v in dirichlet)
    d_nodes = np.unique(mesh.edges[d_edges])
    free_nodes = np.setdiff1d(np.arange(len(mesh.nodes)), d_nodes)
    b = np.zeros(sp_.n)
    if J is not None:
        b = sp_.current_load(J, J_mask)
        b = _project_source(sp_, b, free_nodes)
    b = b + sp_.magnet_load(nu_lin, Br)
    fixed = np.zeros(sp_.n, bool)
    fixed[d_edges] = True
    if gauge == "tree":
        tree = tree_gauge(mesh, d_edges, d_nodes)
        unknown = np.ones(sp_.n, bool)
        unknown[d_edges] = False
        unknown[tree] = False
    elif gauge == "none":
        tree = np.array([], np.int64)
        unknown = ~fixed
    else:
        raise ValueError("gauge must be 'tree' or 'none'")
    u = np.flatnonzero(unknown)
    a = np.zeros(sp_.n) if a0 is None else np.asarray(a0, float).copy()
    a[d_edges] = d_vals
    a[tree] = 0.0

    def state(a):
        B = sp_.curl(a)
        B2 = (B * B).sum(1)
        nu_e = nu_lin.copy()
        dnu_e = np.zeros(M)
        for mask, curve in nl:
            nu_e[mask] = curve.nu(B2[mask])
            dnu_e[mask] = curve.dnu(B2[mask])
        return B, B2, nu_e, dnu_e

    Br2 = (Br * Br).sum(1)
    bJ = b - sp_.magnet_load(nu_lin, Br)

    def functional(a):
        B, B2, _, _ = state(a)
        w = _energy_density(B2, (B * Br).sum(1), nu_lin, nl, Br2)
        return float(np.sum(w * sp_.vol)) - float(bJ @ a)

    def linsolve(Kt, r):
        Kr = Kt[u][:, u]
        if gauge == "tree":
            return _factorise(Kr).solve(-r)
        d = 1.0 / Kr.diagonal()
        x, info = spla.cg(Kr, -r, rtol=cg_tol, maxiter=20 * len(u), M=sp.diags(d))
        if info:
            raise RuntimeError(f"CG did not converge (info={info})")
        return x

    B, B2, nu_e, dnu_e = state(a)
    K = sp_.stiffness(nu_e)
    r = (K @ a - b)[u]
    scale = max(np.linalg.norm(b[u]), np.linalg.norm((K @ a)[u]), 1e-300)
    res = np.linalg.norm(r) / scale
    hist = [res]
    it = 0
    while it < max_iter and res > tol:
        it += 1
        Kt = sp_.tangent(nu_e, dnu_e, B) if nl else K
        d = linsolve(Kt, r)
        s = 1.0
        if nl:
            Pi0 = functional(a)
            slope = float(r @ d)
            if -slope > 1e3 * np.finfo(float).eps * max(abs(Pi0), 1e-300):
                trial = a.copy()
                while s > 1e-6:
                    trial[u] = a[u] + s * d
                    if functional(trial) <= Pi0 + 1e-4 * s * slope:
                        break
                    s *= 0.5
        a[u] = a[u] + s * d
        B, B2, nu_e, dnu_e = state(a)
        K = sp_.stiffness(nu_e)
        r = (K @ a - b)[u]
        res = np.linalg.norm(r) / scale
        hist.append(res)
        if verbose:
            print(f"newton {it}: residual {res:.3e} step {s:.3g}")
        if not nl:
            break
    H = nu_e[:, None] * B
    lin = np.ones(M, bool)
    for mask, _ in nl:
        lin &= ~mask
    H[lin] = nu_lin[lin, None] * (B[lin] - Br[lin])
    w = _energy_density(B2, (B * Br).sum(1), nu_lin, nl, Br2)
    energy = float(np.sum(w * sp_.vol))
    coenergy = float(np.sum(((H * B).sum(1) - w) * sp_.vol))
    info = {"unknowns": int(len(u)), "tree_edges": int(len(tree)), "dirichlet_edges": int(len(d_edges)),
            "residuals": hist, "seconds": time.perf_counter() - t0, "gauge": gauge, "current_load": bJ}
    return MagnetostaticSolution3D(mesh, a, B, H, energy, coenergy, it, float(res), info)


# ---------------------------------------------------------------------------------------------- eddy currents
@dataclass
class EddySolution3D:
    mesh: TetMesh
    a: np.ndarray                  # complex edge degrees of freedom of A* (Wb)
    B: np.ndarray                  # (M, 3) complex flux density phasor per tetrahedron (T, peak)
    omega: float
    sigma: np.ndarray              # (M,) conductivity
    loss: float                    # time-averaged Joule loss in the conductors (W)
    info: dict = field(default_factory=dict)
    space: EdgeSpace | None = None

    def B_at(self, points) -> np.ndarray:
        k = self.mesh.locate(points)
        out = np.full((len(k), 3), np.nan, complex)
        out[k >= 0] = self.B[k[k >= 0]]
        return out

    def J_at(self, points) -> np.ndarray:
        """Eddy current density phasor ``-j omega sigma A*`` at points (A/m^2, peak)."""
        k = self.mesh.locate(points)
        A = self.space.evaluate(self.a, points, cells=k)
        s = np.where(k >= 0, self.sigma[np.maximum(k, 0)], 0.0)
        return -1j * self.omega * s[:, None] * A


def solve_eddy(mesh: TetMesh, nu, sigma, omega: float, J, J_mask=None,
               dirichlet: tuple[np.ndarray, np.ndarray] | None = None, space: EdgeSpace | None = None,
               gauge: str = "tree", rtol: float = 1e-9) -> EddySolution3D:
    """Time-harmonic eddy currents with the modified vector potential ``A*`` (linear media).

    nu, sigma: per tetrahedron (or scalars); J: source current density phasor (M, 3) or callable, zero in the
    conductors. The tree gauge spans the non-conducting nodes only (conductor nodes are roots), so ``A*`` is
    unique in the conductors and carries the electric scalar potential there. ``gauge="tree"`` (default)
    factorises the tree-gauged system (sparse LU: small cases, up to some 1e4 unknowns in seconds);
    ``gauge="none"`` keeps every edge and solves with COCG and a Jacobi preconditioner, which is fast where it
    converges but stalls and diverges on stiff cases (measured on TEAM problem 7: residual 2e-2 after 300
    iterations, then growth), so it is opt-in."""
    t0 = time.perf_counter()
    sp_ = space or EdgeSpace(mesh)
    M = len(mesh.tets)
    nu = np.broadcast_to(np.asarray(nu, float), (M,))
    sigma = np.broadcast_to(np.asarray(sigma, float), (M,)).copy()
    if dirichlet is None:
        d_edges = mesh.boundary_edges
        d_vals = np.zeros(len(d_edges))
    else:
        d_edges, d_vals = (np.asarray(v) for v in dirichlet)
    d_nodes = np.unique(mesh.edges[d_edges])
    cond_nodes = mesh.nodes_in(sigma > 0)
    roots = np.union1d(d_nodes, cond_nodes)
    free_nodes = np.setdiff1d(np.arange(len(mesh.nodes)), roots)
    b = _project_source(sp_, sp_.current_load(J, J_mask), free_nodes).astype(complex)
    tree = tree_gauge(mesh, d_edges, roots) if gauge == "tree" else np.array([], np.int64)
    unknown = np.ones(sp_.n, bool)
    unknown[d_edges] = False
    unknown[tree] = False
    u = np.flatnonzero(unknown)
    K = sp_.stiffness(nu).astype(complex) + 1j * omega * sp_.mass(sigma)
    a = np.zeros(sp_.n, complex)
    a[d_edges] = d_vals
    rhs = (b - K @ a)[u]
    t1 = time.perf_counter()
    Kr = K[u][:, u].tocsr()
    iters = 0
    if gauge == "tree":
        a[u] = _factorise(Kr).solve(rhs)
    elif gauge == "none":
        a[u], iters = cocg(Kr, rhs, 1.0 / Kr.diagonal(), rtol=rtol)
    else:
        raise ValueError("gauge must be 'tree' or 'none'")
    t2 = time.perf_counter()
    B = np.einsum("ki,kid->kd", a[sp_.dofs], sp_.curls)
    Ms = sp_.mass(sigma)
    loss = 0.5 * omega ** 2 * float(np.real(np.conj(a) @ (Ms @ a)))
    info = {"unknowns": int(len(u)), "tree_edges": int(len(tree)), "assembly_s": t1 - t0, "solve_s": t2 - t1,
            "iterations": iters, "gauge": gauge}
    return EddySolution3D(mesh, a, B, omega, sigma, loss, info, sp_)
