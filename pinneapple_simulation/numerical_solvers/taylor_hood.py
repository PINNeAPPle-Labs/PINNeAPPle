"""Steady incompressible Navier-Stokes on a body-fitted triangle / tetrahedron mesh: Taylor-Hood P2/P1 with
scikit-fem, Newton's method from the Stokes solution, sparse direct solves.

The textbook inf-sup-stable pair (Hood & Taylor 1973; Girault & Raviart 1986), with the gradient form of the viscous
term so that an outflow face left free is the do-nothing condition ``nu du/dn - p n = 0`` (as in the Schafer-Turek
benchmark). Forces on a no-slip body are the reactions of the momentum residual at its velocity unknowns, the
variational form that converges at the rate of the energy norm (Babuska & Miller 1984). Linear systems: MKL
PARDISO when pypardiso is installed, else SuperLU (:mod:`.sparse_direct`).

    from pinneapple_simulation.numerical_solvers.taylor_hood import TaylorHood, read_gmsh
    mesh = read_gmsh("mesh.msh")
    th = TaylorHood(mesh, nu=1e-3)
    sol = th.solve(dirichlet=[(on_inlet, inflow), (on_walls, 0.0), (on_body, 0.0)], tol=1e-8)
    sol.force(on_body), sol.pressure(points), sol.velocity(points)

``on_*`` are predicates on facet midpoints ``(d, n) -> bool (n,)``; values are constants or ``fn(X (d, n)) -> (d, n)``.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

__all__ = ["TaylorHood", "TaylorHoodSolution", "read_gmsh", "split_wedges"]


def split_wedges(wedges: np.ndarray) -> np.ndarray:
    """Three tetrahedra per prism (vertices 0-1-2 bottom, 3-4-5 top), each quadrilateral face cut along the diagonal
    from its smallest global vertex, so neighbouring prisms (and tetrahedra) stay conforming (Dompierre et al. 1999).
    """
    out = []
    for w in np.asarray(wedges, int):
        out.extend(_split_one(list(w)))
    return np.asarray(out, int).reshape(-1, 4)


def _split_one(w: list[int]) -> list[list[int]]:
    # rotate / reflect so that the smallest vertex is w[0] (the six symmetries of the prism keep it a prism)
    rots = [(0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3), (2, 0, 1, 5, 3, 4), (3, 5, 4, 0, 2, 1), (4, 3, 5, 1, 0, 2),
            (5, 4, 3, 2, 1, 0)]
    k = int(np.argmin(w))
    v = [w[i] for i in rots[k]]
    # the quad opposite v0 is v1 v2 v5 v4: its diagonal from its smallest vertex
    if min(v[1], v[5]) < min(v[2], v[4]):
        return [[v[0], v[1], v[2], v[5]], [v[0], v[1], v[5], v[4]], [v[0], v[4], v[5], v[3]]]
    return [[v[0], v[1], v[2], v[4]], [v[0], v[4], v[2], v[5]], [v[0], v[4], v[5], v[3]]]


def read_gmsh(path: str):
    """A gmsh .msh as a scikit-fem MeshTri / MeshTet (prism layers split into tetrahedra, orientation fixed)."""
    import meshio
    import skfem
    m = meshio.read(path)
    pts = np.asarray(m.points, float)
    tets = [c.data for c in m.cells if c.type == "tetra"]
    wedges = [c.data for c in m.cells if c.type == "wedge"]
    if tets or wedges:
        T = np.concatenate(tets + ([split_wedges(np.concatenate(wedges))] if wedges else []))
        used = np.unique(T)
        remap = np.full(len(pts), -1)
        remap[used] = np.arange(len(used))
        T = remap[T]
        P = pts[used]
        a, b, c, d = (P[T[:, i]] for i in range(4))
        neg = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) < 0
        T[neg] = T[neg][:, [0, 2, 1, 3]]
        return skfem.MeshTet(P.T.copy(), T.T.copy())
    tri = np.concatenate([c.data for c in m.cells if c.type == "triangle"])
    used = np.unique(tri)
    remap = np.full(len(pts), -1)
    remap[used] = np.arange(len(used))
    tri = remap[tri]
    P = pts[used, :2]
    a, b, c = (P[tri[:, i]] for i in range(3))
    neg = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]) < 0
    tri[neg] = tri[neg][:, [0, 2, 1]]
    return skfem.MeshTri(P.T.copy(), tri.T.copy())


def _tolerant_finder(mesh):
    """Element finder for probes: the candidate (among the 24 nearest centroids) whose barycentric coordinates are
    least negative, so points on the boundary (or within round-off of it) are found; scikit-fem's own finder
    raises on them."""
    from scipy.spatial import cKDTree
    P, T = mesh.p, mesh.t
    tree = cKDTree(P[:, T].mean(axis=1).T)
    k = min(24, T.shape[1])

    def element_finder(mapping=None):
        def finder(*x):
            X = np.stack(x, axis=1)
            cand = tree.query(X, k)[1].reshape(len(X), -1)
            V = P[:, T[:, cand]]                                              # (d, d + 1, n, k)
            A = V[:, 1:] - V[:, :1]                                           # (d, d, n, k)
            A = np.moveaxis(A, (0, 1), (-2, -1))                              # (n, k, d, d): columns = edges
            r = (X[:, None, :] - np.moveaxis(V[:, 0], 0, -1))[..., None]      # (n, k, d, 1)
            lam = np.linalg.solve(A, r)[..., 0]                               # (n, k, d)
            bary = np.concatenate([1 - lam.sum(-1, keepdims=True), lam], axis=-1)
            best = np.argmax(bary.min(-1), axis=1)
            return cand[np.arange(len(X)), best]
        return finder
    return element_finder


@dataclass
class TaylorHoodSolution:
    th: TaylorHood
    u: np.ndarray
    p: np.ndarray
    info: dict = field(default_factory=dict)

    def velocity(self, X: np.ndarray) -> np.ndarray:
        """Velocity at points (n, d)."""
        X = np.atleast_2d(np.asarray(X, float))
        out = np.zeros((len(X), self.th.d))
        idx = self.th.ub.split_indices()
        Ps = self.th.sb.probes(X.T)
        for k in range(self.th.d):
            out[:, k] = Ps @ self.u[idx[k]]
        return out

    def pressure(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        return self.th.pb.probes(X.T) @ self.p

    def force(self, on_body: Callable) -> np.ndarray:
        """Force on the no-slip body (density 1; per unit depth in 2-D): minus the momentum residual at its velocity
        unknowns, component by component."""
        th = self.th
        r = th.momentum_residual(self.u, self.p)
        dofs = th.ub.get_dofs(th.mesh.facets_satisfying(on_body, boundaries_only=True))
        body = np.zeros(th.ub.N, bool)
        body[dofs.all()] = True
        idx = th.ub.split_indices()
        F = np.zeros(th.d)
        for k in range(th.d):
            m = np.zeros(th.ub.N, bool)
            m[idx[k]] = True
            F[k] = -float(r[body & m].sum())
        return F

    def divergence_l2(self) -> float:
        import skfem
        from skfem.helpers import div

        @skfem.Functional
        def dd(w):
            return div(w["u"]) ** 2
        return float(np.sqrt(dd.assemble(self.th.ub, u=self.th.ub.interpolate(self.u))))


class TaylorHood:
    """P2/P1 Taylor-Hood on a scikit-fem triangle or tetrahedron mesh, kinematic viscosity ``nu``, density 1."""

    def __init__(self, mesh, nu: float, intorder: int | None = None):
        import skfem
        t0 = time.perf_counter()
        self.mesh, self.nu = mesh, float(nu)
        self.d = mesh.p.shape[0]
        object.__setattr__(mesh, "element_finder", _tolerant_finder(mesh))
        tri = self.d == 2
        e2 = skfem.ElementTriP2() if tri else skfem.ElementTetP2()
        e1 = skfem.ElementTriP1() if tri else skfem.ElementTetP1()
        io = intorder or 4
        self.ub = skfem.Basis(mesh, skfem.ElementVector(e2), intorder=io)
        self.sb = self.ub.with_element(e2)
        self.pb = self.ub.with_element(e1)
        from skfem.helpers import ddot, div, grad
        nu_ = self.nu

        @skfem.BilinearForm
        def visc(u, v, w):
            return nu_ * ddot(grad(u), grad(v))

        @skfem.BilinearForm
        def divf(u, q, w):
            return -div(u) * q

        self.K = visc.assemble(self.ub)
        self.B = divf.assemble(self.ub, self.pb)                      # (n_p, n_u)
        self.timings = {"assembly_s": time.perf_counter() - t0}

    @property
    def n_unknowns(self) -> int:
        return self.ub.N + self.pb.N

    def _convection(self, u: np.ndarray, jacobian: bool = True):
        import skfem
        from skfem.helpers import dot, grad, mul

        @skfem.BilinearForm
        def jac(du, v, w):
            return dot(mul(grad(du), w["w"]), v) + dot(mul(grad(w["w"]), du), v)

        @skfem.LinearForm
        def res(v, w):
            return dot(mul(grad(w["w"]), w["w"]), v)

        W = self.ub.interpolate(u)
        return (jac.assemble(self.ub, w=W) if jacobian else None), res.assemble(self.ub, w=W)

    def momentum_residual(self, u: np.ndarray, p: np.ndarray) -> np.ndarray:
        _, N = self._convection(u, jacobian=False)
        return self.K @ u + N + self.B.T @ p

    def _dirichlet(self, dirichlet: Sequence) -> tuple[np.ndarray, np.ndarray]:
        idx = self.ub.split_indices()
        comp = np.zeros(self.ub.N, int)
        for k, ii in enumerate(idx):
            comp[ii] = k
        X = self.ub.doflocs
        vals = np.zeros(self.ub.N)
        fixed = np.zeros(self.ub.N, bool)
        for where, value in dirichlet:
            dofs = self.ub.get_dofs(self.mesh.facets_satisfying(where, boundaries_only=True)).all()
            if callable(value):
                V = np.asarray(value(X[:, dofs]), float).reshape(self.d, -1)
                vals[dofs] = V[comp[dofs], np.arange(len(dofs))]
            else:
                v = np.broadcast_to(np.asarray(value, float), (self.d,))
                vals[dofs] = v[comp[dofs]]
            fixed[dofs] = True
        return np.nonzero(fixed)[0], vals

    def solve(self, dirichlet: Sequence, navier_stokes: bool = True, tol: float = 1e-8, max_iter: int = 30,
              pressure_fixed: bool = False, solver: str = "auto") -> TaylorHoodSolution:
        """``dirichlet``: [(facet predicate, value), ...] for the velocity; faces not listed are do-nothing outflow.
        ``pressure_fixed``: pin one pressure (enclosed flows); ``solver``: see :mod:`.sparse_direct`."""
        from .sparse_direct import solve as direct
        t0 = time.perf_counter()
        nu_, n_p = self.ub.N, self.pb.N
        D, vals = self._dirichlet(dirichlet)
        Dall = D.copy()
        if pressure_fixed:
            Dall = np.append(Dall, nu_)
        free = np.ones(nu_ + n_p, bool)
        free[Dall] = False
        Z = sparse.csr_matrix((n_p, n_p))

        def block(A):
            return sparse.bmat([[A, self.B.T], [self.B, Z]], format="csr")

        def lin_solve(A, rhs, xD):
            x = np.zeros(nu_ + n_p)
            x[Dall] = xD
            Af = A[free][:, free]
            x[free] = direct(Af, rhs[free] - A[free][:, Dall] @ xD, solver)
            return x

        xD = np.append(vals[D], 0.0) if pressure_fixed else vals[D]
        ts = time.perf_counter()
        x = lin_solve(block(self.K), np.zeros(nu_ + n_p), xD)
        t_solve = time.perf_counter() - ts
        hist = []
        if navier_stokes:
            r0 = None
            for _it in range(max_iter):
                J, N = self._convection(x[:nu_])
                R = np.concatenate([self.K @ x[:nu_] + N + self.B.T @ x[nu_:], self.B @ x[:nu_]])
                rn = float(np.linalg.norm(R[free]))
                if r0 is None:
                    r0 = rn
                    ax = np.abs(x)
                    floor = 1e-13 * float(np.linalg.norm((abs(self.K) @ ax[:nu_] + np.abs(N)
                                                          + abs(self.B.T) @ ax[nu_:])[free[:nu_]]))
                hist.append(rn)
                if len(hist) > 1 and rn <= max(tol * r0, floor):
                    break
                ts = time.perf_counter()
                dx = lin_solve(block(self.K + J), -R, np.zeros(len(Dall)))
                t_solve += time.perf_counter() - ts
                x = x + dx
            else:
                raise RuntimeError(f"Newton did not converge in {max_iter} iterations: {hist}")
        info = {"unknowns": nu_ + n_p, "velocity_dofs": nu_, "pressure_dofs": n_p, "elements": self.mesh.nelements,
                "newton_iterations": max(len(hist) - 1, 0), "residuals": hist, "solve_s": t_solve,
                "total_s": time.perf_counter() - t0 + self.timings["assembly_s"], **self.timings}
        return TaylorHoodSolution(self, x[:nu_], x[nu_:], info)
