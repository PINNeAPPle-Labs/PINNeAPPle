"""Linear elasticity on an immersed discretisation: -div sigma(u) = f in the part, sigma = lambda tr(eps) I + 2 mu eps,
with prescribed displacements (Nitsche, all or some components), tractions and pressures on its surface.

Weak form (Nitsche for the displacement set Gamma_D, components selected by the projection P):

    int sigma(u) : eps(v) - int_GD (P sigma(u) n) . v - int_GD (P sigma(v) n) . u + int_GD gamma P u . v + s(u, v)
      = int f . v + int_GN t . v - int_GD (P sigma(v) n) . g + int_GD gamma P g . v

``s`` is the same small-cut stabilisation as for heat (ghost penalty on every displacement component scaled by
2 mu + lambda, or the finite cell method's fictitious material). Unknowns are interleaved by node (u_x, u_y, u_z).
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import linalg, sparse

from .assembly import (
    Discretization,
    SurfaceMeasures,
    condition_number,
    ghost_matrix,
    reference_cell,
    scatter,
    scatter_vector,
    solve_linear,
)
from .bcs import assign, evaluate
from .heat import default_beta, default_ghost, field_errors


def lame(E: float, nu: float) -> tuple[float, float]:
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


def _B(G: np.ndarray) -> np.ndarray:
    """Strain-displacement tables (q, nvoigt, nb * d) from physical gradients (q, nb, d); Voigt order xx yy (zz)
    xy (yz zx) with engineering shear strains."""
    q, nb, d = G.shape
    pairs = [(0, 1)] if d == 2 else [(0, 1), (1, 2), (2, 0)]
    B = np.zeros((q, d + len(pairs), nb, d))
    for k in range(d):
        B[:, k, :, k] = G[:, :, k]
    for r, (i, j) in enumerate(pairs):
        B[:, d + r, :, i] = G[:, :, j]
        B[:, d + r, :, j] = G[:, :, i]
    return B.reshape(q, d + len(pairs), nb * d)


def elasticity_matrix(E: float, nu: float, d: int) -> np.ndarray:
    lam, mu = lame(E, nu)
    nv = 3 if d == 2 else 6
    D = np.zeros((nv, nv))
    D[:d, :d] = lam
    D[np.arange(d), np.arange(d)] += 2 * mu
    D[np.arange(d, nv), np.arange(d, nv)] = mu
    return D


def _traction_tables(G: np.ndarray, lam: float, mu: float) -> list[np.ndarray]:
    """Per direction k, S_k (q, d, nb*d): the traction sigma(N_a e_c) e_k of every basis function (component i of
    the vector in axis 1): sigma(N_a e_c)_ik = lam delta_ik d_c N_a + mu (delta_ck d_i N_a + delta_ic d_k N_a)."""
    q, nb, d = G.shape
    out = []
    for k in range(d):
        S = np.zeros((q, d, nb, d))
        for i in range(d):
            for c in range(d):
                coef = np.zeros((q, nb))
                if i == k:
                    coef += lam * G[:, :, c]
                if c == k:
                    coef += mu * G[:, :, i]
                if i == c:
                    coef += mu * G[:, :, k]
                S[:, i, :, c] = coef
        out.append(S.reshape(q, d, nb * d))
    return out


@dataclass
class ElasticSolution:
    disc: Discretization
    u: np.ndarray                      # (n_dofs, d)
    E: float
    nu: float
    info: dict = field(default_factory=dict)
    balance: dict = field(default_factory=dict)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.disc.evaluate(self.u, X, ncomp=self.disc.d)

    def gradient(self, X: np.ndarray) -> np.ndarray:
        return self.disc.evaluate(self.u, X, gradient=True, ncomp=self.disc.d)          # (n, d, d) du_i/dx_j

    def stress(self, X: np.ndarray) -> np.ndarray:
        """Cauchy stress tensors (n, d, d)."""
        Gu = self.gradient(X)
        lam, mu = lame(self.E, self.nu)
        eps = 0.5 * (Gu + Gu.transpose(0, 2, 1))
        tr = np.trace(eps, axis1=1, axis2=2)
        return lam * tr[:, None, None] * np.eye(self.disc.d) + 2 * mu * eps

    def von_mises(self, X: np.ndarray) -> np.ndarray:
        s = self.stress(X)
        if self.disc.d == 2:                                            # plane strain: sigma_zz = nu (sxx + syy)
            szz = self.nu * (s[:, 0, 0] + s[:, 1, 1])
            S = np.zeros((len(s), 3, 3))
            S[:, :2, :2] = s
            S[:, 2, 2] = szz
            s = S
        dev = s - np.trace(s, axis1=1, axis2=2)[:, None, None] / 3 * np.eye(3)
        return np.sqrt(1.5 * np.einsum("nij,nij->n", dev, dev))

    def errors(self, exact, grad=None, fine: bool = True) -> dict:
        return field_errors(self.disc, self.u, exact, grad, fine, ncomp=self.disc.d)

    def max_displacement(self) -> float:
        P = self.disc.volume_points(fine=self.disc.quad.fine is not None)
        X = self.disc.physical(P)
        return float(np.nanmax(np.linalg.norm(self(X), axis=1)))


def _traction(bc, X: np.ndarray, n: np.ndarray, w: np.ndarray, d: int) -> np.ndarray:
    """Traction at boundary points for Traction, Pressure and Force conditions."""
    if hasattr(bc, "total"):
        return np.broadcast_to(np.asarray(bc.total, float) / max(float(w.sum()), 1e-300), (len(X), d)).copy()
    if hasattr(bc, "p"):
        return -evaluate(bc.p, X, n)[:, None] * n
    return evaluate(bc.value, X, n, shape=(d,))


def _cut_elastic(disc: Discretization, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cut, W = disc.cut_weights()
    _, G, _ = disc.fitted_reference
    B = _B(G)
    nd = B.shape[2]
    T = np.einsum("qvi,vw,qwj->qij", B, D, B).reshape(len(G), nd * nd)
    out = np.zeros((len(cut), nd * nd))
    for s in range(0, len(cut), 2000):
        out[s:s + 2000] = W[s:s + 2000] @ T
    return cut, out.reshape(len(cut), nd, nd)


def _body_vector(disc: Discretization, f: Any) -> np.ndarray:
    d, n = disc.d, disc.n_dofs
    out = np.zeros(n * d)
    if f is None or (not callable(f) and not np.any(np.asarray(f, float))):
        return out
    ref = reference_cell(disc.grid)
    inside = disc.quad.inside_cells
    if len(inside):
        P = disc.quad.inside_points()
        fv = evaluate(f, disc.physical(P), shape=(d,)).reshape(len(inside), -1, d)
        out += scatter_vector(disc.dofs.dofs_of(inside), np.einsum("cqk,q,qb->cbk", fv, ref["w"], ref["N"]), n, d)
    cut, W = disc.cut_weights()
    if len(cut):
        N, _, xi = disc.fitted_reference
        X = disc.grid.to_physical(np.repeat(cut, len(xi)), np.tile(xi, (len(cut), 1)))
        fv = evaluate(f, X, shape=(d,)).reshape(len(cut), -1, d)
        out += scatter_vector(disc.dofs.dofs_of(cut), np.einsum("cqk,cq,qb->cbk", fv, W, N), n, d)
    return out


def assemble_elasticity(disc: Discretization, E: float, nu: float, body_force: Any = None, bcs: Sequence = (),
                        stabilization: str = "ghost", ghost: float | None = None, alpha: float = 1e-6,
                        beta: float | None = None):
    t0 = time.perf_counter()
    d, n, p = disc.d, disc.n_dofs, disc.order
    lam, mu = lame(E, nu)
    D = elasticity_matrix(E, nu, d)
    h = disc.grid.size
    ref = reference_cell(disc.grid)
    Bref = _B(ref["G"])
    Kref = np.einsum("q,qvi,vw,qwj->ij", ref["w"], Bref, D, Bref)
    inside = disc.quad.inside_cells
    A = scatter(disc.dofs.dofs_of(inside), np.broadcast_to(Kref, (len(inside),) + Kref.shape), n, d)
    cut, Kc = _cut_elastic(disc, D)
    if stabilization == "fcm":
        Kc = (1 - alpha) * Kc + alpha * Kref[None]
    A = A + scatter(disc.dofs.dofs_of(cut), Kc, n, d)
    b = _body_vector(disc, body_force)
    info = {"stabilization": stabilization}
    scale = 2 * mu + lam
    if stabilization == "ghost":
        ghost = default_ghost(p) if ghost is None else ghost
        info["ghost"] = ghost
        A = A + ghost * scale * ghost_matrix(disc, d)
    S = disc.quad.surface
    if len(S) and bcs:
        Xs = disc.physical(S)
        owner = assign(bcs, Xs, S.normal)
        for i, bc in enumerate(bcs):
            idx = np.nonzero(owner == i)[0]
            if not len(idx):
                continue
            Pi, Xi, ni = S.subset(idx), Xs[idx], S.normal[idx]
            if bc.kind == "neumann":
                t = _traction(bc, Xi, ni, Pi.w, d)
                sm = SurfaceMeasures(disc, Pi, {f"t{c}": t[:, c] for c in range(d)})
                vec = np.stack([sm.vector(f"t{c}", sm.N) for c in range(d)], axis=-1)
                b += scatter_vector(disc.dofs.dofs_of(sm.cells), vec, n, d)
                continue
            comps = np.zeros(d, bool)
            comps[list(range(d)) if bc.components is None else list(bc.components)] = True
            g = evaluate(bc.value, Xi, ni, shape=(d,))
            eig = stabilization != "ghost"
            w = {"1": np.ones(len(idx))}
            for k in range(d):
                w[f"n{k}"] = ni[:, k]
                for c in range(d):
                    w[f"g{c}"] = g[:, c]
                    w[f"g{c}n{k}"] = g[:, c] * ni[:, k]
                    if eig and c >= k:
                        w[f"n{k}{c}"] = ni[:, k] * ni[:, c]
            sm = SurfaceMeasures(disc, Pi, w)
            N, G = sm.N, sm.G
            nq, nb = N.shape
            nd = nb * d
            Sk = _traction_tables(G, lam, mu)                            # traction of the trial functions
            rows = np.repeat(comps[None, :], nb, axis=0).ravel()        # test (b, i) with i in the projection
            cons = 0.0
            for k in range(d):
                Tk = (N[:, :, None, None] * Sk[k][:, None, :, :]).reshape(nq, nd, nd)   # N_b sigma(u)_ik
                cons = cons + (sm.W[f"n{k}"] @ Tk.reshape(nq, -1)).reshape(-1, nd, nd)
            cons = cons * rows[None, :, None]
            sym = cons + cons.transpose(0, 2, 1)
            mass = sm.outer("1", N, N)
            Mfull = np.einsum("cab,ij->caibj", mass, np.diag(comps.astype(float))).reshape(-1, nd, nd)
            gs = 0.0                                                    # int (P sigma(v) n) . g
            for k in range(d):
                for c in range(d):
                    if comps[c]:
                        gs = gs + sm.W[f"g{c}n{k}"] @ Sk[k][:, c, :]
            gN = np.zeros((len(sm.cells), nb, d))
            for c in range(d):
                if comps[c]:
                    gN[:, :, c] = sm.vector(f"g{c}", N)
            gN = gN.reshape(len(sm.cells), -1)
            cells = sm.cells
            if not eig:
                bb = default_beta(p) if beta is None else beta
                gamma = np.full(len(cells), bb * scale / h)
            else:                       # gamma > 2 sup int_Gc |P sigma(v) n|^2 / a_c(v, v), per cut cell
                Pm = np.diag(comps.astype(float))
                Bc = 0.0
                for k in range(d):
                    for m in range(d):
                        key = f"n{min(k, m)}{max(k, m)}"
                        tab = np.einsum("qia,ij,qjb->qab", Sk[k], Pm, Sk[m]).reshape(nq, -1)
                        Bc = Bc + (sm.W[key] @ tab).reshape(-1, nd, nd)
                row = np.searchsorted(cut, cells)
                lamx = np.array([_max_eig(0.5 * (Bc[j] + Bc[j].T), Kc[row[j]]) for j in range(len(cells))])
                gamma = (2.0 if beta is None else beta) * 2.0 * lamx
                info["nitsche_gamma_max"] = float(gamma.max())
            info.setdefault("_gamma", {})[i] = (cells, gamma, comps)
            dofs = disc.dofs.dofs_of(cells)
            A = A + scatter(dofs, -sym + gamma[:, None, None] * Mfull, n, d)
            b += scatter_vector(dofs, (-gs + gamma[:, None] * gN).reshape(len(cells), nb, d), n, d)
        info["boundary_points"] = {type(bc).__name__ + f"[{i}]": int((owner == i).sum()) for i, bc in enumerate(bcs)}
    info["assembly_s"] = time.perf_counter() - t0
    info["nnz"] = int(A.nnz)
    return sparse.csr_matrix(A), b, info


def _max_eig(B: np.ndarray, A: np.ndarray) -> float:
    dg = np.abs(np.diag(A)).max()
    return float(linalg.eigh(B, A + 1e-12 * dg * np.eye(len(A)), eigvals_only=True)[-1])


def rigid_body_modes(disc: Discretization) -> np.ndarray:
    """Near-null space for smoothed-aggregation AMG: translations and rotations at the dof nodes."""
    X = disc.dofs.coords()
    X = X - X.mean(0)
    d, n = disc.d, len(X)
    modes = []
    for k in range(d):
        m = np.zeros((n, d))
        m[:, k] = 1
        modes.append(m)
    rots = [(0, 1)] if d == 2 else [(0, 1), (1, 2), (2, 0)]
    for i, j in rots:
        m = np.zeros((n, d))
        m[:, i], m[:, j] = -X[:, j], X[:, i]
        modes.append(m)
    return np.stack([m.ravel() for m in modes], 1)


def solve_elasticity(disc: Discretization, E: float, nu: float, body_force: Any = None, bcs: Sequence = (),
                     stabilization: str = "ghost", ghost: float | None = None, alpha: float = 1e-6,
                     beta: float | None = None, condition: bool = False, solver: str = "auto") -> ElasticSolution:
    A, b, info = assemble_elasticity(disc, E, nu, body_force, bcs, stabilization, ghost, alpha, beta)
    x, sinfo = solve_linear(A, b, solver, near_null=rigid_body_modes(disc))
    gamma = info.pop("_gamma", {})
    info.update({"solve_" + k: v for k, v in sinfo.items()}, dofs=A.shape[0],
                quadrature_s=disc.timings.get("quadrature_s", 0.0))
    if condition:
        info.update(condition_number(A))
    sol = ElasticSolution(disc, x.reshape(-1, disc.d), E, nu, info)
    sol.balance = force_balance(sol, body_force, bcs, gamma)
    return sol


def force_balance(sol: ElasticSolution, body_force: Any, bcs: Sequence, gamma: dict) -> dict:
    """Equilibrium check: applied forces (body + tractions) against the reactions on the displacement boundary,
    from Nitsche's consistent flux sigma(u) n - gamma P (u - g) (exact for the discrete solution, testing the weak
    form with constant v). The imbalance is relative to the sum of the load magnitudes, since a pressure on a closed
    surface has no net force."""
    disc = sol.disc
    d = disc.d
    applied = _body_vector(disc, body_force).reshape(-1, d).sum(0)
    S = disc.quad.surface
    Xs = disc.physical(S)
    owner = assign(bcs, Xs, S.normal) if bcs else np.full(len(S), -1)
    reaction = np.zeros(d)
    magnitude = float(np.abs(_body_vector(disc, body_force)).sum())     # sum of |load| (net loads can cancel)
    for i, bc in enumerate(bcs):
        m = owner == i
        X, nrm, w = Xs[m], S.normal[m], S.w[m]
        if bc.kind == "neumann":
            t = _traction(bc, X, nrm, w, d)
            applied += (w[:, None] * t).sum(0)
            magnitude += float((w[:, None] * np.abs(t)).sum())
        elif i in gamma:
            cells, gm, comps = gamma[i]
            gg = np.zeros(disc.grid.n_cells)
            gg[cells] = gm
            tr = np.einsum("nij,nj->ni", sol.stress(X), nrm) * comps
            jump = (sol(X) - evaluate(bc.value, X, nrm, shape=(d,))) * comps
            r = w[:, None] * (tr - gg[S.cell[m]][:, None] * jump)
            reaction += r.sum(0)
            magnitude += float(np.abs(r).sum())
    scale = max(magnitude, 1e-300)
    return {"applied_N": applied.tolist(), "reaction_N": reaction.tolist(),
            "imbalance": float(np.abs(applied + reaction).max() / scale)}
