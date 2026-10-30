"""Steady heat conduction on an immersed discretisation: -div(k grad T) = f in the part, with prescribed
temperatures (Nitsche), heat fluxes and convection on its surface.

Weak form on the active cells (Nitsche, Comput. Methods Appl. Mech. Eng. 1971; Hansbo & Hansbo 2002 for cut
elements):

    int k grad T . grad v - int_GD k (dT/dn v + dv/dn T) + int_GD gamma T v + int_GR h T v + s(T, v)
      = int f v - int_GD k dv/dn g + int_GD gamma g v + int_GR h T_inf v + int_GN q v

with ``s`` the small-cut stabilisation: ``"ghost"`` (ghost penalty on the faces of cut cells, Burman 2010; the Nitsche
penalty is then ``beta k / h``), ``"fcm"`` (the finite cell method's fictitious material of conductivity
``alpha k`` filling each cut cell, Parvizian, Duester & Rank 2007; the Nitsche penalty per cut cell from a local
eigenvalue problem, Embar, Dolbow & Harari 2010) or ``"none"``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy import linalg, sparse

from .assembly import (Discretization, SurfaceMeasures, condition_number, ghost_matrix, reference_cell, scatter,
                       scatter_vector, solve_linear)
from .bcs import assign, evaluate


@dataclass
class HeatSolution:
    disc: Discretization
    T: np.ndarray
    info: dict = field(default_factory=dict)
    balance: dict = field(default_factory=dict)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.disc.evaluate(self.T, X)

    def gradient(self, X: np.ndarray) -> np.ndarray:
        return self.disc.evaluate(self.T, X, gradient=True)

    def errors(self, exact, grad=None, fine: bool = True) -> dict:
        """L2 and H1-seminorm errors against an exact field (absolute and relative), integrated with the fine octree
        rule when it was kept (otherwise the moment-fitted one)."""
        return field_errors(self.disc, self.T, exact, grad, fine)

    def max(self) -> float:
        P = self.disc.volume_points(fine=self.disc.quad.fine is not None)
        return float(np.nanmax(self.disc.evaluate(self.T, self.disc.physical(P))))


def field_errors(disc: Discretization, u: np.ndarray, exact, grad=None, fine: bool = True, ncomp: int = 1,
                 chunk: int = 200_000) -> dict:
    fine = fine and disc.quad.fine is not None
    P = disc.volume_points(fine=fine)
    acc = np.zeros(4)
    for s in range(0, len(P), chunk):
        sl = slice(s, s + chunk)
        X = disc.grid.to_physical(P.cell[sl], P.xi[sl])
        w = P.w[sl]
        uh = disc.evaluate(u, X, ncomp=ncomp).reshape(len(X), -1)
        ue = np.asarray(exact(X), float).reshape(len(X), -1)
        acc[0] += np.sum(w * ((uh - ue) ** 2).sum(1))
        acc[1] += np.sum(w * (ue ** 2).sum(1))
        if grad is not None:
            gh = disc.evaluate(u, X, gradient=True, ncomp=ncomp).reshape(len(X), -1)
            ge = np.asarray(grad(X), float).reshape(len(X), -1)
            acc[2] += np.sum(w * ((gh - ge) ** 2).sum(1))
            acc[3] += np.sum(w * (ge ** 2).sum(1))
    out = {"rule": "fine" if fine else "fitted", "L2": float(np.sqrt(acc[0])), "L2_norm": float(np.sqrt(acc[1]))}
    out["L2_rel"] = out["L2"] / max(out["L2_norm"], 1e-300)
    if grad is not None:
        out["H1"], out["H1_norm"] = float(np.sqrt(acc[2])), float(np.sqrt(acc[3]))
        out["H1_rel"] = out["H1"] / max(out["H1_norm"], 1e-300)
    return out


def _cut_stiffness(disc: Discretization) -> tuple[np.ndarray, np.ndarray]:
    """grad.grad element matrices of the cut cells (unit coefficient) from the fitted weights."""
    cut, W = disc.cut_weights()
    _, G, _ = disc.fitted_reference
    nb = G.shape[1]
    T = np.einsum("qik,qjk->qij", G, G).reshape(len(G), nb * nb)
    return cut, (W @ T).reshape(len(cut), nb, nb)


def _cell_vectors(disc: Discretization, f: Any) -> np.ndarray:
    """Load vector of a volume source over inside cells (Gauss) and cut cells (fitted rule)."""
    n = disc.n_dofs
    out = np.zeros(n)
    if not callable(f) and float(f) == 0.0:
        return out
    ref = reference_cell(disc.grid)
    inside = disc.quad.inside_cells
    if len(inside):
        P = disc.quad.inside_points()
        fv = evaluate(f, disc.physical(P)).reshape(len(inside), -1)
        out += scatter_vector(disc.dofs.dofs_of(inside), (fv * ref["w"]) @ ref["N"], n)
    cut, W = disc.cut_weights()
    if len(cut):
        N, _, xi = disc.fitted_reference
        X = disc.grid.to_physical(np.repeat(cut, len(xi)), np.tile(xi, (len(cut), 1)))
        fv = evaluate(f, X).reshape(len(cut), -1)
        out += scatter_vector(disc.dofs.dofs_of(cut), (W * fv) @ N, n)
    return out


def assemble_heat(disc: Discretization, conductivity: float = 1.0, source: Any = 0.0, bcs: Sequence = (),
                  stabilization: str = "ghost", ghost: float | None = None, alpha: float = 1e-6,
                  beta: float | None = None) -> tuple[sparse.csr_matrix, np.ndarray, dict]:
    """Matrix, right-hand side and assembly details of the steady heat problem (see the module docstring)."""
    t0 = time.perf_counter()
    k = float(conductivity)
    n, p = disc.n_dofs, disc.order
    ref = reference_cell(disc.grid)
    h = disc.grid.size
    inside = disc.quad.inside_cells
    A = scatter(disc.dofs.dofs_of(inside), np.broadcast_to(k * ref["K"], (len(inside),) + ref["K"].shape), n)
    cut, Kc = _cut_stiffness(disc)
    Kc = k * Kc
    if stabilization == "fcm":
        Kc = (1 - alpha) * Kc + alpha * k * ref["K"][None]
    A = A + scatter(disc.dofs.dofs_of(cut), Kc, n)
    b = _cell_vectors(disc, source)
    info = {"stabilization": stabilization}
    if stabilization == "ghost":
        ghost = default_ghost(p) if ghost is None else ghost
        info["ghost"] = ghost
        A = A + ghost * k * ghost_matrix(disc)

    S = disc.quad.surface
    if len(S) and bcs:
        Xs = disc.physical(S)
        owner = assign(bcs, Xs, S.normal)
        d = disc.d
        for i, bc in enumerate(bcs):
            idx = np.nonzero(owner == i)[0]
            if not len(idx):
                continue
            Pi, Xi, ni = S.subset(idx), Xs[idx], S.normal[idx]
            if bc.kind == "neumann":
                sm = SurfaceMeasures(disc, Pi, {"q": evaluate(bc.flux, Xi, ni)})
                b += scatter_vector(disc.dofs.dofs_of(sm.cells), sm.vector("q", sm.N), n)
            elif bc.kind == "robin":
                hc = evaluate(bc.h, Xi, ni)
                sm = SurfaceMeasures(disc, Pi, {"h": hc, "hT": hc * evaluate(bc.T_inf, Xi, ni)})
                dofs = disc.dofs.dofs_of(sm.cells)
                A = A + scatter(dofs, sm.outer("h", sm.N, sm.N), n)
                b += scatter_vector(dofs, sm.vector("hT", sm.N), n)
            else:                                                       # Dirichlet by Nitsche
                g = evaluate(bc.value, Xi, ni)
                w = {"1": np.ones(len(idx)), "g": g}
                for a in range(d):
                    w[f"n{a}"], w[f"gn{a}"] = ni[:, a], g * ni[:, a]
                eig = stabilization != "ghost"
                if eig:
                    for a in range(d):
                        for c in range(a, d):
                            w[f"n{a}{c}"] = ni[:, a] * ni[:, c]
                sm = SurfaceMeasures(disc, Pi, w)
                N, G = sm.N, sm.G
                sym = sum(sm.outer(f"n{a}", N, G[:, :, a]) for a in range(d))
                sym = sym + sym.transpose(0, 2, 1)
                mass = sm.outer("1", N, N)
                gdn = sum(sm.vector(f"gn{a}", G[:, :, a]) for a in range(d))
                gN = sm.vector("g", N)
                cells = sm.cells
                if not eig:
                    bb = 10.0 * p ** 2 if beta is None else beta
                    gamma = np.full(len(cells), bb * k / h)
                else:
                    B = sum((1 if a == c else 2) * sm.outer(f"n{a}{c}", G[:, :, a], G[:, :, c])
                            for a in range(d) for c in range(a, d))
                    B = 0.5 * (B + B.transpose(0, 2, 1))
                    row = np.searchsorted(cut, cells)
                    lam = np.array([_max_eig(k * B[j], Kc[row[j]]) for j in range(len(cells))])
                    gamma = (2.0 if beta is None else beta) * 2.0 * lam
                    info["nitsche_gamma_max"] = float(gamma.max())
                    info["nitsche_gamma_median"] = float(np.median(gamma))
                info.setdefault("_gamma", {})[i] = (cells, gamma)
                dofs = disc.dofs.dofs_of(cells)
                A = A + scatter(dofs, -k * sym + gamma[:, None, None] * mass, n)
                b += scatter_vector(dofs, -k * gdn + gamma[:, None] * gN, n)
        info["boundary_points"] = {type(bc).__name__ + f"[{i}]": int((owner == i).sum()) for i, bc in enumerate(bcs)}
    info["assembly_s"] = time.perf_counter() - t0
    info["nnz"] = int(A.nnz)
    return sparse.csr_matrix(A), b, info


def default_ghost(order: int) -> float:
    """Ghost-penalty coefficient: measured on the 2-D manufactured case (verification.conditioning and the
    coefficient sweep of the report), 0.1 for Q1 and 0.01 for Q2 keep the error within 25 % of the finite cell
    method's at a condition number below 3e4, whatever the cut."""
    return 0.1 if order == 1 else 0.01


def _max_eig(B: np.ndarray, A: np.ndarray) -> float:
    """Largest generalised eigenvalue of B x = lambda A x (A regularised on its null space)."""
    d = np.abs(np.diag(A)).max()
    return float(linalg.eigh(B, A + 1e-12 * d * np.eye(len(A)), eigvals_only=True)[-1])


def solve_heat(disc: Discretization, conductivity: float = 1.0, source: Any = 0.0, bcs: Sequence = (),
               stabilization: str = "ghost", ghost: float | None = None, alpha: float = 1e-6,
               beta: float | None = None, condition: bool = False, solver: str = "auto") -> HeatSolution:
    """Assemble and solve; ``condition=True`` also estimates the condition number of the matrix (raw and Jacobi
    scaled). The result carries the timings and the heat balance (source vs boundary fluxes)."""
    A, b, info = assemble_heat(disc, conductivity, source, bcs, stabilization, ghost, alpha, beta)
    T, sinfo = solve_linear(A, b, solver)
    gamma = info.pop("_gamma", {})
    info.update({"solve_" + k: v for k, v in sinfo.items()})
    info["dofs"] = disc.n_dofs
    info["quadrature_s"] = disc.timings.get("quadrature_s", 0.0)
    if condition:
        t0 = time.perf_counter()
        info.update(condition_number(A))
        info["condition_s"] = time.perf_counter() - t0
    sol = HeatSolution(disc, T, info)
    sol.balance = heat_balance(sol, conductivity, source, bcs, gamma)
    return sol


def heat_balance(sol: HeatSolution, conductivity: float, source: Any, bcs: Sequence, gamma: dict | None = None
                 ) -> dict:
    """Energy check: the heat generated inside against the heat leaving through each boundary condition. On the
    Dirichlet part the flux is Nitsche's consistent flux -k dT/dn + gamma (T - g), which is conservative (testing
    the weak form with v = 1); without ``gamma`` it is the plain conductive flux of the discrete field."""
    disc = sol.disc
    gen = float(_cell_vectors(disc, source).sum())          # the rule the load vector used (partition of unity)
    S = disc.quad.surface
    Xs = disc.physical(S)
    owner = assign(bcs, Xs, S.normal) if bcs else np.full(len(S), -1)
    out = {"generated_W": gen}
    total = scale = 0.0
    for i, bc in enumerate(bcs):
        m = owner == i
        X, nrm, w = Xs[m], S.normal[m], S.w[m]
        if bc.kind == "neumann":
            q = -float(np.sum(w * evaluate(bc.flux, X, nrm)))
        elif bc.kind == "robin":
            q = float(np.sum(w * evaluate(bc.h, X, nrm) * (sol(X) - evaluate(bc.T_inf, X, nrm))))
        else:
            flux = -conductivity * np.einsum("ij,ij->i", sol.gradient(X), nrm)
            if gamma and i in gamma:
                cells, gm = gamma[i]
                g = np.zeros(disc.grid.n_cells)
                g[cells] = gm
                flux = flux + g[S.cell[m]] * (sol(X) - evaluate(bc.value, X, nrm))
            q = float(np.sum(w * flux))
        out[f"{type(bc).__name__}[{i}]_out_W"] = q
        total += q
        scale += abs(q)
    out["out_W"] = total
    out["imbalance"] = (total - gen) / max(abs(gen), scale, 1e-300)      # relative to the largest heat flow
    return out
