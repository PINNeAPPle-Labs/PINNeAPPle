"""Transient conduction: Craig-Bampton condensation of the blocks, implicit Euler, and the fine reference.

``C dT/dt + K T = E p(t)``. Each block's cells are written as

    T_c = Psi t + Z p + Phi eta,

with ``Psi = -K_cc^-1 K_cp`` the static constraint modes (its port temperatures ``t``), ``Z p`` the static response to
its own power with the ports held (exact, so a quasi-static history is reproduced exactly), and ``Phi`` the ``k``
slowest fixed-interface modes (``K_cc Phi = C_cc Phi Lambda``, ``Phi^T C_cc Phi = I``) with amplitudes ``eta``
(Craig and Bampton, AIAA J. 6 (1968) 1313; applied here to the heat equation). A Galerkin projection with the test
space ``[Psi; I]`` and ``Phi`` gives, per block,

    [Psi^T C Psi   B^T] d/dt [t  ]   [S  0     ] [t  ]   [W p - Psi^T C Z dp/dt]
    [B             I  ]      [eta] + [0  Lambda] [eta] = [   - Phi^T C Z dp/dt ],     B = Phi^T C Psi,

and the port temperatures are restricted to the interface modes of :mod:`.substructure`. Blocks with the same
variant share all of these matrices. The global system (shared interface modes plus ``k`` amplitudes per block) is
integrated with implicit Euler at a fixed step, factorised once.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from . import fv
from .learned import _face_dofs
from .package import BASE, Package
from .substructure import FLUXES, block_model, interface_modes


@dataclass
class CBBlock:
    K: np.ndarray        # reduced stiffness (d + k)
    M: np.ndarray        # reduced capacity
    W: np.ndarray        # static load map on the ports (d, n2)
    G: np.ndarray        # load-rate map (d + k, n2): [Psi^T C Z; Phi^T C Z]
    Ot: np.ndarray       # observables from (t, eta) (n_obs, d + k)
    Op: np.ndarray       # observables from p (n_obs, n2)
    lam: np.ndarray


def cb_block(pkg_stack, variant, Phi_f: np.ndarray, k: int) -> CBBlock:
    """Craig-Bampton reduction of one block onto ``Phi_f`` interface modes and ``k`` fixed-interface modes."""
    bm = block_model(pkg_stack, variant)
    st = pkg_stack
    props = st.properties(variant)
    sysb = fv.assemble(props["k"], props["dz"], props["R"], st.dx, st.h_top, st.h_bottom, ports=True,
                       rho_c=props["rho_c"])
    nc = sysb.n_cells
    Kcc = sysb.K.tocsr()[:nc, :nc].toarray()
    c = sysb.C
    # fixed-interface modes, C-orthonormal: K Phi = C Phi Lambda
    s = 1 / np.sqrt(c)
    lam, V = np.linalg.eigh(s[:, None] * Kcc * s[None, :])
    Phi = s[:, None] * V[:, :k]
    lam = lam[:k]
    Psi = -bm.Y @ Phi_f                                  # (nc, d) static modes of the reduced ports
    CPsi = c[:, None] * Psi
    B = Phi.T @ CPsi                                     # (k, d)
    d = Psi.shape[1]
    K = np.zeros((d + k, d + k))
    K[:d, :d] = Phi_f.T @ bm.S @ Phi_f
    K[d:, d:] = np.diag(lam)
    M = np.zeros_like(K)
    M[:d, :d] = Psi.T @ CPsi
    M[:d, d:], M[d:, :d], M[d:, d:] = B.T, B, np.eye(k)
    CZ = c[:, None] * bm.Z
    G = np.vstack([Psi.T @ CZ, Phi.T @ CZ])
    Ot = np.hstack([bm.O_t @ Phi_f, np.asarray(bm.O @ Phi)])
    return CBBlock(K, M, Phi_f.T @ bm.W, G, Ot, bm.O_p, lam)


#: Laplace rates (1/s) of the interface snapshots of transient models: steady, and the lid (~0.1 s) to the die
#: (~1e-4 s) time scales of the reference stack
TRANSIENT_SHIFTS = (0.0, 10.0, 1e2, 1e3, 1e4)


class CBTransient:
    """Craig-Bampton model of a package (``m`` interface modes per face, ``k`` interior modes per block)."""

    def __init__(self, pkg: Package, m: int = 8, k: int = 8, shifts: tuple[float, ...] = TRANSIENT_SHIFTS):
        self.pkg, self.m, self.k, self.shifts = pkg, m, k, tuple(shifts)
        self.timings: dict[str, float] = {}

    def prepare(self, dt: float) -> CBTransient:
        t0 = time.perf_counter()
        pkg, st, m, k = self.pkg, self.pkg.stack, self.m, self.k
        base = block_model(st, BASE)
        basis = {o: interface_modes(base, o, m, shifts=self.shifts) for o in ("x", "y")}
        Phi_f = sla.block_diag(basis["x"], basis["x"], basis["y"], basis["y"])
        self.blocks = [cb_block(st, v, Phi_f, k) for v in pkg.variants]
        self.timings["blocks"] = time.perf_counter() - t0
        nbx, nby = pkg.shape
        N = nbx * nby
        fdofs, nfaces = _face_dofs(pkg.shape, m)
        n_face_dofs = nfaces * m
        eta = n_face_dofs + np.arange(N * k).reshape(N, k)
        self.dofs = np.hstack([fdofs, eta])
        self.n = n_face_dofs + N * k
        self.var = pkg.layout.ravel()
        t1 = time.perf_counter()
        D = self.dofs
        dd = D.shape[1]
        rows = np.repeat(D, dd, axis=1).ravel()
        cols = np.tile(D, (1, dd)).ravel()
        Kv = np.stack([self.blocks[v].K for v in self.var]).reshape(N, -1).ravel()
        Mv = np.stack([self.blocks[v].M for v in self.var]).reshape(N, -1).ravel()
        self.Kg = sp.csc_matrix((Kv, (rows, cols)), shape=(self.n, self.n))
        self.Mg = sp.csc_matrix((Mv, (rows, cols)), shape=(self.n, self.n))
        self.dt = dt
        self.lu = fv.factor(self.Mg / dt + self.Kg)
        self.timings["assemble_factor"] = time.perf_counter() - t1
        self.timings["prepare"] = time.perf_counter() - t0
        return self

    def _blockwise(self, attr: str, P: np.ndarray) -> np.ndarray:
        """Per block ``getattr(block, attr) @ P[b]``, grouped by variant."""
        out = None
        for v, blk in enumerate(self.blocks):
            sel = self.var == v
            A = getattr(blk, attr)
            r = P[sel] @ A.T
            if out is None:
                out = np.empty((len(P), A.shape[0]))
            out[sel] = r
        return out

    def run(self, powers: np.ndarray, x0: np.ndarray | None = None, every: int = 1) -> dict:
        """Integrate from rest (or ``x0``) over ``powers`` (n_steps, nx, ny) W per active cell, one per step end.
        Returns the observables per saved step: peak junction temperature and the die/TIM and lid heats."""
        t0 = time.perf_counter()
        pkg = self.pkg
        N = pkg.n_blocks
        n2 = pkg.stack.n ** 2
        d = 4 * self.m
        x = np.zeros(self.n) if x0 is None else x0
        p_prev = np.zeros((N, n2))
        hist = {"Tmax": [], "q_tim": [], "q_top": [], "Tj": []}
        for i, P in enumerate(powers):
            pb = pkg.block_power(P).reshape(N, n2)
            Wp = self._blockwise("W", pb)
            Gd = self._blockwise("G", (pb - p_prev) / self.dt)
            f = np.zeros((N, d + self.k))
            f[:, :d] = Wp
            f -= Gd
            rhs = self.Mg @ x / self.dt
            np.add.at(rhs, self.dofs.ravel(), f.ravel())
            x = self.lu.solve(rhs)
            p_prev = pb
            if (i + 1) % every == 0 or i == len(powers) - 1:
                xb = x[self.dofs]
                obs = None
                for v, blk in enumerate(self.blocks):
                    sel = self.var == v
                    o = xb[sel] @ blk.Ot.T + pb[sel] @ blk.Op.T
                    if obs is None:
                        obs = np.empty((N, o.shape[1]))
                    obs[sel] = o
                tj = obs[:, :n2]
                hist["Tmax"].append(tj.max())
                hist["Tj"].append(tj.reshape(*pkg.shape, n2).copy())
                hist["q_tim"].append(obs[:, n2 + FLUXES.index("q_tim")].sum())
                hist["q_top"].append(obs[:, n2 + FLUXES.index("q_top")].sum())
        out = {k: np.array(v) for k, v in hist.items()}
        out["seconds"] = time.perf_counter() - t0
        out["x"] = x
        return out


def reference_transient(pkg: Package, powers: np.ndarray, dt: float, every: int = 1) -> dict:
    """Fine-grid implicit Euler with the same step (``C / dt + K`` factorised once, or AMG-CG for big grids)."""
    from .substructure import reference_observables
    t0 = time.perf_counter()
    sysm = fv.assemble_package(pkg, transient=True)
    A = (sysm.K + sp.diags(sysm.C / dt)).tocsc()
    direct = A.shape[0] <= 400_000
    lu = fv.factor(A) if direct else None
    t_prep = time.perf_counter() - t0
    T = np.zeros(A.shape[0])
    hist = {"Tmax": [], "q_tim": [], "q_top": [], "Tj": []}
    for i, P in enumerate(powers):
        rhs = sysm.C / dt * T + fv.source_vector(pkg, P)
        T = lu.solve(rhs) if direct else fv.solve(A, rhs, method="amg")[0]
        if (i + 1) % every == 0 or i == len(powers) - 1:
            o = reference_observables(pkg, T.reshape(sysm.shape))
            hist["Tmax"].append(o["Tj"].max())
            hist["Tj"].append(o["Tj"].copy())
            hist["q_tim"].append(o["q_tim"].sum())
            hist["q_top"].append(o["q_top"].sum())
    out = {k: np.array(v) for k, v in hist.items()}
    out["seconds"] = time.perf_counter() - t0
    out["seconds_prepare"] = t_prep
    out["n_cells"] = sysm.n_cells
    return out
