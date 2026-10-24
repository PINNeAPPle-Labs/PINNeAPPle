"""Prior field plus exception patches (track 1c): solve exactly only where a cheap prior does not hold.

1. **Prior.** A cheap field that is right in most of the package:
   ``"keff"`` -- the homogenised solution (:mod:`.homogenize`, one coarse cell per block laterally, the fine rows
   vertically), copied to every fine cell of its block and row; ``"column"`` -- every block as an isolated 1-D
   column with its own power (no lateral spreading). The Keff prior is the default: it keeps lateral spreading,
   which a 1-D column cannot (measured in the report), and costs one coarse solve.
2. **Where it does not hold.** The finite-volume residual of the prior on the fine grid, ``r = K T_prior - f``
   (one sparse product): heat created or destroyed by the prior in each cell, which is large where the prior
   misses a bump, a sub-pitch hotspot or an interface. Blocks are ranked by ``||r||`` over their cells (``select=
   "residual"``) or drawn at random (``"random"``, the baseline at the same budget).
3. **Compute exactly there.** Each selected block plus ``halo`` blocks around it is a patch; the fine problem is
   solved on the union of the patches with the prior as Dirichlet data on the cells around them.
4. **Recombine.** The patch solution replaces the prior in the selected blocks (``blend="core"``) or with a linear
   ramp across the halo (``"ramp"``, continuous). A final scalar correction ``alpha * phi`` (``phi``: the response to
   a uniform power map, one block with adiabatic sides) makes heat out equal heat in (``energy=True``).

This is a defect-correction / one-shot two-level Schwarz idea with a homogenised coarse level and residual-driven
patch selection; :func:`pareto` measures error against the fraction of cells solved exactly.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from . import fv
from . import homogenize as H
from .package import Package
from .substructure import reference_observables


def prior(pkg: Package, kind: str = "keff") -> tuple[np.ndarray, float]:
    """The prior on the fine cells ``(nx, ny, nz)`` and its cost (s)."""
    t0 = time.perf_counter()
    st = pkg.stack
    n = st.n
    nbx, nby = pkg.shape
    if kind == "keff":
        Th, _, _ = H.solve_homogenized(pkg, 1)                         # (nbx, nby, nz)
        T = np.repeat(np.repeat(Th, n, 0), n, 1)
    elif kind == "column":
        eff = H.effective_properties(st)
        dz = st.dz()
        nz = st.nz
        # one row-by-row conduction column per block: tridiagonal, the block's power on its active row
        kz = eff["kz"]
        A = st.pitch ** 2
        G = A / (dz[:-1] / (2 * kz[:-1]) + eff["R"] + dz[1:] / (2 * kz[1:]))
        gt = A / (dz[-1] / (2 * kz[-1]) + 1 / st.h_top)
        gb = A / (dz[0] / (2 * kz[0]) + 1 / st.h_bottom)
        K = np.diag(np.r_[G, 0] + np.r_[0, G]) - np.diag(G, 1) - np.diag(G, -1)
        K[0, 0] += gb
        K[-1, -1] += gt
        resp = np.linalg.solve(K, np.eye(nz)[:, st.active_k])          # column response to 1 W
        Pb = pkg.block_power().sum(-1)                                 # (nbx, nby)
        T = np.repeat(np.repeat(Pb[..., None] * resp[None, None, :], n, 0), n, 1)
    else:
        raise ValueError(f"unknown prior {kind!r}")
    return T, time.perf_counter() - t0


def block_indicator(pkg: Package, sysm, T: np.ndarray, f: np.ndarray) -> np.ndarray:
    """``||K T - f||`` over the cells of each block (W), ``(nbx, nby)``."""
    st = pkg.stack
    n = st.n
    nbx, nby = pkg.shape
    r = (sysm.K @ T.ravel() - f).reshape(nbx, n, nby, n, st.nz)
    return np.sqrt((r ** 2).sum(axis=(1, 3, 4)))


def correct(pkg: Package, T0: np.ndarray, blocks: np.ndarray, *, halo: int = 1, blend: str = "core",
            sysm=None, f=None) -> tuple[np.ndarray, dict]:
    """Solve the fine problem on the patches around ``blocks`` (boolean ``(nbx, nby)``) with ``T0`` outside."""
    t0 = time.perf_counter()
    st = pkg.stack
    n, nz = st.n, st.nz
    nbx, nby = pkg.shape
    sysm = sysm or fv.assemble_package(pkg)
    f = fv.source_vector(pkg) if f is None else f
    patch = blocks.copy()
    for _ in range(halo):
        p2 = patch.copy()
        p2[1:] |= patch[:-1]
        p2[:-1] |= patch[1:]
        p2[:, 1:] |= patch[:, :-1]
        p2[:, :-1] |= patch[:, 1:]
        patch = p2
    cellmask = np.repeat(np.repeat(patch, n, 0), n, 1)[..., None] & np.ones(nz, bool)
    U = np.nonzero(cellmask.ravel())[0]
    T = T0.ravel().copy()
    info = {"patch_blocks": int(patch.sum()), "solved_cells": int(len(U)), "cells": int(T.size)}
    if len(U):
        K = sysm.K.tocsr()
        Kuu = K[U][:, U]
        rhs = f[U] - K[U] @ T + Kuu @ T[U]
        Tu, _ = fv.solve(Kuu, rhs)
        Tp = T.copy()
        Tp[U] = Tu
        if blend == "core":
            core = np.repeat(np.repeat(blocks, n, 0), n, 1)[..., None] & np.ones(nz, bool)
            w = core.ravel().astype(float)
        elif blend == "ramp":
            # 1 in the selected blocks, falling linearly to 0 across the halo (block-distance ramp)
            from scipy.ndimage import distance_transform_cdt
            dist = distance_transform_cdt(~blocks, metric="chessboard").astype(float)
            wb = np.clip(1.0 - dist / (halo + 1.0), 0.0, 1.0)
            w = np.repeat(np.repeat(wb, n, 0), n, 1)[..., None].repeat(nz, -1).ravel()
        else:
            w = cellmask.ravel().astype(float)
        T = (1 - w) * T + w * Tp
    info["seconds"] = time.perf_counter() - t0
    return T.reshape(T0.shape), info


def energy_correct(pkg: Package, T: np.ndarray, sysm) -> tuple[np.ndarray, float]:
    """Add ``alpha * phi`` (``phi``: the response to a uniform map, i.e. one block with adiabatic sides, by symmetry)
    so that the Robin outflow equals the power."""
    from .package import BASE
    from .substructure import block_model
    st = pkg.stack
    n = st.n
    out = (sysm.g_top * T[..., -1]).sum() + (sysm.g_bottom * T[..., 0]).sum()
    bm = block_model(st, BASE)
    p = np.ones(n * n)
    t = np.linalg.solve(bm.S, bm.W @ p)
    phi_b = bm.recover(t, p)                                      # (n, n, nz) per 1 W per active cell
    phi = np.tile(phi_b, (pkg.shape[0], pkg.shape[1], 1))
    out_phi = (sysm.g_top * phi[..., -1]).sum() + (sysm.g_bottom * phi[..., 0]).sum()
    alpha = (pkg.power.sum() - out) / out_phi
    return T + alpha * phi, float(alpha)


def run(pkg: Package, *, budget: float = 0.1, select: str = "residual", halo: int = 1, blend: str = "core",
        prior_kind: str = "keff", energy: bool = True, seed: int = 0, sysm=None, f=None) -> dict:
    """The whole method at one budget (fraction of blocks selected); returns the field and its costs."""
    t0 = time.perf_counter()
    sysm = sysm or fv.assemble_package(pkg)
    f = fv.source_vector(pkg) if f is None else f
    t_asm = time.perf_counter() - t0
    T0, t_prior = prior(pkg, prior_kind)
    t1 = time.perf_counter()
    eta = block_indicator(pkg, sysm, T0, f)
    t_ind = time.perf_counter() - t1
    nsel = int(round(budget * eta.size))
    sel = np.zeros(eta.shape, bool)
    if nsel:
        if select == "residual":
            sel.ravel()[np.argsort(-eta.ravel())[:nsel]] = True
        else:
            sel.ravel()[np.random.default_rng(seed).choice(eta.size, nsel, replace=False)] = True
    T, info = correct(pkg, T0, sel, halo=halo, blend=blend, sysm=sysm, f=f)
    alpha = 0.0
    if energy:
        T, alpha = energy_correct(pkg, T, sysm)
    return {"T": T, "selected": sel, "eta": eta, "alpha": alpha, "info": info,
            "seconds": {"assemble": t_asm, "prior": t_prior, "indicator": t_ind, "patches": info["seconds"],
                        "total": time.perf_counter() - t0}}


def pareto(pkg: Package, budgets=(0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4), *, halos=(0, 1), selects=("residual",
           "random"), prior_kind: str = "keff", seed: int = 0) -> dict:
    """Error against the fraction of cells solved exactly, for each selection rule and halo; with the reference."""
    t0 = time.perf_counter()
    sysm = fv.assemble_package(pkg)
    f = fv.source_vector(pkg)
    Tref, info_ref = fv.solve(sysm.K, f)
    t_ref = time.perf_counter() - t0
    Tref = Tref.reshape(sysm.shape)
    ref = reference_observables(pkg, Tref)
    rows = []
    for sel in selects:
        for halo in halos:
            for b in budgets:
                r = run(pkg, budget=b, select=sel, halo=halo, prior_kind=prior_kind, seed=seed, sysm=sysm, f=f)
                o = reference_observables(pkg, r["T"])
                rows.append({"select": sel, "halo": halo, "budget": b,
                             "solved_fraction": r["info"]["solved_cells"] / r["info"]["cells"],
                             "err_Tmax": abs(float(o["Tj"].max() - ref["Tj"].max())),
                             "err_Tj_max": float(np.abs(o["Tj"] - ref["Tj"]).max()),
                             "err_q_tim": float(np.abs(o["q_tim"] - ref["q_tim"]).sum() / np.abs(ref["q_tim"]).sum()),
                             "err_q_ubump": float(np.abs(o["q_ubump"] - ref["q_ubump"]).sum()
                                                  / np.abs(ref["q_ubump"]).sum()),
                             "energy_balance": float((o["q_top"].sum() + o["q_bottom"].sum()) / pkg.power.sum() - 1),
                             "seconds": r["seconds"]["total"], "alpha": r["alpha"]})
                if sel == "random" and b == 0.0:
                    continue
    return {"blocks": pkg.n_blocks, "cells": int(Tref.size), "Tmax_ref": float(ref["Tj"].max()),
            "reference_s": t_ref, "reference_method": info_ref.method, "prior": prior_kind, "rows": rows}
