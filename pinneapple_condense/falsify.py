"""Falsification studies of the condensation method (``docs/research/condensation_falsification.md``): each function
is one pre-registered test, returns plain JSON-able numbers and is meant to run in a fresh process
(:func:`.benchmark.isolated` with ``module="pinneapple_condense.falsify"``) so that its peak memory is its own.

Thermal hypotheses T1-T10 stress the existing method; the CFD and meshless hypotheses live in :mod:`.flow`,
:mod:`.algebraic` and their studies here.
"""
from __future__ import annotations

import time

import numpy as np

from . import fv
from .hierarchy import Hierarchy
from .package import uniform_package
from .substructure import Condensed, reference_observables


# ---------------------------------------------------------------------------------------------------- T1 / T1b
def t1_reference(nb: int, seed: int = 2, m: int = 12, rtol: float = 1e-10, fine: bool = True,
                 maxiter: int = 2000) -> dict:
    """Reference peak junction temperature of an ``nb x nb`` package: the fine AMG solve (``fine``) and the exact
    condensation (PCG on the unreduced interfaces, preconditioned by the m-mode model) with its convergence trail
    (T1b: iterations and time to 1 mK)."""
    pkg = uniform_package((nb, nb), seed=seed)
    out = {"blocks": nb * nb, "cells": pkg.n_cells}
    if fine:
        t0 = time.perf_counter()
        T, info = fv.solve_package(pkg)
        out["fine"] = {"Tmax": float(reference_observables(pkg, T)["Tj"].max()), "seconds": time.perf_counter() - t0,
                       "method": info.method, "iterations": info.iterations}
        del T
    c = Condensed(pkg, m).prepare()
    st = c.solve()
    t_red = float(c.observables(st)["Tj"].max())
    trail, done, it = [], 0, 1
    U = None
    while done < maxiter:
        r = c.correct(st, rtol=0.0, maxiter=it)            # restarts from the reduced solution: trail of fresh runs
        tm = float(c.observables_fine(r["U"], r["pb"])["Tj"].max())
        trail.append({"iterations": r["iterations"], "Tmax": tm, "rel_residual": r["history"][-1],
                      "seconds": r["seconds"]})
        done, U = r["iterations"], r
        if r["history"][-1] <= rtol or r["iterations"] < it:
            break
        it *= 2
    r = c.correct(st, rtol=rtol, maxiter=maxiter)
    t_exact = float(c.observables_fine(r["U"], r["pb"])["Tj"].max())
    for row in trail:
        row["err_Tmax"] = abs(row["Tmax"] - t_exact)
    to_1mK = next((row for row in trail if row["err_Tmax"] <= 1e-3), None)
    out["exact"] = {"Tmax": t_exact, "iterations": r["iterations"], "seconds": r["seconds"],
                    "rel_residual": r["history"][-1], "prepare_s": c.timings["prepare"], "m": m,
                    "Tmax_reduced": t_red, "trail": trail,
                    "iterations_to_1mK": to_1mK["iterations"] if to_1mK else None,
                    "seconds_to_1mK": to_1mK["seconds"] if to_1mK else None}
    del U
    return out


def t1_point(nb: int, m: int, seed: int = 2) -> dict:
    """The memoised hierarchy with ``m`` interface modes: peak junction temperature, build and solve time."""
    pkg = uniform_package((nb, nb), seed=seed)
    h = Hierarchy(pkg, m).prepare()
    sol = h.solve()
    o = h.observables(sol)
    return {"blocks": nb * nb, "m": m, "Tmax": float(o["Tj"].max()), "prepare_s": h.timings["prepare"],
            "solve_s": sol["seconds"],
            "energy_balance": float((o["q_top"].sum() + o["q_bottom"].sum()) / pkg.power.sum() - 1)}


# ---------------------------------------------------------------------------------------------------- F0 validation
GHIA_RE100 = {   # u on the vertical centreline of the lid-driven cavity, Re = 100 (Ghia, Ghia and Shin 1982, table I)
    "y": [0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703,
          0.0625, 0.0547],
    "u": [0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641, -0.20581, -0.21090, -0.15662, -0.10150,
          -0.06434, -0.04775, -0.04192, -0.03717]}


def f0_validation(poiseuille=(16, 32, 64), cavity=(32, 64, 128), nusselt=(16, 32, 64)) -> dict:
    """The flow solver against plane Poiseuille flow, the Re = 100 cavity and the one-sided-flux parallel-plate
    Nusselt number (5.385)."""
    from . import flow as FL
    out = {"poiseuille": [], "cavity": [], "nusselt": []}
    for ny in poiseuille:
        h, nx = 1.0 / ny, 4 * ny
        g = FL.Grid(nx, ny, h)
        yc = (np.arange(ny) + 0.5) * h
        inlet = 6 * yc * (1 - yc)
        A, b = FL.assemble_oseen(g, 0.1, inlet)
        u, v, p = FL.split(g, FL.solve(A, b))
        dpdx = (p[3 * nx // 4, ny // 2] - p[nx // 4, ny // 2]) / (nx / 2 * h)
        out["poiseuille"].append({"ny": ny, "u_max_rel_err": float(np.abs(u[nx // 2] - inlet).max() / 1.5),
                                  "dpdx": float(dpdx), "dpdx_exact": -1.2,
                                  "dpdx_rel_err": float(abs(dpdx + 1.2) / 1.2), "v_max": float(np.abs(v).max())})
    gy, gu = np.array(GHIA_RE100["y"]), np.array(GHIA_RE100["u"])
    for n in cavity:
        g = FL.Grid(n, n, 1.0 / n, closed=True, lid=1.0)
        t0 = time.perf_counter()
        r = FL.picard(g, nu=0.01, tol=1e-7, relax=0.8, maxiter=300)
        u, _, _ = FL.split(g, r["x"])
        uc = u[n // 2]
        ui = np.interp(gy, (np.arange(n) + 0.5) / n, uc)
        out["cavity"].append({"n": n, "picard_iterations": r["iterations"], "max_dev_vs_ghia": float(np.abs(ui - gu).max()),
                              "u_min": float(uc.min()), "seconds": time.perf_counter() - t0})
    for ny in nusselt:
        h, nx = 1.0 / ny, 12 * ny
        g = FL.Grid(nx, ny, h)
        yc = (np.arange(ny) + 0.5) * h
        A, b = FL.assemble_oseen(g, 0.1, 6 * yc * (1 - yc))
        u, v, _ = FL.split(g, FL.solve(A, b))
        alpha = 0.1                                   # Pe = U D_h / alpha = 20
        As, bs = FL.assemble_scalar(g, u, v, np.full((nx, ny), alpha), np.ones((nx, ny)), np.zeros((nx, ny)))
        bs = bs.reshape(nx, ny)
        bs[:, 0] += h                                 # unit heat flux through the bottom wall
        T = FL.solve(As, bs.ravel()).reshape(nx, ny)
        i = int(0.8 * nx)
        uc = 0.5 * (u[i] + u[i + 1])
        Tb = float((uc * T[i]).sum() / uc.sum())
        Tw = float(T[i, 0] + 0.5 * h / alpha)
        nu_ = 2.0 / (alpha * (Tw - Tb))
        out["nusselt"].append({"ny": ny, "Nu": nu_, "Nu_exact": 5.385, "rel_err": abs(nu_ - 5.385) / 5.385})
    return out


# ---------------------------------------------------------------------------------------------------- CFD cases
def _pins(n_pins, cells=16, pin=8, inlet_blocks=0, outlet_blocks=0):
    from . import flow as FL
    return FL.pin_array(n_pins, cells, pin, 1.0, inlet_cells=inlet_blocks * cells, outlet_cells=outlet_blocks * cells)


def _stokes_system(n_pins, cells=16, pin=8, nu=1.0, inlet_profile=None, adv=None, inlet_blocks=1, outlet_blocks=1):
    """Pin array in a channel strip with free-slip sides (a strip of a periodic array), uniform inlet; lengths in
    unit cells of side 1 (h = 1 / cells)."""
    from . import flow as FL
    solid = _pins(n_pins, cells, pin, inlet_blocks, outlet_blocks)
    nx, ny = solid.shape
    g = FL.Grid(nx, ny, 1.0 / cells, solid, walls="slip")
    inlet = np.ones(ny) if inlet_profile is None else inlet_profile
    A, b = FL.assemble_oseen(g, nu, inlet, adv)
    return g, A, b


def _primal_pressure(g, cells):
    """One pressure unknown per block: the fluid cell nearest the block centre."""
    prim = np.zeros(g.n, bool)
    for bi in range(g.nx // cells):
        for bj in range(g.ny // cells):
            sub = g.ip[bi * cells:(bi + 1) * cells, bj * cells:(bj + 1) * cells]
            ii, jj = np.nonzero(sub >= 0)
            if len(ii):
                c = np.argmin((ii - cells / 2) ** 2 + (jj - cells / 2) ** 2)
                prim[sub[ii[c], jj[c]]] = True
    return prim


def _condense(A, X, F, cells_h, primal=None):
    from . import algebraic as AL
    P = AL.partition(A, X, (cells_h, cells_h), fieldid=F, primal=primal)
    return AL.Condensation(A, P).prepare(), P


def _timed_direct(A, b):
    import scipy.sparse.linalg as spla
    t0 = time.perf_counter()
    x = spla.spsolve(A.tocsc(), b)
    return x, time.perf_counter() - t0


def _reduction_sweep(cond, P, A, b, x_ref, snap_problem, metric, ms):
    """Smallest m whose reduced solution meets 1 % on ``metric(x, x_ref)``; per m: error, reduced size."""
    from . import algebraic as AL
    A2, P2, snaps = snap_problem
    fc2 = AL.faces(P2, A2)
    fc = AL.faces(P, A)
    face_len = np.mean([len(f["idx"]) for f in fc["faces"]]) if fc["faces"] else 0.0
    rows = []
    for m in ms:
        R = AL.Reduced(cond, fc, AL.face_modes(snaps, fc2, m)).prepare()
        x, info = R.solve(b)
        rows.append({"m": m, "error": float(metric(x, x_ref)), "reduced_dofs": R.timings["reduced_dofs"],
                     "separator": R.timings["separator"], "unseen_faces": R.timings["unseen_faces"],
                     "compression": float(face_len / m), "solve_s": info["seconds"]})
    ok = [r for r in rows if r["error"] <= 0.01]
    return {"rows": rows, "mean_face_unknowns": float(face_len), "m_for_1pct": ok[0]["m"] if ok else None,
            "compression_at_1pct": ok[0]["compression"] if ok else None, "full_faces": int(len(fc["full"]))}


def f2_stokes(n_pins: int = 16, cells: int = 16, pin: int = 8, ms=(1, 2, 4, 8, 16), n_snap: int = 24,
              seed: int = 0) -> dict:
    """F2 / F2r: Stokes through an ``n_pins x n_pins`` pin array; exact algebraic condensation against the global
    direct solve, then reduced faces from snapshots of a 4 x 4 array with random smooth inlet profiles."""
    from . import algebraic as AL
    from . import flow as FL
    g, A, b = _stokes_system((n_pins, n_pins), cells, pin)
    X, F = g.coords()
    x_ref, t_direct = _timed_direct(A, b)
    t0 = time.perf_counter()
    cond, P = _condense(A, X, F, 1.0, _primal_pressure(g, cells))
    x, info = cond.solve(b)
    t_cond = time.perf_counter() - t0
    u, v, p = FL.split(g, x_ref)

    def dp(xx):
        pp = FL.split(g, xx)[2]
        return float(np.nanmean(pp[0]))               # pressure at the first column (outlet ghost is 0)

    rng = np.random.default_rng(seed)
    g2, A2, _ = _stokes_system((4, 4), cells, pin)
    snaps = []
    for _ in range(n_snap):
        y = (np.arange(g2.ny) + 0.5) / g2.ny
        prof = 1 + sum(rng.normal(0, 0.3) * np.sin((k + 1) * np.pi * y) for k in range(4))
        _, A2k, b2k = _stokes_system((4, 4), cells, pin, inlet_profile=prof)
        snaps.append(FL.solve(A2k, b2k))
    X2, F2 = g2.coords()
    P2 = AL.partition(A2, X2, (1.0, 1.0), fieldid=F2, primal=_primal_pressure(g2, cells))
    red = _reduction_sweep(cond, P, A, b, x_ref, (A2, P2, snaps),
                           lambda xx, xr: abs(dp(xx) - dp(xr)) / abs(dp(xr)), ms)
    nvel = g.iv.max() + 1
    return {"pins": n_pins * n_pins, "unknowns": int(g.n), "direct_s": t_direct, "condensed_s": t_cond,
            "speedup": t_direct / t_cond, "condensation": cond.timings, "solve": info,
            "exact_rel_err": float(np.abs(x - x_ref).max() / np.abs(x_ref).max()),
            "exact_rel_err_velocity": float(np.abs(x[:nvel] - x_ref[:nvel]).max() / np.abs(x_ref[:nvel]).max()),
            "pressure_drop": dp(x_ref), "pressure_drop_condensed": dp(x), "reduced": red,
            "metric_note": "flow rate is fixed by the velocity inlet; the pressure drop at that flow rate is reported"}


def _scalar_case(n_pins, cells, pin, Pe_b, u, v, g, q_pin=None, T_in=0.0):
    """Advection-diffusion of a scalar through the pin array: uniform diffusivity ``alpha = U L_b / Pe_b`` (U = 1,
    L_b = 1), source ``q_pin`` (default 1) in each pin."""
    from . import flow as FL
    alpha = 1.0 / Pe_b
    k = np.full((g.nx, g.ny), alpha)
    rhoc = np.ones((g.nx, g.ny))
    q = np.where(g.solid, 1.0, 0.0) if q_pin is None else q_pin
    return FL.assemble_scalar(g, u, v, k, rhoc, q, T_in=T_in)


def f1_advection(n_pins: int = 8, cells: int = 16, pin: int = 8, pes=(0.1, 1, 10, 100, 1000), ms=(1, 2, 4, 8, 16),
                 n_snap: int = 24, seed: int = 0) -> dict:
    """F1 / F1r: steady advection-diffusion through the pin array (velocity: the Stokes solution), block Peclet
    sweep; exact condensation of the non-symmetric operator, then reduced faces from a 4 x 4 snapshot problem with
    random pin sources and inlet profiles at the same Peclet number."""
    from . import algebraic as AL
    from . import flow as FL
    g, A, b = _stokes_system((n_pins, n_pins), cells, pin)
    u, v, _ = FL.split(g, FL.solve(A, b))
    g2, A2f, b2f = _stokes_system((4, 4), cells, pin)
    u2, v2, _ = FL.split(g2, FL.solve(A2f, b2f))
    Xs, Xs2 = FL.scalar_coords(g), FL.scalar_coords(g2)
    rng = np.random.default_rng(seed)
    rows = []
    for pe in pes:
        As, bs = _scalar_case(n_pins, cells, pin, pe, u, v, g)
        x_ref, t_direct = _timed_direct(As, bs)
        t0 = time.perf_counter()
        cond, P = _condense(As, Xs, np.zeros(len(Xs), np.int64), 1.0)
        x, _ = cond.solve(bs)
        t_cond = time.perf_counter() - t0
        snaps = []
        for _ in range(n_snap):
            qp = np.where(g2.solid, rng.uniform(0, 2, g2.solid.shape), 0.0)
            A2, b2 = _scalar_case(4, cells, pin, pe, u2, v2, g2, q_pin=qp, T_in=rng.normal())
            snaps.append(FL.solve(A2, b2))
        A2, _ = _scalar_case(4, cells, pin, pe, u2, v2, g2)
        P2 = AL.partition(A2, Xs2, (1.0, 1.0))
        red = _reduction_sweep(cond, P, As, bs, x_ref, (A2, P2, snaps),
                               lambda xx, xr: np.linalg.norm(xx - xr) / np.linalg.norm(xr), ms)
        rows.append({"Pe_b": pe, "exact_rel_err": float(np.abs(x - x_ref).max() / np.abs(x_ref).max()),
                     "direct_s": t_direct, "condensed_s": t_cond, "condensation": cond.timings, "reduced": red,
                     "T_max": float(x_ref.max())})
    return {"pins": n_pins * n_pins, "unknowns": int(As.shape[0]), "rows": rows}


def _cht_grid(n_pins, cells, pin, inlet_blocks=1, outlet_blocks=2):
    from . import flow as FL
    solid = _pins(n_pins, cells, pin, inlet_blocks, outlet_blocks)
    return FL.Grid(solid.shape[0], solid.shape[1], 1.0 / cells, solid, walls="slip")


def _cht_energy(g, u, v, pin_size, k_ratio=100.0, Pr=7.0, Re=10.0):
    """Conjugate energy equation: fluid conductivity 1 / (Re Pr) (lengths in pin sizes, U = 1, rho c = 1 in the fluid),
    solid conductivity ``k_ratio`` times larger and no advection, unit volumetric source in the fins."""
    from . import flow as FL
    kf = pin_size / (Re * Pr)                        # alpha = U d / Pe with d = pin size in unit-cell lengths
    k = np.where(g.solid, k_ratio * kf, kf)
    rhoc = np.where(g.solid, 0.0, 1.0)
    q = np.where(g.solid, 1.0, 0.0)
    return FL.assemble_scalar(g, u, v, k, rhoc, q)


def f3_cht(n_pins=(8, 4), cells: int = 16, pin: int = 8, res=(1, 10, 50, 100), ms=(2, 4, 8, 16), n_snap: int = 16,
           tol: float = 1e-6, maxiter: int = 150, seed: int = 0) -> dict:
    """F3 / F3r: cold plate (fin array in a free-slip strip), Navier-Stokes by Picard, full solve against condensed
    Picard (every Oseen iterate condensed by content), then the conjugate energy equation; reduced faces for the
    flow at the largest Re; the fraction of reversed face flux on block faces."""
    from . import algebraic as AL
    from . import flow as FL
    g = _cht_grid(n_pins, cells, pin)
    d = pin / cells                                  # pin size in unit-cell lengths
    X, F = g.coords()
    prim = _primal_pressure(g, cells)
    inlet = np.ones(g.ny)
    rows = []
    for Re in res:
        nu = d / Re
        t0 = time.perf_counter()
        full = FL.picard(g, nu, inlet, tol=tol, maxiter=maxiter, relax=0.7)
        t_full = time.perf_counter() - t0
        stats = {"distinct": [], "blocks": []}

        def cond_solver(A, b, stats=stats):
            P = AL.partition(A, X, (1.0, 1.0), fieldid=F, primal=prim)
            c = AL.Condensation(A, P).prepare()
            stats["distinct"].append(c.timings["distinct_blocks"])
            stats["blocks"].append(c.timings["blocks"])
            return c.solve(b)[0]
        t0 = time.perf_counter()
        cnd = FL.picard(g, nu, inlet, tol=tol, maxiter=maxiter, relax=0.7, linear_solver=cond_solver)
        t_cnd = time.perf_counter() - t0
        uf, vf, pf = FL.split(g, full["x"])
        uc, vc, pc = FL.split(g, cnd["x"])
        dp_f, dp_c = float(np.nanmean(pf[0])), float(np.nanmean(pc[0]))
        Ae, be = _cht_energy(g, uf, vf, d, Re=Re)
        Te = FL.solve(Ae, be)
        Ae2, be2 = _cht_energy(g, uc, vc, d, Re=Re)
        Tc = FL.solve(Ae2, be2)
        fin = g.solid.ravel()
        rise_f, rise_c = float(Te[fin].max()), float(Tc[fin].max())
        # reversed flux through block faces: u on x-faces at block boundaries that points upstream
        xb = uf[::cells][1:-1]
        reversed_frac = float((np.clip(-xb, 0, None).sum()) / max(np.abs(xb).sum(), 1e-300))
        rows.append({"Re_fin": Re, "picard_full": {"iterations": full["iterations"], "seconds": t_full,
                                                   "converged": bool(full["history"] and full["history"][-1] < tol)},
                     "picard_condensed": {"iterations": cnd["iterations"], "seconds": t_cnd,
                                          "distinct_blocks_last": stats["distinct"][-1], "blocks": stats["blocks"][-1]},
                     "dp_full": dp_f, "dp_condensed": dp_c, "dp_rel_err": abs(dp_c - dp_f) / abs(dp_f),
                     "fin_rise_full": rise_f, "fin_rise_condensed": rise_c,
                     "fin_rise_rel_err": abs(rise_c - rise_f) / abs(rise_f),
                     "reversed_block_face_flux": reversed_frac, "_x": full["x"]})
    # reduced faces for the final Oseen operator at the largest Re, snapshots: the same operator on a smaller array
    Re = res[-1]
    xr = rows[-1].pop("_x")
    for r in rows:
        r.pop("_x", None)
    u, v, _ = FL.split(g, xr)
    A, b = FL.assemble_oseen(g, d / Re, inlet, (u, v))
    cond, P = _condense(A, X, F, 1.0, prim)
    g2 = _cht_grid((4, n_pins[1]), cells, pin)
    r2 = FL.picard(g2, d / Re, np.ones(g2.ny), tol=tol, maxiter=maxiter, relax=0.7)
    u2, v2, _ = FL.split(g2, r2["x"])
    rng = np.random.default_rng(seed)
    snaps = []
    for _ in range(n_snap):
        y = (np.arange(g2.ny) + 0.5) / g2.ny
        prof = 1 + sum(rng.normal(0, 0.3) * np.sin((k + 1) * np.pi * y) for k in range(4))
        A2k, b2k = FL.assemble_oseen(g2, d / Re, prof, (u2, v2))
        snaps.append(FL.solve(A2k, b2k))
    A2, _ = FL.assemble_oseen(g2, d / Re, np.ones(g2.ny), (u2, v2))
    X2, F2 = g2.coords()
    P2 = AL.partition(A2, X2, (1.0, 1.0), fieldid=F2, primal=_primal_pressure(g2, cells))

    def dpm(xx, xr_):
        return abs(np.nanmean(FL.split(g, xx)[2][0]) - np.nanmean(FL.split(g, xr_)[2][0])) / \
            abs(np.nanmean(FL.split(g, xr_)[2][0]))
    x_ref = FL.solve(A, b)
    red = _reduction_sweep(cond, P, A, b, x_ref, (A2, P2, snaps), dpm, ms)
    return {"pins": list(n_pins), "unknowns": int(g.n), "rows": rows, "reduced_at_Re": Re, "reduced": red}


# ---------------------------------------------------------------------------------------------------- T2 T3 T8 T9
def thermal_case(pkg, m: int = 12, rtol: float = 1e-10) -> dict:
    """One package: fine FV peak junction temperature; m-mode condensation (Tmax error, energy-norm effectivity of the
    residual estimate); exact condensation (PCG to ``rtol``) against the fine solve."""
    from .substructure import energy_norm
    s = fv.assemble_package(pkg)
    t0 = time.perf_counter()
    T, info = fv.solve(s.K, fv.source_vector(pkg))
    t_fine = time.perf_counter() - t0
    T = T.reshape(s.shape)
    tm = float(reference_observables(pkg, T)["Tj"].max())
    c = Condensed(pkg, m).prepare()
    st = c.solve()
    t_red = float(c.observables(st)["Tj"].max())
    eta = c.estimate(st)
    en = energy_norm(s.K, (c.field(st) - T).ravel())
    r = c.correct(st, rtol=rtol, maxiter=3000)
    t_ex = float(c.observables_fine(r["U"], r["pb"])["Tj"].max())
    return {"Tmax_fine": tm, "fine_s": t_fine, "fine_method": info.method, "m": m, "Tmax_reduced": t_red,
            "err_reduced": abs(t_red - tm), "effectivity": float(np.sqrt((eta ** 2).sum()) / max(en, 1e-300)),
            "Tmax_exact": t_ex, "exact_vs_fine": abs(t_ex - tm), "exact_iterations": r["iterations"],
            "exact_rel_residual": r["history"][-1], "prepare_s": c.timings["prepare"]}


def _stack(**kw):
    from dataclasses import replace

    from .package import reference_stack
    return replace(reference_stack(), **kw)


def t2_contrast(nb: int = 32, ratios=(1e3, 1e4, 1e5), m: int = 12, seed: int = 2) -> dict:
    """T2: the underfill conductivity lowered so that k_copper / k_underfill = ratio (baseline 401 / 0.5 = 802)."""
    from .package import MATERIALS
    rows = [{"ratio": MATERIALS["copper"][0] / MATERIALS["underfill"][0], "baseline": True,
             **thermal_case(uniform_package((nb, nb), seed=seed), m)}]
    for q in ratios:
        st = _stack(materials=(("underfill", MATERIALS["copper"][0] / q, MATERIALS["underfill"][1]),))
        rows.append({"ratio": q, **thermal_case(uniform_package((nb, nb), st, seed=seed), m)})
    return {"blocks": nb * nb, "rows": rows}


def t3_anisotropy(nb: int = 32, factors=(100.0, 0.01), m: int = 12, seed: int = 2) -> dict:
    """T3: TIM and interposer with in-plane / through-thickness conductivity ratio ``factor``."""
    rows = [{"factor": 1.0, "baseline": True, **thermal_case(uniform_package((nb, nb), seed=seed), m)}]
    for f in factors:
        st = _stack(anisotropy=(("tim", f), ("interposer", f)))
        rows.append({"factor": f, **thermal_case(uniform_package((nb, nb), st, seed=seed), m)})
    return {"blocks": nb * nb, "rows": rows}


def t8_adversarial(nb: int = 16, m: int = 12, seed: int = 2) -> dict:
    """T8: power maps unlike the snapshot set (same total power as the baseline hotspot map)."""
    base = uniform_package((nb, nb), seed=seed)
    n = base.stack.n
    P = base.power.sum()
    nx = nb * n
    i, j = np.meshgrid(np.arange(nx), np.arange(nx), indexing="ij")
    rng = np.random.default_rng(seed)
    maps = {"baseline hotspots": base.power,
            "checkerboard per cell": ((i + j) % 2).astype(float),
            "checkerboard per block": ((i // n + j // n) % 2).astype(float),
            "single-cell spike": ((i == nx // 3) & (j == nx // 2)).astype(float),
            "white noise": rng.random((nx, nx)),
            "block corners only": (((i % n) == 0) & ((j % n) == 0)).astype(float)}
    rows = []
    for name, p in maps.items():
        pkg = uniform_package((nb, nb), seed=seed)
        pkg.power = P * p / p.sum()
        rows.append({"map": name, **thermal_case(pkg, m)})
    return {"blocks": nb * nb, "rows": rows}


def t9_thin(nb: int = 16, m: int = 12, seed: int = 2) -> dict:
    """T9: thin layers -- TIM 30 -> 3 -> 1 um (1 um with one cell row), die 100 -> 20 um."""
    from dataclasses import replace

    from .package import reference_stack
    base = reference_stack()

    def with_layer(st, name, **kw):
        return replace(st, layers=tuple(replace(L, **kw) if L.name == name else L for L in st.layers))
    cases = {"baseline": base,
             "TIM 3 um": with_layer(base, "tim", thickness=3e-6),
             "TIM 1 um, one cell": with_layer(base, "tim", thickness=1e-6, nz=1),
             "die 20 um": with_layer(base, "die", thickness=20e-6),
             "TIM 1 um and die 20 um": with_layer(with_layer(base, "tim", thickness=1e-6, nz=1), "die",
                                                  thickness=20e-6)}
    rows = [{"case": k, **thermal_case(uniform_package((nb, nb), st, seed=seed), m)} for k, st in cases.items()]
    return {"blocks": nb * nb, "rows": rows}


# ---------------------------------------------------------------------------------------------------- T4 / T5
def _t4_package(nb, seed):
    from .learned import default_variant, variant_field
    from .package import Package, hotspot_power, reference_stack
    st = reference_stack()
    shape = (nb, nb)
    theta = variant_field(shape, seed=seed, defects=0)
    rng = np.random.default_rng(seed)
    idx = rng.choice(nb * nb, 4, replace=False)
    th = theta.reshape(-1, 3)
    th[idx[:2], 0] = 0.05                             # nearly open microbumps
    th[idx[2:], 1] = 30.0                             # delaminated TIM
    p = hotspot_power(st, shape, 1e6 * nb * nb * st.pitch ** 2, seed=seed + 3)
    return Package(st, np.arange(nb * nb).reshape(shape), [default_variant(t) for t in th], p), theta


def t4_no_repetition(nb: int = 32, m: int = 12, seed: int = 1, n_train: int = 48) -> dict:
    """T4: every block distinct (continuous random variants, four defects). Against one fine FV solve: the exact
    path (every block condensed, m-mode interfaces, then the PCG correction to the fine solution), and the learned
    path (:func:`.learned.solve_adaptive`, training timed separately)."""
    from .learned import LearnedBlocks, solve_adaptive
    pkg, theta = _t4_package(nb, seed)
    t0 = time.perf_counter()
    T, info = fv.solve_package(pkg)
    t_fine = time.perf_counter() - t0
    tm = float(reference_observables(pkg, T)["Tj"].max())
    del T
    t0 = time.perf_counter()
    c = Condensed(pkg, m).prepare()
    st = c.solve()
    t_build = time.perf_counter() - t0
    t_red = float(c.observables(st)["Tj"].max())
    r = c.correct(st, rtol=1e-8, maxiter=3000)
    t_ex = float(c.observables_fine(r["U"], r["pb"])["Tj"].max())
    exact = {"build_s": t_build, "Tmax_reduced": t_red, "err_reduced": abs(t_red - tm), "correct_s": r["seconds"],
             "correct_iterations": r["iterations"], "Tmax_corrected": t_ex, "err_corrected": abs(t_ex - tm),
             "total_s": t_build + r["seconds"]}
    del c, st, r
    t0 = time.perf_counter()
    L = LearnedBlocks(pkg.stack, m=m, n_train=n_train, seed=seed).fit()
    t_train = time.perf_counter() - t0
    ra = solve_adaptive(L, theta, pkg.power)
    tl = float(ra.obs["Tj"].max())
    learned = {"training_s": t_train, "solve_s": ra.timings["total"], "Tmax": tl, "err": abs(tl - tm),
               "exact_blocks": int(ra.exact_blocks.sum())}
    return {"blocks": nb * nb, "fine_s": t_fine, "fine_method": info.method, "Tmax_fine": tm, "exact": exact,
            "learned": learned}


def t5_variants(nb: int = 64, Vs=(1, 4, 16, 64, 256), m: int = 12, seed: int = 0) -> dict:
    """T5: ``V`` discrete variants at random positions: hierarchy build (memoised subtrees) against the flat
    condensation build, both including the V block condensations."""
    from .learned import default_variant
    rows = []
    rng = np.random.default_rng(seed)
    for V in Vs:
        pkg = uniform_package((nb, nb), seed=2)
        th = np.column_stack([rng.uniform(0.3, 1.0, V), rng.uniform(0.5, 3.0, V), rng.uniform(0.5, 1.0, V)])
        pkg.variants = [default_variant(t) for t in th]
        pkg.layout = rng.integers(0, V, (nb, nb)) if V > 1 else np.zeros((nb, nb), int)
        t0 = time.perf_counter()
        h = Hierarchy(pkg, m).prepare()
        sol = h.solve()
        t_h = time.perf_counter() - t0
        th_ = float(h.observables(sol)["Tj"].max())
        del h, sol
        t0 = time.perf_counter()
        c = Condensed(pkg, m).prepare()
        st = c.solve()
        t_f = time.perf_counter() - t0
        rows.append({"V": V, "hierarchy_s": t_h, "flat_s": t_f, "Tmax_hierarchy": th_,
                     "Tmax_flat": float(c.observables(st)["Tj"].max())})
        del c, st
    return {"blocks": nb * nb, "rows": rows}


# ---------------------------------------------------------------------------------------------------- T6 / T7
def _voxel_partition(pkg, K):
    """Algebraic partition of the package's fine FV unknowns into its blocks (lateral position)."""
    from . import algebraic as AL
    st = pkg.stack
    nx, ny = pkg.shape[0] * st.n, pkg.shape[1] * st.n
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(st.nz), indexing="ij")
    X = np.column_stack([(i.ravel() + 0.5) / st.n, (j.ravel() + 0.5) / st.n])
    F = k.ravel()                                     # z row as the "field": canonical order inside a column
    return AL.partition(K, X, (1.0, 1.0), fieldid=F)


def _nonlinear(pkg, k_of_T, R_of_T, tol=1e-8, maxiter=40, condensed=False, T_amb=300.0):
    """Picard for k(T) and R_c(T): ``k_of_T(k0, Tabs) -> k`` per cell; ``R_of_T(R0, Tabs_interface) -> R``.
    ``condensed``: the cell temperatures entering k and R are block means per z row (each block one linear
    operator per iteration, condensed by :mod:`.algebraic`)."""
    from . import algebraic as AL
    st = pkg.stack
    p0 = pkg.global_properties()
    f = fv.source_vector(pkg)
    nbx, nby, n = pkg.shape[0], pkg.shape[1], st.n
    T = np.zeros((nbx * n, nby * n, st.nz))
    hist, times = [], []
    part = None
    for _ in range(maxiter):
        Tu = T
        if condensed:                                 # block-mean temperature per z row
            Tb = T.reshape(nbx, n, nby, n, st.nz).mean((1, 3))
            Tu = np.repeat(np.repeat(Tb, n, 0), n, 1)
        k = k_of_T(p0["k"], T_amb + Tu)
        kl = k_of_T(p0["k_lat"], T_amb + Tu)
        Ti = 0.5 * (Tu[..., :-1] + Tu[..., 1:])
        R = R_of_T(p0["R"], T_amb + Ti)
        s = fv.assemble((kl, kl, k), p0["dz"], R, st.dx, st.h_top, st.h_bottom)
        t0 = time.perf_counter()
        if condensed:
            if part is None:
                part = _voxel_partition(pkg, s.K)
            c = AL.Condensation(s.K.tocsr(), part).prepare()
            Tn = c.solve(f)[0].reshape(T.shape)
        else:
            Tn = fv.solve(s.K, f)[0].reshape(T.shape)
        times.append(time.perf_counter() - t0)
        d = float(np.abs(Tn - T).max() / max(np.abs(Tn).max(), 1e-300))
        T = Tn
        hist.append(d)
        if d < tol:
            break
    return T, hist, times


def _k_of_T(alpha_si=1.3, alpha_cu=0.1):
    from .package import MATERIALS

    def f(k0, Tabs):
        out = k0.copy()
        si = np.isclose(k0, MATERIALS["silicon"][0])
        cu = np.isclose(k0, MATERIALS["copper"][0])
        out[si] = k0[si] * (300.0 / Tabs[si]) ** alpha_si
        out[cu] = k0[cu] * (300.0 / Tabs[cu]) ** alpha_cu
        return out
    return f


def t6_t7_nonlinear(nb: int = 16, case: str = "k", seed: int = 2, power_scale: float = 2.0) -> dict:
    """T6 (``case="k"``: silicon and copper k(T)) or T7 (``case="R"``: R_c(T) = R_0 (1 + beta (T - T_0)) on the
    die/TIM contact, beta doubling R_c over the temperature range of the linear solution)."""
    pkg = uniform_package((nb, nb), seed=seed)
    pkg.power = pkg.power * power_scale
    ident_k = lambda k0, Tabs: k0                     # noqa: E731
    ident_R = lambda R0, Tabs: R0                     # noqa: E731
    if case == "k":
        k_of_T, R_of_T = _k_of_T(), ident_R
    else:
        st = pkg.stack
        T_lin = fv.solve_package(pkg)[0].reshape(pkg.shape[0] * st.n, pkg.shape[1] * st.n, st.nz)
        die_top = st.layer_slices()["die"].stop - 1
        Ti_lin = 0.5 * (T_lin[..., die_top] + T_lin[..., die_top + 1])   # the interface temperature R_c sees
        lo, hi = 300.0 + float(Ti_lin.min()), 300.0 + float(Ti_lin.max())
        beta = 1.0 / max(hi - lo, 1e-9)

        def R_of_T(R0, Tabs, lo=lo, beta=beta):
            out = R0.copy()
            # rises linearly from R_0 at the coolest to 2 R_0 at the hottest interface temperature of the linear
            # solution, and saturates there (unbounded growth runs away: R up -> T up -> R up)
            out[..., die_top] = R0[..., die_top] * np.clip(1 + beta * (Tabs[..., die_top] - lo), 1.0, 2.0)
            return out
        k_of_T = ident_k
    t0 = time.perf_counter()
    Tf, hf, tf = _nonlinear(pkg, k_of_T, R_of_T)
    t_fine = time.perf_counter() - t0
    t0 = time.perf_counter()
    Tc, hc, tc = _nonlinear(pkg, k_of_T, R_of_T, condensed=True)
    t_cnd = time.perf_counter() - t0
    st = pkg.stack
    tj_f = float(Tf[..., st.active_k].max())
    tj_c = float(Tc[..., st.active_k].max())
    T_lin = fv.solve_package(pkg)[0].reshape(pkg.shape[0] * st.n, pkg.shape[1] * st.n, st.nz)
    return {"case": case, "blocks": nb * nb, "Tmax_fine_nonlinear": tj_f, "Tmax_condensed_nonlinear": tj_c,
            "err": abs(tj_c - tj_f), "Tmax_linear": float(T_lin[..., st.active_k].max()),
            "fine": {"iterations": len(hf), "seconds": t_fine, "history": hf},
            "condensed": {"iterations": len(hc), "seconds": t_cnd, "history": hc}}


# ---------------------------------------------------------------------------------------------------- T10
def _assemble_graded(k, kl, dz, R, dxs, dy, h_top, h_bottom):
    """The FV operator of :func:`.fv.assemble` with a cell width per x column ``dxs`` (no ports)."""
    import scipy.sparse as sp
    nx, ny, nz = k.shape
    dx = np.asarray(dxs, float)[:, None, None]
    dzb = np.asarray(dz, float)[None, None, :]
    idx = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    rows, cols, vals = [], [], []
    diag = np.zeros(nx * ny * nz)

    def couple(i, j, g):
        i, j, g = i.ravel(), j.ravel(), np.broadcast_to(g, i.shape).ravel()
        rows.extend((i, j))
        cols.extend((j, i))
        vals.extend((-g, -g))
        np.add.at(diag, i, g)
        np.add.at(diag, j, g)
    couple(idx[:-1], idx[1:], dy * dzb / (dx[:-1] / (2 * kl[:-1]) + dx[1:] / (2 * kl[1:])))
    couple(idx[:, :-1], idx[:, 1:], dx * dzb / (dy / (2 * kl[:, :-1]) + dy / (2 * kl[:, 1:])))
    Rb = np.broadcast_to(np.asarray(R, float), (nx, ny, nz - 1))
    couple(idx[..., :-1], idx[..., 1:],
           dx * dy / (dzb[..., :-1] / (2 * k[..., :-1]) + Rb + dzb[..., 1:] / (2 * k[..., 1:])))
    A = dx[..., 0] * dy
    np.add.at(diag, idx[..., -1].ravel(), (A / (dz[-1] / (2 * k[..., -1]) + 1 / h_top)).ravel())
    np.add.at(diag, idx[..., 0].ravel(), (A / (dz[0] / (2 * k[..., 0]) + 1 / h_bottom)).ravel())
    rows.append(np.arange(diag.size))
    cols.append(np.arange(diag.size))
    vals.append(diag)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(diag.size, diag.size))


def _graded_case(nb, pitches, seed, power=None):
    st0 = uniform_package((nb, nb), seed=seed)
    st = st0.stack
    props = st0.global_properties()
    n = st.n
    col_pitch = np.array([pitches[i % len(pitches)] for i in range(nb)])
    dxs = np.repeat(col_pitch / n, n)
    K = _assemble_graded(props["k"], props["k_lat"], props["dz"], props["R"], dxs, st.dx, st.h_top, st.h_bottom)
    f = np.zeros((nb * n, nb * n, st.nz))
    p = st0.power if power is None else power
    f[..., st.active_k] = p
    i, j, kk = np.meshgrid(np.arange(nb * n), np.arange(nb * n), np.arange(st.nz), indexing="ij")
    X = np.column_stack([(i.ravel() + 0.5) / n, (j.ravel() + 0.5) / n])   # block-index coordinates
    return K, f.ravel(), X, kk.ravel(), st


def t10_pitch(nb: int = 16, pitches=(40e-6, 20e-6), ms=(2, 4, 8, 12, 16, 24), n_snap: int = 24,
              seed: int = 2) -> dict:
    """T10: columns of blocks alternating in pitch, through algebraic condensation (each pitch its own block
    operator, found by content); baseline: the uniform pitch through the same pipeline. Reduced faces from
    snapshots of a 4 x 4 package with the same column pattern and random power maps."""
    import scipy.sparse.linalg as spla

    from . import algebraic as AL
    out = {}
    for label, pt in (("uniform", (pitches[0],)), ("alternating", tuple(pitches))):
        K, f, X, F, st = _graded_case(nb, pt, seed)
        n = st.n
        t0 = time.perf_counter()
        x_ref = spla.spsolve(K.tocsc(), f)
        t_ref = time.perf_counter() - t0
        tm = float(x_ref.reshape(nb * n, nb * n, st.nz)[..., st.active_k].max())
        P = AL.partition(K, X, (1.0, 1.0), fieldid=F)
        c = AL.Condensation(K, P).prepare()
        x, _ = c.solve(f)
        t_ex = float(x.reshape(nb * n, nb * n, st.nz)[..., st.active_k].max())
        K2, _, X2, F2, _ = _graded_case(4, pt, seed)
        P2 = AL.partition(K2, X2, (1.0, 1.0), fieldid=F2)
        rng = np.random.default_rng(seed)
        snaps = []
        for _ in range(n_snap):
            pw = rng.random((4 * n, 4 * n)) * rng.uniform(0, 2, (4, 4)).repeat(n, 0).repeat(n, 1)
            _, f2, _, _, _ = _graded_case(4, pt, seed, power=pw * f.sum() / pw.sum() / (nb / 4) ** 2)
            snaps.append(spla.spsolve(K2.tocsc(), f2))
        fc2, fc = AL.faces(P2, K2), AL.faces(P, K)
        rows = []
        for m in ms:
            R = AL.Reduced(c, fc, AL.face_modes(snaps, fc2, m)).prepare()
            xm, _ = R.solve(f)
            rows.append({"m": m, "Tmax": float(xm.reshape(nb * n, nb * n, st.nz)[..., st.active_k].max()),
                         "reduced_dofs": R.timings["reduced_dofs"], "unseen_faces": R.timings["unseen_faces"]})
        for r in rows:
            r["err"] = abs(r["Tmax"] - tm)
        out[label] = {"Tmax_fine": tm, "fine_s": t_ref, "Tmax_exact": t_ex, "exact_vs_fine": abs(t_ex - tm),
                      "distinct_blocks": c.timings["distinct_blocks"], "rows": rows}
    return {"blocks": nb * nb, "pitches": list(pitches), **out}


# ---------------------------------------------------------------------------------------------------- meshless
def _fv_truth_interp(g, T, cells):
    """Bilinear interpolant of a cell-centred field on a :mod:`.flow` grid (lengths in unit cells)."""
    from scipy.interpolate import RegularGridInterpolator
    x = (np.arange(g.nx) + 0.5) / cells
    y = (np.arange(g.ny) + 0.5) / cells
    return RegularGridInterpolator((x, y), T.reshape(g.nx, g.ny), bounds_error=False, fill_value=None)


def m1_advection(pins: int = 4, densities=(8, 16, 32), truth_cells: int = 64, pes=(1, 10, 100),
                 methods=("rbf_fd", "gfdm"), seed: int = 0) -> dict:
    """M1: meshless advection-diffusion on point clouds (velocity interpolated from the finest FV Stokes solution)
    against FV at equal unknowns; truth: FV at ``truth_cells`` per unit cell."""
    import scipy.sparse.linalg as spla

    from . import flow as FL
    from . import meshless2d as M
    pf = 0.5
    gt, At, bt = _stokes_system((pins, pins), truth_cells, int(pf * truth_cells))
    ut, vt, _ = FL.split(gt, spla.spsolve(At.tocsc(), bt))
    rows = []
    for pe in pes:
        As, bs = _scalar_case(pins, truth_cells, int(pf * truth_cells), pe, ut, vt, gt)
        Tt = spla.spsolve(As.tocsc(), bs)
        truth = _fv_truth_interp(gt, Tt, truth_cells)
        scale = float(np.abs(Tt).max())
        for c in densities:
            g, A, b = _stokes_system((pins, pins), c, int(pf * c))
            u, v, _ = FL.split(g, spla.spsolve(A.tocsc(), b))
            A2, b2 = _scalar_case(pins, c, int(pf * c), pe, u, v, g)
            t0 = time.perf_counter()
            T = spla.spsolve(A2.tocsc(), b2)
            t_fv = time.perf_counter() - t0
            Xc = FL.scalar_coords(g)
            e_fv = float(np.abs(T - truth(Xc)).max() / scale)
            rows.append({"Pe_b": pe, "density": c, "method": "fv", "unknowns": int(len(T)), "err": e_fv,
                         "solve_s": t_fv})
            X, _, box = M.cloud((pins, pins), 1.0 / c, pf, seed=seed)
            X = M.snap_to_box(X, box, 1.0 / c)
            vel = M.bilinear_mac(gt, ut, vt, X)
            lp = X % 1.0 - 0.5
            inpin = (np.abs(lp[:, 0]) < pf / 2) & (np.abs(lp[:, 1]) < pf / 2) & (X[:, 0] > 1) & \
                (X[:, 0] < 1 + pins)
            for meth in methods:
                t0 = time.perf_counter()
                Am, bm = M.advection_diffusion(X, box, 1.0 / c, vel, 1.0 / pe, inpin.astype(float), method=meth)
                try:
                    Tm = spla.spsolve(Am.tocsc(), bm)
                    err = float(np.abs(Tm - truth(X)).max() / scale)
                    l2 = float(np.linalg.norm(Tm - truth(X)) / np.linalg.norm(truth(X)))
                except Exception as e:                # noqa: BLE001 -- a singular or failed solve is a result
                    err, l2 = float("nan"), float("nan")
                    rows.append({"Pe_b": pe, "density": c, "method": meth, "failed": str(e)[:200]})
                    continue
                rows.append({"Pe_b": pe, "density": c, "method": meth, "unknowns": int(len(X)), "err": err,
                             "rel_l2": l2, "solve_s": time.perf_counter() - t0})
    return {"pins": pins * pins, "truth_cells": truth_cells, "rows": rows}


def m2_stokes(pins: int = 4, densities=(8, 16, 32), truth_cells: int = 64, methods=("rbf_fd", "gfdm"),
              seed: int = 0) -> dict:
    """M2: collocated meshless Stokes on fluid points with pin-wall points; pressure drop against FV."""
    import scipy.sparse.linalg as spla

    from . import flow as FL
    from . import meshless2d as M
    pf = 0.5
    rows = []
    gt, At, bt = _stokes_system((pins, pins), truth_cells, int(pf * truth_cells))
    xt = spla.spsolve(At.tocsc(), bt)
    dp_t = float(np.nanmean(FL.split(gt, xt)[2][0]))
    for c in densities:
        g, A, b = _stokes_system((pins, pins), c, int(pf * c))
        dp = float(np.nanmean(FL.split(g, spla.spsolve(A.tocsc(), b))[2][0]))
        rows.append({"density": c, "method": "fv", "unknowns": int(g.n), "dp": dp, "dp_rel_err": abs(dp - dp_t) / dp_t})
        X, wall, box = M.cloud((pins, pins), 1.0 / c, pf, seed=seed, fluid_only=True, wall_points=True)
        X = M.snap_to_box(X, box, 1.0 / c)
        N = len(X)
        for meth in methods:
            t0 = time.perf_counter()
            Am, bm = M.stokes(X, wall, box, 1.0 / c, method=meth)
            try:
                x = spla.spsolve(Am.tocsc(), bm)
                p = x[2 * N:]
                inl = X[:, 0] < 0.5 / c + 1e-9
                dpm = float(np.mean(p[inl])) if inl.any() else float("nan")
                ok = bool(np.isfinite(x).all())
            except Exception as e:                    # noqa: BLE001
                rows.append({"density": c, "method": meth, "failed": str(e)[:200]})
                continue
            rows.append({"density": c, "method": meth, "unknowns": int(3 * N), "dp": dpm,
                         "dp_rel_err": abs(dpm - dp_t) / dp_t if ok else float("nan"), "finite": ok,
                         "seconds": time.perf_counter() - t0})
    return {"pins": pins * pins, "dp_truth": dp_t, "truth_cells": truth_cells, "rows": rows}


def m3_hybrid(pins: int = 2, densities=(8, 16, 32), truth_cells: int = 64, k_pin: float = 400.0, k_mat: float = 1.0,
              R_c: float = 1e-3, stencils=((12, 2), (20, 2), (20, 3)), seed: int = 0) -> dict:
    """M3: FV-at-interfaces hybrid against FV everywhere at equal unknowns (truth: FV at ``truth_cells``), with the
    error split between the FV band and the meshless points, and the stencil sensitivity."""
    import scipy.sparse.linalg as spla
    from scipy.interpolate import RegularGridInterpolator

    from . import meshless2d as M
    pf = 0.5
    At, bt, _, _ = M.fv_conduction((pins, pins), truth_cells, pf, k_pin, k_mat, R_c)
    Tt = spla.spsolve(At.tocsc(), bt)
    n = pins * truth_cells
    gx = (np.arange(n) + 0.5) / truth_cells
    truth = RegularGridInterpolator((gx, gx), Tt.reshape(n, n), bounds_error=False, fill_value=None)
    scale = float(np.abs(Tt).max())
    rows = []
    for c in densities:
        A, b, X, _ = M.fv_conduction((pins, pins), c, pf, k_pin, k_mat, R_c)
        T = spla.spsolve(A.tocsc(), b)
        rows.append({"density": c, "method": "fv", "unknowns": int(len(T)),
                     "err": float(np.abs(T - truth(X)).max() / scale)})
        for nn, deg in stencils:
            A, b, X, kind, _ = M.hybrid_conduction((pins, pins), c, pf, k_pin, k_mat, R_c, n=nn, degree=deg, seed=seed)
            T = spla.spsolve(A.tocsc(), b)
            e = np.abs(T - truth(X)) / scale
            rows.append({"density": c, "method": f"hybrid n={nn} p={deg}", "unknowns": int(len(T)),
                         "fv_fraction": float((kind == 0).mean()), "err": float(e.max()),
                         "err_band": float(e[kind == 0].max()) if (kind == 0).any() else None,
                         "err_points": float(e[kind == 1].max()) if (kind == 1).any() else None})
    return {"pins": pins * pins, "rows": rows}


def m4_meshless_condensed(pins: int = 4, cells: int = 16, seed: int = 0) -> dict:
    """M4: meshless matrices condensed by :mod:`.algebraic`: the hybrid conduction (replicated point clouds and FV
    bands: identical blocks expected) and the meshless advection-diffusion (velocity differs per block)."""
    import scipy.sparse.linalg as spla

    from . import algebraic as AL
    from . import flow as FL
    from . import meshless2d as M
    out = {}
    A, b, X, kind, _ = M.hybrid_conduction((pins, pins), cells, 0.5, 400.0, 1.0, 1e-3, seed=seed)
    cases = {"hybrid conduction": (A, b, X, (0.0, 0.0))}
    gt, At, bt = _stokes_system((pins, pins), cells, cells // 2)
    ut, vt, _ = FL.split(gt, spla.spsolve(At.tocsc(), bt))
    Xc, _, box = M.cloud((pins, pins), 1.0 / cells, 0.5, seed=seed)
    Xc = M.snap_to_box(Xc, box, 1.0 / cells)
    lp = Xc % 1.0 - 0.5
    inpin = (np.abs(lp[:, 0]) < 0.25) & (np.abs(lp[:, 1]) < 0.25) & (Xc[:, 0] > 1) & (Xc[:, 0] < 1 + pins)
    Am, bm = M.advection_diffusion(Xc, box, 1.0 / cells, M.bilinear_mac(gt, ut, vt, Xc), 0.1, inpin.astype(float))
    cases["advection-diffusion Pe_b=10"] = (Am, bm, Xc, (0.0, 0.0))
    for name, (A, b, X, org) in cases.items():
        t0 = time.perf_counter()
        x_ref = spla.spsolve(A.tocsc(), b)
        t_direct = time.perf_counter() - t0
        t0 = time.perf_counter()
        P = AL.partition(A, X, (1.0, 1.0), origin=org)
        c = AL.Condensation(A.tocsr(), P).prepare()
        x, _ = c.solve(b)
        t_c = time.perf_counter() - t0
        out[name] = {"unknowns": int(A.shape[0]), "exact_rel_err": float(np.abs(x - x_ref).max() / np.abs(x_ref).max()),
                     "direct_s": t_direct, "condensed_s": t_c, "speedup": t_direct / t_c, **c.timings}
    return out


# ---------------------------------------------------------------------------------------------------- E1
def exception_patches(A, b, x0, x_true, block, nbshape, budgets, select="residual", halo=1, seed=0):
    """Generic prior + exception patches: blocks ranked by the norm of ``b - A x0`` over their unknowns (or random),
    the patch (selected blocks + ``halo`` rings) solved with the prior as Dirichlet data outside."""
    import scipy.sparse.linalg as spla
    A = A.tocsr()
    r = b - A @ x0
    nblk = nbshape[0] * nbshape[1]
    eta = np.sqrt(np.bincount(block, weights=r ** 2, minlength=nblk))
    errb = np.sqrt(np.bincount(block, weights=(x0 - x_true) ** 2, minlength=nblk))
    rng = np.random.default_rng(seed)
    rows = []
    for bud in budgets:
        k = int(round(bud * nblk))
        if select == "residual":
            sel = np.argsort(-eta)[:k]
        else:
            sel = rng.choice(nblk, k, replace=False)
        mask = np.zeros(nbshape, bool)
        mask.ravel()[sel] = True
        top = set(np.argsort(-errb)[:k].tolist())
        overlap = len(top & set(sel.tolist())) / k if k else float("nan")
        patch = mask.copy()
        for _ in range(halo):
            p2 = patch.copy()
            p2[1:] |= patch[:-1]
            p2[:-1] |= patch[1:]
            p2[:, 1:] |= patch[:, :-1]
            p2[:, :-1] |= patch[:, 1:]
            patch = p2
        U = np.nonzero(patch.ravel()[block])[0]
        x = x0.copy()
        if len(U):
            Auu = A[U][:, U]
            rhs = b[U] - A[U] @ x0 + Auu @ x0[U]
            x[U] = spla.spsolve(Auu.tocsc(), rhs)
        rows.append({"budget": bud, "solved_fraction": len(U) / len(x0),
                     "err_max": float(np.abs(x - x_true).max() / np.abs(x_true).max()),
                     "err_l2": float(np.linalg.norm(x - x_true) / np.linalg.norm(x_true)), "overlap_top_error": overlap})
    return rows


def _coarse_prior(g_f, cells_f, g_c, T_c, cells_c):
    """A coarse cell-centred solution interpolated to the fine cells (bilinear, nearest at the edges)."""
    from . import flow as FL
    f = _fv_truth_interp(g_c, T_c, cells_c)
    return f(FL.scalar_coords(g_f) * (1.0 / g_f.h) / cells_f)


def e1_cfd(n_pins=8, fine=16, coarse=4, budgets=(0.0, 0.05, 0.1, 0.2, 0.4, 0.6), Pe_b=100.0, Re=50.0,
           seed: int = 0) -> dict:
    """E1: residual-selected against random exception patches on the advection-diffusion case (Pe_b = 100) and on
    the conjugate energy field of the cold plate (Re_fin = 50); prior: the same case on a 4x coarser grid."""
    import scipy.sparse.linalg as spla

    from . import flow as FL
    out = {}
    # advection-diffusion
    sols = {}
    for c in (fine, coarse):
        g, A, b = _stokes_system((n_pins, n_pins), c, c // 2)
        u, v, _ = FL.split(g, spla.spsolve(A.tocsc(), b))
        As, bs = _scalar_case(n_pins, c, c // 2, Pe_b, u, v, g)
        sols[c] = (g, As, bs, spla.spsolve(As.tocsc(), bs))
    g, As, bs, Tt = sols[fine]
    gc, _, _, Tc = sols[coarse]
    x0 = _coarse_prior(g, fine, gc, Tc, coarse)
    Xc = FL.scalar_coords(g)
    nbs = (g.nx // fine, g.ny // fine)
    blk = (np.floor(Xc[:, 0] * 1.0 / (g.h * fine)).astype(int) * nbs[1] + np.floor(Xc[:, 1] / (g.h * fine)).astype(int))
    out["advection Pe_b=100"] = {sel: exception_patches(As, bs, x0, Tt, blk, nbs, budgets, sel, seed=seed)
                                 for sel in ("residual", "random")}
    out["advection Pe_b=100"]["prior_err_max"] = float(np.abs(x0 - Tt).max() / np.abs(Tt).max())
    # CHT energy at Re
    sols = {}
    for c in (fine, coarse):
        g = _cht_grid((n_pins, n_pins // 2), c, c // 2)
        d = 0.5
        r = FL.picard(g, d / Re, np.ones(g.ny), tol=1e-7, maxiter=150, relax=0.7)
        u, v, _ = FL.split(g, r["x"])
        Ae, be = _cht_energy(g, u, v, d, Re=Re)
        sols[c] = (g, Ae, be, spla.spsolve(Ae.tocsc(), be), r["iterations"])
    g, Ae, be, Tt, _ = sols[fine]
    gc, _, _, Tc, _ = sols[coarse]
    x0 = _coarse_prior(g, fine, gc, Tc, coarse)
    Xc = FL.scalar_coords(g)
    nbs = (g.nx // fine, g.ny // fine)
    blk = (np.floor(Xc[:, 0] / (g.h * fine)).astype(int) * nbs[1] + np.floor(Xc[:, 1] / (g.h * fine)).astype(int))
    out[f"cold-plate energy Re={Re:g}"] = {sel: exception_patches(Ae, be, x0, Tt, blk, nbs, budgets, sel, seed=seed)
                                          for sel in ("residual", "random")}
    out[f"cold-plate energy Re={Re:g}"]["prior_err_max"] = float(np.abs(x0 - Tt).max() / np.abs(Tt).max())
    return out
