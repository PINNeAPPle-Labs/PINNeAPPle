"""Condensation falsification campaign: the measured runs of docs/research/condensation_falsification.md.

Each pre-registered test is a function of :mod:`pinneapple_condense.falsify`, run in a fresh process (its peak memory
is that process's). Results go to ``results/<name>.json`` and are kept (rerun one with ``--force <name>``); a run that
fails is recorded with its error, not hidden. ``--only T1,F1`` runs a subset.

Run: ``python examples/condensation_falsification/run.py [--only ...] [--force ...]``.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from pinneapple_condense import benchmark as B

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
MOD = "pinneapple_condense.falsify"


def plan() -> dict[str, list[tuple[str, str, dict]]]:
    """Study name -> list of (result name, function, kwargs)."""
    t1 = [(f"T1_ref_{nb}", "t1_reference", {"nb": nb, "fine": nb <= 64}) for nb in (16, 32, 64, 128)]
    t1 += [(f"T1_{nb}_m{m}", "t1_point", {"nb": nb, "m": m}) for nb in (16, 32, 64, 128) for m in (6, 12, 24, 48)
           if not (nb == 128 and m == 48)]           # m = 48 at 128 x 128: dense top merges beyond 15 GB
    ms = [1, 2, 3, 4, 6, 8, 12, 16]
    cfd = {"F0": [("F0_validation", "f0_validation", {})],
           "F1": [("F1_advection", "f1_advection", {"ms": ms})],
           "F2": [("F2_stokes", "f2_stokes", {"ms": ms})],
           "F3": [("F3_cht", "f3_cht", {"ms": ms})]}
    thermal = {"T2": [("T2_contrast", "t2_contrast", {})],
               "T3": [("T3_anisotropy", "t3_anisotropy", {})],
               "T4": [("T4_no_repetition", "t4_no_repetition", {})],
               "T5": [("T5_variants", "t5_variants", {})],
               # 16 x 16 (pre-registered) ran out of memory in the separator factorisation of the generic layer (kept
               # as a measured failure); 8 x 8 is the reduced scale that fits
               "T6": [("T6_nonlinear_k", "t6_t7_nonlinear", {"case": "k"}),
                      ("T6_nonlinear_k_nb8", "t6_t7_nonlinear", {"case": "k", "nb": 8})],
               "T7": [("T7_nonlinear_R", "t6_t7_nonlinear", {"case": "R"}),
                      ("T7_nonlinear_R_nb8", "t6_t7_nonlinear", {"case": "R", "nb": 8})],
               "T8": [("T8_adversarial", "t8_adversarial", {})],
               "T9": [("T9_thin", "t9_thin", {})],
               "T10": [("T10_pitch", "t10_pitch", {}), ("T10_pitch_nb8", "t10_pitch", {"nb": 8})]}
    meshless = {"M1": [("M1_advection", "m1_advection", {})],
                "M2": [("M2_stokes", "m2_stokes", {})],
                "M3": [("M3_hybrid", "m3_hybrid", {})],
                "M4": [("M4_condensed", "m4_meshless_condensed", {})],
                "E1": [("E1_cfd", "e1_cfd", {})]}
    return {"T1": t1, **cfd, **thermal, **meshless}


def run(name, func, kw, force, timeout=7200):
    path = os.path.join(RES, f"{name}.json")
    if os.path.exists(path) and not force:
        return json.load(open(path))
    print(f"-- {name}: {func}({kw})", flush=True)
    try:
        r = B.isolated(func, timeout=timeout, module=MOD, **kw)
    except Exception as e:                       # recorded, not hidden
        r = {"failed": str(e)[-1500:]}
    json.dump(r, open(path, "w"), indent=1)
    return r


def _load(name):
    p = os.path.join(RES, f"{name}.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    return None if "failed" in r else r


def figures() -> list[str]:
    """Figures from whatever results exist."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = []
    C = {"m6": "#e08a2a", "m12": "#2a6fdb", "m24": "#1f9e77", "m48": "#7a5bd6", "ref": "#555555"}

    def save(fig, name):
        path = os.path.join(FIG, name)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        out.append(path)
    # T1 / T1b
    refs = {nb: _load(f"T1_ref_{nb}") for nb in (16, 32, 64, 128)}
    if any(refs.values()):
        fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
        for m in (6, 12, 24, 48):
            xs, ys = [], []
            for nb, r in refs.items():
                p = _load(f"T1_{nb}_m{m}")
                if r and p:
                    xs.append(nb * nb)
                    ys.append(max(abs(p["Tmax"] - r["exact"]["Tmax"]), 1e-6))
            if xs:
                a.loglog(xs, ys, "o-", color=C[f"m{m}"], label=f"m = {m}")
        a.axhline(0.1, color=C["ref"], ls="--", lw=1)
        a.text(20, 0.115, "0.1 K (T1 threshold)", fontsize=8, color=C["ref"])
        a.set_xlabel("blocks")
        a.set_ylabel("|Tmax error| vs exact [K]")
        a.legend(fontsize=8, frameon=False)
        xs = [nb * nb for nb, r in refs.items() if r]
        b.loglog(xs, [r["exact"]["seconds_to_1mK"] or np.nan for r in refs.values() if r], "o-", color=C["m12"],
                 label="exact correction to 1 mK (m = 12 preconditioner)")
        fx = [nb * nb for nb, r in refs.items() if r and "fine" in r]
        b.loglog(fx, [r["fine"]["seconds"] for r in refs.values() if r and "fine" in r], "s--", color=C["ref"],
                 label="fine FV (AMG-CG), one solve")
        b.set_xlabel("blocks")
        b.set_ylabel("wall time [s]")
        b.legend(fontsize=8, frameon=False)
        for ax in (a, b):
            ax.spines[["top", "right"]].set_visible(False)
        save(fig, "t1_far_field.png")
    # F1 / F2 / F3 reduction
    f1, f2, f3 = _load("F1_advection"), _load("F2_stokes"), _load("F3_cht")
    if f1 or f2 or f3:
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        if f1:
            for r, c in zip(f1["rows"], plt.cm.viridis(np.linspace(0, 0.9, len(f1["rows"]))), strict=True):
                rr = r["reduced"]["rows"]
                ax[0].semilogy([x["m"] for x in rr], [max(x["error"], 1e-9) for x in rr], "o-", color=c,
                               label=f"Pe_b = {r['Pe_b']:g}")
            ax[0].set_title("F1 advection-diffusion: rel. L2 error", fontsize=10)
        for k, (res, title) in enumerate(((f2, "F2 Stokes: pressure-drop error"),
                                           (f3, "F3 cold plate (Re 100): pressure-drop error")), start=1):
            if res:
                rr = res["reduced"]["rows"]
                ax[k].semilogy([x["m"] for x in rr], [max(x["error"], 1e-9) for x in rr], "o-", color=C["m12"])
                ax[k].set_title(title, fontsize=10)
        for a in ax:
            a.axhline(0.01, color=C["ref"], ls="--", lw=1)
            a.set_xlabel("modes per face m")
            a.spines[["top", "right"]].set_visible(False)
        ax[0].legend(fontsize=7, frameon=False)
        save(fig, "cfd_reduction.png")
    # M1 / M3 convergence
    m1, m3 = _load("M1_advection"), _load("M3_hybrid")
    if m1 or m3:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        if m1:
            sty = {"fv": "s-", "rbf_fd": "o-", "gfdm": "^-"}
            for pe, c in zip(sorted({r["Pe_b"] for r in m1["rows"]}), ("#2a6fdb", "#1f9e77", "#e08a2a"), strict=False):
                for meth in ("fv", "rbf_fd", "gfdm"):
                    rr = [r for r in m1["rows"] if r["Pe_b"] == pe and r["method"] == meth and "err" in r]
                    if rr:
                        ax[0].loglog([r["unknowns"] for r in rr], [r["err"] for r in rr], sty[meth], color=c,
                                     label=f"{meth}, Pe_b = {pe:g}", alpha=0.9 if meth == "fv" else 0.7)
            ax[0].set_title("M1 advection-diffusion: max error vs FV truth", fontsize=10)
            ax[0].legend(fontsize=6, frameon=False, ncol=2)
        if m3:
            for meth, sty_ in (("fv", "s-"), ("hybrid n=12 p=2", "o-"), ("hybrid n=20 p=2", "^--"),
                               ("hybrid n=20 p=3", "v:")):
                rr = [r for r in m3["rows"] if r["method"] == meth]
                if rr:
                    ax[1].loglog([r["unknowns"] for r in rr], [r["err"] for r in rr], sty_, label=meth)
            ax[1].set_title("M3 multi-material conduction: max error", fontsize=10)
            ax[1].legend(fontsize=7, frameon=False)
        for a in ax:
            a.set_xlabel("unknowns")
            a.spines[["top", "right"]].set_visible(False)
        save(fig, "meshless_convergence.png")
    # E1
    e1 = _load("E1_cfd")
    if e1:
        cases = [k for k in e1 if isinstance(e1[k], dict) and "residual" in e1[k]]
        fig, ax = plt.subplots(1, len(cases), figsize=(5.5 * len(cases), 3.8))
        for a, k in zip(np.atleast_1d(ax), cases, strict=True):
            for sel, c in (("residual", "#2a6fdb"), ("random", "#8a8a8a")):
                rr = e1[k][sel]
                a.semilogy([r["solved_fraction"] for r in rr], [r["err_l2"] for r in rr], "o-", color=c, label=sel)
            a.set_title(f"E1 {k}", fontsize=10)
            a.set_xlabel("fraction of unknowns solved")
            a.set_ylabel("rel. L2 error")
            a.legend(fontsize=8, frameon=False)
            a.spines[["top", "right"]].set_visible(False)
        save(fig, "e1_exceptions.png")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--force", default="")
    ap.add_argument("--figures", action="store_true", help="only draw the figures from results/")
    a = ap.parse_args()
    if a.figures:
        print("\n".join(figures()))
        return
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    only = set(filter(None, a.only.split(",")))
    force = set(filter(None, a.force.split(",")))
    for study, items in plan().items():
        if only and study not in only:
            continue
        for name, func, kw in items:
            run(name, func, kw, "all" in force or study in force or name in force)
    figures()


if __name__ == "__main__":
    main()
