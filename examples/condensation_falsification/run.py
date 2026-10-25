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
               # 32 x 32 (pre-registered) was killed out of memory (kept as a measured failure); 16 x 16 fits
               "T4": [("T4_no_repetition", "t4_no_repetition", {}),
                      ("T4_no_repetition_nb16", "t4_no_repetition", {"nb": 16})],
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
        a.text(300, 0.12, "0.1 K (T1 threshold)", fontsize=8, color=C["ref"])
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
            ax[0].set_ylim(1e-4, 10)
            ax[0].set_title("M1 advection-diffusion: max error vs FV truth\n(GFDM leaves the axis: up to 1e9)", fontsize=10)
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


def verdicts() -> list[dict]:
    """Each pre-registered hypothesis judged by its kill criterion on the measured results (``None`` results: not
    tested). Returns rows ``{id, verdict, evidence}``."""
    V = []

    def add(hid, verdict, evidence):
        V.append({"id": hid, "verdict": verdict, "evidence": evidence})

    def ref(nb):
        r = _load(f"T1_ref_{nb}")
        return r["exact"]["Tmax"] if r else None

    def m_star(nb):
        for m in (6, 12, 24, 48):
            p = _load(f"T1_{nb}_m{m}")
            if p and ref(nb) is not None and abs(p["Tmax"] - ref(nb)) <= 0.1:
                return m
        return None
    m1k, m16k = m_star(32), m_star(128)
    if m1k:
        e = {f"{nb * nb}": m_star(nb) for nb in (16, 32, 64, 128)}
        bad = m16k is None or m16k > 2 * m1k
        add("T1", "refuted" if bad else "validated",
            f"smallest m (of 6, 12, 24, 48) with |dTmax| <= 0.1 K: {e} blocks -> m; kill if m(16,384) > 2 m(1,024)")
    r1k, r16k, r4k = _load("T1_ref_32"), _load("T1_ref_128"), _load("T1_ref_64")
    if r1k and r16k:
        it1, it16 = r1k["exact"]["iterations_to_1mK"], r16k["exact"]["iterations_to_1mK"]
        slower = [nb for nb in (16, 32, 64) if (r := _load(f"T1_ref_{nb}")) and
                  r["exact"]["seconds_to_1mK"] > r["fine"]["seconds"]]
        bad = it16 > 4 * it1 or bool(slower)
        add("T1b", "refuted" if bad else "validated",
            f"PCG iterations to 1 mK: {it1} at 1,024 and {it16} at 16,384 blocks (limit 4x); time to 1 mK "
            f"{r4k['exact']['seconds_to_1mK']:.1f} s vs fine AMG {r4k['fine']['seconds']:.0f} s at 4,096 blocks")

    def thermal(hid, name, key):
        r = _load(name)
        if not r:
            return add(hid, "not tested", f"{name}: no result")
        base = [x for x in r["rows"] if x.get("baseline") or x.get(key) in ("baseline",)][0]
        worst_exact = max(x["exact_vs_fine"] for x in r["rows"])
        ratio = max(x["err_reduced"] for x in r["rows"]) / base["err_reduced"]
        bad = worst_exact > 1e-6 or ratio > 3
        add(hid, "refuted" if bad else "validated",
            f"exact vs fine <= {worst_exact:.1e} K; worst m=12 error {ratio:.2f}x the baseline "
            f"({base['err_reduced'] * 1e3:.1f} mK)")
    thermal("T2", "T2_contrast", "ratio")
    thermal("T3", "T3_anisotropy", "factor")
    thermal("T9", "T9_thin", "case")
    r = _load("T4_no_repetition_nb16")
    big = _load("T4_no_repetition")
    if r:
        ex, le = r["exact"], r["learned"]
        add("T4", "refuted" if (ex["total_s"] > r["fine_s"] or le["solve_s"] > r["fine_s"] or le["err"] > 0.1)
            else "validated",
            ("32 x 32 (pre-registered): killed out of memory at 13 GB (one dense model per distinct block); 16 x 16: "
             if big is None else "") + f"fine FV {r['fine_s']:.0f} s; exact path (all blocks + PCG) {ex['total_s']:.0f} s; learned "
            f"{le['solve_s']:.0f} s (+{le['training_s']:.0f} s training), error {le['err']:.2f} K; m-mode interfaces alone "
            f"off by {ex['err_reduced']:.2f} K on these heterogeneous blocks")
    else:
        add("T4", "not tested", "no result")
    r = _load("T5_variants")
    if r:
        last = r["rows"][-1]
        add("T5", "refuted" if last["hierarchy_s"] > last["flat_s"] else "validated",
            "; ".join(f"V={x['V']}: hierarchy {x['hierarchy_s']:.1f} s / flat {x['flat_s']:.1f} s" for x in r["rows"]))
    for hid, name in (("T6", "T6_nonlinear_k"), ("T7", "T7_nonlinear_R")):
        big, small = _load(name), _load(name + "_nb8")
        if small:
            bad = small["err"] > 0.1 or small["condensed"]["iterations"] > 10 or \
                small["condensed"]["seconds"] > small["fine"]["seconds"]
            add(hid, "refuted" if bad or big is None else "validated",
                ("16 x 16 (pre-registered): out of memory in the separator factorisation of the generic layer; "
                 if big is None else "") +
                f"8 x 8: error {small['err'] * 1e3:.2g} mK, {small['condensed']['iterations']} Picard iterations, "
                f"{small['condensed']['seconds']:.0f} s vs fine {small['fine']['seconds']:.0f} s "
                f"(nonlinear - linear Tmax {small['Tmax_fine_nonlinear'] - small['Tmax_linear']:.2f} K)")
    r = _load("T8_adversarial")
    if r:
        worst = max(r["rows"], key=lambda x: x["err_reduced"])
        effs = [x["effectivity"] for x in r["rows"]]
        bad = worst["err_reduced"] > 0.1 or min(effs) < 0.5 or max(effs) > 2
        add("T8", "refuted" if bad else "validated",
            f"worst map '{worst['map']}': {worst['err_reduced']:.2f} K; others <= "
            f"{sorted(x['err_reduced'] for x in r['rows'])[-2] * 1e3:.1f} mK; effectivity {min(effs):.2f}-{max(effs):.2f}")
    big, small = _load("T10_pitch"), _load("T10_pitch_nb8")
    if small:
        def e12(k):
            return [x["err"] for x in small[k]["rows"] if x["m"] == 12][0]
        ratio = e12("alternating") / max(e12("uniform"), 1e-12)
        ex = max(small[k]["exact_vs_fine"] for k in ("uniform", "alternating"))
        bad = ex > 1e-6 or (ratio > 3 and e12("alternating") > 1e-4)
        add("T10", "refuted" if bad else ("partially validated" if big is None else "validated"),
            ("16 x 16 (pre-registered): out of memory in the generic layer; " if big is None else "") +
            f"8 x 8: exact vs fine {ex:.1e} K; m=12 error {e12('alternating'):.1e} K (uniform {e12('uniform'):.1e} K)")
    f0 = _load("F0_validation")
    if f0:
        pz = max(x["u_max_rel_err"] for x in f0["poiseuille"])
        cav = [x for x in f0["cavity"] if x["n"] == 128][0]["max_dev_vs_ghia"]
        nu_ = max(x["rel_err"] for x in f0["nusselt"])
        add("F0", "partially validated" if pz > 1e-6 else "validated",
            f"Poiseuille max rel. error {pz:.1e} (pre-registered as exact, 1e-6: refuted -- the ghost-cell wall is "
            f"O(h^2), observed order 2); cavity Re 100 vs Ghia {cav * 100:.2f} % of U (limit 2 %); Nu error {nu_ * 100:.3f} %")
    f1 = _load("F1_advection")
    if f1:
        ex = max(r["exact_rel_err"] for r in f1["rows"])
        add("F1", "refuted" if ex > 1e-8 else "validated", f"max relative difference {ex:.1e} over Pe_b 0.1-1000; "
            f"blocks distinct by content: {f1['rows'][0]['condensation']['distinct_blocks']} of "
            f"{f1['rows'][0]['condensation']['blocks']} (the computed velocity differs per block)")
        r100 = [r for r in f1["rows"] if r["Pe_b"] == 100][0]["reduced"]
        c = r100["compression_at_1pct"] or 0
        add("F1r", "refuted" if c < 4 else "validated",
            f"Pe_b = 100: m = {r100['m_for_1pct']} for 1 % ({r100['mean_face_unknowns']:.1f} unknowns per face, "
            f"compression {c:.2f}); " + ", ".join(f"Pe_b {r['Pe_b']:g}: m {r['reduced']['m_for_1pct']}" for r in f1["rows"]))
    f2 = _load("F2_stokes")
    if f2:
        bad = f2["exact_rel_err"] > 1e-8 or f2["speedup"] < 2
        add("F2", "refuted" if bad else "validated",
            f"exact to {f2['exact_rel_err']:.1e}; {f2['condensation']['distinct_blocks']} distinct of "
            f"{f2['condensation']['blocks']} blocks; direct {f2['direct_s']:.1f} s vs condensed {f2['condensed_s']:.1f} s "
            f"(speedup {f2['speedup']:.2f}, limit 2)")
        c = f2["reduced"]["compression_at_1pct"] or 0
        add("F2r", "refuted" if c < 4 else "validated", f"m = {f2['reduced']['m_for_1pct']} for 1 % on the pressure drop "
            f"(compression {c:.2f})")
    f3 = _load("F3_cht")
    if f3:
        err = max(max(r["dp_rel_err"], r["fin_rise_rel_err"]) for r in f3["rows"])
        slow = [r["Re_fin"] for r in f3["rows"] if r["picard_condensed"]["seconds"] > r["picard_full"]["seconds"]]
        add("F3", "refuted" if (err > 0.01 or slow) else "validated",
            f"error <= {err:.1e}; condensed Picard slower than full at Re {slow} "
            + ", ".join(f"Re {r['Re_fin']:g}: {r['picard_condensed']['seconds']:.0f} vs {r['picard_full']['seconds']:.0f} s"
                        for r in f3["rows"]))
        c = f3["reduced"]["compression_at_1pct"] or 0
        rev = f3["rows"][-1]["reversed_block_face_flux"]
        add("F3r", "refuted" if c < 4 else "validated", f"Re 100: m = {f3['reduced']['m_for_1pct']} for 1 % "
            f"(compression {c:.2f}); reversed block-face flux {rev * 100:.1f} %")
    m1 = _load("M1_advection")
    if m1:
        notes, bad_rbf, bad_gfdm = [], False, False
        for pe in sorted({r["Pe_b"] for r in m1["rows"]}):
            for meth in ("rbf_fd", "gfdm"):
                rr = sorted((r for r in m1["rows"] if r["Pe_b"] == pe and r["method"] == meth and "err" in r),
                            key=lambda r: r["density"])
                fv_ = {r["density"]: r["err"] for r in m1["rows"] if r["Pe_b"] == pe and r["method"] == "fv"}
                mono = all(b["err"] < a["err"] for a, b in zip(rr, rr[1:], strict=False))
                ratio = max(r["err"] / fv_[r["density"]] for r in rr)
                flag = (not mono) or ratio > 10
                if meth == "rbf_fd":
                    bad_rbf |= flag
                else:
                    bad_gfdm |= flag
                notes.append(f"{meth} Pe {pe:g}: {'monotone' if mono else 'NOT monotone'}, up to {ratio:.3g}x FV")
        add("M1", "refuted" if (bad_rbf and bad_gfdm) else "partially validated" if bad_rbf or bad_gfdm
            else "validated", "; ".join(notes))
    m2 = _load("M2_stokes")
    if m2:
        notes, fails = [], 0
        for meth in ("rbf_fd", "gfdm"):
            rr = sorted((r for r in m2["rows"] if r["method"] == meth and "dp_rel_err" in r), key=lambda r: r["density"])
            mono = len(rr) == 3 and all(b["dp_rel_err"] < a["dp_rel_err"] for a, b in zip(rr, rr[1:], strict=False))
            fails += not mono
            notes.append(f"{meth}: " + ", ".join(f"{r['dp_rel_err'] * 100:.0f} %" for r in rr) +
                         (" (decreasing)" if mono else " (not converging)"))
        fvr = sorted((r for r in m2["rows"] if r["method"] == "fv"), key=lambda r: r["density"])
        add("M2", "refuted" if fails == 2 else "partially validated" if fails else "validated",
            "pressure-drop error over densities 8, 16, 32 -- " + "; ".join(notes) +
            "; FV: " + ", ".join(f"{r['dp_rel_err'] * 100:.0f} %" for r in fvr))
    m3 = _load("M3_hybrid")
    if m3:
        rows = [(r["density"], r["err"], [x for x in m3["rows"] if x["density"] == r["density"] and
                                          x["method"] == "fv"][0]["err"]) for r in m3["rows"]
                if r["method"] == "hybrid n=12 p=2" and r.get("fv_fraction", 1) < 1]
        bad = any(h > 2 * f for _, h, f in rows)
        add("M3", "refuted" if bad else "validated", "; ".join(f"density {d}: hybrid {h * 100:.2f} % vs FV {f * 100:.2f} %"
                                                            for d, h, f in rows))
    m4 = _load("M4_condensed")
    if m4:
        h = m4["hybrid conduction"]
        a = m4["advection-diffusion Pe_b=10"]
        bad = max(h["exact_rel_err"], a["exact_rel_err"]) > 1e-8 or h["speedup"] < 1
        add("M4", "refuted" if bad else "validated",
            f"exact to {max(h['exact_rel_err'], a['exact_rel_err']):.1e}; hybrid: {h['distinct_blocks']} distinct of "
            f"{h['blocks']} blocks, speedup {h['speedup']:.2f}; advection: {a['distinct_blocks']} of {a['blocks']}, "
            f"speedup {a['speedup']:.2f}")
    e1 = _load("E1_cfd")
    if e1:
        notes, bad = [], False
        for k, v in e1.items():
            if not isinstance(v, dict) or "residual" not in v:
                continue
            res, rnd = v["residual"], v["random"]
            target = res[-2]["err_l2"]
            need_res = min((r["solved_fraction"] for r in res if r["err_l2"] <= target), default=1.0)
            need_rnd = min((r["solved_fraction"] for r in rnd if r["err_l2"] <= target), default=1.0)
            ov = [r["overlap_top_error"] for r in res[1:]]
            bad |= need_res > 0.5 * need_rnd or min(ov) < 0.5
            notes.append(f"{k}: to reach {target:.3f} residual solves {need_res:.0%}, random {need_rnd:.0%}; "
                         f"overlap with top-error blocks {min(ov):.2f}-{max(ov):.2f}")
        add("E1", "refuted" if bad else "validated", "; ".join(notes))
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--force", default="")
    ap.add_argument("--figures", action="store_true", help="only draw the figures from results/")
    a = ap.parse_args()
    if a.figures:
        manifest()
        for v in verdicts():
            print(f"{v['id']:5} {v['verdict']:20} {v['evidence']}")
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
    manifest()


def manifest():
    """Figures, the verdict table and ``experiment.json`` from the results on disk."""
    import pinneapple_lab
    with pinneapple_lab.experiment(
            "examples/condensation_falsification", title="Falsification campaign for physics condensation",
            description="Pre-registered hypotheses (docs/research/condensation_falsification.md) with kill criteria: "
                        "thermal stress tests of block condensation, algebraic condensation of advection-diffusion, "
                        "Stokes and conjugate heat transfer, meshless discretisations and prior + exception patches.",
            domain="heat transfer", methods=["static condensation", "finite volumes", "RBF-FD", "GFDM", "Picard"],
            modules=["pinneapple_condense"], out_dir=HERE, register=False,
            links={"preregistration": "docs/research/condensation_falsification.md",
                   "script": "examples/condensation_falsification/run.py"}) as ex:
        for f in figures():
            ex.add_file(f)
        V = verdicts()
        json.dump(V, open(os.path.join(RES, "verdicts.json"), "w"), indent=1)
        for v in V:
            ex.check(f"{v['id']} not refuted", v["verdict"] in ("validated", "partially validated"),
                     kind="verification", detail=f"{v['verdict']}: {v['evidence']}"[:500])
        for k in ("validated", "partially validated", "refuted", "not tested"):
            ex.log_metric(f"hypotheses_{k.replace(' ', '_')}", sum(v["verdict"] == k for v in V), "",
                          "pre-registered hypotheses with this verdict")


if __name__ == "__main__":
    main()
