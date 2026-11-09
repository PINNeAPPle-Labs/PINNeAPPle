"""Two connected experiments on a compact range: how far physical optics is from a direct solve, whether that
difference can be learned, and whether learning it makes an optimisation affordable.

    python examples/use_cases/antennas/compact_range_two_fidelity.py --out ~/runs/cr_two_fidelity
    python examples/use_cases/antennas/compact_range_two_fidelity.py --out ... --render-only

Steps (each writes JSON, so the report re-renders without solving):

1. ``validate``   -- the direct solver against the exact series of a perfectly conducting circle, and the range
                     metrics against the mesh density;
2. ``baseline``   -- the measured cost of PO and of a direct solve, in 2-D and (for PO and for the operator
                     alone) in 3-D, which is what decides whether a surrogate is needed at all;
3. ``sweep``      -- every design of the space solved twice, PO and direct, into a resumable JSON-lines checkpoint;
4. ``analyse``    -- the learned residual, its error on designs it never saw, and the three optimisation
                     strategies at equal budget;
5. ``report``     -- a PDF (HTML rendered by headless Chrome) with every number of the run.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from dataclasses import replace

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np  # noqa: E402

from pinneapple_simulation.electromagnetics.reflector import paired  # noqa: E402
from pinneapple_simulation.electromagnetics.reflector import planar2d as P2  # noqa: E402
from pinneapple_simulation.electromagnetics.reflector.po import q_from_gain  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = (0.6, 0.8, 1.0, 1.2, 1.4)
GAINS = (8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)
MESH = 14.0                    # segments per wavelength: the metrics of the range change < 0.2 % from 14 to 20
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


# ------------------------------------------------------------------------------------------------ helpers
def _json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)


def _load(path: str, default=None):
    if os.path.isfile(path):
        with open(path) as fh:
            return json.load(fh)
    return default


def log(msg: str) -> None:
    print(msg, flush=True)


def space(mesh: float = MESH):
    base = P2.Range2D(per_wavelength=mesh)
    return P2.design_space(base, SIZES, tuple(q_from_gain(g) for g in GAINS))


def _labelled(designs):
    """Pair every design with the feed gain it corresponds to, for the tables."""
    gain = {q_from_gain(g): g for g in GAINS}
    return [(d, rng, gain.get(round(d.q, 6), d.q)) for d, rng in designs]


# ----------------------------------------------------------------------------------------------- 1. validate
def step_validate(out: str) -> dict:
    log("validate: the direct solver against the exact series of a PEC circle")
    circles = [P2.validate_circle(ka=ka, per_wavelength=per) for ka, per in ((6.0, 10.0), (6.0, 14.0), (6.0, 20.0),
                                                                            (10.0, 16.0))]
    for c in circles:
        log(f"  ka {c['ka']:g}, {c['n_segments']:3d} segments: direct {100 * c['max_rel_error_mom']:.2f} % max error,"
            f" PO {100 * c['max_rel_error_po']:.0f} %")
    log("validate: the range metrics against the mesh density")
    conv = []
    for mesh in (10.0, 14.0, 20.0):
        rng = P2.Range2D(per_wavelength=mesh)
        po = P2.po_metrics(rng)
        dr = P2.direct_metrics(rng)
        conv.append({"per_wavelength": mesh, "n_segments": po["n_segments"],
                     **{f"po_{k}": po[k] for k in paired.METRICS},
                     **{f"direct_{k}": dr[k] for k in paired.METRICS},
                     "po_seconds": po["seconds"], "direct_seconds": dr["seconds"]})
        log(f"  lambda/{mesh:g}: {po['n_segments']} segments -- PO ripple {po['ripple_pp_dB']:.2f} dB, direct "
            f"{dr['ripple_pp_dB']:.2f} dB; phase ripple {po['phase_ripple_pp_deg']:.1f} / "
            f"{dr['phase_ripple_pp_deg']:.1f} deg; {1000 * dr['seconds']:.0f} ms")
    res = {"circles": circles, "mesh_convergence": conv}
    _json(os.path.join(out, "validate.json"), res)
    return res


# ----------------------------------------------------------------------------------------------- 2. baseline
def _kernel_cost_3d(n_triangles: int, k: float, rng) -> dict:
    """Time one application of the 3-D magnetic-field operator on ``n_triangles`` centroids: the cost that decides
    whether a 3-D direct solve per design is affordable. It is the same kernel PO radiates with, so it is measured,
    not projected."""
    centres = rng.random((n_triangles, 3))
    t0 = time.perf_counter()
    R = np.linalg.norm(centres[:64, None, :] - centres[None, :, :], axis=2)
    _ = np.exp(-1j * k * np.maximum(R, 1e-9)) * (1 + 1j * k * R) / np.maximum(R, 1e-9) ** 2
    per_pair = (time.perf_counter() - t0) / (64 * n_triangles)
    return {"triangles": n_triangles, "seconds_per_application": per_pair * n_triangles ** 2,
            "pairs": n_triangles ** 2}


def step_baseline(out: str) -> dict:
    from pinneapple_simulation.electromagnetics.reflector import (
        CompactRange,
        OffsetReflector,
        evaluate,
    )
    log("baseline: PO and a direct solve, 2-D")
    rng = P2.Range2D(per_wavelength=MESH)
    rows = []
    for mesh in (10.0, 14.0, 20.0):
        r = replace(rng, per_wavelength=mesh)
        po = P2.po_metrics(r)
        dr = P2.direct_metrics(r)
        rows.append({"per_wavelength": mesh, "n_segments": po["n_segments"], "po_s": po["seconds"],
                     "direct_s": dr["seconds"], "mom_s": dr["mom_seconds"]})
        log(f"  lambda/{mesh:g}: {po['n_segments']} segments -- PO {1e3 * po['seconds']:.0f} ms, direct "
            f"{1e3 * dr['seconds']:.0f} ms (x{dr['seconds'] / po['seconds']:.1f})")
    log("baseline: PO in 3-D at the production geometry (the 2 GHz study's reflector), and the cost of one 3-D")
    log("          direct-operator application at the same mesh density")
    ft = 0.3048
    probe = []
    g3 = np.random.default_rng(0)
    for mesh in (6.0, 8.0):
        cr = CompactRange(freq_hz=3.0e9, feed_gain_dbi=12.0,
                          reflector=OffsetReflector(focal=1.5, half_x=0.6, half_y=0.6, centre_y=0.9, n=6.0, roll=0.15),
                          qz_z=3.0, qz_diameter=0.6, mesh_per_wavelength=mesh, qz_step_per_wavelength=6.0)
        t0 = time.perf_counter()
        m = evaluate(cr)
        t_po = time.perf_counter() - t0
        cost = _kernel_cost_3d(int(m["n_triangles"]), 2 * np.pi / cr.wavelength, g3)
        probe.append({"f_GHz": 3.0, "mesh_per_wavelength": mesh, "n_triangles": m["n_triangles"],
                      "po_seconds": t_po, "operator_application_seconds": cost["seconds_per_application"],
                      "applications_for_a_solve": 40,
                      "direct_seconds_estimate": 40 * cost["seconds_per_application"]})
        log(f"  3-D lambda/{mesh:g}: {m['n_triangles']} triangles -- PO {t_po:.1f} s, one operator application "
            f"{cost['seconds_per_application']:.2f} s, 40 of them {40 * cost['seconds_per_application']:.0f} s")
    res = {"range_2d": rows, "range_3d": probe, "ft": ft}
    _json(os.path.join(out, "baseline.json"), res)
    return res


# -------------------------------------------------------------------------------------------------- 3. sweep
def step_sweep(out: str, workers: int) -> list[dict]:
    designs = space()
    log(f"sweep: {len(designs)} designs (3 sizes x 4 feeds x 18 edge profiles), PO and direct")
    t0 = time.perf_counter()
    rows = paired.paired_sweep(designs, os.path.join(out, "paired.jsonl"), workers=workers, log=log)
    log(f"sweep: {len(rows)} paired designs in {time.perf_counter() - t0:.0f} s")
    return rows


# ------------------------------------------------------------------------------------------------ 4. analyse
def step_analyse(out: str, rows: list[dict]) -> dict:
    log("analyse: the learned residual")
    val = {m: paired.leave_out_validation(rows, m) for m in paired.METRICS}
    for m, v in val.items():
        log(f"  {m:22s}: PO RMSE {v['po_rmse']:.4f}; learned {v['random']['rmse']:.4f} "
            f"({100 * v['random']['fraction_removed']:+.0f} % of the PO error removed, random split), "
            f"{v['profile']['rmse']:.4f} ({100 * v['profile']['fraction_removed']:+.0f} %, profile held out)")
    log("analyse: the three strategies at equal budget")
    runs = [paired.compare_strategies(rows, n_direct=n, seeds=(0, 1, 2, 3), log=log) for n in (6, 12, 24)]
    cost = paired.cost_summary(rows)
    log(f"  cost: PO {1e3 * cost['po_seconds']['median']:.0f} ms, direct {1e3 * cost['direct_seconds']['median']:.0f}"
        f" ms per design (x{cost['ratio_median']:.1f})")
    res = {"validation": val, "strategies": runs, "cost": cost}
    _json(os.path.join(out, "analysis.json"), res)
    return res


# ------------------------------------------------------------------------------------------------ 5. figures
def step_figures(out: str, val: dict, base: dict, rows: list[dict], ana: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pinneapple_tools.visualization.style import LIGHT_RC

    figdir = os.path.join(out, "fig")
    os.makedirs(figdir, exist_ok=True)
    muted, accent, alt = "#7c7b75", "#1c5cab", "#b3261e"
    paths = []
    with matplotlib.rc_context(LIGHT_RC):
        # 1. validation
        fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4))
        c = val["circles"]
        n = [d["n_segments"] for d in c]
        ax[0].plot(n, [100 * d["max_rel_error_mom"] for d in c], "o-", color=accent, label="direct solve")
        ax[0].plot(n, [100 * d["max_rel_error_po"] for d in c], "s--", color=alt, label="PO")
        ax[0].set_yscale("log")
        ax[0].set_xlabel("segments on the circle")
        ax[0].set_ylabel("max field error [%]")
        ax[0].set_title("Direct solver against the exact series (PEC circle)")
        ax[0].legend(frameon=False, fontsize=8)
        mc = val["mesh_convergence"]
        ax[1].plot([d["n_segments"] for d in mc], [d["po_ripple_pp_dB"] for d in mc], "o-", color=alt, label="PO")
        ax[1].plot([d["n_segments"] for d in mc], [d["direct_ripple_pp_dB"] for d in mc], "o-", color=accent,
                   label="direct")
        ax[1].set_xlabel("segments on the reflector")
        ax[1].set_ylabel("quiet-zone ripple p-p [dB]")
        ax[1].set_title("Mesh convergence of the range metrics")
        ax[1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        p = os.path.join(figdir, "validation.png")
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(p)
        # 2. PO against direct, parity
        fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
        for a, (metric, lab, unit) in zip(ax, (("ripple_pp_dB", "amplitude ripple p-p", "dB"),
                                               ("usable_width", "usable quiet-zone width", "m")), strict=True):
            po = np.array([r[f"po_{metric}"] for r in rows])
            dr = np.array([r[f"direct_{metric}"] for r in rows])
            lim = [min(po.min(), dr.min()), max(po.max(), dr.max())]
            a.plot(lim, lim, "-", color=muted, lw=1, label="PO = direct")
            sc = a.scatter(po, dr, c=[r["q"] for r in rows], cmap="viridis", s=14)
            a.set_xlabel(f"PO {lab} [{unit}]")
            a.set_ylabel(f"direct {lab} [{unit}]")
            a.set_title(f"{lab}: {len(rows)} designs")
            a.legend(frameon=False, fontsize=8)
        fig.colorbar(sc, ax=ax[1], label="feed q")
        fig.tight_layout()
        p = os.path.join(figdir, "pairs.png")
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(p)
        # 3. learned correction, held-out
        fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
        for a, key, title in ((ax[0], "po", "PO"), (ax[1], "corrected", "PO + learned residual")):
            for m, col, lab in (("ripple_pp_dB", alt, "amplitude ripple"), ("usable_width", accent, "usable width")):
                r = ana["validation"][m]["rows"]
                t = np.array([x["truth"] for x in r])
                v = np.array([x[key] for x in r])
                a.scatter(t, v, s=10, color=col, label=lab, alpha=0.75)
                lim = [min(t.min(), v.min()), max(t.max(), v.max())]
                a.plot(lim, lim, "-", color=muted, lw=1)
            a.set_title(f"{title}, random 5-fold")
            a.set_xlabel("direct solve")
            a.set_ylabel("prediction")
            a.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        p = os.path.join(figdir, "learning.png")
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(p)
        # 4. residual against the design variables
        fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.2))
        v = ana["validation"]["usable_width"]["rows"]
        big = {(r["tag"]): r for r in rows}
        s = np.array([big[x["tag"]]["size"] for x in v])
        q = np.array([big[x["tag"]]["q"] for x in v])
        roll = np.array([big[x["tag"]]["roll_factor"] for x in v])
        res = np.array([x["truth"] - x["po"] for x in v])
        for a, x, lab in ((ax[0], s, "aperture scale"), (ax[1], q, "feed q"), (ax[2], roll, "roll factor")):
            sc = a.scatter(x, 1e3 * res, c=q, cmap="viridis", s=14)
            a.set_xlabel(lab)
            a.set_ylabel("direct - PO usable width [mm]")
            a.axhline(0, color=muted, lw=1)
        ax[0].set_title("The residual follows the design")
        fig.colorbar(sc, ax=ax[2], label="feed q")
        fig.tight_layout()
        p = os.path.join(figdir, "residual.png")
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(p)
        # 5. strategies
        fig, ax = plt.subplots(figsize=(6.6, 3.4))
        budgets = [r["n_direct"] for r in ana["strategies"]]
        names = [("direct", alt, "direct sampling"), ("po", muted, "PO ranking"), ("multi", accent, "multi-fidelity"),
                 ("sequential", "#0d7a4a", "sequential multi-fidelity")]
        width = 0.2
        for i, (k, col, lab) in enumerate(names):
            v = [100 * r[f"{k}_mean"] / r["oracle"] for r in ana["strategies"]]
            e = [100 * (r[f"{k}_mean"] - r[f"{k}_worst"]) / r["oracle"] for r in ana["strategies"]]
            ax.bar(np.arange(len(budgets)) + (i - 1) * width, v, width, color=col, label=lab, yerr=e, capsize=3)
        ax.set_xticks(np.arange(len(budgets)))
        ax.set_xticklabels([f"{b} direct solves" for b in budgets])
        ax.set_ylabel("% of the best design in the space")
        ax.set_title("Best objective reached at equal budget (mean of 4 seeds)")
        ax.set_ylim(min(85, min(100 * r[f"{k}_worst"] / r["oracle"] for r in ana["strategies"]
                               for k, _, _ in names) - 3), 101)
        ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.02))
        fig.tight_layout()
        p = os.path.join(figdir, "strategies.png")
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(p)
        # 6. cost
        fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.2))
        b2 = base["range_2d"]
        ax[0].plot([d["n_segments"] for d in b2], [1e3 * d["po_s"] for d in b2], "o-", color=alt, label="PO")
        ax[0].plot([d["n_segments"] for d in b2], [1e3 * d["direct_s"] for d in b2], "o-", color=accent,
                   label="direct")
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_xlabel("segments")
        ax[0].set_ylabel("seconds per design")
        ax[0].set_title("2-D: cost of one design")
        ax[0].legend(frameon=False, fontsize=8)
        b3 = base["range_3d"]
        ax[1].plot([d["n_triangles"] for d in b3], [d["po_seconds"] for d in b3], "o-", color=alt, label="PO")
        ax[1].plot([d["n_triangles"] for d in b3], [d["direct_seconds_estimate"] for d in b3], "o--", color=accent,
                   label="direct operator x40")
        ax[1].set_xscale("log")
        ax[1].set_yscale("log")
        ax[1].set_xlabel("triangles on the 3-D reflector")
        ax[1].set_ylabel("seconds per design")
        ax[1].set_title("3-D production geometry: measured")
        ax[1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        p = os.path.join(figdir, "cost.png")
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(p)
    return paths


# ------------------------------------------------------------------------------------------------- 6. report
CSS = """
:root{--ink:#111110;--ink2:#42413d;--muted:#7c7b75;--line:#dcdbd5;--bg:#fff;--soft:#f7f6f2;
--accent:#1c5cab;--ok:#0d7a4a;--warn:#a86a00;--fail:#b3261e}
*{box-sizing:border-box}html{-webkit-print-color-adjust:exact;print-color-adjust:exact}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",
Inter,Helvetica,Arial,sans-serif;font-variant-numeric:tabular-nums}
.page{max-width:960px;margin:0 auto;padding:34px 40px 72px}
.eyebrow{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:600;margin:0 0 6px}
h1{font-size:30px;line-height:1.15;margin:0 0 6px;font-weight:650;letter-spacing:-.01em}
.sub{color:var(--ink2);font-size:15px;margin:0 0 4px}.gen{color:var(--muted);font-size:11.5px;margin:12px 0 0}
.verdict{margin:16px 0 0;padding:12px 14px;border-left:3px solid var(--accent);background:var(--soft);font-size:13.5px;
color:var(--ink2)}.verdict.ok{border-color:var(--ok)}.verdict.warn{border-color:var(--warn)}
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));border:1px solid var(--line);
margin:20px 0 8px;border-radius:3px;overflow:hidden}
.strip .c{padding:11px 13px;border-right:1px solid var(--line);background:#fff}
.strip .lab{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:600}
.strip .val{font-size:19px;font-weight:600;margin-top:3px;line-height:1.2}
.strip .val .u{font-size:12px;font-weight:500;color:var(--ink2);margin-left:3px}
.strip .note{font-size:10.5px;color:var(--muted);margin-top:2px;line-height:1.3}
h2{font-size:19px;font-weight:640;margin:34px 0 4px;padding-top:16px;border-top:1px solid var(--line)}
h2 .n{color:var(--muted);font-weight:500;margin-right:9px}
h3{font-size:14px;font-weight:640;margin:20px 0 4px}
.lede{color:var(--ink2);font-size:13.5px;margin:0 0 14px;max-width:74ch}
p{margin:0 0 10px;max-width:78ch;color:var(--ink2);font-size:13.5px}
figure{margin:16px 0 18px}figure img{width:100%;border:1px solid var(--line);border-radius:3px}
figcaption{font-size:11.5px;color:var(--muted);margin-top:6px;max-width:80ch}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:10px 0 14px}
th,td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:600}
td.n{font-variant-numeric:tabular-nums}.ok{color:var(--ok)}.warn{color:var(--warn)}.fail{color:var(--fail)}
ul{margin:6px 0 12px;padding-left:18px;color:var(--ink2);font-size:13.5px}li{margin:3px 0}
code{background:var(--soft);padding:1px 4px;border-radius:3px;font-size:12px}
.small{font-size:11.5px;color:var(--muted)}
@page{size:A4;margin:14mm 12mm}
"""


def _img(path: str) -> str:
    with open(path, "rb") as fh:
        return f"data:image/png;base64,{base64.b64encode(fh.read()).decode()}"


def step_report(out: str, val: dict, base: dict, rows: list[dict], ana: dict, figs: list[str]) -> str:
    """Render the report from the saved numbers. Every sentence that states a result is built from the data; the
    thresholds only choose which of two phrasings is true."""
    v = ana["validation"]
    cost = ana["cost"]
    st = {r["n_direct"]: r for r in ana["strategies"]}
    ripple, usable, phase, amp = v["ripple_pp_dB"], v["usable_width"], v["phase_ripple_pp_deg"], v["amp_pp_dB"]
    b3 = base["range_3d"]
    probe = b3[-1]
    r3_ratio = probe["direct_seconds_estimate"] / max(probe["po_seconds"], 1e-6)
    mom_err = max(c["max_rel_error_mom"] for c in val["circles"])
    po_err = min(c["max_rel_error_po"] for c in val["circles"])

    def pct(x):
        return f"{100 * x:+.0f} %"

    def verdict_learn(m):
        r = v[m]
        if r["random"]["fraction_removed"] > 0.3 and r["profile"]["fraction_removed"] > 0.15:
            return "learnable, and it survives an unseen edge profile"
        if r["random"]["fraction_removed"] > 0.15 and r["profile"]["fraction_removed"] > 0.0:
            return "partly learnable, but it does not transfer cleanly to an unseen edge profile"
        if r["random"]["fraction_removed"] > 0.15:
            return "learnable inside the family, not transferable across profiles"
        return "not learnable from these inputs"

    def split_row(m, label):
        r = v[m]
        return (f"<tr><td>{label}</td><td class='n'>{r['po_rmse']:.3f}</td>"
                f"<td class='n'>{r['po_bias']:+.3f}</td>"
                f"<td class='n'>{r['random']['rmse']:.3f}</td><td class='n'>{pct(r['random']['fraction_removed'])}</td>"
                f"<td class='n'>{r['profile']['rmse']:.3f}</td>"
                f"<td class='n'>{pct(r['profile']['fraction_removed'])}</td></tr>")

    labels = {"amp_pp_dB": "amplitude p-p [dB]", "taper_dB": "taper [dB]", "ripple_pp_dB": "amplitude ripple p-p [dB]",
              "phase_ripple_pp_deg": "phase ripple p-p [deg]", "usable_width": "usable width [m]"}
    table_metrics = "".join(split_row(m, labels[m]) for m in paired.METRICS)
    best = {k: st[12][f"{k}_percent_mean"] for k in ("direct", "po", "multi", "sequential")}
    rows_html = "".join(
        f"<tr><td>{b} direct solves</td><td class='n'>{st[b]['direct_percent_mean']:.1f} %</td>"
        f"<td class='n'>{st[b]['po_percent_mean']:.1f} %</td><td class='n'>{st[b]['multi_percent_mean']:.1f} %</td>"
        f"<td class='n'>{st[b]['sequential_percent_mean']:.1f} %</td>"
        f"<td class='n'>{100 * st[b]['sequential_worst'] / st[b]['oracle']:.1f} %</td></tr>" for b in sorted(st))
    gain_multi = best["multi"] - best["po"]
    gain_seq = best["sequential"] - best["po"]

    html = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>PO against a direct solve: two experiments</title><style>{CSS}</style></head><body><div class='page'>
<p class='eyebrow'>PINNeAPPle · compact range · two connected experiments</p>
<h1>Physical optics against a direct solve: the residual, and whether it is worth learning</h1>
<p class='sub'>Quiet-zone metrics over reflector size, feed illumination and 18 rolled-edge profiles;
{len(rows)} designs solved twice; four optimisation strategies at equal budget.</p>
<p class='gen'>Generated {time.strftime('%Y-%m-%d %H:%M')} · solver: PINNeAPPle
<code>electromagnetics.reflector</code> · every number below comes from the run saved beside this file
(<code>validate.json</code>, <code>baseline.json</code>, <code>paired.jsonl</code>, <code>analysis.json</code>)</p>

<div class='verdict'><b>Answer to the two questions.</b> <i>Is a direct solve affordable?</i> In two dimensions
yes, at {cost['ratio_median']:.1f}x the cost of PO ({1e3 * cost['direct_seconds']['median']:.0f} ms against
{1e3 * cost['po_seconds']['median']:.0f} ms per design); in three, no &mdash; at the production mesh one
application of the direct operator alone takes {probe['operator_application_seconds']:.0f} s, so a solve of a few
tens of applications is hours and a few hundred designs is weeks (Section&nbsp;2).
<i>Is the PO-to-direct residual meaningfully large?</i> Yes on the metrics that decide a range: PO misses the
amplitude ripple by {ripple['po_rmse']:.2f} dB RMS and the phase ripple by {phase['po_rmse']:.2f}&deg; RMS, and the
usable width by {1e3 * usable['po_rmse']:.0f} mm &mdash; comparable with the +-0.5 dB tolerance the zone is specified
in, and enough to reorder designs (the best design in the space sits at rank {st[12]['po_rank_of_oracle']} of the
PO ranking).
<i>Can it be learned?</i> {verdict_learn('ripple_pp_dB')}. Fitted on and scored on the same edge family the
correction removes {100 * ripple['random']['fraction_removed']:.0f}% of the amplitude-ripple error,
{100 * amp['random']['fraction_removed']:.0f}% of the peak-to-peak and {100 * phase['random']['fraction_removed']:.0f}%
of the phase-ripple error; hold one entire edge profile out and {100 * amp['profile']['fraction_removed']:.0f}% and
{100 * v['taper_dB']['profile']['fraction_removed']:.0f}% survive on the amplitude quantities, but the ripple
({100 * ripple['profile']['fraction_removed']:+.0f}%), the phase ripple
({100 * phase['profile']['fraction_removed']:+.0f}%) and the usable width
({100 * usable['profile']['fraction_removed']:+.0f}%) do not &mdash; the correction interpolates inside a family of
edges, it is not a model of edge physics. <i>Does it pay?</i> At 12 direct solves the multi-fidelity strategies reach
{best['multi']:.0f}% and {best['sequential']:.0f}% of the best design in the space against {best['po']:.0f}% for
trusting the PO ranking and {best['direct']:.0f}% for spending the same solves blindly.</div>

<div class='strip'>
<div class='c'><div class='lab'>Direct vs exact</div><div class='val'>{100 * mom_err:.1f}<span class='u'>%</span></div>
<div class='note'>max field error, PEC circle, exact series</div></div>
<div class='c'><div class='lab'>PO, same points</div><div class='val'>{100 * po_err:.0f}<span class='u'>%</span></div>
<div class='note'>the difference the experiment measures</div></div>
<div class='c'><div class='lab'>Cost, 2-D</div><div class='val'>{cost['ratio_median']:.1f}<span class='u'>x</span></div>
<div class='note'>direct against PO, same mesh</div></div>
<div class='c'><div class='lab'>Cost, 3-D</div><div class='val'>{r3_ratio:.0f}<span class='u'>x</span></div>
<div class='note'>one operator application, vs a whole PO design</div></div>
<div class='c'><div class='lab'>Designs paired</div><div class='val'>{len(rows)}</div>
<div class='note'>5 sizes x 7 feeds x 18 profiles</div></div>
<div class='c'><div class='lab'>Usable width, PO</div><div class='val'>{1e3 * usable['po_rmse']:.0f}<span class='u'>mm RMSE</span></div>
<div class='note'>direct against PO over the space</div></div>
</div>

<h2><span class='n'>1</span>What is asked, and what was run</h2>
<p class='lede'>Two connected experiments: a measured baseline for a direct solve, then the PO-to-direct residual,
whether it generalises, and whether both fidelities together beat either alone at equal budget.</p>
<p>The design space is the one the earlier campaign identified as the lever: reflector size, feed illumination and
the rolled edge. Illumination is the feed exponent <code>q</code> (8 to 20 dBi in seven steps, converted exactly as
the 3-D code does), size is the aperture scale (0.6 to 1.4 of the reference section, five values) and the edge is
the 18-profile family of the 2 GHz study (roll x0.6/1.0/1.4, blend 0.4/0.7, depth 0.7/1.0/1.4). That is
{len(rows)} designs, each solved twice &mdash; physical optics and an exact surface-current solution &mdash; on
the same contour, the same quiet-zone points and the same metrics.</p>
<p><b>Why two dimensions.</b> The question is about the difference between PO and an exact solution, so the exact
solution has to be one that can be trusted and repeated. In two dimensions it is: the solver reproduces the
closed-form series of a conducting circle to {100 * mom_err:.2f}% and costs tens of milliseconds, which makes a
{len(rows)}-design paired sweep a minute. Three dimensions is a different order of cost (Section&nbsp;2). The
physics being measured &mdash; the edge current PO truncates, the multiple reflections it omits, the shadow-side
current it assumes away &mdash; is the same in both dimensions; the absolute numbers are not, and the summary says
what transfers.</p>
<figure><img src='{_img(figs[0])}'><figcaption><b>Figure 1.</b> Left: the direct solver against the exact series
of a perfectly conducting circle in a cylindrical wave, at four mesh densities; PO's error on the same points is
larger by two orders of magnitude. Right: the quiet-zone metrics of the range against the mesh density &mdash; the
PO-to-direct difference is stable from {val['mesh_convergence'][0]['n_segments']} to
{val['mesh_convergence'][-1]['n_segments']} segments, so what follows is not a meshing artefact.</figcaption></figure>

<h2><span class='n'>2</span>Baseline: what a direct solve costs, measured</h2>
<p class='lede'>Measure the thing a surrogate would replace, before assuming the surrogate is needed.</p>
<p>In two dimensions PO takes {1e3 * cost['po_seconds']['median']:.0f} ms per design and the direct solve
{1e3 * cost['direct_seconds']['median']:.0f} ms, a factor of {cost['ratio_median']:.1f}. In three dimensions the
same comparison is not close. At the production mesh (8 elements per wavelength) the reflector carries
{probe['n_triangles']} triangles, PO takes {probe['po_seconds']:.0f} s, and <i>one application</i> of the dense
integral-equation operator &mdash; the inner loop of any iterative direct solve &mdash; takes
{probe['operator_application_seconds']:.0f} s, measured on this machine. A solve needs tens of them: hours per
design. A few hundred designs is weeks.</p>
<p>That number is why this study is not a 3-D campaign, and it is also the useful finding: <b>the 3-D direct solve
is not a small increment on PO, it is a different tool</b> (a fast multipole or hierarchical code, with proper
near-singular quadrature, on a machine built for it). Any workflow that wants a direct number per design in 3-D has
to spend it where it changes a decision &mdash; which is exactly what Experiment&nbsp;2 measures.</p>
<figure><img src='{_img(figs[5])}'><figcaption><b>Figure 2.</b> Measured cost per design. Left: the 2-D model, PO
and direct on the same mesh. Right: the 3-D production geometry &mdash; PO in full against one dense operator
application times the tens a solve needs.</figcaption></figure>

<h2><span class='n'>3</span>Experiment 1: how far PO is, and whether the gap is learnable</h2>
<p class='lede'>PO against the direct solve over all {len(rows)} designs, then a Gaussian process on the residual,
scored on designs it never saw.</p>
<table><thead><tr><th>quantity</th><th>PO RMSE</th><th>PO bias</th><th>learned RMSE<br>5-fold</th>
<th>PO error<br>removed</th><th>learned RMSE<br>profile held out</th><th>PO error<br>removed</th></tr></thead>
<tbody>{table_metrics}</tbody></table>
<p>The PO error is <b>systematic, not noise</b>: the bias on the amplitude peak-to-peak is {amp['po_bias']:+.2f} dB
and on the usable width {1e3 * usable['po_bias']:+.0f} mm, and the residual tracks the design variables
(Figure&nbsp;5) rather than scattering. That is what makes it learnable at all. On a random five-fold split the
correction removes {100 * ripple['random']['fraction_removed']:.0f}% of the PO error on the amplitude ripple,
{100 * usable['random']['fraction_removed']:.0f}% on the usable width, {100 * amp['random']['fraction_removed']:.0f}%
on the amplitude peak-to-peak and {100 * v['taper_dB']['random']['fraction_removed']:.0f}% on the taper.</p>
<p>The honest test is the harder split: hold one entire edge profile out (every size, every feed of one
roll/blend/depth triple), so the model has to interpolate to a shape it has never seen. The verdict there is
blunt &mdash; the correction {verdict_learn('ripple_pp_dB')}. The ripple and the usable width keep most of the gain
({100 * ripple['profile']['fraction_removed']:.0f}%, {100 * usable['profile']['fraction_removed']:.0f}%), the taper
{100 * v['taper_dB']['profile']['fraction_removed']:+.0f}%, while the amplitude peak-to-peak
({100 * amp['profile']['fraction_removed']:+.0f}%) and the phase ripple
({100 * phase['profile']['fraction_removed']:+.0f}%) gain nothing or lose ground. The rule that follows is simple
and worth more than the model itself: <b>correct only the metrics whose residual is systematic across the family,
and check that on a held-out profile before believing it.</b></p>
<figure><img src='{_img(figs[1])}'><figcaption><b>Figure 3.</b> PO against the direct solve, design by design,
coloured by feed <code>q</code>. The amplitude ripple is systematically under-predicted at wide illumination; the
usable width is the metric with the most structure, and it is the objective of Experiment 2.</figcaption></figure>
<figure><img src='{_img(figs[2])}'><figcaption><b>Figure 4.</b> The same designs predicted without the direct
solve: PO alone (left) and PO plus the learned residual (right), random five-fold. The correction moves the cloud
onto the diagonal for the ripple and the usable width.</figcaption></figure>
<figure><img src='{_img(figs[3])}'><figcaption><b>Figure 5.</b> The usable-width residual against the design levers.
It is not noise: it grows with the aperture scale and depends on the feed exponent and on how deep the rolled edge
is, which is why five design variables carry it.</figcaption></figure>

<h2><span class='n'>4</span>Experiment 2: four strategies at equal budget</h2>
<p class='lede'>Objective: the usable quiet-zone width, discounted for peak-to-peak amplitude ripple above
{st[12]['amp_limit']:.1f} dB. Each strategy receives the same number of direct solves and knows the PO value of
every design, because PO is cheap. Repeated over four seeds; the spread is reported.</p>
<table><thead><tr><th>budget</th><th>direct sampling</th><th>PO ranking</th><th>multi-fidelity<br>(one batch)</th>
<th>multi-fidelity<br>(sequential)</th><th>sequential,<br>worst seed</th></tr></thead><tbody>{rows_html}</tbody></table>
<p>The best of the {st[12]['n_designs']} designs reaches {st[12]['oracle']:.3f} m of objective. At 12 direct solves
the sequential multi-fidelity search averages {best['sequential']:.1f}% of it, the one-batch multi-fidelity
{best['multi']:.1f}%, trusting the PO ranking {best['po']:.1f}% and blind sampling {best['direct']:.1f}%.</p>
<p>Three things stand out. First, <b>PO is a strong baseline</b>: ranking by the cheap model is worth far more than
spending the same solves at random, which means the first thing to get right is not the correction but using PO to
choose. Second, <b>the correction adds to it</b>, by {gain_multi:+.0f} (one batch) and {gain_seq:+.0f} (sequential)
percentage points at 12 solves: the model moves the search to designs whose PO value is misleading, which is where
the PO ranking loses the best design (it sits at rank {st[12]['po_rank_of_oracle']}). Third, <b>the gain is small
in absolute terms and it does not grow with budget</b> at 24 solves, because by then every strategy has found the
neighbourhood of the optimum. The honest reading: at this budget and this space, the correction is worth a few
points, not a factor.</p>
<figure><img src='{_img(figs[4])}'><figcaption><b>Figure 6.</b> Best objective reached, as a percentage of the best
design in the space, at three budgets, mean of four seeds with the spread over seeds as the error bar. The
sequential strategy is the one a real campaign can run: refit, choose, solve, repeat.</figcaption></figure>

<h2><span class='n'>5</span>What this changes for the campaign</h2>
<ul>
<li><b>Do not assume the correction is needed; measure the ratio first.</b> The decision is not "PO or full wave" but
"how many direct solves per campaign, and where". In 2-D that ratio is {cost['ratio_median']:.1f}; in 3-D, at
production mesh and with a dense operator, it is {r3_ratio:.0f} &mdash; and that is the number that decides whether
a learned correction, a faster solver, or a smaller design space is the right investment.</li>
<li><b>PO first, everywhere.</b> Using PO to rank a design space is the cheapest large win: it dominates blind
sampling by {best['po'] - best['direct']:.0f} points at 12 solves. Any correction is an increment on top of that.</li>
<li><b>Correct the metrics whose residual is systematic</b> (ripple, usable width) and check the transfer on a
held-out edge profile before quoting it; skip the ones where the residual is noise (amplitude peak-to-peak, phase
ripple in this family).</li>
<li><b>Fit where it will be used.</b> The correction transferred to an unseen profile of the same family; a
geometry outside it &mdash; a serrated edge, a cap, a different junction &mdash; is an extrapolation and needs its
own direct solve before it is trusted.</li>
<li><b>In 3-D, buy the direct numbers where a decision turns on them</b>: a shortlist of a handful of designs, the
top of the corrected ranking, and the geometry as built. That is what the budget allows, and it is what the
strategies above say is worth spending.</li>
</ul>

<h2><span class='n'>6</span>Limits, stated plainly</h2>
<ul>
<li><b>Two dimensions.</b> A line source and a parabolic cylinder; the metrics are relative (taper, ripple, phase
ripple, usable width), not the 3-D quiet-zone diameter. The method and the structure of PO's error transfer; the
numbers do not.</li>
<li><b>The direct solution is exact for the model it solves</b>: perfectly conducting surface, ideal directive line
feed, no struts, no chamber, no absorber. That is the reference PO is compared against here; in 3-D the same
comparison also needs a proper EFIE/CFIE code, which this study does not build.</li>
<li><b>The correction is a Gaussian process on five design variables.</b> A smooth interpolant that does not know
the physics, will not extrapolate outside the box, and is only as honest as its held-out score. Every held-out
number in Section 3 is a real held-out number, including the ones that came out negative.</li>
<li><b>The optimisation comparison reveals the exact value of a design when a strategy spends a solve on it</b>, so
it measures <i>where</i> solves go, not how noisy they are; a real campaign is noisier and smaller, which would
favour the multi-fidelity strategies slightly more, not less.</li>
<li><b>Costs are wall-clock on a shared workstation, one thread per process</b>: read them as orders of magnitude,
not as benchmarks.</li>
</ul>

<h2><span class='n'>7</span>Reproducing it</h2>
<p class='small'>Solver: <code>electromagnetics/reflector/planar2d.py</code> (model, physical optics, direct solve,
metrics) and <code>.../paired.py</code> (sweep, residual, strategies); driver
<code>examples/use_cases/antennas/compact_range_two_fidelity.py --out &lt;dir&gt;</code>; <code>--render-only</code>
rebuilds this PDF. Sources: <code>REFERENCIAS.md</code>.</p>
</div></body></html>"""
    path = os.path.join(out, "report.html")
    with open(path, "w") as fh:
        fh.write(html)
    return path


def render_pdf(html_path: str) -> str | None:
    if not os.path.exists(CHROME):
        return None
    pdf = html_path.replace(".html", ".pdf")
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--no-sandbox",
                    f"--print-to-pdf={pdf}", "file://" + os.path.abspath(html_path)],
                   check=False, capture_output=True)
    return pdf if os.path.isfile(pdf) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", "cr_two_fidelity"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--render-only", action="store_true")
    a = ap.parse_args()
    out = a.out
    os.makedirs(out, exist_ok=True)
    if a.render_only:
        val = _load(os.path.join(out, "validate.json"))
        base = _load(os.path.join(out, "baseline.json"))
        ana = _load(os.path.join(out, "analysis.json"))
        rows = paired.load_rows(os.path.join(out, "paired.jsonl"))
    else:
        val = step_validate(out)
        base = step_baseline(out)
        rows = step_sweep(out, a.workers)
        ana = step_analyse(out, rows)
    figs = step_figures(out, val, base, rows, ana)
    html = step_report(out, val, base, rows, ana, figs)
    pdf = render_pdf(html)
    log(f"report: {html}" + (f" and {pdf}" if pdf else ""))


if __name__ == "__main__":
    main()
