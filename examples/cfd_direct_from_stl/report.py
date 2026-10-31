"""Verdicts, tables and figures of the campaign from results/*.json (the pre-registered rule:
pinneapple_simulation.cfd_direct.verdicts)."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from pinneapple_simulation.cfd_direct.verdicts import QUANTITIES, operating_points, profile_errors, verdict

LABEL = {"reference": "R: body-fitted mesh + Taylor-Hood", "lbm": "A: LBM, voxelised", "immersed": "B: immersed cut cell"}
COLOR = {"reference": "#1f4e79", "lbm": "#c55a11", "immersed": "#2e7d32"}


def load(results: Path) -> dict:
    groups = defaultdict(list)
    for p in sorted(results.glob("*.json")):
        if p.name in ("verdicts.json",):
            continue
        r = json.loads(p.read_text())
        if "method" not in r:
            continue
        r.setdefault("ok", True)
        groups[(r["case"], r.get("Re"))].append(r)
    for runs in groups.values():
        profile_errors(runs)
    return groups


def _fmt(x, pct=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{100 * x:.2f} %" if pct else f"{x:.4g}"


def report(results: Path, figures: Path) -> dict:
    figures.mkdir(exist_ok=True)
    groups = load(results)
    out, lines = {}, []
    for (case, re), runs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        op = operating_points(runs, case, re)
        tag = case if re is None else f"{case} Re {re:g}"
        v = {m: verdict(op, m) for m in ("lbm", "immersed")}
        rstar = op.get("reference")
        out[tag] = {"gate_passed": op["gate_passed"], "limits": op.get("limits"),
                    "reference_n": rstar["n"] if rstar else None,
                    "reference_total_s": rstar["times"]["total_s"] if rstar else None, **v}
        lines += [f"### {tag}", "",
                  f"Gate R {'passed' if op['gate_passed'] else '**not passed** (R* = finest reference run)'}; "
                  f"R* = n {rstar['n'] if rstar else '-'}"
                  f" ({rstar['times']['total_s']:.1f} s)" if rstar else "", "",
                  "| method | n | " + " | ".join(f"err {q}" for q in QUANTITIES[case]) +
                  " | err profile | total s | solve s | peak MB | note |",
                  "|---|---|" + "---|" * (len(QUANTITIES[case]) + 5)]
        for r in sorted(runs, key=lambda r: (r["method"], r["n"])):
            if not r["ok"]:
                lines.append(f"| {r['method']} | {r['n']} | " + " | ".join("-" for _ in QUANTITIES[case]) +
                             f" | - | {r.get('wall_s', 0):.0f} | - | {r.get('watchdog_peak_mb', 0):.0f} | "
                             f"{r['why'][:120].replace('|', '/')} |")
                continue
            note = []
            if "converged" in r:
                note.append(f"{r['flow_through']:.1f} FT{'' if r['converged'] else ', not converged'}")
            lines.append(f"| {r['method']} | {r['n']} | " + " | ".join(_fmt(r.get(f"err_{q}")) for q in QUANTITIES[case])
                         + f" | {_fmt(r.get('err_profile'))} | {r['times']['total_s']:.1f} | "
                         f"{r['times']['solve_s']:.1f} | {r.get('peak_rss_mb', 0):.0f} | {'; '.join(note)} |")
        lines += ["", "| candidate | verdict | speed-up (total) | speed-up (solve) | X* n |", "|---|---|---|---|---|"]
        for m, vv in v.items():
            lines.append(f"| {LABEL[m]} | **{vv['verdict']}** | {_fmt(vv.get('speedup'), False)} | "
                         f"{_fmt(vv.get('speedup_solve'), False)} | {vv.get('n', '-')} |"
                         + (f" {vv['why']}" if "why" in vv else ""))
        lines.append("")
    (results / "verdicts.json").write_text(json.dumps(out, indent=1, default=str))
    (results / "summary.md").write_text("\n".join(lines))
    _figures(groups, figures)
    return out


def _figures(groups: dict, figures: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = sorted(groups, key=lambda k: (k[0], k[1] or 0))
    # main quantity against resolution, with the published value
    main = {"C1": ("Cd", 5.57953523384), "C2": ("Cd", None), "C3": ("x1", None)}
    fig, axs = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.2), squeeze=False)
    for ax, k in zip(axs[0], keys, strict=True):
        q, ref = main[k[0]]
        for m in ("reference", "lbm", "immersed"):
            rs = sorted([r for r in groups[k] if r["method"] == m and r["ok"]], key=lambda r: r["n"])
            if rs:
                ax.plot([r["n"] for r in rs], [r[q] for r in rs], "o-", color=COLOR[m], label=LABEL[m], ms=4)
        rv = ref if ref is not None else next((r.get(f"{q}_ref") for r in groups[k] if r.get(f"{q}_ref")), None)
        if rv:
            ax.axhline(rv, color="k", lw=0.8, ls="--")
            ax.axhspan(rv * 0.98, rv * 1.02, color="0.9", zorder=0)
        ax.set_xscale("log", base=2)
        ax.set_title(f"{k[0]}" + ("" if k[1] is None else f" Re {k[1]:g}"), fontsize=9)
        ax.set_xlabel("cells per D (S)", fontsize=8)
        ax.set_ylabel(q, fontsize=8)
        ax.tick_params(labelsize=7)
    axs[0][0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(figures / "quantity_vs_resolution.png", dpi=150)
    plt.close(fig)
    # total time against error
    fig, axs = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.2), squeeze=False)
    for ax, k in zip(axs[0], keys, strict=True):
        q = main[k[0]][0]
        for m in ("reference", "lbm", "immersed"):
            rs = [r for r in groups[k] if r["method"] == m and r["ok"] and np.isfinite(r.get(f"err_{q}", np.nan))]
            if rs:
                rs = sorted(rs, key=lambda r: r["n"])
                ax.loglog([r["times"]["total_s"] for r in rs], [max(r[f"err_{q}"], 1e-5) for r in rs], "o-",
                          color=COLOR[m], label=LABEL[m], ms=4)
        ax.set_title(f"{k[0]}" + ("" if k[1] is None else f" Re {k[1]:g}"), fontsize=9)
        ax.set_xlabel("total time [s]", fontsize=8)
        ax.set_ylabel(f"relative error of {q}", fontsize=8)
        ax.tick_params(labelsize=7)
    axs[0][0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(figures / "time_vs_error.png", dpi=150)
    plt.close(fig)
    # peak memory
    fig, axs = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.0), squeeze=False)
    for ax, k in zip(axs[0], keys, strict=True):
        for m in ("reference", "lbm", "immersed"):
            rs = sorted([r for r in groups[k] if r["method"] == m], key=lambda r: r["n"])
            if rs:
                ax.semilogy([r["n"] for r in rs], [r.get("peak_rss_mb") or r.get("watchdog_peak_mb") or np.nan
                                                   for r in rs], "o-", color=COLOR[m], label=LABEL[m], ms=4)
        ax.set_xscale("log", base=2)
        ax.set_title(f"{k[0]}" + ("" if k[1] is None else f" Re {k[1]:g}"), fontsize=9)
        ax.set_xlabel("cells per D (S)", fontsize=8)
        ax.set_ylabel("peak memory [MB]", fontsize=8)
        ax.tick_params(labelsize=7)
    axs[0][0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(figures / "memory.png", dpi=150)
    plt.close(fig)
