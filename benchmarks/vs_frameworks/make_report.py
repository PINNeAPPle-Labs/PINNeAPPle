"""Tables and figures of the cross-framework benchmark, generated from ``results/*.json``.

    python -m benchmarks.vs_frameworks.make_report      # writes results/TABLES.md and figures/*.png

``REPORT.md`` (the narrative) quotes these tables; every number in it comes from here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pinneapple_tools.benchmark_suite.crossframework.evaluate import paired_ratio, seed_stats

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGS = HERE / "figures"
#: the first three slots of the validated categorical palette (all-pairs safe), fixed per framework family
COLORS = {"pinneapple": "#2a78d6", "pinneapple_nostop": "#2a78d6", "deepxde": "#eb6834", "neuraloperator": "#eb6834",
          "physicsnemo_sym": "#1baf7a", "physicsnemo": "#1baf7a"}
LABEL = {"pinneapple": "PINNeAPPle", "pinneapple_nostop": "PINNeAPPle, early stopping off (patience = iterations)", "deepxde": "DeepXDE", "physicsnemo_sym": "PhysicsNeMo-Sym",
         "physicsnemo": "PhysicsNeMo", "neuraloperator": "neuraloperator"}
ORDER = ["pinneapple", "pinneapple_nostop", "deepxde", "physicsnemo_sym", "physicsnemo", "neuraloperator"]
EXACT_FD_FLOOR = {}


def load(track):
    p = RESULTS / f"{track}.json"
    return json.loads(p.read_text()) if p.exists() else []


def fmt(v, digits=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if isinstance(v, (int, np.integer)):
        return str(v)
    a = abs(v)
    if a != 0 and (a < 1e-2 or a >= 1e4):
        return f"{v:.{digits - 1}e}"
    return f"{v:.{digits}g}"


def cell(values):
    s = seed_stats(values)
    if s["n"] == s["n_failed"]:
        return f"failed ({s['n_failed']}/{s['n']})"
    out = f"{fmt(s['median'])} [{fmt(s['min'])}, {fmt(s['max'])}]"
    if s["n_failed"]:
        out += f" ({s['n_failed']} failed)"
    return out


def metric(r, key):
    if r.get("status") != "ok":
        return float("nan")
    if key in (r.get("metrics") or {}):
        return float(r["metrics"][key])
    v = r.get(key)
    return float(v) if v is not None else float("nan")


def groups(rows):
    g = {}
    for r in rows:
        g.setdefault((r.get("problem"), r.get("framework")), []).append(r)
    return g


def _fws(rows):
    present = {r.get("framework") for r in rows}
    return [f for f in ORDER if f in present]


def table_pinn(track, rows):
    g = groups(rows)
    probs = [p for p in ("burgers", "poisson", "helmholtz", "allen_cahn") if any(k[0] == p for k in g)]
    lines = [f"### Track {track}", "",
             "median [min, max] over seeds; n = seeds per cell (failures counted, not dropped)", "",
             "| problem | framework | n | rel. L2 | max abs err | FD residual RMS | iterations | CPU s | wall s "
             "| ms / iteration (CPU) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for p in probs:
        for fw in _fws(rows):
            rs = sorted(g.get((p, fw), []), key=lambda r: r.get("seed", 0))
            if not rs:
                continue
            ms = [metric(r, "cpu_s") / max(1, r.get("iterations") or 0) * 1e3 if r.get("status") == "ok"
                  else float("nan") for r in rs]
            lines.append(f"| {p} | {LABEL[fw]} | {len(rs)} | {cell([metric(r, 'rel_l2') for r in rs])} | "
                         f"{cell([metric(r, 'max_abs') for r in rs])} | "
                         f"{cell([metric(r, 'residual_rms') for r in rs])} | "
                         f"{cell([float(r.get('iterations') or 'nan') for r in rs])} | "
                         f"{cell([metric(r, 'cpu_s') for r in rs])} | {cell([metric(r, 'wall_s') for r in rs])} | "
                         f"{cell(ms)} |")
    fails = [r for r in rows if r.get("status") != "ok"]
    if fails:
        lines += ["", "Failed runs:", ""] + [f"* `{r['run_id']}`: {r.get('error', '')[:300]}" for r in fails]
    return lines


def table_paired(track, rows, key="rel_l2"):
    """Per-seed ratio framework / PINNeAPPle (same seed = same initial weights in A1)."""
    g = groups(rows)
    lines = ["", f"Paired per-seed ratio of {key} to PINNeAPPle (median [min, max]; 'lower' = seeds where the "
             "framework's error is lower than PINNeAPPle's)", "",
             "| problem | framework | n | ratio | lower |", "|---|---|---|---|---|"]
    for p in sorted({k[0] for k in g}):
        base = {r["seed"]: metric(r, key) for r in g.get((p, "pinneapple"), [])}
        for fw in _fws(rows):
            if fw == "pinneapple":
                continue
            other = {r["seed"]: metric(r, key) for r in g.get((p, fw), [])}
            pr = paired_ratio(other, base)
            if pr["n"]:
                lines.append(f"| {p} | {LABEL[fw]} | {pr['n']} | {fmt(pr['median'])} [{fmt(pr['min'])}, "
                             f"{fmt(pr['max'])}] | {pr['a_better']}/{pr['n']} |")
    return lines


def table_b(rows):
    g = groups(rows)
    lines = ["### Track B (FNO, Darcy 64 x 64)", "", "median [min, max] over seeds", "",
             "| framework | n | test rel. L2 (mean) | test rel. L2 (max) | val rel. L2 | parameters (real) | "
             "s / epoch (CPU) | latency batch 1 [ms] | latency batch 20 [ms] |", "|---|---|---|---|---|---|---|---|---|"]
    for fw in _fws(rows):
        rs = g.get(("darcy", fw), [])
        ok = [r for r in rs if r.get("status") == "ok"]
        lines.append(
            f"| {LABEL[fw]} | {len(rs)} | {cell([metric(r, 'rel_l2_mean') for r in rs])} | "
            f"{cell([metric(r, 'rel_l2_max') for r in rs])} | "
            f"{cell([float(r['val_metrics']['rel_l2_mean']) for r in ok])} | "
            f"{ok[0].get('n_parameters_real') if ok else 'n/a'} | "
            f"{cell([float(np.median(r['epoch_cpu_s'])) for r in ok])} | "
            f"{cell([1e3 * r['latency_batch1_s'] for r in ok])} | {cell([1e3 * r['latency_batch20_s'] for r in ok])} |")
    return lines


def table_c(rows):
    g = groups(rows)
    lines = ["### Track C (inverse Burgers, nu from 2000 noisy samples)", "", "median [min, max] over seeds", "",
             "| framework | n | nu estimate | nu relative error | field rel. L2 | iterations | CPU s |",
             "|---|---|---|---|---|---|---|"]
    for fw in _fws(rows):
        rs = sorted(g.get(("burgers_inverse", fw), []), key=lambda r: r.get("seed", 0))
        if not rs:
            continue
        lines.append(f"| {LABEL[fw]} | {len(rs)} | {cell([metric(r, 'nu_hat') for r in rs])} | "
                     f"{cell([metric(r, 'nu_rel_err') for r in rs])} | {cell([metric(r, 'rel_l2') for r in rs])} | "
                     f"{cell([float(r.get('iterations') or 'nan') for r in rs])} | "
                     f"{cell([metric(r, 'cpu_s') for r in rs])} |")
    return lines + table_paired("C", rows, "nu_rel_err")


def table_timing():
    p = RESULTS / "timing.json"
    if not p.exists():
        return []
    rows = json.loads(p.read_text())
    lines = ["### Cost per iteration on a quiet machine (A1 configuration, one run at a time, one thread)", "",
             "| problem | framework | iterations | ms / iteration (CPU) | ms / iteration (wall) | load before |",
             "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["problem"], ORDER.index(r["framework"]))):
        lines.append(f"| {r['problem']} | {LABEL[r['framework']]} | {r['iterations']} | "
                     f"{fmt(1e3 * r['cpu_s'] / r['iterations'])} | {fmt(1e3 * r['wall_s'] / r['iterations'])} | "
                     f"{fmt(r['loadavg_before'][0])} |")
    return lines


def table_d():
    p = RESULTS / "D.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    lines = ["### Track D (measured ergonomics)", "", "| framework | runner lines (A / B / C) | install size [MB] | "
             "import time [s] |", "|---|---|---|---|"]
    for fw, v in d.items():
        loc = " / ".join(str(v["loc"].get(t, "-")) for t in ("A", "B", "C"))
        lines.append(f"| {LABEL.get(fw, fw)} | {loc} | {fmt(v['install_mb'])} | {fmt(v['import_s'])} |")
    return lines


def figure_pinn(track, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = groups(rows)
    probs = [p for p in ("burgers", "poisson", "helmholtz", "allen_cahn") if any(k[0] == p for k in g)]
    if not probs:
        return None
    fws = [f for f in _fws(rows)]
    fig, axes = plt.subplots(1, len(probs), figsize=(3.2 * len(probs), 3.2), squeeze=False)
    for ax, p in zip(axes[0], probs, strict=True):
        for i, fw in enumerate(fws):
            vals = [metric(r, "rel_l2") for r in g.get((p, fw), [])]
            vals = [v for v in vals if np.isfinite(v)]
            ax.scatter([i] * len(vals), vals, s=36, color=COLORS[fw], edgecolor="white", linewidth=1.5, zorder=3)
            if vals:
                ax.hlines(np.median(vals), i - 0.25, i + 0.25, color="#0b0b0b", linewidth=2, zorder=4)
        allv = [metric(r, "rel_l2") for fw in fws for r in g.get((p, fw), [])]
        allv = [v for v in allv if np.isfinite(v) and v > 0]
        if allv and max(allv) / min(allv) > 5:
            ax.set_yscale("log")
        ax.set_xticks(range(len(fws)), [LABEL[f].replace("PhysicsNeMo-Sym", "PNeMo-Sym") for f in fws],
                      fontsize=8)
        ax.set_title(p, fontsize=10)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0][0].set_ylabel("relative L2 error (test grid)")
    fig.suptitle(f"Track {track}: one dot per seed, bar = median", fontsize=10)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    out = FIGS / f"{track}_rel_l2.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    lines = ["# Cross-framework benchmark: generated tables", "",
             "Generated by `python -m benchmarks.vs_frameworks.make_report` from `results/*.json`. Do not edit.", ""]
    env = RESULTS / "environment.json"
    if env.exists():
        e = json.loads(env.read_text())
        lines += ["## Environment", "", f"* CPU: {e['cpu']} ({e['n_cpus']} logical CPUs), threads per run: "
                  f"{e['threads']}", f"* Python {e['python']}, torch {e['torch']} (CUDA available: "
                  f"{e['cuda_available']}), numpy {e['numpy']}", f"* commit {e.get('git_commit', '?')}",
                  "* versions: " + ", ".join(f"{k} {v}" for k, v in e["versions"].items() if v), ""]
    for track in ("A1", "A2"):
        rows = load(track)
        if rows:
            lines += table_pinn(track, rows)
            if track == "A1":
                lines += table_paired(track, rows)
            f = figure_pinn(track, rows)
            if f:
                lines += ["", f"![{track}](../figures/{f.name})", ""]
    lines += [""] + table_timing() + [""]
    rows = load("B")
    if rows:
        lines += table_b(rows) + [""]
    rows = load("C")
    if rows:
        lines += table_c(rows) + [""]
    lines += table_d()
    (RESULTS / "TABLES.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
