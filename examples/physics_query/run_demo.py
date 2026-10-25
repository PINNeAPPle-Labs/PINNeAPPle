"""Physics query engine, track 2: the measured demonstration of docs/research/physics_condensation.md (section 7).

A 48 x 48 block chip package (3.4 M fine cells). The engine is built once (condensed hierarchies at m and 2 m, and a
calibration solve of the package's own power map), then answers about 5,000 point, region and interface queries:
on the solved map (reads), on a combination of solved maps (superposition), on an unseen map with a rigorous
interval (monotone bound) and with a 0.5 K target (one condensed solve, then reads). Every answer is compared with the
fine finite-volume reference of the same conditions (:func:`pinneapple_condense.query_benchmark.query_demo`, about
10 minutes, most of it the three reference solves).

Run: ``python examples/physics_query/run_demo.py [--force]``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

import pinneapple_lab

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
PATH = os.path.join(RES, "query_demo.json")


def measure(nb: int = 48) -> dict:
    """The demo in a fresh process (its peak memory is that process's)."""
    code = ("import json, sys; from pinneapple_condense.query_benchmark import query_demo; "
            "from pinneapple_condense.benchmark import peak_rss_gb; "
            f"r = query_demo(nb={nb}); r['peak_rss_gb'] = peak_rss_gb(); print('@@' + json.dumps(r, default=str))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    raise RuntimeError(out.stderr[-3000:])


def figure(r: dict, path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"cached": "#2a6fdb", "superposition": "#1f9e77", "bound": "#e08a2a", "condensed": "#7a5bd6"}
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for label, ph in r["phases"].items():
        for meth, s in ph.items():
            if not isinstance(s, dict) or "error_K_or_W" not in s or "max" not in s["error_K_or_W"]:
                continue
            if "points" not in label:
                continue                                  # point temperatures only: one axis in K
            lat = s["latency_s"]["median"] * 1e3
            ax.errorbar(lat, s["error_K_or_W"]["median"],
                        yerr=[[0], [s["error_K_or_W"]["max"] - s["error_K_or_W"]["median"]]],
                        fmt="o", color=colors.get(meth, "#888"), capsize=3, ms=7)
            where = label.split(":")[0] + (" (die interior)" if "interior" in label else "")
            ax.annotate(f"{meth}\n{where}", (lat, s["error_K_or_W"]["median"]), fontsize=7,
                        xytext=(6, -14 if meth == "bound" or "interior" in label else 4), textcoords="offset points")
    ref = np.median(r["reference_solve_s"]) * 1e3
    ax.axvline(ref, color="#555", ls="--", lw=1)
    ax.text(ref, ax.get_ylim()[1] if ax.get_yscale() == "linear" else 1, " fine FV\n reference", fontsize=7,
            va="top", color="#555")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("latency per query, median [ms]")
    ax.set_ylabel("|error| vs fine FV [K] (median, bar to max)")
    ax.set_title(f"{r['blocks']:,} blocks, {r['cells'] / 1e6:.1f} M cells: point temperature queries", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    with pinneapple_lab.experiment(
            "examples/physics_query", title="Physics query engine on a chip package",
            description="query(quantity, where, conditions, accuracy) answered by the cheapest valid method -- read "
                        "a solved state, superpose solved states, a rigorous monotone bound, a new condensed solve -- "
                        "with uncertainty, validity and provenance, against the fine finite-volume reference.",
            domain="heat transfer", methods=["static condensation", "superposition", "M-matrix bounds"],
            modules=["pinneapple_query", "pinneapple_condense.query"], out_dir=HERE, register=False,
            links={"report": "docs/research/physics_condensation.md", "script": "examples/physics_query/run_demo.py"}) \
            as ex:
        if a.force or not os.path.exists(PATH):
            json.dump(measure(), open(PATH, "w"), indent=1)
        r = json.load(open(PATH))
        ex.add_file(figure(r, os.path.join(FIG, "latency_vs_error.png")))
        ex.log_metric("queries", r["queries_total"], "", "queries answered in the demo")
        ex.log_metric("build_s", r["timings"]["calibrate"], "s", "hierarchies at m and 2m plus the calibration solve")
        ex.log_metric("reference_solve_s", float(np.median(r["reference_solve_s"])), "s", "fine FV, AMG-CG, median")
        within, total, inside, n_iv = 0, 0, 0, 0
        for ph in r["phases"].values():
            for s in ph.values():
                if isinstance(s, dict) and "error_within_uncertainty" in s:
                    w, t = map(int, s["error_within_uncertainty"].split("/"))
                    within, total = within + w, total + t
                if isinstance(s, dict) and "interval_contains_truth" in s:
                    w, t = map(int, s["interval_contains_truth"].split("/"))
                    inside, n_iv = inside + w, n_iv + t
        ex.log_metric("error_within_uncertainty", within / total, "fraction", f"{within}/{total} answers")
        ex.check("rigorous intervals contain the fine reference", inside == n_iv, kind="verification",
                 detail=f"{inside}/{n_iv}")
        ex.check("at least 99 % of answers within their stated uncertainty", within >= 0.99 * total,
                 kind="verification", detail=f"{within}/{total}")


if __name__ == "__main__":
    main()
