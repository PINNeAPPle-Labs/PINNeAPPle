"""Adaptive Physics Condensation, track 1: the measured benchmark of docs/research/physics_condensation.md.

Synthetic multilayer chip package (substrate, C4, interposer + TSVs, microbumps, die, TIM, lid), repeated 40 um unit
cells, hotspot power maps, contact resistances. Every study is a function of :mod:`pinneapple_condense.benchmark`,
run in a fresh process (its peak memory is that process's). Results go to ``results/*.json`` (kept: rerun one with
``--force <study>``), figures to ``figures/``.

Run: ``python examples/physics_condensation/run_benchmark.py [--quick] [--force all|<study>]``. The full run takes
about an hour on 4 cores and needs ~12 GB of memory for the largest package (512 x 512 blocks).
"""
from __future__ import annotations

import argparse
import json
import os

import pinneapple_lab
from pinneapple_condense import benchmark as B
from pinneapple_condense import figures as F

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")

FULL = {
    "mesh": ("mesh_convergence", {}),
    "accuracy": ("accuracy_vs_modes", {}),
    "accuracy_cc": ("accuracy_vs_modes", {"modes": "cc", "ms": [2, 8, 24]}),
    "accuracy_32": ("accuracy_vs_modes", {"shape": [32, 32], "ms": [6, 12, 24]}),
    "correction": ("correction_study", {}),
    "refinement": ("refinement_study", {}),
    "keff": ("homogenization_study", {}),
    "transient": ("transient_study", {}),
    "configs": ("configurations", {"nb": 64}),
    "learned_16": ("learned_study", {"nb": 16}),
    "learned_32": ("learned_study", {"nb": 32}),
    "learned_32_small_train": ("learned_study", {"nb": 32, "n_train": 6}),
    "learned_64": ("learned_study", {"nb": 64}),
    "root_ranks": ("root_compressibility", {"nb": 64}),
}
SCALING = {"reference": [4, 8, 16, 32, 48], "flat": [4, 8, 16, 32, 64, 128],
           "hierarchy": [4, 8, 16, 32, 64, 128, 256], "hierarchy_m6": [64, 128, 256, 512]}
QUICK = {"accuracy": ("accuracy_vs_modes", {"shape": [6, 6], "ms": [2, 8]}),
         "keff": ("homogenization_study", {"shape": [6, 6]})}


def run(name, func, kw, force):
    path = os.path.join(RES, f"{name}.json")
    if os.path.exists(path) and not force:
        return json.load(open(path))
    print(f"-- {name}: {func}({kw})", flush=True)
    try:
        r = B.isolated(func, **kw)
    except Exception as e:                     # recorded, not hidden: the report lists what did not fit
        r = {"failed": str(e)[-600:]}
    json.dump(r, open(path, "w"), indent=1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", default="")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    force = set(a.force.split(","))
    studies = QUICK if a.quick else FULL
    with pinneapple_lab.experiment(
            "examples/physics_condensation", title="Adaptive Physics Condensation: chip-package thermal",
            description="Hierarchical static condensation of repeated blocks (microbumps, TSVs) with reduced "
                        "interfaces, error-driven refinement and learned block operators, against a fine "
                        "finite-volume reference and Keff homogenisation.",
            domain="heat transfer", methods=["finite volumes", "static condensation", "nested dissection",
                                             "Craig-Bampton", "POD-GPR"],
            modules=["pinneapple_condense"], out_dir=HERE, register=False,
            links={"report": "docs/research/physics_condensation.md",
                   "script": "examples/physics_condensation/run_benchmark.py"}) as ex:
        R = {k: run(k, f, kw, "all" in force or k in force) for k, (f, kw) in studies.items()}
        if "accuracy" in R:
            acc = R["accuracy"]
            ex.add_file(F.accuracy(acc, os.path.join(FIG, "accuracy_vs_modes.png")))
            r8 = [r for r in acc["rows"] if r["m"] == 8][0]
            ex.log_metric("err_Tmax_m8", r8["err_Tmax"], "K", f"|Tmax condensed m=8 - fine FV|, {acc['shape']} blocks")
            ex.check("energy balance with reduced interfaces", max(abs(r["energy_balance"]) for r in acc["rows"])
                     < 1e-8, kind="physics")
            ex.check("estimator effectivity in [0.5, 2] for m >= 4",
                     all(0.5 <= r["effectivity"] <= 2 for r in acc["rows"] if r["m"] >= 4), kind="verification")
        if "keff" in R and "maps" in R["keff"]:
            ex.add_file(F.keff_maps(R["keff"]["maps"], os.path.join(FIG, "keff_vs_fine.png")))
        if "refinement" in R:
            ex.add_file(F.refinement(R["refinement"], os.path.join(FIG, "refinement.png")))
        if "transient" in R:
            ex.add_file(F.transient(R["transient"], os.path.join(FIG, "transient.png")))
            best = R["transient"]["rows"][-1]
            ex.log_metric("transient_err_Tmax", best["err_Tmax_max"], "K",
                          f"max over time, CB m={best['m']} k={best['k']} vs fine implicit Euler, same dt")
        if not a.quick:
            pts = {}
            for meth, sizes in SCALING.items():
                for nb in sizes:
                    kw = {"nb": nb, "reference": meth == "reference", "flat": meth == "flat",
                          "hier": meth.startswith("hierarchy")}
                    if meth == "hierarchy_m6":
                        kw["m"] = 6
                    r = run(f"scaling_{meth}_{nb}", "scaling_point", kw, "all" in force or "scaling" in force)
                    p = pts.setdefault(nb, {"blocks": nb * nb})
                    if "failed" in r:
                        p[meth] = {"failed": r["failed"]}
                        continue
                    p["cells"] = r["cells"]
                    key = "hierarchy" if meth.startswith("hierarchy") else meth
                    p[meth] = dict(r[key], peak_rss_gb=r["peak_rss_gb"])
            pts = [pts[k] for k in sorted(pts)]
            json.dump(pts, open(os.path.join(RES, "scaling.json"), "w"), indent=1)
            ex.add_file(F.scaling(pts, os.path.join(FIG, "scaling.png")))
            big = [p for p in pts if "prepare_s" in p.get("hierarchy", {})][-1]
            ex.log_metric("largest_blocks", big["blocks"], "blocks", "largest package solved by the hierarchy, m=12")
            ex.log_metric("largest_build_s", big["hierarchy"]["prepare_s"], "s", "hierarchy build, fresh process")
            ex.log_metric("largest_solve_s", big["hierarchy"]["solve_s"], "s", "one power map, after the build")
            ex.log_metric("largest_peak_rss", big["hierarchy"]["peak_rss_gb"], "GB", "peak RSS of the process")
            for p in pts:
                if "Tmax" in p.get("reference", {}) and "Tmax" in p.get("hierarchy", {}):
                    ex.check(f"hierarchy m=12 Tmax within 0.5 K of the fine reference ({p['blocks']} blocks)",
                             abs(p["hierarchy"]["Tmax"] - p["reference"]["Tmax"]) < 0.5, kind="verification",
                             detail=f"{p['hierarchy']['Tmax']:.3f} vs {p['reference']['Tmax']:.3f} K")
            for k in ("learned_32", "learned_64"):
                if k in R and "adaptive_err" in R[k]:
                    ex.check(f"{k}: learned + verified Tmax within 0.05 K of all-exact condensation",
                             R[k]["adaptive_err"]["err_Tmax"] < 0.05, kind="verification")
                    ex.log_metric(f"{k}_speedup", R[k]["exact_s"] / R[k]["adaptive"]["timings"]["total"], "x",
                                  "all-exact per-block condensation time / learned+verified time (training excluded)")


if __name__ == "__main__":
    main()
