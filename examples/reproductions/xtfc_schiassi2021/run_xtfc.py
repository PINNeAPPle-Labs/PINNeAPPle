"""Reproduce the X-TFC paper (Schiassi et al., arXiv:2005.10632v1) with the protocol frozen in TARGETS.md.

    python examples/reproductions/xtfc_schiassi2021/run_xtfc.py                   # full protocol
    python examples/reproductions/xtfc_schiassi2021/run_xtfc.py --smoke           # 2 seeds per case

Writes results.json, RESULTS.md and figures next to this file. Every number is measured by this script; the
paper's numbers are the quoted targets of TARGETS.md.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import pinneapple_lab  # noqa: E402
from pinneapple_analysis.reproduction import evaluate, seed_stats, status  # noqa: E402
from pinneapple_physics.paper_problems_xtfc import XTFC_CASES  # noqa: E402
from pinneapple_simulation.numerical_solvers.xtfc_general import grid, solve_xtfc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_SEED = 20201            # test-set seed, independent of the ELM seeds (TARGETS.md, ambiguity 2)

# Targets quoted in TARGETS.md: (case, field) -> (paper test max, best competitor or None, where)
TARGETS = {
    ("ode1", "y"): (6.66134e-16, 1e-3, "T1, Fig. 3 p. 11; LeNN O(1e-3) average"),
    ("ode2", "y1"): (5.5434e-10, 1e-4, "T2, Fig. 5 p. 14; ANN [24] O(1e-4)"),
    ("ode2", "y2"): (1.58122e-9, 1e-5, "T2, Fig. 6 p. 14; ANN [24] O(1e-5)"),
    ("pde1", "f"): (5.054e-13, 5e-7, "T3, p. 17 / Table 4; Ref. [24] 5e-7"),
    ("pde2", "f"): (7.581e-12, 6e-6, "T4, p. 18 / Table 5; Ref. [24] 6e-6"),
    ("pde3", "f"): (8.977e-11, 1.5e-5, "T5, p. 20 / Table 6; Ref. [24] 1.5e-5"),
    ("pde4", "f"): (5.785e-7, None, "T6, p. 21"),
    ("pde5", "f"): (4.025e-4, None, "T7, p. 23"),
    ("pde6", "z"): (1.657e-8, None, "T8, p. 24"),
    ("pde7", "f"): (2.978e-5, None, "T9, p. 25"),
}
PAPER_MEAN = {("pde1", "f"): 7.640e-14, ("pde2", "f"): 1.322e-12, ("pde3", "f"): 1.068e-11,
              ("pde4", "f"): 6.928e-8, ("pde5", "f"): 2.090e-5, ("pde6", "z"): 1.018e-9, ("pde7", "f"): 8.082e-7}


def test_set(case):
    p = case.problem
    lo, hi = np.array(p.bounds).T
    pts = lo + (hi - lo) * np.random.default_rng(TEST_SEED).random((10_000 if p.ndim > 1 else 1_000, p.ndim))
    return pts


def run_one(case, seed, *, kind="uniform", n=None, neurons=None, map_to_unit=False):
    p = case.problem
    Xtr = grid(p.bounds, n or case.n_train, kind)
    if p.linear:
        max_iter, tol = 1, 0.0
    else:
        max_iter = case.paper_iterations + 5
        tol = case.paper_tol if case.paper_tol is not None else 1e-14
    r = solve_xtfc(p, Xtr, n_neurons=neurons or case.n_neurons, activation=case.activation,
                   weight_range=case.weight_range, seed=seed, max_iter=max_iter, tol=tol, map_to_unit=map_to_unit)
    Xte = test_set(case)
    out = {"seed": seed, "iterations": r.iterations, "converged": r.converged, "lstsq_s": r.lstsq_seconds,
           "total_s": r.total_seconds, "constraint_err": r.constraint_error(), "fields": {}}
    res = r.residual(Xte)
    out["test_residual_max"] = float(max(np.max(np.abs(x)) for x in res)) if all(np.all(np.isfinite(x)) for x in
                                                                                  res) else float("inf")
    for f in p.fields:
        m_test = evaluate(r.predict(Xte, f), case.exact_np(Xte, f), ("max_abs", "mean_abs", "rel_l2"))
        m_train = evaluate(r.predict(Xtr, f), case.exact_np(Xtr, f), ("max_abs", "mean_abs"))
        rec = {"test_max": m_test["max_abs"], "test_mean": m_test["mean_abs"], "test_rel_l2": m_test["rel_l2"],
               "train_max": m_train["max_abs"], "train_mean": m_train["mean_abs"]}
        if case.test_points is not None:
            tp = case.test_points[f]
            rec["paper_points_max"] = evaluate(r.predict(tp, f), case.exact_np(tp, f), ("max_abs",))["max_abs"]
        out["fields"][f] = rec
    return out


def summarise(runs, name, f, key):
    return seed_stats([r["fields"][f][key] for r in runs]).as_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cases", nargs="*", default=sorted(XTFC_CASES))
    a = ap.parse_args()
    n_ode, n_pde = (2, 2) if a.smoke else (1000, 10)
    results = {"protocol": "TARGETS.md", "machine": platform.platform(), "python": platform.python_version(),
               "smoke": a.smoke, "variants": {}}
    variants = [("primary", {})]
    variants_for = {"pde": [("cgl (declared secondary)", {"kind": "cgl"})],
                    "ode1": [("n = L = 50 (declared secondary)", {"n": 50, "neurons": 50})],
                    "ode2": [("inputs mapped to [-1, 1] (post-hoc, not used for status)", {"map_to_unit": True})]}
    with pinneapple_lab.experiment(
            "examples/reproductions/xtfc_schiassi2021", title="X-TFC reproduction (Schiassi et al.)",
            description="The ODE and PDE problems of arXiv:2005.10632v1 solved by X-TFC at the paper's settings, "
                        "many seeds, scored by the neutral evaluator against the frozen targets of TARGETS.md.",
            domain="physics-informed ML", methods=["X-TFC"],
            modules=["pinneapple_simulation.numerical_solvers.xtfc_general", "pinneapple_physics.paper_problems_xtfc",
                     "pinneapple_analysis.reproduction"], out_dir=HERE, register=False,
            links={"targets": "examples/reproductions/xtfc_schiassi2021/TARGETS.md"}) as ex:
        for name in a.cases:
            case = XTFC_CASES[name]
            seeds = range(n_ode if name.startswith("ode") else n_pde)
            extra = variants_for.get(name, variants_for["pde"] if name.startswith("pde") else [])
            for vname, kw in variants + extra:
                t0 = time.perf_counter()
                runs = [run_one(case, s, **kw) for s in seeds]
                wall = time.perf_counter() - t0
                summary = {"wall_s": wall, "n_seeds": len(runs),
                           "iterations": seed_stats([r["iterations"] for r in runs]).as_dict(),
                           "converged": sum(r["converged"] for r in runs),
                           "lstsq_s_median": float(np.median([r["lstsq_s"] for r in runs])),
                           "total_s_median": float(np.median([r["total_s"] for r in runs])),
                           "constraint_err_max": float(max(r["constraint_err"] for r in runs)),
                           "test_residual": seed_stats([r["test_residual_max"] for r in runs]).as_dict(),
                           "fields": {}}
                for f in case.problem.fields:
                    key = "paper_points_max" if case.test_points is not None else "test_max"
                    st = {k: summarise(runs, name, f, k) for k in
                          ("test_max", "test_mean", "test_rel_l2", "train_max", "train_mean")}
                    if case.test_points is not None:
                        st["paper_points_max"] = summarise(runs, name, f, "paper_points_max")
                    paper, competitor, where = TARGETS[(name, f)]
                    med = st[key]["median"]
                    if vname == "primary":
                        st["status"] = status(med, paper, ordering_holds=None if competitor is None
                                              else bool(med < competitor), reduced_scale=a.smoke)
                    st.update({"scored_on": key, "paper": paper, "paper_mean": PAPER_MEAN.get((name, f)),
                               "competitor": competitor, "where": where})
                    summary["fields"][f] = st
                    if vname == "primary":
                        ex.log_metric(f"{name}_{f}_median_test_max", med, "abs error",
                                      f"median over {len(runs)} seeds of max |u - exact| on {key}")
                        ex.check(f"{name}/{f}: {st['status']} (ours {med:.3g} vs paper {paper:.4g})",
                                 st["status"] == "reproduced", kind="reference", label="reproduction")
                if vname == "primary":
                    ex.check(f"{name}: constraints hold exactly (max {summary['constraint_err_max']:.2g})",
                             summary["constraint_err_max"] < 1e-8, kind="physics")
                    ex.check(f"{name}: fresh-point residual median {summary['test_residual']['median']:.2g}",
                             summary["test_residual"]["median"] < 1e-2, kind="physics")
                results["variants"].setdefault(name, {})[vname] = {"summary": summary, "runs": runs}
                parts = []
                for f, st in summary["fields"].items():
                    s = st[st["scored_on"]]
                    parts.append(f"{f}: med {s['median']:.3g} [{s['min']:.2g}, {s['max']:.2g}] {st.get('status', '')}")
                print(f"{name:5s} {vname:58s} {' '.join(parts)}  wall {wall:.1f}s", flush=True)
        out = os.path.join(HERE, "results_smoke.json" if a.smoke else "results.json")
        with open(out, "w") as fh:
            json.dump(results, fh, indent=1)
        if not a.smoke:
            write_report(results)


def write_report(results):
    rows = ["| Case | Field | Target (test max) | Ours: median [min, max] | Scored on | Seeds | Status | "
            "Paper mean | Our mean (median) | lstsq median | Wall (all seeds) |", "|" + "---|" * 11]
    labels, ours, papers, lo, hi = [], [], [], [], []
    for name, vs in results["variants"].items():
        s = vs["primary"]["summary"]
        for f, st in s["fields"].items():
            k = st["scored_on"]
            pm = f"{st['paper_mean']:.3g}" if st["paper_mean"] else "-"
            rows.append(f"| {name} | {f} | {st['paper']:.4g} | {st[k]['median']:.3g} [{st[k]['min']:.2g}, "
                        f"{st[k]['max']:.2g}] | {k} | {s['n_seeds']} | **{st['status']}** | {pm} | "
                        f"{st['test_mean']['median']:.3g} | {s['lstsq_s_median'] * 1e3:.1f} ms | {s['wall_s']:.0f} s |")
            labels.append(f"{name}/{f}")
            ours.append(st[k]["median"])
            lo.append(st[k]["min"])
            hi.append(st[k]["max"])
            papers.append(st["paper"])
    var_rows = ["| Case | Variant | Field | median [min, max] | converged |", "|---|---|---|---|---|"]
    for name, vs in results["variants"].items():
        for vname, v in vs.items():
            s = v["summary"]
            for f, st in s["fields"].items():
                k = st["scored_on"]
                var_rows.append(f"| {name} | {vname} | {f} | {st[k]['median']:.3g} [{st[k]['min']:.2g}, "
                                f"{st[k]['max']:.2g}] | {s['converged']}/{s['n_seeds']} |")
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(labels))
    ax.errorbar(x, ours, yerr=[np.subtract(ours, lo), np.subtract(hi, ours)], fmt="o", label="ours (median, min-max)")
    ax.scatter(x, papers, marker="x", color="k", zorder=3, label="paper (quoted)")
    ax.set_yscale("log")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("max abs test error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "targets_vs_ours.png"), dpi=120)
    with open(os.path.join(HERE, "RESULTS.md"), "w") as fh:
        fh.write("# X-TFC reproduction: measured results\n\nGenerated by `run_xtfc.py` from `results.json`. "
                 f"Machine: {results['machine']}, Python {results['python']}.\n\n## Primary protocol\n\n"
                 + "\n".join(rows) + "\n\n## All variants\n\n" + "\n".join(var_rows)
                 + "\n\n![targets vs ours](targets_vs_ours.png)\n")


if __name__ == "__main__":
    main()
