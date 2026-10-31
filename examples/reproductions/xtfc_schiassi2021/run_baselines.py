"""Baselines against X-TFC on the paper's PDE problems, as fixed in TARGETS.md (Amendment 1).

    python examples/reproductions/xtfc_schiassi2021/run_baselines.py [--smoke]

ELM-only (constraints as least-squares rows) on all 7 PDEs; PINN via pp.solve on P1, P2, P4; FEM via pp.solve on
P1. X-TFC numbers are read from results.json (run_xtfc.py), so every method is scored on the same test points.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import pinneapple as pp
from pinneapple_analysis.reproduction import evaluate, seed_stats
from pinneapple_physics.paper_problems_xtfc import XTFC_CASES
from pinneapple_physics.pde_environment.builder import ProblemBuilder
from pinneapple_simulation.numerical_solvers.xtfc_general import grid, solve_xtfc

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_SEED = 20201


def test_set(case):
    lo, hi = np.array(case.problem.bounds).T
    return lo + (hi - lo) * np.random.default_rng(TEST_SEED).random((10_000, case.problem.ndim))


def exact_np(case):
    return lambda X: case.exact_np(np.asarray(X))[:, None]


def face(dim, at):
    return lambda X, ctx=None: np.isclose(X[:, dim], at, atol=1e-6)


def spec_for(name):
    """The problem in pp.solve's vocabulary (lap u = f convention for poisson)."""
    c = XTFC_CASES[name]
    ex = exact_np(c)
    if name == "pde1":
        b = ProblemBuilder("xtfc_p1").domain(x=(0.0, 1.0), y=(0.0, 1.0)).fields("u").pde("poisson")
        for d in (0, 1):
            for at in (0.0, 1.0):
                b = b.bc("dirichlet", field="u", value=ex, on=face(d, at))
        src = lambda X, ctx=None: (np.exp(-X[:, 0]) * (X[:, 0] - 2 + X[:, 1] ** 3 + 6 * X[:, 1]))[:, None]  # noqa: E731
        return b.build(), {"source_fn": src}
    if name == "pde2":
        b = (ProblemBuilder("xtfc_p2").domain(x=(0.0, 1.0), y=(0.0, 1.0)).fields("u").pde("poisson")
             .bc("dirichlet", field="u", value=0.0, on=face(0, 0.0))
             .bc("dirichlet", field="u", value=0.0, on=face(0, 1.0))
             .bc("dirichlet", field="u", value=0.0, on=face(1, 0.0))
             .bc("neumann", field="u", value=lambda X: (2 * np.sin(np.pi * X[:, 0]))[:, None], on=face(1, 1.0)))
        src = lambda X, ctx=None: ((2 - np.pi ** 2 * X[:, 1] ** 2) * np.sin(np.pi * X[:, 0]))[:, None]  # noqa: E731
        return b.build(), {"source_fn": src}
    if name == "pde4":
        b = (ProblemBuilder("xtfc_p4").domain(x=(0.0, 1.0), t=(0.0, 1.0)).fields("u").pde("heat_equation", alpha=1.0)
             .bc("dirichlet", field="u", value=0.0, on=face(0, 0.0))
             .bc("dirichlet", field="u", value=0.0, on=face(0, 1.0))
             .ic(field="u", fn=lambda X: np.sin(np.pi * X[:, 0])[:, None]))
        return b.build(), {}
    raise KeyError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    n_elm, n_pinn, epochs = (2, 1, 50) if a.smoke else (10, 3, 5000)
    xtfc = json.load(open(os.path.join(HERE, "results.json")))["variants"]
    out = {"amendment": "TARGETS.md, Amendment 1", "smoke": a.smoke, "cases": {}}
    for name in [n for n in sorted(XTFC_CASES) if n.startswith("pde")]:
        case = XTFC_CASES[name]
        f = next(iter(case.problem.fields))
        Xte = test_set(case)
        ref = case.exact_np(Xte)
        rec = {}
        prim = xtfc[name]["primary"]
        rec["X-TFC"] = {"test_max": prim["summary"]["fields"][f]["test_max"],
                        "test_rel_l2": prim["summary"]["fields"][f]["test_rel_l2"],
                        "wall_s_per_seed": prim["summary"]["total_s_median"],
                        "constraint_err_max": prim["summary"]["constraint_err_max"]}
        # ELM-only: same ELM, seeds, neurons, grid; constraints as rows
        t0 = time.perf_counter()
        runs = []
        for s in range(n_elm):
            it = None if case.problem.linear else case.paper_iterations + 5
            r = solve_xtfc(case.problem, grid(case.problem.bounds, case.n_train), n_neurons=case.n_neurons,
                           activation=case.activation, weight_range=case.weight_range, seed=s, max_iter=it,
                           tol=1e-14, constraints="rows")
            m = evaluate(r.predict(Xte), ref, ("max_abs", "rel_l2"))
            runs.append({**m, "constraint_err": r.constraint_error(), "s": r.total_seconds})
        rec["ELM-only"] = {"test_max": seed_stats([x["max_abs"] for x in runs]).as_dict(),
                           "test_rel_l2": seed_stats([x["rel_l2"] for x in runs]).as_dict(),
                           "wall_s_per_seed": (time.perf_counter() - t0) / n_elm,
                           "constraint_err_max": max(x["constraint_err"] for x in runs)}
        if name in ("pde1", "pde2", "pde4"):
            spec, ctx = spec_for(name)
            runs = []
            t0 = time.perf_counter()
            for s in range(n_pinn):
                sol = pp.solve(spec, "pinn", epochs=epochs, n_collocation=900, seed=s, ctx=ctx)
                runs.append(evaluate(np.asarray(sol.predict(Xte)).ravel(), ref, ("max_abs", "rel_l2")))
            rec["PINN (pp.solve)"] = {"test_max": seed_stats([x["max_abs"] for x in runs]).as_dict(),
                                      "test_rel_l2": seed_stats([x["rel_l2"] for x in runs]).as_dict(),
                                      "wall_s_per_seed": (time.perf_counter() - t0) / n_pinn,
                                      "epochs": epochs}
        if name == "pde1":
            spec, ctx = spec_for(name)
            t0 = time.perf_counter()
            sol = pp.solve(spec, "fem", n=29, ctx=ctx)
            m = evaluate(np.asarray(sol.predict(Xte)).ravel(), ref, ("max_abs", "rel_l2"))
            rec["FEM P1 (pp.solve, 30x30 nodes)"] = {"test_max": seed_stats([m["max_abs"]]).as_dict(),
                                                     "test_rel_l2": seed_stats([m["rel_l2"]]).as_dict(),
                                                     "wall_s_per_seed": time.perf_counter() - t0}
        out["cases"][name] = rec
        print(name, {k: f"{v['test_max']['median']:.3g} ({v['wall_s_per_seed']:.2f}s)" for k, v in rec.items()},
              flush=True)
    with open(os.path.join(HERE, "baselines_smoke.json" if a.smoke else "baselines.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    if not a.smoke:
        rows = ["| Case | Method | Test max abs: median [min, max] | rel L2 (median) | Seeds | Wall per seed |",
                "|---|---|---|---|---|---|"]
        for name, rec in out["cases"].items():
            for meth, v in rec.items():
                t = v["test_max"]
                rows.append(f"| {name} | {meth} | {t['median']:.3g} [{t['min']:.2g}, {t['max']:.2g}] | "
                            f"{v['test_rel_l2']['median']:.3g} | {t['n_seeds']} | {v['wall_s_per_seed']:.2f} s |")
        with open(os.path.join(HERE, "BASELINES.md"), "w") as fh:
            fh.write("# Baselines at the budgets of TARGETS.md (Amendment 1)\n\nGenerated by `run_baselines.py`. "
                     "Same 10 000 test points for every method.\n\n" + "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
