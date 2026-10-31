"""Reproduce Gated X-TFC (Dwivedi, Schiassi, Sigovan, Sixou, arXiv:2510.01039v1) with the protocol frozen in
TARGETS.md.

    python examples/reproductions/gated_xtfc_dwivedi2025/run_gated.py --part forward|ungated|inverse|twin|baselines
    python examples/reproductions/gated_xtfc_dwivedi2025/run_gated.py --part report
    python examples/reproductions/gated_xtfc_dwivedi2025/run_gated.py --smoke --part ...   # tiny sizes

Each part writes part_<name>.json next to this file; ``report`` scores them and writes RESULTS.md, figures and
experiment.json.
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
from pinneapple_simulation.numerical_solvers import gated_xtfc as gx  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
X_VAL = (np.arange(800) + 0.5) / 800                 # TARGETS.md ambiguity 2
X_TEST = np.linspace(0.0, 1.0, 20_000)               # ambiguity 3
FORWARD = {1e-2: (4.95e-4, 0.837, 99.93), 1e-3: (1.40e-4, 0.933, 44.43), 1e-4: (5.86e-4, 0.956, 52.76)}
INVERSE = {0.1: (0.0897, 10.3), 0.01: (0.00951, 4.9), 0.005: (0.00537, 7.4), 0.001: (0.00117, 17.0)}
TWIN = (1e-4, 2.97e-3)


def _save(name, obj, smoke):
    with open(os.path.join(HERE, f"part_{name}{'_smoke' if smoke else ''}.json"), "w") as fh:
        json.dump(obj, fh, indent=1)


def _score(sol):
    p = sol.problem
    m = evaluate(sol.predict(X_TEST), p.exact(X_TEST), ("max_abs", "mean_abs", "rel_l2"))
    r = sol.residual(X_TEST[1:-1])
    m["test_residual_max"] = float(np.max(np.abs(r)))
    m["bc_error"] = float(max(abs(sol.predict(np.array([0.0]))[0] - p.u_left),
                              abs(sol.predict(np.array([1.0]))[0] - p.u_right)))
    return m


def part_forward(smoke):
    n = 60 if smoke else 1000
    out = {}
    for nu in FORWARD:
        r = gx.search_gates_nested(gx.convection_diffusion(nu), X_VAL, n_centers=n,
                                   xs_tol=1e-2 if smoke else 1e-4, eps_tol=10.0 if smoke else 1e-1)
        out[str(nu)] = {"xs": r["xs"], "eps_scale": r["eps_scale"], "J_val": r["J"], "n_eval": r["n_evaluations"],
                        "seconds": r["seconds"], **_score(r["sol"])}
        print("forward", nu, {k: out[str(nu)][k] for k in ("xs", "eps_scale", "max_abs", "n_eval", "seconds")},
              flush=True)
    return out


def part_ungated(smoke):
    n = 300 if smoke else 5000
    nu = 1e-4
    t0 = time.perf_counter()
    sol = gx.solve_forward(gx.convection_diffusion(nu), gx.gated_layout([0.5], [10.0], nu, n))
    out = {"n_per_block": n, "seconds": time.perf_counter() - t0, **_score(sol)}
    print("ungated", out, flush=True)
    return out


def _clustered(n):
    return 1 - (1 - np.linspace(0.0, 1.0, n)) ** 3


def part_inverse(smoke):
    seeds = range(2 if smoke else 10)
    nc, ns, ne = (60, 40, 8) if smoke else (400, 300, 30)
    out = {}
    for nu_true in INVERSE:
        runs = []
        for s in seeds:
            x = _clustered(50)
            y = gx.convection_diffusion(nu_true).exact(x) + np.random.default_rng(1000 + s).normal(0, 1e-2, x.size)
            fit = gx.fit_inverse(gx.convection_diffusion, x, y, sigma=1e-2, beta_pde=1e2, n_centers=ns,
                                 n_colloc=nc, n_eval=ne, seed=s)
            runs.append({"seed": s, "nu_hat": fit.nu, "pct_error": 100 * abs(fit.nu - nu_true) / nu_true,
                         "xs": fit.splits[0], "eps_scale": fit.eps_scales[0], "log_evidence": fit.log_evidence,
                         "eta": fit.eta, "seconds": fit.seconds,
                         "field_max_abs": evaluate(fit.solution.predict(X_TEST),
                                                   gx.convection_diffusion(nu_true).exact(X_TEST),
                                                   ("max_abs",))["max_abs"]})
            print("inverse", nu_true, runs[-1]["nu_hat"], f"{runs[-1]['pct_error']:.1f}%", f"{fit.seconds:.0f}s",
                  flush=True)
        out[str(nu_true)] = runs
    return out


def part_twin(smoke):
    nu, _ = TWIN
    seeds = range(2 if smoke else 10)
    n, ne = (60, 8) if smoke else (1200, 60)
    runs = []
    for s in seeds:
        r = gx.search_gates_bo(gx.twin_boundary_layer(nu), X_VAL, split_bounds=[(0.001, 0.2), (0.8, 0.999)],
                               n_centers=n, n_eval=ne, seed=s)
        runs.append({"seed": s, "splits": r["splits"], "eps_scales": r["eps_scales"], "J_val": r["J"],
                     "seconds": r["seconds"], **_score(r["sol"])})
        print("twin", s, runs[-1]["max_abs"], runs[-1]["splits"], f"{r['seconds']:.0f}s", flush=True)
    return runs


def part_baselines(smoke):
    """Equal budget (2000 points, 2000 basis functions) at the three forward nu."""
    import torch

    from pinneapple_simulation.numerical_solvers.xtfc_general import (
        Constraint,
        XTFCProblem,
        grid,
        solve_xtfc,
    )
    n = 60 if smoke else 1000
    seeds = range(2 if smoke else 10)
    out = {}
    for nu in FORWARD:
        p = gx.convection_diffusion(nu)
        rec = {"ungated RBF X-TFC (x_s = 1/2)": _score(gx.solve_forward(p, gx.gated_layout([0.5], [10.0], nu, n)))}

        def exact_t(X, nu=nu):
            return torch.expm1(X[:, 0] / nu) / float(np.expm1(1 / nu)) if 1 / nu < 700 else \
                torch.exp((X[:, 0] - 1) / nu)
        prob = XTFCProblem([(0.0, 1.0)], {"u": [Constraint(0, 0.0), Constraint(0, 1.0)]}, {"u": exact_t},
                           lambda X, D, nu=nu: D[("u", (1,))] - nu * D[("u", (2,))], [("u", (1,)), ("u", (2,))],
                           linear=True)
        for label, wr, mode in (("tanh ELM X-TFC, unif(-1,1)", (-1, 1), "expression"),
                                ("tanh ELM X-TFC, unif(-10,10)", (-10, 10), "expression"),
                                ("tanh PIELM (BC rows), unif(-10,10)", (-10, 10), "rows")):
            vals = []
            for s in seeds:
                r = solve_xtfc(prob, grid(prob.bounds, 2 * n), n_neurons=2 * n, weight_range=wr, seed=s,
                               constraints=mode)
                vals.append(evaluate(r.predict(X_TEST[:, None]), p.exact(X_TEST), ("max_abs",))["max_abs"])
            rec[label] = seed_stats(vals).as_dict()
        out[str(nu)] = rec
        print("baselines", nu, {k: (v["max_abs"] if "max_abs" in v else v["median"]) for k, v in rec.items()},
              flush=True)
    return out


def report():
    def load(n):
        return json.load(open(os.path.join(HERE, f"part_{n}.json")))
    fw, ug, inv, tw, bl = (load(n) for n in ("forward", "ungated", "inverse", "twin", "baselines"))
    rows = ["| Target | Paper | Ours | Seeds | Status |", "|---|---|---|---|---|"]
    statuses = {}
    for nu, (paper, xs, es) in FORWARD.items():
        r = fw[str(nu)]
        st = status(r["max_abs"], paper)
        statuses[f"F nu={nu}"] = st
        rows.append(f"| Forward ν = {nu:g}: max abs error (paper x⋆s {xs}, ε⋆ {es}) | {paper:.3g} | {r['max_abs']:.3g} "
                    f"(x⋆s {r['xs']:.3f}, ε⋆ {r['eps_scale']:.1f}, {r['n_eval']} solves, {r['seconds']:.0f} s) "
                    f"| 1 (deterministic) | **{st}** |")
    g4 = fw[str(1e-4)]["max_abs"]
    first, second = g4 < ug["max_abs"], 1e-3 <= ug["max_abs"] < 1e-2
    st = "reproduced" if first and second else ("partially reproduced" if first else "not reproduced")
    statuses["F4"] = st
    rows.append(f"| Table 1, ν = 1e-4: gated (2000) < ungated (10000), ungated O(1e-3) | O(1e-4) < O(1e-3) | "
                f"{g4:.3g} vs {ug['max_abs']:.3g} (ungated {ug['seconds']:.0f} s) | 1 | **{st}** |")
    for nu, (nu_hat, pct) in INVERSE.items():
        runs = inv[str(nu)]
        ss = seed_stats([r["pct_error"] for r in runs])
        st = status(ss.median, pct)
        statuses[f"I nu={nu}"] = st
        rows.append(f"| Inverse ν = {nu:g}: % error of ν̂ (paper ν̂ = {nu_hat}) | {pct}% | {ss.median:.1f}% "
                    f"[{ss.min:.1f}, {ss.max:.1f}], ν̂ median {np.median([r['nu_hat'] for r in runs]):.3g} | "
                    f"{len(runs)} | **{st}** |")
    ts = seed_stats([r["max_abs"] for r in tw])
    st = status(ts.median, TWIN[1])
    statuses["T1"] = st
    rows.append(f"| Twin boundary layer ν = 1e-4: max abs error | {TWIN[1]:.3g} | {ts.median:.3g} [{ts.min:.2g}, "
                f"{ts.max:.2g}] | {len(tw)} | **{st}** |")
    brows = ["| ν | Method (2000 points, 2000 basis functions) | max abs test error |", "|---|---|---|"]
    for nu, rec in bl.items():
        brows.append(f"| {float(nu):g} | Gated X-TFC (this reproduction) | {fw[nu]['max_abs']:.3g} |")
        for k, v in rec.items():
            val = v["max_abs"] if "max_abs" in v else v["median"]
            extra = "" if "max_abs" in v else f" (median of {v['n_seeds']}; [{v['min']:.2g}, {v['max']:.2g}])"
            brows.append(f"| {float(nu):g} | {k} | {val:.3g}{extra} |")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, nu in zip(axes, FORWARD, strict=True):
        r = fw[str(nu)]
        p = gx.convection_diffusion(nu)
        sol = gx.solve_forward(p, gx.gated_layout([r["xs"]], [r["eps_scale"]], nu, 1000))
        ax.semilogy(X_TEST, np.abs(sol.predict(X_TEST) - p.exact(X_TEST)) + 1e-18)
        ax.axvline(r["xs"], ls="--", color="k", lw=0.8)
        ax.set_title(f"ν = {nu:g}: max |e| {r['max_abs']:.2g} (paper {FORWARD[nu][0]:.2g})", fontsize=9)
        ax.set_xlabel("x")
    axes[0].set_ylabel("|u_h - u_exact|")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "forward_errors.png"), dpi=120)
    with open(os.path.join(HERE, "RESULTS.md"), "w") as fh:
        fh.write("# Gated X-TFC reproduction: measured results\n\nGenerated by `run_gated.py --part report` from "
                 f"the part_*.json files. Machine: {platform.platform()}, Python {platform.python_version()}.\n\n"
                 + "\n".join(rows) + "\n\n## Baselines at equal budget\n\n" + "\n".join(brows)
                 + "\n\n![forward errors](forward_errors.png)\n")
    with pinneapple_lab.experiment(
            "examples/reproductions/gated_xtfc_dwivedi2025", title="Gated X-TFC reproduction (Dwivedi et al.)",
            description="Forward, inverse and twin-boundary-layer cases of arXiv:2510.01039v1 against the frozen "
                        "targets of TARGETS.md, with equal-budget X-TFC / PIELM baselines.",
            domain="physics-informed ML", methods=["Gated X-TFC", "X-TFC", "PIELM"],
            modules=["pinneapple_simulation.numerical_solvers.gated_xtfc", "pinneapple_analysis.reproduction"],
            out_dir=HERE, register=False,
            links={"targets": "examples/reproductions/gated_xtfc_dwivedi2025/TARGETS.md"}) as ex:
        plt.close("all")
        for name, st in statuses.items():
            ex.check(f"{name}: {st}", st == "reproduced", kind="reference", label="reproduction")
        for nu in FORWARD:
            r = fw[str(nu)]
            ex.log_metric(f"forward_max_abs_nu_{nu:g}", r["max_abs"], "abs error", "max |u - exact| on 2e4 points")
            ex.check(f"forward nu={nu:g}: boundary values exact ({r['bc_error']:.1g})", r["bc_error"] < 1e-12,
                     kind="physics")
            ex.check(f"forward nu={nu:g}: fresh-point residual max {r['test_residual_max']:.2g}", True,
                     kind="physics", detail="recorded, no threshold in the paper")
    print("\n".join(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["forward", "ungated", "inverse", "twin", "baselines", "report"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.part == "report":
        report()
        return
    _save(a.part, globals()[f"part_{a.part}"](a.smoke), a.smoke)


if __name__ == "__main__":
    main()
