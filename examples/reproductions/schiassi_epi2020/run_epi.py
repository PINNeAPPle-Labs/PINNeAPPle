"""Reproduce Schiassi et al., X-TFC parameter discovery for SIR / SEIR / SEIRS (arXiv:2008.05554v1), with the
protocol frozen in TARGETS.md.

    python examples/reproductions/schiassi_epi2020/run_epi.py            # full protocol (minutes)
    python examples/reproductions/schiassi_epi2020/run_epi.py --smoke    # 2 seeds, writes *_smoke files

Writes results.json, RESULTS.md, a figure and experiment.json next to this file.
"""
from __future__ import annotations

import argparse
import json
import os
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import pinneapple_lab  # noqa: E402
from pinneapple_analysis.reproduction import seed_stats, status  # noqa: E402
from pinneapple_physics.paper_problems_epi import (  # noqa: E402
    EPI_MODELS,
    fit_shooting,
    synthetic_data,
)
from pinneapple_simulation.numerical_solvers.xtfc_inverse import solve_xtfc_inverse  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# Tables 1-3: delta -> |err| per parameter (and R0, reported only)
TARGETS = {
    "sir": {0: {"beta": 0, "gamma": 0, "R0": 0}, 0.1: {"beta": 4.2e-5, "gamma": 4.2e-5, "R0": 3.2e-4},
            1: {"beta": 4.4e-4, "gamma": 4.2e-4, "R0": 3.2e-3}, 5: {"beta": 2.2e-3, "gamma": 2.1e-3, "R0": 1.56e-2}},
    "seir": {0: {"beta": 0, "gamma": 0, "sigma": 0, "R0": 0},
             0.1: {"beta": 2.9e-3, "gamma": 3.7e-4, "sigma": 4.9e-4, "R0": 4.5e-3},
             1: {"beta": 2.9e-2, "gamma": 3.8e-3, "sigma": 4.8e-3, "R0": 4.5e-2},
             3: {"beta": 8.7e-2, "gamma": 1.2e-2, "sigma": 1.4e-2, "R0": 1.4e-1}},
    "seirs": {0: {"beta": 0, "gamma": 0, "sigma": 0, "zeta": 0, "R0": 0},
              0.1: {"beta": 1.1e-3, "gamma": 2.0e-3, "sigma": 2.6e-3, "zeta": 2.8e-3, "R0": 1.7e-4},
              1: {"beta": 9.3e-3, "gamma": 1.8e-2, "sigma": 2.5e-2, "zeta": 2.5e-2, "R0": 2.1e-4},
              3: {"beta": 1.7e-2, "gamma": 4.7e-2, "sigma": 6.8e-2, "zeta": 5.8e-2, "R0": 9.1e-3}},
}


VARIANTS = {
    "primary": "the frozen protocol of TARGETS.md (data-driven start, eq. 9 stopping)",
    "multistart": "post hoc: the data-driven start plus 8 random starts U(0, 2) per rate; lowest final loss kept",
    "oracle_start": "post hoc diagnostic, not an estimator: Gauss-Newton started at the true parameters",
}
N_RANDOM_STARTS = 8


def _fit(model, t, data, seed, params0=None):
    return solve_xtfc_inverse(model.problem(), t[:, None], data, model.residual, model.unknowns,
                              n_neurons=model.n_neurons, activation=model.activation,
                              weight_range=model.weight_range, seed=seed, map_to_unit=True, tol=1e-12, max_iter=200,
                              params0=params0)


def _errors(model, params):
    tp = model.true_params
    err = {k: abs(params[k] - tp[k]) for k in model.unknowns}
    err["R0"] = abs(params["beta"] / params["gamma"] - tp["beta"] / tp["gamma"])
    return err


def _summary(model, r, t, clean):
    X = t[:, None]
    fit = np.stack([r.predict(X, c) for c in model.compartments], axis=1)
    D = {(c, (a,)): torch.tensor(r.predict(X, c, (a,))) for c in model.compartments for a in (0, 1)}
    res = model.residual(None, D, {k: torch.tensor(v) for k, v in r.params.items()})
    return {"params": r.params, "err": _errors(model, r.params), "iterations": r.iterations, "stop": r.stop_reason,
            "seconds": r.seconds, "loss_rms": r.loss_rms[-1], "field_max_err_vs_clean": float(np.abs(fit - clean).max()),
            "ic_err": float(np.abs(fit[0] - np.array(model.y0)).max()),
            "residual_rms": float(np.sqrt(np.mean([float((x ** 2).mean()) for x in res])))}


def run_case(model, delta, seed):
    t, clean, y = synthetic_data(model, delta, seed=seed)
    data = {c: (t[:, None], y[:, i]) for i, c in enumerate(model.compartments)}
    r = _fit(model, t, data, seed)
    out = {"seed": seed, "params_init": r.params_init, "primary": _summary(model, r, t, clean)}
    rng = np.random.default_rng(1000 + seed)
    best, total = r, r.seconds
    for _ in range(N_RANDOM_STARTS):
        r2 = _fit(model, t, data, seed, {k: float(rng.uniform(0.0, 2.0)) for k in model.unknowns})
        total += r2.seconds
        if np.isfinite(r2.loss_rms[-1]) and r2.loss_rms[-1] < best.loss_rms[-1]:
            best = r2
    out["multistart"] = {**_summary(model, best, t, clean), "seconds": total}
    out["oracle_start"] = _summary(model, _fit(model, t, data, seed, model.true_params), t, clean)
    base = fit_shooting(model, t, y, r.params_init)
    out["baseline"] = {"params": base["params"], "seconds": base["seconds"], "nfev": base["nfev"],
                       "err": _errors(model, base["params"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    n_seeds = 2 if a.smoke else 10
    run_dir = os.path.join(HERE, "runs_smoke" if a.smoke else "runs")
    os.makedirs(run_dir, exist_ok=True)
    results = {"machine": platform.platform(), "smoke": a.smoke, "variants": VARIANTS, "models": {}}
    for name, m in EPI_MODELS.items():
        results["models"][name] = {}
        for delta in TARGETS[name]:
            runs = []
            for s in ([0] if delta == 0 else range(n_seeds)):
                path = os.path.join(run_dir, f"{name}_d{delta}_s{s}.json")
                if os.path.exists(path):
                    with open(path) as fh:
                        runs.append(json.load(fh))
                    continue
                runs.append(run_case(m, delta, s))
                with open(path, "w") as fh:
                    json.dump(runs[-1], fh)
            results["models"][name][str(delta)] = runs
            for v in ("primary", "multistart", "oracle_start", "baseline"):
                med = {k: float(np.median([r[v]["err"][k] for r in runs])) for k in runs[0][v]["err"]}
                print(name, delta, v, {k: f"{x:.2g}" for k, x in med.items()}, flush=True)
    out = os.path.join(HERE, "all_runs_smoke.json" if a.smoke else "all_runs.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1)
    if not a.smoke:
        report(results)


def _fmt(st):
    return f"{st.median:.2g} [{st.min:.2g}, {st.max:.2g}]"


def report(results):
    rows = ["| Model | δ | Parameter | Paper \\|err\\| | Primary: median [min, max] | Status | Multi-start (post hoc) "
            "| Status (post hoc) | Oracle start | Shooting baseline |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    statuses, post_hoc = {}, {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (name, by_delta) in zip(axes, results["models"].items(), strict=True):
        for k in EPI_MODELS[name].unknowns + ("R0",):
            xs, curves = [], {"primary": [], "multistart": [], "paper": []}
            for d, runs in by_delta.items():
                p = TARGETS[name][float(d) if "." in d else int(d)][k]
                st = {v: seed_stats([r[v]["err"][k] for r in runs]) for v in ("primary", "multistart", "oracle_start",
                                                                             "baseline")}
                if k == "R0":
                    s1 = s2 = "reported"
                else:
                    s1, s2 = status(st["primary"].median, p), status(st["multistart"].median, p)
                    statuses[f"{name} δ={d} {k}"], post_hoc[f"{name} δ={d} {k}"] = s1, s2
                rows.append(f"| {name.upper()} | {d} | {k} | {p:.2g} | {_fmt(st['primary'])} | **{s1}** | "
                            f"{_fmt(st['multistart'])} | {s2} | {st['oracle_start'].median:.2g} | "
                            f"{st['baseline'].median:.2g} |")
                if float(d) > 0 and k != "R0":
                    xs.append(float(d))
                    curves["primary"].append(st["primary"].median)
                    curves["multistart"].append(st["multistart"].median)
                    curves["paper"].append(p)
            if k != "R0":
                line = ax.loglog(xs, curves["paper"], "x--", label=f"{k} paper")[0]
                ax.loglog(xs, curves["primary"], "o-", color=line.get_color(), label=f"{k} primary")
                ax.loglog(xs, curves["multistart"], "s:", color=line.get_color(), alpha=0.6,
                          label=f"{k} multi-start")
        ax.set_title(name.upper())
        ax.set_xlabel("noise δ")
        ax.legend(fontsize=6, ncol=2)
    axes[0].set_ylabel("|err| (median over seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "errors_vs_noise.png"), dpi=120)
    plt.close(fig)
    n_rep = sum(v == "reproduced" for v in statuses.values())
    n_post = sum(v == "reproduced" for v in post_hoc.values())
    timing = {name: {v: float(np.median([r[v]["seconds"] for runs in by_delta.values() for r in runs]))
                     for v in ("primary", "multistart", "baseline")} for name, by_delta in results["models"].items()}
    lines = [f"- {n.upper()}: primary {t['primary']:.2g} s, multi-start {t['multistart']:.2g} s, "
             f"shooting {t['baseline']:.2g} s (median per run)" for n, t in timing.items()]
    with open(os.path.join(HERE, "RESULTS.md"), "w") as fh:
        fh.write("# X-TFC parameter discovery (Schiassi et al. 2020): measured results\n\n"
                 f"Generated by `run_epi.py`. Machine: {results['machine']}.\n\n"
                 f"- **Primary (frozen protocol): {n_rep} of {len(statuses)} scored targets reproduced.**\n"
                 f"- Multi-start (post hoc, not the frozen protocol): {n_post} of {len(post_hoc)}.\n"
                 "- Oracle start is a diagnostic (it uses the true values) and is not scored.\n\n"
                 + "\n".join(f"- `{k}`: {v}" for k, v in results["variants"].items()) + "\n\n"
                 + "\n".join(rows) + "\n\nWall time:\n\n" + "\n".join(lines)
                 + "\n\n![errors vs noise](errors_vs_noise.png)\n")
    with pinneapple_lab.experiment(
            "examples/reproductions/schiassi_epi2020", title="X-TFC epidemiological parameter discovery (Schiassi)",
            description="SIR, SEIR and SEIRS parameters recovered by X-TFC from clean and noisy data, against the "
                        "frozen targets of TARGETS.md, with a single-shooting least-squares baseline.",
            domain="physics-informed ML", methods=["X-TFC inverse", "single shooting"],
            modules=["pinneapple_simulation.numerical_solvers.xtfc_inverse", "pinneapple_physics.paper_problems_epi",
                     "pinneapple_analysis.reproduction"], out_dir=HERE, register=False,
            links={"targets": "examples/reproductions/schiassi_epi2020/TARGETS.md"}) as ex:
        for k, v in statuses.items():
            ex.check(f"{k}: {v}", v == "reproduced", kind="reference", label="reproduction")
        for name, by_delta in results["models"].items():
            r0 = by_delta["0"][0]["primary"]
            ex.check(f"{name}: initial values exact ({r0['ic_err']:.1g})", r0["ic_err"] < 1e-9, kind="physics")
            ex.log_metric(f"{name}_residual_rms_clean", r0["residual_rms"], "population/day",
                          "RMS of the fitted system's residual at the data points, clean data")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
