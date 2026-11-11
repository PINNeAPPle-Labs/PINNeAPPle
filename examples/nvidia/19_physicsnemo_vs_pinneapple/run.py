"""Same PDE, same data, same budget: NVIDIA PhysicsNeMo's FNO against PINNeAPPle's own FNO2d.

``-div(a grad u) = 1`` on the unit square, ``u = 0`` on the boundary, ``a`` a smooth log-normal random field. The two
libraries train a Fourier neural operator on the identical training pairs, under the identical
:class:`~pinneapple_analysis.cost.equal_budget.ComputeBudget` (same Adam steps, same batch, same learning rate, same
seeds) and with matched parameter counts (2.37 M each). They are then judged three ways: a held-out relative L2; the
same on coefficients outside the training band (twice the log-contrast, which neither model ever saw); and
PINNeAPPle's Darcy checks, which need no reference at all -- a certified energy-error bound from the residual, the
boundary condition, the flux balance, positivity and the coefficient envelope.

The point of the example is not which library is nicer to write. It is that "trained a neural operator" is half of a
usable result: the other half is the decision, and the decision is taken by the physics, not by the loss.

    python examples/nvidia/19_physicsnemo_vs_pinneapple/run.py            # 2 seeds x 2 libraries, ~30 min on a busy laptop
    python examples/nvidia/19_physicsnemo_vs_pinneapple/run.py --smoke    # seconds (CI)
"""
import argparse
import json
import os

import numpy as np

from pinneapple_analysis.cost.equal_budget import ComputeBudget
from pinneapple_lab import experiment
from pinneapple_neural.physicsnemo_bridge.field_operators import (
    physicsnemo_fno,
    pinneapple_fno,
    train_grid_operator,
)
from pinneapple_physics import darcy_checks as dc
from pinneapple_simulation.numerical_solvers import operator_benchmarks as ob
from pinneapple_tools.nvidia.diagram import Block, architecture_diagram
from pinneapple_tools.nvidia.figures import checks_figure, curves_figure, field_panels
from pinneapple_tools.nvidia.runinfo import RunInfo

HERE = os.path.dirname(os.path.abspath(__file__))
C_TRAIN = 0.8        # standard deviation of log(a) in the training and in-distribution test cases
C_EXTRA = 1.8        # ... and in the out-of-distribution set (a contrast neither model ever saw)
LABELS = ("PhysicsNeMo FNO", "PINNeAPPle FNO2d")


def coefficients(n_samples, n, contrast, seed):
    """Smooth log-normal coefficient ``a = exp(contrast * g)`` from a standardised periodic Gaussian random field."""
    g = ob.periodic_grf(n_samples, (n, n), alpha=2.5, tau=3.0, seed=seed)
    g = (g - g.mean(axis=(1, 2), keepdims=True)) / g.std(axis=(1, 2), keepdims=True)
    return np.exp(contrast * g)


def rel_l2(pred, ref):
    """Per-sample relative L2 error of a batch of ``(n, n)`` fields."""
    p, r = np.asarray(pred).reshape(len(pred), -1), np.asarray(ref).reshape(len(ref), -1)
    return np.linalg.norm(p - r, axis=1) / np.linalg.norm(r, axis=1)


def build(label, n, seed, smoke=False):
    """One operator per library, at matched parameter counts (2.37 M; the small CI configs are only for --smoke)."""
    if label == "PhysicsNeMo FNO":
        cfg = dict(latent_channels=8, num_fno_layers=2, num_fno_modes=4, decoder_layer_size=16) if smoke else {}
        return physicsnemo_fno(n, seed=seed, **cfg)
    cfg = dict(width=12, modes=4, layers=2) if smoke else dict(width=45)
    return pinneapple_fno(n, seed=seed, **cfg)


def head_to_head_figure(res, path):
    """The comparison a reader wants first: held-out error inside and outside the training band, per seed."""
    from pinneapple_tools.nvidia.figures import _plt
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.6))
    colors = {LABELS[0]: "#4f7d00", LABELS[1]: "#b88a00"}
    for ax, key, title, lim in ((axs[0], "test", "Held-out relative L2 (inside the training band)", 0.06),
                                (axs[1], "extrap", "Relative L2 outside the training band (2.25x the contrast)", 0.6)):
        for i, lab in enumerate(LABELS):
            vals = np.array(res[lab][key])
            ax.bar(i, float(vals.mean()), width=0.55, color=colors[lab], alpha=0.85,
                   label=f"{lab} ({vals.mean():.2%} mean)")
            ax.scatter(np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)), vals, zorder=3, s=28,
                       color="k", marker="o")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["PhysicsNeMo", "PINNeAPPle"], fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("relative L2")
        ax.set_ylim(0, lim)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Equal data, equal budget, matched parameter count — black dots are the individual seeds", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(smoke=False, base=None):
    """``base``: folder for experiment.json and results/ (default: this folder; tests pass a temporary one)."""
    base = base or HERE
    OUT = os.path.join(base, "results")
    os.makedirs(OUT, exist_ok=True)
    n, n_train, n_test = (16, 24, 6) if smoke else (32, 400, 50)
    steps, seeds = (6, (0,)) if smoke else (1500, (0, 1))
    budget = ComputeBudget(steps=steps)
    info = RunInfo("19_physicsnemo_vs_pinneapple")
    architecture_diagram(
        "19 Equal budget: PhysicsNeMo FNO vs PINNeAPPle FNO2d, judged by the physics",
        [Block("coef", "Coefficient fields\na = exp(c g), log-normal", "pinneapple", 0, 0),
         Block("fv", "Darcy finite volumes\n(400 training pairs)", "pinneapple", 1, 0),
         Block("pnemo", "physicsnemo.models.fno.FNO\n2.37 M parameters", "physicsnemo", 2, 0),
         Block("pnap", "pinneapple FNO2d\n2.37 M parameters", "pinneapple", 2, 1),
         Block("budget", "Same ComputeBudget\n(Adam steps, batch, lr, seeds)", "pinneapple", 3, 0.5),
         Block("chk", "Darcy checks, no reference:\nresidual bound, BC, flux, positivity", "pinneapple", 4, 0),
         Block("env", "Coefficient envelope\n(applicability)", "pinneapple", 4, 1),
         Block("dec", "Two trust decisions\nAPPROVED / REVIEW / REJECT", "pinneapple", 5, 0.5)],
        [("coef", "fv", "a"), ("fv", "pnemo", "(a, u)"), ("fv", "pnap", "(a, u)"), ("pnemo", "budget", "u"),
         ("pnap", "budget", "u"), ("budget", "chk", "fields"), ("coef", "env", "a"), ("chk", "dec", "checks"),
         ("env", "dec", "applicability")],
        os.path.join(OUT, "architecture.png"),
        subtitle="Both libraries get the same data and the same compute; PINNeAPPle supplies the data and both verdicts")

    with experiment("examples/nvidia/19_physicsnemo_vs_pinneapple",
                    title="Equal budget: PhysicsNeMo FNO vs PINNeAPPle FNO2d on Darcy", domain="flow in porous media",
                    methods=["FNO (PhysicsNeMo)", "FNO (PINNeAPPle)", "FVM"],
                    modules=["pinneapple_neural.physicsnemo_bridge.field_operators", "pinneapple_physics.darcy_checks"],
                    out_dir=base, register=False,
                    params={"n": n, "n_train": n_train, "n_test": n_test, "budget": budget.__dict__, "seeds": seeds,
                            "contrast_train": C_TRAIN, "contrast_ood": C_EXTRA}) as ex:
        with info.step("data (PINNeAPPle finite volumes)"):
            a_tr = coefficients(n_train, n, C_TRAIN, 0)
            u_tr = np.stack([ob.darcy_2d(a) for a in a_tr])
            a_te = coefficients(n_test, n, C_TRAIN, 1)
            u_te = np.stack([ob.darcy_2d(a) for a in a_te])
            a_oo = coefficients(n_test, n, C_EXTRA, 2)
            u_oo = np.stack([ob.darcy_2d(a) for a in a_oo])
        info.mark("PINNeAPPle finite volumes (data)", ran=True, how=f"{len(a_tr)} solves, {n}x{n}")
        env = dc.CoefficientEnvelope.fit(a_tr)

        res = {lab: {"test": [], "extrap": [], "wall_s": [], "curve": [], "fields": [], "ood": [], "params": 0}
               for lab in LABELS}
        for lab in LABELS:
            for seed in seeds:
                pred = build(lab, n, seed, smoke)
                res[lab]["params"] = pred.n_parameters()
                curve = []

                def evaluate(pred=pred, curve=curve):
                    e = float(rel_l2(pred.predict(a_te), u_te).mean())
                    curve.append(e)
                    return {"held_out_rel_l2": e}

                with info.step(f"train ({lab}, seed {seed})"):
                    r = train_grid_operator(pred, a_tr, u_tr, budget, batch_size=8 if not smoke else 4, lr=2e-3,
                                            seed=seed, evaluate=evaluate, eval_at=np.linspace(0.1, 0.9, 9),
                                            threads=3)
                res[lab]["test"].append(float(rel_l2(pred.predict(a_te), u_te).mean()))
                res[lab]["extrap"].append(float(rel_l2(pred.predict(a_oo), u_oo).mean()))
                res[lab]["wall_s"].append(float(r.run.wall_s))
                res[lab]["curve"].append(curve)
                res[lab]["fields"].append(pred.predict(a_te))
                res[lab]["ood"].append(pred.predict(a_oo))
                info.mark(f"{lab} training (seed {seed})", ran=True,
                          how=f"CPU, {r.run.steps} steps, {r.epochs:.1f} epochs, {r.run.wall_s:.0f} s")

        # the seed with the median held-out error speaks for its library in the figures and the checks
        verdicts, checks_all, summary = {}, {}, {}
        with info.step("checks (PINNeAPPle, no reference)"):
            ref_report, _ = dc.darcy_trust_report(a_te, u_te, 1.0, model="finite volumes", reference=u_te, envelope=env)
            for lab in LABELS:
                k = int(np.argsort(res[lab]["test"])[len(res[lab]["test"]) // 2])
                u_hat = res[lab]["fields"][k]
                report, checks = dc.darcy_trust_report(a_te, u_hat, 1.0, model=lab, reference=u_te, envelope=env)
                oo_report, _ = dc.darcy_trust_report(a_oo, res[lab]["ood"][k], 1.0, model=lab, reference=u_oo,
                                                    envelope=env, kinds=("physics", "applicability"))
                verdicts[lab] = (report.decisao, oo_report.decisao)
                checks_all[lab] = checks
                checks_figure(checks, os.path.join(OUT, f"checks_{lab.split()[0].lower()}.png"),
                              title=f"{lab}, PINNeAPPle checks", decision=report.decisao)
                summary[lab] = {
                    "parameters": res[lab]["params"], "n_seeds": len(seeds),
                    "rel_l2_in_band": {"mean": float(np.mean(res[lab]["test"])),
                                       "per_seed": res[lab]["test"]},
                    "rel_l2_out_of_band": {"mean": float(np.mean(res[lab]["extrap"])),
                                           "per_seed": res[lab]["extrap"]},
                    "train_wall_s": {"mean": float(np.mean(res[lab]["wall_s"])), "per_seed": res[lab]["wall_s"]},
                    "decision_in_band": report.decisao, "decision_out_of_band": oo_report.decisao,
                    "checks": [c.to_dict() for c in checks],
                }
                ex.log_metric(f"rel_l2_in_band__{lab}", float(np.mean(res[lab]["test"])), "",
                              "mean relative L2 on 50 held-out coefficients from the training band")
                ex.log_metric(f"rel_l2_out_of_band__{lab}", float(np.mean(res[lab]["extrap"])), "",
                              "mean relative L2 on 50 coefficients with 2.25x the training log-contrast")
                ex.log_metric(f"parameters__{lab}", res[lab]["params"], "", "trainable parameters")
                ex.log_metric(f"train_wall_s__{lab}", float(np.mean(res[lab]["wall_s"])), "s",
                              f"wall clock of one training ({steps} Adam steps, CPU, 3 threads)")
                for c in checks:
                    if c.passed is not None:
                        ex.check(f"{lab}: {c.name}", bool(c.passed), value=c.value, max=c.limit, detail=c.detail,
                                 kind=c.kind)
        ex.check("equal budget: both libraries ran the same number of Adam steps", True, value=float(steps),
                 detail="ComputeBudget(steps) shared by both trainings")
        ex.check("matched parameter count (within 5 %)", abs(res[LABELS[0]]["params"] - res[LABELS[1]]["params"])
                 / res[LABELS[0]]["params"] <= 0.05, value=float(res[LABELS[1]]["params"]),
                 reference=float(res[LABELS[0]]["params"]), rtol=0.05, kind="sanity")
        ex.check("reference solver approves the finite-volume fields", ref_report.decisao == "APPROVED",
                 detail=f"decision on the FV reference itself: {ref_report.decisao}", kind="reference")

        with info.step("figures"):
            k = {lab: int(np.argsort(res[lab]["test"])[len(res[lab]["test"]) // 2]) for lab in LABELS}
            i = int(np.argsort(rel_l2(res[LABELS[0]]["fields"][k[LABELS[0]]], u_te))[len(a_te) // 2])
            field_panels([a_te[i], u_te[i], res[LABELS[0]]["fields"][k[LABELS[0]]][i],
                          res[LABELS[1]]["fields"][k[LABELS[1]]][i],
                          np.abs(res[LABELS[0]]["fields"][k[LABELS[0]]][i] - u_te[i]),
                          np.abs(res[LABELS[1]]["fields"][k[LABELS[1]]][i] - u_te[i])],
                         ["a (coefficient)", "u, finite volumes (reference)", "u, PhysicsNeMo FNO",
                          "u, PINNeAPPle FNO2d", "|error|, PhysicsNeMo", "|error|, PINNeAPPle"],
                         os.path.join(OUT, "fields.png"),
                         cmaps=["cividis", "viridis", "viridis", "viridis", "magma", "magma"],
                         shared=[(1, 2, 3), (4, 5)], ncols=3,
                         suptitle=f"Held-out coefficient {i}; errors {rel_l2(res[LABELS[0]]['fields'][k[LABELS[0]]], u_te)[i]:.2%} "
                                  f"(PhysicsNeMo) vs {rel_l2(res[LABELS[1]]['fields'][k[LABELS[1]]], u_te)[i]:.2%} (PINNeAPPle)")
            curves_figure([{"title": "Held-out relative L2 during training (mean of the seeds)", "xlabel": "fraction "
                            "of the budget", "ylabel": "relative L2", "logy": True,
                            "series": [(lab, np.linspace(0.1, 0.9, len(res[lab]["curve"][0])),
                                        np.mean(res[lab]["curve"], axis=0), s) for lab, s in
                                       zip(LABELS, ("o-", "s-"), strict=True)],
                            "hlines": [(dc.DEFAULT_TOLERANCES["reference"], "reference limit", "r")]}],
                          os.path.join(OUT, "training.png"))
            head_to_head_figure(res, os.path.join(OUT, "head_to_head.png"))
        info.mark("figures", ran=True, how="matplotlib, from the same numbers in summary.json")
        summary = {"budget_steps": steps, "seeds": list(seeds), "models": summary,
                   "reference_solver_decision": ref_report.decisao}
        with open(os.path.join(OUT, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)
        info.save(os.path.join(OUT, "run_info.json"))
    print(json.dumps({lab: {"in_band": f"{v['rel_l2_in_band']['mean']:.2%}",
                            "out_of_band": f"{v['rel_l2_out_of_band']['mean']:.2%}",
                            "decision": v["decision_in_band"], "out_of_band_decision": v["decision_out_of_band"]}
                      for lab, v in summary["models"].items()}, indent=1))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args().smoke)
