"""Physics learning verification, track 3: the measured study of docs/research/physics_condensation.md (section 6).

A 2-D die / TIM / lid section with a fast finite-volume reference (:mod:`pinneapple_analysis.validation.
physics_learning_cases`). Two studies, each kept in ``results/`` (rerun one with ``--force <study>``):

* ``detection`` -- the deliberately defective models at four severities and the controls, the verifier against the
  usual metrics (held-out relative L2, mean squared discrete residual), informative points against random ones at
  equal budgets (:func:`pinneapple_analysis.validation.physics_learning.detection_study`, seconds);
* ``models`` -- real learned models: a POD-GPR surrogate trained on correlated data and on independent samples, and
  a PINN with soft and with hard (architecture) side conditions (minutes: the PINNs train here).

Run: ``python examples/physics_learning_verification/run_study.py [--force all|detection|models]``.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import pinneapple_lab

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")


def detection():
    from pinneapple_analysis.validation.physics_learning import detection_study
    t = time.perf_counter()
    r = detection_study()
    r["seconds_total"] = time.perf_counter() - t
    return r


def models():
    from pinneapple_analysis.validation.physics_learning import baselines, verify
    from pinneapple_analysis.validation.physics_learning_cases import (
        Section,
        pinn_model,
        surrogate_model,
    )
    case = Section()
    out = []

    def rec(m, info, t_train):
        t = time.perf_counter()
        r = verify(m, case, k=48)
        out.append({"model": m.name, "inputs": list(m.inputs), "detected": r.detected, "decision": r.trust.decisao,
                    "failed": [f.line() for f in r.failed], "out_of_scope": [f.check for f in r.out_of_scope],
                    "findings": [{"check": f.check, "passed": f.passed, "provenance": f.provenance,
                                  "line": f.line()} for f in r.findings],
                    "verify_s": time.perf_counter() - t, "train_s": t_train, "info": info, **baselines(m, case)})

    for corr in (True, False):
        t = time.perf_counter()
        rec(surrogate_model(case, correlated=corr), {"correlated": corr}, time.perf_counter() - t)
    for hard in (False, True):
        t = time.perf_counter()
        m, info = pinn_model(case, hard_sides=hard)
        rec(m, {k: v for k, v in info.items() if np.isscalar(v)}, time.perf_counter() - t)
    return out


STUDIES = {"detection": detection, "models": models}


def run(name, force):
    path = os.path.join(RES, f"{name}.json")
    if os.path.exists(path) and not force:
        return json.load(open(path))
    print(f"-- {name}", flush=True)
    r = STUDIES[name]()
    json.dump(r, open(path, "w"), indent=1, default=str)
    return r


def figure_detection(det, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = det["rows"]
    defects = sorted({r["defect"] for r in rows if r["group"] == "defect"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.6, 1]})
    cols = (("verifier", "verifier", "#2a6fdb"), ("flag_loss", "physics loss > limit", "#e08a2a"),
            ("flag_heldout", "held-out L2 > limit", "#8a8a8a"))
    x = np.arange(len(defects))
    for i, (key, lab, c) in enumerate(cols):
        rate = [np.mean([r[key] for r in rows if r["defect"] == d]) for d in defects]
        a.bar(x + (i - 1) * 0.27, rate, 0.27, label=lab, color=c)
    a.set_xticks(x, [d.replace(" ", "\n") for d in defects], fontsize=8)
    a.set_ylabel("detection rate (over severities)")
    a.set_ylim(0, 1.05)
    a.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.15), ncol=3)
    budgets = det["budgets"]
    s = det["summary"]["defect"]
    b.plot(budgets, [s[f"informative_{k}"] for k in budgets], "o-", color="#2a6fdb", label="informative points")
    b.plot(budgets, [s[f"random_{k}"] for k in budgets], "s--", color="#8a8a8a", label="random points")
    b.set_xscale("log")
    b.set_xticks(budgets, [str(k) for k in budgets])
    b.minorticks_off()
    b.set_xlabel("points checked")
    b.set_ylabel("defects detected")
    b.set_ylim(0.5, 1.02)
    b.legend(fontsize=8, frameon=False)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", default="")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    force = set(a.force.split(","))
    with pinneapple_lab.experiment(
            "examples/physics_learning_verification", title="Physics learning verification of learned thermal models",
            description="A verifier independent of the training loss: discrete residual, interface flux, boundary "
                        "conditions, energy balance, symmetry, scaling and causal interventions against a finite-"
                        "volume reference, on deliberately defective models, a POD-GPR surrogate and PINNs.",
            domain="heat transfer", methods=["finite volumes", "PINN", "POD-GPR", "interventions"],
            modules=["pinneapple_analysis.validation.physics_learning"], out_dir=HERE, register=False,
            links={"report": "docs/research/physics_condensation.md",
                   "script": "examples/physics_learning_verification/run_study.py"}) as ex:
        det = run("detection", "all" in force or "detection" in force)
        ex.add_file(figure_detection(det, os.path.join(FIG, "detection.png")))
        s = det["summary"]
        for key, how in (("verifier", "verifier, 48 informative points + probes"),
                         ("flag_loss", "mean squared discrete residual > limit"),
                         ("flag_heldout", "held-out relative L2 > limit")):
            ex.log_metric(f"defect_detection_{key}", s["defect"][key], "fraction",
                          f"{how}; {s['defect']['n']} defective models")
        ex.check("verifier detects every deliberately defective model", s["defect"]["verifier"] == 1.0,
                 kind="verification")
        ex.check("verifier detects more defects than the physics loss or held-out error",
                 s["defect"]["verifier"] > max(s["defect"]["flag_loss"], s["defect"]["flag_heldout"]),
                 kind="verification")
        mod = run("models", "all" in force or "models" in force)
        by = {m["model"]: m for m in mod}
        for m in mod:
            ex.log_metric(f"heldout_rel_l2[{m['model']}]", m["heldout_rel_l2"], "", "200 points, 6 held-out cases")
        hard = [m for m in mod if "hard" in m["model"]]
        if hard:
            side = next(f for f in hard[0]["findings"] if f["check"] == "boundary:side_gradient")
            ex.check("hard side condition credited to the architecture", side["provenance"] == "architecture"
                     and side["passed"], kind="verification")
        corr = [m for m in by.values() if "correlated" in m["model"]]
        if corr:
            ex.check("surrogate on correlated data: low held-out error, rejected on interventions",
                     corr[0]["heldout_rel_l2"] < 0.01 and corr[0]["decision"] == "REJECT", kind="verification")


if __name__ == "__main__":
    main()
