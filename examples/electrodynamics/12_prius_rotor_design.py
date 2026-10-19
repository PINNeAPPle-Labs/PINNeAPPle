"""V-magnet rotor of the Prius motor: FEA + Gaussian-process surrogate + NSGA-II, against NSGA-II on the FEA alone.

Five rotor variables (magnet width and thickness, V depth and opening, bridge), two objectives (maximise torque at
250 A and 50 degrees, minimise magnet mass), torque ripple at most 30 %. The hybrid study (Latin hypercube start,
rounds of GP fit -> NSGA-II on the GP -> FEA on points along the predicted front, ``surrogate_assisted`` of
``pinneapple_design.design_optimizer``) is compared with NSGA-II calling the FEA for every individual, at equal
quality: the hypervolume of the feasible FEA-evaluated designs. Every design of the hybrid front is FEA-evaluated by
construction and re-evaluated at a finer FEA (480 gap nodes, 12 rotor angles).

    python examples/electrodynamics/12_prius_rotor_design.py            # 3 seeds, about 2 h
    python examples/electrodynamics/12_prius_rotor_design.py --quick    # 1 seed, small budgets
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from pinneapple_simulation.electromagnetics.machines import design as D

OUT = Path(__file__).parent / "results"


def main(quick: bool) -> dict:
    warnings.filterwarnings("ignore")
    t0 = time.perf_counter()
    seeds = (0,) if quick else (0, 1, 2)
    runs = []
    for seed in seeds:
        fea = D.rotor_evaluator(n_gap=288, n_positions=6)
        fine = D.rotor_evaluator(n_gap=480, n_positions=12, name="ipm-fea-fine")
        P = D.rotor_design_problem(fea)
        r = D.hybrid_vs_fea_only(P, n_initial=12 if quick else 24, rounds=2 if quick else 4, n_verify=4 if quick else 8,
                                 ga_budget=60 if quick else 2000, surrogate_budget=600 if quick else 2000, seed=seed,
                                 verifier=fine, progress=lambda s: print(s, flush=True))
        r["seed"] = seed
        rows = r["hybrid"]["finer_fea"]["rows"]["rows"]
        r["hybrid"]["finer_fea_summary"] = {
            "front_designs": len(rows),
            "feasible_at_finer_fea": int(sum(bool(x["feasible_reference"]) for x in rows)),
            "torque_rel_diff_max": float(max(abs(x["rel_error"]["torque"]) for x in rows)),
            "torque_rel_diff_mean": float(np.mean([abs(x["rel_error"]["torque"]) for x in rows])),
            "ripple_rel_diff_max": float(max(abs(x["rel_error"]["ripple"]) for x in rows)),
            "ripple_rel_diff_mean": float(np.mean([abs(x["rel_error"]["ripple"]) for x in rows]))}
        print("finer FEA re-evaluation of the hybrid front:", r["hybrid"]["finer_fea_summary"], flush=True)
        runs.append(r)
        print(f"seed {seed}: hybrid {r['hybrid']['fea_calls']} FEA calls, HV {r['hybrid']['hypervolume']:.4f}; "
              f"FEA-only reaches it after {r['fea_only']['calls_to_reach_hybrid']} calls "
              f"(final HV {r['fea_only']['hypervolume']:.4f} after {r['fea_only']['fea_calls']}); "
              f"saving {r['fea_call_saving']}", flush=True)
    # the Prius rotor itself, through the same coarse and fine FEA as the designs
    fea, fine = D.rotor_evaluator(n_gap=288, n_positions=6), D.rotor_evaluator(n_gap=480, n_positions=12)
    x0 = np.array([[getattr(fea.base.rotor, n) for n in fea.names]])
    y, yf = fea.evaluate(x0), fine.evaluate(x0)
    base = dict(zip(fea.names, map(float, x0[0]), strict=True))
    base.update({"torque": float(y["torque"][0]), "ripple": float(y["ripple"][0]),
                 "magnet_mass": float(y["magnet_mass"][0]), "torque_fine": float(yf["torque"][0]),
                 "ripple_fine": float(yf["ripple"][0])})
    print("Prius rotor:", base, flush=True)
    res = {"runs": runs, "seconds": time.perf_counter() - t0, "baseline": base}
    OUT.mkdir(exist_ok=True)
    (OUT / "12_prius_rotor_design.json").write_text(json.dumps(res, indent=1, default=float))
    _figure(runs, base)
    return res


def _figure(runs, base):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for r in runs:
        c = r["fea_only"]["hypervolume_curve"]
        ax[0].plot(np.arange(1, len(c) + 1), c, label=f"NSGA-II on FEA, seed {r['seed']}")
        ax[0].axhline(r["hybrid"]["hypervolume"], ls="--", lw=0.8, color=ax[0].lines[-1].get_color())
        ax[0].plot([r["hybrid"]["fea_calls"]], [r["hybrid"]["hypervolume"]], "o", color=ax[0].lines[-1].get_color())
        f = r["hybrid"]["front"]
        ax[1].plot([d["magnet_mass"] for d in f], [d["torque"] for d in f], "o", label=f"hybrid front, seed {r['seed']}")
    ax[1].plot([base["magnet_mass"]], [base["torque"]], "k*", ms=12, label="Prius rotor (same FEA)")
    ax[0].set_xlabel("designs evaluated by NSGA-II on the FEA (invalid geometries cost no FEA call)")
    ax[0].set_ylabel("hypervolume (normalised)")
    ax[0].set_title("dots: hybrid (GP + NSGA-II) at its FEA count")
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel("magnet mass (kg)")
    ax[1].set_ylabel("torque at 250 A (N m, FEA)")
    ax[1].legend(fontsize=8)
    sv = ", ".join(f"{100 * r['fea_call_saving']:.1f} %" for r in runs if r["fea_call_saving"] is not None)
    fig.text(0.01, 0.01, "Hypervolume of the feasible FEA-evaluated designs (objectives normalised by the ideal and "
             "nadir of both runs of a seed). FEA-call saving of the hybrid at equal hypervolume: " + sv + ". Right: "
             "coarse FEA (288 gap nodes, 6 angles) of the hybrid fronts; every design re-evaluated at 480 nodes, 12 "
             "angles (torque within 1.1 %, ripple up to 30 % higher). The fronts sit at the 1.0 mm bridge bound: the "
             "mechanical strength of the bridge is not a constraint here.", fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "12_prius_rotor_design.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main("--quick" in sys.argv)
