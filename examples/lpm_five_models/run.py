"""The first five PINNeAPPle Large Physics Models, end to end: data, fit, held-out error and OOD report.

Each family is one design space with its own reference (a closed form, or the immersed finite-element solver):

    pp_fin        extruded-fin heat sink          fin temperature               1-D fin theory
    pp_airfoil    Joukowski airfoil               surface Cp                    exact potential flow
    pp_nozzle     converging-diverging nozzle     wall Mach and p/p_chamber      quasi-1-D isentropic
    pp_exchanger  shell-and-tube exchanger        dimensionless temperatures    Kern + Gnielinski + e-NTU
    pp_bracket    L-bracket under a tip load      displacement and von Mises    immersed FEM

The script draws geometries and operating points, runs the reference on each, fits one model per family on batched
designs, and reports on bodies the model never saw. Every number is measured against the family's own reference:
it is a statement about the fit, not a validation of the model for a real part, and the card of each run says so.

    python examples/lpm_five_models/run.py                       # all five, small enough for a laptop
    python examples/lpm_five_models/run.py --families pp_fin pp_nozzle --geometries 40 --epochs 400
"""
from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneapple_lab import experiment
from pinneapple_neural.lpm_pipeline import families, run

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", nargs="+", default=sorted(families()), choices=sorted(families()))
    ap.add_argument("--geometries", type=int, default=24)
    ap.add_argument("--conditions", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    known = families()
    titles = {name: fam.title for name, fam in known.items()}
    rows = {}
    started = time.time()
    for name in a.families:
        family = known[name]
        target = os.path.join(HERE, "results", name)
        summary = run(family, target, n_geometries=a.geometries, conditions_per_geometry=a.conditions,
                      seed=a.seed, epochs=a.epochs, batch_designs=min(24, a.geometries))
        rows[name] = summary
        print(f"{name:12s} held-out {summary['held_out_relative_l2']:.4f} "
              f"(p90 {summary['held_out_relative_l2_p90']:.4f}) vs constant-field "
              f"{summary['constant_field_baseline_relative_l2']:.3f} | OOD probes "
              f"{summary['ood_probes_flagged']} | {summary['seconds']:.0f} s")

    figures = []
    if rows:
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        names = list(rows)
        values = [rows[n]["held_out_relative_l2"] for n in names]
        base = [rows[n]["constant_field_baseline_relative_l2"] for n in names]
        pos = range(len(names))
        ax.bar([p - 0.2 for p in pos], values, width=0.4, label="model (held-out bodies)")
        ax.bar([p + 0.2 for p in pos], base, width=0.4, label="constant field (training average)")
        ax.set_xticks(list(pos))
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylabel("relative L2 against the family's reference")
        ax.set_title("One model per design space: held-out error against the constant-field baseline")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(HERE, "results", "held_out_vs_baseline.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150)
        figures.append(os.path.relpath(path, os.path.dirname(HERE)))

    with experiment("lpm_five_models",
                    title="The first five Large Physics Models: one model per design space",
                    description=("Five design spaces, five references (fin theory, exact potential flow, quasi-1-D "
                                 "isentropic flow, epsilon-NTU and immersed FEM), one trained model each. The models "
                                 "are scored on bodies held out by geometry, against the constant-field baseline "
                                 "and with an out-of-distribution report on probes outside each family."),
                    domain="physics AI / design-space surrogates",
                    methods=["Large Physics Model (geometry code, multi-scale neighbourhoods, ensemble heads, OOD)",
                             "query-to-surface cross-attention", "batched design-space training",
                             "split by geometry, not by design"],
                    modules=["pinneapple_neural.lpm", "pinneapple_neural.lpm_pipeline",
                             "pinneapple_neural.lpm_families"],
                    params={"families": list(a.families), "geometries": a.geometries,
                            "conditions_per_geometry": a.conditions, "epochs": a.epochs, "seed": a.seed},
                    references=["each family's own reference is recorded in its run card (name, citation, "
                                "assumptions) and repeated in the checks below",
                                "Milne-Thomson, Theoretical Hydrodynamics, 5th ed. (1968), ch. 6; Sutton & Biblarz, "
                                "Rocket Propulsion Elements, 9th ed. (2017), ch. 3; Incropera & DeWitt ch. 3 and 8"],
                    links={"pipeline": "pinneapple_neural/lpm_pipeline.py",
                           "families": "pinneapple_neural/lpm_families.py"},
                    register=False, curate=False) as ex:
        for name, summary in rows.items():
            ex.log_metric(f"{name}_held_out_relative_l2", summary["held_out_relative_l2"], unit="1",
                          how=f"relative L2 on bodies held out by geometry ({titles[name]})")
            ex.log_metric(f"{name}_held_out_p90", summary["held_out_relative_l2_p90"], unit="1",
                          how="90th percentile over the held-out designs")
            ex.log_metric(f"{name}_constant_field_baseline", summary["constant_field_baseline_relative_l2"],
                          unit="1", how="relative L2 of predicting the training-average field on the same designs")
            ex.log_metric(f"{name}_ensemble_std_mean", summary["ensemble_std_mean"], unit="field",
                          how="mean standard deviation between the bagged ensemble heads")
            ex.check(f"{name}_beats_the_constant_field",
                     passed=summary["held_out_relative_l2"] < 0.8 * summary["constant_field_baseline_relative_l2"],
                     kind="reference",
                     label=(f"{name}: held-out {summary['held_out_relative_l2']:.4f} against "
                            f"{summary['constant_field_baseline_relative_l2']:.4f} for the constant field"))
            flagged = int(summary["ood_probes_flagged"].split("/")[0])
            total = int(summary["ood_probes_flagged"].split("/")[1])
            ex.check(f"{name}_ood_probes", passed=flagged == total, kind="reference",
                     label=f"{name}: {flagged}/{total} probes outside the family flagged by the OOD rule")

    out = os.path.join(HERE, "results", "five_families.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"summary": rows, "seconds": round(time.time() - started, 1),
                   "experiment": "examples/lpm_five_models"}, handle, indent=2)
    print(f"\n{len(rows)} families in {time.time() - started:.0f} s; written to {out}")


if __name__ == "__main__":
    main()
