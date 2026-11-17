"""How many CFD runs the cheap model buys: the multi-fidelity curve on a real entry capsule corpus.

What it shows
-------------
1. the capsule-aeroheating file is read through the adapter, and the two fidelities are paired by member -- 1.024
   cheap panel-method runs and 64 laminar Navier-Stokes ones, on the same wall, node for node;
2. for three targets (heat load, stagnation heat flux, shoulder heat flux) the curve fits four strategies on N
   expensive drawings and scores them on the ones it did not see: the cheap model as it is, a surrogate from the
   design alone, a learned correction to the cheap model, and the same correction with the cheap model as an input;
3. the report is the answer: eight Navier-Stokes runs plus the panel method already beat the panel method alone on
   the shoulder -- where it is structurally wrong -- and the design alone never gets close, which is why the cheap
   model is worth having at all.

Run: ``python examples/capsule_aeroheating/01_multifidelity.py`` (downloads the 77 MB wall file once, then ~20 s).
``--smoke`` runs the same curve on 16 expensive designs with 2 repeats.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from pinneapple_data.datasets import capsule_aeroheating as cap  # noqa: E402
from pinneapple_data.multifidelity import FidelityPair, learning_curve  # noqa: E402
from pinneapple_lab import experiment  # noqa: E402

TARGETS = {"heat_load": ("W", "the heat flux integrated over the heat shield"),
           "stagnation_heat_flux": ("W m-2", "the heat flux at the stagnation point"),
           "shoulder_heat_flux": ("W m-2", "the highest heat flux on the shoulder ring, where the panel method is "
                                            "blind to how the shoulder changes the flow")}
BUDGETS = (8, 16, 32, 64)


def build_pairs(path: str, targets=TARGETS) -> dict[str, FidelityPair]:
    """One ``FidelityPair`` per target: the cheap model at every design, the expensive one at the 64 that ran.

    The designs are the six inputs scaled by the bounds the file declares (``field_channels``), so a diameter in
    metres and a Mach number are on the same footing before any kernel sees them."""
    pairs = {}
    with cap.open_study(path) as study:
        high, low = cap.pair_fidelities(study)
        names, channels, _ = cap.field_channels(study)
        table = study.table(list(targets))
        values = {target: np.asarray(table[target], dtype=np.float64) for target in targets}
        cheap = np.flatnonzero(np.asarray(study.table(["fidelity"])["fidelity"], dtype=str) == cap.LOW_FIDELITY)
        for target in targets:
            pairs[target] = FidelityPair(
                design_low=channels[cheap], values_low=values[target][cheap],
                design_high=channels[high], values_high=values[target][high],
                low_at_high=values[target][low], names=tuple(names))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="16 expensive designs, 2 repeats")
    parser.add_argument("--data", default=os.path.join(HERE, "_out"), help="where the 77 MB file lives")
    args = parser.parse_args()

    path = cap.download("wall", args.data)
    print(f"file: {path}  ({os.path.getsize(path) / 1e6:.0f} MB)\n")
    pairs = build_pairs(path)
    if args.smoke:                                    # a subset of the expensive designs, the same machinery
        pairs = {t: FidelityPair(design_low=p.design_low, values_low=p.values_low, design_high=p.design_high[:16],
                                 values_high=p.values_high[:16], low_at_high=p.low_at_high[:16], names=p.names)
                 for t, p in pairs.items()}

    budgets = (4, 8, 16) if args.smoke else BUDGETS
    repeats = 2 if args.smoke else 20
    curves, rows = {}, []
    with experiment("capsule_aeroheating", title="How many CFD runs the cheap model buys",
                    domain="aerothermodynamics",
                    modules=["pinneapple_data.datasets.capsule_aeroheating", "pinneapple_data.multifidelity"],
                    links={"script": "examples/capsule_aeroheating/01_multifidelity.py",
                           "dataset": "https://huggingface.co/datasets/bijective/capsule-aeroheating"},
                    references=[cap.CITATION],
                    description="The capsule-aeroheating corpus (1.024 panel-method runs, 64 laminar Navier-Stokes "
                                "ones on the same wall) as a multi-fidelity learning curve: the cheap model, a "
                                "surrogate from the design alone, a learned correction, and the correction with the "
                                "cheap model as an input, at N = 4 to 64 expensive designs.") as ex:
        for target, pair in pairs.items():
            curve = learning_curve(pair, budgets=budgets, repeats=repeats)
            curves[target] = curve
            print(f"=== {target} -- {TARGETS[target][1]}")
            print(f"    {pair.summary()}\n")
            print("\n".join("    " + line for line in curve.table().splitlines()) + "\n")

            unit = TARGETS[target][0]
            cheap = float(np.median(curve.scores["low_fidelity"]["relative_l2"][0]))
            at_eight = float(np.median(curve.scores["difference"]["relative_l2"][0]))
            at_full = float(np.median(curve.scores["difference"]["relative_l2"][-1]))
            geometry = float(np.median(curve.scores["geometry"]["relative_l2"][-1]))
            rows.append({"target": target, "cheap": cheap, "difference_at_8": at_eight, "difference_at_full": at_full,
                         "geometry_at_full": geometry, "unit": unit, "budget_full": int(budgets[-1])})

            ex.log_metric(f"{target}_cheap_model_error", cheap, unit="relative L2",
                          how=f"the panel method's own value against the Navier-Stokes one, all {pair.n_high} "
                              "expensive designs, no fit")
            ex.log_metric(f"{target}_difference_at_{budgets[0]}", at_eight, unit="relative L2",
                          how=f"correction to the cheap model fitted on {budgets[0]} expensive designs, scored on "
                              f"the {pair.n_high - budgets[0]} it did not see, median over {repeats} splits")
            ex.log_metric(f"{target}_difference_at_{budgets[-1]}", at_full, unit="relative L2",
                          how=f"leave-one-out over the {pair.n_high} expensive designs" if budgets[-1] == pair.n_high
                              else f"correction fitted on {budgets[-1]} expensive designs, scored on the rest")
            ex.log_metric(f"{target}_design_only_at_{budgets[-1]}", geometry, unit="relative L2",
                          how="surrogate of the target from the six inputs alone, same split as the correction")
            ex.log_metric(f"{target}_fidelity_gap", pair.low_at_high.mean() / pair.values_high.mean() - 1.0,
                          unit="1", how="mean of the cheap model over mean of the expensive one, minus 1: the bias")
            ex.check(f"{target}: {budgets[0]} expensive designs beat the cheap model",
                     passed=bool(at_eight < cheap), kind="baseline",
                     detail=f"correction {at_eight:.2%} against the cheap model's {cheap:.2%} (relative L2)")
            ex.check(f"{target}: the design alone is worse than the cheap model", passed=bool(geometry > cheap),
                     kind="baseline",
                     detail=f"design-only {geometry:.2%} against the cheap model's {cheap:.2%} at N={budgets[-1]}")

        print(f"{'target':24}{'cheap':>9}{'N=' + str(budgets[0]):>9}{'N=' + str(budgets[-1]):>9}"
              f"{'design only':>13}")
        for row in rows:
            print(f"{row['target']:24}{row['cheap']:9.2%}{row['difference_at_8']:9.2%}"
                  f"{row['difference_at_full']:9.2%}{row['geometry_at_full']:13.2%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
