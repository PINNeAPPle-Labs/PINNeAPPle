"""How many CFD runs the cheap model buys: the multi-fidelity curve on a real entry capsule corpus, as a number and
as a field.

What it shows
-------------
1. the capsule-aeroheating file read through the adapter, the two fidelities paired by member -- 1.024 cheap
   panel-method runs and 64 laminar Navier-Stokes ones, on the same wall, node for node;
2. **per capsule**: for three targets (heat load, stagnation heat flux, shoulder heat flux) four strategies are fitted
   on N expensive drawings and scored on the ones they did not see -- the cheap model as it is, a surrogate from the
   design alone, a learned correction to the cheap model, and that correction with the cheap model's value among its
   inputs;
3. **per field**: the same question about the whole 2.361-node heat-flux field, where the correction is a few modes of
   the relative residual with their amplitudes learned from the six inputs. How deep the correction is (the share of
   the residual energy in the first modes) is measured, not assumed;
4. the two reports are the answer, and they disagree: the field's error is one shape with a design-dependent amplitude,
   which a handful of runs captures, where a scalar target such as the heat load has less structure to exploit.

Run: ``python examples/capsule_aeroheating/01_multifidelity.py`` (downloads the 77 MB wall file once, then ~1 min).
``--smoke`` runs 16 expensive designs with 2 splits.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from pinneapple_data.datasets import capsule_aeroheating as cap  # noqa: E402
from pinneapple_data.multifidelity import (  # noqa: E402
    FidelityPair,
    FieldPair,
    field_learning_curve,
    learning_curve,
    residual_rank,
)
from pinneapple_lab import experiment  # noqa: E402

TARGETS = {"heat_load": ("W", "the heat flux integrated over the heat shield"),
           "stagnation_heat_flux": ("W m-2", "the heat flux at the stagnation point"),
           "shoulder_heat_flux": ("W m-2", "the highest heat flux on the shoulder ring, where the panel method is "
                                            "blind to how the shoulder changes the flow")}
BUDGETS = (8, 16, 32, 64)
MODES = 3


def scalar_pairs(study) -> dict[str, FidelityPair]:
    """One ``FidelityPair`` per target: the cheap model at every design, the expensive one at the 64 that ran.

    The designs are the six inputs scaled by the bounds the file declares (``field_channels``), so a diameter in
    metres and a Mach number are on the same footing before any kernel sees them."""
    high, low = cap.pair_fidelities(study)
    _, channels, _ = cap.field_channels(study)
    fidelity = np.asarray(study.table(["fidelity"])["fidelity"], dtype=str)
    cheap = np.flatnonzero(fidelity == cap.LOW_FIDELITY)
    table = study.table(list(TARGETS))
    return {target: FidelityPair(design_low=channels[cheap], values_low=np.asarray(table[target])[cheap],
                                 design_high=channels[high], values_high=np.asarray(table[target])[high],
                                 low_at_high=np.asarray(table[target])[low])
            for target in TARGETS}


def scalar_report(ex, pairs, budgets, repeats) -> list[dict]:
    rows = []
    for target, pair in pairs.items():
        curve = learning_curve(pair, budgets=budgets, repeats=repeats)
        print(f"=== {target} -- {TARGETS[target][1]}")
        print(f"    {pair.summary()}\n")
        print("\n".join("    " + line for line in curve.table().splitlines()) + "\n")

        cheap = float(np.median(curve.scores["low_fidelity"]["relative_l2"][0]))
        first = float(np.median(curve.scores["difference"]["relative_l2"][0]))
        full = float(np.median(curve.scores["difference"]["relative_l2"][-1]))
        geometry = float(np.median(curve.scores["geometry"]["relative_l2"][-1]))
        rows.append({"target": target, "cheap": cheap, "first": first, "full": full, "geometry": geometry})

        ex.log_metric(f"{target}_cheap_model_error", cheap, unit="relative L2",
                      how=f"the panel method's own value against the Navier-Stokes one, all {pair.n_high} "
                          "expensive designs, no fit")
        ex.log_metric(f"{target}_difference_at_{budgets[0]}", first, unit="relative L2",
                      how=f"correction to the cheap model fitted on {budgets[0]} expensive designs, scored on the "
                          f"{pair.n_high - budgets[0]} it did not see, median over {repeats} splits")
        ex.log_metric(f"{target}_difference_at_{budgets[-1]}", full, unit="relative L2",
                      how=f"leave-one-out over the {pair.n_high} expensive designs" if budgets[-1] == pair.n_high
                          else f"correction fitted on {budgets[-1]} expensive designs, scored on the rest")
        ex.log_metric(f"{target}_design_only_at_{budgets[-1]}", geometry, unit="relative L2",
                      how="surrogate of the target from the six inputs alone, same split as the correction")
        ex.check(f"{target}: {budgets[0]} expensive designs beat the cheap model", passed=bool(first < cheap),
                 kind="baseline", detail=f"correction {first:.2%} against the cheap model's {cheap:.2%} (relative L2)")
        ex.check(f"{target}: the design alone is worse than the cheap model", passed=bool(geometry > cheap),
                 kind="baseline",
                 detail=f"design-only {geometry:.2%} against the cheap model's {cheap:.2%} at N={budgets[-1]}")
    return rows


def field_report(ex, pair: FieldPair, budgets, repeats, modes: int = MODES) -> dict[str, float]:
    """The same question about the whole field, with a low-rank correction instead of a network per node."""
    relative = pair.residual() / pair.run_scale().reshape(-1, 1)
    rank = {k: float(residual_rank(relative, components=k)["share"]) for k in (1, 2, 3, modes) if k <= modes}
    curve = field_learning_curve(pair, budgets=budgets, repeats=repeats, components=modes)

    print("=== the heat-flux field, node by node")
    print(f"    {pair.summary()}")
    print("    energy of the relative residual: " + ", ".join(f"{k} mode(s) {v:.1%}" for k, v in rank.items()) + "\n")
    print("\n".join("    " + line for line in curve.table().splitlines()) + "\n")

    cheap = float(np.median(curve.scores["low_fidelity"]["relative_l2"][0]))
    mean_correction = float(np.median(curve.scores["mean_correction"]["relative_l2"][0]))
    first = float(np.median(curve.scores["pod_correction"]["relative_l2"][0]))
    full = float(np.median(curve.scores["pod_correction"]["relative_l2"][-1]))
    for name, value in (("cheap_field_error", cheap), ("mean_correction_at_lowest_budget", mean_correction),
                        (f"pod_correction_at_{budgets[0]}", first), (f"pod_correction_at_{budgets[-1]}", full)):
        ex.log_metric(name, value, unit="relative L2",
                      how=f"relative L2 over each run's whole {pair.n_nodes}-node field, median over the held-out "
                          f"runs and {repeats} splits")
    for k, value in rank.items():
        ex.log_metric(f"residual_energy_in_{k}_mode{'s' if k > 1 else ''}", value, unit="share",
                      how="share of the energy of the relative residual (each run's residual over the magnitude of "
                          "its own cheap field) held by the first modes")
    ex.log_metric("nodes_per_run", float(pair.n_nodes), unit="nodes", how="the shared wall's node count")
    ex.check("the average correction halves the field's error", passed=bool(mean_correction < cheap / 2),
             kind="baseline",
             detail=f"{mean_correction:.2%} against the cheap field's {cheap:.2%}, with no dependence on the design")
    ex.check(f"{modes} modes learned from the design beat the average correction",
             passed=bool(first < mean_correction), kind="baseline",
             detail=f"{first:.2%} at N={budgets[0]} against {mean_correction:.2%}")
    ex.check("the field correction is a few modes deep", passed=bool(rank[modes] > 0.9), kind="physics",
             detail=f"{rank[modes]:.1%} of the relative residual's energy in {modes} modes")
    return {"cheap": cheap, "mean_correction": mean_correction, "first": first, "full": full}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="16 expensive designs, 2 splits")
    parser.add_argument("--data", default=os.path.join(HERE, "_out"), help="where the 77 MB file lives")
    args = parser.parse_args()

    path = cap.download("wall", args.data)
    budgets = (4, 8, 16) if args.smoke else BUDGETS
    repeats = 2 if args.smoke else 20
    print(f"file: {path}  ({os.path.getsize(path) / 1e6:.0f} MB)\n")

    with cap.open_study(path) as study:
        pairs = scalar_pairs(study)
        field_pair = cap.field_pair(study)
    if args.smoke:                                    # a subset of the expensive runs, the same machinery
        keep = 16
        pairs = {t: FidelityPair(design_low=p.design_low, values_low=p.values_low, design_high=p.design_high[:keep],
                                 values_high=p.values_high[:keep], low_at_high=p.low_at_high[:keep])
                 for t, p in pairs.items()}
        field_pair = FieldPair(design_high=field_pair.design_high[:keep], values_high=field_pair.values_high[:keep],
                               values_low=field_pair.values_low[:keep], position=field_pair.position)

    with experiment("capsule_aeroheating", title="How many CFD runs the cheap model buys",
                    domain="aerothermodynamics",
                    modules=["pinneapple_data.datasets.capsule_aeroheating", "pinneapple_data.multifidelity"],
                    links={"script": "examples/capsule_aeroheating/01_multifidelity.py",
                           "dataset": "https://huggingface.co/datasets/bijective/capsule-aeroheating"},
                    references=[cap.CITATION],
                    description="The capsule-aeroheating corpus (1.024 panel-method runs, 64 laminar Navier-Stokes "
                                "ones on the same wall) as multi-fidelity learning curves, per capsule and per field: "
                                "the cheap model, a surrogate from the design alone, a learned correction, and the "
                                "correction with the cheap model as an input; and, for the 2.361-node field, a few "
                                "modes of the relative residual whose amplitudes are learned from the design.") as ex:
        rows = scalar_report(ex, pairs, budgets, repeats)
        field = field_report(ex, field_pair, budgets, repeats)

    print(f"{'per capsule':24}{'cheap':>9}{'N=' + str(budgets[0]):>9}{'N=' + str(budgets[-1]):>9}"
          f"{'design only':>13}")
    for row in rows:
        print(f"{row['target']:24}{row['cheap']:9.2%}{row['first']:9.2%}{row['full']:9.2%}{row['geometry']:13.2%}")
    print(f"\n{'the field':24}{'cheap':>9}{'mean':>9}{'N=' + str(budgets[0]):>9}{'N=' + str(budgets[-1]):>9}")
    print(f"{'relative L2 per run':24}{field['cheap']:9.2%}{field['mean_correction']:9.2%}"
          f"{field['first']:9.2%}{field['full']:9.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
