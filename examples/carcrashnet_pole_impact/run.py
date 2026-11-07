"""Pole impact of a bumper beam on its crash boxes — the crash benchmark, recreated here.

What is recreated, and with what:

| the benchmark | here |
|---|---|
| 14,742 OpenRadioss pole impacts of a DP1000 beam on DP600 crash boxes (7 design variables, 5 crashworthiness quantities) | the same case, at coupon scale, in the library's explicit hexahedral crash solver: 7 design variables, the same 5 quantities, a fixed mesh so that every design shares its nodes |
| 825 full-vehicle crashes (Neon / Yaris / Silverado) with a part hierarchy of 27-78 edited groups | a three-part coupon (beam + two boxes) and the same mesh refined to 6 and 11 semantic parts, the hierarchy depth the published ablations sweep |
| CrashSolver: per-part local attention -> global part transformer -> interface message passing -> temporal decoder | ``pinneapple_neural.architectures.crash_solver.CrashSolverModel``, the same decomposition |
| baselines: Transolver, GeoTransolver, FIGConvUNet | the library's Transolver and TransolverLite with time as an input channel, plus PointNet/EdgeConv — GeoTransolver and FIGConvUNet are not implemented here and no row pretends to be them |
| 11 tabular surrogates on the bumper-beam corpus (Ridge .. CatBoost) | every family scikit-learn/XGBoost/LightGBM provide here, with hyperparameters fixed across targets (the zero-tuning protocol) |

The solvers are different and so are the numbers: OpenRadioss with shells against this library's solid hexahedra with
J2 plasticity and penalty contact, at coupon scale. What the recreation is for is the *protocol* — one design vector,
one mesh, one split, a field surrogate and a tabular one on the same campaign — and the comparison of families on
equal terms. The run records the solver, the mesh, the design box and any run that left the solver's strain envelope,
so no number here can be read as an OpenRadioss result.

Run: ``python examples/carcrashnet_pole_impact/run.py --designs 120 --parallel 8``
(``--smoke``: 6 designs, short event, 2 epochs).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import pinneapple_lab  # noqa: E402
from pinneapple_arena.fieldbench import (  # noqa: E402
    FieldDataset,
    build_model_specs,
    paired_bootstrap,
    run_benchmark,
    subsample_nodes,
    write_report,
)
from pinneapple_arena.fieldbench.tabular import (  # noqa: E402
    benchmark_tabular,
    feature_sensitivity,
    load_table,
    write_table_report,
)
from pinneapple_simulation.numerical_solvers.crash_cases import DESIGN_BOUNDS  # noqa: E402

INPUT_KEYS = list(DESIGN_BOUNDS)
TARGET_KEYS = ["peak_wall_force", "peak_deceleration", "peak_internal_energy", "peak_plastic_work", "ke_absorption"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", type=int, default=120)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--end-time", type=float, default=1e-3, help="seconds of the event to simulate")
    ap.add_argument("--frames", type=int, default=11)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-parts", type=int, default=3)
    ap.add_argument("--max-nodes", type=int, default=900,
                    help="nodes kept per design for the field benchmark (the published one also scores on a retained "
                         "node subset); the dataset itself keeps every node")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "_out"))
    a = ap.parse_args()

    designs, end_time, epochs = (6, 4e-4, 2) if a.smoke else (a.designs, a.end_time, a.epochs)
    os.makedirs(a.out, exist_ok=True)
    npz = os.path.join(a.out, "pole_impact_fields.npz")
    csv = os.path.join(a.out, "pole_impact_quantities.csv")

    with pinneapple_lab.experiment(
            "examples/carcrashnet_pole_impact",
            title="Pole-impact crash benchmark: quantities and displacement fields",
            description="A bumper beam on two crash boxes driven into a rigid pole: seven design variables to the "
                        "crashworthiness quantities, and the undeformed mesh to the displacement sequence of every "
                        "node. Tabular families and field surrogates are compared on the same campaign, with the "
                        "part hierarchy the hierarchical surrogate exploits.",
            domain="structural crashworthiness",
            methods=["explicit hexahedral dynamics (J2 plasticity, penalty contact)", "gradient-boosted trees",
                     "physics-attention transformer", "hierarchical part-aware surrogate"],
            modules=["pinneapple_simulation.numerical_solvers.crash_fem",
                     "pinneapple_simulation.pipelines.pole_impact", "pinneapple_arena.fieldbench"],
            out_dir=HERE, register=False,
            links={"guide": "docs/guides/crash_and_surface_benchmarks.md",
                   "script": "examples/carcrashnet_pole_impact/run.py"}) as ex:
        # ---------------------------------------------------------------- 1. the campaign
        if a.reuse and os.path.exists(npz) and os.path.exists(csv):
            print(f"reused {npz} and {csv}")
            with open(os.path.join(a.out, "campaign.json")) as fh:
                summary = json.load(fh)["summary"]
        else:
            from pinneapple_simulation.pipelines.pole_impact import build_pole_impact_dataset
            summary = build_pole_impact_dataset(designs, a.out, seed=a.seed, parallel=a.parallel,
                                                end_time=end_time, frames=a.frames, n_parts=a.n_parts)
            print(json.dumps(summary, indent=2)[:800])
        ex.log_metric("designs_simulated", summary["n_samples"], "designs",
                      f"explicit hexahedral dynamics, end time {summary['end_time_s']} s, "
                      f"{summary['nodes']} nodes")
        ex.log_metric("runs_out_of_envelope", summary["out_of_envelope"], "designs",
                      "runs whose elements left the corotational strain envelope")
        ex.log_metric("elements_eroded_mean", summary["eroded_mean"], "elements", "mean per design at the end")
        ex.check("no run left the strain envelope", summary["out_of_envelope"] == 0,
                 detail=f"{summary['out_of_envelope']} of {summary['n_samples']} designs", kind="physics")

        # ---------------------------------------------------------------- 2. the quantities (tabular)
        table = load_table(csv, inputs=INPUT_KEYS, targets=TARGET_KEYS)
        ex.log_metric("tabular_rows", int(table["x"].shape[0]), "rows",
                      f"{table['dropped_out_of_envelope']} dropped as out of envelope")
        tabular = benchmark_tabular(table, seed=42)
        print("\n" + tabular["table"] + "\n")
        sens = feature_sensitivity(table, seed=42)
        write_table_report(tabular, os.path.join(a.out, "tabular_leaderboard.md"), sens)
        best_tab = tabular["ranked"][0]
        ex.log_metric("tabular_best_r2", tabular["scores"][best_tab]["mean"]["r2"], "R^2",
                      f"{best_tab}, mean over {len(TARGET_KEYS)} quantities, {tabular['n_test']} held-out rows")
        if sens.get("peak_wall_force"):
            force_top, force_w = max(sens["peak_wall_force"].items(), key=lambda kv: kv[1])
            ex.log_metric(f"importance_{force_top}", force_w, "relative importance",
                          "permutation importance of the design variables for peak wall force, averaged over the "
                          "tree models")

        # ---------------------------------------------------------------- 3. the fields
        dataset = FieldDataset.load(npz)
        full_nodes = int(np.mean([s.n_points for s in dataset.samples]))
        dataset = subsample_nodes(dataset, a.max_nodes, seed=a.seed)
        ex.log_metric("field_nodes_scored", int(np.mean([s.n_points for s in dataset.samples])), "nodes",
                      f"retained per design out of {full_nodes} in the mesh, the same subset for every model")
        specs = build_model_specs(transient=True, n_frames=a.frames, n_parts=max(a.n_parts, 4), scale="small")
        result = run_benchmark(specs, dataset, epochs=epochs, batch_size=a.batch, device=None, patience=15,
                               seed=a.seed, verbose=True, n_boot=1000)
        print("\n" + result["table"] + "\n")
        write_report(result, os.path.join(a.out, "field_leaderboard.md"))
        best = sorted(result["scores"], key=lambda s: s.rel_l2)[0]
        ex.log_metric("field_best_rel_l2", best.rel_l2, "relative L2",
                      f"{best.name}, mean over {result['n_test']} held-out designs, 95% CI {best.extra['ci95']}")
        ex.log_metric("field_best_r2", best.r2, "R^2", f"{best.name}, pooled over every node and frame")
        if len(result["scores"]) > 1:
            second = sorted(result["scores"], key=lambda s: s.rel_l2)[1]
            paired = paired_bootstrap(best.scores, second.scores)
            ex.log_metric("field_pair_gap", paired["mean_difference"], "relative L2",
                          f"{best.name} minus {second.name}, win rate {paired['win_rate']:.2f}, "
                          f"p_perm {paired['p_permutation']:.3g}")
        ex.check("a field surrogate beats the mean field", best.rel_l2 < 1.0,
                 detail=f"best {best.name} Rel L2 {best.rel_l2:.3f}", kind="regression")
        for name in ("tabular_leaderboard.md", "field_leaderboard.md"):
            ex.add_file(os.path.join(a.out, name))
        print(f"quantities: {csv}\nfields: {npz}")


if __name__ == "__main__":
    main()
