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
    extreme_errors,
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
DESIGN_CHANNELS = len(DESIGN_BOUNDS)


def peak_figure(kpi: dict, test_ids: list, path: str, model: str, conditioning: str) -> str:
    """Reference against predicted peak displacement for every held-out design — the quantity-level parity plot."""
    import matplotlib.pyplot as plt

    ref, got = kpi["reference"] * 1e3, kpi["prediction"] * 1e3           # mm, signed (the beam travels into the pole)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    span = max(ref.max() - ref.min(), got.max() - got.min(), 1e-9)
    lo = min(ref.min(), got.min()) - 0.10 * span
    hi = max(ref.max(), got.max()) + 0.10 * span
    ax.plot([lo, hi], [lo, hi], color="#8b949e", lw=0.9, ls="--", label="perfect prediction")
    sc = ax.scatter(ref, got, c=kpi["relative"] * 100, cmap="magma_r", s=42, edgecolor="k", linewidth=0.4, zorder=3)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("relative error of the peak (%)")
    bias = float(np.mean(got - ref))
    ax.set_xlabel("reference peak displacement (mm), explicit FE")
    ax.set_ylabel(f"predicted peak displacement (mm), {model}")
    ax.set_title(f"{kpi['n']} held-out designs — conditioning: {conditioning}\n"
                 f"mean error {kpi['mean_relative']:.1%}  ·  median {kpi['median_relative']:.1%}  ·  "
                 f"worst {kpi['max_relative']:.1%}  ·  signed bias {bias:+.2f} mm", fontsize=9.5)
    ax.annotate(f"the cloud sits {abs(bias):.2f} mm above the diagonal:\nthe surrogate under-predicts the peak",
                xy=(float(np.mean(ref)), float(np.mean(got))), xytext=(0.04, 0.72), textcoords="axes fraction",
                fontsize=8, color="#5a626d",
                arrowprops={"arrowstyle": "->", "color": "#5a626d", "lw": 0.8})
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path


def surface_figure(sample, prediction, path: str, model: str, sample_id: str) -> str:
    """The deformed bumper beam as a *surface*: reference and surrogate side by side, same colour scale.

    The benchmark is scored on a retained node subset, so the surrogate's field is carried back to the full mesh by
    inverse-distance weighting of its four nearest retained nodes before it is drawn — the shape and the numbers of the
    left panel are the solver's, and the right panel is the surrogate deformed and coloured by its own prediction."""
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from pinneapple_simulation.numerical_solvers.crash_cases import bumper_beam_assembly

    design = dict(sample.meta["design"])
    design.pop("velocity", None)
    nodes = np.asarray(bumper_beam_assembly(**design).mesh.nodes, float) * 1e3     # mm: same unit as the fields
    ref = np.asarray(sample.values)[-1] * 1e3                                      # (N, 3) mm, last frame
    n_sub = prediction.shape[1]
    if n_sub == ref.shape[0]:
        pred = np.asarray(prediction)[-1] * 1e3
    else:                                                                # interpolate back to the full mesh
        sub_xyz = np.asarray(sample.points)[:n_sub] * 1e3
        if "coord_scale" in sample.meta:                                 # stored coordinates are normalised
            sub_xyz = (np.asarray(sample.points)[:n_sub] * float(sample.meta["coord_scale"])
                       + np.asarray(sample.meta["coord_shift"])) * 1e3
        pred_sub = np.asarray(prediction)[-1] * 1e3
        d2 = ((nodes[:, None, :] - sub_xyz[None, :, :]) ** 2).sum(-1)
        k = min(4, n_sub)
        nearest = np.argpartition(d2, k - 1, axis=1)[:, :k]                # four nearest retained nodes, IDW weight
        w = 1.0 / np.maximum(np.take_along_axis(d2, nearest, axis=1), 1e-12)
        pred = (pred_sub[nearest] * w[..., None]).sum(1) / w.sum(1)[:, None]
    mag_ref, mag_pred = np.linalg.norm(ref, axis=1), np.linalg.norm(pred, axis=1)
    vmax = float(max(mag_ref.max(), mag_pred.max()))
    Q = np.asarray(bumper_beam_assembly(**design).mesh.boundary_faces())
    cmap, norm = plt.get_cmap("jet"), Normalize(0.0, vmax)
    fig = plt.figure(figsize=(15, 6.4))
    fig.subplots_adjust(left=0.0, right=0.9, top=0.88, bottom=0.0, wspace=0.0)
    for i, (label, u, m) in enumerate((
            ("reference — explicit FE", ref, mag_ref),
            (f"surrogate — {model}", pred, mag_pred))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        P = nodes[Q] + 2.0 * u[Q]                                        # 2x so the crush is visible
        fc = cmap(norm(m[Q].mean(1)))
        ax.add_collection3d(Poly3DCollection(P, facecolors=fc, edgecolors=(0, 0, 0, 0.35), linewidths=0.15))
        lo, hi = nodes.min(0), nodes.max(0)
        c, span = 0.5 * (lo + hi), (hi - lo)
        ax.set_xlim(c[0] - span[0] / 2, c[0] + span[0] / 2)
        ax.set_ylim(c[1] - span[1] / 2, c[1] + span[1] / 2)
        ax.set_zlim(c[2] - span[2] / 2, c[2] + span[2] / 2)
        ax.set_box_aspect(tuple(span), zoom=1.15)
        ax.view_init(20, -122)
        ax.set_axis_off()
        ax.set_title(f"{label}\npeak u_x = {float(np.min(u[:, 0])):+.2f} mm", fontsize=10)
    cax = fig.add_axes([0.915, 0.18, 0.012, 0.62])
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("|u| (mm), deformed shape scaled 2×")
    fig.suptitle(f"Design {sample_id} — displacement magnitude at the end of the event", fontsize=11)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path


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
    ap.add_argument("--conditioning", choices=["geometry", "design"], default="geometry",
                    help="'design' feeds the seven design variables to the field surrogates as per-node channels, so "
                         "a held-out design can be asked about at all; 'geometry' keeps the original protocol, where "
                         "a model can only interpolate between the geometries it has seen")
    ap.add_argument("--models", default="transolver",
                    help="comma-separated model rows of the field benchmark (see build_model_specs)")
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
        if a.conditioning == "design":
            from pinneapple_simulation.pipelines.pole_impact import with_design_channels
            dataset = with_design_channels(dataset)
        full = dataset                                                    # every node: the figures draw the surface
        full_nodes = int(np.mean([s.n_points for s in dataset.samples]))
        dataset = subsample_nodes(dataset, a.max_nodes, seed=a.seed)
        ex.log_metric("field_nodes_scored", int(np.mean([s.n_points for s in dataset.samples])), "nodes",
                      f"retained per design out of {full_nodes} in the mesh, the same subset for every model")
        ex.log_metric("field_design_channels", DESIGN_CHANNELS if a.conditioning == "design" else 0, "channels",
                      "the seven design variables, mapped to [0, 1] inside their own box and broadcast to every node"
                      if a.conditioning == "design" else "design channels not fed: the models see geometry only")
        # the field is the displacement *vector*: three components per node, so every row is built for three outputs.
        # With out_dim=1 torch broadcast the missing channels and the rows were scored against a three-component field
        # they never predicted — the field benchmark of this example reported numbers for a scalar field until this.
        field_dim = int(np.asarray(dataset.samples[0].values).shape[-1])
        specs = build_model_specs(transient=True, n_frames=a.frames, n_parts=max(a.n_parts, 4), scale="small",
                                  out_dim=field_dim,
                                  extra_dim=DESIGN_CHANNELS if a.conditioning == "design" else 0)
        wanted = {m.strip() for m in a.models.split(",") if m.strip()}
        unknown = wanted - {s["name"] for s in specs}
        if unknown:
            raise SystemExit(f"unknown model row(s) {sorted(unknown)}; available: {[s['name'] for s in specs]}")
        specs = [s for s in specs if s["name"] in wanted]
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

        # ---------------------------------------------------------------- 4. the KPI of the held-out designs
        # The field metrics say how well the whole field is drawn; the design was run for one number. Both come from
        # the same predictions, and the second is what a crash report quotes — the peak displacement of the beam.
        test_ids = dataset.splits["test"]
        truths = [dataset[i].target() for i in test_ids]
        kpi = extreme_errors(result["predictions"][str(best.name)], truths, channel=0, mode="min", n_boot=1000)
        ref_mm = kpi["reference"] * 1e3
        got_mm = kpi["prediction"] * 1e3
        worst = int(np.argmax(kpi["relative"]))
        median = int(np.argsort(kpi["relative"])[len(kpi["relative"]) // 2])
        ex.log_metric("kpi_peak_displacement_mean_rel_error", kpi["mean_relative"], "relative error",
                      f"{best.name}, peak displacement along the impact direction (min u_x), mean over "
                      f"{kpi['n']} held-out designs, 95% CI {kpi['mean_relative_ci95']}")
        ex.log_metric("kpi_peak_displacement_median_rel_error", kpi["median_relative"], "relative error",
                      f"{best.name}, median over the same {kpi['n']} designs")
        ex.log_metric("kpi_peak_displacement_max_rel_error", kpi["max_relative"], "relative error",
                      f"{best.name}, worst held-out design ({test_ids[worst]})")
        print(f"peak displacement on the held-out designs: mean relative error {kpi['mean_relative']:.2%}, "
              f"median {kpi['median_relative']:.2%}, worst {kpi['max_relative']:.2%} ({test_ids[worst]}); "
              f"representative design {test_ids[median]}: reference {ref_mm[median]:.2f} mm, "
              f"surrogate {got_mm[median]:.2f} mm")
        ex.check("the surrogate tracks the peak displacement of held-out designs", kpi["mean_relative"] < 0.25,
                 detail=f"{best.name}, mean relative error {kpi['mean_relative']:.2%} over {kpi['n']} designs",
                 kind="regression")
        with open(os.path.join(a.out, "kpi_peak_displacement.json"), "w") as fh:
            json.dump({"model": str(best.name), "conditioning": a.conditioning, "unit": "m",
                       "channel": 0, "mode": "min (peak displacement along the impact direction)",
                       "test_ids": list(test_ids),
                       "reference_mm": ref_mm.round(4).tolist(), "prediction_mm": got_mm.round(4).tolist(),
                       "relative_error": kpi["relative"].round(6).tolist(),
                       "mean_relative": kpi["mean_relative"], "median_relative": kpi["median_relative"],
                       "max_relative": kpi["max_relative"],
                       "mean_relative_ci95": list(map(float, kpi["mean_relative_ci95"]))}, fh, indent=2)
        ex.add_file(os.path.join(a.out, "kpi_peak_displacement.json"))
        # the predictions themselves, so a figure can be redrawn without training again
        predictions_npz = os.path.join(a.out, "predictions.npz")
        np.savez_compressed(predictions_npz, ids=np.array(list(test_ids)),
                            predicted=np.stack(result["predictions"][str(best.name)]).astype(np.float32),
                            reference=np.stack(truths).astype(np.float32), model=str(best.name))
        ex.add_file(predictions_npz)
        for name in ("tabular_leaderboard.md", "field_leaderboard.md"):
            ex.add_file(os.path.join(a.out, name))
        figures = os.path.join(a.out, "figures")
        os.makedirs(figures, exist_ok=True)
        peak_figure(kpi, test_ids, os.path.join(figures, "peak_displacement_parity.png"), str(best.name),
                    a.conditioning)
        surface_figure(full[test_ids[median]], result["predictions"][str(best.name)][median],
                       os.path.join(figures, "displacement_surface.png"), str(best.name), test_ids[median])
        print(f"quantities: {csv}\nfields: {npz}\nfigures: {figures}")


if __name__ == "__main__":
    main()
