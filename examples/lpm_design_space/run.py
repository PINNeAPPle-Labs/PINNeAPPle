"""One model for a whole design space: a Large Physics Model of potential flow past an ellipse family.

The design space is ``a, b`` in [0.6, 1.4] with a uniform stream along x. The answer is analytic
(Milne-Thomson, *Theoretical Hydrodynamics*, 5th ed., 1968, §9.61), so every number below is measured
against the exact Cp — no solver, no reference data set:

    q/U = (a + b) |sin t| / sqrt(a^2 sin^2 t + b^2 cos^2 t),   Cp = 1 - (q/U)^2

The script trains one model on ``--train`` designs, and reports, on designs it never saw:
the held-out relative L2, the ensemble spread, and the out-of-distribution flag on three probes
that are not in the family (a square, a circle of radius 3, a thin ellipse a = 0.25).

    python examples/lpm_design_space/run.py                                  # 200 designs, ~3 min on a CPU
    python examples/lpm_design_space/run.py --train 1000 --epochs 400        # the scale run (benchmarks/lpm_scale.py)
"""
from __future__ import annotations

import argparse
import os
import time

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneapple_lab import experiment
from pinneapple_neural.lpm import Design, LargePhysicsModel, ellipse_design_space, fit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=int, default=200)
    ap.add_argument("--test", type=int, default=20)
    ap.add_argument("--points", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--backbone", choices=("mlp", "attention"), default="attention")
    a = ap.parse_args()

    torch.manual_seed(0)
    train = ellipse_design_space(a.train, n_points=a.points, seed=0)
    held = ellipse_design_space(a.test, n_points=a.points, seed=99_999)
    model = LargePhysicsModel(dim=2, n_fields=1, radii=(0.15, 0.5, 1.5), width=96, n_heads=4,
                              backbone=a.backbone, attn_layers=2, attn_heads=4)

    t0 = time.time()
    hist = fit(model, train, epochs=a.epochs, lr=2e-3, batch_designs=a.batch)
    seconds = time.time() - t0
    model.eval()

    def rel(d: Design) -> float:
        return float((model.predict(d)["mean"] - d.target).norm() / d.target.norm())

    errors = [rel(d) for d in held]
    held_mean = sum(errors) / len(errors)
    spread = [float(model.predict(d)["std"].mean()) for d in held]      # epistemic spread, Cp units
    probes = {"square": model.ood(Design.square(n=a.points)),
              "circle_r3": model.ood(Design.ellipse(3.0, 3.0, a.points)),
              "thin_ellipse_a025": model.ood(Design.ellipse(0.25, 1.2, a.points))}
    train_mean = sum(rel(d) for d in train) / len(train)
    # baseline: predict, at every surface point, the average Cp of the training designs
    mean_field = torch.stack([d.target for d in train]).mean(0)
    baseline = sum(float((mean_field - d.target).norm() / d.target.norm()) for d in held) / len(held)
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "results"), exist_ok=True)

    with experiment("lpm_design_space", title="One model for a whole design space (Large Physics Model)",
                    description=("A single trained model covers the ellipse family of potential flow: held-out Cp on "
                                 "unseen geometries, an ensemble spread per prediction and an out-of-distribution "
                                 "flag on geometries outside the family, all against the exact Milne-Thomson answer."),
                    domain="fluid mechanics (potential flow)",
                    methods=["Large Physics Model (geometry code, multi-scale neighbourhoods, ensemble heads, OOD)",
                             "query-to-surface cross-attention" if a.backbone == "attention" else "per-query MLP",
                             "batched design-space training"],
                    modules=["pinneapple_neural.lpm"],
                    params={"train_designs": a.train, "held_out_designs": a.test, "surface_points": a.points,
                            "epochs": a.epochs, "batch_designs": a.batch, "backbone": a.backbone},
                    references=["Milne-Thomson, Theoretical Hydrodynamics, 5th ed. (1968), §9.61: surface speed of "
                                "potential flow past an ellipse",
                                "Wu et al., Transolver, ICML 2024, arXiv:2402.02366 (geometry-aware attention "
                                "over the surface points)"],
                    links={"benchmark": "benchmarks/lpm_scale.py", "guide": "benchmarks/lpm_ellipse_design_space.py"},
                    register=False, curate=False) as ex:
        ex.log_metric("train_rel_l2_mean", train_mean, unit="1", how="relative L2 of Cp over the training designs")
        ex.log_metric("heldout_rel_l2_mean", held_mean, unit="1",
                      how="relative L2 of Cp on designs drawn with a different seed, exact analytic reference")
        ex.log_metric("heldout_rel_l2_p90", sorted(errors)[int(0.9 * (len(errors) - 1))], unit="1",
                      how="90th percentile over the held-out designs")
        ex.log_metric("ensemble_spread_mean", sum(spread) / len(spread), unit="Cp",
                      how="mean of the ensemble std over the held-out predictions")
        ex.log_metric("constant_field_baseline_rel_l2", baseline, unit="1",
                      how="relative L2 of predicting the training-average Cp at every point, same held-out designs")
        ex.log_metric("train_seconds", seconds, unit="s", how="wall time of fit() on this machine")
        ex.log_metric("seconds_per_epoch", seconds / a.epochs, unit="s", how="wall time divided by the epochs run")
        ex.log_metric("epochs_to_final_loss", float(hist[-1]), unit="1", how="MSE of the bagged training loss")

        ex.check("heldout_matches_the_exact_answer", value=held_mean, max=0.10, kind="reference",
                 label="held-out Cp within 10% of the Milne-Thomson potential-flow answer")
        ex.check("beats_the_constant_field", passed=held_mean < 0.8 * baseline, kind="reference",
                 label=f"held-out error {held_mean:.3f} against {baseline:.3f} for the constant-field baseline")
        for name, report in probes.items():
            ex.check(f"ood_{name}", passed=bool(report["ood"]), kind="reference",
                     label=f"{name} flagged as outside the trained family (score {report['ood_score']:.2f})")

        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        worst = max(range(len(held)), key=lambda i: errors[i])
        d = held[worst]
        order = torch.argsort(torch.atan2(d.points[:, 1], d.points[:, 0]))
        ax.plot(torch.atan2(d.points[order, 1], d.points[order, 0]).numpy(), d.target[order, 0].numpy(),
                label="exact Cp (Milne-Thomson §9.61)", lw=2)
        ax.plot(torch.atan2(d.points[order, 1], d.points[order, 0]).numpy(),
                model.predict(d)["mean"][order, 0].numpy(), "--", label="LPM prediction", lw=2)
        ax.set_xlabel("surface angle t (rad)")
        ax.set_ylabel("Cp")
        ax.set_title(f"Worst held-out design (a={d.points[:, 0].max():.2f}, b={d.points[:, 1].max():.2f}), "
                     f"relative L2 = {errors[worst]:.3f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(here, "results", "lpm_worst_heldout_cp.png"), dpi=150)

    print(f"held-out relative L2 {held_mean:.4f} | train {train_mean:.4f} | {seconds:.1f} s | "
          f"OOD scores {[round(p['ood_score'], 2) for p in probes.values()]}")


if __name__ == "__main__":
    main()
