"""Multi-seed held-out cold-plate experiment using PINNeAPPle's existing FNO.

Full runs are manual and compute-intensive. --smoke is an integration check
only and must not be cited for accuracy or physical approval.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
import numpy as np

from run import main as train_existing_fno


def run_study(out: str | Path, seeds=(0, 1, 2, 3, 4), *, smoke=False,
              steps: int | None = None, padding=8) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Require unique nonempty list of seeds")
    rows = []
    for seed in seeds:
        seed = int(seed)
        train_seed, test_seed = 1000 + 10 * seed, 1001 + 10 * seed
        summary = train_existing_fno(
            smoke=smoke, base=str(out / f"seed_{seed}"),
            steps=steps, padding=padding, train_seed=train_seed,
            test_seed=test_seed, model_seed=seed,
        )
        rows.append({
            "seed": seed, "train_seed": train_seed, "test_seed": test_seed,
            "train_steps": summary["train_steps"], "decision": summary["decision"],
            "relative_l2": summary["test_rel_l2"],
            "hotspot_error_K": summary["max_hotspot_error_K"],
            "infer_ms_per_layout": summary["infer_ms_per_layout"],
            "solver_ms_per_layout": summary["solver_ms_per_layout"],
            "checks": summary["checks"], "ood": summary["ood"],
            "data_generation_s": summary.get("data_generation_s"),
            "training_wall_s": summary.get("training_wall_s"),
            "mean_baseline_relative_l2": summary.get("baselines", {}).get("training_mean_relative_l2"),
            "nearest_baseline_relative_l2": summary.get("baselines", {}).get("nearest_layout_relative_l2"),
            "provenance": summary.get("provenance"),
        })
    fields = ("relative_l2", "hotspot_error_K", "infer_ms_per_layout", "solver_ms_per_layout")
    stats = {}
    for key in fields:
        a = np.asarray([r[key] for r in rows], dtype=float)
        stats[key] = {"median": float(np.median(a)),
                      "min": float(a.min()), "max": float(a.max())}
    report = {
        "schema": "pinneapple.external-fno-study/1",
        "status": "SMOKE_ONLY" if smoke else "COMPLETED_NOT_INDEPENDENTLY_REPRODUCED",
        "qualification": "Internal multi-seed evaluation; not peer review or external adoption.",
        "notes": [
            "Different model initialization and train/test GRF seeds on every replicate.",
            "No extrapolation generalization claim: held-out GRF draws use the same distribution.",
            "Solver timings exclude dataset generation/training; not an end-to-end speedup.",
            "5 seeds are descriptive; median[min,max] is not a confidence interval.",
            "Physics rejection is a valid negative result; thresholds must stay fixed.",
        ],
        "seeds": list(seeds), "padding": padding, "smoke": smoke,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "summary_stats": stats, "runs": rows,
    }
    (out / "study.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    from report_study import generate_report
    generate_report(out)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/external_validation/coldplate_multiseed")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    args = parser.parse_args()
    print(json.dumps(run_study(args.out, args.seeds, smoke=args.smoke,
                              steps=args.steps, padding=args.padding)["summary_stats"], indent=2))
