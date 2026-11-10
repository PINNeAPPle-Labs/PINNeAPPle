"""Generate a publication-ready *internal* study report from study.json.

This does not run training, alter scientific decisions or imply independent review.
Usage: python examples/fno_coldplate/report_study.py artifacts/external_validation/coldplate_multiseed
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def generate_report(directory: str | Path) -> dict:
    directory = Path(directory)
    study = json.loads((directory / "study.json").read_text())
    runs = study["runs"]
    if not runs:
        raise ValueError("No completed runs in study.json")
    columns = ("seed", "train_steps", "decision", "relative_l2", "hotspot_error_K",
               "infer_ms_per_layout", "solver_ms_per_layout",
               "data_generation_s", "training_wall_s",
               "mean_baseline_relative_l2", "nearest_baseline_relative_l2",
               "ood_relative_l2", "ood_decision")
    rows = []
    for r in runs:
        row = {k: r.get(k) for k in columns}
        row["ood_relative_l2"] = r.get("ood", {}).get("relative_l2")
        row["ood_decision"] = r.get("ood", {}).get("decision")
        rows.append(row)
    with (directory / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# PINNeAPPle cold-plate FNO: multi-seed evaluation",
        "",
        f"**Status:** {study['status']}",
        "",
        "> Internal measurements only. Not independent validation or peer review.",
        "> A physics REJECT must never be converted to an approval based on L2 alone.",
        "",
        "## Protocol",
        f"- Model seeds: {study['seeds']}",
        f"- Padding: {study['padding']}",
        f"- Smoke mode: {study['smoke']}",
        "- Held-out GRF layouts are independent of training draws.",
        "- Nearest-layout and training-mean baselines use training data only.",
        "- Timings include separate dataset generation and training wall times.",
        "- No end-to-end speedup is claimed; training and generation must be amortized.",
        "",
        "## Per-seed measurements",
        "",
        "| Seed | L2 relative | Mean baseline | Nearest baseline | Hotspot error (K) | Physics decision | OOD decision |",
        "|---:|---:|---:|---:|---:|---|---|",
    ]
    def fmt(x):
        return "n/a" if x is None else f"{x:.5g}" if isinstance(x, (int, float)) else str(x)
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(k)) for k in (
            "seed", "relative_l2", "mean_baseline_relative_l2",
            "nearest_baseline_relative_l2", "hotspot_error_K",
            "decision", "ood_decision")) + " |")
    lines += ["", "## Descriptive statistics", ""]
    for key, stats in study["summary_stats"].items():
        lines.append(f"- {key}: median={stats['median']:.6g}, min={stats['min']:.6g}, max={stats['max']:.6g}")
    lines += [
        "", "## Limitations",
        "- Five seeds do not establish a confidence interval or independent adoption.",
        "- Finite-volume solver timings exclude startup; environment and hardware affect comparisons.",
        "- OOD rejection is a guardrail, not proof of OOD prediction accuracy.",
        "- A physics decision of REJECT remains a negative result.",
        "- External reproduction requires another evaluator's exact run logs and commit.",
    ]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [r["seed"] for r in rows]
        fig, ax = plt.subplots(figsize=(8, 4))
        for key, label in (
            ("relative_l2", "FNO"),
            ("mean_baseline_relative_l2", "Train mean"),
            ("nearest_baseline_relative_l2", "Nearest layout"),
        ):
            ys = [r.get(key) for r in rows]
            if all(y is not None for y in ys):
                ax.plot(xs, ys, marker="o", label=label)
        ax.set(xlabel="Model seed", ylabel="Held-out relative L2",
               title="Cold-plate predictive error: model vs baselines")
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / "heldout_error.png", dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([str(x) for x in xs], [r["hotspot_error_K"] for r in rows])
        ax.set(xlabel="Model seed", ylabel="Maximum hotspot error (K)",
               title="Cold-plate hotspot error by seed")
        fig.tight_layout()
        fig.savefig(directory / "hotspot_error.png", dpi=160)
        plt.close(fig)
    except ImportError:
        # CSV and Markdown reports remain available without plotting extras.
        pass
    return {"runs": len(rows), "report": str(directory / "REPORT.md")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    print(json.dumps(generate_report(args.directory), indent=2))
