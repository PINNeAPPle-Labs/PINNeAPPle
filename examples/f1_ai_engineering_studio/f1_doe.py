"""Reproducible F1 concept design-of-experiments (not CFD or validated lap times)."""
from __future__ import annotations
import argparse
import csv
import json
import random
from pathlib import Path

PARAMETERS = {
    "front_wing_angle_deg": (5.0, 18.0),
    "rear_wing_angle_deg": (8.0, 25.0),
    "ride_height_mm": (25.0, 65.0),
    "diffuser_angle_deg": (5.0, 14.0),
}

def generate_variants(count: int = 10, seed: int = 42) -> list[dict]:
    """Latin hypercube DOE with independently shuffled strata per parameter."""
    if count < 1:
        raise ValueError("count must be >= 1")
    rng = random.Random(seed)
    columns = {}
    for name, (low, high) in PARAMETERS.items():
        strata = list(range(count))
        rng.shuffle(strata)
        columns[name] = [
            round(low + (high - low) * (strata[i] + rng.random()) / count, 6)
            for i in range(count)
        ]
    return [{"variant_id": f"f1-{i + 1:03d}",
             **{name: columns[name][i] for name in PARAMETERS}}
            for i in range(count)]

def export_variants(directory: str | Path, count: int = 10, seed: int = 42) -> Path:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    variants = generate_variants(count, seed)
    with (target / "variants.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant_id", *PARAMETERS])
        writer.writeheader()
        writer.writerows(variants)
    (target / "manifest.json").write_text(json.dumps({
        "schema_version": "0.1.0", "kind": "conceptual_doe",
        "seed": seed, "count": count, "parameters": PARAMETERS,
        "simulation_status": "not_run",
        "verification_status": "not_run",
        "warning": "Geometry, CFD, lap time and physical validity are NOT established."
    }, indent=2) + "\n", encoding="utf-8")
    return target

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/f1_doe")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(export_variants(args.output, args.count, args.seed))
