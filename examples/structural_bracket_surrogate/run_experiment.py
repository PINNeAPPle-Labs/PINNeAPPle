"""One-command reconstructed bracket benchmark (CAD→FEM→DeepONet→evaluation).

Run from examples/structural_bracket_surrogate:
    python run_experiment.py --output-dir artifacts/reconstruction

This executes an independent reconstruction, NOT the exact original paper.
Requires CadQuery, Gmsh, NumPy, SciPy and PyTorch. All outputs are tagged
unverified until independent FEM and mesh-convergence gates pass.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from build_benchmark import make_benchmark
from train_prepared import train
from evaluate import evaluate


def run(output_dir: str, *, variants: int = 3, loads: int = 3,
        points: int = 256, mesh_size_mm: float = 5.,
        epochs: int = 100, seed: int = 42):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset = out / "prepared.npz"
    checkpoint = out / "deeponet.pt"
    metrics = out / "metrics.json"
    stages = {}
    try:
        stages["dataset"] = make_benchmark(dataset, variants=variants,
            loads_per_geometry=loads, points_per_case=points,
            mesh_size_mm=mesh_size_mm, seed=seed)
        stages["training"] = train(str(dataset), str(checkpoint), epochs=epochs, seed=seed)
        stages["evaluation"] = evaluate(str(dataset), str(checkpoint), str(metrics))
        status = "completed_unverified"
    except Exception as exc:
        status = "failed"
        stages["failure"] = {"exception_type": type(exc).__name__,
                             "message": str(exc)[:500]}
        raise
    finally:
        manifest = {"workflow": "bracket_reconstructed_v1", "status": status,
                    "seed": seed, "mesh_size_mm": mesh_size_mm,
                    "epochs": epochs, "stages": stages,
                    "exact_source_reproduction": False,
                    "veriphysics_certified": False,
                    "independent_fem_validated": False,
                    "mesh_convergence_validated": False}
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="artifacts/reconstruction")
    p.add_argument("--variants", type=int, default=3)
    p.add_argument("--loads", type=int, default=3)
    p.add_argument("--points", type=int, default=256)
    p.add_argument("--mesh-size-mm", type=float, default=5.)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    print(json.dumps(run(a.output_dir, variants=a.variants, loads=a.loads,
          points=a.points, mesh_size_mm=a.mesh_size_mm,
          epochs=a.epochs, seed=a.seed), indent=2))
