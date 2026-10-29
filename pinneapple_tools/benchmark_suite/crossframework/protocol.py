"""Fixed settings of the cross-framework benchmark protocol and the export every runner produces.

The numbers here are the protocol (``benchmarks/vs_frameworks/PROTOCOL.md``); a runner reads them instead of
choosing its own, so the identical-configuration track cannot drift between frameworks.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from .problems import (
    PINNProblem,
    make_point_set,
    residual_points,
    save_point_set,
    stencil,
    test_points,
)

__all__ = ["A1", "A2", "C", "B", "export_predictions", "ensure_point_set"]

#: Track A1, identical configuration
A1 = {
    "width": 50, "depth": 4, "activation": "tanh", "init": "xavier_uniform (shared arrays per seed), zero bias",
    "dtype": "float32", "optimizer": "adam", "lr": 1e-3, "n_interior": 10_000, "n_boundary": 400,
    "loss_weights": 1.0, "loss": "mean squared PDE residual + mean squared BC error (+ mean squared IC error)",
    "batch": "full batch, the same points every iteration", "grad_clip": None, "lr_schedule": "constant",
    "point_seed": 20260,
}

#: Track A2, each framework's own recipe, the same time cap per run (seconds of process CPU time, one thread)
A2 = {"cap_s": 240.0, "cap_kind": "cpu"}

#: Track C, inverse Burgers
C = {"n_obs": 2000, "noise_rel": 0.01, "obs_seed": 4242, "nu_init": 0.05, "width": 50, "depth": 4, "lr": 1e-3,
     "n_interior": 10_000}

#: Track B, operator learning on Darcy
B = {"resolution": 64, "width": 32, "modes": 12, "layers": 4, "batch_size": 20, "lr": 1e-3, "weight_decay": 1e-4,
     "loss": "per-sample relative L2, averaged over the batch", "schedule": "cosine over all steps"}


def export_predictions(problem: PINNProblem, predict: Callable[[np.ndarray], np.ndarray],
                       batch: int = 20_000) -> dict[str, np.ndarray]:
    """``{"u_test", "u_stencil"}`` from ``predict`` ((N, 2) float32 array -> (N,) or (N, 1)), evaluated in batches
    on the shared test grid and on the residual stencil. Every runner exports exactly this."""
    def run(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        parts = [np.asarray(predict(X[i:i + batch]), dtype=np.float64).reshape(-1) for i in range(0, len(X), batch)]
        return np.concatenate(parts)
    return {"u_test": run(test_points(problem)),
            "u_stencil": run(stencil(problem, residual_points(problem)).reshape(-1, 2))}


def ensure_point_set(problem: PINNProblem, root) -> Path:
    """The A1 point set of ``problem`` under ``root/points/`` (created once; later calls load the same file)."""
    path = Path(root) / "points" / f"{problem.name}.npz"
    if not path.exists():
        save_point_set(make_point_set(problem, n_interior=A1["n_interior"], n_boundary=A1["n_boundary"],
                                      seed=A1["point_seed"]), path)
    return path


def runner_parser(description: str):
    """The command line every runner accepts (the driver calls runners only through it)."""
    import argparse
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--problem", required=True)
    p.add_argument("--mode", default="a1", help="a1 (identical configuration) or a2 (the framework's own recipe)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=0, help="A1 / C: number of optimizer iterations")
    p.add_argument("--cap-s", type=float, default=A2["cap_s"], help="A2: time cap per run [s]")
    p.add_argument("--cap-kind", default=A2["cap_kind"], choices=("cpu", "wall"))
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--root", required=True, help="benchmark root (shared point sets live in ROOT/points)")
    p.add_argument("--out", required=True, help="directory for <run_id>.json / .npz")
    p.add_argument("--run-id", required=True)
    p.add_argument("--epochs", type=int, default=0, help="B: epochs")
    p.add_argument("--n-train", type=int, default=1000, help="B: training samples")
    return p


def base_meta(args, *, track: str, framework: str, source: str, config: dict) -> dict:
    """Metadata every result starts from (the runner fills status, iterations and times)."""
    from .records import FRAMEWORK_PACKAGES, package_versions
    return {"track": track, "problem": args.problem, "framework": framework, "mode": args.mode, "seed": args.seed,
            "threads": args.threads, "device": args.device, "status": "running", "iterations": 0,
            "wall_s": float("nan"), "cpu_s": float("nan"), "loadavg_before": [], "loadavg_after": [],
            "versions": package_versions(FRAMEWORK_PACKAGES.get(framework, []) + ["torch"]),
            "config": config, "source": source}


__all__ += ["runner_parser", "base_meta"]
