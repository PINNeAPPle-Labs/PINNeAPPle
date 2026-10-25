"""Four contrasting PINN architectures trained on the same viscous Burgers problem, ranked by accuracy and by
Arena's physics-aware view.

Problem: ``burgers_1d`` (u_t + u u_x = nu u_xx, u(x, 0) = -sin(pi x) on [-1, 1], nu = 0.01/pi; Raissi, Perdikaris &
Karniadakis 2019, J. Comput. Phys. 378), reference by the Cole-Hopf solution inside the problem class. A steep front
forms near x = 0 around t ~ 0.3 and persists to t = 1.

The run uses Arena's own pieces (:func:`~pinneapple_arena.model_factory.build_model`,
:func:`~pinneapple_arena.trainer.train_pinn`, :func:`~pinneapple_arena.trainer.evaluate_model`,
:func:`~pinneapple_arena.arena.rank_by_accuracy`, :func:`~pinneapple_arena.arena.physics_aware_rank`), one model at
a time so a caller can report progress. ``physics_aware_rank`` never reorders the accuracy ranking; it flags models
whose PDE residual is an outlier of the group (median + n_std * std) and, when ``residual_threshold`` is given,
above that absolute value too. The relative test alone can miss models that are all bad together: with four
models, a model with a residual far above the others raises the group std with it (see the app README for a
measured case).

Per-architecture budgets (``DEFAULT_ARCHITECTURES``) were tuned in the source project for convergence on the front:
Vanilla PINN 4x64 for 12,000 epochs, SIREN 3x64 with omega_0 = 15 for 4,000, PIELM (frozen random hidden layer,
trained output layer) 128 features for 5,000, XTFC (constrained basis, trained output layer) for 400. A full run
takes minutes on a CPU; tests use a few epochs per model.

Architectures left out on purpose: ``vpinn`` builds the same network as ``vanilla_pinn`` and its weak-form loss is
not used by ``train_pinn``, so it would run as a copy of Vanilla PINN; ``hash_grid_mlp`` and ``bench_fourier_mlp``
collapsed towards the trivial u = 0 (low residual, ~90 % error) in the source project's tuning runs.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from .arena import physics_aware_rank, rank_by_accuracy
from .config import ModelConfig, NetworkConfig, TrainingConfig
from .model_factory import build_model
from .problems import get_problem
from .trainer import TrainResult, evaluate_model, train_pinn

__all__ = ["PROBLEM_NAME", "PROBLEM_PARAMS", "DEFAULT_ARCHITECTURES", "ARCHITECTURE_ORDER", "SHOCK_REGION_X",
           "N_STD", "field_fit_slice", "run_benchmark"]

PROBLEM_NAME = "burgers_1d"
PROBLEM_PARAMS: dict[str, float] = {"nu": 0.01 / math.pi}
GRID_N = 64           # evaluation grid (x, t), GRID_N x GRID_N
N_BC = 400            # boundary / initial condition points
N_TRAIN_SUPERVISED = 200   # only sizes the interior sample that fixes the collocation box
N_COL = 4000          # interior collocation points, resampled every run
GRAD_CLIP = 1.0
SCHEDULER = "cosine"
SHOCK_REGION_X = (-0.3, 0.3)   # band of the steepest gradients of the reference, every t
N_STD = 2.0                    # Arena's default outlier rule

DEFAULT_ARCHITECTURES: list[dict[str, Any]] = [
    {"display": "Vanilla PINN", "type": "vanilla_pinn", "epochs": 12000, "lr": 1.5e-3,
     "network": {"hidden": [64, 64, 64, 64]}},
    {"display": "SIREN", "type": "siren", "epochs": 4000, "lr": 1e-3,
     "network": {"hidden": [64, 64, 64], "omega_0": 15.0}},
    {"display": "PIELM", "type": "pielm", "epochs": 5000, "lr": 3e-3, "network": {"hidden_dim": 128}},
    {"display": "XTFC", "type": "xtfc", "epochs": 400, "lr": 2e-3, "network": {"hidden": [32, 32, 32]}},
]
ARCHITECTURE_ORDER: list[str] = [a["display"] for a in DEFAULT_ARCHITECTURES]


def field_fit_slice(xy_eval: np.ndarray, y_ref: np.ndarray, y_pred: np.ndarray, grid_n: int) -> dict[str, Any]:
    """Reference and prediction on the last time row of the evaluation grid (t = 1 for ``burgers_1d``).

    Arena's 1-D time problems build the grid with ``np.meshgrid(x, t)`` ('xy' indexing) and ravel it, so t is
    constant over each block of ``grid_n`` rows and the last block is the final time, x increasing."""
    a, b = (grid_n - 1) * grid_n, grid_n * grid_n
    ref = y_ref[a:b, 0] if y_ref.ndim > 1 else y_ref[a:b]
    pred = y_pred[a:b, 0] if y_pred.ndim > 1 else y_pred[a:b]
    return {"t": float(np.mean(xy_eval[a:b, 1])), "x": xy_eval[a:b, 0].tolist(), "u_real": ref.tolist(),
            "u_pred": pred.tolist()}


def _rel_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    from pinneapple_physics.metrics import relative_l2

    return float(relative_l2(pred, ref))


def run_benchmark(problem_name: str = PROBLEM_NAME, architectures: list[dict[str, Any]] | None = None,
                  seed: int | None = None, progress_cb: Callable[[int, int], None] | None = None,
                  n_col: int = N_COL, grid_n: int = GRID_N, residual_threshold: float | None = None
                  ) -> dict[str, Any]:
    """Train every architecture on the same collocation points, evaluate on the same grid, rank.

    ``seed`` makes the run reproducible (collocation sample and each model's initialisation, ``seed + i``);
    ``None`` is a fresh run. ``progress_cb(i, n)`` is called after each model. Returns a JSON-able dict: per-model
    metrics (relative L2, L-infinity, relative L2 in the front band, PDE residual, times, parameters, the t = 1
    field fit), the accuracy ranking, the physics-aware view and whether the ranking order was kept."""
    architectures = DEFAULT_ARCHITECTURES if architectures is None else architectures
    problem = get_problem(problem_name)
    xy_int, _, xy_bc, y_bc, xy_eval, y_eval, field_names = problem.supervised_data(
        n_train=N_TRAIN_SUPERVISED, n_bc=N_BC, grid_n=grid_n, **PROBLEM_PARAMS)
    in_dim = xy_int.shape[1]
    out_dim = y_eval.shape[1] if y_eval.ndim > 1 else 1
    rng = np.random.default_rng(seed)
    xy_col = rng.uniform(xy_int.min(axis=0), xy_int.max(axis=0), (n_col, in_dim))
    band = (xy_eval[:, 0] >= SHOCK_REGION_X[0]) & (xy_eval[:, 0] <= SHOCK_REGION_X[1])

    train_results: list[TrainResult] = []
    eval_results: list[dict[str, Any]] = []
    per_model: list[dict[str, Any]] = []
    for i, arch in enumerate(architectures):
        model_seed = None if seed is None else seed + i
        cfg = ModelConfig(name=arch["display"], type=arch["type"], network=NetworkConfig(**arch.get("network", {})),
                          training=TrainingConfig(epochs=arch["epochs"], lr=arch.get("lr", 2e-3),
                                                  grad_clip=GRAD_CLIP, scheduler=SCHEDULER, seed=model_seed))
        if model_seed is not None:
            torch.manual_seed(model_seed)          # build_model initialises the weights before train_pinn seeds
        model = build_model(cfg, in_dim=in_dim, out_dim=out_dim)
        tres = train_pinn(model, cfg, pinn_residuals_fn=problem.pinn_residuals, xy_int=xy_col, xy_bc=xy_bc,
                          uv_bc=y_bc, problem_params=PROBLEM_PARAMS, log_interval=10**9)
        t_inf = time.time()
        eres = evaluate_model(tres, cfg, xy_eval=xy_eval, Y_ref=y_eval, field_names=field_names)
        inference_s = time.time() - t_inf
        eres["name"] = arch["display"]
        train_results.append(tres)
        eval_results.append(eres)
        mets = eres["metrics"]
        pred = eres["pred"][:, 0] if eres["pred"].ndim > 1 else eres["pred"]
        ref = eres["ref"][:, 0] if eres["ref"].ndim > 1 else eres["ref"]
        per_model.append({
            "name": arch["display"], "type": arch["type"], "class": type(model).__name__,
            "n_params": int(sum(p.numel() for p in model.parameters())), "epochs": arch["epochs"],
            "lr": arch.get("lr", 2e-3), "train_time_s": tres.train_time, "inference_time_s": inference_s,
            "rel_l2_error": float(np.mean([mets[f"rel_{f}"] for f in field_names if f"rel_{f}" in mets])),
            "max_abs_error": float(np.mean([mets[f"Linf_{f}"] for f in field_names if f"Linf_{f}" in mets])),
            "shock_region_rel_error": _rel_l2(pred[band], ref[band]),
            "physics_residual": tres.physics_residual,
            "field_fit": field_fit_slice(xy_eval, y_eval, eres["pred"], grid_n),
        })
        if progress_cb is not None:
            progress_cb(i + 1, len(architectures))

    accuracy = rank_by_accuracy(train_results, eval_results, field_names)
    pa = physics_aware_rank(train_results, eval_results, field_names, n_std=N_STD,
                            residual_threshold=residual_threshold)
    flagged = set(pa["flagged_high_residual"])
    med, std = pa["group_median"], pa["group_std"]
    return {
        "problem": problem_name, "problem_description": problem.description,
        "problem_equation": "u_t + u u_x = nu u_xx", "nu": PROBLEM_PARAMS["nu"],
        "shock_region_x": list(SHOCK_REGION_X), "field_names": field_names, "n_col": n_col, "n_bc": N_BC,
        "grid_n": grid_n, "seed": seed, "models": per_model,
        "accuracy_ranking": [{"rank": i + 1, "name": n, "mean_rel_error": e, "flagged": n in flagged}
                             for i, (n, e) in enumerate(accuracy)],
        "physics_aware": {"physics_residuals": pa["physics_residuals"],
                          "flagged_high_residual": pa["flagged_high_residual"], "group_median": med,
                          "group_std": std, "n_std": N_STD, "residual_threshold": residual_threshold,
                          "threshold": med + N_STD * std if med is not None and std is not None else None},
        "ranking_order_unchanged": [n for n, _ in accuracy] == [n for n, _ in pa["ranking"]],
    }
