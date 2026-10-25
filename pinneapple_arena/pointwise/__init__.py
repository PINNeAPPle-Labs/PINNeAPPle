"""Pointwise field benchmarks: the problems, reference data, losses, model adapters and training protocol of the
PINNeAPPle Arena app (``apps/pinn_arena``).

An arena problem is a field regression on coordinates: ``(N, in_dim)`` points in, ``(N, out_dim)`` field values
out. For each of the 34 catalog problems (``PROBLEMS_CATALOG``) a reference dataset comes from an analytic or
numerical solution (``DatasetGenerator``), the PDE residual from a PINNeAPPle preset or a native autograd residual
(``PhysicsLossGenerator``), and any registry architecture (``ARCHITECTURES_CATALOG``) is built through
``pp.build_model`` and, if it expects grids or sequences, wrapped so each point is its own sample (``adapters``).
``training`` trains one architecture under the arena protocol (data loss + warmed-up, magnitude-balanced PDE
residual on separate collocation points, cosine LR, early stopping on a validation split, best weights restored)
and scores it on a held-out split; ``leaderboard.aggregate`` ranks architectures over many experiments.

    from pinneapple_arena import pointwise as pw

    gen = pw.DatasetGenerator("burgers_1d", {})
    ref = gen.generate(n_samples=2000)                     # coords (N, 2) = (x, t), solution (N, 1)
    loss_fn = pw.PhysicsLossGenerator("burgers_1d", {}).compile_losses()
    model = pw.instantiate_model("vanilla_pinn", 2, 1)     # ArchitectureUnavailable if the registry can not build it

This differs from ``pinneapple_arena.Arena`` (YAML-driven runs of ``ArenaProblem`` PINN benchmarks with their own
residuals) and from ``pinneapple_arena.regimes`` (equal-budget surrogate rankings per data regime): the problems
here carry UI metadata (equations, parameters, column hints for uploads) and the protocol is the one the app runs.
"""
from __future__ import annotations

from pinneapple_arena.pointwise.adapters import (
    CLOSED_FORM,
    NOT_APPLICABLE,
    POINTWISE_LAYOUTS,
    ArchitectureNotApplicable,
    adapt_model,
    check_applicable,
)
from pinneapple_arena.pointwise.architectures import (
    ARCHITECTURES_CATALOG,
    REGISTRY_KEYS,
    catalog_status,
    registry_key,
)
from pinneapple_arena.pointwise.datasets import DatasetGenerator
from pinneapple_arena.pointwise.geometries import GEOMETRIES_CATALOG, get_geometry_mesh_config
from pinneapple_arena.pointwise.leaderboard import aggregate as aggregate_leaderboard
from pinneapple_arena.pointwise.losses import PhysicsLossGenerator
from pinneapple_arena.pointwise.problems import PROBLEMS_CATALOG
from pinneapple_arena.pointwise.training import (
    ArchitectureUnavailable,
    best_device,
    device_info,
    field_metrics,
    instantiate_model,
    model_forward,
    predict,
    prediction_quality,
    problem_in_dim,
    problem_out_dim,
    rank_models,
    run_inverse_solver,
    train_closed_form_model,
    train_torch_model,
)

__all__ = [
    "ARCHITECTURES_CATALOG", "CLOSED_FORM", "GEOMETRIES_CATALOG", "NOT_APPLICABLE", "POINTWISE_LAYOUTS",
    "PROBLEMS_CATALOG", "REGISTRY_KEYS",
    "ArchitectureNotApplicable", "ArchitectureUnavailable", "DatasetGenerator", "PhysicsLossGenerator",
    "adapt_model", "aggregate_leaderboard", "best_device", "catalog_status", "check_applicable", "device_info",
    "field_metrics", "get_geometry_mesh_config", "instantiate_model", "model_forward", "predict",
    "prediction_quality", "problem_in_dim", "problem_out_dim", "rank_models", "registry_key", "run_inverse_solver",
    "train_closed_form_model", "train_torch_model",
]
