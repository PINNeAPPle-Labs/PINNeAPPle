"""Benchmark of field surrogates over many geometries (surface pressure over a car body, displacement over a crash
structure): containers, metrics with bootstrap intervals, the paired significance tests, one trainer for every row,
and the leaderboard.

    from pinneapple_arena.fieldbench import (FieldSample, FieldDataset, normalize_geometry, subsample_nodes,
                                             run_benchmark, build_model_specs, write_report)
"""
from .metrics import (
                      ModelScore,
                      PatchwiseProtocol,
                      bootstrap_ci,
                      error_percentiles,
                      extreme_errors,
                      extreme_of_field,
                      leaderboard,
                      mean_relative_l2,
                      model_cost,
                      paired_bootstrap,
                      pooled_r2,
                      relative_l2_per_geometry,
                      score_model,
)
from .models import OperatorSetAdapter, available_models, build_model_specs
from .protocol import (
                      FieldDataset,
                      FieldSample,
                      normalize_geometry,
                      run_benchmark,
                      subsample_nodes,
                      write_report,
)
from .trainer import collate_samples, predict_field_model, train_field_model

__all__ = ["FieldSample", "FieldDataset", "normalize_geometry", "subsample_nodes", "run_benchmark", "write_report",
           "train_field_model", "predict_field_model", "collate_samples",
           "build_model_specs", "available_models", "OperatorSetAdapter",
           "relative_l2_per_geometry", "mean_relative_l2", "pooled_r2", "error_percentiles",
           "extreme_of_field", "extreme_errors", "bootstrap_ci",
           "paired_bootstrap", "model_cost", "leaderboard", "score_model", "ModelScore", "PatchwiseProtocol"]
