"""An AirfRANS submission to FluidsBench, from the public release to a validated result package.

The benchmark (fluidsbench.org) scores learned surrogates on realistic CFD with a fixed, benchmark-owned scoring
support, published metric equations and a composite with caps. This pipeline owns the participant's half: read the
release with the library's own feature definition, fit a coordinate network, predict at the native points of every
official test case, and run the benchmark's own validate → evaluate → package → verify chain — then check the resulting
package twice, with :mod:`pinneapple_arena.fluidsbench` and with an independent recomputation of the domain metrics.

    from pinneapple_simulation.pipelines.airfrans_benchmark import CampaignSpec, TrainingSpec, run_campaign

    spec = CampaignSpec(dataset_root="…/Dataset", support_root="…/support/airfrans", checkout="…/fluidsbench-submission")
    result = run_campaign(spec, "work/airfrans-full", submission_id="my-model-full-v1", split_id="full")

Modules: :mod:`.data` (the release, case by case), :mod:`.features` (the eleven input channels), :mod:`.model` (the
network and its physics residuals), :mod:`.train`, :mod:`.predict`, :mod:`.entry` (the participant entry and the schema-v3
package configuration) and :mod:`.submit` (the benchmark's workflow, driven and then checked).

Two things this pipeline deliberately does not do: it does not compute the leaderboard numbers itself — the benchmark's
evaluator does, and the campaign only re-checks them — and it does not claim a split's result is an extrapolation test
unless the model was trained in that split's own regime, which the methodology record states.

Everything except the extraction stage runs without the AirfRANS/VTK stack, on a feature store written once.
"""
from __future__ import annotations

from .campaign import STAGES, CampaignSpec, campaign_from_dict, run_campaign
from .data import AirfRANSStackMissing, case_arrays, case_ids_for, extract_cases, manifest_splits
from .entry import SUPPORTS, write_entry
from .features import FEATURE_NAMES, chord, point_features, sample_indices, scaled_targets
from .model import CoordinateNet, WallAwareNet, physics_loss, wall_layer_profile
from .predict import PredictionSpec, predict_cases
from .submit import OfficialToolchain, Step, cross_check_domain_metrics
from .train import (TrainingSpec, WallSpec, curve_force_loss, load_curve_batch, load_training_points,
                    load_wall_points, pick_device, train, train_wall)
from .wall import (WallTruthMissing, curve_coefficients, curve_features, load_wall_case, slope_from_wall_shear,
                   surface_rank, wall_case_truth, write_wall_store)

__all__ = ["CampaignSpec", "campaign_from_dict", "run_campaign", "STAGES",
           "TrainingSpec", "WallSpec", "train", "train_wall", "curve_force_loss",
           "load_training_points", "load_wall_points", "load_curve_batch", "pick_device",
           "PredictionSpec", "predict_cases",
           "CoordinateNet", "WallAwareNet", "wall_layer_profile", "physics_loss", "FEATURE_NAMES", "point_features", "scaled_targets", "sample_indices",
           "chord",
           "case_arrays", "case_ids_for", "extract_cases", "manifest_splits", "AirfRANSStackMissing",
           "write_entry", "SUPPORTS",
           "OfficialToolchain", "Step", "cross_check_domain_metrics",
           "WallTruthMissing", "wall_case_truth", "write_wall_store", "load_wall_case", "slope_from_wall_shear",
           "curve_coefficients", "curve_features", "surface_rank"]
