"""Writing the participant entry: the per-case discretization records and the schema-v3 package configuration.

Everything here is measured or read — the case list and entity counts come from the approved support release, the
parameter count, epochs, seed, checkpoint digest and campaign times come from the run's own records. Nothing about the
method is invented: the loss, the sampling and the compute are the ones the training stage actually used.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["SUPPORTS", "write_entry"]

#: the three supports the AirfRANS release scores on
SUPPORTS = {"domain": "two-dimensional-domain-native", "curve": "airfoil-curve-native",
            "forces": "airfoil-force-coefficients-native"}

CASE_SET_FOR = {"full": "standard", "scarce": "standard", "aoa_extrapolation": "aoa_extrapolation"}


def _per_case(values: Sequence[int]) -> dict[str, Any]:
    ordered = sorted(values)
    return {"kind": "per_case", "minimum": int(ordered[0]), "median": float(np.median(ordered)),
            "maximum": int(ordered[-1])}


def _split_order(split_file: str | None, cases: Mapping[str, Any]) -> list[str]:
    if split_file and os.path.exists(split_file):
        with open(split_file) as handle:
            ordered = [str(c) for c in json.load(handle)["case_ids"]]
        missing = [c for c in ordered if c not in cases]
        if missing:
            raise ValueError(f"the split lists cases the support release does not carry: {missing[:3]}")
        return ordered
    return sorted(cases)


def write_entry(entry_dir: str, *, submission_id: str, series_id: str, split_id: str, support_root: str,
                training: Mapping[str, Any], prediction: Mapping[str, Any], training_hours: float,
                model_name: str = "PINNeAPPle coordinate network",
                model_type: str = "Coordinate network (point-wise MLP)", hardware: str = "",
                physics_weight: float = 0.0, training_note: str = "", submitted_at: str = "",
                split_file: str | None = None, checkpoint_path: str | None = None,
                submitter: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Write ``entry.json``, ``discretization-cases.jsonl`` and ``package-config.json``; returns their paths."""
    case_set = CASE_SET_FOR.get(split_id, "standard")
    chunk_path = os.path.join(support_root, "case-sets", case_set, "chunk-000.json")
    with open(chunk_path) as handle:
        chunk = json.load(handle)
    cases = {case["case_id"]: {i["support_id"]: int(i["entity_count"]) for i in case["support_instances"]}
             for case in chunk["cases"]}
    domain_counts = _per_case([cases[c][SUPPORTS["domain"]] for c in cases])
    curve_counts = _per_case([cases[c][SUPPORTS["curve"]] for c in cases])
    os.makedirs(entry_dir, exist_ok=True)

    with open(os.path.join(entry_dir, "entry.json"), "w") as handle:
        json.dump({"schema": "fluidsbench-airfrans-entry-v1", "submission_id": submission_id,
                   "split_id": split_id, "force_prediction_source": "field_integrated",
                   "submitted_at": submitted_at}, handle, indent=1)

    ordered = _split_order(split_file, cases)
    with open(os.path.join(entry_dir, "discretization-cases.jsonl"), "w") as handle:
        for case_id in ordered:
            counts = cases[case_id]
            domain, curve = counts[SUPPORTS["domain"]], counts[SUPPORTS["curve"]]
            handle.write(json.dumps({
                "$schema": "https://fluidsbench.org/schemas/v3/discretization-case.schema.json",
                "schema_version": "1.0", "submission_id": submission_id, "dataset_id": "airfrans",
                "split_id": split_id, "case_id": case_id,
                "inference": {
                    "inputs": [
                        {"id": "surface-geometry-input",
                         "entity_counts": [{"entity": "airfoil_curve_points", "count": curve}],
                         "domain": {"kind": "full_dataset_domain"}},
                        {"id": "flow-domain-input",
                         "entity_counts": [{"entity": "flow_domain_points", "count": domain}],
                         "domain": {"kind": "full_dataset_domain"}}],
                    "direct_outputs": [
                        {"id": "coordinate-query-output",
                         "entity_counts": [{"entity": "flow_domain_points", "count": domain}],
                         "domain": {"kind": "full_dataset_domain"}},
                        {"id": "field-integrated-forces",
                         "entity_counts": [{"entity": "force_coefficients", "count": 1}],
                         "domain": {"kind": "full_dataset_domain"}}],
                    "mappings": [
                        {"support_id": SUPPORTS["domain"], "source_output_id": "coordinate-query-output",
                         "support_count": domain, "scored_count": domain, "unmapped_count": 0,
                         "extrapolated_count": 0, "final_coverage_fraction": 1.0},
                        {"support_id": SUPPORTS["curve"], "source_output_id": "coordinate-query-output",
                         "support_count": curve, "scored_count": curve, "unmapped_count": 0,
                         "extrapolated_count": 0, "final_coverage_fraction": 1.0},
                        {"support_id": SUPPORTS["forces"], "source_output_id": "field-integrated-forces",
                         "support_count": 1, "scored_count": 1, "unmapped_count": 0,
                         "extrapolated_count": 0, "final_coverage_fraction": 1.0}]}}) + "\n")

    parameters = int(training["parameters"])
    physics = ("with two differentiable physics terms (wall no-penetration and outer-domain Bernoulli) weighted "
               f"{physics_weight:g}" if physics_weight else "supervised only, no physics term")
    config = {
        "split_id": split_id, "submission_id": submission_id,
        "result_revision": {"series_id": series_id, "version": 1, "supersedes": None,
                            "change_summary": "Initial published result."},
        "participant": {
            "model": model_name, "model_type": model_type, "model_types": [model_type],
            "training_regime": "from_scratch", "target_data_used": "official_train",
            "external_pretraining": False, "pretraining_data": [],
            "methodology": {
                "format": "fluidsbench-method-v1", "record_kind": "submitter_reported",
                "record_note": (f"Coordinate network trained only on the {training_note or 'public AirfRANS training split'}, "
                                f"{physics}. Predictions are made at the native internal points of every official "
                                f"test case, so the mapping to the domain support is the identity and no interpolation "
                                f"of the benchmark side occurs."),
                "architecture": {
                    "description": (f"Point-wise multilayer perceptron: {training['depth']} hidden layers of width "
                                    f"{training['width']} with SiLU activations, {parameters} trainable parameters, "
                                    f"three outputs per point (u/U_inf, v/U_inf, p/U_inf^2)."),
                    "total_parameter_count": parameters, "parameter_count_basis": "exact",
                    "submitter_trainable_parameter_count": parameters,
                    "components": [{"id": "coordinate-mlp", "family": "point-wise multilayer perceptron",
                                    "role": "maps point features to the three flow quantities",
                                    "description": "eleven input channels, hidden width and depth as declared, "
                                                   "three outputs",
                                    "parameter_count": parameters}],
                    "key_hyperparameters": (
                        [{"id": "width", "component_ids": ["coordinate-mlp"], "name": "hidden width",
                          "value": str(training["width"]), "description": "units per hidden layer"},
                         {"id": "depth", "component_ids": ["coordinate-mlp"], "name": "hidden layers",
                          "value": str(training["depth"]), "description": "SiLU blocks before the output layer"},
                         {"id": "batch", "component_ids": ["coordinate-mlp"], "name": "batch size",
                          "value": str(training["batch"]), "description": "sampled points per optimiser step"},
                         {"id": "lr", "component_ids": ["coordinate-mlp"], "name": "learning rate",
                          "value": str(training["lr"]), "description": "AdamW peak, cosine decay"},
                         {"id": "epochs", "component_ids": ["coordinate-mlp"], "name": "epochs",
                          "value": str(training["epochs"]), "description": "passes over the sampled point set"},
                         {"id": "wall-fraction", "component_ids": ["coordinate-mlp"], "name": "wall sampling share",
                          "value": str(training["wall_fraction"]),
                          "description": "share of each sample taken from the wall and the boundary-layer band"}]
                        + ([{"id": "physics-weight", "component_ids": ["coordinate-mlp"],
                             "name": "physics loss weight", "value": f"{physics_weight:g}",
                             "description": "weight of the wall no-penetration and outer-domain Bernoulli residuals"}]
                           if physics_weight else [])),
                    "input_features": [
                        {"id": "position", "component_ids": ["coordinate-mlp"], "name": "point position",
                         "domain": "volume", "component_count": 2,
                         "description": "x and y in metres, divided by the chord."},
                        {"id": "sdf", "component_ids": ["coordinate-mlp"], "name": "distance to the airfoil",
                         "domain": "volume", "component_count": 2,
                         "description": "the AirfRANS signed distance, divided by the chord, and its signed logarithm."},
                        {"id": "normals", "component_ids": ["coordinate-mlp"], "name": "wall normals",
                         "domain": "volume", "component_count": 2,
                         "description": "the airfoil normal at the nearest wall point, zero away from the wall."},
                        {"id": "surface", "component_ids": ["coordinate-mlp"], "name": "wall mask",
                         "domain": "volume", "component_count": 1,
                         "description": "1 on an airfoil node, 0 elsewhere (the AirfRANS surface flag)."},
                        {"id": "freestream", "component_ids": ["coordinate-mlp"], "name": "freestream direction",
                         "domain": "global", "component_count": 2,
                         "description": "unit inlet velocity, i.e. the case's angle of attack."},
                        {"id": "inlet-scale", "component_ids": ["coordinate-mlp"], "name": "inlet velocity",
                         "domain": "global", "component_count": 1,
                         "description": "log of the inlet velocity magnitude, the case's Reynolds-like scale."}],
                    "predicted_fields": [
                        {"field_id": "two-dimensional-domain-native.U", "domain": "domain", "component_count": 2,
                         "component_ids": ["coordinate-mlp"], "production": "direct_model_output",
                         "description": "Cartesian velocity on every native 2D flow-domain point."},
                        {"field_id": "two-dimensional-domain-native.p", "domain": "domain", "component_count": 1,
                         "component_ids": ["coordinate-mlp"], "production": "direct_model_output",
                         "description": "Kinematic pressure on every native 2D flow-domain point."},
                        {"field_id": "airfoil-curve-native.p", "domain": "curve", "component_count": 1,
                         "component_ids": ["coordinate-mlp"], "production": "derived_from_model_output",
                         "description": "Curve pressure selected from the domain prediction at the wall, as the "
                                        "methodology contract allows."}],
                },
                "data_handling": {
                    "normalization": ("per-point scaling by the chord and the inlet velocity: positions and distances "
                                      "divide by the chord, velocity by U_inf, pressure by U_inf^2. No dataset-wide "
                                      "statistics are used, so the weights transfer across cases."),
                    "preprocessing": ("the public AirfRANS preprocessing (cropped domain, airfoil curve, signed "
                                      "distance, wall normals) read through the AirfRANS 0.1.5.1 library; the native "
                                      "point order is preserved end to end."),
                    "sampling": (f"{training['per_case']} points per training case, a quarter of them from the wall "
                                 f"and the boundary-layer band, resampled every epoch over "
                                 f"{training['train_cases']} training cases; every native point of every official "
                                 f"test case is predicted for inference, in memory-safe chunks."),
                },
                "training": {"stages": [{
                    "id": "surrogate", "status": "performed_by_submitter", "component_ids": ["coordinate-mlp"],
                    "description": f"supervised regression on the official training split of the public release, "
                                   f"{physics}; no test data and no external pretraining.",
                    "procedure": {
                        "kind": "gradient_based",
                        "loss": {"description": ("mean squared error over the three scaled outputs (u/U_inf, v/U_inf, "
                                                 "p/U_inf^2) of the sampled points"
                                                 + (f", plus {physics_weight:g} times the wall no-penetration residual "
                                                    f"and the outer-domain Bernoulli residual" if physics_weight
                                                    else "")),
                                 "weighting": "uniform mean over every component and sampled point"},
                        "optimizers": [{"name": "AdamW", "learning_rate": float(training["lr"]),
                                        "schedule": "cosine decay to lr/50, weight decay 1e-5"}],
                        "batch": {"value": int(training["batch"]), "unit": "sampled points",
                                  "gradient_accumulation_steps": 1},
                        "duration": {"epochs": int(training["epochs"]),
                                     "optimizer_steps": int(training["epochs"] *
                                                            (training["train_points"] // training["batch"] + 1))}},
                    "run_count": 1, "stochastic": True, "random_seeds": [int(training["seed"])],
                    "compute": {
                        "hardware": hardware or f"one {training['device']} device, one process",
                        "max_concurrent_device_count": 1, "devices_per_job": 1,
                        "campaign_wall_time_hours": round(float(training_hours), 4),
                        "aggregate_device_hours": round(float(training_hours), 4),
                        "measurement_basis": "measured", "cost_scope": "final_training",
                        "hardware_identity_basis": "machine_recorded",
                        "measurement_notes": f"one process on one device; wall-clock of the training stage, "
                                             f"{training['seconds']} s measured end to end."}}]},
                "checkpoints": [{"id": "weights", "component_ids": ["coordinate-mlp"],
                                 "sha256": training["weights_sha256"], "digest_scope": "raw_loaded_file",
                                 "bytes_description": (f"{os.path.getsize(checkpoint_path)} bytes, torch state dict of "
                                                       f"the trained network" if checkpoint_path
                                                       else "the model.pt written by the training stage"),
                                 "role": "the trained weights loaded for every prediction",
                                 "selection_rule": "the final epoch of the single run; no checkpoint selection"}],
                "inference_compute": {
                    "status": "measured", "hardware": hardware or f"one {prediction['device']} device, one process",
                    "max_concurrent_device_count": 1, "devices_per_job": 1,
                    "hardware_identity_basis": "machine_recorded",
                    "case_count": int(prediction["cases"]),
                    "campaign_wall_time_seconds": float(prediction["seconds"]),
                    "aggregate_device_time_seconds": float(prediction["seconds"]),
                    "includes_preprocessing": True, "includes_mapping": True, "timing_protocol": "legacy_reported",
                    "measurement_notes": ("wall time of the prediction campaign over the complete official split: "
                                          "feature construction, chunked forward passes and packing of every case's "
                                          "native-order arrays. Model load and framework import are excluded.")},
            },
            "submitter_name": (submitter or {}).get("name", ""),
            "institution": (submitter or {}).get("institution", ""),
            "paper_url": (submitter or {}).get("paper_url", ""),
        },
        "reproducibility": {"contract_version": "open-reproducibility-3.0", "access": "public",
                            "public_test_data_use": "evaluation_only", "result_data_license_spdx": "CC-BY-4.0"},
        "spatial_discretization": {
            "training": {
                "surface_input": {"used": True,
                                  "representation": "airfoil curve point cloud with the AirfRANS signed distance and "
                                                    "wall normals",
                                  "entity_counts": [{"entity": "sampled_training_points",
                                                     "count": {"kind": "fixed", "value": training["per_case"]}}],
                                  "native_comparison": {"status": "not_applicable"},
                                  "sampling": {"kind": "resampled_each_epoch",
                                               "method": "uniform random subset per case, resampled every epoch, a "
                                                         "quarter of the points from the wall and the boundary layer",
                                               "seed": int(training["seed"])},
                                  "domain": {"kind": "full_dataset_domain"}, "connectivity": "none"},
                "surface_supervision": {"used": True, "representation": "targets at the wall nodes of the samples",
                                        "entity_counts": [{"entity": "sampled_training_points",
                                                           "count": {"kind": "fixed",
                                                                     "value": training["per_case"]}}],
                                        "native_comparison": {"status": "not_applicable"},
                                        "sampling": {"kind": "resampled_each_epoch",
                                                     "method": "the wall nodes inside each sampled subset",
                                                     "seed": int(training["seed"])},
                                        "domain": {"kind": "full_dataset_domain"}, "connectivity": "none"},
                "volume_input": {"used": True, "representation": "native internal point cloud",
                                 "entity_counts": [{"entity": "sampled_training_points",
                                                    "count": {"kind": "fixed", "value": training["per_case"]}}],
                                 "native_comparison": {"status": "not_applicable"},
                                 "sampling": {"kind": "resampled_each_epoch",
                                              "method": "uniform random subset per case, resampled every epoch",
                                              "seed": int(training["seed"])},
                                 "domain": {"kind": "full_dataset_domain"}, "connectivity": "none"},
                "volume_supervision": {"used": True, "representation": "native velocity and pressure at the samples",
                                       "entity_counts": [{"entity": "sampled_training_points",
                                                          "count": {"kind": "fixed",
                                                                    "value": training["per_case"]}}],
                                       "native_comparison": {"status": "not_applicable"},
                                       "sampling": {"kind": "resampled_each_epoch",
                                                    "method": "uniform random subset per case, resampled every epoch",
                                                    "seed": int(training["seed"])},
                                       "domain": {"kind": "full_dataset_domain"}, "connectivity": "none"},
            },
            "inference": {
                "geometry_dependency": "surface_geometry",
                "surface_input": {"used": True, "case_record_id": "surface-geometry-input",
                                  "representation": "airfoil curve point cloud of the case",
                                  "entity_counts": [{"entity": "airfoil_curve_points", "count": curve_counts}],
                                  "native_comparison": {"status": "not_applicable"},
                                  "sampling": {"kind": "fixed", "method": "every native airfoil point of the case"},
                                  "domain": {"kind": "full_dataset_domain"}, "connectivity": "none"},
                "volume_input": {"used": True, "case_record_id": "flow-domain-input",
                                 "representation": "native internal point cloud of the case",
                                 "entity_counts": [{"entity": "flow_domain_points", "count": domain_counts}],
                                 "native_comparison": {"status": "not_applicable"},
                                 "sampling": {"kind": "fixed", "method": "every native internal point of the case"},
                                 "domain": {"kind": "full_dataset_domain"}, "connectivity": "none"},
                "direct_outputs": [
                    {"id": "coordinate-query-output", "domain": "volume",
                     "representation": {"used": True,
                                        "representation": "per-point queries at the native internal points, in "
                                                          "memory-safe chunks",
                                        "entity_counts": [{"entity": "flow_domain_points", "count": domain_counts}],
                                        "native_comparison": {"status": "not_applicable"},
                                        "sampling": {"kind": "fixed", "method": "every native internal point"},
                                        "domain": {"kind": "full_dataset_domain"}, "connectivity": "none"},
                     "queries_per_forward_pass": domain_counts},
                    {"id": "field-integrated-forces", "domain": "scalar_case",
                     "representation": {"used": True,
                                        "representation": "the case's force coefficients, integrated from the "
                                                          "submitted pressure and velocity field by the official "
                                                          "evaluator",
                                        "entity_counts": [{"entity": "force_coefficients",
                                                           "count": {"kind": "fixed", "value": 1}}],
                                        "native_comparison": {"status": "not_applicable"},
                                        "sampling": {"kind": "none"},
                                        "domain": {"kind": "full_dataset_domain"}, "connectivity": "none"},
                     "queries_per_forward_pass": {"kind": "fixed", "value": 1}}],
                "mappings": [
                    {"support_id": SUPPORTS["domain"], "source_output_id": "coordinate-query-output",
                     "method": {"kind": "identity"}, "implementation": "scripts/airfrans.py evaluate-entry",
                     "extrapolation_policy": "forbidden", "unmapped_fraction": 0.0, "extrapolated_fraction": 0.0,
                     "final_coverage_fraction": 1.0},
                    {"support_id": SUPPORTS["curve"], "source_output_id": "coordinate-query-output",
                     "method": {"kind": "nearest"}, "implementation": "scripts/airfrans.py evaluate-entry",
                     "extrapolation_policy": "forbidden", "unmapped_fraction": 0.0, "extrapolated_fraction": 0.0,
                     "final_coverage_fraction": 1.0},
                    {"support_id": SUPPORTS["forces"], "source_output_id": "field-integrated-forces",
                     "method": {"kind": "identity"},
                     "implementation": "scripts/airfrans.py evaluate-entry (forces integrated from the submitted wall "
                                       "pressure and velocity field)",
                     "extrapolation_policy": "forbidden", "unmapped_fraction": 0.0, "extrapolated_fraction": 0.0,
                     "final_coverage_fraction": 1.0}],
            },
        },
    }
    path = os.path.join(entry_dir, "package-config.json")
    with open(path, "w") as handle:
        json.dump(config, handle, indent=1)
    return {"entry": os.path.join(entry_dir, "entry.json"), "package_config": path,
            "discretization_cases": os.path.join(entry_dir, "discretization-cases.jsonl"),
            "cases": len(ordered), "submitted_at": submitted_at}
