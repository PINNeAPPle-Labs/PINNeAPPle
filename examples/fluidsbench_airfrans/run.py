"""A FluidsBench submission rehearsed end to end, on a reference that can be checked in closed form.

The benchmark itself evaluates AirfRANS, AhmedML, DrivAerML and WindsorML, whose supports run to gigabytes and whose
ground truth is CFD. This example walks the *participant path* — contract, support, mapping, metrics, package,
validation, trust decision — on a case whose exact solution is known, so every number on the way can be checked:

* reference: potential flow around a circular cylinder, the exact solution of the Euler equations at zero Mach. On the
  wall it gives ``|V| = 2U|sin(θ-α)|`` and ``Cp = 1 - 4 sin²(θ-α)``, which this example asserts; its drag is zero
  (d'Alembert), which makes it a legitimate *reference field* and not a legitimate drag benchmark.
* support: two entity sets shaped like the AirfRANS release (`two-dimensional-domain-native` with velocity and kinematic
  pressure, `airfoil-curve-native` with wall pressure), written in the benchmark's own ``manifest → case set → chunk →
  ground-truth table`` layout, with the digests its contract expects.
* splits: three angles of attack are trained on (−4°, 0°, 8°); one is held out *inside* the training range (4°) and two
  are held out *outside* it (12°, 20°). The second is the benchmark's AoA-extrapolation protocol, and it is where the
  trust decision stops agreeing with the score.
* submissions: every model family of ``pinneapple_arena.fieldbench`` at the same budget, one result package per model and
  split, each validated against its own numbers and each given a trust decision.

Run it::

    python examples/fluidsbench_airfrans/run.py                # ~2 min on a laptop
    python examples/fluidsbench_airfrans/run.py --smoke        # one model, a few seconds

Nothing here submits to the leaderboard: the support is a rehearsal written by this example, and every result directory
says so.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from pinneapple_arena.fieldbench import (  # noqa: E402
    FieldDataset,
    FieldSample,
    build_model_specs,
    predict_field_model,
    train_field_model,
)
from pinneapple_arena.fluidsbench import (  # noqa: E402
    Checkout,
    assess_submission,
    build_package,
    load_support_release,
    map_to_support,
    score_support,
    to_table,
    validate_package,
    write_support_release,
)

RADIUS = 0.5
U_INF = 1.0
TRAIN_ANGLES = (-4.0, 0.0, 8.0)
HOLDOUT_ANGLE = 4.0
EXTRAPOLATION_ANGLES = (12.0, 20.0)
CURVE_POINTS = 64
RING_RADIAL, RING_AZIMUTHAL = 8, 40


# ----------------------------------------------------------------------------------- the exact reference
def potential_flow(points: np.ndarray, alpha_deg: float, radius: float = RADIUS, u_inf: float = U_INF) -> np.ndarray:
    """Exact potential flow around a cylinder at angle of attack ``alpha_deg``: ``(u, v, p)`` at ``points``.

    ``u_r = U∞(1 - R²/r²)cos(θ-α)``, ``u_θ = -U∞(1 + R²/r²)sin(θ-α)``; the kinematic pressure follows Bernoulli,
    ``p = ½(U∞² - q²)`` in m²/s² — the unit the AirfRANS release uses for its ``p`` field."""
    alpha = math.radians(float(alpha_deg))
    x, y = np.asarray(points, float)[:, 0], np.asarray(points, float)[:, 1]
    r = np.maximum(np.hypot(x, y), 1e-9)
    theta = np.arctan2(y, x)
    u_r = u_inf * (1.0 - (radius / r) ** 2) * np.cos(theta - alpha)
    u_theta = -u_inf * (1.0 + (radius / r) ** 2) * np.sin(theta - alpha)
    u = u_r * np.cos(theta) - u_theta * np.sin(theta)
    v = u_r * np.sin(theta) + u_theta * np.cos(theta)
    pressure = 0.5 * (u_inf ** 2 - (u ** 2 + v ** 2))
    return np.stack([u, v, pressure], axis=1)


def cylinder_points() -> tuple[np.ndarray, np.ndarray]:
    """The wall curve (closed, ``CURVE_POINTS`` nodes) and the annular 2-D domain, as ``(x, y)`` arrays."""
    theta = np.linspace(0.0, 2.0 * np.pi, CURVE_POINTS, endpoint=False)
    curve = RADIUS * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    radii = np.linspace(1.2 * RADIUS, 4.0 * RADIUS, RING_RADIAL)
    angles = np.linspace(0.0, 2.0 * np.pi, RING_AZIMUTHAL, endpoint=False)
    radius_grid, angle_grid = np.meshgrid(radii, angles, indexing="ij")
    domain = np.stack([(radius_grid * np.cos(angle_grid)).ravel(),
                       (radius_grid * np.sin(angle_grid)).ravel()], axis=1)
    return curve, domain


# ------------------------------------------------------------------------------------ the benchmark contract
CURVE_SUPPORT = {
    "id": "airfoil-curve-native", "domain": "curve",
    "location_definition": {"mode": "materialized_table", "format": "json", "artifact_role": "ground_truth_table",
                            "support_id_rule": {"kind": "artifact_field", "field": "support_id"},
                            "coordinate_fields": ["x", "y"],
                            "weight_rule": {"kind": "artifact_field", "artifact_role": "ground_truth_table",
                                            "field": "weight"},
                            "ordering": "support_id_ascending"},
    "quantities": [{"id": "pressure", "unit": "m^2/s^2", "components": [
        {"id": "pressure", "target_field": "pressure", "target_association": "table_field",
         "prediction_field": "pressure_pred"}]}],
    "metric_bindings": [
        {"metric_id": "surface_pressure_weighted_rel_l2", "quantity_id": "pressure",
         "reduction": "relative_l2_percent", "weighting": "support_weights",
         "dataset_weighting": "boundary_point_dual_length", "aggregation": "per_geometry_then_macro_average"},
        {"metric_id": "surface_pressure_equal_rel_l2", "quantity_id": "pressure", "reduction": "relative_l2_percent",
         "weighting": "uniform", "dataset_weighting": "boundary_nodes_equal",
         "aggregation": "per_geometry_then_macro_average"}],
    "required_coverage": {"count_fraction": 1.0, "weight_fraction": 1.0, "unmapped_count": 0},
    "extrapolation_policy": "forbidden",
}

DOMAIN_SUPPORT = {
    "id": "two-dimensional-domain-native", "domain": "volume",
    "location_definition": {"mode": "materialized_table", "format": "json", "artifact_role": "ground_truth_table",
                            "support_id_rule": {"kind": "artifact_field", "field": "support_id"},
                            "coordinate_fields": ["x", "y"],
                            "weight_rule": {"kind": "artifact_field", "artifact_role": "ground_truth_table",
                                            "field": "weight"},
                            "ordering": "support_id_ascending"},
    "quantities": [
        {"id": "velocity", "unit": "m/s", "components": [
            {"id": "velocity_x", "target_field": "velocity_x", "target_association": "table_field",
             "prediction_field": "velocity_x_pred"},
            {"id": "velocity_y", "target_field": "velocity_y", "target_association": "table_field",
             "prediction_field": "velocity_y_pred"}]},
        {"id": "pressure", "unit": "m^2/s^2", "components": [
            {"id": "pressure", "target_field": "pressure", "target_association": "table_field",
             "prediction_field": "pressure_pred"}]}],
    "metric_bindings": [
        {"metric_id": "flow_domain_velocity_rel_l2", "quantity_id": "velocity", "reduction": "relative_l2_percent",
         "weighting": "uniform", "dataset_weighting": "interior_nodes_equal",
         "aggregation": "per_geometry_then_macro_average"},
        {"metric_id": "flow_domain_velocity_weighted_rel_l2", "quantity_id": "velocity",
         "reduction": "relative_l2_percent", "weighting": "support_weights",
         "dataset_weighting": "interior_point_dual_area", "aggregation": "per_geometry_then_macro_average"},
        {"metric_id": "flow_domain_pressure_rel_l2", "quantity_id": "pressure", "reduction": "relative_l2_percent",
         "weighting": "uniform", "dataset_weighting": "interior_nodes_equal",
         "aggregation": "per_geometry_then_macro_average"}],
    "required_coverage": {"count_fraction": 1.0, "weight_fraction": 1.0, "unmapped_count": 0},
    "extrapolation_policy": "forbidden",
}

REHEARSAL_SPEC = {
    "schema_version": "1.1", "dataset_id": "airfrans-rehearsal", "dataset_name": "AirfRANS (rehearsal)",
    "dataset_version": "airfrans-rehearsal-1", "status": "rehearsal",
    "scoring_support": {"status": "rehearsal", "submissions_open": False,
                        "release_id": "cylinder-native-support-v1", "manifest_sha256": "", "manifest_file": ""},
    "evaluation_reference_version": "pinneapple-fluidsbench-adapter",
    "ranking": {"metric_id": "overall_score", "direction": "higher", "decimal_places": 1,
                "rounding": "decimal_half_up", "method": "competition"},
    "overall_score_composite": {
        "metric_id": "overall_score", "tolerance": 1e-6,
        "components": [
            {"metric_id": "surface_pressure_weighted_rel_l2", "weight": 0.4, "transform": "bounded_error",
             "cap": 30.0},
            {"metric_id": "flow_domain_velocity_rel_l2", "weight": 0.35, "transform": "bounded_error", "cap": 30.0},
            {"metric_id": "flow_domain_pressure_rel_l2", "weight": 0.25, "transform": "bounded_error", "cap": 30.0}]},
    "component_score_groups": {"field_score": ["surface_pressure_weighted_rel_l2", "flow_domain_pressure_rel_l2"],
                               "diagnostic_score": ["flow_domain_velocity_rel_l2"]},
    "splits": [],
}

METHODOLOGY_CONTRACT = {
    "format": "fluidsbench-methodology-contract-v1", "dataset_id": "airfrans-rehearsal",
    "allow_additional_predicted_fields": True,
    "required_predicted_fields": [
        {"field_id": "two-dimensional-domain-native.velocity", "domain": "domain", "component_count": 2},
        {"field_id": "two-dimensional-domain-native.pressure", "domain": "domain", "component_count": 1},
        {"field_id": "airfoil-curve-native.pressure", "domain": "curve", "component_count": 1}]}


def build_case(case_id: str, alpha_deg: float, curve: np.ndarray, domain: np.ndarray) -> dict:
    """One case's ground-truth tables for both supports, and the same points in the model's own frame."""
    curve_values = potential_flow(curve, alpha_deg)
    domain_values = potential_flow(domain, alpha_deg)
    weight = np.full(len(curve), 2.0 * np.pi * RADIUS / len(curve))        # dual segment length of the closed curve
    tables = {
        "airfoil-curve-native": {"columns": {
            "support_id": [f"{case_id}-c{i:03d}" for i in range(len(curve))],
            "x": curve[:, 0], "y": curve[:, 1], "weight": weight, "pressure": curve_values[:, 2]}},
        "two-dimensional-domain-native": {"columns": {
            "support_id": [f"{case_id}-d{i:04d}" for i in range(len(domain))],
            "x": domain[:, 0], "y": domain[:, 1], "weight": np.ones(len(domain)),
            "velocity_x": domain_values[:, 0], "velocity_y": domain_values[:, 1], "pressure": domain_values[:, 2]}},
    }
    points = np.concatenate([curve, domain], axis=0)
    return {"tables": tables, "alpha_deg": alpha_deg, "points": points,
            # the models work in 3-D (z = 0); the support stays in the benchmark's own 2-D frame
            "points3d": np.c_[points, np.zeros(len(points))],
            "values": np.concatenate([curve_values, domain_values], axis=0)}


def write_rehearsal_checkout(root: str, split_cases: dict[str, list[str]], manifest_name: str,
                             manifest_sha256: str) -> str:
    """Write a rehearsal ``benchmark-specs/<id>/`` tree, so the example reads its contract the way a participant does."""
    specs = os.path.join(root, "benchmark-specs", "airfrans-rehearsal")
    os.makedirs(os.path.join(specs, "splits"), exist_ok=True)
    labels = {"full": "Full", "holdout": "Hold-out", "aoa_extrapolation": "AoA extrapolation"}
    splits = []
    for split_id, ids in split_cases.items():
        payload = {"schema_version": "1.0", "dataset_id": "airfrans-rehearsal", "split_id": split_id,
                   "case_set_id": "standard", "split_label": labels[split_id], "case_id_status": "rehearsal",
                   "case_count": len(ids), "case_ids": ids}
        path = os.path.join(specs, "splits", f"{split_id}.json")
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=1)
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        splits.append({"id": split_id, "label": labels[split_id], "index_file": f"splits/{split_id}.json",
                       "case_count": len(ids), "case_set_id": "standard", "case_id_status": "rehearsal",
                       "sha256": digest})
    spec = dict(REHEARSAL_SPEC, splits=splits)
    spec["scoring_support"] = dict(REHEARSAL_SPEC["scoring_support"], manifest_file=manifest_name,
                                   manifest_sha256=manifest_sha256)
    with open(os.path.join(specs, "submission-spec.json"), "w") as handle:
        json.dump(spec, handle, indent=1)
    with open(os.path.join(specs, "methodology-contract.json"), "w") as handle:
        json.dump(METHODOLOGY_CONTRACT, handle, indent=1)
    return root


# ----------------------------------------------------------------------------------- the participant workflow
def _score_cases(release, case_ids: list[str], predictions: list[np.ndarray], cases: dict, curve: np.ndarray) -> list:
    """Map one model's output onto both supports of every scored case and compute the bound metrics."""
    n_curve = len(curve)
    scores = []
    for case_id, values in zip(case_ids, predictions, strict=True):
        points = cases[case_id]["points3d"]
        for support_id, rows, fields in (
                (CURVE_SUPPORT["id"], slice(0, n_curve), {"pressure_pred": values[:n_curve, 2]}),
                (DOMAIN_SUPPORT["id"], slice(n_curve, None),
                 {"velocity_x_pred": values[n_curve:, 0], "velocity_y_pred": values[n_curve:, 1],
                  "pressure_pred": values[n_curve:, 2]})):
            support = release.support(case_id, support_id)
            # the model is queried at the canonical entities themselves: the clean identity mapping
            mapped = map_to_support(support, points=points[rows], fields=fields, method="identity")
            scores.append(score_support(support, mapped.values, coverage=mapped.coverage.as_dict()))
    return scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="tiny budget, one model")
    parser.add_argument("--out", default=os.path.join(HERE, "_out"))
    args = parser.parse_args(argv)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    epochs = 4 if args.smoke else 60
    started = time.time()

    # 1. the reference, and the benchmark-owned support written around it
    curve, domain = cylinder_points()
    angles = {f"train_{alpha:+.0f}": alpha for alpha in TRAIN_ANGLES}
    angles[f"holdout_{HOLDOUT_ANGLE:+.0f}"] = HOLDOUT_ANGLE
    angles.update({f"extrapolation_{alpha:+.0f}": alpha for alpha in EXTRAPOLATION_ANGLES})
    cases = {case_id: build_case(case_id, alpha, curve, domain) for case_id, alpha in angles.items()}
    split_cases = {"full": list(cases),
                   "holdout": [c for c in cases if c.startswith("holdout")],
                   "aoa_extrapolation": [c for c in cases if c.startswith("extrapolation")]}
    manifest_path = write_support_release(
        os.path.join(out, "support"), release_id="cylinder-native-support-v1", dataset_id="airfrans-rehearsal",
        supports=[CURVE_SUPPORT, DOMAIN_SUPPORT],
        cases={case_id: case["tables"] for case_id, case in cases.items()},
        notes="Rehearsal support of the FluidsBench adapter example: potential flow around a cylinder, exact ground truth.")
    with open(manifest_path, "rb") as handle:
        manifest_digest = hashlib.sha256(handle.read()).hexdigest()
    checkout_root = os.path.join(out, "checkout")
    checkout = Checkout(write_rehearsal_checkout(checkout_root, split_cases,
                                                 os.path.relpath(manifest_path, checkout_root), manifest_digest))
    contract = checkout.dataset("airfrans-rehearsal")
    release = load_support_release(manifest_path, "standard")
    print(f"contract {contract.name}: {len(contract.splits)} splits, composite with "
          f"{len(contract.composite.components)} components; support {release.release_id} verified "
          f"({len(release)} cases, {len(curve)} wall + {RING_RADIAL * RING_AZIMUTHAL} domain entities each)")
    print(f"  files hashed when the contract was read: {', '.join(contract.digests())}")

    # 2. one field sample per case, in the support's own entity order
    dataset = FieldDataset(
        name="cylinder-potential-flow",
        samples=[FieldSample(id=case_id, points=case["points3d"], values=case["values"],
                             category=case_id.split("_")[0],
                             thickness=np.full(len(case["points3d"]), case["alpha_deg"] / 20.0, dtype=np.float32),
                             meta={"alpha_deg": case["alpha_deg"]})
                 for case_id, case in cases.items()],
        splits={"train": [c for c in cases if c.startswith("train")],
                "val": [c for c in cases if c.startswith("train")][:1],
                "test": split_cases["holdout"] + split_cases["aoa_extrapolation"]})

    # 3. every model family at the same budget, then the benchmark's own scoring for each split
    specs = build_model_specs(in_dim=3, out_dim=3)
    if args.smoke:
        specs = specs[:1]
    scored_cases = split_cases["holdout"] + split_cases["aoa_extrapolation"]
    bounds = {"aoa": (min(TRAIN_ANGLES), max(TRAIN_ANGLES))}
    conditions = {case_id: {"aoa": case["alpha_deg"]} for case_id, case in cases.items()}
    packages: dict[str, list] = {"holdout": [], "aoa_extrapolation": []}
    summary = []

    def submit(name: str, family: str, notes: str, parameters: int, predicted: list[np.ndarray],
               epochs_run: int, train_seconds: float, inference_seconds: float, device: str,
               record_kind: str = "rehearsal", contender: bool = True) -> list:
        """Score one method on both splits, write its package, validate it, and record its trust decision."""
        scores = _score_cases(release, scored_cases, predicted, cases, curve)
        built = []
        for split_id in ("holdout", "aoa_extrapolation"):
            members = set(split_cases[split_id])
            split_scores = [score for score in scores if score.case_id in members]
            submission_id = f"{name.replace('_', '-')}-{split_id.replace('_', '-')}-v1"
            package = build_package(
                os.path.join(out, "submissions", contract.dataset_id, submission_id),
                submission_id=submission_id, release=release, scores=split_scores, split_id=split_id,
                contract=contract, dataset_version=contract.dataset_version,
                split_sha256=contract.split(split_id).expected_sha256,
                model={"model": f"{name} ({family})", "model_type": "Point-to-point field surrogate",
                       "model_types": [family], "training_regime": "from_scratch",
                       "target_data_used": "official_train", "external_pretraining": False,
                       "pretraining_data": [], "parameter_count_millions": round(parameters / 1e6, 6)},
                submitter={"submitter_name": "PINNeAPPle example", "institution": "", "paper_url": "", "code_url": ""},
                methodology={"format": "fluidsbench-method-v1", "record_kind": record_kind,
                             "architecture": {"description": notes, "total_parameter_count": parameters,
                                              "parameter_count_basis": "exact",
                                              "submitter_trainable_parameter_count": parameters},
                             "data_handling": {"normalization": "per-channel standardisation of the training labels",
                                               "sampling": f"{CURVE_POINTS} wall nodes + "
                                                           f"{RING_RADIAL * RING_AZIMUTHAL} domain nodes per case"},
                             "training": {"stages": [{"name": "surrogate", "epochs": epochs_run,
                                                      "wall_time_seconds": round(train_seconds, 3),
                                                      "seeds": [0]}] if epochs_run else []},
                             "inference_compute": {"status": "measured", "case_count": len(split_cases[split_id]),
                                                   "campaign_wall_time_seconds": round(inference_seconds, 6),
                                                   "includes_mapping": True, "hardware": device}},
                evaluation_command="python examples/fluidsbench_airfrans/run.py",
                discretization={"inference": {
                    "geometry_dependency": "surface_geometry",
                    "direct_outputs": [{"id": "model-output", "domain": "volume",
                                        "representation": {"used": True,
                                                           "representation": "canonical scoring entities"}}],
                    "mappings": []}},
                reproducibility={"contract_version": "open-reproducibility-3.0", "access": "public",
                                 "public_test_data_use": "evaluation_only",
                                 "result_data_license_spdx": "CC-BY-4.0"},
                notes="Rehearsal package written on a synthetic support; not a FluidsBench submission.")
            report = validate_package(package.root, contract=contract)
            trust = assess_submission(split_scores, release=release, contract=contract,
                                      split_case_ids=split_cases[split_id], train_bounds=bounds,
                                      case_conditions={c: conditions[c] for c in split_cases[split_id]})
            if contender:                                         # the oracle is a check, not a competitor
                packages[split_id].append(package)
            summary.append({"model": name, "family": family, "split": split_id, "parameters": parameters,
                            "epochs": epochs_run, "train_seconds": round(train_seconds, 2),
                            "metrics": {k: round(v, 4) for k, v in package.metric_values.items()},
                            "overall_score": (round(package.overall_score, 3)
                                              if package.overall_score is not None else None),
                            "package_valid": report.ok, "trust": trust.decision,
                            "checks": {c.name: c.passed for c in trust.checks}})
            print(f"\n{name} · {split_id} · {parameters} parameters · {epochs_run} epochs · "
                  f"train {train_seconds:.1f}s · inference {inference_seconds:.3f}s")
            print("  " + "  ".join(f"{k}={v:.3f}%" for k, v in package.metric_values.items()
                                   if k not in ("overall_score", "field_score", "diagnostic_score")))
            if package.overall_score is not None:
                print(f"  overall_score {package.overall_score:.4f}")
            print("  package: " + ("PASS" if report.ok else "FAIL — " + "; ".join(report.problems)))
            print(f"  trust: {trust.decision.upper()}" + (f" — {'; '.join(trust.reasons)}" if trust.reasons else ""))
            built.append(package)
        return built

    for spec in specs:
        model = spec["build"]()
        uses_parts = spec.get("uses_parts", False)
        started_training = time.perf_counter()
        history = train_field_model(model, dataset, train_ids=dataset.splits["train"], val_ids=dataset.splits["val"],
                                    epochs=epochs, batch_size=2, lr=2e-3, seed=0, uses_parts=uses_parts)
        train_seconds = time.perf_counter() - started_training
        started_inference = time.perf_counter()
        predicted = predict_field_model(model, dataset, scored_cases, device=history["device"], uses_parts=uses_parts)
        inference_seconds = time.perf_counter() - started_inference
        submit(spec["name"], spec["family"], spec["notes"], int(sum(p.numel() for p in model.parameters())),
               predicted, history["epochs_run"], train_seconds, inference_seconds, history["device"])

    # the pipeline's own sanity check: the reference scored against itself must come out at zero error and full
    # score. Nothing about a model is tested here; everything about the mapping, the reductions and the composite is.
    oracle = submit("reference-oracle", "analytic (zero parameters)", "the reference field itself — a check of the "
                    "evaluation chain, not a contender", 0, [cases[c]["values"] for c in scored_cases],
                    0, 0.0, 0.0, "none", record_kind="rehearsal", contender=False)
    oracle_error = max(abs(p.metric_values[m]) for p in oracle for m in p.metric_values if m.endswith("rel_l2"))
    assert oracle_error < 1e-9, oracle_error
    assert all(abs(p.overall_score - 100.0) < 1e-9 for p in oracle), [p.overall_score for p in oracle]
    print("\nthe reference scored against itself: relative L2 = 0 and overall_score = 100, so the mapping, the "
          "reductions and the composite agree with the ground truth they were given")

    # 4. the two leaderboards, and the reference checks this rehearsal rests on
    tables = {split_id: to_table(packages[split_id], ranking=contract.ranking)
              for split_id in ("holdout", "aoa_extrapolation")}
    for split_id, table in tables.items():
        print(f"\n### {split_id}\n{table}")
    decisions = {split_id: sorted({entry["trust"] for entry in summary if entry["split"] == split_id})
                 for split_id in tables}
    print(f"\ntrust decisions: inside the training range {decisions['holdout']}, "
          f"outside it {decisions['aoa_extrapolation']}")

    wall = potential_flow(curve, 0.0)
    wall_speed_max = float(np.max(np.hypot(wall[:, 0], wall[:, 1])))
    stagnation = float(potential_flow(np.array([[RADIUS, 0.0]]), 0.0)[0, 2])
    assert abs(stagnation - 0.5) < 1e-12, stagnation                      # Cp = 1 at the stagnation point
    assert abs(wall_speed_max - 2.0 * U_INF) < 1e-12, wall_speed_max      # |V| = 2U at the shoulders
    print(f"reference checks: stagnation pressure {stagnation:.6f} = ½U², maximum wall speed "
          f"{wall_speed_max:.6f} = 2U; the drag of this exact solution is zero (d'Alembert), so it is a reference "
          f"field, not a drag benchmark")

    result = {"schema": "pinneapple.fluidsbench.rehearsal/v1",
              "train_angles_deg": list(TRAIN_ANGLES), "holdout_angle_deg": HOLDOUT_ANGLE,
              "extrapolation_angles_deg": list(EXTRAPOLATION_ANGLES),
              "points_per_case": {"curve": CURVE_POINTS, "domain": RING_RADIAL * RING_AZIMUTHAL},
              "budget": {"epochs": epochs, "batch_size": 2, "lr": 2e-3},
              "support_release": release.release_id, "support_manifest_sha256": release.manifest_sha256,
              "contract_digests": contract.digests(), "trust_decisions": decisions,
              "reference": {"stagnation_pressure": stagnation, "wall_speed_max": wall_speed_max, "drag": 0.0,
                            "source": "potential flow around a cylinder (exact Euler solution)"},
              "submissions": summary, "leaderboards": tables,
              "wall_time_seconds": round(time.time() - started, 1)}
    with open(os.path.join(out, "rehearsal.json"), "w") as handle:
        json.dump(result, handle, indent=1)
    with open(os.path.join(out, "leaderboard.md"), "w") as handle:
        handle.write("\n\n".join(f"### {split_id}\n{table}" for split_id, table in tables.items()) + "\n")
    print(f"\nwrote {out}/rehearsal.json and {sum(len(v) for v in packages.values())} result package(s) under "
          f"{out}/submissions")
    assert all(entry["package_valid"] for entry in summary), "every package must validate on its own numbers"
    return 0


def run(smoke: bool = False, out: str | None = None) -> dict:
    """Programmatic entry point (the example runner and the tests call this)."""
    argv = ["--smoke"] if smoke else []
    if out:
        argv += ["--out", out]
    main(argv)
    with open(os.path.join(os.path.abspath(out or os.path.join(HERE, "_out")), "rehearsal.json")) as handle:
        return json.load(handle)


if __name__ == "__main__":
    raise SystemExit(main())
