"""The FluidsBench adapter: contract reading, support chain, mapping, the published metrics, the result package.

The reference numbers in ``test_reproduces_the_published_v3_reference_fixture`` come from the benchmark's own synthetic
fixture, ``examples/v3-template`` in https://github.com/FluidsBench/fluidsbench-submission (Apache-2.0): its ground truth,
its predictions and the per-case and aggregate values its evaluator produced. Reproducing them is the cheapest honest
evidence that this implementation reduces the way the published one does.
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from pinneapple_arena.fluidsbench import (
    Checkout,
    ContractError,
    CoverageError,
    MappingError,
    MetricError,
    aggregate,
    assess_submission,
    build_package,
    competition_rank,
    load_support_release,
    macro_average,
    map_to_support,
    r2,
    relative_l2_from_statistics,
    relative_l2_statistics,
    score_support,
    validate_package,
    write_support_release,
)

# --------------------------------------------------------------------------- fixtures and helpers
PRESSURE_SUPPORT = {
    "id": "volume-points-v1",
    "domain": "volume",
    "location_definition": {
        "mode": "materialized_table", "format": "json", "artifact_role": "ground_truth_table",
        "support_id_rule": {"kind": "artifact_field", "field": "support_id"},
        "coordinate_fields": ["x", "y", "z"],
        "weight_rule": {"kind": "artifact_field", "artifact_role": "ground_truth_table", "field": "weight"},
        "ordering": "support_id_ascending",
    },
    "quantities": [{"id": "pressure", "unit": "Pa", "components": [
        {"id": "pressure", "target_field": "pressure_true", "target_association": "table_field",
         "prediction_field": "pressure_pred"}]}],
    "metric_bindings": [
        {"metric_id": "pressure_rel_l1", "quantity_id": "pressure", "reduction": "relative_l1_percent",
         "weighting": "support_weights", "dataset_weighting": "synthetic_support_weights",
         "aggregation": "per_geometry_then_macro_average", "case_evidence": "metric_value"},
        {"metric_id": "pressure_equal_rel_l2", "quantity_id": "pressure", "reduction": "relative_l2_percent",
         "weighting": "uniform", "dataset_weighting": "entities_equal",
         "aggregation": "per_geometry_then_macro_average", "case_evidence": "metric_value"},
        {"metric_id": "pressure_physical_rel_l2", "quantity_id": "pressure", "reduction": "relative_l2_percent",
         "weighting": "support_weights", "dataset_weighting": "synthetic_support_weights",
         "aggregation": "per_geometry_then_macro_average", "case_evidence": "metric_value"},
        {"metric_id": "pressure_mae", "quantity_id": "pressure", "reduction": "mae", "weighting": "support_weights",
         "dataset_weighting": "synthetic_support_weights", "aggregation": "per_geometry_then_macro_average",
         "case_evidence": "metric_value"},
        {"metric_id": "pressure_rmse", "quantity_id": "pressure", "reduction": "rmse", "weighting": "support_weights",
         "dataset_weighting": "synthetic_support_weights", "aggregation": "per_geometry_then_macro_average",
         "case_evidence": "metric_value"},
    ],
    "required_coverage": {"count_fraction": 1.0, "weight_fraction": 1.0, "unmapped_count": 0},
    "extrapolation_policy": "forbidden",
}

#: The two cases of the published synthetic fixture: ids, coordinates, weights and truth.
FIXTURE_CASES = {
    "case-001": {"support_id": "volume-points-v1", "columns": {
        "support_id": ["s000", "s001", "s002"], "x": [0.0, 1.0, 2.0], "y": [0.0, 0.0, 0.0], "z": [0.0, 0.0, 0.0],
        "weight": [1.0, 2.0, 1.0], "pressure_true": [1.0, 2.0, 4.0]}},
    "case-002": {"support_id": "volume-points-v1", "columns": {
        "support_id": ["s000", "s001", "s002"], "x": [0.0, 1.0, 2.0], "y": [1.0, 1.0, 1.0], "z": [0.0, 0.0, 0.0],
        "weight": [1.0, 1.0, 2.0], "pressure_true": [2.0, 4.0, 6.0]}},
}

#: The predictions of the same fixture, deliberately in a different order than the ground truth.
FIXTURE_PREDICTIONS = {"case-001": (["s002", "s000", "s001"], [5.0, 1.0, 1.0]),
                       "case-002": (["s001", "s002", "s000"], [5.0, 6.0, 1.0])}

#: Per-case and aggregate values the reference evaluator printed for that fixture (percent where the metric is).
REFERENCE = {
    "case-001": {"pressure_rel_l1": 33.333333333333336, "pressure_equal_rel_l2": 30.860669992418384,
                 "pressure_physical_rel_l2": 34.64101615137755, "pressure_mae": 0.75,
                 "pressure_rmse": 0.8660254037844386},
    "case-002": {"pressure_rel_l1": 11.11111111111111, "pressure_equal_rel_l2": 18.89822365046136,
                 "pressure_physical_rel_l2": 14.744195615489714, "pressure_mae": 0.5,
                 "pressure_rmse": 0.7071067811865476},
    "aggregate": {"pressure_rel_l1": 22.22222222222222, "pressure_equal_rel_l2": 24.87944682143987,
                  "pressure_physical_rel_l2": 24.692605883433632, "pressure_mae": 0.625,
                  "pressure_rmse": 0.7865660924854931},
}


@pytest.fixture()
def release(tmp_path):
    manifest = write_support_release(tmp_path / "support", release_id="test-support-v1", dataset_id="synthetic",
                                     supports=[PRESSURE_SUPPORT],
                                     cases={case_id: {"volume-points-v1": payload}
                                            for case_id, payload in FIXTURE_CASES.items()})
    return load_support_release(manifest, "standard")


def _mapped(case, case_id):
    ids, values = FIXTURE_PREDICTIONS[case_id]
    return map_to_support(case, fields={"pressure_pred": values}, method="by_id", support_ids=ids)


def _scores(release):
    scores = []
    for case_id in release.case_ids:
        case = release.support(case_id, "volume-points-v1")
        mapped = _mapped(case, case_id)
        scores.append(score_support(case, mapped.values, coverage=mapped.coverage.as_dict()))
    return scores


def _submission_inputs():
    model = {"model": "Synthetic fixture model", "model_type": "Coordinate network", "model_types": ["Deterministic"],
             "training_regime": "from_scratch", "target_data_used": "official_train", "external_pretraining": False,
             "pretraining_data": [], "parameter_count_millions": 0.001}
    submitter = {"submitter_name": "Example Researcher", "institution": "Example University", "paper_url": "",
                 "code_url": ""}
    methodology = {"format": "fluidsbench-method-v1", "record_kind": "rehearsal",
                   "note": "written by the adapter test; not a real methodology record"}
    reproducibility = {"contract_version": "open-reproducibility-3.0", "access": "public",
                       "public_test_data_use": "evaluation_only", "result_data_license_spdx": "CC-BY-4.0"}
    return {"model": model, "submitter": submitter, "methodology": methodology, "reproducibility": reproducibility}


# ------------------------------------------------------------------------------ the published reductions
def test_reproduces_the_published_v3_reference_fixture(release):
    scores = _scores(release)
    for score in scores:
        expected = REFERENCE[score.case_id]
        for metric_id, value in expected.items():
            assert score.metric_values[metric_id] == pytest.approx(value, rel=1e-12), (score.case_id, metric_id)
    values = aggregate(scores)
    for metric_id, value in REFERENCE["aggregate"].items():
        assert values[metric_id] == pytest.approx(value, rel=1e-12), metric_id


def test_chunk_statistics_are_summed_before_the_square_root():
    truth = np.array([[101325.0], [100980.0], [100410.0]])
    prediction = np.array([[101300.0], [101020.0], [100500.0]])
    weights = np.array([0.10, 0.15, 0.08])
    whole = relative_l2_statistics(truth, prediction, weights, weighting="support_weights")
    chunks = [relative_l2_statistics(truth[s], prediction[s], weights[s], weighting="support_weights")
              for s in (slice(0, 2), slice(2, 3))]
    assert relative_l2_from_statistics(chunks) == pytest.approx(relative_l2_from_statistics([whole]), rel=1e-12)
    chunk_values = [relative_l2_from_statistics([chunk]) for chunk in chunks]
    assert not np.isclose(float(np.mean(chunk_values)), relative_l2_from_statistics([whole]))
    assert whole.entity_count == 3 and whole.total_weight == pytest.approx(0.33)


def test_statistics_reject_what_the_definitions_reject():
    zero_norm = relative_l2_statistics([[0.0]], [[1.0]], [1.0], weighting="support_weights")
    assert zero_norm.denominator == 0.0                    # accumulating is allowed; reporting is not
    with pytest.raises(MetricError):
        relative_l2_from_statistics([zero_norm])
    with pytest.raises(MetricError):
        relative_l2_statistics([[1.0]], [[np.nan]], [1.0], weighting="uniform")
    with pytest.raises(MetricError):
        relative_l2_statistics([[1.0]], [[1.0]], [-1.0], weighting="support_weights")
    with pytest.raises(MetricError):
        r2([[2.0], [2.0]], [[1.0], [2.0]], weighting="uniform")                        # constant truth
    with pytest.raises(MetricError):
        macro_average([])


def test_macro_average_and_competition_rank():
    assert macro_average([1.0, 3.0]) == 2.0
    assert competition_rank([42.44, 42.55, 42.44], direction="higher", decimal_places=1) == [2, 1, 2]
    assert competition_rank([1.04, 1.05, 9.0], direction="lower", decimal_places=1) == [1, 2, 3]


# ---------------------------------------------------------------------------------- the support chain
def test_support_release_verifies_its_chain_and_sorts_by_support_id(release):
    assert release.verified and release.release_id == "test-support-v1" and len(release) == 2
    case = release.support("case-001", "volume-points-v1")
    assert case.support_ids.tolist() == ["s000", "s001", "s002"]
    assert case.coordinates.shape == (3, 3) and case.weights.tolist() == [1.0, 2.0, 1.0]
    np.testing.assert_allclose(case.target("pressure").ravel(), [1.0, 2.0, 4.0])
    assert case.entity_count == 3


def test_a_tampered_ground_truth_is_refused(tmp_path):
    manifest = write_support_release(tmp_path / "support", release_id="tampered-v1", dataset_id="synthetic",
                                     supports=[PRESSURE_SUPPORT],
                                     cases={case_id: {"volume-points-v1": payload}
                                            for case_id, payload in FIXTURE_CASES.items()})
    table = tmp_path / "support" / "case-sets" / "standard" / "data" / "case-001-volume-points-v1-ground-truth.json"
    payload = json.loads(table.read_text())
    payload["columns"]["pressure_true"] = [1.0, 2.0, 9.0]
    table.write_text(json.dumps(payload))
    release = load_support_release(manifest, "standard")            # the chunk itself still hashes
    with pytest.raises(Exception) as error:
        release.support("case-001", "volume-points-v1")             # the table no longer does
    assert "digest" in str(error.value)


def test_an_unsupported_location_mode_says_so(tmp_path):
    native = dict(PRESSURE_SUPPORT)
    native["location_definition"] = dict(PRESSURE_SUPPORT["location_definition"], mode="native_entities")
    manifest = write_support_release(tmp_path / "native", release_id="native-v1", dataset_id="airfrans",
                                     supports=[native],
                                     cases={case_id: {"volume-points-v1": payload}
                                            for case_id, payload in FIXTURE_CASES.items()})
    with pytest.raises(Exception) as error:
        load_support_release(manifest, "standard")
    assert "materialized_table" in str(error.value)


# ------------------------------------------------------------------------------------- the mapping
def test_by_id_joins_exactly_and_identity_needs_every_entity(release):
    case = release.support("case-001", "volume-points-v1")
    mapped = _mapped(case, "case-001")
    np.testing.assert_allclose(mapped.values["pressure_pred"].ravel(), [1.0, 1.0, 5.0])   # in support order
    assert mapped.coverage.coverage_fraction == 1.0 and mapped.coverage.unmapped_count == 0
    with pytest.raises(MappingError):
        map_to_support(case, fields={"pressure_pred": [1.0, 2.0, 3.0]}, method="by_id",
                       support_ids=["s000", "s001", "s999"])                                # unknown id
    with pytest.raises(MappingError):
        map_to_support(case, fields={"pressure_pred": [1.0, 2.0]}, method="identity")    # wrong length


def test_nearest_transfer_reports_unmapped_and_forbidden_extrapolation(release):
    case = release.support("case-001", "volume-points-v1")
    native = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])                 # s002 at x=2.0 has no nearby point
    fields = {"pressure_pred": np.array([1.0, 2.0])}
    mapped = map_to_support(case, points=native, fields=fields, method="nearest", max_distance=0.5,
                            allow_partial=True)
    assert mapped.coverage.unmapped_count == 1 and mapped.coverage.coverage_fraction == pytest.approx(2 / 3)
    assert mapped.coverage.extrapolated_count == 1
    with pytest.raises(CoverageError) as error:                            # forbidding extrapolation is enforced
        map_to_support(case, points=native, fields=fields, method="nearest", max_distance=0.5)
    assert error.value.coverage.unmapped_count == 1


def test_linear_transfer_on_a_grid(release):
    case = release.support("case-001", "volume-points-v1")
    native = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    mapped = map_to_support(case, points=native, fields={"pressure_pred": np.array([1.0, 2.0, 4.0, 3.0])},
                            method="linear")
    np.testing.assert_allclose(mapped.values["pressure_pred"].ravel(), [1.0, 2.0, 4.0])
    assert mapped.coverage.extrapolated_count == 0 and mapped.coverage.coverage_fraction == 1.0


# ---------------------------------------------------------------------------------- the result package
def test_package_round_trip_validates(tmp_path, release):
    scores = _scores(release)
    package = build_package(tmp_path / "submissions" / "synthetic" / "fixture-model-v1",
                            submission_id="fixture-model-v1", release=release, scores=scores, split_id="default",
                            evaluation_command="python -m reference.evaluate_predictions ...", **_submission_inputs())
    assert package.metric_values["pressure_physical_rel_l2"] == pytest.approx(
        REFERENCE["aggregate"]["pressure_physical_rel_l2"], rel=1e-12)
    report = validate_package(package.root)
    assert report.ok, report.problems
    assert any("macro-average" in name for name in report.checked)

    submission, case_metrics = json.loads((package.root / "submission.json").read_text()), json.loads(
        (package.root / "metrics" / "cases.json").read_text())
    assert submission["result_revision"] == {"series_id": "fixture-model", "version": 1, "supersedes": None,
                                             "change_summary": "Initial published result."}
    assert submission["case_metrics"]["sha256"] == package.files["metrics/cases.json"]
    assert case_metrics["case_count"] == 2 and len(case_metrics["cases"]) == 2
    stats = case_metrics["cases"][0]["supports"][0]["metric_sufficient_statistics"]
    assert stats["pressure_physical_rel_l2"]["numerator"] == pytest.approx(3.0)
    assert stats["pressure_physical_rel_l2"]["total_weight"] == pytest.approx(4.0)
    lines = (package.root / "discretization" / "cases.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["inference"]["mappings"][0]["final_coverage_fraction"] == 1.0


def test_package_validation_catches_an_edited_number(tmp_path, release):
    scores = _scores(release)
    package = build_package(tmp_path / "pkg", submission_id="fixture-model-v1", release=release, scores=scores,
                            split_id="default", evaluation_command="…", **_submission_inputs())
    path = package.root / "metrics" / "cases.json"
    payload = json.loads(path.read_text())
    payload["cases"][0]["supports"][0]["metric_values"]["pressure_physical_rel_l2"] = 1.0
    path.write_text(json.dumps(payload))
    report = validate_package(package.root)
    assert not report.ok
    joined = " ".join(report.problems)
    assert "hashes to" in joined                            # the digest chain notices the edit
    assert "sufficient statistics" in joined                # and so does the reduction it contradicts
    assert "macro-average" in joined


# ------------------------------------------------------------------------------------- the trust layer
def test_trust_holds_on_incomplete_coverage_and_reviews_an_unrun_applicability(release, tmp_path):
    scores = _scores(release)
    ledger_path = tmp_path / "ledger.json"
    trust = assess_submission(scores, release=release, split_case_ids=release.case_ids)
    assert trust.decision == "review"                        # applicability was not run: never a neutral pass
    assert trust.ledger.status("fluidsbench/complete entity coverage") == "passed"
    assert trust.ledger.status("fluidsbench/every case is inside the training range") == "not run"

    conditions = {"case-001": {"aoa": 18.5}, "case-002": {"aoa": 4.0}}
    inside = assess_submission(scores, release=release, split_case_ids=release.case_ids,
                               train_bounds={"aoa": (0.0, 15.0)}, case_conditions=conditions)
    assert inside.decision == "review"                       # a case outside the training range is worth saying

    passing = assess_submission(scores, release=release, split_case_ids=release.case_ids,
                                train_bounds={"aoa": (0.0, 15.0)},
                                case_conditions={"case-001": {"aoa": 4.0}, "case-002": {"aoa": 5.0}})
    assert passing.decision == "submit" and passing.passed
    ledger_path.write_text(json.dumps([e.__dict__ for e in passing.ledger.entries], default=str))
    assert ledger_path.exists()


def test_trust_holds_when_a_case_of_the_split_is_missing(release):
    scores = _scores(release)[:1]
    trust = assess_submission(scores, release=release, split_case_ids=release.case_ids)
    assert trust.decision == "hold" and any("whole official split" in r for r in trust.reasons)


# ----------------------------------------------------------------------------------- the contract reader
def _fake_checkout(root):
    specs = root / "benchmark-specs" / "airfrans"
    (specs / "splits").mkdir(parents=True)
    split = {"schema_version": "1.0", "dataset_id": "airfrans", "split_id": "full", "case_set_id": "standard",
             "split_label": "Full", "case_id_status": "official", "case_count": 2, "case_ids": ["a", "b"]}
    split_path = specs / "splits" / "full.json"
    split_path.write_text(json.dumps(split))
    import hashlib

    digest = hashlib.sha256(split_path.read_bytes()).hexdigest()
    spec = {
        "schema_version": "1.1", "dataset_id": "airfrans", "dataset_name": "AirfRANS",
        "dataset_version": "airfrans-native-v1", "status": "official",
        "scoring_support": {"status": "official", "submissions_open": True, "release_id": "support-v1",
                            "manifest_sha256": "0" * 64, "manifest_file": "does-not-exist.json"},
        "evaluation_reference_version": "airfrans-scoring-v2",
        "splits": [{"id": "full", "label": "Full", "index_file": "splits/full.json", "case_count": 2,
                    "case_set_id": "standard", "case_id_status": "official", "sha256": digest}],
        "ranking": {"metric_id": "overall_score", "direction": "higher", "decimal_places": 1,
                    "rounding": "decimal_half_up", "method": "competition"},
        "overall_score_composite": {
            "metric_id": "overall_score", "tolerance": 1e-6,
            "components": [
                {"metric_id": "surface_pressure_rel_l2", "weight": 0.5, "transform": "bounded_error", "cap": 15.0},
                {"metric_id": "cd_r2", "weight": 0.5, "transform": "bounded_quality"}]},
        "component_score_groups": {"field_score": ["surface_pressure_rel_l2"],
                                   "force_score": ["cd_r2"]},
    }
    (specs / "submission-spec.json").write_text(json.dumps(spec))
    (specs / "methodology-contract.json").write_text(json.dumps({
        "dataset_id": "airfrans", "required_predicted_fields": [
            {"field_id": "two-dimensional-domain-native.U", "domain": "domain", "component_count": 2},
            {"field_id": "airfrans-curve-native.p", "domain": "curve", "component_count": 1}]}))
    return root


def test_checkout_reads_the_spec_and_recomputes_the_composite(tmp_path):
    checkout = Checkout(_fake_checkout(tmp_path))
    contract = checkout.dataset("airfrans")
    assert contract.name == "AirfRANS" and contract.status == "official" and contract.open_splits() == ["full"]
    assert contract.split_case_ids("full") == ["a", "b"]
    assert len(contract.required_fields) == 2
    values = {"surface_pressure_rel_l2": 6.0, "cd_r2": 0.93}
    assert contract.composite.score(values) == pytest.approx(0.5 * 60.0 + 0.5 * 93.0)
    assert set(contract.composite.group_scores(values)) == {"field_score", "force_score"}
    assert contract.digests() and all(len(v) == 64 for v in contract.digests().values())


def test_split_digest_mismatch_is_refused(tmp_path):
    checkout = Checkout(_fake_checkout(tmp_path))
    split = tmp_path / "benchmark-specs" / "airfrans" / "splits" / "full.json"
    payload = json.loads(split.read_text())
    payload["case_ids"] = ["a", "b", "c"]
    split.write_text(json.dumps(payload))
    with pytest.raises(ContractError) as error:
        checkout.dataset("airfrans").split_case_ids("full")
    assert "digest" in str(error.value)


def test_checkout_find_names_the_clone_command(tmp_path, monkeypatch):
    monkeypatch.delenv("FLUIDSBENCH_REPO", raising=False)
    with pytest.raises(ContractError) as error:
        Checkout.find()
    assert "git clone" in str(error.value)


def test_unknown_dataset_lists_what_is_there(tmp_path):
    checkout = Checkout(_fake_checkout(tmp_path))
    with pytest.raises(ContractError) as error:
        checkout.dataset("drivaerml")
    assert "airfrans" in str(error.value)
    shutil.rmtree(tmp_path / "benchmark-specs", ignore_errors=True)
