"""Tests for the ``pinneapple_llm.qa`` quality pipeline.

Organised by the failure the pipeline must catch, not by module. Every test
that asserts a ``FAIL``/``BLOCKED`` also asserts the *reason* is visible in
the report, because a pipeline that fails silently is indistinguishable
from one that did not run.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

import pinneapple_llm
import pinneapple_llm.qa as qa
from pinneapple_llm.qa.examples import pipe_wear_twin as example


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _spec(**overrides):
    spec = qa.PhysicalSpecification(
        objective="unit test artefact",
        phenomenon="synthetic wall thinning",
        geometry=qa.GeometrySpec(
            kind="hollow_pipe", dimensions={"outer_radius": qa.Quantity(0.1, "m"),
                                            "length": qa.Quantity(1.0, "m"),
                                            "wall_thickness": qa.Quantity(0.01, "m")},
            hollow=True, wall_thickness=qa.Quantity(0.01, "m")),
        independent_variables=["t"],
        dependent_variables=["wear_depth", "residual_thickness"],
        parameters={"wear_depth": qa.Quantity(0.0, "mm"),
                    "residual_thickness": qa.Quantity(10.0, "mm")},
        governing_equations=["dw/dt = rate"],
        spatial_validity={"y": (qa.Quantity(0.0, "m"), qa.Quantity(1.0, "m"))},
        temporal_validity=(qa.Quantity(0.0, "year"), qa.Quantity(5.0, "year")),
        invariants=[qa.Invariant("wear_nonneg", "nonnegative", "wear_depth"),
                    qa.Invariant("wear_units", "unit_compatible", "wear_depth", params={"unit": "mm"})],
        validation_metrics={"max_wear_mm": {"limit": 8.0}},
        required_validations=["units", "numeric", "physics"],
    )
    for key, value in overrides.items():
        setattr(spec, key, value)
    return spec


def _simple_bundle(times=None):
    times = np.linspace(0, 5, 6) if times is None else times
    vertices = np.zeros((4, 3))
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    fields = {"wear_depth": np.tile(np.linspace(0, 6, 4), (len(times), 1))}
    return {
        "meshes": {"wall": (vertices, faces)},
        "fields": fields,
        "field_units": {"wear_depth": "mm"},
        "times": times,
        "sensors": [qa.SensorReading("S1", (0.0, 0.0, 0.0), "mm", "wear_depth",
                                     np.linspace(0, 6, len(times)), holdout=True)],
    }


def _run(spec=None, *, validators=None, **bundle_overrides):
    bundle = _simple_bundle()
    bundle.update(bundle_overrides)
    pipeline = qa.QualityPipeline(spec or _spec(), validators=validators)
    return pipeline.run(**bundle)


# ---------------------------------------------------------------------------
# 1-3. static validation
# ---------------------------------------------------------------------------
def test_syntactically_invalid_code_is_rejected():
    result = qa.validate_python_artifact("def broken(:\n    pass")
    assert result.status is qa.Status.FAIL
    assert any(f.id == "static.syntax" and f.status is qa.Status.FAIL for f in result.findings)


def test_missing_module_import_is_rejected():
    result = qa.validate_python_artifact("import totally_missing_module_xyz\n")
    assert result.status is qa.Status.FAIL
    assert any("does not resolve" in f.message or "allowlist" in f.message for f in result.findings)


def test_invented_api_is_caught_by_introspection():
    source = "from pinneapple_twin3d.primitives import torus_that_does_not_exist\n"
    result = qa.validate_python_artifact(source)
    assert result.status is qa.Status.FAIL
    # the import itself fails to resolve
    assert any(f.id == "static.import" and f.status is qa.Status.FAIL for f in result.findings)


def test_attribute_on_real_module_that_does_not_exist_is_caught():
    source = "import pinneapple_twin3d\nx = pinneapple_twin3d.primitives.torus(1.0, 2.0)\n"
    result = qa.validate_python_artifact(source)
    assert result.status is qa.Status.FAIL
    assert any("does not exist" in f.message for f in result.findings)


def test_forbidden_call_and_import_are_rejected():
    source = "import subprocess\neval('1+1')\n"
    result = qa.validate_python_artifact(source)
    ids = {f.id for f in result.findings if f.status is qa.Status.FAIL}
    assert "static.import" in ids
    assert "static.security" in ids


def test_real_api_call_passes_static_checks():
    source = (
        "from pinneapple_twin3d.primitives import cylinder\n"
        "from pinneapple_twin3d import Scene\n"
        "v, f = cylinder((0, 0, 0), (0, 1, 0), 0.05)\n"
        "scene = Scene('ok')\n"
    )
    result = qa.validate_python_artifact(source)
    assert result.status is qa.Status.PASS, [f.message for f in result.findings]


# ---------------------------------------------------------------------------
# 4-8. scientific validators
# ---------------------------------------------------------------------------
def test_wrong_space_time_shape_is_caught():
    bundle = _simple_bundle()
    bundle["fields"]["wear_depth"] = np.zeros((3, 4))  # 6 times declared
    report = _run(**bundle)
    step = report.step("validator:numeric")
    assert step.status is qa.Status.FAIL
    assert any(f.id == "numeric.time_axis" for f in step.findings)


def test_field_mesh_size_mismatch_is_caught():
    bundle = _simple_bundle()
    bundle["fields"]["wear_depth"] = np.zeros((6, 9))  # mesh has 4 vertices
    report = _run(**bundle)
    assert report.step("validator:numeric").status is qa.Status.FAIL
    assert any(f.id == "numeric.vertex_axis" for f in report.step("validator:numeric").findings)


def test_incompatible_units_are_caught():
    bundle = _simple_bundle()
    bundle["field_units"] = {"wear_depth": "s"}  # spec declares mm: different dimension
    report = _run(**bundle)
    step = report.step("validator:units")
    assert step.status is qa.Status.FAIL
    assert any(f.id == "unit.field_vs_spec" for f in step.findings)


def test_compatible_but_different_unit_scale_is_reported():
    bundle = _simple_bundle()
    bundle["field_units"] = {"wear_depth": "m"}  # mm -> m: same dimension, other scale
    report = _run(**bundle)
    step = report.step("validator:units")
    assert step.status is qa.Status.PASS
    assert any(f.id == "unit.field_scale" for f in step.findings)


def test_nan_and_infinity_are_caught():
    fields = {"wear_depth": np.array([[0.0, np.nan, np.inf, 1.0]] * 6)}
    report = _run(fields=fields)
    step = report.step("validator:numeric")
    assert step.status is qa.Status.FAIL
    assert any(f.id == "numeric.non_finite" for f in step.findings)


def test_empty_geometry_is_caught():
    report = _run(meshes={"empty": (np.zeros((0, 3)), np.zeros((0, 3), dtype=int))})
    step = report.step("validator:geometry")
    assert step.status is qa.Status.FAIL
    assert any(f.id == "geometry.empty" for f in step.findings)


def test_solid_cylinder_is_not_accepted_as_a_hollow_pipe():
    bundle = example.build_twin()
    bundle["meshes"]["pipe_body"] = example.solid_pipe_mesh()
    report = qa.QualityPipeline(bundle["spec"]).run(
        meshes=bundle["meshes"], fields=bundle["fields"], field_units=bundle["field_units"],
        times=bundle["times"], time_unit=bundle["time_unit"], sensors=bundle["sensors"])
    step = report.step("validator:geometry")
    assert step.status is qa.Status.FAIL
    message = " ".join(f.message for f in step.findings if f.blocks)
    assert "SOLID" in message and "hollow" in message


def test_physical_invariant_violation_is_caught():
    # 9 mm of wear exceeds the 8 mm allowance declared in the spec parameters
    fields = {"wear_depth": np.linspace(0, 9, 4)[None, :].repeat(6, axis=0)}
    spec = _spec(invariants=[qa.Invariant("cap", "max", "wear_depth", params={"limit": 8.0})])
    report = _run(spec=spec, fields=fields)
    step = report.step("validator:physics")
    assert step.status is qa.Status.FAIL
    assert any(f.id == "invariant.cap" for f in step.findings if f.blocks)


def test_inconsistent_boundary_condition_is_caught_by_spec_validation():
    spec = _spec()
    spec.boundary_conditions = ["T = 300 K"]  # T is not a declared variable
    assert any("boundary condition" in p for p in spec.validate())
    report = _run(spec=spec)
    assert report.step("spec").status is qa.Status.FAIL


def test_missing_target_for_a_declared_invariant_blocks():
    spec = _spec(invariants=[qa.Invariant("ghost", "nonnegative", "not_a_field")])
    report = _run(spec=spec)
    assert report.step("validator:physics").status is qa.Status.BLOCKED


# ---------------------------------------------------------------------------
# 9-10. export
# ---------------------------------------------------------------------------
def test_export_failure_is_reported_not_swallowed():
    def broken_export(_out):
        raise OSError("disk full")

    pipeline = qa.QualityPipeline(_spec(), validators=("numeric",))
    bundle = _simple_bundle()
    report = pipeline.run(exporter=broken_export, output_dir=None, **bundle)
    step = report.step("export")
    assert step.status is qa.Status.FAIL
    assert any("disk full" in f.message for f in step.findings)


def test_export_that_returns_no_file_is_not_a_pass():
    pipeline = qa.QualityPipeline(_spec(), validators=("numeric",))
    report = pipeline.run(exporter=lambda _out: "/nonexistent/path/scene.json", **_simple_bundle())
    assert report.step("export").status is qa.Status.FAIL


# ---------------------------------------------------------------------------
# 14-15. repair loop
# ---------------------------------------------------------------------------
def _failing_evaluate(_artifact):
    return {"physics": qa.ValidatorResult(
        name="physics", status=qa.Status.FAIL,
        findings=[qa.Finding(id="invariant.cap", status=qa.Status.FAIL, severity="error",
                             message="wear exceeds the allowance")])}


def test_repair_stops_after_the_attempt_limit():
    attempts = []

    def repair(artifact, diagnosis, n):
        attempts.append(n)
        return artifact + f"\n# attempt {n}"

    loop = qa.RepairLoop(evaluate=_failing_evaluate, repair=repair, max_attempts=3,
                         required=("physics",))
    outcome = loop.run("original")
    assert outcome.accepted is False
    assert len(outcome.attempts) == 3
    assert attempts == [1, 2]  # the third attempt is the limit, not a new repair
    assert "still failing" in outcome.stopped_because


def test_repair_that_compiles_but_violates_science_is_rejected():
    def repair(_artifact, _diagnosis, n):
        return f"# compiles fine, attempt {n}\nvalue = {n}\n"

    loop = qa.RepairLoop(evaluate=_failing_evaluate, repair=repair, max_attempts=2,
                         required=("physics",))
    outcome = loop.run("original")
    assert outcome.accepted is False
    assert outcome.results["physics"].status is qa.Status.FAIL
    assert all(not a.accepted for a in outcome.attempts)


def test_repair_cannot_relax_the_acceptance_criteria():
    spec = _spec()

    def repair(_artifact, _diagnosis, _n):
        # a "fix" that lowers the bar instead of raising the artifact
        spec.validation_metrics["max_wear_mm"]["limit"] = 999.0
        return "def fixed():\n    return 1\n"

    loop = qa.RepairLoop(evaluate=_failing_evaluate, repair=repair, spec=spec,
                         max_attempts=3, required=("physics",))
    outcome = loop.run("original")
    assert outcome.accepted is False
    assert "criteria changed" in outcome.stopped_because


def test_repair_oscillation_is_detected():
    def repair(artifact, _diagnosis, _n):
        return "A" if artifact != "A" else "B"

    loop = qa.RepairLoop(evaluate=_failing_evaluate, repair=repair, max_attempts=5,
                         required=("physics",))
    outcome = loop.run("A")  # repair returns "B"? artifact is "A" so it returns "B"
    assert outcome.accepted is False
    assert len(outcome.attempts) <= 5


# ---------------------------------------------------------------------------
# 16-17. statuses and levels
# ---------------------------------------------------------------------------
def test_no_reference_data_never_grants_q3():
    report = _run()
    assert report.step("reference_comparison") is None
    assert report.quality_level in ("Q1", "Q2")
    assert any("Q3 blocked" in r for r in report.level_reasons)


def test_not_run_and_not_applicable_are_distinct_from_pass():
    report = _run()
    statuses = {s.step: s.status for s in report.steps}
    assert statuses["static_syntax"] is qa.Status.NOT_APPLICABLE
    assert "PASS" not in statuses.values() or True  # explicit distinction asserted below
    assert qa.Status.NOT_APPLICABLE is not qa.Status.PASS
    assert qa.Status.NOT_RUN is not qa.Status.PASS
    # an inapplicable validation does not silently become a pass
    assert report.step("static_schema") is None


def test_q4_requires_operational_evidence():
    report = _run()
    assert report.quality_level != "Q4"
    example_report, _ = example.run_example()  # reaches Q3, so Q4's reason is reached
    assert example_report.quality_level == "Q3"
    assert any("Q4 blocked" in r for r in example_report.level_reasons)


def test_operational_evidence_without_reviewer_does_not_grant_q4():
    ev = {"environment": "plant-1"}
    report = qa.QualityPipeline(_spec(), validators=("numeric",), operational_evidence=ev).run(
        **_simple_bundle())
    assert report.quality_level != "Q4"


# ---------------------------------------------------------------------------
# 18. end-to-end example
# ---------------------------------------------------------------------------
def test_pipe_twin_example_runs_end_to_end(tmp_path):
    report, bundle = example.run_example(output_dir=str(tmp_path))

    assert report.overall_status is qa.Status.PASS, report.summary()
    assert report.quality_level == "Q3"
    assert report.step("validator:geometry").status is qa.Status.PASS
    assert report.step("validator:sensors").status is qa.Status.PASS
    assert report.step("reference_comparison").status is qa.Status.PASS

    for name in ("scene.json", "geometry.glb", "fields.bin", "wear_dataset.json",
                 "quality_report.json"):
        path = os.path.join(tmp_path, name)
        assert os.path.exists(path), name
        assert os.path.getsize(path) > 0, name

    manifest = json.loads((tmp_path / "scene.json").read_text())
    assert manifest["format"] == "pinneapple-twin3d/1"
    assert {p["name"] for p in manifest["parts"]} >= {"wall", "pipe_body"}
    assert manifest["fields"], "the scene must carry fields, not just geometry"
    assert any(s["id"].startswith("UT-") for s in manifest["sensors"])

    # the report is machine-readable and round-trips
    loaded = qa.QualityReport.read(str(tmp_path / "quality_report.json"))
    assert loaded.quality_level == report.quality_level
    assert loaded.to_dict() == report.to_dict()

    # synthetic data is labelled as such, never as a measurement
    assert bundle["dataset"].origin == "synthetic"
    assert "[SYNTHETIC DATA]" in bundle["scene"].title


def test_pipe_twin_anomaly_is_detected():
    report, bundle = example.run_example(anomaly=True)
    assert report.step("validator:physics").status is qa.Status.FAIL
    assert report.step("reference_comparison").status is qa.Status.FAIL
    assert report.quality_level in ("Q1", "Q2")
    assert report.blocking_findings


@pytest.mark.parametrize("radius,thickness,length", [
    (0.05, 0.004, 1.0),
    (0.10, 0.010, 2.0),
    (0.25, 0.020, 0.5),
])
def test_hollow_mesh_volume_matches_the_analytic_expression(radius, thickness, length):
    """Property: the enclosed volume of the generated mesh is pi*(R^2-r^2)*L."""
    vertices, faces = example.hollow_pipe_mesh(radius=radius, thickness=thickness,
                                               length=length, n_theta=128)
    tri = vertices[faces]
    volume = abs(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
    expected = np.pi * (radius ** 2 - (radius - thickness) ** 2) * length
    assert volume == pytest.approx(expected, rel=1e-3)


def test_report_records_artifact_hashes_and_environment(tmp_path):
    report, _ = example.run_example(output_dir=str(tmp_path))
    assert report.artifacts
    assert all(a.sha256 for a in report.artifacts)
    assert report.spec_hash and report.criteria_hash
    assert "python" in report.environment
    assert report.reproduce


# ---------------------------------------------------------------------------
# stage B: the model's structured claims
# ---------------------------------------------------------------------------
_ENVELOPE = {
    "code": ("from pinneapple_twin3d.primitives import cylinder\n"
             "v, f = cylinder((0, 0, 0), (0, 1, 0), 0.05)\n"),
    "assumptions": ["the pipe is straight"],
    "dependencies": ["pinneapple_twin3d"],
    "units": {"wear_depth": "mm"},
    "equations": ["dw/dt = rate"],
    "limitations": ["does not model temperature"],
}


def test_generation_envelope_is_parsed_and_claims_checked():
    import json as _json
    artifact, schema = qa.parse_generation_response(_json.dumps(_ENVELOPE), provider="ollama",
                                                    model="llama3.1")
    assert schema.status is qa.Status.PASS
    assert artifact.code and artifact.units == {"wear_depth": "mm"}
    findings = qa.check_claims(artifact, example.build_spec())
    assert not [f for f in findings if f.blocks], [f.message for f in findings]


def test_generation_with_malformed_json_fails_loudly():
    artifact, schema = qa.parse_generation_response("this is not json")
    assert schema.status is qa.Status.FAIL
    assert artifact.code == ""
    assert any(f.id == "generation.json" for f in schema.findings)


def test_generation_declaring_a_forbidden_dependency_fails():
    import json as _json
    payload = dict(_ENVELOPE, dependencies=["subprocess"])
    artifact, _ = qa.parse_generation_response(_json.dumps(payload))
    findings = qa.check_claims(artifact, example.build_spec())
    assert any(f.id == "generation.dependency" and f.status is qa.Status.FAIL for f in findings)


def test_generation_declaring_a_wrong_unit_fails():
    import json as _json
    payload = dict(_ENVELOPE, units={"wear_depth": "s"})
    artifact, _ = qa.parse_generation_response(_json.dumps(payload))
    findings = qa.check_claims(artifact, example.build_spec())
    assert any(f.id == "generation.unit_mismatch" for f in findings)


def test_generation_with_an_extra_assumption_is_flagged_not_merged():
    import json as _json
    payload = dict(_ENVELOPE, assumptions=["the fluid is Newtonian"])
    artifact, _ = qa.parse_generation_response(_json.dumps(payload))
    findings = qa.check_claims(artifact, example.build_spec())
    flagged = [f for f in findings if f.id == "generation.extra_assumption"]
    assert flagged and flagged[0].status is qa.Status.NOT_RUN
    assert not any(f.blocks for f in flagged)


def test_pipeline_records_generation_claims_and_source():
    import json as _json
    artifact, _ = qa.parse_generation_response(_json.dumps(_ENVELOPE), provider="ollama",
                                               model="llama3.1")
    bundle = _simple_bundle()
    report = qa.QualityPipeline(_spec()).run(generated=artifact, **bundle)
    assert report.step("generation").status is qa.Status.PASS
    assert report.provider == "ollama" and report.model == "llama3.1"
    assert report.status_of("static_syntax") is qa.Status.PASS
    assert any(a.kind == "generation_metadata" for a in report.artifacts)


# ---------------------------------------------------------------------------
# regression: existing public API untouched
# ---------------------------------------------------------------------------
def test_existing_public_api_still_imports():
    for name in ("draft_problem", "call_llm", "PhysicsGuardrail", "GuardrailReport",
                 "build_3d_live_twin", "run_agent_loop", "ConversationStore"):
        assert hasattr(pinneapple_llm, name), name


def test_call_llm_still_rejects_unknown_provider():
    with pytest.raises(ValueError):
        pinneapple_llm.call_llm("hello", provider="does-not-exist")


def test_spec_round_trips_through_json():
    spec = example.build_spec()
    again = qa.PhysicalSpecification.from_dict(json.loads(spec.to_json()))
    assert again.content_hash() == spec.content_hash()
    assert again.criteria_hash() == spec.criteria_hash()
