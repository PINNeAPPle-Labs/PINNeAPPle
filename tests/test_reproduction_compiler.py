"""Regression tests for evidence-gated paper-to-reproduction compilation."""
from pinneapple_llm.reproduction import compile_reproduction, STAGES


def test_rejects_unsupported_numeric_claims():
    source = "A bracket uses a 70 GPa material and a DeepONet model."
    extracted = {
        "title": "Bracket", "domain": "solid mechanics",
        "quantities": [
            {"name": "young", "value": 70, "unit": "GPa", "evidence": "70 GPa"},
            {"name": "load", "value": 5000, "unit": "N", "evidence": "5000 N"},
        ],
        "methods": [{"name": "DeepONet", "evidence": "DeepONet"}],
        "missing_information": ["CAD source"],
    }
    result = compile_reproduction(source, extracted, source_type="paper")
    assert len(result["quantities"]) == 1
    assert len(result["rejected_claims"]) == 1
    assert result["exact_reproduction_ready"] is False
    assert "geometry_source" in result["missing_information"]
    assert [s["stage"] for s in result["stages"]] == list(STAGES)


def test_no_evidence_cannot_become_verified():
    result = compile_reproduction("Some model was used.", {
        "title": "Unknown", "domain": "unknown",
        "quantities": [{"name": "E", "value": 200, "unit": "GPa", "evidence": "steel"}],
        "methods": [], "missing_information": []})
    assert not result["quantities"]
    assert result["rejected_claims"]


def test_artifact_refs_are_only_metadata():
    source = "A 3D solid elasticity benchmark."
    fields = ("geometry_source", "geometry_dimensions", "governing_equations",
              "material_parameters", "boundary_conditions", "mesh_definition",
              "case_sampling", "split_protocol", "model_architecture",
              "feature_encoding", "optimizer", "loss", "training_schedule", "metrics")
    result = compile_reproduction(source, {
        "title": "Demo", "domain": "mechanics",
        "quantities": [], "methods": [], "missing_information": []},
        supplied_artifacts={f: f"artifact:{f}" for f in fields})
    assert result["exact_reproduction_ready"] is False
    assert all(s["status"] != "blocked" for s in result["stages"])
