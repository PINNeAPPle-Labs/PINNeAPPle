import pytest
from pinneapple_atlas import Evidence, Capability, compare_capabilities, build_graph


def test_missing_local_is_not_absent():
    public = Capability("fno", "Fourier Neural Operator", "public", "implemented",
                        (Evidence("https://github.com/neuraloperator/neuraloperator", "code", "NeuralOperator", "public"),))
    coverage = compare_capabilities([public])
    assert coverage[0].classification == "not_assessed"
    graph = build_graph([public])
    assert len(graph["nodes"]) == 2
    assert graph["edges"][0]["relation"] == "supported_by"


def test_implementation_requires_evidence():
    with pytest.raises(ValueError):
        Capability("fno", "FNO", "pinneapple", "validated")


def test_planned_does_not_count_as_implemented():
    local = Capability("fno", "FNO", "pinneapple", "planned")
    assert compare_capabilities([local])[0].classification == "planned"


def test_evidence_origin_must_match():
    with pytest.raises(ValueError):
        Capability("fno", "FNO", "pinneapple", "partial",
                   (Evidence("https://example.org", "paper", "Example", "public"),))
