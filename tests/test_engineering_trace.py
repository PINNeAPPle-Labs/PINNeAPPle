"""Contract tests for engineering requirements and evidence."""
import pytest

from pinneapple_physics.engineering_trace import EngineeringTrace, Evidence, Requirement


def test_traceability_and_report():
    trace = EngineeringTrace()
    trace.add_requirement(Requirement("R1", "Pressure below limit", "spec.pdf#p2", "Pa", upper_bound=100))
    assert trace.status("R1") == "unverified"
    trace.add_evidence(Evidence("E1", "R1", "runs/1/result.json", "pressure_check", True, 95, "Pa"))
    assert trace.status("R1") == "verified"
    assert trace.report()["requirements"][0]["evidence_ids"] == ["E1"]
    trace.add_evidence(Evidence("E2", "R1", "runs/2/result.json", "pressure_check", False, 120, "Pa"))
    assert trace.status("R1") == "failed"


def test_reject_invalid_bounds_and_units():
    with pytest.raises(ValueError):
        Requirement("R1", "Pressure", "spec", "Pa", 200, 100)
    trace = EngineeringTrace()
    trace.add_requirement(Requirement("R1", "Pressure", "spec", "Pa", upper_bound=100))
    with pytest.raises(ValueError, match="Unit mismatch"):
        trace.add_evidence(Evidence("E1", "R1", "run", "check", True, 95, "bar"))
    with pytest.raises(ValueError, match="cannot pass"):
        trace.add_evidence(Evidence("E1", "R1", "run", "check", True, 120, "Pa"))
    with pytest.raises(ValueError, match="Unknown requirement"):
        trace.add_evidence(Evidence("E1", "missing", "run", "check", True))


def test_duplicates_and_missing_requirements():
    trace = EngineeringTrace()
    req = Requirement("R1", "Constraint", "spec")
    trace.add_requirement(req)
    with pytest.raises(ValueError, match="Duplicate requirement"):
        trace.add_requirement(req)
    with pytest.raises(KeyError):
        trace.status("missing")
