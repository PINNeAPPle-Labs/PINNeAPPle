"""Verification adapter contracts."""
import pytest

from pinneapple_physics.engineering_trace import EngineeringTrace, Requirement
from pinneapple_physics.engineering_verification import record_check_result, release_gate


def make_trace():
    trace = EngineeringTrace()
    trace.add_requirement(Requirement("pressure", "Max pressure", "spec#1", "Pa", upper_bound=100))
    return trace


def test_verified_result_opens_release_gate():
    trace = make_trace()
    assert not release_gate(trace)["approved"]
    record_check_result(
        trace, requirement_id="pressure", evidence_id="run1", artifact_uri="runs/1.json",
        result={"check": "pressure_limit", "passed": True, "value": 95.0, "unit": "Pa"},
    )
    assert release_gate(trace)["approved"]


@pytest.mark.parametrize("result", [
    {"check": "pressure_limit", "passed": "false", "value": 95, "unit": "Pa"},
    {"check": "pressure_limit", "value": 95, "unit": "Pa"},
    {"check": "", "passed": True},
    {"check": "pressure_limit", "passed": True, "value": float("nan"), "unit": "Pa"},
    {"check": "pressure_limit", "passed": True, "value": True, "unit": "Pa"},
])
def test_malformed_results_fail_closed(result):
    trace = make_trace()
    with pytest.raises(ValueError):
        record_check_result(
            trace, requirement_id="pressure", evidence_id="run1",
            artifact_uri="runs/1.json", result=result,
        )
    assert trace.status("pressure") == "unverified"
    assert not release_gate(trace)["approved"]


def test_failing_check_blocks_release():
    trace = make_trace()
    record_check_result(
        trace, requirement_id="pressure", evidence_id="run1", artifact_uri="runs/1.json",
        result={"check": "pressure_limit", "passed": False, "value": 120, "unit": "Pa"},
    )
    assert release_gate(trace)["failed"] == ["pressure"]
    assert not release_gate(trace)["approved"]


def test_empty_trace_cannot_be_approved():
    assert not release_gate(EngineeringTrace())["approved"]
