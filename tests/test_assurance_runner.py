from pinneapple_analysis.verification.assurance_runner import run_artifact_assurance
from pinneapple_analysis.verification.assurance_policy import RequiredCheck
from pinneapple_analysis.verification.pipeline_assurance import Artifact, Check, Status


def test_pipeline_runner_blocks_undeclared_solver_evidence():
    cad = Artifact("cad", "geometry", metadata={"length_unit": "mm", "coordinate_frame": "world"})
    mesh = Artifact("mesh", "mesh", parents=("cad",), metadata={
        "length_unit": "mm", "coordinate_frame": "world",
        "node_count": 20, "cell_count": 10,
    })
    report = run_artifact_assurance(
        [cad, mesh],
        [RequiredCheck("geometry", "length_unit_declared"),
         RequiredCheck("mesh", "mesh_counts"),
         RequiredCheck("solver", "independent_residual")],
    )
    assert report.assessment["verdict"] == "pass"
    assert report.decision_gate["decision"] == "blocked"
    assert ("solver", "independent_residual") in report.decision_gate["missing"]


def test_pipeline_runner_detects_unit_mismatch():
    cad = Artifact("cad", "geometry", metadata={"length_unit": "mm", "coordinate_frame": "world"})
    mesh = Artifact("mesh", "mesh", parents=("cad",), metadata={
        "length_unit": "m", "coordinate_frame": "world",
        "node_count": 20, "cell_count": 10,
    })
    report = run_artifact_assurance([cad, mesh], [RequiredCheck("mesh", "lineage:length_unit")])
    assert report.decision_gate["decision"] == "blocked"
    assert report.diagnosis["first_observed_failure_stage"] == "mesh"


def test_pipeline_runner_can_be_eligible_not_approved():
    cad = Artifact("cad", "geometry", metadata={"length_unit": "mm"})
    report = run_artifact_assurance([cad], [RequiredCheck("geometry", "length_unit_declared")])
    assert report.decision_gate["decision"] == "eligible_for_review"
    assert report.decision_gate["review_required"] is True
