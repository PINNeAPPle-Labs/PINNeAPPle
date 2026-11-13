from pinneapple_analysis.verification.pipeline_assurance import Artifact, Check, Status
from pinneapple_analysis.verification.diagnostic_engine import localize, propose_repairs, compare_intervention


def test_localization_does_not_claim_cause():
    artifacts = [Artifact("cad", "geometry"), Artifact("mesh", "mesh", ("cad",))]
    checks = [Check("unit_check", "geometry", Status.PASS),
              Check("mesh_quality", "mesh", Status.FAIL, suggested_action="Remesh")]
    result = localize(artifacts, checks, ["boundary_localized"])
    assert result["first_observed_failure_stage"] == "mesh"
    assert result["root_cause_confirmed"] is False
    assert result["candidate_causes"]
    repairs = propose_repairs(checks)
    assert repairs[0].requires_approval
    assert not repairs[0].permitted_to_execute


def test_intervention_detects_regression_and_missing():
    before = [Check("a", "solver", Status.FAIL), Check("b", "dataset", Status.PASS)]
    after = [Check("a", "solver", Status.PASS), Check("b", "dataset", Status.FAIL)]
    result = compare_intervention(before, after, changed_artifacts=["mesh-v2"])
    assert result["fixed_checks"] == ["solver:a"]
    assert result["regressions"] == ["dataset:b"]
    assert result["eligible_for_approval"] is False
    assert compare_intervention(before, after[:1], changed_artifacts=["mesh-v2"])["missing_rechecks"]
