from pinneapple_analysis.verification.brainstorm_mvp import (
    falsify_scalar_property, calibration_coverage, ood_envelope,
    independent_design_check, sensor_drift, evidence_dossier,
)
from pinneapple_analysis.verification.pipeline_assurance import Status


def test_falsification_finds_counterexample():
    c = falsify_scalar_property("positivity", "training", [-1, 0, 1],
                                lambda x: x, lambda x, y: y >= 0,
                                assumption="Nonnegative solution applies to this PDE")
    assert c.status == Status.FAIL
    assert c.evidence["counterexample_count"] == 1


def test_ood_detects_extrapolation():
    c = ood_envelope("ood", [[0.5], [2.0]], [0.0], [1.0])
    assert c.status == Status.FAIL
    assert c.evidence["outside_indices"] == [1]


def test_calibration_and_design():
    c = calibration_coverage("intervals", [0, 0], [-1, -1], [1, 1],
                             [0, 0], nominal=0.9, tolerance=0.11)
    assert c.status == Status.PASS
    assert independent_design_check("design", 1, 3, atol=0.1, rtol=0).status == Status.FAIL


def test_sensor_drift_and_dossier():
    c = sensor_drift("drift", [0, 0, 5], baseline_mean=0, baseline_std=1)
    assert c.status == Status.FAIL
    report = evidence_dossier([c], context_of_use="experimental digital twin")
    assert report["decision"] == "blocked"
    assert report["counts"]["fail"] == 1
