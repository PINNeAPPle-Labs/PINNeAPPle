from pinneapple_analysis.verification.physics_ai_checks import (
    field_parity, synthetic_recovery, inverse_identifiability,
    holdout_leakage, evaluate_physical_limits, temporal_composition,
)
from pinneapple_analysis.verification.pipeline_assurance import Status


def test_field_parity_detects_export_mismatch():
    assert field_parity("onnx", "export", [1, 2], [1, 2]).status == Status.PASS
    assert field_parity("onnx", "export", [1, 2], [1, 3]).status == Status.FAIL
    assert field_parity("onnx", "export", [1], []).status == Status.INCONCLUSIVE


def test_group_leakage():
    assert holdout_leakage("geometry_split", ["g1"], ["g2"]).status == Status.PASS
    assert holdout_leakage("geometry_split", ["g1"], ["g1"]).status == Status.FAIL


def test_physical_limits_require_assumptions():
    assert evaluate_physical_limits("temp", "solver", [1], lower=0, assumptions=()).status == Status.INCONCLUSIVE
    assert evaluate_physical_limits("temp", "solver", [-1], lower=0, assumptions=("Absolute temperature",)).status == Status.FAIL


def test_inverse_recovery_and_local_rank():
    assert synthetic_recovery("recovery", [1, 2], [1, 2]).status == Status.PASS
    assert inverse_identifiability("rank", [[1, 0], [0, 1]]).status == Status.PASS
    assert inverse_identifiability("rank", [[1, 1], [2, 2]]).status == Status.FAIL


def test_temporal_semigroup():
    assert temporal_composition("semigroup", [1, 2], [1, 2]).status == Status.PASS
    assert temporal_composition("semigroup", [1, 2], [2, 3]).status == Status.FAIL
