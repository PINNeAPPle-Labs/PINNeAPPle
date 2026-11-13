"""The bundle harness: same data, same budget, same metric for a PINN and an FNO, with the gates attached."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_arena.bundle_bench import BenchConfig, prepare, run_benchmark, run_model
from pinneapple_data.schema import AxisSpec, Field, Physics, PhysicsDataBundle

TINY = BenchConfig(field="u", epochs=3, width=8, modes=4, layers=1, hidden=(8,), batch_size=4, train_fraction=0.75)


def travelling_wave(nt: int = 12, nx: int = 16) -> PhysicsDataBundle:
    x = np.linspace(0.0, 1.0, nx)
    t = np.linspace(0.0, 1.0, nt)
    u = (1.0 + 0.2 * np.sin(2 * np.pi * (x[None, :] - 0.3 * t[:, None]))).astype(np.float32)
    return PhysicsDataBundle.from_arrays(
        {"field:u": u, "axis:t": t, "axis:x": x},
        fields=[Field.from_array("u", u, unit="1", axes=("t", "x"))],
        axes=[AxisSpec("t", "s"), AxisSpec("x", "m")],
        physics=Physics(governing="u_t + c u_x = 0"), source="synthetic wave", license="CC0-1.0",
        citation="synthetic", kind="synthetic")


def test_prepare_splits_along_time_not_at_random():
    data = prepare(travelling_wave(), TINY)
    assert list(data.train) == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert list(data.test) == [9, 10, 11]
    assert data.nt == 12 and data.nx == 16


def test_prepare_refuses_a_field_it_cannot_pose():
    x = np.linspace(0, 1, 8)
    t = np.linspace(0, 1, 6)
    field = np.zeros((6, 8, 2), dtype=np.float32)
    bundle = PhysicsDataBundle.from_arrays(
        {"field:v": field, "axis:t": t, "axis:x": x},
        fields=[Field.from_array("v", field, axes=("t", "x"), components=("a", "b"))],
        axes=[AxisSpec("t", "s"), AxisSpec("x", "m")], source="vector", license="CC0-1.0", kind="synthetic")
    with pytest.raises(NotImplementedError, match="handles \\(time, space\\)"):
        prepare(bundle, BenchConfig(field="v"))


def test_both_models_run_and_are_scored_the_same_way():
    bundle = travelling_wave()
    report = run_benchmark(bundle, TINY)
    assert set(report.results) == {"pinn", "fno"}
    for name, result in report.results.items():
        assert result.parameters > 0, name
        assert result.epochs_run == TINY.epochs
        assert len(result.per_sample) == report.n_test
        assert np.isfinite(result.rel_l2)
    assert report.leader() in report.results
    assert report.n_train == 9 and report.n_test == 3


def test_the_reference_gate_measures_the_prediction_against_the_data():
    bundle = travelling_wave()
    report = run_benchmark(bundle, TINY)
    reference = next(g for g in report.gates["gates"] if g["check_id"] == "reference")
    leader = report.results[report.leader()]
    assert reference["value"] == pytest.approx(leader.rel_l2, rel=1e-5)   # float32 field vs float64 metric
    assert reference["status"] in ("passed", "failed")
    integrity = next(g for g in report.gates["gates"] if g["check_id"] == "integrity")
    assert integrity["status"] == "passed"


def test_the_markdown_report_says_what_was_run():
    text = run_benchmark(travelling_wave(), TINY).to_markdown()
    assert "| model | parameters | rel-L2" in text
    assert "Leader" in text and "Verification gates" in text


def test_a_ledger_seals_each_model_score():
    from pinneapple_veriphysics.ledger import EvidenceLedger

    ledger = EvidenceLedger()
    run_benchmark(travelling_wave(), TINY, ledger=ledger)
    assert ledger.status("benchmark/fno") in ("passed", "failed")
    entry = ledger.latest("benchmark/fno")
    assert entry is not None and entry.valid
    assert isinstance(entry.outputs["rel_l2"], float)   # a measured number, not a claim
    assert entry.outputs["field"] == "u"


def test_run_model_rejects_an_unknown_family():
    with pytest.raises(ValueError, match="unknown model"):
        run_model(travelling_wave(), TINY, "transformer")


def test_the_tolerance_decides_ok():
    strict = BenchConfig(field="u", epochs=1, width=8, modes=4, layers=1, hidden=(8,), tolerance=-1.0)
    assert run_benchmark(travelling_wave(), strict).ok is False
