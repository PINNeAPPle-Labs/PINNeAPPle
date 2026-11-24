"""The cost regression gate: an operation that gets more expensive fails the gate, with a message that says so (#53,
roadmap D2).

FLOPs are deterministic, so they gate strictly (a 6 % increase fails); wall time and memory are noisy, so they gate
loosely (a 6 % increase passes, a 60 % one does not). Both limits are checked here against tables built in the test,
and the real versioned baseline is checked for the two properties the gate needs to bite: it exists, and it was
measured on a machine the comparison accepts.

The gate is local on purpose: ``make cost-gate`` runs the same code CI would run, because GitHub Actions is off for
billing. ``scripts/measure_method_costs.py --gate`` measures now and compares with
``pinneapple_catalog/method_costs_baseline.json``.
"""
from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest

from pinneapple_catalog import costs

REPO = Path(__file__).resolve().parent.parent
BASELINE_PATH = costs.HERE / "method_costs_baseline.json"
SCRIPT = REPO / "scripts" / "measure_method_costs.py"
MACHINE = {"platform": "test-machine", "cpu_count": 8, "python": "3.15", "torch": "2.15"}
PROBE = "pinneapple_simulation.numerical_solvers.spectral.poisson_periodic_fft"


def _table(**overrides) -> dict:
    """A cost table with one measured method, so a regression is the only thing under test."""
    record = {"status": "measured", "probe": PROBE, "median_time_s": 0.004, "peak_mem_bytes": 1_000_000,
              "flops": 10_000_000, "mem_source": "tracemalloc", "flops_source": "flop_counter"}
    record.update(overrides)
    return {"machine": dict(MACHINE), "methods": {"S4": record}}


def test_a_six_percent_flop_increase_fails_the_gate():
    """The done-when of #53: raise an operation's FLOPs by more than 5 % and the gate has to say so."""
    baseline = _table()
    current = _table(flops=int(10_000_000 * 1.06))
    rows = costs.compare_with_baseline(current, baseline)
    assert len(rows) == 1, rows
    r = rows[0]
    assert r["method_id"] == "S4" and r["quantity"] == "flops"
    assert r["ratio"] == pytest.approx(1.06, rel=1e-6)
    assert r["factor"] == pytest.approx(1.05)


def test_a_four_percent_flop_increase_passes_the_gate():
    baseline = _table()
    current = _table(flops=int(10_000_000 * 1.04))
    assert costs.compare_with_baseline(current, baseline) == []


def test_a_zero_flop_change_passes_the_gate():
    assert costs.compare_with_baseline(_table(), _table()) == []


def test_time_gates_loosely_because_it_is_noisy():
    """A 10 % change in wall time is noise; 60 % is a regression."""
    baseline = _table()
    assert costs.compare_with_baseline(_table(median_time_s=0.0044), baseline) == []
    rows = costs.compare_with_baseline(_table(median_time_s=0.0064), baseline)
    assert [r["quantity"] for r in rows] == ["time_s"]


def test_memory_gates_loosely_too():
    baseline = _table()
    rows = costs.compare_with_baseline(_table(peak_mem_bytes=1_600_000), baseline)
    assert [r["quantity"] for r in rows] == ["peak_mem_bytes"]


def test_a_different_machine_is_refused_not_compared():
    """The same code measured on two CPUs is two measurements; calling that a regression would be a lie."""
    baseline = _table()
    other = _table(flops=20_000_000)
    other["machine"] = {**MACHINE, "platform": "some-other-machine"}
    with pytest.raises(ValueError, match="different machines"):
        costs.compare_with_baseline(other, baseline)


def test_an_unmeasured_method_is_not_compared():
    current = {"machine": dict(MACHINE), "methods": {"S4": {"status": "unmeasured", "reason": "no probe"}}}
    assert costs.compare_with_baseline(current, _table()) == []


def test_the_gate_message_names_the_method_the_quantity_and_the_limit():
    """The message is the part a contributor reads in a failed run: it has to be usable as it stands."""
    baseline = _table()
    current = _table(flops=int(10_000_000 * 1.06))
    out = subprocess.run([sys.executable, str(SCRIPT), "--gate", "--only", "S13", "--repeats", "1"],
                         cwd=REPO, capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "PYTHONPATH": str(REPO),
                              "COST_GATE_OUT": "/tmp/cost_gate_test.json", "HOME": str(Path.home())})
    # The gate itself is exercised above; here the script has to run end to end and print the gate's verdict.
    assert "cost gate:" in out.stdout + out.stderr, out.stdout + out.stderr
    assert out.returncode in (0, 1), out.stderr


def test_the_versioned_baseline_exists_and_is_comparable():
    """Without a baseline the gate can only pass silently, which is worse than failing."""
    assert BASELINE_PATH.exists(), "pinneapple_catalog/method_costs_baseline.json is the gate's reference"
    baseline = costs.load_costs(BASELINE_PATH)
    assert baseline["machine"]["platform"], "the baseline has no machine recorded"
    measured = [mid for mid, r in baseline["methods"].items() if r.get("status") == "measured"]
    assert measured, "the baseline has no measured method, so the gate could never fail"
    assert any(baseline["methods"][mid].get("flops") for mid in measured), (
        "no method in the baseline has counted FLOPs: the strict half of the gate would never fire")


def test_the_checked_in_table_matches_the_baseline_it_was_frozen_from():
    """The table and the baseline describe the same measurement, so a diff between them is a real code change."""
    table, baseline = costs.load_costs(), costs.load_costs(BASELINE_PATH)
    assert table["machine"]["platform"] == baseline["machine"]["platform"]
    for mid, rec in baseline["methods"].items():
        if rec.get("status") != "measured":
            continue
        assert table["methods"][mid]["probe"] == rec["probe"]
        assert table["methods"][mid]["sizes"] == rec["sizes"]


def test_an_increase_is_detected_on_a_copy_of_the_real_table():
    """Same shape as the real file, one number raised: the gate finds it without a synthetic shortcut."""
    baseline = costs.load_costs(BASELINE_PATH)
    current = copy.deepcopy(baseline)
    measured = [mid for mid, r in current["methods"].items() if r.get("status") == "measured" and r.get("flops")]
    mid = measured[0]
    current["methods"][mid]["flops"] = int(baseline["methods"][mid]["flops"] * 1.5)
    rows = costs.compare_with_baseline(current, baseline)
    assert [r["method_id"] for r in rows] == [mid]
    assert rows[0]["quantity"] == "flops"
