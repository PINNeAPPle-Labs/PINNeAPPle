"""The measured cost table is complete and carries everything a number needs to be trusted (#52, roadmap D1).

These tests do not measure anything (that is ``scripts/measure_method_costs.py``): they check the file that measurement
wrote -- that every method of the catalog (solvers and training methods) is in it, that a measured record has the
machine, the probe, the size axis, the per-size numbers and an exponent with its confidence interval, and that an
unmeasured method says *why* instead of quietly missing.
"""
from __future__ import annotations

import pytest

from pinneapple_analysis.cost import CostRecord, measure, scaling_study
from pinneapple_catalog import costs
from pinneapple_catalog.methods import list_methods
from pinneapple_core.errors import UnknownName

RECORDS = costs.all_costs()
DOC = costs.load_costs()
SCOPE = [m for m in list_methods() if m.category in ("solver", "training")]


def test_the_table_covers_every_solver_and_training_method():
    missing = [m.id for m in SCOPE if m.id not in RECORDS]
    assert not missing, f"methods with no entry in method_costs.json: {missing}"


def test_every_unmeasured_method_states_a_reason():
    blank = [m.id for m in SCOPE
             if RECORDS.get(m.id, {}).get("status") == "unmeasured" and not RECORDS[m.id].get("reason", "").strip()]
    assert not blank, f"unmeasured without a reason: {blank}"


def test_the_machine_is_recorded_with_the_numbers():
    """A wall time without its machine cannot be compared with anything, so the file has to carry it."""
    machine = DOC.get("machine", {})
    for key in ("platform", "cpu_count", "python", "torch"):
        assert key in machine, f"the cost table does not record the machine's {key!r}"


def test_the_file_says_how_it_was_measured():
    assert "scripts/measure_method_costs.py" in DOC.get("how", "")
    assert DOC.get("measured_at"), "the table does not say when it was measured"
    assert DOC.get("commit"), "the table does not say which commit it measured"


@pytest.mark.parametrize("method_id", sorted(mid for mid, r in RECORDS.items() if r.get("status") == "measured"))
def test_a_measured_record_has_time_memory_and_an_exponent(method_id):
    rec = RECORDS[method_id]
    assert rec["probe"] and rec["size_axis"], "the record must name what ran and what n means"
    assert len(rec["per_size"]) >= 3, "a scaling exponent needs at least three sizes"
    assert rec["exponent_ci"][0] <= rec["exponent"] <= rec["exponent_ci"][1]
    assert rec["r2"] == rec["fits"]["time"]["r2"]
    for row in rec["per_size"]:
        assert row["time_s"] > 0 and row["n_runs"] >= 1
        assert "mem_bytes" in row and "flops" in row, "an honest gap is null, never a guess"
    assert rec["mem_source"], "the record must say how memory was sampled (rss or tracemalloc)"
    # FLOPs: either counted by torch, or absent -- and when absent the field says so
    if rec["flops"] is None:
        assert rec["flops_source"] == ""
    else:
        assert rec["flops_source"] in ("flop_counter", "dispatch_counter", "profiler")


def test_at_least_one_method_has_counted_flops_so_the_gate_can_bite():
    """#53's gate is strict on FLOPs; with no counted FLOPs anywhere the strict half would never fire."""
    counted = [mid for mid, r in RECORDS.items() if r.get("status") == "measured" and r.get("flops")]
    assert counted, "no method in the table has a counted FLOP number"


def test_a_method_without_a_record_gets_an_actionable_error():
    with pytest.raises(UnknownName, match="no cost record"):
        costs.method_cost("P1.1")


def test_coverage_has_one_row_per_method_and_names_the_status():
    rows = costs.coverage()
    assert len(rows) == len(SCOPE)
    assert {r["status"] for r in rows} <= {"measured", "unmeasured", "missing"}
    assert not [r for r in rows if r["status"] == "missing"], "method_costs.json is stale: rerun the measurement"


def test_a_probe_runs_live_and_returns_a_record():
    """One probe from the table, measured now: the table's numbers have to be reproducible by the same code path."""
    probe = costs.PROBES["S19"]
    rec = measure(probe.build(2 ** 16), repeats=2, memory=False)
    assert isinstance(rec, CostRecord) and rec.time_s > 0


def test_the_scaling_machinery_reads_an_exponent_off_a_known_growth():
    """The exponent in the table comes from this fit; on a hundred-fold range it recovers a linear cost."""
    study = scaling_study(lambda n: (lambda: sum(range(int(n)))), [10 ** 5, 10 ** 6, 10 ** 7],
                          name="synthetic linear", repeats=3, declared="n")
    assert 0.6 < study.fits["time"].exponent < 1.4
