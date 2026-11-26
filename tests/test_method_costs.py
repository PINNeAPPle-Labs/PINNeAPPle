"""The measured cost table is complete and carries everything a number needs to be trusted (#52, roadmap D1).

These tests do not measure anything (that is ``scripts/measure_method_costs.py``): they check the file that measurement
wrote. Every entry of the catalog has a row -- a measured one carries the machine, the probe, the size axis, the
per-size numbers and an exponent with its confidence interval; an unmeasured one carries a blocker from the closed set
and the specific reason, never "not enough time". The gate itself is exercised here too: a missing row, a table
without its machine and an unexplained method all have to fail it, or the completeness it reports is empty.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from pinneapple_analysis.cost import CostRecord, measure, scaling_study
from pinneapple_catalog import costs
from pinneapple_catalog.methods import list_methods
from pinneapple_core.errors import UnknownName

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "measure_method_costs.py"
DOC_PATH = REPO / "docs" / "dev" / "METHOD_COSTS.md"

RECORDS = costs.all_costs()
DOC = costs.load_costs()
CATALOG = list_methods()
UNMEASURED_IDS = [mid for mid, r in RECORDS.items() if r.get("status") == "unmeasured"]


def _gate_module():
    """``scripts/measure_method_costs.py`` as a module, so the gate is tested where it lives and not through a copy."""
    spec = importlib.util.spec_from_file_location("measure_method_costs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = _gate_module()


def _table(tmp_path: Path, table: dict) -> Path:
    path = tmp_path / "method_costs.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- coverage
def test_the_table_covers_every_catalog_entry():
    """Every id of ``methods.METHODS`` -- solvers, training methods, equations and problems -- has a row."""
    missing = [m.id for m in CATALOG if m.id not in RECORDS]
    assert not missing, f"catalog entries with no row in method_costs.json: {missing}"


def test_every_catalog_entry_has_exactly_one_coverage_row():
    rows = costs.coverage()
    assert [r["id"] for r in rows] == [m.id for m in CATALOG]
    assert {r["status"] for r in rows} <= {"measured", "unmeasured", "missing"}
    assert not [r for r in rows if r["status"] == "missing"], (
        "method_costs.json is stale or unexplained: rerun python scripts/measure_method_costs.py")


def test_an_unmeasured_entry_names_a_blocker_from_the_closed_set():
    """The blocker is what makes the gap countable; a free-text motive alone cannot be counted or rejected."""
    for mid in UNMEASURED_IDS:
        rec = RECORDS[mid]
        assert rec.get("blocker") in costs.BLOCKERS, f"{mid}: blocker {rec.get('blocker')!r} is not in BLOCKERS"
        assert str(rec.get("reason") or "").strip(), f"{mid}: an unmeasured method must say why"


def test_every_blocker_in_the_vocabulary_is_used_by_some_entry():
    """A motive nobody hit is a motive nobody checked: the vocabulary is the set of gaps this table really has."""
    used = {RECORDS[mid]["blocker"] for mid in UNMEASURED_IDS}
    assert set(costs.BLOCKERS) == used, f"blockers never used: {sorted(set(costs.BLOCKERS) - used)}"


def test_an_equation_or_a_problem_is_a_definition_not_a_gap():
    """They are descriptions, not executables; the row says so instead of pretending a probe could exist."""
    row = next(r for r in costs.coverage() if r["category"] == "equation")
    assert row["status"] == "unmeasured" and row["blocker"] == "definition"


# --------------------------------------------------------------------------- the numbers
def test_the_machine_is_recorded_with_the_numbers():
    """A wall time without its machine cannot be compared with anything, so the file has to carry it."""
    machine = DOC.get("machine", {})
    for key in ("platform", "cpu", "cpu_count", "python", "torch"):
        assert key in machine, f"the cost table does not record the machine's {key!r}"
    assert machine["cpu"], "the machine's CPU is recorded as an empty string"


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


def test_a_method_without_a_record_gets_an_actionable_error(tmp_path):
    """Every catalog entry has a row now, so the missing-record path is exercised on a table that lost one."""
    empty = tmp_path / "method_costs.json"
    empty.write_text('{"methods": {}}', encoding="utf-8")
    with pytest.raises(UnknownName, match="no cost record"):
        costs.method_cost("S4", empty)


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


# --------------------------------------------------------------------------- the gate
def test_the_gate_passes_on_the_committed_table():
    assert GATE._check(costs.COSTS_PATH) == 0


def test_the_gate_fails_when_a_catalog_method_has_no_row(tmp_path, capsys):
    table = copy.deepcopy(DOC)
    del table["methods"]["S4"]
    assert GATE._check(_table(tmp_path, table)) == 1
    assert "GAP      S4" in capsys.readouterr().out


def test_the_gate_fails_when_the_machine_is_not_recorded(tmp_path, capsys):
    """No machine, no comparable number: the file is the place it is recorded, so an empty one has to fail."""
    table = copy.deepcopy(DOC)
    table["machine"] = {k: "" for k in table["machine"]}
    assert GATE._check(_table(tmp_path, table)) == 1
    assert "NO MACHINE" in capsys.readouterr().out


def test_the_gate_fails_on_an_unmeasured_row_without_a_blocker(tmp_path):
    table = copy.deepcopy(DOC)
    table["methods"][UNMEASURED_IDS[0]]["blocker"] = ""
    assert GATE._check(_table(tmp_path, table)) == 1


def test_the_gate_fails_on_a_blocker_outside_the_vocabulary(tmp_path):
    table = copy.deepcopy(DOC)
    table["methods"][UNMEASURED_IDS[0]]["blocker"] = "no_time"
    assert GATE._check(_table(tmp_path, table)) == 1


def test_the_gate_fails_on_an_unmeasured_row_without_a_reason(tmp_path):
    table = copy.deepcopy(DOC)
    table["methods"][UNMEASURED_IDS[0]]["reason"] = "  "
    assert GATE._check(_table(tmp_path, table)) == 1


# --------------------------------------------------------------------------- the doc
def test_the_doc_lists_every_catalog_entry():
    """One line per method, linked to the JSON, is the human side of the acceptance; stale rows are worse than none."""
    text = DOC_PATH.read_text(encoding="utf-8")
    for m in CATALOG:
        assert f"| **{m.id}** |" in text, f"{m.id} has no line in docs/dev/METHOD_COSTS.md"


def test_the_doc_matches_the_table():
    """The doc is generated: ``--docs-check`` is the same comparison this asserts, so it cannot drift from the table."""
    assert DOC_PATH.read_text(encoding="utf-8") == GATE.render_doc(DOC, costs.coverage()), (
        "docs/dev/METHOD_COSTS.md is out of date: run python scripts/measure_method_costs.py --docs")
