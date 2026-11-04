"""Gallery groups (#1244): every entry is a bundled flow with a quick configuration, its example flow is the one the
tooling writes, and the committed examples/<group>/<entry>/ holds measured quick and full runs with a valid
manifest."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import pinneapple_flow as pf
from pinneapple_flow.examples import FLOWS, flows, with_quick
from pinneapple_flow.gallery import GROUPS, example_flow

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = [(g, e) for g, spec in GROUPS.items() for e in spec["entries"]]


def test_the_groups_hold_the_industrial_and_base_workflow_flows():
    ind = {s["flow"] for s in GROUPS["industrial"]["entries"].values()}
    assert {"reactor_sizing_packed_bed", "reactor_sizing_cstr", "lithography_opc", "wing_design",
            "ahmed_body_end_to_end"} <= ind
    base = {s["flow"] for s in GROUPS["base_workflows"]["entries"].values()}
    assert {"surrogate_modeling", "use_trained_model", "veriphysics_grade", "inverse_modeling",
            "equation_discovery", "model_benchmarking"} <= base


@pytest.mark.parametrize("group,entry", ENTRIES)
def test_each_entry_is_written_by_the_tooling_and_was_measured(group, entry):
    doc = example_flow(group, entry)
    assert doc["meta"]["example"] == group and doc["id"] == f"example.{group}.{entry}"
    committed = json.loads((FLOWS / group / f"{entry}.json").read_text())
    assert committed == doc                                   # regenerated, not edited by hand
    for d in (doc, with_quick(doc)):
        assert [i for i in pf.validate_flow(d) if i.level == "error"] == []
    assert any(f["name"] == f"{group}/{entry}" for f in flows())
    base = ROOT / "examples" / group / entry
    res = json.loads((base / "results" / "result.json").read_text())
    assert res["flow"] == GROUPS[group]["entries"][entry]["flow"]
    for conf in ("quick", "full"):
        run = res["runs"][conf]
        assert run["status"] == "done" and run["seconds"] > 0 and run["commit"] and run["metrics"]
    from pinneapple_lab import manifest as M
    m = M.load(str(base))
    assert M.validate(m, str(base)) == []
    assert {"run_seconds_quick", "run_seconds_full"} <= {x["name"] for x in m.metrics}


def test_a_quick_configuration_names_nodes_the_flow_has(monkeypatch):
    from pinneapple_flow import gallery
    monkeypatch.setitem(gallery.GROUPS["industrial"]["entries"], "bad",
                        {"flow": "wing_design", "quick": {"nope": {"maxiter": 1}}})
    with pytest.raises(KeyError, match="nope"):
        example_flow("industrial", "bad")
