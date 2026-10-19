"""Decision records and references: the registry is well formed, every heuristic rule cites it or says why not,
confidence goes down without references, LLM references are checked, records reach the store and the report."""
import json
from pathlib import Path

import pinneapple_flow as pf
from pinneapple_flow import references as R
from pinneapple_flow.decisions import (
    Decision,
    collect,
    edits_to_decisions,
    llm_decision,
    report_markdown,
    rule_decision,
    user_decision,
)
from pinneapple_flow.rules import RULES

ROOT = Path(__file__).resolve().parents[1]


def test_registry_entries_are_well_formed():
    reg = R.registry()
    assert len(reg) >= 40
    for key, entry in reg.items():
        assert R.problems(entry) == [], (key, R.problems(entry))
    for key in ("raissi2019pinn", "lu2021deeponet", "li2021fno", "pfaff2021meshgraphnet", "evans2010pde"):
        assert key in reg
    assert R.ref("raissi2019pinn")["doi"] == "10.1016/j.jcp.2018.10.045"


def test_every_rule_cites_the_registry_or_says_why_not():
    reg = R.registry()
    for r in RULES.values():
        assert r.cites or r.uncited, f"rule {r.id} has neither references nor an uncited reason"
        for k in r.cites:
            assert k in reg, (r.id, k)
        for _kind, path, _detail in r.evidence:
            assert (ROOT / path).exists(), (r.id, path)


def test_confidence_downgrade_without_reference():
    cited = rule_decision("method.fd", what="finite differences")
    assert not cited.no_reference and cited.label == "high" and cited.references[0]["status"] == "registry"
    bare = rule_decision("coef.template_default", what="alpha = 0.1")
    assert bare.no_reference and bare.confidence <= 0.35 and bare.label == "low"
    assert "validation" in bare.suggestion and bare.uncited_reason
    with_evidence = rule_decision("validation.threshold", what="2 %")       # uncited but backed by the library
    assert not with_evidence.no_reference and with_evidence.evidence


def test_llm_references_are_checked_never_trusted():
    d = llm_decision("ollama/m", topic="method", what="DeepONet", references=[
        "lu2021deeponet", {"key": "made_up_2031"}, {"title": "x", "doi": "not-a-doi"}, {"title": "y", "doi": "10.1/y"}])
    st = [r["status"] for r in d.references]
    assert st == ["registry", "unverified", "unverified", "unverified"]
    assert "not in the curated registry" in d.references[1]["note"]
    none = llm_decision("ollama/m", topic="route", what="route")
    assert none.no_reference and none.confidence <= 0.35


def test_user_edits_become_user_decisions():
    ds = edits_to_decisions([{"node": "eq", "param": "coefficients", "value": {"alpha": 2e-5},
                              "notes": "handbook", "references": ["carslaw1959heat"]},
                             {"node": "val", "param": "tolerance", "value": 0.05}])
    assert [d.decided_by["kind"] for d in ds] == ["user", "user"]
    assert ds[0].topic == "coefficients" and not ds[0].no_reference
    assert ds[1].no_reference and ds[1].label in ("low", "medium")
    u = user_decision("geo", "dimension", "3-D", 3)
    assert Decision.from_dict(json.loads(json.dumps(u.to_dict()))).what == "3-D"


def test_planner_records_decisions_with_references():
    res = pf.plan("transient heat in a 2D plate with fixed edges, alpha = 0.05", backend="heuristic")
    ds = res["proposals"][0]["flow"]["meta"]["decisions"]
    topics = {d["topic"] for d in ds}
    assert {"equations", "dimension", "bcs", "coefficients", "method", "validation", "route"} <= topics
    coef = next(d for d in ds if d["decided_by"]["id"] == "coef.from_text")
    assert coef["value"] == {"alpha": 0.05}
    assert any(d["no_reference"] for d in ds) and any(not d["no_reference"] for d in ds)
    md = report_markdown(ds)
    assert "no reference" in md and "https://doi.org/" in md


def test_llm_plan_decisions(monkeypatch):
    flow = {"name": "h", "nodes": [{"id": "geo", "type": "geometry.domain", "params": {"dimension": 1}},
                                   {"id": "eq", "type": "equations.system", "params": {"template": "heat"}}],
            "edges": [{"source": "geo", "source_port": "geometry", "target": "eq", "target_port": "geometry"}]}
    answer = {"proposals": [{"title": "t", "rationale": "r", "flow": flow, "decisions": [
        {"node": "eq", "topic": "equations", "what": "heat equation", "references": ["carslaw1959heat"]},
        {"node": "ghost", "topic": "weird", "what": "something", "references": [{"title": "invented", "doi": "10.5/z"}]}]}]}
    res = pf.plan("heat", backend="llm", provider="ollama", model="m", call=lambda p, **k: json.dumps(answer))
    ds = res["proposals"][0]["flow"]["meta"]["decisions"]
    assert ds[0]["decided_by"] == {"kind": "llm", "id": "ollama/m"} and ds[0]["references"][0]["status"] == "registry"
    assert ds[1]["node"] == "" and ds[1]["topic"] == "other" and ds[1]["references"][0]["status"] == "unverified"
    assert ds[-1]["topic"] == "route"


def test_runtime_decisions_store_and_report(tmp_path, monkeypatch):
    monkeypatch.setenv("PINNEAPPLE_FLOW_HOME", str(tmp_path / "home"))
    from pinneapple_flow.store import FlowStore
    flow = pf.Flow("dec", nodes=[
        {"id": "geo", "type": "geometry.domain", "params": {"dimension": 1, "t_range": [0, 0.2]}},
        {"id": "eq", "type": "equations.system", "params": {"template": "heat"}},
        {"id": "fd", "type": "solver.classical", "params": {"n": 21, "n_t": 5}},
        {"id": "val", "type": "validate.compare", "params": {"tolerance": 0.05}},
        {"id": "rep", "type": "export.report", "params": {"format": "markdown"}}],
        edges=[{"source": "geo", "source_port": "geometry", "target": "eq", "target_port": "geometry"},
               {"source": "eq", "source_port": "problem", "target": "fd", "target_port": "problem"},
               {"source": "fd", "source_port": "solution", "target": "val", "target_port": "candidates"},
               {"source": "eq", "source_port": "problem", "target": "val", "target_port": "problem"},
               {"source": "val", "source_port": "metrics", "target": "rep", "target_port": "items"}],
        meta={"user_edits": [{"node": "val", "param": "tolerance", "value": 0.05, "notes": "loose on purpose"}]})
    store = FlowStore(f"sqlite:///{tmp_path / 'f.sqlite'}")
    v = store.save_flow(flow)
    assert any(d["decided_by"]["kind"] == "user" for d in store.get_decisions(flow.id, version=v))
    r = pf.run(flow, workdir=tmp_path / "w", store=store)
    assert r.status == "done", {k: x["error"] for k, x in r.nodes.items()}
    fd = r.report("fd")["decisions"]
    assert fd and fd[0]["decided_by"]["id"] == "method.fd"
    val = r.report("val")["decisions"]
    assert any(d["decided_by"]["kind"] == "user" for d in val)
    assert store.get_decisions(flow.id, run_id=r.run_id)
    text = Path(r.output("rep", "report")["$artifact"]).read_text()
    assert "Decisions and references" in text and "leveque2007fd" not in text      # formatted, not raw keys
    assert "LeVeque" in text
    assert r.report("rep")["metrics"]["decisions"] >= 3
    all_ds = collect(flow, {k: x["report"] for k, x in r.nodes.items()})
    assert any(d["node"] == "fd" for d in all_ds)
