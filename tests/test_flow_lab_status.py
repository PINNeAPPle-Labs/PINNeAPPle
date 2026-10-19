"""Lab seal and "how to raise it" hooks: the letter is read from pinneapple_lab (curation tier today, the A-F seal
scale and curation.next_steps when the Lab provides them), never graded by the flow library."""
import json

import pytest

import pinneapple_flow as pf
from pinneapple_flow import lab_status as ls
from pinneapple_flow.decisions import RAISE_NODES, rule_decision, user_decision
from pinneapple_flow.service import FLOWS_DIR

ITEM = "flow_demo"


def _curation(tmp_path, **rec):
    record = {"tier": "C", "tier_name": "exploratory", "gaps": ["compare with an independent reference"],
              "headline": {"text": "PASS rel_l2 vs exact"}, **rec}
    (tmp_path / "curation.json").write_text(json.dumps({"generated": "now", "items": {ITEM: record}}))
    return str(tmp_path)


def test_seal_status_reads_the_lab_seal(tmp_path):
    lab = _curation(tmp_path)
    st = ls.seal_status(ITEM, lab)
    assert st["letter"] == "C" and st["reason"] == "PASS rel_l2 vs exact" and st["scale"] == "ABCDEF"
    assert st["sealed"] and st["source"] == "pinneapple_lab.seal"          # every letter is sealed (A-F)
    assert ls.seal_status("unknown", lab)["letter"] is None
    assert ls.seal_status(ITEM, str(tmp_path / "empty"))["reason"].startswith("not curated")


def test_e_and_f_show_the_failing_check_and_the_next_step(tmp_path):
    lab = _curation(tmp_path, tier="E", tier_name="known problems",
                    headline={"label": "mass balance 3% > 1%", "kind": "failing"},
                    next_steps=[{"type": "physics", "to": "D", "stage": "physics", "why": "a physics check fails",
                                 "why_pt": "", "short": "make the physics check pass", "short_pt": "", "actions": []}])
    st = ls.seal_status(ITEM, lab)
    assert (st["letter"], st["failing_check"]) == ("E", "mass balance 3% > 1%")
    assert st["hint"] == "D: make the physics check pass"


def test_next_steps_map_lab_actions_to_registered_nodes(tmp_path):
    act = {"id": "trust_card", "kind": "python", "target": "pinneapple.trust_card", "command": "pp.trust_card(...)",
           "text": "run the trust card", "text_pt": "", "params": {}}
    gci = {**act, "id": "gci", "target": "pinneapple_lab.uq.gci_check", "command": "gci_check(ctx, ...)"}
    cli = {**act, "id": "review", "kind": "cli", "target": "cli:pinneapple_lab review", "command": "review"}
    lab = _curation(tmp_path, next_steps=[
        {"type": "reference", "to": "B", "stage": "model", "why": "no independent reference", "why_pt": "",
         "short": "pass an independent check", "short_pt": "", "actions": [act, gci, cli]}])
    out = ls.next_steps(ITEM, lab)
    assert out["available"] and out["gaps"][0]["gap"] == "pass an independent check"
    by_id = {a["id"]: a["nodes"] for a in out["gaps"][0]["actions"]}
    assert by_id["trust_card"] == [{"type": "validate.compare", "params": {}}]
    assert by_id["gci"] == [{"type": "verify.convergence", "params": {}}]
    assert by_id["review"] == []                                       # a CLI step: no node, shown as text
    assert {n["type"] for n in out["gaps"][0]["nodes"]} == {"validate.compare", "verify.convergence"}


def test_next_steps_without_the_lab_api_degrade(tmp_path, monkeypatch):
    from pinneapple_lab import curation
    lab = _curation(tmp_path)
    monkeypatch.delattr(curation, "next_steps", raising=False)
    out = ls.next_steps(ITEM, lab)
    assert not out["available"] and out["message"] == ls.NOT_AVAILABLE
    assert out["gaps"][0]["gap"].startswith("compare") and out["gaps"][0]["nodes"] == []


def test_insert_nodes_wires_by_port_type():
    doc = json.loads((FLOWS_DIR / "heat_1d.json").read_text())
    new, missing = ls.insert_nodes(doc, "pinn", [{"type": "validate.compare"}])
    assert not missing
    into = {e["target_port"]: e["source"] for e in new["edges"] if e["target"] == "compare"}
    assert into["candidates"] == "pinn" and into["reference"] != "pinn"
    assert not [i for i in pf.validate_flow(new) if i.level == "error"]
    with pytest.raises(KeyError):
        ls.insert_nodes(doc, "pinn", [{"type": "os.system"}])


def test_weak_decisions_suggest_registered_nodes():
    for topic, types in RAISE_NODES.items():
        assert types and all(t in pf.REGISTRY for t in types), topic
    weak = user_decision("n", "method", "picked by hand")
    assert weak.no_reference and weak.raise_with == list(RAISE_NODES["method"])
    strong = rule_decision("family.heat", what="heat")
    assert strong.raise_with == []


def test_api_routes(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from pinneapple_flow.studio import load_app
    monkeypatch.setenv("PINNEAPPLE_FLOW_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("PINNEAPPLE_FLOW_DB_URL", raising=False)
    monkeypatch.delenv("FLOW_USER", raising=False)
    app = load_app()
    c = TestClient(app)
    assert c.get("/api/lab/seal", params={"item": "x"}).json()["letter"] is None
    assert "available" in c.get("/api/lab/next-steps", params={"item": "x"}).json()
    doc = json.loads((FLOWS_DIR / "heat_1d.json").read_text())
    r = c.post("/api/flows/insert-nodes", json={"flow": doc, "anchor": "pinn", "nodes": [{"type": "uq.sensitivity"}]})
    assert r.status_code == 200 and r.json()["valid"]
    assert c.post("/api/flows/insert-nodes", json={"flow": doc, "nodes": [{"type": "nope"}]}).status_code == 400
    app.state.service.shutdown()


def test_one_after_run_hook_point(tmp_path, monkeypatch):
    from pinneapple_flow import runtime
    seen = []

    def hook(result, flow):
        seen.append((result.run_id, flow.id))

    def broken(result, flow):
        raise RuntimeError("boom")
    monkeypatch.setattr(runtime, "RUN_FINISHED_HOOKS", [hook, broken])
    f = pf.Flow("hooked", nodes=[{"id": "g", "type": "geometry.domain", "params": {"dimension": 1}}])
    r = pf.run(f, workdir=tmp_path)
    assert r.status == "done" and seen == [(r.run_id, "hooked")]
    assert r.hooks["broken"].startswith("failed: RuntimeError")
    assert r.hooks["export_to_lab"].startswith("skipped") or runtime.lab_export_status() == "available"


def test_finished_run_writes_a_valid_lab_manifest(tmp_path):
    from pinneapple_lab import manifest as M
    f = pf.Flow("heat manifest", nodes=[
        {"id": "geo", "type": "geometry.domain", "params": {"dimension": 1, "t_range": [0, 0.3]}},
        {"id": "eq", "type": "equations.system", "params": {"template": "heat"}},
        {"id": "fd", "type": "solver.classical", "params": {"method": "fd", "n": 21, "n_t": 6}},
        {"id": "val", "type": "validate.compare", "params": {"reference": "exact"}}], edges=[
        {"source": "geo", "source_port": "geometry", "target": "eq", "target_port": "geometry"},
        {"source": "eq", "source_port": "problem", "target": "fd", "target_port": "problem"},
        {"source": "fd", "source_port": "solution", "target": "val", "target_port": "candidates"},
        {"source": "eq", "source_port": "problem", "target": "val", "target_port": "problem"}])
    r = pf.run(f, workdir=tmp_path)
    assert r.hooks["export_to_lab"].startswith("manifest "), r.hooks
    assert "issues" not in r.hooks["export_to_lab"], r.hooks
    m = M.load(r.workdir)
    assert not M.validate(m, base=r.workdir)
    assert "validate.compare" in m.methods and any(x["name"] == "val.FD_rel_l2" for x in m.metrics)
    assert any(c["name"].startswith("val.") for c in m.checks) and m.files["data"] == ["decisions.json"]
    assert m.references and "pinneapple_simulation.numerical_solvers.symbolic_fd.solve_fd" in m.modules
