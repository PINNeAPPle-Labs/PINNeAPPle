"""What another application (the CFD flow mode) builds on: the not_run / degraded node states, a node type whose
library is missing finishing not_run instead of being refused, registering port types / references / rules through
the public API, a table prefix for the store, saving without change, and DOI verification."""
import pytest

import pinneapple_flow as pf
from pinneapple_flow import references as R
from pinneapple_flow.registry import Param, Port, node

if "test.missing_tool" not in pf.REGISTRY:
    @node("test.missing_tool", "needs a tool", category="Test", outputs=[Port("metrics", "metrics")],
          available=lambda: (False, "needs footool on PATH"), calls="footool (test)")
    def _missing(ctx, inputs, params):
        raise AssertionError("must not run")

    @node("test.fallback", "declared fallback", category="Test", outputs=[Port("metrics", "metrics")],
          params=[Param("give_up", "bool", False)], calls="a declared fallback (test)")
    def _fallback(ctx, inputs, params):
        if params["give_up"]:
            ctx.not_run("no input case for this geometry")
        ctx.degrade("coarse correlation instead of the solver")
        return {"metrics": {"value": 1.0}}

    @node("test.consumer", "needs metrics", category="Test",
          inputs=[Port("m", "metrics"), Port("opt", "metrics", required=False)], outputs=[Port("metrics", "metrics")],
          calls="consumes metrics (test)")
    def _consumer(ctx, inputs, params):
        return {"metrics": {"got_optional": inputs.get("opt") is not None}}


def E(a, b, port="m"):
    return {"source": a, "source_port": "metrics", "target": b, "target_port": port}


def test_unavailable_node_finishes_not_run_and_downstream_too(tmp_path):
    f = pf.Flow("x", nodes=[{"id": "a", "type": "test.missing_tool"}, {"id": "b", "type": "test.consumer"}],
                edges=[E("a", "b")])
    r = pf.run(f, workdir=tmp_path)
    assert r.nodes["a"]["status"] == "not_run" and "footool" in r.report("a")["not_run_reason"]
    assert r.nodes["b"]["status"] == "not_run" and "did not run" in r.report("b")["not_run_reason"]


def test_degraded_and_optional_input_from_a_node_that_did_not_run(tmp_path):
    f = pf.Flow("y", nodes=[{"id": "ok", "type": "test.fallback"},
                            {"id": "no", "type": "test.fallback", "params": {"give_up": True}},
                            {"id": "c", "type": "test.consumer"}],
                edges=[E("ok", "c"), E("no", "c", "opt")])
    r = pf.run(f, workdir=tmp_path, cache=False)
    assert r.nodes["ok"]["status"] == "degraded" and r.report("ok")["degraded"]
    assert r.nodes["no"]["status"] == "not_run"
    assert r.nodes["c"]["status"] == "done" and r.output("c", "metrics") == {"got_optional": False}


def test_register_port_type_reference_and_rule():
    pf.register_port_type("cfd_case", "an OpenFOAM case folder (test)", converts_to=["dataset"])
    assert pf.compatible("cfd_case", "dataset") and not pf.compatible("dataset", "cfd_case")
    with pytest.raises(ValueError):
        pf.register_port_type("cfd_case", "something else")
    entry = {"title": "A test reference", "authors": ["A. Author"], "year": 2001, "venue": "Test Journal 1, 1-2",
             "doi": "10.1234/test.2001", "kind": "article", "topics": ["test"]}
    assert pf.register_references({"test2001ref": entry}) == ["test2001ref"]
    assert R.ref("test2001ref")["title"] == "A test reference"
    with pytest.raises(ValueError):
        pf.register_references({"bad": {"title": "no authors"}})
    with pytest.raises(ValueError):
        pf.register_references({"test2001ref": {**entry, "year": 2002}})
    rule = pf.Rule("test.rule", "a test rule", cites=("test2001ref",), topic="method")
    assert pf.register_rule(rule) is rule
    with pytest.raises(ValueError):
        pf.register_rule(pf.Rule("test.unknown", "cites nothing known", cites=("nope",)))
    from pinneapple_flow.decisions import rule_decision
    assert rule_decision("test.rule", what="x").references[0]["key"] == "test2001ref"


def test_store_table_prefix_and_saving_without_change(tmp_path):
    from pinneapple_flow.store import FlowStore
    st = FlowStore(f"sqlite:///{tmp_path}/app.db", table_prefix="flow_")
    assert st.flows.name == "flow_flows" and st.runs.name == "flow_runs"
    doc = {"schema": "pinneapple-flow/1", "id": "same", "name": "n", "nodes": [{"id": "g", "type": "geometry.domain"}],
           "edges": []}
    assert st.save_flow(doc) == 1
    assert st.save_flow(doc) == 1                      # no change: the current version
    assert st.save_flow({**doc, "name": "renamed"}) == 2
    with pytest.raises(ValueError):
        FlowStore(f"sqlite:///{tmp_path}/b.db", table_prefix="Bad-Prefix")


def test_generic_cfd_references_and_online_doi_check(monkeypatch):
    for k in ("launder1974ke", "menter1994sst", "white2011fm", "anderson2003mcf", "hirt1981vof", "elghobashi1994",
              "patankar1980", "incropera2007", "colebrook1939", "finnie1960", "kovasznay1948"):
        assert not R.problems(R.ref(k)), k
    calls = []

    def fake(url, timeout=8.0):
        calls.append(url)
        if "doi.org" in url:
            return 200, {"responseCode": 1}
        return 200, {"message": {"title": ["Two-equation eddy-viscosity turbulence models for engineering "
                                           "applications"]}}
    monkeypatch.setattr(R, "_get_json", fake)
    v = R.verify_doi("10.2514/3.12149", R.ref("menter1994sst")["title"])
    assert v["status"] == "verified" and v["title_match"] is True and v["source"] == "doi.org,crossref"
    monkeypatch.setattr(R, "_get_json", lambda url, timeout=8.0: (None, None))   # offline
    assert R.verify_doi("10.2514/3.12149")["status"] == "unverified"
    monkeypatch.setattr(R, "_get_json", lambda url, timeout=8.0: (404, None))
    assert R.verify_doi("10.9999/none")["status"] == "not found"
