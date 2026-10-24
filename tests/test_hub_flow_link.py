"""The Hub and Flow Studio on one registry (pinneapple_hub.library.flowlink, .compose, .flowruns): the palette and the
Hub index list the same node types with the same titles; every node type gets a ready, valid flow (inputs -> node ->
report) carrying the user's parameters; Hub <-> Flow Studio links round-trip; a flow published from Flow Studio
becomes a tool whose exposed parameters are variables of its Python, and that Python runs; recorded runs of the
bundled flows are current and appear on the pages of the nodes they use."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from pinneapple_hub import library as hub
from pinneapple_hub.library.compose import ComposeError, compose
from pinneapple_hub.library.flowlink import (
    FlowLinkError,
    add_link,
    bundled_flows_using,
    flow_for_node,
    flow_link,
    from_b64url,
)
from pinneapple_hub.library.sandbox import run_snippet

ROOT = Path(__file__).resolve().parents[1]
INDEX = hub.load_index()


def _specs():
    from pinneapple_flow import REGISTRY
    return [s for s in REGISTRY.specs() if not s.type.startswith("custom.")]


def test_palette_and_index_list_the_same_node_types():
    from pinneapple_flow.palette import entry
    items = {i["name"]: i for i in INDEX["items"] if i["id"].startswith("node:")}
    assert set(items) == {s.type for s in _specs()}
    for s in _specs():
        it, pal = items[s.type], entry(s)
        assert it["subkind"] == ("flow node" if pal["curated"] else "library call")
        assert it["title"].startswith(pal["title"]), (s.type, it["title"], pal["title"])
        assert it["extra"]["params_schema"] == s.params_schema()


def test_every_node_type_gets_a_valid_ready_flow():
    from pinneapple_flow import REGISTRY
    for s in _specs():
        out = flow_for_node(s.type)
        assert out["ok"] and out["issues"] == [], (s.type, out["issues"])
        types = [n["type"] for n in out["flow"]["nodes"]]
        assert s.type in types
        if s.outputs and s.type != "export.report":
            report = next(n for n in out["flow"]["nodes"] if n["type"] == "export.report")
            assert any(e["target"] == report["id"] for e in out["flow"]["edges"])
        for p in (p for p in s.inputs if p.required):
            node = out["node"]
            assert any(e["target"] == node and e["target_port"] == p.name for e in out["flow"]["edges"]), (s.type, p)
    assert all(t in REGISTRY for t in ("geometry.domain", "export.report"))


def test_bundled_wiring_is_reused_and_user_parameters_reach_the_flow():
    assert "burgers_1d" in bundled_flows_using("geometry.mesh")
    out = flow_for_node("geometry.mesh", {"n": 41})
    assert "bundled flow" in out["source"]
    mesh = next(n for n in out["flow"]["nodes"] if n["id"] == out["node"])
    assert mesh["params"]["n"] == 41
    bad = flow_for_node("geometry.mesh", {"n": 1})
    assert not bad["ok"] and any("n must be >=" in i["message"] for i in bad["issues"])
    with pytest.raises(FlowLinkError):
        flow_for_node("os.system")


def test_links_round_trip():
    out = flow_for_node("geometry.mesh", {"n": 41})
    link = flow_link(out["flow"], base="http://fs.example/")
    assert link.startswith("http://fs.example/?flow=")
    assert from_b64url(link.split("?flow=", 1)[1]) == out["flow"]
    add = add_link("geometry.mesh", {"n": 41}, base="http://fs.example")
    assert add.startswith("http://fs.example/?add=geometry.mesh&params=")
    assert from_b64url(add.split("&params=", 1)[1]) == {"n": 41}


def test_api_functions_link_to_the_node_that_wraps_them():
    it = next(i for i in INDEX["items"] if i.get("subkind") == "function"
              and i["name"] == "pinneapple_systems.time_series.predictability.predictability_report")
    wraps = {"node:lib.ts.predictability", "node:forecast.lab"}
    assert wraps & set(it["examples"]) and it["links"]["flow_studio"].removeprefix("?node=") in {
        w.removeprefix("node:") for w in wraps}
    lib = [i for i in INDEX["items"] if i.get("subkind") == "library call"]
    assert len(lib) >= 200
    runs = [i for i in lib if (i.get("quality") or {}).get("label") == "runs with its defaults"]
    assert runs and all(i["code"].startswith("from ") and i["run"]["ok"] for i in runs)


def test_a_flow_published_from_flow_studio_becomes_a_tool():
    flow = json.loads((ROOT / "pinneapple_flow" / "flows" / "antenna_vivaldi_limits.json").read_text())
    node = flow["nodes"][0]
    from pinneapple_flow import REGISTRY
    pname = next(p.name for p in REGISTRY[node["type"]].params if p.kind in ("float", "int"))
    out = compose({"name": "Vivaldi limits", "flow": flow, "expose": [{"node": node["id"], "param": pname}]})
    assert out["ok"], out["issues"]
    var = out["expose"][0]["var"]
    assert f"{var} = " in out["code"] and f'["params"][{pname!r}] = {var}' in out["code"]
    r = run_snippet(out["code"], timeout=600)
    assert r["ok"], r["stderr"]
    assert f"{node['id']} done" in r["stdout"]
    with pytest.raises(ComposeError):
        compose({"flow": {"name": "x", "nodes": [{"id": "a", "type": "os.system"}], "edges": []}})
    with pytest.raises(ComposeError):
        compose({"flow": "not a flow"})


def test_bundled_flow_runs_are_recorded_and_shown_on_node_pages():
    from pinneapple_hub.library.flowruns import current, read_flow_runs
    runs = read_flow_runs()
    flows = [i for i in INDEX["items"] if i["kind"] == "flow"]
    assert flows and all(current(runs, f["name"]) for f in flows), "python -m pinneapple_hub.library flowruns"
    done = [f for f in flows if f["results"]["flow_run"]["status"] == "done"]
    assert len(done) >= len(flows) // 2
    f = done[0]
    for nid, n in f["results"]["flow_run"]["nodes"].items():
        page = hub.get(INDEX, f"node:{n['type']}")
        assert any(x["flow"] == f["name"] and x["node"] == nid for x in page["results"]["flow_runs"])


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    pytest.importorskip("fastapi")
    os.environ["HUB_DATA_DIR"] = str(tmp_path_factory.mktemp("hubfl"))
    os.environ["HUB_FLOW_STUDIO_URL"] = "http://fs.example"
    sys.path.insert(0, str(ROOT / "apps" / "hub"))
    from fastapi.testclient import TestClient
    from hub.api import app
    with TestClient(app) as c:
        yield c


def test_api_flowlink_and_publish(client):
    r = client.post("/api/flowlink/node", json={"type": "geometry.mesh", "params": {"n": 41}})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ok"] and d["open"].startswith("http://fs.example/?flow=") and "?add=geometry.mesh" in d["add"]
    assert client.post("/api/flowlink/node", json={"type": "nope"}).status_code == 404
    flow = json.loads((ROOT / "pinneapple_flow" / "flows" / "heat_1d.json").read_text())
    body = {"name": "Heat rod", "flow": flow, "expose": [{"node": "mesh", "param": "n"}]}
    out = client.post("/api/compose", json=body).json()
    assert out["ok"] and out["expose"][0]["var"] == "mesh_n"
    saved = client.post("/api/mine", json=body).json()
    assert saved["extra"]["expose"][0]["param"] == "n" and saved["flow"]["nodes"][0]["type"] == "geometry.domain"
