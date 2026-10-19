"""Flow Studio HTTP API (apps/flow_studio/backend/app.py over pinneapple_flow.service): planning, validation,
previews, storage with versions, examples, a background run with live status, node results, artifact access."""
import time

import pytest

pytest.importorskip("fastapi")
from starlette.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PINNEAPPLE_FLOW_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("PINNEAPPLE_FLOW_DB_URL", raising=False)
    monkeypatch.delenv("FLOW_USER", raising=False)
    from pinneapple_flow.studio import load_app
    app = load_app()
    with TestClient(app) as c:
        yield c
    app.state.service.shutdown()


def small_flow():
    return {"schema": "pinneapple-flow/1", "id": "api-heat", "name": "api heat", "nodes": [
        {"id": "geo", "type": "geometry.domain", "params": {"dimension": 1, "t_range": [0, 0.3]}},
        {"id": "eq", "type": "equations.system", "params": {"template": "heat"}},
        {"id": "fd", "type": "solver.classical", "params": {"method": "fd", "n": 31, "n_t": 6}},
        {"id": "plot", "type": "viz.plot", "params": {}}],
        "edges": [{"source": "geo", "source_port": "geometry", "target": "eq", "target_port": "geometry"},
                  {"source": "eq", "source_port": "problem", "target": "fd", "target_port": "problem"},
                  {"source": "fd", "source_port": "solution", "target": "plot", "target_port": "solution"}]}


def test_catalog_info_schema(client):
    assert client.get("/health").json() == {"ok": True}
    cat = client.get("/api/catalog").json()
    assert len(cat["nodes"]) >= 20 and "problem" in cat["port_types"]
    assert client.get("/api/info").json()["planners"][0] == "heuristic"
    assert client.get("/api/schema").json()["properties"]["schema"]["const"] == "pinneapple-flow/1"
    r = client.get("/")
    assert "content-security-policy" in r.headers and r.headers["x-frame-options"] == "DENY"


def test_plan_validate_and_previews(client):
    res = client.post("/api/plan", json={"text": "transient heat in a 2D plate", "backend": "heuristic"}).json()
    assert res["planner"] == "heuristic"
    v = client.post("/api/validate", json=res["proposals"][0]["flow"]).json()
    assert v["valid"], v["issues"]
    bad = small_flow()
    bad["edges"][1]["target_port"] = "nope"
    assert not client.post("/api/validate", json=bad).json()["valid"]
    eq = client.post("/api/preview/equations", json={"geometry": {"dimension": 3},
                                                     "equations": {"template": "heat"}}).json()
    assert eq["ok"] and eq["geometry"]["coords"] == ["x", "y", "z", "t"]
    eq = client.post("/api/preview/equations", json={"geometry": {"dimension": 1},
                                                     "equations": {"pdes": ["u_t = __import__('os')"]}}).json()
    assert not eq["ok"] and "allowed" in eq["error"]
    arch = client.post("/api/preview/architecture", json={"kind": "pod_mlp", "in_dim": 2, "out_dim": 100,
                                                          "config": {"blocks": [{"type": "dense", "width": 16}]}}).json()
    assert arch["total_params"] > 0 and arch["nodes"][0]["type"] == "Linear"
    assert client.post("/api/plan", json={"text": ""}).status_code == 400


def test_save_versions_examples(client):
    f = small_flow()
    assert client.post("/api/flows", json={"flow": f, "note": "a"}).json() == {"id": "api-heat", "version": 1}
    f["description"] = "second"
    assert client.post("/api/flows", json={"flow": f}).json()["version"] == 2
    assert client.get("/api/flows/api-heat").json()["description"] == "second"
    assert len(client.get("/api/flows/api-heat/versions").json()) == 2
    assert client.get("/api/flows/unknown").status_code == 404
    ex = client.get("/api/examples").json()
    names = {e["name"] for e in ex}
    assert {"heat_1d", "poisson_2d", "planned_digital_twin"} <= names
    assert client.get("/api/examples/heat_1d").json()["schema"] == "pinneapple-flow/1"
    assert client.get("/api/examples/..%2Fschema").status_code == 404


def test_run_status_results_and_artifacts(client):
    r = client.post("/api/runs", json={"flow": small_flow()}).json()
    rid = r["run_id"]
    for _ in range(300):
        st = client.get(f"/api/runs/{rid}").json()
        if st["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(0.2)
    assert st["status"] == "done", st
    assert st["nodes"]["fd"]["status"] == "done"
    res = client.get(f"/api/runs/{rid}/nodes/fd").json()
    assert res["report"]["metrics"]["solve_seconds"]["value"] >= 0
    fig = res["report"]["figures"][0]["path"]
    img = client.get("/api/artifact", params={"path": fig})
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    assert client.get("/api/artifact", params={"path": "/etc/passwd"}).status_code == 403
    assert "log" in res and client.get(f"/api/runs/{rid}/nodes/nope").status_code == 404
    again = client.post("/api/runs", json={"flow": small_flow(), "from_node": "plot"}).json()["run_id"]
    for _ in range(300):
        st = client.get(f"/api/runs/{again}").json()
        if st["status"] in ("done", "failed"):
            break
        time.sleep(0.2)
    assert st["nodes"]["fd"]["status"] == "cached" and st["nodes"]["plot"]["status"] == "done"


def test_flow_json_never_executes_code(client):
    f = small_flow()
    f["nodes"].append({"id": "x", "type": "pinneapple_core.python", "params": {"function": "os:system"}})
    assert client.post("/api/runs", json={"flow": f}).status_code == 400


def test_studio_refuses_public_bind_without_login(monkeypatch):
    from pinneapple_flow.studio import serve
    monkeypatch.delenv("FLOW_USER", raising=False)
    monkeypatch.delenv("FLOW_PASSWORD_HASH", raising=False)
    assert serve(host="0.0.0.0", port=1, open_browser=False, env_file="/nonexistent") == 2
