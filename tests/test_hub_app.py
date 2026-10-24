"""HTTP test of the PINNeAPPle Hub app (apps/hub): catalogue, item pages, file serving limited to the index's
folders, zoo inference against the library's own fixed-input answers, composing, sandbox validation as a job,
downloads and My Hub."""
from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "hub"
pytest.importorskip("fastapi")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["HUB_DATA_DIR"] = str(tmp_path_factory.mktemp("hub"))
    sys.path.insert(0, str(APP))
    from fastapi.testclient import TestClient
    from hub.api import app
    with TestClient(app) as c:
        yield c


def _wait(c, jid, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.3)
    raise TimeoutError(jid)


def test_pages_and_catalogue(client):
    assert client.get("/health").json()["status"] == "ok"
    for path in ("/", "/create", "/mine", "/item/geometry%3Af1_car"):
        r = client.get(path)
        assert r.status_code == 200 and "PINNeAPPle Hub" in r.text
        assert "content-security-policy" in r.headers
    cfg = client.get("/api/config").json()
    assert set(cfg["tiers"]) == set("ABCDEF") and cfg["counts"]["geometry"] >= 17
    home = client.get("/api/home").json()
    assert home["featured"] and len(home["categories"]) == 6
    r = client.get("/api/search", params={"q": "wind turbine", "kind": "geometry"}).json()
    assert r["items"][0]["id"] == "geometry:wind_turbine_nrel5mw"
    a = client.get("/api/search", params={"quality": "A", "limit": 500}).json()
    assert a["total"] and all(i["quality"] == "A" for i in a["items"])
    assert client.get("/api/search", params={"q": "zzzz-nothing"}).json()["total"] == 0


def test_item_page_data_and_files(client):
    it = client.get("/api/items/geometry:f1_car").json()
    assert it["code"].startswith("import pinneapple as pp") and it["run"]["ok"]
    glb = client.get("/files/" + it["extra"]["glb"])
    assert glb.status_code == 200 and glb.content[:4] == b"glTF"
    for f in it["files"]:
        if f.get("path"):
            assert client.get("/files/" + f["path"]).status_code == 200, f
    ex = client.get("/api/items/example:accretion_flow").json()
    assert ex["quality"]["letter"] in "ABCDEF" and ex["results"]["figures"]
    assert client.get("/files/" + ex["results"]["figures"][0]).status_code == 200
    assert client.get("/api/items/nope:x").status_code == 404
    for bad in ("../pyproject.toml", "data/web/../../pyproject.toml", "apps/deploy/.env.example",
                "pinneapple_hub/library/__init__.py", "lab/curation.json/../../AGENTS.md"):
        assert client.get("/files/" + bad).status_code == 404, bad


def test_server_inference_matches_the_library_answers(client):
    fx = json.loads((ROOT / "data" / "web" / "zoo" / "fixtures.json").read_text())["models"]["heatsink_forced_mlp"]
    meta = json.loads((ROOT / "data" / "web" / "zoo" / "heatsink_forced_mlp.json").read_text())
    cols = meta["onnx"]["input_columns"]
    r = client.post("/api/models/heatsink_forced_mlp/predict", json={"inputs": dict(zip(cols, fx["inputs"][0], strict=True))})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["verdict"]["status"] == fx["evaluation"][0]["verdict"]
    for k, v in fx["predict"].items():
        if k in out["outputs"]:
            assert out["outputs"][k] == pytest.approx(v[0], rel=1e-5)
    assert client.post("/api/models/heatsink_forced_mlp/predict", json={"inputs": {"n_fins": 3}}).status_code == 422
    assert client.post("/api/models/nope/predict", json={"inputs": {}}).status_code == 404


def test_compose_validate_download_and_my_hub(client):
    body = {"name": "Rod grid", "blocks": [{"type": "geometry.domain", "params": {"dimension": 1}},
                                           {"type": "geometry.mesh", "params": {"n": 41}}]}
    blocks = client.get("/api/compose/blocks").json()
    assert {"geometry.domain", "geometry.mesh"} <= {b["type"] for b in blocks["blocks"]}
    out = client.post("/api/compose", json=body).json()
    assert out["ok"] and "pf.run(flow" in out["code"]
    assert client.post("/api/compose", json={"blocks": [{"type": "os.system"}]}).status_code == 422
    bad = client.post("/api/compose/validate", json={"blocks": [{"type": "geometry.mesh", "params": {"n": 1}}]})
    assert bad.status_code == 422
    job = client.post("/api/compose/validate", json=body).json()
    j = _wait(client, job["id"])
    assert j["status"] == "done" and j["result"]["ok"], j
    assert "mesh done {'nodes': 41}" in j["result"]["stdout"]
    spec = base64.urlsafe_b64encode(json.dumps(body).encode()).decode().rstrip("=")
    py = client.get(f"/api/compose/download/py?spec={spec}")
    assert py.status_code == 200 and py.text == out["code"] and "rod_grid.py" in py.headers["content-disposition"]
    fl = client.get(f"/api/compose/download/json?spec={spec}").json()
    assert fl["schema"] == "pinneapple-flow/1"
    saved = client.post("/api/mine", json={**body, "job_id": job["id"]}).json()
    assert saved["id"] == "mine:rod_grid" and saved["run"]["status"] == "ok"
    assert [i["id"] for i in client.get("/api/mine").json()["items"]] == ["mine:rod_grid"]
    assert client.get("/api/items/mine:rod_grid").json()["code"] == out["code"]
    assert client.delete("/api/mine/mine:rod_grid").status_code == 200
    assert client.get("/api/mine").json()["items"] == []
