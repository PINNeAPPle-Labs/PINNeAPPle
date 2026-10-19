"""Digital Twin Runtime app (apps/twin_runtime): HTTP smoke test of the routes, the examples and a live twin."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("paho.mqtt.client")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "twin_runtime"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["TWIN_DATA_DIR"] = str(tmp_path_factory.mktemp("twin"))
    sys.path.insert(0, str(APP))
    from twin_runtime.api import app
    with TestClient(app) as c:
        yield c


def _wait(client, job):
    for _ in range(300):
        j = client.get(f"/api/jobs/{job['id']}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_health_meta_page(client):
    assert client.get("/health").json()["status"] == "ok"
    m = client.get("/api/meta").json()
    assert m["db"] == "sqlite" and m["mqtt"] == "embedded"
    assert m["example_rom"]["metrics"]["rollout_rmse_h"] > 0
    assert "Digital Twin Runtime" in client.get("/").text
    assert client.get("/docs").status_code == 200
    assert client.get("/api/example-file/tank_rom.json").status_code == 200
    assert client.get("/api/example-file/../api.py").status_code == 404


def test_replay_example_finds_the_leak(client):
    j = _wait(client, client.post("/api/replay", json={"example": True}).json())
    assert j["status"] == "done", j.get("error")
    res = j["result"]
    raised = [a for a in res["alerts"] if not a["cleared"]]
    assert len(raised) == 1 and raised[0]["data"]["parameter"] == "q_leak"
    assert res["snapshot"]["state"]["q_leak"] == pytest.approx(5e-4, rel=0.2)
    st = client.get(f"/api/jobs/{j['id']}/files/twin.study.json")
    assert st.status_code == 200 and json.loads(st.content)["kind"] == "StudyResult"
    assert client.get(f"/api/jobs/{j['id']}/files/output.log").status_code == 404


def test_replay_upload_with_rom(client):
    csv = (APP / "examples" / "tank_day_with_leak.csv").read_bytes()
    up = client.post("/api/uploads", files={"file": ("rec.csv", csv)}).json()
    j = _wait(client, client.post("/api/replay", json={"upload_id": up["id"], "model": "rom"}).json())
    assert j["status"] == "done", j.get("error")
    assert j["result"]["model"] == "controlled_opinf"
    assert client.post("/api/replay", json={"upload_id": "nope"}).status_code == 404
    assert client.post("/api/replay", json={"example": True, "model": "magic"}).status_code == 422


def test_model_upload_checks_sha(client):
    ref = (APP / "examples" / "tank_rom.json.modelref.json").read_bytes()
    model = (APP / "examples" / "tank_rom.json").read_bytes()
    r = client.post("/api/models", files={"modelref": ("ref.json", ref), "model_file": ("tank_rom.json", model)})
    assert r.status_code == 200 and r.json()["kind"] == "controlled_opinf"
    bad = client.post("/api/models", files={"modelref": ("ref.json", ref), "model_file": ("tank_rom.json", b"{}")})
    assert bad.status_code == 422


def test_live_twin_over_mqtt_and_rest(client):
    r = client.post("/api/live/start", json={"speed": 600}).json()
    assert r["broker_port"] and r["topic"] == "pinneapple/tank-1/sensors"
    deadline = time.time() + 15
    while time.time() < deadline:
        st = client.get("/api/assets/tank-1/state").json()
        if st["n_updates"] >= 20:
            break
        time.sleep(0.2)
    assert st["n_updates"] >= 20 and set(st["state"]) == {"h", "T", "q_leak", "eta"}
    assert client.post("/api/live/tank-1/fault", json={"kind": "leak", "magnitude": 6e-4}).status_code == 200
    assert client.post("/api/live/tank-1/fault", json={"kind": "boom"}).status_code == 422
    live = client.get("/api/assets/tank-1/live").json()
    assert live["forecast"]["t"] and live["limits"]["h"] == [0.3, 1.8]
    assert client.get("/api/assets/tank-1/history?table=estimates&limit=3").json()
    assert json.loads(client.get("/api/assets/tank-1/study").content)["study_type"] == "twin"
    assert client.post("/api/live/tank-1/stop").json()["n_updates"] >= 20

    client.post("/api/live/start", json={"asset": "pump-7", "source": "rest"})
    rows = [{"t": 10.0 * i, "h": 1.2, "T": 25.0, "q_in": 0.00657, "Q": 2.776e5, "T_in": 15.0, "T_amb": 20.0}
            for i in range(5)]
    out = client.post("/api/assets/pump-7/readings", json=rows).json()
    assert out["accepted"] == 5 and out["last"]["status"] in ("normal", "warning")
    assert client.post("/api/assets/pump-7/readings", json={"t": 0.0, "h": 1}).status_code == 422
    client.post("/api/live/pump-7/stop")
    assert client.get("/api/assets/nope/state").status_code == 404


def test_series_flow_job(client):
    j = _wait(client, client.post("/api/series", json={}).json())
    assert j["status"] == "done", j.get("error")
    steps = j["result"]["steps"]
    assert [s["day"] for s in steps] == [13, 14, 15, 17, 20]
    assert "tank_rom.json.modelref.json" in j["result"]["files"]
    assert client.get(f"/api/jobs/{j['id']}/files/optimize.study.json").status_code == 200
