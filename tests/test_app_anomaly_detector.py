"""Physics Anomaly Detector app (apps/anomaly_detector): HTTP smoke test of the routes, the example and the benchmark."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "anomaly_detector"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["ANOMALY_DATA_DIR"] = str(tmp_path_factory.mktemp("anomaly"))
    sys.path.insert(0, str(APP))
    from anomaly_detector.api import app
    with TestClient(app) as c:
        yield c


def _wait(client, job):
    for _ in range(600):
        j = client.get(f"/api/jobs/{job['id']}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_health_meta_page(client):
    assert client.get("/health").json()["status"] == "ok"
    m = client.get("/api/meta").json()
    assert set(m["methods"]) == {"statistical", "timeseries", "balance", "model"}
    assert [b["name"] for b in m["preset"]["balances"]] == ["mass", "energy"]
    assert "Physics Anomaly Detector" in client.get("/").text
    assert client.get("/docs").status_code == 200


def test_example_finds_the_faults_and_explains_them(client):
    j = _wait(client, client.post("/api/analyze", json={"example": True}).json())
    assert j["status"] == "done", j.get("error")
    r = j["result"]
    assert r["score"]["event_recall"] >= 5 / 6
    leak = next(p for p in r["score"]["per_fault"] if p["kind"] == "leak")
    assert leak["detected"] and "balance" in leak["methods"]
    assert any(e["explanations"][0].startswith("mass balance broken by -") for e in r["events"])
    st = json.loads(client.get(f"/api/jobs/{j['id']}/files/anomaly.study.json").content)
    assert st["kind"] == "StudyResult" and st["metrics"]["event_recall"] >= 5 / 6
    csv = client.get(f"/api/jobs/{j['id']}/files/events.csv").text
    assert csv.startswith("start_s,end_s,severity") and len(csv.splitlines()) == len(r["events"]) + 1
    assert client.get(f"/api/jobs/{j['id']}/files/../api.py").status_code == 404


def test_upload_with_custom_balance_and_no_model(client):
    csv = (APP / "examples" / "tank_six_faults.csv").read_bytes()
    up = client.post("/api/uploads", files={"file": ("plant.csv", csv)}).json()
    cols = client.get(f"/api/columns/{up['id']}").json()
    assert "fault" in cols["columns"] and cols["t_range"][0] == 0.0
    body = {"upload_id": up["id"], "normal_until": 7200, "sensors": ["h", "q_in", "q_out"], "methods": ["balance"],
            "balances": [{"name": "volume", "accumulation": "A * h", "inflow": "q_in", "outflow": "q_out",
                          "units": "m3/s"}], "constants": {"A": 1.5}, "model": "none", "label_column": "fault"}
    j = _wait(client, client.post("/api/analyze", json=body).json())
    assert j["status"] == "done", j.get("error")
    assert [c["name"] for c in j["result"]["channels"]] == ["balance:volume"]
    assert j["result"]["score"]["false_alarm_events"] == 0
    bad = dict(body, balances=[{"name": "x", "accumulation": "__import__('os')", "inflow": "q_in"}])
    jb = _wait(client, client.post("/api/analyze", json=bad).json())
    assert jb["status"] == "failed" and "balance 'x'" in jb["error"]
    assert client.post("/api/analyze", json={"upload_id": up["id"]}).status_code == 422
    assert client.post("/api/analyze", json={"upload_id": "nope", "normal_until": 1}).status_code == 404


def test_benchmark_job(client):
    j = _wait(client, client.post("/api/benchmark", json={"seeds": [101]}).json())
    assert j["status"] == "done", j.get("error")
    res = j["result"]["results"]
    assert set(res) == {"statistical", "timeseries", "balance", "model", "combined"}
    assert res["combined"]["event_recall"] >= 5 / 6
    assert client.post("/api/benchmark", json={"seeds": list(range(10))}).status_code == 422
