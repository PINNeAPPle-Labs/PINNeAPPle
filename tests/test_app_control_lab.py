"""Physics-Based Control Lab app (apps/control_lab): HTTP smoke test of the comparison, recommendation and safety."""
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
APP = ROOT / "apps" / "control_lab"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["CONTROL_DATA_DIR"] = str(tmp_path_factory.mktemp("control"))
    sys.path.insert(0, str(APP))
    from control_lab.api import app
    with TestClient(app) as c:
        yield c


def _wait(client, job):
    for _ in range(900):
        j = client.get(f"/api/jobs/{job['id']}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_health_meta_page_and_safety_statement(client):
    assert client.get("/health").json()["status"] == "ok"
    m = client.get("/api/meta").json()
    assert "no command path to real equipment" in m["mode"]
    page = client.get("/").text
    assert "Physics-Based Control Lab" in page and "no command path to real equipment" in page
    assert client.get("/docs").status_code == 200


def test_comparison_mpc_beats_baselines(client):
    body = {"controllers": ["pid", "rules", "lmpc"], "scenario": {"duration": 1800}}
    j = _wait(client, client.post("/api/compare", json=body).json())
    assert j["status"] == "done", j.get("error")
    runs = j["result"]["runs"]
    assert runs["lmpc"]["metrics"]["iae_T_K_s"] < runs["pid"]["metrics"]["iae_T_K_s"]
    assert runs["lmpc"]["metrics"]["iae_T_K_s"] < runs["rules"]["metrics"]["iae_T_K_s"]
    assert runs["lmpc"]["metrics"]["max_violation_T_K"] == 0.0 and runs["lmpc"]["plans"]
    st = json.loads(client.get(f"/api/jobs/{j['id']}/files/control.study.json").content)
    assert st["study_type"] == "control" and "no command path" in st["limits"][0]
    assert client.get(f"/api/jobs/{j['id']}/files/trajectories.csv").text.startswith("controller,t,h,T")


def test_surrogate_mpc_and_validation(client):
    pytest.importorskip("casadi")
    j = _wait(client, client.post("/api/compare", json={"controllers": ["smpc"], "scenario": {"duration": 900}}).json())
    assert j["status"] == "done", j.get("error")
    assert j["result"]["runs"]["smpc"]["failed_solves"] == 0
    assert client.post("/api/compare", json={"controllers": ["smpc"], "spec": {"dt": 10}}).status_code == 422
    assert client.post("/api/compare", json={"controllers": ["boom"]}).status_code == 422
    assert client.post("/api/compare", json={"spec": {"horizn": 3}}).status_code == 422
    assert client.post("/api/compare", json={"scenario": {"duration": 10}}).status_code == 422


def test_recommend_from_a_twin_snapshot(client):
    snap = {"state": {"h": 1.1, "T": 26.0, "q_leak": 3e-4, "eta": 0.97}, "status": "warning"}
    r = client.post("/api/recommend", json={"estimate": snap, "t": 700,
                                            "reading": {"q_in": 0.0066, "Q": 3e5, "T_in": 15, "T_amb": 20}}).json()
    assert "no command path" in r["mode"] and 0 <= r["proposed_next"]["Q"] <= 600e3
    assert len(r["plan"]["pred_T"]) == len(r["reference"]["T_ref"])
    assert client.post("/api/recommend", json={"estimate": {"h": 1.0}}).status_code == 422
