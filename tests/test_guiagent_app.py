"""Engineering GUI Agent app (apps/guiagent): the HTTP API around pinneapple_tools.automation, on the shared runtime
of apps/_shared (job queue, ownership, usage metering).

The applications (FreeCAD, OpenFOAM, ParaView) are replaced by an empty toolchain, so a run is recorded as blocked
at its first step; the full chain is tested in tests/test_automation.py where the applications are installed."""
from __future__ import annotations

import importlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GUIAGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GUIAGENT_FREE_RUNS", "2")
    for var in ("GUIAGENT_USER", "GUIAGENT_MODE", "GUIAGENT_ADMIN_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.syspath_prepend(str(ROOT / "apps" / "guiagent"))
    monkeypatch.syspath_prepend(str(ROOT / "apps" / "_shared"))
    sys.modules.pop("guiagent.api", None)
    api = importlib.import_module("guiagent.api")
    from pinneapple_tools.automation import Toolchain
    api._toolchain = Toolchain()                                 # no applications: steps are blocked, not run
    with TestClient(api.app) as c:
        yield c, api


def _wait(c, run_id, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        r = c.get(f"/api/runs/{run_id}").json()
        if not r["active"] and r["state"] not in ("running", "pending"):
            return r
        time.sleep(0.1)
    raise AssertionError("run did not stop")


def test_pages_workflows_and_interpretation(client):
    c, _ = client
    assert c.get("/health").json()["status"] == "ok"
    assert "Engineering GUI Agent" in c.get("/").text
    assert c.get("/shared/app.css").status_code == 200
    wfs = c.get("/api/workflows").json()
    wf = next(w for w in wfs if w["id"] == "channel_obstacle")
    assert [p["action"] for p in wf["plan"]][:2] == ["freecad.open", "freecad.macro"]
    assert wf["inputs"]["obstacle_height_mm"]["max"] == 80
    it = c.post("/api/interpret", json={"task": "channel with a 6 cm obstacle, inlet 25 mm/s"}).json()
    assert it["workflow"] == "channel_obstacle" and it["resolved_inputs"] == {"obstacle_height_mm": 60.0,
                                                                             "inlet_velocity_mm_s": 25.0}
    assert c.post("/api/interpret", json={"task": "obstacle height 300 mm in the channel"}).status_code == 422
    meta = c.get("/api/meta").json()
    assert meta["mode"] == "public" and meta["uploads"] is False and meta["max_jobs"] == 1
    assert c.get("/api/runtime").json()["free_limits_per_month"]["runs"] == 2


def test_run_is_queued_recorded_and_metered(client):
    c, api = client
    assert c.post("/api/runs", json={"workflow": "channel_obstacle", "inputs": {"obstacle_height_mm": 99}}).status_code == 422
    assert c.post("/api/runs", json={"workflow": "nope"}).status_code == 404
    r = c.post("/api/runs", json={"workflow": "channel_obstacle", "inputs": {"obstacle_height_mm": 30}})
    assert r.status_code == 200
    run_id, job_id = r.json()["run_id"], r.json()["job_id"]
    rec = _wait(c, run_id)
    assert rec["state"] == "blocked" and rec["steps"][0]["state"] == "blocked"
    assert "missing" in rec["steps"][0]["error"] and "folder" not in rec
    assert c.get(f"/api/jobs/{job_id}").json()["status"] == "done"            # the shared queue ran it
    assert c.get(f"/api/runs/{run_id}/report").status_code == 200
    from pinneapple_data.contracts import validate
    sr = c.get(f"/api/runs/{run_id}/study_result").json()
    assert validate(sr) == [] and sr["status"] == "failed" and sr["metadata"]["decision"] == "INCOMPLETE"
    z = zipfile.ZipFile(io.BytesIO(c.get(f"/api/runs/{run_id}/bundle").content))
    assert f"{run_id}/run.json" in z.namelist() and f"{run_id}/owner.json" not in z.namelist()
    assert c.get(f"/api/runs/{run_id}/files/../../etc/passwd").status_code == 404
    assert c.get(f"/api/runs/{run_id}/files/owner.json").status_code == 404
    assert c.post(f"/api/runs/{run_id}/approve", json={"step": "solve"}).status_code == 409
    assert [x["run_id"] for x in c.get("/api/runs").json()] == [run_id]
    assert "runs" in json.dumps(c.get("/api/usage").json())
    assert c.post("/api/workflows", files={"file": ("w.zip", b"PK", "application/zip")}).status_code == 403

    # someone else's run is not visible
    owner = api.RUNS / run_id / "owner.json"
    owner.write_text(json.dumps({**json.loads(owner.read_text()), "owner": "ip:somebody-else"}))
    assert c.get(f"/api/runs/{run_id}").status_code == 404 and c.get("/api/runs").json() == []

    # the free limit (GUIAGENT_FREE_RUNS=2) is enforced by the shared meter
    assert c.post("/api/runs", json={"workflow": "channel_obstacle"}).status_code == 200
    over = c.post("/api/runs", json={"workflow": "channel_obstacle"})
    assert over.status_code == 429 and over.json()["unit"] == "runs"
