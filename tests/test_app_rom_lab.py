"""Reduced Order Model Lab app (apps/rom_lab): HTTP smoke test of the study queue, the examples, the exports and the
run endpoint, on the shared runtime."""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["ROM_DATA_DIR"] = str(tmp_path_factory.mktemp("rom"))
    sys.path.insert(0, str(ROOT / "apps" / "rom_lab"))
    from rom_lab.api import app
    return TestClient(app)


def _wait(client, job):
    for _ in range(600):
        r = client.get(f"/api/jobs/{job['id']}").json()
        if r["status"] in ("done", "failed", "cancelled"):
            return r
        time.sleep(0.2)
    raise AssertionError("study did not finish")


def test_health_meta_and_page(client):
    assert client.get("/health").json()["status"] == "ok"
    m = client.get("/api/meta").json()
    assert {e["id"] for e in m["examples"]} == {"wake", "beam"}
    assert "dmd" in m["methods_by_data"]["time"] and "pod_gpr" in m["methods_by_data"]["parametric"]
    assert "Reduced Order Model Lab" in client.get("/").text
    assert client.get("/docs").status_code == 200


def test_uploaded_time_series_study_exports_and_runs(client):
    x = np.linspace(0, 2 * np.pi, 120)
    t = np.arange(60) * 0.1
    U = np.array([1 + np.sin(x - 2 * s) + 0.3 * np.sin(2 * x + 3 * s) for s in t])
    buf = io.BytesIO()
    np.savez(buf, u=U, times=t, points=np.c_[x, np.zeros_like(x)])
    files = {"files": ("wave.npz", buf.getvalue())}
    insp = client.post("/api/inspect", files=files).json()
    assert insp["n_snapshots"] == 60 and insp["kind"] == "time"
    job = client.post("/api/studies", files=files, data={"options": json.dumps(
        {"methods": ["persistence", "dmd", "opinf"], "ranks": [4, 8], "budget_s": 0.5})}).json()
    r = _wait(client, job)
    assert r["status"] == "done", r.get("error")
    res = r["result"]
    rows = {x["key"]: x for x in res["study"]["rows"]}
    assert rows["dmd_r8"]["error_mean"] < 1e-3 and rows["persistence"]["error_mean"] > 0.5
    assert {c["name"].split(" ")[0] for c in res["checks"]} >= {"best", "held-out"}
    sr = client.get(f"/api/jobs/{job['id']}/files/study_result.json").json()
    assert sr["kind"] == "StudyResult" and sr["study_type"] == "rom" and sr["models"][0]["model_type"] == "rom"
    ref = client.get(f"/api/jobs/{job['id']}/files/models/dmd_r8.modelref.json").json()
    assert ref["metrics_split"].startswith("test") and ref["sha256"]
    npz = client.get(f"/api/jobs/{job['id']}/files/models/dmd_r8.rom.npz").content
    import hashlib
    assert hashlib.sha256(npz).hexdigest() == ref["sha256"]
    onnx = client.get(f"/api/jobs/{job['id']}/onnx/dmd_r8")
    assert onnx.status_code == 200 and len(onnx.content) > 1000
    assert client.get(f"/api/jobs/{job['id']}/onnx/opinf_r8").status_code == 422
    spec = client.get(f"/api/jobs/{job['id']}/gateway/dmd_r8").json()
    assert spec["inputs"] == ["t"] and len(spec["outputs"]) == 120
    rec = client.get(f"/api/jobs/{job['id']}/reconstruction", params={"model": "dmd_r8", "snapshot": 2}).json()
    assert rec["rel_l2"] < 1e-3 and len(rec["xy"]) == 120
    nomesh = client.get(f"/api/jobs/{job['id']}/view3d", params={"model": "dmd_r8", "snapshot": 2})
    assert nomesh.status_code == 422 and "no mesh" in nomesh.json()["detail"]
    run = client.post("/api/run", files={"model": ("m.rom.npz", npz)}, data={"inputs": json.dumps({"times": [5.5, 9.0]})})
    out = run.json()
    assert run.status_code == 200 and len(out["snapshots"]) == 2 and out["in_range"] == [False, False]
    csv = client.post("/api/run", files={"model": ("m.rom.npz", npz)}, data={"inputs": '{"times": [5.5]}', "fmt": "csv"})
    assert csv.text.startswith("t,in_range,u[0]")
    assert client.get("/api/usage").json()["units"]["studies"]["used"] >= 1


@pytest.mark.parametrize("eid,best_err", [("beam", 0.08)])
def test_example_runs_offline(client, eid, best_err):
    r = _wait(client, client.post(f"/api/examples/{eid}").json())
    assert r["status"] == "done", r.get("error")
    best = next(x for x in r["result"]["study"]["rows"] if x["key"] == r["result"]["best_key"])
    assert best["error_mean"] < best_err and r["result"]["wall_s"] < 60
    assert r["result"]["snapshots"]["mesh"]["node_sets"] == ["clamp", "loaded tip"]
    key, job = r["result"]["best_key"], r["id"]
    v = client.get(f"/api/jobs/{job}/view3d", params={"model": key, "snapshot": 1})
    assert v.status_code == 200, v.text
    v = v.json()
    rec = client.get(f"/api/jobs/{job}/reconstruction", params={"model": key, "snapshot": 1, "field": "von_mises"}).json()
    assert v["errors"]["von_mises"]["rel_l2"] == pytest.approx(rec["rel_l2"], rel=1e-9)
    assert v["errors"]["von_mises"]["max"] == pytest.approx(max(abs(e) for e in rec["error"]), rel=1e-9)
    assert "study mean over 20 held-out designs" in v["errors"]["von_mises"]["summary"][0]
    assert {m["kind"] for m in v["view"]["markers"]} >= {"bc", "load", "hotspot"}
    assert v["view"]["deform"]["FEM"]["unit"] == "m" and v["deformation"] == "displacement"
    glb = client.get(v["glb"])
    assert glb.status_code == 200 and glb.content[:4] == b"glTF"
    assert client.get(f"/api/jobs/{job}/view3d", params={"model": key, "snapshot": 99}).status_code == 422


def test_bad_uploads_are_refused(client):
    assert client.post("/api/inspect", files={"files": ("x.txt", b"hello")}).status_code == 422
    assert client.post("/api/studies", files={"files": ("x.npz", b"PK")}, data={"options": "nope"}).status_code == 422
    assert client.post("/api/run", files={"model": ("m.npz", b"junk")}, data={"inputs": "{}"}).status_code == 422
