"""VerifiedPhysics app (apps/verified_physics): the quick routes (knowledge graph, solver recommendation, mesh quality,
geometry classification) and the job routes with the LLM stubbed; full training only with PINNEAPPLE_SLOW=1."""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("networkx")
trimesh = pytest.importorskip("trimesh")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["VPHYS_DATA_DIR"] = str(tmp_path_factory.mktemp("vphys"))
    sys.path.insert(0, str(ROOT / "apps" / "verified_physics"))
    from verified_physics.api import app
    return TestClient(app)


def _wait(c, job_id, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/v1/runs/{job_id}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.3)
    raise TimeoutError(job_id)


def _stub(monkeypatch, preset, kwargs=None):
    import pinneapple_llm as pl
    from pinneapple_llm.draft import DraftResult

    monkeypatch.setattr(pl, "draft_problem", lambda d, **k: DraftResult(preset, dict(kwargs or {}), "stub", ""))


def _box_stl() -> bytes:
    buf = io.BytesIO()
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(buf, file_type="stl")
    return buf.getvalue()


def test_health_and_knowledge(client):
    assert client.get("/health").json()["status"] == "ok"
    s = client.get("/v1/knowledge/stats").json()
    assert s["total_presets"] >= 50
    assert client.get("/v1/knowledge/preset/burgers_1d").json()["pde_kind"] == "burgers"
    assert client.get("/v1/knowledge/preset/nope").status_code == 404
    assert client.get("/v1/knowledge/verified_equations").json()["verified_equations"]


def test_solver_routes(client):
    fam = client.get("/v1/solver_families").json()
    assert fam
    r = client.post("/v1/recommend_solver", json={"preset_name": "laplace_2d"})
    assert r.status_code == 200 and r.json()
    assert client.post("/v1/recommend_solver", json={"preset_name": "nope"}).status_code == 422


def test_mesh_routes(client):
    files = {"file": ("box.stl", _box_stl(), "application/octet-stream")}
    q = client.post("/v1/assess_mesh_quality", files=files)
    assert q.status_code == 200, q.text
    g = client.post("/v1/classify_geometry", files={"file": ("box.stl", _box_stl(), "application/octet-stream")})
    assert g.status_code == 200, g.text
    regions = g.json()["regions"]
    assert len(regions) == 6 and all(r["geometric_candidate"] == "wall" for r in regions) and not g.json()["llm_used"]
    assert client.post("/v1/assess_mesh_quality", files={"file": ("a.txt", b"x", "text/plain")}).status_code == 415


def test_analyze_rejection_and_inverse_scope(client, monkeypatch):
    _stub(monkeypatch, None)
    job = _wait(client, client.post("/v1/analyze", json={"description": "nothing fits", "epochs": 10}).json()["id"])
    assert job["status"] == "done" and job["result"]["rejected"]
    inv = client.post("/v1/inverse_problem", json={"preset_name": "burgers_1d", "observed_x": [[0.0, 0.0]],
                                                   "observed_y": [[0.0]], "unknown_param_names": ["nu"]}).json()
    job = _wait(client, inv["id"])
    assert job["status"] in ("done", "failed")
    if job["status"] == "done":
        assert "error" in job["result"]


@pytest.mark.skipif(not os.environ.get("PINNEAPPLE_SLOW"), reason="slow: trains PINNs (set PINNEAPPLE_SLOW=1)")
def test_analyze_trains_and_verifies(client, monkeypatch):
    _stub(monkeypatch, "burgers_1d", {"nu": 0.02})
    job = _wait(client, client.post("/v1/analyze", json={"description": "Burgers", "epochs": 10,
                                                         "n_collocation": 128}).json()["id"])
    assert job["status"] == "done", job.get("error")
    p = job["result"]["provenance"]
    assert p["drafted_preset"] == "burgers_1d" and p["guardrail_checks"]
