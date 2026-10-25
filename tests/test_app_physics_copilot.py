"""PhysicsCopilot app (apps/physics_copilot): a solve job end to end with the drafting step stubbed (training and
guardrail real), the rejection path, and a real local-LLM run when Ollama with llama3.2:3b is reachable."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _ollama_has(model="llama3.2:3b"):
    try:
        import requests
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        return r.ok and any(m["name"] in (model, f"{model}:latest") for m in r.json().get("models", []))
    except Exception:
        return False


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["COPILOT_DATA_DIR"] = str(tmp_path_factory.mktemp("copilot"))
    sys.path.insert(0, str(ROOT / "apps" / "physics_copilot"))
    from physics_copilot.api import app
    return TestClient(app)


def _wait(c, job_id, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/v1/jobs/{job_id}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.2)
    raise TimeoutError(job_id)


def _stub(monkeypatch, preset, kwargs=None):
    import pinneapple_llm as pl
    from pinneapple_llm.draft import DraftResult

    monkeypatch.setattr(pl, "draft_problem",
                        lambda d, **k: DraftResult(preset, dict(kwargs or {}), "stubbed draft", ""))


@pytest.mark.skipif(not os.environ.get("PINNEAPPLE_SLOW"), reason="slow: trains a 64x4 PINN (set PINNEAPPLE_SLOW=1)")
def test_solve_job_reports_every_check(client, monkeypatch):
    assert client.get("/health").json()["status"] == "ok"
    _stub(monkeypatch, "burgers_1d", {"nu": 0.02})
    job = _wait(client, client.post("/v1/solve", json={"description": "1-D viscous Burgers", "epochs": 10}).json()["id"], timeout=900)
    assert job["status"] == "done", job.get("error")
    r = job["result"]
    assert r["status"] == "done" and r["preset"] == "burgers_1d" and r["checks"]
    assert {c["name"] for c in r["checks"]} >= {"pde_residual"} and isinstance(r["trustworthy"], bool)
    assert client.get("/api/usage").json()["units"]["solves"]["used"] >= 1


def test_no_fit_is_a_rejection(client, monkeypatch):
    assert client.get("/health").json()["status"] == "ok"
    _stub(monkeypatch, None)
    job = _wait(client, client.post("/v1/solve", json={"description": "the stock market", "epochs": 10}).json()["id"], timeout=900)
    assert job["result"]["status"] == "rejected" and "stubbed draft" in job["result"]["reason"]


@pytest.mark.skipif(not _ollama_has(), reason="no local Ollama server with llama3.2:3b")
def test_real_local_llm(client):
    job = _wait(client, client.post("/v1/solve", json={"description": "1D viscous Burgers equation with nu=0.01",
                                                        "epochs": 100}).json()["id"], timeout=900)
    assert job["status"] == "done"
