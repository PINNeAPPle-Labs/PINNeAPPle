"""Model Benchmarking app (apps/model_benchmarking): HTTP smoke test of the routes and the page."""
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


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["MBENCH_DATA_DIR"] = str(tmp_path_factory.mktemp("mbench"))
    os.environ["MBENCH_AUTOSTART"] = "0"
    os.environ["MBENCH_EPOCH_SCALE"] = "0.002"
    sys.path.insert(0, str(ROOT / "apps" / "model_benchmarking"))
    from model_benchmarking.api import app
    with TestClient(app) as c:
        yield c


def test_health_and_page(client):
    assert client.get("/health").json()["status"] == "ok"
    page = client.get("/").text
    assert "Which Model Actually Deserves to Win" in page and "cdn.jsdelivr" not in page
    assert client.get("/docs").status_code == 200


def _wait(fn, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = fn()
        if r:
            return r
        time.sleep(0.3)
    raise TimeoutError


def test_run_and_poll(client):
    st = client.get("/api/sweep_status").json()
    assert st["sweep"]["status"] == "idle" and [b["epochs"] for b in st["budgets"]] == [24, 8, 10, 1]
    assert client.post("/api/run_sweep").json()["started"]
    s = _wait(lambda: (lambda j: j["sweep"] if j["sweep"]["status"] in ("done", "error") else None)(
        client.get("/api/sweep_status").json()))
    assert s["status"] == "done", s["error"]
    assert [m["name"] for m in s["result"]["models"]] == ["Vanilla PINN", "SIREN", "PIELM", "XTFC"]
    assert s["result"]["ranking_order_unchanged"]
