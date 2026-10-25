"""Surrogate Speed app (apps/surrogate_speed): HTTP smoke test of the routes and the page."""
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
    os.environ["SSPEED_DATA_DIR"] = str(tmp_path_factory.mktemp("sspeed"))
    os.environ["SSPEED_AUTOSTART"] = "0"
    os.environ["SSPEED_TRAIN_EPOCHS"] = "60"
    os.environ["SSPEED_GRID_CHECK"] = "0"
    sys.path.insert(0, str(ROOT / "apps" / "surrogate_speed"))
    from surrogate_speed.api import app
    with TestClient(app) as c:
        yield c


def test_health_and_page(client):
    assert client.get("/health").json()["status"] == "ok"
    page = client.get("/").text
    assert "A Fast Answer, Checked" in page and "cdn.jsdelivr" not in page
    assert client.get("/docs").status_code == 200


def _wait(fn, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = fn()
        if r:
            return r
        time.sleep(0.3)
    raise TimeoutError


def test_mesh_field_and_sweep(client):
    m = client.get("/api/mesh.json", params={"design": "dense_thin_fins"}).json()
    assert len(m["indices"]) % 3 == 0 and max(m["indices"]) < len(m["positions"]) // 3
    assert client.get("/api/mesh.json", params={"design": "nope"}).status_code == 404
    assert client.get("/api/thermal_field").json()["theta_fin_base_c"] > 0
    assert client.post("/api/run_sweep").json()["started"]
    s = _wait(lambda: (lambda j: j if j["sweep"]["status"] in ("done", "error") else None)(
        client.get("/api/sweep_status").json()))
    assert s["sweep"]["status"] == "done", s["sweep"]["error"]
    assert s["surrogate_trained"] and len(s["sweep"]["designs"]) == 7
    summ = s["sweep"]["summary"]
    assert summ["max_analytical_vs_fea_diff_pct"] < 2.0 and summ["mean_speedup_factor"] > 1
