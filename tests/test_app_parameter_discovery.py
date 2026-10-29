"""Parameter Discovery app (apps/parameter_discovery): HTTP smoke test of the routes and the page."""
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
    os.environ["PDISC_DATA_DIR"] = str(tmp_path_factory.mktemp("pdisc"))
    os.environ["PDISC_AUTOSTART"] = "0"
    os.environ["PDISC_EPOCHS"] = "30"
    os.environ["PDISC_OBSERVATIONS"] = "32"
    sys.path.insert(0, str(ROOT / "apps" / "parameter_discovery"))
    from parameter_discovery.api import app
    with TestClient(app) as c:
        yield c


def test_health_and_page(client):
    assert client.get("/health").json()["status"] == "ok"
    page = client.get("/").text
    assert "The AI Never Saw 0.15" in page and "cdn.jsdelivr" not in page
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
    assert client.get("/api/discovery/status").json()["status"] == "idle"
    assert client.post("/api/discovery/run").json()["started"]
    assert client.post("/api/discovery/run").status_code in (200, 409)
    s = _wait(lambda: (lambda j: j if j["status"] in ("done", "error") else None)(
        client.get("/api/discovery/status").json()))
    assert s["status"] == "done", s["error"]
    r = s["result"]
    assert r["scenario"]["n_epochs"] == 30 and r["identifiability"]["cramer_rao_standard_error"] > 0
