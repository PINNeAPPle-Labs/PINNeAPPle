"""Design Optimization app (apps/design_optimization): HTTP smoke test of the routes and the page."""
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
    os.environ["DOPT_DATA_DIR"] = str(tmp_path_factory.mktemp("dopt"))
    sys.path.insert(0, str(ROOT / "apps" / "design_optimization"))
    from design_optimization.api import app
    with TestClient(app) as c:
        yield c


def test_health_and_page(client):
    assert client.get("/health").json()["status"] == "ok"
    page = client.get("/").text
    assert "An AI Redesigned This Wing" in page and "cdn.jsdelivr" not in page
    assert client.get("/docs").status_code == 200


def _wait(fn, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = fn()
        if r:
            return r
        time.sleep(0.3)
    raise TimeoutError


def test_scenario_runs_and_cuts_drag(client):
    r = client.get("/api/scenario", params={"seed": 1, "n_iterations": 60}).json()
    assert r["final"]["cd"] < r["baseline"]["cd"] and r["within_bounds"]
    assert len(r["airfoil_final"]["x"]) == 160 and r["n_iterations_run"] <= 60
    assert client.get("/api/scenario", params={"n_iterations": 0}).status_code == 422
