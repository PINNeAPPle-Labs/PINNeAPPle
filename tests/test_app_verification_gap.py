"""Verification Gap app (apps/verification_gap): HTTP smoke test of the routes and the page."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["VGAP_DATA_DIR"] = str(tmp_path_factory.mktemp("vgap"))
    sys.path.insert(0, str(ROOT / "apps" / "verification_gap"))
    from verification_gap.api import app
    with TestClient(app) as c:
        yield c


def test_health_and_page(client):
    assert client.get("/health").json()["status"] == "ok"
    page = client.get("/").text
    assert "The Design That Almost Passed" in page and "cdn.jsdelivr" not in page
    assert client.get("/docs").status_code == 200


def _wait(fn, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = fn()
        if r:
            return r
        time.sleep(0.3)
    raise TimeoutError


def test_scenario_and_mesh(client):
    s = client.get("/api/scenario", params={"seed": 42}).json()
    assert s["nominal_c"] == pytest.approx(495.9025, abs=1e-3) and len(s["samples_c"]) == 200
    assert s["summary"]["verdict"] == "MARGINAL" and s["verification"]["decision"] == "REVIEW"
    assert len(s["verification"]["checks"]) == 6
    assert s["comparison"]["highway"]["temp_c"] < s["comparison"]["city"]["temp_c"]
    assert client.get("/api/scenario", params={"seed": 42}).json()["samples_c"] == s["samples_c"]
    m = client.get("/api/mesh.json").json()
    assert len(m["positions"]) % 3 == 0 and max(m["indices"]) < len(m["positions"]) // 3
