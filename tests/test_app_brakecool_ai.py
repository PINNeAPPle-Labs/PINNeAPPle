"""BrakeCool AI app (apps/brakecool_ai): HTTP smoke test of every route with a quickly trained surrogate.

The surrogate's accuracy and the optimizer are tested in tests/pinneapple_design/; here BRAKE_EPOCHS is small so
the background training finishes in seconds and the routes, the page and the verification card are exercised."""
from __future__ import annotations

import os
import struct
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    old = os.environ.get("BRAKE_EPOCHS")
    os.environ["BRAKE_EPOCHS"] = "300"
    sys.path.insert(0, str(ROOT / "apps" / "brakecool_ai"))
    try:
        from brakecool_ai.api import app
        yield TestClient(app)
    finally:
        sys.path.remove(str(ROOT / "apps" / "brakecool_ai"))
        if old is None:
            os.environ.pop("BRAKE_EPOCHS", None)
        else:
            os.environ["BRAKE_EPOCHS"] = old


def _trained(c, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = c.get("/api/status").json()
        assert s["training_error"] is None, s["training_error"]
        if s["surrogate_trained"]:
            return s
        time.sleep(0.2)
    raise TimeoutError("surrogate training did not finish")


def test_health_page_and_static(client):
    assert client.get("/health").json()["status"] == "ok"
    page = client.get("/")
    assert page.status_code == 200 and "BrakeCool AI" in page.text
    assert "cdn." not in page.text and "/vendor/three/three.module.min.js" in page.text
    assert "script-src 'self'" in page.headers["content-security-policy"]
    assert client.get("/shared/stl.js").status_code == 200
    assert client.get("/vendor/three/OrbitControls.js").status_code == 200


def test_mesh_is_binary_stl(client):
    stl = client.post("/api/mesh.stl", json={"vane_count": 36, "vane_gap_m": 0.008, "vane_thickness_m": 0.004}).content
    n = struct.unpack("<I", stl[80:84])[0]
    assert stl[:5] != b"solid" and n > 0 and len(stl) == 84 + 50 * n


def test_evaluate_uses_the_formula(client):
    r = client.post("/api/evaluate", json={"vane_count": 36, "vane_gap_m": 0.008, "vane_thickness_m": 0.004,
                                           "power_w": 800, "cooling_profile": "highway"}).json()
    assert r["rotor_temperature_c"] == pytest.approx(112.4, abs=0.1) and r["violated_constraints"] == []
    assert client.post("/api/evaluate", json={"vane_count": 36, "vane_gap_m": 0.008, "vane_thickness_m": 0.004,
                                              "power_w": 5000}).status_code == 400


def test_uncertainty(client):
    r = client.post("/api/uncertainty", json={"vane_count": 36, "vane_gap_m": 0.008, "vane_thickness_m": 0.004,
                                              "power_w": 800, "cooling_profile": "highway", "n_samples": 100}).json()
    assert r["verdict"] == "ROBUST" and r["n_samples"] == 100 and r["std_c"] > 0
    assert r["tolerances"]["vane_gap_tolerance_m"] == pytest.approx(1e-4)


def test_optimize_returns_a_verified_design(client):
    _trained(client)
    r = client.post("/api/optimize", json={"power_w": 800, "cooling_profile": "highway"}).json()
    v = r["verification"]
    assert v["overall_verdict"] in ("PASS", "WARNING", "FAIL") and 0 < v["coverage"] < 1
    assert r["optimized"]["design"]["vane_spacing_m"] >= 0.006 - 1e-9
