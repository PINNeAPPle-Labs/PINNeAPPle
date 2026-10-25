"""RideTune AI app (apps/ridetune_ai): HTTP smoke test of every route with a quickly trained surrogate.

The surrogate's accuracy and the optimizer are tested in tests/pinneapple_design/; here RIDE_EPOCHS is small so
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
    old = os.environ.get("RIDE_EPOCHS")
    os.environ["RIDE_EPOCHS"] = "300"
    sys.path.insert(0, str(ROOT / "apps" / "ridetune_ai"))
    try:
        from ridetune_ai.api import app
        yield TestClient(app)
    finally:
        sys.path.remove(str(ROOT / "apps" / "ridetune_ai"))
        if old is None:
            os.environ.pop("RIDE_EPOCHS", None)
        else:
            os.environ["RIDE_EPOCHS"] = old


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
    assert page.status_code == 200 and "RideTune AI" in page.text
    assert "cdn." not in page.text and "/vendor/three/three.module.min.js" in page.text
    assert "script-src 'self'" in page.headers["content-security-policy"]
    assert client.get("/shared/stl.js").status_code == 200
    assert client.get("/vendor/three/OrbitControls.js").status_code == 200


def test_mesh_is_binary_stl(client):
    stl = client.post("/api/mesh.stl", json={"width_m": 0.05, "thickness_m": 0.022}).content
    n = struct.unpack("<I", stl[80:84])[0]
    assert stl[:5] != b"solid" and n > 0 and len(stl) == 84 + 50 * n


def test_evaluate_and_sensitivity(client):
    r = client.post("/api/evaluate", json={"width_m": 0.05, "thickness_m": 0.022, "load_n": 1200}).json()
    assert r["bending_stress_pa"] == pytest.approx(6 * 1200 * 0.35 / (0.05 * 0.022 ** 2))
    s = client.post("/api/sensitivity", json={"width_m": 0.05, "thickness_m": 0.022, "load_n": 1200}).json()
    assert s["dominant_variable"] == "thickness_m" and s["elasticity_ratio"] == pytest.approx(2.0, rel=1e-4)
    assert client.post("/api/sensitivity", json={"width_m": 0.5, "thickness_m": 0.022}).status_code == 400


def test_optimize_returns_a_verified_design(client):
    _trained(client)
    r = client.post("/api/optimize", json={"load_n": 1200, "target_life_cycles": 1e6}).json()
    v = r["verification"]
    assert v["overall_verdict"] in ("PASS", "WARNING", "FAIL") and v["target_life_cycles"] == 1e6
    assert client.post("/api/optimize", json={"load_n": 1200, "target_life_cycles": 1e9}).status_code == 400
