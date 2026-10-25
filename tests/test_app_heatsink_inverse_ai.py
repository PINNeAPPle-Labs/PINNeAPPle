"""HeatSink Design AI app (apps/heatsink_inverse_ai): HTTP smoke test of every route with a quickly trained surrogate.

The surrogate's accuracy and the optimizer are tested in tests/pinneapple_design/; here HSD_EPOCHS is small so
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
    old = os.environ.get("HSD_EPOCHS")
    os.environ["HSD_EPOCHS"] = "300"
    sys.path.insert(0, str(ROOT / "apps" / "heatsink_inverse_ai"))
    try:
        from heatsink_inverse_ai.api import app
        yield TestClient(app)
    finally:
        sys.path.remove(str(ROOT / "apps" / "heatsink_inverse_ai"))
        if old is None:
            os.environ.pop("HSD_EPOCHS", None)
        else:
            os.environ["HSD_EPOCHS"] = old


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
    assert page.status_code == 200 and "HeatSink Design AI" in page.text
    assert "cdn." not in page.text and "/vendor/three/three.module.min.js" in page.text
    assert "script-src 'self'" in page.headers["content-security-policy"]
    assert client.get("/shared/stl.js").status_code == 200
    assert client.get("/vendor/three/OrbitControls.js").status_code == 200


def test_mesh_is_binary_stl(client):
    stl = client.post("/api/mesh.stl", json={"fin_count": 20, "fin_height_m": 0.025, "fin_thickness_m": 0.001}).content
    n = struct.unpack("<I", stl[80:84])[0]
    assert stl[:5] != b"solid" and n > 0 and len(stl) == 84 + 50 * n


def test_evaluate_uses_the_formula(client):
    r = client.post("/api/evaluate", json={"fin_count": 20, "fin_height_m": 0.025, "fin_thickness_m": 0.001,
                                           "power_w": 65, "cooling_profile": "standard_fan"}).json()
    assert r["junction_temperature_c"] == pytest.approx(48.118, abs=1e-3) and r["violated_constraints"] == []
    assert client.post("/api/evaluate", json={"fin_count": 20, "fin_height_m": 0.025, "fin_thickness_m": 0.001,
                                              "power_w": 65, "cooling_profile": "nope"}).status_code == 400
    tight = client.post("/api/evaluate", json={"fin_count": 55, "fin_height_m": 0.02, "fin_thickness_m": 0.0028}).json()
    assert "min_spacing" in tight["violated_constraints"]


def test_cross_check(client):
    r = client.post("/api/cross_check", json={"fin_count": 20, "fin_height_m": 0.025, "fin_thickness_m": 0.001,
                                              "power_w": 65}).json()
    assert r["agreement"]["t_junction_relative_diff"] < 0.01
    assert r["finite_volume"]["energy_balance_rel_error"] < 1e-9
    assert client.post("/api/cross_check", json={"fin_count": 55, "fin_height_m": 0.02,
                                                 "fin_thickness_m": 0.0028}).status_code == 400


def test_optimize_returns_a_verified_design(client):
    _trained(client)
    r = client.post("/api/optimize", json={"power_w": 65, "cooling_profile": "standard_fan"}).json()
    v = r["verification"]
    assert v["overall_verdict"] in ("PASS", "WARNING", "FAIL") and v["trust"]["decision"] in ("APPROVED", "REVIEW",
                                                                                               "REJECT")
    assert 0 < v["coverage"] < 1 and v["real_temperature_c"] > 25
    assert r["optimized"]["design"]["fin_spacing_m"] >= 0.0012 - 1e-9
