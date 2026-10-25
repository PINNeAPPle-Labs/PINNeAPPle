"""PatchDose AI app (apps/patchdose_ai): HTTP smoke test of every route with a quickly trained surrogate.

The surrogate's accuracy and the optimizer are tested in tests/pinneapple_design/; here PATCH_EPOCHS is small so
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
    old = os.environ.get("PATCH_EPOCHS")
    os.environ["PATCH_EPOCHS"] = "300"
    sys.path.insert(0, str(ROOT / "apps" / "patchdose_ai"))
    try:
        from patchdose_ai.api import app
        yield TestClient(app)
    finally:
        sys.path.remove(str(ROOT / "apps" / "patchdose_ai"))
        if old is None:
            os.environ.pop("PATCH_EPOCHS", None)
        else:
            os.environ["PATCH_EPOCHS"] = old


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
    assert page.status_code == 200 and "PatchDose AI" in page.text
    assert "cdn." not in page.text and "/vendor/three/three.module.min.js" in page.text
    assert "script-src 'self'" in page.headers["content-security-policy"]
    assert client.get("/shared/stl.js").status_code == 200
    assert client.get("/vendor/three/OrbitControls.js").status_code == 200


def test_mesh_is_binary_stl(client):
    stl = client.post("/api/mesh.stl", json={"area_m2": 0.002, "membrane_thickness_m": 8e-5, "reservoir_conc_kg_m3": 60}).content
    n = struct.unpack("<I", stl[80:84])[0]
    assert stl[:5] != b"solid" and n > 0 and len(stl) == 84 + 50 * n


def test_evaluate_and_simulate(client):
    body = {"area_m2": 0.002, "membrane_thickness_m": 8e-5, "reservoir_conc_kg_m3": 60}
    r = client.post("/api/evaluate", json=body).json()
    assert r["release_rate_mg_day"] == pytest.approx(64.8) and r["depletion_time_days"] == pytest.approx(1.85185, rel=1e-4)
    s = client.post("/api/simulate", json={**body, "n_points": 50}).json()
    assert len(s["timeseries"]) == 50 and s["duration_days"] == pytest.approx(1.5 * r["depletion_time_days"])
    assert client.post("/api/evaluate", json={**body, "area_m2": -1}).status_code == 400


def test_optimize_returns_a_verified_design(client):
    _trained(client)
    assert client.post("/api/optimize", json={"target_rate_mg_day": 0.1, "target_wear_days": 2}).status_code == 400
    r = client.post("/api/optimize", json={"target_rate_mg_day": 100, "target_wear_days": 2}).json()
    v = r["verification"]
    assert v["overall_verdict"] in ("PASS", "WARNING", "FAIL") and v["target_rate_mg_day"] == 100
    assert 5 <= r["optimized"]["design"]["area_cm2"] <= 40
