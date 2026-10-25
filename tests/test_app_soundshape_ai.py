"""SoundShape AI app (apps/soundshape_ai): HTTP smoke test of every route with a quickly trained surrogate.

The surrogate's accuracy and the optimizer are tested in tests/pinneapple_design/; here SOUND_EPOCHS is small so
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
    old = os.environ.get("SOUND_EPOCHS")
    os.environ["SOUND_EPOCHS"] = "300"
    sys.path.insert(0, str(ROOT / "apps" / "soundshape_ai"))
    try:
        from soundshape_ai.api import app
        yield TestClient(app)
    finally:
        sys.path.remove(str(ROOT / "apps" / "soundshape_ai"))
        if old is None:
            os.environ.pop("SOUND_EPOCHS", None)
        else:
            os.environ["SOUND_EPOCHS"] = old


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
    assert page.status_code == 200 and "SoundShape AI" in page.text
    assert "cdn." not in page.text and "/vendor/three/three.module.min.js" in page.text
    assert "script-src 'self'" in page.headers["content-security-policy"]
    assert client.get("/shared/stl.js").status_code == 200
    assert client.get("/vendor/three/OrbitControls.js").status_code == 200


def test_mesh_is_binary_stl(client):
    stl = client.post("/api/mesh.stl", json={"radius_m": 0.025, "length_m": 0.1}).content
    n = struct.unpack("<I", stl[80:84])[0]
    assert stl[:5] != b"solid" and n > 0 and len(stl) == 84 + 50 * n


def test_evaluate_uses_the_formula(client):
    r = client.post("/api/evaluate", json={"radius_m": 0.025, "length_m": 0.1, "cabinet_volume_l": 20}).json()
    assert 30 < r["resonant_frequency_hz"] < 60 and r["acoustic_field"]["freq_min_hz"] == 22.0
    assert client.post("/api/evaluate", json={"radius_m": 0.025, "length_m": 0.1,
                                              "cabinet_volume_l": 500}).status_code == 400


def test_optimize_and_frontier(client):
    _trained(client)
    r = client.post("/api/optimize", json={"cabinet_volume_l": 20, "target_freq_hz": 40}).json()
    assert r["verification"]["overall_verdict"] in ("PASS", "WARNING", "FAIL")
    f = client.post("/api/pareto_frontier", json={"cabinet_volume_l": 20, "target_freq_hz": 40, "n_points": 3}).json()
    assert len(f["frontier"]) == 3 and all(p["port_volume_m3"] > 0 for p in f["frontier"])
