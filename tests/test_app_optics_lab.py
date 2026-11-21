"""Optics Lab app (apps/optics_lab): HTTP smoke test of the routes and of the checks the page shows.

The imaging physics is tested in tests/test_physics_lithography.py; here the routes, the page and the three checks
the app repeats at the numbers on screen (clear-mask gain, modulation around the resolution limit, Abbe vs SOCS) are
exercised at a small grid."""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    sys.path.insert(0, str(ROOT / "apps" / "optics_lab"))
    sys.path.insert(0, str(ROOT / "apps" / "_shared"))
    try:
        from optics_lab.api import app
        yield TestClient(app)
    finally:
        sys.path.remove(str(ROOT / "apps" / "_shared"))
        sys.path.remove(str(ROOT / "apps" / "optics_lab"))


def test_the_page_and_meta_are_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "Optics Lab" in page.text
    assert "/shared/fieldview3d.js" in page.text
    meta = client.get("/api/meta").json()
    assert {"ArF imersão", "EUV"} <= set(meta["presets"])
    assert meta["grid"]["N"] == 128 and meta["grid"]["L_nm"] == 1024.0


def test_the_image_comes_with_its_checks_and_a_3d_view(client):
    r = client.post("/api/image", json={"wavelength_nm": 193.0, "na": 1.35, "sigma": 0.7,
                                        "shape": "conventional", "threshold": 0.3})
    assert r.status_code == 200, r.text
    d = r.json()
    m, c = d["metrics"], d["checks"]

    # the numbers on screen
    assert 40.0 < m["cd_nm"] < 140.0                     # a 64 nm line at just-printable pitch widens
    assert m["nils"] > 0.5
    assert 80.0 < m["resolution_limit_nm"] < 90.0        # lambda/(NA(1+sigma)) = 193/(1.35*1.7) = 84.1 nm

    # and the checks
    assert c["clear_mask_gain"]["ok"], c["clear_mask_gain"]
    assert c["abbe_vs_socs"]["ok"], c["abbe_vs_socs"]
    assert all(p["ok"] for p in c["modulation_around_the_limit"]), c["modulation_around_the_limit"]

    # the 3-D view is a glTF with the field as a vertex attribute, plus its meta
    glb = client.get(f"/api/view/{d['view']}.glb")
    assert glb.status_code == 200 and glb.content[:4] == b"glTF"
    meta = client.get(f"/api/view/{d['view']}").json()
    assert meta["schema"] == "pinneapple-field-view/v1"
    assert meta["fields"][0]["key"] == "f0"
    assert len(meta["notes"]) == 2
    txt = glb.content.decode("utf-8", "ignore")
    assert "_F0" in txt or struct.unpack("<I", glb.content[4:8])[0] == 2


def test_euv_resolves_the_same_clip_with_a_small_bias(client):
    r = client.post("/api/image", json={"wavelength_nm": 13.5, "na": 0.33, "sigma": 0.5,
                                        "shape": "conventional", "threshold": 0.3, "window": False})
    assert r.status_code == 200, r.text
    m = r.json()["metrics"]
    assert abs(m["cd_bias_nm"]) < 6.0                    # 64 nm lines at 192 nm pitch are easy for EUV
    assert m["resolution_limit_nm"] < 30.0


def test_a_rejected_input_is_422(client):
    assert client.post("/api/image", json={"na": 2.0}).status_code == 422
    assert client.post("/api/image", json={"wavelength_nm": 0.5}).status_code == 422


def test_an_expired_view_is_404(client):
    assert client.get("/api/view/nope.glb").status_code == 404
    assert client.get("/api/view/nope").status_code == 404
