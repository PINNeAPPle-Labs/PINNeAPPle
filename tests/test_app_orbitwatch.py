"""OrbitWatch app (apps/orbitwatch): the source app's tests/test_app.py (upload path end to end, ordering, refusals,
metering through appkit) and the Earth Engine path refusing without credentials."""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("jinja2")
Image = pytest.importorskip("PIL.Image")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["ORBITWATCH_DATA_DIR"] = str(tmp_path_factory.mktemp("orbit"))
    os.environ["ORBITWATCH_FREE_REPORTS"] = "8"
    sys.path.insert(0, str(ROOT / "apps" / "orbitwatch"))
    from orbitwatch.api import app
    return TestClient(app)


def _png(rgb, shape=(32, 32)) -> bytes:
    arr = np.zeros((*shape, 3), dtype=np.uint8)
    arr[:, :] = rgb
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _post(client, pngs, dates):
    return client.post("/v1/detect_changes_upload", files=[("images", (f"{i}.png", p, "image/png"))
                                                           for i, p in enumerate(pngs)], data={"dates": dates})


def test_health_and_reference(client):
    assert client.get("/health").json()["status"] == "ok"
    ref = client.get("/v1/threshold_reference").json()
    assert ref["default_reflectance_scale"] == 10000.0 and ref["preview_stretch"] == 3000.0


def test_no_change_and_change(client):
    same = _png((120, 130, 140))
    r = _post(client, [same, same], ["2024-01-01", "2024-06-01"])
    assert r.status_code == 200 and "No significant change" in r.text and "data:image/png;base64," in r.text
    r = _post(client, [_png((50, 120, 60)), _png((220, 40, 30))], ["2024-01-01", "2024-06-01"])
    assert r.status_code == 200 and "No significant change" not in r.text and "significant" in r.text.lower()


def test_sorted_by_date_not_upload_order(client):
    r = _post(client, [_png((220, 40, 30)), _png((50, 120, 60))], ["2024-06-01", "2024-01-01"])
    assert r.text.index("2024-01-01") < r.text.index("2024-06-01")


def test_earth_engine_without_credentials(client, monkeypatch):
    for k in ("EE_PROJECT_ID", "EE_SERVICE_ACCOUNT_EMAIL", "EE_SERVICE_ACCOUNT_KEY_FILE"):
        monkeypatch.delenv(k, raising=False)
    r = client.post("/v1/detect_changes", json={"lon": -47.87, "lat": -15.61})
    assert r.status_code == 503
    d = r.json()["detail"]
    assert "EE_SERVICE_ACCOUNT_EMAIL" in d or "not installed" in d
    assert client.post("/v1/detect_changes", json={"lat": 1.0}).status_code == 422


def test_refusals_and_metering(client):
    p = _png((10, 10, 10))
    assert _post(client, [p, p], ["2024-01-01"]).status_code == 422
    assert _post(client, [_png((1, 1, 1), (16, 16)), p], ["2024-01-01", "2024-02-01"]).status_code == 422
    assert _post(client, [p], ["2024-01-01"]).status_code == 422
    codes = [_post(client, [p, p], ["2024-01-01", "2024-02-01"]).status_code for _ in range(6)]
    assert 429 in codes and codes[0] == 200                       # 8 free reports per month in this test
