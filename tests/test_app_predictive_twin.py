"""Predictive Twin app (apps/predictive_twin): HTTP smoke test of the routes and the page."""
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
    os.environ["PTWIN_DATA_DIR"] = str(tmp_path_factory.mktemp("ptwin"))
    os.environ["PTWIN_OFFLINE"] = "1"
    sys.path.insert(0, str(ROOT / "apps" / "predictive_twin"))
    from predictive_twin.api import app
    with TestClient(app) as c:
        yield c


def test_health_and_page(client):
    assert client.get("/health").json()["status"] == "ok"
    page = client.get("/").text
    assert "A River's Own Forecast" in page and "cdn.jsdelivr" not in page
    assert client.get("/docs").status_code == 200


def _wait(fn, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = fn()
        if r:
            return r
        time.sleep(0.3)
    raise TimeoutError


def test_offline_scenario_is_labelled_synthetic_and_detects_the_step(client):
    s = client.get("/api/scenario").json()
    assert s["synthetic"] and "SYNTHETIC" in s["gauge"]["site_name"]
    assert s["verdict"] == "DIVERGENCE_DETECTED" and s["detection"]["direction"] == "rising_above_forecast"
    assert len(s["timeseries"]["discharge_cfs"]) == len(s["forecast"]["discharge_cfs"])
    assert s["calibration"]["q0_cfs"] > 0 and s["water_level"]["current_ft"] > 0


@pytest.mark.skipif(not os.environ.get("PINNEAPPLE_NETWORK"), reason="set PINNEAPPLE_NETWORK=1 (live USGS API)")
def test_live_gauge(client, monkeypatch):
    monkeypatch.setenv("PTWIN_OFFLINE", "0")
    r = client.get("/api/scenario")
    assert r.status_code in (200, 502)
    if r.status_code == 200:
        assert not r.json()["synthetic"] and r.json()["gauge"]["site_id"] == "01646500"
