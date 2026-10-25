"""RoverMobility app (apps/rover_mobility): the soil guide comes from the library's sourced soils, and a short
analysis job returns the per-constraint checks next to the guardrail."""
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
    os.environ["ROVERMOB_DATA_DIR"] = str(tmp_path_factory.mktemp("rover"))
    sys.path.insert(0, str(ROOT / "apps" / "rover_mobility"))
    from rover_mobility.api import app
    return TestClient(app)


def test_soil_guide(client):
    assert client.get("/health").json()["status"] == "ok"
    soils = client.get("/v1/soil_parameter_guide").json()["soils"]
    grc = next(s for s in soils if s["name"] == "GRC-1")
    assert grc["c_Pa"] == 250.0 and "Oravec" in grc["sources"]["c"]


@pytest.mark.skipif(not os.environ.get("PINNEAPPLE_SLOW"), reason="slow: trains a 64x4 PINN (set PINNEAPPLE_SLOW=1)")
def test_analysis_job(client):
    job = client.post("/v1/analyze_mobility", json={"epochs": 20, "n_collocation": 128, "n_slip": 6}).json()
    t0 = time.time()
    while job["status"] not in ("done", "failed") and time.time() - t0 < 900:
        time.sleep(0.5)
        job = client.get(f"/v1/jobs/{job['id']}").json()
    assert job["status"] == "done", job.get("error")
    r = job["result"]
    assert len(r["per_slip"]) == 6 and set(r["constraint_checks"]) == {
        "r1_zero_pull_at_zero_slip", "r2_mohr_coulomb", "r3_monotone_up_to_slip_0_4", "r4_torque_covers_thrust"}
    assert r["guardrail"]["checks"] and r["verdict"]
    assert client.post("/v1/analyze_mobility", json={"phi_deg": 95}).status_code == 422
