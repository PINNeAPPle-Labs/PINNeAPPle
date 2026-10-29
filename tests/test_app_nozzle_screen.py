"""NozzleScreen app (apps/nozzle_screen): the analytic estimate (same numbers as the source app's isentropic.py on its
README motor), input validation, and a short PINN screen job whose report matches its own numbers."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MOTOR = {"throat_radius_m": 0.05, "exit_radius_m": 0.12, "p_chamber_pa": 4e6, "T_chamber_k": 2800,
         "gamma": 1.2, "R_gas_j_per_kgk": 345}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["NOZZLE_DATA_DIR"] = str(tmp_path_factory.mktemp("nozzle"))
    sys.path.insert(0, str(ROOT / "apps" / "nozzle_screen"))
    from nozzle_screen.api import app
    return TestClient(app)


def test_analytic_estimate(client):
    assert client.get("/health").json()["status"] == "ok"
    a = client.get("/v1/analytical_only", params=MOTOR).json()["analytic"]
    assert a["mach_exit"] == pytest.approx(2.8879, abs=1e-3) and a["specific_impulse_s"] == pytest.approx(234.97, abs=0.05)
    bad = dict(MOTOR, area_ratio=4.0)
    assert client.get("/v1/analytical_only", params=bad).status_code == 422


@pytest.mark.skipif(not os.environ.get("PINNEAPPLE_SLOW"), reason="slow: trains a 64x4 PINN (set PINNEAPPLE_SLOW=1)")
def test_screen_job_reports_its_disagreement(client):
    job = client.post("/v1/screen_nozzle", json={**MOTOR, "epochs": 10, "n_collocation": 128,
                                                 "n_bc_per_face": 40}).json()
    t0 = time.time()
    while job["status"] not in ("done", "failed") and time.time() - t0 < 900:
        time.sleep(0.5)
        job = client.get(f"/v1/jobs/{job['id']}").json()
    assert job["status"] == "done", job.get("error")
    r = job["result"]
    assert r["disagreement"]["methods_agree"] == (r["disagreement"]["mach_exit_relative"] <= 0.3
                                                  and r["disagreement"]["p_exit_relative"] <= 0.3)
    assert r["verdict"] == "agree" or r["verdict"].startswith("do not trust")
