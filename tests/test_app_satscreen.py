"""SatScreen app (apps/satscreen): the source README's worked conjunction example and J2 propagation through HTTP, and
the description mode with the LLM stubbed (an out-of-range draft is refused, a sane one is used)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = {"mode": "elements", "chief": {"a_km": 6771.0},
           "deputy_relative_state": {"x0_km": 0.05, "y0_km": 1.0, "z0_km": 0.1, "vx0_km_s": 0.0002,
                                     "vy0_km_s": -0.0019, "vz0_km_s": 0.00005},
           "combined_hard_body_radius_km": 0.005, "combined_position_sigma_km": 0.5}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["SATSCREEN_DATA_DIR"] = str(tmp_path_factory.mktemp("sat"))
    sys.path.insert(0, str(ROOT / "apps" / "satscreen"))
    from satscreen.api import app
    return TestClient(app)


def test_worked_example(client):
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/v1/screen_conjunction", json=EXAMPLE)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["closest_approach"]["miss_distance_km"] == pytest.approx(0.3729, abs=1e-4)
    assert out["closest_approach"]["t_tca_s"] == pytest.approx(423.5, abs=0.5)
    assert out["collision_probability"] == pytest.approx(3.79e-5, rel=5e-3) and not out["hard_body_violated"]
    assert client.post("/v1/screen_conjunction", json={"mode": "elements"}).status_code == 422


def test_j2_propagation(client):
    r = client.post("/v1/propagate_j2", json={"satellite": {"a_km": 7000.0, "e": 0.001, "inclination_deg": 98.7},
                                              "duration_orbits": 1.0, "n_points": 50})
    assert r.status_code == 200, r.text
    out = r.json()
    lo, hi = out["radius_range_km"]
    assert 6980 < lo < hi < 7020 and len(out["trajectory"]) == 50


def test_description_mode_checks_the_drafted_values(client, monkeypatch):
    import pinneapple_llm as pl
    from pinneapple_llm import draft as draft_mod
    from pinneapple_llm.draft import DraftResult

    seq = iter([DraftResult("satellite_j2_perturbation", {"mu": "398600441800000", "e": 1}, "bad units", ""),
                DraftResult("satellite_j2_perturbation", {"a": 7000.0, "e": 0.001}, "ok", "")])
    monkeypatch.setattr(draft_mod, "draft_problem", lambda d, **k: next(seq))
    r = client.post("/v1/propagate_j2", json={"mode": "description", "description": "a sun-synchronous LEO orbit",
                                              "n_points": 5})
    assert r.status_code == 200 and r.json()["llm_reasoning"] == "ok"
    monkeypatch.setattr(draft_mod, "draft_problem",
                        lambda d, **k: DraftResult("kepler_two_body_orbit", {}, "wrong preset", ""))
    r = client.post("/v1/screen_conjunction", json={"mode": "description", "description": "two satellites",
                                                    "max_attempts": 2})
    assert r.status_code == 502 and "kepler_two_body_orbit" in r.json()["detail"]
    assert pl.DraftRejected
