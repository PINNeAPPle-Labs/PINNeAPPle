"""Tests for the plate-fin heat-sink physics and the HeatSink Sizer app."""
from __future__ import annotations

import json
import math
import os
import sys

import pytest

from pinneapple_physics.closed_form import plate_fin_heatsink as pfh

APP = os.path.join(os.path.dirname(__file__), "..", "apps", "heatsink_sizer")
sys.path.insert(0, os.path.abspath(APP))

G80 = pfh.HeatSinkGeometry(0.08, 0.08, 0.006, 30, 0.0012, 0.035)


def op(v=2.0, q=100.0):
    return pfh.OperatingPoint(q, 25.0, v, 0.03, 0.03)


# ── closed-form physics ──────────────────────────────────────────────────────

def test_spreading_reduces_to_1d_when_source_covers_base():
    s = pfh.spreading_resistance(200.0, 0.005, 0.0064, 0.0064, 0.3)
    assert s["r_max"] == pytest.approx(s["r_1d"], rel=1e-9)


def test_spreading_grows_as_source_shrinks():
    r = [pfh.spreading_resistance(200.0, 0.005, a * a, 0.01, 0.3)["r_max"] for a in (0.08, 0.04, 0.01)]
    assert r[0] < r[1] < r[2]


def test_teertstra_limits():
    air = pfh.air_properties(25.0)
    g = pfh.HeatSinkGeometry(0.05, 0.3, 0.005, 40, 0.0005, 0.03)
    slow = pfh.forced_convection(g, 0.02, air)          # long channel, low flow: fully developed
    assert slow["nu_b"] == pytest.approx(slow["re_star"] * air["pr"] / 2, rel=0.02)
    g2 = pfh.HeatSinkGeometry(0.1, 0.02, 0.005, 5, 0.001, 0.03)
    fast = pfh.forced_convection(g2, 10.0, air)         # short channel, fast flow: developing plate
    dev = 0.664 * math.sqrt(fast["re_star"]) * air["pr"] ** (1 / 3)
    assert fast["nu_b"] == pytest.approx(dev, rel=0.10)


def test_natural_convection_has_an_optimal_fin_count():
    r = [pfh.evaluate(pfh.HeatSinkGeometry(0.08, 0.10, 0.005, n, 0.0015, 0.035),
                      pfh.OperatingPoint(15, 25, 0, 0.03, 0.03))["r_total_k_w"] for n in (4, 9, 20)]
    assert r[1] < r[0] and r[1] < r[2]      # too few fins: little area; too many: choked flow


def test_forced_resistance_falls_and_pressure_drop_rises_with_airflow():
    a, b = pfh.evaluate(G80, op(1.0)), pfh.evaluate(G80, op(4.0))
    assert b["r_total_k_w"] < a["r_total_k_w"]
    assert b["pressure_drop_pa"] > a["pressure_drop_pa"]


def test_geometry_errors_are_reported():
    bad = pfh.HeatSinkGeometry(0.02, 0.08, 0.006, 30, 0.0012, 0.035)
    with pytest.raises(ValueError, match="do not fit"):
        pfh.evaluate(bad, op())


# ── finite-volume base solve ─────────────────────────────────────────────────

def test_fvm_energy_balance_and_agreement_with_lee():
    from pinneapple_design.thermal.heatsink.field import solve_base_field
    f = solve_base_field(base_width=0.08, base_depth=0.08, base_thickness=0.006, k=209.0,
                         power_w=100.0, r_fins_k_w=0.2, source_width=0.03, source_depth=0.03)
    assert f["energy_balance_rel_error"] < 1e-8
    lee = pfh.spreading_resistance(209.0, 0.006, 0.03 ** 2, 0.08 ** 2, 0.2)["r_max"]
    assert f["theta_max"] / 100.0 - 0.2 == pytest.approx(lee, rel=0.05)


def test_fvm_off_centre_source_moves_hotspot():
    from pinneapple_design.thermal.heatsink.field import solve_base_field
    f = solve_base_field(base_width=0.1, base_depth=0.1, base_thickness=0.004, k=209.0, power_w=50.0,
                         r_fins_k_w=0.3, source_width=0.02, source_depth=0.02, source_x=0.02, source_y=0.08)
    x, y = f["hotspot_xy_m"]
    assert x < 0.04 and y > 0.06


# ── engine / sizer / API ─────────────────────────────────────────────────────

def test_evaluate_design_checks_and_band():
    from pinneapple_design.thermal.heatsink.engine import DesignInput, OperatingInput, evaluate_design
    r = evaluate_design(DesignInput(80, 80, 6, 30, 1.2, 35), OperatingInput(100, 25, 2, 30, 30, t_limit_c=70))
    lo, hi = r["kpis"]["t_source_band_c"]
    assert lo < r["kpis"]["t_source_max_c"] < hi
    assert r["verdict"]["status"] == "meets"
    assert all(c["status"] == "pass" for c in r["checks"])


def test_sizer_closed_form_screen_returns_verified_designs():
    from pinneapple_design.thermal.heatsink.sizer import SizingRequest, size
    res = size(SizingRequest(power_w=60, t_limit_c=75, n_candidates=1500, n_verify=6), surrogates={})
    json.dumps(res)                                   # plain JSON types only (no numpy scalars)
    assert res["status"] == "ok"
    assert all(r["meets_requirements"] for r in res["recommendations"])
    assert res["recommendations"][0]["kpis"]["t_source_band_c"][1] <= 75


def test_sizer_reports_infeasible_requests():
    from pinneapple_design.thermal.heatsink.sizer import SizingRequest, size
    res = size(SizingRequest(power_w=400, t_limit_c=30, max_width_mm=40, max_depth_mm=40,
                             max_height_mm=20, n_candidates=800, n_verify=4), surrogates={})
    assert res["status"] == "no_feasible_design"


def test_api_endpoints():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from heatsink_sizer.api import app
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    assert "al6063" in c.get("/api/meta").json()["materials"]
    body = {"design": {"base_width_mm": 80, "base_depth_mm": 80, "base_thickness_mm": 6, "n_fins": 30,
                       "fin_thickness_mm": 1.2, "fin_height_mm": 35},
            "operating": {"power_w": 100, "air_velocity_m_s": 2}}
    r = c.post("/api/evaluate", json=body)
    assert r.status_code == 200 and r.json()["kpis"]["t_source_max_c"] > 25
    body["design"]["n_fins"] = 80                      # fins no longer fit
    assert c.post("/api/evaluate", json=body).status_code == 422
    assert c.get("/").status_code == 200


def test_sizer_with_trained_surrogate_if_available():
    from pinneapple_design.thermal.heatsink.sizer import SizingRequest, size
    from pinneapple_design.thermal.heatsink.surrogate import load_surrogates
    surs = load_surrogates(os.path.join(APP, "artifacts"))
    if "forced" not in surs:
        pytest.skip("no trained surrogate artifact")
    res = size(SizingRequest(power_w=100, t_limit_c=70, n_candidates=20000, n_verify=8), surs)
    json.dumps(res)
    assert "surrogate" in res["search"]["screening"]["method"]
    assert res["status"] == "ok"
    assert res["search"]["screen_error_on_verified_c"]["mean_abs"] < 2.0


def test_field3d_is_physically_consistent():
    import numpy as np
    from pinneapple_design.thermal.heatsink.engine import DesignInput, OperatingInput, evaluate_design
    r = evaluate_design(DesignInput(80, 80, 6, 30, 1.2, 35), OperatingInput(100, 25, 2, 30, 30))
    f = r["field3d"]
    root = np.array(f["fins"]["root_c"])
    assert root.shape == (30, len(f["base"]["y"]))
    ratio = np.array(f["fin_profile"]["theta_ratio"])
    assert ratio[0] == pytest.approx(1.0) and np.all(np.diff(ratio) < 0)   # tip cooler than root
    top, bottom = np.array(f["base"]["top_c"]), np.array(f["base"]["bottom_c"])
    assert 25 < root.min() and root.max() <= bottom.max() + 1e-9          # heat flows source -> fins
    assert bottom.max() == pytest.approx(f["hotspot"]["t_c"], abs=1e-3)   # sent rounded to 3 dp
    assert top.max() < bottom.max()


def test_api_basic_auth_and_busy_limit(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    import base64
    import threading

    from fastapi.testclient import TestClient
    from heatsink_sizer import api
    monkeypatch.setattr(api.app.state, "auth_user", "eng")
    monkeypatch.setattr(api.app.state, "auth_password", "s3cret")
    c = TestClient(api.app)
    assert c.get("/health").status_code == 200                  # probes stay open
    assert c.get("/api/meta").status_code == 401
    good = {"Authorization": "Basic " + base64.b64encode(b"eng:s3cret").decode()}
    bad = {"Authorization": "Basic " + base64.b64encode(b"eng:nope").decode()}
    assert c.get("/api/meta", headers=bad).status_code == 401
    assert c.get("/api/meta", headers=good).status_code == 200
    monkeypatch.setattr(api._SIZING, "slots", threading.BoundedSemaphore(1))
    api._SIZING.slots.acquire()                                   # another run in progress
    r = c.post("/api/size", json={"power_w": 50, "t_limit_c": 80}, headers=good)
    assert r.status_code == 429


# ── independent cross-checks (different correlations than the model uses) ──

@pytest.mark.parametrize("v", [0.5, 1.0, 2.0, 4.0])
def test_forced_fin_resistance_matches_stephan_eps_ntu(v):
    """Teertstra (model) vs. Stephan developing-flow Nusselt + epsilon-NTU air heating."""
    g = pfh.HeatSinkGeometry(0.08, 0.08, 0.006, 24, 0.0012, 0.035)
    r = pfh.evaluate(g, pfh.OperatingPoint(100, 25, v, 0.03, 0.03))
    air = pfh.air_properties(25 + 0.5 * (r["t_fin_root_c"] - 25))
    b, H, L = g.fin_gap, g.fin_height, g.base_depth
    dh = 2 * b
    re = v * g.base_width / ((g.n_fins - 1) * b) * dh / air["nu"]
    xs = L / (dh * re * air["pr"])
    h = (7.55 + 0.024 * xs ** -1.14 / (1 + 0.0358 * air["pr"] ** 0.17 * xs ** -0.64)) * air["k"] / dh
    k = pfh.MATERIALS[g.material]["k"]
    lc = H + g.fin_thickness / 2
    ml = math.sqrt(2 * h / (k * g.fin_thickness)) * lc
    ha = h * (math.tanh(ml) / ml * 2 * lc * L * g.n_fins + (g.base_width - g.n_fins * g.fin_thickness) * L)
    mcp = air["rho"] * v * g.base_width * H * air["cp"]
    r_ind = 1 / (mcp * (1 - math.exp(-ha / mcp)))
    assert r["resistances_k_w"]["fins_convection"] == pytest.approx(r_ind, rel=0.08)


def test_natural_convection_matches_elenbaas_and_optimum_spacing():
    op = pfh.OperatingPoint(20, 25, 0, 0.03, 0.03)
    g = pfh.HeatSinkGeometry(0.1, 0.1, 0.005, 12, 0.002, 0.03)
    r = pfh.evaluate(g, op)
    el = r["dimensionless"]["elenbaas"]
    air = pfh.air_properties(25 + 0.5 * (r["t_fin_root_c"] - 25))
    h_elenbaas = el / 24 * (1 - math.exp(-35 / el)) ** 0.75 * air["k"] / g.fin_gap
    assert r["h_w_m2k"] == pytest.approx(h_elenbaas, rel=0.05)
    dt = r["t_fin_root_c"] - 25
    s_opt = 2.714 * (0.1 * air["nu"] * air["alpha"] / (9.81 * air["beta"] * dt)) ** 0.25
    best = min(range(5, 40), key=lambda n: pfh.evaluate(
        pfh.HeatSinkGeometry(0.1, 0.1, 0.005, n, 0.002, 0.03), op)["r_total_k_w"])
    gap = (0.1 - best * 0.002) / (best - 1)
    assert gap == pytest.approx(s_opt, rel=0.15)


def test_report_scope_is_specific_to_the_cooling_mode():
    from pinneapple_design.thermal.heatsink.engine import DesignInput, OperatingInput, evaluate_design
    forced = evaluate_design(DesignInput(80, 80, 6, 30, 1.2, 35), OperatingInput(100, 25, 2, 30, 30))
    natural = evaluate_design(DesignInput(100, 100, 5, 12, 2, 30), OperatingInput(20, 25, 0, 30, 30))
    topics = lambda r: {i["topic"]: i["effect"] for i in r["scope"]["items"]}  # noqa: E731
    assert topics(forced)["Air bypass"] == "optimistic" and "Thermal radiation" not in topics(forced)
    assert topics(natural)["Thermal radiation"] == "conservative" and "Air bypass" not in topics(natural)
    for r in (forced, natural):
        assert len(r["scope"]["validated"]) >= 4
        assert all(i["today"] and i["planned"] for i in r["scope"]["items"])
        assert not any(w.startswith(("Radiation is neglected", "Assumes")) for w in r["warnings"])
