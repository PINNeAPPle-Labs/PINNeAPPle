"""The standard satellite-design workflow (#1207): SSO inclination, eclipse geometry, link identities, the closed
design and its propagation-based re-verification."""
import math

import pytest

from pinneapple_design.satellite import budgets as B
from pinneapple_design.satellite import propagate_j2, satellite_design_workflow


def _status(r, code):
    return next(c["status"] for c in r["checks"] if c["code"] == code)


@pytest.fixture(scope="module")
def default_run():
    pytest.importorskip("pymsis")                    # the F10.7 scenarios need NRLMSIS (pinneapple[sky])
    return satellite_design_workflow()


def test_default_design_closes_and_every_check_passes(default_run):
    r = default_run
    assert r["verdict"] == "pass", r["failed"]
    d = r["design"]
    assert 600 <= d["h_km"] <= 750 and d["revisit"]["gsd_m"] <= 6.0
    assert r["lifetime_years"]["high"] >= 3.0 and r["lifetime_years"]["mean"] <= 25.0
    assert d["link"]["margin_db"] >= 3.0
    assert min(t["mass_kg"] for t in r["trade"] if not t["infeasible"]) == pytest.approx(d["mass_kg"])


def test_sun_synchronous_inclination_and_node_drift():
    for h in (500.0, 700.0, 800.0):
        i = B.sso_inclination(h)
        assert abs(B.nodal_rate(B.R_E + h * 1e3, 0.0, math.radians(i)) - B.OMEGA_SUN) < 1e-20
    assert 97.3 < B.sso_inclination(500) < 97.5 and 98.5 < B.sso_inclination(800) < 98.7
    with pytest.raises(ValueError):
        B.sso_inclination(8000.0)


def test_eclipse_fraction_closed_form_matches_geometry():
    for h in (400.0, 700.0):
        for beta in (0.0, 20.0, 50.0, 70.0):
            assert abs(B.eclipse_fraction(h, beta) - B.eclipse_fraction_sampled(h, beta)) < 1e-4
    assert B.eclipse_fraction(700.0, 80.0) == 0.0


def test_propagated_node_drift_and_eclipse():
    p = propagate_j2(700.0, B.sso_inclination(700.0), 1.0, beta_deg=10.0)
    assert abs(p["raan_rate_rad_s"] / B.OMEGA_SUN - 1) < 0.02
    assert abs(p["eclipse_fraction"] - B.eclipse_fraction(p["alt_km_mean"], 10.0)) < 2e-3


def test_link_budget_free_space_loss_and_scaling():
    a = B.link_budget(p_tx_w=1.0, f_hz=437e6, h_km=500, elev_deg=90.0, g_tx_dbi=0.0, g_over_t_db_k=-20.0,
                      rate_bps=9600, ebn0_req_db=10.0)
    assert a["range_km"] == pytest.approx(500.0)
    assert a["fspl_db"] == pytest.approx(20 * math.log10(4 * math.pi * 500e3 * 437e6 / B.C_LIGHT))
    b = B.link_budget(p_tx_w=2.0, f_hz=437e6, h_km=500, elev_deg=90.0, g_tx_dbi=0.0, g_over_t_db_k=-20.0,
                      rate_bps=9600, ebn0_req_db=10.0)
    assert b["ebn0_db"] - a["ebn0_db"] == pytest.approx(10 * math.log10(2.0))


def test_thermal_radiator_and_battery_rules():
    hot = B.thermal_cases((0.1, 0.2, 0.3), 600, B.surface(B.COATINGS[0], 1.0), 10.0, 5.0)
    mli = B.thermal_cases((0.1, 0.2, 0.3), 600, B.surface(B.COATINGS[0], 0.1), 10.0, 5.0)
    assert mli["t_cold_c"] > hot["t_cold_c"]                     # less radiator keeps the cold case warmer
    assert B.battery(10, 5800, 0.35, 1)["dod"] > B.battery(10, 5800, 0.35, 5)["dod"]   # more cycles, less DoD


def test_unreachable_mission_is_reported():
    pytest.importorskip("pymsis")
    r = satellite_design_workflow(mission={"gsd_max_m": 2.0})
    assert r["verdict"] == "fail" and r["failed"] == ["design.feasible"]


def test_the_planner_proposes_the_workflow_in_portuguese_and_english():
    import pinneapple_flow.nodes_ext  # noqa: F401
    from pinneapple_flow.planner.extensions import match
    for text in ("projeto de cubesat em órbita heliossíncrona com orçamento de enlace",
                 "CubeSat design: link budget, reaction wheel sizing and orbit decay"):
        assert match(text)[0][1].id == "satellite.design"
