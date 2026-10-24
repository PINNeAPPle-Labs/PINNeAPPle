"""Rover autolab: mission parsing, operating points, multi-soil multibody runs, soil learning and risk planning."""
import numpy as np
import pytest

pytest.importorskip("torch")

from pinneapple_simulation.rover.autolab import (  # noqa: E402
    RiskPlanner,
    SoilBelief,
    demo_site,
    grade_table,
    max_climbable_grade,
    parse_mission,
    traction_monitor,
)
from pinneapple_simulation.rover.autolab.estimation import soil_theta  # noqa: E402
from pinneapple_simulation.terramechanics import WongReeceModel, get_soil  # noqa: E402
from pinneapple_simulation.vehicle import VehicleSimulator, build_vehicle, viper  # noqa: E402

SITE = demo_site()
WHEEL = viper().wheel


def test_parse_mission_pt_en():
    s = parse_mission("Leve o rover até (37, 15) sem atolar, escorregamento máximo de 40%, minimizando energia.", SITE)
    assert s.goal == (37.0, 15.0) and s.slip_limit == pytest.approx(0.4) and s.objective == "energy"
    assert "start = site start" in s.defaults_used and s.quotes["goal_xy"] == "(37, 15)"
    e = parse_mission("Drive from (3, 6) to (30, 10), slip below 0.3, risk 2 %, shortest distance", SITE)
    assert e.start == (3.0, 6.0) and e.goal == (30.0, 10.0)
    assert e.slip_limit == pytest.approx(0.3) and e.risk_max == pytest.approx(0.02) and e.objective == "distance"
    with pytest.raises(ValueError):
        parse_mission("go to (99, 99)", SITE)


def test_llm_values_need_quotes():
    class Fake:
        def extract(self, text, schema):
            return {"slip_limit": 0.9, "goal_xy": [10, 10], "quotes": {"goal_xy": "(10, 10)", "slip_limit": "not there"}}
    s = parse_mission("go to (10, 10)", SITE, llm=Fake())
    assert s.goal == (10.0, 10.0) and s.slip_limit == pytest.approx(0.4) and s.parser == "llm+rules"


def test_soft_sand_climbs_less_than_regolith():
    sand, grc = SITE.true_soil(2), get_soil("GRC-1")
    ts, tg = grade_table(sand, WHEEL, 450.0), grade_table(grc, WHEEL, 450.0)
    assert max_climbable_grade(ts, 0.4) < max_climbable_grade(tg, 0.4)
    assert ts.at(0.0, "sinkage") > 2 * tg.at(0.0, "sinkage")
    assert SITE.mislabelled_units() == [2]


def test_multisoil_simulator_matches_single_soil_and_follows_route():
    model = build_vehicle(viper())
    soil = get_soil("GRC-1")
    kw = dict(speed=0.2, dt=4e-3, settle=0.5, ramp=0.5, record_stresses=False)
    a = VehicleSimulator(model, SITE.deformable_terrain(cell=0.05, x_range=(0, 6), y_range=(3, 9)), soil).run(
        1.5, start=(3.0, 6.0), **kw)
    b = VehicleSimulator(model, SITE.deformable_terrain(cell=0.05, x_range=(0, 6), y_range=(3, 9)),
                         [soil, get_soil("JSC-1A")], soil_map=lambda x, y: np.zeros(np.shape(x), int)).run(
        1.5, start=(3.0, 6.0), **kw)
    assert np.allclose(a.q, b.q) and np.allclose(a.wheel["slip"], b.wheel["slip"])
    c = VehicleSimulator(model, SITE.deformable_terrain(cell=0.05, x_range=(0, 6), y_range=(3, 9)), soil).run(
        30.0, route=[(3.0, 6.0), (4.0, 6.0)], **kw)
    assert c.meta["ended"] == "goal" and abs(c.q[-1, 0] - 4.0) < 0.35


def test_traction_monitor():
    stop = traction_monitor(0.4, window_s=0.2, t_min=1.0)
    t = iter(np.arange(0.0, 5.0, 0.05))
    assert all(stop({"t": next(t), "slip": np.array([0.9, 0.9])}) is None for _ in range(20))   # start-up: ignored
    assert all(stop({"t": next(t), "slip": np.array([0.1, 0.1])}) is None for _ in range(10))
    out = [stop({"t": next(t), "slip": np.array([0.9, 0.9])}) for _ in range(6)]
    assert out[0] is None and out[-1] and out[-1].startswith("stalled")     # needs a 0.2-s mean above 0.4


def test_belief_recovers_soil_from_noise_free_measurements():
    truth = SITE.true_soil(2)
    rng = np.random.default_rng(0)
    W = rng.uniform(60, 160, 48)
    s = rng.uniform(0.02, 0.5, 48)
    ss = WongReeceModel(truth, WHEEL).steady_state(W, s)
    bel = SoilBelief([u.orbital for u in SITE.units], WHEEL, input_sd=(1e-4, 1e-4))
    meas = {"t": np.arange(48.0), "unit": np.full(48, 2), "load": W, "slip": s, "slip_sink": s,
            "sinkage": ss["sinkage"], "torque": ss["torque"]}
    bel.assimilate(meas, batch=8)
    rec = bel.recovery(2, truth)
    for p in ("log10_k_phi", "phi_deg"):
        assert abs(rec[p]["error"]) < 0.5 * abs(rec[p]["prior_error"])
    assert max(abs(v["z"]) for v in rec.values()) < 3.0


def test_planner_avoids_learned_sand_on_slopes():
    bel = SoilBelief([u.orbital for u in SITE.units], WHEEL)
    pl = RiskPlanner(SITE, bel, WHEEL, 450.0, n_samples=3, cell=0.5)
    a = pl.plan(SITE.start, SITE.goal)
    on_sand = a.profile["unit"] == 2
    assert on_sand.any() and a.profile["grade_deg"][on_sand].max() > 10.0     # the prior route takes the sandy pass
    b2 = bel.units[2]
    b2.mean, b2.cov = soil_theta(SITE.true_soil(2)), np.diag([0.05, 0.05, 0.1, 1.0]) ** 2
    pl2 = RiskPlanner(SITE, bel, WHEEL, 450.0, n_samples=3, cell=0.5)
    b = pl2.plan(SITE.start, SITE.goal)
    assert pl2.evaluate(a.waypoints).max_risk > 0.5 and b.max_risk <= 0.05
    steep_sand = (b.profile["unit"] == 2) & (b.profile["grade_deg"] > 6.0)
    assert not steep_sand.any()


def test_kinematic_wheel_speeds():
    sim = VehicleSimulator(build_vehicle(viper()), SITE.deformable_terrain(cell=0.1, x_range=(0, 2), y_range=(0, 2)),
                           get_soil("GRC-1"), wheel_speeds="kinematic")
    assert np.allclose(sim._wheel_speed_scale(0.0), 1.0)
    s = sim._wheel_speed_scale(0.2)                             # left turn: left wheels (side +1) slower
    side = np.array([c.side for c in sim.spec.corners])
    assert s[side > 0].max() < 1.0 < s[side < 0].min()
