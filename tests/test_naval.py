"""pinneapple_simulation.naval: submarine depth-keeping and ship initial stability."""
import math

import numpy as np

from pinneapple_simulation.naval import (
    ILLUSTRATIVE,
    SubmarineDepth,
    SubmarineParams,
    architectures,
    box_barge,
    depth_study,
    free_surface_correction,
    initial_stability,
    metacentric_height,
    righting_arm,
    roll_period,
)
from pinneapple_systems.control import loop_eigenvalues, simulate_loop


def test_compressible_hull_makes_depth_unstable():
    sub = SubmarineDepth(ILLUSTRATIVE)
    lin = sub.linear()
    assert lin["k"] < 0 and lin["growth_rate"] > 0
    stiff = SubmarineDepth(SubmarineParams(kh=3e-6, kw=4.6e-6)).linear()
    assert stiff["k"] > 0 and stiff["growth_rate"] == 0.0
    assert abs(sub.neutral_depth(ILLUSTRATIVE.trim0) - ILLUSTRATIVE.z_ref) < 1e-6


def test_linear_growth_rate_matches_the_nonlinear_drift():
    sub = SubmarineDepth(SubmarineParams(drag=0.0))
    lam = sub.linear()["growth_rate"]
    # open loop from a 1 cm offset, no drag: z - z_ref grows like cosh(lam t)
    x = np.array([0.0, ILLUSTRATIVE.z_ref + 0.01])
    dt, T = 0.05, 300.0
    for _ in range(int(T / dt)):
        k1 = sub.rhs(x, ILLUSTRATIVE.trim0, 0.0, 0)
        k2 = sub.rhs(x + dt / 2 * k1, ILLUSTRATIVE.trim0, 0.0, 0)
        k3 = sub.rhs(x + dt / 2 * k2, ILLUSTRATIVE.trim0, 0.0, 0)
        k4 = sub.rhs(x + dt * k3, ILLUSTRATIVE.trim0, 0.0, 0)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    assert abs((x[1] - ILLUSTRATIVE.z_ref) / (0.01 * math.cosh(lam * T)) - 1) < 0.01


def test_depth_study_reproduces_the_illustrative_example():
    r = depth_study(tune=True)
    ol = r["open_loop"]
    assert ol["stopped"] and "600" in ol["stopped"] and abs(ol["t_stop"] - 642.0) < 0.5
    rows = {row["name"]: row for row in r["architectures"]}
    tm = rows["pid_target_mass"]
    assert tm["stopped"] is None and tm["stable_linear"]
    assert abs(tm["y_min"] - 99.77) < 0.01 and abs(tm["y_max"] - 102.39) < 0.01
    assert abs(tm["final_trim"] - 18500.0) < 1.0
    fl = rows["pid_pump_flow"]
    assert fl["stable_linear"] is False and fl["stopped"] is not None
    tuned = r["tuned"]["rows"][0]
    assert tuned["stable_linear"] and tuned["stopped"] is None
    assert tuned["y_max"] - 100.0 < tm["y_max"] - 100.0           # tighter than the example's gains


def test_target_mass_loop_eigenvalues_match_the_analytic_polynomial():
    sub = SubmarineDepth(ILLUSTRATIVE)
    lin = sub.linear()
    ctrl, act = architectures()["pid_target_mass"]
    le = loop_eigenvalues(sub.plant(), ctrl, act, [0.0, 100.0], ILLUSTRATIVE.trim0, 100.0, dt=0.25,
                          method="rk4")
    roots = np.roots([lin["m_eff"], 9.81 * 27500, lin["k"] + 9.81 * 500, 9.81 * 5])
    assert le["stable"] and abs(le["max_real"] - roots.real.max()) < 2e-4


def test_robustness_samples_run():
    r = depth_study(tune=False, robustness_samples=4, t_end=300.0)
    assert 0.0 <= r["robustness"]["fraction_stable"] <= 1.0


def test_box_barge_and_roll_period():
    h = box_barge(L=60.0, B=12.0, T=3.0)
    assert abs(h["BM"] - 12.0 ** 2 / (12 * 3.0)) < 1e-12 and h["KB"] == 1.5
    gm = metacentric_height(h["KB"], h["BM"], KG=4.0)
    assert abs(gm - 1.5) < 1e-12
    assert abs(roll_period(gm, k_xx=4.2) - 2 * math.pi * 4.2 / math.sqrt(9.81 * 1.5)) < 1e-12
    assert np.allclose(righting_arm(gm, [0, 10]), gm * np.sin(np.radians([0, 10])))
    assert righting_arm(gm, 20, BM=h["BM"]) > righting_arm(gm, 20)


def test_free_surface_can_make_a_ship_unstable():
    h = box_barge(L=60.0, B=12.0, T=3.0)
    fsc = free_surface_correction([{"l": 30.0, "b": 12.0, "rho": 1025.0}], h["volume"])
    assert abs(fsc - 30 * 12 ** 3 / 12 / h["volume"]) < 1e-12           # 2.0 m > GM 1.5 m
    st = initial_stability(h["KB"], h["BM"], 4.0, h["volume"], B=12.0, tanks=[{"l": 30.0, "b": 12.0,
                                                                                "rho": 1025.0}])
    assert not st["stable"] and st["roll_period"] == math.inf and st["notes"]


def test_simulate_loop_without_control_drifts():
    sub = SubmarineDepth(ILLUSTRATIVE)
    ctrl, act = architectures()["pid_target_mass"]
    r = simulate_loop(sub.plant(), ctrl, act, 100.0, 120.0, 0.25, disturbance=lambda t: 1500.0,
                      method="sequential_euler", u0=ILLUSTRATIVE.trim0)
    assert r["y_max"] > 100.0 and r["stopped"] is None


def test_flow_nodes_run(tmp_path):
    import pinneapple_flow as pf
    nodes = [{"id": "a", "type": "control.linear_analysis", "params": {}},
             {"id": "t", "type": "control.pid_tuning", "params": {"method": "pole_placement"}},
             {"id": "s", "type": "naval.submarine_depth", "params": {"t_end": 300.0}},
             {"id": "g", "type": "naval.ship_stability", "params": {}}]
    r = pf.run(pf.Flow("control", nodes=nodes), workdir=tmp_path)
    assert r.status == "done", {k: v.get("error") for k, v in r.nodes.items()}
    m = r.output("a", "metrics")
    assert m["classification"] == "stable (oscillatory)" and abs(m["gain_margin_db"]["value"] - 15.563) < 1e-2
    assert r.output("s", "metrics")["pid_pump_flow.stable_linear"] is False
