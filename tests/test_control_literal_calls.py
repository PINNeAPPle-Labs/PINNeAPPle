"""Literal calls of the control and naval tools, imported from their defining modules: closed-form checks that also
give the Studio its smoke-run inputs (``python -m pinneapple_studio inputs``)."""
import math

from pinneapple_simulation.naval.hydrostatics import (
    box_barge,
    free_surface_correction,
    initial_stability,
    metacentric_height,
    roll_period,
)
from pinneapple_simulation.naval.submarine import architectures, depth_study
from pinneapple_systems.control.analysis import (
    controllability,
    observability,
    stability,
    step_metrics,
)
from pinneapple_systems.control.tuning import (
    fit_fopdt,
    imc_pid,
    place_pid_second_order,
    simc_pid,
    ziegler_nichols,
)


def test_hydrostatics_literal():
    assert box_barge(L=60.0, B=12.0, T=3.0)["BM"] == 4.0
    assert abs(free_surface_correction(tanks=[{"l": 10.0, "b": 12.0, "rho": 1000.0}], volume=2160.0)
               - 1000.0 * 10.0 * 12.0 ** 3 / 12.0 / (1025.0 * 2160.0)) < 1e-12
    assert metacentric_height(KB=1.5, BM=4.0, KG=4.0) == 1.5
    assert abs(roll_period(GM=1.5, B=12.0) - 2 * math.pi * 4.2 / math.sqrt(9.81 * 1.5)) < 1e-12
    st = initial_stability(KB=1.5, BM=4.0, KG=4.0, volume=2160.0, B=12.0)
    assert st["stable"] and st["meets_min_gm"]


def test_analysis_literal():
    assert stability(A=[[0.0, 1.0], [-4.0, -0.8]])["classification"] == "stable (oscillatory)"
    assert controllability(A=[[0.0, 1.0], [0.0, 0.0]], B=[[0.0], [1.0]])["controllable"]
    assert observability(A=[[0.0, 1.0], [0.0, 0.0]], C=[[1.0, 0.0]])["observable"]
    m = step_metrics(t=[0.0, 1.0, 2.0, 3.0, 4.0], y=[0.0, 0.6, 1.1, 1.0, 1.0], reference=1.0)
    assert abs(m["overshoot"] - 10.0) < 1e-9


def test_tuning_literal():
    f = fit_fopdt(t=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0],
                  y=[0.0, 0.0, 0.787, 1.24, 1.506, 1.75, 1.885, 1.983, 2.0], du=1.0)
    assert abs(f["K"] - 2.0) < 0.05
    assert simc_pid(K=2.0, tau=10.0, theta=1.0)["ti"] == 8.0
    assert imc_pid(K=2.0, tau=10.0, theta=2.0, lam=2.0, kind="pi")["ti"] == 10.0
    assert abs(ziegler_nichols(kind="pid", ku=6.0, tu=4.0)["kc"] - 3.6) < 1e-12
    g = place_pid_second_order(m=2.0, c=0.0, k=-3.0, b=0.5, wn=1.5, zeta=0.7, p=0.4)
    assert max(p.real for p in g["poles"]) < 0


def test_submarine_literal():
    assert set(architectures()) == {"pid_target_mass", "pid_pump_flow"}
    r = depth_study(t_end=120.0, tune=False)
    assert r["linear"]["growth_rate"] > 0


def test_cartpole_literal():
    from pinneapple_systems.control.cartpole import CartPoleParams, cartpole_rhs, cartpole_study
    dx = cartpole_rhs(x=[0.0, 0.0, 0.1, 0.0], u=0.0, d=0.0, t=0.0)
    assert abs(dx[3] - 9.81 * math.sin(0.1) / 0.5) < 1e-12
    r = cartpole_study(t_end=8.0, t_push=6.0)
    assert abs(r["unstable_pole"] - math.sqrt(9.81 / 0.5)) < 1e-6 and r["controllable"]
    assert r["closed_loop_classification"].startswith("stable")
    assert r["catch_time"] < 6.0 and r["push_recovery"] < 2.0 and r["max_accel"] <= 15.0 + 1e-12
    assert r["free_fall_time_to_90deg"] < 1.5
    assert CartPoleParams().length == 0.5
