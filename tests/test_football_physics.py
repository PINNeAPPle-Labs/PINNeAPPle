"""pinneapple_systems.sports.football: physics against closed forms and simulated data (fast, offline); the real
data loaders only when the data are cached or downloads are allowed (PINNEAPPLE_FOOTBALL_DATA=1)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from pinneapple_systems.sports.football import ball as B
from pinneapple_systems.sports.football import kinematics as K
from pinneapple_systems.sports.football import metrics as M
from pinneapple_systems.sports.football import pitch_control as PC
from pinneapple_systems.sports.football import xg as X


# ----------------------------------------------------------------------------------------------------------- ball
def test_vacuum_flight_is_the_parabola():
    t = np.linspace(0, 2, 21)
    P = B.simulate([0, 0, 0], [10, 2, 9], t)
    exact = np.column_stack([10 * t, 2 * t, 9 * t - 0.5 * B.G * t ** 2])
    assert np.max(np.abs(P - exact)) < 1e-9


def test_drag_slows_and_magnus_lifts():
    t = np.linspace(0, 1.5, 16)
    vac = B.simulate([0, 0, 0], [25, 0, 8], t)
    drag = B.simulate([0, 0, 0], [25, 0, 8], t, cd=0.25)
    back = B.simulate([0, 0, 0], [25, 0, 8], t, cd=0.25, magnus=(0, -B.BALL.k(0.2), 0))   # backspin: lift up
    assert drag[-1, 0] < vac[-1, 0]
    assert back[-1, 2] > drag[-1, 2]


def test_batched_simulation_matches_single():
    t = np.linspace(0, 1, 11)
    a = B.simulate([0, 0, 0.1], [20, 1, 7], t, cd=0.3, magnus=(0, 0, 0.003))
    b = B.simulate(np.array([[0, 0, 0.1]] * 2), np.array([[20, 1, 7]] * 2), t, cd=np.array([0.3, 0.3]),
                   magnus=np.array([[0, 0, 0.003]] * 2), g=np.array([B.G, B.G]))
    assert np.allclose(a, b[0]) and np.allclose(a, b[1])


def test_drag_fit_recovers_coefficient_with_honest_interval():
    rng = np.random.default_rng(1)
    t = np.arange(0, 2.0, 0.1)
    obs = B.simulate([0, 0, 0.2], [22, 2, 9], t, cd=0.3) + rng.normal(0, 0.1, (len(t), 3))
    fit = B.fit_flight(t, obs, "drag", sigma=0.1, loss="linear")
    assert abs(fit.cd - 0.3) < 3 * fit.std("cd") + 0.02
    assert abs(fit.speed0 - np.linalg.norm([22, 2, 9])) < 0.5
    free = B.fit_flight(t, obs, "drag_free_g", sigma=0.1, loss="linear")
    assert abs(free.g - B.G) < 1.0


def test_2d_tracking_cannot_identify_launch_elevation():
    t = np.arange(0, 1.6, 0.04)
    r3 = B.fisher_identifiability([0, 0, 0.11], [20, 0, 9], t, observe=(0, 1, 2))
    r2 = B.fisher_identifiability([0, 0, 0.11], [20, 0, 9], t, observe=(0, 1))
    assert r3["std"]["vz0"] < 1.0
    assert r2["std"]["vz0"] > 50 * r3["std"]["vz0"]


def test_extract_flights_finds_a_lob_and_ignores_rolling():
    t = np.arange(0, 3.0, 0.1)
    P = B.simulate([0, 0, 0.11], [15, 0, 11], t)
    z = np.where(P[:, 2] > 0.11, P[:, 2], 0.11)
    roll = np.full(30, 0.11)
    track = {"frame": np.arange(90.0), "x": np.r_[np.zeros(30), P[:, 0], np.zeros(30)],
             "y": np.zeros(90), "z": np.r_[roll, z, roll], "detected": np.ones(90)}
    fl = B.extract_flights(track)
    assert len(fl) == 1 and 4.5 < fl[0].apex < 7.0


def test_rolling_travel_time_closed_form():
    t = B.rolling_travel_time(np.array([10.0, 100.0]), 10.0, 1.0)
    assert np.isclose(t[0], 10 - np.sqrt(100 - 20))          # (v0 - sqrt(v0^2 - 2 a d)) / a
    assert np.isinf(t[1])                                     # stops after 50 m


def test_drag_crisis_band():
    assert B.cd_crisis(1e5) > 0.4 and B.cd_crisis(5e5) < 0.2
    lo, hi = B.DRAG_CRISIS_BAND["re_crit"]
    assert lo < 2.6e5 < hi


# ---------------------------------------------------------------------------------------------------- kinematics
def test_kalman_recovers_velocity_and_covers_truth():
    rng = np.random.default_rng(0)
    dt = 0.04
    t = np.arange(0, 20, dt)
    true = np.column_stack([3 * t + np.sin(t), 2 * np.cos(0.5 * t)])
    v = np.column_stack([3 + np.cos(t), -np.sin(0.5 * t)])
    obs = true + rng.normal(0, 0.1, true.shape)
    obs[100:110] = np.nan                                       # a gap is bridged
    r = K.kalman_smooth(obs[:, None, :], dt, sigma_pos=0.1, jerk=5.0)
    err = np.linalg.norm(r["vel"][25:-25, 0] - v[25:-25], axis=1)
    assert np.sqrt(np.mean(err ** 2)) < 0.25
    z = np.abs(r["vel"][25:-25, 0] - v[25:-25]) / r["vel_std"][25:-25, 0]
    assert 0.85 < np.mean(z < 1.96) <= 1.0
    raw = K.finite_differences(obs[:, None, :], dt)["vel"][25:-25, 0]
    assert np.nanmean(np.linalg.norm(raw - v[25:-25], axis=1)) > 3 * np.mean(err)


def test_metabolic_power_flat_running_is_minetti_constant():
    assert np.isclose(K.metabolic_power(4.0, 0.0), 3.6 * 4.0)
    assert K.metabolic_power(4.0, 2.0) > K.metabolic_power(4.0, 0.0) > K.metabolic_power(4.0, -1.0)


def test_reachable_set_and_projection():
    assert np.isclose(K.reachable_radius(0.0, 1.0, v_max=12.5, a_max=10.0), 5.0)
    pos, vel = np.zeros((1, 2)), np.array([[5.0, 0.0]])
    far = np.array([[40.0, 0.0]])
    proj = K.project_reachable(pos, vel, far, 1.0)
    assert np.linalg.norm(proj - pos) <= K.reachable_radius(5.0, 1.0) + 1e-9
    near = np.array([[5.2, 0.1]])
    assert np.allclose(K.project_reachable(pos, vel, near, 1.0), near)


# ------------------------------------------------------------------------------------------------- pitch control
def _scene(da, dd):
    att = np.full((1, 2, 2), np.nan)
    dfn = np.full((1, 2, 2), np.nan)
    att[0, 0] = [-da, 0]
    dfn[0, 0] = [dd, 0]
    z = np.zeros((1, 2, 2))
    return att, z, dfn, z, np.array([[0.0, -20.0]]), np.array([[0.0, 0.0]])


def test_pitch_control_symmetry_and_monotonicity():
    p = PC.DEFAULT
    sym = PC.control_probability(*_scene(5, 5), p)
    assert abs(sym[0] - 0.5) < 1e-6
    assert PC.control_probability(*_scene(3, 8), p)[0] > 0.8
    assert PC.control_probability(*_scene(8, 3), p)[0] < 0.2


def test_accel_intercept_time_respects_limits():
    p = PC.PCParams(tti="accel", v_max=8.0, a_max=4.0)
    tau = PC.time_to_intercept(np.zeros((1, 2)), np.zeros((1, 2)), np.array([[20.0, 0.0]]), p)[0]
    # from rest: 2 s to reach 8 m/s covering 8 m, then 12 m at 8 m/s
    assert abs(tau - 3.5) < 1e-3
    moving = PC.time_to_intercept(np.zeros((1, 2)), np.array([[6.0, 0.0]]), np.array([[20.0, 0.0]]), p)[0]
    assert moving < tau


# ----------------------------------------------------------------------------------------------------- metrics, xG
def test_metrics():
    y = np.array([0, 0, 1, 1])
    assert M.auc([0.1, 0.2, 0.8, 0.9], y) == 1.0
    assert np.isclose(M.log_loss([0.5] * 4, y), np.log(2))
    assert np.isclose(M.brier([0, 0, 1, 1], y), 0.0)


def test_open_goal_physics_feature():
    shot = {"x": 108.0, "y": 40.0, "goal": 1, "body_part": "Right Foot", "shot_type": "Open Play",
            "technique": "Normal", "first_time": False, "under_pressure": False, "play_pattern": "Regular Play",
            "statsbomb_xg": 0.3, "match_id": 1, "season": "s"}
    empty = X.shots_arrays([{**shot, "freeze": []}])
    keeper = X.shots_arrays([{**shot, "freeze": [(119.0, 40.0, False, True)]}])
    f0, _ = X.physics_features(empty)
    f1, _ = X.physics_features(keeper)
    assert f0[0, 0] == 1.0
    assert f1[0, 0] < 1.0 and f1[0, 1] > -1.0


# ------------------------------------------------------------------------------------------------- real data (opt)
def _data_allowed(rel: str) -> bool:
    from pinneapple_systems.sports.football import data
    return os.environ.get("PINNEAPPLE_FOOTBALL_DATA") == "1" or (data.cache_dir() / rel).exists()


def test_metrica_loader_and_passes():
    if not _data_allowed("metrica/Sample_Game_1_RawEventsData.csv"):
        pytest.skip("Metrica sample data not cached (set PINNEAPPLE_FOOTBALL_DATA=1 to download)")
    from pinneapple_systems.sports.football import data
    try:
        m = data.metrica_match(1)
    except OSError as exc:
        pytest.skip(f"offline: {exc}")
    assert m.players["Home"].shape[2] == 2 and np.nanmax(np.abs(m.ball[:, 0])) < 60
    ps = PC.passes_from_metrica(m)
    assert len(ps) > 800 and 0.75 < ps.y.mean() < 0.95
