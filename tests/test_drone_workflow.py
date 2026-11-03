"""The multirotor workflow (#1201): BEMT against momentum theory and the APC 10 x 4.7 published static data, the
motor power split, Peukert, the LQR margins, the sized drone flown until its battery is empty, the Phantom 4's
published endurance; negative controls."""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from pinneapple_simulation.multirotor.power import Battery, Motor  # noqa: E402
from pinneapple_simulation.multirotor.quad import loop_margins_per_channel, lqr_hover  # noqa: E402
from pinneapple_simulation.multirotor.rotor import (  # noqa: E402
    APC_10X4_7_STATIC,
    apc_10x4_7,
    ideal_twist_check,
    induced_velocity,
    momentum_hover_power,
    prop_coefficients,
)
from pinneapple_simulation.multirotor.workflow import (  # noqa: E402
    build_quad,
    drone_workflow,
    rotor_verification,
    size_quad,
)


def _status(r):
    return {c["code"]: c["status"] for c in r["checks"]}


def test_momentum_theory_and_the_ideal_rotor():
    T, R = 10.0, 0.127
    assert momentum_hover_power(T, R) == pytest.approx(T * induced_velocity(T, R))
    c = ideal_twist_check()
    assert c["rel_error"] < 1e-6 and c["inflow_spread"] < 1e-3      # measured 7.1e-10 and 2.1e-4


def test_apc_published_data_assumed_and_calibrated_polar():
    d = np.asarray(APC_10X4_7_STATIC)
    p = prop_coefficients(apc_10x4_7(), d[:, 0])
    err_t = np.max(np.abs(np.asarray(p["cT"]) / d[:, 1] - 1))
    err_p = np.max(np.abs(np.asarray(p["cP"]) / d[:, 2] - 1))
    assert 0.15 < err_t < 0.25 and 0.30 < err_p < 0.40              # measured 19.1 % and 34.7 %: reported, not hidden
    rv = rotor_verification()
    assert rv["interpolation"]["after"]["cT_max_rel"] < 0.05          # measured 1.5 % on the held-out rows
    assert rv["interpolation"]["after"]["cP_max_rel"] < 0.05          # 1.4 %
    assert rv["extrapolation"]["after"]["cP_max_rel"] < 0.10          # 5.2 % above the training RPMs


def test_motor_power_split_and_peukert():
    m = Motor(kv_rpm=900, r_ohm=0.12, i0=0.6)
    op = m.operating_point(0.05, 600.0)
    assert op["closure"] < 1e-12
    assert m.kt == pytest.approx(60 / (2 * math.pi * 900))
    b = Battery(n_series=4, capacity_ah=5.0, peukert=1.1, rating_hours=1.0)
    assert b.peukert_capacity(5.0) == pytest.approx(5.0)              # at the rated current
    assert b.peukert_capacity(10.0) == pytest.approx(5.0 * 0.5 ** 0.1)
    soc, dt, i = 1.0, 1.0, 10.0                                       # constant-current discharge gives that capacity
    t = 0.0
    while soc > 0:
        soc = b.draw(soc, i, dt)
        t += dt
    assert t / 3600 * i == pytest.approx(b.peukert_capacity(i), rel=1e-3)


def test_lqr_margins_with_motor_lag():
    af = rotor_verification()["interpolation"]["airfoil"]
    q = build_quad(scale=0.8, cells=4, capacity_ah=2.0, payload_kg=0.3, airfoil=af)
    c = lqr_hover(q)
    assert np.max(np.real(c["poles"])) < 0
    mg = loop_margins_per_channel(c, q.tau_motor)
    assert min(x["phase_margin"] for x in mg) >= 45                  # measured 48.3 deg (roll / pitch)
    assert min(x["gain_margin_down_db"] for x in mg) >= 6             # 12.9 dB
    aggressive = lqr_hover(q, r_torque=40.0, q_rate=0.1, q_ang=2.0)
    assert min(x["phase_margin"] for x in loop_margins_per_channel(aggressive, q.tau_motor)) < 45   # 34 deg


def test_drone_workflow_sizes_controls_and_flies_until_empty():
    r = drone_workflow()
    st = _status(r)
    assert r["verdict"] == "pass", r["failed"]
    for code in ("rotor.momentum_identity", "rotor.apc_calibrated_held_out", "rotor.apc_extrapolated",
                 "sizing.feasible", "motor.power_closure", "rotor.hover_power_vs_momentum", "control.margins",
                 "control.step", "control.gust", "mission.empty_battery", "mission.endurance", "mission.tracking",
                 "published.phantom4_hover"):
        assert st[code] == "pass", code
    # measured: 0.937 kg, 8 in, 4S 2 A h; flown 15.04 min (hover estimate 15.26 min); Phantom 4 29.5 min vs 28 min
    assert r["mission"]["empty"]
    assert r["mission"]["endurance_s"] / 60 >= 15.0
    assert abs(r["published"]["rel_error"]) <= 0.25
    assert r["published"]["mass_kg"] == pytest.approx(1.380, rel=1e-3)


def test_impossible_endurance_is_reported():
    af = rotor_verification()["interpolation"]["airfoil"]
    s = size_quad(payload_kg=0.3, endurance_min=300.0, airfoil=af, scales=(1.0,), cells=(4,), capacities=(4.0,))
    assert s["best"] is None
