"""Drive model (``pinneapple_simulation.electromagnetics.machines.drive``): dq steady state under current and voltage
limits, least-loss operating points, efficiency map, vehicle and WLTC drive-cycle energy."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pinneapple_simulation.electromagnetics.machines import verification as mv
from pinneapple_simulation.electromagnetics.machines.drive import (
    Drive,
    LinearDQ,
    Vehicle,
    copper_resistance,
    cycle_energy,
    v_max_svpwm,
    wltc_class3b,
)

DRIVE_CASES = [n for n in mv.CASES if n.startswith(("Drive", "Prius flux map"))]


def test_wltc_trace_is_the_regulation_cycle():
    """GTR 15 class 3b: 1800 s, 23.266 km, 131.3 km/h."""
    t, v = wltc_class3b()
    assert len(t) == 1801 and t[-1] == 1800.0
    assert np.sum(v) / 1000 == pytest.approx(23.266, abs=5e-4)
    assert 3.6 * v.max() == pytest.approx(131.3)
    assert np.sum(3.6 * v) == pytest.approx(83758.6, abs=1e-6)


def test_voltage_limit_and_svpwm():
    assert v_max_svpwm(500.0) == pytest.approx(500.0 / math.sqrt(3))
    m = LinearDQ(4, 0.2, 0.6e-3, 2.4e-3)
    d = Drive(m, R=0.0, I_max=250.0, V_max=v_max_svpwm(500.0))
    w = 6000 * 2 * math.pi / 60
    top = d.max_torque(w)
    assert d.voltage(top["i_d"], top["i_q"], m.pole_pairs * w) <= d.V_max * (1 + 1e-9)
    assert math.hypot(top["i_d"], top["i_q"]) <= d.I_max * (1 + 1e-9)


def test_efficiency_map_is_physical():
    m = LinearDQ(4, 0.2, 0.6e-3, 2.4e-3, core_loss_coeffs=(2.0, 1.3, 2e-3))
    d = Drive(m, R=0.05, I_max=250.0, V_max=v_max_svpwm(500.0), n_d=81)
    em = d.efficiency_map(np.array([100.0, 400.0, 800.0]), np.array([-200.0, 50.0, 200.0, 500.0]))
    eff = em["efficiency"]
    ok = np.isfinite(eff)
    assert ok.sum() >= 6 and np.all((eff[ok] > 0) & (eff[ok] < 1))
    assert np.isnan(eff[2, 3])  # 500 N m is beyond the envelope at 800 rad/s


def test_copper_resistance_closed_form():
    R = copper_resistance(72, 13 * math.pi * 0.456e-3 ** 2, 0.08382, 0.16, temperature=20.0)
    assert R == pytest.approx(1.724e-8 * 72 * 2 * (0.08382 + 0.16) / (13 * math.pi * 0.456e-3 ** 2))
    assert copper_resistance(1, 1e-6, 1.0, 0.0, temperature=120.0) == pytest.approx(2 * 1.724e-2 * 1.393)


def test_braking_beyond_the_envelope_goes_to_friction():
    m = LinearDQ(4, 0.2, 0.6e-3, 2.4e-3)
    d = Drive(m, R=0.0, I_max=60.0, V_max=v_max_svpwm(500.0))
    t, v = wltc_class3b()
    veh = Vehicle(mass=1450.0, c_rr=0.009, cd_a=0.58, wheel_radius=0.30, gear_ratio=4.113)
    with pytest.raises(ValueError):  # 60 A cannot accelerate the car
        cycle_energy(d, veh, t, v, envelope_speeds=7)


@pytest.mark.parametrize("name", DRIVE_CASES)
def test_drive_verification_case(name):
    r = mv.run([name], quick=True)[0]
    assert r.passed, r.row() + " " + r.detail
