"""The rover workflow (#1200): closed forms of the step climb, tip-over, solar day and heater; the lightest rover
for the default Mars mission within its margins; the traverse re-verification (energy closure, battery resized from
the measured depth of discharge); Sojourner's published peak power; negative controls."""
from __future__ import annotations

import math

import pytest

pytest.importorskip("scipy")

from pinneapple_simulation.vehicle.mission_design import (  # noqa: E402
    PLANETS,
    SIGMA_SB,
    heater_power,
    max_step_height,
    published_rover_check,
    rover_workflow,
    solar_array,
    step_climb_force,
    tip_over_angles,
)


def _status(r):
    return {c["code"]: c["status"] for c in r["checks"]}


def test_step_climb_closed_form():
    R, W, mu = 0.1, 10.0, 0.6
    assert step_climb_force(0.0, R, W, mu) == 0.0
    assert step_climb_force(R, R, W, mu) == pytest.approx(W / mu)          # continuous with the vertical face
    assert step_climb_force(R - 1e-9, R, W, mu) == pytest.approx(W / mu, rel=1e-6)
    a = math.acos(1 - 0.05 / R)                                          # a hand-worked edge contact
    n = W / (math.cos(a) + mu * math.sin(a))
    assert step_climb_force(0.05, R, W, mu) == pytest.approx(n * (math.sin(a) - mu * math.cos(a)))
    # a low step the wheel climbs on its own torque: tan a <= mu
    a0 = math.atan(mu)
    assert step_climb_force(R * (1 - math.cos(a0)) * 0.99, R, W, mu) == 0.0
    h = max_step_height(R, W, 5.0, mu, 0.15)
    assert step_climb_force(h, R, W, mu) == pytest.approx(5.0, rel=1e-6)


def test_tip_over_solar_and_heater_closed_forms():
    t = tip_over_angles(track=0.4, wheelbase=0.6, cg_height=0.2)
    assert t["lateral_deg"] == pytest.approx(45.0)
    assert t["longitudinal_deg"] == pytest.approx(math.degrees(math.atan(1.5)))
    pl = PLANETS["mars"]
    s = solar_array(1.0, 0.25, pl)
    assert s["peak_W"] == pytest.approx(0.25 * 586.2 * 0.6)
    assert s["energy_Wh"] == pytest.approx(s["peak_W"] * 2 / math.pi * pl["day_s"] / 2 / 3600)
    assert heater_power(1.0, 0.05, 250.0, 150.0, 0.0) == pytest.approx(0.05 * SIGMA_SB * (250 ** 4 - 150 ** 4))
    assert heater_power(1.0, 0.05, 250.0, 150.0, 1e3) == 0.0


def test_sojourner_published_power_within_the_band():
    p = published_rover_check()
    # measured: 13.9 W against the reported 16 W (-12.9 %); drive power at 1 cm/s 0.14 W
    assert p["within_band"]
    assert p["drivable_on_array"]


def test_rover_workflow_designs_and_reverifies_on_the_traverse():
    r = rover_workflow()
    st = _status(r)
    assert r["verdict"] == "pass", r["failed"]
    for code in ("design.feasible", "mobility.grade", "mobility.step", "mobility.stability", "mobility.sinkage",
                 "power.energy_margin", "traverse.distance", "traverse.battery", "traverse.slip",
                 "traverse.energy_closure", "published.sojourner_power", "published.sojourner_drive"):
        assert st[code] == "pass", code
    d, tr = r["design"], r["traverse"]
    # measured: 12.73 kg, wheels R 0.050 x 0.106 m, array 0.158 m^2, battery 82.2 -> 90.8 W h after the traverse
    assert 5.0 < d["mass_kg"] < 30.0
    assert d["grade_deg"] >= r["mission"]["max_slope_deg"]
    assert tr["energy_closure"] < 1e-9
    assert tr["soc_min"] >= tr["soc_floor"] - 1e-9
    assert r["battery_resized"], "the night-only battery misses the dawn / dusk deficit the traverse measures"


def test_impossible_mission_is_reported():
    r = rover_workflow(mission={"max_slope_deg": 40.0}, grid=3)
    assert _status(r)["design.feasible"] == "fail"
    assert r["verdict"] == "fail"
    assert "traverse" not in r
