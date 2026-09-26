"""Shallow-water FV solver: exact dam breaks (Ritter, Stoker), still-water wall force, conservation,
gates, sensors, CSV and experiment comparison."""
import math

import numpy as np
import pytest

from pinneapple_simulation.numerical_solvers.shallow_water_fv import G, RHO, ShallowWater2D, ritter, stoker


def _channel(nx=400, L=10.0, ny=3):
    sw = ShallowWater2D(nx, ny, L / nx)
    X, _ = sw.centers()
    return sw, X


def _l1_rel(num, ref):
    return float(np.abs(num - ref).sum() / np.abs(ref).sum())


def test_ritter_dry_bed_dam_break():
    sw, X = _channel()
    h0, x0, t = 1.0, 5.0, 0.4
    sw.h[X < x0] = h0
    sw.run(t)
    href, _ = ritter(X[:, 1], t, h0, x0)  # exact solution
    assert _l1_rel(sw.h[:, 1], href) < 0.03
    assert sw.health_report()["ok"]


def test_stoker_wet_bed_dam_break():
    sw, X = _channel()
    h0, h1, x0, t = 1.0, 0.2, 5.0, 0.5
    sw.h[:] = np.where(X < x0, h0, h1)
    sw.run(t)
    href, _ = stoker(X[:, 1], t, h0, h1, x0)
    assert _l1_rel(sw.h[:, 1], href) < 0.02
    # bore position: last cell above the mid-depth between hm and h1
    hm = href.max(where=(X[:, 1] > x0 + 0.5), initial=0)
    front_num = X[:, 1][sw.h[:, 1] > 0.5 * (hm + h1)].max()
    front_ref = X[:, 1][href > 0.5 * (hm + h1)].max()
    assert abs(front_num - front_ref) < 4 * sw.dx


def test_still_water_is_exactly_at_rest_and_wall_force_is_hydrostatic():
    sw = ShallowWater2D(40, 20, 0.05)
    h = 0.3
    sw.h[:] = h
    wall = np.zeros((40, 20), bool)
    wall[-4:, :] = True  # a thick wall at the right end
    sw.add_wall(wall)
    sw.add_wall_force_sensor("wall", wall)
    sw.add_pressure_probe("p_bed", 0.5, 0.5, z=0.0)
    sw.run(0.5)
    assert np.abs(sw.hu).max() < 1e-12 and np.abs(sw.h[~sw.solid] - h).max() < 1e-12
    width = 20 * 0.05
    assert sw.sensors[0].values[-1] == pytest.approx(RHO * G * h * h / 2 * width, rel=1e-9)
    assert sw.sensors[1].values[-1] == pytest.approx(RHO * G * h, rel=1e-9)


def test_gate_release_volume_conservation_sensors_csv_and_compare(tmp_path):
    nx, ny, L = 120, 30, 3.0
    sw = ShallowWater2D(nx, ny, L / nx)
    X, Y = sw.centers()
    sw.h[X < 1.0] = 0.5
    gate = (X > 1.0) & (X < 1.0 + 2 * sw.dx)
    sw.add_gate("gate", gate, release_time=0.1)
    obstacle = (X > 2.2) & (X < 2.4) & (Y > 0.3) & (Y < 0.6)
    sw.add_wall(obstacle)
    sw.add_depth_gauge("G1", 1.5, 0.375)
    sw.add_wall_force_sensor("F_obstacle", obstacle)
    sw.run(1.2)
    rep = sw.health_report()
    assert rep["ok"] and rep["max_volume_rel_drift"] < 1e-12
    g1 = np.array(sw.sensors[0].values)
    t = np.array(sw.times)
    assert g1[t < 0.1].max() == 0.0 and g1.max() > 0.05  # dry until the gate opens, then wets
    assert max(sw.sensors[1].values) > 0  # the surge hits the obstacle
    path = sw.export_csv(str(tmp_path / "run.csv"))
    rows = open(path).read().splitlines()
    assert rows[0].split(",")[:3] == ["t", "G1", "F_obstacle"] and len(rows) == len(sw.times) + 1
    cmp = sw.compare("G1", t[::5], g1[::5])
    assert cmp["rmse"] < 1e-12
