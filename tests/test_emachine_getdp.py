"""Cross-check of the machine FEA with gmsh + GetDP (``machines.getdp``): an independent mesher and solver on the
same geometry and materials. Skipped when the ``gmsh`` and ``getdp`` executables are not installed."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pinneapple_simulation.electromagnetics.machine import belt_winding
from pinneapple_simulation.electromagnetics.machines import M400_50A, IPMMachine, prius_2004
from pinneapple_simulation.electromagnetics.machines import getdp as G

pytestmark = pytest.mark.skipif(not G.available(), reason="gmsh and getdp executables not installed")


@pytest.fixture(scope="module")
def setup():
    g = prius_2004()
    return g, IPMMachine(g, n_gap=576), belt_winding(48, 4)


def test_geo_regions_have_the_polygon_areas(tmp_path, setup):
    g, _, _ = setup
    info = G.write_geo(tmp_path / "m.geo", g, 0.3, n_gap=288)
    assert len(info["magnets"]) == 16 and len(info["slot_areas"]) == 48
    assert info["slot_areas"][0] == pytest.approx(g.stator.slot_area(), rel=2e-3)


def test_no_load_flux_linkage_agrees(setup):
    g, m, lay = setup
    r = G.solve(g, 0.0, (0.0, 0.0, 0.0), M400_50A, m.Br, m.magnet.mu_r, lay, 9.0, n_gap=576)
    psi = m.flux_linkage(m.solve(0.0))
    assert np.max(np.abs(r["psi"] - psi)) < 0.01 * np.max(np.abs(psi))
    assert abs(r["torque"]) < 0.5  # symmetric rotor at the aligned position (GetDP's mesh is not symmetric)


def test_loaded_torque_agrees(setup):
    g, m, lay = setup
    th = m.load_period / 3
    cur = m.phase_currents(th, 250.0, math.radians(50.0))
    r = G.solve(g, th, cur, M400_50A, m.Br, m.magnet.mu_r, lay, 9.0, n_gap=576)
    T = m.torque(m.solve(th, cur))
    assert abs(r["torque"] - T) / T < 0.02
