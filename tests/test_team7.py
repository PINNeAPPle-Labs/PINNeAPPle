"""TEAM workshop problem 7 (``pinneapple_simulation.electromagnetics.fem3d.team7``): the data tables, the coil
current and the probe lines; the full 3-D solve on the coarsest mesh of example 21 runs with PINNEAPPLE_SLOW=1."""
from __future__ import annotations

import os

import numpy as np
import pytest

from pinneapple_simulation.electromagnetics.fem3d import team7


def test_measured_tables_cover_every_line_frequency_and_instant():
    assert len(team7.MEASURED) == 16
    for (line, f, wt), v in team7.MEASURED.items():
        assert f in (50, 200) and wt in (0, 90)
        assert len(v) == (17 if line in ("A1B1", "A2B2") else 12)
    assert team7.MEASURED[("A1B1", 50, 0)][7] == 78.11           # the peak of table 2 (a), x = 126 mm


def test_coil_current_is_uniform_and_counter_clockwise():
    J0 = team7.AMPERE_TURNS / team7.COIL_AREA
    pts = np.array([[0.194, 0.0125, 0.1],      # side y = 0..25 mm: +x
                    [0.2815, 0.1, 0.1],        # side x = 269..294 mm: +y
                    [0.194, 0.1875, 0.1],      # side y = 175..200 mm: -x
                    [0.1065, 0.1, 0.1],        # side x = 94..119 mm: -y
                    [0.244 + 0.0375 / 2**0.5, 0.05 - 0.0375 / 2**0.5, 0.1]])   # a corner arc
    J = team7.coil_current_density(pts)
    assert np.allclose(np.linalg.norm(J, axis=1), J0)
    assert np.allclose(J[:4, :2] / J0, [[1, 0], [0, 1], [-1, 0], [0, -1]])
    assert np.allclose(J[4, :2] / J0, [2**-0.5, 2**-0.5])


def test_eddy_current_probes_lie_in_the_plate():
    for name, P in team7.probe_lines().items():
        if name in ("A3B3", "A4B4"):
            x, y, z = P.T
            hole = (x > 0.018) & (x < 0.126) & (y > 0.018) & (y < 0.126)
            assert np.all((x > 0) & (x < 0.294) & (z > 0) & (z < 0.019) & ~hole)


@pytest.mark.skipif(not os.environ.get("PINNEAPPLE_SLOW"), reason="3-D eddy-current solve (set PINNEAPPLE_SLOW=1)")
def test_team7_coarse_mesh_against_the_measurements(tmp_path):
    pytest.importorskip("gmsh")
    msh = team7.mesh_team7(tmp_path / "t7.msh", h_plate=0.012, h_coil=0.02, h_air=0.08, h_layer=0.01)
    r = team7.run(msh, freqs=(50,), rtol=1e-10)
    f = r["f"][50]
    assert f["info"]["residual"] < 1e-10
    assert f["compare"]["A1B1 wt=0"]["rms_rel_peak"] < 0.1         # 0.068 measured (example 21, coarsest level)
    assert f["loss_W"] == pytest.approx(4.948, rel=0.01)
