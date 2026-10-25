"""pinneapple_simulation.numerical_solvers.fin_cell_fv against the 1-D fin-array closed form, heat balance and grid
convergence."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_lab.uq import gci
from pinneapple_physics.closed_form import fin_array_conduction as fac
from pinneapple_simulation.numerical_solvers.fin_cell_fv import solve_fin_cell

W = D = 0.08
TB, K, TA = 0.003, 205.0, 25.0


def _case(n, h_fin, t, power, h, refine=1):
    pitch = (W - n * t) / (n - 1) + t
    cell = min(t, pitch - t, TB) / (3 * refine)
    return solve_fin_cell(pitch=pitch, t_fin=t, height=h_fin, t_base=TB, depth=D, k=K, h_conv=h,
                          power_per_fin=power / n, t_amb=TA, cell=cell)


def _closed(n, h_fin, t, power, h):
    p = fac.fin_array_profile_params(K, h, n, t, h_fin, D, W, TB, power)
    return fac.fin_array_base_temperature_c(p, TA, power)


def test_heat_balance_and_fin_profile():
    r = _case(20, 0.025, 0.001, 65.0, 35.0)
    assert r["heat_balance_error"] < 1e-8
    assert r["junction_c"] > r["fin_root_c"] > r["fin_tip_c"] > TA
    assert np.isnan(r["T"]).any() and np.nanmax(r["T"]) <= r["junction_c"] + 1e-9


@pytest.mark.parametrize("n,h_fin,t,power,h,tol", [
    (20, 0.025, 0.001, 65.0, 35.0, 0.01),      # baseline: source project's FEniCS cell gave 48.08 C
    (44, 0.020, 0.0006, 65.0, 35.0, 0.01),
    (25, 0.008, 0.0012, 65.0, 35.0, 0.01),
    (8, 0.030, 0.0025, 65.0, 35.0, 0.02),      # thick, sparse fins: 2-D conduction in the base matters more
])
def test_agrees_with_the_one_dimensional_fin_formula(n, h_fin, t, power, h, tol):
    fv = _case(n, h_fin, t, power, h)["junction_c"] - TA
    assert abs(_closed(n, h_fin, t, power, h) - TA - fv) / fv < tol


def test_baseline_matches_the_source_projects_fem_value():
    assert _case(20, 0.025, 0.001, 65.0, 35.0, refine=2)["junction_c"] == pytest.approx(48.08, abs=0.05)


def test_grid_convergence_band_is_small():
    runs = [_case(20, 0.025, 0.001, 65.0, 35.0, refine=r) for r in (1, 2, 4)]
    g = gci([r["junction_c"] - TA for r in runs], [r["cell_size_m"] for r in runs])
    assert g["gci"] < 1e-3


def test_rejects_a_fin_wider_than_its_pitch():
    with pytest.raises(ValueError):
        solve_fin_cell(pitch=0.001, t_fin=0.002, height=0.01, t_base=TB, depth=D, k=K, h_conv=30, power_per_fin=1)
