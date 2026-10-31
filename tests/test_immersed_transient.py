"""Transient heat on the immersed discretisation (pinneapple_simulation.immersed.transient): time and space rates
on a manufactured solution, and a sphere cooled by convection against its series solution. Small grids."""
import numpy as np
import pytest

from pinneapple_simulation import immersed as im
from pinneapple_simulation.immersed import verification as V
from pinneapple_simulation.immersed.transient import solve_transient_heat


def _run(order, n, steps, theta, stabilization="ghost"):
    c = V.manufactured_transient(2)
    d = im.Discretization(c["geometry"], c["size"] / n, order=order, keep_fine=True)
    s = solve_transient_heat(d, 1.0, 1.0, c["source"], c["bcs"], lambda X: c["exact"](X, 0.0), 0.5 / steps, steps,
                             theta, stabilization=stabilization)
    return s.errors(c["exact"])["L2"]


def test_backward_euler_converges_at_first_order_in_time():
    e = [_run(2, 40, k, 1.0) for k in (4, 8, 16)]
    r = np.log2(np.array(e[:-1]) / np.array(e[1:]))
    assert np.all(r > 0.95)                                             # measured 1.03, 1.02


@pytest.mark.parametrize("order,expect", [(1, 1.9), (2, 2.6)])
def test_crank_nicolson_space_rates(order, expect):
    e = [_run(order, n, 64 if order == 1 else 128, 0.5) for n in (10, 20, 40)]
    r = np.log2(np.array(e[:-1]) / np.array(e[1:]))
    assert np.all(r > expect)                                           # measured Q1 2.22, 2.10; Q2 2.92, 2.65


def test_sphere_cooling_matches_the_series_solution():
    c = V.sphere_cooling()
    tend = 0.3 * c["R"] ** 2 / c["alpha"]
    d = im.Discretization(c["geometry"], 2 * c["R"] / 8, order=1, keep_fine=True)
    s = solve_transient_heat(d, c["k"], c["rho_c"], 0.0, c["bcs"], c["T_i"], tend / 40, 40, 0.5)
    assert s.errors(c["exact"])["L2_rel"] < 1.2e-2                      # measured 7.9e-3
