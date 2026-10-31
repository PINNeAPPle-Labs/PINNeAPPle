"""Immersed linear elasticity (pinneapple_simulation.immersed.elasticity) and its conforming P2 reference
(numerical_solvers.tet_fem): manufactured rates, the Lamé thick sphere, equilibrium, total-force loads. Small grids;
the measured studies are in docs/research/immersed_analysis.md."""
import numpy as np
import pytest

from pinneapple_simulation import immersed as im
from pinneapple_simulation.immersed import verification as V


@pytest.mark.parametrize("order,expect", [(1, (1.9, 0.95)), (2, (2.9, 1.9))])
@pytest.mark.parametrize("stabilization", ["ghost", "fcm"])
def test_elastic_manufactured_rates_on_a_disk(order, expect, stabilization):
    rows = V.elastic_convergence(V.elastic_manufactured(2), order, (10, 20, 40), stabilization=stabilization)
    r = V.rates(rows)
    assert min(r["L2"]) > expect[0] and min(r["H1"]) > expect[1]
    # round-off of the solve: FCM's condition number (~1e10) leaves 2e-8 at n = 40 (measured), ghost penalty 4e-9
    assert max(abs(x["force_imbalance"]) for x in rows) < 1e-7


def test_lame_thick_sphere_and_equilibrium():
    r = V.solve_elastic_case(V.lame_sphere(), 6, order=1)                # measured 1.80e-1 at n = 6, 4.61e-2 at 12
    assert r["L2_rel"] < 0.25 and abs(r["force_imbalance"]) < 1e-10
    r2 = V.solve_elastic_case(V.lame_sphere(), 12, order=1)
    assert r2["L2_rel"] < r["L2_rel"] / 3


def test_total_force_is_applied_exactly_whatever_area_is_found():
    geo = im.box((0, 0, 0), (1.0, 0.2, 0.2))
    disc = im.Discretization(geo, 0.1, order=1)
    bcs = [im.Displacement(0.0, where=lambda X, n: (X[:, 0] < 0.05) & (n[:, 0] < -0.5)),
           im.Force((0.0, 0.0, -3.0), where=lambda X, n: (X[:, 0] > 0.9) & (n[:, 0] > 0.5))]
    sol = im.solve_elasticity(disc, 1e3, 0.3, None, bcs)
    assert np.allclose(sol.balance["applied_N"], (0, 0, -3.0), atol=1e-12)
    assert sol.balance["imbalance"] < 1e-10


def test_conforming_p2_elasticity_on_the_lame_sphere():
    try:
        import gmsh  # noqa: F401
    except (ImportError, OSError):
        pytest.skip("gmsh not available")
    from pinneapple_simulation.numerical_solvers import tet_fem as T
    case = V.lame_sphere()
    c = V._shift(3)

    def shell(g):
        a, b = g.model.occ.addSphere(*c, 0.6), g.model.occ.addSphere(*c, 0.3)
        g.model.occ.cut([(3, a)], [(3, b)])
    s = T.solve_elasticity(T.gmsh_mesh(shell, 0.15, 2), case.E, case.nu, None, case.bcs)
    e = s.errors(case.exact, case.grad)
    assert e["L2_rel"] < 1.7e-2 and s.balance["imbalance"] < 1e-8                    # measured 8.45e-3


def test_stl_heat_sink_gets_exactly_its_chip_power(tmp_path):
    try:
        import gmsh  # noqa: F401
    except (ImportError, OSError):
        pytest.skip("gmsh not available")
    from pinneapple_simulation.immersed import parts
    stl = parts.write_stl("heat_sink", str(tmp_path / "hs.stl"), size=0.004)
    case = parts.heat_sink_case(stl)
    geo = im.STLGeometry.from_file(stl)
    assert geo.watertight
    sol = im.solve_heat(im.Discretization(geo, 0.003, order=1), case.conductivity, 0.0, case.bcs)
    assert np.isclose(-sol.balance["HeatInput[0]_out_W"], parts.HS["P"], rtol=1e-12)
    assert abs(sol.balance["imbalance"]) < 1e-10
