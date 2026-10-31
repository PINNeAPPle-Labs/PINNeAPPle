"""Immersed (cut-cell) Stokes and Navier-Stokes: known answers. Poiseuille flow between walls that cut the grid is in
the discrete space (Q2 velocity, Q1 pressure) and is reproduced to round-off; a manufactured solution around a disk
converges at the optimal rates; the condition number does not depend on how the body cuts the cells; the force of
the variational form equals the Nitsche surface flux. Campaign: docs/research/cfd_direct_from_stl.md."""
import numpy as np
import pytest

pytest.importorskip("scipy")

from pinneapple_simulation.immersed import flow  # noqa: E402
from pinneapple_simulation.immersed.geometry import SDFGeometry  # noqa: E402
from pinneapple_simulation.immersed.heat import field_errors  # noqa: E402

A, YC, U0 = 0.3, 0.5013, 1.0          # half-width, centre (off the grid planes) and peak of the channel


def _slab():
    return SDFGeometry(lambda X: np.abs(X[:, 1] - YC) - A, ((0, 0), (1, 1)))


def _poiseuille(X):
    return np.column_stack([U0 * (1 - ((X[:, 1] - YC) / A) ** 2), 0 * X[:, 1]])


@pytest.mark.parametrize("navier_stokes", [False, True])
def test_poiseuille_between_cut_walls_is_exact(navier_stokes):
    fd = flow.FlowDiscretization(None, ((0, 0), (1, 1)), h=1 / 8, fluid=_slab())
    assert len(fd.disc.quad.cut_cells) > 0
    nu = 0.5
    bcs = {"x-": flow.Velocity(_poiseuille), "x+": "outflow"}
    sol = (flow.solve_navier_stokes(fd, nu, bcs, u_ref=U0) if navier_stokes else flow.solve_stokes(fd, nu, bcs))
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(0, 1, 400), rng.uniform(YC - A, YC + A, 400)])
    assert np.max(np.abs(sol.velocity(X) - _poiseuille(X))) < 1e-8
    G = 2 * nu * U0 / A ** 2                                     # dp/dx = -G, p = 0 at the free outflow
    assert np.max(np.abs(sol.pressure(X) - G * (1 - X[:, 0]))) < 1e-7
    # wall shear on the two immersed walls: the force per unit length is 2 nu U0 / A per wall, along +x
    F = sol.force()                                              # the walls run into the inflow face: surface flux only
    assert np.isnan(F["variational"]).all()
    assert F["surface"][0] == pytest.approx(2 * 2 * nu * U0 / A, rel=1e-8)
    assert abs(F["surface"][1]) < 1e-8


C, R = (0.5013, 0.4987), 0.2


def _u(X):
    x, y = X[:, 0], X[:, 1]
    return np.column_stack([np.sin(np.pi * x) * np.cos(np.pi * y), -np.cos(np.pi * x) * np.sin(np.pi * y)])


def _grad_u(X):
    x, y = np.pi * X[:, 0], np.pi * X[:, 1]
    g = np.empty((len(X), 2, 2))
    g[:, 0, 0] = np.pi * np.cos(x) * np.cos(y)
    g[:, 0, 1] = -np.pi * np.sin(x) * np.sin(y)
    g[:, 1, 0] = np.pi * np.sin(x) * np.sin(y)
    g[:, 1, 1] = -np.pi * np.cos(x) * np.cos(y)
    return g.reshape(len(X), -1)


def _force(nu, convection):
    def f(X):
        x, y = np.pi * X[:, 0], np.pi * X[:, 1]
        gp = -np.pi * np.column_stack([np.sin(x) * np.cos(y), np.cos(x) * np.sin(y)])
        out = 2 * np.pi ** 2 * nu * _u(X) + gp
        if convection:
            out = out + np.einsum("nij,nj->ni", _grad_u(X).reshape(-1, 2, 2), _u(X))
        return out
    return f


def _mms(h, nu=1.0, convection=False):
    body = flow.PolygonGeometry(flow.regular_polygon(C, R, 2048))
    fd = flow.FlowDiscretization(body, ((0, 0), (1, 1)), h=h)
    bcs = {f: flow.Velocity(_u) for f in ("x-", "x+", "y-", "y+")}
    if convection:
        sol = flow.solve_navier_stokes(fd, nu, bcs, u_ref=1.0, body_velocity=_u, force=_force(nu, True))
    else:
        sol = flow.solve_stokes(fd, nu, bcs, body_velocity=_u, force=_force(nu, False))
    e = field_errors(fd.disc, sol.x[:2 * fd.n_u], _u, _grad_u, fine=False, ncomp=2)
    return e["L2"], e["H1"], sol


@pytest.mark.parametrize("convection", [False, True])
def test_manufactured_solution_converges_at_the_optimal_rates(convection):
    errs = [_mms(h, nu=1.0 if not convection else 0.1, convection=convection)[:2] for h in (1 / 8, 1 / 16)]
    rate_l2 = np.log2(errs[0][0] / errs[1][0])
    rate_h1 = np.log2(errs[0][1] / errs[1][1])
    assert rate_l2 > 2.8, errs
    assert rate_h1 > 1.8, errs


def test_condition_number_does_not_depend_on_the_cut():
    conds = []
    for delta in (0.4987, 1e-1, 1e-2, 1e-3, 1e-4):
        # the top wall crosses the cell above y = 0.75 at a fluid fraction delta
        top = 0.75 + delta / 8
        fluid = SDFGeometry(lambda X, t=top: np.maximum(0.2013 - X[:, 1], X[:, 1] - t), ((0, 0), (1, 1)))
        fd = flow.FlowDiscretization(None, ((0, 0), (1, 1)), h=1 / 8, fluid=fluid)
        sys = flow.assemble_flow(fd, 1.0)
        idx, _ = flow._constraints(fd, {"x-": "wall", "x+": "wall"})
        free = np.ones(fd.n_unknowns, bool)
        free[idx] = False
        conds.append(np.linalg.cond(sys["L"][free][:, free].toarray()))   # saddle point: no diagonal scaling
    assert max(conds) / min(conds) < 10, conds


def test_polygon_signed_distance():
    poly = flow.PolygonGeometry(np.array([[0, 0], [2, 0], [2, 1], [0, 1]], float))
    phi = poly.phi(np.array([[1, 0.5], [3, 0.5], [1, 2], [0.5, 0.25]]))
    assert np.allclose(phi, [-0.5, 1.0, 1.0, -0.25])
    assert poly.area() == pytest.approx(2.0)
