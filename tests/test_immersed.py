"""Immersed (finite cell / CutFEM) analysis on raw geometry (pinneapple_simulation.immersed): bases, cut-cell
quadrature, SDF / voxel / STL inputs, small-cut stabilisation and steady heat against manufactured and closed-form
solutions. Small grids only: the full studies are in docs/research/immersed_analysis.md."""
import numpy as np
import pytest

from pinneapple_simulation import immersed as im
from pinneapple_simulation.immersed import basis, verification as V
from pinneapple_simulation.immersed.geometry import _closest_on_triangles, icosphere
from pinneapple_simulation.immersed.quadrature import build_quadrature


@pytest.mark.parametrize("p,d", [(1, 2), (2, 2), (1, 3), (2, 3)])
def test_basis_partition_of_unity_and_exact_on_q_p(p, d):
    rng = np.random.default_rng(0)
    xi = rng.random((50, d))
    assert np.allclose(basis.shape(p, xi).sum(1), 1.0)
    assert np.allclose(basis.grad(p, xi).sum(1), 0.0)
    nodes = basis.local_offsets(p, d) / p
    f = lambda x: (x ** p).sum(1) + x.prod(1)                           # noqa: E731 (in Q_p)
    assert np.allclose(basis.shape(p, xi) @ f(nodes), f(xi))


def test_quadrature_disk_area_and_perimeter_converge_with_depth():
    geo = im.ball([0.031, 0.017], 0.7)
    grid = im.BackgroundGrid.around(geo.bounds, 0.1, order=2)
    errs = []
    for depth in (1, 2, 3):
        q = build_quadrature(geo, grid, depth=depth)
        errs.append((abs(q.measure() / (np.pi * 0.49) - 1), abs(q.area() / (2 * np.pi * 0.7) - 1)))
    for (va, aa), (vb, ab) in zip(errs[:-1], errs[1:]):
        assert vb < va / 2.5 and ab < aa / 2.5                          # O((h / 2^depth)^2)
    assert errs[-1][0] < 5e-5 and errs[-1][1] < 2e-5


def test_moment_fitting_reproduces_the_fine_rule_on_q_2p():
    geo = im.ball([0.03, -0.02, 0.01], 0.6)
    grid = im.BackgroundGrid.around(geo.bounds, 0.2, order=1)
    q = build_quadrature(geo, grid, depth=2, keep_fine=True)
    Xf = grid.to_physical(q.fine.cell, q.fine.xi)
    Xc = grid.to_physical(q.volume.cell, q.volume.xi)
    for a in range(3):
        for b in range(3):
            f = lambda X: X[:, 0] ** a * X[:, 1] ** b * X[:, 2] ** (2 - a)          # noqa: E731
            assert np.isclose(np.sum(q.fine.w * f(Xf)), np.sum(q.volume.w * f(Xc)), rtol=1e-10, atol=1e-13)


def test_stl_winding_number_distance_and_sign():
    V_, F = icosphere(0.5, 3)
    geo = im.STLGeometry(V_, F)
    assert geo.watertight
    X = np.random.default_rng(1).uniform(-0.7, 0.7, (4000, 3))
    w = geo.winding_number(X)
    r = np.linalg.norm(X, axis=1)
    far = np.abs(r - 0.5) > 0.02                                        # beyond the faceting of the sphere
    assert np.array_equal(w[far] > 0.5, r[far] < 0.5)
    assert np.array_equal(geo.phi(X) < 0, w > 0.5)                     # pseudonormal sign = winding number
    holey = im.STLGeometry(V_, F[10:])                                 # 10 triangles missing: not watertight
    assert not holey.watertight
    assert np.array_equal(holey.phi(X[far]) < 0, r[far] < 0.5)
    T = V_[F]
    Xs = X[np.abs(r - 0.5) < 0.05][:40]
    brute = np.array([np.sqrt(_closest_on_triangles(np.repeat(x[None], len(T), 0), T[:, 0], T[:, 1], T[:, 2])[0]
                              .min()) for x in Xs])
    assert np.allclose(geo.unsigned_distance(Xs), brute, atol=1e-12)


def test_stl_round_trip(tmp_path):
    V_, F = icosphere(1.0, 1)
    p = im.write_stl(str(tmp_path / "s.stl"), V_, F)
    V2, F2 = im.read_stl(p)
    assert len(F2) == len(F) and np.allclose(np.sort(V2, 0), np.sort(V_, 0), atol=1e-6)


def test_voxel_and_pp_geometry_sdf_inputs():
    from pinneapple_design.geometry.gen.sdf_shapes import sdf3d_sphere
    sph = im.SDFGeometry(lambda X: sdf3d_sphere(X, (0, 0, 0), 0.5), ([-0.5] * 3, [0.5] * 3))
    vox = im.VoxelGeometry.from_geometry(sph, 0.025)
    X = np.random.default_rng(2).uniform(-0.6, 0.6, (3000, 3))
    r = np.linalg.norm(X, axis=1)
    far = np.abs(r - 0.5) > 0.04
    assert np.array_equal(vox.phi(X[far]) < 0, r[far] < 0.5)
    for geo in (sph, vox):
        q = im.Discretization(geo, 0.125, order=1).quad
        assert abs(q.measure() / (4 / 3 * np.pi * 0.125) - 1) < 2e-3


@pytest.mark.parametrize("order,expect", [(1, (1.9, 0.95)), (2, (2.9, 1.9))])
def test_heat_manufactured_rates_on_a_ring(order, expect):
    rows = V.convergence(V.manufactured(2, "annulus", "robin"), order, (10, 20, 40))
    r = V.rates(rows)
    assert min(r["L2"]) > expect[0] and min(r["H1"]) > expect[1]
    assert max(abs(x["energy_imbalance"]) for x in rows) < 1e-8


@pytest.mark.parametrize("order", [1, 2])
def test_heat_dirichlet_nitsche_rates_on_a_disk(order):
    rows = V.convergence(V.manufactured(2, "ball"), order, (10, 20, 40))
    r = V.rates(rows)
    assert min(r["L2"]) > order + 0.9 and min(r["H1"]) > order - 0.1


def test_small_cuts_ghost_penalty_bounds_the_condition_number():
    rows = V.conditioning(1, offsets=(1e-2, 1e-4), stabilizations=("fcm", "ghost"))
    ghost = [r for r in rows if r["stabilization"] == "ghost"]
    fcm = [r for r in rows if r["stabilization"] == "fcm"]
    assert all(r["min_cut_fraction"] < 1e-5 for r in rows)
    assert max(r["cond"] for r in ghost) < 1e3
    assert min(r["cond"] for r in fcm) > 1e8
    assert all(r["L2_rel"] < 2e-2 for r in rows)


def test_sphere_and_cylinder_with_convection():
    s = V.solve_case(V.sphere_convection(), 8, order=2)                 # measured 1.9e-5
    assert s["L2_rel"] < 4e-5 and abs(s["energy_imbalance"]) < 1e-10
    c = V.solve_case(V.cylinder_convection(), 6, order=2)               # measured 2.6e-4 (the rims are rounded
    assert c["L2_rel"] < 5e-4 and abs(c["energy_imbalance"]) < 1e-10    # at the octree-leaf scale)


def test_heat_on_an_stl_sphere_matches_the_closed_form():
    case = V.sphere_convection()
    V_, F = icosphere(0.5, 4, center=V._shift(3))
    geo = im.STLGeometry(V_, F)
    disc = im.Discretization(geo, 0.125, order=1)
    sol = im.solve_heat(disc, case.conductivity, case.source, case.bcs)
    e = sol.errors(case.exact, case.grad)
    assert e["L2_rel"] < 3e-3 and abs(sol.balance["imbalance"]) < 1e-10


def test_conforming_tet_fem_matches_the_closed_form_and_calculix():
    pytest.importorskip("gmsh")
    import shutil
    from pinneapple_simulation.immersed import compare
    part = compare.parts()["sphere"]
    r = compare.run_conforming(part, 2, 0.2)                            # measured 2.8e-5
    assert r["L2_rel"] < 6e-5 and abs(r["energy_imbalance"]) < 1e-10
    p1 = compare.run_conforming(part, 1, 0.2)
    assert p1["L2_rel"] > 10 * r["L2_rel"]
    if shutil.which("ccx"):                                             # the same P2 mesh in CalculiX
        c = compare.run_calculix(part, 0.2)
        assert abs(c["L2_rel"] - r["L2_rel"]) < 0.2 * r["L2_rel"]


def test_equal_accuracy_interpolates_between_measured_runs_only():
    from pinneapple_simulation.immersed.compare import equal_accuracy
    rows = [{"method": "a", "L2_rel": e, "dofs": n, "wall_s": t, "geometry_s": g}
            for e, n, t, g in [(1e-2, 100, 1.0, 0.1), (1e-3, 1000, 10.0, 1.0)]]
    t = equal_accuracy(rows, targets=(np.sqrt(1e-5), 1e-4))
    assert np.isclose(t[0]["dofs"], np.sqrt(1e5)) and np.isnan(t[1]["dofs"])
