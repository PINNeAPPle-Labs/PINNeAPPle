"""3-D edge-element (Nedelec) magnetostatics and eddy currents (``pinneapple_simulation.electromagnetics.fem3d``).

Fast cases on structured box meshes run everywhere; the gmsh sphere and the GetDP cross-checks are skipped when
gmsh or getdp is missing."""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from pinneapple_simulation.electromagnetics.fem3d import box_mesh
from pinneapple_simulation.electromagnetics.fem3d.cases import manufactured_magnetostatic, slab_eddy
from pinneapple_simulation.electromagnetics.fem3d.nedelec import (
    NU0,
    EdgeSpace,
    solve_magnetostatic,
    tree_gauge,
)
from pinneapple_simulation.electromagnetics.materials import illustrative_steel

try:
    import gmsh  # noqa: F401
    HAVE_GMSH = True
except ImportError:  # pragma: no cover
    HAVE_GMSH = False
HAVE_GETDP = shutil.which("getdp") is not None


@pytest.mark.parametrize("gauge", ["tree", "none"])
def test_uniform_field_is_exact(gauge):
    g = np.linspace(-1, 1, 5)
    m = box_mesh(g, g, g)
    sp = EdgeSpace(m)
    B0 = np.array([0.3, -0.2, 1.0])
    de = m.boundary_edges
    s = solve_magnetostatic(m, NU0, dirichlet=(de, sp.edge_values(lambda x: 0.5 * np.cross(B0, x), de)),
                            gauge=gauge, space=sp, cg_tol=1e-13)
    assert np.abs(s.B - B0).max() < 1e-9
    assert s.energy == pytest.approx(0.5 * NU0 * B0 @ B0 * 8.0, rel=1e-10)


def test_tree_spans_the_free_nodes():
    g = np.linspace(0, 1, 5)
    m = box_mesh(g, g, g)
    de = m.boundary_edges
    tree = tree_gauge(m, de, np.unique(m.edges[de]))
    free = len(m.nodes) - len(m.boundary_nodes)
    assert len(tree) == free and len(np.unique(tree)) == free
    assert not np.isin(tree, de).any()


def test_manufactured_solution_converges_first_order():
    e = [manufactured_magnetostatic(n)["error"] for n in (5, 9, 17)]
    orders = np.log2(np.array(e[:-1]) / np.array(e[1:]))
    assert e[-1] < 0.06
    assert np.all(orders > 0.95) and np.all(orders < 1.1)


def test_mass_matrix_integrates_exactly():
    g = np.linspace(0, 1, 4)
    m = box_mesh(g, g, np.linspace(0, 2, 5))
    sp = EdgeSpace(m)
    M = sp.mass(1.0)
    a = sp.edge_values(lambda x: np.column_stack([x[:, 1], -x[:, 0], 0 * x[:, 0]]))   # in the Whitney space
    assert a @ M @ a == pytest.approx(2.0 * 2.0 / 3.0, rel=1e-12)


def test_linked_roots_remove_the_floating_potential():
    # a conductor inside the box: its nodes and the boundary are two root groups; the forest leaves the
    # conductor's floating potential in the null space, one more tree edge removes it
    g = np.linspace(0, 1, 7)
    m = box_mesh(g, g, g, region=lambda c: np.where(np.all(np.abs(c - 0.5) < 0.2, axis=1), 1, 0),
                 names={"air": 0, "plate": 1})
    de = m.boundary_edges
    roots = np.union1d(np.unique(m.edges[de]), m.nodes_in(m.mask("plate")))
    forest = tree_gauge(m, de, roots)
    tree = tree_gauge(m, de, roots, link_roots=True)
    assert len(tree) == len(forest) + 1 and np.isin(forest, tree).all()
    sp = EdgeSpace(m)
    sigma = np.where(m.mask("plate"), 1e7, 0.0)
    K = (sp.stiffness(NU0) + 1j * 2 * np.pi * 50 * sp.mass(sigma)).tocsr()
    for t, singular in ((forest, True), (tree, False)):
        u = np.setdiff1d(np.setdiff1d(np.arange(sp.n), de), t)
        smin = np.linalg.svd(K[u][:, u].toarray(), compute_uv=False)[-1]
        assert (smin < 1e-8 * NU0) == singular


def test_pardiso_and_superlu_agree():
    from pinneapple_simulation.electromagnetics.fem3d.nedelec import _mkl_path

    _mkl_path()
    pytest.importorskip("pypardiso")
    a = slab_eddy(8, nxy=4, gauge="tree", solver="splu")
    b = slab_eddy(8, nxy=4, gauge="tree", solver="pardiso")
    assert b["loss_error"] == pytest.approx(a["loss_error"], abs=1e-10)


def test_slab_eddy_against_closed_form():
    r1 = slab_eddy(8, nxy=4, gauge="tree")
    r2 = slab_eddy(16, nxy=8, gauge="tree")
    assert r2["error_B"] < 0.6 * r1["error_B"]
    assert r2["loss_exact"] == pytest.approx(r2["loss_quadrature"], rel=1e-6)   # formula vs integral of the field
    assert r2["loss_error"] < 0.02


def _iron_cube(current: float):
    g = np.linspace(0, 1, 7)
    m = box_mesh(g, g, g, region=lambda c: np.where(np.all(np.abs(c - 0.5) < 0.2, axis=1), 1, 0),
                 names={"air": 0, "iron": 1})
    sp = EdgeSpace(m)
    j1 = lambda x: np.column_stack([-(x[:, 1] - 0.5), x[:, 0] - 0.5, 0 * x[:, 0]]) * 5e6   # noqa: E731
    ring = np.abs(np.linalg.norm(m.centroids[:, :2] - 0.5, axis=1) - 0.32) < 0.1
    return m, sp, (lambda x: current * j1(x)), ring


def test_newton_converges_and_coenergy_derivative_is_the_flux_linkage():
    curve = illustrative_steel(mu_r_init=2000.0, Js=1.8)
    out = {}
    for current in (0.95, 1.0, 1.05):
        m, sp, J, mask = _iron_cube(current)
        s = solve_magnetostatic(m, NU0, J=J, J_mask=mask, nonlinear={1: curve}, gauge="tree", space=sp,
                                tol=1e-10)
        out[current] = s
    s = out[1.0]
    assert s.residual < 1e-10 and s.iterations < 25
    assert np.max(np.linalg.norm(s.B[m.region == 1], axis=1)) > 1.0      # the iron saturates
    psi = s.source_work() / 1.0
    dW = (out[1.05].coenergy - out[0.95].coenergy) / 0.1
    assert dW == pytest.approx(psi, rel=2e-3)


@pytest.mark.skipif(not HAVE_GMSH, reason="gmsh not installed")
def test_sphere_in_uniform_field(tmp_path):
    from pinneapple_simulation.electromagnetics.fem3d.sphere import (
        SphereCase,
        mesh_sphere,
        solve_native,
    )

    c = SphereCase(mu_r=10.0)
    r = solve_native(mesh_sphere(tmp_path / "s.msh", 0.3, c), c)
    assert r["mean_in"] < 0.05 and r["l2_in"] < 0.05


@pytest.mark.skipif(not (HAVE_GMSH and HAVE_GETDP), reason="gmsh and getdp needed")
def test_sphere_matches_getdp_on_the_same_mesh(tmp_path):
    from pinneapple_simulation.electromagnetics.fem3d import getdp3d, read_msh22
    from pinneapple_simulation.electromagnetics.fem3d.sphere import (
        SphereCase,
        errors,
        mesh_sphere,
        uniform_dirichlet,
    )

    c = SphereCase(mu_r=10.0)
    msh = mesh_sphere(tmp_path / "s.msh", 0.35, c)
    g = getdp3d.sphere(msh, c)
    m = read_msh22(msh)
    sp = EdgeSpace(m)
    s = solve_magnetostatic(m, np.where(m.mask("sphere"), NU0 / c.mu_r, NU0), dirichlet=uniform_dirichlet(m, sp, c),
                            space=sp, cg_tol=1e-12)
    e = errors(m, s.B, c)
    assert e["Bz_mean_in"] == pytest.approx(g["Bz_mean_in"], rel=1e-8)
    assert s.energy == pytest.approx(g["energy"], rel=1e-8)

