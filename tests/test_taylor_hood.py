"""Taylor-Hood P2/P1 reference (scikit-fem): Poiseuille flow is in the discrete space and is reproduced to round-off;
the Schafer-Turek 2D-1 drag on a coarse gmsh mesh is within 1 % of the benchmark; prism splitting stays conforming."""
import numpy as np
import pytest

skfem = pytest.importorskip("skfem")

from pinneapple_simulation.numerical_solvers.taylor_hood import (  # noqa: E402
    TaylorHood,
    split_wedges,
)


def test_poiseuille_is_exact():
    mesh = skfem.MeshTri.init_tensor(np.linspace(0, 2, 9), np.linspace(0, 1, 5))
    th = TaylorHood(mesh, nu=0.1)

    def u(X):
        return np.stack([4 * X[1] * (1 - X[1]), 0 * X[1]])

    sol = th.solve([(lambda x: x[0] < 1e-9, u), (lambda x: (x[1] < 1e-9) | (x[1] > 1 - 1e-9), 0.0)])
    X = np.random.default_rng(1).uniform([0, 0], [2, 1], (50, 2))
    assert np.abs(sol.velocity(X) - u(X.T).T).max() < 1e-10
    assert np.abs(sol.pressure(X) - 0.8 * (2 - X[:, 0])).max() < 1e-9   # dp/dx = -8 nu, p = 0 at the outflow
    assert sol.divergence_l2() < 1e-10


def test_schafer_turek_drag_on_a_coarse_mesh():
    pytest.importorskip("gmsh")
    import tempfile

    from pinneapple_simulation.cfd_direct.runners import reference_c1
    r = reference_c1(8, tempfile.mkdtemp())
    assert r["err_Cd"] < 0.01 and r["err_dp"] < 0.01


def test_prism_split_is_conforming():
    # two prisms sharing a quadrilateral face: the shared face is cut along the same diagonal from both sides
    w = np.array([[0, 1, 2, 3, 4, 5], [1, 6, 2, 4, 7, 5]])
    T = split_wedges(w)
    assert T.shape == (6, 4)
    faces = {}
    for t in T:
        for f in ([0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]):
            key = tuple(sorted(t[f]))
            faces[key] = faces.get(key, 0) + 1
    shared = {k for k in faces if set(k) <= {1, 2, 4, 5}}
    assert all(faces[k] == 2 for k in shared) and len(shared) == 2
