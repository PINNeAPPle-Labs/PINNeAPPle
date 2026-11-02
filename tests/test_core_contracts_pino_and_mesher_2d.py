"""Core coverage (#1129, toward 1.0) for the PINO derivatives and wrapper, and the 2-D mesher.

The spectral derivative of a band-limited periodic function is exact; central differences have a known truncation
error. Mesh areas and angles are compared with the exact geometry (a rectangle, a disc, right isosceles and
equilateral triangles)."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinneapple_design.geometry.mesh import mesher_2d as M  # noqa: E402
from pinneapple_neural.architectures.neural_operators import pino as P  # noqa: E402


def _periodic(n=64, L=2 * math.pi):
    x = torch.arange(n, dtype=torch.float64) * L / n
    return x, torch.sin(3 * x) + 0.5 * torch.cos(5 * x)


# ------------------------------------------------------------------------------------------------ derivatives
def test_spectral_derivative_is_exact_for_a_band_limited_function():
    x, u = _periodic()
    d1 = P.spectral_derivative(u, dim=0, L=2 * math.pi, order=1)
    d2 = P.spectral_derivative(u, dim=0, L=2 * math.pi, order=2)
    assert torch.allclose(d1, 3 * torch.cos(3 * x) - 2.5 * torch.sin(5 * x), atol=1e-10)   # exact
    assert torch.allclose(d2, -9 * torch.sin(3 * x) - 12.5 * torch.cos(5 * x), atol=1e-9)


def test_fd_derivative_has_second_order_error():
    errs = []
    for n in (64, 128):
        x, u = _periodic(n)
        dx = 2 * math.pi / n
        d1 = P.fd_derivative(u, dim=0, dx=dx, order=1)
        errs.append(float((d1 - (3 * torch.cos(3 * x) - 2.5 * torch.sin(5 * x))).abs().max()))
    assert 3.6 < errs[0] / errs[1] < 4.4                                          # halving dx divides it by 4
    with pytest.raises(ValueError):
        P.fd_derivative(u, dim=0, dx=dx, order=3)


def test_compute_derivatives_on_a_2d_grid():
    n = 32
    s = torch.arange(n, dtype=torch.float64) * 2 * math.pi / n
    X, Y = torch.meshgrid(s, s, indexing="ij")
    u = (torch.sin(X) * torch.cos(2 * Y))[None]
    grid = P.GridSpec(L=(2 * math.pi, 2 * math.pi), dims=(-2, -1))
    d = P.compute_derivatives(u, grid=grid, orders=(1, 2))
    assert set(d) == {(0, 1), (0, 2), (1, 1), (1, 2)}
    assert torch.allclose(d[(0, 1)], (torch.cos(X) * torch.cos(2 * Y))[None], atol=1e-10)
    assert torch.allclose(d[(1, 2)], -4 * u, atol=1e-9)
    with pytest.raises(ValueError):
        P.compute_derivatives(u, grid=grid, method="nope")


def test_pino_physics_fn_is_zero_for_an_exact_solution():
    """u = sin(x) solves u'' + u = 0, so the PDE residual vanishes; the BC term compares masked values."""
    x, _ = _periodic()
    u = torch.sin(x)[None]
    spec = P.ResidualSpec(residual=lambda u, deriv, params: deriv[(0, 2)] + params["k2"] * u, orders=(2,))
    mask = torch.zeros_like(u, dtype=torch.bool)
    mask[..., 0] = True
    out = P.pino_physics_fn(u, grid=P.GridSpec(L=(2 * math.pi,), dims=(-1,)), spec=spec, params={"k2": 1.0},
                            bc_mask=mask, bc_value=torch.zeros_like(u))
    assert float(out["pde"]) < 1e-20 and float(out["bc"]) < 1e-20
    assert float(out["physics_total"]) == pytest.approx(float(out["pde"] + out["bc"] + out["ic"]))


class _Identity(P.NeuralOperatorBase):
    def forward(self, x):
        return P.OperatorOutput(y=x, losses={"total": torch.tensor(1.0, dtype=x.dtype)}, extras={})


def test_pino_wrapper_adds_the_weighted_physics_loss():
    m = P.PhysicsInformedNeuralOperator(_Identity(), lambda_pde=2.0, lambda_bc=3.0, lambda_physics=0.5)
    x = torch.ones(1, 4, dtype=torch.float64)
    plain = m(x)
    assert float(plain.losses["total"]) == 1.0
    out = m(x, physics_fn=lambda u, **kw: {"pde": u.sum(), "bc": 1.0}, physics_data={})
    assert float(out.losses["physics/physics_total"]) == pytest.approx(2.0 * 4 + 3.0 * 1)
    assert float(out.losses["total"]) == pytest.approx(1.0 + 0.5 * 11.0)
    with pytest.raises(TypeError):
        m(x, physics_fn=lambda u, **kw: 3.0, physics_data={})


# ------------------------------------------------------------------------------------------------ 2-D mesher
def test_structured_rectangle_mesh_and_its_quality():
    m = M.mesh_rectangle_structured((0.0, 2.0), (0.0, 1.0), nx=8, ny=4)
    assert isinstance(m, M.Mesh2D) and (m.n_vertices, m.n_triangles) == (45, 64)
    assert m.triangle_areas().sum() == pytest.approx(2.0, rel=1e-6)              # exact area
    assert len(m.boundary_points["inlet"]) == 5 and len(m.boundary_points["top"]) == 9
    sq = M.mesh_rectangle_structured((0.0, 1.0), (0.0, 1.0), nx=4, ny=4)
    rep = M.mesh_quality_report(sq)
    assert rep["min_angle"]["min"] == pytest.approx(45.0, abs=1e-3)              # right isosceles triangles
    assert rep["aspect_ratio"]["max"] == pytest.approx((math.sqrt(2) / 2) / (2 - math.sqrt(2)), rel=1e-4)
    assert rep["poor_elements"] == 0 and rep["total_area"] == pytest.approx(1.0, rel=1e-6)


def test_polygon_mesh_covers_the_polygon():
    hexagon = np.array([[math.cos(a), math.sin(a)] for a in np.linspace(0, 2 * math.pi, 7)[:-1]])
    m = M.mesh_polygon_2d(hexagon)
    assert m is not None and m.triangle_areas().sum() == pytest.approx(3 * math.sqrt(3) / 2, rel=1e-6)
    rep = M.mesh_quality_report(m)
    assert rep["aspect_ratio"]["min"] >= 1.0 - 1e-6                               # 1 only for equilateral


def test_sdf_mesh_of_a_disc():
    pytest.importorskip("gmsh")
    m = M.mesh_sdf_2d(lambda p: np.hypot(p[:, 0], p[:, 1]) - 0.5, (-0.6, -0.6), (0.6, 0.6), mesh_size=0.05,
                      boundary_n_points=200)
    assert m is not None and m.n_triangles > 100
    assert m.triangle_areas().sum() == pytest.approx(math.pi * 0.25, rel=0.02)    # polygonal disc
    assert np.all(np.hypot(m.vertices[:, 0], m.vertices[:, 1]) <= 0.5 + 0.02)
