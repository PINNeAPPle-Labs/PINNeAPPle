"""Operators against manufactured solutions on grid, mesh, point cloud and continuous fields."""
import numpy as np
import pytest

from pinneapple_core import (
    Domain, Field, FunctionField, Mesh, curl, div, flux, grad, hessian, integrate, jacobian, laplacian,
)

# scalar s = sin(2x) cos(3y)
S = lambda p: np.sin(2 * p[:, 0]) * np.cos(3 * p[:, 1])
GS = lambda p: np.stack([2 * np.cos(2 * p[:, 0]) * np.cos(3 * p[:, 1]), -3 * np.sin(2 * p[:, 0]) * np.sin(3 * p[:, 1])], 1)
LS = lambda p: -13 * S(p)
_sxy = lambda p: -6 * np.cos(2 * p[:, 0]) * np.sin(3 * p[:, 1])
HS = lambda p: np.stack([np.stack([-4 * S(p), _sxy(p)], 1), np.stack([_sxy(p), -9 * S(p)], 1)], 1)
# vector v = (sin(2x) cos y, cos x sin(2y))
V = lambda p: np.stack([np.sin(2 * p[:, 0]) * np.cos(p[:, 1]), np.cos(p[:, 0]) * np.sin(2 * p[:, 1])], 1)
DV = lambda p: 2 * np.cos(2 * p[:, 0]) * np.cos(p[:, 1]) + 2 * np.cos(p[:, 0]) * np.cos(2 * p[:, 1])
CV = lambda p: -np.sin(p[:, 0]) * np.sin(2 * p[:, 1]) + np.sin(2 * p[:, 0]) * np.sin(p[:, 1])
JV = lambda p: np.stack([
    np.stack([2 * np.cos(2 * p[:, 0]) * np.cos(p[:, 1]), -np.sin(2 * p[:, 0]) * np.sin(p[:, 1])], 1),
    np.stack([-np.sin(p[:, 0]) * np.sin(2 * p[:, 1]), 2 * np.cos(p[:, 0]) * np.cos(2 * p[:, 1])], 1)], 1)

DOM = Domain.box([0, 0], [1, 1])


def _interior(p, margin=0.15):
    return ((p > margin) & (p < 1 - margin)).all(axis=1)


def _rel(a, b, pts):
    m = _interior(pts)
    return np.abs(a - b)[m].max() / np.abs(b).max()


def _supports():
    yield "grid", lambda g: Field.on_grid([np.linspace(0, 1, 81)] * 2, g), 2e-3
    yield "mesh", lambda g: Field.on_mesh(Mesh.structured([0, 0], [1, 1], (40, 40)), g), 1e-2
    yield "cloud", lambda g: Field.on_points(DOM.sample_interior(4000, seed=0), g, domain=DOM), 1.5e-1


@pytest.mark.parametrize("name,make,tol", list(_supports()), ids=[s[0] for s in _supports()])
def test_discrete_operators_match_analytic(name, make, tol):
    s, v = make(S), None
    p = s.points
    v = Field(V(p), s.support)
    assert _rel(grad(s).values, GS(p), p) < tol
    assert _rel(laplacian(s).values, LS(p), p) < tol
    assert _rel(hessian(s).values, HS(p), p) < tol
    assert _rel(div(v).values, DV(p), p) < tol
    assert _rel(curl(v).values, CV(p), p) < tol
    assert _rel(jacobian(v).values, JV(p), p) < tol
    assert jacobian(s).values.shape == (p.shape[0], 1, 2)
    assert laplacian(v).values.shape == (p.shape[0], 2)


def test_discrete_operators_converge_under_refinement():
    errs = []
    for n in (10, 20, 40):
        s = Field.on_mesh(Mesh.structured([0, 0], [1, 1], (n, n)), S)
        errs.append(_rel(grad(s).values, GS(s.points), s.points))
    assert errs[0] > errs[1] > errs[2]
    assert errs[0] / errs[2] > 8  # about second order


def test_curl_3d_on_grid_and_mesh():
    f = lambda p: np.stack([np.sin(p[:, 1]), np.sin(p[:, 2]), np.sin(p[:, 0])], 1)
    exact = lambda p: -np.stack([np.cos(p[:, 2]), np.cos(p[:, 0]), np.cos(p[:, 1])], 1)
    ax = np.linspace(0, 1, 21)
    for fld in (Field.on_grid([ax] * 3, f), Field.on_mesh(Mesh.structured([0] * 3, [1] * 3, (10, 10, 10)), f)):
        c = curl(fld)
        assert c.components == (3,)
        m = ((fld.points > 0.2) & (fld.points < 0.8)).all(axis=1)
        assert np.abs(c.values - exact(fld.points))[m].max() < 2e-2


def test_curl_of_gradient_and_div_of_curl_vanish():
    ax = np.linspace(0, 1, 61)
    phi = Field.on_grid([ax, ax], S)
    c = curl(Field(grad(phi).values, phi.support))
    assert np.abs(c.values[100:-100]).max() < 1e-2


def test_wrong_inputs_raise():
    s = Field.on_grid([np.linspace(0, 1, 9)] * 2, S)
    with pytest.raises(ValueError):
        div(s)
    with pytest.raises(ValueError):
        curl(s)
    with pytest.raises(ValueError):
        hessian(Field(V(s.points), s.support))
    with pytest.raises(TypeError):
        grad(np.zeros(3))
    with pytest.raises(ValueError):
        flux(s)
    with pytest.raises(ValueError):
        grad(lambda x: x[:, 0])  # continuous needs x


# -- flux and the divergence theorem ----------------------------------------
def test_flux_mesh_exact_for_linear_field():
    m = Mesh.structured([0, 0], [2, 1], (6, 4))
    f = Field.on_mesh(m, lambda p: np.stack([1 + p[:, 0], 2 * p[:, 1]], 1))  # div = 3, area = 2
    assert flux(f) == pytest.approx(3 * 2)
    assert flux(f, "x_max") == pytest.approx(3.0 * 1)  # u = 3 on x = 2, length 1
    assert flux(f, "x_min") == pytest.approx(-1.0 * 1)
    assert flux(f, lambda p: p[:, 1] > 0.999) == pytest.approx(2.0 * 2)  # top: v = 2, length 2


def test_flux_equals_integral_of_divergence_on_all_representations():
    # integral of div v over [0,1]^2: (int 2cos2x)(int cos y) + (int 2cos x)(int cos 2y) = 2 sin(2) sin(1)
    exact = 2 * np.sin(2) * np.sin(1)
    grid = Field.on_grid([np.linspace(0, 1, 81)] * 2, V)
    mesh = Field.on_mesh(Mesh.structured([0, 0], [1, 1], (40, 40)), V)
    assert flux(grid) == pytest.approx(exact, abs=2e-3)
    assert flux(mesh) == pytest.approx(exact, abs=5e-3)
    assert integrate(div(mesh)) == pytest.approx(exact, abs=5e-3)
    cloud = Field.on_points(DOM.sample_interior(3000, seed=1), V, domain=DOM)
    assert flux(cloud, n=40_000) == pytest.approx(exact, abs=0.08)


def test_flux_grid_faces_and_3d():
    ax = np.linspace(0, 2, 21)
    ay = np.linspace(0, 1, 11)
    f = Field.on_grid([ax, ay], lambda p: np.stack([p[:, 0], np.zeros(len(p))], 1))
    assert flux(f, "x_max") == pytest.approx(2.0 * 1.0)
    assert flux(f, "x_min") == pytest.approx(0.0)
    assert flux(f) == pytest.approx(2.0)  # div = 1 over area 2
    with pytest.raises(ValueError):
        flux(f, "z_max")
    f3 = Field.on_mesh(Mesh.structured([0] * 3, [1] * 3, (3, 3, 3)), lambda p: p)  # div = 3, volume 1
    assert flux(f3) == pytest.approx(3.0)


def test_flux_over_sdf_domain():
    from pinneapple_design.geometry.csg import CSGCircle

    dom = Domain.from_sdf(CSGCircle(0.0, 0.0, 1.0))
    f = Field.on_points(dom.sample_interior(2000, seed=0), lambda p: p, domain=dom)  # div = 2, area pi
    assert flux(f, n=40_000) == pytest.approx(2 * np.pi, rel=0.05)


# -- continuous --------------------------------------------------------------
torch = pytest.importorskip("torch")


def test_continuous_operators_match_analytic():
    s = FunctionField(lambda x: torch.sin(2 * x[:, 0]) * torch.cos(3 * x[:, 1]), DOM)
    v = FunctionField(lambda x: torch.stack([torch.sin(2 * x[:, 0]) * torch.cos(x[:, 1]),
                                             torch.cos(x[:, 0]) * torch.sin(2 * x[:, 1])], 1), DOM)
    x = torch.tensor(DOM.sample_interior(50, seed=0))
    xn = x.numpy()
    t = lambda a: torch.as_tensor(a)
    assert torch.allclose(grad(s, x), t(GS(xn)))
    assert torch.allclose(laplacian(s, x), t(LS(xn)))
    assert torch.allclose(hessian(s, x), t(HS(xn)))
    assert torch.allclose(div(v, x), t(DV(xn)))
    assert torch.allclose(curl(v, x), t(CV(xn)))
    assert torch.allclose(jacobian(v, x), t(JV(xn)))
    assert jacobian(s, x).shape == (50, 1, 2)
    assert laplacian(v, x).shape == (50, 2)


def test_continuous_accepts_plain_callable_and_3d_curl():
    f = lambda x: torch.stack([torch.sin(x[:, 1]), torch.sin(x[:, 2]), torch.sin(x[:, 0])], 1)
    x = torch.rand(20, 3, dtype=torch.float64)
    c = curl(f, x)
    assert torch.allclose(c, -torch.stack([torch.cos(x[:, 2]), torch.cos(x[:, 0]), torch.cos(x[:, 1])], 1))


def test_continuous_flux_and_integrate():
    v = FunctionField(lambda x: x, Domain.box([0, 0], [2, 1]))  # div = 2, area 2
    assert float(flux(v, n=40_000)) == pytest.approx(4.0, rel=0.03)
    assert float(flux(v, "x_max", n=40_000)) == pytest.approx(2.0 * 1.0, rel=0.05)
    assert float(integrate(FunctionField(lambda x: x[:, 0], v.domain), n=50_000)) == pytest.approx(2.0, rel=0.02)
    with pytest.raises(ValueError):
        flux(lambda x: x)


def test_top_level_namespace_exposes_operators():
    import pinneapple as pp

    assert pp.grad is grad and pp.flux is flux and pp.laplacian is laplacian
