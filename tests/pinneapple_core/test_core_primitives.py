import numpy as np
import pytest

from pinneapple_core import Domain, Field, FunctionField, Geometry, Mesh, PointCloud


# -- Domain ------------------------------------------------------------------
def test_box_domain_sampling_and_volume():
    d = Domain.box([0, -1, 0], [2, 1, 3])
    assert d.dim == 3 and d.volume() == pytest.approx(12.0)
    p = d.sample_interior(500, seed=1)
    assert p.shape == (500, 3) and d.contains(p).all()
    b, n = d.sample_boundary(600, seed=2, return_normals=True)
    assert np.allclose(np.abs(d.sdf(b)), 0, atol=1e-12)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)
    assert not d.contains(np.array([[5.0, 0, 0]])).any()


def test_box_boundary_is_uniform_in_measure():
    d = Domain.box([0, 0], [4, 1])  # long faces carry 4x the length of short ones
    b = d.sample_boundary(20000, seed=0)
    on_long = np.isclose(b[:, 1], 0) | np.isclose(b[:, 1], 1)
    assert on_long.mean() == pytest.approx(8 / 10, abs=0.02)


def test_sdf_domain_from_csg():
    from pinneapple_design.geometry.csg import annulus

    d = Domain.from_sdf(annulus(0.0, 0.0, 0.5, 1.0))
    assert d.volume() == pytest.approx(np.pi * (1 - 0.25), rel=0.02)
    p = d.sample_interior(300, seed=3)
    r = np.linalg.norm(p, axis=1)
    assert (r >= 0.5 - 1e-6).all() and (r <= 1 + 1e-6).all()
    b, n = d.sample_boundary(200, return_normals=True)
    assert np.allclose(np.abs(d.sdf(b)), 0, atol=1e-2) and np.allclose(np.linalg.norm(n, axis=1), 1, atol=1e-3)


def test_domain_validation():
    with pytest.raises(ValueError):
        Domain.box([0, 0], [1, 0])
    with pytest.raises(ValueError):
        Domain.box([0, 0], [1, 1]).contains(np.zeros((3, 3)))


# -- Mesh --------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(5,), (4, 3), (3, 2, 2)])
def test_structured_mesh_volume_and_boundary(shape):
    lo, hi = np.zeros(len(shape)), np.array([1.0, 2.0, 3.0][: len(shape)])
    m = Mesh.structured(lo, hi, shape)
    assert m.volume == pytest.approx(np.prod(hi))
    assert (m.cell_volumes > 0).all()
    # every boundary node lies on a face of the box
    bn = m.points[m.boundary_nodes()]
    on_face = np.isclose(bn, lo).any(axis=1) | np.isclose(bn, hi).any(axis=1)
    assert on_face.all()
    # and every point on a face is a boundary node
    pts = m.points
    all_face = np.isclose(pts, lo).any(axis=1) | np.isclose(pts, hi).any(axis=1)
    assert all_face.sum() == m.boundary_nodes().size


def test_mesh_gradient_exact_for_linear_and_integral_exact_for_p1():
    m = Mesh.structured([0, 0, 0], [1, 2, 1], (3, 4, 2))
    a = np.array([2.0, -3.0, 0.5])
    u = m.points @ a + 1.0
    assert np.allclose(m.gradient(u)[:, 0, :], a)
    assert np.allclose(m.cell_gradient(u)[:, 0, :], a)
    assert m.integrate(u)[0] == pytest.approx(2.0 * (a @ [0.5, 1.0, 0.5] + 1.0))


def test_mesh_interpolation_exact_for_linear_and_nan_outside():
    m = Mesh.structured([0, 0], [1, 1], (6, 6))
    u = 3 * m.points[:, 0] - m.points[:, 1] + 0.25
    q = np.random.default_rng(0).random((200, 2))
    assert np.allclose(m.interpolate(u, q)[:, 0], 3 * q[:, 0] - q[:, 1] + 0.25)
    out = m.interpolate(u, np.array([[1.5, 0.5], [0.5, 0.5]]))
    assert np.isnan(out[0]).all() and not np.isnan(out[1]).any()
    near = m.interpolate(u, np.array([[1.5, 0.5]]), fill="nearest")
    assert not np.isnan(near).any()


def test_mesh_rejects_bad_input():
    with pytest.raises(ValueError):
        Mesh(np.zeros((3, 2)), np.array([[0, 1, 5]]))
    with pytest.raises(ValueError):
        Mesh(np.array([[0.0, 0], [1, 0], [2, 0]]), np.array([[0, 1, 2]]))  # collinear


# -- Geometry ----------------------------------------------------------------
def test_geometry_named_boundaries_and_tags():
    g = Geometry.box([0, 0], [2, 1])
    assert g.boundary_names == ["x_min", "x_max", "y_min", "y_max"]
    p = g.sample_boundary(50, "x_max", seed=0)
    assert np.allclose(p[:, 0], 2.0)
    tags = g.tag(np.array([[0.0, 0.5], [2.0, 0.5], [1.0, 0.5]]))
    assert list(tags) == ["x_min", "x_max", ""]
    with pytest.raises(KeyError):
        g.sample_boundary(5, "nope")


def test_geometry_meshes_a_box_and_an_lshape():
    from pinneapple_design.geometry.csg import lshape

    box = Geometry.box([0, 0], [1, 1]).mesh(0.1)
    assert box.volume == pytest.approx(1.0)
    shape = Geometry.from_csg(lshape(2.0, 2.0, 1.0, 1.0))
    m = shape.mesh(0.15)
    assert m.volume == pytest.approx(shape.domain.volume(), rel=0.1)
    assert shape.domain.contains(m.cell_centroids).all()


# -- Field: grid -------------------------------------------------------------
def test_grid_field_calculus_1d_2d():
    x = np.linspace(0, 1, 201)
    f = Field.on_grid([x], np.sin(np.pi * x))
    assert f.integrate() == pytest.approx(2 / np.pi, abs=1e-4)
    assert np.allclose(f.gradient().values[:, 0], np.pi * np.cos(np.pi * x), atol=2e-3)
    assert f.interpolate(np.array([[0.25]]))[0] == pytest.approx(np.sin(np.pi / 4), abs=1e-4)

    y = np.linspace(0, 2, 41)
    xs = np.linspace(0, 1, 31)
    u = Field.on_grid([xs, y], lambda p: np.stack([p[:, 0] ** 2, 3 * p[:, 1]], axis=1), name="u")
    assert u.components == (2,)
    assert np.allclose(u.divergence().values, 2 * u.points[:, 0] + 3, atol=1e-8)
    assert u.divergence().name == "div(u)"
    s = Field.on_grid([xs, y], lambda p: p[:, 0] * p[:, 1])
    assert s.integrate() == pytest.approx(0.5 * 2.0, abs=1e-9)  # int x dx * int y dy = 0.5 * 2
    assert s.gradient().components == (2,)


def test_grid_accepts_grid_shaped_values_and_vector_integral():
    x, y = np.linspace(0, 1, 11), np.linspace(0, 1, 6)
    arr = np.ones((11, 6, 2)) * np.array([1.0, 2.0])
    f = Field.on_grid([x, y], arr)
    assert np.allclose(f.integrate(), [1.0, 2.0])
    assert f.gradient().components == (2, 2)


# -- Field: mesh -------------------------------------------------------------
def test_mesh_field_divergence_and_integral():
    m = Mesh.structured([0, 0], [1, 1], (10, 10))
    u = Field.on_mesh(m, lambda p: np.stack([2 * p[:, 0], -p[:, 1] + p[:, 0]], axis=1))
    assert np.allclose(u.divergence().values, 1.0)
    s = Field.on_mesh(m, lambda p: p[:, 0] + p[:, 1])
    assert s.integrate() == pytest.approx(1.0)


# -- Field: point cloud ------------------------------------------------------
def test_point_cloud_gradient_interpolation_integral():
    dom = Domain.box([0, 0], [1, 1])
    pts = dom.sample_interior(3000, seed=4)
    f = Field.on_points(pts, lambda p: 2 * p[:, 0] - 3 * p[:, 1] + 1, domain=dom)
    assert np.allclose(f.gradient().values, [2.0, -3.0], atol=1e-6)
    q = np.array([[0.5, 0.5], [0.3, 0.7]])
    assert np.allclose(f.interpolate(q), 2 * q[:, 0] - 3 * q[:, 1] + 1, atol=2e-2)
    assert f.integrate() == pytest.approx(0.5, abs=0.05)  # Monte Carlo
    w = Field.on_points(pts, f.values, weights=np.full(3000, 1 / 3000))
    assert w.integrate() == pytest.approx(f.values.mean())
    with pytest.raises(ValueError):
        Field.on_points(pts, f.values).integrate()


def test_point_cloud_1d_interpolation():
    x = np.sort(np.random.default_rng(0).random(50)) * 2
    f = Field.on_points(x[:, None], 3 * x)
    assert f.interpolate(np.array([[1.0]]))[0] == pytest.approx(3.0)
    assert np.isnan(f.interpolate(np.array([[5.0]]))).all()


def test_field_rejects_wrong_sizes_and_bad_divergence():
    with pytest.raises(ValueError):
        Field.on_points(np.zeros((4, 2)), np.zeros(3))
    f = Field.on_grid([np.linspace(0, 1, 5)] * 2, lambda p: p[:, 0])
    with pytest.raises(ValueError):
        f.divergence()


# -- FunctionField (torch) ---------------------------------------------------
torch = pytest.importorskip("torch")


def test_function_field_derivatives_match_analytic():
    dom = Domain.box([0, 0], [1, 1])
    u = FunctionField(lambda x: torch.sin(x[:, 0]) * x[:, 1] ** 2, dom)
    x = torch.tensor(dom.sample_interior(64, seed=0))
    g = u.gradient(x)
    assert torch.allclose(g[:, 0], torch.cos(x[:, 0]) * x[:, 1] ** 2)
    assert torch.allclose(g[:, 1], 2 * torch.sin(x[:, 0]) * x[:, 1])
    lap = u.laplacian(x)
    assert torch.allclose(lap, -torch.sin(x[:, 0]) * x[:, 1] ** 2 + 2 * torch.sin(x[:, 0]))

    v = FunctionField(lambda x: torch.stack([x[:, 0] ** 2, x[:, 0] * x[:, 1]], dim=1), dom)
    assert torch.allclose(v.divergence(x), 2 * x[:, 0] + x[:, 0])
    assert v.gradient(x).shape == (64, 2, 2)


def test_function_field_keeps_graph_for_training():
    net = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    u = FunctionField(net, Domain.box([0, 0], [1, 1]))
    x = torch.rand(16, 2)
    loss = (u.laplacian(x) ** 2).mean()
    loss.backward()
    # the output bias is a constant: it cannot reach the Laplacian, every other parameter does
    grads = [p.grad for p in net.parameters()][:-1]
    assert all(g is not None and g.abs().sum() > 0 for g in grads)


def test_function_field_integrate_and_to_field():
    dom = Domain.box([0, 0], [2, 1])
    u = FunctionField(lambda x: x[:, 0] * x[:, 1], dom)
    assert float(u.integrate(n=100_000)) == pytest.approx(1.0, abs=0.02)
    m = Mesh.structured([0, 0], [2, 1], (8, 4))
    f = u.to_field(m)
    assert f.integrate() == pytest.approx(1.0, abs=0.02)  # x*y is bilinear: P1 is not exact
    assert u.to_field(PointCloud(m.points)).values.shape == (m.n_points,)
