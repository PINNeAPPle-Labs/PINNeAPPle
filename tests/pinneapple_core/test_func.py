import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinneapple_core import Mesh, func
from pinneapple_core.fem import integrate_p1, mass, solve_poisson, stiffness

D = torch.float64


# -- transforms --------------------------------------------------------------
def test_wrt_by_name_position_and_sequence():
    f = lambda geometry, mu: (mu * geometry ** 2).sum()
    g = torch.tensor([1.0, 2.0], dtype=D)
    mu = torch.tensor(3.0, dtype=D)
    assert torch.allclose(func.grad(f, wrt="geometry")(g, mu), 2 * mu * g)
    assert torch.allclose(func.grad(f, wrt=0)(g, mu), 2 * mu * g)
    assert torch.allclose(func.grad(f, wrt="mu")(g, mu), (g ** 2).sum())
    gg, gm = func.grad(f, wrt=["geometry", "mu"])(g, mu)
    assert torch.allclose(gg, 2 * mu * g) and torch.allclose(gm, (g ** 2).sum())
    with pytest.raises(ValueError):
        func.grad(f, wrt="nope")


def test_jacobian_modes_agree_and_hessian_vmap():
    f = lambda x: torch.stack([x[0] * x[1], x[0] ** 2, torch.sin(x[1])])
    x = torch.tensor([0.5, 1.5], dtype=D)
    exact = torch.tensor([[1.5, 0.5], [1.0, 0.0], [0.0, np.cos(1.5)]], dtype=D)
    for mode in ("auto", "rev", "fwd"):
        assert torch.allclose(func.jacobian(f, mode=mode)(x), exact)
    assert torch.allclose(func.jacrev(f)(x), func.jacfwd(f)(x))
    h = func.hessian(lambda x: (x ** 3).sum())(x)
    assert torch.allclose(h, torch.diag(6 * x))
    batch = torch.rand(5, 2, dtype=D)
    assert func.vmap(func.jacobian(f))(batch).shape == (5, 3, 2)
    with pytest.raises(ValueError):
        func.jacobian(f, mode="bad")


# -- implicit differentiation -------------------------------------------------
def _problem():
    A = torch.tensor([[3.0, 1.0], [1.0, 2.0]], dtype=D)
    res = lambda x, b, c: A @ x + c * x ** 3 - b
    return res


def test_implicit_solve_gradient_matches_theory_and_finite_differences():
    res = _problem()
    b = torch.tensor([1.0, 2.0], dtype=D, requires_grad=True)
    c = torch.tensor(0.5, dtype=D, requires_grad=True)
    x0 = torch.zeros(2, dtype=D)
    x = func.implicit_solve(res, x0, b, c)
    assert res(x, b, c).abs().max() < 1e-9
    J = torch.autograd.functional.jacobian(lambda b_, c_: func.implicit_solve(res, x0, b_, c_), (b, c))
    # theory: dx/db = (dF/dx)^-1, dx/dc = -(dF/dx)^-1 x^3
    xs = x.detach()
    dFdx = torch.tensor([[3.0, 1.0], [1.0, 2.0]], dtype=D) + 3 * c.detach() * torch.diag(xs ** 2)
    assert torch.allclose(J[0], torch.linalg.inv(dFdx), atol=1e-9)
    assert torch.allclose(J[1], -torch.linalg.solve(dFdx, xs ** 3), atol=1e-9)
    eps = 1e-6
    fd = (func.implicit_solve(res, x0, b.detach(), c.detach() + eps)
          - func.implicit_solve(res, x0, b.detach(), c.detach() - eps)) / (2 * eps)
    assert torch.allclose(J[1], fd, atol=1e-6)


def test_implicit_solve_with_external_solver_and_scalar_loss():
    from scipy.optimize import fsolve

    res = _problem()
    calls = []

    def external(r, x0, b, c):  # a black box: scipy, never differentiated
        calls.append(1)
        f = lambda z: r(torch.tensor(z, dtype=D), b, c).numpy()
        return torch.tensor(fsolve(f, x0.numpy(), xtol=1e-13), dtype=D)

    b = torch.tensor([1.0, 2.0], dtype=D, requires_grad=True)
    c = torch.tensor(0.5, dtype=D)
    loss = func.implicit_solve(res, torch.zeros(2, dtype=D), b, c, solver=external).pow(2).sum()
    loss.backward()
    ref = torch.tensor([1.0, 2.0], dtype=D, requires_grad=True)
    func.implicit_solve(res, torch.zeros(2, dtype=D), ref, c).pow(2).sum().backward()
    assert calls and torch.allclose(b.grad, ref.grad, atol=1e-8)


def test_newton_reports_non_convergence():
    with pytest.raises(RuntimeError):
        func.newton_solve(lambda x: x ** 2 + 1.0, torch.tensor([1.0], dtype=D), max_iter=10)


# -- differentiable FEM and geometry gradients ---------------------------------
def test_fem_poisson_is_accurate_and_matrices_are_consistent():
    m = Mesh.structured([0, 0], [1, 1], (24, 24))
    pts = torch.tensor(m.points, dtype=D)
    src = lambda p: 2 * np.pi ** 2 * torch.sin(np.pi * p[:, 0]) * torch.sin(np.pi * p[:, 1])
    u = solve_poisson(pts, m.cells, src, m.boundary_nodes())
    exact = torch.sin(np.pi * pts[:, 0]) * torch.sin(np.pi * pts[:, 1])
    assert (u - exact).abs().max() < 5e-3
    assert mass(pts, m.cells).sum() == pytest.approx(1.0)
    ones = torch.ones(pts.shape[0], dtype=D)
    assert stiffness(pts, m.cells).matmul(ones).abs().max() < 1e-12


def test_geometry_gradient_1d_length_matches_finite_differences_and_theory():
    ref = Mesh.interval(0.0, 1.0, 40)
    bn = ref.boundary_nodes()

    def J(L):
        pts = torch.tensor(ref.points, dtype=D) * L
        return integrate_p1(pts, ref.cells, solve_poisson(pts, ref.cells, 1.0, bn))

    L = torch.tensor(1.7, dtype=D, requires_grad=True)
    J(L).backward()
    eps = 1e-6
    fd = (J(torch.tensor(1.7 + eps, dtype=D)) - J(torch.tensor(1.7 - eps, dtype=D))) / (2 * eps)
    assert L.grad.item() == pytest.approx(fd.item(), rel=1e-7)
    assert L.grad.item() == pytest.approx(1.7 ** 2 / 4, rel=1e-3)  # J = L^3/12 for -u'' = 1


def test_geometry_gradient_2d_shape_parameters_match_finite_differences():
    ref = Mesh.structured([0, 0], [1, 1], (10, 10))
    bn = ref.boundary_nodes()
    base = torch.tensor(ref.points, dtype=D)

    def J(theta):  # rectangle a x b with a bump of height h on top
        a, b, h = theta
        x, y = base[:, 0], base[:, 1]
        pts = torch.stack([a * x, b * y * (1 + h * torch.sin(np.pi * x))], dim=1)
        return integrate_p1(pts, ref.cells, solve_poisson(pts, ref.cells, 1.0, bn))

    theta = torch.tensor([1.3, 0.8, 0.2], dtype=D, requires_grad=True)
    J(theta).backward()
    fd = []
    for i in range(3):
        e = torch.zeros(3, dtype=D)
        e[i] = 1e-6
        fd.append(((J(theta.detach() + e) - J(theta.detach() - e)) / 2e-6).item())
    assert np.allclose(theta.grad.numpy(), fd, rtol=1e-6)


def test_jacobian_of_model_wrt_geometry():
    ref = Mesh.structured([0, 0], [1, 1], (6, 6))
    bn = ref.boundary_nodes()
    base = torch.tensor(ref.points, dtype=D)

    def model(geometry, mu):
        pts = base * geometry
        return mu * solve_poisson(pts, ref.cells, 1.0, bn)

    g = torch.tensor([1.2, 0.9], dtype=D)
    mu = torch.tensor(2.0, dtype=D)
    jac = func.jacobian(model, wrt="geometry")(g, mu)
    assert jac.shape == (base.shape[0], 2)
    e = torch.tensor([1e-6, 0.0], dtype=D)
    fd = (model(g + e, mu) - model(g - e, mu)) / 2e-6
    assert torch.allclose(jac[:, 0], fd, atol=1e-6)


def test_namespace_pp_func():
    import pinneapple as pp

    assert pp.func.grad is func.grad
