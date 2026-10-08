"""A small problem written once against PhysicsBackend runs unchanged on torch and jax."""
import numpy as np
import pytest

import pinneapple as pp
from pinneapple_core.backend import (
    PhysicsBackend, get_physics_backend, list_backends, register_backend, use_backend,
)

pytest.importorskip("torch")


def poisson_objective(bk, L, n=64):
    """-u'' = 1 on (0, L), u = 0 at both ends, centred finite differences; returns the integral of u.
    Written with the backend API only."""
    m = n - 1                                    # interior unknowns
    h = L / n
    A = (2.0 * bk.eye(m) - bk.diag(bk.ones(m - 1), 1) - bk.diag(bk.ones(m - 1), -1)) / h ** 2
    u = bk.solve(A, bk.ones(m))
    x = bk.linspace(0.0, 1.0, m) * (L - 2 * h) + h  # interior nodes
    return bk.integrate(u, x)


def test_same_problem_value_and_gradient_on_both_backends():
    pytest.importorskip("jax")
    results = {}
    for name in ("torch", "jax"):
        bk = get_physics_backend(name)
        L = bk.asarray(1.7)
        value = poisson_objective(bk, L)
        g = bk.grad(lambda L_: poisson_objective(bk, L_))(L)
        results[name] = (float(bk.to_numpy(value)), float(bk.to_numpy(g)))
    assert results["torch"][0] == pytest.approx(results["jax"][0], rel=1e-10)
    assert results["torch"][1] == pytest.approx(results["jax"][1], rel=1e-8)
    # and both are right: J = L^3/12, dJ/dL = L^2/4 up to the O(h^2) quadrature of the interior nodes
    assert results["torch"][1] == pytest.approx(1.7 ** 2 / 4, rel=2e-2)


def test_jit_vmap_jacobian_and_integrate_agree():
    pytest.importorskip("jax")
    out = {}
    for name in ("torch", "jax"):
        bk = get_physics_backend(name)
        f = lambda a: bk.xp.sin(a) * bk.asarray([1.0, 2.0])
        jac = bk.jacobian(f)(bk.asarray(0.3))
        batch = bk.vmap(f)(bk.asarray([0.1, 0.2, 0.3]))
        x = bk.linspace(0, np.pi, 201)
        integral = bk.jit(lambda y, x_: bk.integrate(y, x_))(bk.xp.sin(x), x)
        out[name] = (bk.to_numpy(jac), bk.to_numpy(batch), float(bk.to_numpy(integral)))
    assert np.allclose(out["torch"][0], out["jax"][0]) and np.allclose(out["torch"][1], out["jax"][1])
    assert out["torch"][2] == pytest.approx(2.0, abs=1e-4) == pytest.approx(out["jax"][2], abs=1e-4)
    assert out["torch"][1].shape == (3, 2)


def test_linspace_is_differentiable_in_its_endpoint_on_every_backend():
    pytest.importorskip("jax")
    for name in ("torch", "jax"):
        bk = get_physics_backend(name)
        g = bk.grad(lambda L: bk.linspace(0.0, L, 11).sum())(bk.asarray(2.0))
        assert float(bk.to_numpy(g)) == pytest.approx(5.5)  # d/dL of sum(i L / 10) for i = 0..10
        assert np.allclose(bk.to_numpy(bk.linspace(1.0, 3.0, 5)), [1.0, 1.5, 2.0, 2.5, 3.0])
    assert float(get_physics_backend("torch").linspace(0, 1, 1)[0]) == 0.0


def test_set_backend_and_use_backend_switch_the_active_engine():
    from pinneapple_tools.compute_backends.backend import get_backend, set_backend

    original = get_backend()
    try:
        set_backend("torch")
        assert get_physics_backend().name == "torch"
        with use_backend("torch") as bk:
            assert bk.name == "torch" and get_backend() == "torch" and pp.get_physics_backend().name == "torch"
        set_backend("jax")  # selecting by name works even before jax is imported
        assert get_backend() == "jax"
        set_backend("torch")
        with pytest.raises(ValueError, match="Unknown backend"):
            set_backend("not-a-real-backend")
    finally:
        set_backend(original)


def test_register_a_new_backend():
    class NumpyBackend(PhysicsBackend):
        name = "numpy_test"
        xp = np

        def asarray(self, x, dtype=None): return np.asarray(x, dtype=dtype or np.float64)
        def to_numpy(self, x): return np.asarray(x)
        def solve(self, a, b): return np.linalg.solve(a, b)

    register_backend("numpy_test", NumpyBackend)
    assert "numpy_test" in list_backends()
    bk = get_physics_backend("numpy_test")
    assert bk.integrate([0.0, 1.0, 2.0], dx=1.0) == pytest.approx(2.0)
    assert np.allclose(bk.solve(np.eye(2) * 2, np.array([2.0, 4.0])), [1.0, 2.0])
    with pytest.raises(NotImplementedError):
        bk.grad(lambda x: x)
    with pytest.raises(ValueError):
        register_backend("numpy_test", NumpyBackend)
    with pytest.raises(TypeError):
        register_backend("bad", object)
