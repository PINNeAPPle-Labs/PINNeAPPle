"""pp.compile: derivative reuse gives the same losses and weights with fewer autograd graphs."""
import pytest

torch = pytest.importorskip("torch")

import pinneapple as pp
from pinneapple_physics.compile_api import Benchmark, CompiledProblem
from pinneapple_physics.pinn_solver.compiler import autograd_ops as ao

SMALL = {"epochs": 12, "n_collocation": 128, "hidden_dim": 16, "n_layers": 2}


def test_derivative_cache_reuses_the_same_graph_and_is_scoped():
    x = torch.rand(8, 2, requires_grad=True)
    y = (x ** 3).sum(dim=1, keepdim=True)
    with ao.derivative_cache() as stats:
        g1 = ao.grad(y, x)
        g2 = ao.grad(y, x)
        assert g1 is g2 and (stats["hits"], stats["misses"]) == (1, 1)
        assert ao.grad(y * 2, x) is not g1                      # a different output is a different derivative
    assert ao.grad(y, x) is not ao.grad(y, x)                   # outside the context nothing is cached
    assert torch.allclose(g1, 3 * x ** 2)


def test_jacobian_cache():
    x = torch.rand(5, 2, requires_grad=True)
    Y = torch.cat([x[:, :1] ** 2, x[:, 1:] * 3], dim=1)
    with ao.derivative_cache() as stats:
        a, b = ao.jacobian(Y, x), ao.jacobian(Y, x)
    assert a is b and stats["hits"] == 1


def test_compiled_loss_equals_uncompiled_loss():
    import pinneapple_neural.architectures  # noqa: F401
    from pinneapple_neural.architectures.registry import ModelRegistry
    from pinneapple_physics.pinn_solver.compiler.compile import compile_problem

    spec = pp.PhysicalProblem.from_preset("burgers_1d").to_pde_spec()
    torch.manual_seed(0)
    model = ModelRegistry.build("modified_mlp", in_dim=2, out_dim=1, hidden_dim=16, n_layers=2)
    batch = {"x_col": torch.rand(64, 2) * 2 - 1}
    base = compile_problem(spec)(model, None, dict(batch))
    fast = compile_problem(spec, cache_derivatives=True)(model, None, dict(batch))
    assert set(base) == set(fast)
    for k in base:
        assert torch.allclose(torch.as_tensor(base[k]), torch.as_tensor(fast[k]), rtol=1e-6, atol=1e-8), k


def test_compile_solve_matches_plain_solve_and_benchmark_reports_identical_weights():
    compiled = pp.compile("burgers_1d", optimize="physics")
    assert isinstance(compiled, CompiledProblem) and compiled.options == {"cache_derivatives": True}
    a = compiled.solve("pinn", **SMALL)
    b = pp.solve("burgers_1d", "pinn", **SMALL)
    X = torch.rand(50, 2).numpy() * 2 - 1
    assert abs(a.predict(X) - b.predict(X)).max() < 1e-5
    bench = compiled.benchmark(epochs=6, repeats=1, n_collocation=128, hidden_dim=16, n_layers=2)
    assert isinstance(bench, Benchmark) and bench.identical() and bench.speedup > 0
    assert bench.derivative_calls_saved >= 1 and "speedup" in str(bench)


def test_compile_options_and_errors():
    assert pp.compile("burgers_1d", optimize="none").options == {"cache_derivatives": False}
    with pytest.raises(ValueError):
        pp.compile("burgers_1d", optimize="bogus")
    with pytest.raises(TypeError, match="bare torch model"):
        pp.compile(torch.nn.Linear(2, 1))
    assert pp.compile("burgers_1d").loss_fn() is not None
