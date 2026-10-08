import numpy as np
import pytest

torch = pytest.importorskip("torch")

import pinneapple as pp
from pinneapple_core.optim import Constraint, OptimizeResult, PhysicsOptimizer, eq, ineq, list_methods


def _quadratic(**kw):
    return PhysicsOptimizer({"x": (-3, 3), "y": (-3, 3)}, lambda v: (v[0] - 2) ** 2 + (v[1] - 1) ** 2,
                            [ineq(lambda v: v[0] + v[1] - 2)], **kw)              # optimum (1.5, 0.5), f = 0.5


GRAD = lambda z: np.array([2 * (z[0] - 2), 2 * (z[1] - 1)])
TOL = {"slsqp": 1e-6, "adjoint": 1e-6, "augmented_lagrangian": 1e-3, "differential_evolution": 1e-3,
       "penalty": 3e-2, "bayesian": 2e-2}


@pytest.mark.parametrize("method", list_methods())
def test_every_method_solves_the_same_inequality_constrained_problem(method):
    res = _quadratic().minimize(method, grad=GRAD if method == "adjoint" else None)
    assert isinstance(res, OptimizeResult) and res.method == method and res.feasible
    assert res.x == pytest.approx([1.5, 0.5], abs=TOL[method] * 5)
    assert res.fun == pytest.approx(0.5, abs=TOL[method] * 5)
    assert res.violation <= 1e-3 and res.n_evals > 0 and res.history and res.wall_time_s >= 0


def test_unconstrained_optimum_inside_the_feasible_set_is_not_pushed_away():
    opt = PhysicsOptimizer({"x": (-3, 3), "y": (-3, 3)}, lambda v: (v[0] - 0.5) ** 2 + (v[1] + 0.5) ** 2,
                           [ineq(lambda v: v[0] + v[1] - 2)])
    for m in ("slsqp", "augmented_lagrangian", "differential_evolution"):
        assert opt.minimize(m).x == pytest.approx([0.5, -0.5], abs=1e-2)


def _torsion_problem():
    from pinneapple_core import Mesh
    from pinneapple_core.fem import integrate_p1, solve_poisson

    ref = Mesh.structured([0, 0], [1, 1], (8, 8))
    base, bn = torch.tensor(ref.points), ref.boundary_nodes()

    def rigidity(x):
        pts = base * x
        return integrate_p1(pts, ref.cells, solve_poisson(pts, ref.cells, 1.0, bn))

    opt = PhysicsOptimizer({"a": (0.5, 2.0, 1.6), "b": (0.5, 2.0, 0.8)}, lambda x: -rigidity(x),
                           [eq(lambda x: x[0] * x[1] - 1.0)])
    return opt, rigidity


def test_physics_problem_three_families_agree_on_the_square():
    torch.set_default_dtype(torch.float64)
    try:
        opt, rigidity = _torsion_problem()
        adjoint = lambda z: torch.func.grad(lambda x: -rigidity(x))(torch.tensor(z)).numpy()
        results = {m: opt.minimize(m, grad=adjoint if m == "adjoint" else None)
                   for m in ("slsqp", "adjoint", "augmented_lagrangian", "differential_evolution", "bayesian")}
    finally:
        torch.set_default_dtype(torch.float32)
    for m, r in results.items():
        assert r.feasible and r.x == pytest.approx([1.0, 1.0], abs=0.02), (m, r)
    assert len({r.family for r in results.values()}) == 5
    f = [r.fun for r in results.values()]
    assert max(f) - min(f) < 2e-4                       # the same optimal value from every family
    assert results["slsqp"].fun == pytest.approx(results["adjoint"].fun, abs=1e-12)


def test_dict_argument_numpy_backend_and_vector_constraints():
    opt = PhysicsOptimizer({"x": {"bounds": (0, 4), "x0": 3.0}, "y": (0, 4)},
                           lambda p: (p["x"] - 1) ** 2 + (p["y"] - 3) ** 2,
                           [ineq(lambda p: np.array([p["x"] - 2, p["y"] - 2.5]))],      # a vector of two constraints
                           backend="numpy", argument="dict")
    res = opt.minimize("slsqp")
    assert res.params == pytest.approx({"x": 1.0, "y": 2.5}, abs=1e-5) and res.feasible
    assert opt.minimize("differential_evolution").x == pytest.approx([1.0, 2.5], abs=1e-2)
    with pytest.raises(ValueError, match="backend='torch'"):
        opt.minimize("penalty")


def test_compare_runs_every_method_and_skips_adjoint_without_a_gradient():
    out = _quadratic().compare()
    assert set(out) == set(list_methods()) - {"adjoint"} and all(r.feasible for r in out.values())
    assert "adjoint" in _quadratic().compare(grad=GRAD)
    assert "feasible" in str(out["slsqp"])


def test_validation_errors():
    f = lambda v: v[0] ** 2
    with pytest.raises(ValueError, match="lower bound"):
        PhysicsOptimizer({"x": (1, 0)}, f)
    with pytest.raises(ValueError, match="x0"):
        PhysicsOptimizer({"x": (0, 1, 5)}, f)
    with pytest.raises(ValueError):
        PhysicsOptimizer({}, f)
    with pytest.raises(ValueError):
        PhysicsOptimizer({"x": (0, 1)}, f, backend="jax")
    with pytest.raises(ValueError):
        Constraint(f, "bogus")
    opt = PhysicsOptimizer({"x": (0, 1)}, f)
    with pytest.raises(ValueError, match="unknown method"):
        opt.minimize("magic")
    with pytest.raises(ValueError, match="grad="):
        opt.minimize("adjoint")


def test_infeasible_problem_is_reported_not_hidden():
    opt = PhysicsOptimizer({"x": (0, 1)}, lambda v: v[0], [ineq(lambda v: 2.0 - v[0])])     # needs x >= 2, impossible
    res = opt.minimize("differential_evolution")
    assert not res.feasible and res.violation > 0.5


def test_namespace():
    assert pp.PhysicsOptimizer is PhysicsOptimizer
