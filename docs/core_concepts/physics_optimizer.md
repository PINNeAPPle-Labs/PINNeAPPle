# PhysicsOptimizer

`PhysicsOptimizer(parameters, objective, constraints)` (`pinneapple_core.optim`, also `pp.PhysicsOptimizer`) states a
constrained design or calibration problem once and solves it with any of six methods. Every method returns the same
`OptimizeResult` (parameters, value, largest constraint violation, feasibility, evaluations, family, history, time).

```python
from pinneapple_core.optim import PhysicsOptimizer, eq, ineq

opt = PhysicsOptimizer({"a": (0.5, 2.0, 1.6), "b": (0.5, 2.0, 0.8)},         # name: (lo, hi, start)
                       objective=lambda x: -rigidity(x),                     # torch function of the vector x
                       constraints=[eq(lambda x: x[0] * x[1] - 1.0)])        # a*b = 1
opt.minimize("slsqp")                     # or "adjoint", "penalty", "augmented_lagrangian", "differential_evolution", "bayesian"
opt.compare()                             # all of them on the same problem
```

| Method | Family | Needs |
|---|---|---|
| `slsqp` | sequential quadratic programming with autograd gradients of objective and constraints | torch objective |
| `adjoint` | the same SQP with the objective gradient you supply (an adjoint solve, `implicit_solve`, `torch.func.grad`) | `grad=` |
| `penalty` | projected Adam on the objective plus a growing quadratic penalty | torch objective |
| `augmented_lagrangian` | projected Adam with multiplier updates, so constraints are met exactly | torch objective |
| `differential_evolution` | population search; equality constraints as a thin band of width `tol/2` | nothing |
| `bayesian` | Gaussian-process expected improvement times the probability of feasibility, one GP per constraint | nothing |

Constraints are `ineq(fn)` for `fn(x) <= 0` and `eq(fn)` for `fn(x) == 0`; a function may return a vector. With
`backend="numpy"` the functions are plain numpy and gradient methods use finite differences; `argument="dict"` passes the
parameters by name. A point is feasible when every violation is at most `tol` (default 1e-3); an impossible problem is
reported as `feasible=False`, never hidden.

## Reference problem

`examples/physics_optimizer/01_constrained_shape_six_methods.py`: the torsional rigidity (integral of `u` for
`-laplace(u) = 1`, `u = 0` on the boundary) of an `a x b` rectangle from the differentiable finite-element solver,
maximised at fixed area. The square `a = b = 1` is the known optimum. All six methods find it:

| Method | a, b | value | violation | evaluations |
|---|---|---|---|---|
| slsqp | 1, 1 | -0.0340297 | 1e-14 | 17 |
| adjoint | 1, 1 | -0.0340297 | 1e-14 | 17 |
| penalty | 1.001, 0.999 | -0.0340298 | 3e-6 | 3001 |
| augmented_lagrangian | 1, 1 | -0.0340297 | 2e-9 | 2001 |
| differential_evolution | 1, 1 | -0.0340637 | 5e-4 | 605 |
| bayesian | 0.998, 1.003 | -0.0340975 | 1e-3 | 61 |

The gradient methods are the cheapest in evaluations when gradients exist; Bayesian optimisation reaches the same answer in
61 evaluations without gradients, which is the one to use when each evaluation is an expensive simulation. Optimal control
and MPC are not part of this interface (roadmap X32); the existing `pinneapple_design.design_optimizer` pipeline is unchanged.
