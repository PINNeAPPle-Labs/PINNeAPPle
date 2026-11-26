# Solve a problem

The shortest path from a PDE to a number you can defend: describe the problem, solve it with two methods, and
compare them against a reference. Everything here runs in about ten seconds on a laptop.

## The four lines

```python
import numpy as np
import pinneapple as pp
from pinneapple_physics.pde_environment.builder import ProblemBuilder

# 1. Describe the problem: laplace(u) = f on the unit square, u = 0 on the boundary.
spec = (ProblemBuilder("poisson").domain(x=(0, 1), y=(0, 1)).fields("u").pde("poisson")
        .bc("dirichlet", field="u", value=0.0, on="x_boundary")
        .bc("dirichlet", field="u", value=0.0, on="y_boundary").build())
problem = pp.PhysicalProblem.from_pde_spec(spec)

# 2. The source term and the exact solution this case has, so the answer can be graded.
ctx = {"source_fn": lambda X, ctx=None: -2 * np.pi**2 * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])}
exact = lambda X: (np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1]))[:, None]

# 3. Solve it twice: a classical solver and a network.
fem = pp.solve(problem, "fem", ctx=ctx, n=24)
pinn = pp.solve(problem, "pinn", ctx=ctx, epochs=200, hidden_dim=16, n_layers=2, n_collocation=256)

# 4. Compare on the same points, against the same reference.
table = pp.compare(problem, [fem, pinn], reference="exact", options={"exact": {"fn": exact}})
print(table)
```

On a laptop, that prints:

```text
Comparison on 'poisson' against 'exact' at 4096 points
method          kind       status     train [s]  infer [us/pt]  ref sims  relL2(u)
fem             classical  ok            0.0356           4.89         0  7.540e-03
pinn            neural     ok              2.83          0.463         0  5.637e-02
```

The finite-element solution is the more accurate of the two here (7.5e-3 against 5.6e-2 relative L2) and takes a
fraction of the training time; the network costs almost nothing per point once trained. That is the trade the
catalogue exists to make visible — on this problem, at this budget, the classical method wins, and the table says so
instead of the README claiming otherwise.

## What just happened

| Line | What it does |
|---|---|
| `ProblemBuilder(...)` | Names the domain, the field, the PDE and the boundary conditions; `.build()` returns a spec that is checked for consistency before anything runs. |
| `pp.PhysicalProblem.from_pde_spec` | Turns the spec into the object every method takes. |
| `ctx["source_fn"]` | The right-hand side of the PDE, as a function of the sampled points. |
| `pp.solve(problem, "fem", ...)` | A classical solver through the same interface as any other method. |
| `pp.solve(problem, "pinn", ...)` | A physics-informed network, with the same problem object and no adapter code. |
| `pp.compare(...)` | Evaluates both on the same points against the same reference and adds cost columns. |

## Where to go from here

| You want to | Guide |
|---|---|
| pick the method before writing code | [Choosing a method](choosing_a_method.md), or `pp.recommend("...")` |
| see what a preset takes and gives back | `pp.card(name)`, [pre-trained models and datasets](model_zoo.md) |
| find the parameters of a model from observations | [Identifiability, calibration and inversion](identifiability_calibration_inversion.md) |
| optimise a design rather than solve one | [Design studies](design_studies.md) |
| validate a trained model before trusting it | [`pp.validate`](pp_validate.md) |
| look at the field | [Studio](../core_concepts/studio.md), `pp.viz.field_view(...)` |
| run many cases and keep the evidence | [Running a benchmark](running_a_benchmark.md) |
| check a result in CI | [Physics checks in CI](physics_checks_in_ci.md) |

Every method above is listed with its validated status in `pinneapple_catalog.validated_methods()`; a method that
appears there has a test and a reference behind it, and one that does not says so.
