# From an inverse model to an equation: symbolic distillation

A PINN with an unknown term, a neural ODE or a neural operator learns a law and hides it in weights. The equation
is the part anyone else can read, publish and solve again. `pinneapple_analysis.inverse_problems.symbolic_distillation`
reads it back out of the model and does the part that decides whether the equation is worth anything: it puts the
equation into a solver and compares **dynamics**, not two evaluations.

```python
import numpy as np
from pinneapple_analysis.inverse_problems.symbolic_distillation import (
    DistillationConfig, Support, distill, field_dataset, simulate_evolution, verify)

data = field_dataset(surface, points, coords=["x", "t"],        # surface: a network, a PolynomialSurface, ...
                     columns={"u": ("u", ()), "u_x": ("u", ("x",)), "u_xx": ("u", ("x", "x")),
                              "u_t": ("u", ("t",))},             # read through autograd, never finite differences
                     target="u_t", support=Support.from_reference(points, names=["x", "t"]))
report = distill(data, config=DistillationConfig(backends=("sindy", "gp"), poly_order=1, lhs="u_t",
                                                 units={"u": "K", "u_x": "K/m", "u_xx": "K/m^2"},
                                                 target_unit="K/s"))
print(report.equation(), report.terms, report.identifiability["stable"])

U = simulate_evolution(report.to_text("u"), x=x, t=t, u0=u0)     # the equation back in a solver
report = verify(report, [{"name": "beyond the window", "kind": "out_of_distribution", "features": features,
                          "target": network_values, "reference": exact,
                          "simulator": lambda r, c: simulate_evolution(r.to_text("u"), x=x, t=t, u0=u0).ravel()}])
```

| Stage | Call | Gives |
|---|---|---|
| sampling | `field_dataset` / `module_dataset` / `PolynomialSurface` | named feature columns and one target, read through `torch.autograd` through the model, refused outside `Support` |
| symbolic regression | `distill(data, config=...)` | a Pareto front of complexity against error per arm; `sindy`, `gp`, and `pysr` when installed |
| identifiability | `report.identifiability` | per-term inclusion, coefficient interval and stability over the bootstrap resamples and the ensemble members; unstable terms are dropped from the equation |
| verification | `verify(report, cases)` | the equation against the network that produced it, on held-out and out-of-distribution cases, by evaluation or by integration |
| the record | `report.equation()`, `.latex`, `.code`, `.to_dict()`, `.to_system()`, `.to_pinn()` | the law in four forms and the handoff to `pinneapple_physics.symbolic_pde` |

The SINDy arm reuses `pinneapple_analysis.inverse_problems.missing_term` (`CandidateLibrary`, `SINDyIdentifier`) and
the prune of `discovery`; `module_dataset` also reads a `NeuralTermDiscovery` (its `predict_term`) or a plain numpy
callable, so a learned term network is distilled by the same pipeline.

## Measured: a law read out of a model that implements it exactly

`f = -0.5 x + 2 y^2` in `torch`, 400 points uniform on [-1, 1]², library of every monomial up to degree 2 (six
candidates, four of them not in the law):

| Term | Coefficient | Interval95 (30 replicates) | Inclusion |
|---|---|---|---|
| `x` | -0.4999999999999999 | [-0.5000000000000004, -0.49999999999999956] | 1.00 |
| `y*y` | 2.0 | [1.999999999999999, 2.0000000000000018] | 1.00 |

`f = -0.5*(x) + 2*(y*y)`, held-out relative error **1.1e-15**, fit error 1.1e-16, both checks pass, 0.04 s.
The front of this sample is one point (`complexity 2`, relative error 1.1e-05): with nothing to trade there is
nothing to plot.

## Measured: the noise term the replicates drop

The same law with 12 % noise on the target, 600 points, 60 resamples. `x*x` is the candidate the noise lets into
part of the fits. The coefficients below are the ones the split fit (80 % of the rows) found, which is what the
replicates are refits of; the reported equation is refitted on all the rows, which is why its numbers differ in the
third decimal:

| Term | Coefficient (split fit) | Inclusion | Interval95 | In the equation |
|---|---|---|---|---|
| `x` | -0.5063 | 1.00 | [-0.518, -0.488] | yes |
| `y*y` | 2.0139 | 1.00 | [1.974, 2.036] | yes |
| `x*x` | -0.0317 | 0.72 | [-0.0668, -0.0275] | **no** |

`f = -0.506954*(x) + 1.99536*(y*y)`, held-out 12.7 %. The interval of `x*x` covers only the fits that **had** the
term: every one of them puts it at the same sign, so an interval that includes 0 would contradict the `same_sign`
its own row reports.

The front of the SINDy sweep, one point per complexity:

| Complexity | Held-out relative error | STRidge threshold (absolute) | Expression |
|---|---|---|---|
| 2 | 0.126632 | 0.2524 | `-0.496656*(x) + 2.00074*(y*y)` |
| 4 | 0.124985 | 0.1503 | `0.0076782*(1) - 0.496385*(x) - 0.0287783*(x*x) + 2.0045*(y*y)` |

A longer model that fits worse is dominated and is not on the front; the same model reached by two thresholds is one
point.

## Measured: a heat equation that outlives the surface it was read from

`u_t = D u_xx`, `D = 0.5`, on a periodic grid of 41 points with `u0 = sin x + 0.5 sin 2x`. The source model is a
Chebyshev `PolynomialSurface` fitted to the exact solution on `t` in [0, 0.4] only (1025 points, RMS residual
**3.7e-6**); the distillation never sees `t > 0.4`.

| Quantity | Value |
|---|---|
| Recovered law | `u_t = 0.499945*(u_xx)` |
| `D` against the true 0.5 | relative error **1.1e-4** |
| Unit the coefficient needs | `m^2/s` (from `target_unit="K/s"` and `u_xx` in `K/m^2`) |
| Held-out relative error against the model | 0.29 % (allowed 5 %) |
| Stable terms | `u_xx`; `u` and `u_x` never enter |
| Identifiability | inclusion 1.00, interval [0.49992, 0.49998], 30 replicates |

Integrated from the same initial condition over `t` in [0, 4], ten times the window it was read on, against the
analytic solution:

| Arm | Relative error over `t` in [0, 4] |
|---|---|
| the distilled equation, `simulate_evolution` | **6.8e-5** |
| the surface it was read from, evaluated at the same points | **10.57** (1057 %) |

`verify` writes the verdict from that comparison (`kind="out_of_distribution"`, the simulator replacing the pointwise
evaluation): `equation_wins`, with the check `distill.verification` passing. The equation was not fitted to any of
those points, and the boundary values are the solver's own, never borrowed from the data.

## Measured: the units filter, and what it refuses

`u_t = -1.5 u_x` with `u` dimensionless and `u_x` a rate, `units={"u": "1", "u_x": "1/s"}`, `target_unit="1/s"`,
`dimensionless_coefficients=True`: only a term of unit 1/s can be multiplied by a dimensionless coefficient, so the
library reaches the regression without `1` and `u`:

> 2 candidate terms never reached the regression: the units constraint refused them (1: the term is 1, the left-hand
> side is 1/s: the coefficient would not be dimensionless; u: the term is 1, the left-hand side is 1/s: the
> coefficient would not be dimensionless)

A term the expression language cannot read is refused with the reader's reason, never with a missing unit for a name
that is not a variable, and never with an exception:

> tan(u): the term 'tan(u)' is not one the expression language reads: only ['cos', 'exp', 'log', 'sin', 'sqrt',
> 'tanh'] may be called in a constraint: 'tan(u)'; u^2: the term 'u^2' is not one the expression language reads:
> BitXor is not part of the expression language

A function the language does have follows the dimension rule instead (argument dimensionless, result dimensionless):
`tanh(u)` with `u` in m/s is refused with *"tanh() takes a dimensionless argument; its argument is m/s"*, and `sin(u)`
likewise; `sqrt(u)` with `u` in m/s is refused because a fractional power of a unit has no whole-unit dimension.

## Handing the equation back

```python
report.to_text("u")                 # 'u_t = 0.499945*(u_xx)' -- the compiler's syntax, read by parse_system
system = report.to_system()         # EquationSystem(coords=['x', 't'], fields=['u']) for symbolic_pde
problem, model = report.to_pinn({"x": (0, 2 * np.pi), "t": (0, 1)}, field="u")   # as a known term
report.to_text("v")                 # ValueError: 'v' is not one of the equation's features
```

`to_dict()` / `from_dict()` round-trip the whole record, and every quantity that could not be measured is `None`
with its reason in `report.unavailable` (no interval invented from an ensemble that did not run).

## Limits (measured, not assumed)

- **A candidate that is collinear with another cannot be separated.** The same heat problem with a single harmonic
  (`u = exp(-Dt) sin x`, `D = 0.05`, 693 rows) makes `u_xx = -u` exactly. STRidge then puts the whole weight on one
  of the two columns and reports `u_t = -0.05*(u)` — a *wrong* law with a perfect fit (7.6e-13 relative on the
  held-out rows). The identifiability table refuses to bless it: `u_xx` has inclusion 1.00 but its coefficient
  flickers in sign around zero between resamples (`same_sign` false, interval [-8.5e-11, 1.7e-10]), so it is
  reported as unstable rather than as part of the equation. Two harmonics remove the degeneracy and the report is
  then `u_t = 0.499945*(u_xx)` above. The strong form cannot tell collinear terms apart; the fit alone will not say
  so, which is why the table exists.
- **The bootstrap measures sampling stability, not model error.** A bias shared by every resample survives it: read
  `verify`, which compares against a reference, not only `identifiability`.
- **The SINDy arm inherits the strong form's bias.** Derivatives come from `autograd` through the model, which is the
  point of distilling from a model rather than from a table; a noisy *model* still lets one term stand in for part of
  another (`discovery.discover_pde` has the weak form and names the disagreement).
- **The genetic-programming arm is a search, not a guarantee.** At `population=60, generations=6` on a two-term
  quadratic law it reported `-0.557*sin(sin(x))` (held-out 36 %); it is there for the laws SINDy's library does not
  span, and the arm with the lower held-out error wins.
- **PySR is an extra.** `pip install pysr` and one `python -c "import pysr"` for the Julia side; without it the arm is
  recorded in `report.unavailable` with the command, and the other arms still deliver.
- **A term whose effect is below the fit error of the network cannot be recovered at all.**

## References

Brunton, Proctor & Kutz (2016), *PNAS* 113(15) 3932-3937 (SINDy). Rudy, Brunton, Proctor & Kutz (2017), *Science
Advances* 3(4) e1602614 (PDE-FIND). Cranmer et al. (2020), NeurIPS 33, arXiv:2006.11287 (EQL: a network distilled
into a symbolic form). Fasel, Kutz, Brunton & Brunton (2022), *Proc. R. Soc. A* 478, 20210904 (Ensemble-SINDy — the
replicate/interval design). Angelopoulos & Bates (2022), arXiv:2107.07511 (conformal bands).

Reproduce every number above with `PYTHONPATH=. python -m pytest tests/test_symbolic_distillation.py -q`.
