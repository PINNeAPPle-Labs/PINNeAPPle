# Integral equations of the second kind

The problem compiler carries two kinds that are not differential equations at all: `fredholm2` and `volterra2`, the
Fredholm and Volterra equations of the second kind on an interval.

```
Fredholm   u(x) = f(x) + lambda * int_a^b K(x, s) u(s) ds
Volterra   u(x) = f(x) + lambda * int_a^x K(x, s) u(s) ds
```

They are solved in Nyström form: the integral is replaced by a fixed quadrature on the collocation points themselves,
so the residual the training loop differentiates is an algebraic equation in the values the network already produces.

```
r(x_i) = u(x_i) - f(x_i) - lambda * [ sum_j w_j K(x_i, x_j) u(x_j) + D_i u(x_i) ]
```

`w` are the composite-trapezoid weights on the nodes sorted by `x`, and `D_i` is the analytic integral of the singular
part of the kernel over the self cell of node `i`. There is no extra model evaluation and no autograd through the
kernel: this is the cheapest residual in the compiler, and it is the only one that couples every point to every other.

## Using it

```python
import numpy as np
from pinneapple_physics.pde_environment.spec import PDETermSpec, ProblemSpec
from pinneapple_physics.pinn_solver.compiler.integral_kinds import constant_kernel

spec = ProblemSpec(
    name="fredholm2", dim=1, coords=("x",), fields=("u",),
    pde=PDETermSpec(kind="fredholm2", fields=("u",), coords=("x",), params={"lambda": 0.5}),
    conditions=(), domain_bounds={"x": (0.0, 1.0)},
)
# the free term and the kernel travel in the batch's ctx, like every other source term
batch = {"x_col": grid, "ctx": {"kernel_fn": constant_kernel, "f_fn": my_free_term}}
```

`params` takes `lambda` (alias `lam`, default 1) and `singular`. `ctx` takes `kernel_fn`, an optional `diag_fn` and
`f_fn` (alias `source_fn`, default zero).

`kernel_fn(X, S, ctx)` receives the nodes as two arrays of shape `(N, 1)` and `(M, 1)` and returns an `(N, M)` matrix.
Writing one is a one-liner for a pointwise kernel:

```python
def gaussian_kernel(X, S, ctx):
    r = np.linalg.norm(X[:, None, :] - S[None, :, :], axis=-1)
    return np.exp(-(r / ctx["ell"]) ** 2)
```

## Singular kernels

For `K = ln|x - s|` the diagonal is infinite, so a kernel that returns it pointwise cannot be integrated. The kernel is
split: `kernel_fn` returns the part that is summed numerically (finite everywhere, zero on the diagonal is the usual
choice) and `diag_fn(width, ctx)` returns the analytic self-cell integral of the singular part. Three helpers come
with the module, as matched pairs:

| kernel | `kernel_fn` | `diag_fn` | order |
| --- | --- | --- | --- |
| `K = 1` | `constant_kernel` | not needed | 2 |
| `K = ln|x-s|` | `log_kernel` | `log_diag` | 1 |
| `K = (j/4) H_0^(2)(k r)` | `helmholtz2_kernel` | `helmholtz2_diag` | 1 |

Set `"singular": True` in the params and the kind **refuses to run** without a `diag_fn`. That is deliberate: a
singular kernel integrated straight through its diagonal produces a plausible wrong answer, and a silent wrong answer
is worse than a refusal. Dropping the analytic diagonal costs more than an order of magnitude at the same resolution
— measured at 25 times at 21 nodes for the logarithmic kernel.

`helmholtz2_kernel` is complex, so the equation needs two fields (the `ur`/`ui` or `Er`/`Ei` convention of the
compiler) and returns two residuals. Its diagonal uses the small-argument expansion of the Hankel function and raises
when `k * cell >= 1`, where the expansion stops being valid.

## The orders are measured, not assumed

The trapezoid with an analytic self cell is second order for a bounded kernel and **first order** for a logarithmic
one. The reason is not the diagonal: it is the two cells next to the singularity, where the logarithm is not smooth
and the trapezoid carries an `O(h)` truncation error. Getting second order there needs product integration over the
near cells, which is a documented next step and not something the current rule does.

Volterra is second order too, and the causal mask is not a detail. Solving the same Volterra equation with the future
allowed to contribute -- the Fredholm matrix, which already has a zero diagonal, keeping the half self cell -- gives a
"solution" of 160 where the exact one is bounded by 2.72. A mask error is not a small error, which is why
`tests/test_integral_kinds.py` asserts the difference rather than trusting the assembly.

```
n=  21  bounded residual=3.580e-04   log residual (analytic diagonal)=9.383e-03   log without=2.343e-01
n=  41  bounded residual=8.949e-05   log residual (analytic diagonal)=4.265e-03   log without=1.361e-01
n=  81  bounded residual=2.237e-05   log residual (analytic diagonal)=1.996e-03   log without=7.715e-02
n= 161  bounded residual=5.593e-06   log residual (analytic diagonal)=9.569e-04   log without=4.304e-02
```

`python examples/numerical_solvers/14_integral_equations_nystrom.py` reproduces the table, with the Volterra case and
the eigenvalue discussion below.

## The direct solve is the baseline

`nystrom_matrix(x, kernel_fn, ctx, diag_fn=..., volterra=...)` returns `(A, D)` with the weights folded in, so the
classical Nyström solve of the same discretisation is two lines:

```python
A, D = nystrom_matrix(x, constant_kernel)
u = np.linalg.solve(np.eye(x.size) - lam * (A + np.diag(D)), f)
```

This matters for honesty. A second-kind linear equation on a modest grid is cheap and accurate, and the direct solve
beats a trained network on the single solve every time. The neural arm earns its place where the direct solve stops:
a parametric family (one model instead of a family of matrices), an inverse problem (recover the kernel from data),
and high frequency, where the matrix becomes large and dense.

What the direct path actually reaches, measured on the bounded Fredholm case with `lambda = 0.5` (relative error
against `exp(x)`, second order throughout):

| nodes | 161 | 321 | 641 | 1281 | 2561 |
| --- | --- | --- | --- | --- | --- |
| maximum relative error | 5.6e-6 | 1.4e-6 | 3.5e-7 | 8.7e-8 | 2.2e-8 |

So `1e-6` needs about six hundred nodes, not one hundred and sixty: at second order, every factor of four in the
error costs a factor of two in the spacing. Quoting a target error without the grid that reaches it is how a
discretisation gets mistaken for a model.

## The trained arm, measured

Sections 5 and 6 of the example train a small tanh network on the same 81 nodes and the same discretisation, with
2500 Adam steps and a deterministic seed, and compare it with the direct solve on the same case. The loss is the one
`compile_problem` produces — the trained run is the check that the compiled kind is differentiable and trainable, not
a second implementation of the equation written beside it.

| arm | max error vs `exp(x)` | cost |
| --- | --- | --- |
| trained network, `lambda = 0.5` | 1.7e-3 | seconds of CPU training |
| direct Nyström solve, `lambda = 0.5` | 2.2e-5 | milliseconds |

So the direct solve is roughly eighty times more accurate and answers in the time the training spends on a few steps.
That is the expected result and the example states it rather than hiding it: for one linear solve at a modest grid,
this discretisation is the wrong place to put a network.

The parametric arm is the case where the trade is real. With a fixed free term `f = exp(x)` and the parameter moving
the solution, `u(x) = exp(x) + lambda (e - 1) / (1 - lambda)`, one network trained over `lambda` in `[0.2, 0.8]`
reaches a worst relative error of 1.6e-2 on held-out values of `lambda`. The direct solve reaches 1e-6 there, but it
has to be run once per parameter — and the network is one artefact for the whole band.

The clock does not flatter the model, and the example says so: a hundred direct solves spread over the same band take
tens of milliseconds and are exact, while the training takes tens of seconds. On a problem this small the model
is not a speed story. What it adds is a single differentiable object, continuous in the parameter, which is what a
gradient-based design loop or an inverse problem consumes — and that, not the wall-clock, is the claim to make.

Treat these numbers as a floor, not a benchmark: a better architecture, more steps or a different sampling of the
parameter would move them, and none of that changes the conclusion for the single solve. `lambda` stops at 0.8
because `lambda = 1` is the eigenvalue, where the equation has no unique solution for the direct solve to be compared
against. Note what the trained arm's own residual is — 3.6e-4 at 81 nodes, which is the quadrature error of the rule
itself. Training had reached the floor of the discretisation: no amount of further optimisation improves it without
refining the grid or the rule.

## The eigenvalue, and why a small residual is not a small error

For `K = 1` on `[0, 1]` the operator has a single eigenvalue at `lambda = 1`. There the system is numerically
singular — the condition number at 81 nodes is above `1e15` — while the exact solution still has a vanishing residual,
because the residual measures the quadrature error and not the conditioning. The Fredholm alternative is why: at the
eigenvalue the equation is solvable only for a compatible `f`, and the solution is not unique. Any experiment that
reports only the residual will look successful at `lambda = 1` and mean nothing. Report the two separately.

## Presets

Four presets register the kind at the `pp` level, each with a closed-form solution, so `pp.solve` and any comparison
or benchmark path can score a method against the exact answer without assembling anything by hand:

| preset | equation | kernel | exact solution |
| --- | --- | --- | --- |
| `integral_fredholm_unit_exp` | Fredholm, `lambda = 0.5` | `K = 1` | `exp(x)` |
| `integral_fredholm_separable_exp` | Fredholm, `lambda = 0.5` | `K = x s` (rank one) | `exp(x)` |
| `integral_volterra_unit_exp` | Volterra | `K = 1` | `exp(x)` |
| `integral_log_kernel_linear` | Fredholm, `lambda = 0.5` | `K = ln|x - s|` | `a + b x` |

```python
import pinneapple as pp

problem = pp.PhysicalProblem.from_preset("integral_log_kernel_linear")
exact = pp.solve(problem, "analytic")          # or pp.solve("integral_fredholm_unit_exp", "analytic")
```

The kernel and the free term travel in `pde.params` as callables — the same channel every other kind uses for material
maps and sources — so nothing has to be injected into the batch. The closed forms live in
`pinneapple_physics/closed_form/integral_equations.py` as matched `(solution, source)` pairs, and
`tests/test_integral_presets.py` checks each pair against a numerical integral of the kernel weighted by the claimed
solution. That test exists because the two files can drift apart, and a preset whose source no longer matches its
solution is a benchmark that measures nothing.

`lambda` defaults to 0.5 in every Fredholm preset: at `lambda = 1` the unit kernel is at its eigenvalue and the
comparison is meaningless. The log-kernel preset sets `singular=True` and ships `log_diag`, so it is the one that
exercises the analytic self cell through the public path.

## Scope and limits

- One spatial coordinate per kind. A curve in the plane needs an arclength parametrisation and the tensor-product
  variant is not implemented; both are follow-ups.
- Dense `(N, N)` assembly: adequate to a few thousand nodes, beyond which a fast method is the right tool and is not
  here.
- Second kind only. A first-kind equation is ill-posed without a regularisation, and a hypersingular kernel needs a
  regularisation or a weak-form treatment; the module refuses both rather than approximating them.
- The collocation points **are** the Nyström nodes. Hand the compiler a fixed grid for a deterministic quadrature: a
  resampled collocation set turns the quadrature error into noise, because the nodes, the weights and the diagonal
  move with the batch.

## Catalog

`E72` (Fredholm) and `E73` (Volterra) in `pinneapple_catalog.methods`, both marked `validated` against the closed-form
cases in `tests/test_integral_kinds.py`.
