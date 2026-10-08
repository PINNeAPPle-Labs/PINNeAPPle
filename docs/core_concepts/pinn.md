# PINN / Physics

This layer turns a [ProblemDefinition](problem_definition.md) and a
[Model](model.md) into a differentiable loss: it takes the model's
predictions at collocation points, differentiates them with autograd, and
assembles PDE-residual, boundary, and initial-condition penalties into a
single training objective.

## Where it lives

`pinneapple_physics.pinn_solver` (re-exported at `pinneapple_physics`):

- `compile_problem(spec, weights=None)` — the main entry point. Given a
  `ProblemSpec`, returns a `loss_fn(model, y_hat, batch)` callable that the
  `Trainer` calls every step. It reads `spec.pde.kind` to know which residual
  to build (`laplace`, `poisson`, `burgers`,
  `navier_stokes_incompressible`, `heat`, `wave`, `elasticity`, `darcy`,
  `helmholtz`, `advection`, `reaction_diffusion`, ...).
- `LossWeights` — the four fixed weights `compile_problem` reads directly:
  `w_pde`, `w_bc`, `w_ic`, `w_data`.
- `AdaptiveWeights` — a standalone self-normalizing weighter for arbitrary
  named loss terms (independent of `LossWeights`): it tracks a lagged EMA of
  each term's raw loss and rescales every term's weight relative to the
  currently *hardest* term, clamped to `[1, max_ratio]`.
- Autograd operators (`pinneapple_physics.pinn_solver.compiler.autograd_ops`):
  `grad(y, x)`, `jacobian(Y, x)`, `divergence(...)`, `laplacian(...)`,
  `time_derivative(y, x, t_index)`, `norm_dot_grad(y, x, normals)`, `mse`.
  These are the primitives every built-in residual (and any custom one) is
  built from.
- `Subdomain`, `SubdomainPINN`, `DoMINO` — domain-decomposition PINN
  (split the domain into subdomains, each with its own sub-network, stitched
  by interface conditions), for problems too stiff or too large for a single
  network.
- `LatentConditionedModel`, `sample_latent`, `ensemble_forward`,
  `mean_covariance_loss` — a stochastic/latent-conditioned PINN variant for
  predicting a distribution over solutions rather than a point estimate.

## Compiling and using a loss

```python
from pinneapple_physics import compile_physics  # wraps pinn_solver.compile_problem

losses = compile_physics(spec)          # spec: a ProblemSpec
# losses(model, y_hat, batch) -> {"pde": ..., "bc_<name>": ..., ...}
```

## Symbolic residuals

For PDEs not covered by a built-in `kind`, `pinneapple_physics.symbolic_pde`
compiles a residual straight from a SymPy expression:

```python
import sympy as sp
from pinneapple_physics.symbolic_pde import SymbolicPDE, HardBC

x, y = sp.symbols("x y")
u = sp.Function("u")
expr = u(x, y).diff(x, 2) + u(x, y).diff(y, 2) + 2*sp.pi**2*sp.sin(sp.pi*x)*sp.sin(sp.pi*y)
pde = SymbolicPDE(expr, coord_syms=[x, y], field_syms=[u])
residual_fn = pde.to_residual_fn(model)   # (N,1) PDE residual tensor
```

`HardBC` bakes a boundary condition into the network's output via a distance-
function ansatz (`wrap_model`) instead of penalizing it in the loss;
`PeriodicBC`/`MultiPeriodicBC`/`NeumannBC` cover other constraint styles.

## What this layer does *not* do

It does not train anything — it only produces loss values/tensors from a
model's predictions. Turning that loss into weight updates is the job of
[Solver](solver.md).


## `pp.loss`: loss terms and balancing by name

`pinneapple_core.loss` (`pp.loss`) has the loss terms as plain functions and a `Balancer` that selects the weighting
strategy by name.

| Term | Meaning |
|---|---|
| `pde(residual)` | mean squared residual; `weights=` for a weighted mean, `t=` with `causal_epsilon=` for causal training |
| `boundary(pred, target)` | boundary or initial condition mismatch |
| `conservation(quantity, target, scale)` | conserved quantity or divergence-free constraint |
| `energy(e, mode)` | `conserve`, `nonincrease`, `nondecrease` along time |
| `symmetry(model, x, transform, output_transform)` | invariance or equivariance penalty |
| `supervised(pred, target, kind)` | `mse`, `mae`, `huber`, `relative_l2` |
| `inverse(pred, observed, params, prior, reg)` | misfit plus a prior on the unknown parameters |
| `combine(terms, weights)` | weighted sum; weights for absent terms raise instead of being dropped |

```python
from pinneapple_core import loss
bal = loss.Balancer("gradnorm", names=["pde", "bc"], model=net)       # or "ntk", "relobralo", "curriculum", ...
total = bal({"pde": loss.pde(residual), "bc": loss.boundary(u_b, g)}, step=i)
total.backward()
```

`Balancer.strategies()` lists: `fixed`, `self_adaptive`, `gradnorm`, `loss_ratio`, `ntk`, `relobralo`, `softadapt`,
`augmented_lagrangian`, `inverse_dirichlet`, `lr_annealing`, `pcgrad`, `joint_adaptive`, `auto` (these are the existing
`WeightScheduler` methods behind one name) and `curriculum` (`schedule={"bc": (0, 10)}` ramped over `steps`, linear or
cosine). Strategies that read gradients need `model=`. `self_adaptive` exposes `weight_parameters()` to optimise by
gradient ascent. `examples/vs_physicsnemo/03_pinneapple_active_weight_sched/example.py` now runs on `pp.loss`: with the
same seed its loss and error histories are identical (difference 0.0) to the version with hand-written weights.
