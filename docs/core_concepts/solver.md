# Solver

In this framework's pipeline (Problem → Domain → Model → Physics → **Solver**
→ Backend → Researcher), "Solver" means the *training policy*: which
optimizer, how many stages, in what order, and with what scheduling —
distinct from `pinneapple_simulation.numerical_solvers`, which are classical
FEM/FDM/CFD solvers used to generate reference data (see
[Package Layers](../architecture/package_layers.md) for that distinction).

## Where it lives

The training-policy layer is `pinneapple_neural.trainer` (re-exported at
`pinneapple_neural`, and via the legacy alias `pinneapple_train`):

- `Trainer` + `TrainConfig` — the baseline policy: an Adam optimizer
  (`torch.optim.Adam`, configured by `TrainConfig.lr`/`weight_decay`), with
  optional gradient clipping, AMP, early stopping, checkpointing, and a
  `physics_aware_validation` flag that runs validation under
  `torch.enable_grad()` so PDE-residual validation losses (which need
  second derivatives) don't break inside a `no_grad` block.
- `TwoPhaseTrainer` + `TwoPhaseConfig` — a two-stage policy: Phase 1 fits the
  model to reference/measured data (supervised MSE), Phase 2 switches to the
  physics (PDE residual + BC) loss; `combined=True` instead ramps a physics
  weight in during Phase 1 rather than switching abruptly.
- `TimeMarchingTrainer` — splits `[0, T]` into sequential windows and trains
  a fresh copy of the network per window, using the previous window's
  prediction at its end time as the next window's initial condition (Wight &
  Zhao 2020) — for stiff or long-horizon temporal PDEs.
- `CausalPINNTrainer` + `CausalWeightScheduler` — weights collocation points
  by a decaying function of the cumulative residual at earlier times, so the
  network is pushed to get early times right before late ones (Wang et al.
  2022, "Respecting causality...").
- `DDPPINNTrainer` — the same training loop wrapped for
  `torch.nn.parallel.DistributedDataParallel`.
- `MultiRestartTrainer` — runs multiple random restarts and keeps the best.
- Loss-weight balancers, usable independently of which `Trainer` you pick:
  `WeightScheduler`/`SelfAdaptiveWeights`, `GradNormBalancer`,
  `LossRatioBalancer`, `NTKWeightBalancer`, `ReLoBRaLo`, `SoftAdapt`,
  `AugmentedLagrangian`, `InverseDirichlet`, `PCGrad`, `JointAdaptiveWeights`,
  `AutoBalancer`.

## Using it

```python
from pinneapple_neural import train_model  # Trainer shortcut

result = train_model(model, losses, epochs=5000)
```

`train_model` is a thin wrapper over `Trainer(model, loss_fn).fit(...)`; for
anything beyond the default single-stage Adam policy (two-phase, causal,
time-marching, distributed), construct the specific trainer class directly.

## What Solver does *not* do

It has no opinion on which physics or architecture it's optimizing — it
receives a model and a loss function and drives weight updates. It also
doesn't decide *where* those updates run (CPU/GPU, PyTorch/JAX); that's
[Backend](backend.md).


## One `solve(problem)` for every kind of solver

`pp.solve(problem, method)` takes a `PhysicalProblem`, a `ProblemSpec` or a preset name and returns a `Solution` with
`predict(X)`, `wall_time_s` and `metadata()` (the same keys for every backend: problem fingerprint, method, kind,
fields, coordinates, and the backend's own details). `pp.compare` scores several methods against a reference at the
same points and reports each method's kind.

| Method | Kind | What it is |
|---|---|---|
| `pinn` | neural | physics-informed network trained with `solve_pde` |
| `fem` | classical | P1 finite elements for steady `poisson` / `laplace` on a box with Dirichlet conditions (1D-3D) |
| `reference` | classical | the solver named in the problem's `reference_solver` |
| `external` | external | any code returning grid fields, `runner=callable(problem)` |
| `analytic`, `exact` | analytic | closed form known to the library, or a function you pass |

Register more with `@register_method("name", kind="classical")`; `list_methods(kind=...)` and `list_kinds()` show what is
available. A method that cannot handle a problem raises `MethodNotAvailable` with the reason, never a partial answer.
The source term of `poisson` is passed as `ctx={"source_fn": fn}` and read by both `pinn` and `fem`, so the two solve the
same problem. Example: `examples/solver_api/01_classical_vs_pinn.py` (FEM relative L2 4.2e-3 in 0.02 s, PINN 1.3e-3 in
30 s on the manufactured Poisson problem).

Not yet behind this contract: FNO, SPH, LBM and FVM backends, and time-dependent classical solves of arbitrary presets.
