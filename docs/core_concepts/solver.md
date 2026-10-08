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


## `pp.compile`: solve the same problem with less work

```python
compiled = pp.compile("burgers_1d", optimize="physics")
sol = compiled.solve("pinn", epochs=2000)
print(compiled.benchmark(epochs=100))      # baseline vs compiled CPU time and how far the weights differ
```

`optimize="physics"` memoizes first derivatives within one loss evaluation. The PDE residual is built from several
derivative calls on the same field (the time derivative and the convective term of Burgers both differentiate `u`), and
each call builds an autograd graph; the compiled problem builds it once. The losses are the same, and the trained
weights agree to floating-point summation order. Measured with `benchmark` (80 epochs, 2048 collocation points,
`modified_mlp` 64x4, one CPU thread, best of 3 alternating runs):

| Problem | First-derivative graphs built | Baseline | Compiled | Speedup | Max weight difference |
|---|---|---|---|---|---|
| burgers_1d | 1 of 2 | 3.33 s | 2.84 s | 1.18x | 3e-8 |
| crystal_phonon | 1 of 3 | 3.26 s | 2.52 s | 1.29x | 6e-8 |
| heston_pde_2d | 4 of 6 | 7.49 s | 6.31 s | 1.19x | 5e-8 |
| sod_shock_tube_astro | 5 of 6 | 4.21 s | 3.57 s | 1.18x | 0 |
| threaded_coupling_tc50_rotating | 10 of 11 | 9.58 s | 9.11 s | 1.05x | 2e-8 |

The gain is modest and depends on how many derivatives the PDE repeats. Where the time goes: about half of a PINN
step is the backward pass through the second-order graph, which caching does not touch. Two other ideas were measured
on CPU and rejected: forward-mode (nested `jvp`) derivatives for the Burgers residual were slower (59 ms against 35 ms
per step), and a batched evaluation of the diagonal second derivatives changed the time by less than 10% (41.4 to
41.8 ms in 2D, 57.3 to 51.6 ms in 3D). Not done: kernel-level operator fusion, batching of the boundary and initial
forward passes, memory planning, and compiling a bare model (`pp.compile(model)` raises `TypeError`, since the
optimisations act on the PDE residual a model alone does not define). `solve_pde(..., cache_derivatives=True)` and
`compile_problem(spec, cache_derivatives=True)` expose the same switch.
