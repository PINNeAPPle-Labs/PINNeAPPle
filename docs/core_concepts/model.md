# Model

A **Model** maps coordinates to predicted fields — `u = model(x)`. It knows
nothing about the PDE it is being used to solve; physics enters later, as a
loss computed *on top of* the model's output. This separation is what lets
you swap a SIREN for a Fourier neural operator without touching the physics
code, and swap the PDE without touching the model code.

## Where it lives

Architectures live in `pinneapple_neural.architectures` (re-exported at
`pinneapple_neural` and, for legacy imports, `pinneapple_models`). Every
model subclasses `BaseModel(nn.Module)`
(`pinneapple_neural/architectures/base.py`), which adds framework-wide
conventions on top of a normal `torch.nn.Module`:

- `forward(x)` — the actual coordinate → field mapping (subclass-defined).
- `forward_batch(batch)` — a default dict-based entry point used by the
  `Trainer`/Arena: it reads `batch["x"]` (or `batch["x_col"]` for PINN
  collocation batches) and calls `forward`.
- `save_checkpoint(path, metadata=None)` / a matching load path — writes a
  dict with `state_dict`, `class_name`, and user `metadata` so a checkpoint
  can be reloaded without knowing which class produced it ahead of time.
- `family` / `name` class attributes used by the model registry for
  discovery and reporting.

## Building a model

Rather than importing architecture classes directly, use the registry:

```python
from pinneapple_neural import build_model

model = build_model("SIREN", in_dim=2, out_dim=1, hidden_dim=64, n_layers=4)
```

`build_model` is a thin wrapper over `ModelRegistry.build`
(`pinneapple_neural.architectures.registry`), which looks the name up in the
`ModelCatalog` and instantiates it with the given kwargs — the same registry
the benchmark suite (`pinneapple_tools.benchmark_suite`) uses to build every
model in a comparison run from a plain string name.

## Available architecture families

`pinneapple_neural.architectures` ships several families, all subclassing
`BaseModel`: PINN-style MLPs (`SIREN`, `ModifiedMLP` with Fourier feature
embeddings), coordinate encodings (`HashGridMLP`), neural operators (`AFNO`
and others under `neural_operators`), graph networks (`MeshGraphNet`), plus
transformer, ROM, reservoir-computing, and classical time-series model
families under the same registry. Which family you pick affects only how
coordinates map to fields — everything downstream (loss compilation,
training, benchmarking) is architecture-agnostic.

## What a Model is *not*

A `Model` does not know about boundary conditions, PDE residuals, or
optimizers. Those responsibilities belong to
[ProblemDefinition](problem_definition.md), [PINN / Physics](pinn.md), and
[Solver](solver.md) respectively. The only contract a `Model` has to satisfy
is: given a batch of coordinates, return a batch of predicted field values.


## PhysicsModule: composing models and solvers

`pinneapple_core.module` (also `pp.PhysicsModule`, `pp.Sequential`, `pp.SolverModule`, `pp.Hybrid`) treats a neural
network, a numerical solver and a mix of the two as the same kind of component. Everything is a torch module.

| Class | Use |
|---|---|
| `PhysicsModule` | base class: subclass it and write `forward`; gives `describe()`, `num_parameters()`, a small `fit(x, y)` |
| `Sequential` | chain modules, solvers and plain functions; `(name, module)` pairs name them |
| `SolverModule(fn, params, trainable)` | a differentiable function as a component, e.g. a finite-element solve; autograd flows through it, and the names in `trainable` become parameters (calibration, identification) |
| `SolverModule.from_method(problem, "fem")` | any `pp.solve` method (FEM, external code, ...) as a component; not differentiable, so only what follows it trains |
| `Hybrid(solver, corrector, mode)` | `residual` (`u0 + c`), `multiplicative` (`u0 (1 + c)`) or `replace` (`c`); `corrector_input` chooses what the corrector sees; `return_parts=True` returns solver output and correction separately |

```python
model = Hybrid(SolverModule(coarse_fem_solve), corrector, corrector_input=lambda args, u0: torch.stack([u0, k(args[0]), x], 1))
model.fit(theta, fine_solution, epochs=600, lr=1e-2, batch_size=32)
print(model.describe())
```

`examples/physics_module/01_coarse_fem_plus_neural_correction.py` builds a coarse 6-cell finite-element solve of
`-(k(x) u')' = 1` with oscillating `k` plus a dilated convolutional corrector, trained end to end through the solver:
held-out relative error 0.31 for the coarse solve alone and 0.10 for the hybrid.
`Hybrid(..., detach_solver=True)` stops gradients into the solver. Model discrepancy decomposition and neural-closure
workflows are separate roadmap items (X27, X28).
