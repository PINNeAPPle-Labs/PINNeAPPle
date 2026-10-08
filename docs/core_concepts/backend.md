# Backend

The Backend layer is the execution runtime abstraction: it decides which
numerical library actually runs the tensor operations, and provides the
lower-level device/parallelism utilities the [Solver](solver.md) layer
builds on. Everything above this layer (models, physics, training policy)
is written against a common interface so it does not need to know whether
it is ultimately running on PyTorch or JAX.

## Where it lives

`pinneapple_tools.compute_backends` (re-exported at `pinneapple_tools`, and
via the legacy alias `pinneapple_backend`):

- `Backend` — a namespace of backend-name constants (`Backend.TORCH`,
  `Backend.JAX`).
- `set_backend(name)` / `get_backend()` — a simple global backend selector
  (`"torch"` is the default and requires no extra dependency; `"jax"`
  requires `pip install jax jaxlib`).
- `JAXBackend` — `jit_pinn(model_fn, residual_fn)` JIT-compiles a PINN
  residual; `vmap_residual(single_pt_residual)` vectorizes a per-point
  residual over a batch; `torch_to_jax`/`jax_to_torch` convert tensors
  between the two libraries. `jax_available()` reports whether JAX is
  importable.

```python
from pinneapple_tools.compute_backends import set_backend, get_backend, JAXBackend

set_backend("jax")
print(get_backend())  # "jax"

compiled = JAXBackend.jit_pinn(model_fn, residual_fn)
batched = JAXBackend.vmap_residual(single_pt_residual)
```

PyTorch is the default and by far the more complete path — every model,
trainer, and benchmark task in the framework is written in PyTorch first;
the JAX backend targets a narrower, PINN-residual-specific speedup.

## Device and parallelism utilities

A related but separate set of helpers lives in `pinneapple_neural.trainer`
and is used by `Trainer`/`TwoPhaseTrainer`/etc. to actually place work on
hardware: `best_device()`, `count_gpus()`, `gpu_info()`, `AMPContext`
(mixed precision), `wrap_data_parallel`/`unwrap_model`, `CUDAPrefetcher`,
`GradAccumTrainer` (gradient accumulation), and HPC-scale utilities in
`trainer.hpc` — `FSDPConfig`/`wrap_fsdp`, `wrap_zero_optimizer`,
`CUDAGraphModule`, PowerSGD/Top-K gradient-compression hooks,
`build_torchrun_cmd`/`build_slurm_script`, and `PINNeAPPleProfiler`. These
are PyTorch-specific execution concerns (single-GPU vs. multi-GPU vs.
cluster), one layer below `compute_backends`' torch-vs-JAX abstraction.

## What Backend does *not* do

It has no notion of models, losses, or training loops — only of how tensor
operations for a given backend name are dispatched and accelerated. Which
loss to compute is [PINN / Physics](pinn.md)'s job; when to step the
optimizer is [Solver](solver.md)'s.


## PhysicsBackend: one array, derivative and solve API

`pinneapple_core.backend` defines `PhysicsBackend`, the interface the library's kernel code is written against, with
torch and jax implementations. `pp.set_backend("jax")` (or `with pp.use_backend("jax"):`) selects the engine and
`pp.get_physics_backend()` returns its object.

| Group | Operations |
|---|---|
| arrays | `asarray`, `to_numpy`, `linspace`, `zeros`, `ones`, `eye`, `diag`; elementwise math on `bk.xp` |
| algebra | `matmul`, `solve`, `integrate` (trapezoid) |
| transforms | `grad`, `jacobian`, `vmap`, `jit` (no-op on torch) |

```python
def objective(bk, L):                       # no engine named here
    x = bk.linspace(0.0, L, 101)
    return bk.integrate(bk.xp.sin(x) ** 2, x)

for name in ("torch", "jax"):
    bk = pp.get_physics_backend(name)
    print(name, bk.grad(lambda L: objective(bk, L))(bk.asarray(2.0)))
```

Float64 is the default dtype on both, so the same code gives the same numbers: `tests/pinneapple_core/test_backend.py`
solves a finite-difference Poisson problem and differentiates its objective with respect to the domain length on both
engines (values agree to 1e-10 relative, gradients to 1e-8). Add an engine (CUDA kernels, an FEM or FVM code, an external
solver) with `register_backend("name", MyBackend)`, where `MyBackend` subclasses `PhysicsBackend` and implements the
operations it supports; the rest raise `NotImplementedError`. `jax` is optional and enabled with x64 when first used.
Existing PINN code paths are not yet ported to this interface: today it is the contract new kernel code uses.
