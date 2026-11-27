# NVIDIA Warp kernels for the black-hole accretion solver

Issue [#410](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/410). `pinneapple_physics.blackhole.hydro` is a
vectorised PyTorch finite-volume solver: on a 128 × 64 grid a step is a few hundred small tensor operations and takes
about 9 ms on one CPU core, which makes a run of the length the black-hole weather paper needs (5 × 10⁵ GM/c³) a
matter of many hours. `RIAFConfig(backend="warp")` runs the **same scheme** as [NVIDIA
Warp](https://github.com/NVIDIA/warp) kernels, so the state can live on a GPU and a step can be differentiated through
`wp.Tape`.

```bash
pip install warp-lang                 # or: pip install "pinneapple[warp]"
python -m pinneapple_physics.blackhole.hydro_warp     # the measurement harness, prints the table below
```

```python
from pinneapple_physics.blackhole import AccretionFlow, RIAFConfig, torus_state
from pinneapple_physics.blackhole.hydro_warp import warp_status, tape_accreted_mass

warp_status()      # {'available': ..., 'missing': ..., 'install': ..., 'device': 'cpu', 'version': ...}
flow = AccretionFlow(RIAFConfig(nr=128, ntheta=64, backend="warp"))
flow.set_primitives(torus_state(flow))
flow.run(t_end=100.0, every=10.0)

tape_accreted_mass(flow, steps=8, dt=0.05)   # {'mass': ..., 'dmass_dalpha': ...} from wp.Tape
```

Warp is optional. Without it this module imports, `warp_status()` says what is missing and how to install it, and only
a request for the backend raises (`ImportError`, with the install line) — the same contract as
[NVIDIA Warp as a compute backend](warp_backend.md), which is the general entry point (`warp.fem`, zero-copy tensors,
a differentiable rigid-body step).

## What was ported

The PyTorch solver stays **the reference**, and its arithmetic is not changed by any of this: `hydro` computes the
MUSCL-minmod reconstruction, the HLL fluxes, the two azimuthal viscous stresses, the geometric sources, the safety
caps and the ghost cells, and the kernels repeat those expressions in the same order.

| piece | reference | Warp |
|---|---|---|
| one SSP-RK2 stage (`rhs` + `U -> U + dt k`, per cell) | `AccretionFlow.rhs`, `step` | `stage1` / `stage2`: reconstruction, HLL flux, viscous stresses, divergence, sources and the update **in one kernel per stage**, one thread per cell |
| boundary fluxes (`budget`) | the `fluxes` dict of `rhs` | `budget`, one thread per theta face, for the two radial faces |
| `cons_to_prim_i` + `_bc` | two tensor passes | `convert`: one kernel over the whole grid, interior from the conserved state, ghosts from the mirrored cell |
| the time step | `dt` (CFL + explicit diffusion limit) | the same PyTorch reduction, unchanged: both backends pick bit-identical steps |

Each cell reconstructs its four faces, so a face shared by two cells is computed twice. That is deliberate: it is what
makes the stage one launch instead of three, and on a GPU the launch count is what a step of this size is made of.

Two things the port had to get right, both measurable and both in the code comments:

* **Ghost cells are written from the conserved state, never by reading the primitive buffer.** The first version read
  the neighbouring cell's new value, which the same launch was writing: a race, 3 × 10⁻² wrong on `v_φ`.
* **A buffer that a later kernel overwrites cannot be on the tape.** Warp evaluates a kernel's derivative at the
  values its inputs hold when `Tape.backward` runs, so the tape path allocates fresh state and stage buffers every
  step. With the reused buffers the gradient of the angular momentum was 18 % wrong after a single step.

## Measured: CPU

Apple Silicon (arm64), macOS, `torch` 2.15.0 with one thread, `warp-lang` 1.19.0 on its CPU backend, 20 steps after 3
warm-up steps. "Max rel. difference" is the largest relative difference of the five primitive fields against the
PyTorch run on the same grid at the same time; the boundary budgets agree to 2 × 10⁻¹⁵.

| backend | grid | ms/step | simulated time | s per unit time | vs reference | max rel. difference |
|---|---|---|---|---|---|---|
| torch (reference) | 64 × 32 | 4.14 | 5.0 | 0.017 | 1.00× | 0 |
| warp (CPU) | 64 × 32 | 4.09 | 5.0 | 0.017 | 1.01× | 3.4e-14 |
| torch (reference) | 128 × 64 | 8.82 | 3.0 | 0.058 | 1.00× | 0 |
| warp (CPU) | 128 × 64 | 6.08 | 3.0 | 0.040 | 1.45× | 3.4e-14 |
| torch (reference) | 256 × 128 | 32.97 | 1.9 | 0.35 | 1.00× | 0 |
| warp (CPU) | 256 × 128 | 23.87 | 1.9 | 0.25 | 1.38× | 2.4e-14 |

Read that honestly: **on a CPU the kernels are 1.0–1.45× faster, not the order of magnitude the issue is after.**
Warp's CPU backend is a compiled one-shot kernel per launch, and the fused stage does twice the face work per cell;
the PyTorch reference is already inside CPU BLAS-friendly elementwise territory. Warp's case is the GPU, and there is
no GPU on this machine (see the next section). The `numba` backend (`RIAFConfig(backend="numba")`) remains the fast
CPU path when `numba` is installed; it was not installed here, so it has no row.

Cost per cell above 128 × 64 settles at about 0.73 µs for the kernels and 1.01 µs for the reference — the per-cell
cost is what a larger grid costs, and the number a run of the paper's size is planned with.

## Not measured: GPU

No CUDA device, and `warp-lang` on this machine is a build without CUDA, so the GPU columns of the harness are
skipped, in the table, with that reason:

```
| warp | cuda | 128 x 64 | - | - | - | - | - | not measured: Warp runs on cpu here (no CUDA build of warp-lang) |
| torch | cuda | 128 x 64 | - | - | - | - | - | not measured: this machine has no CUDA device for PyTorch |
```

The kernels are written device-agnostically (the state is a zero-copy view of the torch tensor, and a flow whose
`device` is `cuda` puts the same kernels on the GPU), so `bench(device="cuda")` is the measurement, not a rewrite.
What has **not** been run is everything that only a GPU can show: the speed-up, float64 throughput (a consumer card
computes float64 at roughly 1/32 of float32, which is why a float32 kernel variant is the obvious next step and why
it is not here — the tests require agreement with the reference to round-off, which float32 cannot give), and the
behaviour of the tape on a longer run than a gradient check.

## Differentiability

`hydro.accreted_mass` runs the reference's two stages out of place (a fresh clone per update, `bc_functional` for the
ghost cells, a *fixed* `dt`) and returns the mass through `r_in` as a tensor that keeps the graph, so
`total.backward()` gives d(mass)/d(α). Writing the state in place, as `step` does, is what stops autograd there: the
clamp in the boundary condition was overwritten a line after torch saved it, and a step that overwrites its own
operands cannot be back-propagated.

`dt` is fixed on purpose: the reference picks it from the state, a step that depends on α would have to be
differentiated too, and the number of steps to a physical time is not differentiable at all.

Both paths are checked against a central finite difference of the same observable (24 × 12, 8 steps, h = 10⁻⁴):

| path | d(mass)/dα | against the finite difference |
|---|---|---|
| finite difference (reference, two runs) | 6.051962e-11 | — |
| torch autograd through `hydro.accreted_mass` | 6.051951e-11 | 1.9e-6 relative |
| `wp.Tape` through the Warp kernels | 6.051951e-11 | 1.9e-6 relative, and 1e-15 against the torch autograd value |

The same tape is the route to the rest of the issue: gradients with respect to the torus parameters. Those enter
through `torus_state`, which is numpy, so they are **not** on the tape yet — the piece to write is a Warp (or torch)
version of the initial condition, after which the same one-line `tape.backward` gives d(mass)/d(R_c, a, l_c).

## The other two solvers

`compressible_fv` and `shallow_water_fv` are named in the issue and are **not** ported:
`hydro_warp.WARP_PORTS` says so, with the reason and the reference class, and `hydro_warp.not_ported("shallow_water_fv")`
raises `NotImplementedError` naming the seam instead of pretending otherwise. The seam is the shape of `WarpStepper`: a
stepper with `dt()`/`step(dt)` that keeps the reference's buffers, one kernel holding the reference's `rhs` arithmetic,
and an equivalence test in the shape of `tests/test_hydro_warp.py`. For the shallow-water solver the extra work is its
wall and gate masking and its wet/dry wave speeds, which are value-dependent branches that need tests of their own.

## Tests

`tests/test_hydro_warp.py`, 16 tests: the dispatch (including "asked for and missing", by hiding `warp` from the
import system rather than deleting the package), agreement with the reference to round-off over four set-ups and a
longer run on a larger grid, the budget, the boundary rule (`_bc` against `bc_functional`), the gradient check on the
reference — which runs everywhere — and the same check through the tape, the pinned reference numbers, the seam, and
the harness. Without `warp-lang` the five kernel tests skip with the install line as their reason; they never pass
quietly.
