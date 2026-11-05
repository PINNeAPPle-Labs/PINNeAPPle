# NVIDIA Warp as a compute backend

`pinneapple_tools.compute_backends` reaches a third engine, next to PyTorch (the default) and JAX:
[Warp](https://github.com/NVIDIA/warp) (Apache-2.0). Warp compiles `@wp.kernel` functions to CUDA or to the CPU and
carries a reverse-mode tape (`wp.Tape`) through them. PINNeAPPle already used Warp *indirectly*, through the Newton
rigid-body backend ([Rigid bodies with NVIDIA Newton](newton_rigid_body.md)); this module is the direct entry point,
for the three things a PinT or JAX engine does not give you.

```bash
pip install warp-lang          # or pip install "pinneapple[warp]"
pp.doctor("gpu")               # what is missing for Newton, Warp and PICT on this machine
```

## Zero-copy tensors

```python
import torch
from pinneapple_tools.compute_backends import to_warp, to_torch

x = torch.arange(12, dtype=torch.float64).reshape(4, 3).contiguous()
a = to_warp(x)                       # a view of the torch buffer (DLPack), not a copy
back = to_torch(a)
assert back.data_ptr() == x.data_ptr()
back[0, 0] = 7.0                     # the write lands in x
```

`to_warp` refuses a non-contiguous tensor rather than copying it: a copy is a decision, so it is the caller's to make.

## `warp.fem` weak forms, with the error measured

```python
from pinneapple_tools.compute_backends import fem_poisson, poisson_convergence

sol = fem_poisson(32)                          # -laplace(u) = f, u = 0 on all sides, Q2 on a quad grid
sol.n_dofs, sol.l2_error, sol.cg_iterations, sol.max_residual

for row in poisson_convergence((8, 16, 32, 64), degree=2):
    print(row["resolution"], row["l2_error"], row["order"])
```

The problem is a manufactured one with an analytic Laplacian, `u = x³(1-x)³ y³(1-y)³`, chosen because it is *not*
an eigenfunction of the discrete operator (a `sin(kπx) sin(lπy)` mode is, and it converges in two CG iterations,
hiding an assembly error). Measured on the CPU build, 8 → 64 cells per side:

| Shape functions | L2 error, 8 → 64 | Observed order |
|---|---|---|
| Q1 (`degree=1`) | 7.89e-06 → 1.27e-07 | 1.97 → 2.00 |
| Q2 (`degree=2`) | 4.21e-07 → 8.94e-10 | 2.91 → 2.99 |

Those two orders are what `tests/test_warp_backend.py` asserts. A wrong Jacobian, a missing boundary term or a
source term with the wrong sign shows up as an error that stops falling when the mesh is refined — which is how the
sign error in the first version of this source term was found (it left the solution at `-u` and the error pinned at
`2‖u‖ = 1.665e-4` at every resolution).

## A differentiable step, against PyTorch autograd

`WarpBackend` steps the rigid-body state of `pinneapple_simulation.particle_dynamics.RigidBodyState` — the same
symplectic Euler arithmetic as `RigidBody.step` — as Warp kernels, recorded on a tape:

```python
from pinneapple_tools.compute_backends import WarpBackend

report = WarpBackend(3).gradients_against_torch(dt=0.02, n_steps=4)
assert report["gradients_agree_1e_6"], report["gradient_max_rel_diff"]
print(report["value_max_abs_diff"])          # the state, Warp vs torch: ~1e-16
print(report["gradient_max_rel_diff"])       # d state / d v0, d f, d torque: ~1e-16
```

Writing your own loop uses the same pieces: `state_arrays` → `step_with_tape(..., loss=backend.position_sq_loss)`
→ `tape.backward(loss=loss)` → read `.grad` on the arrays you passed in.

Three traps are worth knowing before writing a kernel of your own, because each one is silent:

- **The step is out of place.** A kernel that overwrites its own input makes Warp evaluate the adjoint at the values
  it just wrote: measured, that moved `d loss / d omega` by 5e-4 relative against autograd. With a separate output
  array the same kernel agrees to 0.0. `WarpBackend.step` returns a new state for this reason.
- **The loss has to be a kernel too.** A loss computed in numpy after the tape closes has no adjoint, and
  `tape.backward` then leaves every gradient at zero without an error.
- **Warp stores quaternions as `(x, y, z, w)`**; `RigidBodyState` uses `(w, x, y, z)`. A mismatch permutes the vector
  part, which the forward pass hides: only the quaternion compared against torch showed it.

## What this backend does not do

- It is **not** a `pp.set_backend("warp")` engine. Warp differentiates through an explicit `wp.Tape`, not through
  the functions you write, so it does not implement the `PhysicsBackend` array/`grad` interface; use it where you
  need kernels with a tape, not as a replacement for the torch/JAX backends.
- The step kernels are **float64 only** (Warp compiles one kernel per concrete type; float64 is the precision every
  gradient check here is made at). A CUDA build is selected automatically on an NVIDIA machine; the checks above run
  on the CPU.
- The gradient with respect to the **mass** is not provided: the reciprocal is taken on the host.

## Where it is checked

`tests/test_warp_backend.py` — the import without `warp-lang` and its install message, the shared buffer, the two
convergence orders, the CG residual, and the gradient agreement in 2-D and 3-D. `python -m pytest
tests/test_warp_backend.py -q` skips the Warp-dependent checks cleanly when the package is absent.
