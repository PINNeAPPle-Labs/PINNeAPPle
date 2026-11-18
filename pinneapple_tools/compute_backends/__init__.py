"""Multi-backend support for PINNeAPPle: PyTorch (default), JAX and NVIDIA Warp.

Provides a backend abstraction layer so PINN code can run on PyTorch or JAX
without modification.  JAX gives ``vmap``/``jit`` speedups on CPU/TPU; the
default PyTorch backend requires no extra dependencies.

Quick start::

    from pinneapple_tools.compute_backends import set_backend, get_backend, JAXBackend

    # Switch to JAX (requires: pip install jax jaxlib)
    set_backend("jax")
    print(get_backend())  # "jax"

    # JIT-compile a PINN. residual_fn receives (u_fn, x), where u_fn is
    # model_fn bound to params (x -> u(x)), so it can take derivatives via
    # jax.grad/jax.hessian -- not a precomputed value, which JAX cannot
    # differentiate.
    compiled = JAXBackend.jit_pinn(model_fn, residual_fn)

    # Vectorise a single-point residual over a batch of collocation points
    batched = JAXBackend.vmap_residual(single_pt_residual)

    # Convert between PyTorch and JAX
    jax_x = JAXBackend.torch_to_jax(torch_tensor)
    torch_x = JAXBackend.jax_to_torch(jax_array)

NVIDIA Warp (``pip install warp-lang``) is a third entry point, for work a PinT/JAX engine cannot do: GPU kernels
with a reverse-mode tape, ``warp.fem`` weak forms and zero-copy torch buffers. It is not a drop-in array engine
(Warp differentiates through an explicit ``wp.Tape``, not through the functions you write), so it has its own
namespace::

    from pinneapple_tools.compute_backends import warp_available, WarpBackend, poisson_convergence

    poisson_convergence((8, 16, 32))          # FEM manufactured-solution order, assembled in warp.fem
    WarpBackend(3).gradients_against_torch()  # d state / d v0 from the tape vs torch.autograd
"""

from .backend import Backend, get_backend, set_backend
from .jax_backend import JAXBackend, jax_available, jax_pinn, jit_pinn, vmap_residual
from .warp_backend import (
    MANUFACTURED_SOLUTION,
    FemPoissonSolution,
    WarpBackend,
    fem_poisson,
    poisson_convergence,
    to_torch,
    to_warp,
    warp_available,
    warp_device,
)

__all__ = [
    # Backend registry
    "Backend",
    "get_backend",
    "set_backend",
    # JAX utilities
    "JAXBackend",
    "jax_available",
    "jax_pinn",
    "jit_pinn",
    "vmap_residual",
    # NVIDIA Warp: zero-copy tensors, warp.fem weak forms, a differentiable rigid-body step
    "WarpBackend",
    "FemPoissonSolution",
    "warp_available",
    "warp_device",
    "to_warp",
    "to_torch",
    "fem_poisson",
    "poisson_convergence",
    "MANUFACTURED_SOLUTION",
]
