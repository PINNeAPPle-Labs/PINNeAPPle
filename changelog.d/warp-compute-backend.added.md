**NVIDIA Warp backend** (`pinneapple_tools.compute_backends.warp_backend`, extra `pinneapple[warp]`): zero-copy
PyTorch tensors through DLPack (`to_warp` / `to_torch`, a non-contiguous tensor is refused rather than copied), a
Poisson problem assembled in `warp.fem` whose error against a manufactured solution is measured at every resolution
(`poisson_convergence`: Q1 order 1.97 → 2.00, Q2 2.91 → 2.99), and `WarpBackend`, one symplectic Euler step of the
`particle_dynamics` rigid-body state recorded on a `wp.Tape`, whose gradients agree with `torch.autograd` to 1e-16 in
2-D and 3-D (the test asserts 1e-6). Three traps that fail silently are documented where they were found: a kernel
that overwrites its own input makes Warp take the adjoint at the new value (5e-4 relative error on `d loss / d
omega`), a loss computed outside the tape leaves every gradient at zero, and Warp stores quaternions as
`(x, y, z, w)` where `RigidBodyState` uses `(w, x, y, z)`. `pp.doctor("gpu")` is a new profile for the NVIDIA stack
(Newton, Warp), one fix per missing item. Docs: `docs/core_concepts/warp_backend.md`.
