- `pinneapple_core.native` (INT-015): native execution backends behind one versioned C ABI, with the NumPy
  implementation as the definition and C as the accelerator.
  - The contract is `pinneapple_core/native/include/pinneapple_native.h` (`pp_abi_version`, `pp_build`, `pp_ops`,
    `pp_call`), so C, C++, Rust (`extern "C"`), Fortran (`iso_c_binding`) and Zig load through the same `ctypes`
    code instead of one bridge per pair of languages. `AbiVersionMismatch` refuses a library built for another
    version, and a library whose advertised signatures differ is refused too.
  - Three kernels, in C and in NumPy: `laplacian_2d`, `heat_step_2d` and `jacobi_2d`, all of the five-point
    stencil with zero Dirichlet data. `NativeKernels` dispatches and says which backend produced the result.
  - Measured on one machine (Apple clang 21, arm64), the two backends timed back to back: single-digit on the
    one-pass kernel and about an order of magnitude on the iterated ones over NumPy's vectorised loop, agreement
    from 0.0 (Laplacian, bit for bit) to machine epsilon, and both backends landing on the stencil's truncation
    error `pi^2 h^2 / 12` against a manufactured solution — `python -m benchmarks.native_backend`.
  - `PINNEAPPLE_NATIVE=0` opts out, `PINNEAPPLE_NATIVE_CC` names a compiler, and a machine without one gets the
    NumPy reference with the reason instead of an error. Damped Jacobi now refuses `omega > 1`: it diverged to
    `inf` there in the first run, so the range is part of the contract.
