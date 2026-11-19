"""Native execution backends: one C ABI, and whichever language implements it.

PINNeAPPle is the orchestration layer, Python the scientific interface, and the compiled code an execution
backend behind a checked boundary. A backend implements four functions from
``pinneapple_core/native/include/pinneapple_native.h`` and this package calls it — in C, C++, Rust
(``extern "C"``), Fortran (``iso_c_binding``) or Zig, without a bridge per pair of languages.

    from pinneapple_core.native import NativeKernels, native_status

    native_status()
    # {'enabled': True, 'compiler': '/usr/bin/cc', 'available': True,
    #  'library': '/Users/.../Caches/pinneapple/native/<hash>/libpinneapple_native.dylib', ...}

    k = NativeKernels()
    k.backend                      # 'native:libpinneapple_native.dylib'
    k.laplacian_2d(u)              # NumPy array; the same numbers as the reference

What is here:

* :mod:`pinneapple_core.native.abi` — the contract: ``ABI_VERSION``, the operations and their signatures, and
  the refusals a caller can receive by name;
* :mod:`pinneapple_core.native.build` — compiles the C source into a user cache, keyed by the fingerprint of
  what it compiles; ``PINNEAPPLE_NATIVE=0`` opts out, ``PINNEAPPLE_NATIVE_CC`` names a compiler;
* :mod:`pinneapple_core.native.loader` — ctypes, with the ABI checked before the first call;
* :mod:`pinneapple_core.native.kernels` — the operations in NumPy (the reference) and the dispatcher that says
  which backend produced a result.

Honest limits, which the interop page repeats: the only backend built and measured here is the C one on this
machine's compiler; a Rust backend speaks the same header but is not built in this repository, and no result
here is claimed for it. A machine without a compiler gets the NumPy reference and a reason, never a wrong
number and never a silent claim of speed.
"""
from .abi import (
    ABI_VERSION,
    OPS,
    AbiVersionMismatch,
    BuildFailed,
    CompilerNotFound,
    NativeCallError,
    NativeError,
    OpSpec,
    UnknownOperation,
)
from .build import build as build_library
from .build import cache_root, compiler, enabled, library_path, status
from .kernels import (
    NativeKernels,
    heat_step_2d_reference,
    jacobi_2d_reference,
    laplacian_2d_reference,
)
from .loader import NativeLibrary, clear_cache, load

__all__ = [
    "ABI_VERSION", "OPS", "OpSpec", "NativeError", "AbiVersionMismatch", "CompilerNotFound", "BuildFailed",
    "NativeCallError", "UnknownOperation",
    "NativeLibrary", "load", "clear_cache",
    "NativeKernels", "laplacian_2d_reference", "heat_step_2d_reference", "jacobi_2d_reference",
    "build_library", "cache_root", "compiler", "enabled", "library_path", "status", "available", "native_status",
]


def available() -> tuple:
    """``(usable, reason)``: whether a native backend can run here, and the sentence that explains it."""
    info = status()
    return bool(info["available"]), str(info["reason"])


def native_status() -> dict:
    """The full status dictionary of :func:`pinneapple_core.native.build.status`."""
    return status()
