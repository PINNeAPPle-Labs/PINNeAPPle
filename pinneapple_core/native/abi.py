"""The contract between PINNeAPPle and a native kernel backend: one C ABI, N languages behind it.

Python stays the scientific interface and PINNeAPPle stays the orchestration layer; C, C++, Rust
(``extern "C"``), Fortran (``iso_c_binding``) or Zig are *execution backends*. The alternative — a bridge per
pair of languages — grows quadratically, and none of those bridges would be tested by this repository.

What a backend implements (the header is ``pinneapple_core/native/include/pinneapple_native.h``, and that file is
the source of truth)::

    int32_t     pp_abi_version(void);
    const char *pp_build(void);
    int32_t     pp_ops(char *buf, int32_t cap, int32_t *needed);
    int32_t     pp_call(const char *op, const double *const *inputs, const int64_t *input_sizes,
                        int32_t n_inputs, const double *params, int32_t n_params,
                        double *out, int64_t out_capacity, int64_t *out_size,
                        char *error, int32_t error_capacity);

Decisions worth knowing, because each one is a way a native layer goes wrong:

* **The version is checked before any call, not after.** A library built against ABI 2 called by an ABI-1
  loader is a struct-layout accident waiting to happen; :class:`AbiVersionMismatch` refuses it by name.
* **Nothing is allocated across the boundary.** The caller owns the output buffer and passes its capacity, so
  there is no ``free`` whose allocator may differ from the one that allocated.
* **A refused call writes nothing into ``out``.** The error sentence — not a code the caller has to translate —
  is what comes back, because the person reading it is a user of the library, not its author.
* **Sizes travel as doubles and are checked to be whole numbers.** The same rule the rest of the repository
  follows: a value that cannot be what it claims is refused rather than truncated.
* **An operation that cannot run is reported, not faked.** :class:`~pinneapple_core.native.kernels.NativeKernels`
  falls back to the NumPy implementation and says which one ran, so no result is ever attributed to a
  backend that did not produce it.

The operations this repository's C backend serves are listed in :data:`OPS`: three kernels of the same
five-point stencil, which is where the per-element cost of an interpreter stops being academic.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ABI_VERSION", "ERROR_NAMES", "OPS", "OpSpec", "NativeError", "AbiVersionMismatch", "CompilerNotFound",
    "BuildFailed", "NativeCallError", "UnknownOperation",
]

#: The ABI this loader speaks. A library reporting a different value is refused (see AbiVersionMismatch).
ABI_VERSION = 1

PP_OK = 0
PP_ERR_UNKNOWN_OP = -1
PP_ERR_BAD_ARGS = -2
PP_ERR_OUTPUT_SIZE = -3
PP_ERR_INTERNAL = -4

#: Return code to the name a message uses.
ERROR_NAMES = {
    PP_OK: "ok",
    PP_ERR_UNKNOWN_OP: "unknown_op",
    PP_ERR_BAD_ARGS: "bad_args",
    PP_ERR_OUTPUT_SIZE: "output_size",
    PP_ERR_INTERNAL: "internal",
}


class NativeError(RuntimeError):
    """Base class of every refusal from the native layer."""


class CompilerNotFound(NativeError):
    """No C compiler on this machine, so the library cannot be built here."""


class BuildFailed(NativeError):
    """The compiler ran and refused the source. Carries the command and its stderr."""

    def __init__(self, message: str, command: tuple[str, ...] = (), stderr: str = "") -> None:
        super().__init__(message)
        self.command = tuple(command)
        self.stderr = stderr


class AbiVersionMismatch(NativeError):
    """The library was built against a different ABI than this loader speaks."""

    def __init__(self, found: int, expected: int = ABI_VERSION) -> None:
        super().__init__(
            f"the library reports ABI {found}, this loader speaks ABI {expected}; "
            "a backend built for another contract is refused rather than called"
        )
        self.found = found
        self.expected = expected


class UnknownOperation(NativeError):
    """The library does not serve the operation that was asked for."""


class NativeCallError(NativeError):
    """A call reached the library and came back non-zero. ``code`` is the ABI return value."""

    def __init__(self, op: str, code: int, message: str) -> None:
        super().__init__(f"{op}: {message} ({ERROR_NAMES.get(code, code)})")
        self.op = op
        self.code = code
        self.message = message


@dataclass(frozen=True)
class OpSpec:
    """What one operation takes and gives, so the Python side can check before crossing the boundary.

    ``params`` are the scalars in signature order and ``inputs`` is how many arrays the op reads; the caller
    passes the output size, so what an op writes is named in ``signature`` rather than declared twice. The C
    side checks all of this again — a test asserts a wrong-sized output is refused, so the Python check is a
    better message, not the only guard.
    """

    name: str
    inputs: int
    params: tuple[str, ...]
    signature: str

    @property
    def n_params(self) -> int:
        return len(self.params)


OPS: dict[str, OpSpec] = {
    "laplacian_2d": OpSpec(
        name="laplacian_2d",
        inputs=1,
        params=("nx", "ny", "h"),
        signature="(u[nx*ny]; nx, ny, h) -> lap[nx*ny]",
    ),
    "heat_step_2d": OpSpec(
        name="heat_step_2d",
        inputs=1,
        params=("nx", "ny", "alpha", "dt", "steps"),
        signature="(u[nx*ny]; nx, ny, alpha, dt, steps) -> u[nx*ny]",
    ),
    "jacobi_2d": OpSpec(
        name="jacobi_2d",
        inputs=2,
        params=("nx", "ny", "h", "steps", "omega"),
        signature="(u0[nx*ny], f[nx*ny]; nx, ny, h, steps, omega) -> u[nx*ny]",
    ),
}


def spec(op: str) -> OpSpec:
    """The spec of ``op``, or :class:`UnknownOperation` naming what this build does serve."""
    try:
        return OPS[op]
    except KeyError:
        raise UnknownOperation(f"{op!r} is not one of {sorted(OPS)}") from None


def parse_listing(listing: str) -> dict[str, str]:
    """``pp_ops()`` output ("name<TAB>signature" per line) as ``{name: signature}``."""
    served: dict[str, str] = {}
    for line in listing.splitlines():
        line = line.rstrip("\n")
        if not line:
            continue
        name, _, signature = line.partition("\t")
        served[name.strip()] = signature.strip()
    return served


def check_specs(served: dict[str, str]) -> str | None:
    """Compare what the library advertises with what this loader expects.

    Returns ``None`` when they agree and a sentence when they do not. The signature strings are compared, not
    just the names: a backend that kept the name and changed the meaning of the params is exactly the
    mistake a name-only check misses.
    """
    missing = sorted(set(OPS) - set(served))
    if missing:
        return f"the library does not serve {missing}"
    differing = [name for name, signature in served.items() if name in OPS and signature != OPS[name].signature]
    if differing:
        details = "; ".join(f"{name}: library {served[name]!r}, loader {OPS[name].signature!r}" for name in differing)
        return f"the signatures disagree ({details})"
    return None
