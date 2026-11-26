"""Loading a native backend through the C ABI with ctypes: no extension module, no rebuild of Python.

``ctypes`` is a deliberate choice over PyO3 or pybind11 for *this* side of the boundary. A wrapped extension
module has to be compiled per interpreter version and per platform, which is the packaging tax the C ABI
exists to avoid; a plain shared library loaded by ctypes works for any library that speaks the header, whoever
wrote it and in whatever language. The performance argument does not change: the call overhead is a few
microseconds, and the kernels here run for milliseconds.

What the loader does before calling anything:

1. resolves the path (:mod:`pinneapple_core.native.build`);
2. reads ``pp_abi_version()`` and refuses a different ABI by name;
3. reads ``pp_ops()`` and compares the advertised signatures with the ones this loader knows, so a backend
   that kept a name and changed its meaning is caught rather than trusted;
4. on every call, checks arity and dtype on the Python side (a better message) and checks the C return code
   (the authority).

Errors from the library arrive as the sentence the C side wrote, never as a bare integer: the person reading
it calls the library, not writes it.
"""
from __future__ import annotations

import ctypes
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .abi import (
    ABI_VERSION,
    PP_OK,
    AbiVersionMismatch,
    NativeCallError,
    UnknownOperation,
    check_specs,
    parse_listing,
    spec,
)
from .build import library_path as _library_path
from .build import status as _build_status

__all__ = ["NativeLibrary", "load", "ERR_CAPACITY"]

ERR_CAPACITY = 512


class NativeLibrary:
    """One loaded library, with the ABI already checked. Build it with :func:`load`."""

    def __init__(self, path: Path, library: ctypes.CDLL) -> None:
        self.path = Path(path)
        self._lib = library
        self._bind()
        self.abi_version = int(self._fn_abi_version())
        if self.abi_version != ABI_VERSION:
            raise AbiVersionMismatch(self.abi_version)
        self.build = self._fn_build().decode("utf-8", "replace")
        self.served: dict[str, str] = parse_listing(self._ops_listing())
        self._checked = False
        # Refuse a backend that kept the names and changed their meaning before anyone calls it.
        self.verify()

    # -- ABI binding -------------------------------------------------------------------------------------
    def _bind(self) -> None:
        lib = self._lib
        double_p = ctypes.POINTER(ctypes.c_double)
        lib.pp_abi_version.restype = ctypes.c_int32
        lib.pp_abi_version.argtypes = []
        lib.pp_build.restype = ctypes.c_char_p
        lib.pp_build.argtypes = []
        lib.pp_ops.restype = ctypes.c_int32
        lib.pp_ops.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
        lib.pp_call.restype = ctypes.c_int32
        lib.pp_call.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(double_p), ctypes.POINTER(ctypes.c_int64), ctypes.c_int32,
            ctypes.POINTER(ctypes.c_double), ctypes.c_int32,
            ctypes.POINTER(ctypes.c_double), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64),
            ctypes.c_char_p, ctypes.c_int32,
        ]
        self._fn_abi_version = lib.pp_abi_version
        self._fn_build = lib.pp_build
        self._ops_listing = self._read_ops
        self._fn_call = lib.pp_call

    def _read_ops(self) -> str:
        needed = ctypes.c_int32(0)
        # First ask how much room the listing needs, then ask for it: the listing can grow with the backend.
        self._lib.pp_ops(None, 0, ctypes.byref(needed))
        capacity = max(int(needed.value), 1)
        buffer = ctypes.create_string_buffer(capacity)
        status = self._lib.pp_ops(buffer, capacity, ctypes.byref(needed))
        if status != PP_OK:
            raise NativeCallError("pp_ops", int(status), "the library could not list its operations")
        return buffer.value.decode("utf-8", "replace")

    # -- the checked surface -----------------------------------------------------------------------------
    def verify(self) -> None:
        """Refuse a library whose signatures are not the ones this loader expects (idempotent).

        Called once when the library is loaded, so a mismatched backend never reaches the dispatcher, and again
        before each call, which costs a boolean.
        """
        if self._checked:
            return
        problem = check_specs(self.served)
        if problem is not None:
            raise NativeCallError("pp_ops", PP_OK, problem)
        self._checked = True

    def ops(self) -> list[str]:
        """The operations this library actually serves, sorted."""
        return sorted(self.served)

    def supports(self, op: str) -> bool:
        return op in self.served

    def call(self, op: str, inputs: Sequence[np.ndarray], params: Sequence[float], out_size: int) -> np.ndarray:
        """Call ``op`` and return a fresh ``float64`` array of ``out_size`` values.

        The arrays in ``inputs`` are made contiguous ``float64`` here, because a non-contiguous array passed to
        C is the classic silent corruption: the kernel would read the buffer as if it were row-major.
        """
        self.verify()
        operation = spec(op)
        if not self.supports(op):
            raise UnknownOperation(f"{op!r} is not served by {self.path.name}; it serves {self.ops()}")
        if len(inputs) != operation.inputs:
            raise NativeCallError(op, PP_OK, f"takes {operation.inputs} input(s), got {len(inputs)}")
        if len(params) != operation.n_params:
            raise NativeCallError(op, PP_OK, f"takes params {operation.params}, got {len(params)}")
        arrays: list[np.ndarray] = [np.ascontiguousarray(a, dtype=np.float64).reshape(-1) for a in inputs]
        pointers = (ctypes.POINTER(ctypes.c_double) * len(arrays))(*(a.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)) for a in arrays)) if arrays else None
        sizes = (ctypes.c_int64 * max(len(arrays), 1))(*(len(a) for a in arrays))
        param_array = (ctypes.c_double * len(params))(*[float(p) for p in params])
        out = np.empty(int(out_size), dtype=np.float64)
        out_len = ctypes.c_int64(0)
        error = ctypes.create_string_buffer(ERR_CAPACITY)
        status = int(self._fn_call(
            op.encode("utf-8"),
            pointers, sizes, len(arrays),
            param_array, len(params),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), int(out_size), ctypes.byref(out_len),
            error, ERR_CAPACITY,
        ))
        if status != PP_OK:
            message = error.value.decode("utf-8", "replace") or "the library returned an error without a message"
            raise NativeCallError(op, status, message)
        if int(out_len.value) != int(out_size):
            raise NativeCallError(op, status, f"the library wrote {int(out_len.value)} values, expected {out_size}")
        return out

    def __repr__(self) -> str:
        return f"NativeLibrary({self.path.name!r}, abi={self.abi_version}, ops={len(self.served)})"


_CACHE: dict[str, NativeLibrary | None] = {}
_LAST_REASON: str | None = None


def load(path: Path | None = None) -> NativeLibrary | None:
    """Load (and cache) the native backend, or return ``None`` when this machine cannot run one.

    ``None`` is not a failure of the caller: the NumPy reference is the answer then. The reason is kept in
    :func:`last_reason` so a report can say why. Pass ``path`` to load a library you built yourself.
    """
    global _LAST_REASON
    key = str(path) if path is not None else "__default__"
    if key in _CACHE:
        return _CACHE[key]
    resolved = Path(path) if path is not None else _library_path()
    if resolved is None:
        _LAST_REASON = _build_status()["reason"]
        _CACHE[key] = None
        return None
    try:
        library = ctypes.CDLL(str(resolved))
    except OSError as exc:
        _LAST_REASON = f"could not load {resolved}: {exc}"
        _CACHE[key] = None
        return None
    try:
        result = NativeLibrary(resolved, library)
    except (AbiVersionMismatch, NativeCallError) as exc:
        # A library that speaks another ABI is refused, not called. That is a real refusal, and the caller
        # has to see it rather than be handed a silently slower fallback.
        _LAST_REASON = str(exc)
        _CACHE[key] = None
        raise
    _LAST_REASON = f"loaded {resolved}"
    _CACHE[key] = result
    return result


def last_reason() -> str | None:
    """Why :func:`load` returned ``None`` the last time it did."""
    return _LAST_REASON


def clear_cache() -> None:
    """Forget the loaded library. Used by tests that change the environment between cases."""
    global _LAST_REASON
    _CACHE.clear()
    _LAST_REASON = None
