"""The kernel operations, once in NumPy and once in C, behind one dispatcher that says which one ran.

This is the shape the whole native layer is built around: **the reference implementation is not a fallback for
a machine without a compiler, it is the definition of what the kernel computes.** The C version exists to be
faster at the same arithmetic, and a test asserts the two agree element by element. A backend that is absent or
built for another ABI therefore never changes an answer — it changes the speed and the sentence a report prints.

::

    from pinneapple_core.native import NativeKernels

    k = NativeKernels()
    k.backend                       # 'native:libpinneapple_native.dylib' or 'reference:numpy'
    u = k.jacobi_2d(u0, f, steps=2000, omega=2 / 3)

The three kernels are the same five-point stencil:

* :meth:`NativeKernels.laplacian_2d` — one pass of the discrete Laplacian, zero Dirichlet on the boundary;
* :meth:`NativeKernels.heat_step_2d` — ``steps`` explicit steps of ``u_t = alpha * laplacian(u)`` on the unit
  square, with the stability limit of the scheme enforced rather than returned as a blowing-up field;
* :meth:`NativeKernels.jacobi_2d` — ``steps`` Jacobi sweeps for ``-laplacian(u) = f``.

Boundary convention, everywhere: the boundary nodes of the grid are Dirichlet zeros. The interior is what the
kernels compute, and the boundary of the input is never read — a kernel that quietly used a stale boundary value
would be a different operator per call.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .build import status as _build_status
from .loader import last_reason as _last_reason
from .loader import load as _load

__all__ = [
    "NativeKernels", "laplacian_2d_reference", "heat_step_2d_reference", "jacobi_2d_reference", "native_status",
]

Array = np.ndarray


def _grid(u: Array, name: str = "u") -> Array:
    """Validate a kernel input and return it as a contiguous ``float64`` 2-D array."""
    array = np.ascontiguousarray(u, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D grid, got shape {array.shape}")
    if min(array.shape) < 3:
        raise ValueError(f"{name} must be at least 3 x 3 (a boundary ring plus one interior node), got {array.shape}")
    return array


def _h_from(size: int) -> float:
    return 1.0 / float(size - 1)


def _stability(alpha: float, dt: float, nx: int, ny: int) -> tuple[float, float]:
    """The per-axis coefficients of the explicit step and its stability check.

    A caller that asks for ``alpha * dt`` above the limit of the scheme gets a refusal with the limit in it,
    not a field multiplied by 1e300 after twenty steps.
    """
    hx, hy = _h_from(nx), _h_from(ny)
    rx, ry = alpha * dt / (hx * hx), alpha * dt / (hy * hy)
    if rx + ry > 0.5:
        limit = 0.5 / (1.0 / (hx * hx) + 1.0 / (hy * hy))
        raise ValueError(
            f"alpha*dt = {alpha * dt:g} exceeds the explicit scheme's limit {limit:g} on a {nx} x {ny} grid "
            "(alpha*dt*(1/hx^2 + 1/hy^2) > 0.5); the step would be unstable"
        )
    return rx, ry


# -- the reference implementations ------------------------------------------------------------------------


def laplacian_2d_reference(u: Array, h: float) -> Array:
    """Five-point Laplacian of the interior; zero on the boundary ring."""
    u = _grid(u)
    if not h > 0.0:
        raise ValueError(f"h must be positive, got {h!r}")
    inv_h2 = 1.0 / (h * h)
    lap = np.zeros_like(u)
    lap[1:-1, 1:-1] = (
        u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1] - 4.0 * u[1:-1, 1:-1]
    ) * inv_h2
    return lap


def heat_step_2d_reference(u: Array, alpha: float, dt: float, steps: int) -> Array:
    """``steps`` explicit steps of ``u_t = alpha * laplacian(u)`` on the unit square."""
    out = _grid(u)
    if not alpha > 0.0:
        raise ValueError(f"alpha must be positive, got {alpha!r}")
    if not dt > 0.0:
        raise ValueError(f"dt must be positive, got {dt!r}")
    steps = int(steps)
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")
    nx, ny = out.shape
    rx, ry = _stability(alpha, dt, nx, ny)
    current = out.copy()
    for _ in range(steps):
        nxt = np.zeros_like(current)
        nxt[1:-1, 1:-1] = (
            current[1:-1, 1:-1]
            + rx * (current[1:-1, :-2] - 2.0 * current[1:-1, 1:-1] + current[1:-1, 2:])
            + ry * (current[:-2, 1:-1] - 2.0 * current[1:-1, 1:-1] + current[2:, 1:-1])
        )
        current = nxt
    return current


def jacobi_2d_reference(u0: Array, f: Array, h: float, steps: int, omega: float = 1.0) -> Array:
    """``steps`` weighted-Jacobi sweeps for ``-laplacian(u) = f`` with zero Dirichlet data."""
    u = _grid(u0, "u0")
    f = _grid(f, "f")
    if u.shape != f.shape:
        raise ValueError(f"u0 and f must have the same shape, got {u.shape} and {f.shape}")
    if not h > 0.0:
        raise ValueError(f"h must be positive, got {h!r}")
    if not 0.0 < omega <= 1.0:
        raise ValueError(
            f"omega must be in (0, 1], got {omega!r}: damped Jacobi is stable only up to 1 "
            "(its iteration matrix has eigenvalues 1 - omega*(1 - mu), and mu reaches cos(pi*h))"
        )
    steps = int(steps)
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")
    h2 = h * h
    keep = 1.0 - omega
    current = u.copy()
    for _ in range(steps):
        nxt = np.zeros_like(current)
        neighbour_sum = current[1:-1, :-2] + current[1:-1, 2:] + current[:-2, 1:-1] + current[2:, 1:-1]
        nxt[1:-1, 1:-1] = keep * current[1:-1, 1:-1] + omega * 0.25 * (neighbour_sum + h2 * f[1:-1, 1:-1])
        current = nxt
    return current


# -- the dispatcher ---------------------------------------------------------------------------------------


class NativeKernels:
    """The kernel operations, on the native backend when this machine has one and on NumPy otherwise.

    ``backend`` says which one produced the last result, so a report never attributes a number to a compiled
    library that did not run. ``NativeKernels(force_reference=True)`` pins the reference, which is what a test
    comparing the two needs.
    """

    def __init__(self, force_reference: bool = False, library: object | None = None) -> None:
        self._library = None if force_reference else (library if library is not None else _load())
        self._reason = "" if force_reference else _last_reason() or ""
        self.backend = "reference:numpy" if self._library is None else f"native:{self._library.path.name}"

    # -- introspection -----------------------------------------------------------------------------------
    @property
    def is_native(self) -> bool:
        return self._library is not None

    def describe(self) -> dict:
        """What this instance will do, and why — the dictionary a report or a log prints."""
        if self._library is None:
            return {"backend": self.backend, "native": False, "reason": self._reason or "forced reference"}
        return {
            "backend": self.backend,
            "native": True,
            "library": str(self._library.path),
            "build": self._library.build,
            "abi_version": self._library.abi_version,
            "ops": self._library.ops(),
        }

    def ops(self) -> Sequence[str]:
        """The operations this instance serves: what the native backend advertises, or all of them."""
        return self._library.ops() if self._library is not None else _OPS_NAMES

    # -- the kernels -------------------------------------------------------------------------------------
    def laplacian_2d(self, u: Array, h: float | None = None) -> Array:
        """Discrete Laplacian of ``u``; ``h`` defaults to the unit square's spacing ``1/(nx-1)``."""
        u = _grid(u)
        h = _h_from(u.shape[0]) if h is None else float(h)
        if not h > 0.0:
            raise ValueError(f"h must be positive, got {h!r}")
        if self._library is not None:
            flat = self._library.call("laplacian_2d", [u], (u.shape[0], u.shape[1], h), u.size)
            return flat.reshape(u.shape)
        return laplacian_2d_reference(u, h)

    def heat_step_2d(self, u: Array, alpha: float, dt: float, steps: int) -> Array:
        """``steps`` explicit heat steps on the unit square, with the scheme's limit enforced first."""
        u = _grid(u)
        alpha, dt, steps = float(alpha), float(dt), int(steps)
        _stability(alpha, dt, u.shape[0], u.shape[1])
        if self._library is not None:
            flat = self._library.call("heat_step_2d", [u], (u.shape[0], u.shape[1], alpha, dt, steps), u.size)
            return flat.reshape(u.shape)
        return heat_step_2d_reference(u, alpha, dt, steps)

    def jacobi_2d(self, u0: Array, f: Array, h: float | None = None, steps: int = 100,
                  omega: float = 1.0) -> Array:
        """``steps`` Jacobi sweeps for ``-laplacian(u) = f``; ``h`` defaults to ``1/(nx-1)``."""
        u0 = _grid(u0, "u0")
        f = _grid(f, "f")
        if u0.shape != f.shape:
            raise ValueError(f"u0 and f must have the same shape, got {u0.shape} and {f.shape}")
        h = _h_from(u0.shape[0]) if h is None else float(h)
        if not h > 0.0:
            raise ValueError(f"h must be positive, got {h!r}")
        steps, omega = int(steps), float(omega)
        if not 0.0 < omega <= 1.0:
            raise ValueError(f"omega must be in (0, 1], got {omega!r}")
        if self._library is not None:
            flat = self._library.call("jacobi_2d", [u0, f], (u0.shape[0], u0.shape[1], h, steps, omega), u0.size)
            return flat.reshape(u0.shape)
        return jacobi_2d_reference(u0, f, h, steps, omega)

    def __repr__(self) -> str:
        return f"NativeKernels(backend={self.backend!r})"


_OPS_NAMES = ("laplacian_2d", "heat_step_2d", "jacobi_2d")


def native_status() -> dict:
    """Whether a native backend is usable here, with the reason — see :func:`pinneapple_core.native.available`."""
    return _build_status()
