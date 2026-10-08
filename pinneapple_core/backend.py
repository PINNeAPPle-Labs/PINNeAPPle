"""PhysicsBackend: one array/derivative/solve API over torch, jax and anything registered.

Code written against a :class:`PhysicsBackend` does not name an engine::

    from pinneapple_core.backend import get_physics_backend

    def energy(bk, L):                        # bk is the active backend
        x = bk.linspace(0.0, L, 101)
        return bk.integrate(bk.sin(x) ** 2, x)

    bk = get_physics_backend()
    dE = bk.grad(lambda L: energy(bk, L))(bk.asarray(2.0))

``pp.set_backend("jax")`` (or ``with use_backend("jax"):``) switches the engine, and the same function
returns the same numbers. Float64 is the default dtype so results agree across engines.

Operations: ``asarray, to_numpy, linspace, zeros, ones, eye, diag, matmul, solve, integrate`` and the
transforms ``grad, jacobian, vmap, jit`` (``jit`` is a no-op where the engine has none). Elementwise
functions (``sin, exp, sum, stack, where, ...``) live on ``bk.xp``, the engine's own numpy-like module.

Add an engine with ``register_backend("name", MyBackend)``; ``MyBackend`` subclasses :class:`PhysicsBackend`.
"""
from __future__ import annotations

import contextlib
from typing import Any, Callable, Dict, Optional, Type

import numpy as np

__all__ = ["PhysicsBackend", "TorchPhysicsBackend", "JaxPhysicsBackend", "register_backend", "list_backends",
           "get_physics_backend", "use_backend"]


class PhysicsBackend:
    """Interface every engine implements. All array arguments and results are native to the engine."""

    name: str = "abstract"
    xp: Any = None  # numpy-like module for elementwise math

    # arrays
    def asarray(self, x, dtype=None): raise NotImplementedError
    def to_numpy(self, x) -> np.ndarray: raise NotImplementedError
    def linspace(self, start, stop, num: int):
        """``num`` points from ``start`` to ``stop``, differentiable in both ends."""
        raise NotImplementedError
    def zeros(self, *shape: int): raise NotImplementedError
    def ones(self, *shape: int): raise NotImplementedError
    def eye(self, n: int): raise NotImplementedError
    def diag(self, v, k: int = 0): raise NotImplementedError

    # linear algebra and integration
    def matmul(self, a, b): raise NotImplementedError
    def solve(self, a, b): raise NotImplementedError

    def integrate(self, y, x=None, dx: float = 1.0, axis: int = -1):
        """Trapezoid rule of ``y`` along ``axis`` over the points ``x`` (or spacing ``dx``)."""
        xp = self.xp
        y = self.asarray(y)
        n = y.shape[axis]
        lo = [slice(None)] * y.ndim
        hi = [slice(None)] * y.ndim
        lo[axis], hi[axis] = slice(0, n - 1), slice(1, n)
        mid = (y[tuple(lo)] + y[tuple(hi)]) / 2.0
        if x is None:
            return mid.sum(axis=axis) * dx
        x = self.asarray(x)
        h = x[1:] - x[:-1]
        shape = [1] * y.ndim
        shape[axis] = n - 1
        return (mid * h.reshape(shape)).sum(axis=axis)

    # transforms
    def grad(self, fn: Callable, argnum: int = 0) -> Callable: raise NotImplementedError
    def jacobian(self, fn: Callable, argnum: int = 0) -> Callable: raise NotImplementedError
    def vmap(self, fn: Callable, in_axes=0) -> Callable: raise NotImplementedError
    def jit(self, fn: Callable) -> Callable: return fn

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class TorchPhysicsBackend(PhysicsBackend):
    name = "torch"

    def __init__(self) -> None:
        import torch

        self._t = torch
        self.xp = torch
        self.dtype = torch.float64

    def asarray(self, x, dtype=None):
        t = self._t
        if isinstance(x, t.Tensor):
            return x if dtype is None else x.to(dtype)
        return t.as_tensor(np.asarray(x), dtype=dtype or self.dtype)

    def to_numpy(self, x) -> np.ndarray:
        return x.detach().cpu().numpy() if isinstance(x, self._t.Tensor) else np.asarray(x)

    def linspace(self, start, stop, num):
        # built from arange so it is differentiable in start and stop (torch.linspace is not)
        t = self._t
        start, stop = self.asarray(start), self.asarray(stop)
        return start + (stop - start) * t.arange(num, dtype=self.dtype) / max(num - 1, 1)

    def zeros(self, *shape):
        return self._t.zeros(*shape, dtype=self.dtype)

    def ones(self, *shape):
        return self._t.ones(*shape, dtype=self.dtype)

    def eye(self, n):
        return self._t.eye(n, dtype=self.dtype)

    def diag(self, v, k=0):
        return self._t.diag(v, k)

    def matmul(self, a, b):
        return self._t.matmul(a, b)

    def solve(self, a, b):
        return self._t.linalg.solve(a, b)

    def grad(self, fn, argnum=0):
        return self._t.func.grad(fn, argnums=argnum)

    def jacobian(self, fn, argnum=0):
        return self._t.func.jacrev(fn, argnums=argnum)

    def vmap(self, fn, in_axes=0):
        return self._t.func.vmap(fn, in_dims=in_axes)


class JaxPhysicsBackend(PhysicsBackend):
    name = "jax"

    def __init__(self) -> None:
        import jax

        jax.config.update("jax_enable_x64", True)  # float64 so results match the other engines
        import jax.numpy as jnp

        self._jax = jax
        self.xp = jnp

    def asarray(self, x, dtype=None):
        return self.xp.asarray(x, dtype=dtype or self.xp.float64)

    def to_numpy(self, x) -> np.ndarray:
        return np.asarray(x)

    def linspace(self, start, stop, num):
        xp = self.xp
        start, stop = self.asarray(start), self.asarray(stop)
        return start + (stop - start) * xp.arange(num, dtype=xp.float64) / max(num - 1, 1)

    def zeros(self, *shape):
        return self.xp.zeros(shape, dtype=self.xp.float64)

    def ones(self, *shape):
        return self.xp.ones(shape, dtype=self.xp.float64)

    def eye(self, n):
        return self.xp.eye(n, dtype=self.xp.float64)

    def diag(self, v, k=0):
        return self.xp.diag(v, k)

    def matmul(self, a, b):
        return self.xp.matmul(a, b)

    def solve(self, a, b):
        return self.xp.linalg.solve(a, b)

    def grad(self, fn, argnum=0):
        return self._jax.grad(fn, argnums=argnum)

    def jacobian(self, fn, argnum=0):
        return self._jax.jacrev(fn, argnums=argnum)

    def vmap(self, fn, in_axes=0):
        return self._jax.vmap(fn, in_axes=in_axes)

    def jit(self, fn):
        return self._jax.jit(fn)


# -- registry ----------------------------------------------------------------
_REGISTRY: Dict[str, Type[PhysicsBackend]] = {"torch": TorchPhysicsBackend, "jax": JaxPhysicsBackend}
_INSTANCES: Dict[str, PhysicsBackend] = {}


def register_backend(name: str, cls: Type[PhysicsBackend]) -> None:
    """Make ``name`` available to ``set_backend`` / ``use_backend``. ``cls`` is instantiated lazily, once."""
    if not (isinstance(cls, type) and issubclass(cls, PhysicsBackend)):
        raise TypeError("cls must be a subclass of PhysicsBackend")
    if name in _REGISTRY:
        raise ValueError(f"backend '{name}' is already registered")
    _REGISTRY[name] = cls


def list_backends() -> list:
    return sorted(_REGISTRY)


def get_physics_backend(name: Optional[str] = None) -> PhysicsBackend:
    """The backend object for ``name``, or for the active backend set with ``set_backend``."""
    from pinneapple_tools.compute_backends.backend import get_backend

    name = name or get_backend()
    if name not in _REGISTRY:
        raise ValueError(f"unknown backend '{name}'; registered: {list_backends()}")
    if name not in _INSTANCES:
        try:
            _INSTANCES[name] = _REGISTRY[name]()
        except ImportError as exc:
            raise ImportError(f"backend '{name}' needs a package that is not installed: {exc}") from exc
    return _INSTANCES[name]


@contextlib.contextmanager
def use_backend(name: str):
    """``with use_backend("jax"):`` runs the block on that backend and restores the previous one."""
    from pinneapple_tools.compute_backends import backend as _b

    previous = _b.get_backend()
    _b.set_backend(name)
    try:
        yield get_physics_backend(name)
    finally:
        _b.set_backend(previous)
