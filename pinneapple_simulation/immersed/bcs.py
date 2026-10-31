"""Boundary conditions on surfaces that are not on the grid.

Each condition applies where its ``where`` selector holds on the boundary quadrature points: ``where(X)`` or
``where(X, n)`` (points and outward normals) returning a boolean mask, or ``None`` for the whole boundary. A point
takes the first condition in the list that selects it; points no condition selects are natural (zero flux, free
surface). Values are constants or callables of the points (``f(X)`` or ``f(X, n)``).
"""
from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def _arity(fn: Callable) -> int:
    try:
        return len([p for p in inspect.signature(fn).parameters.values()
                    if p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
    except (TypeError, ValueError):
        return 1


def evaluate(value: Any, X: np.ndarray, n: np.ndarray | None = None, shape: tuple = ()) -> np.ndarray:
    """A constant, a vector or a callable of (X) or (X, n), as an array (len(X),) + shape."""
    if callable(value):
        out = value(X, n) if _arity(value) >= 2 else value(X)
    else:
        out = value
    return np.broadcast_to(np.asarray(out, float), (len(X),) + shape).astype(float)


def select(where: Callable | None, X: np.ndarray, n: np.ndarray) -> np.ndarray:
    if where is None:
        return np.ones(len(X), bool)
    m = where(X, n) if _arity(where) >= 2 else where(X)
    return np.broadcast_to(np.asarray(m, bool), (len(X),))


@dataclass
class Dirichlet:
    """Prescribed temperature (Nitsche's method)."""
    value: Any = 0.0
    where: Callable | None = None
    kind = "dirichlet"


@dataclass
class HeatFlux:
    """Prescribed heat flux INTO the body [W/m^2] (Neumann); negative values extract heat."""
    flux: Any = 0.0
    where: Callable | None = None
    kind = "neumann"


@dataclass
class Convection:
    """Convection (Robin): the flux out of the body is ``h (T - T_inf)``."""
    h: Any
    T_inf: Any = 0.0
    where: Callable | None = None
    kind = "robin"


@dataclass
class Displacement:
    """Prescribed displacement (Nitsche); ``components`` restricts it to some directions (e.g. (2,) = only u_z,
    a frictionless support)."""
    value: Any = 0.0
    where: Callable | None = None
    components: Sequence[int] | None = None
    kind = "dirichlet"


@dataclass
class Traction:
    """Surface traction [Pa]: a vector, or a callable of (X) / (X, n) returning (n, d)."""
    value: Any
    where: Callable | None = None
    kind = "neumann"


@dataclass
class Pressure:
    """Normal pressure [Pa] pushing on the surface (traction -p n)."""
    p: Any
    where: Callable | None = None
    kind = "neumann"


@dataclass
class Force:
    """A total force [N] (vector) spread uniformly over the surface its selector picks: the applied load is exact
    whatever area the discretisation finds there (rounded edges, facets)."""
    total: Any
    where: Callable | None = None
    kind = "neumann"


def assign(bcs: Sequence, X: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Index of the condition that owns each boundary point (-1: natural)."""
    owner = np.full(len(X), -1, np.int64)
    for i, bc in enumerate(bcs):
        m = select(bc.where, X, n) & (owner < 0)
        owner[m] = i
    return owner
