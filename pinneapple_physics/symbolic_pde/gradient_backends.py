"""Pluggable derivative backends for :class:`SymbolicPDE`.

Design note -- this mirrors a pattern found in PhysicsNeMo's
``PhysicsInformer(grad_method=...)`` (see ``physicsnemo/sym/eq/phy_informer.py``
upstream, Apache-2.0; this module reimplements the *architecture pattern*,
not their code): separate "what is the PDE" (the SymPy expression, unchanged)
from "how do we compute a derivative given the shape of the data". The same
symbolic residual can then be evaluated either

* over scattered collocation points from a continuous model (PINN-style),
  differentiated via ``torch.autograd.grad`` -- the ``"autograd"`` backend,
  which is exactly ``SymbolicPDE``'s original (and still default) behavior, or
* over a structured grid (e.g. an FNO's output), differentiated via
  finite-difference or spectral (FFT) stencils -- the ``"finite_difference"``
  / ``"spectral"`` backends below.

The finite-difference and spectral math here is **not reimplemented**: it
calls straight through to ``pinneapple_neural.architectures.neural_operators
.pino``'s ``fd_derivative`` / ``spectral_derivative``, the same functions
already used (and exercised) by the PINO wrapper. This module only adds the
bookkeeping to (a) map SymPy coordinate names to grid tensor axes/extents and
(b) compose repeated/mixed partial derivatives (e.g. ``u_xx``, or a mixed
``u_xy``) out of those single-axis primitives.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Literal, Sequence

import torch

# NOTE: pino.py's fd_derivative/spectral_derivative are imported lazily,
# inside grid_derivative() below, rather than at module level. Importing
# `pinneapple_neural.architectures.neural_operators.pino` transitively
# triggers `pinneapple_neural`'s architecture registry (~25 model families
# registered on import, see architectures/register_all.py) -- a real but
# heavy cost that every SymbolicPDE user would otherwise pay just for
# importing pinneapple_physics.symbolic_pde, even those using the default
# grad_method="autograd" that never touches grid-based derivatives. The
# lazy import defers that cost to the (rarer) call site that actually needs
# finite-difference/spectral derivatives.

GradMethod = Literal["autograd", "finite_difference", "spectral"]
GridGradMethod = Literal["finite_difference", "spectral"]


@dataclass(frozen=True)
class GridAxisSpec:
    """Maps SymPy coordinate names to axes of a structured-grid field tensor.

    Parameters
    ----------
    coord_names : coordinate symbol names, e.g. ``("x", "t")`` for a 1D
        time-dependent PDE like Burgers'.
    dims : which tensor dimension of the (grid-shaped) field each entry of
        ``coord_names`` corresponds to, e.g. ``(-2, -1)`` for a field tensor
        shaped ``(..., n_x, n_t)``. Same convention as ``pino.GridSpec.dims``.
    L : physical extent spanned by each axis (same convention as
        ``pino.GridSpec.L``) -- used to derive finite-difference spacing
        (``dx = L / n``) and spectral wavenumbers.
    """

    coord_names: Sequence[str]
    dims: Sequence[int]
    L: Sequence[float]

    def axis_of(self, coord_name: str) -> int:
        try:
            return list(self.coord_names).index(coord_name)
        except ValueError as exc:
            raise KeyError(
                f"Coordinate '{coord_name}' not among this SymbolicPDE's grid "
                f"axes {list(self.coord_names)} -- pass a `grid=GridAxisSpec(...)` "
                f"that covers every coordinate the PDE differentiates with "
                f"respect to."
            ) from exc


def grid_derivative(
    field: torch.Tensor,
    wrt: List[str],
    *,
    grid: GridAxisSpec,
    method: GridGradMethod,
) -> torch.Tensor:
    """Compute an arbitrary-order partial derivative of a grid-shaped field.

    ``wrt`` lists coordinate names in differentiation order (the same
    convention ``compiler.py`` already expands SymPy's
    ``Derivative.variable_count`` into for the autograd backend), e.g.
    ``["x", "x"]`` for ``u_xx`` or ``["x", "y"]`` for a mixed ``u_xy``.

    Repeated differentiation along the SAME axis is composed into a single
    ``order=count`` call to the underlying primitive. Differentiation along
    DIFFERENT axes is applied sequentially, one axis at a time -- exact for
    the spectral case (differentiation along different axes commutes; each
    is an independent per-axis multiplication in Fourier space) and the
    standard way to build higher-dimensional finite-difference stencils out
    of 1D ones for the FD case.
    """
    from pinneapple_neural.architectures.neural_operators.pino import (
        fd_derivative,
        spectral_derivative,
    )

    counts = Counter(wrt)
    result = field
    for coord_name, order in counts.items():
        ax = grid.axis_of(coord_name)
        dim = grid.dims[ax]
        L = grid.L[ax]
        n = result.shape[dim]
        if method == "spectral":
            result = spectral_derivative(result, dim=dim, L=L, order=order)
        elif method == "finite_difference":
            if order not in (1, 2):
                raise ValueError(
                    f"grad_method='finite_difference' supports derivative order "
                    f"1 or 2 per axis (pino.fd_derivative's own limit); got "
                    f"order {order} for coordinate '{coord_name}'. A higher "
                    f"order would need chaining first-order calls, not "
                    f"currently wired up here."
                )
            dx = L / n
            result = fd_derivative(result, dim=dim, dx=dx, order=order)
        else:
            raise ValueError(f"Unknown grid grad_method: {method!r}")
    return result
