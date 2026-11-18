"""Integral equations of the second kind as PDE kinds of the problem compiler, in Nyström form.

The equations are Fredholm and Volterra of the second kind over an interval::

    Fredholm   u(x) = f(x) + lambda * int_a^b K(x, s) u(s) ds
    Volterra   u(x) = f(x) + lambda * int_a^x K(x, s) u(s) ds

The integral is replaced by a fixed quadrature on the collocation points themselves, so the residual is an algebraic
equation in the values the network already produces -- no extra model evaluations, no autograd through the kernel::

    r(x_i) = u(x_i) - f(x_i) - lambda * [ sum_j w_j K(x_i, x_j) u(x_j) + D_i u(x_i) ]

with ``w`` the composite-trapezoid weights on the nodes sorted by ``x`` and ``D_i`` the analytic integral of the
singular part of the kernel over the *self cell* of node ``i`` (see below). Because the coupling is a matrix-vector
product, the same assembly gives the classical direct solve, which is what makes the neural solution checkable
against something other than itself.

Weakly singular kernels
-----------------------
For ``K = ln|x - s|`` the diagonal of the matrix is infinite and the naive rule diverges. The standard remedy is
applied here: the kernel is split into a singular part, whose integral over the cell containing ``x_i`` is analytic,
and a remainder integrated numerically::

    int K u ~ sum_{j != i} w_j K(x_i, x_j) u(x_j) + D_i u(x_i),    D_i = int_{x_i - w_i/2}^{x_i + w_i/2} K(x_i, s) ds

The pair ``(kernel, diag)`` is what the caller supplies: ``diag`` is the analytic cell integral of the singular
part, and ``kernel`` is what is summed numerically. For a bounded kernel no ``diag`` is needed and the diagonal is
the trapezoid's own ``w_i K(x_i, x_i)``, which is the same rule.

``D`` is which for the cell of width ``w_i``; for ``ln|s|`` that integral is ``w (ln(w/2) - 1)`` exactly (see
:func:`log_diag`). For Volterra the integration stops at ``x_i``, so only the left half of the self cell counts and
``D`` is halved -- exact for the distance kernels used here, which are even in the local coordinate.

Nodes
-----
The collocation points **are** the Nyström nodes. That is what makes the kind cheap, and it also means a resampled
collocation set turns the quadrature error into noise: for a deterministic quadrature, hand the compiler a fixed
grid. ``nystrom_matrix`` sorts the nodes internally, so the batch may arrive in any order.

Scope of this module: one spatial coordinate, real kernels with one field and complex kernels with two (the
``Er``/``Ei`` convention of :mod:`~pinneapple_physics.pinn_solver.compiler.em_kinds`), and a dense ``(N, N)``
assembly -- adequate up to a few thousand nodes, beyond which a fast method is the right tool and is not here.
Integral equations of the first kind and hypersingular kernels are refused rather than approximated.

References: Atkinson, *The Numerical Solution of Integral Equations of the Second Kind* (Cambridge, 1997), ch. 3-4
(Nyström methods) and Delves & Mohamed, *Computational Methods for Integral Equations* (Cambridge, 1985), ch. 6
(product integration and the cell-integral diagonal).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from .em_kinds import _at, _spatial

__all__ = [
    "INTEGRAL_KINDS", "integral_residuals", "nystrom_matrix", "trapezoid_weights",
    "constant_kernel", "product_kernel", "log_kernel", "log_diag", "helmholtz2_kernel", "helmholtz2_diag",
]

INTEGRAL_KINDS = frozenset({"fredholm2", "volterra2"})


# ------------------------------------------------------------------------------------------------ quadrature
def trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """Composite-trapezoid weights on sorted nodes ``x`` (N,): ``(x[i+1] - x[i-1]) / 2`` inside, half a cell at
    the ends. The weights sum to ``x[-1] - x[0]``, which is also the width of each node's self cell in the sense the
    diagonal replacement needs."""
    x = np.asarray(x, float).reshape(-1)
    n = x.size
    if n < 3:
        raise ValueError(f"integral kinds need at least 3 collocation points, got {n}")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("collocation points must be distinct along the spatial coordinate")
    w = np.empty(n)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


def nystrom_matrix(x: np.ndarray, kernel_fn: Callable, ctx: dict[str, Any] | None = None, *,
                   diag_fn: Callable | None = None, volterra: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Assemble ``(A, D)`` such that ``int K(x_i, s) u(s) ds ~ (A @ u)_i + D_i u_i`` on the nodes ``x``.

    ``A`` has the column weights folded in and a zero diagonal; ``D`` is the self-cell contribution. The pairing
    ``A + diag(D)`` is the Nyström matrix, so the direct solve of ``u = f + lambda K u`` is
    ``np.linalg.solve(np.eye(N) - lambda * (A + np.diag(D)), f_sorted)``. Nodes are sorted internally and the
    returned matrices are in that sorted order; :func:`integral_residuals` maps back.

    ``kernel_fn(X, S, ctx) -> (N, M)`` takes the sorted nodes as two arrays of shape ``(N, 1)`` and ``(M, 1)`` and
    must be finite on the diagonal (return 0 there when the singular part is carried by ``diag_fn``). ``diag_fn
    (width, ctx) -> (N,)`` returns the analytic self-cell integral of the singular part for cell widths ``width``.
    """
    ctx = dict(ctx or {})
    xs = np.asarray(x, float).reshape(-1)
    order = np.argsort(xs)
    xsort = xs[order]
    w = trapezoid_weights(xsort)
    n = xsort.size

    xs_col = xsort[:, None]
    k = np.asarray(kernel_fn(xs_col, xs_col, ctx))
    if k.shape != (n, n):
        raise ValueError(f"kernel_fn must return an (N, N) matrix for {n} nodes, got shape {k.shape}")
    if not np.all(np.isfinite(k)):
        raise ValueError("kernel_fn returned a non-finite value: the singular part belongs in diag_fn, and the "
                         "diagonal of the numerically summed part must be finite (0 is the usual choice)")

    cw = _volterra_width(xsort) if volterra else w
    if diag_fn is not None:
        if volterra:
            # node 0 integrates over [x0, x0]: no self cell, and a zero width would put a nan through the kernel
            d = np.zeros(n, dtype=float)
            d[1:] = np.asarray(diag_fn(cw[1:], ctx)).reshape(-1)
        else:
            d = np.asarray(diag_fn(cw, ctx)).reshape(-1)
        if d.size != n:
            raise ValueError(f"diag_fn must return one value per node ({n}), got {d.size}")
        if not np.all(np.isfinite(d)):
            raise ValueError("diag_fn returned a non-finite value")
        d = d.astype(np.result_type(k.dtype, d.dtype))
    else:
        d = cw * np.diagonal(k)

    a = k * w[None, :]
    np.fill_diagonal(a, 0.0)
    if volterra:
        # only s <= x contributes; the diagonal came off and is replaced by the half cell
        a = np.tril(a, -1)
        d = 0.5 * d
    return a, d


def _volterra_width(xsort: np.ndarray) -> np.ndarray:
    """The width of the half cell that closes the integration for a Volterra row: the backward spacing at every
    node but the first, where the integral over ``[x_0, x_0]`` is empty and the width is zero.

    The Fredholm weight is not used here because it is already a half cell at the two ends, and halving it again
    would give a quarter cell at the last node instead of the half cell the Volterra rule needs."""
    n = xsort.size
    cw = np.zeros(n)
    cw[1:] = xsort[1:] - xsort[:-1]
    return cw


# ------------------------------------------------------------------------------------------------ kernels
def constant_kernel(X: np.ndarray, S: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
    """``K = 1``: the bounded kernel of the textbook Fredholm examples. Finite on the diagonal, so no ``diag_fn``."""
    return np.ones((len(X), len(S)))


def product_kernel(X: np.ndarray, S: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
    """``K(x, s) = x s``: the degenerate (rank-one) kernel, finite on the diagonal, so no ``diag_fn``."""
    return np.asarray(X, float)[:, 0:1] * np.asarray(S, float)[:, 0:1].T


def log_kernel(X: np.ndarray, S: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
    """``ln|x - s|`` with a zero diagonal: the singular part is carried by :func:`log_diag`."""
    r = _distance(X, S)
    k = np.zeros_like(r)
    off = ~np.eye(r.shape[0], r.shape[1], dtype=bool)
    k[off] = np.log(r[off])
    return k


def log_diag(width: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
    """``int_{-w/2}^{w/2} ln|s| ds = w (ln(w/2) - 1)``: the exact self-cell integral of :func:`log_kernel`."""
    w = np.asarray(width, float)
    return w * (np.log(0.5 * w) - 1.0)


def helmholtz2_kernel(X: np.ndarray, S: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
    """``(j/4) H_0^(2)(k r)`` with a zero diagonal: the two-dimensional Helmholtz kernel under ``exp(+j omega t)``
    (outgoing waves are ``H_0^(2)``, the convention of ``pinneapple_simulation.electromagnetics``). Reads ``k`` from
    ``ctx``; complex, so the equation needs two fields."""
    from scipy.special import hankel2

    k0 = float(ctx.get("k", 1.0))
    r = _distance(X, S)
    out = np.zeros(r.shape, complex)
    off = ~np.eye(r.shape[0], r.shape[1], dtype=bool)
    out[off] = 0.25j * hankel2(0, k0 * r[off])
    return out


def helmholtz2_diag(width: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
    """Self-cell integral of :func:`helmholtz2_kernel` from the small-argument expansion
    ``H_0^(2)(z) ~ 1 - (2j/pi)(ln(z/2) + gamma)``::

        int_{-w/2}^{w/2} (j/4) H_0^(2)(k|s|) ds ~ w [ j/4 + (ln(k w / 4) + gamma - 1) / (2 pi) ]

    The expansion is the ``k w << 1`` limit, which is the regime where the kernel is resolvable at all; the diagonal
    is the only place it is used and the trapezoid carries the rest. Raises when ``k w >= 1``.
    """
    w = np.asarray(width, float)
    k0 = float(ctx.get("k", 1.0))
    if np.any(k0 * w >= 1.0):
        raise ValueError(f"helmholtz2_diag: the small-argument expansion needs k * cell < 1, got max {np.max(k0 * w):.3g}")
    gamma = 0.5772156649015329
    return w * (0.25j + (np.log(0.25 * k0 * w) + gamma - 1.0) / (2.0 * np.pi))


def _distance(X: np.ndarray, S: np.ndarray) -> np.ndarray:
    """``(N, M)`` Euclidean distance between the rows of ``X`` and those of ``S``."""
    return np.linalg.norm(np.asarray(X, float)[:, None, :] - np.asarray(S, float)[None, :, :], axis=-1)


# ------------------------------------------------------------------------------------------------ residual
def integral_residuals(kind: str, *, fields: dict[str, torch.Tensor], xcol: torch.Tensor, coords: Sequence[str],
                       params: dict[str, Any], ctx: dict[str, Any], model: Callable, field_names: Sequence[str],
                       split_fields: Callable) -> list[torch.Tensor]:
    """Residuals of a second-kind integral equation at the collocation points, in the compiler's ``(N, 1)`` form.

    ``params``: ``lambda`` (alias ``lam``, default 1), and ``singular`` -- set it to ``True`` to demand a ``diag_fn``
    and refuse to run without one, since a singular kernel silently integrated through its diagonal produces
    plausible garbage. ``ctx`` (or the callables in ``params``): ``kernel_fn``, the optional ``diag_fn``, and
    ``f_fn`` (alias ``source_fn``) for the free term, which defaults to zero.
    """
    if kind not in INTEGRAL_KINDS:
        raise ValueError(f"unknown integral kind {kind!r} (expected one of {sorted(INTEGRAL_KINDS)})")
    p = params
    ctx = {**{k: v for k, v in p.items() if callable(v)}, **(ctx or {})}
    sp = _spatial(coords)
    if len(sp) != 1:
        raise ValueError(f"{kind} is written for one spatial coordinate, got coords {tuple(coords)}")
    kernel_fn = ctx.get("kernel_fn")
    if kernel_fn is None:
        raise ValueError(f"{kind} needs ctx['kernel_fn'] (or a callable in the PDE params)")
    diag_fn = ctx.get("diag_fn")
    if p.get("singular", False) and diag_fn is None:
        raise ValueError(f"{kind} is declared singular but no ctx['diag_fn'] was given: the cell integral of the "
                         "singular part must be supplied analytically, it cannot be integrated numerically")
    lam = float(p.get("lambda", p.get("lam", 1.0)))

    if len(field_names) == 2:
        return _residuals_pair(kind, fields, field_names, xcol, sp, ctx, kernel_fn, diag_fn, lam)
    if len(field_names) != 1:
        raise ValueError(f"{kind} expects one field (real kernel) or two fields Re/Im (complex kernel), "
                         f"got {len(field_names)}")
    u = fields[field_names[0]]
    f = _at(ctx, "f_fn", xcol, 0.0, u, "source_fn")

    order = torch.argsort(xcol.detach()[:, sp[0]])
    a, d = nystrom_matrix(xcol.detach()[:, sp[0]].cpu().numpy(), kernel_fn, ctx, diag_fn=diag_fn,
                          volterra=(kind == "volterra2"))
    if np.iscomplexobj(a):
        raise ValueError(f"{kind}: the kernel is complex, so the equation needs two fields (Re and Im of u); "
                         f"got the single field {field_names[0]!r}")
    at = torch.as_tensor(a, device=u.device, dtype=u.dtype)
    dt = torch.as_tensor(d, device=u.device, dtype=u.dtype)
    us = u[order]
    res_sorted = us - f[order] - lam * (at @ us + dt[:, None] * us)
    return [_unsort(res_sorted, order, u)]


def _residuals_pair(kind, fields, field_names, xcol, sp, ctx, kernel_fn, diag_fn, lam):
    """Complex kernel, complex field: ``u = ur + j ui`` and the residual is returned as its two real parts."""
    ur, ui = fields[field_names[0]], fields[field_names[1]]
    fr = _at(ctx, "f_fn", xcol, 0.0, ur, "source_fn")
    fi = _at(ctx, "f_i_fn", xcol, 0.0, ur, "source_i_fn")
    order = torch.argsort(xcol.detach()[:, sp[0]])
    a, d = nystrom_matrix(xcol.detach()[:, sp[0]].cpu().numpy(), kernel_fn, ctx, diag_fn=diag_fn,
                          volterra=(kind == "volterra2"))
    ar = torch.as_tensor(np.ascontiguousarray(a.real), device=ur.device, dtype=ur.dtype)
    ai = torch.as_tensor(np.ascontiguousarray(a.imag), device=ur.device, dtype=ur.dtype)
    dr = torch.as_tensor(np.ascontiguousarray(np.real(d)), device=ur.device, dtype=ur.dtype)
    di = torch.as_tensor(np.ascontiguousarray(np.imag(d)), device=ur.device, dtype=ur.dtype)
    urs, uis = ur[order], ui[order]
    kr = ar @ urs - ai @ uis + dr[:, None] * urs - di[:, None] * uis
    ki = ar @ uis + ai @ urs + dr[:, None] * uis + di[:, None] * urs
    return [_unsort(urs - fr[order] - lam * kr, order, ur),
            _unsort(uis - fi[order] - lam * ki, order, ur)]


def _unsort(res_sorted: torch.Tensor, order: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Back to the batch order: index ``i`` of the result belongs to the point the caller passed as row ``i``."""
    out = torch.empty_like(like)
    out[order] = res_sorted
    return out
