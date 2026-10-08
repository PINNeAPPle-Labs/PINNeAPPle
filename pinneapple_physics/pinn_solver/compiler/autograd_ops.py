from __future__ import annotations

import contextlib
import contextvars
from typing import Dict, Iterator, Optional, Tuple

import torch

# Derivative cache. Inside ``derivative_cache()`` a repeated ``grad(y, x)`` / ``jacobian(Y, x)`` on the same
# tensors returns the first result instead of building another autograd graph. The cache keeps references to
# ``y`` and ``x`` so their ids cannot be reused while it is alive, and it is dropped on exit.
_CACHE: contextvars.ContextVar[Optional[Dict[Tuple[str, int, int], tuple]]] = contextvars.ContextVar(
    "pinneapple_derivative_cache", default=None)
_STATS = {"hits": 0, "misses": 0}


@contextlib.contextmanager
def derivative_cache() -> Iterator[Dict[str, int]]:
    """Memoize first derivatives within the block (one loss evaluation). Yields hit/miss counters."""
    token = _CACHE.set({})
    _STATS["hits"] = _STATS["misses"] = 0
    try:
        yield _STATS
    finally:
        _CACHE.reset(token)


def _cached(kind: str, y: torch.Tensor, x: torch.Tensor, compute):
    cache = _CACHE.get()
    if cache is None:
        return compute()
    key = (kind, id(y), id(x))
    hit = cache.get(key)
    if hit is not None:
        _STATS["hits"] += 1
        return hit[2]
    _STATS["misses"] += 1
    out = compute()
    cache[key] = (y, x, out)
    return out


def ensure_tensor(y):
    if hasattr(y, "y"):
        return y.y
    return y


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return _cached("grad", y, x, lambda: torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0])


def jacobian(Y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    assert Y.ndim == 2
    return _cached("jac", Y, x, lambda: _jacobian(Y, x))


def _jacobian(Y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    outs = []
    for j in range(Y.shape[1]):
        g = torch.autograd.grad(
            outputs=Y[:, j:j + 1],
            inputs=x,
            grad_outputs=torch.ones_like(Y[:, j:j + 1]),
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        outs.append(g[:, None, :])
    return torch.cat(outs, dim=1)


def divergence(
    V: torch.Tensor,
    x: torch.Tensor,
    coord_indices: list[int] | None = None,
) -> torch.Tensor:
    """Divergence sum_i dV_i/dx_{idx_i} of a vector field V.

    Parameters
    ----------
    V : (N, K) tensor – vector field components.
    x : (N, D) tensor – input coordinates (must have grad enabled).
    coord_indices : optional list of K column indices into `x`, one per
        component of V, to differentiate against. When *None* (default)
        V and x must have the same number of columns and each component
        is paired with the coordinate of the same index. Pass the
        spatial column indices (excluding time) to obtain the spatial
        divergence of a time-dependent vector field.
    """
    indices = coord_indices if coord_indices is not None else list(range(x.shape[1]))
    assert V.shape[1] == len(indices)
    J = jacobian(V, x)
    div = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
    for i, idx in enumerate(indices):
        div = div + J[:, i:i + 1, idx:idx + 1].reshape(-1, 1)
    return div


def laplacian(
    y: torch.Tensor,
    x: torch.Tensor,
    coord_indices: list[int] | None = None,
) -> torch.Tensor:
    """Sum of unmixed second derivatives ∂²y/∂x_i².

    Parameters
    ----------
    y : (N, 1) tensor – scalar field values.
    x : (N, D) tensor – input coordinates (must have grad enabled).
    coord_indices : optional list of column indices to include.
        When *None* (default) all D columns are summed, giving the
        full Laplacian.  Pass only the spatial indices to obtain the
        **spatial** Laplacian (excluding the time coordinate).  For
        example ``coord_indices=[1]`` for a 2-D input ``(t, x)``
        computes ∂²y/∂x² only.
    """
    g = grad(y, x)
    indices = coord_indices if coord_indices is not None else list(range(x.shape[1]))
    out = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
    for i in indices:
        gi = g[:, i:i + 1]
        gii = torch.autograd.grad(
            outputs=gi,
            inputs=x,
            grad_outputs=torch.ones_like(gi),
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0][:, i:i + 1]
        out = out + gii
    return out


def time_derivative(y: torch.Tensor, x: torch.Tensor, t_index: int) -> torch.Tensor:
    g = grad(y, x)
    return g[:, t_index:t_index + 1]


def norm_dot_grad(y: torch.Tensor, x: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    g = grad(y, x)
    return torch.sum(g * normals, dim=1, keepdim=True)


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a - b) ** 2)