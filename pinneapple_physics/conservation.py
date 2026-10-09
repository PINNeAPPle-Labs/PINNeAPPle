"""Conservation as a hard constraint on model outputs: project a predicted field so that its weighted integral
(total mass, energy, charge, water) equals a prescribed value exactly.

Neural surrogates trained with a conservation penalty only conserve approximately, and the error accumulates when a
surrogate is rolled out in time. A projection after the network removes the error by construction and stays
differentiable, so it can sit inside training:

* ``"additive"``: f + (T - <w, f>) / <w, 1>; the smallest correction in the w-weighted L2 norm (a uniform shift);
* ``"multiplicative"``: f * T / <w, f>; keeps the sign and the relative pattern (zero stays zero);
* ``"positive"``: clip negatives, then multiplicative; for fields that must be non-negative (density, humidity,
  concentrations), the standard global mass fixer of tracer transport.

``weights`` are the quadrature weights of the grid (cell areas or volumes, Gauss weights times radius^2 on the
sphere), so <w, f> is the discrete integral. :class:`ConservationProjection` wraps a model; the target comes from the
input (e.g. the mass of the previous state, plus sources).
"""
from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

__all__ = ["integral", "project_integral", "ConservationProjection", "MODES"]

MODES = ("additive", "multiplicative", "positive")
Tensor = torch.Tensor


def integral(field: Tensor, weights: Tensor, dims: int = 1) -> Tensor:
    """Weighted sum over the last ``dims`` dimensions (``weights`` has that trailing shape)."""
    axes = tuple(range(-dims, 0))
    return (field * weights).sum(dim=axes)


def project_integral(field: Tensor, weights: Tensor, target: float | Tensor, mode: str = "additive",
                     dims: int | None = None, eps: float = 1e-30) -> Tensor:
    """Return ``field`` corrected so that ``integral(field, weights) == target`` (per leading batch entry)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    weights = torch.as_tensor(weights, dtype=field.dtype, device=field.device)
    dims = weights.dim() if dims is None else dims
    target = torch.as_tensor(target, dtype=field.dtype, device=field.device)
    expand = (...,) + (None,) * dims
    if mode == "positive":
        field = torch.clamp(field, min=0.0)
    total = integral(field, weights, dims)
    if mode == "additive":
        return field + ((target - total) / weights.sum())[expand]
    if torch.any(total.abs() < eps):
        raise ValueError("multiplicative projection needs a non-zero integral; use mode='additive'")
    return field * (target / total)[expand]


class ConservationProjection(nn.Module):
    """``model`` followed by an exact projection. ``target_fn(x, y_pred)`` gives the conserved total for each batch
    entry (e.g. ``lambda x, y: integral(x_state, w)`` for a closed system); by default the integral of the input field
    ``x`` itself (same grid as the output)."""

    def __init__(self, model: nn.Module, weights: Tensor, mode: str = "additive",
                 target_fn: Callable[[Tensor, Tensor], Tensor] | None = None):
        super().__init__()
        self.model = model
        self.register_buffer("weights", torch.as_tensor(weights, dtype=torch.get_default_dtype()))
        self.mode = mode
        self.target_fn = target_fn

    def forward(self, x: Tensor) -> Tensor:
        y = self.model(x)
        w = self.weights.to(y.dtype)
        target = self.target_fn(x, y) if self.target_fn is not None else integral(x, w, w.dim())
        return project_integral(y, w, target, self.mode, w.dim())
