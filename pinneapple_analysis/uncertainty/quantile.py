"""Quantile (distribution-free) uncertainty estimation.

Provides pinball-loss training for quantile regression models and a general
QuantileHead that wraps any point-prediction model.

Unlike conformal prediction, quantile regression *learns* conditional quantiles
from data rather than relying on post-hoc calibration.  Use
:class:`~pinneapple_uq.conformal.ConformalPredictor` when distribution-free
coverage guarantees are required.

Quick start::

    from pinneapple_analysis.uncertainty import QuantileHead, QuantileLoss, QuantileConfig

    cfg  = QuantileConfig(quantiles=(0.1, 0.5, 0.9))
    head = QuantileHead(base_model, cfg)
    loss = QuantileLoss(cfg.quantiles)

    for x_batch, y_batch in loader:
        y_pred = head(x_batch)          # (B, H, Q)
        metrics = loss(head, y_pred, {"y": y_batch})
        metrics["total"].backward()
        ...
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import torch.nn as nn


def pinball_loss_torch(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    q: float,
) -> torch.Tensor:
    """Pinball (quantile) loss for a single quantile level *q*.

    Parameters
    ----------
    y_pred : Tensor — predicted quantile, shape ``(B, H)`` or ``(N,)``.
    y_true : Tensor — targets, same shape as *y_pred*.
    q : float — quantile level in ``(0, 1)``.

    Returns
    -------
    Tensor — scalar mean pinball loss.
    """
    q = float(q)
    if not 0.0 < q < 1.0:
        raise ValueError(f"quantile level q must be in (0, 1), got {q}")
    if y_pred.shape != y_true.shape:
        # broadcasting a (B, 1) target against (B, H) predictions would silently score the wrong pairs
        raise ValueError(f"y_pred {tuple(y_pred.shape)} and y_true {tuple(y_true.shape)} must have the same shape")
    e = y_true - y_pred
    return torch.mean(torch.maximum(q * e, (q - 1.0) * e))


def crossing_rate(y_pred: torch.Tensor, quantiles: Sequence[float]) -> float:
    """Fraction of predictions whose quantiles are out of order.

    Parameters
    ----------
    y_pred : Tensor — ``(..., Q)``, one column per level in *quantiles*.
    quantiles : the ``Q`` levels, in the order of the last axis (any order).

    Returns
    -------
    float in ``[0, 1]``: the share of points where a higher level is predicted below a lower one.
    """
    levels = torch.as_tensor(list(quantiles), dtype=torch.float64)
    if y_pred.shape[-1] != len(levels):
        raise ValueError(f"y_pred has {y_pred.shape[-1]} columns for {len(levels)} quantile levels")
    ordered = y_pred.detach()[..., torch.argsort(levels)]
    return float((ordered[..., 1:] < ordered[..., :-1]).any(dim=-1).double().mean())


@dataclass
class QuantileConfig:
    """Configuration for quantile prediction.

    Parameters
    ----------
    quantiles : sequence of floats in ``(0, 1)``.
        Default ``(0.1, 0.5, 0.9)`` gives 80 % prediction interval + median.
    """
    quantiles: Sequence[float] = (0.1, 0.5, 0.9)

    def __post_init__(self) -> None:
        _check_levels(self.quantiles)


def _check_levels(quantiles: Sequence[float]) -> None:
    qs = [float(q) for q in quantiles]
    if not qs:
        raise ValueError("at least one quantile level is needed")
    if any(not 0.0 < q < 1.0 for q in qs):
        raise ValueError(f"quantile levels must be in (0, 1), got {qs}")
    if len(set(qs)) != len(qs):
        raise ValueError(f"quantile levels must be distinct, got {qs}")


class QuantileHead(nn.Module):
    """Wrap a point-prediction model and project outputs to multiple quantiles.

    Assumes ``base(x)`` returns ``(B, H)`` or ``(B, H, 1)``.
    Output shape: ``(B, H, Q)`` where *Q* = ``len(cfg.quantiles)``.

    Parameters
    ----------
    base : nn.Module — backbone forecaster (any architecture).
    cfg : QuantileConfig — quantile levels.
    hidden_dim : int — width of the projection head.
    rearrange : bool — sort the ``Q`` outputs at every point so that a higher level is never predicted below a
        lower one (Chernozhukov, Fernandez-Val and Galichon 2010; the sorted estimate is never farther from the
        true quantile function). Default ``True``; ``False`` returns the raw outputs, which can cross,
        especially early in training and away from the data. :func:`crossing_rate` measures it.

    Notes
    -----
    The head sees only the point prediction ``base(x)``, not ``x``, so the width of the interval depends on ``x``
    only through that prediction.
    """

    def __init__(
        self,
        base: nn.Module,
        cfg: QuantileConfig,
        hidden_dim: int = 64,
        rearrange: bool = True,
    ) -> None:
        super().__init__()
        self.base = base
        self.cfg = cfg
        self.rearrange = rearrange
        self.proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(cfg.quantiles)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return quantile predictions, shape ``(B, H, Q)``."""
        y_point = self.base(x)
        if hasattr(y_point, "y"):
            # Unwrap ModelOutput/PINNOutput-style wrapper dataclasses.
            y_point = y_point.y
        if y_point.ndim == 2:
            y_point = y_point.unsqueeze(-1)  # (B, H, 1)
        if y_point.shape[-1] != 1:
            raise ValueError("QuantileHead expects single-target base output (C=1).")
        B, H, _ = y_point.shape
        out = self.proj(y_point.reshape(B * H, 1)).reshape(B, H, -1)
        if self.rearrange:
            order = torch.argsort(torch.as_tensor(list(self.cfg.quantiles)))   # levels need not be given sorted
            out = torch.sort(out, dim=-1).values[..., torch.argsort(order)]
        return out


class QuantileLoss:
    """Multi-quantile pinball loss compatible with the Pinneapple Trainer.

    Parameters
    ----------
    quantiles : sequence of floats.
    """

    def __init__(self, quantiles: Sequence[float]) -> None:
        _check_levels(quantiles)
        self.quantiles = list(quantiles)

    def __call__(
        self,
        model: nn.Module,
        y_pred: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute averaged pinball loss across all quantile levels.

        Parameters
        ----------
        model : nn.Module — unused (required for Trainer compatibility).
        y_pred : Tensor — ``(B, H, Q)``.
        batch : dict — must contain key ``"y"`` with target ``(B, H)`` or ``(B, H, 1)``.

        Returns
        -------
        dict with ``"total"`` key for Trainer.
        """
        y_true = batch["y"]
        if y_true.ndim == 3:
            if y_true.shape[-1] != 1:
                raise ValueError("QuantileLoss expects single-target y with C=1.")
            y_true = y_true[..., 0]  # (B, H)
        elif y_true.ndim != 2:
            raise ValueError(f"Unexpected y shape: {tuple(y_true.shape)}")

        if y_pred.ndim != 3:
            raise ValueError(f"Expected y_pred (B, H, Q), got {tuple(y_pred.shape)}")
        if y_pred.shape[-1] != len(self.quantiles):
            raise ValueError(f"y_pred has {y_pred.shape[-1]} quantile outputs for {len(self.quantiles)} levels "
                             f"{self.quantiles}")

        total: torch.Tensor = sum(  # type: ignore[assignment]
            pinball_loss_torch(y_pred[:, :, qi], y_true, q)
            for qi, q in enumerate(self.quantiles)
        )
        total = total / max(1, len(self.quantiles))
        return {"total": total}
