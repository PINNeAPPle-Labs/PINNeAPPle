from __future__ import annotations
"""Base classes for reduced-order models."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class ROMOutput:
    """
    Output for Reduced Order Models.

    y:
      - reconstructed field or predicted latent/state trajectory
    """
    y: torch.Tensor
    losses: dict[str, torch.Tensor]
    extras: dict[str, Any]


class ROMBase(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean((y_hat - y) ** 2)
