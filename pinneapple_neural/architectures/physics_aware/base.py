from __future__ import annotations
"""Base classes for physics-aware neural network models."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class PhysicsAwareOutput:
    y: torch.Tensor
    losses: dict[str, torch.Tensor]
    extras: dict[str, Any]


class PhysicsAwareBase(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean((y_hat - y) ** 2)
