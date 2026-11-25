"""Base TSOutput and TSModelBase for time series models."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn


@dataclass
class TSOutput:
    """Standard output for time series model predictions."""

    y_hat: torch.Tensor
    extras: dict[str, Any]


class TSModelBase(nn.Module):
    """
    Base opcional para modelos TS.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
