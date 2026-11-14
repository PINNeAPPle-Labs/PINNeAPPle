"""Minimal DeepONet baseline for structural bracket stress prediction.

Requires PyTorch. Input tensors:
  case_features: [B, 31] (material, loads, family, geometry)
  point_features: [B, N, 8] (spatial and boundary features)
Returns von Mises stress predictions [B, N].
This is an architectural starting point, NOT a trained or validated model.
"""
from __future__ import annotations

import torch
from torch import nn


def mlp(input_dim: int, width: int = 128, depth: int = 4) -> nn.Sequential:
    layers: list[nn.Module] = []
    for _ in range(depth):
        layers.extend((nn.Linear(input_dim, width), nn.Tanh()))
        input_dim = width
    return nn.Sequential(*layers)


class BracketDeepONet(nn.Module):
    """Branch/trunk network with an elementwise fusion and nonlinear head."""

    def __init__(self) -> None:
        super().__init__()
        self.branch = mlp(31)
        self.trunk = mlp(8)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, case_features: torch.Tensor, point_features: torch.Tensor) -> torch.Tensor:
        if case_features.ndim != 2 or case_features.shape[-1] != 31:
            raise ValueError("case_features must have shape [B, 31]")
        if point_features.ndim != 3 or point_features.shape[-1] != 8:
            raise ValueError("point_features must have shape [B, N, 8]")
        if case_features.shape[0] != point_features.shape[0]:
            raise ValueError("Batch sizes must match")
        branch = self.branch(case_features).unsqueeze(1)
        trunk = self.trunk(point_features)
        return self.head(branch * trunk).squeeze(-1)


def relative_l2(prediction: torch.Tensor, reference: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if prediction.shape != reference.shape:
        raise ValueError("Prediction and reference shapes must match")
    diff = torch.linalg.vector_norm((prediction - reference).reshape(prediction.shape[0], -1), dim=1)
    norm = torch.linalg.vector_norm(reference.reshape(reference.shape[0], -1), dim=1)
    return (diff / norm.clamp_min(eps)).mean()


def stress_weighted_mse(
    prediction: torch.Tensor, reference: torch.Tensor, alpha: float = 0.7, eps: float = 1e-12
) -> torch.Tensor:
    """Emphasize high-stress regions; weights are detached and normalized per case."""
    if prediction.shape != reference.shape or prediction.ndim != 2:
        raise ValueError("Expected matching [B, N] tensors")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one")
    scale = reference.detach().abs().amax(dim=1, keepdim=True).clamp_min(eps)
    weights = (1 - alpha) + alpha * reference.detach().abs() / scale
    return (weights * (prediction - reference).square()).mean()


if __name__ == "__main__":
    model = BracketDeepONet()
    sample = model(torch.zeros(2, 31), torch.zeros(2, 16, 8))
    assert sample.shape == (2, 16)
    print("Output shape:", tuple(sample.shape))
    print("Trainable parameters:", sum(p.numel() for p in model.parameters()))
