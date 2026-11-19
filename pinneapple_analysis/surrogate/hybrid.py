"""Composable DeepONet encoders: MLP, Fourier-feature MLP and RBF-KAN.

The RBF-KAN option follows the edge-wise learnable univariate basis idea of KAN,
using Gaussian radial bases (FastKAN-style), not exact B-spline KAN.
Branch and trunk can be selected independently. The inherited DeepONet forward
and OperatorOutput contracts remain unchanged.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from pinneapple_neural.architectures.neural_operators.deeponet import DeepONet, mlp

ENCODERS = ("mlp", "rbf_kan", "fourier")


class RBFKANLayer(nn.Module):
    """Per-edge Gaussian basis expansion plus a trainable linear residual."""

    def __init__(self, in_features: int, out_features: int, grid_size: int = 8,
                 grid_min: float = -3.0, grid_max: float = 3.0):
        super().__init__()
        if min(in_features, out_features, grid_size) < 1 or grid_max <= grid_min:
            raise ValueError("invalid RBF-KAN layer dimensions or grid")
        self.register_buffer("grid", torch.linspace(grid_min, grid_max, grid_size))
        self.scale = (grid_max - grid_min) / max(1, grid_size - 1)
        self.weights = nn.Parameter(torch.empty(out_features, in_features, grid_size))
        nn.init.normal_(self.weights, std=0.02 / math.sqrt(in_features * grid_size))
        self.base = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = torch.exp(-((x.unsqueeze(-1) - self.grid) / self.scale).square())
        return torch.einsum("...ig,oig->...o", basis, self.weights) + self.base(torch.nn.functional.silu(x))


class RBFKAN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64,
                 depth: int = 2, grid_size: int = 8):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least one")
        blocks: list[nn.Module] = []
        width = in_dim
        for _ in range(depth):
            blocks.append(RBFKANLayer(width, hidden, grid_size=grid_size))
            width = hidden
        blocks.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FourierFeatures(nn.Module):
    """Deterministic frequencies are registered buffers, not untracked RNG state."""

    def __init__(self, in_dim: int, features: int = 16, scale: float = 2.0):
        super().__init__()
        if features < 1 or scale <= 0:
            raise ValueError("features and scale must be positive")
        freq = torch.arange(1, features + 1, dtype=torch.float32) * scale
        self.register_buffer("frequencies", freq)
        self.in_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phase = x.unsqueeze(-1) * self.frequencies
        return torch.cat([x, phase.sin().flatten(-2), phase.cos().flatten(-2)], dim=-1)


def encoder(kind: str, in_dim: int, out_dim: int, *, hidden: int = 64,
            depth: int = 3, grid_size: int = 8, fourier_features: int = 16) -> nn.Module:
    if kind == "mlp":
        return mlp(in_dim, hidden, out_dim, depth=depth)
    if kind == "rbf_kan":
        return RBFKAN(in_dim, out_dim, hidden, depth=depth, grid_size=grid_size)
    if kind == "fourier":
        features = FourierFeatures(in_dim, fourier_features)
        return nn.Sequential(features, mlp(in_dim * (2 * fourier_features + 1), hidden, out_dim, depth=depth))
    raise ValueError(f"unknown encoder {kind!r}; choose one of {ENCODERS}")


class HybridDeepONet(DeepONet):
    """DeepONet with independently configurable branch and trunk representations."""

    def __init__(self, branch_dim: int, trunk_dim: int, out_dim: int = 1, *,
                 branch: str = "rbf_kan", trunk: str = "mlp", hidden: int = 64,
                 modes: int = 32, depth: int = 3, grid_size: int = 8,
                 fourier_features: int = 16):
        if modes < 1:
            raise ValueError("modes must be positive")
        super().__init__(branch_dim, trunk_dim, out_dim, hidden=hidden, modes=modes, depth=depth)
        options = dict(hidden=hidden, depth=depth, grid_size=grid_size, fourier_features=fourier_features)
        self.branch = encoder(branch, branch_dim, out_dim * modes, **options)
        self.trunk = encoder(trunk, trunk_dim, modes, **options)
        self.architecture = {"branch": branch, "trunk": trunk, "hidden": hidden, "depth": depth,
                             "modes": modes, "grid_size": grid_size, "fourier_features": fourier_features}
