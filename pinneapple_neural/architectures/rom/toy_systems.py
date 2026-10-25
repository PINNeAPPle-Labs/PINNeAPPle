"""Synthetic linear dynamical systems for ROM demonstrations and tests (POD compression, DMD rollout)."""
from __future__ import annotations

import torch


def make_linear_system(D: int, seed: int = 0) -> torch.Tensor:
    """A contractive random linear operator ``A`` of shape ``(D, D)``: a Gaussian matrix scaled by ``1 / sqrt(D)``
    whose singular values are rescaled to a largest value of 0.95 (so ``x_{k+1} = A x_k`` decays)."""
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(D, D, generator=g) / (D**0.5)
    u, s, vh = torch.linalg.svd(M)
    s = 0.95 * (s / s.max())
    return (u * s) @ vh


def generate_sequences(B: int, T: int, D: int, A: torch.Tensor, noise: float = 0.01, seed: int = 123) -> torch.Tensor:
    """``B`` trajectories of ``T`` steps of ``x_{k+1} = A x_k + noise * N(0, I)`` from Gaussian initial states in
    ``D`` dimensions, shape ``(B, T, D)``; deterministic for a given ``seed``."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, D, generator=g)
    xs = [x]
    for _ in range(T - 1):
        x = x @ A.t() + noise * torch.randn_like(x, generator=g)
        xs.append(x)
    return torch.stack(xs, dim=1)
