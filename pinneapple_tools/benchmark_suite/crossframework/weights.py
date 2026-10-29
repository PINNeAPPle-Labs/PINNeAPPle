"""Shared initial weights for the identical-configuration track (A1).

The network of A1 is the same for every framework: a fully connected tanh network ``in -> 50 x 4 -> out``, Xavier
(Glorot) uniform weights, zero biases, float32. To remove the initialisation lottery from the comparison, the weights
of seed ``s`` are drawn once here and every framework loads the same arrays into its own network
(:func:`load_into_linears`), so seed ``s`` starts from the same function in every framework.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["layer_sizes", "xavier_uniform_init", "load_into_linears", "count_parameters"]


def layer_sizes(in_dim: int = 2, out_dim: int = 1, width: int = 50, depth: int = 4) -> list[int]:
    return [in_dim] + [width] * depth + [out_dim]


def xavier_uniform_init(sizes: Sequence[int], seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """``[(W, b), ...]`` with ``W`` of shape ``(fan_out, fan_in)`` (the ``torch.nn.Linear`` layout) drawn from
    ``U(-a, a)``, ``a = sqrt(6 / (fan_in + fan_out))`` (Glorot & Bengio 2010), and ``b = 0``; float32."""
    rng = np.random.default_rng(10_000 + int(seed))
    out = []
    for fi, fo in zip(sizes[:-1], sizes[1:], strict=False):
        a = np.sqrt(6.0 / (fi + fo))
        out.append((rng.uniform(-a, a, size=(fo, fi)).astype(np.float32), np.zeros(fo, dtype=np.float32)))
    return out


def load_into_linears(linears, params: Sequence[tuple[np.ndarray, np.ndarray]]) -> None:
    """Copy ``params`` into a sequence of ``torch.nn.Linear`` modules (shapes must match exactly)."""
    import torch
    linears = list(linears)
    if len(linears) != len(params):
        raise ValueError(f"{len(linears)} linear layers but {len(params)} weight pairs")
    with torch.no_grad():
        for lin, (W, b) in zip(linears, params, strict=False):
            if tuple(lin.weight.shape) != W.shape or tuple(lin.bias.shape) != b.shape:
                raise ValueError(f"shape mismatch: layer {tuple(lin.weight.shape)} vs weights {W.shape}")
            lin.weight.copy_(torch.from_numpy(W))
            lin.bias.copy_(torch.from_numpy(b))


def count_parameters(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))
