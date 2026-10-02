from __future__ import annotations
"""Native Transolver (Physics-Attention) for point clouds / unstructured meshes.

Two variants, both pure PyTorch (no optional dependency, no research-only licence --
unlike ``noether_bridge``):

``Transolver``
    Physics-Attention as described by Wu et al., "Transolver: A Fast Transformer Solver
    for PDEs on General Geometries", ICML 2024 (arXiv:2402.02366). Per head, every point is
    softly assigned to M "physical states" (slices) with a temperature-scaled softmax; slice
    tokens are the *weighted means* of their points, attention runs among the M tokens, and
    the result is broadcast back with the same weights. Cost O(N*M + M^2) instead of O(N^2).

``TransolverLite``
    The simplified slice-attention block popularised in a public weekend-project write-up
    (single softmax assignment, weighted *sums* instead of means, ``nn.MultiheadAttention``
    over slices). Kept so that reproduction studies can run the exact variant they compare
    against.

Both map point features ``(B, N, in_dim)`` to point outputs ``(B, N, out_dim)`` and are
invariant to node ordering (no connectivity is used).
"""
from typing import Optional

import torch
import torch.nn as nn

from .base import NeuralOperatorBase, OperatorOutput


class PhysicsAttention(nn.Module):
    """Multi-head Physics-Attention (slice -> attend -> deslice)."""

    def __init__(self, dim: int, heads: int = 8, slices: int = 32, dropout: float = 0.0):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.h, self.dh, self.m = heads, dim // heads, slices
        self.in_x = nn.Linear(dim, dim)
        self.in_fx = nn.Linear(dim, dim)
        self.to_slice = nn.Linear(self.dh, slices)
        nn.init.orthogonal_(self.to_slice.weight)
        self.temperature = nn.Parameter(torch.full((1, heads, 1, 1), 0.5))
        self.q = nn.Linear(self.dh, self.dh, bias=False)
        self.k = nn.Linear(self.dh, self.dh, bias=False)
        self.v = nn.Linear(self.dh, self.dh, bias=False)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        B, N, _ = x.shape
        fx = self.in_fx(x).view(B, N, self.h, self.dh).transpose(1, 2)          # B h N dh
        xm = self.in_x(x).view(B, N, self.h, self.dh).transpose(1, 2)
        w = torch.softmax(self.to_slice(xm) / self.temperature.clamp(min=0.05), dim=-1)  # B h N M
        norm = w.sum(2)                                                          # B h M
        tok = torch.einsum("bhnm,bhnd->bhmd", w, fx) / (norm[..., None] + 1e-5)  # weighted mean per slice
        q, k, v = self.q(tok), self.k(tok), self.v(tok)
        att = self.drop(torch.softmax(q @ k.transpose(-1, -2) * self.dh ** -0.5, dim=-1))
        tok = att @ v                                                            # B h M dh
        y = torch.einsum("bhmd,bhnm->bhnd", tok, w).transpose(1, 2).reshape(B, N, -1)
        y = self.out(y)
        return (y, w) if return_weights else y


class TransolverBlock(nn.Module):
    def __init__(self, dim, heads, slices, mlp_ratio=2, dropout=0.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = PhysicsAttention(dim, heads, slices, dropout)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class Transolver(NeuralOperatorBase):
    """Point-cloud operator: (B, N, in_dim) -> (B, N, out_dim)."""

    def __init__(self, in_dim: int, out_dim: int, dim: int = 128, depth: int = 4, heads: int = 8,
                 slices: int = 32, mlp_ratio: int = 2, dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(in_dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.placeholder = nn.Parameter(torch.randn(dim) / dim)
        self.blocks = nn.ModuleList([TransolverBlock(dim, heads, slices, mlp_ratio, dropout) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, out_dim))

    def forward(self, x: torch.Tensor, *, y_true: Optional[torch.Tensor] = None, return_loss: bool = False) -> OperatorOutput:
        h = self.embed(x) + self.placeholder
        for blk in self.blocks:
            h = blk(h)
        y = self.head(h)
        losses = {}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]
        return OperatorOutput(y=y, losses=losses, extras={})


class _LiteBlock(nn.Module):
    """Slice-attention block exactly as in the reproduced write-up."""

    def __init__(self, d_model=64, n_heads=4, n_slices=8, dropout=0.1):
        super().__init__()
        self.slice_proj = nn.Linear(d_model, n_slices)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(d_model * 4, d_model))

    def forward(self, x):
        w = torch.softmax(self.slice_proj(x), dim=-1)
        xs = torch.einsum("bns,bnd->bsd", w, x)
        a, _ = self.attn(xs, xs, xs)
        xs = self.norm1(xs + a)
        x = self.norm2(x + torch.einsum("bns,bsd->bnd", w, xs))
        return x + self.ff(x)


class TransolverLite(NeuralOperatorBase):
    """Linear input projection -> ``n_blocks`` slice-attention blocks -> per-node linear head."""

    def __init__(self, in_dim: int = 5, out_dim: int = 101, d_model: int = 64, n_heads: int = 4,
                 n_slices: int = 8, n_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList([_LiteBlock(d_model, n_heads, n_slices, dropout) for _ in range(n_blocks)])
        self.out = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor, *, y_true: Optional[torch.Tensor] = None, return_loss: bool = False) -> OperatorOutput:
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        y = self.out(h)
        losses = {}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]
        return OperatorOutput(y=y, losses=losses, extras={})


__all__ = ["PhysicsAttention", "Transolver", "TransolverLite"]
