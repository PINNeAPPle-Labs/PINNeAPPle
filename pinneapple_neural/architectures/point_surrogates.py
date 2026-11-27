"""Point-cloud surrogates for a field defined over a surface: PointNet, its wide residual variant, and an
EdgeConv (DGCNN-style) model.

These are the point-based family of a surface-field benchmark (surface pressure over a car body, displacement over a
crash structure): the input is the geometry alone — a point set with optional per-point features — and the output is
a value at every input point, so the same weights predict on any geometry and any point count.

Written from the papers:

- PointNet: Qi et al., "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation", CVPR 2017
  (arXiv:1612.00593) — shared per-point MLP, a symmetric (max) pooling to a global feature, the global feature
  concatenated back to every point, a final per-point MLP;
- the wide residual variant with a 4096-d embedding stands in for the "PointNetLarge" row of the vehicle-aerodynamics
  benchmark (a wider PointNet trained with the same protocol, not the published weights);
- EdgeConv: Wang et al., "Dynamic Graph CNN for Learning on Point Sets", ACM TOG 2019 (arXiv:1801.07829) — a
  k-nearest-neighbour graph rebuilt from the features at every layer, edge features from the difference of the
  point and its neighbours, max-pooled per point.

    from pinneapple_neural.architectures.point_surrogates import PointNetSurrogate
    model = PointNetSurrogate(in_dim=3, out_dim=1, dim=1024, residual=False)   # (B, N, 3) -> (B, N, 1)
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["PointNetSurrogate", "EdgeConvSurrogate", "PartPointNetSurrogate"]


def _mlp(in_dim: int, hidden: int, out_dim: int, depth: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for _ in range(depth):
        layers += [nn.Linear(last, hidden), nn.GELU()]
        last = hidden
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class PointNetSurrogate(nn.Module):
    """Shared point MLP -> global max-pool -> per-point decode (PointNet's segmentation head).

    ``dim`` is the width of the shared and global features (1024 for PointNet, 4096 for the wide variant);
    ``residual=True`` adds residual blocks per point, the wider variant trained in the vehicle benchmark;
    ``use_normals``/``extra_dim`` add per-point input features (surface normals, thickness, ...) after the
    coordinates."""

    def __init__(self, in_dim: int = 3, out_dim: int = 1, dim: int = 1024, residual: bool = False,
                 depth: int = 3, dropout: float = 0.3, global_depth: int = 3):
        super().__init__()
        self.in_dim, self.out_dim, self.dim = in_dim, out_dim, dim
        self.embed = _mlp(in_dim, dim, dim, depth=2)
        self.per_point = nn.ModuleList([self._block(dim, dropout) for _ in range(depth)])
        self.residual = residual
        self.global_mlp = _mlp(dim, 2 * dim if residual else dim, dim, depth=1)
        self.head = _mlp(2 * dim, dim, out_dim, depth=2)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _block(dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, **_: torch.Tensor) -> torch.Tensor:
        """``x`` (B, N, in_dim) with an optional ``mask`` (B, N) of valid points -> (B, N, out_dim)."""
        h = self.embed(x)
        for blk in self.per_point:
            h = h + self.drop(blk(h)) if self.residual else blk(h)
        if mask is not None:
            h = h * mask.unsqueeze(-1).to(h.dtype)
            pooled = h.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(dim=1)
        else:
            pooled = h.amax(dim=1)
        g = self.global_mlp(pooled).unsqueeze(1).expand(-1, h.shape[1], -1)
        return self.head(torch.cat([h, g], dim=-1))


class EdgeConvSurrogate(PointNetSurrogate):
    """EdgeConv blocks on a k-nearest-neighbour graph (DGCNN-style), then the global pool and per-point decode.

    The width grows from ``dim // 8`` to ``dim`` over the ``blocks`` edge blocks (as the reference does), the graph
    is rebuilt from the current features when ``dynamic=True``, and the neighbours are taken in the ``in_dim``
    coordinates otherwise. Costs ``B * N * k * width`` per layer, so keep ``N`` and ``k`` modest."""

    def __init__(self, in_dim: int = 3, out_dim: int = 1, dim: int = 1024, k: int = 16, blocks: int = 3,
                 dynamic: bool = True, dropout: float = 0.4, depth: int = 1):
        super().__init__(in_dim=in_dim, out_dim=out_dim, dim=dim, residual=False, depth=depth, dropout=dropout)
        if blocks < 1:
            raise ValueError("blocks must be >= 1")
        self.k, self.dynamic, self.blocks = int(k), bool(dynamic), int(blocks)
        widths = [max(32, dim // (2 ** (blocks - 1 - i))) for i in range(blocks)]     # dim//4, dim//2, dim
        self.widths = widths
        self.edge = nn.ModuleList([_mlp(2 * (in_dim if i == 0 else widths[i - 1]), widths[i], widths[i], depth=2)
                                   for i in range(blocks)])
        self.drop_edges = nn.Dropout(dropout)
        self.fuse = nn.Linear(widths[-1], dim) if widths[-1] != dim else nn.Identity()
        self.head = _mlp(2 * dim, dim, out_dim, depth=2)

    def _neighbours(self, feats: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Indices (B, N, k) of the nearest neighbours, by the features when ``dynamic`` else by the coordinates."""
        src = feats if self.dynamic else coords
        k = min(self.k, src.shape[1])
        return torch.cdist(src, src).topk(k, dim=-1, largest=False).indices

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, **_: torch.Tensor) -> torch.Tensor:
        h = x
        for i, theta in enumerate(self.edge):
            idx = self._neighbours(h if i else x, x[..., :3])                   # (B, N, k)
            b, n, c = h.shape
            nbr = torch.gather(h, 1, idx.reshape(b, n * idx.shape[-1]).unsqueeze(-1).expand(-1, -1, c)) \
                .reshape(b, n, idx.shape[-1], c)                                  # (B, N, k, C)
            edge = torch.cat([h.unsqueeze(2).expand_as(nbr), nbr - h.unsqueeze(2)], dim=-1)
            out = theta(edge).amax(dim=2)                                       # (B, N, width_i)
            h = h + self.drop_edges(out) if (i and out.shape[-1] == h.shape[-1]) else out
        h = self.fuse(h)
        if mask is not None:
            h = h * mask.unsqueeze(-1).to(h.dtype)
            pooled = h.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(dim=1)
        else:
            pooled = h.amax(dim=1)
        g = self.global_mlp(pooled).unsqueeze(1).expand(-1, h.shape[1], -1)
        return self.head(torch.cat([h, g], dim=-1))


class PartPointNetSurrogate(PointNetSurrogate):
    """PointNet with a learned embedding per semantic part, added to the per-point features.

    The part label is the one thing a crash or a car mesh has that a plain point cloud does not; adding it as a
    per-point input is the cheapest way to let a point model use it (the hierarchical way is
    :class:`pinneapple_neural.architectures.crash_solver.CrashSolver`)."""

    def __init__(self, in_dim: int = 3, out_dim: int = 1, dim: int = 1024, n_parts: int = 8,
                 part_dim: int = 32, residual: bool = False, depth: int = 3, dropout: float = 0.3):
        super().__init__(in_dim=in_dim, out_dim=out_dim, dim=dim, residual=residual, depth=depth, dropout=dropout)
        self.n_parts = int(n_parts)
        self.part = nn.Embedding(n_parts + 1, part_dim)                    # +1: padding / unknown part
        self.embed = _mlp(in_dim + part_dim, dim, dim, depth=2)

    def forward(self, x: torch.Tensor, part: torch.Tensor | None = None, mask: torch.Tensor | None = None,
                **_: torch.Tensor) -> torch.Tensor:
        if part is None:
            part = torch.zeros(x.shape[:2], dtype=torch.long, device=x.device)
        x = torch.cat([x, self.part(part.clamp(0, self.n_parts))], dim=-1)
        return super().forward(x, mask=mask)
