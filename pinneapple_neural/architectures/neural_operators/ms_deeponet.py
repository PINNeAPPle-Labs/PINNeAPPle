"""Multi-scale DeepONet: one trunk per spatial scale, each seeing the coordinates stretched by its own factor
(the MscaleDNN idea, Liu, Cai and Xu 2020), so high-frequency parts of the output are learned by trunks that see
them as low frequencies. The branch gives the coefficients of all scales."""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .base import NeuralOperatorBase, OperatorOutput
from .deeponet import mlp


class MultiScaleDeepONet(NeuralOperatorBase):
    """
        y(u, x) = sum_s < B_s(u), T_s(c_s x) > + bias

    ``scales``: number of modes of each trunk; ``scale_factors``: the coordinate stretch c_s of each trunk
    (default 1, 4, 16, ...: one trunk per octave pair).
    """
    def __init__(
        self,
        branch_dim: int,
        trunk_dim: int,
        out_dim: int,
        *,
        hidden: int | Sequence[int] = 128,
        scales: Sequence[int] = (32, 64, 128),
        scale_factors: Sequence[float] | None = None,
        depth: int = 1,
    ):
        super().__init__()
        self.scales = [int(s) for s in scales]
        self.scale_factors = [float(c) for c in (scale_factors or [4.0 ** k for k in range(len(self.scales))])]
        if len(self.scale_factors) != len(self.scales):
            raise ValueError("scale_factors must have one entry per scale")
        self.total_modes = int(sum(self.scales))
        self.out_dim = int(out_dim)
        self.branch = mlp(branch_dim, hidden, self.out_dim * self.total_modes, depth)
        self.trunks = nn.ModuleList([mlp(trunk_dim, hidden, s, depth) for s in self.scales])
        self.bias = nn.Parameter(torch.zeros(self.out_dim))

    def forward(
        self,
        u: torch.Tensor,        # (B, branch_dim)
        coords: torch.Tensor,   # (N, trunk_dim) or (B, N, trunk_dim)
        *,
        y_true: torch.Tensor | None = None,
        return_loss: bool = False,
    ) -> OperatorOutput:
        B = u.shape[0]
        b_all = self.branch(u).view(B, self.out_dim, self.total_modes)
        y = None
        start = 0
        for trunk, s, c in zip(self.trunks, self.scales, self.scale_factors, strict=True):
            t = trunk(c * coords)
            b_s = b_all[:, :, start:start + s]
            term = torch.einsum("bos,ns->bno", b_s, t) if t.dim() == 2 else torch.einsum("bos,bns->bno", b_s, t)
            y = term if y is None else y + term
            start += s
        y = y + self.bias

        losses: dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]
        return OperatorOutput(y=y, losses=losses, extras={})
