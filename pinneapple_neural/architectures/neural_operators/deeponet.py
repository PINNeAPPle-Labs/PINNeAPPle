"""DeepONet operator learning with branch-trunk architecture (Lu et al., Nat. Mach. Intell. 3, 2021)."""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .base import NeuralOperatorBase, OperatorOutput


def mlp(in_dim: int, hidden: int | Sequence[int], out_dim: int, depth: int = 1,
        act: type = nn.GELU, last_act: bool = False) -> nn.Sequential:
    """MLP with hidden widths ``hidden`` (a list) or ``depth`` layers of width ``hidden`` (an int)."""
    widths = list(hidden) if isinstance(hidden, (list, tuple)) else [int(hidden)] * max(1, int(depth))
    layers, d = [], in_dim
    for w in widths:
        layers += [nn.Linear(d, w), act()]
        d = w
    layers.append(nn.Linear(d, out_dim))
    if last_act:
        layers.append(act())
    return nn.Sequential(*layers)


class DeepONet(NeuralOperatorBase):
    """
    Unstacked DeepONet, one set of branch coefficients per output channel:

        y_o(u, x) = sum_{k=1..modes} B_{o,k}(u) T_k(x) + b_o

    Branch net: the input function at fixed sensors, u (B, branch_dim).
    Trunk net: the query coordinates, x (N, trunk_dim) shared by the batch or (B, N, trunk_dim) per sample.

    ``hidden``: width of the hidden layers (int, with ``depth`` layers) or the list of widths. The original paper
    uses three to four hidden layers in each net; ``depth`` defaults to 1 only to keep earlier checkpoints loadable.
    ``trunk_activation``: also apply the activation after the trunk's last layer (as in the paper's code).
    """
    def __init__(
        self,
        branch_dim: int,
        trunk_dim: int,
        out_dim: int,
        hidden: int | Sequence[int] = 128,
        modes: int = 64,
        depth: int = 1,
        trunk_activation: bool = False,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.modes = modes
        self.branch = mlp(branch_dim, hidden, out_dim * modes, depth)
        self.trunk = mlp(trunk_dim, hidden, modes, depth, last_act=trunk_activation)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def _contract(self, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2:                                   # shared query points
            return torch.einsum("bom,nm->bno", b, t)
        return torch.einsum("bom,bnm->bno", b, t)          # per-sample query points

    def forward(
        self,
        u: torch.Tensor,        # (B, branch_dim)
        coords: torch.Tensor,   # (N, trunk_dim) or (B, N, trunk_dim)
        *,
        y_true: torch.Tensor | None = None,
        return_loss: bool = False,
    ) -> OperatorOutput:
        B = u.shape[0]
        b = self.branch(u).view(B, self.out_dim, self.modes)
        y = self._contract(b, self.trunk(coords)) + self.bias

        losses = {"total": torch.tensor(0.0, device=y.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]

        return OperatorOutput(y=y, losses=losses, extras={})
