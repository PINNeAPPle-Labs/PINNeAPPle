"""PINNsFormer as Zhao, Ding & Prakash define it (ICLR 2024, arXiv:2307.11833v3, Sec. 3 and Appendix A).

The architecture follows the official code, AdityaLab/pinnsformer at commit 00d790f5 (``model/pinnsformer.py``,
``util.make_time_sequence``), where the paper is silent on a detail (feed-forward width 256, the WaveAct placement,
the final WaveAct of the encoder and decoder):

- pseudo sequence (eq. 3): one point ``[x, t]`` becomes ``{[x, t], [x, t + dt], ..., [x, t + (k-1) dt]}``;
- spatio-temporal mixer: one linear layer to ``d_model``;
- encoder: self-attention + feed-forward, pre-activated by WaveAct, residual; decoder: encoder-decoder attention
  only (no self-attention), queried by the same embedding as the encoder;
- output MLP ``d_model -> d_hidden -> d_hidden -> out`` with WaveAct;
- WaveAct (eq. 4): ``w1 sin(x) + w2 cos(x)``, ``w1 = w2 = 1`` learnable, one pair per placement.

Attention only mixes the k copies of the same point, so ``u(x, t)`` (the first element of its sequence) is a
pointwise function of ``(x, t)``. :class:`PINNsFormer` in ``pinnsformer.py`` is a different model: it attends
across unrelated collocation points.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .base import PINNBase, PINNOutput

Tensor = torch.Tensor

__all__ = ["WaveAct", "PINNsFormerPaper", "init_official"]


class WaveAct(nn.Module):
    """``w1 sin(x) + w2 cos(x)`` with learnable scalars ``w1 = w2 = 1`` at start (eq. 4)."""

    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), WaveAct(), nn.Linear(d_hidden, d_hidden), WaveAct(),
                         nn.Linear(d_hidden, d_out))


# The attention calls keep need_weights=True (PyTorch's default, as in the official code): it takes the explicit
# softmax path, which supports the second derivatives a PDE residual needs; the fused CPU kernel does not.
class _EncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.ff = _mlp(d_model, d_ff, d_model)
        self.act1, self.act2 = WaveAct(), WaveAct()

    def forward(self, x: Tensor) -> Tensor:
        h = self.act1(x)
        x = x + self.attn(h, h, h)[0]
        return x + self.ff(self.act2(x))


class _DecoderLayer(_EncoderLayer):
    def forward(self, x: Tensor, memory: Tensor) -> Tensor:  # type: ignore[override]
        h = self.act1(x)
        x = x + self.attn(h, memory, memory)[0]
        return x + self.ff(self.act2(x))


def init_official(module: nn.Module) -> None:
    """The official notebooks' init: every ``nn.Linear`` (attention output projections included) gets Xavier-uniform
    weights and biases 0.01; the attention input projections keep PyTorch's own init."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class PINNsFormerPaper(PINNBase):
    """PINNsFormer of Zhao, Ding & Prakash (ICLR 2024); see the module docstring.

    Parameters (paper Table 4 / the official notebooks): ``k = 5``, ``dt = 1e-4``, ``d_model = 32``,
    ``d_hidden = 512``, ``num_layers = 1`` (encoder and decoder), ``heads = 2``; ``d_ff = 256`` is the official
    feed-forward width. ``time_dim`` is the input column that the pseudo sequence shifts (the last by default).
    The official init (:func:`init_official`) is applied at construction; at the defaults with ``in_dim = 2`` the
    model has 453 561 parameters, as the official notebooks print.

    ``forward_tensor(x)``: ``x`` (N, in_dim) -> ``u`` (N, out_dim) at the points (the first element of each
    sequence). ``forward_sequence(x)``: (N, k, out_dim) at the k shifted copies, for the sequential loss (eq. 6).
    A 3-D input (N, k, in_dim) is taken as ready-made sequences.
    """

    def __init__(self, in_dim: int = 2, out_dim: int = 1, *, k: int = 5, dt: float = 1e-4, d_model: int = 32,
                 d_hidden: int = 512, num_layers: int = 1, heads: int = 2, d_ff: int = 256, time_dim: int = -1):
        super().__init__()
        self.in_dim, self.out_dim, self.k, self.dt = int(in_dim), int(out_dim), int(k), float(dt)
        self.time_dim = time_dim % self.in_dim
        self.linear_emb = nn.Linear(self.in_dim, d_model)
        enc, dec = _EncoderLayer(d_model, heads, d_ff), _DecoderLayer(d_model, heads, d_ff)
        # copies of one layer, as the official get_clones (deepcopy) builds them
        self.encoder = nn.ModuleList([copy.deepcopy(enc) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([copy.deepcopy(dec) for _ in range(num_layers)])
        self.enc_act, self.dec_act = WaveAct(), WaveAct()
        self.linear_out = _mlp(d_model, d_hidden, self.out_dim)
        self.inverse_params = nn.ParameterDict()
        init_official(self)

    def make_sequence(self, x: Tensor) -> Tensor:
        """``(N, in_dim) -> (N, k, in_dim)``: copy j has ``t + j * dt`` in ``time_dim`` (eq. 3)."""
        seq = x.unsqueeze(1).repeat(1, self.k, 1)
        shift = torch.zeros(self.k, self.in_dim, dtype=x.dtype, device=x.device)
        shift[:, self.time_dim] = self.dt * torch.arange(self.k, dtype=x.dtype, device=x.device)
        return seq + shift

    def _run(self, seq: Tensor) -> Tensor:
        src = self.linear_emb(seq)
        e = src
        for layer in self.encoder:
            e = layer(e)
        e = self.enc_act(e)
        d = src
        for layer in self.decoder:
            d = layer(d, e)
        return self.linear_out(self.dec_act(d))

    def forward_sequence(self, x: Tensor) -> Tensor:
        return self._run(x if x.ndim == 3 else self.make_sequence(x))

    def forward_tensor(self, x: Tensor) -> Tensor:
        if x.ndim == 3:
            return self._run(x)
        if x.ndim != 2:
            raise ValueError(f"PINNsFormerPaper expects (N, in_dim) or (N, k, in_dim), got {tuple(x.shape)}")
        return self._run(self.make_sequence(x))[:, 0, :]

    def forward(self, x: Tensor, *, physics_fn=None, physics_data=None) -> PINNOutput:
        y = self.forward_tensor(x)
        z = torch.zeros((), device=y.device, dtype=y.dtype)
        losses = {"total": z}
        if physics_fn is not None and physics_data is not None:
            pl = self.physics_loss(physics_fn=physics_fn, physics_data=physics_data)
            losses.update(pl)
            losses["total"] = losses["total"] + losses.get("physics", z)
        return PINNOutput(y=y, losses=losses, extras={})
