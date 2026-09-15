from __future__ import annotations
"""Adaptive Fourier Neural Operator (AFNO).

Reference: Guibas et al., ICLR 2022
  "Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers"
  https://arxiv.org/abs/2111.13587
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.fft as fft

from .base import BaseModel, ModelOutput


class AFNOLayer(nn.Module):
    """Single AFNO layer: block-diagonal token mixing in Fourier space +
    channel MLP.

    The forward pass, following Guibas et al. (2022) exactly (this
    implementation previously used a per-mode dense C×C complex weight,
    which is architecturally an FNO2D spectral convolution wearing the AFNO
    name -- AFNO's own defining features, a resolution-independent
    block-diagonal weight SHARED across every retained frequency and a
    soft-shrinkage sparsification step, were both absent; fixed below):

    1. Apply 2-D real FFT to the spatial axes ``(H, W)`` of the input.
    2. Retain only the ``n_modes_h × n_modes_w`` lowest-frequency components.
    3. Split the channel dimension into ``num_blocks`` blocks and apply a
       2-layer complex MLP (real/imag parts via the standard complex-matmul
       expansion, with a ReLU between the two layers) whose weights are the
       SAME for every retained frequency location -- this is what makes the
       parameter count independent of spatial resolution, AFNO's whole
       point, unlike a per-mode weight.
    4. Soft-shrink (``torch.nn.functional.softshrink``) the resulting
       frequency-domain activations towards zero, promoting a sparse
       frequency representation.
    5. Apply inverse FFT to reconstruct the spatial field, with an
       inner residual connection back to the pre-FFT input.
    6. Add a channel-mixing MLP (analogous to the FFN in a Transformer).

    Args:
        hidden_dim: Channel width ``C`` (must be divisible by ``num_blocks``).
        n_modes_h: Number of Fourier modes to retain in height dimension.
        n_modes_w: Number of Fourier modes to retain in width dimension.
        num_blocks: Number of blocks the channel dim is split into for the
            block-diagonal frequency-domain MLP (default 8, per the paper).
        hidden_size_factor: Width multiplier for the frequency-domain
            block MLP's hidden layer (default 1, per the paper).
        sparsity_threshold: Soft-shrinkage threshold applied to
            frequency-domain activations (default 0.01, per the paper).
        mlp_ratio: Expansion ratio for the channel MLP (hidden = hidden_dim * mlp_ratio).
        dropout: Dropout probability applied inside the channel MLP.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_modes_h: int = 12,
        n_modes_w: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_blocks: int = 8,
        hidden_size_factor: int = 1,
        sparsity_threshold: float = 0.01,
    ):
        super().__init__()
        if hidden_dim % num_blocks != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_blocks ({num_blocks})"
            )
        self.hidden_dim = hidden_dim
        self.n_modes_h = n_modes_h
        self.n_modes_w = n_modes_w
        self.num_blocks = num_blocks
        self.block_size = hidden_dim // num_blocks
        self.sparsity_threshold = sparsity_threshold

        # Block-diagonal 2-layer complex MLP, SHARED across every retained
        # frequency location (no mode index anywhere in these shapes) --
        # the real AFNO mixing mechanism. Index 0/1 on the leading dim of
        # each parameter holds the real/imaginary part respectively.
        scale = 0.02
        bs = self.block_size
        bh = bs * hidden_size_factor
        self.w1 = nn.Parameter(scale * torch.randn(2, num_blocks, bs, bh))
        self.b1 = nn.Parameter(scale * torch.randn(2, num_blocks, bh))
        self.w2 = nn.Parameter(scale * torch.randn(2, num_blocks, bh, bs))
        self.b2 = nn.Parameter(scale * torch.randn(2, num_blocks, bs))

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def _block_mlp(self, x_re: torch.Tensor, x_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """The shared block-diagonal complex MLP. Inputs/outputs shaped
        ``(B, N, num_blocks, block_size)`` where ``N`` is the number of
        retained frequency locations (flattened)."""
        w1, b1, w2, b2 = self.w1, self.b1, self.w2, self.b2
        o1_re = torch.relu(
            torch.einsum("bnki,kio->bnko", x_re, w1[0])
            - torch.einsum("bnki,kio->bnko", x_im, w1[1])
            + b1[0]
        )
        o1_im = torch.relu(
            torch.einsum("bnki,kio->bnko", x_im, w1[0])
            + torch.einsum("bnki,kio->bnko", x_re, w1[1])
            + b1[1]
        )
        o2_re = (
            torch.einsum("bnki,kio->bnko", o1_re, w2[0])
            - torch.einsum("bnki,kio->bnko", o1_im, w2[1])
            + b2[0]
        )
        o2_im = (
            torch.einsum("bnki,kio->bnko", o1_im, w2[0])
            + torch.einsum("bnki,kio->bnko", o1_re, w2[1])
            + b2[1]
        )
        return o2_re, o2_im

    def _fourier_mix(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learned mixing in truncated Fourier space.

        Args:
            x: ``(B, H, W, C)`` spatial field.

        Returns:
            ``(B, H, W, C)`` after Fourier mixing.
        """
        B, H, W, C = x.shape

        # 2-D real FFT over spatial axes  ->  (B, H, W//2+1, C)  complex
        x_ft = fft.rfft2(x, dim=(1, 2), norm="ortho")  # (B, H, W//2+1, C)

        nh = min(self.n_modes_h, H)
        nw = min(self.n_modes_w, x_ft.size(2))

        # Positive + negative frequency rows (top and bottom of H axis)
        nh_pos = nh // 2 + nh % 2  # positive-freq rows
        nh_neg = nh // 2            # negative-freq rows (mirrored)

        out_ft = torch.zeros_like(x_ft)

        def _process(x_c: torch.Tensor) -> torch.Tensor:
            # x_c: (B, nh_part, nw, C) complex -> same shape, block-MLP'd
            # and soft-shrunk with weights SHARED across every (h, w) here.
            bpart, wpart = x_c.shape[1], x_c.shape[2]
            xr = x_c.real.reshape(B, bpart * wpart, self.num_blocks, self.block_size)
            xi = x_c.imag.reshape(B, bpart * wpart, self.num_blocks, self.block_size)
            o_re, o_im = self._block_mlp(xr, xi)
            o_re = torch.nn.functional.softshrink(o_re, lambd=self.sparsity_threshold)
            o_im = torch.nn.functional.softshrink(o_im, lambd=self.sparsity_threshold)
            o_re = o_re.reshape(B, bpart, wpart, C)
            o_im = o_im.reshape(B, bpart, wpart, C)
            return torch.complex(o_re, o_im)

        # Positive-frequency rows
        out_ft[:, :nh_pos, :nw] = _process(x_ft[:, :nh_pos, :nw])

        # Negative-frequency rows  (stored at H-nh_neg : H in rFFT output)
        if nh_neg > 0:
            out_ft[:, H - nh_neg : H, :nw] = _process(x_ft[:, H - nh_neg : H, :nw])

        # Inverse 2-D real FFT  ->  (B, H, W, C), with the AFNO block's own
        # inner residual back to the pre-FFT input (Guibas et al.'s `bias`).
        return fft.irfft2(out_ft, s=(H, W), dim=(1, 2), norm="ortho") + x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, H, W, C)`` spatial field.

        Returns:
            ``(B, H, W, C)`` after Fourier mixing + MLP.
        """
        # Fourier branch (residual)
        x = x + self._fourier_mix(self.norm1(x))
        # Channel MLP branch (residual)
        x = x + self.mlp(self.norm2(x))
        return x


class AFNO(BaseModel):
    """Adaptive Fourier Neural Operator for structured-grid PDEs.

    Processes 2-D spatial fields represented as ``(B, H, W, C)`` tensors.
    Suitable for weather/climate forecasting, turbulence modelling, and other
    applications where inputs live on a regular spatial grid.

    Architecture:
    1. A channel-mixing **input projection** ``in_channels -> hidden_dim``.
    2. ``n_layers`` :class:`AFNOLayer` blocks, each mixing tokens in
       truncated Fourier space and channels via a point-wise MLP.
    3. A channel-mixing **output projection** ``hidden_dim -> out_channels``.

    Args:
        in_channels: Number of input field channels ``C_in``.
        out_channels: Number of output field channels ``C_out``.
        hidden_dim: Internal channel width.
        n_layers: Number of AFNO blocks.
        n_modes_h: Fourier modes kept along the height axis.
        n_modes_w: Fourier modes kept along the width axis.
        mlp_ratio: MLP expansion ratio inside each AFNO block.
        dropout: Dropout probability.
    """

    family: str = "neural_operators"
    name: str = "afno"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int = 64,
        n_layers: int = 4,
        n_modes_h: int = 12,
        n_modes_w: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_channels, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                AFNOLayer(hidden_dim, n_modes_h, n_modes_w, mlp_ratio, dropout)
                for _ in range(n_layers)
            ]
        )
        self.out_proj = nn.Linear(hidden_dim, out_channels)

    def forward(self, x: torch.Tensor) -> ModelOutput:  # type: ignore[override]
        """Forward pass.

        Args:
            x: Input spatial field, shape ``(B, H, W, C_in)``.

        Returns:
            :class:`~pinneapple_models.base.ModelOutput` with ``y`` of shape
            ``(B, H, W, C_out)``.
        """
        h = self.in_proj(x)          # (B, H, W, hidden_dim)
        for block in self.blocks:
            h = block(h)
        return ModelOutput(y=self.out_proj(h))  # (B, H, W, out_channels)

    def forward_batch(self, batch: dict) -> ModelOutput:  # type: ignore[override]
        """Dict-based interface for the Arena/Trainer.

        Expects key ``'x'`` with shape ``(B, H, W, C_in)``.
        """
        return self.forward(batch["x"])
