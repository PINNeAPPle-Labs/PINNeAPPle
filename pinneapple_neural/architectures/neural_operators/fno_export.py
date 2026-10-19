"""ONNX export of Fourier neural operators.

The FNO's spectral convolution uses ``rfft2`` / ``irfft2`` and complex weights; neither the TorchScript nor the
dynamo ONNX exporter of PyTorch 2.x handles complex tensors. For a fixed grid, the truncated transforms are small
dense matrices, so the same layer can be written in real arithmetic:

* forward transform on the retained modes: ``X = E_h^T x E_w`` with ``E = cos - i sin`` restricted to the rows
  ``k1 in [0, m1) U [H - m1, H)`` and the columns ``k2 in [0, m2)``;
* the complex product with the weights split into real and imaginary parts;
* inverse ``irfft2`` of a spectrum that is zero outside those modes:
  ``y = Re(G O V^T c) / (H W)`` with ``G = e^{+i alpha}``, ``V = e^{+i beta}`` and ``c = 1`` for the zero (and the
  Nyquist) column, 2 otherwise -- exactly what ``irfft2`` computes (it discards the imaginary part of those columns).

:func:`to_exportable` returns a copy of a model with every ``_SpectralConv2d`` replaced by :class:`RealSpectralConv2d`
(identical outputs to float32 round-off, checked by :func:`export_onnx`), which then exports with the standard
exporter and runs in onnxruntime.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .fno import _SpectralConv2d

__all__ = ["RealSpectralConv2d", "to_exportable", "export_onnx"]


class _Plain(nn.Module):
    """``model(x)`` with an ``OperatorOutput`` unwrapped to its tensor: the exporter passes only ``x``."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        return getattr(y, "y", y)


class RealSpectralConv2d(nn.Module):
    """Real-arithmetic equivalent of :class:`~.fno._SpectralConv2d` on an ``H x W`` grid."""

    def __init__(self, conv: _SpectralConv2d, H: int, W: int):
        super().__init__()
        W_f = W // 2 + 1
        m1 = min(conv.modes1, H // 2)
        m2 = min(conv.modes2, W_f)
        self.H, self.W, self.out_c = int(H), int(W), conv.out_c
        k1 = np.concatenate([np.arange(m1), np.arange(H - m1, H)])
        k2 = np.arange(m2)
        h, w = np.arange(H), np.arange(W)
        a = 2 * np.pi * np.outer(h, k1) / H                       # (H, 2 m1)
        b = 2 * np.pi * np.outer(w, k2) / W                       # (W, m2)
        c = np.where((k2 == 0) | ((W % 2 == 0) & (k2 == W // 2)), 1.0, 2.0)
        f32 = torch.float32
        for name, v in (("Ch", np.cos(a)), ("Sh", np.sin(a)), ("Cw", np.cos(b)), ("Sw", np.sin(b)), ("c", c)):
            self.register_buffer(name, torch.tensor(v, dtype=f32))
        wts = torch.cat([conv.weights1.detach()[:, :, :m1, :m2], conv.weights2.detach()[:, :, :m1, :m2]], dim=2)
        self.register_buffer("Wr", wts.real.to(f32).contiguous())
        self.register_buffer("Wi", wts.imag.to(f32).contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Pr = x @ self.Cw                                         # (B, C, H, m2): x E_w, E_w = Cw - i Sw
        Pi = -(x @ self.Sw)
        ChT, ShT = self.Ch.transpose(0, 1), self.Sh.transpose(0, 1)
        Xr = ChT @ Pr + ShT @ Pi                                 # E_h^T P, E_h = Ch - i Sh
        Xi = ChT @ Pi - ShT @ Pr
        Or = torch.einsum("bihw,iohw->bohw", Xr, self.Wr) - torch.einsum("bihw,iohw->bohw", Xi, self.Wi)
        Oi = torch.einsum("bihw,iohw->bohw", Xr, self.Wi) + torch.einsum("bihw,iohw->bohw", Xi, self.Wr)
        Zr = self.Ch @ Or - self.Sh @ Oi                         # G O, G = Ch + i Sh
        Zi = self.Ch @ Oi + self.Sh @ Or
        y = (Zr * self.c) @ self.Cw.transpose(0, 1) - (Zi * self.c) @ self.Sw.transpose(0, 1)
        return y / (self.H * self.W)


def to_exportable(model: nn.Module, grid: tuple[int, int]) -> nn.Module:
    """A copy of ``model`` with every 2-D spectral convolution in real arithmetic for an ``H x W`` grid."""
    out = copy.deepcopy(model).eval()

    def swap(mod: nn.Module):
        for name, child in mod.named_children():
            if isinstance(child, _SpectralConv2d):
                setattr(mod, name, RealSpectralConv2d(child, *grid))
            else:
                swap(child)
    swap(out)
    return out.requires_grad_(False)


def export_onnx(model: nn.Module, example: torch.Tensor, path: str | Path, *, grid: tuple[int, int] | None = None,
                opset: int = 17, atol: float = 1e-4) -> float:
    """Export ``model`` (an FNO, or any module containing 2-D spectral convolutions) to ONNX at ``path`` with a dynamic
    batch axis, check onnxruntime against PyTorch on ``example`` and return the largest absolute difference (raises
    when it exceeds ``atol`` times the output scale). ``grid`` defaults to the last two axes of ``example``."""
    grid = grid or tuple(example.shape[-2:])
    model = _Plain(model).eval()
    real = to_exportable(model, grid)
    with torch.no_grad():
        ref = model.eval()(example)
        ref = getattr(ref, "y", ref)
        alt = real(example)
        alt = getattr(alt, "y", alt)
    scale = float(ref.abs().max()) or 1.0
    if not torch.allclose(ref, alt, atol=atol * scale, rtol=0):
        raise RuntimeError(f"real-arithmetic copy differs by {float((ref - alt).abs().max()):.3g}")
    torch.onnx.export(real, (example,), str(path), input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}, opset_version=opset,
                      dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: example.numpy().astype(np.float32)})[0]
    diff = float(np.abs(got - ref.numpy()).max())
    if not math.isfinite(diff) or diff > atol * scale:
        raise RuntimeError(f"onnxruntime differs from PyTorch by {diff:.3g}")
    return diff
