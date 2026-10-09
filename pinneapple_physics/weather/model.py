"""Global forecast network on an equiangular latitude-longitude grid.

A U-Net whose convolutions see the sphere: longitude is periodic (circular padding), and the rows padded beyond a
pole are the rows on the other side of that pole, shifted by half the circle (what a stencil crossing the pole
actually touches). The network maps the state at t and t - 6 h, the fixed fields (land-sea mask, orography) and the
forcing (top-of-atmosphere sunlight at t and t + 6 h, latitude, longitude) to the 6-hour change of the state.
Longer forecasts are made by feeding predictions back (autoregressive rollout); training continues on rollouts of
several steps so that errors made by the network itself are part of what it learns to correct.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def sphere_pad(x: torch.Tensor, p: int) -> torch.Tensor:
    """Pad (B, C, lat, lon) by ``p`` cells: circular in longitude, across the poles in latitude."""
    if p == 0:
        return x
    nlon = x.shape[-1]
    north = torch.roll(x[..., :p, :].flip(-2), nlon // 2, dims=-1)      # rows across the pole, half a turn away
    south = torch.roll(x[..., -p:, :].flip(-2), nlon // 2, dims=-1)
    x = torch.cat([north, x, south], dim=-2)
    return torch.cat([x[..., -p:], x, x[..., :p]], dim=-1)


class SConv(nn.Module):
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.p = k // 2
        self.conv = nn.Conv2d(cin, cout, k)

    def forward(self, x):
        return self.conv(sphere_pad(x, self.p))


class Block(nn.Module):
    """Residual block: two spherical 3x3 convolutions with group norm and GELU."""

    def __init__(self, cin, cout):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), SConv(cin, cout)
        self.n2, self.c2 = nn.GroupNorm(8, cout), SConv(cout, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.c1(F.gelu(self.n1(x)))
        h = self.c2(F.gelu(self.n2(h)))
        return h + self.skip(x)


class SphereUNet(nn.Module):
    """``patch`` > 1 first folds patch x patch cells into channels (pixel unshuffle), so the U-Net runs on a grid that
    is ``patch`` times coarser and the output is unfolded back; a light full-resolution layer then smooths the seams."""

    def __init__(self, n_state: int, n_extra: int, widths=(128, 192, 256, 384), blocks: int = 2, patch: int = 1,
                 refine: int = 64):
        super().__init__()
        self.n_state, self.patch = n_state, patch
        cin = (2 * n_state + n_extra) * patch * patch
        self.inp = SConv(cin, widths[0])
        self.down = nn.ModuleList()
        c = widths[0]
        for w in widths:
            self.down.append(nn.Sequential(*[Block(c if j == 0 else w, w) for j in range(blocks)]))
            c = w
        self.mid = nn.Sequential(Block(c, c), Block(c, c))
        self.up = nn.ModuleList()
        for w in reversed(widths[:-1]):
            self.up.append(nn.Sequential(Block(c + w, w), *[Block(w, w) for _ in range(blocks - 1)]))
            c = w
        self.head = nn.Sequential(nn.GroupNorm(8, c), nn.GELU(), SConv(c, n_state * patch * patch if patch > 1 else n_state))
        if patch > 1:
            self.refine = nn.Sequential(SConv(n_state, refine), nn.GELU(), SConv(refine, n_state))
            nn.init.zeros_(self.refine[-1].conv.weight)
            nn.init.zeros_(self.refine[-1].conv.bias)
        nn.init.zeros_(self.head[-1].conv.weight)
        nn.init.zeros_(self.head[-1].conv.bias)

    def forward(self, x_t, x_prev, extra):
        h = torch.cat([x_t, x_t - x_prev, extra], 1)
        if self.patch > 1:
            h = F.pixel_unshuffle(h, self.patch)
        h = self.inp(h)
        skips = []
        for i, d in enumerate(self.down):
            h = d(h)
            if i < len(self.down) - 1:
                skips.append(h)
                h = F.avg_pool2d(h, 2)
        h = self.mid(h)
        for u in self.up:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = u(torch.cat([h, skips.pop()], 1))
        d = self.head(h)
        if self.patch > 1:
            d = F.pixel_shuffle(d, self.patch)
            d = d + self.refine(d)
        return x_t + d


# ------------------------------------------------------------------ forcing

def toa_insolation(times, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Cosine of the solar zenith angle clipped at 0 (normalised top-of-atmosphere sunlight), (T, lat, lon)."""
    t = np.asarray(times, dtype="datetime64[ns]")
    doy = (t - t.astype("datetime64[Y]")).astype("timedelta64[h]").astype(float) / 24.0
    hour = (t - t.astype("datetime64[D]")).astype("timedelta64[m]").astype(float) / 60.0
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (doy - 80.0) / 365.25)
    la = np.deg2rad(lat)[None, :, None]
    ha = np.deg2rad((hour[:, None, None] * 15.0 + lon[None, None, :]) - 180.0)
    cz = np.sin(la) * np.sin(decl[:, None, None]) + np.cos(la) * np.cos(decl[:, None, None]) * np.cos(ha)
    return np.clip(cz, 0, None).astype(np.float32)


def static_fields(lat: np.ndarray, lon: np.ndarray, constants: np.ndarray) -> np.ndarray:
    la, lo = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    return np.concatenate([constants, np.stack([np.sin(la), np.cos(lo) * np.cos(la), np.sin(lo) * np.cos(la)])]
                          ).astype(np.float32)
