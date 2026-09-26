"""Large Physics Model (LPM) building blocks: one model for a whole design space.

Definition used (Luminary, "Vocabulary of Physics AI", 2026): an LPM is "a single trained model
that covers a design space wide enough to stand in for many individual simulations". In practice
(Luminary SHIFT models; "Demystifying Physics AI: the five stages to scale"; "How Large Physics
Models gain spatial context") that means:

1. **spatial context** for every query point:
   - multi-scale neighbourhoods (fine radius -> local curvature, coarse radius -> where on the
     body the point is), as GeoTransolver's multi-scale ball queries do;
   - Fourier positional encoding of coordinates against the spectral bias toward smooth
     functions (Tancik et al., "Fourier features let networks learn high frequency functions",
     NeurIPS 2020, arXiv:2006.10739);
2. **a geometry code** shared by all points of a design (permutation-invariant pooling, DeepSets:
   Zaheer et al., NeurIPS 2017, arXiv:1703.06114) and conditioning on operating parameters;
3. **query independence**: the prediction at a point does not depend on batching or order;
4. **uncertainty and out-of-distribution flags on every prediction** (deep-ensemble heads,
   Lakshminarayanan et al. 2017, arXiv:1612.01474; distance of the geometry code to the
   training designs);
5. **fine-tuning** on a few new simulations instead of retraining.

This module provides those pieces around a compact network; a production LPM swaps the head for
a geometry-aware transformer (Transolver/GeoTransolver/AB-UPT) and the training set for ~1000
simulations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn


class FourierEncoding(nn.Module):
    """x -> [x, sin(2^k pi x), cos(2^k pi x)] for k = 0..n_freq-1 (per coordinate)."""

    def __init__(self, n_freq: int = 6, include_input: bool = True):
        super().__init__()
        self.register_buffer("freqs", math.pi * 2.0 ** torch.arange(n_freq, dtype=torch.float32))
        self.include_input = include_input

    def out_dim(self, d: int) -> int:
        return d * (2 * len(self.freqs) + int(self.include_input))

    def forward(self, x):
        z = x[..., None] * self.freqs
        parts = [torch.sin(z).flatten(-2), torch.cos(z).flatten(-2)]
        return torch.cat(([x] if self.include_input else []) + parts, dim=-1)


def multiscale_neighbourhood(query, points, normals, radii: Sequence[float]):
    """Per query point and radius: mean offset to neighbours, mean neighbour normal, occupancy.

    query (Q,d), points (P,d), normals (P,d) -> (Q, len(radii) * (2d + 1)). Invariant to the
    order of ``points`` and independent for each query point.
    """
    d2 = torch.cdist(query, points) ** 2
    feats = []
    for r in radii:
        w = (d2 <= r * r).to(query.dtype)
        cnt = w.sum(1, keepdim=True)
        safe = cnt.clamp_min(1.0)
        mean_off = (w @ points) / safe - query
        mean_n = (w @ normals) / safe
        feats += [mean_off / r, mean_n, cnt / points.shape[0]]
    return torch.cat(feats, dim=1)


class GeometryEncoder(nn.Module):
    """DeepSets: per-point MLP on (position, normal) then mean+max pooling -> geometry code."""

    def __init__(self, d: int, width: int = 64, code: int = 32):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(2 * d, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, code))

    def forward(self, points, normals):
        h = self.phi(torch.cat([points, normals], dim=1))
        return self.rho(torch.cat([h.mean(0), h.max(0).values]))


@dataclass
class Design:
    """One simulation of the design space: its geometry, operating point and a target field."""

    points: torch.Tensor  # (P, d) surface samples
    normals: torch.Tensor  # (P, d)
    query: torch.Tensor  # (Q, d) where the field is known/predicted
    params: torch.Tensor  # (p,) operating parameters (may be empty)
    target: Optional[torch.Tensor] = None  # (Q, f)


class LargePhysicsModel(nn.Module):
    def __init__(self, dim: int, n_fields: int, n_params: int = 0, radii: Sequence[float] = (0.1, 0.4, 1.2),
                 n_freq: int = 6, width: int = 128, code: int = 32, n_heads: int = 5):
        super().__init__()
        self.radii = tuple(radii)
        self.enc = FourierEncoding(n_freq)
        self.geo = GeometryEncoder(dim, code=code)
        local = self.enc.out_dim(dim) + len(self.radii) * (2 * dim + 1)
        self.trunk = nn.Sequential(nn.Linear(local + code + n_params, width), nn.GELU(),
                                   nn.Linear(width, width), nn.GELU())
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, n_fields))
                                    for _ in range(n_heads)])
        self.register_buffer("train_codes", torch.zeros(0, code))
        self.register_buffer("param_lo", torch.zeros(n_params))
        self.register_buffer("param_hi", torch.zeros(n_params))

    def _members(self, x: Design) -> torch.Tensor:
        g = self.geo(x.points, x.normals)
        local = torch.cat([self.enc(x.query), multiscale_neighbourhood(x.query, x.points, x.normals, self.radii)], 1)
        cond = [g.expand(len(x.query), -1)] + ([x.params.expand(len(x.query), -1)] if x.params.numel() else [])
        h = self.trunk(torch.cat([local] + cond, 1))
        return torch.stack([hd(h) for hd in self.heads])  # (H, Q, f)

    def forward(self, x: Design) -> torch.Tensor:
        return self._members(x).mean(0)

    @torch.no_grad()
    def predict(self, x: Design) -> Dict[str, torch.Tensor]:
        """Mean, ensemble std (epistemic spread) and an OOD report for this design."""
        m = self._members(x)
        return {"mean": m.mean(0), "std": m.std(0), **self.ood(x)}

    @torch.no_grad()
    def ood(self, x: Design, factor: float = 1.5) -> Dict[str, object]:
        """Geometry code distance to the nearest training design vs. the spread seen in training."""
        if len(self.train_codes) < 2:
            return {"ood": False, "ood_score": 0.0, "reasons": ["no training reference stored"]}
        g = self.geo(x.points, x.normals)
        dist = torch.cdist(g[None], self.train_codes)[0].min()
        tc = self.train_codes
        nn_train = torch.cdist(tc, tc).fill_diagonal_(float("inf")).min(1).values
        ref = torch.quantile(nn_train, 0.95).clamp_min(1e-8)
        reasons = []
        score = float(dist / ref)
        if score > factor:
            reasons.append(f"geometry {score:.2f}x farther from the training designs than usual")
        if x.params.numel() and len(self.param_lo):
            out = (x.params < self.param_lo) | (x.params > self.param_hi)
            if bool(out.any()):
                reasons.append(f"operating parameters {out.nonzero().ravel().tolist()} outside the trained range")
        return {"ood": bool(reasons), "ood_score": score, "reasons": reasons}

    @torch.no_grad()
    def remember_training_set(self, designs: List[Design]) -> None:
        self.train_codes = torch.stack([self.geo(d.points, d.normals) for d in designs])
        if designs[0].params.numel():
            P = torch.stack([d.params for d in designs])
            self.param_lo, self.param_hi = P.min(0).values, P.max(0).values


def fit(model: LargePhysicsModel, designs: List[Design], *, epochs: int = 2000, lr: float = 2e-3,
        freeze_geometry: bool = False, seed: int = 0) -> List[float]:
    """Train (or fine-tune with ``freeze_geometry=True``) on a list of designs.

    Bagging: each ensemble head sees its own bootstrap resample of the designs (fixed per call),
    so the spread between heads reflects how much the training designs constrain a prediction.
    """
    g = torch.Generator().manual_seed(seed)
    for p in model.geo.parameters():
        p.requires_grad_(not freeze_geometry)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n, H = len(designs), len(model.heads)
    counts = torch.stack([torch.bincount(torch.randint(0, n, (n,), generator=g), minlength=n) for _ in range(H)])
    weights = counts.float() / counts.sum(1, keepdim=True)  # (H, n)
    hist = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = 0.0
        for i, d in enumerate(designs):
            if not bool((weights[:, i] > 0).any()):
                continue
            err = ((model._members(d) - d.target) ** 2).mean(dim=(1, 2))  # (H,)
            loss = loss + (weights[:, i] * err).sum() / H
        loss.backward()
        opt.step()
        sched.step()
        hist.append(float(loss.detach()))
    model.remember_training_set(designs)
    return hist
