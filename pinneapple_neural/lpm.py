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

Two backbones are available. ``"mlp"`` (default) is the compact per-query MLP of the first
release. ``"attention"`` adds cross-attention from every query to the surface points of its own
design (:class:`PointAttention`) — the geometry-aware transformer of the production LPMs
(Transolver: Wu et al., ICML 2024, arXiv:2402.02366; GeoTransolver: Herde et al., 2025,
arXiv:2507.13822; AB-UPT: 2025, arXiv:2505.18781), at a cost quadratic in the surface samples.
Both backbones train either one design at a time (:func:`fit`) or in padded batches
(:func:`stack_designs` + ``fit(..., batch_designs=N)``), which is what makes a design space of
~1000 simulations trainable; :func:`ellipse_design_space` builds such a space with the exact
potential-flow answer, so scaling can be measured without a solver.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

__all__ = [
    "Design",
    "DesignBatch",
    "FourierEncoding",
    "GeometryEncoder",
    "LargePhysicsModel",
    "PointAttention",
    "design_space",
    "ellipse_design_space",
    "fit",
    "multiscale_neighbourhood",
    "stack_designs",
]


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


def _multiscale(query, points, normals, point_mask, radii: Sequence[float]):
    """Batched :func:`multiscale_neighbourhood`: (B,Q,d), (B,P,d), (B,P,d), (B,P) -> (B,Q,R*(2d+1))."""
    d2 = torch.cdist(query, points) ** 2  # (B, Q, P)
    feats = []
    total = (point_mask.sum(1) if point_mask is not None else torch.full((query.shape[0],), points.shape[1],
                                                                        device=query.device)).clamp_min(1)
    for r in radii:
        w = (d2 <= r * r).to(query.dtype)
        if point_mask is not None:
            w = w * point_mask[:, None, :].to(query.dtype)
        cnt = w.sum(-1, keepdim=True)  # (B, Q, 1)
        safe = cnt.clamp_min(1.0)
        mean_off = torch.einsum("bqp,bpd->bqd", w, points) / safe - query
        mean_n = torch.einsum("bqp,bpd->bqd", w, normals) / safe
        feats += [mean_off / r, mean_n, cnt / total[:, None, None].to(query.dtype)]
    return torch.cat(feats, dim=-1)


def multiscale_neighbourhood(query, points, normals, radii: Sequence[float]):
    """Per query point and radius: mean offset to neighbours, mean neighbour normal, occupancy.

    query (Q,d), points (P,d), normals (P,d) -> (Q, len(radii) * (2d + 1)). Invariant to the
    order of ``points`` and independent for each query point.
    """
    return _multiscale(query[None], points[None], normals[None], None, radii)[0]


class GeometryEncoder(nn.Module):
    """DeepSets: per-point MLP on (position, normal) then mean+max pooling -> geometry code.

    With a ``mask`` (masked positions of a padded batch) the pooling ignores the padding, so a
    design pooled inside a batch gets the same code as on its own.
    """

    def __init__(self, d: int, width: int = 64, code: int = 32):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(2 * d, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, code))

    def forward(self, points, normals, mask=None):
        single = points.dim() == 2
        pts, nrm = (points[None], normals[None]) if single else (points, normals)
        h = self.phi(torch.cat([pts, nrm], dim=-1))  # (B, P, W)
        if mask is None:
            mask = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        w = mask.to(h.dtype)[..., None]
        mean = (h * w).sum(1) / w.sum(1).clamp_min(1.0)
        mx = h.masked_fill(~mask[..., None], float("-inf")).max(1).values
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
        code = self.rho(torch.cat([mean, mx], dim=-1))
        return code[0] if single else code


class PointAttention(nn.Module):
    """Cross-attention from each query token to the surface points of its own design.

    The geometry-aware transformer block of the production LPMs: every query reads the sampled
    surface (position, normal) of the design it belongs to, keys of other designs in the same
    batch are masked out. ``h`` (B,Q,W) in and out, so it drops after the trunk of
    :class:`LargePhysicsModel`.
    """

    def __init__(self, dim: int, width: int, heads: int = 4):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(2 * dim, width), nn.GELU(), nn.Linear(width, width))
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, h, points, normals, point_mask=None):
        keys = self.emb(torch.cat([points, normals], dim=-1))
        pad = None if point_mask is None else ~point_mask
        a, _ = self.attn(self.norm1(h), keys, keys, key_padding_mask=pad, need_weights=False)
        h = h + a
        return h + self.ff(self.norm2(h))


@dataclass
class Design:
    """One simulation of the design space: its geometry, operating point and a target field."""

    points: torch.Tensor  # (P, d) surface samples
    normals: torch.Tensor  # (P, d)
    query: torch.Tensor  # (Q, d) where the field is known/predicted
    params: torch.Tensor  # (p,) operating parameters (may be empty)
    target: torch.Tensor | None = None  # (Q, f)

    @classmethod
    def ellipse(cls, a: float, b: float, n: int = 128) -> Design:
        """Ellipse ``x = a cos t, y = b sin t`` (``n`` points) in a uniform stream along x, with its exact Cp.

        Surface speed of potential flow past an ellipse (Milne-Thomson, *Theoretical Hydrodynamics*, 5th ed.,
        1968, §9.61): ``q / U = (a + b) |sin t| / sqrt(a^2 sin^2 t + b^2 cos^2 t)``; target ``Cp = 1 - (q/U)^2``
        at the surface points (query = points), no operating parameters, outward unit normals.
        """
        th = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
        pts = torch.stack([a * torch.cos(th), b * torch.sin(th)], 1)
        nrm = torch.stack([b * torch.cos(th), a * torch.sin(th)], 1)
        nrm = nrm / nrm.norm(dim=1, keepdim=True)
        q = (a + b) * torch.sin(th).abs() / torch.sqrt(a ** 2 * torch.sin(th) ** 2 + b ** 2 * torch.cos(th) ** 2)
        return cls(pts, nrm, pts, torch.zeros(0), (1 - q ** 2)[:, None])

    @classmethod
    def square(cls, s: float = 1.0, n: int = 128) -> Design:
        """Square ``[-s, s]^2`` sampled by ``n`` points (counter-clockwise from ``(-s, -s)``), outward normals,
        no target (an out-of-distribution probe for a smooth-body design space)."""
        pts, nr = [], []
        for u in torch.linspace(0, 4, n + 1)[:-1].tolist():
            k, f = int(u), u - int(u)
            pts.append([(-s + 2 * s * f, -s), (s, -s + 2 * s * f), (s - 2 * s * f, s), (-s, s - 2 * s * f)][k])
            nr.append([(0, -1), (1, 0), (0, 1), (-1, 0)][k])
        P = torch.tensor(pts)
        return cls(P, torch.tensor(nr, dtype=torch.float32), P, torch.zeros(0), None)


@dataclass
class DesignBatch:
    """Several designs padded to a common surface-query count, trained in one pass.

    Built by :func:`stack_designs`; the masks tell the pooling, the neighbourhoods and the
    attention which entries are real, so a design trained inside a batch sees exactly what it
    would see alone.
    """

    points: torch.Tensor  # (B, P, d)
    normals: torch.Tensor  # (B, P, d)
    point_mask: torch.Tensor  # (B, P) bool
    query: torch.Tensor  # (B, Q, d)
    query_mask: torch.Tensor  # (B, Q) bool
    params: torch.Tensor  # (B, p)
    target: torch.Tensor | None = None  # (B, Q, f)

    @property
    def batch_size(self) -> int:
        return int(self.points.shape[0])

    def to(self, device) -> DesignBatch:
        """Move every tensor (but not the dataclass) to ``device``."""
        def move(t):
            return None if t is None else t.to(device)

        return DesignBatch(move(self.points), move(self.normals), move(self.point_mask), move(self.query),
                           move(self.query_mask), move(self.params), move(self.target))


def stack_designs(designs: Sequence[Design], device=None) -> DesignBatch:
    """Pad a list of designs into one :class:`DesignBatch` (zero padding + masks)."""
    if not designs:
        raise ValueError("stack_designs needs at least one design")
    dev = torch.device(device) if device is not None else designs[0].points.device
    B = len(designs)
    dim = designs[0].points.shape[1]
    P = max(len(d.points) for d in designs)
    Q = max(len(d.query) for d in designs)
    p_dim = max(int(d.params.numel()) for d in designs)
    fields = next((d.target.shape[1] for d in designs if d.target is not None), None)

    points = torch.zeros(B, P, dim, device=dev)
    normals = torch.zeros(B, P, dim, device=dev)
    point_mask = torch.zeros(B, P, dtype=torch.bool, device=dev)
    query = torch.zeros(B, Q, dim, device=dev)
    query_mask = torch.zeros(B, Q, dtype=torch.bool, device=dev)
    params = torch.zeros(B, p_dim, device=dev)
    target = torch.zeros(B, Q, fields, device=dev) if fields is not None else None
    for i, d in enumerate(designs):
        p, q = len(d.points), len(d.query)
        points[i, :p] = d.points.to(dev)
        normals[i, :p] = d.normals.to(dev)
        point_mask[i, :p] = True
        query[i, :q] = d.query.to(dev)
        query_mask[i, :q] = True
        if d.params.numel():
            params[i, :d.params.numel()] = d.params.to(dev)
        if target is not None and d.target is not None:
            target[i, :q] = d.target.to(dev)
    return DesignBatch(points, normals, point_mask, query, query_mask, params, target)


def ellipse_design_space(n: int, *, n_points: int = 64, a_range: tuple[float, float] = (0.6, 1.4),
                         b_range: tuple[float, float] = (0.6, 1.4), seed: int = 0,
                         device=None) -> list[Design]:
    """``n`` ellipse designs sampled from the family, each with its exact potential-flow Cp.

    A design space with no solver in the loop: the answer is analytic (Milne-Thomson §9.61, see
    :meth:`Design.ellipse`), so a large-scale run measures the model, not a data pipeline.
    """
    g = torch.Generator().manual_seed(seed)
    a = a_range[0] + (a_range[1] - a_range[0]) * torch.rand(n, generator=g)
    b = b_range[0] + (b_range[1] - b_range[0]) * torch.rand(n, generator=g)
    designs = [Design.ellipse(float(ai), float(bi), n_points) for ai, bi in zip(a, b, strict=True)]
    return [d if device is None else Design(d.points.to(device), d.normals.to(device), d.query.to(device),
                                            d.params.to(device), None if d.target is None else d.target.to(device))
            for d in designs]


#: ``design_space`` is :func:`ellipse_design_space` under the generic name used by callers that
#: only need "a design space of ``n`` simulations".
design_space = ellipse_design_space


class LargePhysicsModel(nn.Module):
    """One model for a whole design space.

    ``backbone="mlp"`` (default) is the per-query MLP of the first release; ``backbone="attention"``
    adds ``attn_layers`` :class:`PointAttention` blocks so every query reads the surface of its
    design. Everything else — the geometry code, the ensemble heads, the OOD reference — is shared.
    """

    def __init__(self, dim: int, n_fields: int, n_params: int = 0, radii: Sequence[float] = (0.1, 0.4, 1.2),
                 n_freq: int = 6, width: int = 128, code: int = 32, n_heads: int = 5,
                 backbone: str = "mlp", attn_heads: int = 4, attn_layers: int = 2):
        super().__init__()
        if backbone not in ("mlp", "attention"):
            raise ValueError(f"backbone must be 'mlp' or 'attention', got {backbone!r}")
        self.backbone = backbone
        self.radii = tuple(radii)
        self.enc = FourierEncoding(n_freq)
        self.geo = GeometryEncoder(dim, code=code)
        local = self.enc.out_dim(dim) + len(self.radii) * (2 * dim + 1)
        self.trunk = nn.Sequential(nn.Linear(local + code + n_params, width), nn.GELU(),
                                   nn.Linear(width, width), nn.GELU())
        self.blocks = nn.ModuleList(
            [PointAttention(dim, width, attn_heads) for _ in range(attn_layers if backbone == "attention" else 0)])
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, n_fields))
                                    for _ in range(n_heads)])
        self.register_buffer("train_codes", torch.zeros(0, code))
        self.register_buffer("param_lo", torch.zeros(n_params))
        self.register_buffer("param_hi", torch.zeros(n_params))

    def _members_batch(self, b: DesignBatch) -> torch.Tensor:
        """Ensemble members for a padded batch: (H, B, Q, f)."""
        Q = b.query.shape[1]
        g = self.geo(b.points, b.normals, b.point_mask)  # (B, code)
        local = torch.cat([self.enc(b.query), _multiscale(b.query, b.points, b.normals, b.point_mask, self.radii)], -1)
        cond = [g[:, None, :].expand(-1, Q, -1)]
        if b.params.numel():
            cond.append(b.params[:, None, :].expand(-1, Q, -1))
        h = self.trunk(torch.cat([local] + cond, dim=-1))
        for blk in self.blocks:
            h = blk(h, b.points, b.normals, b.point_mask)
        return torch.stack([hd(h) for hd in self.heads])  # (H, B, Q, f)

    def _on_device(self, x: Design) -> Design:
        """The design on the model's device (a CPU probe evaluated on an MPS/CUDA model)."""
        dev = self.heads[0][0].weight.device
        if x.points.device == dev:
            return x
        return Design(x.points.to(dev), x.normals.to(dev), x.query.to(dev), x.params.to(dev),
                      None if x.target is None else x.target.to(dev))

    def _members(self, x) -> torch.Tensor:
        """(H, Q, f) for one :class:`Design`, (H, B, Q, f) for a :class:`DesignBatch`."""
        if isinstance(x, Design):
            return self._members_batch(stack_designs([self._on_device(x)]))[:, 0]
        return self._members_batch(x)

    def forward(self, x) -> torch.Tensor:
        """Ensemble mean: (Q, f) for a :class:`Design`, (B, Q, f) for a :class:`DesignBatch`."""
        return self._members(x).mean(0)

    @torch.no_grad()
    def predict(self, x: Design) -> dict[str, torch.Tensor]:
        """Mean, ensemble std (epistemic spread) and an OOD report for this design."""
        m = self._members(x)
        return {"mean": m.mean(0), "std": m.std(0), **self.ood(x)}

    @torch.no_grad()
    def ood(self, x: Design, factor: float = 1.5) -> dict[str, object]:
        """Geometry code distance to the nearest training design vs. the spread seen in training."""
        if len(self.train_codes) < 2:
            return {"ood": False, "ood_score": 0.0, "reasons": ["no training reference stored"]}
        x = self._on_device(x)
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
    def remember_training_set(self, designs: list[Design], batch_size: int = 256) -> None:
        """Store the geometry codes (and the parameter range) of the training designs, for OOD."""
        if not designs:
            raise ValueError("remember_training_set needs the training designs")
        dev = self.train_codes.device
        codes = [self.geo(b.points, b.normals, b.point_mask)
                 for b in (stack_designs(designs[i:i + batch_size], device=dev)
                           for i in range(0, len(designs), batch_size))]
        self.train_codes = torch.cat(codes)
        if designs[0].params.numel():
            P = torch.stack([d.params for d in designs]).to(dev)
            self.param_lo, self.param_hi = P.min(0).values, P.max(0).values


def fit(model: LargePhysicsModel, designs: list[Design], *, epochs: int = 2000, lr: float = 2e-3,
        freeze_geometry: bool = False, seed: int = 0, batch_designs: int | None = None,
        device=None) -> list[float]:
    """Train (or fine-tune with ``freeze_geometry=True``) on a list of designs.

    Bagging: each ensemble head sees its own bootstrap resample of the designs (fixed per call),
    so the spread between heads reflects how much the training designs constrain a prediction.
    ``batch_designs`` trains that many designs per optimizer step (padded and masked, so the
    result per design is the same as training it alone); ``None`` keeps the one-design loop.
    """
    g = torch.Generator().manual_seed(seed)
    for p in model.geo.parameters():
        p.requires_grad_(not freeze_geometry)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n, H = len(designs), len(model.heads)
    counts = torch.stack([torch.bincount(torch.randint(0, n, (n,), generator=g), minlength=n) for _ in range(H)])
    weights = counts.float() / counts.sum(1, keepdim=True)  # (H, n)
    if device is None:
        device = model.heads[0][0].weight.device
    elif torch.device(device) != model.heads[0][0].weight.device:
        model.to(device)
    size = n if batch_designs is None else max(1, min(int(batch_designs), n))
    chunks = [list(range(i, min(i + size, n))) for i in range(0, n, size)]
    hist = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = 0.0
        for idx in chunks:
            b = stack_designs([designs[i] for i in idx], device=device)
            pred = model._members_batch(b)  # (H, B, Q, f)
            fields = pred.shape[-1]
            m = b.query_mask.to(pred.dtype)[None, :, :, None]
            err = (((pred - b.target[None]) ** 2) * m).sum(dim=(-1, -2)) / (
                b.query_mask.sum(-1).clamp_min(1).to(pred.dtype)[None, :] * fields)  # (H, B)
            w = weights[:, idx].to(err.device)
            loss = loss + (w * err).sum() / H
        loss.backward()
        opt.step()
        sched.step()
        hist.append(float(loss.detach()))
    model.remember_training_set(designs)
    return hist
