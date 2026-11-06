"""CrashSolver: a part-hierarchy surrogate for field sequences over a structure (crash and, more generally, any
assembly whose mesh is split into structural parts).

The model follows the decomposition described for full-vehicle crash prediction (Elrefaie, Shu, Klenk & Ahmed,
"CarCrashNet", arXiv:2605.07098): rather than one attention pass over the whole mesh — where deformation is local
near the load path but the load path itself runs through the parts — the nodes are grouped by **structural part**,
and the sequence runs

1. **shared local component encoders**: a geometry-aware attention block (physics attention, Wu et al. 2024) applied
   to the nodes of each part on its own, so local deformation is learned on small token sets;
2. **global component transformer**: self-attention over the per-part latent summaries ``z_k``;
3. **interface message passing**: messages along a mesh-derived part graph (which parts touch which), so the load
   path coupling between bumper, rails, supports and cabin is represented and not averaged away;
4. **temporal decoder**: a per-node decoder that emits the displacement at a requested time, from the node's local
   feature, its part context and a Fourier encoding of the time.

Inputs, per sample: nodal coordinates ``X`` (B, N, 3), a wall thickness or any scalar per node ``tau`` (B, N),
semantic part labels ``c`` (B, N, 1-based, 0 for padding) and, optionally, dense part ids ``p`` (B, N) and the part
adjacency (B, K, K) — the interface graph; when it is not given, it is inferred from the shortest distance between
the node sets of two parts. Output: displacements ``U`` (B, T, N, 3) for the requested times, in the same units as the
coordinates, so positions are ``X + U``.

This is a reimplementation of the published design, not a copy of the authors' code (which is not public at the time
of writing), and the mapping described here is the one the paper states; see
``docs/guides/crash_and_surface_benchmarks.md`` for what it reproduces and what it does not.

    from pinneapple_neural.architectures.crash_solver import CrashSolverModel
    model = CrashSolverModel(in_dim=3, hidden=96, local_slices=32, global_layers=2, n_frames=8)
    out = model(x, part=part, thickness=tau, time=t)          # {"u": (B, T, N, 3), "part_tokens": (B, K, hidden)}
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .neural_operators.transolver import PhysicsAttention

__all__ = ["CrashSolverModel", "FourierTime"]


class FourierTime(nn.Module):
    """Fourier features of the time coordinate: ``[t, sin(2^k pi t), cos(2^k pi t)]`` scaled to a fixed band."""

    def __init__(self, dim: int = 16, t_max: float = 1.0):
        super().__init__()
        if dim % 2:
            raise ValueError("dim must be even")
        self.dim = dim
        self.register_buffer("freqs", 2.0 ** torch.arange(dim // 2).float() * torch.pi / max(float(t_max), 1e-12))
        self.out = nn.Sequential(nn.Linear(dim + 1, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """``t`` (B, T) -> (B, T, dim)."""
        a = t.unsqueeze(-1) * self.freqs
        feat = torch.cat([t.unsqueeze(-1), torch.sin(a), torch.cos(a)], dim=-1)
        return self.out(feat)


def _group_ranks(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Stable ordering by part label and the rank of every node inside its part, in the *original* node order."""
    order, sorted_labels, rank_sorted, k = _group_ranks_sorted(labels)
    rank = torch.zeros_like(rank_sorted)
    rank.scatter_(1, order, rank_sorted)
    return order, sorted_labels, rank, k


def _padded_parts(labels: torch.Tensor, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Group node features by part label, padded to the largest part. ``labels`` (B, N) 1-based (0 padding),
    ``feats`` (B, N, C) -> (B, K, N_max, C) and the (B, K, N_max) mask.

    Everything is done in the *sorted* order the grouping creates, so a padded node (label 0) lands in a bucket of its
    own that is dropped before returning — it can never overwrite a rank of a real part."""
    b, n, c = feats.shape
    order, sorted_labels, rank_sorted, k = _group_ranks_sorted(labels)
    if k == 0:
        return torch.zeros((b, 1, 1, c), dtype=feats.dtype, device=feats.device), \
            torch.zeros((b, 1, 1), dtype=torch.bool, device=feats.device)
    n_max = int(rank_sorted.max().item()) + 1
    valid = sorted_labels > 0
    dest = sorted_labels.clamp(min=0) * n_max + rank_sorted                     # (B, N) in sorted order
    flat = torch.zeros((b, (k + 1) * n_max, c), dtype=feats.dtype, device=feats.device)
    flat.scatter_(1, dest.unsqueeze(-1).expand(-1, -1, c),
                  torch.gather(feats, 1, order.unsqueeze(-1).expand(-1, -1, c)) * valid.unsqueeze(-1).to(feats.dtype))
    mask = torch.zeros((b, (k + 1) * n_max), dtype=torch.bool, device=feats.device)
    mask.scatter_(1, dest, valid)
    return flat.view(b, k + 1, n_max, c)[:, 1:].contiguous(), mask.view(b, k + 1, n_max)[:, 1:].contiguous()


def _group_ranks_sorted(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """(sorted order, sorted labels, rank inside the part *in sorted order*, K)."""
    b, n = labels.shape
    k = int(labels.max().item()) if labels.numel() else 0
    order = torch.argsort(labels, dim=1, stable=True)
    sorted_labels = torch.gather(labels, 1, order)
    firsts = torch.full((b, k + 1), n, dtype=torch.long, device=labels.device)
    firsts.scatter_reduce_(1, sorted_labels, torch.arange(n, device=labels.device).expand(b, -1).contiguous(),
                           reduce="amin")
    rank_sorted = torch.arange(n, device=labels.device).expand(b, -1) - torch.gather(firsts, 1, sorted_labels)
    return order, sorted_labels, rank_sorted, k


class CrashSolverModel(nn.Module):
    """The hierarchical part-aware surrogate (see the module docstring).

    Parameters
    ----------
    in_dim:
        Channels of the node input beyond the coordinates (the thickness is passed separately as ``thickness``).
    out_dim:
        Channels of the predicted field (3 for a displacement vector).
    hidden:
        Width of every latent (the paper's ablations show a *narrow* latent is better than a wide one under data
        scarcity, hence 96 by default).
    local_layers, local_slices:
        Depth of the shared per-part encoder and the number of physics-attention slices it assigns each part's nodes
        to (32 in the paper's best configuration).
    global_layers, global_heads:
        Depth and head count of the part-token transformer.
    interface_layers:
        Rounds of message passing on the part graph (0 disables the interface exchange).
    part_dim, n_part_ids, n_classes:
        Width of the part embeddings and the sizes of the dense part-id and semantic-class vocabularies.
    n_frames, time_dim, decoder_layers:
        The temporal decoder: how many times it is asked for at once when ``time`` is not given, the width of the
        Fourier time encoding and the depth of the per-node decoder.
    """

    def __init__(self, in_dim: int = 3, out_dim: int = 3, hidden: int = 96, local_layers: int = 1,
                 local_heads: int = 4, local_slices: int = 32, global_layers: int = 2, global_heads: int = 2,
                 global_slices: int = 16, interface_layers: int = 2, part_dim: int = 32, n_part_ids: int = 64,
                 n_classes: int = 12, n_frames: int = 8, time_dim: int = 16, decoder_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.hidden, self.out_dim, self.n_frames = int(hidden), int(out_dim), int(n_frames)
        self.n_part_ids, self.n_classes = int(n_part_ids), int(n_classes)
        self.interface_layers = int(interface_layers)

        self.node_in = nn.Sequential(nn.Linear(3 + in_dim + 1, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.thickness = nn.Linear(1, hidden)                       # tau: a wall thickness, a gauge
        self.part_id_emb = nn.Embedding(n_part_ids + 1, part_dim)
        self.class_emb = nn.Embedding(n_classes + 1, part_dim)
        self.part_proj = nn.Sequential(nn.Linear(2 * part_dim, hidden), nn.GELU())

        # 1. shared local component encoder (physics attention inside each part)
        self.local = nn.ModuleList([nn.ModuleDict({
            "n1": nn.LayerNorm(hidden), "n2": nn.LayerNorm(hidden),
            "attn": PhysicsAttention(hidden, local_heads, local_slices, dropout),
            "mlp": nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden)),
        }) for _ in range(int(local_layers))])
        self.part_pool = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))

        # 2. global component transformer over the part tokens
        layer = nn.TransformerEncoderLayer(hidden, global_heads, 2 * hidden, dropout, batch_first=True,
                                           norm_first=True)
        self.global_mixer = nn.TransformerEncoder(layer, int(global_layers), enable_nested_tensor=False)

        # 3. interface message passing on the part graph
        self.interface = nn.ModuleList([nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(),
                                                      nn.Linear(hidden, hidden)) for _ in range(int(interface_layers))]
                                      )
        self.interface_norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(int(interface_layers))])

        # 4. temporal decoder
        self.time = FourierTime(time_dim)
        d_in = 3 * hidden + time_dim                                       # node + part context + time + design context
        self.decoder = nn.ModuleList([nn.Sequential(nn.Linear(d_in, 2 * hidden), nn.GELU(),
                                                    nn.Linear(2 * hidden, d_in)) for _ in range(int(decoder_layers))])
        self.dec_norm = nn.ModuleList([nn.LayerNorm(d_in) for _ in range(int(decoder_layers))])
        self.out = nn.Linear(d_in, out_dim)

    # ---------------------------------------------------------------------------------- helpers
    @staticmethod
    def part_adjacency(x: torch.Tensor, part: torch.Tensor, tol: float | None = None) -> torch.Tensor:
        """(B, K, K) boolean graph: parts ``i`` and ``j`` are joined when their node sets come within ``tol``."""
        b, n, _ = x.shape
        k = int(part.max().item()) if part.numel() else 0
        adj = torch.zeros((b, k, k), dtype=torch.bool, device=x.device)
        if k == 0:
            return adj
        if tol is None:                                        # scale from the real nodes only: padding must not move it
            valid = part.unsqueeze(-1) > 0
            span = (x.masked_fill(~valid, float("-inf")).amax(dim=(1, 2))
                    - x.masked_fill(~valid, float("inf")).amin(dim=(1, 2)))
            tol = 0.02 * float(span.mean())
        tol = float(tol)
        d = torch.cdist(x, x)                                                  # (B, N, N), computed once
        for i in range(k):
            mi = (part == i + 1)
            for j in range(i + 1, k):
                pair = mi.unsqueeze(2) & (part == j + 1).unsqueeze(1)          # (B, N, N) node pairs of the two parts
                far = torch.where(pair, d, torch.full_like(d, float("inf"))).amin(dim=(1, 2))
                touch = far <= tol
                adj[:, i, j] = touch
                adj[:, j, i] = touch
        return adj

    def _interface(self, tokens: torch.Tensor, adj: torch.Tensor | None, token_mask: torch.Tensor) -> torch.Tensor:
        if self.interface_layers == 0 or adj is None:
            return tokens
        z = tokens
        for layer, norm in zip(self.interface, self.interface_norm, strict=False):
            msg = torch.zeros_like(z)
            edges = adj & token_mask.unsqueeze(1) & token_mask.unsqueeze(2)
            for i in range(z.shape[1]):
                m_i = edges[:, i]                                              # (B, K) neighbours of part i
                if not bool(m_i.any()):
                    continue
                src = z
                pair = torch.cat([src, src - z[:, i].unsqueeze(1), (src - z[:, i].unsqueeze(1)) ** 2], dim=-1)
                upd = layer(pair) * m_i.unsqueeze(-1).to(z.dtype)
                msg[:, i] = upd.sum(dim=1) / m_i.sum(dim=1, keepdim=True).clamp(min=1)
            z = norm(z + msg)
        return z

    # ---------------------------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, *, part: torch.Tensor | None = None, part_id: torch.Tensor | None = None,
                thickness: torch.Tensor | None = None, extra: torch.Tensor | None = None,
                time: torch.Tensor | None = None, mask: torch.Tensor | None = None,
                part_adj: torch.Tensor | None = None, return_tokens: bool = False):
        """Predict the field at the requested times.

        ``x`` (B, N, 3) coordinates; ``part`` (B, N) 1-based semantic part label (0 = padding); ``part_id``
        (B, N) dense part id; ``thickness`` (B, N); ``extra`` (B, N, in_dim - 3) further node channels (normals,
        temperature, ...); ``time`` (B, T) physical times to decode (defaults to ``n_frames`` evenly spaced times in
        [0, 1], in the same time unit the model was trained with); ``mask`` (B, N) valid points."""
        if part is None:
            part = torch.ones(x.shape[:2], dtype=torch.long, device=x.device)
        if part_id is None:
            part_id = part
        if thickness is None:
            thickness = torch.ones(x.shape[:2], dtype=torch.float32, device=x.device)
        if mask is None:
            mask = part > 0
        if extra is None:
            extra = torch.zeros((*x.shape[:2], self.node_in[0].in_features - 4), dtype=x.dtype, device=x.device)
        b, n, _ = x.shape
        feats = torch.cat([x, extra, thickness.unsqueeze(-1)], dim=-1) * mask.unsqueeze(-1).to(x.dtype)
        h = self.node_in(feats) + self.thickness(thickness.unsqueeze(-1))
        emb = self.part_proj(torch.cat([self.part_id_emb(part_id.clamp(0, self.n_part_ids)),
                                        self.class_emb(part.clamp(0, self.n_classes))], dim=-1))
        h = h + emb

        # 1. per-part local encoder
        h_pad, m_pad = _padded_parts(part, h)                                  # (B, K, N_max, H), (B, K, N_max)
        flat = h_pad.reshape(b * h_pad.shape[1], h_pad.shape[2], self.hidden)
        flat_mask = m_pad.reshape(b * m_pad.shape[1], m_pad.shape[2])
        for blk in self.local:
            z = blk["n1"](flat)
            flat = flat + blk["attn"](z, mask=flat_mask)
            flat = flat + blk["mlp"](blk["n2"](flat))
        flat = flat * flat_mask.unsqueeze(-1).to(flat.dtype)
        h_loc_pad = flat.view(*h_pad.shape)

        # part tokens: the mean of the part's nodes
        w = m_pad.unsqueeze(-1).to(h_pad.dtype)
        tokens = (h_loc_pad * w).sum(dim=2) / w.sum(dim=2).clamp(min=1.0)
        token_mask = m_pad.any(dim=2)                                          # (B, K)

        # 2. global component transformer
        tokens = self.global_mixer(tokens).masked_fill(~token_mask.unsqueeze(-1), 0.0)

        # 3. interface message passing
        if part_adj is None:
            part_adj = self.part_adjacency(x, part)
        tokens = self._interface(tokens, part_adj, token_mask)
        context = tokens.mean(dim=1)

        # node-level features: the local encoding of the node plus the mixed context of its own part
        index = (part - 1).clamp(min=0)
        ctx_node = torch.gather(tokens, 1, index.unsqueeze(-1).expand(-1, -1, self.hidden))
        _, rank = self._node_ranks(part)                                       # rank inside its part
        flat_idx = (index * h_pad.shape[2] + rank).clamp(min=0)
        h_node = torch.gather(h_pad.reshape(b, -1, self.hidden), 1,
                              flat_idx.unsqueeze(-1).expand(-1, -1, self.hidden))
        h_node = h_node * (part > 0).unsqueeze(-1).to(h.dtype)

        # 4. temporal decoder
        if time is None:
            time = torch.linspace(0.0, 1.0, self.n_frames, device=x.device).unsqueeze(0).expand(b, -1)
        t_feat = self.time(time.to(x.dtype))                                    # (B, T, time_dim)
        outs = []
        ctx_b = context.unsqueeze(1).expand(-1, n, -1)
        for ti in range(time.shape[1]):
            zt = torch.cat([h_node, ctx_node, t_feat[:, ti].unsqueeze(1).expand(-1, n, -1), ctx_b], dim=-1)
            for blk, nrm in zip(self.decoder, self.dec_norm, strict=False):
                zt = nrm(zt + blk(zt))
            outs.append(self.out(zt))
        u = torch.stack(outs, dim=1)                                            # (B, T, N, out_dim)
        u = u * (part > 0).unsqueeze(1).unsqueeze(-1).to(u.dtype)
        if return_tokens:
            return {"u": u, "part_tokens": tokens, "node_features": h_node}
        return u

    @staticmethod
    def _node_ranks(part: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(sorted order, rank of each node inside its part) — the inverse of :func:`_padded_parts`."""
        order, _, rank, _ = _group_ranks(part)
        return order, rank
