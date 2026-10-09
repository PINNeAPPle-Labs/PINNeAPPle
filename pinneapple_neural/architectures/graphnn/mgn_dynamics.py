"""Transient mesh dynamics on top of :class:`MeshGraphNet`.

Reproduces the training/inference recipe of Pfaff et al. (ICLR 2021) and of
the PhysicsNeMo ``vortex_shedding_mgn`` example:

* node inputs  = [velocity_t, one-hot node type]   (normalised)
* edge inputs  = [rel. position, |rel. position|]  (normalised, static)
* node outputs = [Δvelocity, pressure_{t+1}]       (normalised)
* training-time noise on the velocity of NORMAL nodes, with the Δv target
  corrected so the net learns to undo the noise
* free-running rollout where non-NORMAL nodes (walls, inlet, obstacle) are
  clamped to their prescribed values.

Meshes may differ per sample; each call processes one graph (B = 1), which is
what the reference examples do as well.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

from .base import GraphBatch
from .mesh_graph_net import MeshGraphNet

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def triangles_to_edges(cells: torch.Tensor) -> torch.Tensor:
    """Bidirectional, de-duplicated edge_index (2, E) from triangle ``cells`` (T, 3)."""
    cells = cells.long()
    e = torch.cat([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]], dim=0)
    e = torch.cat([e, e.flip(1)], dim=0)
    e = torch.unique(e, dim=0)
    return e.t().contiguous()


def edge_features(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Relative displacement (dst - src) and its norm → (E, pos_dim + 1)."""
    rel = pos[edge_index[1]] - pos[edge_index[0]]
    return torch.cat([rel, rel.norm(dim=-1, keepdim=True)], dim=-1)


@dataclass
class MeshGraph:
    """One static mesh: topology, raw edge features and node types."""
    edge_index: torch.Tensor          # (2, E)
    edge_attr: torch.Tensor           # (E, F) raw (un-normalised)
    node_type: torch.Tensor           # (N,) integer ids

    @classmethod
    def from_mesh(cls, pos: torch.Tensor, cells: torch.Tensor,
                  node_type: torch.Tensor) -> MeshGraph:
        ei = triangles_to_edges(cells)
        return cls(ei, edge_features(pos, ei), node_type.long().view(-1))

    def to(self, device) -> MeshGraph:
        return MeshGraph(self.edge_index.to(device), self.edge_attr.to(device),
                         self.node_type.to(device))


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------

class Normalizer(nn.Module):
    """Per-feature standardiser with fixed statistics (stored as buffers)."""

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    @torch.no_grad()
    def fit(self, tensors: Iterable[torch.Tensor]) -> Normalizer:
        n, s, s2 = 0, 0.0, 0.0
        for t in tensors:
            t = t.reshape(-1, t.shape[-1]).double()
            n += t.shape[0]
            s = s + t.sum(0)
            s2 = s2 + (t ** 2).sum(0)
        mean = s / n
        std = (s2 / n - mean ** 2).clamp_min(0).sqrt().clamp_min(self.eps)
        self.mean.copy_(mean.float())
        self.std.copy_(std.float())
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


# ---------------------------------------------------------------------------
# Dynamics wrapper
# ---------------------------------------------------------------------------

class MeshDynamicsMGN(nn.Module):
    """Autoregressive MeshGraphNet for fields ``(vel, p)`` on a transient mesh.

    Parameters
    ----------
    vel_dim:        velocity components (2 for 2-D).
    num_node_types: size of the one-hot node-type encoding.
    edge_dim:       raw edge feature size (``pos_dim + 1``).
    out_p:          also predict a scalar pressure (output = vel_dim + 1).
    mgn_kwargs:     forwarded to :class:`MeshGraphNet` (hidden_dim, n_layers,
                    n_message_passing, activation, ...).
    """

    def __init__(self, vel_dim: int = 2, num_node_types: int = 4,
                 edge_dim: int = 3, out_p: bool = True, **mgn_kwargs) -> None:
        super().__init__()
        self.vel_dim, self.num_node_types, self.out_p = vel_dim, num_node_types, out_p
        out_dim = vel_dim + (1 if out_p else 0)
        self.net = MeshGraphNet(vel_dim + num_node_types, out_dim,
                                edge_in_dim=edge_dim, **mgn_kwargs)
        self.vel_norm = Normalizer(vel_dim)
        self.dvel_norm = Normalizer(vel_dim)
        self.p_norm = Normalizer(1)
        self.edge_norm = Normalizer(edge_dim)

    # -- statistics ---------------------------------------------------------
    @torch.no_grad()
    def fit_stats(self, velocities: Sequence[torch.Tensor],
                  pressures: Sequence[torch.Tensor] | None,
                  graphs: Sequence[MeshGraph]) -> None:
        """``velocities``: list of (T, N, vel_dim); ``pressures``: (T, N, 1)."""
        self.vel_norm.fit(v[:-1] for v in velocities)
        self.dvel_norm.fit(v[1:] - v[:-1] for v in velocities)
        if pressures is not None:
            self.p_norm.fit(p[1:] for p in pressures)
        self.edge_norm.fit(g.edge_attr for g in graphs)

    # -- single step ---------------------------------------------------------
    def _predict_normalised(self, vel: torch.Tensor, g: MeshGraph) -> torch.Tensor:
        onehot = torch.nn.functional.one_hot(g.node_type, self.num_node_types).to(vel.dtype)
        x = torch.cat([self.vel_norm(vel), onehot], dim=-1)[None]
        e = self.edge_norm(g.edge_attr)[None]
        return self.net(GraphBatch(x=x, edge_index=g.edge_index, edge_attr=e)).y[0]

    def step(self, vel: torch.Tensor, g: MeshGraph):
        """vel (N, vel_dim) → (next_vel, pressure or None)."""
        out = self._predict_normalised(vel, g)
        dv = self.dvel_norm.inverse(out[:, : self.vel_dim])
        p = self.p_norm.inverse(out[:, self.vel_dim:]) if self.out_p else None
        return vel + dv, p

    # -- training loss -------------------------------------------------------
    def loss(self, vel_t: torch.Tensor, vel_t1: torch.Tensor,
             p_t1: torch.Tensor | None, g: MeshGraph,
             noise_std: float = 0.02, normal_type: int = 0) -> torch.Tensor:
        """One-step MSE in normalised space, with PhysicsNeMo-style noise.

        The input velocity of NORMAL nodes is perturbed by ``noise_std`` and the
        Δv target is shifted by the same noise so that ``v_in + Δv = v_{t+1}``.
        """
        if noise_std > 0:
            noise = torch.randn_like(vel_t) * noise_std
            noise = noise * (g.node_type == normal_type).unsqueeze(-1).to(vel_t.dtype)
        else:
            noise = torch.zeros_like(vel_t)
        vel_in = vel_t + noise
        dv_target = self.dvel_norm(vel_t1 - vel_in)
        target = dv_target if not self.out_p else torch.cat([dv_target, self.p_norm(p_t1)], -1)
        pred = self._predict_normalised(vel_in, g)
        return torch.mean((pred - target) ** 2)

    # -- rollout -------------------------------------------------------------
    @torch.no_grad()
    def rollout(self, vel_traj: torch.Tensor, g: MeshGraph, n_steps: int,
                free_types: Sequence[int] = (0,)):
        """Free-running rollout from ``vel_traj[0]``.

        Nodes whose type is not in ``free_types`` are overwritten each step by
        the ground-truth (prescribed) value ``vel_traj[t+1]``.

        Returns ``(vel_pred (n_steps+1, N, V), p_pred (n_steps, N, 1) or None)``.
        """
        free = torch.zeros_like(g.node_type, dtype=torch.bool)
        for t in free_types:
            free |= g.node_type == t
        free = free.unsqueeze(-1)
        vel = vel_traj[0]
        vels, ps = [vel], []
        for t in range(n_steps):
            nxt, p = self.step(vel, g)
            vel = torch.where(free, nxt, vel_traj[t + 1])
            vels.append(vel)
            if p is not None:
                ps.append(p)
        return torch.stack(vels), (torch.stack(ps) if ps else None)
