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

Meshes may differ per sample. :meth:`MeshDynamicsMGN.step` and
:meth:`MeshDynamicsMGN.rollout` process one graph (B = 1), which is what the
reference examples do; :meth:`MeshDynamicsMGN.step_batch`,
:meth:`MeshDynamicsMGN.loss_batch` and :func:`train_dynamics_batch` train and
run a batch of meshes of *different* sizes in one forward pass, by flattening
them into a disjoint union with :func:`flatten_meshes` (no edge crosses a
sample boundary, so the union is exact, not an approximation).
"""
from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .base import GraphBatch, GraphOutput
from .mesh_graph_net import MeshGraphNet
from .utils import flatten_graph_batch

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


def flatten_meshes(graphs: Sequence[MeshGraph]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack meshes of *different* sizes into one flat graph, for training with ``B > 1``.

    Nodes are concatenated (``N_total = Σ N_b``) and every ``edge_index`` gets its sample's node offset, so the result
    is the disjoint union of the samples: no edge crosses a sample boundary, and one pass of the processor over the
    union gives exactly the per-sample answers. The node types ride along as the ``x`` of
    :func:`~.utils.flatten_graph_batch` — one call, and its prefix sum is the ``ptr`` that splits the output back.

    Returns
    -------
    (edge_index, edge_attr, node_type, ptr)
        ``edge_index`` ``(2, E_total)`` with offsets applied, ``edge_attr`` ``(E_total, F)`` concatenated,
        ``node_type`` ``(N_total,)``, and ``ptr`` ``(B+1,)`` — sample ``b`` owns ``[ptr[b], ptr[b+1])``.

    Examples
    --------
    >>> a = MeshGraph.from_mesh(torch.rand(4, 2), torch.tensor([[0, 1, 2], [1, 2, 3]]), torch.zeros(4))
    >>> b = MeshGraph.from_mesh(torch.rand(6, 2), torch.tensor([[0, 1, 2], [3, 4, 5]]), torch.zeros(6))
    >>> ei, ea, nt, ptr = flatten_meshes([a, b])
    >>> ptr.tolist(), ei.max().item()
    ([0, 4, 10], 9)
    """
    if not graphs:
        raise ValueError("flatten_meshes needs at least one mesh")
    x_flat, edge_index_flat, _pos, edge_attr_flat, ptr = flatten_graph_batch(
        x_list=[g.node_type.view(-1, 1) for g in graphs],
        edge_index_list=[g.edge_index.long() for g in graphs],
        edge_attr_list=[g.edge_attr for g in graphs],
    )
    return edge_index_flat, edge_attr_flat, x_flat.view(-1).long(), ptr


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

    Notes
    -----
    ``capabilities`` declares what the model asks for and gives back, so the
    registry (and ``pp.list_models()``) reports it instead of the family
    default: the input is a *static mesh plus a trajectory*, the output is a
    *rollout*, not a per-node field on a shareable topology.
    """

    capabilities = {
        "input_kind": "graph_trajectory",       # a static mesh, not a per-sample point cloud
        "expects": ["mesh", "trajectory"],      # keys of forward_batch(): MeshGraph(s) + (T, N, vel_dim)
        "predicts": ["rollout"],                # (n_steps + 1, N, vel_dim) trajectories (+ pressure)
        "supports_physics_loss": False,
    }

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
    def _node_input(self, vel: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
        """[normalised velocity, one-hot node type] → (N, vel_dim + num_node_types)."""
        onehot = torch.nn.functional.one_hot(node_type, self.num_node_types).to(vel.dtype)
        return torch.cat([self.vel_norm(vel), onehot], dim=-1)

    def _predict_normalised(self, vel: torch.Tensor, g: MeshGraph) -> torch.Tensor:
        x = self._node_input(vel, g.node_type)[None]
        e = self.edge_norm(g.edge_attr)[None]
        return self.net(GraphBatch(x=x, edge_index=g.edge_index, edge_attr=e)).y[0]

    def _predict_flat(self, node_inputs: Sequence[torch.Tensor],
                      graphs: Sequence[MeshGraph]) -> tuple[torch.Tensor, torch.Tensor]:
        """One processor pass over the disjoint union of ``graphs`` → ``(out (N_total, out_dim), ptr)``."""
        edge_index, edge_attr, _node_type, ptr = flatten_meshes(graphs)
        batch = GraphBatch(x=torch.cat(node_inputs, dim=0)[None], edge_index=edge_index,
                           edge_attr=self.edge_norm(edge_attr)[None])
        return self.net(batch).y[0], ptr

    @staticmethod
    def _split(out: torch.Tensor, ptr: torch.Tensor) -> list[torch.Tensor]:
        return [out[ptr[b].item():ptr[b + 1].item()] for b in range(ptr.numel() - 1)]

    def _to_fields(self, out: torch.Tensor, vel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(Δv, p) in physical units → next velocity and pressure (or ``None``)."""
        next_vel = vel + self.dvel_norm.inverse(out[:, : self.vel_dim])
        p = self.p_norm.inverse(out[:, self.vel_dim:]) if self.out_p else None
        return next_vel, p

    def step(self, vel: torch.Tensor, g: MeshGraph):
        """vel (N, vel_dim) → (next_vel, pressure or None)."""
        out = self._predict_normalised(vel, g)
        return self._to_fields(out, vel)

    def step_batch(self, vels: Sequence[torch.Tensor],
                   graphs: Sequence[MeshGraph]) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
        """One step for a batch of meshes of *different* sizes, in a single forward pass.

        ``vels[b]`` is ``(N_b, vel_dim)`` and ``graphs[b]`` its mesh; the meshes need not share a topology or a node
        count. Returns ``(next_vels, pressures)``, one entry per sample — the same numbers :meth:`step` gives one
        sample at a time (the union has no edge between samples, so nothing mixes).
        """
        if len(vels) != len(graphs):
            raise ValueError(f"got {len(vels)} velocity blocks for {len(graphs)} meshes")
        if not graphs:
            raise ValueError("step_batch needs at least one mesh")
        out, ptr = self._predict_flat([self._node_input(v, g.node_type) for v, g in zip(vels, graphs, strict=True)],
                                      graphs)
        next_vels, pressures = [], []
        for vel, block in zip(vels, self._split(out, ptr), strict=True):
            nxt, p = self._to_fields(block, vel)
            next_vels.append(nxt)
            pressures.append(p)
        return next_vels, pressures

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

    def loss_batch(self, vel_t: Sequence[torch.Tensor], vel_t1: Sequence[torch.Tensor],
                   p_t1: Sequence[torch.Tensor] | None, graphs: Sequence[MeshGraph],
                   noise_std: float = 0.02, normal_type: int = 0) -> torch.Tensor:
        """The one-step training loss for a batch of trajectories on meshes of different sizes.

        Same target convention as :meth:`loss` (noise on NORMAL nodes, Δv shifted so ``v_in + Δv = v_{t+1}``), but
        the noise is drawn per sample against that sample's own node-type mask and the whole batch goes through the
        processor once. The mean is over all nodes of all samples, so a finer mesh carries proportionally more of the
        gradient — the same weighting the single-mesh loss already has over its nodes.
        """
        if len(vel_t) != len(graphs) or len(vel_t1) != len(graphs):
            raise ValueError("loss_batch needs one velocity block pair per mesh")
        if p_t1 is not None and len(p_t1) != len(graphs):
            raise ValueError("loss_batch needs one pressure block per mesh (or None)")
        if self.out_p and (p_t1 is None or any(p is None for p in p_t1)):
            raise ValueError("loss_batch: out_p=True needs a pressure block per mesh")
        if not graphs:
            raise ValueError("loss_batch needs at least one mesh")
        vels_in, targets = [], []
        for b, (v_t, v_t1, g) in enumerate(zip(vel_t, vel_t1, graphs, strict=True)):
            noise = torch.zeros_like(v_t)
            if noise_std > 0:
                noise = torch.randn_like(v_t) * noise_std
                noise = noise * (g.node_type == normal_type).unsqueeze(-1).to(v_t.dtype)
            vel_in = v_t + noise
            dv_target = self.dvel_norm(v_t1 - vel_in)
            target = dv_target if not self.out_p else torch.cat([dv_target, self.p_norm(p_t1[b])], -1)
            vels_in.append(vel_in)
            targets.append(target)
        node_inputs = [self._node_input(v, g.node_type) for v, g in zip(vels_in, graphs, strict=True)]
        out, _ptr = self._predict_flat(node_inputs, graphs)
        return torch.mean((out - torch.cat(targets, dim=0)) ** 2)

    # -- rollout -------------------------------------------------------------
    @staticmethod
    def _free_mask(node_type: torch.Tensor, free_types: Sequence[int]) -> torch.Tensor:
        free = torch.zeros_like(node_type, dtype=torch.bool)
        for t in free_types:
            free |= node_type == t
        return free.unsqueeze(-1)

    @torch.no_grad()
    def rollout(self, vel_traj: torch.Tensor, g: MeshGraph, n_steps: int,
                free_types: Sequence[int] = (0,)):
        """Free-running rollout from ``vel_traj[0]``.

        Nodes whose type is not in ``free_types`` are overwritten each step by
        the ground-truth (prescribed) value ``vel_traj[t+1]``.

        Returns ``(vel_pred (n_steps+1, N, V), p_pred (n_steps, N, 1) or None)``.
        """
        free = self._free_mask(g.node_type, free_types)
        vel = vel_traj[0]
        vels, ps = [vel], []
        for t in range(n_steps):
            nxt, p = self.step(vel, g)
            vel = torch.where(free, nxt, vel_traj[t + 1])
            vels.append(vel)
            if p is not None:
                ps.append(p)
        return torch.stack(vels), (torch.stack(ps) if ps else None)

    @torch.no_grad()
    def rollout_batch(self, vel_trajs: Sequence[torch.Tensor], graphs: Sequence[MeshGraph], n_steps: int,
                      free_types: Sequence[int] = (0,)) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
        """Free-running rollouts for a batch of trajectories on meshes of different sizes.

        ``vel_trajs[b]`` is ``(T, N_b, vel_dim)`` with ``T >= n_steps + 1``; the prescribed nodes of each sample are
        clamped to that sample's own trajectory, as in :meth:`rollout`. Returns ``(vels, pressures)`` with
        ``vels[b]`` ``(n_steps + 1, N_b, vel_dim)`` and ``pressures[b]`` ``(n_steps, N_b, 1)`` or ``None``.
        """
        if len(vel_trajs) != len(graphs):
            raise ValueError(f"got {len(vel_trajs)} trajectories for {len(graphs)} meshes")
        free = [self._free_mask(g.node_type, free_types) for g in graphs]
        vels = [[traj[0]] for traj in vel_trajs]
        ps: list[list[torch.Tensor]] = [[] for _ in graphs]
        for t in range(n_steps):
            nxt, p = self.step_batch([v[-1] for v in vels], graphs)
            for b in range(len(graphs)):
                vels[b].append(torch.where(free[b], nxt[b], vel_trajs[b][t + 1]))
                if p[b] is not None:
                    ps[b].append(p[b])
        return ([torch.stack(v) for v in vels],
                [torch.stack(p) if p else None for p in ps])

    # -- Arena / GNNAdapter interface ----------------------------------------
    def forward_batch(self, batch: Mapping[str, Any]) -> GraphOutput:
        """Dict-based interface used by the Arena / :class:`GNNAdapter`.

        Accepted keys
        -------------
        ``mesh`` / ``mesh_list`` / ``graph`` / ``graphs``
            One :class:`MeshGraph` or a list of them — of different sizes. Aligned with ``trajectory`` /
            ``trajectory_list`` when both are lists.
        ``trajectory`` / ``velocity`` / ``trajectories``
            ``(T, N, vel_dim)`` for one mesh, or a list of such blocks, one per mesh.
        ``pressure`` / ``pressures`` / ``p``
            ``(T, N, 1)`` blocks, required when ``mode="loss"`` and ``out_p``.
        ``mode`` / ``task``
            ``"rollout"`` (default): free-running rollout of ``n_steps``; ``"step"``: one step from ``trajectory[0]``;
            ``"loss"``: the one-step training loss.
        ``n_steps``, ``free_types``, ``noise_std``, ``normal_type``
            Forwarded to the corresponding method.
        ``y_true`` / ``y``
            Optional rollout target, ``(n_steps + 1, N, vel_dim)`` (or one per mesh) for the padded MSE.

        Returns
        -------
        GraphOutput
            ``y`` padded to ``(B, n_steps + 1, N_max, vel_dim)`` for a rollout (``(B, N_max, vel_dim)`` for a step),
            the loss in ``losses`` when ``y_true`` is given, and the exact per-sample blocks in
            ``extras["vel_list"]`` / ``extras["p_list"]`` next to ``extras["ptr"]`` and ``extras["mask"]``.
        """
        graphs = _as_blocks(_first(batch, ("mesh", "mesh_list", "graph", "graphs")))
        trajs = _as_blocks(_first(batch, ("trajectory", "velocity", "trajectories", "trajectory_list")))
        pressures = _as_blocks(_first(batch, ("pressure", "pressures", "p")))
        mode = str(batch.get("mode", batch.get("task", "rollout"))).lower()
        if not graphs or not trajs:
            raise KeyError("forward_batch needs 'mesh'/'graph'/'graphs' and 'trajectory'/'velocity'")
        if len(trajs) == 1 and len(graphs) > 1:
            trajs = trajs * len(graphs)
        if len(pressures) == 1 and len(graphs) > 1:
            pressures = pressures * len(graphs)
        if len(trajs) != len(graphs):
            raise ValueError(f"got {len(trajs)} trajectories for {len(graphs)} meshes")
        normal_type = int(batch.get("normal_type", 0))

        if mode == "loss":
            if self.out_p and (not pressures or any(p is None for p in pressures)):
                raise KeyError("mode='loss' with out_p=True needs 'pressure' blocks: (T, N, 1) per mesh")
            p_t1 = [p[1] for p in pressures] if pressures else None
            loss = self.loss_batch([t[0] for t in trajs], [t[1] for t in trajs], p_t1, graphs,
                                   noise_std=float(batch.get("noise_std", 0.02)), normal_type=normal_type)
            n_max = max(g.node_type.numel() for g in graphs)
            return GraphOutput(y=trajs[0].new_zeros((len(graphs), n_max, self.vel_dim)),
                               losses={"total": loss, "mse": loss}, extras={"ptr": None, "mask": None})

        if mode == "step":
            vels, ps = self.step_batch([t[0] for t in trajs], graphs)
            y, mask = _pad_blocks(vels)
        elif mode == "rollout":
            n_steps = int(batch.get("n_steps", min(t.shape[0] - 1 for t in trajs)))
            vels, ps = self.rollout_batch(trajs, graphs, n_steps, tuple(batch.get("free_types", (normal_type,))))
            y, mask = _pad_blocks(vels, node_axis=1)
        else:
            raise ValueError(f"mode must be 'rollout', 'step' or 'loss'; got {mode!r}")

        losses: dict[str, torch.Tensor] = {"total": y.new_zeros(())}
        target = _first(batch, ("y_true", "y"))
        if target is not None:
            y_true, _ = _pad_blocks(_as_blocks(target), node_axis=1 if mode == "rollout" else 0)
            if y_true.shape != y.shape:
                raise ValueError(f"y_true {tuple(y_true.shape)} does not match the prediction {tuple(y.shape)}")
            m = mask[..., None].to(y.dtype)
            losses["mse"] = (((y - y_true) ** 2) * m).sum() / (m.sum() * y.shape[-1]).clamp_min(1.0)
            losses["total"] = losses["mse"]
        return GraphOutput(y=y, losses=losses,
                           extras={"vel_list": vels, "p_list": ps, "mask": mask, "ptr": None})


def _first(batch: Mapping[str, Any], names: Sequence[str]):
    """The first key of ``names`` present (and not ``None``) in ``batch``."""
    for name in names:
        if batch.get(name) is not None:
            return batch[name]
    return None


def _as_blocks(value) -> list:
    """``None`` → ``[]``; a single object → ``[object]``; a sequence of blocks stays a list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _pad_blocks(blocks: Sequence[torch.Tensor], *, node_axis: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad per-sample blocks along their node axis into one tensor plus a ``(B, N_max)`` validity mask.

    ``node_axis=0`` for ``(N_b, ...)`` blocks (one step), ``1`` for ``(T, N_b, ...)`` ones (a rollout); the result is
    ``(B, *shape with the node axis at N_max)``.
    """
    shapes = {tuple(s for i, s in enumerate(b.shape) if i != node_axis) for b in blocks}
    if len(shapes) != 1:
        raise ValueError(f"blocks disagree outside the node axis: {sorted(shapes)}")
    n_max = max(b.shape[node_axis] for b in blocks)
    shape = list(blocks[0].shape)
    shape[node_axis] = n_max
    out = blocks[0].new_zeros((len(blocks), *shape))
    mask = blocks[0].new_zeros((len(blocks), n_max))
    for b, block in enumerate(blocks):
        sel = [slice(0, block.shape[node_axis]) if i == node_axis else slice(None) for i in range(len(shape))]
        out[(b, *sel)] = block
        mask[b, :block.shape[node_axis]] = 1.0
    return out, mask


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_dynamics(model: MeshDynamicsMGN, data: Sequence[tuple[MeshGraph, torch.Tensor, torch.Tensor]],
                   *, steps: int, lr: float = 1e-4, lr_final: float = 1e-6,
                   noise_std: float = 0.02, device: torch.device | str = "cpu", log_every: int = 200,
                   seed: int = 0) -> list[float]:
    """One-step training on random ``(trajectory, time)`` pairs with Adam and exponential LR decay.

    ``data`` holds ``(graph, velocity (T, N, V), pressure (T, N, 1))`` triples; the LR decays from ``lr`` to
    ``lr_final`` over ``steps`` (the schedule of Pfaff et al. 2021 and of PhysicsNeMo ``vortex_shedding_mgn``).
    Call :meth:`MeshDynamicsMGN.fit_stats` first. Returns the mean loss of every ``log_every`` steps.
    """
    rng = np.random.default_rng(seed)
    model.to(device).train()
    data = [(g.to(device), v.to(device), p.to(device)) for g, v, p in data]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gamma = (lr_final / lr) ** (1.0 / max(steps, 1))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    hist, run, t0 = [], 0.0, time.time()
    for it in range(1, steps + 1):
        g, v, p = data[rng.integers(len(data))]
        t = int(rng.integers(v.shape[0] - 1))
        loss = model.loss(v[t], v[t + 1], p[t + 1], g, noise_std=noise_std)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run += loss.item()
        if it % log_every == 0:
            hist.append(run / log_every)
            print(f"  step {it:6d}/{steps}  loss {hist[-1]:.4e}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{time.time() - t0:6.0f}s", flush=True)
            run = 0.0
    return hist


def train_dynamics_batch(model: MeshDynamicsMGN, data: Sequence[tuple[MeshGraph, torch.Tensor, torch.Tensor]],
                         *, steps: int, lr: float = 1e-4, lr_final: float = 1e-6, batch_size: int = 2,
                         noise_std: float = 0.02, device: torch.device | str = "cpu", log_every: int = 200,
                         seed: int = 0) -> list[float]:
    """The same schedule as :func:`train_dynamics`, with ``batch_size`` trajectories per step on meshes of any sizes.

    Every iteration draws ``batch_size`` distinct ``(graph, trajectory, time)`` triples and takes one
    :meth:`MeshDynamicsMGN.loss_batch` step, so the gradient is an average over samples whose meshes need not share
    a topology or a node count — the batching through :func:`flatten_meshes`. ``batch_size=1`` reproduces
    :func:`train_dynamics` (up to the noise draws); ``steps`` counts optimiser steps, not samples, so a run of the
    same ``steps`` costs the same wall time per step as ``B = 1`` in the processor but sees ``batch_size`` times more
    data. Returns the mean loss of every ``log_every`` steps.
    """
    rng = np.random.default_rng(seed)
    size = max(int(batch_size), 1)
    if size > len(data):
        raise ValueError(f"batch_size={size} exceeds the {len(data)} trajectories available")
    model.to(device).train()
    data = [(g.to(device), v.to(device), p.to(device)) for g, v, p in data]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gamma = (lr_final / lr) ** (1.0 / max(steps, 1))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    hist, run, t0 = [], 0.0, time.time()
    for it in range(1, steps + 1):
        pick = rng.choice(len(data), size=size, replace=False)
        graphs, v_t, v_t1, p_t1 = [], [], [], []
        for b in pick:
            g, v, p = data[b]
            t = int(rng.integers(v.shape[0] - 1))
            graphs.append(g)
            v_t.append(v[t])
            v_t1.append(v[t + 1])
            p_t1.append(p[t + 1])
        loss = model.loss_batch(v_t, v_t1, p_t1, graphs, noise_std=noise_std)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run += loss.item()
        if it % log_every == 0:
            hist.append(run / log_every)
            print(f"  step {it:6d}/{steps}  loss {hist[-1]:.4e}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"B={size}  {time.time() - t0:6.0f}s", flush=True)
            run = 0.0
    return hist


@torch.no_grad()
def eval_rollout(model: MeshDynamicsMGN, data: Sequence[tuple[MeshGraph, torch.Tensor, torch.Tensor]], *,
                 n_steps: int, device: torch.device | str = "cpu", normal_type: int = 0) -> dict[str, float]:
    """Velocity RMSE on NORMAL nodes: mean over trajectories of the ``n_steps`` free rollout and of one step.

    Returns ``{"rollout_rmse", "one_step_rmse", "n_steps"}``.
    """
    model.to(device).eval()
    roll, one = [], []
    for g, v, _p in data:
        g, v = g.to(device), v.to(device)
        n = min(n_steps, v.shape[0] - 1)
        pred, _ = model.rollout(v, g, n, free_types=(normal_type,))
        free = g.node_type == normal_type
        roll.append(((pred[1:] - v[1 : n + 1])[:, free] ** 2).mean().sqrt().item())
        nxt, _ = model.step(v[0], g)
        one.append(((nxt - v[1])[free] ** 2).mean().sqrt().item())
    return {"rollout_rmse": float(np.mean(roll)), "one_step_rmse": float(np.mean(one)), "n_steps": n_steps}


def frozen_ic_rmse(data: Sequence[tuple[MeshGraph, torch.Tensor, torch.Tensor]], *, n_steps: int,
                   normal_type: int = 0) -> float:
    """Baseline for :func:`eval_rollout`: velocity RMSE on NORMAL nodes when the velocity stays at its t = 0 value
    over the same horizon, mean over trajectories."""
    base = []
    for g, v, _p in data:
        free = g.node_type == normal_type
        n = min(n_steps, v.shape[0] - 1)
        base.append(((v[0:1] - v[1 : n + 1])[:, free] ** 2).mean().sqrt().item())
    return sum(base) / len(base)
