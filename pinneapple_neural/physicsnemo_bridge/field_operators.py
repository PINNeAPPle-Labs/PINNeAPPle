"""PhysicsNeMo field operators on a structured grid -> PINNeAPPle predictors (pattern 1: PhysicsNeMo trains,
PINNeAPPle checks).

A neural operator maps a coefficient field on an ``n x n`` grid to a solution field on the same grid (Darcy,
heat conduction with a variable conductivity, ...). This module builds such a model from PhysicsNeMo's own
implementations, trains it under a :class:`pinneapple_analysis.cost.equal_budget.ComputeBudget`, and exposes it as a
NumPy predictor ``predict(a) -> u`` that PINNeAPPle's checks (``pinneapple_physics.darcy_checks``, ``pp.trust_card``,
``pinneapple check``) consume. Nothing here re-implements a PhysicsNeMo model:

``physicsnemo_fno(n, ...)``
    ``physicsnemo.models.fno.FNO`` (2-D, one input channel, coordinate features added by PhysicsNeMo).
``physicsnemo_meshgraphnet(n, ...)``
    ``physicsnemo.models.meshgraphnet.MeshGraphNet`` on the graph of the grid (:func:`grid_graph`: one node per
    cell, edges to the four neighbours, edge features ``dx, dy, |d|`` in cell units; node features: normalised
    coefficient, cell-centre coordinates, boundary flag). Needs PyTorch Geometric and ``torch_scatter``, as
    PhysicsNeMo's MeshGraphNet does.
``GridFieldPredictor``
    the trained model plus its input / output normalisation; ``predict`` takes and returns NumPy arrays
    ``(batch, n, n)``; ``save`` / ``load`` keep the PhysicsNeMo constructor arguments with the weights.
``train_grid_operator``
    supervised training (per-sample relative L2 loss, Adam, cosine learning rate over the budget) driven by
    ``train_within_budget``, so two models trained with the same ``ComputeBudget`` got the same compute.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from pinneapple_neural.physicsnemo_bridge.adapters import _require_physicsnemo

__all__ = ["grid_graph", "GridFieldPredictor", "physicsnemo_fno", "physicsnemo_meshgraphnet",
           "train_grid_operator", "TrainingResult"]


def grid_graph(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Graph of an ``n x n`` cell-centred grid: ``edge_index (2, E)`` (both directions, node ``i * n + j``) and
    ``edge_attr (E, 3)`` = ``(dx, dy, |d|)`` of each edge in cell units."""
    idx = np.arange(n * n).reshape(n, n)
    src = [idx[:-1, :].ravel(), idx[1:, :].ravel(), idx[:, :-1].ravel(), idx[:, 1:].ravel()]
    dst = [idx[1:, :].ravel(), idx[:-1, :].ravel(), idx[:, 1:].ravel(), idx[:, :-1].ravel()]
    m = n * (n - 1)
    d = [np.tile([1.0, 0.0], (m, 1)), np.tile([-1.0, 0.0], (m, 1)), np.tile([0.0, 1.0], (m, 1)),
         np.tile([0.0, -1.0], (m, 1))]
    ei = np.stack([np.concatenate(src), np.concatenate(dst)])
    dxy = np.concatenate(d)
    ea = np.concatenate([dxy, np.linalg.norm(dxy, axis=1, keepdims=True)], axis=1)
    return ei.astype(np.int64), ea.astype(np.float32)


class _GridOperator(nn.Module):
    """``(batch, n, n)`` coefficient -> ``(batch, n, n)`` solution around a PhysicsNeMo model."""

    def __init__(self, net: nn.Module, kind: str, n: int, a_mean: float, a_std: float, u_scale: float):
        super().__init__()
        self.net, self.kind, self.n = net, kind, int(n)
        self.register_buffer("a_mean", torch.tensor(float(a_mean)))
        self.register_buffer("a_std", torch.tensor(float(a_std)))
        self.register_buffer("u_scale", torch.tensor(float(u_scale)))
        if kind == "meshgraphnet":
            ei, ea = grid_graph(n)
            self.register_buffer("edge_index", torch.from_numpy(ei), persistent=False)
            self.register_buffer("edge_attr", torch.from_numpy(ea), persistent=False)
            c = (np.arange(n) + 0.5) / n
            X, Y = np.meshgrid(c, c, indexing="ij")
            bnd = np.zeros((n, n))
            bnd[0, :] = bnd[-1, :] = bnd[:, 0] = bnd[:, -1] = 1.0
            static = np.stack([X, Y, bnd], axis=-1).reshape(n * n, 3).astype(np.float32)
            self.register_buffer("static", torch.from_numpy(static), persistent=False)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        x = (a - self.a_mean) / self.a_std
        b = x.shape[0]
        if self.kind == "fno":
            y = self.net(x[:, None])[:, 0]
        else:
            from torch_geometric.data import Data
            nn_ = self.n * self.n
            node = torch.cat([x.reshape(b, nn_, 1), self.static.expand(b, nn_, 3)], dim=-1).reshape(b * nn_, 4)
            off = (torch.arange(b, device=a.device) * nn_).repeat_interleave(self.edge_index.shape[1])
            ei = self.edge_index.repeat(1, b) + off[None]
            ea = self.edge_attr.repeat(b, 1)
            y = self.net(node, ea, Data(edge_index=ei, num_nodes=b * nn_)).reshape(b, self.n, self.n)
        return y * self.u_scale


@dataclass
class GridFieldPredictor:
    """A trained PhysicsNeMo model on an ``n x n`` grid as a NumPy predictor ``predict(a) -> u``."""
    module: _GridOperator
    config: dict[str, Any]
    name: str = ""

    @property
    def kind(self) -> str:
        return self.module.kind

    @property
    def n(self) -> int:
        return self.module.n

    @property
    def physicsnemo_model(self) -> nn.Module:
        """The underlying ``physicsnemo`` model instance."""
        return self.module.net

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.module.net.parameters()))

    def fit_normalisation(self, a: np.ndarray, u: np.ndarray) -> None:
        """Input mean / std and output scale from the training data."""
        self.module.a_mean.fill_(float(np.mean(a)))
        self.module.a_std.fill_(float(np.std(a)) or 1.0)
        self.module.u_scale.fill_(float(np.std(u)) or 1.0)

    @torch.no_grad()
    def predict(self, a: np.ndarray, batch_size: int = 16) -> np.ndarray:
        a = np.asarray(a, dtype=np.float32)
        single = a.ndim == 2
        a = a[None] if single else a
        if a.shape[-2:] != (self.n, self.n):
            raise ValueError(f"this model works on {self.n} x {self.n} grids, got {a.shape[-2:]}")
        self.module.eval()
        out = [self.module(torch.from_numpy(a[i:i + batch_size])).numpy() for i in range(0, len(a), batch_size)]
        u = np.concatenate(out).astype(np.float64)
        return u[0] if single else u

    __call__ = predict

    def save(self, path: str | Path) -> Path:
        """Weights (``.pt``, a plain state dict) and ``.json`` with the constructor arguments."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.module.state_dict(), path.with_suffix(".pt"))
        path.with_suffix(".json").write_text(json.dumps({"name": self.name, **self.config}, indent=1))
        return path.with_suffix(".pt")

    @classmethod
    def load(cls, path: str | Path) -> GridFieldPredictor:
        path = Path(path)
        cfg = json.loads(path.with_suffix(".json").read_text())
        name = cfg.pop("name", "")
        kind = cfg.pop("kind")
        build = physicsnemo_fno if kind == "fno" else physicsnemo_meshgraphnet
        pred = build(**cfg)
        pred.name = name
        pred.module.load_state_dict(torch.load(path.with_suffix(".pt"), weights_only=True))
        return pred


def physicsnemo_fno(n: int, *, latent_channels: int = 32, num_fno_layers: int = 4, num_fno_modes: int = 12,
                    decoder_layers: int = 1, decoder_layer_size: int = 64, padding: int = 8,
                    seed: int | None = 0) -> GridFieldPredictor:
    """A 2-D ``physicsnemo.models.fno.FNO`` (coefficient -> solution) wrapped as a :class:`GridFieldPredictor`."""
    _require_physicsnemo()
    from physicsnemo.models.fno import FNO
    if seed is not None:
        torch.manual_seed(seed)
    cfg = dict(latent_channels=latent_channels, num_fno_layers=num_fno_layers, num_fno_modes=num_fno_modes,
               decoder_layers=decoder_layers, decoder_layer_size=decoder_layer_size, padding=padding)
    net = FNO(in_channels=1, out_channels=1, dimension=2, coord_features=True, **cfg)
    return GridFieldPredictor(_GridOperator(net, "fno", n, 0.0, 1.0, 1.0),
                              {"kind": "fno", "n": n, "seed": seed, **cfg}, "PhysicsNeMo FNO")


def physicsnemo_meshgraphnet(n: int, *, processor_size: int = 15, hidden_dim: int = 64,
                             aggregation: str = "sum", seed: int | None = 0) -> GridFieldPredictor:
    """A ``physicsnemo.models.meshgraphnet.MeshGraphNet`` on the graph of the ``n x n`` grid (:func:`grid_graph`)."""
    _require_physicsnemo()
    from physicsnemo.models.meshgraphnet import MeshGraphNet
    if seed is not None:
        torch.manual_seed(seed)
    net = MeshGraphNet(input_dim_nodes=4, input_dim_edges=3, output_dim=1, processor_size=processor_size,
                       hidden_dim_processor=hidden_dim, hidden_dim_node_encoder=hidden_dim,
                       hidden_dim_edge_encoder=hidden_dim, hidden_dim_node_decoder=hidden_dim,
                       aggregation=aggregation)
    cfg = {"kind": "meshgraphnet", "n": n, "processor_size": processor_size, "hidden_dim": hidden_dim,
           "aggregation": aggregation, "seed": seed}
    return GridFieldPredictor(_GridOperator(net, "meshgraphnet", n, 0.0, 1.0, 1.0), cfg, "PhysicsNeMo MeshGraphNet")


@dataclass
class TrainingResult:
    """What :func:`train_grid_operator` spent and reached."""
    run: Any                                   # pinneapple_analysis.cost.equal_budget.BudgetRun
    epochs: float
    final_loss: float
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"epochs": self.epochs, "final_loss": self.final_loss, **self.run.to_dict()}


def train_grid_operator(pred: GridFieldPredictor, a: np.ndarray, u: np.ndarray, budget, *, batch_size: int = 8,
                        lr: float = 1e-3, weight_decay: float = 1e-5, seed: int = 0,
                        evaluate=None, eval_at=(), threads: int | None = None) -> TrainingResult:
    """Train ``pred`` on ``(a, u)`` pairs until ``budget`` (a ``ComputeBudget``) is spent.

    Loss: mean over the batch of the relative L2 error per sample (the loss of the FNO paper). Mini-batches are
    drawn without replacement, epoch by epoch, from a fixed seed. ``evaluate`` / ``eval_at`` are passed to
    ``train_within_budget`` (evaluation time is not charged to the model).
    """
    from pinneapple_analysis.cost.equal_budget import cosine_lr, set_lr, train_within_budget
    if threads:
        torch.set_num_threads(int(threads))
    a = np.asarray(a, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32)
    pred.fit_normalisation(a, u)
    A, U = torch.from_numpy(a), torch.from_numpy(u)
    m = pred.module
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    state = {"order": rng.permutation(len(a)), "pos": 0, "seen": 0}

    def step(progress: float) -> float:
        m.train()
        if state["pos"] + batch_size > len(a):
            state["order"], state["pos"] = rng.permutation(len(a)), 0
        idx = torch.from_numpy(state["order"][state["pos"]:state["pos"] + batch_size])
        state["pos"] += batch_size
        state["seen"] += len(idx)
        set_lr(opt, cosine_lr(progress, lr))
        out = m(A[idx])
        tgt = U[idx]
        loss = (torch.linalg.vector_norm((out - tgt).flatten(1), dim=1)
                / torch.linalg.vector_norm(tgt.flatten(1), dim=1)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return float(loss.detach())

    run = train_within_budget(step, budget, count_first_step=False, evaluate=evaluate, eval_at=eval_at)
    m.eval()
    losses = [h["loss"] for h in run.history if h.get("loss") is not None]
    return TrainingResult(run, state["seen"] / len(a), float(losses[-1]) if losses else float("nan"), run.history)
