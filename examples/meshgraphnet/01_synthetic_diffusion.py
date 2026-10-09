"""Sanity experiment: MeshDynamicsMGN learns heat diffusion on random meshes.

Self-contained (no download). Ground truth is explicit graph-Laplacian
diffusion on a Delaunay triangulation of random points; the Dirichlet boundary
nodes are held at a random constant. A correct MeshGraphNet must beat the
"copy previous state" baseline on a multi-step rollout over unseen meshes.

    python examples/meshgraphnet/01_synthetic_diffusion.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from _common import eval_rollout, pick_device, train_dynamics
from scipy.spatial import Delaunay

from pinneapple_neural.architectures.graphnn.mgn_dynamics import MeshDynamicsMGN, MeshGraph


def make_sample(rng, n_pts=150, T=30, alpha=0.2):
    pts = rng.random((n_pts, 2))
    cells = Delaunay(pts).simplices
    pos = torch.from_numpy(pts).float()
    nt = torch.zeros(n_pts, dtype=torch.long)
    bnd = (pos[:, 0] < 0.05) | (pos[:, 0] > 0.95)
    nt[bnd] = 3
    g = MeshGraph.from_mesh(pos, torch.from_numpy(cells), nt)
    src, dst = g.edge_index
    deg = torch.zeros(n_pts).index_add_(0, dst, torch.ones(dst.numel()))
    u = torch.zeros(n_pts, 1)
    u[bnd] = float(rng.uniform(-1, 1))
    u = u + torch.exp(-((pos - torch.from_numpy(rng.random(2)).float()) ** 2).sum(-1, keepdim=True) / 0.02)
    traj = [u]
    for _ in range(T):
        agg = torch.zeros_like(u).index_add_(0, dst, u[src])
        un = u + alpha * (agg / deg.unsqueeze(-1) - u)
        un[bnd] = u[bnd]
        u = un
        traj.append(u)
    v = torch.stack(traj)
    return g, v, torch.zeros_like(v)  # (graph, state, dummy pressure)


def run(n_train=24, n_test=6, n_pts=150, steps=2500, rollout=20, log_every=250, seed=0):
    """Train on ``n_train`` random meshes, roll out ``rollout`` steps on ``n_test`` unseen ones; returns the
    rollout RMSE next to the frozen-initial-state baseline. tests/test_mesh_dynamics_mgn.py runs a reduced size."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    train = [make_sample(rng, n_pts) for _ in range(n_train)]
    test = [make_sample(rng, n_pts) for _ in range(n_test)]
    dev = pick_device()
    model = MeshDynamicsMGN(vel_dim=1, num_node_types=4, edge_dim=3, out_p=False, hidden_dim=32,
                            n_layers=2, n_message_passing=4, activation="relu")
    model.fit_stats([v for _, v, _ in train], None, [g for g, _, _ in train])
    train_dynamics(model, train, steps=steps, lr=2e-3, lr_final=1e-5, noise_std=0.0, device=dev,
                   log_every=log_every)
    m = eval_rollout(model, test, n_steps=rollout, device=dev)
    base = np.mean([((v[0:1] - v[1:rollout + 1])[:, g.node_type == 0] ** 2).mean().sqrt().item()
                    for g, v, _ in test])
    m["frozen_ic_baseline_rmse"] = float(base)
    return m


if __name__ == "__main__":
    m = run()
    print(json.dumps(m, indent=2))
    Path("examples/meshgraphnet/_out").mkdir(parents=True, exist_ok=True)
    Path("examples/meshgraphnet/_out/synthetic_diffusion.json").write_text(json.dumps(m, indent=2))
    assert m["rollout_rmse"] < m["frozen_ic_baseline_rmse"], "MGN failed to beat the frozen baseline"
