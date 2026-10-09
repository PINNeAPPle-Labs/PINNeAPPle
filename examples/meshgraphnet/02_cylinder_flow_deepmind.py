"""MeshGraphNet on DeepMind's ``cylinder_flow`` (Pfaff et al., ICLR 2021).

Same task as PhysicsNeMo's ``vortex_shedding_mgn``: predict the next-step
velocity/pressure of vortex shedding behind a cylinder on irregular triangle
meshes, then roll out autoregressively.

Reduced budget by default (a byte-prefix of the TFRecord files: a few dozen
trajectories, fewer message-passing layers) so it runs on a laptop. Use
``--train-mb 13000 --mp 15 --hidden 128 --steps 500000`` for the paper setup.

    pip install tfrecord
    python examples/meshgraphnet/02_cylinder_flow_deepmind.py
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import torch
from _common import (
    CYLINDER_TYPE_MAP,
    cylinder_flow_to_tensors,
    download_prefix,
    eval_rollout,
    pick_device,
    read_trajectories,
    train_dynamics,
)

from pinneapple_neural.architectures.graphnn.mgn_dynamics import MeshDynamicsMGN

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="examples/meshgraphnet/_data/cylinder_flow")
ap.add_argument("--out", default="examples/meshgraphnet/_out/cylinder_flow")
ap.add_argument("--train-mb", type=int, default=400)
ap.add_argument("--valid-mb", type=int, default=60)
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--mp", type=int, default=15)
ap.add_argument("--rollout", type=int, default=100)
ap.add_argument("--lr", type=float, default=1e-3, help="paper: 1e-4 over ~1e6 steps")
ap.add_argument("--lr-final", type=float, default=1e-5)
ap.add_argument("--noise", type=float, default=0.003, help="paper/PhysicsNeMo: 0.02 over ~1e6 steps")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

torch.manual_seed(a.seed)
dev = pick_device()
data_dir, out = Path(a.data), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
print(f"device={dev}")

download_prefix("cylinder_flow", "train", data_dir, a.train_mb * 1_000_000)
download_prefix("cylinder_flow", "valid", data_dir, a.valid_mb * 1_000_000)
tr = [cylinder_flow_to_tensors(t, CYLINDER_TYPE_MAP) for t in read_trajectories(data_dir, "train", 10_000)]
va = [cylinder_flow_to_tensors(t, CYLINDER_TYPE_MAP) for t in read_trajectories(data_dir, "valid", 10_000)]
print(f"train trajectories={len(tr)}  valid trajectories={len(va)}")

model = MeshDynamicsMGN(vel_dim=2, num_node_types=4, edge_dim=3, hidden_dim=a.hidden,
                        n_layers=2, n_message_passing=a.mp, activation="relu")
model.fit_stats([v for _, v, _ in tr], [p for _, _, p in tr], [g for g, _, _ in tr])
print(f"params={sum(p.numel() for p in model.parameters()):,}")

hist = train_dynamics(model, tr, steps=a.steps, lr=a.lr, lr_final=a.lr_final,
                     device=dev, noise_std=a.noise, seed=a.seed)
metrics = eval_rollout(model, va, n_steps=a.rollout, device=dev)

# baseline: "velocity stays at its t=0 value" over the same horizon
base = []
for g, v, _ in va:
    free = g.node_type == 0
    n = min(a.rollout, v.shape[0] - 1)
    base.append(((v[0:1] - v[1 : n + 1])[:, free] ** 2).mean().sqrt().item())
metrics["frozen_ic_baseline_rmse"] = sum(base) / len(base)
metrics.update(train_traj=len(tr), valid_traj=len(va), steps=a.steps, lr=a.lr, noise=a.noise, hidden=a.hidden, mp=a.mp,
               final_train_loss=hist[-1] if hist else None)
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))

# figure: |v| ground truth vs prediction at the last rollout step of the first valid trajectory
g, v, _ = va[0]
n = min(a.rollout, v.shape[0] - 1)
pred, _ = model.rollout(v.to(dev), g.to(dev), n)
pos = torch.from_numpy(read_trajectories(data_dir, "valid", 1)[0]["mesh_pos"])
cells = torch.from_numpy(read_trajectories(data_dir, "valid", 1)[0]["cells"])
tri = mtri.Triangulation(pos[:, 0], pos[:, 1], cells)
fig, ax = plt.subplots(3, 1, figsize=(8, 7))
for k, (title, f) in enumerate([("ground truth |v|", v[n].norm(dim=-1)),
                                ("MeshGraphNet |v|", pred[n].cpu().norm(dim=-1)),
                                ("abs. error", (pred[n].cpu() - v[n]).norm(dim=-1))]):
    c = ax[k].tripcolor(tri, f, shading="gouraud")
    ax[k].set_title(f"{title} (step {n})")
    ax[k].set_aspect("equal")
    fig.colorbar(c, ax=ax[k])
fig.tight_layout()
fig.savefig(out / "rollout.png", dpi=120)
