"""PINNeAPPle port of PhysicsNeMo ``examples/cfd/vortex_shedding_mgn`` (NOT RUN YET).

Same dataset (DeepMind ``cylinder_flow``), same graph/features, same
hyper-parameters as ``vortex_shedding_mgn/conf/config.yaml``:

  node inputs  6  = [u_t, v_t, one-hot(4: NORMAL/INFLOW/OUTFLOW/WALL)]
  edge inputs  3  = [dx, dy, |d|]
  outputs      3  = [Δu, Δv, p_{t+1}]        (all z-score normalised)
  hidden 128, 2 MLP layers, 15 message-passing layers, sum aggregation, ReLU
  Adam lr 1e-4, exponential decay 0.9999991 per iteration, batch size 1
  25 epochs x 400 train trajectories x 300 time steps, noise std 0.02
  test: 10 trajectories x 300-step rollout

Usage (full download is 13.6 GB train + 1.4 GB test):
    pip install tfrecord
    python examples/meshgraphnet/physicsnemo_parity/vortex_shedding_mgn.py \
        --train-mb 13650 --test-mb 1360
Smoke test:  --epochs 1 --num-train 2 --num-train-steps 20 --num-test 1 --num-test-steps 20
Reference numbers to compare against: PhysicsNeMo's ``inference.py`` rollout error
on the same test trajectories (same RMSE definition: velocity, NORMAL nodes).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import (CYLINDER_TYPE_MAP, cylinder_flow_to_tensors, download_prefix, eval_rollout,  # noqa: E402
                      pick_device, read_trajectories)
from pinneapple_neural.architectures.graphnn.mgn_dynamics import MeshDynamicsMGN  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="examples/meshgraphnet/_data/cylinder_flow")
ap.add_argument("--out", default="examples/meshgraphnet/_out/parity_vortex_shedding")
ap.add_argument("--train-mb", type=int, default=13650)
ap.add_argument("--test-mb", type=int, default=1360)
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--num-train", type=int, default=400)
ap.add_argument("--num-train-steps", type=int, default=300)
ap.add_argument("--num-test", type=int, default=10)
ap.add_argument("--num-test-steps", type=int, default=300)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--lr-decay", type=float, default=0.9999991)
ap.add_argument("--noise-std", type=float, default=0.02)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
dev = pick_device()
data_dir, out = Path(a.data), Path(a.out)
out.mkdir(parents=True, exist_ok=True)

download_prefix("cylinder_flow", "train", data_dir, a.train_mb * 1_000_000)
download_prefix("cylinder_flow", "test", data_dir, a.test_mb * 1_000_000)
tr = [cylinder_flow_to_tensors(t, CYLINDER_TYPE_MAP) for t in read_trajectories(data_dir, "train", a.num_train)]
te = [cylinder_flow_to_tensors(t, CYLINDER_TYPE_MAP) for t in read_trajectories(data_dir, "test", a.num_test)]
tr = [(g, v[: a.num_train_steps + 1], p[: a.num_train_steps + 1]) for g, v, p in tr]
print(f"device={dev} train={len(tr)} test={len(te)}")

model = MeshDynamicsMGN(vel_dim=2, num_node_types=4, edge_dim=3, hidden_dim=128, n_layers=2,
                        n_message_passing=15, activation="relu").to(dev)
model.fit_stats([v for _, v, _ in tr], [p for _, _, p in tr], [g for g, _, _ in tr])
tr = [(g.to(dev), v.to(dev), p.to(dev)) for g, v, p in tr]
opt = torch.optim.Adam(model.parameters(), lr=a.lr)
sched = torch.optim.lr_scheduler.ExponentialLR(opt, a.lr_decay)

t0 = time.time()
for ep in range(a.epochs):
    model.train()
    total, n = 0.0, 0
    for gi in rng.permutation(len(tr)):
        g, v, p = tr[gi]
        for t in rng.permutation(v.shape[0] - 1):
            loss = model.loss(v[t], v[t + 1], p[t + 1], g, noise_std=a.noise_std)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total, n = total + loss.item(), n + 1
    print(f"epoch {ep:3d} loss {total / n:.4e}  {time.time() - t0:7.0f}s", flush=True)
    torch.save(model.state_dict(), out / "mgn.pt")

metrics = eval_rollout(model, te, n_steps=a.num_test_steps, device=dev)
metrics.update(epochs=a.epochs, train_traj=len(tr), test_traj=len(te))
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
