"""PINNeAPPle port of PhysicsNeMo ``examples/cfd/lagrangian_mgn`` (NOT RUN YET).

Particle-based fluids (DeepMind *Learning to Simulate* datasets, e.g. Water-2D)
with MeshGraphNet on a radius graph rebuilt at every step. Follows
``conf/experiment/water.yaml`` + ``conf/model/mgn_2d.yaml``:

  node inputs 22 = [pos(2), 5 x velocity history(10), boundary features(4), one-hot type(6)]
  edge inputs 3  = [dx, dy, exp(-d^2/r^2)]   radius r = metadata (0.015 for Water)
  output 2 = normalised acceleration; semi-implicit Euler update (dt folded into the stats)
  noise_std 3e-4 random-walk noise on positions of non-kinematic particles
  Adam lr 1e-4 -> 1e-6 cosine, batch 20 (samples), 20 epochs

Differences vs PhysicsNeMo (documented, not hidden): PINNeAPPle's MeshGraphNet has a single
``hidden_dim`` (128 here) instead of 256-wide encoders/decoder with a 128-wide processor, and uses
the dense (B, N, F) convention, so a "batch" of 20 samples is 20 sequential graphs accumulated
into one optimiser step (gradient accumulation, same expected gradient).

Data:  https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate
    pip install tfrecord
    python examples/meshgraphnet/physicsnemo_parity/lagrangian_mgn.py --dataset Water --download
Smoke test: --epochs 1 --num-seq 2 --steps-per-epoch 20 --num-test-seq 1 --rollout-steps 20
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import pick_device  # noqa: E402
from pinneapple_neural.architectures.graphnn.base import GraphBatch  # noqa: E402
from pinneapple_neural.architectures.graphnn.mesh_graph_net import MeshGraphNet  # noqa: E402

L2S_URL = "https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets"
KINEMATIC_ID = 3

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="Water")
ap.add_argument("--data", default="examples/meshgraphnet/_data")
ap.add_argument("--out", default="examples/meshgraphnet/_out/parity_lagrangian")
ap.add_argument("--download", action="store_true", help="fetch metadata + train/valid tfrecords with curl")
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--num-seq", type=int, default=1000)
ap.add_argument("--steps-per-epoch", type=int, default=None, help="default: num_seq * (T - history - 1)")
ap.add_argument("--batch", type=int, default=20)
ap.add_argument("--num-test-seq", type=int, default=30)
ap.add_argument("--rollout-steps", type=int, default=206)
ap.add_argument("--history", type=int, default=5)
ap.add_argument("--num-types", type=int, default=6)
ap.add_argument("--noise-std", type=float, default=3e-4)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
dev = pick_device()
root = Path(a.data) / a.dataset
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
if a.download:
    root.mkdir(parents=True, exist_ok=True)
    for f in ("metadata.json", "train.tfrecord", "valid.tfrecord"):
        if not (root / f).exists():
            subprocess.run(["curl", "-sSf", "-o", str(root / f), f"{L2S_URL}/{a.dataset}/{f}"], check=True)
meta = json.loads((root / "metadata.json").read_text())
dim, radius, bounds = meta["dim"], meta["default_connectivity_radius"], meta["bounds"][0]
vm, vs, am, as_ = (torch.tensor(meta[k]).reshape(1, dim).float() for k in ("vel_mean", "vel_std", "acc_mean", "acc_std"))
vm, vs, am, as_ = (t.to(dev) for t in (vm, vs, am, as_))


def read_sequences(split, n):
    from tfrecord.reader import tfrecord_loader
    it = tfrecord_loader(str(root / f"{split}.tfrecord"), None, description={"key": "int", "particle_type": "byte"},
                         sequence_description={"position": "byte"})
    seqs = []
    for ctx, seq in it:
        if len(seqs) >= n:
            break
        ptype = torch.from_numpy(np.frombuffer(ctx["particle_type"], dtype=np.int64).copy())
        pos = np.stack([np.frombuffer(b, dtype=np.float32) for b in seq["position"]])
        pos = torch.from_numpy(pos.reshape(pos.shape[0], -1, dim).copy())
        seqs.append((pos, F.one_hot(ptype, a.num_types).float()))
    return seqs


def boundary_feature(pos):
    d = torch.cat([pos - bounds[0], bounds[1] - pos], -1)
    f = torch.exp(-(d ** 2) / radius ** 2)
    f[d > radius] = 0
    return f


def build(pos_t, vel_hist, ntype):
    """pos_t (N,d), vel_hist (H,N,d) normalised → node feats, edge_index, edge_attr (radius graph, self-loops)."""
    x = torch.cat([pos_t, vel_hist.permute(1, 0, 2).flatten(1), boundary_feature(pos_t), ntype], -1)
    ei = torch.nonzero(torch.cdist(pos_t, pos_t) < radius).t().contiguous()
    disp = pos_t[ei[1]] - pos_t[ei[0]]
    dist = torch.exp(-(disp.norm(dim=-1, keepdim=True) ** 2) / radius ** 2)
    return x, ei, torch.cat([disp, dist], -1)


def random_walk_noise(T, N):
    nv = T - 1
    vel_noise = (a.noise_std / nv ** 0.5) * torch.randn(nv, N, dim, device=dev)
    vel_noise = vel_noise.cumsum(0)
    pos_noise = torch.cat([torch.zeros(1, N, dim, device=dev), vel_noise.cumsum(0)])
    pos_noise[-1] = pos_noise[-2]  # target noise = current noise, cancels in the velocity difference
    return pos_noise


feat_dim = dim + a.history * dim + 2 * dim + a.num_types
net = MeshGraphNet(feat_dim, dim, edge_in_dim=dim + 1, hidden_dim=128, n_layers=2, n_message_passing=10,
                   activation="relu").to(dev)


def predict(pos_t, vel_hist, ntype):
    x, ei, ea = build(pos_t, vel_hist, ntype)
    return net(GraphBatch(x=x[None], edge_index=ei, edge_attr=ea[None])).y[0]


def sample_loss(seq):
    pos, ntype = seq[0].to(dev), seq[1].to(dev)
    T = pos.shape[0]
    t0 = int(rng.integers(0, T - a.history - 1))
    w = pos[t0: t0 + a.history + 2].clone()
    material = (ntype[:, KINEMATIC_ID] == 0).float().unsqueeze(-1)
    w = w + random_walk_noise(*w.shape[:2]) * material
    vel = w[1:] - w[:-1]
    acc = vel[-1] - vel[-2]
    vel_n = (vel - vm) / vs
    acc_n = (acc - am) / as_
    pred = predict(w[-2], vel_n[:-1], ntype)
    mask = material.squeeze(-1) > 0
    return ((pred[mask] - acc_n[mask]) ** 2).mean()


@torch.no_grad()
def rollout(seq, steps):
    net.eval()
    pos, ntype = seq[0].to(dev), seq[1].to(dev)
    h = a.history
    window = pos[: h + 1].clone()  # positions t0..t0+h
    mat = (ntype[:, KINEMATIC_ID] == 0).unsqueeze(-1)
    preds = []
    for k in range(steps):
        vel = window[1:] - window[:-1]
        acc_n = predict(window[-1], (vel - vm) / vs, ntype)
        vnext = vel[-1] + (acc_n * as_ + am)
        pnext = torch.clamp(window[-1] + vnext, bounds[0] + 1e-3, bounds[1] - 1e-3)
        pnext = torch.where(mat, pnext, pos[h + 1 + k])  # kinematic particles follow ground truth
        preds.append(pnext)
        window = torch.cat([window[1:], pnext[None]])
    pred = torch.stack(preds)
    gt = pos[h + 1: h + 1 + steps]
    return ((pred - gt)[:, mat.squeeze(-1)] ** 2).mean(-1).mean().item()  # position MSE over particles/steps


train = read_sequences("train", a.num_seq)
test = read_sequences("valid", a.num_test_seq)
print(f"device={dev} train_seq={len(train)} test_seq={len(test)} node_feat={feat_dim}")
steps_per_epoch = a.steps_per_epoch or len(train) * (train[0][0].shape[0] - a.history - 1)
total_updates = a.epochs * max(steps_per_epoch // a.batch, 1)
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_updates, eta_min=1e-6)

t0 = time.time()
for ep in range(a.epochs):
    net.train()
    run = 0.0
    n_upd = max(steps_per_epoch // a.batch, 1)
    for _ in range(n_upd):
        opt.zero_grad(set_to_none=True)
        for _ in range(a.batch):
            loss = sample_loss(train[int(rng.integers(len(train)))]) / a.batch
            loss.backward()
            run += loss.item()
        opt.step()
        sched.step()
    print(f"epoch {ep:3d} loss {run / n_upd:.4e}  {time.time() - t0:7.0f}s", flush=True)
    torch.save(net.state_dict(), out / "mgn.pt")

res = {"rollout_position_mse": float(np.mean([rollout(s, a.rollout_steps) for s in test])),
       "rollout_steps": a.rollout_steps, "epochs": a.epochs, "train_seq": len(train), "dataset": a.dataset}
(out / "metrics.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
