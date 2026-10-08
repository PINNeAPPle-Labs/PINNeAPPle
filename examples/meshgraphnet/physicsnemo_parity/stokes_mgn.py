"""PINNeAPPle port of PhysicsNeMo ``examples/cfd/stokes_mgn`` (NOT RUN YET).

Stokes flow in a pipe with a random polygonal obstacle (FEniCS data, NGC
``physicsnemo_datasets_stokes_flow``). One-shot regression geometry → (u, v, p),
no time stepping. Hyper-parameters follow ``stokes_mgn/conf/config.yaml``:

  node inputs 7 = [x, y, one-hot(marker, 5)]   edge inputs 3 = [dx, dy, |d|]
  outputs 3 = [u, v, p] (z-score)   hidden 256, 15 message-passing layers, ReLU
  Adam lr 1e-4, final lr = 1% of initial (per-epoch geometric decay)
  500 epochs, 500 train / 10 valid / 10 test samples, batch size 1, MSE

Data (files ``results/*.vtp``; requires ``pip install pyvista``):
    wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/physicsnemo/physicsnemo_datasets_stokes_flow/0.0.1/files?redirect=true&path=results_polygon.zip' -O results_polygon.zip
    unzip results_polygon.zip   # → results/
    python examples/meshgraphnet/physicsnemo_parity/stokes_mgn.py --data results
Smoke test: --epochs 2 --num-train 8 --num-eval 2

Extra (PhysicsNeMo's ``pi_fine_tuning.py`` — PDE-residual fine tuning with an MLP) is a
separate item: use ``pinneapple_physics`` Stokes residuals; not part of this script.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import pick_device  # noqa: E402
from pinneapple_neural.architectures.graphnn.base import GraphBatch  # noqa: E402
from pinneapple_neural.architectures.graphnn.mesh_graph_net import MeshGraphNet  # noqa: E402
from pinneapple_neural.architectures.graphnn.mgn_dynamics import Normalizer, edge_features, triangles_to_edges  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="results")
ap.add_argument("--out", default="examples/meshgraphnet/_out/parity_stokes")
ap.add_argument("--epochs", type=int, default=500)
ap.add_argument("--num-train", type=int, default=500)
ap.add_argument("--num-eval", type=int, default=10)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
dev = pick_device()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)


def load(path):
    import pyvista as pv
    m = pv.read(path)
    pos = torch.from_numpy(np.asarray(m.points[:, :2])).float()
    faces = m.faces.reshape(-1, 4)[:, 1:]  # triangles
    marker = torch.from_numpy(np.asarray(m.point_data["marker"])).long()
    y = torch.stack([torch.from_numpy(np.asarray(m.point_data[k])).float() for k in ("u", "v", "p")], -1)
    ei = triangles_to_edges(torch.from_numpy(faces))
    x = torch.cat([pos, torch.nn.functional.one_hot(marker, 5).float()], -1)
    return x, ei, edge_features(pos, ei), y


files = sorted(Path(a.data).glob("*.vtp"), key=lambda f: int(re.search(r"\d+", f.stem).group()))
rng.shuffle(files)  # PhysicsNeMo preprocess.py: random 80/10/10 split
n = len(files)
tr_f, va_f, te_f = files[: int(.8 * n)][: a.num_train], files[int(.8 * n): int(.9 * n)][: a.num_eval], files[int(.9 * n):][: a.num_eval]
tr, va, te = ([load(f) for f in fs] for fs in (tr_f, va_f, te_f))
print(f"device={dev} train={len(tr)} valid={len(va)} test={len(te)}")

xn, en, yn = Normalizer(7), Normalizer(3), Normalizer(3)
# PhysicsNeMo normalises only pos/u/v/p (not the one-hot marker): keep one-hot raw.
xn.fit(t[0] for t in tr); xn.mean[2:] = 0; xn.std[2:] = 1
en.fit(t[2] for t in tr); yn.fit(t[3] for t in tr)
net = MeshGraphNet(7, 3, edge_in_dim=3, hidden_dim=256, n_layers=2, n_message_passing=15, activation="relu").to(dev)
xn, en, yn = xn.to(dev), en.to(dev), yn.to(dev)
opt = torch.optim.Adam(net.parameters(), lr=a.lr)
sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.01 ** (1.0 / a.epochs))  # stepped per epoch


def predict(s):
    x, ei, ea, _ = s
    return net(GraphBatch(x=xn(x.to(dev))[None], edge_index=ei.to(dev), edge_attr=en(ea.to(dev))[None])).y[0]


def evaluate(ds):
    net.eval()
    with torch.no_grad():
        err = [((yn.inverse(predict(s)) - s[3].to(dev)) ** 2).mean(0).sqrt() / s[3].to(dev).std(0) for s in ds]
    return torch.stack(err).mean(0).tolist()  # relative RMSE [u, v, p]


t0 = time.time()
for ep in range(a.epochs):
    net.train()
    tot = 0.0
    for i in rng.permutation(len(tr)):
        loss = ((predict(tr[i]) - yn(tr[i][3].to(dev))) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tot += loss.item()
    sched.step()
    if ep % 10 == 0 or ep == a.epochs - 1:
        print(f"epoch {ep:4d} loss {tot / len(tr):.4e}  valid relRMSE(u,v,p) {evaluate(va)}  {time.time() - t0:6.0f}s", flush=True)

res = {"test_rel_rmse_uvp": evaluate(te), "valid_rel_rmse_uvp": evaluate(va), "epochs": a.epochs, "train": len(tr)}
torch.save({"net": net.state_dict()}, out / "mgn.pt")
(out / "metrics.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
