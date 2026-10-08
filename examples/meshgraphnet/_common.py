"""Shared helpers for the MeshGraphNet examples (data loading, train/eval loops).

Nothing here is specific to a dataset except the DeepMind ``meshgraphnets``
TFRecord reader (``cylinder_flow``), which needs ``pip install tfrecord``.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch

from pinneapple_neural.architectures.graphnn.mgn_dynamics import MeshDynamicsMGN, MeshGraph

DM_URL = "https://storage.googleapis.com/dm-meshgraphnets"


def pick_device() -> torch.device:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")  # tiny graphs: MPS launch latency dominates


# ---------------------------------------------------------------------------
# DeepMind meshgraphnets TFRecord
# ---------------------------------------------------------------------------

def download_prefix(dataset: str, split: str, dest: Path, nbytes: int) -> Path:
    """Download meta.json and the first ``nbytes`` of ``<split>.tfrecord``.

    TFRecord is a sequential format, so a byte prefix holds whole leading
    trajectories (a truncated last record is dropped by the reader).
    """
    dest.mkdir(parents=True, exist_ok=True)
    meta = dest / "meta.json"
    if not meta.exists():
        subprocess.run(["curl", "-sSf", "-o", str(meta), f"{DM_URL}/{dataset}/meta.json"], check=True)
    rec = dest / f"{split}.tfrecord"
    if not rec.exists():
        subprocess.run(["curl", "-sSf", "-r", f"0-{nbytes - 1}", "-o", str(rec),
                        f"{DM_URL}/{dataset}/{split}.tfrecord"], check=True)
    return rec


def read_trajectories(data_dir: Path, split: str, max_traj: int) -> List[Dict[str, np.ndarray]]:
    from tfrecord.reader import tfrecord_loader

    meta = json.loads((data_dir / "meta.json").read_text())
    desc = {k: "byte" for k in meta["field_names"]}
    out: List[Dict[str, np.ndarray]] = []
    it = tfrecord_loader(str(data_dir / f"{split}.tfrecord"), None, description=desc)
    while len(out) < max_traj:
        try:
            rec = next(it)
        except StopIteration:
            break
        except Exception:  # truncated tail of a prefix download
            break
        traj = {}
        for name, spec in meta["features"].items():
            arr = np.frombuffer(bytes(rec[name]),dtype=np.dtype(spec["dtype"])).copy()
            arr = arr.reshape(spec["shape"][0], -1, spec["shape"][-1])
            traj[name] = arr[0] if spec["type"] == "static" else arr
        out.append(traj)
    return out


def cylinder_flow_to_tensors(traj: Dict[str, np.ndarray], node_type_map: Dict[int, int]):
    """DeepMind trajectory → (MeshGraph, velocity (T,N,2), pressure (T,N,1)).

    ``node_type_map`` collapses raw ids to contiguous classes, e.g. the four
    PhysicsNeMo classes: NORMAL=0, INFLOW=1, OUTFLOW=2, WALL=3.
    """
    pos = torch.from_numpy(traj["mesh_pos"]).float()
    cells = torch.from_numpy(traj["cells"]).long()
    raw = traj["node_type"][:, 0]
    nt = torch.from_numpy(np.vectorize(node_type_map.get)(raw)).long()
    g = MeshGraph.from_mesh(pos, cells, nt)
    return g, torch.from_numpy(traj["velocity"]).float(), torch.from_numpy(traj["pressure"]).float()


# raw DeepMind ids: 0 NORMAL, 4 INFLOW, 5 OUTFLOW, 6 WALL_BOUNDARY
CYLINDER_TYPE_MAP = {0: 0, 4: 1, 5: 2, 6: 3}


# ---------------------------------------------------------------------------
# Generic training / evaluation for MeshDynamicsMGN
# ---------------------------------------------------------------------------

def train_dynamics(model: MeshDynamicsMGN, data: List[Tuple[MeshGraph, torch.Tensor, torch.Tensor]],
                   *, steps: int, lr: float = 1e-4, lr_final: float = 1e-6,
                   noise_std: float = 0.02, device: torch.device, log_every: int = 200,
                   seed: int = 0) -> List[float]:
    """Random-(trajectory, time) one-step training with exponential LR decay."""
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


@torch.no_grad()
def eval_rollout(model: MeshDynamicsMGN, data, *, n_steps: int, device: torch.device) -> Dict[str, float]:
    """Mean rollout RMSE of velocity over ``n_steps`` plus 1-step RMSE."""
    model.to(device).eval()
    roll, one = [], []
    for g, v, p in data:
        g, v = g.to(device), v.to(device)
        n = min(n_steps, v.shape[0] - 1)
        pred, _ = model.rollout(v, g, n)
        free = (g.node_type == 0)
        roll.append(((pred[1:] - v[1 : n + 1])[:, free] ** 2).mean().sqrt().item())
        nxt, _ = model.step(v[0], g)
        one.append(((nxt - v[1])[free] ** 2).mean().sqrt().item())
    return {"rollout_rmse": float(np.mean(roll)), "one_step_rmse": float(np.mean(one)), "n_steps": n_steps}
