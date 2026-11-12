"""Fitting the coordinate network on a split of the public release, with the optional physics terms."""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .features import FEATURE_NAMES, WALL_FRACTION, point_features, sample_indices, scaled_targets
from .model import CoordinateNet, physics_loss

__all__ = ["TrainingSpec", "load_training_points", "train", "pick_device"]


@dataclass(frozen=True)
class TrainingSpec:
    """Everything one run of the surrogate is defined by — recorded verbatim in the methodology record."""
    train_key: str = "full_train"
    train_cases: int = 640
    val_cases: int = 40
    points_per_case: int = 4096
    epochs: int = 30
    batch_size: int = 8192
    learning_rate: float = 2e-3
    width: int = 256
    depth: int = 4
    seed: int = 0
    physics_weight: float = 0.0
    wall_fraction: float = WALL_FRACTION

    def as_dict(self) -> dict[str, Any]:
        return {"train_key": self.train_key, "train_cases": self.train_cases, "val_cases": self.val_cases,
                "per_case": self.points_per_case, "epochs": self.epochs,
                "batch": self.batch_size, "lr": self.learning_rate,
                "width": self.width, "depth": self.depth, "seed": self.seed,
                "physics_weight": self.physics_weight, "wall_fraction": self.wall_fraction}


def pick_device(preferred: str | None = None) -> str:
    """``cuda``, then ``mps``, then ``cpu``; a caller that names one gets exactly that one."""
    import torch

    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_training_points(store: str, case_ids: Sequence[str], spec: TrainingSpec, *,
                         points_per_case: int | None = None, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Sample points from every case and turn them into the network's input and target blocks."""
    rng = np.random.default_rng(seed)
    per_case = points_per_case or spec.points_per_case
    features, targets = [], []
    for case_id in case_ids:
        with np.load(os.path.join(store, f"{case_id}.npz")) as data:
            take = sample_indices(data["features"], per_case, rng, spec.wall_fraction)
            features.append(point_features(data["features"][take], data["airfoil_points"]))
            targets.append(scaled_targets(data["targets"][take], data["features"][take]))
    return np.concatenate(features), np.concatenate(targets)


def train(store: str, manifest: dict[str, Any], out_dir: str, spec: TrainingSpec,
          device: str | None = None, progress=None) -> dict[str, Any]:
    """Fit one network, write ``model.pt`` and ``training.json``, and return the record.

    The point pool stays in host memory and only the current batch goes to the device: gathering rows out of a
    multi-million-row tensor on the GPU costs more than the forward and backward passes it feeds."""
    import torch

    pool = sorted(manifest.get(spec.train_key) or [])
    if not pool:
        raise ValueError(f"the manifest has no cases for {spec.train_key!r}")
    train_ids = pool[:spec.train_cases]
    val_ids = pool[-spec.val_cases:]
    device = pick_device(device)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)

    started = time.time()
    x_train, y_train = load_training_points(store, train_ids, spec, seed=spec.seed)
    x_val, y_val = load_training_points(store, val_ids, spec, points_per_case=min(spec.points_per_case, 2048),
                                        seed=spec.seed + 1)
    model = CoordinateNet(spec.width, spec.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=spec.epochs,
                                                          eta_min=spec.learning_rate / 50)
    x_host, y_host = torch.from_numpy(x_train), torch.from_numpy(y_train)
    val_x = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    history = []
    n = len(x_host)
    for epoch in range(1, spec.epochs + 1):
        model.train()
        order = torch.randperm(n)
        running, seen = 0.0, 0
        for start in range(0, n, spec.batch_size):
            index = order[start:start + spec.batch_size]
            batch_x = x_host[index].to(device, non_blocking=True)
            batch_y = y_host[index].to(device, non_blocking=True)
            prediction = model(batch_x)
            loss = ((prediction - batch_y) ** 2).mean()
            if spec.physics_weight:
                loss = loss + spec.physics_weight * physics_loss(prediction, batch_x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(index)
            seen += len(index)
        schedule.step()
        model.eval()
        with torch.no_grad():
            validation = float(((model(val_x) - val_y) ** 2).mean())
        history.append({"epoch": epoch, "train": running / max(seen, 1), "val": validation})
        if progress:
            progress(epoch, spec.epochs, running / max(seen, 1), validation)

    weights = os.path.join(out_dir, "model.pt")
    torch.save({"state_dict": model.state_dict(), "width": spec.width, "depth": spec.depth,
                "features": list(FEATURE_NAMES), "targets": ["u/Uinf", "v/Uinf", "p/Uinf2"]}, weights)
    record = {"variant": "physics" if spec.physics_weight else "plain", **spec.as_dict(),
              "parameters": int(sum(p.numel() for p in model.parameters())), "device": device,
              "train_points": int(len(x_train)), "val_points": int(len(x_val)),
              "final_val_mse_standardized": history[-1]["val"], "history": history,
              "weights_sha256": hashlib.sha256(open(weights, "rb").read()).hexdigest(),
              "seconds": round(time.time() - started, 1)}
    with open(os.path.join(out_dir, "training.json"), "w") as handle:
        json.dump(record, handle, indent=1)
    return record
