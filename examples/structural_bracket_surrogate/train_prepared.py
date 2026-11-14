"""Train DeepONet on *prepared multi-case* feature arrays, never a single FEM case.

Input NPZ schema:
  branch [C,31], trunk [C,P,8], stress_pa [C,P], group [C]
Cases sharing a geometry group must never span train and validation.
The 31/8 feature encoding is intentionally external until fully specified.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from deeponet import BracketDeepONet, stress_weighted_mse, relative_l2


def group_split(groups: np.ndarray, val_fraction: float = 0.2, seed: int = 42):
    unique = np.unique(groups)
    if len(unique) < 2 or not (0 < val_fraction < 1):
        raise ValueError("Need at least two geometry groups and valid fraction")
    rng = np.random.default_rng(seed)
    unique = rng.permutation(unique)
    nval = max(1, min(len(unique)-1, round(len(unique)*val_fraction)))
    val = np.isin(groups, unique[:nval])
    return np.flatnonzero(~val), np.flatnonzero(val)


def train(dataset: str, output: str, epochs: int = 100, seed: int = 42):
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    torch.manual_seed(seed)
    with np.load(dataset, allow_pickle=False) as data:
        branch = np.asarray(data["branch"], dtype=np.float32)
        trunk = np.asarray(data["trunk"], dtype=np.float32)
        stress = np.asarray(data["stress_pa"], dtype=np.float32)
        groups = np.asarray(data["group"])
    if branch.ndim != 2 or branch.shape[1] != 31:
        raise ValueError("branch must have shape [cases,31]")
    if trunk.ndim != 3 or trunk.shape[0] != len(branch) or trunk.shape[2] != 8:
        raise ValueError("trunk must have shape [cases,points,8]")
    if stress.shape != trunk.shape[:2]:
        raise ValueError("stress_pa must have shape [cases,points]")
    if groups.shape != (len(branch),):
        raise ValueError("group must have one label per case")
    if not all(np.isfinite(x).all() for x in (branch, trunk, stress)):
        raise ValueError("Dataset contains nonfinite values")
    train_idx, val_idx = group_split(groups, seed=seed)
    # Fit all normalizers on train cases only. Zero-variance features remain unchanged.
    bmean = branch[train_idx].mean(axis=0)
    bstd = branch[train_idx].std(axis=0).clip(min=1e-6)
    tmean = trunk[train_idx].reshape(-1,8).mean(axis=0)
    tstd = trunk[train_idx].reshape(-1,8).std(axis=0).clip(min=1e-6)
    yscale = max(float(np.percentile(np.abs(stress[train_idx]), 95)), 1e-12)
    b = torch.from_numpy((branch-bmean)/bstd)
    t = torch.from_numpy((trunk-tmean)/tstd)
    y = torch.from_numpy(stress/yscale)
    model = BracketDeepONet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        prediction = model(b[train_idx], t[train_idx])
        loss = stress_weighted_mse(prediction, y[train_idx])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_prediction = model(b[val_idx], t[val_idx])
            val_error = relative_l2(val_prediction, y[val_idx])
        history.append({"epoch": epoch+1, "train_weighted_mse": float(loss.item()),
                        "val_relative_l2": float(val_error.item())})
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "normalization": {
            "branch_mean": bmean.tolist(), "branch_std": bstd.tolist(),
            "trunk_mean": tmean.tolist(), "trunk_std": tstd.tolist(),
            "stress_scale_pa": yscale,
        },
        "split": {"train_cases": train_idx.tolist(), "validation_cases": val_idx.tolist()},
        "history": history, "seed": seed, "feature_schema": "external_31_8_not_yet_verified",
    }, out)
    return history[-1]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset", help="NPZ with branch, trunk, stress_pa and group")
    p.add_argument("--output", default="artifacts/deeponet.pt")
    p.add_argument("--epochs", type=int, default=100)
    args = p.parse_args()
    print(json.dumps(train(args.dataset, args.output, epochs=args.epochs), indent=2))
