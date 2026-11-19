"""Evaluate held-out bracket stress predictions and source-family generalization.

Run: python evaluate.py prepared.npz model.pt --output metrics.json
Checkpoint must be produced by train_prepared.py on the same dataset.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from deeponet import BracketDeepONet


def evaluate(dataset: str, checkpoint: str, output: str):
    with np.load(dataset, allow_pickle=False) as data:
        branch = np.asarray(data["branch"], dtype=np.float32)
        trunk = np.asarray(data["trunk"], dtype=np.float32)
        stress = np.asarray(data["stress_pa"], dtype=np.float32)
        groups = np.asarray(data["group"])
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("PyTorch version must support weights_only=True") from exc
    val = np.asarray(state["split"]["validation_cases"], dtype=int)
    train = np.asarray(state["split"]["train_cases"], dtype=int)
    if len(val) == 0 or len(np.intersect1d(train, val)):
        raise ValueError("Invalid checkpoint split")
    if set(groups[train]).intersection(set(groups[val])):
        raise ValueError("Geometry group leakage detected")
    norm = state["normalization"]
    b = torch.tensor((branch[val] - np.asarray(norm["branch_mean"])) /
                     np.asarray(norm["branch_std"]), dtype=torch.float32)
    t = torch.tensor((trunk[val] - np.asarray(norm["trunk_mean"])) /
                     np.asarray(norm["trunk_std"]), dtype=torch.float32)
    model = BracketDeepONet()
    model.load_state_dict(state["state_dict"])
    model.eval()
    with torch.no_grad():
        prediction = model(b, t).numpy() * float(norm["stress_scale_pa"])
    target = stress[val]
    errors = np.linalg.norm(prediction-target, axis=1) / np.maximum(
        np.linalg.norm(target, axis=1), 1e-12)
    hotspot = np.abs(prediction.max(axis=1) - target.max(axis=1)) / np.maximum(
        target.max(axis=1), 1e-12)
    result = {
        "validation_cases": len(val),
        "validation_groups": np.unique(groups[val]).tolist(),
        "relative_l2_mean": float(errors.mean()),
        "relative_l2_median": float(np.median(errors)),
        "relative_l2_p95": float(np.percentile(errors, 95)),
        "peak_stress_relative_error_mean": float(hotspot.mean()),
        "model_verified": False,
        "source_reproduction": False,
        "warning": "No independent physical verification; sampled point maxima are not mesh-global maxima",
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("checkpoint")
    p.add_argument("--output", default="artifacts/evaluation.json")
    a = p.parse_args()
    print(json.dumps(evaluate(a.dataset, a.checkpoint, a.output), indent=2))
