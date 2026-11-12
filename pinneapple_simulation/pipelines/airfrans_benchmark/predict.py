"""Running a trained network over a split and packing the participant entry the benchmark workflow consumes."""
from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .features import point_features
from .model import CoordinateNet
from .train import pick_device

__all__ = ["PredictionSpec", "predict_cases"]


@dataclass(frozen=True)
class PredictionSpec:
    """How the inference campaign is run and how its cost is measured."""
    chunk: int = 200_000
    device: str | None = None
    overwrite: bool = True


DEFAULT_PREDICTION = PredictionSpec()


def predict_cases(store: str, weights: str, case_ids: Sequence[str], entry_dir: str,
                  spec: PredictionSpec | None = None, progress=None) -> dict[str, Any]:
    """Write ``<entry>/predictions/<case_id>.npz`` for every case, in the unchanged native point order.

    The arrays are the four the AirfRANS participant workflow reads: ``velocity`` (N, 2), ``pressure`` (N,),
    ``internal_points`` (N, 2) and ``airfoil_points`` (M, 2) — finite, denormalised, native order. A reused entry
    directory is cleared first, because the workflow refuses a directory that carries another split's predictions."""
    import torch

    spec = spec or DEFAULT_PREDICTION
    device = pick_device(spec.device)
    payload = torch.load(weights, map_location="cpu", weights_only=False)
    model = CoordinateNet(payload["width"], payload["depth"]).eval().to(device)
    model.load_state_dict(payload["state_dict"])
    out_dir = os.path.join(entry_dir, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    if spec.overwrite:
        for name in os.listdir(out_dir):
            if name.endswith(".npz"):
                os.remove(os.path.join(out_dir, name))

    started, written = time.time(), []
    for done, case_id in enumerate(case_ids, 1):
        with np.load(os.path.join(store, f"{case_id}.npz")) as data:
            features = data["features"]
            airfoil = data["airfoil_points"]
            inlet = float(data["inlet_velocity"][0])
        feats = point_features(features, airfoil)
        chunks = []
        with torch.no_grad():
            for start in range(0, len(feats), spec.chunk):
                block = torch.from_numpy(feats[start:start + spec.chunk]).to(device)
                chunks.append(model(block).cpu().numpy())
        scaled = np.concatenate(chunks, axis=0).astype(np.float64)
        np.savez_compressed(os.path.join(out_dir, f"{case_id}.npz"),
                            velocity=scaled[:, :2] * inlet, pressure=scaled[:, 2] * inlet ** 2,
                            internal_points=np.asarray(features[:, :2], np.float64),
                            airfoil_points=np.asarray(airfoil, np.float64))
        written.append(case_id)
        if progress:
            progress(done, len(case_ids))
    record = {"weights": os.path.abspath(weights), "cases": len(written), "device": device,
              "seconds": round(time.time() - started, 1)}
    with open(os.path.join(entry_dir, "predict.json"), "w") as handle:
        json.dump(record, handle, indent=1)
    return record
