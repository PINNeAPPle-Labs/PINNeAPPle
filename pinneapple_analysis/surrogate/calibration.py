"""Fine-tune trusted, already-trained torch surrogates against real measurements.

Uses the pretrained model's scalers and coordinate grid; never fits normalization
to held-out observations. The old validity domain is not silently expanded.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from .data import SurrogateData
from .methods import NonNegative, _Torch
from .model import SurrogateModel
from .search import real_data_splits
from .splits import Splits
from .study import evaluate


@dataclass
class CalibrationResult:
    model: SurrogateModel
    history: list[dict[str, float]]
    holdout: dict[str, Any]
    baseline: dict[str, Any]
    splits: Splits

    def to_dict(self) -> dict[str, Any]:
        return {"history": self.history, "baseline": self.baseline, "holdout": self.holdout,
                "splits": self.splits.to_dict(), "model_card": self.model.card}


def fine_tune_real(model: SurrogateModel, observations: SurrogateData, *, split: Splits | None = None,
                   seed: int = 0, epochs: int = 150, lr: float = 1e-4, batch: int = 64,
                   patience: int = 20, weight_decay: float = 1e-5, anchor_weight: float = 0.05,
                   freeze: tuple[str, ...] = (), allow_ood: bool = False) -> CalibrationResult:
    """Adapt MLP, FNO, DeepONet and hybrid DeepONet using independent real-data splits.

    No checkpoint is approved automatically; held-out and shift data affect only
    the final report, never optimizer steps or early stopping.
    """
    import torch

    if model.input_names != observations.input_names or model.output_names != observations.output_names:
        raise ValueError("measurement names/order differ from the pretrained surrogate")
    if tuple(model.grid_shape) != tuple(observations.grid_shape):
        raise ValueError("observation field geometry differs from the pretrained grid")
    if model.grid_shape and (model.coords is None or observations.coords is None or
                             not np.allclose(model.coords, observations.coords, rtol=1e-6, atol=1e-8)):
        raise ValueError("observation coordinates differ; align mesh/coordinates before calibration")
    if epochs < 1 or patience < 1 or batch < 1 or lr <= 0 or anchor_weight < 0 or weight_decay < 0:
        raise ValueError("invalid fine-tuning hyperparameters")
    if not allow_ood and any(x["ood"] for x in model.gate.assess(observations.X)):
        raise ValueError("real measurements include out-of-domain inputs; set allow_ood=True explicitly")
    if any(f not in ("branch", "trunk") for f in freeze):
        raise ValueError("freeze supports only branch and trunk")
    original = model.fitted
    nonnegative = isinstance(original, NonNegative)
    core = original.inner if nonnegative else original
    if not isinstance(core, _Torch):
        raise ValueError("fine_tune_real supports torch-based surrogates; sklearn/GBTs cannot be fine-tuned")
    sp = split or real_data_splits(observations, seed=seed)
    parts = [np.asarray(getattr(sp, n), dtype=int) for n in ("train", "val", "test", "shift")]
    if min(len(parts[0]), len(parts[1]), len(parts[2])) < 1:
        raise ValueError("train/validation/test observations must be present")
    if len(np.unique(np.concatenate(parts))) != sum(map(len, parts)):
        raise ValueError("overlapping calibration partitions")
    if np.any(np.concatenate(parts) >= observations.n) or np.any(np.concatenate(parts) < 0):
        raise ValueError("calibration index outside observations")
    if observations.groups is not None:
        group_sets = [set(observations.groups[p].tolist()) for p in parts]
        if any(group_sets[i] & group_sets[j] for i in range(4) for j in range(i + 1, 4)):
            raise ValueError("real-world group leakage across partitions")

    f = copy.deepcopy(core)
    device = next(f.module.parameters()).device
    dtype = next(f.module.parameters()).dtype
    f32 = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)
    width = int(np.prod(observations.Y.shape[1:]))
    Yn = observations.Y.reshape(observations.n, width).copy()
    if nonnegative:
        mask = original.mask
        if np.any(Yn[:, mask] < 0):
            raise ValueError("nonnegative outputs contain negative measurements")
        Yn[:, mask] = np.sqrt(Yn[:, mask])
    Z = f32((observations.X - f.xm) / f.xs)
    T = f32((Yn - f.ym) / f.ys)
    if f.extra:
        # Frozen physical geometry is part of the model, not learned from new targets.
        f.extra = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in f.extra.items()}
    old = copy.deepcopy(f.module).eval()
    for p in old.parameters():
        p.requires_grad_(False)
    for side in freeze:
        if not any(name.startswith(side + ".") for name, _ in f.module.named_parameters()):
            raise ValueError(f"model has no {side} subnetwork to freeze")
    for name, p in f.module.named_parameters():
        if any(name.startswith(side + ".") for side in freeze):
            p.requires_grad_(False)
    trainable = [p for p in f.module.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("fine-tuning froze all model parameters")
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    best_loss, best_state, best_epoch = float("inf"), copy.deepcopy(f.module.state_dict()), 0
    hist: list[dict[str, float]] = []
    for epoch in range(epochs):
        f.module.train()
        order = rng.permutation(sp.train)
        for ix in range(0, len(order), batch):
            ids = order[ix:ix + batch]
            prediction = f.forward_fn(f.module, Z[ids], f.extra)
            with torch.no_grad():
                baseline = f.forward_fn(old, Z[ids], f.extra)
            loss = (prediction - T[ids]).square().mean()
            if anchor_weight:
                loss = loss + anchor_weight * (prediction - baseline).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        f.module.eval()
        with torch.no_grad():
            score = float((f.forward_fn(f.module, Z[sp.val], f.extra) - T[sp.val]).square().mean())
        hist.append({"epoch": float(epoch + 1), "validation_mse_normalized": score})
        if score < best_loss:
            best_loss, best_epoch, best_state = score, epoch + 1, copy.deepcopy(f.module.state_dict())
        if epoch + 1 - best_epoch >= patience:
            break
    f.module.load_state_dict(best_state)
    f.module.eval()
    f.info = {**f.info, "fine_tuned_real": {"best_epoch": best_epoch, "validation_mse_normalized": best_loss}}
    new_fitted = NonNegative(f, original.mask) if nonnegative else f
    card = dict(model.card)
    card.pop("model_sha256", None)
    card.pop("metrics", None)
    card.pop("baseline_metrics", None)
    card["parent_model_sha256"] = model.card.get("model_sha256")
    card["dataset_sha256"] = observations.sha256()
    card["real_data_calibration"] = {"seed": seed, "best_epoch": best_epoch, "lr": lr, "batch": batch,
                                     "anchor_weight": anchor_weight, "frozen": list(freeze),
                                     "allow_ood": allow_ood, "splits": sp.to_dict()}
    card["trust"] = {"decision": "not_verified",
                     "reason": "adapted checkpoint requires independent domain and physics review"}
    adapted = SurrogateModel(new_fitted, model.gate, input_names=model.input_names,
                             output_names=model.output_names, grid_shape=model.grid_shape,
                             coords=model.coords, coord_names=model.coord_names, card=card)
    scale = observations.Y[sp.train].reshape(len(sp.train), -1).std(0) + 1e-12
    final, baseline = {}, {}
    for name in ("val", "test", "shift"):
        ids = np.asarray(getattr(sp, name), dtype=int)
        if not len(ids):
            continue
        Y = observations.Y[ids].reshape(len(ids), -1)
        final[name] = evaluate(observations.kind, adapted.fitted.predict(observations.X[ids]), Y,
                               scale, observations.output_names)
        baseline[name] = evaluate(observations.kind, original.predict(observations.X[ids]), Y,
                                  scale, observations.output_names)
    adapted.card["metrics"] = final
    adapted.card["baseline_metrics"] = baseline
    return CalibrationResult(adapted, hist, final, baseline, sp)
