"""Training and inference of a set-to-set field surrogate: one geometry in, the field on its own points out.

The library trains pointwise fields (``pinneapple_arena.pointwise.training``) and grid operators, but a surrogate over
*many geometries* needs a different batching: every sample is a geometry with its own point count, and the batch is
padded to the largest one, with a mask that keeps the padding out of the pooling and the loss. That is what this
trainer does, plus the bookkeeping a benchmark row needs (target standardisation from the training split only,
early stopping on the validation split, best-weights restore, gradient clipping, and the cost of a forward pass).

    from pinneapple_arena.fieldbench.trainer import train_field_model, predict_field_model
    hist = train_field_model(model, dataset, train_ids, val_ids, epochs=30, device="cpu")
    preds = predict_field_model(model, dataset, dataset.splits["test"], device=hist["device"])

Nothing here is tuned per architecture: the same loop trains every model row of a benchmark, which is the point of a
benchmark (a different budget per model makes the comparison meaningless). When a model needs a different budget the
caller records it in the row's notes.
"""
from __future__ import annotations

import time

import numpy as np
import torch

__all__ = ["train_field_model", "predict_field_model", "collate_samples"]


def _best_device() -> str:
    try:
        from pinneapple_neural.trainer import best_device
        return str(best_device()).split(":")[0]
    except Exception:
        return "cpu"


def collate_samples(dataset, ids, *, standardize: dict | None = None, uses_parts: bool = True,
                    uses_extra: bool = True) -> dict:
    """Pad a list of samples into one batch.

    Returns a dict with ``x`` (B, N, 3), ``mask`` (B, N), the optional ``part``/``part_id``/``thickness``/``extra``
    /``time`` channels, ``y`` (B, N, C) or (B, T, N, C) standardised when ``standardize`` is given, ``channels``, and
    the point count of every sample (to slice the padded prediction back). ``extra`` is the sample's own per-node
    channels — the design vector of a parametric benchmark, broadcast to its nodes — padded with zeros like the rest,
    and left out entirely when ``uses_extra`` is false (the geometry-only ablation reads the same dataset)."""
    samples = [dataset[i] for i in ids]
    n_max = max(s.n_points for s in samples)
    b = len(samples)
    transient = samples[0].transient
    first = np.asarray(samples[0].values)
    channels = int(first.shape[-1])
    n_frames = int(first.shape[0]) if transient else 0
    x = np.zeros((b, n_max, 3), np.float32)
    mask = np.zeros((b, n_max), bool)
    part = np.zeros((b, n_max), np.int64)
    part_id = np.zeros((b, n_max), np.int64)
    thick = np.ones((b, n_max), np.float32)
    with_extra = next((s for s in samples if s.extra is not None), None) if uses_extra else None
    extra = None if with_extra is None else np.zeros((b, n_max, int(np.asarray(with_extra.extra).shape[-1])),
                                                     np.float32)
    y = np.zeros((b, n_frames, n_max, channels) if transient else (b, n_max, channels), np.float32)
    for i, s in enumerate(samples):
        n = s.n_points
        x[i, :n] = s.points
        mask[i, :n] = True
        part[i, :n] = np.asarray(s.parts, np.int64) if s.parts is not None else 1
        part_id[i, :n] = np.asarray(s.part_ids, np.int64) if s.part_ids is not None else part[i, :n]
        thick[i, :n] = np.asarray(s.thickness, np.float32) if s.thickness is not None else 1.0
        if extra is not None:
            if s.extra is None:
                raise ValueError(f"sample {s.id} carries no extra channels while the rest of the batch does: the "
                                 f"design vector of a parametric benchmark is either on every sample or on none")
            extra[i, :n] = np.asarray(s.extra, np.float32).reshape(n, extra.shape[-1])
        v = np.asarray(s.values, np.float32).reshape((n_frames, n, channels) if transient else (n, channels))
        if transient:
            y[i, :, :n] = v
        else:
            y[i, :n] = v
    out = {"x": torch.from_numpy(x), "mask": torch.from_numpy(mask), "y": torch.from_numpy(y),
           "channels": channels, "transient": transient, "lengths": [s.n_points for s in samples]}
    if uses_parts:
        out["part"] = torch.from_numpy(part)
    out["part_id"] = torch.from_numpy(part_id)
    out["thickness"] = torch.from_numpy(thick)
    if extra is not None:
        out["extra"] = torch.from_numpy(extra)
    if samples[0].times is not None:
        out["time"] = torch.from_numpy(np.stack([np.asarray(s.times, np.float32) for s in samples]))
    return out


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over the valid points (and frames) only.

    A prediction with fewer channels than the target does not raise on its own: torch broadcasts it, the loss comes out
    finite, and the benchmark then scores a one-component field against a three-component displacement — which is how a
    row built for the wrong ``out_dim`` reported a plausible error for a model that never predicted the field."""
    if pred.ndim == 4:                                     # (B, T, N, C)
        if target.ndim == 3:
            target = target.unsqueeze(1).expand(-1, pred.shape[1], -1, -1)
        m = mask[:, None, :, None].to(pred.dtype)
    else:
        if target.ndim == 4:
            target = target[:, 0]
        m = mask[:, :, None].to(pred.dtype)
    if pred.shape[-1] != target.shape[-1]:
        raise ValueError(f"the model predicts {pred.shape[-1]} channel(s) and the field has {target.shape[-1]}: build "
                         f"the row for the field it is scored on (build_model_specs(out_dim=...))")
    m = m.expand_as(pred)
    return (((pred - target) ** 2) * m).sum() / m.sum().clamp(min=1.0)


def _inputs_of(batch: dict, device: str) -> dict:
    return {k: v.to(device) for k, v in batch.items()
            if k in ("x", "part", "part_id", "thickness", "extra", "time", "mask")}


def train_field_model(model, dataset, *, train_ids, val_ids, epochs: int = 30, batch_size: int = 2,
                      lr: float = 1e-3, weight_decay: float = 1e-5, device: str | None = None, patience: int = 10,
                      seed: int = 0, standardize: bool = True, grad_clip: float = 1.0, uses_parts: bool = True,
                      uses_extra: bool = True, verbose: bool = False) -> dict:
    """Train one surrogate on the training split, early-stop on the validation split, restore the best weights."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device = device or _best_device()
    dataset_ids = [dataset[i] for i in train_ids]
    stats = None
    if standardize:
        flat = np.concatenate([np.asarray(s.values, np.float64).reshape(-1, np.asarray(s.values).shape[-1])
                               for s in dataset_ids], axis=0)               # every point of every training design
        stats = {"mean": flat.mean(axis=0).astype(np.float32), "std": (flat.std(axis=0) + 1e-8).astype(np.float32)}
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=lr / 100.0)
    rng = np.random.default_rng(seed)
    best = {"val": float("inf"), "epoch": -1, "state": None}
    history = {"train": [], "val": []}
    n_train = len(train_ids)

    def loss_of(ids) -> float:
        batch = collate_samples(dataset, ids, standardize=stats, uses_parts=uses_parts, uses_extra=uses_extra)
        with torch.no_grad():
            pred = model(**_inputs_of(batch, device))
            return float(_masked_mse(pred, batch["y"].to(device), batch["mask"].to(device)))

    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(n_train)
        running, seen = 0.0, 0
        for start in range(0, n_train, batch_size):
            ids = [train_ids[k] for k in order[start:start + batch_size]]
            batch = collate_samples(dataset, ids, standardize=stats, uses_parts=uses_parts, uses_extra=uses_extra)
            pred = model(**_inputs_of(batch, device))
            loss = _masked_mse(pred, batch["y"].to(device), batch["mask"].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            running += float(loss.detach()) * len(ids)
            seen += len(ids)
        sched.step()
        tr = running / max(seen, 1)
        va = loss_of(val_ids) if val_ids else float("nan")
        history["train"].append(tr)
        history["val"].append(va)
        if val_ids and va < best["val"]:
            best = {"val": va, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        if verbose:
            print(f"epoch {epoch + 1}/{epochs}  train {tr:.4g}  val {va:.4g}")
        if val_ids and epoch - best["epoch"] >= patience:
            break
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    model._pinneapple_standardize = stats                                       # noqa: SLF001  (undone in predict)
    # the cost of one geometry (batch of one, the whole test point set) — measured, not estimated
    sample = dataset[val_ids[0] if val_ids else train_ids[0]]
    inputs = {k: torch.from_numpy(np.ascontiguousarray(v))
              for k, v in sample.inputs(with_parts=uses_parts, with_extra=uses_extra).items()}
    with torch.no_grad():
        for _ in range(2):
            model(**{k: v.to(device) for k, v in inputs.items()})
        t0 = time.perf_counter()
        for _ in range(5):
            model(**{k: v.to(device) for k, v in inputs.items()})
        latency = (time.perf_counter() - t0) / 5
    return {"history": history, "best_val": best["val"], "best_epoch": best["epoch"], "device": device,
            "standardize": stats, "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
            "latency_ms": latency * 1e3, "throughput_per_s": 1.0 / max(latency, 1e-9),
            "val_loss": best["val"], "epochs_run": len(history["train"])}


def predict_field_model(model, dataset, ids, *, device: str = "cpu", uses_parts: bool = True,
                        uses_extra: bool = True, batch_size: int = 2) -> list[np.ndarray]:
    """Predict every sample of ``ids``; returns one array per sample with the field's own shape (physical units)."""
    model = model.to(device).eval()
    out = []
    stats = getattr(model, "_pinneapple_standardize", None)
    mean = None if not stats else np.asarray(stats["mean"], np.float32)
    std = None if not stats else np.asarray(stats["std"], np.float32)
    for start in range(0, len(ids), batch_size):
        chunk = ids[start:start + batch_size]
        batch = collate_samples(dataset, chunk, standardize=None, uses_parts=uses_parts, uses_extra=uses_extra)
        inputs = {k: v.to(device) for k, v in batch.items()
                  if k in ("x", "part", "part_id", "thickness", "extra", "time", "mask")}
        with torch.no_grad():
            pred = model(**inputs).cpu().numpy()
        if mean is not None:                                   # back to physical units
            pred = pred * std.reshape(-1) + mean.reshape(-1)
        for j, sample_id in enumerate(chunk):
            s = dataset[sample_id]
            n = s.n_points
            p = np.asarray(pred[j])
            # a model may answer with the padded batch width and, for a transient field, with a frame axis that a
            # steady sample does not have: reduce to the sample's own shape before slicing
            if not s.transient:
                while p.ndim > 2:
                    p = p[0] if p.shape[0] == 1 else p.reshape(-1, p.shape[-1])
                p = p[:n, :]
            else:
                if p.ndim == 2:
                    p = p[None]
                p = p[:, :n, :]
            channels = int(np.asarray(s.values).shape[-1])
            if p.shape[-1] != channels:
                raise ValueError(f"sample {s.id} has a {channels}-component field and the model answered with "
                                 f"{p.shape[-1]}: the row was built for another field")
            out.append(p[..., 0] if p.shape[-1] == 1 else p)
    return out
