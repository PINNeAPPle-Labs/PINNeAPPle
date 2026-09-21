"""A joint, multi-case extension of :mod:`dense_volume_operator`: one
:class:`~pinneapple_neural.architectures.neural_operators.fno.FNO3d`
trained across SEVERAL real dense-in-time volume time series that share
the same physical problem but differ in a scalar (or low-dim) physical
parameter ``theta`` -- e.g. several real OpenFOAM channel-flow runs at
different friction Reynolds numbers ``Re_tau``.

Why a new module instead of extending :func:`dense_volume_operator
.train_dense_volume_operator` in place: that function's signature and
checkpoint schema are a real, shipped contract (``PINNeAPPle-SplashCFD``'s
``channel_wale_fno3d`` model was trained and is served through it) --
changing its meaning out from under existing callers would be a breaking
change, not an addition. This module is purely additive: it reuses
``FNO3d`` completely unmodified (no new architecture code), it just adds
``theta`` as extra constant input channels, broadcast over the spatial
grid -- the exact mechanism ``FNO3d``'s own ``use_grid=True`` already uses
for coordinate channels, generalised from "channels the network can't
predict, only condition on" (grid coordinates) to "a channel that also
happens to be a channel the network can't predict, only condition on"
(the physical parameter). See ``PINNeAPPle-SplashCFD/ROADMAP.md`` item 1
("The parametric/multi-case PINN track") for the product motivation.

Design decisions worth stating explicitly:

- **Global (not per-case) normalization.** ``dense_volume_operator``
  normalizes with one case's own mean/std. A parametric model must be
  queryable at a ``theta`` that was never trained on (that's the entire
  point -- interpolating across ``Re_tau``, not memorizing a finite set
  of cases), so there is no "this case's std" to look up at inference
  time for an unseen ``theta``. Normalization here is computed ONCE over
  ALL training cases pooled together and stored in the checkpoint,
  exactly like a single global case would be.
- **theta is normalized too**, log-scale min/max to ``[-1, 1]`` over the
  training cases' ``theta`` values (physical parameters like ``Re_tau``
  are typically swept over multiplicative/log-spaced ranges, not linear
  ones -- e.g. 150..5200 spans 1.5 decades). The log-min/max are stored
  in the checkpoint so a query at an interpolated ``theta`` (never seen
  in training) normalizes consistently.
- **theta is a condition, not a predicted state.** The network's
  ``out_channels`` stays equal to the state dimension (e.g. 4 for
  u,v,w,p) -- theta is constant along a real trajectory (it is a fixed
  property of which physical case you're in, not part of what evolves in
  time), so :func:`rollout_parametric_dense_volume_operator` re-attaches
  the same (normalized) theta channel(s) after every autoregressive step
  before feeding the prediction back in, rather than asking the network
  to also predict/carry it forward.
- **One case per gradient step.** Each training step samples a batch's
  start indices from a single, randomly chosen (weighted by that case's
  number of real training pairs) case -- simpler and more general than
  requiring every case to share the exact same grid shape to be mixed
  within one physical batch tensor (this module does require a shared
  ``(D, H, W)`` across cases today via a plain ``torch.stack`` inside a
  batch, but keeps cases logically separate so that requirement is the
  only one, not "same everything").
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from pinneapple_neural.architectures.neural_operators.fno import FNO3d
from pinneapple_neural.trainer.checkpoint import Checkpoint
from pinneapple_neural.workflows.dense_volume_operator import _best_device

__all__ = [
    "ParametricCase",
    "ParametricDenseVolumeOperatorConfig",
    "train_parametric_dense_volume_operator",
    "load_parametric_dense_volume_operator",
    "predict_single_step_parametric",
    "rollout_parametric_dense_volume_operator",
]


@dataclass
class ParametricCase:
    """One real training case: a dense-in-time volume sequence plus the
    scalar (or low-dim) physical parameter(s) that produced it."""

    name: str
    frames: np.ndarray  # (T, C, D, H, W), real units, NOT normalized
    theta: Sequence[float]  # raw (un-normalized) physical parameter value(s), e.g. [Re_tau]


@dataclass
class ParametricDenseVolumeOperatorConfig:
    state_channels: int  # e.g. 4 for (u, v, w, p) -- NOT counting theta channels
    theta_dim: int = 1
    width: int = 16
    modes: int = 8
    layers: int = 4
    epochs: int = 600
    lr: float = 2e-3
    val_frac: float = 0.15  # held-out TAIL of each case's own time series (temporal generalization)
    batch_size: int = 4
    rollout_steps: int = 20
    log_every: int = 25
    device: str = "auto"
    theta_log_scale: bool = True  # log10-transform theta before min/max normalization (see module docstring)
    pushforward_max_steps: int = 4
    pushforward_prob: float = 0.5
    pushforward_warmup_epochs: int = 50

    @property
    def in_channels(self) -> int:
        return self.state_channels + self.theta_dim


def _theta_stats(cases: Sequence[ParametricCase], log_scale: bool) -> Tuple[np.ndarray, np.ndarray]:
    theta_mat = np.asarray([c.theta for c in cases], dtype=np.float64)  # (n_cases, theta_dim)
    if log_scale:
        if (theta_mat <= 0).any():
            raise ValueError("theta_log_scale=True requires all theta values to be strictly positive")
        theta_mat = np.log10(theta_mat)
    tmin = theta_mat.min(axis=0)
    tmax = theta_mat.max(axis=0)
    span = np.clip(tmax - tmin, 1e-8, None)  # guard a degenerate single-case sweep
    return tmin, span


def _normalize_theta(theta: Sequence[float], tmin: np.ndarray, span: np.ndarray, log_scale: bool) -> np.ndarray:
    t = np.asarray(theta, dtype=np.float64)
    if log_scale:
        t = np.log10(t)
    return (2.0 * (t - tmin) / span - 1.0).astype(np.float32)  # -> [-1, 1]


def _broadcast_theta(theta_n: np.ndarray, D: int, H: int, W: int) -> torch.Tensor:
    """(theta_dim,) normalized -> (theta_dim, D, H, W) constant field."""
    t = torch.as_tensor(theta_n, dtype=torch.float32).view(-1, 1, 1, 1)
    return t.expand(-1, D, H, W).clone()


def train_parametric_dense_volume_operator(
    cases: Sequence[ParametricCase],
    cfg: ParametricDenseVolumeOperatorConfig,
    *, seed: int = 0,
) -> Tuple[Checkpoint, List[Dict[str, Any]]]:
    """Jointly train one theta-conditioned :class:`FNO3d` across ``cases``.

    Mirrors :func:`dense_volume_operator.train_dense_volume_operator`'s
    single-step + pushforward training loop (same stabilization, same
    reason -- see that module's docstring), generalised to sample batches
    from a randomly chosen case (weighted by its number of real training
    pairs) at each step instead of always the one case there is.
    """
    if len(cases) < 2:
        raise ValueError("train_parametric_dense_volume_operator needs >=2 cases -- a single case is just "
                          "dense_volume_operator.train_dense_volume_operator")

    device = _best_device(cfg.device)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    D, H, W = cases[0].frames.shape[-3:]
    for c in cases:
        if c.frames.shape[-3:] != (D, H, W):
            raise ValueError(f"case '{c.name}' has grid shape {c.frames.shape[-3:]}, expected {(D, H, W)} "
                              f"(all cases must share a grid shape to be batched together)")
        if c.frames.shape[1] != cfg.state_channels:
            raise ValueError(f"case '{c.name}' has {c.frames.shape[1]} state channels, cfg says {cfg.state_channels}")
        if len(c.theta) != cfg.theta_dim:
            raise ValueError(f"case '{c.name}' has theta_dim {len(c.theta)}, cfg says {cfg.theta_dim}")

    all_frames = np.concatenate([c.frames for c in cases], axis=0)
    mean = all_frames.mean(axis=(0, 2, 3, 4), keepdims=True)
    std = all_frames.std(axis=(0, 2, 3, 4), keepdims=True).clip(min=1e-8)

    theta_min, theta_span = _theta_stats(cases, cfg.theta_log_scale)

    # Per-case: normalized frames tensor, normalized theta broadcast field, train/val split index.
    case_data = []
    for c in cases:
        frames_n = (c.frames - mean) / std
        X = torch.as_tensor(frames_n, dtype=torch.float32)  # (T, state_c, D, H, W)
        theta_n = _normalize_theta(c.theta, theta_min, theta_span, cfg.theta_log_scale)
        theta_field = _broadcast_theta(theta_n, D, H, W)  # (theta_dim, D, H, W), constant across time

        T = X.shape[0]
        n_pairs = T - 1
        n_val = max(1, int(cfg.val_frac * n_pairs)) if n_pairs > 1 else 0
        n_train = max(1, n_pairs - n_val)
        case_data.append({
            "name": c.name, "X": X, "theta_field": theta_field, "theta_n": theta_n,
            "n_train": n_train, "n_val": n_val, "n_pairs": n_pairs,
        })

    net = FNO3d(
        in_channels=cfg.in_channels, out_channels=cfg.state_channels, width=cfg.width,
        modes1=cfg.modes, modes2=cfg.modes, modes3=cfg.modes, layers=cfg.layers, use_grid=True,
    ).to(device)
    n_params = sum(p.numel() for p in net.parameters())

    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)
    loss_fn = nn.MSELoss()

    def _with_theta(x_state: torch.Tensor, theta_field: torch.Tensor) -> torch.Tensor:
        # x_state: (B, state_c, D, H, W); theta_field: (theta_dim, D, H, W) -> broadcast over batch
        tb = theta_field.unsqueeze(0).expand(x_state.shape[0], -1, -1, -1, -1).to(x_state.device)
        return torch.cat([x_state, tb], dim=1)

    # Val tensors, one concatenated batch per case (small: val_frac default 0.15).
    val_batches = []
    for cd in case_data:
        if cd["n_val"] == 0:
            continue
        s, e = cd["n_train"], cd["n_train"] + cd["n_val"]
        Xv = cd["X"][s:e].to(device)
        Yv = cd["X"][s + 1:e + 1].to(device)
        val_batches.append((cd["name"], Xv, Yv, cd["theta_field"]))

    weights = np.asarray([cd["n_train"] for cd in case_data], dtype=np.float64)
    weights /= weights.sum()

    history: List[Dict[str, Any]] = []
    t_start = time.time()
    bs = max(1, cfg.batch_size)
    steps_per_epoch = max(1, int(round(sum(cd["n_train"] for cd in case_data) / bs)))

    for epoch in range(cfg.epochs):
        net.train()
        use_pushforward = epoch >= cfg.pushforward_warmup_epochs and cfg.pushforward_max_steps > 1
        epoch_loss = 0.0

        for _ in range(steps_per_epoch):
            ci = rng.choice(len(case_data), p=weights)
            cd = case_data[ci]
            n_train = cd["n_train"]
            idx = rng.integers(0, n_train, size=min(bs, n_train))
            theta_field = cd["theta_field"]

            opt.zero_grad(set_to_none=True)

            if use_pushforward and rng.random() < cfg.pushforward_prob:
                k_max_per_sample = [min(cfg.pushforward_max_steps, n_train - int(s)) for s in idx]
                k = max(1, min(k_max_per_sample))
                cur = cd["X"][idx].to(device)
                with torch.no_grad():
                    for _ in range(k - 1):
                        cur = net(_with_theta(cur, theta_field)).y
                target = cd["X"][[int(s) + k for s in idx]].to(device)
                pred = net(_with_theta(cur, theta_field)).y
                loss = loss_fn(pred, target)
            else:
                x_batch = cd["X"][idx].to(device)
                y_batch = cd["X"][[int(s) + 1 for s in idx]].to(device)
                pred = net(_with_theta(x_batch, theta_field)).y
                loss = loss_fn(pred, y_batch)

            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        sched.step()
        epoch_loss /= steps_per_epoch

        if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
            net.eval()
            with torch.no_grad():
                val_losses = {}
                for name, Xv, Yv, theta_field in val_batches:
                    if Xv.shape[0] == 0:
                        continue
                    val_losses[name] = loss_fn(net(_with_theta(Xv, theta_field)).y, Yv).item()
                val_loss = float(np.mean(list(val_losses.values()))) if val_losses else float("nan")
            elapsed = time.time() - t_start
            history.append({
                "epoch": epoch, "train_mse": epoch_loss, "val_mse": val_loss,
                "val_mse_by_case": val_losses, "elapsed_s": elapsed, "used_pushforward": use_pushforward,
            })

    checkpoint = Checkpoint(
        model_state=net.state_dict(), optim_state=opt.state_dict(),
        cfg={"state_channels": cfg.state_channels, "theta_dim": cfg.theta_dim, "in_channels": cfg.in_channels,
             "out_channels": cfg.state_channels},
        meta={
            "width": cfg.width, "modes": cfg.modes, "layers": cfg.layers,
            "mean": mean.tolist(), "std": std.tolist(),
            "theta_min": theta_min.tolist(), "theta_span": theta_span.tolist(),
            "theta_log_scale": cfg.theta_log_scale,
            "case_names": [c.name for c in cases], "case_theta": [list(c.theta) for c in cases],
            "epochs": cfg.epochs, "device": device, "n_params": n_params,
            "pushforward_max_steps": cfg.pushforward_max_steps, "pushforward_prob": cfg.pushforward_prob,
            "pushforward_warmup_epochs": cfg.pushforward_warmup_epochs,
        },
    )
    return checkpoint, history


def load_parametric_dense_volume_operator(checkpoint: Checkpoint, *, device: str = "cpu") -> nn.Module:
    meta, cfg = checkpoint.meta, checkpoint.cfg
    net = FNO3d(
        in_channels=cfg["in_channels"], out_channels=cfg["out_channels"],
        width=meta["width"], modes1=meta["modes"], modes2=meta["modes"], modes3=meta["modes"],
        layers=meta["layers"], use_grid=True,
    ).to(device)
    net.load_state_dict(checkpoint.model_state)
    net.eval()
    return net


def _theta_field_from_checkpoint(checkpoint: Checkpoint, theta: Sequence[float], D: int, H: int, W: int) -> torch.Tensor:
    meta = checkpoint.meta
    theta_min = np.asarray(meta["theta_min"], dtype=np.float64)
    theta_span = np.asarray(meta["theta_span"], dtype=np.float64)
    theta_n = _normalize_theta(theta, theta_min, theta_span, bool(meta["theta_log_scale"]))
    return _broadcast_theta(theta_n, D, H, W)


def predict_single_step_parametric(
    checkpoint: Checkpoint, input_frame: np.ndarray, theta: Sequence[float], *, device: str = "cpu",
) -> np.ndarray:
    """One forward pass at a given (possibly never-trained-on) ``theta``.
    ``input_frame``: (state_c, D, H, W), real units in, real units out."""
    net = load_parametric_dense_volume_operator(checkpoint, device=device)
    mean = np.array(checkpoint.meta["mean"], dtype=np.float32)
    std = np.array(checkpoint.meta["std"], dtype=np.float32)
    _, D, H, W = input_frame.shape
    theta_field = _theta_field_from_checkpoint(checkpoint, theta, D, H, W).to(device)

    x_n = torch.as_tensor((input_frame[None] - mean) / std, device=device, dtype=torch.float32)
    tb = theta_field.unsqueeze(0)
    with torch.no_grad():
        pred_n = net(torch.cat([x_n, tb], dim=1)).y[0].cpu().numpy()
    return pred_n * std[0] + mean[0]


def rollout_parametric_dense_volume_operator(
    checkpoint: Checkpoint, initial_frame: np.ndarray, theta: Sequence[float], n_steps: int, *, device: str = "cpu",
) -> np.ndarray:
    """Autoregressive rollout at a given ``theta``, re-attaching the
    (constant) theta channel after every step -- see module docstring
    ("theta is a condition, not a predicted state"). Returns
    ``(n_steps + 1, state_c, D, H, W)``, index 0 = ``initial_frame``."""
    net = load_parametric_dense_volume_operator(checkpoint, device=device)
    mean = np.array(checkpoint.meta["mean"], dtype=np.float32)
    std = np.array(checkpoint.meta["std"], dtype=np.float32)
    _, D, H, W = initial_frame.shape
    theta_field = _theta_field_from_checkpoint(checkpoint, theta, D, H, W).to(device)
    tb = theta_field.unsqueeze(0)

    cur_n = torch.as_tensor((initial_frame[None] - mean) / std, device=device, dtype=torch.float32)
    out = [initial_frame.astype(np.float32)]
    with torch.no_grad():
        for _ in range(n_steps):
            cur_n = net(torch.cat([cur_n, tb], dim=1)).y
            pred = cur_n[0].cpu().numpy() * std[0] + mean[0]
            out.append(pred)
    return np.stack(out, axis=0)
