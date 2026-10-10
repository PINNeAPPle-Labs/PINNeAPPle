"""Black-hole weather forecasting: the U-Net of Duarte, Nemmen & Navarro (2022), a trust horizon and a physics check.

The original (TensorFlow, MIT licence, github.com/black-hole-group/DL_BH_fluids) is ported layer for layer:

* input: 5 consecutive log-normalised density snapshots, stacked as channels (time, not physical variables);
  output: the next 5 snapshots;
* U-Net with 5x5 convolutions and LeakyReLU (Keras default slope 0.3), filters 2f, 4f, 8f, 16f, a 16f
  bottleneck, nearest upsampling with skip connections and a linear 1x1 output (f = 32 in the paper);
* density normalisation (log rho - log rho_min) / (log rho_max - log rho_min) and a crop of the tenuous outer
  atmosphere and of the polar cells;
* the multi-simulation loss ``MAE + alpha * MAE(high-density pixels, y > 0.5)``. The one-simulation loss of the
  paper weights fixed pixel boxes of its 256 x 192 grid, which do not transfer to another grid, so the
  grid-independent loss is used for both set-ups.

Added here (issue #399):

* ``rollout`` + ``lead_time_scores``: error by lead time against persistence and against the time-mean flow,
  and the **trust horizon** -- the lead time up to which the forecast beats persistence and keeps a pattern
  correlation above a threshold (same idea as ``pinneapple_physics.weather.evaluate.horizons``);
* ``MassEnvelope``: a check that needs no ground truth. The mass in the window can only change by what crosses
  its edges, and the training simulations bound how fast that happens; a rollout whose mass moves faster than
  anything seen in training is creating or destroying gas (the "artificial mass injection" the original
  authors report as the cause of drift).
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DuarteUNet", "DensityCodec", "make_blocks", "train_forecaster", "rollout", "lead_time_scores",
           "trust_horizon", "MassEnvelope", "MassProjection", "load_forecaster", "window_mass"]


# ---------------------------------------------------------------------- data
@dataclass
class DensityCodec:
    """Log-normalisation and crop shared by training and inference (stored with the checkpoint)."""
    log_min: float
    log_max: float
    r_cells: int                       # keep radial cells [0, r_cells)
    theta_trim: int                    # drop this many cells at each pole

    @classmethod
    def fit(cls, rho: np.ndarray, r_cells: int, theta_trim: int) -> "DensityCodec":
        crop = rho[..., :r_cells, theta_trim:rho.shape[-1] - theta_trim]
        lg = np.log10(np.clip(crop, 1e-30, None))
        return cls(float(lg.min()), float(lg.max()), int(r_cells), int(theta_trim))

    def crop(self, a: np.ndarray) -> np.ndarray:
        return a[..., :self.r_cells, self.theta_trim:a.shape[-1] - self.theta_trim]

    def encode(self, rho: np.ndarray) -> np.ndarray:
        lg = np.log10(np.clip(self.crop(rho), 1e-30, None))
        return ((lg - self.log_min) / (self.log_max - self.log_min)).astype(np.float32)

    def decode(self, x: np.ndarray) -> np.ndarray:
        """Normalised -> density (cropped window)."""
        return 10.0 ** (np.asarray(x, np.float64) * (self.log_max - self.log_min) + self.log_min)


def make_blocks(x: np.ndarray, k: int = 5, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Frames (T, H, W) -> inputs (N, k, H, W) = frames i..i+k-1 and targets = frames i+k..i+2k-1."""
    idx = np.arange(0, len(x) - 2 * k + 1, stride)
    X = np.stack([x[i:i + k] for i in idx])
    Y = np.stack([x[i + k:i + 2 * k] for i in idx])
    return X, Y


# ---------------------------------------------------------------------- model
class _Conv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 5, padding=2), nn.LeakyReLU(0.3))


def _block(cin, cout):
    return nn.Sequential(_Conv(cin, cout), _Conv(cout, cout))


class DuarteUNet(nn.Module):
    """The U-Net of Duarte et al. (2022) (``create_auto_encoder`` in the original ``src/models.py``).

    ``residual=True`` (not in the paper) predicts the change from the last input frame instead of the frames
    themselves: the output is ``x[:, -1:] + net(x)``. When the flow changes little between frames, the absolute
    prediction spends its accuracy on reproducing the state; the residual one starts from persistence."""

    def __init__(self, filters: int = 32, frames: int = 5, residual: bool = False):
        super().__init__()
        self.residual = residual
        m = filters
        c = [2 * m, 4 * m, 8 * m, 16 * m]
        self.e1, self.e2, self.e3, self.e4 = _block(frames, c[0]), _block(c[0], c[1]), _block(c[1], c[2]), _block(c[2], c[3])
        self.mid = _block(c[3], c[3])
        self.d4 = _block(2 * c[3], c[3])
        self.d3 = _block(c[3] + c[2], c[2])
        self.d2 = _block(c[2] + c[1], c[1])
        self.d1 = _block(c[1] + c[0], c[0])
        self.out = nn.Conv2d(c[0], frames, 1)
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")   # Keras he_normal
                nn.init.zeros_(mod.bias)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        b = self.mid(F.max_pool2d(e4, 2))
        up = lambda a: F.interpolate(a, scale_factor=2, mode="nearest")
        d = self.d4(torch.cat([up(b), e4], 1))
        d = self.d3(torch.cat([up(d), e3], 1))
        d = self.d2(torch.cat([up(d), e2], 1))
        d = self.d1(torch.cat([up(d), e1], 1))
        y = self.out(d)
        return x[:, -1:] + y if self.residual else y


def duarte_loss(pred, target, alpha: float = 8.0):
    """MAE over the window + alpha * MAE over the high-density pixels (target > 0.5)."""
    total = (pred - target).abs().mean()
    hd = target > 0.5
    high = (pred - target).abs()[hd].mean() if hd.any() else pred.new_zeros(())
    return total + alpha * high


# ---------------------------------------------------------------------- training
@dataclass
class TrainConfig:
    filters: int = 32
    frames: int = 5
    residual: bool = False
    epochs: int = 40
    batch_size: int = 16
    lr: float = 2e-4                   # the paper uses 5e-4 with batch 64; at batch 16 that diverges here
    alpha: float = 8.0
    clip: float = 1.0                  # gradient-norm clipping (not in the paper; keeps the deep U-Net stable)
    bf16: bool = False                 # CPU autocast (helps only on CPUs with AMX)
    seed: int = 0
    max_minutes: float = 1e9


def train_forecaster(X: np.ndarray, Y: np.ndarray, Xv: np.ndarray, Yv: np.ndarray, cfg: TrainConfig,
                     codec: DensityCodec, out_dir: str, *, log=print) -> Dict[str, list]:
    """Adam + the Duarte loss; keeps the weights with the best validation loss in ``out_dir/forecaster.pt``."""
    torch.manual_seed(cfg.seed)
    os.makedirs(out_dir, exist_ok=True)
    model = DuarteUNet(cfg.filters, cfg.frames, cfg.residual)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    Xvt, Yvt = torch.from_numpy(Xv), torch.from_numpy(Yv)
    hist = {"train": [], "val": [], "seconds": []}
    best, t0 = math.inf, time.time()
    rng = np.random.default_rng(cfg.seed)
    for ep in range(cfg.epochs):
        model.train()
        perm = rng.permutation(len(Xt))
        tot = 0.0
        for i in range(0, len(perm), cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=cfg.bf16):
                pred = model(Xt[b])
            loss = duarte_loss(pred.float(), Yt[b], cfg.alpha)
            opt.zero_grad()
            loss.backward()
            if cfg.clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            tot += loss.item() * len(b)
        val = evaluate_loss(model, Xvt, Yvt, cfg)
        hist["train"].append(tot / len(Xt))
        hist["val"].append(val)
        hist["seconds"].append(time.time() - t0)
        if val < best:
            best = val
            save_forecaster(os.path.join(out_dir, "forecaster.pt"), model, cfg, codec, {"epoch": ep, "val": val})
        log(f"epoch {ep:3d}  train {hist['train'][-1]:.4f}  val {val:.4f}  ({hist['seconds'][-1]/60:.1f} min)")
        if (time.time() - t0) / 60 > cfg.max_minutes:
            log("time budget reached")
            break
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(hist, f)
    return hist


@torch.no_grad()
def evaluate_loss(model, X, Y, cfg, batch=32) -> float:
    model.eval()
    tot = 0.0
    for i in range(0, len(X), batch):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=cfg.bf16):
            pred = model(X[i:i + batch])
        tot += float(duarte_loss(pred.float(), Y[i:i + batch], cfg.alpha)) * len(X[i:i + batch])
    return tot / len(X)


def save_forecaster(path, model, cfg: TrainConfig, codec: DensityCodec, meta: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "train": asdict(cfg), "codec": asdict(codec), "meta": meta}, path)


def load_forecaster(path) -> Tuple[DuarteUNet, DensityCodec, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    tc = ck["train"]
    model = DuarteUNet(tc["filters"], tc["frames"], tc.get("residual", False))
    model.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
    model.eval()
    return model, DensityCodec(**ck["codec"]), ck


# ---------------------------------------------------------------------- forecasting and scores
@torch.no_grad()
def rollout(model: nn.Module, x0: np.ndarray, n_blocks: int, *, clip: bool = False,
            mass_projection: Optional["MassProjection"] = None) -> np.ndarray:
    """Iterative forecast: feed each predicted block back in. x0 (k, H, W) -> (n_blocks * k, H, W).

    ``mass_projection``: after each block, rescale every frame so that the window mass changes no faster than
    the training simulations allow (see :class:`MassProjection`)."""
    model.eval()
    x = torch.from_numpy(np.asarray(x0, np.float32))[None]
    out = []
    m_last = None if mass_projection is None else mass_projection.mass(x0[-1:])[0]
    for _ in range(n_blocks):
        x = model(x)
        if clip:
            x = x.clamp(0, 1)
        if mass_projection is not None:
            xb, m_last = mass_projection(x[0].numpy(), m_last)
            x = torch.from_numpy(xb)[None]
        out.append(x[0].numpy())
    return np.concatenate(out, 0)


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / math.sqrt(float((a * a).sum() * (b * b).sum()) + 1e-30))


def lead_time_scores(pred: np.ndarray, truth: np.ndarray, last_input: np.ndarray, mean_state: np.ndarray,
                     weights: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Per lead (frame) on normalised log-density: MAE of the forecast, of persistence (the last input frame)
    and of the time-mean flow; the anomaly correlation with the truth, anomalies taken from the time-mean flow
    (the ACC of weather forecasting); and the **tendency correlation**, between the forecast change and the true
    change since the last input frame -- the stricter test when the flow drifts away from its time mean, where
    the ACC is high for any forecast that follows the drift."""
    w = np.ones_like(truth[0]) if weights is None else weights / weights.mean()
    T = len(truth)
    mae = np.array([float((np.abs(pred[t] - truth[t]) * w).mean()) for t in range(T)])
    pers = np.array([float((np.abs(last_input - truth[t]) * w).mean()) for t in range(T)])
    clim = np.array([float((np.abs(mean_state - truth[t]) * w).mean()) for t in range(T)])
    acc = np.array([_corr((pred[t] - mean_state) * np.sqrt(w), (truth[t] - mean_state) * np.sqrt(w)) for t in range(T)])
    tend = np.array([_corr((pred[t] - last_input) * np.sqrt(w), (truth[t] - last_input) * np.sqrt(w)) for t in range(T)])
    return {"mae": mae, "persistence": pers, "climatology": clim, "acc": acc, "tendency": tend}


def trust_horizon(scores: Dict[str, np.ndarray], frame_dt: float, acc_min: float = 0.6, block: int = 1) -> Dict[str, float]:
    """Lead time up to which the forecast stays useful: before its anomaly correlation first drops below
    ``acc_min`` (the weather convention), and before its error first exceeds persistence / the time mean.

    ``block``: judge whole predicted blocks (the forecaster emits ``block`` frames at a time) by their mean."""
    if block > 1:
        nb = len(scores["mae"]) // block
        scores = {k: np.asarray(v[:nb * block]).reshape(nb, block).mean(1) for k, v in scores.items()}
        frame_dt = frame_dt * block
    lead = frame_dt * (1 + np.arange(len(scores["mae"])))

    def first(mask):
        k = np.nonzero(mask)[0]
        return float(lead[k[0] - 1]) if len(k) and k[0] > 0 else (0.0 if len(k) else float(lead[-1]))

    out = {"acc": first(scores["acc"] < acc_min),
           "beats_persistence": first(scores["mae"] > scores["persistence"]),
           "beats_climatology": first(scores["mae"] > scores["climatology"])}
    if "tendency" in scores:
        out["tendency"] = first(scores["tendency"] < acc_min)
    return out


@dataclass
class MassEnvelope:
    """Bound on how fast the mass in the forecast window can change, learned from the training simulations.

    ``rate_max`` is the largest |d ln M / dt| between consecutive training frames (times a margin). A forecast
    whose window mass moves faster is not obeying mass conservation; the first lead where that happens is a
    trust limit found without any ground truth.
    """
    rate_max: float
    margin: float = 1.5

    @classmethod
    def fit(cls, masses: Sequence[np.ndarray], frame_dt: float, margin: float = 1.5) -> "MassEnvelope":
        rates = [np.abs(np.diff(np.log(m))) / frame_dt for m in masses]
        return cls(float(max(r.max() for r in rates)), margin)

    def violations(self, masses: np.ndarray, frame_dt: float, m0: Optional[float] = None) -> np.ndarray:
        m = np.asarray(masses, float) if m0 is None else np.concatenate([[m0], masses])
        rate = np.abs(np.diff(np.log(m))) / frame_dt
        out = rate > self.margin * self.rate_max
        return out if m0 is not None else np.concatenate([[False], out])

    def first_violation(self, masses, frame_dt, m0=None) -> Optional[int]:
        v = np.nonzero(self.violations(masses, frame_dt, m0))[0]
        return int(v[0]) if len(v) else None


@dataclass
class MassProjection:
    """Project forecast frames onto the mass budget seen in training: the window mass may change between frames
    by at most ``envelope.rate_max`` (relative, per unit time), so a frame that gains or loses more is rescaled
    (uniformly in density, i.e. a shift in normalised log density) to the nearest allowed mass. The rate is
    bounded, not prescribed, so real accretion and outflow still pass."""
    codec: DensityCodec
    cell_volume: np.ndarray
    envelope: MassEnvelope
    frame_dt: float

    def mass(self, x: np.ndarray) -> np.ndarray:
        return window_mass(self.codec.decode(x), self.cell_volume)

    def __call__(self, frames: np.ndarray, m_prev: float):
        out = np.array(frames, np.float32, copy=True)
        span = self.codec.log_max - self.codec.log_min
        lim = self.envelope.rate_max * self.frame_dt
        for i in range(len(out)):
            m = float(self.mass(out[i:i + 1])[0])
            lo, hi = m_prev * math.exp(-lim), m_prev * math.exp(lim)
            target = min(max(m, lo), hi)
            if target != m:
                out[i] += np.float32(math.log10(target / m) / span)
            m_prev = target
        return out, m_prev


def window_mass(rho: np.ndarray, cell_volume: np.ndarray) -> np.ndarray:
    """Mass in the (cropped) window for density frames (T, H, W) and cell volumes (H, W)."""
    return (np.asarray(rho, float) * cell_volume[None]).sum((1, 2))
