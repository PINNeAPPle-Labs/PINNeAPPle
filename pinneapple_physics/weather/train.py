"""Training of the global forecast network: one 6-hour step first, then rollouts of growing length.

Loss: mean squared error of the normalised state, weighted by the area of each grid cell (cosine of latitude) and
by channel (geopotential and temperature at 500/850 hPa and the surface fields count more: they are what forecasts
are judged on). Mixed precision bf16 on CPUs with AMX/AVX-512 bf16, fp32 elsewhere. Checkpoints keep the best model
on the validation years (2-day rollout error).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .data import Era5Store
from .model import SphereUNet, static_fields, toa_insolation

PRIORITY = {"z500": 3.0, "t850": 3.0, "t2m": 2.0, "msl": 2.0, "u10": 1.5, "v10": 1.5, "tp6": 1.0}


class Forecaster:
    """A trained network with everything needed to roll it out from an ERA5 time."""

    def __init__(self, store: Era5Store, model: SphereUNet):
        self.store, self.model = store, model
        self.static = torch.from_numpy(static_fields(store.lat, store.lon, store.constants))

    def extra(self, times) -> torch.Tensor:
        t = np.asarray(times, dtype="datetime64[ns]")
        sun = toa_insolation(np.concatenate([t, t + np.timedelta64(6, "h")]), self.store.lat, self.store.lon)
        sun = torch.from_numpy(sun.reshape(2, len(t), *sun.shape[1:]).transpose(1, 0, 2, 3).copy())
        return torch.cat([sun, self.static.expand(len(t), -1, -1, -1)], 1)

    @torch.no_grad()
    def rollout(self, x_t: np.ndarray, x_prev: np.ndarray, t0, steps: int, perturb: np.ndarray | None = None
                ) -> np.ndarray:
        """Normalised forecast (members, steps + 1, channel, lat, lon) from states at t0 and t0 - 6 h. ``perturb``
        (members, channel, lat, lon) is added to the initial state of each member (ensemble)."""
        self.model.eval()
        n = 1 if perturb is None else len(perturb)
        a = torch.from_numpy(np.repeat(np.asarray(x_t, np.float32)[None], n, 0))
        b = torch.from_numpy(np.repeat(np.asarray(x_prev, np.float32)[None], n, 0))
        if perturb is not None:
            a = a + torch.from_numpy(perturb.astype(np.float32))
            b = b + torch.from_numpy(perturb.astype(np.float32))
        out = [a.numpy()]
        t = np.datetime64(t0, "ns")
        for _ in range(steps):
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=_bf16()):
                y = self.model(a, b, self.extra([t] * n)).float()
            b, a = a, y
            t = t + np.timedelta64(6, "h")
            out.append(a.numpy())
        return np.stack(out, 1)

    def save(self, path: str | Path, config: dict):
        torch.save({"state_dict": self.model.state_dict(), "config": config}, path)

    @classmethod
    def load(cls, path: str | Path, store: Era5Store) -> Forecaster:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        m = SphereUNet(len(store.channels), 7, tuple(cfg["widths"]), cfg["blocks"], cfg["patch"])
        m.load_state_dict(ck["state_dict"])
        return cls(store, m)


def _bf16() -> bool:
    try:
        return torch.backends.mkldnn.is_available() and "bf16" in open("/proc/cpuinfo").read()
    except OSError:
        return False


def _weights(store: Era5Store) -> torch.Tensor:
    cw = torch.tensor([PRIORITY.get(c, 1.0) for c in store.channels])
    cw = cw / cw.mean()
    lw = torch.from_numpy(store.lat_weights().astype(np.float32))
    return cw[None, :, None, None] * lw[None, None, :, None]


def train(store_path: str | Path, out: str | Path, *, train_years=(1990, 2017), val_years=(2018, 2019),
          widths=(192, 256, 384), blocks=2, patch=2, batch=16, hours=6.0,
          schedule=((1, 0.45), (2, 0.2), (4, 0.2), (8, 0.15)), lr=8e-4, seed=0, log=print,
          init_from: str | Path | None = None, resume: bool = True) -> Path:
    """Train for ``hours`` of wall time split by ``schedule`` ((rollout steps, fraction of the time), ...).

    ``out/last.pt`` (weights, optimiser, phase and time spent in it) is written every 200 steps; with ``resume`` a
    run that was interrupted continues from it. ``init_from``: start from the weights of another checkpoint."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    store = Era5Store(store_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"widths": list(widths), "blocks": blocks, "patch": patch, "train_years": list(train_years),
           "val_years": list(val_years), "schedule": [list(s) for s in schedule], "hours": hours}
    model = SphereUNet(len(store.channels), 7, widths, blocks, patch)
    fc = Forecaster(store, model)
    W = _weights(store)
    tr = store.years(*train_years)
    va = store.years(*val_years)
    va = va[(va >= 1) & (va + 8 < len(store.times))][::37][:48]          # fixed validation starts
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total = hours * 3600.0
    t_start, best, history, step = time.time(), math.inf, [], 0
    use_bf16 = _bf16()
    start_phase, spent_in_phase = 0, 0.0
    if init_from is not None:
        model.load_state_dict(torch.load(init_from, map_location="cpu", weights_only=False)["state_dict"])
        log(f"weights from {init_from}")
    if resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        start_phase, spent_in_phase = ck["phase"], ck["spent_in_phase"]
        best, history, step = ck["best"], ck["history"], ck["step"]
        rng = np.random.default_rng(seed + step)
        log(f"resumed at step {step}, phase {start_phase}, {spent_in_phase / 60:.0f} min into it")

    def batch_at(idx, k):
        x = np.stack([store.state[i - 1:i + k + 1] for i in idx]).astype(np.float32)   # (B, k+2, C, H, W)
        return torch.from_numpy(x), store.times[idx]

    def val_error(k=8):
        model.eval()
        x, t0 = batch_at(va, k)
        err = 0.0
        with torch.no_grad():
            a, b = x[:, 1], x[:, 0]
            for s in range(k):
                with torch.autocast("cpu", dtype=torch.bfloat16, enabled=use_bf16):
                    y = model(a, b, fc.extra(t0 + np.timedelta64(6 * s, "h"))).float()
                b, a = a, y
            err = float((((a - x[:, k + 1]) ** 2) * W).mean())
        model.train()
        return err

    elapsed_before = 0.0
    for ph, (k, frac) in enumerate(schedule):
        if ph < start_phase:
            continue
        budget = frac * total - (spent_in_phase if ph == start_phase else 0.0)
        offset = (spent_in_phase if ph == start_phase else 0.0)
        t_phase = time.time()
        phase_lr = lr if k == 1 else lr * 0.3
        sched_steps = None
        n_in_phase = 0
        while time.time() - t_phase < budget:
            idx = rng.choice(tr[(tr >= 1) & (tr + k < len(store.times))], batch, replace=False)
            x, t0 = batch_at(idx, k)
            frac_done = (offset + time.time() - t_phase) / (frac * total)
            for g in opt.param_groups:                                # cosine decay inside each phase
                g["lr"] = phase_lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(frac_done, 1.0))))
            a, b = x[:, 1], x[:, 0]
            loss = 0.0
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=use_bf16):
                for s in range(k):
                    y = model(a, b, fc.extra(t0 + np.timedelta64(6 * s, "h"))).float()
                    loss = loss + (((y - x[:, s + 2]) ** 2) * W).mean() / k
                    b, a = a, y
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            n_in_phase += 1
            if step % 200 == 0:
                v = val_error()
                history.append({"step": step, "rollout": k, "loss": float(loss.detach()), "val_2day": v,
                                "minutes": (time.time() - t_start) / 60})
                log(f"step {step} rollout {k} loss {float(loss):.4f} val(2 days) {v:.4f} "
                    f"{(time.time() - t_start) / 60:.0f} min")
                if v < best:
                    best = v
                    fc.save(out / "model.pt", cfg)
                (out / "history.json").write_text(json.dumps(history))
                torch.save({"state_dict": model.state_dict(), "optimizer": opt.state_dict(), "phase": ph,
                            "spent_in_phase": offset + time.time() - t_phase, "best": best, "history": history,
                            "step": step, "config": cfg}, out / "last.tmp")
                (out / "last.tmp").replace(out / "last.pt")
        elapsed_before += time.time() - t_phase
        _ = sched_steps, n_in_phase, elapsed_before
    v = val_error()
    if v < best:
        fc.save(out / "model.pt", cfg)
    (out / "history.json").write_text(json.dumps(history))
    return out / "model.pt"
