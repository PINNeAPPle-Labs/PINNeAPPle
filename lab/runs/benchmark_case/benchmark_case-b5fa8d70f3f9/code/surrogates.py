"""Surrogates for the heated-channel digital twin (layer 2) and their shared data handling.

All models map (state at t_k, actuator samples over [t_k, t_k+0.4]) -> state at t_{k+1} and
are rolled out autoregressively. State = (u, v, p, T) on the 128x32 CFD grid, normalised with
training-set statistics only.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[2] / "PINNeAPPle"))
from pinneapple_neural.architectures.neural_operators.fno import FNO2d  # noqa: E402
from pinneapple_neural.architectures.neural_operators.deeponet import DeepONet  # noqa: E402

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SMOKE = os.environ.get("SMOKE") == "1"
RES = ROOT / ("results_smoke" if SMOKE else "results")
RES.mkdir(exist_ok=True)
FIELDS = ("u", "v", "p", "T")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(RES / "log.txt", "a") as fh:
        fh.write(s + "\n")


class ChannelData:
    def __init__(self):
        z = np.load(ROOT / "data" / "trajectories.npz")
        self.meta = json.loads((ROOT / "data" / "meta.json").read_text())
        self.F = z["fields"]                                   # (R, K+1, 4, nx, ny)
        self.split = z["split"]
        Uf, Qf = z["U_fine"], z["Q_fine"]                      # (R, 401) at 0.1
        K = self.F.shape[1] - 1
        idx = np.arange(K)[:, None] * 4 + np.arange(5)[None]   # 5 samples per 0.4 interval
        self.act = np.concatenate([Uf[:, idx], Qf[:, idx]], -1).astype(np.float32)  # (R, K, 10)
        self.qoi = {k: z[f"qoi_{k}"] for k in ("T_out", "T_max", "dp", "heat_in", "heat_out", "energy")}
        tr = self.split == "train"
        self.mu = self.F[tr].mean((0, 1, 3, 4)).reshape(1, 4, 1, 1).astype(np.float32)
        self.sd = self.F[tr].std((0, 1, 3, 4)).reshape(1, 4, 1, 1).astype(np.float32)
        self.a_mu = self.act[tr].mean((0, 1)).astype(np.float32)
        self.a_sd = (self.act[tr].std((0, 1)) + 1e-6).astype(np.float32)
        self.Fn = ((self.F - self.mu) / self.sd).astype(np.float32)
        self.actn = ((self.act - self.a_mu) / self.a_sd).astype(np.float32)
        self.nx, self.ny = self.F.shape[-2:]
        self.dx, self.dy = self.meta["L"] / self.nx, 1.0 / self.ny
        self.Pe = self.meta["Re"] * self.meta["Pr"]
        xc = np.array(self.meta["xc"])
        self.heater = torch.tensor((xc >= 2.0) & (xc <= 4.0), dtype=torch.float32)
        self.dt = self.meta["dt_record"]

    def runs(self, name):
        return np.where(self.split == name)[0]

    def pairs(self, runs):
        X = self.Fn[runs, :-1].reshape(-1, 4, self.nx, self.ny)
        A = self.actn[runs].reshape(-1, 10)
        Y = self.Fn[runs, 1:].reshape(-1, 4, self.nx, self.ny)
        return X, A, Y

    def denorm(self, x):
        mu = torch.as_tensor(self.mu, device=x.device) if torch.is_tensor(x) else self.mu
        sd = torch.as_tensor(self.sd, device=x.device) if torch.is_tensor(x) else self.sd
        return x * sd + mu


# ---------------------------------------------------------------------------
# physics residuals (physical units) -- used as metrics for every model and as a loss for PI-FNO
# ---------------------------------------------------------------------------

def divergence(uv, dx, dy):
    """Central-difference divergence of cell-centred (u, v), interior cells. uv: (B, 2, nx, ny)."""
    du = (uv[:, 0, 2:, 1:-1] - uv[:, 0, :-2, 1:-1]) / (2 * dx)
    dv = (uv[:, 1, 1:-1, 2:] - uv[:, 1, 1:-1, :-2]) / (2 * dy)
    return du + dv


def energy_residual(s0, s1, act_phys, D: ChannelData):
    """Global energy balance over one record interval, normalised by the heat input scale.
    d/dt int T = (1/Pe) Q L_h - int u T dy|outlet  (inlet conduction neglected)."""
    E0 = s0[:, 3].sum((1, 2)) * D.dx * D.dy
    E1 = s1[:, 3].sum((1, 2)) * D.dx * D.dy
    q_int = act_phys[:, 5:10].mean(1)                       # mean heater power over the interval
    Lh = float(D.heater.sum()) * D.dx
    heat_in = q_int * Lh / D.Pe
    out0 = (s0[:, 0, -1] * s0[:, 3, -1]).sum(1) * D.dy
    out1 = (s1[:, 0, -1] * s1[:, 3, -1]).sum(1) * D.dy
    r = (E1 - E0) / D.dt - heat_in + 0.5 * (out0 + out1)
    return r / (2.0 * Lh / D.Pe)                            # relative to the maximum heat input


def qois(s, D: ChannelData):
    """Same definitions applied to CFD truth and to predictions (cell-centred approximations)."""
    u, T, p = s[:, 0, -1], s[:, 3, -1], s[:, 2]
    T_out = (u * T).sum(-1) / np.maximum(u.sum(-1), 1e-9)
    return {"T_out": T_out, "T_max": s[:, 3].reshape(len(s), -1).max(1),
            "dp": p[:, 0].mean(-1) - p[:, -1].mean(-1)}


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

class PaddedFNO(nn.Module):
    """PINNeAPPle FNO2d with zero padding (the channel is not periodic) and actuator channels."""

    def __init__(self, width=48, modes=(24, 12), layers=4, pad=(32, 16)):  # 160x48 grid: much faster FFTs on MPS than 144x40
        super().__init__()
        self.pad = pad
        self.fno = FNO2d(4 + 10, 4, width=width, modes1=modes[0], modes2=modes[1], layers=layers, use_grid=True)

    def forward(self, x, a):
        B, _, H, W = x.shape
        ach = a[:, :, None, None].expand(B, a.shape[1], H, W)
        z = F.pad(torch.cat([x, ach], 1), (0, self.pad[1], 0, self.pad[0]))
        dz = self.fno(z).y[:, :, :H, :W]
        return x + dz


class ConvAE(nn.Module):
    """Convolutional autoencoder (128x32 -> latent) + MLP latent stepper with actuator input."""

    def __init__(self, latent=64):
        super().__init__()
        ch = (32, 64, 128)
        self.enc = nn.Sequential(nn.Conv2d(4, ch[0], 3, 2, 1), nn.GELU(), nn.Conv2d(ch[0], ch[1], 3, 2, 1), nn.GELU(),
                                 nn.Conv2d(ch[1], ch[2], 3, 2, 1), nn.GELU(), nn.Flatten(), nn.Linear(ch[2] * 16 * 4, latent))
        self.dec_in = nn.Linear(latent, ch[2] * 16 * 4)
        self.dec = nn.Sequential(nn.ConvTranspose2d(ch[2], ch[1], 4, 2, 1), nn.GELU(), nn.ConvTranspose2d(ch[1], ch[0], 4, 2, 1),
                                 nn.GELU(), nn.ConvTranspose2d(ch[0], 4, 4, 2, 1))
        self.step_net = nn.Sequential(nn.Linear(latent + 10, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, latent))

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(self.dec_in(z).view(-1, 128, 16, 4))

    def step(self, z, a):
        return z + self.step_net(torch.cat([z, a], 1))

    def forward(self, x, a):
        return self.decode(self.step(self.encode(x), a))


class POD:
    def __init__(self, D: ChannelData, runs, energy=0.9999, cap=40):
        X = D.Fn[runs].reshape(-1, 4, D.nx * D.ny)
        self.modes, self.r = [], []
        for f in range(4):
            U, S, Vt = np.linalg.svd(X[::3, f], full_matrices=False)  # every 3rd snapshot (memory)
            r = int(min(cap, np.searchsorted(np.cumsum(S ** 2) / (S ** 2).sum(), energy) + 1))
            self.modes.append(Vt[:r].astype(np.float32))
            self.r.append(r)
        self.nx, self.ny = D.nx, D.ny

    def encode(self, x):  # x (B,4,nx,ny) normalised
        x = x.reshape(len(x), 4, -1)
        return np.concatenate([x[:, f] @ self.modes[f].T for f in range(4)], 1)

    def decode(self, a):
        out, i = [], 0
        for f in range(4):
            out.append(a[:, i:i + self.r[f]] @ self.modes[f])
            i += self.r[f]
        return np.stack(out, 1).reshape(len(a), 4, self.nx, self.ny)

    def torch_decode(self, a):
        out, i = [], 0
        for f in range(4):
            M = torch.as_tensor(self.modes[f], device=a.device)
            out.append(a[:, i:i + self.r[f]] @ M)
            i += self.r[f]
        return torch.stack(out, 1).reshape(len(a), 4, self.nx, self.ny)


class LatentMLP(nn.Module):
    def __init__(self, r, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(r + 10, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, r))

    def forward(self, a, u):
        return a + self.net(torch.cat([a, u], 1))


class DeepONetStepper(nn.Module):
    """PINNeAPPle DeepONet: branch = (POD coefficients of the current state, actuator samples),
    trunk = (x, y); outputs the 4 fields at the next record time."""

    def __init__(self, r, coords, hidden=256, modes=128):
        super().__init__()
        self.don = DeepONet(branch_dim=r + 10, trunk_dim=2, out_dim=4, hidden=hidden, modes=modes)
        self.register_buffer("coords", coords)

    def forward(self, a, u, idx=None):
        c = self.coords if idx is None else self.coords[idx]
        return self.don(torch.cat([a, u], 1), c).y          # (B, N, 4)


# ---------------------------------------------------------------------------
# training helpers
# ---------------------------------------------------------------------------

def seq_batches(D, runs, horizon, bs, rng, stride=1):
    """Random (run, start) windows of length horizon+1 for rollout training (every `stride`-th start, random offset)."""
    K = D.Fn.shape[1] - 1
    off = int(rng.integers(stride))
    starts = [(r, k) for r in runs for k in range(off, K - horizon + 1, stride)]
    order = rng.permutation(len(starts))
    for i in range(0, len(order), bs):
        sel = [starts[j] for j in order[i:i + bs]]
        X = np.stack([D.Fn[r, k:k + horizon + 1] for r, k in sel])     # (B, h+1, 4, nx, ny)
        A = np.stack([D.actn[r, k:k + horizon] for r, k in sel])       # (B, h, 10)
        Ap = np.stack([D.act[r, k:k + horizon] for r, k in sel])
        yield (torch.tensor(X, device=DEV), torch.tensor(A, device=DEV), torch.tensor(Ap, device=DEV))


def train_field_model(D, model, name, *, epochs, horizon=4, lr=1e-3, bs=16, phys_lambda=0.0, patience=5, seed=0, stride=3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr, va = D.runs("train"), D.runs("val")
    if SMOKE:
        tr, va, epochs = tr[:3], va[:2], 1
    best, state, bad, hist, t0 = 1e9, None, 0, [], time.time()
    mu, sd = torch.tensor(D.mu, device=DEV), torch.tensor(D.sd, device=DEV)
    for ep in range(epochs):
        model.train()
        tl, n = 0.0, 0
        for X, A, Ap in seq_batches(D, tr, horizon, bs, rng, stride):
            x = X[:, 0]
            loss = 0.0
            for k in range(horizon):
                x_new = model(x, A[:, k])
                loss = loss + F.mse_loss(x_new, X[:, k + 1])
                if phys_lambda > 0:
                    s0, s1 = x * sd + mu, x_new * sd + mu
                    div = divergence(s1[:, :2], D.dx, D.dy)
                    er = energy_residual(s0, s1, Ap[:, k], D)
                    loss = loss + phys_lambda * ((div ** 2).mean() + (er ** 2).mean())
                x = x_new
            loss = loss / horizon
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += float(loss.detach()) * len(X)
            n += len(X)
        sch.step()
        v = rollout_error(D, lambda x0, acts: rollout_field(model, x0, acts), va)
        hist.append({"epoch": ep + 1, "train": tl / n, "val_rollout_rel_l2": v})
        log(f"  [{name}] ep {ep + 1}: train {tl / n:.4e}  val rollout relL2 {v:.4f} ({time.time() - t0:.0f}s)")
        if np.isfinite(v) and (v < best or state is None):
            best, bad, state = v, 0, {k: x.detach().clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if state is not None:  # a model whose validation rollout never stayed finite keeps its last weights (reported as such)
        model.load_state_dict(state)
    model.eval()
    return model, {"history": hist, "train_s": time.time() - t0, "best_val": best,
                   "n_params": sum(p.numel() for p in model.parameters())}


@torch.no_grad()
def rollout_field(model, x0, acts):
    """x0 (4,nx,ny) normalised numpy; acts (K,10) normalised -> (K+1,4,nx,ny) numpy."""
    x = torch.tensor(x0[None], device=DEV)
    out = [x0]
    for k in range(len(acts)):
        x = model(x, torch.tensor(acts[k:k + 1], device=DEV))
        out.append(x[0].float().cpu().numpy())
    return np.stack(out)


def rollout_error(D, roll, runs):
    errs = []
    for r in runs:
        P = roll(D.Fn[r, 0], D.actn[r])
        errs.append(np.linalg.norm(P[1:] - D.Fn[r, 1:]) / np.linalg.norm(D.Fn[r, 1:]))
    return float(np.mean(errs))
