"""Reproduction (and improvement) of Ligthart, "Pedestrian Dead Reckoning using data-driven and
physics-informed machine learning", MSc thesis, TU Delft 2025 -- with PINNeAPPle, on the public RIDI dataset.

The thesis used a proprietary Leica dataset (25 h) and Leica's EKF. Here:
  * data     : RIDI (Yan et al. 2018), 94 smartphone sequences, 2.7 h, 11 people, Tango VIO poses as
               ground truth; person-disjoint split 60/20/20 as in the thesis;
  * networks : PINNeAPPle Conv1DModel (dilated residual 1-D CNN) backbone with a velocity + log-std head;
               "adapted TLIO" (IMU in the local/world frame) and "DIVE" (gravity-aligned, yaw-free frame);
  * training : MSE for 10 epochs then maximum likelihood, lr as in the thesis scale, random accelerometer /
               gyroscope biases, gravity-direction perturbation and random yaw, re-drawn every epoch;
  * EKF      : PINNeAPPle ExtendedKalmanFilter, state (p, v), IMU mechanisation with the phone's own
               orientation (game rotation vector aligned in heading at the segment start), network velocity
               as the measurement (the thesis' EKF additionally estimates attitude and biases);
  * PINN     : grid-output network over a 4 s window (velocity at 50 Hz), physics loss from the
               accelerometer measurement model with the quadratic-fit (central difference) derivative,
               lambda in {0, 0.01, 0.1, 1};
  * improve  : yaw-equivariant test-time augmentation, validation-fitted bias correction in the heading
               frame, and variance recalibration -- aimed at the bias the thesis identified as the bottleneck.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[2] / "PINNeAPPle"))
from pinneapple_neural.architectures.convolutions.conv1d import Conv1DModel  # noqa: E402
from pinneapple_analysis.state_estimation.kalman import ExtendedKalmanFilter  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
RES = ROOT / ("results_smoke" if SMOKE else "results")
RES.mkdir(exist_ok=True)
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
G = 9.81
HZ = 200
SPLIT = {"train": ["hang", "dan", "huayi", "ma", "hao"], "val": ["ruixuan", "zhicheng"],
         "test": ["shali", "tang", "xiaojing", "yajie"]}
OUT = {}


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(RES / "log.txt", "a") as fh:
        fh.write(s + "\n")


def save():
    (RES / "results.json").write_text(json.dumps(OUT, indent=1, default=float))


# ---------------------------------------------------------------------------- data
class Seq:
    def __init__(self, path):
        d = pd.read_csv(path)
        t = d.time.values * 1e-9
        tu = np.arange(t[0], t[-1], 1.0 / HZ)                      # uniform 200 Hz resampling
        it = lambda c: np.stack([np.interp(tu, t, d[k].values) for k in c], 1)
        self.name = Path(path).parts[-3]
        self.t = tu - tu[0]
        self.acc = it(["acce_x", "acce_y", "acce_z"])
        self.gyr = it(["gyro_x", "gyro_y", "gyro_z"])
        self.pos = it(["pos_x", "pos_y", "pos_z"])
        q = Rot.from_quat(d[["ori_x", "ori_y", "ori_z", "ori_w"]].values).as_quat()
        qr = Rot.from_quat(d[["rv_x", "rv_y", "rv_z", "rv_w"]].values).as_quat()
        self.R = Rot.from_quat(_interp_quat(t, q, tu))             # ground-truth body->world (Tango)
        self.Rrv = Rot.from_quat(_interp_quat(t, qr, tu))          # phone AHRS (game rotation vector)
        # ground-truth velocity: central difference of Tango positions smoothed over 0.1 s
        from scipy.ndimage import uniform_filter1d
        self.vel = uniform_filter1d(np.gradient(self.pos, 1.0 / HZ, axis=0), 20, axis=0)
        self.Rm = self.R.as_matrix().astype(np.float32)            # (N,3,3) body->world, for vectorised windows
        self.acc_w = self.R.apply(self.acc)                        # world-frame specific force (GT attitude)
        self.gyr_w = self.R.apply(self.gyr)


def _interp_quat(t, q, tu):
    from scipy.spatial.transform import Slerp
    return Slerp(t, Rot.from_quat(q))(np.clip(tu, t[0], t[-1])).as_quat()


def load(split):
    base = ROOT / "data" / "data_publish_v2"
    paths = sorted(p for u in SPLIT[split] for p in glob.glob(str(base / f"{u}_*" / "processed" / "data.csv")))
    if SMOKE:
        paths = paths[:3]
    return [Seq(p) for p in paths]


# ---------------------------------------------------------------------------- features
def rotvec(Rm):
    """Vectorised rotation matrix -> rotation vector (axis * angle), (N,3,3) -> (N,3)."""
    tr = np.clip((np.trace(Rm, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)
    ang = np.arccos(tr)
    w = np.stack([Rm[:, 2, 1] - Rm[:, 1, 2], Rm[:, 0, 2] - Rm[:, 2, 0], Rm[:, 1, 0] - Rm[:, 0, 1]], 1)
    s = np.sin(ang)
    k = np.where(s > 1e-6, ang / (2 * np.maximum(s, 1e-12)), 0.5)   # small-angle limit: w/2
    out = w * k[:, None]
    near_pi = ang > np.pi - 1e-3                                    # formula degenerates near 180 deg
    if near_pi.any():
        out[near_pi] = Rot.from_matrix(Rm[near_pi]).as_rotvec()
    return out


def yaw_rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def feats_tlio(acc_w, gyr_w):
    return np.concatenate([acc_w, gyr_w], 1)          # (L, 6), world ("local") frame


def feats_dive(R, acc, idx_end):
    """Gravity-aligned, yaw-free frame of the window end; specific force minus gravity + rotation vector."""
    yaw = R[idx_end].as_euler("zyx")[0]
    Ry = Rot.from_euler("z", yaw)
    Rg = Ry.inv() * R                                  # rotations relative to the heading at window end
    f = Rg.apply(acc) - np.array([0, 0, G])
    return np.concatenate([f, Rg.as_rotvec()], 1), yaw


class WindowSet(torch.utils.data.Dataset):
    """1-s windows (200 samples) with velocity at the window end, re-augmented each epoch."""

    def __init__(self, seqs, kind, stride=10, augment=False, L=HZ):
        self.seqs, self.kind, self.aug, self.L = seqs, kind, augment, L
        self.idx = [(i, e) for i, s in enumerate(seqs) for e in range(L, len(s.t), stride)]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i, e = self.idx[k]
        s = self.seqs[i]
        sl = slice(e - self.L, e)
        Rm = s.Rm[sl]
        acc_w, gyr_w = s.acc_w[sl], s.gyr_w[sl]
        if self.aug:
            ba, bg = np.random.uniform(-0.02, 0.02, 3), np.random.uniform(-0.002, 0.002, 3)
            acc_w = acc_w + Rm @ ba                                    # body-frame bias, seen in the world frame
            gyr_w = gyr_w + Rm @ bg
            ax = np.random.normal(size=2); ax = np.r_[ax / np.linalg.norm(ax), 0]
            Rp = Rot.from_rotvec(ax * np.random.uniform(0, 0.04)).as_matrix()   # gravity-direction error
            acc_w, gyr_w, Rm = acc_w @ Rp.T, gyr_w @ Rp.T, Rp @ Rm
        v = s.vel[e - 1]
        if self.kind == "tlio":
            x = np.concatenate([acc_w, gyr_w], 1)
            if self.aug:
                Y = yaw_rot(np.random.uniform(0, 2 * np.pi))
                x = np.concatenate([x[:, :3] @ Y.T, x[:, 3:] @ Y.T], 1)
                v = Y @ v
            return torch.tensor(x.T, dtype=torch.float32), torch.tensor(v, dtype=torch.float32), torch.zeros(1)
        yaw = np.arctan2(Rm[-1, 1, 0], Rm[-1, 0, 0])
        Yt = yaw_rot(-yaw)
        Rg = Yt @ Rm                                                   # relative to the heading at window end
        f = acc_w @ Yt.T - np.array([0, 0, G])                         # = Rg acc - g
        x = np.concatenate([f, rotvec(Rg)], 1)
        return torch.tensor(x.T, dtype=torch.float32), torch.tensor(Yt @ v, dtype=torch.float32), torch.tensor([yaw])


# ---------------------------------------------------------------------------- models
class VelNet(nn.Module):
    """PINNeAPPle Conv1DModel backbone -> pooled -> velocity (3) + log-std (3)."""

    def __init__(self, cin=6, hidden=64, blocks=6):
        super().__init__()
        self.body = Conv1DModel(cin, hidden, hidden_channels=hidden, num_blocks=blocks, kernel_size=5,
                                dilation_schedule="exponential", norm="group")
        self.head = nn.Sequential(nn.Linear(2 * hidden, 128), nn.GELU(), nn.Linear(128, 6))

    def forward(self, x):
        h = self.body(x)
        h = h.y if hasattr(h, "y") else h
        z = torch.cat([h.mean(-1), h[..., -1]], 1)
        o = self.head(z)
        return o[:, :3], o[:, 3:].clamp(-5, 3)


class GridNet(nn.Module):
    """Grid-output PINN variant: 4 s input pooled to 50 Hz -> velocity at every 50 Hz sample."""

    def __init__(self, cin=6, hidden=64, blocks=6):
        super().__init__()
        self.body = Conv1DModel(cin, 3, hidden_channels=hidden, num_blocks=blocks, kernel_size=5,
                                dilation_schedule="exponential", norm="group")

    def forward(self, x):                               # x (B, 6, 800)
        x = F.avg_pool1d(x, 4)
        y = self.body(x)
        return y.y if hasattr(y, "y") else y             # (B, 3, 200)


def train_velnet(kind, tr, va, epochs_mse=10, epochs_ml=30, lr=1e-4, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    dtr = WindowSet(tr, kind, augment=True)
    dva = WindowSet(va, kind, stride=20)
    ltr = torch.utils.data.DataLoader(dtr, batch_size=256, shuffle=True, num_workers=0)
    lva = torch.utils.data.DataLoader(dva, batch_size=512)
    m = VelNet().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    best, state, bad, hist, t0 = 1e9, None, 0, [], time.time()
    total = (1, 1) if SMOKE else (epochs_mse, epochs_ml)
    for ep in range(sum(total)):
        ml = ep >= total[0]
        m.train()
        for x, v, _ in ltr:
            x, v = x.to(DEV), v.to(DEV)
            mu, ls = m(x)
            loss = ((mu - v) ** 2).sum(1).mean() if not ml else (2 * ls + (mu - v) ** 2 * torch.exp(-2 * ls)).sum(1).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()
        vl, n = 0.0, 0
        with torch.no_grad():
            for x, v, _ in lva:
                mu, ls = m(x.to(DEV))
                v = v.to(DEV)
                vl += float(((2 * ls + (mu - v) ** 2 * torch.exp(-2 * ls)).sum(1) if ml else ((mu - v) ** 2).sum(1)).sum()); n += len(v)
        vl /= n
        hist.append({"epoch": ep + 1, "phase": "ML" if ml else "MSE", "val": vl})
        log(f"  [{kind}] ep {ep + 1} ({'ML' if ml else 'MSE'}): val {vl:.4f} ({time.time() - t0:.0f}s)")
        if ep == total[0]:
            best, bad = 1e9, 0          # new objective -> reset early stopping
        if vl < best:
            best, bad, state = vl, 0, {k: x.clone() for k, x in m.state_dict().items()}
        else:
            bad += 1
            if ml and bad >= 5:
                break
    m.load_state_dict(state)
    m.eval()
    return m, {"history": hist, "train_s": time.time() - t0, "n_params": sum(p.numel() for p in m.parameters())}


def train_grid(tr, va, lam, epochs=25, lr=1e-4, seed=0):
    """Grid-output network, data loss on all 50 Hz samples + lam * accelerometer-model physics loss."""
    torch.manual_seed(seed); np.random.seed(seed)
    L = 4 * HZ

    def batches(seqs, stride, aug):
        idx = [(i, e) for i, s in enumerate(seqs) for e in range(L, len(s.t), stride)]
        np.random.shuffle(idx)
        for b in range(0, len(idx), 64):
            X, V, Rb, A = [], [], [], []
            for i, e in idx[b:b + 64]:
                s = seqs[i]
                sl = slice(e - L, e)
                acc, gyr, R = s.acc[sl].copy(), s.gyr[sl].copy(), s.R[sl]
                if aug:
                    acc += np.random.uniform(-0.02, 0.02, 3); gyr += np.random.uniform(-0.002, 0.002, 3)
                X.append(feats_tlio(R.apply(acc), R.apply(gyr)).T)
                V.append(s.vel[sl][3::4].T)                                     # 50 Hz targets
                Rb.append(R.as_matrix()[3::4])                                  # body->world at 50 Hz
                A.append(s.acc[sl].reshape(-1, 4, 3).mean(1))                   # measured specific force, 50 Hz
            yield (torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(V), dtype=torch.float32),
                   torch.tensor(np.array(Rb), dtype=torch.float32), torch.tensor(np.array(A), dtype=torch.float32))

    m = GridNet().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    best, state, bad, t0 = 1e9, None, 0, time.time()
    dt = 4.0 / HZ
    gvec = torch.tensor([0, 0, G], device=DEV)
    ep_total = 1 if SMOKE else epochs
    for ep in range(ep_total):
        use_phys = lam > 0 and ep >= (0 if SMOKE else 10)     # thesis: 10 data-only epochs first
        m.train()
        for X, V, Rb, A in batches(tr, 20, True):
            X, V, Rb, A = X.to(DEV), V.to(DEV), Rb.to(DEV), A.to(DEV)
            vp = m(X)
            loss = ((vp - V) ** 2).mean()
            if use_phys:
                a = (vp[:, :, 2:] - vp[:, :, :-2]) / (2 * dt)              # quadratic-fit derivative (interior)
                a = a.transpose(1, 2)                                     # (B, T-2, 3) world
                f_pred = torch.einsum("btji,btj->bti", Rb[:, 1:-1], a + gvec)   # R^T (a + g)
                loss = loss + lam * ((f_pred - A[:, 1:-1]) ** 2).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()
        with torch.no_grad():
            err = [float(((m(X.to(DEV))[:, :, -1] - V.to(DEV)[:, :, -1]) ** 2).sum(1).mean()) for X, V, _, _ in batches(va, 80, False)]
        v = float(np.mean(err))
        log(f"  [grid lam={lam}] ep {ep + 1}: val end-velocity MSE {v:.4f} ({time.time() - t0:.0f}s)")
        if v < best:
            best, bad, state = v, 0, {k: x.clone() for k, x in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 5 and ep >= 12:
                break
    m.load_state_dict(state)
    m.eval()
    return m, {"best_val": best, "train_s": time.time() - t0}


# ---------------------------------------------------------------------------- prediction helpers
class Predictor:
    """Wraps a trained model: velocity (world) + std at window end, optional TTA / bias correction."""

    def __init__(self, kind, model, tta=0, corr=None, var_scale=1.0, grid=False):
        self.kind, self.m, self.tta, self.corr, self.vs, self.grid = kind, model, tta, corr, var_scale, grid
        self.const_std = None

    @torch.no_grad()
    def __call__(self, acc, gyr, R):
        if self.grid:
            x = feats_tlio(R.apply(acc), R.apply(gyr)).T[None]
            v = self.m(torch.tensor(x, dtype=torch.float32, device=DEV))[0, :, -1].cpu().numpy()
            return v, self.const_std
        if self.kind == "tlio":
            x = feats_tlio(R.apply(acc), R.apply(gyr))
            if self.tta:
                vs, ss = [], []
                for k in range(self.tta):
                    Y = yaw_rot(2 * np.pi * k / self.tta)
                    xx = np.concatenate([x[:, :3] @ Y.T, x[:, 3:] @ Y.T], 1)
                    mu, ls = self.m(torch.tensor(xx.T[None], dtype=torch.float32, device=DEV))
                    vs.append(Y.T @ mu[0].cpu().numpy()); ss.append(np.exp(ls[0].cpu().numpy()))
                v, s = np.mean(vs, 0), np.mean(ss, 0)
            else:
                mu, ls = self.m(torch.tensor(x.T[None], dtype=torch.float32, device=DEV))
                v, s = mu[0].cpu().numpy(), np.exp(ls[0].cpu().numpy())
        else:
            x, yaw = feats_dive(R, acc, len(acc) - 1)
            mu, ls = self.m(torch.tensor(x.T[None], dtype=torch.float32, device=DEV))
            v, s = yaw_rot(yaw) @ mu[0].cpu().numpy(), np.exp(ls[0].cpu().numpy())
        if self.corr is not None:                       # bias correction in the heading frame
            yaw = np.arctan2(v[1], v[0])
            Y = yaw_rot(yaw)
            vh = Y.T @ v
            vh = self.corr["scale"] * vh + self.corr["offset"]
            v = Y @ vh
        return v, s * self.vs


def standalone(pred, seqs, stride=40, attitude="gt"):
    errs, bias_h, z = [], [], []
    for s in seqs:
        for e in range(HZ * (4 if pred.grid else 1), len(s.t), stride):
            L = HZ * (4 if pred.grid else 1)
            sl = slice(e - L, e)
            R = s.R[sl] if attitude == "gt" else None
            v, sd = pred(s.acc[sl], s.gyr[sl], R)
            vt = s.vel[e - 1]
            errs.append(np.linalg.norm(v - vt))
            yaw = np.arctan2(vt[1], vt[0])
            if np.linalg.norm(vt[:2]) > 0.3:
                bias_h.append(yaw_rot(yaw).T @ (v - vt))       # error in the TRUE heading frame
            if sd is not None:
                z.append((v - vt) / np.maximum(sd, 1e-6))
    errs = np.array(errs)
    bh = np.array(bias_h)
    out = {"median": float(np.median(errs)), "p95": float(np.percentile(errs, 95)), "mean": float(errs.mean()),
           "n": len(errs), "bias_heading_frame_mean": bh.mean(0).tolist(), "bias_heading_frame_se": (bh.std(0) / np.sqrt(len(bh))).tolist()}
    if z:
        z = np.array(z)
        out["z_rms"] = np.sqrt((z ** 2).mean(0)).tolist()
    return out


def fit_bias_correction(pred, seqs, stride=40):
    """Per-axis affine map in the predicted heading frame, fitted on validation people only."""
    P, T = [], []
    for s in seqs:
        for e in range(HZ, len(s.t), stride):
            v, _ = pred(s.acc[e - HZ:e], s.gyr[e - HZ:e], s.R[e - HZ:e])
            if np.linalg.norm(v[:2]) < 0.3:
                continue
            Y = yaw_rot(np.arctan2(v[1], v[0]))
            P.append(Y.T @ v); T.append(Y.T @ s.vel[e - 1])
    P, T = np.array(P), np.array(T)
    scale, offset = np.ones(3), np.zeros(3)
    for k in range(3):
        A = np.c_[P[:, k], np.ones(len(P))]
        scale[k], offset[k] = np.linalg.lstsq(A, T[:, k], rcond=None)[0]
    return {"scale": scale, "offset": offset}


def var_scale(pred, seqs):
    z = []
    for s in seqs:
        for e in range(HZ, len(s.t), 40):
            v, sd = pred(s.acc[e - HZ:e], s.gyr[e - HZ:e], s.R[e - HZ:e])
            z.append((v - s.vel[e - 1]) / sd)
    return float(np.sqrt((np.array(z) ** 2).mean()))


# ---------------------------------------------------------------------------- EKF dead reckoning
def ekf_segment(pred, s, i0, i1, upd_every=0.1, use_nn=True):
    """Dead reckoning on [i0, i1): IMU mechanisation at 50 Hz + network velocity updates at 10 Hz.
    Attitude = phone rotation vector, heading-aligned to the ground truth at the segment start only."""
    R_rv = s.Rrv
    off = (s.R[i0] * R_rv[i0].inv()).as_euler("zyx")[0]
    Roff = Rot.from_euler("z", off)
    Rest = lambda sl: Roff * R_rv[sl]
    dt = 4.0 / HZ
    x = np.r_[s.pos[i0], s.vel[i0]]
    Fm = np.eye(6); Fm[:3, 3:] = np.eye(3) * dt
    state = {"a": np.zeros(3)}
    f = lambda x: np.r_[x[:3] + x[3:] * dt + 0.5 * state["a"] * dt ** 2, x[3:] + state["a"] * dt]
    h = lambda x: x[3:]
    Hm = np.zeros((3, 6)); Hm[:, 3:] = np.eye(3)
    sa = 0.3
    Q = np.diag([(0.5 * sa * dt ** 2) ** 2] * 3 + [(sa * dt) ** 2] * 3)
    ekf = ExtendedKalmanFilter(6, 3, f, h, Q=Q, R=np.eye(3) * 0.1, F_jac=lambda x: Fm, H_jac=lambda x: Hm)
    ekf.initialize(x, np.diag([1e-4] * 3 + [1e-2] * 3))
    L = HZ * (4 if pred.grid else 1)
    k_upd = int(round(upd_every / dt))
    traj = [x[:3].copy()]
    for k, i in enumerate(range(i0, i1 - 4, 4)):
        sl = slice(i, i + 4)
        state["a"] = (Rest(sl).apply(s.acc[sl]) - np.array([0, 0, G])).mean(0)
        ekf.predict()
        if use_nn and k % k_upd == 0 and i + 4 - L >= 0:
            w = slice(i + 4 - L, i + 4)
            v, sd = pred(s.acc[w], s.gyr[w], Rest(w))
            ekf.R = np.diag(np.maximum(sd, 0.02) ** 2)
            ekf.update(v)
        traj.append(ekf.x[:3].copy())
    traj = np.array(traj)
    gt = s.pos[i0:i1:4][:len(traj)]
    dist = np.linalg.norm(np.diff(gt[:, :2], axis=0), axis=1).sum()
    err = np.linalg.norm(traj[-1, :2] - gt[-1, :2])
    vel_err = np.linalg.norm(ekf.x[3:] - s.vel[min(i1 - 1, len(s.vel) - 1)])
    return {"drift_rate_pct": float(100 * err / max(dist, 1e-6)), "final_err_m": float(err), "dist_m": float(dist),
            "final_vel_err": float(vel_err)}


def dead_reckoning(pred, seqs, seg_s=90.0):
    out = []
    for s in seqs:
        n = int(seg_s * HZ)
        starts = list(range(HZ * 5, len(s.t) - n, n)) or ([HZ * 5] if len(s.t) > HZ * 40 else [])
        for i0 in starts:
            i1 = min(i0 + n, len(s.t))
            out.append(ekf_segment(pred, s, i0, i1))
    d = np.array([o["drift_rate_pct"] for o in out])
    return {"n_segments": len(out), "drift_median_pct": float(np.median(d)) if len(d) else None,
            "drift_p95_pct": float(np.percentile(d, 95)) if len(d) else None, "segments": out}


# ---------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    if os.environ.get("RESUME") == "1" and (RES / "results.json").exists():
        OUT.update(json.loads((RES / "results.json").read_text()))
    tr, va, te = load("train"), load("val"), load("test")
    OUT["data"] = {s: {"n_seq": len(x), "hours": sum(q.t[-1] for q in x) / 3600, "people": SPLIT[s]} for s, x in
                   (("train", tr), ("val", va), ("test", te))}
    log(OUT["data"])
    # --- reproduction, chapter 3: TLIO-adapted vs DIVE
    models = {}
    for kind in ("tlio", "dive"):
        if os.environ.get("RESUME") == "1" and f"ch3_{kind}" in OUT and (RES / f"{kind}.pt").exists():
            # finished before the 2026-09-25 machine reboot: reload weights, keep the stored evaluation
            models[kind] = VelNet().to(DEV)
            models[kind].load_state_dict(torch.load(RES / f"{kind}.pt", map_location=DEV))
            log(kind, "reloaded from", RES / f"{kind}.pt")
            continue
        m, info = train_velnet(kind, tr, va)
        models[kind] = m
        p = Predictor(kind, m)
        OUT[f"ch3_{kind}"] = {"train": info, "standalone_gt_attitude": standalone(p, te),
                              "dead_reckoning_90s": dead_reckoning(p, te)}
        log(kind, json.dumps({k: v for k, v in OUT[f"ch3_{kind}"]["standalone_gt_attitude"].items()}),
            OUT[f"ch3_{kind}"]["dead_reckoning_90s"]["drift_median_pct"])
        save()
        torch.save(m.state_dict(), RES / f"{kind}.pt")
    # IMU-only integration (no network) as the reference for "what the network buys"
    OUT["imu_only_dead_reckoning_90s"] = _imu_only(te)
    save()
    # --- reproduction, chapter 4: grid-output PINN variants
    OUT["ch4_grid_pinn"] = {}
    for lam in ((0.0, 0.1) if SMOKE else (0.0, 0.01, 0.1, 1.0)):
        m, info = train_grid(tr, va, lam)
        p = Predictor("tlio", m, grid=True)
        # constant per-axis std from validation residuals (the thesis used a linear uncertainty model)
        res = []
        for s in va:
            for e in range(4 * HZ, len(s.t), 80):
                v, _ = p(s.acc[e - 4 * HZ:e], s.gyr[e - 4 * HZ:e], s.R[e - 4 * HZ:e])
                res.append(v - s.vel[e - 1])
        p.const_std = np.array(res).std(0)
        OUT["ch4_grid_pinn"][str(lam)] = {"train": info, "standalone_gt_attitude": standalone(p, te),
                                          "dead_reckoning_90s": dead_reckoning(p, te)}
        log("grid", lam, OUT["ch4_grid_pinn"][str(lam)]["standalone_gt_attitude"]["median"],
            OUT["ch4_grid_pinn"][str(lam)]["dead_reckoning_90s"]["drift_median_pct"])
        save()
    # --- improvements on the best chapter-3 network (chosen on VALIDATION drift, never test)
    best_kind = min(("tlio", "dive"), key=lambda k: standalone(Predictor(k, models[k]), va)["median"])
    m = models[best_kind]
    variants = {"baseline": Predictor(best_kind, m)}
    if best_kind == "tlio":
        variants["+ yaw TTA (8)"] = Predictor("tlio", m, tta=8)
    base_for_corr = variants.get("+ yaw TTA (8)", variants["baseline"])
    corr = fit_bias_correction(base_for_corr, va)
    variants["+ TTA + bias correction"] = Predictor(best_kind, m, tta=base_for_corr.tta, corr=corr)
    vs = var_scale(variants["+ TTA + bias correction"], va)
    variants["+ TTA + bias corr. + variance recalibration"] = Predictor(best_kind, m, tta=base_for_corr.tta, corr=corr, var_scale=vs)
    OUT["improvements"] = {"base_kind": best_kind, "bias_correction": {k: v.tolist() for k, v in corr.items()},
                           "variance_scale": vs, "variants": {}}
    for name, p in variants.items():
        OUT["improvements"]["variants"][name] = {"standalone_gt_attitude": standalone(p, te), "dead_reckoning_90s": dead_reckoning(p, te)}
        log("improve", name, OUT["improvements"]["variants"][name]["standalone_gt_attitude"]["median"],
            OUT["improvements"]["variants"][name]["dead_reckoning_90s"]["drift_median_pct"])
        save()
    veriphysics(variants, te)
    OUT["runtime_s"] = time.time() - t0
    save()


def _imu_only(seqs):
    class Null:
        grid = False
    out = []
    for s in seqs:
        n = int(90 * HZ)
        for i0 in (list(range(HZ * 5, len(s.t) - n, n)) or ([HZ * 5] if len(s.t) > HZ * 40 else [])):
            out.append(ekf_segment(Null(), s, i0, min(i0 + n, len(s.t)), use_nn=False))
    d = np.array([o["drift_rate_pct"] for o in out])
    return {"n_segments": len(out), "drift_median_pct": float(np.median(d)), "drift_p95_pct": float(np.percentile(d, 95))}


def veriphysics(variants, te):
    from pinneapple_analysis.uncertainty.calibration import CalibrationMetrics
    from pinneapple_analysis.verification.physics_confidence_score import CalibrationSummary, compute_physics_confidence
    from pinneapple_data.physics_case import PhysicsCase
    from pinneapple_pdb.benchmarks import BenchmarkEntry, register_benchmark
    xs, ys, idx = [], [], []
    for j, s in enumerate(te):
        for e in range(HZ, len(s.t), 100):
            xs.append((j, e)); ys.append(s.vel[e - 1])
    xs, ys = np.array(xs, np.float32), np.array(ys, np.float32)

    @register_benchmark("ridi_test_people_velocity")
    def _b():
        return BenchmarkEntry(name="ridi_test_people_velocity", description="Tango VIO velocity, RIDI test people",
                              reference_source="RIDI (Yan, Shan, Furukawa, ECCV 2018)", x_vars=("seq", "idx"),
                              y_vars=("vx", "vy", "vz"), reference_x=xs, reference_y=ys)
    out = {}
    for name, p in variants.items():
        P, S = [], []
        for (j, e) in xs.astype(int):
            s = te[j]
            v, sd = p(s.acc[e - HZ:e], s.gyr[e - HZ:e], s.R[e - HZ:e])
            P.append(v); S.append(sd)
        P, S = np.array(P), np.array(S)
        case = PhysicsCase(name=name, results={"seq": xs[:, 0], "idx": xs[:, 1], "vx": P[:, 0], "vy": P[:, 1], "vz": P[:, 2]},
                           reference_benchmark="ridi_test_people_velocity")
        b = case.validate_against_benchmark()
        ece = CalibrationMetrics.expected_calibration_error(torch.tensor(P.ravel()), torch.tensor(ys.ravel()), torch.tensor(S.ravel()))
        cov = CalibrationMetrics.coverage_at_level(torch.tensor(P.ravel()), torch.tensor(ys.ravel()), torch.tensor(S.ravel()), alpha=0.1)
        sc = compute_physics_confidence(benchmark_comparison=b,
                                        calibration_metrics=CalibrationSummary(ece=ece, coverage=cov, target_coverage=0.9))
        out[name] = {"overall_score": sc.overall_score, "coverage": sc.coverage, "components": [c.__dict__ for c in sc.components]}
        log(sc.summary())
    out["not_run"] = {"physics_guardrail": "no compiled PDE ProblemSpec for IMU kinematics",
                      "numerical_convergence": "no discretisation parameter", "geometry_ood": "no geometry input"}
    OUT["veriphysics"] = out


if __name__ == "__main__":
    main()
