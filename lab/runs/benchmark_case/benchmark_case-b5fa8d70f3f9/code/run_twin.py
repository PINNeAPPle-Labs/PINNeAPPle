"""Heated-channel digital twin: surrogates (layer 2), sparse sensing (layer 3), control (layer 4).

  python run_twin.py            -> results/results.json + artefacts
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surrogates import (  # noqa: E402
    DEV, RES, ROOT, SMOKE, ChannelData, ConvAE, DeepONetStepper, LatentMLP, PaddedFNO, POD, divergence,
    energy_residual, log, qois, rollout_error, rollout_field, train_field_model,
)

OUT = {}


def save():
    (RES / "results.json").write_text(json.dumps(OUT, indent=1, default=float))


# ---------------------------------------------------------------------------
# latent models
# ---------------------------------------------------------------------------

def fit_dmdc(D, pod, runs, ridge=1e-3):
    X, A, Y = D.pairs(runs)
    a0, a1 = pod.encode(X), pod.encode(Y)
    Z = np.concatenate([a0, A, np.ones((len(A), 1), np.float32)], 1)
    W = np.linalg.solve(Z.T @ Z + ridge * np.eye(Z.shape[1]), Z.T @ a1)
    return W


def dmdc_roll(pod, W):
    def roll(x0, acts):
        a = pod.encode(x0[None])
        out = [x0]
        for u in acts:
            a = np.concatenate([a, u[None], np.ones((1, 1), np.float32)], 1) @ W
            out.append(pod.decode(a)[0])
        return np.stack(out)
    return roll


def train_latent_mlp(D, pod, epochs, horizon=8, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr, va = D.runs("train"), D.runs("val")
    if SMOKE:
        tr, va, epochs = tr[:3], va[:2], 2
    Atr = {r: pod.encode(D.Fn[r]) for r in np.concatenate([tr, va])}
    r_dim = sum(pod.r)
    sc = np.concatenate([Atr[r] for r in tr]).std(0)
    sc = np.maximum(sc, 1e-2 * sc.max())  # floor: near-constant modes must not blow up the scaling
    m = LatentMLP(r_dim).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scT = torch.tensor(sc, device=DEV, dtype=torch.float32)
    best, state, t0 = 1e9, None, time.time()
    K = D.Fn.shape[1] - 1
    for ep in range(epochs):
        starts = [(r, k) for r in tr for k in range(K - horizon + 1)]
        order = rng.permutation(len(starts))
        for i in range(0, len(order), 64):
            sel = [starts[j] for j in order[i:i + 64]]
            a = torch.tensor(np.stack([Atr[r][k:k + horizon + 1] for r, k in sel]), device=DEV) / scT
            u = torch.tensor(np.stack([D.actn[r, k:k + horizon] for r, k in sel]), device=DEV)
            z, loss = a[:, 0], 0.0
            for k in range(horizon):
                z = m(z, u[:, k])
                loss = loss + F.mse_loss(z, a[:, k + 1])
            opt.zero_grad()
            (loss / horizon).backward()
            opt.step()
        sch.step()
        roll = latent_mlp_roll(pod, m, sc)
        v = rollout_error(D, roll, va)
        if np.isfinite(v) and (v < best or state is None):
            best, state = v, {k: x.clone() for k, x in m.state_dict().items()}
        if (ep + 1) % 10 == 0 or SMOKE:
            log(f"  [POD-MLP] ep {ep + 1}: val rollout relL2 {v:.4f}")
    if state is not None:
        m.load_state_dict(state)
    m.eval()
    return m, sc, {"best_val": best, "train_s": time.time() - t0, "n_params": sum(p.numel() for p in m.parameters())}


def latent_mlp_roll(pod, m, sc):
    scT = torch.tensor(sc, device=DEV, dtype=torch.float32)

    @torch.no_grad()
    def roll(x0, acts):
        z = torch.tensor(pod.encode(x0[None]), device=DEV) / scT
        out = [x0]
        for u in acts:
            z = m(z, torch.tensor(u[None], device=DEV))
            out.append(pod.decode((z * scT).cpu().numpy())[0])
        return np.stack(out)
    return roll


def train_deeponet(D, pod, epochs, seed=0, n_pts=2048):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr, va = D.runs("train"), D.runs("val")
    if SMOKE:
        tr, va, epochs = tr[:3], va[:2], 1
    xs, ys = np.meshgrid(np.array(D.meta["xc"]) / 10.0, np.array(D.meta["yc"]), indexing="ij")
    coords = torch.tensor(np.stack([xs.ravel(), ys.ravel()], 1), dtype=torch.float32)
    X, A, Y = D.pairs(tr)
    a0 = pod.encode(X)
    sc = a0.std(0)
    sc = np.maximum(sc, 1e-2 * sc.max())
    m = DeepONetStepper(sum(pod.r), coords).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Yf = torch.tensor(Y.reshape(len(Y), 4, -1).transpose(0, 2, 1))
    best, state, t0 = 1e9, None, time.time()
    for ep in range(epochs):
        order = rng.permutation(len(X))
        for i in range(0, len(order), 64):
            b = order[i:i + 64]
            idx = torch.tensor(rng.choice(coords.shape[0], n_pts, replace=False))
            pred = m(torch.tensor(a0[b] / sc, device=DEV), torch.tensor(A[b], device=DEV), idx.to(DEV))
            loss = F.mse_loss(pred, Yf[b][:, idx].to(DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()
        v = rollout_error(D, deeponet_roll(D, pod, m, sc), va)
        if np.isfinite(v) and (v < best or state is None):
            best, state = v, {k: x.clone() for k, x in m.state_dict().items()}
        log(f"  [DeepONet] ep {ep + 1}: val rollout relL2 {v:.4f} ({time.time() - t0:.0f}s)")
    if state is not None:
        m.load_state_dict(state)
    m.eval()
    return m, sc, {"best_val": best, "train_s": time.time() - t0, "n_params": sum(p.numel() for p in m.parameters())}


def deeponet_roll(D, pod, m, sc):
    @torch.no_grad()
    def roll(x0, acts):
        x, out = x0, [x0]
        for u in acts:
            a = torch.tensor(pod.encode(x[None]) / sc, device=DEV, dtype=torch.float32)
            y = m(a, torch.tensor(u[None], device=DEV))[0].cpu().numpy()      # (N, 4)
            x = y.T.reshape(4, D.nx, D.ny)
            out.append(x)
        return np.stack(out)
    return roll


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def evaluate(D, name, roll, families):
    res = {}
    for fam in families:
        runs = D.runs(fam)
        per, tim = [], []
        for r in runs:
            t0 = time.time()
            P = roll(D.Fn[r, 0], D.actn[r])
            tim.append(time.time() - t0)
            Pp, Tp = D.denorm(P), D.F[r]
            rel = {f: float(np.linalg.norm(Pp[1:, i] - Tp[1:, i]) / np.linalg.norm(Tp[1:, i])) for i, f in enumerate("uvpT")}
            qp, qt = qois(Pp, D), qois(Tp, D)
            # frames right after an inlet-velocity jump carry impulsive pressure spikes: report dp both ways
            jump = np.abs(np.diff(D.act[r, :, 4] - D.act[r, :, 0])) > 0.05
            calm = np.ones(len(Pp), bool); calm[1:][np.abs(D.act[r, :, 4] - D.act[r, :, 0]) > 0.05] = False
            q_err = {k: float(np.sqrt(((qp[k] - qt[k]) ** 2).mean())) for k in qp}
            q_err["dp_calm_frames"] = float(np.sqrt(((qp["dp"] - qt["dp"])[calm] ** 2).mean()))
            Pt = torch.tensor(Pp)
            div = divergence(Pt[:, :2], D.dx, D.dy).abs().mean().item()
            er = energy_residual(Pt[:-1], Pt[1:], torch.tensor(D.act[r]), D).abs().mean().item()
            finite = bool(np.isfinite(P).all())
            per.append({"run": int(r), "rel_l2": rel, "qoi_rmse": q_err, "div_mean_abs": div,
                        "energy_residual_mean_abs": er, "finite": finite})
        agg = lambda key: {k: float(np.mean([p[key][k] for p in per])) for k in per[0][key]}
        res[fam] = {"rel_l2": agg("rel_l2"), "qoi_rmse": agg("qoi_rmse"),
                    "div_mean_abs": float(np.mean([p["div_mean_abs"] for p in per])),
                    "energy_residual_mean_abs": float(np.mean([p["energy_residual_mean_abs"] for p in per])),
                    "all_finite": all(p["finite"] for p in per), "rollout_seconds": float(np.mean(tim)),
                    "per_run": per}
        log(f"  {name} [{fam}] relL2 u {res[fam]['rel_l2']['u']:.3f} v {res[fam]['rel_l2']['v']:.3f} "
            f"p {res[fam]['rel_l2']['p']:.3f} T {res[fam]['rel_l2']['T']:.3f} | Tout {res[fam]['qoi_rmse']['T_out']:.4f} "
            f"Tmax {res[fam]['qoi_rmse']['T_max']:.4f} | energy res {res[fam]['energy_residual_mean_abs']:.3f}")
    return res


def truth_physics(D, families):
    out = {}
    for fam in families:
        d, e = [], []
        for r in D.runs(fam):
            Pt = torch.tensor(D.F[r])
            d.append(divergence(Pt[:, :2], D.dx, D.dy).abs().mean().item())
            e.append(energy_residual(Pt[:-1], Pt[1:], torch.tensor(D.act[r]), D).abs().mean().item())
        out[fam] = {"div_mean_abs": float(np.mean(d)), "energy_residual_mean_abs": float(np.mean(e))}
    return out


# ---------------------------------------------------------------------------
# layer 3 -- sparse sensing with the PINNeAPPle EnKF
# ---------------------------------------------------------------------------

SENSORS = [("T1", 3, 5.0, 0.5), ("T2", 3, 9.0, 0.25), ("P1", 2, 0.5, 0.5)]  # (name, field, x, y)
NOISE = {3: 0.002, 2: 0.02}


def _simulate_truth(args):
    """CFD 'plant' with a heater gain error: the real heat flux is gain * commanded power."""
    run, gain, U_fine, Q_fine = args
    sys.path.insert(0, str(ROOT.parents[2] / "PINNeAPPle"))
    from pinneapple_simulation.numerical_solvers.thermal_channel_2d import ChannelConfig, ThermalChannel2D
    t_f = np.arange(0, 40.0001, 0.1)
    s = ThermalChannel2D(ChannelConfig(nx=128, ny=32, dt=0.01, record_every=0.4))
    s.reset(float(U_fine[0]), float(Q_fine[0]) * gain)
    s.run(lambda t: float(U_fine[0]), lambda t: float(Q_fine[0]) * gain, 15.0, record=False)
    s.t = 0.0
    rec = s.run(lambda t: float(np.interp(t, t_f, U_fine)), lambda t: gain * float(np.interp(t, t_f, Q_fine)), 40.0)
    return run, gain, rec["fields"]


def layer3(D, pod, lat_m, sc):
    from pinneapple_analysis.state_estimation.kalman import EnsembleKalmanFilter
    z = np.load(ROOT / "data" / "trajectories.npz")
    runs = D.runs("test_same_family")[:4].tolist() + D.runs("test_unseen_sinusoid")[:2].tolist()
    if SMOKE:
        runs = runs[:1]
    jobs = [(r, g, z["U_fine"][r], z["Q_fine"][r]) for r in runs for g in (1.0, 1.3)]
    with ProcessPoolExecutor(4) as ex:
        truths = list(ex.map(_simulate_truth, jobs))
    xc, yc = np.array(D.meta["xc"]), np.array(D.meta["yc"])
    sens_idx = [(f, int(np.abs(xc - x).argmin()), int(np.abs(yc - y).argmin())) for _, f, x, y in SENSORS]
    r_dim = sum(pod.r)
    scT = torch.tensor(sc, device=DEV, dtype=torch.float32)
    # observation operator is linear in the POD coefficients: y = H a + c
    Hrows = []
    for f, i, j in sens_idx:
        off = sum(pod.r[:f])
        row = np.zeros(r_dim)
        row[off:off + pod.r[f]] = pod.modes[f][:, i * D.ny + j]
        Hrows.append(row)
    Hm = np.array(Hrows)
    sd_obs = np.array([NOISE[f] / D.sd[0, f, 0, 0] for f, _, _ in sens_idx])  # noise in normalised units
    rng = np.random.default_rng(0)
    res = {"sensors": SENSORS, "noise": {str(k): v for k, v in NOISE.items()}, "cases": []}

    @torch.no_grad()
    def step_latent(a, u_norm, gain):
        u = u_norm.copy()
        q_phys = u[5:] * D.a_sd[5:] + D.a_mu[5:]
        u[5:] = (gain * q_phys - D.a_mu[5:]) / D.a_sd[5:]          # forecast with the current gain estimate
        zt = torch.tensor(a[None] / sc, device=DEV, dtype=torch.float32)
        return (lat_m(zt, torch.tensor(u[None], device=DEV)) * scT)[0].cpu().numpy()

    for run, gain, fields in truths:
        truth_n = (fields - D.mu) / D.sd
        K = len(truth_n) - 1
        obs = np.array([[truth_n[k, f, i, j] for f, i, j in sens_idx] for k in range(K + 1)]) + rng.normal(0, sd_obs, (K + 1, len(sens_idx)))
        a0 = pod.encode(truth_n[:1])[0]
        # (a) open loop
        a, ol = a0.copy(), [a0]
        for k in range(K):
            a = step_latent(a, D.actn[run, k], 1.0)
            ol.append(a)
        ol = pod.decode(np.array(ol))
        # (b) EnKF on the latent state, (c) EnKF with augmented heater-gain parameter
        def run_enkf(augment):
            n = r_dim + (1 if augment else 0)
            state = {"k": 0}
            f = lambda x: np.concatenate([step_latent(x[:r_dim], D.actn[run, state["k"]], x[-1] if augment else 1.0),
                                          x[r_dim:]])
            h = lambda x: Hm @ x[:r_dim]
            Q = np.diag(np.concatenate([(0.02 * sc) ** 2, [1e-5] if augment else []]))
            enkf = EnsembleKalmanFilter(n, len(sens_idx), f, h, Q=Q, R=np.diag(sd_obs ** 2), n_ens=48 if not SMOKE else 8,
                                        inflation=1.02, seed=1)
            x0 = np.concatenate([a0, [1.0] if augment else []])
            P0 = np.diag(np.concatenate([(0.05 * sc) ** 2, [0.2 ** 2] if augment else []]))
            enkf.initialize(x0, P0)
            est, gains = [a0], [1.0]
            for k in range(K):
                state["k"] = k
                out = enkf.step(obs[k + 1])
                est.append(out["x"][:r_dim])
                gains.append(float(out["x"][-1]) if augment else 1.0)
            return pod.decode(np.array(est)), gains
        enkf_s, _ = run_enkf(False)
        enkf_g, gains = run_enkf(True)
        # (d) gappy POD: sensors only, ridge-regularised least squares on the POD coefficients
        lam = 1e-2
        gp = [np.linalg.solve(Hm.T @ Hm + lam * np.eye(r_dim), Hm.T @ o) for o in obs]
        gappy = pod.decode(np.array(gp))

        def err(P):
            Pp, Tt = D.denorm(P), fields
            eT = float(np.linalg.norm(Pp[1:, 3] - Tt[1:, 3]) / np.linalg.norm(Tt[1:, 3]))
            q_p, q_t = qois(Pp, D), qois(Tt, D)
            return {"T_rel_l2": eT, "T_out_rmse": float(np.sqrt(((q_p["T_out"] - q_t["T_out"]) ** 2).mean())),
                    "T_max_rmse": float(np.sqrt(((q_p["T_max"] - q_t["T_max"]) ** 2).mean()))}
        case = {"run": int(run), "family": str(D.split[run]), "true_gain": gain, "open_loop": err(ol),
                "enkf_state": err(enkf_s), "enkf_state_and_gain": err(enkf_g), "gappy_pod_sensors_only": err(gappy),
                "gain_estimate_final": gains[-1], "gain_trace": gains[::5]}
        res["cases"].append(case)
        log(f"  L3 run {run} gain {gain}: OL {case['open_loop']['T_rel_l2']:.3f}  EnKF {case['enkf_state']['T_rel_l2']:.3f}  "
            f"EnKF+gain {case['enkf_state_and_gain']['T_rel_l2']:.3f} (gain est {gains[-1]:.3f})  gappy {case['gappy_pod_sensors_only']['T_rel_l2']:.3f}")
        if len(res["cases"]) == 1:
            np.savez_compressed(RES / "layer3_example.npz", truth=fields[:, 3], ol=D.denorm(ol)[:, 3], enkf=D.denorm(enkf_g)[:, 3],
                                gappy=D.denorm(gappy)[:, 3], gains=np.array(gains), sens=np.array(sens_idx))
    return res


# ---------------------------------------------------------------------------
# layer 4 -- open-loop optimisation through the differentiable surrogate, verified by CFD
# ---------------------------------------------------------------------------

def layer4(D, pod, lat_m, sc, target=0.05, w_q=0.002, n_seg=10):
    from pinneapple_simulation.numerical_solvers.thermal_channel_2d import ChannelConfig, ThermalChannel2D
    K = D.Fn.shape[1] - 1
    x0 = D.Fn[D.runs("val")[0], 0]      # a realistic initial state (validation, not test)
    U_cmd = 1.0
    scT = torch.tensor(sc, device=DEV, dtype=torch.float32)
    mu, sdv = torch.tensor(D.mu, device=DEV), torch.tensor(D.sd, device=DEV)
    theta = torch.zeros(n_seg, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.05)
    a_mu, a_sd = torch.tensor(D.a_mu, device=DEV), torch.tensor(D.a_sd, device=DEV)

    def simulate(theta):
        q = 2.0 * torch.sigmoid(theta)                                    # heater power in [0, 2]
        z = torch.tensor(pod.encode(x0[None]), device=DEV, dtype=torch.float32) / scT
        touts = []
        for k in range(K):
            qk = q[min(k * n_seg // K, n_seg - 1)]
            u_phys = torch.cat([torch.full((5,), U_cmd, device=DEV), qk.repeat(5)])
            z = lat_m(z, ((u_phys - a_mu) / a_sd)[None])
            s = pod.torch_decode(z * scT) * sdv + mu
            touts.append((s[0, 0, -1] * s[0, 3, -1]).sum() / s[0, 0, -1].sum())
        return q, torch.stack(touts)

    t0 = time.time()
    for it in range(20 if SMOKE else 300):
        q, tout = simulate(theta)
        cost = ((tout - target) ** 2).mean() + w_q * (q ** 2).mean()
        opt.zero_grad()
        cost.backward()
        opt.step()
    with torch.no_grad():
        q, tout = simulate(theta)
    q = q.cpu().numpy()
    opt_s = time.time() - t0
    # verify in CFD from the same initial state (validation trajectory's first frame)
    x_init = D.F[D.runs("val")[0], 0]
    s = ThermalChannel2D(ChannelConfig(nx=128, ny=32, dt=0.01, record_every=0.4))
    s.reset(U_cmd, float(q[0]))
    s.u[:, 1:-1] = 0.0
    # initialise the CFD state from the recorded fields (cell-centred -> staggered by averaging)
    uc, vc, pc, Tc = x_init
    s.u[1:-1, 1:-1] = 0.5 * (uc[1:] + uc[:-1]); s.u[0, 1:-1] = s.inlet_profile(U_cmd); s.u[-1, 1:-1] = uc[-1]
    s.v[1:-1, 1:-1] = 0.5 * (vc[:, 1:] + vc[:, :-1]); s.p[1:-1, 1:-1] = pc; s.T[1:-1, 1:-1] = Tc
    s._apply_bc()
    t_cfd = time.time()
    rec = s.run(lambda t: U_cmd, lambda t: float(q[min(int(t / 40.0 * n_seg), n_seg - 1)]), 40.0)
    t_cfd = time.time() - t_cfd
    q_cfd = qois(rec["fields"], D)["T_out"]
    tout = tout.cpu().numpy()
    res = {"target_T_out": target, "q_segments": q.tolist(), "optimisation_seconds": opt_s, "cfd_seconds": t_cfd,
           "surrogate_tracking_rmse": float(np.sqrt(((tout - target) ** 2).mean())),
           "cfd_tracking_rmse": float(np.sqrt(((q_cfd[1:] - target) ** 2).mean())),
           "surrogate_vs_cfd_T_out_rmse": float(np.sqrt(((tout - q_cfd[1:]) ** 2).mean())),
           "tout_surrogate": tout.tolist(), "tout_cfd": q_cfd.tolist()}
    log(f"  L4: surrogate tracking {res['surrogate_tracking_rmse']:.4f}  CFD tracking {res['cfd_tracking_rmse']:.4f}"
        f"  gap {res['surrogate_vs_cfd_T_out_rmse']:.4f}  q={np.round(q, 2)}")
    return res


# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    D = ChannelData()
    families = ["test_same_family", "test_unseen_sinusoid", "test_unseen_edge_steps"]
    OUT["data"] = {"n_runs": int(len(D.split)), "splits": {s: int((D.split == s).sum()) for s in np.unique(D.split)},
                   "cfd_seconds_per_run_mean": float(np.mean(D.meta["solver_seconds"])), "grid": [D.nx, D.ny]}
    OUT["truth_physics_floor"] = truth_physics(D, families)
    ep = 1 if SMOKE else 20
    pod = POD(D, D.runs("train"))
    OUT["pod_modes"] = pod.r
    models = {}
    # 1. POD + DMDc
    t0 = time.time()
    W = fit_dmdc(D, pod, D.runs("train"))
    models["POD + DMDc (linear latent)"] = (dmdc_roll(pod, W), {"train_s": time.time() - t0})
    # 2. POD + MLP latent dynamics
    lat_m, sc, info = train_latent_mlp(D, pod, epochs=2 if SMOKE else 200)
    models["POD + MLP latent dynamics"] = (latent_mlp_roll(pod, lat_m, sc), info)
    # 3. Convolutional autoencoder + latent stepper
    cae = ConvAE().to(DEV)
    X, _, _ = D.pairs(D.runs("train"))
    opt = torch.optim.AdamW(cae.parameters(), lr=1e-3)
    Xt = torch.tensor(X)
    for e in range(1 if SMOKE else 40):                         # stage 1: reconstruction
        for i in torch.randperm(len(Xt)).split(32):
            xb = Xt[i].to(DEV)
            loss = F.mse_loss(cae.decode(cae.encode(xb)), xb)
            opt.zero_grad(); loss.backward(); opt.step()
    cae, info = train_field_model(D, cae, "CAE", epochs=ep, horizon=4, lr=5e-4)
    models["Conv autoencoder + latent MLP"] = (lambda x0, a, m=cae: rollout_field(m, x0, a), info)
    # 4. FNO (data only) and 5. physics-informed FNO (lambda chosen on validation)
    fno, info = train_field_model(D, PaddedFNO(), "FNO", epochs=ep, horizon=4)
    models["FNO (PINNeAPPle FNO2d)"] = (lambda x0, a, m=fno: rollout_field(m, x0, a), info)
    pi_cands = {}
    for lam in ((0.1,) if SMOKE else (0.1, 1.0)):
        m, info = train_field_model(D, PaddedFNO(), f"PI-FNO lam={lam}", epochs=ep, horizon=4, phys_lambda=lam)
        pi_cands[lam] = (m, info)
    lam_best = min(pi_cands, key=lambda l: pi_cands[l][1]["best_val"])
    pim, info = pi_cands[lam_best]
    info["lambda"] = lam_best
    info["lambda_candidates"] = {str(l): c[1]["best_val"] for l, c in pi_cands.items()}
    models["Physics-informed FNO"] = (lambda x0, a, m=pim: rollout_field(m, x0, a), info)
    # 6. DeepONet
    don, dsc, info = train_deeponet(D, pod, epochs=1 if SMOKE else 40)
    models["DeepONet (PINNeAPPle)"] = (deeponet_roll(D, pod, don, dsc), info)
    OUT["models"] = {}
    for name, (roll, info) in models.items():
        OUT["models"][name] = {"train": {k: v for k, v in info.items() if k != "history"},
                               "val_rollout_rel_l2": rollout_error(D, roll, D.runs("val")),
                               "test": evaluate(D, name, roll, families)}
        OUT["models"][name]["history"] = info.get("history")
        save()
    # which model is best on validation (never on test)
    OUT["selected_on_val"] = min(OUT["models"], key=lambda k: OUT["models"][k]["val_rollout_rel_l2"])
    torch.save({"fno": fno.state_dict(), "pi_fno": pim.state_dict(), "lat": lat_m.state_dict(), "sc": sc}, RES / "models.pt")
    # example rollout for figures
    r = D.runs("test_unseen_sinusoid")[0]
    np.savez_compressed(RES / "example_rollout.npz", truth=D.F[r], act=D.act[r],
                        **{n.split(" ")[0].replace("+", "").replace("(", ""): D.denorm(models[n][0](D.Fn[r, 0], D.actn[r]))
                           for n in ("FNO (PINNeAPPle FNO2d)", "POD + MLP latent dynamics", "Physics-informed FNO")})
    log("== layer 3")
    OUT["layer3"] = layer3(D, pod, lat_m, sc)
    save()
    log("== layer 4")
    OUT["layer4"] = layer4(D, pod, lat_m, sc)
    OUT["runtime_s"] = time.time() - t_start
    save()
    veriphysics(D, models)
    save()
    log("done", OUT["runtime_s"])


def veriphysics(D, models):
    from pinneapple_analysis.verification.convergence import richardson_extrapolate
    from pinneapple_analysis.verification.physics_confidence_score import compute_physics_confidence
    from pinneapple_data.physics_case import PhysicsCase
    from pinneapple_pdb.benchmarks import BenchmarkEntry, register_benchmark
    ver = json.loads((ROOT / "results" / "verification.json").read_text())
    gc = ver["grid_convergence"]["T_out"]
    conv = richardson_extrapolate(*gc["values"], r=2.0)
    runs = D.runs("test_unseen_sinusoid")
    xs = np.array([(r, k) for r in runs for k in range(1, D.F.shape[1])], np.float32)
    ys = np.concatenate([qois(D.F[r], D)["T_out"][1:] for r in runs]).astype(np.float32)[:, None]

    @register_benchmark("heated_channel_Tout_unseen_sinusoid")
    def _b():
        return BenchmarkEntry(name="heated_channel_Tout_unseen_sinusoid", description="CFD outlet temperature, unseen actuator family",
                              reference_source="PINNeAPPle ThermalChannel2D (verified: Poiseuille, Nu=5.385)",
                              x_vars=("run", "frame"), y_vars=("T_out",), reference_x=xs, reference_y=ys)
    out = {}
    for name, (roll, _) in models.items():
        yp = np.concatenate([qois(D.denorm(roll(D.Fn[r, 0], D.actn[r])), D)["T_out"][1:] for r in runs])
        yp = yp.astype(np.float64)
        if not np.isfinite(yp).all():   # a diverged rollout gets no score; reported as such
            out[name] = {"overall_score": None, "coverage": 0.0, "note": "rollout produced non-finite values"}
            continue
        case = PhysicsCase(name=name, results={"run": xs[:, 0], "frame": xs[:, 1], "T_out": yp},
                           reference_benchmark="heated_channel_Tout_unseen_sinusoid")
        bench = case.validate_against_benchmark()
        if not np.isfinite(bench.relative_l2_error):
            out[name] = {"overall_score": None, "coverage": 0.0, "note": "rollout diverged (overflow in the error)"}
            continue
        sc = compute_physics_confidence(benchmark_comparison=bench, convergence_result=conv)
        out[name] = {"overall_score": sc.overall_score, "coverage": sc.coverage, "components": [c.__dict__ for c in sc.components]}
        log(sc.summary())
    out["note"] = ("numerical_convergence is the CFD reference's own 3-grid GCI on T_out (shared by all surrogates); "
                   "physics_guardrail not run: no compiled ProblemSpec for this time-dependent actuated NS+energy system; "
                   "uq_calibration not run: the surrogates are deterministic single models")
    OUT["veriphysics"] = out


if __name__ == "__main__":
    main()
