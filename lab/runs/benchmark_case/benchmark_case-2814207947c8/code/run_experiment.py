"""Full experiment for the PINNeAPPle Physics Digital Twin paper.

Phases (challenge roadmap): data -> virtual sensors -> baseline ML -> physics model ->
PINN (PINNeAPPle) -> parameter identification -> validation -> anomaly detection ->
what-if -> insights. Every number in results/ is produced by this script.

Split: train 2015-2021, validation 2022 (model/hyper-parameter selection and early
stopping only), test 2023-2024 (touched once, at the end).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import os
import numpy as np
import torch

SMOKE = os.environ.get("SMOKE") == "1"
EP = 1 if SMOKE else 25
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[2] / "PINNeAPPle"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from twin_core import (  # noqa: E402
    C_ASSUMED, H, PhysicsModel, Standardizer, ThermalParams, ThermalPINN, Windows, load_weather,
    make_windows, physics_residual, rhs, simulate, split_indices, to_t,
)

RES = ROOT / "results"
RES.mkdir(exist_ok=True)
torch.set_num_threads(6)
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def one_hour_step(Tprev, k, Ta, W, S, dq, p, substeps=4):
    """Integrate the ODE from tau=k to k+1 starting at Tprev (N,)."""
    N = Tprev.shape[0]
    T = Tprev.clone()
    h = 1.0 / substeps
    dqc = dq[:, None]
    for j in range(substeps):
        t = torch.full((N, 1), k + j * h)
        Tc = T[:, None]
        k1 = rhs(Tc, t, Ta, W, S, dqc, p)
        k2 = rhs(Tc + 0.5 * h * k1, t + 0.5 * h, Ta, W, S, dqc, p)
        k3 = rhs(Tc + 0.5 * h * k2, t + 0.5 * h, Ta, W, S, dqc, p)
        k4 = rhs(Tc + h * k3, t + h - 1e-7, Ta, W, S, dqc, p)
        T = T + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)[:, 0]
    return T


@torch.no_grad()
def discrete_physics_residual(pred: np.ndarray, w: Windows, p) -> np.ndarray:
    """r_h = T_hat(h) - Phi_1h(T_hat(h-1)) under the identified ODE, K/h. Shape (N,H)."""
    T0, Ta, W, S, dq, _ = to_t(w)
    P = torch.as_tensor(pred, dtype=torch.float32)
    prev = torch.cat([T0[:, None], P[:, :-1]], 1)
    r = [P[:, k] - one_hour_step(prev[:, k], k, Ta, W, S, dq, p) for k in range(H)]
    return torch.stack(r, 1).numpy()


def metrics(pred, w: Windows):
    e = pred - w.Y
    dT_obs = w.Y - w.T0[:, None]
    out = {}
    for h in (1, 3, 6):
        eh = e[:, h - 1]
        out[f"h{h}"] = {
            "MAE": float(np.abs(eh).mean()),
            "RMSE": float(np.sqrt((eh ** 2).mean())),
            "rel_L2_dT": float(np.sqrt((eh ** 2).mean()) / np.sqrt((dT_obs[:, h - 1] ** 2).mean())),
            "bias": float(eh.mean()),
        }
    return out


def block_bootstrap_rmse(err, starts, n_boot=1000, block=24 * 7, seed=0):
    """95% CI of RMSE with weekly block bootstrap (temporal autocorrelation)."""
    rng = np.random.default_rng(seed)
    b = (starts - starts.min()) // block
    ub = np.unique(b)
    groups = [np.where(b == u)[0] for u in ub]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        vals.append(np.sqrt((err[idx] ** 2).mean()))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ---------------------------------------------------------------------------
# baselines (data-driven, no explicit physics)
# ---------------------------------------------------------------------------

def fit_baselines(wtr, wva):
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.multioutput import MultiOutputRegressor

    Xtr, Xva = wtr.context(), wva.context()
    st = Standardizer(Xtr)
    Ytr = wtr.Y - wtr.T0[:, None]
    models, tuning = {}, {}

    def val_rmse(m):
        p = m.predict(st(Xva)) + wva.T0[:, None]
        return float(np.sqrt(((p - wva.Y) ** 2).mean()))

    def pick(name, candidates):
        best, best_s, rows = None, 1e9, []
        for hp, mk in candidates:
            t0 = time.time()
            m = mk().fit(st(Xtr), Ytr)
            s = val_rmse(m)
            rows.append({"hp": hp, "val_rmse_all_h": s, "fit_s": time.time() - t0})
            log(f"  {name} {hp}: val RMSE={s:.4f}")
            if s < best_s:
                best, best_s = m, s
        tuning[name] = rows
        models[name] = best

    pick("Linear (Ridge)", [(f"alpha={a}", (lambda a=a: Ridge(alpha=a))) for a in (0.1, 10.0, 1000.0)])
    pick("Random Forest", [(f"max_depth={d},min_leaf={l}",
                            (lambda d=d, l=l: RandomForestRegressor(200, max_depth=d, min_samples_leaf=l,
                                                                    max_features=0.5, n_jobs=6, random_state=0)))
                           for d, l in ((12, 5), (20, 5), (None, 20))])
    pick("Gradient Boosting", [(f"lr={lr},leaves={lv}",
                                (lambda lr=lr, lv=lv: MultiOutputRegressor(HistGradientBoostingRegressor(
                                    learning_rate=lr, max_leaf_nodes=lv, max_iter=600, early_stopping=True,
                                    validation_fraction=0.1, random_state=0))))
                               for lr, lv in ((0.05, 31), (0.1, 63))])
    pick("MLP", [(f"hidden={h},alpha={a}",
                  (lambda h=h, a=a: MLPRegressor(hidden_layer_sizes=h, alpha=a, early_stopping=True, max_iter=300,
                                                 random_state=0, learning_rate_init=1e-3)))
                 for h, a in (((128, 128), 1e-4), ((256, 256, 128), 1e-3))])
    return models, st, tuning


class SkWrap:
    def __init__(self, m, st, name):
        self.m, self.st, self.name = m, st, name

    def predict(self, w):
        return self.m.predict(self.st(w.context())) + w.T0[:, None]


# ---------------------------------------------------------------------------
# PINN training
# ---------------------------------------------------------------------------

def augment_contexts(w: Windows, rng) -> Windows:
    """Physics-only collocation contexts: perturbed forcing + internal heat. No labels are used."""
    n = len(w)
    S = np.clip(w.S * rng.uniform(0.5, 1.6, (n, 1)), 0, 1300).astype(np.float32)
    W = (w.W * rng.uniform(0.2, 2.0, (n, 1))).astype(np.float32)
    Ta = (w.Ta + rng.uniform(-3, 3, (n, 1))).astype(np.float32)
    dq = (rng.uniform(0, 1, n) < 0.5) * rng.uniform(0, 150, n) * 3600 / C_ASSUMED
    return Windows(T0=w.T0, Ta=Ta, W=W, S=S, RH=w.RH, hour=w.hour, Y=w.Y, start=w.start, dq=dq.astype(np.float32))


def train_pinn(wtr, wva, lam, *, init: ThermalParams, seed=0, epochs=30, bs=512, n_col=4, augment=True, tag=""):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    st = Standardizer(wtr.context())
    model = ThermalPINN(ctx_dim=wtr.context().shape[1], init=init)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    steps = epochs * math.ceil(len(wtr) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    cva = torch.as_tensor(st(wva.context()))
    T0v, *_ = to_t(wva)
    tau_d = torch.arange(1, H + 1, dtype=torch.float32)
    best, best_state, hist = 1e9, None, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(wtr))
        agg = {"data": 0.0, "phys": 0.0, "n": 0}
        for i in range(0, len(perm), bs):
            b = wtr.subset(perm[i:i + bs])
            T0, Ta, W, S, dq, Y = to_t(b)
            c = torch.as_tensor(st(b.context()))
            Tp = model(c, T0, tau_d.expand(len(b), H))
            l_data = ((Tp - Y) ** 2).mean()
            loss = l_data
            l_phys = torch.tensor(0.0)
            if lam > 0:
                tau = torch.rand(len(b), n_col) * H
                r, _ = physics_residual(model, c, T0, Ta, W, S, dq, tau)
                l_phys = (r ** 2).mean()
                if augment:
                    ba = augment_contexts(b, rng)
                    T0a, Taa, Wa, Sa, dqa, _ = to_t(ba)
                    ca = torch.as_tensor(st(ba.context()))
                    ra, _ = physics_residual(model, ca, T0a, Taa, Wa, Sa, dqa, torch.rand(len(b), n_col) * H)
                    l_phys = 0.5 * (l_phys + (ra ** 2).mean())
                loss = l_data + lam * l_phys
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            agg["data"] += float(l_data) * len(b)
            agg["phys"] += float(l_phys) * len(b)
            agg["n"] += len(b)
        model.eval()
        with torch.no_grad():
            pv = model(cva, T0v, tau_d.expand(len(wva), H)).numpy()
        v = float(np.sqrt(((pv - wva.Y) ** 2).mean()))
        p = [float(x) for x in model.params()]
        hist.append({"epoch": ep + 1, "train_data_mse": agg["data"] / agg["n"], "train_phys_mse": agg["phys"] / agg["n"],
                     "val_rmse": v, "a": p[0], "b0": p[1], "b1": p[2], "q": p[3]})
        if v < best:
            best, best_state = v, {k: x.clone() for k, x in model.state_dict().items()}
        log(f"  PINN{tag} lam={lam} seed={seed} ep={ep + 1}: train_data={hist[-1]['train_data_mse']:.4f} "
            f"phys={hist[-1]['train_phys_mse']:.4f} val_rmse={v:.4f} params={np.round(p, 5).tolist()}")
    model.load_state_dict(best_state)
    model.train_seconds = time.time() - t0
    return PinnWrap(model, st, lam), hist


class PinnWrap:
    def __init__(self, model, st, lam):
        self.model, self.st, self.lam = model, st, lam
        self.name = "PINN (PINNeAPPle)" if lam > 0 else "NN, same architecture, no physics loss"

    @torch.no_grad()
    def predict(self, w: Windows):
        T0, *_ = to_t(w)
        c = torch.as_tensor(self.st(w.context()))
        out = []
        for i in range(0, len(w), 4096):
            out.append(self.model(c[i:i + 4096], T0[i:i + 4096],
                                  torch.arange(1, H + 1, dtype=torch.float32).expand(len(T0[i:i + 4096]), H)))
        return torch.cat(out).numpy()

    def continuous_residual(self, w: Windows, n_col=16, seed=123):
        """|dT/dt - f| (K/h) on FRESH random collocation times, with the PINN's own parameters."""
        g = torch.Generator().manual_seed(seed)
        T0, Ta, W, S, dq, _ = to_t(w)
        c = torch.as_tensor(self.st(w.context()))
        rs = []
        for i in range(0, len(w), 2048):
            sl = slice(i, i + 2048)
            tau = torch.rand(len(T0[sl]), n_col, generator=g) * H
            r, _ = physics_residual(self.model, c[sl], T0[sl], Ta[sl], W[sl], S[sl], dq[sl], tau)
            rs.append(r.detach())
        r = torch.cat(rs)
        return {"mean_abs_K_per_h": float(r.abs().mean()), "rms_K_per_h": float(r.pow(2).mean().sqrt())}


class OdeWrap:
    def __init__(self, pm: PhysicsModel):
        self.pm, self.name = pm, "Physics ODE (identified)"

    def predict(self, w):
        return self.pm.predict(w)


class Persistence:
    name = "Persistence"

    def predict(self, w):
        return np.repeat(w.T0[:, None], H, 1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    d = load_weather()
    tr, va, te = split_indices(d)
    if SMOKE:  # fast code-path check only
        tr, va, te = tr[::25], va[::10], te[::10]
    wtr, wva, wte = make_windows(d, tr), make_windows(d, va), make_windows(d, te)
    R = {"data": {"n_rows": len(d), "start": str(d.time.iloc[0]), "end": str(d.time.iloc[-1]),
                  "n_windows": {"train": len(wtr), "val": len(wva), "test": len(wte)},
                  "summary": d.drop(columns=["time"]).describe().round(3).to_dict()}}

    # Phase 5/7 -- physics model + parameter identification (least squares, multi-step shooting)
    log("== Physics ODE identification")
    t0 = time.time()
    pm = PhysicsModel().fit(wtr, iters=20 if SMOKE else 300)
    R["physics_ode"] = {"params": pm.params.__dict__, "physical": pm.params.physical(), "fit_s": time.time() - t0}
    log(" ", pm.params, pm.params.physical())

    # sensitivity: re-identify on each training year separately (parameter stability)
    yrs = d.time.dt.year.to_numpy()[wtr.start]
    R["physics_ode"]["per_year"] = {}
    for y in np.unique(yrs):
        p_y = PhysicsModel().fit(wtr.subset(yrs == y), iters=10 if SMOKE else 200).params
        R["physics_ode"]["per_year"][int(y)] = p_y.__dict__
        log(f"  year {y}: {p_y}")

    # EKI cross-check (PINNeAPPle inverse_problems), derivative-free, on a random subset
    log("== EKI cross-check")
    from pinneapple_analysis.inverse_problems.ensemble_kalman import EKIConfig, EnsembleKalmanInversion
    rng = np.random.default_rng(1)
    sub = wtr.subset(rng.choice(len(wtr), 400, replace=False))
    T0s, Tas, Ws, Ss, dqs, Ys = to_t(sub)

    def G(theta):
        out = []
        for th in theta:
            p = (torch.tensor(math.exp(th[0])), torch.tensor(math.exp(th[1])), torch.tensor(math.exp(th[2])),
                 torch.tensor(th[3]))
            with torch.no_grad():
                out.append(simulate(T0s, Tas, Ws, Ss, dqs, p, 2).numpy().ravel())
        return np.array(out)

    cfg = EKIConfig(n_ensemble=40, n_iterations=2 if SMOKE else 25, noise_std=0.35, init_spread=0.5, seed=0)
    eki = EnsembleKalmanInversion(G, cfg)
    eki.run(Ys.numpy().ravel().astype(np.float64), np.array([math.log(1e-3), math.log(0.15), math.log(0.02), 0.1]))
    th = eki.theta
    eki_p = np.column_stack([np.exp(th[:, 0]), np.exp(th[:, 1]), np.exp(th[:, 2]), th[:, 3]])
    R["eki"] = {"mean": dict(zip(["a", "b0", "b1", "q"], eki_p.mean(0).tolist())),
                "std": dict(zip(["a", "b0", "b1", "q"], eki_p.std(0).tolist())), "n_windows": 400,
                "ensemble": 40, "iterations": cfg.n_iterations}
    log("  EKI:", R["eki"])

    # Phase 4 -- data-driven baselines
    log("== Baselines")
    sk, st, tuning = fit_baselines(wtr, wva)
    R["baseline_tuning"] = tuning

    # Phase 6 -- PINN, lambda selected on validation
    log("== PINN lambda sweep")
    sweep = {}
    pinns = {}
    for lam in (0.0, 0.1, 1.0, 10.0):
        m, hist = train_pinn(wtr, wva, lam, init=None, seed=0, epochs=EP)
        pv = m.predict(wva)
        sweep[lam] = {"val_rmse": float(np.sqrt(((pv - wva.Y) ** 2).mean())),
                      "val_residual": m.continuous_residual(wva) if lam > 0 else None,
                      "params": [float(x) for x in m.model.params()], "history": hist,
                      "train_s": m.model.train_seconds}
        pinns[lam] = m
    lam_best = min((l for l in sweep if l > 0), key=lambda l: sweep[l]["val_rmse"])
    R["pinn_sweep"] = {str(k): v for k, v in sweep.items()}
    R["pinn_lambda_selected"] = lam_best
    log("  selected lambda:", lam_best)

    # seed robustness for the selected PINN and for the no-physics twin
    seed_runs = {"pinn": [pinns[lam_best]], "nn": [pinns[0.0]]}
    for s in (1, 2):
        seed_runs["pinn"].append(train_pinn(wtr, wva, lam_best, init=None, seed=s, epochs=EP)[0])
        seed_runs["nn"].append(train_pinn(wtr, wva, 0.0, init=None, seed=s, epochs=EP)[0])

    # ---------------- test evaluation (single pass) ----------------
    log("== Test evaluation")
    models = {"Persistence": Persistence(), "Physics ODE (identified)": OdeWrap(pm)}
    for k, m in sk.items():
        models[k] = SkWrap(m, st, k)
    models["NN (no physics loss)"] = pinns[0.0]
    models["PINN (PINNeAPPle)"] = pinns[lam_best]
    p_ode = pm.ptuple()
    R["test"] = {}
    preds_te = {}
    r_obs = discrete_physics_residual(wte.Y, wte, p_ode)
    R["observed_physics_residual"] = {"mean_abs_K_per_h": float(np.abs(r_obs).mean())}
    for name, m in models.items():
        t0 = time.time()
        p = m.predict(wte)
        dt_inf = (time.time() - t0) / len(wte) * 1000
        preds_te[name] = p
        rr = discrete_physics_residual(p, wte, p_ode)
        R["test"][name] = {"metrics": metrics(p, wte), "inference_ms_per_1000": dt_inf * 1000,
                           "physics_residual_discrete_mean_abs": float(np.abs(rr).mean()),
                           "rmse_h6_ci95": block_bootstrap_rmse(p[:, 5] - wte.Y[:, 5], wte.start)}
        if isinstance(m, PinnWrap) and m.lam > 0:
            R["test"][name]["physics_residual_continuous"] = m.continuous_residual(wte)
        log(f"  {name}: {R['test'][name]['metrics']['h1']['RMSE']:.3f} / {R['test'][name]['metrics']['h3']['RMSE']:.3f}"
            f" / {R['test'][name]['metrics']['h6']['RMSE']:.3f}  resid={R['test'][name]['physics_residual_discrete_mean_abs']:.3f}")
    R["train_seconds"] = {"PINN (PINNeAPPle)": pinns[lam_best].model.train_seconds,
                          "NN (no physics loss)": pinns[0.0].model.train_seconds,
                          "Physics ODE (identified)": R["physics_ode"]["fit_s"],
                          **{k: sum(r["fit_s"] for r in v) for k, v in tuning.items()}}
    R["seed_spread"] = {}
    for key, lst in seed_runs.items():
        rm = [float(np.sqrt(((m.predict(wte)[:, 5] - wte.Y[:, 5]) ** 2).mean())) for m in lst]
        R["seed_spread"][key] = {"rmse_h6_test": rm, "mean": float(np.mean(rm)), "std": float(np.std(rm))}
        if key == "pinn":
            R["seed_spread"][key]["params"] = [[float(x) for x in m.model.params()] for m in lst]
    # overfitting check: train vs val vs test RMSE (all horizons)
    R["generalization"] = {}
    for name in ("Gradient Boosting", "Random Forest", "MLP", "PINN (PINNeAPPle)", "NN (no physics loss)",
                 "Physics ODE (identified)"):
        m = models[name]
        sub_tr = wtr.subset(np.arange(0, len(wtr), 7))
        R["generalization"][name] = {s: float(np.sqrt(((m.predict(ww) - ww.Y) ** 2).mean()))
                                     for s, ww in (("train", sub_tr), ("val", wva), ("test", wte))}
    np.savez_compressed(RES / "test_predictions.npz", start=wte.start, Y=wte.Y, T0=wte.T0,
                        **{k.replace(" ", "_").replace("(", "").replace(")", "").replace(",", ""): v
                           for k, v in preds_te.items()})

    # ---------------- extrapolation: radiation regime ----------------
    log("== Extrapolation (radiation regime)")
    smax_tr, smax_va, smax_te = wtr.S.max(1), wva.S.max(1), wte.S.max(1)
    thr_tr, thr_te = 750.0, 900.0
    wtr_lo, wva_lo = wtr.subset(smax_tr <= thr_tr), wva.subset(smax_va <= thr_tr)
    wte_hi = wte.subset(smax_te > thr_te)
    pm_lo = PhysicsModel().fit(wtr_lo, iters=10 if SMOKE else 200)
    sk_lo, st_lo, _ = fit_baselines(wtr_lo, wva_lo)
    pinn_lo = train_pinn(wtr_lo, wva_lo, lam_best, init=None, seed=0, epochs=EP, tag="-extrap")[0]
    nn_lo = train_pinn(wtr_lo, wva_lo, 0.0, init=None, seed=0, epochs=EP, tag="-extrap")[0]
    ex_models = {"Persistence": Persistence(), "Physics ODE (identified)": OdeWrap(pm_lo),
                 "Linear (Ridge)": SkWrap(sk_lo["Linear (Ridge)"], st_lo, ""),
                 "Random Forest": SkWrap(sk_lo["Random Forest"], st_lo, ""),
                 "Gradient Boosting": SkWrap(sk_lo["Gradient Boosting"], st_lo, ""),
                 "MLP": SkWrap(sk_lo["MLP"], st_lo, ""),
                 "NN (no physics loss)": nn_lo, "PINN (PINNeAPPle)": pinn_lo}
    R["extrapolation"] = {"train_rule": f"max S in window <= {thr_tr} W/m2", "test_rule": f"max S in window > {thr_te} W/m2",
                          "n_train": len(wtr_lo), "n_test": len(wte_hi), "params_ode": pm_lo.params.__dict__,
                          "params_pinn": [float(x) for x in pinn_lo.model.params()], "results": {}}
    for name, m in ex_models.items():
        R["extrapolation"]["results"][name] = metrics(m.predict(wte_hi), wte_hi)
        log(f"  {name}: h6 RMSE={R['extrapolation']['results'][name]['h6']['RMSE']:.3f}")

    # ---------------- what-if ----------------
    log("== What-if")
    rng = np.random.default_rng(7)
    morning = np.where((wte.hour >= 8) & (wte.hour <= 10))[0]
    wbase = wte.subset(rng.choice(morning, min(600, len(morning)), replace=False))
    dq50 = 50 * 3600 / C_ASSUMED

    def scen(w, S=1.0, Wf=1.0, dTa=0.0, dq=0.0):
        return Windows(T0=w.T0, Ta=(w.Ta + dTa).astype(np.float32), W=(w.W * Wf).astype(np.float32),
                       S=np.clip(w.S * S, 0, 1300).astype(np.float32), RH=w.RH, hour=w.hour, Y=w.Y, start=w.start,
                       dq=np.full(len(w), dq, np.float32))

    scenarios = {"Solar x1.4": dict(S=1.4), "Wind x0.35 (e.g. 14->5 km/h)": dict(Wf=0.35),
                 "Ambient +1 K": dict(dTa=1.0), "Internal heat +50 W/m2": dict(dq=dq50),
                 "Combined (all four)": dict(S=1.4, Wf=0.35, dTa=1.0, dq=dq50)}
    wi_models = {k: models[k] for k in ("Physics ODE (identified)", "PINN (PINNeAPPle)", "NN (no physics loss)",
                                        "Gradient Boosting", "Random Forest", "MLP", "Linear (Ridge)")}
    base_pred = {k: m.predict(wbase) for k, m in wi_models.items()}
    R["whatif"] = {"n_states": len(wbase), "reference": "identified physics ODE (hypothesis, not observed data)",
                   "scenarios": {}}
    for sname, kw in scenarios.items():
        ws = scen(wbase, **kw)
        ref = wi_models["Physics ODE (identified)"].predict(ws)[:, 5] - base_pred["Physics ODE (identified)"][:, 5]
        row = {}
        for k, m in wi_models.items():
            dT = m.predict(ws)[:, 5] - base_pred[k][:, 5]
            row[k] = {"mean_dT6": float(dT.mean()), "std_dT6": float(dT.std()),
                      "mae_vs_physics": float(np.abs(dT - ref).mean()),
                      "sign_agreement": float((np.sign(dT) == np.sign(ref)).mean())}
            if isinstance(m, PinnWrap) and m.lam > 0:
                row[k]["physics_residual"] = m.continuous_residual(ws)
        R["whatif"]["scenarios"][sname] = row
        log(f"  {sname}: ODE {row['Physics ODE (identified)']['mean_dT6']:+.3f}  PINN {row['PINN (PINNeAPPle)']['mean_dT6']:+.3f}"
            f"  NN {row['NN (no physics loss)']['mean_dT6']:+.3f}  GB {row['Gradient Boosting']['mean_dT6']:+.3f}")

    # ---------------- anomaly detection ----------------
    log("== Anomaly detection")
    R["anomaly"] = anomaly_experiment(d, te, va, models["PINN (PINNeAPPle)"], wva)

    # ---------------- VeriPhysics confidence ----------------
    log("== VeriPhysics")
    R["veriphysics"] = veriphysics_block(models, wva, wte, pm)

    # save artifacts for the app
    torch.save({"state": pinns[lam_best].model.state_dict(), "mu": pinns[lam_best].st.mu,
                "sd": pinns[lam_best].st.sd, "ctx_dim": wtr.context().shape[1], "lam": lam_best},
               RES / "pinn_twin.pt")
    (RES / "physics_params.json").write_text(json.dumps(pm.params.__dict__, indent=2))
    R["runtime_total_s"] = time.time() - t_start
    (RES / "results.json").write_text(json.dumps(R, indent=1, default=float))
    (RES / "log.txt").write_text("\n".join(LOG))
    log("done in", R["runtime_total_s"])


def anomaly_experiment(d, te_idx, va_idx, twin: PinnWrap, wva: Windows):
    from pinneapple_systems.digital_twin.monitoring import AnomalyMonitor, ThresholdDetector

    T = d["soil_temperature_0_to_7cm"].to_numpy(np.float32)

    def residuals(Tseries, idx_targets):
        """1-h and 6-h residuals obs - twin prediction for target hours idx_targets."""
        dd = d.copy()
        dd["soil_temperature_0_to_7cm"] = Tseries
        w1 = make_windows(dd, idx_targets - 1)
        w6 = make_windows(dd, idx_targets - 6)
        r1 = Tseries[w1.start + 1] - twin.predict(w1)[:, 0]
        r6 = Tseries[w6.start + 6] - twin.predict(w6)[:, 5]
        return r1, r6

    va_t = va_idx[(va_idx >= 6)]
    va_t = va_t[va_t + H < len(d)]
    r1v, r6v = residuals(T, va_t)
    thr = {"warn_1h": float(np.quantile(np.abs(r1v), 0.99)), "anom_1h": float(np.quantile(np.abs(r1v), 0.999)),
           "warn_6h": float(np.quantile(np.abs(r6v), 0.99)), "anom_6h": float(np.quantile(np.abs(r6v), 0.999))}
    # naive detector: temperature outside the training-period 0.5-99.5% band
    lo, hi = np.quantile(T[: va_idx.min()], [0.005, 0.995])
    thr["naive_lo"], thr["naive_hi"] = float(lo), float(hi)

    tgt = te_idx[(te_idx >= te_idx.min() + 6)]
    tgt = tgt[tgt + H < len(d)]
    rng = np.random.default_rng(11)
    Tc = T.copy()
    label = np.zeros(len(T), int)  # 0 clean, 1 spike, 2 drift, 3 stuck
    events = []
    busy = np.zeros(len(T), bool)
    kinds = [("spike", 1, 1)] * 150 + [("drift", 2, 24)] * 30 + [("stuck", 3, 12)] * 30
    for kind, code, dur in kinds:
        for _ in range(1000):
            s = int(rng.choice(tgt[:-40]))
            if not busy[s - 12: s + dur + 12].any():
                break
        busy[s - 12: s + dur + 12] = True
        if kind == "spike":
            Tc[s] += rng.choice([-1, 1]) * rng.uniform(2, 6)
        elif kind == "drift":
            Tc[s: s + dur] += np.linspace(0, rng.uniform(2, 4), dur)
        else:
            Tc[s: s + dur] = Tc[s - 1]
        label[s: s + dur] = code
        events.append((kind, s, dur))
    r1, r6 = residuals(Tc, tgt)
    mon = AnomalyMonitor()
    warn = ThresholdDetector({"r1": thr["warn_1h"], "r6": thr["warn_6h"]})
    anom = ThresholdDetector({"r1": thr["anom_1h"], "r6": thr["anom_6h"]})
    mon.add_detector(warn)
    status = np.zeros(len(T), int)  # 0 NORMAL 1 WARNING 2 ANOMALY
    for i, t in enumerate(tgt):
        obs = {"r1": float(r1[i]), "r6": float(r6[i])}
        zero = {"r1": 0.0, "r6": 0.0}
        if anom.check(float(t), "T_facility", obs, zero):
            status[t] = 2
        elif warn.check(float(t), "T_facility", obs, zero):
            status[t] = 1
    naive = np.zeros(len(T), bool)
    naive[tgt] = (Tc[tgt] < lo) | (Tc[tgt] > hi)

    clean = np.isin(np.arange(len(T)), tgt) & (label == 0) & ~busy
    out = {"thresholds_K": thr, "n_clean_hours": int(clean.sum()),
           "false_alarm_rate": {"twin_WARNING_or_ANOMALY": float((status[clean] >= 1).mean()),
                                "twin_ANOMALY": float((status[clean] == 2).mean()),
                                "naive_band": float(naive[clean].mean())},
           "by_type": {}}
    for kind in ("spike", "drift", "stuck"):
        ev = [e for e in events if e[0] == kind]
        det_tw, det_an, det_nv, delays = 0, 0, 0, []
        for _, s, dur in ev:
            span = np.arange(s, s + dur + 6)  # allow detection up to 6 h after the fault window
            hit = np.where(status[span] >= 1)[0]
            det_tw += len(hit) > 0
            det_an += (status[span] == 2).any()
            det_nv += naive[span].any()
            if len(hit):
                delays.append(int(hit[0]))
        out["by_type"][kind] = {"n": len(ev), "recall_twin_warn_or_anom": det_tw / len(ev),
                                "recall_twin_anomaly": det_an / len(ev), "recall_naive": det_nv / len(ev),
                                "median_delay_h": float(np.median(delays)) if delays else None}
    # example segment for the paper figure
    k = [e for e in events if e[0] == "drift"][0][1]
    seg = np.arange(k - 48, k + 48)
    out["example"] = {"t": seg.tolist(), "T_clean": T[seg].tolist(), "T_corrupt": Tc[seg].tolist(),
                      "status": status[seg].tolist()}
    tpos = np.isin(np.arange(len(T)), tgt)
    np.savez_compressed(RES / "anomaly_series.npz", tgt=tgt, r1=r1, r6=r6, status=status[tgt], label=label[tgt],
                        Tc=Tc[tgt], T=T[tgt])
    log("  anomaly:", json.dumps(out["by_type"]), out["false_alarm_rate"])
    return out


def veriphysics_block(models, wva, wte, pm: PhysicsModel):
    from pinneapple_analysis.uncertainty.calibration import CalibrationMetrics
    from pinneapple_analysis.verification.convergence import richardson_extrapolate
    from pinneapple_analysis.verification.physics_confidence_score import CalibrationSummary, compute_physics_confidence
    from pinneapple_data.physics_case import PhysicsCase
    from pinneapple_pdb.benchmarks import BenchmarkEntry, register_benchmark

    # held-out benchmark: observed 6-h temperature change on the test period
    xs = np.column_stack([wte.start, np.full(len(wte), 6)]).astype(np.float32)
    ys = (wte.Y[:, 5] - wte.T0).astype(np.float32)[:, None]

    @register_benchmark("natal_thermal_mass_dT6h_2023_2024")
    def _bench():
        return BenchmarkEntry(name="natal_thermal_mass_dT6h_2023_2024",
                              description="Observed 6-h change of soil_temperature_0_to_7cm, Natal, 2023-2024 (held-out test)",
                              reference_source="Open-Meteo Historical Weather API (ERA5-Land)",
                              x_vars=("window_start", "horizon_h"), y_vars=("dT",), reference_x=xs, reference_y=ys)

    # numerical convergence of the physics simulator used by the what-if engine (RK4 sub-steps 1,2,4)
    T0, Ta, W, S, dq, _ = to_t(wte.subset(np.arange(0, len(wte), 10)))
    with torch.no_grad():
        f = [simulate(T0, Ta, W, S, dq, pm.ptuple(), n)[:, 5].numpy() for n in (1, 2, 4)]
    conv = richardson_extrapolate(f[0], f[1], f[2], r=2.0)

    out = {"convergence": {"observed_order": conv.observed_order, "gci_fine": conv.gci_fine,
                           "asymptotic_ratio": conv.asymptotic_ratio, "is_asymptotic": conv.is_asymptotic}}
    for name in ("PINN (PINNeAPPle)", "Physics ODE (identified)", "Gradient Boosting", "NN (no physics loss)"):
        m = models[name]
        pv, pt = m.predict(wva), m.predict(wte)
        sd = (pv - wva.Y).std(0)  # per-horizon Gaussian predictive std estimated on validation only
        yt = torch.tensor(wte.Y[:, 5]); yp = torch.tensor(pt[:, 5]); ys_ = torch.full_like(yp, float(sd[5]))
        ece = CalibrationMetrics.expected_calibration_error(yp, yt, ys_)
        cov = CalibrationMetrics.coverage_at_level(yp, yt, ys_, alpha=0.1)
        case = PhysicsCase(name=name, results={"window_start": xs[:, 0], "horizon_h": xs[:, 1],
                                               "dT": pt[:, 5] - wte.T0}, reference_benchmark="natal_thermal_mass_dT6h_2023_2024")
        bench = case.validate_against_benchmark()
        score = compute_physics_confidence(calibration_metrics=CalibrationSummary(ece=ece, coverage=cov, target_coverage=0.9,
                                                                                sharpness=float(sd[5])),
                                           benchmark_comparison=bench,
                                           convergence_result=conv if name in ("PINN (PINNeAPPle)", "Physics ODE (identified)") else None)
        out[name] = {"overall_score": score.overall_score, "coverage": score.coverage,
                     "components": [c.__dict__ for c in score.components], "summary": score.summary(),
                     "ece": ece, "coverage_90": cov, "sigma_h6": float(sd[5]),
                     "rel_l2_dT6": bench.relative_l2_error}
        log(score.summary())
    out["not_run"] = {"physics_guardrail": "PhysicsGuardrail requires a compiled PDE ProblemSpec; this ODE is forced by "
                                           "external time series, which ProblemSpec cannot express. The physics residual is "
                                           "reported separately on fresh collocation points.",
                      "geometry_ood": "no geometry input in this problem"}
    return out


if __name__ == "__main__":
    main()
