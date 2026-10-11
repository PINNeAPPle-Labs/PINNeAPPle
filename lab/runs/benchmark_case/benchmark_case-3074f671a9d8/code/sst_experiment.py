"""Satellite SST image forecasting with PINNeAPPle neural operators.

Data: NOAA OISST v2.1 daily SST (AVHRR-based), Brazil-Malvinas Confluence, 64x64 @ 0.25 deg.
Task: given the last 5 daily SST images, forecast the next 7 daily images (leads 1..7).

Leakage control
  * chronological split: train 1982-2012, validation 2013-2017, test 2018-2024;
  * the climatology and every normalisation constant come from the training years only;
  * a sample never straddles a split boundary (inputs and targets inside the same period);
  * hyper-parameters and early stopping use validation only; test is evaluated once.

Usage: python sst_experiment.py [hpo|final|eval|all]
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[2] / "PINNeAPPle"))
from pinneapple_neural.architectures.neural_operators.fno import FNO2d  # noqa: E402
from pinneapple_neural.architectures.convolutions.conv2d import Conv2DModel  # noqa: E402

RES = ROOT / ("results_smoke" if os.environ.get("SMOKE") == "1" else "results")
RES.mkdir(exist_ok=True)
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
LAGS, LEADS = 5, 7
SPLITS = {"train": (1982, 2012), "val": (2013, 2017), "test": (2018, 2024)}
SMOKE = os.environ.get("SMOKE") == "1"


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(RES / "log.txt", "a") as fh:
        fh.write(s + "\n")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

class SSTData:
    def __init__(self):
        z = np.load(ROOT / "data" / "oisst_bmc_1982_2024.npz")
        self.sst = z["sst"].astype(np.float32)
        self.time = z["time"].astype("datetime64[D]")
        self.lat, self.lon = z["lat"], z["lon"]
        self.ocean = ~np.isnan(self.sst).all(0)
        self.year = self.time.astype("datetime64[Y]").astype(int) + 1970
        self.doy = (self.time - self.time.astype("datetime64[Y]")).astype(int)
        tr = (self.year >= SPLITS["train"][0]) & (self.year <= SPLITS["train"][1])
        # climatology: 3 annual harmonics per pixel, fitted on training years only
        w = 2 * np.pi * self.doy / 365.25
        B = np.column_stack([np.ones_like(w)] + [f(k * w) for k in (1, 2, 3) for f in (np.sin, np.cos)])
        X = np.nan_to_num(self.sst.reshape(len(self.sst), -1))
        coef, *_ = np.linalg.lstsq(B[tr], X[tr], rcond=None)
        self.clim = (B @ coef).reshape(self.sst.shape).astype(np.float32)
        anom = np.where(self.ocean, np.nan_to_num(self.sst) - self.clim, 0.0).astype(np.float32)
        self.scale = float(anom[tr][:, self.ocean].std())
        self.anom = anom / self.scale                       # dimensionless anomaly, 0 on land
        self.sin, self.cos = np.sin(w).astype(np.float32), np.cos(w).astype(np.float32)

    def indices(self, split):
        """Valid forecast origins t (last input day) fully inside `split`."""
        y0, y1 = SPLITS[split]
        inside = (self.year >= y0) & (self.year <= y1)
        t = np.arange(LAGS - 1, len(self.sst) - LEADS)
        ok = inside[t - LAGS + 1] & inside[t + LEADS]
        return t[ok]


class Batcher:
    """Builds (input, target) tensors on the device from origin indices."""

    def __init__(self, D: SSTData):
        self.A = torch.tensor(D.anom, device=DEV)
        self.mask = torch.tensor(D.ocean, device=DEV, dtype=torch.float32)
        self.sin = torch.tensor(D.sin, device=DEV)
        self.cos = torch.tensor(D.cos, device=DEV)
        self.lag = torch.arange(-LAGS + 1, 1, device=DEV)
        self.lead = torch.arange(1, LEADS + 1, device=DEV)

    def __call__(self, t):
        t = torch.as_tensor(t, device=DEV)
        x = self.A[t[:, None] + self.lag]                     # (B, LAGS, H, W)
        y = self.A[t[:, None] + self.lead]                    # (B, LEADS, H, W)
        B, _, Hh, Ww = x.shape
        extra = torch.stack([self.mask.expand(B, Hh, Ww),
                             self.sin[t][:, None, None].expand(B, Hh, Ww),
                             self.cos[t][:, None, None].expand(B, Hh, Ww)], 1)
        return torch.cat([x, extra], 1), y


# ---------------------------------------------------------------------------
# models (PINNeAPPle architectures) -- residual formulation: predict a(t+k) - a(t)
# ---------------------------------------------------------------------------

class Forecaster(nn.Module):
    def __init__(self, kind, **hp):
        super().__init__()
        cin = LAGS + 3
        if kind == "fno":
            self.core = FNO2d(cin, LEADS, width=hp["width"], modes1=hp["modes"], modes2=hp["modes"],
                              layers=hp["layers"], use_grid=True)
        elif kind == "cnn":
            self.core = Conv2DModel(cin, LEADS, hidden_channels=hp["hidden"], num_blocks=hp["blocks"],
                                    kernel_size=3, dropout=hp.get("dropout", 0.0))
        else:
            raise ValueError(kind)
        self.kind = kind

    def forward(self, x):
        out = self.core(x)
        out = out.y if hasattr(out, "y") else out
        return x[:, LAGS - 1: LAGS] + out  # last observed anomaly + learned increment


def masked_mse(p, y, m):
    return (((p - y) ** 2) * m).sum() / (m.sum() * p.shape[0] * p.shape[1])


def train_model(kind, hp, D, bat, seed=0, epochs=30, patience=5, bs=32, lr=1e-3, tag=""):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr, va = D.indices("train"), D.indices("val")
    if SMOKE:
        tr, va, epochs = tr[:640], va[:320], 1
    model = Forecaster(kind, **hp).to(DEV)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    m = bat.mask
    best, best_state, bad, hist = 1e9, None, 0, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(tr)
        tl, n = 0.0, 0
        for i in range(0, len(perm), bs):
            x, y = bat(perm[i:i + bs])
            loss = masked_mse(model(x), y, m)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += float(loss) * len(x)
            n += len(x)
        sched.step()
        vl = evaluate_loss(model, va, bat)
        hist.append({"epoch": ep + 1, "train_mse": tl / n, "val_mse": vl, "t": time.time() - t0})
        log(f"  [{kind}{tag} {hp} s{seed}] ep {ep + 1}: train {tl / n:.4f}  val {vl:.4f}  ({time.time() - t0:.0f}s)")
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                log("  early stop")
                break
    model.load_state_dict(best_state)
    return model, {"best_val_mse": best, "history": hist, "n_params": n_par, "train_s": time.time() - t0,
                   "epochs_run": len(hist)}


@torch.no_grad()
def evaluate_loss(model, idx, bat, bs=64):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(idx), bs):
        x, y = bat(idx[i:i + bs])
        tot += float(masked_mse(model(x), y, bat.mask)) * len(x)
        n += len(x)
    return tot / n


@torch.no_grad()
def predict(model, idx, bat, bs=64):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        x, _ = bat(idx[i:i + bs])
        out.append(model(x).float().cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------

def baseline_preds(D: SSTData, idx, rho=None):
    """Returns dict name -> (N, LEADS, H, W) anomaly forecasts (normalised units)."""
    a_t = D.anom[idx]
    out = {"Persistence": np.repeat(a_t[:, None], LEADS, 1),
           "Climatology": np.zeros((len(idx), LEADS) + a_t.shape[1:], np.float32)}
    if rho is not None:
        out["Damped persistence (AR1)"] = rho[None] * a_t[:, None]
    return out


def fit_damped_persistence(D: SSTData):
    """Per-pixel, per-lead least-squares coefficient a(t+k) ~ rho_k a(t), on training years."""
    tr = D.indices("train")
    a0 = D.anom[tr]
    rho = np.zeros((LEADS,) + a0.shape[1:], np.float32)
    for k in range(1, LEADS + 1):
        ak = D.anom[tr + k]
        rho[k - 1] = (a0 * ak).sum(0) / np.maximum((a0 * a0).sum(0), 1e-6)
    return rho


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def radial_spectrum(f):
    """Radially averaged 2D power spectrum of fields (N, H, W) -> (H//2,)."""
    F = np.abs(np.fft.fftshift(np.fft.fft2(f), axes=(-2, -1))) ** 2
    H, W = f.shape[-2:]
    yy, xx = np.indices((H, W))
    r = np.hypot(yy - H // 2, xx - W // 2).astype(int)
    P = F.mean(0)
    return np.array([P[r == k].mean() for k in range(H // 2)])


def score(pred, D: SSTData, idx):
    """pred: normalised anomaly forecasts (N, L, H, W). Metrics in degC over ocean pixels."""
    y = np.stack([D.anom[idx + k] for k in range(1, LEADS + 1)], 1)
    oc = D.ocean
    e = (pred - y)[..., oc] * D.scale            # (N, L, P)
    yo, po = y[..., oc] * D.scale, pred[..., oc] * D.scale
    out = {"rmse": np.sqrt((e ** 2).mean((0, 2))).tolist(), "mae": np.abs(e).mean((0, 2)).tolist(),
           "bias": e.mean((0, 2)).tolist()}
    # anomaly correlation coefficient (spatial, per forecast, then averaged)
    yc = yo - yo.mean(-1, keepdims=True)
    pc = po - po.mean(-1, keepdims=True)
    acc = (yc * pc).sum(-1) / np.sqrt((yc ** 2).sum(-1) * (pc ** 2).sum(-1) + 1e-12)
    out["acc"] = acc.mean(0).tolist()
    # monthly block bootstrap CI for RMSE at leads 1 and 7
    months = D.time[idx].astype("datetime64[M]")
    groups = [np.where(months == m)[0] for m in np.unique(months)]
    rng = np.random.default_rng(0)
    for L in (0, LEADS - 1):
        v = []
        se = (e[:, L] ** 2).mean(-1)
        for _ in range(1000):
            g = rng.integers(0, len(groups), len(groups))
            v.append(np.sqrt(se[np.concatenate([groups[i] for i in g])].mean()))
        out[f"rmse_ci95_lead{L + 1}"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    # spectral fidelity at lead 7 (blurring diagnostic): ratio of predicted/observed power per wavenumber
    ps, ys = radial_spectrum(pred[:, -1]), radial_spectrum(y[:, -1])
    out["spectral_ratio_lead7"] = (ps / ys).tolist()
    return out


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

CANDIDATES = [
    ("fno", {"width": 32, "modes": 12, "layers": 4}),
    ("fno", {"width": 48, "modes": 20, "layers": 4}),
    ("cnn", {"hidden": 48, "blocks": 6}),
    ("cnn", {"hidden": 64, "blocks": 8, "dropout": 0.1}),
]


def stage_hpo(D, bat):
    rows = []
    for kind, hp in CANDIDATES:
        _, info = train_model(kind, hp, D, bat, seed=0, epochs=25, patience=4, tag="-hpo")
        rows.append({"kind": kind, "hp": hp, **{k: v for k, v in info.items()}})
        (RES / "hpo.json").write_text(json.dumps(rows, indent=1))
    return rows


def stage_final(D, bat):
    rows = json.loads((RES / "hpo.json").read_text())
    final = {}
    for kind in ("fno", "cnn"):
        best = min((r for r in rows if r["kind"] == kind), key=lambda r: r["best_val_mse"])
        for seed in (0, 1, 2):
            model, info = train_model(kind, best["hp"], D, bat, seed=seed, epochs=40, patience=6, tag="-final")
            torch.save(model.state_dict(), RES / f"{kind}_seed{seed}.pt")
            final[f"{kind}_seed{seed}"] = {"hp": best["hp"], **info}
            (RES / "final_training.json").write_text(json.dumps(final, indent=1))
    return final


def load_models():
    final = json.loads((RES / "final_training.json").read_text())
    models = {}
    for key, info in final.items():
        kind = key.split("_")[0]
        m = Forecaster(kind, **info["hp"]).to(DEV)
        m.load_state_dict(torch.load(RES / f"{key}.pt", map_location=DEV))
        models[key] = m
    return models, final


def stage_eval(D, bat):
    from pinneapple_analysis.uncertainty.calibration import CalibrationMetrics
    from pinneapple_analysis.uncertainty.conformal import ConformalPredictor
    from pinneapple_analysis.verification.physics_confidence_score import CalibrationSummary, compute_physics_confidence
    from pinneapple_data.physics_case import PhysicsCase
    from pinneapple_pdb.benchmarks import BenchmarkEntry, register_benchmark

    models, final = load_models()
    va, te = D.indices("val"), D.indices("test")
    tr_sub = D.indices("train")[::10]
    rho = fit_damped_persistence(D)
    R = {"data": {"n_days": int(len(D.sst)), "start": str(D.time[0]), "end": str(D.time[-1]),
                  "grid": [64, 64], "ocean_fraction": float(D.ocean.mean()), "anomaly_scale_degC": D.scale,
                  "n_origins": {s: int(len(D.indices(s))) for s in SPLITS}},
         "training": final, "hpo": json.loads((RES / "hpo.json").read_text())}

    def ens(kind, idx):
        ps = [predict(m, idx, bat) for k, m in models.items() if k.startswith(kind)]
        return np.mean(ps, 0), ps

    preds = {"test": {}, "val": {}, "train": {}}
    for split, idx in (("test", te), ("val", va), ("train", tr_sub)):
        preds[split].update(baseline_preds(D, idx, rho))
        for kind, label in (("fno", "FNO2d (PINNeAPPle)"), ("cnn", "ResNet-CNN (PINNeAPPle Conv2DModel)")):
            mean, members = ens(kind, idx)
            preds[split][label + " ensemble"] = mean
            if split == "test":
                R.setdefault("seed_members", {})[label] = [score(p, D, idx)["rmse"] for p in members]

    R["scores"] = {s: {k: score(p, D, idx) for k, p in preds[s].items()}
                   for s, idx in (("test", te), ("val", va), ("train", tr_sub))}
    for k in R["scores"]["test"]:
        log(f"  TEST {k}: rmse L1 {R['scores']['test'][k]['rmse'][0]:.3f} L7 {R['scores']['test'][k]['rmse'][-1]:.3f}"
            f"  acc L7 {R['scores']['test'][k]['acc'][-1]:.3f}")
    # seasonal breakdown (test, lead 7)
    mon = D.time[te].astype("datetime64[M]").astype(int) % 12 + 1
    seas = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
    y7 = D.anom[te + LEADS][:, D.ocean] * D.scale
    R["seasonal_rmse_lead7"] = {name: {s: float(np.sqrt((((p[:, -1][:, D.ocean] * D.scale) - y7)[np.isin(mon, ms)] ** 2).mean()))
                                       for s, ms in seas.items()} for name, p in preds["test"].items()}
    best_name = min((k for k in preds["test"] if "ensemble" in k),
                    key=lambda k: np.mean(R["scores"]["val"][k]["rmse"]))
    R["selected_model_on_val"] = best_name

    # ---- uncertainty: per-pixel, per-lead split-conformal intervals calibrated on validation ----
    def resid(split, idx):
        y = np.stack([D.anom[idx + k] for k in range(1, LEADS + 1)], 1)
        return preds[split][best_name], y
    pv, yv = resid("val", va)
    pt, yt = resid("test", te)
    oc = D.ocean
    unc = {}
    for L in (0, 2, LEADS - 1):
        xc = torch.tensor(pv[:, L][:, oc].reshape(-1, 1) * D.scale)
        yc = torch.tensor(yv[:, L][:, oc].reshape(-1, 1) * D.scale)
        cp = ConformalPredictor(lambda x: x, alpha=0.1)   # model = the already computed forecast
        cp.calibrate(xc, yc)
        xt = torch.tensor(pt[:, L][:, oc].reshape(-1, 1) * D.scale)
        ytt = torch.tensor(yt[:, L][:, oc].reshape(-1, 1) * D.scale)
        cov = float(cp.coverage(xt, ytt))
        sd = torch.tensor(((pv[:, L] - yv[:, L])[:, oc].std(0) * D.scale))  # per-pixel sigma from validation
        sdt = sd[None].expand(len(te), -1).reshape(-1, 1)
        ece = CalibrationMetrics.expected_calibration_error(xt, ytt, sdt)
        cov_g = CalibrationMetrics.coverage_at_level(xt, ytt, sdt, alpha=0.1)
        unc[f"lead{L + 1}"] = {"conformal_halfwidth_degC": float(cp.quantile), "conformal_coverage_test": cov,
                               "gaussian_ece": ece, "gaussian_coverage90": cov_g, "mean_sigma_degC": float(sd.mean())}
        log(f"  UQ lead {L + 1}: {unc[f'lead{L + 1}']}")
    R["uncertainty"] = unc

    # ---- VeriPhysics: benchmark agreement on held-out test + calibration ----
    sub = te[::5]
    js = np.searchsorted(te, sub)
    oc_idx = np.flatnonzero(oc)
    leads_used = (0, 2, 6)
    xs = np.array([(t, L + 1, p) for t in sub for L in leads_used for p in oc_idx], np.float32)
    ys = np.array([D.anom[t + L + 1].ravel()[oc_idx] for t in sub for L in leads_used], np.float32).reshape(-1, 1) * D.scale

    @register_benchmark("oisst_bmc_test_2018_2024")
    def _b():
        return BenchmarkEntry(name="oisst_bmc_test_2018_2024",
                              description="NOAA OISST v2.1 SST anomalies, Brazil-Malvinas Confluence, held-out 2018-2024",
                              reference_source="NOAA PSL OISST v2.1 (psl.noaa.gov)", x_vars=("origin", "lead", "pixel"),
                              y_vars=("sst_anom",), reference_x=xs, reference_y=ys)

    vp = {}
    for name in list(preds["test"].keys()):
        p = preds["test"][name]
        yp = np.array([p[j, L].ravel()[oc_idx] for j in js for L in leads_used], np.float32).reshape(-1) * D.scale
        case = PhysicsCase(name=name, results={"origin": xs[:, 0], "lead": xs[:, 1], "pixel": xs[:, 2], "sst_anom": yp},
                           reference_benchmark="oisst_bmc_test_2018_2024")
        bench = case.validate_against_benchmark()
        calib = None
        if name == best_name:
            u = unc[f"lead{LEADS}"]
            calib = CalibrationSummary(ece=u["gaussian_ece"], coverage=u["gaussian_coverage90"], target_coverage=0.9,
                                       sharpness=u["mean_sigma_degC"])
        sc = compute_physics_confidence(benchmark_comparison=bench, calibration_metrics=calib)
        vp[name] = {"overall_score": sc.overall_score, "coverage": sc.coverage,
                    "components": [c.__dict__ for c in sc.components], "rel_l2": bench.relative_l2_error,
                    "rmse": bench.rmse, "n_points": bench.n_compared_points}
        log(sc.summary())
    vp["not_run"] = {"physics_guardrail": "no governing PDE is imposed on a purely data-driven image forecaster",
                     "numerical_convergence": "no discretisation-refinement parameter in a learned forecaster",
                     "geometry_ood": "no geometry input"}
    R["veriphysics"] = vp
    # figures data
    k = int(np.argmin(np.abs(te - np.searchsorted(D.time, np.datetime64("2021-07-15")))))
    np.savez_compressed(RES / "figure_data.npz", t=te[k], obs=np.stack([D.sst[te[k] + j] for j in range(0, LEADS + 1)]),
                        clim=np.stack([D.clim[te[k] + j] for j in range(1, LEADS + 1)]),
                        **{n.split(" ")[0].replace("(", "").replace(")", ""): preds["test"][n][k] * D.scale
                           for n in preds["test"]}, ocean=D.ocean, lat=D.lat, lon=D.lon)
    (RES / "results.json").write_text(json.dumps(R, indent=1, default=float))
    log("eval done; selected:", best_name)


def main(stage):
    t0 = time.time()
    D = SSTData()
    bat = Batcher(D)
    log(f"== stage {stage} on {DEV}; origins:", {s: len(D.indices(s)) for s in SPLITS}, "scale", D.scale)
    if stage in ("hpo", "all"):
        stage_hpo(D, bat)
    if stage in ("final", "all"):
        stage_final(D, bat)
    if stage in ("eval", "all"):
        stage_eval(D, bat)
    log(f"stage {stage} finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
