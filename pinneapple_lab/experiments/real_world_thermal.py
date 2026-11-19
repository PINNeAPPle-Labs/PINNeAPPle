"""KIT LLEC measured thermal dynamics benchmark, no synthetic data fallback.

Includes one-step RC-regularized comparison and an optional continuous-time
lumped ODE PINN. Neither model solves a spatial PDE. Source: Zenodo 21157950.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from ..spec import Experiment, register

DOI = "10.5281/zenodo.21157950"
URL = "https://zenodo.org/records/21157950"
REQUIRED = ("time", "temperature", "outdoor", "heat_w")


def load_real(path, mapping):
    """Mapping keys must map actual temperature, air and delivered HEAT in W."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Real measurements missing: {path}; obtain from {URL}")
    if set(mapping) != set(REQUIRED):
        raise ValueError(f"Explicit mapping required for {REQUIRED}")
    data = pd.read_csv(path, low_memory=False)
    if set(mapping.values()) - set(data.columns):
        raise ValueError("Unrecognized sensor column; inspect Zenodo Metadata.json")
    a = data[[mapping[k] for k in REQUIRED]].copy()
    a.columns = list(REQUIRED)
    # KIT CSVW time is numeric hours, not necessarily an ISO timestamp.
    numeric_time = pd.to_numeric(a["time"], errors="coerce")
    if numeric_time.notna().all():
        a["time"] = numeric_time.astype(float) * 3600.0
    else:
        parsed = pd.to_datetime(a["time"], utc=True, errors="coerce")
        if parsed.isna().any():
            raise ValueError("Invalid timestamp values")
        a["time"] = parsed.astype("int64") / 1e9
    for k in REQUIRED[1:]:
        a[k] = pd.to_numeric(a[k], errors="coerce")
    if a.isna().any().any() or len(a) < 301:
        raise ValueError("Missing/non-numeric measurements or too few samples")
    t = a["time"].to_numpy(dtype=float)
    dt = np.diff(t)
    if np.any(dt <= 0) or not np.allclose(dt, np.median(dt), atol=0.25, rtol=0):
        raise ValueError("Temporal gaps, duplicate timestamps or irregular sampling")
    if not (10 <= dt[0] <= 3600):
        raise ValueError("Unexpected sampling interval")
    return a, float(np.median(dt)), hashlib.sha256(path.read_bytes()).hexdigest()


def chronological(n):
    a, b = int(n*.6), int(n*.8)
    return {"train": np.arange(a), "val": np.arange(a,b), "test": np.arange(b,n)}


def rc_baseline(x, y, train):
    """First-order Euler step, fitted on train only: ΔT = a(Te-T)+bQ+c."""
    from scipy.optimize import lsq_linear
    design = np.column_stack((x[:,1]-x[:,0], x[:,2], np.ones(len(x))))
    fit = lsq_linear(design[train], (y-x[:,0])[train],
                     bounds=([0,0,-np.inf],[.999,np.inf,np.inf]))
    if not fit.success:
        raise RuntimeError("RC calibration failed")
    return fit.x, x[:,0]+design@fit.x


def score(y, pred):
    e = np.asarray(y)-np.asarray(pred)
    return {"rmse_c": float(np.sqrt(np.mean(e*e))),
            "mae_c": float(np.mean(np.abs(e))),
            "max_abs_c": float(np.max(np.abs(e)))}


def train_network(x, y, split, prior, physics, epochs, seed, device, return_predictor=False):
    import pinneapple as pp
    torch.manual_seed(seed)
    net = pp.build_model("vanilla_pinn", in_dim=3, out_dim=1,
                         hidden=[32,32]).to(device)
    mean, std = x[split["train"]].mean(0), x[split["train"]].std(0).clip(1e-6)
    xt = torch.tensor((x-mean)/std, dtype=torch.float32, device=device)
    dy = torch.tensor((y-x[:,0])[:,None], dtype=torch.float32, device=device)
    rc = torch.tensor((prior-x[:,0])[:,None], dtype=torch.float32, device=device)
    ti = torch.tensor(split["train"], device=device)
    vi = torch.tensor(split["val"], device=device)
    def evaluate(z):
        out = net(z)
        return out.y if hasattr(out, "y") else out
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best = float("inf")
    best_state = None
    stale = 0
    hist = []
    for epoch in range(epochs):
        net.train()
        opt.zero_grad(set_to_none=True)
        estimate = evaluate(xt[ti])
        data = (estimate-dy[ti]).square().mean()
        residual = (estimate-rc[ti]).square().mean() if physics else 0
        loss = data + (0.25 * residual if physics else 0)
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            val = (evaluate(xt[vi])-dy[vi]).square().mean().item()
        hist.append([float(loss.item()), float(val)])
        if val < best - 1e-10:
            best, stale = val, 0
            best_state = {k: v.detach().clone() for k,v in net.state_dict().items()}
        else:
            stale += 1
        if stale >= 80:
            break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        result = evaluate(xt).cpu().numpy().ravel()+x[:,0]
    if return_predictor:
        def predict_external(external_x):
            values = torch.tensor((np.asarray(external_x)-mean)/std,
                                  dtype=torch.float32,device=device)
            with torch.no_grad():
                return evaluate(values).cpu().numpy().ravel()+np.asarray(external_x)[:,0]
        return result, np.asarray(hist), epoch+1, predict_external
    return result, np.asarray(hist), epoch+1


@register
class KITRealThermal(Experiment):
    name = "kit_real_thermal"
    version = "1"
    description = "KIT observed building temperature: calibrated RC vs MLP and physics-regularized PINNeAPPle network"
    tags = ["real-data", "thermal", "baseline", "physics-informed"]
    references = [URL, DOI]
    limitations = [
        "One-step prediction with observed lagged room temperature, not autonomous rollout",
        "RC regularization is discrete; optional continuous model is a lumped ODE, not a spatial PDE",
        "Requires matching delivered heating power and zone temperature, not electric input power",
        "Only chronological within-campaign split; independent campaign test pending",
        "Single seed, no uncertainty estimates or error-conservation assessment",
    ]
    params = {"csv": "", "mapping_json": "", "epochs": 400, "seed": 0, "device": "cpu", "continuous_pinn": True, "physics_weight": 1.0}

    def run(self, ctx):
        p = ctx.params
        if not p["csv"] or not p["mapping_json"]:
            raise ValueError("Pass -p csv=/data/Exp1_tot.csv -p mapping_json=/data/mapping.json")
        mapping = json.loads(Path(p["mapping_json"]).read_text(encoding="utf-8"))
        with ctx.stage("load_real_measurements"):
            a, dt, digest = load_real(p["csv"], mapping)
        ctx.input("provenance", {"doi": DOI, "sha256": digest, "columns": mapping,
                                 "n_rows": len(a), "dt_seconds": dt,
                                 "task": "one_step_ahead", "split": "chronological 60/20/20"})
        t = a["temperature"].to_numpy(dtype=float)
        ambient = a["outdoor"].to_numpy(dtype=float)
        heat = a["heat_w"].to_numpy(dtype=float) / 1000  # kW
        x = np.column_stack((t[:-1], ambient[:-1], heat[:-1]))
        y = t[1:]
        split = chronological(len(y))
        with ctx.stage("calibrated_numerical_rc"):
            coefficients, rc = rc_baseline(x, y, split["train"])
        ctx.output("rc_coefficients", dict(zip(("a_step","b_kw","c_step"), map(float, coefficients))))
        predictions = {"rc": rc}
        history = {}
        timing = {}
        for name, physics in (("data_only", False), ("rc_regularized", True)):
            with ctx.stage("train_" + name):
                start = time.perf_counter()
                pred, hist, steps = train_network(x, y, split, rc, physics,
                                                  int(p["epochs"]), int(p["seed"]), p["device"])
                timing[name] = {"seconds": time.perf_counter()-start, "steps": steps}
                predictions[name], history[name] = pred, hist
        if p.get("continuous_pinn", True):
            from .kit_continuous_thermal import train_continuous_pinn
            with ctx.stage("train_continuous_ode_pinn"):
                start = time.perf_counter()
                # Observed temperature is supervised on training indices only.
                # The full time series is used as an inference grid, not labels.
                full_split = chronological(len(t))
                continuous_pred, continuous_hist, physical = train_continuous_pinn(
                    a["time"].to_numpy(dtype=float),
                    ambient, a["heat_w"].to_numpy(dtype=float), t,
                    full_split["train"], full_split["val"],
                    seed=int(p["seed"]), epochs=int(p["epochs"]),
                    device=p["device"], physics_weight=float(p["physics_weight"]))
                # y[i] is the observation at original time i+1.
                predictions["continuous_ode_pinn"] = continuous_pred[1:]
                history["continuous_ode_pinn"] = continuous_hist[:,[0,2]]
                timing["continuous_ode_pinn"] = {
                    "seconds": time.perf_counter()-start,
                    "steps": len(continuous_hist)}
                ctx.output("continuous_ode_physical_parameters", physical)
        for model, pred in predictions.items():
            for partition, indices in split.items():
                for metric, val in score(y[indices],pred[indices]).items():
                    ctx.metric(f"{model}_{partition}_{metric}", val)
        ctx.output("training_times", timing)
        test = split["test"]
        ctx.output("test_predictions", np.column_stack([y[test]]+[pred[test] for pred in predictions.values()]))
        ctx.check("physics_model_beats_calibrated_solver",
                  passed=score(y[test],predictions["rc_regularized"][test])["rmse_c"] <
                         score(y[test],rc[test])["rmse_c"],
                  kind="baseline", detail="Held-out RMSE comparison, failed checks are reported")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11,4))
        ax.plot(y[test], label="measured", lw=1)
        for name, pred in predictions.items():
            ax.plot(pred[test], label=name, alpha=.75)
        ax.set(xlabel="Held-out step", ylabel="Room temperature (C)")
        ax.legend()
        ctx.figure("test_measured_vs_predictions", fig)
        fig, ax = plt.subplots(figsize=(7,4))
        for name, hist in history.items():
            ax.semilogy(hist[:,1], label=name)
        ax.set(xlabel="Epoch", ylabel="Validation MSE (C²)")
        ax.legend()
        ctx.figure("validation_loss", fig)
