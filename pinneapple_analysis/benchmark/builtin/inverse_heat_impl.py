"""Inverse heat benchmark implementation (moved from PINNeAPPle-Benchmark experiment 09): the convection coefficient
h of the heat-spreader plate from 8 thermocouples; the plate PINN (run_plate under a budget) against surrogate +
calibration (MLP, DeepONet on finite-volume runs) and the finite-volume model itself + calibration. The data
directory is set by ``set_data_dir``.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pinneapple_analysis.cost.equal_budget import (
    ComputeBudget,
    cosine_lr,
    set_lr,
    time_per_query,
    train_within_budget,
)
from pinneapple_analysis.inverse_problems.heat_transfer import (
    PLATE,
    PLATE_SENSORS_MM,
    _golden,
    run_plate,
)
from pinneapple_neural.architectures.neural_operators.deeponet import DeepONet, mlp
from pinneapple_simulation.numerical_solvers.heat_fv_reference import interp_grid, plate_2d

from ..comparison import save_comparison

_DIR = {"data": None, "smoke": False}


def set_data_dir(prepared, smoke: bool = False) -> None:
    _DIR["data"], _DIR["smoke"] = Path(prepared), smoke


def single_thread():
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)


NOISE, H_GUESS = 0.5, 60.0
H_RANGE, P_RANGE = (5.0, 100.0), (2.0, 20.0)
CASES = {"id": [(10.0, 25.0), (5.0, 60.0), (15.0, 10.0)], "ood": [(10.0, 150.0), (10.0, 200.0)]}
SPACES = {
    "mlp": {"width": [64, 128], "depth": [3, 4], "lr": [1e-3, 3e-3]},
    "deeponet": {"hidden": [64, 128], "depth": [3, 4], "p": [32, 64], "lr": [1e-3, 3e-3]},
    "pinn": {"width": [32, 48], "depth": [3, 4], "n_colloc": [1500, 2500], "lr": [1e-3, 2e-3]},
}
FV = dict(lx=PLATE["lx"], ly=PLATE["ly"], t=PLATE["t"], k=PLATE["k"], t_air=PLATE["t_air"], device=PLATE["device"])
SENS = np.asarray(PLATE_SENSORS_MM, float) / 1000


def fv_field(h, power):
    return plate_2d(h, nx=100, ny=60, power=power, **FV)


def readings(power, h_true, seed):
    x, y, t_ref = fv_field(h_true, power)
    rng = np.random.default_rng(seed + 100)
    return interp_grid(x, y, t_ref, SENS[:, 0], SENS[:, 1]) + rng.normal(0, NOISE, len(SENS)), t_ref


# ------------------------------------------------------------------------------------------------- data
def _fv_one(hp):
    single_thread()
    t0 = time.process_time()
    _, _, T = fv_field(*hp)
    return T.astype(np.float32), time.process_time() - t0


def generate(out: Path, n_train: int, n_val: int, workers: int = 4) -> dict:
    rng = np.random.default_rng(21)

    def lhs(n):
        u = (np.argsort(rng.random((2, n)), axis=1).T + rng.random((n, 2))) / n
        return np.c_[H_RANGE[0] * (H_RANGE[1] / H_RANGE[0]) ** u[:, 0], P_RANGE[0] + u[:, 1] * (P_RANGE[1] - P_RANGE[0])]
    hp = {"train": lhs(n_train), "val": lhs(n_val)}
    out_arrays, cost = {}, {}
    with ProcessPoolExecutor(max(1, workers)) as ex:
        for s, v in hp.items():
            res = list(ex.map(_fv_one, [tuple(r) for r in v]))
            out_arrays[f"{s}_hp"], out_arrays[f"{s}_T"] = v, np.stack([r[0] for r in res])
            cost[s] = {"n": len(v), "solver_core_s": float(sum(r[1] for r in res))}
    x, y, _ = fv_field(25.0, 10.0)
    np.savez_compressed(Path(out) / "inverse_heat.npz", x=x, y=y, **out_arrays)
    total = sum(c["solver_core_s"] for c in cost.values())
    return {"problem": "inverse_heat", "splits": cost, "solver_core_s_total": total,
            "solver_core_s_per_sample": total / sum(c["n"] for c in cost.values()), "train_set_core_s": total,
            "h_range": H_RANGE, "p_range": P_RANGE}


# ------------------------------------------------------------------------------------------------- surrogates
class Surrogate(nn.Module):
    """(log h, power) normalised -> (T - T_air) field (60 x 100), normalised."""

    def __init__(self, method, cfg, xy):
        super().__init__()
        self.method = method
        if method == "mlp":
            self.net = mlp(2, cfg["width"], 60 * 100, depth=cfg["depth"])
        else:
            self.net = DeepONet(2, 2, 1, hidden=cfg["hidden"], modes=cfg["p"], depth=cfg["depth"])
            self.register_buffer("xy", xy)

    def forward(self, z):
        if self.method == "mlp":
            return self.net(z).reshape(len(z), 60, 100)
        return self.net(z, self.xy).y.reshape(len(z), 60, 100)


class Data:
    def __init__(self):
        d = np.load(_DIR["data"] / "inverse_heat.npz")
        self.x, self.y = d["x"], d["y"]
        self.hp = {s: d[f"{s}_hp"] for s in ("train", "val")}
        self.T = {s: d[f"{s}_T"] - FV["t_air"] for s in ("train", "val")}
        self.ts = float(self.T["train"].std())
        X, Y = np.meshgrid(self.x / FV["lx"], self.y / FV["lx"])
        self.xy = torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=torch.float32)

    @staticmethod
    def z(h, p):
        h, p = np.atleast_1d(h), np.atleast_1d(p)
        lh = (np.log(h) - math.log(H_RANGE[0])) / math.log(H_RANGE[1] / H_RANGE[0]) * 2 - 1
        return torch.tensor(np.c_[lh, (p - P_RANGE[0]) / (P_RANGE[1] - P_RANGE[0]) * 2 - 1], dtype=torch.float32)


def train_surrogate(method: str, cfg: dict, seed: int, budget: ComputeBudget):
    D = Data()
    torch.manual_seed(seed)
    model = Surrogate(method, cfg, D.xy)
    Z = D.z(*D.hp["train"].T)
    T = torch.tensor(D.T["train"] / D.ts, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(seed)
    bs = 20

    def step(progress):
        set_lr(opt, cosine_lr(progress, cfg["lr"]))
        idx = torch.from_numpy(rng.integers(0, len(Z), bs))
        loss = ((model(Z[idx]) - T[idx]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss.detach()

    run = train_within_budget(step, budget)
    model.eval()
    with torch.no_grad():
        pv = model(D.z(*D.hp["val"].T)).numpy() * D.ts
    tv = D.T["val"]
    val = float(np.mean(np.linalg.norm((pv - tv).reshape(len(tv), -1), axis=1) / np.linalg.norm(tv.reshape(len(tv), -1), axis=1)))
    return model, D, run, val


def calibrate(field_fn, power, t_meas, x, y):
    """Golden-section search of h in [1, 300] W/m2K for the sensor misfit of field_fn(h, power) (degC field)."""
    sse = lambda h: float(np.sum((interp_grid(x, y, field_fn(h, power), SENS[:, 0], SENS[:, 1]) - t_meas) ** 2))  # noqa: E731
    calls = {"n": 0}

    def counted(h):
        calls["n"] += 1
        return sse(h)
    return _golden(counted, 1.0, 300.0), calls["n"]


def surrogate_job(method: str, cfg: dict, seed: int, budget: ComputeBudget, *, final: bool,
                  comparison_path=None) -> dict:
    model, D, run, val = train_surrogate(method, cfg, seed, budget)
    out = {"train": run.to_dict(), "val_rel_l2": val, "n_params": sum(p.numel() for p in model.parameters())}
    if not final:
        return out

    def field(h, p):
        with torch.no_grad():
            return model(D.z(h, p)).numpy()[0] * D.ts + FV["t_air"]
    cases = []
    for kind, lst in CASES.items():
        for power, h_true in lst:
            t_meas, t_ref = readings(power, h_true, seed)
            t0 = time.perf_counter()
            h_hat, n_eval = calibrate(field, power, t_meas, D.x, D.y)
            dt = time.perf_counter() - t0
            T = field(h_hat, power)
            cases.append({"kind": kind, "power": power, "h_true": h_true, "h_hat": h_hat,
                          "h_rel_err": abs(h_hat - h_true) / h_true, "max_field_err_C": float(np.abs(T - t_ref).max()),
                          "hot_spot_err_C": float(abs(T.max() - t_ref.max())), "calibration_s": dt, "n_eval": n_eval})
            if kind == "id" and power == CASES["id"][0][0]:        # the first case, "id0": the report's figure
                X, Y = np.meshgrid(D.x, D.y, indexing="xy")        # the field is (ny, nx), so x runs fast
                save_comparison(comparison_path, t_ref, T, x=X, y=Y, quantity="T", unit="°C", instance="id0")
    out["cases"] = cases
    z1 = D.z(25.0, 10.0)
    out["infer_batch1"] = time_per_query(lambda: model(z1), repeats=30, warmup=5)
    return out


def pinn_job(cfg: dict, seed: int, budget: ComputeBudget, *, kind: str, case: int, comparison_path=None) -> dict:
    power, h_true = (CASES[kind][case] if kind in CASES else (8.0, 40.0))   # "val": a search case of its own
    r = run_plate(power=power, h_true=h_true, noise=NOISE, h_guess=H_GUESS, seed=seed, budget=budget,
                  width=cfg["width"], depth=cfg["depth"], n_colloc=cfg["n_colloc"], lr=cfg["lr"])
    t_ref = np.asarray(r["field_reference"])
    if kind == "id" and case == 0:                                  # the first case, "id0": the report's figure
        X, Y = np.meshgrid(np.asarray(r["x_mm"]) / 1000, np.asarray(r["y_mm"]) / 1000, indexing="xy")
        save_comparison(comparison_path, t_ref, np.asarray(r["field_pinn"]), x=X, y=Y, quantity="T", unit="°C",
                        instance="id0")
    return {"train": r["train"], "kind": kind, "case": case, "power": power, "h_true": h_true, "h_hat": r["h_pinn"],
            "h_rel_err": abs(r["h_pinn"] - h_true) / h_true, "max_field_err_C": r["max_field_error"],
            "hot_spot_err_C": abs(r["hot_spot_pinn"] - r["hot_spot_reference"]),
            "field_ref_max": float(t_ref.max())}


def classical_job(seed: int) -> dict:
    """Golden-section search through the finite-volume model itself (the solver as the 'surrogate')."""
    single_thread()
    cases = []
    x, y, _ = fv_field(25.0, 10.0)
    for kind, lst in CASES.items():
        for power, h_true in lst:
            t_meas, t_ref = readings(power, h_true, seed)
            t0 = time.process_time()
            h_hat, n_eval = calibrate(lambda h, p: fv_field(h, p)[2], power, t_meas, x, y)
            dt = time.process_time() - t0
            T = fv_field(h_hat, power)[2]
            cases.append({"kind": kind, "power": power, "h_true": h_true, "h_hat": h_hat,
                          "h_rel_err": abs(h_hat - h_true) / h_true, "max_field_err_C": float(np.abs(T - t_ref).max()),
                          "hot_spot_err_C": float(abs(T.max() - t_ref.max())), "calibration_core_s": dt,
                          "n_eval": n_eval})
    return {"cases": cases}


