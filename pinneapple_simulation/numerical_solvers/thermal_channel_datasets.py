"""Heated-channel CFD dataset (zoo ``thermal_channel_cfd``): verification and actuated trajectories.

The digital-twin benchmark of PINNeAPPle-Benchmark experiment 05, on the library's 2-D incompressible solver
:class:`.thermal_channel_2d.ThermalChannel2D` (channel L = 10, Re 100, Pr 0.71, heater on the bottom wall at
x in [2, 4]): the inlet velocity U(t) and the heater power Q(t) are driven by actuator trajectories, and the solver
records u, v, p, T every 0.4 time units over a 40-unit horizon after a 15-unit warm-up.

* :func:`verify`: Poiseuille flow on three grids, the fully developed Nusselt number (5.385) with the global energy
  balance, and a three-grid Richardson / GCI study of the twin's quantities of interest;
* :func:`simulate` / :data:`PLAN` / :func:`generate`: 82 trajectories in five splits (random steps and ramps for
  train / val / test, and two unseen actuation families), written as ``<split>[-NN].npz`` + ``meta.json``.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .thermal_channel_2d import ChannelConfig, ThermalChannel2D

__all__ = ["U_RANGE", "Q_RANGE", "WARMUP", "HORIZON", "REC", "PLAN", "SPLITS", "make_actuation", "eval_act",
           "simulate", "verify", "generate"]

U_RANGE, Q_RANGE = (0.6, 1.4), (0.0, 2.0)
WARMUP, HORIZON, REC = 15.0, 40.0, 0.4
GRID = dict(nx=128, ny=32, dt=0.01)

PLAN = ([("random_steps_ramps", s, "train") for s in range(48)]
        + [("random_steps_ramps", 1000 + s, "val") for s in range(8)]
        + [("random_steps_ramps", 2000 + s, "test_same_family") for s in range(12)]
        + [("sinusoid_chirp", 3000 + s, "test_unseen_sinusoid") for s in range(8)]
        + [("edge_steps", 4000 + s, "test_unseen_edge_steps") for s in range(6)])
SPLITS = ("train", "val", "test_same_family", "test_unseen_sinusoid", "test_unseen_edge_steps")
SHARD_ROWS = 12


# --------------------------------------------------------------------------- verification
def verify() -> dict:
    """Poiseuille (3 grids), fully developed Nusselt number (2 grids) and the three-grid study of T_out, T_max, dp."""
    from pinneapple_analysis.verification.convergence import richardson_extrapolate
    out: dict = {}
    pois = []
    for nx, ny, dt in ((64, 16, 0.02), (128, 32, 0.01), (256, 64, 0.005)):
        s = ThermalChannel2D(ChannelConfig(nx=nx, ny=ny, dt=dt))
        s.reset(1.0, 0.0, fully_developed=False)
        s.run(lambda t: 1.0, lambda t: 0.0, 30.0, record=False)
        f = s.fields()
        ex = 6 * s.yc * (1 - s.yc)
        i = int(0.8 * nx)
        pois.append({"grid": [nx, ny], "u_profile_max_err": float(np.abs(f["u"][i] - ex).max()),
                     "dpdx_mid": float((f["p"][int(0.5 * nx)].mean() - f["p"][int(0.8 * nx)].mean())
                                       / (0.3 * s.cfg.L)),
                     "dpdx_exact": 12.0 / s.cfg.Re, "max_divergence": s.divergence()})
    out["poiseuille"] = pois
    nus = []
    for nx, ny, dt in ((128, 16, 0.01), (256, 32, 0.005)):
        s = ThermalChannel2D(ChannelConfig(nx=nx, ny=ny, L=32.0, Re=50.0, heater=(1.0, 30.0), dt=dt))
        s.reset(1.0, 1.0, fully_developed=True)
        s.run(lambda t: 1.0, lambda t: 1.0, 160.0, record=False)
        T = s.T[1:-1, 1:-1]
        u = 0.5 * (s.u[1:, 1:-1] + s.u[:-1, 1:-1])
        Tb = (u * T).sum(1) / u.sum(1)
        Tw = s.T[1:-1, 1] + 0.5 * s.dy                 # wall temperature from the flux condition (Q = 1)
        sel = (s.xc > 15) & (s.xc < 28)
        Nu = 2.0 / (Tw - Tb)
        q = s.qoi()
        nus.append({"grid": [nx, ny], "Nu_mean": float(Nu[sel].mean()), "Nu_reference": 5.385,
                    "rel_err": float(Nu[sel].mean() / 5.385 - 1),
                    "energy_imbalance_rel": float((q["heat_in"] - q["heat_out"]) / q["heat_in"])})
    out["nusselt"] = nus
    vals = []
    for nx, ny, dt in ((64, 16, 0.02), (128, 32, 0.01), (256, 64, 0.005)):
        s = ThermalChannel2D(ChannelConfig(nx=nx, ny=ny, dt=dt))
        s.reset(1.0, 1.0)
        s.run(lambda t: 1.0 + 0.3 * np.sin(0.3 * t), lambda t: 1.0 if t < 10 else 1.8, 20.0, record=False)
        vals.append(s.qoi())
    conv = {}
    for k in ("T_out", "T_max", "dp"):
        r = richardson_extrapolate(vals[0][k], vals[1][k], vals[2][k], r=2.0)
        conv[k] = {"values": [v[k] for v in vals], "observed_order": r.observed_order, "gci_fine": r.gci_fine,
                   "gci_coarse": r.gci_coarse, "extrapolated": float(r.extrapolated_value)}
    out["grid_convergence"] = conv
    return out


# --------------------------------------------------------------------------- actuation
def _piecewise_linear(rng, lo, hi, t_end, hold=(3.0, 8.0)):
    knots, vals, t = [0.0], [rng.uniform(lo, hi)], 0.0
    while t < t_end + WARMUP:
        t += rng.uniform(*hold)
        knots.append(t)
        if rng.uniform() < 0.5:                        # half of the transitions are (near-)steps, half are ramps
            knots.append(t + 0.05)
            vals.append(vals[-1])
        vals.append(rng.uniform(lo, hi))
    return np.array(knots[:len(vals)]), np.array(vals)


def make_actuation(family: str, seed: int):
    """(U spec, Q spec) of one trajectory: ``random_steps_ramps`` (training family), ``sinusoid_chirp`` and
    ``edge_steps`` (unseen families)."""
    rng = np.random.default_rng(seed)
    T_tot = WARMUP + HORIZON
    if family == "random_steps_ramps":
        ku, vu = _piecewise_linear(rng, *U_RANGE, T_tot)
        kq, vq = _piecewise_linear(rng, *Q_RANGE, T_tot)
        return ("pl", ku.tolist(), vu.tolist()), ("pl", kq.tolist(), vq.tolist())
    if family == "sinusoid_chirp":
        return (("sin", 1.0, rng.uniform(0.2, 0.4), rng.uniform(0.05, 0.5), rng.uniform(0, 6)),
                ("sin", 1.0, rng.uniform(0.6, 1.0), rng.uniform(0.05, 0.5), rng.uniform(0, 6)))
    if family == "edge_steps":
        su = rng.choice([U_RANGE[0], U_RANGE[1]])
        sq = rng.choice([Q_RANGE[0], Q_RANGE[1]])
        tu, tq = WARMUP + rng.uniform(5, 20), WARMUP + rng.uniform(5, 20)
        return (("pl", [0, tu, tu + 0.05, T_tot + 1], [su, su, sum(U_RANGE) - su, sum(U_RANGE) - su]),
                ("pl", [0, tq, tq + 0.05, T_tot + 1], [sq, sq, sum(Q_RANGE) - sq, sum(Q_RANGE) - sq]))
    raise ValueError(family)


def eval_act(spec, t: float, which: str) -> float:
    lo, hi = (U_RANGE if which == "U" else Q_RANGE)
    if spec[0] == "pl":
        return float(np.interp(t, spec[1], spec[2]))
    _, _center, amp, f0, ph = spec
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    tt = t - WARMUP                                    # chirp: the frequency grows from f0 to 2 f0
    return float(np.clip(mid + amp * half * np.sin(2 * np.pi * (f0 * tt + 0.5 * f0 * tt ** 2 / HORIZON) + ph),
                         lo, hi))


def simulate(args) -> dict:
    """One trajectory: warm-up, then the recorded horizon (fields every ``REC``)."""
    family, seed = args
    return _trajectory(family, seed, GRID, WARMUP, HORIZON, REC)


def _trajectory(family: str, seed: int, grid: dict, warmup: float, horizon: float, rec_every: float) -> dict:
    au, aq = make_actuation(family, seed)
    s = ThermalChannel2D(ChannelConfig(**grid, record_every=rec_every))
    s.reset(eval_act(au, 0, "U"), eval_act(aq, 0, "Q"))
    t0 = time.time()
    s.run(lambda t: eval_act(au, t, "U"), lambda t: eval_act(aq, t, "Q"), warmup, record=False)
    s.t = 0.0
    rec = s.run(lambda t: eval_act(au, t + warmup, "U"), lambda t: eval_act(aq, t + warmup, "Q"), horizon)
    fine = np.arange(0, horizon + 1e-9, 0.1)
    return {"family": family, "seed": seed, "fields": rec["fields"].astype(np.float32),
            "U": np.array(rec["U"], np.float32), "Q": np.array(rec["Q"], np.float32),
            "qoi": {k: np.array([q[k] for q in rec["qoi"]], np.float32) for k in rec["qoi"][0]},
            "U_fine": np.array([eval_act(au, warmup + t, "U") for t in fine], np.float32),
            "Q_fine": np.array([eval_act(aq, warmup + t, "Q") for t in fine], np.float32),
            "seconds": time.time() - t0}


def _run(task) -> dict:
    """Runs one task in a worker of :func:`generate`.

    The recipe's parameters travel *with the task*: a spawned worker (the default on macOS and Windows)
    re-imports this module and would otherwise read the module defaults instead of the grid and horizon
    the caller asked for.
    """
    family, seed, grid, warmup, horizon, rec_every = task
    globals().update(GRID=grid, WARMUP=warmup, HORIZON=horizon, REC=rec_every)   # eval_act reads them
    return _trajectory(family, seed, grid, warmup, horizon, rec_every)


def generate(out: Path, workers: int = 1, name: str = "thermal_channel_cfd") -> dict:
    """Run :data:`PLAN` and write ``out/<name>/<split>[-NN].npz`` and ``meta.json``."""
    from .operator_datasets import write_npz
    t0 = time.time()
    tasks = [(f, s, dict(GRID), WARMUP, HORIZON, REC) for f, s, _ in PLAN]
    if workers <= 1:
        runs = [_run(task) for task in tasks]
    else:
        with ProcessPoolExecutor(workers) as ex:
            runs = list(ex.map(_run, tasks))
    d = Path(out) / name
    files, rows = [], {}
    for split in SPLITS:
        rs = [r for r, (_, _, sp) in zip(runs, PLAN, strict=True) if sp == split]
        parts = [rs[i:i + SHARD_ROWS] for i in range(0, len(rs), SHARD_ROWS)]
        for j, part in enumerate(parts):
            fn = f"{split}.npz" if len(parts) == 1 else f"{split}-{j:02d}.npz"
            cols = {"family": np.array([r["family"] for r in part]), "seed": np.array([r["seed"] for r in part]),
                    **{k: np.stack([r[k] for r in part]) for k in ("fields", "U", "Q", "U_fine", "Q_fine")},
                    **{f"qoi_{k}": np.stack([r["qoi"][k] for r in part]) for k in part[0]["qoi"]}}
            write_npz(d / fn, {f"{split}__{k}": v for k, v in cols.items()})
            files.append(fn)
        rows[split] = len(rs)
    s = ThermalChannel2D(ChannelConfig(**GRID))
    meta = {"name": name, "files": files, "rows": rows, **GRID, "L": 10.0, "Re": 100.0, "Pr": 0.71,
            "heater": [2.0, 4.0], "dt_record": REC, "warmup": WARMUP, "horizon": HORIZON,
            "U_range": U_RANGE, "Q_range": Q_RANGE, "xc": s.xc.tolist(), "yc": s.yc.tolist(),
            "solver_seconds": [r["seconds"] for r in runs], "wall_seconds": time.time() - t0,
            "workers": workers, "cpu": os.cpu_count()}
    (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta
