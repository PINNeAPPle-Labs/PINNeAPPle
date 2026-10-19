"""Confined cylinder wake with the D2Q9 lattice-Boltzmann solver: one run, its verification and the zoo dataset
``cylinder_wake_lbm``.

A cylinder of diameter ``D`` lattice cells sits 3 D downstream of a uniform (plug) inlet in a channel of height
``height`` D and length ``length`` D (blockage D/H = 1/height), with no-slip walls (bounce-back) and a pressure
outlet. :func:`simulate` runs ``conv_time`` convective units (t U / D) and keeps the last ``keep_time``: vorticity
snapshots normalised by U/D, the transverse-velocity probe 2 D behind the cylinder, the wake regime (steady or
shedding) and the Strouhal number (Hann-windowed, peak-refined spectrum of the probe).

* :func:`verify`: the developed channel profile against plane Poiseuille flow (low Re, two grids), and a resolution
  study at Re 100 (D = 20, 30, 40 at the same physical time and sampling);
* :data:`PLAN` / :func:`generate`: the dataset, Re 20..100 without a turbulence model, written as
  ``<split>[-NN].npz`` + ``meta.json`` (split by Reynolds number).
"""
from __future__ import annotations

import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

__all__ = ["SETUP", "PLAN", "SPLITS", "simulate", "probe_strouhal", "verify", "generate"]

SETUP = {"D": 20, "height": 5, "length": 12, "u_in": 0.1, "conv_time": 200.0, "keep_time": 40.0, "save_dt": 0.5,
         "Cs": 0.0}
# Re <= 100: at D = 20 the Strouhal number is within 0.7 % of D = 40 at Re 100, but 4.4 % below D = 30 at Re 120
# (tau 0.55, near the BGK stability limit), so higher Reynolds numbers are left out
PLAN = {"train": [20.0, 30.0, 50.0, 60.0, 80.0, 90.0], "test": [40.0, 70.0, 100.0]}
SPLITS = ("train", "test")
SHARD_ROWS = 320                       # 4 runs of 80 snapshots per file
SHEDDING_AMPLITUDE = 0.02              # rms of the probe's transverse velocity (/U) above which the wake sheds


def probe_strouhal(signal: np.ndarray, dt: float) -> float:
    """Strouhal number of a probe record sampled every ``dt`` convective units: strongest peak of the
    Hann-windowed, 4x zero-padded spectrum with Gaussian peak interpolation (a fraction of a bin)."""
    from pinneapple_physics.physics_audit.regime import dominant_frequency
    return float(dominant_frequency(signal, dt=dt, discard=0.0, window=True, refine=True, pad=4))


def simulate(Re: float, D: int = SETUP["D"], height: int = SETUP["height"], length: int = SETUP["length"],
             u_in: float = SETUP["u_in"], conv_time: float = SETUP["conv_time"],
             keep_time: float | None = SETUP["keep_time"], save_dt: float = SETUP["save_dt"],
             Cs: float = SETUP["Cs"]) -> dict:
    """One run. Returns the setup, the convective times ``t`` and vorticity frames ``vorticity`` (n, nx, ny) of the
    last ``keep_time`` convective units (``None``: the second half of the run), the probe ``probe_uy`` (/U) over
    the same times, ``wake_amplitude`` (its rms), ``amplitude_growth`` (probe rms of the last quarter of the window
    over the first: ~1 for a saturated wake), ``regime`` and ``strouhal`` (0 for a steady wake), and the ``solid``
    mask."""
    import torch

    from .lbm import LBMSolver, _d2q9_tensors, _equilibrium_2d, cylinder_mask
    nx, ny = length * D, height * D
    cx, cy = 3.0 * D, ny / 2 + 0.5                     # half-cell offset breaks the symmetry
    mask = cylinder_mask(nx, ny, cx, cy, D / 2.0)
    re_channel = Re * (ny - 2) / D                     # the solver defines Re on the channel height
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = LBMSolver(nx=nx, ny=ny, Re=re_channel, u_in=u_in, obstacle_mask=mask, Cs=Cs)
    steps_per_unit = D / u_in                          # lattice steps per convective unit
    save_every = max(1, int(round(save_dt * steps_per_unit)))
    steps = int(round(conv_time * steps_per_unit / save_every)) * save_every
    cxv, cyv, wv, _ = _d2q9_tensors(torch.device("cpu"))
    ux0 = torch.where(mask.bool(), 0.0, float(u_in))   # fluid at rest inside the obstacle: no start-up shock
    f0 = _equilibrium_2d(torch.ones(nx, ny), ux0, torch.zeros(nx, ny), cxv, cyv, wv)
    t0 = time.time()
    with torch.no_grad():
        out = solver(f0, steps=steps, save_every=save_every)
    ux = np.asarray(torch.stack(out.extras["trajectory_ux"]), dtype=np.float32)
    uy = np.asarray(torch.stack(out.extras["trajectory_uy"]), dtype=np.float32)
    n_keep = len(ux) // 2 if keep_time is None else min(len(ux), int(round(keep_time / (save_every / steps_per_unit))))
    keep = slice(len(ux) - n_keep, len(ux))
    ux, uy = ux[keep], uy[keep]
    t = (np.arange(len(out.extras["trajectory_ux"])) + 1)[keep] * save_every / steps_per_unit
    solid = mask.numpy().astype(bool)
    w = (np.gradient(uy, axis=1) - np.gradient(ux, axis=2)) / u_in * D
    w[:, solid] = 0.0
    probe = uy[:, int(cx + 2 * D), int(ny / 2)] / u_in
    amp = float(np.std(probe))
    q = max(1, len(probe) // 4)
    growth = float(np.std(probe[-q:]) / max(np.std(probe[:q]), 1e-12))
    shedding = amp > SHEDDING_AMPLITUDE
    return {"Re": float(Re), "D": D, "nx": nx, "ny": ny, "cylinder_center": [cx, cy], "u_in": u_in, "Cs": Cs,
            "tau": 1.0 / solver.omega, "t": t.astype(np.float32), "vorticity": w.astype(np.float32),
            "probe_uy": probe.astype(np.float32), "wake_amplitude": amp, "amplitude_growth": growth,
            "regime": "shedding" if shedding else "steady",
            "strouhal": probe_strouhal(probe, save_every / steps_per_unit) if shedding else 0.0,
            "solid": solid, "finite": bool(np.isfinite(ux).all() and np.isfinite(uy).all()),
            "seconds": time.time() - t0}


def _channel_profile(D: int, height: int, length: int, u_in: float, Re: float) -> dict:
    """Empty channel with the dataset's inlet and walls: developed profile 3/4 down the channel against the plane
    Poiseuille parabola of the same flow rate between walls half a cell outside the boundary nodes."""
    import torch

    from .lbm import LBMSolver
    nx, ny = length * D, height * D
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = LBMSolver(nx=nx, ny=ny, Re=Re * (ny - 2) / D, u_in=u_in)
    nu = u_in * D / Re
    steps = int(round(max(4 * length * D / u_in, 3 * (ny - 2) ** 2 / nu)))   # 4 flow-throughs, 3 viscous times
    with torch.no_grad():
        out = solver(steps=steps, save_every=10 ** 9)
    u = out.extras["ux"].numpy()[int(0.75 * nx), 1:-1].astype(float)
    h = ny - 2.0                                       # fluid nodes 1..ny-2, walls half a cell beyond them
    s = np.arange(1, ny - 1) - 0.5                     # distance of each fluid node from the lower wall
    q = u.mean()                                       # midpoint rule for the mean velocity
    exact = 6 * q * (s / h) * (1 - s / h)
    return {"grid": [nx, ny], "u_max_over_mean": float(u.max() / q), "u_max_over_mean_exact": 1.5,
            "profile_max_rel_err": float(np.abs(u - exact).max() / exact.max())}


def verify(Ds=(20, 30, 40), Re: float = 100.0, channel_Re: float = 2.0) -> dict:
    """Developed channel profile (plane Poiseuille; at ``channel_Re`` on D the entrance length is ~2.5 D, so the
    profile is developed 9 D downstream) on the dataset grid and a 2x coarser one, and the resolution study of the
    wake at ``Re``. At the dataset's Reynolds numbers the 12 D channel is far shorter than the entrance length, so
    the cylinder sees a nearly uniform inflow with thin wall boundary layers."""
    out = {"channel": [_channel_profile(D, SETUP["height"], SETUP["length"], SETUP["u_in"], channel_Re)
                       for D in (Ds[0] // 2, Ds[0])]}
    study = []
    for D in Ds:
        r = simulate(Re, D=D)
        cx, cy = r["cylinder_center"]
        w2 = r["vorticity"][:, int(cx + 2 * D), int(cy)]
        study.append({"D": D, "tau": r["tau"], "strouhal": r["strouhal"], "wake_amplitude": r["wake_amplitude"],
                      "vorticity_rms_2D": float(np.std(w2)), "seconds": r["seconds"]})
    fine = study[-1]
    for s in study:
        for k in ("strouhal", "wake_amplitude", "vorticity_rms_2D"):
            s[f"{k}_rel_diff_vs_D{fine['D']}"] = float(abs(s[k] / fine[k] - 1))
    out["resolution_study"] = study
    return out


def _run(args) -> dict:
    Re, setup = args
    return simulate(Re, **setup)


def generate(out: Path, workers: int = 1, name: str = "cylinder_wake_lbm") -> dict:
    """Run :data:`PLAN` and write ``out/<name>/<split>[-NN].npz`` (one row per snapshot) and ``meta.json``."""
    from .operator_datasets import write_npz
    t0 = time.time()
    res = [r for r in (Re for s in SPLITS for Re in PLAN[s])]
    with ProcessPoolExecutor(max(1, workers)) as ex:
        runs = dict(zip(res, ex.map(_run, [(Re, dict(SETUP)) for Re in res]), strict=True))
    d = Path(out) / name
    files, rows = [], {}
    for split in SPLITS:
        rs = [runs[Re] for Re in PLAN[split]]
        cols = {"Re": np.concatenate([np.full(len(r["t"]), r["Re"]) for r in rs]),
                "t": np.concatenate([r["t"] for r in rs]),
                "vorticity": np.concatenate([r["vorticity"] for r in rs]),
                "regime": np.concatenate([np.full(len(r["t"]), r["regime"]) for r in rs]),
                "strouhal": np.concatenate([np.full(len(r["t"]), r["strouhal"]) for r in rs]),
                "wake_amplitude": np.concatenate([np.full(len(r["t"]), r["wake_amplitude"]) for r in rs])}
        n = len(cols["t"])
        starts = range(0, n, SHARD_ROWS)
        for j, i0 in enumerate(starts):
            fn = f"{split}.npz" if len(starts) == 1 else f"{split}-{j:02d}.npz"
            write_npz(d / fn, {f"{split}__{k}": v[i0:i0 + SHARD_ROWS] for k, v in cols.items()})
            files.append(fn)
        rows[split] = n
    r0 = next(iter(runs.values()))
    meta = {"name": name, "files": files, "rows": rows, **SETUP, "nx": r0["nx"], "ny": r0["ny"],
            "cylinder_center": r0["cylinder_center"], "solid": np.argwhere(r0["solid"]).tolist(),
            "runs": {f"{Re:g}": {k: runs[Re][k] for k in ("regime", "strouhal", "wake_amplitude", "amplitude_growth",
                                                         "tau", "finite", "seconds")} for Re in runs},
            "wall_seconds": time.time() - t0, "workers": workers, "cpu": os.cpu_count()}
    (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta
