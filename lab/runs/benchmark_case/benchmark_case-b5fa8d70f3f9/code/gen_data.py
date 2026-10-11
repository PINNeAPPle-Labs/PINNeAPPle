"""Reference CFD for the heated-channel digital twin: verification + dataset generation.

  python gen_data.py verify   -> results/verification.json
  python gen_data.py dataset  -> data/trajectories.npz (+ data/meta.json)
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[2] / "PINNeAPPle"))
from pinneapple_simulation.numerical_solvers.thermal_channel_2d import ChannelConfig, ThermalChannel2D  # noqa: E402

(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "data").mkdir(exist_ok=True)
U_RANGE, Q_RANGE = (0.6, 1.4), (0.0, 2.0)
WARMUP, HORIZON, REC = 15.0, 40.0, 0.4


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def verify():
    out = {}
    # 1) Poiseuille flow on three grids
    pois = []
    for nx, ny, dt in ((64, 16, 0.02), (128, 32, 0.01), (256, 64, 0.005)):
        s = ThermalChannel2D(ChannelConfig(nx=nx, ny=ny, dt=dt))
        s.reset(1.0, 0.0, fully_developed=False)
        s.run(lambda t: 1.0, lambda t: 0.0, 30.0, record=False)
        f = s.fields()
        ex = 6 * s.yc * (1 - s.yc)
        i = int(0.8 * nx)
        pois.append({"grid": [nx, ny], "u_profile_max_err": float(np.abs(f["u"][i] - ex).max()),
                     "dpdx_mid": float((f["p"][int(0.5 * nx)].mean() - f["p"][int(0.8 * nx)].mean()) / (0.3 * s.cfg.L)),
                     "dpdx_exact": 12.0 / s.cfg.Re, "max_divergence": s.divergence()})
        print("poiseuille", pois[-1], flush=True)
    out["poiseuille"] = pois

    # 2) fully developed Nusselt number, bottom wall uniform flux, top adiabatic (Nu_Dh = 5.385)
    nus = []
    for nx, ny, dt in ((128, 16, 0.01), (256, 32, 0.005)):
        cfg = ChannelConfig(nx=nx, ny=ny, L=32.0, Re=50.0, heater=(1.0, 30.0), dt=dt)
        s = ThermalChannel2D(cfg)
        s.reset(1.0, 1.0, fully_developed=True)
        s.run(lambda t: 1.0, lambda t: 1.0, 160.0, record=False)
        T = s.T[1:-1, 1:-1]
        u = 0.5 * (s.u[1:, 1:-1] + s.u[:-1, 1:-1])
        Tb = (u * T).sum(1) / u.sum(1)
        Tw = s.T[1:-1, 1] + 0.5 * s.dy          # wall temperature from the flux condition (Q=1)
        sel = (s.xc > 15) & (s.xc < 28)
        Nu = 2.0 / (Tw - Tb)
        # global energy balance at steady state: heat in = advected heat out
        q = s.qoi()
        nus.append({"grid": [nx, ny], "Nu_mean": float(Nu[sel].mean()), "Nu_std": float(Nu[sel].std()),
                    "Nu_reference": 5.385, "rel_err": float(Nu[sel].mean() / 5.385 - 1),
                    "energy_in": q["heat_in"], "energy_out": q["heat_out"],
                    "energy_imbalance_rel": float((q["heat_in"] - q["heat_out"]) / q["heat_in"])})
        print("nusselt", nus[-1], flush=True)
    out["nusselt"] = nus

    # 3) three-grid convergence of the digital-twin QoIs on a transient scenario
    from pinneapple_analysis.verification.convergence import richardson_extrapolate
    Uf = lambda t: 1.0 + 0.3 * np.sin(0.3 * t)
    Qf = lambda t: 1.0 if t < 10 else 1.8
    vals = []
    for nx, ny, dt in ((64, 16, 0.02), (128, 32, 0.01), (256, 64, 0.005)):
        s = ThermalChannel2D(ChannelConfig(nx=nx, ny=ny, dt=dt))
        s.reset(1.0, 1.0)
        s.run(Uf, Qf, 20.0, record=False)
        vals.append(s.qoi())
        print("grid", nx, ny, vals[-1], flush=True)
    conv = {}
    for k in ("T_out", "T_max", "dp"):
        r = richardson_extrapolate(vals[0][k], vals[1][k], vals[2][k], r=2.0)
        conv[k] = {"values": [v[k] for v in vals], "observed_order": r.observed_order, "gci_fine": r.gci_fine,
                   "gci_coarse": r.gci_coarse, "asymptotic_ratio": r.asymptotic_ratio, "is_asymptotic": r.is_asymptotic,
                   "extrapolated": float(r.extrapolated_value)}
    out["grid_convergence"] = conv
    (ROOT / "results" / "verification.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(conv, indent=1))


# ---------------------------------------------------------------------------
# actuator trajectories
# ---------------------------------------------------------------------------

def piecewise_linear(rng, lo, hi, t_end, hold=(3.0, 8.0)):
    knots, vals, t = [0.0], [rng.uniform(lo, hi)], 0.0
    while t < t_end + WARMUP:
        t += rng.uniform(*hold)
        knots.append(t)
        # half of the transitions are (near-)steps, half are ramps
        if rng.uniform() < 0.5:
            knots.append(t + 0.05)
            vals.append(vals[-1])
        vals.append(rng.uniform(lo, hi))
    knots, vals = np.array(knots[:len(vals)]), np.array(vals)
    return knots, vals


def make_actuation(family, seed):
    rng = np.random.default_rng(seed)
    T_tot = WARMUP + HORIZON
    if family == "random_steps_ramps":
        ku, vu = piecewise_linear(rng, *U_RANGE, T_tot)
        kq, vq = piecewise_linear(rng, *Q_RANGE, T_tot)
        return ("pl", ku.tolist(), vu.tolist()), ("pl", kq.tolist(), vq.tolist())
    if family == "sinusoid_chirp":   # unseen family: smooth oscillations with drifting frequency
        return (("sin", 1.0, rng.uniform(0.2, 0.4), rng.uniform(0.05, 0.5), rng.uniform(0, 6)),
                ("sin", 1.0, rng.uniform(0.6, 1.0), rng.uniform(0.05, 0.5), rng.uniform(0, 6)))
    if family == "edge_steps":       # unseen family: large jumps between the range edges
        su = rng.choice([U_RANGE[0], U_RANGE[1]])
        sq = rng.choice([Q_RANGE[0], Q_RANGE[1]])
        tu, tq = WARMUP + rng.uniform(5, 20), WARMUP + rng.uniform(5, 20)
        return (("pl", [0, tu, tu + 0.05, T_tot + 1], [su, su, sum(U_RANGE) - su, sum(U_RANGE) - su]),
                ("pl", [0, tq, tq + 0.05, T_tot + 1], [sq, sq, sum(Q_RANGE) - sq, sum(Q_RANGE) - sq]))
    raise ValueError(family)


def eval_act(spec, t, which):
    lo, hi = (U_RANGE if which == "U" else Q_RANGE)
    if spec[0] == "pl":
        return float(np.interp(t, spec[1], spec[2]))
    _, center, amp, f0, ph = spec
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    # chirp: frequency grows linearly from f0 to 2 f0 over the trajectory
    tt = t - WARMUP
    return float(np.clip(mid + amp * half * np.sin(2 * np.pi * (f0 * tt + 0.5 * f0 * tt ** 2 / HORIZON) + ph), lo, hi))


def simulate(args):
    family, seed = args
    au, aq = make_actuation(family, seed)
    s = ThermalChannel2D(ChannelConfig(nx=128, ny=32, dt=0.01, record_every=REC))
    s.reset(eval_act(au, 0, "U"), eval_act(aq, 0, "Q"))
    t0 = time.time()
    s.run(lambda t: eval_act(au, t, "U"), lambda t: eval_act(aq, t, "Q"), WARMUP, record=False)
    s.t = 0.0
    shift = WARMUP
    rec = s.run(lambda t: eval_act(au, t + shift, "U"), lambda t: eval_act(aq, t + shift, "Q"), HORIZON)
    # actuator values for every record interval (commanded at the start of each interval)
    return {"family": family, "seed": seed, "fields": rec["fields"].astype(np.float32),
            "U": np.array(rec["U"], np.float32), "Q": np.array(rec["Q"], np.float32),
            "qoi": {k: np.array([q[k] for q in rec["qoi"]], np.float32) for k in rec["qoi"][0]},
            "U_fine": np.array([eval_act(au, shift + t, "U") for t in np.arange(0, HORIZON + 1e-9, 0.1)], np.float32),
            "Q_fine": np.array([eval_act(aq, shift + t, "Q") for t in np.arange(0, HORIZON + 1e-9, 0.1)], np.float32),
            "seconds": time.time() - t0, "spec": (au, aq)}


PLAN = ([("random_steps_ramps", s, "train") for s in range(48)]
        + [("random_steps_ramps", 1000 + s, "val") for s in range(8)]
        + [("random_steps_ramps", 2000 + s, "test_same_family") for s in range(12)]
        + [("sinusoid_chirp", 3000 + s, "test_unseen_sinusoid") for s in range(8)]
        + [("edge_steps", 4000 + s, "test_unseen_edge_steps") for s in range(6)])


def dataset(workers=4):
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        runs = list(ex.map(simulate, [(f, s) for f, s, _ in PLAN]))
    split = np.array([sp for _, _, sp in PLAN])
    np.savez_compressed(ROOT / "data" / "trajectories.npz",
                        fields=np.stack([r["fields"] for r in runs]), U=np.stack([r["U"] for r in runs]),
                        Q=np.stack([r["Q"] for r in runs]), U_fine=np.stack([r["U_fine"] for r in runs]),
                        Q_fine=np.stack([r["Q_fine"] for r in runs]), split=split,
                        family=np.array([r["family"] for r in runs]), seed=np.array([r["seed"] for r in runs]),
                        **{f"qoi_{k}": np.stack([r["qoi"][k] for r in runs]) for k in runs[0]["qoi"]})
    s = ThermalChannel2D(ChannelConfig(nx=128, ny=32))
    meta = {"nx": 128, "ny": 32, "L": 10.0, "Re": 100.0, "Pr": 0.71, "heater": [2.0, 4.0], "dt_solver": 0.01,
            "dt_record": REC, "warmup": WARMUP, "horizon": HORIZON, "U_range": U_RANGE, "Q_range": Q_RANGE,
            "xc": s.xc.tolist(), "yc": s.yc.tolist(), "solver_seconds": [r["seconds"] for r in runs],
            "wall_seconds_total": time.time() - t0, "specs": [str(r["spec"]) for r in runs]}
    (ROOT / "data" / "meta.json").write_text(json.dumps(meta, indent=1))
    print("dataset done", time.time() - t0)


if __name__ == "__main__":
    {"verify": verify, "dataset": dataset}[sys.argv[1]]()
