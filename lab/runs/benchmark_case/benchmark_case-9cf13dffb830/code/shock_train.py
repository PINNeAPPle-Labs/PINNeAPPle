"""Supersonic duct flow with a fixed-pressure outlet (shock train / pseudo-shock) with PINNeAPPle's
compressible finite-volume solver.

Stages (python shock_train.py <stage> ...):
  verify   -- inviscid standing normal shock at M = 2 vs Rankine-Hugoniot
  sweep    -- 2-D viscous duct, M_in = 2, back-pressure ratios p_b/p_in in P_SWEEP
  grid     -- three-grid study of the shock-train leading-edge position (Richardson / GCI)
  duct3d   -- one 3-D square-duct case (top and side views, as in the CONVERGE picture)
Results: ../results/<stage>*.json|npz
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[2] / "PINNeAPPle"))
from pinneapple_simulation.numerical_solvers.compressible_fv import CompressibleConfig, CompressibleFV  # noqa: E402

RES = ROOT / "results"
RES.mkdir(exist_ok=True)
G, M_IN, L, H = 1.4, 2.0, 12.0, 1.0
P_SWEEP = [2.0, 2.5, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25]
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(RES / "log.txt", "a") as fh:
        fh.write(s + "\n")


def leading_edge(p_center, x, p_in, thr=1.15):
    """First attempt, kept for the record: first x where the centre-line pressure exceeds thr * p_in.
    It is triggered by the gradual compression of the growing wall boundary layers even when no shock
    train exists (p_b/p_in = 2: core supersonic to the exit, yet this returned x = 2.49), so it is not
    the shock-train position."""
    i = np.where(p_center > thr * p_in)[0]
    return float(x[i[0]]) if len(i) else float("nan")


def sonic_front(M_center, x):
    """Shock-train leading edge used in the paper: first x where the centre-line Mach number drops below 1
    (the first near-normal shock of the train). NaN when the core stays supersonic to the exit (no train)."""
    i = np.where(M_center < 1.0)[0]
    return float(x[i[0]]) if len(i) else float("nan")


def run_duct(pb_ratio, shape=(480, 40), Re=2e4, t_end=45.0, three_d=False, tag=""):
    nd = len(shape)
    cache = RES / f"case_{tag}pb{pb_ratio:.2f}_{'x'.join(map(str, shape))}.json"
    if cache.exists():                                  # finished before an interruption
        return json.loads(cache.read_text())
    lengths = (L, H) if nd == 2 else (L, H, H)
    inflow = (1.0, M_IN) + (0.0,) * (nd - 1) + (1 / G,)
    bc = {"x-": "supersonic_inflow", "x+": "pressure_outlet", "y-": "wall", "y+": "wall"}
    if nd == 3:
        bc.update({"z-": "wall", "z+": "wall"})
    cfg = CompressibleConfig(shape=shape, lengths=lengths, mu_ref=M_IN / Re, inflow=inflow, p_out=pb_ratio / G, bc=bc,
                             device=DEV, dtype=torch.float32, limiter="minmod", cfl=0.4)
    s = CompressibleFV(cfg)
    W = np.zeros((nd + 2,) + tuple(shape), np.float32)
    W[0] = 1.0; W[1] = M_IN; W[-1] = 1 / G
    s.set_state(W)
    x = (np.arange(shape[0]) + 0.5) * L / shape[0]
    jc = shape[1] // 2
    hist = {"t": [], "x_s": [], "x_sonic": [], "residual": []}
    t0 = time.time()
    next_rec = 0.0
    while s.t < t_end:
        U_old = s.W[s.interior()][0].clone()
        dt = s.step()
        if s.t >= next_rec:
            Wi = s.W[s.interior()]
            pc = (Wi[-1][:, jc] if nd == 2 else Wi[-1][:, jc, jc]).float().cpu().numpy()
            Mn = s.mach()
            mc = (Mn[:, jc] if nd == 2 else Mn[:, jc, jc]).float().cpu().numpy()
            hist["x_sonic"].append(sonic_front(mc, x))
            res = float(((Wi[0] - U_old).abs().mean()) / dt)
            hist["t"].append(s.t); hist["x_s"].append(leading_edge(pc, x, 1 / G)); hist["residual"].append(res)
            next_rec += 0.5
            if not np.isfinite(res):
                raise FloatingPointError(f"diverged at t={s.t}")
    Wi = s.W[s.interior()].float().cpu().numpy()
    Mach = s.mach().float().cpu().numpy()
    out = {"pb_ratio": pb_ratio, "shape": list(shape), "Re": Re, "t_end": s.t, "seconds": time.time() - t0, "history": hist}
    xs_tail = np.array(hist["x_s"][-10:])
    out["x_s_final"] = hist["x_s"][-1]
    xo = np.array(hist["x_sonic"][-10:])
    out["x_sonic_final"] = hist["x_sonic"][-1]
    out["x_sonic_last10_range"] = float(np.nanmax(xo) - np.nanmin(xo)) if np.isfinite(xo).any() else None
    out["no_train"] = bool(not np.isfinite(out["x_sonic_final"]))
    out["x_s_last5_range"] = float(np.nanmax(xs_tail) - np.nanmin(xs_tail)) if np.isfinite(xs_tail).any() else None
    if nd == 2:
        out["p_wall"] = Wi[-1][:, 0].tolist(); out["p_center"] = Wi[-1][:, jc].tolist(); out["M_center"] = Mach[:, jc].tolist()
        np.savez_compressed(RES / f"field_{tag}pb{pb_ratio:.2f}_{shape[0]}x{shape[1]}.npz", W=Wi, Mach=Mach, x=x)
    else:
        np.savez_compressed(RES / f"field3d_{tag}pb{pb_ratio:.2f}.npz", Mach_top=Mach[:, :, shape[2] // 2], Mach_side=Mach[:, shape[1] // 2, :],
                            p_center=Wi[-1][:, jc, jc], x=x)
        out["p_center"] = Wi[-1][:, jc, jc].tolist(); out["M_center"] = Mach[:, jc, jc].tolist()
    unstart = np.isfinite(out["x_sonic_final"]) and out["x_sonic_final"] < 0.3   # train pushed to the inlet
    out["unstarted"] = bool(unstart)
    cache.write_text(json.dumps(out))
    log(f"[{tag}pb={pb_ratio}] shape={shape} x_sonic={out['x_sonic_final']:.3f} (last-10 range {out['x_sonic_last10_range']}) x_s(first attempt)={out['x_s_final']:.3f} (last-5 range {out['x_s_last5_range']}) {out['seconds']:.0f}s")
    return out


def verify():
    # inviscid quasi-1-D standing normal shock: M1 = 2 -> theory M2 = 0.5774, rho2/rho1 = 2.6667, p2/p1 = 4.5
    cfg = CompressibleConfig(shape=(400, 4), lengths=(4.0, 0.04), inflow=(1.0, M_IN, 0.0, 1 / G), p_out=4.5 / G,
                             bc={"x-": "supersonic_inflow", "x+": "pressure_outlet", "y-": "periodic", "y+": "periodic"},
                             device="cpu", dtype=torch.float64, limiter="vanleer")
    s = CompressibleFV(cfg)
    x = (np.arange(400) + 0.5) * 0.01
    W = np.zeros((4, 400, 4)); W[0] = 1; W[1] = M_IN; W[3] = 1 / G
    xs0 = 2.0                                           # initialise with the exact jump at x = 2 (neutrally stable position)
    m = x > xs0
    W[0][m] = 2.6667; W[1][m] = M_IN / 2.6667; W[3][m] = 4.5 / G
    s.set_state(W)
    while s.t < 30:
        s.step()
    Wi = s.W[s.interior()][:, :, 0].numpy()
    down = Wi[:, -20:].mean(1)
    M2 = down[1] / np.sqrt(G * down[3] / down[0])
    xs = x[np.argmax(np.abs(np.diff(Wi[3])))]
    th = {"M2": ((1 + 0.2 * 4) / (1.4 * 4 - 0.2)) ** 0.5, "rho_ratio": 2.4 * 4 / (0.4 * 4 + 2), "p_ratio": 1 + 2 * 1.4 / 2.4 * 3}
    out = {"M2": float(M2), "rho_ratio": float(down[0]), "p_ratio": float(down[3] * G), "theory": th, "shock_x_after_t30": float(xs)}
    log("verify:", out)
    (RES / "verify.json").write_text(json.dumps(out, indent=1))


def main(stage):
    if stage == "verify":
        verify()
    elif stage == "sweep":
        out = [run_duct(pb, tag="sweep_") for pb in P_SWEEP]
        (RES / "sweep.json").write_text(json.dumps(out, indent=1))
    elif stage == "grid":
        out = [run_duct(3.0, shape=sh, tag="grid_") for sh in ((240, 20), (360, 30), (540, 45))]
        (RES / "grid.json").write_text(json.dumps(out, indent=1))
    elif stage == "duct3d":
        out = run_duct(3.0, shape=(360, 30, 30), Re=1e4, t_end=40.0, tag="3d_")
        (RES / "duct3d.json").write_text(json.dumps(out, indent=1))
    elif stage.startswith("duct3d:"):
        # second 3-D case: at p_b/p_in = 3 the square duct (four walls, Re 1e4) pushed the train almost to the inlet,
        # so a lower back pressure is run to show a train inside the duct as in the target picture
        pb = float(stage.split(":")[1])
        out = run_duct(pb, shape=(360, 30, 30), Re=1e4, t_end=40.0, tag="3d_")
        (RES / f"duct3d_pb{pb:.2f}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    for st in sys.argv[1:]:
        main(st)
