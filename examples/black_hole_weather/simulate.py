"""Generate the accretion-flow simulations used to train and test the black-hole weather forecaster (issue #399).

Each run starts from an equilibrium torus around a Schwarzschild black hole (Paczynski-Wiita potential) and
evolves the viscous, radiatively inefficient flow that forms, saving the primitives every ``--every`` M.
Runs checkpoint their state, so an interrupted run resumes where it stopped:

    python examples/black_hole_weather/simulate.py --out data/bh --name PL0SS0.1 --alpha 0.1 --viscosity SS
    python examples/black_hole_weather/simulate.py --out data/bh --name PL0SS0.3 --alpha 0.3 --viscosity SS

Output per run (folder ``<out>/<name>``): ``frames_XXXX.npy`` chunks of (n, 5, nr, ntheta) float32 primitives
(rho, v_r, v_theta, v_phi, p), ``times.npy``, ``mdot.npy``, ``grid.npz``, ``config.json`` and ``state.pt``.
``load_run`` reads a run back as one array.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from pinneapple_physics.blackhole import AccretionFlow, RIAFConfig, torus_state


def load_run(folder: str):
    """-> dict(t, frames (T, 5, nr, ntheta), mdot, grid, config)."""
    chunks = sorted(glob.glob(os.path.join(folder, "frames_*.npy")))
    frames = np.concatenate([np.load(c) for c in chunks]) if chunks else np.zeros((0,))
    t = np.load(os.path.join(folder, "times.npy"))
    mdot = np.load(os.path.join(folder, "mdot.npy"))
    n = min(len(frames), len(t))
    with open(os.path.join(folder, "config.json")) as f:
        cfg = json.load(f)
    return {"t": t[:n], "frames": frames[:n], "mdot": mdot[:n], "grid": dict(np.load(os.path.join(folder, "grid.npz"))),
            "config": cfg}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--viscosity", default="SS")
    ap.add_argument("--torus-a", type=float, default=0.0)
    ap.add_argument("--nr", type=int, default=128)
    ap.add_argument("--ntheta", type=int, default=64)
    ap.add_argument("--t-end", type=float, default=10000.0)
    ap.add_argument("--every", type=float, default=10.0)
    ap.add_argument("--chunk", type=int, default=50, help="frames per saved chunk / checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    torch.set_num_threads(1)

    folder = os.path.join(a.out, a.name)
    os.makedirs(folder, exist_ok=True)
    cfg = RIAFConfig(nr=a.nr, ntheta=a.ntheta, alpha=a.alpha, viscosity=a.viscosity, torus_a=a.torus_a, seed=a.seed)
    flow = AccretionFlow(cfg)
    state_path = os.path.join(folder, "state.pt")
    if os.path.exists(state_path):
        st = torch.load(state_path, weights_only=False)
        flow.W.copy_(st["W"])
        flow.t, flow.boundary, flow.mdot_in = st["t"], st["boundary"], st["mdot"]
        times, mdot, k_chunk = list(st["times"]), list(st["mdots"]), st["k_chunk"]
        print(f"resuming {a.name} at t = {flow.t:.1f} M ({len(times)} frames)", flush=True)
    else:
        flow.set_primitives(torus_state(flow))
        times, mdot, k_chunk = [0.0], [0.0], 0
        np.save(os.path.join(folder, f"frames_{k_chunk:04d}.npy"), flow.primitives()[None].astype(np.float32))
        k_chunk += 1
        np.savez(os.path.join(folder, "grid.npz"), **flow.grid())
        with open(os.path.join(folder, "config.json"), "w") as f:
            json.dump({**{k: v for k, v in asdict(cfg).items() if k not in ("dtype",)}, "every": a.every,
                       "t_end": a.t_end, "name": a.name}, f, indent=1)

    buf, t0 = [], time.time()
    t_next = times[-1] + a.every
    while flow.t < a.t_end - 1e-9:
        flow.step(min(flow.dt(), t_next - flow.t))
        if not np.isfinite(flow.mdot_in):
            raise FloatingPointError(f"{a.name}: non-finite state at t = {flow.t:.2f}")
        if flow.t >= t_next - 1e-9:
            buf.append(flow.primitives().astype(np.float32))
            times.append(flow.t)
            mdot.append(flow.mdot_in)
            t_next += a.every
            if len(buf) == a.chunk or flow.t >= a.t_end - 1e-9:
                np.save(os.path.join(folder, f"frames_{k_chunk:04d}.npy"), np.stack(buf))
                k_chunk += 1
                buf = []
                np.save(os.path.join(folder, "times.npy"), np.asarray(times))
                np.save(os.path.join(folder, "mdot.npy"), np.asarray(mdot))
                torch.save({"W": flow.W.clone(), "t": flow.t, "boundary": dict(flow.boundary), "mdot": flow.mdot_in,
                            "times": times, "mdots": mdot, "k_chunk": k_chunk}, state_path)
                el = time.time() - t0
                print(f"{a.name}: t = {flow.t:8.1f} M  frames {len(times)}  mdot {flow.mdot_in:.4g}  "
                      f"capped {flow.capped}  wall {el/60:.1f} min", flush=True)
    print(f"{a.name}: done", flush=True)


if __name__ == "__main__":
    main()
