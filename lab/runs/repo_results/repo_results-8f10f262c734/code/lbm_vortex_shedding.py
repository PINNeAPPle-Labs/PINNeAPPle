"""Vortex shedding with the D2Q9 LBM solver: Strouhal number of a cylinder and a NACA 4412.

- ``cylinder``: Re = 100 (on the diameter). Reference: Williamson's fit for the laminar shedding
  regime, St = 0.2665 - 1.018 / sqrt(Re) = 0.1647 at Re = 100 (C. H. K. Williamson, "Vortex
  dynamics in the cylinder wake", Annu. Rev. Fluid Mech. 28 (1996) 477-539). The channel here has
  no-slip walls and a uniform inlet; confinement (blockage D/H = 1/height) raises St relative to the
  unconfined value, so results are reported for more than one channel height.
- ``naca4412``: NACA 4412 at 20 deg angle of attack, Re = 500 on the chord (the Karman street of
  the case in ROADMAP.md §11). There is no single published Strouhal number for this exact case;
  the usual yardstick for bluff shedding is St ~ 0.15-0.2 on the projected height c*sin(alpha)
  (Fage & Johansen, Proc. R. Soc. A 116 (1927) 170-197, inclined flat plates), reported as such.

St comes from the dominant frequency of the cross-stream velocity at a probe in the wake,
after discarding the start-up transient.

    python -m benchmarks.lbm_vortex_shedding --case cylinder --steps 30000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from pinneapple_simulation.numerical_solvers.lbm import (
    LBMSolver, _d2q9_tensors, _lbm_step_2d, _macroscopic_2d, cylinder_mask, naca4_mask,
)


def run(case: str, steps: int, u_in: float = 0.1, scale: int = 20, height: int = 10):
    if case == "cylinder":
        D, Re = scale, 100.0
        nx, ny = 22 * D, height * D  # blockage D/H = 1/height
        cxc, cyc = 5 * D, ny / 2 + 0.5  # half-cell offset breaks the symmetry so shedding starts
        mask = cylinder_mask(nx, ny, cxc, cyc, D / 2)
        L, probe = D, (int(cxc + 2 * D), int(ny / 2))
        ref = 0.2665 - 1.018 / math.sqrt(Re)
    elif case == "naca4412":
        c, aoa, Re = 4 * scale, 20.0, 500.0
        nx, ny = 7 * c, 3 * c
        x_le, y_le = 1.5 * c, ny / 2 + 0.25 * c
        mask = naca4_mask(nx, ny, "4412", c, aoa, x_le=x_le, y_le=y_le)
        L = c * math.sin(math.radians(aoa))  # projected height
        probe = (int(x_le + 2.0 * c), int(y_le - 0.3 * c))
        ref = None
    else:
        raise ValueError(case)
    # LBMSolver defines Re on the channel height (ny - 2): convert so that nu = u_in * L / Re_case
    Re_channel = Re * (ny - 2) / (L if case == "cylinder" else c)
    solver = LBMSolver(nx=nx, ny=ny, Re=Re_channel, u_in=u_in, obstacle_mask=mask)
    dev = torch.device("cpu")
    cx, cy, w, opp = _d2q9_tensors(dev)
    solid = solver.solid
    rho0 = torch.ones(nx, ny)
    ux0 = torch.where(solid, 0.0, u_in)  # fluid at rest inside the obstacle: no start-up shock
    from pinneapple_simulation.numerical_solvers.lbm import _equilibrium_2d
    f = _equilibrium_2d(rho0, ux0, torch.zeros(nx, ny), cx, cy, w)
    series = np.empty(steps)
    t0 = time.time()
    for n in range(steps):
        f = _lbm_step_2d(f, solver.omega, solid, cx, cy, w, opp, u_in=u_in, rho_out=1.0, Cs=0.0)
        rho, ux, uy = _macroscopic_2d(f[:, probe[0]:probe[0] + 1, probe[1]:probe[1] + 1], cx, cy)
        series[n] = float(uy)
        if not math.isfinite(series[n]):
            raise FloatingPointError(f"diverged at step {n}")
    seconds = time.time() - t0
    s = series[steps // 2:] - series[steps // 2:].mean()
    spec = np.abs(np.fft.rfft(s * np.hanning(len(s))))
    freqs = np.fft.rfftfreq(len(s), d=1.0)
    k = int(np.argmax(spec[1:]) + 1)
    # parabolic peak interpolation for sub-bin frequency accuracy
    if 1 <= k < len(spec) - 1:
        a, b, g = np.log(spec[k - 1] + 1e-30), np.log(spec[k] + 1e-30), np.log(spec[k + 1] + 1e-30)
        k = k + 0.5 * (a - g) / (a - 2 * b + g)
    freq = k * (freqs[1] - freqs[0])
    St = freq * L / u_in
    out = {"case": case, "St": St, "reference_St": ref,
           "rel_diff_vs_reference": None if ref is None else (St - ref) / ref,
           "length_scale_lattice": L, "u_in": u_in, "Re": Re, "omega": solver.omega, "grid": [nx, ny],
           "steps": steps, "periods_analysed": (steps // 2) * freq, "probe_uy_amplitude": float(s.std() * math.sqrt(2)),
           "seconds": round(seconds, 1)}
    if case == "naca4412":
        out["reference_note"] = "no single published value; bluff-body yardstick St ~ 0.15-0.2 on c*sin(alpha)"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="cylinder", choices=["cylinder", "naca4412"])
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--scale", type=int, default=20)
    ap.add_argument("--height", type=int, default=10, help="channel height in diameters (cylinder)")
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    r = run(args.case, args.steps, scale=args.scale, height=args.height)
    r["blockage"] = 1.0 / args.height if args.case == "cylinder" else None
    print(json.dumps(r, indent=1))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out", f"lbm_vortex_shedding_{args.case}" + (f"_h{args.height}" if args.case == "cylinder" else "") + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(r, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
