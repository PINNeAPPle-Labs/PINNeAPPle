"""Pulsatile flow in the library's aortic arch with three-element Windkessel outlets (OpenFOAM pimpleFoam).

    python examples/biology/aorta_windkessel.py --out runs/aorta_wk [--cycles 4] [--cell 1.5e-3] [--carreau]

Inflow: half-sine ejection of 70 mL in 0.3 s every 1 s (plug profile at the aortic annulus). Outlets: one RCR per
outlet from ``split_rcr`` (Murray d^3 flow split, 93 mmHg mean pressure, 9 % proximal resistance, total compliance
1e-8 m^3/Pa): round adult values for a demonstration, not patient data. Blood Newtonian (1060 kg/m^3, 3.5 mPa s) or
Carreau (Cho and Kensey 1991) with ``--carreau``. Laminar on a snappyHexMesh mesh (no wall layers).

Checks reported (last cycle): mass imbalance per window; cycle-mean outlet flow fractions against the 0-D
resistance split; cycle-mean outlet pressure against Q_mean (Rp + Rd); systolic / diastolic inlet pressure; how far
the outlet pressures are from their Windkessels (explicit coupling); cycle-to-cycle change of the inlet pressure
(periodicity). No published or Vascular Model Repository case is compared: vascularmodel.com was refused by the
network of the session that wrote this.
Writes <out>/result.json and <out>/waveforms.png.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from pinneapple_simulation.hemodynamics import MMHG, half_sine_inflow, outlet_flows, split_rcr
from pinneapple_simulation.hemodynamics.aorta import CoupledRun, setup_case


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--cell", type=float, default=1.5e-3)
    ap.add_argument("--carreau", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    case = out / "case"
    period = 1.0
    if not (case / "setup.json").exists():
        setup_case(case, cell_size=a.cell, level=1, inflow=half_sine_inflow, period=period, cycles=a.cycles,
                   viscosity="carreau" if a.carreau else "newtonian")
    info = json.loads((case / "setup.json").read_text())
    diam = {k: v["diameter"] for k, v in info["geometry"]["caps"].items() if k.startswith("outlet")}
    wk = split_rcr(diam)
    run = CoupledRun(case, wk, rho=info["physics"]["rho"])
    hist_file = out / "history.json"
    if hist_file.exists():
        run.history = json.loads(hist_file.read_text())
        for k, w in wk.items():                     # restore the Windkessel states from the last window
            r = run.history[-1]
            w.Pc = r[f"P_wk_{k}"] - w.Rp * r[f"Q_{k}"]
    for c in range(len(run.history) // 100, a.cycles):
        run.run((c + 1) * period, verbose=False)
        hist_file.write_text(json.dumps(run.history))
        print(f"cycle {c + 1}: p_inlet range "
              f"{min(r['p_inlet'] for r in run.history[-100:]) / MMHG:.1f}-"
              f"{max(r['p_inlet'] for r in run.history[-100:]) / MMHG:.1f} mmHg", flush=True)
    chk = run.check(period)
    names = list(wk)
    q0 = outlet_flows(chk["Q_inlet_mean"], [wk[k].Rp + wk[k].Rd for k in names])
    for k, q in zip(names, q0, strict=True):
        o = chk["outlets"][k]
        o["flow_fraction_0d"] = float(q / chk["Q_inlet_mean"])
        o["p_mean_0d_mmHg"] = float(o["Q_mean"] * (wk[k].Rp + wk[k].Rd) / MMHG)
        o["p_mean_mmHg"] = o["p_mean"] / MMHG
    h = run.history
    pin = np.array([r["p_inlet"] for r in h]) / MMHG
    last, prev = pin[-100:], pin[-200:-100]
    res = {"mesh": info["mesh"], "physics": info["physics"], "cycles": a.cycles, "check": chk,
           "systolic_mmHg": float(last.max()), "diastolic_mmHg": float(last.min()),
           "mean_inlet_mmHg": float(last.mean()),
           "periodicity_max_change_mmHg": float(np.max(np.abs(last - prev))) if len(prev) == 100 else None,
           "windkessels": {k: {"Rp": w.Rp, "Rd": w.Rd, "C": w.C} for k, w in wk.items()}}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ("mesh", "physics", "windkessels")}, indent=1,
                     default=float))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = np.array([r["t"] for r in h])
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(t, pin, label="inlet")
        for k in names:
            ax[0].plot(t, np.array([r[f"p3d_{k}"] for r in h]) / MMHG, label=k)
            ax[1].plot(t, np.array([r[f"Q_{k}"] for r in h]) * 1e6, label=k)
        ax[1].plot(t, np.array([r["Q_inlet"] for r in h]) * 1e6, "k", label="inlet")
        ax[0].set(xlabel="t [s]", ylabel="pressure [mmHg]")
        ax[1].set(xlabel="t [s]", ylabel="flow [mL/s]")
        ax[0].legend(fontsize=7)
        ax[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "waveforms.png", dpi=110)
    except ImportError:
        pass
    return res


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
