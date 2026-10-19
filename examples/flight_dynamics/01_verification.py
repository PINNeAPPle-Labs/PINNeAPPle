"""Verification of the flight-dynamics library: turbulence spectra and variances against MIL-F-8785C, trim of the
four aircraft held for 10 s stick fixed, energy / angular-momentum conservation of the integrator, the
lattice-in-the-loop lift against the steady vortex lattice. Writes results/verification.json and the PSD figure.

    python examples/flight_dynamics/01_verification.py          (about 2 minutes)
"""
import json
import os

import pinneapple as pp
from pinneapple_simulation.flight import verification
from pinneapple_simulation.flight.figures import psd_figure

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "..", "docs", "assets", "flight_dynamics")
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
os.makedirs(ASSETS, exist_ok=True)

rep = verification.run()
curves = rep.pop("_curves")
psd_figure(curves[:3] + curves[-2:], os.path.join(ASSETS, "turbulence_psd.png"))
json.dump(pp.to_jsonable(rep), open(os.path.join(HERE, "results", "verification.json"), "w"), indent=1)
for c in rep["turbulence"]["cases"]:
    print(f"{c['generator']:14s} {c['model']:10s} {c['component']}: sigma {c['sigma']:.3f} / {c['sigma_target']:.3f}"
          f" m/s, PSD ratio {c['psd_ratio_min']:.2f}..{c['psd_ratio_max']:.2f}")
for r in rep["trim"]:
    print(f"{r['aircraft']:18s} trim residual {r['residual']:.1e}, 10 s hold: dV {r['dV_max']:.3f} m/s, "
          f"dh {r['dh_max']:.3f} m, dnz {r['dnz_max']:.1e}")
