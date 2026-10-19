"""The four aircraft through the same gust: a vertical 1-cosine up-gust of 7.62 m/s (25 ft/s) over 2H = 100 m at
3000 m, each at its own cruise speed, stick fixed. Writes results/same_gust.json (peak load factor at the CG and in
the cabin, pitch and height excursions, the quasi-steady and Pratt estimates for this gust) and saves the runs for
the side-by-side render of 05_render_clips.py.

    python examples/flight_dynamics/07_same_gust.py      (about a minute)
"""
import json
import math
import os
import pickle

import numpy as np

import pinneapple as pp
from pinneapple_simulation.flight import catalog
from pinneapple_simulation.flight.control import StickFixed
from pinneapple_simulation.flight.encounter import gust_sensitivity
from pinneapple_simulation.flight.simulate import simulate
from pinneapple_simulation.flight.trim import stability_derivatives, trim
from pinneapple_simulation.flight.turbulence import DiscreteGust, pratt_gust_load
from pinneapple_simulation.flight.verification import ENCOUNTER_SPEED

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_out")
H_ALT, U, HALF = 3000.0, 7.62, 50.0
LEAD = 1.5                                       # s of level flight before the gust

res = {"gust": {"amplitude_m_s": U, "half_length_m": HALF, "altitude_m": H_ALT, "shape": "full 1-cosine, up"},
       "aircraft": {}}
for name in catalog.names():
    ac = catalog.build(name)
    V = ENCOUNTER_SPEED[name]
    tr = trim(ac, V, H_ALT)
    d = stability_derivatives(ac, tr)
    g = DiscreteGust(amplitude=U, length=HALF, start=LEAD * V, shape="full")
    T = LEAD + 2 * HALF / V + 6.0
    h = simulate(ac, tr.x, StickFixed(tr.controls), T, 0.005, wind=g, record_gamma=True, record_every=2)
    rho = ac.atmosphere(tr.x)[0]
    sig = rho / 1.225
    p = pratt_gust_load(ac.wing_loading, d["CL_alpha"], ac.af.mac, U * math.sqrt(sig), V * math.sqrt(sig), rho=rho)
    dn = h.nz - h.nz[0]
    r = {"V": V, "wing_loading": ac.wing_loading, "cl_alpha": d["CL_alpha"], "gust_time_s": 2 * HALF / V,
         "dnz_cg_max": float(dn.max()), "dnz_cg_min": float(dn.min()),
         "t_peak_s": float(h.t[int(np.argmax(dn))] - LEAD),
         "theta_max_deg": float(np.degrees(np.abs(h.euler[:, 1] - h.euler[0, 1]).max())),
         "dh_max_m": float(np.abs(h.altitude - H_ALT).max()),
         "cabin_dnz_max": {k: float((v - v[0]).max()) for k, v in h.cabin_nz.items()},
         "quasi_steady_dn": gust_sensitivity(ac, V, rho, d["CL_alpha"]) * U, "pratt_dn_for_this_gust": p["dn"],
         "pratt_K_g": p["K_g"], "mass_ratio": p["mu"]}
    res["aircraft"][name] = r
    os.makedirs(os.path.join(OUT, f"{name}_gust_same"), exist_ok=True)
    pickle.dump({"hist": h, "wind": g}, open(os.path.join(OUT, f"{name}_gust_same", "run.pkl"), "wb"))
    print(f"{name:18s} V {V:5.0f} m/s  gust {r['gust_time_s']:.2f} s  dnz max {r['dnz_cg_max']:+.3f}  "
          f"(quasi-steady {r['quasi_steady_dn']:.3f}, Pratt K_g {p['K_g']:.2f}: {p['dn']:.3f})  "
          f"theta {r['theta_max_deg']:.2f} deg")
json.dump(pp.to_jsonable(res), open(os.path.join(HERE, "results", "same_gust.json"), "w"), indent=1)
