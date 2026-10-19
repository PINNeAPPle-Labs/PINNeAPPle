"""Trim, stability derivatives, dynamic modes and the mass properties of the four aircraft.

    python examples/flight_dynamics/02_trim_and_stability.py    (under a minute)

Writes results/stability.json: per aircraft the trim at its cruise point, the non-dimensional derivatives from the
full model (vortex lattice + fin + fuselage), the eigenvalues of the 12-state linearisation, the mass, inertia, CG
and the sources / assumptions of every input.
"""
import json
import os

import pinneapple as pp
from pinneapple_simulation.flight import catalog
from pinneapple_simulation.flight.trim import classify_modes, modes, stability_derivatives, trim
from pinneapple_simulation.flight.verification import CRUISE

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}
for name in catalog.names():
    ac = catalog.build(name)
    V, h = CRUISE[name]
    tr = trim(ac, V, h)
    d = stability_derivatives(ac, tr)
    m = classify_modes(modes(ac, tr)["eigenvalues"])
    out[name] = {"type": ac.info["type"], "mass": ac.mass, "inertia_body": ac.inertia, "cg_mac": ac.info["cg_mac"],
                 "wing_area": ac.af.wing_area, "span": ac.af.span, "mac": ac.af.mac, "wing_loading": ac.wing_loading,
                 "cd0": ac.aero.cd0, "oswald_lattice": ac.aero.oswald, "trim": tr.as_dict(), "derivatives": d,
                 "modes": m, "info": ac.info, "gear": [{"name": g.name, "contact": g.contact, "k": g.stiffness,
                                                         "c": g.damping} for g in ac.gear]}
    print(f"{name:18s} m {ac.mass:7.0f} kg  W/S {ac.wing_loading:6.0f} Pa  CLa {d['CL_alpha']:.2f}  "
          f"SM {d['static_margin']:.2f}  Cmq {d['Cm_q']:.1f}  Clp {d['Cl_p']:.2f}  Cnb {d['Cn_beta']:.3f}  "
          f"modes {[round(o['wn'], 2) for o in m['oscillatory']]} rad/s")
json.dump(pp.to_jsonable(out), open(os.path.join(HERE, "results", "stability.json"), "w"), indent=1)
