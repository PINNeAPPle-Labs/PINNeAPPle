"""The four aircraft in the same severe turbulence: load-factor increments, roll and pitch excursions, cabin
accelerations (RMS and peaks), related to wing loading and lift-curve slope (quasi-steady gust sensitivity, Pratt).

    python examples/flight_dynamics/04_turbulence_encounters.py [--quick]   (about an hour; --quick: 1 seed, 30 s)

Turbulence: von Karman, MIL-F-8785C severe (probability of exceedance 1e-5) at 3000 m, the same sigma and scale
length for every aircraft, as a frozen 3-D random-mode field (spanwise variation); each aircraft at its own speed.
Runs stick fixed and with the autopilot (altitude, heading, speed hold), three field realisations each; one run per
aircraft with a frozen line (no spanwise variation) to show the roll the spanwise gradients cause; light and
moderate with the autopilot; a 1-cosine gust against the Pratt formula. Writes results/turbulence.json and figures.
"""
import json
import os
import pickle
import sys

import numpy as np

import pinneapple as pp
from pinneapple_simulation.flight import catalog
from pinneapple_simulation.flight.encounter import pratt_check, turbulence_encounter
from pinneapple_simulation.flight.turbulence import turbulence_spec
from pinneapple_simulation.flight.verification import ENCOUNTER_SPEED
from pinneapple_simulation.flight.visuals import flight_scene
from pinneapple_tools.visualization.studio.web import screenshot, web_viewer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_out")
ASSETS = os.path.join(HERE, "..", "..", "docs", "assets", "flight_dynamics")
QUICK = "--quick" in sys.argv
H = 3000.0
DURATION = 30.0 if QUICK else 120.0
SEEDS = [0] if QUICK else [0, 1, 2]
U_DE = 7.62                                     # 25 ft/s equivalent gust for the Pratt comparison (same for all)

spec = turbulence_spec(H, "severe")
res = {"spec": spec.as_dict(), "duration_s": DURATION, "seeds": SEEDS, "runs": [], "line_field": [], "intensity": [],
       "pratt": []}


def job(name, kind, ctl, seed, severity="severe", field="3d"):
    """One encounter (run in a worker process); keeps the run and its viewer for the autopilot, seed-0 case."""
    ac = catalog.build(name)
    V = ENCOUNTER_SPEED[name]
    if kind == "pratt":
        return kind, name, pratt_check(ac, V, H, U_DE)
    keep = kind == "main" and ctl == "autopilot" and seed == 0
    s = turbulence_spec(H, severity)
    h, m, tr, d, wind = turbulence_encounter(ac, V, H, s, duration=DURATION, controller=ctl, seed=seed, field=field,
                                             record_gamma=keep)
    m.update(seed=seed, severity=severity)
    if keep:
        d_out = os.path.join(OUT, f"{name}_turbulence")
        os.makedirs(d_out, exist_ok=True)
        pickle.dump({"hist": h, "wind": wind}, open(os.path.join(d_out, "run.pkl"), "wb"))
        sc = flight_scene(ac, h, n_frames=200, wind=wind, runway=False,
                          title=f"{ac.info['type']}: severe turbulence, {H:.0f} m")
        web_viewer(sc, os.path.join(d_out, "viewer"))
    print(f"{name:18s} {kind:9s} {ctl:11s} {severity:8s} seed {seed}: dnz rms {m['dnz_cg_rms']:.3f} "
          f"[{m['dnz_cg_min']:+.2f}, {m['dnz_cg_max']:+.2f}] phi rms {m['phi_rms_deg']:.1f} deg", flush=True)
    return kind, name, m


def save_moderate(name):
    """A moderate-turbulence run with the autopilot for the visuals, for an aircraft whose severe run departs."""
    ac = catalog.build(name)
    h, m, tr, d, wind = turbulence_encounter(ac, ENCOUNTER_SPEED[name], H, turbulence_spec(H, "moderate"),
                                             duration=DURATION, controller="autopilot", seed=0, record_gamma=True)
    d_out = os.path.join(OUT, f"{name}_turbulence_moderate")
    os.makedirs(d_out, exist_ok=True)
    pickle.dump({"hist": h, "wind": wind}, open(os.path.join(d_out, "run.pkl"), "wb"))
    web_viewer(flight_scene(ac, h, n_frames=200, wind=wind, runway=False,
                            title=f"{ac.info['type']}: moderate turbulence, {H:.0f} m"), os.path.join(d_out, "viewer"))
    print(name, "moderate (visuals):", {k: m[k] for k in ("dnz_cg_rms", "phi_rms_deg", "departed")}, flush=True)


if __name__ == "__main__" and "--moderate-visuals" in sys.argv:
    # python 04_turbulence_encounters.py --moderate-visuals motor_glider
    for n in sys.argv[sys.argv.index("--moderate-visuals") + 1:]:
        save_moderate(n)
elif __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    names = catalog.names()
    jobs = [(n, "main", c, sd, "severe", "3d") for n in names for c in ("stick_fixed", "autopilot") for sd in SEEDS]
    jobs += [(n, "line", "stick_fixed", 0, "severe", "line") for n in names]
    jobs += [(n, "intensity", "autopilot", 0, sev, "3d") for n in names for sev in ("light", "moderate")]
    jobs += [(n, "pratt", "", 0, "severe", "3d") for n in names]
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as ex:
        done = list(ex.map(job, *zip(*jobs, strict=True)))
    order = {n: i for i, n in enumerate(names)}
    for kind, _, m in sorted(done, key=lambda r: order[r[1]]):
        {"main": res["runs"], "line": res["line_field"], "intensity": res["intensity"], "pratt": res["pratt"]}[kind].append(m)
    for name in catalog.names():
        screenshot(os.path.join(OUT, f"{name}_turbulence", "viewer"),
                   os.path.join(OUT, f"{name}_turbulence", "viewer.png"), frame=120)

    # mean and spread over the realisations
    summary = []
    for name in catalog.names():
        for ctl in ("stick_fixed", "autopilot"):
            rows = [r for r in res["runs"] if r["aircraft"] == name and r["controller"] == ctl]
            keys = ("dnz_cg_rms", "dnz_cg_max", "dnz_cg_min", "phi_rms_deg", "phi_peak_deg", "theta_rms_deg",
                    "theta_peak_deg", "ny_cg_rms", "alpha_max_deg", "quasi_steady_dnz_rms", "wind_w_rms")
            s = {"aircraft": name, "controller": ctl, "n": len(rows), "wing_loading": rows[0]["wing_loading"],
                 "cl_alpha": rows[0]["cl_alpha"], "V": rows[0]["V"], "gust_sensitivity": rows[0]["gust_sensitivity"]}
            for k in keys:
                v = np.array([r[k] for r in rows])
                s[k] = float(v.mean())
                s[k + "_spread"] = float(v.max() - v.min())
            s["cabin"] = {st: {q: float(np.mean([r["cabin"][st][q] for r in rows])) for q in rows[0]["cabin"][st]}
                          for st in rows[0]["cabin"]}
            summary.append(s)
    res["summary"] = summary
    json.dump(pp.to_jsonable(res), open(os.path.join(HERE, "results", "turbulence.json"), "w"), indent=1)
    # figures and tables: 06_tables.py (from this JSON and the saved runs)
