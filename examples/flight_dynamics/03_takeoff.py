"""Take-off of the four aircraft: ground roll, rotation at V_R, lift-off, climb over the 35 ft screen; Raymer's
closed form with the same aircraft model; the A320-class run against published field lengths; a 15 kt crosswind
case and a 1-cosine down-gust during rotation for each aircraft; 3-D viewers with screenshots.

    python examples/flight_dynamics/03_takeoff.py [--quick]     (about 10 minutes; --quick: nominal runs only)

Writes results/takeoff.json, docs/assets/flight_dynamics/takeoff*.png and the viewers in _out/<case>/viewer/
(open with ``python -m http.server -d examples/flight_dynamics/_out/a320_takeoff/viewer``).
"""
import json
import math
import os
import pickle
import sys

import pinneapple as pp
from pinneapple_simulation.flight import catalog
from pinneapple_simulation.flight.figures import lateral_figure, takeoff_figure
from pinneapple_simulation.flight.takeoff import (
    raymer_variants,
    simulate_takeoff,
    takeoff_metrics,
)
from pinneapple_simulation.flight.turbulence import KT, DiscreteGust, ShearWind
from pinneapple_simulation.flight.visuals import flight_scene
from pinneapple_tools.visualization.studio.web import screenshot, web_viewer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_out")
ASSETS = os.path.join(HERE, "..", "..", "docs", "assets", "flight_dynamics")
QUICK = "--quick" in sys.argv
KIND = {"light_ga": "light", "motor_glider": "light", "regional_turboprop": "transport", "a320": "transport"}
CROSSWIND = 15 * KT                       # MIL-F-8785C log profile, 15 kt at 20 ft, from the right (090)
GUST = {"amplitude": 4.0, "length": 30.0}  # 1-cosine down-gust of 4 m/s over 60 m, met at the rotation point


def save_case(case, ac, hist, wind=None):
    d = os.path.join(OUT, case)
    os.makedirs(d, exist_ok=True)
    pickle.dump({"hist": hist, "wind": wind}, open(os.path.join(d, "run.pkl"), "wb"))
    sc = flight_scene(ac, hist, n_frames=160, wind=wind, title=f"{ac.info['type']}: {case.split('_', 1)[1]}")
    web_viewer(sc, os.path.join(d, "viewer"))
    return d


res = {"published": {}, "nominal": {}, "closed_form": {}, "crosswind": {}, "gust_at_rotation": {}}
runs, cross = {}, {}
for name in catalog.names():
    ac = catalog.build(name)
    res["published"][name] = ac.info["published"]
    h = simulate_takeoff(ac, kind=KIND[name], record_gamma=True)
    m = takeoff_metrics(ac, h)
    m["schedule"] = h.events["schedule"]
    res["nominal"][name] = m
    runs[name] = h
    rv = raymer_variants(ac, m, KIND[name])
    cf, cf_sim = rv["raymer_1p1Vs"], rv["raymer_sim_VLOF"]
    res["closed_form"][name] = {"raymer_1p1Vs": cf, "raymer_with_sim_VLOF_and_rotation_time": cf_sim,
                                "sim_over_raymer": m["distance_35ft_m"] / cf["total_35ft_m"],
                                "sim_over_raymer_same_VLOF": m["distance_35ft_m"] / cf_sim["total_35ft_m"]}
    print(f"{name:18s} ground roll {m['ground_roll_m']:6.0f} m  35 ft {m['distance_35ft_m']:6.0f} m  "
          f"(Raymer {cf['total_35ft_m']:6.0f} m)  V_LOF {m['V_LOF_kt']:5.1f} kt  theta_LOF {m['theta_LOF_deg']:4.1f}  "
          f"climb {m.get('climb_gradient_pct', float('nan')):4.1f} %")
    d = save_case(f"{name}_takeoff", ac, h)
    screenshot(os.path.join(d, "viewer"), os.path.join(d, "viewer.png"), frame=110)
    if QUICK:
        continue
    # crosswind from the right, log profile
    w = ShearWind(w20=CROSSWIND, from_deg=90.0)
    hc = simulate_takeoff(ac, kind=KIND[name], wind=w, record_gamma=True)
    mc = takeoff_metrics(ac, hc)
    res["crosswind"][name] = {"w20_kt": CROSSWIND / KT, "from_deg": 90.0, **mc}
    cross[name] = hc
    save_case(f"{name}_crosswind", ac, hc, w)
    # down-gust met where the nominal run reached V_R
    xR = float(h.events["x_R"][0])
    g = DiscreteGust(amplitude=GUST["amplitude"], length=GUST["length"], start=xR, shape="full",
                     direction=(0.0, 0.0, 1.0))
    hg = simulate_takeoff(ac, kind=KIND[name], wind=g, record_gamma=True)
    mg = takeoff_metrics(ac, hg)
    res["gust_at_rotation"][name] = {**GUST, "start_x_m": xR, "direction": "down", **mg,
                                     "delta_distance_35ft_m": mg.get("distance_35ft_m", math.nan) - m["distance_35ft_m"]}
    save_case(f"{name}_gust", ac, hg, g)
    print(f"   crosswind: max offset {mc['max_lateral_deviation_ground_m']:.2f} m, bank {mc['max_bank_deg']:.1f} deg,"
          f" rudder {mc['max_rudder_deg']:.1f} deg | gust: 35 ft at {mg.get('distance_35ft_m', math.nan):.0f} m, "
          f"tail clearance {mg['tail_clearance_min_m']:.2f} m")

takeoff_figure(runs, res["nominal"], os.path.join(ASSETS, "takeoff.png"))
if cross:
    lateral_figure(cross, os.path.join(ASSETS, "takeoff_crosswind.png"),
                   "15 kt crosswind")
ac = catalog.build("a320")
res["a320_vs_published"] = {
    "sim_all_engines_35ft_m": res["nominal"]["a320"]["distance_35ft_m"], "sim_mass_kg": ac.mass,
    "published": ac.info["published"]["a320_tofl"],
    "note": ("the certified take-off field length is the longer of the one-engine-inoperative distance and 1.15 x the "
             "all-engines distance to 35 ft, at 78 t; the simulation is all engines, no margin, at 73.5 t")}
json.dump(pp.to_jsonable(res), open(os.path.join(HERE, "results", "takeoff.json"), "w"), indent=1)
