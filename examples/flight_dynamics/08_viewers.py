"""Rebuild the interactive three.js viewers and their screenshots from the runs saved by 03_takeoff.py,
04_turbulence_encounters.py and 07_same_gust.py, and copy a few screenshots into docs/assets/flight_dynamics/.

    python examples/flight_dynamics/08_viewers.py
    python -m http.server -d examples/flight_dynamics/_out/a320_turbulence/viewer      # then open localhost:8000
"""
import os
import pickle
import shutil

from pinneapple_simulation.flight import catalog
from pinneapple_simulation.flight.visuals import flight_scene
from pinneapple_tools.visualization.studio.web import screenshot, web_viewer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_out")
ASSETS = os.path.join(HERE, "..", "..", "docs", "assets", "flight_dynamics")
CASES = {"takeoff": "take-off", "crosswind": "15 kt crosswind take-off", "gust": "down-gust at rotation",
         "turbulence": "severe turbulence", "turbulence_moderate": "moderate turbulence", "gust_same": "1-cosine gust"}
PUBLISH = {"a320_takeoff": 110, "light_ga_crosswind": 70, "a320_turbulence": 120,
           "regional_turboprop_turbulence": 120, "motor_glider_turbulence_moderate": 120}

for name in catalog.names():
    ac = catalog.build(name)
    for case, title in CASES.items():
        f = os.path.join(OUT, f"{name}_{case}", "run.pkl")
        if not os.path.exists(f):
            continue
        run = pickle.load(open(f, "rb"))
        flying = case.startswith(("turbulence", "gust_same"))
        sc = flight_scene(ac, run["hist"], n_frames=200, wind=run["wind"], runway=not flying,
                          wisps=flying, title=f"{ac.info['type']}: {title}")
        d = os.path.join(OUT, f"{name}_{case}", "viewer")
        web_viewer(sc, d)
        key = f"{name}_{case}"
        png = screenshot(d, os.path.join(OUT, key, "viewer.png"), frame=PUBLISH.get(key, 120))
        if key in PUBLISH and png:
            shutil.copy(png, os.path.join(ASSETS, f"viewer_{key}.png"))
        print(key, png, flush=True)
