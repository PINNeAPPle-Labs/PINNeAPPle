"""The two 3-D views of the flight: the globe with the ascent drawn on it, and the launcher in a chase camera.

Each page is one self-contained HTML file (three.js inlined), so it opens from ``file://`` with no server: play,
scrub the timeline, change the playback speed, and on the chase view fly with the vehicle.

    python examples/rocket_launch/03_flight_views.py            (about 1 minute)

Writes _out/globe/index.html and _out/vehicle/index.html.
"""
import json
import os

import pinneapple as pp
from pinneapple_simulation.launch import Site, Target, simulate_ascent, small_launcher
from pinneapple_simulation.launch.ascent import GuidanceConfig
from pinneapple_simulation.launch.visuals import flight_scene

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_out")
os.makedirs(OUT, exist_ok=True)

SITE = Site("Tanegashima", 30.4, 130.97)
TARGET = Target(500.0)

mission = os.path.join(HERE, "results", "mission.json")
if os.path.exists(mission):                     # reuse the design the mission script converged to
    d = json.load(open(mission))["design"]["params"]
    cfg = GuidanceConfig(pitch_end_altitude_km=d["pitch_end_altitude_km"], pitch_exponent=d["pitch_exponent"])
else:
    cfg = GuidanceConfig()

hist = simulate_ascent(small_launcher(), SITE, TARGET, guidance=cfg, dt=0.1, t_max=1500.0)
orb = hist.final_orbit()
print(f"{orb['perigee_km']:.0f} x {orb['apogee_km']:.0f} km, e = {orb['e']:.4f}, cut-off at "
      f"T+{hist.events['cutoff_s']:.1f} s")

for view in ("globe", "vehicle"):
    sc = flight_scene(hist, view=view, n_frames=300)
    path = pp.viz.web_viewer(sc, os.path.join(OUT, view), single_file=True)
    size = os.path.getsize(path) / 1e6
    print(f"{view:8s} view -> {path}  ({size:.1f} MB, opens from file://)")
