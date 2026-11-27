"""The mission-control page: the ground track on a map, the telemetry charts and the tiles, on one time cursor.

    python examples/rocket_launch/04_mission_control.py         (about 40 s)

Writes _out/mission_control.html (one file, opens from anywhere) and the frames of the map under
_out/mission_frames/.
"""
import json
import os

from pinneapple_simulation.launch import Site, Target, simulate_ascent, small_launcher
from pinneapple_simulation.launch.ascent import GuidanceConfig
from pinneapple_simulation.launch.visuals import mission_control

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_out")
os.makedirs(OUT, exist_ok=True)

SITE = Site("Tanegashima", 30.4, 130.97)
TARGET = Target(500.0)

mission = os.path.join(HERE, "results", "mission.json")
cfg = GuidanceConfig()
if os.path.exists(mission):
    d = json.load(open(mission))["design"]["params"]
    cfg = GuidanceConfig(pitch_end_altitude_km=d["pitch_end_altitude_km"], pitch_exponent=d["pitch_exponent"])

hist = simulate_ascent(small_launcher(), SITE, TARGET, guidance=cfg, dt=0.1, t_max=1500.0)
path = mission_control(hist, os.path.join(OUT, "mission_control.html"), n_frames=110,
                       title="Concept launcher to 500 km: mission control")
print("mission control ->", path, f"({os.path.getsize(path) / 1e6:.1f} MB)")
