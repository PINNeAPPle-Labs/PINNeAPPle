"""The mission: size the vehicle so the ascent ends on a 500 km near-circular orbit from Tanegashima, fly it at a fine
step, and write the numbers.

The stage masses of ``small_launcher`` are not free parameters. They come out of the closure run here: the three
stages are sized from a velocity budget, the vehicle is flown, the apogee the second stage leaves is read back, and
the budget is scaled by what the transfer was short or long — until the transfer apogee *is* the target altitude and
the kick stage circularises there. The vehicle shipped in ``pinneapple_simulation.launch.vehicle.small_launcher``
(``DELTA_V_SPLIT``) is the converged result of exactly this calculation.

    python examples/rocket_launch/02_mission.py [--quick]        (--quick: skip the closure, fly the shipped design)

Writes results/mission.json, results/design.json (the closure's path) and two profiles in
docs/assets/rocket_launch/.
"""
import json
import os
import sys
import time

import numpy as np

import pinneapple as pp
from pinneapple_simulation.launch import Site, Target, simulate_ascent, small_launcher
from pinneapple_simulation.launch.ascent import GuidanceConfig
from pinneapple_simulation.launch.sizing import close, delta_v_required
from pinneapple_simulation.launch.vehicle import DELTA_V_SPLIT
from pinneapple_simulation.launch.visuals import flight_figures

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
ASSETS = os.path.join(HERE, "..", "..", "docs", "assets", "rocket_launch")
QUICK = "--quick" in sys.argv
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(ASSETS, exist_ok=True)

SITE = Site("Tanegashima", 30.4, 130.97)         # 30.400 N, 130.969 E: the Tanegashima Space Center
TARGET = Target(500.0)                           # circular, at the latitude of the site -> a due-east launch

shipped = small_launcher()
budget = delta_v_required(TARGET.altitude_km, SITE.lat_deg)
print(f"velocity budget: {budget['circular_speed_m_s']:.0f} m/s of orbital speed, less the {budget['pad_speed_m_s']:.0f}"
      f" m/s the pad already has, plus the losses -> about {budget['total_m_s']:.0f} m/s")
print(f"shipped design: {shipped.liftoff_mass_kg:,.0f} kg at lift-off, {shipped.payload_kg:.0f} kg payload "
      f"({100 * shipped.payload_fraction:.2f} %), ideal velocity {shipped.delta_v_vac_m_s:.0f} m/s")

# ------------------------------------------------------------------ the design, closed on the simulation
if QUICK:
    design = {"rows": [], "note": "--quick: the closure was not run; the shipped design was flown"}
    vehicle, cfg = shipped, GuidanceConfig()
else:
    t0 = time.time()
    print("\nclosing the velocity budget on the simulation (one knob: the total ideal velocity)...")
    vehicle, _, report = close(SITE, TARGET, payload_kg=200.0, dt=0.2, verbose=True)
    cfg = GuidanceConfig(pitch_end_altitude_km=report["pitch_end_altitude_km"],
                         pitch_exponent=report["pitch_exponent"])
    design = dict(report, seconds=round(time.time() - t0, 1),
                  shipped_delta_v_m_s=shipped.delta_v_vac_m_s,
                  shipped_split=list(DELTA_V_SPLIT))
    json.dump(pp.to_jsonable(design), open(os.path.join(RESULTS, "design.json"), "w"), indent=1)
    print(f"closure: {len(report['rows'])} simulations in {design['seconds']:.0f} s -> "
          f"{vehicle.liftoff_mass_kg:,.0f} kg, ideal {vehicle.delta_v_vac_m_s:.0f} m/s")

# ------------------------------------------------------------------ the flight, at a fine step
hist = simulate_ascent(vehicle, SITE, TARGET, guidance=cfg, dt=0.05, t_max=3000.0)
orb = hist.final_orbit()
s = hist.summary()
reserve = float(hist.events.get("propellant_fraction_left", 0.0))

print(f"\nstage 1 and 2 cut-off at T+{hist.events['separation_s'][-1]:.1f} s, transfer apogee "
      f"{hist.events['apogee_altitude_km']:.1f} km, {hist.events['coast_s']:.0f} s of coast, kick stage to "
      f"T+{s['cutoff_time_s']:.1f} s")
print(f"orbit: {orb['perigee_km']:.1f} x {orb['apogee_km']:.1f} km, e = {orb['e']:.5f}, "
      f"i = {orb['inclination_deg']:.2f} deg, period {orb['period_s'] / 60:.2f} min")
print(f"max q {s['max_q_pa'] / 1e3:.1f} kPa at {s['max_q_altitude_km']:.1f} km and Mach {s['max_mach']:.2f}; "
      f"max acceleration {s['max_acceleration_g']:.2f} g")
print("losses [m/s]: " + ", ".join(f"{k.replace('dv_', '').replace('_m_s', '')} {v:.0f}"
                                   for k, v in hist.losses.items()))

mission = {
    "site": {"name": SITE.name, "lat_deg": SITE.lat_deg, "lon_deg": SITE.lon_deg, "azimuth_deg": hist.azimuth_deg},
    "target": {"altitude_km": TARGET.altitude_km, "inclination_deg": orb["inclination_deg"]},
    "vehicle": vehicle.summary(),
    "design": {"params": {"pitch_end_altitude_km": cfg.pitch_end_altitude_km, "pitch_exponent": cfg.pitch_exponent},
               "closure": design},
    "flight": s,
    "orbit": {k: float(v) for k, v in orb.items()},
    "events": pp.to_jsonable(hist.events),
}
# the full telemetry (one sample per second) is a separate file: the record above stays readable in a diff
telemetry = {k: np.asarray(v).tolist() for k, v in hist.telemetry(step=max(1, int(round(1.0 / 0.05)))).items()}
with open(os.path.join(RESULTS, "telemetry.json"), "w") as f:
    json.dump(pp.to_jsonable(telemetry), f, separators=(",", ":"))
json.dump(pp.to_jsonable(mission), open(os.path.join(RESULTS, "mission.json"), "w"), indent=1)
figures = flight_figures(hist, ASSETS)
print("\nwrote", os.path.join(RESULTS, "mission.json"), "and", ", ".join(os.path.basename(f) for f in figures))
