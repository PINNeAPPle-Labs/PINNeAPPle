"""The ascent model against its references: the US Standard Atmosphere 1976, the rotation of the pad, the Tsiolkovsky
velocity of each stage, the velocity bookkeeping of the flight, the inserted orbit against the target and against an
independent element solver, and one revolution of the orbit (energy, momentum, period).

    python examples/rocket_launch/01_verification.py            (about 30 s)

Writes results/verification.json. Every number printed comes from a check with a stated reference.
"""
import json
import os

import pinneapple as pp
from pinneapple_simulation.launch import Site, Target, simulate_ascent, small_launcher, verification

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

SITE = Site("Tanegashima", 30.4, 130.97)
TARGET = Target(500.0)

vehicle = small_launcher()
print(f"{vehicle.name}: liftoff {vehicle.liftoff_mass_kg:,.0f} kg, payload {vehicle.payload_kg:.0f} kg "
      f"({100 * vehicle.payload_fraction:.2f} %), ideal velocity {vehicle.delta_v_vac_m_s:.0f} m/s, "
      f"T/W {vehicle.thrust_to_weight_liftoff:.2f}")
for st, dv, above in zip(vehicle.stages, vehicle.stage_delta_v_m_s, vehicle.stage_masses_above, strict=True):
    print(f"  {st.name}: {st.propellant_kg:7.0f} kg propellant, {above:6.0f} kg above, Isp {st.isp_vac_s:.0f} s, "
          f"ideal {dv:6.0f} m/s, thrust {st.thrust_vac_n / 1e3:6.1f} kN (vac)")

hist = simulate_ascent(vehicle, SITE, TARGET, dt=0.1, t_max=1500.0)
rep = verification.run(hist, site=SITE, target=TARGET)
json.dump(pp.to_jsonable(rep), open(os.path.join(RESULTS, "verification.json"), "w"), indent=1)

for name, group in rep["groups"].items():
    print(f"\n{name}: {'PASS' if group['pass'] else 'FAIL'}")
    for c in group["checks"]:
        ref = c.get("reference")
        ref_s = f"vs {ref:12.6g}" if isinstance(ref, (int, float)) else ""
        print(f"  {'ok ' if c.get('pass', True) else 'FAIL'} {c['name']:34s} {c.get('value', float('nan')):12.6g} "
              f"{c.get('unit', ''):7s} {ref_s:20s} rel {c.get('rel_error', 0.0):.2e}")
print(f"\n{rep['n_checks']} checks, {rep['n_failed']} failed")
assert rep["pass"], rep["failed"]
