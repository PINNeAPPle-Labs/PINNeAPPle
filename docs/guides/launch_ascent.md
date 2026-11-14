# Ascent to orbit: a launcher from the pad to a 500 km near-circular orbit

`pinneapple_simulation.launch` flies a launch vehicle from a pad to a target orbit: a three-degree-of-freedom ascent
(position, velocity and mass in an Earth-centred inertial frame) through the US Standard Atmosphere 1976, with a
pitch schedule, a coast to apogee and a circularisation burn, and the velocity bookkeeping that closes on the
Tsiolkovsky budget. The worked case is a concept vehicle carrying 200 kg from Tanegashima to a 500 km near-circular
orbit ([`examples/rocket_launch`](../examples/rocket_launch/README.md)).

```python
from pinneapple_simulation.launch import Site, Target, simulate_ascent, small_launcher

hist = simulate_ascent(small_launcher(), Site("Tanegashima", 30.4, 130.97), Target(500.0))
print(hist.final_orbit())          # a, e, i, perigee, apogee, period of the inserted orbit
print(hist.summary())              # max q, max acceleration, cut-off time, the losses
print(hist.losses)                 # thrust, gravity, drag, steering [m/s] and their closure
```

## The model

| Part | What it does | Where |
|---|---|---|
| Atmosphere | US Standard Atmosphere 1976 from the ground to 1000 km: seven lapse-rate layers to 84.852 km (geopotential), the upper table above it | `atmosphere.py` |
| Vehicle | stages with dry/propellant mass, vacuum and sea-level impulse and thrust, drag area; the slender-launcher `CD(Mach)` table | `vehicle.py` |
| Ascent | RK4 on 3-DOF in ECI; thrust along a commanded direction; gravity (J2 optional); drag on the air-relative velocity; the Earth's rotation under the pad | `ascent.py` |
| Sizing | the inverse Tsiolkovsky relation per stage, and the closure that solves the velocity budget on the simulation | `sizing.py` |
| Checks | atmosphere against the published table, pad rotation, elements against `pinneapple_physics.astro.orbits`, one revolution of the orbit, the velocity bookkeeping | `verification.py` |
| Displays | a globe with the ascent, a chase view of the launcher, and a mission-control page | `visuals.py` |

The guidance has three parts and no hidden magic: a pitch schedule in altitude (vertical rise, pull-over, thrust
along the local horizon above `pitch_end_altitude_km`), a **coast** to apogee after the second stage, and a
**circularisation** burn at apogee along the velocity the circular orbit still lacks, cut off on the osculating
perigee.

## The velocity bookkeeping closes

Every loss is integrated over the whole flight — including the coast — so

```
dv_thrust - dv_gravity - dv_drag - dv_steering = |v_cutoff| - |v_liftoff|
```

holds to the quadrature error of the integrator step (`hist.losses["closure_m_s"]`, a thousandth of the thrust
velocity at the default step). That identity is the useful check: gravity loss is the part of the thrust spent
holding altitude, steering loss the part spent turning the vector, and both are numbers a launch team quotes.
Note that the steering term is measured against the **inertial** velocity, so the first seconds of vertical thrust —
perpendicular to the pad's eastward 400 m/s — count there.

## Sizing the vehicle

```python
from pinneapple_simulation.launch.sizing import close, delta_v_required, size_vehicle

budget = delta_v_required(500.0, 30.4)         # orbital speed, less the pad's rotation, plus losses
veh = size_vehicle(200.0, delta_v=(3513.0, 5270.0, 950.0))
vehicle, hist, report = close(Site("Tanegashima", 30.4, 130.97), Target(500.0), verbose=True)
```

`close` reads the apogee the second stage leaves and scales the ideal velocity until that apogee *is* the target
altitude — the same loop a launch team runs when it closes a design against its losses. It takes three simulations
and about fifteen seconds for this mission.

## Displays

```python
import pinneapple as pp
from pinneapple_simulation.launch.visuals import flight_scene, mission_control

pp.viz.web_viewer(flight_scene(hist, view="globe"), "globe/", single_file=True)      # the Earth and the track
pp.viz.web_viewer(flight_scene(hist, view="vehicle"), "chase/", single_file=True)    # the launcher, up close
mission_control(hist, "mission_control.html")                                        # map, charts, tiles
```

Two scenes, because the globe (6 371 km) and the vehicle (18 m) are 350 000 apart: no single camera holds both. The
globe view draws the Natural Earth 1:110m land mask and the path coloured by speed; the chase view follows the
launcher with its plume and trail. Both pages are single files with three.js inlined: they open from `file://`.

## What the checks establish, and what they do not

`verification.run()` (42 checks, `examples/rocket_launch/01_verification.py`) establishes the model against its
references: the atmosphere to a few parts per million of the published table, the pad's rotation to machine
precision, the osculating elements against an independent solver (`pinneapple_physics.astro.orbits`, itself checked
against JPL Horizons in `tests/test_astro_accuracy.py`), energy and angular momentum over one revolution, and the
velocity bookkeeping. The target orbit is reached to a few kilometres of perigee.

It does **not** validate a vehicle: no structural loads, no sloshing, no thermal, no winds, no navigation, no engine
out, no dispersions. The vehicle of the example is a concept and every coefficient is an assumption, listed in the
example's README. A simulation is not a flight.
