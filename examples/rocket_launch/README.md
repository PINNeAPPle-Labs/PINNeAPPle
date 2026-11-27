# A launcher from Tanegashima to a 500 km orbit: an ascent, a 3-D flight and a mission-control display

A concept study in the same spirit as the AI-designed rocket demo that went around: a vehicle that carries **200 kg**
to a **near-circular orbit at 500 km**, launched due east from **Tanegashima** (30.400 N, 130.969 E), flown as a
three-degree-of-freedom ascent, drawn on a globe and replayed in a mission-control page.

Nothing here is a flight demonstration. The vehicle is a **concept**: every structural coefficient, impulse and
thrust-to-weight below is an assumption of this study, listed with what it stands on. What *is* verified is the
model: the atmosphere against the published 1976 tables, the pad's rotation, the stage velocities against
Tsiolkovsky, the inserted orbit against the target and against an independent element solver, and one revolution of
that orbit against energy and angular-momentum conservation. `01_verification.py` prints all 42 checks.

## The mission

| | |
|---|---|
| Site | Tanegashima, 30.400 N, 130.969 E (0 m) |
| Launch azimuth | 90.0 deg (due east: an inclination equal to the site's latitude, 30.40 deg) |
| Target | circular, 500 km; ideal orbital speed 7 616.6 m/s there |
| The pad already has | 400.7 m/s eastwards (`omega R cos(latitude)`) |
| Payload | 200 kg (1.6 % of the lift-off mass) |

The flight has three parts:

1. **Pitch program** (stages one and two): a 10 s vertical rise, then the thrust pitched above the local horizon on
   a schedule in altitude, from 90 deg at 1.5 km to 0 deg at 300 km (exponent 0.6: a gentle pull-over where the
   dynamic pressure is highest). The second stage leaves a **transfer orbit with its apogee at the target
   altitude**.
2. **Coast to apogee**: about five minutes of ballistic flight, thrust off.
3. **Circularisation** (kick stage): at apogee the last stage burns prograde along the velocity the circular orbit
   still lacks, and cuts off when the osculating perigee reaches the target altitude.

## The vehicle (a concept, all numbers assumed)

| Stage | Propellant | Dry | Isp (vac / sea level) | Thrust (vac) |
|---|---|---|---|---|
| S1 | 8 500 kg | 660 kg | 311 / 261 s | 212 kN |
| S2 | 2 570 kg | 301 kg | 348 / 306 s | 34.5 kN |
| S3 (kick) | 118 kg | 22 kg | 292 s | 3.9 kN |
| Payload | 200 kg + 60 kg fairing | | | |

The masses are not typed in: `sizing.size_vehicle` builds them from a velocity split, and `sizing.close` solves that
split on the simulation until the transfer apogee is the target altitude and the kick stage circularises there
(`DELTA_V_SPLIT` in `pinneapple_simulation/launch/vehicle.py` is the converged result). The assumptions that go in:

- **impulses**: 311 s of vacuum impulse for kerolox, 348 s for methalox, 292 s for a solid kick stage (the range the
  small-launcher literature quotes for each chemistry);
- **structural coefficients**: 0.072, 0.105, 0.16 dry over wet — the range of a pressure-fed upper stage and a solid
  motor;
- **thrust-to-weight**: 1.45 at lift-off (below 1.3 a launcher loses too much to gravity; above 1.7 it wastes
  structure);
- **drag**: `CD_MACH` in `vehicle.py`, the slender-body curve (0.20 subsonic, 0.48 at Mach 1.1, 0.20 above Mach 8) on
  a 1.30 m stack;
- **atmosphere**: US Standard Atmosphere 1976, dry, no wind (NASA-TM-X-74335).

## What the model measures on that flight

Quoted from `results/mission.json` (step 0.05 s):

- cut-off of the first two stages at T+376 s, transfer apogee 502 km, 297 s of coast, kick stage to T+748 s;
- **inserted orbit: 486 x 501 km, e = 0.001, i = 30.40 deg, period 94.4 min**;
- **max dynamic pressure 36.8 kPa** at 10.5 km (Mach 11.4 happens higher, at low density);
- **max acceleration 5.54 g**;
- losses: 1 808 m/s of gravity, 74 m/s of drag, 840 m/s of steering against an ideal 9 940 m/s of thrust velocity,
  which closes to 2 m/s.

## Files

| | |
|---|---|
| `01_verification.py` | 42 checks of the model against its references → `results/verification.json` |
| `02_mission.py` | the design closure and the fine flight → `results/mission.json`, `docs/assets/rocket_launch/*.png` |
| `03_flight_views.py` | two self-contained pages (three.js inlined): `_out/globe/index.html`, `_out/vehicle/index.html` |
| `04_mission_control.py` | `_out/mission_control.html`: ground-track map, telemetry charts and tiles on one time cursor |

Open the pages directly (`file://` works: everything is inlined) or serve them
(`python -m http.server -d examples/rocket_launch/_out`). The viewers have play/pause, a time slider, playback speed
and, on the chase view, a follow camera.

## What this does not establish

- The vehicle is not designed for real: no structural loads, no sloshing, no thermal, no mass growth, no margins.
- The guidance is a schedule plus a terminal burn, not a flight algorithm: no throttling, no winds, no dispersions,
  no navigation errors, no engine-out cases.
- The orbit is two-body: no J2 precession (the model can integrate J2, the inserted orbit does not carry it), no
  third bodies, no solar pressure, no drag in orbit.
- The 3-D pages are a drawing of the trajectory, not a render of a vehicle: the launcher is a mesh of revolution with
  the concept's length and diameter.
