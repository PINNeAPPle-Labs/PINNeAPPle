# Flight dynamics, turbulence and take-off

`pinneapple_simulation.flight` flies a rigid aircraft in six degrees of freedom. The wing and tail loads come from the
library's vortex lattice (`pinneapple_design.aero.vlm`), solved again at every step with the local air velocity at
each panel. The other parts are thrust, gravity, landing-gear contact with the runway, MIL-F-8785C / MIL-HDBK-1797
turbulence and gusts, trim, a simple pilot / autopilot, and animated 3-D scenes for the browser viewer and Blender.

```python
from pinneapple_simulation import flight

ac = flight.catalog.build("a320")                  # light_ga, regional_turboprop, a320, motor_glider
tr = flight.trim(ac, V=128.0, altitude=3000.0)     # alpha, elevator, throttle for steady level flight
d = flight.stability_derivatives(ac, tr)           # CL_alpha, Cm_q, Cl_p, Cn_beta, ... and the static margin

hist = flight.simulate_takeoff(ac)                 # brake release -> V_R -> lift-off -> 400 ft
m = flight.takeoff_metrics(ac, hist)               # ground roll, V_LOF, theta, distance to 35 ft, climb gradient
cf = flight.raymer_takeoff(ac)                     # closed-form estimate with the same aircraft model

spec = flight.turbulence_spec(3000.0, "severe")    # von Karman, MIL-F-8785C intensities and scale lengths
hist, metrics, *_ = flight.turbulence_encounter(ac, 128.0, 3000.0, spec, duration=60.0)

from pinneapple_tools.visualization.studio.web import web_viewer
web_viewer(flight.visuals.flight_scene(ac, hist), "viewer/")   # time slider, gust arrows, wing loading, HUD
```

Every number on this page was produced by the scripts in `examples/flight_dynamics/` and is stored in
`examples/flight_dynamics/results/*.json`. The tables are filled from those files by `06_tables.py`.
Where an input is an assumption, it is listed as one in `Aircraft.info["assumptions"]` and below.

## Gallery (Blender Cycles)

The four aircraft through the same 1-cosine gust. The translucent wisps are the frozen gust field, blue for down and
red for up. The wings carry the lattice loading. The inset is n_z against time. Clip:
[same_gust_grid.mp4](../assets/flight_dynamics/clips/same_gust_grid.mp4).

![Four aircraft through the same gust](../assets/flight_dynamics/clips/same_gust_grid.png)

| take-off (low chase camera, n_z trail) | turbulence (haze, gust wisps, n_z sparkline) |
|---|---|
| ![](../assets/flight_dynamics/clips/light_ga_takeoff.png) [mp4](../assets/flight_dynamics/clips/light_ga_takeoff.mp4) | ![](../assets/flight_dynamics/clips/light_ga_turbulence.png) [mp4](../assets/flight_dynamics/clips/light_ga_turbulence.mp4) |
| ![](../assets/flight_dynamics/clips/regional_turboprop_takeoff.png) [mp4](../assets/flight_dynamics/clips/regional_turboprop_takeoff.mp4) | ![](../assets/flight_dynamics/clips/regional_turboprop_turbulence.png) [mp4](../assets/flight_dynamics/clips/regional_turboprop_turbulence.mp4) |
| ![](../assets/flight_dynamics/clips/a320_takeoff.png) [mp4](../assets/flight_dynamics/clips/a320_takeoff.mp4) | ![](../assets/flight_dynamics/clips/a320_turbulence.png) [mp4](../assets/flight_dynamics/clips/a320_turbulence.mp4) |
| ![](../assets/flight_dynamics/clips/motor_glider_takeoff.png) [mp4](../assets/flight_dynamics/clips/motor_glider_takeoff.mp4) | ![](../assets/flight_dynamics/clips/motor_glider_turbulence.png) [mp4](../assets/flight_dynamics/clips/motor_glider_turbulence.mp4) (moderate: the severe runs depart, and even the moderate run spends 7 % of the time past the stall angle, see below) |

## The four aircraft

| key | type | geometry | mass, engines |
|---|---|---|---|
| `light_ga` | Cessna 172-class single piston | the aero optimizer's baseline `Airframe` | 1100 kg, 120 kW (the optimizer's calibrated C172-class defaults) |
| `regional_turboprop` | ATR 72-600-class twin turboprop | planform and gear of the `turboprop_atr72` geometry-library entry; its mesh for the visuals | 23,000 kg, 2 x 2,475 shp normal take-off power (ATR brochure 2023, factsheet 2020) |
| `a320` | A320-class twin turbofan | the aero optimizer's baseline `Airliner` | 73,500 kg (the optimizer's A320-class MTOW), 2 x 27,000 lbf (CFM56-5B4 rating, CFM / EASA TCDS E.003) |
| `motor_glider` | high-aspect-ratio self-launching motor glider | parametric `Airframe` with the Stemme S10's span (23 m) and area (18.7 m², AR 28.3) | 850 kg, 84.5 kW (Stemme S10 / S10-VT) |

Masses and inertias come from a component breakdown, with the CG put at 25 % MAC (30 % for the glider). The
component masses are the wing (the Raymer general-aviation or transport equation, as the aero optimizer uses them),
the tails, engines, fuel, payload or occupants, and the fuselage and systems. Zero-lift drag is the aero optimizer's
component build-up (fuselage, fin, nacelles, wing and tail friction with form factors, miscellaneous). Induced drag
uses K from the lattice's Trefftz-plane drag.

<!-- table:aircraft -->
| aircraft | mass [kg] | S [m²] | span [m] | W/S [Pa] | Ixx [kg m²] | Iyy | Izz |
|---|---|---|---|---|---|---|---|
| Light GA (C172 class) | 1100 | 16.2 | 11.02 | 666 | 1890 | 2905 | 4505 |
| Turboprop (ATR 72 class) | 23000 | 61.0 | 27.05 | 3698 | 288155 | 520555 | 754446 |
| Airliner (A320 class) | 73500 | 122.6 | 34.13 | 5879 | 1385361 | 4014283 | 5200376 |
| Motor glider (AR 28) | 850 | 18.7 | 23.00 | 446 | 9076 | 1895 | 10780 |
<!-- /table:aircraft -->

Assumptions, per aircraft, are in `results/stability.json` (`info.assumptions`). The main ones:

- Take-off CLmax is 1.6 (GA, flaps 10°), 2.1 (turboprop, flaps 15°), 2.0 (A320 class, CONF 1+F) and 1.45 (glider).
  These values set V_R and the stall margin and are not computed.
- Flaps are modelled as a rotation of the panel normals by tau·delta on the flapped strips. Tau is the thin-airfoil
  flap effectiveness × 0.85. Slats are not modelled.
- The gear positions are taken from the geometry where it has gear (ATR 72 entry). On the A320 class they are chosen
  so the nose gear carries ~7 % of the weight. On the light aircraft the mains sit at a 15° tip-back angle.
- Tyre rolling friction is 0.02.
- Propellers have efficiency 0.80 (0.75 on the glider) and a static figure of merit of 0.70.

## Models

**Rigid body.**
- Axes: earth NED, body x forward / y right / z down.
- State: position, body velocity, a unit quaternion for the attitude, and body rates. Euler angles are only an
  output, and `quat_to_euler` stays finite at θ = ±90°.
- Equations: `x' = C v`, `v' = F/m − ω × v`, `ω' = I⁻¹(M − ω × Iω)`.
- Integration: RK4 with the quaternion renormalised after each step.

**Aerodynamics (quasi-steady vortex lattice in the loop).**
- At every evaluation, the relative air velocity at each control point (flight velocity, `ω × r`, and the wind sampled
  per spanwise strip) sets the right-hand side of the lattice.
- The influence matrix depends only on the geometry, so it is LU-factorised once.
- Forces come from Kutta-Joukowski on the bound legs with the local velocity. Lift, pitching moment, pitch and roll
  damping, downwash at the tail and the dihedral effect therefore all come from one model. So does the rolling moment
  of a gust that differs from one wing to the other.
- Control surfaces rotate their panels' normals (elevator, ailerons, flaps).
- Added by build-up:
  - the fin as a lifting line, with the Helmbold lift slope at 1.55 × its aspect ratio and the rudder through tau;
  - the fuselage Munk moments in pitch and yaw (Raymer eqs. 16.25, 16.47);
  - zero-lift drag, gear and flap drag;
  - induced drag, with McCormick's ground-effect factor at the wing height.

**Engines.**
- Turbofan: Mattingly's high-bypass lapse T/T_SL = δ₀(1 − 0.49 √M).
- Propeller: T = ηP/√(V² + V₀²), so T equals the momentum-theory static thrust at rest and ηP/V in flight.

**Gear.** Each leg is a vertical spring-damper with a rolling resistance μN and a tyre side force that saturates at
the friction limit. The nose wheel steers. Springs and dampers are sized for a 0.25 m static compression and a
damping ratio of 0.6.

**Trim and derivatives.**
- `trim` zeroes u̇, ẇ and q̇ for given speed, altitude and flight-path angle.
- `stability_derivatives` takes central differences of the full model.
- `modes` gives the eigenvalues of the 12-state linearisation.

**Pilot and autopilot.**
- The pitch, roll and yaw loops are designed per aircraft from its own Cm_δe, Cl_δa, Cn_δr and inertias, for a
  target natural frequency and damping, and scheduled with the dynamic pressure.
- Outer loops: altitude through pitch attitude, heading through bank, airspeed through the throttle, and a washed-out
  yaw damper.
- Take-off schedule (an assumed technique, not a certified one):
  - full throttle and the take-off flap;
  - nose wheel and rudder hold the centreline, ailerons hold the wings level;
  - at V_R the pitch attitude goes up at 3°/s to 12° (transports) or at 6°/s to 11° (light aircraft);
  - V_R is 1.05 V_s (transports) or 1.10 V_s (light aircraft), with V_s the take-off-flap stall speed;
  - after lift-off the pitch attitude is trimmed to fly 1.25 V_s (1.30 V_s for light aircraft), and bank holds the
    track along the runway.

**Turbulence** (`flight.turbulence`, MIL-F-8785C §3.7 and MIL-HDBK-1797 §4.9):

- Intensities and scale lengths:
  - Low altitude (h < 1000 ft): σ_w = 0.1 W₂₀ with W₂₀ = 15 / 30 / 45 kt. σ_u = σ_v = σ_w / (0.177 + 0.000823 h)^0.4.
    L_w = h and L_u = L_v = h / (0.177 + 0.000823 h)^1.2.
  - Medium and high altitude (h > 2000 ft): isotropic. σ comes from the exceedance-probability chart: light 1e-2,
    moderate 1e-3, severe 1e-5. L is 1750 ft (Dryden) or 2500 ft (von Kármán).
  - Between 1000 and 2000 ft, values are interpolated linearly.
- Spectra:
  - Dryden: Φ_u = σ²(2L/π)/(1 + (LΩ)²), Φ_w = σ²(L/π)(1 + 3(LΩ)²)/(1 + (LΩ)²)².
  - von Kármán: the same with exponents 5/6 and 11/6 and the 1.339 factor.
- Generators:
  - `FrozenLine`: spectral synthesis along the path, with exact amplitudes and random phases.
  - `DrydenFilter`: the forming filters with white noise, discretised exactly (Van Loan).
  - `FrozenField3D`: isotropic, divergence-free 3-D von Kármán turbulence from random Fourier modes (Kraichnan 1970).
    Its energy spectrum E(k) projects exactly onto the MIL 1-D spectra (checked by quadrature).
  - `DiscreteGust`: the 1-cosine gust, as the MIL ramp or the full shape.
  - `ShearWind`: the logarithmic wind profile.
- The high-altitude intensity table is the MIL-F-8785C Figure 7 table as used by common implementations of the
  specification. It is reproduced in `HIGH_ALTITUDE_SIGMA_FT_S`.

<!-- table:derivatives -->
| aircraft | CLα | Cmα | static margin | Cmq | Clβ | Cnβ | Clp | Cnr | Cmδe | Clδa | Cnδr | oscillatory modes ωn [rad/s] / ζ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Light GA (C172 class) | 5.12 | -1.50 | 0.29 | -18.44 | -0.042 | 0.064 | -0.50 | -0.106 | -1.68 | -0.193 | -0.066 | 5.10 / 0.54, 2.14 / 0.20, 0.21 / 0.10 |
| Turboprop (ATR 72 class) | 5.73 | -1.76 | 0.31 | -54.49 | -0.085 | 0.135 | -0.60 | -0.146 | -2.64 | -0.187 | -0.114 | 1.95 / 0.45, 1.45 / 0.09, 0.11 / 0.05 |
| Airliner (A320 class) | 5.26 | -1.94 | 0.37 | -44.59 | -0.217 | 0.080 | -0.47 | -0.246 | -2.38 | -0.072 | -0.129 | 1.44 / 0.40, 1.00 / 0.11, 0.10 / 0.02 |
| Motor glider (AR 28) | 6.02 | -0.32 | 0.05 | -21.49 | -0.083 | 0.044 | -0.69 | -0.028 | -1.13 | -0.345 | -0.042 | 2.00 / 0.84, 1.19 / 0.27, 0.27 / 0.04 |
<!-- /table:derivatives -->

The oscillatory modes are listed fastest first: short period, Dutch roll, phugoid.

## Verification

**Turbulence against the target spectra.** Each generator, component and spectral form is checked. The PSD ratio is
the band-averaged Welch PSD divided by the target, in eight log bands between 4/L and the resolution.

The frozen line has exact amplitudes, so its variance is the band-limited integral of the target spectrum. The
forming filter is a stochastic process, so 200,000 samples leave a few per cent of scatter.

The 3-D field is one realisation of 96 × 32 random modes with dx_min = 2 m: the band is k ≤ k_max = π/(2 dx_min) =
0.785 rad/m, marked on the right-hand plot. Its isotropic energy spectrum matches the MIL forms to 1e-3 by quadrature
(`tests/test_flight_turbulence.py`).

Along a line, the band-averaged w spectrum follows the target between 5/L and 0.2 rad/m (the band ratios in the
table). Above that it falls off: from 0.2 rad/m to 0.95 k_max the ratios are 0.12–0.62 (u), 0.52–1.0 (v) and
0.30–0.85 (w) (`psd_ratio_high_band` in `results/verification.json`). The reason is that a line spectrum at k₁ needs
modes with |k| > k₁, and there are none beyond k_max. The field therefore represents wavelengths down to about
4 × 2π/k_max = 16 dx_min, and the line spectrum at high k is spiky because the modes are discrete.

The encounters use dx_min = MAC/4 (128 log-spaced shells, so the mode count does not grow with k_max). They therefore
resolve gusts down to wavelengths of about 4 MAC. 4 MAC / span is 0.47 for the A320 class (MAC 4.04 m, span 34.13 m), 0.54 for the light GA (1.48 m, 11.02 m),
0.34 for the ATR class (2.30 m, 27.05 m) and 0.15 for the glider (0.86 m, 23.0 m), from `results/stability.json`.
Shorter gusts, which would add rolling and pitching moments of their own, are filtered out.

![PSD of generated turbulence against MIL-F-8785C](../assets/flight_dynamics/turbulence_psd.png)

<!-- table:psd -->
| generator | form | comp. | σ target [m/s] | σ generated [m/s] | PSD ratio min | PSD ratio max |
|---|---|---|---|---|---|---|
| FrozenLine | dryden | u | 7.050 | 7.038 | 0.99 | 1.01 |
| FrozenLine | dryden | v | 7.050 | 7.042 | 0.99 | 1.01 |
| FrozenLine | dryden | w | 7.050 | 7.042 | 0.99 | 1.01 |
| FrozenLine | von_karman | u | 7.050 | 7.019 | 0.99 | 1.02 |
| FrozenLine | von_karman | v | 7.050 | 7.019 | 0.99 | 1.01 |
| FrozenLine | von_karman | w | 7.050 | 7.019 | 0.99 | 1.04 |
| DrydenFilter | dryden | u | 2.879 | 2.823 | 0.97 | 1.07 |
| DrydenFilter | dryden | v | 2.879 | 2.894 | 0.93 | 1.05 |
| DrydenFilter | dryden | w | 2.879 | 2.795 | 0.96 | 1.06 |
| FrozenField3D | von_karman | u | 6.991 | 6.917 | 0.79 | 1.09 |
| FrozenField3D | von_karman | v | 6.991 | 7.465 | 0.84 | 1.34 |
| FrozenField3D | von_karman | w | 6.991 | 6.341 | 0.66 | 1.18 |
<!-- /table:psd -->

**Trim.** Each aircraft is trimmed at its cruise point and flown stick-fixed for 10 s. The 1e-9 m/s airspeed change
is the regularisation constant in the airspeed norm, not a drift.

<!-- table:trim -->
| aircraft | V [m/s] | h [m] | α [deg] | δe [deg] | throttle | trim residual [m/s²] | 10 s: max |ΔV| [m/s] | max |Δh| [m] | max |Δθ| [deg] |
|---|---|---|---|---|---|---|---|---|---|
| Light GA (C172 class) | 55 | 1500 | 3.00 | -1.61 | 0.74 | 1.44e-14 | 1.00e-09 | 0.0000 | 3.98e-16 |
| Turboprop (ATR 72 class) | 120 | 3000 | 1.45 | -0.83 | 0.70 | 3.80e-15 | 1.00e-09 | 0.0000 | 0.0000 |
| Airliner (A320 class) | 128 | 3000 | 6.84 | -3.56 | 0.30 | 1.12e-14 | 1.00e-09 | 0.0000 | 0.0000 |
| Motor glider (AR 28) | 30 | 1500 | 3.98 | -2.42 | 0.16 | 1.99e-14 | 1.00e-09 | 0.0000 | 1.59e-15 |
<!-- /table:trim -->

**Energy and angular momentum.** The test case is the A320-class body, spinning and translating under gravity alone
for 20 s. Kinetic plus potential energy and the angular momentum in NED must be conserved. The error falls with the
RK4 order as the step is halved.

<!-- table:conservation -->
| Δt [s] | relative energy error (20 s) | relative angular-momentum error | abs(norm(q) - 1) |
|---|---|---|---|
| 0.020 | 4.40e-10 | 2.68e-09 | 0.00 |
| 0.010 | 8.03e-12 | 1.67e-10 | 0.00 |
| 0.005 | 1.62e-12 | 1.05e-11 | 0.00 |
<!-- /table:conservation -->

**Lattice in the loop vs the steady lattice.** The lift of the in-the-loop model (Kutta-Joukowski with the inclined
free stream) is compared with the linearised steady solution at α = 0 and 4°. The two agree to O(α²).

<!-- table:lattice -->
| aircraft | α [deg] | CL lattice in the loop | CL steady lattice | relative difference |
|---|---|---|---|---|
| Light GA (C172 class) | 0 | 0.1617 | 0.1617 | 4.00e-11 |
| Light GA (C172 class) | 4 | 0.5194 | 0.5201 | 0.0013 |
| Turboprop (ATR 72 class) | 0 | 0.4257 | 0.4257 | 4.00e-11 |
| Turboprop (ATR 72 class) | 4 | 0.8254 | 0.8268 | 0.0016 |
| Airliner (A320 class) | 0 | 0.1863 | 0.1863 | 4.00e-11 |
| Airliner (A320 class) | 4 | 0.5564 | 0.5572 | 0.0014 |
| Motor glider (AR 28) | 0 | 0.5238 | 0.5238 | 4.00e-11 |
| Motor glider (AR 28) | 4 | 0.9457 | 0.9473 | 0.0017 |
<!-- /table:lattice -->

**Take-off closed form.** Raymer's take-off distance (Aircraft Design: A Conceptual Approach, §17.8) is evaluated with
the same thrust, lift, drag and ground-effect model as the simulation. The table below shows it twice: once with
Raymer's V_LOF = 1.1 V_s and his 1 s / 3 s rotation, and once with the simulated V_LOF and rotation time.

## Take-off

Runs: ISA, sea level, no wind, from rest on the gear at full throttle, stopped at 400 ft. "To 35 ft" is the distance
at which the CG has climbed 35 ft above its rest height.

![Take-off trajectories](../assets/flight_dynamics/takeoff.png)

<!-- table:takeoff -->
| aircraft | V_R [kt] | V_LOF [kt] | θ at lift-off [deg] | α at lift-off [deg] | ground roll [m] | to 35 ft [m] | V at 35 ft [kt] | climb gradient [%] | Raymer, V_LOF = 1.1 V_s [m] | sim vs Raymer [%] | Raymer, simulated V_LOF and rotation time [m] | sim vs Raymer [%] | min tail clearance [m] | max CL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Light GA (C172 class) | 55.7 | 62.4 | 6.8 | 6.5 | 244 | 403 | 67.9 | 13.6 | 311 | 29.6 | 410 | -1.7 | 0.95 | 1.10 |
| Turboprop (ATR 72 class) | 109.5 | 119.7 | 6.3 | 6.0 | 1060 | 1291 | 123.2 | 7.3 | 1299 | -0.6 | 1492 | -13.5 | 0.23 | 1.81 |
| Airliner (A320 class) | 141.4 | 155.6 | 9.7 | 9.6 | 1346 | 1691 | 168.2 | 10.4 | 1703 | -0.7 | 1882 | -10.1 | 0.18 | 1.57 |
| Motor glider (AR 28) | 47.9 | 54.3 | 1.6 | 1.3 | 186 | 286 | 57.7 | 16.8 | 250 | 14.5 | 326 | -12.4 | 1.17 | 1.07 |
<!-- /table:takeoff -->

- **Raymer's standard assumptions** (V_LOF = 1.1 V_s, 3 s / 1 s rotation): the simulated distance to 35 ft is within
  1 % for the two transports. It is 30 % longer for the light GA and 15 % longer for the glider. Their simulated
  lift-off speeds are well above 1.1 V_s, because the pilot model rotates to a fixed attitude and lifts off below the
  assumed CLmax (see the max CL column).
- **Closed form fed the simulated V_LOF and rotation time:** it is 2–14 % longer than the simulation. Its transition
  arc (n = 1.2 at 1.15 V_s) is more conservative than the flown climb-out.

**A320 class against published values.** The certified take-off field length is the longer of two distances: the
one-engine-inoperative distance, and 1.15 × the all-engines distance to 35 ft. It is published for the 78 t MTOW. The
simulation runs all engines, with no margin, at 73.5 t. The two numbers are of the same order but are not the same
quantity.

<!-- table:a320 -->
| quantity | value |
|---|---|
| simulated all-engines distance to 35 ft, 73500 kg, ISA sea level | 1691 m |
| 1.15 x the simulated all-engines distance (the all-engines part of the certified rule) | 1691 m x 1.15 = 1945 m |
| published | A320-200, MTOW 78 t, sea level ISA: take-off field length about 2,090-2,100 m (certified, one engine inoperative and 1.15 x all-engines distance, whichever is longer); e.g. https://www.skytamer.com/Airbus_A320.html, https://aircraftinvestigation.info/airplanes/Airbus_A320-200_CFM56.html (which also quotes about 1,825 m at 73.5 t); secondary sources. The primary source, Airbus' A320 Aircraft Characteristics - Airport and Maintenance Planning, was not checked (it could not be opened from the environment of this study). |
<!-- /table:a320 -->

**Crosswind.** The wind is 15 kt at 20 ft from the right, with the MIL-F-8785C logarithmic profile. The pilot keeps
the centreline with the nose wheel and the rudder and the wings level with the ailerons. After lift-off it holds the
track, so the aircraft crabs into the wind.

![Crosswind take-off](../assets/flight_dynamics/takeoff_crosswind.png)

<!-- table:crosswind -->
| aircraft | max offset on the ground [m] | max bank [deg] | max rudder [deg] | max aileron [deg] | heading at lift-off [deg] | min wing-tip clearance [m] | to 35 ft [m] |
|---|---|---|---|---|---|---|---|
| Light GA (C172 class) | 3.48 | 11.09 | 7.35 | 1.77 | 5.05 | 1.90 | 384 |
| Turboprop (ATR 72 class) | 1.01 | 1.25 | 5.68 | 4.45 | 1.76 | 3.95 | 1278 |
| Airliner (A320 class) | 1.51 | 4.99 | 11.62 | 14.90 | 0.19 | 2.67 | 1666 |
| Motor glider (AR 28) | 1.36 | 5.64 | 11.15 | 3.01 | 4.76 | 0.63 | 277 |
<!-- /table:crosswind -->

**Gust during rotation.** A 1-cosine down-gust of 4 m/s over 60 m is placed where the nominal run reached V_R.

<!-- table:gust -->
| aircraft | to 35 ft [m] | change vs nominal [m] | V_LOF [kt] | θ at lift-off [deg] | min tail clearance [m] |
|---|---|---|---|---|---|
| Light GA (C172 class) | 394 | -9 | 60.8 | 9.2 | 0.71 |
| Turboprop (ATR 72 class) | 1293 | 2 | 119.6 | 6.3 | 0.22 |
| Airliner (A320 class) | 1693 | 1 | 155.6 | 9.8 | 0.18 |
| Motor glider (AR 28) | 284 | -2 | 54.2 | 2.9 | 1.09 |
<!-- /table:gust -->

## Turbulence encounters

The turbulence is the same for all four aircraft: von Kármán, MIL-F-8785C severe (exceedance probability 1e-5) at
3000 m, σ = 7.05 m/s, L = 762 m. It is a frozen 3-D field, so the two wings meet different gusts.

Each aircraft flies at its own speed for 120 s: 55, 120 and 128 m/s, and the glider at 40 m/s. Glider pilots speed up in rough air, and at 30 m/s its trim leaves little α margin to the clean CLmax (an assumption of the study). Each case is run stick-fixed and
with the autopilot (altitude, heading and speed hold), in three realisations of the field. The first 5 s are
discarded.

Cabin accelerations are the specific force at the passenger stations: `a_cg + ω̇ × r + ω × (ω × r)`. Load factors
are in body axes, so steady level flight gives n_z = cos θ.

<!-- table:turbulence -->
| aircraft | controls | V [m/s] | W/S [Pa] | A = ρVCLα/2(W/S) [g per m/s] | A·σ_w [g] | RMS Δn_z CG [g] | spread over seeds | max Δn_z | min Δn_z | RMS bank [deg] | peak bank [deg] | RMS pitch [deg] | RMS n_y CG [g] | worst cabin station: RMS / peak Δn_z [g] | max α [deg] | α at clean CLmax [deg] | model validity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Light GA (C172 class) | stick fixed | 55 | 666 | 0.1919 | 1.328 | 0.586 | 0.078 | 2.47 | -1.42 | 22.6 | 63.1 | 13.1 | 0.050 | rear seat: 0.609 / 2.59 | 12.7 | 14.1 | valid |
| Light GA (C172 class) | autopilot | 55 | 666 | 0.1919 | 1.091 | 0.368 | 0.045 | 1.46 | -1.23 | 7.0 | 18.8 | 4.4 | 0.032 | rear seat: 0.389 / 1.52 | 16.4 | 14.1 | beyond stall |
| Turboprop (ATR 72 class) | stick fixed | 120 | 3698 | 0.0846 | 0.626 | 0.566 | 0.072 | 2.18 | -1.08 | 14.9 | 39.0 | 12.7 | 0.067 | aft row: 0.590 / 2.26 | 10.1 | 11.8 | beyond stall |
| Turboprop (ATR 72 class) | autopilot | 120 | 3698 | 0.0846 | 0.616 | 0.314 | 0.028 | 0.98 | -1.01 | 4.5 | 12.5 | 2.9 | 0.034 | aft row: 0.338 / 1.14 | 8.6 | 11.8 | valid |
| Airliner (A320 class) | stick fixed | 128 | 5879 | 0.0520 | 0.358 | 0.287 | 0.072 | 1.16 | -0.80 | 14.7 | 39.4 | 6.4 | 0.040 | aft row: 0.331 / 1.32 | 14.1 | 14.1 | beyond stall |
| Airliner (A320 class) | autopilot | 128 | 5879 | 0.0520 | 0.353 | 0.214 | 0.047 | 0.69 | -0.71 | 3.1 | 8.6 | 2.7 | 0.026 | aft row: 0.246 / 0.84 | 13.1 | 14.1 | valid |
| Motor glider (AR 28) | stick fixed | 40 | 446 | 0.2471 | 1.522 | – | – | – | – | – | – | – | – | – | – | 7.4 | departed |
| Motor glider (AR 28) | autopilot | 40 | 446 | 0.2471 | 1.510 | – | – | – | – | – | – | – | – | – | – | 7.4 | departed |
<!-- /table:turbulence -->

"Model validity" compares each run with the model's limits, using the stall angle of attack (α at the assumed clean
CLmax, from the trim) as the limit of the linear lattice. A run is "beyond stall" if any realisation spends time above
it. It is "departed" if the bank passes 90° or α goes 10° past stall. A departed run describes a loss of control the
quasi-steady linear model cannot represent, so its numbers are not a gust response.

- **Motor glider in severe turbulence:** at 40 m/s, atan(σ_w/V) = 10.0° of RMS gust angle meets 6.5° of α margin
  from the trim. All three autopilot realisations depart, and one of the three stick-fixed ones does. Their numbers
  are blanked in the tables. Only the glider's light-turbulence row stays
  valid. In moderate turbulence it spends 7.0 % of the time past the stall angle, without departing; its clip
  and trace show that moderate run.
- **Light GA with the autopilot (severe):** 0.01–1.5 % of the time past the stall angle, depending on the
  realisation.
- **Turboprop and A320 class, stick fixed:** at most 0.2 % of the time past it in one realisation.
- **Turboprop and A320 class with the autopilot:** never reach the stall angle.

These fractions are `time_above_stall_frac` in `results/turbulence.json`.
- **Stick-fixed runs:** over 120 s they include the unattended spiral and phugoid excursions (large bank angles), so
  their RMS values are not gust response alone. The autopilot rows are the comparison between aircraft.

![Load-factor and bank traces](../assets/flight_dynamics/turbulence_traces.png)

![RMS load factor at the CG and in the cabin](../assets/flight_dynamics/turbulence_rms.png)

**Wing loading and lift-curve slope.** A rigid wing held fixed in a vertical gust w would see
Δn = A·w with A = ρ V CL_α / (2 W/S). A·σ_w is therefore an upper bound for the RMS increment: plunge and pitch
relieve it. The Pratt-Walker formula adds the alleviation factor K_g = 0.88 μ / (5.3 + μ), with the mass ratio
μ = 2(W/S) / (ρ c̄ CL_α g), for a 1-cosine gust of half-length 12.5 c̄. Here U_de = 25 ft/s EAS, the same for every
aircraft.

The simulation is quasi-steady, without Küssner / Wagner lift lag. Its discrete-gust peak should therefore lie
between Pratt and the unalleviated value.

![Gust sensitivity and the Pratt check](../assets/flight_dynamics/gust_sensitivity.png)

<!-- table:pratt -->
| aircraft | V [m/s] | mass ratio μ | K_g | Δn, no alleviation | Δn Pratt | Δn simulated | sim / Pratt |
|---|---|---|---|---|---|---|---|
| Light GA (C172 class) | 55 | 19.7 | 0.693 | 1.698 | 1.177 | 1.283 | 1.090 |
| Turboprop (ATR 72 class) | 120 | 62.9 | 0.812 | 0.748 | 0.607 | 0.673 | 1.109 |
| Airliner (A320 class) | 128 | 62.1 | 0.811 | 0.460 | 0.373 | 0.407 | 1.092 |
| Motor glider (AR 28) | 40 | 19.1 | 0.689 | 2.185 | 1.506 | 1.794 | 1.191 |
<!-- /table:pratt -->

**The same gust for all four.** One physical gust: a vertical 1-cosine up-gust of 7.62 m/s over 100 m (H = 50 m)
at 3000 m, stick fixed, each aircraft at its cruise speed (`07_same_gust.py`). The slow, lightly loaded aircraft
meet it for longer, so they have time to plunge and pitch into it. Pratt's K_g was derived for H = 12.5 MAC and is
shown only for reference.

<!-- table:same_gust -->
| aircraft | V [m/s] | W/S [Pa] | gust duration [s] | peak Δn_z CG [g] | peak cabin Δn_z [g] | quasi-steady A·U [g] | Pratt (for H = 12.5 MAC) [g] | max Δθ [deg] | max Δh [m] |
|---|---|---|---|---|---|---|---|---|---|
| Light GA (C172 class) | 55 | 666 | 1.82 | 0.625 | rear seat: 0.652 | 1.463 | 1.014 | 5.40 | 4.4 |
| Turboprop (ATR 72 class) | 120 | 3698 | 0.83 | 0.571 | aft row: 0.627 | 0.644 | 0.523 | 1.31 | 1.6 |
| Airliner (A320 class) | 128 | 5879 | 0.78 | 0.357 | aft row: 0.429 | 0.396 | 0.321 | 0.86 | 1.4 |
| Motor glider (AR 28) | 40 | 446 | 2.50 | 0.939 | pilot: 0.934 | 1.883 | 1.297 | 3.21 | 15.5 |
<!-- /table:same_gust -->

**What the spanwise variation does.** The same aircraft are run stick-fixed in a frozen line, where the gust is
uniform across the span, at the same intensity. Without spanwise variation, the only rolling input is sideslip
through the dihedral effect.

<!-- table:spanwise -->
| aircraft | RMS bank, 3-D field [deg] | RMS bank, frozen line [deg] | RMS Δn_z, 3-D field [g] | RMS Δn_z, frozen line [g] |
|---|---|---|---|---|
| Light GA (C172 class) | 22.6 | 12.2 | 0.586 | 0.412 |
| Turboprop (ATR 72 class) | 14.9 | 2.4 | 0.566 | 0.350 |
| Airliner (A320 class) | 14.7 | 8.3 | 0.287 | 0.250 |
| Motor glider (AR 28) (departed stick fixed in severe turbulence: not shown) |  |  |  |  |
<!-- /table:spanwise -->

**Intensity.** Light, moderate and severe turbulence, autopilot on, one realisation for light and moderate.

<!-- table:intensity -->
| aircraft | severity | σ [m/s] | RMS Δn_z [g] | max Δn_z [g] | RMS bank [deg] | max α [deg] | model validity |
|---|---|---|---|---|---|---|---|
| Airliner (A320 class) | light | 1.84 | 0.051 | 0.18 | 0.7 | 8.5 | valid |
| Airliner (A320 class) | moderate | 2.88 | 0.080 | 0.28 | 1.1 | 9.8 | valid |
| Airliner (A320 class) | severe (mean of seeds) | 7.05 | 0.214 | 0.69 | 3.1 | 13.1 | valid |
| Light GA (C172 class) | light | 1.84 | 0.101 | 0.31 | 2.4 | 7.4 | valid |
| Light GA (C172 class) | moderate | 2.88 | 0.152 | 0.54 | 2.8 | 9.7 | valid |
| Light GA (C172 class) | severe (mean of seeds) | 7.05 | 0.368 | 1.46 | 7.0 | 16.4 | beyond stall |
| Motor glider (AR 28) | light | 1.84 | 0.112 | 0.46 | 1.7 | 6.0 | valid |
| Motor glider (AR 28) | moderate | 2.88 | 0.159 | 0.47 | 2.8 | 13.5 | beyond stall |
| Motor glider (AR 28) | severe (mean of seeds) | 7.05 | – | – | – | – | departed |
| Turboprop (ATR 72 class) | light | 1.84 | 0.077 | 0.23 | 1.1 | 3.3 | valid |
| Turboprop (ATR 72 class) | moderate | 2.88 | 0.127 | 0.35 | 1.6 | 4.5 | valid |
| Turboprop (ATR 72 class) | severe (mean of seeds) | 7.05 | 0.314 | 0.98 | 4.5 | 8.6 | valid |
<!-- /table:intensity -->

## 3-D visualisation

`flight.visuals.flight_scene(ac, hist, wind=...)` builds a studio `Scene` with an `Animation`. It contains:

- the aircraft mesh (the parametric airframe, or the geometry-library mesh for the ATR class) moved by the recorded
  attitude and position;
- the runway with its markings;
- the trajectory trail;
- the vertical gust w as vertical arrows on a heading-aligned grid around the aircraft, coloured blue (down) to red
  (up), and as wisps along the flown track;
- the lattice loading ΔCp on a sheet just above the wing and tail skins;
- a HUD with airspeed, height, n_z, pitch and bank.

`web_viewer` writes the browser viewer, which has play / pause, a time slider, playback speed, a follow camera and a
vectors toggle. `blender.render_animation` renders Cycles frames with the HUD burnt in and encodes MP4 / GIF.

The viewers are written by `03_takeoff.py` and `04_turbulence_encounters.py` into `examples/flight_dynamics/_out/`
(not committed). Open one with `python -m http.server -d examples/flight_dynamics/_out/a320_takeoff/viewer`.

| | |
|---|---|
| ![A320-class take-off in the viewer](../assets/flight_dynamics/viewer_a320_takeoff.png) | ![GA crosswind take-off](../assets/flight_dynamics/viewer_light_ga_crosswind.png) |
| ![A320-class in severe turbulence](../assets/flight_dynamics/viewer_a320_turbulence.png) | ![ATR-class in severe turbulence](../assets/flight_dynamics/viewer_regional_turboprop_turbulence.png) |

The "Streamlines" button shows the gust wisps; "Vectors" toggles the arrows.

Clips are rendered by `05_render_clips.py` from the saved runs. They are 720 × 405, 12 Cycles samples and 96 frames
at 12 fps, kept small enough to commit, in `docs/assets/flight_dynamics/clips/`. Raise `--size`, `--samples` and
`--frames` (with `--keep-in-out`) for full-resolution renders.

Clips: see the gallery at the top of this page.

## Reproduce

```bash
python examples/flight_dynamics/01_verification.py          # PSD, trim, conservation, lattice checks (~2 min)
python examples/flight_dynamics/02_trim_and_stability.py    # trim, derivatives, modes, mass properties (<1 min)
python examples/flight_dynamics/03_takeoff.py               # take-offs, crosswind, gust, viewers (~10 min)
python examples/flight_dynamics/04_turbulence_encounters.py # encounters, Pratt (~25 min on 4 cores)
python examples/flight_dynamics/07_same_gust.py             # the four aircraft through one gust (~1 min)
python examples/flight_dynamics/04_turbulence_encounters.py --moderate-visuals motor_glider
python examples/flight_dynamics/05_render_clips.py          # Blender clips (needs bpy; ~3 h on 4 cores with haze)
python examples/flight_dynamics/06_tables.py                # the tables and turbulence figures on this page
python examples/flight_dynamics/08_viewers.py               # rebuild the viewers and screenshots
pytest tests/test_flight_turbulence.py tests/test_flight_dynamics.py tests/test_studio_animation.py
```

## Limits

- **Quasi-steady aerodynamics.**
  - Lift follows the angle of attack instantly: there is no Wagner or Küssner lift build-up and no downwash lag
    (Cm_α̇).
  - Short gusts are therefore felt fully. This is why the discrete-gust peaks exceed Pratt's.
- **Linear lattice.**
  - The lattice has no stall and no section-drag polar. Only the stall margin (maximum CL reached against the assumed
    take-off CLmax) is reported.
  - Thickness, compressibility and viscous effects on lift are absent. The aero optimizer's Prandtl-Glauert stretch
    is not used here, because all runs are at M ≤ 0.43.
- **No aeroelasticity.** The wing is rigid. On the high-aspect-ratio glider and the airliner, wing bending would
  change both the gust loads and the cabin accelerations.
- **Ground effect** acts on the induced drag only (McCormick factor). The lift increase near the ground is not
  modelled.
- **Turbulence fields.**
  - The 3-D field is isotropic, with one scale length (L_w). At low altitude it keeps the MIL intensities but not
    their different scale lengths.
  - The frozen-field (Taylor) hypothesis is used throughout.
- **Propeller and engine effects.** Slipstream, torque, P-factor and gyroscopic moments are not modelled, and the
  engines respond instantly to the throttle.
- **Aircraft data.** The aircraft are "class" models, not the real types. Planforms, masses, inertias, take-off
  CLmax, gear stiffness and the pilot technique are estimates or assumptions (listed above and in
  `Aircraft.info`). Only the published figures quoted with their source are data.
- **One-engine-out take-off and braking are not simulated.** The certified field length therefore cannot be
  reproduced; only the all-engines distance is compared.
