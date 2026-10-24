How far does an airliner roll before it clears 35 ft, and what does a severe gust do to the wing? Answering that usually means a separate flight-dynamics code with its own aerodynamics tables.

In PINNeAPPle we built `pinneapple_simulation.flight`: a 6-DOF rigid aircraft whose wing and tail loads come from the library's vortex lattice, solved again at every time step with the local air velocity at each panel. Thrust, landing gear on the runway, trim, a simple pilot/autopilot and MIL-F-8785C turbulence and gusts sit around it. Four aircraft classes: a C172-class single, an ATR 72-class turboprop, an A320-class airliner and an AR 28 motor glider.

What the committed runs show:
- Take-off, A320 class: 1691 m to 35 ft in the simulation vs 1703 m from Raymer's closed form with the same aircraft model (-0.7 %).
- Turbulence generator: the band-averaged PSD of the frozen-line generator stays within 0.99-1.04 of the MIL-F-8785C target.
- Discrete gust: simulated peak load factor is 1.09-1.19 x the Pratt formula, as expected for a quasi-steady model without lift lag.

We also kept the results that don't flatter the model. The light GA flies 29.6 % past Raymer, because its pilot model lifts off well above 1.1 V_s. The 3-D turbulence field only holds 0.66-1.34 of the target and drops off above 0.2 rad/m. In severe turbulence the motor glider departs, so its numbers are blanked rather than reported as a gust response. Every run is checked against the stall angle and labelled as valid, beyond stall or departed.

All the numbers come from the scripts in examples/flight_dynamics and the JSON files committed next to them. The clips are rendered from the same runs in Blender Cycles.

Guide: https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main/docs/guides/flight_dynamics.md

#FlightDynamics #Aerospace #AircraftDesign #CFD #Simulation #ScientificComputing #OpenSource #Python

---
## Media

1. `media/flight-dynamics-6dof/flight_dynamics_6dof.mp4` (70 s, 1080x1350). It shows a 6-DOF aircraft flown on a vortex lattice: an A320-class and a light GA take-off, distance to 35 ft against Raymer, MIL-F-8785C severe turbulence (A320 class and light GA), the generated turbulence PSD against the MIL spectrum, and four aircraft passing through the same 1-cosine gust. It closes on an end card with the repo URL.

## Sources

- 1691 m (A320 class, simulated distance to 35 ft): examples/flight_dynamics/results/takeoff.json -> nominal.a320.distance_35ft_m = 1691.22
- 1703 m (Raymer §17.8, V_LOF = 1.1 V_s): examples/flight_dynamics/results/takeoff.json -> closed_form.a320.raymer_1p1Vs.total_35ft_m = 1703.04
- -0.7 %: takeoff.json -> closed_form.a320.sim_over_raymer = 0.99306; also results/tables.md "takeoff" table, column "sim vs Raymer [%]"
- 29.6 % (light GA vs Raymer): takeoff.json -> closed_form.light_ga.sim_over_raymer = 1.29561; tables.md takeoff row Light GA
- 0.99-1.04 (FrozenLine PSD ratio, Dryden and von Kármán, u/v/w): examples/flight_dynamics/results/verification.json -> turbulence.cases[0..5].psd_ratio_min (min 0.9885) / psd_ratio_max (max 1.0370)
- 0.66-1.34 (FrozenField3D PSD ratio): verification.json -> turbulence.cases[9..11].psd_ratio_min (min 0.6582) / psd_ratio_max (max 1.3388)
- 0.2 rad/m (upper limit where the 3-D field follows the target): docs/guides/flight_dynamics.md, Verification section ("between 5/L and 0.2 rad/m")
- 1.09-1.19 x Pratt: examples/flight_dynamics/results/turbulence.json -> pratt[*].ratio_sim_to_pratt (1.0898, 1.1091, 1.0915, 1.1912)
- Motor glider departs in severe turbulence: turbulence.json -> runs[*].departed for aircraft motor_glider (true in 1 of 3 stick-fixed and 3 of 3 autopilot realisations); tables.md "turbulence" table, model validity = "departed"
- Video only: σ = 7.05 m/s (turbulence.json -> spec.sigma_w = 7.0505); ATR class 1291 m vs 1299 m (takeoff.json -> nominal.regional_turboprop.distance_35ft_m = 1291.33, closed_form.regional_turboprop.raymer_1p1Vs.total_35ft_m = 1299.41); same-gust peak Δn_z 0.357 g A320 class / 0.939 g glider and gust 7.62 m/s over 100 m (examples/flight_dynamics/results/same_gust.json -> aircraft.a320.dnz_cg_max, aircraft.motor_glider.dnz_cg_max, gust.amplitude_m_s, gust.half_length_m = 50)
