# PINNeAPPle Lab curation

Generated 2026-10-18 23:14 UTC. Tiers: **A flagship** (product / paper / marketing once reviewed), **B solid** (demos, use cases, datasets), **C plausible** (partly checked), **D exploratory** (not validated), **E known problems** (a failing check), **F not usable** (crashed, diverged, failed physics or contradicts the reference). Gates, not averages: see `pinneapple_lab/curation.py`.

| experiment | tier | best score | runs (A/B/C/D/E/F) | usable | product | paper | marketing | data | reviewed |
|---|---|---|---|---|---|---|---|---|---|
| `bar_wear` | **A** flagship | 100.0 | 8/0/0/0/0/0 | 100 % | - | - | - | ready | no |
| `solid_fem` | **A** flagship | 100.0 | 3/0/0/0/0/0 | 100 % | - | - | - | ready | no |
| `particle_suspension` | **A** flagship | 97.9 | 1/0/0/0/0/0 | 100 % | - | - | - | ready | no |
| `pipe_flow` | **A** flagship | 97.9 | 2/1/0/0/0/0 | 100 % | - | - | - | ready | no |
| `calculix_case` | **A** flagship | 95.7 | 2/3/0/0/0/0 | 100 % | - | - | - | ready | no |
| `repo_results` | **A** flagship | 94.7 | 3/20/0/1/1/0 | 92 % | - | - | - | - | no |
| `benchmark_case` | **B** solid | 100.0 | 0/7/0/0/0/0 | 100 % | - | - | - | - | no |
| `rom_study` | **B** solid | 98.0 | 0/2/0/0/0/0 | 100 % | - | - | - | ready | no |
| `car_lbm` | **B** solid | 94.3 | 0/40/0/0/0/0 | 100 % | - | - | - | ready | no |
| `vehicle_cfd` | **B** solid | 93.8 | 0/1/0/0/0/0 | 100 % | - | - | - | ready | no |
| `pendulum_video` | **B** solid | 92.9 | 0/1/0/0/0/0 | 100 % | - | - | - | ready | no |
| `bar_wear_ranking` | **B** solid | 92.0 | 0/1/0/0/0/0 | 100 % | - | - | - | ready | no |
| `manifest` | **B** solid | 91.9 | 0/3/0/1/0/0 | 75 % | - | - | - | - | no |
| `bricks` | **B** solid | 91.2 | 0/3/0/0/0/0 | 100 % | - | - | - | - | no |
| `kepler_law` | **B** solid | 89.4 | 0/1/0/0/0/0 | 100 % | - | - | - | ready | no |
| `lorenz_discovery` | **B** solid | 89.4 | 0/1/0/0/0/0 | 100 % | - | - | - | - | no |
| `oscillator_discovery` | **B** solid | 89.4 | 0/1/0/0/0/0 | 100 % | - | - | - | ready | no |
| `em_pinn` | **B** solid | 88.5 | 0/2/0/0/0/2 | 50 % | - | - | - | - | no |
| `accretion_flow` | **B** solid | 84.9 | 0/6/0/0/0/0 | 100 % | - | - | - | ready | no |
| `cylinder_lbm` | **B** solid | 84.9 | 0/6/0/1/0/2 | 67 % | - | - | - | ready | no |
| `bondi_accretion` | **B** solid | 83.6 | 0/6/0/0/0/0 | 100 % | - | - | - | ready | no |
| `heat_xtfc` | **B** solid | 83.6 | 0/8/0/0/0/4 | 67 % | - | - | - | ready | no |
| `oscillator` | **B** solid | 83.6 | 0/60/0/0/0/0 | 100 % | - | - | - | ready | no |
| `example_script` | **D** exploratory | 86.2 | 0/0/0/5/0/3 | 0 % | - | - | - | - | no |

## Trust card: six questions per item

Data and geometry (was the physical problem represented correctly?), model (can it represent the dynamics?), physical constraints (equations and boundary conditions respected?), benchmark (against a solver or reference data?), uncertainty (where may it not be reliable?), engineering decision (adequate for the intended use?). ✓ answered by a passing check, ~ partial evidence, ? open, ✗ a failing check.

| item | tier | data & geometry | model | physics | benchmark | uncertainty | decision |
|---|---|---|---|---|---|---|---|
| `bar_wear/60/40 brass` | A | ~ | ✓ | ✓ | ✓ | ~ | ✓ |
| `bar_wear/PTFE` | A | ~ | ✓ | ✓ | ✓ | ~ | ✓ |
| `bar_wear/ferritic stainless steel` | A | ~ | ✓ | ✓ | ✓ | ~ | ✓ |
| `bar_wear/hardened tool steel` | A | ~ | ✓ | ✓ | ✓ | ~ | ✓ |
| `bar_wear/mild steel` | A | ~ | ✓ | ✓ | ✓ | ~ | ✓ |
| `bar_wear/polyethylene` | A | ~ | ✓ | ✓ | ✓ | ~ | ✓ |
| `bar_wear/stellite` | A | ~ | ✓ | ✓ | ✓ | ~ | ✓ |
| `bar_wear/tungsten carbide` | A | ~ | ✓ | ✓ | ✓ | ~ | ✓ |
| `calculix_case/l_bracket` | A | ~ | ✓ | ✓ | ✓ | ~ | ✓ |
| `calculix_case/plate_hole` | A | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `particle_suspension` | A | ✓ | ✓ | ✓ | ✓ | ~ | ✓ |
| `pipe_flow/bend_90` | A | ~ | ✓ | ✓ | ✓ | ~ | ✓ |
| `pipe_flow/laminar_pipe` | A | ~ | ✓ | ✓ | ✓ | ~ | ✓ |
| `repo_results/ahmed_body` | A | ~ | ✓ | ✓ | ✓ | ~ | ✓ |
| `repo_results/em_verification` | A | ~ | ✓ | ✓ | ✓ | ~ | ~ |
| `repo_results/prius_pmsm` | A | ✓ | ✓ | ✓ | ✓ | ~ | ✓ |
| `solid_fem/cantilever` | A | ~ | ✓ | ✓ | ✓ | ~ | ✓ |
| `solid_fem/simply_supported` | A | ~ | ✓ | ✓ | ✓ | ~ | ✓ |
| `solid_fem/torsion` | A | ~ | ✓ | ✓ | ✓ | ~ | ✓ |
| `accretion_flow` | B | ~ | ? | ✓ | ? | ~ | ✓ |
| `bar_wear_ranking` | B | ? | ? | ✓ | ? | ? | ~ |
| `benchmark_case/bumper_crash` | B | ✓ | ? | ? | ✓ | ~ | ✓ |
| `benchmark_case/heated_channel` | B | ~ | ✓ | ? | ✓ | ~ | ✓ |
| `benchmark_case/pdr` | B | ~ | ? | ? | ✓ | ~ | ~ |
| `benchmark_case/shock_train` | B | ~ | ? | ✓ | ? | ~ | ✓ |
| `benchmark_case/soil_twin` | B | ~ | ? | ? | ✓ | ~ | ✓ |
| `benchmark_case/sst` | B | ~ | ? | ? | ✓ | ~ | ✓ |
| `benchmark_case/terramechanics` | B | ~ | ? | ✓ | ✓ | ~ | ✓ |
| `bondi_accretion` | B | ~ | ✓ | ? | ✓ | ? | ~ |
| `bricks/bridge` | B | ~ | ✓ | ✓ | ? | ~ | ~ |
| `bricks/cantilever` | B | ~ | ✓ | ✓ | ? | ~ | ~ |
| `bricks/tower` | B | ~ | ✓ | ✓ | ? | ~ | ~ |
| `calculix_case/buckling_column` | B | ? | ✓ | ? | ✓ | ~ | ✓ |
| `calculix_case/fin_heat` | B | ? | ✓ | ? | ✓ | ~ | ✓ |
| `calculix_case/modal_cantilever` | B | ? | ✓ | ? | ✓ | ~ | ✓ |
| `car_lbm` | B | ✓ | ✓ | ✓ | ? | ~ | ✓ |
| `cylinder_lbm` | B | ✓ | ? | ✓ | ? | ~ | ✓ |
| `em_pinn/em_skin_effect` | B | ✓ | ? | ? | ✓ | ~ | ~ |
| `em_pinn/em_wire_az` | B | ✓ | ? | ? | ✓ | ~ | ~ |
| `heat_xtfc` | B | ? | ✓ | ? | ✓ | ~ | ✓ |
| `kepler_law` | B | ~ | ✓ | ? | ✓ | ~ | ✓ |
| `lorenz_discovery` | B | ? | ✓ | ? | ✓ | ~ | ~ |
| `manifest/examples/bricks` | B | ~ | ✓ | ? | ✓ | ? | ~ |
| `manifest/examples/enso_forecasting` | B | ~ | ✓ | ? | ✓ | ? | ~ |
| `manifest/examples/flight_dynamics` | B | ✓ | ✓ | ✓ | ? | ~ | ~ |
| `oscillator` | B | ✓ | ✓ | ? | ✓ | ? | ~ |
| `oscillator_discovery` | B | ? | ✓ | ? | ✓ | ? | ~ |
| `pendulum_video` | B | ? | ✓ | ? | ✓ | ~ | ✓ |
| `pipe_flow/kenics_mixer` | B | ~ | ? | ✓ | ✓ | ~ | ✓ |
| `repo_results/airfoil_handoff` | B | ✓ | ✓ | ? | ✓ | ~ | ~ |
| `repo_results/airfoil_surrogate` | B | ~ | ? | ? | ? | ? | ~ |
| `repo_results/airliner_cfd` | B | ~ | ✓ | ✓ | ? | ~ | ~ |
| `repo_results/bh_forecast` | B | ~ | ? | ? | ✓ | ~ | ✓ |
| `repo_results/burgers_pinn` | B | ~ | ✓ | ? | ✓ | ~ | ✓ |
| `repo_results/concorde_aoa` | B | ~ | ? | ✓ | ? | ~ | ✓ |
| `repo_results/design_from_intent` | B | ✓ | ✓ | ✓ | ? | ~ | ~ |
| `repo_results/fin_inverse_2d` | B | ~ | ✓ | ? | ✓ | ~ | ~ |
| `repo_results/fin_inverse_3d` | B | ~ | ✓ | ? | ✓ | ~ | ✓ |
| `repo_results/free_surface_trough` | B | ✓ | ? | ✓ | ? | ~ | ~ |
| `repo_results/heatsink_surrogate` | B | ~ | ? | ? | ? | ? | ~ |
| `repo_results/lbm_strouhal` | B | ~ | ✓ | ? | ✓ | ~ | ✓ |
| `repo_results/lunar_rover` | B | ~ | ? | ✓ | ? | ✓ | ✓ |
| `repo_results/meshgraphnet` | B | ~ | ? | ? | ✓ | ~ | ✓ |
| `repo_results/motor_spm` | B | ~ | ✓ | ✓ | ? | ~ | ~ |
| `repo_results/near_random_forecasting` | B | ✓ | ✓ | ? | ✓ | ~ | ~ |
| `repo_results/prius_drive` | B | ~ | ✓ | ✓ | ? | ~ | ~ |
| `repo_results/prius_end_effects` | B | ✓ | ✓ | ✓ | ? | ~ | ~ |
| `repo_results/prius_thermal_demag` | B | ✓ | ✓ | ✓ | ? | ~ | ~ |
| `repo_results/topology_optimization` | B | ~ | ✓ | ? | ✓ | ~ | ~ |
| `rom_study/beam_parametric` | B | ~ | ✓ | ? | ✓ | ✓ | ✓ |
| `rom_study/cylinder_wake` | B | ✓ | ✓ | ✓ | ✓ | ~ | ✓ |
| `vehicle_cfd/car` | B | ~ | ? | ✓ | ? | ~ | ✓ |
| `example_script/examples/getting_started/01_harmonic_oscillator.py` | D | ✓ | ? | ? | ? | ~ | ? |
| `example_script/examples/getting_started/02_damped_oscillator.py` | D | ✓ | ? | ? | ? | ~ | ? |
| `example_script/examples/getting_started/05_logistic_growth.py` | D | ✓ | ? | ? | ? | ~ | ? |
| `example_script/examples/getting_started/06_lotka_volterra.py` | D | ✓ | ? | ? | ? | ? | ? |
| `example_script/examples/getting_started/07_nonlinear_pendulum.py` | D | ✓ | ? | ? | ? | ~ | ? |
| `manifest/examples/pinneapple_and_physicsnemo` | D | ~ | ? | ? | ? | ? | ? |
| `repo_results/weather_era5` | D | ~ | ? | ? | ? | ~ | ? |
| `repo_results/benchmark_arena` | E | ✓ | ? | ? | ✓ | ~ | ✗ |
| `em_pinn/em_tm_cavity` | F | ✓ | ? | ? | ✗ | ~ | ✗ |
| `em_pinn/em_two_dielectric_capacitor` | F | ✓ | ? | ? | ✗ | ~ | ✗ |
| `example_script/examples/getting_started/03_heat_diffusion_1d.py` | F | ✗ | ? | ? | ? | ~ | ✗ |
| `example_script/examples/getting_started/04_wave_equation_1d.py` | F | ✗ | ? | ? | ? | ? | ✗ |
| `example_script/examples/getting_started/08_van_der_pol.py` | F | ✗ | ? | ? | ? | ~ | ✗ |

## E and F: the main problem and the next step

| item | tier | main problem | next step |
|---|---|---|---|
| `repo_results/benchmark_arena` | E | dataset audit diversity.effective · effective sample count 1 of 54 (no generator stated | D: fix the failing check |
| `em_pinn/em_tm_cavity` | F | Ez vs closed form (+2 more) | E: agree with the reference, or pass an independent physics check |
| `em_pinn/em_two_dielectric_capacitor` | F | phi vs closed form | E: agree with the reference, or pass an independent physics check |
| `example_script/examples/getting_started/03_heat_diffusion_1d.py` | F | crashed · TimeoutError: 03_heat_diffusion_1d.py did not finish in 600.0 s | E: make the run finish without an error |
| `example_script/examples/getting_started/04_wave_equation_1d.py` | F | crashed · TimeoutError: 04_wave_equation_1d.py did not finish in 600.0 s | E: make the run finish without an error |
| `example_script/examples/getting_started/08_van_der_pol.py` | F | crashed · TimeoutError: 08_van_der_pol.py did not finish in 600.0 s | E: make the run finish without an error |

## What each experiment needs next

- `accretion_flow` (B): compare with an independent reference (exact solution, experiment, published value)
- `bar_wear_ranking` (B): compare with an independent reference (exact solution, experiment, published value); add figures (a movie, a render or a 3-D view for communication)
- `benchmark_case` (B): compare with an independent reference (exact solution, experiment, published value); test on unseen inputs (held-out designs, out-of-distribution cases); add figures (a movie, a render or a 3-D view for communication)
- `bondi_accretion` (B): show it beats a baseline or satisfies a physical law it was not trained on; add figures (a movie, a render or a 3-D view for communication)
- `bricks` (B): compare with an independent reference (exact solution, experiment, published value)
- `car_lbm` (B): compare with an independent reference (exact solution, experiment, published value)
- `cylinder_lbm` (B): compare with an independent reference (exact solution, experiment, published value)
- `em_pinn` (B): show it beats a baseline or satisfies a physical law it was not trained on; test on unseen inputs (held-out designs, out-of-distribution cases); add figures (a movie, a render or a 3-D view for communication)
- `example_script` (D): compare with an independent reference (exact solution, experiment, published value); show it beats a baseline or satisfies a physical law it was not trained on; add figures (a movie, a render or a 3-D view for communication)
- `heat_xtfc` (B): show it beats a baseline or satisfies a physical law it was not trained on; add figures (a movie, a render or a 3-D view for communication)
- `kepler_law` (B): show it beats a baseline or satisfies a physical law it was not trained on; add figures (a movie, a render or a 3-D view for communication)
- `lorenz_discovery` (B): show it beats a baseline or satisfies a physical law it was not trained on; add figures (a movie, a render or a 3-D view for communication)
- `manifest` (B): compare with an independent reference (exact solution, experiment, published value); add figures (a movie, a render or a 3-D view for communication); document it (docs page, references)
- `oscillator` (B): show it beats a baseline or satisfies a physical law it was not trained on; add figures (a movie, a render or a 3-D view for communication)
- `oscillator_discovery` (B): show it beats a baseline or satisfies a physical law it was not trained on; add figures (a movie, a render or a 3-D view for communication)
- `pendulum_video` (B): show it beats a baseline or satisfies a physical law it was not trained on
- `rom_study` (B): compare with an independent reference (exact solution, experiment, published value)
- `vehicle_cfd` (B): compare with an independent reference (exact solution, experiment, published value)
