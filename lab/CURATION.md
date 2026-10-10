# PINNeAPPle Lab curation

Generated 2026-10-10 23:00 UTC. Tiers: **A flagship** (product / paper / marketing once reviewed), **B solid** (demos, use cases, datasets), **C exploratory**, **D not usable**. Gates, not averages: see `pinneapple_lab/curation.py`.

| experiment | tier | best score | runs (A/B/C/D) | usable | product | paper | marketing | data | reviewed |
|---|---|---|---|---|---|---|---|---|---|
| `bar_wear` | **A** flagship | 100.0 | 8/0/0/0 | 100 % | - | - | - | ready | no |
| `solid_fem` | **A** flagship | 100.0 | 3/0/0/0 | 100 % | - | - | - | ready | no |
| `particle_suspension` | **A** flagship | 97.9 | 1/0/0/0 | 100 % | - | - | - | ready | no |
| `pipe_flow` | **A** flagship | 97.9 | 2/1/0/0 | 100 % | - | - | - | ready | no |
| `calculix_case` | **A** flagship | 95.7 | 2/3/0/0 | 100 % | - | - | - | ready | no |
| `repo_results` | **A** flagship | 92.9 | 1/9/0/0 | 100 % | - | - | - | ready | no |
| `benchmark_case` | **B** solid | 100.0 | 0/7/0/0 | 100 % | - | - | - | - | no |
| `rom_study` | **B** solid | 95.4 | 0/1/0/1 | 50 % | - | - | - | ready | no |
| `car_lbm` | **B** solid | 94.3 | 0/40/0/1 | 98 % | - | - | - | ready | no |
| `pendulum_video` | **B** solid | 92.9 | 0/1/0/0 | 100 % | - | - | - | ready | no |
| `bar_wear_ranking` | **B** solid | 92.0 | 0/1/0/0 | 100 % | - | - | - | ready | no |
| `vehicle_cfd` | **B** solid | 90.3 | 0/1/0/1 | 50 % | - | - | - | ready | no |
| `kepler_law` | **B** solid | 89.4 | 0/1/0/0 | 100 % | - | - | - | ready | no |
| `lorenz_discovery` | **B** solid | 89.4 | 0/1/0/0 | 100 % | - | - | - | - | no |
| `oscillator_discovery` | **B** solid | 89.4 | 0/1/0/0 | 100 % | - | - | - | ready | no |
| `accretion_flow` | **B** solid | 84.9 | 0/6/0/0 | 100 % | - | - | - | ready | no |
| `cylinder_lbm` | **B** solid | 84.9 | 0/6/1/2 | 67 % | - | - | - | ready | no |
| `bondi_accretion` | **B** solid | 83.6 | 0/6/0/0 | 100 % | - | - | - | ready | no |
| `heat_xtfc` | **B** solid | 83.6 | 0/8/0/4 | 67 % | - | - | - | ready | no |
| `oscillator` | **B** solid | 83.6 | 0/60/0/0 | 100 % | - | - | - | ready | no |
| `example_script` | **C** exploratory | 86.2 | 0/0/6/3 | 0 % | - | - | - | - | no |

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
| `calculix_case/buckling_column` | B | ? | ✓ | ? | ✓ | ~ | ✓ |
| `calculix_case/fin_heat` | B | ? | ✓ | ? | ✓ | ~ | ✓ |
| `calculix_case/modal_cantilever` | B | ? | ✓ | ? | ✓ | ~ | ✓ |
| `car_lbm` | B | ✓ | ✓ | ✓ | ? | ~ | ✓ |
| `cylinder_lbm` | B | ✓ | ? | ✓ | ? | ~ | ✓ |
| `heat_xtfc` | B | ? | ✓ | ? | ✓ | ~ | ✓ |
| `kepler_law` | B | ~ | ✓ | ? | ✓ | ~ | ✓ |
| `lorenz_discovery` | B | ? | ✓ | ? | ✓ | ~ | ~ |
| `oscillator` | B | ✓ | ✓ | ? | ✓ | ? | ~ |
| `oscillator_discovery` | B | ? | ✓ | ? | ✓ | ? | ~ |
| `pendulum_video` | B | ? | ✓ | ? | ✓ | ~ | ✓ |
| `pipe_flow/kenics_mixer` | B | ~ | ? | ✓ | ✓ | ~ | ✓ |
| `repo_results/airfoil_surrogate` | B | ~ | ? | ? | ? | ? | ~ |
| `repo_results/bh_forecast` | B | ~ | ? | ? | ✓ | ~ | ✓ |
| `repo_results/burgers_pinn` | B | ~ | ✓ | ? | ✓ | ~ | ✓ |
| `repo_results/concorde_aoa` | B | ~ | ? | ✓ | ? | ~ | ✓ |
| `repo_results/fin_inverse_2d` | B | ~ | ✓ | ? | ✓ | ~ | ~ |
| `repo_results/fin_inverse_3d` | B | ~ | ✓ | ? | ✓ | ~ | ✓ |
| `repo_results/heatsink_surrogate` | B | ~ | ? | ? | ? | ? | ~ |
| `repo_results/lbm_strouhal` | B | ~ | ✓ | ? | ✓ | ~ | ✓ |
| `repo_results/meshgraphnet` | B | ~ | ? | ? | ✓ | ~ | ✓ |
| `rom_study/beam_parametric` | B | ~ | ✓ | ? | ✓ | ✓ | ✓ |
| `vehicle_cfd/car` | B | ~ | ? | ✓ | ? | ~ | ✓ |
| `example_script/examples/getting_started/01_harmonic_oscillator.py` | C | ✓ | ? | ? | ? | ~ | ? |
| `example_script/examples/getting_started/02_damped_oscillator.py` | C | ✓ | ? | ? | ? | ~ | ? |
| `example_script/examples/getting_started/05_logistic_growth.py` | C | ✓ | ? | ? | ? | ~ | ? |
| `example_script/examples/getting_started/06_lotka_volterra.py` | C | ✓ | ? | ? | ? | ? | ? |
| `example_script/examples/getting_started/07_nonlinear_pendulum.py` | C | ✓ | ? | ? | ? | ~ | ? |
| `example_script/examples/getting_started/03_heat_diffusion_1d.py` | D | ✗ | ? | ? | ? | ~ | ✗ |
| `example_script/examples/getting_started/04_wave_equation_1d.py` | D | ✗ | ? | ? | ? | ? | ✗ |
| `example_script/examples/getting_started/08_van_der_pol.py` | D | ✗ | ? | ? | ? | ~ | ✗ |
| `rom_study/cylinder_wake` | D | ✗ | ✓ | ✓ | ✓ | ~ | ✗ |
| `vehicle_cfd/rocket` | D | ~ | ✗ | ✓ | ✗ | ~ | ✗ |

## What each experiment needs next

- `accretion_flow` (B): compare with an independent reference (exact solution, experiment, published value)
- `bar_wear_ranking` (B): compare with an independent reference (exact solution, experiment, published value); add figures (a movie, a render or a 3-D view for communication)
- `benchmark_case` (B): compare with an independent reference (exact solution, experiment, published value); test on unseen inputs (held-out designs, out-of-distribution cases); add figures (a movie, a render or a 3-D view for communication)
- `bondi_accretion` (B): show it beats a baseline or satisfies a physical law it was not trained on; add figures (a movie, a render or a 3-D view for communication)
- `car_lbm` (B): compare with an independent reference (exact solution, experiment, published value)
- `cylinder_lbm` (B): compare with an independent reference (exact solution, experiment, published value)
- `example_script` (C): compare with an independent reference (exact solution, experiment, published value); show it beats a baseline or satisfies a physical law it was not trained on; add figures (a movie, a render or a 3-D view for communication)
- `heat_xtfc` (B): show it beats a baseline or satisfies a physical law it was not trained on; add figures (a movie, a render or a 3-D view for communication)
- `kepler_law` (B): show it beats a baseline or satisfies a physical law it was not trained on; add figures (a movie, a render or a 3-D view for communication)
- `lorenz_discovery` (B): show it beats a baseline or satisfies a physical law it was not trained on; add figures (a movie, a render or a 3-D view for communication)
- `oscillator` (B): show it beats a baseline or satisfies a physical law it was not trained on; add figures (a movie, a render or a 3-D view for communication)
- `oscillator_discovery` (B): show it beats a baseline or satisfies a physical law it was not trained on; add figures (a movie, a render or a 3-D view for communication)
- `pendulum_video` (B): show it beats a baseline or satisfies a physical law it was not trained on
- `rom_study` (B): compare with an independent reference (exact solution, experiment, published value)
- `vehicle_cfd` (B): compare with an independent reference (exact solution, experiment, published value); document it (docs page, references)
