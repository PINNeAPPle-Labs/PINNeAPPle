# PINNeAPPle Lab curation

Generated 2026-10-10 21:51 UTC. Tiers: **A flagship** (product / paper / marketing once reviewed), **B solid** (demos, use cases, datasets), **C exploratory**, **D not usable**. Gates, not averages: see `pinneapple_lab/curation.py`.

| experiment | tier | best score | runs (A/B/C/D) | usable | product | paper | marketing | data | reviewed |
|---|---|---|---|---|---|---|---|---|---|
| `bar_wear` | **A** flagship | 100.0 | 8/0/0/0 | 100 % | - | - | - | ready | no |
| `solid_fem` | **A** flagship | 100.0 | 3/0/0/0 | 100 % | - | - | - | ready | no |
| `pipe_flow` | **A** flagship | 97.9 | 2/0/0/0 | 100 % | - | - | - | ready | no |
| `calculix_case` | **A** flagship | 95.7 | 2/3/0/0 | 100 % | - | - | - | ready | no |
| `repo_results` | **A** flagship | 92.9 | 1/9/0/0 | 100 % | - | - | - | ready | no |
| `benchmark_case` | **B** solid | 100.0 | 0/7/0/0 | 100 % | - | - | - | - | no |
| `rom_study` | **B** solid | 98.0 | 0/2/0/0 | 100 % | - | - | - | ready | no |
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
