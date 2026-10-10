# PINNeAPPle Lab datasets

25 datasets, 2321 samples from completed (validated) runs. Arrays are not in git: rebuild a dataset with its sweep command (cached runs are skipped), then export it as one file.

## `accretion_flow` / `frames`

Primitive fields (rho, v_r, v_theta, v_phi, p) on the (r, theta) grid

186 samples from 6 runs; every sample also carries the run parameters (alpha, backend, every, nr, ntheta, seed, t_end, torus_a, viscosity).

| field | kind | shape / type | range | units |
|---|---|---|---|---|
| frame | array | [5, 64, 32] float32 | -0.9 .. 0.982 | G=M=c=1 |
| t | scalar | float  | 0 .. 600 | GM/c^3 |
| mdot | scalar | float  | 0 .. 33.03 |  |
| alpha | scalar | float  | 0.2 .. 0.2 |  |
| viscosity | scalar | str  | SS |  |
| torus_a | scalar | float  | 0 .. 0 |  |

Schema ranges are those of one run (`accretion_flow-4a0c1426e768`).

```bash
python -m pinneapple_lab sweep accretion_flow -g alpha=0.05,0.1,0.2 -g viscosity=SS,ST
python -m pinneapple_lab export accretion_flow frames accretion_flow_frames.npz
```

## `benchmark_case` / `arrays`

Arrays behind the figures of PINNeAPPle-Benchmark/experiments/08_supersonic_shock_train

84 samples from 5 runs; every sample also carries the run parameters (case).

| field | kind | shape / type | range | units |
|---|---|---|---|---|
| array | array | variable float32 | -0.7762 .. 11.99 |  |
| file | scalar | str  | field3d_3d_pb2.50.npz, field3d_3d_pb3.00.npz, field_grid_pb3.00_240x20.npz, field_grid_pb3.00_360x30.npz, field_grid_pb3.00_540x45.npz, field_sweep_pb2.00_480x40.npz |  |
| key | scalar | str  | Mach_top, Mach_side, p_center, x, W, Mach |  |
| case | scalar | str  | shock_train |  |

Schema ranges are those of one run (`benchmark_case-9cf13dffb830`).

```bash
python -m pinneapple_lab sweep benchmark_case -g case=bumper_crash,heated_channel,pdr,shock_train,soil_twin,sst,terramechanics
python -m pinneapple_lab export benchmark_case arrays benchmark_case_arrays.npz
```

## `benchmark_case` / `lead_skill`

Test RMSE (degC) per forecast lead for every model

5 samples from 1 runs; every sample also carries the run parameters (case).

| field | kind | shape / type | range | units |
|---|---|---|---|---|
| model | scalar | str  | Persistence, Climatology, Damped persistence (AR1), FNO2d (PINNeAPPle) ensemble, ResNet-CNN (PINNeAPPle Conv2DModel) ensemble |  |
| rmse | array | [7] float64 | 0.291 .. 1.574 |  |
| mae | array | [7] float64 | 0.1931 .. 1.195 |  |

Schema ranges are those of one run (`benchmark_case-3074f671a9d8`).

```bash
python -m pinneapple_lab sweep benchmark_case -g case=bumper_crash,heated_channel,pdr,shock_train,soil_twin,sst,terramechanics
python -m pinneapple_lab export benchmark_case lead_skill benchmark_case_lead_skill.npz
```

## `benchmark_case` / `sweep`

Shock-train leading-edge position against back-pressure ratio

8 samples from 1 runs; every sample also carries the run parameters (case).

| field | kind | shape / type | range | units |
|---|---|---|---|---|
| pb_ratio | scalar | float  | 2 .. 4.25 |  |
| x_shock | scalar | float  | 0.0125 .. 2.487 |  |
| unstarted | scalar | bool  | False, True |  |
| no_train | scalar | bool  | True, False |  |

Schema ranges are those of one run (`benchmark_case-9cf13dffb830`).

```bash
python -m pinneapple_lab sweep benchmark_case -g case=bumper_crash,heated_channel,pdr,shock_train,soil_twin,sst,terramechanics
python -m pinneapple_lab export benchmark_case sweep benchmark_case_sweep.npz
```

## `bondi_accretion` / `profiles`

Radial Bondi profiles: exact and simulated

6 samples from 6 runs; every sample also carries the run parameters (backend, cs_inf, gamma, nr, t_end).

| field | kind | shape / type | range | units |
|---|---|---|---|---|
| r | array | [64] float64 | 4.153 .. 386.4 |  |
| rho_exact | array | [64] float64 | 1.28 .. 49.98 |  |
| rho_sim | array | [64] float64 | 1.287 .. 51.28 |  |
| v_exact | array | [64] float64 | -0.8602 .. -0.003879 |  |
| v_sim | array | [64] float64 | -0.8968 .. -0.004556 |  |
| cs_inf | scalar | float  | 0.1 .. 0.1 |  |
| gamma | scalar | float  | 1.4 .. 1.4 |  |
| mdot | scalar | float  | 9318 .. 9318 |  |

Schema ranges are those of one run (`bondi_accretion-3a8c3789e109`).

```bash
python -m pinneapple_lab sweep bondi_accretion -g cs_inf=0.08,0.1,0.14 -g gamma=1.3,1.4
python -m pinneapple_lab export bondi_accretion profiles bondi_accretion_profiles.npz
```

## `car_lbm` / `flow`

Car geometry (mask, signed distance) and the time-mean flow it produces (velocity / U, pressure coefficient), with drag and lift

24 samples from 24 runs; every sample also carries the run parameters (Cs, Re, clearance, diffuser_deg, hood, length_cells, nose, slant_deg, steps, tunnel_height, u_in, windshield_deg).

| field | kind | shape / type | range | units |
|---|---|---|---|---|
| mask | array | [320, 160] uint8 | 0 .. 1 |  |
| sdf | array | [320, 160] float32 | -0.1562 .. 3.582 | car lengths |
| mean_ux | array | [320, 160] float32 | -0.1728 .. 1.401 | U |
| mean_uy | array | [320, 160] float32 | -0.3832 .. 1.002 | U |
| mean_cp | array | [320, 160] float32 | -1.057 .. 1.735 | - |
| Cd | scalar | float  | 1.429 .. 1.429 |  |
| Cl | scalar | float  | 0.8153 .. 0.8153 |  |
| slant_deg | scalar | float  | 7.859 .. 7.859 |  |
| windshield_deg | scalar | float  | 23.8 .. 23.8 |  |
| hood | scalar | float  | 0.5333 .. 0.5333 |  |
| clearance | scalar | float  | 0.06627 .. 0.06627 |  |
| diffuser_deg | scalar | float  | 4.68 .. 4.68 |  |
| nose | scalar | float  | 0.2579 .. 0.2579 |  |

Schema ranges are those of one run (`car_lbm-ddba6505f43d`).

```bash
python -m pinneapple_lab sweep car_lbm -n 24
python -m pinneapple_lab export car_lbm flow car_lbm_flow.npz
```

## `car_lbm` / `vorticity`

Wake vorticity snapshots (normalised by U / car length)

1152 samples from 24 runs; every sample also carries the run parameters (Cs, Re, clearance, diffuser_deg, hood, length_cells, nose, slant_deg, steps, tunnel_height, u_in, windshield_deg).

| field | kind | shape / type | range | units |
|---|---|---|---|---|
| vorticity | array | [320, 160] float16 | -50.47 .. 56.25 |  |
| frame | scalar | int  | 0 .. 47 |  |
| Cd | scalar | float  | 1.429 .. 1.429 |  |
| slant_deg | scalar | float  | 7.859 .. 7.859 |  |
| windshield_deg | scalar | float  | 23.8 .. 23.8 |  |
| hood | scalar | float  | 0.5333 .. 0.5333 |  |
| clearance | scalar | float  | 0.06627 .. 0.06627 |  |
| diffuser_deg | scalar | float  | 4.68 .. 4.68 |  |
| nose | scalar | float  | 0.2579 .. 0.2579 |  |

Schema ranges are those of one run (`car_lbm-ddba6505f43d`).

```bash
python -m pinneapple_lab sweep car_lbm -n 24
python -m pinneapple_lab export car_lbm vorticity car_lbm_vorticity.npz
```

## `cylinder_lbm` / `vorticity`

Vorticity snapshots (normalised by U/D) labelled by regime and Re

560 samples from 7 runs; every sample also carries the run parameters (Cs, D, Re, height, length, save_every, seed, steps, u_in).

| field | kind | shape / type | range | units |
|---|---|---|---|---|
| vorticity | array | [240, 100] float16 | -17.69 .. 17.69 | U/D |
| Re | scalar | float  | 40 .. 40 |  |
| regime | scalar | str  | steady |  |
| step | scalar | int  | 8000 .. 1.59e+04 |  |
| strouhal | scalar | float  | 0 .. 0 |  |

Schema ranges are those of one run (`cylinder_lbm-de04dbbd3e8c`).

```bash
python -m pinneapple_lab sweep cylinder_lbm -g Cs=0.0,0.1 -g Re=100,140,180,20,40,60,80
python -m pinneapple_lab export cylinder_lbm vorticity cylinder_lbm_vorticity.npz
```

## `heat_xtfc` / `fields`

u(x, t) of the heat equation on a 101 x 51 grid

8 samples from 8 runs; every sample also carries the run parameters (activation, alpha, n_basis, n_collocation, seed).

| field | kind | shape / type | range | units |
|---|---|---|---|---|
| u | array | [101, 51] float32 | 0 .. 1 | - |
| u_exact | array | [101, 51] float32 | 0 .. 1 |  |
| alpha | scalar | float  | 1 .. 1 |  |

Schema ranges are those of one run (`heat_xtfc-fe8b4a9756b7`).

```bash
python -m pinneapple_lab sweep heat_xtfc -g activation=sin,tanh -g alpha=0.01,0.1,1.0 -g n_basis=20,60
python -m pinneapple_lab export heat_xtfc fields heat_xtfc_fields.npz
```

## `kepler_law` / `orbits`

Measured orbits (NASA fact sheets)

19 samples from 1 runs; every sample also carries the run parameters (exponent_grid, seed).

| field | kind | shape / type | range | units |
|---|---|---|---|---|
| system | scalar | str  | Sun, Jupiter, Saturn |  |
| body | scalar | str  | Mercury, Venus, Earth, Mars, Jupiter, Saturn |  |
| a | scalar | float  | 1.855e+08 .. 5.906e+12 | m |
| P | scalar | float  | 8.139e+04 .. 7.824e+09 | s |

Schema ranges are those of one run (`kepler_law-2c94b68c58a3`).

```bash
python -m pinneapple_lab sweep kepler_law
python -m pinneapple_lab export kepler_law orbits kepler_law_orbits.npz
```

## `oscillator` / `trajectories`

Oscillator trajectories x(t) with parameters and regime

60 samples from 60 runs; every sample also carries the run parameters (dt, method, omega, seed, t_end, zeta).

| field | kind | shape / type | range | units |
|---|---|---|---|---|
| t | array | [1357] float32 | 0 .. 20 | s |
| x | array | [1357] float32 | -0.9266 .. 1 | m |
| x_exact | array | [1357] float32 | -0.7396 .. 1 |  |
| zeta | scalar | float  | 0.09549 .. 0.09549 |  |
| omega | scalar | float  | 9.654 .. 9.654 |  |
| method | scalar | str  | euler |  |
| regime | scalar | str  | underdamped |  |

Schema ranges are those of one run (`oscillator-3588216c6c4b`).

```bash
python -m pinneapple_lab sweep oscillator -n 60
python -m pinneapple_lab export oscillator trajectories oscillator_trajectories.npz
```

## `oscillator_discovery` / `discovered_laws`

Recovered oscillator parameters per trajectory

20 samples from 1 runs; every sample also carries the run parameters (max_runs, seed, threshold).

| field | kind | shape / type | range | units |
|---|---|---|---|---|
| zeta_true | scalar | float  | 0.07064 .. 1.494 |  |
| zeta_found | scalar | float  | 0.09797 .. 1.495 |  |
| omega_true | scalar | float  | 0.5589 .. 7.398 |  |
| omega_found | scalar | float  | 0.5589 .. 7.398 |  |
| terms | scalar | str  | x,x', x,x',x², x,x',x x' |  |

Schema ranges are those of one run (`oscillator_discovery-d0a08fc3b799`).

```bash
python -m pinneapple_lab sweep oscillator_discovery
python -m pinneapple_lab export oscillator_discovery discovered_laws oscillator_discovery_discovered_laws.npz
```

## `pendulum_video` / `pendulum`

Video frames, measured angle and the true angle

1 samples from 1 runs; every sample also carries the run parameters (L, damping, fps, g, n_test, noise, predict_seconds, seconds, seed, size, test_width, theta0).

| field | kind | shape / type | range | units |
|---|---|---|---|---|
| frames | array | [600, 160, 160] uint8 | 0 .. 255 |  |
| theta_measured | array | [600] float32 | -2.077 .. 2.393 |  |
| theta_true | array | [600] float32 | -2.078 .. 2.4 |  |
| g | scalar | float  | 9.81 .. 9.81 |  |
| L | scalar | float  | 0.8 .. 0.8 |  |
| damping | scalar | float  | 0.15 .. 0.15 |  |
| fps | scalar | int  | 60 .. 60 |  |

Schema ranges are those of one run (`pendulum_video-7a4bc9625e0e`).

```bash
python -m pinneapple_lab sweep pendulum_video
python -m pinneapple_lab export pendulum_video pendulum pendulum_video_pendulum.npz
```

## `repo_results` / `coefficients`

Ahmed body force coefficients (OpenFOAM, medium mesh)

1 samples from 1 runs; every sample also carries the run parameters (source).

| field | kind | shape / type | range | units |
|---|---|---|---|---|
| CD | scalar | float  | 0.2977 .. 0.2977 |  |
| CL | scalar | float  | 0.3368 .. 0.3368 |  |
| CD_pressure | scalar | float  | 0.2429 .. 0.2429 |  |
| CD_friction | scalar | float  | 0.05474 .. 0.05474 |  |
| cells | scalar | int  | 2.039e+05 .. 2.039e+05 |  |
| CD_measured | scalar | float  | 0.285 .. 0.285 |  |

Schema ranges are those of one run (`repo_results-76a9a3bf2c2f`).

```bash
python -m pinneapple_lab sweep repo_results -g source=ahmed_body,airfoil_surrogate,bh_forecast,burgers_pinn,concorde_aoa,fin_inverse_2d,fin_inverse_3d,heatsink_surrogate,lbm_strouhal,meshgraphnet
python -m pinneapple_lab export repo_results coefficients repo_results_coefficients.npz
```

## `repo_results` / `fields`

Burgers u(t, x): PINN prediction and exact solution

1 samples from 1 runs; every sample also carries the run parameters (source).

| field | kind | shape / type | range | units |
|---|---|---|---|---|
| x | array | [200] float32 | -1 .. 1 | - |
| t | array | [200] float32 | 0 .. 1 | - |
| u_pred | array | [200, 200] float32 | -0.9766 .. 0.9796 |  |
| u_exact | array | [200, 200] float32 | -1 .. 1 |  |
| nu | scalar | float  | 0.01 .. 0.01 |  |
| model | scalar | str  | pinn_mlp |  |
| source | scalar | str  | burgers_pinn |  |

Schema ranges are those of one run (`repo_results-96e87a293dea`).

```bash
python -m pinneapple_lab sweep repo_results -g source=ahmed_body,airfoil_surrogate,bh_forecast,burgers_pinn,concorde_aoa,fin_inverse_2d,fin_inverse_3d,heatsink_surrogate,lbm_strouhal,meshgraphnet
python -m pinneapple_lab export repo_results fields repo_results_fields.npz
```

## `repo_results` / `held_out_error`

Surrogate error on held-out heat-sink designs, per output

3 samples from 1 runs; every sample also carries the run parameters (source).

| field | kind | shape / type | range | units |
|---|---|---|---|---|
| mode | scalar | str  | forced, natural |  |
| output | scalar | str  | thermal resistance (base to air), pressure drop |  |
| n_test | scalar | int  | 750 .. 900 |  |
| mean_abs_pct | scalar | float  | 0.7115 .. 1.041 |  |
| p95_abs_pct | scalar | float  | 2.086 .. 2.696 |  |
| max_abs_pct | scalar | float  | 5.88 .. 10 |  |
| pct_within_5pct | scalar | float  | 99.22 .. 99.89 |  |

Schema ranges are those of one run (`repo_results-33073ba61b6c`).

```bash
python -m pinneapple_lab sweep repo_results -g source=ahmed_body,airfoil_surrogate,bh_forecast,burgers_pinn,concorde_aoa,fin_inverse_2d,fin_inverse_3d,heatsink_surrogate,lbm_strouhal,meshgraphnet
python -m pinneapple_lab export repo_results held_out_error repo_results_held_out_error.npz
```

## `repo_results` / `lead_time_skill`

Mean absolute error vs lead time of the black-hole flow forecasters and of persistence, in and out of distribution

4 samples from 1 runs; every sample also carries the run parameters (source).

| field | kind | shape / type | range | units |
|---|---|---|---|---|
| model | scalar | str  | f16, res16 |  |
| test | scalar | str  | PL0SS0.1, PL0SS0.3 |  |
| lead | array | [60] float64 | 20 .. 1200 |  |
| mae | array | [60] float64 | 0.0004255 .. 0.0391 |  |
| persistence | array | [60] float64 | 0.0002477 .. 0.01935 |  |
| climatology | array | [60] float64 | 0.1007 .. 0.1446 |  |
| acc | array | [60] float64 | 0.961 .. 1 |  |
| horizon_beats_persistence | scalar | float  | 0 .. 1200 |  |

Schema ranges are those of one run (`repo_results-269f1207b6f2`).

```bash
python -m pinneapple_lab sweep repo_results -g source=ahmed_body,airfoil_surrogate,bh_forecast,burgers_pinn,concorde_aoa,fin_inverse_2d,fin_inverse_3d,heatsink_surrogate,lbm_strouhal,meshgraphnet
python -m pinneapple_lab export repo_results lead_time_skill repo_results_lead_time_skill.npz
```

## `repo_results` / `polar`

LBM-LES aerodynamic coefficients vs angle of attack (Re 300, coarse)

5 samples from 1 runs; every sample also carries the run parameters (source).

| field | kind | shape / type | range | units |
|---|---|---|---|---|
| aoa_deg | scalar | float  | 0 .. 20 |  |
| CL | scalar | float  | -0.03325 .. 0.5881 |  |
| CD | scalar | float  | 0.109 .. 0.615 |  |
| CM | scalar | float  | -0.2873 .. -0.01799 |  |
| CY | scalar | float  | 0.007121 .. 0.02407 |  |
| Cp_min | scalar | float  | -1.506 .. -1.488 |  |
| Cp_max | scalar | float  | 1.174 .. 1.283 |  |
| enstrophy | scalar | float  | 377.6 .. 438.7 |  |
| CL_polhamus | scalar | float  | 0 .. 1.026 |  |

Schema ranges are those of one run (`repo_results-ae7814d939d6`).

```bash
python -m pinneapple_lab sweep repo_results -g source=ahmed_body,airfoil_surrogate,bh_forecast,burgers_pinn,concorde_aoa,fin_inverse_2d,fin_inverse_3d,heatsink_surrogate,lbm_strouhal,meshgraphnet
python -m pinneapple_lab export repo_results polar repo_results_polar.npz
```

## `repo_results` / `rollout_scores`

MeshGraphNet rollout and one-step RMSE vs the frozen-initial-condition baseline

2 samples from 1 runs; every sample also carries the run parameters (source).

| field | kind | shape / type | range | units |
|---|---|---|---|---|
| case | scalar | str  | synthetic_diffusion, cylinder_flow |  |
| rollout_rmse | scalar | float  | 0.02968 .. 0.0711 |  |
| one_step_rmse | scalar | float  | 0.009105 .. 0.02652 |  |
| baseline_rmse | scalar | float  | 0.09619 .. 0.1104 |  |
| n_steps | scalar | int  | 20 .. 50 |  |

Schema ranges are those of one run (`repo_results-cbac6f31f681`).

```bash
python -m pinneapple_lab sweep repo_results -g source=ahmed_body,airfoil_surrogate,bh_forecast,burgers_pinn,concorde_aoa,fin_inverse_2d,fin_inverse_3d,heatsink_surrogate,lbm_strouhal,meshgraphnet
python -m pinneapple_lab export repo_results rollout_scores repo_results_rollout_scores.npz
```

## `repo_results` / `strouhal`

LBM vortex-shedding runs: Strouhal number, set-up and probe amplitude

3 samples from 1 runs; every sample also carries the run parameters (source).

| field | kind | shape / type | range | units |
|---|---|---|---|---|
| case | scalar | str  | cylinder, naca4412 |  |
| tag | scalar | str  | cylinder_h10, cylinder_h20, naca4412 |  |
| Re | scalar | float  | 100 .. 500 |  |
| St | scalar | float  | 0.1742 .. 0.2068 |  |
| St_reference | scalar | NoneType  | 0.1647 .. 0.1647 |  |
| length_scale_cells | scalar | float  | 16 .. 27.36 |  |
| u_in | scalar | float  | 0.1 .. 0.1 |  |
| steps | scalar | int  | 2e+04 .. 2e+04 |  |
| periods | scalar | float  | 7.558 .. 11.97 |  |
| probe_uy_amplitude | scalar | float  | 0.0577 .. 0.07266 |  |

Schema ranges are those of one run (`repo_results-fa808c1e7a31`).

```bash
python -m pinneapple_lab sweep repo_results -g source=ahmed_body,airfoil_surrogate,bh_forecast,burgers_pinn,concorde_aoa,fin_inverse_2d,fin_inverse_3d,heatsink_surrogate,lbm_strouhal,meshgraphnet
python -m pinneapple_lab export repo_results strouhal repo_results_strouhal.npz
```

## `repo_results` / `temperature_fields_2d`

2d temperature field: finite-volume reference and PINN reconstructions from noisy sensors (one per noise draw)

5 samples from 1 runs; every sample also carries the run parameters (source).

| field | kind | shape / type | range | units |
|---|---|---|---|---|
| T_pinn | array | [60, 100] float32 | 62.3 .. 79.38 |  |
| T_reference | array | [60, 100] float32 | 62.5 .. 79.13 |  |
| draw | scalar | int  | 0 .. 4 |  |
| h_pinn | scalar | float  | 14.39 .. 15.29 |  |
| h_true | scalar | float  | 15 .. 15 |  |
| case | scalar | str  | 2d_plate |  |

Schema ranges are those of one run (`repo_results-8f10f262c734`).

```bash
python -m pinneapple_lab sweep repo_results -g source=ahmed_body,airfoil_surrogate,bh_forecast,burgers_pinn,concorde_aoa,fin_inverse_2d,fin_inverse_3d,heatsink_surrogate,lbm_strouhal,meshgraphnet
python -m pinneapple_lab export repo_results temperature_fields_2d repo_results_temperature_fields_2d.npz
```

## `repo_results` / `temperature_fields_3d`

3d temperature field: finite-volume reference and PINN reconstructions from noisy sensors (one per noise draw)

3 samples from 1 runs; every sample also carries the run parameters (source).

| field | kind | shape / type | range | units |
|---|---|---|---|---|
| T_pinn | array | [12, 40, 40] float32 | 65.96 .. 74.56 |  |
| T_reference | array | [12, 40, 40] float32 | 65.91 .. 74.48 |  |
| draw | scalar | int  | 0 .. 2 |  |
| h_pinn | scalar | float  | 137.9 .. 141 |  |
| h_true | scalar | float  | 150 .. 150 |  |
| case | scalar | str  | 3d_block |  |

Schema ranges are those of one run (`repo_results-e9574ff86021`).

```bash
python -m pinneapple_lab sweep repo_results -g source=ahmed_body,airfoil_surrogate,bh_forecast,burgers_pinn,concorde_aoa,fin_inverse_2d,fin_inverse_3d,heatsink_surrogate,lbm_strouhal,meshgraphnet
python -m pinneapple_lab export repo_results temperature_fields_3d repo_results_temperature_fields_3d.npz
```

## `repo_results` / `test_points`

Held-out airfoil cases: OpenFOAM coefficients and surrogate predictions

120 samples from 1 runs; every sample also carries the run parameters (source).

| field | kind | shape / type | range | units |
|---|---|---|---|---|
| model | scalar | str  | mlp, gnn |  |
| case | scalar | str  | s003_a0, s003_a1, s003_a2, s003_a3, s007_a0, s007_a1 |  |
| alpha | scalar | float  | -1.494 .. 13.19 | deg |
| true | array | [3] float64 | -0.08594 .. 1.493 |  |
| pred | array | [3] float64 | -0.08529 .. 1.493 |  |

Schema ranges are those of one run (`repo_results-d8795900891a`).

```bash
python -m pinneapple_lab sweep repo_results -g source=ahmed_body,airfoil_surrogate,bh_forecast,burgers_pinn,concorde_aoa,fin_inverse_2d,fin_inverse_3d,heatsink_surrogate,lbm_strouhal,meshgraphnet
python -m pinneapple_lab export repo_results test_points repo_results_test_points.npz
```

## `vehicle_cfd` / `streamlines`

Streamlines (points) and the speed along them, |U|/U

40 samples from 1 runs; every sample also carries the run parameters (alpha, iterations, keep_case, procs, resolution, samples, slant_deg, speed, style, surface_level, vehicle).

| field | kind | shape / type | range | units |
|---|---|---|---|---|
| points | array | [103, 3] float32 | -1.38 .. 10.16 |  |
| speed | array | [103] float32 | 0.1634 .. 1.332 |  |

Schema ranges are those of one run (`vehicle_cfd-19b6c3f7bb85`).

```bash
python -m pinneapple_lab sweep vehicle_cfd
python -m pinneapple_lab export vehicle_cfd streamlines vehicle_cfd_streamlines.npz
```

## `vehicle_cfd` / `surface`

Skin pressure and friction coefficients on the wall faces (half model, y >= 0) with the force per face

1 samples from 1 runs; every sample also carries the run parameters (alpha, iterations, keep_case, procs, resolution, samples, slant_deg, speed, style, surface_level, vehicle).

| field | kind | shape / type | range | units |
|---|---|---|---|---|
| xyz | array | [5130, 3] float32 | 0 .. 4.6 | m |
| cp | array | [5130] float32 | -4.151 .. 0.9996 |  |
| cf | array | [5130] float32 | 1.561e-06 .. 0.01143 |  |
| force | array | [5130, 3] float32 | -1.743 .. 3.826 |  |
| vehicle | scalar | str  | car |  |
| CD | scalar | float  | 0.2883 .. 0.2883 |  |
| CL | scalar | float  | 0.01798 .. 0.01798 |  |

Schema ranges are those of one run (`vehicle_cfd-19b6c3f7bb85`).

```bash
python -m pinneapple_lab sweep vehicle_cfd
python -m pinneapple_lab export vehicle_cfd surface vehicle_cfd_surface.npz
```
