# PINNeAPPle Lab catalogue

208 runs, generated 2026-10-18 16:42.

| experiment | runs | completed | failed validation | failed | datasets (samples) |
|---|---|---|---|---|---|
| [accretion_flow](#accretion_flow) | 6 | 6 | 0 | 0 | frames (186) |
| [bar_wear](#bar_wear) | 8 | 8 | 0 | 0 | wear_history (8) |
| [bar_wear_ranking](#bar_wear_ranking) | 1 | 1 | 0 | 0 | ranking (8) |
| [benchmark_case](#benchmark_case) | 7 | 7 | 0 | 0 | arrays (84), sweep (8), lead_skill (5) |
| [bondi_accretion](#bondi_accretion) | 6 | 6 | 0 | 0 | profiles (6) |
| [bricks](#bricks) | 3 | 3 | 0 | 0 | - |
| [calculix_case](#calculix_case) | 5 | 5 | 0 | 0 | fields (5) |
| [car_lbm](#car_lbm) | 40 | 40 | 0 | 0 | flow (40), vorticity (1920) |
| [cylinder_lbm](#cylinder_lbm) | 9 | 7 | 2 | 0 | vorticity (720) |
| [em_pinn](#em_pinn) | 4 | 2 | 2 | 0 | - |
| [example_script](#example_script) | 8 | 5 | 0 | 3 | - |
| [heat_xtfc](#heat_xtfc) | 12 | 8 | 4 | 0 | fields (12) |
| [kepler_law](#kepler_law) | 1 | 1 | 0 | 0 | orbits (19) |
| [lorenz_discovery](#lorenz_discovery) | 1 | 1 | 0 | 0 | - |
| [oscillator](#oscillator) | 60 | 60 | 0 | 0 | trajectories (60) |
| [oscillator_discovery](#oscillator_discovery) | 1 | 1 | 0 | 0 | discovered_laws (20) |
| [particle_suspension](#particle_suspension) | 1 | 1 | 0 | 0 | suspension (31) |
| [pendulum_video](#pendulum_video) | 1 | 1 | 0 | 0 | pendulum (1) |
| [pipe_flow](#pipe_flow) | 3 | 3 | 0 | 0 | cross_sections (14), sections (3), wall (3) |
| [repo_results](#repo_results) | 25 | 24 | 1 | 0 | strouhal (3), test_points (120), polar (5), coefficients (1), temperature_fields_2d (5), rollout_scores (2), temperature_fields_3d (3), lead_time_skill (4), fields (1), held_out_error (3), torque_vs_current_angle (13), single_wheel_validation (5) |
| [rom_study](#rom_study) | 2 | 2 | 0 | 0 | beam_designs (100), wake (1) |
| [solid_fem](#solid_fem) | 3 | 3 | 0 | 0 | fields (3) |
| [vehicle_cfd](#vehicle_cfd) | 1 | 1 | 0 | 0 | streamlines (40), surface (1) |

## accretion_flow

A hot torus accreting onto a Schwarzschild black hole (viscous 2.5-D hydro): density movie, accretion rate and mass / angular-momentum budgets. Frames form a dataset for forecasting.

| run | status | alpha | backend | every | nr | ntheta | seed | t_end | torus_a | viscosity | angmom_budget_error | capped_cells_last_step | frames | mass_budget_error | mdot_final | mdot_mean | checks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [d611a6352627](runs/accretion_flow/accretion_flow-d611a6352627) | completed | 0.05 | numba | 20.0 | 64 | 32 | 0 | 600.0 | 0.0 | SS | 6.086e-06 | 39 | 31 | 8.263e-16 | 12 | 5.965 | 3/3 |
| [2b4c5b53625d](runs/accretion_flow/accretion_flow-2b4c5b53625d) | completed | 0.05 | numba | 20.0 | 64 | 32 | 0 | 600.0 | 0.0 | ST | 1.407e-06 | 41 | 31 | 4.131e-16 | 7.284 | 3.834 | 3/3 |
| [2e487e9f3b08](runs/accretion_flow/accretion_flow-2e487e9f3b08) | completed | 0.1 | numba | 20.0 | 64 | 32 | 0 | 600.0 | 0.0 | SS | 6.517e-06 | 30 | 31 | 0 | 17.06 | 13.6 | 3/3 |
| [9ee81c81f58c](runs/accretion_flow/accretion_flow-9ee81c81f58c) | completed | 0.1 | numba | 20.0 | 64 | 32 | 0 | 600.0 | 0.0 | ST | 2.104e-06 | 38 | 31 | 0 | 11.75 | 8.522 | 3/3 |
| [4a0c1426e768](runs/accretion_flow/accretion_flow-4a0c1426e768) | completed | 0.2 | numba | 20.0 | 64 | 32 | 0 | 600.0 | 0.0 | SS | 3.978e-06 | 12 | 31 | 3.03e-11 | 19.49 | 23.11 | 3/3 |
| [069c9c145e88](runs/accretion_flow/accretion_flow-069c9c145e88) | completed | 0.2 | numba | 20.0 | 64 | 32 | 0 | 600.0 | 0.0 | ST | 7.946e-07 | 24 | 31 | 1.446e-15 | 16.15 | 15.31 | 3/3 |

![accretion_flow](runs/accretion_flow/accretion_flow-069c9c145e88/figures/density.png)

![accretion_flow](runs/accretion_flow/accretion_flow-4a0c1426e768/figures/density.png)

## bar_wear

A crowned bar end slid on a flat counterface: Archard wear with the contact pressure redistributed on an elastic layer, from the first Hertz-like contact through running-in to steady flat-punch wear; one run per material (classic Archard & Hirst data).

| run | status | crown_radius_mm | distance_runins | length_mm | load_N | material | render | width_mm | K_mm3_per_Nm | distance_m | initial_peak_pressure_MPa | life_to_1mm_km | running_in_distance_m | steady_pressure_MPa | checks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [1e7e0db2cc3a](runs/bar_wear/bar_wear-1e7e0db2cc3a) | completed | 400.0 | 2.0 | 20.0 | 50.0 | PTFE | True | 10.0 | 0.0005 | 1333 | 1.245 | 8 | 649.9 | 0.2494 | 4/4 |
| [238870e1e723](runs/bar_wear/bar_wear-238870e1e723) | completed | 400.0 | 2.0 | 20.0 | 50.0 | mild steel | True | 10.0 | 0.003763 | 177.1 | 9.344 | 1.063 | 88.68 | 0.2494 | 4/4 |
| [4a62551f9d4b](runs/bar_wear/bar_wear-4a62551f9d4b) | completed | 400.0 | 2.0 | 20.0 | 50.0 | 60/40 brass | True | 10.0 | 0.0006316 | 1056 | 7.28 | 6.333 | 528.7 | 0.2494 | 4/4 |
| [bfb09c28a716](runs/bar_wear/bar_wear-bfb09c28a716) | completed | 400.0 | 2.0 | 20.0 | 50.0 | hardened tool steel | True | 10.0 | 1.529e-05 | 4.359e+04 | 9.344 | 261.5 | 2.182e+04 | 0.2494 | 4/4 |
| [dde80d2b6c0a](runs/bar_wear/bar_wear-dde80d2b6c0a) | completed | 400.0 | 2.0 | 20.0 | 50.0 | stellite | True | 10.0 | 7.971e-06 | 8.364e+04 | 9.615 | 501.8 | 4.187e+04 | 0.2494 | 4/4 |
| [da9f0a75bd03](runs/bar_wear/bar_wear-da9f0a75bd03) | completed | 400.0 | 2.0 | 20.0 | 50.0 | ferritic stainless steel | True | 10.0 | 6.8e-06 | 9.804e+04 | 9.179 | 588.2 | 4.908e+04 | 0.2494 | 4/4 |
| [b6099cd17ea5](runs/bar_wear/bar_wear-b6099cd17ea5) | completed | 400.0 | 2.0 | 20.0 | 50.0 | polyethylene | True | 10.0 | 7.647e-07 | 8.718e+05 | 1.569 | 5231 | 4.309e+05 | 0.2494 | 4/4 |
| [ccdb0fae6186](runs/bar_wear/bar_wear-ccdb0fae6186) | completed | 400.0 | 2.0 | 20.0 | 50.0 | tungsten carbide | True | 10.0 | 7.692e-08 | 8.667e+06 | 13.21 | 5.2e+04 | 4.336e+06 | 0.2494 | 4/4 |

![bar_wear](runs/bar_wear/bar_wear-ccdb0fae6186/figures/wear_profiles.png)

![bar_wear](runs/bar_wear/bar_wear-b6099cd17ea5/figures/wear_profiles.png)

## bar_wear_ranking

Every bar_wear run side by side: specific wear rate, life to 1 mm of wear at the same load and the ranking of the materials, checked against the ranking of their Archard coefficients K = k/H.

| run | status | seed | best_material_life_km | materials | worst_material_life_km | checks |
|---|---|---|---|---|---|---|
| [0f0d5ccb2871](runs/bar_wear_ranking/bar_wear_ranking-0f0d5ccb2871) | completed | 0 | 5.2e+04 | 8 | 1.063 | 1/1 |

![bar_wear_ranking](runs/bar_wear_ranking/bar_wear_ranking-0f0d5ccb2871/figures/wear_life_ranking.png)

## benchmark_case

Cases from the PINNeAPPle Labs landing page that live in the public PINNeAPPle-Benchmark and PINNeAPPle-Climate repositories, imported with their headline claim re-checked, figures, the arrays behind the paper's figures and the source that produced them.

| run | status | case | Climatology|rmse_lead7_degC | Conv autoencoder + latent MLP|test_same_family|T_rel_l2 | Conv autoencoder + latent MLP|test_unseen_sinusoid|T_rel_l2 | Damped persistence (AR1)|rmse_lead7_degC | DeepONet (PINNeAPPle)|test_same_family|T_rel_l2 | DeepONet (PINNeAPPle)|test_unseen_sinusoid|T_rel_l2 | checks |
|---|---|---|---|---|---|---|---|---|---|
| [2814207947c8](runs/benchmark_case/benchmark_case-2814207947c8) | completed | soil_twin |  |  |  |  |  |  | 2/2 |
| [2a494685fed3](runs/benchmark_case/benchmark_case-2a494685fed3) | completed | bumper_crash |  |  |  |  |  |  | 2/2 |
| [e5981ba8bbc9](runs/benchmark_case/benchmark_case-e5981ba8bbc9) | completed | terramechanics |  |  |  |  |  |  | 2/2 |
| [b5fa8d70f3f9](runs/benchmark_case/benchmark_case-b5fa8d70f3f9) | completed | heated_channel |  | 0.09345 | 0.1409 |  | 0.8398 | 0.8314 | 2/2 |
| [b8f46b4fe34f](runs/benchmark_case/benchmark_case-b8f46b4fe34f) | completed | pdr |  |  |  |  |  |  | 2/2 |
| [3074f671a9d8](runs/benchmark_case/benchmark_case-3074f671a9d8) | completed | sst | 1.574 |  |  | 1.018 |  |  | 1/1 |
| [9cf13dffb830](runs/benchmark_case/benchmark_case-9cf13dffb830) | completed | shock_train |  |  |  |  |  |  | 2/2 |

![benchmark_case](runs/benchmark_case/benchmark_case-9cf13dffb830/figures/fig_duct3d.png)

![benchmark_case](runs/benchmark_case/benchmark_case-3074f671a9d8/figures/fig1_region.png)

## bondi_accretion

Spherical accretion onto a Schwarzschild black hole (Paczynski-Wiita potential): the hydro solver started from the exact transonic Bondi solution must keep it steady.

| run | status | backend | cs_inf | gamma | nr | t_end | density_rel_error_max | density_rel_error_median | mdot_rel_error | sonic_radius | checks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [2606e5a668d9](runs/bondi_accretion/bondi_accretion-2606e5a668d9) | completed | numba | 0.08 | 1.3 | 64 | 300.0 | 0.06416 | 0.0004116 | 0.006652 | 48.81 | 2/2 |
| [b2e48df5c514](runs/bondi_accretion/bondi_accretion-b2e48df5c514) | completed | numba | 0.08 | 1.4 | 64 | 300.0 | 0.04122 | 0.0002985 | 0.007202 | 38.4 | 2/2 |
| [74ab4e4f9274](runs/bondi_accretion/bondi_accretion-74ab4e4f9274) | completed | numba | 0.1 | 1.3 | 64 | 300.0 | 0.04508 | 0.0005541 | 0.007008 | 33.19 | 2/2 |
| [3a8c3789e109](runs/bondi_accretion/bondi_accretion-3a8c3789e109) | completed | numba | 0.1 | 1.4 | 64 | 300.0 | 0.02593 | 0.0004488 | 0.007716 | 26.83 | 2/2 |
| [24fd3dba0cb6](runs/bondi_accretion/bondi_accretion-24fd3dba0cb6) | completed | numba | 0.14 | 1.3 | 64 | 300.0 | 0.02397 | 0.0009125 | 0.007964 | 19.4 | 2/2 |
| [42fdab9aa232](runs/bondi_accretion/bondi_accretion-42fdab9aa232) | completed | numba | 0.14 | 1.4 | 64 | 300.0 | 0.02302 | 0.0007038 | 0.008969 | 16.44 | 2/2 |

![bondi_accretion](runs/bondi_accretion/bondi_accretion-42fdab9aa232/figures/bondi_profile.png)

![bondi_accretion](runs/bondi_accretion/bondi_accretion-24fd3dba0cb6/figures/bondi_profile.png)

## bricks

Self-modifying interlocking-brick builds: rigid-body statics with stud clutch judge every candidate, an evolutionary and a surrogate-guided search redesign it at equal evaluation budget.

| run | status | budget | case | gif | newton | seeds | best_parts | best_score | evaluations_per_run | evolution_best_mean | evolution_best_std | evolution_success_rate | checks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [f119f554b32e](runs/bricks/bricks-f119f554b32e) | completed | 200 | tower | True | True | 3 | 14 | 134.4 | 200 | 65.07 | 5.439 | 1 | 9/9 |
| [5d008732488d](runs/bricks/bricks-5d008732488d) | completed | 200 | cantilever | True | True | 3 | 8 | 119.9 | 200 | 95.9 | 17.28 | 1 | 9/9 |
| [0613c73f2edf](runs/bricks/bricks-0613c73f2edf) | completed | 200 | bridge | True | True | 3 | 5 | -5 | 200 | -6 | 1.414 | 1 | 9/9 |

![bricks](runs/bricks/bricks-0613c73f2edf/figures/comparison.png)

![bricks](runs/bricks/bricks-5d008732488d/figures/comparison.png)

## calculix_case

CalculiX studies built in Python (gmsh tetrahedra or structured hexahedra, node and face sets picked by geometry): stress concentration, a filleted bracket, modes, buckling and a cooling fin, each against its closed-form reference, with mesh convergence where it matters.

| run | status | case | render | samples | Kt_extrapolated | Kt_gci | Kt_net | Kt_net_heywood | buckling_load_kN | elements_mesh_0 | checks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [0f01ecf25f1a](runs/calculix_case/calculix_case-0f01ecf25f1a) | completed | modal_cantilever | True | 48 |  |  |  |  |  |  | 2/2 |
| [66bac137ed69](runs/calculix_case/calculix_case-66bac137ed69) | completed | buckling_column | True | 48 |  |  |  |  | 27.71 |  | 1/1 |
| [63ffe30172aa](runs/calculix_case/calculix_case-63ffe30172aa) | completed | fin_heat | True | 48 |  |  |  |  |  |  | 1/1 |
| [bf67ee84a752](runs/calculix_case/calculix_case-bf67ee84a752) | completed | l_bracket | True | 48 |  |  |  |  |  | 1.017e+04 | 3/3 |
| [a10042673f52](runs/calculix_case/calculix_case-a10042673f52) | completed | plate_hole | True | 48 | 2.577 | 0.01654 | 2.591 | 2.512 |  | 8976 | 5/5 |

![calculix_case](runs/calculix_case/calculix_case-a10042673f52/figures/von_mises_fea.png)

![calculix_case](runs/calculix_case/calculix_case-bf67ee84a752/figures/von_mises_fea.png)

## car_lbm

Virtual wind tunnel for a parametric 2-D car body (Ahmed-type, six design parameters): lattice-Boltzmann LES with moving road; drag and lift by momentum exchange, mean flow, wake vorticity movie. The training data of car_surrogate.

| run | status | Cs | Re | clearance | diffuser_deg | hood | length_cells | nose | slant_deg | steps | tunnel_height | u_in | windshield_deg | Cd | Cd_rms_fluctuation | Cl | Strouhal_frontal_height | mean_flow_divergence | checks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [01d0849fcd6b](runs/car_lbm/car_lbm-01d0849fcd6b) | completed | 0.1 | 500.0 | 0.10491302009992373 | 5.461080029102873 | 0.47558637617623484 | 64 | 0.23766190040470328 | 17.573344271776932 | 8000 | 2.5 | 0.08 | 33.7005406387837 | 1.652 | 0.1415 | 0.6574 | 0.12 | 0.0003255 | 3/3 |
| [4d9a736b17f6](runs/car_lbm/car_lbm-4d9a736b17f6) | completed | 0.1 | 500.0 | 0.11331448782654147 | 7.385592469388693 | 0.5832302256632885 | 64 | 0.38743462610548784 | 24.253267461795566 | 8000 | 2.5 | 0.08 | 44.8746927155908 | 1.726 | 0.1511 | 0.623 | 0.12 | 0.0003555 | 3/3 |
| [45dc6edd8075](runs/car_lbm/car_lbm-45dc6edd8075) | completed | 0.1 | 500.0 | 0.04420219793570932 | 3.096879659750433 | 0.7036141634121883 | 64 | 0.172873897568471 | 29.351291763576526 | 8000 | 2.5 | 0.08 | 27.458272594421672 | 1.419 | 0.1327 | 0.9005 | 0.12 | 0.0003347 | 3/3 |
| [e53e2fb92ce4](runs/car_lbm/car_lbm-e53e2fb92ce4) | completed | 0.1 | 500.0 | 0.08876116952700108 | 1.6370071920665097 | 0.6579749193675155 | 64 | 0.9010132214973694 | 38.385153445928765 | 8000 | 2.5 | 0.08 | 46.13870805749418 | 1.583 | 0.1485 | 0.7771 | 0.12 | 0.0003397 | 3/3 |
| [ddba6505f43d](runs/car_lbm/car_lbm-ddba6505f43d) | completed | 0.1 | 500.0 | 0.06626744237767704 | 4.680410054629916 | 0.5333490371439392 | 64 | 0.2578837703608595 | 7.859149391493312 | 8000 | 2.5 | 0.08 | 23.80118433362152 | 1.429 | 0.1304 | 0.8153 | 0.54 | 0.0003337 | 3/3 |
| [2cb049a4733a](runs/car_lbm/car_lbm-2cb049a4733a) | completed | 0.1 | 500.0 | 0.10732489244322849 | 9.845365217349599 | 0.7118501614614828 | 64 | 0.3149993674616432 | 26.77029063473322 | 8000 | 2.5 | 0.08 | 31.8915046095035 | 1.724 | 0.1554 | 0.6273 | 0.12 | 0.0003441 | 3/3 |
| [000726cc8ec0](runs/car_lbm/car_lbm-000726cc8ec0) | completed | 0.1 | 500.0 | 0.061661105064526364 | 0.20766016835299797 | 0.7260152743349797 | 64 | 0.6766599554625821 | 39.571066958634866 | 8000 | 2.5 | 0.08 | 52.973223826672104 | 1.516 | 0.1416 | 0.9295 | 0.12 | 0.0003317 | 3/3 |
| [500798921b1a](runs/car_lbm/car_lbm-500798921b1a) | completed | 0.1 | 500.0 | 0.11475370756281755 | 5.729865836576346 | 0.7190826852498715 | 64 | 0.8559137644101507 | 28.49125646557014 | 8000 | 2.5 | 0.08 | 20.310987054895573 | 1.712 | 0.1539 | 0.6115 | 0.12 | 0.0003417 | 3/3 |
| [cb2e9c63cc61](runs/car_lbm/car_lbm-cb2e9c63cc61) | completed | 0.1 | 500.0 | 0.07274344790554883 | 8.346858074327661 | 0.5964247811473138 | 64 | 0.9366265863524385 | 16.93475252107471 | 8000 | 2.5 | 0.08 | 34.454211175683994 | 1.498 | 0.1385 | 0.7661 | 0.84 | 0.0003373 | 3/3 |
| [e255b8b0b291](runs/car_lbm/car_lbm-e255b8b0b291) | completed | 0.1 | 500.0 | 0.057079043317894866 | 5.034454038587483 | 0.4718672657632555 | 64 | 0.12202142567173069 | 20.524049578941625 | 8000 | 2.5 | 0.08 | 36.41959146036871 | 1.409 | 0.1244 | 0.8292 | 0.54 | 0.0003316 | 3/3 |
| [a3d566adf5bc](runs/car_lbm/car_lbm-a3d566adf5bc) | completed | 0.1 | 500.0 | 0.08243011555927618 | 0.6900933239975455 | 0.5103498917846542 | 64 | 0.8190239880414447 | 20.830058654430164 | 8000 | 2.5 | 0.08 | 40.04553189974173 | 1.534 | 0.1381 | 0.8765 | 0.84 | 0.0003289 | 3/3 |
| [1ea0200595dd](runs/car_lbm/car_lbm-1ea0200595dd) | completed | 0.1 | 500.0 | 0.11849481918323354 | 2.9982372146871517 | 0.585227166254487 | 64 | 0.14572451173904352 | 36.97350084370443 | 8000 | 2.5 | 0.08 | 32.38217949331721 | 1.692 | 0.1477 | 0.6139 | 0.12 | 0.0003317 | 3/3 |
| [bdfcd6b96b2e](runs/car_lbm/car_lbm-bdfcd6b96b2e) | completed | 0.1 | 500.0 | 0.05065970456656483 | 7.036997037190515 | 0.7276516168001711 | 64 | 0.7190267443132328 | 36.0693007067628 | 8000 | 2.5 | 0.08 | 40.941741968269 | 1.401 | 0.1291 | 0.846 | 0.12 | 0.0003335 | 3/3 |
| [9ebb460d2144](runs/car_lbm/car_lbm-9ebb460d2144) | completed | 0.1 | 500.0 | 0.08091485139627642 | 1.4281689190153664 | 0.5568957650846434 | 64 | 0.5926930194012445 | 38.13742195460627 | 8000 | 2.5 | 0.08 | 36.6295315075885 | 1.534 | 0.1344 | 0.8532 | 0.84 | 0.00034 | 3/3 |
| [2c4ed9779ce1](runs/car_lbm/car_lbm-2c4ed9779ce1) | completed | 0.1 | 500.0 | 0.09016306292060514 | 7.95633085007501 | 0.5643642732990674 | 64 | 0.3462423335661612 | 12.316106051637387 | 8000 | 2.5 | 0.08 | 44.28385531564803 | 1.604 | 0.1556 | 0.7049 | 0.12 | 0.0003166 | 3/3 |
| [ee898deb4ed5](runs/car_lbm/car_lbm-ee898deb4ed5) | completed | 0.1 | 500.0 | 0.05350546427273401 | 3.4801429862490902 | 0.5714062750076697 | 64 | 0.5420369172684172 | 11.64841105880631 | 8000 | 2.5 | 0.08 | 47.83421097652759 | 1.461 | 0.133 | 0.9251 | 0.12 | 0.0003459 | 3/3 |
| [17377963edf2](runs/car_lbm/car_lbm-17377963edf2) | completed | 0.1 | 500.0 | 0.11715810914885852 | 8.030845353569394 | 0.5067529186378905 | 64 | 0.96839207648284 | 5.926845187052316 | 8000 | 2.5 | 0.08 | 41.119668545099756 | 1.694 | 0.1591 | 0.5942 | 0.12 | 0.0003541 | 3/3 |
| [0ebce63174b6](runs/car_lbm/car_lbm-0ebce63174b6) | completed | 0.1 | 500.0 | 0.076599387745781 | 1.0229524782879023 | 0.45779239380774783 | 64 | 0.6575411125688053 | 34.21417782597189 | 8000 | 2.5 | 0.08 | 49.24154069454904 | 1.534 | 0.1315 | 0.8505 | 0.84 | 0.0003334 | 3/3 |
| [4a95cc22c2e6](runs/car_lbm/car_lbm-4a95cc22c2e6) | completed | 0.1 | 500.0 | 0.08615509314470199 | 3.9969678954616334 | 0.6944279090087614 | 64 | 0.35419101807246856 | 18.40765680297593 | 8000 | 2.5 | 0.08 | 38.213349243245666 | 1.598 | 0.1574 | 0.7439 | 0.12 | 0.000343 | 3/3 |
| [04698aa89723](runs/car_lbm/car_lbm-04698aa89723) | completed | 0.1 | 500.0 | 0.07152636190280295 | 2.279189121275397 | 0.6837450836478541 | 64 | 0.8439131244788948 | 30.506424762936646 | 8000 | 2.5 | 0.08 | 53.262487290974036 | 1.543 | 0.144 | 0.8287 | 0.84 | 0.0003344 | 3/3 |
| [6f431a30a9bf](runs/car_lbm/car_lbm-6f431a30a9bf) | completed | 0.1 | 500.0 | 0.07426215797658264 | 6.294201889784224 | 0.5225840528142548 | 64 | 0.7291459323942165 | 10.964295840841691 | 8000 | 2.5 | 0.08 | 35.424904204383004 | 1.527 | 0.1414 | 0.8017 | 0.12 | 0.0003382 | 3/3 |
| [8a675bd1fb06](runs/car_lbm/car_lbm-8a675bd1fb06) | completed | 0.1 | 500.0 | 0.048266413284699355 | 2.143738233457255 | 0.6301580651583292 | 64 | 0.7979863962879993 | 19.332015400106524 | 8000 | 2.5 | 0.08 | 46.94389570083578 | 1.403 | 0.1331 | 0.941 | 0.12 | 0.0003328 | 3/3 |
| [0f01b8aa2bd5](runs/car_lbm/car_lbm-0f01b8aa2bd5) | completed | 0.1 | 500.0 | 0.08426136849161456 | 4.111568259072409 | 0.48232927647789897 | 64 | 0.639916070933903 | 25.981404398859816 | 8000 | 2.5 | 0.08 | 42.32387813496943 | 1.551 | 0.1328 | 0.7627 | 0.84 | 0.0003455 | 3/3 |
| [6940a081d993](runs/car_lbm/car_lbm-6940a081d993) | completed | 0.1 | 500.0 | 0.10016252462223196 | 5.937598047831958 | 0.5470375596514846 | 64 | 0.1832358528217956 | 27.391242731384285 | 8000 | 2.5 | 0.08 | 50.3851182429822 | 1.688 | 0.1518 | 0.688 | 0.12 | 0.0003458 | 3/3 |
| [46021785d3b3](runs/car_lbm/car_lbm-46021785d3b3) | completed | 0.1 | 500.0 | 0.1038127847308005 | 4.797639312039529 | 0.5290379728490794 | 64 | 0.5734160789163087 | 9.90442422210112 | 8000 | 2.5 | 0.08 | 28.94812054736 | 1.653 | 0.1467 | 0.6708 | 0.12 | 0.0003308 | 3/3 |
| [a21e93923403](runs/car_lbm/car_lbm-a21e93923403) | completed | 0.1 | 500.0 | 0.11053848859934676 | 9.47860694038998 | 0.45608690552060066 | 64 | 0.47887825679844614 | 7.308247008189791 | 8000 | 2.5 | 0.08 | 22.79870600449359 | 1.636 | 0.148 | 0.584 | 0.12 | 0.0003373 | 3/3 |
| [8b2e43cfcd6f](runs/car_lbm/car_lbm-8b2e43cfcd6f) | completed | 0.1 | 500.0 | 0.10861282115841372 | 9.05429871157693 | 0.642435195612088 | 64 | 0.037861658442381166 | 32.71689396333619 | 8000 | 2.5 | 0.08 | 21.192736081731557 | 1.684 | 0.1424 | 0.5954 | 0.12 | 0.0003372 | 3/3 |
| [7f14ab3828f8](runs/car_lbm/car_lbm-7f14ab3828f8) | completed | 0.1 | 500.0 | 0.05966558862000414 | 7.69227774754666 | 0.6120806308921014 | 64 | 0.877288851830406 | 34.88193951677232 | 8000 | 2.5 | 0.08 | 28.031980274121253 | 1.459 | 0.1269 | 0.845 | 0.84 | 0.0003416 | 3/3 |
| [07a280c137da](runs/car_lbm/car_lbm-07a280c137da) | completed | 0.1 | 500.0 | 0.09923984691942359 | 2.516900881318515 | 0.6464343951006367 | 64 | 0.5241356927271144 | 23.760274283789663 | 8000 | 2.5 | 0.08 | 50.927803759223416 | 1.692 | 0.1609 | 0.7648 | 0.12 | 0.000348 | 3/3 |
| [14188f61c799](runs/car_lbm/car_lbm-14188f61c799) | completed | 0.1 | 500.0 | 0.04637428677308144 | 4.368350646939291 | 0.6268079606119336 | 64 | 0.4643844015452907 | 14.834618466600833 | 8000 | 2.5 | 0.08 | 54.95460855373447 | 1.374 | 0.1349 | 0.8747 | 0.12 | 0.0003292 | 3/3 |

![car_lbm](runs/car_lbm/car_lbm-14188f61c799/figures/forces.png)

![car_lbm](runs/car_lbm/car_lbm-07a280c137da/figures/forces.png)

## cylinder_lbm

2D channel flow past a cylinder with the D2Q9 lattice-Boltzmann solver (pinneapple_simulation.numerical_solvers.cylinder_wake). Measures the wake regime (steady or vortex shedding) and the Strouhal number; saves vorticity snapshots labelled by regime and Re, a dataset for vision (classification / clustering of flow regimes, #415).

| run | status | Cs | D | Re | height | length | save_every | seed | steps | u_in | regime_shedding | strouhal | wake_amplitude | checks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [de04dbbd3e8c](runs/cylinder_lbm/cylinder_lbm-de04dbbd3e8c) | completed | 0.0 | 20 | 40 | 5 | 12 | 100 | 0 | 16000 | 0.1 | 0 | 0 | 4.7e-05 | 1/1 |
| [418823cceeb5](runs/cylinder_lbm/cylinder_lbm-418823cceeb5) | completed | 0.0 | 20 | 20 | 5 | 12 | 100 | 0 | 16000 | 0.1 | 0 | 0 | 6.269e-06 | 2/2 |
| [5e53b105e4b0](runs/cylinder_lbm/cylinder_lbm-5e53b105e4b0) | completed | 0.0 | 20 | 60 | 5 | 12 | 100 | 0 | 16000 | 0.1 | 1 | 0.225 | 0.04966 | 2/2 |
| [4347d6fb9da9](runs/cylinder_lbm/cylinder_lbm-4347d6fb9da9) | completed | 0.0 | 20 | 80 | 5 | 12 | 100 | 0 | 16000 | 0.1 | 1 | 0.25 | 0.3688 | 2/2 |
| [e254136572c5](runs/cylinder_lbm/cylinder_lbm-e254136572c5) | completed | 0.0 | 20 | 100 | 5 | 12 | 100 | 0 | 16000 | 0.1 | 1 | 0.25 | 0.5283 | 2/2 |
| [e3321156b9d5](runs/cylinder_lbm/cylinder_lbm-e3321156b9d5) | failed_validation | 0.0 | 20 | 140 | 5 | 12 | 100 | 0 | 16000 | 0.1 | 0 | 0 |  | 0/1 |
| [b20bedf5d841](runs/cylinder_lbm/cylinder_lbm-b20bedf5d841) | failed_validation | 0.0 | 20 | 180 | 5 | 12 | 100 | 0 | 16000 | 0.1 | 0 | 0 |  | 0/1 |
| [d79b20689d8c](runs/cylinder_lbm/cylinder_lbm-d79b20689d8c) | completed | 0.1 | 20 | 180 | 5 | 12 | 100 | 0 | 16000 | 0.1 | 1 | 0.25 | 0.8159 | 2/2 |
| [ea9cf6215d98](runs/cylinder_lbm/cylinder_lbm-ea9cf6215d98) | completed | 0.1 | 20 | 140 | 5 | 12 | 100 | 0 | 16000 | 0.1 | 1 | 0.25 | 0.7312 | 2/2 |

![cylinder_lbm](runs/cylinder_lbm/cylinder_lbm-ea9cf6215d98/figures/vorticity.png)

![cylinder_lbm](runs/cylinder_lbm/cylinder_lbm-d79b20689d8c/figures/vorticity.png)

## em_pinn

Physics-informed networks on the electromagnetics presets with closed-form solutions (electrostatics, magnetostatics, a TM cavity, the skin effect), scored against the exact field.

| run | status | case | epochs | n_collocation | seed | tolerance | n_parameters | rel_l2_Ai | rel_l2_Ar | rel_l2_Az | rel_l2_Ez | rel_l2_Hx | checks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [d750874a636b](runs/em_pinn/em_pinn-d750874a636b) | failed_validation | em_tm_cavity | 3000 | 2048 | 0 | 0.05 | 2.1e+04 |  |  |  | 1.33 | 0.9954 | 1/4 |
| [276d8c0f9edf](runs/em_pinn/em_pinn-276d8c0f9edf) | failed_validation | em_two_dielectric_capacitor | 3000 | 2048 | 0 | 0.05 | 2.086e+04 |  |  |  |  |  | 1/2 |
| [435dc2e1ca9c](runs/em_pinn/em_pinn-435dc2e1ca9c) | completed | em_skin_effect | 3000 | 2048 | 0 | 0.05 | 2.093e+04 | 0.008055 | 0.002579 |  |  |  | 3/3 |
| [dc339d690726](runs/em_pinn/em_pinn-dc339d690726) | completed | em_wire_az | 3000 | 2048 | 0 | 0.05 | 2.086e+04 |  |  | 0.01594 |  |  | 2/2 |

![em_pinn](runs/em_pinn/em_pinn-dc339d690726/figures/em_wire_az_pinn_vs_exact.png)

![em_pinn](runs/em_pinn/em_pinn-435dc2e1ca9c/figures/em_skin_effect_pinn_vs_exact.png)

## example_script

Runs an example or use case of the repository and stores what it produced: figures, JSON outputs and their numbers as metrics, arrays as the 'artifacts' dataset, the console output and the code. The catalogue of these runs is also the health report of the examples.

| run | status | args | isolated | script | timeout | exit_code | files_produced | script_seconds | stdout.relative_l2 | stdout.relative_l2_error | checks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [5a77c5897b1d](runs/example_script/example_script-5a77c5897b1d) | completed |  | True | examples/getting_started/01_harmonic_oscillator.py | 600.0 | 0 | 1 | 94.8 |  | 0.915 | 1/1 |
| [0ba6838478e4](runs/example_script/example_script-0ba6838478e4) | completed |  | True | examples/getting_started/02_damped_oscillator.py | 600.0 | 0 | 1 | 137 | 0.5001 |  | 1/1 |
| [f151824795db](runs/example_script/example_script-f151824795db) | failed |  | True | examples/getting_started/03_heat_diffusion_1d.py | 600.0 | -1 | 0 | 600.1 | 0.0006818 |  | 0/1 |
| [511688b89ce8](runs/example_script/example_script-511688b89ce8) | failed |  | True | examples/getting_started/04_wave_equation_1d.py | 600.0 | -1 | 0 | 600.2 |  |  | 0/1 |
| [57ffa2145319](runs/example_script/example_script-57ffa2145319) | completed |  | True | examples/getting_started/05_logistic_growth.py | 600.0 | 0 | 1 | 52.72 | 0.007669 |  | 1/1 |
| [2ab998651866](runs/example_script/example_script-2ab998651866) | completed |  | True | examples/getting_started/06_lotka_volterra.py | 600.0 | 0 | 1 | 163.6 |  |  | 1/1 |
| [3c41e93eed63](runs/example_script/example_script-3c41e93eed63) | completed |  | True | examples/getting_started/07_nonlinear_pendulum.py | 600.0 | 0 | 1 | 219.4 | 1.03 |  | 1/1 |
| [3bdda3a8019a](runs/example_script/example_script-3bdda3a8019a) | failed |  | True | examples/getting_started/08_van_der_pol.py | 600.0 | -1 | 0 | 600.2 | 1.04 |  | 0/1 |

![example_script](runs/example_script/example_script-3c41e93eed63/figures/07_nonlinear_pendulum.png)

![example_script](runs/example_script/example_script-2ab998651866/figures/06_lotka_volterra.png)

## heat_xtfc

1D heat equation u_t = alpha u_xx on [0,1] x [0,1], u0 = sin(pi x), solved by X-TFC (pinneapple_simulation.numerical_solvers.xtfc_pde) vs exp(-alpha pi^2 t) sin(pi x).

| run | status | activation | alpha | n_basis | n_collocation | seed | max_abs_error | rel_l2_error | checks |
|---|---|---|---|---|---|---|---|---|---|
| [4c4adfeb5fc2](runs/heat_xtfc/heat_xtfc-4c4adfeb5fc2) | completed | tanh | 0.01 | 20 | 25 | 0 | 0.0001011 | 4.552e-05 | 1/1 |
| [2abcbcb6d7ce](runs/heat_xtfc/heat_xtfc-2abcbcb6d7ce) | completed | sin | 0.01 | 20 | 25 | 0 | 6.423e-07 | 4.075e-07 | 1/1 |
| [dbea79e4124e](runs/heat_xtfc/heat_xtfc-dbea79e4124e) | completed | tanh | 0.01 | 60 | 25 | 0 | 3.779e-07 | 2.085e-07 | 1/1 |
| [bb25d72596fb](runs/heat_xtfc/heat_xtfc-bb25d72596fb) | completed | sin | 0.01 | 60 | 25 | 0 | 1.173e-10 | 4.992e-11 | 1/1 |
| [7eef4ced7058](runs/heat_xtfc/heat_xtfc-7eef4ced7058) | failed_validation | tanh | 0.1 | 20 | 25 | 0 | 0.001695 | 0.001413 | 0/1 |
| [87910a1ec638](runs/heat_xtfc/heat_xtfc-87910a1ec638) | completed | sin | 0.1 | 20 | 25 | 0 | 2.171e-05 | 2.087e-05 | 1/1 |
| [94269745d788](runs/heat_xtfc/heat_xtfc-94269745d788) | completed | tanh | 0.1 | 60 | 25 | 0 | 2.814e-06 | 2.026e-06 | 1/1 |
| [22bc6dad5c93](runs/heat_xtfc/heat_xtfc-22bc6dad5c93) | completed | sin | 0.1 | 60 | 25 | 0 | 9.989e-10 | 6.945e-10 | 1/1 |
| [c7a9f86bbfb0](runs/heat_xtfc/heat_xtfc-c7a9f86bbfb0) | failed_validation | tanh | 1.0 | 20 | 25 | 0 | 0.031 | 0.06418 | 0/1 |
| [71e21e11957e](runs/heat_xtfc/heat_xtfc-71e21e11957e) | failed_validation | sin | 1.0 | 20 | 25 | 0 | 0.02571 | 0.04364 | 0/1 |
| [4ae0ba0dabeb](runs/heat_xtfc/heat_xtfc-4ae0ba0dabeb) | failed_validation | tanh | 1.0 | 60 | 25 | 0 | 0.0008588 | 0.001397 | 0/1 |
| [fe8b4a9756b7](runs/heat_xtfc/heat_xtfc-fe8b4a9756b7) | completed | sin | 1.0 | 60 | 25 | 0 | 0.0001163 | 0.0002403 | 1/1 |

![heat_xtfc](runs/heat_xtfc/heat_xtfc-fe8b4a9756b7/figures/solution.png)

![heat_xtfc](runs/heat_xtfc/heat_xtfc-4ae0ba0dabeb/figures/solution.png)

## kepler_law

Discover Kepler's third law from real orbital data (planets, moons of Jupiter and Saturn): sparse regression over power laws finds P ∝ a^n, n is compared with 3/2, the central masses come out as GM, and all three systems collapse onto P = 2π sqrt(a^3 / GM).

| run | status | exponent_grid | seed | Jupiter_GM_rel_error | Jupiter_exponent | Saturn_GM_rel_error | Saturn_exponent | Sun_GM_rel_error | Sun_exponent | checks |
|---|---|---|---|---|---|---|---|---|---|---|
| [2c94b68c58a3](runs/kepler_law/kepler_law-2c94b68c58a3) | completed | 0.25 | 0 | 0.0004799 | 1.5 | 0.00123 | 1.501 | 0.0005239 | 1.5 | 7/7 |

![kepler_law](runs/kepler_law/kepler_law-2c94b68c58a3/figures/kepler.png)

## lorenz_discovery

AI-Lorenz style equation discovery (De Florio, Kevrekidis & Karniadakis 2024): noisy, subsampled observations of the Lorenz system are fitted window by window with an extreme learning machine (the free function of X-TFC), whose analytic derivatives feed SINDy over quadratic monomials; the recovered system is integrated and compared with the true attractor.

| run | status | beta | dt | noise | rho | seed | sigma | subsample | t_end | threshold | beta | max_coefficient_rel_error | prediction_horizon_lyapunov_times | rho | sigma | checks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [f76bfaeab75f](runs/lorenz_discovery/lorenz_discovery-f76bfaeab75f) | completed | 2.6666666666666665 | 0.005 | 0.01 | 28.0 | 0 | 10.0 | 4 | 20.0 | 0.08 | 2.665 | 0.01778 | 3.801 | 28.03 | 10.02 | 2/2 |

![lorenz_discovery](runs/lorenz_discovery/lorenz_discovery-f76bfaeab75f/figures/lorenz_discovery.png)

## oscillator

Damped harmonic oscillator x'' + 2 zeta omega x' + omega^2 x = 0: integrator vs exact solution.

| run | status | dt | method | omega | seed | t_end | zeta | max_abs_error | omega_dt | rmse | checks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [0442136ce607](runs/oscillator/oscillator-0442136ce607) | completed | 0.012885592063752185 | euler | 9.223687118766923 | 0 | 20.0 | 1.3870553097049836 | 0.01122 | 0.1189 | 0.001669 | 2/2 |
| [97053ea14b88](runs/oscillator/oscillator-97053ea14b88) | completed | 0.0019641163321943204 | rk4 | 1.616674352282638 | 0 | 20.0 | 0.9473678965549043 | 8.651e-13 | 0.003175 | 1.834e-13 | 2/2 |
| [057e3f2dd66a](runs/oscillator/oscillator-057e3f2dd66a) | completed | 0.004954188819558289 | rk4 | 0.8244625852776364 | 0 | 20.0 | 0.11056792267363594 | 7.663e-12 | 0.004085 | 4.589e-12 | 2/2 |
| [04d4a6349562](runs/oscillator/oscillator-04d4a6349562) | completed | 0.0033250672987854705 | rk4 | 0.8032061536399775 | 0 | 20.0 | 0.5397375515521012 | 3.134e-13 | 0.002671 | 1.096e-13 | 2/2 |
| [791703425065](runs/oscillator/oscillator-791703425065) | completed | 0.002861067635189222 | rk4 | 1.5742812178309347 | 0 | 20.0 | 0.7256122669373342 | 2.373e-12 | 0.004504 | 5.814e-13 | 2/2 |
| [8c284c3882d4](runs/oscillator/oscillator-8c284c3882d4) | completed | 0.005801881913762104 | symplectic | 1.9526458502758697 | 0 | 20.0 | 0.04183649717882347 | 0.005657 | 0.01133 | 0.003244 | 2/2 |
| [d3a71b25ea04](runs/oscillator/oscillator-d3a71b25ea04) | completed | 0.0018811366140411916 | rk4 | 0.7003211418465322 | 0 | 20.0 | 0.29797721549084555 | 3.336e-14 | 0.001317 | 1.539e-14 | 2/2 |
| [d2d90700043b](runs/oscillator/oscillator-d2d90700043b) | completed | 0.001055816294361119 | symplectic | 0.7621527673671995 | 0 | 20.0 | 1.195670633238918 | 0.0002274 | 0.0008047 | 7.451e-05 | 2/2 |
| [80fc2d7c5f12](runs/oscillator/oscillator-80fc2d7c5f12) | completed | 0.03694120763470103 | euler | 2.044821255379227 | 0 | 20.0 | 1.0971380066677485 | 0.01039 | 0.07554 | 0.002761 | 2/2 |
| [1f46f99b6b59](runs/oscillator/oscillator-1f46f99b6b59) | completed | 0.0021788817648673024 | euler | 5.746692058560777 | 0 | 20.0 | 1.266508884513013 | 0.001336 | 0.01252 | 0.0002345 | 2/2 |
| [ff857ab7a67f](runs/oscillator/oscillator-ff857ab7a67f) | completed | 0.0031943332502127175 | symplectic | 3.5075541267039467 | 0 | 20.0 | 1.2061388066810794 | 0.003156 | 0.0112 | 0.0004816 | 2/2 |
| [5e5f0a9633be](runs/oscillator/oscillator-5e5f0a9633be) | completed | 0.02776797432596831 | rk4 | 0.8791711072634142 | 0 | 20.0 | 1.4942129249724063 | 9.124e-09 | 0.02441 | 1.829e-09 | 2/2 |
| [66dd9684d466](runs/oscillator/oscillator-66dd9684d466) | completed | 0.0342501479986307 | rk4 | 5.492535571800476 | 0 | 20.0 | 0.6552918685651878 | 7.593e-06 | 0.1881 | 9.987e-07 | 2/2 |
| [22cdb530644f](runs/oscillator/oscillator-22cdb530644f) | completed | 0.01400097499631986 | symplectic | 2.6316058876702564 | 0 | 20.0 | 1.2457818708666115 | 0.01023 | 0.03685 | 0.0018 | 2/2 |
| [5aa1c7f37cbb](runs/oscillator/oscillator-5aa1c7f37cbb) | completed | 0.012216311567499392 | euler | 4.693201574143115 | 0 | 20.0 | 0.8515679480642693 | 0.0116 | 0.05733 | 0.00179 | 2/2 |
| [bd18adec8a8f](runs/oscillator/oscillator-bd18adec8a8f) | completed | 0.04567944555451056 | rk4 | 6.25597781266593 | 0 | 20.0 | 0.07063719533483889 | 0.0002926 | 0.2858 | 9.52e-05 | 2/2 |
| [2e7a5fa8790a](runs/oscillator/oscillator-2e7a5fa8790a) | completed | 0.04761448609092085 | rk4 | 0.6355991112461176 | 0 | 20.0 | 1.0541126816618527 | 9.45e-09 | 0.03026 | 2.906e-09 | 2/2 |
| [a79515969a1a](runs/oscillator/oscillator-a79515969a1a) | completed | 0.0014884765278563973 | rk4 | 5.9586179318190675 | 0 | 20.0 | 0.30937867491241605 | 6.419e-11 | 0.008869 | 1.008e-11 | 2/2 |
| [9d7e625ccd11](runs/oscillator/oscillator-9d7e625ccd11) | completed | 0.0014291466263210968 | symplectic | 1.8647229988586413 | 0 | 20.0 | 0.2579184541639241 | 0.001236 | 0.002665 | 0.0003398 | 2/2 |
| [2b869c0c3d7e](runs/oscillator/oscillator-2b869c0c3d7e) | completed | 0.022334036970504845 | euler | 0.5193579949374877 | 0 | 20.0 | 0.9172834258819436 | 0.002054 | 0.0116 | 0.0009867 | 2/2 |
| [d1d5ccd8a219](runs/oscillator/oscillator-d1d5ccd8a219) | completed | 0.0037774365605841524 | symplectic | 3.877764715355997 | 0 | 20.0 | 0.12946429695435932 | 0.007169 | 0.01465 | 0.001834 | 2/2 |
| [c3bf80d06918](runs/oscillator/oscillator-c3bf80d06918) | completed | 0.04108334858474703 | symplectic | 1.2559363221498139 | 0 | 20.0 | 0.20990640405542466 | 0.02463 | 0.0516 | 0.008944 | 2/2 |
| [5f030afd29d1](runs/oscillator/oscillator-5f030afd29d1) | completed | 0.002558128640927086 | euler | 1.315303781720961 | 0 | 20.0 | 0.8001456148776995 | 0.0007248 | 0.003365 | 0.0002078 | 2/2 |
| [0ae45f86d504](runs/oscillator/oscillator-0ae45f86d504) | completed | 0.00108844325803353 | euler | 8.816440753699911 | 0 | 20.0 | 1.0315623678187527 | 0.001418 | 0.009596 | 0.0001757 | 2/2 |
| [85d06988710e](runs/oscillator/oscillator-85d06988710e) | completed | 0.0036395998173508106 | euler | 3.7075809366017505 | 0 | 20.0 | 1.1605297203557239 | 0.001658 | 0.01349 | 0.0003407 | 2/2 |
| [c35c9bbd1fe7](runs/oscillator/oscillator-c35c9bbd1fe7) | completed | 0.02515979142526455 | symplectic | 1.1814625816513138 | 0 | 20.0 | 0.4526480309176831 | 0.0125 | 0.02973 | 0.003598 | 2/2 |
| [907c90d29816](runs/oscillator/oscillator-907c90d29816) | completed | 0.006318138191817069 | euler | 7.622271840820334 | 0 | 20.0 | 1.3658289986509138 | 0.004585 | 0.04816 | 0.0007398 | 2/2 |
| [24119609f45a](runs/oscillator/oscillator-24119609f45a) | completed | 0.011371309939945087 | symplectic | 4.1504895252131755 | 0 | 20.0 | 0.33451060674716326 | 0.0212 | 0.0472 | 0.003556 | 2/2 |
| [158a279ad88f](runs/oscillator/oscillator-158a279ad88f) | completed | 0.021348309540644487 | rk4 | 1.1000024296478488 | 0 | 20.0 | 0.9681323484519061 | 2.775e-09 | 0.02348 | 6.996e-10 | 2/2 |
| [65ee0f1ccc6c](runs/oscillator/oscillator-65ee0f1ccc6c) | completed | 0.03159900215656863 | symplectic | 6.576081418270312 | 0 | 20.0 | 1.29134665027671 | 0.05807 | 0.2078 | 0.006358 | 2/2 |

![oscillator](runs/oscillator/oscillator-65ee0f1ccc6c/figures/trajectory.png)

![oscillator](runs/oscillator/oscillator-158a279ad88f/figures/trajectory.png)

## oscillator_discovery

Uses the oscillator trajectories already in the lab database: for each one, SINDy over {x, x', x², x x', x'²} finds x'' = -ω² x - 2ζω x' and recovers ζ and ω from the data alone.

| run | status | max_runs | seed | threshold | n_trajectories | omega_rel_error_median | zeta_abs_error_median | checks |
|---|---|---|---|---|---|---|---|---|
| [d0a08fc3b799](runs/oscillator_discovery/oscillator_discovery-d0a08fc3b799) | completed | 200 | 0 | 0.02 | 20 | 4.667e-06 | 3.776e-06 | 2/2 |

![oscillator_discovery](runs/oscillator_discovery/oscillator_discovery-d0a08fc3b799/figures/oscillator_rediscovery.png)

## particle_suspension

Solids suspension in a stirred tank as the impeller speeds up: soft-sphere DEM particles with drag, buoyancy and turbulent dispersion in an analytic, divergence-free stirred-tank flow; process video (particles coloured by speed, turning impeller, live charts), suspension curves and the just-suspended speed against Zwietering's correlation.

| run | status | d_mm | dt | frame_every | n | ramp_s | render | rho_p | rpm_max | samples | seed | size | t_end | rpm_90pct_suspended | solids_mass_ratio | suspended_fraction_end | terminal_velocity_m_s | top_third_fraction_end | zwietering_njs_S4_rpm | checks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [b7feb1ba16dd](runs/particle_suspension/particle_suspension-b7feb1ba16dd) | completed | 3.0 | 0.00025 | 0.25 | 12000 | 25.0 | True | 1200.0 | 400.0 | 16 | 0 | [800, 640] | 30.0 | 288 | 0.07695 | 0.9568 | 0.1097 | 0.7736 | 237.1 | 6/6 |

![particle_suspension](runs/particle_suspension/particle_suspension-b7feb1ba16dd/figures/stirred_tank_early.png)

## pendulum_video

Law -> video -> law -> video. A large-amplitude damped pendulum is filmed (rendered with camera noise and motion blur); its angle is measured from the frames; weak-form SINDy discovers θ'' = -(g/L) sin θ - c θ' from a library that also offers θ, θ³, cos θ, θ'|θ'|, ...; the discovered law is re-simulated and re-rendered, and the predicted video is compared with the observed one, including after the observation window.

| run | status | L | damping | fps | g | n_test | noise | predict_seconds | seconds | seed | size | test_width | theta0 | angle_measurement_rmse_rad | damping_discovered | forecast_rmse_rad | g_discovered | g_rel_error | n_terms | checks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [7a4bc9625e0e](runs/pendulum_video/pendulum_video-7a4bc9625e0e) | completed | 0.8 | 0.15 | 60 | 9.81 | 120 | 0.04 | 6.0 | 10.0 | 0 | 160 | 2.0 | 2.4 | 0.01516 | 0.152 | 0.009488 | 9.787 | 0.002353 | 2 | 3/3 |

![pendulum_video](runs/pendulum_video/pendulum_video-7a4bc9625e0e/figures/pendulum_discovery.png)

## pipe_flow

Internal flow in OpenFOAM on a structured O-grid swept along the pipe: laminar pipe against Hagen-Poiseuille, a turbulent 90 degree bend against Colebrook and Ito, a Kenics static mixer against an empty pipe (mixing of a passive scalar, pressure-drop ratio).

| run | status | D | case | elements | iterations | keep_case | procs | render | samples | K_bend_excess | K_bend_total | K_ito | Re | U_mean_m_s | cells | checks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [0c34ca2835fe](runs/pipe_flow/pipe_flow-0c34ca2835fe) | completed | 0.05 | bend_90 | 6 | 1500 | False | 2 | True | 64 | 0.1276 | 0.1956 | 0.2304 | 5e+04 | 1 | 6.24e+04 | 4/4 |
| [64f176a5fafc](runs/pipe_flow/pipe_flow-64f176a5fafc) | completed | 0.05 | laminar_pipe | 6 | 1500 | False | 2 | True | 64 |  |  |  | 500 | 0.01 | 3.84e+04 | 3/3 |
| [07a353a549b9](runs/pipe_flow/pipe_flow-07a353a549b9) | completed | 0.05 | kenics_mixer | 6 | 1500 | False | 2 | True | 64 |  |  |  | 50 | 0.001 | 3.448e+05 | 4/4 |

![pipe_flow](runs/pipe_flow/pipe_flow-07a353a549b9/figures/mixing_and_pressure.png)

![pipe_flow](runs/pipe_flow/pipe_flow-64f176a5fafc/figures/profile_and_pressure.png)

## repo_results

Results already produced by repository scripts (PINNs, inverse problems, LBM, MeshGraphNet, black-hole forecasts, electric machines, electromagnetics, airfoil and airliner CFD, the lunar rover, forecasting studies, design apps), re-validated against their references and stored as datasets.

| run | status | source | CD | CD_friction | CD_pressure | CL | CL_max | CL_rms_vs_polhamus | checks |
|---|---|---|---|---|---|---|---|---|---|
| [96e87a293dea](runs/repo_results/repo_results-96e87a293dea) | completed | burgers_pinn |  |  |  |  |  |  | 1/1 |
| [fa808c1e7a31](runs/repo_results/repo_results-fa808c1e7a31) | completed | lbm_strouhal |  |  |  |  |  |  | 3/3 |
| [cbac6f31f681](runs/repo_results/repo_results-cbac6f31f681) | completed | meshgraphnet |  |  |  |  |  |  | 2/2 |
| [ae7814d939d6](runs/repo_results/repo_results-ae7814d939d6) | completed | concorde_aoa |  |  |  |  | 0.5881 | 0.2409 | 2/2 |
| [8f10f262c734](runs/repo_results/repo_results-8f10f262c734) | completed | fin_inverse_2d |  |  |  |  |  |  | 3/3 |
| [e9574ff86021](runs/repo_results/repo_results-e9574ff86021) | completed | fin_inverse_3d |  |  |  |  |  |  | 3/3 |
| [33073ba61b6c](runs/repo_results/repo_results-33073ba61b6c) | completed | heatsink_surrogate |  |  |  |  |  |  | 3/3 |
| [269f1207b6f2](runs/repo_results/repo_results-269f1207b6f2) | completed | bh_forecast |  |  |  |  |  |  | 2/2 |
| [d8795900891a](runs/repo_results/repo_results-d8795900891a) | completed | airfoil_surrogate |  |  |  |  |  |  | 2/2 |
| [76a9a3bf2c2f](runs/repo_results/repo_results-76a9a3bf2c2f) | completed | ahmed_body | 0.2977 | 0.05474 | 0.2429 | 0.3368 |  |  | 2/2 |
| [4104d166d1c0](runs/repo_results/repo_results-4104d166d1c0) | completed | prius_pmsm |  |  |  |  |  |  | 12/12 |
| [32624513331b](runs/repo_results/repo_results-32624513331b) | completed | prius_end_effects |  |  |  |  |  |  | 4/4 |
| [b1eaa75f0af8](runs/repo_results/repo_results-b1eaa75f0af8) | completed | prius_thermal_demag |  |  |  |  |  |  | 5/5 |
| [4de224ca1ac2](runs/repo_results/repo_results-4de224ca1ac2) | completed | prius_drive |  |  |  |  |  |  | 6/6 |
| [c9fd2d4178ee](runs/repo_results/repo_results-c9fd2d4178ee) | completed | em_verification |  |  |  |  |  |  | 33/33 |
| [baf5389acdaf](runs/repo_results/repo_results-baf5389acdaf) | completed | motor_spm |  |  |  |  |  |  | 1/1 |
| [b2b7314f5a6d](runs/repo_results/repo_results-b2b7314f5a6d) | completed | airfoil_handoff |  |  |  |  |  |  | 8/8 |
| [3fe077b4c485](runs/repo_results/repo_results-3fe077b4c485) | completed | airliner_cfd | 0.04615 | 0.01479 | 0.03137 | 0.5893 |  |  | 3/3 |
| [dd2cd168e8d1](runs/repo_results/repo_results-dd2cd168e8d1) | completed | lunar_rover |  |  |  |  |  |  | 3/3 |
| [caf5921ef263](runs/repo_results/repo_results-caf5921ef263) | completed | near_random_forecasting |  |  |  |  |  |  | 5/5 |
| [156fa6f6214f](runs/repo_results/repo_results-156fa6f6214f) | completed | design_from_intent |  |  |  |  |  |  | 8/8 |
| [ea46a4b7c9e3](runs/repo_results/repo_results-ea46a4b7c9e3) | completed | topology_optimization |  |  |  |  |  |  | 5/5 |
| [6b9418caaf82](runs/repo_results/repo_results-6b9418caaf82) | completed | free_surface_trough |  |  |  |  |  |  | 5/5 |
| [835ff767e739](runs/repo_results/repo_results-835ff767e739) | failed_validation | benchmark_arena |  |  |  |  |  |  | 5/6 |
| [4cb729015e4a](runs/repo_results/repo_results-4cb729015e4a) | completed | weather_era5 |  |  |  |  |  |  | 0/0 |

![repo_results](runs/repo_results/repo_results-4cb729015e4a/figures/weather_pnw_heat_dome_2021_preview_day3.png)

![repo_results](runs/repo_results/repo_results-6b9418caaf82/figures/trough_checks.png)

## rom_study

Reduced-order models on the lab's own solvers: POD / DMD / Operator Inference forecasting a cylinder wake, and a parametric POD-RBF / POD-GPR model of a 3-D FEM cantilever, each against its baseline (persistence, nearest design).

| run | status | Re | case | n_test | n_train | save_every | seed | steps | St_dmd | St_probe | displacement_coverage_90 | displacement_max_error | displacement_median_error | displacement_nearest_design_median_error | checks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [61475bf029de](runs/rom_study/rom_study-61475bf029de) | completed | 100.0 | beam_parametric | 20 | 80 | 50 | 0 | 30000 |  |  | 0.8208 | 0.001829 | 0.0005344 | 0.05243 | 6/6 |
| [9580a00926ca](runs/rom_study/rom_study-9580a00926ca) | completed | 100.0 | cylinder_wake | 20 | 80 | 50 | 0 | 30000 | 0.2444 | 0.2444 |  |  |  |  | 6/6 |

![rom_study](runs/rom_study/rom_study-9580a00926ca/figures/forecast_error.png)

![rom_study](runs/rom_study/rom_study-61475bf029de/figures/rom_errors.png)

## solid_fem

3-D linear elasticity with hexahedra with incompatible modes (C3D8I): a cantilever, a simply supported beam and a bar in torsion, checked against beam and Saint-Venant theory and against CalculiX on the same mesh, drawn as a post-processor draws them (deformed mesh, von Mises bands).

| run | status | calculix | case | refine | render | samples | bending_stress_MPa | calculix_max_rel_difference | deflection_mm | dofs | elements | max_displacement_mm | checks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [4b486451e11b](runs/solid_fem/solid_fem-4b486451e11b) | completed | True | cantilever | 1 | True | 64 | 22.5 | 2.617e-06 | 1.908 | 8235 | 1920 | 1.913 | 4/4 |
| [eda68173545c](runs/solid_fem/solid_fem-eda68173545c) | completed | True | simply_supported | 1 | True | 64 | 150.3 | 1.643e-06 | 3.082 | 1.482e+04 | 3840 | 3.135 | 4/4 |
| [fad914757a27](runs/solid_fem/solid_fem-fad914757a27) | completed | True | torsion | 1 | True | 64 |  | 1.121e-06 |  | 2.079e+04 | 5760 | 0.6283 | 4/4 |

![solid_fem](runs/solid_fem/solid_fem-fad914757a27/figures/displacement_fea.png)

![solid_fem](runs/solid_fem/solid_fem-eda68173545c/figures/displacement_fea.png)

## vehicle_cfd

A parametric 3-D road car or launch vehicle in OpenFOAM (snappyHexMesh + simpleFoam, half model): drag, lift, skin pressure, streamlines and wake, checked against reference ranges (car) or Barrowman's stability equations (rocket); Blender Cycles renders and a 3-D viewer.

| run | status | alpha | iterations | keep_case | procs | resolution | samples | slant_deg | speed | style | surface_level | vehicle | CD | CD_friction | CD_pressure | CL | Re | cells | checks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [19b6c3f7bb85](runs/vehicle_cfd/vehicle_cfd-19b6c3f7bb85) | completed | 4.0 | 600 | False | 2 | coarse | 96 | 22.0 | 30.0 | fastback | None | car | 0.2883 | 0.03308 | 0.2552 | 0.01798 | 9.2e+06 | 1.518e+05 | 2/2 |
