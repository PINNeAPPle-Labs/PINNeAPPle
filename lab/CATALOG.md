# PINNeAPPle Lab catalogue

97 runs, generated 2026-10-10 17:10.

| experiment | runs | completed | failed validation | failed | datasets (samples) |
|---|---|---|---|---|---|
| [accretion_flow](#accretion_flow) | 6 | 6 | 0 | 0 | frames (186) |
| [bondi_accretion](#bondi_accretion) | 6 | 6 | 0 | 0 | profiles (6) |
| [cylinder_lbm](#cylinder_lbm) | 9 | 7 | 2 | 0 | vorticity (720) |
| [heat_xtfc](#heat_xtfc) | 12 | 8 | 4 | 0 | fields (12) |
| [kepler_law](#kepler_law) | 1 | 1 | 0 | 0 | orbits (19) |
| [lorenz_discovery](#lorenz_discovery) | 1 | 1 | 0 | 0 | - |
| [oscillator](#oscillator) | 60 | 60 | 0 | 0 | trajectories (60) |
| [oscillator_discovery](#oscillator_discovery) | 1 | 1 | 0 | 0 | discovered_laws (20) |
| [pendulum_video](#pendulum_video) | 1 | 1 | 0 | 0 | pendulum (1) |

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

## cylinder_lbm

2D channel flow past a cylinder with the D2Q9 lattice-Boltzmann solver (pinneapple_simulation.numerical_solvers.lbm). Measures the wake regime (steady or vortex shedding) and the Strouhal number; saves vorticity snapshots labelled by regime and Re, a dataset for vision (classification / clustering of flow regimes, #415).

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

## pendulum_video

Law -> video -> law -> video. A large-amplitude damped pendulum is filmed (rendered with camera noise and motion blur); its angle is measured from the frames; weak-form SINDy discovers θ'' = -(g/L) sin θ - c θ' from a library that also offers θ, θ³, cos θ, θ'|θ'|, ...; the discovered law is re-simulated and re-rendered, and the predicted video is compared with the observed one, including after the observation window.

| run | status | L | damping | fps | g | n_test | noise | predict_seconds | seconds | seed | size | test_width | theta0 | angle_measurement_rmse_rad | damping_discovered | forecast_rmse_rad | g_discovered | g_rel_error | n_terms | checks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [7a4bc9625e0e](runs/pendulum_video/pendulum_video-7a4bc9625e0e) | completed | 0.8 | 0.15 | 60 | 9.81 | 120 | 0.04 | 6.0 | 10.0 | 0 | 160 | 2.0 | 2.4 | 0.01516 | 0.152 | 0.009488 | 9.787 | 0.002353 | 2 | 3/3 |

![pendulum_video](runs/pendulum_video/pendulum_video-7a4bc9625e0e/figures/pendulum_discovery.png)
