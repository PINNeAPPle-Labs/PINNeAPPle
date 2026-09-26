# Catálogo de métodos do PINNeAPPle

> Gerado por `scripts/build_method_status.py` a partir de `pinneapple_catalog/methods.py` e dos testes. Não editar à mão: rode o script de novo.

**Regra (D5 de `PEDIDOS_2026-09-24.md`):** ✅ = existe teste que compara o método com uma referência independente (solução fechada, solução manufaturada, valor publicado ou tabelado), na mesma função de teste. 🟡 = há teste, mas nenhum compara com referência. ⚪ = nenhum teste encontrado. Só ✅ entra na lista oficial (`validated_methods()`). O resto fica **pendente: paper + benchmark**; nenhum código foi apagado.

Referências: "no código" = já citadas no docstring do módulo; "adicionadas" = fonte canônica incluída no catálogo porque o código não citava nada. Todo arXiv foi conferido contra o título em 2026-09-24.

| Categoria | ✅ | 🟡 | ⚪ |
|---|---|---|---|
| A. Solvers (S) | 7 | 13 | 6 |
| B. Métodos de treino (T) | 1 | 12 | 9 |
| C. Equações físicas (E) | 36 | 21 | 8 |
| D. Problemas físicos (P) | 27 | 8 | 31 |

## A. Solvers (S)

### S1 — Finite differences 1D/2D/3D
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/pinneapple_solvers/test_fdm_robin.py::test_robin_heat_transient_relaxes_to_same_steady_state`; `PINNeAPPle/tests/pinneapple_solvers/test_fdm_robin.py::test_robin_poisson_matches_analytical_linear_profile`
- **Código:** `pinneapple_simulation/numerical_solvers/fdm.py`, `pinneapple_simulation/numerical_solvers/fdm3d.py`
- **Referências adicionadas:** LeVeque, Finite Difference Methods for Ordinary and Partial Differential Equations, SIAM, 2007

### S2 — Finite elements (linear) + nonlinear beam FEM
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/pinneapple_physics/test_turbulence_selector.py::test_unknown_solver_family_raises`
- **Código:** `pinneapple_simulation/numerical_solvers/fem.py`, `pinneapple_simulation/numerical_solvers/nonlinear_beam_fem.py`
- **Referências adicionadas:** Hughes, The Finite Element Method: Linear Static and Dynamic FEA, Dover, 2000 · Zienkiewicz, Taylor & Zhu, The Finite Element Method: Its Basis and Fundamentals, 7th ed., 2013

### S3 — Finite volumes
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_simulation/numerical_solvers/fvm.py`
- **Referências adicionadas:** Versteeg & Malalasekera, An Introduction to Computational Fluid Dynamics: The Finite Volume Method, 2nd ed., Pearson, 2007 · LeVeque, Finite Volume Methods for Hyperbolic Problems, Cambridge University Press, 2002

### S4 — Spectral (Fourier pseudo-spectral)
- **Estado:** 🟡 testado, sem referência
- **Revisão manual:** the hit compares gradient backends with each other (consistency), not the spectral solver with an independent reference
- **Evidência:** `PINNeAPPle/tests/test_gradient_backend_consistency.py::test_gradient_backends_agree_laplace_2d`; `PINNeAPPle/tests/test_gradient_backend_consistency.py::test_gradient_backends_agree_reaction_diffusion_2d`
- **Código:** `pinneapple_simulation/numerical_solvers/spectral.py`
- **Referências adicionadas:** Trefethen, Spectral Methods in MATLAB, SIAM, 2000 · Canuto, Hussaini, Quarteroni & Zang, Spectral Methods: Fundamentals in Single Domains, Springer, 2006

### S5 — Lattice Boltzmann D2Q9 / D3Q19 (+ Smagorinsky LES)
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle-CFD/tests/test_e04_metodos.py::test_nunca_recomenda_lbm`; `PINNeAPPle-CFD/tests/test_e04_recomendacao.py::test_ca_e4_01_laminar_e_turbulento`; `PINNeAPPle/tests/pinneapple_physics/test_turbulence_selector.py::test_lbm_laminar_is_zero` (+17)
- **Código:** `pinneapple_simulation/numerical_solvers/lbm.py`
- **Referências no código:** Zou & He, Phys. Fluids 9 (1997) 1591 -- pressure/velocity boundary conditions · Hou, Sterling, Chen & Doolen (1994) -- Smagorinsky LES-LBM
- **Referências adicionadas:** Krueger et al., The Lattice Boltzmann Method: Principles and Practice, Springer, 2017

### S6 — SPH / ISPH / DFSPH
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_simulation/numerical_solvers/sph.py`, `pinneapple_simulation/numerical_solvers/isph.py`, `pinneapple_simulation/numerical_solvers/dfsph.py`
- **Referências no código:** Morris, Fox & Zhu, J. Comput. Phys. 136 (1997) -- SPH viscosity · Cummins & Rudman, J. Comput. Phys. 152 (1999) 584-607 -- ISPH · Shao & Lo, Adv. Water Resour. 26 (2003) 787-800 -- ISPH · Bender & Koschier, SCA 2015 / IEEE TVCG 2017 -- DFSPH
- **Referências adicionadas:** Monaghan, Smoothed particle hydrodynamics, Rep. Prog. Phys. 68 (2005) 1703

### S7 — Meshfree RBF collocation (Kansa)
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_simulation/numerical_solvers/meshfree.py`
- **Referências adicionadas:** Kansa, Multiquadrics, Comput. Math. Appl. 19(8-9) (1990) 147-161

### S8 — X-TFC (IVP, PDE, subdomains)
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_xtfc_pde.py::test_heat_1d_matches_closed_form_solution`; `PINNeAPPle/tests/test_xtfc_subdomain.py::test_extra_param_reuses_features_and_shifts_solution`; `PINNeAPPle/tests/test_xtfc_subdomain.py::test_stiff_chain_matches_closed_form_and_scipy` (+1)
- **Código:** `pinneapple_simulation/numerical_solvers/xtfc_ivp.py`, `pinneapple_simulation/numerical_solvers/xtfc_pde.py`, `pinneapple_simulation/numerical_solvers/xtfc_subdomain.py`
- **Referências no código:** Schiassi et al., Extreme Theory of Functional Connections, arXiv:2005.10632 · Schiassi et al., PI X-TFC for parameter discovery, arXiv:2008.05554 · Leake & Mortari, Deep Theory of Functional Connections, arXiv:1812.08625

### S9 — Discrete-time PINN / implicit Runge-Kutta Gauss-Legendre
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_discrete_time_pinn.py::test_discrete_time_rk_residual_shapes_and_zero_for_matching_data`; `PINNeAPPle/tests/test_discrete_time_pinn.py::test_registered_in_solver_registry_and_forward_runs`
- **Código:** `pinneapple_simulation/numerical_solvers/discrete_time_pinn.py`, `pinneapple_simulation/numerical_solvers/irk_gauss_legendre.py`
- **Referências no código:** Raissi et al., Physics Informed Deep Learning (Part II), arXiv:1711.10566
- **Referências adicionadas:** Raissi, Perdikaris & Karniadakis, Physics Informed Deep Learning (Part I), arXiv:1711.10561 · Hairer & Wanner, Solving Ordinary Differential Equations II: Stiff and DAE Problems, 2nd ed., Springer, 1996

### S10 — Stiff kinetics ODE
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_xtfc_subdomain.py::test_stiff_chain_matches_closed_form_and_scipy`
- **Código:** `pinneapple_simulation/numerical_solvers/stiff_kinetics_ode.py`
- **Referências adicionadas:** Hairer & Wanner, Solving Ordinary Differential Equations II: Stiff and DAE Problems, 2nd ed., Springer, 1996

### S11 — TVD advection
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_simulation/numerical_solvers/tvd_advection.py`
- **Referências adicionadas:** Harten, High resolution schemes for hyperbolic conservation laws, J. Comput. Phys. 49 (1983) 357-393 · Sweby, High resolution schemes using flux limiters, SIAM J. Numer. Anal. 21(5) (1984) 995-1011

### S12 — Immersed boundary (FDM)
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_simulation/numerical_solvers/immersed_boundary_fdm.py`
- **Referências adicionadas:** Peskin, The immersed boundary method, Acta Numerica 11 (2002) 479-517 · Mittal & Iaccarino, Immersed boundary methods, Annu. Rev. Fluid Mech. 37 (2005) 239-261

### S13 — Beam BVP FDM / 3D elasticity FDM
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_simulation/numerical_solvers/beam_bvp_fdm.py`, `pinneapple_simulation/numerical_solvers/elasticity3d_fdm.py`
- **Referências adicionadas:** Timoshenko & Goodier, Theory of Elasticity, 3rd ed., McGraw-Hill, 1970 · LeVeque, Finite Difference Methods for Ordinary and Partial Differential Equations, SIAM, 2007

### S14 — Eddy current FDM
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_inspection.py::test_eddy_current_synthetic_matches_solver_convention`; `PINNeAPPle/tests/test_inspection.py::test_train_eddy_current_smoke`
- **Código:** `pinneapple_simulation/numerical_solvers/eddy_current_fdm.py`
- **Referências adicionadas:** Jackson, Classical Electrodynamics, 3rd ed., Wiley, 1999

### S15 — Bekker-Wong terramechanics
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_bekker_wong_terramechanics_satisfying_solution_gives_zero_residual`
- **Código:** `pinneapple_simulation/numerical_solvers/bekker_wong.py`, `pinneapple_simulation/particle_dynamics/terramechanics.py`
- **Referências no código:** Wong 1978
- **Referências adicionadas:** Bekker, Introduction to Terrain-Vehicle Systems, University of Michigan Press, 1969 · Wong, Theory of Ground Vehicles, Wiley, 1978 (4th ed. 2008)

### S16 — N-body (REBOUND)
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_nbody_rebound.py::test_cr3bp_l4_stays_fixed_in_rotating_frame`
- **Código:** `pinneapple_simulation/numerical_solvers/nbody_rebound.py`
- **Referências no código:** Rein & Liu, REBOUND, Astron. Astrophys. 537 (2012) A128

### S17 — JAX-CFD
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_jax_cfd_solver.py::test_jax_cfd_registered_in_registry`; `PINNeAPPle/tests/test_jax_cfd_solver.py::test_jax_cfd_solver_raises_clear_error_when_missing`; `PINNeAPPle/tests/test_jax_cfd_solver.py::test_registry_build_returns_working_solver`
- **Código:** `pinneapple_simulation/numerical_solvers/jax_cfd_solver.py`
- **Referências no código:** Dresdner et al., Learning to correct spectral methods, arXiv:2207.00556
- **Referências adicionadas:** Kochkov et al., Machine learning accelerated CFD, PNAS 118(21) (2021), arXiv:2102.01010

### S18 — Particle dynamics: rigid body, MPM, molecular dynamics
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_mpm_simulator`; `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_mpm_state`; `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_particle_system_is_abstract` (+5)
- **Código:** `pinneapple_simulation/particle_dynamics/`
- **Referências no código:** Stomakhin et al., A material point method for snow simulation, ACM TOG 32(4) (2013) · Hu et al., A moving least squares material point method, ACM TOG 37(4) (2018) · Swope, Andersen, Berens & Wilson, J. Chem. Phys. 76 (1982) 637 -- velocity Verlet
- **Referências adicionadas:** Featherstone, Rigid Body Dynamics Algorithms, Springer, 2008

### S19 — Signal decomposition: CEEMDAN, EEMD, VMD, HHT, SSA, SST, STL, FFT, wavelet
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_solver_registry_build`
- **Código:** `pinneapple_simulation/numerical_solvers/ceemdan.py`, `pinneapple_simulation/numerical_solvers/eemd.py`, `pinneapple_simulation/numerical_solvers/vmd.py`, `pinneapple_simulation/numerical_solvers/hilbert_huang.py`, `pinneapple_simulation/numerical_solvers/ssa.py`, `pinneapple_simulation/numerical_solvers/sst.py`, `pinneapple_simulation/numerical_solvers/stl.py`, `pinneapple_simulation/numerical_solvers/fft.py`, `pinneapple_simulation/numerical_solvers/wavelet.py`
- **Referências no código:** Torres et al., CEEMDAN, ICASSP 2011 · Wu & Huang, EEMD, Adv. Adapt. Data Anal. 1(1) (2009) · Dragomiretskiy & Zosso, Variational Mode Decomposition, IEEE TSP 62(3) (2014)
- **Referências adicionadas:** Huang et al., The empirical mode decomposition and the Hilbert spectrum, Proc. R. Soc. A 454 (1998) 903-995 · Golyandina, Nekrutkin & Zhigljavsky, Analysis of Time Series Structure: SSA and Related Techniques, 2001 · Daubechies, Lu & Wu, Synchrosqueezed wavelet transforms, Appl. Comput. Harmon. Anal. 30 (2011) 243-261 · Cleveland et al., STL: a seasonal-trend decomposition procedure based on loess, J. Off. Stat. 6(1) (1990) · Cooley & Tukey, Math. Comput. 19 (1965) 297-301 -- FFT · Mallat, A Wavelet Tour of Signal Processing, 3rd ed., Academic Press, 2009

### S20 — Co-simulation (graph + engine, algebraic loops by Tarjan SCC)
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_cosim_analytical_node`; `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_cosim_blackbox_node`; `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_cosim_graph_and_engine` (+8)
- **Código:** `pinneapple_systems/cosimulation/`
- **Referências adicionadas:** Gomes et al., Co-simulation: a survey, ACM Comput. Surv. 51(3) (2018) · Tarjan, Depth-first search and linear graph algorithms, SIAM J. Comput. 1(2) (1972) 146-160

### S21 — Digital twin state estimation: KF/EKF/UKF/EnKF, prognostics, signal reconstruction
- **Estado:** 🟡 testado, sem referência
- **Revisão manual:** the only 'reference' hit is an OPC-UA live-stream test; no state-estimation result is compared with an independent reference
- **Evidência:** `PINNeAPPle/tests/test_digital_twin_streams_scada_live.py::test_opcua_stream_reads_real_values_from_a_real_server`; `PINNeAPPle/tests/test_telemetry_conditioning.py::test_first_order_filter_matches_closed_form_step_response_with_irregular_sampling`
- **Código:** `pinneapple_systems/digital_twin/`, `pinneapple_neural/architectures/classical_ts/`
- **Referências adicionadas:** Kalman, A new approach to linear filtering and prediction problems, J. Basic Eng. 82 (1960) 35-45 · Julier & Uhlmann, Unscented filtering and nonlinear estimation, Proc. IEEE 92(3) (2004) 401-422 · Evensen, Sequential data assimilation with a nonlinear QG model, J. Geophys. Res. 99(C5) (1994) 10143 · Simon, Optimal State Estimation, Wiley, 2006

### S22 — 1D process components (pipe network, transient pipe, heat exchanger, valve, reaction, polytropic path, compressor map)
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_advection_dispersion_reaction_solver`; `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_process_component_exceptions`; `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_transient_pipe`
- **Código:** `pinneapple_systems/process_components/`
- **Referências no código:** PennWell 2003 (compressor similarity map)
- **Referências adicionadas:** Incropera, DeWitt, Bergman & Lavine, Fundamentals of Heat and Mass Transfer, 7th ed., Wiley, 2011 · White, Fluid Mechanics, 7th ed., McGraw-Hill, 2011 · IEC 60534-2-1 / ANSI/ISA-75.01.01 -- control valve sizing · Schultz, The polytropic analysis of centrifugal compressors, J. Eng. Power 84(1) (1962) 69-82 · Bell, Wronski, Quoilin & Lemort, CoolProp, Ind. Eng. Chem. Res. 53(6) (2014) 2498-2508

### S23 — Explicit equation-system solver (evaluation order)
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_process_component_exceptions`
- **Código:** `pinneapple_systems/process_components/explicit_equation_system.py`
- **Referências adicionadas:** Tarjan, Depth-first search and linear graph algorithms, SIAM J. Comput. 1(2) (1972) 146-160

### S24 — External solvers: OpenFOAM, FEniCS, ANSYS/CFD formats, MATLAB, Modelica/FMI, MuJoCo, Genesis, TurboDesigner
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle-CFD/tests/test_e02_modelo_matematico.py::test_caso_exportado_le_no_foamdictionary_e_roda_simplefoam`; `PINNeAPPle-CFD/tests/test_e02_modelo_matematico.py::test_exportacao_todos_os_patches_em_todos_os_campos_e_reprodutivel`; `PINNeAPPle-CFD/tests/test_e03_geometria_cliente.py::test_api_upload_confirmar_previa_e_malha_cliente` (+63)
- **Código:** `pinneapple_simulation/external_solvers/`
- **Referências adicionadas:** Weller, Tabor, Jasak & Fureby, A tensorial approach to CFD, Comput. Phys. 12(6) (1998) 620-631 (OpenFOAM) · Baratta et al., DOLFINx: the next generation FEniCS problem solving environment, 2023 · Blochwitz et al., The Functional Mockup Interface, Modelica Conference 2011 · Todorov, Erez & Tassa, MuJoCo, IROS 2012

### S25 — Causal telemetry conditioning (Hampel one-step, rate limit, first-order filter, gaps)
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_telemetry_conditioning.py::test_first_order_filter_matches_closed_form_step_response_with_irregular_sampling`
- **Código:** `pinneapple_systems/digital_twin/conditioning.py`
- **Referências no código:** Hampel, The influence curve and its role in robust estimation, J. Am. Stat. Assoc. 69 (1974) 383-393 · Pearson, Outliers in process modeling and identification, IEEE Trans. Control Syst. Technol. 10(1) (2002) 55-63

### S26 — 2D shallow-water finite volumes (MUSCL + HLL, wet/dry, walls, gates, virtual sensors, health checks)
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_shallow_water_fv.py::test_ritter_dry_bed_dam_break`
- **Código:** `pinneapple_simulation/numerical_solvers/shallow_water_fv.py`
- **Referências adicionadas:** Toro, Shock-Capturing Methods for Free-Surface Shallow Flows, Wiley, 2001 · Ritter, Die Fortpflanzung der Wasserwellen, Z. Vereines Deutscher Ingenieure 36 (1892) 947-954 · Stoker, Water Waves, Interscience, 1957 (wet-bed dam break)


## B. Métodos de treino (T)

### T1 — Base trainer: AMP, gradient accumulation, auto batch size, CUDA graphs, profiler
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/pinneapple_train/test_trainer_minimal.py::test_trainer_runs_one_epoch`; `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_backtest_runner`; `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_cosim_trainer_fits` (+4)
- **Código:** `pinneapple_neural/trainer/trainer.py`, `pinneapple_neural/trainer/hpc.py`
- **Referências adicionadas:** Micikevicius et al., Mixed Precision Training, ICLR 2018, arXiv:1710.03740 · Kingma & Ba, Adam, ICLR 2015, arXiv:1412.6980

### T2 — Distributed: DDP, FSDP, SLURM, torchrun, PowerSGD
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_neural/trainer/distributed.py`, `pinneapple_neural/trainer/hpc.py`
- **Referências no código:** Vogels, Karimireddy & Jaggi, PowerSGD, NeurIPS 2019, arXiv:1905.13727
- **Referências adicionadas:** Li et al., PyTorch Distributed, VLDB 2020, arXiv:2006.15704 · Zhao et al., PyTorch FSDP, VLDB 2023, arXiv:2304.11277

### T3 — Causal training
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_neural/trainer/causal.py`
- **Referências no código:** Wang, Sankaran & Perdikaris, Respecting causality is all you need for training PINNs, arXiv:2203.07404

### T4 — Time marching
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_neural/trainer/time_marching.py`
- **Referências no código:** Wight & Zhao, Solving Allen-Cahn and Cahn-Hilliard equations using adaptive PINNs, arXiv:2007.04542

### T5 — Two phases Adam -> L-BFGS + L-BFGS finetune
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/pinneapple_neural/test_lbfgs_finetune.py::test_multi_round_converges_a_toy_quadratic`; `PINNeAPPle/tests/pinneapple_neural/test_lbfgs_finetune.py::test_on_round_end_callback_invoked_correctly`; `PINNeAPPle/tests/pinneapple_neural/test_lbfgs_finetune.py::test_post_round_eval_overrides_reported_loss` (+3)
- **Código:** `pinneapple_neural/trainer/two_phase.py`, `pinneapple_neural/trainer/lbfgs_finetune.py`
- **Referências adicionadas:** Raissi, Perdikaris & Karniadakis, Physics-informed neural networks, J. Comput. Phys. 378 (2019) 686-707 · Liu & Nocedal, On the limited memory BFGS method, Math. Program. 45 (1989) 503-528

### T6 — Multi-restart
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_neural/trainer/multistart.py`
- **Referências:** nenhuma fonte encontrada — pendente de paper

### T7 — Adaptive residual-based collocation
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_neural/trainer/collocation.py`
- **Referências no código:** Lu et al., DeepXDE, SIAM Rev. 63(1) (2021), arXiv:1907.04502 · Wu et al., non-adaptive and residual-based adaptive sampling for PINNs, CMAME 2023, arXiv:2207.10289

### T8 — Loss balancing: ReLoBRaLo, SoftAdapt, PCGrad, Augmented Lagrangian, Inverse Dirichlet, AutoBalancer
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle-CFD/tests/test_e08_funcao_perda.py::test_balanceador_inexistente_e_existente`; `PINNeAPPle-CFD/tests/test_e08_funcao_perda.py::test_perda_combinada_nomeada_com_balanceador_e_nan`
- **Código:** `pinneapple_neural/trainer/loss_balancer.py`
- **Referências no código:** Bischof & Kraus, Multi-objective loss balancing for PINNs, arXiv:2110.09813 · Heydari et al., SoftAdapt, arXiv:1912.12355 · van der Meer, Oosterlee & Borovykh, Optimally weighted loss functions for PDEs, 2022
- **Referências adicionadas:** Yu et al., Gradient Surgery for Multi-Task Learning (PCGrad), NeurIPS 2020, arXiv:2001.06782 · Lu et al., PINNs with hard constraints for inverse design (augmented Lagrangian), arXiv:2102.04626 · Maddu et al., Inverse-Dirichlet weighting, arXiv:2107.00940

### T9 — Weight schedulers: SA-PINN, GradNorm, LossRatio, NTK
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_neural/trainer/weight_scheduler.py`
- **Referências no código:** McClenny & Braga-Neto, Self-Adaptive PINNs, arXiv:2009.04544 · Chen et al., GradNorm, ICML 2018, arXiv:1711.02257 · Wang, Yu & Perdikaris, When and why PINNs fail to train: an NTK perspective, arXiv:2007.14527

### T10 — Domain decomposition: XPINN, DoMINO/D3M
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/pinneapple_analysis/test_architecture_recommendation.py::test_scarce_data_no_solver_recommends_pinn`
- **Código:** `pinneapple_neural/architectures/pinns/xpinn.py`, `pinneapple_physics/pinn_solver/domino.py`
- **Referências no código:** Li et al., D3M: a deep domain decomposition method for PDEs, arXiv:1909.12236
- **Referências adicionadas:** Jagtap & Karniadakis, Extended PINNs (XPINNs), Commun. Comput. Phys. 28(5) (2020) 2002-2041

### T11 — Inverse PINN (learnable physical parameters)
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/pinneapple_analysis/test_architecture_recommendation.py::test_inverse_problem_adds_inverse_pinn_regardless_of_other_axes`; `PINNeAPPle/tests/test_decision_adapter_tree_arena.py::test_arena_catalog_family_mapping`
- **Código:** `pinneapple_neural/trainer/inverse_pinn.py`
- **Referências adicionadas:** Raissi, Perdikaris & Karniadakis, Physics-informed neural networks, J. Comput. Phys. 378 (2019) 686-707

### T12 — Gray-box + symbolic regression (PySR)
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_neural/trainer/graybox.py`
- **Referências adicionadas:** Cranmer, Interpretable ML for science with PySR and SymbolicRegression.jl, arXiv:2305.01582

### T13 — Uncertainty: heteroscedastic, B-PINN, MC Dropout, deep ensembles
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_deep_ensemble`; `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_edge_runtime`; `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_mc_dropout_standalone` (+10)
- **Código:** `pinneapple_neural/trainer/heteroscedastic.py`, `pinneapple_neural/architectures/pinns/bpinn.py`, `pinneapple_systems/component_modeling/`
- **Referências no código:** Yang, Meng & Karniadakis, B-PINNs, J. Comput. Phys. 425 (2021), arXiv:2003.06097
- **Referências adicionadas:** Gal & Ghahramani, Dropout as a Bayesian approximation, ICML 2016, arXiv:1506.02142 · Lakshminarayanan, Pritzel & Blundell, Deep ensembles, NeurIPS 2017, arXiv:1612.01474 · Kendall & Gal, What uncertainties do we need in Bayesian deep learning, NeurIPS 2017, arXiv:1703.04977

### T14 — SINDy
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_residual_analyzer`; `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_sindy_identifier`
- **Código:** `pinneapple_neural/architectures/rom/sindy.py`
- **Referências adicionadas:** Brunton, Proctor & Kutz, Discovering governing equations from data (SINDy), PNAS 113(15) (2016), arXiv:1509.03580

### T15 — Adaptive hyperparameter sweep / architecture search (arena NAS)
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_neural/trainer/adaptive_sweep.py`, `pinneapple_arena/nas.py`
- **Referências:** nenhuma fonte encontrada — pendente de paper

### T16 — Noether trainer (research only)
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_noether_research_only.py::test_dataset_and_trainer_refused`
- **Código:** `pinneapple_neural/trainer/noether_trainer.py`
- **Referências adicionadas:** Alkin et al., AB-UPT, arXiv:2502.09692

### T17 — Post-training advisor / audit
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_neural/trainer/advisor.py`, `pinneapple_neural/trainer/audit.py`
- **Referências:** nenhuma fonte encontrada — pendente de paper

### T18 — Meta-learning: MAML, Reptile
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_adaptation_and_active_learning.py::test_meta_train_reptile_actually_trains_and_adapts`; `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_maml_trainer_trains`; `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_meta_benchmark_pipeline_run` (+3)
- **Código:** `pinneapple_adaptation/meta_learning/`
- **Referências no código:** Finn, Abbeel & Levine, MAML, ICML 2017, arXiv:1703.03400 · Nichol, Achiam & Schulman, On first-order meta-learning algorithms (Reptile), arXiv:1803.02999

### T19 — Transfer learning (freeze, adapters, parametric)
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_parametric_family_transfer`; `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_physics_transfer_adapter`; `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_transfer_trainer_finetunes`
- **Código:** `pinneapple_adaptation/transfer_learning/`
- **Referências:** nenhuma fonte encontrada — pendente de paper

### T20 — Self-scaled quasi-Newton: BFGS / SSBFGS / SSBroyden (new, 2026-09-24)
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_self_scaled_qn.py::test_small_pinn_beats_tolerance_quickly`
- **Código:** `pinneapple_neural/trainer/self_scaled_qn.py`
- **Referências no código:** Urban, Stefanou & Pons, Unveiling the optimization process of PINNs, J. Comput. Phys. 523 (2025) 113656, arXiv:2405.04230 -- eqs. (7)-(23), Appendix B · Jnini, Kiyani, Shukla et al., Curvature-Aware Optimization for High-Accuracy PINNs, arXiv:2604.05230 (benchmarks: Euler/HLLC, Helmholtz, inviscid Burgers, Stokes, PK-PD) · Wang, Teng & Perdikaris, gradient pathologies in PINNs, SIAM J. Sci. Comput. 43(5) (2021), arXiv:2001.04536 -- Helmholtz benchmark

### T21 — Autoresearch loop (fixed-budget trials, keep/revert, random or LLM proposer)
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_arena_autoresearch.py::test_loop_keeps_improvements_reverts_the_rest_and_logs`; `PINNeAPPle/tests/test_arena_autoresearch.py::test_real_pinn_template_runs_and_reports_the_metric`
- **Código:** `pinneapple_arena/autoresearch.py`
- **Referências adicionadas:** karpathy/autoresearch (MIT), https://github.com/karpathy/autoresearch

### T22 — Large Physics Model blocks (Fourier encoding, multi-scale neighbourhoods, geometry code, ensemble UQ, OOD score, fine-tuning)
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_lpm.py::test_fourier_encoding_shape`; `PINNeAPPle/tests/test_lpm.py::test_neighbourhood_features_do_not_depend_on_surface_point_order`; `PINNeAPPle/tests/test_lpm.py::test_query_independence_permutation_and_subsets`
- **Código:** `pinneapple_neural/lpm.py`
- **Referências no código:** Luminary, Vocabulary of Physics AI / How Large Physics Models gain spatial context (2026) · Tancik et al., Fourier features let networks learn high frequency functions, NeurIPS 2020, arXiv:2006.10739 · Zaheer et al., Deep Sets, NeurIPS 2017, arXiv:1703.06114 · Lakshminarayanan, Pritzel & Blundell, Deep ensembles, NeurIPS 2017, arXiv:1612.01474
- **Referências adicionadas:** Milne-Thomson, Theoretical Hydrodynamics, §9.61 (ellipse potential flow, benchmark reference)


## C. Equações físicas (E)

### E1 — Laplace
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_gradient_backend_consistency.py::test_default_grad_method_autograd_unchanged_manufactured_solution`; `PINNeAPPle/tests/test_gradient_backend_consistency.py::test_gradient_backends_agree_laplace_2d`; `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_laplace_2d_exact_solution_gives_zero_residual` (+2)
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Evans, Partial Differential Equations, 2nd ed., AMS, 2010

### E2 — Poisson
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/pinneapple_solvers/test_fdm_robin.py::test_robin_poisson_matches_analytical_linear_profile`; `PINNeAPPle/tests/test_codegen.py::test_fdm_poisson_2d_matches_manufactured_solution`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Evans, Partial Differential Equations, 2nd ed., AMS, 2010

### E3 — Helmholtz
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions_batch2.py::test_helmholtz_eigenfunction`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Evans, Partial Differential Equations, 2nd ed., AMS, 2010

### E4 — Heat (steady anisotropic / transient)
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_codegen.py::test_fdm_heat_1d_matches_closed_form`; `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_heat_equation_steady_anisotropic_exact_solution_gives_zero_residual`; `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_heat_equation_steady_exact_solution_gives_zero_residual` (+1)
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Incropera, DeWitt, Bergman & Lavine, Fundamentals of Heat and Mass Transfer, 7th ed., Wiley, 2011 · Evans, Partial Differential Equations, 2nd ed., AMS, 2010

### E5 — Wave
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions_batch2.py::test_wave_standing_mode`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Evans, Partial Differential Equations, 2nd ed., AMS, 2010

### E6 — Advection-diffusion / convection
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions_batch2.py::test_advection_diffusion_decaying_travelling_wave`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** LeVeque, Finite Difference Methods for Ordinary and Partial Differential Equations, SIAM, 2007

### E7 — Burgers (viscous / inviscid)
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_codegen.py::test_fdm_burgers_1d_matches_traveling_front`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Burgers, A mathematical model illustrating the theory of turbulence, Adv. Appl. Mech. 1 (1948) 171-199

### E8 — Reaction-diffusion 1D/2D (Gray-Scott)
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_gradient_backend_consistency.py::test_gradient_backends_agree_reaction_diffusion_2d`; `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_reaction_diffusion_2d_exact_solution_gives_zero_residual`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Pearson, Complex patterns in a simple system, Science 261 (1993) 189-192

### E9 — Fisher-KPP
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_pdb_benchmark_catalog.py::test_pdebench_diffusion_reaction_benchmark_matches_ablowitz_zeppetella_front`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Fisher, The wave of advance of advantageous genes, Ann. Eugen. 7(4) (1937) 355-369 · Ablowitz & Zeppetella, Bull. Math. Biol. 41(6) (1979) 835-840 -- exact travelling wave

### E10 — Incompressible Navier-Stokes 2D/3D
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_lid_driven_cavity_domain_2d`; `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_lid_driven_cavity_domain_3d`; `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_lid_driven_cavity_solver_3d` (+10)
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Batchelor, An Introduction to Fluid Dynamics, Cambridge University Press, 1967 · Ghia, Ghia & Shin, J. Comput. Phys. 48 (1982) 387-411 -- lid-driven cavity benchmark

### E11 — Navier-Stokes in a rotating frame
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_ns_rotating_frame_solid_body_rotation_gives_zero_residual`; `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_ns_rotating_frame_wrong_solution_gives_nonzero_residual`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Batchelor, An Introduction to Fluid Dynamics, Cambridge University Press, 1967

### E12 — Stokes
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions_batch2.py::test_stokes_polynomial_flow`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Batchelor, An Introduction to Fluid Dynamics, Cambridge University Press, 1967

### E13 — Brinkman
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions_batch2.py::test_brinkman_boundary_layer_profile`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Brinkman, A calculation of the viscous force exerted by a flowing fluid on a dense swarm of particles, Appl. Sci. Res. A1 (1949) 27-34

### E14 — Darcy
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_pdb_benchmark_catalog.py::test_list_benchmarks_contains_lane_emden_entries`; `PINNeAPPle/tests/test_pdb_benchmark_catalog.py::test_pdebench_darcy_benchmark_field_is_physically_sane`; `PINNeAPPle/tests/test_pdb_benchmark_catalog.py::test_pdebench_darcy_benchmark_reference_source_cites_pdebench` (+2)
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Darcy, Les fontaines publiques de la ville de Dijon, 1856 · Takamoto et al., PDEBench, arXiv:2210.07182 (Darcy 2D benchmark)

### E15 — Buckley-Leverett two-phase
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Buckley & Leverett, Mechanism of fluid displacement in sands, Trans. AIME 146 (1942) 107-116

### E16 — Compressible Euler 1D/2D/axisymmetric/rotating 3D
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_astrophysics_validation.py::test_euler_compressible_1d_smooth_advected_pulse_gives_near_zero_residual`; `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_compressible_euler_rotating_3d_matches_independent_closed_form`; `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_k_omega_sst_3d_matches_independent_closed_form`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics, 3rd ed., Springer, 2009 · Sod, J. Comput. Phys. 27 (1978) 1-31 -- shock tube

### E17 — Shallow water 1D/2D
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_shallow_water_fv.py::test_ritter_dry_bed_dam_break`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Toro, Shock-Capturing Methods for Free-Surface Shallow Flows, Wiley, 2001

### E18 — Stommel gyre
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_stommel_gyre_2d_exact_solution_gives_zero_residual`; `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_stommel_gyre_2d_wrong_solution_gives_nonzero_residual`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Stommel, The westward intensification of wind-driven ocean currents, Trans. AGU 29(2) (1948) 202-206

### E19 — Plane-stress / plane-strain linear elasticity
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_cartesian_breadth.py::test_cartesian_architecture_preset_trains_meaningfully`; `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_elasticity_2d_exact_solution_gives_zero_residual`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Timoshenko & Goodier, Theory of Elasticity, 3rd ed., McGraw-Hill, 1970

### E20 — Axisymmetric elasticity (+ torsion) / Lame
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_axisymmetric_elasticity_torsion_exact_solution_gives_zero_residual`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Timoshenko & Goodier, Theory of Elasticity, 3rd ed., McGraw-Hill, 1970

### E21 — Neo-Hookean hyperelasticity
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Holzapfel, Nonlinear Solid Mechanics, Wiley, 2000

### E22 — Euler-Bernoulli beam
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_beam_nonlinear_fem.py::test_small_load_cantilever_matches_linear_closed_form_tip_deflection`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`, `pinneapple_systems/process_components/beam_statics.py`
- **Referências adicionadas:** Timoshenko & Goodier, Theory of Elasticity, 3rd ed., McGraw-Hill, 1970

### E23 — von Karman beam / plate
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_beam_nonlinear_fem.py::test_small_load_cantilever_matches_linear_closed_form_tip_deflection`; `PINNeAPPle/tests/test_nonlinear_beam_fem_transient.py::test_cantilever_first_natural_frequency_matches_closed_form`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`, `pinneapple_systems/process_components/beam_nonlinear_fem.py`
- **Referências adicionadas:** Timoshenko & Woinowsky-Krieger, Theory of Plates and Shells, 2nd ed., McGraw-Hill, 1959

### E24 — Thermoelasticity
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Boley & Weiner, Theory of Thermal Stresses, Wiley, 1960

### E25 — Phase-field fracture
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Bourdin, Francfort & Marigo, J. Mech. Phys. Solids 48 (2000) 797-826 · Miehe, Hofacker & Welschinger, Comput. Methods Appl. Mech. Eng. 199 (2010) 2765-2778

### E26 — Biot poroelasticity
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Biot, General theory of three-dimensional consolidation, J. Appl. Phys. 12 (1941) 155-164

### E27 — Maxwell TE
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions_batch2.py::test_maxwell_te_plane_wave`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Jackson, Classical Electrodynamics, 3rd ed., Wiley, 1999

### E28 — Bekker-Wong terramechanics
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_bekker_wong_terramechanics_satisfying_solution_gives_zero_residual`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Bekker, Introduction to Terrain-Vehicle Systems, University of Michigan Press, 1969 · Wong, Theory of Ground Vehicles, Wiley, 1978 (4th ed. 2008)

### E29 — Compressor meanline 1D
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_compressible_euler_rotating_3d_matches_independent_closed_form`; `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_compressor_meanline_1d_exact_solution_gives_zero_residual`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Dixon & Hall, Fluid Mechanics and Thermodynamics of Turbomachinery, 7th ed., Elsevier, 2014

### E30 — Phonon Boltzmann transport 1D (gray)
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_astrophysics_validation.py::test_schwarzschild_light_bending_perturbative_solution_near_zero_residual`; `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_k_omega_sst_3d_matches_independent_closed_form`; `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_phonon_bte_1d_gray_exact_solution_gives_zero_residual`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Chen, Nanoscale Energy Transport and Conversion, Oxford University Press, 2005

### E31 — Black-Scholes
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_black_scholes_1d_exact_solution_gives_zero_residual`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Black & Scholes, The pricing of options and corporate liabilities, J. Polit. Econ. 81(3) (1973) 637-654

### E32 — Heston
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Heston, A closed-form solution for options with stochastic volatility, Rev. Financ. Stud. 6(2) (1993) 327-343

### E33 — Opinion dynamics (continuum Hegselmann-Krause)
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions_batch2.py::test_opinion_dynamics_stationary_kink`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Hegselmann & Krause, Opinion dynamics and bounded confidence, JASSS 5(3) (2002)

### E34 — Kepler two-body orbit
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_astrophysics_validation.py::test_kepler_two_body_orbit_exact_gives_near_zero_residual`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Vallado, Fundamentals of Astrodynamics and Applications, 4th ed., Microcosm Press, 2013

### E35 — Circular restricted three-body problem (planar, synodic)
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_astrophysics_validation.py::test_cr3bp_l1_l2_l3_equilibria_exact_gives_near_zero_residual`; `PINNeAPPle/tests/test_astrophysics_validation.py::test_cr3bp_l4_equilibrium_exact_gives_near_zero_residual`; `PINNeAPPle/tests/test_nbody_rebound.py::test_cr3bp_l4_stays_fixed_in_rotating_frame`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Szebehely, Theory of Orbits: The Restricted Problem of Three Bodies, Academic Press, 1967

### E36 — J2 perturbation
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_astrophysics_validation.py::test_satellite_j2_perturbation_reduces_to_two_body_when_j2_zero`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Vallado, Fundamentals of Astrodynamics and Applications, 4th ed., Microcosm Press, 2013

### E37 — Clohessy-Wiltshire relative motion
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_astrophysics_validation.py::test_space_debris_cw_exact_gives_near_zero_residual`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Clohessy & Wiltshire, Terminal guidance system for satellite rendezvous, J. Aerosp. Sci. 27(9) (1960) 653-658

### E38 — Euler rigid-body attitude
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_astrophysics_validation.py::test_spacecraft_attitude_exact_gives_near_zero_residual`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Markley & Crassidis, Fundamentals of Spacecraft Attitude Determination and Control, Springer, 2014

### E39 — Lane-Emden polytrope
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_astrophysics_validation.py::test_lane_emden_n0_exact_gives_near_zero_residual`; `PINNeAPPle/tests/test_astrophysics_validation.py::test_lane_emden_n1_exact_gives_near_zero_residual`; `PINNeAPPle/tests/test_lane_emden_numerical_validation.py::test_lane_emden_astrophysically_standard_n_matches_published_tables` (+2)
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Chandrasekhar, An Introduction to the Study of Stellar Structure, 1939 · Hansen, Kawaler & Trimble, Stellar Interiors, 2nd ed., Springer, 2004 (Table 4.1)

### E40 — Shakura-Sunyaev accretion disk
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_astrophysics_validation.py::test_shakura_sunyaev_exact_flux_gives_near_zero_residual`; `PINNeAPPle/tests/test_astrophysics_validation.py::test_shakura_sunyaev_exact_matches_module_reference_function`; `PINNeAPPle/tests/test_shakura_sunyaev_disk_validation.py::test_shakura_sunyaev_far_field_teff_matches_r_minus_three_quarters_power_law` (+3)
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Shakura & Sunyaev, Black holes in binary systems, Astron. Astrophys. 24 (1973) 337-355

### E41 — Schwarzschild weak-field light bending
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_astrophysics_validation.py::test_schwarzschild_light_bending_perturbative_solution_near_zero_residual`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Misner, Thorne & Wheeler, Gravitation, W. H. Freeman, 1973

### E42 — SIR epidemic
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions_batch2.py::test_sir_logistic_limit`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Kermack & McKendrick, Proc. R. Soc. A 115 (1927) 700-721

### E43 — Pharmacokinetics two-compartment
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions_batch2.py::test_pk_two_compartment_eigen_solution`
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`
- **Referências adicionadas:** Gibaldi & Perrier, Pharmacokinetics, 2nd ed., Marcel Dekker, 1982

### E44 — Reaction network / Arrhenius / mass action
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_advection_dispersion_reaction_solver`; `PINNeAPPle/tests/test_reaction_kinetics.py::test_arrhenius_rate_constant_increases_with_temperature_for_positive_activation_energy`; `PINNeAPPle/tests/test_reaction_kinetics.py::test_arrhenius_requires_temperature` (+4)
- **Código:** `pinneapple_physics/pinn_solver/compiler/compile.py`, `pinneapple_systems/process_components/reaction_kinetics.py`
- **Referências adicionadas:** Fogler, Elements of Chemical Reaction Engineering, 5th ed., Prentice Hall, 2016

### E45 — Hagen-Poiseuille
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle-CFD/tests/test_e02_modelo_matematico.py::test_correlacoes_de_referencia`; `PINNeAPPle-CFD/tests/test_e04_execucao.py::test_correlacoes_classicas`; `PINNeAPPle-CFD/tests/test_e10_comparacao.py::test_ca_e10_02_tubo_reto_colebrook_e_hagen_poiseuille` (+2)
- **Código:** `PINNeAPPle-CFD/pinneapple_cfd/etapas/e02_modelo_matematico/correlacoes.py`
- **Referências adicionadas:** White, Fluid Mechanics, 7th ed., McGraw-Hill, 2011

### E46 — Blasius friction factor
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_pdb_benchmark_catalog.py::test_blasius_benchmark_matches_published_wall_shear_parameter`; `PINNeAPPle/tests/test_pdb_benchmark_catalog.py::test_blasius_benchmark_reference_source_cites_published_value`
- **Código:** `PINNeAPPle-CFD/.../correlacoes.py`
- **Referências adicionadas:** White, Fluid Mechanics, 7th ed., McGraw-Hill, 2011

### E47 — Colebrook-White
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle-CFD/tests/test_e02_modelo_matematico.py::test_correlacoes_de_referencia`; `PINNeAPPle-CFD/tests/test_e04_execucao.py::test_correlacoes_classicas`; `PINNeAPPle-CFD/tests/test_e09_treino.py::test_tubo_reto_atinge_meta_mvp_e_sinaliza_ood` (+5)
- **Código:** `PINNeAPPle-CFD/.../correlacoes.py`, `pinneapple_systems/process_components/pipe_network_1d.py`
- **Referências adicionadas:** Colebrook, Turbulent flow in pipes, J. Inst. Civ. Eng. 11 (1939) 133-156

### E48 — Darcy-Weisbach pressure drop
- **Estado:** ⚪ sem teste
- **Código:** `PINNeAPPle-CFD/.../correlacoes.py`
- **Referências adicionadas:** White, Fluid Mechanics, 7th ed., McGraw-Hill, 2011

### E49 — Herschel-Bulkley + Metzner-Reed (non-Newtonian pipe flow)
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_non_newtonian_pipe_flow.py::test_herschel_bulkley_reduces_to_newtonian_at_zero_yield_stress_and_n_equal_1`; `PINNeAPPle/tests/test_non_newtonian_pipe_flow.py::test_laminar_friction_factor_matches_hagen_poiseuille_16_over_Re`; `PINNeAPPle/tests/test_non_newtonian_pipe_flow.py::test_turbulent_friction_factor_matches_blasius_correlation` (+1)
- **Código:** `pinneapple_systems/process_components/non_newtonian_pipe_flow.py`
- **Referências adicionadas:** Herschel & Bulkley, Kolloid-Z. 39 (1926) 291-300 · Metzner & Reed, Flow of non-Newtonian fluids, AIChE J. 1(4) (1955) 434-440

### E50 — DNV-RP-O501 erosion (straight pipe and bend)
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle-CFD/tests/test_e01_descricao_problema.py::test_dureza_brinell_entra_na_ficha_e_libera_o_ecrc`; `PINNeAPPle-CFD/tests/test_e01_descricao_problema.py::test_ficha_aceita_pelo_e2`; `PINNeAPPle-CFD/tests/test_e02_modelo_matematico.py::test_ca_e2_03_sem_velocidade_nem_vazao_nao_gera_caso` (+43)
- **Código:** `PINNeAPPle-CFD/pinneapple_cfd/etapas/e02_modelo_matematico/erosao.py`
- **Referências adicionadas:** DNV-RP-O501 rev. 4.2, Managing Sand Production and Erosion, 2007/2011

### E51 — Lame thick-walled cylinder
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_physics_knowledge_base.py::test_sourced_presets_actually_build_a_problem_spec`; `PINNeAPPle/tests/test_pipe_stress_mechanics.py::test_lame_hoop_inner_exceeds_outer_for_a_thick_wall_under_internal_pressure`; `PINNeAPPle/tests/test_pipe_stress_mechanics.py::test_lame_hoop_stress_matches_thin_wall_pressure_vessel_formula_at_small_t_over_r`
- **Código:** `pinneapple_systems/process_components/pipe_stress_mechanics.py`
- **Referências adicionadas:** Timoshenko & Goodier, Theory of Elasticity, 3rd ed., McGraw-Hill, 1970

### E52 — von Mises equivalent stress
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_beam_statics.py::test_von_mises_stress_matches_pure_bending_limit_when_shear_is_zero`; `PINNeAPPle/tests/test_pipe_stress_mechanics.py::test_von_mises_pure_uniaxial_tension_equals_the_applied_stress`; `PINNeAPPle/tests/test_pipe_stress_mechanics.py::test_von_mises_triaxial_is_zero_under_a_purely_hydrostatic_stress_state` (+1)
- **Código:** `pinneapple_systems/process_components/pipe_stress_mechanics.py`, `pinneapple_systems/process_components/beam_statics.py`
- **Referências adicionadas:** von Mises, Goettinger Nachrichten, Math.-Phys. Kl. (1913) 582-592 · Budynas & Nisbett, Shigley's Mechanical Engineering Design, 10th ed., McGraw-Hill, 2015

### E53 — Euler buckling + constrained rod buckling
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_pipe_stress_mechanics.py::test_euler_buckling_matches_textbook_pinned_pinned_closed_form`
- **Código:** `pinneapple_systems/process_components/pipe_stress_mechanics.py`
- **Referências no código:** Paslay & Dawson 1964; Mitchell 1988 -- constrained rod buckling
- **Referências adicionadas:** Timoshenko & Gere, Theory of Elastic Stability, 2nd ed., McGraw-Hill, 1961

### E54 — Beam-column moment amplification factor
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_pipe_stress_mechanics.py::test_beam_column_amplification_factor_is_one_at_zero_axial_load`; `PINNeAPPle/tests/test_pipe_stress_mechanics.py::test_beam_column_amplification_factor_matches_1_over_1_minus_ratio_below_the_clip`; `PINNeAPPle/tests/test_pipe_stress_mechanics.py::test_beam_column_amplification_factor_stays_finite_at_the_critical_load`
- **Código:** `pinneapple_systems/process_components/pipe_stress_mechanics.py`
- **Referências adicionadas:** Timoshenko & Gere, Theory of Elastic Stability, 2nd ed., McGraw-Hill, 1961

### E55 — S-N curve + Goodman + Miner
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/pinneapple_physics/closed_form/test_cantilever_fatigue.py::test_allowable_stress_round_trips_with_fatigue_life`; `PINNeAPPle/tests/pinneapple_physics/closed_form/test_cantilever_fatigue.py::test_position_profile_matches_root_at_x0_and_zero_at_tip`; `PINNeAPPle/tests/pinneapple_physics/closed_form/test_cantilever_fatigue.py::test_root_stress_matches_ridetune_ai_baseline` (+9)
- **Código:** `pinneapple_systems/process_components/fatigue_analysis.py`, `pinneapple_physics/closed_form/cantilever_fatigue.py`
- **Referências adicionadas:** Miner, Cumulative damage in fatigue, J. Appl. Mech. 12 (1945) A159-A164 · Budynas & Nisbett, Shigley's Mechanical Engineering Design, 10th ed., McGraw-Hill, 2015

### E56 — Torsional stick-slip with Stribeck friction
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_torsional_stickslip.py::test_larger_static_kinetic_gap_produces_more_severe_stick_slip_near_breakout_speed`; `PINNeAPPle/tests/test_torsional_stickslip.py::test_n_nodes_below_three_is_rejected`; `PINNeAPPle/tests/test_torsional_stickslip.py::test_result_shapes_are_internally_consistent` (+3)
- **Código:** `pinneapple_systems/process_components/torsional_stickslip.py`
- **Referências adicionadas:** Armstrong-Helouvry, Dupont & Canudas de Wit, A survey of models, analysis tools and compensation methods for the control of machines with friction, Automatica 30(7) (1994) 1083-1138

### E57 — Heat exchanger (effectiveness-NTU)
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_artifact_registry.py::test_model_store_save_load_latest_and_promote`; `PINNeAPPle/tests/test_artifact_registry.py::test_triton_export_produces_config_and_onnx`; `PINNeAPPle/tests/test_component_library.py::test_checkpoint_round_trip_reconstructs_from_component_name` (+5)
- **Código:** `pinneapple_systems/process_components/heat_exchanger.py`
- **Referências adicionadas:** Incropera, DeWitt, Bergman & Lavine, Fundamentals of Heat and Mass Transfer, 7th ed., Wiley, 2011

### E58 — Fin array conduction
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/pinneapple_physics/closed_form/test_fin_array_conduction.py::test_matches_brakecool_ai_baseline_radial_profile`; `PINNeAPPle/tests/pinneapple_physics/closed_form/test_fin_array_conduction.py::test_matches_heatsink_design_ai_baseline_exactly`; `PINNeAPPle/tests/pinneapple_physics/closed_form/test_fin_array_conduction.py::test_radial_profile_supports_a_different_corrected_length_than_the_resistance_network` (+1)
- **Código:** `pinneapple_physics/closed_form/fin_array_conduction.py`
- **Referências adicionadas:** Incropera, DeWitt, Bergman & Lavine, Fundamentals of Heat and Mass Transfer, 7th ed., Wiley, 2011

### E59 — Control valve (Cv, compressible / incompressible)
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_process_components.py::test_installed_cv_monotonic_and_bounded`
- **Código:** `pinneapple_systems/process_components/control_valve.py`
- **Referências adicionadas:** IEC 60534-2-1 / ANSI/ISA-75.01.01 -- flow equations for sizing control valves

### E60 — Polytropic compression path
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_process_components.py::test_expansion_direction_reduces_temperature_and_efficiency_convention_is_bounded`
- **Código:** `pinneapple_systems/process_components/polytropic_path.py`
- **Referências adicionadas:** Schultz, The polytropic analysis of centrifugal compressors, J. Eng. Power 84(1) (1962) 69-82

### E61 — Real gas equation of state (CoolProp)
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_process_component_exceptions`; `PINNeAPPle/tests/test_breadth_six_packages.py::test_breadth_transient_pipe`
- **Código:** `pinneapple_systems/process_components/real_gas_eos.py`
- **Referências adicionadas:** Bell, Wronski, Quoilin & Lemort, CoolProp, Ind. Eng. Chem. Res. 53(6) (2014) 2498-2508

### E62 — Compressor similarity map
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_systems/process_components/similarity_map.py`
- **Referências no código:** PennWell 2003
- **Referências adicionadas:** Dixon & Hall, Fluid Mechanics and Thermodynamics of Turbomachinery, 2014

### E63 — Helmholtz resonator
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/pinneapple_physics/closed_form/test_helmholtz_resonator.py::test_larger_cabinet_lowers_frequency`; `PINNeAPPle/tests/pinneapple_physics/closed_form/test_helmholtz_resonator.py::test_larger_port_volume_raises_frequency`; `PINNeAPPle/tests/pinneapple_physics/closed_form/test_helmholtz_resonator.py::test_matches_soundshape_ai_baseline` (+1)
- **Código:** `pinneapple_physics/closed_form/helmholtz_resonator.py`
- **Referências adicionadas:** Kinsler, Frey, Coppens & Sanders, Fundamentals of Acoustics, 4th ed., Wiley, 2000

### E64 — Membrane diffusion
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/pinneapple_physics/closed_form/test_membrane_diffusion.py::test_concentration_profile_is_linear_and_matches_flux_gradient`; `PINNeAPPle/tests/pinneapple_physics/closed_form/test_membrane_diffusion.py::test_decay_constant_independent_of_area_and_equals_inverse_depletion_time`; `PINNeAPPle/tests/pinneapple_physics/closed_form/test_membrane_diffusion.py::test_depletion_time_independent_of_area` (+2)
- **Código:** `pinneapple_physics/closed_form/membrane_diffusion.py`
- **Referências adicionadas:** Crank, The Mathematics of Diffusion, 2nd ed., Oxford University Press, 1975

### E65 — Cantilever fatigue
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/pinneapple_physics/closed_form/test_cantilever_fatigue.py::test_allowable_stress_round_trips_with_fatigue_life`; `PINNeAPPle/tests/pinneapple_physics/closed_form/test_cantilever_fatigue.py::test_position_profile_matches_root_at_x0_and_zero_at_tip`; `PINNeAPPle/tests/pinneapple_physics/closed_form/test_cantilever_fatigue.py::test_root_stress_matches_ridetune_ai_baseline` (+1)
- **Código:** `pinneapple_physics/closed_form/cantilever_fatigue.py`
- **Referências adicionadas:** Budynas & Nisbett, Shigley's Mechanical Engineering Design, 10th ed., McGraw-Hill, 2015


## D. Problemas físicos (P)

### P1.1 — Academic: burgers_1d
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_codegen.py::test_fdm_burgers_1d_matches_traveling_front`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E7

### P1.2 — Academic: channel_flow_3d
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_physics_guardrail.py::test_conservation_mass_continuity_exact_divergence_free_field_is_near_zero`; `PINNeAPPle/tests/test_physics_guardrail.py::test_conservation_mass_continuity_nondivergence_free_field_is_clearly_flagged`; `PINNeAPPle/tests/test_physics_guardrail.py::test_dimensional_analysis_navier_stokes_positive_Re_passes` (+1)
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E10

### P1.3 — Academic: darcy_pressure_only_3d
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E14

### P1.4 — Academic: helmholtz_acoustics_3d
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E3

### P1.5 — Academic: laplace_2d
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_gradient_backend_consistency.py::test_default_grad_method_autograd_unchanged_manufactured_solution`; `PINNeAPPle/tests/test_gradient_backend_consistency.py::test_gradient_backends_agree_laplace_2d`; `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_laplace_2d_exact_solution_gives_zero_residual` (+2)
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E1

### P1.6 — Academic: lid_driven_cavity_3d
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E10

### P1.7 — Academic: ns_incompressible_2d
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_codegen.py::test_classify_physics_shape_on_real_presets`; `PINNeAPPle/tests/test_codegen.py::test_fenics_stokes_script_is_valid_and_has_expected_api`; `PINNeAPPle/tests/test_codegen.py::test_pdeparser_rejects_uncovered_kind` (+4)
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E10

### P1.8 — Academic: pipe_flow_3d
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E10

### P1.9 — Academic: poisson_2d
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_codegen.py::test_fdm_poisson_2d_matches_manufactured_solution`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E2

### P1.10 — Academic: reaction_diffusion_2d
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_gradient_backend_consistency.py::test_gradient_backends_agree_reaction_diffusion_2d`; `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_reaction_diffusion_2d_exact_solution_gives_zero_residual`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E8

### P1.11 — Academic: steady_heat_conduction_3d
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/pinneapple_analysis/test_retrieval.py::test_corpus_includes_real_presets`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E4

### P1.12 — Academic: transient_heat_3d
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E4

### P1.13 — Academic: wave_ultrasound_3d
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E5

### P2.1 — Solids: axisymmetric_linear_elasticity_2d
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E20

### P2.2 — Solids: linear_elasticity_3d
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_codegen.py::test_classify_physics_shape_on_real_presets`; `PINNeAPPle/tests/test_codegen.py::test_fenics_elasticity_contact_script_is_valid_and_has_expected_api`; `PINNeAPPle/tests/test_codegen.py::test_fenics_elasticity_without_contact_has_config_helper` (+2)
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E19

### P2.3 — Solids: linear_elasticity_3d_industry
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_missing_presets_registered.py::test_linear_elasticity_3d_industry_does_not_collide_with_structural_variant`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E19

### P2.4 — Solids: material_fracture_2d
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E25

### P2.5 — Solids: plane_strain_2d
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E19

### P2.6 — Solids: plane_stress_2d
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_cartesian_breadth.py::test_cartesian_architecture_preset_trains_meaningfully`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E19

### P2.7 — Solids: rotary_coupling_torsion
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E20

### P2.8 — Solids: thermoelasticity_2d
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E24

### P2.9 — Solids: thick_walled_cylinder_lame
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_physics_knowledge_base.py::test_sourced_presets_actually_build_a_problem_spec`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E20

### P2.10 — Solids: threaded_coupling_tc50_box
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E20

### P2.11 — Solids: threaded_coupling_tc50_pin
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E20

### P2.12 — Solids: threaded_coupling_tc50_rotating
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E20

### P2.13 — Solids: von_mises_2d
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E19, E52

### P3.1 — Aero / automotive: aircraft_wing_aerodynamics
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E10

### P3.2 — Aero / automotive: aircraft_wing_structural
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E19

### P3.3 — Aero / automotive: car_brake_thermal
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_heat_equation_transient_axisymmetric_exact_solution_gives_zero_residual`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E4

### P3.4 — Aero / automotive: car_external_aero
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E10

### P3.5 — Aero / automotive: car_suspension_fatigue
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E19, E55

### P3.6 — Aero / automotive: rocket_nozzle_cfd
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E16

### P3.7 — Aero / automotive: rocket_structural
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E19

### P4.1 — Turbomachinery: axial_compressor_cascade_2d
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E16

### P4.2 — Turbomachinery: axial_compressor_meanline
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_compressor_meanline_1d_exact_solution_gives_zero_residual`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E29

### P4.3 — Turbomachinery: axial_compressor_stage_3d
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_compressible_euler_rotating_3d_matches_independent_closed_form`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E16

### P4.4 — Turbomachinery: fan_cooler_cfd
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_ns_rotating_frame_solid_body_rotation_gives_zero_residual`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E10

### P5.1 — Industrial thermal: cpu_heatsink_thermal
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_heat_equation_steady_exact_solution_gives_zero_residual`; `PINNeAPPle/tests/test_physics_guardrail.py::test_end_to_end_cpu_heatsink_thermal_exact_solution_is_trustworthy`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E4

### P5.2 — Industrial thermal: datacenter_airflow_2d
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E10

### P5.3 — Industrial thermal: datacenter_cfd_3d
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E10

### P5.4 — Industrial thermal: datacenter_server_thermal
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_heat_equation_steady_exact_solution_gives_zero_residual`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E4

### P5.5 — Industrial thermal: furnace_combustion_zone
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E6, E44

### P5.6 — Industrial thermal: industrial_furnace_thermal
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_heat_equation_steady_exact_solution_gives_zero_residual`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E4

### P5.7 — Industrial thermal: pcb_thermal
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_heat_equation_steady_anisotropic_exact_solution_gives_zero_residual`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E4

### P5.8 — Industrial thermal: refractory_lining
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_heat_equation_steady_exact_solution_gives_zero_residual`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E4

### P6.1 — Multidisciplinary: black_scholes_1d
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_black_scholes_1d_exact_solution_gives_zero_residual`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E31

### P6.2 — Multidisciplinary: climate_atmosphere_2d
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E6

### P6.3 — Multidisciplinary: climate_ocean_gyre
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E18

### P6.4 — Multidisciplinary: crystal_phonon
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_phonon_bte_1d_gray_exact_solution_gives_zero_residual`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E30

### P6.5 — Multidisciplinary: drug_diffusion_tissue
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E6

### P6.6 — Multidisciplinary: heston_pde_2d
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E32

### P6.7 — Multidisciplinary: opinion_dynamics_2d
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions_batch2.py::test_opinion_dynamics_stationary_kink`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E33

### P6.8 — Multidisciplinary: pk_two_compartment
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions_batch2.py::test_pk_two_compartment_eigen_solution`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E43

### P6.9 — Multidisciplinary: sir_epidemic
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E42

### P7.1 — Astro / space: cr3bp_planar_synodic
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_astrophysics_validation.py::test_cr3bp_l1_l2_l3_equilibria_exact_gives_near_zero_residual`; `PINNeAPPle/tests/test_astrophysics_validation.py::test_cr3bp_l4_equilibrium_exact_gives_near_zero_residual`; `PINNeAPPle/tests/test_nbody_rebound.py::test_cr3bp_l4_stays_fixed_in_rotating_frame`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E35

### P7.2 — Astro / space: kepler_two_body_orbit
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_astrophysics_validation.py::test_kepler_two_body_orbit_exact_gives_near_zero_residual`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E34

### P7.3 — Astro / space: lane_emden_polytrope
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_astrophysics_validation.py::test_lane_emden_n0_exact_gives_near_zero_residual`; `PINNeAPPle/tests/test_astrophysics_validation.py::test_lane_emden_n1_exact_gives_near_zero_residual`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E39

### P7.4 — Astro / space: nfw_dark_matter_potential
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_astrophysics_validation.py::test_nfw_potential_exact_gives_near_zero_residual`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E2

### P7.5 — Astro / space: satellite_j2_perturbation
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_astrophysics_validation.py::test_satellite_j2_perturbation_reduces_to_two_body_when_j2_zero`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E36

### P7.6 — Astro / space: schwarzschild_light_bending_weak_field
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_astrophysics_validation.py::test_schwarzschild_light_bending_perturbative_solution_near_zero_residual`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E41

### P7.7 — Astro / space: shakura_sunyaev_accretion_disk
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_shakura_sunyaev_disk_validation.py::test_shakura_sunyaev_numerical_integration_matches_closed_form`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E40

### P7.8 — Astro / space: sod_shock_tube_astro
- **Estado:** ⚪ sem teste
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E16

### P7.9 — Astro / space: space_debris_cw_relative_motion
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_astrophysics_validation.py::test_space_debris_cw_exact_gives_near_zero_residual`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E37

### P7.10 — Astro / space: spacecraft_attitude_euler_rotation
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_astrophysics_validation.py::test_spacecraft_attitude_exact_gives_near_zero_residual`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E38

### P8.1 — Terramechanics: bekker_wong_surrogate_2d
- **Estado:** ✅ validado
- **Evidência:** `PINNeAPPle/tests/test_manufactured_solutions.py::test_audit_physics_bekker_wong_terramechanics_satisfying_solution_gives_zero_residual`
- **Código:** `pinneapple_physics/pde_environment/presets/`
- **Equações:** E28

### P9 — PINNeAPPle-CFD v1: pipes + particle erosion (CFD-04 pilot)
- **Estado:** 🟡 testado, sem referência
- **Evidência:** `PINNeAPPle-CFD/tests/test_e11_relatorio_pdf.py::test_api_rota_pdf_e_pdf_disponivel`; `PINNeAPPle-CFD/tests/test_e11_relatorio_pdf.py::test_carregador_por_arquivo_nao_deixa_modulo_meio_carregado`; `PINNeAPPle-CFD/tests/test_jornada.py::test_campanha_so_cobre_o_cfd04`
- **Código:** `PINNeAPPle-CFD/pinneapple_cfd/`
- **Equações:** E10, E45, E46, E47, E48, E50
