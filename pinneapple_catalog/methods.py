"""Every solver, training method, equation and problem in PINNeAPPle, with
its code, its references and its validation status.

IDs match the inventory of 2026-09-24 (``docs/dev/PEDIDOS_2026-09-24.md``):
S = solver, T = training method, E = physical equation, P = physical problem.

References come in two kinds, kept apart on purpose:

- ``refs_in_code``: what the module's own docstring already cites.
- ``refs_added``: the canonical source (original paper or standard textbook)
  added here because the code cites nothing. Every arXiv id in both lists
  was checked against the paper title on 2026-09-24 (that check found and
  fixed a wrong id: X-TFC cited arXiv:2005.01219, a metasurface paper, for
  Leake & Mortari; the right id is arXiv:1812.08625).

Validation status (decision D5) is NOT written by hand: it is computed by
``scripts/build_method_status.py`` from ``tests/`` and stored in
``method_status.json``. An item is ``validated`` when a test that exercises
it compares against an independent reference (closed form, published
value or manufactured solution); ``tested`` when tests exist but none is a
reference comparison; ``untested`` otherwise. Anything not ``validated``
is "pending: paper + benchmark" and is left out of ``validated_methods()``.
No code is deleted because of this status.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

CATEGORIES = {"S": "solver", "T": "training", "E": "equation", "P": "problem"}


@dataclass(frozen=True)
class Method:
    id: str
    name: str
    code: Tuple[str, ...]
    probes: Tuple[str, ...]  # strings whose presence in a test file means the test exercises this item
    refs_in_code: Tuple[str, ...] = ()
    refs_added: Tuple[str, ...] = ()
    equations: Tuple[str, ...] = ()  # problems only: the E ids they use

    @property
    def category(self) -> str:
        return CATEGORIES[self.id[0]]

    @property
    def references(self) -> Tuple[str, ...]:
        return self.refs_in_code + self.refs_added


def _m(id, name, code, probes, refs_in_code=(), refs_added=(), equations=()):
    return Method(id, name, tuple(code), tuple(probes), tuple(refs_in_code), tuple(refs_added), tuple(equations))


NS = "pinneapple_simulation/numerical_solvers/"
TR = "pinneapple_neural/trainer/"
PC = "pinneapple_systems/process_components/"
CF = "pinneapple_physics/closed_form/"
COMPILE = "pinneapple_physics/pinn_solver/compiler/compile.py"

# Standard references reused below.
LEVEQUE_FD = "LeVeque, Finite Difference Methods for Ordinary and Partial Differential Equations, SIAM, 2007"
HAIRER = "Hairer & Wanner, Solving Ordinary Differential Equations II: Stiff and DAE Problems, 2nd ed., Springer, 1996"
TIMO_ELAST = "Timoshenko & Goodier, Theory of Elasticity, 3rd ed., McGraw-Hill, 1970"
TIMO_STAB = "Timoshenko & Gere, Theory of Elastic Stability, 2nd ed., McGraw-Hill, 1961"
INCROPERA = "Incropera, DeWitt, Bergman & Lavine, Fundamentals of Heat and Mass Transfer, 7th ed., Wiley, 2011"
WHITE = "White, Fluid Mechanics, 7th ed., McGraw-Hill, 2011"
EVANS = "Evans, Partial Differential Equations, 2nd ed., AMS, 2010"
TORO = "Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics, 3rd ed., Springer, 2009"
BATCHELOR = "Batchelor, An Introduction to Fluid Dynamics, Cambridge University Press, 1967"
VALLADO = "Vallado, Fundamentals of Astrodynamics and Applications, 4th ed., Microcosm Press, 2013"
SHIGLEY = "Budynas & Nisbett, Shigley's Mechanical Engineering Design, 10th ed., McGraw-Hill, 2015"
JACKSON = "Jackson, Classical Electrodynamics, 3rd ed., Wiley, 1999"
RAISSI = "Raissi, Perdikaris & Karniadakis, Physics-informed neural networks, J. Comput. Phys. 378 (2019) 686-707"
RAISSI_I = "Raissi, Perdikaris & Karniadakis, Physics Informed Deep Learning (Part I), arXiv:1711.10561"
WONG = "Wong, Theory of Ground Vehicles, Wiley, 1978 (4th ed. 2008)"
BEKKER = "Bekker, Introduction to Terrain-Vehicle Systems, University of Michigan Press, 1969"

METHODS: List[Method] = [
    # ── A. Solvers ───────────────────────────────────────────────────────
    _m("S1", "Finite differences 1D/2D/3D", [NS + "fdm.py", NS + "fdm3d.py"], ["numerical_solvers.fdm", "'fdm'", '"fdm"'],
       refs_added=[LEVEQUE_FD]),
    _m("S2", "Finite elements (linear) + nonlinear beam FEM", [NS + "fem.py", NS + "nonlinear_beam_fem.py"],
       ["numerical_solvers.fem", "nonlinear_beam_fem", '"fem"'],
       refs_added=["Hughes, The Finite Element Method: Linear Static and Dynamic FEA, Dover, 2000",
                   "Zienkiewicz, Taylor & Zhu, The Finite Element Method: Its Basis and Fundamentals, 7th ed., 2013"]),
    _m("S3", "Finite volumes", [NS + "fvm.py"], ["numerical_solvers.fvm", '"fvm"'],
       refs_added=["Versteeg & Malalasekera, An Introduction to Computational Fluid Dynamics: The Finite Volume "
                   "Method, 2nd ed., Pearson, 2007",
                   "LeVeque, Finite Volume Methods for Hyperbolic Problems, Cambridge University Press, 2002"]),
    _m("S4", "Spectral (Fourier pseudo-spectral)", [NS + "spectral.py"], ["numerical_solvers.spectral", '"spectral"'],
       refs_added=["Trefethen, Spectral Methods in MATLAB, SIAM, 2000",
                   "Canuto, Hussaini, Quarteroni & Zang, Spectral Methods: Fundamentals in Single Domains, Springer, 2006"]),
    _m("S5", "Lattice Boltzmann D2Q9 / D3Q19 (+ Smagorinsky LES)", [NS + "lbm.py"], ["numerical_solvers.lbm", '"lbm', "LBM"],
       refs_in_code=["Zou & He, Phys. Fluids 9 (1997) 1591 -- pressure/velocity boundary conditions",
                     "Hou, Sterling, Chen & Doolen (1994) -- Smagorinsky LES-LBM"],
       refs_added=["Krueger et al., The Lattice Boltzmann Method: Principles and Practice, Springer, 2017"]),
    _m("S6", "SPH / ISPH / DFSPH", [NS + "sph.py", NS + "isph.py", NS + "dfsph.py"],
       ["numerical_solvers.sph", "numerical_solvers.isph", "numerical_solvers.dfsph", '"sph"', '"isph"', '"dfsph"'],
       refs_in_code=["Morris, Fox & Zhu, J. Comput. Phys. 136 (1997) -- SPH viscosity",
                     "Cummins & Rudman, J. Comput. Phys. 152 (1999) 584-607 -- ISPH",
                     "Shao & Lo, Adv. Water Resour. 26 (2003) 787-800 -- ISPH",
                     "Bender & Koschier, SCA 2015 / IEEE TVCG 2017 -- DFSPH"],
       refs_added=["Monaghan, Smoothed particle hydrodynamics, Rep. Prog. Phys. 68 (2005) 1703"]),
    _m("S7", "Meshfree RBF collocation (Kansa)", [NS + "meshfree.py"], ["numerical_solvers.meshfree", "rbf_collocation"],
       refs_added=["Kansa, Multiquadrics, Comput. Math. Appl. 19(8-9) (1990) 147-161"]),
    _m("S8", "X-TFC (IVP, PDE, subdomains)", [NS + "xtfc_ivp.py", NS + "xtfc_pde.py", NS + "xtfc_subdomain.py"],
       ["xtfc_ivp", "xtfc_pde", "xtfc_subdomain"],
       refs_in_code=["Schiassi et al., Extreme Theory of Functional Connections, arXiv:2005.10632",
                     "Schiassi et al., PI X-TFC for parameter discovery, arXiv:2008.05554",
                     "Leake & Mortari, Deep Theory of Functional Connections, arXiv:1812.08625"]),
    _m("S9", "Discrete-time PINN / implicit Runge-Kutta Gauss-Legendre", [NS + "discrete_time_pinn.py", NS + "irk_gauss_legendre.py"],
       ["discrete_time_pinn", "irk_gauss_legendre"],
       refs_in_code=["Raissi et al., Physics Informed Deep Learning (Part II), arXiv:1711.10566"],
       refs_added=[RAISSI_I, HAIRER]),
    _m("S10", "Stiff kinetics ODE", [NS + "stiff_kinetics_ode.py"], ["stiff_kinetics_ode"], refs_added=[HAIRER]),
    _m("S11", "TVD advection", [NS + "tvd_advection.py"], ["tvd_advection"],
       refs_added=["Harten, High resolution schemes for hyperbolic conservation laws, J. Comput. Phys. 49 (1983) 357-393",
                   "Sweby, High resolution schemes using flux limiters, SIAM J. Numer. Anal. 21(5) (1984) 995-1011"]),
    _m("S12", "Immersed boundary (FDM)", [NS + "immersed_boundary_fdm.py"], ["immersed_boundary"],
       refs_added=["Peskin, The immersed boundary method, Acta Numerica 11 (2002) 479-517",
                   "Mittal & Iaccarino, Immersed boundary methods, Annu. Rev. Fluid Mech. 37 (2005) 239-261"]),
    _m("S13", "Beam BVP FDM / 3D elasticity FDM", [NS + "beam_bvp_fdm.py", NS + "elasticity3d_fdm.py"],
       ["beam_bvp_fdm", "elasticity3d_fdm"], refs_added=[TIMO_ELAST, LEVEQUE_FD]),
    _m("S14", "Eddy current FDM", [NS + "eddy_current_fdm.py"], ["eddy_current"], refs_added=[JACKSON]),
    _m("S15", "Bekker-Wong terramechanics", [NS + "bekker_wong.py", "pinneapple_simulation/particle_dynamics/terramechanics.py"],
       ["bekker_wong", "terramechanics"], refs_in_code=["Wong 1978"], refs_added=[BEKKER, WONG]),
    _m("S16", "N-body (REBOUND)", [NS + "nbody_rebound.py"], ["nbody_rebound", "rebound"],
       refs_in_code=["Rein & Liu, REBOUND, Astron. Astrophys. 537 (2012) A128"]),
    _m("S17", "JAX-CFD", [NS + "jax_cfd_solver.py"], ["jax_cfd"],
       refs_in_code=["Dresdner et al., Learning to correct spectral methods, arXiv:2207.00556"],
       refs_added=["Kochkov et al., Machine learning accelerated CFD, PNAS 118(21) (2021), arXiv:2102.01010"]),
    _m("S18", "Particle dynamics: rigid body, MPM, molecular dynamics", ["pinneapple_simulation/particle_dynamics/"],
       ["particle_dynamics", "RigidBody", "mpm", "molecular_dynamics"],
       refs_in_code=["Stomakhin et al., A material point method for snow simulation, ACM TOG 32(4) (2013)",
                     "Hu et al., A moving least squares material point method, ACM TOG 37(4) (2018)",
                     "Swope, Andersen, Berens & Wilson, J. Chem. Phys. 76 (1982) 637 -- velocity Verlet"],
       refs_added=["Featherstone, Rigid Body Dynamics Algorithms, Springer, 2008"]),
    _m("S19", "Signal decomposition: CEEMDAN, EEMD, VMD, HHT, SSA, SST, STL, FFT, wavelet",
       [NS + f for f in ("ceemdan.py", "eemd.py", "vmd.py", "hilbert_huang.py", "ssa.py", "sst.py", "stl.py", "fft.py", "wavelet.py")],
       ["ceemdan", "eemd", "vmd", "hilbert_huang", "numerical_solvers.ssa", "numerical_solvers.sst",
        "numerical_solvers.stl", "numerical_solvers.fft", "wavelet"],
       refs_in_code=["Torres et al., CEEMDAN, ICASSP 2011", "Wu & Huang, EEMD, Adv. Adapt. Data Anal. 1(1) (2009)",
                     "Dragomiretskiy & Zosso, Variational Mode Decomposition, IEEE TSP 62(3) (2014)"],
       refs_added=["Huang et al., The empirical mode decomposition and the Hilbert spectrum, Proc. R. Soc. A 454 (1998) 903-995",
                   "Golyandina, Nekrutkin & Zhigljavsky, Analysis of Time Series Structure: SSA and Related Techniques, 2001",
                   "Daubechies, Lu & Wu, Synchrosqueezed wavelet transforms, Appl. Comput. Harmon. Anal. 30 (2011) 243-261",
                   "Cleveland et al., STL: a seasonal-trend decomposition procedure based on loess, J. Off. Stat. 6(1) (1990)",
                   "Cooley & Tukey, Math. Comput. 19 (1965) 297-301 -- FFT",
                   "Mallat, A Wavelet Tour of Signal Processing, 3rd ed., Academic Press, 2009"]),
    _m("S20", "Co-simulation (graph + engine, algebraic loops by Tarjan SCC)", ["pinneapple_systems/cosimulation/"],
       ["cosimulation", "CoSimGraph", "CoSimEngine"],
       refs_added=["Gomes et al., Co-simulation: a survey, ACM Comput. Surv. 51(3) (2018)",
                   "Tarjan, Depth-first search and linear graph algorithms, SIAM J. Comput. 1(2) (1972) 146-160"]),
    _m("S21", "Digital twin state estimation: KF/EKF/UKF/EnKF, prognostics, signal reconstruction",
       ["pinneapple_systems/digital_twin/", "pinneapple_neural/architectures/classical_ts/"],
       ["digital_twin", "classical_ts.ekf", "classical_ts.ukf", "classical_ts.enkf", "Kalman", "EnKF", "UKF"],
       refs_added=["Kalman, A new approach to linear filtering and prediction problems, J. Basic Eng. 82 (1960) 35-45",
                   "Julier & Uhlmann, Unscented filtering and nonlinear estimation, Proc. IEEE 92(3) (2004) 401-422",
                   "Evensen, Sequential data assimilation with a nonlinear QG model, J. Geophys. Res. 99(C5) (1994) 10143",
                   "Simon, Optimal State Estimation, Wiley, 2006"]),
    _m("S22", "1D process components (pipe network, transient pipe, heat exchanger, valve, reaction, polytropic path, "
              "compressor map)", [PC], ["process_components"],
       refs_in_code=["PennWell 2003 (compressor similarity map)"],
       refs_added=[INCROPERA, WHITE, "IEC 60534-2-1 / ANSI/ISA-75.01.01 -- control valve sizing",
                   "Schultz, The polytropic analysis of centrifugal compressors, J. Eng. Power 84(1) (1962) 69-82",
                   "Bell, Wronski, Quoilin & Lemort, CoolProp, Ind. Eng. Chem. Res. 53(6) (2014) 2498-2508"]),
    _m("S23", "Explicit equation-system solver (evaluation order)", [PC + "explicit_equation_system.py"],
       ["explicit_equation_system"],
       refs_added=["Tarjan, Depth-first search and linear graph algorithms, SIAM J. Comput. 1(2) (1972) 146-160"]),
    _m("S24", "External solvers: OpenFOAM, FEniCS, ANSYS/CFD formats, MATLAB, Modelica/FMI, MuJoCo, Genesis, TurboDesigner",
       ["pinneapple_simulation/external_solvers/"], ["external_solvers", "openfoam", "fenics"],
       refs_added=["Weller, Tabor, Jasak & Fureby, A tensorial approach to CFD, Comput. Phys. 12(6) (1998) 620-631 (OpenFOAM)",
                   "Baratta et al., DOLFINx: the next generation FEniCS problem solving environment, 2023",
                   "Blochwitz et al., The Functional Mockup Interface, Modelica Conference 2011",
                   "Todorov, Erez & Tassa, MuJoCo, IROS 2012"]),
    # ── B. Training ──────────────────────────────────────────────────────
    _m("T1", "Base trainer: AMP, gradient accumulation, auto batch size, CUDA graphs, profiler",
       [TR + "trainer.py", TR + "hpc.py"], ["trainer.trainer", "from pinneapple_neural.trainer import", "Trainer("],
       refs_added=["Micikevicius et al., Mixed Precision Training, ICLR 2018, arXiv:1710.03740",
                   "Kingma & Ba, Adam, ICLR 2015, arXiv:1412.6980"]),
    _m("T2", "Distributed: DDP, FSDP, SLURM, torchrun, PowerSGD", [TR + "distributed.py", TR + "hpc.py"],
       ["DDPPINNTrainer", "FSDPConfig", "PowerSGD", "trainer.distributed"],
       refs_in_code=["Vogels, Karimireddy & Jaggi, PowerSGD, NeurIPS 2019, arXiv:1905.13727"],
       refs_added=["Li et al., PyTorch Distributed, VLDB 2020, arXiv:2006.15704",
                   "Zhao et al., PyTorch FSDP, VLDB 2023, arXiv:2304.11277"]),
    _m("T3", "Causal training", [TR + "causal.py"], ["CausalPINNTrainer", "CausalWeightScheduler", "trainer.causal"],
       refs_in_code=["Wang, Sankaran & Perdikaris, Respecting causality is all you need for training PINNs, arXiv:2203.07404"]),
    _m("T4", "Time marching", [TR + "time_marching.py"], ["TimeMarchingTrainer", "time_marching"],
       refs_in_code=["Wight & Zhao, Solving Allen-Cahn and Cahn-Hilliard equations using adaptive PINNs, arXiv:2007.04542"]),
    _m("T5", "Two phases Adam -> L-BFGS + L-BFGS finetune", [TR + "two_phase.py", TR + "lbfgs_finetune.py"],
       ["TwoPhaseTrainer", "lbfgs_finetune", "two_phase"],
       refs_added=[RAISSI, "Liu & Nocedal, On the limited memory BFGS method, Math. Program. 45 (1989) 503-528"]),
    _m("T6", "Multi-restart", [TR + "multistart.py"], ["MultiRestartTrainer", "multistart"]),
    _m("T7", "Adaptive residual-based collocation", [TR + "collocation.py"], ["AdaptiveCollocation", "trainer.collocation"],
       refs_in_code=["Lu et al., DeepXDE, SIAM Rev. 63(1) (2021), arXiv:1907.04502",
                     "Wu et al., non-adaptive and residual-based adaptive sampling for PINNs, CMAME 2023, arXiv:2207.10289"]),
    _m("T8", "Loss balancing: ReLoBRaLo, SoftAdapt, PCGrad, Augmented Lagrangian, Inverse Dirichlet, AutoBalancer",
       [TR + "loss_balancer.py"], ["ReLoBRaLo", "SoftAdapt", "PCGrad", "AugmentedLagrangian", "InverseDirichlet",
                                   "AutoBalancer", "loss_balancer"],
       refs_in_code=["Bischof & Kraus, Multi-objective loss balancing for PINNs, arXiv:2110.09813",
                     "Heydari et al., SoftAdapt, arXiv:1912.12355",
                     "van der Meer, Oosterlee & Borovykh, Optimally weighted loss functions for PDEs, 2022"],
       refs_added=["Yu et al., Gradient Surgery for Multi-Task Learning (PCGrad), NeurIPS 2020, arXiv:2001.06782",
                   "Lu et al., PINNs with hard constraints for inverse design (augmented Lagrangian), arXiv:2102.04626",
                   "Maddu et al., Inverse-Dirichlet weighting, arXiv:2107.00940"]),
    _m("T9", "Weight schedulers: SA-PINN, GradNorm, LossRatio, NTK", [TR + "weight_scheduler.py"],
       ["SelfAdaptiveWeights", "GradNormBalancer", "LossRatioBalancer", "NTKWeightBalancer", "weight_scheduler"],
       refs_in_code=["McClenny & Braga-Neto, Self-Adaptive PINNs, arXiv:2009.04544",
                     "Chen et al., GradNorm, ICML 2018, arXiv:1711.02257",
                     "Wang, Yu & Perdikaris, When and why PINNs fail to train: an NTK perspective, arXiv:2007.14527"]),
    _m("T10", "Domain decomposition: XPINN, DoMINO/D3M", ["pinneapple_neural/architectures/pinns/xpinn.py",
                                                          "pinneapple_physics/pinn_solver/domino.py"],
       ["XPINN", "xpinn", "DoMINO", "pinn_solver.domino"],
       refs_in_code=["Li et al., D3M: a deep domain decomposition method for PDEs, arXiv:1909.12236"],
       refs_added=["Jagtap & Karniadakis, Extended PINNs (XPINNs), Commun. Comput. Phys. 28(5) (2020) 2002-2041"]),
    _m("T11", "Inverse PINN (learnable physical parameters)", [TR + "inverse_pinn.py"], ["InversePINN", "inverse_pinn"],
       refs_added=[RAISSI]),
    _m("T12", "Gray-box + symbolic regression (PySR)", [TR + "graybox.py"], ["GrayBox", "graybox"],
       refs_added=["Cranmer, Interpretable ML for science with PySR and SymbolicRegression.jl, arXiv:2305.01582"]),
    _m("T13", "Uncertainty: heteroscedastic, B-PINN, MC Dropout, deep ensembles",
       [TR + "heteroscedastic.py", "pinneapple_neural/architectures/pinns/bpinn.py", "pinneapple_systems/component_modeling/"],
       ["Heteroscedastic", "BayesianPINN", "bpinn", "mc_dropout", "MCDropout", "DeepEnsemble", "component_modeling"],
       refs_in_code=["Yang, Meng & Karniadakis, B-PINNs, J. Comput. Phys. 425 (2021), arXiv:2003.06097"],
       refs_added=["Gal & Ghahramani, Dropout as a Bayesian approximation, ICML 2016, arXiv:1506.02142",
                   "Lakshminarayanan, Pritzel & Blundell, Deep ensembles, NeurIPS 2017, arXiv:1612.01474",
                   "Kendall & Gal, What uncertainties do we need in Bayesian deep learning, NeurIPS 2017, arXiv:1703.04977"]),
    _m("T14", "SINDy", ["pinneapple_neural/architectures/rom/sindy.py"], ["SINDy", "rom.sindy"],
       refs_added=["Brunton, Proctor & Kutz, Discovering governing equations from data (SINDy), PNAS 113(15) (2016), "
                   "arXiv:1509.03580"]),
    _m("T15", "Adaptive hyperparameter sweep / architecture search (arena NAS)",
       [TR + "adaptive_sweep.py", "pinneapple_arena/nas.py"], ["adaptive_sweep", "AdaptiveSweep", "arena.nas", "arena_nas"]),
    _m("T16", "Noether trainer (research only)", [TR + "noether_trainer.py"], ["NoetherSurrogateTrainer", "noether_trainer"],
       refs_added=["Alkin et al., AB-UPT, arXiv:2502.09692"]),
    _m("T17", "Post-training advisor / audit", [TR + "advisor.py", TR + "audit.py"], ["trainer.advisor", "trainer.audit"]),
    _m("T18", "Meta-learning: MAML, Reptile", ["pinneapple_adaptation/meta_learning/"], ["MAML", "Reptile", "meta_learning"],
       refs_in_code=["Finn, Abbeel & Levine, MAML, ICML 2017, arXiv:1703.03400",
                     "Nichol, Achiam & Schulman, On first-order meta-learning algorithms (Reptile), arXiv:1803.02999"]),
    _m("T19", "Transfer learning (freeze, adapters, parametric)", ["pinneapple_adaptation/transfer_learning/"],
       ["transfer_learning"]),
    _m("T20", "Self-scaled quasi-Newton: BFGS / SSBFGS / SSBroyden (new, 2026-09-24)", [TR + "self_scaled_qn.py"],
       ["SelfScaledQuasiNewton", "self_scaled_qn"],
       refs_in_code=["Urban, Stefanou & Pons, Unveiling the optimization process of PINNs, J. Comput. Phys. 523 "
                     "(2025) 113656, arXiv:2405.04230 -- eqs. (7)-(23), Appendix B",
                     "Jnini, Kiyani, Shukla et al., Curvature-Aware Optimization for High-Accuracy PINNs, "
                     "arXiv:2604.05230 (benchmarks: Euler/HLLC, Helmholtz, inviscid Burgers, Stokes, PK-PD)",
                     "Wang, Teng & Perdikaris, gradient pathologies in PINNs, SIAM J. Sci. Comput. 43(5) (2021), "
                     "arXiv:2001.04536 -- Helmholtz benchmark"]),
    # ── C. Equations (PINN compiler kinds + closed forms) ────────────────
    _m("E1", "Laplace", [COMPILE], ['"laplace"', "laplace_2d"], refs_added=[EVANS]),
    _m("E2", "Poisson", [COMPILE], ['"poisson"', "poisson_2d"], refs_added=[EVANS]),
    _m("E3", "Helmholtz", [COMPILE], ['"helmholtz"', "helmholtz_acoustics"], refs_added=[EVANS]),
    _m("E4", "Heat (steady anisotropic / transient)", [COMPILE], ['"heat_equation', "transient_heat", "steady_heat"],
       refs_added=[INCROPERA, EVANS]),
    _m("E5", "Wave", [COMPILE], ['"wave_equation"', "wave_ultrasound"], refs_added=[EVANS]),
    _m("E6", "Advection-diffusion / convection", [COMPILE], ['"advection_diffusion"', '"convection"'], refs_added=[LEVEQUE_FD]),
    _m("E7", "Burgers (viscous / inviscid)", [COMPILE], ['"burgers"', '"inviscid_burgers"', "burgers_1d"],
       refs_added=["Burgers, A mathematical model illustrating the theory of turbulence, Adv. Appl. Mech. 1 (1948) 171-199"]),
    _m("E8", "Reaction-diffusion 1D/2D (Gray-Scott)", [COMPILE], ['"reaction_diffusion', "reaction_diffusion_2d"],
       refs_added=["Pearson, Complex patterns in a simple system, Science 261 (1993) 189-192"]),
    _m("E9", "Fisher-KPP", [COMPILE], ['"fisher_kpp"', "diffusion_reaction"],
       refs_added=["Fisher, The wave of advance of advantageous genes, Ann. Eugen. 7(4) (1937) 355-369",
                   "Ablowitz & Zeppetella, Bull. Math. Biol. 41(6) (1979) 835-840 -- exact travelling wave"]),
    _m("E10", "Incompressible Navier-Stokes 2D/3D", [COMPILE], ["navier_stokes_incompressible", "ns_incompressible",
                                                                 "lid_driven_cavity", "Ghia"],
       refs_added=[BATCHELOR, "Ghia, Ghia & Shin, J. Comput. Phys. 48 (1982) 387-411 -- lid-driven cavity benchmark"]),
    _m("E11", "Navier-Stokes in a rotating frame", [COMPILE], ["incompressible_navier_stokes_rotating_frame"],
       refs_added=[BATCHELOR]),
    _m("E12", "Stokes", [COMPILE], ['"stokes"'], refs_added=[BATCHELOR]),
    _m("E13", "Brinkman", [COMPILE], ['"brinkman"'],
       refs_added=["Brinkman, A calculation of the viscous force exerted by a flowing fluid on a dense swarm of "
                   "particles, Appl. Sci. Res. A1 (1949) 27-34"]),
    _m("E14", "Darcy", [COMPILE], ['"darcy"', "darcy_pressure_only", "pdebench_darcy"],
       refs_added=["Darcy, Les fontaines publiques de la ville de Dijon, 1856",
                   "Takamoto et al., PDEBench, arXiv:2210.07182 (Darcy 2D benchmark)"]),
    _m("E15", "Buckley-Leverett two-phase", [COMPILE], ["buckley_leverett"],
       refs_added=["Buckley & Leverett, Mechanism of fluid displacement in sands, Trans. AIME 146 (1942) 107-116"]),
    _m("E16", "Compressible Euler 1D/2D/axisymmetric/rotating 3D", [COMPILE], ["euler_compressible", "compressible_euler",
                                                                             "sod_shock_tube", "rocket_nozzle"],
       refs_added=[TORO, "Sod, J. Comput. Phys. 27 (1978) 1-31 -- shock tube"]),
    _m("E17", "Shallow water 1D/2D", [COMPILE], ["shallow_water"],
       refs_added=["Toro, Shock-Capturing Methods for Free-Surface Shallow Flows, Wiley, 2001"]),
    _m("E18", "Stommel gyre", [COMPILE], ["stommel", "climate_ocean_gyre"],
       refs_added=["Stommel, The westward intensification of wind-driven ocean currents, Trans. AGU 29(2) (1948) 202-206"]),
    _m("E19", "Plane-stress / plane-strain linear elasticity", [COMPILE], ["linear_elasticity_plane_stress", "plane_stress",
                                                                          "plane_strain", "linear_elasticity_3d"],
       refs_added=[TIMO_ELAST]),
    _m("E20", "Axisymmetric elasticity (+ torsion) / Lame", [COMPILE], ["axisymmetric_linear_elasticity",
                                                                         "thick_walled_cylinder_lame", "lame_hoop"],
       refs_added=[TIMO_ELAST]),
    _m("E21", "Neo-Hookean hyperelasticity", [COMPILE], ["hyperelasticity_neo_hookean"],
       refs_added=["Holzapfel, Nonlinear Solid Mechanics, Wiley, 2000"]),
    _m("E22", "Euler-Bernoulli beam", [COMPILE, PC + "beam_statics.py"], ['"euler_bernoulli_beam"', "beam_statics", "solve_beam"],
       refs_added=[TIMO_ELAST]),
    _m("E23", "von Karman beam / plate", [COMPILE, PC + "beam_nonlinear_fem.py"], ["von_karman", "nonlinear_beam"],
       refs_added=["Timoshenko & Woinowsky-Krieger, Theory of Plates and Shells, 2nd ed., McGraw-Hill, 1959"]),
    _m("E24", "Thermoelasticity", [COMPILE], ["thermoelasticity"],
       refs_added=["Boley & Weiner, Theory of Thermal Stresses, Wiley, 1960"]),
    _m("E25", "Phase-field fracture", [COMPILE], ["phase_field_fracture", "material_fracture"],
       refs_added=["Bourdin, Francfort & Marigo, J. Mech. Phys. Solids 48 (2000) 797-826",
                   "Miehe, Hofacker & Welschinger, Comput. Methods Appl. Mech. Eng. 199 (2010) 2765-2778"]),
    _m("E26", "Biot poroelasticity", [COMPILE], ["biot_poroelasticity"],
       refs_added=["Biot, General theory of three-dimensional consolidation, J. Appl. Phys. 12 (1941) 155-164"]),
    _m("E27", "Maxwell TE", [COMPILE], ["maxwell_te"], refs_added=[JACKSON]),
    _m("E28", "Bekker-Wong terramechanics", [COMPILE], ["bekker_wong_terramechanics", "bekker_wong_surrogate"],
       refs_added=[BEKKER, WONG]),
    _m("E29", "Compressor meanline 1D", [COMPILE], ["compressor_meanline", "axial_compressor"],
       refs_added=["Dixon & Hall, Fluid Mechanics and Thermodynamics of Turbomachinery, 7th ed., Elsevier, 2014"]),
    _m("E30", "Phonon Boltzmann transport 1D (gray)", [COMPILE], ["phonon_bte", "crystal_phonon"],
       refs_added=["Chen, Nanoscale Energy Transport and Conversion, Oxford University Press, 2005"]),
    _m("E31", "Black-Scholes", [COMPILE], ["black_scholes"],
       refs_added=["Black & Scholes, The pricing of options and corporate liabilities, J. Polit. Econ. 81(3) (1973) 637-654"]),
    _m("E32", "Heston", [COMPILE], ["heston"],
       refs_added=["Heston, A closed-form solution for options with stochastic volatility, Rev. Financ. Stud. 6(2) (1993) 327-343"]),
    _m("E33", "Opinion dynamics (continuum Hegselmann-Krause)", [COMPILE], ["opinion_dynamics"],
       refs_added=["Hegselmann & Krause, Opinion dynamics and bounded confidence, JASSS 5(3) (2002)"]),
    _m("E34", "Kepler two-body orbit", [COMPILE], ["kepler"], refs_added=[VALLADO]),
    _m("E35", "Circular restricted three-body problem (planar, synodic)", [COMPILE], ["cr3bp"],
       refs_added=["Szebehely, Theory of Orbits: The Restricted Problem of Three Bodies, Academic Press, 1967"]),
    _m("E36", "J2 perturbation", [COMPILE], ["j2_perturbation", "j2_secular"], refs_added=[VALLADO]),
    _m("E37", "Clohessy-Wiltshire relative motion", [COMPILE], ["cw_relative_motion", "space_debris"],
       refs_added=["Clohessy & Wiltshire, Terminal guidance system for satellite rendezvous, J. Aerosp. Sci. 27(9) (1960) 653-658"]),
    _m("E38", "Euler rigid-body attitude", [COMPILE], ["spacecraft_attitude"],
       refs_added=["Markley & Crassidis, Fundamentals of Spacecraft Attitude Determination and Control, Springer, 2014"]),
    _m("E39", "Lane-Emden polytrope", [COMPILE], ["lane_emden"],
       refs_added=["Chandrasekhar, An Introduction to the Study of Stellar Structure, 1939",
                   "Hansen, Kawaler & Trimble, Stellar Interiors, 2nd ed., Springer, 2004 (Table 4.1)"]),
    _m("E40", "Shakura-Sunyaev accretion disk", [COMPILE], ["shakura_sunyaev"],
       refs_added=["Shakura & Sunyaev, Black holes in binary systems, Astron. Astrophys. 24 (1973) 337-355"]),
    _m("E41", "Schwarzschild weak-field light bending", [COMPILE], ["schwarzschild"],
       refs_added=["Misner, Thorne & Wheeler, Gravitation, W. H. Freeman, 1973"]),
    _m("E42", "SIR epidemic", [COMPILE], ["sir_ode", "sir_epidemic"],
       refs_added=["Kermack & McKendrick, Proc. R. Soc. A 115 (1927) 700-721"]),
    _m("E43", "Pharmacokinetics two-compartment", [COMPILE], ["pk_two_compartment"],
       refs_added=["Gibaldi & Perrier, Pharmacokinetics, 2nd ed., Marcel Dekker, 1982"]),
    _m("E44", "Reaction network / Arrhenius / mass action", [COMPILE, PC + "reaction_kinetics.py"],
       ["reaction_kinetics", "arrhenius", "ReactionNetwork"],
       refs_added=["Fogler, Elements of Chemical Reaction Engineering, 5th ed., Prentice Hall, 2016"]),
    _m("E45", "Hagen-Poiseuille", ["PINNeAPPle-CFD/pinneapple_cfd/etapas/e02_modelo_matematico/correlacoes.py"],
       ["hagen_poiseuille"], refs_added=[WHITE]),
    _m("E46", "Blasius friction factor", ["PINNeAPPle-CFD/.../correlacoes.py"], ["f_blasius", "blasius"], refs_added=[WHITE]),
    _m("E47", "Colebrook-White", ["PINNeAPPle-CFD/.../correlacoes.py", PC + "pipe_network_1d.py"], ["colebrook"],
       refs_added=["Colebrook, Turbulent flow in pipes, J. Inst. Civ. Eng. 11 (1939) 133-156"]),
    _m("E48", "Darcy-Weisbach pressure drop", ["PINNeAPPle-CFD/.../correlacoes.py"], ["dp_darcy", "darcy_weisbach"],
       refs_added=[WHITE]),
    _m("E49", "Herschel-Bulkley + Metzner-Reed (non-Newtonian pipe flow)", [PC + "non_newtonian_pipe_flow.py"],
       ["non_newtonian_pipe_flow", "herschel_bulkley", "metzner_reed"],
       refs_added=["Herschel & Bulkley, Kolloid-Z. 39 (1926) 291-300",
                   "Metzner & Reed, Flow of non-Newtonian fluids, AIChE J. 1(4) (1955) 434-440"]),
    _m("E50", "DNV-RP-O501 erosion (straight pipe and bend)", ["PINNeAPPle-CFD/pinneapple_cfd/etapas/e02_modelo_matematico/erosao.py"],
       ["dnv_curva", "dnv_reto", "erosao"], refs_added=["DNV-RP-O501 rev. 4.2, Managing Sand Production and Erosion, 2007/2011"]),
    _m("E51", "Lame thick-walled cylinder", [PC + "pipe_stress_mechanics.py"], ["lame_hoop_stress", "thick_walled_cylinder"],
       refs_added=[TIMO_ELAST]),
    _m("E52", "von Mises equivalent stress", [PC + "pipe_stress_mechanics.py", PC + "beam_statics.py"],
       ["von_mises"], refs_added=["von Mises, Goettinger Nachrichten, Math.-Phys. Kl. (1913) 582-592", SHIGLEY]),
    _m("E53", "Euler buckling + constrained rod buckling", [PC + "pipe_stress_mechanics.py"],
       ["euler_critical_buckling", "constrained_rod_buckling"],
       refs_in_code=["Paslay & Dawson 1964; Mitchell 1988 -- constrained rod buckling"], refs_added=[TIMO_STAB]),
    _m("E54", "Beam-column moment amplification factor", [PC + "pipe_stress_mechanics.py"],
       ["moment_amplification"], refs_added=[TIMO_STAB]),
    _m("E55", "S-N curve + Goodman + Miner", [PC + "fatigue_analysis.py", CF + "cantilever_fatigue.py"],
       ["miners_rule", "goodman", "sn_curve", "fatigue"],
       refs_added=["Miner, Cumulative damage in fatigue, J. Appl. Mech. 12 (1945) A159-A164", SHIGLEY]),
    _m("E56", "Torsional stick-slip with Stribeck friction", [PC + "torsional_stickslip.py"],
       ["torsional_stickslip", "stribeck"],
       refs_added=["Armstrong-Helouvry, Dupont & Canudas de Wit, A survey of models, analysis tools and compensation "
                   "methods for the control of machines with friction, Automatica 30(7) (1994) 1083-1138"]),
    _m("E57", "Heat exchanger (effectiveness-NTU)", [PC + "heat_exchanger.py"], ["heat_exchanger", "HeatExchanger"],
       refs_added=[INCROPERA]),
    _m("E58", "Fin array conduction", [CF + "fin_array_conduction.py"], ["fin_array"], refs_added=[INCROPERA]),
    _m("E59", "Control valve (Cv, compressible / incompressible)", [PC + "control_valve.py"], ["control_valve", "installed_cv"],
       refs_added=["IEC 60534-2-1 / ANSI/ISA-75.01.01 -- flow equations for sizing control valves"]),
    _m("E60", "Polytropic compression path", [PC + "polytropic_path.py"], ["polytropic_path"],
       refs_added=["Schultz, The polytropic analysis of centrifugal compressors, J. Eng. Power 84(1) (1962) 69-82"]),
    _m("E61", "Real gas equation of state (CoolProp)", [PC + "real_gas_eos.py"], ["real_gas_eos", "GasState"],
       refs_added=["Bell, Wronski, Quoilin & Lemort, CoolProp, Ind. Eng. Chem. Res. 53(6) (2014) 2498-2508"]),
    _m("E62", "Compressor similarity map", [PC + "similarity_map.py"], ["similarity_map", "evaluate_map"],
       refs_in_code=["PennWell 2003"], refs_added=["Dixon & Hall, Fluid Mechanics and Thermodynamics of Turbomachinery, 2014"]),
    _m("E63", "Helmholtz resonator", [CF + "helmholtz_resonator.py"], ["helmholtz_resonator"],
       refs_added=["Kinsler, Frey, Coppens & Sanders, Fundamentals of Acoustics, 4th ed., Wiley, 2000"]),
    _m("E64", "Membrane diffusion", [CF + "membrane_diffusion.py"], ["membrane_diffusion"],
       refs_added=["Crank, The Mathematics of Diffusion, 2nd ed., Oxford University Press, 1975"]),
    _m("E65", "Cantilever fatigue", [CF + "cantilever_fatigue.py"], ["cantilever_fatigue"], refs_added=[SHIGLEY]),
]

# ── D. Problems: the 65 registered presets, grouped; each points at its equations ──
_PRESET_GROUPS = {
    "P1": ("Academic", {
        "burgers_1d": ["E7"], "laplace_2d": ["E1"], "poisson_2d": ["E2"], "reaction_diffusion_2d": ["E8"],
        "ns_incompressible_2d": ["E10"], "lid_driven_cavity_3d": ["E10"], "channel_flow_3d": ["E10"],
        "pipe_flow_3d": ["E10"], "darcy_pressure_only_3d": ["E14"], "helmholtz_acoustics_3d": ["E3"],
        "wave_ultrasound_3d": ["E5"], "steady_heat_conduction_3d": ["E4"], "transient_heat_3d": ["E4"]}),
    "P2": ("Solids", {
        "plane_stress_2d": ["E19"], "plane_strain_2d": ["E19"], "linear_elasticity_3d": ["E19"],
        "linear_elasticity_3d_industry": ["E19"], "axisymmetric_linear_elasticity_2d": ["E20"],
        "thick_walled_cylinder_lame": ["E20"], "von_mises_2d": ["E19", "E52"], "thermoelasticity_2d": ["E24"],
        "material_fracture_2d": ["E25"], "rotary_coupling_torsion": ["E20"], "threaded_coupling_tc50_box": ["E20"],
        "threaded_coupling_tc50_pin": ["E20"], "threaded_coupling_tc50_rotating": ["E20"]}),
    "P3": ("Aero / automotive", {
        "aircraft_wing_aerodynamics": ["E10"], "aircraft_wing_structural": ["E19"], "rocket_nozzle_cfd": ["E16"],
        "rocket_structural": ["E19"], "car_external_aero": ["E10"], "car_brake_thermal": ["E4"],
        "car_suspension_fatigue": ["E19", "E55"]}),
    "P4": ("Turbomachinery", {
        "axial_compressor_meanline": ["E29"], "axial_compressor_cascade_2d": ["E16"],
        "axial_compressor_stage_3d": ["E16"], "fan_cooler_cfd": ["E10"]}),
    "P5": ("Industrial thermal", {
        "datacenter_airflow_2d": ["E10"], "datacenter_cfd_3d": ["E10"], "datacenter_server_thermal": ["E4"],
        "cpu_heatsink_thermal": ["E4"], "pcb_thermal": ["E4"], "industrial_furnace_thermal": ["E4"],
        "furnace_combustion_zone": ["E6", "E44"], "refractory_lining": ["E4"]}),
    "P6": ("Multidisciplinary", {
        "climate_atmosphere_2d": ["E6"], "climate_ocean_gyre": ["E18"], "black_scholes_1d": ["E31"],
        "heston_pde_2d": ["E32"], "sir_epidemic": ["E42"], "pk_two_compartment": ["E43"],
        "drug_diffusion_tissue": ["E6"], "opinion_dynamics_2d": ["E33"], "crystal_phonon": ["E30"]}),
    "P7": ("Astro / space", {
        "kepler_two_body_orbit": ["E34"], "cr3bp_planar_synodic": ["E35"], "satellite_j2_perturbation": ["E36"],
        "space_debris_cw_relative_motion": ["E37"], "spacecraft_attitude_euler_rotation": ["E38"],
        "lane_emden_polytrope": ["E39"], "shakura_sunyaev_accretion_disk": ["E40"],
        "schwarzschild_light_bending_weak_field": ["E41"], "nfw_dark_matter_potential": ["E2"],
        "sod_shock_tube_astro": ["E16"]}),
    "P8": ("Terramechanics", {"bekker_wong_surrogate_2d": ["E28"]}),
}
for _pid, (_group, _presets) in _PRESET_GROUPS.items():
    for _i, (_preset, _eqs) in enumerate(sorted(_presets.items()), start=1):
        METHODS.append(_m(f"{_pid}.{_i}", f"{_group}: {_preset}",
                          ["pinneapple_physics/pde_environment/presets/"], [_preset], equations=_eqs))
METHODS.append(_m("P9", "PINNeAPPle-CFD v1: pipes + particle erosion (CFD-04 pilot)",
                  ["PINNeAPPle-CFD/pinneapple_cfd/"], ["CFD-04"], equations=["E10", "E45", "E46", "E47", "E48", "E50"]))

_BY_ID: Dict[str, Method] = {}
for _meth in METHODS:
    if _meth.id in _BY_ID:
        raise ValueError(f"duplicate method id: {_meth.id}")
    _BY_ID[_meth.id] = _meth

_STATUS_PATH = os.path.join(os.path.dirname(__file__), "method_status.json")


def get_method(method_id: str) -> Method:
    try:
        return _BY_ID[method_id]
    except KeyError:
        raise KeyError(f"unknown method '{method_id}'") from None


def list_methods(category: Optional[str] = None) -> List[Method]:
    if category is None:
        return list(METHODS)
    return [m for m in METHODS if m.category == category]


def method_status() -> Dict[str, Dict]:
    """``{id: {"status": ..., "reference_tests": [...], "other_tests": [...]}}`` from the last scan."""
    try:
        with open(_STATUS_PATH) as f:
            return json.load(f)["methods"]
    except FileNotFoundError:
        return {}


def validated_methods(category: Optional[str] = None) -> List[Method]:
    """The official list: only items whose last scan found a reference-comparison test."""
    status = method_status()
    return [m for m in list_methods(category) if status.get(m.id, {}).get("status") == "validated"]
