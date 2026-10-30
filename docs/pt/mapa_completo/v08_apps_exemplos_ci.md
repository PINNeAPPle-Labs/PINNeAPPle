# Volume 8 — Apps, exemplos, reproduções, scripts e CI

> Parte do [Mapa completo do PINNeAPPle](../mapa_completo.md). Gerado por `docs/pt/_mapa/gerar.py` a partir do inventário do commit `e53cdcfb04`.

O que cada app, exemplo, reprodução de paper, script, benchmark e workflow de CI faz. A coluna *testes* conta funções de teste em arquivos que citam o caminho.

## 1. Apps (`apps/`)

64 itens.

| item | o que faz | arquivos .py | testes |
|---|---|---:|---:|
| `apps/_shared` | apps/_shared: what every app gets: `appkit.py` is the glue every app under `apps/` imports (`apps/_shared` is on `PYTHONPATH` in each Dockerfile). The logic behind it lives in the library, with tests: | 1 | 23 |
| `apps/aero_optimizer` | Aircraft Design Optimizer: **A single-aisle airliner (A320 / 737 class), searched by a genetic algorithm, checked in OpenFOAM 3D, shown with its flow fields.** | 9 | 0 |
| `apps/anomaly_detector` | Physics Anomaly Detector: **Detect abnormal behaviour with data *and* physical knowledge: which balance broke, by how much, and which sensors are involved.** | 2 | 4 |
| `apps/benchmark_arena` | Physics AI Benchmark Arena (day 22): **Live:** https://arena.pinneapple.org · **Port:** 8105 · **Prefix:** `ARN` | 3 | 5 |
| `apps/brakecool_ai` | BrakeCool AI: Inverse design of the radial vanes of a vented brake rotor (vane count, vent gap, vane thickness) under a sustained heat load, through a frozen neural surrogate re-checked against the physics, plus a Monte Carlo of machining tolerances. Imported from PINNeAPPle-apps (`brakecool_ai`); the logic lives in `pinneapple_design.surrogate_inverse.brake_rotor`, this folder holds routes and the page. | 2 | 5 |
| `apps/calibration_lab` | Model Calibration Lab (day 18): **Fit model parameters to real measurements, with intervals only where the data determine them, checked on measurements the fit never saw.** Live at <https://calibration.pinneapple.org> (port 8101). | 2 | 4 |
| `apps/control_lab` | Physics-Based Control Lab: **Use physical models to compute better control actions, and compare them fairly with what runs today.** | 2 | 4 |
| `apps/data_health` | Engineering Data Health: **Upload a dataset, get a physics-aware quality report.** | 2 | 0 |
| `apps/data_standardizer` | Engineering Data Standardizer: **Many files, many conventions, one physical schema.** | 2 | 0 |
| `apps/deploy` | Deploy the apps on one server: HeatSink Sizer, PCB Hotspot, Engineering Data Health, Engineering Data Standardizer, Simulation Metadata, the Form Compiler, Inverse Heat Lab, Simulation Preflight, Mesh Quality, Simulation Comparator, Engineering Model Lineage and the Simulation Interoperability Hub, the Aircraft Design Optimizer, the PINNeAPPle Observatory, the Inverse Model Studio, the Parameter Identifiability Lab, the Model Calibration Lab, the Engineering Evidence Report, the Physics Validation Lab and the PINNeAPPle Lab … | 0 | 9 |
| `apps/design_optimization` | Design Optimization: **A wing section redesigned live: less profile drag at the same lift, every step re-checked on the physics.** | 2 | 2 |
| `apps/design_requirements` | Form Compiler: **Pick the document format, drop the datasheets and specifications, get the form or datasheet compiled, with the source of every value and a list of what is missing or contradictory.** | 3 | 0 |
| `apps/engineering_optimizer` | Engineering Design Optimizer: **Better designs within physical and manufacturing limits, found by black-box optimisation on a fast model and confirmed by a high-fidelity one.** Day 17 of the PINNeAPPle app series; live at `https://optimize.pinneapple.org`. | 2 | 3 |
| `apps/evidence_report` | Engineering Evidence Report: **Turn experiments and simulations into reproducible technical evidence. A test is passed only if it ran.** | 2 | 5 |
| `apps/flow_studio` | PINNeAPPle Flow Studio: A node-based canvas for engineering flows, in the style of n8n: describe a problem, get one or more proposed flows, edit the equations and the methods, run, and click any node to see what it produced (metrics, logs, figures, a three.js 3-D view, the network architecture, tables). Everything a node does is a call into the PINNeAPPle library (`pinneapple_flow` plus the existing packages); this folder is only the HTTP shell and the React frontend. The same flows run without the GUI: `python -m … | 1 | 19 |
| `apps/flowtrace` | FlowTrace: **Upload a video or an image sequence of a seeded flow; get a cross-correlation PIV velocity field back. Any camera.** | 2 | 3 |
| `apps/governed_card` | GovernedCard: **A model card that cannot be issued without a physics check of the model behind it.** | 2 | 3 |
| `apps/guiagent` | Engineering GUI Agent: **guiagent.pinneapple.org** · port 8115 · day 27 extra · library: `pinneapple_tools.automation` ([guide](../../docs/guides/engineering_gui_agent.md)) | 2 | 2 |
| `apps/heatsink_inverse_ai` | HeatSink Design AI: Gradient-based inverse design of a straight-fin CPU heat sink (fin count, height, thickness on a fixed 80 x 80 mm aluminium base) through a frozen neural surrogate, with every answer re-checked against the physics it learned. Imported from PINNeAPPle-apps (`heatsink_design_ai`); the logic now lives in `pinneapple_design.surrogate_inverse.heatsink_fins` and this folder holds only routes and the page. | 2 | 5 |
| `apps/heatsink_sizer` | HeatSink Sizer: **Plate-fin heat sinks, sized and verified by physics — in seconds.** | 7 | 0 |
| `apps/helio_tfc` | HelioTFC: **Where is the solar cycle's envelope heading? EMD of the daily sunspot number, the Hilbert amplitude of the ~11-year component, and a fit whose end values are tied exactly.** | 2 | 4 |
| `apps/hub` | PINNeAPPle Hub: The catalogue of ready-made tools, trained models, geometries, datasets, flows and examples of PINNeAPPle: search, see an item with its recorded run, copy its code, download its files, open it in Flow Studio, run a zoo model in the browser or on the server, and compose your own tool from existing blocks. Docs: [docs/apps/hub.md](../../docs/apps/hub.md). | 2 | 14 |
| `apps/identifiability_lab` | Parameter Identifiability Lab (day 25): **Before estimating parameters, check that the measurements can determine them.** Live at | 2 | 4 |
| `apps/interop_hub` | Simulation Interoperability Hub: **Upload your simulation result. Get it in a neutral engineering representation.** | 2 | 0 |
| `apps/inverse_heat` | Inverse Heat Lab: **The convection coefficient h from a few thermocouples, with a physics-informed neural network, in 1D, 2D and 3D, and every answer checked against an independent solution.** | 7 | 0 |
| `apps/inverse_studio` | Inverse Model Studio (day 21): **From measurements back to the unknown parameters and fields: one problem, five methods, the same data, checked.** Live at <https://inverse-studio.pinneapple.org> (port 8104). This is a separate app from the [Inverse Heat Lab](https://inverse.pinneapple.org) (`apps/inverse_heat`), which is unchanged. The studio reuses that lab's physics (the fin and plate physics-informed networks and the finite-volume references) through `pinneapple_analysis.inverse_problems`, and links to it. | 2 | 3 |
| `apps/mesh_quality` | Mesh Quality: **Where your mesh is bad, how bad, and what to do about it, before the solver finds out.** | 2 | 0 |
| `apps/meshbridge` | MeshBridge: **Upload a CFD / FEM / CAD mesh file; get its points and fields as NPZ, or a repaired triangle surface.** | 2 | 4 |
| `apps/meshless_lab` | Meshless Method Lab: **Meshless or mesh-based: measure where each wins on your problem.** Day 26 of the PINNeAPPle apps program; live at | 2 | 5 |
| `apps/model_benchmarking` | Model Benchmarking: **Four PINN architectures trained live on the same viscous Burgers problem, ranked by accuracy and by a check on the PDE residual.** | 2 | 2 |
| `apps/model_gateway` | Physics AI Model Gateway (day 31): **Live:** https://models.pinneapple.org · **Port:** 8114 · **Prefix:** `MDL` | 2 | 6 |
| `apps/model_lineage` | Engineering Model Lineage: **How was this result produced? The digital thread from CAD to prediction.** | 2 | 0 |
| `apps/nozzle_screen` | NozzleScreen: **Is this rocket-nozzle idea in the right ballpark? An ideal-nozzle estimate in microseconds, and a PINN cross-check that says when the two disagree.** | 2 | 2 |
| `apps/observatory` | PINNeAPPle Observatory: **The sky and the solar system as they are right now, from where you stand, in 3-D.** | 3 | 0 |
| `apps/orbitwatch` | OrbitWatch: **What changed at this place between two satellite images? Per-band reflectance change and NDVI change, as a self-contained HTML report.** | 2 | 5 |
| `apps/parameter_discovery` | Parameter Discovery: **Can a physics-informed network recover a hidden material property from a few noisy readings and the PDE alone?** | 2 | 2 |
| `apps/patchdose_ai` | PatchDose AI: Smallest membrane-controlled release patch (area, membrane thickness, reservoir concentration) for a target release rate and wear time, found through a frozen neural surrogate and re-checked against Fick's law, plus the release curve over the wear time. Imported from PINNeAPPle-apps (`patchdose_ai`); the logic lives in `pinneapple_design.surrogate_inverse.transdermal_patch`. | 2 | 4 |
| `apps/pcb_hotspot` | PCB Hotspot: **Board-level PCB thermal analysis: junction temperatures, hotspots and what fixes them, in about a second.** | 4 | 0 |
| `apps/physics_copilot` | PhysicsCopilot: **A physics problem in plain language -> a trained PINN and the guardrail's re-computed report of it.** | 2 | 3 |
| `apps/physics_design` | Physics Design Optimizer: **Optimise parametric physical designs with explicit objectives, constraints and physics models, gradient-aware: compare gradient, evolutionary and Bayesian methods on the same problem and budget, and see which candidates the high-fidelity model confirms.** Day 23 of the PINNeAPPle app series; live at `https://design.pinneapple.org`. | 2 | 2 |
| `apps/pinn_arena` | PINNeAPPle Arena: **Port:** 8130 · **Prefix:** `PINN_ARENA` · **Service:** `pinnarena` (apps/deploy) | 21 | 25 |
| `apps/playground` | PINNeAPPle Playground: Draw an airfoil, size a heat sink or drive a lunar-rover wheel: a PINNeAPPle zoo model answers in the browser, and the physics checks and the trust verdict sit next to every number. It is a static site, so nothing is installed and no server runs any code. The state lives in the URL. | 1 | 2 |
| `apps/predictive_twin` | Predictive Twin: **A digital twin of a real river gauge: calibrated on its own recent past, forecasting the present, flagging when the river stops following the forecast.** | 2 | 3 |
| `apps/ridetune_ai` | RideTune AI: Lighter cross-section (width x thickness) of a suspension arm modelled as a rectangular cantilever, for a fatigue life target, found through a frozen neural surrogate of the bending stress and re-checked against the closed form, plus local sensitivities of life and mass. Imported from PINNeAPPle-apps (`ridetune_ai`); the logic lives in `pinneapple_design.surrogate_inverse.suspension_arm`. | 2 | 4 |
| `apps/rom_lab` | Reduced Order Model Lab: **Cheaper models of your simulations, with the error on held-out data in sight.** Day 19 of the PINNeAPPle apps program; live at <https://rom.pinneapple.org> (port 8102). | 3 | 4 |
| `apps/rover_mobility` | RoverMobility: **Rigid-wheel mobility screening: drawbar pull against slip from a Bekker-Wong PINN, with each physics constraint re-checked on its own.** | 2 | 2 |
| `apps/satscreen` | SatScreen: **Quick conjunction screening: closest approach and a collision-probability estimate in Clohessy-Wiltshire relative motion, plus two-body + J2 propagation.** | 2 | 3 |
| `apps/sensitivity_lab` | Engineering Sensitivity Lab: **Which parameters really control the result, and which can you freeze?** | 2 | 7 |
| `apps/simulation_campaign` | Simulation Campaign Planner: **Which simulations really need to run, in which order, and the case files to run them.** | 2 | 8 |
| `apps/simulation_comparator` | Simulation Comparator: **How far is this result from the reference, field by field, and where.** | 2 | 0 |
| `apps/simulation_metadata` | Simulation Metadata API: **Drop a simulation, get its solver, mesh, boundary conditions, parameters and convergence verdict.** | 2 | 0 |
| `apps/simulation_preflight` | Simulation Preflight: **Before spending hours of solver time, find out whether your simulation is set up correctly.** | 2 | 5 |
| `apps/simulation_runner` | Simulation Workflow Runner (day 27): **runner.pinneapple.org** · port 8110 · `RUNNER_*` environment | 2 | 4 |
| `apps/soundshape_ai` | SoundShape AI: Bass-reflex port radius and length for a target tuning frequency, found through a frozen neural surrogate and re-checked against the Helmholtz formula, plus a sweep of the trade-off between a small port and an accurate tuning. Imported from PINNeAPPle-apps (`soundshape_ai`); the logic lives in `pinneapple_design.surrogate_inverse.reflex_port`. | 2 | 4 |
| `apps/surrogate_speed` | Surrogate Speed: **A trained surrogate answers in microseconds; how much faster is that than a numerical solve, and how far apart are they?** | 2 | 2 |
| `apps/surrogate_studio` | Surrogate Studio (day 14): **Live:** https://surrogate.pinneapple.org · **Port:** 8097 · **Prefix:** `SUR` | 2 | 4 |
| `apps/text2part` | Text2Part: **Describe a part in plain language; get a mesh (STL) or a parametric CAD file (STEP) and its measured size.** | 2 | 4 |
| `apps/twin_ops` | Twin Ops — a commissioned digital twin, end to end: **HX-201, a fouled shell-and-tube exchanger.** The example exists to show the whole product, not a plot: a signed contract, a preflight that can refuse, a held-out backtest, an evidence file, and a drift ladder for when the plant moves away from the model. | 1 | 24 |
| `apps/twin_runtime` | Digital Twin Runtime: **Put a validated physical model or surrogate to work next to operational data, and know when to stop trusting it.** | 2 | 6 |
| `apps/uncertainty_lab` | Engineering Uncertainty Lab: **Know when a prediction is reliable, and when the case must go to the high-fidelity solver.** | 2 | 8 |
| `apps/validation_lab` | Physics Validation Lab: **Is this model or result physically and numerically consistent? Every check, with the record that it ran.** | 3 | 5 |
| `apps/verification_gap` | Verification Gap: **A design that passed its nominal safety check, and what its manufacturing tolerances do to that pass.** | 2 | 2 |
| `apps/verified_physics` | VerifiedPhysics: **The LLM proposes the physics; PINNeAPPle executes it and re-checks it mechanically, with the evidence attached.** | 2 | 5 |
| `apps/zoo_space` |  | 0 | 0 |

## 2. Exemplos (`examples/`)

88 itens.

| item | o que faz | arquivos .py | testes |
|---|---|---:|---:|
| `examples/README.md` | # Pinneapple Examples | 0 | 0 |
| `examples/architectures` | Pinneapple Models — Showcase Examples: These examples are **curated demos** focused on the full breadth of `pinneapple_models`: registry/catalog usage, neural operators, PINNs (including inverse parameters), ROMs, GNNs, and time-series transformers. | 12 | 0 |
| `examples/arena_pipelines` | 00_list_datasets.py — List and inspect all available PINNeAPPle datasets. | 10 | 0 |
| `examples/autolab` | The self-optimizing engineer (PINNeAPPle Autonomous Physics Lab, demo 2: thermal). | 1 | 0 |
| `examples/autolab_aero` | The AI Aerodynamicist (PINNeAPPle Autonomous Physics Lab, stage 1: the scientific cycle without an agent). | 1 | 0 |
| `examples/autolab_aero_agent` | The AI Aerodynamicist, stage 2: an agent reads the mission and runs the lab only through controlled tools. | 1 | 0 |
| `examples/autolab_digital_twin` | The digital twin that knows when it is wrong (PINNeAPPle Autonomous Physics Lab, demo 3). | 1 | 0 |
| `examples/autolab_rover` | PINNeAPPle Autonomous Physics Lab, demo 4: the rover that learns the planet. | 1 | 0 |
| `examples/benchmark` | An equal-budget benchmark in one call: PINN vs FNO vs DeepONet on 1-D Burgers. | 1 | 0 |
| `examples/benchmark_suite` | PINNeAPPle Arena — Examples: These examples are meant to **show the full potential of `pinneapple_arena`**: | 20 | 0 |
| `examples/black_hole_weather` | Black-hole weather: forecasting the gas around a black hole, and knowing for how long to trust it: This is a reproduction and extension of **Duarte, Nemmen & Navarro (2022)**, *Black hole weather forecasting with deep learning: a pilot study*: - MNRAS 512, 5848; [arXiv:2102.06242](https://arxiv.org/abs/2102.06242); - original code (MIT): [black-hole-group/DL_BH_fluids](https://github.com/black-hole-group/DL_BH_fluids). | 5 | 20 |
| `examples/bricks` | Self-modifying interlocking-brick builds: simulate, judge by physics, redesign, repeat. | 1 | 20 |
| `examples/calculix_cantilever` | Viga em balanço (CalculiX): Execução de referência do solver CalculiX (`ccx`) de uma viga em balanço de 1,0 × 0,1 × 0,1 (m). | 0 | 0 |
| `examples/calculix_pyvista_cantilever.py` | CalculiX cantilever -> PyVista image: von Mises stress on the deformed beam. | 1 | 0 |
| `examples/chain_of_whys` | Chain of whys: four projects, four scales, the same reasoning (pinneapple_reason). | 1 | 0 |
| `examples/condensation_falsification` | Condensation falsification campaign: the measured runs of docs/research/condensation_falsification.md. | 1 | 0 |
| `examples/core_primitives` | Poisson PINN on the core primitives (Domain, Geometry, FunctionField, Mesh, Field). | 2 | 0 |
| `examples/data_api` | A PINN, a neural operator and an inverse problem trained from the same loader API. | 1 | 0 |
| `examples/data_pipeline` | 03_prefetch_streaming_gpu.py | 8 | 0 |
| `examples/design_from_intent` | Design from intent: five requests through the same pipeline. | 1 | 0 |
| `examples/electrodynamics` | Electrodynamics & Electromagnetics Examples: Physics-informed neural network (PINN) solutions to classical electrostatics, magnetostatics and electromagnetics problems. | 17 | 20 |
| `examples/end_to_end` |  | 6 | 0 |
| `examples/enso_forecasting` | El Nino (ENSO) prediction: data-driven, physical, physics-informed and hybrid models under one honest protocol. | 1 | 20 |
| `examples/first_example` | Your first PINNeAPPle result in about a minute on a laptop CPU: solve Burgers' equation with a physics-informed network, check it against the exact solution and plot both. | 1 | 0 |
| `examples/flight_dynamics` | Verification of the flight-dynamics library: turbulence spectra and variances against MIL-F-8785C, trim of the four aircraft held for 10 s stick fixed, energy / angular-momentum conservation of the integrator, the lattice-in-the-loop lift against the steady vortex lattice. Writes … | 8 | 20 |
| `examples/flow_studio` | Write the example flows of Flow Studio into ``pinneapple_flow/flows/`` (the flows the GUI's "Open..." lists). | 3 | 0 |
| `examples/free_surface` | Free-surface flow (interFoam, VOF): `01_runner_slope_study.py`: a tapping jet falling into a melt runner held by a weir; the floor slope (+2°, 0°, -2°) changes the depth of metal under the jet. 2-D sections with `pp.cfd.FreeSurfaceFlow`, k sampled 5 cm above the floor while the solver runs and averaged in time. Needs OpenFOAM. Results: `docs/core_concepts/free_surface.md`. | 1 | 0 |
| `examples/freecad_openfoam` | FreeCAD -> OpenFOAM pipelines: three validated worked studies, a parameter sweep and a surrogate. | 1 | 0 |
| `examples/geomechanics` | Geomechanics with the material point method: `01_granular_column_collapse.py`: sand columns of aspect ratio 0.5 to 3 collapse on a rough floor; runout against Lube et al. (2005); frames written in GeoTaichi's layout and read back through the GeoTaichi bridge. About 6 minutes on one core. Results and limits: `docs/core_concepts/geomechanics_mpm.md`. | 1 | 0 |
| `examples/geometry` | pinneapple_geom examples: This folder focuses on geometry + meshing workflows that typically appear in Physics-AI / PINN pipelines: | 8 | 0 |
| `examples/geometry_library` | Wind on the CAARC standard tall building with OpenFOAM, through the generic STL path of ``pp.cfd``. | 3 | 0 |
| `examples/geophysics` | Earth-system blocks in one script (about one minute on a laptop CPU): | 1 | 0 |
| `examples/getting_started` | Getting Started Examples — Tier 1: Explorer: These examples are your **first step into Physics AI**. Each one: | 10 | 0 |
| `examples/handoff_cases` | Engineering hand-off cases: Two self-contained case packages for testing an engineering-simulation agent (or an engineer) from a cold start. Each folder holds the task (`brief.md`), the geometry, the simulation set-up and scripts, results that were really computed, and verified references; `<case>.zip` is the same folder as one archive. | 13 | 0 |
| `examples/hpo_experiments` | pinneapple_researcher examples: These examples are designed to show the full workflow: | 8 | 0 |
| `examples/industrial_presets` | Presets industriais: Problemas de engenharia já montados, resolvidos e validados com o PINNeAPPle. Cada pasta tem a ficha (entradas SI com a fonte de cada valor padrão), os resultados rodados aqui, a validação com o erro medido e as figuras. Regenere com `python -m pinneapple_presets report all`. | 0 | 0 |
| `examples/inspection` | pinneapple_analysis.inspection example 01: eddy-current NDE end to end. | 1 | 0 |
| `examples/latent` | Latent surrogate with a trust card: error, speed and calibration against the full solver. | 1 | 0 |
| `examples/lbm` | 3-D lattice Boltzmann wind tunnel: drag on a sphere against Stokes' law and the Schiller-Naumann correlation, in 32-bit and 16-bit storage of the distributions. | 1 | 0 |
| `examples/meshgraphnet` | MeshGraphNet in PINNeAPPle: Implementation: - `pinneapple_neural/architectures/graphnn/mesh_graph_net.py`: the encoder, processor and decoder network, registered as `mgn` / `meshgraphnet`. - `mgn_dynamics.py`: the transient training recipe of Pfaff et al. It covers normalization, training noise, the Δv target, and autoregressive rollout with boundary nodes held fixed. | 6 | 6 |
| `examples/model_zoo` | Model zoo quickstart: load a pre-trained model, predict, compare with the reference, validate, benchmark and fine-tune -- every step on the shipped, verified airfoil surrogate and its dataset. | 1 | 0 |
| `examples/modelica` | Modelica models as Physics AI data: Models: `models/DrainingTank.mo`, `models/LadleCooling.mo`. OpenModelica is used from the PATH or from Docker (`docker pull openmodelica/openmodelica:v1.24.4-minimal`). Results and discussion: `docs/core_concepts/modelica.md`. | 4 | 6 |
| `examples/molecular` | A protein on a gold surface: orient, build the Au(111) slab, run a short OpenMM MD and visualise it. | 1 | 0 |
| `examples/motor_magnetics` | Surface-PM machine (4 poles, 24 slots, nonlinear iron): cogging torque, torque vs current angle, back-EMF and losses with the 2D A_z FEM of pinneapple_simulation.electromagnetics. | 1 | 0 |
| `examples/near_random_forecasting` | Visual pack of the near-random forecasting study: four Blender Cycles renders and two 2-D figures. | 2 | 20 |
| `examples/non_hermitian` | Exceptional-point tuning in PT-symmetric dimers: the no-go theorem (Zenodo 23287622) checked symbolically and numerically, the three routes around it, the hypothesis map as a physics audit, and the figure. | 1 | 0 |
| `examples/numerical_solvers` | pinneapple_solvers examples: This folder contains runnable scripts showcasing the **numerical-solvers layer**: | 13 | 0 |
| `examples/nvidia` | PINNeAPPle with the NVIDIA stack: Every example puts PINNeAPPle at the centre (the problem, the reference solvers, the geometry checks, the physics checks and the trust decision) and joins it with a different combination of stacks. The glue lives in `pinneapple_tools.nvidia`; each example is a thin `run.py` that writes `results/architecture.png` (PINNeAPPle's role and each stack's role, and what was *not* run here), result figures, `summary.json` and `run_info.json` (versions, hardware, timings, which components ran). | 25 | 39 |
| `examples/openfoam_campaign` | OpenFOAM parameter campaign: One base case, six runs that change the jet speed and elevation, split between two machines. | 1 | 0 |
| `examples/pde_environment` | End-to-end demonstration for the astrophysics specialization: train a PINN on the `kepler_two_body_orbit` preset and compare the trained network's predicted trajectory against the exact (Kepler's-equation) solution. | 9 | 10 |
| `examples/physics_audit` | Physics audit on pipe friction: models judged by their physics, not only by their past errors. | 1 | 0 |
| `examples/physics_check` | A stand-in for a drag surrogate Cd(Re, aoa) fitted on wind-tunnel points (Re 1e5 .. 1e6, aoa 0 .. 10 deg). | 7 | 22 |
| `examples/physics_condensation` | Adaptive Physics Condensation, track 1: the measured benchmark of docs/research/physics_condensation.md. | 1 | 0 |
| `examples/physics_db` | Quickstart: discover a dataset via CMR/earthaccess and inspect variables. | 7 | 0 |
| `examples/physics_ensemble` | Side by side: the five-family ensemble reacting only to past errors (left) and with the current-case physics residual against the level each model has in its own training regime (right, ``residual_lookahead`` + ``residual_baseline``), on the same stream of cases. | 5 | 3 |
| `examples/physics_learning_verification` | Physics learning verification, track 3: the measured study of docs/research/physics_condensation.md (section 6). | 1 | 0 |
| `examples/physics_module` | Hybrid model: a coarse finite-element solve plus a neural correction, trained end to end. | 1 | 0 |
| `examples/physics_optimizer` | One constrained design problem through one interface, six optimizers. | 1 | 0 |
| `examples/physics_query` | Physics query engine, track 2: the measured demonstration of docs/research/physics_condensation.md (section 7). | 1 | 0 |
| `examples/pinn_solver` | pinneapple_physics.pinn_solver examples: This folder focuses on the **PINN-specific** layer of PINNeAPPle. | 10 | 0 |
| `examples/pinneapple_and_physicsnemo` | Wave equation 1D u_tt - (c^2 u_x)_x = 0 | 4 | 0 |
| `examples/pinneapple_benchmark` | 01_pde_comparison.py — PINNeAPPle PDE Architecture Benchmark | 2 | 0 |
| `examples/pinneapple_cosim` | 01_spring_mass_pinn.py — 2-DOF Coupled Spring-Mass System with Two PINN Models | 1 | 0 |
| `examples/problem_designer` | pinneapple_problemdesign examples: This folder showcases **problem elicitation + specification + planning**. | 5 | 22 |
| `examples/proof_graph` | Proof-graph experiment: does the no-go theorem for linear exceptional-point tuning (PT dimers, Zenodo 23287622) extend to PT tetramers whose bonds are modulated independently? An open question of the paper (Q2), attacked as a graph of assumptions, lemmas and a theorem, each with exact (sympy) and … | 1 | 0 |
| `examples/qualitative` | Qualitative preview of heat-sink variants, then the quantitative check. | 3 | 0 |
| `examples/quickstart` | 01 — First look: what is installed, and what is already solved for you. | 13 | 0 |
| `examples/reproductions` | Reproduce Gated X-TFC (Dwivedi, Schiassi, Sigovan, Sixou, arXiv:2510.01039v1) with the protocol frozen in TARGETS.md. | 3 | 0 |
| `examples/results_chassis` |  | 0 | 0 |
| `examples/rigid_body` | Rigid bodies with NVIDIA Newton: check the solver, then run a small scene. | 1 | 0 |
| `examples/science` | PINNeAPPle science: experiments: Every row is a quantity computed by an experiment of `pinneapple_science` (`pp.science`) and compared with its reference; the values come from the committed `result.json` files. Regenerate with `python -m pinneapple_science report all`. The map of every area (available, partial, planned) is [ATLAS.md](ATLAS.md) and [atlas.html](atlas.html). | 0 | 0 |
| `examples/scratch` | scratch: This directory holds working and experimental scripts that were used during development (benchmarking snippets, one-off result-generation and report-embedding utilities, etc.) but were never organized into the curated topic-folder examples (`examples/architectures/`, `examples/geometry/`, and so on). They may be less polished, less documented, or narrower in scope than the rest of `examples/`, and some assume they're run from the repository root with the project installed in editable mode — … | 6 | 0 |
| `examples/security` | Secure Physics AI pipeline, end to end, with pinneapple_security. | 2 | 0 |
| `examples/solver_api` | The same problem through one call: a classical FEM solve and a PINN, compared against the exact solution. | 1 | 0 |
| `examples/solvers` | Examples: time-series decomposition solvers. | 3 | 0 |
| `examples/sports` | Ball flight: drag, Magnus and gravity from tracked positions; where the inverse problem is posed, and whether the physics predicts real broadcast-tracking flights better than data-only extrapolation and boosted trees. | 5 | 0 |
| `examples/studio` | Any simulation result in 3D, with three calls: load, export, render. | 7 | 6 |
| `examples/systems` | Component registry + toolbox + physics-informed fit, end to end. | 1 | 0 |
| `examples/time_series` | PINNeAPPle Time Series Examples: This folder contains runnable scripts that demonstrate **forecasting workflows** supported by `pinneapple_timeseries`. | 10 | 0 |
| `examples/topology_optimization` | Half MBB beam and cantilever: SIMP compliance minimisation with pinneapple_design.topology. | 3 | 0 |
| `examples/trainer` | pinneapple_train example 02: DataModule + robust preprocessing. | 5 | 0 |
| `examples/use_cases` | Compact range quiet zone by physical optics: validation, feed gain x reflector size (x rolled edge) sweep, a surrogate with held-out error and the trade-off map. Generic example parameters (3 GHz, F = 1.5 m, a 1.2 m square section with a superellipse junction, a 0.6 m quiet zone at 3 m). | 34 | 17 |
| `examples/validation` | PINN, steady heat conduction: -lap T = q on the unit square, T = 0 on the walls (manufactured source, exact T). | 12 | 4 |
| `examples/validation_arena` | Run an external model (for instance the FNO or MeshGraphNet of a PhysicsNeMo experiment) through the hardened defence of the validation arena (#707), and compare each of its outputs with the reference solver. | 2 | 0 |
| `examples/visualizations` | Run all PINNeAPPle visualization examples. | 8 | 0 |
| `examples/vs_physicsnemo` | PINNeAPPle vs PhysicsNeMo — Comparison & Integration Guide: > As linhas marcadas como medidas vêm do benchmark entre frameworks > ([`benchmarks/vs_frameworks/REPORT.md`](../../benchmarks/vs_frameworks/REPORT.md)): só CPU (Intel Xeon, 1 thread por run), > 3 a 5 seeds, nvidia-physicsnemo 2.2.2 e nvidia-physicsnemo.sym 2.4.0. Nada foi medido em GPU, que é onde o > PhysicsNeMo é otimizado. As outras linhas descrevem funcionalidades e não foram medidas. | 7 | 22 |
| `examples/weather_forecasting` | Weather forecasting on ERA5: how many days a model is good for, and extreme events: A global forecast model trained with `pinneapple_physics.weather` on ERA5 (WeatherBench2, public, no account), scored by lead time against persistence, climatology and the published IFS HRES, Pangu-Weather, Keisler and NeuralGCM forecasts on the same 2020 cases, with the useful horizon (ACC >= 0.6) and the no-skill horizon (RMSE reaches climatology), a per-forecast horizon from an ensemble, and seven extreme events forecast against what happened. | 2 | 6 |
| `examples/worldmodel` | Physics World Model — end-to-end example. | 1 | 0 |

## 3. Reproduções de papers

_Sem texto._

## 4. Scripts (`scripts/`)

24 itens.

| item | o que faz | arquivos .py | testes |
|---|---|---:|---:|
| `scripts/README.md` | # scripts/ — Runbook do MVP (Flow Obstacle 2D) | 0 | 0 |
| `scripts/app_helpers_baseline.txt` | # Existing duplicates of library helpers in apps/ and examples/ (scripts/check_app_helpers.py). | 0 | 4 |
| `scripts/arena` |  | 5 | 0 |
| `scripts/astro` | Build the compact star and constellation catalogues shipped in pinneapple_physics/astro/data. | 3 | 0 |
| `scripts/build_method_status.py` | Compute the validation status of every item in ``pinneapple_catalog.methods`` from the test suites. | 1 | 2 |
| `scripts/check_app_helpers.py` | Flag reusable logic that is written in apps/ or examples/ instead of being imported from the library. | 1 | 0 |
| `scripts/check_dist.py` | Check the built distributions: every wheel package is in both the sdist and the wheel, and imports when the sdist is installed in a clean virtual environment (roadmap I5, #92). | 1 | 0 |
| `scripts/check_example_imports.py` | Check that every ``pinneapple*`` import in examples/ and templates/ resolves. | 1 | 0 |
| `scripts/check_experiment_manifests.py` | CI gate: every examples/* folder has a valid experiment.json and every Lab run reads as a valid manifest. | 1 | 0 |
| `scripts/check_release_version.py` | Fail unless the git tag, pyproject.toml, CITATION.cff, pinneapple.__version__ and CHANGELOG.md agree on the release version. | 1 | 0 |
| `scripts/cosmos` | scripts/cosmos — Cosmos (LLM / World Foundation Model) integration: Esses scripts assumem que você tem um endpoint HTTP para Cosmos (ou gateway interno) que aceita JSON e retorna JSON. | 3 | 0 |
| `scripts/dev` |  | 1 | 0 |
| `scripts/example_imports_baseline.txt` | examples/architectures/50_timeseries_transformer_forecast_toy.py | 0 | 2 |
| `scripts/fault_injection_matrix.py` | Regenerate the detection matrix of the validation checks (#697): every check against the fault library on Darcy 2-D and Burgers 1-D. | 1 | 0 |
| `scripts/geometry_library.py` | Generate, verify and publish the geometry library files. | 1 | 21 |
| `scripts/issues` | Snapshot the open issues of the repository into the contribute page (docs/org/contribute/index.html). | 1 | 0 |
| `scripts/mvp` | scripts/mvp — scripts locais “cola-tudo” do MVP: Aqui ficam scripts para: - gerar/copiar JSONs do bundle (conditions/manifest) - validar bundle contra schema + presença de arquivos - rodar uma matriz de runs da Arena (native + physicsnemo) | 3 | 0 |
| `scripts/omniverse` | scripts/omniverse — executar dentro do Omniverse Kit: Esses scripts rodam DENTRO do Omniverse Code / Isaac Sim. | 2 | 0 |
| `scripts/publish_issues.sh` | #!/usr/bin/env bash | 0 | 0 |
| `scripts/reference` | Build the module reference at docs/org/reference/: every package, subpackage and module of the repository with its docstring and public API (read from the source with ``ast``, nothing is imported), plus the curated explanations and the examples run by ``run_examples.py`` (code, printed output, … | 3 | 0 |
| `scripts/run_colab_notebooks.py` | Execute the Colab notebooks headless, top to bottom, and check that the committed copies carry no outputs. | 1 | 0 |
| `scripts/update_imports.py` | Step 1: Update all Python import statements to use the new mega-module paths. | 1 | 0 |
| `scripts/validate_model_card.py` | Governance-as-code gate for the model hub (see ``ROADMAP_PHYSICS_AI_HUB.md``, P1.5): reject a model-card push with no computed ``validation_metrics`` or no ``reference_source`` citation, the same schema check ``ModelCard.validate()`` already implements -- this script is just the CI-callable/CLI … | 1 | 8 |
| `scripts/zoo` | Build the zoo's bundled assets, model cards and ``manifest.json`` from the committed sources. | 10 | 28 |

## 5. Benchmarks (`benchmarks/`)

7 itens.

| item | o que faz | arquivos .py | testes |
|---|---|---:|---:|
| `benchmarks/data_io_bench.py` |  | 1 | 0 |
| `benchmarks/lbm_vortex_shedding.py` | Vortex shedding with the D2Q9 LBM solver: Strouhal number of a cylinder and a NACA 4412. | 1 | 0 |
| `benchmarks/lpm_ellipse_design_space.py` | Large Physics Model on an exact design space: potential-flow Cp on ellipses. | 1 | 0 |
| `benchmarks/shard_balance_bench.py` |  | 1 | 0 |
| `benchmarks/ssqn_helmholtz.py` | Paper benchmark: 2D Helmholtz with BFGS vs SSBFGS vs SSBroyden vs Adam. | 1 | 0 |
| `benchmarks/ssqn_paper_benchmarks.py` | Paper benchmarks for the self-scaled quasi-Newton optimizers (BFGS / SSBFGS / SSBroyden vs Adam). | 1 | 0 |
| `benchmarks/vs_frameworks` | Cross-framework benchmark: runners, driver and report (see PROTOCOL.md). | 15 | 0 |

## 6. Integração contínua (`.github/workflows/`)

| workflow | nome | gatilhos | jobs | o que faz |
|---|---|---|---|---|
| `docs.yml` | Deploy Docs to GitHub Pages | push, workflow_dispatch, contents | deploy |  |
| `draft-pdf.yml` | Draft PDF | push, pull_request, workflow_dispatch | paper |  |
| `extras.yml` | Install extras | push, pull_request, workflow_dispatch | extras |  |
| `observatory.yml` | Observatory (browser tests) | push, pull_request, workflow_dispatch | browser |  |
| `physics-check.yml` | Physics check (pinneapple check) | push, pull_request, workflow_dispatch, contents | passing-fixtures, failing-fixtures, darcy-trust, unit-tests |  |
| `playground.yml` | Zoo web export and Playground (ONNX parity, JS runtime, browser smoke test) | push, pull_request, workflow_dispatch | onnx-parity, browser, docker |  |
| `publish-zoo.yml` | Publish zoo to the Hugging Face Hub | workflow_dispatch, push, contents, group | publish, geometries |  |
| `release.yml` | Release to PyPI | push | build, publish |  |
| `security.yml` | Security | pull_request, push, schedule, workflow_dispatch, contents (cron `17 5 * * 1`) | scan, tests, image |  |
| `tests.yml` | Tests (audit + core suite) | push, pull_request, workflow_dispatch | test, generated, first-example, colab-notebooks |  |
| `training-callbacks.yml` | Training callbacks (Lightning, PhysicsNeMo, Hugging Face) | push, pull_request, workflow_dispatch, contents | callbacks |  |
| `validation-gallery.yml` | Validation gallery | push, pull_request, workflow_dispatch, contents | gallery |  |

## 7. Documentação existente (`docs/`)

190 arquivos Markdown. Lista com o título de cada um:

| arquivo | título |
|---|---|
| `docs/SUMMARY.md` | SUMMARY |
| `docs/apps/anomaly_detector.md` | Physics Anomaly Detector (day 28) |
| `docs/apps/benchmark_arena.md` | Physics AI Benchmark Arena |
| `docs/apps/calibration_lab.md` | Model Calibration Lab (day 18) |
| `docs/apps/control_lab.md` | Physics-Based Control Lab (day 29) |
| `docs/apps/evidence_report.md` | Engineering Evidence Report (day 30) |
| `docs/apps/hub.md` | PINNeAPPle Hub |
| `docs/apps/identifiability_lab.md` | Parameter Identifiability Lab (day 25) |
| `docs/apps/inverse_studio.md` | Inverse Model Studio (day 21) |
| `docs/apps/meshless_lab.md` | Meshless Method Lab (day 26) |
| `docs/apps/model_gateway.md` | Physics AI Model Gateway |
| `docs/apps/rom_lab.md` | Reduced Order Model Lab (day 19) |
| `docs/apps/surrogate_studio.md` | Surrogate Studio |
| `docs/apps/twin_runtime.md` | Digital Twin Runtime (day 20) |
| `docs/apps/validation_lab.md` | Physics Validation Lab (day 24) |
| `docs/architecture/execution_model.md` | Execution Model |
| `docs/architecture/package_layers.md` | Package Layers |
| `docs/architecture/system_overview.md` | System Overview |
| `docs/archive/overview_en.md` | PINNeAPPle — Library Overview |
| `docs/archive/overview_pt.md` | PINNeAPPle — Visão Geral da Biblioteca |
| `docs/assets/apps/README.md` | App images (30 apps with PINNeAPPle, days 1–14) |
| `docs/assets/proof_graph/pt_chain_no_go_report.md` | No linear exceptional-point tuning in PT-symmetric tetramers |
| `docs/core_concepts/adaptive_forecasting.md` | Adaptive forecasting: switching and combining models online |
| `docs/core_concepts/astro.md` | Positional astronomy and astrophysics (`pinneapple_physics.astro`) |
| `docs/core_concepts/astroml.md` | astroML: periodograms, deconvolved densities and fits with errors |
| `docs/core_concepts/backend.md` | Backend |
| `docs/core_concepts/black_hole_weather.md` | Black-hole weather: forecasting accretion flows with a trust horizon |
| `docs/core_concepts/core_primitives.md` | Core primitives: Domain, Geometry, Mesh, Field |
| `docs/core_concepts/dashboard.md` | Animation-centred dashboards |
| `docs/core_concepts/earth_system.md` | Earth-system building blocks and data assimilation |
| `docs/core_concepts/fitmesh.md` | From a problem description to checked CAD, fit-for-purpose meshes and one report |
| `docs/core_concepts/free_surface.md` | Free-surface flow: runners, channels, launders |
| `docs/core_concepts/geomechanics_mpm.md` | Geomechanics with the material point method |
| `docs/core_concepts/geometry_domain.md` | Geometry & Domain |
| `docs/core_concepts/lab.md` | PINNeAPPle Lab: experiments at scale, and a database of results and datasets |
| `docs/core_concepts/lbm_wind_tunnel.md` | Lattice Boltzmann wind tunnel and the FluidX3D bridge |
| `docs/core_concepts/lunar_rover_terramechanics.md` | Lunar rover terramechanics in 3D |
| `docs/core_concepts/model.md` | Model |
| `docs/core_concepts/modelica.md` | Modelica models as Physics AI data |
| `docs/core_concepts/molecular_visualization.md` | Molecular visualisation: proteins on metal surfaces |
| `docs/core_concepts/motor_and_topopt_audit.md` | Audit: electric-motor magnetics and topology optimization |
| `docs/core_concepts/multiscale_ensembles.md` | Multiscale ensembles: scales, filters and closures |
| `docs/core_concepts/newton_rigid_body.md` | Rigid bodies with NVIDIA Newton |
| `docs/core_concepts/non_hermitian.md` | Non-Hermitian physics: PT-symmetric dimers and exceptional-point tuning |
| `docs/core_concepts/openfoam_acceleration.md` | Faster implicit pressure solves in OpenFOAM |
| `docs/core_concepts/openfoam_campaigns.md` | OpenFOAM parameter campaigns on several machines |
| `docs/core_concepts/overview.md` | Core Concepts |
| `docs/core_concepts/physical_sample.md` | PhysicalSample |
| `docs/core_concepts/physics_audit.md` | Physics audit: which model to use, when to trust it, when to revisit the physics |
| `docs/core_concepts/physics_ensembles.md` | Adaptive ensembles of physics models |
| `docs/core_concepts/physics_optimizer.md` | PhysicsOptimizer |
| `docs/core_concepts/pinn.md` | PINN / Physics |
| `docs/core_concepts/problem_definition.md` | ProblemDefinition |
| `docs/core_concepts/proof_graphs.md` | Proof graphs: mathematics with the same evidence discipline |
| `docs/core_concepts/qualitative_preview.md` | Qualitative preview: physical impact of geometry changes before the numbers |
| `docs/core_concepts/reduced_order_models.md` | Reduced-order models |
| `docs/core_concepts/researcher_benchmarking.md` | Researcher & Benchmarking |
| `docs/core_concepts/security.md` | Security for scientific and industrial Physics AI (`pinneapple_security`) |
| `docs/core_concepts/solver.md` | Solver |
| `docs/core_concepts/studio.md` | 3D studio and external flow |
| `docs/core_concepts/training_pipeline.md` | Training Pipeline |
| `docs/core_concepts/veriphysics.md` | Verification and evidence (`pinneapple_veriphysics`) |
| `docs/core_concepts/viewers_and_twin_standards.md` | Other 3-D viewers and digital-twin standards |
| `docs/dev/ANALISE_FEATURES_COMERCIAIS.md` | O que as ferramentas comerciais e os outros CFDs têm que o PINNeAPPle ainda não tem |
| `docs/dev/AUDIT_REPORT.md` | Evidenced audit report |
| `docs/dev/CATALOGO_METODOS.md` | Catálogo de métodos do PINNeAPPle |
| `docs/dev/LATENT_AND_CYBER_EVALUATION.md` | Latent-space methods and cybersecurity applications: evaluation and roadmap |
| `docs/dev/LIBRARY_CONSOLIDATION.md` | Library consolidation: reusable logic outside the library |
| `docs/dev/MARKET_SCAN_LOVABLE_PHYSICS_AI_2026-09-13.md` | Pesquisa: Physics AI / Simulação / CAE / Digital Twin em lovable.app |
| `docs/dev/PEDIDOS_2026-09-24.md` | Pedidos e insights de 2026-09-24 — registro e decisões |
| `docs/dev/REA_INTEGRATION.md` | REA no PINNeAPPle — integração, fluxo de harvest e mapa do ecossistema |
| `docs/dev/ROADMAP_PHYSICS_AI_HUB.md` | PINNeAPPle → "the Hugging Face of Physics AI": full roadmap |
| `docs/dev/issues/001-geometric-similarity-score.md` |  |
| `docs/dev/issues/002-active-simulation-loop.md` |  |
| `docs/dev/issues/003-raw-geometry-field-surrogate.md` |  |
| `docs/dev/issues/004-rom-fmu-export.md` |  |
| `docs/dev/issues/005-pinneapple-mcp-server.md` |  |
| `docs/dev/issues/006-topology-optimization-simp.md` |  |
| `docs/dev/issues/007-solution-adaptive-amr.md` |  |
| `docs/dev/issues/008-multi-fidelity-cokriging.md` |  |
| `docs/dev/issues/009-hybrid-rom-online-calibration.md` |  |
| `docs/dev/issues/010-pce-sobol.md` |  |
| `docs/dev/issues/011-geometry-encoders.md` |  |
| `docs/dev/issues/012-lbm-d3q27-cumulant-voxel.md` |  |
| `docs/dev/issues/013-ifno-incremental-modes.md` |  |
| `docs/dev/issues/014-multiadam.md` |  |
| `docs/dev/issues/015-mapdl-rst-reader-pyfluent.md` |  |
| `docs/dev/issues/016-browser-pde-preview.md` |  |
| `docs/dev/issues/017-physicsnemo-missing-models.md` |  |
| `docs/dev/issues/018-rea-native-ghidra.md` |  |
| `docs/dev/issues/019-adjoint-mma-bspline.md` |  |
| `docs/dev/issues/020-design-study-object.md` |  |
| `docs/dev/issues/021-rbf-mesh-morphing.md` |  |
| `docs/dev/issues/022-inference-cache-viewer.md` |  |
| `docs/dev/issues/023-the-well-adapter.md` |  |
| `docs/dev/issues/024-procedural-pde-datasets.md` |  |
| `docs/dev/issues/025-neural-corrected-stepper.md` |  |
| `docs/dev/issues/026-differentiable-linear-solves.md` |  |
| `docs/dev/issues/027-nektar-bridge.md` |  |
| `docs/dev/issues/028-jaxfluids-bridge.md` |  |
| `docs/dev/issues/029-paraview-catalyst-insitu.md` |  |
| `docs/dev/issues/030-nn-in-weak-forms.md` |  |
| `docs/dev/issues/031-warp-backend.md` |  |
| `docs/dev/issues/032-rbf-kriging-surrogates-outliers.md` |  |
| `docs/dev/issues/033-onshape-configuration-sampler.md` |  |
| `docs/dev/issues/034-rescale-job-connector.md` |  |
| `docs/dev/issues/035-comsol-mph-bridge.md` |  |
| `docs/dev/model_dataset_inventory.md` | Model and dataset inventory for the zoo (2026-10-13) |
| `docs/dev/pr/feat-rea-bridge.md` | Summary |
| `docs/getting_started/first_pinn.md` | First PINN |
| `docs/getting_started/installation.md` | Installation |
| `docs/getting_started/project_layout.md` | Project Layout |
| `docs/getting_started/quickstart.md` | Quickstart |
| `docs/getting_started/start_here.md` | Start here (docs) |
| `docs/guides/adding_an_experiment.md` | Adding an experiment / Como adicionar um experimento |
| `docs/guides/agent_workbench.md` | Agent Workbench: coding and engineering agents on a local model |
| `docs/guides/antenna_design.md` | Antenna design: an end-fire Vivaldi with an external FEM solver |
| `docs/guides/autonomous_lab.md` | Autonomous Physics Lab: an AI that manages evidence for engineering decisions |
| `docs/guides/autonomous_physics_lab.md` | Autonomous Physics Lab |
| `docs/guides/campaign_planner.md` | Simulation Campaign Planner: which simulations need to run |
| `docs/guides/chain_of_whys.md` | Chain of whys: why each element of a design exists, computed |
| `docs/guides/check_power.md` | Check power: how often each check catches a known defect |
| `docs/guides/check_power_matrix.md` | Check power: the detection matrix |
| `docs/guides/choosing_a_method.md` | Choosing a method: PINN, neural operator, graph network or the solver |
| `docs/guides/colab_notebooks.md` | Colab notebooks |
| `docs/guides/compact_range.md` | Compact range quiet zone: feed gain × reflector size |
| `docs/guides/dataset_audit.md` | Dataset physics audit: does the data solve its PDE? |
| `docs/guides/design_from_intent.md` | Design from intent |
| `docs/guides/design_report.md` | Design reports without an LLM: `Datasheet` and the self-planning study |
| `docs/guides/design_studies.md` | Design studies: one problem, many optimisers, verified candidates |
| `docs/guides/electromagnetics.md` | Electromagnetics and electrodynamics |
| `docs/guides/engineering_design_optimizer.md` | Engineering Design Optimizer (app, day 17) |
| `docs/guides/engineering_gui_agent.md` | Engineering GUI agent: run engineering software and keep the evidence |
| `docs/guides/enso_forecasting.md` | Predicting El Niño: physical, physics-informed, data-driven and hybrid models under one protocol |
| `docs/guides/evidence_ledger.md` | Evidence ledger: a test passed only if it ran |
| `docs/guides/flight_dynamics.md` | Flight dynamics, turbulence and take-off |
| `docs/guides/flow_schema.md` | The flow schema (`pinneapple-flow/1`) and how another app reuses it |
| `docs/guides/fmi_compatibility.md` | FMI import compatibility |
| `docs/guides/football_physics.md` | Football: where physics helps on public tracking data |
| `docs/guides/freecad_openfoam_pipelines.md` | FreeCAD -> OpenFOAM pipelines |
| `docs/guides/geometry_library.md` | Geometry library |
| `docs/guides/identifiability_calibration_inversion.md` | Identifiability, calibration and inverse problems |
| `docs/guides/industrial_presets.md` | Industrial presets |
| `docs/guides/interlocking_bricks.md` | Self-modifying brick builds |
| `docs/guides/mcp_server.md` | PINNeAPPle from any coding agent: the MCP server |
| `docs/guides/metamorphic_tests.md` | Metamorphic tests: checking a surrogate without ground truth |
| `docs/guides/model_zoo.md` | Pre-trained models and datasets (`pp.zoo`, `pp.datasets`) |
| `docs/guides/near_random_forecasting.md` | Forecasting near-random series without overfitting |
| `docs/guides/observatory.md` | The PINNeAPPle Observatory |
| `docs/guides/physics_checks_in_ci.md` | Physics checks in CI |
| `docs/guides/physics_design_optimizer.md` | Physics Design Optimizer (app, day 23) |
| `docs/guides/physics_residual_attack_detection.md` | Detect false data injection with physics residuals |
| `docs/guides/physicsnemo_trust.md` | PhysicsNeMo trains, PINNeAPPle says when to trust |
| `docs/guides/pinn_validation.md` | PINN validation: what the training loss does not say |
| `docs/guides/pinneapple_llm_quality_pipeline.md` | Quality pipeline for LLM-generated physics and twin artifacts |
| `docs/guides/playground.md` | Playground: physics surrogates in the browser |
| `docs/guides/pp_validate.md` | pp.validate: one call, one validation report |
| `docs/guides/recreate_a_wear_twin.md` | Recreate a 3D wear twin (refractory, liner, pipe wall) |
| `docs/guides/residual_error_bound.md` | From residual to error bound: how wrong is this prediction, without the true answer? |
| `docs/guides/rollout_validation.md` | Rollout validation: when an autoregressive surrogate stops being trustworthy |
| `docs/guides/rom_and_meshless_studies.md` | Reduced-order models and meshless methods: measured, not assumed |
| `docs/guides/rover_learns_the_planet.md` | The rover that learns the planet |
| `docs/guides/running_a_benchmark.md` | Running a benchmark |
| `docs/guides/scale_aware_ensembles.md` | Choose a model by the scale of the question |
| `docs/guides/science.md` | Science by area (`pp.science`) |
| `docs/guides/sensitivity_lab.md` | Engineering Sensitivity Lab |
| `docs/guides/spectral_fidelity.md` | Spectral fidelity: what relative L2 does not see |
| `docs/guides/surrogates_arena_gateway.md` | Surrogates from simulation results: study, regime arena, gateway |
| `docs/guides/topology_optimization.md` | Topology optimisation |
| `docs/guides/training_callbacks.md` | Physics checks during training (Lightning, PhysicsNeMo, Hugging Face) |
| `docs/guides/trusted_latent_surrogate.md` | A latent surrogate that says when not to trust it |
| `docs/guides/uncertainty_and_sensitivity.md` | Uncertainty and sensitivity studies on any model |
| `docs/guides/uncertainty_lab.md` | Engineering Uncertainty Lab |
| `docs/guides/unit_cell_benchmark.md` | Periodic unit cell: an FNO over frequency × scan, and an equal-budget benchmark against Bayesian optimisation |
| `docs/guides/validation_arena.md` | Red team / blue team for the validation checks |
| `docs/guides/validation_gallery.md` | Validation gallery: one worked example per model family and per physics |
| `docs/guides/validation_methods_reference.md` | Validation methods reference |
| `docs/guides/workflow_runner.md` | Simulation Workflow Runner: campaigns as traceable, resumable workflows |
| `docs/guides/zoo_catalogue.md` | Zoo catalogue |
| `docs/index.md` | PINNeAPPle |
| `docs/lab/coverage.md` | Experiments coverage |
| `docs/philosophy.md` | Philosophy |
| `docs/research/condensation_falsification.md` | Falsification campaign for physics condensation — pre-registration |
| `docs/research/physics_condensation.md` | Adaptive Physics Condensation: geometry-independent thermal models of chip packages |
| `docs/security/audit.md` | Security audit: the PINNeAPPle web apps and their deployment |
| `docs/security/runbook.md` | Operator runbook: PINNeAPPle apps |
| `docs/security/server.md` | Server hardening: the host that runs the PINNeAPPle apps |
| `docs/strategy/STRATEGY_PHYSICS_AI.md` | Physics AI Strategy — structure and first implementation |
| `docs/validation_status.md` | Validation status |
| `docs/why.md` | Why PINNeAPPle? |
