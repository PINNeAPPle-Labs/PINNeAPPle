# Changelog

All notable changes to PINNeAPPle (the `pinneapple` package on PyPI) are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow
[Semantic Versioning](https://semver.org/) with one caveat: **while the major version is 0, a minor release
(0.x.0) may contain breaking changes.** Every breaking or behavior-changing entry is listed under
**Changed** and starts with "**Breaking:**" or "**Behavior:**".

How this file is maintained is described in [CONTRIBUTING.md](CONTRIBUTING.md#changelog). The entries for
0.5.0 and later were reconstructed from the git history and checked against the files published on PyPI.


## [Unreleased]

### Added
- Interior-PM machine FEA (`pinneapple_simulation.electromagnetics.machines`): parametric V-magnet rotors and
  round-bottom slots, conforming mirrored sector meshes with a moving band, M400-50A B-H and loss data, stacking
  factor, table and Bertotti iron-loss models, and `IPMMachine` (Maxwell-stress and co-energy torque, cogging, ripple,
  flux linkage, back-EMF, dq flux linkages, frozen-permeability L_d / L_q, MTPA, iron and magnet losses). Verified
  against a closed form (magnetised ring, second order), exact areas, torque consistency (co-energy, dq power
  balance, frozen-permeability superposition) and the published Prius torque curve (Yang et al., IEMDC 2013);
  example `examples/electrodynamics/10_pmsm_prius.py`, tests `tests/test_emachine_pmsm.py`.
  `analytic.magnetized_ring` added.
- Electromagnetics and electrodynamics (`pinneapple_simulation.electromagnetics`, `pp.em`; guide
  `docs/guides/electromagnetics.md`):
  - SI core: CODATA 2022 constants (`constants`), SI dimensions of every field and a dimensional check of Maxwell's
    equations (`units.check_equations`, `units.check_units`); `pinneapple_data.physical_units` gains T, mT, Wb,
    coulomb, farad, henry and siemens.
  - Media (`media`): isotropic, anisotropic (`Anisotropic`, uniaxial crystals, plane-wave indices, reciprocity) and
    dispersive Drude / Lorentz / Debye poles with auxiliary-differential-equation coefficients for time-domain
    solvers, phasor convention `exp(+j omega t)`; `hysteresis`: Froehlich-Kennelly curve (plugs into the Newton
    solver) and the Jiles-Atherton model.
  - References in closed form: `statics` (capacitors, exact Biot-Savart of segments, loops in elliptic integrals,
    solenoids, mutual and self inductance), `waves` (Fresnel, multilayers, skin depth, waveguides, cavities, dipoles,
    TMz cylinder scattering, surface plasmons), `mie` (dielectric and conducting spheres, checked against miepython
    and Bohren & Huffman's published values).
  - `electrostatics.solve_electrostatic` (P1 FEM, capacitance from energy), `mesh.rect_mesh`.
  - The electric-machine work of the `feat/motor-magnetics-topopt` branch: A_z magnetostatics with Newton for
    nonlinear B-H, magnets and (anti-)periodic constraints; time-harmonic eddy currents in solid conductors; SPM and
    reluctance machine model (Maxwell-stress and co-energy torque, cogging, back-EMF, losses, adjoint
    sensitivities); `pinneapple_design.topology` (SIMP, filter, projection, OC, MMA; compliance, thermal,
    Stokes-Brinkman and rotor problems; guide `docs/guides/topology_optimization.md`).
  - `verification`: 19 cases against closed forms, published values and miepython with measured orders; the table
    in the guide is the output of `examples/electrodynamics/07_em_verification.py`.
  - Physics checks of any field (`pinneapple_physics.maxwell_checks`, `pp.maxwell`): Gauss, Faraday,
    Ampere-Maxwell and continuity residuals with a grid-halving test that tells discretisation error from a field
    that breaks the law, Poynting energy balance, PEC faces, tangential continuity, S-parameter reciprocity and
    passivity, and a deploy decision. `pinneapple check` runs them with `physics = "maxwell"` (PC102, PC301, PC103;
    tolerances `maxwell-gauss`, `maxwell-faraday`, `maxwell-ampere`, `maxwell-continuity`, `maxwell-energy`).
  - PDE compiler kinds `electrostatics`, `magnetostatics_az`, `eddy_current_az`, `maxwell_tm`, `maxwell_3d`,
    `maxwell_tm_frequency` and the integral forms `electrostatics_integral` and `maxwell_tm_integral` (Gauss's law
    and Ampere-Maxwell / Faraday on control squares, which see material interfaces); presets `em_*` with closed-form
    solutions registered for `pp.solve(problem, "analytic")`.
  - Optional extra `em` (miepython); catalog items S29-S39 and E66-E71.
- Surrogate Studio app (day 14, `apps/surrogate_studio`, surrogate.pinneapple.org, port 8097). It turns uploaded
  simulation results (CSV, Parquet, JSON, HDF5, NPZ or a `DatasetSpec`) into a surrogate:
  - an equal-budget comparison of every applicable method on held-out and distribution-shift splits;
  - the chosen model hosted at `/v1/models/<name>/predict` (the Model Gateway API) with its OOD gate;
  - a limitations report and a trust decision;
  - downloads as contracts (`StudyResult`, `ModelRef`, `DatasetSpec`), a report, ONNX and TorchScript.
  It runs on the shared job queue, API keys and usage metering. Two offline examples run in about 10 s. Deployed with
  `docker-compose.yml` (service `surrogate`), the Caddyfile and `.env.example`.
- `pinneapple_analysis.study_contracts`: the shared contracts for uncertainty and sensitivity studies. A
  `ParameterSpace` becomes study parameters (uniform on the bounds, other distributions from
  `metadata["distributions"]`; categorical and log-scaled parameters are refused) and back exactly; a `ModelRef`
  (`zoo:<name>`, ONNX) becomes a `StudyModel` and back; `uncertainty_result` / `sensitivity_result` write a
  `StudyResult` with the checks that ran. The Engineering Sensitivity Lab runs its studies in the shared job queue
  with metering (`studies`, `evaluations`, `parameters`) and reads and writes these contracts; so does the Engineering
  Uncertainty Lab (`studies`, `scenarios`, `propagation_samples`, `hf_runs`; a `ModelRef` and a `ParameterSpace` in, a
  `StudyResult` out).
- `train_gnn` (`pinneapple_design.aero.train`) gains `checkpoint_dir` and `checkpoint_every`. Every few epochs it
  saves the full training state (weights, optimiser, epoch, elapsed time, history, shuffling RNG). A later call with
  the same settings resumes from it to the same weights as an uninterrupted run, and a checkpoint written with other
  settings is refused (#780). `scripts/zoo/train_airfoil_gnn.py` trains the airfoil graph network on the regenerated
  campaign with checkpoints, and `build_assets.py --airfoil-gnn` gates and packages it. The first such network was
  rejected by the gate: its coefficients pass, but its field errors and stagnation pressure fail (see the inventory).
- Meshless Method Lab app (day 26, `apps/meshless_lab`, meshless.pinneapple.org, port 8109): a Poisson, Helmholtz or
  heat problem (manufactured solution or source + boundary value) on a preset, polygon or point-cloud domain, solved
  by Kansa, RBF-FD, embedded finite differences, linear FEM and a PINN in a queued refinement study on the shared
  runtime; error, cost, condition number, noise amplification, observed order, the Kansa shape-parameter sweep, error
  maps and the winner of each criterion; `StudyResult` contract and CSV/JSON downloads; four offline examples
  (smooth L-shape, corner singularity, heat on a plate with a hole, Helmholtz on the disk).
  `pinneapple_simulation.meshless.contracts` builds the contract.
- Reduced Order Model Lab app (day 19, `apps/rom_lab`, rom.pinneapple.org, port 8102): upload field snapshots
  (OpenFOAM times, VTU/VTK/FRD series, NPZ/HDF5, CSV/Parquet or a `DatasetSpec` table), queue a ROM study on the
  shared runtime, compare POD + RBF / GPR, DMD, Operator Inference and an equal-budget neural surrogate on held-out
  times or designs (error vs rank vs speed, energy, reconstructions), and export `.rom.npz`, ONNX for the model
  gateway, `ModelRef` and `StudyResult` contracts; `POST /api/run` runs an exported ROM. Two offline examples from
  the library's solvers (LBM cylinder wake, solid-FEM cantilever designs). `pinneapple_neural.architectures.rom
  .contracts` builds the contracts.
- Engineering GUI Agent app (`apps/guiagent`, guiagent.pinneapple.org, port 8115). It takes a task sentence or a
  workflow and runs it in FreeCAD, CfdOF, OpenFOAM and ParaView inside a separate, optional heavy Docker image. The page
  shows the step timeline with checks and screenshots, asks for approval before the solver, and delivers the HTML
  evidence report, the StudyResult and a zip of the run. Runs go through the shared runtime (job queue, ownership,
  metering of runs, compute seconds and storage). In the public mode only the committed workflows run; uploads need
  self-hosted mode. The Simulation Workflow Runner registers the `guiagent` step kind. Deployed through the compose service `guiagent` (profile `guiagent`) and a Caddy block.
- Engineering GUI agent (`pinneapple_tools.automation`): runs a task in engineering software and keeps the evidence.
  A structured workflow (YAML/JSON: inputs with bounds, steps with preconditions, expected files and confirmation
  categories, acceptance criteria) is executed step by step through adapters for FreeCAD and CfdOF (Python API
  inside the FreeCAD GUI on Xvfb, observed with xdotool), OpenFOAM (command line), ParaView (`pvpython`), Playwright
  and the acceptance criteria. The runner stops on the first failure and before expensive, overwriting, destructive
  or external steps until a person approves them, and records commands, logs, files with SHA-256, screenshots and
  checks; `write_report` gives a self-contained HTML evidence report and `to_study_result` a
  `pinneapple_data.contracts.StudyResult`. `runner_step.register()` makes it a step kind (`guiagent`) of the Simulation
  Workflow Runner.
  `interpret` maps a sentence to a committed workflow and its inputs (Ollama optional). Criteria: `mass_balance`,
  `inflow`, `pressure_drop` (from `pinneapple_check`'s finite-volume reader), `converged`, `artifact`,
  `physics_lint`. Example `channel_obstacle`: FreeCAD project, macro, CfdOF case, OpenFOAM v1912, ParaView export;
  [guide](docs/guides/engineering_gui_agent.md).
- Design studies in `pinneapple_design.design_optimizer` (guide: `docs/guides/design_studies.md`): one problem
  description (`DesignSpace` with continuous and integer variables, `Objective`, `Constraint` with physical,
  manufacturing and validity kinds, `DesignProblem`), evaluators for Python functions, zoo models, torch modules,
  external solvers run as commands and model references (`from_model_ref`), and `run(problem, method, budget=...)`
  for `random`, `nsga2`, `pymoo-nsga2` (optional pymoo), `scipy-de`, `scipy-slsqp`, `bayesian` and `gradient`
  (autograd) under one evaluation budget. `compare_methods` tabulates hypervolume, cost and convergence on the same
  budget; `verify_candidates` re-evaluates chosen designs with a reference and reports surrogate errors and a
  verification status; `surrogate_assisted` runs GP active learning against a reference; `feasibility_report` lists
  active constraints and margins; `hypervolume` is exact in 2-D. Presets: plate-fin heat sink (zoo MLP, closed
  form, FVM reference) and CST airfoil (zoo ensemble), with STL/STEP design files.
- `pinneapple_design.design_optimizer`: `problem_from_spec` / `reference_for_spec` (problems from JSON specs: presets,
  zoo models, tables), `study_result` and `to_parameter_space` (the shared data contracts, when
  `pinneapple_data.contracts` is installed), `surrogate_from_table` with held-out metrics and
  `model_ref_for_surrogate`, `table_evaluator` (offline solver results as a reference), and a `progress` callback on
  `run`, `compare_methods` and `surrogate_assisted` (an exception from it, e.g. a cancelled job, stops the run).
- `MethodComparison.union()`: every run's evaluations in one record with the run that produced each row.
- `pinneapple_design.geometry_library.exporters.stl_bytes`: the binary STL writer for any triangle mesh.
- Inverse Model Studio app (day 21, `apps/inverse_studio`, inverse-studio.pinneapple.org, port 8104). It is a new app,
  separate from the Inverse Heat Lab, which is unchanged.
  - It formulates an inverse problem once (measurements, model, unknown parameters, boundary conditions or field
    zones) and solves it with least squares, the adjoint gradient, autodiff, Bayesian regularisation and the
    Inverse Heat Lab's PINNs on the same data.
  - It shows the reconstructed fields, the residuals, held-out ranking, the identifiability diagnosis, and truth
    recovery on the examples.
  - It exports a `StudyResult` and runs on the shared job queue and usage metering.
  - Docs page: `docs/apps/inverse_studio.md`.
- Parameter Identifiability Lab app (day 25, `apps/identifiability_lab`, identifiability.pinneapple.org, port 8108).
  Before an experiment or a calibration, it checks which parameters a measurement plan can determine. It reports the
  Fisher rank and conditioning, the ambiguous combinations, local sensitivity and leverage, profile likelihood,
  posterior correlations, and D-optimal extra measurements. It runs on the shared job queue and usage metering, reads
  tables and a `ParameterSpace`, and exports a `StudyResult`. Shared page pieces are in
  `apps/_shared/static/shared/inverse.js`; the docs page is `docs/apps/identifiability_lab.md`.
- Model Calibration Lab app (day 18, `apps/calibration_lab`, calibration.pinneapple.org, port 8101). It fits model
  parameters to measurements (CSV, Excel, Parquet, JSON, sensor exports). The model can be built in, a formula, or a
  table of simulated runs.
  - Jacobian and bootstrap intervals are given only for identifiable parameters ("not estimable" otherwise, from the
    day-25 core).
  - Before/after comparison, and validation on held-out or extrapolated measurements.
  - The other estimators (Bayesian, autodiff, adjoint, PINN) run on the same data, ranked by held-out error.
  - It exports a `StudyResult` and a `ModelRef`, and runs on the shared job queue and usage metering.
  - Docs page: `docs/apps/calibration_lab.md`.
- Identifiability, calibration and inverse-problem core in `pinneapple_analysis.inverse_problems` (one core for the
  Parameter Identifiability Lab, Model Calibration Lab and Inverse Model Studio apps; guide
  `docs/guides/identifiability_calibration_inversion.md`). It provides forward models (`ExpressionModel`,
  `CallableModel`, `TabulatedModel` for simulated results, and `PinFinModel`, `RodModel` with zone-wise h(x), exact
  sensitivities and a one-solve adjoint, `PlateModel`, `LumpedCoolingModel`, `ArrheniusModel`), `Parameter` and
  `Observations` (read from CSV, Excel, Parquet or JSON tables).
  - `analyze_identifiability` combines Fisher rank and conditioning, the ambiguous combinations, measurement
    leverage, profile likelihood, an adaptive-Metropolis posterior, and D-optimal suggestions for extra
    measurements.
  - `calibrate` gives Jacobian and bootstrap intervals only for identifiable parameters ("not estimable" otherwise)
    and validates on held-out or extrapolated rows.
  - `compare_methods` / `solve_inverse` run least squares, adjoint, autodiff, Bayesian and PINN on the same data.
  - Ten presets with known answers.
  - `inverse_problems.contracts` maps parameters, reports, calibrations and comparisons to the shared app contracts
    (`ParameterSpace`, a `StudyResult` of type identifiability, calibration or inverse, and a `ModelRef` with
    held-out metrics). It needs `pinneapple_data.contracts`.
- `heat_transfer.run_plate` accepts measured `readings` at `sensors_mm` and an air temperature `t_air`, so the plate
  PINN can be run on given data (the synthetic default is unchanged).
- PhysicsNeMo trains, PINNeAPPle checks (2-D Darcy):
  - `pinneapple_physics.darcy_checks`: physics checks of a predicted Darcy field `-div(a grad u) = f`. They use
    the reference solver's own finite-volume discretisation and need no reference solution. `residual_error_bound`
    turns the residual into a guaranteed bound on the energy-norm error (one fast sine transform). The module also
    checks the boundary condition, net boundary flux against the source, and positivity. `CoefficientEnvelope`
    checks applicability against the training coefficients. `calibrate_tolerances` derives limits from a model's own
    predictions on held-out in-distribution inputs (a drift monitor that needs no labels). `residual_l2_bound`
    certifies the L2 error. `PROFILES` / `profile_verdicts` give use profiles (`fields`, `fluxes`) with a-priori
    limits, their reasons, and a graded ratio value / limit. `sample_checks` checks one prediction exactly as
    `pinneapple check` does, and `check_config_entry` writes its configuration. `estimated_error`,
    `estimated_flux_error`, `regional_errors` and `use_verdicts` say where a prediction errs and which uses it can
    serve (interior field; walls and boundary fluxes; inputs), from the residual alone. `darcy_checks`,
    `darcy_trust_report` and `trust_report` give the deploy decision for a batch.
  - `check_nodal_darcy_batch` (and `nodal_darcy_residual`, `solve_nodal_darcy`): the same kind of checks for
    training data stored at the nodes of a finite-difference grid, such as PhysicsNeMo's `Darcy2D` datapipe. It
    uses the diversity measures of `pinneapple_data.audit`; the audit's own physics check covers cell-centred data.
  - `pinneapple_neural.physicsnemo_bridge.field_operators`: `physicsnemo_fno` and `physicsnemo_meshgraphnet` (on the
    grid graph, `grid_graph`) build PhysicsNeMo's own models as NumPy predictors (`GridFieldPredictor`, with
    save / load). `train_grid_operator` trains them under a `ComputeBudget`.
  - `pinneapple check`: `physics = "darcy"` for cell-centred grid results with fields `a` and `u` (PC301 error
    and L2 bounds, PC101 flux balance, PC201 `u = 0`, PC402 positivity). An `envelope` table checks the coefficient
    against its training envelope under PC401. Tolerances: `darcy-energy`, `darcy-l2`, `darcy-flux`, `darcy-bc` and
    `darcy-positivity`.
  - `operator_datasets.darcy_set` (Darcy pairs from shifted input distributions, solved like the zoo dataset) and
    `block_average`. `darcy_inputs` takes `tau`, `high` and `low`.
  - `pinneapple_tools.visualization.trust_panels`: field panels with verdict badges, residual and error maps, as PNG
    or GIF.
  - Example `examples/vs_physicsnemo/07_physicsnemo_trains_pinneapple_checks` and the guide
    `docs/guides/physicsnemo_trust.md`. The `darcy-trust` job of the physics-check workflow runs the GitHub Action
    on the exported predictions.
- Veriphysics scope of use (`pinneapple_veriphysics.scope`): `ScopeOfUse` lists which uses a result can serve,
  where it fails and why (measured value against the limit, what the use needs, and whether the evidence is
  certified, estimated, a reference comparison or an input check). Input checks are preconditions, not uses.
  `scope_from_verdicts` and `scope_from_batch` build it from per-use verdicts such as `darcy_checks.use_verdicts`.
  `DecisionRecord.scope_of_use` carries it into the Decision Record and the Evidence Report PDF, which gains a
  "Where this result can be used" section with the figure. The PhysicsNeMo Darcy example writes one per model
  (`python run.py report`).
- Validation gallery (`examples/validation/`, #705): 12 CPU examples covering PINN, FNO / DeepONet, MeshGraphNet,
  POD / DMD / OpInf, MLP / Gaussian process / least squares, neural ODE vs Hamiltonian network, a diffusion model and
  external models (ONNX, PhysicsNeMo), on heat, Darcy, reaction-diffusion, elasticity, a cylinder wake, advection,
  molecular dynamics, magnetostatics and cantilever modes. Each ends in `pp.validate`, a trust card and
  `pinneapple check` outputs (`validation.gallery.conclude`), and catches one real failure on a broken variant. CI
  workflow `validation-gallery.yml` runs them at smoke size. Also new: `pp.validate(sampler, {"generative": ...})`
  (per-sample physics, diversity and memorisation of a generative model), state-vector rollouts (`ndim = 0`) with
  user invariants, `numerical_solvers.reaction_diffusion` (Allen-Cahn, Fisher-KPP) and
  `pinneapple_neural.utils.fit_supervised`.
- Physics checks during training (`pinneapple_neural.callbacks`, #704): `PhysicsMonitor` runs the checks of
  `pp.validate` on a validation batch every N steps, warns or stops when a physics metric worsens for K checks while
  the loss improves, and writes the report and trust card at the end; adapters for PyTorch Lightning
  (`PhysicsValidationCallback`), PhysicsNeMo training loops (`PhysicsNeMoHook`, `LaunchLogger`) and the Hugging Face
  Trainer (`PhysicsTrainerCallback`), none of them a dependency. `residual_bound.ResidualLoss`: a differentiable
  physics loss in the reference discretisation. CI workflow `training-callbacks.yml` runs them against the real
  frameworks.
- PINN validation (`pinneapple_analysis.validation.pinn`, #703): residual at fresh grid, random and adaptive points;
  detectors for a near-trivial solution, propagation failure in time, boundary / initial conditions traded off
  against the residual and spectral bias; exact-solution gates (Burgers by Cole-Hopf, manufactured Poisson,
  Kovasznay); `PINNMonitor` runs the checks during training and stops on a failure mode. Checks `pinn.*` in
  `pp.validate` with `problem={"pinn": ...}`.
- Rollout validation of autoregressive surrogates (`pinneapple_analysis.validation.rollout`, #701): invariant
  drift, high-band energy over the horizon, divergence time (trust horizon) and, for chaotic systems, long-run
  statistics (value histogram, decorrelation time, band energies) against the noise floor of the reference. Checks
  `rollout.*` in `pp.validate` with `problem={"rollout": {...}}`. Kuramoto–Sivashinsky solver
  (`pinneapple_simulation.numerical_solvers.kuramoto_sivashinsky`).
- Operations library (`pinneapple_systems.operations`) for running validated models next to plant data: a
  heated-tank plant with mass and energy balances, sensors and six fault kinds (`plant`); operator inference with
  inputs fitted on whole trajectories with held-out rollout errors, also as CasADi expressions (`rom`); a digital
  twin runtime with EnKF state and parameter estimation, forecasts with input uncertainty, rule-based alerts, drift
  monitoring with `assess_drift`, open-loop revalidation and a SQLite/Postgres history (`twin`); physics anomaly
  detection with statistical, AR/CUSUM, conservation-balance and model-residual channels, events with explanations
  and a labelled fault benchmark (`anomaly`); linear (SLSQP) and nonlinear (CasADi/IPOPT or SciPy) MPC compared with
  PID and rule-based baselines in closed loop, simulation / recommendation only (`control`); the days 13-31 chain
  campaign -> surrogate -> uncertainty -> optimizer -> twin through the data contracts (`series`); `ModelRef` and
  `StudyResult` builders (`contracts`).
- Embedded MQTT 3.1.1 broker (`pinneapple_systems.digital_twin.io.mqtt_broker.MQTTBroker`: QoS 0/1, retained
  messages, wildcards, optional login) for demos and single-asset edge setups; works with `MQTTStream`.
- `PIDController` takes optional `output_limits` (saturation with anti-windup by conditional integration) and a
  feed-forward `bias`; without them it behaves as before.
- Validation report as an artifact (`pinneapple_analysis.validation.report_artifact`, #706): JSON schema
  `validation-report/v1` with provenance hashes, `pp.validate.diff(old, new)` (lost / gained checks, moved metrics,
  validity-envelope changes), a model-card section, an SVG badge and a shields.io endpoint, signatures with
  `pinneapple_security.signing`. Stored reports for the zoo models (`pinneapple_hub.zoo.validation_reports`,
  `scripts/zoo/validation_reports.py build|check`), shown in their Hub cards; the zoo publishing workflow refuses a
  model version that loses a check the stored report passed.
- `pp.validate(model, problem, data=None, reference=None, level="standard", query=None)` (#696): one validation
  report for a zoo model, a PyTorch / PhysicsNeMo module, an ONNX file or a callable. It composes the applicability
  envelope, the guaranteed residual bound and the calibrated error bound, flux balance, reference error, spectral
  fidelity, metamorphic relations, the data audit, the zoo gate and robustness; reports pass / warn / fail /
  not-applicable per check, a credibility level per factor (modelled on NASA-STD-7009A and ASME V&V 40) and a deploy
  decision; outputs JSON (`validation-report/v1`), HTML, text, the trust report and `pinneapple check` formats
  (SARIF, JUnit). Statuses, factors and check IDs live in `pinneapple_analysis.validation.schema`.
  `pp.validate.<name>` still resolves the names of `pinneapple_analysis.validation`.
- `pinneapple_neural.architectures.neural_operators.fno_export`: ONNX export of FNOs (spectral convolutions in real
  arithmetic, checked against PyTorch with onnxruntime).
- Metamorphic tests for surrogates (`pinneapple_analysis.validation.metamorphic`, #698): symmetry, similarity
  scaling, invariance, monotonicity, superposition and resolution consistency as declarative relations with a
  tolerance and a sampler; random and (with `hypothesis`) property-based search; the worst violation is returned as a
  counterexample. `pinneapple_hub.zoo.relations.check_zoo_relations` runs the relations of the airfoil and forced
  heat-sink zoo models. `pinneapple check`: new rule PC303 `metamorphic-relation` and a `relations` key on
  `[[model]]`. Docs: "Metamorphic tests".
- Dataset physics audit (`pinneapple_data.audit`, `pp.datasets.audit`, #700): per-sample consistency with the stated
  PDE in the dataset's own discretisation (built in for Darcy, reusing `operator_benchmarks.darcy_2d`), declared
  column constraints, diversity (effective sample count and input spectrum against the stated generator, exact and
  near duplicates), train/test leakage by nearest neighbour, units and missing values; an audit card (JSON,
  markdown) stored for the bundled zoo datasets and shown in their Hub card. `pinneapple check`: `[[dataset]]` targets
  and rules PC451 (physics), PC452 (diversity), PC453 (leakage). Docs: "Dataset physics audit".
- Residual-to-error calibration (`pinneapple_analysis.uncertainty.residual_bound`, #699): a guaranteed a-posteriori
  bound `||u_h - u||_A <= ||r||_{L^-1} / sqrt(min k)` for Darcy and steady heat conduction in the discretisation of
  the reference solver, and a split-conformal map residual -> error (`ResidualErrorCalibrator`, `coverage_study`)
  whose coverage is measured and reported as broken under an input shift; `to_check` for trust reports.
  `operator_benchmarks.darcy_matrix` exposes the finite-volume matrix of `darcy_2d`. Docs: "From residual to error
  bound".
- FMI import compatibility: `check_fmus` / `check_fmu` (`pinneapple_simulation.external_solvers.modelica.fmi_compat`)
  run FMUs through `simulate_fmu` in every interface they offer and tabulate what passed; `FMUSimConfig.fmi_type`
  selects Co-Simulation or Model Exchange. The new "FMI import compatibility" page, regenerated by
  `examples/modelica/fmi_compatibility.py`, lists the Modelica Reference-FMUs 0.0.39 (FMI 1.0, 2.0 and 3.0, CS and
  ME) and a pythonfmu FMU: 38 of 38 runs pass on x86_64-linux.
- Engineering Evidence Report app (day 30, `apps/evidence_report`, evidence.pinneapple.org, port 8113). It
  turns the outputs of a study into a report (HTML, PDF) with methodology, conditions, baseline comparison,
  tests, limitations and artifact references by SHA-256 and URL, and into an evidence package (zip with a SHA-256
  manifest). A test is printed as passed only when its own evidence record shows it ran; otherwise it reads "not
  run". Contracts are validated against `pinneapple_data.contracts` and every report is exported as a StudyResult
  (`evidence_package.to_study_result`); API keys and usage metering come from the shared app runtime. The example
  runs a real FEM-vs-PINN study. Deployed with `apps/deploy` (service `evidence`).
- Physics Validation Lab app (day 24, `apps/validation_lab`, validate.pinneapple.org, port 8107). Give it
  predictions or results with a form, a `pinneapple check` configuration or a zoo ModelRef. It reports numerical
  verification, validation against data and physical consistency separately, lists the groups that did not run,
  keeps an evidence record per check and produces a reproducible report (JSON, HTML, configuration, ledger). It
  has a REST API and a GitHub Action recipe; uploaded code is refused. Deployed with `apps/deploy` (service
  `validate`).
- ROM study (`pinneapple_neural.architectures.rom.study`): `rom_study(snapshots)` holds out later times,
  interleaved times or unseen parameter points, fits POD + RBF, POD + GPR, the nearest snapshot, DMD, Operator
  Inference and a neural surrogate trained within the same wall-clock budget (`train_within_budget`) at several
  ranks, and reports the held-out error, the POD projection floor, fit and inference time, speed-up over the
  full-order model, the singular values and the rank / error / speed Pareto front. Every fitted model is a
  `ReducedModel` with `predict(times=...)` / `predict(params=...)` and a portable `.npz` form (arrays + JSON
  manifest, no pickle). `ParametricPOD.to_arrays` / `from_arrays` rebuild an identical predictor.
- Snapshot matrices (`pinneapple_data.snapshots`): `load_snapshots` stacks OpenFOAM time directories, series of
  VTU/VTK/FRD/meshio results, `.npz`/`.npy`/HDF5 arrays and CSV/Parquet tables (a row per snapshot) into a
  `SnapshotSet` that keeps the field layout, times or parameters, and locations.
- Meshless study (`pinneapple_simulation.meshless`): Kansa (global multiquadric collocation), RBF-FD (PHS r³ +
  polynomials), finite differences on an embedded grid (Shortley-Weller), P1 finite elements on the Delaunay mesh of
  the nodes and a PINN on the same nodes, for Poisson, Helmholtz and heat problems with Dirichlet conditions on 2-D
  domains (square, disk, L-shape, annulus, plate with a hole, polygons, point clouds). `meshless_study` reports the
  error against a manufactured solution (sympy) or a fine FEM reference with its own error estimate, setup and solve
  time, condition number, noise amplification, observed order, a Kansa shape-parameter sweep and the winner of each
  criterion.
- `pinneapple_lab.experiments.rom.cylinder_wake` and `cantilever`: the wake and cantilever generators of
  `rom_study`, now public.
- Surrogates from simulation results (`pinneapple_analysis.surrogate`): `run_study` trains every applicable method
  (polynomial ridge, Gaussian process, gradient-boosted trees, MLP, FNO, DeepONet, and a PINN when a PDE is
  declared) on the same splits and `ComputeBudget`, measures them on a held-out split and a distribution-shift split
  (held-out groups, the edge of the input space, or one end of an input), selects by validation error and packages
  the result as a `SurrogateModel` (predict with an OOD gate, save / load with a sha256 check, TorchScript and ONNX
  export, ModelRef fields). The limitations report gives validity bounds, the OOD gate, where the errors concentrate
  and the PDE residual of predicted fields; a `TrustReport` decides. Readers for CSV, Parquet, JSON, HDF5 and NPZ;
  examples that need no download (the OpenFOAM airfoil campaign, exact Burgers fields and an FDM baseline);
  `surrogate.contracts` converts data, models, studies and arena results to and from the shared contracts
  (`DatasetSpec`, `ModelRef`, `StudyResult` of `pinneapple_data.contracts`). Guide:
  `docs/guides/surrogates_arena_gateway.md`.
- Regime arena (`pinneapple_arena.regimes.run_regime_arena`): ranks methods per regime (interpolation, few data,
  extrapolation, geometry change, steep fronts) over several seeds with equal data, budget and stopping rule, with
  training and inference cost, model size, physics violations of declared constraints, numerical-solver baselines
  and an audit of the splits by `pinneapple_data.audit`.
- Model gateway (`pinneapple_registry.ModelGateway`, on `ModelStore`): versioned endpoints for surrogates, zoo
  models and ONNX graphs, one `predict` that enforces each version's OOD policy (warn or refuse; callers may only
  tighten it), a run log, operational metrics (calls, rows, errors, OOD rows, latency p50 / p95) and exports.
  `pinneapple_registry.gateway_api.gateway_router` is its REST shape (`/v1/models/...`), shared by the apps that
  serve models. `GatewayClient` calls it from Python with the standard library only. A `remote` kind serves another
  app's model through this gateway, with local validity bounds, policy, logging and metering. `publish_ref` publishes
  from a `ModelRef` (`zoo:<name>` or an http(s) gateway URI).
- `pinneapple_data.audit.constraint_mask`: the evaluator of declared constraints, public for checking predictions.
- Engineering Sensitivity Lab (`apps/sensitivity_lab`, day 16, sensitivity.pinneapple.org, port 8099): Sobol indices
  with bootstrap intervals, Morris screening, interactions, PDP/ICE curves, autodiff derivatives and the freeze
  recommendation with its re-run check, for a model, runs done in a solver (design CSV round trip) or a dataset.
  Deployed as service `sensitivity` (`SEN_*` variables).
- Engineering Uncertainty Lab (`apps/uncertainty_lab`, day 15, uncertainty.pinneapple.org, port 8098): split-conformal
  intervals with coverage measured on held-out data, applicability alerts, a routing rule checked against the heat-sink
  physics engine, and propagation of uncertain inputs, for zoo models or uploaded ONNX graphs. Deployed as service
  `uncertainty` (`UQL_*` variables).
- Sensitivity analysis package `pinneapple_analysis.sensitivity` (`pp.sensitivity`, `pp.sobol_indices`): first-order,
  total and second-order Sobol indices with bootstrap intervals (Saltelli 2010 / Jansen 1999 / Saltelli 2002
  estimators, checked against the exact Ishigami and Sobol' G indices), indices from runs done elsewhere
  (`saltelli_design` + `sobol_from_runs`), given-data first-order indices, Morris elementary effects, partial
  dependence and ICE curves, local derivatives by autodiff or central differences, and a freeze recommendation that
  uses the upper confidence bound plus a direct re-run check (`verify_freeze`); `sensitivity_study` runs them
  together on one model and `study_cost` gives its number of evaluations beforehand.
- Uncertainty-study helpers in `pp.uncertainty`: input distributions (uniform, normal, truncated normal, lognormal,
  triangular) sampled by MC, LHS or Sobol (`distributions`); `propagate_space` (`pp.propagate_uncertainty`);
  least-squares polynomial chaos with leave-one-out Q2 and Sobol indices (`polynomial_chaos`); split-conformal
  intervals on any model's predictions with the coverage measured on held-out data, reliability across levels and
  per-group coverage (`conformal_study`, `pp.conformal_study`); a range + k-NN density applicability gate
  (`ApplicabilityDomain`) and the rule that routes cases to the high-fidelity solver (`route_cases`);
  `uncertainty_study` runs them together for one model output.
- `pinneapple_analysis.study_models.StudyModel` (`pp.StudyModel`): one interface for zoo models (with their torch
  module for autodiff, ensemble spread and reference splits), ONNX graphs and Python callables.
- `pinneapple_design.thermal.heatsink.surrogate.reference_solve`: the heat-sink physics engine on one design given in
  the dataset's columns (the high-fidelity answer the zoo surrogates approximate), and `reference_solve_batch`.
- Simulation workflows (`pinneapple_core.workflows`). A workflow is a DAG of steps (dependencies, cycle check,
  per-case expansion with `workflow_from_cases`) run by `WorkflowRunner`, which provides:
  - per-step states, logs, attempts and outputs in SQLite;
  - retries; failures that cancel only dependent steps (or `fail_fast`);
  - resume of interrupted runs; cancel;
  - skip-done by content hash (kind, parameters, declared input files, upstream hashes).

  Executors are a plugin registry: `local`, `docker` (`docker run --network none`), and `slurm` (sbatch scripts; dry
  run by default). Step adapters are a registry too: `command`, `python`, `openfoam` (reuses
  `openfoam.campaign.edit_file` / `parse_solver_log`), and `calculix` (`ccx` + `read_frd`). `dataset_from_run` turns
  finished runs into a `DatasetSpec` and its table. Whitelisted templates: slab heating (exact solution), CalculiX
  cantilever vs Euler-Bernoulli, and the OpenFOAM cavity.
- Day 27 app: Simulation Workflow Runner (`apps/simulation_runner`, runner.pinneapple.org, port 8110). It runs case
  tables (e.g. Campaign Planner exports) through the templates, with traceable runs, resume, rerun with skip-done,
  Slurm script generation, and dataset downloads (CSV / Parquet / HDF5 / JSON + DatasetSpec). Usage is metered as
  jobs run, compute seconds and storage. Public servers run only the templates; `RUNNER_SELF_HOSTED=1` enables
  custom workflows.
- Simulation Campaign Planner (day 13, `apps/simulation_campaign`, campaign.pinneapple.org, port 8096): plan which
  simulations to run over a `ParameterSpace` and a budget. It returns a design of experiments, coverage before and
  after, repeats of existing runs dropped, a run order, OpenFOAM / CalculiX / Python case files, and a `StudyResult`
  plus a `DatasetSpec` skeleton. The library behind it:
  - `pinneapple_data.doe`: LHS, maximin, Sobol, Halton, full and fractional factorial with minimum-aberration
    generators, and sequential and adaptive infill; coverage metrics, redundancy, priorities, and `plan_campaign`.
  - `pinneapple_simulation.external_solvers.campaign_cases`: the case writers.
- Evidence ledger, Validation Lab and evidence package in `pinneapple_veriphysics`:
  - `ledger`: `EvidenceEntry` / `EvidenceLedger`. Each entry records the check, the SHA-256 of its inputs, its
    outputs, start and finish times and the verdict, sealed with the SHA-256 of its content. `status()` is
    "passed" or "failed" only for a complete, intact entry; anything else reads "not run" (an edited entry, a
    crash, a missing hash or time). `record_checks` records the checks of a trust card (N/A checks get no entry).
  - `validation_lab`: runs `pinneapple check` on the inputs of a folder and groups the outcomes into numerical
    verification, validation against data and physical consistency. Per target, it lists the groups in which
    nothing ran. Every outcome gets a ledger entry; the report holds the input manifest with hashes and the
    configuration that reproduces it. `config_from_spec` writes the configuration from a plain description;
    `assert_no_code` refuses models that would run uploaded code (zoo models only). `render_html` writes the report.
  - `evidence_package`: reads other tools' outputs (Validation Lab reports, ledgers, `pinneapple check` JSON,
    Simulation Comparator results, StudyResult / ModelRef / DatasetSpec JSON, metrics tables, any file by hash).
    It builds a report with methodology, conditions, baseline comparison, tests, limitations and artifact
    references, renders it as HTML and as PDF (`pinneapple_tools.reporting.paperkit`) and packs a zip with a
    SHA-256 manifest (`verify_package`). A test is printed as passed only when the source's own ledger records
    its run; a bare claim reads "not run".
- Data contracts shared by the apps (`pinneapple_data.contracts`): `ParameterSpace` (bounds, units, log scale,
  integer and categorical parameters, constants, safe constraint expressions, unit-hypercube mapping, feasibility),
  `DatasetSpec` (columns with roles and units, files with SHA-256, splits, provenance), `ModelRef` (held-out metrics
  only, validity range) and `StudyResult` (table, metrics, checks that need a record, artifacts, stated limits).
  JSON/YAML load and save, validation with readable errors, JSON Schema export (`schemas/`), and CSV / Parquet /
  HDF5 tables with a `.contract.json` sidecar (embedded in Parquet and HDF5 too).
- App server runtime (`pinneapple_core.service`, standard library only): a persistent job queue (SQLite; thread or
  process workers; progress, logs, cancel, retry, automatic retries, time limit, max concurrent jobs across server
  workers, resume after a worker dies), hashed API keys, usage metering with monthly free limits from
  `<PREFIX>_FREE_<UNIT>`, and an upload store with size and type limits. `apps/_shared/appkit.py`
  `install_runtime` exposes them under `/api` (`/api/jobs`, `/api/keys`, `/api/usage`, `/api/uploads`) and accepts
  `X-API-Key` alongside the per-app login; documented in `apps/_shared/README.md`.
- Observatory: space journey from your place to the target (#723). `pinneapple_physics.astro.journey`: heliocentric
  ICRS positions of the observer (Earth from the JPL ephemeris plus the site), the Solar System bodies at the date and
  the stars from their parallax (Hipparcos 2007 for the brightest); galactic coordinates (IAU 1958 on J2000 axes);
  distance, one-way light time and travel times at walking, airliner, New Horizons, Voyager 1 and light speed; the
  target's apparent magnitude (m = M + 5 log10(d / 10 pc) for stars, inverse square for bodies) and angular diameter
  versus the distance left; scale markers (Moon distance, 1 au, heliopause, 1 light-year, nearest star) and the Sun's
  magnitude seen from a star. The app's Journey view lifts off from your latitude and longitude, leaves the atmosphere
  and flies across the Solar System or through the neighbouring stars in 3-D with a logarithmic speed control,
  counters (travelled, left, light time, size and brightness of the target, trip times) and, on arrival, a look-around
  view with the info card (#724); at a star, "Look back home" shows the Sun as a star of the right magnitude.
  `/api/journey/{plan,scene}`; "Fly there" from the Learn card. Tests: `tests/test_astro_journey.py` (arrival distance
  against the ephemeris and the parallax to 0.1 %, the Sun at about +0.5 from Alpha Centauri, light time 1.28 s to the
  Moon and 8.3 min to the Sun) and `apps/observatory/tests/e2e/journey.spec.js` (Mars in English, Sirius in Portuguese).
- Observatory: observing session mode (#722). `pinneapple_physics.astro.observing`: visibility of any target (body,
  star, Messier object) with altitude, air mass, extinction per band, sky brightness (Krisciunas & Schaefer 1991
  moonlight + twilight), parallactic angle and field rotation, rise / transit / set for fixed objects (within a
  minute of Skyfield), the best window; telescope optics (magnification, true field, exit pupil, Dawes and Rayleigh
  limits, limiting magnitude, image scale, camera field) with presets; GoTo slews with a trapezoidal profile, horizon
  limit and meridian flip; survey cut-outs from CDS hips2fits (DSS2 colour), cached on the server and credited on
  screen. `conditions.fetch_cloud_forecast` / `sky_view`: "real sky" from the Open-Meteo cloud forecast, "perfect
  sky" without weather, and a fallback to perfect sky without a forecast. The app's Observe tab plans the night,
  sets up the telescope, slews and tracks the target in an eyepiece or camera view at 1x to 3600x (field rotation,
  extinction, sky brightness, seeing, clouds), with a session log (notes, snapshots, conditions); offline the field
  is drawn from the star catalogue. `/api/observe/{plan,optics,slew,weather,presets,field.jpg}`. Tests:
  `tests/test_astro_observing.py` (Skyfield, Kasten & Young, optics formulas, analytic field rotation) and
  `apps/observatory/tests/e2e/observe.spec.js` (a 2-hour session at 600x stays centred; a forced overcast forecast
  hides the view and is logged).
- Observatory: Learn mode with facts and curiosities about each object (#724). `pinneapple_physics.astro.facts` and a
  versioned dataset in `pinneapple_physics/astro/data/facts/` (the Sun, planets, Moon, dwarf planets and major moons,
  the brightest stars, the 88 constellations, the 110 Messier objects, the ISS, Hubble and Tiangong): every number
  carries its source (NASA fact sheets, Hipparcos 2007, IAU, d3-celestial, published distances), text written in
  English and Portuguese. Cards give key numbers with everyday comparisons, three to five curiosities, history,
  missions, how to see the object tonight from your place (`facts.tonight`) and a quiz; search by name in both
  languages and a "Did you know?" tip on start-up. The app has a Learn tab, the start of the card in the Object tab,
  and `/api/facts/{card,search,list,tip}`. Constellation areas are computed from the IAU boundaries
  (`scripts/astro/build_facts.py`). Tests: `tests/test_astro_facts.py` (NASA fact sheet values, DE421 cross-checks,
  IAU areas, sources and both languages) and `apps/observatory/tests/e2e/learn.spec.js`.
- Observatory: Portuguese and English interface (#721). A language switcher (EN / PT) in the top bar; the default
  follows the browser, the choice is kept in the URL (`#lang=pt`) and in local storage. Every string the page shows is
  in the message catalogues `static/i18n/en.json` / `pt.json`, including check names, verdicts, units and API errors
  (the server sends a message code in `X-Error-Code`); numbers and dates follow the locale. Constellations use their
  Portuguese names (`pinneapple_physics.astro.stars.constellation_names`; the two halves of Serpens now get their own
  names and label points), stars keep their IAU names. Library checks and validation rows carry a stable `id` for
  translation. Tests: catalogue parity and a lint for hard-coded text (`tests/test_observatory_i18n.py`), and a
  Playwright test that switches to Portuguese and back (`apps/observatory/tests/e2e`). Screenshots in both languages.
- Data contracts shared by the apps (`pinneapple_data.contracts`): `ParameterSpace` (bounds, units, log scale,
  integer and categorical parameters, constants, safe constraint expressions, unit-hypercube mapping, feasibility),
  `DatasetSpec` (columns with roles and units, files with SHA-256, splits, provenance), `ModelRef` (held-out metrics
  only, validity range) and `StudyResult` (table, metrics, checks that need a record, artifacts, stated limits).
  JSON/YAML load and save, validation with readable errors, JSON Schema export (`schemas/`), and CSV / Parquet /
  HDF5 tables with a `.contract.json` sidecar (embedded in Parquet and HDF5 too).
- App server runtime (`pinneapple_core.service`, standard library only): a persistent job queue (SQLite; thread or
  process workers; progress, logs, cancel, retry, automatic retries, time limit, max concurrent jobs across server
  workers, resume after a worker dies), hashed API keys, usage metering with monthly free limits from
  `<PREFIX>_FREE_<UNIT>`, and an upload store with size and type limits. `apps/_shared/appkit.py`
  `install_runtime` exposes them under `/api` (`/api/jobs`, `/api/keys`, `/api/usage`, `/api/uploads`) and accepts
  `X-API-Key` alongside the per-app login; documented in `apps/_shared/README.md`.
- Zoo dataset `airfoil_rans_cst_re4e6_fields` (Hub-only, 186 MB in 5 files): the cell fields p, Ux, Uy, nut and
  cell centres of the 436 OpenFOAM runs of `airfoil_rans_cst_re4e6`, with the same splits. The campaign was
  regenerated, and its coefficients reproduce the published table. `write_field_dataset`
  (`pinneapple_design.aero.field_dataset`) writes it, sharding splits larger than 120 runs.
- Zoo dataset `thermal_channel_cfd` (Hub-only, 360 MB in 8 files): the actuated heated channel of Benchmark
  experiment 05. It holds 82 trajectories of u, v, p, T on 128 × 32 with 101 snapshots, the actuators and the
  quantities of interest. The splits are train 48 / val 8 / test 12 of random steps and ramps, plus two actuation
  families unseen in training. The card carries the solver checks (Poiseuille, Nusselt number) and a three-grid GCI.
  The recipe is `pinneapple_simulation.numerical_solvers.thermal_channel_datasets` and the command line is
  `scripts/zoo/thermal_channel_data.py`.
- Spectral fidelity metrics (`pinneapple_analysis.validation.spectral`, #702): radially binned energy spectra
  (periodic, or mean-removed and Hann-windowed), energy ratio, phase error and coherence per wavenumber band, a
  high-wavenumber bias score, structure functions with flatness, and errors in gradients, vorticity, strain rate and
  wall flux. Reported by `pinneapple check` (new rule PC704 `spectral-bias`, tolerance `hf-bias`) and by the
  `pp.benchmark` field problems ("HF energy bias" column). Docs: "Spectral fidelity".
- Zoo dataset `navier_stokes_2d_li2021_nu1e-3` (Hub-only, 919 MB in 12 shards of 100 runs): 2-D incompressible
  Navier–Stokes in vorticity form, ν = 10⁻³, with the forcing and initial-vorticity distribution of Li et al. (2021).
  It holds 1200 runs (1000 train / 200 test) solved pseudo-spectrally on 128² and stored on 64² at t = 0, 1, …, 50.
  `pp.datasets.load(name, split="test")` downloads only the two test shards.
- Validation arena (`pinneapple_analysis.validation.arena`, `pp.validate.arena(problem, checks, budget=...)`, #707):
  a red team / blue team loop on top of the fault-injection suite. The red team scores the defence with the fault
  library as mutation operators (mutation score), searches for escapes (wrong outputs every check accepts; differential
  evolution, CMA-ES or L-BFGS-B over mutants and physics-consistent attacks that solve a nearby problem exactly) and
  for false alarms (correct cases from the verified solvers that a check rejects). The blue team calibrates one
  combined test with split-conformal quantiles (`uncertainty.conformal.conformal_quantile`) so the family-wise
  false-alarm rate is stated (`alpha`), and adds checks from the library until every stored escape is rejected.
  Escapes and false alarms are stored as regression cases; `pinneapple check --arena-cases` replays them (the
  Darcy 2-D and Burgers 1-D cases ship with the library). `audit_model(problem, predict)` runs an external model
  through the hardened defence. Docs: "Red team / blue team for the validation checks"; example:
  `examples/validation_arena/`. `uncertainty.conformal` gains `conformal_quantile` (any array) and
  `conformal_quantile_valid`, and keeps float64 scores in float64.
- Fault-injection suite (`pinneapple_analysis.validation.fault_injection`, #697): measures how often each validation
  check catches known defects. A fault library (model: bias, scale, high-frequency noise, smoothing, wrong boundary
  values, a source-term sign flip, out-of-range use, a stale model; data: a discretisation coefficient off by a
  constant, duplicated or near-duplicate samples, train/test leakage, unit mix-ups, swapped channels) is injected into
  correct Darcy 2-D and Burgers 1-D cases from the verified `operator_benchmarks` solvers, and `run_fault_injection`
  returns the detection matrix (check x fault x magnitude), the false-alarm rate of each check on clean cases, the
  minimal detectable magnitude, ROC AUC and the true error of each fault; `check_power()` is the "check power"
  section of a validation report. Checks reuse `pinneapple_check.numerics`, the physics monitor and the solvers'
  own stencils. `scripts/fault_injection_matrix.py` regenerates the matrix (docs: "Check power"); CI runs it at a
  small size. `operator_benchmarks.darcy_operator` exposes the Darcy finite-volume matrix (with per-axis weights)
  that `darcy_2d` solves; `benchmark.validation.field_features` is now public.
- Black-hole weather, zoo version (#399). The library gains the tools, and the example scripts only call them:
  - `pinneapple_physics.blackhole.runs`: generate, resume and read simulations;
  - `convergence`: a resolution study with conservative remap between non-nested grids, Richardson order and GCI,
    and statistical convergence with batch-mean and seed intervals;
  - `dataset`: zoo packing, one file per split, by trajectory;
  - `skill`: persistence, climatology and DMD baselines, moving-block bootstrap skill against lead time, physics
    checks;
  - `figures`.
- `forecast` gains training on whole trajectories with a multi-step rollout loss, input noise, window-mass and
  mass-change penalties, replicate padding, and per-epoch checkpoints that resume.
- `RIAFConfig(perturbation_kind="modes")`: the same seeded initial perturbation on every grid.
- `pinneapple_analysis.verification.forecast_skill`: `skill_vs_lead`, block bootstrap and batch-means intervals.
- Hub-only dataset `blackhole_riaf_density_256`: four-grid resolution study in its card, split by trajectory.
- The retrained forecaster is not shipped: it fails the mass-conservation check
  (`docs/dev/model_dataset_inventory.md`).
- `pp.benchmark` (`pinneapple_analysis.benchmark`) and the `pinneapple benchmark prepare|run|report` command: equal-budget
  benchmarks in one call (problems x methods under one `ComputeBudget`, equal hyper-parameter search, seeds, budget
  levels, an equal-FLOPs arm, an `out_dir` cache that resumes and refuses mixed configurations). A prepare stage saves
  and hashes the solver configuration, geometry, data and its cost; validation runs PINNeAPPle's own stack on them
  (solver verification, GCI, preflight, mesh quality, y+, physics audit, applicability envelope, trust card);
  correctness gates (architecture, training, determinism) run by default and mark rows INVALID or "did not converge";
  an advisor lists diagnostics with evidence and configuration edits. Built-in problems: `burgers`, `darcy`,
  `navier_stokes`, `airfoil`, `inverse_heat`; methods: `pinn`, `fno`, `deeponet`, `mgn`, `mlp`, `mlp_fields`,
  `pinn_rans`; `ArrayProblem` and `register_problem` / `register_method` for your own. Docs: "Running a benchmark";
  example: `examples/benchmark/`.
- `pp.benchmark`: `budget_study` (the best configuration again at longer budgets, a table per instance and
  `figures/budget_study.png`) and `ablations` (a setup audit: changes to the best configuration at the full budget,
  next to the unchanged run). The airfoil RANS PINN gets two options, off by default: an exact far field and
  collocation points even in log r (`airfoil_impl.RANS_FAR`, `RANS_MIX`).
- `pp.benchmark` gates: the airfoil PINNs' exact-solution check runs their own training residuals on solutions that
  exercise every term (a harmonic stream function; Kovasznay flow with an eddy viscosity) and must fail with a flipped
  sign; `Problem.main_metric_for(method)` judges "did not converge" on what each method predicts (Cd for the
  integrals-only MLP, the velocity field for the field models: potential flow has zero drag by construction).
- Equal-budget method comparison (#299), `pinneapple_analysis.cost.equal_budget`: `ComputeBudget` (wall clock,
  FLOPs, steps), `train_within_budget` (stops at the budget, budget-aware schedules, evaluation off the clock),
  `count_flops` (PyTorch's operator formulas plus FFTs, and unlike `FlopCounterMode` it runs around
  `torch.autograd.grad`, so PINN residuals are counted), `equal_budget_search` (the same number of trials for every
  method), `AmortizedCost` (data, training and inference cost per query, break-even), `relative_l2`,
  `time_per_query`, `hardware_record`.
- `pinneapple_simulation.numerical_solvers.operator_benchmarks`: the reference solvers of the standard
  operator-learning benchmarks (Li et al. 2021) written from the equations and checked on exact solutions: batched
  pseudo-spectral Burgers and 2-D vorticity Navier-Stokes (ETDRK4), finite-volume Darcy, periodic Gaussian random
  fields, and `FourierInterpolant` (a differentiable periodic interpolant for hard initial conditions in PINNs).
- `train_within_budget(checkpoint=..., resume=...)`: long budgeted runs save their accounting and continue the same
  budget after an interruption.
- `pinneapple_design.aero.panel`: a Hess-Smith panel method (inviscid reference for airfoils: Cl, Cp, Cm, velocity
  anywhere), checked against thin-airfoil theory.
- `run_plate` (Inverse Heat Lab) takes an optional `budget=ComputeBudget(...)` and the network and collocation
  sizes; without them it behaves as before.
- Docs page "Choosing a method" (`docs/guides/choosing_a_method.md`): a decision table backed by the equal-budget
  benchmark (PINNeAPPle-Benchmark experiment 09), linking every row to the library entry points.
- `CONTRIBUTING.md` explains who reviews pull requests, the best-effort response targets, and how someone becomes a
  maintainer.
- `pp.zoo.catalogue()` and the docs page `docs/guides/zoo_catalogue.md` generated from it. The page lists every zoo
  model and dataset: version, task, held-out metrics, trust decision, size (bundled or Hub-only), licence and
  Hugging Face link. A test keeps the committed page in sync with the manifest.
- `pp.` shortcuts for the helpers consolidated from the apps, examples and satellite repositories: `pp.uncertainty`,
  `pp.closed_form`, `pp.excitation`, `pp.reporting` (modules) and `pp.as_tensor`, `pp.to_jsonable`, `pp.dominant_frequency`,
  `pp.assess_drift`, `pp.propagate_tolerances`, `pp.score_against_reference` (resolved on first use).
- Zoo datasets for operator learning (Hub-only), the problems and input distributions of Li et al. (2021),
  generated with PINNeAPPle's verified solvers:
  - `burgers_1d_li2021`: viscous Burgers, ν = 0.1, 1200 trajectories on 1024 points, 11 snapshots. The grid and
    time-step study gives a 4e-12 relative L2 discretisation error; the solver is checked against Cole–Hopf.
  - `darcy_2d_li2021`: 1200 permeability / pressure pairs on 128 × 128. Each sample is solved on 512² and averaged
    onto the cells. The grid study against 1024² gives 4.3e-4 relative L2 (Richardson estimate about 8e-4).
  - Both have a 1000 / 200 train / test split.
- `pinneapple_simulation.numerical_solvers.operator_benchmarks` (reference solvers and Gaussian random fields of
  the three standard problems) and `operator_datasets` (the dataset recipes: inputs, solves, convergence study,
  generation). `scripts/zoo/operator_data.py` is the command-line front end.
- `PhysicsEnsemble.run_delayed(queries, references, available_at, issue_times=None)`: prequential evaluation where
  each label is used only once it exists (pending updates replayed in order of availability), returning an
  `EnsembleRun` like `run`; `PhysicsEnsemble.last_state()` and `update(..., state=...)` score a stored prediction
  when its label arrives later. `hstack_gifs` in `pinneapple_tools.visualization.dashboard.static` stitches
  animations side by side (replaces the `stitch` helper of the physics-ensemble examples).
- Surrogate monitoring and sensor design: `pinneapple_analysis.uncertainty.assess_drift` (nRMSE, bias, interval
  coverage, per-feature KS input shift, out-of-range fraction, Mann-Kendall residual trend and an ok / review /
  retrain decision), `uncertainty.propagate` (Monte-Carlo tolerance propagation with Wilson intervals on the
  probability of meeting limits) and `pinneapple_analysis.data_assimilation.sensor_placement` (NumPy Gaussian process
  with an anisotropic Matern kernel, closed-form leave-one-out, marginal-likelihood fit, and greedy sensor placement
  by integrated-variance reduction after Krause, Singh & Guestrin 2008).
- Library homes for helpers that lived in examples: `pinneapple_neural.utils.as_tensor` (model output ->
  prediction Tensor); `pinneapple_data.datasets.deepmind_mgn` (DeepMind meshgraphnets partial download over HTTP
  Range, TFRecord reader, `cylinder_flow` conversion) and `train_dynamics` / `eval_rollout` in
  `graphnn.mgn_dynamics`; `pinneapple_systems.excitation` (step, ramp, sinusoid, PRBS, chirp, ... actuator signals and
  rate limit); `NASAPowerClient` and `OpenMeteoClient` in `pinneapple_data.datasets.external_sources`;
  `pinneapple_physics.solid.axisymmetric` (strains, stresses, equilibrium residual and von Mises in (r, z) with
  optional torsion, tested against the Lame cylinder); `burgers_periodic_solve` / `burgers_periodic_pairs` in
  `pinneapple_data.datasets.physics` (pseudo-spectral, checked against Cole-Hopf). The examples import them.
  `best_device(allow_mps=False)` skips Apple MPS.
- Helpers that lived inside the apps are now library API, and the apps call them: `pinneapple_core.jsonable.to_jsonable`
  (one JSON sanitiser instead of six private copies), `pinneapple_data.cae.dataset.preview` / `field_scalar`,
  `cae.report.worst_cells`, `cae.compare.tolerance_verdicts`, `Lineage.report` / `flags` / `Artifact.verified`,
  `pinneapple_data.preflight.attach_snippets` / `drop_results` / `is_result_file`, `formfill.compile_documents` /
  `check_overrides` / `output_filename` / `FormSpec.describe`, and `unified.before_after_series`. The heat-sink and PCB
  apps import the library modules directly (their alias modules stay for old imports).
- Zoo models as ONNX, with per-query trust checks:
  - `pp.zoo.export_onnx` / `onnx_parity` (`pinneapple_hub.zoo.export`): one graph from the card's physical inputs
    to its physical outputs, with pre- and post-processing included. The export is refused above 1e-5 difference
    from PyTorch on the zoo's held-out inputs.
  - `pp.zoo.query_checks` / `check_spec` (`pinneapple_hub.zoo.checks`): range (PC601), physical output (PC402/403),
    ensemble spread (PC602) and the gate's physics checks at the query (PC302), plus a verdict (pass / warn / fail).
    As in the gate, the rover's dFx/ds check follows the sign of the recorded Wong–Reece reference (v1.1.0).
    Design constraints are reported apart.
  - `pp.zoo.export_web` and `python -m pinneapple_hub.zoo.web DIR`: a browser bundle (graph with checks, card, spec,
    reference data). The JS runtime `pinneapple_hub/zoo/web/{onnx,zoo}.js` ships with it.
  - The airfoil predictor also returns `LD` and `LD_std` (L/D per ensemble member).
- CST airfoil geometry:
  - `pinneapple_design.aero.geometry`: `handles`, `shape_from_handles`, `out_of_bounds`.
  - `pinneapple_design.aero.geometry_onnx`: the geometry and the handle solve as ONNX graphs.
- `pp.geometries.export_web`: gallery index and thumbnails for the browser.
- PINNeAPPle Playground (`apps/playground`): a static web page in which the zoo models answer live in the browser,
  with the trust verdict next to every number. It has four modes:
  - airfoil: drag the CST shape or pick a NACA preset; the polar against held-out OpenFOAM and XFOIL;
  - heat sink: R with its conformal 95 % bound, and ΔP;
  - VIPER rover wheel;
  - the geometry library in 3-D.

  The app is UI code only, built on the library features above. The state lives in the URL, the page works offline,
  and the mobile layout is supported. `engine=ort` runs onnxruntime-web instead of the built-in interpreter. CI
  (`.github/workflows/playground.yml`) runs the Python and JS tests of the export and the runtime, a Playwright
  smoke test, and a Docker build. Deploy config: `apps/playground/Dockerfile`, a `playground` service in
  `apps/deploy` (port 8095), and an opt-in GitHub Pages step (`PLAYGROUND_PAGES`) in the docs workflow.
  Guide: `docs/guides/playground.md`.
- Multiscale adaptive ensembles (`pinneapple_physics.multiscale`, #599-#604). Models declare the scales they resolve;
  queries declare the scales they need.
  - Filters: box, Gaussian and sharp spectral, with exact transfer functions. Restriction and prolongation, post-filter
    realizability, variable-width filters and their commutation error.
  - 2-D vorticity DNS and LES (pseudo-spectral, ETDRK4).
  - Closures: Smagorinsky, dynamic Smagorinsky, gradient (Clark). A-priori tests (stress and tendency correlation,
    energy transfer, backscatter), a-posteriori tests (spectrum, dissipation, long-run stability) and a label-free
    closure health check (enstrophy growth, spectral pile-up, Germano-identity error).
  - Multiscale residuals split into subfilter, commutation and model parts.
  - Cross-scale consistency check, which names the integral that broke.
  - `ScaleProfile`, `ScaleQuery` and `ScaleAwareRouter`: a fidelity ladder from surrogate to LES with closure to
    resolved solver, with a "not covered" answer when no model resolves the requested scale.
  - Scale-linking models (PyTorch): a scale-conditioned FNO with a filtering-consistency loss, a differentiable LES,
    and a closure trained a posteriori through the solver.
  - Decaying 2-D turbulence benchmark, ensemble comparison and animation.
  - Results (180 held-out queries, 256² DNS reference):
    - The scale-aware ensembles never answer below a model's resolved scale. The scale-blind ensemble, once cost
      counts, does so on 25-29 queries, with 32-36 % error at the finest width against 3.3 %.
    - In mean error per unit cost the scale-aware ensembles do not beat the scale-blind one on this small 2-D
      case, where the 128² DNS costs about as much as the LES.
    - The closure check flags static Smagorinsky on every query (Germano error 1.35, dissipation 1.5-7.9× too
      high) and never the dynamic, gradient or learned closures.
    - The closure trained a posteriori is the best LES: spectrum error 0.008-0.013 against 0.04-0.05.
    - The scale-conditioned FNO beats the scale-blind network at the held-out width but not fixed-width FNOs
      post-filtered exactly.
  - Example `examples/physics_ensemble/multiscale_turbulence.py`; docs page "Multiscale ensembles" and the guide
    "Choose a model by the scale of the question".
- `PhysicsEnsemble(prediction_checks=..., error_tolerance=...)`: label-free checks of each prediction, and an error
  tolerance per context, both escalate through the router's fidelity ladder. A model that fails is still scored.
- `pinneapple_tools.reporting.paperkit`: the report figure style (`setup_mpl`, `savefig`, `CAT`/`SEQ`/`DIV`
  palettes), `fmt`, `sanitize` and `Paper`, an A4 PDF builder for numbered technical reports (reportlab, from the
  `veriphysics` extra, is imported only when a `Paper` is built). Moved from the benchmark and climate repos.
- Closed-form flow and heat helpers moved from benchmarks, lab experiments and examples into the library:
  `closed_form.pipe_flow` (laminar / Blasius / Colebrook / Haaland friction factors with validity warnings, Darcy and
  parallel-plate pressure drop, `first_cell_height` for a target y+), `closed_form.compressible` (isentropic ratios,
  `normal_shock`, Rayleigh `pitot_pressure_ratio`, modified-Newtonian `cp_max_newtonian`),
  `closed_form.heat.slab_transient` (Fourier series, Carslaw & Jaeger) and `burgers.burgers_sine_inviscid_exact` (entropy solution).
- `physics_audit.regime.dominant_frequency` (Hann window, rfft, parabolic peak refinement) for Strouhal numbers;
  `fitmesh.size_layers` (geometric prism-layer stack); `pinneapple_data.cae.courant_number` / `wall_distance` and
  their `mesh_*` wrappers; `simulation_metadata.openfoam.read_dat` / `parse_dat` for function-object tables;
  `lpm.Design.ellipse` / `Design.square`.
- Helpers the experiment repositories used to copy: `pinneapple_analysis.uncertainty.resampling` (`block_bootstrap`,
  moving-block bootstrap after Kunsch 1989; `group_bootstrap`; `skill_score` = 1 - E|model|/E|ref| with a paired
  block, group or iid interval; blocked `jackknife`), `CalibrationSummary.from_predictions` and
  `score_against_reference` (benchmark registration, `PhysicsCase` comparison, calibration and
  `compute_physics_confidence` for several models in one call) in
  `pinneapple_analysis.verification.physics_confidence_score`, `pinneapple_neural.architectures.rom.DMDc` (DMD with
  control, Proctor, Brunton & Kutz 2016), `pinneapple_physics.metrics.rollout_relative_l2`,
  `pinneapple_data.datasets.download.fetch_cached` (URL cache with retries and exponential backoff) and
  `openradioss.read_energy_error` (energy-balance error from the engine log).
- `scripts/check_app_helpers.py` (run in CI by `tests/test_app_helpers.py`): flags new local helper modules and new
  copies of library helpers in `apps/` and `examples/`, against a baseline that can only shrink. The rule
  ("reusable logic goes in the library; apps and examples only compose library calls") is in `AGENTS.md` and
  `CONTRIBUTING.md`; the inventory of what is being moved is `docs/dev/LIBRARY_CONSOLIDATION.md`.
- Model zoo: `heatsink_natural_mlp`, the natural-convection plate-fin heat-sink surrogate. It is trained with
  monotonicity constraints so that a more conductive material or a wider heat source never raises the resistance.
  Error against 750 held-out physics-engine designs: R 0.47 % mean, 1.5 % p95. The 95 % conformal bound covers
  97.1 %. Trust report `APPROVED`: monotonicity violations are 0 % for conductivity, source width and power (limit
  1 %). The app's unconstrained surrogate had the conductivity sign wrong on 5 % of designs.
- `pinneapple_neural.trainer.monotonic`: `monotonicity_penalty` (a hinge on input-output derivatives, in physical
  units, with a margin) and `monotone_violation_share`.
- `pinneapple_design.thermal.heatsink.monotone.train_monotone`: the training recipe of the constrained heat-sink
  surrogate. It uses the same data, features and split as the app, the library `Trainer` with the penalty on
  training and random in-range designs, then an L-BFGS polish.
- Zoo infrastructure for large datasets:
  - Hub-only datasets: one `<split>.npz` per split, optionally sharded into `<split>-NN.npz`. `pp.datasets.load`
    downloads only the requested split, concatenates the shards and checks every file's SHA-256.
  - `pp.datasets.split_files`.
  - `scripts/zoo/build_assets.py --hub-only DIR` enters the Hub-only datasets, and `--heatsink-natural DIR --only
    heatsink_natural_mlp` rebuilds the new model; neither touches the published files of the other items.
  - `publish_to_hub.py --staging DIR` and `--verify --allow-unpublished`.
  - A `staging_ref` input in the publish workflow.
  - Hub cards now end with a citation section (PINNeAPPle, Zenodo DOI 10.5281/zenodo.23343062).
- Zoo model `rover_contact_grc1_viper` v1.1.0 (model and dataset), trained on the library's current physics (GRC-1
  with slip-sinkage, n1 = 0.8): held-out Fx 0.220 N, Fz 0.470 N, My 0.071 N m RMSE. v1.0.0 (trained without
  slip-sinkage) stays published and listed under `superseded`; `pp.zoo.load(name, version=...)` and
  `pp.zoo.versions(name)` (with a `SupersededModelWarning`).
- Physics guard for surrogates: `WongReeceModel.physics_config()` / `from_physics_config()` and
  `terramechanics.physics_mismatch`; the zoo card records the physics a model was trained on (`data_lineage.physics`),
  `ZooContactSurrogate.check_physics` raises a `PhysicsMismatchWarning` and the rover dashboard notes any difference.
- Field overlays: `SurfaceView(field_saturation=...)`, `dashboard.ColorBar(saturation=...)` and
  `vehicle.dashboard.render_frames(clim_percentiles=...)` (colour range from the stress the ruts actually carry).
- `scripts/zoo/build_assets.py --only NAME` rebuilds one item and keeps the published files of the others.
- Rover dashboard with a semi-transparent stress map and the zoo PINN:
  - `SurfaceView(field_glow=..., far_field=...)`: a field map can sit semi-transparently on the shaded 3-D relief
    (`field_glow=0` is a pure albedo tint, so rut geometry, grouser marks and Sun shading show through) and continue
    onto the far grid; `vehicle.viz.frame_view` and `vehicle.dashboard.render_frames` take `field_alpha` and
    `field_glow`.
  - `terramechanics.surrogate.ZooContactSurrogate` gives a model-zoo contact model the `ContactSurrogate` interface;
    `vehicle_dashboard` accepts zoo models directly and lists the model, its held-out error and its error along the
    run.
  - `examples/studio/07_lunar_rover_dashboard.py` uses `pp.zoo.load("rover_contact_grc1_viper")` and colours the
    ruts by normal stress at 0.6 opacity with a kPa colour bar.
- Colab notebooks in `notebooks/colab/`, each self-contained (the first cell installs `pinneapple>=0.7.0` and picks
  the GPU, a parameter cell sets the sizes): a first PINN with its physics checks, an airfoil surrogate from
  `pp.zoo` validated against stored OpenFOAM runs, an equal-budget benchmark, the lunar rover terramechanics
  dashboard as a GIF, and a "verify a model" notebook where the trust card catches bad predictions. `scripts/run_colab_notebooks.py` runs
  them headless (`--smoke` for tiny sizes) and checks that the committed copies carry no outputs; a CI job runs them
  in smoke mode. "Open in Colab" badges in the README and the docs (Guides > Colab notebooks).
- `pp.trust_card` (`pinneapple_analysis.trust.field_card`): physics checks of any field on a grid with NumPy only
  (finite values, PDE residual of Laplace / Poisson / heat by finite differences, Dirichlet boundary values per face,
  conservation by the divergence theorem, maximum principle, reference error, applicability against the training
  bounds), turned into the deploy decision of `TrustReport` (APPROVED / REVIEW / REJECT) and the six questions of the
  lab trust card.
- `AGENTS.md` for coding agents (what the library is, the APIs to use first, how to validate any result, layout,
  tests, what never to do), with `GEMINI.md` and `CLAUDE.md` pointing to it; `tests/test_agents_md.py` runs every
  Python block in it.
- MCP server `pinneapple-mcp` (`pinneapple_mcp`, extra `pinneapple[mcp]`, MCP Python SDK 1.x or 2.x, stdio): tools
  `capabilities`, `list_methods` / `describe_method` (with the catalog's validation status), `list_presets`,
  `solve` and `benchmark` (equal training budget) as jobs with a trust card per result, `trust_card` for any field
  or file, `zoo_list` / `zoo_predict` / `zoo_validate` for the model zoo, and `job_status` / `list_jobs` /
  `cancel_job`. Problems are JSON (a preset or a box problem whose values are safely parsed expressions);
  results are saved under a working directory and returned as paths; jobs run in a separate process with a
  wall-time limit, a memory limit and the network switched off. Added to `.mcp.json`; set-up for Gemini
  CLI, Claude Code, Cursor, Codex and a Colab cell in `docs/guides/mcp_server.md`.
- Latent surrogate with a trust card (`pinneapple_neural.architectures.rom.TrustedLatentSurrogate`): wraps a POD
  latent model (`ParametricPOD`) with three per-query checks calibrated on held-out designs — a normalised
  split-conformal bound on the relative field error (Gaussian-process spread or a bootstrap ensemble of latent
  regressors), a conformal p-value of the distance to the training designs (design envelope), and an optional
  physics residual of the decoded field — and returns the library's `TrustReport` (APPROVED / REVIEW / REJECT).
  Example `examples/latent/trusted_latent_surrogate.py` (2-D advection-diffusion-reaction, every answer checked
  against a full solve). Guide: `docs/guides/trusted_latent_surrogate.md`.
- Physics-residual false-data-injection detection for cyber-physical systems (`pinneapple_security.cps`): grey-box
  residual components (`MassBalance`, `FlowLaw`) whose constants are identified from trusted data, a per-channel
  CUSUM detector calibrated to a false-alarm budget, localisation of the broken balance, ROC AUC and episode metrics.
  Reproducible benchmark `pinneapple_security.cps_bench` (three-tank water network, six attacks, data-driven
  baselines: autoregressive twin, PCA Q statistic, isolation forest; nominal and shifted operating regimes).
  Example `examples/security/02_physics_residual_fdi.py`. Guide: `docs/guides/physics_residual_attack_detection.md`.
- Evaluation of latent-space methods and new cybersecurity applications, ranked, with a roadmap:
  `docs/dev/LATENT_AND_CYBER_EVALUATION.md`.
- Geometry library `pp.geometries` (`pinneapple_design.geometry_library`): 17 realistic parametric 3-D geometries
  for Physics-AI, each with the code that regenerates it and ready-made files. Uniform API: `list()`, `info(name)`
  (dimensions, units, parameters with ranges, sources of the real-world numbers with estimates flagged, frame,
  physics uses), `build(name, **params)`, `load(name, format, lod)`, `export(...)`, `render(...)`.
  - First entries wrapped from the existing models (their classes keep working): A320-class airliner, road car,
    Formula car, two-stage launcher, NASA VIPER.
  - New: NREL 5-MW wind turbine (published chord / twist table), ATR 72-class turboprop, H145-class helicopter,
    F450-class quadcopter, KCS-class container ship, Wigley hull, high-speed train, Falcon 9-class launcher,
    pin-fin heat sink, closed centrifugal impeller, CAARC tall building with an urban block, human aortic arch.
  - Every body closed and outward (watertight, no self-intersections), metres, documented frame, two levels of
    detail; `Geometry.solid()` fuses the bodies into one watertight surface (manifold3d).
  - Formats: binary STL, OBJ, glTF, OpenUSD, STEP (faceted B-rep that OpenCASCADE reads back as solids).
  - Light files committed in `data/geometries/`; every file of both levels on the Hugging Face Hub
    (`PINNeAPPle-Labs/geometry-library`) with SHA-256 in a manifest; the loader downloads on demand into a cache
    and falls back to `build()` offline. The `geometries` job of `publish-zoo.yml` rebuilds, verifies, uploads.
  - Software renderer (no GPU) for thumbnails, a contact sheet and a turntable GIF; gallery page
    `docs/guides/geometry_library.md`; examples: CAARC building wind with OpenFOAM through the STL path, BEM power
    curve of the 5-MW turbine, vortex-lattice and friction drag of the airliner.
- Positional astronomy and astrophysics (`pinneapple_physics.astro`): JPL SPK reader (DE421), time scales, IAU 2006
  precession and nutation, apparent places with light time, deflection and aberration, refraction, airmass and
  extinction; rise/transit/set, twilight, Moon phase, magnitudes, oppositions, elongations, lunar and solar eclipses,
  transits; SGP4 satellite passes with magnitudes; Yale BSC5 stars with luminosity, radius and HR diagram; Kepler
  elements, N-body integrators (leapfrog, Yoshida-4, RK4, DOP853) with conservation checks; a physics-informed orbit
  surrogate with conformal bands and a trust verdict; black-hole lensing of the real sky. Validated against USNO NOVAS,
  JPL Horizons, the USNO almanac, NASA eclipse predictions and skyfield (`validation.run_all()`). Extra `sky`.
  Docs: `docs/core_concepts/astro.md`.
- PINNs and machine learning in the observatory (`pinneapple_physics.astro`): orbit-decay PINN with NRLMSIS 2.0
  density and an input-uncertainty envelope, checked against RK integration and SGP4 (`decay`); small-body
  propagator from JPL SBDB elements checked against JPL Horizons (Ceres) and the JPL close-approach table (Apophis
  2029) (`smallbodies`, `validation_summary()`); the orbit PINN of `surrogate.train_orbit_pinn` for any small body
  (Apophis: 0.11″ over a year); a learned propagator for many near-Earth asteroids that fails its own gate and says
  so (`neural_propagator.trust_report()`); a Kp forecaster with
  conformal intervals and the aurora probability at the observer's geomagnetic latitude, verified on observed Kp
  (`spaceweather`); clustering and classification of the HR diagram against the catalogue's MK types (`stars_ml`);
  Open-Meteo ensemble post-processing for observing conditions (`conditions`); a thin accretion disc traced along
  exact Schwarzschild null geodesics and lensed in front of the real sky (`lensing.trace_thin_disc`, `disc_image`,
  `accretion_view`). `pp.astro` points to `pinneapple_physics.astro`. In the app: decay card in Satellites, space weather, conditions, asteroids
  and close approaches, stellar ML and the black-hole panel in Analyses, planet-size control and asteroid orbits in
  the solar-system view. Optional dependency `pymsis` (extra `sky`).
- PINNeAPPle Observatory app (`apps/observatory`): the sky and the solar system now from the visitor's location in
  3-D (three.js), with satellites, astrophysics analyses, the orbit surrogate with its checks, and the validation table;
  deploy service `observatory` (`OBS_DOMAIN`). Guide: `docs/guides/observatory.md`.
- `pinneapple check`, a physics linter for CI (`pinneapple_check`; also `python -m pinneapple.check`): fails when a
  simulation result or a Physics-AI model violates physics. 29 rules with ids, explanations and fix hints
  (`--list-rules`, `--explain PC101`): conservation of mass, energy and momentum, divergence, conserved totals in
  time (PC1xx), boundary and initial conditions, wall velocity (PC2xx), PDE residual of stored fields and of a
  model's predictions (PC3xx), declared bounds, physical admissibility, non-finite values (PC4xx), units and
  parameter dimensions (PC5xx), queries outside a model's training range or in a gap of its data (PC6xx), field
  and quantity-of-interest regression against a reference (PC7xx), hashes, signatures, unsafe checkpoints and
  untracked artefacts (PC8xx), method status in the catalog (PC9xx).
  - Reads OpenFOAM cases (with `phi`, so continuity is checked on the solver's own flux), VTK/VTU, NPZ/HDF5 grids
    and point tables, CSV of quantities of interest; models as a `file.py:function`, a PINNeAPPle checkpoint with
    its architecture name (scanned before loading; training range from its metadata), a `pp.zoo` model (range from
    its card) or a hub model. Physics:
    incompressible flow, heat, linear elasticity, generic conservation law.
  - `pinneapple-check.toml` or `[tool.pinneapple.check]` in `pyproject.toml`, zero-config discovery, `--select` /
    `--ignore` prefixes, `--format text|json|sarif|junit|github|summary`, a baseline to adopt it gradually, exit
    codes 0/1/2. The JSON report carries the trust card of each target and feeds the veriphysics applicability map.
  - Reuses `pinneapple_data` readers and units, the `physics_audit` training envelope, field metrics, the method
    catalog and `pinneapple_security`, without importing torch.
  - `action.yml`: a composite GitHub Action (pinned to its own ref or a PyPI version, pip cache, SARIF upload to
    code scanning, annotations, job summary); example workflow in `examples/physics_check/workflow.yml`.
  - `examples/physics_check/`: a passing and a failing fixture for every rule (real icoFoam runs among them), used
    by `tests/test_physics_check.py` and by `.github/workflows/physics-check.yml`. Docs:
    `docs/guides/physics_checks_in_ci.md`.
  - New extra `check` (meshio, h5py, cryptography) and the `pinneapple` command (`pinneapple check ...`; it also
    forwards `pinneapple benchmark ...` to `pinneapple_analysis.benchmark` when installed).
- Rover realism (`pinneapple_simulation.terramechanics`, `.vehicle`, `pinneapple_tools.visualization.studio`,
  `pinneapple_design.geometry.vehicle_cad`):
  - Slip-sinkage calibration with held-out points: `testbed.calibrate_holdout` fits the slip-dependent sinkage
    exponent `n = n0 + n1 s` (Ding et al. 2010, 2011) and K on every other published single-wheel point and reports
    the errors on the held-out points before and after (`testbed.holdout_split`, `plot_validation(calibrated=...)`).
    `WongReeceModel(slip_sinkage="drive" | "both" | "none")`, `sinkage_exponent()`, `forces(..., sinkage_slip=...)`.
  - 3-D ruts: `DeformableTerrain` rut walls at the angle of repose (`wall_angle_deg`) and rounded berms holding the
    displaced soil (`berm`), as separate layers; the default renders show ruts by their geometry, the stress map
    is an option (`overlay=`).
  - Ground to the horizon: `SurfaceView(horizon=..., far_fn=...)` adds a curved annulus past the lunar horizon;
    `studio.planetary.border_visibility` checks that no mesh border is visible from a camera.
  - Dashboard exports at small sizes: `font_scale` with text-aware gaps, `write_video(gif_font_scale=...)` lays the
    GIF out natively at its size, `dashboard.static.write_gif` writes a palette-optimised GIF.
  - VIPER CAD from public NASA figures: chamfered body, radiator and MLI front, pan-tilt mast head, box-beam
    suspension arms with actuators, the TRIDENT drill (`vehicle_cad.drill_assembly`) and new primitives
    (`loft`, `extrude_polygon`, `chamfered_box`, `box_beam`, `ladder_rail`, `helix_sweep`, `mli_blanket`,
    `actuator`, `pan_tilt_head`); sources and estimates in `VIPER_SOURCES`.
- Model zoo `pp.zoo` and dataset hub `pp.datasets` (`pinneapple_hub.zoo`): pre-trained models and ready-made
  datasets you can load in one line. Only items that passed a quality gate ship (an independent reference, physics
  checks, a recorded validity range; for datasets, a verified solver or a public source with a compatible licence).
  - Models:
    - `airfoil_cst_rans_mlp`: CST airfoil Cl/Cd/Cm, Re 4e6 RANS. Cd 0.54 % on unseen shapes, 2.8 % against XFOIL.
    - `heatsink_forced_mlp`: plate-fin resistance and pressure drop, mean error 0.71 %.
    - `rover_contact_grc1_viper`: wheel-soil contact PINN, retrained for 16000 epochs; Fx RMSE 0.21 N.
  - Datasets: the 436-run airfoil RANS campaign with its grid study and XFOIL polar, the forced and natural heat-sink
    datasets, Wong-Reece contact forces, and published single-wheel tests.
  - `list`, `info` (a `ModelCard` with task, units, validity range, metrics vs reference, training data, licence,
    citation, trust report), `load`, `predict` (with an out-of-range warning), `evaluate` (reproduces the card's
    metrics on the stored reference), `validate` (a `TrustReport` with physics checks), `benchmark` (through
    `pp.benchmark`, against the same network trained from scratch) and `finetune`. `finetune` refuses to save a
    model that is worse than the original on the original test set unless `force=True`.
  - Weights and datasets are listed in `pinneapple_hub/zoo/manifest.json` with SHA-256 and versions. Small files are
    bundled (2.5 MB); the rest is downloaded lazily from the Hugging Face Hub (`PINNeAPPle-Labs/<name>`) and cached.
    Errors are explicit for offline use, checksum mismatches and items not published yet.
  - Workflow "Publish zoo to the Hugging Face Hub": it uploads everything when the `HF_TOKEN` secret is set, and
    skips with a warning otherwise.
  - Supporting material: an inventory of every candidate model and dataset and its verdict
    (`docs/dev/model_dataset_inventory.md`), a guide (`docs/guides/model_zoo.md`) and
    `examples/model_zoo/quickstart.py`.
- `ModelCard` has optional `task`, `version`, `inputs`, `outputs`, `validity_range` and `known_limitations` fields,
  which are rendered in `to_markdown()`.
- Two engineering hand-off case packages in `examples/handoff_cases/` (each also as a zip): `airfoil_ld` (max L/D of a
  2-D airfoil, OpenFOAM simpleFoam k-omega SST, CST geometry, the 436-run campaign, baseline against XFOIL, a
  surrogate optimum checked in CFD) and `free_surface_trough` (water jet into an open trough, interFoam VOF via
  `pp.cfd.FreeSurfaceFlow`, two meshes with mass balance, impingement, splash, floor shear and mean k).
- Aero optimizer logic in the library: `pinneapple_design.aero.train` (load OpenFOAM runs, split by shape, metrics,
  MLP-ensemble and MeshGraphNet training/prediction, `train_surrogates` -> the `Engine` bundle),
  `pinneapple_design.aero.dataset` (run plan, `run_job`, `steady_values`, the one averaging rule for unsteady runs),
  `pinneapple_design.aero.verify` (Pareto picks checked in OpenFOAM), `pinneapple_design.aero.case.case_zip` (a
  ready OpenFOAM case with a stand-alone `COEF_SCRIPT`), `pinneapple_design.aero.optimize.infeasibility_reasons`
  and `pinneapple_design.design_optimizer.representatives` (best per objective + balanced pick of a Pareto front).

### Changed
- **Behavior:** `BayesianDesignOptimizer` scores its acquisition on all candidate points with one GP prediction (same
  values, several times faster) and seeds the GP hyper-parameter restarts with its own seed, so a run is reproducible.
- **Behavior:** `EvolutionaryDesignOptimizer` in multi-objective mode now draws parents by the NSGA-II
  crowded-comparison tournament (front rank, then crowding distance) instead of by the first objective, so the
  population spreads along the front.
- `pp.benchmark`: adding entries to `budget_study` or `ablations` reuses an existing `out_dir` (they only add jobs); changing
  or removing an entry is still refused. Before, any change to them was refused.
- **Behavior:** `pinneapple_analysis.cost.measure` now counts FLOPs of code that calls `torch.autograd.grad` with
  the hook-free counter (`flops_source == "dispatch_counter"`) instead of the profiler's matmul-only lower bound.
- `compressible_fv.sod_exact` delegates to `astrophysics.sod_exact_solution` (same output to round-off; covers every wave pattern);
  `internal_flow.colebrook` delegates to `closed_form.pipe_flow.colebrook`; the LBM shedding benchmark, the
  cylinder / car / ROM lab experiments, the SSQN benchmarks and the missile example use the library functions above.
- Examples, benchmarks, scripts and a notebook use the library helpers instead of local copies: autograd
  `grad`/`laplacian`/`divergence`, `pinneapple_physics.metrics`, `pinneapple_lab.uq.gci`, `latin_hypercube`,
  `use_cfd_style(dark=True)`, `compressible_fv.sod_exact`, the `pinneapple_data.cae.foam` readers, `set_seed`,
  `best_device` and `load_leaderboard` (20 lines fewer in `scripts/app_helpers_baseline.txt`).
  `heat_fv_reference.footprint_area` is public (`_footprint_area` kept as an alias) and `interp_grid`/`convergence`
  are in its `__all__`; the `fin_convection_inverse/fv_reference.py` shim is removed.
  `examples/architectures/04_pinns.py` uses the real `PINNFactory` instead of stubs that ignored the PDE residuals.
- **Behavior:** the zoo gate of `rover_contact_grc1_viper` checks that dFx/ds has the sign of the recorded Wong-Reece
  reference (with slip-sinkage the pull need not grow with slip at a fixed sinkage) instead of dFx/ds >= 0.
- **Behavior:** GRC-1 and JSC-1A carry n1 = 0.8 (estimate from the five single-wheel calibrations; was 0), so their
  sinkage grows with driving slip. The validation soils keep n1 = 0. In `VehicleSimulator` the slip that sets the
  exponent lags the wheel's slip over `sink_length` (0.15 m) of rim travel.
- **Behavior:** `vehicle.dashboard.render_frames` and `vehicle_dashboard` default to the geometry-only view (no
  stress colour map, no colour bar); pass `field=` / `field_label=` for the overlay.
- `apps/aero_optimizer`: `tools/train_surrogate.py`, `tools/generate_dataset.py` and `tools/verify.py` are thin CLIs
  over the library; the API uses the library picks, reasons and case zip (same responses);
  `tools/render_airliner.py` uses the studio `colorbar` (tick labels now follow the viewer's format) and
  `aero.render_blender` the studio `jet`.

### Fixed
- `eddy_current_fdm`: `annular_current_source` injected a current that depended on how the coil band fell on the
  grid (it normalised by the nominal band area; 0.75 N I in one case); `axial_flux_density` returned half the field
  on the axis; `eddy_current_density` had the wrong sign for the solver's `exp(+j omega t)` convention. Together they
  explain the coil field at a fraction of the loop formula recorded in the validation gallery.
- `solve_magnetostatic` (electromagnetics): the Newton line search could stall at a 1e-7 residual when the Armijo
  test compared energy changes below round-off; near convergence it now takes the full step.
- `pinneapple check`: `FV.patch_sums` (per-patch flux and heat-flow totals of the balance checks PC101/PC103) read
  the boundary arrays with the internal-face offset and returned 0 for every patch; the net balance was right.
- `pinneapple_analysis.inverse_problems.heat_transfer.__all__` listed an undefined `FinInput`, so
  `from ... import *` failed.
- App job queue (`pinneapple_core.service.jobs`): the final log line of a job is written before its final status, so
  a caller that sees `done`, `failed` or `cancelled` also sees the whole log (a race made
  `test_job_runs_with_progress_logs_and_workdir` fail intermittently).
- Form Compiler (`pinneapple_data.formfill`): vendor datasheets laid out as one grid now compile. A unit in its own
  column ("Design pressure | 1.5 | MPa g") is attached to the value; a label naming a pair ("(Int/Ext)", "Max/Min")
  splits "1.5 / NA" or "60 / 10" into both items and drops the "NA" part; a nozzle schedule printed beside the other
  data is read as the nozzle table and its header is no longer taken as an answer (it filled Service with "Qty");
  "label: value" inside one cell, "Shell inside diameter" (longest label wins), notes such as "NB registration is not
  required" and "Wind loads shall be as per UBC97" are read; a bare yes/no is no longer a text answer. New U-DR-1
  labels: TL-TL length, temperature (int/ext), corrosion allowance (internal), operating "internal pressure" and
  "temperature max/min", location, supports; `Field.reject` lists values that are not an answer to an item.
- `pp.validate` on Darcy audits the training data with the problem's own source `f` (it assumed `f = 1`), and
  records callables in the provenance by name (a rollout spec with an invariant function could not be saved).
- Rollout validation: a user invariant named like a built-in (`energy`) is no longer replaced by the built-in or
  scaled like it; the spectral band ignores shells below 1e-6 of the peak energy (round-off, not content).
- `pp.validate` runs PhysicsNeMo field models given without a channel axis (PhysicsNeMo raises `ValueError`).
- MC dropout (`pinneapple_analysis.uncertainty.mc_dropout`): `MCDropoutWrapper` applied dropout after every
  `nn.Linear`, including the output layer, so about `p` of the samples were exactly zero and the reported std was
  `|mean| sqrt(p / (1 - p))` whatever the model. Dropout now acts on the input of every `nn.Linear` but the first
  (the hidden activations, Gal & Ghahramani 2016), and the sampled mean and std match the closed-form moments of a
  two-layer network. Sampling no longer leaves the model in eval mode or resets the caller's global torch RNG when a
  seed is given, and `n_samples = 1` (a NaN std) now raises a `ValueError`.
- Quantile regression (`pinneapple_analysis.uncertainty.quantile`): `QuantileLoss` scored the first `Q` outputs
  when the model produced more outputs than levels and raised a bare `IndexError` when it produced fewer; the pinball
  loss broadcast a `(B, 1)` target against `(B, H)` predictions; a level outside (0, 1) such as 1.5 was accepted and
  gave a loss that rewards over-prediction without bound. All of these now raise a clear `ValueError`, as do empty
  or repeated levels in `QuantileConfig` and `QuantileLoss`.
- Encounter feasibility (`pinneapple_analysis.uncertainty.reachability`): contact was looked for only at the
  `n` time samples, so a fast flyby whose regions overlapped between two samples was reported as impossible (a
  closest approach of 0.05 against a radius of 0.1 at 2000 units per second came out `possible=False`). The closest
  approach of the two centres is now sampled, and every interval whose lower bound reaches zero is searched. A
  negative uncertainty, acceleration or latency, or a non-positive horizon, now raises a `ValueError`.
- Veriphysics robustness studies (`pinneapple_veriphysics.robustness`): an extrapolation margin of 0 (or a negative
  one) left no point outside the training box and `_sample_shell` looped forever, hanging `ensemble_study` and
  `extrapolation_study`; it now raises a clear `ValueError`. The evaluation points came from the global NumPy
  random state, so the same study gave different evidence on every run; sampling is now seeded and reproducible.
- Zoo query checks (`pp.zoo.query_checks`, the Playground's trust panel): the range checks run in float32 (the
  torch module and its ONNX graph), but the bounds had a slack of only 1e-9, so a query exactly at the edge of the
  training range was rejected as "outside the validity range". `T1 = 0.08` on `airfoil_cst_rans_mlp`, a value in
  its training data, arrived as 0.0799999982 and failed PC601. The slack now covers two float32 epsilons; the
  browser bundles in `data/web/zoo/` carry the new limits.
- `pinneapple check`: a grid with 2 points along an axis made `numpy.gradient` raise "Shape of array too small" in
  the residual and the balances, and the run ended with a traceback, so no other target was reported. Such an axis
  now uses first-order ends, and a residual with no interior point is noted as not evaluated instead of passing on
  an empty set. A check that still raises is reported as a PC001 error on its target, and the run goes on with the
  remaining targets.
- CalculiX Docker image (`pinneapple_simulation.external_solvers.calculix.runner.ensure_image`): every build attempt
  pulled the Debian base from public.ecr.aws, which answers 429 ("Data limit exceeded") on busy shared CI runners,
  so the retries failed the same way and the real-CalculiX test turned unrelated pull requests red. Each retry now
  takes the next registry serving the same official image (`BASE_IMAGES`: public.ecr.aws, mirror.gcr.io, Docker
  Hub; identical digest); `PINNEAPPLE_CALCULIX_BASE` puts another base first.
- `pinneapple_physics.conservation.project_integral(mode="positive")` returned a negative field when the target
  integral was negative, breaking the non-negativity the mode exists for; it now raises a clear `ValueError`, also
  for a field that is <= 0 everywhere. A zero field with a zero target is returned unchanged in the multiplicative
  modes instead of dividing 0 by 0.
- `TrustGate` OOD score (`pinneapple_analysis.trust`): a batch of query points was scored by the chi-squared p-value
  of its mean point, so two points 50 standard deviations on either side of the training mean averaged to the mean
  and scored 0.9998, "in distribution". **Behavior:** each point is now tested on its own and `ood_score` is the
  fraction of query points whose Mahalanobis p-value is at least `ood_level` (new argument, default 0.01: inside the
  99 % region of the fitted Gaussian). `TrustGate.ood_p_values(x)` returns the per-point p-values; NaN coordinates
  count as outside and a wrong number of coordinates raises a clear `ValueError`. `TrustScore.details` reports
  `ood_fraction_inside`, `ood_p_value_min` and `ood_level` (replacing `ood_p_value`).
- `pp.trust_card` / `field_checks`: an exactly harmonic field that is linear along an axis (`u = x + 2y`,
  `T = 300 + x + y`) failed the critical PDE-residual check and was rejected, because its finite-difference terms are
  pure round-off and the residual was judged against that round-off. The residual scale now has a floor at the
  round-off of the stencil, computed from the dtype of the data as written (float32 fields get float32 epsilon).
  Grids with 2 points on an axis now report the conservation check as not run instead of raising NumPy's
  "Shape of array too small" error, and torch tensors (including ones that require grad) are accepted as values.
  New tests check the residual against exact Laplace, Poisson and heat solutions (second-order convergence on
  uniform and stretched grids) and the conservation check against the divergence theorem.
- `pinneapple check` on tensor-product grids (`pinneapple_check.numerics.GridD`): second derivatives were the
  derivative of the derivative, which next to the boundary divides the one-sided edge value by h (an O(h) error),
  so the PDE residual (PC301) converged at order 1.5 instead of 2 and an exact harmonic field on a 17 x 17 grid
  scored 0.095, above the 0.05 default. Pure second derivatives now use the three-point stencil (non-uniform
  spacing allowed); the residual of the same field is 0.017 and converges at order 2 on uniform and stretched grids.
- Playground: the "Model on the Hugging Face Hub" link opened a 404 page. `pp.zoo.export_web` now takes the Hub
  repository from the zoo manifest (`hf_repo`, e.g. `PINNeAPPle-Labs/airfoil-cst-rans-mlp`) instead of the model name.
- Black-hole simulations: resuming a run from its checkpoint failed on the outer-boundary state (`AccretionFlow._outer`).
- Observatory, black-hole lensing panel (Analyses): the render came out almost black, with only the shadow and a
  grey ring. `pinneapple_physics.astro.lensing.catalog_sky` now draws each Yale BSC star with a magnitude-dependent
  footprint (`star_splat`: width grows with brightness from 0.1 deg at V 6.5, so faint stars never fall between
  pixels; peak ~ sqrt(flux)) and a brighter, tinted Milky Way; `tone_map` replaces the app's ad-hoc exposure.
  `lensed_sky` now traces with the exact Binet tracer of `trace_thin_disc` and takes the asymptotic escape direction
  in closed form: the old trace stopped at 1.5 r_cam and used the local tangent, which left concentric banding and
  under-deflected rays. The deflection now matches the weak-field value (2M/b)(1 + cos θ) + 15πM²/(4b²) within
  1.5 % (new test). New `lensing_view` (used by `/api/analysis/lensing.png`) adds the thin accretion disc of
  `accretion_view` when the 60° field has fewer than 2 stars brighter than V 2.0 (`bright_stars_in_field`), with a
  disc auto / on / off control in the panel; the default camera distance is 120 GM/c². Screenshot
  `docs/assets/apps/14-observatory-lensing.jpg` regenerated with the hole in front of Sagittarius.
- Observatory, solar-system view: the planets were a few pixels across. The planet-size slider now defaults to ×12
  (range 1–24, `?size=` in the URL), the glow halo is brighter, and the Earth follows the slider (its hand-set
  orientation matrix used to drop the scale). The Moon's exaggerated distance and the observer pin scale with it,
  so the Moon stays outside the enlarged Earth. Screenshot `docs/assets/apps/14-observatory-system.jpg` regenerated.
- Observatory, solar-system view: the overview now frames every planet inside the free part of the canvas (left of
  the side panel, above the time controls; Jupiter used to sit under the panel). It looks from 50° above the
  ecliptic, from the side opposite the Earth, and fits the projected planets by iteration. The Earth shader lacked
  the linear-to-sRGB output conversion, so the Earth rendered several times too dark; it now has it, plus the blue of
  the sunlit atmosphere on the day side. The Sun's light and an ambient floor were raised so that Mercury and Venus
  show their sunlit side and their surface at large phase angles. Screenshot
  `docs/assets/apps/14-observatory-system.jpg` regenerated.
- `pinneapple_data.preflight`: `freestreamPressure` / `freestream` fix the pressure level; external-aerodynamics cases
  were reported as having no pressure reference.
- Geometry library: `build()` snaps vertices to a 0.1 um grid, so the files rebuilt on another machine (the Hub
  publishing job) are bit-identical to the manifest; the publishing job pins the NumPy / SciPy / manifold3d
  versions the manifest was generated with. All files and the manifest were regenerated. The self-intersection
  check measures distances about each triangle pair's centre with a size-relative tolerance (no false hits on flat
  caps far from the origin or on nearly parallel neighbours).

### Documentation
- Validation methods reference: measured behaviour of `crps_gaussian` / `crps_ensemble` (fair score unbiased for
  small ensembles) and of `block_bootstrap` (percentile intervals undercover: 0.93 at a nominal 0.95 for iid n = 100,
  0.90 for a strongly correlated AR(1) series), and first-order residuals of `pinneapple check` on point clouds.
- New guide "Validation methods reference" (`docs/guides/validation_methods_reference.md`, under Guides): every
  check of `pinneapple check`, `pp.trust_card`, `TrustGate`, `pinneapple_analysis.validation`, `.uncertainty`,
  `.verification`, veriphysics, the physics audit monitor and the zoo query gate, with what it detects, its
  assumptions, default thresholds, references and known limitations; a table that maps the status vocabularies of
  these modules onto pass / warn / fail / not-applicable; and how check power is measured (ASME V&V 40 and
  NASA-STD-7009A credibility, mutation testing, metamorphic relations, adversarial robustness of neural operators,
  the arena proposed in #707).

## [0.7.0] - 2026-10-13

### Added
- Animation-centred dashboards for any simulation (`pinneapple_tools.visualization.dashboard`, `pp.viz.Dashboard`):
  an animation in the middle, charts synced to its time cursor (time series with a cursor, or fixed curves with
  markers that move with the frames), KPI tiles, a parameter panel and a colour bar.
  - Exports: one self-contained interactive HTML file, where scrubbing a chart moves the animation; a static PNG;
    a dashboard GIF/MP4.
  - The HTML embeds VP9 WebM and H.264 sources, reuses the studio viewer's `studio-core.js`, and has an optional
    live three.js view (the studio's vendored three.js, lunar lighting preset).
  - `pinneapple_simulation.vehicle.dashboard` builds it for vehicle runs: drawbar pull, torque and sinkage versus
    slip with live wheel operating points; per-wheel sinkage and load along the traverse; speed, slip, and the
    PINN against Wong-Reece with its error.
  - Example: `examples/studio/07_lunar_rover_dashboard.py`, which writes `docs/assets/studio/rover-*`. Docs:
    `docs/core_concepts/dashboard.md`.
  - Planetary renders: the vehicle's shadow now continues onto the far-field terrain beyond the detailed patch
    (`studio.shadows.cast_shadow(..., supersample=n)` gives a smooth shadow fraction on coarse grids, and
    `shade_terrain` accepts it); `vehicle.viz.frame_view(trail=...)` shifts the detailed window back along the path.
- Realistic planetary rover simulation, as library features (`pp.terramechanics`, `pp.multibody`, `pp.vehicle`):
  - `pinneapple_simulation.terramechanics`: a soil database where every parameter records its source and how far
    it was checked (GRC-1, JSC-1A, the soils of three published single-wheel test series); the Wong-Reece wheel
    (Bekker pressure-sinkage, Janosi-Hanamoto shear) with slip-sinkage, grousers (equivalent shear radius, Ding et
    al. 2011), multi-pass loading (Wong's unloading-reloading, prior shear), bulldozing and lateral force (Ishigami
    et al. 2007); a deformable terrain with ruts, grouser marks, contact-stress maps and rigid rocks; a single-wheel
    testbed with published points bundled; PINN training for wheel-soil contact with a held-out error report.
  - `pinneapple_simulation.multibody`: floating-base revolute trees with linearly implicit steps (checked against
    free fall, a pendulum period and momentum and energy conservation).
  - `pinneapple_simulation.vehicle`: vehicle description, multibody model and simulator (suspension, steering,
    wheel inertia, motor torque limits, speed control, load-levelling suspension), scenarios, renders and
    animations; `viper()` is NASA VIPER at real scale from cited public figures, with the estimates listed.
  - `pinneapple_design.geometry.vehicle_cad`: parametric vehicle CAD (wheels with rims, spokes and grousers or
    chevrons, panels, radiators, dishes, camera heads), centre of mass and inertia from the meshes, glTF/STL.
  - Studio: lighting and sky presets for PyVista, Blender and the browser viewer (`studio.lighting`, black lunar
    sky with a low Sun), heightfield Sun shadows and Lunar-Lambert regolith shading (`studio.shadows`), planetary
    surface renders with field colour maps (`studio.planetary`), and new PBR materials.
  - Example: `examples/use_cases/terramechanics/lunar_rover_3d.py` is now a thin script over these features. Docs:
    `docs/core_concepts/lunar_rover_terramechanics.md`.
- Lunar rover terramechanics in 3D (`pinneapple_simulation.rover`): a parametric six-wheel rocker-bogie rover
  (glTF/STL export) on lunar terrain (grade, ramp, craters with Pike's 1977 depth ratio, rocks, soil regions) with
  Bekker-Wong wheels. Each step solves a quasi-static equilibrium of body, rockers, bogies and differential, with
  longitudinal motion in speed or slip mode. Outputs per wheel: sinkage, slip, drawbar pull, load, torque and
  power; per rover: attitude and energy per metre. The maximum grade comes from bisection.
  - Verification: Bekker's closed form, the standalone solver, the balance laws, the drawbar-pull grade estimate,
    and surrogate errors with range flags.
  - Example: `examples/use_cases/terramechanics/lunar_rover_3d.py`. Docs:
    `docs/core_concepts/lunar_rover_terramechanics.md`.
  - `pinneapple_twin3d.Scene.add_motion`: rigid part motion per time step, played by the bundled viewer.
- Faster implicit pressure solves in OpenFOAM (`pinneapple_simulation.accelerated_foam`): `pinneWarmStart`, an
  OpenFOAM linear-solver wrapper (C++ source in the package, built against the installed OpenFOAM by
  `build.build_plugin`) that projects each pressure system on recent solutions (or takes a guess from Python
  through named pipes) before calling OpenFOAM's own PCG/GAMG, and can dump every system; readers to SciPy/PyTorch
  sparse matrices, CG with Jacobi/DIC/AMG and mixed precision at OpenFOAM's stopping test, POD and learned
  warm starts, and benchmarks against plain OpenFOAM. `FreeSurfaceFlow` takes `pressure_solver=` and `libs=`.
  Measured results (including where it does not help): `docs/core_concepts/openfoam_acceleration.md`.
- Fit-for-purpose geometry and mesh (`pinneapple_design.fitmesh`): reads a problem description (English or
  Portuguese) and asks for what is missing; suggests reference geometries with their published sources (NACA wing,
  Ahmed body, cylinder, sphere, 90-degree pipe bend, backward-facing step, plate with a hole, cantilever beam,
  plate-fin heat sink), parameters and ranges; builds STEP/STL variants checked against exact volumes and the
  reference objects; plans the mesh for screening, design or verification (y+ and prism layers from skin friction,
  domain, cell and cost estimate, mesh family); generates Gmsh and snappyHexMesh meshes with quality from
  `pinneapple_data.cae`; computes the grid-convergence index; writes one JSON + HTML report with a verdict per
  stage. Optional language-model help at every step (local Ollama or hosted) with validated outputs and isolated
  execution of model-written CadQuery. HTTP API in `fitmesh.api`. Docs: `docs/core_concepts/fitmesh.md`.
- astroML bridge (`pinneapple_physics.pde_environment.presets.astroml_bridge`): Lomb-Scargle and multi-term
  periodograms for unevenly sampled time series with Scargle false-alarm levels and a bootstrap, `best_period`,
  extreme deconvolution (`XDGMM`) with per-point errors, Bayesian Blocks and Knuth histograms, line fits with errors
  on y (WLS) or on both axes (`TLS_logL`), opt-in `astroML.datasets` loaders, and `validate()`, four synthetic
  offline checks (period recovered to 5e-5, XD means, a step change point, line slopes). The module imports without
  astroML. Optional dependency: `pip install "pinneapple[astroml]"`. Docs: `docs/core_concepts/astroml.md`.
- NVIDIA Newton rigid-body backend (`pinneapple_simulation.external_solvers.newton_physics`): `RigidScene` (sphere,
  box, capsule, cylinder and mesh bodies, revolute and fixed joints, ground with friction and restitution),
  `simulate()` with the XPBD, semi-implicit, Featherstone or MuJoCo-Warp solvers on CPU or GPU, `RigidTrajectory`
  (energy, UPD export, torch `RigidBodyState`) and `validate()`, six closed-form checks (free fall, Coulomb sliding,
  pendulum period, restitution, torque-free spin, resting contact). XPBD passes all six on CPU at 1 ms. Optional
  dependency: `pip install newton`. Docs: `docs/core_concepts/newton_rigid_body.md`.
- Lattice Boltzmann wind tunnel (`pinneapple_simulation.numerical_solvers.lbm_tunnel`), in PyTorch on CPU or GPU:
  D3Q19/D3Q27, SRT or TRT collision, Smagorinsky LES, DDF shifting with optional FP16 storage (half the memory),
  Guo forcing split for TRT, halfway bounce-back with moving walls, momentum-exchange forces, inlet/outlet/open/wall/
  moving/periodic faces, STL voxelization, SI unit conversion and legacy VTK I/O. Validated on Poiseuille flow, Stokes
  drag and sphere drag at Re = 20 (`examples/lbm/01_sphere_wind_tunnel.py`).
  FluidX3D bridge (`external_solvers.fluidx3d`): writes a wind-tunnel `main_setup()` and the needed defines into the
  user's own FluidX3D checkout, runs it, and reads forces and `.vtk` fields into arrays or PhysicalSamples. FluidX3D
  is not bundled, and its license forbids commercial and military use.
  Docs: `docs/core_concepts/lbm_wind_tunnel.md`.
- OpenFOAM parameter campaigns (`pinneapple_simulation.external_solvers.openfoam.campaign`): one base case run
  with many inputs, from a JSON plan or from Python. Per run: dictionary edits with values computed from the
  parameters (arithmetic only), a mesh swap, a patch rebuilt to a given area, an optional hook, decomposition
  (scotch or simple) and MPI. Lines are sampled while the solver runs, and solver health is read from the log.
  Machines split the runs by worker tag or shard, claim runs with lock files, skip finished runs, and can push
  each result to git. CLI: `python -m pinneapple_simulation.external_solvers.openfoam.campaign`. Example:
  `examples/openfoam_campaign/`. Docs: `docs/core_concepts/openfoam_campaigns.md`.
- Geomechanics MPM (`pinneapple_simulation.particle_dynamics.geo_mpm`): MLS-MPM in physical units (2-D or 3-D)
  with Drucker-Prager sand (plane strain in 2-D), a weakly compressible fluid and an elastic solid, Coulomb walls,
  several bodies, frames in GeoTaichi's layout. GeoTaichi bridge (`external_solvers.geotaichi`): runs GeoTaichi
  scripts as a separate process and reads its MPM/DEM frames into PhysicalSamples. Example: granular column
  collapse against Lube et al. (2005). Docs: `docs/core_concepts/geomechanics_mpm.md`; remaining features: #471.
- Modelica to Physics AI (`pinneapple_simulation.external_solvers.modelica`): `omc.build_fmu` compiles `.mo` files
  into FMI 2.0 FMUs with OpenModelica (local omc or the official Docker image, no OMPython), `omc.simulate_mo`
  simulates with parameter overrides, `sweep.fmu_sweep` runs Latin-hypercube sweeps into an `FMUDataset` (split by
  run, save/load). Examples `examples/modelica/`: FMU against omc, a ladle-cooling surrogate tested on unseen runs
  (melt RMSE 0.8 K against 19 K for the nearest training run), and a PINN identifying a discharge coefficient from
  noisy readings. Docs: `docs/core_concepts/modelica.md`.
- Free-surface flow (`pp.cfd.FreeSurfaceFlow`, `pinneapple_simulation.numerical_solvers.free_surface`): interFoam
  VOF cases with inlets given as velocity vectors, open boundaries, an initial liquid level (gravity may be tilted),
  k-omega SST, and `LiveLineSampler`, which samples lines while the solver writes and can delete the sampled times
  (long transients without function objects or a full disk). `pinneapple_design.geometry.trough.TroughSection`:
  2-D melt-runner section with a weir and a slope. Example `examples/free_surface/01_runner_slope_study.py`.
  Docs: `docs/core_concepts/free_surface.md`.

- `pinneapple_twin3d.wear`: a generator of 3D wear (refractory / liner) digital twins from a spec. `ZoneSpec` /
  `VesselSpec` describe a part as surfaces of revolution cut into cells; `WearDataset` holds wear depth per cell and
  reading (validation, CSV and JSON contract shared with the web suite); `forecast` (repair-aware linear remaining-life
  fit with a confidence grade), `backtest`; `optimize` (simulates run-to-limit, fixed-calendar and predictive grouped
  maintenance policies and recommends the cheapest one without breach); `synthetic_campaign` (clearly labelled synthetic
  data); four presets (steel ladle, pig-iron ladle, BOF converter, RH degasser) and `export_wear_twin`
  (viewer scene + `wear_dataset.json` + `report.json`). Guide: `docs/guides/recreate_a_wear_twin.md`; example:
  `examples/use_cases/refractory_wear_twin/`; tests: `tests/test_twin3d_wear.py` (Python checked against the JavaScript core).
  Also: a fifth preset (`smelting_reduction_reactor`), a 90 % interval on the remaining life (`remaining_lo/hi`), `calibrate` (optimizer
  assumptions from a stop log and the repair history, each parameter tagged calibrated / default / business input),
  `pinneapple_twin3d.primitives` (revolve, sweep, loft, wing, box, cylinder) and `pinneapple_twin3d.gallery` (runner channel,
  mixing tank, oil pipeline, airliner, launch vehicle, drone, lunar rover, wind turbine, satellite; illustrative fields).

- Formula car (`pinneapple_design.geometry.vehicles3d.FormulaCar`): parametric open-wheel car (tub, floor and
  diffuser, NACA-section front and rear wings with a DRS flap, helmet, halo, wheels) with per-component CFD bodies and
  a rasterised frontal area. Example `examples/studio/06_formula_car_cfd.py`: OpenFOAM half model with a moving road
  at 70 m/s, DRS closed and open, per-component forces, skin Cp, streamlines, mid-plane and wake slices, viewer,
  Plotly and Rerun. `render_scene` gains `slice_crop`, and long colour-bar titles move to the corner text.
- Proof graphs (`pinneapple_analysis.proof`): mathematical arguments as graphs of assumptions, lemmas, theorems and
  conjectures with exact (symbolic, exhaustive, integer/degree) and numerical checks (random tests, counterexample
  search); statuses PROVED / ARGUED / SUPPORTED / REFUTED / OPEN propagate along dependencies; report, Mermaid and
  figure. Experiment `examples/proof_graph/pt_chain_no_go.py`: the no-go theorem for linear exceptional-point tuning
  extended to PT tetramers with independently modulated bonds (ARGUED: every algebraic step machine-checked for all
  degrees, one continuity step written), N >= 5 supported numerically. Docs: `docs/core_concepts/proof_graphs.md`.
- Viewer bridges (`pinneapple_tools.visualization.integrations`): Rerun (scenes, molecules, time series, ensemble
  runs, proof and knowledge graphs, reports; live or `.rrd`), Plotly standalone HTML (scenes, molecules, 3-D
  fields), Trame web app around PyVista. Extra `viz3d`.
- Digital-twin standards (`pinneapple_systems.digital_twin.standards`): Asset Administration Shell v3 (Eclipse
  BaSyx), DTDL v3, NGSI-LD, Eclipse Ditto and a live OPC UA server, each carrying the physical validity of the
  twin's models. Extra `twin`. Docs: `docs/core_concepts/viewers_and_twin_standards.md`.
- Molecular visualisation (`pinneapple_tools.visualization.molecular`, `pp.molecular`): PDB / mdtraj molecules, fcc
  metal slabs (8 metals, (111)/(100)/(110)) sized to a footprint, orientation on a surface (principal axes, active site
  up), off-screen stills (space-filling or backbone, coloured by residue class, element, chain or height), trajectory
  frames and MP4, MD dashboard (energy, temperature, RMSD, height, contacts), glTF scene and a step-by-step console
  log. Example `examples/molecular/protein_on_gold.py`: a protein on Au(111) with OpenMM (Amber14 + GBn2, INTERFACE
  gold Lennard-Jones as a Steele wall). Docs: `docs/core_concepts/molecular_visualization.md`.
- Non-Hermitian physics (`pinneapple_physics.non_hermitian`, `pp.non_hermitian`): PT-symmetric dimers and chains,
  exceptional points in closed form and from the matrix alone, eigenvector coalescence, Petermann factor; a symbolic
  check of exceptional-point tuning laws against the no-go theorem for PT-symmetric dimers (Zenodo 23287622) with
  coupling classification, the three routes around it (square-root coupling, feedback, non-reciprocal hopping) and
  the theorem's hypotheses as a physics-audit profile; figure and docs (`docs/core_concepts/non_hermitian.md`).
- `fmi` extra (`fmpy`, `pythonfmu`) and a test that builds an FMI 2.0 Co-Simulation FMU and simulates it through
  `simulate_fmu` / `fmu_to_upd` (matches the exact damped-oscillator solution to 1e-5).

- Physics audit (`pinneapple_physics.physics_audit`, `pp.physics_audit`): which model to use, when to trust it and
  when its physical formulation must be revisited. Assumption auditor (catalogue of common assumptions with soft
  validity ranges on dimensionless numbers, training envelopes with nearest-neighbour distance, `UNKNOWN` instead of a
  silent pass), physics validity monitor without ground truth (residual, conservation, bounds, boundary, constitutive,
  invariant and envelope checks; alarms on critical failures, persistence and residual drift), physics-aware routing
  (compatibility prior on the ensemble weights, sleeping experts, escalation to the solver), failure attribution
  (regime change, extrapolation, PDE / conservation / boundary violation, numerical resolution, measurement, missing
  physics, with evidence and confidence), a knowledge graph of phenomena, equations, assumptions, mechanisms and
  models, and ranked improvement recommendations; `diagnose()` writes a report. Example
  `examples/physics_audit/pipe_regimes.py`: across laminar, transitional and rough turbulent pipe flow the routed
  ensemble has 0.38 % mean error against 1.49 % for history-only weights, and a low-residual, in-domain Blasius
  prediction with 14.5 % error is attributed to missing physics. Docs: `docs/core_concepts/physics_audit.md`.
- `PhysicsEnsemble(router=...)`: physics-aware prior on the weights per case; experts the router puts to sleep are
  not used and not penalised by that case. Predictions also return `prior`, `awake` and `escalate`.

- Module reference at `docs/org/reference/` (published under `/PINNeAPPle/org/reference/`): every package and
  module with its docstring and public API read from the source, plus curated explanations and 38 examples that are
  executed to record their output and plots (`scripts/reference/run_examples.py`, `scripts/reference/build.py`).
  Every package page lists all its subpackages and modules as a folding tree with a one-line description (from
  the docstring, from a curated note for subpackages without one, or built from what the module defines).
- Contribute page at `docs/org/contribute/`: the open issues grouped by theme (inferred from titles and labels), area,
  release, priority, effort and umbrella issue, with filters, search and where to start; a snapshot is embedded
  (`scripts/issues/build.py`, refreshed by the docs workflow) and the page refreshes from the GitHub API when it can.

- PINNeAPPle Lab (`pinneapple_lab`): experiment runner and database. Runs keyed by
  experiment, version and parameters (cached), statuses with validation checks, inputs/outputs/metrics/figures,
  sharded datasets with cards, SQLite index, sweeps (grid / Latin hypercube, parallel), dataset export and a Markdown
  catalogue; CLI `python -m pinneapple_lab`. Built-in experiments: oscillator, heat_xtfc, bondi_accretion,
  accretion_flow, cylinder_lbm, bh_forecast. Black-hole hydro: numba backend (`RIAFConfig(backend="numba")`).
  `repo_results` imports results already produced by repository scripts (Burgers PINN, fin inverse PINN 2D/3D, LBM
  Strouhal, MeshGraphNet, delta-wing polar, black-hole forecast skill), re-validates them against their references
  and stores them as datasets. `report` also writes `lab/DATASETS.md` (schema, units, counts, rebuild and export
  commands per dataset).
  `example_script` runs any script of `examples/` (examples and use cases) into the lab: figures, JSON and printed
  metrics, arrays as a dataset, console output and code, with the checkout left clean; `python -m pinneapple_lab
  examples [--run]`. `python -m pinneapple_lab serve` serves the catalogue as a web app (run files, JSON API,
  dataset downloads, optional login); Dockerfile in `pinneapple_lab/deploy` and a `lab` service in `apps/deploy`.
  Cached runs whose dataset shards are missing (a fresh checkout) are recomputed.
  Curation (`pinneapple_lab.curation`, `python -m pinneapple_lab curate` / `review`): every run and experiment graded
  A flagship / B solid / C exploratory / D not usable by gates over ten evidence dimensions (validation, reference,
  baseline, physics, generalization, uncertainty, reproducibility, data, assets, documentation) plus a human review,
  with readiness for product, paper, marketing and training data; Veriphysics vocabulary (verified / inferred /
  unsupported, not run is never a pass). `ctx.check(kind=...)`; tiers, filter and quality panel in the catalogue;
  `/api/curation`. Also: `benchmark_case` (landing-page cases), 3-D `vehicle_cfd` (road car, launch vehicle),
  `car_lbm` / `car_surrogate` (Physics AI on car geometry). Reports for one item or a filtered set
  (`python -m pinneapple_lab brief`, `/report.{html,md,pdf}`) and a showcase gallery in the catalogue.
  `pinneapple_physics.tribology`: Archard wear of a bar end with contact pressure on an elastic layer (implicit
  wear-contact step), Archard & Hirst material table; lab experiments `bar_wear` (one run per material) and
  `bar_wear_ranking`.
  `pinneapple_simulation.numerical_solvers.internal_flow`: pipes, bends and Kenics static mixers in OpenFOAM
  (structured O-grid swept along a centreline, mixer elements cut by snappyHexMesh, passive scalar by
  scalarTransportFoam), Colebrook and Ito bend-loss references; lab experiment `pipe_flow`.
  `pinneapple_simulation.numerical_solvers.solid_fem`: 3-D linear elasticity with C3D8I hexahedra (incompatible
  modes), structured box meshes, CalculiX cross-check, post-processor style figures; lab experiment `solid_fem`.
  `ParametricPOD` (POD + RBF / per-coefficient Gaussian process / linear / nearest regression, amplitude-shape split,
  predictive standard deviation) and `latin_hypercube` in `pinneapple_neural.architectures.rom`; lab experiment
  `rom_study`; docs page on reduced-order models; tests for POD, DMD, Operator Inference and the parametric ROM.
  Particle process videos: `pinneapple_simulation.numerical_solvers.particles` (soft-sphere DEM with drag, buoyancy,
  turbulent dispersion; analytic stirred-tank flow; Zwietering just-suspended speed) and
  `pinneapple_tools.visualization.studio.particles` (Cycles point-cloud animation with moving equipment, video
  composer with live charts, colour bar and clock; GIF and MP4); lab experiment `particle_suspension`.
  `pp.fea` (`pinneapple_simulation.external_solvers.calculix.study`): CalculiX studies from Python, with gmsh
  tetrahedra or hexahedral meshes, node and face sets by geometry, `Static` (pressure, gravity, nonlinear geometry,
  plasticity), `Frequency`, `Buckle` and `Heat` steps, and results as arrays (displacement, stress, reactions,
  frequencies, modes, buckling factors, temperature); `fea_figure` draws tetrahedral meshes and any nodal field. Lab
  experiment `calculix_case`.
  Lab trust card: every run and item answers six questions (data and geometry, model, physical constraints,
  benchmark, uncertainty, engineering decision) from its own evidence, in the catalogue, reports and CURATION.md.
  `DeepONet(depth=..., trunk_activation=...)` and per-sample query points; `MeshGraphNet(absolute_pos=False,
  decoder_layers=2)` for the translation-invariant, MLP-decoder form of Pfaff et al.
  `pinneapple_lab.uq`: coverage check of predictive intervals and grid convergence index (two or three meshes)
  as `kind="uncertainty"` checks.
  `python -m pinneapple_lab site docs/lab`: the self-contained public catalogue, published with the docs site under
  `/lab/`; README experiment gallery (`docs/assets/lab`).

- Black-hole weather (#399): `pinneapple_physics.blackhole`: axisymmetric viscous accretion solver (Paczyński-Wiita, α-viscosity, torus;
  validated against conservation, torus equilibrium and exact Bondi accretion).
  A torch port of the Duarte, Nemmen & Navarro (2022) U-Net, plus a residual variant.
  Lead-time and tendency scores, the mass check and mass projection.
  `twin.accretion_scene` (Twin3D cutaway) and `raytrace` (Schwarzschild ray tracing with lensed star field, the
  "Interstellar" view).
  Twin3D: Blender renders (`Scene.render_blender`), and viewer URL parameters `cmap`, `view`, `zoom`, `theme`,
  `range`, plus the inferno colormap.

- `pinneapple_veriphysics` (also `pp.veriphysics`): the verification and evidence layer of Veriphysics, now part of PINNeAPPle under
  Apache-2.0. `DecisionRecord` (recommendation, trust score and coverage, per-check evidence, alternatives), the Evidence Report PDF
  (`render_evidence_report_pdf`, optional `reportlab`: `pip install pinneapple[veriphysics]`), the applicability map (evidence chain,
  8-item checklist, tested variable envelope), measured robustness studies, `formulate_and_recommend` and the execution log. The job
  queue, HTTP API, billing and web app stay in the Veriphysics product, which imports this package.
- Adaptive physics ensembles tell regime changes from sensor noise (`PhysicsEnsemble`):
  `residual_lookahead` scores every model on the current case by its PDE residual before choosing, against the
  level it has in its own domain (`residual_baseline`, helper `residual_baseline()`), so a regime change is seen on
  its first case; `measurement_check(query)` blocks physically implausible readings from switching the model and
  from updating the weights (`EnsembleRun.suspect`); `lookahead_persistence=N` makes a model's residual count only
  when its last N readings agree. Examples `compare_lookahead_gif.py` and `sensor_noise_gif.py` with side-by-side GIFs.

- Global weather forecasting on ERA5 (`pinneapple_physics.weather`, first part): WeatherBench2 data (public bucket,
  64 x 32 grid) as a normalised local store, a U-Net whose convolutions are periodic in longitude and cross the
  poles, training on rollouts of growing length, area-weighted RMSE and ACC by lead with useful and no-skill
  horizons against persistence, climatology and the published IFS HRES, Pangu-Weather, Keisler and NeuralGCM
  forecasts, seven extreme-weather case studies (later ones read from ARCO ERA5 and averaged onto the grid) and
  forecast-versus-ERA5 globes and regional maps with a skill strip. Trained results and docs follow.

- Qualitative preview of geometries and geometry changes (`pinneapple_design.qualitative`, `pp.qualitative`): for
  an objective written in plain Portuguese or English, each variant (or each change to a part of an `Assembly`) gets
  the expected direction and strength of the change, the mechanisms behind it, the parts that drove it, side effects,
  a confidence level with reasons and a surface map, from cheap face-by-face models (`ExternalFlow`,
  `ConvectiveCooling`, `Cantilever`, `ScalingModel`); `part_sensitivity` ranks where to act first and `quantify`
  compares the expectations with an accurate computation. Includes a 3-D voxel FEM with incompatible-mode bricks
  (`voxel_fem_cantilever`) and three worked examples (heat-sink fins, beam stiffness per mass, Ahmed body slant).

- Animated view of adaptive physics ensembles (`pinneapple_physics.ensemble_viz.animate_ensemble`): GIF of the
  prediction against the reference field, the chosen model and the weights, case by case
  (`PhysicsEnsemble.run(..., keep_predictions=True)`). Example with five families (FNO, multiscale MeshGraphNet,
  DeepONet, PINN, CNN), each trained on its own regime: the ensemble picks each one in its regime, 7.8 % error over
  the stream against 24 % for the best single model.

- Adaptive ensembles of physics models (`pinneapple_physics.ensemble.PhysicsEnsemble`, `pp.ensemble`): online
  selection or combination of neural operators, graph networks, PINNs, numerical solvers and closed forms, learning
  from reference fields, from the physics residual alone (no ground truth) and from cost (adaptive fidelity), with
  lazy evaluation of the leading model, spread-based conformal intervals, an optional conservation projection,
  adapters for `pp.solve` solutions and torch modules, and `fit_static_weights` (convex weights with cross-validated
  error). Exactly solvable benchmark `pinneapple_physics.advection_diffusion_1d`. The online-learning core
  (Fixed-Share, AdaHedge, adaptive conformal) moved to `pinneapple_physics.online_learning` and is shared with the
  adaptive forecaster; AdaHedge's mixability gap is now computed with log-sum-exp (no underflow at large learning rates).

- Adaptive forecasting (`pinneapple_systems.time_series.AdaptiveForecaster`): online switching and combination of
  forecasting models from their past out-of-sample errors only (Fixed-Share exponential weights per horizon, AdaHedge
  over the learning and switching rates, adaptive conformal intervals), a pool of fast experts (naive, drift, SES,
  damped Holt, Holt-Winters, Theta, moving average, ridge AR) and `LagRegressorExpert` for any tabular ML model.
  No look-ahead is tested explicitly. On a seasonal -> trend -> random-walk series the 1-step MAE is 21 % below the
  best single model in hindsight.

- Earth-system building blocks (`pinneapple_simulation.geophysics`, `pp.geophysics`): spectral-transform shallow-water
  model on the sphere with Williamson et al. (1992) test cases 2, 5 and 6 and mass / energy / potential-enstrophy
  diagnostics (#258); 1-D Richards equation in mixed form (Celia et al. 1990) with van Genuchten-Mualem, Brooks-Corey,
  Clapp-Hornberger/Campbell and Gardner soils, verified against the exact Gardner infiltration profile (#260);
  two-layer energy-balance climate model (Held et al. 2010; Geoffroy et al. 2013) with exact integration, step
  response, ECS/TCR and parameter fitting (#259).
- 4D-Var through autograd for any differentiable PyTorch model, with Lorenz-96/63, an adjoint gradient check and an
  identical-twin experiment (`pinneapple_analysis.data_assimilation`, `pp.assimilation`).
- Exact conservation for surrogates: `pinneapple_physics.conservation.project_integral` and `ConservationProjection`
  (additive, multiplicative and positive global fixers, differentiable).

- `pinneapple_security` (also `pp.security`): data and process security for scientific and industrial Physics AI.
  Merkle manifests of datasets and model folders; Ed25519/HMAC signatures in DSSE envelopes; in-toto + SLSA provenance
  of experiments and a CycloneDX SBOM; a hash-chained, tamper-evident audit trail; AES-256-GCM encryption at rest; PII
  detection (CPF/CNPJ, cards and IBANs checksum-validated), keyed pseudonymisation, k-anonymity and solver-log
  sanitising; differential privacy (Laplace/Gaussian mechanisms, RDP accountant of the sampled Gaussian, DP-SGD with
  `torch.func`); checkpoint scanning and `safe_load`; adversarial sensitivity of surrogates; physics-residual CUSUM and
  replay detection of manipulated sensor data; secret scanning; data labels and a handling policy (personal data,
  export control); an evidence map to NIST CSF 2.0, IEC 62443-3-3, ISO/IEC 27001:2022, 21 CFR Part 11, LGPD/GDPR, SSDF
  and SLSA; signed `ModelStore` versions and model cards; `python -m pinneapple_security` command line. Extra:
  `pip install pinneapple[security]`. Example: `examples/security/01_secure_physics_ai_pipeline.py`.

- Public validation status page `docs/validation_status.md` (#97): per catalog item, validated / tested / no test with the number of reference and other tests, generated by `scripts/build_method_status.py --public` from the test scan (85 validated, 53 tested, 41 without a test on 2026-10-09). A test and the release workflow check it matches `method_status.json`.

- Process optimizer (Data Health app, Optimize tab): forward-chaining time-series cross-validation on the training period chooses the L2 regularisation of the KPI model; the fold errors, their mean and spread and the chosen strength are in the report (`model.cv`) and on the page (#81).

- Experiment tracking adapter (#113): `pp.Experiment(..., tracker="mlflow" | "wandb" | callable)` sends the flat config, every metric as `<metric>/<field>`, the wall time, the problem fingerprint and the saved record (`result.json`, `model.pt`) to the tracker; `pinneapple_physics.tracking.log_result` does the same for an existing result. Extra: `pip install pinneapple[tracking]`.

- TrustReport persisted with the model (#157, decision D2): `ModelCard.attach_trust_report` / `override_trust` and `ModelStore.save(trust_report=...)` / `set_trust_report` / `promote(override_reason=...)`. A REJECT blocks publication (card validation, hub push, promotion to staging/production) unless overridden with a recorded reason. `TrustReport.from_dict` restores a stored report.

- Cost columns of the benchmark protocol (#55): `pp.compare` tables and the benchmark suite leaderboards report training time, inference cost per point (µs, best of three) and the number of reference simulations the method consumed (`n_reference_sims`; from `Solution.info['n_reference_simulations']` or the task's `n_reference_simulations`, 0 for physics-only training).
- `examples/first_example/first_example.py`: solve, check against the exact solution and plot in 11 lines, about a
  minute on CPU; a CI job runs it from a clean install of the wheel (#44).
- `scripts/check_dist.py`: the built sdist and wheel must contain every package and the viewer assets, and the
  sdist must install in a clean venv and import every package. It runs in the test workflow and before a release
  (#92).
- MeshGraphNet (`pinneapple_neural.architectures.graphnn`, registered as `mgn`/`meshgraphnet`) and the transient
  recipe `MeshDynamicsMGN` (Pfaff et al. 2021), with examples in `examples/meshgraphnet/` (synthetic diffusion,
  DeepMind cylinder_flow, PhysicsNeMo parity scripts); `examples/vs_physicsnemo/06` now trains this MGN
  (#232, #234, #237).
- 3D studio (`pp.viz`, `pinneapple_tools.visualization.studio`): a `Scene` for any geometry (STL/OBJ, any result
  `pinneapple_data.cae` reads, arrays) with per-vertex fields, streamlines and slice planes; export to glTF (fields as
  vertex attributes), OpenUSD and STL; Blender Cycles renders on the jet colour scale with a colour bar; a browser
  viewer that ships its own three.js. Vectors show as magnitude, stress tensors as von Mises.
- Studio input and viewer: `Scene.from_file` reads glTF/GLB (node transforms, fields from `_NAME` attributes) and
  VTK `.vtp` (with `vtk`). The browser viewer has colour-range, slice-position, line-density and edge controls on a
  shared core (`studio-core.js`, also used by the aircraft app and tested with node). Slices can be grouped into
  stacks, and `ExternalFlow` samples four wake planes. `Scene.decimate` and `web_viewer(max_faces=...)` keep large
  meshes usable; the docs give measured sizes and load times. The digital twin writes its GLB with the studio writer
  and gains `to_studio()`. three.js ships once, in the library, and the apps serve that copy.
- External flow in OpenFOAM for any body (`pp.cfd.ExternalFlow`): snappyHexMesh, simpleFoam k-ω SST with wall
  functions, half model, moving road; forces, skin Cp/Cf, streamlines, mid-plane and wake slices, `to_scene()`.
  Ahmed body at 25°: CD 0.321 (coarse) and 0.298 (medium) against about 0.285 measured. Closed bodies in
  `pp.bodies` (Ahmed body, sphere, cylinder, box); `pp.cae` reaches `pinneapple_data.cae`.
- Aircraft Design Optimizer (`apps/aero_optimizer`, service `aero`, library `pinneapple_design.aero`): airfoil
  (CST) and wing-area design for a light aircraft. 436 OpenFOAM runs (simpleFoam, k-ω SST, y⁺ < 1) on an O-grid with
  the same topology for every design train a MeshGraphNet (flow field + Cl, Cd, Cm; drag within 1.5 % on unseen
  airfoils) and an MLP ensemble (0.5 %, with uncertainty). NSGA-II with constraint domination searches top speed,
  CO₂ per 100 km and stall speed under stall (CS-23), thickness, trim and stall-margin requirements; designs that miss
  one, and designs the surrogates disagree on, are shown with the reason. Pareto designs verified in OpenFOAM agree
  within 0.1 kt / 0.1 % / 0.3 kt and are added back to training. Every design downloads as a ready OpenFOAM case.
- Aircraft Design Optimizer, airliner: parametric single-aisle airliner (`pinneapple_design.aero.airliner`: fuselage
  with windows and livery, swept wing with sharklets, turbofans, tails; glTF with PBR, USD, STL); compressible vortex
  lattice, Korn wave drag, transport weights and Breguet mission calibrated on the A320ceo (`airliner3d`); NSGA-II
  over 13 variables for CO₂ per passenger-km, cruise Mach and Vref (`optimize_airliner`); OpenFOAM 3D half-model runs
  (`case3d`) with skin Cp/Cf, streamlines and slices, shown in a three.js viewer and Blender Cycles renders on the
  jet colour scale.
- Five CAE apps, each deployed as a service in `apps/deploy`:
  - Simulation Preflight (`apps/simulation_preflight`, service `preflight`, library `pinneapple_data.preflight`):
    checks an OpenFOAM case or a CalculiX deck before it runs. Each finding has a PASS/WARNING/FAIL verdict, where it
    is (file and line), its severity, an explanation, the fix and an exportable checklist. Rules were validated on
    real solver runs, including two failures the solver never reports: a model with no supports (ccx reports "Job
    finished" with a 1.8e11 mm displacement) and inconsistent units (first frequency 0.16 Hz instead of 209 Hz).
  - Mesh Quality (`apps/mesh_quality`, service `mesh`, library `pinneapple_data.cae`): checkMesh-equivalent
    finite-volume metrics (identical to OpenFOAM v1912 on pitzDaily, a Gmsh tetrahedral mesh and a sheared channel).
    Also element metrics (scaled Jacobian, skewness, edge ratio) for Gmsh/VTK/Abaqus/CalculiX meshes, histograms, a 3D
    heatmap, the worst elements, problem regions and recommended actions. Reads OpenFOAM, `.inp`, `.frd`, STL and
    anything meshio reads.
  - Simulation Comparator (`apps/simulation_comparator`, service `compare`, `pinneapple_data.cae.compare`): reference
    vs candidate, in four modes: simulation/simulation, simulation/experiment, simulation/AI and AI/experiment. Gives
    global and per-field error (relative L2, MAE, RMSE, max, p99, bias, NRMSE, R²) and an error map. Fields are matched
    by name and component and interpolated between different meshes. Checks: Ghia lid-driven cavity 1.26 % → 0.19 %
    with mesh refinement; a PINN vs finite volumes 0.39 %.
  - Engineering Model Lineage (`apps/model_lineage`, service `lineage`, library `pinneapple_data.lineage`): the
    digital thread as a graph of artifacts, each with file, version, software, parameters, timestamp, sha256, owner
    and origin. Auto-detected from OpenFOAM/Gmsh/CalculiX folders and PINNeAPPle exports, or declared in a
    `lineage.json` whose hashes are verified. Checks: hash mismatch, stale outputs, missing inputs, cycles, mixed
    revisions. Answers upstream/downstream questions and exports W3C PROV-JSON and Markdown.
  - Simulation Interoperability Hub (`apps/interop_hub`, service `interop`, `pinneapple_data.cae.dataset`): any
    result becomes the neutral `pinneapple.physical_dataset/1` (geometry, mesh, coordinates, fields with quantity and
    unit, metadata). It exports to VTK `.vtu` (OpenFOAM polyhedra rebuilt as standard cells or VTK polyhedra), HDF5,
    Parquet, CSV, NPZ, JSON and a PINNeAPPle dataset (UPD Zarr).
- Inverse Heat Lab (`apps/inverse_heat`): the convection coefficient h from a few thermocouples with an inverse PINN,
  in 1D (a pin fin, trained live on demo readings or your own, cross-checked by a least-squares fit of the analytic
  solution), 2D (a heat-spreader plate, trained live and compared with a finite-volume solution cell by cell) and 3D
  (a chip under a block, the full offline run in an interactive 3D view with layer slices). Every case shows its complete
  script (`pip install pinneapple`, then run), filled with the page's inputs and readings, to copy or download; the
  three scripts are run by the tests. Deployed as the `inverse`
  service in `apps/deploy`.
- `pinneapple_core` (roadmap X3, #187), reachable as `pp.core`: `Domain`, `Geometry`, `Mesh` and `Field` as shared
  primitives. A `Field` holds values on a tensor grid, a simplex mesh (1D/2D/3D) or a point cloud and answers
  `gradient()`, `divergence()`, `interpolate(x)` and `integrate()` the same way on each (trapezoid on a grid, exact
  P1 on a mesh, local least squares and Monte Carlo/quadrature on a cloud). `FunctionField` offers the same calls for a
  torch callable with autograd derivatives that keep the graph, including `laplacian`. `Domain` covers boxes in any
  dimension and the 2D CSG shapes (interior and boundary sampling with normals, `contains`, signed distance);
  `Geometry` adds named boundaries and a mesher. Examples in `examples/core_primitives/`: a Poisson PINN
  (relative L2 error 5e-3) and a Fourier neural operator on grid fields.
- `pp.transforms` (roadmap X13, #197): `Scale`, `Nondimensionalize`, `Coordinate` (polar, cylindrical, spherical, log), `Symmetry`
  (reflect, rotate, with vector fields), `Periodic` and `FourierFeatures`, composable with `>>`, invertible where applicable, applied to
  tables, `PhysicsDataset` and (the scaling ones) `PhysicalProblem`, with a provenance record per step (`problem.metadata["transforms"]`,
  `dataset.provenance`) holding the fingerprints before and after. Nondimensionalising Burgers gives `nu' = nu T / L^2`, checked against the
  exact solution. `PhysicsDataset` gained a `provenance` list.
- Physics data API (roadmap X12, #196): `pinneapple_core.data` with `PhysicsDataset`, `DataLoader` (datasets and samplers as
  sources, one `Batch` with a group per source, `physics_aware` sets `requires_grad` on differentiated points) and the samplers
  `CollocationSampler` (uniform, lhs, sobol, any `Domain`), `BoundarySampler` (normals, named parts), `MeshSampler`,
  `TrajectorySampler`, `AdaptiveSampler` (residual-proportional) and `ActiveSampler` (top acquisition score). A Poisson PINN, an
  operator and an inverse problem for the diffusivity train from the same loader API.
- `PhysicsOptimizer` (roadmap X11, #195): `PhysicsOptimizer(parameters, objective, constraints)` in `pinneapple_core.optim`
  with six methods behind `minimize(method)` and `compare()`: `slsqp`, `adjoint` (supplied gradient), `penalty`,
  `augmented_lagrangian`, `differential_evolution` and `bayesian` (constrained expected improvement). The same
  constrained problem (maximum torsional rigidity of a rectangle at fixed area, with the differentiable FEM) is solved by
  all six and gives the square. `Result.feasible` reports infeasible problems instead of hiding them.
- `pp.loss` (roadmap X10, #194): `pde` (optionally causal), `boundary`, `conservation`, `energy`, `symmetry`, `supervised`,
  `inverse`, `combine`, and `Balancer(strategy, names, model=)` choosing the weighting by name: the 13 existing
  `WeightScheduler` methods (fixed, self_adaptive, gradnorm, loss_ratio, ntk, relobralo, softadapt, augmented_lagrangian,
  inverse_dirichlet, lr_annealing, pcgrad, joint_adaptive, auto) plus `curriculum`. The SA-PINN example
  (`examples/vs_physicsnemo/03_pinneapple_active_weight_sched`) now uses it and reproduces its loss and error histories exactly.
- `pp.compile` (roadmap X9, #193): `pp.compile(problem, optimize="physics")` returns a `CompiledProblem` (`solve`, `loss_fn`,
  `benchmark`). The optimisation is derivative reuse: first derivatives of the same field are built once per loss
  evaluation (`solve_pde` and `compile_problem` take `cache_derivatives=True`). Measured speedup of PINN training 1.05x to
  1.29x on five presets with trained weights equal to 6e-8; forward-mode and batched second derivatives were tried and are
  not faster on CPU. Details and the full table in `docs/core_concepts/solver.md`.
- `PhysicsModule` (roadmap X8, #192): `pinneapple_core.module` with `PhysicsModule`, `Sequential`, `SolverModule` (a
  differentiable function with optional trainable parameters, or any `pp.solve` method via `from_method`) and `Hybrid`
  (solver plus neural correction, `residual` / `multiplicative` / `replace`). A coarse finite-element solve with a
  convolutional correction trains end to end through the solver (held-out relative error 0.31 coarse alone, 0.10 hybrid).
- `PhysicsBackend` (roadmap X7, #191): `pinneapple_core.backend` gives torch and jax one API (`asarray`, `matmul`, `solve`,
  `integrate`, `grad`, `jacobian`, `vmap`, `jit`, array constructors, `xp` for elementwise math), a registry
  (`register_backend`) for further engines, `pp.use_backend(...)` as a context manager and `pp.get_physics_backend()`.
  `set_backend` now accepts any registered name. A finite-difference Poisson problem and the gradient of its objective
  with respect to the domain length run unchanged on torch and jax with the same results.
- One `solve(problem)` contract across solver kinds (roadmap X6, #190): every method has a `kind` (`classical`, `neural`,
  `analytic`, `external`, ...), every `Solution` has `metadata()` with the same keys, and `compare` reports the kind.
  New methods: `fem` (P1 finite elements for steady Poisson/Laplace on a box, on `pinneapple_core.fem`) and `external`
  (any code that returns grid fields). On a manufactured Poisson problem FEM and PINN solve the identical
  `PhysicalProblem` through the same call (relative L2 4.2e-3 and 1.3e-3). `register_method(name, kind=...)`,
  `list_methods(kind=)`, `list_kinds()`.
- `pp.func` (roadmap X5, #189): `grad`, `jacobian`, `jacrev`, `jacfwd`, `hessian`, `vmap` over `torch.func` with
  `wrt=` by argument name (`jacobian(model, wrt="geometry")`); `implicit_solve` differentiates through a nonlinear
  solve by the implicit function theorem (one adjoint solve, any external forward solver); `pinneapple_core.fem`
  is a differentiable P1 Poisson solver giving gradients with respect to mesh vertices, checked against finite
  differences.
- Physics operators (roadmap X4, #188): `pp.grad`, `div`, `curl`, `laplacian`, `jacobian`, `hessian`, `integrate` and
  `flux` (module `pinneapple_core.operators`) with one call on a continuous field (torch callable, autograd), a grid,
  a mesh or a point cloud, checked against manufactured solutions on each. `flux` obeys the divergence theorem
  (exact for P1 fields on a mesh, trapezoid on a grid, Monte Carlo on a cloud or continuous field). Point-cloud
  gradients now use a local quadratic fit (second order; 0.25% against 3.5% before on the test field).
  New helpers: `Mesh.boundary_geometry()` (outward normals and facet measures), `Domain.boundary_measure()`.
- Example `examples/use_cases/fin_convection_inverse`: the convection coefficient of a pin fin identified from five
  noisy thermocouples with an inverse PINN (`PINNFactory`, h trainable and used in the tip condition too). Over 10
  noise draws h = 24.8 ± 0.4 W/m²K (true 25), the dissipated heat within 0.2 %, as accurate as a least-squares fit of
  the analytic profile without needing it; results and figures in `results/`.
  The same inverse problem in 2D (a heat-spreader plate with an 8 W device: h = 14.8 ± 0.3 for a true 15,
  hot spot 79.2 °C vs 79.1) and 3D (a 10 W chip under a steel block, thermocouples only on top:
  chip temperature 74.6 °C vs 74.5, h 149 from the energy balance on the learned field, the
  trainable h -7 %), checked against an independent finite-volume solver (`fv_reference.py`).
- Form Compiler (`apps/design_requirements`, library `pinneapple_data.formfill`): pick a document format, drop the
  datasheets and specifications, and get every item the format asks for with the document, page and text each value
  came from, the conflicts between documents and the missing required items. Supported formats: ASME Section VIII
  Div. 1 Form U-DR-1 (fills the official fillable form; `pinneapple_data.udr1` keeps the shortcuts) and compiled
  datasheets with the data of API 520/526 (relief valves), TEMA/API 660 (shell-and-tube exchangers, shell and tube
  side columns), API 650 (storage tanks) and API 610 (centrifugal pumps), rendered as PDF (`render_datasheet`). Formats
  are `FormSpec` data and round-trip through JSON; `Compilation.record()` exports values, units, SI values and sources
  for other forms and calculations. Optional extraction with a local LLM served by Ollama (standard-library HTTP,
  documents stay on your network) keeps a value only when its quoted source is in the document and the value is in
  the quote. Scanned pages are read with OCR (Tesseract, `pinneapple_data.formfill.ocr`): pages are deskewed, table
  grid lines are erased before recognition and used to rebuild the cells, borderless tables are split at wide gaps,
  and every OCR value carries Tesseract's confidence (under 85 % it is flagged for checking against the scan).
- Front door for using the library: `pp.solve(problem, method)` and `pp.compare(problem, methods, reference=...)` over
  `PhysicalProblem`, preset names or PDE specs, with methods `pinn`, `analytic`, `exact`, `reference` and
  `@pp.register_method` for your own. `compare` scores every method on the same points and reports a failing method
  instead of stopping. The `reference` method refuses solver output that does not cover every coordinate and field (#28).
- `pp.Experiment`: one reproducible run (seeds Python, NumPy and PyTorch, deterministic algorithms while it runs),
  scored against a reference, with the problem fingerprint, environment and versions; `ExperimentResult.save()` writes a
  JSON record and the model weights. Two runs with the same seed give bit-identical predictions on CPU (#29).
- `pp.metrics`: `relative_l2`, `rmse`, `mae`, `max_abs`, `relative_linf`, `r2`, `per_field` and `summary`, one convention
  for the whole library: per field, float64, relative errors `nan` where the reference norm is zero (#32).
- `pinneapple_physics.closed_form.burgers.burgers_sine_exact`: the Cole-Hopf solution of the `burgers_1d` preset
  (Gauss-Hermite quadrature with a shifted exponent, so no overflow at small viscosity), used by the `analytic` method.
- `PhysicalProblem` (`pp.PhysicalProblem`, with `pp.Parameter` and `pp.Quantity`): one description of a physics problem
  (coordinates, fields, PDE, conditions, domain or geometry, units, quantities of interest, solver choice, task and intent).
  Parameters carry a unit and a role (`fixed`, `design`, `uncertain`, `unknown`), so the same object can drive a forward
  solve, design optimization, uncertainty quantification or parameter discovery. `validate()` lists inconsistencies before a
  solver sees them, `to_dict()`/`from_dict()` round-trip through JSON (a condition callable that is not serialized comes back
  as a placeholder that fails loudly), and `fingerprint()` gives a stable hash for provenance. The three existing classes are
  adapters, unchanged: `from_pde_spec`/`to_pde_spec` round-trip all 65 presets exactly, and `from_physics_case`/
  `to_physics_case` and `from_problem_design`/`to_problem_design` cover the other two (#27).

### Fixed

- `om_runner._read_om_result` dropped `time` (dataset 0 of the OpenModelica result file).

- `simulate_fmu` failed on every FMU when no parameters were given (`start_values=None` passed to fmpy).
- `sample_uniform_box` only accepted 3-D boxes, so `CollocationSampler(strategy="uniform")` failed in 1-D and 2-D;
  it now samples boxes of any dimension.
- Registry: two model or problem versions saved within the same second got the same version name and the second
  silently overwrote the first; a suffix (`_2`, `_3`, ...) now keeps both.
- Arena: an FNO was trained on random mini-batches of scattered points reshaped as if they were an image (and
  `fno2d` built the 1-D operator); it now trains on whole ordered grids (half the evaluation resolution, so the
  evaluation also tests super-resolution) and refuses scattered points. DeepONets train as operators (branch: the
  instance's input function, trunk: the query points). Compiled PINN loss terms that fail raise instead of being
  dropped, and an unknown or unbuildable model type raises instead of becoming an MLP under that name.
- `DynamicModeDecomposition` and `OperatorInference` stored their operators in float32 whatever the data, so a
  float64 fit failed at `rollout` and `eig`; they now keep the data's dtype.
- `MultiScaleDeepONet` fed every trunk the same unscaled coordinates (no multi-scale effect) and kept an unused
  base trunk; each trunk now sees the coordinates stretched by its own `scale_factors` (MscaleDNN).
- The Arena built DeepONets with a list `hidden` that the class rejected, and the benchmark API with argument
  names that do not exist; both silently returned an MLP under the DeepONet name. The Arena now builds the
  DeepONet; the pointwise benchmark API raises a clear error.
- Masked losses of `MeshGraphNet`, `GraphNeuralNetwork` and `EquivariantGNN` averaged over padded nodes too (half
  the true loss at 50 % padding); they now average over the valid nodes.
- The digital twin viewer (`pinneapple_twin3d`) is in English: labels, tooltips, status and error messages;
  numbers use the browser's locale (#306).
- The benchmark suite computes every error through `pp.metrics` (new `metrics.pooled` for the single leaderboard
  number, same values as before); its inline formulas are gone (#32).
- `ExternalFlow` without bodies and `trust_report.Check` with an unknown status raise a clear `ValueError` in
  English instead of a `TypeError` or a Portuguese message.
- `UPDZarrStore` writes with `Group.create_array` on Zarr 3 (it used the deprecated `create_dataset`), and a test covers the
  write/read round trip on Zarr 2 and 3 (#239).
- `serialization.load_zarr` called `UPDZarrStore.iter_samples`, which did not exist, so every call failed; the store
  now has it.
- `pp.solve(..., "pinn", ctx=...)` ignored `ctx`, so a Poisson source term never reached the network (the solve returned
  the zero field); it is now passed to `solve_pde`.
- Fixed `templates/30_zarr_data_pipeline.py` to use the current UPD and Zarr APIs.
- `physical_units`: `mPa` (and `mPa·s`, `mW`) was read as `MPa` (`MW`), a factor of 10⁹, through the case-insensitive
  fallback; milli and mega prefixes are no longer interchanged.
- **Behavior:** `solve_pde` (and so `pp.pipeline`) silently dropped every boundary and initial condition defined by a
  selector function, which is most presets. It drew points inside the box and kept those the selector accepted, but a
  boundary (`t == 0`, `x == -1`) has zero volume, so no point was ever kept and the network learned the trivial solution.
  On `burgers_1d` (nu = 0.01/pi) the relative L2 error of a 4000-epoch PINN goes from 1.02 to 0.023. Conditions are now
  sampled inside the box and on each of its faces, and a condition that still selects no point raises an error. The
  same flaw in the dataset generator (`_sample_callable_condition`) returned the origin repeated `n` times, and a selector
  that raised was treated as "select everything"; both fixed.
  Once these conditions had points, three more gaps they had hidden came up and are fixed: first-order Neumann/Robin
  conditions now get the outward normal of the box face their points lie on (`batch["n_bc"]` was never built; points on a
  curved boundary inside the box are skipped with a warning, since a selector cannot give their normal), conditions on
  different fields are laid out over all model fields instead of failing to stack, and the plane `t = 0` (or `x = 0`)
  is sampled when it lies inside the range, as for an initial condition at `t = 0` on `[-T, T]`.
- Six presets could not be compiled: `pcb_thermal`, `cpu_heatsink_thermal`, `datacenter_airflow_2d`, `datacenter_cfd_3d`,
  `datacenter_server_thermal` and `car_suspension_fatigue` declared heat-flux, convection or traction boundary targets
  (`q_heat`, `h`/`T_ref`, `tx`/`ty`) without the `thermal_bc` or `traction_map` that tells the compiler how to resolve them, so
  building the loss raised `KeyError` (the notes in `tag_geometry.py` said they were already resolvable). They now declare it,
  and a test evaluates each loss. Found by `PhysicalProblem.validate()`.

- `pp.compare` now accepts precomputed `Solution` objects in `methods`, reusing their predictions and training time
  instead of solving again. It validates coordinate and field compatibility and gives duplicate method names unique
  row labels.

### Documentation
- JOSS paper draft (`paper/paper.md`, `paper/paper.bib`) and a workflow that builds its PDF (`draft-pdf.yml`).
  `CITATION.cff` lists both authors and points to this repository; the Code of Conduct is now the full Contributor
  Covenant 2.1; `CONTRIBUTING.md` explains how to report bugs and get help; the README has a statement of need.

### Known issues
- The classical reference path for `burgers_1d` (`fdm` solver) overflows and returns a grid over `x` only, without the time
  axis; `pp.solve(..., "reference")` refuses it. Use `"analytic"` for this preset.
- `industrial_furnace_thermal` still does not compile: its hot-face condition combines convection and radiation, which
  `thermal_bc` does not support, and `insulation_interface` needs a two-region interface model. Documented in the preset notes.

## [0.6.3] - 2026-10-04

### Added
- `LRAnnealing` loss balancer (Wang, Teng & Perdikaris 2021), wired into `WeightScheduler` as
  `method="lr_annealing"` (#23).
- OpenRadioss connector: deck editing, Docker runner and VTK reader
  (`pinneapple_simulation.external_solvers.openradioss`) (#24).
- Native Transolver (`PhysicsAttention`, `Transolver`, `TransolverLite`), registered as `transolver_native`,
  `physics_attention_transolver` and `transolver_lite` (#24).
- Solvers: HLLC/MUSCL compressible finite volume (`compressible_fv`, checked against Sod's exact solution and the
  Rankine–Hugoniot normal-shock jump), pseudo-spectral 3-D Navier–Stokes DNS (`ns3d_spectral`) and a 2-D
  heated-channel incompressible solver (`thermal_channel_2d`) (#24).

- `pinneapple_analysis.cost`: computational-cost validation per operation. `measure` (median time with spread, peak
  memory, exact PyTorch FLOPs), `scaling_study` (fits `cost ~ n^k` with a bootstrap confidence interval and compares
  it with a declared complexity such as `"n log n"`), `Budget` (time, memory, FLOPs, exponent limits) and `CostLedger`
  (saved baseline; FLOPs gate strictly, time and memory loosely). `pinn_operation_costs` splits one PINN training
  step into forward, first and second derivatives, backward and optimizer step; `profile_ops` gives a per-operator table.
- `pinneapple_tools.visualization.pyvista_bridge` and the `pinneapple[pyvista]` extra: CalculiX model plus `.frd`
  results to a PyVista grid (displacement, von Mises, optional exaggerated warp; C3D4/C3D8/C3D10/C3D20 families, checked
  against beam theory and box volume), `pinneapple_twin3d` scenes to `MultiBlock`, `.vtu` export for ParaView and
  off-screen PNG/GIF rendering. `can_render()` probes for an OpenGL context in a subprocess because VTK crashes the
  interpreter when none exists; on a headless server use `xvfb-run -a`.
- `pinneapple[cost]` extra (psutil) for CPU memory sampling in the cost module.
- GitHub issue forms (bug report, feature request) and a pull request template; CI runs a blocking tier for the public
  API, cost module and the PyVista and CalculiX bridges.
- Example `examples/calculix_pyvista_cantilever.py`.
- Example `examples/calculix_cantilever/`: a real CalculiX cantilever reference run (input deck, results and README) (#143).
- Release automation: `.github/workflows/release.yml` publishes to PyPI only from a `vX.Y.Z` tag, through Trusted Publishing
  (no stored token), after `scripts/check_release_version.py` confirms that the tag, `pyproject.toml`, `CITATION.cff`,
  `pinneapple.__version__` and a dated `CHANGELOG.md` heading agree, and after the built wheel imports in a clean environment.
- `.github/workflows/extras.yml`: each optional extra installs in a clean environment and `pp.info()` reports no broken package.
- `scripts/check_example_imports.py` and `tests/test_example_imports.py`: every `pinneapple*` import in `examples/` and `templates/`
  must resolve. 156 imports in 57 files were already broken (they predate the refactor into the mega-modules); they are listed in
  `scripts/example_imports_baseline.txt`, new breakage fails CI and a fixed file must leave the list.

### Changed
- **Behavior:** the `numerical_convergence` component of `PhysicsConfidenceScore` is now multiplied by an order-plausibility factor: an
  observed order outside [0.5, 4.0] (the convention already used by the CFD grid-convergence check) is not a believable asymptotic rate, even
  when the GCI arithmetic is self-consistent, so scores for such studies drop. The same change touches the parametric dense-volume
  operator workflow (#130).
- **Behavior:** `bekker_wong` terramechanics rewritten: input validation, break points at the stress kinks,
  sign-aware shear for braking, Brent's method for sinkage, no fabricated values when the solver fails, and a
  vectorised, autograd-differentiable batched Gauss–Legendre quadrature (#24).
- **Behavior:** terramechanics preset: removed a false physics constraint (`Fx(s=0)=0`, which the solver itself
  violates) and added a verified one (`dFz/dz >= 0`) (#24).

### Fixed
- `from pinneapple import *` and `getattr` on 24 names in `pinneapple.__all__` failed: lazy aliases such as `pp.uq`, `pp.geom`,
  `pp.dt`, `pp.export` pointed at module names that never existed (`pinneapple_uq`, `pinneapple_geom`, ...) and the
  streamline, world-model and scene imports used names that are not exported. Aliases now point to the real packages
  (for example `pp.uq` is `pinneapple_analysis.uncertainty`); `pp.serve` is removed (no such module);
  `plot_streamlines_2d_model` is `plot_streamlines_2d_from_model`; the world-model names `CosmosAdapter`,
  `PhysicsVideoDataset`, `SimToRealAdapter`, `PhysicalScene` and `SceneObject` never existed and are replaced by
  `PhysicsWorldModel`, `WorldModelDataset`, `WorldModelTrainer` and `PhysicsScenario`. A test now resolves every name.
- `templates/08_csg_geometry.py` and `templates/09_flow_visualization.py` imported names from the wrong module or a name that does not exist.
- `run_custom_solver` (`pinneapple_tools.sandbox`) could not run on Linux with the default memory limit: the child was started with
  `-m`, which imports the whole `pinneapple_tools` package (and torch) before the script, and under `RLIMIT_AS` of 1 GB the loader
  failed to map `libtorch_cuda.so`, so every call returned `ProcessTerminated`. The child now runs the runner file by path and loads only the stdlib.
- Test suite: the OpenFOAM graceful-degradation test passes whether or not OpenFOAM is installed; the Triton export test skips when `onnx`
  is missing; two problem-design bridge tests no longer use a hard-coded macOS path and the import check reads the AST instead of grepping
  (a docstring mention and stale `.pyc` files were false alarms).
- `pp.info()` reported "optional deps missing" for modules that do not exist; it now imports the real packages and lists
  optional third-party dependencies with the extra that installs each.
- `LICENSE` was a truncated Apache-2.0 text (end of section 4 and the appendix were missing); replaced with the canonical text.

### Documentation
- Recorded two failed attempts to fix mass conservation in the immersed-boundary `channel` mode (both reverted,
  both measured worse than the current behavior) (#25).

## [0.6.2] - 2026-10-02

### Added
- Validation batch 4: exact-solution tests for FEM Q1, Kansa RBF, axisymmetric eddy current, the compressor
  similarity map, Buckley–Leverett, neo-Hookean, thermoelasticity, Biot (Terzaghi mode) and Heston (84 validated
  items in total) (#19).
- Example: interactive shallow-water dam-break demo viewed in Twin3D (#21).
- Benchmarks: the PK-PD benchmark was re-run for real with all four optimizer variants (#20).
- Reference applications in the repository under `apps/` (not part of the wheel): HeatSink Sizer, PCB Hotspot,
  Engineering Data Health (with an Optimize tab), Engineering Data Standardizer and Simulation Metadata API, a shared
  app kit, and a single-server Docker/Caddy deployment (also usable behind an existing reverse proxy) (#18).
- `pinneapple_app` (repository only, not packaged): experiments apply real physics, runs are seeded and split into
  train, calibration and test sets, and every model gets a surrogate trust report (held-out error, convergence,
  generalization, uncertainty with split conformal coverage, physics checks) (#18).

### Changed
- **Behavior:** `GradNormBalancer` update is now `w * target / ||grad(w L)||`, with the global L2 gradient norm over
  the shared layer. Weights now reach the GradNorm target in one step and stay stationary for unchanged losses (#17).

### Fixed
- `GradNormBalancer` divided by the weighted gradient norm twice, so it converged to square-root balancing and
  oscillated (#17).
- FEM and Kansa solvers crashed on any Dirichlet condition (an `isinstance` check against the `DirichletBC` factory).
  A shared `_bc` helper now accepts the builder contract `(X, ctx)` and one-argument callables, and tells an arity
  mismatch from a real `TypeError` by inspecting the signature (#19).
- Kansa multiquadric Laplacian used `(d-2)` instead of `(d-1)`, and Kansa now honours the input dtype (#19).
- `axial_flux_density` returned `-B_z` and dropped the quadrature part (#19).
- The neo-Hookean residual differentiated the Cauchy stress in reference coordinates; it now uses `Div_X P` (#19).
- `pinneapple_app` experiments never applied physics: a bare `try/except` hid a `KeyError` every epoch, boundaries
  were forced to `u = 0`, custom equations were a `* 0.0` placeholder, and leaderboard metrics used the training
  points (repository only) (#18).

### Known issues
- The immersed-boundary `channel` mode does not conserve mass (documented in the module).

## [0.6.1] - 2026-09-27

### Added
- CalculiX bridge: `.inp` and `.frd` I/O, a `ccx` runner (Debian-based Docker image built from the bundled
  Dockerfile), and `pinneapple-ccx`, a drop-in `ccx` command whose fields come from a PINN or surrogate. It has a
  validity-envelope guard, a `--fallback` to the real `ccx` and a provenance JSON per job. Validated: C3D20R
  cantilever tip deflection within 0.94 % of Timoshenko theory.
- Large Physics Model blocks (`pinneapple_neural.lpm`): Fourier positional encoding, multi-scale neighbourhood
  features, DeepSets geometry code, operating-parameter conditioning, bagged-ensemble heads, a geometry-code
  out-of-distribution score and fine-tuning with a frozen geometry encoder. 3.1 % mean relative L2 on unseen designs
  in a potential-flow benchmark, with out-of-range designs flagged.
- `pinneapple_analysis.uncertainty.posterior_metrics`: PosteriorBench metrics (multi-scale-kernel MMD, sliced
  Wasserstein, radially averaged power-spectrum error, mean/std relative L2), ported from
  neuraloperator/PosteriorBench (MIT).
- Twin3D: OpenFOAM case to scene (wall patches and per-face values over time, ASCII and binary polyMesh), OpenUSD export
  with time-sampled fields, scan cleaning and MuJoCo/MJCF import, plus viewer fixes.
- `pinneapple_systems.digital_twin.conditioning`: causal per-signal telemetry conditioning (Hampel, rate limit,
  dt-aware filter, gap and stale handling).
- `pinneapple_arena.autoresearch`: fixed-budget trial loop with keep/revert over a tunable block, with random-search
  and LLM proposers.
- Advanced CadQuery builders (flanged pipe bend, tee, concentric reducer, helical coil, axial fan, involute spur gear,
  pin-fin heat sink) and `register_advanced_templates`.
- 2-D shallow-water finite-volume solver (MUSCL-HLL, wet/dry, drawable walls, probes and health checks), validated on
  Ritter and Stoker dam breaks.
- Geometry retrieval (D2 descriptors with a caption index), a research-only set-based encounter-feasibility module,
  and benchmarks for the Helmholtz, stiff PK-PD, inviscid Burgers and Sod problems with independent references.

### Fixed
- The sdist did not include `pinneapple_decision`, so `import pinneapple_arena` failed from the sdist since 0.6.0.
  A test now checks that every wheel package is in the sdist include list (#16).
- `beam_bvp_fdm` boundary conditions made every deflection first order (-4.8 % at `nx = 100`); ghost-node boundary
  conditions make it second order for simply supported, cantilever and fixed-fixed beams.
- `spectral.poisson_periodic_fft` used float32 wavenumbers, capping a float64 solve at about 1e-7; it is now exact to
  round-off.
- `tvd_advection_rhs` used the downwind slope in the limiter in both flow directions, so the scheme was not TVD (22 %
  overshoot on a square wave); the upwind ratio of Sweby (1984) is used now.
- `TabulatedSignal` converts tensors with `.numpy()` (NumPy 2 deprecation).

## [0.6.0] - 2026-09-25

### Added
- `pinneapple_decision`: `PhysicsDecisionEngine`, a probabilistic layer that decides which experiment to run next (model,
  training strategy, validation) and never predicts physical results. It includes `pp.decide`, a problem adapter, a
  decision tree, an Arena decision mode and surrogate families for KPI regression, POD reduced-order models, point-cloud
  operators and hybrids (#10, #11, #12).
- `pinneapple_catalog.resources` (58 public datasets, CAD sets, pretrained models and benchmarks with checked licenses;
  research-only items warn and are refused when `PINNEAPPLE_COMMERCIAL_MODE=1`) and `pinneapple_catalog.methods` (every
  solver, training method, equation and problem with code location, references and a validation status computed from the
  tests) (#14).
- `pinneapple_twin3d`: 3-D digital-twin scene export (glTF, transient fields, sensors with alarm envelopes) and a bundled
  three.js viewer (#14).
- `SelfScaledQuasiNewton` trainer: BFGS, SSBFGS and SSBroyden (Urban, Stefanou & Pons 2025) (#14).
- PINNFactory: undeclared functions of the independent variables become exogenous signals fed from tensors or
  `TabulatedSignal` (#14).
- `hybrid_surrogate_physics` workflow: a surrogate of intermediate quantities followed by an explicit physics post-model,
  with Gaussian-process and PCA surrogates (#13).
- Pluggable `grad_method` for `SymbolicPDE` (`autograd`, `finite_difference`, `spectral`).
- Geometry out-of-distribution guardrail (diagonal Mahalanobis) as a fifth component of `PhysicsConfidenceScore`.
- Robin and radiative-Robin boundary conditions in `FDMSolver` (#7).
- `pinneapple_physics.closed_form`: engineering formulas ported back from PINNeAPPle-apps.
- Parametric (Re_tau-conditioned) dense-volume operator workflow.
- Analytic tag-geometry fixtures for `solve_pde`, real standard geometry for 29 of the 40 tag-based presets, and new
  compiler mechanisms (`normal_stress_field`, `thermal_bc`).
- CI: pre-commit configuration that enforces the ruff settings of `pyproject.toml`.

### Changed
- **Behavior:** `PhysicsConfidenceScore` now has five components instead of four, so scores change.
- **Behavior:** the `euler_bernoulli_beam` residual used the wrong sign for a compressive axial load; it now uses the
  beam-column equation `EI w'''' + F w''`, so compressive loads destabilize the beam as they should.
- **Breaking:** `AleatoricHead` takes the input `x` and a new required `in_dim` constructor argument. Before, its
  log-variance head received the model's point prediction, so the variance was not heteroscedastic.
- **Behavior:** `AFNOLayer` was rewritten to match Guibas et al. (2022) (block-diagonal weights shared across frequencies
  and frequency-domain soft-shrinkage). Before, it was a per-mode spectral convolution under the AFNO name. Checkpoints
  trained with the old layer do not load.
- **Behavior:** Noether integrations are marked research-only, behind a license guard (#9).
- **Behavior:** `solve_pde()` now fails loudly when tag-based conditions have no real geometry input instead of continuing.
- CI no longer hides failures with `|| true`.
- Repository URLs and the citation version were updated after the move to the `PINNeAPPle-Labs` organization, and a dead
  `Changelog` link was removed from `pyproject.toml`.

### Fixed
- Arena fed graph models their own targets as node features (target leak) (#8).
- Conformal prediction used linear interpolation for the quantile instead of the exact order statistic
  `ceil((n+1)(1-alpha))`, which understated intervals for small calibration sets.
- `MCDropoutWrapper` and the uncertainty decomposition called `model.train()` on the whole model, which switched BatchNorm
  to batch statistics and permanently changed the caller's running statistics. Only Dropout-family modules are activated now.
- `DFSPH` alpha was missing the stabilizing `sum_j ||m_j grad W_ij||^2` term of Bender and Koschier.
- `STLDomainBatchBuilder` ignored the `y_bc` of Neumann and Robin conditions.

## [0.5.0] - 2026-09-11

First release on PyPI. The git tag `v0.5.0` points to 2026-09-08; the files uploaded to PyPI on 2026-09-11 also contain
the work committed up to 2026-09-11 12:53 (-03:00), for example the evidence graph, retrieval and the OPC-UA/Modbus
adapters listed below.

### Added
- New packages: `pinneapple_registry` (local artifact registry for models, datasets and experiments),
  `pinneapple_perception` (physics from images, video and audio, with a PIV extractor validated on a known shift),
  `pinneapple_pdb` (named benchmark catalog), `pinneapple_hub` (model hub with a governance gate:
  `scripts/validate_model_card.py` rejects a model card without `validation_metrics` and `reference_source`), and
  `pinneapple_llm` modules for constrained CAD/geometry drafting and a LoRA fine-tuning pipeline.
- `process_components` and `component_library`: generic process-engineering physics and reference component models.
- Astrophysics and space specialization: Kepler two-body, Clohessy–Wiltshire relative motion, J2 perturbation, spacecraft
  attitude, CR3BP, GR light bending and Shakura–Sunyaev disk presets.
- Verification: `PhysicsGuardrail` dimensional-analysis and conservation checks, reference-data fetch from UPD zarr, grid
  convergence analysis, `evidence_graph`, tool and architecture recommendation with an adversarial critique, and the
  verification modules moved into `pinneapple_analysis.verification`.
- Analysis and twins: Kalman-filter state estimation as a general `pinneapple_analysis` module, trend-based remaining
  useful life in `digital_twin`, OPC-UA and Modbus stream adapters with live simulator tests, and MQTT/Kafka broker
  integration tests.
- Retrieval: local semantic search over a real corpus, live arXiv search, and `answer_with_context` for grounded RAG answers.
- Simulation: OpenFOAM binary and polyMesh readers, WALE LES, ANSYS Fluent coupling, IGES import, LBM3D obstacles and LES
  with a turbulence-model selector, and CGNS/Exodus/Fluent readers checked against real writers.
- Arena: physics-aware model ranking and architecture plus hyperparameter search; `PhysicsCase` as a bridge between
  geometry, physics spec, solver and results; a splash-archive adapter with a dense-volume neural-operator workflow;
  a VLM dataset-curation module.
- Dozens of previously unsupported PDE kinds now compile (steady Navier–Stokes, steady heat, plane stress and strain,
  rotating-frame flows, Black–Scholes, Heston, phase-field fracture, axisymmetric variants and more).

### Fixed
- 152 scientific and mathematical findings from a full audit across 29 modules (equation errors, sign conventions, unit
  mismatches, discretization bugs, deviations from the cited papers, broken API signatures).
- `pytest tests/` aborted at collection with four `ModuleNotFoundError`s, silently blocking every test; fixed together with
  four more bugs it exposed.
- `TrimeshBridge.load()` crashed on current trimesh, `export_onnx` failed with torch 2.5 and later, three structural
  presets silently dropped the Lamé lambda, and the LLM CSG cut-operand ordering was ambiguous.
- Packaging: removed a console-script entry point to a missing module (`pinneapple_pdb.cli`) and a license classifier that
  conflicted with the SPDX expression; added an explicit sdist include list (20 MB to 1.8 MB).

### Known issues
- The `Changelog` project URL in the 0.5.0 package metadata points to a `CHANGELOG.md` that did not exist when it was
  published. This file is that changelog.

## Before 0.5.0

Development before the first PyPI release is not itemized. In short: the initial open-source release (2026-01-23), the
PINN Arena and benchmark suite, the refactor into the current mega-modules (2026-05-06), the rename from `pinneaple` to
`pinneapple` (2026-05-18), and the Physics Synthetic Data Factory and the 9-stage pipeline (2026-06-04).

[Unreleased]: https://github.com/PINNeAPPle-Labs/PINNeAPPle/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/PINNeAPPle-Labs/PINNeAPPle/compare/v0.6.3...v0.7.0
[0.6.3]: https://github.com/PINNeAPPle-Labs/PINNeAPPle/compare/v0.6.2...v0.6.3
[0.6.2]: https://github.com/PINNeAPPle-Labs/PINNeAPPle/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/PINNeAPPle-Labs/PINNeAPPle/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/PINNeAPPle-Labs/PINNeAPPle/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/PINNeAPPle-Labs/PINNeAPPle/releases/tag/v0.5.0
