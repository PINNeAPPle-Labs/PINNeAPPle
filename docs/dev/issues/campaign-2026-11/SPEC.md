# SPEC — Experiment campaign issues (PINNeAPPle)

Everything in this folder produces GitHub issues for `PINNeAPPle-Labs/PINNeAPPle`.
Read this file before writing any issue body.

## Hard rules

1. **English only.** No Portuguese anywhere in an issue body.
2. **No author/assistant attribution.** No "Generated with", no "Co-Authored-By", no mention of Claude, Anthropic,
   an AI, an assistant or a chat session. The issues are written as ordinary engineering specifications.
3. **No invented results.** This campaign is about experiments that have not been run yet. Numbers in an issue are
   *targets*, *acceptance thresholds* or *budgets* and must be phrased as such ("acceptance: GCI < 5 %"). Never
   write "we measured", "the run produced", "results show" unless the number exists in this repository today.
4. **Separate what exists from what must be built.** Every issue has a section `## What already exists in
   PINNeAPPle (verified)` and `## What is missing (the work)`. Only name modules, functions and paths from the
   verified inventory below (or from a file you actually read in the repository). If a capability is not in the
   inventory, describe it as something to build — do not guess an API name.
5. **No fabricated citations or DOIs.** Reference literature as `Author (Year), Title, Venue` where you are certain
   of it, and datasets by their real name and publisher (`NOAA OISST v2.1`, `ERA5`, `WOD`, `CMIP6`). Never invent a
   DOI, a URL, a paper title or a measured value from a paper. A short reference list is fine; a wrong one is not.
6. **Honesty about physics.** Every issue states the regime where its model is valid and the effect it cannot show
   (rarefied flow, continuum breakdown, surface tension dominance, information-theoretic limits, energy-condition
   violation, and so on). A claim that a simulation can prove a technology works is a defect.
7. **Write the body only.** No title line, no YAML front matter, no HTML comments. The title comes from the backlog
   file, the labels come from the backlog file, and the series line is added automatically at creation time.

## Mandatory body structure

Use exactly these `##` headings, in this order. Keep the headings verbatim; a checker greps for them.

```
## Summary
     2-4 sentences: the object, the physics, the demonstration, and the claim it can support.

## Question
     The falsifiable question this experiment answers, written as one sentence.

## Why run it
     What it buys: scientific value, engineering value, visual value. Concrete, not marketing.

## Physical model
     Governing equations in plain text/LaTeX-ish, the dimensionless groups that decide the regime,
     the validity range, and what the model deliberately leaves out.

## Geometry and inputs
     How the 3D geometry is generated (parametric CAD, SDF, lattice, mesh from a solver, procedural growth),
     what is optimised, and the datasets or boundary conditions it needs.

## Solver and engine
     Which solver family, and specifically which PINNeAPPle module or external solver (OpenFOAM solver name,
     CalculiX, FEniCS, ...). State the alternative if the first does not exist.

## What already exists in PINNeAPPle (verified)
     Bullet list of the real modules/functions from the inventory that this experiment reuses.

## What is missing (the work)
     Numbered list of the pieces that do not exist today. Each item says what it does and where it lives
     (package/module to add). This is the actual scope of the issue.

## Experiment protocol
     Ordered steps with the case matrix (parameter sweeps, number of cases), convergence policy, seeds,
     compute budget (CPU-hours, an order of magnitude is enough), and what is logged.

## Baselines, references and verification
     The reference solution or analytic result each number is checked against (closed form, mesh refinement /
     GCI, published correlation, conserved quantity, limiting case), and the baselines the optimiser must beat
     (do nothing, uniform lattice, fixed geometry, hand-designed shape).

## Metrics
     Each metric with symbol, definition, unit, and how it is computed from the solver output.

## Deliverables
     Dataset (format, size), figures, the video/render, the report/manifest, the artefacts a reader can re-run.

## Acceptance criteria
     Checkable statements. Good: "re-solved on a 2x refined mesh, the optimum's pressure drop changes by less than
     5 %". Bad: "the design performs well".

## Realizability and evidence class
     Which evidence axes this experiment can fill (mathematical_consistency, numerical_stability,
     physical_plausibility, experimental_validation, technological_feasibility, uncertainty_sensitivity) and which
     stay not assessed. For speculative subjects, the PLAUSIBILITY_CLASS today and what would have to be true to
     move up one class.

## Risks and failure modes
     The three to five ways this experiment fails or misleads, and the mitigation for each.

## Out of scope
     2-4 bullets.

## References
     Real, verifiable sources only. Datasets with publisher; literature as Author (Year), Title, Venue.
```

## Verified capability inventory (as of this writing)

Only these APIs are confirmed to exist. Cite `module.path` or `function()` — do not invent new ones.

**Core / trust**
- `pp.PhysicalProblem`, `pp.solve(problem, method)`, `pp.compare(problem, methods, reference=)`, `pp.trust_card(...)`,
  `pp.validate(model, problem, data=, level=)`, `pp.recommend(text)`, `pp.card(preset)`, `pp.doctor(profile)`,
  `pinneapple check` / `pinneapple_check` (balances, numerics, physics, fv, evidence, rules), `pp.list_presets()`,
  `pp.presets.names()/.load(name).card()/.run()/.validate()/.invert()/.optimize()`.
- `pinneapple_analysis.verification`: `convergence`, `physics_confidence_score`, `evidence_graph`,
  `solver_orchestration`, `geometry_ood_guardrail`, `causal_discrepancy`, `tool_recommendation`.
- `pinneapple_analysis.validation`: `rollout.validate_rollout`, `pinn.validate_pinn`,
  `physics_learning.verify(model, case)`; `pinneapple_analysis.uncertainty.margin.margin_study`,
  `corner_stackup`; `pinneapple_analysis.sensitivity`; `pinneapple_analysis.cost.equal_budget` +
  `profiler`; `pinneapple_check` evidence/trust report.

**Experiment recording and speculative physics**
- `pinneapple_lab.experiment(name, title=, domain=, methods=, params=)` with `ex.log_metric`, `ex.check`,
  `ex.add_file`; `python -m pinneapple_lab scaffold|register|seal --explain`; manifests, A-F grade, Lab seal.
- `pp.futures`: `ask(question) -> Brief`, `run(name, dim=, quick=)`, `names()`, `atlas()`, `validate()`;
  `pinneapple_futures.Scenario/Proposal/Input/Claim/Hypothesis/Derived`,
  `pinneapple_futures.plausibility.assess` with axes `mathematical_consistency`, `numerical_stability`,
  `physical_plausibility`, `experimental_validation`, `technological_feasibility`, `uncertainty_sensitivity`
  and verdicts `SUPPORTED|CONDITIONAL|CONTRADICTED|NOT ASSESSED` (physics) and
  `FEASIBLE|CONDITIONAL|NOT FEASIBLE|NOT ASSESSED` (engineering);
  `pinneapple_futures.schema.PLAUSIBILITY_CLASSES = established|extrapolated|hypothetical|fictional`,
  `PROVENANCE_KINDS = constant|measured|design_input|derived|hypothesis`, `STAGES =
  speculation|formalization|modeling|experimentation|assessment`; existing demonstrators
  `pinneapple_futures/labs/exoplanet.py`, `mars_habitat.py`, `programmable_matter.py` (the last solves
  `d/dx[k dT/dx] = 0` with `k(T)`, graded `k(x)` and a switching `k` in closed form);
  docs `docs/guides/futures_lab.md`.
- `pp.science.atlas()`, `pp.science.experiments(area=, level=, kind=, dim=)`, `pp.science.run(name, dim=)`;
  `pinneapple_science.registry.experiments`; docs `docs/guides/science.md`.

**Geometry generation and design**
- `pinneapple_design.geometry.gen`: `sdf_shapes`, `primitives`, `domains`, `domains3d`, `airfoil`, `car2d`,
  `cadquery_gen`, `cadquery_advanced`, `genetic_sdf`, `threaded_connection`.
- `pinneapple_design.geometry`: `csg`, `mesh`, `mesh_collocator`, `ops`, `sample`, `io`, `builders`, `optimize`,
  `trough`, `vehicle_cad`, `vehicles3d`, `retrieval`, `bodies`.
- `pinneapple_design.topology`: `optimize`, `DesignChain`, `TopOptResult`, `finite_difference_check`,
  `DensityFilter`, `SymmetryMap`, `heaviside`, `MMA`, `oc_update`, `Compliance2D`, `Q4Grid`,
  `StokesDissipation2D` (Borrvall–Petersson), `ThermalCompliance2D`, `elasticity_ke`, `conduction_ke`;
  `pinneapple_design/topology/magnetic.py`; 2-D problems today (3-D is work to do);
  docs `docs/guides/topology_optimization.md`.
- `pinneapple_design.aero`: `aircraft`, `aircraft3d`, `airliner`, `airliner3d`, `airframe`, `panel`, `vlm`,
  `mesh`, `case`, `case3d`, `dataset`, `field_dataset`, `optimize`, `optimize3d`, `optimize_airliner`,
  `surrogate`, `train`, `verify`, `render_blender`, `buildup`, `geometry`, `geometry_onnx`.
- `pinneapple_design.thermal`: `heatsink/` (`engine`, `field`, `sizer`, `surrogate`, `trust`, `monotone`),
  `fin_array_speed`, `fin_cell`, `board_cooling` (mission-style run with `run_mission`).
- `pinneapple_design.bricks`: `Assembly`, `analyse_assembly`, `CantileverGoal`, `BridgeGoal`, `TowerGoal`,
  `evolve`, `surrogate_search`, `compare`, `visuals` (interlocking-brick statics, Newton replay, search).
- `pinneapple_design.surrogate_inverse`: `core.fit_surrogate`, `core.descend`, `core.VerificationCard`,
  `core.trust_report`; cases `heatsink_fins`, `brake_rotor`, `transdermal_patch`, `suspension_arm`, `reflex_port`.
- `pp.design(text, answers=, out_dir=)` and `pinneapple_design.intent` (goal → verified design or refusal).
- `pinneapple_reason`: `reason(*cases.roof())`, `chain.explain()`, `write_report(chain, out)`,
  `LIBRARY.mode(name).evaluate(params)`, `register_mode`, `register_countermeasure`.

**Solvers (in-process)**
- `pinneapple_simulation.numerical_solvers`: `shallow_water_fv` (MUSCL+HLL, Toro; validated against Ritter and
  Stoker), `free_surface`, `sph`, `isph`, `dfsph`, `sph_boundaries`, `lbm`, `lbm_tunnel`, `ns3d_spectral`,
  `compressible_fv`, `stable_fluids`, `tvd_advection`, `elasticity3d_fdm`, `solid_fem`, `plane_stress_fem`,
  `beam_fe`, `nonlinear_beam_fem`, `beam_bvp_fdm`, `immersed_boundary_fdm`, `reaction_diffusion`,
  `kuramoto_sivashinsky`, `nbody_rebound`, `pipe_network`, `water_hammer_moc`, `thermal_channel_2d`,
  `eddy_current_fdm`, `fdm`, `fdm3d`, `fem`, `fvm`, `spectral`, `meshfree`, `symbolic_fd`, `problem_runner`,
  `openfoam_bridge`, `fenics_bridge`, `jax_cfd_solver`, `external_flow`, `internal_flow`, `cfd_pipeline`,
  `cylinder_wake`, `stl`, `discrete_time_pinn`, `xtfc_ivp`/`xtfc_pde`/`xtfc_subdomain`/`xtfc_general`,
  `operator_datasets`, `operator_benchmarks`, `ssa`, `sst`, `emd`-family decompositions, `hilbert_huang`.
- `pinneapple_simulation.external_solvers`: `openfoam`, `calculix`, `fenics`, `fluidx3d`, `genesis`, `geotaichi`,
  `mujoco`, `newton_physics`, `modelica`, `matlab`, `ansys`, `openradioss`, `turbodesigner`, `cfd_formats`,
  `campaign_cases`. OpenFOAM and CalculiX are separate installs and must be called through these bridges.
- `pinneapple_simulation.particle_dynamics`: `molecular_dynamics` (Lennard-Jones + velocity Verlet, differentiable),
  `particles` (SPH), `mpm`, `geo_mpm`, `rigid_body`, `terramechanics`.
- `pinneapple_simulation.meshless`: `pinn`, `stencils`, `problems`, `study`, `domains`, `methods`, `contracts`.
- `pinneapple_simulation.electromagnetics`: `statics`, `fdtd`, `waves`, `mie`, `media`, `materials`, `fem`, `eddy`,
  `hysteresis`, `machine`, `machines`, `antenna` (external FEM runner, DOE/NSGA-II/active learning),
  `reflector` (physical optics quiet zones), `periodic` (Floquet unit cells, FNO vs Bayesian optimisation),
  `postprocess`, `analytic`, `verification`, `mesh`, `units`; plus `pp.em`, `pp.maxwell.maxwell_checks`.
- `pinneapple_simulation.flight`: `catalog.build("a320")`, `trim`, `stability_derivatives`, `simulate_takeoff`,
  `takeoff_metrics`, `raymer_takeoff`, `turbulence_encounter`, `turbulence_spec`, `pratt_check`,
  `visuals.flight_scene`, `verification.run`; modules `aero`, `aircraft`, `control`, `encounter`, `gear`, `mass`,
  `propulsion`, `rigid_body`, `simulate`, `trim`, `figures`.
- `pinneapple_simulation.rover.autolab`: `RoverMission(demo_site(), text).run(out_dir)`, `SoilBelief`,
  `RiskPlanner`, `grade_table`, `compare_contact_models`; `VehicleSimulator(model, terrain, [soils])`.
- `pinneapple_simulation.multibody`: `arm`, `tree`; `robot_sim`: `arms`, `learn`, `sim`, `spec`, `checks`;
  `vehicle`: `build`, `scenarios`, `sim`, `spec`, `viper`, `viz`, `dashboard`.
- `pinneapple_simulation.geophysics`: `energy_balance`, `richards`, `sphere_shallow_water`.
- `pinneapple_simulation.pipelines.freecad_openfoam`: `PipelineSpec(..., template="pipe_bend")`,
  `run_pipeline(spec, folder, inputs)`, `studies.pipe_bend()/heated_pipe()/sphere()`,
  `Sweep(spec, doe(bounds, n), folder).run(parallel=)`, `fit_surrogate(rows, inputs, outputs)`.
- `pinneapple_simulation.accelerated_foam`: `case`, `solvers`, `learned`, `warm_start`, `ldu`, `benchmark`,
  `guess_server`, `build`.
- `pinneapple_tools.automation`: `Workflow.load`, `Runner(wf, folder, inputs).run(approve=)`, `interpret`.

**Neural models and training**
- `pinneapple_neural.architectures`: `neural_operators` (`fno`, `deeponet`, `ms_deeponet`, `pino`, `uno`,
  `transolver`, `gno`), `pinns`, `physics_aware`, `graphnn`, `recurrent`, `classical_ts`
  (`arima`, `tcn`, `var`, `kalman`, `ekf`, `enkf`, `ukf`), `afno`, `siren`, `reservoir_computing`, `rom`,
  `transformers`, `convolutions`, `continuous`, `autoencoders`, `hash_grid`, `modified_mlp`, `adapters`,
  `registry`, `instantiate`, `catalog`.
- `pinneapple_neural.callbacks.PhysicsMonitor(problem, reference=, every=)`; `model registry`/zoo:
  `pp.zoo.list()/load(name)/validate(name, inputs)`, `pp.datasets.load(name, split=)`,
  `pinneapple_catalog.validated_methods()`, `pinneapple_catalog.methods.method_status()`.
- `pinneapple_adaptation.transfer_learning`.

**Data**
- `pinneapple_data.datasets`: `timeseries` (synthetic ODE cases), `real_world`, `physics`, `geometry`,
  `external_sources` (`noaa` — NOAA ERDDAP/NCEI, `copernicus`, `nasa_open_data`, `nasa_power`, `open_meteo`,
  `cptec_inpe`, `brazil_data_cube`, `materials_project`, `nomad`, `nrel`, `designsafe`, `openfoam_tutorials`,
  `su2`), `deepmind_mgn`, `download`, `registry`; zarr stores and iterables (`zarr_store`, `zarr_iterable`,
  `zarr_shard_iterable`, `prefetch_adaptive`); `pinneapple_data.contracts`, `lineage`, `audit`, `doe`,
  `parameter_sampling`, `active_learning`, `data_health`.
- `pinneapple_arena.external`: `MODELS_BY_SLUG`, `call_infer(slug)`, `run_benchmark`, `licence_counts()`.

**Time series and forecasting (the ENSO precedent)**
- `pinneapple_systems.time_series.enso`: `dataset.build_dataset()`, `dataset.to_series(ds)`,
  `study.protocol(months)`, `study.run_model(name, data, prot)`, `study.event_hindcast`, `pipeline.solver_checks()`;
  programme `examples/enso_forecasting/run_study.py`, committed results in `examples/enso_forecasting/results/`,
  guide `docs/guides/enso_forecasting.md`, tests `tests/test_enso.py`.
- `pinneapple_systems.time_series.forecast_lab`: `study.run_zoo`, `study.evaluate`, `presets.case(name)`,
  `Protocol(horizons)`, `baselines`, `hybrid`, `learned`, `physical`, `plots`, `renders`;
  `pinneapple_systems.time_series.predictability.predictability_report(y)`.

**Visualisation, rendering, video, world models**
- `pinneapple_twin3d.scene`: build/export a scene with `scene.json`, `geometry.glb`, `fields.bin`, optional
  `with_viewer=True` (bundled web viewer) and `usd=True` (OpenUSD, `add_physics`, `set_gravity`); modules
  `gallery`, `openfoam`, `primitives`, `scan`, `wear`, `usd`, `serve`.
- `pinneapple_blender.export` (point cloud + fields → numbered `.ply` sequence), `pinneapple_blender.render.build_scene`
  (headless Blender subprocess), `pinneapple_blender.cycles` (`CyclesScene`, `add_surface`, `add_plane`, `camera`,
  `render`, `render_frames`, `frames_to_video`, `available()`).
- `pp.dashboard.Dashboard(...).to_png/.to_html/.to_video(gif=)`, `pp.viz.field_view(mesh, fields, deform=,
  markers=)`, `pinneapple_analysis.validation.error_map.error_map(points, pred, ref, regions=)`.
- `pinneapple_tools.nvidia.omniverse.FieldStage(path).add_mesh(..., fields=, color_by=)`, `.add_time_samples`,
  `.set_trust`, `omniverse.validate_stage(path)`.
- `pinneapple_worldmodel.physical_world`: `dataset(n, seed)` (videos with exact labels), `physics_checks(video)`,
  `corrupt(video, kind)`, `WorldModel().fit(videos)`; `.inspection_data`, `.robot_perception`.
- `pinneapple_flow.plan/run` and the Flow Studio app (`apps/flow_studio`).

**Autonomy layer**
- `pinneapple_autolab`: `AutonomousLab(mission, simulator, folder).run()`, `Mission`, `Limit`, `Simulator`,
  `FunctionSimulator`, `compare_models`, `ValidatedDomain`, `assess_candidates`, `choose_next`,
  `domains.racecar.RaceCarAero`, `flagship_mission()`, `agent.run_agent(text, simulator, folder, domain=)`,
  `agent.interpret_mission`, `core.act`, `dashboard.build_dashboard(folder, out, replay=True)`.
- `pinneapple_query` (`Engine`, `Planner`, `Method`, `Offer`, `Answer`), `pinneapple_condense`,
  `pinneapple_physics.symbolic_pde.text.parse_system`, `pp.closed_form` (pipe flow, compressible, heat,
  rocket nozzle), `pinneapple_physics.astro.conjunction`.

**Not in the library today (do not pretend otherwise)**
Acoustics beyond the 1-D NDT wave FDTD in `pinneapple_analysis.inspection.acoustic` and the ultrasonic phased
array module; optical ray tracing / BRDF / radiometry; a 3-D incompressible Navier-Stokes solver in-process beyond
`ns3d_spectral`, `lbm`, `stable_fluids` and the OpenFOAM bridges; DEM/contact beyond `terramechanics` and the
external `geotaichi`/`genesis` bridges; fracture and damage mechanics; shell and membrane finite elements;
phase-field and free-boundary multiphase; rigid-body–fluid coupling; general relativity and geodesic integration;
quantum circuit simulation (`pinneapple_quantum` exists — check its API before citing it); atomistic/DFT;
multi-agent construction planning; assembly sequencing for modular matter.

## Tone

Write like an engineer who has to defend the experiment: specific, quantitative, unimpressed by its own idea.
No marketing adjectives. No emoji. No exclamation marks. British or American spelling, consistently within one
issue (American preferred).
