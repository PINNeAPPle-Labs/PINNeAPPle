# AGENTS.md

Guide for coding agents (Gemini CLI, Antigravity, Claude Code, Codex, Cursor, ...) working with or on PINNeAPPle.
Every Python block in this file is run by `tests/test_agents_md.py`; keep them short and correct.

## What PINNeAPPle is

A Physics AI library: one problem description (`pp.PhysicalProblem`) solved by physics-informed networks, neural
operators, graph networks or classical solvers (FDM, FEM, OpenFOAM, CalculiX bridges) through one call, and a trust
layer that checks every result against the physics (residuals, conservation, boundary conditions, references,
applicability) before anyone uses it.

## Install

```bash
pip install pinneapple                 # core: numpy, scipy, torch, sympy, matplotlib
pip install "pinneapple[cfd]"          # + geometry, meshing, 3-D viewing (OpenFOAM and CalculiX are separate installs)
pip install "pinneapple[enterprise]"   # + serving, signatures, the CI linter, the hub, the MCP server
pip install -e ".[dev]"                # from a checkout, with pytest and ruff
pinneapple doctor start                # what this machine still needs, with the fix per line
```

On Colab: the notebooks in [`notebooks/colab/`](notebooks/colab/README.md) install it in their first cell.

## Use these APIs first

| Task | API |
|---|---|
| Is this installation ready for the work intended (`start`, `cfd`, `research`, `enterprise`, `all`) | `pp.doctor(profile)` -> report with `ok`, one fix per missing item |
| Which preset, method and model for a problem described in a sentence (English or Portuguese), and why | `pp.recommend("...")` -> `Recommendation` (`.preset`, `.method`, `.model`, `.explain()`) |
| What a preset takes, what it gives back, and the smallest program that uses it | `pp.card(name)` -> `Card` (`.coordinates`, `.fields`, `.snippet`, `.requires`) |
| A project of the user's own, outside this repository (`plain` / `research` / `cfd`) | `pp.new("name", template=..., preset=...)`, `pinneapple new`; `pp.new` refuses to overwrite |
| Use a pre-trained, verified model or a dataset | `pp.zoo.list()`, `pp.zoo.load(name).predict(...)`, `pp.zoo.validate(name, inputs)`, `pp.datasets.load(name, split=...)` |
| Find a ready-made tool, trained model, geometry, dataset, flow or example with code that runs; compose a custom tool from nodes | `pinneapple_hub.library.search(load_index(), q, kind=...)`, `pinneapple_hub.library.compose.compose({...})`; the Hub app `apps/hub` ([docs](docs/apps/hub.md)) |
| Describe a problem | `pp.PhysicalProblem.from_preset(name)`, `ProblemBuilder(...)...build()` + `pp.PhysicalProblem.from_pde_spec`, `pp.list_presets()` |
| Solve it | `pp.solve(problem, method, **options)`: `"pinn"`, `"fem"`, `"analytic"`, `"reference"`, `"exact"`, `"external"` (`pp.list_methods()`) |
| A preset whose boundaries are named geometry tags | `pp.solve(preset, method="pinn")` sends the preset's own box or cylinder (`pde_environment/presets/tag_geometry.py`, 32 of the 40 tagged presets) and records it in `Solution.info["geometry"]`; the other 8 describe a real airfoil/car body/furnace and still raise `TagConditionsUnresolved` |
| Compare methods on the same points | `pp.compare(problem, methods, reference=...)` |
| Check a result | `pp.trust_card(grid, values, pde=..., dirichlet=..., reference=..., train_bounds=...)` (decision APPROVED / REVIEW / REJECT) |
| Validate a model (zoo, PyTorch, ONNX, callable) | `pp.validate(model, problem, data=..., query=..., level="quick" / "standard" / "thorough")` (credibility per factor, decision pass / warn / fail; [docs](docs/guides/pp_validate.md)) |
| Validate an autoregressive rollout (trust horizon, drift, long-run statistics) | `pinneapple_analysis.validation.rollout.validate_rollout(pred, ndim, ref, long_ref=...)` or `pp.validate(step, {"rollout": {...}}, data=...)` ([docs](docs/guides/rollout_validation.md)) |
| Validate a trained PINN (fresh-point residual, failure modes, exact solutions) | `pinneapple_analysis.validation.pinn.validate_pinn(net, problem)` or `pp.validate(net, {"pinn": "burgers_1d"})`; `PINNMonitor` during training ([docs](docs/guides/pinn_validation.md)) |
| Physics checks during training (Lightning, PhysicsNeMo, Hugging Face) | `pinneapple_neural.callbacks.PhysicsMonitor(problem, reference=..., every=N)` with `PhysicsValidationCallback` / `PhysicsNeMoHook` / `PhysicsTrainerCallback` ([docs](docs/guides/training_callbacks.md)) |
| Worked validation examples per model family and physics | `examples/validation/NN_*.py` (`run(smoke=True)`), each ending in `pp.validate` + trust card + `pinneapple check` ([docs](docs/guides/validation_gallery.md)) |
| Deploy decision from your own checks | `pinneapple_analysis.trust.TrustReport`, `Check` |
| Which methods are validated | `pinneapple_catalog.validated_methods()`, `pinneapple_catalog.methods.method_status()` |
| Third-party physics-ML libraries: references (repository, licence with its source, citation, install) and run one through its own package with an honest status | `pinneapple_arena.external`: `MODELS_BY_SLUG[slug]`, `call_infer(slug)`, `run_benchmark(select_slugs(category=...))`, `licence_counts()`; `python -m pinneapple_arena.external --list` |
| Equal-budget benchmark | `pp.compare` with the same epochs / collocation / seeds per method (below); `pp.benchmark.run(...)` where installed |
| A field, a prediction against its reference, or a deformed shape on its mesh in an app | `pp.viz.field_view(mesh, fields, deform=..., markers=...)` + `pinneapple_analysis.validation.error_map.error_map(points, pred, ref, regions=...)`; page module `/shared/fieldview3d.js` ([docs](docs/core_concepts/studio.md)) |
| Animations with synced charts | `pp.dashboard.Dashboard(...)`: `.to_png`, `.to_html`, `.to_video(gif=...)` |
| Lint results and models for physics in CI | `pinneapple check [paths]` (conservation, BCs, residual, units, OOD, regression; [docs](docs/guides/physics_checks_in_ci.md)) |
| Helpers that used to live in apps and examples | `pp.to_jsonable`, `pp.as_tensor`, `pp.dominant_frequency`, `pp.closed_form` (pipe flow, compressible, heat), `pp.uncertainty` (resampling, drift, tolerances), `pp.assess_drift`, `pp.propagate_tolerances`, `pp.score_against_reference`, `pp.excitation`, `pp.reporting.paperkit` ([inventory](docs/dev/LIBRARY_CONSOLIDATION.md)) |
| Electromagnetics: references, materials, solvers, physics checks of any field | `pp.em` (statics, waves, Mie, media, FEM), `pp.maxwell.maxwell_checks(grid, fields, ...)`, `pinneapple check` with `physics = "maxwell"`, `pp.em.verification.run()` ([docs](docs/guides/electromagnetics.md)) |
| Printed end-fire antennas with an external FEM solver (EMerge): limits, checks, runner, surrogates, DOE / NSGA-II / active learning | `pinneapple_simulation.electromagnetics.antenna`: `bounds.min_length(10, 2e9)`, `antenna_checks(row)`, `ava.layout(AVAParams(...))`, `runner.ava_simulator(dir).run(designs, freqs)`, `surrogates.lodo(table, target)`, `study.open_study(out)` / `study.package_study(dir, out)`, 3-D (balloon, near field, viewer, Blender) `study.render_3d({name: params}, out)` ([docs](docs/guides/antenna_design.md)) |
| Compact-range quiet zones by physical optics; periodic unit cells (Floquet, FNO vs Bayesian optimisation at equal budget) | `pinneapple_simulation.electromagnetics.reflector`: `CompactRange(...)`, `evaluate(cr)`, `study.validate(cr)`, `study.qz_study(cr, gains, sides, rolls, out)`; design against a spec (both polarisations, principal cuts, GP with 95 % coverage, design maps, Pareto front, smallest reflector verified by PO): `reflector.design.qz_report(cr, Spec(1.0, xpol_max_db=-25))`, `design.doe(...)`, `design.fit_qz_surrogate(table)`, `design.smallest_reflector(...)`, `design.run_design_study(...)`, `design.generic_2ghz()`; `pinneapple_simulation.electromagnetics.periodic`: `tcda.simulate(TCDACell(), freqs, scans)`, `benchmark.campaign`, `benchmark.fit_fno`, `benchmark.equal_budget` ([docs](docs/guides/compact_range.md), [docs](docs/guides/unit_cell_benchmark.md)) |
| Electric machines: interior-PM FEA (torque, cogging, L_d/L_q, MTPA, losses) | `pinneapple_simulation.electromagnetics.machines`: `IPMMachine(prius_2004())`, `.operating_point(I_peak=, gamma=)`, `.mtpa(I)`, `.inductances(i_d, i_q)`, `.losses(rpm, i_d, i_q)`; drive: `machines.drive.FluxMap.from_fea(m, I_max)`, `Drive(fm, R, I_max, V_max).max_torque(w)` / `.operating_point(T, w)` / `.efficiency_map(...)`, `cycle_energy(drive, Vehicle(...), *wltc_class3b())`; verification `machines.verification.run()` ([docs](docs/guides/electromagnetics.md#interior-pm-traction-motor-the-2004-toyota-prius)) |
| 3-D magnetostatics and eddy currents on edge (Nedelec) elements: nonlinear B-H by Newton, energy and co-energy, GetDP as the reference on the same gmsh mesh | `pinneapple_simulation.electromagnetics.fem3d`: `solve_magnetostatic(mesh, nu, J=, Br=, nonlinear={tag: curve})`, `solve_eddy(mesh, nu, sigma, omega, J)`, `box_mesh` / `read_msh22`, `sphere.SphereCase` + `mesh_sphere` + `solve_native`, `getdp3d.sphere(msh, case)`; Flow node `em.edge3d_verification` ([docs](docs/guides/electromagnetics.md#3-d-edge-elements-the-solver-path-for-3-d-machines)) |
| Skew and end effects of a PM machine in 3-D against multi-slice 2-D (cogging, same mesh for every angle and skew) | `pinneapple_simulation.electromagnetics.machines3d`: `SPM3D(SPMMachine(...), n_theta=720, stack_layers=6, end_length=0.02, end_layers=1)`, `.cogging(thetas, step_skew(...))`, `.multislice(...)`, `skew.skew_study(model, totals, checkpoint=...)`; Flow node `em.spm_skew_3d` ([docs](docs/guides/electromagnetics.md#3-d-skew-cogging-in-3-d-against-multi-slice-2-d)) |
| Run FreeCAD / CfdOF / OpenFOAM / ParaView and keep the evidence | `pinneapple_tools.automation`: `Workflow.load(...)`, `Runner(wf, folder, inputs).run(approve=[...])`, `interpret(task, catalog())` ([guide](docs/guides/engineering_gui_agent.md)) |
| A parametric FreeCAD model to a checked OpenFOAM result, headless and reproducible (patches by rule, snappyHexMesh, simpleFoam / pimpleFoam, passive scalar, sections, references); sweeps over the FreeCAD parameters (one process per case, resume) and a surrogate with held-out error | `pinneapple_simulation.pipelines.freecad_openfoam`: `PipelineSpec(..., template="pipe_bend")`, `run_pipeline(spec, folder, inputs)`, `studies.pipe_bend()` / `heated_pipe()` / `sphere()`, `Sweep(spec, doe(bounds, n), folder).run(parallel=2)`, `fit_surrogate(rows, inputs, outputs)`; Flow nodes `cfd.*` ([guide](docs/guides/freecad_openfoam_pipelines.md)) |
| Flight dynamics: 6-DOF aircraft on the vortex lattice, MIL-F-8785C turbulence, take-off, gust encounters, animated 3-D scenes | `pinneapple_simulation.flight`: `catalog.build("a320")`, `trim(ac, V, h)`, `stability_derivatives`, `simulate_takeoff(ac)` + `takeoff_metrics` + `raymer_takeoff`, `turbulence_encounter(ac, V, h, turbulence_spec(h, "severe"))`, `pratt_check`, `visuals.flight_scene(ac, hist, wind=...)`, `verification.run()` ([docs](docs/guides/flight_dynamics.md)) |
| Interlocking-brick builds: statics with stud clutch, Newton replay, self-modifying search | `pinneapple_design.bricks`: `Assembly`, `analyse_assembly`, `CantileverGoal` / `BridgeGoal` / `TowerGoal`, `evolve`, `surrogate_search`, `compare`, `visuals` ([docs](docs/guides/interlocking_bricks.md)) |
| A plain-language goal to a verified, buildable design (or the questions it needs, or a refusal saying what is missing) | `pp.design(text, answers=..., out_dir=...)`, `pinneapple design "<goal>"`, MCP `design`; templates in `pinneapple_design.intent` ([docs](docs/guides/design_from_intent.md)) |
| A rover mission in plain language: drive, get stuck, learn the soil from the wheels, replan with the learned risk, verify (multibody VIPER, several soils, Wong-Reece, EKF, risk-aware routes, trust report, Cycles renders) | `pinneapple_simulation.rover.autolab`: `RoverMission(demo_site(), text).run(out_dir)`, `SoilBelief`, `RiskPlanner`, `grade_table`, `compare_contact_models`; `VehicleSimulator(model, terrain, [soils], soil_map=...).run(route=..., stop=...)`; Flow node `rover.learn_planet` ([docs](docs/guides/rover_learns_the_planet.md)) |
| Why a design needs each of its elements, computed (naive concept -> failure mode with its mechanism and numbers -> countermeasure -> check again), for any scale | `pinneapple_reason`: `reason(*cases.roof())`, `chain.explain()`, `write_report(chain, out)`, `LIBRARY.mode(name).evaluate(params)`, `register_mode` / `register_countermeasure`; MCP `reason`; Flow nodes `reason.*` ([docs](docs/guides/chain_of_whys.md)) |
| A mission in plain language planned and run through controlled library tools, with evidence and a trust report (board cooling: learn the temperature field, optimise a heat sink + fans, verify on physics) | `pinneapple_autolab` (`Tool`, `Toolbox`, `TemplatePlanner`, `LLMPlanner`, `execute`), `pinneapple_design.thermal.board_cooling.run_mission(text, out_dir=...)`; a twin that tells a changed process from a wrong sensor from a model out of its domain (confusion matrix on controlled cases): `pinneapple_systems.digital_twin.pipe_loop.run_mission(out_dir=...)`; Flow nodes `autolab.board_cooling`, `autolab.pipe_twin` ([docs](docs/guides/autonomous_physics_lab.md)) |
| Forecast a noisy or near-random series without fooling yourself (predictability first, purged walk-forward, untouched test, DM tests) | `pinneapple_systems.time_series.predictability.predictability_report(y)`, `pinneapple_systems.time_series.forecast_lab` (`study.run_zoo(data, specs, Protocol(horizons))`, `presets.case(name)`) ([docs](docs/guides/near_random_forecasting.md)) |
| A problem description to a runnable flow of library calls (node graph, GUI or headless) | `pinneapple_flow.plan(text)`, `pinneapple_flow.run(flow)`, `python -m pinneapple_flow run flow.json` / `studio`; equations as safe text `pinneapple_physics.symbolic_pde.text.parse_system`, any parsed system by finite differences `pinneapple_simulation.numerical_solvers.symbolic_fd.solve_fd` ([schema](docs/guides/flow_schema.md), [app](apps/flow_studio/README.md)) |
| Predict El Niño: ENSO data (OISST, WOD thermocline, ERA5), recharge / delayed oscillators, an intermediate coupled model with checks against theory, PINNs, LIM / CNN, EnKF, the forecast_lab protocol | `pinneapple_systems.time_series.enso`: `dataset.build_dataset()`, `dataset.to_series(ds)`, `study.protocol(ds.months)`, `study.run_model(name, data, prot)`, `pipeline.solver_checks()` ([docs](docs/guides/enso_forecasting.md)) |
| Football physics on public data (SkillCorner, Metrica, StatsBomb 360): ball-flight ODE and inverse problem with UQ, a gravity audit of tracking data, Spearman pitch control fitted by likelihood, Kalman-RTS kinematics and metabolic power, trajectory prediction with physical limits, xG features; each against data-only baselines at equal data | `pinneapple_systems.sports.football`: `ball.fit_flight(t, xyz, "drag")`, `ball_study.gravity_audit(flights)`, `pitch_control.control_probability(...)` / `fit_params`, `kinematics.kalman_smooth(xy, dt)`, `kinematics.metabolic_power(v, a)`, `pc_study.compare`, `xg.compare`; data `data.metrica_match(1)`, `data.skillcorner_ball(id)`, `data.statsbomb_shots()`; Flow nodes `football.*` ([docs](docs/guides/football_physics.md)) |
| Derivative trees from one potential (thermodynamics, hyperelasticity, machines, mechanics): every property by differentiation, Legendre transforms with a convexity check, Maxwell relations, consistency of a model or table with one potential; potential-based surrogates (measured: less accurate than a direct surrogate on CO2 near Tc, but consistent) | `pp.potential`: `Potential(fn, vars, names=...)` / `Potential.from_sympy`, `tree(pot, depth, at=...)`, `legendre(pot, swap={"V": "p"}, conjugate_sign=-1, bounds=...)`, `convexity`, `maxwell_relations`, `consistency_check(model, "helmholtz_T_rho", at=...)`; adapters `thermo` (`ideal_gas`, `van_der_waals`, `peng_robinson`, `gibbs`), `hyperelastic.neo_hookean`, `mechanics.double_pendulum` / `hamiltonian`, `machines.machine_coenergy`; `surrogate.PotentialSurrogate` / `DirectSurrogate` / `fit`; Flow nodes `potential.*` ([docs](docs/guides/derivative_trees.md)) |
| Inverse design through a frozen neural surrogate of a closed-form model, every answer re-checked against the model (verification card + trust report); five worked cases | `pinneapple_design.surrogate_inverse`: `core.fit_surrogate(X, y, bounds, log_output=...)`, `core.descend(loss, x0, bounds=..., reparam="sigmoid")`, `core.VerificationCard(...).trust_report()`; cases `heatsink_fins`, `brake_rotor`, `transdermal_patch`, `suspension_arm`, `reflex_port` (`train_surrogate`, `optimize_design`, `verify_design`, `mesh`) |
| A ready-made industrial case (water hammer, pipe network, heat sink, power PCB, footbridge, ...): data sheet with sourced defaults, solve with your parameters, references with measured error, surrogate, inversion, optimisation | `pp.presets.names()`, `pp.presets.load(name).card()` / `.run(**params)` / `.run(fidelity="surrogate")` / `.validate()` / `.invert()` / `.optimize()`; `python -m pinneapple_presets list`; Flow node `preset.run` ([docs](docs/guides/industrial_presets.md)) |
| Science by area: an atlas of every area (available / partial / planned, pointing to the code that covers it) and experiments at academic or research level, theory or virtual bench, in 0D to 3D + time, each checked against cited references with a trust report and a Lab seal (electromagnetism first) | `pp.science.atlas()`, `pp.science.experiments(area=, level=, kind=, dim=)`, `pp.science.run(name, dim=...)` -> `Result` (`.comparisons`, `.trust`, `.save(dir)`), `pinneapple_science.em.fdtd_study.run_fdtd(FDTDConfig.from_dict({...}))`; `python -m pinneapple_science atlas \| list \| run \| report`; Flow nodes `science.run`, `science.atlas` ([docs](docs/guides/science.md)) |
| Record an experiment so the Lab, the site and the Flow Studio see it (manifest, A-F grade, next steps) | `with pinneapple_lab.experiment("name", title=..., domain=...) as ex:` + `ex.log_metric` / `ex.check` (figures saved with `savefig` are recorded); `python -m pinneapple_lab scaffold <name>`, `register <folder>`, `seal --explain <item>`; format `pinneapple_lab.manifest` ([docs](docs/guides/adding_an_experiment.md)) |
| A mission (objectives, limits, variables, simulation budget) on any simulator: plan, checked build, simulate, surrogates compared on held-out data, NSGA-II, triage (favourable / uncertain / out of domain), next simulation chosen by what is unknown, verified recommendation, ledger | `pinneapple_autolab`: `AutonomousLab(mission, simulator, folder).run()`, `Mission`, `Limit(lower_frac_of_baseline=...)`, `Simulator` / `FunctionSimulator`, `compare_models`, `ValidatedDomain`, `assess_candidates`, `choose_next`; racing-car OpenFOAM domain `domains.racecar.RaceCarAero`, `flagship_mission()`; driven by an agent: `agent.run_agent(text, simulator, folder, domain=racecar_domain())` (`provider="anthropic"` for an LLM), `agent.interpret_mission`, `agent.lab_toolbox`, `core.act`; dashboard `dashboard.build_dashboard(folder, out, replay=True)` / `dashboard.serve(folder, out)`, Flow node `autolab.aerodynamicist` ([docs](docs/guides/autonomous_lab.md)) |
| An end-to-end CFD surrogate study on any simulator, stage by stage: DOE with space-filling metrics and resumable, costed runs; grid convergence (GCI); dataset with a held-out region; surrogate families at equal budget with cross-conformal intervals, physics checks on predictions and a written selection rule; Sobol, NSGA-II, acquisition; amortised inverse (MDN); twin page; evidence report. Flagship: the Ahmed body in OpenFOAM vs Ahmed et al. (1984) | `pinneapple_autolab.study`: `campaign.plan_doe` / `run_campaign` / `cost_table`, `mesh_study.grid_convergence`, `dataset.build_dataset` + `region_split`, `benchmark.benchmark(ds, targets, field, physics=...)` -> `Board.select()`, `extras.sobol_on_surrogate` / `pareto_on_surrogate` / `acquire`, `twin.twin_payload` + `write_twin`, `report.write_report` / `evidence_package`; `pinneapple_analysis.inverse_problems.amortized.fit_mdn`; Ahmed: `pinneapple_autolab.domains.ahmed` (`AhmedAero`, `ahmed_cad`, `import_body`, `qa_checks`) and `.ahmed_study` (one call per stage); app `apps/ahmed_twin`; Flow nodes `study.*` ([docs](docs/guides/ahmed_body_end_to_end.md)) |
| Results to NVIDIA Omniverse (OpenUSD with fields, timeline, trust decision), PhysicsNeMo surrogates checked by PINNeAPPle, architecture figures | `pinneapple_tools.nvidia`: `omniverse.FieldStage(path).add_mesh(..., fields=, color_by=)`, `.add_time_samples`, `.set_trust`, `omniverse.validate_stage(path)`, `diagram.architecture_diagram(...)`; worked examples in `examples/nvidia` ([README](examples/nvidia/README.md)) |
| Synthetic physics data and checks for world models (Cosmos alternative): videos with exact labels, video physics checks, inspection samples as USD scene variations, robot camera data from the robot sim | `pinneapple_worldmodel.physical_world`: `dataset(n, seed)`, `physics_checks(video)`, `corrupt(video, kind)`, `WorldModel().fit(videos)`; `.inspection_data.inspection_sample(x, y, r)`; `.robot_perception.arm_motions(arm, n, seed)`, `write_motion_stage`, `camera_frames`, `perception_checks` |
| Chip-package thermal by condensation of repeated blocks (microbumps, TSVs): fine FV reference, Keff baseline, block Schur complements reused across identical blocks, reduced interfaces with an error estimate and refinement, nested condensation with memoised subtrees, learned block operators with verification, Craig-Bampton transient | `pinneapple_condense`: `package.uniform_package((nb, nb))`, `fv.solve_package(pkg)`, `substructure.Condensed(pkg, m).prepare().solve()` + `.observables` / `.recover(state, I, J)` / `.estimate` / `.refine`, `hierarchy.Hierarchy(pkg, m)`, `homogenize.solve_homogenized`, `learned.LearnedBlocks(stack).fit()` + `solve_adaptive`, `transient.CBTransient`; Flow node `condense.package_thermal` ([report](docs/research/physics_condensation.md)) |
| Does a learned thermal model (PINN, surrogate) know the physics beyond its loss: discrete residual, interface flux, boundary, energy, symmetry, scaling and causal interventions against an FV reference; scope, point sufficiency, informative points, architecture-vs-learned provenance | `pinneapple_analysis.validation.physics_learning`: `verify(model, case)` -> `VerificationReport` (`.findings`, `.failed`, `.out_of_scope`, `.trust`), `detection_study()`, `baselines`; cases in `physics_learning_cases` (`Section`, `defective_models`, `surrogate_model`, `pinn_model`); Flow node `analysis.physics_learning` ([report](docs/research/physics_condensation.md)) |
| Physics queries with a planner: `query(quantity, where, time, conditions, accuracy)` answered by the cheapest valid method (read a solved state, superposition, rigorous monotone bound, condensed solve, fine reference) with value, units, uncertainty, validity and provenance; stable physical references (`die@u,v[,w]`, `ubump[I,J]`, `interface:die/tim`, `lid:top`) | `pinneapple_query`: `Engine`, `Planner`, `Method`/`Offer`/`Answer`, `parse_ref`; chip package: `pinneapple_condense.query.package_engine(pkg)` then `eng.query("T", "die@0.5,0.5", conditions={"power": P}, accuracy=0.1)`; Flow node `query.package` ([report](docs/research/physics_condensation.md)) |
| Biology-informed modelling: SBML models as ODE systems (simulate, sensitivities, PINN problem, SBML export), identifiability before fitting, SBINN parameter inference, BINN term discovery in reaction-diffusion, nonlinear PK/PD and PBPK, excitable media (FitzHugh-Nagumo, Aliev-Panfilov) and activation maps, Windkessel-coupled aortic flow in OpenFOAM | `pinneapple_physics.biology`: `read_sbml(path)`, `biomodel(12)`, `ODESystem.simulate / sensitivities / to_problem_spec / forward_model / to_sbml`, `models.ultradian_glucose_insulin`, `models.pk_one_compartment_mm`, `models.minimal_pbpk`, `excitable.aliev_panfilov_2d`, `cell_migration`; `pinneapple_analysis.inverse_problems`: `ode_model.ODEForwardModel` + `identifiability.analyze_identifiability`, `sbinn.SBINN`, `binn.ReactionDiffusionBINN`, `activation_mapping.EikonalPINN`; `pinneapple_simulation.hemodynamics`: `Windkessel`, `split_rcr`, `aorta.setup_case` + `aorta.CoupledRun` ([docs](docs/guides/bio_informed.md)) |
| Conjunction screening and an ideal rocket nozzle in closed form | `pinneapple_physics.astro.conjunction.screen_cw_conjunction(n, state0, sigma_km=...)`, `collision_probability_2d`, `propagate_two_body_j2`; `pp.closed_form.rocket_nozzle.nozzle_performance(...)` |
| Does a design that passes at nominal still pass with its manufacturing tolerances (Monte Carlo p95, corner stack-up, six checks, decision) | `pinneapple_analysis.uncertainty.margin.margin_study(fn, nominal, tolerances, limit)`, `corner_stackup` |
| A river gauge as a digital twin (USGS data at run time, baseflow recession, sustained-departure alarm) | `pinneapple_systems.digital_twin.io.usgs.fetch_iv(site)`, `pinneapple_systems.digital_twin.recession.run_twin(timestamps, values)` |
| Everything from an agent tool call | the MCP server `pinneapple-mcp` ([docs](docs/guides/mcp_server.md)) |
| Coding and engineering agents on a local model (web app, VS Code, terminal), conversations kept for fine-tuning | `python -m pinneapple_llm.workbench serve --workspace <project>` (`doctor`, `eval`, `finetune`), `pinneapple_llm.workbench.Session` ([docs](docs/guides/agent_workbench.md)) |

Check the zoo before training a surrogate: a zoo model comes with its held-out errors, physics checks and validity
range. The geometry library (`pp.geometries`) is being added; use it once `hasattr(pp, "geometries")` is true.

### Pre-trained models and datasets

```python
import pinneapple as pp

print(pp.zoo.list(), pp.datasets.list())
model = pp.zoo.load("airfoil_cst_rans_mlp")           # weights checked against their SHA-256
naca2412 = dict(T0=0.169591, T1=0.144959, T2=0.131433, M0=0.024922, M1=0.071088, M2=0.058960)
out = model.predict(**naca2412, alpha_deg=[4.0])
print(out["Cl"], out["Cl_std"])                         # mean and ensemble spread (OpenFOAM: Cl 0.6625)
report = pp.zoo.validate("airfoil_cst_rans_mlp", inputs={**naca2412, "alpha_deg": [4.0]})
assert report.decisao == "APPROVED"                     # card metrics reproduced, physics checks, validity range
test = pp.datasets.load("airfoil_rans_cst_re4e6", split="test")
print(test.n_rows, test.columns[:4])
```

### Before any physics: is this machine ready, and what should I run

```python
import pinneapple as pp

report = pp.doctor("start", verbose=False)          # start | cfd | research | enterprise | all
assert report.ok, [c.name for c in report.failures]  # each failure carries the command that fixes it

rec = pp.recommend("heat conduction in a heat sink cooled by a fan")
print(rec.preset, rec.method, rec.confidence, rec.matched)   # a preset, a method that runs, the terms matched
print(pp.card(rec.preset))                           # what it takes, what it gives back, the program to run
```

### Solve and compare

```python
import numpy as np
import pinneapple as pp
from pinneapple_physics.pde_environment.builder import ProblemBuilder

# laplace(u) = f on the unit square, u = 0 on the boundary, exact u = sin(pi x) sin(pi y)
spec = (ProblemBuilder("poisson").domain(x=(0, 1), y=(0, 1)).fields("u").pde("poisson")
        .bc("dirichlet", field="u", value=0.0, on="x_boundary")
        .bc("dirichlet", field="u", value=0.0, on="y_boundary").build())
problem = pp.PhysicalProblem.from_pde_spec(spec)
ctx = {"source_fn": lambda X, ctx=None: -2 * np.pi**2 * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])}
exact = lambda X: (np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1]))[:, None]

fem = pp.solve(problem, "fem", ctx=ctx, n=24)
pinn = pp.solve(problem, "pinn", ctx=ctx, epochs=200, hidden_dim=16, n_layers=2, n_collocation=256)
table = pp.compare(problem, [fem, pinn], reference="exact", options={"exact": {"fn": exact}})
print(table)                                   # relative L2, RMSE, max error, train and inference cost
assert table.best() == "fem"
```

### Validate any result (never report a field without this)

```python
import numpy as np
import pinneapple as pp

x = y = np.linspace(0, 1, 41)
X, Y = np.meshgrid(x, y, indexing="ij")
T = np.sin(np.pi * X) * np.sinh(np.pi * Y) / np.sinh(np.pi)          # any field on a grid, from any code

card = pp.trust_card({"x": x, "y": y}, T, pde="laplace",
                     dirichlet={"x_min": 0.0, "x_max": 0.0, "y_min": 0.0,
                                "y_max": lambda P: np.sin(np.pi * P[:, 0])},
                     train_bounds={"x": (0, 1), "y": (0, 1)})
print(card)                                    # one line per check, PASS / FAIL / N/A
assert card.decision == "APPROVED"

bad = pp.trust_card({"x": x, "y": y}, 1.2 * T, pde="laplace", dirichlet={"y_max": lambda P: np.sin(np.pi * P[:, 0])})
assert bad.decision == "REJECT"                # harmonic, but the boundary condition is wrong
```

The checks a result needs, and where they come from:

| Check | How |
|---|---|
| PDE residual | `pp.trust_card(..., pde="laplace" / "poisson" / "heat", source=...)` on a grid; for a model, `pinneapple_analysis.validation.PhysicsValidator` |
| Boundary conditions | `dirichlet={face: value or callable}`; `PhysicsValidator.add_boundary_check` |
| Conservation | automatic in `trust_card` for steady problems (net boundary flux = integral of the source); `PhysicsValidator.add_conservation_check` |
| Reference comparison | `reference=` array or callable; `pp.compare(..., reference=...)` |
| Applicability (OOD) | `train_bounds=` (and `query_points=`); `pinneapple_analysis.trust.TrustGate` for model inputs |

### Equal-budget comparison

```python
import numpy as np
import pinneapple as pp

problem = pp.PhysicalProblem.from_preset("burgers_1d", nu=0.01 / np.pi)
budget = {"epochs": 100, "n_collocation": 256, "hidden_dim": 16, "n_layers": 2}
runs = [pp.solve(problem, "pinn", architecture=arch, seed=s, **budget)
        for arch in ("modified_mlp", "siren") for s in (0, 1)]
cmp = pp.compare(problem, runs, reference="analytic")
print(cmp)                                     # same points, same reference, same budget for every run
```

### Why each element of a design exists

```python
from pinneapple_reason import cases, reason

chain = reason(*cases.roof())                  # flat tiles -> pitch -> ventilated ceiling -> radiant barrier
print(chain.explain("en"))                     # every step: the failure, its mechanism and numbers, the fix, why
assert chain.resolved and chain.path()[0][1:] == ("lap_water_ingress", "increase_slope")
```

### A ready-made industrial case

```python
import pinneapple as pp

p = pp.presets.load("water_hammer_valve_closure")      # pp.presets.names() lists them; p.card() is the data sheet
r = p.run(t_close=10.0)                                # SI inputs; the defaults carry their sources
print(r)                                               # outputs, decision (pressure vs PN16), trust report
assert r.trust.decisao == "APPROVED" and p.validate()["all_pass"]   # Joukowsky, rigid-column theory, convergence
fast = p.run(t_close=2.0)                              # too fast: over PN16 and the line cavitates
assert fast.decision["verdict"] == "NOT OK" and fast.trust.decisao != "APPROVED"
```

### Science by area

```python
import pinneapple as pp

atlas = pp.science.atlas()                             # 20 areas; each subarea available / partial / planned
assert pp.science.validate() == []                     # every pointer names code that exists
print([e["name"] for e in pp.science.experiments(dim="3D+t")])
r = pp.science.run("em.hall_effect")                   # 2-D Hall bar vs the Drude coefficient 1/(n q)
print(r)                                               # computed vs reference with its source, checks, trust
assert r.passed and r.trust.decisao == "APPROVED"
```

### Dashboard of a time-dependent result

```python
import numpy as np
import pinneapple as pp
from pinneapple_tools.visualization.dashboard import Chart, Dashboard, Series, Tile

t = np.linspace(0, 2, 12)
theta = 0.5 * np.cos(3 * t)
frames = [np.full((90, 160, 3), int(100 + 100 * (v + 0.5)), dtype=np.uint8) for v in theta]
db = Dashboard("Pendulum", t, frames, charts=[Chart("Angle", "angle [rad]", "time [s]", [Series("theta", theta)])],
               tiles=[Tile("Max angle", float(np.abs(theta).max()), "rad")])
db.to_png("pendulum.png", k=5)
```

### Validation status of a method

```python
from pinneapple_catalog import methods

status = methods.method_status()
validated = [m.id for m in methods.validated_methods("solver")]
print(len(validated), "validated solvers, e.g.", validated[:5])
assert status["S2"]["status"] == "validated"   # linear finite elements: compared with a reference in a test
```

## Project layout

| Path | Contents |
|---|---|
| `pinneapple/` | the `pp` namespace: re-exports and lazy shortcuts (`pp.solve`, `pp.trust_card`, `pp.dashboard`, ...) |
| `pinneapple_physics/` | problems and presets, the PDE compiler, `solve`/`compare`, metrics, physics audit |
| `pinneapple_neural/` | architectures (model registry), trainers, predictors |
| `pinneapple_simulation/` | numerical solvers, external solvers (OpenFOAM, CalculiX), terramechanics, multibody, vehicles |
| `pinneapple_analysis/` | validation, trust (`TrustReport`, `trust_card`), uncertainty, inverse problems |
| `pinneapple_design/` | geometry, CAD, aero (CST airfoils), design optimisation |
| `pinneapple_tools/` | visualisation (studio, dashboard), export, benchmark suite |
| `pinneapple_catalog/` | method catalog and validation status |
| `pinneapple_lab/` | experiment runner, database, trust card per run |
| `pinneapple_mcp/` | the MCP server for agents |
| `pinneapple_flow/` | engineering flows: schema, node catalog, planner, runtime, store (Flow Studio's library) |
| `pinneapple_reason/` | chain of whys: requirements, failure-mode library by physical domain, reasoning engine, reports |
| `tests/`, `examples/`, `notebooks/colab/`, `docs/` | tests, runnable scripts, Colab notebooks, documentation (mkdocs) |

## Apps from PINNeAPPle-apps and PINNeAPPle-arena

| App | Folder | What it does | Library APIs it uses |
|---|---|---|---|
| Verification Gap | `apps/verification_gap` | A brake rotor that passes at nominal dimensions, Monte Carlo under its machining tolerances: p95, corner stack-up, six checks, TrustReport decision | `pinneapple_analysis.uncertainty.margin.margin_study`, `pinneapple_design.surrogate_inverse.brake_rotor` |
| Design Optimization | `apps/design_optimization` | A NACA section redesigned live for less profile drag at the same lift (MLP surrogate + Adam steps on the design, every iterate re-checked on the physics) | `pinneapple_design.design_optimizer.airfoil_drag.run_design_optimization`, `pinneapple_physics.closed_form.thin_airfoil` |
| Model Benchmarking | `apps/model_benchmarking` | Vanilla PINN, SIREN, PIELM and XTFC trained live on viscous Burgers, ranked by accuracy with Arena's physics-aware residual flag | `pinneapple_arena.showcase.run_benchmark` |
| Parameter Discovery | `apps/parameter_discovery` | A PINN recovers the diffusivity of the 1-D heat equation from noisy readings, with the Cramer-Rao bound of those readings | `pinneapple_analysis.inverse_problems.heat_diffusivity.run_inverse_discovery` |
| Predictive Twin | `apps/predictive_twin` | Baseflow-recession twin of a live USGS river gauge: calibrate, forecast, flag a sustained 3-sigma departure | `pinneapple_systems.digital_twin.recession.run_twin`, `pinneapple_systems.digital_twin.io.usgs.fetch_iv` |
| Surrogate Speed | `apps/surrogate_speed` | A heat-sink surrogate timed against a finite-volume fin-cell solve, with their agreement, over seven designs | `pinneapple_design.thermal.fin_array_speed`, `pinneapple_design.surrogate_inverse.heatsink_fins`, `pinneapple_design.thermal.fin_cell.solve_fin_cell` |
| PINNeAPPle Arena | `apps/pinn_arena` | trains registry architectures head to head on 34 problems (reference data, PDE residual, held-out scores), a leaderboard of this server's runs, and a catalog of 95 third-party physics-ML libraries run through their own packages | `pinneapple_arena.pointwise` (`DatasetGenerator`, `PhysicsLossGenerator`, `instantiate_model`, `train_torch_model`, `field_metrics`, `aggregate_leaderboard`), `pinneapple_arena.external` (`MODELS`, `call_infer`, `run_finetune`, `probe_dataset`) |
| HeatSink Design AI | `apps/heatsink_inverse_ai` | fin count / height / thickness of a straight-fin heat sink by descent through a frozen MLP surrogate, re-checked against the fin formula; finite-volume fin-cell cross-check | `pinneapple_design.surrogate_inverse.heatsink_fins`, `pinneapple_design.thermal.fin_cell`, `pinneapple_physics.closed_form.fin_array_conduction` |
| BrakeCool AI | `apps/brakecool_ai` | vented brake-rotor vanes under a sustained load, same loop; Monte Carlo of machining tolerances | `pinneapple_design.surrogate_inverse.brake_rotor`, `pinneapple_analysis.uncertainty.tolerance.propagate` |
| PatchDose AI | `apps/patchdose_ai` | smallest membrane-controlled release patch for a target rate and wear time (generic coefficients, not a medical tool); release over time | `pinneapple_design.surrogate_inverse.transdermal_patch`, `pinneapple_physics.closed_form.membrane_diffusion` |
| RideTune AI | `apps/ridetune_ai` | lighter cantilever-arm section for a fatigue life (Basquin); local elasticities | `pinneapple_design.surrogate_inverse.suspension_arm`, `pinneapple_analysis.sensitivity.local_derivatives` |
| SoundShape AI | `apps/soundshape_ai` | bass-reflex port for a tuning frequency; port-volume vs accuracy sweep | `pinneapple_design.surrogate_inverse.reflex_port`, `pinneapple_physics.closed_form.helmholtz_resonator` |
| Text2Part | `apps/text2part` | plain-language part to a checked CSG / CadQuery recipe, built and measured (local LLM) | `pinneapple_llm.draft_part`, `measure_part`, `export_recipe` |
| MeshBridge | `apps/meshbridge` | CGNS / Exodus / Fluent / Abaqus / STL / STEP to an NPZ point bundle or a repaired surface | `pinneapple_simulation.external_solvers.cfd_formats.convert` |
| PhysicsCopilot | `apps/physics_copilot` | plain-language problem to a trained PINN and its guardrail report (local LLM) | `pinneapple_analysis.verification.pipeline.analyze` |
| VerifiedPhysics | `apps/verified_physics` | the same pipeline with candidates, UQ, interventions, inverse problems, knowledge graph, geometry and mesh checks | `pinneapple_analysis.verification` (`pipeline`, `uncertainty_inverse`, `causal_discrepancy`, `knowledge_graph`, ...) |
| SatScreen | `apps/satscreen` | closest approach and 2-D Gaussian Pc in Clohessy-Wiltshire motion, J2 propagation | `pinneapple_physics.astro.conjunction`, `presets.astrophysics.cw_exact_velocity`, `pinneapple_llm.draft_preset_kwargs` |
| FlowTrace | `apps/flowtrace` | cross-correlation PIV from a video or image frames | `pinneapple_perception.video_piv`, `pinneapple_perception.piv_tools` |
| NozzleScreen | `apps/nozzle_screen` | ideal rocket nozzle and a PINN cross-check with their disagreement | `pinneapple_physics.closed_form.rocket_nozzle`, `pinneapple_analysis.verification.nozzle_crosscheck` |
| RoverMobility | `apps/rover_mobility` | Bekker-Wong surrogate drawbar pull against slip, constraints R1-R4 checked one by one | `pinneapple_simulation.terramechanics.mobility` |
| GovernedCard | `apps/governed_card` | model cards issued only from a guardrail run against the model | `pinneapple_hub.verified_card.certify_model` |
| HelioTFC | `apps/helio_tfc` | solar-cycle envelope: EMD, Hilbert amplitude, end-point-constrained ELM fit (live SILSO data) | `pinneapple_systems.time_series.solar_cycle` |
| OrbitWatch | `apps/orbitwatch` | band reflectance / NDVI change between dated images, HTML report (Sentinel-2 via Earth Engine) | `pinneapple_perception.change_detection`, `pinneapple_perception.sentinel2` |

## Tests and lint

```bash
pytest tests/test_public_api.py tests/test_manufactured_solutions.py -q    # the must-pass core
pytest tests/test_agents_md.py tests/test_mcp_server.py -q                  # this file and the MCP server
python scripts/run_colab_notebooks.py --smoke                               # every Colab notebook, tiny sizes
pinneapple check <result dirs or models>                                    # physics linter (pinneapple_check)
python scripts/check_app_helpers.py                                         # apps/examples only compose library calls
python scripts/check_experiment_manifests.py                                # every examples/* folder has a valid experiment.json
ruff check <changed files>
```

CI (`.github/workflows/tests.yml`): public API, distribution contents, manufactured-solution physics tests, the first
example from a clean wheel, and the Colab notebooks in smoke mode. A user-visible change goes in a new file
`changelog.d/<slug>.<category>.md` (see `changelog.d/README.md`), never in `CHANGELOG.md` itself, so parallel pull
requests do not conflict.

## Generated files

These are built where they are used and git-ignored; never commit them (a committed copy makes every other open pull
request conflict on the next merge):

| File | Built by | Where |
|---|---|---|
| `data/web/hub/index.json` | `python -m pinneapple_hub.library build` | on first `hub.load_index()` (a cache), at `docker build` of `apps/hub` |
| `docs/flow_studio/COVERAGE.md` | `python -m pinneapple_flow coverage -o docs/flow_studio/COVERAGE.md` | the docs build (`tools/gen_flow_docs.py`) |
| `docs/flow_studio/EXAMPLES_AS_FLOWS.md` | `python -m pinneapple_flow example-flows doc` | the docs build (`tools/gen_flow_docs.py`) |

Run the command to see one locally (and `build` again after changing a Hub source: the cached index is not refreshed
on its own). What is committed are the measurements they are built from: the Hub's recorded snippet runs, one file per
item in `data/web/hub/snippet_runs/` (`python -m pinneapple_hub.library run --stale` records new or changed snippets),
and the flow runs in `pinneapple_flow/flows/examples/measured.json`. The `generated` job of
`.github/workflows/tests.yml` builds all three from a fresh clone and fails when a snippet has no passing run of its
current code, a module has neither a flow node nor a reason, an example folder has neither a flow nor a reason, or the
docs site does not build.

## Never do

- Do not put reusable logic (physics, numerics, geometry, data processing, metrics, plotting helpers, trust checks,
  IO, solvers, training loops) in `apps/`, `examples/`, `scripts/` or notebooks: it goes in a `pinneapple*` package
  with a public API and tests, and the app or example imports it. Apps keep only glue (routes, schemas, UI, deploy).
  `python scripts/check_app_helpers.py` flags new local helper modules and copies of library helpers; the inventory is
  [docs/dev/LIBRARY_CONSOLIDATION.md](docs/dev/LIBRARY_CONSOLIDATION.md).
- Do not write a raw PyTorch training loop when `pp.solve`, a trainer or a registered model already does the job.
- Do not report a result (a number, a field, a plot, "it works") without its physics checks: a trust card or the
  equivalent residual, boundary-condition, conservation and reference checks, and say which ones failed.
- Do not call a method validated unless `pinneapple_catalog` says so; "tested" is not "validated".
- Do not copy GPL (or other copyleft) code into this Apache-2.0 repository; reimplement from the paper and cite it.
- Do not invent reference values or citations; every reference number needs its source.
- Do not widen tolerances or skip a failing check to get a pass; fix the cause or report the failure.
- Do not commit a generated file (see "Generated files"); commit what it is built from and run its command locally.
- Do not add an `examples/*` folder without a valid `experiment.json`: wrap the script in
  `pinneapple_lab.experiment(...)` (or start from `python -m pinneapple_lab scaffold <name>`, or run
  `python -m pinneapple_lab register <folder>`). CI fails otherwise (`scripts/check_experiment_manifests.py`), and a
  metric in a manifest needs its measured value, unit and how it was measured.
