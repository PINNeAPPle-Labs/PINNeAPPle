# Library consolidation: reusable logic outside the library

Rule: **reusable logic goes in the library; apps, examples, scripts, notebooks and the satellite repos only compose
library calls.** A user gets every feature with `pip install pinneapple`, through a `pp.` shortcut or a public module.
App-specific glue (FastAPI routes, request schemas, HTML/JS, deploy files, auth) stays in the app.

This page is the inventory behind that rule (audit of 2026-10-13, `main` at `8e8ff8c`). For every piece of reusable
logic found outside the `pinneapple*` packages it records where it is, what it does, whether the library already has
it, and the verdict:

- **move**: no library equivalent; it goes into the library with a public API, docstrings and tests, and the caller
  imports it from there (a thin shim stays where outside users may import the old path);
- **switch**: the library already has it; the caller is switched to the library;
- **glue**: app- or example-specific (routes, UI, deploy, narrative demo code); stays;
- **stays**: kept on purpose for the reason given (frozen record, release tooling, external reference replica, ...).

The pull requests that implemented it and what is still to do are at the end. `scripts/check_app_helpers.py` (run in CI) keeps new
duplicates out; see [AGENTS.md](../../AGENTS.md) and [CONTRIBUTING.md](../../CONTRIBUTING.md).

Work streams that were in flight during the audit are left alone and audited after they merge: observatory AI
features, playground, zoo v2, black-hole zoo (`examples/black_hole_weather`, `pinneapple_physics/blackhole`),
multiscale ensemble (`pinneapple_physics/ensemble.py`), rover dashboard v3 (`dashboard/static.py`, studio,
terramechanics), equal-budget benchmark / benchmark API (`pinneapple_analysis/benchmark`, `aero/panel.py`,
`inverse_problems/heat_transfer.py`, Benchmark `exp/09`).

## Summary

| Area | Files audited | move | switch | glue / stays |
|---|---|---|---|---|
| `apps/*` (15 apps) | 51 .py | 22 | 6 | routes, schemas, UI, asset builders |
| `examples/` | 282 .py | 10 | 14 | ~240 scripts that already only call the library |
| `scripts/`, `benchmarks/`, `projects/`, `templates/`, `notebooks/` | 60 .py, 7 notebooks | 4 | 9 | CI / release / docs tooling, tutorials |
| `lab/` | 293 .py | 0 | 0 | all are frozen run snapshots (`pinneapple_lab.runner._snapshot_code`) |
| PINNeAPPle-Benchmark (`experiments/`) | 43 .py | 6 | 7 | the experiments' own models and physics |
| PINNeAPPle-Climate | 22 .py | 4 | 3 | domain experts (ETAS, hurricane, tsunami baselines) |
| PINNeAPPle-CFD (private) | generic modules only | 10 (reimplemented from first principles) | 8 | everything tied to a client case stays private |
| pinnfactory | 6 .py | 0 | 0 | standalone PyPI package, ancestor of `pinn_factory`; no dependency on pinneapple |
| PINNeAPPle-Automotive | 0 .py | 0 | 0 | documentation only |

## apps/

Most apps are already thin: their physics lives in `pinneapple_design.aero`, `pinneapple_design.thermal.heatsink`,
`pinneapple_physics.{pcb,astro}`, `pinneapple_data.{cae,data_health,unified,formfill,lineage,preflight,simulation_metadata}`
and `pinneapple_analysis.inverse_problems.heat_transfer`.

| Where | What it does | In the library | Verdict | Target |
|---|---|---|---|---|
| `apps/_shared/appkit.py` `install`, `BusyLimiter` | HTTP Basic auth, static mounts, 429 limiter | — | glue | — |
| `aero_optimizer/api.py` `_clean` | JSON sanitiser (NaN to null, numpy to Python, rounding) | 8 private copies (`data_health._clean`, `process_optimizer._clean_json`, `pinneapple_mcp.tools._clean`, `pinneapple_mcp.jobs._jsonable`, `pinneapple_check.result._clean`, `fitmesh.api._jsonable`, `pinneapple_lab.context._jsonable`, ...) | move | `pinneapple_core.jsonable.to_jsonable` |
| `aero_optimizer/api.py` `_picks`, `_picks_al` | fastest / greenest / safest / balanced design on a Pareto front | `design_optimizer.pareto` has the front, not the picks | move | `pinneapple_design.design_optimizer.pareto.representatives` |
| `aero_optimizer/api.py` constraint "reasons" tally (twice) | counts why designs were infeasible | — | move | `pinneapple_design.aero.optimize.infeasibility_reasons` |
| `aero_optimizer/api.py` `openfoam_case` zip, `COEF_SCRIPT` | case + run script + coefficient script as one zip | `aero.case.write_case` writes the case only | move | `pinneapple_design.aero.case.case_zip` |
| `aero_optimizer/tools/train_surrogate.py` | trains the MLP ensemble and the MeshGraphNet on the OpenFOAM runs, holds out whole shapes | models in `aero.surrogate`, no training | move | `pinneapple_design.aero.train` |
| `aero_optimizer/tools/verify.py` | spreads verification picks on the front, averages unsteady runs, evaluates a CFD polar | parts in `aero.aircraft` | move | `pinneapple_design.aero.verify` |
| `aero_optimizer/tools/generate_dataset.py` `plan`, `one` | LHS shapes x angle bins + NACA reference polars, one OpenFOAM run to npz | `aero.case.run_case` | move | `pinneapple_design.aero.dataset` |
| `aero_optimizer/tools/render_airliner.py` `colorbar` | burns a colour bar into an image | `pinneapple_tools.visualization.studio.colormap.colorbar` (same code) | switch | — |
| `aero_optimizer/tools/{build_assets,render_gallery,verify3d,verify3d_airliner}.py` | build the app's assets | — | stays (asset build CLIs) | — |
| `heatsink_sizer/{engine,field,sizer,trust,surrogate}.py` | aliases of `pinneapple_design.thermal.heatsink.*` | yes | switch (aliases kept as shims) | — |
| `pcb_hotspot/{engine,solver}.py` | aliases of `pinneapple_physics.pcb.*` | yes | switch (aliases kept as shims) | — |
| `inverse_heat/engine.py` | alias of `pinneapple_analysis.inverse_problems.heat_transfer` | yes | switch after the benchmark-api stream merges (it edits that module) | — |
| `inverse_heat/scripts/*.py` `plate_fv`, `block_fv`, `fin_theta` | the copy-paste scripts the app shows | `numerical_solvers.heat_fv_reference`, `heat_transfer.fin_exact` | stays: complete stand-alone scripts by design, run by `tests/test_inverse_heat.py` | — |
| `inverse_heat/api.py` `_start` job thread | progress-reporting job runner | `pinneapple_mcp.jobs` (different shape) | glue | — |
| `data_health/api.py` | calls private `process_optimizer._clean_json` | — | switch to `to_jsonable` | — |
| `data_standardizer/api.py` `_charts`, `_down` | raw vs standardised series per variable, downsampled | private downsamplers in the library | move | `pinneapple_data.unified.before_after_series` |
| `design_requirements/api.py` `_compile`, `_spec_meta`, `_parse_overrides`, `_filename` | read documents, extract rules (LLM optional), compile a form; describe a spec; validate overrides | parts in `pinneapple_data.formfill` | move | `formfill.compile_documents`, `FormSpec.describe`, `formfill.check_overrides` |
| `interop_hub/api.py` `_preview` | boundary surface + up to 6 scalar fields of a converted dataset | `studio.scene.scalar_of` | move | `pinneapple_data.cae.dataset.preview` |
| `mesh_quality/api.py` `_worst_cells` | the n worst cells per metric with their centres | — | move | `pinneapple_data.cae.report.worst_cells` |
| `model_lineage/api.py` `_graph_payload`, `_is_lineage_json` | per-artifact flags, hash verification, overall status | — | move | `Lineage.summary`, `Lineage.looks_like` |
| `simulation_comparator/api.py` tolerance roll-up | per-field PASS / FAIL against a tolerance | — | move | `cae.compare.compare(..., tolerance=)` |
| `simulation_preflight/api.py` `_snippets`, `_is_result` | source lines around each finding; detect result files | — | move | `pinneapple_data.preflight.attach_snippets`, `drop_results` |
| `observatory/api.py` `target` window, `_target_vec`, `_body_now`, `_events`, `engine_integrators`, `_satellites` | observing window, target resolution, body summary, merged event list, integrator comparison, TLE merging | parts in `pinneapple_physics.astro` | move, **after** the observatory AI stream merges | `astro.phenomena.observing_window`, `astrometry.resolve_target`, `phenomena.body_summary`, `phenomena.events`, `nbody.compare_integrators`, `satellites.load_tles` |
| `observatory/tools/make_textures.py` | procedural planet textures | — | stays (asset build) | — |
| `simulation_metadata/api.py` | calls the library | — | glue | — |

## examples/

282 files; about 240 are plain scripts over library APIs. `examples/scratch/` is scratch code, not shipped as
examples. Cross-cutting: ~60 files insert the repo root in `sys.path` and 71 lines choose the device inline
(`pp.best_device()` exists); these are cleaned up as the files are touched.

| Where | What it does | In the library | Verdict | Target |
|---|---|---|---|---|
| `meshgraphnet/_common.py` (5 scripts + a test) | DeepMind meshgraphnets TFRecord download and reader, conversion to `MeshGraph`, one-step training, rollout RMSE | none (`pick_device` = `pp.best_device`) | move | `pinneapple_data.datasets.deepmind_mgn`, `pinneapple_neural.architectures.graphnn.mgn_dynamics.train_dynamics` / `eval_rollout` |
| `_pred`, `unwrap_pred`, `FactoryTensorAdapter`, `TensorOut` (~12 files) | model output (`PINNOutput` / `ModelOutput`) to a tensor | 8 private copies | move | `pinneapple_neural.utils.as_tensor` |
| `use_cases/heated_channel_twin/...` `shape_step`, `prbs`, `chirp`, `sawtooth`, `smoothed_random_walk`, ... | actuator excitation signals for system identification | none | move | `pinneapple_systems.excitation` |
| `use_cases/solid_mechanics/*`, `drill_pipe_rotating_pipeline.py`, `run_simulations_rotating.py` | axisymmetric strains / stresses / equilibrium residual / von Mises (with torsion) by autograd, 3 copies | partial (preset residual, Voigt von Mises) | move | `pinneapple_physics.solid.axisymmetric` |
| `physics_db/01_*` `fetch_power_daily`, `power_json_to_samples`; `pinn_solver/06_*` `fetch_open_meteo` | NASA POWER and Open-Meteo clients | other clients in `datasets.external_sources` | move | `NASAPowerClient`, `OpenMeteoClient` |
| `vs_physicsnemo/04_*` `generate_burgers_data` | random periodic IC to u(T) operator pairs | none | move | `pinneapple_data.datasets.physics.burgers_periodic_pairs` |
| `use_cases/missile_aero/*` `pitot_pressure_ratio`, `cp_max_newtonian` | Rayleigh Pitot formula, modified Newtonian Cp | none | move (the example's `cf_van_driest` is a crude temperature-ratio correction, not Van Driest II: not moved) | `pinneapple_physics.closed_form.compressible` |
| `use_cases/crash_surrogate/*` `abramowicz_jones_solver` | axial crush of a square tube | none | stays for now (one caller, needs a published reference test) | — |
| `physics_ensemble/compare_lookahead_gif.py` `stitch` | GIFs side by side | — | move, after the rover-dashboard stream merges (`dashboard/static.py`) | `dashboard.static.hstack_gifs` |
| `black_hole_weather/simulate.py` `load_run`, checkpoint loop | run storage for the accretion flow | — | move, after the black-hole zoo stream merges | `pinneapple_physics.blackhole` |
| `handoff_cases/airfoil_ld/simulation/airfoil_case.py` `run`, `residuals` | simpleFoam run, residual parsing | `aero.case.run_case`, `simulation_metadata.openfoam.parse_log` | switch | — |
| `handoff_cases/airfoil_ld/simulation/summarise_baseline.py` `gci` | grid convergence index | `pinneapple_lab.uq.gci` | switch | — |
| `handoff_cases/free_surface_trough/*` field / patch / log readers | ASCII OpenFOAM readers | `pinneapple_data.cae.foam`, `pinneapple_twin3d.openfoam` | switch | — |
| `pinneapple_benchmark/01_*` `_build_reference`; `architectures/04_pinns.py` `burgers_fdm_forced`, local `PINNFactory` | Burgers references; a copy of the factory | `datasets.physics._load_burgers_1d`, `closed_form.burgers`, `pinn_solver.factory` | switch | — |
| `_grad`, `laplacian`, `du_dx` (7 files) | autograd derivatives | `pinn_solver.compiler.autograd_ops` | switch | — |
| `l2_rel`, `rel_l2`, `relative_l2`, `compute_metrics` (8 files) | error metrics | `pinneapple_physics.metrics` | switch | — |
| `_dark_fig`, `dark_ax`, `_setup_dark`, `_style` (~10 files) | dark matplotlib theme | `pinneapple_tools.visualization.style.use_cfd_style` | switch | — |
| `time_series/07`, `08` duplicated helpers | seed, LSTM, rolling evaluation | `set_seed`, `LSTMForecaster`, `BacktestRunner` | switch | — |
| `_lhs_sample`, `_lhs`, `lhs_sample` | Latin hypercube | `pinneapple_data.parameter_sampling` | switch | — |
| `_jsonable`, `jsonable` | JSON sanitiser | `to_jsonable` (above) | switch | — |
| `vs_physicsnemo/05,06` `validate_physics`, `validate_cfd_fields` | hand-made divergence / BC checks | `pp.trust_card`, `validation.conservation` | switch | — |
| `use_cases/fin_convection_inverse/fv_reference.py` | re-export shim of `heat_fv_reference` | yes | switch | — |
| `use_cases/terramechanics/benchmark_vs_omnilrs.py` `OmniLRSSolver` | replica of an external tool used as a baseline | — | stays | — |
| `benchmark_suite/14_*` 2-D Q4 FEM, cavity and cylinder FDM | demo solvers | none | stays until validated (Ghia et al. cavity, analytic plate) | — |
| `physics_check/generate.py`, `physics_ensemble/*` experts, `worldmodel`, `arena_pipelines`, `geometry` demos | fixtures and narrative demos | — | glue | — |
| `handoff_cases/airfoil_ld/simulation/xfoil_reference.py` | XFOIL subprocess runner | — | stays (external reference tool) | — |

## scripts/, benchmarks/, projects/, templates/, notebooks/, lab/

| Where | What it does | In the library | Verdict | Target |
|---|---|---|---|---|
| `benchmarks/lpm_ellipse_design_space.py` `ellipse`, `square` (imported by `tests/test_lpm.py`) | LPM design from an analytic shape | none | move | `pinneapple_neural.lpm.Design.ellipse`, `Design.square` |
| `benchmarks/lbm_vortex_shedding.py` inline Strouhal estimator (also in `pinneapple_lab/experiments/{cylinder,car,rom}.py`) | dominant frequency of a probe signal (Hann, rfft, parabolic peak) | `regime.strouhal` takes the frequency | move | `pinneapple_physics.physics_audit.regime.dominant_frequency` |
| `benchmarks/ssqn_paper_benchmarks.py` `Burgers.exact` | inviscid Burgers, sine IC, by characteristics | viscous Cole-Hopf only | move | `pinneapple_physics.closed_form.burgers.burgers_sine_inviscid_exact` |
| `benchmarks/ssqn_paper_benchmarks.py` `sod_exact` | exact Sod solution | `compressible_fv.sod_exact` (and a second library copy) | switch | — |
| `benchmarks/ssqn_*` `rel_l2`, `grad` | metric, autograd | `pinneapple_physics.metrics`, `autograd_ops` | switch | — |
| `scripts/arena/leaderboard.py` | prints the leaderboard | `runner.leaderboard.load_leaderboard` | switch | — |
| `scripts/*` CI, release, docs, zoo, geometry, astro catalog builders; `cosmos`, `omniverse`, `mvp` glue | tooling | — | stays | — |
| `projects/scripts/pipeline/*` | one example project | — | glue (`rmse` switched to `pinneapple_physics.metrics`) | — |
| `templates/01-35` | tutorials; 18 hand-write autograd residuals on purpose | `autograd_ops` | stays (teaching) | — |
| `notebooks/PINNeAPPle_Demo_Heat3D_Laplace.ipynb` `laplace_residual`, `show_slices` | 3-D Laplacian, mid-plane slices | `autograd_ops.laplacian`, `visualization.voxel.plot_voxel_slice` | switch | — |
| `notebooks/PINNeAPPle_Climate_Digital_Twin_OpenMeteo.ipynb` `VirtualWeatherSensor` | dataframe replay as a sensor stream | `digital_twin.io.sensors.SensorRegistry` | switch | — |
| `notebooks/colab/*` `load_trust_card`, `repo_file` | fallback for pinneapple 0.7.0 without `pp.trust_card` | `pp.trust_card` | stays until the next release | — |
| `lab/runs/**/code/*.py` (293 files) | code snapshots taken by `pinneapple_lab.runner._snapshot_code` | live code in `pinneapple_lab/experiments`, examples, satellite repos | stays (frozen records: editing them breaks reproducibility) | — |

## Satellite repos

Both PINNeAPPle-Benchmark and PINNeAPPle-Climate find the library through `sys.path.insert(<repo>/../PINNeAPPle)`
and declare no dependency; the consolidation PRs there add a `requirements.txt`.

| Where | What it does | In the library | Verdict | Target |
|---|---|---|---|---|
| Benchmark `experiments/02,03,04,05,07` `veriphysics*` blocks, Climate `01_sst` `stage_eval` | register benchmark, `PhysicsCase.validate_against_benchmark`, calibration summary, physics confidence score (6 copies) | the parts only | move | `pinneapple_analysis.verification.physics_confidence_score.score_against_reference`, `CalibrationSummary.from_predictions` |
| Climate `hazards_forecasting/src/common.py` `block_bootstrap`, `skill`, `cluster_skill`; Benchmark `02` `block_bootstrap_rmse`; `06` `jackknife` | block / group bootstrap CIs, skill score with paired CI, blocked jackknife (9 experiments) | i.i.d. bootstrap only | move | `pinneapple_analysis.uncertainty.resampling` |
| Climate `common.py` `run_delayed` | prequential run with delayed labels (pokes the private `ens._last`) | `PhysicsEnsemble.run` without delay | move, after the multiscale-ensemble stream merges (`ensemble.py`) | `PhysicsEnsemble.run_delayed` |
| `paperkit/paperkit.py` (Benchmark, byte-identical copy in Climate; 10 paper scripts) | PDF paper builder and figure style | — | move | `pinneapple_tools.reporting.paperkit` |
| Benchmark `05` `fit_dmdc`, `dmdc_roll` | DMD with control, rollout | DMD without control | move | `pinneapple_neural.architectures.rom.dmd.DMDc` |
| Benchmark `05` `rollout_error` | mean relative rollout error | — | move | `pinneapple_physics.metrics.rollout_relative_l2` |
| Benchmark `03` `energy_error` | OpenRadioss energy error from `engine.log` | — | move | `openradioss.read_energy_error` |
| Climate download helpers `_fetch`, `get` | URL fetch with retry and a file cache (3 scripts) | private `_fetch` without retry | move | `pinneapple_data.datasets.download.fetch_cached` |
| Benchmark `04` `metrics`, `auroc`, `lhs`, `calibration`; `03` `pod_gp`, `nearest`; `05` `POD`; Climate `radial_spectrum`, ACC | metrics, AUROC, LHS, POD + GP, spectra | `pinneapple_physics.metrics`, `pinneapple_security.cps.roc_auc`, `rom.parametric`, `rom.pod.POD`, `posterior_metrics.radial_power_spectrum`, `weather.evaluate` | switch | — |
| Benchmark `02` soil model, `04` Bekker-Wong constraint net, `06` lattice / 3-SAT maths, `07` PDR nets, `08` shock-train detection; Climate ETAS, hurricane, tsunami experts | the experiments' own contribution | — | stays | — |
| Benchmark `exp/09-equal-budget` branch | equal-budget work stream | — | in flight, not touched | — |
| pinnfactory `pinn_generator.py` | stand-alone SymPy-to-loss PINN generator (PyPI `pinnfactory`) | `pinn_solver.factory.pinn_factory` is its evolved superset | stays (independent package) | — |
| PINNeAPPle-Automotive | documentation only | — | nothing to move | — |

### PINNeAPPle-CFD (private)

Only generic algorithms were read. Nothing from that repository is copied: each item below is **re-implemented
from textbooks or published papers** (cited in the docstrings) and tested against independent references. Client
cases, geometry, campaigns, reports and deployment code stay private.

| Generic capability | In the library | Verdict | Target |
|---|---|---|---|
| solver-log parsing, checkMesh metrics, polyMesh reader, dictionary parser, conformal intervals, adaptive collocation, loss balancing, von Mises | yes (`simulation_metadata.openfoam`, `cae.foam`, `cae.quality`, `uncertainty.conformal`, `trainer.loss_balancer`, `solid_fem.von_mises`) | already in library | — |
| pipe-flow friction factors and pressure drops (Colebrook, Haaland, Blasius, Hagen-Poiseuille, parallel plates) with validity ranges | Colebrook only | reimplement | `pinneapple_physics.closed_form.pipe_flow` |
| first-cell height for a target y+ in a pipe | flat plate only | reimplement | `closed_form.pipe_flow.first_cell_height` |
| prism-layer sizing (count, expansion, total thickness) | private helper | reimplement | `pinneapple_design.fitmesh.layers.size_layers` |
| per-cell Courant number and wall distance on an unstructured mesh | none | reimplement | `pinneapple_data.cae.metrics` |
| transient 1-D slab conduction, Fourier series | none | reimplement | `pinneapple_physics.closed_form.heat.slab_transient` |
| function-object `.dat` table reader | partial | reimplement | `pinneapple_data.simulation_metadata.openfoam.read_dat` |
| streamlines on an unstructured finite-volume mesh | structured grids only | later (larger, needs its own validation) | `pinneapple_data.cae.streamlines` |
| strict OpenFOAM dictionary value editor | looser `campaign.set_entry` | later | `external_solvers.openfoam.dict_edit` |
| solid-particle erosion models (published forms, no default coefficients) | none | later | `closed_form.erosion` |
| surrogate drift monitor, Monte-Carlo tolerance propagation, sparse-sensor GP and sensor placement, MCMC inverse | none | later | `pinneapple_analysis.uncertainty` |

## Library-side duplicates found on the way

- Three `latin_hypercube` functions (`aero.geometry`, `rom.parametric`, `modelica.sweep`).
- Two exact Sod solutions (`compressible_fv.sod_exact`, `presets.astrophysics.sod_exact_solution`).
- AUROC only in `pinneapple_security.cps.roc_auc`.
- `aero/render_blender.py:jet` duplicates `studio/colormap.py:jet`.
- About eight private JSON sanitisers and eight private model-output unwrappers (replaced by `to_jsonable` and
  `as_tensor`).

## Work streams merged during the consolidation (audited against main 4966ee6)

Playground (#649), multiscale ensemble (#661), zoo v2 (#654), rover dashboard v3 (#650) and rover zoo v1.1 (#651) put their
reusable logic in the library (`pinneapple_hub/zoo/{checks,export,web}`, `pinneapple_design/aero/geometry_onnx.py`,
`pinneapple_design/geometry_library/web.py`, `pinneapple_physics/multiscale`, `pinneapple_design/thermal/heatsink/monotone.py`,
`pinneapple_neural/trainer/monotonic.py`, `pinneapple_simulation/terramechanics`). What they added outside it is glue:

| Where | What it does | Verdict |
|---|---|---|
| `apps/playground/build.py`, `site/js/*`, `tools/`, deploy files | static-site build, browser UI over the zoo web bundle (checks come from the ONNX graph) | glue |
| `examples/physics_ensemble/multiscale_turbulence.py` | wires `pinneapple_physics.multiscale` and the architectures for one benchmark | glue |
| `examples/studio/07_lunar_rover_dashboard.py` | loads the zoo contact model, renders the dashboard | glue |
| `scripts/zoo/build_assets.py` `hub_only_entry`, `_carry_versions` | zoo release manifest | stays for now; candidate for `pinneapple_hub.zoo._artifacts` once the open zoo PRs (#656, #674) land |
| `scripts/zoo/publish_to_hub.py` | command line of the zoo's Hub publication | glue: `plan`, `publish`, the Parquet copies, cards, Space and every Hub check are in `pinneapple_hub.zoo.publish`, `.parquet`, `.hub_cards` |
| `apps/zoo_space/` (`index.html`, `app.js`, `space.css`) | page of the static zoo demo Space | UI over the data of `pinneapple_hub.zoo.demo.export_space` and the runtime `pinneapple_hub/zoo/web`; charts and widgets are the Playground's |

## Pull requests

All merged, each with its tests (against closed forms, published values or independent algorithms) and green CI.

| Area | Pull request |
|---|---|
| Inventory, rule in AGENTS.md / CONTRIBUTING.md, CI guard `scripts/check_app_helpers.py` | [#652](https://github.com/PINNeAPPle-Labs/PINNeAPPle/pull/652) |
| Data and CAE apps: `to_jsonable`, CAE preview / worst cells / tolerance verdicts, lineage report, pre-flight snippets, form compilation, unified before/after series | [#653](https://github.com/PINNeAPPle-Labs/PINNeAPPle/pull/653) |
| Satellite helpers: resampling CIs, `score_against_reference`, `DMDc`, rollout L2, `fetch_cached`, OpenRadioss energy error | [#655](https://github.com/PINNeAPPle-Labs/PINNeAPPle/pull/655) |
| Closed forms (pipe flow, compressible, heat, inviscid Burgers), `dominant_frequency`, layer sizing, FV Courant / wall distance, `.dat` reader, LPM designs, one Sod and one Colebrook | [#657](https://github.com/PINNeAPPle-Labs/PINNeAPPle/pull/657) |
| `pinneapple_tools.reporting.paperkit` | [#658](https://github.com/PINNeAPPle-Labs/PINNeAPPle/pull/658) |
| Example helpers: `as_tensor`, DeepMind MGN reader and training, excitation signals, NASA POWER / Open-Meteo, axisymmetric elasticity, periodic Burgers pairs | [#659](https://github.com/PINNeAPPle-Labs/PINNeAPPle/pull/659) |
| Aero optimizer: training, dataset, verification, Pareto representatives, case zip | [#660](https://github.com/PINNeAPPle-Labs/PINNeAPPle/pull/660) |
| Examples, benchmarks, scripts and a notebook switched to existing library helpers | [#678](https://github.com/PINNeAPPle-Labs/PINNeAPPle/pull/678) |
| Drift monitoring, tolerance propagation, GP sensor placement | [#679](https://github.com/PINNeAPPle-Labs/PINNeAPPle/pull/679) |
| `PhysicsEnsemble.run_delayed`, `hstack_gifs` | [#680](https://github.com/PINNeAPPle-Labs/PINNeAPPle/pull/680) |
| `pp.` shortcuts | [#681](https://github.com/PINNeAPPle-Labs/PINNeAPPle/pull/681) |
| PINNeAPPle-Benchmark: experiments use the library helpers; paperkit from the library | [Benchmark#7](https://github.com/PINNeAPPle-Labs/PINNeAPPle-Benchmark/pull/7), [Benchmark#8](https://github.com/PINNeAPPle-Labs/PINNeAPPle-Benchmark/pull/8) |
| PINNeAPPle-Climate: library helpers, paperkit, `run_delayed` | [Climate#16](https://github.com/PINNeAPPle-Labs/PINNeAPPle-Climate/pull/16), [Climate#17](https://github.com/PINNeAPPle-Labs/PINNeAPPle-Climate/pull/17), [Climate#18](https://github.com/PINNeAPPle-Labs/PINNeAPPle-Climate/pull/18) |

## Still to do

| Item | Why it waits |
|---|---|
| Observatory analyses (`observing_window`, `resolve_target`, `body_summary`, `events`, `compare_integrators`, `load_tles`) | the observatory AI work stream is still open |
| Black-hole run storage (`load_run`, checkpoint loop) | the black-hole zoo work stream is still open |
| `apps/inverse_heat/engine.py` alias | the benchmark-API stream edits `inverse_problems/heat_transfer.py` |
| Zoo manifest and Hub verification in `scripts/zoo/` | open zoo PRs #656, #674 |
| Physics-ensemble examples: `ic_coefficients`, `ParametricPINN`, the Adam + cosine training loop, ring-graph adapters, selection timeline (duplicated between `five_model_families_gif.py`, `fno_pinn_gnn_solvers.py`, `compare_lookahead_gif.py`, `sensor_noise_gif.py`) | next pass |
| Unstructured-mesh streamlines, strict OpenFOAM dictionary editor, particle-erosion models, MCMC inverse with identifiability, rollout evaluator against persistence | larger items that need their own validation; to be written from the literature |
| Abramowicz–Jones crush, the 2-D demo solvers of `benchmark_suite/14` | need a published reference test before they move |
| `airfoil_case.residuals`, trough boundary reader, time-series LSTM / rolling evaluation | the library versions behave differently; the examples keep theirs |
| `pinneapple_simulation/.../pipe_network_1d.colebrook_white_f`, three library `latin_hypercube` functions, private model-output unwrappers | library-side duplicates with different edge behaviour; to be unified with care |
