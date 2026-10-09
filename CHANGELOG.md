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

[Unreleased]: https://github.com/PINNeAPPle-Labs/PINNeAPPle/compare/v0.6.3...HEAD
[0.6.3]: https://github.com/PINNeAPPle-Labs/PINNeAPPle/compare/v0.6.2...v0.6.3
[0.6.2]: https://github.com/PINNeAPPle-Labs/PINNeAPPle/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/PINNeAPPle-Labs/PINNeAPPle/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/PINNeAPPle-Labs/PINNeAPPle/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/PINNeAPPle-Labs/PINNeAPPle/releases/tag/v0.5.0
