- `pinneapple_pipeline` / `pp.SurrogatePipeline`: one object for the whole cycle of a surrogate model — problem →
  data → preprocess → model → training → export → verification — as a facade over what the library already does
  (presets and condition sampling, reference and analytic solves, `design` of experiments, the supervised surrogate
  study, `train_pinn`/`validate_pinn`, model export), plus `pp.pipeline` and `python -m pinneapple.pipeline
  validate|plan|run|resume|report|presets`. A configuration is twenty-one dataclasses that round-trip through YAML
  and reject an unknown key instead of ignoring it (`pp.PipelineConfig`; `pp.SurrogateConfig` is
  `pinneapple_design`'s and stays untouched).
- Ten VeriPhysics gates decide what a run may be used for, each returning `PASS`/`WARN`/`FAIL`/`NOT_EVALUATED`/
  `NOT_APPLICABLE` with the value, the threshold, the evidence, the limitations and the next step: problem,
  geometry, discretization (a real grid-refinement study of the reference solver), simulation, dataset (coverage,
  duplicates, split leakage), model (held-out error against the criterion of the use case), physics (PDE residual
  and boundary/initial misfit on fresh points), applicability (validity envelope and OOD), export (the exported file
  compared against the in-memory model) and inference. An unevaluated gate is never reported as a pass, and
  `fail_policy: block` raises `PipelineRejected` with the report attached. `state.json` makes `run_until`,
  `resume` and `status` survive a new process: a finished reference campaign is not solved twice.
- Measured on the four use cases in `examples/surrogate_pipeline/`: the parametric surrogate of the Burgers
  viscosity (18 cases, GP vs MLP on held-out cases, 13 s, `nrmse` 1.8e-4) reports 7 pass / 1 warn / 2 not
  applicable, the warn being the distribution-shift corner held out on purpose; the physics-only PINN scores
  `relative_l2` 0.068 against the Cole-Hopf solution with a boundary/initial misfit of 6.9e-3 and an ONNX file that
  agrees with the in-memory model to 1.6e-7; the symbolic heat equation is refused at 300 epochs (the conditions
  traded against the residual, named by `validate_pinn`) and passes at 2000 with the L-BFGS polish.
- Two things the pipeline had to say out loud, both found while building it: the `solver_config` override of
  `pinneapple_physics.solving` only reaches the solver inside `params` (a top-level `{"nx": ...}` is merged into the
  spec and ignored, which makes a refinement study compare a run with itself), and the reference solutions of
  `laplace_2d`, `poisson_2d` and `climate_ocean_gyre` are identically zero at the sampled points, so the `model`
  and `discretization` gates report `NOT_EVALUATED` there instead of a perfect score.
- Five more blocks, each backed by an instrument that already existed and each tested end to end: `geometry`
  (an STL read by `pinneapple_data.cae.io.read_stl` and checked by `surface_checks` — watertightness, open and
  non-manifold edges, winding, enclosed volume and the bounding box in metres, so the `geometry` gate can fail a
  surface that does not bound a volume; `source: builder` builds it with `pinneapple_design.geometry.bodies` and
  writes it to `<run>/geometry/surface.stl`, so the gate measures a file and the `geometry` figure draws it);
  `materials` (a key of `pinneapple_design.intent.materials.MATERIALS` applied
  to the problem's own parameters, each value carrying the datasheet it was read from, Lamé constants recomputed, and
  only exact property names matched — `nu` is the Poisson ratio of a solid and the viscosity of a flow);
  `optimization` (grid or random search over the PINN's training knobs, chosen on validation points that are not the
  ones the `model` gate reports); `adaptive_learning` (`doe.design(method="infill")` over the cases already solved,
  solve, grow the dataset, repeat, until the target or the budget); `deployment`
  (`pinneapple_tools.model_services`, with a trust function built from the validated domain and the gates, and
  `serve.py` written into the run folder so the endpoint restarts from the artefacts).
- `early_stopping` no longer raises `ConfigError`, and it means three documented things: `PINNMonitor` on the
  symbolic backend (it stops when a physics check keeps failing), a chunked `solve_pde` loop on the preset/spec
  backend that keeps the best held-out physics loss, and the study's own `budget_seconds` for a supervised run.
- `pinneapple check` is now a second opinion rather than a duplicate: every run writes `verification/predict.py`,
  `queries.csv` and `pinneapple-check.toml`, runs the checker over them, and the `applicability`, `export` and
  `physics` gates quote its outcomes (PC4xx/PC6xx, PC8xx, PC1xx-PC3xx) when it has any, falling back to their own
  measurement — and the report says which. `verification.check_physics` names the physics the outputs belong to;
  when the field names do not match, the run records that the physics was not applied instead of pretending.
- `pp.SurrogatePipeline.from_run(folder)` reloads a finished run — configuration, gates, dataset and the trained
  network (`model/physics/` checkpoint plus the recipe) — which is what makes `deploy()`, the checker's `predict.py`
  and `resume` in a new process work on a physics-only model. `predict_array(X)` is the `(N, n_in) -> (N, n_out)`
  surface those consumers use.
