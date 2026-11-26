- `pinneapple_pipeline` / `pp.SurrogatePipeline`: one object for the whole cycle of a surrogate model — problem →
  data → preprocess → model → training → export → verification — as a facade over what the library already does
  (presets and condition sampling, reference and analytic solves, `design` of experiments, the supervised surrogate
  study, `train_pinn`/`validate_pinn`, model export), plus `pp.pipeline` and `python -m pinneapple.pipeline
  validate|plan|run|resume|report|presets`. A configuration is seventeen dataclasses that round-trip through YAML
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
