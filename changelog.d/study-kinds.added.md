Studies other than design optimisation:

- `pinneapple_flow.studies`: ten kinds of study (design optimisation, simulation, surrogate, inverse problem,
  uncertainty and sensitivity, digital twin, forecast, control design, coupled multiphysics, experimental data
  analysis), each with its own steps, checks and what "verified" means, words in English and Portuguese, and
  `classify_study(goal)`.
- `pinneapple_flow.study_steps`: steps that take a model as data (a *model reference*: a library tool, fixed inputs,
  varied inputs, outputs), so a pipeline can chain them across sandboxed steps: `sensitivity_step` (Sobol),
  `uncertainty_step` (Monte Carlo with a convergence check and exceedance probability), `identify_step` (calibration
  with intervals and held-out rows), `analyze_step` (predictability) and `forecast_step` (AR scored per horizon
  against persistence and the mean, out of sample). Each returns its checks.
