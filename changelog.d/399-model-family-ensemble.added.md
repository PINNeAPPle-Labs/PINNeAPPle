- `pinneapple_physics.blackhole.family` (#399, plan item 4): the model family of the black-hole weather forecast and
  its ensemble, on the forecast contract of `forecast.rollout` (5 frames of normalised log density in, the next 5 out),
  so every member is scored by the same `skill.evaluate_skill` as the numbers the module already reports. Members: the
  paper's `DuarteUNet` (residual), `FourierForecaster` (the repo's `FNO2d` on the low modes of the r-theta plane) and
  `SphereBackbone` (the `SphereUNet` of `pinneapple_physics.weather`, with `grid_pad` in place of its spherical
  padding: zeros in r, replication in theta, because the accretion grid is a meridional half-plane and not a globe).
  Aggregations: `EqualWeightEnsemble` (the plain mean, an exact identity on identical members) and
  `adaptive_ensemble` (`PhysicsEnsemble` in combine mode, fixed-share/AdaHedge weights learned prequentially, tilted by
  the mass rate of the rollout -- label-free, as `residual_lookahead` asks). `evaluate_family` reports each member and
  both aggregations against persistence, DMD and the time-mean flow with the moving-block bootstrap, the paired
  ensemble-against-best-single comparison, and walks the best single model on the validation trajectory only.
  `skill.evaluate_skill` now also carries the anomaly correlation of every model by lead and the lead where it falls
  below 0.6 (`acc_by_lead` and `acc_horizon_M` in the model report, from the new `skill.acc_by_lead` and
  `skill.first_below` helpers), which is what the trust bar of the animation draws.
  `figures.plot_family` and `figures.forecast_gif` (Earth-2 style: simulation and forecast side by side, the trust bar
  underneath, palettised frames so the GIF stays small) and `examples/black_hole_weather/family.py` regenerate both.
