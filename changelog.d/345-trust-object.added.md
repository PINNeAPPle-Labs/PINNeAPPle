- The trust layer as one object (#345): `pp.trust(model, data, problem)` returns a `Trust`
  (`pinneapple_analysis.trust.record`, schema `pinneapple.trust/1`) — the record a prediction carries and a consumer
  reads. It holds `in_domain`/`domain_score`, the `horizon` (useful and no-skill lead), the `residuals` per law with
  the in-domain level of the same model, the `measurement` checks on the readings the prediction used, the calibrated
  `uncertainty` (level, measured coverage, width, ECE) and the `provenance` (model and data fingerprints, versions),
  with `to_json()`, `from_json()`, `badge()` (good / fading / no_skill) and `badge_reason()`. Every field is a
  `TrustValue`: a number with the source it came from, or `not_measured` with the reason — a measured value without a
  source, a `not_measured` carrying a value and a reference without its own source are refused at construction.
  `badge()` is a decision over measured gates, not a weighted score: no_skill when a measured gate proves the
  prediction carries no information (the input is outside the training distribution, or the requested lead is past
  the no-skill lead), fading when a measured gate fails or a gate was not measured, good when every gate was measured
  and passed. Adapters read the existing measurements instead of repeating them:
  `adapters.from_ensemble` (`PhysicsEnsemble` residuals, spread, interval and suspect reading),
  `weather_horizon` (`pinneapple_physics.weather.evaluate.horizons`), `rollout_horizon`
  (`validation.rollout.divergence_time`), `calibration_uncertainty`
  (`uncertainty.calibration.CalibrationMetrics`) and `verification_measurement` (`pinneapple_decision.verifiers`).
