- Bring your own foundation model (#348), and the checks that say where it stops: `pinneapple_physics.pretrained`
  wraps a model trained somewhere else as an `Expert` of `PhysicsEnsemble`, under the same residual, measurement and
  horizon checks as the models trained here. `REGISTRY` knows the open models worth wrapping (the WeatherBench2
  forecasts IFS HRES, Pangu-Weather, GraphCast and NeuralGCM, Prithvi for satellite imagery, SAM 2 and
  Depth-Anything for vision) with the source, the envelope and the **licence**, with the date and place it was read or
  an explicit `NOASSERTION`; `known_models()` and `spec()` read it, and `licence_report()` collects the two lists a
  deployment asks about (nobody read the licence at the source; the licence forbids commercial use).
  `from_pretrained_expert(name, adapter)` returns the expert with that record in its provenance, and the four adapters
  cover the ways a model arrives -- `WeatherBench2Adapter` (the published archive, through the same loader the skill
  curves use), `TorchCheckpointAdapter` (a checkpoint in this repository's layout, rolled out autoregressively),
  `ArrayArchiveAdapter` (a local `.npz`: the offline path) and `HuggingFaceAdapter` (the Hub, or `local_path`). An
  adapter that cannot load raises `PretrainedUnavailable` naming what is missing and how to point it at a local copy;
  it never returns a prediction it did not make, and the ensemble counts that expert as failed for the case. The cost
  of a call may be a function of the query (a forecast that grows with its lead time is priced per case), and
  `EnsembleRun.summary()` reports the licences of the experts that ran.
- The checks a domain makes with no reference field (#348), in `pinneapple_physics.domain_checks`: `advection_residual`
  (the part of a 6-hour change the wind cannot explain -- 0.0028 for a field that is exactly advected, 1.000 for the
  same field sitting still under that wind, 0.974 for noise), `global_drift` and `dry_static_energy` (the area-weighted
  total of `cp T + g z` does not wander: 0.000 for a constant, 0.040 spreads per day for a 5 % step over 6 h),
  `hand` and `flood_mask_score` (water on terrain: a mask inside the 12 m depth scores 1.000, the same mask with 3 m of
  water 0.306, and a mask that wets the outlet without its donors 0.800), and `temporal_residual` with
  `lucas_kanade` (frames translated by two pixels: flow residual 5e-9; frames of noise: 0.974 with 7.5 % of the pixels
  teleporting). Each is a `ValidityCheck` and `as_prediction_check` turns it into the `(score, reason)` callable
  `PhysicsEnsemble(prediction_checks=...)` weights the experts with. The demo over 2020 with the three weather models
  side by side is the next slice: it needs the archive subset and a trained checkpoint.
