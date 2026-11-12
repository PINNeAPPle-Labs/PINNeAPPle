- `pinneapple_simulation.pipelines.airfrans_benchmark`: the AirfRANS → FluidsBench submission campaign as a pipeline —
  extract the public release case by case with the AirfRANS feature definition (`extract_cases`, the only stage that
  needs the pinned VTK stack), build the eleven scaled input channels with the wall and boundary-layer band deliberately
  oversampled (`point_features`, `sample_indices`), fit the coordinate network with the optional wall no-penetration and
  outer-domain Bernoulli residuals (`train`, `physics_loss`, `TrainingSpec`), predict every native point of every
  official test case and pack the participant entry (`predict_cases`), write the per-case discretization records and the
  schema-v3 package configuration (`write_entry`), and drive the benchmark's own validate → evaluate → package → verify →
  contributor-validator chain before re-checking the package (`OfficialToolchain`, `cross_check_domain_metrics`). One
  call runs the whole thing (`CampaignSpec`, `run_campaign`), and `python -m
  pinneapple_simulation.pipelines.airfrans_benchmark` runs any stage from a JSON spec
  ([guide](docs/guides/airfrans_fluidsbench.md)).
