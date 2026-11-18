- `pinneapple_arena.fieldbench` can now rank **parametric** field surrogates. `FieldSample.extra` carries per-node
  channels (the design vector of a campaign, broadcast to its nodes), `collate_samples`/`train_field_model` batch them,
  `build_model_specs(extra_dim=...)` builds the rows that read them and `run_benchmark` keeps them away from the rows
  that do not. `extreme_of_field` and `extreme_errors` score the extreme of one channel — the peak displacement of a
  crash design, the number the design was run for — next to the field metrics.
  `pinneapple_simulation.pipelines.pole_impact.design_vector` and `with_design_channels` map a design into `[0, 1]` and
  bring a campaign written before these channels existed forward without re-running the solver.
- **Behavior:** the batched target is now standardised with the training statistics the trainer was already computing
  and the prediction was already undoing. Before, `collate_samples` ignored the statistics, so the loss was in raw
  units (~1e-6 for a displacement in metres) while `predict_field_model` still multiplied by the standard deviation —
  a prediction returned scaled by it. The two halves agree now, and a test pins the correspondence.
- **Behavior:** a prediction whose channel count does not match the field is refused by the masked loss, by
  `predict_field_model` and by `relative_l2_per_geometry` (which now compares shapes, not sizes). A row built with
  `out_dim=1` for a three-component displacement used to be broadcast against the missing channels and scored anyway.
