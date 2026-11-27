- `pinneapple_data.ashton.physicsnemo_models` (#409): DoMINO from the installed PhysicsNeMo (2.2.2) as a baseline
  member of the Ashton fixture — the same `Case` objects, the same metrics, the same `benchmark_table` the library's own
  models report into, with the training budget carried in `model_name` so a comparison against it is a comparison.
  `x_meshgraphnet_availability()` and `require_x_meshgraphnet()` fail cleanly naming `torch-geometric` and `torch-scatter`
  (absent here), and the table row reads `awaiting_data` instead of an invented number. `availability()` answers per
  component with a reason for every "no" and imports nothing.
- `pinneapple_data.ashton.physicsnemo_trust` (#409): the model as a `PhysicsEnsemble` member with the embedding the trust
  layer reads, and `cross_family_with_domino` — train on one geometry family of the fixture, walk towards the other, and
  report the field and Cd error, the embedding distance, the flagged fraction and the conformal coverage at every step.
  Measured here: the embedding catches the shift in both directions (flagged fraction 1,000 at t = 0,6 going α → β and at
  t = 0,4 going β → α) while the residual ratio does not (it rises in one direction and falls in the other), and the 90 %
  band leaks from 0,90 to 0,86 rather than failing.
