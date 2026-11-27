- The physical data graph (#156): `pinneapple_data.lineage` gains the `problem` and `validation` kinds and
  `Lineage.trace()`, which answers, for a prediction, the problem, the geometry, the simulation runs, the dataset
  version, the model version and the validation record — and names in `missing` the groups nothing recorded instead
  of leaving a hole. `from_registry(registry, problem_id, prediction=...)` writes the edges the four
  `pinneapple_registry` stores already keep (the `problem_id` every record is filed under, the dataset a model's
  metadata names, the run it cites, the trust report or `ValidationReport` stored with the version) and refuses to
  invent the ones they do not: a model that never named a dataset is linked to the problem only, and its node and
  the trace both say so. New page: `docs/core_concepts/physical_data_graph.md`.
