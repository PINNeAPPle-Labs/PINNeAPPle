- `pinneapple_neural/lpm_stages/`: the design-space pipeline with a boundary per stage, starting with the four that can
  stand alone today. **`space`** samples the family's declared ranges with a low-discrepancy rule (Sobol by default,
  Latin hypercube or uniform on request), re-draws a geometry the family's `validate` refuses instead of shortening the
  dataset, and carries the active half: `refine` proposes the next bodies around the designs whose confidence sits
  nearest the decision threshold — the worst-informed body is deliberately *not* proposed, because a body the model does
  not understand at all teaches it less per simulation than one it almost understands. **`trust`** calibrates the raw
  OOD ratio into a confidence in [0, 1] with the knots stored, so the table can be published, and writes the held-out
  report with the constant-field baseline beside it and the gate table that says which gates ran and which are not
  applicable to a point cloud (conservation, boundary and residual need quadrature weights; they are reported with the
  reason, never skipped silently). **`card`** is the v2 record: dataset identity as the SHA-256 of the manifest, the
  distribution of the training parameters, the commit, the calibration knots, the gates, and the licence with the usage
  it allows — a model touched by `cc-by-nc-*` data is refused if declared `client-delivery`. **`registry`** turns the
  cards on disk into the four questions an engineer asks: which families have a model, which beats its baseline *and*
  passes its tolerance, which dataset it came from, which gate failed. The families also gain `geometry_from_unit`, the
  hook that lets a lattice sampler draw the parameter box directly (`pp_fin` derives its base width from the fin count
  and pitch, and without it the reference receives a geometry with a missing key).
