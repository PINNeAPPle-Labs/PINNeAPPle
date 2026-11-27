- **The Flow coverage gate is green again, with three curated reasons**: `pinneapple_analysis.report` (the one-page
  experiment report and its PDF -- a view of the manifest, the trust card, the quality checklist and `costs.json`, not
  a computation a node performs), `pinneapple_analysis.surrogate_loop` (the surrogate-in-the-loop loop with a
  verification budget, which chooses the calls a campaign makes rather than being one of them, the reason
  `pinneapple_data.adaptive_sampling` already gives) and `pinneapple_physics.constraint_engine` (one constraint object
  applied by the trainer's penalty, the optimizer's `ineq`/`eq` and the deploy guard). Each landed in a merge without a
  node or a stated reason, and `tests/test_flow_library_nodes.py` failed on main for it.
