- **The hub index is green again: every snippet has a recorded run of its current code**: 239 of the 967 snippets
  in `data/web/hub/snippet_runs/` had none -- the auto-call catalog grew (the `pde_environment` presets, the closed
  forms, the trainer and validation helpers) faster than the recording did, so
  `python -m pinneapple_hub.library check` failed for each of them and the `generated` job of `Tests (audit + core
  suite)` was red on every push. The records are written by `python -m pinneapple_hub.library run --stale` (224 runs,
  none of them failing), which is the same command the weekly refresh uses.
