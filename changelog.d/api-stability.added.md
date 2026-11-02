- `pinneapple_stability` sets out what PINNeAPPle promises on the way to 1.0 (#1127):
  - **core** (`CORE`, `CORE_PP`): semantic versioning.
  - **experimental**: every other module, unchanged but without a compatibility promise.
  - **labs** packages: these also issue an `ExperimentalWarning` on first import. Silence it with `PINNEAPPLE_QUIET_EXPERIMENTAL=1`.

  The Studio catalog gives every tool a `stability` field. `docs/api_stability.md` describes the policy and the path from experimental to core. Nothing is removed.
