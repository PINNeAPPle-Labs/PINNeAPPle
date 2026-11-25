- **The core API snapshot carries the contact preset's names**: `api/core.json` did not list
  `pinneapple_presets.cases.contact` — `BoltedJoint` with its ten methods, and the two module helpers `runs_dir` and
  `case_run_dir` — so `tests/test_deprecation_and_api.py::test_core_api_matches_the_committed_snapshot` failed on
  `main` after the preset that made contact with friction a capability of the catalogue (#1507, PR #2517). The
  snapshot is regenerated from the source (`python -m pinneapple_stability.api --write`); nothing else changed.
