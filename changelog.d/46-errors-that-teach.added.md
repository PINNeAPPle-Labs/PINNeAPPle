- `pinneapple_core.errors` — errors that teach (#46, roadmap A5). An exception now carries the action that clears the
  failure, not only the description of it: `PINNeAPPleError` renders a second line (`Fix: pass repeats=5`) and exposes
  it as `.fix`, and each subclass also inherits the standard exception a caller already catches (`BadValue` is a
  `ValueError`, `MissingExtra` an `ImportError`, `UnknownName` a `KeyError`, `NotConfigured` a `RuntimeError`,
  `NotSupported` a `NotImplementedError`) — better messages, unchanged `except` clauses.
- The thirty most common failures a user of this library hits were audited and rewritten: the cost profiler
  (`measure`, `scaling_study`, `expected_exponent`, `Budget.validate`), the equal-budget toolkit, the method and
  benchmark lookups, the benchmark config, the solver registry, `TrustGate`, the `pp.tensor` unit rules, the
  optional-dependency errors (pyarrow, h5py, the hub extra, onnxruntime, the retrieval extra, the dynamics backends).
  The table lives in `pinneapple_core.errors.AUDITED` and is rendered into `docs/dev/errors_that_teach.md` by
  `python scripts/build_error_audit.py`; `tests/test_errors_that_teach.py` triggers every one of them and refuses an
  entry whose error comes back without its fix or without its standard exception type.
