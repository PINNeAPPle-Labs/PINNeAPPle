- **Cost measured per catalog method, with the machine written down** (`pinneapple_catalog/method_costs.json`,
  `scripts/measure_method_costs.py`, #52): every entry of `pinneapple_catalog.methods.METHODS` now has a row in the
  versioned table -- `measured` with the median time and its interquartile range per size, the peak memory and how it
  was sampled (`rss` / `tracemalloc`), the FLOPs and their source (`flop_counter` / `profiler` / none, which is an
  answer and not a gap), the scaling exponent with a bootstrap 95 % interval over the sizes it was fitted on, and the
  machine (CPU by name, cores, platform, Python and torch, commit) the numbers came from; or `unmeasured` with a
  blocker from a closed set (`needs_cuda`, `needs_external_solver`, `needs_optional_package`, `needs_input`,
  `not_a_single_operation`, `definition`) and the specific reason. New probes measure the finite-difference SOR solve
  (S1), the Q1 FEM assembly and CG (S2), the finite-volume explicit Euler step (S3), the TVD upwind right-hand side
  (S11) and one epoch of the base trainer on CPU (T1); each fits a real exponent instead of declaring one.
  `--check` fails on a catalog entry with no row, on a table that lost its machine, and on an unexplained gap, and
  `docs/dev/METHOD_COSTS.md` is generated from the table by `--docs` so the per-method line cannot drift from it.
