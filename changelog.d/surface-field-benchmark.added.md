- `pinneapple_arena.fieldbench`: the benchmark of field surrogates over many geometries — containers
  (`FieldSample`, `FieldDataset` with named splits and a held-out-category split), metrics with the conventions two
  vehicle benchmarks use (per-geometry relative L2 averaged over geometries, pooled R², error percentiles, bootstrap
  intervals, paired bootstrap / sign-flip / Wilcoxon comparisons), one trainer for every row (padded batches, masked
  loss, target standardisation from the training split, early stopping), model rows built through one interface, the
  three-resolution protocol (`PatchwiseProtocol`) and a markdown leaderboard; plus the tabular counterpart
  (`benchmark_tabular`, `feature_sensitivity`) that compares ten regressor families on the same split with fixed
  hyperparameters.
