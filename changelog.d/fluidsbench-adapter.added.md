- `pinneapple_arena.fluidsbench`: an adapter for the FluidsBench benchmark that reads the published contract from a
  checkout of the benchmark repository (`Checkout.dataset(id)` — status, open splits with their digests, the
  scoring-support release, the composite and its caps, the required predicted fields), verifies the benchmark-owned
  support chain (`load_support_release`), maps a model's own output onto the canonical entities and declares the
  coverage it reached (`map_to_support`, identity / by id / nearest / linear), computes the published reductions with
  chunk-safe sufficient statistics and per-geometry macro-averaging (`relative_l2_statistics`,
  `relative_l2_from_statistics`, `aggregate`), writes the schema-v3 result package with its digest chain
  (`build_package`), recomputes that package's identities (`validate_package`, optionally running the benchmark's own
  validator) and turns scope / integrity / coverage / applicability into a `submit` / `review` / `hold` decision
  recorded in a VeriPhysics evidence ledger (`assess_submission`). `python -m pinneapple_arena.fluidsbench` exposes
  `datasets`, `dataset`, `support` and `validate`; `examples/fluidsbench_airfrans` rehearses the whole path on
  potential flow around a cylinder ([guide](docs/guides/fluidsbench.md)).
