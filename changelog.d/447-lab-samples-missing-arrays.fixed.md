- `pinneapple_lab.store.LabStore.samples` skips a run whose dataset arrays are not on disk instead of raising
  `FileNotFoundError` (#447). The shards are not versioned, so after a `reindex` or in a fresh clone every run
  that was committed as metadata only got in the way: `car_surrogate` died on the 40 committed car LBM runs
  before it could train on the 36 that do have data. The caller's own "needs at least N runs" message now
  reports what is usable, and the arrays come back with `python -m pinneapple_lab sweep <experiment> --force`.
