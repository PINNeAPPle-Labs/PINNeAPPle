- `pinneapple_neural/lpm.py`, `pinneapple_neural/lpm_pipeline.py`: the multiscale neighbourhood features are computed
  **once per design**, at dataset time, and stored in the `.npz`; the model reads them from the batch instead of
  rebuilding the query-to-surface distance matrix on every forward. `Design.local` and `DesignBatch.local` carry them,
  `stack_designs` pads them and falls back to computing when only some designs have them, and a cache built for another
  radius set is ignored rather than trusted (the model checks its width against `len(radii) * (2d + 1)`). Measured on
  the campaign's own data: **1.6× per epoch on `pp_wing3d`** (960 points, batch 24) and **2.2× on `pp_fin`** (111
  points), with the fit's loss history identical to 1e-6. Not the 50× that Luminary reports from caching neighbour
  indices: their bottleneck was data loading starving a GPU, and this distance matrix is only part of our forward — the
  claim is the measurement, not the analogy. `train()` takes the radii from the manifest, so the cache and the model
  cannot drift apart.
