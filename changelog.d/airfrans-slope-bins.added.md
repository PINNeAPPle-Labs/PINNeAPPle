### Added

- `WallSpec.slope_bins` (and the campaign's `--slope-bins`): train the wall readout on each chord bin's mean slope
  instead of each node's own. The truth curve is representable to 12.1 % by 33 bins a side — under the 20 % cap that
  the wall-shear term needs — while the per-node target carries the node-to-node variation the evaluator's gradient
  reconstruction amplifies and a per-case MLP cannot know. Off by default; the feature cache keys on the bin count, so
  every arm that does not use it keeps its cached sample.
- `WallSpec.slope_width` / `WallAwareNet(slope_width=...)`: the readout's hidden width, carried in the checkpoint, because
  it is the whole near-wall story and a 512-wide head cannot load into a 256-wide one.
- `WallSpec.slope_floor` is now reachable from the campaign (`--slope-floor`), which is how the metric-weighting
  question — the slope loss is a per-point relative error while the benchmark weights by `τ²` — can be measured.

- A second stage (`init_from`) no longer refuses to start when the readout was widened: the trunk and the buffers
  transfer and only the mismatched readout tensors are skipped, with the skipped groups printed. Training a 512-wide
  readout from a 256-wide checkpoint is exactly what the width option is for.
