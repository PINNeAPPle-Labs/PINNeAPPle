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

Measured end to end: on the 60-case holdout the binned target is the first change to move the final checkpoint without
epoch selection (wall shear 33.63 % → 27.22 % at 48 bins, 23.80 % at 24 bins with 32 epochs, `Cd R²` −0.14 → +0.39, +5.9
points of the composite proxy). Packaged at 760 cases over the same trunk as the 75.44 entry, the official test scored it
**68.09** — wall shear 51.12 % and `Cd R²` 0.178 — so the local gain did not transfer. The flag stays, off by default,
with both numbers next to it.
