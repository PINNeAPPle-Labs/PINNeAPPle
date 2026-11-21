### Added

- `WallAwareNet(..., stream=True)`: the outer velocity as the curl of a scalar stream function — one head output fewer,
  and `∂u/∂x + ∂v/∂y = 0` identically instead of by luck. The derivative is taken with respect to the scaled
  coordinates the network already reads, so the freestream adds with zero divergence and the wall layer is untouched:
  no-slip stays exact where its weight vanishes. `WallSpec.stream` / `TrainingSpec.stream` carry the flag into the
  methodology record and every checkpoint, and the campaign's `--stream` reproduces the arm.

  What justified it was measured first, on the 600-case model: outside the wall layer the predicted field's divergence
  is **16 to 52 times** the release's own (0.44–0.76 against 0.008–0.041 in units of `(c/U∞)·∇·u`), while the release
  sits on the reconstruction's own noise floor. The obvious cheaper way to impose continuity — the inner-layer relation
  `v = −½ a′(x) d²` between the wall and the readout's own slope — was tried and rejected: the release's own flow
  violates it by a factor of 17 in the same band, so it is not a relation a real flow satisfies at this scale.

At the reference budget it is the campaign's clearest case of a metric's *shape* deciding an experiment: wall shear
44.90 % → **32.65 %** (the largest wall gain of any 200-case arm) and 7.5 proxy points lost, because both wall shears are
past the 20 % cap and score zero while the field it gives up is scored.

At 560 cases the wall gain does not reproduce — **38.45 % against 35.80 %** for the same recipe without the stream — so
the 200-case advantage was a small-data effect rather than the architecture's; the wall pressure is the one channel it
improves (2.12 % → 1.19 %), and the composite proxy moves 41.02 → 41.59.
