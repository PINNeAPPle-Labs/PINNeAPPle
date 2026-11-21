### Added

- `boundary_layer.shape_correction` and `panel.panel_shape_correction`: the wall-shear *shape* to go with the flat-plate
  law's level — the integral boundary layer's `τ_BL / τ_placa` along the section, driven by the panel solve's edge
  velocity, damped by `CORRECTION_GAIN` and read at every node by chord position from its nearest contour point.
- `point_features(..., correction=True)` appends it as one channel (column `len(FEATURE_NAMES)`), `WallAwareNet`
  multiplies its flat-plate prior by it, and `WallSpec` / `TrainingSpec` carry the flag into the feature cache and the
  checkpoint. Measured, it is neutral on the 61-case holdout — wall shear 44.56 % against 44.90 % for the same recipe
  without it, wall pressure 2.70 % against 2.97 % — so it is off by default and the number travels with the flag.

### Fixed

- `panel_shape_correction` fed the integral boundary layer an edge velocity in units of the freestream where Reynolds
  number needs metres per second, which ran every case at a thirty-fold lower Reynolds number; and it read the wall
  normals from the raw point block's columns 4:6 — `(n_y, surface)` — where the release leaves them zero off the
  surface, so the channel was a constant one over the whole field.
- `edge_solution` raised `IndexError` on a surface where Michel's criterion never trips, because the turbulent march was
  started at the station past the end of the chord instead of being skipped.
- `WallSpec` was missing the `correction` field its own `as_dict` read.
