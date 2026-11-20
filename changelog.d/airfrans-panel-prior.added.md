### Added

- `pinneapple_simulation.pipelines.airfrans_benchmark.panel`: a Hess–Smith panel method as the network's inviscid prior
  — `airfoil_contour` rebuilds an ordered, counter-clockwise section from the release's unordered point cloud,
  `camber_line` and `zero_lift_angle` supply the prescribed circulation from thin-airfoil theory, and `panel_prior`
  returns the field at the release's own nodes in the network's units. Validated against the exact cylinder solution
  (0.046 % at 240 panels, either contour orientation) and Kutta–Joukowski.
- `point_features(..., panel=True)` appends those three channels, `CoordinateNet` and `WallAwareNet` take a `panel`
  flag, and `TrainingSpec` / `WallSpec` carry it into the methodology record, the feature cache and the checkpoint.
