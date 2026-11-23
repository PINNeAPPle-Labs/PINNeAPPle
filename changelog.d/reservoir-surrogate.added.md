- Added a reservoir-surrogate workflow, `pinneapple_neural.workflows.reservoir_surrogate`, which
  trains a registered neural operator from well controls to the water-saturation and pressure
  fields of a waterflood study. It reads the two-file HDF5 layout a simulation campaign writes
  (statics plus per-scenario fields, NaN outside the active cells), refuses a dataset with
  non-finite values inside the active cells, and trains any of `hadamard_deeponet`,
  `chebyshev_branch_trunk` or `cpno` through each one's own calling convention. The score is
  k-fold over the training scenarios plus a separate holdout file, reported as relative L2 on
  the active cells only, next to `breakthrough_time`, which measures water arrival against the
  cell's own starting saturation rather than an absolute level. The workflow never runs a
  solver: it starts from the HDF5 the campaign produced, which is what lets the library, a CLI
  and an API share it without any of them needing a container.
