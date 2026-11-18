- `pinneapple_neural.architectures.crash_solver.CrashSolverModel`: a hierarchical, part-aware field surrogate — shared
  per-part physics attention, a transformer over the part tokens, message passing on the part interface graph and a
  Fourier-time decoder that emits a displacement sequence — together with
  `pinneapple_neural.architectures.point_surrogates` (`PointNetSurrogate`, its wide residual variant,
  `EdgeConvSurrogate`, `PartPointNetSurrogate`). `PhysicsAttention`/`TransolverBlock` now take a mask, so a padded
  component of a batch is encoded exactly as it would be alone.
