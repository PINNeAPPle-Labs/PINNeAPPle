- `pp.load_checkpoint(path)` rebuilds a model saved by `save_checkpoint` in one call: checkpoints now record the
  registry name and constructor arguments. Checkpoints written before this cannot be rebuilt without them and say so
  (#941).
- Every registered solver has `solve(...)`, and `SolverCatalog().describe(name)` returns its constructor, entry point
  and an example call (#957).
- `pinneapple_physics.pinn_solver.compiler.pointwise_residual(spec)`: the pointwise PDE residual a spec trains on.
