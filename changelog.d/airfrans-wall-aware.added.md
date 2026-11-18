- `pinneapple_simulation.pipelines.airfrans_benchmark.wall`: the wall quantities the FluidsBench AirfRANS evaluator
  derives from a submitted field instead of reading from it — the dimensionless wall slope (`-wallShearStress/nu` in
  units of `U_inf/chord`), the airfoil polyline quadrature and the four force coefficients recomputed from a predicted
  curve, differentiable so a training loss sees the same numbers the evaluator will. `write_wall_store` caches the
  per-case truth (slope at every surface node, segment lengths and normals, reference coefficients) once, in about a
  second a case, and `curve_coefficients` reproduces `Simulation.force_coefficient` to five decimals.
- `WallAwareNet`: a residual coordinate network with three heads (outer velocity, wall slope, pressure) and an
  analytic wall layer `U = A(d)U_outer + (1-A(d)) s d` with `A(d) = 1 - exp(-(d/lambda)^2)`. No-slip is exact rather
  than penalised, and the first derivative at the wall is the slope head, which is what the benchmark reconstructs to
  score wall shear and drag.
- `shape_descriptor`: the case's upper and lower airfoil surfaces on 32 chord bins, appended as 64 per-case channels.
  A point-wise network cannot tell two airfoils apart at the same location; a nearest-neighbour probe puts the wall
  slope error at 75.7 % without the descriptor and 42.1 % with it.
- `WallSpec` / `train_wall` / `curve_force_loss`: the wall-aware objective — the data term, a relative L2 on the wall
  slope, and the squared drag and lift error of the coefficients reconstructed from the predicted curve.
- `docs/guides/fluidsbench_datasets.md`: what each open FluidsBench dataset needs in storage and compute, measured
  against this machine, and the exact next command once one that fits exists.
