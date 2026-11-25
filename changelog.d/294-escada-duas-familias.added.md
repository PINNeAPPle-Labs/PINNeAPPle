- `pinneapple_simulation.cfd_validation.ladder` (family one, NACA 0012) now reports the wall pressure: every RANS rung
  carries its `Cp`, and the cost-accuracy table gains `Cp_error_rms` and `Cp_error_max_abs` against the finest rung,
  compared on the finest rung's wall stations after interpolating along the surface arclength — the norms are
  `sqrt(mean(dCp^2))` and `max(|dCp|)` in units of the dynamic pressure.
- `pinneapple_simulation.cfd_validation.ladder_ahmed` is the second family the issue asked for: the Ahmed body's six
  slant angles from Ahmed, Ramm & Faltin (1984), read from the campaign's committed records on the campaign mesh and on
  the L2 mesh. Every rung gets the drag and lift against the fine rung, the skin `Cp` vertex by vertex with the region
  the largest difference falls in, the distance to the measured drag as a second reference, and the Spearman rank
  correlation of the six designs. The finding: the campaign mesh ranks five of the six as the fine mesh does and
  inverts the 30 and 40 degree pair, and refining to L2 moves the mean drag *away* from the measurement (16.9 % to
  23.7 %), so the remaining error is the model's and not the mesh's.
- `examples/cfd/classical_cases/run_cases.py --only ladder-ahmed` writes `results/ladder_ahmed.json`; the README of the
  example carries both cost-accuracy tables, and `MULTIFIDELITY` exports each family's rungs as data levels for the
  multi-fidelity workflow (#35).
- The rungs are also priced in CPU seconds (`cpu_seconds`, the solver's own `ExecutionTime`) beside the wall time, so a
  table measured on a loaded machine can still say what a rung costs.
