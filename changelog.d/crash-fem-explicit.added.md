- `pinneapple_simulation.numerical_solvers.crash_fem`: an explicit structural crash solver — central-difference
  time integration at the exact critical step of the mesh, 2x2x2 Gauss integration, corotational kinematics, J2
  plasticity with isotropic hardening and element erosion, penalty contact against a rigid wall or pole (with
  friction), the energies of the run and the crashworthiness quantities (`CrashSolver`, `Material`, `RigidPole`,
  `RigidWall`, `CrashResult`). `crash_cases` builds the bumper-beam pole-impact coupon that drives it (seven design
  variables, a fixed mesh with 3 to 11 semantic parts), and `pinneapple_simulation.pipelines.pole_impact` turns a
  sweep of it into a displacement-sequence dataset plus a quantities table.
