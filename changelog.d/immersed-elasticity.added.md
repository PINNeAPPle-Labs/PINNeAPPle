- `pinneapple_simulation.immersed`: linear elasticity on the immersed discretisation (issue #966, part 2):
  `solve_elasticity` with Nitsche displacements (all or some components), tractions, pressures and total forces
  (`Force`, spread over the area the selector finds), ghost-penalty or FCM stabilisation and an equilibrium check;
  `verification.elastic_manufactured` and `verification.lame_sphere`. Conforming P2 elasticity in
  `numerical_solvers.tet_fem.solve_elasticity`. `immersed.parts`: a finned heat sink and an L-bracket given only as
  STL, solved by both methods from the same file and compared with a fine conforming reference.
