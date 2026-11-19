- **Added:** a `ProblemSpec` face for five solvers that only had a direct API, and the documented-scope entries
  that let `select_solver_family` route to them: `fdm3d` (`HeatConduction3D`, 3-D transient heat on a box with
  cold walls, optional volumetric source) and `solid_fem` (`SolidFEM`, 3-D linear elasticity on C3D8I hexahedra
  with clamped faces, a face traction and a body force, plus the new `SolidFEM.load_body`), `wave3d_fdm`
  (`ElasticWave3D`, the 3-D acoustic wave equation), `darcy2d_fvm` (`operator_benchmarks.darcy_2d`, `-div(a grad
  u) = f` on the unit square) and `reaction_diffusion1d` (`ReactionDiffusion1D`, Allen-Cahn and Fisher-KPP on a
  periodic interval). Each refuses, by name, a spec it cannot honour instead of solving it with a boundary it
  does not have. With the entries in
  `_CLASSICAL_SOLVER_SCOPE`, 3-D heat no longer falls through to the PINN, and the plan that recommends a family
  reads the documented scope rather than what the machine happens to have installed. The table grew the exclusion
  it was missing: `fdm`'s and `fvm`'s "diffusion" is the linear equation, so neither claims reaction-diffusion.
