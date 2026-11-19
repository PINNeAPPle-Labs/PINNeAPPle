- **Added:** a `ProblemSpec` face for every solver in the library that only had a direct API, and the documented-scope
  entries that let `select_solver_family` route to them, so a plan stops calling a covered problem a gap: `fdm3d`
  (`HeatConduction3D`, 3-D transient heat with cold walls and a source), `solid_fem` (`SolidFEM`, C3D8I hexahedra
  with clamps, a face traction and a body force, plus the new `SolidFEM.load_body`), `wave3d_fdm` (`ElasticWave3D`),
  `wave2d_fdm` (`wave2d.Wave2D`, new: the 2-D acoustic wave equation between the 1-D and 3-D solvers that existed),
  `darcy2d_fvm` (`operator_benchmarks.darcy_2d`, `-div(a grad u) = f` on the unit square), `heat1d_fdm`
  (`heat1d.Heat1D`, new: Crank-Nicolson with the ends held at a temperature), `elliptic3d_fdm` (`elliptic3d`, new:
  3-D Poisson/Laplace/Helmholtz by seven-point differences and red-black SOR) and `reaction_diffusion_nd`
  (`reaction_diffusion.ReactionDiffusion`, new: the ETDRK4 departure in 2-D and 3-D, with the reaction `"none"` for
  pure diffusion). `fem` gained 2-D plane stress and plane strain (`linear_elasticity`, `plane_stress`/
  `plane_strain` and the presets' kind names, at dim 2 only), so 2-D elasticity is solved by Q1 elements instead of
  falling through to Poisson. Each of them refuses, by name, a spec it cannot honour instead of solving it with a
  boundary it does not have.
- **Changed:** `_CLASSICAL_SOLVER_SCOPE` entries can now say `not_kind_substrings`, and the matcher honours it in
  `select_solver_family` and in `pinneapple_advisor.compose.documented_families`. The substring claim was too coarse
  in a way that mattered: `fdm`'s and `fvm`'s "diffusion" is the linear equation with a source `q(x, t)`, and
  `fem`'s "linear_elasticity" is contained in "nonlinear_elasticity", so both were claiming problems they would have
  answered with the wrong operator.
