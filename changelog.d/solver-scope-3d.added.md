- **Added:** a `ProblemSpec` face for two solvers that only had a direct API, and the documented-scope entries
  that let `select_solver_family` route to them: `fdm3d` (`HeatConduction3D`, 3-D transient heat on a box with
  cold walls, optional volumetric source) and `solid_fem` (`SolidFEM`, 3-D linear elasticity on C3D8I hexahedra
  with clamped faces, a face traction and a body force, plus the new `SolidFEM.load_body`). Both refuse, by name,
  a spec they cannot honour instead of solving it with a boundary they do not have. With the entries in
  `_CLASSICAL_SOLVER_SCOPE`, 3-D heat no longer falls through to the PINN, and the plan that recommends a family
  reads the documented scope rather than what the machine happens to have installed.
