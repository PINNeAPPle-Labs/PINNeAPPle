- **FEniCSx exercised, and the dead FEniCS wrapper fixed**: the `fenicsx` backend of the solver contract now pins the
  DOLFINx release it speaks (0.8-0.9) as a *version requirement* -- on a machine with 0.11 installed,
  `check_environment()` reports `dolfinx version: 0.11.0 is outside what this bridge speaks ... (fix: conda install -c
  conda-forge fenics-dolfinx=0.9)` instead of a `TypeError` inside the first run, and the bridge builds its function
  space through `dolfinx.fem.functionspace` so it works on 0.8 and 0.9 alike. It also returns the coordinates of its
  degrees of freedom (`extras["dof_coordinates"]`, aligned with the field vector) and no longer splits a scalar
  temperature into fake `ux`/`uy` components, which is what makes a result checkable. New
  `SolverCase.conduction` + `conduction_profile`: steady `-k T'' = source` with a closed form, solved through the
  contract and measured against it (a linear profile comes back to 1.8e-6 of its range, a sourced one to 2e-6 at
  every refinement). `external_solvers.fenics.solve_and_package` -- which imported `FEniCSBridge`, a class that never
  existed, and called it with keyword arguments it does not take -- now translates `FEniCSConfig` into the problem spec
  the bridge reads (`problem_spec()`), packages the fields with their points and raises `FEniCSUnavailable` with the
  install command when FEniCS is missing; the strict `xfail` that recorded the bug is now a passing test. COMSOL is
  declared out of scope in the contract, in the code and in `docs/guides/external_solver_contract.md`: no
  implementation exists here and one that cannot be licensed and exercised would only move the failure into the run.
