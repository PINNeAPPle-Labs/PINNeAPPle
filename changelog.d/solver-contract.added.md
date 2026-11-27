- **One interface for external solvers**: `pinneapple_simulation.external_solvers` now has a contract
  (`SolverCase`, `ExternalSolver`, `check_environment()`, `prepare()`, `run()`, `status()`, `results()`,
  `cancel()`) with one definition of "finished" (`classify_run`: a divergence outranks the exit code, so `diverged`
  and `failed` are never the same answer), and six implementations behind it: OpenFOAM, CalculiX, FEniCSx, ANSYS
  Fluent, MATLAB, and the in-process hexahedral FEM. `check_environment()` never raises: it returns what is missing
  (executable, version, licence, environment, Python module) with the evidence and the fix, and
  `raise_if_missing()` turns that into one readable error. Add a solver by writing one small class instead of reading
  the previous one's case layout; `pinneapple_core.workflows.adapters` now asks the same contract whether OpenFOAM or
  CalculiX is usable, so a workflow step and the library cannot disagree. Same cantilever through CalculiX (native
  `ccx` 2.23) and the in-process FEM: 0.9937 and 0.9894 of the Timoshenko tip deflection, in one schema with
  provenance naming each backend and its version.
