Eight bugs found while writing `GETTING_STARTED.md` are fixed, each with a regression test in
`tests/test_v1_known_bugs.py`:

- `pp.validate(model, {"pinn": spec})` accepts a `ProblemSpec` (and `"ctx"` for a source term) and checks the
  residual the solver trains on (#939).
- The Arena builds `ModifiedMLP`; `ModifiedMLP` also takes its activation by name (#940).
- `pp.solve("burgers_1d", "reference")` uses the preset's initial condition, covers the whole time span and keeps
  the explicit step stable; it now agrees with the Cole-Hopf solution to within 4 % (#942).
- `pp.quickstart` prints one `name (kind)` per condition, and benchmark reports name the problem (#945).
- `SolverRegistry.build` names the constructor when given wrong arguments, and `SolverCatalog.get` works (#957).
