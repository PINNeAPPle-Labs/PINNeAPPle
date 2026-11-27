- `pinneapple_analysis.inverse_problems.symbolic_distillation` (#400): read an equation out of a trained inverse model
  and verify it as an equation. `field_dataset` / `module_dataset` sample a model (a network, a `PolynomialSurface`, a
  `NeuralTermDiscovery` or any numpy callable) inside the `Support` it was fitted on, with derivatives through
  `torch.autograd`; `distill` fits SINDy (`CandidateLibrary`, `SINDyIdentifier`) and genetic programming
  (`gp_symbolic`) over the same samples, reports the Pareto front of complexity against error, refits every bootstrap
  resample and every ensemble member to keep only the terms that survive, and writes an `EquationReport` (SymPy,
  LaTeX, Python, per-term intervals and units, the reasoning behind every quantity it could not measure).
  `simulate_evolution` puts the equation back into `pinneapple_physics.symbolic_pde` and integrates it, and `verify`
  scores it against the network that produced it; on a heat equation read from a surface fitted on `t` in [0, 0.4],
  the distilled law recovers `D = 0.499945` (true 0.5) and reproduces the trajectory over `t` in [0, 4] to 6.8e-5,
  where the surface itself is off by 1057 %. PySR stays an optional extra. Guide:
  `docs/guides/symbolic_distillation.md`.
- **Behavior:** `pinneapple_data.expressions`: `tanh` is in the function table (`FUNCTIONS` and the torch table), so
  `pinneapple_analysis.inverse_problems.gp_symbolic.parse_expression` reads the `tanh` the genetic-programming engine
  writes (it was accepted by the engine's own operator table and refused by the gate in front of the parser).
- `pinneapple_analysis.inverse_problems.gp_symbolic`: a search without unary operators (`operators=()`) no longer
  breeds a one-argument `*` node that every evaluation crashed on; an operator the expression language does not have
  (for example `u ^ 2`) is refused with `PolicyViolation` instead of leaking a `KeyError`.
