- Dynamics toolkit integration:
  - `pinneapple_analysis.dynamics.first_passage.survival_from_density` and `survival_from_solution`: the
    hitting-time distribution from a Fokker-Planck density with an absorbing boundary, from an array or a
    `pp.solve` result (survival, density of tau, hazard, restricted mean, with diagnostics).
  - `examples/dynamics_toolkit/`: the five frameworks end to end, each against an independent reference
    (first passage from a `pp.solve` PINN, the Koopman spectrum of a PINN, sensor times by Fisher information,
    Buckingham pi with curve collapse, early-warning indicators with controls), with `experiment.json`.
  - Catalog items S42-S46 for the five frameworks; `docs/guides/dynamics_toolkit.md` is now the index, with the
    example's measured results.
