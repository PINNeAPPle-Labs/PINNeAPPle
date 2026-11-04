- `pinneapple_problemdesign.run_draft.run_draft(draft, geometry, method="auto"|"fd"|"pinn", budget)` runs a
  `formulate` draft as a job on a box: second-order finite differences (Crank-Nicolson with Newton for a
  temperature-dependent source, ghost-point flux conditions) for a scalar diffusion-reaction law, else a PINN on the
  draft's own loss compared with the FD solution; every result states the manufactured-solution order on three
  grids, the balance of the drafted law, the time-step change and every assumption with its criterion on the box.
  A draft that cannot be discretised (an algebraic relation, coupled flow fields, an advective term, a nonlinear
  condition, a missing constant) is refused with the reason. Flow node `problemdesign.run_draft`, bundled flow
  `run_drafted_model.json`, planner capability `problemdesign.run_draft` (#1246).
