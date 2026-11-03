# Reactor sizing: cooled CSTR (propylene glycol)

The smallest reactor meeting the conversion, hot-spot and runaway limits: the reactor model, a learned model only when its measured cost pays off, optimisation, and the design re-verified by the model (pinneapple_physics.reactors.design).

Flow: `pinneapple_flow/flows/reactor_sizing_cstr.json` (as the example flow `industrial/reactor_cstr`). Run it with `python -m pinneapple_flow example-flows run industrial/reactor_cstr [--quick]`; regenerate this folder with `python -m pinneapple_flow example-flows gallery industrial --entries reactor_cstr`.

Quick configuration (parameter overrides per node): `{"design": {"maxiter": 8, "popsize": 6}}`.

Every number below was measured by a run of the flow on the commit and date stated; none is typed in.

## quick

done in 11.9 s (commit d83a70bc, 2026-11-03).

| metric | value | how |
|---|---|---|
| `design.design.V` | 2.18209 | optimum, re-solved by the model |
| `design.design.UA` | 3062.18 | optimum, re-solved by the model |
| `design.model.X` | 0.821378 | reactor model at the design |
| `design.model.T_max` | 329.685 | reactor model at the design |
| `design.model.yield` | 0.821378 | reactor model at the design |
| `design.model.selectivity` | 1 | reactor model at the design |
| `design.cost.solve_seconds` | 0.0627037 | measured: mean of three solves |
| `report.decisions` | 0 |  |
| `report.decisions_without_reference` | 0 |  |
| `report.items` | 1 |  |

| check | status | detail |
|---|---|---|
| design.conversion | pass | margin 0.02138 at the design, computed by the model |
| design.hot_spot | pass | margin 0.3148 at the design, computed by the model |
| model.balances | pass | mole and energy balances of every tank closed to 1.4e-15 (relative to the feed and its heat flow) |
| model.stoichiometry | pass | the flow change lies in the span of the reactions (residual 3.8e-17 of the feed) |
| safety.stable_state | pass | every tank on a stable steady state (slope condition); steady states per tank: [1] |

## full

done in 27.3 s (commit d83a70bc, 2026-11-03).

| metric | value | how |
|---|---|---|
| `design.design.V` | 1.86415 | optimum, re-solved by the model |
| `design.design.UA` | 2588.22 | optimum, re-solved by the model |
| `design.model.X` | 0.800935 | reactor model at the design |
| `design.model.T_max` | 329.972 | reactor model at the design |
| `design.model.yield` | 0.800935 | reactor model at the design |
| `design.model.selectivity` | 1 | reactor model at the design |
| `design.cost.solve_seconds` | 0.0874055 | measured: mean of three solves |
| `report.decisions` | 0 |  |
| `report.decisions_without_reference` | 0 |  |
| `report.items` | 1 |  |

| check | status | detail |
|---|---|---|
| design.conversion | pass | margin 0.0009346 at the design, computed by the model |
| design.hot_spot | pass | margin 0.02775 at the design, computed by the model |
| model.balances | pass | mole and energy balances of every tank closed to 8.9e-15 (relative to the feed and its heat flow) |
| model.stoichiometry | pass | the flow change lies in the span of the reactions (residual 7.7e-18 of the feed) |
| safety.stable_state | pass | every tank on a stable steady state (slope condition); steady states per tank: [1] |
