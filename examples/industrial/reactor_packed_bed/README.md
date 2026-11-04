# Reactor sizing: cooled packed bed (o-xylene to phthalic anhydride)

The smallest reactor meeting the conversion, hot-spot and runaway limits: the reactor model, a learned model only when its measured cost pays off, optimisation, and the design re-verified by the model (pinneapple_physics.reactors.design).

Flow: `pinneapple_flow/flows/reactor_sizing_packed_bed.json` (as the example flow `industrial/reactor_packed_bed`). Run it with `python -m pinneapple_flow example-flows run industrial/reactor_packed_bed [--quick]`; regenerate this folder with `python -m pinneapple_flow example-flows gallery industrial --entries reactor_packed_bed`.

Quick configuration (parameter overrides per node): `{"design": {"maxiter": 20, "popsize": 8}}`.

Every number below was measured by a run of the flow on the commit and date stated; none is typed in.

## quick

done in 6.9 s (commit d83a70bc, 2026-11-03).

| metric | value | how |
|---|---|---|
| `design.design.V` | 0.00109553 | optimum, re-solved by the model |
| `design.design.Tc` | 645.124 | optimum, re-solved by the model |
| `design.model.X` | 0.750671 | reactor model at the design |
| `design.model.T_max` | 672.262 | reactor model at the design |
| `design.model.P_out` | 101325 | reactor model at the design |
| `design.model.yield` | 0.750671 | reactor model at the design |
| `design.model.selectivity` | 1 | reactor model at the design |
| `design.cost.solve_seconds` | 0.0106866 | measured: mean of three solves |
| `design.safety.runaway_margin_K` | 14 | Morbidelli-Varma boundary minus the coolant |
| `report.decisions` | 0 |  |
| `report.decisions_without_reference` | 0 |  |
| `report.items` | 1 |  |

| check | status | detail |
|---|---|---|
| design.conversion | pass | margin 0.0006709 at the design, computed by the model |
| design.hot_spot | pass | margin 0.8885 at the design, computed by the model |
| model.converged | pass | conversion changes by 3.5e-09 with a 100x tighter tolerance |
| model.stoichiometry | pass | the flow change lies in the span of the reactions (residual 9.3e-17 of the feed) |
| model.levenspiel | pass | area under F_A0/(-r_A) vs X is 0.0010955 m^3 for a volume of 0.0010955 m^3 (9.2e-06) |
| safety.semenov | inconclusive | psi = 0.539 against 1/e: above the zeroth-order bound, which is conservative for a reaction that consumes its reactant; the sensitivity margin decides |
| safety.runaway_margin | pass | coolant 14.0 K below the runaway boundary (Morbidelli-Varma), required 10 K |

## full

done in 16.2 s (commit d83a70bc, 2026-11-03).

| metric | value | how |
|---|---|---|
| `design.design.V` | 0.00107254 | optimum, re-solved by the model |
| `design.design.Tc` | 645.468 | optimum, re-solved by the model |
| `design.model.X` | 0.75 | reactor model at the design |
| `design.model.T_max` | 673.15 | reactor model at the design |
| `design.model.P_out` | 101325 | reactor model at the design |
| `design.model.yield` | 0.75 | reactor model at the design |
| `design.model.selectivity` | 1 | reactor model at the design |
| `design.cost.solve_seconds` | 0.0120638 | measured: mean of three solves |
| `design.safety.runaway_margin_K` | 14 | Morbidelli-Varma boundary minus the coolant |
| `report.decisions` | 0 |  |
| `report.decisions_without_reference` | 0 |  |
| `report.items` | 1 |  |

| check | status | detail |
|---|---|---|
| design.conversion | pass | margin 8.309e-08 at the design, computed by the model |
| design.hot_spot | pass | margin 2.631e-05 at the design, computed by the model |
| model.converged | pass | conversion changes by 4.5e-09 with a 100x tighter tolerance |
| model.stoichiometry | pass | the flow change lies in the span of the reactions (residual 2.0e-17 of the feed) |
| model.levenspiel | pass | area under F_A0/(-r_A) vs X is 0.0010725 m^3 for a volume of 0.0010725 m^3 (9.2e-06) |
| safety.semenov | inconclusive | psi = 0.545 against 1/e: above the zeroth-order bound, which is conservative for a reaction that consumes its reactant; the sensitivity margin decides |
| safety.runaway_margin | pass | coolant 14.0 K below the runaway boundary (Morbidelli-Varma), required 10 K |
