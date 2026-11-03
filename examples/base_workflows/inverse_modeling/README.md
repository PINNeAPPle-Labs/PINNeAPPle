# Inverse modeling: identifiability, estimate, trust checks

Identifiability before the fit, estimation with held-out rows and intervals, and the checks that say whether the estimate can be trusted: fit, held-out, bounds, practical identifiability, white residuals, synthetic recovery (pinneapple_analysis.inverse_problems.workflow).

Flow: `pinneapple_flow/flows/inverse_modeling.json` (as the example flow `base_workflows/inverse_modeling`). Run it with `python -m pinneapple_flow example-flows run base_workflows/inverse_modeling [--quick]`; regenerate this folder with `python -m pinneapple_flow example-flows gallery base_workflows --entries inverse_modeling`.

Quick configuration (parameter overrides per node): `{"inverse": {"bootstrap": 20, "n_recovery": 4}}`.

Every number below was measured by a run of the flow on the commit and date stated; none is typed in.

## quick

done in 0.1 s (commit d83a70bc, 2026-11-03).

| metric | value | how |
|---|---|---|
| `inverse.estimate.A` | 5.4888e+06 | 95 % interval [3201244.5519733867, 9411012.405284008] |
| `inverse.estimate.Ea` | 60290.7 | 95 % interval [58698.9655650849, 61925.60850752983] |
| `inverse.next_measurement.T=320, tau=60` | 1.92853e-05 | D-optimal gain in log det F |
| `inverse.next_measurement.T=330, tau=60` | 1.66524e-05 | D-optimal gain in log det F |
| `inverse.next_measurement.T=310, tau=60` | 1.54207e-05 | D-optimal gain in log det F |
| `report.decisions` | 0 |  |
| `report.decisions_without_reference` | 0 |  |
| `report.items` | 1 |  |

| check | status | detail |
|---|---|---|
| identify.structural | pass | Fisher information rank 2 of 2 at the initial guess (condition number 1.32e+05) |
| identify.fits_data | pass | reduced chi-square 0.562 against the stated noise (<= 2) |
| identify.held_out | pass | held-out rows: mean (residual/sigma)^2 1.56 (<= 4) |
| identify.inside_bounds | pass | every estimate inside its bounds |
| identify.practical | pass | every parameter practically identifiable |
| identify.white_residuals | pass | runs test of the residual signs along T: p = 0.556 (>= 0.05: no structure left) |
| identify.recovery | pass | 4 synthetic refits at the estimate: the 95 % intervals cover it in A 100%, Ea 100% (>= 73% at that count) |

## full

done in 0.4 s (commit d83a70bc, 2026-11-03).

| metric | value | how |
|---|---|---|
| `inverse.estimate.A` | 5.4888e+06 | 95 % interval [3201244.5519733867, 9411012.405284008] |
| `inverse.estimate.Ea` | 60290.7 | 95 % interval [58698.9655650849, 61925.60850752983] |
| `inverse.next_measurement.T=320, tau=60` | 1.92853e-05 | D-optimal gain in log det F |
| `inverse.next_measurement.T=330, tau=60` | 1.66524e-05 | D-optimal gain in log det F |
| `inverse.next_measurement.T=310, tau=60` | 1.54207e-05 | D-optimal gain in log det F |
| `report.decisions` | 0 |  |
| `report.decisions_without_reference` | 0 |  |
| `report.items` | 1 |  |

| check | status | detail |
|---|---|---|
| identify.structural | pass | Fisher information rank 2 of 2 at the initial guess (condition number 1.32e+05) |
| identify.fits_data | pass | reduced chi-square 0.562 against the stated noise (<= 2) |
| identify.held_out | pass | held-out rows: mean (residual/sigma)^2 1.56 (<= 4) |
| identify.inside_bounds | pass | every estimate inside its bounds |
| identify.practical | pass | every parameter practically identifiable |
| identify.white_residuals | pass | runs test of the residual signs along T: p = 0.556 (>= 0.05: no structure left) |
| identify.recovery | pass | 10 synthetic refits at the estimate: the 95 % intervals cover it in A 100%, Ea 100% (>= 81% at that count) |
