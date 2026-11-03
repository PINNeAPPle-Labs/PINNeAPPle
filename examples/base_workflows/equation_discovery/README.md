# Equation discovery: sparse regression checked on held-out data

SINDy (ODEs) or PDE-FIND (PDEs): the model chosen on held-out data, bootstrap inclusion and intervals, the discovered system simulated against data it never fitted (pinneapple_analysis.inverse_problems.discovery).

Flow: `pinneapple_flow/flows/equation_discovery.json` (as the example flow `base_workflows/equation_discovery`). Run it with `python -m pinneapple_flow example-flows run base_workflows/equation_discovery [--quick]`; regenerate this folder with `python -m pinneapple_flow example-flows gallery base_workflows --entries equation_discovery`.

Quick configuration (parameter overrides per node): `{"discover": {"n_bootstrap": 20}}`.

Every number below was measured by a run of the flow on the commit and date stated; none is typed in.

## quick

done in 3.1 s (commit d83a70bc, 2026-11-03).

| metric | value | how |
|---|---|---|
| `discover.savgol_window` | 41 | chosen by the held-out rollout |
| `report.decisions` | 0 |  |
| `report.decisions_without_reference` | 0 |  |
| `report.items` | 1 |  |

| check | status | detail |
|---|---|---|
| discover.rollout | pass | the discovered system integrated from the held-out window's first state over 0.998 time units: RMS error 0.99% of the signal (allowed 10%) |
| discover.stable_terms | pass | every selected term is kept in at least 80 % of the bootstrap refits |

## full

done in 3.1 s (commit d83a70bc, 2026-11-03).

| metric | value | how |
|---|---|---|
| `discover.savgol_window` | 41 | chosen by the held-out rollout |
| `report.decisions` | 0 |  |
| `report.decisions_without_reference` | 0 |  |
| `report.items` | 1 |  |

| check | status | detail |
|---|---|---|
| discover.rollout | pass | the discovered system integrated from the held-out window's first state over 0.998 time units: RMS error 0.99% of the signal (allowed 10%) |
| discover.stable_terms | pass | every selected term is kept in at least 80 % of the bootstrap refits |
