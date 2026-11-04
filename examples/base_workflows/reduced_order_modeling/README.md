# Reduced-order modeling: POD-Galerkin, DMD and operator inference checked against the full model

Snapshots of a full-order model (or a file), the POD rank from the singular-value energy, POD-Galerkin, DMD and operator inference on one protocol, checked on held-out parameters and on extrapolation in time, with stability, the invariant and the measured speed-up (pinneapple_analysis.reduced_order.workflow).

Flow: `pinneapple_flow/flows/reduced_order_modeling.json` (as the example flow `base_workflows/reduced_order_modeling`). Run it with `python -m pinneapple_flow example-flows run base_workflows/reduced_order_modeling [--quick]`; regenerate this folder with `python -m pinneapple_flow example-flows gallery base_workflows --entries reduced_order_modeling`.

Quick configuration (parameter overrides per node): `{"rom": {"repeats": 1}}`.

Every number below was measured by a run of the flow on the commit and date stated; none is typed in.

## quick

done in 1.7 s (commit ca74b8a, 2026-11-03).

| metric | value | how |
|---|---|---|
| `rom.rank` | 9 | POD modes 8 (+ the constant mode when the invariant is the mass); energy 0.99999006 |
| `rom.projection_error` | 0.0192619 | floor of any ROM in this basis on the held-out runs (interpolation window 0.00098) |
| `rom.pod_galerkin.interpolation_error` | 0.00112514 | held-out mu [0.0625, 0.0875, 0.11249999999999999, 0.1375], t <= 1.5 |
| `rom.pod_galerkin.extrapolation_error` | 0.0208757 | 1.5 < t <= 2 |
| `rom.pod_galerkin.speedup` | 6.91839 | online 9.08 ms, measured |
| `rom.dmd.interpolation_error` | 0.00294448 | held-out mu [0.0625, 0.0875, 0.11249999999999999, 0.1375], t <= 1.5 |
| `rom.dmd.extrapolation_error` | 0.0251276 | 1.5 < t <= 2 |
| `rom.dmd.speedup` | 90.5939 | online 0.592 ms, measured |
| `rom.opinf.interpolation_error` | 0.000980584 | held-out mu [0.0625, 0.0875, 0.11249999999999999, 0.1375], t <= 1.5 |
| `rom.opinf.extrapolation_error` | 0.0193097 | 1.5 < t <= 2 |
| `rom.opinf.speedup` | 5.192 | online 7.56 ms, measured |
| `report.decisions` | 0 |  |
| `report.decisions_without_reference` | 0 |  |
| `report.items` | 1 |  |

| check | status | detail |
|---|---|---|
| pod_galerkin.interpolation | pass | max rel. L2 error 0.00113 on mu = [0.0625, 0.0875, 0.1125, 0.1375], t <= 1.5 (<= 0.02; projection floor 0.00098) |
| pod_galerkin.extrapolation | pass | max rel. L2 error 0.0209 for 1.5 < t <= 2 (<= 0.05; projection floor 0.019) |
| pod_galerkin.stability | pass | max Re(lambda) linearized about the uniform state, fluctuation modes = -0.08605; growth of the linearized model over the horizon 0.8419 (<= 1.01) |
| pod_galerkin.invariant | pass | mass drift 2.8e-16 (<= 1e-06) |
| pod_galerkin.speedup | pass | online 9.08 ms against the FOM: x6.92 (worst held-out parameter, best of 1) |
| dmd.interpolation | pass | max rel. L2 error 0.00294 on mu = [0.0625, 0.0875, 0.1125, 0.1375], t <= 1.5 (<= 0.02; projection floor 0.00098) |
| dmd.extrapolation | pass | max rel. L2 error 0.0251 for 1.5 < t <= 2 (<= 0.05; projection floor 0.019) |
| dmd.stability | fail | max /lambda/ (discrete) linearized about the uniform state, fluctuation modes = 1.022; growth of the linearized model over the horizon 76.68 (<= 1.01) |
| dmd.invariant | pass | mass drift 1.5e-12 (<= 1e-06) |
| dmd.speedup | pass | online 0.592 ms against the FOM: x90.6 (worst held-out parameter, best of 1) |
| opinf.interpolation | pass | max rel. L2 error 0.000981 on mu = [0.0625, 0.0875, 0.1125, 0.1375], t <= 1.5 (<= 0.02; projection floor 0.00098) |
| opinf.extrapolation | pass | max rel. L2 error 0.0193 for 1.5 < t <= 2 (<= 0.05; projection floor 0.019) |
| opinf.stability | fail | max Re(lambda) linearized about the uniform state, fluctuation modes = 0.3734; growth of the linearized model over the horizon 2.11 (<= 1.01) |
| opinf.invariant | pass | mass drift 1.5e-11 (<= 1e-06) |
| opinf.speedup | pass | online 7.56 ms against the FOM: x5.19 (worst held-out parameter, best of 1) |

## full

done in 3.4 s (commit d83a70bc, 2026-11-03).

| metric | value | how |
|---|---|---|
| `report.decisions` | 0 |  |
| `report.decisions_without_reference` | 0 |  |
| `report.items` | 1 |  |
| `rom.dmd.extrapolation_error` | 0.0251276 | 1.5 < t <= 2 |
| `rom.dmd.interpolation_error` | 0.00294448 | held-out mu [0.0625, 0.0875, 0.11249999999999999, 0.1375], t <= 1.5 |
| `rom.dmd.speedup` | 107.818 | online 0.399 ms, measured |
| `rom.opinf.extrapolation_error` | 0.0193097 | 1.5 < t <= 2 |
| `rom.opinf.interpolation_error` | 0.000980584 | held-out mu [0.0625, 0.0875, 0.11249999999999999, 0.1375], t <= 1.5 |
| `rom.opinf.speedup` | 4.65521 | online 8.69 ms, measured |
| `rom.pod_galerkin.extrapolation_error` | 0.0208757 | 1.5 < t <= 2 |
| `rom.pod_galerkin.interpolation_error` | 0.00112514 | held-out mu [0.0625, 0.0875, 0.11249999999999999, 0.1375], t <= 1.5 |
| `rom.pod_galerkin.speedup` | 3.4597 | online 8.85 ms, measured |
| `rom.projection_error` | 0.0192619 | floor of any ROM in this basis on the held-out runs (interpolation window 0.00098) |
| `rom.rank` | 9 | POD modes 8 (+ the constant mode when the invariant is the mass); energy 0.99999006 |

| check | status | detail |
|---|---|---|
| pod_galerkin.interpolation | pass | max rel. L2 error 0.00113 on mu = [0.0625, 0.0875, 0.1125, 0.1375], t <= 1.5 (<= 0.02; projection floor 0.00098) |
| pod_galerkin.extrapolation | pass | max rel. L2 error 0.0209 for 1.5 < t <= 2 (<= 0.05; projection floor 0.019) |
| pod_galerkin.stability | pass | max Re(lambda) linearized about the uniform state, fluctuation modes = -0.08605; growth of the linearized model over the horizon 0.8419 (<= 1.01) |
| pod_galerkin.invariant | pass | mass drift 2.8e-16 (<= 1e-06) |
| pod_galerkin.speedup | pass | online 8.85 ms against the FOM: x3.46 (worst held-out parameter, best of 3) |
| dmd.interpolation | pass | max rel. L2 error 0.00294 on mu = [0.0625, 0.0875, 0.1125, 0.1375], t <= 1.5 (<= 0.02; projection floor 0.00098) |
| dmd.extrapolation | pass | max rel. L2 error 0.0251 for 1.5 < t <= 2 (<= 0.05; projection floor 0.019) |
| dmd.stability | fail | max /lambda/ (discrete) linearized about the uniform state, fluctuation modes = 1.022; growth of the linearized model over the horizon 76.68 (<= 1.01) |
| dmd.invariant | pass | mass drift 1.5e-12 (<= 1e-06) |
| dmd.speedup | pass | online 0.399 ms against the FOM: x108 (worst held-out parameter, best of 3) |
| opinf.interpolation | pass | max rel. L2 error 0.000981 on mu = [0.0625, 0.0875, 0.1125, 0.1375], t <= 1.5 (<= 0.02; projection floor 0.00098) |
| opinf.extrapolation | pass | max rel. L2 error 0.0193 for 1.5 < t <= 2 (<= 0.05; projection floor 0.019) |
| opinf.stability | fail | max Re(lambda) linearized about the uniform state, fluctuation modes = 0.3734; growth of the linearized model over the horizon 2.11 (<= 1.01) |
| opinf.invariant | pass | mass drift 1.5e-11 (<= 1e-06) |
| opinf.speedup | pass | online 8.69 ms against the FOM: x4.66 (worst held-out parameter, best of 3) |
