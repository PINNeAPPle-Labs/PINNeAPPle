# Multiphysics coupling: one-way, partitioned with Aitken, monolithic, checked

Two physics participants through named interface fields, coupled one-way, two-way partitioned (Gauss-Seidel with Aitken relaxation) and monolithically; the strategy from the measured coupling strength; convergence, interface balance and a closed-form coupled solution (pinneapple_simulation.multiphysics.workflow).

Flow: `pinneapple_flow/flows/multiphysics_coupling.json` (as the example flow `base_workflows/multiphysics_coupling`). Run it with `python -m pinneapple_flow example-flows run base_workflows/multiphysics_coupling [--quick]`; regenerate this folder with `python -m pinneapple_flow example-flows gallery base_workflows --entries multiphysics_coupling`.

Quick configuration (parameter overrides per node): `{}`.

Every number below was measured by a run of the flow on the commit and date stated; none is typed in.

## quick

done in 0.1 s (commit d83a70bc, 2026-11-03).

| metric | value | how |
|---|---|---|
| `couple.coupling.contraction` | 0.350626 | spectral radius of the plain partitioned map, measured from its first iterations |
| `couple.one_way_error.T_max_K` | 0.118843 | relative to the two-way solution |
| `couple.one_way_error.voltage_drop_V` | 0.106521 | relative to the two-way solution |
| `couple.one_way_error.resistance_ohm` | 0.106521 | relative to the two-way solution |
| `couple.one_way.iterations` | 1 | 0.014 ms |
| `couple.gauss_seidel.iterations` | 22 | 0.399 ms |
| `couple.gauss_seidel_aitken.iterations` | 9 | 0.233 ms |
| `couple.monolithic.iterations` | 408 | 44.8 ms |
| `report.decisions` | 0 |  |
| `report.decisions_without_reference` | 0 |  |
| `report.items` | 1 |  |

| check | status | detail |
|---|---|---|
| coupling.convergence | pass | Gauss-Seidel + Aitken: 9 iterations to an interface residual 3.7e-11 (<= 1e-10); plain Gauss-Seidel: 22 iterations |
| coupling.monolithic | pass | monolithic residual 6.2e-11 (from 4.5), 408 residual evaluations |
| coupling.partitioned_vs_monolithic | pass | largest relative difference of the quantities of interest 2.5e-11 (<= 1e-6) |
| interface.joule_power | pass | electric power I dV = 0.740261 W; heat received by the thermal model 0.740261 W; relative imbalance 4.1e-11 (<= 0.001) |
| energy.global | pass | heat conducted out of the ends 0.740293 W against I dV = 0.740261 W; relative imbalance 4.3e-05 (<= 0.001) |
| closed_form | pass | two-way against T - T0 = (cos(m(x - L/2))/cos(mL/2) - 1)/alpha, dV = rho0 J (2/m) tan(mL/2): field error 1.1e-05, T_max_K 3.7e-06, voltage_drop_V 3.8e-06, resistance_ohm 3.8e-06 (<= 0.001) |
| one_way_vs_two_way | info | one-way error against two-way: T_max_K 0.119, voltage_drop_V 0.107, resistance_ohm 0.107; one-way against the closed form: field error 0.357 |
| recommendation | info | gauss_seidel_aitken: two-way needed (one-way error 0.119); the plain partitioned map contracts (rho = 0.351 < 1), so Gauss-Seidel with Aitken relaxation converges and reuses each solver as is |

## full

done in 0.1 s (commit d83a70bc, 2026-11-03).

| metric | value | how |
|---|---|---|
| `couple.coupling.contraction` | 0.350626 | spectral radius of the plain partitioned map, measured from its first iterations |
| `couple.one_way_error.T_max_K` | 0.118843 | relative to the two-way solution |
| `couple.one_way_error.voltage_drop_V` | 0.106521 | relative to the two-way solution |
| `couple.one_way_error.resistance_ohm` | 0.106521 | relative to the two-way solution |
| `couple.one_way.iterations` | 1 | 0.0154 ms |
| `couple.gauss_seidel.iterations` | 22 | 0.43 ms |
| `couple.gauss_seidel_aitken.iterations` | 9 | 0.286 ms |
| `couple.monolithic.iterations` | 408 | 46 ms |
| `report.decisions` | 0 |  |
| `report.decisions_without_reference` | 0 |  |
| `report.items` | 1 |  |

| check | status | detail |
|---|---|---|
| coupling.convergence | pass | Gauss-Seidel + Aitken: 9 iterations to an interface residual 3.7e-11 (<= 1e-10); plain Gauss-Seidel: 22 iterations |
| coupling.monolithic | pass | monolithic residual 6.2e-11 (from 4.5), 408 residual evaluations |
| coupling.partitioned_vs_monolithic | pass | largest relative difference of the quantities of interest 2.5e-11 (<= 1e-6) |
| interface.joule_power | pass | electric power I dV = 0.740261 W; heat received by the thermal model 0.740261 W; relative imbalance 4.1e-11 (<= 0.001) |
| energy.global | pass | heat conducted out of the ends 0.740293 W against I dV = 0.740261 W; relative imbalance 4.3e-05 (<= 0.001) |
| closed_form | pass | two-way against T - T0 = (cos(m(x - L/2))/cos(mL/2) - 1)/alpha, dV = rho0 J (2/m) tan(mL/2): field error 1.1e-05, T_max_K 3.7e-06, voltage_drop_V 3.8e-06, resistance_ohm 3.8e-06 (<= 0.001) |
| one_way_vs_two_way | info | one-way error against two-way: T_max_K 0.119, voltage_drop_V 0.107, resistance_ohm 0.107; one-way against the closed form: field error 0.357 |
| recommendation | info | gauss_seidel_aitken: two-way needed (one-way error 0.119); the plain partitioned map contracts (rho = 0.351 < 1), so Gauss-Seidel with Aitken relaxation converges and reuses each solver as is |
