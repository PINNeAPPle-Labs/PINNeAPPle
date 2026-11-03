# Design optimization: geometry and parameters, re-verified on refined meshes

Geometry dimensions and parameters of a tool (a gmsh-parametrized cantilever solved by plane-stress FEM by default) optimised under constraints; the optimum re-solved on refined meshes (Richardson, grid convergence index), the coarse-mesh error corrected, checks on convergence, constraints, sensitivities, robustness and the known optimum, and the STEP of the optimum (pinneapple_design.design_optimizer.workflow).

Flow: `pinneapple_flow/flows/design_optimization.json` (as the example flow `base_workflows/design_optimization`). Run it with `python -m pinneapple_flow example-flows run base_workflows/design_optimization [--quick]`; regenerate this folder with `python -m pinneapple_flow example-flows gallery base_workflows --entries design_optimization`.

Quick configuration (parameter overrides per node): `{}`.

Every number below was measured by a run of the flow on the commit and date stated; none is typed in.

## quick

done in 9.8 s (commit b5c5838, 2026-11-03).

| metric | value | how |
|---|---|---|
| `design.cost.solve_seconds` | 0.0216784 | measured: one tool call at the search mesh |
| `design.design.h_root` | 0.151249 | optimum |
| `design.design.h_tip` | 0.0504862 | optimum |
| `design.design.thickness` | 0.01 | optimum |
| `design.verified.mass_kg` | 7.91812 | Richardson over the meshes [8, 16, 32]; grid convergence index 0.0 |
| `design.verified.stress_max_Pa` | 1.47437e+08 | Richardson over the meshes [8, 16, 32]; grid convergence index 0.0016837805364138601 |
| `design.verified.tip_deflection_m` | 0.00589746 | Richardson over the meshes [8, 16, 32]; grid convergence index 0.0025981908465616886 |
| `design.robust.objective_cost` | 0.00861623 | relative objective paid for 95 % feasibility under the tolerances |
| `design.reference.objective` | 7.85 | SLSQP from 8 seeded starts on the reference model |
| `report.decisions` | 0 |  |
| `report.decisions_without_reference` | 0 |  |
| `report.items` | 1 |  |

| check | status | detail |
|---|---|---|
| surrogate.decision | pass | no surrogate: cost: 0.0216784 s per solve x 480 evaluations = 10.4056 s, within the direct budget of 600 s -> no learned model; call the solver directly. The model families below are the ones to reach for if the study grows. |
| optimizer.converged | pass | 2 starts agree to 0.000% (at most 1.0%); mesh correction after 2 search(es) changed by 0.008% |
| mesh.quality | pass | worst element over the search and verification meshes: SICN 0.509 (above 0.1) |
| mesh.convergence | pass | grid convergence index at the optimum, worst tip_deflection_m: 0.260% (at most 5%); observed orders mass_kg n/a, stress_max_Pa 1.95, tip_deflection_m 1.95 |
| mesh.coarse_error.mass_kg | info | search mesh 7.91812 vs converged 7.91812 (+0.00%); corrected in the search |
| mesh.coarse_error.stress_max_Pa | info | search mesh 1.3678e+08 vs converged 1.47437e+08 (-7.23%); corrected in the search |
| mesh.coarse_error.tip_deflection_m | info | search mesh 0.00523237 vs converged 0.00589746 (-11.28%); corrected in the search |
| constraints.stress_max_Pa | pass | stress_max_Pa <= 1.5e+08: verified 1.47437e+08 (mesh band ±0.17%); violation 0.000% (at most 0.5%) |
| constraints.tip_deflection_m | pass | tip_deflection_m <= 0.008: verified 0.00589746 (mesh band ±0.26%); violation 0.000% (at most 0.5%) |
| optimality.kkt | pass | stationarity residual 0.00% of /grad objective/ with active stress_max_Pa, thickness at lower bound; multipliers stress_max_Pa 2.686e-08, thickness >= 0.01 397 |
| robustness.tolerances | pass | 95.0% of 200 designs within the tolerances h_root ±0.0015, h_tip ±0.0005, thickness ±0.0001 satisfy every constraint (target 95%); back-off for 95 %: stress_max_Pa 2.505e+06, tip_deflection_m 0.0001401 |
| reference.known_optimum | pass | nominal mass_kg: 7.85047 vs the reference optimum 7.85 (0.01%, at most 2%); the reference model at that design violates its limits by 0.00% (at most 2%); largest design difference 1.33% of a range (optima need not be unique) |
| export.geometry | pass | optimum.step read back: volume 0.00100868 m3 vs the design's 0.00100868 m3 |

## full

done in 7.2 s (commit d83a70bc, 2026-11-03).

| metric | value | how |
|---|---|---|
| `design.cost.solve_seconds` | 0.00780194 | measured: one tool call at the search mesh |
| `design.design.h_root` | 0.151249 | optimum |
| `design.design.h_tip` | 0.0504862 | optimum |
| `design.design.thickness` | 0.01 | optimum |
| `design.reference.objective` | 7.85 | SLSQP from 8 seeded starts on the reference model |
| `design.robust.objective_cost` | 0.00861623 | relative objective paid for 95 % feasibility under the tolerances |
| `design.verified.mass_kg` | 7.91812 | Richardson over the meshes [8, 16, 32]; grid convergence index 0.0 |
| `design.verified.stress_max_Pa` | 1.47437e+08 | Richardson over the meshes [8, 16, 32]; grid convergence index 0.0016837805364138601 |
| `design.verified.tip_deflection_m` | 0.00589746 | Richardson over the meshes [8, 16, 32]; grid convergence index 0.0025981908465616886 |
| `report.decisions` | 0 |  |
| `report.decisions_without_reference` | 0 |  |
| `report.items` | 1 |  |

| check | status | detail |
|---|---|---|
| surrogate.decision | pass | no surrogate: cost: 0.00780194 s per solve x 480 evaluations = 3.74493 s, within the direct budget of 600 s -> no learned model; call the solver directly. The model families below are the ones to reach for if the study grows. |
| optimizer.converged | pass | 2 starts agree to 0.000% (at most 1.0%); mesh correction after 2 search(es) changed by 0.008% |
| mesh.quality | pass | worst element over the search and verification meshes: SICN 0.509 (above 0.1) |
| mesh.convergence | pass | grid convergence index at the optimum, worst tip_deflection_m: 0.260% (at most 5%); observed orders mass_kg n/a, stress_max_Pa 1.95, tip_deflection_m 1.95 |
| mesh.coarse_error.mass_kg | info | search mesh 7.91812 vs converged 7.91812 (+0.00%); corrected in the search |
| mesh.coarse_error.stress_max_Pa | info | search mesh 1.3678e+08 vs converged 1.47437e+08 (-7.23%); corrected in the search |
| mesh.coarse_error.tip_deflection_m | info | search mesh 0.00523237 vs converged 0.00589746 (-11.28%); corrected in the search |
| constraints.stress_max_Pa | pass | stress_max_Pa <= 1.5e+08: verified 1.47437e+08 (mesh band ±0.17%); violation 0.000% (at most 0.5%) |
| constraints.tip_deflection_m | pass | tip_deflection_m <= 0.008: verified 0.00589746 (mesh band ±0.26%); violation 0.000% (at most 0.5%) |
| optimality.kkt | pass | stationarity residual 0.00% of /grad objective/ with active stress_max_Pa, thickness at lower bound; multipliers stress_max_Pa 2.686e-08, thickness >= 0.01 397 |
| robustness.tolerances | pass | 95.0% of 200 designs within the tolerances h_root ±0.0015, h_tip ±0.0005, thickness ±0.0001 satisfy every constraint (target 95%); back-off for 95 %: stress_max_Pa 2.505e+06, tip_deflection_m 0.0001401 |
| reference.known_optimum | pass | nominal mass_kg: 7.85047 vs the reference optimum 7.85 (0.01%, at most 2%); the reference model at that design violates its limits by 0.00% (at most 2%); largest design difference 1.33% of a range (optima need not be unique) |
| export.geometry | pass | optimum.step read back: volume 0.00100868 m3 vs the design's 0.00100868 m3 |
