# Computational lithography: ILT mask correction over the process window

Partially coherent imaging (Abbe / Hopkins SOCS), threshold resist, inverse lithography through the exact model or a learned operator when its measured cost pays off, and the mask re-verified by the full Abbe model over focus and dose (pinneapple_physics.lithography.design).

Flow: `pinneapple_flow/flows/lithography_opc.json` (as the example flow `industrial/lithography_opc`). Run it with `python -m pinneapple_flow example-flows run industrial/lithography_opc [--quick]`; regenerate this folder with `python -m pinneapple_flow example-flows gallery industrial --entries lithography_opc`.

Quick configuration (parameter overrides per node): `{}`.

Every number below was measured by a run of the flow on the commit and date stated; none is typed in.

## quick

done in 32.7 s (commit b5c5838, 2026-11-03).

| metric | value | how |
|---|---|---|
| `opc.epe.uncorrected_nm` | 9.95804 | full Abbe image, best focus and dose |
| `opc.epe.corrected_nm` | 0.363636 | full Abbe image, best focus and dose |
| `opc.pw.worst_epe_nm` | 1.83217 |  |
| `opc.pw.dof_nm` | 100 |  |
| `opc.mask.min_feature_nm` | 16 |  |
| `opc.cost.image_seconds` | 0.0154313 | measured: one exact image |
| `report.decisions` | 0 |  |
| `report.decisions_without_reference` | 0 |  |
| `report.items` | 1 |  |

| check | status | detail |
|---|---|---|
| litho.epe_nominal | pass | mean EPE 0.36 nm at best focus and dose (uncorrected mask: 9.96 nm), full Abbe image |
| litho.process_window | pass | worst mean EPE 1.83 nm over focus [-50.0, 0.0, 50.0] nm x dose [0.95, 1.0, 1.05] (uncorrected: 16.52 nm) |
| litho.improves | pass | the corrected mask prints closer to the target than the target itself used as the mask |
| mask.min_feature | pass | smallest mask feature or space 16.0 nm, rule 16 nm |
| model.kernels_vs_abbe | pass | largest intensity difference between the 0.995-energy kernels and the full Abbe sum on the mask: 1.12e-03 |

## full

done in 27.2 s (commit d83a70bc, 2026-11-03).

| metric | value | how |
|---|---|---|
| `opc.cost.image_seconds` | 0.00729325 | measured: one exact image |
| `opc.epe.corrected_nm` | 0.363636 | full Abbe image, best focus and dose |
| `opc.epe.uncorrected_nm` | 9.95804 | full Abbe image, best focus and dose |
| `opc.mask.min_feature_nm` | 16 |  |
| `opc.pw.dof_nm` | 100 |  |
| `opc.pw.worst_epe_nm` | 1.83217 |  |
| `report.decisions` | 0 |  |
| `report.decisions_without_reference` | 0 |  |
| `report.items` | 1 |  |

| check | status | detail |
|---|---|---|
| litho.epe_nominal | pass | mean EPE 0.36 nm at best focus and dose (uncorrected mask: 9.96 nm), full Abbe image |
| litho.process_window | pass | worst mean EPE 1.83 nm over focus [-50.0, 0.0, 50.0] nm x dose [0.95, 1.0, 1.05] (uncorrected: 16.52 nm) |
| litho.improves | pass | the corrected mask prints closer to the target than the target itself used as the mask |
| mask.min_feature | pass | smallest mask feature or space 16.0 nm, rule 16 nm |
| model.kernels_vs_abbe | pass | largest intensity difference between the 0.995-energy kernels and the full Abbe sum on the mask: 1.12e-03 |
