# Ahmed body: CFD to surrogate twin

The Ahmed body from geometry (parametric and OpenCASCADE CAD) through an OpenFOAM campaign (k-omega SST, prism layers, checks per run), grid convergence and validation against Ahmed et al. (1984), a dataset with a held-out region, surrogate families with calibrated intervals and physics checks, the slant angle inferred from a drag measurement, the least-drag slant and clearance re-verified against the CFD (or stated unverified), and a digital twin page.

Flow: `pinneapple_flow/flows/ahmed_body_end_to_end.json` (as the example flow `industrial/ahmed_design`). Run it with `python -m pinneapple_flow example-flows run industrial/ahmed_design [--quick]`; regenerate this folder with `python -m pinneapple_flow example-flows gallery industrial --entries ahmed_design`.

Quick configuration (parameter overrides per node): `{"geometry": {"cad": false}, "models": {"budget_s": 3.0}}`.

Every number below was measured by a run of the flow on the commit and date stated; none is typed in.

## quick

done in 184.5 s (commit d83a70bc, 2026-11-03).

| metric | value | how |
|---|---|---|
| `geometry.volume_m3` | 0.10959 | parametric body, divergence theorem on its triangles |
| `cfd.runs_ok` | 84 | run records with ok = true |
| `cfd.cpu_hours` | 48.825 | wall time x processes, summed over the runs |
| `inverse.slant_median_deg` | 18.5684 | median of 4000 MDN posterior samples |
| `report.cad_vs_parametric_cd_rel` | 0.0127671 | CD of the imported CAD body over the parametric body, baseline, L1 mesh, OpenFOAM |
| `report.cd_baseline_L0` | 0.289 | OpenFOAM, 84223 cells, window mean of iterations 500-1000 |
| `report.cd_baseline_L1` | 0.296565 | OpenFOAM, 207080 cells, window mean of iterations 500-1000 |
| `report.cd_baseline_L2` | 0.270744 | OpenFOAM, 534472 cells, window mean of iterations 500-1000 |
| `report.cd_baseline_L3` | 0.263979 | OpenFOAM, 1398073 cells, window mean of iterations 500-1000 |
| `report.gci_cd_finest` | 0.0109835 | grid-convergence index of CD on the finest mesh (Celik 2008) |
| `report.validation_cd_rel_error_median` | 0.109131 | median |CD_cfd / CD_measured - 1| over the six slant angles of Ahmed et al. (1984), L1 mesh |
| `report.validation_cd_rel_error_max` | 0.475055 | largest of the same |
| `report.doe_discrepancy` | 0.00392649 | centred L2 discrepancy of the 48-point maximin LHS |
| `report.doe_discrepancy_random` | 0.0141119 | median over 20 random designs of 48 points |
| `report.CD_mlp_ensemble_test_rmse` | 0.00580779 | mlp_ensemble on the test split, fitted on train + val, measured once after selection |
| `report.CD_mlp_ensemble_test_coverage90` | 0.9 | share of test cases inside the cross-conformal 90 % interval |
| `report.CD_mlp_ensemble_extrapolation_rmse` | 0.0549058 | mlp_ensemble on the extrapolation split, fitted on train + val, measured once after selection |
| `report.CD_mlp_ensemble_extrapolation_coverage90` | 0.7 | share of extrapolation cases inside the cross-conformal 90 % interval |
| `report.CL_gp_test_rmse` | 0.0204383 | gp on the test split, fitted on train + val, measured once after selection |
| `report.CL_gp_test_coverage90` | 0.9 | share of test cases inside the cross-conformal 90 % interval |
| `report.CL_gp_extrapolation_rmse` | 0.184066 | gp on the extrapolation split, fitted on train + val, measured once after selection |
| `report.CL_gp_extrapolation_coverage90` | 0.7 | share of extrapolation cases inside the cross-conformal 90 % interval |
| `report.CD_pressure_mlp_ensemble_test_rmse` | 0.00996185 | mlp_ensemble on the test split, fitted on train + val, measured once after selection |
| `report.CD_pressure_mlp_ensemble_test_coverage90` | 0.8 | share of test cases inside the cross-conformal 90 % interval |
| `report.CD_pressure_mlp_ensemble_extrapolation_rmse` | 0.0615752 | mlp_ensemble on the extrapolation split, fitted on train + val, measured once after selection |
| `report.CD_pressure_mlp_ensemble_extrapolation_coverage90` | 0.9 | share of extrapolation cases inside the cross-conformal 90 % interval |
| `report.CD_friction_mlp_ensemble_test_rmse` | 0.00102236 | mlp_ensemble on the test split, fitted on train + val, measured once after selection |
| `report.CD_friction_mlp_ensemble_test_coverage90` | 0.8 | share of test cases inside the cross-conformal 90 % interval |
| `report.CD_friction_mlp_ensemble_extrapolation_rmse` | 0.00113136 | mlp_ensemble on the extrapolation split, fitted on train + val, measured once after selection |
| `report.CD_friction_mlp_ensemble_extrapolation_coverage90` | 1 | share of extrapolation cases inside the cross-conformal 90 % interval |
| `report.cp_pod_gp_test_rel_l2_mean` | 0.0986457 | pod_gp on the test split, fitted on train + val, measured once after selection |
| `report.cp_pod_gp_test_coverage90` | 0.892176 | share of test cases inside the cross-conformal 90 % interval |
| `report.cp_pod_gp_extrapolation_rel_l2_mean` | 0.199554 | pod_gp on the extrapolation split, fitted on train + val, measured once after selection |
| `report.cp_pod_gp_extrapolation_coverage90` | 0.833638 | share of extrapolation cases inside the cross-conformal 90 % interval |
| `report.sobol_total_CD_slant_deg` | 0.961736 | total Sobol index on the selected surrogate, Saltelli n=4096 |
| `report.sobol_total_CD_clearance` | 0.0532791 | total Sobol index on the selected surrogate, Saltelli n=4096 |
| `report.sobol_total_CD_speed` | 0.0144963 | total Sobol index on the selected surrogate, Saltelli n=4096 |
| `report.sobol_total_CL_slant_deg` | 0.942418 | total Sobol index on the selected surrogate, Saltelli n=4096 |
| `report.sobol_total_CL_clearance` | 0.0604931 | total Sobol index on the selected surrogate, Saltelli n=4096 |
| `report.sobol_total_CL_speed` | 6.89179e-06 | total Sobol index on the selected surrogate, Saltelli n=4096 |
| `report.sobol_total_CD_friction_slant_deg` | 0.0889582 | total Sobol index on the selected surrogate, Saltelli n=4096 |
| `report.sobol_total_CD_friction_clearance` | 0.0390279 | total Sobol index on the selected surrogate, Saltelli n=4096 |
| `report.sobol_total_CD_friction_speed` | 0.929848 | total Sobol index on the selected surrogate, Saltelli n=4096 |
| `report.pareto_pick0_cd_abs_error` | 0.00101337 | |CD CFD - CD predicted| at a Pareto pick, OpenFOAM L1 |
| `grid.CD_error_0deg` | 0.0770807 | (CFD - measured) / measured, Ahmed 1984 |
| `grid.CD_error_12.5deg` | 0.111856 | (CFD - measured) / measured, Ahmed 1984 |
| `grid.CD_error_25deg` | 0.0430798 | (CFD - measured) / measured, Ahmed 1984 |
| `grid.CD_error_30deg` | -0.106407 | (CFD - measured) / measured, Ahmed 1984 |
| `grid.CD_error_35deg` | 0.475055 | (CFD - measured) / measured, Ahmed 1984 |
| `grid.CD_error_40deg` | 0.202044 | (CFD - measured) / measured, Ahmed 1984 |
| `grid.gci_CD_fine` | 0.0109835 | grid convergence index of the finest mesh, Celik et al. 2008 |
| `design.design.slant_deg` | 9.78139 |  |
| `design.design.clearance` | 0.0300279 |  |
| `design.design.speed` | 40 |  |
| `design.predicted.CD` | 0.250797 | mlp_ensemble refitted on 73 CFD runs |
| `design.predicted.CD_half_width_90` | 0.00409647 | mlp_ensemble refitted on 73 CFD runs |
| `design.predicted.CL` | -0.0859743 | gp refitted on 73 CFD runs |
| `design.predicted.CL_half_width_90` | 0.151051 | gp refitted on 73 CFD runs |
| `design.baseline.CD` | 0.290901 | 25 deg slant, 0.05 m clearance, same models |

| check | status | detail |
|---|---|---|
| design.outside_drag_crisis | pass | slant 9.78 deg outside the 28-32 deg drag-crisis band |
| design.no_extrapolation | pass | inside the sampled box (slant <= 34 deg, clearance 0.03-0.08 m) |
| verify.cfd | inconclusive | unverified: no OpenFOAM here and the nearest recorded run (pareto_00) is 0.084 away in the unit cube (> 0.05); run the design case |

## full

done in 346.3 s (commit d83a70bc, 2026-11-03).

| metric | value | how |
|---|---|---|
| `geometry.volume_m3` | 0.10959 | parametric body, divergence theorem on its triangles |
| `geometry.cad_volume_m3` | 0.109932 | imported STL of the OpenCASCADE body |
| `cfd.runs_ok` | 84 | run records with ok = true |
| `cfd.cpu_hours` | 48.825 | wall time x processes, summed over the runs |
| `inverse.slant_median_deg` | 18.5079 | median of 4000 MDN posterior samples |
| `report.cad_vs_parametric_cd_rel` | 0.0127671 | CD of the imported CAD body over the parametric body, baseline, L1 mesh, OpenFOAM |
| `report.cd_baseline_L0` | 0.289 | OpenFOAM, 84223 cells, window mean of iterations 500-1000 |
| `report.cd_baseline_L1` | 0.296565 | OpenFOAM, 207080 cells, window mean of iterations 500-1000 |
| `report.cd_baseline_L2` | 0.270744 | OpenFOAM, 534472 cells, window mean of iterations 500-1000 |
| `report.cd_baseline_L3` | 0.263979 | OpenFOAM, 1398073 cells, window mean of iterations 500-1000 |
| `report.gci_cd_finest` | 0.0109835 | grid-convergence index of CD on the finest mesh (Celik 2008) |
| `report.validation_cd_rel_error_median` | 0.109131 | median |CD_cfd / CD_measured - 1| over the six slant angles of Ahmed et al. (1984), L1 mesh |
| `report.validation_cd_rel_error_max` | 0.475055 | largest of the same |
| `report.doe_discrepancy` | 0.00392649 | centred L2 discrepancy of the 48-point maximin LHS |
| `report.doe_discrepancy_random` | 0.0141119 | median over 20 random designs of 48 points |
| `report.CD_mlp_ensemble_test_rmse` | 0.00580779 | mlp_ensemble on the test split, fitted on train + val, measured once after selection |
| `report.CD_mlp_ensemble_test_coverage90` | 0.9 | share of test cases inside the cross-conformal 90 % interval |
| `report.CD_mlp_ensemble_extrapolation_rmse` | 0.0549058 | mlp_ensemble on the extrapolation split, fitted on train + val, measured once after selection |
| `report.CD_mlp_ensemble_extrapolation_coverage90` | 0.7 | share of extrapolation cases inside the cross-conformal 90 % interval |
| `report.CL_gp_test_rmse` | 0.0204383 | gp on the test split, fitted on train + val, measured once after selection |
| `report.CL_gp_test_coverage90` | 0.9 | share of test cases inside the cross-conformal 90 % interval |
| `report.CL_gp_extrapolation_rmse` | 0.184066 | gp on the extrapolation split, fitted on train + val, measured once after selection |
| `report.CL_gp_extrapolation_coverage90` | 0.7 | share of extrapolation cases inside the cross-conformal 90 % interval |
| `report.CD_pressure_mlp_ensemble_test_rmse` | 0.00996185 | mlp_ensemble on the test split, fitted on train + val, measured once after selection |
| `report.CD_pressure_mlp_ensemble_test_coverage90` | 0.8 | share of test cases inside the cross-conformal 90 % interval |
| `report.CD_pressure_mlp_ensemble_extrapolation_rmse` | 0.0615752 | mlp_ensemble on the extrapolation split, fitted on train + val, measured once after selection |
| `report.CD_pressure_mlp_ensemble_extrapolation_coverage90` | 0.9 | share of extrapolation cases inside the cross-conformal 90 % interval |
| `report.CD_friction_mlp_ensemble_test_rmse` | 0.00102236 | mlp_ensemble on the test split, fitted on train + val, measured once after selection |
| `report.CD_friction_mlp_ensemble_test_coverage90` | 0.8 | share of test cases inside the cross-conformal 90 % interval |
| `report.CD_friction_mlp_ensemble_extrapolation_rmse` | 0.00113136 | mlp_ensemble on the extrapolation split, fitted on train + val, measured once after selection |
| `report.CD_friction_mlp_ensemble_extrapolation_coverage90` | 1 | share of extrapolation cases inside the cross-conformal 90 % interval |
| `report.cp_pod_gp_test_rel_l2_mean` | 0.0986457 | pod_gp on the test split, fitted on train + val, measured once after selection |
| `report.cp_pod_gp_test_coverage90` | 0.892176 | share of test cases inside the cross-conformal 90 % interval |
| `report.cp_pod_gp_extrapolation_rel_l2_mean` | 0.199554 | pod_gp on the extrapolation split, fitted on train + val, measured once after selection |
| `report.cp_pod_gp_extrapolation_coverage90` | 0.833638 | share of extrapolation cases inside the cross-conformal 90 % interval |
| `report.sobol_total_CD_slant_deg` | 0.961736 | total Sobol index on the selected surrogate, Saltelli n=4096 |
| `report.sobol_total_CD_clearance` | 0.0532791 | total Sobol index on the selected surrogate, Saltelli n=4096 |
| `report.sobol_total_CD_speed` | 0.0144963 | total Sobol index on the selected surrogate, Saltelli n=4096 |
| `report.sobol_total_CL_slant_deg` | 0.942418 | total Sobol index on the selected surrogate, Saltelli n=4096 |
| `report.sobol_total_CL_clearance` | 0.0604931 | total Sobol index on the selected surrogate, Saltelli n=4096 |
| `report.sobol_total_CL_speed` | 6.89179e-06 | total Sobol index on the selected surrogate, Saltelli n=4096 |
| `report.sobol_total_CD_friction_slant_deg` | 0.0889582 | total Sobol index on the selected surrogate, Saltelli n=4096 |
| `report.sobol_total_CD_friction_clearance` | 0.0390279 | total Sobol index on the selected surrogate, Saltelli n=4096 |
| `report.sobol_total_CD_friction_speed` | 0.929848 | total Sobol index on the selected surrogate, Saltelli n=4096 |
| `report.pareto_pick0_cd_abs_error` | 0.00101337 | |CD CFD - CD predicted| at a Pareto pick, OpenFOAM L1 |
| `grid.CD_error_0deg` | 0.0770807 | (CFD - measured) / measured, Ahmed 1984 |
| `grid.CD_error_12.5deg` | 0.111856 | (CFD - measured) / measured, Ahmed 1984 |
| `grid.CD_error_25deg` | 0.0430798 | (CFD - measured) / measured, Ahmed 1984 |
| `grid.CD_error_30deg` | -0.106407 | (CFD - measured) / measured, Ahmed 1984 |
| `grid.CD_error_35deg` | 0.475055 | (CFD - measured) / measured, Ahmed 1984 |
| `grid.CD_error_40deg` | 0.202044 | (CFD - measured) / measured, Ahmed 1984 |
| `grid.gci_CD_fine` | 0.0109835 | grid convergence index of the finest mesh, Celik et al. 2008 |
| `design.design.slant_deg` | 9.78139 |  |
| `design.design.clearance` | 0.0300279 |  |
| `design.design.speed` | 40 |  |
| `design.predicted.CD` | 0.250797 | mlp_ensemble refitted on 73 CFD runs |
| `design.predicted.CD_half_width_90` | 0.00409647 | mlp_ensemble refitted on 73 CFD runs |
| `design.predicted.CL` | -0.0859743 | gp refitted on 73 CFD runs |
| `design.predicted.CL_half_width_90` | 0.151051 | gp refitted on 73 CFD runs |
| `design.baseline.CD` | 0.290901 | 25 deg slant, 0.05 m clearance, same models |

| check | status | detail |
|---|---|---|
| design.outside_drag_crisis | pass | slant 9.78 deg outside the 28-32 deg drag-crisis band |
| design.no_extrapolation | pass | inside the sampled box (slant <= 34 deg, clearance 0.03-0.08 m) |
| verify.cfd | inconclusive | unverified: no OpenFOAM here and the nearest recorded run (pareto_00) is 0.084 away in the unit cube (> 0.05); run the design case |
