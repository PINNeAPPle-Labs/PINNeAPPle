# The AI Aerodynamicist: lower drag at 70 m/s

> Find a lower-drag aerodynamic design at 70 m/s. Preserve the required downforce, stay within the geometry constraints, and validate the final design using CFD.

## Plan

- Objective: min drag [N]
- Limits: downforce >= 1 x baseline; aero balance (front share) >= 0.9 x baseline and <= 1.1 x baseline
- Budget: 20 reference simulations = 1 baseline + 10 space-filling + 6 chosen one at a time + 3 final verifications
- Fixed conditions: speed_m_s = 70.0, rho_kg_m3 = 1.225, nu_m2_s = 1.5e-05, ground = moving road at the flow speed, model = half model, symmetry plane y = 0, turbulence = steady RANS k-omega SST, wall functions, wheels = static walls, solver = OpenFOAM simpleFoam
- Hypotheses to test:
  - the rear flap is the largest single source of drag that can be traded for downforce elsewhere
  - a steeper diffuser adds underfloor downforce at little drag until the flow separates from the ramp
  - a lower floor strengthens the ground effect (more downforce) within the clearance limit
  - front-wing angle moves the aero balance forward; rear-wing angle moves it back

## Solver qualification (baseline design, outside the budget)

| variant | cells | wall [s] | drag_N | downforce_N | balance_front |
|---|---|---|---|---|---|
| medium | 212869 | 428 | 3031 (-8.8 %) | 4327 (-43.7 %) | 0.8939 (-3.4 %) |
| campaign | 424842 | 679 | 3315 (-0.2 %) | 7038 (-8.4 %) | 1.015 (+9.6 %) |
| fine | 818620 | 1529 | 3322 (+0.0 %) | 7685 (+0.0 %) | 0.9258 (+0.0 %) |

Differences are relative to the `fine` variant; the campaign uses `campaign`.

## Budget

20 of 20 reference simulations, 11840 s of solver wall time, 0 failed.

## Surrogates (k-fold cross-validation on the final data)

| output | family | RMSE | rel. error | R² | 95 % coverage | fit [s] |
|---|---|---|---|---|---|---|
| drag_N | gp | 105.2 | 2.66 % | 0.699 | 65 % | 0.06 |
| drag_N | mlp_ensemble **(chosen)** | 90.27 | 2.35 % | 0.778 | 94 % | 0.29 |
| drag_N | quadratic | skipped: a full quadratic in 6 variables has 28 coefficients, more than the simulations | | | | |
| downforce_N | gp | 941.9 | 11.10 % | 0.126 | 65 % | 0.08 |
| downforce_N | mlp_ensemble **(chosen)** | 829.8 | 10.21 % | 0.322 | 82 % | 0.32 |
| downforce_N | quadratic | skipped: a full quadratic in 6 variables has 28 coefficients, more than the simulations | | | | |
| balance_front | gp | 0.0801 | 5.94 % | -0.060 | 53 % | 0.07 |
| balance_front | mlp_ensemble **(chosen)** | 0.05655 | 4.31 % | 0.472 | 82 % | 0.34 |
| balance_front | quadratic | skipped: a full quadratic in 6 variables has 28 coefficients, more than the simulations | | | | |

Surface cp surrogate: POD (12 modes, 94.7 % of the variance) + gp, 5-fold held-out RMSE per region: all 0.098, front_wing 0.235, rear_wing 0.207, floor 0.108, body 0.070, diffuser 0.116

## Decisions (each prediction made before its simulation)

| run | judged | rule | predicted -> CFD | after |
|---|---|---|---|---|
| run012-acquisition | out_of_domain | best prediction is out of domain | drag_N 2894 -> 2870 (0.8 %); downforce_N 7279 -> 7338 (0.8 %); balance_front 1.051 -> 1.1 (4.5 %) | verified |
| run013-acquisition | out_of_domain | best prediction is out of domain | drag_N 2843 -> 2879 (1.2 %); downforce_N 7357 -> 7390 (0.4 %); balance_front 0.9754 -> 0.9559 (2.0 %) | verified |
| run014-acquisition | uncertain | resolve uncertainty where the decision changes | drag_N 2859 -> 2897 (1.3 %); downforce_N 7291 -> 7811 (6.7 %); balance_front 1.007 -> 1.017 (1.0 %) | verified |
| run015-acquisition | uncertain | resolve uncertainty where the decision changes | drag_N 2869 -> 2830 (1.4 %); downforce_N 7223 -> 6947 (4.0 %); balance_front 1.112 -> 1.104 (0.7 %) | rejected by reference |
| run016-acquisition | uncertain | resolve uncertainty where the decision changes | drag_N 2849 -> 3036 (6.1 %); downforce_N 7256 -> 6777 (7.1 %); balance_front 1.067 -> 1.111 (4.0 %) | rejected by reference |
| run017-acquisition | uncertain | resolve uncertainty where the decision changes | drag_N 2790 -> 2902 (3.9 %); downforce_N 7672 -> 7384 (3.9 %); balance_front 1.102 -> 1.163 (5.2 %) | rejected by reference |
| run018-verification | favourable | final verification | drag_N 2875 -> 2884 (0.3 %); downforce_N 7328 -> 7257 (1.0 %); balance_front 1.078 -> 1.078 (0.1 %) | verified |
| run019-verification | favourable | final verification | drag_N 2898 -> 2795 (3.7 %); downforce_N 7985 -> 7319 (9.1 %); balance_front 1.097 -> 1.073 (2.3 %) | verified |
| run020-verification | favourable | final verification | drag_N 2975 -> 2867 (3.8 %); downforce_N 7950 -> 7566 (5.1 %); balance_front 1.09 -> 1.055 (3.3 %) | verified |

## Recommendation

best simulated design run019-verification: drag_N 2795 N vs baseline 3315 (-15.7 %), every limit met in the reference simulation

- drag_N: baseline 3315 -> 2795 (-15.67 %)
- downforce_N: baseline 7038 -> 7319 (limit: downforce >= 7038.38 N)
- balance_front: baseline 1.015 -> 1.073 (limit: 0.913191 <= aero balance (front share) <= 1.11612)

## The recommendation on the fine mesh (outside the budget)

Both designs on the fine mesh (803,259 cells for the recommendation). Noise = standard deviation of the output over the averaged iterations of each fine run.

| output | baseline (fine) | recommended (fine) | change (fine) | change / noise | change (campaign mesh) |
|---|---|---|---|---|---|
| drag_N | 3322 ± 65 | 2859 ± 53 | -13.95 % | -5.5 | -15.67 % |
| downforce_N | 7685 ± 2.8e+02 | 7745 ± 2.5e+02 | +0.79 % | +0.2 | +3.98 % |
| balance_front | 0.9258 ± 0.032 | 1.029 ± 0.016 | +11.18 % | +2.9 | +5.72 % |

On the fine mesh the balance limit (aero balance (front share) >= 0.9 x baseline and <= 1.1 x baseline) of the fine baseline is 0.8332 to 1.018; the recommendation's 1.029 is outside it. The campaign mesh moves the baseline's balance by +9.6 % against the fine mesh, so the balance limit is the least resolved of the three outputs.

## Limit margins against the solver's own iteration noise

margin = distance to the limit in the CFD (negative: violated); noise = standard deviation of the output over the averaged iterations of that run. A verdict with |margin / noise| < 2 is within the solver's noise.

| run | output | margin | noise | margin / noise | verdict |
|---|---|---|---|---|---|
| run012-acquisition | downforce_N | +299.1 | 187 | +1.6 | verified |
| run012-acquisition | balance_front | +0.01604 | 0.0197 | +0.8 | verified |
| run013-acquisition | downforce_N | +351.2 | 133 | +2.6 | verified |
| run013-acquisition | balance_front | +0.04269 | 0.0106 | +4.0 | verified |
| run014-acquisition | downforce_N | +772.1 | 337 | +2.3 | verified |
| run014-acquisition | balance_front | +0.09866 | 0.0575 | +1.7 | verified |
| run015-acquisition | downforce_N | -91.82 | 95.1 | -1.0 | rejected by reference |
| run015-acquisition | balance_front | +0.01185 | 0.00699 | +1.7 | rejected by reference |
| run016-acquisition | downforce_N | -261.9 | 164 | -1.6 | rejected by reference |
| run016-acquisition | balance_front | +0.004753 | 0.0166 | +0.3 | rejected by reference |
| run017-acquisition | downforce_N | +346.1 | 54.2 | +6.4 | rejected by reference |
| run017-acquisition | balance_front | -0.04705 | 0.0199 | -2.4 | rejected by reference |
| run018-verification | downforce_N | +218.7 | 41.4 | +5.3 | verified |
| run018-verification | balance_front | +0.0377 | 0.00474 | +8.0 | verified |
| run019-verification | downforce_N | +280.5 | 149 | +1.9 | verified |
| run019-verification | balance_front | +0.04347 | 0.0188 | +2.3 | verified |
| run020-verification | downforce_N | +527.3 | 116 | +4.5 | verified |
| run020-verification | balance_front | +0.06063 | 0.0151 | +4.0 | verified |

![pareto](figures/pareto.png)

![models](figures/models.png)

![learning](figures/learning.png)

![abc](figures/trust_abc.png)

![cp](figures/cp_numerical_prediction_error.png)

![baseline_cp_top](figures/baseline_cp_top.png)

![baseline_cp_bottom](figures/baseline_cp_bottom.png)

![baseline_streamlines](figures/baseline_streamlines.png)

![baseline_cp_paraview](figures/baseline_cp_paraview.png)

![baseline_wireframe](figures/baseline_wireframe.png)

![baseline_midplane_cp](figures/baseline_midplane_cp.png)

![baseline_wake_vorticity](figures/baseline_wake_vorticity.png)

![baseline_cycles_streamlines](figures/baseline_cycles_streamlines.png)

![baseline_cycles_cp](figures/baseline_cycles_cp.png)

![best_cp_top](figures/best_cp_top.png)

![best_cp_bottom](figures/best_cp_bottom.png)

![best_streamlines](figures/best_streamlines.png)

![best_cp_paraview](figures/best_cp_paraview.png)

![best_wireframe](figures/best_wireframe.png)

![best_midplane_cp](figures/best_midplane_cp.png)

![best_wake_vorticity](figures/best_wake_vorticity.png)

![best_cycles_streamlines](figures/best_cycles_streamlines.png)

![best_cycles_cp](figures/best_cycles_cp.png)

![dashboard](figures/dashboard.png)

## Environment

```json
{
 "platform": "Linux-6.18.44-fc-v153-x86_64-with-glibc2.39",
 "python": "3.13.17",
 "cpu_count": 4,
 "machine": "x86_64",
 "cpu": "Intel(R) Xeon(R) Processor @ 2.10GHz",
 "mem_gb": 15.7,
 "numpy": "2.5.3",
 "scipy": "1.18.1",
 "sklearn": "1.9.1",
 "torch": "2.14.1+cu130",
 "pinneapple": "0.7.0",
 "git_commit": "13ee053288fb01ed9170eee8484d576946037ce0"
}
```
