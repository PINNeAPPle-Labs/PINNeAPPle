# Relatório de confiança — PINNeAPPle Autonomous Physics Lab - rover mission

**Decisão:** REVIEW  
**Score:** 0.92  ·  **Cobertura:** 100%

| Checagem | Status | Crítica | Valor | Limite | Fonte |
|---|---|---|---|---|---|
| goal_reached | supports | sim | 1 | 1 | multibody run of the final route |
| slip_within_limit | supports | sim | 0.214524 | 0.4 | 2-s mean front-wheel slip after the 2-s start-up, final run |
| energy_covers_climb | supports | não | 2869.08 | 1364.86 | drive energy (sum of max(tau omega, 0)) vs m g x total climb of the final run (no regeneration; kinetic energy at 0.2 m/s is negligible) |
| time_step_converged | supports | não | 0.00616627 | 0.05 | first seconds of the final route at dt = 0.004 s and dt/2 |
| planner_vs_multibody_energy | supports | não | 0.130846 | 0.25 | quasi-static planner (learned belief) vs multibody, final route |
| planner_vs_multibody_slip | contradicts | não | 0.0540689 | 0.05 | RMS of predicted slip - 2-s mean front-wheel slip along the final route |
| replanned_route_risk | supports | não | 0 | 0.05 | share of belief samples that exceed the slip limit |
| failed_route_risk_learned | supports | não | 1 | 0.05 | first route re-evaluated with the learned belief |
| learned_soil_predictive | supports | não | 0.0355344 | 0.2 | held-out last quarter of the stall unit's measurements |
| learned_soil_calibrated | supports | não | 1.98296 | 3 | known truth of the synthetic site: max |error| / posterior sd |
| learned_soil_closer_to_truth | supports | não | — | — | known truth: |error| vs prior's |
| models_used_in_validity | supports | não | — | — | planner uses the analytic Wong-Reece law; zoo surrogate only compared |

## Motivos

- planner_vs_multibody_slip: falhou
