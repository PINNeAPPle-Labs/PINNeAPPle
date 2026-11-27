# Relatório de confiança — laminar_pipe/simulation

**Decisão:** APPROVED  
**Score:** 1.00  ·  **Cobertura:** 100%

| Checagem | Status | Crítica | Valor | Limite | Fonte |
|---|---|---|---|---|---|
| pressure_drop: the solver against the Hagen-Poiseuille closed form | supports | sim | 2.3e-05 | 0.05 | — |
| pressure_drop: change under refinement | supports | sim | 0.006621 | 0.02 | — |
| the flow is laminar, where the closed form applies | supports | sim | 529.6 | 2300 | — |
| the discrete profile is positive everywhere | supports | não | 3.9269e-06 | — | — |
| the finite-difference solve is within the family's tolerance of the closed form | supports | não | 2.3e-05 | 0.05 | — |
