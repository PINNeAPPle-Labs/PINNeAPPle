# Relatório de confiança — heat_wall/simulation

**Decisão:** APPROVED  
**Score:** 1.00  ·  **Cobertura:** 100%

| Checagem | Status | Crítica | Valor | Limite | Fonte |
|---|---|---|---|---|---|
| cold_face_temperature: the solver against the series-resistance closed form | supports | sim | 2e-06 | 0.05 | — |
| cold_face_temperature: change under refinement | supports | sim | 1e-06 | 0.02 | — |
| the closed form returns a finite far-face temperature | supports | não | 572.952 | — | — |
| the solution is finite | supports | não | 572.953 | — | — |
| energy balance of the discrete system | supports | não | -9.07652e-12 | 1e-08 | — |
| the two-dimensional solve is within the family's tolerance of the closed form | supports | não | 2e-06 | 0.05 | — |
