# Relatório de confiança — cantilever_beam/simulation

**Decisão:** APPROVED  
**Score:** 1.00  ·  **Cobertura:** 100%

| Checagem | Status | Crítica | Valor | Limite | Fonte |
|---|---|---|---|---|---|
| tip_deflection: the solver against the Euler-Bernoulli closed form (plane strain) | supports | sim | 0.001279 | 0.05 | — |
| tip_deflection: change under refinement | supports | sim | 0.005368 | 0.02 | — |
| the beam is slender enough for Euler-Bernoulli | supports | sim | 25 | 10 | — |
| the solution is finite | supports | não | 0.000135243 | — | — |
| the root reaction balances the applied load | supports | não | 500 | -500 | — |
| the solver is within the family's tolerance of the closed form | supports | não | 0.001279 | 0.05 | — |
