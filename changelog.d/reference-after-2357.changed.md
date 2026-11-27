- Referência de módulos regerada (#2357, #2358): os dois PRs entraram com API pública nova — sensibilidade com
  caminhos posteriores, amostragem adaptativa, fidelidade por requisição, otimização robusta, e o laço de orçamento,
  o PINN de estudo e o `DesignPipeline` do design-study — sem rodar `scripts/reference/build.py`, o que deixou
  `docs/org/reference/` 4 arquivos atrás do código. `reference.yml --check` estava vermelho em `main` por isso.
