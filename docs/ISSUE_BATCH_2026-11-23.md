# Lote 23/11/2026 — estrutura operacional

**Fonte:** `docs/EXECUTION_PLAN_2026.md` (espelhado nos dois repositórios).
**Meta nominal:** 69 issues PINNeAPPle + 22 Engineering API = 91. A meta não autoriza fechamento sem evidência.

## Preparar sem executar testes ou GitHub Actions

Gerar o manifesto nominal sem acesso à rede:

```bash
python scripts/issue_batch_2026.py --date 2026-11-23 --offline
```

Reconciliar as 91 issues reais (somente GET, não modifica GitHub):

```bash
GH_TOKEN=<token> python scripts/issue_batch_2026.py --date 2026-11-23
```

O token é opcional para issues públicas, mas necessário para repositórios privados.
Os relatórios são gerados localmente em `.local/issue-batches/2026-11-23.json` e `.md`.
**Não commitar tokens nem logs sensíveis.** O script não cria PRs, não dispara workflows,
não fecha issues e não executa testes. Se o GitHub estiver indisponível, registra
`unavailable` e retorna código 2; não inventa estados.

## Agrupamento inicial — requer triagem do conteúdo completo

1. **Contratos de validação e evidência**: começar por PINNeAPPle #31, #38, #40;
   examinar interoperabilidade com VeriPhysics e os registros de execução.
2. **Pipelines e modelos**: PINNeAPPle #33, #34, #35, #52; separar tarefas de
   arquitetura de experimentos que exigem resultados numéricos medidos.
3. **Usabilidade e erros**: PINNeAPPle #46; inventariar mensagens antes de refatorar.
4. **Engineering API**: #27 (itens remanescentes do Studio), #28 (controle/naval),
   #29 (laboratório numérico), #30 (atlas físico), #31 (wizard), #32-#34 (Engineer).
   Confirmar PINNeAPPle_COMMIT, schemas e dependências antes de qualquer UI.
5. **Demais issues**: usar o manifesto completo; estas prioridades não excluem
   nenhuma das 91. Classificar com dados reais e separar P0/P1, S/M/L e bloqueios.

## Evidência mínima por issue

- Issue real, estado atual, labels, título, critérios de aceite, dependências e PRs.
- Módulos alterados e escopo preciso; relação N:M entre PRs e issues.
- Evidências de implementação e revisão; testes **não executados** devem permanecer
  explicitamente como `not_run`, nunca como `passed`.
- Benchmarks: referência física, unidades, seeds, métricas e limites de validade.
- `completed` apenas após revisão, merge e validação efetiva.
- `partial` ou `blocked` com próxima ação explícita em qualquer outro caso.

## Continuação

A estrutura deste lote é **preparatória**. Ela não declara 91 issues implementadas.
As PRs de produto abertas em paralelo precisam ser relacionadas aos critérios de
aceite de cada issue antes de qualquer fechamento.
