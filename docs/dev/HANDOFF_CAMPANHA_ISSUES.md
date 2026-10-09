# Retomada: campanha de issues do PINNeAPPle (série X)

Documento de passagem de bastão. Escrito em 2026-10-09 para retomar o trabalho depois, por qualquer sessão.

## O que foi pedido

Passar pelas issues abertas do repositório `PINNeAPPle-Labs/PINNeAPPle`, uma por vez, na ordem, só as sem responsável ou atribuídas a `barrosyan`. Para cada uma: implementar, testar, commitar, dar push, abrir PR, mesclar e comentar na issue.

Regras do dono (valem sempre):

- Commits, PRs, issues e comentários saem **só em nome de `barrosyan`** (git: `Yan Barros <yanbarrosyan@gmail.com>`), **sem nenhuma menção a Claude ou IA**. Isso vale mesmo que alguma ferramenta sugira linhas de atribuição.
- Sem pressa; uma de cada vez, para garantir que está correta.
- Mesclar os próprios PRs depois do CI verde, comentar na issue e fechá-la.
- O dono reclamou de uso alto de CPU: nada de testes longos ou muitos workers. Rodar subconjuntos e com 1 thread quando der.

## Fluxo usado por issue (funcionou 11 vezes)

1. `gh issue view N`, ler o "Done when" (critério de pronto) e responder exatamente a ele.
2. `gh issue comment N` avisando que vai começar e o plano (a própria issue pede isso).
3. `git checkout -b feat/xN-...` a partir de `main` atualizada.
4. Implementar em `pinneapple_core/` (ou onde fizer sentido), com testes em `tests/pinneapple_core/`.
5. Documentar em `docs/core_concepts/` e acrescentar entrada no `CHANGELOG.md` (seção Unreleased).
6. Se o módulo for novo e visível em `pp.*`, registrar em `pinneapple/__init__.py` (`_SUBMODULES` ou `_LAZY_ATTRS`) e em `pyproject.toml` (lista `packages` e sdist, só para pacotes de topo).
7. `git commit` (inglês, sem menção a IA), `git push -u`, `gh pr create` com seção "Done-when check" e "Limits".
8. `gh pr checks N --watch --interval 20` (rodar com timeout alto ou em segundo plano), `gh pr merge N --merge --delete-branch`.
9. `gh issue comment N` com resultado e ressalvas honestas. A issue fecha sozinha pelo "Closes #N".

## Ambiente

- Não existe `python`; usar `python3` e `PYTHONPATH=.` (o repositório não é instalado). Há `.venv` no repositório.
- Testes do que foi feito: `PYTHONPATH=. python3 -m pytest tests/pinneapple_core tests/test_packaging_lists.py -q`.
- jax está instalado neste computador; os testes de jax usam `importorskip`.
- `tests/test_breadth_six_packages.py` dá **segfault** neste computador, com e sem as mudanças (pré-existente, não é nosso).
- Evitar `sleep` encadeado; esperar CI com `gh pr checks N --watch`.

## Estado: série X (feita)

| Issue | Item | PR | Onde | Resultado / ressalva |
|---|---|---|---|---|
| #187 | X3 | #290 | `pinneapple_core` (`domain`, `geometry`, `mesh`, `field`) | Field/Mesh/Domain/Geometry com gradient/divergence/interpolate/integrate |
| #188 | X4 | #291 | `operators.py` | grad, div, curl, laplacian, jacobian, hessian, integrate, flux nas 4 representações; gradiente em nuvem passou a usar ajuste quadrático local |
| #189 | X5 | #300 | `func.py`, `fem.py` | `pp.func` com `wrt=`, `implicit_solve`, FEM P1 diferenciável; gradiente de geometria bate com diferenças finitas |
| #190 | X6 | #301 | `pinneapple_physics/solving.py` | `kind` por método, `Solution.metadata()`, backends `fem` e `external`; corrigido bug (`ctx` ignorado no `pinn`). FNO/SPH/LBM/FVM **não** entraram |
| #191 | X7 | #302 | `backend.py` | PhysicsBackend torch+jax; achado: `torch.linspace` não é diferenciável nos extremos |
| #192 | X8 | #303 | `module.py` | PhysicsModule, Sequential, SolverModule, Hybrid; híbrido FEM grosseiro + correção neural (erro 0,31 → 0,10) |
| #193 | X9 | #304 | `pinneapple_physics/compile_api.py` | `pp.compile` com reuso de derivadas: **ganho modesto, 1,05x a 1,29x**. Forward-mode e Hessiana em lote foram medidos e **não** ajudaram |
| #194 | X10 | #307 | `loss.py` | `pp.loss` + `Balancer` (13 estratégias existentes + currículo); exemplo SA-PINN reproduz o original exatamente |
| #195 | X11 | #308 | `optim.py` | `PhysicsOptimizer` com 6 métodos; mesmo problema restrito nos 6 |
| #196 | X12 | #309 | `data.py` | DataLoader + 6 amostradores; PINN, operador e problema inverso na mesma API |
| #197 | X13 | #310 | `transforms.py` | Scale, Nondimensionalize, Coordinate, Symmetry, Periodic, FourierFeatures + proveniência |

Páginas de documentação criadas/estendidas: `docs/core_concepts/core_primitives.md`, `physics_optimizer.md`, `backend.md`, `model.md`, `pinn.md`, `solver.md`, `training_pipeline.md`, `problem_definition.md`.

## Em andamento: X14 (#198), `pp.distributed`

Branch **`feat/x14-distributed`** (já no GitHub, commit WIP, **sem PR**).

Já escrito: `pinneapple_core/distributed.py` (map/sweep/ensemble com joblib/loky e fallback, `partition_mesh` com halo e `map_mesh`, `decompose_box`, `train_decomposed` com Schwarz aditivo), registro em `pinneapple/__init__.py` e `pinneapple_core/__init__.py`, e `tests/pinneapple_core/test_distributed.py`.

O que falta:

1. **Validar os testes** com poucos workers. A suíte passou de 10 minutos e foi interrompida a pedido do dono (CPU). Limitar `workers=` (por exemplo 2) e o sweep de 64 casos (`dist.sweep(..., workers=64)` vira `min(cpu_count, ...)` ou reduzir o tamanho do problema).
2. Resultado já medido do Schwarz: 2 subdomínios, sobreposição 0,4, 8 iterações, erro relativo L2 de 0,2% em 2,8 s; com 4 subdomínios converge devagar (informação anda um subdomínio por iteração).
3. Critério de pronto da issue: "um treino com decomposição de domínio e um sweep de 64 simulações rodam pelo módulo". Falta confirmar os dois com os testes leves.
4. Documentar (`docs/core_concepts/`), `CHANGELOG`, abrir PR, mesclar, comentar e fechar a #198. Dependência opcional: `joblib` (considerar extra em `pyproject.toml`).

## Pendente

- **Série X restante: #199 a #227** (X15 a X40), cerca de 29 issues. Ver a lista com `gh issue list --state open --limit 400`. Algumas apoiam-se no que já existe (por exemplo X16 registries, X21 verificação dimensional, X32 controle MPC).
- **Experimentos com dados abertos: #228 a #289** (clima, CFD, aeroespacial). Muitos dependem de GPU ou de baixar dados grandes; combinar antes de começar.
- **Backlog antigo: #6 a #186** (itens B, A, D, E, F, G...), cerca de 140 issues.
- Total aberto em 2026-10-09: **229** (fechadas: 41).

## Decisões e lições (para não repetir)

- Pacote novo `pinneapple_core` concentra a infraestrutura (X3 em diante). Submódulos que importam torch são carregados sob demanda (`__getattr__` em `pinneapple_core/__init__.py`).
- Verificar contra um **caso de referência numérico** (solução manufaturada, diferenças finitas, tabela clássica) e confirmar que os testes falham no código antigo.
- Ser honesto no PR e na issue sobre o que ficou de fora e sobre ganhos modestos.
- O `compile_problem` mantém `LossWeights` próprio; `pp.loss` é a "porta de entrada" sobre os balanceadores já existentes.

## Como retomar (exemplos de pedido)

- "Termine o X14: reduza os workers dos testes de `test_distributed.py`, valide, documente e abra o PR."
- "Continue a série X a partir da #199, uma por vez, seguindo `docs/dev/HANDOFF_CAMPANHA_ISSUES.md`."

## Campanha paralela: contribuições em código aberto de terceiros

Tudo (catálogo, ferramentas, dados, estado dos PRs, regras, memória do assistente) está em `docs/dev/oss_contribuicoes/` (comece por `README.md` e `OSS_CAMPAIGN.md`).
