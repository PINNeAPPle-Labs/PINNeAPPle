# Mapa completo do PINNeAPPle

Documento de referência, em português, de todo o repositório: 41 pacotes, 2.142 módulos, 2.757 classes e 6.130 funções públicas. Gerado a partir do inventário automático do commit `e53cdcfb04`.

Como o texto passa de 150 páginas, ele está dividido em volumes por camada. Este arquivo é o volume 0: visão geral, arquitetura e sumário.

## Volumes

| volume | conteúdo | pacotes |
|---|---|---|
| [Volume 1 — Núcleo, porta de entrada e catálogo](mapa_completo/v01_nucleo.md) | A porta de entrada `pp.*`, os primitivos do núcleo, o catálogo de métodos, o registro de artefatos e os shims de compatibilidade. | `pinneapple`, `pinneapple_core`, `pinneapple_catalog`, `pinneapple_registry`, `pinneapple_orchestration`, `pinneapple_models`, `pinneapple_train`, `pinneapple_solvers` |
| [Volume 2 — Física: problemas, equações e ciência](mapa_completo/v02_fisica.md) | Como um problema físico é descrito e resolvido (`PhysicalProblem`, `pp.solve`), as equações, os presets industriais, o desenho de problemas, os experimentos por área, a condensação e as consultas físicas. | `pinneapple_physics`, `pinneapple_presets`, `pinneapple_problemdesign`, `pinneapple_science`, `pinneapple_condense`, `pinneapple_query` |
| [Volume 3 — Simulação numérica](mapa_completo/v03_simulacao.md) | Solvers clássicos (FDM, FEM, FVM, espectral, SPH, LBM), pontes para solvers externos, eletromagnetismo, voo, multicorpos, veículos e rovers. | `pinneapple_simulation` |
| [Volume 4 — Aprendizado de máquina e dados](mapa_completo/v04_aprendizado.md) | Arquiteturas e treino de redes (PINN, operadores neurais, grafos), dados e amostragem, adaptação, modelos de mundo, percepção, computação quântica e a arena de benchmarks. | `pinneapple_neural`, `pinneapple_data`, `pinneapple_pdb`, `pinneapple_adaptation`, `pinneapple_arena`, `pinneapple_worldmodel`, `pinneapple_perception`, `pinneapple_quantum` |
| [Volume 5 — Confiança, validação e decisão](mapa_completo/v05_confianca.md) | A camada que confere cada resultado contra a física: validação, incerteza, inversão, o linter `pinneapple check`, registros de evidência, segurança, decisão e a cadeia de porquês. | `pinneapple_analysis`, `pinneapple_check`, `pinneapple_veriphysics`, `pinneapple_security`, `pinneapple_decision`, `pinneapple_reason` |
| [Volume 6 — Projeto e sistemas](mapa_completo/v06_projeto_sistemas.md) | Geometria, otimização de projeto, projeto a partir de intenção, sistemas dinâmicos, séries temporais, co-simulação e gêmeos digitais. | `pinneapple_design`, `pinneapple_systems`, `pinneapple_twin3d` |
| [Volume 7 — Ferramentas, orquestração e produtos](mapa_completo/v07_ferramentas_produtos.md) | Ferramentas (benchmarks, exportação, visualização, automação de GUIs de engenharia, NVIDIA), o Flow, o Hub, o Lab, os LLMs, o servidor MCP, o laboratório autônomo e o app web. | `pinneapple_tools`, `pinneapple_blender`, `pinneapple_flow`, `pinneapple_hub`, `pinneapple_lab`, `pinneapple_llm`, `pinneapple_mcp`, `pinneapple_autolab`, `pinneapple_app` |
| [Volume 8 — Apps, exemplos, reproduções, scripts e CI](mapa_completo/v08_apps_exemplos_ci.md) | 64 apps, 88 entradas de exemplos, reproduções de papers, scripts, benchmarks e 12 workflows de CI. | — |
| [Volume 9 — Índice alfabético, glossário e apêndice numérico](mapa_completo/v09_indice_glossario_apendice.md) | Índice de todas as classes e funções públicas, glossário e números. | — |

## Visão geral

### O que é o PINNeAPPle

O PINNeAPPle é uma biblioteca Python de IA para física. A ideia central tem duas partes:

1. **Uma descrição, muitos métodos.** O problema físico é descrito uma vez (`pp.PhysicalProblem`, ou o nome de um
   preset). Depois ele é resolvido por redes informadas pela física (PINN), operadores neurais, redes de grafos ou
   solvers clássicos (diferenças finitas, elementos finitos, pontes para OpenFOAM e CalculiX), sempre pela mesma
   chamada: `pp.solve(problema, método)`.
2. **Uma camada de confiança.** Antes de alguém usar um resultado, ele é conferido contra a física: resíduo da
   equação, conservação, condições de contorno, comparação com referências e domínio de validade. A saída é uma
   decisão explícita (`APPROVED` / `REVIEW` / `REJECT` no `pp.trust_card`; `pass` / `warn` / `fail` no
   `pp.validate`).

Em volta desse núcleo há camadas de projeto (geometria, otimização, projeto a partir de um texto), sistemas
(séries temporais, co-simulação, gêmeos digitais), orquestração (Flow, Lab, laboratório autônomo, LLMs, servidor
MCP) e produtos (Hub, apps web). O pacote de topo `pinneapple` (importado como `pp`) é uma camada fina: cada nome
dele delega para um dos pacotes `pinneapple_*`.

### Arquitetura em camadas

A tabela resume as camadas na ordem em que um problema passa por elas. Cada camada depende só da interface da
anterior (um `ProblemSpec`, coordenadas amostradas, a função de perda, um modelo treinado), e não de herança ou de
um objeto compartilhado. Por isso dá para trocar uma peça sem mexer nas outras.

| camada | pacotes principais | papel |
|---|---|---|
| Porta de entrada | `pinneapple` (`pp.*`), `pinneapple_core.onboarding` | `pp.doctor`, `pp.recommend`, `pp.card`, `pp.new`, `pp.solve`, `pp.compare`, `pp.validate` |
| Primitivos | `pinneapple_core` | `Field`, `Mesh`, `Domain`, `Geometry`, operadores diferenciais, motor de workflows |
| Problema | `pinneapple_physics` (`physical_problem`, `pde_environment`, `symbolic_pde`), `pinneapple_presets`, `pinneapple_problemdesign` | descrever o problema: equações, contornos, presets, desenho guiado |
| Solução clássica | `pinneapple_simulation`, `pinneapple_condense` | FDM, FEM, FVM, SPH, LBM, pontes externas, eletromagnetismo, voo, veículos, condensação |
| Aprendizado | `pinneapple_neural`, `pinneapple_data`, `pinneapple_adaptation`, `pinneapple_worldmodel`, `pinneapple_quantum` | arquiteturas, treino, dados, transferência, modelos de mundo |
| Confiança | `pinneapple_analysis`, `pinneapple_check`, `pinneapple_veriphysics`, `pinneapple_security` | validação, incerteza, inversão, linter de física em CI, registros de evidência, assinaturas |
| Decisão e raciocínio | `pinneapple_decision`, `pinneapple_reason`, `pinneapple_query` | qual experimento rodar a seguir, cadeia de porquês, consultas físicas com planejador |
| Projeto e sistemas | `pinneapple_design`, `pinneapple_systems`, `pinneapple_twin3d` | geometria, otimização, projeto por intenção, séries temporais, gêmeos digitais |
| Catálogo e registro | `pinneapple_catalog`, `pinneapple_registry`, `pinneapple_hub` | métodos com status de validação, recursos com licença, artefatos, zoo de modelos |
| Orquestração | `pinneapple_flow`, `pinneapple_lab`, `pinneapple_autolab`, `pinneapple_llm`, `pinneapple_mcp`, `pinneapple_orchestration` | grafos de nós, experimentos registrados, missões autônomas, agentes |
| Ferramentas e produtos | `pinneapple_tools`, `pinneapple_arena`, `pinneapple_blender`, `pinneapple_app`, `apps/` | benchmarks, exportação, visualização, automação de GUIs, apps web |

Os pacotes `pinneapple_models`, `pinneapple_train` e `pinneapple_solvers` são shims de compatibilidade: só
reexportam `pinneapple_neural.architectures`, `pinneapple_neural.trainer` e
`pinneapple_simulation.numerical_solvers`, para que caminhos de import antigos continuem funcionando.

### Como as peças se conectam

```mermaid
flowchart TD
    U["Usuário / agente<br/>texto, preset ou PhysicalProblem"] --> PP["pp.* (pinneapple)<br/>doctor · recommend · card · new"]
    PP --> PROB["Problema<br/>pinneapple_physics · presets · problemdesign"]
    PROB --> SOLVE{"pp.solve(problema, método)"}
    SOLVE -->|pinn| NEU["pinneapple_neural<br/>arquiteturas + trainer"]
    SOLVE -->|fem / reference / analytic| SIM["pinneapple_simulation<br/>solvers clássicos + pontes externas"]
    NEU --> DATA["pinneapple_data<br/>amostragem, datasets"]
    CORE["pinneapple_core<br/>Field · Mesh · Domain · operadores"] -.-> NEU
    CORE -.-> SIM
    SOLVE --> SOL["Solution"]
    SOL --> TRUST["Confiança<br/>pp.validate · pp.trust_card · pinneapple check<br/>pinneapple_analysis · veriphysics"]
    TRUST -->|APPROVED / pass| USE["Uso: projeto, gêmeo digital, relatório"]
    TRUST -->|REVIEW / REJECT| DEC["pinneapple_decision<br/>próximo experimento"]
    DEC --> SOLVE
    CAT["pinneapple_catalog<br/>status de validação dos métodos"] -.-> SOLVE
    FLOW["pinneapple_flow<br/>plan(texto) → grafo → run"] --> SOLVE
    HUB["pinneapple_hub<br/>zoo, datasets, biblioteca"] --> NEU
    LAB["pinneapple_lab<br/>experimentos registrados"] --> TRUST
    USE --> DES["pinneapple_design · pinneapple_systems"]
    APPS["apps/ · pinneapple_app · MCP · LLM · autolab"] --> FLOW
    APPS --> PP
```

Em palavras:

- **`pp.*`** é a porta de entrada. `pp.doctor` diz o que falta instalar, `pp.recommend` sugere preset, método e
  modelo a partir de uma frase, `pp.card` mostra o que um preset recebe e devolve, e `pp.new` cria um projeto do
  usuário fora do repositório.
- **`pp.solve` / `pp.compare`** (em `pinneapple_physics.solving`) recebem um `PhysicalProblem`, um `ProblemSpec`
  ou o nome de um preset e despacham para o método pedido (`"pinn"`, `"fem"`, `"analytic"`, `"reference"`,
  `"exact"`, `"external"`; `pp.list_methods()` lista os registrados). O resultado é um `Solution`.
- **Confiança**: `pp.validate` (em `pinneapple_analysis.validation.validate`) e `pp.trust_card` (em
  `pinneapple_analysis.trust.field_card`) conferem resultados e modelos; `pinneapple check` roda as mesmas
  verificações em CI.
- **`pinneapple_flow`** transforma um texto num grafo de nós tipados (`plan`), e o executa sem interface ou no
  Flow Studio (`run`).
- **`pinneapple_hub`** guarda e entrega modelos treinados e verificados (`pp.zoo`), datasets (`pp.datasets`) e uma
  biblioteca pesquisável de ferramentas, geometrias, fluxos e exemplos.
- **`pinneapple_catalog`** lista cada solver, método de treino, equação e problema com o código, as referências e
  o status de validação calculado a partir dos testes (`validated` / `tested` / `untested`).
- **Apps** (`apps/` e `pinneapple_app`) são interfaces web e ferramentas de demonstração montadas sobre a
  biblioteca; o servidor MCP (`pinneapple_mcp`) expõe as mesmas funções para agentes de código.

### A porta de entrada `pp.*`

`import pinneapple as pp` expõe 297 nomes (versão `0.7.0`). A tabela diz de onde cada um vem e o que faz.

| nome | como é carregado | origem | o que faz |
|---|---|---|---|
| `pp.ActiveLearningConfig` | import direto | `pinneapple_data.ActiveLearningConfig` | Configuração do active learning: estratégia, número de candidatos, de pontos escolhidos por rodada, iniciais, iterações, peso de exploração e semente. |
| `pp.AdaptiveCollocationTrainer` | import direto | `pinneapple_data.AdaptiveCollocationTrainer` | Envolve um PINN e uma função física, mantém o próprio conjunto de colocação e acrescenta pontos escolhidos pela estratégia de active learning durante o treino. |
| `pp.AFNO` | import direto | `pinneapple_neural.architectures.AFNO` | Operador AFNO completo para campos 2D `(B, H, W, C)`: projeção de entrada, camadas AFNO e projeção de saída. |
| `pp.annulus` | import direto | `pinneapple_design.geometry.annulus` | Anel fechado (tubo com tampas planas) em torno do eixo z. |
| `pp.arena` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_arena` | Pacote raiz: reexporta configuração, `Arena`, rankings, problemas, fábrica de modelos, treino, visualização, `define_problem`, `DatasetProblem`, presets de dataset, modo de decisão e NAS. |
| `pp.as_tensor` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_neural.utils.as_tensor` | Extrai o tensor de previsão de um Tensor, dicionário de campos, tupla ou objeto de saída (`PINNOutput`, `ModelOutput` etc.). |
| `pp.assess_drift` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_analysis.uncertainty.drift.assess_drift` | Checa um lote de predições quanto a perda de acurácia, má calibração, mudança de entrada e tendência do resíduo. |
| `pp.assimilation` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_analysis.data_assimilation` | Assimilação de dados: 4D-Var por autograd sobre os métodos de ensemble (EKI) e filtros de Kalman, mais reconstrução GP e posicionamento de sensores. |
| `pp.astro` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_physics.astro` | Pacote de astronomia posicional e astrofísica do observatório PINNeAPPle, em NumPy próprio e validado contra USNO NOVAS, JPL Horizons e skyfield; reexporta `Observer`, `Time`, `apparent` e afins. |
| `pp.auto_residual` | import direto | `pinneapple_physics.symbolic_pde.auto_residual` | Calcula valores do campo e derivadas nomeadas via autograd. |
| `pp.Backend` | import direto | `pinneapple_tools.compute_backends.Backend` | Namespace com as constantes de nome de backend (`TORCH`, `JAX`). |
| `pp.backend` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_tools.compute_backends` | Suporte multi-backend: reexporta o seletor PyTorch/JAX e os utilitários JAX. |
| `pp.batched_inference` | import direto | `pinneapple_neural.trainer.batched_inference` | Inferência em lotes de um modelo sobre um tensor grande (`pinneapple_neural.predictor.infer`), com tamanho de lote, dispositivo e AMP opcionais. |
| `pp.benchmark` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_analysis.benchmark` | Entrada do benchmark com orçamento igual (preparar -> validar -> rodar -> relatório); reexporta `run`, `prepare`, `load`, o registro e `ArrayProblem`. |
| `pp.BenchmarkConfig` | import direto | `pinneapple_tools.benchmark_suite.BenchmarkConfig` | Configuração do benchmark: número de pontos de colocação/contorno/inicial, épocas, taxa de aprendizado, pesos das perdas, dispositivo, semente e limiar de convergência. |
| `pp.BenchmarkResult` | import direto | `pinneapple_tools.benchmark_suite.BenchmarkResult` | Resultado de um par (problema, modelo): métricas, histórico, tempo, número de parâmetros e posição no ranking. |
| `pp.BenchmarkTaskBase` | import direto | `pinneapple_tools.benchmark_suite.BenchmarkTaskBase` | Base abstrata de tarefa física: amostragem de colocação, contorno e condição inicial, resíduo da EDP, grade de avaliação e montagem de batch; os métodos abstratos levantam `NotImplementedError`. |
| `pp.best_device` | import direto | `pinneapple_neural.trainer.best_device` | Retorna o melhor dispositivo disponível: CUDA, depois MPS (se permitido), depois CPU. |
| `pp.best_method` | import direto | `pinneapple_core.onboarding.catalog.best_method` | O método mais barato que roda hoje para o preset: analítico, senão solver de referência, senão rede treinada. |
| `pp.bodies` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_design.geometry.bodies` | Corpos fechados triangulados para escoamento externo: corpo de Ahmed, esfera, cilindro, caixa e anel. |
| `pp.BoxConstraint` | import direto | `pinneapple_design.design_optimizer.BoxConstraint` | Restrição de limites `theta_min <= theta <= theta_max` com barreira ReLU unilateral. |
| `pp.build_digital_twin` | import direto | `pinneapple_systems.digital_twin.build_digital_twin` | Fábrica de conveniência para `DigitalTwin` com as opções mais comuns. |
| `pp.build_model` | definido em `pinneapple/__init__.py` | `pinneapple.build_model` | Constrói um modelo pelo nome no `ModelRegistry` (`ModelRegistry.build(name, **kwargs)`). |
| `pp.CADToCFDPipeline` | import direto | `pinneapple_simulation.numerical_solvers.CADToCFDPipeline` | Pipeline completo: arquivo CAD → malha → solução NS → dados de treino PINN, com comparação contra um PINN. |
| `pp.cae` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_data.cae` | Camada de dados CAE: lê malhas e resultados de OpenFOAM, CalculiX/Abaqus, Gmsh, VTK, STL e tabelas de pontos num único tipo `Mesh`, com qualidade de malha, Courant e distância à parede. |
| `pp.card` | import direto | `pinneapple_core.onboarding.card` | Monta o `Card` de um preset; nome desconhecido levanta `KeyError` com sugestões próximas. |
| `pp.CauchyMisfit` | import direto | `pinneapple_analysis.inverse_problems.CauchyMisfit` | Desajuste de Cauchy/Lorentz, de cauda pesada e muito robusto. |
| `pp.CausalPINNTrainer` | import direto | `pinneapple_neural.trainer.CausalPINNTrainer` | Treinador PINN que aplica os pesos causais ao resíduo da EDP junto com perdas de condição inicial/contorno; permite consultar e plotar os pesos. |
| `pp.CausalWeightScheduler` | import direto | `pinneapple_neural.trainer.CausalWeightScheduler` | Calcula pesos causais por fatia de tempo, w_k = exp(-epsilon * soma das perdas das fatias anteriores), e a perda ponderada e uma métrica de violação de causalidade. |
| `pp.cfd` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_simulation.numerical_solvers.external_flow` | Escoamento externo em torno de qualquer corpo no OpenFOAM numa chamada: STL → snappyHexMesh → simpleFoam (k-omega SST) → forças, coeficientes de pele, linhas de corrente, cortes e cena 3D. Requer OpenFOAM instalado. |
| `pp.CFDMesh` | import direto | `pinneapple_simulation.numerical_solvers.CFDMesh` | Malha não estruturada para CFD, criada de STL, STEP ou CSG, com nós de contorno e plot 2D. |
| `pp.channel_with_hole` | import direto | `pinneapple_design.geometry.channel_with_hole` | Canal retangular com obstáculo circular. |
| `pp.ChannelDomain2D` | import direto | `pinneapple_design.geometry.ChannelDomain2D` | Canal retangular com entrada à esquerda e saída à direita. |
| `pp.ChannelWithObstacleDomain2D` | import direto | `pinneapple_design.geometry.ChannelWithObstacleDomain2D` | Canal retangular com obstáculo circular. |
| `pp.circle` | import direto | `pinneapple_design.geometry.circle` | Atalho: `SDF` de círculo. |
| `pp.closed_form` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_physics.closed_form` | Subpacote de modelos físicos em forma fechada (sem EDP), portados de ferramentas de produtos para serem compartilhados e testados; o `__init__` importa só parte dos módulos (não importa `eddy_coil`, `electromagnetics`, `heat_exchanger`, `pipe_insulation`, `pump_npsh`, `thin_airfoil`). |
| `pp.CollocationConfig` | import direto | `pinneapple_data.CollocationConfig` | Configuração da geração de colocação: estratégia e número de pontos interiores, de contorno e de condição inicial, entre outros. |
| `pp.CollocationSampler` | import direto | `pinneapple_data.CollocationSampler` | Gerador único de pontos de colocação: construtores `from_bounds`, `from_problem_spec`, `from_domain`, `from_mesh`, `from_stl`, e `sample`/`sample_adaptive` que devolvem o dict de batch PINN. |
| `pp.compare` | import direto | `pinneapple_physics.solving.compare` | Resolve ou reaproveita cada método e pontua contra a referência nos mesmos pontos. |
| `pp.Comparison` | import direto | `pinneapple_physics.solving.Comparison` | Tabela de `compare`: métricas de cada método contra a referência nos mesmos pontos, com `best`. |
| `pp.compile` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_physics.compile_api.compile` | Prepara um `PhysicalProblem`, `ProblemSpec` ou nome de preset para resolver mais rápido. |
| `pp.ComposedObsOperator` | import direto | `pinneapple_analysis.inverse_problems.ComposedObsOperator` | Empilha vários operadores de observação num vetor único. |
| `pp.CompositeObjective` | import direto | `pinneapple_design.design_optimizer.CompositeObjective` | Soma ponderada de vários objetivos. |
| `pp.CompositeRegularizer` | import direto | `pinneapple_analysis.inverse_problems.CompositeRegularizer` | Soma ponderada de vários regularizadores. |
| `pp.compute_isosurface` | import direto | `pinneapple_neural.predictor.compute_isosurface` | Isosuperfície de um campo 3D por marching cubes (exige `scikit-image`). |
| `pp.compute_pareto_front` | import direto | `pinneapple_design.design_optimizer.compute_pareto_front` | Máscara booleana das soluções Pareto-ótimas. |
| `pp.compute_streamlines` | import direto | `pinneapple_neural.predictor.compute_streamlines` | Linhas de corrente 2D por integração RK4. |
| `pp.ComputeBudget` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_analysis.cost.equal_budget.ComputeBudget` | Orçamento de treino em segundos, FLOPs e/ou passos (None = sem limite). |
| `pp.conformal_study` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_analysis.uncertainty.conformal_study.conformal_study` | Calibra em `(y_cal, pred_cal)` e mede a cobertura em `(y_test, pred_test)`. |
| `pp.ConformalPredictor` | import direto | `pinneapple_analysis.uncertainty.ConformalPredictor` | Preditor conformal split: calibra num conjunto reservado e dá intervalos com cobertura garantida. |
| `pp.ConstraintSet` | import direto | `pinneapple_design.design_optimizer.ConstraintSet` | Agrega várias restrições: soma de penalidades e checagem de todas. |
| `pp.ContinuousAdjointSolver` | import direto | `pinneapple_design.design_optimizer.ContinuousAdjointSolver` | Solver adjunto por diferenciação automática: resolve a equação adjunta, calcula a sensibilidade de forma e roda a otimização. |
| `pp.core` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_core` | Ponto de entrada do pacote: exporta `Domain`, `Geometry`, `Mesh`, `Field`, `FunctionField`, `Grid`, `PointCloud` e os operadores, e carrega `func`, `fem`, `loss`, `optim`, `data`, `transforms` etc. preguiçosamente no primeiro acesso. |
| `pp.count_gpus` | import direto | `pinneapple_neural.trainer.count_gpus` | Retorna o número de GPUs CUDA disponíveis. |
| `pp.csg_annulus` | import direto | `pinneapple_design.geometry.csg_annulus` | Alias de `annulus` em `pinneapple_design.geometry`: domínio CSG em forma de coroa circular. |
| `pp.CSGCircle` | import direto | `pinneapple_design.geometry.CSGCircle` | Círculo 2D. |
| `pp.CSGDifference` | import direto | `pinneapple_design.geometry.CSGDifference` | Diferença A menos B: max(sdf_a, -sdf_b). |
| `pp.CSGEllipse` | import direto | `pinneapple_design.geometry.CSGEllipse` | Elipse 2D alinhada aos eixos. |
| `pp.CSGIntersection` | import direto | `pinneapple_design.geometry.CSGIntersection` | Interseção booleana de duas formas: max das SDFs. |
| `pp.CSGPolygon` | import direto | `pinneapple_design.geometry.CSGPolygon` | Polígono qualquer dado por vértices ordenados. |
| `pp.CSGRectangle` | import direto | `pinneapple_design.geometry.CSGRectangle` | Retângulo alinhado aos eixos. |
| `pp.CSGUnion` | import direto | `pinneapple_design.geometry.CSGUnion` | União booleana de duas formas: min das SDFs. |
| `pp.curl` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_core.operators.curl` | Rotacional: vetor em 3D, escalar (componente z) em 2D. |
| `pp.dashboard` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_tools.visualization.dashboard` | Painéis centrados numa animação para qualquer simulação: animação no meio, gráficos sincronizados ao cursor de tempo, KPIs, parâmetros e barra de cor; exporta HTML interativo, PNG e GIF/MP4. |
| `pp.data` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_data` | Raiz do pacote: importa, cada um dentro de `try/except` (falha silenciosa), STL, `PhysicalSample`, armazenamento Zarr, collate, colocação, active learning, normalizadores, tipos UPD, splits e construtores de batch; os `datasets` são importados sem proteção. |
| `pp.datasets` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_hub.zoo.datasets` | `pp.datasets`: datasets verificados em uma linha, guardados como npz sem pickle; os pequenos vêm no pacote, os outros são baixados do Hub, conferidos por SHA-256 e cacheados. |
| `pp.DDPPINNTrainer` | import direto | `pinneapple_neural.trainer.DDPPINNTrainer` | Treinador PINN multi-GPU com DDP: inicializa o grupo de processos, sincroniza gradientes e pode lançar workers via `mp.spawn`. |
| `pp.DDPTrainerConfig` | import direto | `pinneapple_neural.trainer.DDPTrainerConfig` | Configuração do treino DDP: backend, número de processos, endereço/porta do mestre, clip de gradiente, AMP e frequência de log. |
| `pp.decide` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_decision.api.decide` | Decide qual experimento rodar a seguir para o problema e objetivo dados. |
| `pp.decision` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_decision` | Fachada do pacote: reexporta tipos, decisor, backends, laço de decisão, seletores, árvore, verificadores e a API `decide`/`execute`/`verify`/`run_tree`; importar o pacote não carrega torch. |
| `pp.decision_engine` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_decision.api.decision_engine` | Devolve o motor padrão; qualquer argumento (ou `reset=True`) cria um novo, sem carregar a evidência antiga. |
| `pp.DEFAULT_MODELS` | import direto | `pinneapple_tools.benchmark_suite.DEFAULT_MODELS` | Lista padrão de `ModelSpec` usada pela suíte de benchmarks (`pinneapple_tools.benchmark_suite.benchmark`). |
| `pp.design` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_design.intent.design` | Projeta a partir de uma meta em linguagem natural; devolve `DesignPackage`, `NeedsInput` ou `OutOfScope`. |
| `pp.design_opt` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_design.design_optimizer` | Ponto de entrada do otimizador de projeto: reexporta o laço fechado (espaço de parâmetros, surrogate, otimizador, refinamento por PINN, Pareto) e a camada de estudos de projeto. |
| `pp.DesignOptConfig` | import direto | `pinneapple_design.design_optimizer.DesignOptConfig` | Configuração do `DesignOptLoop`: iterações, configuração do otimizador interno, refinamento top-k e outros. |
| `pp.DesignOptimizerConfig` | import direto | `pinneapple_design.design_optimizer.DesignOptimizerConfig` | Configuração única dos back-ends (método gradient/bayesian/evolutionary, iterações, taxa de aprendizado, etc.). |
| `pp.DesignOptLoop` | import direto | `pinneapple_design.design_optimizer.DesignOptLoop` | Orquestra o laço fechado: parametriza, avalia com o surrogate, otimiza, refina por PINN e monta a frente de Pareto. |
| `pp.DesignOptResult` | import direto | `pinneapple_design.design_optimizer.DesignOptResult` | Resultado completo de um `DesignOptLoop`: melhor design, histórico, frente de Pareto, com gráfico de convergência e resumo. |
| `pp.DigitalTwin` | import direto | `pinneapple_systems.digital_twin.DigitalTwin` | Gêmeo digital que funde um modelo substituto treinado com dados de sensores vivos; gerencia fluxos, laço de atualização, callbacks, predição e histórico. |
| `pp.DigitalTwinConfig` | import direto | `pinneapple_systems.digital_twin.DigitalTwinConfig` | Configuração do `DigitalTwin`: dispositivo, lote, AMP, intervalo de atualização, fila, histórico, assimilação e parâmetros de anomalia. |
| `pp.DirichletBC` | import direto | `pinneapple_physics.symbolic_pde.DirichletBC` | Condição de Dirichlet suave por penalidade MSE. |
| `pp.div` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_core.operators.div` | Divergência de um campo vetorial. |
| `pp.doctor` | import direto | `pinneapple_core.onboarding.doctor` | `pp.doctor`: verifica a instalação contra um perfil (`start`, `cfd`, `research`, `enterprise`, `all`) e imprime um comando de correção por falha. |
| `pp.dominant_frequency` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_physics.physics_audit.regime.dominant_frequency` | Frequência do pico espectral mais forte de um sinal amostrado (por exemplo desprendimento de vórtices). |
| `pp.DoMINO` | import direto | `pinneapple_physics.pinn_solver.DoMINO` | Treinador de PINN por decomposição de domínio: particiona, combina as saídas e penaliza descontinuidade na interface. |
| `pp.DragAdjointObjective` | import direto | `pinneapple_design.design_optimizer.DragAdjointObjective` | Objetivo de força de arrasto (pressão + viscoso) aproximado como média ponderada sobre pontos de superfície. |
| `pp.DragObjective` | import direto | `pinneapple_design.design_optimizer.DragObjective` | Proxy de arrasto: média de \|u\|^2 + peso x p^2 sobre o campo (sem normais de superfície). |
| `pp.dt` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_systems.digital_twin` | Runtime de gêmeo digital sobre modelos substitutos: reexporta `DigitalTwin`, estados, fluxos de sensores, filtros de Kalman, detectores de anomalia, pipeline de tempo real, reconstrução de sinais, condicionamento de telemetria e prognóstico (RUL). |
| `pp.dynamics` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_simulation.particle_dynamics` | Simulação dinâmica diferenciável em PyTorch puro: corpos rígidos, MPM (MLS-MPM) e SPH; reexporta também os parâmetros de solo/roda de Bekker-Wong. |
| `pp.EKIConfig` | import direto | `pinneapple_analysis.inverse_problems.EKIConfig` | Configuração da EKI: tamanho do ensemble, iterações, ruído, espalhamento inicial, regularização, prior, tolerância de colapso e semente. |
| `pp.ellipse` | import direto | `pinneapple_design.geometry.ellipse` | Atalho: `SDF` de elipse. |
| `pp.em` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_simulation.electromagnetics` | Pacote de eletromagnetismo em SI: referências em forma fechada, materiais, solvers FEM 2-D em A_z, FDTD, ondas e máquinas elétricas; reexporta as classes e funções principais dos submódulos. |
| `pp.ensemble` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_physics.ensemble` | Ensemble adaptativo de especialistas físicos: pesos atualizados online por erro contra referência, resíduo físico e custo, nos modos `select` ou `combine`, com intervalos conformais e detecção de suspeitos. |
| `pp.EnsembleKalmanInversion` | import direto | `pinneapple_analysis.inverse_problems.EnsembleKalmanInversion` | EKI padrão (Iglesias et al. 2013): inicializa o ensemble, aplica passos tipo Kalman e para por colapso do ensemble. |
| `pp.EnsembleUQ` | import direto | `pinneapple_analysis.uncertainty.EnsembleUQ` | Envelope de Deep Ensemble que dá média e incerteza a partir dos membros. |
| `pp.env` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_physics.pde_environment` | Ponto de entrada do ambiente de problemas: reexporta `ProblemSpec`, `PDETermSpec`, as condições de contorno, `ProblemBuilder`, o registro de presets, presets acadêmicos/CFD/industriais, os resíduos de turbulência e a base de famílias de EDP. O bloco de presets estruturais é importado dentro de `try/except ImportError`. [sem teste direto] |
| `pp.excitation` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_systems.excitation` | Sinais de excitação de atuador em NumPy (degrau, rampa, senoide, escada aleatória, PRBS, chirp, dente de serra, passeio aleatório suavizado, pulsos gaussianos, AM de dois tons) para identificação de sistemas e treino de surrogates; separa famílias de treino e famílias reservadas (`TRAIN_FAMILIES`, `HELDOUT_FAMILIES`). |
| `pp.execute` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_decision.api.execute` | Executa uma decisão emitida pelo mesmo motor, rechecando as restrições antes. |
| `pp.Experiment` | import direto | `pinneapple_physics.experiment.Experiment` | Método aplicado a um problema com semente, opções e referência opcional; `run` executa de forma reprodutível. |
| `pp.ExperimentResult` | import direto | `pinneapple_physics.experiment.ExperimentResult` | Registro de uma execução: config, fingerprint, solução, métricas, tempo, ambiente; salva e carrega em JSON. |
| `pp.export` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_tools.model_export` | Exportação de modelos `nn.Module` para TorchScript e ONNX e de arrays de coordenadas/campos para CSV e NPZ. [sem teste] |
| `pp.export_csv` | import direto | `pinneapple_tools.model_export.export_csv` | Grava coordenadas e campos num arquivo CSV com cabeçalho. |
| `pp.export_npz` | import direto | `pinneapple_tools.model_export.export_npz` | Grava coordenadas e campos num `.npz` comprimido do NumPy. |
| `pp.export_onnx` | import direto | `pinneapple_tools.model_export.export_onnx` | Exporta qualquer `nn.Module` para ONNX com nomes de entrada/saída, opset e eixos dinâmicos. |
| `pp.export_torchscript` | import direto | `pinneapple_tools.model_export.export_torchscript` | Exporta qualquer `nn.Module` para TorchScript (trace com `example_input`, ou script se não houver exemplo ou o trace falhar). |
| `pp.fea` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_simulation.external_solvers.calculix.study` | Estudos CalculiX em poucas linhas: malha (hexaedros, tetraedros gmsh ou deck existente), conjuntos por geometria, um passo de análise (estático, frequência, flambagem, calor), execução e resultados em arrays. |
| `pp.FlowVisualizer` | import direto | `pinneapple_neural.predictor.FlowVisualizer` | Classe de alto nível com linhas de corrente, contornos de pressão, módulo de velocidade e vorticidade a partir de um modelo. |
| `pp.flux` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_core.operators.flux` | Fluxo de um campo vetorial pela fronteira, `integral f . n dS`. |
| `pp.FourierFeatureEmbedding` | import direto | `pinneapple_neural.architectures.FourierFeatureEmbedding` | Embedding [cos, sin](2π B x) com matriz `B` aleatória ou treinável. |
| `pp.func` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_core.func` | Transformações de função (`grad`, `jacrev`, `jacfwd`, `jacobian`, `hessian`, `vmap`) sobre `torch.func` com `wrt` por nome ou posição, e `implicit_solve`, que diferencia através de um solver por uma solução adjunta. |
| `pp.GaussianMisfit` | import direto | `pinneapple_analysis.inverse_problems.GaussianMisfit` | Desajuste Gaussiano (L2/Mahalanobis). |
| `pp.geom` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_design.geometry` | Subpacote de geometria (antigo `pinneapple_geom`): reexporta `GeometrySpec`, `GeometryAsset`, `MeshData`, SDFs, CSG, domínios físicos 2D/3D, malhadores, RBF e `MeshCollocator`; o construtor STL é importado de forma tolerante (vira `None` se faltar dependência). |
| `pp.geometries` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_design.geometry_library` | Fachada da biblioteca de geometrias (exposta como `pp.geometries`): listar, construir, exportar, renderizar miniaturas, carregar arquivos pré-gerados e exportar a galeria web. |
| `pp.geophysics` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_simulation.geophysics` | Blocos do sistema terrestre verificados contra soluções exatas ou testes padrão: águas rasas na esfera, equação de Richards e modelo de balanço de energia de duas camadas. |
| `pp.get_backend` | import direto | `pinneapple_tools.compute_backends.get_backend` | Devolve o nome do backend ativo. |
| `pp.get_pde_family` | import direto | `pinneapple_physics.pde_environment.get_pde_family` | Busca uma `PDEFamily` por tipo ou apelido (sem diferenciar caixa). |
| `pp.get_physics_backend` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_core.backend.get_physics_backend` | Devolve o objeto backend pelo nome ou o backend ativo. |
| `pp.get_preset` | import direto | `pinneapple_physics.pde_environment.get_preset` | Cria um preset pelo nome, repassando argumentos. |
| `pp.get_rank` | import direto | `pinneapple_neural.trainer.get_rank` | Retorna o rank do processo atual (0 fora de modo distribuído). |
| `pp.get_rans_preset` | import direto | `pinneapple_physics.pde_environment.get_rans_preset` | Cria um objeto de resíduo de turbulência pelo nome; ValueError para nome desconhecido. |
| `pp.get_world_size` | import direto | `pinneapple_neural.trainer.get_world_size` | Retorna o número total de processos (1 fora de modo distribuído). |
| `pp.GlobalSensitivity` | import direto | `pinneapple_analysis.inverse_problems.GlobalSensitivity` | Sensibilidade global por variância (índices de Sobol pelo método de Saltelli). |
| `pp.gpu_info` | import direto | `pinneapple_neural.trainer.gpu_info` | Retorna um dict de informações por GPU disponível. |
| `pp.grad` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_core.operators.grad` | Gradiente: `(N, d)` para escalar, `(N, c, d)` para `c` componentes. |
| `pp.GradNormBalancer` | import direto | `pinneapple_neural.trainer.GradNormBalancer` | GradNorm: ajusta pesos para manter as normas de gradiente de cada termo numa camada compartilhada proporcionais à média. |
| `pp.HardBC` | import direto | `pinneapple_physics.symbolic_pde.HardBC` | Condição de contorno rígida por função distância: u = g + phi * rede, satisfeita exatamente; embrulha o modelo. |
| `pp.HashGridEncoding` | import direto | `pinneapple_neural.architectures.HashGridEncoding` | Codificação hash multirresolução com interpolação entre os cantos do voxel em cada nível. |
| `pp.HashGridMLP` | import direto | `pinneapple_neural.architectures.HashGridMLP` | MLP pequeno sobre a codificação hash, para representar campos 3D. |
| `pp.hessian` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_core.operators.hessian` | Hessiana `(N, d, d)` de um campo escalar. |
| `pp.HeteroscedasticMisfit` | import direto | `pinneapple_analysis.inverse_problems.HeteroscedasticMisfit` | Desajuste Gaussiano com variância de ruído por observação. |
| `pp.HuberMisfit` | import direto | `pinneapple_analysis.inverse_problems.HuberMisfit` | Desajuste de Huber (ou pseudo-Huber), robusto a outliers moderados. |
| `pp.Hybrid` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_core.module.Hybrid` | Solver mais correção neural (por exemplo modo `residual`), treinável através do solver. |
| `pp.IdentifiabilityAnalyzer` | import direto | `pinneapple_analysis.inverse_problems.IdentifiabilityAnalyzer` | Analisa a identificabilidade a partir dos autovalores da matriz de Fisher. |
| `pp.identify_pde` | import direto | `pinneapple_physics.pde_environment.identify_pde` | Pontua as famílias conhecidas contra uma descrição em texto livre por sobreposição de palavras-chave. |
| `pp.infer_on_grid_1d` | import direto | `pinneapple_neural.predictor.infer_on_grid_1d` | Avalia o modelo numa grade retangular x-t. |
| `pp.infer_on_grid_2d` | import direto | `pinneapple_neural.predictor.infer_on_grid_2d` | Avalia o modelo numa grade retangular x-y. |
| `pp.inference` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_neural.predictor` | Inferência pós-treino (antigo `pinneapple_inference`): reexporta inferência em lote e em grade, gráficos de campos, erro, perda, comparação, fluxo 3D e otimização de projeto. [sem teste direto] |
| `pp.info` | definido em `pinneapple/__init__.py` | `pinneapple.info` | Imprime versão, dispositivos, número de presets e modelos, pacotes importáveis e dependências opcionais com o extra de pip. |
| `pp.IntegralObsOperator` | import direto | `pinneapple_analysis.inverse_problems.IntegralObsOperator` | Funcional integral aproximado por quadratura ponderada com núcleo opcional. |
| `pp.integrate` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_core.operators.integrate` | Integral sobre o suporte (trapézio na grade, P1 exato na malha, quadratura ou Monte Carlo na nuvem). |
| `pp.inverse` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_analysis.inverse_problems` | Reexporta o kit de problemas inversos (modelos de ruído, regularizadores, operadores de observação, EKI, sensibilidade, termos faltantes, `InverseProblemSolver`, e o núcleo de identificabilidade/calibração/studio). O docstring ainda usa o nome antigo `pinneapple_inverse`. |
| `pp.InverseProblemSolver` | import direto | `pinneapple_analysis.inverse_problems.InverseProblemSolver` | Resolvedor unificado de problemas inversos com informação física. |
| `pp.InverseSolverConfig` | import direto | `pinneapple_analysis.inverse_problems.InverseSolverConfig` | Configuração do `InverseProblemSolver`: método, iterações, taxa de aprendizado e opções de EKI. |
| `pp.is_distributed` | import direto | `pinneapple_neural.trainer.is_distributed` | Diz se há um grupo de processos distribuído ativo. |
| `pp.IteratedEKI` | import direto | `pinneapple_analysis.inverse_problems.IteratedEKI` | EKI regularizada por Tikhonov (TEKI, Chada et al. 2020). |
| `pp.jacobian` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_core.operators.jacobian` | Jacobiana `(N, c, d)` de um campo. |
| `pp.JAXBackend` | import direto | `pinneapple_tools.compute_backends.JAXBackend` | Conjunto de utilitários JAX para PINNs sobre modelos Equinox ou Flax: `jit_pinn`, `vmap_residual`, `grad_fn` e conversões de tensores. |
| `pp.jit_pinn` | import direto | `pinneapple_tools.compute_backends.jit_pinn` | Alias de módulo para `JAXBackend.jit_pinn` (compila com JIT o resíduo do PINN). |
| `pp.KOmegaSSTResiduals` | import direto | `pinneapple_physics.pde_environment.KOmegaSSTResiduals` | Resíduos do modelo k-omega SST (RANS) para PINN, em 2D ou 3D conforme a entrada, com viscosidade turbulenta e resíduos de canal. |
| `pp.laplacian` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_core.operators.laplacian` | Soma das segundas derivadas, por componente em campos vetoriais. |
| `pp.LCurveSelector` | import direto | `pinneapple_analysis.inverse_problems.LCurveSelector` | Escolhe lambda pela máxima curvatura da curva L, chamando uma função de treino para cada valor. |
| `pp.LidDrivenCavityDomain2D` | import direto | `pinneapple_design.geometry.LidDrivenCavityDomain2D` | Cavidade quadrada com tampa móvel. |
| `pp.LinearObsOperator` | import direto | `pinneapple_analysis.inverse_problems.LinearObsOperator` | Operador linear `H(u) = A u` (ex.: tomografia). |
| `pp.list_methods` | import direto | `pinneapple_physics.solving.list_methods` | Nomes dos métodos registrados, opcionalmente filtrados por tipo. |
| `pp.list_models` | definido em `pinneapple/__init__.py` | `pinneapple.list_models` | Lista em ordem os nomes de modelos registrados no `ModelRegistry`. |
| `pp.list_pde_families` | import direto | `pinneapple_physics.pde_environment.list_pde_families` | Lista ordenada dos tipos canônicos de EDP. |
| `pp.list_presets` | import direto | `pinneapple_physics.pde_environment.list_presets` | Lista ordenada de todos os presets registrados. |
| `pp.LocalSensitivity` | import direto | `pinneapple_analysis.inverse_problems.LocalSensitivity` | Sensibilidade linearizada em torno de um ponto nominal, com Jacobiano por diferenças finitas opcional. |
| `pp.loss` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_core.loss` | Termos de perda como funções simples (`pde`, `boundary`, `supervised`, `conservation`, `energy`, `symmetry`, `inverse`), soma ponderada (`combine`) e o `Balancer` de pesos dinâmicos escolhido por nome. |
| `pp.LossRatioBalancer` | import direto | `pinneapple_neural.trainer.LossRatioBalancer` | Mantém as razões entre perdas (via EMA) próximas de valores-alvo em relação à perda da EDP. |
| `pp.lshape` | import direto | `pinneapple_design.geometry.lshape` | Domínio em L. |
| `pp.MassConservationConstraint` | import direto | `pinneapple_design.design_optimizer.MassConservationConstraint` | Penaliza o quadrado médio da divergência estimada do campo de velocidade. |
| `pp.maxwell` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_physics.maxwell_checks` | Verificações das equações de Maxwell sobre campos numa grade (de PINN, FNO, solver ou medição), sem referência: resíduos relativos com teste em `2h`, balanço de Poynting, contorno PEC, reciprocidade e passividade; alimenta `pinneapple check` (`physics = "maxwell"`). |
| `pp.maybe_compile` | import direto | `pinneapple_neural.trainer.maybe_compile` | Aplica `torch.compile` se o PyTorch suportar; caso contrário devolve o modelo original. |
| `pp.MCDropoutWrapper` | import direto | `pinneapple_analysis.uncertainty.MCDropoutWrapper` | Envolve um `nn.Module` e liga dropout na inferência por hooks; avisa quando não há ativação oculta para descartar. |
| `pp.MeshGraphNet` | import direto | `pinneapple_neural.architectures.MeshGraphNet` | GNN encoder-processador-decoder para simulação em malha não estruturada com conectividade fixa. |
| `pp.meta` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_adaptation.meta_learning` | Reexporta configurações, amostrador e treinadores de meta-learning e oferece `meta_train` (constrói e treina) e `meta_adapt` (adapta a uma tarefa nova). |
| `pp.MetaBenchmarkConfig` | import direto | `pinneapple_tools.benchmark_suite.MetaBenchmarkConfig` | Configuração do benchmark de meta-aprendizado: épocas meta, passos internos, taxas de aprendizado, tarefas por lote, valores de K, algoritmos, pontos e pesos de perda. |
| `pp.MetaBenchmarkFamily` | import direto | `pinneapple_tools.benchmark_suite.MetaBenchmarkFamily` | Uma família paramétrica de EDPs: fábricas de tarefas de treino e avaliação, faixas de parâmetros e parâmetros de avaliação. |
| `pp.MetaBenchmarkPipeline` | import direto | `pinneapple_tools.benchmark_suite.MetaBenchmarkPipeline` | Pipeline que roda o meta-treino e a avaliação K-shot, monta leaderboard, curvas de erro por K e salva/carrega resultados. |
| `pp.MetaBenchmarkResult` | import direto | `pinneapple_tools.benchmark_suite.MetaBenchmarkResult` | Resultado de uma família, parâmetro de avaliação, algoritmo e valor de K: métricas e tempo. |
| `pp.MethodNotAvailable` | import direto | `pinneapple_physics.solving.MethodNotAvailable` | Exceção: o método pedido não pode resolver este problema. |
| `pp.metrics` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_physics.metrics` | `pp.metrics`: L2, L2 relativo, RMSE, MAE, erro máximo, L∞ relativo e R² por campo, mais resumo, métrica agregada para rankings e erro de rollout temporal. |
| `pp.ModelRegistry` | import direto | `pinneapple_neural.architectures.ModelRegistry` | Registro central por nome: `register` (decorador), `list`, `families`, `spec`, `build` e `build_with_report`. |
| `pp.models` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_models` | [shim] Reexporta registro, catálogo e arquiteturas de `pinneapple_neural.architectures` para código legado; marcado como obsoleto (use `pinneapple_neural.architectures`). [sem teste direto] |
| `pp.ModelSpec` | import direto | `pinneapple_tools.benchmark_suite.ModelSpec` | Descrição de um modelo candidato: nome, fábrica que o constrói, descrição, tipo de entrada (`input_kind`) e família. |
| `pp.ModifiedMLP` | import direto | `pinneapple_neural.architectures.ModifiedMLP` | MLP com Fourier features e portas U/V tipo highway em cada camada. |
| `pp.molecular` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_tools.visualization.molecular` | Visualização molecular: proteínas e moléculas, sozinhas ou adsorvidas em superfícies metálicas, como imagens, trajetórias MP4, cenas glTF e painéis de dinâmica molecular. |
| `pp.MPMSimulator` | import direto | `pinneapple_simulation.particle_dynamics.MPMSimulator` | Simulador MPM diferenciável (`nn.Module`) com materiais elástico, fluido ou neve, em 2D/3D. |
| `pp.MPMState` | import direto | `pinneapple_simulation.particle_dynamics.MPMState` | Contêiner do estado das partículas MPM: posição, velocidade, gradiente de deformação F, matriz afim C e Jp. |
| `pp.multibody` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_simulation.multibody` | Dinâmica de corpos rígidos em coordenadas reduzidas: uma base flutuante com árvores de juntas de revolução; reexporta a árvore e os utilitários de braço. |
| `pp.naca_parametric` | import direto | `pinneapple_design.design_optimizer.naca_parametric` | Gera a superfície de um aerofólio NACA de 4 dígitos como coordenadas de pontos de controle (`pinneapple_design.geometry.gen.airfoil`). |
| `pp.NeumannBC` | import direto | `pinneapple_physics.symbolic_pde.NeumannBC` | Condição de Neumann suave pela derivada normal via autograd. |
| `pp.new` | definido em `pinneapple/__init__.py` | `pinneapple.new` | Cria um projeto do usuário fora do repositório (`pinneapple_core.onboarding.new_project`); recusa sobrescrever. |
| `pp.new_project` | import direto | `pinneapple_core.onboarding.new_project` | Escreve um projeto independente a partir de um template e preset e devolve a pasta. |
| `pp.non_hermitian` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_physics.non_hermitian` | Física não hermitiana: dímeros e cadeias PT-simétricos, pontos excepcionais (EPs) e se eles podem ser sintonizados; numpy, sympy para a parte simbólica e matplotlib para as figuras. |
| `pp.NSFlowSolver` | import direto | `pinneapple_simulation.numerical_solvers.NSFlowSolver` | Solver de Navier-Stokes incompressível na malha, com condições de contorno e exportação de dados PINN. |
| `pp.NTKWeightBalancer` | import direto | `pinneapple_neural.trainer.NTKWeightBalancer` | Pesos inversamente proporcionais ao traço do NTK de cada termo, estimado por Hutchinson. |
| `pp.Parameter` | import direto | `pinneapple_physics.physical_problem.Parameter` | Parâmetro físico nomeado com valor, unidade, papel (fixo, projeto, incerto, desconhecido), limites e distribuição. |
| `pp.ParetoFront` | import direto | `pinneapple_design.design_optimizer.ParetoFront` | Contêiner de uma frente de Pareto: objetivos, parâmetros e máscara, com filtro, hipervolume e gráfico. |
| `pp.ParticleSystem` | import direto | `pinneapple_simulation.particle_dynamics.ParticleSystem` | Base de sistemas de partículas com busca de vizinhos por força bruta O(N²); `forward` levanta NotImplementedError. |
| `pp.pde_from_sympy` | import direto | `pinneapple_physics.symbolic_pde.pde_from_sympy` | Cria um `SymbolicPDE` a partir de uma expressão de resíduo SymPy. |
| `pp.PeriodicBC` | import direto | `pinneapple_physics.symbolic_pde.PeriodicBC` | Periodicidade exata numa coordenada por mudança de variável (seno/cosseno). |
| `pp.PhysicalProblem` | import direto | `pinneapple_physics.physical_problem.PhysicalProblem` | Problema físico completo (geometria, EDP, condições, parâmetros, solver, intenção) com validação, serialização, fingerprint e adaptadores. |
| `pp.physics_audit` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_physics.physics_audit` | Auditoria física: qual modelo usar, quando confiar nele e quando sua formulação física deve ser revista; `diagnose` junta avaliação e diagnóstico num caso. Numpy puro, sem torch. |
| `pp.PhysicsDomain2D` | import direto | `pinneapple_design.geometry.PhysicsDomain2D` | Base dos domínios 2D: SDF, amostragem interna e por região, lote PINN e manifesto de condições. |
| `pp.PhysicsModule` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_core.module.PhysicsModule` | Classe base de modelos e solvers com `kind`, `describe()`, `num_parameters()` e um `fit` simples. |
| `pp.PhysicsOptimizer` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_core.optim.PhysicsOptimizer` | Parâmetros com limites, objetivo e restrições, minimizados pelo método escolhido em `minimize`. |
| `pp.PhysicsScenario` | import direto | `pinneapple_worldmodel.PhysicsScenario` | Especificação completa de um problema físico para o world model. |
| `pp.PhysicsSurrogate` | import direto | `pinneapple_design.design_optimizer.PhysicsSurrogate` | Envoltório fino de um `nn.Module` theta -> u com previsão, jacobiano, treino em dataset e checkpoint. |
| `pp.PhysicsValidator` | import direto | `pinneapple_analysis.validation.PhysicsValidator` | Orquestra uma suíte de checagens físicas sobre um modelo treinado. |
| `pp.PhysicsWorldModel` | import direto | `pinneapple_worldmodel.PhysicsWorldModel` | World model FNO: `(state_t, context) -> state_{t+1}` com formulação residual. |
| `pp.pinn` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_physics.pinn_solver` | Subpacote do compilador de PINN; reexporta `compile_problem`, pesos de perda, DoMINO e utilidades de PINN estocástico. |
| `pp.PINNArenaBenchmark` | import direto | `pinneapple_tools.benchmark_suite.PINNArenaBenchmark` | Suíte que roda todas as combinações tarefa × modelo, monta leaderboard, melhor modelo por problema, tabela-resumo, gráficos e salva/carrega resultados. |
| `pp.PINNRefinement` | import direto | `pinneapple_design.design_optimizer.PINNRefinement` | Refina a previsão do surrogate treinando uma PINN com perda de dados + resíduo de EDP, para um ou para os k melhores designs. |
| `pp.plot_isosurface_3d` | import direto | `pinneapple_neural.predictor.plot_isosurface_3d` | Desenha a isosuperfície 3D de um campo escalar previsto. |
| `pp.plot_streamlines_2d_from_model` | import direto | `pinneapple_neural.predictor.plot_streamlines_2d_from_model` | Desenha linhas de corrente do campo de velocidade previsto por um modelo. |
| `pp.plot_volume_slice` | import direto | `pinneapple_neural.predictor.plot_volume_slice` | Desenha uma fatia 2D de um campo 3D previsto. |
| `pp.PointObsOperator` | import direto | `pinneapple_analysis.inverse_problems.PointObsOperator` | Avalia o modelo num conjunto fixo de posições de sensores. |
| `pp.presets` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_presets` | Presets industriais: problemas de engenharia já montados, resolvidos e validados com o PINNeAPPle, acessíveis por `pp.presets.names()`/`load()`; reexporta o registro e as classes base. |
| `pp.profiles` | import direto | `pinneapple_core.onboarding.profiles` | Perfis aceitos pelo `doctor` e a frase que descreve cada um. |
| `pp.propagate_tolerances` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_analysis.uncertainty.tolerance.propagate` | Propaga tolerâncias independentes por `fn` via Monte Carlo. |
| `pp.propagate_uncertainty` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_analysis.uncertainty.tolerance.propagate_space` | Propaga entradas com qualquer distribuição por `fn`. |
| `pp.qualitative` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_design.qualitative` | Reexporta `preview`, `part_sensitivity`, `Assembly` e os modelos qualitativos. |
| `pp.Quantity` | import direto | `pinneapple_physics.physical_problem.Quantity` | Quantidade de interesse que o usuário quer da solução, com unidade e campo de origem. |
| `pp.quantum` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_quantum` | Fachada do Módulo Quântico (PQM): reexporta configuração de circuito, backends, modelos, perdas de Schrödinger, datasets, treinador e presets, e oferece os atalhos `build_circuit` e `backend_info`. [experimental] [sem teste] |
| `pp.quickstart` | definido em `pinneapple/__init__.py` | `pinneapple.quickstart` | Imprime o resumo de um preset (EDP, campos, coordenadas, domínio, condições) e os próximos passos sugeridos; devolve o spec. |
| `pp.recommend` | import direto | `pinneapple_core.onboarding.recommend` | `pp.recommend`: transforma uma frase em preset, método e modelo, com o motivo e os termos que casaram; é pontuação por palavras-chave e pode errar. [sem teste direto] |
| `pp.rectangle` | import direto | `pinneapple_design.geometry.rectangle` | Atalho: `SDF` de retângulo. |
| `pp.register_method` | import direto | `pinneapple_physics.solving.register_method` | Decorador que registra um método de solução sob um nome e um tipo. |
| `pp.register_preset` | import direto | `pinneapple_physics.pde_environment.register_preset` | Decorador que registra uma função fábrica de preset por nome. |
| `pp.reporting` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_tools.reporting` | Relatórios: estilo de figuras e construtor de PDF (`paperkit`) e datasheet técnico (`datasheet`); não é importado automaticamente pelo pacote. |
| `pp.ResidualBasedAL` | import direto | `pinneapple_data.ResidualBasedAL` | Escolhe pontos de colocação onde o resíduo da EDP é maior, pegando os top-k (RAR) ou sorteando com peso proporcional ao resíduo (RAD). |
| `pp.RigidBody` | import direto | `pinneapple_simulation.particle_dynamics.RigidBody` | Corpo rígido único com massa, inércia e integrador simplético. |
| `pp.RigidBodyState` | import direto | `pinneapple_simulation.particle_dynamics.RigidBodyState` | Contêiner mutável do estado (posição, orientação, velocidades) de vários corpos rígidos. |
| `pp.RigidBodySystem` | import direto | `pinneapple_simulation.particle_dynamics.RigidBodySystem` | Sistema de vários corpos rígidos com gravidade opcional. |
| `pp.run_parallel_sweep` | import direto | `pinneapple_neural.trainer.run_parallel_sweep` | Executa um sweep em grade em paralelo (threads ou processos) e devolve a lista de resultados. |
| `pp.run_tree` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_decision.api.run_tree` | Percorre a árvore de decisão de Physics AI (por padrão `physics_ai_tree()`). |
| `pp.science` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_science` | Ponto de entrada (`pp.science`): reexporta `atlas`, `experiments`, `run` e os tipos centrais. |
| `pp.score_against_reference` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_analysis.verification.physics_confidence_score.score_against_reference` | Pontua as predições de vários modelos contra um conjunto de referência separado. |
| `pp.security` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_security` | Ponto de entrada do pacote; o docstring é o índice das capacidades e diz as dependências (biblioteca padrão e NumPy; PyTorch para DP-SGD e carga segura; `cryptography` para Ed25519 e AES-GCM). |
| `pp.SelfAdaptiveWeights` | import direto | `pinneapple_neural.trainer.SelfAdaptiveWeights` | Pesos de perda aprendíveis (SA-PINN) atualizados por subida de gradiente, crescendo nos termos mais difíceis. |
| `pp.sensitivity` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_analysis.sensitivity` | Análise de sensibilidade global e local; reexporta Sobol, Morris, curvas de efeito, o estudo completo e as funções de teste. |
| `pp.Sequential` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_core.module.Sequential` | Encadeia componentes (módulos, solvers ou funções), a saída de um é a entrada do próximo. |
| `pp.set_backend` | import direto | `pinneapple_tools.compute_backends.set_backend` | Define o backend ativo, validando o nome contra o registro de `pinneapple_core.backend`; nome desconhecido levanta `ValueError`. |
| `pp.ShapeParametrization` | import direto | `pinneapple_design.design_optimizer.ShapeParametrization` | Parametrização de forma para otimização por gradiente (pontos de controle FFD, curvas de Bézier ou família NACA), com deformação de malha. |
| `pp.SineLayer` | import direto | `pinneapple_neural.architectures.SineLayer` | Camada senoidal com inicialização de pesos de Sitzmann et al. |
| `pp.SIREN` | import direto | `pinneapple_neural.architectures.SIREN` | Rede SIREN com `omega_0` configurável e última camada linear por padrão. |
| `pp.sobol_indices` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_analysis.sensitivity.sobol.sobol_indices` | Índices de Sobol de primeira ordem e totais (e de segunda ordem, opcional) de `fn`. |
| `pp.Solution` | import direto | `pinneapple_physics.solving.Solution` | Resultado de `solve`: função `predict`, tempo, histórico, modelo, grade e tipo do método, com `metadata` uniforme. |
| `pp.solve` | import direto | `pinneapple_physics.solving.solve` | Resolve o problema com o método pedido (nome registrado ou callable). |
| `pp.SolverModule` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_core.module.SolverModule` | Um solver como componente, com parâmetros treináveis opcionais; `from_method` usa um backend de `pp.solve` (não diferenciável). |
| `pp.solvers` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_solvers` | [shim] Pacote de compatibilidade (obsoleto) que reexporta os solvers de `pinneapple_simulation.numerical_solvers`; os solvers opcionais viram `None` se o import falhar. [sem teste direto do `__init__`] |
| `pp.SpalartAllmarasResiduals` | import direto | `pinneapple_physics.pde_environment.SpalartAllmarasResiduals` | Resíduos do modelo de uma equação de Spalart-Allmaras. |
| `pp.SparsityRegularizer` | import direto | `pinneapple_analysis.inverse_problems.SparsityRegularizer` | Regularizador L1/LASSO com aproximação suave. |
| `pp.SPHParticles` | import direto | `pinneapple_simulation.particle_dynamics.SPHParticles` | SPH fracamente compressível: densidade por soma de kernel, pressão de Tait, viscosidade e gravidade. |
| `pp.StructuralObjective` | import direto | `pinneapple_design.design_optimizer.StructuralObjective` | Proxy de tensão de von Mises pela média dos quadrados do campo. |
| `pp.StudentTMisfit` | import direto | `pinneapple_analysis.inverse_problems.StudentTMisfit` | Log-verossimilhança negativa de Student-t; `nu` controla a robustez. |
| `pp.StudyModel` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_analysis.study_models.StudyModel` | Envelope de modelo para estudos: predição, nomes, unidades, limites, desvio opcional e construtores de função, zoo e ONNX. |
| `pp.Subdomain` | import direto | `pinneapple_physics.pinn_solver.Subdomain` | Subdomínio retangular com sobreposição e testes de pertinência e interface. |
| `pp.SubdomainPINN` | import direto | `pinneapple_physics.pinn_solver.SubdomainPINN` | PINN de um único subdomínio. |
| `pp.suggest_problem_spec` | import direto | `pinneapple_physics.pde_environment.suggest_problem_spec` | Sugere argumentos de construção de `ProblemSpec` a partir de uma descrição em texto livre. |
| `pp.SurrogateConfig` | import direto | `pinneapple_design.design_optimizer.SurrogateConfig` | Configuração do `PhysicsSurrogate`: modelo (nome registrado ou MLP), dimensões e camadas. |
| `pp.SweepConfig` | import direto | `pinneapple_neural.trainer.SweepConfig` | Configuração do sweep em grade: grade de parâmetros, número de jobs, backend (threads/processos) e timeout. |
| `pp.symbolic` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_physics.symbolic_pde` | Compilador simbólico de EDP: transforma resíduos SymPy em funções PyTorch com autograd e oferece condições de contorno rígidas e suaves; reexporta `SymbolicPDE`, `HardBC` etc. [sem teste direto do __init__] |
| `pp.SymbolicPDE` | import direto | `pinneapple_physics.symbolic_pde.SymbolicPDE` | Resíduo de EDP simbólico compilado para função PyTorch (autograd ou grade). |
| `pp.t_junction` | import direto | `pinneapple_design.geometry.t_junction` | Junção em T: canal horizontal mais ramo vertical. |
| `pp.terramechanics` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_simulation.terramechanics` | Terramecânica de rodas de rovers planetários: solos com proveniência, modelo de roda de Wong-Reece e extensões, terreno deformável com sulcos, bancada de roda única com dados publicados e substitutos PINN do contato. |
| `pp.ThermalEfficiencyObjective` | import direto | `pinneapple_design.design_optimizer.ThermalEfficiencyObjective` | Eficiência térmica (fluxo de calor sobre queda de pressão proxy), devolvida negativa para minimizar. |
| `pp.ThroughputMonitor` | import direto | `pinneapple_neural.trainer.ThroughputMonitor` | Mede amostras/s, memória da GPU e utilização (com `pynvml`, se houver) por época. |
| `pp.TikhonovRegularizer` | import direto | `pinneapple_analysis.inverse_problems.TikhonovRegularizer` | Regularizador de Tikhonov/ridge, com prior e covariância opcionais. |
| `pp.TimeMarchingTrainer` | import direto | `pinneapple_neural.trainer.TimeMarchingTrainer` | Treina uma rede por janela de tempo, congelando a predição no fim de cada janela como condição inicial da seguinte; avalia e plota o resultado completo. |
| `pp.to_jsonable` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_core.jsonable.to_jsonable` | Devolve uma cópia de um objeto que `json.dumps` aceita, com opção de arredondar e tratar não-finitos. |
| `pp.TotalVariationRegularizer` | import direto | `pinneapple_analysis.inverse_problems.TotalVariationRegularizer` | Variação total anisotrópica para parâmetros de campo 1D/2D. |
| `pp.train` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_train` | [shim] Pacote de compatibilidade (marcado como obsoleto) que reexporta o treinador, callbacks e métricas de `pinneapple_neural.trainer`; nomes opcionais viram `None` se o import falhar. [sem teste direto do `__init__`] |
| `pp.TrainConfig` | import direto | `pinneapple_neural.trainer.TrainConfig` | Configuração do `Trainer`: épocas, taxa de aprendizado, clip, AMP, dispositivo, semente, determinismo, log, DDP e validação com física. |
| `pp.Trainer` | import direto | `pinneapple_neural.trainer.Trainer` | Treinador genérico: `fit(train_loader, val_loader, cfg)` roda épocas com perdas tensor/dict, métricas, early stopping, checkpoint e log JSONL. |
| `pp.transfer` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_adaptation.transfer_learning` | Reexporta `TransferConfig`, `TransferTrainer`, utilitários de congelamento, `ParametricFamilyTransfer` e `PhysicsTransferAdapter`. [sem teste direto do `__init__`] |
| `pp.TransferBenchmarkConfig` | import direto | `pinneapple_tools.benchmark_suite.TransferBenchmarkConfig` | Configuração: épocas e taxas da fonte e do ajuste fino, estratégias, aquecimento, pontos e pesos das perdas. |
| `pp.TransferBenchmarkPipeline` | import direto | `pinneapple_tools.benchmark_suite.TransferBenchmarkPipeline` | Pipeline que roda todas as estratégias de transferência e os baselines do zero, monta leaderboard, gráficos e salva/carrega resultados. |
| `pp.TransferBenchmarkResult` | import direto | `pinneapple_tools.benchmark_suite.TransferBenchmarkResult` | Resultado de (cenário, alvo, estratégia): métricas, histórico, tempo e parâmetros treináveis e totais. |
| `pp.TransferScenario` | import direto | `pinneapple_tools.benchmark_suite.TransferScenario` | Um cenário de transferência: uma tarefa fonte e N tarefas alvo, com rótulos. |
| `pp.transforms` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_core.transforms` | Transformações invertíveis e componíveis (`>>`) sobre tabelas, `PhysicsDataset` e, para as de escala, `PhysicalProblem`, registrando proveniência a cada `apply`; as não-escalares levantam `NotImplementedError` em nível de problema. |
| `pp.trust_card` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_analysis.trust.field_card.trust_card` | Roda `field_checks` e produz a decisão de deploy e as seis perguntas do cartão de confiança. |
| `pp.TwoPhaseConfig` | import direto | `pinneapple_neural.trainer.TwoPhaseConfig` | Configuração das duas fases: épocas e taxas de cada fase, modo combinado e rampa do peso da física. |
| `pp.TwoPhaseHistory` | import direto | `pinneapple_neural.trainer.TwoPhaseHistory` | Histórico de perdas das duas fases (incluindo componentes EDP e contorno na fase 2). |
| `pp.TwoPhaseTrainer` | import direto | `pinneapple_neural.trainer.TwoPhaseTrainer` | Executa a fase supervisionada e depois a fase de física, com otimizadores e históricos separados. |
| `pp.uncertainty` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_analysis.uncertainty` | Quantificação de incerteza (antigo `pinneapple_uq`); reexporta aleatória, epistêmica, conformal, calibração, quantis, reamostragem, drift, tolerâncias e `uq_predict`. |
| `pp.UnnormModel` | import direto | `pinneapple_neural.trainer.UnnormModel` | Envolve um modelo treinado em alvos normalizados para devolver predições na escala física (y * std + mean). |
| `pp.uq` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_analysis.uncertainty` | Quantificação de incerteza (antigo `pinneapple_uq`); reexporta aleatória, epistêmica, conformal, calibração, quantis, reamostragem, drift, tolerâncias e `uq_predict`. |
| `pp.uq_predict` | import direto | `pinneapple_analysis.uncertainty.uq_predict` | Interface unificada de predição com incerteza (método escolhido por `method`). |
| `pp.use_backend` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_core.backend.use_backend` | Gerenciador de contexto que roda um bloco num backend e restaura o anterior. |
| `pp.validate` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_analysis.validation.validate.pp_validate` | Uma chamada, um relatório de validação de um modelo (zoo, PyTorch, ONNX ou função) contra um problema, com credibilidade por fator e decisão pass / warn / fail (`pp_validate = _PPValidate()`). |
| `pp.validate_model` | import direto | `pinneapple_analysis.validation.validate_model` | Validação rápida de uma vez a partir de um `ProblemSpec` ou argumentos explícitos. |
| `pp.vehicle` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_simulation.vehicle` | Veículos planetários com rodas: descrição (chassi + módulos de canto) com CAD, modelo multicorpo construído a partir dela e simulador em terreno deformável com rodas de Wong-Reece. |
| `pp.verify` | preguiçoso (`_LAZY_ATTRS`) | `pinneapple_decision.api.verify` | Verifica um resultado; só resultados verificados viram evidência. |
| `pp.veriphysics` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_veriphysics` | Camada de verificação e evidência do Veriphysics dentro do PINNeAPPle; o `__init__` só documenta os submódulos (decisão, relatório, aplicabilidade, escopo, robustez, recomendação, ledger, Validation Lab). |
| `pp.viz` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_tools.visualization.studio` | Studio do PINNeAPPle: visualização 3D realista de qualquer geometria e sua física (cena, visualizador web, Blender, PyVista); exporta `Scene` e atalhos de render. |
| `pp.vmap_residual` | import direto | `pinneapple_tools.compute_backends.vmap_residual` | Alias de módulo para `JAXBackend.vmap_residual` (vetoriza um resíduo de um ponto sobre um lote). |
| `pp.WeightMinimizationObjective` | import direto | `pinneapple_design.design_optimizer.WeightMinimizationObjective` | Proxy de volume pela norma L2 do vetor de parâmetros. |
| `pp.WeightScheduler` | import direto | `pinneapple_neural.trainer.WeightScheduler` | Interface única: `step(losses)` aplica os pesos atuais, atualiza conforme o método e expõe parâmetros aprendíveis quando houver. |
| `pp.WeightSchedulerConfig` | import direto | `pinneapple_neural.trainer.WeightSchedulerConfig` | Configuração do agendamento de pesos: método (de `self_adaptive` até `auto`), pesos iniciais, frequência, taxas, limites e decaimento EMA. |
| `pp.worldmodel` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_worldmodel` | Ponto de entrada do pacote: reexporta o pipeline, o orquestrador, cenários, modelos e fábricas; o docstring descreve o pipeline de cinco estágios do modelo de fundação física. |
| `pp.WorldModelConfig` | import direto | `pinneapple_worldmodel.WorldModelConfig` | Hiperparâmetros do `PhysicsWorldModel`. |
| `pp.WorldModelDataset` | import direto | `pinneapple_worldmodel.WorldModelDataset` | Dataset de previsão de um ou vários passos construído de uma lista de trajetórias. |
| `pp.WorldModelTrainer` | import direto | `pinneapple_worldmodel.WorldModelTrainer` | Treina um `PhysicsWorldModel` num `WorldModelDataset`. |
| `pp.zoo` | submódulo preguiçoso (`_SUBMODULES`) | `pinneapple_hub.zoo` | `pp.zoo`: modelos pré-treinados e verificados em uma linha (listar, carregar, prever, avaliar, validar, consultar checagens, exportar, fazer benchmark e fine-tune com guarda). |



## Como este documento foi feito

1. `docs/pt/_mapa/inventario.py` percorre o repositório por AST e grava `inventario.json`.
2. O texto em português de cada pacote, módulo e símbolo foi escrito lendo o código e está em `docs/pt/_mapa/prosa/*.json`; `docs/pt/_mapa/checar_prosa.py` confere que nada ficou sem texto.
3. `docs/pt/_mapa/gerar.py` monta os volumes. Quando falta texto em português para um símbolo, a tabela mostra a primeira linha da docstring original (em inglês).
4. `docs/pt/_mapa/pdf.js` gera o PDF.
