# Volume 7 — Ferramentas, orquestração e produtos

> Parte do [Mapa completo do PINNeAPPle](../mapa_completo.md). Gerado por `docs/pt/_mapa/gerar.py` a partir do inventário do commit `e53cdcfb04`.

Ferramentas (benchmarks, exportação, visualização, automação de GUIs de engenharia, NVIDIA), o Flow, o Hub, o Lab, os LLMs, o servidor MCP, o laboratório autônomo e o app web.

## Sumário

1. [`pinneapple_tools`](#1-pinneapple_tools)
2. [`pinneapple_blender`](#2-pinneapple_blender)
3. [`pinneapple_flow`](#3-pinneapple_flow)
4. [`pinneapple_hub`](#4-pinneapple_hub)
5. [`pinneapple_lab`](#5-pinneapple_lab)
6. [`pinneapple_llm`](#6-pinneapple_llm)
7. [`pinneapple_mcp`](#7-pinneapple_mcp)
8. [`pinneapple_autolab`](#8-pinneapple_autolab)
9. [`pinneapple_app`](#9-pinneapple_app)

## 1. `pinneapple_tools`

### Objetivo

`pinneapple_tools` é a caixa de ferramentas de apoio do PINNeAPPle: tudo o que fica em volta do treino e da solução de um problema físico, mas não é o modelo nem o solver em si. Reúne visualização no estilo CFD (`visualization`), exportação de modelos (`model_export`), o pipeline de literatura (`hpo_experiments`), a suíte de benchmarks (`benchmark_suite`) e a camada de backend PyTorch/JAX (`compute_backends`). Vários desses submódulos eram pacotes separados; o `__init__` ainda expõe os nomes antigos (`viz`, `export`, `researcher`, `arena`, `backend`) como aliases de compatibilidade.

Além disso, o pacote tem ferramentas de engenharia mais recentes: coeficientes aerodinâmicos a partir de pressão na superfície (`aero_coefficients`), qualidade de datasets e curadoria por modelo de visão-linguagem (`dataset_quality`), métricas e gráficos genéricos de avaliação (`model_evaluation`), relatórios técnicos em PDF/HTML (`reporting`), a política que decide se um problema precisa de surrogate (`surrogate_policy`) e a execução de scripts de solver de usuários em outro processo com limites de recurso (`sandbox`).

Três submódulos são sistemas inteiros. `automation` é o "agente de GUI de engenharia": executa um workflow em YAML (FreeCAD, CfdOF, OpenFOAM, ParaView, web) passo a passo, para quando um passo pede confirmação humana, e grava evidências (comandos, logs, SHA-256 dos arquivos, screenshots, checagens) num relatório HTML e num `StudyResult`. `model_services` publica modelos do zoo (`pp.zoo`), o trust card e o extrator de intenção como microserviços HTTP com as rotas do NVIDIA NIM, cada resposta com o seu relatório de confiança. `rea_bridge` usa a ferramenta externa REA para estudar implementações de referência, com uma trava de licença que decide o que pode ser reaproveitado.

O pacote se liga ao resto do repositório em vários pontos: `automation` usa `pinneapple_check` nas checagens, `pinneapple_simulation.pipelines.freecad_openfoam` nos adaptadores e `pinneapple_data.contracts.StudyResult` no relatório, e se registra como passo `guiagent` do runner de `pinneapple_core.workflows` (usado pelos apps `guiagent` e `simulation_runner`). `model_services` serve modelos de `pp.zoo` e checagens de `pinneapple_physics`. `rea_bridge.tools` registra ferramentas no `PhysicsToolRegistry` de `pinneapple_worldmodel` para o loop de agente de `pinneapple_llm`. `reporting` é usado por estudos de `pinneapple_simulation` e por `pinneapple_veriphysics`.

Esta parte cobre quatro submódulos de `pinneapple_tools`. `benchmark_suite` (o "Arena") roda e compara modelos em problemas físicos: tarefas clássicas (Burgers, calor, onda, Poisson, Allen–Cahn, Taylor–Green, escoamentos), backends de treino (PyTorch nativo, registro de modelos, DeepXDE, JAX, PhysicsNeMo Sym), benchmarks de transferência e meta-aprendizado, pipelines declarativos e o protocolo do benchmark contra outros frameworks. `hpo_experiments` é, apesar do nome, o antigo pesquisador de literatura (arXiv, GitHub e um LLM Gemini). `nvidia` é a cola com PhysicsNeMo, OpenUSD/Omniverse, Isaac Sim e USD BRep. `visualization` reúne gráficos no estilo CFD, painéis animados, pontes para PyVista/Plotly/Rerun/Trame e o `studio` de cenas 3D.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 177 | 34.350 | 155 | 384 | 465 (35 arquivos) | 107 |

Pasta de testes dedicada: `tests/pinneapple_tools` (30 funções de teste).

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `aero_coefficients` | Utilitários em NumPy para Cp, integração de forças de pressão/cisalhamento numa superfície triangulada e conversão para CL/CD/CM nos eixos do vento. | 1 | 4 | 7 |
| `automation` | Agente de GUI de engenharia: executa workflows estruturados em FreeCAD, CfdOF, OpenFOAM, ParaView e interfaces web e entrega arquivos mais a evidência do que aconteceu. | 22 | 62 | 32 |
| `benchmark_suite` | Arena de benchmarks: tarefas físicas, backends de treino, comparação multi-modelo, transferência, meta-aprendizado, pipelines declarativos e o benchmark entre frameworks. | 52 | 118 | 155 |
| `compute_backends` | Seletor global de backend (PyTorch por padrão, JAX opcional) e utilitários JAX de jit/vmap para resíduos de PINN. | 3 | 8 | 156 |
| `dataset_quality` | Checagens estatísticas de qualidade de datasets em nuvem de pontos e curadoria de sequências com um modelo de visão-linguagem (VLM). | 3 | 16 | 16 |
| `hpo_experiments` | Pesquisador automático de literatura (antigo `pinneapple_researcher`): descoberta no arXiv/GitHub, base de conhecimento, extração e reprodução com LLM Gemini. | 27 | 44 | 0 |
| `model_evaluation` | Métricas genéricas de avaliação de modelos (regressão, bootstrap, sensibilidade, calibração) e os gráficos correspondentes em matplotlib. | 3 | 12 | 5 |
| `model_export` | Exportação de modelos para TorchScript e ONNX e de campos (coordenadas + valores) para CSV e NPZ. | 1 | 4 | 0 |
| `model_services` | Modelos do PINNeAPPle servidos como microserviços HTTP com as rotas do NVIDIA NIM, cada resposta com relatório de confiança; inclui cliente e servidor. | 6 | 17 | 39 |
| `nvidia` | Integração com a pilha NVIDIA: stages OpenUSD, robôs para Isaac Sim, surrogates PhysicsNeMo, geometria exata USD BRep, figuras e registro de execução. | 11 | 49 | 39 |
| `rea_bridge` | Ponte para a ferramenta REA (engenharia reversa) com trava de licença, mapa de funcionalidades e livro de registro do que foi estudado. | 6 | 20 | 22 |
| `reporting` | Estilo de figuras e construtores de relatório técnico: PDF com reportlab (`paperkit`) e datasheet HTML/PDF/Markdown (`datasheet`). | 3 | 14 | 14 |
| `sandbox` | Executa um script de solver fornecido pelo usuário em outro processo com limites de recurso POSIX; não é uma fronteira de segurança. | 2 | 1 | 13 |
| `surrogate_policy` | Decide, pelo tempo medido de uma solução real, se um problema precisa de surrogate treinado ou se o solver pode ser chamado direto. | 1 | 2 | 5 |
| `visualization` | Visualização no estilo CFD, painéis animados, pontes para visualizadores 3D externos e o studio de cenas 3D realistas (web, Blender, PyVista). | 35 | 165 | 46 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_tools`

Ponto de entrada do pacote: reexporta visualização, exportação, benchmarks, backends e qualidade de dados, mantém aliases antigos (`viz`, `export`, `researcher`, `arena`, `backend`) e define três atalhos. [sem teste]

<small>`pinneapple_tools/__init__.py` · 237 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot` | função | `(model, x, *, field_name='u', dim=None, **kwargs)` | Atalho que avalia o modelo em `x` sem gradiente e desenha a previsão com `plot_scalar` (1D ou 2D conforme a dimensão). |
| `export_model` | função | `(model, path, fmt='torchscript', **kwargs) -> str` | Atalho que despacha para `export_torchscript`, `export_onnx`, `export_csv` ou `export_npz` conforme `fmt`; formato desconhecido levanta `ValueError`. |
| `run_benchmark` | função | `(models, tasks=None, **bench_kwargs) -> 'BenchmarkResult'` | Atalho que monta um `BenchmarkConfig` e roda `PINNArenaBenchmark` sobre os modelos dados. |

#### `pinneapple_tools.aero_coefficients`

Funções compartilhadas de aerodinâmica em NumPy puro: Cp, integração de força e momento sobre malha triangulada e rotação para os eixos do vento (CL, CD, CM).

<small>`pinneapple_tools/aero_coefficients.py` · 293 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AeroCoefficients` | classe | `(CL, CD, CM, CY)` | Dataclass com os coeficientes adimensionais CL, CD, CM e a força lateral CY, nos eixos do vento. |
| `compute_cp` | função | `(p, p_inf, q_inf) -> np.ndarray` | Calcula o coeficiente de pressão `Cp = (p - p_inf) / q_inf`; levanta `ValueError` se `q_inf` for zero. |
| `integrate_surface_forces` | função | `(vertices, faces, face_normals, face_pressure, face_shear=None, moment_center=None) -> Tu…)` | Soma pressão (e cisalhamento opcional) por face de uma superfície triangulada e devolve força e momento líquidos em torno de um centro. |
| `force_to_coefficients` | função | `(force, moment, q_inf, ref_area, ref_length, alpha_deg, *, drag_axis='x', lift_axis='z', …)` | Converte força/momento nos eixos do corpo em CL/CD/CM nos eixos do vento, girando pelo ângulo de ataque no plano (`drag_axis`, `lift_axis`). |

#### `pinneapple_tools.automation`

Agente de GUI de engenharia ("Selenium para engenharia computacional"): expõe `Runner`, `Workflow`, `interpret`, `catalog` e afins para executar tarefas em softwares de engenharia com evidência.

<small>`pinneapple_tools/automation/__init__.py` · 84 linhas · testes: 32</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `example_workflow` | função | `(name) -> Path` | Devolve o caminho do `workflow.yaml` de um exemplo versionado em `automation/examples/<name>/`. |
| `catalog` | função | `(folder=None) -> dict[str, Workflow]` | Carrega todos os workflows de uma pasta (por padrão os exemplos versionados), indexados pelo id. |

#### `pinneapple_tools.automation.adapters`

Pacote de adaptadores, um por aplicação; importá-lo registra todos.

<small>`pinneapple_tools/automation/adapters/__init__.py` · 16 linhas · testes: 16</small>

#### `pinneapple_tools.automation.adapters.base`

Contrato comum dos adaptadores (`Adapter`, `ActionResult`, `StepContext`) e o registro global de adaptadores. [sem teste]

<small>`pinneapple_tools/automation/adapters/base.py` · 135 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ActionFailed` | classe | `()` | Exceção para uma ação que rodou e falhou; a mensagem diz o motivo e o log diz onde. |
| `ActionResult` | classe | `(outputs, artifacts, evidence, checks, actions)` | Resultado de uma ação: saídas para os passos seguintes, arquivos produzidos, imagens de evidência, checagens e ações registradas. |
| `StepContext` | classe | `(run_dir, step_dir, toolchain, display, workflow_dir, criteria, previous, timeout_s, comm…)` | Contexto em que um passo roda: pastas, toolchain, display, critérios, registros anteriores, timeout, e helpers para rodar comandos com log. |
| `Adapter` | classe | `(requires, actions, needs_display)` | Classe base de um adaptador: declara o que requer, as ações que oferece e se precisa de display, e despacha `run(action, params, ctx)` para `do_<ação>`. |
| `register` | função | `(adapter) -> Adapter` | Registra uma instância de adaptador no registro global pelo nome. |
| `get_adapter` | função | `(name) -> Adapter` | Devolve o adaptador registrado com o nome dado. |
| `list_adapters` | função | `() -> dict[str, Adapter]` | Devolve o dicionário de todos os adaptadores registrados. |
| `known_actions` | função | `() -> set[str]` | Conjunto de todas as ações `adaptador.ação` conhecidas. |

#### `pinneapple_tools.automation.adapters.checks`

Adaptador `verify` (avalia os critérios de aceitação do workflow sobre os arquivos do run) e adaptador `web` (abre uma UI web com Playwright e captura). [sem teste]

<small>`pinneapple_tools/automation/adapters/checks.py` · 71 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `VerifyAdapter` | classe | `()` | Adaptador `verify`: a ação `criteria` avalia cada critério do workflow e devolve as checagens. |
| `WebAdapter` | classe | `()` | Adaptador `web`: a ação `capture` abre uma página com Playwright e grava screenshot; só disponível com Playwright instalado. |

#### `pinneapple_tools.automation.adapters.freecad`

Adaptadores FreeCAD e CfdOF: rodam um script dentro da GUI do FreeCAD no display virtual, capturam a tela e a vista 3D, e montam a análise CfdOF. [sem teste]

<small>`pinneapple_tools/automation/adapters/freecad.py` · 195 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FreeCADAdapter` | classe | `()` | Adaptador `freecad`: ações `open` (descreve documento, parâmetros e sólido) e `macro` (roda uma macro com `ARGS`/`RESULT`). |
| `CfdOFAdapter` | classe | `()` | Adaptador `cfdof`: ação `setup` monta a análise CfdOF (laminar, estacionária, incompressível) e grava o caso de malha e o caso OpenFOAM. |
| `run_in_freecad_gui` | função | `(script, args, ctx, name, view=None, hold_s=60.0) -> tuple[dict[str, Any], ActionResult]` | Roda um script na GUI do FreeCAD com `ARGS`, espera o sinal de pronto, tira screenshots e devolve o `RESULT` e a evidência. |

#### `pinneapple_tools.automation.adapters.openfoam`

Adaptador OpenFOAM pela linha de comando: roda `Allmesh`/`Allrun`, lê os logs com `parse_log` e aplica correções de compatibilidade do CfdOF com OpenFOAM v1912.

<small>`pinneapple_tools/automation/adapters/openfoam.py` · 223 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OpenFOAMAdapter` | classe | `()` | Adaptador `openfoam`: ações `mesh`, `solve`, `prepare` e `post`, com checagens de malha, fim do solver e convergência dos resíduos. |
| `patch_cfdof_allmesh` | função | `(mesh_case) -> str \| None` | Edita o `Allmesh` do CfdOF para usar a sintaxe `-scale` em versões ESI sem `FOAM_API`; devolve a descrição da edição ou None. |
| `patch_cfdof_inlet` | função | `(case) -> list[str]` | Troca entradas `fixedNormalInletOutletVelocity` do CfdOF por `fixedValue` com a mesma velocidade normal (versões ESI antes da v2006). |
| `solver_log` | função | `(case) -> tuple[Path \| None, dict[str, Any]]` | Encontra o log do solver de um caso (`log.simpleFoam`, `log.foamRun`, ...) e devolve o caminho e o conteúdo interpretado. |
| `residual_plot` | função | `(parsed, path, title) -> Path \| None` | Desenha o histórico de resíduos de um log interpretado num PNG (evidência do tipo `plot`). |

#### `pinneapple_tools.automation.adapters.paraview`

Adaptador ParaView via `pvpython`: abre um caso OpenFOAM no último tempo, exporta um campo de célula para CSV e renderiza uma fatia em PNG. [sem teste]

<small>`pinneapple_tools/automation/adapters/paraview.py` · 70 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ParaViewAdapter` | classe | `()` | Adaptador `paraview`: ação `export` (CSV com uma linha por célula e imagem do tipo `render`). |

#### `pinneapple_tools.automation.adapters.pipeline`

Adaptadores do pipeline FreeCAD -> OpenFOAM de `pinneapple_simulation`: só convertem os resultados desse pipeline em `ActionResult`. [sem teste]

<small>`pinneapple_tools/automation/adapters/pipeline.py` · 57 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FreeCADHeadlessAdapter` | classe | `()` | Adaptador `freecadcmd`: ação `export` roda o FreeCAD sem tela (ou um substituto NumPy com `backend: fake`) e exporta STEP e STL por patch. |
| `pipeline_call` | função | `(fn, params, ctx) -> ActionResult` | Chama uma função do pipeline com os parâmetros e a pasta do passo e converte erros comuns em `ActionFailed`. |

#### `pinneapple_tools.automation.display`

Display virtual Xvfb, screenshots e os três drivers de GUI (xdotool, Playwright, pywinauto) usados quando a aplicação não tem API utilizável. [sem teste]

<small>`pinneapple_tools/automation/display.py` · 278 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GuiUnavailable` | classe | `()` | Exceção: o driver de GUI não pode rodar aqui (ferramenta ausente, plataforma errada, sem display). |
| `VirtualDisplay` | classe | `(width=1600, height=1000, depth=24, reuse=False)` | Display Xvfb que vive durante um bloco `with`; fornece o ambiente `DISPLAY` e screenshots via Pillow. |
| `XdotoolDriver` | classe | `(display)` | Encontra e age sobre janelas X11 com `xdotool` (esperar, levantar, teclar, clicar), registrando cada comando. |
| `PlaywrightDriver` | classe | `(headless=True, executable_path=None)` | Dirige uma UI web com Playwright (abrir, clicar, preencher, esperar texto, screenshot), registrando cada ação. |
| `PywinautoDriver` | classe | `(backend='uia')` | Interface para GUIs de desktop Windows via `pywinauto`; no Linux se declara indisponível. |

#### `pinneapple_tools.automation.examples.channel_obstacle.build_project`

Script para `freecadcmd` que constrói o projeto FreeCAD paramétrico do exemplo `channel_obstacle` (canal com obstáculo, dimensões numa planilha). [sem teste]

<small>`pinneapple_tools/automation/examples/channel_obstacle/build_project.py` · 33 linhas · **sem teste direto**</small>

#### `pinneapple_tools.automation.interpret`

Transforma uma tarefa (frase, dict ou JSON/YAML) num workflow do catálogo e seus inputs, por regras e com LLM opcional (Ollama); uma frase só escolhe workflows revisados e define inputs.

<small>`pinneapple_tools/automation/interpret.py` · 155 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Interpretation` | classe | `(workflow, inputs, method, notes, score)` | Resultado da interpretação: workflow escolhido, inputs, método (regras ou LLM), notas e pontuação. |
| `parse_quantity` | função | `(text, unit) -> float \| None` | Extrai o número de um texto e converte para a unidade pedida ("4 cm" -> 40 em mm); None se não houver número. |
| `interpret` | função | `(task, catalog, llm=None, min_score=0.15) -> Interpretation` | Converte a tarefa num workflow do catálogo e seus inputs, validando limites; recusa tarefas sem workflow correspondente. |

#### `pinneapple_tools.automation.record`

Registros de um run: `StepRecord` por passo e `RunRecord` em volta, gravados como `run.json`; um passo só conta como aprovado se a ação rodou e todas as checagens e artefatos estão ok.

<small>`pinneapple_tools/automation/record.py` · 171 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Check` | classe | `(name, passed, detail, value, threshold, source)` | Uma checagem: nome, se passou, detalhe, valor medido, limite e fonte. |
| `Evidence` | classe | `(path, kind, caption, sha256)` | Uma imagem do que aconteceu, com o tipo explícito (`screen`, `window`, `render`, `plot`), legenda e SHA-256. |
| `FileRecord` | classe | `(path, bytes, sha256, kind)` | Registro de um arquivo produzido: caminho, tamanho, SHA-256 e tipo. |
| `StepRecord` | classe | `(id, title, action, mode, state, started, ended, commands, logs, artifacts, evidence, che…)` | Registro de um passo: estado, horários, comandos, logs, artefatos, evidências, checagens, saídas, confirmação e erro. |
| `RunRecord` | classe | `(run_id, workflow, inputs, folder, state, created, finished, steps, environment, criteria…)` | Registro do run inteiro (workflow, inputs, passos, ambiente, critérios, mensagem), com `save`/`load` em `run.json`. |
| `file_record` | função | `(path, root) -> FileRecord` | Calcula tamanho e SHA-256 de um arquivo; para pasta, tamanho total e hash sobre os hashes ordenados dos arquivos. |

#### `pinneapple_tools.automation.report`

Relatório de evidências de um run: HTML autocontido com imagens embutidas e o mesmo conteúdo como `StudyResult` de `pinneapple_data.contracts`. [sem teste]

<small>`pinneapple_tools/automation/report.py` · 234 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `write_report` | função | `(run, path) -> Path` | Grava o relatório HTML autocontido do run (inputs, ferramentas, linha do tempo dos passos, critérios, erros). |
| `to_study_result` | função | `(run)` | Converte o run num `StudyResult` (`study_type="runs"`, `app="guiagent"`). |
| `write_study_result` | função | `(run, path) -> Path` | Grava o `StudyResult` do run em JSON. |

#### `pinneapple_tools.automation.runner`

Executa um workflow passo a passo: pula passos já aprovados, bloqueia sem pré-requisitos, para em confirmações (`expensive`, `overwrite`, ...) e na primeira falha, e reescreve `run.json` a cada mudança. [sem teste direto]

<small>`pinneapple_tools/automation/runner.py` · 290 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Runner` | classe | `(workflow, run_dir, inputs=None, toolchain=None, on_event=None, report=True)` | Executor de um workflow numa pasta de run, com inputs, aprovações e callback de eventos; `run()` executa e `decline()` encerra um run que pediu confirmação. |
| `plan` | função | `(workflow, toolchain=None) -> list[dict[str, Any]]` | Mostra o que o workflow precisaria sem rodar nada: por passo, adaptador, modo, se a aplicação está instalada e as confirmações pedidas. |

#### `pinneapple_tools.automation.runner_step`

Registra o tipo de passo `guiagent` no runner de `pinneapple_core.workflows`, para que um passo de DAG rode um workflow de automação inteiro e devolva o veredito.

<small>`pinneapple_tools/automation/runner_step.py` · 95 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GuiAgentStep` | classe | `(toolchain=None)` | Implementação do passo `guiagent`: roda o workflow pedido na pasta do passo e devolve estado, critérios, métricas e caminhos dos relatórios. |
| `register` | função | `(toolchain=None) -> GuiAgentStep` | Registra o tipo de passo `guiagent` no runner de workflows e devolve o adaptador. |

#### `pinneapple_tools.automation.scripts.cfdof_setup`

Script executado dentro da GUI do FreeCAD que monta uma análise CfdOF pela API Python (grupos de faces, material, malha gmsh) e grava os casos. [sem teste]

<small>`pinneapple_tools/automation/scripts/cfdof_setup.py` · 112 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `on_plane` | função | `(face, plane)` | Diz se uma face do sólido está sobre um dos planos da caixa envolvente (`x_min`, `x_max`, ...), para classificar entrada, saída e paredes. |

#### `pinneapple_tools.automation.scripts.freecad_gui_job`

Script lançado na GUI do FreeCAD que executa o job (script com `ARGS`/`RESULT`), ajusta a vista, sinaliza ao adaptador para tirar screenshots e sai. [sem teste]

<small>`pinneapple_tools/automation/scripts/freecad_gui_job.py` · 68 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `pump` | função | `(seconds)` | Mantém o laço de eventos da GUI do FreeCAD rodando por alguns segundos. |
| `run_job` | função | `()` | Executa o script do job dentro do FreeCAD, grava o resultado e coordena os arquivos `ready`/`release` com o adaptador. |

#### `pinneapple_tools.automation.scripts.freecad_inspect`

Script rodado dentro do FreeCAD que abre um documento e descreve objetos, parâmetros da planilha e o sólido pedido em `RESULT`. [sem teste]

<small>`pinneapple_tools/automation/scripts/freecad_inspect.py` · 27 linhas · **sem teste direto**</small>

#### `pinneapple_tools.automation.scripts.paraview_export`

Script para `pvpython` que abre um caso OpenFOAM, exporta um campo de célula para CSV e salva uma imagem da fatia. [sem teste]

<small>`pinneapple_tools/automation/scripts/paraview_export.py` · 92 linhas · **sem teste direto**</small>

#### `pinneapple_tools.automation.toolchain`

Descobre onde estão instalados FreeCAD, freecadcmd, CfdOF, OpenFOAM, gmsh e pvpython (variáveis de ambiente, locais usuais, PATH) e suas versões. [sem teste]

<small>`pinneapple_tools/automation/toolchain.py` · 109 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Toolchain` | classe | `(freecad, freecadcmd, cfdof, foam_bashrc, gmsh, pvpython)` | Caminhos das aplicações de engenharia, com `discover()`, a lista do que falta e as versões encontradas. |

#### `pinneapple_tools.automation.verify`

Critérios de aceitação de um run calculados dos arquivos produzidos (balanço de massa, vazão de entrada, queda de pressão, convergência, artefato, métrica, `physics_lint`) com as checagens de `pinneapple_check`.

<small>`pinneapple_tools/automation/verify.py` · 176 linhas · testes: 30</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `patch_fluxes` | função | `(case) -> dict[str, Any]` | Fluxo volumétrico [m³/s] por patch de contorno no último tempo, mais o líquido e o total. |
| `patch_mean` | função | `(case, field, patches) -> dict[str, float]` | Média ponderada por área de um campo escalar em cada patch nomeado, no último tempo. |
| `patch_area` | função | `(case, patch) -> float` | Área [m²] de um patch de contorno da malha. |
| `inflow` | função | `(case, inlet, velocity, tol=0.01) -> Check` | Checa se a vazão na entrada bate com `velocity` vezes a área do patch, dentro de `tol`. |
| `mass_balance` | função | `(case, inlet, outlet, tol=0.01) -> Check` | Checa \|entrada + saída\| / \|entrada\| <= `tol` com os fluxos dos patches. |
| `pressure_drop` | função | `(case, inlet, outlet, density=None, min_pa=None, max_pa=None) -> tuple[Check, float]` | Calcula a queda de pressão média entre entrada e saída (cinemática ou vezes `density`) e checa contra `min_pa`/`max_pa`. |
| `evaluate_criterion` | função | `(kind, params, run_dir, previous) -> Check` | Avalia um critério do workflow pelo tipo e devolve um `Check`. |

#### `pinneapple_tools.automation.workflow`

Workflow estruturado em YAML/JSON: inputs com limites, passos com pré-condições, artefatos esperados, confirmações e critérios, com templates `{inputs.x}`, `{steps.id.out}`. [sem teste direto]

<small>`pinneapple_tools/automation/workflow.py` · 274 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `WorkflowError` | classe | `(errors)` | Exceção para workflow ou inputs inválidos; `errors` lista todos os problemas encontrados. |
| `ArtifactSpec` | classe | `(path, kind, min_bytes, description)` | Arquivo ou pasta que um passo deve deixar, relativo à pasta do run, com tamanho mínimo. |
| `InputSpec` | classe | `(default, min, max, unit, description, choices)` | Especificação de um input: padrão, mínimo, máximo, unidade, descrição e escolhas; `check` valida um valor. |
| `Step` | classe | `(id, action, title, description, params, after, requires, expects, confirm, timeout_s)` | Um passo do workflow: ação `adaptador.ação`, parâmetros, dependências, requisitos, artefatos esperados, confirmações e timeout. |
| `Criterion` | classe | `(id, kind, title, params)` | Critério de aceitação do resultado, avaliado pela ação `verify.criteria`. |
| `Workflow` | classe | `(id, title, steps, description, application, inputs, criteria, limits, source)` | O workflow completo: carrega de arquivo, valida, resolve inputs e expõe passos e pasta de origem. |
| `render` | função | `(value, ctx) -> Any` | Preenche recursivamente os templates de um valor a partir do contexto; um template inteiro preserva o tipo do valor referido. |

#### `pinneapple_tools.benchmark_suite`

Ponto de entrada da suíte de benchmarks: reexporta `Arena`, `PINNArenaBenchmark`, registros de tarefas e backends, relatórios e, dentro de `try/except`, os pipelines de física, séries temporais, transferência e meta-aprendizado.

<small>`pinneapple_tools/benchmark_suite/__init__.py` · 108 linhas · testes: 2</small>

#### `pinneapple_tools.benchmark_suite.api`

API de alto nível `Arena`: liga uma `ProblemSpec` (ou um preset, ou um YAML) ao pipeline de geração de dados, treino e métricas.

<small>`pinneapple_tools/benchmark_suite/api.py` · 591 linhas · testes: 153</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ArenaResult` | classe | `(spec, model, history, metrics, artifacts, model_id, elapsed_s)` | Resultado de `Arena.run()`: especificação, modelo treinado, histórico de perdas, métricas finais, artefatos e tempo gasto. |
| `ArenaCompareResult` | classe | `(spec, results)` | Resultado de `Arena.compare()` com vários modelos, com `leaderboard()` para ordenar por métrica. |
| `Arena` | classe | `(spec) -> None` | Executor de alto nível criado por `from_spec`, `from_preset` ou `from_yaml`, que treina um modelo (`run`) ou compara vários (`compare`). |

#### `pinneapple_tools.benchmark_suite.backends`

Pacote dos backends de treino do Arena; importa e reexporta os cinco backends e o protocolo `Backend`. [sem teste]

<small>`pinneapple_tools/benchmark_suite/backends/__init__.py` · 17 linhas · **sem teste direto**</small>

#### `pinneapple_tools.benchmark_suite.backends.base`

Define o protocolo `Backend`: um objeto com `name` e um método `train(bundle, run_cfg)`.

<small>`pinneapple_tools/benchmark_suite/backends/base.py` · 20 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Backend` | classe | `(name)` | Protocolo de backend de treino: recebe um `BundleData` e uma configuração e devolve saídas de treino. |

#### `pinneapple_tools.benchmark_suite.backends.deepxde_backend`

Backend DeepXDE registrado como `deepxde`: baseline supervisionado nos sensores do bundle (x, y, u, v, p opcional). [sem teste]

<small>`pinneapple_tools/benchmark_suite/backends/deepxde_backend.py` · 113 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DeepXDEBackend` | classe | `(name)` | Treina uma rede DeepXDE só com os dados de sensores e devolve `predict_fn` (N,2)->(N,3); falha com `RuntimeError` se `deepxde` não estiver instalado. |

#### `pinneapple_tools.benchmark_suite.backends.jax_pinn`

Backend PINN completo em JAX para o caso `FlowObstacle2D` (Navier–Stokes 2D estacionário), registrado como `jax_pinn`. [sem teste]

<small>`pinneapple_tools/benchmark_suite/backends/jax_pinn.py` · 335 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `JAXPINNBackend` | classe | `(name)` | Treina em JAX com resíduo de momento e continuidade, perda de contorno e perda opcional em sensores; devolve `predict_fn` e métricas de teste. |

#### `pinneapple_tools.benchmark_suite.backends.native_pinn`

Backend PINN nativo em PyTorch para Navier–Stokes 2D estacionário, sem dependências externas além do torch. [sem teste]

<small>`pinneapple_tools/benchmark_suite/backends/native_pinn.py` · 287 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MLP` | classe | `(in_dim, out_dim, hidden, activation='tanh')` | Perceptron multicamada totalmente conectado com ativação escolhida por nome (tanh, relu, gelu, silu). |
| `NativePINNBackend` | classe | `()` | Treina um `MLP` em PyTorch com resíduo de Navier–Stokes nos pontos de colocação e perda de contorno do bundle. |

#### `pinneapple_tools.benchmark_suite.backends.physicsnemo_sym`

Backend que monta um PINN de Navier–Stokes com as abstrações do PhysicsNeMo Sym (domínio, restrições, solver) e usa o laço de treino dele.

<small>`pinneapple_tools/benchmark_suite/backends/physicsnemo_sym.py` · 240 linhas · testes: 2</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsNeMoSymBackend` | classe | `()` | Treina um PINN 2D estacionário com PhysicsNeMo Sym; exige `nvidia-physicsnemo.sym` e não garante devolver um modelo PyTorch. |

#### `pinneapple_tools.benchmark_suite.backends.pinneapple_models_backend`

Backend "multi-modelo" do Arena: treina qualquer modelo do `ModelRegistry` com batch unificado e perda física compilada quando suportada. [sem teste]

<small>`pinneapple_tools/benchmark_suite/backends/pinneapple_models_backend.py` · 200 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PinneappleModelsBackend` | classe | `()` | Carrega o bundle, monta o batch, compila a perda física se houver especificação, treina com o `Trainer` (opcionalmente DDP) e exporta previsões. |

#### `pinneapple_tools.benchmark_suite.benchmark`

PINN Arena: benchmark multi-problema e multi-arquitetura que treina vários modelos em várias tarefas físicas e gera ranking, tabelas e gráficos. A classe base de tarefa levanta `NotImplementedError` nos métodos abstratos.

<small>`pinneapple_tools/benchmark_suite/benchmark.py` · 1.309 linhas · testes: 155 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BenchmarkConfig` | classe | `(n_col, n_bc, n_ic, epochs, lr, weight_pde, weight_bc, weight_ic, device, seed, n_eval, c…)` | Configuração do benchmark: número de pontos de colocação/contorno/inicial, épocas, taxa de aprendizado, pesos das perdas, dispositivo, semente e limiar de convergência. |
| `ModelSpec` | classe | `(name, factory, description, input_kind, family)` | Descrição de um modelo candidato: nome, fábrica que o constrói, descrição, tipo de entrada (`input_kind`) e família. |
| `BenchmarkResult` | classe | `(problem_id, model_id, metrics, history, elapsed_s, n_params, rank)` | Resultado de um par (problema, modelo): métricas, histórico, tempo, número de parâmetros e posição no ranking. |
| `BenchmarkTaskBase` | classe | `(task_id, in_dim, out_dim)` | Base abstrata de tarefa física: amostragem de colocação, contorno e condição inicial, resíduo da EDP, grade de avaliação e montagem de batch; os métodos abstratos levantam `NotImplementedError`. |
| `NotApplicableError` | classe | `()` | Exceção para sinalizar que uma tarefa não monta batch para um dado `input_kind`; o próprio módulo não a levanta, só os chamadores. |
| `PINNArenaBenchmark` | classe | `(tasks, model_specs=None, config=None) -> None` | Suíte que roda todas as combinações tarefa × modelo, monta leaderboard, melhor modelo por problema, tabela-resumo, gráficos e salva/carrega resultados. |
| `all_model_specs` | função | `() -> List[ModelSpec]` | Gera um `ModelSpec` para cada modelo do registro de modelos, marcado com o `input_kind` real da família. |

#### `pinneapple_tools.benchmark_suite.bundle`

Pacote de leitura e validação de bundles de dados do Arena (reexporta `loader` e `schema`). [sem teste]

<small>`pinneapple_tools/benchmark_suite/bundle/__init__.py` · 7 linhas · **sem teste direto**</small>

#### `pinneapple_tools.benchmark_suite.bundle.loader`

Carrega um bundle do disco (manifesto, condições, pontos de colocação e contorno, sensores) e o valida contra um esquema.

<small>`pinneapple_tools/benchmark_suite/bundle/loader.py` · 149 linhas · testes: 2</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BundleData` | classe | `(root, manifest, conditions, points_collocation, points_boundary, sensors)` | Contêiner imutável de um bundle validado: raiz, manifesto, condições, pontos de colocação, pontos de contorno e sensores. |
| `load_bundle` | função | `(bundle_root, *, schema, require_sensors=False) -> BundleData` | Lê e valida um bundle no disco segundo um `BundleSchema`, exigindo sensores quando pedido. |

#### `pinneapple_tools.benchmark_suite.bundle.schema`

Esquema de validação de bundles lido de YAML: versão, arquivos, regiões, chaves do manifesto e colunas de sensores obrigatórias. [sem teste]

<small>`pinneapple_tools/benchmark_suite/bundle/schema.py` · 119 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BundleSchema` | classe | `(schema_version, required_files, required_regions, manifest_required_keys, sensors_requir…)` | Definição do esquema de um bundle, com `validate_bundle_root` para verificar uma pasta. |
| `load_bundle_schema` | função | `(path) -> BundleSchema` | Carrega um `BundleSchema` a partir de um arquivo YAML. |

#### `pinneapple_tools.benchmark_suite.crossframework`

Partes neutras do benchmark entre frameworks (PINNeAPPle contra PhysicsNeMo/PhysicsNeMo-Sym, DeepXDE, neuraloperator): problemas, pontos e pesos compartilhados, avaliador único, estatísticas e registros de execução.

<small>`pinneapple_tools/benchmark_suite/crossframework/__init__.py` · 17 linhas · testes: 20</small>

#### `pinneapple_tools.benchmark_suite.crossframework.darcy`

Trilha B do benchmark entre frameworks: dados de Darcy (`darcy_2d_li2021`), normalização e o laço único de treino de operadores. [sem teste]

<small>`pinneapple_tools/benchmark_suite/crossframework/darcy.py` · 182 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DarcyData` | classe | `(a, u, a_mean, a_std, u_mean, u_std)` | As três partições (treino, validação, teste) de Darcy com a normalização calculada no treino; `x` normaliza entradas e `decode` desfaz a normalização. |
| `load_split` | função | `(split, resolution=64) -> tuple[np.ndarray, np.ndarray]` | Devolve `(a, u)` de uma partição na resolução pedida, com média em blocos quando a resolução é menor. |
| `load_darcy` | função | `(resolution=64, n_train=N_TRAIN) -> DarcyData` | Carrega as três partições de Darcy e monta um `DarcyData` normalizado. |
| `train_operator` | função | `(model, data, *, epochs, seed, batch_size=20, lr=0.001, weight_decay=0.0001, log=None) ->…)` | Treina um operador (B,1,r,r)->(B,1,r,r) com o laço do protocolo e devolve histórico, tempos por época e previsões decodificadas. |
| `run_track_b` | função | `(argv, *, framework, source, build, model_config) -> None` | Corpo comum dos três executores da trilha B: cada executor só informa como sua biblioteca constrói o FNO. |
| `inference_latency` | função | `(model, data, batch, repeats=20, warmup=3) -> float` | Mediana do tempo de parede de uma passada direta sem gradientes sobre um lote de teste. |

#### `pinneapple_tools.benchmark_suite.crossframework.evaluate`

Avaliador único do benchmark entre frameworks (problemas diretos, inverso e operador) e estatísticas sobre sementes.

<small>`pinneapple_tools/benchmark_suite/crossframework/evaluate.py` · 141 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `evaluate_pinn` | função | `(problem, u_test, u_stencil=None, *, reference=None) -> dict[str, float]` | Métricas de uma execução direta a partir das previsões exportadas (erro na grade de teste e resíduo pelo estêncil). |
| `evaluate_inverse` | função | `(nu_hat, nu_true, u_pred, u_ref) -> dict[str, float]` | Execução inversa: erro relativo do coeficiente identificado e erro do campo nos pontos de teste. |
| `per_sample_rel_l2` | função | `(pred, ref) -> np.ndarray` | Erro L2 relativo por amostra, sobre todos os eixos menos o primeiro (aprendizado de operadores). |
| `evaluate_operator` | função | `(pred, ref) -> dict[str, float]` | Execução de operador: média, mediana e máximo do L2 relativo por amostra no teste. |
| `seed_stats` | função | `(values) -> dict[str, object]` | Mediana, mínimo, máximo e n dos valores finitos, contando à parte as falhas (NaN/inf) em `n_failed`. |
| `paired_ratio` | função | `(a, b) -> dict[str, object]` | Razão por semente `a / b` nas sementes em comum: mediana, extremos e quantas vezes `a < b`. |
| `sign_test_p` | função | `(k, n) -> float` | Valor-p exato bilateral do teste do sinal; o docstring avisa que com 5 sementes o menor p possível é 0.0625. |
| `summarize` | função | `(rows, keys, by) -> list[dict[str, object]]` | Agrupa linhas de resultado por campos e aplica `seed_stats` a cada métrica. |

#### `pinneapple_tools.benchmark_suite.crossframework.inverse`

Trilha C do benchmark entre frameworks: identificar a viscosidade de Burgers a partir de amostras ruidosas.

<small>`pinneapple_tools/benchmark_suite/crossframework/inverse.py` · 28 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `observations` | função | `(n=C['n_obs'], noise_rel=C['noise_rel'], seed=C['obs_seed']) -> dict[str, np.ndarray]` | Gera as observações do problema inverso: pontos (x, t), valores ruidosos e limpos e o desvio do ruído, com semente fixa do protocolo. |

#### `pinneapple_tools.benchmark_suite.crossframework.problems`

Problemas PINN diretos e inverso do benchmark entre frameworks (Burgers, Poisson, Helmholtz, Allen–Cahn), com referências, conjuntos de pontos, grade de teste e resíduo por diferenças finitas.

<small>`pinneapple_tools/benchmark_suite/crossframework/problems.py` · 314 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PINNProblem` | classe | `(name, coords, bounds, params, time_dependent, fd_step, citation, test_grid, extra)` | Um problema do benchmark: coordenadas, limites, parâmetros, se depende do tempo, passo do estêncil, citação e grade de teste; expõe condição inicial, contorno, fonte, solução exata e resíduo. |
| `get_problem` | função | `(name) -> PINNProblem` | Busca um problema por nome no dicionário `PROBLEMS`; levanta `KeyError` com a lista disponível se não existir. |
| `make_point_set` | função | `(problem, *, n_interior=10000, n_boundary=400, seed=20260) -> dict[str, np.ndarray]` | Gera os pontos de treino que todo framework carrega (protocolo A1): colocação interna uniforme, pontos e valores de contorno e, se houver tempo, condição inicial. |
| `point_set_digest` | função | `(points) -> str` | SHA-256 dos arrays do conjunto de pontos, para cada execução registrar em quais pontos treinou. |
| `save_point_set` | função | `(points, path) -> str` | Grava o conjunto de pontos em `.npz` e devolve o seu digest. |
| `load_point_set` | função | `(path) -> dict[str, np.ndarray]` | Lê um conjunto de pontos de um arquivo `.npz`. |
| `test_points` | função | `(problem) -> np.ndarray` | Grade de teste estruturada (float64, N x 2), a mesma para todos os frameworks. |
| `residual_points` | função | `(problem, n=2000, seed=777) -> np.ndarray` | Pontos internos novos, nunca usados no treino, para medir o resíduo, afastados do contorno por um passo de estêncil. |
| `stencil` | função | `(problem, X) -> np.ndarray` | Estêncil de 5 pontos em torno de cada ponto de resíduo, no formato (N, 5, 2). |
| `residual_from_stencil` | função | `(problem, X, values) -> np.ndarray` | Calcula o resíduo da EDP a partir dos valores no estêncil por diferenças centrais de segunda ordem. |
| `allen_cahn_reference` | função | `(d=0.0001, n=2048, dt=0.0001, t_end=1.0, n_save=1000) -> tuple[np.ndarray, np.ndarray, np…)` | Solução de referência numérica de Allen–Cahn periódica em [-1, 1) com u(x,0) = x² cos(πx). |
| `callable_exact` | função | `(problem) -> Callable[[np.ndarray], np.ndarray]` | Devolve a solução de referência como função `X -> u` (vetor coluna), para APIs que pedem um callable de referência. |
| `describe` | função | `(problem) -> dict[str, object]` | Descrição do problema pronta para JSON, guardada em todo arquivo de resultado. |

#### `pinneapple_tools.benchmark_suite.crossframework.protocol`

Configurações fixas do protocolo (trilhas A1, A2, B, C, lidas de `benchmarks/vs_frameworks/PROTOCOL.md`) e a exportação que todo executor produz.

<small>`pinneapple_tools/benchmark_suite/crossframework/protocol.py` · 97 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `export_predictions` | função | `(problem, predict, batch=20000) -> dict[str, np.ndarray]` | Avalia a função de previsão em lotes na grade de teste e no estêncil de resíduo e devolve `u_test` e `u_stencil`. |
| `ensure_point_set` | função | `(problem, root) -> Path` | Cria uma única vez o conjunto de pontos A1 do problema em `root/points/` e reutiliza o mesmo arquivo depois. |
| `runner_parser` | função | `(description)` | Parser de linha de comando comum a todos os executores (o driver só os chama por ele). |
| `base_meta` | função | `(args, *, track, framework, source, config) -> dict` | Metadados iniciais de cada resultado (trilha, framework, fonte, configuração); o executor completa status, iterações e tempos. |

#### `pinneapple_tools.benchmark_suite.crossframework.records`

Registros de execução do benchmark entre frameworks: ambiente, threads, carga da máquina, limites de tempo e esquema do arquivo de resultado. [sem teste]

<small>`pinneapple_tools/benchmark_suite/crossframework/records.py` · 166 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Clock` | classe | `(wall_s, cpu_s, loadavg_before, loadavg_after)` | Tempo de parede e de CPU do processo de um bloco, com a carga média antes e depois. |
| `TimeCap` | classe | `(seconds, kind='cpu')` | Orçamento de tempo por execução, medido em CPU do processo ou em tempo de parede, com `restart`, `used` e `expired`. |
| `setup_threads` | função | `(n) -> None` | Fixa o número de threads do processo (variáveis OMP/MKL/OpenBLAS e `torch.set_num_threads`); deve ser chamado antes de importar o framework. |
| `loadavg` | função | `() -> list` | Devolve a carga média do sistema operacional. |
| `cpu_model` | função | `() -> str` | Devolve o nome do modelo de CPU da máquina. |
| `package_versions` | função | `(names) -> dict[str, str \| None]` | Devolve a versão instalada de cada pacote pedido (None se ausente). |
| `environment` | função | `(threads) -> dict[str, Any]` | Registro da máquina e das bibliotecas, guardado junto aos resultados. |
| `write_run` | função | `(out_dir, run_id, meta, arrays) -> Path` | Grava `<run_id>.json` (metadados) e `<run_id>.npz` (previsões exportadas) na pasta de saída. |
| `read_run` | função | `(out_dir, run_id)` | Lê de volta os metadados e as previsões de uma execução gravada. |
| `validate_result` | função | `(meta) -> list` | Lista os campos obrigatórios ausentes nos metadados de um resultado (vazia quando válido). |

#### `pinneapple_tools.benchmark_suite.crossframework.weights`

Pesos iniciais compartilhados da trilha de configuração idêntica (A1), gerados em numpy e copiados para camadas PyTorch.

<small>`pinneapple_tools/benchmark_suite/crossframework/weights.py` · 48 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `layer_sizes` | função | `(in_dim=2, out_dim=1, width=50, depth=4) -> list[int]` | Lista de tamanhos de camada de um MLP a partir da dimensão de entrada, saída, largura e profundidade. |
| `xavier_uniform_init` | função | `(sizes, seed) -> list[tuple[np.ndarray, np.ndarray]]` | Gera pesos Xavier uniformes (Glorot & Bengio 2010) no layout de `torch.nn.Linear`, com bias zero, a partir de uma semente. |
| `load_into_linears` | função | `(linears, params) -> None` | Copia os pesos gerados para uma sequência de `torch.nn.Linear` (as formas precisam bater exatamente). |
| `count_parameters` | função | `(module) -> int` | Conta os parâmetros de um módulo PyTorch. |

#### `pinneapple_tools.benchmark_suite.deeponet_ground_truth`

Os quatro solvers de referência do artigo DeepONet (Lu, Jin & Karniadakis 2019): dada uma função de entrada amostrada de um GRF, calculam a solução verdadeira de cada sistema dinâmico. [sem teste]

<small>`pinneapple_tools/benchmark_suite/deeponet_ground_truth.py` · 155 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `solve_ode_linear` | função | `(draw, x_query) -> np.ndarray` | Solução exata da EDO linear pela antiderivada do próprio spline do GRF. |
| `solve_ode_nonlinear` | função | `(draw, x_query) -> np.ndarray` | Solução de referência da EDO não linear dirigida pela função de entrada, nos pontos pedidos. |
| `solve_pendulum` | função | `(draw, x_query, k=1.0) -> np.ndarray` | Solução do pêndulo forçado pela função de entrada; devolve o ângulo em cada ponto de consulta. |
| `pendulum_rk4_reference` | função | `(draw, x_max, k, n_steps=20000) -> float` | Integrador RK4 de passo fixo escrito à mão, só para conferir a integração adaptativa em scripts de validação (não para produção). |
| `solve_diffusion_reaction` | função | `(draw, nx=100, nt=100, D=0.01, k=0.01, x_max=1.0, t_max=1.0) -> 'tuple[np.ndarray, np.nda…)` | Difusão-reação com Crank–Nicolson para a difusão, uma correção de Picard por passo para a reação e algoritmo de Thomas. |

#### `pinneapple_tools.benchmark_suite.io`

Pacote de entrada/saída da suíte; reexporta os utilitários YAML. [sem teste]

<small>`pinneapple_tools/benchmark_suite/io/__init__.py` · 6 linhas · **sem teste direto**</small>

#### `pinneapple_tools.benchmark_suite.io.yamlx`

Funções finas para ler e escrever YAML como dicionário. [sem teste]

<small>`pinneapple_tools/benchmark_suite/io/yamlx.py` · 56 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `load_yaml` | função | `(path) -> Dict[str, Any]` | Lê um arquivo YAML e devolve o conteúdo como dicionário. |
| `dump_yaml` | função | `(obj) -> str` | Serializa um objeto Python para texto YAML. |

#### `pinneapple_tools.benchmark_suite.meta_benchmark`

Benchmark de meta-aprendizado para PINNs: treina MAML (primeira ordem) e Reptile em famílias paramétricas de EDPs e compara a adaptação em K passos com treino do zero.

<small>`pinneapple_tools/benchmark_suite/meta_benchmark.py` · 937 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MetaBenchmarkConfig` | classe | `(n_meta_epochs, n_inner_steps, inner_lr, outer_lr, n_tasks_per_batch, k_shots, algorithms…)` | Configuração do benchmark de meta-aprendizado: épocas meta, passos internos, taxas de aprendizado, tarefas por lote, valores de K, algoritmos, pontos e pesos de perda. |
| `MetaBenchmarkFamily` | classe | `(name, train_task_factory, eval_task_factory, param_ranges, eval_params)` | Uma família paramétrica de EDPs: fábricas de tarefas de treino e avaliação, faixas de parâmetros e parâmetros de avaliação. |
| `MetaBenchmarkResult` | classe | `(family_name, eval_params, algorithm, k_shot, metrics, elapsed_s)` | Resultado de uma família, parâmetro de avaliação, algoritmo e valor de K: métricas e tempo. |
| `MetaBenchmarkPipeline` | classe | `(families, config=None) -> None` | Pipeline que roda o meta-treino e a avaliação K-shot, monta leaderboard, curvas de erro por K e salva/carrega resultados. |
| `cfg_str` | função | `(results) -> str` | Texto curto com o número de passos do baseline `scratch_full`, tirado dos resultados. |

#### `pinneapple_tools.benchmark_suite.physics_pipeline`

`PhysicsPipeline`: pipeline físico padrão em 6 passos (problema, geometria, geração de pontos e perdas, solver de referência, modelos, treino e exportação com ranking). [sem teste]

<small>`pinneapple_tools/benchmark_suite/physics_pipeline.py` · 1.332 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsPipeline` | classe | `(problem, geometry=None, load_generate_data='generate', source=None, metrics='auto', coll…)` | Pipeline declarativo que recebe um problema e uma lista de modelos, gera dados de referência com um solver numérico, treina cada modelo com a perda física e produz um `BenchmarkReport`. |

#### `pinneapple_tools.benchmark_suite.pipeline.dataset_builder`

Reexportação fina de `pinneapple_data.adapters.pinn_batch_builders`, com o alias legado `BundleDataLike = PINNBatch`. [shim] [sem teste]

<small>`pinneapple_tools/benchmark_suite/pipeline/dataset_builder.py` · 22 linhas · **sem teste direto**</small>

#### `pinneapple_tools.benchmark_suite.pipeline.model_selector`

Ajudante para escolher modelos do `ModelRegistry` por tipo de entrada, chaves esperadas, suporte a perda física e tags. [sem teste]

<small>`pinneapple_tools/benchmark_suite/pipeline/model_selector.py` · 39 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `select_models` | função | `(*, input_kind, expects=None, supports_physics_loss=None, tags_any=None) -> List[str]` | Lista os nomes de modelos registrados que batem com o `input_kind` e os filtros opcionais. |

#### `pinneapple_tools.benchmark_suite.registry`

Registros globais de tarefas e backends do Arena, com decoradores de registro e funções de busca. [sem teste direto]

<small>`pinneapple_tools/benchmark_suite/registry.py` · 77 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `register_task` | função | `(task_cls) -> Type['ArenaTask']` | Decorador que registra uma classe de tarefa pelo seu `task_id`. |
| `get_task` | função | `(task_id) -> Type['ArenaTask']` | Devolve a classe de tarefa registrada com o id dado. |
| `list_tasks` | função | `() -> List[str]` | Lista os ids de tarefas registradas. |
| `register_backend` | função | `(backend_cls) -> Type['Backend']` | Decorador que registra uma classe de backend pelo seu `name`. |
| `get_backend` | função | `(name) -> Type['Backend']` | Devolve a classe de backend registrada com o nome dado. |
| `list_backends` | função | `() -> List[str]` | Lista os nomes de backends registrados. |

#### `pinneapple_tools.benchmark_suite.report`

Relatório estruturado dos benchmarks de física e de séries temporais: resultados por modelo, leaderboard, melhor modelo e exportação JSON. [sem teste direto]

<small>`pinneapple_tools/benchmark_suite/report.py` · 148 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ModelRunResult` | classe | `(model_id, n_params, training_time_s, metrics, history, param_estimates, rank, error_mess…)` | Resultado de um modelo: id, número de parâmetros, tempo de treino, métricas, histórico, parâmetros estimados, posição e mensagem de erro. |
| `BenchmarkReport` | classe | `(benchmark_type, created_at, version, problem_info, config, model_results, leaderboard, b…)` | Relatório completo de um benchmark, com `to_dict`, `to_json`, `save` e `print_summary`. |

#### `pinneapple_tools.benchmark_suite.runner`

Pacote dos executores de linha de comando do Arena (comparação, leaderboard, sweep, YAML, pipeline completo). [sem teste]

<small>`pinneapple_tools/benchmark_suite/runner/__init__.py` · 25 linhas · **sem teste direto**</small>

#### `pinneapple_tools.benchmark_suite.runner.compare`

Compara várias pastas de execução já concluídas, gerando tabela e gráficos. [sem teste]

<small>`pinneapple_tools/benchmark_suite/runner/compare.py` · 151 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `compare_runs` | função | `(run_dirs, *, out_dir='runs/arena_compare', title='arena_compare', make_plots=True) -> Di…)` | Lê as pastas de execução, junta métricas, escreve um resumo e, opcionalmente, gráficos de comparação. |

#### `pinneapple_tools.benchmark_suite.runner.leaderboard`

Leitura e atualização de um leaderboard em arquivo JSON. [sem teste]

<small>`pinneapple_tools/benchmark_suite/runner/leaderboard.py` · 43 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `load_leaderboard` | função | `(path) -> List[Dict[str, Any]]` | Carrega o leaderboard de um arquivo JSON. |
| `update_leaderboard` | função | `(path, row) -> None` | Acrescenta uma linha ao leaderboard em JSON. |

#### `pinneapple_tools.benchmark_suite.runner.metrics`

Utilitário pequeno para converter valores de métricas em float. [sem teste]

<small>`pinneapple_tools/benchmark_suite/runner/metrics.py` · 30 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ensure_float_dict` | função | `(d) -> Dict[str, float]` | Converte para float os valores de um dicionário quando possível. |

#### `pinneapple_tools.benchmark_suite.runner.report`

Grava os artefatos de uma execução (relatório, métricas, resumo) em disco. [sem teste]

<small>`pinneapple_tools/benchmark_suite/runner/report.py` · 67 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `write_run_artifacts` | função | `(*, artifacts_dir, run_id, report, metrics, summary) -> Path` | Persiste relatório, métricas e resumo de uma execução na pasta de artefatos e devolve o caminho. |

#### `pinneapple_tools.benchmark_suite.runner.run_arena_yaml`

Executa um experimento Arena completo a partir de um YAML unificado (preset do problema, vários modelos, métricas, inferência e visualização). [sem teste]

<small>`pinneapple_tools/benchmark_suite/runner/run_arena_yaml.py` · 563 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `run_arena_experiment` | função | `(yaml_path, *, out_dir=None) -> Dict[str, Any]` | Lê o YAML, treina cada modelo do experimento, avalia e devolve um resumo por modelo. |
| `main` | função | `() -> None` | Ponto de entrada de linha de comando para `run_arena_experiment`. |

#### `pinneapple_tools.benchmark_suite.runner.run_benchmark`

Executa um benchmark do Arena a partir de configurações de tarefa, execução e esquema de bundle. [sem teste]

<small>`pinneapple_tools/benchmark_suite/runner/run_benchmark.py` · 132 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `run_benchmark` | função | `(*, artifacts_dir, task_cfg_path, run_cfg_path, bundle_schema_path) -> Dict[str, Any]` | Carrega o bundle, escolhe o backend, treina, calcula as métricas da tarefa e grava os artefatos. |

#### `pinneapple_tools.benchmark_suite.runner.run_models_from_cfg`

Treina um modelo do registro a partir de uma configuração de execução (usado pelo sweep). [sem teste]

<small>`pinneapple_tools/benchmark_suite/runner/run_models_from_cfg.py` · 265 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `run_one` | função | `(*, run_cfg, run_dir, bundle_root=None) -> Dict[str, Any]` | Roda uma configuração de modelo numa pasta de execução e devolve as métricas. |
| `main` | função | `()` | Ponto de entrada de linha de comando para `run_one`. |

#### `pinneapple_tools.benchmark_suite.runner.run_pipeline`

Pipeline de ponta a ponta configurado por YAML: geometria, malha, solver (OpenFOAM, FEniCS ou embutido), dataset, treino, inferência e relatório. [sem teste]

<small>`pinneapple_tools/benchmark_suite/runner/run_pipeline.py` · 1.444 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `run_full_pipeline` | função | `(config_path_or_dict, *, out_dir=None) -> Dict[str, Any]` | Executa todas as etapas do pipeline a partir de um caminho de config ou dicionário e devolve o resumo. |
| `main` | função | `() -> None` | Ponto de entrada de linha de comando para `run_full_pipeline`. |

#### `pinneapple_tools.benchmark_suite.runner.run_sweep`

Sweep de benchmark sobre vários modelos, com paralelismo entre modelos (um processo por GPU, fila, ou DDP por modelo via torchrun). [sem teste]

<small>`pinneapple_tools/benchmark_suite/runner/run_sweep.py` · 208 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SweepJob` | classe | `(model_name, run_cfg)` | Um trabalho do sweep: nome do modelo e sua configuração de execução. |
| `run_sweep` | função | `(*, models, base_run_cfg, bundle_root, out_dir='runs/arena_sweep', parallelism='process',…)` | Distribui os modelos entre GPUs ou processos e produz uma pasta de execução por modelo. |

#### `pinneapple_tools.benchmark_suite.tasks`

Pacote das tarefas do Arena: reexporta `ArenaTask`, os casos de escoamento (obstáculo 2D, cavidade 3D) e as tarefas clássicas de benchmark.

<small>`pinneapple_tools/benchmark_suite/tasks/__init__.py` · 29 linhas · testes: 2</small>

#### `pinneapple_tools.benchmark_suite.tasks.allen_cahn_1d`

Tarefa de benchmark Allen–Cahn 1D com referência numérica opcional.

<small>`pinneapple_tools/benchmark_suite/tasks/allen_cahn_1d.py` · 135 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AllenCahn1DTask` | classe | `(eps=0.01, build_reference=True) -> None` | Tarefa Allen–Cahn 1D: amostragem de pontos, condições de contorno e inicial, resíduo da EDP e grade de avaliação. |

#### `pinneapple_tools.benchmark_suite.tasks.base`

Base das tarefas do Arena e o resultado padrão de uma avaliação.

<small>`pinneapple_tools/benchmark_suite/tasks/base.py` · 53 linhas · testes: 2</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TaskResult` | classe | `(metrics, artifacts)` | Saída padrão de uma avaliação de tarefa: métricas e artefatos. |
| `ArenaTask` | classe | `(task_id)` | Classe base de tarefa do Arena; `compute_metrics` padrão calcula L2 relativo e erro máximo das saídas do backend. |

#### `pinneapple_tools.benchmark_suite.tasks.burgers_1d`

Tarefa de benchmark Burgers 1D (viscosidade padrão 0.01/π) com referência numérica opcional.

<small>`pinneapple_tools/benchmark_suite/tasks/burgers_1d.py` · 152 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Burgers1DTask` | classe | `(nu=0.01 / np.pi, build_reference=True) -> None` | Tarefa Burgers 1D: amostragem, condições de contorno e inicial, resíduo, grade de avaliação e avaliação. |

#### `pinneapple_tools.benchmark_suite.tasks.flow_obstacle_2d`

Tarefa de escoamento 2D estacionário em torno de obstáculo, avaliada por modelo torch, função de previsão ou arrays. [sem teste]

<small>`pinneapple_tools/benchmark_suite/tasks/flow_obstacle_2d.py` · 263 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FlowObstacle2DTask` | classe | `(task_id, n_collocation_eval, n_boundary_eval, n_sensors_eval)` | Calcula resíduo de Navier–Stokes, erro de contorno e L2 em sensores conforme o que o backend devolveu. |

#### `pinneapple_tools.benchmark_suite.tasks.heat_1d`

Tarefa de benchmark equação do calor 1D com solução exata.

<small>`pinneapple_tools/benchmark_suite/tasks/heat_1d.py` · 85 linhas · testes: 153</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Heat1DTask` | classe | `(alpha=0.01) -> None` | Tarefa do calor 1D: solução exata, amostragem, resíduo e grade de avaliação. |

#### `pinneapple_tools.benchmark_suite.tasks.lid_driven_cavity_3d`

Benchmark de cavidade com tampa móvel 3D, com validação contra dados de linha central de Ghia et al. (1982). [sem teste]

<small>`pinneapple_tools/benchmark_suite/tasks/lid_driven_cavity_3d.py` · 394 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LidDrivenCavity3DTask` | classe | `(Re=100.0, size=1.0, lid_velocity=1.0) -> None` | Tarefa de cavidade 3D: métricas, perfil de referência da literatura (Re mais próximo entre os tabelados) e comparação com a previsão. |

#### `pinneapple_tools.benchmark_suite.tasks.navier_stokes_tgv_2d`

Tarefa de benchmark Navier–Stokes 2D com o vórtice de Taylor–Green (solução exata). [sem teste]

<small>`pinneapple_tools/benchmark_suite/tasks/navier_stokes_tgv_2d.py` · 183 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NavierStokesTGV2DTask` | classe | `(nu=0.01) -> None` | Tarefa Taylor–Green 2D: campos exatos u, v, p, amostragem, resíduo, grade e avaliação. |

#### `pinneapple_tools.benchmark_suite.tasks.poisson_2d`

Tarefa de benchmark Poisson 2D com solução exata e termo fonte. [sem teste]

<small>`pinneapple_tools/benchmark_suite/tasks/poisson_2d.py` · 98 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Poisson2DTask` | classe | `()` | Tarefa Poisson 2D: solução exata, forçamento, amostragem, resíduo e grade de avaliação. |

#### `pinneapple_tools.benchmark_suite.tasks.wave_1d`

Tarefa de benchmark equação da onda 1D com solução exata. [sem teste]

<small>`pinneapple_tools/benchmark_suite/tasks/wave_1d.py` · 103 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Wave1DTask` | classe | `(c=1.0) -> None` | Tarefa da onda 1D: solução exata, amostragem, condições, resíduo e grade de avaliação. |

#### `pinneapple_tools.benchmark_suite.timeseries_pipeline`

`TimeSeriesBenchmarkSpec`: benchmark declarativo de séries temporais que reaproveita datasets, modelos, treino e métricas existentes. [sem teste]

<small>`pinneapple_tools/benchmark_suite/timeseries_pipeline.py` · 579 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TimeSeriesBenchmarkSpec` | classe | `(source, models=('lstm',), metrics=('mse', 'mae', 'rmse'), target_cols=None, horizon=50, …)` | Especificação declarativa (fonte, modelos, métricas, horizonte, janela) cujo `run()` treina cada modelo e devolve um `BenchmarkReport`. |

#### `pinneapple_tools.benchmark_suite.transfer_benchmark`

Benchmark de transferência para PINNs: treina um modelo fonte e o ajusta a problemas alvo com várias estratégias, comparando com treino do zero.

<small>`pinneapple_tools/benchmark_suite/transfer_benchmark.py` · 798 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TransferBenchmarkConfig` | classe | `(n_source_epochs, lr_source, n_finetune_epochs, lr_finetune, lr_base, lr_scratch, strateg…)` | Configuração: épocas e taxas da fonte e do ajuste fino, estratégias, aquecimento, pontos e pesos das perdas. |
| `TransferScenario` | classe | `(name, source_task, target_tasks, source_label, target_labels)` | Um cenário de transferência: uma tarefa fonte e N tarefas alvo, com rótulos. |
| `TransferBenchmarkResult` | classe | `(scenario_name, target_label, strategy, metrics, history, elapsed_s, n_params_trainable, …)` | Resultado de (cenário, alvo, estratégia): métricas, histórico, tempo e parâmetros treináveis e totais. |
| `TransferBenchmarkPipeline` | classe | `(scenarios, config=None) -> None` | Pipeline que roda todas as estratégias de transferência e os baselines do zero, monta leaderboard, gráficos e salva/carrega resultados. |

#### `pinneapple_tools.compute_backends`

Suporte multi-backend: reexporta o seletor PyTorch/JAX e os utilitários JAX.

<small>`pinneapple_tools/compute_backends/__init__.py` · 44 linhas · **sem teste direto**</small>

#### `pinneapple_tools.compute_backends.backend`

Seletor global do backend ativo: "torch" (padrão), "jax" ou outro nome registrado em `pinneapple_core.backend`.

<small>`pinneapple_tools/compute_backends/backend.py` · 53 linhas · testes: 156</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Backend` | classe | `(TORCH, JAX)` | Namespace com as constantes de nome de backend (`TORCH`, `JAX`). |
| `set_backend` | função | `(name) -> None` | Define o backend ativo, validando o nome contra o registro de `pinneapple_core.backend`; nome desconhecido levanta `ValueError`. |
| `get_backend` | função | `() -> str` | Devolve o nome do backend ativo. |

#### `pinneapple_tools.compute_backends.jax_backend`

Utilitários JAX para PINNs (jit, vmap, gradiente, conversão torch<->jax); degrada sem erro de import quando JAX não está instalado.

<small>`pinneapple_tools/compute_backends/jax_backend.py` · 268 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `JAXBackend` | classe | `()` | Conjunto de utilitários JAX para PINNs sobre modelos Equinox ou Flax: `jit_pinn`, `vmap_residual`, `grad_fn` e conversões de tensores. |
| `jax_available` | função | `() -> bool` | Diz se o JAX pode ser importado. |
| `jit_pinn` | função | `(model_fn, residual_fn) -> Callable` | Alias de módulo para `JAXBackend.jit_pinn` (compila com JIT o resíduo do PINN). |
| `vmap_residual` | função | `(residual_fn, batch_size=64) -> Callable` | Alias de módulo para `JAXBackend.vmap_residual` (vetoriza um resíduo de um ponto sobre um lote). |
| `jax_pinn` | função | `(model_fn, residual_fn, vectorise=True) -> Callable` | Atalho que compila com JIT e, opcionalmente, vetoriza com vmap o resíduo de um PINN. |

#### `pinneapple_tools.dataset_quality`

Qualidade de datasets: reexporta as checagens estatísticas de `checks` e a curadoria por VLM de `vlm_curation`.

<small>`pinneapple_tools/dataset_quality/__init__.py` · 64 linhas · **sem teste direto**</small>

#### `pinneapple_tools.dataset_quality.checks`

Checagens genéricas sobre um dataset `(coords, fields)`: completude, consistência, distribuição, outliers e faixas físicas, sem dependência de framework. [sem teste direto]

<small>`pinneapple_tools/dataset_quality/checks.py` · 214 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `analyze_completeness` | função | `(fields) -> Dict[str, Any]` | Fração de NaN por campo e uma nota geral de completude. |
| `validate_consistency` | função | `(coords, fields, monotonic_name_hints=_MONOTONIC_NAME_HINTS) -> Dict[str, Any]` | Checa se os campos têm o mesmo número de pontos que `coords` e, por heurística de nome, se campos cumulativos/temporais são não decrescentes. |
| `analyze_distribution` | função | `(fields) -> Dict[str, Any]` | Resumo estatístico por campo com `scipy.stats.describe` (média, variância, assimetria, curtose). |
| `detect_outliers` | função | `(fields, random_state=42) -> Dict[str, Any]` | Ajusta um IsolationForest em todos os campos numéricos juntos e reporta a fração de pontos marcados como outliers. |
| `field_range_issues` | função | `(fields, ranges) -> list` | Checa cada campo contra uma faixa física `{nome: (lo, hi)}` e devolve frases com os problemas. |
| `build_1d_linear_interpolant` | função | `(z, y) -> 'Any'` | Constrói um interpolante linear por partes diferenciável (torch) a partir de amostras 1D, para avaliar resíduos de EDP sobre dados estáticos. |

#### `pinneapple_tools.dataset_quality.vlm_curation`

Curadoria de datasets sintéticos com um modelo de visão-linguagem: julga a plausibilidade física de um clipe por sequência e filtra pelo rótulo/confiança, sem inventar julgamentos (resposta inválida vira rótulo neutro com confiança 0.5).

<small>`pinneapple_tools/dataset_quality/vlm_curation.py` · 504 linhas · testes: 16 · palavras-chave no código (heurística; o texto acima diz o status real): placeholder/stub citado no docstring</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CurationRecord` | classe | `(id, label, confidence, reason, raw_text)` | Julgamento de uma sequência: id, rótulo, confiança, motivo e texto bruto do modelo. |
| `build_consistency_prompt` | função | `(pde_description, *, labels=DEFAULT_LABELS, extra_instructions=None) -> str` | Monta o prompt que pede ao VLM para julgar a consistência física de um clipe contra uma descrição da física dada. |
| `extract_json` | função | `(text) -> Optional[dict]` | Extrai o primeiro objeto JSON `{...}` de um texto livre; devolve None em vez de levantar erro. |
| `normalize_label` | função | `(label, *, labels=DEFAULT_LABELS, fallback=None) -> str` | Normaliza um rótulo possivelmente malformado para o conjunto fechado de rótulos, com fallback no rótulo do meio. |
| `clamp_confidence` | função | `(x, *, lo=0.0, hi=1.0) -> float` | Converte para float e limita a `[lo, hi]`; valor ilegível vira o ponto médio. |
| `parse_curation_response` | função | `(seq_id, text, *, labels=DEFAULT_LABELS, reason_max_chars=300, raw_text_max_chars=1200) -…)` | Converte uma geração do VLM num `CurationRecord` sem nunca levantar erro. |
| `load_vlm` | função | `(model_name, *, device=None, dtype=None, model_cls=None, attn_implementation='sdpa')` | Carrega um modelo de chat de visão-linguagem do Hugging Face e seu processador (precisa do extra `vlm_curation`). |
| `curate_sequence` | função | `(model, processor, clip_path, prompt, *, seq_id='', fps=4, max_new_tokens=512, labels=DEF…)` | Roda uma geração do VLM sobre um clipe e interpreta o resultado. |
| `curate_dataset` | função | `(index, *, out_dir, model_name='nvidia/Cosmos-Reason2-2B', model=None, processor=None, mo…)` | Faz a curadoria de todas as sequências de um índice com um VLM e grava metadados por sequência e agregados. |
| `filter_curated` | função | `(index, curation_records, *, keep_labels=('consistent',), min_confidence=0.55, id_field='…)` | Filtra o índice para as entradas com rótulo aceito e confiança mínima, copiando os dados opcionalmente. |

#### `pinneapple_tools.hpo_experiments`

Pesquisador automático de literatura (antigo `pinneapple_researcher`, apesar do nome "hpo"): descobre artigos e repositórios sobre um tema, monta uma base de conhecimento, extrai problema/solução com LLM e tenta reproduzir. [experimental] [sem teste]

<small>`pinneapple_tools/hpo_experiments/__init__.py` · 14 linhas · **sem teste direto**</small>

#### `pinneapple_tools.hpo_experiments.agents`

Pacote dos agentes LLM do pesquisador (extrator, reprodutor, verificador). [sem teste]

<small>`pinneapple_tools/hpo_experiments/agents/__init__.py` · 6 linhas · **sem teste direto**</small>

#### `pinneapple_tools.hpo_experiments.agents.extractor`

Agente que usa um LLM para extrair problema e solução estruturados dos artefatos da base de conhecimento. [sem teste]

<small>`pinneapple_tools/hpo_experiments/agents/extractor.py` · 101 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ExtractorAgent` | classe | `(provider)` | Envia trechos de um artigo ou repositório ao provedor LLM e devolve um `ExtractedProblemSolution`. |

#### `pinneapple_tools.hpo_experiments.agents.prompts`

Textos de sistema (prompts) dos agentes de extração, reprodução e verificação. [sem teste]

<small>`pinneapple_tools/hpo_experiments/agents/prompts.py` · 125 linhas · **sem teste direto**</small>

#### `pinneapple_tools.hpo_experiments.agents.reproducer`

Agente que pede ao LLM os arquivos de um projeto de reprodução a partir de um item e de trechos da base. [sem teste]

<small>`pinneapple_tools/hpo_experiments/agents/reproducer.py` · 98 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ReproduceResult` | classe | `(project_dir, run_instructions)` | Resultado da reprodução: pasta do projeto gerado e instruções de execução. |
| `ReproducerAgent` | classe | `(provider)` | Gera, via LLM, um projeto de reprodução numa pasta de saída. |

#### `pinneapple_tools.hpo_experiments.agents.verifier`

Agente que lê a árvore do projeto e os logs de execução e propõe diagnóstico e patches. [sem teste]

<small>`pinneapple_tools/hpo_experiments/agents/verifier.py` · 75 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `VerifyResult` | classe | `(status, diagnosis, patches, notes)` | Resposta do verificador: status, diagnóstico, lista de patches e notas. |
| `VerifierAgent` | classe | `(provider)` | Pede ao LLM um diagnóstico de por que o projeto não roda e patches para corrigi-lo. |

#### `pinneapple_tools.hpo_experiments.config`

Configuração do pesquisador: tema, número de artigos e repositórios, pesos do ranking, filtros do GitHub e arXiv, pastas e chunking. [sem teste]

<small>`pinneapple_tools/hpo_experiments/config.py` · 42 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ResearcherConfig` | classe | `(topic, k_papers, k_repos, w_lexical, w_signals, w_freshness, w_quality, github_min_stars…)` | Dataclass com o tema da busca, quantidades, pesos heurísticos do ranking, filtros e diretórios de saída e cache. |

#### `pinneapple_tools.hpo_experiments.kb`

Pacote da base de conhecimento (chunking e armazenamento). [sem teste]

<small>`pinneapple_tools/hpo_experiments/kb/__init__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_tools.hpo_experiments.kb.chunking`

Divide texto em pedaços com sobreposição para indexação. [sem teste]

<small>`pinneapple_tools/hpo_experiments/kb/chunking.py` · 29 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TextChunk` | classe | `(chunk_id, text)` | Um pedaço de texto com seu identificador. |
| `chunk_text` | função | `(text, *, chunk_chars=1800, overlap=200) -> List[TextChunk]` | Corta um texto em pedaços de tamanho fixo em caracteres, com sobreposição entre eles. |

#### `pinneapple_tools.hpo_experiments.kb.store`

Armazenamento de uma execução da base: manifesto, artefatos (artigos e repositórios) e `chunks.jsonl` para RAG. [sem teste]

<small>`pinneapple_tools/hpo_experiments/kb/store.py` · 142 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `KBStore` | classe | `(*, out_dir)` | Persiste uma execução da base: cria a pasta, grava manifesto, textos dos artefatos, pedaços em JSONL e o índice. |
| `slugify` | função | `(s) -> str` | Converte um texto num nome seguro para arquivo ou pasta. |
| `ensure_dir` | função | `(p) -> str` | Cria a pasta se não existir e devolve o caminho. |

#### `pinneapple_tools.hpo_experiments.models`

Modelos de dados do pesquisador: itens ranqueados, resultado da descoberta, artefatos e índice da base, e problema/solução extraído. [sem teste]

<small>`pinneapple_tools/hpo_experiments/models.py` · 75 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RankedItem` | classe | `(type, id, title, url, score, why, meta)` | Um artigo ou repositório ranqueado: tipo, id, título, URL, pontuação, justificativa e metadados. |
| `DiscoveryResult` | classe | `(topic, papers, repos, meta)` | Resultado da descoberta: tema, listas de artigos e de repositórios ranqueados e metadados. |
| `KBArtifact` | classe | `(type, id, title, url, path, meta)` | Um artefato (artigo ou repositório) persistido dentro de uma execução da base. |
| `KBIndex` | classe | `(topic, run_dir, chunks_path, manifest_path)` | Índice leve que aponta para a pasta da execução, o JSONL de pedaços e o manifesto. |
| `ExtractedProblemSolution` | classe | `(source_type, source_id, title, problem, solution, equations, data_requirements, training…)` | Saída do `ExtractorAgent`: problema, solução, equações, requisitos de dados, receita de treino, métricas e limitações. |

#### `pinneapple_tools.hpo_experiments.pipelines`

Pacote dos pipelines do pesquisador (descoberta, base, extração, reprodução, exportação). [sem teste]

<small>`pinneapple_tools/hpo_experiments/pipelines/__init__.py` · 8 linhas · **sem teste direto**</small>

#### `pinneapple_tools.hpo_experiments.pipelines.build_kb`

Pipeline que baixa artigos (PDF do arXiv) e arquivos-chave de repositórios e grava tudo no `KBStore`. [sem teste]

<small>`pinneapple_tools/hpo_experiments/pipelines/build_kb.py` · 141 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_kb` | função | `(cfg, discovery) -> KBIndex` | Monta a base de conhecimento a partir do resultado da descoberta e devolve o `KBIndex`. |

#### `pinneapple_tools.hpo_experiments.pipelines.discover`

Pipeline de descoberta: busca artigos no arXiv e repositórios no GitHub sobre o tema e os ranqueia por heurísticas. [sem teste]

<small>`pinneapple_tools/hpo_experiments/pipelines/discover.py` · 201 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `discover` | função | `(cfg) -> DiscoveryResult` | Busca e ranqueia artigos e repositórios para a configuração dada e devolve um `DiscoveryResult`. |

#### `pinneapple_tools.hpo_experiments.pipelines.export_problemdesign`

Exporta soluções extraídas num esquema estilo ProblemDesign em JSON; o arquivo complementar só é escrito quando `pinneapple_problemdesign` pode ser importado. [sem teste]

<small>`pinneapple_tools/hpo_experiments/pipelines/export_problemdesign.py` · 169 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `to_problemdesign_dict` | função | `(x) -> Dict[str, Any]` | Converte um problema/solução extraído num dicionário estável (fonte, problema, solução, benchmark, pontos fortes e fracos, alternativas, melhorias). |
| `export_problemdesign` | função | `(items, *, out_path) -> str` | Grava os achados em JSON e, se `pinneapple_problemdesign` estiver disponível, um `*_pinneapple_spec.json` complementar. |

#### `pinneapple_tools.hpo_experiments.pipelines.extract_problem_solutions`

Pipeline que roda o `ExtractorAgent` sobre os itens da base. [sem teste]

<small>`pinneapple_tools/hpo_experiments/pipelines/extract_problem_solutions.py` · 39 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `extract_problem_solutions` | função | `(*, kb_index, out_dir=None, max_items=20) -> List[ExtractedProblemSolution]` | Extrai até `max_items` pares problema/solução da base e os grava opcionalmente numa pasta. |

#### `pinneapple_tools.hpo_experiments.pipelines.reproduce`

Pipeline de reprodução: gera um projeto via LLM, compila, roda `run.py`/`main.py` e aplica patches do verificador em laço. [sem teste]

<small>`pinneapple_tools/hpo_experiments/pipelines/reproduce.py` · 79 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `reproduce` | função | `(*, item, kb_index_dir, out_dir=None, max_fix_iters=2) -> str` | Gera e tenta fazer rodar um projeto de reprodução de um item, com até `max_fix_iters` rodadas de correção, e devolve a pasta. |

#### `pinneapple_tools.hpo_experiments.providers`

Pacote dos provedores de LLM (só Gemini). [sem teste]

<small>`pinneapple_tools/hpo_experiments/providers/__init__.py` · 4 linhas · **sem teste direto**</small>

#### `pinneapple_tools.hpo_experiments.providers.gemini_provider`

Provedor LLM que chama a API Gemini (`generateContent`) diretamente, com modo JSON, ou via um gateway legado. [sem teste]

<small>`pinneapple_tools/hpo_experiments/providers/gemini_provider.py` · 240 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GeminiProviderConfig` | classe | `(api_base, api_version, model, api_key, timeout_s, gateway_base_url, gateway_route)` | Configuração imutável do provedor: base e versão da API, modelo, chave, timeout e dados do gateway opcional. |
| `GeminiProvider` | classe | `(*, api_key=None, model=None, api_base=None, api_version=None, timeout_s=None, gateway_ba…)` | Cliente Gemini com `generate`, que aceita um esquema JSON para respostas estruturadas; precisa de chave de API e de `requests`. |

#### `pinneapple_tools.hpo_experiments.ranking`

Heurísticas de ranking: sobreposição léxica, frescor por decaimento exponencial e normalização. [sem teste]

<small>`pinneapple_tools/hpo_experiments/ranking.py` · 71 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `tokenize` | função | `(text) -> List[str]` | Quebra um texto em tokens minúsculos. |
| `lexical_overlap_score` | função | `(query, doc) -> float` | Fração dos tokens únicos da consulta presentes no documento (0 a 1). |
| `safe_parse_iso8601` | função | `(dt) -> Optional[datetime]` | Converte uma data ISO 8601 em `datetime`, devolvendo None se falhar. |
| `freshness_score` | função | `(dt, *, half_life_days=365.0) -> float` | Pontuação de frescor por decaimento exponencial: 1 hoje, cerca de 0.5 na meia-vida. |
| `normalize` | função | `(xs) -> List[float]` | Normaliza uma lista de números para o intervalo [0, 1]. |
| `sort_ranked` | função | `(items) -> List[RankedItem]` | Ordena itens ranqueados pela pontuação, do maior para o menor. |

#### `pinneapple_tools.hpo_experiments.sources`

Pacote das fontes externas (arXiv e GitHub). [sem teste]

<small>`pinneapple_tools/hpo_experiments/sources/__init__.py` · 12 linhas · **sem teste direto**</small>

#### `pinneapple_tools.hpo_experiments.sources.arxiv_pdf`

Baixa o PDF de um artigo do arXiv, extrai o texto e o divide em seções por heurística. [sem teste]

<small>`pinneapple_tools/hpo_experiments/sources/arxiv_pdf.py` · 79 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `arxiv_pdf_url` | função | `(arxiv_id, use_export_subdomain=True) -> str` | Monta a URL do PDF de um id do arXiv. |
| `download_pdf` | função | `(url, out_path, timeout_s=60) -> str` | Baixa um PDF para um caminho local. |
| `extract_pdf_text` | função | `(pdf_path) -> str` | Extrai o texto de um PDF. |
| `split_sections` | função | `(text) -> Dict[str, str]` | Divide o texto em seções procurando títulos por palavras-chave; o docstring avisa que não é perfeito. |

#### `pinneapple_tools.hpo_experiments.sources.arxiv_source`

Busca mínima de artigos na API Atom do arXiv. [sem teste]

<small>`pinneapple_tools/hpo_experiments/sources/arxiv_source.py` · 90 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ArxivPaper` | classe | `(arxiv_id, title, summary, url, published, authors, categories)` | Metadados de um artigo do arXiv: id, título, resumo, URL, data, autores e categorias. |
| `ArxivSource` | classe | `()` | Faz buscas na API Atom do arXiv e devolve `ArxivPaper`. |

#### `pinneapple_tools.hpo_experiments.sources.github_fetcher`

Lê README, árvore de arquivos e arquivos-chave de um repositório pela API REST do GitHub. [sem teste]

<small>`pinneapple_tools/hpo_experiments/sources/github_fetcher.py` · 179 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RepoFile` | classe | `(path, sha, size, url)` | Um arquivo do repositório: caminho, sha, tamanho e URL. |
| `GithubFetcher` | classe | `(token=None, api_base='https://api.github.com')` | Cliente da API REST do GitHub para branch padrão, README, árvore recursiva, texto de arquivos e escolha de arquivos-chave. |

#### `pinneapple_tools.hpo_experiments.sources.github_source`

Busca de repositórios pela API de busca do GitHub. [sem teste]

<small>`pinneapple_tools/hpo_experiments/sources/github_source.py` · 90 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GithubRepo` | classe | `(full_name, title, url, description, stars, forks, language, updated_at, license)` | Metadados de um repositório: nome, URL, descrição, estrelas, forks, linguagem, data de atualização e licença. |
| `GithubSource` | classe | `(token=None)` | Busca repositórios no GitHub; o token é opcional, mas recomendado (`GITHUB_TOKEN`). |

#### `pinneapple_tools.hpo_experiments.utils`

Pacote de utilitários do pesquisador. [sem teste]

<small>`pinneapple_tools/hpo_experiments/utils/__init__.py` · 10 linhas · **sem teste direto**</small>

#### `pinneapple_tools.hpo_experiments.utils.verify_runtime`

Verificação de execução de um projeto: árvore de arquivos, `py_compile` e execução rápida de `run.py` ou `main.py` em subprocesso. [sem teste]

<small>`pinneapple_tools/hpo_experiments/utils/verify_runtime.py` · 74 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RunResult` | classe | `(ok, step, returncode, stdout, stderr)` | Resultado de uma etapa: ok, nome da etapa, código de retorno, stdout e stderr. |
| `project_tree` | função | `(cwd, max_files=400) -> str` | Lista em texto os arquivos do projeto, até um limite. |
| `py_compile_all` | função | `(cwd) -> RunResult` | Compila todos os `.py` do projeto com `python -m py_compile`. |
| `smoke_run` | função | `(cwd, timeout_s=180) -> RunResult` | Executa `run.py` ou `main.py` com timeout; se não houver ponto de entrada, devolve ok sem rodar nada. |

#### `pinneapple_tools.model_evaluation`

Pacote de avaliação de modelos; o `__init__` está vazio e o conteúdo fica em `metrics` e `visualization`. [sem teste direto]

<small>`pinneapple_tools/model_evaluation/__init__.py` · 1 linhas · **sem teste direto**</small>

#### `pinneapple_tools.model_evaluation.metrics`

Métricas genéricas sobre arrays simples: regressão, perfis por eixo, MAPE, piores pontos, intervalo bootstrap, sensibilidades e calibração.

<small>`pinneapple_tools/model_evaluation/metrics.py` · 262 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `regression_metrics` | função | `(y_true, y_pred, field_names=None) -> Dict[str, Any]` | RMSE, MAE, erro relativo e R², no total e por coluna de saída. |
| `axis_binned_profile` | função | `(coord, field, n_bins=40) -> Dict[str, np.ndarray]` | Agrupa um campo em faixas de uma coordenada e tira a média em cada faixa (por exemplo, perfil de velocidade na direção normal à parede). |
| `mape` | função | `(y_true, y_pred, eps=1e-08) -> float` | Erro percentual absoluto médio, como fração, com `eps` contra alvos perto de zero. |
| `worst_n_points` | função | `(coords, abs_residuals, field_names, n=20) -> List[Dict[str, Any]]` | Índices e coordenadas dos `n` pontos com maior erro absoluto somado entre os campos. |
| `bootstrap_confidence_interval` | função | `(residuals, n_boot=1000, ci=90.0, seed=42) -> Dict[str, Any]` | Intervalo de confiança bootstrap da média dos resíduos. |
| `one_at_a_time_sensitivity` | função | `(model_fn, base_inputs, param_columns, perturb_frac=0.1, default_nominal=1.0) -> Dict[str…)` | Sensibilidade de primeira ordem, uma coluna por vez, por diferença finita em torno da média. |
| `gradient_sensitivity` | função | `(model, x, n_out=None) -> Dict[str, Any]` | Sensibilidade exata por autograd: média de \|d saída / d entrada\| sobre as linhas de `x` (precisa de um `torch.nn.Module` diferenciável). |
| `calibration_curve` | função | `(y_true, y_pred, residual_std, levels=(0.5, 0.7, 0.8, 0.9, 0.95)) -> Dict[str, Any]` | Cobertura empírica de um intervalo gaussiano com `residual_std` em cada nível de confiança nominal. |

#### `pinneapple_tools.model_evaluation.visualization`

Gráficos genéricos de avaliação em matplotlib: histogramas, real vs previsto e curva de calibração. [sem teste]

<small>`pinneapple_tools/model_evaluation/visualization.py` · 95 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_value_histograms` | função | `(fields, out_dir, bins=40) -> List[str]` | Um histograma por campo dos valores brutos (perfil de dataset sem verdade de referência). |
| `plot_error_histograms` | função | `(abs_residuals, field_names, out_dir, bins=40) -> List[str]` | Um histograma por campo de saída de \|previsão - verdade\|. |
| `plot_real_vs_predicted` | função | `(y_true, y_pred, out_path) -> str` | Dispersão previsto vs real com a reta y=x. |
| `plot_calibration_curve` | função | `(levels, empirical_coverage, out_path) -> str` | Cobertura empírica vs nominal contra a diagonal ideal. |

#### `pinneapple_tools.model_export`

Exportação de modelos `nn.Module` para TorchScript e ONNX e de arrays de coordenadas/campos para CSV e NPZ. [sem teste]

<small>`pinneapple_tools/model_export/__init__.py` · 197 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `export_torchscript` | função | `(model, path, example_input=None) -> str` | Exporta qualquer `nn.Module` para TorchScript (trace com `example_input`, ou script se não houver exemplo ou o trace falhar). |
| `export_onnx` | função | `(model, path, example_input, input_names=None, output_names=None, opset_version=17, dynam…)` | Exporta qualquer `nn.Module` para ONNX com nomes de entrada/saída, opset e eixos dinâmicos. |
| `export_csv` | função | `(coords, fields, path, *, coord_names=None, field_names=None) -> str` | Grava coordenadas e campos num arquivo CSV com cabeçalho. |
| `export_npz` | função | `(coords, fields, path, *, coord_names=None, field_names=None) -> str` | Grava coordenadas e campos num `.npz` comprimido do NumPy. |

#### `pinneapple_tools.model_services`

Modelos prontos do PINNeAPPle como microserviços locais no formato NVIDIA NIM (`/v1/health`, `/v1/models`, `/v1/infer`, `/v1/chat/completions`), cada resposta com o relatório de confiança.

<small>`pinneapple_tools/model_services/__init__.py` · 27 linhas · testes: 39</small>

#### `pinneapple_tools.model_services.__main__`

Linha de comando `python -m pinneapple_tools.model_services serve` (host, porta, `--no-zoo`, `--api-key`). [sem teste]

<small>`pinneapple_tools/model_services/__main__.py` · 27 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `main` | função | `(argv=None) -> int` | Lê os argumentos, monta o registro padrão e roda o servidor HTTP. |

#### `pinneapple_tools.model_services.client`

Cliente único para Model Services e NVIDIA NIM (mesmas rotas): saúde, modelos, inferência e chat. [sem teste direto]

<small>`pinneapple_tools/model_services/client.py` · 77 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ServiceError` | classe | `(status, message)` | Erro HTTP de um serviço, com status e mensagem. |
| `ModelServiceClient` | classe | `(base_url, api_key=None, timeout=60.0)` | Cliente HTTP de um endpoint `/v1` (local ou NIM com `api_key`): `ready`, `models`, `infer`, `chat`. |

#### `pinneapple_tools.model_services.grid`

Serviços de operador em grade: um modelo de campo treinado (por exemplo um FNO do PhysicsNeMo) e o solver de referência na mesma rota `/v1/infer`, com as checagens de Darcy baseadas em resíduo.

<small>`pinneapple_tools/model_services/grid.py` · 50 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `darcy_trust` | função | `(a, u, f=1.0, envelope=None) -> dict` | Roda as checagens de Darcy (`pinneapple_physics.darcy_checks`) sobre coeficiente e solução e devolve decisão e checagens. |
| `grid_operator_service` | função | `(predictor, name, *, envelope=None, f=1.0, card=None, task='Darcy-type field operator') -…)` | Embala um preditor de campo `a -> u` como `ModelService` com a confiança de Darcy e o envelope de treino. |
| `darcy_solver_service` | função | `(name='pinneapple-darcy-fv', f=1.0) -> ModelService` | O solver de volumes finitos de Darcy como serviço, com as mesmas checagens: o fallback quando o surrogate não é aprovado. |

#### `pinneapple_tools.model_services.registry`

`ModelService` (função com card e função de confiança) e o registro dos serviços embutidos: zoo, trust card e `pinneapple-intent`. [sem teste direto]

<small>`pinneapple_tools/model_services/registry.py` · 200 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ModelService` | classe | `(id, kind, task, predict, trust, card, version, calls)` | Um serviço: id, tipo (`infer` ou `chat`), tarefa, função de previsão, função de confiança e card; `run` devolve saídas, confiança, avisos e tempos. |
| `Registry` | classe | `(services=())` | Registro de serviços por id (`add`, `get`, `list`). |
| `jsonable` | função | `(o)` | Converte recursivamente arrays e escalares NumPy em tipos JSON, com não finitos virando None. |
| `zoo_service` | função | `(name) -> ModelService` | Serviço de um modelo do zoo: previsão do modelo e confiança vinda de `pp.zoo.validate` nos mesmos inputs. |
| `trust_card_service` | função | `() -> ModelService` | Serviço de checagens físicas: recebe eixos, valores e EDP e devolve o trust card do campo. |
| `intent_service` | função | `() -> ModelService` | Serviço `pinneapple-intent`: o extrator determinístico de `pinneapple_design.intent` atrás de uma rota de chat no estilo OpenAI. |
| `default_registry` | função | `(zoo=True) -> Registry` | Monta o registro com os serviços embutidos (o zoo é opcional). |
| `dumps` | função | `(o) -> bytes` | Serializa um objeto para bytes JSON usando `jsonable`. |
| `infer` | função | `(model, inputs_json) -> dict` | Faz uma requisição a um serviço embutido no próprio processo, sem servidor, e devolve a mesma resposta da rota HTTP. |

#### `pinneapple_tools.model_services.server`

Servidor HTTP dos serviços só com a biblioteca padrão (`ThreadingHTTPServer`), com as rotas do NIM e chave de API opcional. [sem teste direto]

<small>`pinneapple_tools/model_services/server.py` · 116 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `make_server` | função | `(registry, host='127.0.0.1', port=8000, api_key=None)` | Cria o `ThreadingHTTPServer` que serve o registro; com `api_key` exige `Authorization: Bearer`. |
| `serve_in_thread` | função | `(registry, host='127.0.0.1', port=0, api_key=None)` | Inicia o servidor numa thread e devolve `(server, base_url)`. |

#### `pinneapple_tools.nvidia`

Cola entre o PINNeAPPle e a pilha NVIDIA (PhysicsNeMo, Omniverse/OpenUSD, Isaac Sim, USD BRep): o PINNeAPPle fornece física, solvers de referência e checagens; o NVIDIA treina, mostra ou simula. Todas as dependências NVIDIA são opcionais.

<small>`pinneapple_tools/nvidia/__init__.py` · 24 linhas · **sem teste direto**</small>

#### `pinneapple_tools.nvidia._brep_worker`

Processo auxiliar isolado que constrói uma peça a partir de uma receita com `usd_brep` e a mede; roda no interpretador que tiver `usd-brep` (wheel CPython 3.12). [sem teste]

<small>`pinneapple_tools/nvidia/_brep_worker.py` · 157 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `main` | função | `(req_path, out_path)` | Lê a requisição JSON, constrói e mede a peça e grava o resultado no arquivo de saída. |

#### `pinneapple_tools.nvidia.brep`

Geometria exata com NVIDIA USD BRep: propriedades de massa do sólido exato, curvatura das faces, tesselações em várias tolerâncias com desvio medido e checagens de confiança da geometria.

<small>`pinneapple_tools/nvidia/brep.py` · 247 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PartRecipe` | classe | `(ops, result)` | Receita de peça como lista de operações (caixa, cilindro, esfera, união, diferença, transformações, filetes, extrusão), serializável em dicionário. |
| `BRepReport` | classe | `(raw, meshes, seconds)` | Relatório da avaliação de uma peça: dados brutos, malhas por tolerância e tempo; dá raio mínimo de curvatura e acesso às malhas. |
| `brep_python` | função | `() -> str \| None` | Caminho do interpretador que tem `usd_brep`: o atual ou o de `$PINNEAPPLE_BREP_PYTHON`, se existir. |
| `brep_available` | função | `() -> bool` | Diz se há um interpretador com `usd_brep` disponível. |
| `evaluate_part` | função | `(recipe, *, chord_tolerances=(0.001, 0.0001), density=1.0, angle_deg=25.0, max_edge_lengt…)` | Constrói a receita com USD BRep (via processo auxiliar) e mede massa, curvatura e tesselações nas tolerâncias pedidas. |
| `geometry_checks` | função | `(rep, *, element_size=None, points_per_radius=4.0, volume_rtol=0.001) -> list[dict[str, A…)` | Checagens da geometria entregue a um solver ou surrogate: sólido manifold, erro de corda abaixo da tolerância, tamanho de malha e volume. |
| `planar_face_mesh` | função | `(V, F, *, axis=2, value, atol=1e-09) -> tuple[np.ndarray, np.ndarray]` | Extrai os triângulos de uma tesselação que estão num plano `x[axis] = value`, como malha 2D. |
| `vertex_normals` | função | `(V, F) -> np.ndarray` | Normais unitárias por vértice ponderadas pela área dos triângulos. |
| `mesh_thickness` | função | `(V, F, *, normals=None, eps=1e-07, chunk=256) -> np.ndarray` | Espessura de parede em cada vértice de uma malha fechada por lançamento de raio (Möller–Trumbore) ao longo da normal interna. |
| `point_mesh_distance` | função | `(P, V, F, chunk=128) -> np.ndarray` | Distância de cada ponto ao vértice ou centróide de triângulo mais próximo, um limite superior barato da distância à malha. |

#### `pinneapple_tools.nvidia.diagram`

Desenha diagramas de arquitetura de um pipeline PINNeAPPle + NVIDIA só com matplotlib (sem graphviz).

<small>`pinneapple_tools/nvidia/diagram.py` · 118 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Block` | classe | `(key, label, owner, col, row, ran)` | Um bloco do diagrama: chave, rótulo, dono (PINNeAPPle ou NVIDIA), posição na grade e se a etapa rodou. |
| `architecture_diagram` | função | `(title, blocks, edges, path, *, subtitle='', width=2.6, height=1.0, dpi=150) -> str` | Desenha blocos e arestas (com rótulo opcional) e salva um PNG. |

#### `pinneapple_tools.nvidia.figures`

Figuras de resultado comuns aos exemplos NVIDIA: painéis de campos, gráfico de checagens e curvas de treino. [sem teste]

<small>`pinneapple_tools/nvidia/figures.py` · 140 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `field_panels` | função | `(fields, titles, path, *, cmaps=None, extent=(0, 1, 0, 1), suptitle='', units=None, share…)` | Desenha vários campos 2D lado a lado, com faixas de cor compartilhadas opcionais, e salva a figura. |
| `check_rows` | função | `(checks) -> list[dict[str, Any]]` | Normaliza checagens de vários formatos em linhas `{name, value, limit, passed, detail}`. |
| `checks_figure` | função | `(checks, path, *, title='Physics checks', decision='', dpi=130) -> str` | Gráfico de barras valor/limite por checagem em escala log, com linha tracejada no limite e decisão no título. |
| `curves_figure` | função | `(panels, path, *, suptitle='', dpi=130) -> str` | Figura com vários painéis de curvas (séries, escala log opcional, linhas horizontais). |

#### `pinneapple_tools.nvidia.heatsink`

Projetos de dissipador de aletas planas avaliados de duas formas: sólido exato do USD BRep (volume, área, massa) e modelo térmico e hidráulico de forma fechada do PINNeAPPle.

<small>`pinneapple_tools/nvidia/heatsink.py` · 113 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Duty` | classe | `(power_w, t_ambient_c, air_velocity_m_s, source, base_t, length, width, material, root_fi…)` | Condição de operação do dissipador: potência, temperatura ambiente, velocidade do ar, fonte, base, dimensões, material e filete de raiz. |
| `evaluate_design` | função | `(n_fins, fin_height, fin_t, duty=None, *, brep=True, chord_tolerances=()) -> dict` | Avalia um projeto: resultado térmico/hidráulico de forma fechada e, com `brep`, volume, área e massa do sólido exato. |
| `sample_designs` | função | `(n, *, n_range=(6, 20), h_range=(0.015, 0.045), t_range=(0.001, 0.003), min_gap=0.002, wi…)` | Amostra projetos por hipercubo latino dentro das faixas, com folga mínima entre aletas. |
| `design_table` | função | `(designs, duty=None, *, brep=True) -> list[dict]` | Avalia uma lista de projetos e devolve uma linha por projeto. |
| `verify` | função | `(n_fins, fin_height, fin_t, duty=None, **kw) -> dict` | Avaliação decisiva (forma fechada + sólido exato) de um projeto proposto por um surrogate. |
| `surrogate_service` | função | `(net, X_train, *, name=…, t_ambient=25.0)` | Embala um surrogate de dissipador como Model Service do PINNeAPPle, com relatório de confiança que sinaliza projetos fora da região de treino. |

#### `pinneapple_tools.nvidia.isaac`

Robôs prontos para Isaac Sim em OpenUSD (`UsdPhysics`), suas checagens e o script do Isaac Sim que reproduz um movimento; escrito para a API 5.x, que o 6.x ainda mantém como depreciada.

<small>`pinneapple_tools/nvidia/isaac.py` · 239 linhas · testes: 39 · palavras-chave no código (heurística; o texto acima diz o status real): shim ou obsoleto</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `author_articulation` | função | `(stage, spec) -> str` | Escreve uma articulação (corpos, juntas, massas) em `/World/<nome>` de um stage USD e devolve o caminho do prim. |
| `validate_physics` | função | `(path, *, expect_bodies=None, expect_joints=None) -> dict` | `validate_stage` mais a leitura `UsdPhysics` e as checagens de massa e juntas de que o simulador depende. |
| `isaac_replay_script` | função | `(usd_path, robot_prim, traj_npz, out_npz, *, mode='effort', physics_dt=1.0 / 600.0, headl…)` | Gera o script autônomo do Isaac Sim que reproduz uma trajetória (ângulos e torques) e grava o resultado; exige Isaac Sim com GPU RTX. |
| `compare_replay` | função | `(npz, t, q, tau) -> dict` | Compara a reprodução do Isaac Sim com o PINNeAPPle: erro máximo de rastreamento e erro RMS de esforço relativo ao torque de pico. |

#### `pinneapple_tools.nvidia.omniverse`

Resultados do PINNeAPPle como stages OpenUSD para o Omniverse (escritos e checados com `usd-core`): campos como primvars, validador de stage e pré-visualização lida de volta do disco.

<small>`pinneapple_tools/nvidia/omniverse.py` · 646 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FieldStage` | classe | `(path, title='PINNeAPPle result', *, meters_per_unit=1.0, fps=24.0)` | Stage USD que guarda campos do PINNeAPPle (malhas, curvas, pontos, amostras no tempo, animação rígida e veredito de confiança) e é gravado com `save()`. |
| `require_pxr` | função | `()` | Importa `pxr` (OpenUSD) ou falha com dica de instalação. |
| `field_colors` | função | `(values, cmap='inferno', vrange=None) -> np.ndarray` | Cores RGB lineares de valores através de um mapa de cores do matplotlib (sRGB convertido para linear, como o USD espera). |
| `grid_surface` | função | `(X, Y, Z=None) -> tuple[np.ndarray, np.ndarray]` | Triângulos de uma grade estruturada de pontos (vértices e faces). |
| `extrude_profile` | função | `(xy, length, n=2, axis=1) -> tuple[np.ndarray, np.ndarray]` | Extruda um perfil 2D fechado ao longo de um eixo e devolve a parede lateral como triângulos. |
| `tube` | função | `(path, radius, n_around=12) -> tuple[np.ndarray, np.ndarray]` | Tubo de raio dado em torno de uma polilinha 3D, como malha de triângulos. |
| `validate_stage` | função | `(path, *, required_primvars=None, compliance=True) -> dict[str, Any]` | Reabre um arquivo USD e o checa (prims, malhas, curvas, amostras no tempo, eixo, unidade, primvars obrigatórios, conformidade). |
| `usd_validators` | função | `(stage) -> dict[str, Any]` | Roda todos os validadores registrados no OpenUSD (`pxr.UsdValidation`), inclusive os de `UsdPhysics`, sobre um stage. |
| `read_stage_geometry` | função | `(path, time=None, exclude=()) -> list[dict[str, Any]]` | Lê as malhas e curvas de um arquivo USD como arrays, com as cores `displayColor` num tempo dado. |
| `render_stage_png` | função | `(path, png, *, time=None, elev=28, azim=-60, title='', colorbar=None, size=(7.5, 5.5), dp…)` | Desenha em PNG (matplotlib) as malhas e curvas de um USD lido do disco, com suas cores. |
| `kit_render_script` | função | `(usd_path, png_path, *, frames=60) -> str` | Gera um script Python para rodar no Omniverse Kit/USD Composer que abre o stage e grava uma captura RTX; exige Omniverse com GPU RTX. |
| `render_stage_gif` | função | `(path, gif, times, *, fps=6.0, label=None, **kw) -> str` | GIF animado de um stage com amostras no tempo, um `render_stage_png` por tempo. |
| `rasterize_topdown` | função | `(path, H, W, *, bounds=None, time=None, exclude=(), background=(0.0, 0.0, 0.0), view='top…)` | Vista ortográfica (por padrão de cima) das malhas de um USD como imagem RGB (H, W, 3). |

#### `pinneapple_tools.nvidia.parts`

Peças paramétricas (suporte em L, dissipador de aletas) como `PartRecipe`, com volumes de forma fechada para checar o sólido exato.

<small>`pinneapple_tools/nvidia/parts.py` · 68 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `l_bracket` | função | `(*, base=0.08, height=0.06, width=0.05, t=0.006, fillet=0.005, bolt_d=0.009, web_hole_d=0…)` | Receita de um suporte em L com filete interno, furos de parafuso na base e furo na alma. |
| `l_bracket_volume` | função | `(*, base=0.08, height=0.06, width=0.05, t=0.006, fillet=0.005, bolt_d=0.009, web_hole_d=0…)` | Volume de forma fechada do suporte em L. |
| `heat_sink` | função | `(*, n_fins, fin_height, fin_t, base_t=0.005, length=0.1, width=0.1, root_fillet=0.0) -> P…)` | Receita de um dissipador de aletas planas igualmente espaçadas sobre uma base. |
| `heat_sink_geometry` | função | `(*, n_fins, fin_height, fin_t, base_t=0.005, length=0.1, width=0.1, root_fillet=0.0) -> d…)` | Volume e áreas de forma fechada do dissipador (área molhada e área total). |

#### `pinneapple_tools.nvidia.physicsnemo`

Surrogates pequenos do PhysicsNeMo (`FullyConnected`) para física tabular (poucas entradas, poucas saídas), treinados e checados à maneira do PINNeAPPle.

<small>`pinneapple_tools/nvidia/physicsnemo.py` · 146 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MLPSurrogate` | classe | `(in_features, out_features, layer_size, num_layers, activation_fn, seed, x_lo, x_hi, y_lo…)` | Surrogate MLP do PhysicsNeMo com normalização pela caixa de treino: `fit`, `predict`, `in_box` (domínio de validade), `save` e `load`. |
| `sign_violations` | função | `(f, axes, axis, output, sign=1, *, tol=0.0) -> float` | Fração das células de uma grade onde a diferença finita de uma saída ao longo de uma entrada tem o sinal oposto ao esperado (checagem de monotonia). |
| `relative_errors` | função | `(pred, ref) -> np.ndarray` | Erro RMS por saída relativo ao RMS da saída de referência. |

#### `pinneapple_tools.nvidia.runinfo`

Registra o que uma execução usou: versões dos pacotes NVIDIA e PINNeAPPle, hardware e tempos por etapa.

<small>`pinneapple_tools/nvidia/runinfo.py` · 89 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RunInfo` | classe | `(name='')` | Registro de execução com tempos por etapa (`step`), marcação de componentes que rodaram ou não (`mark`), e exportação para dicionário ou arquivo. |
| `environment` | função | `() -> dict[str, Any]` | Python, sistema, CPU, GPU (CUDA via torch, se houver) e versões dos pacotes relevantes. |

#### `pinneapple_tools.rea_bridge`

Estudo de implementações de referência com a ferramenta REA (Reverse Engineer Anything), sob uma trava de licença; é uma terceira fonte para o pipeline de pesquisa.

<small>`pinneapple_tools/rea_bridge/__init__.py` · 33 linhas · testes: 22</small>

#### `pinneapple_tools.rea_bridge.client`

Wrapper fino da CLI do REA: resolve o executável (`PINNEAPPLE_REA_BIN`, `rea` no PATH ou `npx`), roda comandos e interpreta o JSON, sem completar resultados. [sem teste direto]

<small>`pinneapple_tools/rea_bridge/client.py` · 216 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ReaUnavailable` | classe | `()` | Exceção: nenhum executável do REA foi encontrado. |
| `ReaError` | classe | `(message, *, returncode, stderr_tail)` | Exceção: o REA rodou mas saiu com erro ou não imprimiu JSON. |
| `ReaResult` | classe | `(command, payload, progress, stderr_tail)` | Uma invocação do REA: comando, payload JSON, eventos de progresso e final do stderr, com acesso ao id de evidência, SHA-256 e limitações. |
| `ReaClient` | classe | `(command=None, *, timeout_s=900.0, cwd=None)` | Roda comandos do REA (versão, doctor, análise de app JavaScript, layout de binário, inspeção de artefato) e interpreta a saída. |
| `resolve_rea_command` | função | `(env=None) -> List[str]` | Devolve o prefixo de comando que roda o REA ou levanta `ReaUnavailable`. |

#### `pinneapple_tools.rea_bridge.evidence`

Converte a evidência de análise JavaScript do REA num mapa compacto de funções, chamadas, timers e armazenamento, e localiza funcionalidades por palavra-chave. [sem teste direto]

<small>`pinneapple_tools/rea_bridge/evidence.py` · 222 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FunctionRef` | classe | `(node_id, name, module, start_line, end_line)` | Referência a uma função: nó, nome, módulo e intervalo de linhas. |
| `FeatureMap` | classe | `(evidence_id, subject_sha256, root, functions, calls, callers, timers, storage, limitatio…)` | Mapa de funcionalidades de uma aplicação: funções, grafo de chamadas, timers, armazenamento, limitações do REA e módulos de terceiros. |
| `FeatureHit` | classe | `(function, matched, snippet, callees, callers)` | Uma função encontrada por `locate`: termos casados, trecho do código, chamadas e chamadores. |
| `summarize_js_evidence` | função | `(evidence, *, vendor_globs=DEFAULT_VENDOR_GLOBS, include_vendor=False) -> FeatureMap` | Constrói um `FeatureMap` a partir de um payload de evidência `analyze_javascript_application`. |
| `locate` | função | `(fmap, keywords, *, root=None, max_hits=20, snippet_lines=40) -> List[FeatureHit]` | Funções cujo nome ou corpo menciona alguma palavra-chave, ordenadas pelo número de ocorrências. |
| `exported_functions_from_layout` | função | `(evidence) -> List[str]` | Lista as funções exportadas de um binário ELF a partir da evidência de `inspect-binary-layout`. |

#### `pinneapple_tools.rea_bridge.harvest`

Livro de registro JSON do que foi estudado, sob qual licença, e o que resultou; recusa achados `adapted` sob política `clean_room` ou `reference_only`. [sem teste direto]

<small>`pinneapple_tools/rea_bridge/harvest.py` · 126 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LicenseGateError` | classe | `()` | Exceção: um achado pede mais reuso do que a licença do projeto permite. |
| `Finding` | classe | `(title, summary, source_refs, status, method, pinneapple_refs, note)` | Um achado (ideia, técnica ou funcionalidade): onde está no original, status (`idea`, `adapted`, `reimplemented`, `exists`, `rejected`) e o que o PINNeAPPle fez. |
| `HarvestEntry` | classe | `(project, url, commit, licenses, policy, findings, rea_version, evidence_id, subject_sha2…)` | Um projeto estudado: URL, commit, licenças, política, achados e dados da evidência do REA. |
| `HarvestLedger` | classe | `(path)` | Arquivo JSON de entradas por nome de projeto, com `put`, `save` e `summary`. |

#### `pinneapple_tools.rea_bridge.licenses`

Trava de licença: classifica um projeto estudado em `adapt`, `clean_room` ou `reference_only` pela licença mais restritiva detectada. [sem teste direto]

<small>`pinneapple_tools/rea_bridge/licenses.py` · 179 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LicensePolicy` | classe | `(policy, licenses, attribution_required, reason, notes)` | Política de reuso de um projeto (licenças, atribuição exigida, motivo); `may_copy_code` diz se código pode ser adaptado. |
| `policy_for` | função | `(spdx_ids) -> LicensePolicy` | Política mais restritiva sobre uma lista de identificadores SPDX. |
| `detect_licenses` | função | `(root, *, max_bytes=200000) -> List[str]` | Identificadores SPDX reconhecidos nos arquivos de licença e nos manifestos (`package.json`, `pyproject.toml`, `setup.py`) de uma pasta. |
| `policy_for_path` | função | `(root) -> LicensePolicy` | Detecta as licenças de uma pasta e devolve a política correspondente. |

#### `pinneapple_tools.rea_bridge.tools`

Registra a ponte REA como ferramentas do `PhysicsToolRegistry` para o loop de agente de `pinneapple_llm`; as ferramentas REA só entram quando o executável existe. [sem teste direto]

<small>`pinneapple_tools/rea_bridge/tools.py` · 94 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `register_rea_tools` | função | `(registry) -> List[str]` | Registra `reference_license_policy` e, se o REA estiver disponível, `rea_analyze_javascript` e `rea_locate_feature`; devolve os nomes. |

#### `pinneapple_tools.reporting`

Relatórios: estilo de figuras e construtor de PDF (`paperkit`) e datasheet técnico (`datasheet`); não é importado automaticamente pelo pacote.

<small>`pinneapple_tools/reporting/__init__.py` · 27 linhas · testes: 5</small>

#### `pinneapple_tools.reporting.datasheet`

Datasheet técnico de um estudo, montado só a partir de dados (sem LLM): faixa de KPIs, seções numeradas, plano de execução, itens a confirmar; HTML, PDF via Chromium e Markdown.

<small>`pinneapple_tools/reporting/datasheet.py` · 570 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `KPI` | classe | `(label, value, unit, note, tone)` | Uma célula da faixa de números principais: rótulo, valor, unidade, nota e tom. |
| `Table` | classe | `(rows, caption, header, widths, align, emphasis, note)` | Tabela de duas ou mais colunas com cabeçalho, legenda, larguras, alinhamento e nota. |
| `Figure` | classe | `(path, caption, wide, max_width)` | Imagem com legenda, em largura total ou dividindo a linha. |
| `Note` | classe | `(text, kind, title)` | Caixa de destaque do tipo `info`, `ok`, `warn` ou `fail`. |
| `Step` | classe | `(title, detail, status, cost)` | Um passo do plano de execução: o que faz, por que existe, custo e situação. |
| `Section` | classe | `(title, number, lede, blocks)` | Seção numerada com abertura e blocos (parágrafos, notas, tabelas, figuras, linhas, equações, listas). |
| `Datasheet` | classe | `(title, subtitle, eyebrow, generated, verified_by, kpis_, sections, plan_, confirm_, limi…)` | O documento inteiro; `build` grava HTML e, se houver Chromium ou Playwright, PDF; também gera Markdown. |
| `find_chrome` | função | `() -> str \| None` | Procura um executável Chromium/Chrome capaz de imprimir HTML em PDF; None se não houver. |

#### `pinneapple_tools.reporting.paperkit`

Estilo único de figuras matplotlib (ordem de cores fixa, mapas `seq_blue` e `div_blue_red`) e `Paper`, um construtor de relatório A4 em PDF com reportlab (opcional).

<small>`pinneapple_tools/reporting/paperkit.py` · 429 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Paper` | classe | `(title, subtitle='', authors='PINNeAPPle Labs', affiliation=…, date=None, report_id='', f…)` | Relatório técnico A4 com reportlab/platypus: título, resumo, seções, equações, figuras e tabelas numeradas, referências. |
| `setup_mpl` | função | `() -> None` | Aplica o estilo de relatório ao `matplotlib.rcParams` e registra os mapas `SEQ` e `DIV`. |
| `savefig` | função | `(fig, path, **kwargs) -> str` | Salva a figura criando as pastas, fecha e devolve o caminho. |
| `fmt` | função | `(x, nd=3) -> str` | Formata um número de forma compacta para uma célula de tabela. |
| `sanitize` | função | `(text) -> str` | Troca sequências de sobrescritos/subscritos Unicode pela marcação `<super>`/`<sub>` do reportlab. |
| `para` | função | `(text, style, **kw)` | Cria um `Paragraph` do reportlab a partir do texto sanitizado. |

#### `pinneapple_tools.sandbox`

Pacote para executar scripts de solver não confiáveis em outro processo com limites de recurso; o próprio docstring avisa que não é um sandbox real.

<small>`pinneapple_tools/sandbox/__init__.py` · 11 linhas · **sem teste direto**</small>

#### `pinneapple_tools.sandbox.custom_solver_runner`

Roda `solve(params)` de um script do usuário num subprocesso com `RLIMIT_AS`, `RLIMIT_CPU` e timeout de relógio; não restringe arquivos, rede nem syscalls.

<small>`pinneapple_tools/sandbox/custom_solver_runner.py` · 318 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `run_custom_solver` | função | `(script_path, params=None, *, timeout=60.0, memory_limit_mb=512) -> Dict[str, Any]` | Executa a função `solve(params)` do script num subprocesso limitado e devolve o resultado ou o erro. |

#### `pinneapple_tools.surrogate_policy`

Política baseada em tempo medido: decide se um problema precisa de surrogate cronometrando uma solução real, com cache JSON e limiar.

<small>`pinneapple_tools/surrogate_policy.py` · 92 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SurrogatePolicyResult` | classe | `(problem_id, needs_model, reason, measured_s)` | Resultado da política: problema, se precisa de modelo, motivo e tempo medido. |
| `SurrogatePolicy` | classe | `(cache_path, *, slow_threshold_s=1.0)` | Classifica problemas (sem solver direto, geometria cara, ou tempo medido acima de `slow_threshold_s`) e sugere uma arquitetura por heurística simples. |

#### `pinneapple_tools.visualization`

Visualização no estilo CFD (antigo `pinneapple_viz`): reexporta estilo, campos 2D/3D, gráficos de PINN, saídas de solvers, animações, malhas, voxels e vórtices; os subpacotes `dashboard` e `studio` cobrem painéis animados e cenas 3D realistas.

<small>`pinneapple_tools/visualization/__init__.py` · 116 linhas · **sem teste direto**</small>

#### `pinneapple_tools.visualization.animation`

Animações da evolução no tempo de soluções de EDP (campo escalar e linhas de corrente), salvas em GIF/MP4 com matplotlib. [sem teste]

<small>`pinneapple_tools/visualization/animation.py` · 205 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `animate_scalar_field` | função | `(trajectory, x=None, y=None, *, t_vals=None, title='Field evolution', field_label='u', cm…)` | Anima uma série temporal de campos escalares 2D com escala de cor fixa e salva opcionalmente. |
| `animate_streamlines` | função | `(u_traj, v_traj, x=None, y=None, *, speed_traj=None, t_vals=None, title='Velocity evoluti…)` | Anima linhas de corrente de uma série temporal de campos de velocidade, coloridas pela velocidade. |
| `make_gif` | função | `(frames, x, y, save_path, *, t_vals=None, cmap=DEFAULT_CMAP, dpi=80, fps=10) -> None` | Atalho que salva uma lista de quadros de campo 2D como GIF. |

#### `pinneapple_tools.visualization.dashboard`

Painéis centrados numa animação para qualquer simulação: animação no meio, gráficos sincronizados ao cursor de tempo, KPIs, parâmetros e barra de cor; exporta HTML interativo, PNG e GIF/MP4.

<small>`pinneapple_tools/visualization/dashboard/__init__.py` · 18 linhas · testes: 16</small>

#### `pinneapple_tools.visualization.dashboard.html`

Exportação HTML interativa de um `Dashboard` num único arquivo autônomo (sem servidor, sem internet).

<small>`pinneapple_tools/visualization/dashboard/html.py` · 429 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `payload` | função | `(db) -> dict` | Converte tudo que a página desenha em dados serializáveis em JSON. |
| `scene3d_payload` | função | `(bodies, poses, terrain=None, lighting=None, axes='z_up') -> dict` | Dados da vista 3D ao vivo (corpos, poses, terreno, iluminação) para a página. |
| `write_html` | função | `(db, path, *, fps=12, max_width=1280, quality=26) -> str` | Grava a página HTML autônoma do painel e devolve o caminho. |

#### `pinneapple_tools.visualization.dashboard.spec`

Modelo de dados puro (arrays numpy e textos) do painel: animação, gráficos, KPIs, parâmetros e barra de cor; os exportadores ficam em `html` e `static`. [sem teste direto]

<small>`pinneapple_tools/visualization/dashboard/spec.py` · 190 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Series` | classe | `(label, y, color, dash, x, width)` | Uma linha de gráfico, com um valor por quadro ou por ponto `x`. |
| `Marker` | classe | `(label, x, y, color)` | Um ponto que se move com os quadros num gráfico de fase. |
| `Chart` | classe | `(title, ylabel, xlabel, series, markers, x, kind, slot, ylim, note)` | Um gráfico do painel: título, eixos, séries, marcadores, tipo, posição (slot), limites e nota. |
| `Tile` | classe | `(label, value, unit, fmt, note)` | Um KPI: escalar fixo ou um valor por quadro que acompanha o cursor. |
| `Param` | classe | `(label, value, unit)` | Um parâmetro mostrado no painel lateral (rótulo, valor, unidade). |
| `ColorBar` | classe | `(title, cmap, lo, hi, saturation)` | Barra de cor ao lado da animação, com saturação ajustável para imitar uma sobreposição semitransparente. |
| `Dashboard` | classe | `(title, t, frames, charts, tiles, params, subtitle, colorbar, footer, time_label, time_un…)` | O painel completo (título, tempos, quadros, gráficos, KPIs, parâmetros, barra de cor, cena 3D), com `to_html`, `to_png` e `to_video`. |

#### `pinneapple_tools.visualization.dashboard.static`

Exportações PNG e GIF/MP4 de um `Dashboard`, desenhadas com matplotlib no estilo escuro do studio.

<small>`pinneapple_tools/visualization/dashboard/static.py` · 352 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `write_png` | função | `(db, path, k=None, size=(1920, 1080), dpi=100, font_scale=1.0) -> str` | Grava o painel num quadro dado (padrão: o último) como PNG. |
| `write_gif` | função | `(path, frames, fps, colors=255, loop=0) -> str` | GIF animado de quadros RGB com uma paleta adaptativa única para todo o clipe. |
| `hstack_gifs` | função | `(paths_or_frames, out, fps=10, colors=128, background='white', mismatch='error', loop=0)` | Junta várias animações lado a lado num só GIF, quadro a quadro (por exemplo, duas execuções comparadas). |
| `write_video` | função | `(db, gif=None, mp4=None, *, fps=12, size=(1920, 1080), gif_size=(1280, 720), gif_every=1,…)` | Grava o painel animado como GIF e/ou MP4 (H.264). |

#### `pinneapple_tools.visualization.electromagnetics`

Figuras e animações de campos eletromagnéticos: mapas com setas de Poynting, filmes de propagação (GIF com Pillow, MP4 com ffmpeg) e diagramas de radiação. [sem teste]

<small>`pinneapple_tools/visualization/electromagnetics.py` · 134 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `field_map` | função | `(field, extent, path=None, *, outline=None, vectors=None, title='', label='', cmap='RdBu_…)` | Desenha um campo 2D com contorno de material e setas opcionais, escala simétrica por padrão. |
| `field_movie` | função | `(frames, extent, path, *, times=None, outline=None, title='', label='', cmap='RdBu_r', fp…)` | Anima uma lista de campos 2D com escala de cor simétrica fixa, calculada de todos os quadros. |
| `radiation_pattern` | função | `(theta, values, path=None, *, reference=None, labels=('computed', 'analytic'), db=True, f…)` | Gráfico polar de um diagrama de campo distante normalizado, em dB opcional, contra uma referência opcional. |

#### `pinneapple_tools.visualization.fields`

Gráficos de campos escalares e vetoriais 2D/3D no estilo CFD (contornos, fatias, setas, linhas de corrente, comparação e erro). [sem teste]

<small>`pinneapple_tools/visualization/fields.py` · 361 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_scalar` | função | `(x, y, field, *, title='Scalar field', label='', cmap=DEFAULT_CMAP, n_levels=64, show_con…)` | Contorno preenchido de um campo escalar 2D, com linhas de contorno opcionais. |
| `plot_scalar_3d` | função | `(x, y, z, field, *, slice_axis='z', slice_val=None, title='Scalar field (3-D slice)', cma…)` | Plota uma fatia 2D de um campo escalar 3D ao longo de um eixo. |
| `plot_vectors` | função | `(x, y, u, v, *, title='Vector field', scale=None, density=20, color_by_mag=True, cmap='ra…)` | Gráfico de setas (quiver), coloridas pela magnitude, com campo escalar de fundo opcional. |
| `plot_streamlines` | função | `(x, y, u, v, *, title='Streamlines', density=1.5, color_by_mag=True, cmap='plasma', linew…)` | Linhas de corrente de um campo de velocidade em grade estruturada, com fundo opcional. |
| `compare_fields` | função | `(x, y, fields, *, cmap=DEFAULT_CMAP, n_cols=3, figsize_per_panel=(5, 4), suptitle='', sho…)` | Contornos de vários campos lado a lado para comparação. |
| `plot_error` | função | `(x, y, pred, ref, *, relative=True, title='Prediction error', cmap='hot', show=False) -> …)` | Mostra o erro ponto a ponto \|pred − ref\| (ou relativo) no domínio. |

#### `pinneapple_tools.visualization.integrations`

Pontes dos resultados do PINNeAPPle para outros visualizadores 3D (Plotly, Rerun, Trame). [sem teste direto]

<small>`pinneapple_tools/visualization/integrations/__init__.py` · 11 linhas · **sem teste direto**</small>

#### `pinneapple_tools.visualization.integrations.plotly_export`

HTML 3D interativo e autônomo com Plotly (girar, zoom, valores ao passar o mouse), para compartilhar um resultado sem servidor.

<small>`pinneapple_tools/visualization/integrations/plotly_export.py` · 94 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `scene_html` | função | `(scene, path, field=None, cmap='Turbo', title='', include_plotlyjs=True) -> str` | Exporta uma cena 3D (superfícies, campo opcional) para HTML Plotly. |
| `molecule_html` | função | `(mol, path, xyz=None, slab=None, metal='Au', title='', include_plotlyjs=True) -> str` | Exporta uma molécula, opcionalmente sobre uma placa metálica, para HTML Plotly. |
| `field_slices_html` | função | `(x, y, z, values, path, iso=None, cmap='Viridis', title='', include_plotlyjs=True) -> str` | Exporta um campo 3D em grade como três fatias medianas e, com `iso`, uma isosuperfície, em HTML Plotly. |

#### `pinneapple_tools.visualization.integrations.rerun_bridge`

Ponte para o visualizador Rerun (rerun.io): envia resultados ao vivo ou grava num arquivo `.rrd`; exige `rerun-sdk`.

<small>`pinneapple_tools/visualization/integrations/rerun_bridge.py` · 197 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RerunLogger` | classe | `(app_id='pinneapple', path=None, spawn=False, connect=None)` | Registrador Rerun que envia pontos, malhas, cenas, moléculas, escalares, conjuntos de execuções, grafos e texto, para arquivo, visualizador novo ou já aberto. |

#### `pinneapple_tools.visualization.integrations.trame_app`

App web Trame (Kitware) em volta de um plotter PyVista, servido a partir do Python, para malhas e campos grandes demais para um HTML autônomo.

<small>`pinneapple_tools/visualization/integrations/trame_app.py` · 37 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_app` | função | `(plotter, title='PINNeAPPle', local_rendering=False, server_name=None)` | Cria um servidor trame que mostra a janela de render do plotter, com renderização no servidor ou no navegador (vtk.js). |

#### `pinneapple_tools.visualization.mesh`

Utilitários de visualização de malhas e geometria (malha 2D, grupos de contorno, nuvem de pontos). [sem teste]

<small>`pinneapple_tools/visualization/mesh.py` · 137 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_mesh` | função | `(nodes, elems, *, node_values=None, cmap=DEFAULT_CMAP, edge_color='#4c9be8', edge_lw=0.5,…)` | Plota uma malha 2D de triângulos ou quadriláteros, com valores nodais opcionais. |
| `plot_boundary` | função | `(nodes, boundary_groups, *, title='Boundary groups', show=False) -> Figure` | Mostra os grupos de contorno rotulados de uma malha. |
| `plot_point_cloud` | função | `(points, values=None, *, cmap='plasma', s=3.0, title='Point cloud', show=False, dim3=Fals…)` | Dispersão 2D ou 3D de uma nuvem de pontos, colorida por valores opcionais. |

#### `pinneapple_tools.visualization.molecular`

Visualização molecular: proteínas e moléculas, sozinhas ou adsorvidas em superfícies metálicas, como imagens, trajetórias MP4, cenas glTF e painéis de dinâmica molecular.

<small>`pinneapple_tools/visualization/molecular.py` · 503 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Molecule` | classe | `(xyz, elements, names, resnames, resids, chains, frames)` | Molécula com posições, elementos, nomes, resíduos, cadeias e quadros; lê PDB ou mdtraj e dá massas, raios, classes e traço C-alfa. |
| `StepLog` | classe | `(title='', color=None, stream=None)` | Log de console passo a passo de um pipeline: passos numerados, detalhes indentados, tempo decorrido e cor em terminal. |
| `residue_class` | função | `(resname) -> str` | Classifica um resíduo pelo nome em hidrofóbico, polar, positivo, negativo, especial ou outro, para colorir. |
| `fcc_slab` | função | `(metal='Au', facet='111', nx=10, ny=10, layers=4, a=None) -> tuple[np.ndarray, np.ndarray]` | Posições atômicas de uma placa cúbica de face centrada com a faceta dada no topo, e os vetores da célula no plano. |
| `slab_for_footprint` | função | `(xyz, metal='Au', facet='111', padding=8.0, layers=4) -> tuple[np.ndarray, np.ndarray, tu…)` | Placa metálica que cobre a projeção x-y de uma molécula mais uma margem. |
| `orient_on_surface` | função | `(xyz, masses=None, active_site=None, gap=3.0, surface_z=0.0) -> np.ndarray` | Gira a molécula para deitar sua face mais larga sobre a superfície e a posiciona a uma distância dada. |
| `rmsd` | função | `(frames, ref, align=True) -> np.ndarray` | RMSD de cada quadro em relação a uma referência, com alinhamento de Kabsch opcional. |
| `height_above` | função | `(frames, surface_z, masses=None) -> np.ndarray` | Altura do centro de massa acima do plano da superfície, por quadro. |
| `contacts` | função | `(frames, surface_z, cutoff=3.5) -> np.ndarray` | Número de átomos a menos de um corte da superfície, por quadro. |
| `render` | função | `(mol, path, *, xyz=None, slab=None, metal='Au', color_by='residue', style='spheres', size…)` | Gera uma imagem estática da molécula, opcionalmente sobre uma placa, com cores e estilo escolhidos. |
| `render_frames` | função | `(mol, frames, out_dir, *, slab=None, metal='Au', color_by='residue', style='spheres', siz…)` | Gera um PNG por quadro de trajetória com câmera fixa ou em órbita lenta. |
| `encode_mp4` | função | `(pngs, out, fps=12, crf=22) -> str` | Codifica PNGs em MP4 H.264 com o ffmpeg empacotado pelo `imageio-ffmpeg`. |
| `md_dashboard` | função | `(path, t_ps, *, potential=None, temperature=None, rmsd_A=None, height_A=None, contacts_n=…)` | Figura com um painel por observável de dinâmica molecular ao longo do tempo (energia, temperatura, RMSD, altura, contatos). |
| `to_glb` | função | `(mol, path, *, xyz=None, slab=None, metal='Au', max_atoms=20000) -> str` | Exporta a cena como glTF binário (esferas por átomo agrupadas por classe e placa metálica) para o visualizador web. |

#### `pinneapple_tools.visualization.pinn`

Visualizações específicas de PINN: curvas de treino, campos de resíduo, pontos de colocação e comparação previsão × referência. [sem teste]

<small>`pinneapple_tools/visualization/pinn.py` · 341 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_loss_history` | função | `(history, *, keys=None, log_scale=True, title='Training loss history', show=False) -> Fig…)` | Plota curvas de perda de um histórico de treino, em escala log por padrão. |
| `plot_multi_loss` | função | `(history, *, groups=None, log_scale=True, title='Loss components', show=False) -> Figure` | Painéis separados por grupos de componentes de perda. |
| `plot_collocation` | função | `(interior_pts, boundary_pts=None, initial_pts=None, data_pts=None, *, title='Collocation …)` | Dispersão 2D dos conjuntos de pontos internos, de contorno, iniciais e de dados. |
| `plot_pinn_prediction` | função | `(x, y, pred, ref=None, *, field_name='u', cmap=DEFAULT_CMAP, show=False) -> Figure` | Painéis referência, previsão e erro (ou só a previsão, sem referência). |
| `plot_pde_residual` | função | `(x, y, residual, *, title='PDE residual', cmap='hot', log_scale=True, show=False) -> Figu…)` | Plota \|resíduo da EDP\| nos pontos de colocação, em escala log opcional. |
| `plot_gradient_magnitude` | função | `(x, y, grad_x, grad_y, *, title='Gradient magnitude \|∇u\|', cmap='plasma', show=False) -> …)` | Plota \|∇u\| a partir das derivadas parciais. |
| `predict_and_plot` | função | `(model, x_coords, y_coords, *, ref=None, field_name='u', device=None, cmap=DEFAULT_CMAP, …)` | Roda a inferência do modelo numa grade e plota o painel de previsão. |

#### `pinneapple_tools.visualization.pyvista_bridge`

Ponte PyVista: transforma malhas e campos do PINNeAPPle (inclusive resultados CalculiX e cenas `twin3d`) em imagens 3D de qualidade e arquivos VTK.

<small>`pinneapple_tools/visualization/pyvista_bridge.py` · 195 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `von_mises` | função | `(stress6) -> np.ndarray` | Tensão equivalente de von Mises a partir das seis componentes do CalculiX. |
| `grid_from_arrays` | função | `(points, cells, vtk_cell_type, point_data=None)` | Monta uma grade não estruturada a partir de pontos, células e tipo de célula VTK. |
| `calculix_grid` | função | `(model, frd=None, step=-1, warp=None)` | Grade PyVista de um modelo CalculiX com deslocamento, \|U\|, tensões e von Mises, se presentes. |
| `scene_to_multiblock` | função | `(scene, time_index=0, field_name=None)` | Converte uma `Scene` do `pinneapple_twin3d` num `pyvista.MultiBlock` (uma superfície por peça). |
| `export_vtk` | função | `(mesh, path) -> str` | Grava `.vtu`, `.vtp`, `.vtk` ou `.vtm` (pronto para ParaView), com o formato escolhido pela extensão. |
| `can_render` | função | `() -> bool` | Testa num subprocesso se dá para criar um contexto OpenGL fora da tela (o VTK derruba o interpretador em vez de levantar erro). |
| `render_png` | função | `(mesh, path, *, scalars=None, cmap='turbo', title='', show_edges=False, window_size=(1400…)` | Renderiza uma malha em PNG fora da tela, com escalares, mapa de cor e câmera escolhidos. |
| `render_gif` | função | `(mesh, path, *, scalars=None, cmap='turbo', title='', n_frames=60, elevation=25.0, window…)` | Gira a câmera em volta da malha e grava um GIF animado. |

#### `pinneapple_tools.visualization.solver`

Visualiza objetos `SolverOutput` de solvers FDM, FEM, FVM e sem malha. [sem teste]

<small>`pinneapple_tools/visualization/solver.py` · 198 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_solver_output` | função | `(output, *, x=None, y=None, title='Solver result', cmap=DEFAULT_CMAP, show=False) -> Figu…)` | Escolhe automaticamente o gráfico adequado para um `SolverOutput`. |
| `plot_fem_result` | função | `(output, *, component=0, title='FEM solution', cmap=DEFAULT_CMAP, show_mesh=True, show=Fa…)` | Plota uma solução FEM sobre sua malha Q1. |
| `plot_fvm_result` | função | `(output, *, frame=-1, title='FVM solution', cmap=DEFAULT_CMAP, show=False) -> Figure` | Plota um passo de tempo de uma trajetória FVM. |
| `plot_residuals` | função | `(output, *, title='Solver residuals', show=False) -> Figure` | Gráfico de barras dos resíduos/perdas guardados em `output.losses`. |

#### `pinneapple_tools.visualization.studio`

Studio do PINNeAPPle: visualização 3D realista de qualquer geometria e sua física (cena, visualizador web, Blender, PyVista); exporta `Scene` e atalhos de render.

<small>`pinneapple_tools/visualization/studio/__init__.py` · 47 linhas · testes: 41</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `render` | função | `(*a, **kw)` | Atalho para o render Blender Cycles de `studio.blender.render`. |
| `web_viewer` | função | `(*a, **kw)` | Atalho para a página interativa de `studio.web.web_viewer`. |
| `serve` | função | `(*a, **kw)` | Atalho que grava o visualizador numa pasta temporária e o serve (`studio.web.serve`). |

#### `pinneapple_tools.visualization.studio.animation`

Animação no tempo de uma `Scene`: corpos rígidos em trajetória, setas por quadro, rastros, campos por vértice e números na tela, para o visualizador web e o Blender.

<small>`pinneapple_tools/visualization/studio/animation.py` · 434 linhas · testes: 27</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Animation` | classe | `(times, tracks, vectors, trails, surface_fields, hud, camera, follow_group, follow_offset…)` | Dados por quadro sobre uma cena estática (trilhas, vetores, rastros, campos de superfície, HUD, câmera), com métodos para adicionar e consultar poses. |
| `matrix_from_quat` | função | `(q) -> np.ndarray` | Matriz de rotação de um ou vários quaternions unitários [w, x, y, z]. |
| `quat_from_matrix` | função | `(R) -> np.ndarray` | Quaternion unitário de uma matriz de rotação pelo método de Shepperd (estável numericamente). |
| `slerp` | função | `(q0, q1, f) -> np.ndarray` | Interpolação esférica entre quaternions unitários pelo arco mais curto. |

#### `pinneapple_tools.visualization.studio.blender`

Renders Blender Cycles de qualquer `Scene` (exige `pip install bpy` ou rodar dentro do Blender): imagens estáticas, animações e quadros de alta qualidade.

<small>`pinneapple_tools/visualization/studio/blender.py` · 1.296 linhas · testes: 27</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `render` | função | `(scene, out, *, field=None, field_range=None, lines=False, line_range=None, slice=None, s…)` | Renderiza uma cena (ou um `.glb`) em imagem com campos, linhas, fatias, vista e fundo escolhidos. |
| `blender_axes` | função | `(axes) -> np.ndarray` | Matriz de rotação dos eixos da cena para as coordenadas do Blender. |
| `render_animation` | função | `(scene, out_dir, *, frames=None, size=(960, 540), samples=16, camera='chase', camera_offs…)` | Renderiza a animação da cena quadro a quadro, com câmera de perseguição, HUD e sparklines opcionais. |
| `render_still` | função | `(scene, out_png, frame, *, size=(1920, 1080), samples=64, **kw) -> str` | Renderiza um único quadro da animação em alta qualidade como PNG. |

#### `pinneapple_tools.visualization.studio.colormap`

Escala de cor CFD (jet) e barra de cor gravada numa imagem, com ticks idênticos aos do visualizador web.

<small>`pinneapple_tools/visualization/studio/colormap.py` · 93 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `jet` | função | `(t) -> Tuple[float, float, float]` | Cor sRGB jet para t em [0, 1]. |
| `ramp` | função | `(t, stops=None) -> tuple[float, float, float]` | Cor sRGB de uma escala linear por partes (padrão jet) em t. |
| `decimals` | função | `(lo, hi) -> int` | Número de casas decimais que distingue cinco ticks em [lo, hi] (mesma regra do JavaScript do studio). |
| `fmt` | função | `(v, nd) -> str` | Formata um rótulo de tick como o JavaScript imprimiria (decimal fixo ou notação exponencial). |
| `ticks` | função | `(lo, hi, n=5, nd=None) -> List[str]` | Rótulos de tick igualmente espaçados, idênticos aos do visualizador web. |
| `colorbar` | função | `(path, title, lo, hi, nd=None, lo_txt='', hi_txt='', corner='bottom-right', stops=None) -…)` | Desenha uma barra de cor com 5 ticks e título num canto de uma imagem existente. |

#### `pinneapple_tools.visualization.studio.compose`

Mapas de cor para animações (jet e divergente), barras de cor gravadas em imagens e montagem de várias imagens ou vídeos em grade.

<small>`pinneapple_tools/visualization/studio/compose.py` · 238 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `stops` | função | `(cmap)` | Pontos de parada da escala de cor pedida. |
| `cmap_color` | função | `(cmap, t) -> tuple[float, float, float]` | Cor sRGB de um mapa em t (NaN vira 0). |
| `symmetric_range` | função | `(values) -> tuple[float, float]` | Faixa simétrica (−m, m) com m o maior \|valor\| finito, para mapas divergentes. |
| `default_range` | função | `(values, cmap='jet') -> tuple[float, float]` | Faixa padrão: mín..máx para jet, simétrica em torno de zero para mapa divergente. |
| `colorbar_image` | função | `(width, title, lo, hi, cmap='jet', scale=1.0, background=(255, 255, 255, 225))` | Barra de cor (título, gradiente, cinco ticks) como imagem RGBA do PIL. |
| `draw_colorbars` | função | `(path, bars, corner='bottom-right', width_frac=0.3) -> str` | Grava várias barras de cor empilhadas num canto de uma imagem (sobrescreve o arquivo). |
| `compose_grid` | função | `(images, out_png, labels=None, ncols=2, colorbar=None, background=(233, 237, 242)) -> str` | Monta várias imagens numa grade com rótulos e uma barra de cor comum. |
| `compose_grid_video` | função | `(mp4_paths, out_mp4, labels=None, ncols=2, colorbar=None, crf=28, webm=False, fps=None, b…)` | Monta vários vídeos numa grade com ffmpeg `xstack`. |

#### `pinneapple_tools.visualization.studio.fieldview`

Campos sobre uma malha para as vistas 3D dos apps: superfície de contorno de uma malha `pinneapple_data.cae` com vários campos, deformação e regiões marcadas, como glTF e JSON.

<small>`pinneapple_tools/visualization/studio/fieldview.py` · 179 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FieldView` | classe | `(scene, meta)` | A vista pronta: cena e metadados, com `glb()` e `to_dict()`. |
| `colour_range` | função | `(values, kind='value') -> dict[str, Any]` | Faixa de cor de um campo: valores de mín a máx; erros de 0 ao percentil 99, marcando quando houve corte. |
| `length_factor` | função | `(unit, to='m') -> float \| None` | Fator de conversão de uma unidade de comprimento para outra (None se não for comprimento). |
| `displacement_factor` | função | `(name, unit, components, mesh_unit='m') -> float \| None` | Se um campo vetorial nodal é um deslocamento, o fator que o converte para a unidade de comprimento da malha. |
| `subsample` | função | `(points, n=MAX_MARKER_POINTS) -> np.ndarray` | No máximo n pontos espalhados por igual no array (para marcadores). |
| `field_view` | função | `(mesh, fields, *, deform=None, markers=(), max_faces=VIEW_MAX_FACES, title='', axes='z_up…)` | Monta o `FieldView` de uma malha com campos, deformação e marcadores. |

#### `pinneapple_tools.visualization.studio.lighting`

Presets de iluminação e céu compartilhados pelos renderizadores do studio (PyVista, Blender Cycles, three.js).

<small>`pinneapple_tools/visualization/studio/lighting.py` · 171 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LightingPreset` | classe | `(name, sun_elevation_deg, sun_azimuth_deg, sun_intensity, sun_color, ambient, background_…)` | Um preset de luz: elevação, azimute, intensidade e cor do Sol, ambiente, fundo, sombras e Terra no céu. |
| `get_preset` | função | `(preset) -> LightingPreset` | Devolve um `LightingPreset` pelo nome. |
| `apply_pyvista` | função | `(plotter, preset='lunar_south_pole', target=(0.0, 0.0, 0.0), extent=10.0, earth=None, sha…)` | Aplica o preset a um plotter PyVista: fundo, Sol direcional com sombras e Terra opcional. |
| `add_earth_pyvista` | função | `(plotter, preset, target, extent)` | Adiciona a Terra como esfera distante com o tamanho angular correto, iluminada pela direção do Sol. |
| `threejs_config` | função | `(preset='lunar_south_pole') -> dict` | O preset como dados simples para os visualizadores no navegador. |
| `blender_world` | função | `(preset='lunar_south_pole', bpy=None)` | Configura a cena do Blender: cor do mundo, lâmpada Sol com tamanho angular e intensidade do preset. |

#### `pinneapple_tools.visualization.studio.particles`

Animações de partículas (milhares de esferas coloridas por um valor) dentro de equipamentos que podem se mover, renderizadas com Blender Cycles e compostas com gráficos ao vivo. [sem teste]

<small>`pinneapple_tools/visualization/studio/particles.py` · 323 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `render_particle_frames` | função | `(frames, d, out_dir, *, equipment=None, field='speed', field_range=None, view=(-1.0, -1.2…)` | Renderiza quadros de partículas com Blender Cycles, com equipamento, campo de cor e câmera. |
| `compose_video` | função | `(images, times, out, *, charts=(), colorbar=None, cmap='viridis', clock='Time: {t:.1f}', …)` | Compõe cada imagem com gráficos que se desenham ao longo do tempo, barra de cor e relógio, em GIF ou MP4. |
| `encode_mp4` | função | `(pngs, out, fps=10) -> str` | MP4 H.264 de uma sequência PNG usando o sequenciador do Blender (sem binário ffmpeg). |

#### `pinneapple_tools.visualization.studio.planetary`

Renders PyVista de um veículo numa superfície planetária: terreno por mapa de alturas com fotometria e sombras do Sol, peças com materiais, rochas, mapas de campo e horizonte.

<small>`pinneapple_tools/visualization/studio/planetary.py` · 348 linhas · testes: 17</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SurfaceView` | classe | `(X, Y, Z, parts, rocks, preset, terrain_field, field_alpha, field_glow, field_saturation,…)` | Descrição completa da vista de superfície: terreno, peças, rochas, preset de luz, campos, mapa de cor, exposição, Terra e terreno distante. |
| `horizon_distance` | função | `(height, planet_radius=MOON_RADIUS) -> float` | Distância até o horizonte de uma esfera lisa vista de uma altura dada (padrão: raio da Lua). |
| `ring_height` | função | `(view, x, y)` | Altura do anel de horizonte em (x, y), misturando o terreno distante com o nível da borda. |
| `terrain_height` | função | `(view, x, y)` | Altura do solo renderizado em (x, y): trecho detalhado, grade distante e anel de horizonte. |
| `border_points` | função | `(view, n=720) -> np.ndarray` | Pontos na borda externa do solo renderizado. |
| `border_visibility` | função | `(view, camera, size=(1600, 900), view_angle=35.0, samples=400, margin=0.0) -> dict` | Verifica se a borda externa do solo aparece vista da câmera, projetando os pontos de borda. |
| `render_surface` | função | `(view, path, camera, *, size=(1600, 900), view_angle=35.0, title='', subtitle='', plotter…)` | Renderiza a vista de superfície com uma câmera dada, em arquivo ou como imagem. |

#### `pinneapple_tools.visualization.studio.pyvista_render`

Renders fora da tela de uma `Scene` do studio com PyVista (VTK), a alternativa sem Blender para máquinas sem `bpy`.

<small>`pinneapple_tools/visualization/studio/pyvista_render.py` · 157 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `render_scene` | função | `(scene, path, *, field=None, cmap='jet', clim=None, lines=False, slice_name=None, view='i…)` | Renderiza a cena com campo, linhas, fatia, vista e chão opcionais num PNG. |

#### `pinneapple_tools.visualization.studio.scene`

Cena 3D para qualquer geometria: superfícies com materiais e campos por vértice, linhas de corrente e planos de fatia, com leitura (STL, OBJ, glTF, VTP) e escrita (glTF, USDA, STL).

<small>`pinneapple_tools/visualization/studio/scene.py` · 682 linhas · testes: 46</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Surface` | classe | `(name, vertices, faces, material, fields, group)` | Uma superfície triangulada com material, campos por vértice e grupo. |
| `Lines` | classe | `(points, values, name, label, cmap, alpha, radius, color_range)` | Polilinhas (por exemplo, linhas de corrente) com valores opcionais por ponto e mapa de cor. |
| `Slice` | classe | `(name, origin, u, v, grid, label, group)` | Plano retangular com uma grade de valores (NaN onde não há valor). |
| `Scene` | classe | `(surfaces, lines, slices, axes, title, labels, lighting, materials, animation)` | A cena completa (superfícies, linhas, fatias, eixos, iluminação, materiais, animação), criada de arrays, arquivo, malha ou peças, com dizimação e `save`. |
| `decimate_surface` | função | `(srf, target_faces) -> 'Surface'` | Simplifica uma superfície por agrupamento de vértices numa grade, ajustada por bisseção para chegar perto do número de faces pedido. |
| `scalar_of` | função | `(a) -> np.ndarray` | Um valor por entrada: o próprio, a magnitude de um vetor ou a tensão de von Mises de um tensor de 6 componentes. |
| `vertex_normals` | função | `(V, F) -> np.ndarray` | Normais por vértice de uma malha de triângulos. |
| `read_stl` | função | `(path) -> List[Tuple[str, np.ndarray, np.ndarray]]` | Lê um STL e devolve (nome, vértices, faces) por sólido, com vértices duplicados fundidos. |
| `read_obj` | função | `(path) -> Tuple[np.ndarray, np.ndarray]` | Lê vértices e faces de um arquivo OBJ. |
| `read_gltf` | função | `(path, axes='z_up') -> List[Surface]` | Lê as malhas de um `.glb`/`.gltf` com transformações aplicadas, convertidas para os eixos da cena. |
| `read_vtp` | função | `(path, material='grey') -> Surface` | Lê um VTK PolyData (`.vtp`) com seus dados de ponto e célula; exige `vtk`. |
| `write_glb` | função | `(surfaces, T, materials, generator='PINNeAPPle', groups=None) -> bytes` | Escreve glTF 2.0 binário com materiais PBR, e cada campo por vértice como atributo. |
| `write_usda` | função | `(surfaces, materials, title='scene', root='Scene') -> str` | Escreve OpenUSD em texto: uma Mesh por superfície com materiais `UsdPreviewSurface` e campos como primvars. |
| `write_stl` | função | `(surfaces, solid='scene') -> bytes` | Escreve um STL binário de todas as superfícies. |

#### `pinneapple_tools.visualization.studio.shadows`

Sombras do Sol e sombreamento de superfície planetária para terrenos por mapa de alturas, independentes do renderizador.

<small>`pinneapple_tools/visualization/studio/shadows.py` · 171 linhas · testes: 17</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `terrain_self_shadow` | função | `(Z, cell, sun_dir, max_distance=None) -> np.ndarray` | Máscara booleana das células onde o próprio terreno esconde o Sol. |
| `points_in_terrain_shadow` | função | `(P, X, Y, Z, sun_dir, step=None) -> np.ndarray` | Para cada ponto 3D, diz se o terreno bloqueia o Sol. |
| `cast_shadow` | função | `(meshes, X, Y, Z, sun_dir, iterations=3, supersample=1) -> np.ndarray` | Máscara das células da grade que ficam na sombra de malhas de triângulos (por exemplo, um veículo). |
| `lunar_lambert` | função | `(normals, sun_dir, view_dir, L=1.0) -> np.ndarray` | Refletância Lunar-Lambert de McEwen (1991), sem o albedo, para normais, Sol e direção de vista. |
| `relief_noise` | função | `(X, Y, rms, wavelengths=(0.6, 8.0), n=40, seed=7, slope=1.2)` | Relevo aleatório suave ancorado no mundo, como soma de ondas planas com direções e fases aleatórias. |
| `grid_normals` | função | `(Z, cell) -> np.ndarray` | Normais de um mapa de alturas em grade regular. |
| `shade_terrain` | função | `(X, Y, Z, sun_dir, camera_position, albedo=(0.45, 0.44, 0.42), shadow=None, exposure=1.0,…)` | Cores do terreno com fotometria Lunar-Lambert, sombra, exposição e luz de preenchimento. |

#### `pinneapple_tools.visualization.studio.solver_view`

Resultado de uma execução de solver (CalculiX ou OpenFOAM) como vista 3D para conferir antes de usar os números: malha, campos, forma deformada e regiões de contorno e carga.

<small>`pinneapple_tools/visualization/studio/solver_view.py` · 153 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `find_result` | função | `(folder) -> tuple[str, Path] \| None` | Identifica numa pasta um resultado CalculiX (`.frd`) ou um caso OpenFOAM; devolve None se não achar. |
| `solver_result_view` | função | `(folder, *, max_faces=VIEW_MAX_FACES) -> tuple[FieldView, dict[str, Any]]` | Monta o `FieldView` e um resumo do resultado do solver na pasta; levanta `ValueError` quando não há resultado. |

#### `pinneapple_tools.visualization.studio.web`

Visualizador interativo de uma `Scene` no navegador (three.js embutido na biblioteca, sem internet).

<small>`pinneapple_tools/visualization/studio/web.py` · 197 linhas · testes: 27</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `web_viewer` | função | `(scene, out_dir, title=None, max_faces=MAX_FACES, single_file=False, filename='index.html…)` | Grava o visualizador da cena numa pasta (ou num arquivo único) e devolve o caminho do `index.html`. |
| `serve` | função | `(scene, port=8000, open_browser=True, block=True, out_dir=None, max_faces=MAX_FACES)` | Grava o visualizador e o serve numa porta local, abrindo o navegador opcionalmente. |
| `screenshot` | função | `(viewer_dir, out_png, frame=None, size=(1280, 720), setup_js=None, timeout_ms=60000) -> s…)` | Captura um PNG do visualizador gravado com navegador headless (Playwright); devolve None se não for possível. |

#### `pinneapple_tools.visualization.style`

Mapas de cor, temas escuro/claro e padrões de figura no estilo CFD. [sem teste]

<small>`pinneapple_tools/visualization/style.py` · 118 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `use_cfd_style` | função | `(dark=True) -> None` | Aplica globalmente o tema CFD escuro ou claro no matplotlib. |
| `get_cmap` | função | `(name=DEFAULT_CMAP)` | Devolve um mapa de cor do matplotlib por apelido do PINNeAPPle ou nome padrão. |
| `make_figure` | função | `(nrows=1, ncols=1, **kwargs)` | Cria uma figura matplotlib já estilizada. |

#### `pinneapple_tools.visualization.trust_panels`

Painéis de campo lado a lado com um veredito de confiança sob cada modelo: qual previsão dá para usar e onde ela quebra a física.

<small>`pinneapple_tools/visualization/trust_panels.py` · 167 linhas · testes: 2</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Panel` | classe | `(title, values, kind, verdict, verdicts, note, inset)` | Um painel: campo, resíduo (log) ou erro (linear a partir de 0), com veredito e nota. |
| `Row` | classe | `(label, panels)` | Uma linha da figura: rótulo e lista de painéis. |
| `trust_panels` | função | `(rows, *, title='', footer='', cell=2.4, residual_range=None, error_max=None, error_label…)` | Figura com uma linha por `Row`: painéis de campo, barra de cor, resíduos e erros, com vereditos. |
| `trust_panels_gif` | função | `(frames, path, *, duration_ms=1600, **kwargs) -> Path` | Uma figura `trust_panels` por quadro, gravada como GIF em laço. |

#### `pinneapple_tools.visualization.vortex`

Identificação de vórtices e visualização de escoamento: vorticidade, critério Q, λ₂, enstrofia, dissipação e painéis prontos (inclusive para LBM).

<small>`pinneapple_tools/visualization/vortex.py` · 675 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `compute_vorticity_2d` | função | `(u, v, dx=1.0, dy=1.0) -> np.ndarray` | Componente z da vorticidade ω_z = ∂v/∂x − ∂u/∂y por diferenças finitas. |
| `compute_q_criterion_2d` | função | `(u, v, dx=1.0, dy=1.0) -> np.ndarray` | Critério Q para escoamento 2D incompressível. |
| `compute_vorticity_3d` | função | `(u, v, w, dx=1.0, dy=1.0, dz=1.0) -> np.ndarray` | Vetor vorticidade ω = ∇×u completo para escoamento 3D. |
| `compute_strain_rate_tensor_3d` | função | `(u, v, w, dx=1.0, dy=1.0, dz=1.0)` | Tensor taxa de deformação simétrico S_ij para escoamento 3D. |
| `compute_q_criterion_3d` | função | `(u, v, w, dx=1.0, dy=1.0, dz=1.0) -> np.ndarray` | Critério Q 3D: Q = 0.5(‖Ω‖² − ‖S‖²). |
| `compute_enstrophy` | função | `(vorticity) -> float` | Enstrofia: 0.5 vezes a soma de \|ω\|². |
| `compute_dissipation_rate` | função | `(strain_rate_tensor_components, nu) -> np.ndarray` | Taxa de dissipação viscosa ε = 2ν S_ij S_ij. |
| `compute_lambda2_3d` | função | `(u, v, w, dx=1.0, dy=1.0, dz=1.0) -> np.ndarray` | Critério λ₂ de Jeong & Hussain (1995); núcleo de vórtice onde λ₂ < 0. |
| `plot_vorticity` | função | `(x, y, omega, *, title='Vorticity ω_z', cmap='RdBu_r', symmetric=True, obstacle_mask=None…)` | Mapa de vorticidade com mapa de cor divergente centrado em zero e máscara de obstáculo opcional. |
| `plot_q_criterion_2d` | função | `(x, y, Q, vel_mag=None, *, title='Q-criterion', q_levels=30, highlight_positive=True, cma…)` | Gráfico do campo Q 2D, destacando regiões Q > 0, com velocidade de fundo opcional. |
| `plot_vortex_identification` | função | `(x, y, u, v, *, dx=1.0, dy=1.0, title=…, obstacle_mask=None, figsize=(11, 5), show=False)…)` | Calcula Q a partir da velocidade e plota com linhas de corrente sobrepostas. |
| `plot_q_criterion_3d` | função | `(x, y, z, Q, *, level=0.01, vel_mag=None, title='Q-criterion iso-surface', cmap='rainbow'…)` | Isosuperfície 3D do critério Q, colorida opcionalmente pela velocidade. |
| `plot_lbm_flow` | função | `(ux, uy, rho=None, obstacle_mask=None, *, dx=1.0, dy=1.0, title='LBM Flow Field', figsize…)` | Painel de 4 gráficos para escoamento LBM: magnitude da velocidade, vorticidade, critério Q e pressão (via densidade), com obstáculos em cinza. |
| `plot_flow_panel` | função | `(x, y, ux, uy, pressure=None, obstacle_mask=None, *, title='Flow Field Analysis', figsize…)` | Painel de 4 gráficos: pressão, \|v\| com linhas de corrente, critério Q e vorticidade. |

#### `pinneapple_tools.visualization.voxel`

Visualização de objetos `VoxelGrid` (fatias, ocupação 3D e histograma). [sem teste]

<small>`pinneapple_tools/visualization/voxel.py` · 127 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_voxel_slice` | função | `(grid, *, axis='z', index=None, cmap=DEFAULT_CMAP, title='', show=False) -> Figure` | Plota uma fatia 2D de uma `VoxelGrid` 3D ao longo de um eixo. |
| `plot_voxel_3d` | função | `(grid, *, threshold=0.5, cmap=DEFAULT_CMAP, alpha=0.6, title='Voxel occupancy', show=Fals…)` | Dispersão 3D dos voxels ocupados acima de um limiar, limitada a um número máximo. |
| `plot_voxel_histogram` | função | `(grid, *, bins=50, title='Voxel value distribution', show=False) -> Figure` | Histograma dos valores não nulos dos voxels. |

### Exemplos de uso reais do repositório

**Experimento Arena a partir de YAML** — `examples/benchmark_suite/02_arena_from_yaml.py`

```python
from pinneapple_tools.benchmark_suite.runner.run_arena_yaml import run_arena_experiment


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "examples" / "benchmark_suite" / "configs" / "experiment_burgers_1d.yaml"
    out_dir = repo_root / "data" / "artifacts" / "experiments" / "burgers_1d"
    ...
    result = run_arena_experiment(config_path, out_dir=out_dir)
```

**Campo de temperatura num stage OpenUSD, validado e pré-visualizado** — `examples/nvidia/07_omniverse_physicsnemo_coldplate/run.py`

```python
st = FieldStage(usd, "Cold plate, PhysicsNeMo FNO temperature")
path = st.add_mesh("ColdPlate", V, F, fields={"temperature": T.ravel(), "conductivity": (a_te[i] * K0).ravel(),
                                              "temperature_reference": (T_COOL + DT * u_te[i]).ravel(),
                                              "estimated_error": (DT * np.abs(est)).ravel()},
                   units={"temperature": "K", "conductivity": "W/m/K", "temperature_reference": "K",
                          "estimated_error": "K"}, color_by="temperature")
st.set_trust(path, decision, [c.to_dict() for c in checks], model="physicsnemo.models.fno.FNO",
             solver="pinneapple finite volumes")
st.save()
v = validate_stage(usd, required_primvars={path: ["temperature", "conductivity"]})
```

**Painel animado de um campo no tempo** — `examples/quickstart/11_visualisation.py`

```python
from pinneapple_tools.visualization.dashboard import Chart, Dashboard, Series, Tile
...
dashboard = Dashboard(
    "Diffusing blobs",
    t,
    frames,
    subtitle="T(x, y, t): two Gaussian blobs diffusing and decaying",
    charts=[Chart("peak amplitude", "amplitude [a.u.]", "time [s]",
                  [Series("peak", np.asarray(peaks))])],
    tiles=[Tile("final peak", peaks[-1], "a.u."), Tile("frames", len(frames), "")],
)
written = dashboard.to_video(gif=f"{OUT}/04_field_4d.gif")
```

**Coeficiente de pressão com `compute_cp`** — `tests/pinneapple_tools/test_aero_coefficients.py`

```python
def test_compute_cp_basic():
    p = np.array([100.0, 150.0, 50.0])
    p_inf = 100.0
    q_inf = 50.0
    cp = compute_cp(p, p_inf, q_inf)
    np.testing.assert_allclose(cp, [0.0, 1.0, -1.0])
```

**Frase -> workflow do catálogo de automação** — `tests/test_automation.py`

```python
def test_sentence_picks_the_workflow_and_converts_units():
    cat = catalog()
    it = interpret("Run the channel with the obstacle height set to 5.5 cm at an inlet velocity of 3 cm/s", cat)
    assert it.workflow.id == "channel_obstacle" and it.method == "rules"
    assert it.inputs == {"obstacle_height_mm": pytest.approx(55.0), "inlet_velocity_mm_s": pytest.approx(30.0)}
    with pytest.raises(WorkflowError, match="outside"):
        interpret("channel obstacle height 95 mm", cat)
    with pytest.raises(WorkflowError, match="no committed workflow"):
        interpret("bake a lemon cake", cat)
```

**Servir o zoo como microserviço NIM e consultar** — `examples/nvidia/06_model_services_zoo/run.py`

```python
        srv, url = serve_in_thread(default_registry())
        c = ModelServiceClient(url)
        try:
            with info.step("service discovery"):
                ready = c.ready()
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 70 de 177.
- Funções de teste em arquivos que importam algum módulo do pacote: 465.

### Limitações conhecidas

- `benchmark_suite/__init__.py` importa `PhysicsPipeline`, `TimeSeriesBenchmarkSpec`, `run_full_pipeline` e os benchmarks de transferência e meta-aprendizado dentro de `try/except Exception: pass`: se uma dependência falhar, o nome some sem aviso.
- Vários docstrings da `benchmark_suite` ainda citam nomes antigos de pacotes (`pinneapple_arena`, `pinneapple_environment`, `pinneapple_pinn`, `examples/pinneapple_arena/`), embora o código já importe os pacotes atuais.
- Os backends DeepXDE, JAX e PhysicsNeMo Sym dependem de pacotes opcionais e falham com `RuntimeError`/`ImportError` se eles não estiverem instalados; DeepXDE, JAX, nativo e multi-modelo não têm teste direto.
- `BenchmarkTaskBase` levanta `NotImplementedError` nos métodos abstratos; os runners de linha de comando (`runner/*`), `physics_pipeline` e `timeseries_pipeline` não têm teste direto.
- `hpo_experiments` não tem nenhum teste, depende de rede (arXiv, GitHub) e de uma chave da API Gemini, e o pipeline `reproduce` executa com `python run.py`/`main.py` código gerado por LLM, sem sandbox.
- `export_problemdesign` só escreve o arquivo complementar se `pinneapple_problemdesign` puder ser importado (o pacote existe no repositório; o import fica dentro de um try).
- O docstring de `pinneapple_tools.nvidia` lista os módulos `nim`, `cosmos` e `testing`, que não existem na pasta; Isaac Sim, Omniverse Kit RTX e USD BRep (wheel CPython 3.12) são externos e opcionais.
- Partes do `visualization` exigem pacotes opcionais (`bpy` para Blender, `pyvista`/`vtk`, `rerun-sdk`, `plotly`, `trame`, `playwright` para capturas, ffmpeg para MP4); muitos módulos de gráficos clássicos (`fields`, `pinn`, `solver`, `mesh`, `voxel`, `animation`, `style`) não têm teste direto.
- `sandbox.custom_solver_runner` não é uma fronteira de segurança: só limita memória virtual, CPU e tempo de relógio; o script ainda acessa arquivos, rede e pode criar processos (aviso do próprio docstring).
- `automation` depende de aplicações externas (FreeCAD, CfdOF, OpenFOAM, gmsh, ParaView, Xvfb, xdotool, Playwright); sem elas os passos ficam bloqueados. `PywinautoDriver` só funciona no Windows. Os adaptadores, o runner, o display e o relatório não têm teste direto (os testes cobrem workflow/interpretação/registro/verificação por outros módulos).
- As correções de compatibilidade em `adapters.openfoam` (`patch_cfdof_allmesh`, `patch_cfdof_inlet`) são específicas do OpenFOAM v1912 / versões ESI antigas com CfdOF.
- `dataset_quality.vlm_curation` precisa do extra `vlm_curation` (`transformers`, `accelerate`) e, para checkpoints hospedados, de GPU e download do Hugging Face; não há substituto offline para o julgamento do VLM.
- `compute_backends.jax_backend` exige JAX (e Equinox ou Flax para os modelos); sem JAX o módulo importa, mas as funções não rodam.
- `reporting.paperkit.Paper` precisa de `reportlab`; `reporting.datasheet` só gera PDF com Chromium/Chrome ou Playwright (sem eles, só HTML).
- O atalho `export_model` passa o modelo como primeiro argumento também para `csv`/`npz`, cujas funções esperam `coords` e `fields`; para esses formatos é melhor chamar `export_csv`/`export_npz` direto. `model_export` e o `__init__` do pacote não têm teste direto.
- `surrogate_policy.recommend_architecture` é uma heurística fixa (`vanilla_pinn` ou `modified_mlp` pelo número de observações), não um recomendador aprendido; `rea_bridge` precisa do REA (Node.js) instalado ou de `npx`.

## 2. `pinneapple_blender`

### Objetivo

_Sem texto de objetivo._

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 5 | 616 | 1 | 7 | 15 (2 arquivos) | 2 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `blender_addon` |  | 1 | 1 | 0 |
| `cycles` |  | 1 | 3 | 8 |
| `export` |  | 1 | 3 | 7 |
| `render` |  | 1 | 1 | 7 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_blender`

Blender bridge: export a PINNeAPPle field/trajectory as a ``.ply`` sequence (``export.py``, pure Python + numpy, no Blender needed), and optionally build/save a Blender scene from it via a real local Blender installation (``render.py``, subprocess bridge to Blender's own ``bpy``, mirroring the …

<small>`pinneapple_blender/__init__.py` · 15 linhas · **sem teste direto**</small>

#### `pinneapple_blender.blender_addon.import_pinneapple_sequence`

Runs INSIDE Blender's own Python (``bpy``) -- never imported by regular PINNeAPPle code, same reasoning as ``cfd_formats/abaqus_reader.py``'s ``_abaqus_odb_export_script.py``: ``bpy`` is Blender's bundled interpreter, not pip-installable into a normal environment, so this is launched as a …

<small>`pinneapple_blender/blender_addon/import_pinneapple_sequence.py` · 95 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `main` | função | `()` |  |

#### `pinneapple_blender.cycles`

In-process Cycles renderer: build a small scene from NumPy arrays and render it with Blender's ``bpy`` module.

<small>`pinneapple_blender/cycles.py` · 292 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CyclesScene` | classe | `(resolution=(960, 540), samples=64, background=(0.97, 0.97, 0.96), ambient=0.6, transpare…)` | A fresh Blender scene rendered with Cycles on the CPU. |
| `available` | função | `() -> bool` | True when the ``bpy`` module can be imported. |
| `frames_to_video` | função | `(frames, path, *, fps=24, crf=28) -> str` | Assemble PNG frames (named ``frame_%04d.png`` in one directory) into an H.264 MP4 with ffmpeg. |

#### `pinneapple_blender.export`

Export a PINNeAPPle result (a point cloud + one or more scalar/vector fields, optionally over several time steps) to a sequence of ``.ply`` files Blender can import natively (built-in importer, no add-on required just to look at one frame) -- the PINNeAPPle-side half of the Blender bridge; …

<small>`pinneapple_blender/export.py` · 153 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `write_ply` | função | `(path, points, colors=None, scalar_fields=None) -> None` | Write a point cloud (no faces -- a vertex-only PLY, which Blender's importer turns into a mesh with only vertices, ready for a Point Cloud Visualizer-style geometry-nodes setup, or straightforward as a particle/point render). ``colors``: (N,3) uint8 RGB, optional. ``scalar_fields``: extra named … |
| `export_scene` | função | `(points, field, out_dir, *, field_name='field', cmap='viridis', frame_index=0, extra_fiel…)` | Export one frame (one time instant / one realisation) as ``<out_dir>/frame_<NNNN>.ply``. Call once per frame with increasing ``frame_index`` to build a sequence -- see :func:`export_trajectory` for the common "I already have every frame's arrays" case in one call. |
| `export_trajectory` | função | `(points, fields_over_time, out_dir, *, field_name='field', cmap='viridis') -> list` | Export a whole sequence at once: ``points`` fixed (N, 3), one ``field`` array (N,) per frame in ``fields_over_time``. This is the natural shape for a PINN surrogate's own output -- evaluate the model at the same spatial points across a range of ``t`` values and pass the resulting list of (N,) … |

#### `pinneapple_blender.render`

Subprocess bridge to a real local Blender installation -- same pattern as ``cfd_formats.abaqus_reader.export_odb_fields``: never guess at Blender's internals from outside, shell out to Blender's own bundled Python (``blender --background --python <script>``) running genuine ``bpy`` code …

<small>`pinneapple_blender/render.py` · 61 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_scene` | função | `(ply_dir, *, output_blend=None, blender_executable='blender', timeout=600) -> str` | Run Blender headless to import a ``.ply`` sequence (as written by ``pinneapple_blender.export``) into a scene, optionally saving it as a ``.blend`` file. |

### Exemplos de uso reais do repositório

_Nenhum exemplo encontrado no repositório para este pacote._

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 3 de 5.
- Funções de teste em arquivos que importam algum módulo do pacote: 15.

### Limitações conhecidas

- _Nenhuma registrada._

## 3. `pinneapple_flow`

### Objetivo

Esta parte cobre o núcleo do motor de fluxos do `pinneapple_flow` e os seus nós: o documento de fluxo (`schema`), o registro de tipos de nó com portas tipadas e formulários de parâmetros (`registry`), a execução sobre o motor de workflows de `pinneapple_core.workflows` (`runtime`), os nós embutidos (`nodes`) e os nós que reproduzem os estudos de `examples/*` (`nodes_ext`), os nós gerados a partir da API pública (`libcalls`, `autocalls`, `wrappers`), o planejador que transforma uma descrição em texto em propostas de fluxo (`planner`), a paleta do Flow Studio (`palette`, `studio`), os registros de decisão com referências checadas (`decisions`, `references`), a identificação de equações e o gêmeo digital automático (`identify`, `twin`), os surrogates treináveis (`learn`) e a ligação com o selo do PINNeAPPle Lab (`lab_status`, `examples`).

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 89 | 18.086 | 23 | 381 | 263 (26 arquivos) | 58 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `__main__` |  | 1 | 10 | 10 |
| `autocalls` | Gera nós `auto.<módulo>.<função>` a partir das funções públicas da biblioteca que podem ser chamadas com segurança por um formulário, gravados em `autocalls.json`. | 1 | 6 | 10 |
| `config` |  | 1 | 5 | 0 |
| `coverage` |  | 1 | 6 | 10 |
| `decisions` | Registros de decisão de um fluxo (quem decidiu, por quê, com quais referências e evidências, com que confiança) e a seção de relatório em Markdown/HTML. | 1 | 11 | 22 |
| `examples` | Inventário de `examples/*` como fluxos: configurações rápidas, execução, comparações medidas e o documento `EXAMPLES_AS_FLOWS.md`. | 1 | 15 | 8 |
| `identify` | Identifica as equações que governam um problema a partir do texto (heurística offline ou LLM) e as descobre a partir de dados por regressão esparsa. | 1 | 4 | 7 |
| `lab_status` | Lê o selo do PINNeAPPle Lab de um item validado e diz que nós inserir no fluxo para subir de letra. | 1 | 5 | 9 |
| `learn` | Surrogates para nós de fluxo (blocos MLP, DeepONet, FNO, POD + MLP): construção, prévia como grafo de camadas, treino e busca de hiperparâmetros. | 1 | 14 | 0 |
| `libcalls` | Lista curada de chamadas públicas da biblioteca expostas como nós, com formulários gerados por introspecção e tipos de nó definidos pelo usuário. | 1 | 10 | 10 |
| `libcalls_catalog` |  | 1 | 0 | 0 |
| `nodes` | Tipos de nó embutidos: problema, dados, solvers, PINN, surrogates, métodos, validação, sistemas, raciocínio, veículo, visualização. | 20 | 91 | 27 |
| `nodes_ext` | Tipos de nó que reproduzem os estudos de `examples/*` como fluxos, um módulo por domínio, e as capacidades do planejador para eles. | 38 | 112 | 8 |
| `palette` | A paleta de nós do Flow Studio: seções em ordem de fluxo de engenharia, títulos legíveis, resumos e ranking de busca. | 1 | 12 | 29 |
| `planner` | Planejador: de uma descrição do problema a propostas de fluxo, por heurística offline, por LLM validado ou pelo assistente guiado. | 6 | 23 | 16 |
| `references` | Registro curado de referências (`references.yaml`) e a checagem (registro, DOI/URL) por que passa toda referência citada. | 1 | 10 | 13 |
| `registry` | Tipos de nó: portas tipadas, esquemas de parâmetros, a chamada de biblioteca de cada um e o decorador `node` que os registra. | 1 | 6 | 46 |
| `rules` |  | 1 | 3 | 13 |
| `runtime` | Executa um fluxo sobre o motor de workflows de `pinneapple_core.workflows`, com cache, re-execução parcial, artefatos e ganchos de fim de execução. | 1 | 15 | 9 |
| `schema` | O documento de fluxo `pinneapple-flow/1` (nós, arestas, metadados): carregar, salvar, validar e JSON Schema. | 1 | 9 | 27 |
| `service` |  | 1 | 1 | 29 |
| `store` |  | 1 | 3 | 23 |
| `studio` | Sobe o Flow Studio (`apps/flow_studio`, FastAPI + frontend React) em localhost, exigindo login para outros endereços. | 1 | 3 | 38 |
| `templates` |  | 1 | 2 | 0 |
| `twin` | Gêmeo digital montado automaticamente a partir de um sistema de equações identificado, sobre o `TwinRuntime` (filtro de Kalman por ensemble). | 1 | 3 | 7 |
| `types` |  | 1 | 2 | 7 |
| `wrappers` | Pontos de entrada com argumentos JSON para capacidades da biblioteca cuja API recebe objetos, para virarem nós de chamada de biblioteca. | 1 | 18 | 0 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_flow`

PINNeAPPle Flow: engineering flows as typed node graphs, planned from a description, run headless or in Flow Studio.

<small>`pinneapple_flow/__init__.py` · 91 linhas · testes: 231</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `register_references` | função | `(entries)` | Add curated references for another package's rules (see :func:`pinneapple_flow.references.register_references`). |
| `register_rule` | função | `(r)` | Add a heuristic rule that cites registered references (see :func:`pinneapple_flow.rules.register_rule`). |
| `run` | função | `(flow, **kw)` | Run a flow headless (see :func:`pinneapple_flow.runtime.run`). |
| `plan` | função | `(text, **kw)` | Proposals for a problem description (see :func:`pinneapple_flow.planner.plan`). |
| `catalog` | função | `() -> list[dict]` | Every node type as JSON (ports, parameter schema, availability). |

#### `pinneapple_flow.__main__`

Command line: ``python -m pinneapple_flow <command>``.

<small>`pinneapple_flow/__main__.py` · 212 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `cmd_run` | função | `(a) -> int` |  |
| `cmd_plan` | função | `(a) -> int` |  |
| `cmd_validate` | função | `(a) -> int` |  |
| `cmd_catalog` | função | `(a) -> int` |  |
| `cmd_examples` | função | `(a) -> int` |  |
| `cmd_coverage` | função | `(a) -> int` |  |
| `cmd_autocalls` | função | `(a) -> int` |  |
| `cmd_references` | função | `(a) -> int` |  |
| `cmd_studio` | função | `(a) -> int` |  |
| `main` | função | `(argv=None) -> int` |  |

#### `pinneapple_flow.autocalls`

Varre os módulos públicos que nenhum nó curado cobre e gera nós `auto.*` só para funções com argumentos de tipo simples e sem caminhos, URLs, comandos, `eval` ou escrita de arquivo; o resultado fica em `autocalls.json` e é regenerado por `python -m pinneapple_flow autocalls`.

<small>`pinneapple_flow/autocalls.py` · 236 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `scan` | função | `(module) -> tuple[list[dict[str, Any]], str]` | Gera as entradas de nó de um módulo, ou a razão de ele não ter nenhuma. |
| `generate` | função | `(modules=None) -> dict[str, Any]` | Varre os módulos (por padrão todos os que os nós curados não cobrem) e devolve os dados de `autocalls.json`. |
| `smoke` | função | `(calls, *, timeout=30.0, workers=6) -> None` | Chama cada função gerada com os padrões do formulário num interpretador novo em pasta vazia e grava o resultado em `entry["check"]`. |
| `prune` | função | `(data) -> None` | Descarta entradas cuja checagem mostra que devolvem uma função ou classe, e dá essa razão ao módulo que ficar vazio. |
| `load` | função | `() -> dict[str, Any]` | Lê `autocalls.json` (ou devolve listas vazias se o arquivo não existe). |
| `write` | função | `(data) -> None` | Grava os dados em `autocalls.json`. |

#### `pinneapple_flow.config`

Settings from the environment (and an optional ``.env`` file): LLM provider, database URL, folders, server.

<small>`pinneapple_flow/config.py` · 99 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Settings` | classe | `(host, port, db_url, llm_provider, llm_model, llm_base_url, llm_api_key)` |  |
| `load_dotenv` | função | `(path='.env', *, override=False) -> dict[str, str]` | Read ``KEY=value`` lines (``#`` comments, optional quotes) into ``os.environ``; returns what was read. |
| `home` | função | `() -> Path` |  |
| `data_root` | função | `() -> Path` |  |
| `safe_data_path` | função | `(name) -> Path` | A path inside the data folder (no absolute paths elsewhere, no ``..`` escapes). |

#### `pinneapple_flow.coverage`

Which PINNeAPPle capabilities are flow nodes, and why the others are not.

<small>`pinneapple_flow/coverage.py` · 183 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `packages` | função | `() -> list[str]` |  |
| `modules` | função | `(pkg) -> list[str]` |  |
| `node_targets` | função | `(auto=True) -> dict[str, set[str]]` | module path -> node types that call into it (``auto``: with the nodes generated from the public API). |
| `coverage` | função | `(auto=True) -> list[dict[str, str]]` | One row per package and first-level module: the node types that reach it, or why there are none (a curated reason, or the one the public-API scan of :mod:`pinneapple_flow.autocalls` recorded). |
| `counts` | função | `(rows=None) -> dict[str, int]` |  |
| `markdown` | função | `() -> str` |  |

#### `pinneapple_flow.decisions`

Define `Decision` e as fábricas de decisões de regra, de usuário e de LLM; a confiança vem de quem decidiu e da qualidade das referências (registro > verificada > não verificada), limitada a 0.35 sem referência nem evidência.

<small>`pinneapple_flow/decisions.py` · 236 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Decision` | classe | `(topic, what, decided_by, rationale, value, references, evidence, node, notes, uncited_re…)` | Registro de uma decisão do fluxo: tema, o que foi decidido, por quem, justificativa, valor, referências, evidências, confiança e rótulo. |
| `score` | função | `(d) -> Decision` | Calcula a confiança de uma decisão a partir de quem decidiu, das referências e das evidências; sem nenhuma, fica no máximo 0.35. |
| `rule_decision` | função | `(rule_id, *, what, node='', value=None, rationale='', topic=None, extra=()) -> Decision` | Cria a decisão tomada por uma regra heurística, com as referências e evidências da regra. |
| `user_decision` | função | `(node, topic, what, value=None, *, notes='', references=(), user='user') -> Decision` | Cria a decisão de uma edição do usuário, com notas e referências checadas no registro ou por DOI/URL. |
| `llm_decision` | função | `(provider_model, *, topic, what, node='', value=None, rationale='', references=()) -> Dec…)` | Cria a decisão proposta por um LLM; cada referência é checada e nada é aceito só pela palavra do modelo. |
| `edits_to_decisions` | função | `(edits) -> list[Decision]` | Converte as edições em `flow.meta["user_edits"]` em decisões de usuário. |
| `collect` | função | `(flow, reports=None) -> list[dict[str, Any]]` | Junta todas as decisões de um fluxo: do planejador/LLM, das edições do usuário e as tomadas pelos nós durante a execução. |
| `by_node` | função | `(decisions) -> dict[str, list[dict[str, Any]]]` | Agrupa decisões pelo nó a que se referem. |
| `summary` | função | `(decisions) -> dict[str, Any]` | Resumo das decisões: total, quantas sem referência, contagem por rótulo e referências citadas. |
| `report_markdown` | função | `(decisions, title='Decisions and references') -> str` | Gera a seção "Decisions and references" do relatório em Markdown. |
| `report_html` | função | `(decisions, title='Decisions and references') -> str` | Gera a mesma seção em HTML com referências clicáveis. |

#### `pinneapple_flow.examples`

Liga cada pasta de `examples/*` aos seus fluxos em `flows/examples/`, roda os fluxos (completo ou rápido), grava os resultados em `measured.json` e gera `EXAMPLES_AS_FLOWS.md`; `problems` é o portão de CI.

<small>`pinneapple_flow/examples.py` · 410 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `inventory` | função | `() -> dict[str, Any]` | Lê `flows/examples/inventory.json` (estado e razão de cada pasta de exemplo). |
| `measured` | função | `() -> dict[str, Any]` | Lê `flows/examples/measured.json` (resultados medidos das execuções dos fluxos de exemplo). |
| `example_folders` | função | `(root=None) -> list[str]` | Lista as pastas de `examples/*` que são exemplos (sem rascunhos, `_out`, `_runs` e pastas de resultados). |
| `flows` | função | `() -> list[dict[str, Any]]` | Lista todos os fluxos de exemplo com nome, caminho, título, domínio, nós e se têm configuração rápida. |
| `flow_path` | função | `(name) -> Path` | Caminho do JSON de um fluxo de exemplo pelo nome; `KeyError` se não existir. |
| `with_quick` | função | `(doc) -> dict[str, Any]` | Devolve o fluxo com a configuração rápida (`meta.quick`) aplicada. |
| `run_example` | função | `(name, *, quick=False, workdir=None, record=True, on_update=None) -> dict[str, Any]` | Roda um fluxo de exemplo (completo ou rápido) e, com `record`, grava resultado e comparações em `measured.json`. |
| `register_example_run` | função | `(result, flow) -> str` | Gancho de fim de execução que grava a execução de um fluxo de exemplo como manifesto de experimento e o importa no Lab do Flow Studio. |
| `register_hook` | função | `() -> None` | Adiciona `register_example_run` a `RUN_FINISHED_HOOKS` (uma vez só). |
| `scan` | função | `(example) -> dict[str, Any]` | Por script de um exemplo, lista as chamadas de biblioteca que ele importa e quais um nó declara. |
| `status` | função | `(example, inv=None, meas=None) -> dict[str, Any]` | Linha do inventário de um exemplo: fluxos, estado, razão e comparação medida. |
| `problems` | função | `() -> list[str]` | Portão de CI: pastas de exemplo sem fluxo nem razão, estados inválidos e fluxos que citam exemplo inexistente. |
| `markdown` | função | `() -> str` | Gera o texto de `docs/flow_studio/EXAMPLES_AS_FLOWS.md`. |
| `planner_rows` | função | `(rows) -> list[str]` | Tabela do planejador: por exemplo com fluxo, se o planejador heurístico propõe a capacidade em primeiro lugar a partir do título. |
| `main` | função | `(argv) -> int` | CLI `python -m pinneapple_flow example-flows` (list, run, doc, check). |

#### `pinneapple_flow.identify`

Classifica sistemas de equações candidatos para uma descrição (palavras-chave, unidades, valores explícitos e, opcionalmente, um LLM validado) e ajusta uma EDP a dados em grade por regressão esparsa (STRidge/SINDy).

<small>`pinneapple_flow/identify.py` · 256 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `identify` | função | `(text, *, dimension=None, units=None, n=3, llm=None, call=None) -> dict[str, Any]` | Devolve sistemas de equações candidatos ordenados para uma descrição, com confiança, raciocínio e decisões; um LLM opcional acrescenta um candidato validado. |
| `discover_pde` | função | `(axes, values, coords, *, field='u', threshold=0.05, trim=3) -> dict[str, Any]` | Ajusta `u_t` (e `u_tt`) a termos candidatos por regressão esparsa sobre dados em grade e devolve a equação, coeficientes, R² e família. |
| `confirm` | função | `(candidate, found) -> dict[str, Any]` | Diz se a equação descoberta concorda com um candidato (mesma família e, quando possível, coeficientes). |
| `to_json` | função | `(res) -> str` | Serializa o resultado em JSON convertendo arrays numpy em listas. |

#### `pinneapple_flow.lab_status`

Consulta `pinneapple_lab.curation` e `pinneapple_lab.seal` (detectados em tempo de execução) para mostrar a letra do selo e os próximos passos; o Flow Studio nunca dá nota sozinho.

<small>`pinneapple_flow/lab_status.py` · 211 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `seal_status` | função | `(item, lab=None) -> dict[str, Any]` | Letra do selo de um item do Lab com sua razão; `None` quando o item nunca foi curado. |
| `nodes_for` | função | `(action='', command='', params=None, target='') -> list[dict[str, Any]]` | Tipos de nó registrados que executam uma ação do Lab (pelo tipo ou pela chamada de biblioteca que declaram), curados primeiro. |
| `next_steps` | função | `(item, lab=None) -> dict[str, Any]` | O que falta ao item para a próxima letra, com a ação do Lab e os nós que a executam. |
| `lab_items` | função | `(reports) -> list[str]` | Itens do Lab gravados pelos nós `validate.lab` de uma execução. |
| `insert_nodes` | função | `(flow, anchor, items) -> tuple[dict[str, Any], list[str]]` | Insere nós num fluxo e liga cada entrada obrigatória a partir de um nó âncora ou de nós compatíveis; devolve o fluxo e as entradas não ligadas. |

#### `pinneapple_flow.learn`

Monta surrogates torch a partir das arquiteturas da biblioteca num formato único de dataset (`A`, `Y`, `grid`), normaliza, treina com Adam e mede erros em partições separadas. [sem teste]

<small>`pinneapple_flow/learn.py` · 366 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `default_config` | função | `(kind, in_dim, out_dim) -> dict[str, Any]` | Arquitetura sugerida para um tipo de surrogate e os tamanhos dos dados. |
| `blocks_net` | função | `(in_dim, out_dim, blocks)` | Monta um `torch.nn.Sequential` a partir de blocos editáveis (`dense`, `residual`, `dropout`, `layernorm`) com camada linear final. |
| `build` | função | `(cfg, in_dim, out_dim, *, n_fields=1, grid=None)` | Constrói o surrogate torch de uma configuração, mapeando `(B, in_dim)` para `(B, out_dim)` em unidades normalizadas. |
| `count_parameters` | função | `(model) -> int` | Conta os parâmetros treináveis de um modelo. |
| `layer_graph` | função | `(model) -> dict[str, Any]` | Grafo das camadas folha com contagem de parâmetros, agrupadas por bloco (ramo e tronco da DeepONet em grupos separados). |
| `architecture` | função | `(cfg, in_dim, out_dim, *, n_fields=1, grid=None) -> dict[str, Any]` | Prévia de uma arquitetura sem treinar: grafo de camadas, número de parâmetros e a configuração usada. |
| `split` | função | `(n, val=0.15, test=0.15, seed=0)` | Divide índices em treino, validação e teste de forma aleatória com semente. |
| `rel_l2_rows` | função | `(pred, true) -> float` | Média por amostra de \|\|pred - true\|\| / \|\|true\|\|. |
| `fit` | função | `(cfg, data, *, epochs=500, lr=0.002, batch=32, seed=0, n_fields=1, splits=None, callback=…)` | Treina o surrogate com Adam e agenda cosseno, guardando os pesos de menor erro de validação; devolve modelo e métricas por partição. |
| `save` | função | `(model, cfg, path, *, n_fields=1, meta=None) -> None` | Salva configuração, tamanhos e pesos do surrogate com `torch.save`. |
| `load` | função | `(obj)` | Reconstrói um surrogate a partir do que `save` gravou. |
| `search_space` | função | `(spec) -> dict[str, tuple]` | Converte uma especificação simples de espaço de busca em parâmetros de `AdaptiveSweepConfig`. |
| `apply_trial` | função | `(cfg, trial) -> tuple[dict[str, Any], dict[str, Any]]` | Separa os valores de uma tentativa em mudanças de arquitetura e opções de treino. |
| `tune` | função | `(cfg, data, space, *, n_trials=12, epochs=200, seed=0, n_fields=1, log=print)` | Busca de hiperparâmetros (Optuna TPE se instalado, senão busca aleatória) minimizando o erro de validação em partições fixas. |

#### `pinneapple_flow.libcalls`

Mantém `CALLS`, a lista de chamadas permitidas; gera o esquema de parâmetros a partir da assinatura e só executa entradas da lista, inclusive para nós `custom.*` criados pelo usuário.

<small>`pinneapple_flow/libcalls.py` · 409 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LibCall` | classe | `(key, target, title, category, description, refs, inputs, exclude, overrides, init, requi…)` | Entrada da lista permitida: alvo `módulo:atributo`, título, categoria, referências, entradas vindas de portas, parâmetros excluídos ou sobrescritos e tipo de saída. |
| `resolve` | função | `(target)` | Devolve o objeto chamável de um alvo da lista permitida. |
| `introspect` | função | `(call) -> tuple[list[Param], list[str]]` | Lê os parâmetros de tipo simples de uma chamada e os nomes que não puderam ser expostos. |
| `summarize` | função | `(ctx, result, depth=0) -> dict[str, Any]` | Transforma o que uma chamada devolveu em entradas de relatório e um resultado JSON. |
| `spec_for` | função | `(call, *, type_=None, title=None, fixed=None, defaults=None, hidden=()) -> NodeSpec` | Monta o `NodeSpec` de uma `LibCall`, com valores fixos, padrões e parâmetros ocultos opcionais. |
| `wrapped` | função | `(call) -> list[str]` | Para um wrapper de `pinneapple_flow.wrappers`, lê no código-fonte os módulos da biblioteca que ele importa. |
| `custom_spec` | função | `(doc) -> NodeSpec` | Cria um tipo de nó `custom.*` do usuário sobre uma entrada de `CALLS`; `ValueError` se citar outra coisa ou parâmetro inexistente. |
| `library_specs` | função | `() -> list[NodeSpec]` | Os `NodeSpec` de todas as entradas de `CALLS`. |
| `auto_specs` | função | `() -> list[NodeSpec]` | Os `NodeSpec` dos nós gerados em `autocalls.json`; os módulos só são importados quando o nó roda. |
| `add` | função | `(*calls) -> None` | Acrescenta `LibCall`s a `CALLS`, recusando chaves duplicadas. |

#### `pinneapple_flow.libcalls_catalog`

The curated allowlist of library calls exposed as nodes (``lib.<key>``). Parameter forms come from the signatures; this file only says which callable, where it sits in the palette, which inputs come from ports and which references back it. Add a capability by adding an entry.

<small>`pinneapple_flow/libcalls_catalog.py` · 119 linhas · **sem teste direto**</small>

#### `pinneapple_flow.nodes`

Importa os módulos de nós embutidos, registra os nós de chamada de biblioteca (`library_specs` e `auto_specs`) e depois os nós de exemplos de `nodes_ext`.

<small>`pinneapple_flow/nodes/__init__.py` · 29 linhas · testes: 17</small>

#### `pinneapple_flow.nodes._common`

Peças comuns dos nós embutidos: documentos de problema, artefatos de solução em grade, soluções exatas, interpolação e gráficos. [sem teste direto]

<small>`pinneapple_flow/nodes/_common.py` · 186 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `coords_of` | função | `(geometry) -> list[str]` | Lista de coordenadas de uma geometria. |
| `bounds_of` | função | `(problem) -> dict[str, tuple[float, float]]` | Limites por eixo de um problema, incluindo `t` quando há tempo. |
| `system_of` | função | `(problem, *, coefficients=None)` | O `EquationSystem` analisado de um documento de problema. |
| `need_problem` | função | `(v) -> dict[str, Any]` | Garante que o valor é um problema com geometria e equações, senão falha o passo. |
| `grid_points` | função | `(axes, coords) -> np.ndarray` | Pontos de uma grade produto dos eixos, como matriz `(N, d)`. |
| `save_solution` | função | `(ctx, name, axes, values, coords, fields, method, label='', extra=None) -> dict[str, Any]` | Grava uma solução em grade como artefato npz com eixos, campos e método. |
| `load_solution` | função | `(v) -> dict[str, Any]` | Lê um artefato de solução em grade (eixos, valores, coordenadas, campos). |
| `exact_function` | função | `(problem)` | Função da solução exata do problema, ou `None`. |
| `interpolate` | função | `(sol, X) -> np.ndarray` | Interpola linearmente uma solução em grade em pontos dados. |
| `eval_axes` | função | `(problem, n=None, n_t=None) -> dict[str, np.ndarray]` | Grade de avaliação de um problema: pontos por eixo dados, a malha anexada ou um padrão por dimensão. |
| `plot_solution` | função | `(sol, title='', field=0)` | Figura de uma solução: curvas no tempo (1-D), mapa (2-D) ou cortes no plano médio (3-D). |

#### `pinneapple_flow.nodes.analysis`

Nós de validação `validate.compare` (erros contra referência e checagens de física do trust card) e `validate.lab` (curadoria do PINNeAPPle Lab com letra A-F e selo na figura). [sem teste direto]

<small>`pinneapple_flow/nodes/analysis.py` · 306 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `compare` | função | `(ctx, inputs, params)` | Nó `validate.compare`: mede cada candidato contra a referência nos mesmos pontos (L2 relativo, erro máximo e RMS) e as checagens de física. |
| `lab` | função | `(ctx, inputs, params)` | Nó `validate.lab`: grava a validação como execução do Lab, dá a letra A-F e carimba o selo na figura só nos níveis A e B. |

#### `pinneapple_flow.nodes.autolab`

Nós do Autonomous Physics Lab (`pinneapple_autolab`): resfriamento de placa, gêmeo de tubulação que sabe quando erra e o aerodinamicista com OpenFOAM como referência. [sem teste direto]

<small>`pinneapple_flow/nodes/autolab.py` · 143 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `board_cooling` | função | `(ctx, inputs, params)` | Nó `autolab.board_cooling`: acha o dissipador mais leve e a velocidade do ventilador de uma placa dentro dos limites, com verificação em malhas finas e relatório de confiança. |
| `pipe_twin` | função | `(ctx, inputs, params)` | Nó `autolab.pipe_twin`: gêmeo de tubulação que distingue mudança de processo, sensor errado e modelo fora do domínio, medido em casos controlados. |
| `aerodynamicist` | função | `(ctx, inputs, params)` | Nó `autolab.aerodynamicist`: reduz o arrasto de um carro de corrida paramétrico mantendo downforce e balanço, com OpenFOAM como referência e surrogates comparados. |

#### `pinneapple_flow.nodes.cfd_pipeline`

Nós do pipeline FreeCAD -> OpenFOAM (`pinneapple_simulation.pipelines.freecad_openfoam`): estudo, geometria, execução, varredura e surrogate da varredura; precisam do OpenFOAM instalado. [sem teste direto]

<small>`pinneapple_flow/nodes/cfd_pipeline.py` · 172 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `study` | função | `(ctx, inputs, params)` | Nó `cfd.study`: modelo FreeCAD paramétrico com regras de patches, malha, física, seções e referências. |
| `geometry` | função | `(ctx, inputs, params)` | Nó `cfd.geometry`: ajusta a planilha, recalcula, exporta e nomeia os patches, sem solver. |
| `run` | função | `(ctx, inputs, params)` | Nó `cfd.run`: CAD, malha, solução, pós-processamento e checagens de um caso. |
| `sweep` | função | `(ctx, inputs, params)` | Nó `cfd.sweep`: varredura de parâmetros retomável, um processo por caso. |
| `surrogate` | função | `(ctx, inputs, params)` | Nó `cfd.surrogate`: todos os surrogates aplicáveis com o mesmo orçamento, erro medido em casos separados. |

#### `pinneapple_flow.nodes.condense`

Nó de condensação estática hierárquica de um encapsulamento de chip (`pinneapple_condense`). [sem teste direto]

<small>`pinneapple_flow/nodes/condense.py` · 64 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `package_thermal` | função | `(ctx, inputs, params)` | Nó `condense.package_thermal`: resolve um encapsulamento multicamada sintético condensando blocos repetidos e relata temperatura de junção e balanço de energia. |

#### `pinneapple_flow.nodes.data`

Nós de dados: observações sintéticas de uma solução, leitura de CSV, conjuntos paramétricos e o benchmark Burgers com condições iniciais de campo aleatório gaussiano. [sem teste direto]

<small>`pinneapple_flow/nodes/data.py` · 194 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `observe` | função | `(ctx, inputs, params)` | Nó `data.observe`: observações pontuais de uma solução (sensores sintéticos) com ruído, guardando os coeficientes verdadeiros. |
| `ingest` | função | `(ctx, inputs, params)` | Nó `data.ingest`: lê medições de CSV colado ou de arquivo da pasta de dados. |
| `parametric` | função | `(ctx, inputs, params)` | Nó `data.parametric`: conjunto de treino para surrogates com um solve por diferenças finitas por amostra. |
| `grf_burgers` | função | `(ctx, inputs, params)` | Nó `data.grf_burgers`: condições iniciais de campo aleatório gaussiano mapeadas para a solução de Burgers viscosa (solver pseudo-espectral). |
| `dataset_of` | função | `(v)` | Lê um dataset de operador (`A`, `Y`, `grid`) de um artefato, senão falha o passo. |

#### `pinneapple_flow.nodes.methods`

Nós de métodos que envolvem implementações da biblioteca: convergência de malha, filtro de Kalman, propagação de incerteza, ROMs, MeshGraphNet, transferência, meta-aprendizado, arena de PINNs, NSGA-II e controle. [sem teste direto]

<small>`pinneapple_flow/nodes/methods.py` · 657 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `convergence` | função | `(ctx, inputs, params)` | Nó `verify.convergence`: três malhas aninhadas, ordem observada, extrapolação de Richardson e GCI. |
| `kalman` | função | `(ctx, inputs, params)` | Nó `assimilate.filter`: estima estados de canais ruidosos com os filtros de Kalman da biblioteca. |
| `propagate` | função | `(ctx, inputs, params)` | Nó `uq.propagate`: propaga incerteza uniforme nas entradas do surrogate por amostragem ou caos polinomial. |
| `rom_fit` | função | `(ctx, inputs, params)` | Nó `rom.fit`: ajusta ROMs aos snapshots e compara os erros de previsão com o mesmo orçamento. |
| `meshgraphnet` | função | `(ctx, inputs, params)` | Nó `gnn.meshgraphnet`: treina uma MeshGraphNet num campo de difusão em malhas aleatórias e compara com o estado inicial congelado. |
| `transfer` | função | `(ctx, inputs, params)` | Nó `transfer.pinn`: fine-tuning de uma PINN treinada no problema alvo contra treino do zero com as mesmas épocas. |
| `meta_learn` | função | `(ctx, inputs, params)` | Nó `meta.learn`: meta-aprende uma inicialização de PINN numa família de problemas e compara a adaptação com inicialização aleatória. |
| `arena` | função | `(ctx, inputs, params)` | Nó `benchmark.arena`: mesmos problemas, orçamento e épocas para todos os modelos, com L2 relativo e posição por problema. |
| `nsga2` | função | `(ctx, inputs, params)` | Nó `optimize.nsga2`: busca um problema de projeto com NSGA-II (ou um baseline) e relata a frente de Pareto. |
| `control` | função | `(ctx, inputs, params)` | Nó `control.compare`: compara controladores em malha fechada no tanque aquecido da biblioteca (só simulação, sem comando a equipamento). |

#### `pinneapple_flow.nodes.physics_learning`

Nó que verifica se um modelo térmico aprendido conhece a física com um verificador independente da perda de treino (`pinneapple_analysis.validation.physics_learning`). [sem teste direto]

<small>`pinneapple_flow/nodes/physics_learning.py` · 62 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `physics_learning` | função | `(ctx, inputs, params)` | Nó `analysis.physics_learning`: checa resíduo, fluxo, contorno, balanço de energia, simetria, escala e intervenções causais contra referência de volumes finitos. |

#### `pinneapple_flow.nodes.presets`

Nós de presets industriais (`pinneapple_presets`): rodar um caso pronto com parâmetros próprios e validá-lo contra os casos de referência.

<small>`pinneapple_flow/nodes/presets.py` · 60 linhas · testes: 27</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `run_preset` | função | `(ctx, inputs, params)` | Nó `preset.run`: resolve um caso industrial pronto com as entradas do usuário. |
| `validate_preset` | função | `(ctx, inputs, params)` | Nó `preset.validate`: roda os casos de referência do preset e mede o erro. |

#### `pinneapple_flow.nodes.problem`

Nós de problema: domínio (1-D, 2-D, 3-D), malha, sistema de equações em texto validado por parser seguro e o problema genérico que identifica as equações. [sem teste direto]

<small>`pinneapple_flow/nodes/problem.py` · 227 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `geometry_doc` | função | `(params) -> dict[str, Any]` | Monta o documento de geometria (dimensão, limites, tempo) a partir dos parâmetros, checando os intervalos. |
| `domain` | função | `(ctx, inputs, params)` | Nó `geometry.domain`: dimensão, limites por eixo e intervalo de tempo. |
| `mesh` | função | `(ctx, inputs, params)` | Nó `geometry.mesh`: grade estruturada com nós por eixo e níveis de tempo salvos. |
| `equations` | função | `(ctx, inputs, params)` | Nó `equations.system`: EDPs, condições e coeficientes em texto, analisados por parser seguro e validados. |
| `require_problem` | função | `(v)` | Atalho para `need_problem`. |
| `generic` | função | `(ctx, inputs, params)` | Nó `problem.generic`: identifica as equações de qualquer problema descrito, com confiança e referências, e as descobre a partir de dados quando houver. |

#### `pinneapple_flow.nodes.query`

Nó de consulta física de um encapsulamento de chip, respondida pelo método válido mais barato com incerteza e procedência (`pinneapple_query`). [sem teste direto]

<small>`pinneapple_flow/nodes/query.py` · 46 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `package_queries` | função | `(ctx, inputs, params)` | Nó `query.package`: responde cada consulta pelo método mais barato que atinge a meta e diz qual, com incerteza. |

#### `pinneapple_flow.nodes.reason`

Nós de raciocínio de engenharia (`pinneapple_reason`): requisito, conceito, modo de falha, verificação, contramedida, cadeia de porquês e relatório.

<small>`pinneapple_flow/nodes/reason.py` · 183 linhas · testes: 17</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `requirement` | função | `(ctx, inputs, params)` | Nó `reason.requirement`: o que o projeto deve fazer e quais modos de falha o impediriam. |
| `concept` | função | `(ctx, inputs, params)` | Nó `reason.concept`: uma versão do projeto com parâmetros, elementos e origem de cada valor. |
| `failure_mode` | função | `(ctx, inputs, params)` | Nó `reason.failure_mode`: um modo de falha no conceito com demanda, capacidade e mecanismo. |
| `verify` | função | `(ctx, inputs, params)` | Nó `reason.verify`: todos os modos de falha de todos os requisitos e o primeiro que falha. |
| `countermeasure` | função | `(ctx, inputs, params)` | Nó `reason.countermeasure`: tenta as contramedidas da primeira falha e aplica a escolhida. |
| `chain` | função | `(ctx, inputs, params)` | Nó `reason.chain`: ciclo checar-corrigir a partir do conceito ingênuo até todos os requisitos passarem. |
| `report` | função | `(ctx, inputs, params)` | Nó `reason.report`: a cadeia como relatório com linha do tempo, margens, figuras e relatório de confiança. |

#### `pinneapple_flow.nodes.science`

Nós que rodam um experimento de `pinneapple_science` contra a referência citada e desenham o atlas das áreas cobertas.

<small>`pinneapple_flow/nodes/science.py` · 65 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `run_experiment` | função | `(ctx, inputs, params)` | Nó `science.run`: roda um experimento científico, cada grandeza contra sua referência citada. |
| `atlas` | função | `(ctx, inputs, params)` | Nó `science.atlas`: mapa das áreas científicas disponíveis, parciais e planejadas. |

#### `pinneapple_flow.nodes.solvers`

Nós de solver: clássico (diferenças finitas, elementos finitos), OpenFOAM, PINN, PINN inverso e calibração por mínimos quadrados. [sem teste direto]

<small>`pinneapple_flow/nodes/solvers.py` · 362 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `classical` | função | `(ctx, inputs, params)` | Nó `solver.classical`: resolve o sistema editado com método clássico e relata custo e resíduo. |
| `openfoam` | função | `(ctx, inputs, params)` | Nó `solver.openfoam`: roda um caso OpenFOAM (blockMesh + solver) e lê a saúde do log; precisa do OpenFOAM no PATH. |
| `pinn` | função | `(ctx, inputs, params)` | Nó `train.pinn`: treina uma PINN nas equações editadas (Adam e L-BFGS) e a valida em pontos novos. |
| `discover` | função | `(ctx, inputs, params)` | Nó `inverse.discover`: identifica coeficientes desconhecidos junto com o campo a partir de observações (PINN inversa). |
| `calibrate` | função | `(ctx, inputs, params)` | Nó `inverse.calibrate`: calibração por mínimos quadrados com o solver de diferenças finitas como modelo direto. |

#### `pinneapple_flow.nodes.surrogates`

Nós de surrogate sobre `pinneapple_flow.learn`: treino com arquitetura editável, HPO, fine-tuning e índices de Sobol. [sem teste direto]

<small>`pinneapple_flow/nodes/surrogates.py` · 274 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `surrogate` | função | `(ctx, inputs, params)` | Nó `surrogate.train`: treina um surrogate num dataset e relata erros de validação e teste. |
| `hpo` | função | `(ctx, inputs, params)` | Nó `surrogate.hpo`: busca arquitetura e treino na validação e retreina o melhor, sem usar o teste na busca. |
| `finetune` | função | `(ctx, inputs, params)` | Nó `train.finetune`: continua o treino de um surrogate com dados novos e compara o erro antes e depois. |
| `sensitivity` | função | `(ctx, inputs, params)` | Nó `uq.sensitivity`: índices de Sobol de uma grandeza do campo em relação às entradas do surrogate. |

#### `pinneapple_flow.nodes.systems`

Nós de sistemas: gêmeo digital do tanque aquecido, previsão do ENSO (baixa dados e precisa de statsmodels), laboratório de previsão de séries e decolagem com a biblioteca de voo. [sem teste direto]

<small>`pinneapple_flow/nodes/systems.py` · 424 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `twin` | função | `(ctx, inputs, params)` | Nó `twin.connect`: alimenta o runtime do gêmeo (Kalman por ensemble no tanque aquecido) com leituras e devolve estimativas, alertas e previsão. |
| `enso` | função | `(ctx, inputs, params)` | Nó `forecast.enso`: previsão do índice Niño 3.4 com o protocolo da biblioteca de ENSO; baixa as observações uma vez. |
| `forecast` | função | `(ctx, inputs, params)` | Nó `forecast.lab`: testes de previsibilidade e depois um conjunto de previsores em validação walk-forward com período de teste intocado. |
| `takeoff` | função | `(ctx, inputs, params)` | Nó `flight.takeoff`: simula uma decolagem com a biblioteca de voo 6-DOF e compara a distância com a forma fechada de Raymer. |

#### `pinneapple_flow.nodes.twin_nodes`

Gêmeo digital automático: nó de dados ao vivo (sensores simulados, arquivo, MQTT, OPC UA) e nó que monta o gêmeo a partir das equações identificadas. [sem teste direto]

<small>`pinneapple_flow/nodes/twin_nodes.py` · 279 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `stream` | função | `(ctx, inputs, params)` | Nó `data.stream`: leituras de sensores simuladas, de arquivo ou ao vivo via MQTT / OPC UA. |
| `twin_auto` | função | `(ctx, inputs, params)` | Nó `twin.auto`: monta o gêmeo a partir das equações identificadas, com alertas e painel. |

#### `pinneapple_flow.nodes.vehicle`

Nós do rover planetário de `pinneapple_simulation.vehicle`: veículo, solo, terreno, travessia multicorpo, modelo de roda Wong-Reece, PINN do model zoo, painel animado e a missão que aprende o solo.

<small>`pinneapple_flow/nodes/vehicle.py` · 479 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `vehicle_presets` | função | `() -> dict[str, Any]` | Funções de `pinneapple_simulation.vehicle` que constroem um `VehicleSpec`. |
| `build_spec` | função | `(v)` | Constrói o veículo a partir do preset e da massa, com geometria de roda editável. |
| `build_soil` | função | `(v)` | Constrói o solo do banco de terramecânica com parâmetros sobrescritos. |
| `build_site` | função | `(v)` | Constrói o terreno pelo cenário escolhido. |
| `load_run` | função | `(v) -> dict[str, Any]` | Lê a travessia gravada por um nó `vehicle.simulate`, conferindo o checksum. |
| `virtual_display` | função | `(size=(1920, 1080))` | Display para pyvista: o atual ou um Xvfb iniciado durante o bloco. |
| `vehicle_spec` | função | `(ctx, inputs, params)` | Nó `vehicle.spec`: rover da biblioteca (NASA VIPER) com massa e roda editáveis. |
| `soil` | função | `(ctx, inputs, params)` | Nó `terramechanics.soil`: solo do banco de terramecânica (parâmetros de Bekker / Wong-Reece) com sobrescrita. |
| `site` | função | `(ctx, inputs, params)` | Nó `scenario.site`: terreno deformável com cratera e rocha, ou rampa plana. |
| `simulate` | função | `(ctx, inputs, params)` | Nó `vehicle.simulate`: dinâmica multicorpo do rover sobre o terreno com rodas Wong-Reece. |
| `wheel_model` | função | `(ctx, inputs, params)` | Nó `terramechanics.wheel_model`: modelo de roda Wong-Reece e suas curvas de bancada na carga estática. |
| `build_wheel_model` | função | `(v)` | Constrói o `WongReeceModel` a partir do solo, do veículo e das opções. |
| `zoo_surrogate` | função | `(ctx, inputs, params)` | Nó `terramechanics.zoo_surrogate`: PINN de contato da roda do model zoo (baixada uma vez, checksum verificado) com seus erros do model card. |
| `dashboard` | função | `(ctx, inputs, params)` | Nó `vehicle.dashboard`: painel animado da travessia com gráficos sincronizados, em HTML, PNG, MP4/GIF. |
| `learn_planet` | função | `(ctx, inputs, params)` | Nó `rover.learn_planet`: missão em que o rover atola, aprende o solo pelas rodas (EKF iterado) e replaneja, num terreno sintético de solos conhecidos. |
| `ctx_metrics` | função | `(nums) -> dict[str, float]` | Métricas numéricas planas do resumo de uma missão do rover. |

#### `pinneapple_flow.nodes.viz`

Nós de visualização e exportação: gráficos, visualizador three.js, render (Blender Cycles se `bpy` existe, senão matplotlib) e relatório autocontido. [sem teste direto]

<small>`pinneapple_flow/nodes/viz.py` · 237 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot` | função | `(ctx, inputs, params)` | Nó `viz.plot`: uma figura por solução (curvas, mapa de cores ou cortes). |
| `field3d` | função | `(ctx, inputs, params)` | Nó `viz.field3d`: visualização interativa three.js do campo. |
| `render` | função | `(ctx, inputs, params)` | Nó `viz.render`: imagem e animação de um campo 2-D com Blender Cycles ou, sem `bpy`, matplotlib. |
| `report` | função | `(ctx, inputs, params)` | Nó `export.report`: relatório autocontido com métricas, parâmetros e figuras embutidas. |

#### `pinneapple_flow.nodes_ext`

Importa os módulos de nós por domínio que reproduzem `examples/*`, liga o gancho que grava execuções de exemplo como experimentos do Lab e coloca cada nó na seção da paleta da sua categoria. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/__init__.py` · 58 linhas · **sem teste direto**</small>

#### `pinneapple_flow.nodes_ext.aero`

Nó de aerodinâmica de avião do exemplo `geometry_library/03_airliner_vlm`: malha de vórtices do `airliner_a320` e arrasto de atrito. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/aero.py` · 71 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `airliner_vlm` | função | `(ctx, inputs, params)` | Nó `aero.airliner_vlm`: inclinação da sustentação, ponto neutro, trimagem, arrasto induzido, CD0 da malha e L/D. |

#### `pinneapple_flow.nodes_ext.architectures`

Nós dos exemplos `examples/architectures`: listagem do registro de modelos, construção por nome com passagem direta em tensores falsos e compressão POD com rollout DMD. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/architectures.py` · 172 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `registry_listing` | função | `(ctx, inputs, params)` | Nó `architectures.registry`: famílias e modelos do registro e o catálogo por família. |
| `forward` | função | `(ctx, inputs, params)` | Nó `architectures.forward`: um modelo do registro construído por nome e rodado em tensores falsos (forma de saída, parâmetros, perdas, gradiente). |
| `pod_dmd` | função | `(ctx, inputs, params)` | Nó `architectures.pod_dmd`: erros de reconstrução POD e de rollout POD + DMD num sistema linear sintético. |

#### `pinneapple_flow.nodes_ext.arena_pipelines`

Nós de `examples/arena_pipelines`: registro de datasets, especificações de benchmark físico e de séries temporais e a execução multi-modelo da `Arena`. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/arena_pipelines.py` · 205 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `datasets` | função | `(ctx, inputs, params)` | Nó `arena.datasets`: ids de datasets por categoria e os arrays que cada id carrega. |
| `physics_spec` | função | `(ctx, inputs, params)` | Nó `arena.physics_spec`: todos os modelos da especificação num problema físico, métricas e leaderboard. |
| `timeseries_spec` | função | `(ctx, inputs, params)` | Nó `arena.timeseries_spec`: todos os modelos de previsão da especificação no mesmo corte lookback/horizonte. |
| `arena_run` | função | `(ctx, inputs, params)` | Nó `arena.run`: todos os modelos treinados e avaliados nos mesmos pontos, com erros por campo e tempo de treino. |

#### `pinneapple_flow.nodes_ext.astro`

Nós de acreção em buraco negro (`examples/black_hole_weather`): simulação de toro RIAF e estatísticas de janela do estudo de resolução. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/astro.py` · 117 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `simulate` | função | `(ctx, inputs, params)` | Nó `blackhole.simulate`: evolui o fluxo de acreção a partir de um toro de equilíbrio (potencial de Paczynski-Wiita). |
| `statistics` | função | `(ctx, inputs, params)` | Nó `blackhole.statistics`: taxa média de acreção, perfis e erro de persistência na janela. |

#### `pinneapple_flow.nodes_ext.autolab_caps`

Registra capacidades do planejador para fluxos de exemplo cujos nós vivem em outros módulos (presets industriais, experimentos de ciência, missões do laboratório autônomo). [sem teste direto]

<small>`pinneapple_flow/nodes_ext/autolab_caps.py` · 42 linhas · **sem teste direto**</small>

#### `pinneapple_flow.nodes_ext.benchmarks`

Nós de benchmark: o benchmark de mesmo orçamento (`pinneapple_analysis.benchmark.run`) e a suíte de `examples/benchmark_suite` (Arena por YAML, transferência, meta-aprendizado). [sem teste direto]

<small>`pinneapple_flow/nodes_ext/benchmarks.py` · 239 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `equal_budget` | função | `(ctx, inputs, params)` | Nó `benchmark.equal_budget`: todos os métodos em todos os problemas com o mesmo `ComputeBudget`, busca e sementes. |
| `arena_yaml` | função | `(ctx, inputs, params)` | Nó `benchmark.arena_yaml`: preset, dados de referência e todos os modelos do YAML treinados nos mesmos pontos. |
| `transfer` | função | `(ctx, inputs, params)` | Nó `benchmark.transfer`: pré-treino numa fonte e fine-tuning com cada estratégia de congelamento contra treino do zero. |
| `meta` | função | `(ctx, inputs, params)` | Nó `benchmark.meta`: meta-treino por família de EDP e adaptação em K passos contra treino do zero. |

#### `pinneapple_flow.nodes_ext.bricks`

Nós dos tijolos encaixáveis auto-modificáveis (`examples/bricks`): checagens que controlam o laço e as duas estratégias de busca com o mesmo orçamento. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/bricks.py` · 83 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `validate` | função | `(ctx, inputs, params)` | Nó `bricks.validate`: valor simulado de cada checagem, se passou, e a decisão de confiança. |
| `search` | função | `(ctx, inputs, params)` | Nó `bricks.search`: as duas estratégias em cada semente com o mesmo orçamento e a melhor construção. |

#### `pinneapple_flow.nodes_ext.cfd`

Nós de CFD com OpenFOAM: a calha de superfície livre com jato (interFoam) de `examples/free_surface` e a campanha de `examples/openfoam_campaign`. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/cfd.py` · 210 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `trough_jet` | função | `(ctx, inputs, params)` | Nó `cfd.trough_jet`: um caso de inclinação da calha com jato (k médio perto do fundo, pico, profundidade, volume). |
| `ratio` | função | `(ctx, inputs, params)` | Nó `cfd.ratio`: razão de uma métrica de cada caso pela do caso base. |
| `campaign` | função | `(ctx, inputs, params)` | Nó `cfd.campaign`: caso base e cada execução do plano com sua saúde e perfis amostrados. |

#### `pinneapple_flow.nodes_ext.core_ts`

Nós de geofísica (água rasa na esfera, 4D-Var em Lorenz-96, Richards), do `PhysicsOptimizer`, do híbrido FEM + correção neural e de séries temporais. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/core_ts.py` · 270 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `earth_system` | função | `(ctx, inputs, params)` | Nó `geo.earth_system`: onda de Rossby-Haurwitz, 4D-Var em Lorenz-96 e infiltração de Richards com balanço de massa. |
| `torsion_optimizer` | função | `(ctx, inputs, params)` | Nó `core.torsion_optimizer`: maximiza a rigidez torcional de um retângulo com vários otimizadores. |
| `hybrid_fem` | função | `(ctx, inputs, params)` | Nó `core.hybrid_fem`: FEM grosseira com corretor Conv1d treinado contra FEM fina. |
| `regime_series` | função | `(n=900, seed=0) -> np.ndarray` | Série do exemplo de previsão adaptativa: sazonal, depois tendência, depois passeio aleatório. |
| `adaptive` | função | `(ctx, inputs, params)` | Nó `ts.adaptive`: especialistas ponderados pelos erros recentes, MAE do conjunto, cobertura do intervalo e trocas de líder. |
| `naive_baselines` | função | `(ctx, inputs, params)` | Nó `ts.naive_baselines`: baselines ingênuo, sazonal ingênuo e drift numa série sintética. |

#### `pinneapple_flow.nodes_ext.design`

Nós de projeto: otimização topológica (`pinneapple_design.topology`), checagem de gradiente, rotor de relutância, projeto a partir de intenção e prévia qualitativa do Ahmed body. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/design.py` · 321 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `topology2d` | função | `(ctx, inputs, params)` | Nó `design.topology2d`: otimização topológica 2-D (viga MBB, balanço, térmica, Stokes). |
| `topology_gradcheck` | função | `(ctx, inputs, params)` | Nó `design.topology_gradcheck`: sensibilidades adjuntas contra diferenças centrais. |
| `reluctance_rotor` | função | `(ctx, inputs, params)` | Nó `design.reluctance_rotor`: núcleo de rotor de 4 polos por otimização topológica, reavaliado com ferro não linear. |
| `from_intent` | função | `(ctx, inputs, params)` | Nó `design.from_intent`: de um objetivo em linguagem natural a um projeto verificado, perguntas ou recusa. |
| `ahmed_slant` | função | `(ctx, inputs, params)` | Nó `qualitative.ahmed_slant`: variação de arrasto prevista por inclinação contra a medida, com baixa confiança perto da separação. |

#### `pinneapple_flow.nodes_ext.electromagnetics`

Nós dos estudos de `examples/motor_magnetics` e `examples/electrodynamics`: tabela de verificação, máquina de ímãs na superfície e o motor IPM do Prius 2004 (torque, mapa de fluxo, acionamento, skew, tolerâncias). [sem teste direto]

<small>`pinneapple_flow/nodes_ext/electromagnetics.py` · 512 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `em_verification` | função | `(ctx, inputs, params)` | Nó `em.verification`: cada solver eletromagnético contra resultado independente, com erros e ordens observadas. |
| `spm_machine` | função | `(ctx, inputs, params)` | Nó `em.spm_machine`: máquina de ímãs na superfície com torque por tensor de Maxwell e por trabalho virtual. |
| `prius_machine` | função | `(ctx, inputs, params)` | Nó `em.prius_machine`: motor do Prius 2004, torque contra ângulo de corrente comparado com Yang et al. (IEMDC 2013). |
| `prius_flux_map` | função | `(ctx, inputs, params)` | Nó `em.prius_flux_map`: psi_d, psi_q e torque numa grade i_d x i_q, com perdas. |
| `prius_drive` | função | `(ctx, inputs, params)` | Nó `em.prius_drive`: envelope MTPA + enfraquecimento de campo, mapa de eficiência e energia no ciclo WLTC. |
| `prius_skew` | função | `(ctx, inputs, params)` | Nó `em.prius_skew`: cogging, torque e ripple contra o skew em degraus e harmônicos da FEM contra o fator de skew. |
| `prius_tolerances` | função | `(ctx, inputs, params)` | Nó `em.prius_tolerances`: dispersão de torque e ripple pelas tolerâncias dos ímãs e do entreferro. |

#### `pinneapple_flow.nodes_ext.em_antenna`

Nó da zona quieta de um compact range de antenas por óptica física, com validação e varredura de projeto. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/em_antenna.py` · 67 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `compact_range` | função | `(ctx, inputs, params)` | Nó `antenna.compact_range`: zona quieta do projeto de referência, validação e varredura de projeto. |

#### `pinneapple_flow.nodes_ext.em_prius`

Nós dos estudos longos do motor do Prius: projeto de rotor (NSGA-II com surrogate GP), rede térmica em ciclos WLTC, verificação cruzada com gmsh + GetDP e efeitos de extremidade; os dois últimos precisam de gmsh e GetDP. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/em_prius.py` · 315 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `rotor_design` | função | `(ctx, inputs, params)` | Nó `em.prius_rotor_design`: busca híbrida com surrogate contra só FEA (hipervolume e chamadas de FEA). |
| `prius_thermal` | função | `(ctx, inputs, params)` | Nó `em.prius_thermal`: rede térmica concentrada com perdas WLTC e margens de desmagnetização por grau de NdFeB. |
| `getdp_crosscheck` | função | `(ctx, inputs, params)` | Nó `em.prius_getdp_crosscheck`: FEA do PINNeAPPle contra um modelo independente gmsh + GetDP da mesma máquina. |
| `end_effects` | função | `(ctx, inputs, params)` | Nó `em.prius_end_effects`: o estudo de efeitos de extremidade do exemplo 16 (precisa de gmsh e GetDP). |

#### `pinneapple_flow.nodes_ext.examples`

O nó `examples.compare`, que confere números do fluxo contra os resultados versionados do exemplo com tolerância `atol + rtol |ref|`, o nó `examples.flatten` e ajudantes comuns dos nós de exemplo.

<small>`pinneapple_flow/nodes_ext/examples.py` · 309 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Written` | classe | `(text)` | Número de referência que guarda o texto como foi escrito (preserva as casas decimais para a tolerância de arredondamento). |
| `save_object` | função | `(ctx, name, obj) -> dict[str, Any]` | Grava um objeto Python como artefato pickle do nó. |
| `load_object` | função | `(v) -> Any` | Lê o objeto de um artefato de `save_object` (só artefatos escritos por nós desta execução). |
| `examples_root` | função | `() -> Path \| None` | Pasta `examples` do checkout (`$PINNEAPPLE_REPO/examples` ou ao lado dos pacotes); `None` numa wheel instalada. |
| `example_dir` | função | `(name) -> Path` | Pasta de um exemplo pelo nome, falhando se não existir ou se a pasta `examples` não estiver presente. |
| `doc_table_value` | função | `(ref) -> float` | Primeiro número de uma célula de tabela Markdown em `docs/`, indicada por `doc:<caminho>\|<linha>\|<coluna>`. |
| `reference_value` | função | `(folder, ref) -> float` | O número de referência que uma expressão `ref` aponta numa pasta de exemplo (arquivo de resultados, manifesto, nó ou tabela de doc). |
| `rounding_half_unit` | função | `(x) -> float` | Meia unidade do último dígito de um número como o Python o escreve; zero com 12 ou mais dígitos significativos. |
| `metric_value` | função | `(reports, name) -> float` | Valor de `node.metric` nos relatórios dos nós anteriores. |
| `compare` | função | `(ctx, inputs, params)` | Nó `examples.compare`: compara cada número medido pelo fluxo com o versionado no exemplo, dentro da tolerância. |
| `flatten` | função | `(ctx, inputs, params)` | Nó `examples.flatten`: transforma cada número de um resultado aninhado numa métrica para `examples.compare` checar. |

#### `pinneapple_flow.nodes_ext.flight`

Nós de dinâmica de voo (`examples/flight_dynamics`) sobre `pinneapple_simulation.flight`: aeronaves, verificação, trimagem, decolagem, turbulência, rajada e cenas 3-D. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/flight.py` · 588 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `aircraft` | função | `(ctx, inputs, params)` | Nó `flight.aircraft`: modelos de aeronave (asa e cauda em malha de vórtices, fuselagem, trem, motores, massas com fontes). |
| `verification` | função | `(ctx, inputs, params)` | Nó `flight.verification`: as checagens do script de verificação do exemplo. |
| `trim_stability` | função | `(ctx, inputs, params)` | Nó `flight.trim_stability`: trimagem em cruzeiro, derivadas adimensionais e autovalores da linearização de 12 estados. |
| `takeoff_cases` | função | `(ctx, inputs, params)` | Nó `flight.takeoff_cases`: corrida, rotação, decolagem e subida, com forma fechada de Raymer, vento cruzado e rajada. |
| `turbulence_encounter` | função | `(ctx, inputs, params)` | Nó `flight.turbulence_encounter`: cada aeronave na mesma turbulência, com fatores de carga e acelerações na cabine. |
| `pratt` | função | `(ctx, inputs, params)` | Nó `flight.pratt`: pico de fator de carga numa rajada 1-cosseno contra a fórmula de Pratt-Walker. |
| `same_gust` | função | `(ctx, inputs, params)` | Nó `flight.same_gust`: mesma rajada para todas as aeronaves, com picos no CG e na cabine e as estimativas quase-estática e de Pratt. |
| `scene` | função | `(ctx, inputs, params)` | Nó `flight.scene`: cenas 3-D animadas dos voos como visualizadores three.js e entrada dos nós de render Blender. |

#### `pinneapple_flow.nodes_ext.football`

Nós de futebol (`examples/sports`): problema inverso da trajetória da bola, suavização de rastreamento (dados simulados) e pitch control numa partida de amostra da Metrica (baixada uma vez). [sem teste direto]

<small>`pinneapple_flow/nodes_ext/football.py` · 76 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `football_ball_flight_inverse` | função | `(ctx, inputs, params)` | Nó `football.ball_flight_inverse`: chutes simulados com ruído ajustados pela EDO de arrasto-Magnus, com cobertura dos intervalos e comparação de Cramér-Rao 3-D contra 2-D. |
| `football_tracking_smoothing` | função | `(ctx, inputs, params)` | Nó `football.tracking_smoothing`: erro de velocidade e potência metabólica por diferenças, Savitzky-Golay e suavizador Kalman-RTS. |
| `football_pass_pitch_control` | função | `(ctx, inputs, params)` | Nó `football.pass_pitch_control`: pitch control contra regressão logística e árvores em passes de partidas de amostra da Metrica. |

#### `pinneapple_flow.nodes_ext.forecasting`

Nós de previsão: checagens do solver de ENSO (sem dados) e estudos de `forecast_lab` (caso, sobreajuste, Lyapunov de Mackey-Glass); alguns casos precisam de download e terminam `not_run` sem ele. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/forecasting.py` · 201 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `enso_solver_checks` | função | `(ctx, inputs, params)` | Nó `enso.solver_checks`: velocidades de ondas, ordem de convergência e resposta de Gill do modelo acoplado, e osciladores contra seus autovalores. |
| `forecast_case` | função | `(ctx, inputs, params)` | Nó `forecast_lab.case`: previsibilidade e todos os modelos num caso, com walk-forward purgado e teste intocado. |
| `lyapunov` | função | `(ctx, inputs, params)` | Nó `forecast_lab.lyapunov`: estimativa de Benettin do maior expoente de Lyapunov de Mackey-Glass. |
| `overfitting` | função | `(ctx, inputs, params)` | Nó `forecast_lab.overfitting`: pontuações de vários esquemas de validação em variantes sobreajustadas e regularizadas contra o teste intocado. |

#### `pinneapple_flow.nodes_ext.geomechanics`

Nós de colapso de coluna granular (MPM contra a lei de Lube et al. 2005), PINN de inspeção por correntes parasitas e biblioteca de componentes. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/geomechanics.py` · 169 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `lube_2005` | função | `(a) -> float` | Lei de espalhamento do colapso 2-D de Lube et al. (2005) usada como referência. |
| `column_collapse` | função | `(ctx, inputs, params)` | Nó `geo.column_collapse`: colapso de coluna de areia seca por MPM, espalhamento final contra a lei de Lube. |
| `eddy_current` | função | `(ctx, inputs, params)` | Nó `inspection.eddy_current`: varredura sintética de sonda por FDM e uma `InspectionPINN` treinada nela. |
| `component_fit` | função | `(ctx, inputs, params)` | Nó `systems.component_fit`: componentes e toolboxes registrados e um componente ajustado a alvos aleatórios com semente (demonstração da API, não da física). |

#### `pinneapple_flow.nodes_ext.lbm`

Nós de Lattice Boltzmann (`examples/lbm`): arrasto de esfera no túnel 3-D contra a lei de Stokes e contra Schiller & Naumann. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/lbm.py` · 153 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `sphere_stokes` | função | `(ctx, inputs, params)` | Nó `lbm.sphere_stokes`: esfera num cubo com campo de Stokes nas faces; arrasto por troca de momento contra 6 pi rho nu R U. |
| `sphere_drag` | função | `(ctx, inputs, params)` | Nó `lbm.sphere_drag`: Cd da esfera no túnel contra Schiller & Naumann (1933). |

#### `pinneapple_flow.nodes_ext.learned_sim`

Nó de MeshGraphNet treinada num prefixo dos dados `cylinder_flow` da DeepMind, com rollout na validação. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/learned_sim.py` · 112 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `mgn_cylinder_flow` | função | `(ctx, inputs, params)` | Nó `gnn.mgn_cylinder_flow`: velocidade no passo seguinte do desprendimento de vórtices, em rollout, contra a velocidade congelada. |

#### `pinneapple_flow.nodes_ext.non_hermitian`

Nós de física não-hermitiana (`examples/non_hermitian`): teorema no-go de sintonia de pontos excepcionais em dímeros PT-simétricos, checado simbólica e numericamente. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/non_hermitian.py` · 177 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ep_laws` | função | `(ctx, inputs, params)` | Nó `nonhermitian.ep_laws`: lei do ponto excepcional por classe de acoplamento e se o teorema se aplica. |
| `ep_numeric` | função | `(ctx, inputs, params)` | Nó `nonhermitian.ep_numeric`: gamma_EP² obtido do hamiltoniano contra o alvo linear e ponto fixo com realimentação. |
| `hypothesis_map` | função | `(ctx, inputs, params)` | Nó `nonhermitian.hypothesis_map`: três projetos auditados contra as hipóteses do teorema. |
| `ep_figure` | função | `(ctx, inputs, params)` | Nó `nonhermitian.ep_figure`: figura de quatro painéis e limiar de quebra PT em cadeias uniformes. |

#### `pinneapple_flow.nodes_ext.pde_env`

Nó do exemplo `pde_environment`: preset com seletor por etiqueta treinado via `solve_pde`, termos de perda por etiqueta antes e depois. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/pde_env.py` · 76 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `tagged_preset` | função | `(ctx, inputs, params)` | Nó `pde_env.tagged_preset`: termos de perda por etiqueta de um preset antes e depois do treino na geometria real. |

#### `pinneapple_flow.nodes_ext.pinn_examples`

Nós dos exemplos de chamada única: PINN de Burgers contra Cole-Hopf com `pp.solve` e FEM contra PINN em Poisson com `pp.compare`. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/pinn_examples.py` · 99 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `burgers_first` | função | `(ctx, inputs, params)` | Nó `pinn.burgers_first`: perfil da PINN de Burgers na frente íngreme contra a solução de Cole-Hopf. |
| `poisson_fem_vs_pinn` | função | `(ctx, inputs, params)` | Nó `pinn.poisson_fem_vs_pinn`: FEM e PINN numa chamada contra a solução exata de Poisson. |

#### `pinneapple_flow.nodes_ext.pinn_solver`

Nó que avalia a perda simbólica de `PINNFactory` num campo nulo (`examples/pinn_solver`). [sem teste direto]

<small>`pinneapple_flow/nodes_ext/pinn_solver.py` · 57 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `factory_loss` | função | `(ctx, inputs, params)` | Nó `pinn_solver.factory_loss`: perda de `PINNFactory` compilada pelo SymPy e avaliada no campo nulo. |

#### `pinneapple_flow.nodes_ext.problemdesign`

Nó de projeto de problema sem LLM (`examples/problem_designer`): validadores, plano FNO-first e relatório de um `ProblemSpec` parcial. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/problemdesign.py` · 84 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `offline_report` | função | `(ctx, inputs, params)` | Nó `problemdesign.offline_report`: validadores, plano e relatório de uma especificação parcial. |

#### `pinneapple_flow.nodes_ext.render`

Nós de render de cenas animadas com Blender Cycles e grade de clipes com barra de cor; o `bpy` roda em processo separado e o vídeo precisa de ffmpeg. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/render.py` · 161 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `blender_clip` | função | `(ctx, inputs, params)` | Nó `render.blender_clip`: clipe Cycles de cada cena com HUD, em MP4, mais uma imagem. |
| `clip_grid` | função | `(ctx, inputs, params)` | Nó `render.clip_grid`: clipes lado a lado com rótulos e uma barra de cor. |

#### `pinneapple_flow.nodes_ext.render_worker`

Script que roda o Blender em processo próprio para `nodes_ext.render`, porque importar `bpy` pode derrubar o processo e o Cycles prende o interpretador por minutos. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/render_worker.py` · 74 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check` | função | `() -> int` | Confere se o `bpy` importa e funciona. |
| `render` | função | `(scenes_pkl, options_json, out_dir) -> int` | Renderiza as cenas de um pickle com as opções JSON na pasta de saída. |
| `main` | função | `(argv) -> int` | Entrada de linha de comando do worker (`--check` ou render). |

#### `pinneapple_flow.nodes_ext.security`

Nó de segurança ciberfísica: resíduos físicos contra injeção de dados falsos na rede de três tanques, comparados com detectores baseados em dados. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/security.py` · 70 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `fdi_benchmark` | função | `(ctx, inputs, params)` | Nó `security.fdi_benchmark`: AUC, taxa de detecção, atraso e localização por detector, regime e ataque; a isolation forest precisa de scikit-learn. |

#### `pinneapple_flow.nodes_ext.solver_registry`

Nós sobre o `SolverRegistry` (`examples/solvers`): decomposições de séries temporais, caixa SPH e Poisson por FDM e espectral. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/solver_registry.py` · 211 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `decompose` | função | `(ctx, inputs, params)` | Nó `solvers.decompose`: soma de senos com ruído decomposta por cada solver registrado. |
| `sph_box` | função | `(ctx, inputs, params)` | Nó `solvers.sph_box`: trajetória SPH, caixa final e fração de partículas dentro, com aviso se saírem. |
| `poisson_fdm_spectral` | função | `(ctx, inputs, params)` | Nó `solvers.poisson_fdm_spectral`: Poisson por SOR e por FFT, a diferença entre eles e cada um contra a solução exata. |

#### `pinneapple_flow.nodes_ext.solvers`

Nós de `examples/numerical_solvers` e `examples/rigid_body`: gás de Lennard-Jones, ruptura de barragem em água rasa e caixa com atrito (precisa do NVIDIA Newton). [sem teste direto]

<small>`pinneapple_flow/nodes_ext/solvers.py` · 172 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `lj_gas` | função | `(ctx, inputs, params)` | Nó `md.lj_gas`: gás de Lennard-Jones com Verlet e energia total contra a inicial. |
| `dam_break` | função | `(ctx, inputs, params)` | Nó `swe.dam_break`: ruptura de barragem em água rasa com relatório de saúde (volume, positividade, finitude). |
| `crate_and_pendulum` | função | `(ctx, inputs, params)` | Nó `newton.crate_and_pendulum`: distância de parada da caixa com atrito de Coulomb contra v0²/(2 mu g); precisa do NVIDIA Newton. |

#### `pinneapple_flow.nodes_ext.structures`

Nós do CalculiX (`examples/calculix_cantilever`): deck, execução do `ccx` (nativo ou Docker) e leitura do `.frd` com comparação com Euler-Bernoulli. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/structures.py` · 146 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `deck` | função | `(ctx, inputs, params)` | Nó `calculix.deck`: deck de entrada da viga em balanço. |
| `solve` | função | `(ctx, inputs, params)` | Nó `calculix.solve`: roda o `ccx` no deck e falha o passo se o retorno não for zero. |
| `results` | função | `(ctx, inputs, params)` | Nó `calculix.results`: deflexão na ponta, maior \|sigma_xx\| e deflexão de Euler-Bernoulli P L³/(3 E I). |

#### `pinneapple_flow.nodes_ext.surrogate_inverse`

Nó de projeto inverso guiado por surrogate (`pinneapple_design.surrogate_inverse`, as ferramentas de `apps/*_ai`) com verificação na forma fechada. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/surrogate_inverse.py` · 59 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `surrogate_inverse` | função | `(ctx, inputs, params)` | Nó `design.surrogate_inverse`: treina o surrogate do caso, otimiza o projeto por ele e verifica na forma fechada. |

#### `pinneapple_flow.nodes_ext.use_cases`

Nó do problema inverso da aleta (`examples/use_cases/fin_convection_inverse`): coeficiente h por PINN e por mínimos quadrados. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/use_cases.py` · 137 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `pin_fin` | função | `(ctx, inputs, params)` | Nó `inverse.pin_fin`: h e o campo da aleta aprendidos juntos por PINN contra ajuste de mínimos quadrados da forma fechada. |

#### `pinneapple_flow.nodes_ext.validation`

Nós de validação: linter físico `pinneapple_check`, avaliação do model zoo, grafo de prova PT, arena de validação red/blue team e auditoria do solver de referência. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/validation.py` · 213 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `physics_check` | função | `(ctx, inputs, params)` | Nó `check.physics`: roda todas as regras do linter físico nos alvos e compara as regras quebradas com a lista do exemplo. |
| `zoo_evaluate` | função | `(ctx, inputs, params)` | Nó `zoo.evaluate`: reproduz as métricas do model card, checagens físicas, comparação com XFOIL e fine-tune protegido. |
| `pt_chain` | função | `(ctx, inputs, params)` | Nó `proof.pt_chain`: grafo de hipóteses, lemas e teorema com checagens exatas e numéricas para tetrâmeros PT. |
| `arena` | função | `(ctx, inputs, params)` | Nó `validate.arena`: red team procura saídas erradas que passam e blue team ajusta as checagens até pegar todas. |
| `audit` | função | `(ctx, inputs, params)` | Nó `validate.audit_model`: o solver de referência julgado pela defesa entregue (sem escapes nem falsos alarmes). |

#### `pinneapple_flow.nodes_ext.validation_gallery`

Nós da galeria de validação (`examples/validation`): um modelo bom e uma variante quebrada que uma checagem nomeada de `pp.validate` deve pegar. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/validation_gallery.py` · 197 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `pinn_heat` | função | `(ctx, inputs, params)` | Nó `gallery.pinn_heat`: PINN de calor e variante com peso de contorno zero, que `pinn.boundary` deve pegar. |
| `pod_dmd_wake` | função | `(ctx, inputs, params)` | Nó `gallery.pod_dmd_wake`: POD + DMD bom e quebrado na esteira, que `rollout.invariants` deve pegar. |
| `opinf_advection` | função | `(ctx, inputs, params)` | Nó `gallery.opinf_advection`: ROMs de advecção-difusão com base completa e cortada, que `rollout.spectrum` deve pegar. |
| `zoo_range` | função | `(ctx, inputs, params)` | Nó `gallery.zoo_range`: modelo do zoo dentro e fora da faixa de validade, que `applicability.range` deve pegar. |

#### `pinneapple_flow.nodes_ext.viz`

Nó "qualquer resultado em 3-D": lê arquivos CAE ou geometria, exporta glTF/USD, gera visualizador three.js e, com `bpy`, renderiza. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/viz.py` · 92 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `any_result_3d` | função | `(ctx, inputs, params)` | Nó `viz.any_result_3d`: carrega, exporta, visualiza e renderiza qualquer resultado em 3-D. |

#### `pinneapple_flow.nodes_ext.worldmodel`

Nó do modelo de mundo físico (`pinneapple_worldmodel`): datasets dos cenários, um especialista FNO por cenário e as tarefas de benchmark. [sem teste direto]

<small>`pinneapple_flow/nodes_ext/worldmodel.py` · 69 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `specialists` | função | `(ctx, inputs, params)` | Nó `worldmodel.specialists`: por cenário, parâmetros do especialista e erros de passo e rollout em cada tarefa. |

#### `pinneapple_flow.palette`

Descreve a paleta do Flow Studio: seções, título e resumo de cada nó, dica de instalação e busca que coloca nós curados acima dos `auto.*`.

<small>`pinneapple_flow/palette.py` · 299 linhas · testes: 29</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `place` | função | `(type_, section) -> None` | Coloca um tipo de nó registrado por outro módulo numa seção da paleta. |
| `sections` | função | `() -> list[dict[str, Any]]` | As seções da paleta em ordem, com título, ícone e descrição. |
| `is_auto` | função | `(spec) -> bool` | Diz se o tipo de nó é gerado (`auto.*`). |
| `section_of` | função | `(spec) -> str` | Seção de um tipo de nó: pela etiqueta `section:`, pelo mapa curado ou pela categoria. |
| `package_of` | função | `(spec) -> str` | Pacote da biblioteca que um nó envolve, pela primeira chamada declarada. |
| `humanize` | função | `(name) -> str` | Converte um nome em snake_case num título legível, mantendo siglas em maiúsculas. |
| `summary` | função | `(text, limit=110) -> str` | Primeira frase de uma docstring, sem marcação reST, cortada num limite de caracteres. |
| `install_hint` | função | `(reason) -> str` | O que fazer com um nó indisponível, a partir da razão. |
| `entry` | função | `(spec, avail=None) -> dict[str, Any]` | Campos de paleta de um tipo de nó (seção, título, resumo, etiquetas, pacote, curado, dica de instalação). |
| `score` | função | `(e, query) -> float` | Relevância de uma entrada do catálogo para uma busca; título pesa mais e nós curados ficam acima dos gerados. |
| `search` | função | `(query, catalog=None, limit=50) -> list[dict[str, Any]]` | Entradas do catálogo que casam com a busca, melhores primeiro. |
| `layout` | função | `(catalog=None) -> list[dict[str, Any]]` | A paleta inteira como seções em ordem com seus tipos de nó. |

#### `pinneapple_flow.planner`

Ponto de entrada `plan`: escolhe o planejador heurístico, o LLM ou `auto` (LLM se houver provedor configurado) e volta para a heurística quando o LLM falha, registrando o motivo.

<small>`pinneapple_flow/planner/__init__.py` · 57 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plan` | função | `(text, *, backend='auto', provider='', model='', base_url='', api_key='', n=3, call=None)…)` | Gera propostas de fluxo para uma descrição com o planejador escolhido, caindo para a heurística quando o LLM falha. |

#### `pinneapple_flow.planner.build`

`FlowBuilder` para montar fluxos em código (nós, arestas, decisões) e `layout`, que posiciona os nós da esquerda para a direita.

<small>`pinneapple_flow/planner/build.py` · 66 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FlowBuilder` | classe | `(name, description='')` | Construtor de fluxos em código: adiciona nós, liga portas, registra decisões e devolve o fluxo ou uma proposta. |
| `layout` | função | `(nodes, edges, dx=270.0, dy=150.0)` | Posiciona os nós em colunas pelo caminho mais longo desde uma fonte e em linhas pela ordem de aparição. |

#### `pinneapple_flow.planner.extensions`

Gancho público de extensão do planejador: outros pacotes declaram `Capability` (palavras-chave, nós, modelo de fluxo) e o planejador heurístico as propõe para descrições que casam, com uma decisão por escolha.

<small>`pinneapple_flow/planner/extensions.py` · 168 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Capability` | classe | `(id, title, domain, keywords, nodes, description, inputs, downstream, references, uncited…)` | Capacidade declarada para o planejador: id, título, domínio, palavras-chave com pesos, nós, referências, modelo de fluxo e pontuação mínima. |
| `register_capability` | função | `(cap) -> Capability` | Adiciona ou substitui (mesmo id) uma capacidade e sua regra de decisão. |
| `capabilities` | função | `() -> list[Capability]` | Todas as capacidades registradas, em ordem de id. |
| `score` | função | `(cap, text) -> tuple[float, list[str]]` | Pontuação de uma capacidade para um texto (soma dos pesos das palavras-chave encontradas), zero se faltar um grupo obrigatório. |
| `match` | função | `(text) -> list[tuple[float, Capability, list[str]]]` | Capacidades cuja pontuação atinge `min_score`, melhores primeiro. |
| `template_doc` | função | `(name, quick=False) -> dict[str, Any]` | Lê o fluxo de exemplo usado como modelo, opcionalmente na configuração rápida. |
| `proposal` | função | `(cap, text, det, hits, s) -> dict[str, Any]` | Monta a proposta de fluxo de uma capacidade com as decisões que explicam a escolha. |
| `proposals` | função | `(text, det=None, *, limit=2) -> list[dict[str, Any]]` | Fluxos das capacidades que melhor casam com o texto (no máximo `limit`). |
| `describe` | função | `() -> list[dict[str, Any]]` | Todas as capacidades em JSON, para o assistente da interface e o prompt do planejador LLM. |

#### `pinneapple_flow.planner.heuristic`

Planejador offline e determinístico: palavras-chave em inglês e português, grandezas com unidades e valores de coeficientes detectam a física e o pedido, e cada proposta é um fluxo do catálogo.

<small>`pinneapple_flow/planner/heuristic.py` · 573 linhas · testes: 11</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `detect` | função | `(text) -> dict[str, Any]` | Tudo o que as heurísticas leem no texto (família de EDP, dimensão, regime, condições, pedidos), mostrado ao usuário. |
| `family_length_ok` | função | `(family) -> bool` | Diz se a família aceita o comprimento do domínio dado (falso para `burgers`). |
| `plan_heuristic` | função | `(text, *, max_proposals=3) -> dict[str, Any]` | Plano heurístico completo: capacidades registradas primeiro, depois fluxos montados conforme a física e os pedidos detectados. |
| `sensor_count` | função | `(text) -> int \| None` | Lê no texto o número de sensores (1 a 64), em inglês ou português, ou `None`. |
| `sensor_layout` | função | `(n, dim) -> list` | Posições de `n` sensores espalhadas no intervalo ou quadrado unitário. |

#### `pinneapple_flow.planner.llm`

Planejador por LLM via `pinneapple_llm.call_llm` (Ollama, endpoint compatível com OpenAI, OpenAI ou Anthropic): catálogo e esquema vão no prompt, a resposta JSON é validada e reparada. [sem teste direto; os testes usam `plan` com `call` falso]

<small>`pinneapple_flow/planner/llm.py` · 228 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `catalog_prompt` | função | `() -> str` | Texto do catálogo de nós (portas, parâmetros e descrição) para o prompt. |
| `build_prompt` | função | `(text, n=2) -> str` | Prompt completo com o catálogo, o esquema e um exemplo de resposta JSON. |
| `repair` | função | `(flow_doc) -> tuple[dict[str, Any], list[str]]` | Reconstrói um fluxo válido a partir da saída do LLM e lista os reparos feitos. |
| `plan_llm` | função | `(text, *, provider, model='', base_url='', api_key='', n=2, call=None) -> dict[str, Any]` | Propostas de um LLM, cada uma validada e reparada; levanta erro em falha do provedor ou saída inútil. |

#### `pinneapple_flow.planner.wizard`

Assistente guiado de "New flow" (problema, física, geometria/dados, métodos, validação, saídas) que transforma respostas numa descrição para o planejador heurístico.

<small>`pinneapple_flow/planner/wizard.py` · 89 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `describe` | função | `(answers) -> str` | A descrição em texto que o planejador lê para as respostas do assistente. |
| `plan_from_answers` | função | `(answers, *, max_proposals=3) -> dict[str, Any]` | Resultado do planejador para as respostas do assistente, com a capacidade escolhida em primeiro lugar. |

#### `pinneapple_flow.references`

Carrega o registro curado `references.yaml`, aceita referências de outros pacotes, verifica DOIs online (doi.org e Crossref) e formata citações.

<small>`pinneapple_flow/references.py` · 216 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `registry` | função | `() -> dict[str, dict[str, Any]]` | Lê `references.yaml` mais as referências registradas por outros pacotes. |
| `register_references` | função | `(entries) -> list[str]` | Acrescenta referências curadas (dict ou YAML) checando o formato; recusa chave existente com conteúdo diferente. |
| `ref` | função | `(key) -> dict[str, Any]` | Uma referência do registro pela chave; `KeyError` se não existir. |
| `link` | função | `(r) -> str` | Link clicável de uma referência: DOI, senão URL, senão vazio. |
| `problems` | função | `(entry) -> list[str]` | Lista o que está errado numa entrada do registro (vazia se bem formada). |
| `verify_doi` | função | `(doi, title='') -> dict[str, Any]` | Resolve um DOI online (doi.org e Crossref) e devolve `verified`, `not found` ou `unverified` (sem rede). |
| `verify_registry` | função | `() -> dict[str, dict[str, Any]]` | Verifica online todas as entradas do registro que têm DOI. |
| `check` | função | `(reference, *, online=None) -> dict[str, Any]` | Classifica uma referência citada como `registry`, `verified`, `not found` ou `unverified`. |
| `cite` | função | `(*keys, supports='') -> list[dict[str, Any]]` | Referências do registro para as chaves dadas. |
| `format_reference` | função | `(r) -> str` | Formata uma referência como texto (autores, ano, título, veículo, link, estado). |

#### `pinneapple_flow.registry`

Define `Port`, `Param`, `NodeSpec` e `Registry` e o decorador `node`, que registra `fn(ctx, inputs, params)` como tipo de nó.

<small>`pinneapple_flow/registry.py` · 256 linhas · testes: 46</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Port` | classe | `(name, type, required, multiple, help)` | Porta de entrada ou saída de um nó: nome, tipo, se é obrigatória ou múltipla e ajuda. |
| `Param` | classe | `(name, kind, default, help, choices, minimum, maximum, max_length, advanced, label)` | Parâmetro de um nó com tipo, padrão, escolhas e limites; gera seu JSON Schema e checa valores. |
| `NodeSpec` | classe | `(type, title, category, fn, inputs, outputs, params, description, calls, requires, availa…)` | Especificação de um tipo de nó: função, portas, parâmetros, descrição, chamadas de biblioteca, requisitos e disponibilidade. |
| `Registry` | classe | `()` | Catálogo de tipos de nó: registrar, registrar nós do usuário, buscar e listar o catálogo. |
| `palette_entry` | função | `(spec, avail) -> dict[str, Any]` | Atalho para `palette.entry` de um `NodeSpec`. |
| `node` | função | `(type_, title, *, category, inputs=(), outputs=(), params=(), description='', calls='', r…)` | Decorador que registra `fn(ctx, inputs, params) -> {porta: valor}` como tipo de nó. |

#### `pinneapple_flow.rules`

The heuristic rules behind every automatic decision of a flow, with the references they rest on.

<small>`pinneapple_flow/rules.py` · 208 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Rule` | classe | `(id, description, cites, uncited, evidence, topic)` |  |
| `register_rule` | função | `(r) -> Rule` | Add a heuristic rule (another package's decisions). It must cite registered references (:func:`pinneapple_flow.references.register_references` first) or say why it has none; an existing id is refused unless the rule is the same. |
| `rule` | função | `(rule_id) -> Rule` |  |

#### `pinneapple_flow.runtime`

Converte cada nó num `Step` do motor de workflows (ordem, logs, retentativas, cancelamento e reaproveitamento de saídas inalteradas); valores grandes viram artefatos em arquivo e o fim de cada execução passa por `on_run_finished`.

<small>`pinneapple_flow/runtime.py` · 523 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NotRun` | classe | `(reason)` | Exceção levantada por um nó que não pode rodar aqui: ele termina `not_run` e os nós que dependem dele também. |
| `NodeContext` | classe | `(node_id, node_type, workdir, log, should_stop=lambda: False, meta=None)` | O que a função de um nó recebe além de entradas e parâmetros: pasta, log e ajudantes para métricas, tabelas, figuras, decisões e artefatos. |
| `FlowNodeAdapter` | classe | `()` | Adaptador do motor que roda um nó no processo: resolve entradas a partir das saídas anteriores e chama o tipo registrado. |
| `FlowRunResult` | classe | `(run_id, status, nodes, workdir, seconds, hooks)` | Resultado de uma execução: id, estado, nós, pasta, segundos e o que cada gancho fez; dá acesso a saídas e relatórios. |
| `default_workdir` | função | `() -> Path` | Pasta de trabalho padrão (`$PINNEAPPLE_FLOW_HOME/work` ou `~/.pinneapple_flow/work`). |
| `artifact` | função | `(path, kind) -> dict[str, Any]` | Cria o valor `{"$artifact": caminho, "kind": tipo}` que referencia um arquivo. |
| `is_artifact` | função | `(v) -> bool` | Diz se um valor é uma referência de artefato. |
| `load_artifact` | função | `(v)` | Lê o conteúdo de um artefato: npz como dict de arrays, json como objeto, pt como objeto torch, png como caminho. |
| `to_workflow` | função | `(flow, *, nodes=None, force=frozenset(), token='', max_parallel=1) -> Workflow` | Converte o fluxo no `Workflow` do motor; nós em `force` recebem um token novo para rodar de novo. |
| `node_states` | função | `(runner, run_id, *, with_logs=False) -> dict[str, dict[str, Any]]` | Estado, erro e saídas de cada nó de uma execução, marcando `not_run` e `degraded`. |
| `register_custom_nodes` | função | `(docs) -> list[str]` | Registra tipos de nó do usuário validados por `custom_spec`. |
| `run` | função | `(flow, *, workdir=None, from_node=None, only=None, run_id=None, cache=True, max_parallel=…)` | Roda um fluxo sem interface e devolve um `FlowRunResult`, com cache, re-execução a partir de um nó e paralelismo opcional. |
| `on_run_finished` | função | `(result, flow) -> None` | Ponto único por onde passa toda execução terminada; chama a exportação para o Lab e os ganchos registrados sem nunca falhar a execução. |
| `lab_export_status` | função | `() -> str` | Diz se `pinneapple_lab.manifest.from_flow_run` está disponível. |
| `export_to_lab` | função | `(result, flow) -> str` | Grava o manifesto de experimento da execução (`experiment.json`) com o `from_flow_run` do Lab. |

#### `pinneapple_flow.schema`

O documento de fluxo `pinneapple-flow/1` com `Flow`, `Node`, `Edge` e `Issue`; `validate_flow` acusa tipos e parâmetros desconhecidos, portas incompatíveis, entradas faltando e ciclos.

<small>`pinneapple_flow/schema.py` · 307 linhas · testes: 27</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FlowError` | classe | `(message, issues=None)` | Exceção para documento de fluxo malformado ou inválido, com a lista de problemas. |
| `Node` | classe | `(id, type, params, position, label)` | Um nó do fluxo: id, tipo, parâmetros, posição e rótulo. |
| `Edge` | classe | `(source, source_port, target, target_port, id)` | Uma aresta do fluxo: nó e porta de origem, nó e porta de destino. |
| `Flow` | classe | `(name, nodes, edges, id, description, meta, schema)` | O documento de fluxo com nós, arestas e metadados, com ordem topológica, ancestrais e leitura/gravação em JSON. |
| `Issue` | classe | `(level, message, node, edge, param)` | Um problema de validação com nível, mensagem e o nó, aresta ou parâmetro envolvido. |
| `as_flow` | função | `(flow) -> Flow` | Obtém um `Flow` a partir de um Flow, dict, string JSON ou caminho de arquivo. |
| `validate_flow` | função | `(flow, registry=None) -> list[Issue]` | Lista todos os problemas de um fluxo; lista vazia significa válido. |
| `check_flow` | função | `(flow, registry=None) -> Flow` | Devolve o fluxo ou levanta `FlowError` com os erros (avisos passam). |
| `flow_json_schema` | função | `() -> dict[str, Any]` | JSON Schema (draft 2020-12) do documento de fluxo. |

#### `pinneapple_flow.service`

The operations Flow Studio (or any other front end, e.g. the PINNeAPPle-CFD app) exposes, as plain methods.

<small>`pinneapple_flow/service.py` · 352 linhas · testes: 29</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FlowService` | classe | `(*, db_url=None, workdir=None, settings=None, max_runs=1)` |  |

#### `pinneapple_flow.store`

Where flows, their versions, runs and per-node outputs live: SQLite by default, any SQLAlchemy URL otherwise.

<small>`pinneapple_flow/store.py` · 261 linhas · testes: 23</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FlowStore` | classe | `(url=None, *, table_prefix=None)` | ``table_prefix``: put the tables under a prefix (``flow_`` -> ``flow_flows``, ``flow_runs``, ...) when they share an application's database; default ``$PINNEAPPLE_FLOW_TABLE_PREFIX`` or none. |
| `default_url` | função | `() -> str` |  |
| `redact` | função | `(url) -> str` | The URL with any password replaced (for logs and the GUI). |

#### `pinneapple_flow.studio`

Serve o app `apps/flow_studio` com uvicorn; recusa ouvir fora de localhost sem `FLOW_USER` e hash de senha, salvo `allow_insecure`.

<small>`pinneapple_flow/studio.py` · 64 linhas · testes: 38</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `app_dir` | função | `() -> Path` | Pasta do app Flow Studio (`$PINNEAPPLE_FLOW_APP` ou `apps/flow_studio` do checkout). |
| `load_app` | função | `()` | Carrega o app FastAPI de `apps/flow_studio/backend/app.py` (precisa do checkout do repositório). |
| `serve` | função | `(host=None, port=None, *, open_browser=True, env_file='.env', allow_insecure=False) -> int` | Sobe o Flow Studio com uvicorn e abre o navegador; recusa endereço não local sem login. |

#### `pinneapple_flow.templates`

Equation templates per PDE family and dimension: the PDEs, conditions, coefficients and (when one is known) the exact solution a planner proposes and a user then edits.

<small>`pinneapple_flow/templates.py` · 114 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `template` | função | `(family, dimension=1) -> dict[str, Any]` | PDEs, conditions, coefficients, domain and exact solution for ``family`` in ``dimension`` (1-3) spatial dimensions. ValueError for an unknown family or a dimension the family does not have here. |
| `retarget` | função | `(equations, dimension) -> dict[str, Any]` | The equations of a template family moved to another dimension (the 2-D <-> 3-D switch). PDEs written with ``lap()`` keep their text; conditions on ``boundary`` stay; a known template is rebuilt for the new dimension. |

#### `pinneapple_flow.twin`

`PDETwinModel` transforma o sistema identificado num modelo de gêmeo (campo em grade grossa, coeficientes como estados de passeio aleatório, um estado por sensor); `simulate_sensors` e `run_twin` simulam e assimilam leituras. Só primeira ordem no tempo.

<small>`pinneapple_flow/twin.py` · 281 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PDETwinModel` | classe | `(system, bounds, *, n=21, dt=0.01, estimate=None, coefficients=None)` | O sistema identificado como modelo de gêmeo: estados do campo em grade grossa, coeficientes estimados e sensores interpolados. |
| `simulate_sensors` | função | `(plant, *, sensors, t_end, every=1, noise=0.0, coefficients=None, fault=None, seed=0) -> …)` | Leituras simuladas de um ativo (em geral uma grade mais fina), com ruído gaussiano e falha opcional num instante. |
| `run_twin` | função | `(model, readings, *, sensors, sigma_obs, sigma_process=0.0001, sigma_param=None, limits=N…)` | Assimila leituras no modelo com o `TwinRuntime` e devolve o runtime e um resumo (estimativas, alertas, coeficientes calibrados). |

#### `pinneapple_flow.types`

Port types of the flow graph and the rule that says which output may feed which input.

<small>`pinneapple_flow/types.py` · 71 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `compatible` | função | `(source, target) -> bool` | True when an output of type ``source`` may feed an input of type ``target``. |
| `register_port_type` | função | `(name, description, *, converts_to=(), converts_from=()) -> None` | Add a port type (for another package's node types, e.g. a CFD case). ``converts_to`` / ``converts_from``: existing types it may feed / be fed from although the names differ. Re-registering the same description is a no-op; a different one for an existing name is refused, so documents stay portable. |

#### `pinneapple_flow.wrappers`

Funções com argumentos JSON que montam os objetos com os construtores da própria biblioteca e chamam a biblioteca; nada aqui calcula física. [sem teste direto]

<small>`pinneapple_flow/wrappers.py` · 292 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `solve_preset` | função | `(preset='burgers_1d', method='analytic', architecture='modified_mlp', epochs=50, hidden_d…)` | `pp.solve` num preset da biblioteca; para `pinn`, a arquitetura é qualquer variante PINN do registro de modelos. |
| `compare_preset` | função | `(preset='burgers_1d', methods=('analytic', 'pinn'), reference='analytic', architecture='m…)` | `pp.compare` de métodos num preset contra uma referência nos mesmos pontos. |
| `lbm_channel` | função | `(nx=64, ny=32, Re=100.0, u_in=0.05, steps=200, save_every=50) -> dict` | Escoamento em canal por Lattice Boltzmann via `SolverRegistry.build('lbm')`. |
| `beam_deflection` | função | `(L_m=1.0, E_Pa=200000000000.0, I_m4=8.33e-06, q_N_per_m=1000.0, nx=200, bc_type=0) -> dict` | Deflexão de viga de Euler-Bernoulli por diferenças finitas (`beam_bvp_fdm`). |
| `xtfc_pde` | função | `(pde='burgers_1d', n_basis=60, n_query=40) -> dict` | Resolve uma EDP com o solver X-TFC (`xtfc_pde`). |
| `wheel_on_soil` | função | `(soil='GRC-1', radius_m=0.25, width_m=0.2, load_N=100.0, slip=0.2, gravity=1.62) -> dict` | Modelo roda-solo de Wong-Reece em regime permanente (`pinneapple_simulation.terramechanics`). |
| `meshless_compare` | função | `(kind='poisson', exact='sin(pi*x)*sin(pi*y)', domain='square', methods=…, k=4.0, alpha=1.…)` | Compara métodos sem malha e com malha num problema (`meshless_study`), com checagem das expressões pela biblioteca. |
| `rom_compare` | função | `(Y=None, A=None, methods=(), ranks=(), budget_s=5.0, seed=0) -> dict` | Compara modelos de ordem reduzida em snapshots paramétricos (`rom_study`). |
| `parametric_pod` | função | `(A=None, Y=None, r=16, regressor='rbf', test_fraction=0.2, seed=0) -> dict` | Ajusta `ParametricPOD` a um dataset e mede o L2 relativo em amostras separadas. |
| `surrogate_study` | função | `(X=None, Y=None, example='none', methods=('ridge', 'gp', 'gbt'), budget=10.0, seeds=1) ->…)` | Compara surrogates clássicos (ridge, GP, árvores, rede) com `pinneapple_analysis.surrogate.run_study`. |
| `calculix_cantilever` | função | `(length_m=1.0, b_m=0.1, h_m=0.1, nx=20, element_type='C3D20R', run=True) -> dict` | Escreve o deck de uma viga em balanço com a biblioteca e roda o CalculiX se `ccx` (ou sua imagem Docker) existir; senão só escreve o deck. |
| `fmu_simulate` | função | `(file='', stop_time=1.0, step_size=0.01) -> dict` | Simula um FMU da pasta de dados (precisa de fmpy). |
| `newton_validate` | função | `(solver='xpbd', cases=…) -> dict` | Valida solvers de corpo rígido contra formas fechadas (precisa do NVIDIA Newton). |
| `openfoam_available` | função | `() -> dict` | Diz se o OpenFOAM está instalado (`blockMesh` no PATH). |
| `model_architecture` | função | `(name='fno', in_dim=1, out_dim=1, hidden=64) -> dict` | Constrói qualquer modelo do registro com argumentos padrão e tamanhos dados e conta seus parâmetros. |
| `doe_design` | função | `(parameters=None, n=16, method='lhs', seed=0, out_dir='') -> dict` | Plano de experimentos com `pinneapple_data.doe.design` sobre um espaço de parâmetros, gravado como CSV. |
| `geometry_export` | função | `(name='caarc_building', fmt='stl', lod='light', out_dir='') -> dict` | Constrói uma geometria paramétrica de `pinneapple_design.geometry_library` e a exporta (STL, OBJ, glTF, USD ou STEP). |
| `design_intent` | função | `(text=…, seed=0, budget=200, n_screen=500, out_dir='') -> dict` | `pinneapple_design.intent.design`: de um objetivo em linguagem natural a um projeto verificado, às perguntas que faltam ou à recusa. |

### Exemplos de uso reais do repositório

**Rodar um fluxo, reaproveitar o cache e re-executar a partir de um nó** — `tests/test_flow_runtime.py`

```python
def test_run_measures_against_exact_then_caches(home):
    f = heat_flow()
    r = pf.run(f, workdir=home / "w")
    assert r.status == "done", {k: v["error"] for k, v in r.nodes.items()}
    rel = r.report("val")["metrics"]["FD_rel_l2"]["value"]
    assert rel < 1e-3
    sol = r.load("fd", "solution")
    assert sol["values"].shape == (41, 11, 1)
    r2 = pf.run(f, workdir=home / "w")
    assert {v["status"] for v in r2.nodes.values()} == {"cached"}
    r3 = pf.run(f, workdir=home / "w", from_node="fd")
    st = {k: v["status"] for k, v in r3.nodes.items()}
    assert st["fd"] == "done" and st["val"] == "done" and st["geo"] == "cached"
```

**Toda proposta do planejador heurístico valida contra o catálogo** — `tests/test_flow_planner.py`

```python
def test_every_proposal_validates(text):
    res = plan(text, backend="heuristic")
    assert res["planner"] == "heuristic" and res["proposals"]
    for p in res["proposals"]:
        assert [i for i in pf.validate_flow(p["flow"]) if i.level == "error"] == []
        assert p["flow"]["meta"]["planner"] == "heuristic"
```

**Nó indisponível termina not_run e propaga para baixo** — `tests/test_flow_extensions.py`

```python
def test_unavailable_node_finishes_not_run_and_downstream_too(tmp_path):
    f = pf.Flow("x", nodes=[{"id": "a", "type": "test.missing_tool"}, {"id": "b", "type": "test.consumer"}],
                edges=[E("a", "b")])
    r = pf.run(f, workdir=tmp_path)
    assert r.nodes["a"]["status"] == "not_run" and "footool" in r.report("a")["not_run_reason"]
    assert r.nodes["b"]["status"] == "not_run" and "did not run" in r.report("b")["not_run_reason"]
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 31 de 89.
- Funções de teste em arquivos que importam algum módulo do pacote: 263.

### Limitações conhecidas

- Quase todos os módulos de `nodes` e `nodes_ext` não têm teste que os importe diretamente; são exercitados pelos testes de fluxos de exemplo e de execução, não por testes unitários próprios.
- Vários nós dependem de ferramentas externas e ficam indisponíveis sem elas: OpenFOAM (`solver.openfoam`, `cfd.*`), gmsh + GetDP (`em.prius_getdp_crosscheck`, `em.prius_end_effects`), NVIDIA Newton, Blender `bpy` e ffmpeg, CalculiX `ccx`, fmpy, scikit-learn, statsmodels.
- Alguns nós baixam dados na primeira execução (ENSO, partidas de amostra da Metrica, casos de `forecast_lab`, PINN do model zoo) e terminam `not_run` quando o download não é possível.
- `runtime.on_run_finished` diz que a exportação para o PINNeAPPle Lab só funciona quando `pinneapple_lab.manifest.from_flow_run` existe; senão `lab_export_status` informa que não está disponível e nada é gravado. `lab_status` também detecta a API do Lab em tempo de execução.
- `twin.PDETwinModel` só aceita sistemas de primeira ordem no tempo.
- Os nós de `nodes_ext` e `examples.example_dir` precisam da pasta `examples` de um checkout (ou `PINNEAPPLE_REPO`); numa wheel instalada eles falham. `studio.load_app` também depende do checkout com `apps/flow_studio`.
- `studio.serve` recusa ouvir fora de localhost sem `FLOW_USER` e hash de senha, salvo `allow_insecure` explícito.
- `learn` e `planner.llm` não têm teste direto; o planejador LLM depende de um provedor configurado e volta para a heurística quando falha.

## 4. `pinneapple_hub`

### Objetivo

`pinneapple_hub` reúne tudo o que o PINNeAPPle publica e distribui pronto: modelos treinados e verificados, datasets verificados, cartões de modelo e o índice do Hub de ferramentas. O pacote raiz expõe só o cliente genérico de hub: `push_to_hub` e `from_pretrained` sobre a infraestrutura do Hugging Face Hub (sem hospedagem própria), mais o `ModelCard`, contrato de reprodutibilidade em que `validation_metrics` calculadas são obrigatórias. `verified_card` cria esses cartões só a partir de uma rodada do `PhysicsGuardrail` de `pinneapple_llm`.

O subpacote `zoo` é o que aparece como `pp.zoo` e `pp.datasets`: modelos pré-treinados (aerofólio, dissipadores de calor, contato de roda de rover) com faixa de validade, métricas contra referência guardada, checagens físicas que alimentam um `TrustReport` de `pinneapple_analysis`, fine-tune com guarda no conjunto de teste original e benchmark via `pp.benchmark`. Os arquivos são verificados por SHA-256 contra o `manifest.json`, cacheados e baixados sob demanda do Hugging Face Hub. O zoo também exporta ONNX (com as checagens por consulta dentro do grafo), safetensors, Parquet para o visualizador de datasets do Hub, cartões README, o Space estático de demonstração e o pacote web do Playground.

O subpacote `library` gera o índice do Hub (app `apps/hub`): coleta ferramentas (funções públicas, nós do Flow Studio, receitas, ferramentas MCP), modelos, geometrias, datasets, fluxos e exemplos do repositório, cada item com um snippet executável cuja saída gravada é conferida por teste. Ele se liga a `pinneapple_flow` (nós, fluxos embutidos, links para o Flow Studio, composição de ferramentas próprias) e a `pinneapple_design` (biblioteca de geometrias).

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 31 | 7.260 | 22 | 170 | 183 (18 arquivos) | 8 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `hub` | `push_to_hub` e `from_pretrained`: envio e reconstrução de modelos no Hugging Face Hub com config e cartão. | 1 | 2 | 0 |
| `library` | Índice do Hub de ferramentas: coleta de itens, snippets executáveis, execuções gravadas, busca, composição e links com o Flow Studio. | 10 | 64 | 22 |
| `model_card` | `ModelCard`: contrato de reprodutibilidade com métricas de validação obrigatórias. | 1 | 1 | 21 |
| `verified_card` | Cartões de modelo cujas métricas vêm só de uma rodada do guardrail físico; recusa os demais. | 1 | 4 | 6 |
| `zoo` | `pp.zoo` e `pp.datasets`: modelos e datasets verificados, com checagens, exportação e publicação no Hugging Face Hub. | 17 | 121 | 125 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_hub`

Pacote raiz: reexporta `ModelCard`, `push_to_hub` e `from_pretrained`.

<small>`pinneapple_hub/__init__.py` · 12 linhas · testes: 122</small>

#### `pinneapple_hub.hub`

Envio e download de modelos PINNeAPPle no Hugging Face Hub, com o arquivo lateral `pinneapple_config.json` que permite reconstruir a arquitetura; exige o extra `hub` (`huggingface_hub`). [sem teste]

<small>`pinneapple_hub/hub.py` · 151 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `push_to_hub` | função | `(model, repo_id, model_card, *, architecture=None, architecture_config=None, private=Fals…)` | Envia pesos, config de arquitetura e cartão de modelo de um modelo treinado para um repositório do Hugging Face Hub. |
| `from_pretrained` | função | `(repo_id, *, revision='main', device='cpu', token=None)` | Baixa e reconstrói, via `ModelRegistry`, um modelo enviado com `push_to_hub`. |

#### `pinneapple_hub.library`

Índice do Hub: todos os itens prontos do repositório numa lista gerada (`data/web/hub/index.json`), com busca, facetas, destaques, execução e verificação dos snippets.

<small>`pinneapple_hub/library/__init__.py` · 303 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `collect` | função | `(only=None) -> list[dict[str, Any]]` | Itens de todas as fontes (`sources.COLLECTORS`), ids únicos, ordenados por tipo e título. |
| `run_file` | função | `(item_id, runs=RUNS) -> Path` | Arquivo que guarda a execução gravada de um item (`snippet_runs/<id>.json`). |
| `read_runs` | função | `(path=RUNS) -> dict[str, Any]` | Todas as execuções gravadas, por id de item. |
| `write_run` | função | `(item_id, rec, runs=RUNS) -> Path` | Grava a execução de um item no seu arquivo em `snippet_runs/`. |
| `attach_runs` | função | `(items, runs) -> None` | Liga a cada item a execução gravada do snippet atual (mesmo hash), senão a do exemplo da referência, senão `not run`. |
| `facets` | função | `(items) -> dict[str, Any]` | Contagens por tipo, subtipo, domínio, categoria e qualidade, para os filtros. |
| `quality_key` | função | `(it) -> str` | Valor de filtro da qualidade de um item: nota do Lab, decisão do zoo ou "not graded". |
| `featured` | função | `(items, n=8) -> list[str]` | Destaques por regra fixa: modelos aprovados do zoo, exemplos com selo A e figuras, geometrias com miniatura, depois fluxos. |
| `build` | função | `(only=None, runs=None) -> dict[str, Any]` | Monta o índice inteiro em memória (sem gravar). |
| `write_index` | função | `(index, path=INDEX) -> Path` | Grava o índice em JSON. |
| `load_index` | função | `(path=INDEX, *, rebuild=False) -> dict[str, Any]` | Lê o índice; o padrão é um cache reconstruído das fontes quando falta ou com `rebuild=True`. |
| `get` | função | `(index, item_id) -> dict[str, Any]` | Devolve um item do índice pelo id. |
| `score` | função | `(it, query) -> float` | Relevância de um item para uma consulta (0 = sem match); todas as palavras devem casar, com peso maior no título. |
| `search` | função | `(index, query='', *, kind=None, domain=None, quality=None, subkind=None, category=None, l…)` | Itens que casam com a consulta e os filtros (tipo, domínio, qualidade, subtipo, categoria), ordenados por relevância. |
| `record_runs` | função | `(items, *, stale_only=False, timeout=300.0, ids=None, workers=4, log=print) -> dict[str, …)` | Roda os snippets (todos, só os desatualizados ou ids escolhidos) em paralelo no sandbox e grava as saídas. |
| `check` | função | `(index) -> list[str]` | Problemas de um índice: snippet sem execução aprovada do código atual, arquivo ou link inexistente, entre outros. |

#### `pinneapple_hub.library.__main__`

Linha de comando `python -m pinneapple_hub.library build | run | check` (e `flowruns`). [sem teste]

<small>`pinneapple_hub/library/__main__.py` · 47 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `main` | função | `(argv=None) -> int` | Ponto de entrada da linha de comando do índice do Hub. |

#### `pinneapple_hub.library.categories`

As categorias de tarefa das ferramentas do Hub: cada ferramenta recebe exatamente uma, pela regra de prefixo mais longa sobre a chave de origem; receitas são listadas uma a uma.

<small>`pinneapple_hub/library/categories.py` · 174 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `key` | função | `(it) -> str` | Chave de origem de uma ferramenta, no formato que as regras de prefixo leem (`node:`, `lib:`, `fn:`, `mcp:`). |
| `category_of` | função | `(it) -> str \| None` | Slug da categoria de uma ferramenta, ou `None` se nenhuma regra casa (um teste proíbe isso). |
| `table` | função | `() -> list[dict[str, str]]` | Tabela das categorias (slug, título e descrição). |

#### `pinneapple_hub.library.compose`

"Crie a sua": compõe uma ferramenta a partir de blocos existentes (tipos de nó do Flow Studio, fluxo inicial opcional, geometria) e devolve o documento de fluxo e o Python que o roda.

<small>`pinneapple_hub/library/compose.py` · 191 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ComposeError` | classe | `()` | Erro de composição (`ValueError`): especificação inválida ou blocos demais. |
| `blocks` | função | `() -> list[dict[str, Any]]` | Blocos que o compositor oferece: todos os tipos de nó da paleta do Flow Studio, curados primeiro. |
| `compose` | função | `(spec) -> dict[str, Any]` | Monta o fluxo a partir da especificação (nome, blocos, início, parâmetros expostos), liga entradas e valida. |
| `code_for` | função | `(doc, *, geometry='', workdir='hub_tool', expose=None) -> str` | Python que reconstrói o fluxo, valida, roda sem interface e imprime as métricas de cada nó. |

#### `pinneapple_hub.library.flowlink`

Liga o Hub ao Flow Studio sobre o mesmo registro: monta um fluxo pronto em volta de qualquer tipo de nó e gera links entre os dois apps.

<small>`pinneapple_hub/library/flowlink.py` · 173 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FlowLinkError` | classe | `()` | Erro ao montar um fluxo para um tipo de nó (por exemplo tipo desconhecido). |
| `bundled_flows_using` | função | `(type_) -> list[str]` | Nomes dos fluxos embutidos que contêm um nó do tipo dado. |
| `flow_for_node` | função | `(type_, params=None, *, report=True) -> dict[str, Any]` | Monta entradas → ferramenta → relatório para um tipo de nó, reaproveitando um fluxo embutido ou escolhendo produtores para as entradas. |
| `b64url` | função | `(obj) -> str` | Codifica um objeto JSON em base64 para URL. |
| `from_b64url` | função | `(text) -> Any` | Decodifica o base64 de URL de volta para o objeto JSON. |
| `flow_link` | função | `(flow, *, base) -> str` | URL do Flow Studio que abre um fluxo (`?flow=`). |
| `add_link` | função | `(type_, params=None, *, base) -> str` | URL do Flow Studio que adiciona um nó a um fluxo escolhido pelo usuário (`?add=`). |

#### `pinneapple_hub.library.flowruns`

Execuções gravadas dos fluxos embutidos (`pinneapple_flow/flows`), mostradas na página de cada fluxo e de cada tipo de nó usado; válidas enquanto o SHA-256 do arquivo do fluxo não muda.

<small>`pinneapple_hub/library/flowruns.py` · 109 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `flow_hash` | função | `(name) -> str` | Hash SHA-256 curto do arquivo JSON de um fluxo embutido. |
| `read_flow_runs` | função | `(path=FLOW_RUNS) -> dict[str, Any]` | Lê `flow_runs.json` com as execuções gravadas. |
| `current` | função | `(runs, name) -> dict[str, Any] \| None` | A execução gravada de um fluxo, só se foi gravada para o arquivo como está agora. |
| `record` | função | `(names=None, *, timeout=900.0, workers=2, log=print) -> dict[str, Any]` | Roda os fluxos embutidos no sandbox e grava `flow_runs.json`. |
| `runs_for_node` | função | `(runs, type_) -> list[dict[str, Any]]` | Cada nó de um tipo nas execuções atuais: fluxo, id do nó, status, métricas e resumo. |

#### `pinneapple_hub.library.myhub`

"Meu Hub": ferramentas compostas e salvas por uma pessoa, como itens do Hub numa pasta local (um JSON cada).

<small>`pinneapple_hub/library/myhub.py` · 87 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MyHubError` | classe | `()` | Erro do Meu Hub (`ValueError`), por exemplo limite de ferramentas atingido ou ferramenta grande demais para salvar. |
| `MyHub` | classe | `(root)` | Pasta local de ferramentas próprias com `save`, `list`, `get` e `delete`. |

#### `pinneapple_hub.library.sandbox`

Roda um snippet num interpretador separado e de vida curta, guardando o que imprimiu e as figuras; tem limites de tempo e, em POSIX, de memória, CPU e arquivo, mas não bloqueia rede.

<small>`pinneapple_hub/library/sandbox.py` · 104 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `code_hash` | função | `(code) -> str` | SHA-256 curto de um snippet; uma execução gravada só é atual enquanto o snippet tem esse hash. |
| `run_snippet` | função | `(code, *, timeout=60.0, memory_mb=None, cpu_s=None, max_figures=4, env=None) -> dict[str,…)` | Executa o código em diretório temporário com matplotlib Agg, sem GPU e com limites, e devolve `ok`, stdout, segundos e figuras PNG. |

#### `pinneapple_hub.library.snippets`

Código mínimo e executável de cada item do Hub, rodado por `python -m pinneapple_hub.library run` e pelos testes; curto, offline e rápido. [sem teste]

<small>`pinneapple_hub/library/snippets.py` · 121 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `model` | função | `(name, example) -> str` | Snippet de um modelo do zoo: carrega, prevê num exemplo e valida. |
| `dataset` | função | `(name, split) -> str` | Snippet de um dataset: carrega um split ou mostra informações e licença. |
| `geometry` | função | `(name) -> str` | Snippet de uma geometria da biblioteca de geometrias. |
| `flow` | função | `(name) -> str` | Snippet que carrega e roda um fluxo embutido. |
| `node` | função | `(type_) -> str` | Snippet de um tipo de nó do Flow Studio. |
| `auto_call` | função | `(target, node_type, defaults=None) -> str` | Chamada gerada de uma função da biblioteca com os padrões do formulário do nó. |
| `api` | função | `(dotted) -> str` | Snippet de uma função pública pelo nome pontilhado. |
| `mcp` | função | `(name, *, call) -> str` | Snippet de uma ferramenta do servidor MCP. |
| `example` | função | `(folder) -> str` | Snippet de uma pasta de exemplo. |
| `lab_item` | função | `(key) -> str` | Snippet de um item do Lab. |
| `symbols` | função | `(code) -> list[str]` | Nomes `pp.xxx` chamados por um snippet (para tags e links). |

#### `pinneapple_hub.library.sources`

Coletores dos itens do Hub: cada um lê uma fonte existente do repositório (manifesto do zoo, biblioteca de geometrias, fluxos, registro de nós, AGENTS.md, servidor MCP, exemplos); nada é escrito à mão. [sem teste]

<small>`pinneapple_hub/library/sources.py` · 579 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `rel` | função | `(p) -> str` | Caminho relativo à raiz do repositório, em formato POSIX. |
| `domain` | função | `(name, tags=()) -> str` | Domínio físico de um item pela regra do Lab (`pinneapple_lab.manifest.domain_of`). |
| `used_by_examples` | função | `(*needles) -> list[str]` | Pastas de exemplo cujos scripts contêm um dos termos dados como string ou chamada. |
| `models` | função | `() -> list[dict[str, Any]]` | Itens de modelos do zoo (manifesto, cartões, exemplo do teste, grafos ONNX do navegador). |
| `datasets` | função | `() -> list[dict[str, Any]]` | Itens dos datasets do manifesto do zoo. |
| `geometries` | função | `() -> list[dict[str, Any]]` | Itens da biblioteca de geometrias de `pinneapple_design`. |
| `flows` | função | `() -> list[dict[str, Any]]` | Itens dos fluxos embutidos do Flow Studio. |
| `nodes` | função | `() -> list[dict[str, Any]]` | Itens de cada tipo de nó da paleta do Flow Studio (curados e chamadas de biblioteca geradas). |
| `node_for_callable` | função | `() -> dict[str, str]` | Mapa `módulo:função` → tipo de nó que a envolve. |
| `recipes` | função | `() -> list[dict[str, Any]]` | Itens de receitas: blocos Python do AGENTS.md e exemplos da referência de módulos. |
| `apis` | função | `() -> list[dict[str, Any]]` | Um item por linha da tabela "Use these APIs first" do AGENTS.md. |
| `agent_tools` | função | `() -> list[dict[str, Any]]` | Ferramentas do servidor MCP, lidas do código-fonte (funções `@server.tool()`). |
| `examples` | função | `() -> list[dict[str, Any]]` | Itens das pastas de exemplos. |

#### `pinneapple_hub.model_card`

`ModelCard`: contrato de reprodutibilidade de todo modelo enviado ao hub (arquitetura, config de treino, linhagem dos dados, métricas de validação calculadas e decisão de confiança).

<small>`pinneapple_hub/model_card.py` · 180 linhas · testes: 21</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ModelCard` | classe | `(name, description, task, version, inputs, outputs, validity_range, known_limitations, ar…)` | Cartão de modelo com tarefa, entradas/saídas, faixa de validade, limitações, arquitetura, equações e métricas; `validate()` lista problemas e há exportação para dict, JSON e Markdown. |

#### `pinneapple_hub.verified_card`

Cartões de modelo cujas `validation_metrics` vêm só de uma rodada do `PhysicsGuardrail` contra o modelo; cartão que não passa em `ModelCard.validate()` é recusado.

<small>`pinneapple_hub/verified_card.py` · 122 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CardRefused` | classe | `(problems)` | Exceção: o cartão gerado do guardrail não passou em `ModelCard.validate()`; `.problems` diz por quê. |
| `validation_metrics_from_report` | função | `(report) -> dict[str, float]` | Converte um relatório do guardrail em métricas: valor de cada checagem numérica e `<nome>_passed` (1.0/0.0) para as de passa/falha. |
| `card_from_guardrail` | função | `(spec, report, *, name, preset, preset_kwargs=None, architecture, architecture_config, tr…)` | Monta um `ModelCard` a partir de um relatório do guardrail e levanta `CardRefused` se ele não valida. |
| `certify_model` | função | `(preset, name, *, preset_kwargs=None, architecture='modified_mlp', architecture_config=No…)` | Constrói ou carrega um modelo para um preset, treina se não há pesos, roda o guardrail e devolve o cartão verificado. |

#### `pinneapple_hub.zoo`

`pp.zoo`: modelos pré-treinados e verificados em uma linha (listar, carregar, prever, avaliar, validar, consultar checagens, exportar, fazer benchmark e fine-tune com guarda).

<small>`pinneapple_hub/zoo/__init__.py` · 341 linhas · testes: 125</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FinetuneRejectedError` | classe | `()` | Exceção: o modelo ajustado ficou pior que o original no teste original (use `force=True` para salvar mesmo assim). |
| `FinetuneResult` | classe | `(model, original, before, after, accepted, tolerance, history, new_data_metrics, saved_to)` | Resultado de `finetune`: modelo novo, original, métricas antes/depois, se foi aceito, tolerância, histórico e onde foi salvo. |
| `export_onnx` | função | `(name, path, *, checks=False, check_parity=True, offline=None) -> dict[str, Any]` | Exporta um modelo para ONNX com pré e pós-processamento dentro do grafo e confere paridade com o PyTorch. |
| `onnx_parity` | função | `(name, onnx_model, X=None, *, offline=None) -> float` | Maior diferença escalada entre um grafo ONNX e o modelo PyTorch. |
| `query_checks` | função | `(name, inputs=None, *, offline=None, **named) -> builtins.list` | Checagens de confiança e veredito para cada linha de entrada. |
| `check_spec` | função | `(name, *, offline=None) -> dict[str, Any]` | Especificação declarativa das checagens por consulta de um modelo. |
| `export_web` | função | `(name, directory, *, offline=None) -> dict[str, Any]` | Grafo ONNX com checagens mais o JSON de cartão e spec, para o navegador. |
| `list` | função | `(task=None) -> builtins.list` | Nomes dos modelos publicados, com filtro opcional por tarefa ou tag. |
| `info` | função | `(name, *, offline=None, version=None) -> ModelCard` | Cartão do modelo: tarefa, entradas/saídas com unidades, faixa de validade, métricas, dados de treino, licença e citação. |
| `load` | função | `(name, *, offline=None, version=None) -> ZooModel` | Preditor pronto, com pesos conferidos pelo SHA-256 do manifesto; `version` carrega uma versão substituída. |
| `versions` | função | `(name) -> builtins.list[dict[str, str]]` | Todas as versões de um modelo, da mais nova para a mais antiga, com o motivo de cada substituição. |
| `predict` | função | `(name, inputs=None, **named) -> dict[str, np.ndarray]` | Atalho para `load(name).predict(...)`. |
| `catalogue` | função | `() -> str` | Página Markdown com todos os modelos e datasets: métricas, decisão de confiança, tamanho e link do Hub. |
| `status` | função | `(name) -> dict[str, str]` | De onde cada arquivo do modelo seria lido agora (bundled, cached ou remote), sem baixar. |
| `evaluate` | função | `(name, dataset=None, *, split=None, offline=None, rtol=None) -> dict[str, Any]` | Métricas do modelo contra os casos de referência guardados. |
| `validate` | função | `(name, inputs=None, *, offline=None)` | `TrustReport` do modelo: métricas do cartão reproduzidas, checagens físicas e faixa de validade. |
| `finetune` | função | `(name, new_data, *, epochs=200, lr=0.0001, batch=256, weight_decay=0.0, tolerance=0.02, s…)` | Ajusta o modelo com dados novos e só aceita se não piorar no teste original além da tolerância (com replay opcional). |
| `benchmark` | função | `(name, *, budget_s=60.0, seeds=1, baselines=('retrain',), out_dir='zoo_benchmark', force=…)` | Coloca o modelo do zoo em `pp.benchmark` contra a mesma arquitetura treinada do zero no orçamento dado. |

#### `pinneapple_hub.zoo._artifacts`

Armazém de artefatos do zoo: manifesto, checagem SHA-256, cache local e download preguiçoso do Hugging Face Hub.

<small>`pinneapple_hub/zoo/_artifacts.py` · 237 linhas · testes: 81</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ZooError` | classe | `()` | Classe base dos erros do zoo. |
| `ArtifactNotPublishedError` | classe | `()` | O artefato está no manifesto mas ainda não está no Hugging Face Hub. |
| `ArtifactUnavailableError` | classe | `()` | O artefato não está empacotado nem em cache e não pode ser baixado (offline ou erro de rede). |
| `ChecksumMismatchError` | classe | `()` | O SHA-256 de um arquivo difere do manifesto: corrompido, truncado ou adulterado. |
| `sha256_file` | função | `(path, chunk=1 << 20) -> str` | SHA-256 de um arquivo, lido em blocos. |
| `manifest` | função | `() -> dict[str, Any]` | O `manifest.json` lido (relido quando o arquivo muda). |
| `cache_dir` | função | `() -> Path` | Pasta de cache local dos artefatos. |
| `is_offline` | função | `(offline=None) -> bool` | Diz se o zoo opera offline: pelo argumento, senão pelas variáveis `PINNEAPPLE_OFFLINE` ou `HF_HUB_OFFLINE`. |
| `entry` | função | `(kind, name, version=None) -> dict[str, Any]` | Entrada do manifesto de um modelo ou dataset, opcionalmente de uma versão específica. |
| `versions` | função | `(kind, name) -> list[dict[str, str]]` | Todas as versões de um item, da mais nova à mais antiga, com revisão e motivo de substituição. |
| `revisions` | função | `(ent) -> list[str]` | Revisões do Hub que guardam os arquivos de uma entrada, na ordem em que são tentadas. |
| `fetch` | função | `(kind, name, filename, *, offline=None, version=None) -> Path` | Caminho local de um arquivo de modelo ou dataset: empacotado, em cache ou baixado e conferido. |
| `status` | função | `(kind, name) -> dict[str, str]` | De onde cada arquivo de um item seria lido agora (bundled, cached ou remote), sem baixar. |

#### `pinneapple_hub.zoo._bench`

`pp.zoo.benchmark`: compara o modelo publicado (sem treino) com a mesma arquitetura treinada do zero sob o `ComputeBudget` de `pinneapple_analysis.benchmark`. [sem teste]

<small>`pinneapple_hub/zoo/_bench.py` · 224 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `run_benchmark` | função | `(model, *, budget_s, seeds, baselines, out_dir, force, log, **kwargs)` | Monta o problema a partir do dataset do modelo e roda `pp.benchmark` com os métodos `zoo_pretrained` e `zoo_retrain`. |

#### `pinneapple_hub.zoo._catalog`

Descrição estática de todo modelo e dataset publicado (fonte de `manifest.json` e dos `model_card.json`); só entra o que passou no portão do zoo. [sem teste]

<small>`pinneapple_hub/zoo/_catalog.py` · 715 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `hf_repo` | função | `(kind, name) -> str` | Repositório do Hub de um modelo/dataset: `PINNeAPPle-Labs/<nome com hífens>`. |

#### `pinneapple_hub.zoo._models`

Adaptadores dos modelos do zoo: preditor com faixa de validade, métricas, checagens físicas e ganchos de fine-tune; a base `ZooModel` levanta `NotImplementedError` nos ganchos abstratos.

<small>`pinneapple_hub/zoo/_models.py` · 657 linhas · testes: 7 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OutOfRangeWarning` | classe | `()` | Aviso: entradas fora da faixa de validade (a previsão é extrapolação). |
| `ZooModel` | classe | `(name, card, config, state, spec)` | Classe base de um modelo carregado: `predict`, `in_range`, `reference`, avaliação e `save`; subclasses implementam construção e métricas. |
| `AirfoilMLP` | classe | `()` | Adaptador do ensemble MLP de aerofólio CST: prevê Cl, Cd, Cm e L/D com o desvio do ensemble. |
| `AirfoilGNN` | classe | `()` | Adaptador MeshGraphNet do campo próximo do aerofólio (p, Ux, Uy, viscosidade turbulenta) e coeficientes; não há modelo com esse adaptador no manifesto atual. |
| `HeatsinkForced` | classe | `()` | Adaptador do dissipador em convecção forçada: prevê resistência térmica e queda de pressão. |
| `HeatsinkNatural` | classe | `()` | Adaptador do dissipador em convecção natural, sem velocidade do ar nem queda de pressão, treinado com restrições de monotonicidade. |
| `RoverContact` | classe | `()` | Adaptador do contato roda-solo de rover: prevê Fx, Fz e My. |
| `SupersededModelWarning` | classe | `()` | Aviso: foi carregada uma versão substituída de um modelo do zoo. |
| `load_dir` | função | `(directory) -> ZooModel` | Carrega um modelo salvo com `ZooModel.save` (por exemplo uma cópia ajustada). |
| `load_model` | função | `(name, *, offline=None, version=None) -> ZooModel` | Carrega um modelo do zoo pelo nome, verificando os arquivos. |
| `copy_model` | função | `(m) -> ZooModel` | Cópia independente de um modelo carregado. |
| `fit` | função | `(m, X, Y, *, epochs=300, lr=0.0001, batch=256, weight_decay=0.0, seed=0, progress=None) -…)` | Adam sobre os alvos normalizados do próprio modelo (alvos NaN ignorados); devolve o histórico de perda. |

#### `pinneapple_hub.zoo.catalogue`

Catálogo do zoo como uma página Markdown com métricas, tamanho e link do Hub de cada item. [sem teste]

<small>`pinneapple_hub/zoo/catalogue.py` · 73 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `markdown` | função | `() -> str` | Gera a página do catálogo em Markdown. |

#### `pinneapple_hub.zoo.checks`

Checagens de confiança por consulta (faixa de validade, região esparsa, saída não física ou não finita, resíduo) com IDs de regra de `pinneapple check`, e o veredito pass/warn/fail que elas formam.

<small>`pinneapple_hub/zoo/checks.py` · 358 linhas · testes: 34</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `has_query_checks` | função | `(zm) -> bool` | Diz se o modelo tem checagens por consulta; senão só `in_range` vale. |
| `check_spec` | função | `(zm) -> dict[str, Any]` | Descrição declarativa das checagens de um modelo (uma entrada por coluna de saída). |
| `checks_module` | função | `(zm, predictor=None)` | Módulo `torch` que mapeia linhas de consulta para valores das checagens; é o mesmo que vai para o grafo ONNX. |
| `check_values` | função | `(zm, X) -> tuple[np.ndarray, np.ndarray]` | Valores e auxiliares das checagens para linhas `X` em unidades físicas. |
| `compare` | função | `(value, op, limit) -> bool` | Compara `valor op limite`; NaN nunca passa. |
| `evaluate` | função | `(spec, values, aux) -> dict[str, Any]` | Status, detalhes e veredito para uma linha de valores (o navegador faz o mesmo com as saídas do grafo). |
| `verdict` | função | `(checks) -> dict[str, str]` | Pior status das checagens de confiança (restrições de projeto excluídas), com título e explicação. |
| `query_checks` | função | `(model, inputs, *, offline=None) -> list[dict[str, Any]]` | Checagens e veredito para cada linha de entrada de um modelo. |

#### `pinneapple_hub.zoo.datasets`

`pp.datasets`: datasets verificados em uma linha, guardados como npz sem pickle; os pequenos vêm no pacote, os outros são baixados do Hub, conferidos por SHA-256 e cacheados.

<small>`pinneapple_hub/zoo/datasets.py` · 199 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DatasetInfo` | classe | `(name, version, title, description, license, source, verification, splits, columns, citat…)` | Metadados de um dataset: versão, título, descrição, licença, fonte, verificação, splits, colunas, citação e repositório. |
| `DatasetSplit` | classe | `(data, name, split, ds_info)` | Um split: `{coluna: array}` com nome, split e informações; tem ajudantes como `xy`. |
| `audit_card` | função | `(name) -> dict[str, Any] \| None` | Cartão de auditoria física guardado de um dataset, ou `None` se não foi auditado. |
| `audit` | função | `(name, *, offline=None, save=False, **kw)` | Auditoria física de um dataset via `pinneapple_data.audit.audit_zoo_dataset`. |
| `list` | função | `() -> builtins.list[str]` | Nomes dos datasets publicados. |
| `info` | função | `(name) -> DatasetInfo` | `DatasetInfo` de um dataset. |
| `split_files` | função | `(name, split=None) -> builtins.list[str]` | Arquivos que guardam um split (`data.npz`, `<split>.npz` ou fragmentos `<split>-NN.npz`). |
| `load` | função | `(name, split=None, *, offline=None)` | Carrega um split (`DatasetSplit`) ou todos como dict. |
| `status` | função | `(name) -> dict[str, str]` | De onde os arquivos seriam lidos agora (bundled, cached ou remote), sem baixar. |

#### `pinneapple_hub.zoo.demo`

Dados do Space estático de demonstração do zoo: varreduras de cada entrada com banda de incerteza e veredito das checagens, e a escrita dos arquivos do Space.

<small>`pinneapple_hub/zoo/demo.py` · 189 linhas · testes: 21</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `models` | função | `() -> list[str]` | Nomes dos modelos que a demonstração oferece. |
| `trust` | função | `(name) -> dict[str, Any]` | Relatório de confiança do modelo como um todo: decisão e motivos. |
| `controls` | função | `(name) -> list[dict[str, Any]]` | Uma entrada por variável de entrada: nome, unidade, descrição e faixa. |
| `sweep` | função | `(name, values, vary, *, n=61, margin=0.15) -> dict[str, Any]` | Previsões e checagens ao varrer uma entrada um pouco além da faixa, com as outras fixas. |
| `bands` | função | `(outputs) -> dict[str, dict[str, Any]]` | Saídas principais e como desenhar a incerteza de cada uma (desvio do ensemble ou limite conformal). |
| `precompute` | função | `(name, *, n=SWEEP_POINTS) -> dict[str, Any]` | Tudo o que o Space mostra de um modelo sem rodá-lo: controles, saídas, varreduras e checagens. |
| `export_space` | função | `(directory, models_=None) -> dict[str, Any]` | Grava dados e runtime do Space estático num diretório. |

#### `pinneapple_hub.zoo.export`

Exportação ONNX dos modelos do zoo com pré e pós-processamento no grafo e, opcionalmente, as checagens por consulta; também safetensors. Adaptador sem exportação levanta `NotImplementedError`.

<small>`pinneapple_hub/zoo/export.py` · 317 linhas · testes: 55 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `predictor_module` | função | `(zm)` | Módulo `torch` que leva as entradas do cartão às saídas, base do grafo ONNX. |
| `output_names` | função | `(zm) -> dict[str, list[str]]` | Mapa saída ONNX → chaves de `predict` das suas colunas. |
| `as_graph_outputs` | função | `(zm, pred) -> dict[str, np.ndarray]` | Saída de `predict` reorganizada como as saídas do grafo. |
| `output_scale` | função | `(zm) -> dict[str, np.ndarray]` | Escala de tolerância de paridade por coluna (relativa para saídas positivas modeladas em log). |
| `parity_error` | função | `(zm, ref, got) -> float` | Maior \|obtido - referência\| / escala sobre todas as colunas de saída. |
| `export_bytes` | função | `(zm, *, checks=False) -> bytes` | Bytes do grafo ONNX de um modelo carregado. |
| `export_onnx` | função | `(name, path, *, checks=False, check_parity=True, offline=None) -> dict[str, Any]` | Exporta um modelo do zoo para ONNX num caminho, com checagem de paridade opcional. |
| `parity_inputs` | função | `(zm, *, n_random=500, seed=7) -> np.ndarray` | Entradas de teste e de referência independente do modelo mais pontos aleatórios dentro da faixa. |
| `run_onnx` | função | `(data, X, query=None) -> dict[str, np.ndarray]` | Roda um grafo no onnxruntime (CPU, uma thread) e devolve as saídas em float64. |
| `onnx_parity` | função | `(name, onnx_model, X=None, *, offline=None) -> float` | Maior diferença escalada entre as previsões do grafo e as do modelo PyTorch. |
| `checks_parity` | função | `(name, onnx_model, X=None, *, offline=None) -> dict[str, float]` | Compara as checagens de um grafo exportado com `checks=True` com as do Python. |
| `export_safetensors` | função | `(name, path, *, offline=None) -> dict[str, Any]` | Grava os pesos do modelo como safetensors. |
| `load_safetensors` | função | `(name, path, *, offline=None) -> ZooModel` | Carrega o modelo do zoo com pesos lidos de um arquivo safetensors. |

#### `pinneapple_hub.zoo.hub_cards`

Cartões README dos repositórios do zoo no Hugging Face: front matter YAML lido pelo Hub e o texto para pessoas.

<small>`pinneapple_hub/zoo/hub_cards.py` · 368 linhas · testes: 21</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `space_url` | função | `() -> str` | URL do Space de demonstração do zoo. |
| `front_matter` | função | `(meta) -> str` | Bloco YAML de front matter a partir de um dict. |
| `parse_front_matter` | função | `(readme) -> dict[str, Any]` | Lê o bloco YAML do topo de um README (`{}` se não há). |
| `code_blocks` | função | `(readme, lang='python') -> list[str]` | Blocos de código cercados de um README numa linguagem. |
| `split_order` | função | `(name) -> list[str]` | Splits de um dataset com train/val/cal/test primeiro e os outros na ordem do manifesto. |
| `dataset_card` | função | `(name, layout) -> str` | README de um repositório de dataset: front matter (configs, features, licença, tags) e o cartão. |
| `model_example` | função | `(zm) -> dict[str, Any]` | Um caso de referência do teste do modelo: entradas, saídas do modelo e referência. |
| `model_card` | função | `(name, *, example, onnx_info=None, files=()) -> str` | README de um repositório de modelo: metadados, uso, faixa de validade, cartão de confiança, relatório e arquivos. |
| `space_readme` | função | `() -> str` | README do Space estático de demonstração. |
| `example_json` | função | `(example) -> str` | Serializa o exemplo do modelo em JSON. |

#### `pinneapple_hub.zoo.parquet`

Cópias Parquet determinísticas dos datasets do zoo para o visualizador do Hugging Face e `datasets.load_dataset`; os npz continuam a fonte de verdade.

<small>`pinneapple_hub/zoo/parquet.py` · 281 linhas · testes: 21</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `read_npz` | função | `(paths) -> dict[str, dict[str, np.ndarray]]` | Lê npz com arrays `<split>__<coluna>` em `{split: {coluna: array}}`, juntando fragmentos. |
| `plan_split` | função | `(arrays, *, explode_row_bytes=EXPLODE_ROW_BYTES, max_shard_bytes=MAX_SHARD_BYTES, row_gro…)` | Decide como um split vira Parquet (explodir amostras grandes, fragmentos, grupos de linhas). |
| `write_dataset` | função | `(name, data, out_dir, *, split_order=(), column_order=(), explode_row_bytes=EXPLODE_ROW_B…)` | Escreve os arquivos Parquet de um dataset e devolve o layout. |
| `read_parquet` | função | `(paths) -> dict[str, np.ndarray]` | Lê de volta os arrays npz de um split a partir dos fragmentos Parquet (inverso de `write_dataset`). |
| `parquet_files` | função | `(layout) -> list[str]` | Todos os arquivos Parquet de um layout. |
| `features_yaml` | função | `(schema) -> list[dict[str, Any]]` | Lista `dataset_info.features` de um schema Parquet, no formato que `datasets` escreve. |

#### `pinneapple_hub.zoo.publish`

Publica o zoo no Hugging Face Hub (um repositório por item em `PINNeAPPle-Labs` mais o Space) e verifica o que o Hub serve; usado por `scripts/zoo/publish_to_hub.py` e pelo workflow de publicação.

<small>`pinneapple_hub/zoo/publish.py` · 611 linhas · testes: 49</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plan` | função | `(staging=None, *, need_files=True, hub_fallback=False, work=None, only=()) -> list[dict]` | Lista de itens a publicar com tipo, nome, repositório, revisão e arquivos. |
| `prepare` | função | `(it, work) -> dict` | Grava os arquivos derivados de um item (Parquet e cartão, ou safetensors, ONNX e exemplo) e o README. |
| `publish` | função | `(items, token, *, work=None, tags=True) -> list[str]` | Envia cada item num commit, marca com a revisão e devolve os que falharam. |
| `build_space` | função | `(out, app_dir, playground_dir, models=None) -> list[str]` | Monta o Space estático de demonstração num diretório. |
| `publish_space` | função | `(token, app_dir, playground_dir, *, work=None, from_hub=True) -> list[str]` | Monta e envia o Space num commit. |
| `verify_files` | função | `(items, allow_unpublished=False) -> int` | Baixa anonimamente cada arquivo do manifesto e confere o SHA-256. |
| `viewer_status` | função | `(repo, *, timeout=2400, interval=30, get=_get_json, sleep=time.sleep) -> dict[str, Any]` | Estado do visualizador de datasets de um repositório, esperando até o tempo limite. |
| `verify_viewer` | função | `(names=None, *, timeout=2400, interval=30, get=_get_json, sleep=time.sleep) -> int` | Confere o visualizador de todos os datasets. |
| `verify_model` | função | `(name, work) -> list[str]` | Problemas encontrados ao carregar do Hub e checar um modelo publicado. |
| `verify_models` | função | `(names=None) -> int` | `verify_model` para todos os modelos, anônimo, sem cópias empacotadas e com cache vazio. |
| `verify_space` | função | `(*, timeout=2400, interval=30, runtime=None, fetch=_fetch, sleep=time.sleep) -> int` | Espera o Space rodar e servir a página; 0 quando está OK. |

#### `pinneapple_hub.zoo.relations`

Relações metamórficas que a física de cada modelo do zoo implica (simetria, monotonicidade), checadas sem solução de referência via `pinneapple_analysis.validation.metamorphic`.

<small>`pinneapple_hub/zoo/relations.py` · 95 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check_zoo_relations` | função | `(model, name=None, *, n=64, seed=0, search='random', relations=None) -> MetamorphicReport` | Roda as relações de um modelo do zoo e devolve um `MetamorphicReport`. |

#### `pinneapple_hub.zoo.validation_reports`

Relatórios `pp.validate` guardados de cada modelo do zoo, mostrados no cartão e comparados com um novo antes de uma versão; uma checagem que deixa de passar bloqueia a publicação.

<small>`pinneapple_hub/zoo/validation_reports.py` · 59 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build` | função | `(name)` | Valida o modelo e grava o relatório. |
| `stored` | função | `(name) -> dict[str, Any] \| None` | Lê o relatório guardado de um modelo, ou `None`. |
| `check_release` | função | `(name, model=None)` | Diferença entre o relatório atual (ou de um modelo novo) e o guardado, mais o relatório novo. |
| `card_section` | função | `(name) -> str` | Bloco Markdown do relatório guardado para o cartão do Hub (vazio se não há). |

#### `pinneapple_hub.zoo.web`

Modelos do zoo para o navegador: grafos ONNX com checagens, cartões em JSON e o runtime JS (`onnx.js`, `zoo.js`) usado pelo Playground.

<small>`pinneapple_hub/zoo/web/__init__.py` · 160 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `runtime_dir` | função | `() -> Path` | Pasta do runtime JS (`onnx.js`, `zoo.js`). |
| `reference_data` | função | `(zm) -> dict[str, Any]` | Pontos de referência para plotar ao lado das previsões (por exemplo rodadas OpenFOAM e polar XFOIL no aerofólio). |
| `export_web` | função | `(name, directory, *, offline=None) -> dict[str, Any]` | Grava `<nome>.onnx` e `<nome>.json` para o navegador e devolve o JSON. |
| `fixtures` | função | `(*, offline=None) -> dict[str, Any]` | Respostas do Python para entradas fixas (dentro e fora da faixa), usadas nos testes JS. |
| `export_bundle` | função | `(directory, models=MODELS, *, offline=None) -> dict[str, Any]` | Tudo o que o Playground carrega: cada modelo, grafos de geometria CST e `fixtures.json`. |

#### `pinneapple_hub.zoo.web.__main__`

`python -m pinneapple_hub.zoo.web [DIR]`: grava o pacote web do zoo (padrão `data/web/zoo`). [sem teste]

<small>`pinneapple_hub/zoo/web/__main__.py` · 23 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `main` | função | `(argv=None) -> int` | Ponto de entrada que chama `export_bundle` no diretório dado. |

### Exemplos de uso reais do repositório

**Quickstart do zoo: listar, carregar e prever** — `examples/model_zoo/quickstart.py`

```python
import pinneapple as pp

print(pp.zoo.list())
print(pp.datasets.list())
card = pp.zoo.info("airfoil_cst_rans_mlp")
print(card.task)

model = pp.zoo.load("airfoil_cst_rans_mlp")
naca2412 = dict(T0=0.169591, T1=0.144959, T2=0.131433, M0=0.024922, M1=0.071088, M2=0.058960)  # NACA 2412 (CST)
alphas = np.array([0.0, 2.0, 4.0, 6.0])
pred = model.predict(**naca2412, alpha_deg=alphas)
```

**Fine-tune com guarda no teste original** — `examples/model_zoo/quickstart.py`

```python
new = pp.datasets.load("airfoil_rans_cst_re4e6", split="naca_reference")    # stand-in for your own new runs
ft = pp.zoo.finetune("airfoil_cst_rans_mlp", new, epochs=50, lr=1e-5)
print(ft.summary())
if ft.accepted:
    ft.save(tempfile.mkdtemp())
else:
    print("not saved: the fine-tuned model is worse on the original test set (force=True overrides)")
```

**Busca no índice do Hub** — `pinneapple_hub/library/__init__.py`

```python
from pinneapple_hub import library as hub
index = hub.load_index()                              # data/web/hub/index.json (generated, not committed)
hits = hub.search(index, "airfoil", kind="model")     # ranked items
item = hub.get(index, "geometry:f1_car")
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 23 de 31.
- Funções de teste em arquivos que importam algum módulo do pacote: 183.

### Limitações conhecidas

- `push_to_hub`/`from_pretrained` exigem o extra `hub` (`huggingface_hub`) e, para enviar, uma conta e token do Hugging Face; o módulo `hub.py` não tem teste direto.
- Datasets e modelos que não vêm empacotados precisam de rede no primeiro uso; offline, levantam `ArtifactUnavailableError`, e um arquivo com SHA-256 diferente do manifesto levanta `ChecksumMismatchError`.
- Re-auditar um cartão existente só refaz a checagem de esquema: os pesos e os kwargs exatos do preset não viajam no JSON do cartão, então as alegações físicas não podem ser re-executadas a partir dele (docstring de `verified_card`).
- O sandbox de snippets limita tempo e, em POSIX, memória, CPU e tamanho de arquivo, mas não bloqueia rede; o docstring pede rodar o app sem saída de rede quando os snippets vêm de visitantes.
- `ZooModel` levanta `NotImplementedError` nos ganchos abstratos e `export.predictor_module` levanta `NotImplementedError` para adaptador sem exportação ONNX; o adaptador `AirfoilGNN` existe mas nenhum modelo do manifesto atual o usa.
- Sem teste direto: `hub`, `library.snippets`, `library.sources`, `zoo._bench`, `zoo._catalog`, `zoo.catalogue` e as linhas de comando `__main__`.
- Os cartões do Hub declaram `inference: false`: nenhum provedor de inferência do Hub roda esses modelos; o Space de demonstração é estático e avalia ONNX no navegador.

## 5. `pinneapple_lab`

### Objetivo

O `pinneapple_lab` é o laboratório de experimentos do PINNeAPPle. Ele roda muitos experimentos de forma reprodutível e acumula um banco de resultados e de datasets. Um experimento é uma classe com parâmetros padrão e um método `run(ctx)`; o runner dá a cada execução um id derivado do nome, da versão e dos parâmetros, grava entradas, saídas, métricas, checagens de validação, figuras e datasets numa pasta própria e indexa tudo em SQLite (`LabStore`). Rodar de novo com os mesmos parâmetros devolve o resultado em cache.

Por cima desse banco existe uma camada de curadoria. `curation.curate` avalia cada execução em dimensões calculadas só a partir do que foi gravado (checagens, métricas, arquivos, código, datasets, documentação e uma revisão humana opcional) e dá uma nota de A (flagship) a F (não utilizável). As notas são portões, não média ponderada: uma checagem que falhou nunca é escondida por uma figura bonita. `next_steps` diz o que falta para a próxima letra e qual ação do PINNeAPPle fecharia cada lacuna; `seal` carimba a letra em figuras, renders e vídeos; `explorer`, `html`, `server`, `reports` e `showcase` publicam tudo como página, API, relatório ou painel animado.

O pacote se liga ao resto da biblioteca de várias formas. Os experimentos embutidos em `pinneapple_lab.experiments` chamam solvers e modelos de outros pacotes (`pinneapple_simulation`, `pinneapple_physics`, `pinneapple_neural`, `pinneapple_design`) e conferem os resultados contra soluções exatas ou referências publicadas. O módulo `manifest` define o formato `experiment.json`, lido pelo site, pela curadoria e pelo Flow Studio, e converte para manifesto resultados de `pp.solve` (`from_solution`), tabelas de `pp.compare` (`from_comparison`), workflows de `pinneapple_core` e execuções do Flow Studio. Qualquer script de `examples/` vira item do Lab com `pinneapple_lab.experiment(...)` (`capture`).

A linha de comando `python -m pinneapple_lab` reúne tudo: `list`, `run`, `sweep`, `status`, `report`, `serve`, `curate`, `review`, `brief`, `seal`, `stamp`, `improve`, `register`, `scaffold`, `manifests --check` (o portão de CI dos manifestos), `showcase` e `export`.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 42 | 11.775 | 45 | 119 | 96 (9 arquivos) | 27 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `__main__` | Linha de comando `python -m pinneapple_lab`: rodar, varrer, curar, carimbar, servir e exportar. | 1 | 1 | 59 |
| `capture` | `experiment(...)`: embrulha um script para que ele deixe um manifesto, um item do Lab e uma nota. | 1 | 3 | 0 |
| `context` | `RunContext`, o objeto que o experimento recebe para gravar tudo o que produz. | 1 | 1 | 0 |
| `curation` | Curadoria: avalia execuções em dimensões de evidência e dá notas A-F com prontidão por uso. | 1 | 17 | 65 |
| `dataset` | Datasets gerados pelas execuções, gravados em shards `.npz` com um cartão descritivo. | 1 | 2 | 0 |
| `experiments` | Experimentos embutidos (ODE, PDE, LBM, FEM, CFD, descoberta de leis, ROM, importação de resultados do repositório). | 22 | 39 | 20 |
| `explorer` | Página de experimentos com selo A-F, cartão de confiança, métricas, mídia e filtros. | 1 | 11 | 6 |
| `html` | Catálogo navegável do Lab em uma única página HTML autocontida. | 1 | 6 | 14 |
| `improve` | Comando `improve`: mostra as ações para a próxima letra e, se pedido, roda as executáveis e recura. | 1 | 2 | 0 |
| `manifest` | Formato `experiment.json` (manifesto de experimento), conversões, validação e portão de CI. | 1 | 19 | 29 |
| `next_steps` | Lacunas entre a letra atual e a próxima, cada uma com as ações do PINNeAPPle que a fecham. | 1 | 9 | 31 |
| `reports` | Relatórios em HTML, Markdown ou PDF de um item ou de um conjunto filtrado. | 1 | 6 | 20 |
| `results_rules` | Regras que extraem métricas e checagens dos arquivos de resultados que um exemplo versiona. | 1 | 6 | 14 |
| `runner` | Execução de um experimento ou de varreduras (grade ou amostras), com cache e paralelismo. | 1 | 3 | 0 |
| `seal` | Selo de qualidade PINNeAPPle: badge com a letra da curadoria, carimbado em figuras e vídeos. | 1 | 21 | 39 |
| `server` | Servidor HTTP (só biblioteca padrão) do catálogo, arquivos das execuções, API JSON e downloads. | 1 | 2 | 26 |
| `showcase` | Painel animado (MP4, GIF, PNG) com os melhores resultados, montado a partir da curadoria. | 1 | 5 | 3 |
| `spec` | Classe base `Experiment` e registro de experimentos por nome. | 1 | 4 | 0 |
| `store` | `LabStore`: pastas de execução mais índice SQLite, consultas, exportação de datasets e catálogo. | 1 | 2 | 0 |
| `uq` | Checagens de incerteza para o cartão de confiança: cobertura de intervalos, GCI e espalhamento de ensemble. | 1 | 5 | 20 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_lab`

Ponto de entrada do pacote: reexporta `Experiment`, `register`, `run`, `sweep`, `LabStore`, `experiment` e afins.

<small>`pinneapple_lab/__init__.py` · 43 linhas · testes: 71</small>

#### `pinneapple_lab.__main__`

Linha de comando `python -m pinneapple_lab` com subcomandos para rodar, varrer, curar, revisar, carimbar, servir, gerar relatórios e exportar datasets.

<small>`pinneapple_lab/__main__.py` · 366 linhas · testes: 59</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `main` | função | `(argv=None) -> int` | Monta o `argparse` com todos os subcomandos e despacha para as funções do pacote; devolve o código de saída. |

#### `pinneapple_lab.capture`

Embrulha um script ou exemplo (`with experiment(...) as ex:` ou decorador) para gravar parâmetros, métricas, checagens e arquivos num manifesto e registrá-lo no Lab. [sem teste direto]

<small>`pinneapple_lab/capture.py` · 168 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Capture` | classe | `(name, *, title='', description='', domain='', methods=(), modules=(), params=None, out_d…)` | Contexto/decorador que acumula parâmetros, métricas (`log_metric`), checagens (`check`) e arquivos (`add_file`) e, ao sair, escreve o manifesto e registra o item no Lab. |
| `experiment` | função | `(name, **kw) -> Capture` | Cria um `Capture`, usável como `with experiment(...) as ex:` ou como `@experiment(...)`. |
| `register_manifest` | função | `(path, lab=None, *, curate=True) -> str` | Importa um manifesto válido no Lab como item `manifest/<id>`, recura e devolve a chave do item; recusa manifesto inválido. |

#### `pinneapple_lab.context`

Define o `RunContext`, que o experimento recebe em `run(ctx)` e que grava tudo na pasta da execução e no índice. [sem teste direto]

<small>`pinneapple_lab/context.py` · 217 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RunContext` | classe | `(run_dir, params, experiment, run_id, seed=None)` | Objeto entregue a `Experiment.run`: dá os parâmetros e grava log, estágios, entradas, saídas, figuras, imagens, GIFs, métricas, checagens e datasets na pasta da execução. |

#### `pinneapple_lab.curation`

Curadoria das execuções: avalia dimensões de evidência (validação, referência, baseline, física, incerteza, reprodutibilidade etc.) e atribui notas A-F com prontidão para produto, artigo, marketing ou dados de treino. Usa o vocabulário de `pinneapple_veriphysics.applicability`.

<small>`pinneapple_lab/curation.py` · 970 linhas · testes: 65</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Dimension` | classe | `(name, score, status, tier, evidence)` | Uma dimensão avaliada: nota 0-1 (ou None = não rodou), status, nível de evidência e lista de evidências. |
| `Assessment` | classe | `(run_id, experiment, status, tier, score, coverage, dimensions, readiness, gaps, reviewed…)` | Avaliação completa de uma execução: letra, nota, cobertura, dimensões, prontidão por uso e lacunas. |
| `infer_check_kind` | função | `(check) -> str` | Deduz de que uma checagem é evidência (referência, benchmark, física etc.) pelo nome, pelo valor de referência ou pelo texto de detalhe. |
| `load_reviews` | função | `(root) -> dict[str, Any]` | Lê as revisões humanas gravadas em `<lab>/reviews.json`. |
| `save_review` | função | `(root, experiment, **fields) -> dict[str, Any]` | Grava (mesclando) a revisão humana de um experimento em `<lab>/reviews.json`. |
| `assess_run` | função | `(store, row, *, meta=None, reviews=None, siblings=None) -> Assessment` | Avalia uma execução (uma linha de `LabStore.runs()`) e devolve seu `Assessment`. |
| `path_to_next` | função | `(tier, issue, checks, dims, evidence, ok, is_ml, ctx, *, reviewed=False) -> list[Any]` | Lista as lacunas entre a letra atual e a próxima, cada uma com ações do PINNeAPPle, a partir das mesmas evidências da nota. |
| `next_steps` | função | `(item, lab=None) -> list[Any]` | Lê do registro de curadoria o caminho para a próxima letra de um item, como lista de `Gap`. |
| `problem` | função | `(status, checks, error=None) -> dict[str, Any] \| None` | Diz qual regra E ou F a execução cai (travou, divergiu, física falhou, contradiz a referência...) ou None se tudo passou. |
| `problem_headline` | função | `(issue) -> dict[str, Any]` | Resumo de um item E ou F: a checagem principal que falhou, a regra e a dica do próximo passo. |
| `check_line` | função | `(check, max_chars=64) -> str` | Uma linha que diz contra o que a checagem comparou (rótulo dado pelo experimento ou nome da checagem mais detalhe). |
| `headline` | função | `(checks, tier, labels=None) -> dict[str, Any]` | Evidência que um selo ou bloco do showcase pode imprimir, tirada só das checagens que passaram. |
| `trust_stages` | função | `(d, checks, metrics, dims, tier, readiness, meta, is_ml, limitations) -> dict[str, dict[s…)` | Responde às seis perguntas do cartão de confiança (dados, modelo, física, benchmark, incerteza...) com base nas evidências da execução. |
| `evidence_stages` | função | `(d, checks, metrics, dims, is_ml, limitations) -> dict[str, dict[str, Any]]` | As cinco primeiras perguntas do cartão de confiança, que não dependem da letra e alimentam a regra C/D. |
| `plan_experiment` | função | `(runs, meta, review) -> dict[str, Any]` | Plano de um experimento: limitações, ações que fecham lacunas (com esforço) e a execução mais próxima de estar pronta para cada uso. |
| `assess_experiment` | função | `(runs) -> dict[str, Any]` | Avalia um experimento inteiro: melhor execução, contagem por letra e consistência entre execuções. |
| `curate` | função | `(store, path_json=None, path_md=None) -> dict[str, Any]` | Avalia todas as execuções e experimentos e escreve `curation.json` e `CURATION.md` na pasta do Lab. |

#### `pinneapple_lab.dataset`

Datasets das execuções: amostras gravadas em shards `.npz` com índice `samples.jsonl` e `card.json` (esquema, unidades, proveniência, checksum). [sem teste direto]

<small>`pinneapple_lab/dataset.py` · 145 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DatasetWriter` | classe | `(folder, *, name, description='', units=None, shard_size=256, license='', provenance=None)` | Escreve amostras de um dataset em shards, com bufferização, e gera o cartão descritivo ao fechar. |
| `read_dataset` | função | `(folder) -> Iterator[dict[str, Any]]` | Itera sobre as amostras de uma pasta de dataset escrita por `DatasetWriter`. |

#### `pinneapple_lab.experiments`

Pacote dos experimentos embutidos; cada módulo registra seus experimentos ao ser importado. [sem teste direto]

<small>`pinneapple_lab/experiments/__init__.py` · 42 linhas · **sem teste direto**</small>

#### `pinneapple_lab.experiments.accretion`

Experimentos de hidrodinâmica de buraco negro sobre `pinneapple_physics.blackhole`: acreção de Bondi e um toro acretando. [sem teste direto]

<small>`pinneapple_lab/experiments/accretion.py` · 126 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BondiAccretion` | classe | `()` | Experimento `bondi_accretion`: o solver hidro, iniciado na solução exata de Bondi (potencial de Paczynski-Wiita), deve mantê-la estacionária. |
| `AccretionFlowRun` | classe | `()` | Experimento `accretion_flow`: toro quente acretando num buraco negro de Schwarzschild, com filme de densidade, taxa de acreção e balanços; os quadros viram dataset. |

#### `pinneapple_lab.experiments.blackhole_forecast`

Experimento de previsão do "tempo" de buraco negro (Duarte, Nemmen & Navarro 2022) que embrulha `examples/black_hole_weather`; precisa do checkout do repositório. [sem teste direto]

<small>`pinneapple_lab/experiments/blackhole_forecast.py` · 86 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BlackHoleForecast` | classe | `()` | Experimento `bh_forecast`: U-Net que prevê a densidade do fluxo de acreção, avaliada por horizonte contra persistência. |

#### `pinneapple_lab.experiments.bricks`

Construções de blocos encaixáveis (`pinneapple_design.bricks`) que se modificam até atingir uma meta (balanço, ponte, torre), verificadas por equilíbrio estático. [sem teste direto]

<small>`pinneapple_lab/experiments/bricks.py` · 97 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BricksExperiment` | classe | `()` | Experimento `bricks`: verifica a física contra formas fechadas e depois muta a construção até cumprir a meta. |
| `make_goal` | função | `(case)` | Devolve a meta (`CantileverGoal`, `BridgeGoal` ou `TowerGoal`) de um caso pelo nome. |

#### `pinneapple_lab.experiments.calculix_cases`

Casos de CalculiX via `pinneapple_simulation.external_solvers.calculix.study` (placa com furo, suporte em L, modal de viga...) contra referências em forma fechada. Depende de CalculiX/gmsh. [sem teste direto]

<small>`pinneapple_lab/experiments/calculix_cases.py` · 283 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CalculixCase` | classe | `()` | Experimento `calculix_case`: roda um caso de CalculiX e compara tensão, convergência de malha e equilíbrio com fórmulas fechadas. |

#### `pinneapple_lab.experiments.car`

Aerodinâmica de um perfil 2-D de carro paramétrico em túnel de vento virtual (LBM D2Q9 com LES) e um surrogate de IA física treinado nessas execuções.

<small>`pinneapple_lab/experiments/car.py` · 438 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CarLBM` | classe | `()` | Experimento `car_lbm`: um desenho de carro no túnel virtual, com arrasto e sustentação, campos médios e filme de vorticidade; gera dataset. |
| `CarSurrogate` | classe | `()` | Experimento `car_surrogate`: treina um FNO (forma -> escoamento médio) e um ensemble de MLPs (parâmetros -> Cd, Cl) sobre as execuções `car_lbm`. |
| `simulate_car` | função | `(shape, *, length_cells=64, Re=500.0, u_in=0.08, Cs=0.1, steps=8000, save_every=50, keep_…)` | Roda o túnel de vento virtual para um perfil (ou máscara sólida) e devolve campos, históricos e coeficientes. |

#### `pinneapple_lab.experiments.cylinder`

Escoamento em torno de cilindro por lattice-Boltzmann: regime, número de Strouhal e imagens de vorticidade rotuladas. [sem teste direto]

<small>`pinneapple_lab/experiments/cylinder.py` · 61 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CylinderLBM` | classe | `()` | Experimento `cylinder_lbm`: canal 2-D com cilindro no solver D2Q9, medindo regime e Strouhal pelo espectro refinado. |

#### `pinneapple_lab.experiments.discovery`

Descoberta de leis físicas a partir de dados: Kepler com dados orbitais reais, pêndulo filmado, Lorenz no estilo AI-Lorenz e oscilador, com SINDy e ELM. [sem teste direto]

<small>`pinneapple_lab/experiments/discovery.py` · 548 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `KeplerLaw` | classe | `()` | Experimento `kepler_law`: regressão esparsa sobre leis de potência redescobre a terceira lei de Kepler a partir de dados orbitais. |
| `PendulumVideo` | classe | `()` | Experimento `pendulum_video`: lei -> vídeo -> lei -> vídeo; mede o ângulo nos quadros, descobre a equação por SINDy fraco e re-simula. |
| `LorenzDiscovery` | classe | `()` | Experimento `lorenz_discovery`: ajusta observações ruidosas de Lorenz por janelas com ELM e descobre as equações. |
| `OscillatorDiscovery` | classe | `()` | Experimento `oscillator_discovery`: usa trajetórias do oscilador já no banco e recupera ζ e ω por SINDy. |
| `elm_fit` | função | `(t, y, n_features=300, ridge=1e-08, seed=0, scale=None)` | Ajuste por mínimos quadrados com features tanh aleatórias fixas (extreme learning machine), com derivadas primeira e segunda. |
| `windowed_elm` | função | `(t, y, window=200, overlap=40, **kw)` | Ajustes ELM em janelas sobrepostas para registros longos e caóticos; devolve y e y' costurados. |
| `weak_form_2nd_order` | função | `(t, u, du_hat, library, width, n_test, p=4)` | Monta a forma fraca de u'' = f(u, u') com funções-teste, evitando derivar os dados (SINDy fraco). |
| `sindy` | função | `(Theta, target, names, threshold=None, max_terms=3, min_contribution=0.03)` | Seleção esparsa de lei: testa todos os subconjuntos de até `max_terms` termos e fica com o de menor BIC. |

#### `pinneapple_lab.experiments.em_pinn`

PINNs nos presets de eletromagnetismo com solução fechada (`em_*`), resolvidos com `pp.solve(problem, "pinn")` e medidos com `pp.compare`. [sem teste direto]

<small>`pinneapple_lab/experiments/em_pinn.py` · 109 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EMPinn` | classe | `()` | Experimento `em_pinn`: treina a PINN de um caso `em_*` e mede o erro relativo contra o campo exato. |

#### `pinneapple_lab.experiments.examples`

Transforma cada script de `examples/` em experimento do Lab: roda como um usuário e coleta figuras, JSON, arrays, saída de console e código.

<small>`pinneapple_lab/experiments/examples.py` · 290 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ExampleScript` | classe | `()` | Experimento `example_script`: roda um script de exemplo e guarda o que ele produziu como figuras, métricas, dataset `artifacts` e código. |
| `discover_examples` | função | `(repo=REPO) -> list[dict[str, str]]` | Lista os scripts executáveis de `examples/**/*.py`, sem helpers e pastas de rascunho, agrupados por pasta. |

#### `pinneapple_lab.experiments.landing`

Importa como execuções do Lab os casos da landing page que vivem em repositórios companheiros públicos (PINNeAPPle-Benchmark, PINNeAPPle-Climate). [sem teste direto]

<small>`pinneapple_lab/experiments/landing.py` · 292 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BenchmarkCase` | classe | `()` | Experimento `benchmark_case`: importa um experimento finalizado de repositório companheiro e reconfere a afirmação principal com o `results.json` dele. |
| `source_root` | função | `(key, lab_root=None) -> str \| None` | Acha o checkout local de um repositório companheiro (variável de ambiente, pasta irmã ou clone raso). |

#### `pinneapple_lab.experiments.manifest_import`

Experimento `manifest`: traz um `experiment.json` para o Lab como execução, para ser curado A-F como qualquer item. [sem teste direto]

<small>`pinneapple_lab/experiments/manifest_import.py` · 45 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ManifestImport` | classe | `()` | Experimento `manifest`: grava parâmetros, métricas, checagens e cópias de figuras de um manifesto; falha se o manifesto registra erro. |

#### `pinneapple_lab.experiments.ode`

Oscilador harmônico amortecido: integradores (Euler, Euler simplético, RK4) contra a solução exata. [sem teste direto]

<small>`pinneapple_lab/experiments/ode.py` · 95 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Oscillator` | classe | `()` | Experimento `oscillator`: integra x'' + 2ζωx' + ω²x = 0 e compara com a solução exata; gera dataset de trajetórias. |
| `exact` | função | `(t, zeta, omega, x0=1.0, v0=0.0)` | Solução exata do oscilador amortecido nos casos sub, crítico e superamortecido. |
| `integrate` | função | `(method, zeta, omega, dt, t_end, x0=1.0, v0=0.0)` | Integra o oscilador com o método escolhido (`euler`, `symplectic`, `rk4`) e passo fixo. |

#### `pinneapple_lab.experiments.particles`

Suspensão de sólidos em tanque agitado: partículas lagrangianas (DEM, arrasto, empuxo, dispersão) num escoamento analítico, renderizadas como vídeo de processo. [sem teste direto]

<small>`pinneapple_lab/experiments/particles.py` · 156 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ParticleSuspension` | classe | `()` | Experimento `particle_suspension`: simula a suspensão e confere velocidade de sedimentação e rotação de suspensão contra Schiller-Naumann e Zwietering. |

#### `pinneapple_lab.experiments.pde`

Experimentos de EDP resolvidos com os solvers da biblioteca e conferidos contra soluções exatas. [sem teste direto]

<small>`pinneapple_lab/experiments/pde.py` · 49 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HeatXTFC` | classe | `()` | Experimento `heat_xtfc`: equação do calor 1-D resolvida por X-TFC contra exp(-α π² t) sin(π x). |

#### `pinneapple_lab.experiments.pipes`

Escoamento interno em tubos e misturadores estáticos com OpenFOAM (`internal_flow`): Hagen-Poiseuille, curva de 90 graus e misturador Kenics. Depende de OpenFOAM. [sem teste direto]

<small>`pinneapple_lab/experiments/pipes.py` · 307 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PipeFlow` | classe | `()` | Experimento `pipe_flow`: roda um caso do solver de tubo e compara fator de atrito e perda com 64/Re, Colebrook e Ito. |

#### `pinneapple_lab.experiments.repo_results`

Importa como execuções validadas os resultados já produzidos por scripts do repositório, reconferindo-os contra sua referência. [sem teste direto]

<small>`pinneapple_lab/experiments/repo_results.py` · 317 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RepoResults` | classe | `()` | Experimento `repo_results`: lê os artefatos de uma fonte (`source`), reconfere e grava métricas, figuras e datasets. |

#### `pinneapple_lab.experiments.repo_sources`

Mais fontes para `repo_results` (máquinas elétricas, aerofólio, CFD de avião, rover lunar, otimização topológica, modelo de clima etc.), cada uma transformando números versionados em checagens. [sem teste direto]

<small>`pinneapple_lab/experiments/repo_sources.py` · 636 linhas · **sem teste direto**</small>

#### `pinneapple_lab.experiments.rom`

Modelos de ordem reduzida (`pinneapple_neural.architectures.rom`) sobre os solvers do Lab: POD, DMD e Operator Inference na esteira de cilindro e ROM paramétrico de viga FEM. [sem teste direto]

<small>`pinneapple_lab/experiments/rom.py` · 399 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ROMStudy` | classe | `()` | Experimento `rom_study`: roda um caso de ROM e confere energia dos modos POD e previsão contra persistência. |
| `cylinder_wake` | função | `(Re, steps, save_every, D=20)` | Gera snapshots de velocidade do escoamento em torno de cilindro (LBM D2Q9) em unidades de rede. |
| `cantilever` | função | `(H, W, angle_deg, *, L=2.0, mesh=(30, 3, 6), F=10000.0)` | Resolve por FEM sólido 3-D uma viga engastada de aço com carga na face livre e devolve o `FEMResult`. |

#### `pinneapple_lab.experiments.structures`

Mecânica estrutural 3-D com o solver `solid_fem` (hexaedros de 8 nós com modos incompatíveis), conferida contra teoria de vigas e torção e contra CalculiX. [sem teste direto]

<small>`pinneapple_lab/experiments/structures.py` · 225 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SolidFEMExperiment` | classe | `()` | Experimento `solid_fem`: roda um caso (balanço, biapoiada, torção...) e compara flecha, tensão e taxa de torção com fórmulas fechadas e CalculiX. |
| `build` | função | `(case, refine=1)` | Monta malha, modelo e referências em forma fechada de um caso. |

#### `pinneapple_lab.experiments.vehicles3d`

Veículos 3-D (carro e foguete) em OpenFOAM com RANS, renders no Blender e visualizador three.js. Depende de OpenFOAM e Blender. [sem teste direto]

<small>`pinneapple_lab/experiments/vehicles3d.py` · 162 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `VehicleCFD` | classe | `()` | Experimento `vehicle_cfd`: gera o veículo, roda RANS em meio modelo e confere coeficientes contra faixas de referência ou estabilidade de Barrowman. |

#### `pinneapple_lab.experiments.wear`

Desgaste por deslizamento de barras de vários materiais (`pinneapple_physics.tribology`), com Archard e pressão redistribuída. [sem teste direto]

<small>`pinneapple_lab/experiments/wear.py` · 238 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BarWear` | classe | `()` | Experimento `bar_wear`: desliza uma barra coroada até assentar e confere volume desgastado, contato inicial e taxa estacionária. |
| `BarWearRanking` | classe | `()` | Experimento `bar_wear_ranking`: compara todas as execuções `bar_wear` e confere o ranking dos materiais pelo coeficiente de Archard. |

#### `pinneapple_lab.explorer`

Página de experimentos: cada item com selo A-F, cartão de confiança, métricas, próximos passos, figuras, vídeos e links, servida em `/experiments` ou exportada como HTML estático.

<small>`pinneapple_lab/explorer.py` · 387 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `proper` | função | `(title) -> str` | Gera um título a partir de uma chave, com siglas em maiúsculas. |
| `collect` | função | `(store, examples=None) -> dict[str, Any]` | Reúne todos os experimentos da página (curadoria e manifestos), contagens por letra, facetas de filtro e lista de cobertura. |
| `coverage` | função | `(data) -> list[dict[str, Any]]` | Tabela experimento -> curado sim/não (e letra), uma linha por entrada. |
| `coverage_md` | função | `(data) -> str` | Gera em Markdown o relatório de cobertura da curadoria. |
| `filter_entries` | função | `(entries, *, letter=None, domain=None, method=None, ready=False, q=None, sort='letter') -…)` | Aplica os filtros da página (letra, domínio, método, pronto, busca, ordenação), também usados por `/api/experiments`. |
| `media_urls` | função | `(data, mode='server') -> dict[str, Any]` | Preenche as URLs de mídia, miniatura e selo, como rotas do servidor ou caminhos relativos do export estático. |
| `render` | função | `(data) -> str` | Renderiza a página com os dados embutidos no template. |
| `media_path` | função | `(lab_root, entry, item) -> str \| None` | Resolve o arquivo por trás de um item de mídia (arquivo de execução ou do repositório). |
| `stamped_jpeg` | função | `(path, key, cur, width=1280) -> bytes \| None` | Devolve a figura carimbada com o selo do item como JPEG de largura limitada. |
| `seal_png` | função | `(key, cur, lang='en') -> bytes` | Renderiza o selo de um item como bytes PNG. |
| `write_static` | função | `(store, out_dir, *, examples=None, media=True) -> str` | Escreve a mesma página como arquivos estáticos (`index.html` e `media/`). |

#### `pinneapple_lab.html`

Catálogo navegável do Lab como uma página HTML autocontida, com dados e miniaturas embutidos.

<small>`pinneapple_lab/html.py` · 197 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `thumbnail_bytes` | função | `(path, width=360) -> bytes \| None` | Miniatura JPEG de uma imagem (primeiro quadro de GIF), ou None se não der para ler. |
| `showcase_score` | função | `(path) -> int` | Pontua o nome de uma figura por palavras-chave para escolher a mais vistosa. |
| `collect` | função | `(store, *, thumbs_per_run=1, max_runs_per_experiment=400, thumbs='embed') -> dict[str, An…)` | Reúne tudo o que o catálogo mostra, com miniaturas embutidas ou apontando para a rota `thumb/` do servidor. |
| `write_html` | função | `(store, path=None, *, standalone=True, **kw) -> str` | Escreve a página do catálogo, como documento completo ou só o corpo. |
| `render_page` | função | `(data, *, standalone=True) -> str` | Renderiza a página do catálogo com os dados de `collect` embutidos. |
| `write_site` | função | `(store, out_dir) -> str` | Escreve um catálogo público autocontido para hospedagem estática (GitHub Pages), sem caminhos locais. |

#### `pinneapple_lab.improve`

Comando `improve`: imprime lacunas e ações para a próxima letra e, com `--run`, executa só as ações registradas como executáveis e recura. [sem teste direto]

<small>`pinneapple_lab/improve.py` · 70 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `runnable_actions` | função | `(gaps) -> list[Any]` | Filtra as ações registradas e executáveis das lacunas, sem repetição. |
| `improve` | função | `(item, store, *, run=False, lang='en', log=print) -> int` | Imprime lacunas e ações de um item; com `run=True` roda os comandos do Lab e recura; devolve 0 ou 1. |

#### `pinneapple_lab.manifest`

Manifestos de experimento (`experiment.json`, formato `pinneapple.experiment/1`): conversões de execuções, soluções, comparações e fluxos, validação, varredura do repositório e portão de CI. (A marca de placeholder é falso positivo: o docstring diz que métrica nunca é placeholder.)

<small>`pinneapple_lab/manifest.py` · 650 linhas · testes: 29 · palavras-chave no código (heurística; o texto acima diz o status real): placeholder/stub citado no docstring</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Manifest` | classe | `(id, title, description, domain, methods, modules, params, scripts, metrics, checks, file…)` | Dataclass do manifesto: id, título, domínio, métodos, módulos, parâmetros, scripts, métricas, checagens, arquivos e proveniência. |
| `save` | função | `(m, path) -> str` | Grava o manifesto em um caminho (uma pasta recebe `experiment.json`). |
| `load` | função | `(path) -> Manifest` | Lê o manifesto de um `experiment.json`, de uma pasta que o contém ou de uma pasta de execução do Lab. |
| `validate` | função | `(m, base=None) -> list[str]` | Lista os problemas de um manifesto (esquema, id, título, métricas finitas com `how`, arquivos existentes). |
| `from_run_dir` | função | `(d) -> Manifest` | Converte uma pasta de execução do Lab em manifesto. |
| `write_for_run` | função | `(d) -> str \| None` | Escreve `experiment.json` numa pasta de execução terminada; nunca levanta exceção. |
| `from_workflow_run` | função | `(runner, run_id) -> Manifest` | Converte uma execução de `pinneapple_core.workflows` em manifesto. |
| `from_solution` | função | `(sol, title=None) -> Manifest` | Converte um resultado de `pp.solve` em manifesto (usado por `Solution.to_experiment`). |
| `from_comparison` | função | `(cmp, title=None) -> Manifest` | Converte uma tabela de `pp.compare` em manifesto, com todas as métricas de cada método. |
| `from_flow_run` | função | `(run, out_dir=None) -> Manifest` | Gancho do Flow Studio: converte uma execução de fluxo terminada em manifesto e, opcionalmente, grava. |
| `domain_of` | função | `(name, tags=()) -> str` | Deduz o domínio (por padrões de texto) a partir do nome e das tags; padrão `general`. |
| `methods_of` | função | `(text) -> list[str]` | Lista rótulos de métodos cujos nomes aparecem no texto, como faceta de filtro. |
| `manifest_from_dir` | função | `(folder, lab_curation=None) -> Manifest` | Monta um manifesto com o que uma pasta de exemplo já contém (README, docstring, imports, figuras, resultados). |
| `lab_items_for` | função | `(rel, cur) -> list[str]` | Lista os itens do Lab que avaliam uma pasta de exemplo, melhor letra primeiro. |
| `from_results` | função | `(folder, rules=None) -> Manifest` | Manifesto da pasta acrescido das métricas e checagens extraídas dos resultados versionados por `results_rules`. |
| `example_dirs` | função | `(root=None) -> list[str]` | Lista as pastas de `examples/*` que precisam ter manifesto. |
| `scan` | função | `(roots=None, *, lab=None) -> list[tuple[str, Manifest]]` | Encontra todos os manifestos sob as raízes dadas e todas as execuções do Lab. |
| `check_repo` | função | `(examples=None, lab=None) -> list[str]` | Portão de CI: uma linha por problema nos manifestos dos exemplos e nas execuções do Lab; vazio se tudo ok. |
| `scaffold` | função | `(name, root=None, *, title='', domain='') -> str` | Cria `examples/<name>/` com `run.py` ligado a `pinneapple_lab.experiment`, README e manifesto inicial; recusa pasta existente. |

#### `pinneapple_lab.next_steps`

Registro de tipos de lacuna e de ações: diz o que falta para a próxima letra e qual ação do PINNeAPPle fecha cada lacuna.

<small>`pinneapple_lab/next_steps.py` · 399 linhas · testes: 31</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Action` | classe | `(id, kind, target, command, text, text_pt, params, runnable, argv)` | Ação sugerida: chamada Python ou comando CLI, com experimento e parâmetros preenchidos e textos em inglês e português. |
| `Gap` | classe | `(type, to, stage, why, why_pt, short, short_pt, check, measured, actions)` | Lacuna entre o item e a próxima letra: tipo, estágio do cartão de confiança, porquê, checagem relacionada e ações. |
| `measured` | função | `(check) -> dict[str, Any] \| None` | Extrai o valor medido de uma checagem e seu limite, ou None se não houver número. |
| `measured_text` | função | `(m, lang='en') -> str` | Formata o valor medido contra o limite como texto em inglês ou português. |
| `context_for` | função | `(experiment, params, run_dir, tags, is_ml, cls_params) -> dict[str, Any]` | Monta o contexto (resolução, semente, se é ML etc.) usado para preencher as ações de um item. |
| `make_action` | função | `(aid, ctx) -> Action \| None` | Preenche uma ação registrada para o item, ou None se a condição dela não vale. |
| `make_gap` | função | `(gtype, to, ctx, *, check=None, why=None, meas=None) -> Gap` | Cria uma lacuna de tipo registrado com as ações preenchidas; `KeyError` para tipo desconhecido. |
| `group` | função | `(gaps) -> list[dict[str, Any]]` | Agrupa lacunas que compartilham as mesmas ações, na ordem em que aparecem. |
| `explain` | função | `(gaps, lang='en', title='') -> str` | Escreve as lacunas como texto, com porquê, valor medido e ações agrupadas. |

#### `pinneapple_lab.reports`

Relatórios do Lab (HTML, Markdown ou PDF) de um item ou de vários filtrados, tudo tirado do registro e da curadoria. PDF precisa de `reportlab`.

<small>`pinneapple_lab/reports.py` · 428 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `select` | função | `(cur, *, items=None, experiment=None, tier=None, use=None, ready=None, tag=None, tags_of=…)` | Seleciona chaves de itens da curadoria pelos filtros; `use` ordena por distância a esse uso. |
| `gather` | função | `(store, cur, key, n_figures=3) -> dict[str, Any]` | Junta tudo o que o relatório diz de um item. |
| `render_html` | função | `(store, groups, title, filters='') -> str` | Renderiza o relatório como HTML. |
| `render_md` | função | `(groups, title, filters='') -> str` | Renderiza o relatório como Markdown com tabela de itens. |
| `render_pdf` | função | `(groups, title, filters='') -> bytes` | Renderiza o relatório como PDF com `reportlab`. |
| `write_report` | função | `(store, out, *, keys=None, title=None, fmt=None, filters='', cur=None, **select_kw) -> str` | Seleciona os itens e escreve o relatório no formato indicado pela extensão ou por `fmt`. |

#### `pinneapple_lab.results_rules`

Regras (`Metric`, `Flag`, `Bound`, `CheckList`) que transformam resultados versionados de um exemplo em métricas e checagens de manifesto, com limites citados como `arquivo:linha`. (A marca de placeholder é falso positivo.)

<small>`pinneapple_lab/results_rules.py` · 219 linhas · testes: 14 · palavras-chave no código (heurística; o texto acima diz o status real): placeholder/stub citado no docstring</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Metric` | classe | `(file, path, name, unit, how)` | Regra que lê uma métrica de um arquivo de resultados por caminho, com nome, unidade e `how`. |
| `Flag` | classe | `(file, path, name, kind, detail, invert)` | Regra de checagem a partir de um veredito booleano que o próprio estudo gravou. |
| `Bound` | classe | `(file, path, name, kind, literal, source, op, absolute, unit)` | Regra que compara um valor medido com o limite de aceitação citado no teste do estudo (`arquivo:linha`). |
| `CheckList` | classe | `(file, path, name, kind, value_key, reference_key, tolerance_key, limit_key, passed_key)` | Regra que lê uma lista de registros de checagem gravados pelo estudo (valor, referência, tolerância ou limite). |
| `extract` | função | `(folder, rules) -> tuple[list[dict], list[dict]]` | Aplica as regras e devolve (métricas, checagens); regra sem arquivo ou caminho não gera nada. |
| `check_citations` | função | `(repo) -> list[str]` | Confere que cada literal de `Bound` aparece na linha citada do arquivo citado. |

#### `pinneapple_lab.runner`

Executa um experimento ou varreduras por grade ou amostragem (hipercubo latino), com cache por id e paralelismo por processos. [sem teste direto; coberto pelos testes do pacote]

<small>`pinneapple_lab/runner.py` · 243 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RunResult` | classe | `(run_id, experiment, status, dir, metrics, seconds, error)` | Resultado de uma execução: id, experimento, status (inclusive `cached:<status>`), pasta, métricas, tempo e erro. |
| `run` | função | `(experiment, params=None, *, root=None, force=False, tags=None, raise_errors=False) -> Ru…)` | Roda um experimento com parâmetros sobre os padrões; não repete execução já concluída, salvo `force`. |
| `sweep` | função | `(experiment, *, grid=None, samples=None, space=None, fixed=None, method='lhs', seed=0, n_…)` | Roda um experimento sobre uma grade e/ou amostras aleatórias do espaço, opcionalmente em paralelo (`n_jobs`). |

#### `pinneapple_lab.seal`

Selo de qualidade PINNeAPPle: badge com a letra dada pela curadoria (nunca escolhida à mão), carimbado em figuras, renders, GIFs e vídeos. Vídeos usam ffmpeg.

<small>`pinneapple_lab/seal.py` · 829 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SealError` | classe | `()` | Erro quando o item não pode receber selo porque sua letra está fora das letras pedidas. |
| `Seal` | classe | `(item, tier, title, reason, reason_kind, curated, experiment, run_id, hint, reason_pt, hi…)` | Dados de um selo: item, letra, título, motivo (manchete da curadoria), tipo do motivo e execução. |
| `load_curation` | função | `(lab=None) -> dict[str, Any]` | Carrega o registro de curadoria de uma pasta, caminho, `LabStore`, dict ou do Lab padrão. |
| `parse_tiers` | função | `(tiers) -> tuple[str, ...]` | Converte `"A,B"`, lista ou None numa tupla de letras válidas. |
| `items` | função | `(lab=None, tiers=SEALABLE) -> list[str]` | Lista as chaves de itens curados com uma das letras pedidas. |
| `seal_for` | função | `(item, lab=None, tiers=None) -> Seal` | Monta o selo de um item curado; `KeyError` se desconhecido e `SealError` se a letra estiver fora do filtro. |
| `run_dir` | função | `(seal, lab=None) -> str` | Devolve a pasta da execução a que o selo se refere. |
| `draw_text` | função | `(draw, xy, text, font, fill, anchor='ls') -> None` | Desenha texto com uma marca inicial (✓, ◐, ✗ ou ↑) na fonte DejaVu Sans. |
| `render_seal` | função | `(seal, *, height=96, fmt='png', reason=True, compact=None, opacity=1.0, lang='en', next_s…)` | Renderiza o badge como imagem PIL RGBA ou SVG, com forma compacta em tamanhos pequenos e idioma en/pt. |
| `contrast_ratio` | função | `(fg, bg=BADGE_BG) -> float` | Razão de contraste WCAG 2 entre duas cores `#rrggbb`. |
| `save_seal` | função | `(seal, path, **kw) -> str` | Grava o badge como SVG ou imagem raster, conforme a extensão. |
| `pick_corner` | função | `(frames, w, h, margin, avoid=()) -> str` | Escolhe o canto onde o badge cobre menos conteúdo, evitando caixas proibidas. |
| `corner_scores` | função | `(frames, w, h, margin, avoid=()) -> dict[str, float]` | Mede o quanto cada canto utilizável está ocupado. |
| `stamp_image` | função | `(image, seal, *, corner='auto', size=0.24, opacity=0.95, margin=0.02, reason=True, avoid=…)` | Carimba uma imagem PIL, array ou arquivo, no canto mais vazio ou numa faixa extra se todos estiverem ocupados. |
| `stamp_figure` | função | `(fig, seal, *, corner='auto', size=0.24, opacity=0.95, margin=0.015, reason=True, extend=…)` | Carimba uma figura matplotlib no lugar, evitando barras de cor. |
| `sealed_savefig` | função | `(seal, **stamp_kw)` | Context manager em que todo `Figure.savefig` carimba a figura antes de salvar. |
| `stamp_video` | função | `(src, seal, out, *, corner='auto', size=0.24, opacity=0.95, margin=0.02, reason=True, ext…)` | Carimba um GIF quadro a quadro ou um MP4/WebM/MOV via overlay do ffmpeg. |
| `stamp_file` | função | `(path, seal, out=None, **kw) -> str` | Carimba um arquivo de imagem, GIF ou vídeo e devolve o caminho de saída; recusa SVG e PDF. |
| `stamp_all` | função | `(lab=None, out_dir='export/sealed', *, tiers=SEALABLE, only=None, lang='en', **kw) -> dic…)` | Modo lote: carimba todas as figuras e vídeos dos itens nas letras pedidas, com selos SVG/PNG e manifesto. |
| `legend_seals` | função | `(lang='en') -> list[Seal]` | Um selo de exemplo por letra, com a linha que explica o significado da letra. |
| `render_legend` | função | `(lang='en', height=72)` | Renderiza a legenda dos seis selos, A a F, em fundo claro e escuro. |

#### `pinneapple_lab.server`

Servidor HTTP só com biblioteca padrão: página do catálogo, arquivos das execuções, API JSON, miniaturas sob demanda e downloads de datasets.

<small>`pinneapple_lab/server.py` · 382 linhas · testes: 26</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LabServer` | classe | `(root=None, *, refresh=30.0, user=None, password=None)` | Catálogo e arquivos de uma pasta do Lab; `handler()` dá a classe que atende as requisições. |
| `serve` | função | `(root=None, *, host='127.0.0.1', port=8093, refresh=30.0) -> None` | Serve o catálogo até ser interrompido. |

#### `pinneapple_lab.showcase`

Painel animado 1080 x 1350 com os melhores resultados do Lab (destaque, grade de itens A e fileira de B), gerado da curadoria como MP4, GIF e PNG.

<small>`pinneapple_lab/showcase.py` · 589 linhas · testes: 3</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Tile` | classe | `(item, tier, figure, title, evidence, evidence_kind, seal, frame, lang)` | Um bloco do painel: item, letra, figura, título, linha de evidência e selo. |
| `evidence_line` | função | `(rec, lang='en', short=False) -> str` | Escolhe a linha de evidência do bloco a partir da manchete da curadoria, no idioma pedido. |
| `choose_tiles` | função | `(lab=None, picks=None, lang='en', n_grid=9, n_b=2) -> dict[str, Any]` | Escolhe destaque, grade (A) e fileira B a partir da curadoria e das escolhas manuais `picks`. |
| `fit_crop` | função | `(img, aspect, *, photo=None) -> tuple[tuple[int, int, int, int], bool]` | Calcula o recorte de uma imagem para caber no bloco e se precisa de tarjas. |
| `panel` | função | `(lab=None, *, lang='en', out_dir='docs/assets/showcase', picks=None, seconds=8.0, fps=15,…)` | Renderiza o painel e devolve os arquivos escritos e os blocos usados. |

#### `pinneapple_lab.spec`

Classe base `Experiment` e registro por nome; `Experiment.run` da base levanta `NotImplementedError` (deve ser sobrescrito). [sem teste direto]

<small>`pinneapple_lab/spec.py` · 92 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Experiment` | classe | `(name, version, description, tags, params, space, code_files, references, limitations, ca…)` | Classe base de experimento: defina `name`, `params` e `run(ctx)`; a versão entra no id da execução. |
| `register` | função | `(cls) -> type[Experiment]` | Decorador de classe que registra o experimento por nome para o runner e a CLI. |
| `get` | função | `(name) -> type[Experiment]` | Busca um experimento pelo nome, carregando os embutidos ou importando `modulo:Classe`. |
| `available` | função | `() -> list[str]` | Lista os nomes de todos os experimentos registrados. |

#### `pinneapple_lab.store`

Banco do Lab: pastas `runs/<experimento>/<run_id>` mais índice `index.sqlite`, consultas, tabelas, exportação de datasets e catálogos. [sem teste direto; coberto pelos testes do pacote]

<small>`pinneapple_lab/store.py` · 285 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LabStore` | classe | `(root=None)` | Banco de experimentos numa raiz: indexa execuções, consulta, monta tabelas, lista e exporta datasets e escreve catálogos. |
| `default_root` | função | `() -> str` | Raiz padrão do Lab: `$PINNEAPPLE_LAB` ou `./lab`. |

#### `pinneapple_lab.uq`

Checagens de incerteza para o cartão de confiança: cobertura de intervalos preditivos, índice de convergência de malha (GCI) e espalhamento de ensemble.

<small>`pinneapple_lab/uq.py` · 95 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `coverage` | função | `(y_true, mean, std, level=0.9) -> float` | Fração dos valores verdadeiros dentro do intervalo gaussiano central de nível dado. |
| `coverage_check` | função | `(ctx, name, y_true, mean, std, level=0.9, tol=0.07) -> float` | Grava a cobertura empírica em dados não vistos e checa se fica perto do nível nominal. |
| `gci` | função | `(values, h, order=None, safety=None) -> dict` | Índice de convergência de malha do valor mais fino, com ordem observada para três malhas (Celik et al. 2008). |
| `gci_check` | função | `(ctx, name, values, h, order=None, max_band=0.05) -> dict` | Grava a faixa GCI e o valor extrapolado de Richardson e checa se a faixa fica abaixo de `max_band`. |
| `ensemble_spread` | função | `(values) -> float` | Desvio padrão relativo de resultados repetidos. |

### Exemplos de uso reais do repositório

**Executar, reaproveitar o cache e varrer parâmetros** — `tests/test_lab.py`

```python
def test_run_records_everything_and_is_cached(tmp_path):
    root = str(tmp_path)
    r = run("_test_square", {"a": 2.0}, root=root)
    assert r.status == "completed" and r.metrics["y_max"] == 2.0
    ...
    again = run("_test_square", {"a": 2.0}, root=root)
    assert again.status == "cached:completed" and again.run_id == r.run_id
```

**Embrulhar um script de exemplo com pinneapple_lab.experiment** — `examples/chain_of_whys/chain_of_whys.py`

```python
    with pinneapple_lab.experiment(
            "examples/chain_of_whys", title="Chain of whys",
            description="Engineering reasoning from physics: naive concept -> computed failure mode -> countermeasure "
                        "-> check again, for a roof, a footbridge, a power board and a pumped main.",
            domain="design and geometry", methods=["closed-form checks", "FD beam", "3-D board FV", "MOC"],
            modules=["pinneapple_reason"], out_dir=HERE, register=False,
            links={"guide": "docs/guides/chain_of_whys.md", "script": "examples/chain_of_whys/chain_of_whys.py"}) as ex:
        for name in a.case or list(cases.CASES):
            ...
            ex.check(f"{name}: every requirement passes on the final version", chain.resolved, kind="physics",
                     detail=" -> ".join(f"{m}:{c}" for _, m, c in chain.path()))
```

**Varredura paralela e linha de comando** — `tests/test_lab.py`

```python
    res = sweep("oscillator", grid={"zeta": [0.1, 0.4], "method": ["rk4", "symplectic"]}, n_jobs=2, root=root)
    assert all(r.ok for r in res)
    assert cli(["--root", root, "status"]) == 0
    assert cli(["--root", root, "run", "oscillator", "-p", "zeta=0.3"]) == 0
    assert cli(["--root", root, "report"]) == 0
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 15 de 42.
- Funções de teste em arquivos que importam algum módulo do pacote: 96.

### Limitações conhecidas

- Vários experimentos embutidos dependem de ferramentas externas: CalculiX e gmsh (`calculix_case`), OpenFOAM (`pipe_flow`, `vehicle_cfd`), Blender (renders de `vehicle_cfd`) e ffmpeg (carimbo de vídeos e `showcase`).
- `bh_forecast`, `example_script` e `repo_results` leem arquivos do repositório e só funcionam num checkout do código-fonte; `benchmark_case` precisa de repositórios companheiros (variável de ambiente, pasta irmã ou clone raso).
- A maioria dos módulos de `pinneapple_lab.experiments` não tem teste direto; `runner`, `store`, `context` e `dataset` são exercitados pelos testes que importam o pacote (`tests/test_lab.py`), não por testes próprios.
- A curadoria só classifica a evidência que a execução gravou: um experimento sem checagens fica no máximo em D, e as letras A exigem revisão humana para virar produto, artigo ou marketing.
- `Experiment.run` da classe base levanta `NotImplementedError`; cada experimento precisa sobrescrevê-lo.
- Relatórios em PDF precisam de `reportlab`; selos e miniaturas usam Pillow e matplotlib.
- A curadoria reutiliza o vocabulário de `pinneapple_veriphysics.applicability`, importado de forma opcional.

## 6. `pinneapple_llm`

### Objetivo

O `pinneapple_llm` usa modelos de linguagem (LLMs) para rascunhar partes de um pipeline de IA física, sempre atrás de uma camada de verificação com base física. A regra de projeto, repetida nos docstrings, é que o LLM só escolhe e parametriza algo que o PINNeAPPle já implementa e testa: um preset de `ProblemSpec` de `pinneapple_physics` (`draft_problem`), um gerador de geometria de `pinneapple_design.geometry` (`draft_geometry`, `draft_mesh_recipe`, `draft_cadquery_template`) ou os campos de `DigitalTwinConfig` de `pinneapple_systems` (`draft_digital_twin`). Ele nunca escreve física, CAD ou gêmeo digital do zero, e cada nome e argumento devolvido é conferido contra o registro real.

O `PhysicsGuardrail` (`guardrail`) é o mecanismo anti-alucinação: recalcula sinais numéricos independentes (resíduo da EDP em pontos novos, sanidade dimensional dos parâmetros, balanço de conservação quando há forma fechada e erro contra dados de referência) e se recusa a chamar de confiável um resultado que não passa. O próprio docstring avisa que isso não garante que o resultado esteja certo. `research` busca artigos e repositórios reais no arXiv e GitHub (o LLM só resume), `org_knowledge` responde perguntas a partir de um índice de busca local de `pinneapple_analysis.retrieval`, e `agent_loop` roda um laço de agente que chama ferramentas reais de `pinneapple_worldmodel`.

Todas as chamadas passam por `_dispatch.call_llm`, que fala com Anthropic, OpenAI (ou servidor compatível) e Ollama local. `local_llm` e `conversation_store` dão um caminho totalmente local (Ollama mais um log SQLite das conversas), e `finetune` faz fine-tuning LoRA de um modelo local com esse log.

Dois subpacotes maiores completam o pacote. `qa` é um pipeline de garantia de qualidade em estágios A-G (especificação física tipada, geração, validação estática, execução em sandbox, validadores científicos, comparação com referência independente e reparo limitado), com relatório reprodutível e níveis Q0-Q4 dados pelo que de fato rodou. `workbench` é o Agent Workbench: uma equipe de agentes de código e engenharia rodando num modelo local, com app web (FastAPI), ferramentas confinadas a uma pasta, aprovações, banco SQLite/PostgreSQL e fine-tuning nas próprias sessões.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 49 | 10.750 | 66 | 99 | 242 (20 arquivos) | 28 |

Pasta de testes dedicada: `tests/pinneapple_llm` (4 funções de teste).

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `_dispatch` | Despacho de uma chamada de LLM para Anthropic, OpenAI/compatível ou Ollama, com log opcional. | 1 | 1 | 0 |
| `agent_loop` | Laço de agente em vários passos em que o LLM escolhe, a cada turno, qual ferramenta real chamar. | 1 | 3 | 15 |
| `cad_draft` | Rascunho de geometria nova pelo LLM: receitas CSG de primitivas ou templates paramétricos CadQuery. | 1 | 7 | 22 |
| `catalog_match` | `resolve_problem`: pré-filtro por palavras-chave mais LLM para achar um preset de EDP, sem nunca levantar exceção. | 1 | 2 | 4 |
| `conversation_store` | Log SQLite de todas as chamadas de LLM, para auditoria e como dataset de fine-tuning. | 1 | 2 | 0 |
| `draft` | Rascunho de `ProblemSpec`: o LLM escolhe e parametriza um preset existente de `pinneapple_physics`. | 1 | 5 | 23 |
| `finetune` | Fine-tuning LoRA de um modelo causal local (transformers + peft) com os dados do log de conversas. | 1 | 3 | 0 |
| `geometry_draft` | Rascunho de geometria: o LLM escolhe um gerador existente de `pinneapple_design.geometry`. | 1 | 2 | 0 |
| `guardrail` | `PhysicsGuardrail`: checagens numéricas recalculadas que decidem se um resultado pode ser chamado de confiável. | 1 | 3 | 60 |
| `local_llm` | Provedor local Ollama: iniciar servidor, baixar modelo e fazer chamadas de chat. | 1 | 3 | 0 |
| `org_knowledge` | Respostas do LLM fundamentadas nos resultados reais de um índice de busca local. | 1 | 2 | 4 |
| `qa` | Pipeline de garantia de qualidade (estágios A-G e níveis Q0-Q4) para artefatos gerados por LLM. | 23 | 80 | 48 |
| `research` | Busca real de literatura no arXiv e GitHub, com resumo opcional pelo LLM. | 1 | 2 | 0 |
| `twin_draft` | Rascunho de configuração de gêmeo digital e montagem de um gêmeo 3-D ao vivo. | 1 | 3 | 0 |
| `workbench` | Agent Workbench: equipe de agentes de código e engenharia em modelo local, com app web e fine-tuning. | 12 | 47 | 37 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_llm`

Ponto de entrada do pacote: reexporta as funções de rascunho, o `PhysicsGuardrail`, a busca de literatura e o caminho local.

<small>`pinneapple_llm/__init__.py` · 61 linhas · testes: 128</small>

#### `pinneapple_llm._dispatch`

Despacho de chamadas de LLM independente de provedor (`anthropic`, `openai`, `openai_compatible`, `ollama`), com log opcional por chamada num `ConversationStore`. [sem teste direto]

<small>`pinneapple_llm/_dispatch.py` · 85 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `call_llm` | função | `(prompt, *, provider='anthropic', model=None, api_key=None, system='', json_mode=False, m…)` | Envia um prompt ao provedor pedido e devolve o texto bruto da resposta. |

#### `pinneapple_llm.agent_loop`

Laço de agente em vários passos: o LLM decide a cada turno qual ferramenta real chamar (por padrão `PhysicsToolRegistry` de `pinneapple_worldmodel`), observa o resultado e segue.

<small>`pinneapple_llm/agent_loop.py` · 331 linhas · testes: 15</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AgentStep` | classe | `(tool_name, tool_args, tool_result, error, llm_reasoning)` | Um turno do laço: ferramenta escolhida, argumentos, resultado real (ou erro) e o raciocínio declarado pelo LLM. |
| `AgentLoopResult` | classe | `(steps, final_answer, stopped_reason)` | Rastro completo de uma execução de `run_agent_loop`. |
| `run_agent_loop` | função | `(goal, tool_registry=None, *, llm_provider='anthropic', model=None, max_steps=8, llm_call…)` | Roda o laço rumo a um objetivo: monta o prompt com histórico e ferramentas disponíveis, chama a ferramenta escolhida ou encerra com resposta final. |

#### `pinneapple_llm.cad_draft`

Geração de geometria nova pelo LLM a partir de blocos já existentes: receitas CSG de primitivas de malha ou templates paramétricos CadQuery (sólido B-rep exportável em STEP).

<small>`pinneapple_llm/cad_draft.py` · 633 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CadRecipeResult` | classe | `(kind, recipe, name, params, reasoning, raw_response)` | Resultado de um rascunho de CAD: tipo (`mesh_recipe` ou `cadquery_template`), receita validada ou nome do template e parâmetros. |
| `draft_mesh_recipe` | função | `(description, *, provider='anthropic', model=None, api_key=None, conversation_store=None)…)` | Pede ao LLM uma receita de geometria (possivelmente combinada por CSG) para uma descrição em linguagem natural; não constrói a malha. |
| `draft_cadquery_template` | função | `(description, *, provider='anthropic', model=None, api_key=None, conversation_store=None)…)` | Pede ao LLM para escolher e parametrizar um template CadQuery paramétrico; requer `cadquery` opcional. |
| `build_recipe` | função | `(result)` | Constrói a geometria descrita: `MeshData` para receita de malha ou objeto CadQuery para template. |
| `export_recipe` | função | `(result, out_path, *, fmt=None) -> str` | Exporta a geometria construída para arquivo (STL via `trimesh`; STEP se `cadquery` estiver instalado). |
| `measure_part` | função | `(result) -> Dict[str, Any]` | Constrói uma receita de malha e mede volume, extensão, estanqueidade e se havia motor booleano disponível. |
| `draft_part` | função | `(description, *, tier='quick', provider='ollama', model=None, api_key=None, max_attempts=…)` | Rascunha uma peça construível, tentando de novo quando o rascunho é recusado; nível `quick` (CSG) ou `engineering` (CadQuery). |

#### `pinneapple_llm.catalog_match`

Frente em dois estágios para `draft_problem`: pré-filtro por sobreposição de palavras-chave e depois o LLM; se o LLM falhar, devolve o melhor casamento por palavras. Nunca levanta exceção.

<small>`pinneapple_llm/catalog_match.py` · 128 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ResolveResult` | classe | `(preset, kwargs, reasoning, used_llm, shortlist)` | Resultado da resolução: preset, kwargs, raciocínio, se o LLM foi usado e a lista curta. |
| `resolve_problem` | função | `(description, *, provider='anthropic', model=None, api_key=None, conversation_store=None,…)` | Resolve uma descrição para um preset de EDP registrado, com fallback determinístico por palavras-chave. |

#### `pinneapple_llm.conversation_store`

Log SQLite (só biblioteca padrão) de cada chamada de LLM do pacote, para auditoria e como dataset de `finetune`. [sem teste direto]

<small>`pinneapple_llm/conversation_store.py` · 120 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ConversationRecord` | classe | `(id, timestamp, module, provider, model, system_prompt, user_prompt, response, metadata)` | Um registro de conversa: horário, módulo, provedor, modelo, prompts, resposta e metadados. |
| `ConversationStore` | classe | `(db_path=_DEFAULT_DB_PATH)` | Invólucro fino de SQLite que grava, consulta e exporta conversas como JSONL. |

#### `pinneapple_llm.draft`

Rascunho de `ProblemSpec` restrito a escolher e parametrizar um preset existente de `pinneapple_physics.list_presets()`; nomes e argumentos inventados são rejeitados.

<small>`pinneapple_llm/draft.py` · 231 linhas · testes: 23</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DraftResult` | classe | `(preset, kwargs, reasoning, raw_response)` | Resultado do rascunho: preset escolhido, kwargs, raciocínio e resposta bruta. |
| `DraftRejected` | classe | `(message, attempts)` | Exceção quando nenhuma tentativa produziu rascunho aceitável; `.attempts` diz por que cada uma foi recusada. |
| `draft_problem` | função | `(description, *, provider='anthropic', model=None, api_key=None, conversation_store=None)…)` | Pede ao LLM para escolher e parametrizar um preset para uma descrição em linguagem natural; não constrói o `ProblemSpec`. |
| `check_kwargs_in_bounds` | função | `(kwargs, bounds) -> Dict[str, float]` | Converte kwargs rascunhados para floats finitos e confere cada um contra limites (lo, hi). |
| `draft_preset_kwargs` | função | `(description, expected_preset, *, bounds=None, provider='ollama', model=None, api_key=Non…)` | Repete `draft_problem` até obter o preset esperado com kwargs dentro dos limites; levanta `DraftRejected` ao esgotar as tentativas. |

#### `pinneapple_llm.finetune`

Fine-tuning LoRA de um modelo causal local no formato Hugging Face (transformers + peft, extra `finetune`) com os pares registrados no log; não faz fine-tuning direto de modelo Ollama. [sem teste direto]

<small>`pinneapple_llm/finetune.py` · 116 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FinetuneConfig` | classe | `(base_model, output_dir, lora_r, lora_alpha, lora_dropout, learning_rate, num_train_epoch…)` | Configuração do fine-tuning: modelo base, pasta de saída e hiperparâmetros de LoRA e treino. |
| `prepare_dataset` | função | `(store, out_jsonl, *, module=None) -> str` | Exporta os pares (prompt, resposta) do log para JSONL, para inspeção antes de treinar. |
| `finetune_lora` | função | `(dataset_jsonl, cfg)` | Roda um fine-tuning LoRA real com `transformers.Trainer` e `peft` sobre o JSONL preparado. |

#### `pinneapple_llm.geometry_draft`

Rascunho de geometria: o LLM escolhe e parametriza um gerador existente (domínios 2-D/3-D nomeados ou `naca_parametric`), conferido contra o registro. [sem teste direto]

<small>`pinneapple_llm/geometry_draft.py` · 121 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GeometryDraftResult` | classe | `(generator, kind, kwargs, reasoning, raw_response)` | Resultado: gerador escolhido, tipo (`domain2d`, `domain3d`, `airfoil`), kwargs, raciocínio e resposta bruta. |
| `draft_geometry` | função | `(description, *, provider='anthropic', model=None, api_key=None, conversation_store=None)…)` | Pede ao LLM para escolher e parametrizar um gerador de geometria; não constrói a geometria. |

#### `pinneapple_llm.guardrail`

`PhysicsGuardrail`: roda checagens independentes e recalculadas (sanidade de parâmetros, análise dimensional, resíduo da EDP, conservação de massa ou calor, erro contra referência) e só marca o resultado como confiável se passar. Não garante correção.

<small>`pinneapple_llm/guardrail.py` · 1.069 linhas · testes: 60</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CheckResult` | classe | `(name, passed, detail, value, threshold)` | Resultado de uma checagem: nome, se passou, detalhe, valor e limite. |
| `GuardrailReport` | classe | `(checks)` | Relatório do guardrail: checagens que rodaram, as puladas, veredito de confiança e resumo. |
| `PhysicsGuardrail` | classe | `(spec, *, residual_threshold=0.01, n_check_points=4096, conservation_n_points_per_face=20…)` | Executa a sequência fixa de checagens contra um modelo treinado e o `ProblemSpec` em que ele foi treinado. |

#### `pinneapple_llm.local_llm`

Provedor local Ollama via HTTP: iniciar `ollama serve`, baixar o modelo e fazer chamadas de chat sem dados saindo da máquina. [sem teste direto]

<small>`pinneapple_llm/local_llm.py` · 119 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `start_server` | função | `(*, host='127.0.0.1', port=11434, wait_s=10.0) -> subprocess.Popen` | Inicia `ollama serve` como subprocesso em segundo plano; falha se o `ollama` não estiver instalado. |
| `ensure_model_pulled` | função | `(model, *, host='127.0.0.1', port=11434, timeout=3600) -> None` | Roda `ollama pull` do modelo se ele ainda não estiver presente. |
| `call_ollama` | função | `(prompt, model, api_key=None, system='', *, host='127.0.0.1', port=11434, timeout=300, js…)` | Chamada de chat a um servidor Ollama local, com erro claro se ele não estiver acessível. |

#### `pinneapple_llm.org_knowledge`

Respostas do LLM fundamentadas nos resultados reais de um `RetrievalIndex` local de `pinneapple_analysis.retrieval` ("LLM local que conhece o código da organização").

<small>`pinneapple_llm/org_knowledge.py` · 133 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OrgAnswer` | classe | `(question, context_used, answer)` | Resposta: pergunta, trechos de contexto realmente usados (fonte, título, score) e o texto da resposta. |
| `answer_with_context` | função | `(question, retrieval_index, *, top_k=5, provider='ollama', model='llama3.2:3b', api_key=N…)` | Busca no índice e pede ao LLM uma resposta baseada só nesses resultados. |

#### `pinneapple_llm.qa`

Garantia de qualidade científica e industrial para artefatos do `pinneapple_llm`: estágios A-G e níveis Q0-Q4.

<small>`pinneapple_llm/qa/__init__.py` · 51 linhas · testes: 48</small>

#### `pinneapple_llm.qa.examples`

Exemplos de referência executáveis do pipeline de qualidade (importados sob demanda).

<small>`pinneapple_llm/qa/examples/__init__.py` · 10 linhas · testes: 39</small>

#### `pinneapple_llm.qa.examples.pipe_wear_twin`

Exemplo ponta a ponta: gêmeo digital 3-D de um tubo com desgaste de parede, construído com `pinneapple_twin3d` e passado pelos estágios A-G.

<small>`pinneapple_llm/qa/examples/pipe_wear_twin.py` · 342 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `hollow_pipe_mesh` | função | `(*, radius=OUTER_DIAMETER_M / 2, thickness=WALL_THICKNESS_M, length=LENGTH_M, n_theta=64)…)` | Malha de tubo realmente oco (paredes externa e interna e duas coroas nas pontas) por revolução. |
| `solid_pipe_mesh` | função | `(*, radius=OUTER_DIAMETER_M / 2, length=LENGTH_M, n_theta=64) -> Tuple[np.ndarray, np.nda…)` | Cilindro maciço com as mesmas medidas externas, usado nos testes como falha que o validador de geometria precisa pegar. |
| `build_spec` | função | `(*, anomaly=False) -> PhysicalSpecification` | Monta a `PhysicalSpecification` do exemplo (com ou sem condição de anomalia). |
| `build_twin` | função | `(*, anomaly=False) -> Dict[str, Any]` | Monta a cena do gêmeo de desgaste com malhas, campos e sensores. |
| `export_twin` | função | `(bundle, output_dir=None) -> str` | Escreve os arquivos reais do visualizador e o contrato do hub; devolve o `scene.json`. |
| `analytical_reference` | função | `(bundle) -> ReferenceData` | Espessura residual em forma fechada para desgaste uniforme; é verificação de fórmulas, não de física. |
| `run_example` | função | `(*, output_dir=None, anomaly=False, write_report=True) -> Tuple[QualityReport, Dict[str, …)` | Roda os estágios A-G sobre o gêmeo sintético e devolve o relatório. |

#### `pinneapple_llm.qa.examples.pipe_wear_twin_main`

Script que roda o exemplo de referência e imprime o relatório de qualidade; `--anomaly` demonstra a falha das invariantes. [sem teste direto]

<small>`pinneapple_llm/qa/examples/pipe_wear_twin_main.py` · 37 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `main` | função | `(argv=None) -> int` | Lê argumentos, roda `run_example` e imprime o relatório; devolve 0 ou 1. |

#### `pinneapple_llm.qa.generation`

Estágio B: captura as alegações estruturadas do LLM (premissas, dependências, unidades, equações, limitações) e as confronta com a especificação. [sem teste direto]

<small>`pinneapple_llm/qa/generation.py` · 240 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GeneratedArtifact` | classe | `(code, assumptions, dependencies, units, equations, limitations, provider, model, elapsed…)` | Proposta do LLM mais suas alegações estruturadas; não confiável por construção. |
| `build_prompt` | função | `(spec) -> str` | Monta um prompt que carrega a especificação inteira, não uma paráfrase. |
| `parse_generation_response` | função | `(raw, *, provider='', model='', elapsed_s=0.0) -> Tuple[GeneratedArtifact, StaticResult]` | Faz parse e checa o esquema da resposta do LLM; não levanta exceção com JSON malformado. |
| `check_claims` | função | `(artifact, spec) -> List[Finding]` | Confronta as declarações do modelo com a especificação e a tabela de unidades; devolve achados. |
| `validate_generated` | função | `(artifact, spec, *, filename='artifact.py') -> Tuple[List[Finding], Any]` | Checa as alegações e roda a validação estática completa do código devolvido. |
| `generate_artifact` | função | `(spec, *, provider='anthropic', model=None, api_key=None, conversation_store=None) -> Gen…)` | Pede a um provedor um artefato mais envelope de alegações, via `_dispatch.call_llm`. |

#### `pinneapple_llm.qa.levels`

Níveis de qualidade Q0-Q4, cumulativos e dados só pelo que rodou; Q4 exige evidência operacional fornecida por humano. [sem teste direto]

<small>`pinneapple_llm/qa/levels.py` · 182 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `evaluate_level` | função | `(report, spec=None) -> Tuple[str, List[str]]` | Devolve (nível, motivos), com os motivos de ter parado ali. |
| `blocking_reasons` | função | `(report) -> List[str]` | Motivos legíveis pelos quais o relatório não pode ser tratado como aprovado. |
| `level_description` | função | `(level) -> str` | Descrição textual de um nível. |

#### `pinneapple_llm.qa.pipeline`

O pipeline de qualidade: liga os estágios A-G em uma execução reprodutível. [sem teste direto]

<small>`pinneapple_llm/qa/pipeline.py` · 455 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PipelineConfig` | classe | `(validators, run_artifacts, source_name, static_policy, repair_attempts)` | Configuração do pipeline: validadores, se roda artefatos, política estática e tentativas de reparo. |
| `QualityPipeline` | classe | `(spec, *, validators=None, sandbox=None, run_artifacts=False, static_policy=None, referen…)` | Executa os estágios sobre uma especificação e um artefato e devolve o `QualityReport`. |

#### `pinneapple_llm.qa.references`

Estágio F: comparação com dados de referência independentes; recusa inventar referência ou limite de aceitação. [sem teste direto]

<small>`pinneapple_llm/qa/references.py` · 202 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ReferenceData` | classe | `(name, origin, version, values, unit, limitations, used_for_calibration)` | Referência de comparação, com origem, versão e limitações obrigatórias. |
| `compare` | função | `(predictions, reference, criteria, *, subset='validation', groups=None, baseline=None) ->…)` | Compara previsões com a referência sob critérios dados pelo chamador; só as métricas pedidas bloqueiam. |
| `split_mask` | função | `(n, holdout_fraction=0.3, seed=0) -> np.ndarray` | Máscara booleana determinística que marca o subconjunto de validação (holdout). |

#### `pinneapple_llm.qa.repair`

Estágio G: diagnóstico estruturado e reparo automático limitado, que rejeita reparos que mudem os critérios da especificação. [sem teste direto]

<small>`pinneapple_llm/qa/repair.py` · 296 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Diagnosis` | classe | `(failed_steps, findings, confirmed_cause, probable_cause, constraints_to_preserve, tests_…)` | Por que uma tentativa foi rejeitada, num formato que o passo de reparo consome. |
| `RepairAttempt` | classe | `(attempt, accepted, status, diagnosis, diff, notes, duration_s)` | Registro de uma tentativa de reparo: aceita ou não, status, diagnóstico e diff. |
| `RepairOutcome` | classe | `(accepted, artifact, attempts, results, stopped_because)` | Resultado do laço de reparo: aceito, artefato final, tentativas e motivo da parada. |
| `RepairLoop` | classe | `(*, evaluate, repair, spec=None, required=(), max_attempts=DEFAULT_MAX_ATTEMPTS, on_attem…)` | Laço limitado gerar -> validar -> diagnosticar -> reparar. |
| `diagnose` | função | `(results, *, required=()) -> Diagnosis` | Junta os resultados de validadores que falharam num diagnóstico estruturado. |

#### `pinneapple_llm.qa.report`

Relatório de qualidade reprodutível: status por estágio (sem booleano "ok"), hashes dos artefatos, versões e comando de reprodução. [sem teste direto]

<small>`pinneapple_llm/qa/report.py` · 327 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Status` | classe | `()` | Enum de status: PASS, FAIL, BLOCKED, NOT_RUN, NOT_APPLICABLE. |
| `Finding` | classe | `(id, status, message, severity, evidence, justification, limitations)` | Um fato observado, com a evidência que o produziu. |
| `StepOutcome` | classe | `(step, status, findings, duration_s, metadata)` | Resultado de um estágio do pipeline, com seus achados. |
| `ArtifactRecord` | classe | `(name, kind, path, sha256, size_bytes, metadata)` | Registro de um artefato: nome, tipo, caminho, sha256, tamanho e metadados. |
| `QualityReport` | classe | `(run_id, created_at, spec_id, spec_hash, criteria_hash, provider, model, steps, artifacts…)` | Tudo o que uma execução do pipeline observou e o que ela pode afirmar; serializa para JSON. |
| `environment_snapshot` | função | `() -> Dict[str, Any]` | Versões e plataforma, para rastrear o relatório até um ambiente. |

#### `pinneapple_llm.qa.sandbox`

Estágio D: execução controlada de artefatos em subprocesso com ambiente limpo e limites de recursos POSIX; não é sandbox de sistema operacional. [sem teste direto]

<small>`pinneapple_llm/qa/sandbox.py` · 389 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `IsolationLevel` | classe | `()` | Enum do nível de isolamento: bloqueado ou subprocesso. |
| `SandboxLimits` | classe | `(timeout_s, cpu_s, memory_mb, file_size_mb, open_files, max_processes, write_bytes)` | Tetos de recursos (CPU, memória, tamanho de arquivo etc.); None desliga o limite. |
| `SandboxResult` | classe | `(status, isolation, workdir, exit_code, stdout, stderr, duration_s, timed_out, violations…)` | Resultado da execução: status, código de saída, saídas, tempo, timeout, violações e arquivos produzidos. |
| `SandboxExecutor` | classe | `(*, limits=None, allow_execution=False, allow_network=False, base_dir=None, keep_workdir=…)` | Roda um artefato Python gerado num processo separado e limitado. |
| `hash_bytes` | função | `(blob) -> str` | SHA-256 em hexadecimal de um bloco de bytes. |

#### `pinneapple_llm.qa.spec`

Estágio A: especificação física tipada e serializável, separando o que veio do usuário, de fonte citada, de premissa ou de estimativa. [sem teste direto]

<small>`pinneapple_llm/qa/spec.py` · 496 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Provenance` | classe | `()` | Enum de proveniência: USER, SOURCE, ASSUMPTION, ESTIMATED. |
| `SpecError` | classe | `()` | Erro para especificação estruturalmente inválida. |
| `Quantity` | classe | `(value, unit, provenance, source, uncertainty)` | Número com unidade, proveniência e incerteza opcional. |
| `Invariant` | classe | `(name, kind, target, params, severity, description)` | Restrição física declarativa (tipo mais parâmetros) que o artefato deve cumprir. |
| `Observation` | classe | `(sensor_id, quantity, unit, position, timestamps, time_unit, values, provenance, holdout)` | Leitura de sensor: onde, o quê, quando e se pode ser usada para validação. |
| `GeometrySpec` | classe | `(kind, coordinate_system, dimensions, hollow, wall_thickness, notes)` | Geometria, domínio e sistema de coordenadas com unidades explícitas. |
| `PhysicalSpecification` | classe | `(objective, phenomenon, geometry, independent_variables, dependent_variables, parameters,…)` | Saída do estágio A: o pedido do usuário expresso de forma checável, com hash dos critérios. |
| `quantity_map` | função | `(**kwargs) -> Dict[str, Quantity]` | Atalho para montar um dicionário de `Quantity` a partir de tuplas (valor, unidade). |

#### `pinneapple_llm.qa.static`

Estágio C: validação estática por AST e introspecção real dos módulos PINNeAPPle (sintaxe, imports, contratos de API, assinaturas, segurança, esquema JSON). Não é sandbox nem prova de correção. [sem teste direto]

<small>`pinneapple_llm/qa/static.py` · 596 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `StaticPolicy` | classe | `(allowed_roots, forbidden_roots, forbidden_calls, forbidden_attr_calls, allowed_write_roo…)` | Política estática: raízes de import permitidas e proibidas, chamadas proibidas e pastas onde se pode escrever. |
| `StaticResult` | classe | `(findings, resolved_imports)` | Resultado da validação estática: achados e imports resolvidos. |
| `check_syntax` | função | `(source, filename='<artifact>') -> List[Finding]` | Checa a compilação; um `SyntaxError` vira achado. |
| `check_imports` | função | `(source, policy) -> Tuple[List[Finding], Dict[str, Any]]` | Recusa imports proibidos e relata os desconhecidos. |
| `check_api_contracts` | função | `(source, resolved) -> List[Finding]` | Resolve de verdade cada atributo `pinneapple_*` usado pelo artefato, pegando APIs inventadas. |
| `check_signatures` | função | `(source, resolved) -> List[Finding]` | Compara chamadas literais com a `inspect.signature` real. |
| `check_security` | função | `(source, policy, write_roots=()) -> List[Finding]` | Recusa criação de processos, execução de código, rede e escrita fora do escopo. |
| `check_json_artifact` | função | `(data, schema=None, name='artifact') -> List[Finding]` | Valida um artefato JSON contra um subconjunto pequeno de JSON Schema. |
| `validate_python_artifact` | função | `(source, *, policy=None, filename='<artifact>', write_roots=(), json_schema=None) -> Stat…)` | Roda todas as checagens estáticas e devolve um `StaticResult`. |
| `validate_json_artifact` | função | `(payload, schema=None, name='artifact') -> StaticResult` | Valida um payload JSON e devolve um `StaticResult`. |
| `load_json_file` | função | `(path) -> Any` | Lê um arquivo JSON. |

#### `pinneapple_llm.qa.units`

Análise dimensional sem dependências para o pipeline, com gramática e tabela de unidades fixas; unidade desconhecida levanta `UnitError`. [sem teste direto]

<small>`pinneapple_llm/qa/units.py` · 327 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `UnitError` | classe | `()` | Erro de unidade não interpretável ou conversão incompatível. |
| `Unit` | classe | `(name, dim, scale, offset)` | Unidade em escala SI (`si = valor * escala + offset`) com dimensão. |
| `parse_unit` | função | `(text) -> Unit` | Converte texto de unidade num `Unit`. |
| `are_compatible` | função | `(a, b) -> bool` | Diz se duas unidades têm a mesma dimensão física. |
| `convert` | função | `(value, from_unit, to_unit) -> float` | Converte um valor entre unidades compatíveis. |
| `dimension_of` | função | `(unit) -> Dim` | Devolve a dimensão de uma unidade. |
| `dimension_str` | função | `(unit) -> str` | Devolve a dimensão de uma unidade como texto. |

#### `pinneapple_llm.qa.validators`

Validadores científicos do estágio E e seu registro; os embutidos se registram na importação. [sem teste direto]

<small>`pinneapple_llm/qa/validators/__init__.py` · 19 linhas · **sem teste direto**</small>

#### `pinneapple_llm.qa.validators.base`

Protocolo de validador, contexto compartilhado e registro; inaplicável vira NOT_APPLICABLE e falta de dado vira BLOCKED, nunca PASS. [sem teste direto]

<small>`pinneapple_llm/qa/validators/base.py` · 164 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SensorReading` | classe | `(id, position, unit, quantity, series, holdout, position_unit)` | Um sensor: posição, unidade e série no eixo de tempo do artefato. |
| `ValidationContext` | classe | `(spec, meshes, fields, field_units, times, time_unit, sensors, payload, model, extra)` | Tudo o que um validador pode ver (especificação, malhas, campos, tempos, sensores, modelo). |
| `ValidatorResult` | classe | `(name, status, findings, evidence, limitations, duration_s)` | Resultado de um validador: status, achados, evidência, limitações e tempo. |
| `register` | função | `(name) -> Callable[[Validator], Validator]` | Decorador que registra um validador por nome; nome repetido dá erro. |
| `available` | função | `() -> List[str]` | Lista os nomes dos validadores registrados. |
| `get` | função | `(name) -> Validator` | Devolve o validador pelo nome, com erro claro se desconhecido. |
| `run` | função | `(name, ctx) -> ValidatorResult` | Executa um validador pelo nome e mede o tempo. |
| `not_applicable` | função | `(name, reason, **evidence) -> ValidatorResult` | Monta um resultado NOT_APPLICABLE com o motivo. |
| `blocked` | função | `(name, reason, **evidence) -> ValidatorResult` | Monta um resultado BLOCKED com o motivo. |

#### `pinneapple_llm.qa.validators.geometry`

Validação de geometria, incluindo a checagem de corpo oco: compara o volume da malha com o implicado pelas dimensões da especificação. [sem teste direto]

<small>`pinneapple_llm/qa/validators/geometry.py` · 206 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check_geometry` | função | `(ctx) -> ValidatorResult` | Valida as malhas do artefato (volume, oco vs maciço); NOT_APPLICABLE sem malhas. |

#### `pinneapple_llm.qa.validators.guardrail`

Ponte para o `PhysicsGuardrail`: aplica-se só quando há modelo treinado e `ProblemSpec`. [sem teste direto]

<small>`pinneapple_llm/qa/validators/guardrail.py` · 70 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check_guardrail` | função | `(ctx) -> ValidatorResult` | Roda o `PhysicsGuardrail` e transforma cada checagem num achado. |

#### `pinneapple_llm.qa.validators.numeric`

Integridade numérica: valores finitos, formas coerentes com tempos e malhas, campos degenerados. [sem teste direto]

<small>`pinneapple_llm/qa/validators/numeric.py` · 108 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check_numeric` | função | `(ctx) -> ValidatorResult` | Checa finitude, forma e degenerescência dos campos numéricos do artefato. |

#### `pinneapple_llm.qa.validators.physics`

Avalia as invariantes físicas declaradas na especificação sobre os arrays do artefato; alvo ausente vira BLOCKED. [sem teste direto]

<small>`pinneapple_llm/qa/validators/physics.py` · 203 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check_physics` | função | `(ctx) -> ValidatorResult` | Avalia cada invariante declarada contra os dados do artefato. |

#### `pinneapple_llm.qa.validators.sensors`

Consistência de sensores: posição, tempo e grandeza existentes, e separação entre leituras de calibração e de validação. [sem teste direto]

<small>`pinneapple_llm/qa/validators/sensors.py` · 123 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check_sensors` | função | `(ctx) -> ValidatorResult` | Confere as leituras de sensores contra o domínio, o eixo de tempo e as grandezas modeladas. |

#### `pinneapple_llm.qa.validators.temporal`

Consistência temporal: ordenação, número de amostras e janela de validade. [sem teste direto]

<small>`pinneapple_llm/qa/validators/temporal.py` · 99 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check_temporal` | função | `(ctx) -> ValidatorResult` | Confere o eixo de tempo e as contagens de amostras; NOT_APPLICABLE em artefato estacionário. |

#### `pinneapple_llm.qa.validators.units`

Consistência dimensional entre geometria, parâmetros, campos e sensores; não prova que as equações estão certas. [sem teste direto]

<small>`pinneapple_llm/qa/validators/units.py` · 142 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check_units` | função | `(ctx) -> ValidatorResult` | Confere que cada grandeza declarada tem unidade válida e compatível com a especificação. |

#### `pinneapple_llm.research`

Busca de literatura e repositórios no arXiv e GitHub com as fontes reais de `pinneapple_tools.hpo_experiments.sources`; o LLM só resume ou ordena os resultados reais. [sem teste direto]

<small>`pinneapple_llm/research.py` · 85 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ResearchReport` | classe | `(query, papers, repos, llm_summary)` | Relatório de busca: consulta, artigos, repositórios e resumo opcional do LLM. |
| `search_literature` | função | `(query, *, k_papers=5, k_repos=5, summarize=False, provider='anthropic', model=None, api_…)` | Busca artigos e repositórios reais e, opcionalmente, pede ao LLM um resumo deles. |

#### `pinneapple_llm.twin_draft`

Rascunho de gêmeo digital: o LLM propõe valores para os campos de `DigitalTwinConfig`; o gêmeo é sempre construído por `build_digital_twin`. [sem teste direto]

<small>`pinneapple_llm/twin_draft.py` · 156 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TwinDraftResult` | classe | `(config_kwargs, reasoning, raw_response)` | Resultado: kwargs de configuração propostos, raciocínio e resposta bruta. |
| `draft_digital_twin` | função | `(description, *, provider='anthropic', model=None, api_key=None, conversation_store=None)…)` | Pede ao LLM kwargs de `DigitalTwinConfig` para uma descrição; não constrói o gêmeo. |
| `build_3d_live_twin` | função | `(model, field_names, *, export_dir, points, config_kwargs=None, **build_kwargs)` | Constrói um `DigitalTwin` real com um gancho que grava cada atualização como quadro `.ply` para `pinneapple_blender`. |

#### `pinneapple_llm.workbench`

Agent Workbench: um agente líder e especialistas (explorador, pesquisador, programador, testador, revisor, engenheiro de física, relator) num modelo local.

<small>`pinneapple_llm/workbench/__init__.py` · 22 linhas · testes: 20</small>

#### `pinneapple_llm.workbench.__main__`

Linha de comando `python -m pinneapple_llm.workbench` (ou `pinneapple-workbench`): `serve`, `doctor`, `pull`, `run`, `eval`, `finetune`, `vscode`.

<small>`pinneapple_llm/workbench/__main__.py` · 274 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `cmd_serve` | função | `(args)` | Sobe o app web com uvicorn. |
| `cmd_doctor` | função | `(args)` | Verifica máquina, servidor de modelo e modelo e diz o que corrigir. |
| `cmd_pull` | função | `(args)` | Baixa com Ollama o modelo recomendado (ou o indicado). |
| `cmd_run` | função | `(args)` | Roda uma tarefa no terminal, sem navegador. |
| `cmd_eval` | função | `(args)` | Mede um modelo nas tarefas embutidas de avaliação. |
| `cmd_finetune` | função | `(args)` | Exporta o dataset do banco, treina um adaptador LoRA ou o serve. |
| `cmd_vscode` | função | `(args)` | Instala a extensão do VS Code que abre o workbench ao lado do editor. |
| `main` | função | `(argv=None)` | Monta o `argparse` dos subcomandos e despacha. |

#### `pinneapple_llm.workbench.agents`

Definições dos agentes: um Markdown por agente com front matter e prompt; versões do usuário ou do projeto substituem as embutidas pelo nome.

<small>`pinneapple_llm/workbench/agents.py` · 65 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Agent` | classe | `(name, title, description, tools, prompt, source, extra)` | Um agente: nome, título, descrição, ferramentas e prompt de sistema. |
| `parse_agent` | função | `(text, source='') -> Agent` | Lê um arquivo Markdown de agente (front matter mais corpo) num `Agent`. |
| `load_agents` | função | `(workspace=None, user_dir=USER_DIR) -> tuple[dict[str, Agent], str]` | Carrega todos os agentes (embutidos, do usuário, do workspace) e as regras comuns. |

#### `pinneapple_llm.workbench.backend`

Backends de chat com ferramentas: Ollama, qualquer servidor compatível com OpenAI e um backend roteirizado para testes; `Backend.chat` da base levanta `NotImplementedError`.

<small>`pinneapple_llm/workbench/backend.py` · 262 linhas · testes: 20 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ToolCall` | classe | `(name, arguments, id)` | Uma chamada de ferramenta pedida pelo modelo: nome, argumentos e id. |
| `Reply` | classe | `(content, tool_calls, usage)` | Resposta do modelo: texto, chamadas de ferramenta e uso de tokens. |
| `Backend` | classe | `(model, tool_mode)` | Interface base de backend (`chat`, `models`), com modo de ferramenta nativo ou JSON. |
| `OllamaBackend` | classe | `(model, host='http://127.0.0.1:11434', *, num_ctx=32768, temperature=0.2, tool_mode='nati…)` | Backend sobre `/api/chat` de um servidor Ollama local. |
| `OpenAIBackend` | classe | `(model, base_url='http://127.0.0.1:1234/v1', api_key='local', *, temperature=0.2, tool_mo…)` | Backend sobre `/v1/chat/completions` de servidor compatível com OpenAI (LM Studio, llama.cpp, vLLM, MLX ou hospedado). |
| `ScriptedBackend` | classe | `(script, model='scripted')` | Backend que repete respostas roteirizadas ou chama uma função, para testes e demos sem modelo. |
| `parse_json_tool_call` | função | `(text, tool_names) -> ToolCall \| None` | Lê uma chamada de ferramenta em JSON de uma resposta; None se não houver. |
| `make_backend` | função | `(kind='ollama', model='', **kw) -> Backend` | Cria o backend pelo tipo (`ollama`, `openai`, `lmstudio`, `llamacpp`, `vllm`, `mlx`, ...). |

#### `pinneapple_llm.workbench.evaluate`

Benchmark pequeno e checável do workbench na sua máquina e modelo: cada tarefa monta um projeto mínimo e confere o resultado mecanicamente.

<small>`pinneapple_llm/workbench/evaluate.py` · 128 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TaskResult` | classe | `(name, passed, seconds, model_calls, prompt_tokens, completion_tokens, detail)` | Resultado de uma tarefa: passou, tempo, chamadas ao modelo e tokens. |
| `run_eval` | função | `(backend, names=None, on_event=None) -> list[TaskResult]` | Roda as tarefas de avaliação com um backend e devolve os resultados. |
| `format_results` | função | `(results, model) -> str` | Formata os resultados como tabela de texto. |
| `as_dicts` | função | `(results) -> list[dict]` | Converte os resultados em lista de dicts. |

#### `pinneapple_llm.workbench.finetune`

Fine-tuning do modelo local nas próprias sessões: exporta trajetórias do banco (com segredos mascarados) e treina LoRA com `mlx_lm` (Apple Silicon) ou TRL + PEFT (CUDA).

<small>`pinneapple_llm/workbench/finetune.py` · 178 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TrainConfig` | classe | `(model, adapter_path, iters, batch_size, num_layers, learning_rate, max_seq_length, lora_…)` | Configuração do treino: modelo, pasta do adaptador e hiperparâmetros de LoRA. |
| `redact` | função | `(text) -> str` | Mascara tokens e chaves que parecem segredos num texto. |
| `export_dataset` | função | `(store, out_dir, *, min_rating=0, mode='trajectories', agents=None, val_fraction=0.1, see…)` | Escreve `train.jsonl` e `valid.jsonl` a partir do banco, filtrando por avaliação; devolve as contagens. |
| `mlx_train_command` | função | `(data_dir, cfg) -> list[str]` | Monta a linha de comando de `mlx_lm.lora`. |
| `train_mlx` | função | `(data_dir, cfg=None, *, dry_run=False) -> str` | Faz fine-tuning LoRA com `mlx_lm` e devolve a pasta do adaptador. |
| `serve_mlx_command` | função | `(cfg=None, port=8080) -> list[str]` | Monta a linha de comando de `mlx_lm.server` com o adaptador. |
| `train_peft` | função | `(data_dir, cfg) -> str` | Faz fine-tuning LoRA com TRL + PEFT e devolve a pasta do adaptador. |

#### `pinneapple_llm.workbench.models`

Escolhe o modelo local a partir da memória da máquina.

<small>`pinneapple_llm/workbench/models.py` · 75 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Machine` | classe | `(system, arch, memory_gb, apple_silicon)` | Dados da máquina: sistema, arquitetura, memória e se é Apple Silicon. |
| `Recommendation` | classe | `(min_memory_gb, model, approx_size_gb, context, note)` | Recomendação de modelo: memória mínima, tag Ollama, tamanho, contexto e nota. |
| `detect_machine` | função | `() -> Machine` | Detecta sistema, arquitetura e memória da máquina. |
| `recommend` | função | `(memory_gb=None) -> Recommendation` | Modelo padrão para a memória dada (ou detectada). |

#### `pinneapple_llm.workbench.runtime`

O laço de agentes: o líder conversa, chama ferramentas, mantém um plano e delega a especialistas; cada passo vira evento armazenado. (A marca de placeholder vem da ferramenta `todo`, não de código incompleto.)

<small>`pinneapple_llm/workbench/runtime.py` · 301 linhas · testes: 20 · palavras-chave no código (heurística; o texto acima diz o status real): placeholder/stub citado no docstring</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Cancelled` | classe | `()` | Exceção usada para cancelar um turno em andamento. |
| `SessionSettings` | classe | `(max_steps, max_depth, context_chars, keep_recent, language)` | Limites da sessão: passos por agente, profundidade de delegação, tamanho de contexto e compactação. |
| `Session` | classe | `(backend, workspace, *, policy=None, settings=None, on_event=None, agents=None, common=No…)` | Uma sessão de trabalho: envia mensagens, roda o laço do líder, atende aprovações e perguntas e pode ser cancelada. |

#### `pinneapple_llm.workbench.server`

App web FastAPI sobre `Session` e `Store`, com frontend estático; escuta em 127.0.0.1 por padrão e aceita token de acesso.

<small>`pinneapple_llm/workbench/server.py` · 318 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Manager` | classe | `(store, backend_factory, default_workspace, language='')` | Gerencia sessões vivas (uma thread por turno) sobre o banco. |
| `create_app` | função | `(*, backend='ollama', model='', host='', base_url='', api_key='', workspace='.', db=DEFAU…)` | Cria o app FastAPI com as rotas da API e o frontend. |

#### `pinneapple_llm.workbench.store`

Banco do workbench (SQLite local ou PostgreSQL): sessões, eventos, chamadas ao modelo e avaliações para fine-tuning. [sem teste direto]

<small>`pinneapple_llm/workbench/store.py` · 158 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Store` | classe | `(url=DEFAULT_DB)` | Acesso ao banco: cria e lista sessões, grava eventos, chamadas ao modelo e feedback, e dá estatísticas. |

#### `pinneapple_llm.workbench.tools`

Ferramentas dos agentes confinadas a uma pasta: arquivos, busca, comandos, jobs em segundo plano, internet, GitHub e exibição de arquivos, com política de aprovação.

<small>`pinneapple_llm/workbench/tools.py` · 355 linhas · testes: 37</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ToolError` | classe | `()` | Erro de ferramenta devolvido ao agente. |
| `ApprovalPolicy` | classe | `(mode)` | Política de aprovação (`ask`, `auto`, `yolo`) que decide quais ações pedem confirmação. |
| `Workspace` | classe | `(root, policy=None, approve=None, run_timeout=600, web=True)` | Pasta que os agentes podem ler e alterar; implementa as ferramentas e pede aprovação nas ações protegidas. |

#### `pinneapple_llm.workbench.web`

Busca na internet (SearXNG, Brave, Tavily ou DuckDuckGo) e leitura de páginas para os agentes.

<small>`pinneapple_llm/workbench/web.py` · 142 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `parse_duckduckgo` | função | `(page) -> list[dict]` | Extrai título, URL e trecho dos resultados da página HTML do DuckDuckGo. |
| `web_search` | função | `(query, max_results=8) -> list[dict]` | Busca com o primeiro provedor configurado e devolve `[{title, url, snippet}]`. |
| `html_to_text` | função | `(page) -> str` | Converte HTML em texto legível. |
| `fetch_text` | função | `(url, max_chars=12000) -> str` | Lê o texto de uma página ou arquivo; links `blob` do GitHub são lidos crus. |

### Exemplos de uso reais do repositório

**Especificação física do pipeline de qualidade** — `docs/guides/pinneapple_llm_quality_pipeline.md`

```python
import pinneapple_llm.qa as qa

spec = qa.PhysicalSpecification(
    objective="Track pipe wall wear over five years",
    phenomenon="erosive wall thinning",
    geometry=qa.GeometrySpec(
        kind="hollow_pipe",
        dimensions={"outer_radius": qa.Quantity(0.10, "m"),
                    "length": qa.Quantity(2.0, "m")},
        hollow=True,
        wall_thickness=qa.Quantity(0.010, "m"),
    ),
```

**resolve_problem sem credenciais cai no casamento por palavras** — `tests/test_catalog_match.py`

```python
def test_resolve_problem_falls_back_without_llm_credentials():
    # No API key is configured in this environment -- resolve_problem must
    # never raise, and must fall back to the deterministic keyword match.
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        os.environ.pop(var, None)

    result = resolve_problem("burgers equation turbulence one dimensional")
    assert result.used_llm is False
    assert result.preset is not None
```

**Subir o Agent Workbench com modelo local** — `docs/guides/agent_workbench.md`

```python
brew install ollama            # or the app from https://ollama.com; it uses the Mac's GPU (Metal)
ollama serve &                 # the Ollama app does this for you
pip install "pinneapple[workbench]"

python -m pinneapple_llm.workbench doctor       # memory, Ollama, model, Python packages: what to fix
python -m pinneapple_llm.workbench pull         # downloads the model recommended for your memory
python -m pinneapple_llm.workbench serve --workspace ~/code/my-project --language Português
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 21 de 49.
- Funções de teste em arquivos que importam algum módulo do pacote: 242.

### Limitações conhecidas

- O `PhysicsGuardrail` não garante que um resultado esteja correto (o docstring diz isso explicitamente); a checagem de conservação só existe para as famílias de EDP com balanço conhecido (massa e condução de calor).
- Os provedores hospedados (Anthropic, OpenAI) exigem chave de API e o extra `llm`; o caminho local exige Ollama instalado, sem fallback embutido.
- `finetune` e `workbench.finetune` dependem de extras pesados (`transformers`, `peft`, `trl`, `mlx_lm`); `finetune` não treina modelos Ollama diretamente.
- `cad_draft`: templates paramétricos e exportação STEP precisam de `cadquery`; sem motor booleano (`manifold3d`) as operações CSG viram concatenação das partes.
- `qa.sandbox` é isolamento por subprocesso com limites POSIX, não um sandbox de sistema operacional; `qa.static` não prova que o código faz o que diz.
- `qa.units` entende só uma gramática e uma tabela de unidades fixas; unidade desconhecida levanta `UnitError`.
- O nível Q4 do pipeline de qualidade não pode ser alcançado dentro do repositório: exige evidência operacional fornecida por humano.
- O workbench executa comandos na máquina do usuário; o servidor escuta em 127.0.0.1 e não deve ser exposto à rede. Modelos locais não igualam os maiores modelos hospedados em tarefas longas, segundo o próprio guia.

## 7. `pinneapple_mcp`

### Objetivo

`pinneapple_mcp` expõe o PINNeAPPle como servidor MCP (Model Context Protocol) via stdio. Qualquer agente de código que fale MCP pode resolver, comparar e validar problemas físicos com a biblioteca, e cada resultado volta com suas verificações físicas e um cartão de confiança (APPROVED / REVIEW / REJECT).

As ferramentas ficam em `pinneapple_mcp.tools` como funções Python comuns, com argumentos e retornos em JSON, utilizáveis sem MCP. Elas chamam o resto da biblioteca: `pp.solve` e `pp.list_methods` (em `pinneapple_physics.solving`), o catálogo de métodos, os presets de problemas, `pp.zoo` e `pp.datasets`, o design por intenção (`pinneapple_design.intent`) e a cadeia de porquês (`pinneapple_reason`). Um problema pode ser um preset ou um 'problema em caixa' descrito campo a campo em JSON, com expressões das coordenadas analisadas por `ast` (nunca `eval`).

O servidor tem padrões seguros (`config`): trabalho longo (`solve`, `benchmark`, `design`, `reason`) roda como job num processo separado com limite de tempo, limite de memória e rede desligada; resultados vão para um diretório de trabalho e só se leem arquivos de raízes permitidas. `import pinneapple_mcp` não importa o SDK MCP; ele só é carregado por `server` (SDK 1.x `FastMCP` ou 2.x `MCPServer`).

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 8 | 1.541 | 2 | 31 | 30 (3 arquivos) | 3 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `__main__` | Permite rodar o servidor com `python -m pinneapple_mcp`. | 1 | 0 | 0 |
| `config` | Limites e locais do servidor (diretório de trabalho, tempo, memória, jobs, épocas, pontos, rede, raízes) com variáveis de ambiente. | 1 | 3 | 10 |
| `expressions` | Compila expressões aritméticas seguras das coordenadas em funções NumPy, sem `eval`. | 1 | 2 | 10 |
| `jobs` | Executa trabalho longo como jobs em processos separados, com estado gravado em disco. | 1 | 5 | 0 |
| `problems` | Monta um problema físico a partir de JSON: preset ou problema em caixa com condições de contorno. | 1 | 2 | 0 |
| `server` | Servidor MCP stdio que registra todas as ferramentas com assinaturas tipadas. | 1 | 2 | 10 |
| `tools` | As ferramentas MCP como funções puras com entrada e saída JSON. | 1 | 17 | 30 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_mcp`

Ponto de entrada do pacote; `build_server` e `main` importam `server` só quando chamados, para não exigir o SDK MCP no import.

<small>`pinneapple_mcp/__init__.py` · 34 linhas · testes: 30</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_server` | função | `(*args, **kwargs)` | Importa `server` sob demanda e devolve o objeto servidor MCP. |
| `main` | função | `(argv=None)` | Importa `server` sob demanda e roda o servidor stdio. |

#### `pinneapple_mcp.__main__`

Chama `server.main()` para `python -m pinneapple_mcp`. [sem teste direto]

<small>`pinneapple_mcp/__main__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_mcp.config`

`Config` com padrões seguros, cada um sobrescrevível por variável de ambiente `PINNEAPPLE_MCP_*`; `get_config`/`set_config` mantêm a configuração global.

<small>`pinneapple_mcp/config.py` · 91 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Config` | classe | `(workdir, max_seconds, max_memory_mb, max_jobs, max_epochs, max_points, max_file_mb, allo…)` | Configuração do servidor: diretório, tempo e memória por job, jobs simultâneos, épocas, pontos, tamanho de arquivo, rede e raízes de leitura. |
| `get_config` | função | `() -> Config` | Devolve a configuração ativa. |
| `set_config` | função | `(cfg) -> Config` | Define a configuração ativa e a devolve. |

#### `pinneapple_mcp.expressions`

Analisa texto com `ast` aceitando só números, nomes de coordenadas, `pi`/`e`, operadores aritméticos e uma lista fixa de funções NumPy; qualquer outra coisa levanta `ValueError`.

<small>`pinneapple_mcp/expressions.py` · 105 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `compile_expression` | função | `(text, coords) -> Callable[[np.ndarray], np.ndarray]` | Compila o texto numa função `f(X) -> (N,)` das coordenadas, ou levanta `ValueError`. |
| `value` | função | `(v, coords)` | Mantém números e compila strings como expressão. |

#### `pinneapple_mcp.jobs`

Cada job roda num processo filho com `RLIMIT_AS`, conexões de rede não locais bloqueadas e diretório próprio com `request.json`, `status.json` e `log.txt`; o estado sobrevive a reinícios (job sem processo vira `lost`). [sem teste direto pelo inventário; exercitado por `tests/test_mcp_server.py` via `tools`]

<small>`pinneapple_mcp/jobs.py` · 199 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `running` | função | `() -> list[str]` | Ids dos jobs com processo ainda vivo. |
| `submit` | função | `(kind, args, cfg, max_seconds=None) -> str` | Inicia um job de um tipo de `tools.JOB_KINDS` num novo processo e devolve o id. |
| `status` | função | `(job_id, cfg, wait_seconds=0.0) -> dict[str, Any]` | Estado de um job, seu resultado quando pronto e os arquivos do diretório, podendo esperar até `wait_seconds`. |
| `cancel` | função | `(job_id, cfg) -> dict[str, Any]` | Mata o processo de um job em execução. |
| `list_jobs` | função | `(cfg, limit=20) -> list[dict[str, Any]]` | Lista os jobs mais recentes com estado. |

#### `pinneapple_mcp.problems`

Constrói `BuiltProblem` a partir de JSON: preset com `preset_kwargs` ou problema em caixa (domínio, campos, EDP, fonte, condições de Dirichlet, solução exata). [sem teste direto pelo inventário]

<small>`pinneapple_mcp/problems.py` · 156 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BuiltProblem` | classe | `(problem, ctx, exact, dirichlet, source, spec)` | Problema montado: `PhysicalProblem`, contexto, solução exata, condições de Dirichlet e fonte; `check_pde` diz qual EDP estacionária pode ser verificada. |
| `build_problem` | função | `(spec) -> BuiltProblem` | Monta um `BuiltProblem` a partir do JSON, ou levanta `ValueError` com o motivo. |

#### `pinneapple_mcp.server`

Monta o servidor MCP (SDK 2.x `MCPServer` ou 1.x `FastMCP`) com todas as ferramentas e CLI `pinneapple-mcp` com opções de limites.

<small>`pinneapple_mcp/server.py` · 284 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_server` | função | `(config=None)` | Cria o objeto servidor MCP com todas as ferramentas do PINNeAPPle. |
| `main` | função | `(argv=None) -> None` | CLI do servidor: lê opções de limites, ajusta a configuração e roda em stdio. |

#### `pinneapple_mcp.tools`

Implementa as ferramentas: capacidades, catálogo de métodos, presets, `trust_card`, `solve` e `benchmark` como jobs, zoo, jobs, `design` e as ferramentas de `pinneapple_reason`.

<small>`pinneapple_mcp/tools.py` · 667 linhas · testes: 30</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `capabilities` | função | `() -> dict[str, Any]` | Informa versão, APIs opcionais presentes, dispositivo, limites e diretório de trabalho. |
| `list_methods` | função | `(category='all', validated_only=False) -> dict[str, Any]` | Métodos de `solve` agrupados por tipo e o catálogo de métodos com status de validação. |
| `describe_method` | função | `(method) -> dict[str, Any]` | Descreve um método de `solve` com suas opções, ou um item do catálogo com código, referências e testes. |
| `list_presets` | função | `(contains=None) -> dict[str, Any]` | Lista os presets de problema utilizáveis em `solve` e `benchmark`. |
| `trust_card` | função | `(file=None, field=None, grid=None, values=None, pde=None, source=None, alpha=1.0, dirichl…)` | Verificações físicas de um campo em grade (inline ou arquivo) e o cartão de confiança com decisão. |
| `solve` | função | `(problem, method='pinn', options=None, reference=None, reference_options=None, grid_point…)` | Resolve um problema com um método num job; o resultado traz métricas, verificações, cartão de confiança e arquivos. |
| `benchmark` | função | `(problem, methods=('pinn', 'fem'), budget=None, seeds=1, reference=None, options=None, gr…)` | Roda vários métodos num problema com o mesmo orçamento de treino e pontua na mesma grade, num job. |
| `zoo_list` | função | `(task=None) -> dict[str, Any]` | Lista modelos pré-treinados verificados do zoo e datasets do hub. |
| `zoo_predict` | função | `(name, inputs) -> dict[str, Any]` | Previsões de um modelo do zoo, marcando as linhas fora da faixa de validade. |
| `zoo_validate` | função | `(name, inputs=None) -> dict[str, Any]` | Relatório de confiança de um modelo do zoo, opcionalmente checando se entradas estão na faixa válida. |
| `job_status` | função | `(job_id, wait_seconds=0.0) -> dict[str, Any]` | Estado de um job e, quando pronto, o resultado com os caminhos dos arquivos. |
| `list_jobs` | função | `(limit=20) -> dict[str, Any]` | Jobs mais recentes com estado. |
| `cancel_job` | função | `(job_id) -> dict[str, Any]` | Para um job em execução matando o processo. |
| `design` | função | `(text, answers=None, verify=True, seed=0, max_seconds=None, wait_seconds=0.0) -> dict[str…)` | Design a partir de um objetivo em linguagem natural: responde na hora `out_of_scope` ou `needs_input`, senão inicia um job que gera o pacote de design. |
| `reason` | função | `(case=None, requirements=None, concept=None, lang='pt', max_steps=12, max_seconds=None, w…)` | Roda a cadeia de porquês de `pinneapple_reason` num job, aplicando contramedidas até cada requisito passar. |
| `reason_library` | função | `(domain=None, lang='pt') -> dict[str, Any]` | Lista os modos de falha por domínio físico, as contramedidas e os casos resolvidos de `pinneapple_reason`. |
| `reason_check` | função | `(mode, params, elements=(), limits=None, lang='pt') -> dict[str, Any]` | Avalia um modo de falha nos parâmetros dados (SI): demanda, capacidade, margem, mecanismo e fontes. |

### Exemplos de uso reais do repositório

**Problema em caixa resolvido por FEM via ferramenta MCP** — `tests/test_mcp_server.py`

```python
PLATE = {"name": "plate", "domain": {"x": [0, 1], "y": [0, 1]}, "fields": ["T"], "pde": "laplace",
         "bcs": [{"kind": "dirichlet", "field": "T", "on": "y_max", "value": "sin(pi*x)"},
                 {"kind": "dirichlet", "field": "T", "on": "y_min", "value": 0},
                 {"kind": "dirichlet", "field": "T", "on": "x_boundary", "value": 0}],
         "exact": "sin(pi*x)*sinh(pi*y)/sinh(pi)"}
...
def test_solve_job_fem_with_checks(cfg):
    st = tools.solve(PLATE, "fem", {"n": 24}, grid_points_per_axis=25, wait_seconds=180)
    assert st["status"] == "done", st
    res = st["result"]
    assert res["reference"] == "exact" and res["metrics"]["relative_l2"]["T"] < 0.02
    assert res["decision"] == "APPROVED", res["trust_cards"]
```

**Instalação e execução do servidor** — `docs/guides/mcp_server.md`

```python
pip install "pinneapple[mcp]"
pinneapple-mcp --help            # or: python -m pinneapple_mcp
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 5 de 8.
- Funções de teste em arquivos que importam algum módulo do pacote: 30.

### Limitações conhecidas

- O servidor precisa do extra `pinneapple[mcp]` (SDK MCP 1.x ou 2.x e `pydantic`); as ferramentas em `tools` funcionam sem ele.
- Limites padrão por job: 600 s, 8192 MB de espaço de endereçamento, 2 jobs simultâneos, 20000 épocas, 250000 pontos e arquivos de até 200 MB; o limite de memória usa `resource` e é ignorado em plataformas sem ele.
- A rede dentro dos jobs fica desligada substituindo `socket.connect` (só conexões locais e Unix passam), a menos que `PINNEAPPLE_MCP_ALLOW_NETWORK=1`.
- Expressões aceitam só aritmética das coordenadas e uma lista fixa de funções NumPy; qualquer outra construção é recusada.
- `problems.BuiltProblem.check_pde` só verifica resíduo para EDPs estacionárias `laplace`, `poisson` e `heat_equation`.
- `jobs`, `problems` e `__main__` não têm teste direto pelo inventário (são exercitados indiretamente por `tests/test_mcp_server.py`).

## 8. `pinneapple_autolab`

### Objetivo

`pinneapple_autolab` é o "Autonomous Physics Lab": uma IA que administra evidências para tomar decisões de engenharia com um simulador de referência caro. Em vez de confiar num surrogate, o laboratório separa o que foi simulado (evidência) do que foi só previsto (rotulado como previsão até uma simulação confirmar), e entrega como recomendação sempre um projeto simulado, com o seu relatório de confiança e o livro de registro (ledger).

O pacote tem duas camadas. A primeira é o laço de agente (`core`): uma missão em linguagem natural vira um plano de chamadas controladas de biblioteca (`Tool`, `Toolbox`, `Step`), escolhidas por um planejador determinístico (`TemplatePlanner`) ou por um modelo de linguagem (`LLMPlanner`, `LLMStepPlanner`), e cada chamada fica registrada como `Evidence`. A segunda é o ciclo científico (`lab.AutonomousLab`): planejar, checar geometria e condições de contorno, simular, aprender (famílias de surrogates comparadas por validação cruzada, com incerteza), otimizar (NSGA-II sobre o surrogate), avaliar cada candidato (domínio validado, probabilidade de cumprir os limites, erro de campo esperado na região crítica), escolher a próxima simulação pelo que ainda é desconhecido, verificar e entregar.

As regras são impostas em código: orçamento de simulações com reserva para verificação (`Budget`, `BudgetExhaustedError`), limites relativos ao baseline que só valem depois de o baseline ser simulado (`UnresolvedLimitError`), condições de operação que o laboratório não pode mudar, e uma auditoria (`agent.audit_claims`) que exige que todo número da resposta final venha das ferramentas. O domínio de exemplo é `domains.racecar`: um carro de fórmula paramétrico (não o carro de nenhuma equipe) simulado em OpenFOAM, com a missão "menos arrasto mantendo a downforce".

Liga-se ao resto do repositório pelas variáveis e objetivos de `pinneapple_design.design_optimizer.study`, pela geometria `pinneapple_design.geometry.vehicles3d.FormulaCar`, pelo `TrustReport` de `pinneapple_analysis.trust`, por `pinneapple_llm` nos planejadores com LLM e pelo extrator `pinneapple_design.intent` na leitura de missões. É usado pelo nó `autolab` de `pinneapple_flow`, por agentes em `pinneapple_design.thermal.board_cooling` e `pinneapple_systems.digital_twin.pipe_loop`, e pelos exemplos `examples/autolab*`.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 16 | 3.596 | 38 | 39 | 37 (5 arquivos) | 10 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `agent` | O laboratório dirigido por um agente: texto da missão -> `Mission` -> chamadas de ferramentas sobre o laboratório -> resposta auditada. | 1 | 9 | 6 |
| `core` | Laço de agente genérico: ferramentas, caixa de ferramentas, planejadores (modelo fixo ou LLM), execução e registro de evidências. | 1 | 13 | 0 |
| `dashboard` | Painel HTML com quatro áreas sincronizadas (geometria, física, modelos, decisões) sobre o estado real de uma sessão, estático, ao vivo ou em replay. | 1 | 5 | 5 |
| `domain` | Domínio validado: diz se um projeto está perto o bastante dos projetos simulados para a previsão valer. | 1 | 2 | 0 |
| `domains` | Simuladores e missões do laboratório, um módulo por domínio físico (hoje `racecar`). | 2 | 11 | 11 |
| `lab` | O laboratório autônomo: o ciclo planejar -> simular -> aprender -> otimizar -> avaliar -> verificar -> entregar sobre qualquer simulador. | 1 | 2 | 10 |
| `ledger` | Livro de registro de evidências: cada simulação e cada evento da sessão, em ordem, com orçamento. | 1 | 4 | 0 |
| `mission` | A missão em números: variáveis, objetivos, limites, baseline, orçamento e tolerâncias de aceitação. | 1 | 4 | 0 |
| `qualification` | Qualificação do solver antes da campanha: quanto da resposta depende da malha e do número de iterações. | 1 | 1 | 0 |
| `replay` | Simulador que responde a partir de um ledger gravado quando o projeto é exatamente o mesmo, e roda o real caso contrário. | 1 | 1 | 6 |
| `report` | Figuras e relatório de engenharia em Markdown de uma sessão, feitos só a partir do snapshot. | 1 | 8 | 0 |
| `simulator` | Interface do simulador de referência (`check`, `run`) e um simulador a partir de função Python. | 1 | 3 | 0 |
| `surrogates` | Surrogates do solver (GP, ensemble de MLP, superfície de resposta quadrática, POD de campo) comparados por validação cruzada. | 1 | 9 | 0 |
| `trust` | Triagem dos candidatos do otimizador (favorável, incerto, fora do domínio, improvável) e a regra de escolha da próxima simulação. | 1 | 5 | 0 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_autolab`

Ponto de entrada do Autonomous Physics Lab: reexporta missão, ledger, laboratório, simuladores, surrogates, domínio validado e triagem.

<small>`pinneapple_autolab/__init__.py` · 70 linhas · testes: 31</small>

#### `pinneapple_autolab.agent`

Laboratório dirigido por agente: lê a missão em texto, expõe uma caixa fechada de ferramentas (sem ferramenta para mudar condições, limites ou solver) e audita a resposta final.

<small>`pinneapple_autolab/agent.py` · 488 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MissionRefused` | classe | `()` | Exceção: o pedido não pode ser executado como escrito; traz o motivo e o que falta. |
| `MissionDomain` | classe | `(title, variables, baseline, output_words, units, operating_point, acceptance, conditions…)` | O que um simulador pode receber como pedido: variáveis, baseline, palavras que nomeiam saídas, ponto de operação fixo e tolerâncias padrão. |
| `MissionReading` | classe | `(mission, readings, assumptions)` | Resultado da leitura de uma missão: a `Mission`, os trechos lidos e as suposições feitas. |
| `LabPlanner` | classe | `(lab, n_doe=None)` | Planejador determinístico de passos sobre `lab_toolbox` (sem rede): baseline, projetos que preenchem o espaço, ciclos aprender/avaliar/simular, verificação e resposta. |
| `AgentResult` | classe | `(reading, refused, answer, audit, run, summary, wall_s, planner)` | Resultado de `run_agent`: leitura da missão, recusa, resposta, auditoria, execução, resumo, tempo e planejador usado. |
| `interpret_mission` | função | `(text, dom, llm=None) -> MissionReading` | Lê o texto numa `Mission` sobre o domínio dado; um LLM opcional só é aceito onde escolhe opções fechadas e cita palavras do texto. |
| `lab_toolbox` | função | `(lab, reading=None) -> Toolbox` | Caixa fechada de ferramentas com que o agente dirige o laboratório; cada ferramenta devolve o medido ou o previsto (rotulado). |
| `audit_claims` | função | `(answer, tool_log, lab, recommended_run=None, rtol=0.005) -> dict[str, Any]` | Confere se cada número da resposta aparece no que as ferramentas devolveram e se o run recomendado existe, teve sucesso e cumpre os limites. |
| `run_agent` | função | `(text, simulator, folder, *, domain, planner=None, provider=None, model=None, llm=None, m…)` | Interpreta o texto, monta o laboratório e deixa o agente dirigi-lo (planejador dado, LLM se houver provedor, senão `LabPlanner`). |

#### `pinneapple_autolab.core`

Laço de agente missão -> plano -> chamadas controladas -> evidência: o agente nunca roda código próprio, só ferramentas registradas com argumentos checados. [sem teste direto]

<small>`pinneapple_autolab/core.py` · 297 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Tool` | classe | `(name, fn, description, inputs, summary)` | Uma chamada de biblioteca permitida: `fn(state, **args) -> dict`, descrição, entradas aceitas e resumo para o log. |
| `Toolbox` | classe | `(kind, tools=None)` | Conjunto controlado de ferramentas de um tipo de missão (`add`, `names`, `check`, `describe`). |
| `Step` | classe | `(tool, args, why)` | Um passo do plano: ferramenta, argumentos e justificativa. |
| `Plan` | classe | `(mission, steps, planner, notes)` | Plano de uma missão: passos, planejador usado e notas. |
| `Planner` | classe | `()` | Interface base de planejador que produz um `Plan`. |
| `TemplatePlanner` | classe | `(steps)` | Planejador determinístico com a sequência fixa de passos de um tipo de missão. |
| `LLMPlanner` | classe | `(fallback, *, provider='anthropic', model=None, llm=None, required=())` | Um modelo de linguagem ordena as ferramentas num plano JSON; ferramenta ou argumento desconhecido faz cair no plano modelo. |
| `Evidence` | classe | `(step, tool, args, why, summary, seconds, ok, error)` | Registro de uma chamada: passo, ferramenta, argumentos, justificativa, resumo do resultado, tempo, sucesso e erro. |
| `Run` | classe | `(plan, state, evidence)` | Resultado de uma execução: plano, estado com as saídas por ferramenta e a lista de evidências. |
| `StepPlanner` | classe | `(name)` | Interface de planejador que escolhe o próximo passo depois de ver os resultados (None quando termina). |
| `LLMStepPlanner` | classe | `(*, provider='anthropic', model=None, llm=None, rules='')` | Um modelo de linguagem escolhe cada próximo passo vendo missão, ferramentas e evidências; passo inválido é recusado e registrado. |
| `execute` | função | `(plan, toolbox, state=None, *, stop_on_error=True, log=None) -> Run` | Executa os passos do plano em ordem, só pela caixa de ferramentas, com estado compartilhado. |
| `act` | função | `(mission, toolbox, planner, state=None, *, max_steps=200, max_refusals=5, log=None) -> Run` | Laço decidir-e-agir: pede um passo, checa, executa, registra, até o planejador terminar ou atingir os limites de passos ou recusas. |

#### `pinneapple_autolab.dashboard`

Painel da sessão a partir dos arquivos (`ledger.json`, `events.jsonl`, snapshot, resumo): página estática, servidor ao vivo, screenshot e vídeo do replay.

<small>`pinneapple_autolab/dashboard.py` · 233 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `live_state` | função | `(folder, out_dir=None, tail=120) -> dict[str, Any]` | Estado da sessão lido dos arquivos, com um `status` do que está acontecendo agora. |
| `build_dashboard` | função | `(folder, out_dir, *, scenes=None, title='', state=None, live=False, replay=False, figures…)` | Grava `index.html` do painel com o estado embutido, opcionalmente ao vivo (polling de `state.json`) ou com replay. |
| `serve` | função | `(folder, out_dir, port=8120, host='127.0.0.1', block=True, **kw)` | Serve o painel ao vivo: a página e `/state.json` relido a cada requisição. |
| `capture` | função | `(html, png, *, width=1600, height=1100, full_page=True, replay_to=None, wait_ms=5000) -> …)` | Screenshot de uma página do painel (precisa de `playwright`). |
| `record_replay` | função | `(html, mp4, *, frames=60, fps=4.0, hold_s=3.0, width=1600, height=2040, out_width=1280, c…)` | Vídeo MP4 do replay do painel via ffmpeg (precisa de `playwright`). |

#### `pinneapple_autolab.domain`

Domínio validado: distância de um projeto ao projeto simulado mais próximo, em unidades do espaçamento típico dos simulados, mais a faixa coberta por variável. [sem teste direto]

<small>`pinneapple_autolab/domain.py` · 72 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DomainVerdict` | classe | `(inside, ratio, nearest, reasons)` | Veredito de domínio: dentro ou fora, razão de distância, vizinho mais próximo e motivos em palavras. |
| `ValidatedDomain` | classe | `(U, names=None, bounds=None, max_ratio=1.5, margin=0.05)` | Domínio dos projetos simulados; `assess` diz se um projeto está dentro (razão até `max_ratio` e variáveis na faixa com margem). |

#### `pinneapple_autolab.domains`

Pacote dos simuladores e missões por domínio físico. [sem teste direto]

<small>`pinneapple_autolab/domains/__init__.py` · 2 linhas · **sem teste direto**</small>

#### `pinneapple_autolab.domains.racecar`

Simulador do "AI Aerodynamicist": carro de fórmula paramétrico (`FormulaCar`, não de nenhuma equipe) em OpenFOAM (snappyHexMesh + simpleFoam k-omega SST), com a missão principal e o painel da sessão.

<small>`pinneapple_autolab/domains/racecar.py` · 464 linhas · testes: 11</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RaceCarAero` | classe | `(resolution='campaign', iterations=800, procs=4, n_lines=40, keep_cases=False, conditions…)` | Simulador de aerodinâmica externa do carro em OpenFOAM: checa o projeto, roda o caso e devolve forças, balanço e Cp de superfície por região. |
| `make_car` | função | `(params)` | Constrói um `FormulaCar` a partir dos parâmetros. |
| `force_convergence` | função | `(history, window=2) -> dict[str, float]` | Variação relativa de arrasto e downforce nos últimos intervalos gravados. |
| `solver_log_summary` | função | `(case) -> dict[str, Any]` | Resumo do log: última iteração, resíduos finais, se o checkMesh passou, células e tempos. |
| `render_run` | função | `(params, flow_npz, out_prefix, title='', size=(1200, 675)) -> dict[str, str]` | Renders PyVista de um projeto simulado: Cp de superfície e linhas de corrente. |
| `cycles_renders` | função | `(params, flow_npz, out_prefix, title='', samples=64, size=(1600, 900)) -> dict[str, str]` | Renders Blender Cycles de um projeto simulado (precisa de `bpy`; sem ele devolve {}). |
| `flow_scene` | função | `(params, flow_npz, title='')` | Cena do estúdio 3D de um projeto simulado (partes do carro, Cp/Cf, linhas de corrente, fatias). |
| `build_dashboard` | função | `(work, out_dir, summary, *, figures=None, figure_runs=None, viewers=True, replay=True, re…)` | Painel de uma sessão do carro com renders do projeto original e do recomendado, visualizadores 3D e a sessão gravada. |
| `recorded_dashboard` | função | `(results_dir, out_dir) -> str` | Painel de uma sessão gravada só a partir dos arquivos versionados, sem OpenFOAM e sem visualizador 3D. |
| `racecar_domain` | função | `()` | Descrição `MissionDomain` do que o simulador do carro pode receber, para `interpret_mission`. |
| `flagship_mission` | função | `(budget=30, reserve_verify=3) -> Mission` | A missão principal: menor arrasto a 70 m/s mantendo a downforce, dentro das restrições geométricas, validado por CFD. |

#### `pinneapple_autolab.lab`

O laboratório autônomo sobre qualquer simulador, com cada etapa disponível como método e registrada no ledger; impõe orçamento, checagens antes de simular e recomendação só de projeto simulado.

<small>`pinneapple_autolab/lab.py` · 546 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AutonomousLab` | classe | `(mission, simulator, folder, *, families=…, field_family='gp', seed=0, nsga_budget=3000, …)` | O laboratório: missão, simulador, pasta, famílias de surrogates, região crítica e tolerância de campo; métodos para cada etapa do ciclo e `run()` para o ciclo inteiro. |
| `showcase` | função | `(decisions, baseline) -> dict[str, dict[str, Any]]` | Escolhe por regra fixa três projetos simulados que o laboratório julgou de forma diferente antes de simular (incerto, favorável, fora do domínio). |

#### `pinneapple_autolab.ledger`

Livro de evidências: cada simulação (por quê, com o quê, custo, resultado) e cada evento, em ordem; é a única fonte de números para relatório, painel e agente. [sem teste direto]

<small>`pinneapple_autolab/ledger.py` · 185 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BudgetExhaustedError` | classe | `()` | Exceção: o orçamento de simulações da missão acabou. |
| `RunRecord` | classe | `(id, purpose, params, ok, outputs, message, cost_s, started, reason, predicted, meta, fie…)` | Uma simulação de referência: id, propósito, parâmetros, sucesso, saídas, mensagem, custo, motivo e o que o surrogate previa antes. |
| `Ledger` | classe | `(folder, max_runs=None, max_wall_s=None, listeners=None)` | Runs e eventos de uma sessão numa pasta, com controle de orçamento, `events.jsonl` incremental e `ledger.json`. |
| `environment` | função | `(extra=None) -> dict[str, Any]` | Hardware e software da máquina: CPU, núcleos, memória, sistema, versões de Python e pacotes, commit do git. |

#### `pinneapple_autolab.mission`

Missão em números: variáveis, objetivos, limites (absolutos ou relativos ao baseline), baseline, orçamento, aceitação e condições fixas. [sem teste direto]

<small>`pinneapple_autolab/mission.py` · 161 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `UnresolvedLimitError` | classe | `()` | Exceção: um limite relativo ao baseline foi usado antes de o baseline ser simulado. |
| `Budget` | classe | `(max_runs, reserve_verify, max_wall_s)` | Orçamento: máximo de simulações, reserva para a verificação final e tempo total opcional. |
| `Limit` | classe | `(name, lower, upper, lower_frac_of_baseline, upper_frac_of_baseline, unit, kind, label)` | Limite `lower <= saída <= upper`, absoluto ou como fração do baseline, com unidade e tipo (físico, geometria, validade). |
| `Mission` | classe | `(title, statement, variables, objectives, constraints, baseline, budget, acceptance, cond…)` | O que o laboratório deve fazer; resolve limites relativos com as saídas do baseline e serializa de/para dict. |

#### `pinneapple_autolab.qualification`

Qualificação do solver: roda o mesmo projeto em variantes de malha/iterações (fora do orçamento) e mede a diferença relativa de cada saída contra a referência. [sem teste direto]

<small>`pinneapple_autolab/qualification.py` · 49 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `solver_qualification` | função | `(sim, params, variants, folder, reference, campaign, outputs=None) -> dict[str, Any]` | Roda as variantes num ledger próprio e devolve, por saída, a diferença relativa de cada variante em relação à `reference`. |

#### `pinneapple_autolab.replay`

Simulador de replay: devolve a medição gravada quando o projeto está no ledger (sem interpolar) e roda o simulador real caso contrário.

<small>`pinneapple_autolab/replay.py` · 68 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ReplaySimulator` | classe | `(simulator, ledger_path, fields_dir=None, atol=1e-09)` | Envolve um simulador real e um ledger gravado; marca a origem do replay em `meta`. |

#### `pinneapple_autolab.report`

Figuras e relatório de engenharia de uma sessão, só a partir do snapshot; nada aqui calcula um número físico. [sem teste direto]

<small>`pinneapple_autolab/report.py` · 348 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `shrink_png` | função | `(path, colors=256) -> str` | Regrava um PNG com paleta adaptativa para reduzir o tamanho. |
| `pareto_front` | função | `(F) -> np.ndarray` | Índices das linhas não dominadas (todas as colunas minimizadas), ordenados pela primeira coluna. |
| `fig_pareto` | função | `(snap, path, x='drag_N', y='downforce_N', scale=0.001, units='kN', annotate=None) -> str` | Figura arrasto x downforce com candidatos, frente do surrogate, projetos simulados, baseline e limite. |
| `fig_models` | função | `(snap, path, outputs, labels=None, scale=None) -> str` | Figura das previsões por validação cruzada contra a referência, com intervalos de 95 %, e notas das famílias. |
| `fig_learning` | função | `(snap, path, outputs, objective, acceptance) -> str` | Figura do erro de cada previsão feita antes da simulação e do melhor objetivo viável contra simulações gastas. |
| `fig_field_triptych` | função | `(points, ref, pred, path, title='', clim=(-1.5, 1.0), err_max=0.3, label='Cp', mirror=Tru…)` | Referência \| previsão \| \|erro\| de um campo de superfície, vistas de cima e de baixo. |
| `fig_trust_cards` | função | `(cards, path, outputs, units=None) -> str` | Um painel por caso de confiança: previsão ± 1,96 std, referência quando simulada e decisão. |
| `write_report` | função | `(summary, path, figures=None, qualification=None, extra='') -> str` | Relatório de engenharia em Markdown: missão, plano, qualificação, orçamento, modelos, decisões, recomendação e ambiente. |

#### `pinneapple_autolab.simulator`

Contrato do simulador de referência (`check` antes, `run` com custo depois; falha nunca devolve saídas inventadas) e um simulador de função para testes. Os métodos da base levantam `NotImplementedError` (abstratos). [sem teste direto]

<small>`pinneapple_autolab/simulator.py` · 91 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SimOutcome` | classe | `(outputs, ok, message, cost_s, fields, meta)` | Resultado de uma simulação: saídas, sucesso, mensagem, custo, campos opcionais e metadados. |
| `Simulator` | classe | `(name, fidelity, outputs, field_names, regions)` | Classe base: define nome, fidelidade, saídas, campos e regiões; subclasses implementam `check` e `run`. |
| `FunctionSimulator` | classe | `(fn, outputs, *, name='function', checker=None, fail=None, field_names=(), regions=None, …)` | Simulador a partir de `fn(params) -> {saída: valor}`, com checador e falha simulada opcionais (testes, problemas analíticos). |

#### `pinneapple_autolab.surrogates`

Surrogates com incerteza (GP Matern 5/2, ensemble de MLP, superfície quadrática, POD de campo) e a comparação por validação cruzada k-fold; a base `_Model` é abstrata (`NotImplementedError`). [sem teste direto]

<small>`pinneapple_autolab/surrogates.py` · 260 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GPModel` | classe | `(restarts=1, seed=0)` | Processo gaussiano Matern 5/2 com escalas ARD (scikit-learn), com desvio padrão preditivo. |
| `MLPEnsemble` | classe | `(n=5, hidden=(32, 32), seed=0, max_iter=600)` | `n` MLPs pequenos em reamostragens bootstrap (scikit-learn); desvio = espalhamento dos membros. |
| `QuadraticRSM` | classe | `(alpha=1e-06)` | Regressão ridge numa base quadrática completa; desvio = desvio residual do próprio ajuste. |
| `ModelBoard` | classe | `(table, chosen, models, cv_pred, n_train, cv_y)` | Tabela de notas por validação cruzada, família escolhida por saída e modelos ajustados. |
| `FieldSurrogate` | classe | `(family='gp', energy=0.99, max_modes=12)` | POD de campos em pontos fixos mais uma regressão por modo; desvio do campo vem dos modos. |
| `make_model` | função | `(family, **kw) -> _Model` | Cria um modelo da família pedida; família desconhecida levanta `ValueError`. |
| `cross_validate` | função | `(family, U, y, k=5, seed=0) -> dict[str, Any]` | Validação cruzada k-fold de uma família numa saída: métricas e previsões fora da amostra. |
| `compare_models` | função | `(U, Y, families=…, k=5, seed=0, select='rmse') -> ModelBoard` | Valida todas as famílias em todas as saídas, escolhe a de menor métrica por saída (penalizando as mal calibradas) e reajusta. |
| `field_errors` | função | `(pred, ref, regions=None, std=None) -> dict[str, dict[str, float]]` | RMSE, erro máximo e cobertura de 95 % de um campo previsto, no total e por região. |

#### `pinneapple_autolab.trust`

Triagem de candidatos: probabilidade de viabilidade, melhoria esperada restrita, domínio validado e erro de campo na região crítica, com quatro rótulos e a regra da próxima simulação. [sem teste direto]

<small>`pinneapple_autolab/trust.py` · 262 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CandidateAssessment` | classe | `(index, params, predicted, std, p_feasible, cei, domain, label, reasons, report, critical…)` | Avaliação de um candidato: previsão e desvio, P(viável), EI restrita, domínio, rótulo, motivos, relatório e, depois, a referência. |
| `NextExperiment` | classe | `(candidate, rule, reason, alternatives)` | A próxima simulação escolhida: candidato, regra, motivo e alternativas. |
| `assess_candidates` | função | `(X, names, pred, objectives, constraints, domain_verdicts, best_value, acceptance, model_…)` | Avalia os candidatos do otimizador e atribui `favourable`, `uncertain`, `out_of_domain` ou `unlikely`. |
| `choose_next` | função | `(cands, objective, exclude=()) -> NextExperiment` | Escolhe a próxima simulação pela regra escrita no docstring do módulo. |
| `verification_checks` | função | `(c, reference, ok, acceptance, constraints, message='') -> CandidateAssessment` | Atualiza uma avaliação com o resultado de referência: erros relativos medidos, confirmação e viabilidade real. |

### Exemplos de uso reais do repositório

**Missão de brinquedo com simulador analítico** — `tests/test_autolab.py`

```python
def toy_sim(**kw):
    return FunctionSimulator(toy, ["drag", "down"], field_names=["cp"], regions={"tail": PTS > 0.7}, **kw)


def toy_mission(runs=12, reserve=1):
    return Mission("toy", "lower drag, keep downforce", [Variable("a", 0, 1), Variable("b", 0, 1)],
                   [Objective("drag", "min")], [Limit("down", lower_frac_of_baseline=1.0)],
                   baseline={"a": 0.5, "b": 0.5}, budget=Budget(runs, reserve),
                   acceptance={"drag": 0.03, "down": 0.03})
```

**O laboratório na missão do carro (docstring do pacote)** — `pinneapple_autolab/__init__.py`

```python
from pinneapple_autolab import AutonomousLab
from pinneapple_autolab.domains.racecar import RaceCarAero, flagship_mission

lab = AutonomousLab(flagship_mission(), RaceCarAero(), "runs/aero", critical_region="diffuser")
summary = lab.run()
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 6 de 16.
- Funções de teste em arquivos que importam algum módulo do pacote: 37.

### Limitações conhecidas

- `domains.racecar` precisa de OpenFOAM (snappyHexMesh, simpleFoam); a campanha completa do exemplo leva horas em 4 núcleos (docstring de `examples/autolab_aero/autolab_aero.py`). Renders pedem PyVista e, para Cycles, `bpy`.
- `dashboard.capture` e `dashboard.record_replay` precisam de `playwright` (e ffmpeg para o vídeo).
- Os planejadores com LLM (`LLMPlanner`, `LLMStepPlanner`, `run_agent` com `provider`) precisam de chave/serviço de um provedor; sem isso usa-se o planejador determinístico.
- `LabPlanner` é declarado como substituto de um modelo de linguagem, não um agente inteligente: segue uma sequência fixa de etapas.
- `Simulator` e a base dos surrogates são abstratos (`NotImplementedError`); cada domínio novo precisa implementar `check` e `run`.
- Os surrogates dependem de scikit-learn; a recomendação final é sempre um projeto simulado, então o orçamento de simulações limita o que pode ser recomendado.
- `core`, `domain`, `ledger`, `mission`, `qualification`, `report`, `simulator`, `surrogates` e `trust` não têm teste direto (são exercitados pelos testes que importam o pacote raiz).

## 9. `pinneapple_app`

### Objetivo

_Sem texto de objetivo._

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 18 | 3.672 | 29 | 53 | 48 (3 arquivos) | 11 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `backend` |  | 18 | 82 | 41 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_app.backend`

_Módulo sem docstring e sem texto._

<small>`pinneapple_app/backend/__init__.py` · 1 linhas · **sem teste direto**</small>

#### `pinneapple_app.backend.core`

_Módulo sem docstring e sem texto._

<small>`pinneapple_app/backend/core/__init__.py` · 1 linhas · **sem teste direto**</small>

#### `pinneapple_app.backend.core.admin_auth`

pinneapple_app.backend.core.admin_auth — a minimal, generic admin gate.

<small>`pinneapple_app/backend/core/admin_auth.py` · 31 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `require_admin` | função | `(x_admin_token=Header(default='')) -> None` | FastAPI dependency: raises 401 unless ``X-Admin-Token`` matches ``PINNEAPPLE_ADMIN_TOKEN``. If that env var is unset, admin routes are refused entirely (503) rather than silently open — there is no "admin token not configured, so let everyone in" fallback. |

#### `pinneapple_app.backend.core.benchmark`

Benchmark report generator — produces plots and tables from ExperimentResult.

<small>`pinneapple_app/backend/core/benchmark.py` · 208 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_benchmark_payload` | função | `(result) -> Dict[str, Any]` | Convert an ExperimentResult into a JSON-serialisable benchmark payload. |

#### `pinneapple_app.backend.core.collocation`

Collocation point generation strategies for pinneapple_app.

<small>`pinneapple_app/backend/core/collocation.py` · 208 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CollocationConfig` | classe | `(strategy, n_interior, n_boundary, n_initial, seed, use_geometry, geometry_sdf_fn, domain…)` | Configuration for collocation point generation. |
| `generate_points` | função | `(config, bounds, *, include_time=False, t_range=(0.0, 1.0)) -> Dict[str, np.ndarray]` | Generate collocation points from a config and domain bounds. |

#### `pinneapple_app.backend.core.data_pipeline`

Data generation pipeline — runs a solver to produce synthetic training data.

<small>`pinneapple_app/backend/core/data_pipeline.py` · 163 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DataConfig` | classe | `(solver_key, n_snapshots, grid_resolution, t_end, use_solver)` | Configuration for synthetic data generation. |
| `DataBundle` | classe | `(x_col, x_bnd, x_ic, u_ref, solver_outputs, meta)` | Output of the data pipeline, consumed by ExperimentRunner. |
| `run_data_pipeline` | função | `(problem, collocation_cfg, data_cfg, *, device='cpu', verbose=True) -> DataBundle` | Execute data generation: (optionally) run solver + sample collocation pts. |

#### `pinneapple_app.backend.core.experiment`

Experiment runner: trains multiple models on the same problem and collects results.

<small>`pinneapple_app/backend/core/experiment.py` · 475 linhas · testes: 7 · palavras-chave no código (heurística; o texto acima diz o status real): experimental</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ModelRunConfig` | classe | `(name, extra_kwargs, weight_override)` |  |
| `ExperimentConfig` | classe | `(experiment_id, problem_name, models, metrics, epochs, lr, device, batch_size, seed, grad…)` | Full configuration for one experimental run (one problem, N models). |
| `ModelResult` | classe | `(name, metrics, loss_history, physics_loss_history, bc_loss_history, train_time_s, n_para…)` |  |
| `ExperimentResult` | classe | `(experiment_id, config, model_results, completed_at)` |  |
| `ExperimentRunner` | classe | `(config, data_bundle, problem)` | Trains all configured models and streams progress via an async generator. |

#### `pinneapple_app.backend.core.model_catalog`

Model catalog: lists available models, enriches with metadata, and recommends.

<small>`pinneapple_app/backend/core/model_catalog.py` · 195 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ModelEntry` | classe | `(name, family, description, supports_physics_loss, tags, recommended_for)` |  |
| `list_models` | função | `(family=None) -> List[ModelEntry]` | Return all available models, optionally filtered by family. |
| `recommend_for_problem` | função | `(problem_family, n=5) -> List[str]` | Return top-N recommended model names for a given problem family. |
| `get_model_info` | função | `(name) -> Optional[ModelEntry]` |  |

#### `pinneapple_app.backend.core.physics_adapter`

Physics adapters: turn a ``ProblemDefinition`` into a trainable physics loss.

<small>`pinneapple_app/backend/core/physics_adapter.py` · 552 linhas · testes: 7 · palavras-chave no código (heurística; o texto acima diz o status real): placeholder/stub citado no docstring; levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ConditionInfo` | classe | `(key, name, kind, fields, weight, n_points, how, target_rms)` |  |
| `PhysicsAdapter` | classe | `()` | Shared interface used by the experiment runner and the report. |
| `PresetPhysics` | classe | `(spec, *, w_physics=None, w_bc=None)` |  |
| `CustomPhysics` | classe | `(problem, *, w_physics=None, w_bc=None)` |  |
| `unwrap` | função | `(out) -> torch.Tensor` | Model output -> plain (N, F) tensor (PINNOutput/ModelOutput carry ``.y``). |
| `spatial_coords` | função | `(coords) -> List[str]` |  |
| `all_faces` | função | `(coords) -> List[Face]` |  |
| `faces_for_tag` | função | `(tag, coords) -> Optional[List[Face]]` | Box-face convention for the tags in ``_BOX_TAGS``; ``None`` otherwise. |
| `sample_faces` | função | `(bounds, coords, faces, n, rng) -> Tuple[np.ndarray, np.ndarray]` | Uniform points on the given box faces, with outward unit normals. |
| `sample_initial` | função | `(bounds, coords, n, rng) -> np.ndarray` |  |
| `sample_interior` | função | `(bounds, coords, n, rng) -> np.ndarray` |  |
| `characteristic_length` | função | `(bounds, coords) -> float` | Mean spatial extent (time extent for ODE-only problems). |
| `parse_equation` | função | `(expr, coords, fields)` | ``"u_t + u*u_x - 0.01*u_xx"`` -> SymPy expression in u(x, t). |
| `parse_location` | função | `(location, bounds, coords) -> List[Face]` | ``""``/``"all"``/``"boundary"``, ``"x=0"``, ``"x_min"``, ``"left"``... |
| `build_physics` | função | `(problem, *, weight_override=None) -> Optional[PhysicsAdapter]` | Adapter for the problem, or ``None`` when it defines no physics at all. |

#### `pinneapple_app.backend.core.problem`

Problem definition layer for pinneapple_app.

<small>`pinneapple_app/backend/core/problem.py` · 184 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BoundaryConditionSpec` | classe | `(kind, location, value, field)` | Human-readable BC for custom problems. |
| `InitialConditionSpec` | classe | `(expression, field)` | Human-readable IC for custom problems. |
| `EquationSpec` | classe | `(expression, label, field)` | A single PDE residual equation (string or SymPy expression). |
| `ProblemDefinition` | classe | `(kind, name, spec, dim, domain_bounds, pde_family, is_time_dependent, preset_name, equati…)` | Unified problem container used throughout pinneapple_app. |
| `load_preset` | função | `(preset_name) -> ProblemDefinition` | Load a ProblemDefinition from the environment preset registry. |
| `define_custom` | função | `(name, equations, bcs, ics=None, domain_bounds=None, dim=2, pde_family=FAMILY_GENERIC, is…)` | Build a ProblemDefinition from user-supplied equations, BCs, and ICs. |

#### `pinneapple_app.backend.core.problem_registry`

Problem registry: metadata, solver recommendations, and model suggestions per preset.

<small>`pinneapple_app/backend/core/problem_registry.py` · 469 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `get_problem_meta` | função | `(preset_name) -> Dict[str, Any]` | Return metadata dict for a preset, with sensible defaults for unknowns. |
| `recommend_models` | função | `(preset_name) -> list` |  |
| `recommended_solver` | função | `(preset_name) -> str` |  |
| `all_problems` | função | `() -> list` |  |

#### `pinneapple_app.backend.core.surrogate_report`

Surrogate quality report: the "can I trust this model?" section of a run.

<small>`pinneapple_app/backend/core/surrogate_report.py` · 552 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `error_section` | função | `(model, fields, X_test, Y_test, reference_source) -> Dict[str, Any]` |  |
| `convergence_section` | função | `(loss_history) -> Dict[str, Any]` |  |
| `generalization_section` | função | `(model, X_train, Y_train, X_test, Y_test, res_train, res_unseen) -> Dict[str, Any]` |  |
| `variables_section` | função | `(model, adapter, X_eval, supervised) -> Dict[str, Any]` |  |
| `weights_section` | função | `(model, adapter, final_raw, w_data, data_loss) -> Dict[str, Any]` |  |
| `uncertainty_section` | função | `(model, snapshots, fields, X_eval, X_cal, Y_cal, X_test, Y_test, coverage=0.9) -> Dict[st…)` |  |
| `field_scales` | função | `(adapter, pred) -> np.ndarray` | Per-field magnitude: max of the predicted spread, the declared range and the RMS of the Dirichlet/initial values imposed on that field. |
| `residual_scale` | função | `(model, adapter, X, pred) -> float` | Typical size of the PDE's individual terms, to normalise its residual. |
| `physics_checks_section` | função | `(model, adapter, X_eval, raw_unseen, raw_initial, pred_eval) -> List[Dict[str, Any]]` |  |
| `verdict` | função | `(report) -> Dict[str, Any]` |  |
| `build_report` | função | `(*, model, adapter, loss_history, final_raw_train, raw_initial, raw_unseen, res_train, X_…)` | Assemble all sections. ``split`` holds numpy/tensor train/cal/test arrays. |

#### `pinneapple_app.backend.main`

FastAPI application entry point for PINNeAPPle App.

<small>`pinneapple_app/backend/main.py` · 112 linhas · testes: 41</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `health` | função | `()` |  |
| `metrics` | função | `() -> str` | Prometheus text-exposition format — deliberately hand-rolled (no new dependency) rather than pulling in ``prometheus-client`` for two gauges. See ``pinneapple_app/deploy/monitoring/prometheus.yml`` for the scrape config pointed at this endpoint. |
| `info` | função | `()` | Return library capabilities summary. |

#### `pinneapple_app.backend.routers`

_Módulo sem docstring e sem texto._

<small>`pinneapple_app/backend/routers/__init__.py` · 1 linhas · **sem teste direto**</small>

#### `pinneapple_app.backend.routers.admin`

pinneapple_app.backend.routers.admin — internal curation/status view: cross-references the problem catalog, the model registry (``pinneapple_registry.ArtifactRegistry``), and dataset scenarios into one dashboard-ready payload. Gated by ``core.admin_auth.require_admin`` (a generic bearer token, see …

<small>`pinneapple_app/backend/routers/admin.py` · 84 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ModelSummaryItem` | classe | `(problem_id, latest_model_version, stage, dataset_scenarios)` |  |
| `AdminSummaryResponse` | classe | `(registry_root, n_presets, n_architectures, models)` |  |
| `PromoteRequest` | classe | `(problem_id, version, stage)` |  |
| `admin_summary` | função | `() -> AdminSummaryResponse` |  |
| `promote_model` | função | `(req) -> Dict[str, Any]` | Promotes a stored model version to a new stage. Being behind ``require_admin`` already IS the confirmation gate here (unlike ``pinneapple_orchestration``'s flow-level ``confirm=True`` flag, which guards an unattended pipeline step instead of a human clicking a button in an admin UI). |

#### `pinneapple_app.backend.routers.experiments`

Experiments API router — launch, monitor, and retrieve experiments.

<small>`pinneapple_app/backend/routers/experiments.py` · 266 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CollocationRequest` | classe | `(strategy, n_interior, n_boundary, n_initial, seed, use_geometry, domain_name)` |  |
| `DataRequest` | classe | `(solver_key, n_snapshots, grid_resolution, t_end, use_solver)` |  |
| `ModelRequest` | classe | `(name, extra_kwargs)` |  |
| `ExperimentRequest` | classe | `(problem_name, custom_problem, collocation, data, models, metrics, epochs, lr, device, se…)` |  |
| `ExperimentStatusResponse` | classe | `(experiment_id, status, progress, message)` |  |
| `launch_experiment` | função | `(req, background_tasks)` | Launch a new experiment asynchronously. Returns experiment_id immediately. |
| `get_status` | função | `(experiment_id)` |  |
| `get_results` | função | `(experiment_id)` |  |
| `list_experiments` | função | `()` |  |
| `experiment_ws` | função | `(websocket, experiment_id)` | WebSocket endpoint — streams training progress events as JSON. |

#### `pinneapple_app.backend.routers.models`

Models API router.

<small>`pinneapple_app/backend/routers/models.py` · 84 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ModelListItem` | classe | `(name, family, description, supports_physics_loss, tags, recommended_for)` |  |
| `MetricItem` | classe | `(key, label)` |  |
| `list_models` | função | `(family=None)` | Return all available models, optionally filtered by family. |
| `list_families` | função | `()` | Return all model families. |
| `recommend_models` | função | `(problem_family, n=5)` | Return top-N recommended model names for a problem family. |
| `list_metrics` | função | `()` | Return all available evaluation metrics. |
| `get_model` | função | `(name)` | Return metadata for a single model. |

#### `pinneapple_app.backend.routers.problems`

Problems API router.

<small>`pinneapple_app/backend/routers/problems.py` · 86 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PresetListItem` | classe | `(name, family, description, tags, time_dependent, recommended_models, recommended_solver)` |  |
| `CustomProblemRequest` | classe | `(name, equations, boundary_conditions, initial_conditions, domain_bounds, dim, pde_family…)` |  |
| `list_problems` | função | `()` | Return all available preset problems with metadata. |
| `get_problem` | função | `(name)` | Return metadata for a single preset problem. |
| `validate_custom_problem` | função | `(req)` | Validate a custom problem definition without running it. |

### Exemplos de uso reais do repositório

_Nenhum exemplo encontrado no repositório para este pacote._

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 7 de 18.
- Funções de teste em arquivos que importam algum módulo do pacote: 48.

### Limitações conhecidas

- _Nenhuma registrada._
