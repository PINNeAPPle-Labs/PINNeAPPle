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
| `benchmark_suite` |  | 52 | 118 | 155 |
| `compute_backends` | Seletor global de backend (PyTorch por padrão, JAX opcional) e utilitários JAX de jit/vmap para resíduos de PINN. | 3 | 8 | 156 |
| `dataset_quality` | Checagens estatísticas de qualidade de datasets em nuvem de pontos e curadoria de sequências com um modelo de visão-linguagem (VLM). | 3 | 16 | 16 |
| `hpo_experiments` |  | 27 | 44 | 0 |
| `model_evaluation` | Métricas genéricas de avaliação de modelos (regressão, bootstrap, sensibilidade, calibração) e os gráficos correspondentes em matplotlib. | 3 | 12 | 5 |
| `model_export` | Exportação de modelos para TorchScript e ONNX e de campos (coordenadas + valores) para CSV e NPZ. | 1 | 4 | 0 |
| `model_services` | Modelos do PINNeAPPle servidos como microserviços HTTP com as rotas do NVIDIA NIM, cada resposta com relatório de confiança; inclui cliente e servidor. | 6 | 17 | 39 |
| `nvidia` |  | 11 | 49 | 39 |
| `rea_bridge` | Ponte para a ferramenta REA (engenharia reversa) com trava de licença, mapa de funcionalidades e livro de registro do que foi estudado. | 6 | 20 | 22 |
| `reporting` | Estilo de figuras e construtores de relatório técnico: PDF com reportlab (`paperkit`) e datasheet HTML/PDF/Markdown (`datasheet`). | 3 | 14 | 14 |
| `sandbox` | Executa um script de solver fornecido pelo usuário em outro processo com limites de recurso POSIX; não é uma fronteira de segurança. | 2 | 1 | 13 |
| `surrogate_policy` | Decide, pelo tempo medido de uma solução real, se um problema precisa de surrogate treinado ou se o solver pode ser chamado direto. | 1 | 2 | 5 |
| `visualization` |  | 35 | 165 | 46 |

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

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/benchmark_suite/__init__.py` · 108 linhas · testes: 2</small>

#### `pinneapple_tools.benchmark_suite.api`

Arena — high-level API for running PINNeAPPle resolution pipelines.

<small>`pinneapple_tools/benchmark_suite/api.py` · 591 linhas · testes: 153</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ArenaResult` | classe | `(spec, model, history, metrics, artifacts, model_id, elapsed_s)` | Container returned by Arena.run() and Arena.compare(). |
| `ArenaCompareResult` | classe | `(spec, results)` | Container returned by Arena.compare() with multiple models. |
| `Arena` | classe | `(spec) -> None` | High-level runner that connects ``ProblemSpec`` to the training pipeline. |

#### `pinneapple_tools.benchmark_suite.backends`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/benchmark_suite/backends/__init__.py` · 17 linhas · **sem teste direto**</small>

#### `pinneapple_tools.benchmark_suite.backends.base`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/benchmark_suite/backends/base.py` · 20 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Backend` | classe | `(name)` | Training backend protocol. |

#### `pinneapple_tools.benchmark_suite.backends.deepxde_backend`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/benchmark_suite/backends/deepxde_backend.py` · 113 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DeepXDEBackend` | classe | `(name)` | DeepXDE backend (baseline supervisionado em sensores). |

#### `pinneapple_tools.benchmark_suite.backends.jax_pinn`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/benchmark_suite/backends/jax_pinn.py` · 335 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `JAXPINNBackend` | classe | `(name)` | Full PINN backend in JAX for FlowObstacle2D (steady 2D Navier–Stokes). |

#### `pinneapple_tools.benchmark_suite.backends.native_pinn`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/benchmark_suite/backends/native_pinn.py` · 287 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MLP` | classe | `(in_dim, out_dim, hidden, activation='tanh')` | Fully connected multilayer perceptron (MLP). |
| `NativePINNBackend` | classe | `()` | Native PINN training backend for steady 2D Navier-Stokes problems. |

#### `pinneapple_tools.benchmark_suite.backends.physicsnemo_sym`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/benchmark_suite/backends/physicsnemo_sym.py` · 240 linhas · testes: 2</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsNeMoSymBackend` | classe | `()` | Backend integration for PhysicsNeMo Sym (formerly Modulus Sym). |

#### `pinneapple_tools.benchmark_suite.backends.pinneapple_models_backend`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/benchmark_suite/backends/pinneapple_models_backend.py` · 200 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PinneappleModelsBackend` | classe | `()` | Train any model from ModelRegistry. |

#### `pinneapple_tools.benchmark_suite.benchmark`

PINN Arena — multi-architecture, multi-problem benchmark suite.

<small>`pinneapple_tools/benchmark_suite/benchmark.py` · 1.309 linhas · testes: 155 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BenchmarkConfig` | classe | `(n_col, n_bc, n_ic, epochs, lr, weight_pde, weight_bc, weight_ic, device, seed, n_eval, c…)` |  |
| `ModelSpec` | classe | `(name, factory, description, input_kind, family)` |  |
| `BenchmarkResult` | classe | `(problem_id, model_id, metrics, history, elapsed_s, n_params, rank)` |  |
| `BenchmarkTaskBase` | classe | `(task_id, in_dim, out_dim)` | Abstract base for physics benchmark tasks. |
| `NotApplicableError` | classe | `()` | Raised (by callers, not by this module) when a task can't build a batch for a given input_kind. ``run()`` doesn't raise this itself — it relies on ``task.build_batch()`` returning ``None`` — but callers that pre-check applicability (e.g. BiaML's ``run_benchmark()``) can use this to signal "not … |
| `PINNArenaBenchmark` | classe | `(tasks, model_specs=None, config=None) -> None` | Multi-problem, multi-architecture PINN benchmark suite. |
| `all_model_specs` | função | `() -> List[ModelSpec]` | Return a ModelSpec for every model in ``pinneapple_models``, tagged with its real ``input_kind`` (pointwise_coords / sequence / graph / grid* / dynamics / operator_branch_trunk / autoencoder — see ``pinneapple_neural.architectures.registry.ModelSpec.input_kind``, now accurately declared per family … |

#### `pinneapple_tools.benchmark_suite.bundle`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/benchmark_suite/bundle/__init__.py` · 7 linhas · **sem teste direto**</small>

#### `pinneapple_tools.benchmark_suite.bundle.loader`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/benchmark_suite/bundle/loader.py` · 149 linhas · testes: 2</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BundleData` | classe | `(root, manifest, conditions, points_collocation, points_boundary, sensors)` | Immutable container for a validated PINNeAPPle bundle. |
| `load_bundle` | função | `(bundle_root, *, schema, require_sensors=False) -> BundleData` | Load and validate a PINNeAPPle bundle from disk. |

#### `pinneapple_tools.benchmark_suite.bundle.schema`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/benchmark_suite/bundle/schema.py` · 119 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BundleSchema` | classe | `(schema_version, required_files, required_regions, manifest_required_keys, sensors_requir…)` | Schema definition for validating a PINNeAPPle bundle. |
| `load_bundle_schema` | função | `(path) -> BundleSchema` | Load a bundle schema definition from a YAML configuration file. |

#### `pinneapple_tools.benchmark_suite.crossframework`

Neutral parts of the cross-framework benchmark (PINNeAPPle vs PhysicsNeMo / PhysicsNeMo-Sym, DeepXDE, neuraloperator): problem definitions and references, the shared point sets and initial weights, the one evaluator, statistics over seeds, run records and the operator-learning data and training …

<small>`pinneapple_tools/benchmark_suite/crossframework/__init__.py` · 17 linhas · testes: 20</small>

#### `pinneapple_tools.benchmark_suite.crossframework.darcy`

Track B of the cross-framework benchmark: the Darcy data, its normalisation and the one training loop.

<small>`pinneapple_tools/benchmark_suite/crossframework/darcy.py` · 182 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DarcyData` | classe | `(a, u, a_mean, a_std, u_mean, u_std)` | The three splits and the training-split normalisation. |
| `load_split` | função | `(split, resolution=64) -> tuple[np.ndarray, np.ndarray]` | ``(a, u)`` of ``split`` in ``("train", "val", "test")`` at ``resolution`` (float32, (n, r, r)). |
| `load_darcy` | função | `(resolution=64, n_train=N_TRAIN) -> DarcyData` |  |
| `train_operator` | função | `(model, data, *, epochs, seed, batch_size=20, lr=0.001, weight_decay=0.0001, log=None) ->…)` | Train ``model`` ((B, 1, r, r) -> (B, 1, r, r), normalised) with the protocol's loop. Returns the history (loss per epoch, wall and CPU time per epoch) and the decoded predictions on the validation and test splits. |
| `run_track_b` | função | `(argv, *, framework, source, build, model_config) -> None` | Shared body of the three track-B runners: the runner only says how its library builds the FNO. |
| `inference_latency` | função | `(model, data, batch, repeats=20, warmup=3) -> float` | Median wall time [s] of one forward pass on ``batch`` test inputs (no gradients). |

#### `pinneapple_tools.benchmark_suite.crossframework.evaluate`

The one evaluator of the cross-framework benchmark, and the statistics over seeds.

<small>`pinneapple_tools/benchmark_suite/crossframework/evaluate.py` · 141 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `evaluate_pinn` | função | `(problem, u_test, u_stencil=None, *, reference=None) -> dict[str, float]` | Metrics of a forward run from its exported predictions. |
| `evaluate_inverse` | função | `(nu_hat, nu_true, u_pred, u_ref) -> dict[str, float]` | Inverse run: relative error of the identified coefficient and the field error on the test points. |
| `per_sample_rel_l2` | função | `(pred, ref) -> np.ndarray` | Per-sample relative L2 over all axes but the first (operator learning). |
| `evaluate_operator` | função | `(pred, ref) -> dict[str, float]` | Operator run: mean, median and max of the per-sample relative L2 on the test split. |
| `seed_stats` | função | `(values) -> dict[str, object]` | Median, min, max and n of the finite values; ``n_failed`` counts NaN / inf (failures are not dropped silently: they are counted, and the median is over the finite values only). |
| `paired_ratio` | função | `(a, b) -> dict[str, object]` | Per-seed ratio ``a / b`` over the seeds both have (finite): median, min, max, and how many seeds ``a < b``. |
| `sign_test_p` | função | `(k, n) -> float` | Two-sided exact sign test p-value for ``k`` successes in ``n`` paired trials (ties excluded beforehand). With n = 5 the smallest attainable p is 0.0625: five seeds cannot show significance at 0.05 by this test. |
| `summarize` | função | `(rows, keys, by) -> list[dict[str, object]]` | Group ``rows`` by the ``by`` fields and give :func:`seed_stats` of each metric in ``keys``. |

#### `pinneapple_tools.benchmark_suite.crossframework.inverse`

Track C of the cross-framework benchmark: identify the viscosity of Burgers from noisy samples.

<small>`pinneapple_tools/benchmark_suite/crossframework/inverse.py` · 28 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `observations` | função | `(n=C['n_obs'], noise_rel=C['noise_rel'], seed=C['obs_seed']) -> dict[str, np.ndarray]` | ``{"X", "u", "u_clean", "sigma"}``: points (n, 2) as (x, t), noisy and clean values (n, 1), float32. |

#### `pinneapple_tools.benchmark_suite.crossframework.problems`

The forward and inverse PINN problems of the cross-framework benchmark, with their references.

<small>`pinneapple_tools/benchmark_suite/crossframework/problems.py` · 314 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PINNProblem` | classe | `(name, coords, bounds, params, time_dependent, fd_step, citation, test_grid, extra)` | One benchmark problem. ``coords`` are ``("x", "t")`` or ``("x", "y")``; ``bounds`` one ``(lo, hi)`` each. |
| `get_problem` | função | `(name) -> PINNProblem` |  |
| `make_point_set` | função | `(problem, *, n_interior=10000, n_boundary=400, seed=20260) -> dict[str, np.ndarray]` | The training points every framework loads (protocol A1): ``X_f`` interior collocation points (uniform), ``X_b``/``u_b`` boundary points and values, and for time-dependent problems ``X_0``/``u_0`` initial points. |
| `point_set_digest` | função | `(points) -> str` | SHA-256 over the arrays (sorted by name) so every run can record which point set it trained on. |
| `save_point_set` | função | `(points, path) -> str` | Write the point set to ``path`` (.npz) and return its digest. |
| `load_point_set` | função | `(path) -> dict[str, np.ndarray]` |  |
| `test_points` | função | `(problem) -> np.ndarray` | The structured test grid (float64, N x 2): the same points for every framework. |
| `residual_points` | função | `(problem, n=2000, seed=777) -> np.ndarray` | Fresh interior points (never used in training; a different generator stream) for the residual, kept one stencil step away from the boundary. |
| `stencil` | função | `(problem, X) -> np.ndarray` | The 5-point stencil around each residual point: (N, 5, 2) = center, a+h, a-h, b+h, b-h. A runner predicts on ``stencil(...).reshape(-1, 2)`` and exports the values; :func:`residual_from_stencil` turns them into the residual. |
| `residual_from_stencil` | função | `(problem, X, values) -> np.ndarray` | Residual at ``X`` from the stencil values (N, 5) by second-order central differences (float64). |
| `allen_cahn_reference` | função | `(d=0.0001, n=2048, dt=0.0001, t_end=1.0, n_save=1000) -> tuple[np.ndarray, np.ndarray, np…)` | ``(x, t, u)`` of ``u_t = d u_xx + 5 u - 5 u^3``, periodic on ``[-1, 1)``, ``u(x, 0) = x^2 cos(pi x)``. |
| `callable_exact` | função | `(problem) -> Callable[[np.ndarray], np.ndarray]` | ``X -> u`` column vector, for APIs that take a reference callable (``pp.compare(..., reference="exact")``). |
| `describe` | função | `(problem) -> dict[str, object]` | JSON-ready description (stored in every result file). |

#### `pinneapple_tools.benchmark_suite.crossframework.protocol`

Fixed settings of the cross-framework benchmark protocol and the export every runner produces.

<small>`pinneapple_tools/benchmark_suite/crossframework/protocol.py` · 97 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `export_predictions` | função | `(problem, predict, batch=20000) -> dict[str, np.ndarray]` | ``{"u_test", "u_stencil"}`` from ``predict`` ((N, 2) float32 array -> (N,) or (N, 1)), evaluated in batches on the shared test grid and on the residual stencil. Every runner exports exactly this. |
| `ensure_point_set` | função | `(problem, root) -> Path` | The A1 point set of ``problem`` under ``root/points/`` (created once; later calls load the same file). |
| `runner_parser` | função | `(description)` | The command line every runner accepts (the driver calls runners only through it). |
| `base_meta` | função | `(args, *, track, framework, source, config) -> dict` | Metadata every result starts from (the runner fills status, iterations and times). |

#### `pinneapple_tools.benchmark_suite.crossframework.records`

Run records of the cross-framework benchmark: environment, threads, load, time caps, the result file schema.

<small>`pinneapple_tools/benchmark_suite/crossframework/records.py` · 166 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Clock` | classe | `(wall_s, cpu_s, loadavg_before, loadavg_after)` | Wall and process CPU time of a block, with the load average before and after. |
| `TimeCap` | classe | `(seconds, kind='cpu')` | A per-run time budget, measured as process CPU time (``kind="cpu"``) or wall time (``kind="wall"``). |
| `setup_threads` | função | `(n) -> None` | Pin the thread count of this process (call before importing torch / a framework): OMP / MKL / OpenBLAS environment variables and ``torch.set_num_threads``. |
| `loadavg` | função | `() -> list` |  |
| `cpu_model` | função | `() -> str` |  |
| `package_versions` | função | `(names) -> dict[str, str \| None]` |  |
| `environment` | função | `(threads) -> dict[str, Any]` | Machine and library record stored next to the results. |
| `write_run` | função | `(out_dir, run_id, meta, arrays) -> Path` | Write ``<run_id>.json`` (metadata) and ``<run_id>.npz`` (exported predictions) into ``out_dir``. |
| `read_run` | função | `(out_dir, run_id)` |  |
| `validate_result` | função | `(meta) -> list` | Missing schema fields of a result (empty when valid). |

#### `pinneapple_tools.benchmark_suite.crossframework.weights`

Shared initial weights for the identical-configuration track (A1).

<small>`pinneapple_tools/benchmark_suite/crossframework/weights.py` · 48 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `layer_sizes` | função | `(in_dim=2, out_dim=1, width=50, depth=4) -> list[int]` |  |
| `xavier_uniform_init` | função | `(sizes, seed) -> list[tuple[np.ndarray, np.ndarray]]` | ``[(W, b), ...]`` with ``W`` of shape ``(fan_out, fan_in)`` (the ``torch.nn.Linear`` layout) drawn from ``U(-a, a)``, ``a = sqrt(6 / (fan_in + fan_out))`` (Glorot & Bengio 2010), and ``b = 0``; float32. |
| `load_into_linears` | função | `(linears, params) -> None` | Copy ``params`` into a sequence of ``torch.nn.Linear`` modules (shapes must match exactly). |
| `count_parameters` | função | `(module) -> int` |  |

#### `pinneapple_tools.benchmark_suite.deeponet_ground_truth`

The four canonical operator-learning ground-truth solvers from Lu, Jin & Karniadakis 2019 (DeepONet), arXiv:1910.03193, Section 4: given a GRF-sampled input function u(x) (`pinneapple_data.grf_sampler.GRFDraw`), each solver computes the TRUE solution of a different dynamical system driven by u. …

<small>`pinneapple_tools/benchmark_suite/deeponet_ground_truth.py` · 155 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `solve_ode_linear` | função | `(draw, x_query) -> np.ndarray` | Exact via the GRF spline's own antiderivative (piecewise-cubic, so integration is exact for the spline itself, not an ODE-solver approximation of it). |
| `solve_ode_nonlinear` | função | `(draw, x_query) -> np.ndarray` |  |
| `solve_pendulum` | função | `(draw, x_query, k=1.0) -> np.ndarray` | Returns s1 (angle) at each query point. |
| `pendulum_rk4_reference` | função | `(draw, x_max, k, n_steps=20000) -> float` | Independent hand-written fixed-step RK4 integrator, useful ONLY to cross-check solve_ivp's adaptive integration in a validation script -- not meant for production use (solve_pendulum's adaptive RK45 is both faster and, by default tolerance, at least as accurate). |
| `solve_diffusion_reaction` | função | `(draw, nx=100, nt=100, D=0.01, k=0.01, x_max=1.0, t_max=1.0) -> 'tuple[np.ndarray, np.nda…)` | Crank-Nicolson for the diffusion term + one Picard correction per step for the k*s^2 reaction term, tridiagonal (Thomas algorithm) solves. Returns (x_grid, t_grid, s) with s.shape == (nt, nx). |

#### `pinneapple_tools.benchmark_suite.io`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/benchmark_suite/io/__init__.py` · 6 linhas · **sem teste direto**</small>

#### `pinneapple_tools.benchmark_suite.io.yamlx`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/benchmark_suite/io/yamlx.py` · 56 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `load_yaml` | função | `(path) -> Dict[str, Any]` | Load a YAML file and return its content as a dictionary. |
| `dump_yaml` | função | `(obj) -> str` | Serialize a Python object into a YAML-formatted string. |

#### `pinneapple_tools.benchmark_suite.meta_benchmark`

Meta-Learning Benchmark for PINNs.

<small>`pinneapple_tools/benchmark_suite/meta_benchmark.py` · 937 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MetaBenchmarkConfig` | classe | `(n_meta_epochs, n_inner_steps, inner_lr, outer_lr, n_tasks_per_batch, k_shots, algorithms…)` |  |
| `MetaBenchmarkFamily` | classe | `(name, train_task_factory, eval_task_factory, param_ranges, eval_params)` | A parametric PDE family for meta-learning. |
| `MetaBenchmarkResult` | classe | `(family_name, eval_params, algorithm, k_shot, metrics, elapsed_s)` |  |
| `MetaBenchmarkPipeline` | classe | `(families, config=None) -> None` | Meta-learning benchmark pipeline. |
| `cfg_str` | função | `(results) -> str` |  |

#### `pinneapple_tools.benchmark_suite.physics_pipeline`

PhysicsPipeline — standard 6-step PINNeAPPle physics pipeline.

<small>`pinneapple_tools/benchmark_suite/physics_pipeline.py` · 1.332 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsPipeline` | classe | `(problem, geometry=None, load_generate_data='generate', source=None, metrics='auto', coll…)` | Standard 6-step PINNeAPPle physics pipeline. |

#### `pinneapple_tools.benchmark_suite.pipeline.dataset_builder`

Arena dataset builder — thin re-export from pinneapple_data.adapters.pinn_batch_builders.

<small>`pinneapple_tools/benchmark_suite/pipeline/dataset_builder.py` · 22 linhas · **sem teste direto**</small>

#### `pinneapple_tools.benchmark_suite.pipeline.model_selector`

Model selection helpers.

<small>`pinneapple_tools/benchmark_suite/pipeline/model_selector.py` · 39 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `select_models` | função | `(*, input_kind, expects=None, supports_physics_loss=None, tags_any=None) -> List[str]` |  |

#### `pinneapple_tools.benchmark_suite.registry`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/benchmark_suite/registry.py` · 77 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `register_task` | função | `(task_cls) -> Type['ArenaTask']` | Decorator to register a task by its ``task_id``. |
| `get_task` | função | `(task_id) -> Type['ArenaTask']` |  |
| `list_tasks` | função | `() -> List[str]` |  |
| `register_backend` | função | `(backend_cls) -> Type['Backend']` | Decorator to register a backend by its ``name``. |
| `get_backend` | função | `(name) -> Type['Backend']` |  |
| `list_backends` | função | `() -> List[str]` |  |

#### `pinneapple_tools.benchmark_suite.report`

Benchmark report: structured output for PhysicsBenchmarkSpec and TimeSeriesBenchmarkSpec.

<small>`pinneapple_tools/benchmark_suite/report.py` · 148 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ModelRunResult` | classe | `(model_id, n_params, training_time_s, metrics, history, param_estimates, rank, error_mess…)` |  |
| `BenchmarkReport` | classe | `(benchmark_type, created_at, version, problem_info, config, model_results, leaderboard, b…)` |  |

#### `pinneapple_tools.benchmark_suite.runner`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/benchmark_suite/runner/__init__.py` · 25 linhas · **sem teste direto**</small>

#### `pinneapple_tools.benchmark_suite.runner.compare`

Compare multiple trained model runs.

<small>`pinneapple_tools/benchmark_suite/runner/compare.py` · 151 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `compare_runs` | função | `(run_dirs, *, out_dir='runs/arena_compare', title='arena_compare', make_plots=True) -> Di…)` | Compare completed run directories. |

#### `pinneapple_tools.benchmark_suite.runner.leaderboard`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/benchmark_suite/runner/leaderboard.py` · 43 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `load_leaderboard` | função | `(path) -> List[Dict[str, Any]]` | Load a leaderboard JSON file. |
| `update_leaderboard` | função | `(path, row) -> None` | Append a new entry to the leaderboard JSON file. |

#### `pinneapple_tools.benchmark_suite.runner.metrics`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/benchmark_suite/runner/metrics.py` · 30 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ensure_float_dict` | função | `(d) -> Dict[str, float]` | Convert dictionary values to floats when possible. |

#### `pinneapple_tools.benchmark_suite.runner.report`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/benchmark_suite/runner/report.py` · 67 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `write_run_artifacts` | função | `(*, artifacts_dir, run_id, report, metrics, summary) -> Path` | Persist run artifacts (report, metrics, summary) to disk. |

#### `pinneapple_tools.benchmark_suite.runner.run_arena_yaml`

Run a complete Arena experiment from a unified YAML config.

<small>`pinneapple_tools/benchmark_suite/runner/run_arena_yaml.py` · 563 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `run_arena_experiment` | função | `(yaml_path, *, out_dir=None) -> Dict[str, Any]` | Run a complete Arena experiment from a YAML config file. |
| `main` | função | `() -> None` |  |

#### `pinneapple_tools.benchmark_suite.runner.run_benchmark`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/benchmark_suite/runner/run_benchmark.py` · 132 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `run_benchmark` | função | `(*, artifacts_dir, task_cfg_path, run_cfg_path, bundle_schema_path) -> Dict[str, Any]` |  |

#### `pinneapple_tools.benchmark_suite.runner.run_models_from_cfg`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/benchmark_suite/runner/run_models_from_cfg.py` · 265 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `run_one` | função | `(*, run_cfg, run_dir, bundle_root=None) -> Dict[str, Any]` |  |
| `main` | função | `()` |  |

#### `pinneapple_tools.benchmark_suite.runner.run_pipeline`

End-to-end pipeline: geometry → solver → dataset → train → predict → report.

<small>`pinneapple_tools/benchmark_suite/runner/run_pipeline.py` · 1.444 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `run_full_pipeline` | função | `(config_path_or_dict, *, out_dir=None) -> Dict[str, Any]` | Run the complete end-to-end pipeline. |
| `main` | função | `() -> None` |  |

#### `pinneapple_tools.benchmark_suite.runner.run_sweep`

Run a benchmark sweep over multiple models.

<small>`pinneapple_tools/benchmark_suite/runner/run_sweep.py` · 208 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SweepJob` | classe | `(model_name, run_cfg)` |  |
| `run_sweep` | função | `(*, models, base_run_cfg, bundle_root, out_dir='runs/arena_sweep', parallelism='process',…)` | Run a sweep. |

#### `pinneapple_tools.benchmark_suite.tasks`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/benchmark_suite/tasks/__init__.py` · 29 linhas · testes: 2</small>

#### `pinneapple_tools.benchmark_suite.tasks.allen_cahn_1d`

Allen-Cahn 1D benchmark task.

<small>`pinneapple_tools/benchmark_suite/tasks/allen_cahn_1d.py` · 135 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AllenCahn1DTask` | classe | `(eps=0.01, build_reference=True) -> None` |  |

#### `pinneapple_tools.benchmark_suite.tasks.base`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/benchmark_suite/tasks/base.py` · 53 linhas · testes: 2</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TaskResult` | classe | `(metrics, artifacts)` | Standard output for a task evaluation. |
| `ArenaTask` | classe | `(task_id)` | Base class for Arena tasks. |

#### `pinneapple_tools.benchmark_suite.tasks.burgers_1d`

Burgers 1D benchmark task.

<small>`pinneapple_tools/benchmark_suite/tasks/burgers_1d.py` · 152 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Burgers1DTask` | classe | `(nu=0.01 / np.pi, build_reference=True) -> None` |  |

#### `pinneapple_tools.benchmark_suite.tasks.flow_obstacle_2d`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/benchmark_suite/tasks/flow_obstacle_2d.py` · 263 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FlowObstacle2DTask` | classe | `(task_id, n_collocation_eval, n_boundary_eval, n_sensors_eval)` | Task definition for the 2D steady flow around obstacle benchmark. |

#### `pinneapple_tools.benchmark_suite.tasks.heat_1d`

Heat equation 1D benchmark task.

<small>`pinneapple_tools/benchmark_suite/tasks/heat_1d.py` · 85 linhas · testes: 153</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Heat1DTask` | classe | `(alpha=0.01) -> None` |  |

#### `pinneapple_tools.benchmark_suite.tasks.lid_driven_cavity_3d`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/benchmark_suite/tasks/lid_driven_cavity_3d.py` · 394 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LidDrivenCavity3DTask` | classe | `(Re=100.0, size=1.0, lid_velocity=1.0) -> None` | 3D Lid-Driven Cavity Flow benchmark. |

#### `pinneapple_tools.benchmark_suite.tasks.navier_stokes_tgv_2d`

Navier-Stokes 2D Taylor-Green Vortex benchmark task.

<small>`pinneapple_tools/benchmark_suite/tasks/navier_stokes_tgv_2d.py` · 183 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NavierStokesTGV2DTask` | classe | `(nu=0.01) -> None` |  |

#### `pinneapple_tools.benchmark_suite.tasks.poisson_2d`

Poisson 2D benchmark task.

<small>`pinneapple_tools/benchmark_suite/tasks/poisson_2d.py` · 98 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Poisson2DTask` | classe | `()` |  |

#### `pinneapple_tools.benchmark_suite.tasks.wave_1d`

Wave equation 1D benchmark task.

<small>`pinneapple_tools/benchmark_suite/tasks/wave_1d.py` · 103 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Wave1DTask` | classe | `(c=1.0) -> None` |  |

#### `pinneapple_tools.benchmark_suite.timeseries_pipeline`

TimeSeriesBenchmarkSpec — declarative time-series benchmark pipeline.

<small>`pinneapple_tools/benchmark_suite/timeseries_pipeline.py` · 579 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TimeSeriesBenchmarkSpec` | classe | `(source, models=('lstm',), metrics=('mse', 'mae', 'rmse'), target_cols=None, horizon=50, …)` | Declarative time-series benchmark pipeline. |

#### `pinneapple_tools.benchmark_suite.transfer_benchmark`

Transfer Learning Benchmark for PINNs.

<small>`pinneapple_tools/benchmark_suite/transfer_benchmark.py` · 798 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TransferBenchmarkConfig` | classe | `(n_source_epochs, lr_source, n_finetune_epochs, lr_finetune, lr_base, lr_scratch, strateg…)` |  |
| `TransferScenario` | classe | `(name, source_task, target_tasks, source_label, target_labels)` | One transfer scenario: one source task and N target tasks. |
| `TransferBenchmarkResult` | classe | `(scenario_name, target_label, strategy, metrics, history, elapsed_s, n_params_trainable, …)` |  |
| `TransferBenchmarkPipeline` | classe | `(scenarios, config=None) -> None` | Benchmark comparing transfer learning strategies against from-scratch baselines. |

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

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/hpo_experiments/__init__.py` · 14 linhas · **sem teste direto**</small>

#### `pinneapple_tools.hpo_experiments.agents`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/hpo_experiments/agents/__init__.py` · 6 linhas · **sem teste direto**</small>

#### `pinneapple_tools.hpo_experiments.agents.extractor`

ExtractorAgent for extracting problem/solution from KB artifacts.

<small>`pinneapple_tools/hpo_experiments/agents/extractor.py` · 101 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ExtractorAgent` | classe | `(provider)` |  |

#### `pinneapple_tools.hpo_experiments.agents.prompts`

Agent system prompts for extract, reproduce, and verify.

<small>`pinneapple_tools/hpo_experiments/agents/prompts.py` · 125 linhas · **sem teste direto**</small>

#### `pinneapple_tools.hpo_experiments.agents.reproducer`

ReproducerAgent for generating reproduction patches from repo/project.

<small>`pinneapple_tools/hpo_experiments/agents/reproducer.py` · 98 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ReproduceResult` | classe | `(project_dir, run_instructions)` |  |
| `ReproducerAgent` | classe | `(provider)` |  |

#### `pinneapple_tools.hpo_experiments.agents.verifier`

VerifierAgent for verifying project runnability and suggesting patches.

<small>`pinneapple_tools/hpo_experiments/agents/verifier.py` · 75 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `VerifyResult` | classe | `(status, diagnosis, patches, notes)` |  |
| `VerifierAgent` | classe | `(provider)` |  |

#### `pinneapple_tools.hpo_experiments.config`

ResearcherConfig for topic, k_papers, k_repos, ranking weights.

<small>`pinneapple_tools/hpo_experiments/config.py` · 42 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ResearcherConfig` | classe | `(topic, k_papers, k_repos, w_lexical, w_signals, w_freshness, w_quality, github_min_stars…)` | Researcher configuration. |

#### `pinneapple_tools.hpo_experiments.kb`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/hpo_experiments/kb/__init__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_tools.hpo_experiments.kb.chunking`

Text chunking for KB indexing with overlap.

<small>`pinneapple_tools/hpo_experiments/kb/chunking.py` · 29 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TextChunk` | classe | `(chunk_id, text)` |  |
| `chunk_text` | função | `(text, *, chunk_chars=1800, overlap=200) -> List[TextChunk]` |  |

#### `pinneapple_tools.hpo_experiments.kb.store`

KB store for artifacts, indexing, and persistence.

<small>`pinneapple_tools/hpo_experiments/kb/store.py` · 142 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `KBStore` | classe | `(*, out_dir)` | Persists a KB run: - manifest.json (ranked items + metadata) - artifacts (papers/, repos/) - chunks.jsonl (chunked text for RAG) |
| `slugify` | função | `(s) -> str` |  |
| `ensure_dir` | função | `(p) -> str` |  |

#### `pinneapple_tools.hpo_experiments.models`

RankedItem, DiscoveryResult data models for researcher output.

<small>`pinneapple_tools/hpo_experiments/models.py` · 75 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RankedItem` | classe | `(type, id, title, url, score, why, meta)` |  |
| `DiscoveryResult` | classe | `(topic, papers, repos, meta)` |  |
| `KBArtifact` | classe | `(type, id, title, url, path, meta)` | A persisted artifact (paper/repo) stored inside a KB run folder. |
| `KBIndex` | classe | `(topic, run_dir, chunks_path, manifest_path)` | A light-weight index pointing to chunked documents (JSONL). |
| `ExtractedProblemSolution` | classe | `(source_type, source_id, title, problem, solution, equations, data_requirements, training…)` | Output of ExtractorAgent. Keep it generic but structured enough to map into pinneapple_problemdesign later. |

#### `pinneapple_tools.hpo_experiments.pipelines`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/hpo_experiments/pipelines/__init__.py` · 8 linhas · **sem teste direto**</small>

#### `pinneapple_tools.hpo_experiments.pipelines.build_kb`

KB build pipeline: papers and repos to KBStore.

<small>`pinneapple_tools/hpo_experiments/pipelines/build_kb.py` · 141 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_kb` | função | `(cfg, discovery) -> KBIndex` |  |

#### `pinneapple_tools.hpo_experiments.pipelines.discover`

Discovery pipeline: papers and repos from topic query.

<small>`pinneapple_tools/hpo_experiments/pipelines/discover.py` · 201 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `discover` | função | `(cfg) -> DiscoveryResult` |  |

#### `pinneapple_tools.hpo_experiments.pipelines.export_problemdesign`

Export extracted solutions to ProblemDesign format.

<small>`pinneapple_tools/hpo_experiments/pipelines/export_problemdesign.py` · 169 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `to_problemdesign_dict` | função | `(x) -> Dict[str, Any]` | ProblemDesign-like schema (stable, explicit): - source - problem - solution - benchmark - strengths - weaknesses - alternatives - future_improvements |
| `export_problemdesign` | função | `(items, *, out_path) -> str` | Write researcher findings to *out_path* (JSON) and, when ``pinneapple_problemdesign`` is available, also write a companion ``*_pinneapple_spec.json`` with partial ``ProblemSpec`` mappings and generated Pinneapple API objects for any item that has enough physics info. |

#### `pinneapple_tools.hpo_experiments.pipelines.extract_problem_solutions`

Extract problem/solution pairs from KB via ExtractorAgent.

<small>`pinneapple_tools/hpo_experiments/pipelines/extract_problem_solutions.py` · 39 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `extract_problem_solutions` | função | `(*, kb_index, out_dir=None, max_items=20) -> List[ExtractedProblemSolution]` |  |

#### `pinneapple_tools.hpo_experiments.pipelines.reproduce`

Reproduce pipeline: apply patches and verify project runnability.

<small>`pinneapple_tools/hpo_experiments/pipelines/reproduce.py` · 79 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `reproduce` | função | `(*, item, kb_index_dir, out_dir=None, max_fix_iters=2) -> str` |  |

#### `pinneapple_tools.hpo_experiments.providers`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/hpo_experiments/providers/__init__.py` · 4 linhas · **sem teste direto**</small>

#### `pinneapple_tools.hpo_experiments.providers.gemini_provider`

Gemini API provider for LLM calls.

<small>`pinneapple_tools/hpo_experiments/providers/gemini_provider.py` · 240 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GeminiProviderConfig` | classe | `(api_base, api_version, model, api_key, timeout_s, gateway_base_url, gateway_route)` |  |
| `GeminiProvider` | classe | `(*, api_key=None, model=None, api_base=None, api_version=None, timeout_s=None, gateway_ba…)` | Default: calls Gemini Developer API directly (models.generateContent). |

#### `pinneapple_tools.hpo_experiments.ranking`

Ranking heuristics: lexical overlap, freshness, quality scores.

<small>`pinneapple_tools/hpo_experiments/ranking.py` · 71 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `tokenize` | função | `(text) -> List[str]` |  |
| `lexical_overlap_score` | função | `(query, doc) -> float` | v0.1 lexical overlap score (0..1). Simple: fraction of unique query tokens present in doc tokens. |
| `safe_parse_iso8601` | função | `(dt) -> Optional[datetime]` |  |
| `freshness_score` | função | `(dt, *, half_life_days=365.0) -> float` | Exponential decay freshness score (0..1). score=1 now, ~0.5 at half_life_days, decays afterwards. |
| `normalize` | função | `(xs) -> List[float]` |  |
| `sort_ranked` | função | `(items) -> List[RankedItem]` |  |

#### `pinneapple_tools.hpo_experiments.sources`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/hpo_experiments/sources/__init__.py` · 12 linhas · **sem teste direto**</small>

#### `pinneapple_tools.hpo_experiments.sources.arxiv_pdf`

Arxiv PDF download and text extraction with section splitting.

<small>`pinneapple_tools/hpo_experiments/sources/arxiv_pdf.py` · 79 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `arxiv_pdf_url` | função | `(arxiv_id, use_export_subdomain=True) -> str` |  |
| `download_pdf` | função | `(url, out_path, timeout_s=60) -> str` |  |
| `extract_pdf_text` | função | `(pdf_path) -> str` |  |
| `split_sections` | função | `(text) -> Dict[str, str]` | Heuristic: find headings by keyword matches; slice between them. Works “ok” for many arXiv PDFs but not perfect (PDF layout can be messy). |

#### `pinneapple_tools.hpo_experiments.sources.arxiv_source`

Arxiv API source for paper search and metadata.

<small>`pinneapple_tools/hpo_experiments/sources/arxiv_source.py` · 90 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ArxivPaper` | classe | `(arxiv_id, title, summary, url, published, authors, categories)` |  |
| `ArxivSource` | classe | `()` | Minimal arXiv Atom API search. |

#### `pinneapple_tools.hpo_experiments.sources.github_fetcher`

GitHub API fetcher for repo file tree and content.

<small>`pinneapple_tools/hpo_experiments/sources/github_fetcher.py` · 179 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RepoFile` | classe | `(path, sha, size, url)` |  |
| `GithubFetcher` | classe | `(token=None, api_base='https://api.github.com')` | Fetches README, repo tree, and key files using GitHub REST API. |

#### `pinneapple_tools.hpo_experiments.sources.github_source`

GitHub API source for repository search.

<small>`pinneapple_tools/hpo_experiments/sources/github_source.py` · 90 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GithubRepo` | classe | `(full_name, title, url, description, stars, forks, language, updated_at, license)` |  |
| `GithubSource` | classe | `(token=None)` | GitHub Search API wrapper (repos discovery). Token optional but recommended (GITHUB_TOKEN). |

#### `pinneapple_tools.hpo_experiments.utils`

_Módulo sem docstring e sem texto._

<small>`pinneapple_tools/hpo_experiments/utils/__init__.py` · 10 linhas · **sem teste direto**</small>

#### `pinneapple_tools.hpo_experiments.utils.verify_runtime`

Runtime verification: project tree, py_compile, smoke run.

<small>`pinneapple_tools/hpo_experiments/utils/verify_runtime.py` · 74 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RunResult` | classe | `(ok, step, returncode, stdout, stderr)` |  |
| `project_tree` | função | `(cwd, max_files=400) -> str` |  |
| `py_compile_all` | função | `(cwd) -> RunResult` |  |
| `smoke_run` | função | `(cwd, timeout_s=180) -> RunResult` |  |

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

pinneapple_tools.nvidia -- PINNeAPPle with the NVIDIA stack (PhysicsNeMo, Omniverse / OpenUSD, Isaac Sim, Cosmos, NIM).

<small>`pinneapple_tools/nvidia/__init__.py` · 24 linhas · **sem teste direto**</small>

#### `pinneapple_tools.nvidia._brep_worker`

USD BRep worker: builds a part from a recipe and measures it. Standalone (usd_brep + numpy + stdlib only), so it runs in whichever interpreter has ``usd-brep`` (its wheel is CPython 3.12):

<small>`pinneapple_tools/nvidia/_brep_worker.py` · 157 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `main` | função | `(req_path, out_path)` |  |

#### `pinneapple_tools.nvidia.brep`

Exact geometry with NVIDIA USD BRep (the SMLib NURBS B-rep kernel, ``pip install usd-brep``), measured for PINNeAPPle: mass properties of the exact solid, curvature of every face, tessellations at several chord tolerances with their measured deviation from the exact surfaces, and B-rep USD export.

<small>`pinneapple_tools/nvidia/brep.py` · 247 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PartRecipe` | classe | `(ops, result)` |  |
| `BRepReport` | classe | `(raw, meshes, seconds)` |  |
| `brep_python` | função | `() -> str \| None` | The interpreter that has ``usd_brep``: this one, else ``$PINNEAPPLE_BREP_PYTHON`` (if it exists). |
| `brep_available` | função | `() -> bool` |  |
| `evaluate_part` | função | `(recipe, *, chord_tolerances=(0.001, 0.0001), density=1.0, angle_deg=25.0, max_edge_lengt…)` | Build ``recipe`` with USD BRep and measure it (see the module docstring). ``vertex_curvature``: the exact surface curvature at every vertex of the finest mesh (``rep.vertex_curvature``, 1/m) and the B-rep face each vertex lies on (``rep.vertex_face``). |
| `geometry_checks` | função | `(rep, *, element_size=None, points_per_radius=4.0, volume_rtol=0.001) -> list[dict[str, A…)` | Trust checks of the geometry handed to a solver or a surrogate (the finest tessellation): manifold solid, chord error below the tolerance asked, mesh volume and area against the exact solid, and (with ``element_size``) the element size against the smallest radius of curvature. |
| `planar_face_mesh` | função | `(V, F, *, axis=2, value, atol=1e-09) -> tuple[np.ndarray, np.ndarray]` | The triangles of a tessellation lying in the plane ``x[axis] = value`` (e.g. the top face of a plate), as a 2-D mesh ``(V2 (n, 2), F2 (m, 3))`` in the other two coordinates, renumbered. |
| `vertex_normals` | função | `(V, F) -> np.ndarray` | Area-weighted unit vertex normals (outward for a consistently oriented closed mesh). |
| `mesh_thickness` | função | `(V, F, *, normals=None, eps=1e-07, chunk=256) -> np.ndarray` | Wall thickness at every vertex of a closed mesh: distance along the inward normal to the first triangle hit (Moller-Trumbore ray casting); ``inf`` where the ray leaves without a hit. |
| `point_mesh_distance` | função | `(P, V, F, chunk=128) -> np.ndarray` | Distance from each point to the nearest triangle vertex or centroid (a cheap upper bound on the distance to the mesh, used to sample the deviation of a mesh from points of an exact surface). |

#### `pinneapple_tools.nvidia.diagram`

Architecture diagrams of a PINNeAPPle + NVIDIA pipeline, drawn by code (matplotlib only, no graphviz).

<small>`pinneapple_tools/nvidia/diagram.py` · 118 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Block` | classe | `(key, label, owner, col, row, ran)` |  |
| `architecture_diagram` | função | `(title, blocks, edges, path, *, subtitle='', width=2.6, height=1.0, dpi=150) -> str` | Draw ``blocks`` and ``edges`` (``(src, dst)`` or ``(src, dst, label)``) and save a PNG at ``path``. |

#### `pinneapple_tools.nvidia.figures`

Result figures shared by the NVIDIA examples: field panels, check tables, training curves.

<small>`pinneapple_tools/nvidia/figures.py` · 140 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `field_panels` | função | `(fields, titles, path, *, cmaps=None, extent=(0, 1, 0, 1), suptitle='', units=None, share…)` | ``fields[i][ix, iy]`` drawn with x to the right and y up; ``shared``: groups of panel indices on one range; ``ncols``: panels per row (default: one row). |
| `check_rows` | função | `(checks) -> list[dict[str, Any]]` | Checks of any supported kind as ``[{name, value, limit, passed, detail}]``. |
| `checks_figure` | função | `(checks, path, *, title='Physics checks', decision='', dpi=130) -> str` | Bar of value / limit per check (limit / value for a lower bound, ``"kind": "min"``; log scale; the dashed line is the limit). Checks without a limit are drawn as pass / fail markers. |
| `curves_figure` | função | `(panels, path, *, suptitle='', dpi=130) -> str` | ``panels``: ``[{"title", "xlabel", "ylabel", "series": [(label, x, y, style?)], "logy"?, "hlines"?}]``. |

#### `pinneapple_tools.nvidia.heatsink`

Plate-fin heat-sink designs evaluated two ways at once: the exact solid from NVIDIA USD BRep (volume, area, mass) and PINNeAPPle's closed-form thermal and hydraulic model (``pinneapple_physics.closed_form.plate_fin_heatsink``: Teertstra forced convection, fin efficiency, spreading resistance, …

<small>`pinneapple_tools/nvidia/heatsink.py` · 113 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Duty` | classe | `(power_w, t_ambient_c, air_velocity_m_s, source, base_t, length, width, material, root_fi…)` |  |
| `evaluate_design` | função | `(n_fins, fin_height, fin_t, duty=None, *, brep=True, chord_tolerances=()) -> dict` | One design: closed-form thermal / hydraulic result and (``brep``) the exact solid's volume, area, mass. |
| `sample_designs` | função | `(n, *, n_range=(6, 20), h_range=(0.015, 0.045), t_range=(0.001, 0.003), min_gap=0.002, wi…)` | Latin-hypercube designs inside the ranges with at least ``min_gap`` between fins. |
| `design_table` | função | `(designs, duty=None, *, brep=True) -> list[dict]` |  |
| `verify` | função | `(n_fins, fin_height, fin_t, duty=None, **kw) -> dict` | Closed form + exact solid for a design a surrogate proposed (the evaluation that decides). |
| `surrogate_service` | função | `(net, X_train, *, name=…, t_ambient=25.0)` | A heat-sink surrogate ``(n_fins, fin_height, fin_t) -> (T rise, dp, mass)`` as a PINNeAPPle Model Service. Inputs ``{"designs": [[n, H, t], ...]}``; the trust report flags designs outside the training box and outputs that break physics (a negative temperature rise, pressure drop or mass). |

#### `pinneapple_tools.nvidia.isaac`

Isaac Sim-ready robots in OpenUSD (``UsdPhysics``), their checks, and the Isaac Sim script that replays a motion.

<small>`pinneapple_tools/nvidia/isaac.py` · 239 linhas · testes: 39 · palavras-chave no código (heurística; o texto acima diz o status real): shim ou obsoleto</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `author_articulation` | função | `(stage, spec) -> str` | Write ``spec`` under ``/World/<name>``; returns that prim path. ``stage``: a ``FieldStage`` or ``Usd.Stage``. |
| `validate_physics` | função | `(path, *, expect_bodies=None, expect_joints=None) -> dict` | :func:`validate_stage` plus the ``UsdPhysics`` parse and the mass / joint checks a simulator relies on. |
| `isaac_replay_script` | função | `(usd_path, robot_prim, traj_npz, out_npz, *, mode='effort', physics_dt=1.0 / 600.0, headl…)` | Standalone Isaac Sim script (run with Isaac Sim's ``python.sh``). ``traj_npz`` holds ``t``, ``q`` [rad], ``tau`` [N m] (joint order of the USD articulation); ``mode="effort"`` applies ``tau`` open loop, ``"position"`` tracks ``q`` with the drives. Writes ``t``, ``q`` and ``effort`` … |
| `compare_replay` | função | `(npz, t, q, tau) -> dict` | Max tracking error [rad] and RMS effort error relative to the peak torque, Isaac Sim replay vs PINNeAPPle. |

#### `pinneapple_tools.nvidia.omniverse`

PINNeAPPle results as OpenUSD stages for NVIDIA Omniverse (authored and checked with ``usd-core``).

<small>`pinneapple_tools/nvidia/omniverse.py` · 646 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FieldStage` | classe | `(path, title='PINNeAPPle result', *, meters_per_unit=1.0, fps=24.0)` | A USD stage holding PINNeAPPle fields. ``save()`` writes it (``.usda`` text or ``.usd`` / ``.usdc``). |
| `require_pxr` | função | `()` | ``pxr`` (OpenUSD), with an install hint when it is missing. |
| `field_colors` | função | `(values, cmap='inferno', vrange=None) -> np.ndarray` | ``(n, 3)`` linear RGB of ``values`` through a matplotlib colour map (sRGB -> linear, as USD expects). |
| `grid_surface` | função | `(X, Y, Z=None) -> tuple[np.ndarray, np.ndarray]` | Triangles of a structured ``(nx, ny)`` grid of points: ``(V (nx*ny, 3), F (2(nx-1)(ny-1), 3))``. |
| `extrude_profile` | função | `(xy, length, n=2, axis=1) -> tuple[np.ndarray, np.ndarray]` | A closed 2-D profile ``(m, 2)`` (in the plane normal to ``axis``) extruded over ``[0, length]`` in ``n`` stations: the side wall as triangles. Vertex ``k * m + i`` is profile point ``i`` at station ``k``. |
| `tube` | função | `(path, radius, n_around=12) -> tuple[np.ndarray, np.ndarray]` | A tube of ``radius`` around a polyline ``path (p, 3)``: vertex ``k * n_around + j`` is ring ``k``. |
| `validate_stage` | função | `(path, *, required_primvars=None, compliance=True) -> dict[str, Any]` | Re-open a USD file and check it. Returns ``{"ok", "problems", "prims", "meshes", "curves", "points", "time_samples", "up_axis", "meters_per_unit", "compliance"}``. ``required_primvars``: prim path -> primvar names that must be there. |
| `usd_validators` | função | `(stage) -> dict[str, Any]` | Every validator OpenUSD registers (``pxr.UsdValidation``: stage metadata, composition, shading, geometry subsets, and the ``UsdPhysics`` rigid-body, collider, joint and articulation checkers) run on ``stage``, or the legacy ``UsdUtils.ComplianceChecker`` on builds without ``UsdValidation``. |
| `read_stage_geometry` | função | `(path, time=None, exclude=()) -> list[dict[str, Any]]` | Meshes and curves of a USD file as arrays: ``[{"path", "kind", "points", "faces" or "counts", "colors"}]`` (colours from ``displayColor`` at ``time``, a time code; default: the first sample). ``exclude``: prim path prefixes to leave out (e.g. a ground plane). |
| `render_stage_png` | função | `(path, png, *, time=None, elev=28, azim=-60, title='', colorbar=None, size=(7.5, 5.5), dp…)` | Draw the meshes and curves of a USD file (read back from disk) in their ``displayColor``; ``colorbar``: ``(label, lo, hi, cmap)``; ``flat``: axes scaled to the data in x, y and a squashed z (long, low scenes such as a pipeline). A file check, not an Omniverse RTX render. |
| `kit_render_script` | função | `(usd_path, png_path, *, frames=60) -> str` | Python to run inside Omniverse Kit / USD Composer (Window > Script Editor, or ``kit --exec``): opens the stage and writes an RTX viewport capture. Needs Omniverse with an RTX GPU; not runnable with ``usd-core`` alone. Save it next to the stage: the paths are resolved from the script's own folder. |
| `render_stage_gif` | função | `(path, gif, times, *, fps=6.0, label=None, **kw) -> str` | An animated GIF of a time-sampled stage: :func:`render_stage_png` at each time code in ``times``; ``label(t)`` gives each frame's title. |
| `rasterize_topdown` | função | `(path, H, W, *, bounds=None, time=None, exclude=(), background=(0.0, 0.0, 0.0), view='top…)` | An orthographic view of the meshes of a USD file, ``(H, W, 3)`` sRGB in [0, 1]: every triangle is filled with its vertices' ``displayColor`` interpolated (barycentric), nearest on top. ``view``: ``"top"`` (looking down -z: x right, y up) or ``"side"`` (looking along +y: x right, z up). ``bounds``: … |

#### `pinneapple_tools.nvidia.parts`

Parametric parts as :class:`~pinneapple_tools.nvidia.brep.PartRecipe` data, with their closed-form volumes so the exact solid can be checked against them.

<small>`pinneapple_tools/nvidia/parts.py` · 68 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `l_bracket` | função | `(*, base=0.08, height=0.06, width=0.05, t=0.006, fillet=0.005, bolt_d=0.009, web_hole_d=0…)` | An L bracket: base plate and web of thickness ``t``, an inner fillet, two bolt holes in the base and one hole through the web (none when ``web_hole_d`` is 0). |
| `l_bracket_volume` | função | `(*, base=0.08, height=0.06, width=0.05, t=0.006, fillet=0.005, bolt_d=0.009, web_hole_d=0…)` |  |
| `heat_sink` | função | `(*, n_fins, fin_height, fin_t, base_t=0.005, length=0.1, width=0.1, root_fillet=0.0) -> P…)` | A plate-fin heat sink: a base ``length`` (along the fins, x) by ``width`` (y) and ``n_fins`` equally spaced fins of thickness ``fin_t`` and height ``fin_height`` on top; ``root_fillet`` rounds the inner fin roots (an extrusion has them; the closed-form thermal model does not). |
| `heat_sink_geometry` | função | `(*, n_fins, fin_height, fin_t, base_t=0.005, length=0.1, width=0.1, root_fillet=0.0) -> d…)` | Closed-form volume and areas (``wetted_area``: fins and the exposed base top; ``total_area``: every face). The root fillets add ``r^2 (1 - pi/4)`` of section each to the volume; the areas ignore them. |

#### `pinneapple_tools.nvidia.physicsnemo`

Small PhysicsNeMo surrogates for tabular physics (a few inputs -> a few outputs), trained and checked the PINNeAPPle way. Grid operators (FNO, MeshGraphNet) are in :mod:`pinneapple_neural.physicsnemo_bridge`.

<small>`pinneapple_tools/nvidia/physicsnemo.py` · 146 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MLPSurrogate` | classe | `(in_features, out_features, layer_size, num_layers, activation_fn, seed, x_lo, x_hi, y_lo…)` |  |
| `sign_violations` | função | `(f, axes, axis, output, sign=1, *, tol=0.0) -> float` | Fraction of cells of the tensor grid ``axes`` (one 1-D array per input) where the forward difference of ``f(X)[:, output]`` along input ``axis`` has the sign opposite to ``sign`` by more than ``tol``. |
| `relative_errors` | função | `(pred, ref) -> np.ndarray` | Per-output RMS error relative to the RMS of the reference output. |

#### `pinneapple_tools.nvidia.runinfo`

What a run used: versions of the NVIDIA and PINNeAPPle packages, the hardware, and step timings.

<small>`pinneapple_tools/nvidia/runinfo.py` · 89 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RunInfo` | classe | `(name='')` |  |
| `environment` | função | `() -> dict[str, Any]` | Python, OS, CPU, GPU (CUDA through torch, if any) and the installed versions of the relevant packages. |

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

pinneapple_viz — CFD-style visualisation for PINNeAPPle.

<small>`pinneapple_tools/visualization/__init__.py` · 116 linhas · **sem teste direto**</small>

#### `pinneapple_tools.visualization.animation`

Time-evolution animations for PDE solutions — CFD style.

<small>`pinneapple_tools/visualization/animation.py` · 205 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `animate_scalar_field` | função | `(trajectory, x=None, y=None, *, t_vals=None, title='Field evolution', field_label='u', cm…)` | Animate a time series of 2-D scalar fields. |
| `animate_streamlines` | função | `(u_traj, v_traj, x=None, y=None, *, speed_traj=None, t_vals=None, title='Velocity evoluti…)` | Animated streamline plot for a time series of 2-D velocity fields. |
| `make_gif` | função | `(frames, x, y, save_path, *, t_vals=None, cmap=DEFAULT_CMAP, dpi=80, fps=10) -> None` | Convenience wrapper: save a list of 2-D field frames as a GIF. |

#### `pinneapple_tools.visualization.dashboard`

Animation-centred dashboards for any simulation: an animation in the middle, charts synced to its time cursor, KPI tiles, a parameter panel and a colour bar; exported as interactive HTML (scrubbing a chart moves the animation), a static PNG and a dashboard GIF/MP4.

<small>`pinneapple_tools/visualization/dashboard/__init__.py` · 18 linhas · testes: 16</small>

#### `pinneapple_tools.visualization.dashboard.html`

Interactive HTML export of a :class:`~.spec.Dashboard`: one self-contained file (no server, no internet).

<small>`pinneapple_tools/visualization/dashboard/html.py` · 429 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `payload` | função | `(db) -> dict` | Everything the page draws, as JSON-able data. |
| `scene3d_payload` | função | `(bodies, poses, terrain=None, lighting=None, axes='z_up') -> dict` | Data for the live 3D view. |
| `write_html` | função | `(db, path, *, fps=12, max_width=1280, quality=26) -> str` | Write the self-contained page; returns ``path``. |

#### `pinneapple_tools.visualization.dashboard.spec`

What an animation-centred dashboard shows: an animation (frames), charts synced to its time cursor, KPI tiles, a parameter panel and a colour bar. Pure data (numpy arrays and strings); the exporters are in :mod:`.html` and :mod:`.static`.

<small>`pinneapple_tools/visualization/dashboard/spec.py` · 190 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Series` | classe | `(label, y, color, dash, x, width)` | A line. ``y`` has one value per frame (series charts) or per ``x`` point (phase-chart curves). |
| `Marker` | classe | `(label, x, y, color)` | A point that moves with the frames (phase charts): ``x[k], y[k]`` at frame k. |
| `Chart` | classe | `(title, ylabel, xlabel, series, markers, x, kind, slot, ylim, note)` |  |
| `Tile` | classe | `(label, value, unit, fmt, note)` | A KPI: a scalar, or one value per frame (the tile follows the cursor). |
| `Param` | classe | `(label, value, unit)` |  |
| `ColorBar` | classe | `(title, cmap, lo, hi, saturation)` | Colour bar beside the animation; ``saturation`` < 1 shows the colour map muted the way a semi-transparent, desaturated field overlay draws it (``studio.planetary.SurfaceView.field_saturation``). |
| `Dashboard` | classe | `(title, t, frames, charts, tiles, params, subtitle, colorbar, footer, time_label, time_un…)` | ``frames``: a list of RGB arrays (H, W, 3) or image paths, one per entry of ``t`` [s]. |

#### `pinneapple_tools.visualization.dashboard.static`

PNG and GIF/MP4 exports of a :class:`~.spec.Dashboard`, drawn with matplotlib in the studio's dark style.

<small>`pinneapple_tools/visualization/dashboard/static.py` · 352 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `write_png` | função | `(db, path, k=None, size=(1920, 1080), dpi=100, font_scale=1.0) -> str` | The dashboard at frame ``k`` (default: the last) as a PNG. |
| `write_gif` | função | `(path, frames, fps, colors=255, loop=0) -> str` | Animated GIF of RGB ``frames`` with one shared adaptive palette (octree quantisation of a few frames spread over the clip, which keeps small areas of distinct colour -- chart lines, a colour bar -- that a median cut merges away) and frame differencing: after the first frame, pixels whose palette … |
| `hstack_gifs` | função | `(paths_or_frames, out, fps=10, colors=128, background='white', mismatch='error', loop=0)` | Several animations side by side in one GIF, frame by frame (e.g. two runs of the same case compared). |
| `write_video` | função | `(db, gif=None, mp4=None, *, fps=12, size=(1920, 1080), gif_size=(1280, 720), gif_every=1,…)` | GIF and/or MP4 (H.264, ``size``) of the dashboard over ``frames`` (default all). The GIF keeps every ``gif_every``-th frame at the same playback speed. With ``gif_font_scale`` the GIF is laid out natively at ``gif_size`` with the text scaled by that factor (readable on small screens and social … |

#### `pinneapple_tools.visualization.electromagnetics`

Figures and animations for electromagnetic fields: field maps with Poynting arrows, wave-propagation movies (GIF with Pillow, MP4 with ffmpeg) and radiation patterns.

<small>`pinneapple_tools/visualization/electromagnetics.py` · 134 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `field_map` | função | `(field, extent, path=None, *, outline=None, vectors=None, title='', label='', cmap='RdBu_…)` | Draw one 2-D field (``[ix, iy]``) with an optional material outline and a vector field ``vectors = (vx, vy)`` (e.g. the time-averaged Poynting vector) as arrows every ``stride`` nodes. Returns the matplotlib figure; saves it when ``path`` is given. |
| `field_movie` | função | `(frames, extent, path, *, times=None, outline=None, title='', label='', cmap='RdBu_r', fp…)` | Animate a list of 2-D fields with a fixed symmetric colour scale (set from all frames) and the geometry outline; ``.gif`` is written with Pillow, ``.mp4`` with ffmpeg. |
| `radiation_pattern` | função | `(theta, values, path=None, *, reference=None, labels=('computed', 'analytic'), db=True, f…)` | Polar plot of a far-field pattern (normalised to its maximum), optionally against a reference; ``db`` plots ``20 log10 \|F\|`` clipped at ``floor_db``. |

#### `pinneapple_tools.visualization.fields`

2-D / 3-D scalar and vector field plots — CFD style.

<small>`pinneapple_tools/visualization/fields.py` · 361 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_scalar` | função | `(x, y, field, *, title='Scalar field', label='', cmap=DEFAULT_CMAP, n_levels=64, show_con…)` | Filled contour plot of a 2-D scalar field. |
| `plot_scalar_3d` | função | `(x, y, z, field, *, slice_axis='z', slice_val=None, title='Scalar field (3-D slice)', cma…)` | Plot a 2-D slice of a 3-D scalar field. |
| `plot_vectors` | função | `(x, y, u, v, *, title='Vector field', scale=None, density=20, color_by_mag=True, cmap='ra…)` | Quiver plot with optional background scalar field. |
| `plot_streamlines` | função | `(x, y, u, v, *, title='Streamlines', density=1.5, color_by_mag=True, cmap='plasma', linew…)` | Streamline plot (requires structured or interpolated regular grid). Handles scattered points by interpolating onto a regular grid. |
| `compare_fields` | função | `(x, y, fields, *, cmap=DEFAULT_CMAP, n_cols=3, figsize_per_panel=(5, 4), suptitle='', sho…)` | Side-by-side contour comparison of several fields. |
| `plot_error` | função | `(x, y, pred, ref, *, relative=True, title='Prediction error', cmap='hot', show=False) -> …)` | Visualise pointwise \|pred - ref\| (or relative error) on the domain. |

#### `pinneapple_tools.visualization.integrations`

Bridges from PINNeAPPle results to other 3-D viewers.

<small>`pinneapple_tools/visualization/integrations/__init__.py` · 11 linhas · **sem teste direto**</small>

#### `pinneapple_tools.visualization.integrations.plotly_export`

Self-contained interactive 3-D HTML with Plotly: one file that opens in any browser (rotate, zoom, hover values), to share a result without a server or the PINNeAPPle viewer.

<small>`pinneapple_tools/visualization/integrations/plotly_export.py` · 94 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `scene_html` | função | `(scene, path, field=None, cmap='Turbo', title='', include_plotlyjs=True) -> str` |  |
| `molecule_html` | função | `(mol, path, xyz=None, slab=None, metal='Au', title='', include_plotlyjs=True) -> str` |  |
| `field_slices_html` | função | `(x, y, z, values, path, iso=None, cmap='Viridis', title='', include_plotlyjs=True) -> str` | ``values[i, j, k]`` on the grid ``x[i], y[j], z[k]``: three mid-plane slices and, with ``iso``, an isosurface. |

#### `pinneapple_tools.visualization.integrations.rerun_bridge`

Rerun (rerun.io) bridge: stream PINNeAPPle results into the Rerun viewer, live or into an ``.rrd`` file.

<small>`pinneapple_tools/visualization/integrations/rerun_bridge.py` · 197 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RerunLogger` | classe | `(app_id='pinneapple', path=None, spawn=False, connect=None)` | ``path``: write an ``.rrd`` file; ``spawn``: open a live viewer; ``connect``: a gRPC URL of a running viewer. With none of them the data stay in memory (useful in tests). |

#### `pinneapple_tools.visualization.integrations.trame_app`

Trame (Kitware) web app around a PyVista plotter: the VTK scene rendered on the server or in the browser, with a small control panel, served from Python (no JavaScript to write). Good for heavy meshes and fields that are too big for a standalone HTML file, inside a digital-twin dashboard or on a …

<small>`pinneapple_tools/visualization/integrations/trame_app.py` · 37 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_app` | função | `(plotter, title='PINNeAPPle', local_rendering=False, server_name=None)` | A trame server showing ``plotter``'s render window (``local_rendering``: vtk.js in the browser instead of images streamed from the server). Returns the server; call ``server.start()`` to serve it. |

#### `pinneapple_tools.visualization.mesh`

Mesh and geometry visualisation utilities.

<small>`pinneapple_tools/visualization/mesh.py` · 137 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_mesh` | função | `(nodes, elems, *, node_values=None, cmap=DEFAULT_CMAP, edge_color='#4c9be8', edge_lw=0.5,…)` | Plot a 2-D mesh (triangles or quads). |
| `plot_boundary` | função | `(nodes, boundary_groups, *, title='Boundary groups', show=False) -> Figure` | Visualise labelled boundary groups. |
| `plot_point_cloud` | função | `(points, values=None, *, cmap='plasma', s=3.0, title='Point cloud', show=False, dim3=Fals…)` | 2-D (or 3-D) scatter of a point cloud, optionally coloured by values. |

#### `pinneapple_tools.visualization.molecular`

Molecular visualisation: proteins and molecules, alone or adsorbed on metal surfaces, rendered as stills, MP4 trajectories, 3-D scenes (glTF for the PINNeAPPle web viewer) and MD dashboards.

<small>`pinneapple_tools/visualization/molecular.py` · 503 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Molecule` | classe | `(xyz, elements, names, resnames, resids, chains, frames)` |  |
| `StepLog` | classe | `(title='', color=None, stream=None)` | Step-by-step console log of a pipeline: numbered steps, details indented under them, elapsed time, colour on a terminal (plain text otherwise or with ``color=False``); every line is also kept in ``lines``. |
| `residue_class` | função | `(resname) -> str` |  |
| `fcc_slab` | função | `(metal='Au', facet='111', nx=10, ny=10, layers=4, a=None) -> tuple[np.ndarray, np.ndarray]` | Atom positions (n, 3) of an fcc slab with the given top facet, top layer at z = 0, and the in-plane cell vectors (2, 3) of the periodic box it fills exactly (so the lattice is not strained by periodicity). |
| `slab_for_footprint` | função | `(xyz, metal='Au', facet='111', padding=8.0, layers=4) -> tuple[np.ndarray, np.ndarray, tu…)` | A slab under a molecule: whole lattice repeats covering its x-y footprint plus ``padding`` on each side, centred under it. Returns (positions, cell, (nx, ny)). |
| `orient_on_surface` | função | `(xyz, masses=None, active_site=None, gap=3.0, surface_z=0.0) -> np.ndarray` | Rotate a molecule so its thinnest principal direction is the surface normal (the broadest face lies on the surface), turn ``active_site`` atoms (indices) away from the surface (else the flatter side faces down), and place its lowest atom ``gap`` above ``surface_z``. |
| `rmsd` | função | `(frames, ref, align=True) -> np.ndarray` | RMSD of each frame against ``ref`` (Kabsch-aligned when ``align``). |
| `height_above` | função | `(frames, surface_z, masses=None) -> np.ndarray` | Height of the centre of mass above the surface plane, per frame. |
| `contacts` | função | `(frames, surface_z, cutoff=3.5) -> np.ndarray` | Number of atoms within ``cutoff`` (A) of the surface plane, per frame. |
| `render` | função | `(mol, path, *, xyz=None, slab=None, metal='Au', color_by='residue', style='spheres', size…)` | A still of ``mol`` (positions ``xyz`` or its own) on ``slab``. ``style``: "spheres" (space-filling) or "cartoon" (backbone tube and side-chain atoms); ``color_by``: residue class, element, chain or height. |
| `render_frames` | função | `(mol, frames, out_dir, *, slab=None, metal='Au', color_by='residue', style='spheres', siz…)` | One PNG per frame, with a fixed camera (or a slow orbit of ``orbit_deg`` over the whole run) so motion is readable. Returns the paths. |
| `encode_mp4` | função | `(pngs, out, fps=12, crf=22) -> str` | H.264 MP4 from PNG frames with the ffmpeg binary bundled by ``imageio-ffmpeg``. |
| `md_dashboard` | função | `(path, t_ps, *, potential=None, temperature=None, rmsd_A=None, height_A=None, contacts_n=…)` | A panel per observable given, over time: potential energy (kJ/mol), temperature (K, with the target), backbone RMSD (A), height of the centre of mass above the surface (A), atoms in contact with the surface. |
| `to_glb` | função | `(mol, path, *, xyz=None, slab=None, metal='Au', max_atoms=20000) -> str` | The scene as glTF binary: one low-poly sphere per atom, merged per residue class, and the slab as metal; opens in the PINNeAPPle web viewer and any glTF viewer. |

#### `pinneapple_tools.visualization.pinn`

PINN-specific visualisations: training curves, residual fields, collocation points.

<small>`pinneapple_tools/visualization/pinn.py` · 341 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_loss_history` | função | `(history, *, keys=None, log_scale=True, title='Training loss history', show=False) -> Fig…)` | Plot loss curves from a training history list. |
| `plot_multi_loss` | função | `(history, *, groups=None, log_scale=True, title='Loss components', show=False) -> Figure` | Multi-panel loss breakdown. |
| `plot_collocation` | função | `(interior_pts, boundary_pts=None, initial_pts=None, data_pts=None, *, title='Collocation …)` | Scatter plot of collocation point sets in 2-D. |
| `plot_pinn_prediction` | função | `(x, y, pred, ref=None, *, field_name='u', cmap=DEFAULT_CMAP, show=False) -> Figure` | Two- or three-panel comparison: [Reference] [Prediction] [Error]. If ref is None, shows only the prediction. |
| `plot_pde_residual` | função | `(x, y, residual, *, title='PDE residual', cmap='hot', log_scale=True, show=False) -> Figu…)` | Plot \|PDE residual\| at collocation points. Optionally log-scaled for better dynamic range. |
| `plot_gradient_magnitude` | função | `(x, y, grad_x, grad_y, *, title='Gradient magnitude \|∇u\|', cmap='plasma', show=False) -> …)` | Plot \|∇u\| from separate partial derivative arrays. |
| `predict_and_plot` | função | `(model, x_coords, y_coords, *, ref=None, field_name='u', device=None, cmap=DEFAULT_CMAP, …)` | Run model inference and immediately plot the prediction panel. |

#### `pinneapple_tools.visualization.pyvista_bridge`

PyVista bridge: turn PINNeAPPle meshes and fields into publication-quality 3-D images.

<small>`pinneapple_tools/visualization/pyvista_bridge.py` · 195 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `von_mises` | função | `(stress6) -> np.ndarray` | Von Mises equivalent stress from CalculiX components ``[SXX, SYY, SZZ, SXY, SYZ, SZX]``. |
| `grid_from_arrays` | função | `(points, cells, vtk_cell_type, point_data=None)` | Unstructured grid from ``points (N, 3)`` and ``cells (M, k)`` (0-based point indices). |
| `calculix_grid` | função | `(model, frd=None, step=-1, warp=None)` | PyVista grid of a CalculiX model, with ``DISP``, ``\|U\|``, ``STRESS`` and ``VON_MISES`` if present. |
| `scene_to_multiblock` | função | `(scene, time_index=0, field_name=None)` | ``pinneapple_twin3d.Scene`` to a ``pyvista.MultiBlock`` (one surface per part). |
| `export_vtk` | função | `(mesh, path) -> str` | Write ``.vtu`` / ``.vtp`` / ``.vtk`` / ``.vtm`` (ParaView-ready); format chosen by extension. |
| `can_render` | função | `() -> bool` | True if an off-screen OpenGL context can be created here. Probed in a subprocess (VTK crashes the interpreter instead of raising when there is none). |
| `render_png` | função | `(mesh, path, *, scalars=None, cmap='turbo', title='', show_edges=False, window_size=(1400…)` | Render ``mesh`` (grid, PolyData or MultiBlock) to a PNG. ``scalars`` is a point/cell array name. |
| `render_gif` | função | `(mesh, path, *, scalars=None, cmap='turbo', title='', n_frames=60, elevation=25.0, window…)` | Orbit camera around ``mesh`` and write an animated GIF (turntable). |

#### `pinneapple_tools.visualization.solver`

Visualise SolverOutput objects from FDM / FEM / FVM / meshfree solvers.

<small>`pinneapple_tools/visualization/solver.py` · 198 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_solver_output` | função | `(output, *, x=None, y=None, title='Solver result', cmap=DEFAULT_CMAP, show=False) -> Figu…)` | Auto-plot a SolverOutput. |
| `plot_fem_result` | função | `(output, *, component=0, title='FEM solution', cmap=DEFAULT_CMAP, show_mesh=True, show=Fa…)` | Plot FEM solution on its Q1 mesh. |
| `plot_fvm_result` | função | `(output, *, frame=-1, title='FVM solution', cmap=DEFAULT_CMAP, show=False) -> Figure` | Plot a single time-step from an FVM trajectory. |
| `plot_residuals` | função | `(output, *, title='Solver residuals', show=False) -> Figure` | Bar chart of named residual/loss values stored in output.losses. |

#### `pinneapple_tools.visualization.studio`

PINNeAPPle studio: realistic 3D visualisation of any geometry and its physics.

<small>`pinneapple_tools/visualization/studio/__init__.py` · 47 linhas · testes: 41</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `render` | função | `(*a, **kw)` | Blender Cycles render; see :func:`pinneapple_tools.visualization.studio.blender.render`. |
| `web_viewer` | função | `(*a, **kw)` | Interactive page; see :func:`pinneapple_tools.visualization.studio.web.web_viewer`. |
| `serve` | função | `(*a, **kw)` | Write the viewer to a temporary folder and serve it; see :func:`.web.serve`. |

#### `pinneapple_tools.visualization.studio.animation`

Time animation of a :class:`~.scene.Scene`: rigid bodies moving along a trajectory, per-frame arrows, trails, per-vertex fields and head-up numbers, for the browser viewer (:mod:`.web`) and Blender (:mod:`.blender`).

<small>`pinneapple_tools/visualization/studio/animation.py` · 434 linhas · testes: 27</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Animation` | classe | `(times, tracks, vectors, trails, surface_fields, hud, camera, follow_group, follow_offset…)` | Per-frame data on top of a static :class:`~.scene.Scene`; see the module docstring. |
| `matrix_from_quat` | função | `(q) -> np.ndarray` | Rotation matrix (3, 3) (or (..., 3, 3)) of unit quaternion(s) ``[w, x, y, z]`` (normalised here). |
| `quat_from_matrix` | função | `(R) -> np.ndarray` | Unit quaternion ``[w, x, y, z]`` (w >= 0) of a rotation matrix (3, 3) or a stack (..., 3, 3) (Shepperd's method: the largest of the four squared components is taken first, so it is stable). |
| `slerp` | função | `(q0, q1, f) -> np.ndarray` | Spherical interpolation between unit quaternions ``[w, x, y, z]`` (shortest arc), f in [0, 1]. |

#### `pinneapple_tools.visualization.studio.blender`

Blender Cycles renders of any :class:`~.scene.Scene` (``pip install bpy``, or run inside Blender).

<small>`pinneapple_tools/visualization/studio/blender.py` · 1.296 linhas · testes: 27</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `render` | função | `(scene, out, *, field=None, field_range=None, lines=False, line_range=None, slice=None, s…)` | Render ``scene`` (a Scene or a .glb path) to ``out`` (.png or .jpg). Returns ``out``. |
| `blender_axes` | função | `(axes) -> np.ndarray` | Rotation matrix scene axes -> Blender coordinates (what the glTF export + import does to the surfaces). |
| `render_animation` | função | `(scene, out_dir, *, frames=None, size=(960, 540), samples=16, camera='chase', camera_offs…)` | Render ``scene.animation`` (an :class:`~.animation.Animation`) frame by frame with Cycles (CPU, denoised), burn the HUD into each PNG and encode an MP4 (and a GIF). Returns ``{"frames": [png, ...], "mp4": path \| None, "gif": path \| None}``. |
| `render_still` | função | `(scene, out_png, frame, *, size=(1920, 1080), samples=64, **kw) -> str` | One high-quality frame of ``scene.animation`` (frame index ``frame``) as a PNG at ``out_png``; takes every option of :func:`render_animation` (camera, dof, motion_blur, haze, hud_keys, sparkline, trails, lines, ...). With motion blur the neighbouring frames are keyframed too, so the blur is that of … |

#### `pinneapple_tools.visualization.studio.colormap`

The CFD colour scale (jet: blue = low, red = high) and a colour bar burnt into an image.

<small>`pinneapple_tools/visualization/studio/colormap.py` · 93 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `jet` | função | `(t) -> Tuple[float, float, float]` | sRGB jet colour for t in [0, 1], as floats in [0, 1]. |
| `ramp` | função | `(t, stops=None) -> tuple[float, float, float]` | sRGB colour of a piecewise-linear scale (default :data:`JET`) at t in [0, 1], floats in [0, 1]. |
| `decimals` | função | `(lo, hi) -> int` | Decimals that tell five ticks over [lo, hi] apart (same rule as studio-core.js). |
| `fmt` | função | `(v, nd) -> str` | A tick label: fixed with ``nd`` decimals, exponent form beyond 1e4 or below 1e-3 (as JavaScript prints it). |
| `ticks` | função | `(lo, hi, n=5, nd=None) -> List[str]` | Evenly spaced tick labels from lo to hi, identical to the browser viewer's. |
| `colorbar` | função | `(path, title, lo, hi, nd=None, lo_txt='', hi_txt='', corner='bottom-right', stops=None) -…)` | Draw a colour bar (jet, or the scale ``stops`` such as :data:`RDBU`) with 5 ticks and a title in a corner of the image at ``path`` (overwritten). |

#### `pinneapple_tools.visualization.studio.compose`

Colour maps for animations (jet and diverging blue-white-red), colour bars burnt into images, and multi-panel pictures and videos (a grid of renders with labels and one common colour bar).

<small>`pinneapple_tools/visualization/studio/compose.py` · 238 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `stops` | função | `(cmap)` |  |
| `cmap_color` | função | `(cmap, t) -> tuple[float, float, float]` | sRGB colour (floats in [0, 1]) of ``cmap`` at t in [0, 1] (NaN -> 0). |
| `symmetric_range` | função | `(values) -> tuple[float, float]` | (-m, m) with m the largest finite \|value\| (a diverging map keeps zero at its neutral midpoint). |
| `default_range` | função | `(values, cmap='jet') -> tuple[float, float]` | min..max for jet, symmetric about zero for a diverging map. |
| `colorbar_image` | função | `(width, title, lo, hi, cmap='jet', scale=1.0, background=(255, 255, 255, 225))` | A colour bar (title, gradient, five ticks) as an RGBA PIL image ``width`` pixels wide. |
| `draw_colorbars` | função | `(path, bars, corner='bottom-right', width_frac=0.3) -> str` | Burn colour bars ``(title, lo, hi, cmap)`` into the image at ``path`` (overwritten), stacked from ``corner``. |
| `compose_grid` | função | `(images, out_png, labels=None, ncols=2, colorbar=None, background=(233, 237, 242)) -> str` | One picture of ``images`` on a grid of ``ncols`` columns (every image scaled to the first one's size), a label in each cell's top-right corner and one common colour bar ``(title, lo, hi, cmap)`` under the grid. |
| `compose_grid_video` | função | `(mp4_paths, out_mp4, labels=None, ncols=2, colorbar=None, crf=28, webm=False, fps=None, b…)` | A grid video of ``mp4_paths`` (ffmpeg ``xstack``; every clip scaled to the first one's size, the shorter ones hold their last frame), with the labels and one colour bar strip ``(title, lo, hi, cmap)`` burnt in. H.264 ``crf`` (yuv420p, even size, faststart); ``webm=True`` also writes a VP9 .webm … |

#### `pinneapple_tools.visualization.studio.fieldview`

Fields on a mesh for the apps' 3-D views: the boundary surface of a ``pinneapple_data.cae`` mesh with several fields (values and errors), optional deformed shapes and marked regions, as a glTF plus a small JSON description.

<small>`pinneapple_tools/visualization/studio/fieldview.py` · 179 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FieldView` | classe | `(scene, meta)` |  |
| `colour_range` | função | `(values, kind='value') -> dict[str, Any]` | ``{"lo", "hi", "max", "clipped"}``: values span min..max; errors span 0..99th percentile (``clipped`` when the maximum is above it). |
| `length_factor` | função | `(unit, to='m') -> float \| None` | Factor from ``unit`` to ``to`` when ``unit`` is a length (``mm`` -> m: 1e-3), else None. |
| `displacement_factor` | função | `(name, unit, components, mesh_unit='m') -> float \| None` | If a nodal vector field is a displacement, the factor that converts it to the mesh's length unit: its unit is a length, or it has no unit and is named like one (``D``, ``U``, ``disp``, ``displacement``). Else None. |
| `subsample` | função | `(points, n=MAX_MARKER_POINTS) -> np.ndarray` | At most ``n`` of the points, evenly spread over the array (markers do not need every node). |
| `field_view` | função | `(mesh, fields, *, deform=None, markers=(), max_faces=VIEW_MAX_FACES, title='', axes='z_up…)` | Build the view. ``fields``: dicts with ``name``, ``values`` (one per point or per cell, vectors as magnitude, 6-component tensors as von Mises), optional ``where`` ("point" / "cell", from the length), ``label``, ``unit``, ``kind`` ("value" or "error"), ``group`` (shared colour range), ``deform`` (a … |

#### `pinneapple_tools.visualization.studio.lighting`

Lighting and sky presets shared by the studio renderers (PyVista, Blender Cycles, three.js viewer).

<small>`pinneapple_tools/visualization/studio/lighting.py` · 171 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LightingPreset` | classe | `(name, sun_elevation_deg, sun_azimuth_deg, sun_intensity, sun_color, ambient, background_…)` |  |
| `get_preset` | função | `(preset) -> LightingPreset` |  |
| `apply_pyvista` | função | `(plotter, preset='lunar_south_pole', target=(0.0, 0.0, 0.0), extent=10.0, earth=None, sha…)` | Set the background, a directional Sun (with shadows when the preset asks for them), a weak fill light and, optionally, Earth in the sky of a PyVista ``plotter``. |
| `add_earth_pyvista` | função | `(plotter, preset, target, extent)` | Earth as a shaded sphere far away, with the right angular size, lit from the Sun's direction. |
| `threejs_config` | função | `(preset='lunar_south_pole') -> dict` | The preset as plain data for the browser viewers (background, sun direction and intensity, fill). |
| `blender_world` | função | `(preset='lunar_south_pole', bpy=None)` | Configure the current Blender scene: world colour, one Sun lamp with the preset's angular size and strength, and (for black-sky presets) no world light. Returns the Sun object. |

#### `pinneapple_tools.visualization.studio.particles`

Particle animations: thousands of spheres coloured by a value (speed, temperature, species), inside glass or steel equipment that can move (a turning impeller), rendered frame by frame with Blender Cycles; then composed with live charts that draw themselves as time runs, a colour bar and a clock, …

<small>`pinneapple_tools/visualization/studio/particles.py` · 323 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `render_particle_frames` | função | `(frames, d, out_dir, *, equipment=None, field='speed', field_range=None, view=(-1.0, -1.2…)` | Render every ``every``-th frame to ``out_dir/frame_XXXX.png``; the camera frames the equipment (or the particles) once. Returns the PNG paths. |
| `compose_video` | função | `(images, times, out, *, charts=(), colorbar=None, cmap='viridis', clock='Time: {t:.1f}', …)` | One composed frame per image: the render on the left, the ``charts`` (label, t, y) on the right drawn up to the current time (full axes from the start, so the line grows), a colour bar over the render and a clock. Writes a GIF to ``out``; with ``mp4`` also ``out`` with .mp4 through Blender's FFmpeg … |
| `encode_mp4` | função | `(pngs, out, fps=10) -> str` | H.264 MP4 of a PNG sequence with Blender's sequencer (no ffmpeg binary needed). |

#### `pinneapple_tools.visualization.studio.planetary`

Renders of a vehicle on a planetary surface with PyVista: heightfield terrain with baked photometry and hard Sun shadows (:mod:`.shadows`), vehicle parts with their studio materials, rocks, optional field colour maps on the terrain and on the parts (with a colour bar), Earth in the sky and a title.

<small>`pinneapple_tools/visualization/studio/planetary.py` · 348 linhas · testes: 17</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SurfaceView` | classe | `(X, Y, Z, parts, rocks, preset, terrain_field, field_alpha, field_glow, field_saturation,…)` |  |
| `horizon_distance` | função | `(height, planet_radius=MOON_RADIUS) -> float` | Distance [m] to the horizon of a smooth sphere seen from ``height`` metres above it. |
| `ring_height` | função | `(view, x, y)` | Height of the horizon annulus [m] at (x, y): ``far_fn`` (or the grid's edge level) blended into the edge level over ``horizon_blend`` metres, a long-wavelength relief faded in with distance, and the curvature drop. |
| `terrain_height` | função | `(view, x, y)` | Height of the rendered ground at (x, y): the detailed patch, the far grid, then the horizon annulus. |
| `border_points` | função | `(view, n=720) -> np.ndarray` | Points (n, 3) on the outer border of the rendered ground (the annulus rim, else the far grid's edge). |
| `border_visibility` | função | `(view, camera, size=(1600, 900), view_angle=35.0, samples=400, margin=0.0) -> dict` | Is the outer border of the ground mesh visible from ``camera``? Every border point is projected through the camera; a point inside the frame (grown by ``margin`` pixels) counts as visible unless the ground between the camera and the point hides it (``samples`` steps along the sight line, geometric … |
| `render_surface` | função | `(view, path, camera, *, size=(1600, 900), view_angle=35.0, title='', subtitle='', plotter…)` | Render ``view`` with ``camera = (position, focal_point[, up])``. Returns the image path or (with ``return_image``) the RGB array. |

#### `pinneapple_tools.visualization.studio.pyvista_render`

Off-screen renders of a studio :class:`~.scene.Scene` with PyVista (VTK): the Blender-free counterpart of :func:`pinneapple_tools.visualization.studio.render`, for machines without ``bpy``.

<small>`pinneapple_tools/visualization/studio/pyvista_render.py` · 157 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `render_scene` | função | `(scene, path, *, field=None, cmap='jet', clim=None, lines=False, slice_name=None, view='i…)` | ``slice_crop``: (s0, s1, t0, t1), the part of the slice to show as fractions of its ``u`` and ``v`` edges. ``wireframe``: a colour (e.g. "#3B82F6") draws every surface as its mesh edges in that colour instead. ``bar_title``: a short colour-bar title with its units (e.g. "Cp"); by default the … |

#### `pinneapple_tools.visualization.studio.scene`

A 3D scene for any geometry: surfaces with materials and per-vertex fields, streamlines, slice planes.

<small>`pinneapple_tools/visualization/studio/scene.py` · 682 linhas · testes: 46</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Surface` | classe | `(name, vertices, faces, material, fields, group)` | One triangulated surface. ``fields``: name -> one value per vertex. |
| `Lines` | classe | `(points, values, name, label, cmap, alpha, radius, color_range)` | Polylines (e.g. streamlines), each with optional per-point values (e.g. speed). ``cmap``: "jet" or the diverging "coolwarm" / "bwr"; ``color_range`` (default: 1st..99th percentile, symmetric for a diverging map); ``alpha`` < 1 and ``radius`` (tube radius; default from the scene size) for soft … |
| `Slice` | classe | `(name, origin, u, v, grid, label, group)` | A rectangular plane with a value grid: ``origin`` + s*``u`` + t*``v`` for s, t in [0, 1]. ``grid[j, i]``: row j along v, column i along u; NaN = no value (inside a body). |
| `Scene` | classe | `(surfaces, lines, slices, axes, title, labels, lighting, materials, animation)` |  |
| `decimate_surface` | função | `(srf, target_faces) -> 'Surface'` | Vertex clustering on a uniform grid whose cell size is searched (bisection) to land near ``target_faces``. Fast and robust for display; thin features below the cell size may close up. |
| `scalar_of` | função | `(a) -> np.ndarray` | One value per entry: itself, the magnitude of a vector, the von Mises stress of a 6-component tensor (xx, yy, zz, xy, yz, zx). |
| `vertex_normals` | função | `(V, F) -> np.ndarray` |  |
| `read_stl` | função | `(path) -> List[Tuple[str, np.ndarray, np.ndarray]]` | (name, vertices, faces) per solid; duplicate vertices merged. |
| `read_obj` | função | `(path) -> Tuple[np.ndarray, np.ndarray]` |  |
| `read_gltf` | função | `(path, axes='z_up') -> List[Surface]` | Triangle meshes of a .glb/.gltf (embedded, data-URI or external buffers), node transforms applied, converted from glTF's y-up to the scene ``axes``. Float vertex attributes starting with ``_`` become fields (their original names when the file was written by this module). |
| `read_vtp` | função | `(path, material='grey') -> Surface` | VTK PolyData (.vtp) with its point and cell data (vectors as magnitude). Needs ``pip install vtk``. |
| `write_glb` | função | `(surfaces, T, materials, generator='PINNeAPPle', groups=None) -> bytes` | glTF 2.0 binary with PBR materials (clearcoat, transmission, emissive). Every per-vertex field becomes a float attribute ``_NAME`` (upper case; three.js reads it back as ``_name``). ``groups``: node name -> indices of surfaces to parent under one node (e.g. a propeller that spins). |
| `write_usda` | função | `(surfaces, materials, title='scene', root='Scene') -> str` | OpenUSD text: one Mesh per surface with UsdPreviewSurface materials; z up, metres (Omniverse, usdview). Per-vertex fields become ``primvars:<name>`` (float, vertex interpolation). |
| `write_stl` | função | `(surfaces, solid='scene') -> bytes` | Binary STL of all surfaces. |

#### `pinneapple_tools.visualization.studio.shadows`

Sun shadows and planetary-surface shading for heightfield terrains, independent of the renderer.

<small>`pinneapple_tools/visualization/studio/shadows.py` · 171 linhas · testes: 17</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `terrain_self_shadow` | função | `(Z, cell, sun_dir, max_distance=None) -> np.ndarray` | Boolean (ny, nx): True where the heightfield ``Z`` (rows along y, columns along x, spacing ``cell``) hides the Sun (``sun_dir`` points from the ground to the Sun). |
| `points_in_terrain_shadow` | função | `(P, X, Y, Z, sun_dir, step=None) -> np.ndarray` | Boolean per point (N, 3): the heightfield (X, Y, Z) blocks the Sun as seen from the point. |
| `cast_shadow` | função | `(meshes, X, Y, Z, sun_dir, iterations=3, supersample=1) -> np.ndarray` | Boolean (ny, nx): cells of the regular grid (X, Y, Z) in the shadow of the triangle ``meshes`` [(V, F), ...]. |
| `lunar_lambert` | função | `(normals, sun_dir, view_dir, L=1.0) -> np.ndarray` | McEwen (1991) Lunar-Lambert reflectance (without the albedo) for unit ``normals`` (N, 3); ``view_dir`` (3,) or (N, 3) points from the surface to the camera. |
| `relief_noise` | função | `(X, Y, rms, wavelengths=(0.6, 8.0), n=40, seed=7, slope=1.2)` | World-anchored smooth random relief [m] at points (X, Y): a sum of ``n`` plane waves with random directions and phases, wavelengths log-spaced in ``wavelengths`` and amplitudes ~ wavelength^slope, scaled to ``rms``. The same (x, y) always gives the same height, so a moving window of terrain does … |
| `grid_normals` | função | `(Z, cell) -> np.ndarray` |  |
| `shade_terrain` | função | `(X, Y, Z, sun_dir, camera_position, albedo=(0.45, 0.44, 0.42), shadow=None, exposure=1.0,…)` | (ny, nx, 3) colours in [0, 1]: Lunar-Lambert photometry, shadows from ``shadow`` (bool grid or 0-1 fraction), a weak fill light, a filmic shoulder. ``tint`` (ny, nx, 3) multiplies the albedo (e.g. a field colour map). |

#### `pinneapple_tools.visualization.studio.solver_view`

The result of a solver run as a 3-D view, to check it before its numbers are used: the mesh, its fields, the deformed shape (CalculiX displacement, with a scale factor) and the regions where the case defines its boundary conditions and loads.

<small>`pinneapple_tools/visualization/studio/solver_view.py` · 153 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `find_result` | função | `(folder) -> tuple[str, Path] \| None` | ``("calculix", <job>.frd)`` or ``("openfoam", <case dir>)`` for a step folder (or the case itself), else None. |
| `solver_result_view` | função | `(folder, *, max_faces=VIEW_MAX_FACES) -> tuple[FieldView, dict[str, Any]]` | View of the CalculiX or OpenFOAM result in ``folder`` (see the module docstring). Raises ``ValueError`` when there is no result to show. |

#### `pinneapple_tools.visualization.studio.web`

Interactive browser view of a :class:`~.scene.Scene` (three.js, shipped with the library; no internet needed).

<small>`pinneapple_tools/visualization/studio/web.py` · 197 linhas · testes: 27</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `web_viewer` | função | `(scene, out_dir, title=None, max_faces=MAX_FACES, single_file=False, filename='index.html…)` | Write the viewer for ``scene`` into ``out_dir``; returns the path of its index.html. Surfaces above ``max_faces`` triangles in total are decimated for the page (the scene itself is not changed); None keeps all. ``single_file``: write only ``out_dir/filename``, with scene.json, scene.glb, the viewer … |
| `serve` | função | `(scene, port=8000, open_browser=True, block=True, out_dir=None, max_faces=MAX_FACES)` | Write the viewer (to ``out_dir`` or a temporary folder) and serve it on ``port``. ``block=False`` returns the server (call ``.shutdown()``); otherwise serves until Ctrl+C. |
| `screenshot` | função | `(viewer_dir, out_png, frame=None, size=(1280, 720), setup_js=None, timeout_ms=60000) -> s…)` | PNG of the viewer written by :func:`web_viewer` in ``viewer_dir`` (or of a single-file page: the path of the .html, opened from file://), taken in headless Chromium (needs ``pip install playwright``; ``PLAYWRIGHT_CHROMIUM`` or /opt/pw-browsers/chromium is used when the bundled browser is missing). … |

#### `pinneapple_tools.visualization.style`

CFD-style colour maps, themes and defaults for PINNeAPPle visualisation.

<small>`pinneapple_tools/visualization/style.py` · 118 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `use_cfd_style` | função | `(dark=True) -> None` | Apply CFD dark/light theme globally. |
| `get_cmap` | função | `(name=DEFAULT_CMAP)` | Return a matplotlib colormap by PINNeAPPle alias or standard name. |
| `make_figure` | função | `(nrows=1, ncols=1, **kwargs)` | Create a styled figure. |

#### `pinneapple_tools.visualization.trust_panels`

Side-by-side field panels with a trust verdict under each model: the figure (and GIF) of "which prediction can I use, and where does it break the physics".

<small>`pinneapple_tools/visualization/trust_panels.py` · 167 linhas · testes: 2</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Panel` | classe | `(title, values, kind, verdict, verdicts, note, inset)` | One field. ``kind``: ``"field"`` (shared linear scale per row), ``"residual"`` (log magnitude) or ``"error"`` (magnitude on a linear scale from 0 to ``error_max`` of :func:`trust_panels`). ``inset``: draw a dashed square that many cells inside the border (e.g. the edge of an interior region). |
| `Row` | classe | `(label, panels)` |  |
| `trust_panels` | função | `(rows, *, title='', footer='', cell=2.4, residual_range=None, error_max=None, error_label…)` | Figure with one row per :class:`Row`: its ``"field"`` panels, a colour bar, its ``"residual"`` / ``"error"`` panels and their colour bar (labelled ``error_label``); every row has the same number of panels of each group. |
| `trust_panels_gif` | função | `(frames, path, *, duration_ms=1600, **kwargs) -> Path` | One :func:`trust_panels` figure per frame, written as a looping GIF. ``kwargs`` go to ``trust_panels`` (pass the same ``residual_range`` to every frame for a fixed residual scale). |

#### `pinneapple_tools.visualization.vortex`

Vortex identification and flow visualization — Q-criterion, λ₂, vorticity.

<small>`pinneapple_tools/visualization/vortex.py` · 675 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `compute_vorticity_2d` | função | `(u, v, dx=1.0, dy=1.0) -> np.ndarray` | z-component of vorticity: ω_z = ∂v/∂x − ∂u/∂y. |
| `compute_q_criterion_2d` | função | `(u, v, dx=1.0, dy=1.0) -> np.ndarray` | Q-criterion for 2-D incompressible flow. |
| `compute_vorticity_3d` | função | `(u, v, w, dx=1.0, dy=1.0, dz=1.0) -> np.ndarray` | Full vorticity vector field ω = ∇×u for 3-D flow: ω_x = ∂w/∂y − ∂v/∂z ω_y = ∂u/∂z − ∂w/∂x ω_z = ∂v/∂x − ∂u/∂y |
| `compute_strain_rate_tensor_3d` | função | `(u, v, w, dx=1.0, dy=1.0, dz=1.0)` | Symmetric strain-rate tensor S_ij = 0.5(∂u_i/∂x_j + ∂u_j/∂x_i) for 3-D flow. |
| `compute_q_criterion_3d` | função | `(u, v, w, dx=1.0, dy=1.0, dz=1.0) -> np.ndarray` | Q-criterion for 3-D flow:  Q = 0.5(‖Ω‖²_F − ‖S‖²_F) |
| `compute_enstrophy` | função | `(vorticity) -> float` | Enstrophy: 0.5 * sum(\|ω\|²). |
| `compute_dissipation_rate` | função | `(strain_rate_tensor_components, nu) -> np.ndarray` | Viscous/turbulent dissipation rate: ε = 2ν S_ij S_ij (double contraction of the strain-rate tensor). |
| `compute_lambda2_3d` | função | `(u, v, w, dx=1.0, dy=1.0, dz=1.0) -> np.ndarray` | λ₂ vortex criterion (Jeong & Hussain, 1995). Vortex core: λ₂ < 0. |
| `plot_vorticity` | função | `(x, y, omega, *, title='Vorticity ω_z', cmap='RdBu_r', symmetric=True, obstacle_mask=None…)` | Coloured vorticity map, symmetric diverging colormap centred on zero. |
| `plot_q_criterion_2d` | função | `(x, y, Q, vel_mag=None, *, title='Q-criterion', q_levels=30, highlight_positive=True, cma…)` | 2-D Q-criterion field. |
| `plot_vortex_identification` | função | `(x, y, u, v, *, dx=1.0, dy=1.0, title=…, obstacle_mask=None, figsize=(11, 5), show=False)…)` | One-shot wrapper: compute Q from velocity field and plot. Also overlays streamlines on the Q background. |
| `plot_q_criterion_3d` | função | `(x, y, z, Q, *, level=0.01, vel_mag=None, title='Q-criterion iso-surface', cmap='rainbow'…)` | 3-D Q-criterion iso-surface coloured by velocity magnitude. |
| `plot_lbm_flow` | função | `(ux, uy, rho=None, obstacle_mask=None, *, dx=1.0, dy=1.0, title='LBM Flow Field', figsize…)` | 4-panel LBM dashboard: [0] Velocity magnitude  [1] Vorticity ω_z [2] Q-criterion         [3] Pressure (density proxy) Solid obstacles are masked in grey. |
| `plot_flow_panel` | função | `(x, y, ux, uy, pressure=None, obstacle_mask=None, *, title='Flow Field Analysis', figsize…)` | 4-panel flow analysis: pressure, \|v\| + streamlines, Q-criterion, vorticity. Mirrors the post-processing style of commercial CFD dashboards. |

#### `pinneapple_tools.visualization.voxel`

Visualisation of VoxelGrid objects.

<small>`pinneapple_tools/visualization/voxel.py` · 127 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_voxel_slice` | função | `(grid, *, axis='z', index=None, cmap=DEFAULT_CMAP, title='', show=False) -> Figure` | Plot a single 2-D slice through a 3-D VoxelGrid. |
| `plot_voxel_3d` | função | `(grid, *, threshold=0.5, cmap=DEFAULT_CMAP, alpha=0.6, title='Voxel occupancy', show=Fals…)` | Scatter-based 3-D visualisation of occupied voxels. |
| `plot_voxel_histogram` | função | `(grid, *, bins=50, title='Voxel value distribution', show=False) -> Figure` | Histogram of non-zero voxel values. |

### Exemplos de uso reais do repositório

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

Node types that reproduce the ``examples/*`` studies as flows, one module per domain, registered through the public API (:func:`pinneapple_flow.registry.node`, :func:`pinneapple_flow.types.register_port_type`). Each node wraps library calls; the flows that use them are under …

<small>`pinneapple_flow/nodes_ext/__init__.py` · 58 linhas · **sem teste direto**</small>

#### `pinneapple_flow.nodes_ext.aero`

Aircraft aerodynamics from the geometry library (``examples/geometry_library/03_airliner_vlm``): the vortex lattice of the ``airliner_a320`` entry (:func:`pinneapple_design.aero.vlm.solve`) and the friction-drag build-up of its mesh (:func:`pinneapple_design.aero.buildup.friction_drag`).

<small>`pinneapple_flow/nodes_ext/aero.py` · 71 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `airliner_vlm` | função | `(ctx, inputs, params)` | Lift slope, neutral point (wing + tail), trim at the cruise lift coefficient, induced drag; CD0 from the library mesh; L/D. Published A320 class: CL_alpha about 5.5-6 /rad, cruise L/D about 16-18. |

#### `pinneapple_flow.nodes_ext.architectures`

The model registry examples (``examples/architectures``): the registry and catalog listing (00, 01, the Group-B check of 60), building registry models by name with a forward pass on dummy tensors (00, 60) and POD compression with a DMD rollout on a synthetic linear system (30).

<small>`pinneapple_flow/nodes_ext/architectures.py` · 172 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `registry_listing` | função | `(ctx, inputs, params)` | The registry's families and models, a few model specs and the catalog by family. This is what the registry tour and the model list print. |
| `forward` | função | `(ctx, inputs, params)` | One registry model built by name and run on dummy tensors. Reports the output shape, the parameter count, the losses it returns and, on request, the input-gradient norm (gradient flow for PINN use). |
| `pod_dmd` | função | `(ctx, inputs, params)` | POD reconstruction and POD + DMD rollout errors on a synthetic linear system. Trajectories of a contractive random linear system are compressed by POD (rank r), advanced by DMD on the POD coefficients and decoded. |

#### `pinneapple_flow.nodes_ext.arena_pipelines`

The Arena pipelines of ``examples/arena_pipelines``: the dataset registry (00), the physics and time-series benchmark specs (01, 02, :mod:`pinneapple_tools.benchmark_suite`) and the multi-model Arena run from a config (04, 05, :class:`pinneapple_arena.Arena`). Each run writes its figures and tables …

<small>`pinneapple_flow/nodes_ext/arena_pipelines.py` · 205 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `datasets` | função | `(ctx, inputs, params)` | The dataset ids per category and, for each requested id, the arrays it loads (or the error). |
| `physics_spec` | função | `(ctx, inputs, params)` | Every model of the spec on the problem (forward or inverse), metrics against the reference, a leaderboard and, for an inverse run, the identified parameters. |
| `timeseries_spec` | função | `(ctx, inputs, params)` | Every forecasting model of the spec on the same lookback / horizon split, metrics on the test part. |
| `arena_run` | função | `(ctx, inputs, params)` | Data from the problem's reference, every model trained and evaluated on the same points; per model the L2 and relative error of each field, the training time and (PINN family) the final physics residual. |

#### `pinneapple_flow.nodes_ext.astro`

Black-hole accretion nodes (``examples/black_hole_weather``): one RIAF torus simulation around a Schwarzschild black hole (:func:`pinneapple_physics.blackhole.runs.simulate_run`) and the time-window statistics the resolution study uses …

<small>`pinneapple_flow/nodes_ext/astro.py` · 117 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `simulate` | função | `(ctx, inputs, params)` | Evolve the viscous, radiatively inefficient flow that forms from an equilibrium torus (Paczynski-Wiita potential), saving the primitives every ``every`` M (examples/black_hole_weather/simulate.py). |
| `statistics` | função | `(ctx, inputs, params)` | Mean accretion rate, density and specific angular momentum profiles and the persistence error at leads of 20, 100 and 1000 M over the window (the quantities of the resolution study, results/convergence.json). |

#### `pinneapple_flow.nodes_ext.autolab_caps`

Planner capabilities for the example flows of nodes that live elsewhere: the industrial presets (:mod:`pinneapple_flow.nodes.presets`), the science experiments (:mod:`pinneapple_flow.nodes.science`) and the autonomous-lab missions (:mod:`pinneapple_flow.nodes.autolab`, ``rover.learn_planet``).

<small>`pinneapple_flow/nodes_ext/autolab_caps.py` · 42 linhas · **sem teste direto**</small>

#### `pinneapple_flow.nodes_ext.benchmarks`

Benchmark harnesses as flows: the equal-budget benchmark (``examples/benchmark``, :func:`pinneapple_analysis. benchmark.run`) and the benchmark suite of ``examples/benchmark_suite`` (:mod:`pinneapple_tools.benchmark_suite`): the Arena experiment from the example's YAML config (02), and the …

<small>`pinneapple_flow/nodes_ext/benchmarks.py` · 239 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `equal_budget` | função | `(ctx, inputs, params)` | Every method on every problem under the same ComputeBudget, the same hyper-parameter search effort and seeds; the result table (mean over seeds at the full budget), the correctness gates and the advisor's diagnostics. The run folder is the node's (config.yaml, prepared/, runs/, table.md, … |
| `arena_yaml` | função | `(ctx, inputs, params)` | The problem preset, its reference data from the built-in solver, every model of the config trained on the same points; per model the best validation loss, the metrics on the reference points and the time. |
| `transfer` | função | `(ctx, inputs, params)` | Pre-train on a source member of each family, fine-tune on targets with each freezing strategy (and from scratch); relative L2 per scenario, target and strategy. |
| `meta` | função | `(ctx, inputs, params)` | Meta-train on each PDE family, adapt K steps to held-out members against training from scratch; relative L2 per family, algorithm and K. |

#### `pinneapple_flow.nodes_ext.bricks`

Self-modifying interlocking-brick builds (``examples/bricks/self_modifying_bricks``): the closed-form and Newton checks that gate the loop (:func:`pinneapple_design.bricks.validation.validate`) and the two search strategies at the same evaluation budget and seeds …

<small>`pinneapple_flow/nodes_ext/bricks.py` · 83 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `validate` | função | `(ctx, inputs, params)` | Every check's simulated value (``check.<i>``, in the library's order, the check named in the note), whether it passed, and the trust decision. |
| `search` | função | `(ctx, inputs, params)` | Both strategies on every seed at the same budget: best score per seed, median (a seed with no feasible build counts as the worst), mean, spread, success rate; and the best build found. |

#### `pinneapple_flow.nodes_ext.cfd`

CFD nodes on OpenFOAM: the free-surface trough of ``examples/free_surface`` (a 2-D section with a liquid bath fed by a jet, interFoam VOF with k-omega SST, :class:`pinneapple_simulation.numerical_solvers.free_surface.FreeSurfaceFlow`) and the run campaign of ``examples/openfoam_campaign`` …

<small>`pinneapple_flow/nodes_ext/cfd.py` · 210 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `trough_jet` | função | `(ctx, inputs, params)` | examples/free_surface/01_runner_slope_study.py for one slope: time-mean k a few cm above the floor in the impact zone, its peak, the depth under the jet and how settled the liquid volume is. |
| `ratio` | função | `(ctx, inputs, params)` | ``<case>.<metric> / <baseline>.<metric>`` for every upstream case (e.g. k relative to the 0 degree floor). |
| `campaign` | função | `(ctx, inputs, params)` | examples/openfoam_campaign: the base case written by the library, then each run of the plan (jet speed and angle) with its health (Courant, phase bounds, volume) and the sampled profiles. |

#### `pinneapple_flow.nodes_ext.core_ts`

Nodes for ``examples/geophysics`` (shallow water on the sphere, 4D-Var on Lorenz-96, Richards infiltration), ``examples/physics_optimizer`` (:class:`pinneapple_core.optim.PhysicsOptimizer` on a constrained shape), ``examples/physics_module`` (:class:`pinneapple_core.module.Hybrid`: a coarse FEM …

<small>`pinneapple_flow/nodes_ext/core_ts.py` · 270 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `earth_system` | função | `(ctx, inputs, params)` | examples/geophysics/01_sphere_and_assimilation.py: the Rossby-Haurwitz wave (Williamson case 6) and the drift of its conserved quantities, 4D-Var on Lorenz-96 (background and analysis RMSE), Richards infiltration (Celia et al. 1990) and its mass-balance ratio. |
| `torsion_optimizer` | função | `(ctx, inputs, params)` | Maximise the torsional rigidity of an a x b rectangle with a b = 1 by SLSQP, the adjoint, a penalty, an augmented Lagrangian, differential evolution and Bayesian optimisation (examples/physics_optimizer). |
| `hybrid_fem` | função | `(ctx, inputs, params)` | -(k(x) u')' = 1 with k = 1 + 0.8 sin(2 pi f x + phase): a 6-cell FEM and a dilated Conv1d corrector trained end to end against a 128-cell FEM; held-out relative error of the coarse FEM alone and of the hybrid (examples/physics_module/01_coarse_fem_plus_neural_correction.py). |
| `regime_series` | função | `(n=900, seed=0) -> np.ndarray` | The series of examples/time_series/adaptive_forecasting.py: seasonal, then a trend, then a random walk. |
| `adaptive` | função | `(ctx, inputs, params)` | Experts weighted by their recent errors on a seasonal -> trend -> random-walk series: the ensemble's MAE against every expert, the 90 % interval coverage and when the leading model changes (examples/time_series/adaptive_forecasting.py). |
| `naive_baselines` | função | `(ctx, inputs, params)` | examples/time_series/00_quickstart.py: a seasonal + trend + noise series, time-ordered train / validation windows, the naive, seasonal-naive and drift baselines and their errors. |

#### `pinneapple_flow.nodes_ext.design`

Design nodes: topology optimization (``examples/topology_optimization``) on :mod:`pinneapple_design.topology` (SIMP compliance of the MBB beam and the cantilever, thermal compliance, Stokes dissipation of the double pipe, the iron of a reluctance rotor for torque, the sensitivity check) and design …

<small>`pinneapple_flow/nodes_ext/design.py` · 321 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `topology2d` | função | `(ctx, inputs, params)` | examples/topology_optimization/01_mbb_beam.py and 02_heat_and_stokes_flow.py. |
| `topology_gradcheck` | função | `(ctx, inputs, params)` | The adjoint sensitivities of the filtered, projected MBB compliance against central differences on the largest components (examples/topology_optimization/01_mbb_beam.py). |
| `reluctance_rotor` | função | `(ctx, inputs, params)` | examples/topology_optimization/03_reluctance_rotor.py: 4-pole rotor core, 4-fold symmetry, adjoint sensitivities; the final design re-evaluated with nonlinear iron and with the co-energy torque. |
| `from_intent` | função | `(ctx, inputs, params)` | A goal in plain language to a verified, buildable design, the questions it needs, or a refusal saying what is missing (examples/design_from_intent/design_from_intent.py). |
| `ahmed_slant` | função | `(ctx, inputs, params)` | examples/qualitative/03_ahmed_slant.py: the preview's expected drag change per slant against the measured one, its direction and the rank correlation; low confidence near the separation angle. |

#### `pinneapple_flow.nodes_ext.electromagnetics`

Electromagnetics nodes: the studies of ``examples/motor_magnetics`` and ``examples/electrodynamics`` on :mod:`pinneapple_simulation.electromagnetics` (the verification table, the surface-PM machine, the 2004 Prius interior-PM motor: no-load, torque against current angle and MTPA, flux map and drive …

<small>`pinneapple_flow/nodes_ext/electromagnetics.py` · 512 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `em_verification` | função | `(ctx, inputs, params)` | Every electromagnetics solver and reference against an independent result: errors and observed orders (examples/electrodynamics/07_em_verification.py). |
| `spm_machine` | função | `(ctx, inputs, params)` | examples/motor_magnetics/01_spm_cogging_and_torque.py: 4 poles, 24 slots, nonlinear iron; torque by Maxwell stress (Arkkio) and by virtual work, side by side. |
| `prius_machine` | função | `(ctx, inputs, params)` | examples/electrodynamics/10_pmsm_prius.py: the 2004 Prius motor (48 slots, 8 poles, V-shaped NdFeB, M400-50A), torque against current angle compared with Yang et al. (IEMDC 2013). |
| `prius_flux_map` | função | `(ctx, inputs, params)` | psi_d, psi_q and torque on an i_d x i_q grid (and loss data on a coarser grid) of the Prius motor. |
| `prius_drive` | função | `(ctx, inputs, params)` | examples/electrodynamics/11_prius_drive.py: MTPA + flux weakening envelope, the efficiency map and the battery energy over the WLTC class 3b cycle. |
| `prius_skew` | função | `(ctx, inputs, params)` | examples/electrodynamics/15_prius_skew.py: cogging, torque and ripple against the step skew, and the skewed back-EMF harmonics against the closed-form skew factor (the 3-D VTU export is left to the script). |
| `prius_tolerances` | função | `(ctx, inputs, params)` | examples/electrodynamics/17_prius_tolerances.py: torque and ripple spread from the tolerances of the magnet remanence, thickness, width and the air gap; standardised regression coefficients; probability within spec. |

#### `pinneapple_flow.nodes_ext.em_antenna`

Antenna measurement ranges (``examples/use_cases/antennas/compact_range_qz``): the quiet zone of a compact range by physical optics (:mod:`pinneapple_simulation.electromagnetics.reflector`), its validation against the closed form and geometrical optics, and the feed gain x reflector size x rolled …

<small>`pinneapple_flow/nodes_ext/em_antenna.py` · 67 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `compact_range` | função | `(ctx, inputs, params)` | The generic example (3 GHz, F = 1.5 m, 1.2 m section, 0.6 m quiet zone at 3 m): the reference design's quiet zone (ripple, taper, usable diameter, cross-polar), the validation and the design sweep. |

#### `pinneapple_flow.nodes_ext.em_prius`

The longer Prius-motor studies of ``examples/electrodynamics`` (12-14, 16) as nodes on :mod:`pinneapple_simulation.electromagnetics.machines`: rotor design (GP-surrogate NSGA-II against FEA-only), the thermal network over WLTC cycles with demagnetisation margins, the cross-check against gmsh + …

<small>`pinneapple_flow/nodes_ext/em_prius.py` · 315 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `rotor_design` | função | `(ctx, inputs, params)` | examples/electrodynamics/12_prius_rotor_design.py: hypervolume and FEA calls of the hybrid search against FEA-only, the hybrid front re-evaluated on a finer FEA, and the Prius rotor through the same evaluators. |
| `prius_thermal` | função | `(ctx, inputs, params)` | examples/electrodynamics/13_prius_thermal.py: magnet temperature sweep, a lumped thermal network driven by the WLTC losses (peak node temperatures), the time to 180 C at peak load, and demagnetisation margins per NdFeB grade at the hottest magnet temperature. |
| `getdp_crosscheck` | função | `(ctx, inputs, params)` | examples/electrodynamics/14_prius_getdp_crosscheck.py: torque against current angle, cogging, back-EMF and apparent inductances, PINNeAPPle's FEA against an independent gmsh + GetDP model of the same machine. |
| `end_effects` | função | `(ctx, inputs, params)` | examples/electrodynamics/16_prius_end_effects.py. |

#### `pinneapple_flow.nodes_ext.examples`

The node that checks a flow against the committed results of the example it reproduces (``examples/<name>``), and the helpers the example-domain nodes share (object artifacts, the checkout's ``examples`` folder).

<small>`pinneapple_flow/nodes_ext/examples.py` · 309 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Written` | classe | `(text)` | A reference number with the text it was written as (``1.40`` keeps its two decimals). |
| `save_object` | função | `(ctx, name, obj) -> dict[str, Any]` | A Python object (flight histories, wind fields, studio scenes) as a pickle artifact of this node. |
| `load_object` | função | `(v) -> Any` | The object of an artifact written by :func:`save_object` (only artifacts a node of this run wrote). |
| `examples_root` | função | `() -> Path \| None` | The checkout's ``examples`` folder (``$PINNEAPPLE_REPO/examples`` or next to the packages), None when this is an installed wheel without it. |
| `example_dir` | função | `(name) -> Path` |  |
| `doc_table_value` | função | `(ref) -> float` | ``doc:<path under docs/>\|<row key>\|<column header>``: the first number in that cell of a markdown table (the row whose first cell is the key; ``−`` and ``×`` are read as minus and nothing). |
| `reference_value` | função | `(folder, ref) -> float` | The number ``ref`` names in an example folder (see the module docstring). |
| `rounding_half_unit` | função | `(x) -> float` | Half a unit of the last digit of ``x`` as Python writes it (0.00147 -> 5e-6); 0 for a number written with 12 or more significant digits (not rounded). |
| `metric_value` | função | `(reports, name) -> float` | ``node.metric`` from the reports of the upstream nodes (a metric with its ``how`` is ``{"value": ...}``). |
| `compare` | função | `(ctx, inputs, params)` | Each comparison: the flow's measured number against the example's committed one, within the tolerance. |
| `flatten` | função | `(ctx, inputs, params)` | Every number of a nested result as a metric (``a.b.0.c``), so ``examples.compare`` can check the result of a node that returns its numbers in an output instead of recording them; strings and booleans are kept too. |

#### `pinneapple_flow.nodes_ext.flight`

Flight-dynamics nodes: the studies of ``examples/flight_dynamics`` (verification, trim and stability, take-off with crosswind and gust cases, turbulence encounters, the Pratt check, the same 1-cosine gust for every aircraft) as nodes on :mod:`pinneapple_simulation.flight`. Each study node takes the …

<small>`pinneapple_flow/nodes_ext/flight.py` · 588 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `aircraft` | função | `(ctx, inputs, params)` | The aircraft models (lattice wing and tail, fin, fuselage, gear, engines, mass properties with sources). |
| `verification` | função | `(ctx, inputs, params)` | The checks of examples/flight_dynamics/01_verification.py, metric names as in its manifest. |
| `trim_stability` | função | `(ctx, inputs, params)` | Trim at the cruise point, non-dimensional derivatives of the full model and the eigenvalues of the 12-state linearisation (examples/flight_dynamics/02_trim_and_stability.py). |
| `takeoff_cases` | função | `(ctx, inputs, params)` | examples/flight_dynamics/03_takeoff.py: ground roll, rotation, lift-off, climb over the 35 ft screen, Raymer's closed form, a crosswind and a down-gust case. |
| `turbulence_encounter` | função | `(ctx, inputs, params)` | examples/flight_dynamics/04_turbulence_encounters.py: each aircraft at its own speed through the same turbulence; dnz, roll and pitch excursions, cabin accelerations, as the mean over the realisations. |
| `pratt` | função | `(ctx, inputs, params)` | Simulated peak dn for a 12.5-MAC 1-cosine gust against the Pratt-Walker formula (and unalleviated). |
| `same_gust` | função | `(ctx, inputs, params)` | examples/flight_dynamics/07_same_gust.py: peak load factor at the CG and in the cabin, pitch and height excursions, the quasi-steady and Pratt estimates for this gust. |
| `scene` | função | `(ctx, inputs, params)` | Animated 3-D scenes of the kept flights (aircraft, lattice loading, n_z trail, the gust field), as three.js viewers (examples/flight_dynamics/08_viewers.py) and as the input of the Blender render nodes. |

#### `pinneapple_flow.nodes_ext.football`

Football nodes: the ball-flight inverse problem and the player-tracking smoothing check of ``examples/sports`` (simulated data, no download), and pass-outcome pitch control on a Metrica sample match (downloads it once).

<small>`pinneapple_flow/nodes_ext/football.py` · 76 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `football_ball_flight_inverse` | função | `(ctx, inputs, params)` | Simulated kicks (drag crisis, random spin) observed with noise are fitted back with the drag-Magnus ODE: errors of launch speed and drag coefficient, coverage of the Laplace 95 % intervals, and the Cramer-Rao comparison of 3-D against 2-D tracking (no ball height: launch elevation and spin are not … |
| `football_tracking_smoothing` | função | `(ctx, inputs, params)` | Simulated runs with known speed and acceleration, observed with noise: speed and metabolic-power error of raw differences, Savitzky-Golay and the constant-acceleration Kalman-RTS smoother (noise fitted by maximum likelihood), and the coverage of the smoother's 95 % velocity intervals. |
| `football_pass_pitch_control` | função | `(ctx, inputs, params)` | Spearman-type pitch control (fitted on one Metrica sample match) against logistic regression and boosted trees on geometric features, scored on the passes of the other match (log-loss, Brier, AUC, calibration). |

#### `pinneapple_flow.nodes_ext.forecasting`

Forecasting nodes: the ENSO solver checks of ``examples/enso_forecasting`` (no data needed) and the ``forecast_lab`` studies of ``examples/near_random_forecasting`` (one case: predictability, model zoo, surrogate-trained twins; the overfitting study; the Mackey-Glass Lyapunov convergence). The ENSO …

<small>`pinneapple_flow/nodes_ext/forecasting.py` · 201 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `enso_solver_checks` | função | `(ctx, inputs, params)` | The intermediate coupled model's shallow-water solver (Kelvin and Rossby wave speeds, convergence order, the Gill steady response against the analytic solution) and the recharge / delayed oscillators against their eigenvalues (examples/enso_forecasting/run_study.py, results/solver_checks.json). No … |
| `forecast_case` | função | `(ctx, inputs, params)` | examples/near_random_forecasting/run_study.py for one case: predictability first, then every model under the purged walk-forward protocol with an untouched test period. |
| `lyapunov` | função | `(ctx, inputs, params)` | Benettin estimate of the largest Lyapunov exponent of Mackey-Glass (tau = 17) against the step, the length of the run and the seed (results/lyapunov_mackey_glass.json). |
| `overfitting` | função | `(ctx, inputs, params)` | Shuffled k-fold, walk-forward without purge, purged walk-forward and purged k-fold scores of deliberately overfitting and regularised variants, against the untouched test score. |

#### `pinneapple_flow.nodes_ext.geomechanics`

Geomechanics, inspection and component-library nodes: the granular column collapse of ``examples/geomechanics`` (:class:`pinneapple_simulation.particle_dynamics.geo_mpm.GeoMPM`, MLS-MPM with Drucker-Prager sand, against the runout law of Lube et al. 2005), the eddy-current inspection PINN of …

<small>`pinneapple_flow/nodes_ext/geomechanics.py` · 169 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `lube_2005` | função | `(a) -> float` | Lube et al. (2005) two-dimensional collapse runout (L_inf - L0) / L0: 1.2 a below a = 1.8, 1.9 a^(2/3) above 2.8, the mean of both in the transition (as examples/geomechanics/01_granular_column_collapse.py). |
| `column_collapse` | função | `(ctx, inputs, params)` | Columns of dry cohesionless sand against a back wall on a rough floor, released at t = 0; the final runout against the law Lube et al. measured, the final height (examples/geomechanics/01_granular_column_collapse.py). |
| `eddy_current` | função | `(ctx, inputs, params)` | A synthetic probe scan from the axisymmetric complex-Helmholtz FDM solver, then an InspectionPINN trained on it (examples/inspection/01_eddy_current_pinn.py). |
| `component_fit` | função | `(ctx, inputs, params)` | examples/systems/01_component_library_toolbox.py: the registered components and toolboxes, one component built and fitted. The example fits random targets (a demonstration of the API, not of the physics); this node seeds them. |

#### `pinneapple_flow.nodes_ext.lbm`

Lattice Boltzmann nodes (``examples/lbm/01_sphere_wind_tunnel.py``): drag on a sphere in the 3-D wind tunnel of :class:`pinneapple_simulation.numerical_solvers.lbm_tunnel.WindTunnelLBM`, against Stokes' law (:func:`~pinneapple_simulation.numerical_solvers.lbm_tunnel.stokes_sphere_field` far field …

<small>`pinneapple_flow/nodes_ext/lbm.py` · 153 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `sphere_stokes` | função | `(ctx, inputs, params)` | A sphere in a cube whose six sides hold the analytic Stokes far field, started from that field; the momentum-exchange drag against F = 6 pi rho nu R U. |
| `sphere_drag` | função | `(ctx, inputs, params)` | Inlet at x-, outlet at x+, open sides; Cd from the mean of the last force samples, against Schiller & Naumann (1933); with both storages, the relative Cd difference and the largest velocity difference. |

#### `pinneapple_flow.nodes_ext.learned_sim`

Learned mesh simulators on public data (``examples/meshgraphnet/02_cylinder_flow_deepmind``): MeshGraphNet on a byte prefix of DeepMind's ``cylinder_flow`` TFRecords (:mod:`pinneapple_data.datasets.deepmind_mgn`), trained and rolled out with …

<small>`pinneapple_flow/nodes_ext/learned_sim.py` · 112 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `mgn_cylinder_flow` | função | `(ctx, inputs, params)` | Next-step velocity of vortex shedding behind a cylinder on irregular triangle meshes, rolled out autoregressively on the validation trajectories; against the velocity frozen at t = 0. |

#### `pinneapple_flow.nodes_ext.non_hermitian`

Non-Hermitian physics nodes (``examples/non_hermitian/ep_tuning.py``): the no-go theorem for exceptional-point tuning in PT-symmetric dimers (Zenodo 23287622) checked symbolically (:func:`pinneapple_physics.non_hermitian.analyze_ep_tuning` on …

<small>`pinneapple_flow/nodes_ext/non_hermitian.py` · 177 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ep_laws` | função | `(ctx, inputs, params)` | Per coupling class: the EP law gamma_EP^2(lam), whether it equals the linear target J^2 + lam J G exactly, its degree in lam, the four assumptions (A PT symmetry, B dimension 2, C polynomial coupling, D reciprocity) and whether the theorem applies. |
| `ep_numeric` | função | `(ctx, inputs, params)` | gamma_EP^2 found from the Hamiltonian alone (route C: closed form, the matrix is not PT symmetric) against the target J^2 + lam J G, and the route-B feedback fixed point. |
| `hypothesis_map` | função | `(ctx, inputs, params)` | Three designs audited against the theorem's assumptions: all VALID means linear tuning is out of reach; a VIOLATED assumption names the route that opened up. |
| `ep_figure` | função | `(ctx, inputs, params)` | The four-panel figure (spectrum across the EP, EP laws by coupling class, coupling plane, chains) and, for uniform chains with \|b\| = 1, the PT-breaking threshold kappa_N^2 = gamma_EP^2 and the order of the EP there. |

#### `pinneapple_flow.nodes_ext.pde_env`

The PDE environment examples (``examples/pde_environment``): a ``selector_type="tag"`` preset trained through ``solve_pde`` on its analytic real-geometry fixture, with the per-tag loss terms before and after training (``07_tagged_presets_real_geometry``).

<small>`pinneapple_flow/nodes_ext/pde_env.py` · 76 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `tagged_preset` | função | `(ctx, inputs, params)` | Per-tag loss terms of a tagged preset before and after training on its real geometry. The boundary points per tag come from the preset's box / cylinder fixture, a modified MLP is trained on them with ``solve_pde``, and every compiled loss term (``pde``, ``bc_<tag>``, ``total``) is reported before … |

#### `pinneapple_flow.nodes_ext.pinn_examples`

The one-call solver examples: ``examples/first_example`` (a Burgers PINN against the Cole-Hopf solution with ``pp.solve``) and ``examples/solver_api`` (a classical FEM and a PINN on the manufactured Poisson problem with ``pp.compare``).

<small>`pinneapple_flow/nodes_ext/pinn_examples.py` · 99 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `burgers_first` | função | `(ctx, inputs, params)` | examples/first_example/first_example.py: u_t + u u_x = nu u_xx on [-1, 1]; the PINN's profile at t = 0.75 (the steep front) against the Cole-Hopf closed form. |
| `poisson_fem_vs_pinn` | função | `(ctx, inputs, params)` | examples/solver_api/01_classical_vs_pinn.py: laplace(u) = f on the unit square with u = sin(pi x) sin(pi y), one call for both methods against the exact solution. |

#### `pinneapple_flow.nodes_ext.pinn_solver`

The PINN-solver examples (``examples/pinn_solver``): the symbolic loss of ``PINNFactory`` evaluated on a field (``01_symbolic_loss_basic``). The compiler examples (Poisson 2-D, Burgers 1-D, wave 1-D) and the inverse heat problem are flows of the generic equations / train.pinn / inverse.discover …

<small>`pinneapple_flow/nodes_ext/pinn_solver.py` · 57 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `factory_loss` | função | `(ctx, inputs, params)` | A PINNFactory loss evaluated for the zero field. The residual strings are compiled by SymPy into a torch loss and evaluated at random collocation points: the total and each component. |

#### `pinneapple_flow.nodes_ext.problemdesign`

Problem design without an LLM (``examples/problem_designer/03_offline_spec_to_report``): a partly known :class:`pinneapple_problemdesign.schema.ProblemSpec`, its validators (warnings and safe assumptions), the FNO-first plan and the Markdown / JSON design report.

<small>`pinneapple_flow/nodes_ext/problemdesign.py` · 84 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `offline_report` | função | `(ctx, inputs, params)` | Validators, plan and report of a partly known spec, as in the example (gaps empty). |

#### `pinneapple_flow.nodes_ext.render`

Render nodes for animated studio scenes: Blender Cycles clips and stills (:func:`pinneapple_tools.visualization.studio.blender.render_animation` / ``render_still``) and a labelled grid of clips with one colour bar (:mod:`pinneapple_tools.visualization.studio.compose`). They take the ``anim_scene`` …

<small>`pinneapple_flow/nodes_ext/render.py` · 161 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `blender_clip` | função | `(ctx, inputs, params)` | Cycles (CPU, denoised) clip of each scene with the HUD burnt in, encoded to MP4, plus a still. |
| `clip_grid` | função | `(ctx, inputs, params)` | The clips side by side (e.g. the four aircraft through the same gust) with labels and one colour bar; the still grid uses each clip's marked frame (the gust peak). |

#### `pinneapple_flow.nodes_ext.render_worker`

Blender rendering in its own process, for :mod:`pinneapple_flow.nodes_ext.render`: ``bpy`` is never imported by the flow engine or the Flow Studio server (a ``bpy`` wheel built against another numpy can crash the process that imports it, and Cycles holds the interpreter for minutes).

<small>`pinneapple_flow/nodes_ext/render_worker.py` · 74 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check` | função | `() -> int` |  |
| `render` | função | `(scenes_pkl, options_json, out_dir) -> int` |  |
| `main` | função | `(argv) -> int` |  |

#### `pinneapple_flow.nodes_ext.security`

Cyber-physical security (``examples/security/02_physics_residual_fdi``): physics residuals against false data injection on the three-tank water network, against three data-driven detectors (:func:`pinneapple_security.cps_bench.run_benchmark`).

<small>`pinneapple_flow/nodes_ext/security.py` · 70 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `fdi_benchmark` | função | `(ctx, inputs, params)` | Per detector, regime and attack: episode AUC, detection rate, median delay (minutes, median over the seeds' medians) and localisation, averaged over the seeds as the example script prints them; false alarms per day on clean data. The isolation forest needs scikit-learn (skipped by the library … |

#### `pinneapple_flow.nodes_ext.solver_registry`

Nodes over the solver registry (``examples/solvers``): every solver built by name through :class:`pinneapple_simulation.numerical_solvers.SolverRegistry` after :func:`~pinneapple_simulation.numerical_solvers.register_all` -- the time-series decompositions (EEMD, VMD, wavelet, STL, SSA, …

<small>`pinneapple_flow/nodes_ext/solver_registry.py` · 211 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `decompose` | função | `(ctx, inputs, params)` | A sum of sines plus Gaussian noise decomposed by each registered solver (examples/solvers/ 01_time_series_decompositions.py, with a seeded noise draw). Per solver: the output shape and, for the additive decompositions (EEMD, CEEMDAN, VMD, SSA, STL), how far the sum of the components is from the … |
| `sph_box` | função | `(ctx, inputs, params)` | examples/solvers/02_sph_box.py: the trajectory, the final bounding box and the fraction of particles still inside the box (a warning when particles leave it: the run is then not a valid fluid simulation). |
| `poisson_fdm_spectral` | função | `(ctx, inputs, params)` | examples/solvers/03_poisson_solvers.py: the source f with u = 0 on the boundary (SOR) and with periodic boundaries (FFT), their mean squared difference as the script prints it, and each against the exact solution of its own equation: the registry's FDM solver solves lap(u) = -f (u = f / (2 (k … |

#### `pinneapple_flow.nodes_ext.solvers`

Numerical-solver nodes from ``examples/numerical_solvers``: the NVE Lennard-Jones gas (:class:`pinneapple_simulation.particle_dynamics.molecular_dynamics.MolecularDynamicsSimulator`, velocity Verlet, periodic box) and the shallow-water dam break with a gate and an obstacle …

<small>`pinneapple_flow/nodes_ext/solvers.py` · 172 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `lj_gas` | função | `(ctx, inputs, params)` | Particles on a lattice at the LJ well separation with a small random kick; velocity-Verlet steps; total energy against its initial value (examples/numerical_solvers/12_molecular_dynamics_lj_gas.py). |
| `dam_break` | função | `(ctx, inputs, params)` | examples/numerical_solvers/13_shallow_water_twin3d_browser_demo.py without the Twin3D export and its blocking server: the run and its health report (volume conservation, positivity, finiteness). |
| `crate_and_pendulum` | função | `(ctx, inputs, params)` | examples/rigid_body/newton_quickstart.py: where the crate stops against Coulomb friction, v0^2 / (2 mu g). |

#### `pinneapple_flow.nodes_ext.structures`

Structures nodes: CalculiX (``examples/calculix_cantilever``) through :mod:`pinneapple_simulation.external_solvers.calculix`: a deck (the example's committed one or a cantilever written by the library), ``ccx`` (native or its Docker image), and the results of an ``.frd`` file (the run's or the …

<small>`pinneapple_flow/nodes_ext/structures.py` · 146 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `deck` | função | `(ctx, inputs, params)` | The input deck of the cantilever (mesh, material, fixed root, tip load). |
| `solve` | função | `(ctx, inputs, params)` |  |
| `results` | função | `(ctx, inputs, params)` | Tip deflection (the most negative u_z at the free end), the largest \|sigma_xx\|, and the Euler-Bernoulli tip deflection P L^3 / (3 E I) of the same beam. |

#### `pinneapple_flow.nodes_ext.surrogate_inverse`

Surrogate-driven inverse design (:mod:`pinneapple_design.surrogate_inverse`, the five design tools of ``apps/*_ai``): sample the design space, label it with a closed form, train an MLP, descend through the frozen surrogate and re-check the final design against the closed form (verification card and …

<small>`pinneapple_flow/nodes_ext/surrogate_inverse.py` · 59 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `surrogate_inverse` | função | `(ctx, inputs, params)` | Train the case's surrogate, optimise the design through it, verify it against the closed form. |

#### `pinneapple_flow.nodes_ext.use_cases`

Use-case nodes (``examples/use_cases``): the pin-fin inverse problem of ``fin_convection_inverse`` -- the convection coefficient h from five noisy thermocouples, by a PINN with h as a trainable parameter (:class:`pinneapple_physics.pinn_solver.factory.pinn_factory.PINNFactory`) and by a …

<small>`pinneapple_flow/nodes_ext/use_cases.py` · 137 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `pin_fin` | função | `(ctx, inputs, params)` | examples/use_cases/fin_convection_inverse/fin_convection_inverse.py: theta'' - m^2 theta = 0, theta(0) = 1, -k theta'(L) = h theta(L); h and the field learned together from the fin equation and the readings, against a least-squares fit of the closed form; the heat the fin dissipates from the … |

#### `pinneapple_flow.nodes_ext.validation`

Validation nodes for ``examples/physics_check``, ``examples/vs_physicsnemo`` (07, the check replay), ``examples/model_zoo``, ``examples/proof_graph`` and ``examples/validation_arena``: the physics linter (:func:`pinneapple_check.run`) against the rules each fixture must break, the zoo's evaluate / …

<small>`pinneapple_flow/nodes_ext/validation.py` · 213 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `physics_check` | função | `(ctx, inputs, params)` | Run every rule on the declared targets (OpenFOAM cases, fields, QoI tables, surrogates, models) and compare the rules each target breaks with the example's list (examples/physics_check, the CI gate's fixtures). |
| `zoo_evaluate` | função | `(ctx, inputs, params)` | examples/model_zoo/quickstart.py: the card's held-out metrics reproduced, the physics checks and validity range, the comparison with XFOIL, and a guarded fine-tune. |
| `pt_chain` | função | `(ctx, inputs, params)` | A graph of assumptions, lemmas and a theorem with exact (sympy) and numerical checks: does the no-go theorem for linear exceptional-point tuning extend to PT tetramers (examples/proof_graph/pt_chain_no_go.py)? |
| `arena` | função | `(ctx, inputs, params)` | The red team looks for wrong outputs that pass every check (escapes) and correct ones that fail (false alarms); the blue team calibrates a combined test and adds checks until every stored escape is caught (examples/validation_arena/run_arena.py). |
| `audit` | função | `(ctx, inputs, params)` | The reference solver itself judged by the shipped defence: it should have no escapes and no false alarms (examples/validation_arena/external_model.py). |

#### `pinneapple_flow.nodes_ext.validation_gallery`

The validation gallery (``examples/validation``) as flows: a model of one family on one physics, validated with ``pp.validate``, and a broken variant of it that one named check must catch (:func:`pinneapple_analysis.validation.gallery.conclude`: report, trust card, SARIF and the failure record).

<small>`pinneapple_flow/nodes_ext/validation_gallery.py` · 197 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `pinn_heat` | função | `(ctx, inputs, params)` | -lap T = q on the unit square, T = 0 on the walls (manufactured source): the PINN as trained and the same PINN trained with boundary weight 0, both through ``pp.validate(net, {"pinn": problem})``; ``pinn.boundary`` must catch the second. |
| `pod_dmd_wake` | função | `(ctx, inputs, params)` | POD + DMD fitted on the saturated wake (good) and on the growing transient (broken), rolled out from the same frame against the LBM reference; ``rollout.invariants`` must catch the amplifying ROM. |
| `opinf_advection` | função | `(ctx, inputs, params)` | Periodic advection-diffusion of a tracer: ROMs with the full basis (good) and cut at ``energy_bad`` (broken), rolled out against the exact solution with mass conserved; ``rollout.spectrum`` must catch the smearing. |
| `zoo_range` | função | `(ctx, inputs, params)` | The zoo model on test designs of its dataset (good) and on two of them moved to a velocity and power outside the card's validity range (broken); ``applicability.range`` must catch the second. |

#### `pinneapple_flow.nodes_ext.viz`

Any result in 3-D (``examples/studio/01_any_result_in_3d``): a file ``pinneapple_data.cae`` reads (OpenFOAM case or zip, CalculiX .frd, Abaqus .inp, Gmsh, VTK) or STL / OBJ geometry, as a :class:`pinneapple.viz.Scene`, exported to glTF and USD, an interactive three.js viewer and, when bpy is …

<small>`pinneapple_flow/nodes_ext/viz.py` · 92 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `any_result_3d` | função | `(ctx, inputs, params)` | examples/studio/01_any_result_in_3d.py: load, export, view, render. |

#### `pinneapple_flow.nodes_ext.worldmodel`

Physics world model (``examples/worldmodel/01_generate_train_evaluate``): trajectory datasets from the built-in scenarios (:func:`pinneapple_worldmodel.generate_datasets`), one specialist FNO world model per scenario (:func:`pinneapple_worldmodel.train_specialist`) and the library's benchmark tasks …

<small>`pinneapple_flow/nodes_ext/worldmodel.py` · 69 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `specialists` | função | `(ctx, inputs, params)` | Per scenario: parameters of the specialist and, per benchmark task of that scenario, next-step and rollout RMSE, relative L2, the divergence step and the overall score. |

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

_Nenhum exemplo encontrado no repositório para este pacote._

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 31 de 89.
- Funções de teste em arquivos que importam algum módulo do pacote: 263.

### Limitações conhecidas

- _Nenhuma registrada._

## 4. `pinneapple_hub`

### Objetivo

_Sem texto de objetivo._

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 31 | 7.260 | 22 | 170 | 183 (18 arquivos) | 8 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `hub` |  | 1 | 2 | 0 |
| `library` |  | 10 | 64 | 22 |
| `model_card` |  | 1 | 1 | 21 |
| `verified_card` |  | 1 | 4 | 6 |
| `zoo` |  | 17 | 121 | 125 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_hub`

Model hub client: ``push_to_hub``/``from_pretrained`` + ``ModelCard``.

<small>`pinneapple_hub/__init__.py` · 12 linhas · testes: 122</small>

#### `pinneapple_hub.hub`

``push_to_hub`` / ``from_pretrained`` for PINNeAPPle models, built on the Hugging Face Hub's own storage/auth/versioning infrastructure rather than a new hosting service this repository would have to build, operate, and secure itself. This is a deliberate, common pattern (many domain-specific …

<small>`pinneapple_hub/hub.py` · 151 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `push_to_hub` | função | `(model, repo_id, model_card, *, architecture=None, architecture_config=None, private=Fals…)` | Upload a trained model's weights + architecture config + model card to a Hugging Face Hub model repo. |
| `from_pretrained` | função | `(repo_id, *, revision='main', device='cpu', token=None)` | Download and reconstruct a model pushed with :func:`push_to_hub`. |

#### `pinneapple_hub.library`

The PINNeAPPle Hub index: every ready-made thing in the repository -- tools (public functions, Flow Studio nodes, recipes, MCP tools), trained models, geometries, datasets, flows and examples -- as one generated list, each item with a runnable snippet whose recorded run is checked by a test.

<small>`pinneapple_hub/library/__init__.py` · 303 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `collect` | função | `(only=None) -> list[dict[str, Any]]` | Items from every source (:data:`.sources.COLLECTORS`), ids unique, sorted by kind then title. |
| `run_file` | função | `(item_id, runs=RUNS) -> Path` | The file holding the recorded run of one item: ``snippet_runs/<id with unsafe characters as _>.json``. |
| `read_runs` | função | `(path=RUNS) -> dict[str, Any]` | Every recorded run, by item id (each file under ``snippet_runs/`` carries its item's ``id``). |
| `write_run` | função | `(item_id, rec, runs=RUNS) -> Path` |  |
| `attach_runs` | função | `(items, runs) -> None` | Set ``run`` on each item: the recorded run of its current snippet (same code hash), else the module reference's own run of a recipe, else ``{"status": "not run"}``. |
| `facets` | função | `(items) -> dict[str, Any]` |  |
| `quality_key` | função | `(it) -> str` | The filter value of an item's quality: a Lab letter, a zoo decision, or "not graded". |
| `featured` | função | `(items, n=8) -> list[str]` | Highlights, by a fixed rule: approved zoo models, A-sealed examples with figures, geometries with a thumbnail, then flows; each with a passing recorded run. |
| `build` | função | `(only=None, runs=None) -> dict[str, Any]` | The whole index (not written; see :func:`write_index`). |
| `write_index` | função | `(index, path=INDEX) -> Path` |  |
| `load_index` | função | `(path=INDEX, *, rebuild=False) -> dict[str, Any]` | The index at ``path``. The default index is a cache: built from the sources (and written) when it is missing or ``rebuild`` is set; rebuild it after changing a source with ``python -m pinneapple_hub.library build``. |
| `get` | função | `(index, item_id) -> dict[str, Any]` |  |
| `score` | função | `(it, query) -> float` | Relevance of an item to a query; 0 = no match. Every query word must match: title weighs most, then name and tags, then summary, domain and description. |
| `search` | função | `(index, query='', *, kind=None, domain=None, quality=None, subkind=None, category=None, l…)` | Items matching ``query`` and the filters (``quality``: a letter, a comma list of letters, a zoo decision in lower case, or "not graded"), best first; without a query, in index order. |
| `record_runs` | função | `(items, *, stale_only=False, timeout=300.0, ids=None, workers=4, log=print) -> dict[str, …)` | Run the snippets (all, the ones not recorded for their current code, or ``ids``) in ``workers`` parallel interpreters, save their figures under ``data/web/hub/figures`` and write one ``snippet_runs/<id>.json`` each. Recipes the module reference already ran are reused. |
| `check` | função | `(index) -> list[str]` | Problems of an index: a snippet without a passing run of its current code, a file or link path that does not exist in the repository, an example id that is not an item. |

#### `pinneapple_hub.library.__main__`

``python -m pinneapple_hub.library build | run [--stale] [ids ...] | check`` (see the package docstring).

<small>`pinneapple_hub/library/__main__.py` · 47 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `main` | função | `(argv=None) -> int` |  |

#### `pinneapple_hub.library.categories`

The 20 task categories of the Hub's tools. Every tool gets exactly one, by the longest rule prefix that matches its key; the key names where the tool comes from (``node:<flow node type>``, ``lib:<module>:<function>`` for a generated library call, ``fn:<dotted name>``, ``mcp:<agent tool>``). Recipes …

<small>`pinneapple_hub/library/categories.py` · 174 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `key` | função | `(it) -> str` | Where a tool comes from, as the prefix rules read it. |
| `category_of` | função | `(it) -> str \| None` | The slug of a tool's category, or None when no rule matches (a test forbids that). |
| `table` | função | `() -> list[dict[str, str]]` |  |

#### `pinneapple_hub.library.compose`

"Create your own": a custom tool composed from existing blocks (Flow Studio node types, optionally a bundled flow to start from and a library geometry), returned as a flow document and the Python that runs it.

<small>`pinneapple_hub/library/compose.py` · 191 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ComposeError` | classe | `()` |  |
| `blocks` | função | `() -> list[dict[str, Any]]` | The blocks the composer offers: every node type of Flow Studio's palette (curated nodes first, then the generated library calls), by palette section, with its ports and parameters. |
| `compose` | função | `(spec) -> dict[str, Any]` | See the module doc. ``spec``: ``name``, ``description``, ``blocks`` ([{type, params}]), optional ``start`` (a bundled flow name to append the blocks to) and ``geometry`` (a geometry library entry the code loads). |
| `code_for` | função | `(doc, *, geometry='', workdir='hub_tool', expose=None) -> str` | Python that rebuilds the flow, validates it, runs it headless and prints what each node measured; exposed parameters are variables set before the run. |

#### `pinneapple_hub.library.flowlink`

The Hub and Flow Studio on one registry: a ready flow around any node type, and links between the two apps.

<small>`pinneapple_hub/library/flowlink.py` · 173 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FlowLinkError` | classe | `()` |  |
| `bundled_flows_using` | função | `(type_) -> list[str]` | Names of the bundled flows that contain a node of ``type_``. |
| `flow_for_node` | função | `(type_, params=None, *, report=True) -> dict[str, Any]` | See the module doc. Raises :class:`FlowLinkError` for an unknown type; parameter problems and unwired inputs come back in ``issues``. |
| `b64url` | função | `(obj) -> str` |  |
| `from_b64url` | função | `(text) -> Any` |  |
| `flow_link` | função | `(flow, *, base) -> str` | Flow Studio URL that opens ``flow`` (``?flow=``). |
| `add_link` | função | `(type_, params=None, *, base) -> str` | Flow Studio URL that adds a node of ``type_`` (with ``params``) to a flow the user picks (``?add=``). |

#### `pinneapple_hub.library.flowruns`

Recorded runs of the bundled flows (``pinneapple_flow/flows``), shown in the Hub on each flow's page and on the page of every node type the flow uses.

<small>`pinneapple_hub/library/flowruns.py` · 109 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `flow_hash` | função | `(name) -> str` |  |
| `read_flow_runs` | função | `(path=FLOW_RUNS) -> dict[str, Any]` |  |
| `current` | função | `(runs, name) -> dict[str, Any] \| None` | The recorded run of ``name`` if it was recorded for the flow file as it is now. |
| `record` | função | `(names=None, *, timeout=900.0, workers=2, log=print) -> dict[str, Any]` | Run the bundled flows (all, or ``names``) and write ``flow_runs.json``. |
| `runs_for_node` | função | `(runs, type_) -> list[dict[str, Any]]` | Every node of ``type_`` in a current recorded flow run: flow, node id, status, metrics, summary. |

#### `pinneapple_hub.library.myhub`

"My Hub": the tools a person composed and saved, as Hub items in a local folder (one JSON file each).

<small>`pinneapple_hub/library/myhub.py` · 87 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MyHubError` | classe | `()` |  |
| `MyHub` | classe | `(root)` |  |

#### `pinneapple_hub.library.sandbox`

Run a code snippet in a short-lived, separate interpreter and keep what it printed and drew.

<small>`pinneapple_hub/library/sandbox.py` · 104 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `code_hash` | função | `(code) -> str` | Short SHA-256 of a snippet: a recorded run is current only while its snippet has this hash. |
| `run_snippet` | função | `(code, *, timeout=60.0, memory_mb=None, cpu_s=None, max_figures=4, env=None) -> dict[str,…)` | Run ``code`` (see the module doc). Returns ``ok``, ``stdout``, ``stderr`` (last lines), ``seconds``, ``timed_out``, ``code_hash`` and ``figures`` (a list of PNG ``bytes``, at most ``max_figures``). |

#### `pinneapple_hub.library.snippets`

The minimal, runnable code of each Hub item. Every snippet built here is run by ``python -m pinneapple_hub.library run`` (output recorded in ``data/web/hub/snippet_runs/``) and again by ``tests/test_hub_library.py``; keep them short, offline and fast.

<small>`pinneapple_hub/library/snippets.py` · 121 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `model` | função | `(name, example) -> str` |  |
| `dataset` | função | `(name, split) -> str` |  |
| `geometry` | função | `(name) -> str` |  |
| `flow` | função | `(name) -> str` |  |
| `node` | função | `(type_) -> str` |  |
| `auto_call` | função | `(target, node_type, defaults=None) -> str` | A generated library call that runs with its defaults (the node's parameter form): call it with them and show what it returns. |
| `api` | função | `(dotted) -> str` |  |
| `mcp` | função | `(name, *, call) -> str` |  |
| `example` | função | `(folder) -> str` |  |
| `lab_item` | função | `(key) -> str` |  |
| `symbols` | função | `(code) -> list[str]` | ``pp.xxx`` names a snippet calls (for tags and links). |

#### `pinneapple_hub.library.sources`

Where the Hub's items come from. Each collector reads one existing source of the repository and returns items in the Hub format (see :mod:`pinneapple_hub.library`); nothing here is written by hand.

<small>`pinneapple_hub/library/sources.py` · 579 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `rel` | função | `(p) -> str` |  |
| `domain` | função | `(name, tags=()) -> str` | The physical domain, by the Lab's own rule (``pinneapple_lab.manifest.domain_of``): on the first word of ``name`` (the item's own name) when that decides it, else on the whole text and the tags. |
| `used_by_examples` | função | `(*needles) -> list[str]` | Example folders whose scripts contain one of ``needles`` as a quoted string or a call. |
| `models` | função | `() -> list[dict[str, Any]]` |  |
| `datasets` | função | `() -> list[dict[str, Any]]` |  |
| `geometries` | função | `() -> list[dict[str, Any]]` |  |
| `flows` | função | `() -> list[dict[str, Any]]` |  |
| `nodes` | função | `() -> list[dict[str, Any]]` | Every node type of Flow Studio's palette (``pinneapple_flow.REGISTRY``: the curated nodes and the generated library calls), so the palette and the Hub list the same tools from the same registry. |
| `node_for_callable` | função | `() -> dict[str, str]` | ``module:function`` -> the node type that wraps it (generated library calls by their target, curated nodes by the ``module:function`` or ``module.function`` they declare in ``calls``). |
| `recipes` | função | `() -> list[dict[str, Any]]` |  |
| `apis` | função | `() -> list[dict[str, Any]]` | One item per row of AGENTS.md's "Use these APIs first" table, for the functions that row names. |
| `agent_tools` | função | `() -> list[dict[str, Any]]` | The MCP server's tools, read from the source (``@server.tool()`` functions). |
| `examples` | função | `() -> list[dict[str, Any]]` |  |

#### `pinneapple_hub.model_card`

``ModelCard``: the reproducibility contract every hub-pushed model carries.

<small>`pinneapple_hub/model_card.py` · 180 linhas · testes: 21</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ModelCard` | classe | `(name, description, task, version, inputs, outputs, validity_range, known_limitations, ar…)` |  |

#### `pinneapple_hub.verified_card`

Model cards whose ``validation_metrics`` come only from a guardrail run against the model, refused otherwise.

<small>`pinneapple_hub/verified_card.py` · 122 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CardRefused` | classe | `(problems)` | The card built from a guardrail run did not pass ``ModelCard.validate()``; ``.problems`` lists why. |
| `validation_metrics_from_report` | função | `(report) -> dict[str, float]` | ``{check name: value}`` for numeric checks, ``{<name>_passed: 1.0 \| 0.0}`` for pass/fail-only checks. |
| `card_from_guardrail` | função | `(spec, report, *, name, preset, preset_kwargs=None, architecture, architecture_config, tr…)` | A ``ModelCard`` for a model of preset ``preset`` from its guardrail ``report`` (not validated here). |
| `certify_model` | função | `(preset, name, *, preset_kwargs=None, architecture='modified_mlp', architecture_config=No…)` | Train (or load ``state_dict``: a dict or ``torch.save`` bytes) a model of ``preset``, check it, build its card. |

#### `pinneapple_hub.zoo`

``pp.zoo``: pre-trained, verified PINNeAPPle models in one line -- load, predict, evaluate against the stored reference, validate (physics checks + validity range), benchmark and fine-tune with a held-out guard.

<small>`pinneapple_hub/zoo/__init__.py` · 341 linhas · testes: 125</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FinetuneRejectedError` | classe | `()` | The fine-tuned model is worse than the original on the original test set (pass ``force=True`` to save it). |
| `FinetuneResult` | classe | `(model, original, before, after, accepted, tolerance, history, new_data_metrics, saved_to)` |  |
| `export_onnx` | função | `(name, path, *, checks=False, check_parity=True, offline=None) -> dict[str, Any]` | Export a model to ONNX (pre/post-processing included; parity with PyTorch checked to 1e-5). See :func:`pinneapple_hub.zoo.export.export_onnx`. |
| `onnx_parity` | função | `(name, onnx_model, X=None, *, offline=None) -> float` | Largest scaled difference between an ONNX graph and the PyTorch model (:mod:`pinneapple_hub.zoo.export`). |
| `query_checks` | função | `(name, inputs=None, *, offline=None, **named) -> builtins.list` | Per-query trust checks and verdict for each input row (:mod:`pinneapple_hub.zoo.checks`). |
| `check_spec` | função | `(name, *, offline=None) -> dict[str, Any]` | The declarative spec of a model's query checks (:func:`pinneapple_hub.zoo.checks.check_spec`). |
| `export_web` | função | `(name, directory, *, offline=None) -> dict[str, Any]` | ONNX graph with query checks + JSON card/spec for the browser (:mod:`pinneapple_hub.zoo.web`). |
| `list` | função | `(task=None) -> builtins.list` | Names of the shipped models (optionally only those whose task or tags mention ``task``). |
| `info` | função | `(name, *, offline=None, version=None) -> ModelCard` | The model card (``ModelCard``): task, inputs/outputs with units, validity range, metrics against the reference, training data, licence, citation and the stored trust report. ``print(card.to_markdown())``. |
| `load` | função | `(name, *, offline=None, version=None) -> ZooModel` | A ready-to-use predictor (weights verified against the manifest's SHA-256). ``version`` loads a superseded release (see :func:`versions`; a ``SupersededModelWarning`` says why it was superseded). |
| `versions` | função | `(name) -> builtins.list[dict[str, str]]` | Every release of a model, newest first, with the reason each older one was superseded. |
| `predict` | função | `(name, inputs=None, **named) -> dict[str, np.ndarray]` | ``load(name).predict(...)`` in one call. |
| `catalogue` | função | `() -> str` | Markdown page of every model and dataset: metrics, trust decision, size, Hugging Face link (``docs/guides/zoo_catalogue.md`` is this page). |
| `status` | função | `(name) -> dict[str, str]` | Where each file of a model resolves from right now (bundled \| cached \| remote), without downloading. |
| `evaluate` | função | `(name, dataset=None, *, split=None, offline=None, rtol=None) -> dict[str, Any]` | Metrics of a model against reference cases. |
| `validate` | função | `(name, inputs=None, *, offline=None)` | Trust report (``pinneapple_analysis.trust.TrustReport``) of a model: its card metrics reproduced on the stored reference, the acceptance limits of the zoo gate, the model's physics checks and -- if ``inputs`` are given -- whether they are inside the validity range (an out-of-range warning is raised … |
| `finetune` | função | `(name, new_data, *, epochs=200, lr=0.0001, batch=256, weight_decay=0.0, tolerance=0.02, s…)` | Fine-tune a copy of a zoo model on ``new_data`` (``(inputs, outputs)``, a dataset split or a mapping of columns; NaN outputs are ignored) and check it on the model's ORIGINAL held-out test set. |
| `benchmark` | função | `(name, *, budget_s=60.0, seeds=1, baselines=('retrain',), out_dir='zoo_benchmark', force=…)` | Benchmark a zoo model with ``pp.benchmark``: the shipped model against the same architecture retrained from scratch on the same training split under ``budget_s`` seconds (CPU), on the same held-out test set, with ``pp.benchmark``'s gates, report and trust card. Returns its ``BenchmarkResult``. … |

#### `pinneapple_hub.zoo._artifacts`

Artifact store of the zoo: manifest, SHA-256 checks, local cache and lazy download from the Hugging Face Hub.

<small>`pinneapple_hub/zoo/_artifacts.py` · 237 linhas · testes: 81</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ZooError` | classe | `()` | Base class of the zoo's errors. |
| `ArtifactNotPublishedError` | classe | `()` | The artifact is in the manifest but not (yet) on the Hugging Face Hub. |
| `ArtifactUnavailableError` | classe | `()` | The artifact is neither bundled nor cached and cannot be downloaded (offline or network error). |
| `ChecksumMismatchError` | classe | `()` | A file's SHA-256 differs from the manifest: corrupt, truncated or tampered with. |
| `sha256_file` | função | `(path, chunk=1 << 20) -> str` |  |
| `manifest` | função | `() -> dict[str, Any]` | The parsed ``manifest.json`` (re-read when the file changes). |
| `cache_dir` | função | `() -> Path` |  |
| `is_offline` | função | `(offline=None) -> bool` |  |
| `entry` | função | `(kind, name, version=None) -> dict[str, Any]` | The manifest entry of a model (``kind="models"``) or dataset (``kind="datasets"``); with ``version``, the entry of that release -- the current one or one listed under ``superseded`` (its files are on the Hub, under its own tag; the reason it was superseded is in ``superseded_reason``). |
| `versions` | função | `(kind, name) -> list[dict[str, str]]` | Every release of an item, newest first: version, revision and, for older ones, why they were superseded. |
| `revisions` | função | `(ent) -> list[str]` | Hub revisions holding an entry's files, in the order they are tried: its ``revision`` and, when that is a republication of the same files (``v1.0.0.post1``: new card, Parquet or safetensors copies), the tag of the version itself (``v1.0.0``). Every file is checked against the manifest's SHA-256 … |
| `fetch` | função | `(kind, name, filename, *, offline=None, version=None) -> Path` | Local path of one file of a model or dataset (``version``: a superseded release instead of the current one), verified against the manifest's SHA-256. |
| `status` | função | `(kind, name) -> dict[str, str]` | Where each file of an item currently resolves from without downloading: bundled \| cached \| remote. |

#### `pinneapple_hub.zoo._bench`

``pp.zoo.benchmark``: a zoo model inside ``pp.benchmark`` (``pinneapple_analysis.benchmark``).

<small>`pinneapple_hub/zoo/_bench.py` · 224 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `run_benchmark` | função | `(model, *, budget_s, seeds, baselines, out_dir, force, log, **kwargs)` |  |

#### `pinneapple_hub.zoo._catalog`

Static description of every shipped model and dataset (the source of truth ``scripts/zoo/build_assets.py`` turns into ``manifest.json`` + ``model_card.json`` files). Only items that passed the zoo's gate are listed here; the candidates that did not, and why, are in …

<small>`pinneapple_hub/zoo/_catalog.py` · 715 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `hf_repo` | função | `(kind, name) -> str` | Hub repo of a model/dataset: PINNeAPPle-Labs/<name with dashes>. |

#### `pinneapple_hub.zoo._models`

Zoo model adapters: a ready-to-use predictor per shipped model, with its validity range, metrics against the stored reference, physics checks (feeding a ``TrustReport``) and the hooks fine-tuning needs.

<small>`pinneapple_hub/zoo/_models.py` · 657 linhas · testes: 7 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OutOfRangeWarning` | classe | `()` | Inputs outside the model's validity range (the prediction is an extrapolation). |
| `ZooModel` | classe | `(name, card, config, state, spec)` | Base class: a loaded zoo model. Subclasses implement ``_build``, ``_encode``/``_decode`` and the metrics. |
| `AirfoilMLP` | classe | `()` |  |
| `AirfoilGNN` | classe | `()` | MeshGraphNet on the near field of the campaign's O-grid: p, Ux, Uy, log(1 + nut/nu) at the graph nodes and Cl, Cd, Cm from a pooled wall readout. Coefficient metrics and checks are the MLP adapter's; the field metrics are relative L2 errors on the held-out shapes of the field dataset (steady runs). |
| `HeatsinkForced` | classe | `()` |  |
| `HeatsinkNatural` | classe | `()` | Natural convection: no air velocity input and no pressure drop; trained with monotonicity constraints in conductivity and source width (``pinneapple_design.thermal.heatsink.monotone``). |
| `RoverContact` | classe | `()` |  |
| `SupersededModelWarning` | classe | `()` | A superseded release of a zoo model was loaded. |
| `load_dir` | função | `(directory) -> ZooModel` | Load a model saved with :meth:`ZooModel.save` (e.g. a fine-tuned copy). |
| `load_model` | função | `(name, *, offline=None, version=None) -> ZooModel` |  |
| `copy_model` | função | `(m) -> ZooModel` |  |
| `fit` | função | `(m, X, Y, *, epochs=300, lr=0.0001, batch=256, weight_decay=0.0, seed=0, progress=None) -…)` | Adam on the model's own normalised targets (NaN targets are masked); returns the loss per epoch. |

#### `pinneapple_hub.zoo.catalogue`

The zoo catalogue: one Markdown page listing every model and dataset with its metrics, size and Hub link.

<small>`pinneapple_hub/zoo/catalogue.py` · 73 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `markdown` | função | `() -> str` | The catalogue page (Markdown). |

#### `pinneapple_hub.zoo.checks`

Per-query trust checks of the zoo models, and the verdict they add up to.

<small>`pinneapple_hub/zoo/checks.py` · 358 linhas · testes: 34</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `has_query_checks` | função | `(zm) -> bool` | Whether the model has per-query checks (:func:`query_checks`); otherwise only ``in_range`` applies. |
| `check_spec` | função | `(zm) -> dict[str, Any]` | The declarative description of a model's query checks: ``checks`` (one entry per column of the ``checks`` output: id, rule, name, op, limit, severity, kind, group, detail template), ``aux`` (names of the ``aux`` columns the templates use), ``groups`` (display name and rule of grouped checks) and … |
| `checks_module` | função | `(zm, predictor=None)` | ``torch.nn.Module``: query rows (m, d) -> (values (m, C), aux (m, A)) of :func:`check_spec`. |
| `check_values` | função | `(zm, X) -> tuple[np.ndarray, np.ndarray]` | (values, aux) of the query checks for rows ``X`` (physical inputs, card order). |
| `compare` | função | `(value, op, limit) -> bool` | ``value op limit``; NaN never passes. |
| `evaluate` | função | `(spec, values, aux) -> dict[str, Any]` | Statuses, details and the verdict for one row of check values (the browser does the same with the graph's outputs). |
| `verdict` | função | `(checks) -> dict[str, str]` | The worst status of the trust checks (design constraints excluded) with its title and explanation. |
| `query_checks` | função | `(model, inputs, *, offline=None) -> list[dict[str, Any]]` | The query checks and the verdict for each input row (a mapping of named columns or an (n, d) array). |

#### `pinneapple_hub.zoo.datasets`

``pp.datasets``: ready-made, verified datasets in one line.

<small>`pinneapple_hub/zoo/datasets.py` · 199 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DatasetInfo` | classe | `(name, version, title, description, license, source, verification, splits, columns, citat…)` |  |
| `DatasetSplit` | classe | `(data, name, split, ds_info)` | One split: ``{column: np.ndarray}`` (all columns have ``n_rows`` entries) plus ``name``, ``split``, ``info``. |
| `audit_card` | função | `(name) -> dict[str, Any] \| None` | The stored physics audit card of a dataset (``pinneapple_data.audit``), or None when it was not audited. |
| `audit` | função | `(name, *, offline=None, save=False, **kw)` | Physics audit of a dataset (:func:`pinneapple_data.audit.audit_zoo_dataset`): residual of every sample against its PDE where one is stated, declared constraints, diversity, duplicates, train/test leakage and units. ``save=True`` stores the card with the package (shown in … |
| `list` | função | `() -> builtins.list[str]` | Names of the shipped datasets. |
| `info` | função | `(name) -> DatasetInfo` |  |
| `split_files` | função | `(name, split=None) -> builtins.list[str]` | Files holding ``split`` (all files when None): ``data.npz``, or ``<split>.npz`` / ``<split>-NN.npz``. |
| `load` | função | `(name, split=None, *, offline=None)` | A split (:class:`DatasetSplit`), or ``{split: DatasetSplit}`` for all of them when ``split`` is None. |
| `status` | função | `(name) -> dict[str, str]` | Where the files resolve from right now (bundled \| cached \| remote), without downloading. |

#### `pinneapple_hub.zoo.demo`

What the zoo's demo Space shows, computed by the library: each model's prediction along a sweep of each input, with its ensemble spread or conformal bound and where its query checks warn or fail, and the files of the static Space that shows them.

<small>`pinneapple_hub/zoo/demo.py` · 189 linhas · testes: 21</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `models` | função | `() -> list[str]` | Names of the zoo models the demo offers. |
| `trust` | função | `(name) -> dict[str, Any]` | The model's trust report as a whole (``pp.zoo.validate``): decision and the reasons. |
| `controls` | função | `(name) -> list[dict[str, Any]]` | One entry per input of the model: name, unit, description, range (validity box, or the held-out span for inputs bounded only through derived quantities) and a starting value (the first held-out test case). |
| `sweep` | função | `(name, values, vary, *, n=61, margin=0.15) -> dict[str, Any]` | Predictions and query checks at ``values`` and along ``vary`` from ``margin`` below to ``margin`` above its range (so the sweep shows where the model stops being trusted). |
| `bands` | função | `(outputs) -> dict[str, dict[str, Any]]` | The main outputs of a model and how each one's uncertainty is drawn: ``{"std": "<name>_std"}`` (ensemble spread, drawn as +-2 std) or ``{"upper": "<name with upper95>"}`` (split-conformal 95 % upper bound). |
| `precompute` | função | `(name, *, n=SWEEP_POINTS) -> dict[str, Any]` | Everything the static Space shows for one model without running it: the inputs (:func:`controls`), the outputs with units and bands (:func:`bands`), the model's trust decision (:func:`trust`), the prediction and the query checks at the held-out case, and one :func:`sweep` per input (x, every … |
| `export_space` | função | `(directory, models_=None) -> dict[str, Any]` | Write the data and runtime of the static Space into ``directory``: ``models/<name>.{onnx,json,sweeps.json}``, ``lib/{onnx,zoo}.js`` and ``index.json`` (the models, in order, with their task and trust decision). |

#### `pinneapple_hub.zoo.export`

ONNX export of zoo models and their per-query trust checks.

<small>`pinneapple_hub/zoo/export.py` · 317 linhas · testes: 55 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `predictor_module` | função | `(zm)` | ``torch.nn.Module`` mapping the card's inputs (n, d) to a tuple of outputs (see :func:`output_names`). |
| `output_names` | função | `(zm) -> dict[str, list[str]]` | ONNX output -> the ``predict`` keys of its columns. |
| `as_graph_outputs` | função | `(zm, pred) -> dict[str, np.ndarray]` | ``predict`` output laid out like the graph's outputs. |
| `output_scale` | função | `(zm) -> dict[str, np.ndarray]` | Per column, the scale of the parity tolerance: NaN = relative to the value (positive, log-modelled outputs: Cd, R, dP; L/D relative to max(\|L/D\|, y_std(Cl) / Cd), the error a Cl within tolerance implies), otherwise the network's standardisation scale. |
| `parity_error` | função | `(zm, ref, got) -> float` | Largest \|got - ref\| / scale over every output column (see :func:`output_scale`). Spreads (``std``, ``LD_std``) are measured relative to the quantity they spread around. |
| `export_bytes` | função | `(zm, *, checks=False) -> bytes` | The ONNX graph of a loaded zoo model (see :func:`export_onnx`). |
| `export_onnx` | função | `(name, path, *, checks=False, check_parity=True, offline=None) -> dict[str, Any]` | Export a zoo model to ONNX at ``path``. The graph maps the card's inputs (input ``x``, shape (n, d)) to its outputs; with ``checks=True`` it also takes ``query`` (m, d) and returns the query-check values (``checks``, ``aux``; see :func:`pinneapple_hub.zoo.checks.check_spec`). Unless … |
| `parity_inputs` | função | `(zm, *, n_random=500, seed=7) -> np.ndarray` | The model's held-out and independent-reference inputs (every evaluation split) plus random points inside the validity box (when the box describes every input). |
| `run_onnx` | função | `(data, X, query=None) -> dict[str, np.ndarray]` | Run a graph in onnxruntime (CPU, one thread) and return its outputs by name (float64). |
| `onnx_parity` | função | `(name, onnx_model, X=None, *, offline=None) -> float` | Largest scaled difference (:func:`parity_error`) between an exported graph's predictions and ``ZooModel.predict`` on ``X`` (default :func:`parity_inputs`). |
| `checks_parity` | função | `(name, onnx_model, X=None, *, offline=None) -> dict[str, float]` | Compare the query checks of a graph exported with ``checks=True`` with :func:`pinneapple_hub.zoo.checks.check_values`: the largest value difference (relative to max(\|value\|, 1)) and the number of rows where a status differs although the value is farther than ``CHECK_TOL`` from its limit (float32 … |
| `export_safetensors` | função | `(name, path, *, offline=None) -> dict[str, Any]` | Write the model's weights (the tensors of ``weights.pt``) as safetensors at ``path``; the file's metadata names the model, its version and adapter. Returns ``{"path", "bytes", "sha256", "tensors"}``. |
| `load_safetensors` | função | `(name, path, *, offline=None) -> ZooModel` | The zoo model ``name`` (config and card from the zoo) with its weights read from a safetensors file. |

#### `pinneapple_hub.zoo.hub_cards`

The README cards of the zoo's Hugging Face repos: YAML front matter the Hub reads, and the text people read.

<small>`pinneapple_hub/zoo/hub_cards.py` · 368 linhas · testes: 21</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `space_url` | função | `() -> str` |  |
| `front_matter` | função | `(meta) -> str` |  |
| `parse_front_matter` | função | `(readme) -> dict[str, Any]` | The YAML block at the top of a README (``{}`` when there is none). |
| `code_blocks` | função | `(readme, lang='python') -> list[str]` | The fenced code blocks of a README in language ``lang``. |
| `split_order` | função | `(name) -> list[str]` | Splits of a dataset with train / val / cal / test first, then the others in manifest order. |
| `dataset_card` | função | `(name, layout) -> str` | README of a dataset repo: front matter (configs, dataset_info, licence, task categories, sizes, tags) and the card. ``layout`` is what :func:`pinneapple_hub.zoo.parquet.write_dataset` returned for this dataset. |
| `model_example` | função | `(zm) -> dict[str, Any]` | A held-out reference case of the model (first row of its ``test`` split): the inputs (6 significant digits), the model's outputs for exactly those inputs, the stored reference outputs, the query-check verdict and the model's trust decision. The "How to use" snippet prints these numbers and the … |
| `model_card` | função | `(name, *, example, onnx_info=None, files=()) -> str` | README of a model repo. ``example``: :func:`model_example`; ``onnx_info``: what :func:`pinneapple_hub.zoo.export.export_onnx` returned (None: no ONNX file); ``files``: the derived files published next to the manifest's (``model.safetensors``, ``model.onnx``). |
| `space_readme` | função | `() -> str` | README of the demo Space: the configuration of a static Space (``index.html``, no server), the models and datasets it shows, and what the page does. |
| `example_json` | função | `(example) -> str` |  |

#### `pinneapple_hub.zoo.parquet`

Parquet copies of the zoo datasets, for the Hugging Face dataset viewer and ``datasets.load_dataset``.

<small>`pinneapple_hub/zoo/parquet.py` · 281 linhas · testes: 21</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `read_npz` | função | `(paths) -> dict[str, dict[str, np.ndarray]]` | ``{split: {column: array}}`` from npz files whose arrays are named ``<split>__<column>``; a split spread over several files (shards ``<split>-NN.npz``, in the given order) is concatenated along the rows. |
| `plan_split` | função | `(arrays, *, explode_row_bytes=EXPLODE_ROW_BYTES, max_shard_bytes=MAX_SHARD_BYTES, row_gro…)` | How one split is laid out: ``samples`` (npz rows), ``steps`` (0, or the length of the exploded axis), ``exploded`` columns, ``rows`` (Parquet rows), ``row_bytes`` and the rows per shard and per row group. |
| `write_dataset` | função | `(name, data, out_dir, *, split_order=(), column_order=(), explode_row_bytes=EXPLODE_ROW_B…)` | Write ``{split: {column: array}}`` as Parquet under ``out_dir/data/<config>/`` and return the layout: ``{"configs": [{"name", "default", "columns", "features" (dataset_info YAML), "splits": [{"name", "samples", "steps", "exploded", "rows", "files" (relative to out_dir), "paths", "bytes"}]}]}``. |
| `read_parquet` | função | `(paths) -> dict[str, np.ndarray]` | The npz arrays of one split back from its Parquet shards (in order): the inverse of :func:`write_dataset`. |
| `parquet_files` | função | `(layout) -> list[str]` | Every Parquet file of a layout (paths relative to the dataset's root). |
| `features_yaml` | função | `(schema) -> list[dict[str, Any]]` | The ``dataset_info.features`` list of a Parquet schema, as ``datasets.Features._to_yaml_list`` writes it. |

#### `pinneapple_hub.zoo.publish`

Publish the zoo to the Hugging Face Hub and check what the Hub serves (``scripts/zoo/publish_to_hub.py`` is the command line, ``.github/workflows/publish-zoo.yml`` runs it).

<small>`pinneapple_hub/zoo/publish.py` · 611 linhas · testes: 49</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plan` | função | `(staging=None, *, need_files=True, hub_fallback=False, work=None, only=()) -> list[dict]` | Every item to publish: ``kind``, ``name``, ``repo``, ``repo_type``, ``revision``, ``files`` (manifest files to upload: name -> local path), ``hub_only`` and ``source`` ("local", or "hub" for a Hub-only dataset converted from the Hub copy, prepared when it is published). Items with local files are … |
| `prepare` | função | `(it, work) -> dict` | Write the item's derived files into ``work`` and its README: Parquet copies and the dataset card, or the safetensors and ONNX copies, the example and the model card. Fills ``it["derived"]`` (path in repo -> local path) and ``it["readme"]``. |
| `publish` | função | `(items, token, *, work=None, tags=True) -> list[str]` | Upload every item (one commit each) and tag it with its revision; returns the items that failed (each failure is printed and the others are still published). A Hub-only dataset from the Hub copy is prepared here, one at a time, and its downloads deleted afterwards (the npz are already on the Hub, … |
| `build_space` | função | `(out, app_dir, playground_dir, models=None) -> list[str]` | Write the static demo Space into ``out``: the data and runtime of :func:`pinneapple_hub.zoo.demo.export_space`, the page of ``app_dir`` (``apps/zoo_space``), the Playground's charts, widgets, styles and icon (``playground_dir`` = ``apps/playground/site``) and the README (Space configuration). … |
| `publish_space` | função | `(token, app_dir, playground_dir, *, work=None, from_hub=True) -> list[str]` | Build the static demo Space (:func:`build_space`) and upload it in one commit, deleting the files of an earlier version that the new one does not have. With ``from_hub`` the sweeps are computed from the models and data downloaded from the Hub (the bundled copies ignored), i.e. from what was … |
| `verify_files` | função | `(items, allow_unpublished=False) -> int` | Download every manifest file anonymously from the item's revision (or the version tag it republishes) and compare its SHA-256 with the manifest; returns the number of failures. With ``allow_unpublished``, a Hub-only item that is not on the Hub yet (HTTP 404) is reported but not counted; a wrong … |
| `viewer_status` | função | `(repo, *, timeout=2400, interval=30, get=_get_json, sleep=time.sleep) -> dict[str, Any]` | The dataset viewer's state of one dataset repo: ``is-valid`` (polled until ``preview`` and ``viewer`` are true or ``timeout``), the configs and splits it found and, per split, the ``first-rows`` answer (rows, truncated cells, error). |
| `verify_viewer` | função | `(names=None, *, timeout=2400, interval=30, get=_get_json, sleep=time.sleep) -> int` | Check the dataset viewer of every dataset (anonymous datasets-server API): ``is-valid`` must say preview and viewer, every split of the manifest must be listed, and ``first-rows`` of every split must return rows. Prints one block per dataset; returns the number of datasets that fail. |
| `verify_model` | função | `(name, work) -> list[str]` | Problems found with the published model ``name`` (empty when none): it is loaded from the Hub with the zoo's loader (bundled copy ignored, fresh cache); its card metrics must be reproduced on the stored reference (``pp.zoo.evaluate``); the card's example must be reproduced by the PyTorch weights, … |
| `verify_models` | função | `(names=None) -> int` | :func:`verify_model` for every model, anonymously (no token), with the bundled copies ignored and an empty cache; returns the number of models with problems. |
| `verify_space` | função | `(*, timeout=2400, interval=30, runtime=None, fetch=_fetch, sleep=time.sleep) -> int` | Wait for the demo Space to run and serve its page; 0 when the stage is RUNNING and the page, ``index.json`` and every model's sweeps load (anonymously), 1 otherwise. Prints the stage, hardware and what was fetched. |

#### `pinneapple_hub.zoo.relations`

Metamorphic relations the physics of each zoo model implies (#698), checked without any reference solution by :func:`check_zoo_relations` (``pinneapple_analysis.validation.metamorphic``).

<small>`pinneapple_hub/zoo/relations.py` · 95 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check_zoo_relations` | função | `(model, name=None, *, n=64, seed=0, search='random', relations=None) -> MetamorphicReport` | Run the relations of a zoo model (``model``: a loaded zoo model or anything with the same ``predict``). |

#### `pinneapple_hub.zoo.validation_reports`

Stored validation reports of the zoo models (#706): one ``pp.validate`` report per model, kept with the package, shown in the model's Hub card, and compared with a fresh report before a release.

<small>`pinneapple_hub/zoo/validation_reports.py` · 59 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build` | função | `(name)` | Validate the zoo model ``name`` and store its report. |
| `stored` | função | `(name) -> dict[str, Any] \| None` |  |
| `check_release` | função | `(name, model=None)` | ``(diff, fresh report)`` of the model as it is now (or ``model``, e.g. a new version) against its stored report; ``diff.regressed`` blocks the release. Without a stored report the diff is against an empty one. |
| `card_section` | função | `(name) -> str` | The markdown block of the stored report for the model's Hub card ("" when none is stored). |

#### `pinneapple_hub.zoo.web`

Zoo models for the browser: ONNX graphs with their query checks, model cards and the JS runtime.

<small>`pinneapple_hub/zoo/web/__init__.py` · 160 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `runtime_dir` | função | `() -> Path` | Folder of the JS runtime (``onnx.js``, ``zoo.js``). |
| `reference_data` | função | `(zm) -> dict[str, Any]` | Reference points to plot next to the predictions: for the airfoil, the held-out OpenFOAM runs and the XFOIL polar of each NACA preset; for the rover, the held-out Wong-Reece states. |
| `export_web` | função | `(name, directory, *, offline=None) -> dict[str, Any]` | Write ``<name>.onnx`` and ``<name>.json`` for the browser into ``directory``; returns the JSON content. |
| `fixtures` | função | `(*, offline=None) -> dict[str, Any]` | Python answers for fixed inputs: predictions and query-check evaluations of each model (in range and out of range), and the geometry of a few CST shapes. The JS runtime tests and the Playground's browser test compare against these. |
| `export_bundle` | função | `(directory, models=MODELS, *, offline=None) -> dict[str, Any]` | Everything the Playground loads: each model (:func:`export_web`), the CST geometry graphs and fixtures. |

#### `pinneapple_hub.zoo.web.__main__`

python -m pinneapple_hub.zoo.web [DIR]: write the browser bundle of the zoo (default data/web/zoo).

<small>`pinneapple_hub/zoo/web/__main__.py` · 23 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `main` | função | `(argv=None) -> int` |  |

### Exemplos de uso reais do repositório

_Nenhum exemplo encontrado no repositório para este pacote._

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 23 de 31.
- Funções de teste em arquivos que importam algum módulo do pacote: 183.

### Limitações conhecidas

- _Nenhuma registrada._

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

_Sem texto de objetivo._

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 49 | 10.750 | 66 | 99 | 242 (20 arquivos) | 28 |

Pasta de testes dedicada: `tests/pinneapple_llm` (4 funções de teste).

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `_dispatch` |  | 1 | 1 | 0 |
| `agent_loop` |  | 1 | 3 | 15 |
| `cad_draft` |  | 1 | 7 | 22 |
| `catalog_match` |  | 1 | 2 | 4 |
| `conversation_store` |  | 1 | 2 | 0 |
| `draft` |  | 1 | 5 | 23 |
| `finetune` |  | 1 | 3 | 0 |
| `geometry_draft` |  | 1 | 2 | 0 |
| `guardrail` |  | 1 | 3 | 60 |
| `local_llm` |  | 1 | 3 | 0 |
| `org_knowledge` |  | 1 | 2 | 4 |
| `qa` |  | 23 | 80 | 48 |
| `research` |  | 1 | 2 | 0 |
| `twin_draft` |  | 1 | 3 | 0 |
| `workbench` |  | 12 | 47 | 37 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_llm`

LLM-assisted physics-AI pipeline drafting, gated by a physics-grounded verification layer.

<small>`pinneapple_llm/__init__.py` · 61 linhas · testes: 128</small>

#### `pinneapple_llm._dispatch`

Provider-agnostic LLM call dispatch, shared by ``draft.py``, ``geometry_draft.py``, ``research.py`` and ``twin_draft.py`` -- one place that knows how to reach Anthropic, OpenAI, or a local Ollama server, and one place that logs every call to a :class:`ConversationStore` when one is configured (see …

<small>`pinneapple_llm/_dispatch.py` · 85 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `call_llm` | função | `(prompt, *, provider='anthropic', model=None, api_key=None, system='', json_mode=False, m…)` | Dispatch a single prompt to whichever provider is requested and return the raw text response. See individual provider modules for what ``**provider_kwargs`` accepts (``host``/``port``/``timeout`` for ``"ollama"``; ``base_url`` for ``"openai"`` / ``"openai_compatible"``, i.e. any server speaking the … |

#### `pinneapple_llm.agent_loop`

A genuine multi-step LLM agent loop: the LLM itself decides, turn by turn, which real tool to call next, observes the result, and decides the next step -- grounded in :func:`pinneapple_llm._dispatch.call_llm` (a real, single-call LLM dispatch across anthropic/openai/ollama) plus a real tool …

<small>`pinneapple_llm/agent_loop.py` · 331 linhas · testes: 15</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AgentStep` | classe | `(tool_name, tool_args, tool_result, error, llm_reasoning)` | One turn of the loop: the tool the LLM chose (or ``None`` for a final-answer / error turn that called no tool), the arguments it supplied, the real result of invoking that tool (or ``error`` set instead when invocation failed), and the LLM's stated reasoning for the turn, if the provider/prompt … |
| `AgentLoopResult` | classe | `(steps, final_answer, stopped_reason)` | Full trace of a :func:`run_agent_loop` run. |
| `run_agent_loop` | função | `(goal, tool_registry=None, *, llm_provider='anthropic', model=None, max_steps=8, llm_call…)` | Run a genuine multi-step agent loop toward *goal*. |

#### `pinneapple_llm.cad_draft`

LLM-assisted CAD/geometry generation -- the same constrained pattern as ``draft.py``'s ``draft_problem`` and ``geometry_draft.py``'s ``draft_geometry``, extended to genuinely NEW (not just named-catalog) geometry: instead of picking one fixed preset name, the LLM composes a recipe out of …

<small>`pinneapple_llm/cad_draft.py` · 633 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CadRecipeResult` | classe | `(kind, recipe, name, params, reasoning, raw_response)` |  |
| `draft_mesh_recipe` | função | `(description, *, provider='anthropic', model=None, api_key=None, conversation_store=None)…)` | Ask an LLM to compose a (possibly CSG-combined) geometry recipe for a natural-language description, e.g. "a box with a cylindrical hole through it" or "two overlapping spheres fused together". |
| `draft_cadquery_template` | função | `(description, *, provider='anthropic', model=None, api_key=None, conversation_store=None)…)` | Ask an LLM to pick + parametrise a genuinely parametric CAD template (real B-rep solid, true STEP-exportable) for a natural-language description, e.g. "a cold plate with an internal cooling channel" or "a heat sink plate with 12 fins". |
| `build_recipe` | função | `(result)` | Actually build the geometry a :class:`CadRecipeResult` describes. |
| `export_recipe` | função | `(result, out_path, *, fmt=None) -> str` | Export a built recipe to a real CAD/mesh file on disk. |
| `measure_part` | função | `(result) -> Dict[str, Any]` | Build a ``"mesh_recipe"`` and measure it: volume, bounding-box extents, watertightness, and whether a boolean engine was available. Without one (``manifold3d``, extra ``geometry``) ``build_mesh`` replaces every cut/union by a concatenation of the parts (a UserWarning only), so a washer comes out as … |
| `draft_part` | função | `(description, *, tier='quick', provider='ollama', model=None, api_key=None, max_attempts=…)` | Draft a buildable part, retrying refused drafts: ``tier="quick"`` asks :func:`draft_mesh_recipe` (CSG of primitives), ``tier="engineering"`` asks :func:`draft_cadquery_template` (parametric CadQuery template). |

#### `pinneapple_llm.catalog_match`

pinneapple_llm.catalog_match — ``resolve_problem()``: a two-stage, never-raises front end to ``draft_problem()``.

<small>`pinneapple_llm/catalog_match.py` · 128 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ResolveResult` | classe | `(preset, kwargs, reasoning, used_llm, shortlist)` |  |
| `resolve_problem` | função | `(description, *, provider='anthropic', model=None, api_key=None, conversation_store=None,…)` | Resolve ``description`` to a registered PDE preset — the shortlist step always runs; the LLM step is attempted next and its result used if it succeeds, else the top keyword match is returned instead. Never raises. |

#### `pinneapple_llm.conversation_store`

SQLite-backed conversation log for every ``pinneapple_llm`` LLM call.

<small>`pinneapple_llm/conversation_store.py` · 120 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ConversationRecord` | classe | `(id, timestamp, module, provider, model, system_prompt, user_prompt, response, metadata)` |  |
| `ConversationStore` | classe | `(db_path=_DEFAULT_DB_PATH)` | Thin SQLite wrapper. Safe to construct repeatedly (opens/creates the DB file each time); not held open across calls, so it's safe to share a path across processes/threads without extra locking beyond what SQLite itself provides. |

#### `pinneapple_llm.draft`

LLM-assisted ``ProblemSpec`` drafting -- deliberately constrained to selecting and parametrising one of PINNeAPPle's own already-implemented, already-testable presets, never to inventing new PDE machinery from scratch.

<small>`pinneapple_llm/draft.py` · 231 linhas · testes: 23</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DraftResult` | classe | `(preset, kwargs, reasoning, raw_response)` |  |
| `DraftRejected` | classe | `(message, attempts)` | No acceptable draft after every attempt; ``.attempts`` lists why each one was refused. |
| `draft_problem` | função | `(description, *, provider='anthropic', model=None, api_key=None, conversation_store=None)…)` | Ask an LLM to pick + parametrise a ``ProblemSpec`` preset for a natural-language problem description. |
| `check_kwargs_in_bounds` | função | `(kwargs, bounds) -> Dict[str, float]` | Coerce drafted kwargs to finite floats and check each one against ``bounds`` ({name: (lo, hi)}). |
| `draft_preset_kwargs` | função | `(description, expected_preset, *, bounds=None, provider='ollama', model=None, api_key=Non…)` | Draft kwargs for ONE known preset: retry ``draft_problem`` until it picks ``expected_preset`` with kwargs that pass :func:`check_kwargs_in_bounds`. Returns ``(kwargs, reasoning, attempts_used)``; raises :class:`DraftRejected` after ``max_attempts`` refusals (a wrong preset, no fit, a hallucinated … |

#### `pinneapple_llm.finetune`

LoRA fine-tuning of a local causal LM on data logged via ``conversation_store.ConversationStore``.

<small>`pinneapple_llm/finetune.py` · 116 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FinetuneConfig` | classe | `(base_model, output_dir, lora_r, lora_alpha, lora_dropout, learning_rate, num_train_epoch…)` |  |
| `prepare_dataset` | função | `(store, out_jsonl, *, module=None) -> str` | Export logged (prompt, response) pairs to a JSONL file, ready for :func:`finetune_lora`. Thin wrapper over ``ConversationStore .export_jsonl`` -- kept as a separate step (not folded into ``finetune_lora``) so the dataset can be inspected/filtered/versioned before spending any GPU time on it. |
| `finetune_lora` | função | `(dataset_jsonl, cfg)` | Run a real LoRA fine-tune (``transformers.Trainer`` + ``peft .LoraConfig``) of ``cfg.base_model`` on the (prompt, response) pairs in ``dataset_jsonl`` (as written by :func:`prepare_dataset`). |

#### `pinneapple_llm.geometry_draft`

LLM-assisted geometry drafting -- same constrained pattern as ``draft.py``'s ``draft_problem``, applied to CAD/domain generation instead of PDE problem specs: the LLM selects and parametrises one of ``pinneapple_design.geometry``'s own already-implemented generators …

<small>`pinneapple_llm/geometry_draft.py` · 121 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GeometryDraftResult` | classe | `(generator, kind, kwargs, reasoning, raw_response)` |  |
| `draft_geometry` | função | `(description, *, provider='anthropic', model=None, api_key=None, conversation_store=None)…)` | Ask an LLM to pick + parametrise a geometry generator for a natural-language description (e.g. "a NACA 2412 airfoil" or "a lid driven cavity in 3D"). |

#### `pinneapple_llm.guardrail`

``PhysicsGuardrail``: the actual anti-hallucination mechanism.

<small>`pinneapple_llm/guardrail.py` · 1.069 linhas · testes: 60</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CheckResult` | classe | `(name, passed, detail, value, threshold)` |  |
| `GuardrailReport` | classe | `(checks)` |  |
| `PhysicsGuardrail` | classe | `(spec, *, residual_threshold=0.01, n_check_points=4096, conservation_n_points_per_face=20…)` | Runs a fixed sequence of independent checks against a trained model + the ``ProblemSpec`` it was trained on. |

#### `pinneapple_llm.local_llm`

Local LLM provider (Ollama) -- so ``pinneapple_llm`` doesn't require a paid API/external network call at all: run a model entirely on your own machine, no data leaves it.

<small>`pinneapple_llm/local_llm.py` · 119 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `start_server` | função | `(*, host='127.0.0.1', port=11434, wait_s=10.0) -> subprocess.Popen` | Launch ``ollama serve`` as a background subprocess (no-op / raises if ``ollama`` is not installed -- there is no bundled fallback, same reasoning as every other "requires the real local tool" bridge in this package). |
| `ensure_model_pulled` | função | `(model, *, host='127.0.0.1', port=11434, timeout=3600) -> None` | ``ollama pull <model>`` if it isn't already present locally. Blocks until the pull completes (can be slow for a large model the first time -- ``timeout`` bounds it). |
| `call_ollama` | função | `(prompt, model, api_key=None, system='', *, host='127.0.0.1', port=11434, timeout=300, js…)` | Chat-completion call against a local Ollama server. Raises a clear error (not a generic connection-refused traceback) if no server is reachable -- call :func:`start_server` first, or run ``ollama serve`` yourself. |

#### `pinneapple_llm.org_knowledge`

Grounded LLM answers over a real local :class:`pinneapple_analysis.retrieval.RetrievalIndex` -- the "local LLM that knows about the organisation's own code" building block, built for (and by) ``org-management``'s knowledge-base requirement, but placed here rather than in that downstream repo for …

<small>`pinneapple_llm/org_knowledge.py` · 133 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OrgAnswer` | classe | `(question, context_used, answer)` |  |
| `answer_with_context` | função | `(question, retrieval_index, *, top_k=5, provider='ollama', model='llama3.2:3b', api_key=N…)` | Answer *question* grounded in *retrieval_index*'s real search results (any object exposing a ``search(query, top_k=...) -> List[SearchResult]`` method -- i.e. a real ``pinneapple_analysis.retrieval.RetrievalIndex``, matching that class's own interface). |

#### `pinneapple_llm.qa`

Scientific and industrial quality assurance for ``pinneapple_llm`` artifacts.

<small>`pinneapple_llm/qa/__init__.py` · 51 linhas · testes: 48</small>

#### `pinneapple_llm.qa.examples`

Runnable reference examples for the quality pipeline.

<small>`pinneapple_llm/qa/examples/__init__.py` · 10 linhas · testes: 39</small>

#### `pinneapple_llm.qa.examples.pipe_wear_twin`

End-to-end reference example: a 3D digital twin of a pipe under wall wear.

<small>`pinneapple_llm/qa/examples/pipe_wear_twin.py` · 342 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `hollow_pipe_mesh` | função | `(*, radius=OUTER_DIAMETER_M / 2, thickness=WALL_THICKNESS_M, length=LENGTH_M, n_theta=64)…)` | A genuinely hollow pipe: outer wall, inner wall and two end annuli. |
| `solid_pipe_mesh` | função | `(*, radius=OUTER_DIAMETER_M / 2, length=LENGTH_M, n_theta=64) -> Tuple[np.ndarray, np.nda…)` | A filled cylinder of the same outer dimensions. |
| `build_spec` | função | `(*, anomaly=False) -> PhysicalSpecification` |  |
| `build_twin` | função | `(*, anomaly=False) -> Dict[str, Any]` |  |
| `export_twin` | função | `(bundle, output_dir=None) -> str` | Write the real viewer files and the hub contract; returns ``scene.json``. |
| `analytical_reference` | função | `(bundle) -> ReferenceData` | Closed-form uniform-wear residual thickness, per cell and per time. |
| `run_example` | função | `(*, output_dir=None, anomaly=False, write_report=True) -> Tuple[QualityReport, Dict[str, …)` | Run stages A-G over the synthetic pipe twin and return the report. |

#### `pinneapple_llm.qa.examples.pipe_wear_twin_main`

Run the reference example and print its quality report::

<small>`pinneapple_llm/qa/examples/pipe_wear_twin_main.py` · 37 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `main` | função | `(argv=None) -> int` |  |

#### `pinneapple_llm.qa.generation`

Stage B informed: the LLM's structured claims, and why they are not evidence.

<small>`pinneapple_llm/qa/generation.py` · 240 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GeneratedArtifact` | classe | `(code, assumptions, dependencies, units, equations, limitations, provider, model, elapsed…)` | The LLM's proposal plus its structured claims. Untrusted by construction. |
| `build_prompt` | função | `(spec) -> str` | A prompt carrying the specification, not a paraphrase of it. |
| `parse_generation_response` | função | `(raw, *, provider='', model='', elapsed_s=0.0) -> Tuple[GeneratedArtifact, StaticResult]` | Parse and schema-check an LLM response. Never raises on malformed JSON. |
| `check_claims` | função | `(artifact, spec) -> List[Finding]` | Confront the model's declarations with the specification and the unit table. |
| `validate_generated` | função | `(artifact, spec, *, filename='artifact.py') -> Tuple[List[Finding], Any]` | Claim checks + the full static validation of the returned code. |
| `generate_artifact` | função | `(spec, *, provider='anthropic', model=None, api_key=None, conversation_store=None) -> Gen…)` | Ask a provider for an artifact + envelope, through the existing dispatch. |

#### `pinneapple_llm.qa.levels`

Quality levels Q0-Q4, awarded from what actually ran -- never from an LLM.

<small>`pinneapple_llm/qa/levels.py` · 182 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `evaluate_level` | função | `(report, spec=None) -> Tuple[str, List[str]]` | Return ``(level, reasons)`` -- the reasons explain why it stopped there. |
| `blocking_reasons` | função | `(report) -> List[str]` | Human-readable reasons the report cannot be treated as approved. |
| `level_description` | função | `(level) -> str` |  |

#### `pinneapple_llm.qa.pipeline`

The quality pipeline: stages A-G wired into one reproducible run.

<small>`pinneapple_llm/qa/pipeline.py` · 455 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PipelineConfig` | classe | `(validators, run_artifacts, source_name, static_policy, repair_attempts)` |  |
| `QualityPipeline` | classe | `(spec, *, validators=None, sandbox=None, run_artifacts=False, static_policy=None, referen…)` |  |

#### `pinneapple_llm.qa.references`

Stage F: comparison against independent reference data.

<small>`pinneapple_llm/qa/references.py` · 202 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ReferenceData` | classe | `(name, origin, version, values, unit, limitations, used_for_calibration)` | A reference the artifact is compared against -- never invented here. |
| `compare` | função | `(predictions, reference, criteria, *, subset='validation', groups=None, baseline=None) ->…)` | Compare ``predictions`` with ``reference.values`` under ``criteria``. |
| `split_mask` | função | `(n, holdout_fraction=0.3, seed=0) -> np.ndarray` | A deterministic boolean mask marking the validation (holdout) subset. |

#### `pinneapple_llm.qa.repair`

Stage G: structured diagnosis and bounded automatic repair.

<small>`pinneapple_llm/qa/repair.py` · 296 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Diagnosis` | classe | `(failed_steps, findings, confirmed_cause, probable_cause, constraints_to_preserve, tests_…)` | Why an attempt was rejected, in the form a repair step can consume. |
| `RepairAttempt` | classe | `(attempt, accepted, status, diagnosis, diff, notes, duration_s)` |  |
| `RepairOutcome` | classe | `(accepted, artifact, attempts, results, stopped_because)` |  |
| `RepairLoop` | classe | `(*, evaluate, repair, spec=None, required=(), max_attempts=DEFAULT_MAX_ATTEMPTS, on_attem…)` | Bounded generate -> validate -> diagnose -> repair loop. |
| `diagnose` | função | `(results, *, required=()) -> Diagnosis` | Turn failing validator results into one structured diagnosis. |

#### `pinneapple_llm.qa.report`

Reproducible quality report -- the machine-readable output of every run.

<small>`pinneapple_llm/qa/report.py` · 327 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Status` | classe | `()` |  |
| `Finding` | classe | `(id, status, message, severity, evidence, justification, limitations)` | One observed fact, with the evidence that produced it. |
| `StepOutcome` | classe | `(step, status, findings, duration_s, metadata)` | The result of one pipeline stage. |
| `ArtifactRecord` | classe | `(name, kind, path, sha256, size_bytes, metadata)` |  |
| `QualityReport` | classe | `(run_id, created_at, spec_id, spec_hash, criteria_hash, provider, model, steps, artifacts…)` | Everything one pipeline run observed, and what it is allowed to claim. |
| `environment_snapshot` | função | `() -> Dict[str, Any]` | Versions and platform, so a report can be traced to an environment. |

#### `pinneapple_llm.qa.sandbox`

Stage D: controlled execution of generated artifacts.

<small>`pinneapple_llm/qa/sandbox.py` · 389 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `IsolationLevel` | classe | `()` |  |
| `SandboxLimits` | classe | `(timeout_s, cpu_s, memory_mb, file_size_mb, open_files, max_processes, write_bytes)` | Resource ceilings. ``None`` disables the corresponding limit. |
| `SandboxResult` | classe | `(status, isolation, workdir, exit_code, stdout, stderr, duration_s, timed_out, violations…)` |  |
| `SandboxExecutor` | classe | `(*, limits=None, allow_execution=False, allow_network=False, base_dir=None, keep_workdir=…)` | Run a generated Python artifact in a separate, limited process. |
| `hash_bytes` | função | `(blob) -> str` |  |

#### `pinneapple_llm.qa.spec`

Typed, serializable physical specification -- stage A of the quality pipeline.

<small>`pinneapple_llm/qa/spec.py` · 496 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Provenance` | classe | `()` |  |
| `SpecError` | classe | `()` | Raised when a specification is structurally invalid. |
| `Quantity` | classe | `(value, unit, provenance, source, uncertainty)` | A number with a unit, a provenance tag, and (optionally) an uncertainty. |
| `Invariant` | classe | `(name, kind, target, params, severity, description)` | A physical constraint the artifact must satisfy. Declarative, not code. |
| `Observation` | classe | `(sensor_id, quantity, unit, position, timestamps, time_unit, values, provenance, holdout)` | A sensor reading: where, what, when, and whether it may be used for validation. |
| `GeometrySpec` | classe | `(kind, coordinate_system, dimensions, hollow, wall_thickness, notes)` | Geometry, domain and coordinate system, with units made explicit. |
| `PhysicalSpecification` | classe | `(objective, phenomenon, geometry, independent_variables, dependent_variables, parameters,…)` | Stage A output: what the user asked for, expressed so it can be checked. |
| `quantity_map` | função | `(**kwargs) -> Dict[str, Quantity]` | ``quantity_map(nu=(1e-6, "m^2/s"))`` -- small construction helper. |

#### `pinneapple_llm.qa.static`

Stage C: static and structural validation of generated artifacts.

<small>`pinneapple_llm/qa/static.py` · 596 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `StaticPolicy` | classe | `(allowed_roots, forbidden_roots, forbidden_calls, forbidden_attr_calls, allowed_write_roo…)` |  |
| `StaticResult` | classe | `(findings, resolved_imports)` |  |
| `check_syntax` | função | `(source, filename='<artifact>') -> List[Finding]` | Compile-only check. A ``SyntaxError`` is reported, never raised. |
| `check_imports` | função | `(source, policy) -> Tuple[List[Finding], Dict[str, Any]]` | Refuse forbidden imports; report unknown/unresolvable ones with evidence. |
| `check_api_contracts` | função | `(source, resolved) -> List[Finding]` | Resolve every ``pinneapple_*`` attribute the artifact touches, for real. |
| `check_signatures` | função | `(source, resolved) -> List[Finding]` | Compare literal call sites against the real ``inspect.signature``. |
| `check_security` | função | `(source, policy, write_roots=()) -> List[Finding]` | Refuse process spawning, code execution, network and out-of-scope writes. |
| `check_json_artifact` | função | `(data, schema=None, name='artifact') -> List[Finding]` | Validate a JSON artifact against a small, explicit schema subset. |
| `validate_python_artifact` | função | `(source, *, policy=None, filename='<artifact>', write_roots=(), json_schema=None) -> Stat…)` | Run every static check and return one :class:`StaticResult`. |
| `validate_json_artifact` | função | `(payload, schema=None, name='artifact') -> StaticResult` |  |
| `load_json_file` | função | `(path) -> Any` |  |

#### `pinneapple_llm.qa.units`

Dimensional analysis for the quality pipeline -- dependency-free, deliberately small.

<small>`pinneapple_llm/qa/units.py` · 327 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `UnitError` | classe | `()` | Raised for an unparsable unit string or an incompatible conversion. |
| `Unit` | classe | `(name, dim, scale, offset)` | An SI-scaled unit: ``si_value = value * scale + offset``. |
| `parse_unit` | função | `(text) -> Unit` | Parse a unit string into a :class:`Unit`. Raises :class:`UnitError`. |
| `are_compatible` | função | `(a, b) -> bool` | True if two unit strings have the same physical dimension. |
| `convert` | função | `(value, from_unit, to_unit) -> float` | Convert ``value`` between two compatible units. Raises on incompatibility. |
| `dimension_of` | função | `(unit) -> Dim` |  |
| `dimension_str` | função | `(unit) -> str` |  |

#### `pinneapple_llm.qa.validators`

Scientific validators and their registry (stage E).

<small>`pinneapple_llm/qa/validators/__init__.py` · 19 linhas · **sem teste direto**</small>

#### `pinneapple_llm.qa.validators.base`

Validator protocol, shared context and registry.

<small>`pinneapple_llm/qa/validators/base.py` · 164 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SensorReading` | classe | `(id, position, unit, quantity, series, holdout, position_unit)` | One sensor: where, what unit, and the series over the artifact's time axis. |
| `ValidationContext` | classe | `(spec, meshes, fields, field_units, times, time_unit, sensors, payload, model, extra)` | Everything a validator may look at. Absent data means "not supplied". |
| `ValidatorResult` | classe | `(name, status, findings, evidence, limitations, duration_s)` |  |
| `register` | função | `(name) -> Callable[[Validator], Validator]` |  |
| `available` | função | `() -> List[str]` |  |
| `get` | função | `(name) -> Validator` |  |
| `run` | função | `(name, ctx) -> ValidatorResult` |  |
| `not_applicable` | função | `(name, reason, **evidence) -> ValidatorResult` |  |
| `blocked` | função | `(name, reason, **evidence) -> ValidatorResult` |  |

#### `pinneapple_llm.qa.validators.geometry`

Geometry validation, including the hollow-body check the spec calls out.

<small>`pinneapple_llm/qa/validators/geometry.py` · 206 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check_geometry` | função | `(ctx) -> ValidatorResult` |  |

#### `pinneapple_llm.qa.validators.guardrail`

Bridge to the existing :class:`pinneapple_llm.guardrail.PhysicsGuardrail`.

<small>`pinneapple_llm/qa/validators/guardrail.py` · 70 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check_guardrail` | função | `(ctx) -> ValidatorResult` |  |

#### `pinneapple_llm.qa.validators.numeric`

Numeric integrity: finite values, shapes, degenerate fields.

<small>`pinneapple_llm/qa/validators/numeric.py` · 108 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check_numeric` | função | `(ctx) -> ValidatorResult` |  |

#### `pinneapple_llm.qa.validators.physics`

Physical invariants declared by the specification (stage E, part 3).

<small>`pinneapple_llm/qa/validators/physics.py` · 203 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check_physics` | função | `(ctx) -> ValidatorResult` |  |

#### `pinneapple_llm.qa.validators.sensors`

Sensor / observation consistency.

<small>`pinneapple_llm/qa/validators/sensors.py` · 123 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check_sensors` | função | `(ctx) -> ValidatorResult` |  |

#### `pinneapple_llm.qa.validators.temporal`

Temporal consistency: ordering, sample counts, validity window.

<small>`pinneapple_llm/qa/validators/temporal.py` · 99 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check_temporal` | função | `(ctx) -> ValidatorResult` |  |

#### `pinneapple_llm.qa.validators.units`

Dimensional consistency across geometry, parameters, fields and sensors.

<small>`pinneapple_llm/qa/validators/units.py` · 142 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check_units` | função | `(ctx) -> ValidatorResult` |  |

#### `pinneapple_llm.research`

Literature/repository search integrated with an LLM -- built entirely on PINNeAPPle's own already-existing, real API-backed search (``pinneapple_tools.hpo_experiments.sources.ArxivSource``/ ``GithubSource``, which call the real arXiv and GitHub search APIs), never an LLM asked to "recall" papers or …

<small>`pinneapple_llm/research.py` · 85 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ResearchReport` | classe | `(query, papers, repos, llm_summary)` |  |
| `search_literature` | função | `(query, *, k_papers=5, k_repos=5, summarize=False, provider='anthropic', model=None, api_…)` | Search arXiv + GitHub for real papers/repositories matching ``query``, optionally with an LLM summary of the (real) results. |

#### `pinneapple_llm.twin_draft`

LLM-assisted digital twin drafting -- same constrained pattern as ``draft.py``/``geometry_draft.py``: the LLM proposes values for ``pinneapple_systems.digital_twin.DigitalTwinConfig``'s own real, introspected fields (update interval, assimilation method, anomaly threshold, ...), never a free-form …

<small>`pinneapple_llm/twin_draft.py` · 156 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TwinDraftResult` | classe | `(config_kwargs, reasoning, raw_response)` |  |
| `draft_digital_twin` | função | `(description, *, provider='anthropic', model=None, api_key=None, conversation_store=None)…)` | Ask an LLM to propose ``DigitalTwinConfig`` kwargs for a natural-language description (e.g. "a twin that re-infers every 100ms and flags anomalies aggressively"). |
| `build_3d_live_twin` | função | `(model, field_names, *, export_dir, points, config_kwargs=None, **build_kwargs)` | Build a real ``DigitalTwin`` (``pinneapple_systems.digital_twin .build_digital_twin``) plus a bound "3D export" hook: every time the twin's own state updates, write the current field prediction at ``points`` as the next frame in a ``pinneapple_blender`` ``.ply`` sequence in ``export_dir`` -- so … |

#### `pinneapple_llm.workbench`

PINNeAPPle Agent Workbench: a team of coding and engineering agents running on a local model.

<small>`pinneapple_llm/workbench/__init__.py` · 22 linhas · testes: 20</small>

#### `pinneapple_llm.workbench.__main__`

``python -m pinneapple_llm.workbench <command>`` (also ``pinneapple-workbench``).

<small>`pinneapple_llm/workbench/__main__.py` · 274 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `cmd_serve` | função | `(args)` |  |
| `cmd_doctor` | função | `(args)` |  |
| `cmd_pull` | função | `(args)` |  |
| `cmd_run` | função | `(args)` |  |
| `cmd_eval` | função | `(args)` |  |
| `cmd_finetune` | função | `(args)` |  |
| `cmd_vscode` | função | `(args)` |  |
| `main` | função | `(argv=None)` |  |

#### `pinneapple_llm.workbench.agents`

The agent definitions: one Markdown file per agent, with a small front matter (name, title, description, tools) and the system prompt as the body. ``prompts/_common.md`` is appended to every agent.

<small>`pinneapple_llm/workbench/agents.py` · 65 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Agent` | classe | `(name, title, description, tools, prompt, source, extra)` |  |
| `parse_agent` | função | `(text, source='') -> Agent` |  |
| `load_agents` | função | `(workspace=None, user_dir=USER_DIR) -> tuple[dict[str, Agent], str]` | All agents by name (built-in, then the user folder, then the workspace folder), and the common rules. |

#### `pinneapple_llm.workbench.backend`

Chat-with-tools backends: Ollama, any OpenAI-compatible server (LM Studio, llama.cpp, vLLM, MLX), and a scripted backend for tests.

<small>`pinneapple_llm/workbench/backend.py` · 262 linhas · testes: 20 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ToolCall` | classe | `(name, arguments, id)` |  |
| `Reply` | classe | `(content, tool_calls, usage)` |  |
| `Backend` | classe | `(model, tool_mode)` |  |
| `OllamaBackend` | classe | `(model, host='http://127.0.0.1:11434', *, num_ctx=32768, temperature=0.2, tool_mode='nati…)` | ``/api/chat`` of a local Ollama server (``ollama serve``). |
| `OpenAIBackend` | classe | `(model, base_url='http://127.0.0.1:1234/v1', api_key='local', *, temperature=0.2, tool_mo…)` | ``/v1/chat/completions`` of any OpenAI-compatible server: LM Studio, llama.cpp ``llama-server``, vLLM, ``mlx_lm.server``, or a hosted endpoint. |
| `ScriptedBackend` | classe | `(script, model='scripted')` | Replays a list of replies, or calls ``fn(messages, tools) -> Reply``: for tests and demos without a model. |
| `parse_json_tool_call` | função | `(text, tool_names) -> ToolCall \| None` | Read ``{"tool": ..., "arguments": {...}}`` (or ``name``/``args``) from a reply; None when there is none. |
| `make_backend` | função | `(kind='ollama', model='', **kw) -> Backend` |  |

#### `pinneapple_llm.workbench.evaluate`

A small, checkable benchmark of the workbench on your machine and model.

<small>`pinneapple_llm/workbench/evaluate.py` · 128 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TaskResult` | classe | `(name, passed, seconds, model_calls, prompt_tokens, completion_tokens, detail)` |  |
| `run_eval` | função | `(backend, names=None, on_event=None) -> list[TaskResult]` |  |
| `format_results` | função | `(results, model) -> str` |  |
| `as_dicts` | função | `(results) -> list[dict]` |  |

#### `pinneapple_llm.workbench.finetune`

Fine-tune the local model on your own workbench sessions.

<small>`pinneapple_llm/workbench/finetune.py` · 178 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TrainConfig` | classe | `(model, adapter_path, iters, batch_size, num_layers, learning_rate, max_seq_length, lora_…)` |  |
| `redact` | função | `(text) -> str` |  |
| `export_dataset` | função | `(store, out_dir, *, min_rating=0, mode='trajectories', agents=None, val_fraction=0.1, see…)` | Write ``train.jsonl`` / ``valid.jsonl``; returns the counts. |
| `mlx_train_command` | função | `(data_dir, cfg) -> list[str]` |  |
| `train_mlx` | função | `(data_dir, cfg=None, *, dry_run=False) -> str` | LoRA fine-tune with ``mlx_lm`` on Apple Silicon; returns the adapter folder. ``--mask-prompt`` trains on the final assistant turn of each example. |
| `serve_mlx_command` | função | `(cfg=None, port=8080) -> list[str]` |  |
| `train_peft` | função | `(data_dir, cfg) -> str` | LoRA fine-tune with Hugging Face TRL + PEFT (CUDA). ``cfg.model`` is a Hugging Face model id; returns the adapter folder. |

#### `pinneapple_llm.workbench.models`

Which local model to run, from the machine's memory.

<small>`pinneapple_llm/workbench/models.py` · 75 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Machine` | classe | `(system, arch, memory_gb, apple_silicon)` |  |
| `Recommendation` | classe | `(min_memory_gb, model, approx_size_gb, context, note)` |  |
| `detect_machine` | função | `() -> Machine` |  |
| `recommend` | função | `(memory_gb=None) -> Recommendation` | The default model for a machine with ``memory_gb`` of memory (detected when omitted). |

#### `pinneapple_llm.workbench.runtime`

The agent loop: the lead talks to you, calls tools, keeps a plan and delegates parts of the work to the specialist agents, each running its own loop with its own prompt and tools.

<small>`pinneapple_llm/workbench/runtime.py` · 301 linhas · testes: 20 · palavras-chave no código (heurística; o texto acima diz o status real): placeholder/stub citado no docstring</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Cancelled` | classe | `()` |  |
| `SessionSettings` | classe | `(max_steps, max_depth, context_chars, keep_recent, language)` |  |
| `Session` | classe | `(backend, workspace, *, policy=None, settings=None, on_event=None, agents=None, common=No…)` |  |

#### `pinneapple_llm.workbench.server`

The workbench web app: FastAPI routes over :class:`~.runtime.Session` and the :class:`~.store.Store`, plus the static single-page frontend in ``static/`` (no build step).

<small>`pinneapple_llm/workbench/server.py` · 318 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Manager` | classe | `(store, backend_factory, default_workspace, language='')` | Live sessions (one worker thread per running turn) over the store. |
| `create_app` | função | `(*, backend='ollama', model='', host='', base_url='', api_key='', workspace='.', db=DEFAU…)` |  |

#### `pinneapple_llm.workbench.store`

The workbench database: a local SQLite file, or a PostgreSQL server you already run.

<small>`pinneapple_llm/workbench/store.py` · 158 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Store` | classe | `(url=DEFAULT_DB)` |  |

#### `pinneapple_llm.workbench.tools`

The tools the agents use, confined to one workspace folder.

<small>`pinneapple_llm/workbench/tools.py` · 355 linhas · testes: 37</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ToolError` | classe | `()` |  |
| `ApprovalPolicy` | classe | `(mode)` |  |
| `Workspace` | classe | `(root, policy=None, approve=None, run_timeout=600, web=True)` | A folder the agents may read and change. ``approve(tool, args) -> bool`` is asked before guarded actions. |

#### `pinneapple_llm.workbench.web`

Internet search and page fetching for the agents.

<small>`pinneapple_llm/workbench/web.py` · 142 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `parse_duckduckgo` | função | `(page) -> list[dict]` |  |
| `web_search` | função | `(query, max_results=8) -> list[dict]` | ``[{title, url, snippet}]`` from the first configured provider. |
| `html_to_text` | função | `(page) -> str` |  |
| `fetch_text` | função | `(url, max_chars=12000) -> str` | The readable text of a page (HTML stripped) or a raw text file; GitHub ``blob`` links are read raw. |

### Exemplos de uso reais do repositório

_Nenhum exemplo encontrado no repositório para este pacote._

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 21 de 49.
- Funções de teste em arquivos que importam algum módulo do pacote: 242.

### Limitações conhecidas

- _Nenhuma registrada._

## 7. `pinneapple_mcp`

### Objetivo

_Sem texto de objetivo._

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 8 | 1.541 | 2 | 31 | 30 (3 arquivos) | 3 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `__main__` |  | 1 | 0 | 0 |
| `config` |  | 1 | 3 | 10 |
| `expressions` |  | 1 | 2 | 10 |
| `jobs` |  | 1 | 5 | 0 |
| `problems` |  | 1 | 2 | 0 |
| `server` |  | 1 | 2 | 10 |
| `tools` |  | 1 | 17 | 30 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_mcp`

PINNeAPPle as an MCP server: solve, benchmark and validate physics problems from any coding agent.

<small>`pinneapple_mcp/__init__.py` · 34 linhas · testes: 30</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_server` | função | `(*args, **kwargs)` |  |
| `main` | função | `(argv=None)` |  |

#### `pinneapple_mcp.__main__`

_Módulo sem docstring e sem texto._

<small>`pinneapple_mcp/__main__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_mcp.config`

Limits and locations of the MCP server. Every value has a safe default and an environment variable.

<small>`pinneapple_mcp/config.py` · 91 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Config` | classe | `(workdir, max_seconds, max_memory_mb, max_jobs, max_epochs, max_points, max_file_mb, allo…)` |  |
| `get_config` | função | `() -> Config` |  |
| `set_config` | função | `(cfg) -> Config` |  |

#### `pinneapple_mcp.expressions`

Safe arithmetic expressions of the coordinates, e.g. ``"sin(pi*x)*sinh(pi*y)/sinh(pi)"``.

<small>`pinneapple_mcp/expressions.py` · 105 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `compile_expression` | função | `(text, coords) -> Callable[[np.ndarray], np.ndarray]` | ``f(X) -> (N,)`` for points ``X`` of shape ``(N, len(coords))``. Raises ``ValueError`` on anything that is not plain arithmetic of the coordinates. |
| `value` | função | `(v, coords)` | A number stays a number; a string becomes ``compile_expression``. |

#### `pinneapple_mcp.jobs`

Long work as jobs: a separate process per job, with a wall-time limit, a memory limit and no network.

<small>`pinneapple_mcp/jobs.py` · 199 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `running` | função | `() -> list[str]` |  |
| `submit` | função | `(kind, args, cfg, max_seconds=None) -> str` | Start job ``kind`` (a key of ``tools.JOB_KINDS``) in a new process; returns its id. |
| `status` | função | `(job_id, cfg, wait_seconds=0.0) -> dict[str, Any]` | Status of a job (``queued``, ``running``, ``done``, ``failed``, ``timeout``, ``cancelled``, ``lost``), its result when done, and the files in its directory. ``wait_seconds`` waits up to that long for it to finish. |
| `cancel` | função | `(job_id, cfg) -> dict[str, Any]` |  |
| `list_jobs` | função | `(cfg, limit=20) -> list[dict[str, Any]]` |  |

#### `pinneapple_mcp.problems`

A physics problem from JSON: a preset, or a box problem described field by field.

<small>`pinneapple_mcp/problems.py` · 156 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BuiltProblem` | classe | `(problem, ctx, exact, dirichlet, source, spec)` |  |
| `build_problem` | função | `(spec) -> BuiltProblem` | ``BuiltProblem`` from the JSON description above. Raises ``ValueError`` with the reason on bad input. |

#### `pinneapple_mcp.server`

The stdio MCP server: ``pinneapple-mcp`` or ``python -m pinneapple_mcp``.

<small>`pinneapple_mcp/server.py` · 284 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_server` | função | `(config=None)` | The MCP server object (SDK 2.x ``MCPServer`` or 1.x ``FastMCP``) with every PINNeAPPle tool. |
| `main` | função | `(argv=None) -> None` |  |

#### `pinneapple_mcp.tools`

The MCP tools as plain functions: JSON-like arguments in, JSON-like dictionaries out.

<small>`pinneapple_mcp/tools.py` · 667 linhas · testes: 30</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `capabilities` | função | `() -> dict[str, Any]` | What this installation offers: version, optional APIs present, device, limits and the working directory. |
| `list_methods` | função | `(category='all', validated_only=False) -> dict[str, Any]` | Solve methods usable by ``solve`` (grouped by kind) and the method catalog (solvers, training methods, equations, problems) with the validation status of each item. |
| `describe_method` | função | `(method) -> dict[str, Any]` | A solve method (e.g. ``pinn``, ``fem``) with its options, or a catalog item (e.g. ``S2``) with its code, references, validation status and the tests that give it. |
| `list_presets` | função | `(contains=None) -> dict[str, Any]` | Problem presets usable as ``{"preset": name}`` in ``solve`` and ``benchmark``. |
| `trust_card` | função | `(file=None, field=None, grid=None, values=None, pde=None, source=None, alpha=1.0, dirichl…)` | Physics checks of a field on a grid and its trust card (decision APPROVED / REVIEW / REJECT). |
| `solve` | função | `(problem, method='pinn', options=None, reference=None, reference_options=None, grid_point…)` | Solve ``problem`` with ``method`` in a job; the result carries metrics against the reference, the physics checks and the trust card, and the paths of the saved prediction and figures. Returns the job status (``wait_seconds`` > 0 waits for the result up to that long; otherwise poll ``job_status``). |
| `benchmark` | função | `(problem, methods=('pinn', 'fem'), budget=None, seeds=1, reference=None, options=None, gr…)` | Run several methods on one problem with the same training budget (epochs, collocation points, seeds for every neural method) and score them on the same grid against one reference; each run gets a trust card. Returns the job status (see ``solve``). |
| `zoo_list` | função | `(task=None) -> dict[str, Any]` | Pre-trained, verified models of the zoo (with what they predict, their validity range and held-out errors) and the datasets of the dataset hub. |
| `zoo_predict` | função | `(name, inputs) -> dict[str, Any]` | Predictions of zoo model ``name`` for ``inputs`` ({input name: value or list}), with the mask of rows inside the model's validity range (rows outside it are reported, never silently extrapolated). |
| `zoo_validate` | função | `(name, inputs=None) -> dict[str, Any]` | Trust report of zoo model ``name``: its card metrics reproduced on the stored reference, the zoo acceptance limits, the model's physics checks and, with ``inputs``, whether they are inside its validity range. |
| `job_status` | função | `(job_id, wait_seconds=0.0) -> dict[str, Any]` | Status of a job and, when it is done, its result with the paths of its files. |
| `list_jobs` | função | `(limit=20) -> dict[str, Any]` | The most recent jobs with their status. |
| `cancel_job` | função | `(job_id) -> dict[str, Any]` | Stop a running job (its process is killed). |
| `design` | função | `(text, answers=None, verify=True, seed=0, max_seconds=None, wait_seconds=0.0) -> dict[str…)` | Design from a plain-language goal. Out-of-scope requests and missing values are answered at once (status ``out_of_scope`` with what is needed, or ``needs_input`` with the questions); an in-scope request starts a job whose result is the design package (verdict, design, margins, verification, files). |
| `reason` | função | `(case=None, requirements=None, concept=None, lang='pt', max_steps=12, max_seconds=None, w…)` | Chain of whys (pinneapple_reason): from a naive concept, check every requirement's failure modes, apply the best countermeasure for the first failure, repeat. Starts a job; its result is the path (version, failure found, countermeasure), the explanation and the report files. |
| `reason_library` | função | `(domain=None, lang='pt') -> dict[str, Any]` | The failure modes (by physical domain) and countermeasures pinneapple_reason knows, and the worked cases. |
| `reason_check` | função | `(mode, params, elements=(), limits=None, lang='pt') -> dict[str, Any]` | Run one failure mode of the library on the given parameters (SI units): demand, capacity, margin, the mechanism in words, every measured number, sources, applicability and self-checks. |

### Exemplos de uso reais do repositório

_Nenhum exemplo encontrado no repositório para este pacote._

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 5 de 8.
- Funções de teste em arquivos que importam algum módulo do pacote: 30.

### Limitações conhecidas

- _Nenhuma registrada._

## 8. `pinneapple_autolab`

### Objetivo

_Sem texto de objetivo._

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 16 | 3.596 | 38 | 39 | 37 (5 arquivos) | 10 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `agent` |  | 1 | 9 | 6 |
| `core` |  | 1 | 13 | 0 |
| `dashboard` |  | 1 | 5 | 5 |
| `domain` |  | 1 | 2 | 0 |
| `domains` |  | 2 | 11 | 11 |
| `lab` |  | 1 | 2 | 10 |
| `ledger` |  | 1 | 4 | 0 |
| `mission` |  | 1 | 4 | 0 |
| `qualification` |  | 1 | 1 | 0 |
| `replay` |  | 1 | 1 | 6 |
| `report` |  | 1 | 8 | 0 |
| `simulator` |  | 1 | 3 | 0 |
| `surrogates` |  | 1 | 9 | 0 |
| `trust` |  | 1 | 5 | 0 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_autolab`

pinneapple_autolab -- the Autonomous Physics Lab: an AI that manages evidence to make engineering decisions.

<small>`pinneapple_autolab/__init__.py` · 70 linhas · testes: 31</small>

#### `pinneapple_autolab.agent`

The lab driven by an agent: a mission in words -> a Mission -> tool calls on the lab -> an audited answer.

<small>`pinneapple_autolab/agent.py` · 488 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MissionRefused` | classe | `()` | The request cannot be run as written (with the reason and what is needed). |
| `MissionDomain` | classe | `(title, variables, baseline, output_words, units, operating_point, acceptance, conditions…)` | What a simulator can be asked: its variables and baseline, outputs by the words that name them, the fixed operating point, default acceptance tolerances. ``output_words``: output -> words (e.g. "drag_N": ["drag"]); ``operating_point``: name -> (value, unit, regex capturing the value in a request). |
| `MissionReading` | classe | `(mission, readings, assumptions)` |  |
| `LabPlanner` | classe | `(lab, n_doe=None)` | Deterministic step planner over :func:`lab_toolbox` (no network): read the mission, plan, baseline, space-filling designs, then (learn, assess, follow the lab's rule, simulate) until only the verification reserve is left, verify, read the status, answer. A stand-in for a language model, not a … |
| `AgentResult` | classe | `(reading, refused, answer, audit, run, summary, wall_s, planner)` |  |
| `interpret_mission` | função | `(text, dom, llm=None) -> MissionReading` | Read ``text`` into a Mission over ``dom``. ``llm`` (optional) has ``extract(text, schema) -> dict`` (e.g. :class:`pinneapple_design.intent.CallLLM`); its answer is accepted only where it names closed options and quotes words present in the text. |
| `lab_toolbox` | função | `(lab, reading=None) -> Toolbox` | The closed tool set an agent drives ``lab`` with. Every tool returns what the lab measured or predicted (predictions are labelled), rounded to 5 significant digits; ``answer`` records the final answer. |
| `audit_claims` | função | `(answer, tool_log, lab, recommended_run=None, rtol=0.005) -> dict[str, Any]` | Every number in ``answer`` must match (within ``rtol``, or as a rounding) a number the tools returned or a run id/variable/budget figure; the recommended run must exist, have succeeded and meet every limit. |
| `run_agent` | função | `(text, simulator, folder, *, domain, planner=None, provider=None, model=None, llm=None, m…)` | Interpret ``text``, build the lab and let the agent drive it through :func:`lab_toolbox`. Planner: the given ``planner``, else an :class:`~pinneapple_autolab.core.LLMStepPlanner` when ``provider`` or ``llm`` is given ("anthropic", "openai", "openai_compatible", "ollama"), else :class:`LabPlanner`. |

#### `pinneapple_autolab.core`

Mission -> plan -> controlled tool calls -> evidence: the agent loop of the Autonomous Physics Lab.

<small>`pinneapple_autolab/core.py` · 297 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Tool` | classe | `(name, fn, description, inputs, summary)` | A library call the agent may make: ``fn(state, **args) -> dict`` (the dict lands in ``state[name]``). ``inputs``: argument name -> description (the only arguments accepted); ``summary``: result -> short dict for the evidence log (default: the scalar entries of the result). |
| `Toolbox` | classe | `(kind, tools=None)` | The controlled set of tools of one mission kind. |
| `Step` | classe | `(tool, args, why)` |  |
| `Plan` | classe | `(mission, steps, planner, notes)` |  |
| `Planner` | classe | `()` |  |
| `TemplatePlanner` | classe | `(steps)` | The fixed sequence of steps of a mission kind. |
| `LLMPlanner` | classe | `(fallback, *, provider='anthropic', model=None, llm=None, required=())` | A language model orders the toolbox's tools into a plan; the template plan is the fallback. |
| `Evidence` | classe | `(step, tool, args, why, summary, seconds, ok, error)` |  |
| `Run` | classe | `(plan, state, evidence)` |  |
| `StepPlanner` | classe | `(name)` | Chooses the next step after seeing every result so far (``None``: done). |
| `LLMStepPlanner` | classe | `(*, provider='anthropic', model=None, llm=None, rules='')` | A language model chooses each next step from the toolbox, seeing the mission, the tools and the evidence so far (summaries only: the numbers the tools returned). It answers ``{"tool", "args", "why"}`` or ``{"done": true}``. An unknown tool or argument is not run: the refusal is recorded as evidence … |
| `execute` | função | `(plan, toolbox, state=None, *, stop_on_error=True, log=None) -> Run` | Run the plan's steps in order through the toolbox only. Each tool sees the shared ``state`` (earlier outputs by tool name, plus whatever the caller put there; ``_plan`` and ``_evidence`` hold the plan and the evidence so far) and returns a dict stored under its name. |
| `act` | função | `(mission, toolbox, planner, state=None, *, max_steps=200, max_refusals=5, log=None) -> Run` | Decide-then-act loop: ask ``planner`` for one step, check it against the toolbox, run it, record the evidence, repeat until the planner is done, ``max_steps`` or ``max_refusals`` consecutive refused / failed steps. Same :class:`Tool` / :class:`Evidence` contract as :func:`execute`; the returned … |

#### `pinneapple_autolab.dashboard`

The lab's dashboard: four synchronised areas over the real state of a session.

<small>`pinneapple_autolab/dashboard.py` · 233 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `live_state` | função | `(folder, out_dir=None, tail=120) -> dict[str, Any]` | The session's state from its files: ``ledger.json`` (runs), ``events.jsonl`` (what is happening), ``snapshot.json`` / ``summary.json`` when written (in ``folder`` or ``out_dir``). ``status`` says what is going on now: ``running`` (the run being simulated), ``training``, ``phase`` (last stage), … |
| `build_dashboard` | função | `(folder, out_dir, *, scenes=None, title='', state=None, live=False, replay=False, figures…)` | Write ``out_dir/index.html``: the dashboard with the current state embedded (``live=True``: it also polls ``state.json`` next to it, see :func:`serve`; ``replay=True``: a slider and a play button step through the recorded events, every number the recorded one). ``scenes``: name -> studio Scene … |
| `serve` | função | `(folder, out_dir, port=8120, host='127.0.0.1', block=True, **kw)` | Serve the dashboard of ``folder`` live: ``/`` the page (rebuilt once), ``/state.json`` the state read from the session's files at every request. ``block=False`` returns the server (``.shutdown()`` to stop). |
| `capture` | função | `(html, png, *, width=1600, height=1100, full_page=True, replay_to=None, wait_ms=5000) -> …)` | Screenshot of a dashboard page (``replay_to``: the replay at that event). Needs ``playwright``. |
| `record_replay` | função | `(html, mp4, *, frames=60, fps=4.0, hold_s=3.0, width=1600, height=2040, out_width=1280, c…)` | An MP4 of the dashboard's replay (the recorded events, evenly sampled), via ffmpeg. Needs ``playwright``. |

#### `pinneapple_autolab.domain`

Where the surrogate has evidence: the validated domain of a set of simulated designs.

<small>`pinneapple_autolab/domain.py` · 72 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DomainVerdict` | classe | `(inside, ratio, nearest, reasons)` |  |
| `ValidatedDomain` | classe | `(U, names=None, bounds=None, max_ratio=1.5, margin=0.05)` |  |

#### `pinneapple_autolab.domains`

Simulators and missions of the Autonomous Physics Lab, one module per physical domain.

<small>`pinneapple_autolab/domains/__init__.py` · 2 linhas · **sem teste direto**</small>

#### `pinneapple_autolab.domains.racecar`

The AI Aerodynamicist's simulator: a parametric open-wheel racing car in OpenFOAM, and its flagship mission.

<small>`pinneapple_autolab/domains/racecar.py` · 464 linhas · testes: 11</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RaceCarAero` | classe | `(resolution='campaign', iterations=800, procs=4, n_lines=40, keep_cases=False, conditions…)` | OpenFOAM external aerodynamics of :class:`FormulaCar`. ``resolution``: a key of :data:`MESHES` ("coarse" 0.14 M cells, "medium" 0.21 M, "fine" 0.82 M cells at the baseline, "campaign": medium with the wings and the floor at the fine levels); ``iterations`` of simpleFoam. Forces are written every … |
| `make_car` | função | `(params)` |  |
| `force_convergence` | função | `(history, window=2) -> dict[str, float]` | Relative change of drag and downforce over the last ``window`` written intervals (e.g. the last 200 iterations when forces are written every 100). |
| `solver_log_summary` | função | `(case) -> dict[str, Any]` | Last iteration, final initial residuals per field, whether checkMesh passed, snappy cells and timings. |
| `render_run` | função | `(params, flow_npz, out_prefix, title='', size=(1200, 675)) -> dict[str, str]` | PyVista renders of one simulated design from its saved flow scene: surface Cp (jet, from above and below) and the streamlines coloured by speed. Returns {name: path}. |
| `cycles_renders` | função | `(params, flow_npz, out_prefix, title='', samples=64, size=(1600, 900)) -> dict[str, str]` | Blender Cycles renders of a simulated design (needs ``bpy``): the streamlines coloured by speed on black, and the surface Cp in the studio, each with its jet colour bar. Returns {} without ``bpy``. |
| `flow_scene` | função | `(params, flow_npz, title='')` | The studio Scene of a simulated design (car parts + skin Cp/Cf, streamlines, slices) from its flow file. |
| `build_dashboard` | função | `(work, out_dir, summary, *, figures=None, figure_runs=None, viewers=True, replay=True, re…)` | The lab dashboard of a race-car session: renders of the original and the recommended design (surface Cp, surface mesh, streamlines, mid-plane Cp, wake vorticity) from their saved flow files, their 3-D studio viewers, and the recorded session (``work``: the lab folder with ledger.json and … |
| `recorded_dashboard` | função | `(results_dir, out_dir) -> str` | The dashboard of a recorded session from its committed files only (``ledger.json``, ``events.jsonl``, ``summary.json``, ``snapshot.json`` and the renders in ``figures/``): no OpenFOAM, no flow files, so no 3-D viewer; the replay steps through the recorded events. |
| `racecar_domain` | função | `()` | What the race-car simulator can be asked, for :func:`pinneapple_autolab.agent.interpret_mission`. |
| `flagship_mission` | função | `(budget=30, reserve_verify=3) -> Mission` | "Find a lower-drag aerodynamic design at 70 m/s. Preserve the required downforce, stay within the geometry constraints, and validate the final design using CFD." |

#### `pinneapple_autolab.lab`

The autonomous physics lab: plan -> build -> simulate -> learn -> optimise -> verify -> deliver, on any simulator.

<small>`pinneapple_autolab/lab.py` · 546 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AutonomousLab` | classe | `(mission, simulator, folder, *, families=…, field_family='gp', seed=0, nsga_budget=3000, …)` | ``critical_region``: name of a simulator region whose field error decides the trust in a constraint (e.g. the diffuser for downforce); ``field_tol``: the tolerated predicted std of that field there. ``probe(params, **options)`` runs another fidelity (e.g. a finer mesh) that never joins the training … |
| `showcase` | função | `(decisions, baseline) -> dict[str, dict[str, Any]]` | Three simulated designs the lab judged differently before simulating them, by a fixed rule: |

#### `pinneapple_autolab.ledger`

The evidence ledger: every simulation the lab ran (why, with what, what it cost, what came out) and every event of the session, in the order they happened.

<small>`pinneapple_autolab/ledger.py` · 185 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BudgetExhaustedError` | classe | `()` | The mission's simulation budget has been spent. |
| `RunRecord` | classe | `(id, purpose, params, ok, outputs, message, cost_s, started, reason, predicted, meta, fie…)` | One reference simulation. ``outputs`` holds only what the solver produced (empty when it failed); ``predicted`` what the surrogate said before the run (for the held-out error), ``reason`` why it was run. |
| `Ledger` | classe | `(folder, max_runs=None, max_wall_s=None, listeners=None)` | Runs and events of one lab session, kept in ``folder``. ``listeners``: callables ``f(event_dict)`` called on every event (a live dashboard, a test). |
| `environment` | função | `(extra=None) -> dict[str, Any]` | Hardware and software of this machine: CPU, cores, memory, OS, Python and package versions, git commit. |

#### `pinneapple_autolab.mission`

A mission as numbers: what to improve, what must hold, what may change, how much simulation it may spend.

<small>`pinneapple_autolab/mission.py` · 161 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `UnresolvedLimitError` | classe | `()` | A limit relative to the baseline was used before the baseline was simulated. |
| `Budget` | classe | `(max_runs, reserve_verify, max_wall_s)` | Simulation budget. ``max_runs`` reference simulations in total, of which ``reserve_verify`` are kept for the final verification of the chosen designs; ``max_wall_s`` (optional) caps the summed solver wall time. |
| `Limit` | classe | `(name, lower, upper, lower_frac_of_baseline, upper_frac_of_baseline, unit, kind, label)` | ``lower <= output <= upper``; a bound can instead be a fraction of the baseline's value (``lower_frac_of_baseline=1.0``: at least the baseline's). ``kind``: "physical", "geometry", "validity". |
| `Mission` | classe | `(title, statement, variables, objectives, constraints, baseline, budget, acceptance, cond…)` | What the lab is asked to do. ``acceptance``: relative tolerance per output for a prediction to count as confirmed by the reference solver (\|prediction - reference\| / \|reference\| <= tol). ``conditions``: the fixed operating and boundary conditions (speed, ground, ...); the lab never changes them. |

#### `pinneapple_autolab.qualification`

Before a campaign: how much of the reference solver's answer is the mesh and the iteration count?

<small>`pinneapple_autolab/qualification.py` · 49 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `solver_qualification` | função | `(sim, params, variants, folder, reference, campaign, outputs=None) -> dict[str, Any]` |  |

#### `pinneapple_autolab.replay`

A simulator that answers from a ledger of measured runs when it holds the exact design, and runs the real simulator otherwise.

<small>`pinneapple_autolab/replay.py` · 68 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ReplaySimulator` | classe | `(simulator, ledger_path, fields_dir=None, atol=1e-09)` |  |

#### `pinneapple_autolab.report`

Figures and the engineering report of a lab session, drawn only from its snapshot (ledger runs, candidates, model board, decisions): nothing here computes a physical number.

<small>`pinneapple_autolab/report.py` · 348 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `shrink_png` | função | `(path, colors=256) -> str` | Re-save a PNG with an adaptive palette of ``colors`` (renders and dense scatter plots shrink 3-4x). |
| `pareto_front` | função | `(F) -> np.ndarray` | Indices of the non-dominated rows of ``F`` (all columns minimised), sorted by the first column. |
| `fig_pareto` | função | `(snap, path, x='drag_N', y='downforce_N', scale=0.001, units='kN', annotate=None) -> str` |  |
| `fig_models` | função | `(snap, path, outputs, labels=None, scale=None) -> str` |  |
| `fig_learning` | função | `(snap, path, outputs, objective, acceptance) -> str` |  |
| `fig_field_triptych` | função | `(points, ref, pred, path, title='', clim=(-1.5, 1.0), err_max=0.3, label='Cp', mirror=Tru…)` | Reference \| prediction \| \|error\| from above and from below (half-model points mirrored), colormap jet. |
| `fig_trust_cards` | função | `(cards, path, outputs, units=None) -> str` | One panel per card: ``{"tag", "label", "reasons", "predicted", "std", "reference", "decision", "action"}``; bars = prediction +- 1.96 std, diamond = reference (when simulated). |
| `write_report` | função | `(summary, path, figures=None, qualification=None, extra='') -> str` | The engineering report in Markdown: mission, plan, solver qualification, budget, models, decisions, recommendation (a simulated design), environment. |

#### `pinneapple_autolab.simulator`

The reference simulator the lab calls: a solver run on one design, its checks before it runs, its cost after.

<small>`pinneapple_autolab/simulator.py` · 91 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SimOutcome` | classe | `(outputs, ok, message, cost_s, fields, meta)` |  |
| `Simulator` | classe | `(name, fidelity, outputs, field_names, regions)` | Base class: set ``name``, ``fidelity``, ``outputs`` (and ``field_names`` / ``field_points`` / ``regions`` if it returns fields), implement :meth:`check` and :meth:`run`. |
| `FunctionSimulator` | classe | `(fn, outputs, *, name='function', checker=None, fail=None, field_names=(), regions=None, …)` | ``fn(params) -> {output: value}`` (optionally also ``"__fields__": {name: array}``); ``checker(params) -> list[str]``; ``fail(params) -> str \| None`` makes a run fail with that message (to test failure paths). |

#### `pinneapple_autolab.surrogates`

Surrogates of the reference solver, compared on data they did not see, each with an uncertainty.

<small>`pinneapple_autolab/surrogates.py` · 260 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GPModel` | classe | `(restarts=1, seed=0)` |  |
| `MLPEnsemble` | classe | `(n=5, hidden=(32, 32), seed=0, max_iter=600)` |  |
| `QuadraticRSM` | classe | `(alpha=1e-06)` |  |
| `ModelBoard` | classe | `(table, chosen, models, cv_pred, n_train, cv_y)` | Cross-validated scores of every family on every output, the choice per output and the fitted models. |
| `FieldSurrogate` | classe | `(family='gp', energy=0.99, max_modes=12)` | POD + per-mode regression of fields sampled on fixed points. ``energy``: fraction of the snapshot variance the retained modes keep (capped at ``max_modes`` and at n_snapshots - 1). |
| `make_model` | função | `(family, **kw) -> _Model` |  |
| `cross_validate` | função | `(family, U, y, k=5, seed=0) -> dict[str, Any]` | k-fold cross-validation of one family on one output; returns the metrics and the held-out predictions. |
| `compare_models` | função | `(U, Y, families=…, k=5, seed=0, select='rmse') -> ModelBoard` | Cross-validate every family on every output, choose the lowest held-out ``select`` metric per output (a family whose 95 % interval holds fewer than half the held-out points is only chosen if nothing else is available), refit the chosen family on all data. |
| `field_errors` | função | `(pred, ref, regions=None, std=None) -> dict[str, dict[str, float]]` | RMSE, max error and (with ``std``) the 95 % coverage of a predicted field, overall and per region. |

#### `pinneapple_autolab.trust`

Can this prediction be used for a decision? Triage of surrogate candidates and the choice of the next simulation.

<small>`pinneapple_autolab/trust.py` · 262 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CandidateAssessment` | classe | `(index, params, predicted, std, p_feasible, cei, domain, label, reasons, report, critical…)` |  |
| `NextExperiment` | classe | `(candidate, rule, reason, alternatives)` |  |
| `assess_candidates` | função | `(X, names, pred, objectives, constraints, domain_verdicts, best_value, acceptance, model_…)` | Assess candidates ``X`` (physical units, columns ``names``). ``pred``: output -> (mean, std) arrays; ``best_value``: the best simulated feasible value of the first objective (None: no feasible simulation yet, then the EI is replaced by the predicted improvement over the candidates' worst); … |
| `choose_next` | função | `(cands, objective, exclude=()) -> NextExperiment` | The next simulation, by the written rule in the module docstring. |
| `verification_checks` | função | `(c, reference, ok, acceptance, constraints, message='') -> CandidateAssessment` | Update an assessment with the reference result: measured relative errors, the confirmation check and the feasibility under the reference values. |

### Exemplos de uso reais do repositório

_Nenhum exemplo encontrado no repositório para este pacote._

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 6 de 16.
- Funções de teste em arquivos que importam algum módulo do pacote: 37.

### Limitações conhecidas

- _Nenhuma registrada._

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
