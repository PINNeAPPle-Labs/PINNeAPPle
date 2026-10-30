# Volume 5 — Confiança, validação e decisão

> Parte do [Mapa completo do PINNeAPPle](../mapa_completo.md). Gerado por `docs/pt/_mapa/gerar.py` a partir do inventário do commit `e53cdcfb04`.

A camada que confere cada resultado contra a física: validação, incerteza, inversão, o linter `pinneapple check`, registros de evidência, segurança, decisão e a cadeia de porquês.

## Sumário

1. [`pinneapple_analysis`](#1-pinneapple_analysis)
2. [`pinneapple_check`](#2-pinneapple_check)
3. [`pinneapple_veriphysics`](#3-pinneapple_veriphysics)
4. [`pinneapple_security`](#4-pinneapple_security)
5. [`pinneapple_decision`](#5-pinneapple_decision)
6. [`pinneapple_reason`](#6-pinneapple_reason)

## 1. `pinneapple_analysis`

### Objetivo

`pinneapple_analysis` reúne as ferramentas que olham para um modelo já treinado (ou para um conjunto de resultados de simulação) e perguntam: ele é confiável, quanto custa, quais parâmetros importam e onde ele pode ser usado. Esta parte do mapa cobre a raiz do pacote e os submódulos de incerteza (`uncertainty`), confiança (`trust`), sensibilidade (`sensitivity`), custo (`cost`), benchmark com orçamento igual (`benchmark`), surrogates (`surrogate`), assimilação de dados (`data_assimilation`, `state_estimation`), grafos de prova (`proof`), reprodução de artigos (`reproduction`), busca semântica (`retrieval`) e os contratos de estudo (`study_models`, `study_contracts`). Os submódulos `validation` e `inverse_problems` ficam em outra parte.

A raiz oferece dois atalhos: `analyze_model` (validação física + quantificação de incerteza numa chamada) e `invert` (problema inverso com erro gaussiano e regularização de Tikhonov via `InverseProblemSolver`). O submódulo `uncertainty` era o antigo `pinneapple_uq` e junta MC Dropout, ensembles, cabeças aleatóricas, predição conformal, calibração, regressão quantílica, propagação de tolerâncias, caos polinomial, drift e o limite de erro a partir do resíduo da EDP.

Vários submódulos seguem a mesma disciplina: todo número vem de uma execução medida, com dados de teste que nunca escolhem o modelo, e cada resultado vira checagens registradas (`Check`, `TrustReport`, cartão de confiança de seis perguntas do `pinneapple_lab`). O `benchmark` compara métodos (PINN, FNO, DeepONet, MeshGraphNet, MLP) sob o mesmo `ComputeBudget`, com portões de correção e um conselheiro que sugere ajustes; o `surrogate` treina todos os candidatos aplicáveis num conjunto de dados e empacota o melhor com um portão OOD.

A ligação com o resto vem pelos atalhos de `pinneapple` (`pp.benchmark`, `pp.uq`/`pp.uncertainty`, `pp.sensitivity`, `pp.assimilation`, `pp.ComputeBudget`, `pp.trust_card`), pelos contratos de `pinneapple_data.contracts` (`ParameterSpace`, `ModelRef`, `StudyResult`) usados pelos apps de incerteza, sensibilidade e surrogate, pelo zoo de `pinneapple_hub` (modelos `zoo:<nome>`) e pelo `pinneapple_reason`, que monta seu relatório de confiança com `pinneapple_analysis.trust.TrustReport`.

Esta parte cobre quatro submódulos de `pinneapple_analysis`. `inspection` gera sinais sintéticos de ensaio não destrutivo e treina PINNs com a mesma EDP. `inverse_problems` estima parâmetros a partir de medições, diz antes se eles são identificáveis e compara métodos (mínimos quadrados, adjunto, autograd, bayesiano, PINN, EKI). `validation` produz o relatório de `pp.validate` e as checagens que o alimentam, além da injeção de falhas e da arena red team / blue team. `verification` reúne os blocos do motor de verificação física (análise dimensional, convergência de malha, qualidade de malha, escore de confiança, recomendação de arquitetura, solver e ferramentas), servidos por `apps/verified_physics` e usados por outros apps.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 159 | 42.004 | 249 | 450 | 939 (83 arquivos) | 56 |

Pasta de testes dedicada: `tests/pinneapple_analysis` (169 funções de teste).

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `benchmark` | Benchmarks de métodos de IA física com orçamento igual: preparar artefatos, validar, rodar com cache e gerar relatório com checagens e diagnósticos. | 18 | 113 | 44 |
| `cost` | Mede custo computacional (tempo, memória, FLOPs, escala, orçamentos, regressões) e oferece o laço de treino que para num orçamento fixo. | 4 | 25 | 55 |
| `data_assimilation` | Assimilação de dados: 4D-Var forte por autograd, modelos caóticos de teste e reconstrução de campo por processo gaussiano com posicionamento guloso de sensores. | 4 | 13 | 7 |
| `inspection` | Suíte de ensaio não destrutivo (NDE): gera sinais de inspeção sintéticos para seis modalidades e treina uma PINN com a mesma EDP que gerou os dados. | 8 | 21 | 13 |
| `inverse_problems` | Problemas inversos: modelos diretos, identificabilidade, calibração, estimadores (mínimos quadrados, adjunto, autograd, MCMC, EKI, PINN), regularização, ruído, sensibilidade e descoberta de termos faltantes. | 19 | 95 | 151 |
| `proof` | Grafos de prova: argumentos matemáticos como grafos de hipóteses, lemas e teoremas com checagens automáticas e status que se propaga. | 3 | 14 | 11 |
| `reproduction` | Ferramentas neutras para reproduzir números de artigos: métricas de erro, estatística por semente e regra de status fixada antes da execução. | 1 | 10 | 5 |
| `retrieval` | Busca semântica local sobre um corpus real (docstrings, presets, código e, opcionalmente, artigos do arXiv) com embeddings de sentence-transformers. | 3 | 11 | 32 |
| `sensitivity` | Análise de sensibilidade global e local: Sobol com intervalos, Morris, curvas de efeito, derivadas e a regra de quais parâmetros podem ser congelados. | 6 | 21 | 28 |
| `state_estimation` | Filtros de Kalman estendido (EKF) e de ensemble (EnKF) para estimar estado sequencialmente a partir de observações. | 2 | 4 | 22 |
| `study_contracts` | Converte espaços de parâmetros, modelos e resultados de estudos de incerteza e sensibilidade para os contratos de `pinneapple_data.contracts`. | 1 | 7 | 4 |
| `study_models` | Interface única `StudyModel` (`predict(X) -> (n, m)` com nomes, unidades e faixa de validade) para estudos de incerteza e sensibilidade. | 1 | 3 | 25 |
| `surrogate` | Estudo de surrogates a partir de resultados de simulação: escolhe candidatos, compara com orçamento igual, mede em dados não vistos e empacota o melhor. | 11 | 41 | 32 |
| `trust` | Pontuação de confiança antes de consumir a predição de um modelo: `TrustGate`, cartão de confiança de campos e `TrustReport` com decisão de deploy. | 4 | 11 | 13 |
| `uncertainty` | Quantificação de incerteza (antigo `pinneapple_uq`): aleatória, epistêmica, conformal, calibração, quantis, tolerâncias, drift, caos polinomial e limites a partir do resíduo. | 21 | 79 | 185 |
| `validation` | Validação de modelos de IA física: o relatório único `pp.validate`, checagens clássicas de PINN (conservação, contorno, simetria), métricas de campo e espectrais, testes metamórficos, rollout, injeção de falhas e a arena red team / blue team. | 33 | 151 | 151 |
| `verification` | Blocos do motor de verificação física: números adimensionais, convergência de malha (GCI), qualidade de malha, rotulagem de geometria, guarda de fora-de-distribuição, grafo de conhecimento, escore de confiança, recomendações e pipeline com LLM. | 19 | 78 | 43 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_analysis`

Raiz do pacote de análise de modelos: reexporta validação, UQ e problemas inversos e oferece os atalhos `analyze_model` e `invert`.

<small>`pinneapple_analysis/__init__.py` · 173 linhas · testes: 56</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `analyze_model` | função | `(model, spec, x_test, *, uq_method='mc_dropout', run_validate=True, run_uq=True) -> dict` | Roda validação física e quantificação de incerteza numa chamada e devolve um dicionário combinado. |
| `invert` | função | `(model, y_obs, sensor_locs, *, noise_std=0.01, lambda_reg=0.001, method='adam', n_iters=2…)` | Atalho de problema inverso com desajuste gaussiano e regularização de Tikhonov, via `InverseProblemSolver`. |

#### `pinneapple_analysis.benchmark`

Entrada do benchmark com orçamento igual (preparar -> validar -> rodar -> relatório); reexporta `run`, `prepare`, `load`, o registro e `ArrayProblem`.

<small>`pinneapple_analysis/benchmark/__init__.py` · 34 linhas · testes: 44</small>

#### `pinneapple_analysis.benchmark.__main__`

Permite rodar `python -m pinneapple_analysis.benchmark`, repassando os argumentos à CLI. [sem teste]

<small>`pinneapple_analysis/benchmark/__main__.py` · 7 linhas · **sem teste direto**</small>

#### `pinneapple_analysis.benchmark.advisor`

Diagnósticos medidos no próprio run (históricos, níveis de orçamento, checagens, dados) com severidade, evidência e edição de configuração sugerida. [sem teste]

<small>`pinneapple_analysis/benchmark/advisor.py` · 222 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Diagnosis` | classe | `(severity, topic, subject, finding, evidence, suggestion, edit)` | Um diagnóstico: severidade, tópico, evidência medida, sugestão e, quando possível, a edição exata da configuração. |
| `diagnose` | função | `(cfg, plan, extracted, checks, arch_cfgs=None, loss_terms=None) -> list[Diagnosis]` | Gera a lista de diagnósticos a partir da configuração, do plano, das métricas extraídas e das checagens. |
| `apply_suggestions` | função | `(cfg, diagnoses, path) -> BenchmarkConfig` | Grava uma nova configuração com todas as edições sugeridas, para o usuário revisar e rodar. |

#### `pinneapple_analysis.benchmark.api`

API `pp.benchmark`: `prepare`, `run`, `load`, `report` e `BenchmarkResult`, com portões de correção e o conselheiro. [sem teste direto]

<small>`pinneapple_analysis/benchmark/api.py` · 284 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BenchmarkResult` | classe | `(ws, cfg, plan, arch, banners, blocked)` | Resultado de um benchmark aberto: tabela, relatório, checagens, diagnósticos, exportações e checagem pontual de determinismo. |
| `prepare` | função | `(config=None, out_dir='benchmark_out', *, force=False, adopt=False, log=print, **kwargs) …)` | Cria e valida os artefatos (config do solver, geometria, dados, custo, hashes) em `out_dir/prepared/` antes de qualquer treino. |
| `run` | função | `(config=None, out_dir='benchmark_out', *, force=False, prepare_missing=True, stages=…, lo…)` | Roda o benchmark; retoma do cache no mesmo `out_dir` e recusa uma configuração diferente sem `force`. |
| `load` | função | `(out_dir) -> BenchmarkResult` | Abre uma pasta de benchmark terminada ou parcial. |
| `report` | função | `(out_dir) -> BenchmarkResult` | Abre a pasta com `load` e regenera o relatório. |

#### `pinneapple_analysis.benchmark.builtin`

Importa e registra os problemas e métodos embutidos; `airfoil` e `inverse_heat` só entram se suas dependências opcionais importarem. [sem teste]

<small>`pinneapple_analysis/benchmark/builtin/__init__.py` · 9 linhas · **sem teste direto**</small>

#### `pinneapple_analysis.benchmark.builtin.airfoil`

Problema `airfoil` (aerofólio 2-D, OpenFOAM simpleFoam k-omega SST, Re 4e6) com métodos MLP, POD + MLP, MeshGraphNet e PINN por geometria; `prepare` exige OpenFOAM. [sem teste]

<small>`pinneapple_analysis/benchmark/builtin/airfoil.py` · 654 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AirfoilProblem` | classe | `()` | Problema registrado `airfoil`: prepara dados OpenFOAM, verifica o solver, define colunas e métricas do aerofólio 2-D. |
| `MGNMethod` | classe | `()` | Método `mgn` (MeshGraphNet) cujo treino é delegado aos ganchos do problema. |
| `MLPMethod` | classe | `()` | Método `mlp` que prevê os coeficientes integrais via ganchos do problema. |
| `MLPFieldsMethod` | classe | `()` | Método `mlp_fields` (POD + MLP) que prevê campos via ganchos do problema. |
| `PINNRansMethod` | classe | `()` | Método `pinn_rans`: PINN RANS por instância (uma geometria e um ângulo). |
| `openfoam_available` | função | `() -> bool` | Diz se o bashrc do OpenFOAM existe (`FOAM_BASHRC` ou o caminho padrão). |

#### `pinneapple_analysis.benchmark.builtin.airfoil_impl`

Implementação do benchmark do aerofólio vinda do experimento 09 do PINNeAPPle-Benchmark: treino de MLP em ensemble, MeshGraphNet, POD + MLP e PINNs de escoamento potencial e RANS. [sem teste]

<small>`pinneapple_analysis/benchmark/builtin/airfoil_impl.py` · 599 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PotentialNet` | classe | `(cfg, alpha_deg)` | Rede de escoamento potencial com circulação treinável e condição de parede imposta na forma da função de corrente. |
| `RansNet` | classe | `(cfg, alpha_deg, wall)` | Rede (u, v, p) com não-deslizamento exato na parede, para a PINN RANS. |
| `EddyViscosity` | classe | `(model, p0, e)` | Viscosidade turbulenta por comprimento de mistura de Prandtl com amortecimento de van Driest, congelada a cada passo (linearização de Picard). |
| `set_data_dir` | função | `(prepared) -> None` | Define a pasta de dados preparados usada pelas funções de treino. |
| `load_runs` | função | `(dirs)` | Carrega os runs aplicando o filtro do app aero (status utilizável, sem estol profundo, oscilantes promediados). |
| `split` | função | `(recs, frac=0.15, seed=3)` | Divisão do app aero: uma fração das formas LHS reservada com todos os seus ângulos; os polares NACA à parte. |
| `splits` | função | `()` | Monta os conjuntos treino, validação (formas sorteadas), teste, NACA e OOD a partir dos runs. |
| `eval_splits` | função | `(final, S)` | Lista os conjuntos a avaliar: teste, NACA e OOD no run final, só validação na busca. |
| `coefs` | função | `(recs)` | Extrai a matriz (Cl, Cd, Cm) dos registros. |
| `coef_scores` | função | `(true, pred) -> dict` | Erros absolutos de Cl e Cm e erro relativo de Cd, por run e em média. |
| `field_scores` | função | `(true, pred) -> dict` | L2 relativo por run da velocidade e da pressão nos nós do grafo. |
| `train_mlp` | função | `(cfg, seed, budget, *, final) -> dict` | Treina o ensemble de MLPs dos coeficientes dentro do orçamento e devolve as métricas. |
| `graph_data` | função | `(lay, recs)` | Monta features de nós e arestas e alvos de campo e coeficientes para o MeshGraphNet. |
| `train_mgn` | função | `(cfg, seed, budget, *, final, batch=4) -> dict` | Treina o MeshGraphNet (campos e coeficientes) dentro do orçamento. |
| `train_mlp_fields` | função | `(cfg, seed, budget, *, final) -> dict` | Baseline de campo: POD dos campos de treino e MLP dos parâmetros de forma e ângulo para os coeficientes POD. |
| `find_case` | função | `(case) -> dict` | Acha um registro de caso pelo id nos conjuntos de avaliação. |
| `collocation_pool` | função | `(shape, d_min) -> torch.Tensor` | Centros de célula da O-grid do caso dentro do raio do domínio, usados como pontos de colocação. |
| `wall_distance` | função | `(x, y, p0, e)` | Distância ao polígono da parede, diferenciável quase em toda parte. |
| `rans_residuals` | função | `() -> list[str]` | Resíduos das equações RANS estacionárias incompressíveis com viscosidade turbulenta (continuidade e momento x e y). |
| `train_pinn` | função | `(cfg, seed, budget, *, case, kind='potential', ckpt=None) -> dict` | Treina uma PINN (potencial ou RANS) para um aerofólio num ângulo, sem dados de simulação. |

#### `pinneapple_analysis.benchmark.builtin.fields`

Problemas de operador campo -> campo em grade, o `ArrayProblem` para arrays do usuário e os métodos genéricos FNO, DeepONet e PINN por instância; ganchos base levantam NotImplementedError.

<small>`pinneapple_analysis/benchmark/builtin/fields.py` · 614 linhas · testes: 29 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FieldData` | classe | `(prepared)` | Arrays preparados com normalizadores gaussianos pontuais calculados só no treino. |
| `FieldOperatorProblem` | classe | `()` | Base dos problemas grade -> grade; reporta L2 relativo do campo e erro relativo de uma integral nos conjuntos PINN, teste e OOD. |
| `ArrayProblem` | classe | `(id, *, train, val, test, ood=None, data_cost_s=None, qoi=None, pinn_setup=None, title=No…)` | Envolve arrays do próprio usuário como problema de benchmark; sem `pinn_setup` o PINN levanta NotImplementedError. |
| `FNOMethod` | classe | `()` | Método `fno` (Fourier Neural Operator) para problemas de campo, com checagem de modos de Fourier. |
| `DeepONetMethod` | classe | `()` | Método `deeponet` para problemas de campo, com checagem das dimensões de branch e trunk. |
| `PINNMethod` | classe | `()` | Método `pinn`: uma PINN por instância via `pinn_setup` do problema ou seu próprio `train_pinn`. |
| `train_operator` | função | `(problem, prepared, method, cfg, seed, budget, *, final=True, **_) -> dict` | Treina um operador (FNO ou DeepONet) com Adam dentro do orçamento e devolve as métricas. |
| `train_field_pinn` | função | `(problem, prepared, cfg, seed, budget, *, split, instance, **_)` | Treina uma PINN para uma instância usando `PINNFactory` e avalia contra o campo verdadeiro. |
| `pinn_checks` | função | `(problem, prepared, cfg) -> list[Check]` | Portões da PINN: derivadas autograd contra diferenças finitas, resíduo de uma solução exata e ajuste de um conjunto mínimo. |
| `finite` | função | `(x) -> bool` | Diz se um valor converte para float finito. |

#### `pinneapple_analysis.benchmark.builtin.inverse_heat`

Problema `inverse_heat`: estimar o coeficiente de convecção h de uma placa a partir de 8 termopares. [sem teste]

<small>`pinneapple_analysis/benchmark/builtin/inverse_heat.py` · 201 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `InverseHeatProblem` | classe | `()` | Problema registrado `inverse_heat`: prepara dados de volume finito, verifica o solver e define as métricas do coeficiente h. |

#### `pinneapple_analysis.benchmark.builtin.inverse_heat_impl`

Implementação do benchmark de calor inverso: PINN da placa contra surrogate + calibração (MLP, DeepONet) e contra o volume finito com busca de seção áurea. [sem teste]

<small>`pinneapple_analysis/benchmark/builtin/inverse_heat_impl.py` · 228 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Surrogate` | classe | `(method, cfg, xy)` | Rede que mapeia (log h, potência) normalizados para o campo de temperatura 60 x 100 normalizado. |
| `Data` | classe | `()` | Carrega os dados preparados e normaliza as entradas do surrogate. |
| `set_data_dir` | função | `(prepared, smoke=False) -> None` | Define a pasta de dados preparados e o modo smoke. |
| `single_thread` | função | `()` | Força uma thread (OMP e torch) no processo atual. |
| `fv_field` | função | `(h, power)` | Campo de temperatura da placa pelo modelo de volume finito para um h e uma potência. |
| `readings` | função | `(power, h_true, seed)` | Leituras ruidosas dos termopares interpoladas do campo de referência. |
| `generate` | função | `(out, n_train, n_val, workers=4) -> dict` | Gera em paralelo os campos de treino e validação por hipercubo latino em (h, potência) e mede o custo. |
| `train_surrogate` | função | `(method, cfg, seed, budget)` | Treina o surrogate MLP ou DeepONet dentro do orçamento. |
| `calibrate` | função | `(field_fn, power, t_meas, x, y)` | Busca de seção áurea de h que minimiza o desajuste nos sensores de um campo dado. |
| `surrogate_job` | função | `(method, cfg, seed, budget, *, final) -> dict` | Job de surrogate + calibração: treina e, no run final, estima h nos casos. |
| `pinn_job` | função | `(cfg, seed, budget, *, kind, case) -> dict` | Job da PINN da placa num caso: estima h e mede erros de campo e de ponto quente. |
| `classical_job` | função | `(seed) -> dict` | Busca de seção áurea usando o próprio modelo de volume finito como 'surrogate'. |

#### `pinneapple_analysis.benchmark.builtin.standard`

Problemas padrão de aprendizado de operador (Burgers, Darcy, Navier-Stokes de Li et al. 2021) com formulação PINN de condições rígidas e solução exata para o portão de resíduo.

<small>`pinneapple_analysis/benchmark/builtin/standard.py` · 478 linhas · testes: 16 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Burgers` | classe | `()` | Problema `burgers` (1-D periódico, u(.,0) -> u(.,T)) com formulação PINN e solução exata. |
| `Darcy` | classe | `()` | Problema `darcy` (-div(a grad u) = 1, u = 0 na borda, a -> u) com formulação PINN. |
| `NavierStokes` | classe | `()` | Problema `navier_stokes` (vorticidade 2-D periódica, w(.,0) -> w(.,T)) com formulação PINN. |
| `Bilinear` | classe | `(a)` | Interpolante bilinear de um campo centrado em células e seu gradiente, sem autograd. |
| `HardNet` | função | `(problem, cfg, init=None, t_end=10.0)` | Cria uma `NeuralNetwork` da biblioteca com features de entrada e condições iniciais/de contorno rígidas. |

#### `pinneapple_analysis.benchmark.checks`

Portões de correção de um run: checagens de arquitetura, de cada job, de determinismo, de dispersão entre sementes e o status de cada linha. [sem teste direto]

<small>`pinneapple_analysis/benchmark/checks.py` · 175 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `architecture_gates` | função | `(plan) -> list[Check]` | Roda as checagens de arquitetura de cada par problema/método; um portão que quebra conta como falho. |
| `job_checks` | função | `(rec) -> list[Check]` | Checagens de um job terminado, a partir do seu arquivo de run. |
| `training_gates` | função | `(ws, plan) -> tuple[list[Check], dict[str, list[Check]]]` | Todas as checagens de jobs agregadas por (problema, método) mais as checagens entre runs. |
| `determinism_check` | função | `(plan, pid, mid, steps=20) -> Check` | Roda um job duas vezes com a mesma semente e orçamento de passos; os resultados devem bater. |
| `seed_spread` | função | `(values, subj, what) -> Check` | Checa se a dispersão relativa entre sementes está dentro da tolerância do `pinneapple_veriphysics`. |
| `row_status` | função | `(arch, train, skipped, did_not_converge) -> str` | Status de uma linha da tabela: checks skipped, INVALID, did not converge ou com avisos. |
| `summarise` | função | `(checks) -> dict[str, Any]` | Conta checagens por status. |

#### `pinneapple_analysis.benchmark.cli`

Comando `pinneapple benchmark prepare|run|report` (também `python -m pinneapple_analysis.benchmark`).

<small>`pinneapple_analysis/benchmark/cli.py` · 64 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `main` | função | `(argv=None) -> int` | Ponto de entrada da CLI `pinneapple benchmark prepare\|run\|report`. |

#### `pinneapple_analysis.benchmark.config`

`BenchmarkConfig`: os mesmos campos vindos de kwargs Python ou de YAML, resolvidos e com hash. [sem teste direto]

<small>`pinneapple_analysis/benchmark/config.py` · 148 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BenchmarkConfig` | classe | `(problems, methods, budget, problem_budgets, budget_levels, seeds, n_trials, search_fract…)` | Tudo de que um run depende (problemas, métodos, orçamentos, sementes, busca, FLOPs iguais, workers); lê e grava YAML e calcula hash. |

#### `pinneapple_analysis.benchmark.registry`

Classes base `Problem` e `Method`, `Check`, `Column` e o registro com decoradores `register_problem`/`register_method`; os ganchos não implementados levantam NotImplementedError. [sem teste direto]

<small>`pinneapple_analysis/benchmark/registry.py` · 251 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Check` | classe | `(name, stage, subject, passed, value, threshold, detail, severity)` | Um portão ou item de validação; `passed` None significa não aplicável, sempre com motivo. |
| `Column` | classe | `(key, label, set, metric, kind, pct)` | Uma coluna da tabela: uma métrica das instâncias de um conjunto, de campo ou integral. |
| `Problem` | classe | `(**options)` | Classe base de problema de benchmark; subclasses definem `id`, `title` e os ganchos de que precisam. |
| `Method` | classe | `(id, title, per_instance, outputs)` | Classe base de método de benchmark (espaço de hiperparâmetros, treino, checagens de arquitetura). |
| `register_problem` | função | `(problem_id)` | Decorador que registra uma classe de problema sob um id. |
| `register_method` | função | `(method_id)` | Decorador que registra uma classe de método sob um id. |
| `get_problem` | função | `(p) -> Problem` | Devolve a instância de problema registrada para um id (ou a própria instância). |
| `get_method` | função | `(m) -> Method` | Devolve a instância de método registrada para um id (ou a própria instância). |
| `problems` | função | `() -> list[str]` | Lista os ids de problemas registrados. |
| `methods` | função | `() -> list[str]` | Lista os ids de métodos registrados. |

#### `pinneapple_analysis.benchmark.report`

Relatório do benchmark: tabelas com n em cada cabeçalho, custo de dados, break-even, seções de checagens, validação, cartão de confiança e diagnósticos, figuras e summary.json.

<small>`pinneapple_analysis/benchmark/report.py` · 648 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `fmt_s` | função | `(x) -> str` | Formata segundos em µs, ms, s ou h. |
| `collect` | função | `(plan) -> dict` | Lê os arquivos de run e monta {(problema, método): {nível: {semente: métricas}}} mais custos. |
| `equal_flops_table` | função | `(plan, col, pid) -> dict \| None` | Linhas do braço de FLOPs iguais: métrica principal nas instâncias comuns a todos e no conjunto completo. |
| `cell` | função | `(seeds, set_, metric) -> dict \| None` | Célula da tabela: média e desvio entre sementes de uma métrica num conjunto. |
| `build_tables` | função | `(plan, col, row_status) -> dict` | Monta as linhas por problema com células, custos e status. |
| `markdown` | função | `(plan, tables, checks, validation, trust, diagnoses, banners, studies=None) -> str` | Escreve o relatório Markdown completo (hardware, tabelas, checagens, validação, confiança, diagnósticos, estudos). |
| `write_csv` | função | `(path, plan, tables) -> None` | Grava as tabelas em CSV, uma linha por célula. |
| `figures` | função | `(plan, col, tables, fig_dir, banners) -> list[str]` | Desenha as figuras do benchmark (erro contra orçamento, resumo) com matplotlib. |
| `summary_metrics` | função | `(prob)` | Linhas da figura de resumo de um problema: rótulo, tipo, métricas de teste e OOD. |
| `to_dataframe` | função | `(plan, tables)` | Converte as tabelas num `pandas.DataFrame`. |
| `collect_studies` | função | `(plan) -> dict` | Lê dos runs o estudo de orçamento e as ablações (auditoria do setup). |
| `studies_markdown` | função | `(plan, studies) -> list[str]` | Seções Markdown do estudo de orçamento e das ablações. |
| `study_metric_key` | função | `(prob, method_id, pts) -> str \| None` | Escolhe a métrica da curva do estudo de orçamento: a métrica de convergência do próprio método, ou a primeira coluna. |
| `studies_figure` | função | `(plan, studies, fig_dir) -> list[str]` | Figura das curvas do estudo de orçamento, um painel por par problema/método. |

#### `pinneapple_analysis.benchmark.runner`

Maquinário do benchmark: workspace em out_dir, estágio de preparo, cache de jobs e estágios de busca, final e FLOPs iguais, um processo por job. [sem teste direto]

<small>`pinneapple_analysis/benchmark/runner.py` · 397 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ConfigMismatch` | classe | `()` | Erro quando o `out_dir` já tem uma configuração diferente e `force` não foi dado. |
| `PreparationError` | classe | `()` | Erro quando os artefatos preparados faltam, estão velhos ou falharam na validação. |
| `Workspace` | classe | `(out_dir, config)` | A pasta `out_dir`: caminhos de runs e preparados, guarda de configuração e leitura de resultados. |
| `Plan` | classe | `(ws, cfg, problems, methods)` | Especificações dos jobs de um benchmark (busca, final, estudo, ablação, FLOPs iguais), com ids compatíveis com o experimento 09. |
| `prepare_problem` | função | `(ws, problem, *, n_workers=1, force=False, adopt=False, log=print) -> dict[str, Any]` | Cria ou reutiliza os artefatos de um problema, verifica o solver, checa os dados e roda a validação; devolve o manifesto. |
| `require_prepared` | função | `(ws, problem, allow_failed) -> dict` | Exige artefatos preparados, atuais e com hashes corretos; senão levanta `PreparationError`. |
| `run_job` | função | `(spec, out_dir, objects=None) -> str` | Roda um job neste processo e grava `runs/<id>.json` com o resultado ou o traceback. |
| `execute_jobs` | função | `(ws, jobs, *, n_workers, objects=None, log=print) -> None` | Executa os jobs pendentes, os mais longos primeiro, em processos paralelos ou em série. |
| `lvl` | função | `(x) -> str` | Rótulo de nível de orçamento no formato `L0.333`. |

#### `pinneapple_analysis.benchmark.validation`

Aplica a pilha de validação do PINNeAPPle (auditoria física, envelope, GCI, malha, y+, preflight) aos artefatos preparados e monta o cartão de confiança.

<small>`pinneapple_analysis/benchmark/validation.py` · 159 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `physics_audit_check` | função | `(subject, name, family, equations, assumptions, features) -> Check` | Checagem de auditoria física de uma formulação (família, equações, hipóteses). |
| `envelope_checks` | função | `(subject, train, test, ood, features=None) -> list[Check]` | Checagens de envelope de aplicabilidade: teste dentro e OOD fora do envelope do treino. |
| `gci_check` | função | `(subject, quantity, values_fine_to_coarse, sizes_fine_to_coarse, max_gci=0.05) -> Check` | Checagem do índice de convergência de malha (GCI) de uma grandeza contra um limite. |
| `mesh_quality_check` | função | `(subject, mesh, target='cfd') -> Check` | Checagem de qualidade de malha para um alvo (por padrão CFD). |
| `preflight_check` | função | `(subject, case_dir) -> Check` | Checagem preflight de um caso de solver. |
| `yplus_check` | função | `(subject, first_cell, length, velocity, nu, target=1.0) -> Check` | y+ da primeira célula pela estimativa de placa plana contra um alvo. |
| `field_features` | função | `(x) -> dict[str, float]` | Descritores escalares de um campo de entrada (amplitude e centróide espectral) para o envelope. |
| `field_validation` | função | `(problem, out, grid_study=None) -> list[Check]` | Checagens de validação de um problema de campo, incluindo estudo de grade opcional. |
| `trust_card` | função | `(out_dir, checks, tier, limitations) -> dict[str, Any]` | Responde as seis perguntas do cartão de confiança do `pinneapple_lab` a partir das checagens do benchmark. |

#### `pinneapple_analysis.cost`

Validação de custo computacional por operação; reexporta `profiler`, `equal_budget` e `torch_ops`.

<small>`pinneapple_analysis/cost/__init__.py` · 50 linhas · testes: 42</small>

#### `pinneapple_analysis.cost.equal_budget`

Comparação com orçamento igual: `ComputeBudget`, laço `train_within_budget`, contagem de FLOPs (inclusive `autograd.grad`), busca aleatória e custo amortizado.

<small>`pinneapple_analysis/cost/equal_budget.py` · 446 linhas · testes: 55</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ComputeBudget` | classe | `(wall_s, flops, steps)` | Orçamento de treino em segundos, FLOPs e/ou passos (None = sem limite). |
| `BudgetRun` | classe | `(steps, wall_s, flops, flops_per_step, stopped_by, budget, history, eval_wall_s)` | O que um treino com orçamento gastou e como foi (passos, tempo, FLOPs, motivo da parada, histórico). |
| `SearchTrial` | classe | `(config, score, info)` | Uma tentativa da busca: configuração, score e informações. |
| `SearchResult` | classe | `(best_config, best_score, trials, n_trials, seed)` | Resultado da busca: melhor configuração, melhor score e todas as tentativas. |
| `AmortizedCost` | classe | `(data_s, train_s, infer_s, per_query_train_s)` | Custo de responder novas consultas (dados, treino, inferência) com total, por consulta e ponto de equilíbrio. |
| `count_flops` | função | `(fn) -> tuple[int, dict[str, int], Any]` | Executa `fn()` uma vez e conta os FLOPs de forward e backward, inclusive `torch.autograd.grad`. |
| `cosine_lr` | função | `(progress, lr0, final_fraction=0.03) -> float` | Taxa de aprendizado com decaimento cosseno ao longo do orçamento. |
| `set_lr` | função | `(optimizer, lr) -> None` | Define a taxa de aprendizado em todos os grupos do otimizador. |
| `train_within_budget` | função | `(step, budget, *, flops_per_step=None, count_first_step=True, evaluate=None, eval_at=(), …)` | Chama `step(progress)` até gastar o orçamento, com avaliações, log e checkpoints opcionais. |
| `equal_budget_search` | função | `(space, objective, *, n_trials, seed=0, minimize=True) -> SearchResult` | Busca aleatória sobre uma grade com exatamente `n_trials` configurações distintas. |
| `relative_l2` | função | `(pred, true, *, sample_axis=0, eps=1e-12) -> np.ndarray` | Erro L2 relativo por amostra. |
| `time_per_query` | função | `(fn, *, repeats=20, warmup=3, n_per_call=1, no_grad=True) -> dict[str, float]` | Tempo mediano de `fn()` por instância respondida, por padrão sob `no_grad`. |
| `hardware_record` | função | `() -> dict[str, Any]` | Registro de CPU, núcleos, memória, GPU, versões e threads para acompanhar uma tabela de orçamento. |

#### `pinneapple_analysis.cost.profiler`

Mede custo de uma operação, ajusta lei de escala `cost ~ n^k`, checa orçamentos e detecta regressões contra uma linha de base salva.

<small>`pinneapple_analysis/cost/profiler.py` · 437 linhas · testes: 11</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CostRecord` | classe | `(name, n_runs, time_s, time_min_s, time_iqr_s, times_s, peak_mem_bytes, mem_source, flops…)` | Custo medido de uma operação: tempos, dispersão, memória de pico, FLOPs e tamanho. |
| `ScalingFit` | classe | `(quantity, exponent, ci, r2, coefficient, declared, expected_exponent, verdict)` | Ajuste log-log de uma grandeza contra o tamanho, com expoente, intervalo, R² e veredito contra a complexidade declarada. |
| `ScalingStudy` | classe | `(name, sizes, records, fits)` | Medidas de uma operação em vários tamanhos e seus ajustes de escala. |
| `Budget` | classe | `(max_time_s, max_mem_bytes, max_flops, max_time_exponent)` | Limites de tempo, memória, FLOPs e expoente de uma operação; limites None não são checados. |
| `BudgetExceeded` | classe | `()` | Exceção levantada quando uma operação passa do orçamento. |
| `Regression` | classe | `(name, quantity, baseline, current, ratio)` | Uma regressão de custo detectada contra a linha de base (grandeza, valores e razão). |
| `CostLedger` | classe | `()` | Registro de operações nomeadas com seus custos; salva uma linha de base e compara runs posteriores. |
| `measure` | função | `(fn, *args, repeats=5, warmup=1, name=None, flops=True, memory=True, size=None, **kwargs)…)` | Mede o custo de `fn(*args, **kwargs)` com repetições e aquecimento. |
| `expected_exponent` | função | `(declared, sizes) -> float` | Inclinação log-log da complexidade declarada nos tamanhos usados. |
| `scaling_study` | função | `(make_call, sizes, *, name='operation', repeats=5, warmup=1, declared=None, declared_flop…)` | Mede uma operação em vários tamanhos e ajusta `cost ~ n^k` para tempo e FLOPs. |

#### `pinneapple_analysis.cost.torch_ops`

Custo por etapa de um passo de treino PINN e tabela de custo por operador PyTorch via profiler. [sem teste]

<small>`pinneapple_analysis/cost/torch_ops.py` · 84 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `pinn_operation_costs` | função | `(model, x, *, repeats=5, warmup=1, optimizer=None) -> Dict[str, CostRecord]` | Custo de cada etapa de um passo PINN em pontos de colocação. |
| `profile_ops` | função | `(fn, *, top=15, sort_by='self_cpu_time_total') -> List[Dict[str, Any]]` | Tabela por operador PyTorch de uma chamada: chamadas, tempo próprio de CPU e FLOPs. |

#### `pinneapple_analysis.data_assimilation`

Assimilação de dados: 4D-Var por autograd sobre os métodos de ensemble (EKI) e filtros de Kalman, mais reconstrução GP e posicionamento de sensores.

<small>`pinneapple_analysis/data_assimilation/__init__.py` · 16 linhas · testes: 6</small>

#### `pinneapple_analysis.data_assimilation.models`

Modelos caóticos pequenos e diferenciáveis em PyTorch (Lorenz-96, Lorenz-63, passo RK4) para testar assimilação. [sem teste direto]

<small>`pinneapple_analysis/data_assimilation/models.py` · 39 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `rk4` | função | `(f, x, dt) -> torch.Tensor` | Um passo de Runge-Kutta de 4ª ordem de `f`. |
| `lorenz96_tendency` | função | `(x, F=8.0) -> torch.Tensor` | Tendência dx/dt do modelo Lorenz-96 com forçante F. |
| `lorenz96_step` | função | `(x, dt=0.05, F=8.0) -> torch.Tensor` | Um passo RK4 do Lorenz-96. |
| `lorenz63_step` | função | `(x, dt=0.01, sigma=10.0, rho=28.0, beta=8.0 / 3.0) -> torch.Tensor` | Um passo RK4 do Lorenz-63. |

#### `pinneapple_analysis.data_assimilation.sensor_placement`

Reconstrução de campo escalar por processo gaussiano (kernel Matérn) a partir de sensores pontuais e escolha gulosa dos próximos sensores.

<small>`pinneapple_analysis/data_assimilation/sensor_placement.py` · 236 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GaussianProcessField` | classe | `(lengthscales=1.0, variance=1.0, nu=2.5, noise=1e-10, mean=0.0) -> None` | Regressão por processo gaussiano de um campo escalar a partir de sensores pontuais, com LOO e ajuste de hiperparâmetros. |
| `SensorPlacement` | classe | `(indices, points, integrated_variance)` | Resultado do posicionamento guloso: índices, pontos e variância integrada. |
| `matern_kernel` | função | `(A, B, lengthscales=1.0, variance=1.0, nu=2.5) -> np.ndarray` | Covariância de Matérn anisotrópica entre dois conjuntos de pontos. |
| `greedy_sensor_placement` | função | `(gp, candidates, n_sensors, domain=None, weights=None, existing=None) -> SensorPlacement` | Escolhe sensores entre candidatos minimizando gulosamente a variância posterior integrada. |

#### `pinneapple_analysis.data_assimilation.var4d`

4D-Var de restrição forte sobre qualquer modelo PyTorch diferenciável, com o adjunto vindo do autograd, checagem de gradiente e experimento gêmeo em Lorenz-96. [sem teste direto]

<small>`pinneapple_analysis/data_assimilation/var4d.py` · 174 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Observation` | classe | `(step, y, H, R)` | Observações num passo do modelo, com operador H e covariância de erro R. |
| `Var4DResult` | classe | `(x0, trajectory, cost_initial, cost_final, cost_history, n_iterations, background_term, o…)` | Resultado do 4D-Var: estado inicial analisado, trajetória, histórico do custo e termos de fundo e observação. |
| `Var4D` | classe | `(model, B_sqrt, dtype=torch.float64)` | 4D-Var de restrição forte sobre um modelo `model(x) -> x_next` com covariância de fundo dada pela raiz. |
| `gradient_check` | função | `(var, x0, xb, obs, n_steps=None, eps=1e-06, seed=0) -> dict[str, float]` | Compara a derivada direcional do custo pelo autograd com uma diferença finita central. |
| `twin_experiment` | função | `(n=40, window_steps=20, obs_every=2, obs_fraction=0.5, obs_std=1.0, background_std=1.0, d…)` | Experimento gêmeo de 4D-Var no Lorenz-96 e RMSE de fundo, análise e previsão. |

#### `pinneapple_analysis.inspection`

Pacote de inspeção/NDE: reexporta `InspectionPINN`, `train_generic_inspection_pinn` e as funções `generate_*`, `make_*_physics_loss` e `train_*` de seis modalidades (correntes parasitas, MFL, acústica, EMAT, ondas guiadas, phased array). Todos os sinais são sintéticos e ilustrativos, nunca dados medidos.

<small>`pinneapple_analysis/inspection/__init__.py` · 151 linhas · testes: 13</small>

#### `pinneapple_analysis.inspection.acoustic`

Gera, por diferenças finitas no tempo (FDTD 1D), o espectro de condição de uma barra golpeada (impact-echo/ressonância), em que um defeito reduz a velocidade de onda local; inclui perda física da equação da onda e treino da PINN. [sem teste]

<small>`pinneapple_analysis/inspection/acoustic.py` · 227 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `generate_acoustic_synthetic` | função | `(*, L=1.0, c0=5000.0, n_points=60, duration=0.0015, defect_position=0.6, defect_width=0.0…)` | Simula por FDTD 1D a barra sem e com defeito e devolve séries temporais, espectros e os dados de treino da PINN para o deslocamento `u(x, t)`. |
| `make_acoustic_physics_loss` | função | `(x_grid, c_field) -> PhysicsLossFn` | Cria a perda física `(model, batch) -> (loss, comps)` da equação da onda `u_tt - c(x)^2 u_xx = 0` via autograd, com `c(x)` tirado do mapa FDTD fixo. |
| `train_acoustic` | função | `(synthetic_data=None, *, hidden=(64, 64, 64), epochs=200, lr=0.001, batch_size=32, device…)` | Gera os dados (se não vierem), monta uma `InspectionPINN` de entrada (x, t) e treina com a perda acústica; devolve modelo, saída do treino e dados. |

#### `pinneapple_analysis.inspection.base`

Base comum da inspeção: a MLP `InspectionPINN` (subclasse de `PINNBase`) e o wrapper de treino que usa o `Trainer` de `pinneapple_neural` com `CombinedLoss` (supervisionada + hook de física). [sem teste]

<small>`pinneapple_analysis/inspection/base.py` · 251 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `InspectionPINN` | classe | `(in_dim, out_dim, hidden=(64, 64, 64), activation='tanh', *, modality='generic', dtype=to…)` | MLP coordenadas -> sinal de inspeção, no padrão de `VanillaPINN`; a física entra por um resíduo externo (`make_*_physics_loss`). |
| `train_generic_inspection_pinn` | função | `(model, synthetic_data, physics_loss_fn=None, *, epochs=200, lr=0.001, batch_size=32, val…)` | Converte um dict de arrays em lotes `{x, y, x_col}`, liga a perda física opcional por `PhysicsLossHook` e chama `Trainer.fit`, devolvendo o dict do treino. |

#### `pinneapple_analysis.inspection.eddy_current`

Sinal sintético de varredura de sonda de correntes parasitas, construído sobre o solver axissimétrico de Helmholtz complexo de `pinneapple_simulation.numerical_solvers.eddy_current_fdm`, mais o treino da PINN com esse mesmo resíduo. A varredura lateral é uma aproximação axissimétrica de um problema 3D. [sem teste]

<small>`pinneapple_analysis/inspection/eddy_current.py` · 281 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `generate_eddy_current_synthetic` | função | `(scan_positions=None, *, nr=24, nz=40, r_bounds=(0.001, 0.05), z_bounds=(-0.03, 0.03), om…)` | Gera um conjunto sintético de varredura de sonda de correntes parasitas resolvendo o Helmholtz axissimétrico para cada posição de defeito. |
| `make_eddy_current_physics_loss` | função | `(omega, mu, sigma) -> PhysicsLossFn` | Cria a perda física com as partes real e imaginária do operador de Helmholtz complexo axissimétrico, com J=0 (válido no volume do condutor). |
| `train_eddy_current` | função | `(synthetic_data=None, *, hidden=(64, 64, 64), epochs=200, lr=0.001, batch_size=32, device…)` | Gera os dados se preciso, monta uma `InspectionPINN` e treina com a perda do resíduo de Helmholtz. |

#### `pinneapple_analysis.inspection.emat`

Gerador simplificado de sinal EMAT: a força de Lorentz vira um termo de força escalar (Gaussiana no espaço, tone-burst no tempo) numa equação da onda elástica 1D; não resolve o acoplamento eletromagnético-elástico completo. [sem teste]

<small>`pinneapple_analysis/inspection/emat.py` · 267 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `generate_emat_synthetic` | função | `(*, L=0.3, rho=7800.0, G0=80000000000.0, n_points=60, duration=6e-05, coil_position=0.05,…)` | Gera um sinal EMAT pitch-catch sintético: um tone-burst tipo Lorentz lança uma onda elástica que um defeito (queda local de rigidez) reflete/atenua. |
| `make_emat_physics_loss` | função | `(x_grid, c_field, coil_x0, footprint_sigma, burst_freq, n_cycles, t0_burst, J0_over_rho) …)` | Cria a perda física da equação da onda forçada `u_tt - c(x)^2 u_xx - F(x,t) = 0`, com `F` em forma fechada. |
| `train_emat` | função | `(synthetic_data=None, *, hidden=(64, 64, 64), epochs=200, lr=0.001, batch_size=32, device…)` | Gera os dados EMAT se preciso e treina uma `InspectionPINN` com a perda da onda forçada. |

#### `pinneapple_analysis.inspection.guided_wave_ultrasonic`

Ondas guiadas em 1D com a equação dispersiva tipo Klein-Gordon `u_tt = c0^2 u_xx - beta(x) u`, como substituto qualitativo das curvas de Rayleigh-Lamb; o defeito é um aumento local de `beta`. [sem teste]

<small>`pinneapple_analysis/inspection/guided_wave_ultrasonic.py` · 210 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `generate_guided_wave_synthetic` | função | `(*, L=1.0, c0=3000.0, beta0=0.0, n_points=80, duration=0.0006, burst_freq=50000.0, n_cycl…)` | Gera um conjunto pulso-eco de onda guiada: um tone-burst injetado em x=0 é parcialmente refletido por um aumento local de `beta(x)`. |
| `make_guided_wave_physics_loss` | função | `(x_grid, beta_field, c0) -> PhysicsLossFn` | Cria a perda física da equação dispersiva `u_tt - c0^2 u_xx + beta(x) u = 0`. |
| `train_guided_wave_ultrasonic` | função | `(synthetic_data=None, *, hidden=(64, 64, 64), epochs=200, lr=0.001, batch_size=32, device…)` | Gera os dados de onda guiada se preciso e treina uma `InspectionPINN` com a perda dispersiva. |

#### `pinneapple_analysis.inspection.magnetic_flux_leakage`

Verdade de referência por diferenças finitas 2D para MFL (vazamento de fluxo magnético) com permeabilidade não linear (saturação tipo Frohlich-Kennelly, iteração de Picard) e treino PIML com um resíduo simplificado. [sem teste]

<small>`pinneapple_analysis/inspection/magnetic_flux_leakage.py` · 308 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `solve_mfl_nonlinear` | função | `(nx, ny, Lx, Ly, phi0, mu_r_max, H_sat, defect_mask, n_picard=6) -> Tuple[np.ndarray, np.…)` | Resolve o potencial escalar magnetostático não linear do MFL por iteração de Picard, com a região de defeito fixa em `mu0`. |
| `generate_magnetic_flux_leakage_synthetic` | função | `(*, nx=36, ny=14, Lx=0.2, Ly=0.02, phi0=500.0, mu_r_max=500.0, H_sat=2000.0, defect_cente…)` | Resolve a verdade de referência MFL e lê o campo de vazamento `Bz` na superfície como sinal de varredura ao longo de x. |
| `make_mfl_physics_loss` | função | `(x_grid, y_grid, mu_field) -> PhysicsLossFn` | Cria uma perda física simplificada `mu * laplaciano(phi)`, que descarta o termo cruzado `grad(mu)·grad(phi)` (exata só onde `mu` varia pouco). |
| `train_magnetic_flux_leakage` | função | `(synthetic_data=None, *, hidden=(64, 64, 64), epochs=200, lr=0.001, batch_size=32, device…)` | Gera os dados MFL se preciso e treina uma `InspectionPINN` com a perda magnetostática simplificada. |

#### `pinneapple_analysis.inspection.phased_array_ultrasonic`

B-scan sintético de um array linear em pulso-eco (wavelets de Ricker) reconstruído por delay-and-sum / Total Focusing Method; a perda física de Helmholtz é declarada como regularizador, não lei exata. [sem teste]

<small>`pinneapple_analysis/inspection/phased_array_ultrasonic.py` · 205 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `generate_phased_array_synthetic` | função | `(*, n_elements=24, array_length=0.02, c=5900.0, burst_freq=5000000.0, defect_x=0.0, defec…)` | Gera um B-scan pulso-eco sintético de um array de elementos e sua reconstrução por delay-and-sum (TFM). |
| `make_phased_array_physics_loss` | função | `(k0) -> PhysicsLossFn` | Cria a perda de consistência de Helmholtz 2D `laplaciano(p) + k0^2 p = 0` aplicada à intensidade da imagem (um regularizador, não lei física exata). |
| `train_phased_array_ultrasonic` | função | `(synthetic_data=None, *, hidden=(64, 64, 64), epochs=200, lr=0.001, batch_size=32, device…)` | Gera o B-scan se preciso e treina uma `InspectionPINN` com o regularizador de Helmholtz. |

#### `pinneapple_analysis.inverse_problems`

Reexporta o kit de problemas inversos (modelos de ruído, regularizadores, operadores de observação, EKI, sensibilidade, termos faltantes, `InverseProblemSolver`, e o núcleo de identificabilidade/calibração/studio). O docstring ainda usa o nome antigo `pinneapple_inverse`.

<small>`pinneapple_analysis/inverse_problems/__init__.py` · 261 linhas · testes: 17</small>

#### `pinneapple_analysis.inverse_problems.calibration`

Calibração de modelo: separa medições de ajuste e de validação, faz mínimos quadrados ponderados, roda a identificabilidade no ponto calibrado (parâmetro não estimável fica sem intervalo) e dá intervalos por Jacobiano e por bootstrap. [sem teste direto]

<small>`pinneapple_analysis/inverse_problems/calibration.py` · 241 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CalibrationResult` | classe | `(parameters, fit, metrics, rows, identifiability, bootstrap, warnings, seconds)` | Resultado da calibração: parâmetros, ajuste, métricas, linhas, identificabilidade, bootstrap, avisos e tempo; `theta` dá os valores e `to_dict` serializa. |
| `residual_metrics` | função | `(pred, obs) -> dict[str, Any]` | Calcula RMSE, MAE, erro máximo, viés, chi2 e chi2 por ponto da predição contra as medições, no total e por observável. |
| `calibrate` | função | `(model, obs, parameters=None, *, holdout=None, holdout_fraction=0.0, holdout_by=None, los…)` | Calibra os parâmetros contra as observações com validação separada, perda robusta opcional, bootstrap e identificabilidade. |

#### `pinneapple_analysis.inverse_problems.contracts`

Converte relatórios de identificabilidade, calibrações e comparações de métodos em `StudyResult` de `pinneapple_data.contracts`, e parâmetros em/de `ParameterSpace`, para encadear os apps.

<small>`pinneapple_analysis/inverse_problems/contracts.py` · 246 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `parameters_from_space` | função | `(space, nominal=None) -> list[Parameter]` | Cria a lista de `Parameter` a partir de um `ParameterSpace`; parâmetros categóricos não são estimáveis e levantam erro. |
| `to_parameter_space` | função | `(params, name, description='', groups=None)` | Cria um `ParameterSpace`: parâmetros livres viram contínuos nos seus limites e os fixos viram constantes. |
| `identifiability_study` | função | `(report, params, *, app='identifiability', title='', model='', study_id=None)` | Converte um relatório de identificabilidade em `StudyResult` (uma linha por parâmetro, posto e condicionamento como métricas). |
| `calibration_study` | função | `(result, params, *, app='calibration', title='', model='', model_spec=None, study_id=None)` | Converte um resultado de calibração em `StudyResult`, com um `ModelRef` do modelo calibrado quando há métricas de validação. |
| `inverse_study` | função | `(comparison, params, *, app='inverse', title='', study_id=None)` | Converte a saída de `compare_methods` em `StudyResult` com uma linha por método e parâmetro. |

#### `pinneapple_analysis.inverse_problems.counterfactual_attribution`

Atribuição exata por valores de Shapley de uma variação medida de um escalar entre poucas causas candidatas, via rodadas contrafactuais de qualquer modelo; inclui um classificador de ambiguidade quando duas causas não se distinguem.

<small>`pinneapple_analysis/inverse_problems/counterfactual_attribution.py` · 152 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CandidateCause` | classe | `(name, apply_baseline, apply_current, signature_measurements)` | Causa candidata: nome, funções que a colocam no estado de base ou atual, e medições que a assinam. |
| `ShapleyAttributionResult` | classe | `(measured_delta, contributions, residual, baseline_value, current_value)` | Resultado da atribuição: variação medida, contribuição por causa, resíduo de fechamento (deve ser ~0) e valores de base e atual. |
| `ExactShapleyAttributor` | classe | `(value_fn, baseline_state)` | Avalia uma função de valor em todas as combinações de causas e devolve a decomposição exata de Shapley da variação. |
| `AmbiguityResult` | classe | `(indeterminate, reason, recommended_measurement)` | Diz se a atribuição é indeterminada, por quê, e qual medição é recomendada para desempatar. |
| `classify_ambiguity` | função | `(contributions, contribution_uncertainty, signature_measurements, available_measurements)…)` | Decide se as contribuições são indistinguíveis dadas as incertezas e as medições disponíveis. |

#### `pinneapple_analysis.inverse_problems.ensemble_kalman`

Ensemble Kalman Inversion (EKI) sem derivadas: atualiza um ensemble de parâmetros só com avaliações do modelo direto; inclui a versão regularizada por Tikhonov (TEKI).

<small>`pinneapple_analysis/inverse_problems/ensemble_kalman.py` · 454 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EKIConfig` | classe | `(n_ensemble, n_iterations, noise_cov, noise_std, init_spread, lambda_reg, prior_mean, pri…)` | Configuração da EKI: tamanho do ensemble, iterações, ruído, espalhamento inicial, regularização, prior, tolerância de colapso e semente. |
| `EKIHistory` | classe | `(iterations, data_misfit, ensemble_spread, theta_mean_history)` | Histórico de convergência da EKI (desajuste, espalhamento, média de theta por iteração). |
| `EnsembleKalmanInversion` | classe | `(forward_fn, config=None) -> None` | EKI padrão (Iglesias et al. 2013): inicializa o ensemble, aplica passos tipo Kalman e para por colapso do ensemble. |
| `IteratedEKI` | classe | `()` | EKI regularizada por Tikhonov (TEKI, Chada et al. 2020). |

#### `pinneapple_analysis.inverse_problems.estimation`

Estimadores comuns a identificabilidade, calibração e studio: mínimos quadrados não lineares em parâmetros escalados, L-BFGS-B com gradiente exato (adjunto ou autograd) e Metropolis adaptativo. [sem teste direto]

<small>`pinneapple_analysis/inverse_problems/estimation.py` · 284 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Prior` | classe | `(mean, std, L, weight)` | Prior Gaussiano nos parâmetros escalados mais penalidade opcional de suavidade `weight * \|L z\|^2`. |
| `FitResult` | classe | `(z, theta, pred, chi2, prior_penalty, jac, nfev, success, message, seconds, derivatives, …)` | Resultado de um ajuste: z, theta, predições, chi2, penalidade do prior, Jacobiano, avaliações e diagnóstico; `dof` dá os graus de liberdade. |
| `PosteriorSamples` | classe | `(names, samples_z, samples, acceptance, n_evaluations, seconds)` | Amostras da posterior (em z e unidades físicas) com taxa de aceitação; `summary` dá média, mediana, desvio, intervalos e correlações. |
| `least_squares_fit` | função | `(model, params, obs, *, z0=None, free=None, prior=None, loss='linear', max_nfev=200, deri…)` | Minimiza chi2 (mais prior) nos parâmetros escalados por região de confiança, com subconjunto livre e perdas robustas opcionais. |
| `gradient_fit` | função | `(model, params, obs, *, z0=None, prior=None, gradient='adjoint', max_iter=300) -> FitResu…)` | L-BFGS-B no desajuste com gradiente exato pelo adjunto do modelo ou por autograd. |
| `metropolis` | função | `(model, params, obs, *, z0, cov0=None, n_steps=4000, burn=None, prior=None, seed=0, thin=…)` | Metropolis de passeio aleatório adaptativo (Haario et al. 2001) em z, com verossimilhança Gaussiana e limite de tempo. |

#### `pinneapple_analysis.inverse_problems.forward`

Linguagem comum dos problemas inversos: `Parameter`, `ParameterSet`, `Observations`, a base `ForwardModel` (cujo `predict` levanta NotImplementedError até ser sobrescrito) e modelos por função, expressão de texto ou tabela; mais o Jacobiano em coordenadas escaladas. [sem teste direto]

<small>`pinneapple_analysis/inverse_problems/forward.py` · 484 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Parameter` | classe | `(name, value, lower, upper, unit, log, fixed, description, truth)` | Um parâmetro ajustável com valor nominal, limites, unidade, escala log opcional, flag de fixo e valor verdadeiro opcional. |
| `ParameterSet` | classe | `(parameters)` | Conjunto de parâmetros livres e o mapa entre theta (dict em unidades físicas) e z (vetor escalado). |
| `Observations` | classe | `(inputs, sigma, values, observable, meta)` | Medições: entradas por coluna, observável de cada linha, valores (opcionais antes do experimento) e sigma do ruído; permite subconjunto, divisão e concatenação. |
| `ForwardModel` | classe | `(name, inputs, outputs, cost, description)` | Classe base de modelo direto: define entradas, saídas e parâmetros padrão; `predict` levanta NotImplementedError se não for sobrescrito. |
| `CallableModel` | classe | `(fn, *, inputs, outputs=('y',), parameters=(), name='callable', cost='cheap', description…)` | Embrulha qualquer função Python ou solver externo `fn(theta, inputs)` como modelo direto. |
| `ExpressionModel` | classe | `(expressions, *, inputs, parameters, name='expression', description='')` | Modelo escrito como texto, uma expressão por observável. |
| `TabulatedModel` | classe | `(table, *, parameters, inputs, output, name='simulated results', smoothing=0.0, observabl…)` | Usa uma tabela de resultados simulados como modelo, interpolando sobre parâmetros e entradas. |
| `count_evaluations` | função | `(model) -> int` | Devolve quantas vezes o modelo foi avaliado (contador interno `_n_eval`). |
| `jacobian` | função | `(model, params, z, obs, *, method='auto', rel_step=0.0001) -> tuple[np.ndarray, np.ndarra…)` | Calcula predições e Jacobiano em relação a z, pelo método do modelo, autograd ou diferenças finitas. |

#### `pinneapple_analysis.inverse_problems.heat_diffusivity`

Recupera a difusividade `alpha` da equação do calor 1D a partir de leituras ruidosas num instante, com uma PINN que aprende campo e `alpha` juntos, e relata a informação de Fisher/limite de Cramér-Rao.

<small>`pinneapple_analysis/inverse_problems/heat_diffusivity.py` · 162 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `u_exact` | função | `(t, x, alpha) -> np.ndarray` | Solução exata `exp(-pi^2 alpha t) sin(pi x)` da equação do calor do caso de teste. |
| `identifiability_report` | função | `(t_obs, x_obs, alpha, noise_std) -> dict[str, Any]` | Informação de Fisher e limite de Cramér-Rao para `alpha` a partir das leituras com ruído Gaussiano. |
| `run_inverse_discovery` | função | `(true_alpha=TRUE_ALPHA, initial_guess=INITIAL_GUESS, noise_std=NOISE_STD, n_observations=…)` | Gera as leituras, treina rede e `alpha` juntos e devolve um resultado serializável com histórico, `alpha` recuperado e identificabilidade. |

#### `pinneapple_analysis.inverse_problems.heat_transfer`

Transferência de calor inversa: estima o coeficiente de convecção `h` de uma aleta (1D, também com leituras do usuário) e de uma placa 2D por PINN; o caso 3D não é treinado aqui, é servido a partir de resultados gravados em `examples/use_cases/fin_convection_inverse/results/3d`.

<small>`pinneapple_analysis/inverse_problems/heat_transfer.py` · 325 linhas · testes: 15</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `InputError` | classe | `()` | Erro de entrada inválida (subclasse de `ValueError`) dos casos de transferência de calor. |
| `fin_exact` | função | `(xi, h, k, d, length) -> np.ndarray` | Temperatura adimensional de uma aleta pino com ponta convectiva (Incropera, Tabela 3.4, caso A). |
| `fin_q_exact` | função | `(h, k, d, length, dt) -> float` | Calor que sai da base da aleta pela solução analítica. |
| `run_fin` | função | `(*, k, d_mm, length_mm, t_base, t_air, sensors_mm, readings, h_guess=100.0, steps=2000, s…)` | Aprende o perfil de temperatura e `h` de uma aleta a partir de leituras de termopares. |
| `fin_synthetic_readings` | função | `(*, k, d_mm, length_mm, t_base, t_air, sensors_mm, h_true, noise, seed=0) -> list[float]` | Gera leituras sintéticas de termopares da aleta analítica mais ruído Gaussiano. |
| `run_plate` | função | `(*, power, h_true, noise, h_guess=60.0, steps=1500, seed=0, progress=None, budget=None, w…)` | Gera leituras sintéticas da placa pelo modelo de volumes finitos e treina a PINN para T(x, y) e `h`, comparando com o modelo. |
| `plate_precomputed` | função | `() -> dict` | Carrega o resultado gravado da placa 2D (campo da PINN e de referência) da pasta de resultados do exemplo. |
| `block_precomputed` | função | `() -> dict` | Carrega o resultado gravado do bloco 3D (campo da PINN, referência e sorteios de `h`) da pasta de resultados do exemplo. |

#### `pinneapple_analysis.inverse_problems.identifiability`

Responde se os parâmetros podem ser recuperados das medições antes de estimar: informação de Fisher, sensibilidade local, verossimilhança de perfil e posterior, mais projeto D-ótimo de novas medições. (A flag 'experimental' vem de 'experimental design', não de status.) [sem teste direto]

<small>`pinneapple_analysis/inverse_problems/identifiability.py` · 457 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): experimental</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `IdentifiabilityReport` | classe | `(parameters, analysis_point, data, n_observations, rank, n_parameters, condition_number, …)` | Relatório de identificabilidade: posto, condicionamento, autovalores, correlações, ambiguidades e status por parâmetro. |
| `d_optimal_design` | função | `(F, Jc_w, n_pick=3, distinct=True) -> list[dict[str, Any]]` | Seleção gulosa D-ótima de medições candidatas a partir da informação de Fisher, com o ganho de log det de cada escolha. |
| `profile_likelihood` | função | `(model, params, obs, z_best, chi2_min, std_z, *, n_points=15, max_nfev=60, time_budget=60…)` | Perfil de verossimilhança de cada parâmetro livre, com intervalo de 95% e veredito. |
| `analyze_identifiability` | função | `(model, obs, parameters=None, *, candidates=None, n_suggest=3, profile='auto', n_profile=…)` | Roda os métodos de identificabilidade sobre modelo e medições e sugere novas medições. |

#### `pinneapple_analysis.inverse_problems.missing_term`

Identificação de termos faltantes numa EDP parcialmente conhecida: biblioteca de candidatos, regressão esparsa SINDy (STRidge/LASSO), análise do resíduo de uma PINN treinada e um termo neural treinado em conjunto.

<small>`pinneapple_analysis/inverse_problems/missing_term.py` · 500 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CandidateLibrary` | classe | `(poly_order=2, include_trig=False, custom_terms=None) -> None` | Monta a matriz Θ de termos candidatos (monômios, trigonométricos, personalizados) a partir de dados de colocação. |
| `SINDyResult` | classe | `(coefficients, active_terms, active_coefficients, residual_norm, n_terms_selected)` | Resultado da regressão SINDy: coeficientes, termos ativos e norma do resíduo; `equation` escreve a forma simbólica. |
| `SINDyIdentifier` | classe | `(threshold=0.001, max_iter=20, alpha=1e-05, method='stridge') -> None` | Identifica termos faltantes por regressão esparsa (Brunton et al. 2016). |
| `ResidualAnalysisResult` | classe | `(residual_field, sindy_result, coords, mean_abs_residual, max_abs_residual)` | Saída de `ResidualAnalyzer.analyze`: campo de resíduo, resultado SINDy opcional e estatísticas. |
| `ResidualAnalyzer` | classe | `(pde_residual_fn, library=None, identifier=None) -> None` | Avalia o resíduo da EDP conhecida numa PINN treinada e opcionalmente roda SINDy sobre ele. |
| `NeuralTermConfig` | classe | `(hidden_dims, activation, term_weight, data_weight, n_iters, lr, device)` | Configuração do treino conjunto PINN + termo neural. |
| `NeuralTermDiscovery` | classe | `(known_residual_fn, config=None) -> None` | Descobre um termo faltante parametrizado por uma rede neural e permite destilá-lo. |

#### `pinneapple_analysis.inverse_problems.models`

Modelos diretos físicos para identificabilidade, calibração e inversão: aleta analítica, aleta em volumes finitos com zonas de `h` (com adjunto), placa dissipadora, resfriamento concentrado e cinética de Arrhenius. [sem teste direto]

<small>`pinneapple_analysis/inverse_problems/models.py` · 476 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): experimental</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PinFinModel` | classe | `(d_mm=5.0, length_mm=50.0)` | Aleta pino analítica com ponta convectiva (caso do Inverse Heat Lab), com autograd e caminho PINN para `h`. |
| `RodModel` | classe | `(d_mm=5.0, length_mm=100.0, n_zones=1, n_cells=200)` | Aleta em volumes finitos com `h` por zonas (um campo), com sensibilidades exatas, gradiente adjunto e gêmeo em torch. |
| `PlateModel` | classe | `(nx=60, ny=36)` | Placa dissipadora do Inverse Heat Lab resolvida por volumes finitos, com cache por `h`. |
| `LumpedCoolingModel` | classe | `(area_m2=0.01)` | Curva de resfriamento concentrado; `h` e C só aparecem como `h A / C`, logo os dois livres são estruturalmente ambíguos. |
| `ArrheniusModel` | classe | `()` | Conversão de primeira ordem de Arrhenius; A e Ea são ambíguos com uma só temperatura. |

#### `pinneapple_analysis.inverse_problems.noise_models`

Modelos de ruído / termos de desajuste de dados em PyTorch: Gaussiano, Huber, Cauchy, Student-t e heteroscedástico.

<small>`pinneapple_analysis/inverse_problems/noise_models.py` · 260 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DataMisfitBase` | classe | `()` | Base abstrata de termo de desajuste / log-verossimilhança negativa. |
| `GaussianMisfit` | classe | `(noise_cov=None, noise_std=1.0, reduction='mean') -> None` | Desajuste Gaussiano (L2/Mahalanobis). |
| `HuberMisfit` | classe | `(delta=1.0, noise_std=1.0, smooth=True) -> None` | Desajuste de Huber (ou pseudo-Huber), robusto a outliers moderados. |
| `CauchyMisfit` | classe | `(gamma=1.0, noise_std=1.0) -> None` | Desajuste de Cauchy/Lorentz, de cauda pesada e muito robusto. |
| `StudentTMisfit` | classe | `(nu=5.0, noise_std=1.0) -> None` | Log-verossimilhança negativa de Student-t; `nu` controla a robustez. |
| `HeteroscedasticMisfit` | classe | `(log_noise_var) -> None` | Desajuste Gaussiano com variância de ruído por observação. |

#### `pinneapple_analysis.inverse_problems.obs_operator`

Operadores de observação H (estado -> observáveis): pontual, linear, integral e composto. O docstring cita um `BoundaryObsOperator` que não existe no arquivo.

<small>`pinneapple_analysis/inverse_problems/obs_operator.py` · 284 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ObsOperatorBase` | classe | `()` | Base abstrata de operador de observação; oferece `jacobian_theta`. |
| `PointObsOperator` | classe | `(sensor_locations, output_indices=None) -> None` | Avalia o modelo num conjunto fixo de posições de sensores. |
| `LinearObsOperator` | classe | `(A, x_eval) -> None` | Operador linear `H(u) = A u` (ex.: tomografia). |
| `IntegralObsOperator` | classe | `(x_quad, weights, kernel_fn=None) -> None` | Funcional integral aproximado por quadratura ponderada com núcleo opcional. |
| `ComposedObsOperator` | classe | `(operators, x_obs_list=None) -> None` | Empilha vários operadores de observação num vetor único. |

#### `pinneapple_analysis.inverse_problems.presets`

Problemas inversos prontos com resposta conhecida (medições sintéticas com semente fixa) e construção de modelos a partir de JSON, para apps e agentes. [sem teste direto]

<small>`pinneapple_analysis/inverse_problems/presets.py` · 326 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Preset` | classe | `(id, title, summary, model_spec, parameters, observations, validation, candidates, method…)` | Problema pronto: modelo em JSON, parâmetros, observações, validação, métodos e lição; `build` monta os objetos. |
| `model_from_spec` | função | `(spec) -> ForwardModel` | Constrói um `ForwardModel` a partir de um dict JSON (aleta, expressão, tabela etc.). |
| `synthetic` | função | `(model, params, obs, seed, rel_sigma=False) -> Observations` | Gera observações sintéticas: modelo na verdade mais ruído N(0, sigma) com semente, arredondado à resolução do instrumento. |
| `get_preset` | função | `(name) -> Preset` | Devolve um preset pelo nome. |
| `list_presets` | função | `(tag=None) -> list[dict[str, Any]]` | Lista os presets, opcionalmente filtrados por tag. |

#### `pinneapple_analysis.inverse_problems.regularization`

Regularizadores diferenciáveis em PyTorch (Tikhonov, L1, variação total, composto) e escolha automática de lambda pela curva L.

<small>`pinneapple_analysis/inverse_problems/regularization.py` · 329 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RegularizerBase` | classe | `()` | Base abstrata de regularizador R(θ) com peso. |
| `TikhonovRegularizer` | classe | `(lambda_reg, prior=None, covariance=None) -> None` | Regularizador de Tikhonov/ridge, com prior e covariância opcionais. |
| `SparsityRegularizer` | classe | `(lambda_reg, eps=1e-06) -> None` | Regularizador L1/LASSO com aproximação suave. |
| `TotalVariationRegularizer` | classe | `(lambda_reg, eps=1e-06) -> None` | Variação total anisotrópica para parâmetros de campo 1D/2D. |
| `CompositeRegularizer` | classe | `(terms=None) -> None` | Soma ponderada de vários regularizadores. |
| `LCurveResult` | classe | `(lambdas, data_misfits, regularization_norms, optimal_lambda, curvatures)` | Resultado da curva L: lambdas, desajustes, normas, curvaturas e lambda ótimo. |
| `LCurveSelector` | classe | `(train_fn, lambdas=None) -> None` | Escolhe lambda pela máxima curvatura da curva L, chamando uma função de treino para cada valor. |

#### `pinneapple_analysis.inverse_problems.sensitivity`

Análise de sensibilidade: Jacobiano local e matriz de Fisher, análise de identificabilidade por autovalores com critérios D/A/E e índices de Sobol globais pelo método de Saltelli.

<small>`pinneapple_analysis/inverse_problems/sensitivity.py` · 439 linhas · testes: 151 · palavras-chave no código (heurística; o texto acima diz o status real): experimental</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LocalSensitivityResult` | classe | `(jacobian, fisher_information, singular_values, sensitivity_indices, param_names)` | Resultado da sensibilidade local: Jacobiano, Fisher, valores singulares e índices. |
| `LocalSensitivity` | classe | `(forward_fn, noise_std=1.0, param_names=None) -> None` | Sensibilidade linearizada em torno de um ponto nominal, com Jacobiano por diferenças finitas opcional. |
| `IdentifiabilityResult` | classe | `(eigenvalues, eigenvectors, condition_number, d_optimality, pseudo_d_optimality, a_optima…)` | Resultado da análise de identificabilidade por Fisher: autovalores, condicionamento, critérios D/A/E e índices identificáveis. |
| `IdentifiabilityAnalyzer` | classe | `(tol=1e-06) -> None` | Analisa a identificabilidade a partir dos autovalores da matriz de Fisher. |
| `SobolResult` | classe | `(S1, ST, param_names, n_samples)` | Índices de Sobol de primeira ordem e totais. |
| `GlobalSensitivity` | classe | `(forward_fn, param_bounds, n_samples=1024, seed=None) -> None` | Sensibilidade global por variância (índices de Sobol pelo método de Saltelli). |

#### `pinneapple_analysis.inverse_problems.solver`

`InverseProblemSolver`: liga modelo (PINN ou `nn.Module`), operador de observação, desajuste e regularização, otimizando por Adam, L-BFGS ou EKI.

<small>`pinneapple_analysis/inverse_problems/solver.py` · 426 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `InverseSolverConfig` | classe | `(method, n_iters, lr, lbfgs_max_iter, device, grad_clip, print_every, eki_n_ensemble, eki…)` | Configuração do `InverseProblemSolver`: método, iterações, taxa de aprendizado e opções de EKI. |
| `InverseSolverResult` | classe | `(theta_opt, final_misfit, final_regularization, final_total_loss, loss_history, n_iters, …)` | Resultado: theta ótimo, desajuste, regularização, perda total, histórico, método e convergência. |
| `InverseProblemSolver` | classe | `(model, obs_operator, data_misfit=None, regularizer=None, config=None, extra_loss_fn=None…)` | Resolvedor unificado de problemas inversos com informação física. |

#### `pinneapple_analysis.inverse_problems.studio`

Interface única para formular um problema inverso, resolvê-lo com vários métodos (`least_squares`, `adjoint`, `autodiff`, `bayesian`, `pinn`) e ranqueá-los pelo erro nas medições de validação. [sem teste direto]

<small>`pinneapple_analysis/inverse_problems/studio.py` · 317 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `InverseProblem` | classe | `(model, observations, parameters, validation, name, description, prior_std, smoothness, f…)` | Problema inverso: modelo, observações, parâmetros, validação, prior e zonas de campo. |
| `available_methods` | função | `(problem) -> dict[str, dict[str, Any]]` | Diz quais métodos se aplicam ao problema e por que não, quando não se aplicam. |
| `solve_inverse` | função | `(problem, method, *, identifiability=None, pinn_steps=1500, mcmc_steps=3000, seed=0, prog…)` | Resolve o problema com um método e devolve um dict serializável com estimativas, incerteza, resíduos e métricas. |
| `compare_methods` | função | `(problem, methods=…, *, identifiability=True, pinn_steps=1500, seed=0, progress=None) -> …)` | Roda vários métodos sobre os mesmos dados e os ranqueia pelo RMSE de validação. |

#### `pinneapple_analysis.proof`

Entrada dos grafos de prova; reexporta `ProofGraph` e as fábricas de checagem.

<small>`pinneapple_analysis/proof/__init__.py` · 29 linhas · testes: 11</small>

#### `pinneapple_analysis.proof.graph`

Modelo de grafo de prova com nós (hipótese, definição, lema, teorema, conjectura), checagens exatas ou numéricas e status ASSUMED/PROVED/ARGUED/SUPPORTED/REFUTED/OPEN. [sem teste direto]

<small>`pinneapple_analysis/proof/graph.py` · 334 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CheckOutcome` | classe | `(name, passed, exact, detail, counterexample, seconds)` | Resultado de uma checagem: passou, exata ou não, detalhe, contraexemplo e tempo. |
| `Check` | classe | `(name, fn)` | Uma checagem nomeada com a função que a executa. |
| `Node` | classe | `(id, kind, statement, depends_on, checks, argument, covered, supported_by, status, outcom…)` | Um nó do grafo (hipótese, definição, lema, teorema, conjectura) com dependências, checagens e status. |
| `ProofGraph` | classe | `(title='')` | O grafo de prova: adiciona nós, executa em ordem de dependência, escreve relatório e desenha em Mermaid ou matplotlib. |
| `symbolic_identity` | função | `(name, lhs, rhs=0, simplify=None) -> Check` | Checagem exata: `simplify(lhs - rhs) == 0` no sympy. |
| `symbolic_true` | função | `(name, statement) -> Check` | Checagem exata: uma relação sympy que simplifica para True. |
| `exhaustive` | função | `(name, predicate, domain, describe='') -> Check` | Checagem exata: um predicado sobre todos os elementos de um conjunto finito. |
| `numeric` | função | `(name, predicate, sampler, n=2000, seed=0) -> Check` | Checagem numérica: um predicado em amostras aleatórias (evidência; falha é contraexemplo). |
| `counterexample_search` | função | `(name, violation, bounds, tol=1e-08, seed=0, maxiter=300, restarts=3) -> Check` | Busca de contraexemplo por evolução diferencial; valor abaixo de `-tol` refuta. |
| `custom` | função | `(name, fn) -> Check` | Checagem livre que devolve (passou, exata, detalhe). |

#### `pinneapple_analysis.proof.pt_chains`

Grafo de prova que testa se o teorema de impossibilidade de sintonia linear de ponto excepcional vale para tetrâmeros PT; o teorema fica ARGUED e cadeias N >= 5 seguem como conjectura. [experimental]

<small>`pinneapple_analysis/proof/pt_chains.py` · 262 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `chain_hamiltonian` | função | `(gamma, bonds) -> np.ndarray` | Hamiltoniano de uma cadeia PT de N sítios com ganho e perda nas pontas. |
| `threshold_numeric` | função | `(bonds, gmax=None, tol=1e-11) -> float` | Menor gamma em que o espectro deixa de ser real, por varredura e bissecção na matriz. |
| `tetramer_threshold_branches` | função | `(A, B) -> list[float]` | Valores candidatos de gamma² dos ramos I e II do limiar do tetrâmero. |
| `build_pt_chain_graph` | função | `(max_deg=4, n_numeric=300, search_iters=120, searches=True) -> ProofGraph` | Monta o grafo de prova do tetrâmero PT; `searches=False` omite as buscas de contraexemplo lentas. |

#### `pinneapple_analysis.reproduction`

Avaliador único de métricas de erro de artigos, estatística por semente sem escolha da melhor e regra de status (reproduzido, parcial, não reproduzido, escala reduzida).

<small>`pinneapple_analysis/reproduction.py` · 139 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SeedStats` | classe | `(values, median, min, max, n_failed)` | Valor de cada semente com mediana e mín-máx; sem escolha da melhor semente. |
| `rel_l2` | função | `(pred, ref) -> float` | Erro L2 relativo; `inf` quando a predição não é finita. |
| `rmae` | função | `(pred, ref) -> float` | Erro absoluto relativo `sum \|pred - ref\| / sum \|ref\|`. |
| `max_abs` | função | `(pred, ref) -> float` | Maior erro absoluto. |
| `mean_abs` | função | `(pred, ref) -> float` | Erro absoluto médio. |
| `mse` | função | `(pred, ref) -> float` | Erro quadrático médio. |
| `rmse` | função | `(pred, ref) -> float` | Raiz do erro quadrático médio. |
| `evaluate` | função | `(pred, ref, metrics=('rel_l2', 'rmae', 'max_abs')) -> dict[str, float]` | Calcula as métricas nomeadas de `pred` contra `ref`. |
| `seed_stats` | função | `(values) -> SeedStats` | Mediana e mín-máx entre sementes; semente divergente conta como `inf`. |
| `status` | função | `(ours_median, paper, *, spread=None, ordering_holds=None, reduced_scale=False, factor=2.0…)` | Status de um alvo pela regra do protocolo: reproduced, partially reproduced, not reproduced ou reduced-scale. |

#### `pinneapple_analysis.retrieval`

Busca semântica local; oferece `build_default_index` e exige o extra `retrieval` (sentence-transformers).

<small>`pinneapple_analysis/retrieval/__init__.py` · 58 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_default_index` | função | `(*, model_name=DEFAULT_MODEL_NAME, include_literature=False, literature_topics=None, lite…)` | Monta numa chamada um índice sobre o corpus padrão, opcionalmente com artigos do arXiv. |

#### `pinneapple_analysis.retrieval.corpus`

Fontes reais do corpus: docstrings de módulos, resumos de presets, arquivos de código fatiados e artigos do arXiv buscados ao vivo (opcional).

<small>`pinneapple_analysis/retrieval/corpus.py` · 360 linhas · testes: 32</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Document` | classe | `(doc_id, source, title, text)` | Um documento do corpus: id, fonte, título e texto. |
| `fetch_literature_documents` | função | `(query, *, k=5, timeout_s=30) -> List[Document]` | Busca artigos reais no arXiv para uma consulta; falha de rede levanta erro. |
| `build_literature_corpus` | função | `(topics=None, *, k_per_topic=5, timeout_s=30) -> List[Document]` | Busca artigos para vários tópicos, sem duplicatas; pula com aviso um tópico que falhar. |
| `build_corpus` | função | `(*, include_literature=False, literature_topics=None, literature_k_per_topic=5) -> List[D…)` | Monta o corpus atual (docstrings, presets e, se pedido, literatura), recalculado a cada chamada. |
| `iter_repo_files` | função | `(repo_root, *, extensions=DEFAULT_CODE_EXTENSIONS, exclude_dirs=DEFAULT_EXCLUDE_DIRS, max…)` | Percorre um repositório e lista arquivos pelas extensões, pulando pastas excluídas e arquivos grandes. |
| `chunk_text` | função | `(text, *, max_chars=4000, overlap=200) -> List[str]` | Fatia texto em janelas de caracteres com sobreposição, preferindo cortar em linha em branco. |
| `file_fingerprint` | função | `(path) -> str` | Hash sha256 do conteúdo atual de um arquivo, para detectar mudanças. |
| `code_repo_documents` | função | `(repo_root, *, repo_name=None, extensions=DEFAULT_CODE_EXTENSIONS, exclude_dirs=DEFAULT_E…)` | Gera documentos de código-fonte fatiados de um clone local de repositório. |

#### `pinneapple_analysis.retrieval.index`

Índice semântico local com embeddings de sentence-transformers e similaridade de cosseno; salva e carrega o índice.

<small>`pinneapple_analysis/retrieval/index.py` · 133 linhas · testes: 17</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SearchResult` | classe | `(document, score)` | Um resultado de busca: documento e score de similaridade. |
| `RetrievalIndex` | classe | `(documents, embeddings, model_name) -> None` | Índice consultável com o embedding de todos os documentos; `build`, `search`, `save` e `load`. |

#### `pinneapple_analysis.sensitivity`

Análise de sensibilidade global e local; reexporta Sobol, Morris, curvas de efeito, o estudo completo e as funções de teste.

<small>`pinneapple_analysis/sensitivity/__init__.py` · 42 linhas · testes: 24</small>

#### `pinneapple_analysis.sensitivity.effects`

Curvas de dependência parcial e ICE, jacobiano local, recomendação de congelar parâmetros e verificação direta do congelamento. [sem teste direto]

<small>`pinneapple_analysis/sensitivity/effects.py` · 157 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `effect_curves` | função | `(fn, space, n_grid=21, n_background=128, n_ice=20, seed=0, outputs=None) -> dict` | Curvas de dependência parcial e ICE de cada entrada. |
| `local_derivatives` | função | `(fn, x0, space=None, torch_fn=None, rel_step=0.0001, outputs=None) -> dict` | Jacobiano dy/dx num ponto, por autograd quando há função torch, senão por diferenças finitas. |
| `freeze_recommendation` | função | `(indices, threshold=0.01, morris_result=None) -> list[dict]` | Por entrada: freeze, inconclusive ou keep, pelo intervalo do índice total contra um limiar. |
| `verify_freeze` | função | `(fn, space, frozen, n=1024, seed=0, at=None) -> dict` | Compara o modelo com todas as entradas variando e com as congeladas fixas; devolve razão de variância e mudanças pontuais. |

#### `pinneapple_analysis.sensitivity.morris`

Triagem de Morris por efeitos elementares, com custo `r (d + 1)` avaliações e intervalos bootstrap de mu*.

<small>`pinneapple_analysis/sensitivity/morris.py` · 111 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MorrisResult` | classe | `(names, outputs, r, levels, delta, n_evals, mu, mu_star, sigma, mu_star_ci, level, effect…)` | Resultado de Morris: mu, mu*, sigma, intervalos de mu* e ranking por saída. |
| `morris_design` | função | `(d, r=20, levels=4, seed=0) -> tuple[np.ndarray, np.ndarray, np.ndarray]` | Trajetórias de Morris no hipercubo unitário, a entrada movida em cada passo e o sinal. |
| `morris` | função | `(fn, space, r=20, levels=4, seed=0, n_boot=500, level=0.95, outputs=None) -> MorrisResult` | Triagem de Morris de `fn` sobre o espaço com `r` trajetórias. |

#### `pinneapple_analysis.sensitivity.sobol`

Índices de Sobol de primeira ordem, totais e de segunda ordem com intervalos bootstrap, a partir de um modelo, de avaliações externas ou de uma amostra existente. [sem teste direto]

<small>`pinneapple_analysis/sensitivity/sobol.py` · 255 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SobolIndices` | classe | `(names, outputs, n_base, n_evals, variance, S1, ST, S1_ci, ST_ci, S2, S2_ci, level, n_boo…)` | Índices de Sobol S1, ST (e S2) por saída e entrada, com intervalos bootstrap. |
| `saltelli_design` | função | `(space, n, second_order=False, seed=0) -> tuple[np.ndarray, dict]` | Entradas do desenho de Saltelli empilhadas e o layout para lê-las de volta. |
| `sobol_from_evaluations` | função | `(Y, layout, outputs=None, n_boot=200, level=0.95, seed=0, X=None) -> SobolIndices` | Índices a partir de saídas avaliadas em outro lugar sobre o desenho de Saltelli. |
| `sobol_indices` | função | `(fn, space, n=1024, second_order=False, n_boot=200, level=0.95, seed=0, outputs=None) -> …)` | Índices de Sobol de primeira ordem e totais (e de segunda ordem, opcional) de `fn`. |
| `first_order_given_data` | função | `(X, y, names=None, n_bins=None, n_boot=200, level=0.95, seed=0) -> dict` | Índices de primeira ordem a partir de uma amostra existente, por classes de mesma contagem com correção de viés. |
| `sobol_from_runs` | função | `(space, n, run_ids, Y, second_order=False, seed=0, n_boot=200, level=0.95, outputs=None) …)` | Índices a partir de runs do desenho feitos fora (OpenFOAM, CalculiX); run faltante levanta ValueError. |

#### `pinneapple_analysis.sensitivity.study`

Estudo de sensibilidade completo (Morris, Sobol, curvas, derivadas, regra de congelamento e sua checagem) e a previsão do número de avaliações. [sem teste direto]

<small>`pinneapple_analysis/sensitivity/study.py` · 136 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `study_cost` | função | `(d, n_sobol=1024, second_order=False, r_morris=20, n_grid=15, n_background=64, n_verify=5…)` | Número de avaliações do modelo em cada parte do estudo de sensibilidade. |
| `sensitivity_study` | função | `(model, space, outputs=None, *, n_sobol=1024, second_order=False, r_morris=20, n_grid=15,…)` | Roda o estudo de sensibilidade completo sobre um modelo e um espaço de parâmetros. |

#### `pinneapple_analysis.sensitivity.testfunctions`

Funções de teste com índices de Sobol em forma fechada (Ishigami, função G de Sobol) para conferir estimadores.

<small>`pinneapple_analysis/sensitivity/testfunctions.py` · 53 linhas · testes: 28</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ishigami` | função | `(X, a=7.0, b=0.1) -> np.ndarray` | Função de Ishigami. |
| `ishigami_space` | função | `() -> list[dict]` | Espaço de parâmetros da função de Ishigami. |
| `ishigami_indices` | função | `(a=7.0, b=0.1) -> dict` | Variância e índices de Sobol exatos da função de Ishigami. |
| `sobol_g` | função | `(X, a) -> np.ndarray` | Função G de Sobol com coeficientes `a`. |
| `sobol_g_space` | função | `(d) -> list[dict]` | Espaço de parâmetros da função G com `d` entradas. |
| `sobol_g_indices` | função | `(a) -> dict` | Variância e índices exatos da função G. |

#### `pinneapple_analysis.state_estimation`

Blocos de estimação sequencial de estado; código movido de `pinneapple_systems.digital_twin`, que agora mantém um shim de reexportação.

<small>`pinneapple_analysis/state_estimation/__init__.py` · 49 linhas · testes: 22 · palavras-chave no código (heurística; o texto acima diz o status real): shim ou obsoleto</small>

#### `pinneapple_analysis.state_estimation.kalman`

Filtro de Kalman estendido e filtro de Kalman de ensemble estocástico, mais a análise EnKF com localização de Gaspari-Cohn.

<small>`pinneapple_analysis/state_estimation/kalman.py` · 346 linhas · testes: 6 · palavras-chave no código (heurística; o texto acima diz o status real): shim ou obsoleto</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ExtendedKalmanFilter` | classe | `(n_state, n_obs, f, h, Q=None, R=None, F_jac=None, H_jac=None, *, eps_fd=1e-05) -> None` | Filtro de Kalman estendido com jacobianos dados ou por diferenças finitas. |
| `EnsembleKalmanFilter` | classe | `(n_state, n_obs, f, h, Q=None, R=None, n_ens=100, inflation=1.02, seed=0) -> None` | Filtro de Kalman de ensemble estocástico com inflação. |
| `gaspari_cohn` | função | `(d) -> np.ndarray` | Correlação de suporte compacto de Gaspari-Cohn para localização. |
| `enkf_analysis` | função | `(ens, y, hx, r_var, *, loc_xy=None, loc_yy=None, inflation=1.0, rng=None) -> np.ndarray` | Passo de análise EnKF com observações perturbadas, localização opcional e inflação. |

#### `pinneapple_analysis.study_contracts`

Ponte entre estudos de incerteza/sensibilidade e os contratos compartilhados (`ParameterSpace`, `ModelRef`, `StudyResult`).

<small>`pinneapple_analysis/study_contracts.py` · 237 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `is_parameter_space` | função | `(obj) -> bool` | Diz se um objeto é um contrato `ParameterSpace`. |
| `space_from_contract` | função | `(space) -> tuple[list[dict], dict[str, Any]]` | Lê `(parâmetros, fixos)` de um `ParameterSpace`. |
| `space_to_contract` | função | `(params, name='study', fixed=None, units=None)` | Grava os parâmetros de um estudo como `ParameterSpace`, com as distribuições nos metadados. |
| `model_from_ref` | função | `(ref, onnx=None, outputs=None) -> StudyModel` | Cria um `StudyModel` de uma referência `zoo:<nome>` ou ONNX. |
| `model_to_ref` | função | `(model)` | Cria um `ModelRef` de um modelo de estudo, com faixa de validade e métricas do cartão. |
| `uncertainty_result` | função | `(out, model, study_id, app='uncertainty', title='')` | Converte o resultado de `uncertainty_study` num `StudyResult` de incerteza. |
| `sensitivity_result` | função | `(out, model, study_id, app='sensitivity', title='')` | Converte o resultado de `sensitivity_study` num `StudyResult` de sensibilidade. |

#### `pinneapple_analysis.study_models`

`StudyModel`: um só envelope de modelo para estudos (função, modelo do zoo, ONNX ou funções de teste embutidas).

<small>`pinneapple_analysis/study_models.py` · 211 linhas · testes: 25</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `StudyModel` | classe | `(name, kind, input_names, output_names, predict_fn, units, bounds, description, torch_fn,…)` | Envelope de modelo para estudos: predição, nomes, unidades, limites, desvio opcional e construtores de função, zoo e ONNX. |
| `builtin` | função | `(name) -> StudyModel` | Modelo de teste embutido (`ishigami` ou `sobol_g`) com índices de Sobol exatos nos metadados. |
| `columns` | função | `(table, names) -> np.ndarray` | Empilha colunas nomeadas de uma tabela num array (n, k). |

#### `pinneapple_analysis.surrogate`

Surrogates a partir de resultados de simulação; reexporta dados, divisões, candidatos, estudo, limites e o modelo empacotado.

<small>`pinneapple_analysis/surrogate/__init__.py` · 40 linhas · testes: 32</small>

#### `pinneapple_analysis.surrogate.contracts`

Leitura de tabelas com papéis de coluna (`DatasetSpec`) e conversão do estudo de surrogate e da arena de regimes para os contratos compartilhados.

<small>`pinneapple_analysis/surrogate/contracts.py` · 217 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `read_table_with_roles` | função | `(path, *, max_rows=None)` | Lê uma tabela e os papéis de coluna do `DatasetSpec` embutido ou do arquivo `.contract.json`. |
| `data_from_spec` | função | `(df, spec, *, group=None, grid_shape=None, meta=None) -> SurrogateData` | Monta `SurrogateData` a partir dos papéis de coluna do spec. |
| `dataset_spec` | função | `(data, name, *, app='', description='')` | Cria um `DatasetSpec` para os dados de surrogate. |
| `model_ref` | função | `(model, *, name, version, uri, app='', dataset_name='')` | Cria um `ModelRef` para um modelo de surrogate publicado. |
| `study_result` | função | `(study, *, id, app='surrogate', ref=None, title='')` | Converte o resultado do estudo de surrogate num `StudyResult`. |
| `arena_study_result` | função | `(result, *, id, app='arena', title='')` | Converte o resultado da arena de regimes num `StudyResult`, uma linha por método e regime. |

#### `pinneapple_analysis.surrogate.data`

Dados de treino de surrogate (`SurrogateData`): parâmetros de entrada para escalares ou campos em grade, lidos de tabelas, arrays ou snapshots.

<small>`pinneapple_analysis/surrogate/data.py` · 365 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SurrogateData` | classe | `(X, Y, input_names, output_names, coords, coord_names, groups, meta)` | Dados de surrogate: entradas X e saídas Y escalares ou campos, com nomes, coordenadas, grupos e hash. |
| `grid_coords` | função | `(shape, ranges=None) -> np.ndarray` | Coordenadas de uma grade produto tensorial na ordem `ij`. |
| `inspect_table` | função | `(df, roles=None) -> dict[str, Any]` | Diz o que uma tabela oferece: colunas numéricas, campos, colunas de texto e entradas/saídas sugeridas. |
| `read_table` | função | `(source, name=None, *, fmt=None)` | Lê uma tabela de caminho ou bytes (CSV, Parquet, JSON, HDF5, NPZ) num DataFrame. |
| `field_columns` | função | `(columns, name) -> list[str]` | Colunas de um campo indexado, em ordem. |
| `from_table` | função | `(df, inputs, outputs, *, group=None, grid_shape=None, coords=None, coord_names=(), meta=N…)` | Escolhe entradas e saídas de um DataFrame; descarta linhas com valores faltantes. |
| `from_arrays` | função | `(arrays, *, input_names=None, output_names=None, meta=None) -> SurrogateData` | Cria `SurrogateData` de um dicionário de arrays (layout de export NPZ ou HDF5). |
| `lattice_order` | função | `(points) -> tuple[tuple[int, ...], np.ndarray] \| None` | Detecta se os pontos formam uma grade produto tensorial e devolve forma e ordem. |
| `from_snapshots` | função | `(snaps, *, meta=None) -> SurrogateData` | Converte um `SnapshotSet` paramétrico em reticulado num conjunto de campos. |
| `snapshot_values` | função | `(snaps, data, Y) -> np.ndarray` | Leva campos previstos no reticulado de volta às linhas no layout dos snapshots. |

#### `pinneapple_analysis.surrogate.examples`

Conjuntos de exemplo sem download: aerofólio RANS do zoo, campos de Burgers exatos, viga cantilever 3-D do ROM Lab e um baseline de diferenças finitas. [sem teste direto]

<small>`pinneapple_analysis/surrogate/examples.py` · 126 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `airfoil_rans` | função | `() -> SurrogateData` | Conjunto do zoo de aerofólios CST com RANS (entradas CST e ângulo, saídas Cl e Cd). |
| `burgers_fields` | função | `(n=96, nu_range=(0.004, 0.1), nx=48, nt=12, t_max=0.9, seed=0) -> SurrogateData` | Campos exatos de Burgers viscoso para várias viscosidades. |
| `cantilever_3d` | função | `(path, n_timed=2) -> SurrogateData` | Viga cantilever 3-D FEM do ROM Lab como conjunto de campos (deslocamento e von Mises). |
| `burgers_fdm_baseline` | função | `(nx=201, cfl=0.4)` | Baseline numérico de diferenças finitas para `burgers_fields`, interpolado na grade do conjunto. |

#### `pinneapple_analysis.surrogate.heldout`

Erros de um surrogate de campo nos dados reservados, por campo, por região do corpo e nas grandezas escalares que o engenheiro lê do campo.

<small>`pinneapple_analysis/surrogate/heldout.py` · 100 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `region_rel_l2` | função | `(pred, true, regions) -> dict[str, float]` | L2 relativo restrito aos pontos de cada região. |
| `heldout_report` | função | `(snaps, data, model, idx, *, grow=0.1) -> dict[str, Any]` | Relatório de erros do modelo nas amostras reservadas: por campo, região, amostra e grandezas escalares. |

#### `pinneapple_analysis.surrogate.limits`

Onde o surrogate pode ser usado: limites de validade, portão OOD por k-vizinhos e onde os erros se concentram. [sem teste direto]

<small>`pinneapple_analysis/surrogate/limits.py` · 147 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OODGate` | classe | `(X_train, names, knn_limit=3.0)` | Portão OOD: limites de validade das entradas e distância de k-vizinhos normalizada. |
| `per_sample_error` | função | `(pred, true, scale) -> np.ndarray` | Erro por amostra: RMS das saídas normalizadas pelo desvio do treino. |
| `limitations` | função | `(gate, X_eval, err, sets, X_train, *, field_err=None, coords=None, coord_names=(), metric…)` | Resume onde o surrogate erra: erros por conjunto, por região de entrada e mapa de erro de campo. |

#### `pinneapple_analysis.surrogate.methods`

Candidatos de surrogate (ridge, GP, GBT, MLP, FNO, DeepONet, PINN), regras de aplicabilidade e treino sob `ComputeBudget`.

<small>`pinneapple_analysis/surrogate/methods.py` · 502 linhas · testes: 18 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TrainContext` | classe | `(Xtr, Ytr, Xva, Yva, budget, seed, grid_shape, coords, pde, options)` | O que um método recebe: arrays, escaladores do treino, orçamento, semente, layout do campo e EDP. |
| `Fitted` | classe | `(method, n_params, info)` | Um candidato treinado com `predict(X_raw) -> Y_raw` e tamanho em bytes. |
| `NonNegative` | classe | `(inner, mask)` | Envelope que treina na raiz quadrada de saídas marcadas e prevê seu quadrado, evitando valores negativos. |
| `Candidate` | classe | `(id, title, family, iterative, fit, check)` | Registro de um candidato: id, título, família, se é iterativo, função de treino e checagem de aplicabilidade. |
| `candidate_ids` | função | `() -> list[str]` | Lista os ids dos candidatos registrados. |
| `applicable` | função | `(kind, n_train, grid_shape=(), has_pde=False) -> dict[str, str \| None]` | Para cada candidato, None se aplicável ou o motivo de não ser. |
| `fit_candidate` | função | `(method, ctx) -> tuple[Fitted, float]` | Treina um candidato e devolve o modelo e o tempo de treino. |

#### `pinneapple_analysis.surrogate.model`

`SurrogateModel`: predição com portão OOD, salvar/carregar, exportar (só candidatos PyTorch; outros levantam NotImplementedError) e cartão do modelo. [sem teste direto]

<small>`pinneapple_analysis/surrogate/model.py` · 179 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SurrogateModel` | classe | `(fitted, gate, *, input_names, output_names, grid_shape=(), coords=None, coord_names=(), …)` | Surrogate treinado e empacotado: predição com portão OOD, salvar/carregar, `ModelRef` e export TorchScript/ONNX. |

#### `pinneapple_analysis.surrogate.pde`

Declarações de EDP (heat, burgers, advection) que dizem qual resíduo o candidato PINN soma à perda de dados. [sem teste direto]

<small>`pinneapple_analysis/surrogate/pde.py` · 142 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PDEDeclaration` | classe | `(type, coord_index, coefficients)` | Declaração de EDP (tipo, coordenadas, coeficientes) com resíduo em torch e por diferenças finitas. |
| `parse_pde` | função | `(decl, coord_names, input_names) -> PDEDeclaration \| None` | Valida uma declaração de EDP contra os nomes de coordenadas e entradas; None passa adiante. |

#### `pinneapple_analysis.surrogate.splits`

Divisões treino/validação/teste e um conjunto de mudança de distribuição retirado antes da divisão aleatória. [sem teste direto]

<small>`pinneapple_analysis/surrogate/splits.py` · 105 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Splits` | classe | `(train, val, test, shift, shift_kind, shift_description, seed, info)` | Índices de treino, validação, teste e mudança de distribuição, com descrição do tipo de mudança. |
| `make_splits` | função | `(data, *, test_frac=0.2, val_frac=0.15, shift='auto', shift_frac=0.15, seed=0) -> Splits` | Cria índices disjuntos; o conjunto de mudança sai antes da divisão aleatória. |

#### `pinneapple_analysis.surrogate.study`

`run_study`: treina cada candidato aplicável nos mesmos dados e orçamento, mede em dados não vistos, escolhe pela validação e empacota com limitações e `TrustReport`.

<small>`pinneapple_analysis/surrogate/study.py` · 380 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `StudyResult` | classe | `(data, splits, budget, seeds, methods, best, model, limitations, trust, metric, created, …)` | Resultado do estudo de surrogate: métodos, melhor, modelo, limitações, confiança, ranking e Markdown. |
| `evaluate` | função | `(kind, pred, true, scale, output_names) -> dict[str, Any]` | Métricas das predições achatadas contra a verdade (nRMSE para escalares, L2 relativo para campos). |
| `run_study` | função | `(data, *, methods='auto', budget=10.0, seeds=1, splits=None, split_options=None, pde=None…)` | Roda o estudo de surrogate completo sob um orçamento por run de treino. |
| `nonnegative_mask` | função | `(data) -> np.ndarray \| None` | Máscara das saídas declaradas não negativas em `meta['nonnegative']`. |

#### `pinneapple_analysis.trust`

Pontuação de confiança antes de consumir predições; reexporta `TrustGate`, `TrustReport` e o cartão de campo. [sem teste direto do __init__]

<small>`pinneapple_analysis/trust/__init__.py` · 20 linhas · **sem teste direto**</small>

#### `pinneapple_analysis.trust.field_card`

Cartão de confiança de um campo em grade: checagens de sanidade, física, referência e generalização que viram decisão de deploy.

<small>`pinneapple_analysis/trust/field_card.py` · 358 linhas · testes: 12</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FieldCheck` | classe | `(name, kind, passed, value, threshold, critical, detail)` | Uma checagem de campo (sanity, physics, reference, generalization); `passed` None quando não pôde rodar. |
| `FieldTrustCard` | classe | `(checks, stages, decision, reasons, name)` | Cartão de confiança de um campo: checagens, estágios, decisão e motivos, com saída em Markdown. |
| `field_checks` | função | `(grid, values, *, pde=None, source=None, alpha=1.0, time_coord='t', dirichlet=None, refer…)` | Roda as checagens de um campo escalar em grade; checagem sem entrada aparece como não executada, com motivo. |
| `trust_card` | função | `(grid, values, *, name='field', is_ml=True, limitations=(), min_coverage=0.5, **kwargs) -…)` | Roda `field_checks` e produz a decisão de deploy e as seis perguntas do cartão de confiança. |

#### `pinneapple_analysis.trust.trust_gate`

`TrustGate`: combina OOD, resíduo físico e variância de ensemble num escore antes de a predição alimentar um gêmeo digital.

<small>`pinneapple_analysis/trust/trust_gate.py` · 199 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TrustScore` | classe | `(combined, ood_score, residual_score, ensemble_score, details)` | Escore combinado e as partes OOD, resíduo e ensemble de uma predição. |
| `TrustGate` | classe | `(*, ood_weight=0.4, residual_weight=0.4, ensemble_weight=0.2, ood_level=0.01)` | Ajusta o domínio de treino e pontua uma predição combinando OOD, resíduo e variância de ensemble. |

#### `pinneapple_analysis.trust.trust_report`

`TrustReport`: junta checagens independentes numa decisão APPROVED, REVIEW ou REJECT com cobertura e score; API com nomes em português. [experimental]

<small>`pinneapple_analysis/trust/trust_report.py` · 179 linhas · testes: 13 · palavras-chave no código (heurística; o texto acima diz o status real): experimental</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Check` | classe | `(name, status, critical, weight, value, limit, source, reason)` | Uma checagem do relatório (status supports, contradicts ou not_run, crítica, peso, valor, limite, fonte). |
| `TrustReport` | classe | `(checks, min_coverage, min_score, model, details)` | Junta checagens e calcula cobertura, score e decisão APPROVED/REVIEW/REJECT com motivos. |
| `checagem_erro_relativo` | função | `(nome, erro, limite, *, fonte='', critica=False, peso=1.0, motivo_nao_executada='') -> Ch…)` | Cria uma checagem de acurácia: erro relativo menor ou igual ao limite. |
| `checagem_booleana` | função | `(nome, passou, *, fonte='', critica=False, peso=1.0, motivo_nao_executada='') -> Check` | Cria uma checagem booleana de consistência física ou comparação qualitativa. |
| `from_trust_score` | função | `(ts, *, limite_sub_score=0.7, fonte='TrustGate') -> List[Check]` | Converte o `TrustScore` do `TrustGate` em checagens de OOD, resíduo e ensemble. |

#### `pinneapple_analysis.uncertainty`

Quantificação de incerteza (antigo `pinneapple_uq`); reexporta aleatória, epistêmica, conformal, calibração, quantis, reamostragem, drift, tolerâncias e `uq_predict`.

<small>`pinneapple_analysis/uncertainty/__init__.py` · 186 linhas · testes: 39</small>

#### `pinneapple_analysis.uncertainty.aleatoric`

Incerteza aleatória por regressão heteroscedástica: cabeça de log-variância e perda NLL gaussiana.

<small>`pinneapple_analysis/uncertainty/aleatoric.py` · 164 linhas · testes: 157</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AleatoricHead` | classe | `(base, out_dim, in_dim, *, hidden=64) -> None` | Envolve um modelo determinístico e acrescenta uma cabeça que aprende a log-variância. |
| `aleatoric_nll_loss` | função | `(mean, log_var, target, *, min_log_var=-10.0, max_log_var=10.0) -> Tensor` | Perda NLL gaussiana heteroscedástica com log-variância limitada. |

#### `pinneapple_analysis.uncertainty.applicability`

Domínio de aplicabilidade (portão de densidade k-NN + faixa) e regra que manda para o solver de alta fidelidade os casos que o modelo não pode responder.

<small>`pinneapple_analysis/uncertainty/applicability.py` · 163 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ApplicabilityDomain` | classe | `(names, lo, hi, center, scale, k, level, threshold, n_ref, bounds_source, _tree, _gate)` | Portão de densidade k-NN e de faixa sobre entradas padronizadas. |
| `route_cases` | função | `(status, pred, lower, upper, tolerance=None, limits=None) -> list[dict]` | Regra de roteamento por caso: surrogate ou high_fidelity, com os motivos. |
| `routing_summary` | função | `(routes) -> dict` | Contagens de cada rota e de cada código de motivo. |

#### `pinneapple_analysis.uncertainty.calibration`

Métricas de calibração e nitidez para regressão probabilística (ECE, cobertura, curva de calibração, NLL gaussiana).

<small>`pinneapple_analysis/uncertainty/calibration.py` · 302 linhas · testes: 185</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CalibrationMetrics` | classe | `()` | Métodos estáticos de calibração e nitidez (ECE, cobertura, dados de curva, nitidez, NLL). |

#### `pinneapple_analysis.uncertainty.conformal`

Predição conformal split para modelos PyTorch, com quantil de amostra finita e checagem de tamanho mínimo de calibração.

<small>`pinneapple_analysis/uncertainty/conformal.py` · 281 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ConformalPredictor` | classe | `(model, alpha=0.1) -> None` | Preditor conformal split: calibra num conjunto reservado e dá intervalos com cobertura garantida. |
| `conformal_quantile` | função | `(scores, alpha) -> float` | Quantil conformal split de qualquer array de escores de não conformidade. |
| `conformal_quantile_valid` | função | `(n, alpha) -> bool` | Diz se `n` escores de calibração bastam para a garantia de amostra finita no nível `alpha`. |

#### `pinneapple_analysis.uncertainty.conformal_study`

Intervalos conformais para predições NumPy de qualquer modelo, com cobertura medida num teste separado e cobertura por grupo.

<small>`pinneapple_analysis/uncertainty/conformal_study.py` · 195 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ConformalStudy` | classe | `(alpha, q, normalized, n_cal, n_test, coverage, coverage_ci, mean_width, median_width, le…)` | Calibração conformal e sua avaliação reservada (cobertura, intervalo, larguras, cobertura por grupo). |
| `conformal_scores` | função | `(y, pred, scale=None) -> np.ndarray` | Escores de não conformidade `\|y - pred\|`, divididos pela escala quando dada. |
| `conformal_quantile` | função | `(scores, alpha=0.1) -> float` | Quantil conformal de amostra finita dos escores. |
| `conformal_interval` | função | `(pred, q, scale=None) -> tuple[np.ndarray, np.ndarray]` | Intervalo `pred -+ q` (vezes a escala quando normalizado). |
| `min_calibration_size` | função | `(alpha) -> int` | Menor conjunto de calibração com quantil finito no nível `1 - alpha`. |
| `coverage_by_group` | função | `(covered, groups, labels=None, level=0.95, width=None) -> list[dict]` | Cobertura medida por grupo, mostrando subcobertura escondida pela garantia marginal. |
| `quantile_bins` | função | `(x, n_bins=5) -> tuple[np.ndarray, list[str]]` | Classes de mesma contagem de `x` com rótulos. |
| `conformal_study` | função | `(y_cal, pred_cal, y_test, pred_test, alpha=0.1, scale_cal=None, scale_test=None, level=0.…)` | Calibra em `(y_cal, pred_cal)` e mede a cobertura em `(y_test, pred_test)`. |
| `split_indices` | função | `(n, fractions=(0.5, 0.5), seed=0) -> list[np.ndarray]` | Conjuntos disjuntos aleatórios de índices com as frações dadas. |

#### `pinneapple_analysis.uncertainty.core`

Tipo `UQResult` e a interface unificada `uq_predict` que despacha para MC Dropout, ensemble e outros métodos. [sem teste direto]

<small>`pinneapple_analysis/uncertainty/core.py` · 265 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `UQResult` | classe | `(mean, std, aleatoric_std, epistemic_std, samples, metadata)` | Contêiner de predição com média, desvio, partes aleatória e epistêmica, amostras e intervalos. |
| `uq_predict` | função | `(model, x, method='mc_dropout', device=None, **kwargs) -> UQResult` | Interface unificada de predição com incerteza (método escolhido por `method`). |

#### `pinneapple_analysis.uncertainty.decomposition`

Separa a variância preditiva em parte aleatória e epistêmica a partir de passes estocásticos de um modelo com cabeça aleatória.

<small>`pinneapple_analysis/uncertainty/decomposition.py` · 143 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `decompose_uncertainty` | função | `(model, x, *, n_samples=100, has_aleatoric=True, device=None) -> UQResult` | Separa a incerteza preditiva em aleatória e epistêmica. |

#### `pinneapple_analysis.uncertainty.distributions`

Entradas incertas de um estudo: uma distribuição por parâmetro, amostrada por Monte Carlo, hipercubo latino ou Sobol.

<small>`pinneapple_analysis/uncertainty/distributions.py` · 187 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `UncertainParameter` | classe | `(name, dist, params, unit, nominal)` | Uma entrada incerta com distribuição, parâmetros, unidade, valor nominal, ppf, cdf e suporte. |
| `as_parameter` | função | `(spec, name=None) -> UncertainParameter` | Cria um parâmetro a partir de dict, par (low, high) uniforme ou `UncertainParameter`. |
| `as_space` | função | `(space) -> list[UncertainParameter]` | Cria a lista de parâmetros de lista, dicionário ou objeto com `parameters`. |
| `to_values` | função | `(space, U) -> np.ndarray` | Leva pontos do hipercubo unitário a valores pelos inversos das CDFs. |
| `unit_sample` | função | `(d, n, method='lhs', seed=0) -> np.ndarray` | Amostra o hipercubo unitário por MC, hipercubo latino ou Sobol embaralhado. |
| `sample_space` | função | `(space, n, method='lhs', seed=0) -> np.ndarray` | Amostra `n` pontos dos parâmetros do espaço. |

#### `pinneapple_analysis.uncertainty.drift`

Monitoramento de drift de um surrogate em uso: acurácia, calibração, mudança de entrada e tendência do resíduo, com decisão ok/review/retrain.

<small>`pinneapple_analysis/uncertainty/drift.py` · 208 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DriftThresholds` | classe | `(nrmse_review, nrmse_retrain, bias_z, coverage_z, ks_alpha, oob_review, oob_retrain, tren…)` | Limiares da decisão ok/review/retrain de `assess_drift`. |
| `DriftReport` | classe | `(n, nrmse, bias, bias_z, level, coverage, ks_stat, ks_pvalue, out_of_bounds, trend_tau, t…)` | Resultado de `assess_drift` com nRMSE, viés, cobertura, KS, fora de faixa, tendência e decisão. |
| `assess_drift` | função | `(y_true, y_pred, sigma=None, X_ref=None, X_new=None, bounds=None, level=0.95, thresholds=…)` | Checa um lote de predições quanto a perda de acurácia, má calibração, mudança de entrada e tendência do resíduo. |

#### `pinneapple_analysis.uncertainty.ensemble`

Incerteza por Deep Ensemble: média e dispersão de vários modelos treinados de forma independente.

<small>`pinneapple_analysis/uncertainty/ensemble.py` · 251 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EnsembleConfig` | classe | `(n_members, diversity_reg, seed)` | Configuração do Deep Ensemble (número de membros, regularização de diversidade, semente). |
| `EnsembleUQ` | classe | `(models, config=None) -> None` | Envelope de Deep Ensemble que dá média e incerteza a partir dos membros. |

#### `pinneapple_analysis.uncertainty.margin`

Estudo de margem por Monte Carlo: um projeto que passa nas dimensões nominais ainda passa com as tolerâncias de fabricação?

<small>`pinneapple_analysis/uncertainty/margin.py` · 130 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `corner_stackup` | função | `(fn, nominal, tolerances) -> dict[str, Any]` | Avalia `fn` nos 2^d cantos da caixa de tolerâncias. |
| `summarize_samples` | função | `(values, limit, marginal_fraction=0.05) -> dict[str, Any]` | Estatísticas da distribuição e veredito pelo p95 contra um limite superior. |
| `margin_study` | função | `(fn, nominal, tolerances, limit, *, n=200, n_small=50, dist='uniform', seed=None, check_s…)` | Estudo de margem Monte Carlo contra um limite superior, com checagens do próprio estudo. |

#### `pinneapple_analysis.uncertainty.mc_dropout`

MC Dropout para incerteza epistêmica: envolve qualquer `nn.Module` e mantém o dropout ativo na inferência.

<small>`pinneapple_analysis/uncertainty/mc_dropout.py` · 414 linhas · testes: 163</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MCDropoutConfig` | classe | `(n_samples, dropout_p, seed)` | Configuração do MC Dropout (amostras, probabilidade, semente). |
| `MCDropoutWrapper` | classe | `(model, config) -> None` | Envolve um `nn.Module` e liga dropout na inferência por hooks; avisa quando não há ativação oculta para descartar. |
| `MCDropout` | classe | `(model, config=None) -> None` | Classe de conveniência que aplica MC Dropout a qualquer modelo chamável. |

#### `pinneapple_analysis.uncertainty.pce`

Caos polinomial (Legendre) por mínimos quadrados: média, variância e índices de Sobol a partir de uma regressão.

<small>`pinneapple_analysis/uncertainty/pce.py` · 123 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PCEResult` | classe | `(names, degree, n_terms, n_evals, mean, std, S1, ST, loo_q2, alphas, coef)` | Expansão ajustada: média, desvio, índices S1/ST, Q² leave-one-out e coeficientes. |
| `multi_indices` | função | `(d, degree) -> np.ndarray` | Todos os multi-índices de grau total até `degree`. |
| `legendre_design` | função | `(xi, alphas) -> np.ndarray` | Matriz de desenho de produtos de Legendre ortonormais. |
| `n_terms` | função | `(d, degree) -> int` | Número de termos de uma expansão de grau total em `d` variáveis. |
| `polynomial_chaos` | função | `(fn, space, degree=3, n=None, seed=0, max_terms=500) -> PCEResult` | Ajusta um caos de Legendre a `fn` em amostras de hipercubo latino do espaço. |

#### `pinneapple_analysis.uncertainty.posterior_metrics`

Métricas de comparação de posteriores para solvers inversos generativos (MMD, Wasserstein fatiado, espectro radial, CRPS).

<small>`pinneapple_analysis/uncertainty/posterior_metrics.py` · 172 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `normalize_weights` | função | `(weights, n) -> np.ndarray` | Valida e normaliza pesos de amostras (uniformes quando None). |
| `weighted_mmd` | função | `(pred, ref, pred_weights=None, ref_weights=None) -> float` | MMD ponderado com kernel gaussiano e banda pela mediana das distâncias. |
| `weighted_wasserstein_1d` | função | `(x, y, wx=None, wy=None) -> float` | Distância de Wasserstein-1 entre duas amostras 1-D ponderadas. |
| `grf_direction` | função | `(h, w, rng, alpha=2.0, tau=3.0) -> np.ndarray` | Direção aleatória suave de norma unitária vinda de um campo gaussiano aleatório. |
| `sliced_wasserstein` | função | `(pred, ref, pred_weights=None, ref_weights=None, n_projections=128, seed=0, alpha=2.0, ta…)` | Wasserstein fatiado entre ensembles de campos por projeções GRF aleatórias. |
| `radial_power_spectrum` | função | `(samples, weights=None) -> np.ndarray` | Espectro de potência radial médio ponderado de amostras 2-D. |
| `spectral_relative_geomean` | função | `(pred_spectrum, ref_spectrum, eps=1e-10) -> float` | Média geométrica do erro relativo entre dois espectros. |
| `evaluate_posterior` | função | `(pred, ref, pred_weights=None, ref_weights=None, *, n_projections=128, seed=0, swd_mean=N…)` | Compara um ensemble posterior previsto com um de referência (menor é melhor). |
| `crps_gaussian` | função | `(mu, sigma, y) -> np.ndarray` | CRPS em forma fechada para uma preditiva gaussiana. |
| `crps_ensemble` | função | `(samples, y, *, fair=False) -> np.ndarray` | CRPS de ensemble, com opção de normalização justa. |

#### `pinneapple_analysis.uncertainty.quantile`

Regressão quantílica: cabeça de quantis, perda pinball e taxa de cruzamento de quantis.

<small>`pinneapple_analysis/uncertainty/quantile.py` · 212 linhas · testes: 161</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `QuantileConfig` | classe | `(quantiles)` | Configuração dos níveis de quantil. |
| `QuantileHead` | classe | `(base, cfg, hidden_dim=64, rearrange=True) -> None` | Envolve um modelo pontual e projeta a saída em vários quantis, com reordenação opcional. |
| `QuantileLoss` | classe | `(quantiles) -> None` | Perda pinball de múltiplos quantis compatível com o Trainer. |
| `pinball_loss_torch` | função | `(y_pred, y_true, q) -> torch.Tensor` | Perda pinball de um nível de quantil. |
| `crossing_rate` | função | `(y_pred, quantiles) -> float` | Fração de predições com quantis fora de ordem. |

#### `pinneapple_analysis.uncertainty.reachability`

Viabilidade de encontro por conjuntos alcançáveis (risco de colisão, rendezvous); emite ResearchOnlyWarning e é recusado em modo comercial. [experimental]

<small>`pinneapple_analysis/uncertainty/reachability.py` · 115 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BodyState` | classe | `(position, velocity, pos_uncertainty, vel_uncertainty, max_accel, latency)` | Estado de um corpo com incertezas de posição e velocidade, aceleração máxima e latência; dá centro e raio da bola alcançável. |
| `encounter_feasibility` | função | `(a, b, horizon, *, contact_distance=0.0, n=2001, refine=True) -> Dict[str, object]` | Margem entre as bolas alcançáveis de dois corpos no horizonte; diz se o contato é possível. |
| `time_to_reach_region` | função | `(chaser, target_center, target_radius=0.0, horizon=1000000.0) -> Optional[float]` | Primeiro instante em que uma região alvo estática fica dentro da bola alcançável. |

#### `pinneapple_analysis.uncertainty.resampling`

Intervalos de confiança por reamostragem para dados dependentes: bootstrap em blocos e por grupo, skill score e jackknife.

<small>`pinneapple_analysis/uncertainty/resampling.py` · 206 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `block_bootstrap` | função | `(x, stat=np.mean, block=20, n=2000, seed=0, level=0.95) -> tuple[float, float, float]` | Bootstrap de blocos móveis de uma estatística para amostra com dependência serial. |
| `group_bootstrap` | função | `(x, groups, stat=np.mean, n=2000, seed=0, level=0.95) -> tuple[float, float, float]` | Bootstrap por grupos inteiros com reposição. |
| `skill_score` | função | `(err_model, err_ref, *, block=None, groups=None, n=2000, seed=0, level=0.95) -> tuple[flo…)` | Skill score `1 - E\|e_model\| / E\|e_ref\|` com intervalo bootstrap pareado. |
| `jackknife` | função | `(samples, fn, n_blocks=20)` | Estimativa jackknife de um bloco removido e erro padrão. |

#### `pinneapple_analysis.uncertainty.residual_bound`

Transforma o resíduo da EDP de uma predição (sem precisar da resposta verdadeira) num limite de erro, garantido em norma de energia ou calibrado por conformal.

<small>`pinneapple_analysis/uncertainty/residual_bound.py` · 242 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LinearSPDProblem` | classe | `(A, f, L, c, weight, shape, _L_solve)` | Problema linear `A u = f` com A SPD, com resíduo, norma de energia, norma dual e solução. |
| `CoverageReport` | classe | `(label, coverage, target, n, holds, tolerance, median_bound, note)` | Relatório da cobertura medida de um limite contra o alvo. |
| `ResidualLoss` | classe | `(problems)` | Perda física diferenciável (resíduo relativo) sobre um conjunto fixo de problemas, em PyTorch esparso. |
| `ResidualErrorCalibrator` | classe | `(alpha=0.1, mode='ratio')` | Mapa conformal split de uma medida de resíduo para um limite de erro. |
| `darcy_problem` | função | `(a, f=1.0) -> LinearSPDProblem` | Problema de Darcy na discretização de `operator_benchmarks.darcy_2d`. |
| `heat_conduction_problem` | função | `(k, q) -> LinearSPDProblem` | Condução estacionária com T = 0 na borda, mesmo operador de Darcy. |
| `energy_error_bound` | função | `(problem, u) -> dict[str, float]` | Limite garantido do erro em norma de energia a partir só do resíduo. |
| `coverage_study` | função | `(res, err, *, alpha=0.1, n_cal=None, n_splits=200, seed=0) -> dict[str, float]` | Cobertura do limite em modo ratio sobre várias divisões aleatórias de calibração e teste. |

#### `pinneapple_analysis.uncertainty.study`

Estudo de incerteza de uma saída: intervalos conformais medidos, portão de aplicabilidade, triagem de casos e propagação de entradas incertas.

<small>`pinneapple_analysis/uncertainty/study.py` · 220 linhas · testes: 12</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `uncertainty_study` | função | `(model, output, X_cal, y_cal, X_test, y_test, *, alpha=0.1, scale='absolute', X_ref=None,…)` | Roda o estudo de incerteza de uma saída de um `StudyModel`. |

#### `pinneapple_analysis.uncertainty.tolerance`

Propagação Monte Carlo de tolerâncias de entrada por qualquer modelo (JCGM 101), com probabilidade de ficar dentro dos limites e intervalo de Wilson.

<small>`pinneapple_analysis/uncertainty/tolerance.py` · 166 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ToleranceResult` | classe | `(mean, std, percentiles, n, inputs, outputs, p_within, p_within_ci, p_within_each)` | Resultado de `propagate`: média, desvio, percentis e probabilidade de ficar dentro dos limites. |
| `wilson_interval` | função | `(k, n, level=0.95) -> tuple[float, float]` | Intervalo de score de Wilson para uma proporção binomial. |
| `propagate` | função | `(fn, nominal, tolerances, n=10000, dist='normal', limits=None, seed=0, level=0.95, percen…)` | Propaga tolerâncias independentes por `fn` via Monte Carlo. |
| `propagate_space` | função | `(fn, space, n=4096, method='lhs', limits=None, seed=0, level=0.95, percentiles=(1, 5, 50,…)` | Propaga entradas com qualquer distribuição por `fn`. |
| `src` | função | `(X, y) -> tuple[np.ndarray, float]` | Coeficientes de regressão padronizados e R² do ajuste linear: quanto da dispersão cada tolerância explica. |

#### `pinneapple_analysis.validation`

Reexporta as checagens clássicas (`PhysicsValidator`, `ConservationCheck`, `BoundaryCheck`, `SymmetryCheck`, `validate_model`), comparações com solução analítica/solver, funções espectrais, metamórficas e do relatório-artefato. O docstring vem depois de `from __future__` e por isso não vira docstring do módulo.

<small>`pinneapple_analysis/validation/__init__.py` · 82 linhas · testes: 21</small>

#### `pinneapple_analysis.validation.analytical`

Compara as predições de um modelo com uma solução analítica em pontos aleatórios do domínio, ou com dados de um solver de referência (RMSE, L2 relativo, erro máximo). [sem teste direto]

<small>`pinneapple_analysis/validation/analytical.py` · 187 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `compare_to_analytical` | função | `(model, analytical_fn, coord_names, domain_bounds, n_points=5000, field_names=None, devic…)` | Compara as predições do modelo com uma função analítica em pontos amostrados no domínio. |
| `validate_against_solver` | função | `(model, solver_data, coord_names, field_names, device='cpu') -> dict` | Compara as predições do modelo com dados de referência de um solver. |

#### `pinneapple_analysis.validation.arena`

Red team / blue team para validação: geradores adversariais de casos errados porém plausíveis contra as checagens, e um laço que as endurece; o módulo é chamável como `run_arena`.

<small>`pinneapple_analysis/validation/arena/__init__.py` · 43 linhas · testes: 12</small>

#### `pinneapple_analysis.validation.arena.__main__`

Ponto de entrada `python -m pinneapple_analysis.validation.arena`: roda a arena em Darcy 2D e Burgers 1D. [sem teste direto]

<small>`pinneapple_analysis/validation/arena/__main__.py` · 49 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `main` | função | `(argv=None) -> int` | Executa a arena pela linha de comando e devolve o código de saída. |

#### `pinneapple_analysis.validation.arena.blue`

Blue team: limiares calibrados com taxa de falso alarme declarada (teste conformal dividido combinado) e endurecimento com checagens extras até rejeitar as fugas guardadas. [sem teste direto]

<small>`pinneapple_analysis/validation/arena/blue.py` · 201 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Defence` | classe | `(checks, thresholds, kind, alpha, alpha_cal, scales, statistic_threshold, n_calibration, …)` | Conjunto de checagens com um limiar cada (rejeita quando algum valor passa do limiar), com estatística combinada e descrição. |
| `score_matrix` | função | `(checks, cases) -> dict[str, np.ndarray]` | Calcula o valor de cada checagem em cada caso. |
| `calibrate_scores` | função | `(checks, values, alpha=0.01, *, safety=0.5, seed=0, false_alarms=None, kind=…, floor=0.01…)` | Calibra os limiares a partir de valores já calculados nos casos limpos, mantendo os falsos alarmes guardados abaixo do limiar. |
| `calibrate` | função | `(checks, clean_cases, alpha=0.01, *, safety=0.5, seed=0, false_alarms=()) -> Defence` | Teste conformal dividido combinado das checagens sobre casos limpos. |
| `harden` | função | `(current, library, clean_cases, *, escapes=(), false_alarms=(), alpha=0.01, safety=0.5, s…)` | Calibra a defesa e adiciona checagens da biblioteca até rejeitar todas as fugas guardadas (ou nenhuma ajudar). |

#### `pinneapple_analysis.validation.arena.cases`

Guarda casos de regressão da arena e os reexecuta (usado por `pinneapple check --arena-cases`). [sem teste direto]

<small>`pinneapple_analysis/validation/arena/cases.py` · 108 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ReplayResult` | classe | `(path, problem, rows)` | Resultado da reexecução de um arquivo de casos; aponta regressões. |
| `save_cases` | função | `(path, problem, defence, findings, meta=None) -> Path` | Grava problema, defesa e achados num arquivo de casos. |
| `load_cases` | função | `(path) -> dict[str, Any]` | Lê um arquivo de casos. |
| `builtin_cases` | função | `() -> list[Path]` | Lista os arquivos de casos que vêm com a biblioteca (um por problema). |
| `load_defence` | função | `(path) -> tuple[Any, Defence]` | Devolve o problema e a defesa endurecida guardados num arquivo de casos. |
| `replay` | função | `(path) -> ReplayResult` | Regenera cada caso guardado e o confere contra a defesa guardada. |

#### `pinneapple_analysis.validation.arena.generators`

Geradores do red team: famílias parametrizadas de saídas erradas porém plausíveis que a busca de evasão explora. [sem teste direto]

<small>`pinneapple_analysis/validation/arena/generators.py` · 164 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Generator` | classe | `(name, bounds, fn, problems, physics_consistent, description)` | Família parametrizada de saídas erradas porém plausíveis, com limites e flag de consistência física. |
| `get_generator` | função | `(name) -> Generator` | Devolve um gerador pelo nome. |
| `generators_for` | função | `(problem) -> list[Generator]` | Lista os geradores aplicáveis a um problema. |

#### `pinneapple_analysis.validation.arena.loop`

Laço da arena: rodadas de red team e depois blue team, o relatório e a auditoria de um modelo externo contra uma defesa. [sem teste direto]

<small>`pinneapple_analysis/validation/arena/loop.py` · 396 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ArenaBudget` | classe | `(n_bases, evals, fa_seeds, fa_evals, n_calibration, n_holdout, n_mutation_cases, n_datase…)` | Quanto o red team busca e quantos casos limpos o blue team usa para calibrar. |
| `RoundReport` | classe | `(round, defence_before, defence_after, mutation_before, mutation_after, escapes, attempts…)` | Relatório de uma rodada: defesa antes/depois, fugas, falsos alarmes e escore de mutação. |
| `ArenaReport` | classe | `(problem, alpha, min_error, rounds, defence, stored, settings, _problem)` | Relatório completo da arena, com rodadas, defesa final, casos guardados e saída em Markdown/JSON. |
| `ModelAudit` | classe | `(problem, defence, true_error, rejected, seeds, min_error)` | Saídas de um modelo julgadas por uma defesa contra o solver de referência (fugas e falsos alarmes). |
| `run_arena` | função | `(problem='darcy_2d', checks=None, *, budget=None, rounds=1, alpha=0.01, safety=0.5, min_e…)` | Roda rodadas de red team e blue team num problema e devolve o `ArenaReport`. |
| `audit_model` | função | `(problem, predict, defence=None, *, n_cases=50, seed=0, min_error=0.1, varied=True, **pro…)` | Passa um modelo externo (FNO, MeshGraphNet, PINN...) por uma defesa. |

#### `pinneapple_analysis.validation.arena.red`

Red team: busca fugas (erradas mas aceitas) e falsos alarmes (corretas mas rejeitadas) contra uma defesa, mais o escore de mutação. [sem teste direto]

<small>`pinneapple_analysis/validation/arena/red.py` · 178 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Finding` | classe | `(kind, problem, seed, variation, generator, params, true_error, values, margin, found, me…)` | Uma fuga, um falso alarme ou a melhor tentativa de um deles, com parâmetros e erro verdadeiro. |
| `evasion_search` | função | `(problem, defence, generator, base, *, budget=200, method='de', seed=0, min_error=0.1) ->…)` | Busca a pior saída de um gerador que a defesa ainda aceita. |
| `false_alarm_search` | função | `(problem, defence, *, seeds, budget=100, method='de', search_seed=0) -> list[Finding]` | Para cada semente, busca o caso correto com maior margem de rejeição da defesa. |
| `mutation_score` | função | `(matrix, thresholds, min_error=0.1) -> dict[str, Any]` | Fração dos mutantes com erro relevante que os limiares rejeitam, no total e por falha. |
| `mutation_escapes` | função | `(matrix, thresholds, *, seed, min_error=0.1, per_fault=1) -> list[Finding]` | Lista os mutantes relevantes que passam, como achados reproduzíveis. |

#### `pinneapple_analysis.validation.arena.search`

Minimizadores caixa-preta no cubo unitário para o red team: evolução diferencial (SciPy), CMA-ES e L-BFGS-B por diferenças finitas. [sem teste direto]

<small>`pinneapple_analysis/validation/arena/search.py` · 130 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SearchResult` | classe | `(x, fun, evaluations, method)` | Resultado de uma minimização: ponto, valor, avaliações e método. |
| `cmaes` | função | `(f, dim, budget, seed=0, sigma0=0.3, x0=None) -> SearchResult` | CMA-ES (mu/mu_w, lambda) no cubo [0, 1]^dim. |
| `differential_evolution` | função | `(f, dim, budget, seed=0) -> SearchResult` | Evolução diferencial do SciPy dentro de um orçamento de avaliações. |
| `lbfgs` | função | `(f, dim, budget, seed=0) -> SearchResult` | L-BFGS-B com gradiente por diferenças finitas a partir de alguns pontos iniciais. |
| `minimize` | função | `(f, dim, budget, method='de', seed=0) -> SearchResult` | Minimiza uma função no cubo unitário com o método escolhido (`de`, `cmaes` ou `lbfgs`). |

#### `pinneapple_analysis.validation.boundary`

Checagens de condições de contorno de uma PINN treinada (Dirichlet, Neumann, periodicidade) consultando o modelo em pontos de borda.

<small>`pinneapple_analysis/validation/boundary.py` · 255 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BoundaryCheck` | classe | `(device='cpu') -> None` | Checa condições de contorno consultando o modelo em pontos de borda. |

#### `pinneapple_analysis.validation.conservation`

Checagens de leis de conservação (massa, energia, integral genérica) por integração numérica sobre o domínio.

<small>`pinneapple_analysis/validation/conservation.py` · 375 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ConservationCheck` | classe | `(device='cpu') -> None` | Checa leis de conservação por integração numérica no domínio. |

#### `pinneapple_analysis.validation.core`

Estruturas de dados do validador clássico: `CheckResult` e o `ValidationReport` agregado (não confundir com o de `validate.py`). [sem teste direto]

<small>`pinneapple_analysis/validation/core.py` · 175 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CheckResult` | classe | `(name, passed, value, threshold, description, unit)` | Resultado de uma checagem física: nome, aprovado, valor, limiar e descrição. |
| `ValidationReport` | classe | `(model_name, timestamp, checks)` | Relatório agregado do validador clássico, com resumo em tabela, `to_dict` e `save`. |

#### `pinneapple_analysis.validation.error_map`

Mostra onde um campo está errado: erro ponto a ponto contra a referência, aglomerados de erro alto (hotspots), participação por região e um texto que compara com o número global.

<small>`pinneapple_analysis/validation/error_map.py` · 182 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ErrorMap` | classe | `(error, rel_l2, max_error, rms_error, hotspots, regions, unit, label)` | Resultado de `error_map`: erro por ponto, L2 relativo, erro máximo e RMS, hotspots e regiões. |
| `pointwise_error` | função | `(pred, ref) -> np.ndarray` | Calcula `\|pred - ref\|` por ponto (norma Euclidiana para vetores). |
| `error_hotspots` | função | `(points, err, *, n=5, quantile=0.9, link=None, weights=None) -> list[dict[str, Any]]` | Agrupa os pontos de erro acima de um quantil em aglomerados e devolve os de maior erro. |
| `grow_region` | função | `(points, idx, radius) -> np.ndarray` | Devolve os pontos a menos de um raio de um conjunto de índices. |
| `error_by_region` | função | `(err, regions, *, weights=None) -> list[dict[str, Any]]` | Por região nomeada: participação nos pontos e no erro quadrático e razão do RMS local para o global. |
| `error_map` | função | `(points, pred, ref, *, regions=None, weights=None, n_hotspots=5, quantile=0.9, unit='', l…)` | Calcula erro ponto a ponto, hotspots e participação por região de uma predição contra a referência. |

#### `pinneapple_analysis.validation.fault_injection`

Injeção de falhas: mede com que frequência cada checagem de validação pega defeitos conhecidos de modelo e de dados.

<small>`pinneapple_analysis/validation/fault_injection/__init__.py` · 36 linhas · testes: 20</small>

#### `pinneapple_analysis.validation.fault_injection.__main__`

Ponto de entrada `python -m pinneapple_analysis.validation.fault_injection`: regenera a matriz de detecção (poder das checagens). [sem teste direto]

<small>`pinneapple_analysis/validation/fault_injection/__main__.py` · 59 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `main` | função | `(argv=None) -> int` | Regenera a matriz de detecção pela linha de comando e devolve o código de saída. |

#### `pinneapple_analysis.validation.fault_injection.checks`

As checagens em teste, como objetos `ValidityCheck` de `physics_audit.monitor` que pontuam um `Case`, incluindo checagens de conjunto de dados. [sem teste direto]

<small>`pinneapple_analysis/validation/fault_injection/checks.py` · 289 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `score` | função | `(check, case) -> float` | Valor de uma checagem num caso (`inf` quando não dá para calcular). |
| `flags` | função | `(check, case, tolerance=None) -> bool` | Diz se a checagem rejeita o caso, com tolerância opcional. |
| `check_library` | função | `(problem) -> dict[str, Any]` | Todas as checagens disponíveis para um problema, com tolerância ajustada à mão. |
| `default_checks` | função | `(problem) -> list` | As checagens que um usuário roda hoje (rodada 0 da arena). |
| `dataset_checks` | função | `(problem, case_checks=None) -> list` | Checagens de um conjunto de dados inteiro: quase-duplicatas, vazamento treino/teste e pior valor por amostra. |

#### `pinneapple_analysis.validation.fault_injection.faults`

Biblioteca de falhas: defeitos controlados da saída de um modelo ou de um conjunto de dados, cada um com uma magnitude. [sem teste direto]

<small>`pinneapple_analysis/validation/fault_injection/faults.py` · 248 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Fault` | classe | `(name, target, apply, magnitudes, documented, unit, description)` | Uma falha controlada de modelo ou de dados, com magnitudes e função `apply`. |
| `model_faults` | função | `(problem=None) -> list[Fault]` | Lista as falhas de saída de modelo, opcionalmente filtradas por problema. |
| `data_faults` | função | `(problem=None) -> list[Fault]` | Lista as falhas de conjunto de dados, opcionalmente filtradas por problema. |
| `get_fault` | função | `(name) -> Fault` | Devolve uma falha pelo nome. |

#### `pinneapple_analysis.validation.fault_injection.matrix`

Matriz de detecção: cada checagem contra cada falha e magnitude e contra casos limpos, com taxas de detecção, falso alarme, AUC e magnitude mínima detectável. [sem teste direto]

<small>`pinneapple_analysis/validation/fault_injection/matrix.py` · 282 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DetectionMatrix` | classe | `(problem, level, tolerances, clean, scores, errors, faults, min_rate, trials)` | Valores das checagens em casos limpos e em cada falha e magnitude, com taxas de detecção, falso alarme e AUC. |
| `FaultInjectionReport` | classe | `(problem, model, data, settings)` | Relatório da injeção de falhas (nível modelo e nível dados), com poder das checagens e saída em Markdown. |
| `auc` | função | `(clean, faulty) -> float` | AUC ROC de um escore separando casos com falha de casos limpos (Mann-Whitney). |
| `fault_rng` | função | `(seed, fault, magnitude, case_seed) -> np.random.Generator` | Fluxo aleatório de uma injeção de falha, para reproduzir um teste isolado. |
| `run_fault_injection` | função | `(problem='darcy_2d', *, checks=None, data_checks=None, model_faults=None, data_faults=Non…)` | Monta a matriz de detecção das checagens contra a biblioteca de falhas num problema. |

#### `pinneapple_analysis.validation.fault_injection.problems`

Problemas de benchmark com solver de referência verificado (Darcy 2D, Burgers 1D) como fonte de casos corretos; a base `BenchmarkProblem` tem ganchos que levantam NotImplementedError até a subclasse implementar. [sem teste direto]

<small>`pinneapple_analysis/validation/fault_injection/problems.py` · 274 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Case` | classe | `(problem, inputs, field, params, seed, variation, meta)` | Um par entrada/saída de um problema, reproduzível por semente e variação. |
| `Dataset` | classe | `(train, test)` | Divisão de treino e teste de casos. |
| `BenchmarkProblem` | classe | `(**options)` | Base de problema com solver de referência verificado; subclasses implementam a solução e o espaço de variação. |
| `DarcyProblem` | classe | `()` | Problema de Darcy `-div(a grad u) = f` no quadrado unitário com `u = 0` na borda. |
| `BurgersProblem` | classe | `()` | Problema de Burgers viscoso 1D periódico. |
| `get_problem` | função | `(problem, **options) -> BenchmarkProblem` | Devolve um problema pelo nome (`darcy_2d`, `burgers_1d`) ou o próprio objeto. |

#### `pinneapple_analysis.validation.field_metrics`

Métricas de erro no estilo PDEBench para campos em grade, só NumPy (RMSE, nRMSE, erro de contorno, conservado, por faixa de frequência, Pearson).

<small>`pinneapple_analysis/validation/field_metrics.py` · 154 linhas · testes: 17</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FieldErrorReport` | classe | `(rmse, nrmse, max_error, crmse, brmse, frmse_low, frmse_mid, frmse_high, bands, vrmse, pe…)` | Conjunto de métricas de erro de um campo em grade, serializável com `to_dict`. |
| `spectral_error_shells` | função | `(pred, target, ndim) -> np.ndarray` | Erro quadrático médio por casca de número de onda radial inteiro. |
| `field_error_metrics` | função | `(pred, target, *, ndim, bands=(4, 12), eps=1e-12) -> FieldErrorReport` | Calcula o `FieldErrorReport` de uma predição contra o alvo. |

#### `pinneapple_analysis.validation.gallery`

Final comum dos exemplos de `examples/validation/`: relatório do modelo bom, trust card, saídas lidas por `pinneapple check` e uma falha real pega numa variante quebrada.

<small>`pinneapple_analysis/validation/gallery.py` · 67 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GalleryResult` | classe | `(name, family, physics, good, bad, caught, expected, files)` | Resultado de um exemplo da galeria: relatórios bom e ruim, se a falha foi pega e arquivos gravados. |
| `conclude` | função | `(name, good, bad, *, expect, family, physics, out_dir=None) -> GalleryResult` | Fecha um exemplo da galeria e levanta `AssertionError` se a checagem esperada não pegar a falha. |

#### `pinneapple_analysis.validation.metamorphic`

Testes metamórficos para substitutos: relações que a física garante (simetria, invariância, monotonicidade, superposição, consistência de resolução), testáveis sem solução de referência; a base `Relation.violations` levanta NotImplementedError.

<small>`pinneapple_analysis/validation/metamorphic.py` · 497 linhas · testes: 9 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Relation` | classe | `(name, tol, output, atol, scale, per_sample, kind)` | Base de relação metamórfica; `violations` devolve a violação relativa por amostra e a evidência. |
| `Symmetry` | classe | `(transform_in, transform_out)` | Relação de simetria `f(T x) = S(x, f(x))`. |
| `Similarity` | classe | `(transform_in, dimensionless)` | Relação de semelhança: uma grandeza adimensional não muda sob a transformação da entrada. |
| `Invariance` | classe | `(transform_in)` | Relação de invariância `f(T x) = f(x)`. |
| `Monotonicity` | classe | `(input, direction, step, increase, bounds)` | A saída não diminui (ou não aumenta) quando uma entrada cresce. |
| `Superposition` | classe | `(affine, homogeneity, linear_inputs, combine)` | Relação de superposição (aditividade e, opcionalmente, homogeneidade) de modelos lineares ou afins. |
| `ResolutionConsistency` | classe | `(restrict, quantity)` | Grandezas integrais devem ser iguais quando a entrada muda de resolução. |
| `RelationResult` | classe | `(name, kind, passed, worst, tol, n_tested, counterexample, search, note)` | Resultado de uma relação: aprovação, pior violação, contraexemplo e nota. |
| `MetamorphicReport` | classe | `(results)` | Relatório de todas as relações metamórficas testadas. |
| `box_sampler` | função | `(bounds, fixed=None, log=()) -> Sampler` | Amostrador uniforme (ou log-uniforme) numa caixa de entradas nomeadas. |
| `field_sampler` | função | `(make) -> Sampler` | Amostrador de lotes de campos a partir de um gerador de uma amostra. |
| `reflect_field` | função | `(axis) -> Callable[[np.ndarray], np.ndarray]` | Cria a reflexão de um lote de campos ao longo de um eixo espacial. |
| `rotate_field` | função | `(k=1, axes=(0, 1)) -> Callable[[np.ndarray], np.ndarray]` | Cria a rotação de um lote de campos por quartos de volta. |
| `check_relations` | função | `(model, relations, sampler, *, n=64, seed=0, search='random', max_examples=50) -> Metamor…)` | Roda cada relação em amostras aleatórias e, com `search='hypothesis'`, faz busca por propriedades (cai para busca aleatória se `hypothesis` faltar). |
| `relations_from_spec` | função | `(spec) -> list[Relation]` | Cria relações de um modelo tabular a partir de uma lista declarativa (TOML/JSON). |

#### `pinneapple_analysis.validation.physics_learning`

Verifica se um modelo aprendeu a física ou só uma perda baixa: escolhe pontos informativos, checa resíduo discreto, respostas a intervenções e suficiência das entradas, e compara com as métricas usuais.

<small>`pinneapple_analysis/validation/physics_learning.py` · 499 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Finding` | classe | `(check, passed, value, limit, detail, provenance, points)` | Um achado da verificação: checagem, aprovação, valor, limite e detalhe. |
| `VerificationReport` | classe | `(model, findings, points, reasons, insufficient, trust)` | Relatório de verificação de aprendizado da física, com achados, pontos e confiança. |
| `select_points` | função | `(case, theta, k=48, seed=0, interventions=INTERVENTIONS, pool=3000) -> tuple[np.ndarray, …)` | Escolhe pontos informativos pela resposta da referência a cada intervenção e por cotas perto de interfaces, fonte e bordas. |
| `random_points` | função | `(case, k, seed=0) -> np.ndarray` | Sorteia pontos aleatórios do caso (base de comparação). |
| `fv_residual` | função | `(model, case, th, rows) -> np.ndarray` | Resíduo de volumes finitos dos valores do próprio modelo nas células dadas, relativo à escala de calor. |
| `structural_checks` | função | `(model, case, theta, X, probes=True) -> list[Finding]` | Checagens no campo do modelo que não precisam de referência: resíduo, interfaces, bordas e balanço global de calor. |
| `intervention_checks` | função | `(model, case, theta, X, interventions=INTERVENTIONS) -> list[Finding]` | Compara a resposta do modelo a cada intervenção com a resposta da referência. |
| `sufficiency` | função | `(model, case, theta, X) -> tuple[Finding, list[dict]]` | Aponta parâmetros físicos fora das entradas do modelo que mudam a temperatura de referência acima da tolerância. |
| `verify` | função | `(model, case=None, theta=None, *, k=48, seed=0, points=None, probes=True) -> Verification…)` | Roda todas as checagens do módulo num modelo e devolve o `VerificationReport`. |
| `baselines` | função | `(model, case=None, *, n_theta=12, n_points=400, seed=1) -> dict` | Calcula o que as métricas usuais dizem (erro L2 em pontos aleatórios e perda física média). |
| `detection_study` | função | `(severities=(0.25, 0.5, 1.0, 2.0), budgets=(4, 8, 16, 48), *, seed=0, extra_models=None) …)` | Estudo de detecção do verificador contra as métricas usuais, com modelos defeituosos e de controle. |

#### `pinneapple_analysis.validation.physics_learning_cases`

Caso de verificação para `physics_learning`: seção transversal die/TIM/tampa aquecida na junção, com solver de referência, modelos de controle, modelos defeituosos, substituto POD+GP e PINN.

<small>`pinneapple_analysis/validation/physics_learning_cases.py` · 359 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Section` | classe | `(nx, nz, _cache)` | Grade e solver de referência da seção transversal die/TIM/tampa. |
| `Model` | classe | `(name, fn, inputs, defect, reinit)` | Modelo sob teste `fn(X, theta) -> T`, com entradas, defeito e cópia não treinada opcional. |
| `good_models` | função | `(sec, amplitudes=(0.003,), modes=(1,)) -> list[Model]` | Modelos de controle: a referência e a referência com erros suaves que respeitam contorno e simetria. |
| `defective_models` | função | `(sec, severity=1.0) -> list[Model]` | Modelos precisos pelas métricas usuais mas errados num aspecto físico, com severidade ajustável. |
| `training_distribution` | função | `(n, rng, *, correlated=True) -> list[dict]` | Amostras de parâmetros como as de treino, com resfriamento correlacionado à potência. |
| `surrogate_model` | função | `(sec, n_train=48, *, correlated=True, seed=0) -> Model` | Substituto POD + processo Gaussiano do campo, treinado em soluções de referência. |
| `pinn_model` | função | `(sec, *, hard_sides=False, adam_steps=3000, lbfgs_steps=500, n_collocation=3000, seed=0, …)` | PINN do caso nominal treinada com `train_pinn`, com uma saída por camada e condições de interface e contorno. |

#### `pinneapple_analysis.validation.pinn`

Validação de PINN: resíduo em pontos nunca usados no treino, detectores dos modos de falha conhecidos, comparação com solução exata e um monitor durante o treino; traz problemas prontos (Burgers, Poisson, Kovasznay).

<small>`pinneapple_analysis/validation/pinn.py` · 576 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PINNProblem` | classe | `(name, bounds, residual, boundary, initial, exact, lhs, forcing, time_axis, scale, residu…)` | Descrição de um problema para `validate_pinn`: domínio, resíduo, contorno, condição inicial, solução exata e escalas. |
| `PINNReport` | classe | `(problem, checks, evidence)` | Relatório de validação de PINN com checagens, evidência e modos de falha detectados. |
| `PINNMonitor` | classe | `(problem, *, every=100, warmup=0, patience=2, stop_on=…, n_points=1024, grid=(32, 32), **…)` | Roda as checagens de `validate_pinn` a cada tantas épocas durante o treino, com parada opcional. |
| `d` | função | `(u, X, i, order=1)` | Derivada de ordem dada de `u` em relação a uma coordenada, por autograd. |
| `validate_pinn` | função | `(model, problem, *, n_points=4096, grid=None, train_points=None, tolerances=None, seed=0)…)` | Valida uma PINN treinada num problema e devolve o `PINNReport`. |
| `mlp` | função | `(n_in, n_out, width=32, depth=4)` | Cria uma MLP tanh, a rede usual de PINN. |
| `train_pinn` | função | `(problem, model=None, *, n_collocation=2000, n_condition=400, adam_steps=1500, lbfgs_step…)` | Treina uma PINN com restrições suaves (Adam e depois L-BFGS) e monitor opcional. |
| `cole_hopf` | função | `(x, t, nu, n=2001) -> np.ndarray` | Solução exata de Burgers viscoso com `u(x, 0) = -sin(pi x)` pela transformada de Cole-Hopf. |
| `burgers_1d` | função | `(nu=0.05 / math.pi) -> PINNProblem` | Problema de Burgers 1D (Raissi et al. 2019, viscosidade maior por padrão) com solução exata. |
| `poisson_2d` | função | `(kx=1, ky=2, exact=True) -> PINNProblem` | Problema de Poisson 2D manufaturado no quadrado unitário. |
| `kovasznay` | função | `(re=40.0) -> PINNProblem` | Escoamento de Kovasznay (Navier-Stokes estacionário) com saídas (u, v, p). |

#### `pinneapple_analysis.validation.report_artifact`

O relatório de validação como artefato: esquema JSON versionado, hashes de proveniência, diferença entre dois relatórios, seção para model card do Hugging Face, badge e assinaturas.

<small>`pinneapple_analysis/validation/report_artifact.py` · 341 linhas · testes: 26</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ReportDiff` | classe | `(old, new, lost, gained, changed, added, removed, moved, envelope, decision)` | Diferença entre dois relatórios: checagens perdidas, ganhas, alteradas e decisão. |
| `as_dict` | função | `(report) -> dict[str, Any]` | Converte relatório (objeto, dict ou caminho JSON) em dict. |
| `check_report` | função | `(report) -> list[str]` | Lista os problemas de um relatório contra o esquema e o vocabulário de IDs; vazio se válido. |
| `hash_model` | função | `(model) -> str \| None` | SHA-256 dos parâmetros de um modelo ou do arquivo ONNX; None para um callable simples. |
| `hash_data` | função | `(*objs) -> str \| None` | SHA-256 dos dados usados num relatório. |
| `diff` | função | `(old, new, *, rtol=0.1) -> ReportDiff` | Calcula o que mudou entre dois relatórios de validação. |
| `model_card_section` | função | `(report, *, title='Physics validation') -> str` | Gera um bloco Markdown para model card do Hugging Face. |
| `shields_endpoint` | função | `(report, label='physics-validated') -> dict[str, Any]` | Gera o JSON de um badge de endpoint do shields.io. |
| `badge` | função | `(report, label='physics-validated') -> str` | Gera um badge SVG autocontido com o resultado das checagens. |
| `sign_report` | função | `(report, key) -> dict[str, Any]` | Assina o relatório com `pinneapple_security.signing.sign_json`. |
| `verify_report` | função | `(envelope, key) -> dict[str, Any]` | Verifica a assinatura de um envelope e devolve o relatório, levantando erro se não confere. |

#### `pinneapple_analysis.validation.rollout`

Validação de rollout de substitutos autorregressivos: deriva de invariantes, espectro de energia ao longo do horizonte, tempo de divergência ('trust horizon') e estatísticas de longo prazo.

<small>`pinneapple_analysis/validation/rollout.py` · 326 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RolloutReport` | classe | `(checks, evidence)` | Relatório das checagens de rollout com evidência. |
| `rollout` | função | `(step, u0, n_steps) -> np.ndarray` | Gera a trajetória de um modelo autorregressivo, parando cedo se o estado deixar de ser finito. |
| `invariant_series` | função | `(traj, ndim, invariants=('mass', 'energy')) -> dict[str, np.ndarray]` | Séries temporais dos invariantes (massa, energia) de uma trajetória. |
| `invariant_drift` | função | `(pred, ndim, ref=None, *, dt=1.0, invariants=('mass', 'energy'), window=1) -> dict[str, d…)` | Taxa e deriva máxima de cada invariante, e desvio contra a referência se houver. |
| `spectral_drift` | função | `(pred, ndim, ref=None, *, ref_spectrum=None, periodic=True, window=1, floor=1e-06) -> dic…)` | Viés de energia em alto número de onda a cada horizonte, contra a referência. |
| `divergence_time` | função | `(pred, ref, ndim, *, dt=1.0, err_tol=0.2, spec_tol=0.25, inv_tol=0.05, periodic=True, inv…)` | Primeiro horizonte em que cada critério cruza a tolerância, e o 'trust horizon' (o mais cedo). |
| `long_run_statistics` | função | `(pred, ref, ndim, *, dt=1.0, burn_in=0, periodic=True) -> dict[str, Any]` | Compara medidas invariantes de uma corrida longa do modelo com uma corrida longa de referência. |
| `validate_rollout` | função | `(pred, ndim, ref=None, *, dt=1.0, conserved=(), invariants=('mass', 'energy'), long_ref=N…)` | Roda as checagens `rollout.*` de uma trajetória (aprova / falha / não aplicável). |

#### `pinneapple_analysis.validation.schema`

Vocabulário do relatório de validação num só lugar: status, fatores e níveis de credibilidade e os IDs estáveis de checagens usados por `pp.validate`, trust card e `pinneapple check`.

<small>`pinneapple_analysis/validation/schema.py` · 120 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CheckSpec` | classe | `(id, factor, title, rule, strength, critical, better)` | Especificação de uma checagem: ID, fator de credibilidade, título, regra, força e criticidade. |
| `spec_of` | função | `(check_id) -> CheckSpec` | Devolve a especificação de um ID de checagem; sub-checagens herdam da família. |

#### `pinneapple_analysis.validation.spectral`

Fidelidade espectral de um campo previsto (só NumPy): espectro de energia, viés em alto número de onda, erro de fase, coerência, funções de estrutura e erros de gradiente.

<small>`pinneapple_analysis/validation/spectral.py` · 369 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SpectralComparison` | classe | `(k, E_pred, E_ref, ratio, phase_error, coherence, bands, band_ratio, band_phase_error, ba…)` | Comparação espectral: espectros, razão, erro de fase, coerência e viés por faixa. |
| `StructureFunctions` | classe | `(r, S, S3_signed, flatness)` | Funções de estrutura por defasagem, com flatness. |
| `SpectralFidelityReport` | classe | `(rel_l2, spectrum, structure_pred, structure_ref, gradients, tol, checks)` | Relatório de fidelidade espectral com L2 relativo, espectro, funções de estrutura e gradientes. |
| `energy_spectrum` | função | `(u, ndim=None, *, periodic=True, window='hann', kmax=None) -> tuple[np.ndarray, np.ndarra…)` | Espectro de energia por cascas de um campo. |
| `fit_power_law` | função | `(k, E, kmin, kmax) -> tuple[float, float]` | Ajusta uma lei de potência `E = C k^slope` num intervalo de k. |
| `spectral_comparison` | função | `(pred, ref, ndim, *, periodic=True, window='hann', bands=(1 / 3, 2 / 3), floor=1e-10) -> …)` | Compara os espectros de predição e referência. |
| `high_wavenumber_bias` | função | `(pred, ref, ndim, **kw) -> float` | Viés de energia na faixa alta de número de onda. |
| `structure_functions` | função | `(u, ndim, *, orders=(2, 3), lags=None, periodic=True) -> StructureFunctions` | Calcula funções de estrutura ao longo de cada eixo espacial. |
| `gradient_metrics` | função | `(pred, ref, ndim, spacing=1.0, *, periodic=False, vector_axis=None, walls=(), conductivit…)` | Erros RMS relativos de grandezas baseadas em gradiente. |
| `spectral_fidelity` | função | `(pred, ref, ndim, *, periodic=True, window='hann', bands=(1 / 3, 2 / 3), tol=0.25, spacin…)` | Fidelidade espectral: aprova quando o viés de alta frequência fica dentro da tolerância. |

#### `pinneapple_analysis.validation.symmetry`

Checagens de simetria (reflexão e rotação 2D) de uma PINN treinada.

<small>`pinneapple_analysis/validation/symmetry.py` · 280 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SymmetryCheck` | classe | `(device='cpu') -> None` | Checa propriedades de simetria espacial de um modelo treinado. |

#### `pinneapple_analysis.validation.validate`

`pp.validate`: uma chamada escolhe as checagens que se aplicam ao modelo e ao problema, roda-as e devolve um `ValidationReport` com evidência, credibilidade por fator e decisão de implantação, em níveis `quick`, `standard` e `thorough`.

<small>`pinneapple_analysis/validation/validate.py` · 759 linhas · testes: 23</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ValidationCheck` | classe | `(id, status, summary, value, threshold, evidence)` | Uma checagem do relatório: ID, status, resumo, valor, limiar e evidência. |
| `ValidationReport` | classe | `(model, problem, level, checks, provenance)` | Relatório de `pp.validate` com decisão, veredito, credibilidade e saídas JSON, HTML e trust card. |
| `as_field_model` | função | `(model) -> Callable[[np.ndarray], np.ndarray]` | Converte modelo torch, ONNX ou callable num operador de campo em NumPy. |
| `validate` | função | `(model, problem=None, data=None, reference=None, level='standard', *, query=None, method=…)` | Valida um modelo num problema no nível pedido e devolve o `ValidationReport`. |

#### `pinneapple_analysis.validation.validator`

`PhysicsValidator`: orquestra uma suíte de checagens (conservação, contorno, simetria, personalizada) sobre um modelo treinado, inclusive a partir de um `ProblemSpec`.

<small>`pinneapple_analysis/validation/validator.py` · 357 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsValidator` | classe | `(model, coord_names, domain_bounds, device='cpu') -> None` | Orquestra uma suíte de checagens físicas sobre um modelo treinado. |
| `validate_model` | função | `(model, spec, coord_names=None, domain_bounds=None, device='cpu') -> ValidationReport` | Validação rápida de uma vez a partir de um `ProblemSpec` ou argumentos explícitos. |

#### `pinneapple_analysis.verification`

Reexporta os blocos de verificação física trazidos do produto `physics_verification_engine`; o serviço HTTP sobre eles é `apps/verified_physics`. (A flag 'experimental' vem de uma citação ao guardrail experimental do PhysicsNeMo, não do status.) [sem teste direto do pacote; os submódulos têm testes]

<small>`pinneapple_analysis/verification/__init__.py` · 366 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): experimental</small>

#### `pinneapple_analysis.verification.architecture_critique`

Revisão adversarial de um desenho de arquitetura de IA física por um LLM, contra uma lista fixa de modos de falha, com veredito e severidade por categoria.

<small>`pinneapple_analysis/verification/architecture_critique.py` · 267 linhas · testes: 12</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CritiqueFinding` | classe | `(category, verdict, severity, reasoning)` | Um achado da revisão: categoria, veredito, severidade e justificativa. |
| `AdversarialReviewReport` | classe | `(architecture_description, findings, overall_reasoning, provider, raw_llm_response)` | Relatório da revisão adversarial, com achados, justificativa geral e resposta bruta do LLM. |
| `run_adversarial_review` | função | `(architecture_description, spec=None, *, provider='ollama', model=None, api_key=None, con…)` | Roda a revisão adversarial de uma descrição de arquitetura contra a lista fixa de modos de falha, usando um provedor de LLM. |

#### `pinneapple_analysis.verification.architecture_recommendation`

Recomenda qual família de arquitetura neural do PINNeAPPle usar primeiro, a partir de disponibilidade de dados, solver, generalização por parâmetro/geometria e problema direto ou inverso; nunca afirma acurácia, só adequação ao escopo documentado.

<small>`pinneapple_analysis/verification/architecture_recommendation.py` · 453 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ArchitectureCandidate` | classe | `(name, registry_key, category, when_to_use, strengths, weaknesses, source, implementation…)` | Uma família de arquitetura e quando seu propósito documentado se aplica, com forças, fraquezas e fonte. |
| `ArchitectureRecommendation` | classe | `(recommended, reasoning, decision_path, alternatives, caveats)` | Resultado da recomendação, com caminho de decisão auditável, alternativas e ressalvas. |
| `recommend_architecture` | função | `(*, n_high_fidelity_simulations=0, has_lots_of_data=None, has_analytical_solver=False, ne…)` | Recomenda a família de arquitetura a usar primeiro a partir das características reais do problema. |

#### `pinneapple_analysis.verification.causal_discrepancy`

Dois itens do roadmap: `run_intervention` reexecuta o pipeline com um parâmetro alterado e relata variações percentuais reais; `fit_discrepancy_model` ajusta uma correção de discrepância de forma (Kennedy & O'Hagan 2001) por uma MLP. Cada intervenção retreina uma PINN do zero.

<small>`pinneapple_analysis/verification/causal_discrepancy.py` · 483 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `InterventionRunResult` | classe | `(param_value, final_loss, loss_pct_change, residual_value, residual_pct_change, guardrail…)` | Resultado de uma rodada de intervenção: valor do parâmetro, perda, resíduo, confiabilidade do guardrail e variações percentuais. |
| `InterventionReport` | classe | `(base_description, param_name, baseline_rejected, rejection_reason, query_point, baseline…)` | Relatório da intervenção: rodada de base, rodadas perturbadas e motivo de rejeição se a base não for confiável. |
| `DiscrepancyModel` | classe | `(physics_model, correction_model, in_dim, out_dim, n_train, n_test, variance_explained_te…)` | Modelo `y(x) ~= physics_model(x) + delta(x)` com a correção `delta` dada por uma MLP. |
| `run_intervention` | função | `(base_description, param_name, param_values, *, physical_parameters=None, provider='ollam…)` | Roda o pipeline como base e de novo para cada valor do parâmetro, relatando variações percentuais reais. |
| `fit_discrepancy_model` | função | `(spec, physics_model, reference_x, reference_y, *, hidden_dim=32, n_layers=2, epochs=400,…)` | Ajusta uma MLP de discrepância sobre o resíduo real entre referência e modelo físico. |

#### `pinneapple_analysis.verification.convergence`

Automação de estudo de convergência de malha: extrapolação de Richardson e Grid Convergence Index de Roache com três malhas, e um driver genérico que chama `solve_fn(resolução)`.

<small>`pinneapple_analysis/verification/convergence.py` · 477 linhas · testes: 43</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ConvergenceResult` | classe | `(observed_order, extrapolated_value, gci_fine, gci_coarse, asymptotic_ratio, is_asymptoti…)` | Resultado de extrapolação de Richardson com três malhas: ordem observada, valor extrapolado, GCI e checagem de regime assintótico. |
| `MeshIndependenceStudyResult` | classe | `(resolutions, values, refinement_ratios, refinement_ratio_used, convergence)` | Estudo completo: valores por resolução e o `ConvergenceResult` das três malhas mais finas. |
| `richardson_extrapolate` | função | `(f_coarse, f_medium, f_fine, r, *, p_assumed=None, safety_factor=_DEFAULT_SAFETY_FACTOR, …)` | Extrapola por Richardson uma grandeza em três resoluções e calcula o GCI de Roache (1994). |
| `mesh_independence_study` | função | `(solve_fn, resolutions, quantity_extractor, *, refinement_ratio=None, p_assumed=None, saf…)` | Chama um solver qualquer em resoluções crescentes e extrapola as três mais finas. |

#### `pinneapple_analysis.verification.dimensional_analysis`

Calcula números adimensionais por definições de livro (Reynolds, Prandtl, Péclet, Mach, Froude, Weber, Damköhler, correlação de Nusselt) e classifica o regime de escoamento conforme a geometria indicada.

<small>`pinneapple_analysis/verification/dimensional_analysis.py` · 175 linhas · testes: 27</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DimensionlessNumbers` | classe | `(reynolds, prandtl, peclet_heat, peclet_mass, mach, froude, weber, damkohler, nusselt_cor…)` | Os números adimensionais calculáveis com os dados fornecidos; campos ficam `None` quando faltam entradas. |
| `compute_dimensionless_numbers` | função | `(*, velocity=None, length=None, density=None, dynamic_viscosity=None, kinematic_viscosity…)` | Calcula todos os números adimensionais que o subconjunto de parâmetros permite. |
| `classify_flow_regime` | função | `(re, *, geometry='internal_pipe') -> str` | Classifica o regime de escoamento pelo Reynolds com limiares próprios da geometria informada. |

#### `pinneapple_analysis.verification.evidence_graph`

Grafo de evidências de uma execução (networkx): liga o registro de proveniência ao escore de confiança e às comparações entre solvers, e explica em texto por que confiar.

<small>`pinneapple_analysis/verification/evidence_graph.py` · 235 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_evidence_graph` | função | `(provenance, confidence=None, comparisons=None) -> 'nx.DiGraph'` | Monta o grafo de evidências de uma execução. |
| `evidence_summary` | função | `(g) -> Dict[str, Any]` | Resume o grafo: contagem de evidências por tipo e por apoio/contradição, sem recalcular a confiabilidade. |
| `explain_trust` | função | `(g) -> str` | Gera um texto 'por que confiar' a partir das evidências reais do grafo. |

#### `pinneapple_analysis.verification.forecast_skill`

Skill de previsão contra uma previsão de referência, com intervalos por bootstrap em blocos móveis e médias em lotes que respeitam a correlação serial.

<small>`pinneapple_analysis/verification/forecast_skill.py` · 96 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SkillResult` | classe | `(leads, diff, skill_score, horizon, n_starts, block)` | Resultado de skill por antecedência: diferenças, escore de skill, horizonte e tamanho de bloco. |
| `block_bootstrap_indices` | função | `(n, block, n_boot, rng) -> np.ndarray` | Índices reamostrados por blocos móveis de amostras consecutivas. |
| `block_bootstrap_mean` | função | `(d, block, n_boot=2000, seed=0, level=0.95) -> tuple[float, float, float]` | Média e intervalo por bootstrap em blocos móveis. |
| `block_bootstrap_ratio` | função | `(a, b, block, n_boot=2000, seed=0, level=0.95) -> tuple[float, float, float]` | Escore de skill `1 - mean(a)/mean(b)` e seu intervalo por bootstrap pareado em blocos. |
| `batch_means_ci` | função | `(x, n_batches=8, level=0.95) -> tuple[float, float]` | Média de uma série correlacionada e meia-largura do intervalo por médias em lotes (t de Student). |
| `skill_vs_lead` | função | `(err_model, err_ref, leads, block, *, n_boot=2000, level=0.95, seed=0) -> SkillResult` | Compara duas previsões pareadas por antecedência e calcula o horizonte em que o modelo é melhor com a confiança pedida. |

#### `pinneapple_analysis.verification.geometry_intelligence`

Rotulagem semântica de regiões de uma malha de superfície (entrada, saída, parede) por segmentação geométrica por normais, com desambiguação opcional por LLM, e sugestão de tipos de condição de contorno.

<small>`pinneapple_analysis/verification/geometry_intelligence.py` · 756 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BoundaryLoop` | classe | `(n_vertices, closed, perimeter, is_planar, planarity_rms_deviation, enclosed_area, circul…)` | Um laço de borda fechado de um patch de superfície, com perímetro, planaridade, área e circularidade. |
| `SurfaceRegion` | classe | `(region_id, face_ids, n_faces, area, area_fraction_of_total_surface, area_fraction_of_lar…)` | Um patch conexo da segmentação por normais, com as propriedades geométricas usadas pelo classificador. |
| `GeometryClassification` | classe | `(regions, semantic_labels, boundary_condition_types, ambiguous_region_ids, llm_used, llm_…)` | Resultado da classificação: regiões, rótulos semânticos, tipos de condição de contorno e regiões ambíguas. |
| `classify_geometry` | função | `(mesh, problem_description='', *, provider=None, model=None, api_key=None, angle_toleranc…)` | Classifica cada superfície de borda da malha numa região com sentido físico, com etapa opcional de LLM. |

#### `pinneapple_analysis.verification.geometry_ood_guardrail`

Sinaliza quando uma geometria de entrada está fora da distribuição das geometrias de referência, por distância de Mahalanobis sobre descritores de forma. (A flag 'experimental' vem da citação ao PhysicsNeMo.)

<small>`pinneapple_analysis/verification/geometry_ood_guardrail.py` · 456 linhas · testes: 39 · palavras-chave no código (heurística; o texto acima diz o status real): experimental</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GeometryOODResult` | classe | `(score, mahalanobis_sq, dof, status, n_reference, feature_values, feature_z_scores, top_d…)` | Saída de uma consulta: escore, Mahalanobis ao quadrado, status e descritores que mais desviam. |
| `GeometryOODGuardrail` | classe | `(*, warn_p=0.05, reject_p=0.01)` | Ajusta uma distribuição Gaussiana por descritor sobre geometrias de referência e pontua geometrias novas. |
| `extract_geometry_features` | função | `(mesh) -> Dict[str, float]` | Extrai de uma malha os descritores de forma listados em `FEATURE_NAMES`. |
| `default_reference_meshes` | função | `() -> List[MeshData]` | Catálogo genérico de malhas de referência (caixa, cilindro, canal) gerado por `pinneapple_design`. |

#### `pinneapple_analysis.verification.knowledge_graph`

Grafo de conhecimento físico (networkx) construído na hora a partir do registro de presets e dos arquivos de teste de validação (lidos por AST): fenômeno, equação, parâmetros, preset, referência, método de verificação. (As flags vêm de palavras no texto, não do status.)

<small>`pinneapple_analysis/verification/knowledge_graph.py` · 644 linhas · testes: 14 · palavras-chave no código (heurística; o texto acima diz o status real): experimental; placeholder/stub citado no docstring</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_knowledge_graph` | função | `() -> 'nx.MultiDiGraph'` | Constrói o grafo de conhecimento a partir do registro de presets e dos testes de validação atuais. |
| `find_presets_for_phenomenon` | função | `(g, phenomenon) -> List[str]` | Lista os presets cuja equação pertence a um fenômeno. |
| `find_verified_equations` | função | `(g) -> List[str]` | Lista os tipos de EDP com ao menos uma aresta de verificação confirmada por AST. |
| `explain_preset` | função | `(g, preset_name) -> Dict[str, Any]` | Resumo estruturado de um preset (equação, fenômeno, parâmetros, citações, verificação) com texto pronto para contexto de LLM. |
| `graph_stats` | função | `(g) -> Dict[str, Any]` | Contagens de auditoria do grafo, incluindo frações de presets com citação e com verificação. |

#### `pinneapple_analysis.verification.mesh_intelligence`

Avalia se uma malha triangular de superfície serve para a física pedida: razão de aspecto e assimetria por triângulo, estimativa da altura da primeira célula por y+ e relatório do que falhou; ortogonalidade é declarada não calculável em malha de superfície.

<small>`pinneapple_analysis/verification/mesh_intelligence.py` · 475 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TriangleQualityStats` | classe | `(aspect_ratio_convention, aspect_ratio, aspect_ratio_mean, aspect_ratio_max, worst_aspect…)` | Métricas de qualidade por triângulo e suas estatísticas resumo. |
| `OrthogonalityResult` | classe | `(computable, reason)` | Indica que a ortogonalidade não é calculável numa malha de superfície e por quê. |
| `YPlusEstimate` | classe | `(computable, reason, target_yplus, reynolds_length, skin_friction_coefficient, wall_shear…)` | Estimativa da altura da primeira célula para um y+ alvo, com campos `None` quando faltam entradas. |
| `MeshQualityReport` | classe | `(n_faces, n_vertices, triangle_quality, orthogonality, yplus, is_good_enough, failures, t…)` | Relatório de qualidade da malha: métricas, estimativa de y+, se é boa o bastante e o que falhou. |
| `estimate_required_first_cell_height` | função | `(*, reynolds_length, velocity, kinematic_viscosity, target_yplus=1.0) -> YPlusEstimate` | Estima a altura da primeira célula para atingir o y+ alvo numa camada limite turbulenta de placa plana. |
| `assess_mesh_quality` | função | `(mesh, *, aspect_ratio_threshold=100.0, skewness_threshold=0.95, max_bad_element_fraction…)` | Avalia se a malha é boa o bastante e devolve um relatório estruturado. |

#### `pinneapple_analysis.verification.nozzle_crosscheck`

Triagem de bocal de foguete por dois métodos sem código comum: uma PINN do preset `rocket_nozzle_cfd` e o bocal isentrópico ideal 1D, com a discordância entre eles.

<small>`pinneapple_analysis/verification/nozzle_crosscheck.py` · 146 linhas · testes: 3</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `scaled_field_head` | função | `(net, scale, offset)` | Embrulha uma rede para que `y = net(x) * scale + offset` por campo de saída. |
| `train_nozzle_pinn` | função | `(*, throat_radius, exit_radius, nozzle_length, p_chamber, T_chamber, gamma, R_gas, epochs…)` | Treina a PINN do preset `rocket_nozzle_cfd`, roda o guardrail e devolve as médias no plano de saída. |
| `screen_nozzle` | função | `(*, throat_radius, exit_radius, p_chamber, T_chamber, gamma=1.2, R_gas=287.0, nozzle_leng…)` | Estimativa do bocal ideal e, com `run_pinn`, a estimativa da PINN e a discordância entre as duas. |

#### `pinneapple_analysis.verification.physics_confidence_score`

`PhysicsConfidenceScore`: escore agregado e transparente (média simples) só das checagens já executadas que o chamador passa, sempre com a fração de cobertura.

<small>`pinneapple_analysis/verification/physics_confidence_score.py` · 777 linhas · testes: 36</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CalibrationSummary` | classe | `(ece, coverage, target_coverage, sharpness)` | Números de calibração de UQ já calculados (ECE, cobertura, nitidez). |
| `ConfidenceComponent` | classe | `(name, score, source_summary)` | Uma checagem já executada convertida em escore de 0 a 1 por fórmula documentada. |
| `PhysicsConfidenceScore` | classe | `(overall_score, components, coverage)` | Escore agregado (média simples dos componentes) com a cobertura das checagens executadas. |
| `compute_physics_confidence` | função | `(*, guardrail_report=None, convergence_result=None, calibration_metrics=None, benchmark_c…)` | Agrega as checagens fornecidas num `PhysicsConfidenceScore`. |
| `score_against_reference` | função | `(name, x, y_ref, preds, *, x_vars, y_vars, sigma=None, convergence=None, guardrail=None, …)` | Pontua as predições de vários modelos contra um conjunto de referência separado. |

#### `pinneapple_analysis.verification.pipeline`

Pipeline descrição do problema -> análise dimensional -> preset rascunhado por LLM -> treino da PINN -> guardrail -> seleção do melhor candidato; bloqueante, o treino acontece dentro da chamada.

<small>`pinneapple_analysis/verification/pipeline.py` · 166 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CandidateResult` | classe | `(architecture, final_loss, guardrail_trustworthy, guardrail_checks, residual_value, model)` | Um candidato treinado: arquitetura, perda final, resultado do guardrail, resíduo e modelo. |
| `AnalysisResult` | classe | `(provenance, best_candidate, all_candidates, rejected, rejection_reason)` | Resultado do pipeline: proveniência, melhor candidato, todos os candidatos e rejeição. |
| `rank_key` | função | `(candidate) -> tuple` | Chave de ordenação: não confiável primeiro descartado, depois resíduo recalculado e perda de treino. |
| `guardrail_checks_as_dicts` | função | `(report) -> list[dict[str, Any]]` | Converte as checagens de um `GuardrailReport` em dicts simples. |
| `analyze` | função | `(description, *, physical_parameters=None, flow_geometry='internal_pipe', provider='ollam…)` | Roda o pipeline completo para uma descrição de problema. |

#### `pinneapple_analysis.verification.provenance`

Registro estruturado de proveniência de uma execução (descrição, adimensionais, preset rascunhado, arquitetura etc.), com gravar e carregar em JSON. (A flag 'placeholder' vem do texto 'not a placeholder'.)

<small>`pinneapple_analysis/verification/provenance.py` · 67 linhas · testes: 8 · palavras-chave no código (heurística; o texto acima diz o status real): placeholder/stub citado no docstring</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ProvenanceRecord` | classe | `(run_id, created_unix, problem_description, dimensionless_numbers, flow_regime, drafted_p…)` | Registro de proveniência de uma execução, com `to_dict`, `save` e `load`. |

#### `pinneapple_analysis.verification.solver_orchestration`

Orquestração entre famílias de solvers do PINNeAPPle: lista o que está disponível na máquina, recomenda uma família para um `ProblemSpec` e compara dois resultados nos mesmos pontos.

<small>`pinneapple_analysis/verification/solver_orchestration.py` · 729 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SolverRecommendation` | classe | `(recommended_family, reasoning, matched_rule, fallback_families, caveats)` | Resultado de `select_solver_family`: família recomendada, regra usada, alternativas e ressalvas. |
| `ComparisonReport` | classe | `(label_a, label_b, n_points, max_abs_diff, mean_abs_diff, rmse, relative_rmse, worst_inde…)` | Comparação numérica entre dois resultados nos mesmos pontos (diferença máxima, média, RMSE). |
| `list_available_solver_families` | função | `() -> Dict[str, dict]` | Verifica na hora quais famílias de solver este processo consegue usar na máquina. |
| `select_solver_family` | função | `(spec, dimensionless_numbers=None, *, catalog=None) -> SolverRecommendation` | Recomenda uma família de solver disponível para o `ProblemSpec` e explica por quê. |
| `compare_solvers` | função | `(spec, model_or_result_a, model_or_result_b, *, query_points=None, labels=('solver_a', 's…)` | Compara dois resultados do mesmo `ProblemSpec` nos mesmos pontos. |

#### `pinneapple_analysis.verification.tool_recommendation`

Recomenda ferramentas externas (CFD, FEM, multifísica, malha, visualização, UQ) cujo escopo documentado cobre o problema, com catálogo inicial de ferramentas, empresas e orientação comprar-ou-construir.

<small>`pinneapple_analysis/verification/tool_recommendation.py` · 601 linhas · testes: 18</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ExternalTool` | classe | `(name, category, license, typical_cost_tier, pde_kinds, dims, strengths, caveats, source)` | Uma ferramenta externa nomeada e seu escopo documentado. |
| `ToolRecommendation` | classe | `(recommended, reasoning, alternatives, caveats)` | Resultado de `recommend_tools`: recomendadas, justificativa, alternativas e ressalvas. |
| `Player` | classe | `(name, focus_areas, tool_keys, notes)` | Uma empresa ou ecossistema e suas áreas de foco, ligado às ferramentas do catálogo. |
| `recommend_tools` | função | `(spec, dimensionless_numbers=None, *, catalog=None, category=None) -> ToolRecommendation` | Recomenda ferramentas externas cujo escopo documentado cobre o `ProblemSpec`. |
| `get_tools_by_category` | função | `(category, *, catalog=None) -> List[str]` | Lista as ferramentas do catálogo numa categoria (ex.: malha, visualização, UQ). |
| `get_player_for_tool` | função | `(tool_key, *, player_catalog=None) -> Optional[str]` | Diz qual empresa faz a ferramenta, ou `None` para projetos comunitários. |
| `buy_vs_build_recommendation` | função | `(need, *, guidance=None) -> Optional[str]` | Busca a orientação qualitativa comprar-ou-construir para uma necessidade; `None` se não coberta. |
| `list_buy_vs_build_needs` | função | `(*, guidance=None) -> List[str]` | Lista as necessidades cobertas pela tabela comprar-ou-construir. |

#### `pinneapple_analysis.verification.uncertainty_inverse`

Liga as classes reais de UQ e de problemas inversos ao pipeline de verificação: relatório de incerteza por tipo e estimativa de parâmetros; o inverso só suporta `heat_equation_steady` com o parâmetro `k`, e outros casos levantam NotImplementedError.

<small>`pinneapple_analysis/verification/uncertainty_inverse.py` · 752 linhas · testes: 13 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `InverseParamAdapter` | classe | `(base, unknown_param_names, initial_guess=None) -> None` | Dá a qualquer modelo um `nn.ParameterDict` `inverse_params`, como `InverseProblemSolver` exige. |
| `UncertaintyReport` | classe | `(epistemic_std, aleatoric_std, parameter_std, total_std, method, n_samples, metadata)` | Relatório de incerteza separado por tipo (epistêmica, aleatória, de parâmetro, total). |
| `InverseProblemResult` | classe | `(estimated_params, parameter_uncertainty, converged, final_misfit, method, n_iters, loss_…)` | Resultado da estimativa de parâmetros: valores, incerteza, convergência, desajuste e histórico. |
| `quantify_uncertainty` | função | `(spec, model, *, method='ensemble', n_samples=20, models=None, x_eval=None, n_eval=200, s…)` | Quantifica a incerteza preditiva de um modelo treinado com as classes reais de `pinneapple_analysis.uncertainty`. |
| `solve_inverse_problem` | função | `(spec, observed_data, unknown_param_names, *, model=None, initial_guess=None, n_outer=5, …)` | Estima parâmetros a partir de dados esparsos com `InverseProblemSolver` num esquema alternado campo/parâmetro. |

### Exemplos de uso reais do repositório

**Correntes parasitas: varredura sintética e treino da PINN** — `examples/inspection/01_eddy_current_pinn.py`

```python
data = generate_eddy_current_synthetic(
    nr=24, nz=40, scan_positions=np.linspace(-0.01, 0.01, 9),
)
...
out = train_eddy_current(
    synthetic_data=data,
    hidden=[64, 64, 64],
    epochs=60,
    lr=1e-3,
    batch_size=32,
    log_dir="examples/_out/inspection_eddy_current",
    run_name="eddy_current_demo",
)
```

**Problema inverso a partir de um preset** — `examples/quickstart/05_inverse_problem.py`

```python
import pinneapple_analysis.inverse_problems as pinv

preset = pinv.get_preset("fin_h")
built = preset.build()
problem = pinv.InverseProblem(
    model=built["model"], observations=built["observations"],
    parameters=built["parameters"], validation=built["validation"], name="fin_h",
)
result = pinv.solve_inverse(problem, "least_squares")
comparison = pinv.compare_methods(problem, methods=["least_squares", "bayesian"])
print("best:", comparison["best"])
```

**pp.validate num FNO exportado em ONNX (Darcy 2D)** — `examples/validation/02_fno_darcy_onnx.py`

```python
data = {"train": {"a": a[:n_train], "u": u[:n_train]}, "test": {"a": a[n_train:], "u": u[n_train:]}}
good = pp.validate(str(path), "darcy_2d", data=data, level="standard")
shifted, _ = darcy_set(8, n_grid, seed=7, high=100.0, low=1.0)
bad = pp.validate(str(path), "darcy_2d", data={"train": data["train"]}, query=shifted, level="standard")
return conclude("02_fno_darcy_onnx", good, bad, expect=CATCHES, family=FAMILY, physics=PHYSICS, out_dir=out_dir)
```

**Números adimensionais conferidos com cálculo à mão** — `tests/pinneapple_analysis/test_verification_dimensional_analysis.py`

```python
def test_reynolds_number_water_in_pipe_matches_hand_calculation():
    # rho=998 kg/m3, U=2 m/s, L=0.05 m (pipe diameter), mu=1.002e-3 Pa.s
    # Re = rho*U*L/mu = 998*2*0.05/1.002e-3 = 99600.8 (hand-computed)
    dn = compute_dimensionless_numbers(
        velocity=2.0, length=0.05, density=998.0, dynamic_viscosity=1.002e-3,
    )
    assert dn.reynolds == pytest.approx(99600.8, rel=1e-3)
```

**Benchmark com orçamento igual: PINN, FNO e DeepONet em Burgers** — `examples/benchmark/01_equal_budget.py`

```python
import pinneapple as pp

full = "--full" in sys.argv
res = pp.benchmark.run(
    problems=["burgers"],
    methods=["pinn", "fno", "deeponet"],
    budget=pp.ComputeBudget(wall_s=300),          # per training run, one core; or flops=..., steps=...
    budget_levels=[1 / 9, 1 / 3, 1.0],            # error-versus-budget curve
    seeds=3, n_trials=4, search_fraction=1 / 3,   # the same search effort for every method
    equal_flops=True,                             # the extra equal-FLOPs arm
    n_workers=4, threads_per_run=1,
    smoke=not full,
    out_dir="results/benchmark_burgers" + ("" if full else "_smoke"),
)
print(res.table().to_string())
```

**Grafo de prova do tetrâmero PT** — `examples/proof_graph/pt_chain_no_go.py`

```python
from pinneapple_analysis.proof.pt_chains import build_pt_chain_graph

g = build_pt_chain_graph(max_deg=4, n_numeric=300, search_iters=40)
status = g.run()
print(g.report())
print("\nMermaid:\n" + g.to_mermaid())
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 103 de 159.
- Funções de teste em arquivos que importam algum módulo do pacote: 939.

### Limitações conhecidas

- `inspection`: todos os sinais são sintéticos e ilustrativos; as simplificações (EMAT escalar, Klein-Gordon no lugar de Rayleigh-Lamb, varredura axissimétrica em correntes parasitas, MFL sem termo cruzado, Helmholtz como regularizador no phased array) estão declaradas nos docstrings. Os submódulos não têm teste direto; só o pacote é importado por `tests/test_inspection.py`.
- `inverse_problems`: convivem duas camadas, o kit PyTorch antigo (`InverseProblemSolver`, misfits, regularizadores, EKI) e o núcleo novo em NumPy/SciPy (`forward`, `estimation`, `identifiability`, `calibration`, `studio`, `presets`); vários módulos do núcleo novo não têm teste direto.
- `inverse_problems.heat_transfer`: o caso 3D não é treinado sob demanda, é lido de resultados gravados; o docstring de `obs_operator` cita um `BoundaryObsOperator` inexistente.
- `validation`: há duas classes `ValidationReport` diferentes (`core.ValidationReport` do validador clássico e `validate.ValidationReport` de `pp.validate`).
- `validation.arena` e `validation.fault_injection`: só os pacotes são importados por testes; os submódulos (blue, red, loop, matrix, faults etc.) não têm teste direto, e só Darcy 2D e Burgers 1D vêm prontos.
- `validation.metamorphic`: a busca por propriedades usa `hypothesis` se instalado; senão cai para busca aleatória.
- `verification`: vários módulos (`pipeline`, `causal_discrepancy`, `architecture_critique`, `geometry_intelligence`) dependem de um provedor de LLM (`ollama` por padrão, servidor local); `evidence_graph` e `knowledge_graph` exigem `networkx`.
- `verification.uncertainty_inverse`: `solve_inverse_problem` só suporta `heat_equation_steady` com o parâmetro `k`; o resto levanta NotImplementedError.
- `verification.nozzle_crosscheck`: o README de `apps/nozzle_screen` avisa que a PINN e o bocal ideal discordam hoje, porque o resíduo do preset está em unidades SI brutas; a metade PINN é uma demonstração.
- `benchmark.builtin.airfoil` precisa do OpenFOAM para `prepare` (`/usr/share/openfoam/etc/bashrc` ou `FOAM_BASHRC`); `airfoil` e `inverse_heat` só são registrados se suas dependências opcionais importarem.
- Vários módulos do benchmark (`advisor`, `api`, `runner`, `checks`, `config`, `registry`, `airfoil*`, `inverse_heat*`) não têm teste que os importe diretamente; a cobertura vem pelo `__init__` e pela CLI.
- Os ganchos das classes base `Problem`, `Method`, `FieldOperatorProblem` e dos candidatos de surrogate levantam NotImplementedError até serem implementados; `SurrogateModel.export` só funciona para os candidatos PyTorch (mlp, fno, deeponet, pinn).
- `retrieval` exige o extra `retrieval` (sentence-transformers, modelo local padrão `all-MiniLM-L6-v2`) e, com `include_literature=True`, rede para o arXiv; o `__init__` de `retrieval` não tem teste direto.
- `uncertainty.reachability` é marcado RESEARCH ONLY: emite ResearchOnlyWarning e é recusado com `PINNEAPPLE_COMMERCIAL_MODE=1`; modela espaço livre, sem gravidade.
- `trust.trust_report` é marcado experimental e tem API em português (`decisao`, `motivos`, `checagem_*`); as regras de decisão são escolhas de projeto, não normas.
- `proof.pt_chains` não prova o teorema por completo: o status fica ARGUED, e cadeias com N >= 5 seguem como conjectura apoiada só por busca de contraexemplos.
- A garantia conformal só vale com calibração suficiente (`n >= ceil(1/alpha) - 1`); com menos pontos o quantil é cortado no maior escore e a garantia não se aplica.

## 2. `pinneapple_check`

### Objetivo

`pinneapple_check` é um linter de física para CI. Ele falha um pull request quando um resultado de simulação ou um modelo de Physics-AI viola a física: conservação, unidades, condições de contorno e iniciais, resíduo da EDP, limites, uso fora da distribuição de treino, regressão contra uma referência gravada e integridade dos artefatos. Cada violação tem um id de regra estável (`PC101`...), severidade, explicação e dica de correção.

O uso é pela linha de comando `pinneapple check [paths]` (ou `python -m pinneapple_check`), pela GitHub Action (`action.yml` na raiz do repositório) ou por Python com `pinneapple_check.run`. Funciona sem configuração (descobre casos OpenFOAM e arquivos de campo e infere a física pelos nomes) ou com `pinneapple-check.toml`. A saída sai em texto, JSON, SARIF, JUnit, anotações do GitHub ou resumo Markdown, e uma baseline permite adotar o checador num repositório que já tem violações.

O pacote importa só NumPy na partida e reaproveita módulos de outros pacotes sem executar seus `__init__` (`_reuse`): leitores OpenFOAM e de unidades de `pinneapple_data`, o envelope de treino de `pinneapple_physics.physics_audit`, métricas de `pinneapple_analysis`, o catálogo de métodos (`pinneapple_catalog`, regra PC901), `pinneapple_security` para hashes, assinaturas e varredura de checkpoints, e os estágios do trust card de `pinneapple_lab`. O relatório vira evidência para o trust card e para o mapa de aplicabilidade de `pinneapple_veriphysics`. O mesmo `cli.main` serve de despachante do comando `pinneapple` (doctor, recommend, card, new, design, benchmark).

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 23 | 4.547 | 22 | 71 | 144 (12 arquivos) | 11 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `__main__` | Entrada `python -m pinneapple_check`. | 1 | 0 | 0 |
| `_reuse` | Importa módulos isolados de outros pacotes PINNeAPPle sem seus `__init__`. | 1 | 14 | 0 |
| `balances` | Balanços globais de massa, calor e força em grades tensoriais. | 1 | 3 | 0 |
| `baseline` | Baseline para falhar só em violações novas. | 1 | 3 | 0 |
| `checks` | As checagens de resultados, modelos, datasets e integridade. | 6 | 15 | 22 |
| `cli` | Linha de comando `pinneapple check` e despachante `pinneapple`. | 1 | 2 | 54 |
| `config` | Leitura de `pinneapple-check.toml` e descoberta de alvos. | 1 | 10 | 5 |
| `engine` | Orquestra configuração, alvos e resultados num `Report`. | 1 | 1 | 5 |
| `evidence` | Trust card e checagens de salvaguarda para o veriphysics. | 1 | 2 | 22 |
| `fields` | Leitura de resultados em `FieldData`. | 1 | 12 | 16 |
| `fv` | Operadores e resíduos de volumes finitos em malhas OpenFOAM. | 1 | 4 | 16 |
| `models` | Carga segura de modelos como funções. | 1 | 5 | 22 |
| `numerics` | Derivadas em grade, nuvem de pontos e modelo. | 1 | 5 | 5 |
| `physics` | Equações governantes e tabela de parâmetros. | 1 | 4 | 5 |
| `report` | Formatos de saída (texto, JSON, SARIF, JUnit, GitHub, resumo). | 1 | 7 | 22 |
| `result` | `Outcome` e `Report`. | 1 | 3 | 0 |
| `rules` | Tabela de regras PC com seleção por prefixo. | 1 | 3 | 0 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_check`

Ponto de entrada do `pinneapple check`, um linter de física para CI: reexporta `run` e as classes de relatório; importa só NumPy na partida.

<small>`pinneapple_check/__init__.py` · 39 linhas · testes: 74</small>

#### `pinneapple_check.__main__`

Permite `python -m pinneapple_check [paths]`, equivalente a `pinneapple check [paths]`. [sem teste direto]

<small>`pinneapple_check/__main__.py` · 8 linhas · **sem teste direto**</small>

#### `pinneapple_check._reuse`

Carrega módulos isolados de outros pacotes (leitores OpenFOAM e de unidades do `pinneapple_data`, envelope do `physics_audit`, métricas do `pinneapple_analysis`, catálogo, estágios do `pinneapple_lab`, checagens Darcy/Maxwell) sem executar o `__init__` desses pacotes. [sem teste direto]

<small>`pinneapple_check/_reuse.py` · 98 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `leaf` | função | `(dotted) -> types.ModuleType` | Importa um módulo pelo nome pontilhado sem executar o `__init__` dos pacotes pais. |
| `foam` | função | `()` | Leitor OpenFOAM de `pinneapple_data.cae.foam`. |
| `cae_io` | função | `()` | Módulo de E/S CAE de `pinneapple_data.cae.io`. |
| `cae_geometry` | função | `()` | Geometria de malha de `pinneapple_data.cae.geometry`. |
| `units` | função | `()` | Unidades físicas de `pinneapple_data.physical_units`. |
| `envelope_cls` | função | `()` | Classe `Envelope` de `pinneapple_physics.physics_audit.assumptions`. |
| `field_metrics` | função | `()` | Métricas de campo de `pinneapple_analysis.validation.field_metrics`. |
| `spectral` | função | `()` | Métricas espectrais de `pinneapple_analysis.validation.spectral`. |
| `metamorphic` | função | `()` | Testes metamórficos de `pinneapple_analysis.validation.metamorphic`. |
| `dataset_audit` | função | `()` | Auditoria de dataset de `pinneapple_data.audit`. |
| `method_catalog` | função | `()` | Catálogo de métodos de `pinneapple_catalog.methods`. |
| `trust_stages` | função | `()` | Estágios do trust card (`STAGES`) de `pinneapple_lab.curation`. |
| `darcy_checks` | função | `()` | Checagens de Darcy de `pinneapple_physics.darcy_checks`. |
| `maxwell_checks` | função | `()` | Checagens de Maxwell de `pinneapple_physics.maxwell_checks`. |

#### `pinneapple_check.balances`

Balanços globais na fronteira de uma caixa em grade tensorial: fluxo líquido de volume, fluxo líquido de calor e força resultante, cada um com o fluxo absoluto total para normalizar. [sem teste direto]

<small>`pinneapple_check/balances.py` · 85 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `grid_mass` | função | `(coords, names, vel) -> dict` | Fluxo líquido de volume pela fronteira da grade e o fluxo absoluto total. |
| `grid_heat` | função | `(coords, names, T, k, q) -> dict` | Fluxo de calor que entra no domínio (k dT/dn) mais a fonte. |
| `grid_force` | função | `(coords, names, disp, E, nu, f) -> dict` | Resultante das trações de fronteira σ·n mais a força de corpo (deformação plana em 2D). |

#### `pinneapple_check.baseline`

Linha de base: adota o checador num repositório que já tem violações e falha só nas novas. [sem teste direto]

<small>`pinneapple_check/baseline.py` · 34 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `load_baseline` | função | `(path) -> set[str]` | Lê o arquivo de baseline como conjunto de impressões de violações. |
| `write_baseline` | função | `(report, path) -> int` | Grava as violações atuais do relatório como baseline e devolve quantas. |
| `apply_baseline` | função | `(report, path) -> None` | Marca no relatório as violações que já estão na baseline. |

#### `pinneapple_check.checks`

Pacote das checagens executadas por `run`: resultados, modelos, conjuntos de dados e integridade de artefatos. [sem teste direto]

<small>`pinneapple_check/checks/__init__.py` · 9 linhas · **sem teste direto**</small>

#### `pinneapple_check.checks.common`

Checagens comuns a resultados e modelos: valores finitos (PC403), limites declarados (PC401), admissibilidade física (PC402), unidades e referências. [sem teste direto]

<small>`pinneapple_check/checks/common.py` · 227 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Ctx` | classe | `(cfg, target, enabled, outcomes)` | Contexto de um alvo onde os resultados são coletados, com regras ativas, tolerâncias e localização. |
| `fmt` | função | `(x) -> str` | Formata um número para mensagens. |
| `admissible` | função | `(name, unit) -> tuple[float \| None, float \| None, str] \| None` | Limites físicos rígidos (baixo, alto, descrição) inferidos do nome e da unidade de um campo. |
| `check_values` | função | `(ctx, name, v, unit, loc, where=None) -> None` | Aplica PC403 (não finito), PC401 (limites declarados) e PC402 (admissibilidade) a um campo. |
| `check_units` | função | `(ctx, name, file_unit, declared, values, loc, used) -> None` | Confere a unidade do arquivo contra a declarada e a plausibilidade das magnitudes (regras PC5xx). |
| `rel_l2` | função | `(a, b) -> float` | Erro L2 relativo entre dois arrays. |

#### `pinneapple_check.checks.datasets`

Checagens de um conjunto de treino (`[[dataset]]`) usando a auditoria de `pinneapple_data.audit`: regras PC451 (física/restrições), PC452 (diversidade/duplicatas) e PC453 (vazamento treino/teste). [sem teste direto]

<small>`pinneapple_check/checks/datasets.py` · 53 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check_dataset` | função | `(ctx, t) -> None` | Roda a auditoria física de um conjunto de treino e emite PC451-PC453. |

#### `pinneapple_check.checks.integrity`

Integridade de artefatos com `pinneapple_security`: manifestos SHA-256 (PC801, PC804), assinaturas Ed25519 destacadas (PC802) e pins de hash por alvo. [sem teste direto]

<small>`pinneapple_check/checks/integrity.py` · 112 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check_pin` | função | `(ctx, path, expected) -> None` | Confere o hash SHA-256 de um arquivo contra o pin declarado. |
| `check_integrity` | função | `(ctx, cfg) -> None` | Confere manifestos e assinaturas declarados na configuração. |

#### `pinneapple_check.checks.models`

Checagens de um modelo: carga segura, status do método no catálogo, entradas dentro da faixa de treino, física das predições em pontos amostrados (resíduo, divergência, balanços), condições de contorno/iniciais e regressão. [sem teste direto]

<small>`pinneapple_check/checks/models.py` · 397 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `read_table` | função | `(path) -> tuple[dict[str, np.ndarray], dict[str, int]]` | Lê colunas de um CSV ou arrays de um NPZ, com o número de linha de cada registro do CSV. |
| `check_model` | função | `(ctx, t) -> LoadedModel \| None` | Carrega e checa um alvo de modelo, devolvendo o `LoadedModel` ou None. |

#### `pinneapple_check.checks.results`

Checagens de um resultado gravado (caso OpenFOAM, grade, nuvem de pontos, tabela de QoI): valores, unidades, contorno, resíduos da EDP, balanços e regressão contra referência.

<small>`pinneapple_check/checks/results.py` · 707 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `evaluate_expr` | função | `(value, coords, n) -> np.ndarray` | Avalia um valor declarado: número, vetor ou expressão das coordenadas (ex.: 'sin(pi*x)'). |
| `check_result` | função | `(ctx, t) -> FieldData \| None` | Carrega e checa um alvo de resultado, devolvendo o `FieldData` ou None. |
| `report_equations` | função | `(ctx, eqs, why, where, rule_div, rule_res, tol_div, tol_res, loc, subject_prefix='') -> N…)` | Emite os resultados de divergência e resíduo de um conjunto de equações. |
| `check_balance` | função | `(ctx, rule, what, b, tol, loc, unit='', closed_scale=None, subject=None) -> None` | Emite o resultado de um balanço global contra a tolerância. |

#### `pinneapple_check.cli`

Linha de comando `pinneapple check [paths]` (formatos, seleção de regras, baseline, `--explain`) e o despachante `pinneapple <comando>` (doctor/recommend/card/new, check, design, benchmark).

<small>`pinneapple_check/cli.py` · 207 linhas · testes: 54</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check_main` | função | `(argv=None) -> int` | Executa `pinneapple check` com os argumentos e devolve o código de saída. |
| `main` | função | `(argv=None) -> int` | Despacha `pinneapple <comando>`: doctor/recommend/card/new, check, e design/benchmark quando instalados. |

#### `pinneapple_check.config`

Configuração por `pinneapple-check.toml` (ou `[tool.pinneapple.check]` no `pyproject.toml`): alvos de resultado, modelo, dataset e integridade, seleção de regras, tolerâncias e descoberta automática sem configuração.

<small>`pinneapple_check/config.py` · 534 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ConfigError` | classe | `()` | Erro de configuração inutilizável (código de saída 2). |
| `Reference` | classe | `(path, fields, rel_l2, qoi)` | Referência gravada para regressão: caminho, campos, tolerância L2 relativa e QoIs. |
| `ResultTarget` | classe | `(kind, time, fields, flux, conserved, envelope)` | Alvo de resultado: tipo, tempo, campos, fluxos, quantidades conservadas e envelope. |
| `ModelTarget` | classe | `(architecture, model_config, hub, zoo, inputs, outputs, train_range, train_samples, queri…)` | Alvo de modelo: arquitetura, hub/zoo, entradas/saídas, faixa de treino, consultas, domínio, método e relações. |
| `DatasetTarget` | classe | `(zoo, inputs, constraints, train, test, audit)` | Alvo de dataset: zoo, entradas, restrições, partições treino/teste e opções de auditoria. |
| `Integrity` | classe | `(manifests, public_key, signed, line)` | Configuração de integridade: manifestos, chave pública e arquivos assinados. |
| `Config` | classe | `(root, file, select, ignore, extend_select, fail_on, baseline, severity, exclude, results…)` | Configuração completa: raiz, seleção/ignorar de regras, fail_on, baseline, severidades, exclusões e alvos. |
| `find_config` | função | `(start) -> tuple[Path \| None, dict[str, Any]]` | Sobe a partir de `start` até o primeiro `pinneapple-check.toml` ou `pyproject.toml` com a seção do checador. |
| `load_config` | função | `(file=None, start=None, root=None) -> Config` | Interpreta um arquivo de configuração (encontrado a partir de `start` se não for dado). |
| `discover` | função | `(paths, exclude, root) -> list[ResultTarget]` | Descobre resultados sob os caminhos quando a configuração não declara nenhum: casos OpenFOAM e arquivos de campo. |

#### `pinneapple_check.engine`

Executa as checagens: configuração -> alvos -> resultados (`Outcome`) -> `Report`, aplicando seleção de regras e baseline.

<small>`pinneapple_check/engine.py` · 93 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `run` | função | `(paths=None, *, config=None, select=None, ignore=None, extend_select=None, baseline=None,…)` | Checa os alvos da configuração e devolve o `Report`. |

#### `pinneapple_check.evidence`

Traduz o relatório em evidência: trust card por alvo (os seis estágios de `pinneapple_lab.curation`) e as checagens de salvaguarda lidas por `pinneapple_veriphysics.applicability.build_applicability_map`.

<small>`pinneapple_check/evidence.py` · 66 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `trust_card` | função | `(report) -> dict[str, dict[str, Any]]` | Trust card por alvo: para cada estágio, título, pergunta, status e evidências. |
| `guardrail_checks` | função | `(report, target) -> list[dict[str, Any]]` | Lista `{name, passed, detail}` para o veriphysics, uma entrada por nome de evidência. |

#### `pinneapple_check.fields`

Lê resultados de qualquer formato (OpenFOAM, NPZ/HDF5, meshio, tabela de QoI) para uma estrutura única, `FieldData`.

<small>`pinneapple_check/fields.py` · 434 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LoadError` | classe | `()` | Erro ao ler um resultado. |
| `FieldData` | classe | `(kind, fields, units, files, source, axes, points, mesh, geom, boundary, patches, phi, ti…)` | Estrutura única de um resultado: campos, unidades, eixos ou pontos, malha, contorno, fluxo `phi`, tempo, snapshots e QoIs. |
| `scalar` | função | `(v) -> np.ndarray` | Converte um valor em array escalar. |
| `velocity_components` | função | `(fd) -> list[np.ndarray] \| None` | Velocidade como lista de componentes (de `U`/`u` ou de escalares `u, v, w`). |
| `velocity_name` | função | `(fd) -> str \| None` | Nome do campo de velocidade encontrado, ou None. |
| `load_result` | função | `(path, kind=None, time=None, fields=None) -> FieldData` | Lê um resultado escolhendo o leitor pelo tipo ou pela extensão. |
| `load_openfoam` | função | `(case, time=None, only=None) -> FieldData` | Lê um caso OpenFOAM (malha, campos, patches, `phi`) num instante. |
| `fv_geometry` | função | `(mesh) -> dict[str, np.ndarray]` | Centros e vetores de área das faces, centros e volumes das células, e direções resolvidas. |
| `read_surface_scalar` | função | `(b, poly) -> tuple[np.ndarray, np.ndarray]` | Lê um `surfaceScalarField` (`phi`) nas faces internas e de fronteira. |
| `load_arrays` | função | `(path, kind=None) -> FieldData` | Lê NPZ/HDF5 como grade (se houver coordenadas 1D) ou nuvem de pontos, com unidades opcionais. |
| `load_meshio` | função | `(path) -> FieldData` | Lê uma malha via meshio para `FieldData`. |
| `load_qoi` | função | `(path) -> FieldData` | Lê uma tabela de quantidades de interesse. |

#### `pinneapple_check.fv`

Operadores de volumes finitos sobre uma malha OpenFOAM (teorema de Gauss nas faces de cada célula) e os resíduos conservativos e balanços de fronteira construídos sobre eles.

<small>`pinneapple_check/fv.py` · 189 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FV` | classe | `(fd)` | Operadores de volumes finitos (fluxo, gradiente, laplaciano, somas por patch) sobre uma malha OpenFOAM. |
| `fv_residuals` | função | `(fv, physics, p)` | Resíduos por célula (por unidade de volume) das equações da física, com os motivos das que não rodaram. |
| `fv_mass_balance` | função | `(fv) -> dict[str, Any] \| None` | Balanço de massa na fronteira usando `phi` ou a velocidade interpolada. |
| `fv_heat_balance` | função | `(fv, p) -> dict[str, Any] \| None` | Balanço de calor na fronteira da malha. |

#### `pinneapple_check.models`

Carrega um modelo como função simples `(N, n_inputs) -> {saída: (N,)}`, com varredura de segurança do checkpoint antes de carregar.

<small>`pinneapple_check/models.py` · 192 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ModelLoadError` | classe | `()` | Erro ao carregar um modelo. |
| `UnsafeCheckpoint` | classe | `(report)` | Checkpoint recusado pela varredura de segurança, com o relatório. |
| `LoadedModel` | classe | `(fn, inputs, outputs, train_range=None, file=None)` | Modelo carregado como função, com entradas, saídas, faixa de treino e arquivo. |
| `load_model` | função | `(target) -> LoadedModel` | Carrega o modelo de um alvo (checkpoint, hub ou zoo). |
| `scan` | função | `(p)` | Varre um checkpoint com `pinneapple_security` antes de carregar. |

#### `pinneapple_check.numerics`

Derivadas de campos escalares nomeados em três discretizações (grade, nuvem de pontos por mínimos quadrados, modelo por diferenças centrais) atrás de uma interface única.

<small>`pinneapple_check/numerics.py` · 243 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GridD` | classe | `(axes, comps)` | Derivadas por diferenças finitas numa grade tensorial. |
| `PointD` | classe | `(points, comps, axes, k=None)` | Derivadas por ajuste quadrático de mínimos quadrados ponderados em nuvem de pontos; pontos na casca ficam de fora. |
| `ModelD` | classe | `(f, X, inputs, h, spans=None)` | Derivadas por diferenças centrais de um modelo nos pontos `X`. |
| `components` | função | `(fields, vector_ndim, ncomp=3) -> dict[str, np.ndarray]` | Separa campos vetoriais em escalares nomeados (`U` -> `u, v, w`). |
| `rel_rms` | função | `(residual, terms, mask=None, floor=0.0) -> float` | RMS do resíduo sobre RMS da soma dos termos absolutos (0 = exato, 1 = termos não se equilibram). |

#### `pinneapple_check.physics`

Equações governantes (escoamento incompressível, calor, elasticidade, conservação, Darcy, Maxwell) escritas sobre os provedores de derivadas, mais a tabela de parâmetros com dimensões e padrões; o docstring ainda diz 'quatro físicas'.

<small>`pinneapple_check/physics.py` · 195 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Equation` | classe | `(name, residual, terms, mask, description, scale)` | Uma equação avaliada: nome, resíduo, termos, máscara, descrição e escala. |
| `parse_parameters` | função | `(physics, given) -> tuple[dict[str, Any], list[tuple[str, str]]]` | Converte parâmetros para SI e lista os de dimensão errada (regra PC504). |
| `infer_physics` | função | `(names) -> str \| None` | Infere a física só pelos nomes dos campos (ex.: U e p -> escoamento incompressível). |
| `equations` | função | `(physics, D, p, conserved=None) -> tuple[list[Equation], list[str]]` | Resíduos da física num provedor de derivadas, com os motivos das equações que não puderam rodar. |

#### `pinneapple_check.report`

Formatos de saída: `text`, `json`, `sarif` (GitHub code scanning), `junit`, `github` (anotações) e `summary` em Markdown para o job summary.

<small>`pinneapple_check/report/__init__.py` · 216 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `render` | função | `(report, fmt, **kw) -> str` | Renderiza o relatório no formato pedido. |
| `text` | função | `(report, color=None, show_passed=False, show_fixes=True, **_) -> str` | Saída de texto legível, com cores e dicas de correção. |
| `to_json` | função | `(report, **_) -> str` | Saída JSON do relatório. |
| `sarif` | função | `(report, **_) -> str` | Saída SARIF para o GitHub code scanning. |
| `junit` | função | `(report, **_) -> str` | Saída JUnit XML. |
| `github` | função | `(report, **_) -> str` | Anotações por workflow commands do GitHub. |
| `summary` | função | `(report, **_) -> str` | Resumo em Markdown para o job summary do GitHub Action. |

#### `pinneapple_check.result`

O que uma checagem produz: um `Outcome` por (regra, alvo, sujeito), aprovado ou não, e o `Report` com código de saída. [sem teste direto]

<small>`pinneapple_check/result.py` · 119 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Location` | classe | `(file, line, column)` | Arquivo, linha e coluna de uma violação. |
| `Outcome` | classe | `(rule, target, subject, passed, message, location, value, threshold, severity, baselined,…)` | Uma avaliação de regra: aprovada ou violação, mensagem, valor, limite, severidade e dica. |
| `Report` | classe | `(outcomes, targets, config_file, root, notes)` | Relatório com todos os resultados, violações, erros, avisos, código de saída e trust card. |

#### `pinneapple_check.rules`

Tabela de regras do `pinneapple check`: cada regra tem id (`PC101`), nome, severidade padrão, estágio do trust card, explicação e dica de correção; seleção por prefixos no estilo ruff. [sem teste direto]

<small>`pinneapple_check/rules.py` · 290 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Rule` | classe | `(id, name, title, severity, stage, evidence, explanation, fix)` | Regra do checador: id, nome, severidade, estágio do trust card, explicação e correção. |
| `get_rule` | função | `(rule_id) -> Rule` | Busca uma regra pelo id. |
| `select_rules` | função | `(select=None, ignore=None, extend_select=None) -> set[str]` | Ids de regras ativas por prefixos no estilo ruff (`select`, `ignore`, `extend_select`). |

### Exemplos de uso reais do repositório

**Uso por Python (docstring do pacote)** — `pinneapple_check/__init__.py`

```python
from pinneapple_check import run
report = run(["cases/"])
assert not report.errors, [o.message for o in report.errors]
```

**GitHub Action e linha de comando** — `README.md`

```python
# .github/workflows/physics-check.yml, in a job with `permissions: security-events: write`
- uses: actions/checkout@v4
- uses: PINNeAPPle-Labs/PINNeAPPle@main    # pin a tag or a commit SHA

pinneapple check cases/ models/            # locally: same rules, readable output, exit code 1 on a violation
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 12 de 23.
- Funções de teste em arquivos que importam algum módulo do pacote: 144.

### Limitações conhecidas

- Os resíduos e balanços têm erro de discretização próprio; em malhas grossas o valor é dominado por ele, por isso as tolerâncias padrão são frouxas (docs/guides/validation_methods_reference.md).
- Condições de contorno checadas são só Dirichlet; Neumann só aparece pelos balanços. Um resíduo pequeno não detecta condição de contorno ou parâmetro errado que a EDP ainda satisfaz.
- A detecção de fora-da-distribuição (PC601/PC602) usa a caixa de treino e distâncias padronizadas; não cobre bem entradas correlacionadas.
- Unidades precisam estar no arquivo ou declaradas; a admissibilidade (PC402) infere a grandeza pelo nome e unidade do campo.
- torch só é necessário para carregar checkpoints `.pt`; `design` e `benchmark` no despachante só funcionam se esses pacotes estiverem instalados.
- Muitos submódulos (`checks.*`, `rules`, `result`, `baseline`, `balances`, `_reuse`) não têm teste que os importe diretamente; são exercitados pelos testes de `run`/CLI.
- O docstring de `physics` fala em 'quatro físicas', mas o código trata seis (escoamento incompressível, calor, elasticidade, conservação, Darcy, Maxwell).

## 3. `pinneapple_veriphysics`

### Objetivo

`pinneapple_veriphysics` é a camada de verificação e evidência do Veriphysics, extraída para dentro do PINNeAPPle (Apache-2.0); a fila de execução, a API e a cobrança ficam no Veriphysics. O pacote responde a pergunta "posso usar este resultado, para quê e com que evidência?" em vez de dar só aprovado/reprovado.

A regra central é não inventar números nem vereditos: `decision`, `applicability` e `evidence_report` só classificam e renderizam fatos que o pipeline já calculou, e uma checagem que não rodou fica `NOT_RUN`. O `ledger` reforça isso: uma checagem só conta como aprovada se há um registro íntegro, com hash, de que ela rodou. `scope` traduz vereditos em usos atendidos e não atendidos, e `robustness` mede extrapolação, ensemble, sensibilidade a pesos e deslocamento de parâmetros com treino real.

O pacote se liga ao resto assim: o `validation_lab` roda as regras do `pinneapple check` (`pinneapple_check`) e as agrupa em verificação numérica, validação contra dados e consistência física; `build_applicability_map` lê as checagens de salvaguarda de `pinneapple_check.evidence`; `recommend` usa `pinneapple_analysis.verification` e um rascunho de LLM (`pinneapple_llm.draft_problem`) para sugerir solver e arquitetura; `evidence_package` gera relatórios e `StudyResult` (`pinneapple_data.contracts`). Os apps `apps/validation_lab` e `apps/evidence_report` são as interfaces web sobre esses módulos.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 13 | 3.411 | 17 | 38 | 98 (11 arquivos) | 1 |

Pasta de testes dedicada: `tests/pinneapple_veriphysics` (49 funções de teste).

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `applicability` | Mapa de aplicabilidade com checklist de oito itens e envelope de variáveis. | 1 | 5 | 37 |
| `decision` | `DecisionRecord` com recomendação, trust score e alternativas. | 1 | 1 | 18 |
| `evidence_package` | Relatório e pacote zip de evidência reprodutível. | 1 | 9 | 22 |
| `evidence_report` | PDF do Evidence Report. | 1 | 1 | 18 |
| `example_cases` | Casos de exemplo fixos para o recomendador. | 1 | 2 | 0 |
| `execution_log` | Coletor de log de passos do pipeline. | 1 | 3 | 5 |
| `ledger` | Ledger de evidência com hashes. | 1 | 8 | 22 |
| `mesh_views` | Vistas 3D de erro contra referência em malha. | 1 | 1 | 3 |
| `recommend` | Formulação do problema e recomendação de método. | 1 | 3 | 1 |
| `robustness` | Estudos de robustez medidos. | 1 | 6 | 5 |
| `scope` | Escopo de uso de um resultado. | 1 | 4 | 5 |
| `validation_lab` | `pinneapple check` agrupado em três categorias com ledger. | 1 | 12 | 42 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_veriphysics`

Camada de verificação e evidência do Veriphysics dentro do PINNeAPPle; o `__init__` só documenta os submódulos (decisão, relatório, aplicabilidade, escopo, robustez, recomendação, ledger, Validation Lab).

<small>`pinneapple_veriphysics/__init__.py` · 35 linhas · testes: 8</small>

#### `pinneapple_veriphysics.applicability`

Mapa de aplicabilidade de uma corrida: classifica fatos já produzidos em VERIFIED, INFERRED ou UNSUPPORTED, responde um checklist fixo de oito itens e registra o envelope de variáveis testado; não roda checagem nem inventa número.

<small>`pinneapple_veriphysics/applicability.py` · 488 linhas · testes: 37</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Tolerances` | classe | `(plateau_rel, spread_rel, ece, coverage)` | Tolerâncias de aceitação do mapa (platô, dispersão, ECE, cobertura), substituíveis por critérios acordados. |
| `ChecklistItem` | classe | `(key, question, status, tier, summary, evidence)` | Um item do checklist de revisão: pergunta, status, nível (tier), resumo e evidência. |
| `VariableEnvelope` | classe | `(name, kind, verified_range, verified_point, extrapolation_supported, note)` | Faixa ou ponto verificado de uma variável e se a extrapolação é suportada. |
| `ApplicabilityMap` | classe | `(checklist, envelope, verified, inferred, unsupported, tolerances)` | O mapa completo: checklist, envelope e listas verificado/inferido/sem suporte, com `to_dict` e `render`. |
| `build_applicability_map` | função | `(*, architecture, coords, fields, domain_bounds, physical_parameters, guardrail_checks, c…)` | Classifica os fatos de uma corrida (checagens de salvaguarda, confiança, perda, domínio) num `ApplicabilityMap`. |

#### `pinneapple_veriphysics.decision`

`DecisionRecord`, a saída para humanos de uma corrida: recomendação, trust score e cobertura, evidência por checagem e alternativas, montado só de objetos que o orquestrador já produziu.

<small>`pinneapple_veriphysics/decision.py` · 157 linhas · testes: 18</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DecisionRecord` | classe | `(run_id, problem_description, recommended_strategy, reasoning, trust_score, trust_coverag…)` | Registro de decisão de uma corrida: estratégia recomendada, raciocínio, trust score, alternativas e ferramentas externas, com `render`. |

#### `pinneapple_veriphysics.evidence_package`

Evidência de engenharia: transforma arquivos de experimentos e simulações num relatório reprodutível (JSON, HTML, PDF, figuras) e num pacote zip com manifesto SHA-256 verificável.

<small>`pinneapple_veriphysics/evidence_package.py` · 638 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Source` | classe | `(name, kind, sha256, bytes, raw, url, data, contract, problems)` | Um arquivo de entrada com tipo detectado, SHA-256, tamanho, dados lidos e problemas. |
| `load_source` | função | `(name, data, url='') -> Source` | Lê um arquivo e decide o que ele é; conteúdo desconhecido é mantido como arquivo referenciado. |
| `build_report` | função | `(meta, sources) -> dict[str, Any]` | Monta o relatório como dados a partir dos metadados e das fontes (baseline contra candidato, testes). |
| `make_figures` | função | `(report, outdir) -> dict[str, str]` | Figuras PNG com os números do próprio relatório. |
| `render_html` | função | `(report, figures=None) -> str` | HTML autocontido do relatório com figuras embutidas. |
| `render_pdf` | função | `(report, figures=None, path=None) -> bytes` | PDF do relatório via `pinneapple_tools.reporting.paperkit.Paper` (precisa de reportlab). |
| `to_study_result` | função | `(report, app='evidence') -> dict[str, Any]` | Converte o relatório num `StudyResult` (`pinneapple_data.contracts`) para o próximo app. |
| `build_package` | função | `(report, sources, *, html=None, pdf=None, figures=None) -> bytes` | Zip com relatório, ledger, figuras, fontes originais e `MANIFEST.json` com SHA-256 de cada arquivo. |
| `verify_package` | função | `(data) -> list[str]` | Lista os problemas de um pacote (vazio quando todos os arquivos batem com o manifesto). |

#### `pinneapple_veriphysics.evidence_report`

Gera o Evidence Report em PDF a partir de um `DecisionRecord` já calculado (precisa do opcional `reportlab`).

<small>`pinneapple_veriphysics/evidence_report.py` · 206 linhas · testes: 18</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `render_evidence_report_pdf` | função | `(decision) -> bytes` | Renderiza um `DecisionRecord` como PDF e devolve os bytes. |

#### `pinneapple_veriphysics.example_cases`

Catálogo fixo de casos de exemplo para o recomendador (Burgers 1D e dois casos de turbomáquina), que contorna o LLM para esses casos conhecidos porque o preenchimento de kwargs pelo LLM se mostrou instável. [sem teste direto]

<small>`pinneapple_veriphysics/example_cases.py` · 228 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ExampleCase` | classe | `(key, label, preset, preset_kwargs, description, physical_parameters, flow_geometry, epoc…)` | Um caso de exemplo: preset, kwargs, parâmetros físicos, geometria, épocas, colocação e limiar de resíduo. |
| `get_example_case` | função | `(key) -> ExampleCase` | Busca um caso pela chave ou levanta ValueError listando as chaves registradas. |

#### `pinneapple_veriphysics.execution_log`

Coletor de log de execução seguro para threads, para a API/UI verem os passos reais do pipeline enquanto ele roda.

<small>`pinneapple_veriphysics/execution_log.py` · 51 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LogEntry` | classe | `(timestamp_unix, message)` | Uma entrada do log: instante e mensagem. |
| `LogCollector` | classe | `() -> None` | Coletor passado a `analyze()`/`formulate_and_recommend()` para observar os passos; pode ser lido de outra thread. |
| `emit` | função | `(log, message) -> None` | Registra no log se houver um, senão não faz nada. |

#### `pinneapple_veriphysics.ledger`

Ledger de evidência só de acréscimo: uma checagem só aparece como aprovada se há um registro íntegro (com hash) de que ela rodou.

<small>`pinneapple_veriphysics/ledger.py` · 242 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EvidenceEntry` | classe | `(check_id, name, category, tool, tool_version, inputs_sha256, outputs, started_at, finish…)` | Uma execução registrada de checagem: ferramenta, hash das entradas, saídas, tempos, resultado e hash da entrada. |
| `EvidenceLedger` | classe | `(entries)` | Lista só de acréscimo de `EvidenceEntry`; o status de uma checagem é o da última entrada válida. |
| `canonical_json` | função | `(obj) -> str` | JSON canônico (chaves ordenadas, sem espaços) para hashes estáveis. |
| `sha256_bytes` | função | `(data) -> str` | SHA-256 de bytes. |
| `sha256_json` | função | `(obj) -> str` | SHA-256 do JSON canônico de um objeto. |
| `sha256_file` | função | `(path, chunk=1 << 20) -> str` | SHA-256 de um arquivo lido em blocos. |
| `utc_now` | função | `(t=None) -> str` | Instante UTC em texto ISO. |
| `record_checks` | função | `(ledger, checks, *, prefix, inputs_sha256, started_at, finished_at, tool='', tool_version…)` | Registra no ledger as checagens de um trust card. |

#### `pinneapple_veriphysics.mesh_views`

Vistas 3D das comparações com referência de uma corrida do Validation Lab: predição, referência e erro pontual na malha (OpenFOAM, VTU/VTK) com as regiões de maior erro.

<small>`pinneapple_veriphysics/mesh_views.py` · 164 linhas · testes: 3</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `mesh_views` | função | `(workdir, config=None, *, report=None, max_faces=None, n_regions=8, mesh_unit='m') -> lis…)` | Uma vista 3D por alvo de resultado com referência: campo, erros e regiões. |

#### `pinneapple_veriphysics.recommend`

Metade de formulação e escolha de método do `pipeline.analyze()`: números adimensionais, regime, família de solver, arquitetura e ferramentas externas, com uma chamada `draft_problem` de LLM (`pinneapple_llm`) e sem treino.

<small>`pinneapple_veriphysics/recommend.py` · 295 linhas · testes: 1</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CandidateDetail` | classe | `(key, name, category, is_recommended, when_to_use, license, typical_cost_tier, player, st…)` | Uma opção documentada (arquitetura ou ferramenta externa) com prós, contras, licença e custo típico. |
| `RecommendationResult` | classe | `(rejected, rejection_reason, dimensionless_numbers, flow_regime, drafted_preset, drafted_…)` | Resultado da formulação e recomendação: rejeição, números adimensionais, regime, preset rascunhado, família de solver e candidatos. |
| `formulate_and_recommend` | função | `(description, *, physical_parameters=None, flow_geometry='internal_pipe', provider='ollam…)` | Formula o problema a partir de uma descrição e recomenda família de solver, arquitetura e ferramentas externas. |

#### `pinneapple_veriphysics.robustness`

Estudos de robustez medidos que alimentam o mapa de aplicabilidade: extrapolação, ensemble (UQ não calibrada sem referência), sensibilidade aos pesos da perda, cobertura das condições e deslocamento de parâmetros.

<small>`pinneapple_veriphysics/robustness.py` · 300 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ensemble_study` | função | `(spec, build_model, best_model, *, n_members, epochs, n_collocation, n_eval=2048, margins…)` | Treina sementes extras e mede a discordância entre membros no interior e fora do domínio. |
| `extrapolation_study` | função | `(spec, model, *, residual_threshold, ensemble=None, margins=DEFAULT_MARGINS, n_eval=2048,…)` | Resíduo (e dispersão do ensemble) em cascas fora do domínio de treino, dando o limite de extrapolação medido. |
| `weight_sensitivity_study` | função | `(spec, build_model, best_model, *, epochs, n_collocation, factors=DEFAULT_WEIGHT_FACTORS,…)` | Retreina com o peso da EDP escalado e mede quanto a predição muda. |
| `calibration_against_reference` | função | `(members, reference_x, reference_y) -> dict[str, Any]` | Calibração do ensemble contra dados de referência via `CalibrationMetrics`. |
| `condition_coverage` | função | `(spec, n=50000) -> dict[str, Any]` | Conta quantos pontos cada condição amostrada automaticamente seleciona com o mesmo amostrador do solver. |
| `parameter_shift_study` | função | `(make_spec, base_kwargs, build_model, base_model, *, residual_threshold, epochs, n_colloc…)` | Generalização em parâmetros: quanto o modelo se afasta quando parâmetros físicos são deslocados. |

#### `pinneapple_veriphysics.scope`

Escopo de uso: quais usos um resultado atende, onde falha e por quê, além de um veredito único aprova/reprova.

<small>`pinneapple_veriphysics/scope.py` · 148 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `UseScope` | classe | `(use, ok, reason, why, evidence, values, limits, fraction_ok, precondition)` | Um uso de um resultado: utilizável, não utilizável ou não avaliado, com motivo e evidência. |
| `ScopeOfUse` | classe | `(subject, uses, figure, notes)` | Os usos que um resultado atende ou não, com figura e notas opcionais. |
| `scope_from_verdicts` | função | `(subject, verdicts, *, evidence='estimated', figure=None, notes=()) -> ScopeOfUse` | Monta um `ScopeOfUse` a partir de vereditos por uso. |
| `scope_from_batch` | função | `(subject, verdicts, *, min_fraction=0.9, evidence='estimated', figure=None, notes=()) -> …)` | Monta um `ScopeOfUse` sobre um lote de vereditos por amostra (uso OK se a fração mínima passar). |

#### `pinneapple_veriphysics.validation_lab`

Validation Lab: roda as checagens do `pinneapple check` sobre predições, resultados ou um modelo do zoo e relata em três grupos (verificação numérica, validação contra dados, consistência física), com registro no ledger para cada checagem.

<small>`pinneapple_veriphysics/validation_lab.py` · 459 linhas · testes: 42</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CodeNotAllowed` | classe | `()` | Erro quando a configuração importaria ou desserializaria código enviado pelo usuário. |
| `LabReport` | classe | `(check_report, ledger, config_text, inputs, started_at, finished_at, versions, title)` | Relatório do lab: relatório do check, ledger, configuração, entradas, versões, categorias e decisão. |
| `category_of` | função | `(rule_id) -> str \| None` | Grupo de uma regra do `pinneapple check` (None para PC002, que é cobertura). |
| `to_toml` | função | `(doc) -> str` | Escreve um `pinneapple-check.toml` a partir de um dicionário. |
| `model_ref_target` | função | `(ref) -> dict[str, Any]` | Tabela `[[model]]` a partir de uma referência a modelo do zoo. |
| `config_from_spec` | função | `(spec) -> str` | Gera `pinneapple-check.toml` a partir de uma descrição simples do que checar. |
| `assert_no_code` | função | `(cfg) -> None` | Recusa alvos que executariam código das entradas (função Python, checkpoint, download do Hub). |
| `write_inputs` | função | `(root, files) -> list[str]` | Grava arquivos enviados sob `root`, recusando nomes que saiam dele. |
| `tables_to_csv` | função | `(root, config_text=None) -> tuple[str \| None, dict[str, str]]` | Converte Parquet e tabelas JSON em CSV ao lado dos originais. |
| `run_validation` | função | `(workdir, config=None, *, allow_code=False, title='Validation') -> LabReport` | Roda o `pinneapple check` nas entradas de `workdir` e devolve o `LabReport`. |
| `summarize` | função | `(lab) -> dict[str, Any]` | Visão compacta de um relatório do lab: decisão e contagens por grupo. |
| `render_html` | função | `(lab) -> str` | HTML autocontido de um relatório do lab, sem scripts nem arquivos externos. |

### Exemplos de uso reais do repositório

**Escopo de uso e Evidence Report para modelos PhysicsNeMo** — `examples/vs_physicsnemo/07_physicsnemo_trains_pinneapple_checks/run.py`

```python
    from pinneapple_veriphysics.decision import DecisionRecord
    from pinneapple_veriphysics.evidence_report import render_evidence_report_pdf
    from pinneapple_veriphysics.scope import scope_from_batch
...
            verdicts = [dc.use_verdicts(data["a"][i], preds[(r, m)][i], DARCY["f"], envelope=envelope)
                        for i in range(len(data["a"]))]
            label = {"in_distribution": "test set"}.get(r, SHIFTS.get(r, {}).get("label", r))
            scopes[r] = scope_from_batch(f"PhysicsNeMo {LABEL[m]}, {label}", verdicts, evidence=evidence)
```

**Validation Lab na API do app** — `apps/validation_lab/validation_lab/api.py`

```python
            (Path(d) / CONFIG_NAME).write_text(config, encoding="utf-8")
            with _HEAVY:
                lab = run_validation(d, allow_code=allow_code).to_dict()
                lab["views3d"] = _views(d, lab)
        except HTTPException:
            raise
        except CodeNotAllowed as e:
            raise HTTPException(422, str(e)) from e
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 12 de 13.
- Funções de teste em arquivos que importam algum módulo do pacote: 98.

### Limitações conhecidas

- `evidence_report` e `evidence_package.render_pdf` precisam do opcional `reportlab`.
- `recommend.formulate_and_recommend` depende de um LLM via `pinneapple_llm` (provedor padrão `ollama`); o docstring de `example_cases` registra que o LLM preenche kwargs de preset de forma instável, por isso há o catálogo fixo.
- `example_cases` cobre só Burgers 1D e dois casos de turbomáquina; presets com seletores por tag (`industrial_furnace_thermal`, `pipe_flow_3d`) ficam fora de escopo, segundo o docstring.
- A dispersão do ensemble em `robustness` é UQ não calibrada a menos que haja dados de referência; um resíduo pequeno fora do domínio não prova que o modelo acerta a solução verdadeira.
- `validation_lab` recusa por padrão alvos que executariam código do usuário (função Python, checkpoint, download do Hub); só modelos do zoo e arquivos de dados são aceitos sem `allow_code`.
- `example_cases` não tem teste direto e `recommend` tem pouca cobertura de teste.

## 4. `pinneapple_security`

### Objetivo

_Sem texto de objetivo._

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 20 | 3.398 | 33 | 93 | 105 (10 arquivos) | 0 |

Pasta de testes dedicada: `tests/pinneapple_security` (60 funções de teste).

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `__main__` |  | 1 | 1 | 10 |
| `anomaly` |  | 1 | 3 | 6 |
| `archives` |  | 1 | 7 | 22 |
| `attestation` |  | 1 | 7 | 10 |
| `audit` |  | 1 | 1 | 54 |
| `classification` |  | 1 | 4 | 10 |
| `compliance` |  | 1 | 3 | 10 |
| `cps` |  | 1 | 9 | 5 |
| `cps_bench` |  | 1 | 7 | 5 |
| `dp` |  | 1 | 7 | 9 |
| `encryption` |  | 1 | 9 | 6 |
| `integrate` |  | 1 | 8 | 10 |
| `integrity` |  | 1 | 8 | 14 |
| `model_security` |  | 1 | 6 | 6 |
| `passwords` |  | 1 | 4 | 56 |
| `privacy` |  | 1 | 11 | 9 |
| `secrets_scan` |  | 1 | 4 | 10 |
| `signing` |  | 1 | 12 | 25 |
| `web` |  | 1 | 15 | 40 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_security`

pinneapple_security — data and process security for scientific and industrial Physics AI.

<small>`pinneapple_security/__init__.py` · 67 linhas · testes: 64</small>

#### `pinneapple_security.__main__`

``python -m pinneapple_security <command>``: manifests, signatures, scans and log sanitising from the shell.

<small>`pinneapple_security/__main__.py` · 143 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `main` | função | `(argv=None) -> int` |  |

#### `pinneapple_security.anomaly`

Physics-based detection of manipulated sensor data (false data injection, spoofing, replay) in industrial streams.

<small>`pinneapple_security/anomaly.py` · 102 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ResidualCUSUM` | classe | `(k, h, mu, sigma, fitted, s_hi, s_lo, t, alarms)` |  |
| `balance_residual` | função | `(inflows, outflows, accumulation=None) -> np.ndarray` | Conservation residual Σin − Σout − d(storage)/dt per sample (mass, energy or charge balance). |
| `replay_score` | função | `(x, window=50) -> dict[str, Any]` | Smallest normalised distance between the last ``window`` samples and any earlier window. Near 0 → the latest data repeats an old recording sample for sample (replay attack) — live noise never matches that closely. |

#### `pinneapple_security.archives`

Open uploaded zip archives safely: no member outside the destination (zip-slip), no symbolic links, and limits on the number of members, the total uncompressed size, the size of one member and the compression ratio (zip bombs).

<small>`pinneapple_security/archives.py` · 153 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `UnsafeArchive` | classe | `()` | The archive breaks a safety rule or a limit. |
| `ArchiveLimits` | classe | `(max_members, max_total_bytes, max_member_bytes, max_ratio)` |  |
| `check_member_name` | função | `(name) -> str` | The member name as a safe relative POSIX path, or :class:`UnsafeArchive`. |
| `check_zip` | função | `(z, limits=DEFAULT_LIMITS) -> list[zipfile.ZipInfo]` | Check names, link entries and declared sizes; return the file members (directories left out). |
| `safe_extract_zip` | função | `(source, dest, limits=DEFAULT_LIMITS) -> list[Path]` | Extract ``source`` (path, bytes, file object or ZipFile) under ``dest``; return the written files. |
| `read_members` | função | `(source, limits=DEFAULT_LIMITS, skip=None) -> dict[str, bytes]` | All file members in memory, ``{safe name: bytes}``, under the same checks and counted limits. ``skip``: a predicate on the member name for members to leave out (not read, not counted). |
| `gunzip` | função | `(data, max_bytes=1 << 30) -> bytes` | Decompress gzip ``data``, refusing output above ``max_bytes`` (a gzip bomb) without producing it first. |

#### `pinneapple_security.attestation`

Provenance attestations (in-toto Statement v1 with a SLSA Provenance v1 predicate) and a CycloneDX SBOM.

<small>`pinneapple_security/attestation.py` · 153 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `subject` | função | `(name, path=None, sha256=None) -> dict[str, Any]` |  |
| `git_commit` | função | `(path='.') -> dict[str, Any] \| None` | Commit of the repository containing ``path`` and whether the work tree is dirty (None outside git). |
| `provenance` | função | `(external_parameters, dependencies=(), started_on=None, finished_on=None, builder_id=…, i…)` |  |
| `statement` | função | `(subjects, predicate, predicate_type=SLSA_PROVENANCE) -> dict[str, Any]` |  |
| `attest_files` | função | `(artifacts, parameters, inputs=(), repo='.') -> dict[str, Any]` | Statement for output files (``name → path`` or paths) built from input files with ``parameters``. |
| `attest_experiment` | função | `(result, artifacts=(), inputs=(), repo='.') -> dict[str, Any]` | Statement for a ``pp.Experiment`` result: the record itself (by digest) and any saved files are the subjects, the config and problem fingerprint are the parameters, the metrics go in ``internalParameters``. |
| `sbom` | função | `(name='pinneapple-environment', version='', packages=None) -> dict[str, Any]` | CycloneDX 1.5 SBOM of the installed distributions (all, or only ``packages``). |

#### `pinneapple_security.audit`

Tamper-evident audit trail: an append-only JSON-lines log where every record carries the hash of the previous one.

<small>`pinneapple_security/audit.py` · 133 linhas · testes: 54</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AuditLog` | classe | `(path, key=None)` |  |

#### `pinneapple_security.classification`

Data classification and handling policy for datasets, models and results.

<small>`pinneapple_security/classification.py` · 106 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DataLabel` | classe | `(level, owner, personal_data, export_control, licence, retention_days, allowed_countries,…)` |  |
| `Decision` | classe | `(allowed, reasons, requires)` |  |
| `HandlingPolicy` | classe | `(publish_levels, share_levels, cloud_levels)` | Default rules (organisation-specific: subclass or edit). ``approvals``: approvals the caller already holds, e.g. {"owner", "dpo", "export_officer"}. |
| `combine` | função | `(labels) -> DataLabel` | Label of something derived from all ``labels``: strictest level, any personal data, every export control. |

#### `pinneapple_security.compliance`

Which pinneapple_security capability produces evidence for which control of the frameworks industry and research are audited against — and a self-assessment report from the evidence a project actually has.

<small>`pinneapple_security/compliance.py` · 106 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `controls_for` | função | `(capability) -> list[dict[str, Any]]` |  |
| `assess` | função | `(evidence, frameworks=()) -> dict[str, Any]` | ``evidence``: capability → anything truthy (a path, a verification result with ``ok``, True). A control is ``covered`` when at least one of its capabilities has evidence, ``partial`` when some do but a verification failed. |
| `report_markdown` | função | `(result, title=…) -> str` |  |

#### `pinneapple_security.cps`

Physics-residual detection of false data injection in cyber-physical systems (water, process, HVAC, grids).

<small>`pinneapple_security/cps.py` · 305 linhas · testes: 5 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ResidualComponent` | classe | `(name)` |  |
| `MassBalance` | classe | `(level, inflows, outflows, dt, window, area, name)` | Storage balance of one volume over a window of ``window`` samples: ``area * (level[t] - level[t-w]) - dt * sum(inflows - outflows)[t-w:t]``, in volume units (m^3 for water). |
| `FlowLaw` | classe | `(flow, up, down, exponent, midpoint, coef, name)` | Flow through a valve, orifice, pipe or weir driven by a head difference: ``q^2 = a + b * (up - down)`` (turbulent square-root law, exponent 1/2; ``exponent`` 1 gives a linear laminar or pump-curve law). ``a`` absorbs the elevation offset between the two levels. Residual ``q - q_law`` in flow units. … |
| `CusumAlarm` | classe | `(k, h, upper_only)` | Alarm logic shared by every detector here: standardised scores ``self.z(D)`` (T, m) → CUSUM → threshold calibrated to a false-alarm budget. Subclasses define ``z`` and ``channels``; ``upper_only`` for scores where only large values are suspicious. Physics and data-driven detectors use the same … |
| `PhysicsResidualDetector` | classe | `(components, k, h, mu_, sigma_)` | Grey-box physics residuals + per-channel two-sided CUSUM, threshold calibrated to a false-alarm budget. |
| `cusum` | função | `(z, k=0.5, h=None, upper_only=False) -> tuple[np.ndarray, list[int]]` | Two-sided CUSUM of standardised scores ``z`` (T,) or (T, m). Returns the statistic (T,), the maximum over channels and both sides, and the alarm times when ``h`` is given (the sums restart after each alarm, so the alarm count is a false-alarm count on clean data). NaNs (warm-up) leave the sums … |
| `roc_auc` | função | `(scores, labels) -> float` | Area under the ROC curve (Mann-Whitney U with average ranks for ties). NaN scores are dropped. |
| `window_auc` | função | `(stat_attacked, episodes, stat_clean) -> float` | Episode-level ROC AUC: the largest statistic inside each attack episode against the largest statistic in every non-overlapping window of the same length of a clean trace. Separability of attacked from clean windows, independent of any threshold. |
| `episode_metrics` | função | `(alarms, episodes, n, samples_per_day=1440, grace=60) -> dict[str, Any]` | Detection rate and delay per attack episode ``(start, end)`` (end exclusive) and false alarms per day outside the episodes. An alarm in ``[start, end)`` detects the episode; the delay is in samples. Alarms in the ``grace`` samples after an episode (the attack being withdrawn is itself a … |

#### `pinneapple_security.cps_bench`

Open, reproducible benchmark for false-data-injection detection: a three-tank gravity water network.

<small>`pinneapple_security/cps_bench.py` · 331 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PCAResidualDetector` | classe | `(variance)` | Squared prediction error (Q statistic) of a PCA model of the sensors and their one-step increments. |
| `ARResidualDetector` | classe | `(lags, ridge, upper_only)` | Data-driven linear twin: each sensor predicted from ``lags`` past samples of every sensor (ridge regression); per-sensor two-sided CUSUM of the standardised one-step prediction residuals. |
| `IsolationForestDetector` | classe | `(n_estimators, seed, max_fit)` | Isolation forest (Liu, Ting & Zhou 2008) on sensors and their increments; needs scikit-learn. |
| `simulate_three_tank` | função | `(days=7.0, seed=0, dt=60.0, substeps=12, demand_scale=1.0, pump_band=(1.0, 2.5)) -> dict[…)` | Simulate the plant; returns sensor columns (``SENSORS``) plus the true states ``true_h1..3``. ``demand_scale`` and ``pump_band`` (T1 levels that switch the pump on and off) change the operating regime without changing the physics: a detector that learned the physics should not care, one that … |
| `inject` | função | `(D, kind, episodes) -> dict[str, np.ndarray]` | Copy of ``D`` with the reported readings rewritten in each ``(start, end)`` episode. |
| `physics_detector` | função | `(dt=60.0, window=1, k=0.5) -> PhysicsResidualDetector` | Grey-box twin of the plant: three storage balances and two valve laws, constants identified from data. |
| `run_benchmark` | função | `(seed=0, train_days=7.0, calib_days=7.0, test_days=10.0, episodes=20, length=240, false_a…)` | Fit every detector on a trusted trace, calibrate it to ``false_alarms_per_day`` on a second clean trace, then, for each operating regime (nominal, and a shifted regime the detectors never saw), score one clean test trace and one attacked copy per attack with ``episodes`` episodes of ``length`` … |

#### `pinneapple_security.dp`

Differential privacy: noise mechanisms, private statistics, a privacy accountant and DP-SGD for PyTorch models.

<small>`pinneapple_security/dp.py` · 155 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RDPAccountant` | classe | `()` | Privacy spent by ``steps`` applications of the sampled Gaussian mechanism (sampling rate ``q``, noise multiplier ``sigma`` = noise std / clipping norm). |
| `DPSGD` | classe | `(model, optimizer, loss_fn, noise_multiplier, max_grad_norm, sample_rate, seed=None)` | DP-SGD wrapper: ``step(xb, yb)`` computes per-sample gradients of ``loss_fn(model_output, y)`` (one scalar per sample), clips each to ``max_grad_norm``, adds N(0, (σ·C)²) noise to the sum and lets ``optimizer`` apply the noisy mean. ``sample_rate`` = batch size / dataset size (Poisson sampling is … |
| `laplace_mechanism` | função | `(value, sensitivity, epsilon, rng=None) -> Any` |  |
| `gaussian_sigma` | função | `(sensitivity, epsilon, delta) -> float` | σ = Δ₂ √(2 ln(1.25/δ)) / ε (Dwork & Roth Thm A.1; stated for ε ≤ 1). |
| `gaussian_mechanism` | função | `(value, sensitivity, epsilon, delta, rng=None) -> Any` |  |
| `dp_mean` | função | `(x, lower, upper, epsilon, rng=None) -> float` | ε-DP mean with values clipped to [lower, upper] and the record count treated as public. |
| `dp_histogram` | função | `(x, bins, epsilon, rng=None) -> np.ndarray` | ε-DP counts per bin (one record changes one count by 1); negative noisy counts are set to 0. |

#### `pinneapple_security.encryption`

Encryption at rest: AES-256-GCM for bytes, files and arrays, with keys from a passphrase (scrypt) or random.

<small>`pinneapple_security/encryption.py` · 106 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DecryptionError` | classe | `()` |  |
| `derive_key` | função | `(passphrase, salt) -> bytes` |  |
| `new_key` | função | `() -> bytes` |  |
| `encrypt_bytes` | função | `(data, secret) -> bytes` |  |
| `decrypt_bytes` | função | `(blob, secret) -> bytes` |  |
| `encrypt_file` | função | `(path, secret, out=None, remove_plain=False) -> str` |  |
| `decrypt_file` | função | `(path, secret, out=None) -> str` |  |
| `encrypt_array` | função | `(a, secret) -> bytes` |  |
| `decrypt_array` | função | `(blob, secret)` |  |

#### `pinneapple_security.integrate`

pinneapple_security in the rest of the library: signed model versions in the ModelStore, signed model cards and TrustReports, attested experiments, audited actions.

<small>`pinneapple_security/integrate.py` · 134 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `secure_save` | função | `(store, problem_id, model, signing_key, *, metadata=None, stage='development', trust_repo…)` | ``ModelStore.save`` + a manifest of the version folder, signed, + a safety scan of the checkpoint + an audit record. Returns the version. |
| `resign_version` | função | `(store, problem_id, version, signing_key, *, actor='', audit=None, reason='') -> str` | Re-sign a version after a legitimate change (e.g. ``ModelStore.promote`` rewrites metadata.json); the audit record keeps the old and new roots, so the change stays traceable. |
| `verify_version` | função | `(store, problem_id, version, key) -> dict[str, Any]` | Signature valid, manifest matches the signed root, files match the manifest, checkpoint scan safe. |
| `sign_record` | função | `(obj, key) -> dict[str, Any]` |  |
| `verify_record` | função | `(envelope, key, expected=None) -> Any` |  |
| `sign_model_card` | função | `(card, key, path=None) -> dict[str, Any]` | Envelope over ``card.to_dict()`` (TrustReport and override included); written next to the card if ``path``. |
| `verify_model_card` | função | `(card, envelope, key) -> dict[str, Any]` |  |
| `attest_and_sign_experiment` | função | `(result, key, directory=None, inputs=(), repo='.') -> dict[str, Any]` | In-toto/SLSA statement for a ``pp.Experiment`` result, signed. With ``directory``, the result is saved there first, its files become subjects, and ``attestation.json`` is written next to them. |

#### `pinneapple_security.integrity`

Integrity: content digests of files, arrays and records, and Merkle manifests of datasets and model folders.

<small>`pinneapple_security/integrity.py` · 139 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Manifest` | classe | `(files, algorithm, root)` | Files of a folder with their digests. ``files``: relative path (POSIX) → {"sha256", "size"}. |
| `sha256_bytes` | função | `(data) -> str` |  |
| `sha256_file` | função | `(path) -> str` |  |
| `canonical_json` | função | `(obj) -> bytes` | Canonical JSON bytes (sorted keys, compact separators, UTF-8); non-JSON values go through ``str``. |
| `digest_record` | função | `(obj) -> str` |  |
| `digest_array` | função | `(a) -> str` | SHA-256 of an array's dtype, shape and C-ordered bytes (NumPy arrays and PyTorch tensors). |
| `merkle_root` | função | `(leaves) -> str` | Merkle root (SHA-256) of hex leaf digests; an odd node is paired with itself. Empty input → hash of b''. |
| `build_manifest` | função | `(folder, include=(), exclude=…) -> Manifest` | Manifest of every file under ``folder`` (glob ``include`` filters; ``exclude`` drops names, folders, paths). |

#### `pinneapple_security.model_security`

Model security: scan checkpoints for code execution before loading, load safely, and measure adversarial sensitivity of surrogates.

<small>`pinneapple_security/model_security.py` · 165 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `UnsafeModelError` | classe | `()` |  |
| `ScanReport` | classe | `(path, globals, dangerous, unknown, errors)` |  |
| `scan_pickle_bytes` | função | `(data, name='<bytes>') -> ScanReport` |  |
| `scan_checkpoint` | função | `(path) -> ScanReport` | Scan a ``torch.save`` zip archive (every ``*.pkl`` member) or a legacy/plain pickle file. |
| `safe_load` | função | `(path, map_location='cpu', expected_sha256=None)` | Verify the digest (if given), scan, then ``torch.load(weights_only=True)``. Raises :class:`UnsafeModelError`. |
| `adversarial_sensitivity` | função | `(model, X, eps=0.01, steps=20, step_frac=0.25, input_range=None) -> dict[str, Any]` | Worst-case relative output change under an L∞ input perturbation of ``eps`` × the input range (per feature). |

#### `pinneapple_security.passwords`

Password hashing for logins: scrypt from the standard library by default, argon2id when ``argon2-cffi`` is installed, and verification of bcrypt hashes (``$2b$...``, as written by ``caddy hash-password`` or ``htpasswd -B``) when ``bcrypt`` is installed.

<small>`pinneapple_security/passwords.py` · 119 linhas · testes: 56</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `is_hash` | função | `(value) -> bool` | Whether ``value`` looks like a hash this module can verify (as opposed to a plain-text password). |
| `hash_password` | função | `(password, scheme='auto') -> str` | A salted hash of ``password``. ``scheme``: ``"scrypt"``, ``"argon2"`` (needs argon2-cffi) or ``"auto"`` (argon2 when installed, else scrypt). |
| `verify_password` | função | `(password, stored) -> bool` | Whether ``password`` matches the hash ``stored``. Unknown formats and missing libraries give False. |
| `needs_rehash` | função | `(stored) -> bool` | True for a scrypt hash weaker than the current parameters (argon2: its own check). |

#### `pinneapple_security.privacy`

Personal and sensitive data in text and tables: detection, redaction, pseudonymisation, k-anonymity, log sanitising.

<small>`pinneapple_security/privacy.py` · 180 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Finding` | classe | `(kind, start, end, text)` |  |
| `valid_cpf` | função | `(s) -> bool` |  |
| `valid_cnpj` | função | `(s) -> bool` |  |
| `luhn_ok` | função | `(s) -> bool` |  |
| `iban_ok` | função | `(s) -> bool` |  |
| `find_pii` | função | `(text, kinds=PII_KINDS) -> list[Finding]` | Non-overlapping findings, earlier kinds in ``PII_KINDS`` winning ties (a CNPJ is not also a phone). |
| `redact` | função | `(text, kinds=PII_KINDS, label=…) -> str` |  |
| `pseudonymize` | função | `(value, key, prefix='id_', length=16) -> str` | Keyed, deterministic token for ``value`` (HMAC-SHA256, truncated to ``length`` hex characters). |
| `pseudonymize_column` | função | `(values, key, prefix='id_') -> list[str]` |  |
| `k_anonymity` | função | `(rows, quasi_identifiers, k=5) -> dict[str, Any]` | ``rows``: a pandas DataFrame or a list of dicts. Returns the table's k, the groups smaller than ``k`` and the share of rows in them. |
| `sanitize_log` | função | `(text, extra=(), keep_paths=True) -> tuple[str, dict[str, int]]` | Strip personal and infrastructure identifiers from a log. ``extra``: literal strings to hide too (project or client names). Returns the clean text and how many of each kind were replaced. |

#### `pinneapple_security.secrets_scan`

Find credentials committed to code, notebooks, configs and solver cases before they are shared or published.

<small>`pinneapple_security/secrets_scan.py` · 116 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SecretFinding` | classe | `(rule, path, line, excerpt)` |  |
| `shannon_entropy` | função | `(s) -> float` |  |
| `scan_text` | função | `(text, path='<text>', entropy_threshold=4.5) -> list[SecretFinding]` |  |
| `scan_path` | função | `(root, exts=None, max_bytes=5000000, entropy_threshold=4.5) -> list[SecretFinding]` |  |

#### `pinneapple_security.signing`

Signatures for artifacts and records: Ed25519 (public-key, needs ``cryptography``) or HMAC-SHA256 (shared key, stdlib).

<small>`pinneapple_security/signing.py` · 189 linhas · testes: 25</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `VerificationError` | classe | `()` |  |
| `VerifyKey` | classe | `(algorithm, raw)` | Public half (Ed25519) or the shared secret (HMAC). ``raw``: 32 bytes. |
| `SigningKey` | classe | `(algorithm, raw)` |  |
| `pae` | função | `(payload_type, payload) -> bytes` | DSSE pre-authentication encoding. |
| `generate_key` | função | `() -> SigningKey` | New Ed25519 key (needs ``cryptography``). |
| `hmac_key` | função | `(secret=None) -> SigningKey` | Shared-secret key (stdlib only). Anyone who can verify can also sign: use it inside one organisation. |
| `sign_bytes` | função | `(data, key) -> dict[str, str]` |  |
| `verify_bytes` | função | `(data, signature, key) -> None` |  |
| `sign_json` | função | `(payload, key, payload_type=PAYLOAD_TYPE) -> dict[str, Any]` | DSSE envelope over the canonical JSON of ``payload``. |
| `verify_json` | função | `(envelope, key) -> Any` | Check the envelope with ``key``; returns the payload. Raises :class:`VerificationError`. |
| `sign_file` | função | `(path, key, sig_path=None) -> str` | Detached signature ``<path>.sig`` (JSON) over the file's SHA-256 and name. |
| `verify_file` | função | `(path, key, sig_path=None) -> dict[str, Any]` |  |

#### `pinneapple_security.web`

HTTP hardening for the PINNeAPPle web apps (ASGI: FastAPI / Starlette), with no dependency beyond the standard library at import time.

<small>`pinneapple_security/web.py` · 496 linhas · testes: 40</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SecurityHeaders` | classe | `(app, csp=None, hsts=…, frame_options='DENY', extra=None)` | ASGI middleware adding the security headers to every response (an existing header set by the route wins). |
| `BodyLimit` | classe | `(app, max_bytes, limits=None)` | ASGI middleware: 413 for a request body above ``max_bytes`` (``Content-Length`` checked first, then the bytes actually received, so chunked uploads can not pass the limit). ``limits``: per path prefix overrides. |
| `RateRule` | classe | `(name, count, seconds, methods, prefixes)` | ``count`` requests per ``seconds`` for requests whose method is in ``methods`` (empty: any) and whose path starts with one of ``prefixes`` (empty: any). |
| `RateLimiter` | classe | `(rules=(), max_keys=50000, clock=time.monotonic)` | Token buckets per ``(rule, key)``: a burst up to ``count`` and a steady ``count/seconds`` refill. Memory is bounded (least recently used keys are dropped past ``max_keys``). Thread-safe. |
| `LoginThrottle` | classe | `(max_failures, window, lock_seconds, clock, max_keys, _fails, _locked, _lock)` | Lock a client IP or a user name out for ``lock_seconds`` after ``max_failures`` failed logins within ``window`` seconds. A success clears the IP's and the user's failure count. |
| `UnsafeURL` | classe | `()` | A URL the server must not fetch. |
| `AuthAudit` | classe | `(app_name, path=None, key=None)` | Auth and admin events (login ok/failed/locked, API key created/revoked, admin denied, rate limited) as one JSON log line each on the ``pinneapple.security`` logger, and, with ``path``, in a hash-chained :class:`pinneapple_security.audit.AuditLog` (``key``: HMAC key for the chain). Secrets are never … |
| `client_ip` | função | `(peer, forwarded_for, trusted=DEFAULT_TRUSTED_PROXIES) -> str` | The caller's address. ``X-Forwarded-For`` is used only when the direct peer is a trusted proxy (by default a loopback or private address: the Docker network Caddy sits on), and then the right-most entry that is not itself a trusted proxy is taken. A client that reaches the app directly from a … |
| `inline_hashes` | função | `(page) -> tuple[set[str], set[str]]` | CSP hashes of the inline ``<script>`` blocks (including import maps) and of the inline event handlers (``onsubmit="..."``) of an HTML page. |
| `build_csp` | função | `(script_hashes=(), handler_hashes=(), *, connect_src=(), img_src=(), frame_ancestors="'no…)` | The Content-Security-Policy of an app page: everything from the app's own origin, the app's own inline scripts by hash (no ``'unsafe-inline'`` for scripts), WebGL/three.js friendly (``blob:`` workers and images). ``frame_src``: other origins the page may embed; ``frame_ancestors``: who may embed it … |
| `parse_rate` | função | `(spec, default=None) -> tuple[float, float] \| None` | ``"30/60"`` (30 requests per 60 s), ``"30/min"``, ``"5/s"``, ``"1000/h"``; ``off``/``0`` → None. |
| `same_origin_ok` | função | `(method, headers, allowed_origins=()) -> bool` | False for a state-changing browser request coming from another site (cross-site request forgery: a page elsewhere posting to the app with the user's cached Basic-auth credentials). Uses the browser-set ``Sec-Fetch-Site`` header, then ``Origin`` compared with ``Host`` (``X-Forwarded-Host`` behind a … |
| `check_cors_origins` | função | `(origins, allow_credentials) -> list[str]` | Validate a CORS allow-list: full ``scheme://host[:port]`` origins only, and no ``*`` with credentials. |
| `check_frame_origins` | função | `(origins) -> list[str]` | Validate origins allowed to embed an app (``frame-ancestors``) or to be embedded by it (``frame-src``): full ``scheme://host[:port]`` origins only; no ``*``, no scheme wildcards, no paths. |
| `check_outbound_url` | função | `(url, allow_hosts=(), resolve=None) -> str` | Refuse a URL a user asked the server to call unless it is http(s) to a public address, or its host is in ``allow_hosts`` (e.g. ``surrogate`` for an app on the same Docker network). Loopback, link-local (cloud metadata at 169.254.169.254), private, multicast and reserved addresses are refused, for … |

### Exemplos de uso reais do repositório

_Nenhum exemplo encontrado no repositório para este pacote._

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 20 de 20.
- Funções de teste em arquivos que importam algum módulo do pacote: 105.

### Limitações conhecidas

- _Nenhuma registrada._

## 5. `pinneapple_decision`

### Objetivo

_Sem texto de objetivo._

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 12 | 2.499 | 37 | 20 | 44 (3 arquivos) | 9 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `adapter` |  | 1 | 2 | 0 |
| `api` |  | 1 | 5 | 16 |
| `backends` |  | 1 | 3 | 0 |
| `constraints` |  | 1 | 5 | 0 |
| `debias` |  | 1 | 8 | 21 |
| `decider` |  | 1 | 1 | 0 |
| `loop` |  | 1 | 7 | 0 |
| `schema` |  | 1 | 10 | 0 |
| `selectors` |  | 1 | 4 | 0 |
| `tree` |  | 1 | 9 | 0 |
| `verifiers` |  | 1 | 3 | 0 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_decision`

pinneapple_decision -- probabilistic decision layer over the Physics AI pipeline.

<small>`pinneapple_decision/__init__.py` · 116 linhas · testes: 44</small>

#### `pinneapple_decision.adapter`

Problem adapter: real ``ProblemSpec`` objects and free-form dicts -> decision state.

<small>`pinneapple_decision/adapter.py` · 508 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AdaptedProblem` | classe | `(problem, given, inferred, unknown, unrecognized, source)` | Result of :func:`adapt_problem`. |
| `adapt_problem` | função | `(obj) -> AdaptedProblem` | Adapt a ``ProblemSpec``, a free-form dict, or ``None`` to the decision layer. |

#### `pinneapple_decision.api`

Module-level API used by ``import pinneapple as pp; pp.decide(...)``.

<small>`pinneapple_decision/api.py` · 59 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `decision_engine` | função | `(executor=None, verifier=None, decider=None, store=None, *, reset=False) -> PhysicsDecisi…)` | Return the default engine. Passing any argument (or ``reset=True``) replaces it with a new engine built from those arguments (evidence is not carried over). |
| `decide` | função | `(problem=None, objective=None, *, state=None, choice=None, engine=None) -> Decision` | Decide which experiment to run next. ``problem`` may be a ``ProblemSpec`` or a free-form dict (see ``adapter.py``). Facts that could not be determined are listed in ``decision.diagnostics["problem_facts"]["unknown"]``. |
| `execute` | função | `(decision, problem=None, *, engine=None) -> ExecutionResult` | Execute a decision issued by the same engine (constraints are re-checked first). |
| `verify` | função | `(result, *, engine=None) -> Verification` | Verify a result; only verified results become evidence for the next decision. |
| `run_tree` | função | `(problem, tree=None, *, max_experiments=3, engine=None)` | Walk the physics-AI decision tree (``tree.physics_ai_tree()`` by default). |

#### `pinneapple_decision.backends`

Scoring backends: turn (state, choice, options in a given order) into scores.

<small>`pinneapple_decision/backends.py` · 153 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ScoringBackend` | classe | `(name, has_label_prior)` |  |
| `RuleBasedBackend` | classe | `(rules=None)` | Deterministic backend over per-choice rule functions (see ``selectors.py``). |
| `LogProbLLMBackend` | classe | `(logprob_fn, *, template=DEFAULT_TEMPLATE, name='llm-logprob')` | Scores options by next-token log-probability of their letter label. |

#### `pinneapple_decision.constraints`

Hard constraints on options: physics/geometry, resources, implementation.

<small>`pinneapple_decision/constraints.py` · 137 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Constraint` | classe | `(name, check)` |  |
| `NoFeasibleOptionError` | classe | `(choice, excluded)` |  |
| `ConstraintViolationError` | classe | `()` | Raised when a decision is about to be executed but its option no longer passes. |
| `resolve` | função | `(constraint) -> Constraint` |  |
| `apply_constraints` | função | `(choice, state) -> Tuple[List[str], Dict[str, str]]` | Return ``(feasible_options, {excluded_option: reason})`` in the choice's option order. |

#### `pinneapple_decision.debias`

L0 debiasing without labels (AnyJev-like; own implementation, see README).

<small>`pinneapple_decision/debias.py` · 161 linhas · testes: 21</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BatchPrior` | classe | `(min_prior_n=8)` | Running mean of L0 distributions per (choice name, feasible option set). |
| `log_softmax` | função | `(xs) -> List[float]` |  |
| `softmax_dict` | função | `(logits) -> Dict[str, float]` |  |
| `normalize` | função | `(p) -> Dict[str, float]` |  |
| `rotations` | função | `(options, max_permutations=None) -> List[List[str]]` |  |
| `raw_distribution` | função | `(score_fn, options) -> Dict[str, float]` |  |
| `permutation_marginalized` | função | `(score_fn, options, *, combine='geometric', max_permutations=None) -> Tuple[Dict[str, flo…)` | Distribution averaged over cyclic rotations of the option order. |
| `apply_prior` | função | `(dist, prior, strength=1.0) -> Dict[str, float]` |  |

#### `pinneapple_decision.decider`

PhysicsDecider: constraints -> scoring -> L0 debiasing -> Decision.

<small>`pinneapple_decision/decider.py` · 119 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsDecider` | classe | `(backend=None, *, debias=True, prior='content_free', prior_strength=1.0, combine='geometr…)` | Decide which option of a :class:`PhysicsChoice` to run next. |

#### `pinneapple_decision.loop`

OBSERVE -> UNDERSTAND -> DECIDE -> EXECUTE -> VALIDATE -> LEARN.

<small>`pinneapple_decision/loop.py` · 215 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ValidationRequiredError` | classe | `()` | A result was offered as evidence before it was verified. |
| `Executor` | classe | `()` |  |
| `CallableExecutor` | classe | `(fn)` | Wrap ``fn(decision, problem) -> ExecutionResult \| dict``. |
| `EvidenceStore` | classe | `(path=None)` | In-memory evidence log, optionally persisted as JSON. |
| `DecisionLoop` | classe | `(executor=None, verifier=None, decider=None, store=None)` |  |
| `PhysicsDecisionEngine` | classe | `()` | High-level facade: ``decide(problem, objective=...)`` / ``decide(state={...})``, ``execute(decision)``, ``verify(result)``. |
| `choice_for_objective` | função | `(objective, has_previous_result=False) -> PhysicsChoice` |  |

#### `pinneapple_decision.schema`

Core types of the decision layer: choices, decisions, evidence.

<small>`pinneapple_decision/schema.py` · 284 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DecisionLevel` | classe | `()` | What was done to the backend's scores before reading them. |
| `OptionInfo` | classe | `(name, description, implementation, representations, supports_varying_geometry, relative_…)` | Metadata of one option, used by rules and constraints (never by the LLM to decide). |
| `PhysicsChoice` | classe | `(name, question, options, constraints, option_info)` | A typed question with a closed set of options. |
| `ExecutionResult` | classe | `(status, metrics, diagnostics, artifacts, decision_id)` | What an executor returns. ``diagnostics`` are short tags a rule or LLM can read on the next decision (e.g. ``"residual_localized"``, ``"late_time_error"``). |
| `Verification` | classe | `(passed, checks, tags, source, notes)` | Outcome of verifying one :class:`ExecutionResult`. |
| `Decision` | classe | `(choice, question, selected, probabilities, level, requires_validation, calibrated_confid…)` |  |
| `Evidence` | classe | `(decision, result, verification)` | One OBSERVE -> DECIDE -> EXECUTE -> VALIDATE record. |
| `DecisionState` | classe | `(problem, previous_result, verification, history, facts)` | Everything a backend may look at: the problem plus the evidence so far. |
| `problem_from_spec` | função | `(spec) -> Dict[str, Any]` | Map a ``pinneapple_problemdesign.ProblemSpec`` to this module's problem dict. |
| `dumps` | função | `(obj) -> str` |  |

#### `pinneapple_decision.selectors`

The three MVP decisions: model, training strategy, validation strategy.

<small>`pinneapple_decision/selectors.py` · 358 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ModelSelector` | classe | `()` |  |
| `TrainingStrategySelector` | classe | `()` |  |
| `ValidationStrategySelector` | classe | `()` |  |
| `FactQuestion` | classe | `(name, question, fact, hints=None)` | A yes/no question whose answer is a *fact* about the problem (decision tree). |

#### `pinneapple_decision.tree`

Declarative decision tree walked by :class:`~pinneapple_decision.loop.DecisionLoop`.

<small>`pinneapple_decision/tree.py` · 297 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ChoiceNode` | classe | `(id, choice, edges, fact, fact_options, plan_key, plan_updates)` |  |
| `ExecuteNode` | classe | `(id, next, plan_updates)` |  |
| `ValidateNode` | classe | `(id, on_pass, on_fail, plan_updates)` |  |
| `TerminalNode` | classe | `(id, outcome, plan_updates)` |  |
| `DecisionTree` | classe | `(name, root, nodes)` |  |
| `TreeStep` | classe | `(node, kind, selected, resolved_by, decision_id, passed, note)` |  |
| `TreeRun` | classe | `(tree, outcome, path, plan, experiments, evidence, facts)` |  |
| `physics_ai_tree` | função | `() -> DecisionTree` | The tree proposed for the physics-AI workflow (see module docstring). |
| `run_tree` | função | `(loop, problem, tree=None, *, max_experiments=3, max_steps=200) -> TreeRun` | Walk ``tree`` with ``loop`` (a :class:`DecisionLoop`). See the module docstring. |

#### `pinneapple_decision.verifiers`

Verifiers: turn an ExecutionResult into a Verification.

<small>`pinneapple_decision/verifiers.py` · 92 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Verifier` | classe | `()` |  |
| `ThresholdVerifier` | classe | `(thresholds=None)` |  |
| `VeriPhysicsVerifier` | classe | `(min_trust=70.0, min_coverage=0.5, fallback=None)` |  |

### Exemplos de uso reais do repositório

_Nenhum exemplo encontrado no repositório para este pacote._

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 3 de 12.
- Funções de teste em arquivos que importam algum módulo do pacote: 44.

### Limitações conhecidas

- _Nenhuma registrada._

## 6. `pinneapple_reason`

### Objetivo

_Sem texto de objetivo._

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 18 | 3.757 | 17 | 60 | 17 (1 arquivos) | 8 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `__main__` |  | 1 | 1 | 0 |
| `cases` |  | 1 | 5 | 17 |
| `chain` |  | 1 | 13 | 17 |
| `engine` |  | 1 | 3 | 0 |
| `library` |  | 1 | 11 | 17 |
| `llm` |  | 1 | 2 | 17 |
| `modes` |  | 7 | 20 | 17 |
| `narrative` |  | 1 | 2 | 0 |
| `props` |  | 1 | 6 | 17 |
| `report` |  | 1 | 4 | 0 |
| `trust` |  | 1 | 5 | 0 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_reason`

pinneapple_reason: the chain of whys of an engineering design, computed.

<small>`pinneapple_reason/__init__.py` · 77 linhas · testes: 17</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `write_report` | função | `(chain, out_dir, lang='pt', **kw)` | Write chain.json, report.md, report.html, the figures and the trust card to ``out_dir``. |
| `trust_report` | função | `(chain, **kw)` |  |
| `run_case` | função | `(name, **kw) -> Chain` | Reason through one of :data:`cases.CASES`. |
| `verify` | função | `(concept, requirements) -> list[Verification]` | Run every failure mode of ``requirements`` on ``concept`` (no countermeasures). |
| `describe_library` | função | `(lang='pt') -> dict` | The library: failure modes by domain and their countermeasures. |

#### `pinneapple_reason.__main__`

``python -m pinneapple_reason``: run a case, list the library, explain a saved chain.

<small>`pinneapple_reason/__main__.py` · 45 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `main` | função | `(argv=None) -> int` |  |

#### `pinneapple_reason.cases`

Worked cases at different scales and in different fields, each starting from the naive concept.

<small>`pinneapple_reason/cases.py` · 150 linhas · testes: 17</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `roof` | função | `()` | A house roof in a warm, humid climate. |
| `footbridge` | função | `()` | A 30 m steel footbridge (two welded girders, 2.5 m deck). |
| `power_board` | função | `()` | A 20 A MOSFET switch on a 4-layer board inside an enclosure at 40 °C. |
| `pumped_main` | função | `()` | A pumping station lifting water from a sump into a tank, and the gravity main from the tank to a valve. |
| `get` | função | `(name)` |  |

#### `pinneapple_reason.chain`

The chain of whys: requirement -> naive concept -> physical check -> failure found -> countermeasure -> check ...

<small>`pinneapple_reason/chain.py` · 409 linhas · testes: 17</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Requirement` | classe | `(id, text, modes, limits, source)` | What the design must do (``function``) and the failure modes that would stop it (``modes``). |
| `Concept` | classe | `(name, params, elements, description, sources, version)` | One version of the design: its parameters (SI units), the elements it has and where each value came from. |
| `Validity` | classe | `(name, value, lo, hi, note)` | One applicability condition of the model a check used: ``lo <= value <= hi`` (either bound may be None). |
| `Verification` | classe | `(mode, requirement, passed, demand, capacity, unit, margin, demand_label, capacity_label,…)` | The result of one failure mode checked on one concept. |
| `Finding` | classe | `(requirement, mode, title, verification)` | The first failure mode that occurs: which requirement it breaks and the number that proves it. |
| `Option` | classe | `(countermeasure, element, title, feasible, reason, changes, cost, effect, target_margin_b…)` | A countermeasure the engine tried for a finding: what it changed, what it costs and what it did to every requirement (margins before / after), or why it could not be applied. |
| `Step` | classe | `(version, concept, verifications, finding, options, chosen, why)` | Version ``version`` of the design, every check run on it, the first failure (if any), the countermeasures tried for it and the one chosen (``chosen``: index into ``options``). |
| `Chain` | classe | `(title, requirements, steps, status, notes, meta)` | The whole reasoning: requirements, every version of the design and why each one was made. |
| `tr` | função | `(pt, en) -> dict[str, str]` | A text in Portuguese and English. |
| `text_of` | função | `(t, lang='pt') -> str` | The text in ``lang`` (falls back to the other language, or to ``str(t)``). |
| `param_text` | função | `(name, value, lang='pt') -> str` | ``name = value unit`` with the parameter's name in ``lang`` (raw name and value when unknown). |
| `element_label` | função | `(name, lang='pt') -> str` | The name of an element in ``lang`` (the raw name when it is not a built-in one). |
| `cost_text` | função | `(cost, lang='pt') -> str` | What a countermeasure costs, in words: its ``what`` template filled with the quantity ({q}; {pct} as a percentage, {mm} in millimetres), or ``quantity unit``. |

#### `pinneapple_reason.engine`

The reasoning engine: check every requirement, find the first failure, try the countermeasures, keep the best, check again, until every requirement passes.

<small>`pinneapple_reason/engine.py` · 211 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Advisor` | classe | `()` |  |
| `Reasoner` | classe | `(library=LIBRARY, advisor=None, max_steps=12, progress=None)` |  |
| `reason` | função | `(requirements, concept, title='', *, advisor=None, max_steps=12, library=LIBRARY, progres…)` | Build the chain of whys for ``requirements`` starting from the naive ``concept``. |

#### `pinneapple_reason.library`

The library of failure modes and countermeasures, by physical domain.

<small>`pinneapple_reason/library.py` · 246 linhas · testes: 17</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Infeasible` | classe | `(pt, en)` | A countermeasure cannot be applied to this concept (the message says why, in both languages). |
| `Need` | classe | `(name, unit, label)` | A parameter a failure mode reads: SI unit and meaning. |
| `FailureMode` | classe | `(name, domain, title, summary, evaluate_fn, needs, sources, countermeasures, uncertain, k…)` | A failure mode and its quantitative check. |
| `Countermeasure` | classe | `(name, element, title, effect, addresses, apply_fn, cost_fn, unit_cost, unit_cost_note, r…)` | A change that acts on the mechanism of a failure mode. |
| `Library` | classe | `()` |  |
| `param` | função | `(p, name, default=None) -> Any` |  |
| `self_check` | função | `(name, passed, value=None, limit=None, source='', reason='') -> Check` | An internal check of a computation (residual, closure, agreement with an independent solution). |
| `register_mode` | função | `(m, replace=False) -> FailureMode` |  |
| `register_countermeasure` | função | `(c, replace=False) -> Countermeasure` |  |
| `validity` | função | `(name, value, lo=None, hi=None, note='') -> Validity` |  |
| `iter_known_answers` | função | `(names=None)` | ``(mode, case)`` for every known-answer case of the library (or of ``names``). |

#### `pinneapple_reason.llm`

An LLM as an advisor of the reasoning: it may propose candidate countermeasures and a first concept, never a number that decides anything.

<small>`pinneapple_reason/llm.py` · 63 linhas · testes: 17</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ExtractLLM` | classe | `()` |  |
| `LLMAdvisor` | classe | `(llm, max_proposals=3)` |  |

#### `pinneapple_reason.modes`

Built-in failure modes and countermeasures, one module per physical domain (imported on first use of the library). Add a domain by writing a module like these and calling ``register_mode`` / ``register_countermeasure``.

<small>`pinneapple_reason/modes/__init__.py` · 6 linhas · **sem teste direto**</small>

#### `pinneapple_reason.modes._util`

Shared helpers of the built-in failure modes.

<small>`pinneapple_reason/modes/_util.py` · 69 linhas · testes: 17</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `verdict` | função | `(*, demand, capacity, unit, scale=None, demand_label, capacity_label, mechanism, method, …)` | A verification that passes when ``demand <= capacity``; margin = (capacity - demand) / scale. |
| `q` | função | `(value, unit, label) -> list` |  |
| `target_margin` | função | `(p) -> float` |  |
| `size_to_pass` | função | `(mode, p, values, make, limits=None, add=()) -> tuple[dict[str, Any], Any]` | The first candidate (in order) whose changes make ``mode`` pass with the case's design margin. |
| `fmt` | função | `(x, nd=3) -> str` |  |

#### `pinneapple_reason.modes.corrosion`

Corrosion: galvanic coupling of two metals in an aerated electrolyte.

<small>`pinneapple_reason/modes/corrosion.py` · 91 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `faraday_rate` | função | `(i, metal) -> float` | Uniform wall-loss rate (m/s) of ``metal`` dissolving at current density ``i`` (A/m^2). |

#### `pinneapple_reason.modes.fluids`

Liquids in pipes: pump cavitation (NPSH), water hammer on valve closure, column separation.

<small>`pinneapple_reason/modes/fluids.py` · 328 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `npsh` | função | `(p) -> dict` |  |
| `main_wave_speed` | função | `(p) -> float` |  |
| `transient` | função | `(p, instantaneous_frictionless=False) -> dict` |  |

#### `pinneapple_reason.modes.structural`

Structures: bending strength, deflection, lateral-torsional buckling and pedestrian-induced resonance of steel girders; flexural buckling of a column; fatigue (Basquin); wind uplift of a roof covering.

<small>`pinneapple_reason/modes/structural.py` · 571 linhas · testes: 17</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `i_section` | função | `(h, b, tf, tw) -> dict` |  |
| `crowd_load` | função | `(span) -> float` | EN 1991-2 5.3.2.1 footbridge uniformly distributed load, Pa. |
| `girder_loads` | função | `(p) -> dict` |  |
| `chi` | função | `(lam, alpha) -> float` | EN 1993-1-1 6.3.1.2 / 6.3.2.2 reduction factor for slenderness ``lam`` and imperfection factor ``alpha``. |
| `m_cr` | função | `(sec, L_cr, C1=1.0) -> float` | Elastic critical moment of a doubly symmetric I-beam, load at the shear centre (Timoshenko & Gere). |
| `frequency` | função | `(p) -> dict` |  |
| `pedestrian_response` | função | `(p) -> dict` |  |

#### `pinneapple_reason.modes.thermal`

Heat: condensation under a roof on a clear night, radiant heat from a sun-baked roof, Joule heating of a power device.

<small>`pinneapple_reason/modes/thermal.py` · 547 linhas · testes: 17</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `roof_network` | função | `(p, case) -> dict` | Steady temperatures (°C) and heat fluxes (W/m^2) of the roof for ``case`` = "night" or "day". |
| `electrothermal` | função | `(p) -> dict` | Junction temperature with P = I^2 R_on(T_j) on the concept's thermal path (fixed point). |

#### `pinneapple_reason.modes.water`

Water: rain getting through the laps of a roof covering (capillarity, wind pressure, ponding / falling film).

<small>`pinneapple_reason/modes/water.py` · 213 linhas · testes: 17</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `lap_heads` | função | `(p) -> dict` | Heads acting on the lap (m of water) and the lap's rise, for the parameters ``p``. |
| `minimum_slope` | função | `(p, margin=0.0) -> float` | Smallest pitch (deg) at which the lap holds ``(1 + margin)`` times the head, by bisection (the film depth falls as the slope rises). ``inf`` if not even 75° is enough. |

#### `pinneapple_reason.narrative`

The chain told in words: one paragraph per version, each claim followed by the number that proves it.

<small>`pinneapple_reason/narrative.py` · 60 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `option_line` | função | `(o, lang) -> str` |  |
| `explain` | função | `(chain, lang='pt') -> str` |  |

#### `pinneapple_reason.props`

Physical constants, properties and small closed-form relations the failure modes share, each with its source.

<small>`pinneapple_reason/props.py` · 76 linhas · testes: 17</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `air` | função | `(t_c, p_pa=101325.0) -> dict[str, float]` | Air properties at ``t_c`` (k, nu, Pr, rho, cp, alpha, beta). |
| `saturation_pressure` | função | `(t_c) -> float` | Saturation vapour pressure of water over a flat water surface, Pa (Magnus form, Alduchov & Eskridge 1996; within 0.4 % of the reference values for -40..50 °C). |
| `dew_point` | função | `(t_c, rh) -> float` | Dew-point temperature, °C, of air at ``t_c`` with relative humidity ``rh`` (0-1) (inverse Magnus form). |
| `sky_temperature` | função | `(t_air_k) -> float` | Effective clear-sky temperature, K: T_sky = 0.0552 T_air^1.5 (Swinbank 1963). |
| `wind_convection` | função | `(v) -> float` | External convection coefficient of a surface in wind, W/(m^2 K): h = 2.8 + 3.0 V (Watmuff et al. 1977). |
| `atmospheric_pressure` | função | `(z_m) -> float` | Standard-atmosphere pressure at altitude ``z_m``, Pa (ISO 2533, troposphere). |

#### `pinneapple_reason.report`

Reports of a chain: ``chain.json``, ``report.md``, ``report.html`` (self-contained), ``trust.json`` and figures.

<small>`pinneapple_reason/report.py` · 487 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `timeline_figure` | função | `(chain, path, lang='pt') -> Path` | Version cards left to right; the arrow between two cards says what failed (with its numbers) and what was added. |
| `margins_figure` | função | `(chain, path, lang='pt') -> Path` |  |
| `demand_capacity_figure` | função | `(chain, path, lang='pt') -> Path` |  |
| `write_report` | função | `(chain, out_dir, lang='pt', card=True) -> dict[str, Path]` | Write the chain, its figures, the trust report and the reports; returns the paths. |

#### `pinneapple_reason.trust`

Trust of a chain: the checks behind the final design, as a :class:`~pinneapple_analysis.trust.TrustReport` and as the six-stage trust card of :func:`pinneapple_lab.curation.trust_stages`.

<small>`pinneapple_reason/trust.py` · 108 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `known_answer_checks` | função | `(modes) -> list[Check]` |  |
| `uncertainty_checks` | função | `(chain) -> list[Check]` | Worst margin of each final check when one uncertain premise at a time moves by +- its relative spread. |
| `chain_checks` | função | `(chain) -> list[tuple[Check, str]]` | ``(check, kind)`` for the final version of ``chain``. |
| `trust_report` | função | `(chain, **kw) -> TrustReport` |  |
| `trust_card` | função | `(chain, out_dir='.', report=None) -> dict[str, dict[str, Any]]` | The six questions (data and geometry, model, physics, benchmark, uncertainty, decision) answered from the chain's own checks with :func:`pinneapple_lab.curation.trust_stages`. |

### Exemplos de uso reais do repositório

_Nenhum exemplo encontrado no repositório para este pacote._

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 10 de 18.
- Funções de teste em arquivos que importam algum módulo do pacote: 17.

### Limitações conhecidas

- _Nenhuma registrada._
