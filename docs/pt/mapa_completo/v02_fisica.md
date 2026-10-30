# Volume 2 — Física: problemas, equações e ciência

> Parte do [Mapa completo do PINNeAPPle](../mapa_completo.md). Gerado por `docs/pt/_mapa/gerar.py` a partir do inventário do commit `e53cdcfb04`.

Como um problema físico é descrito e resolvido (`PhysicalProblem`, `pp.solve`), as equações, os presets industriais, o desenho de problemas, os experimentos por área, a condensação e as consultas físicas.

## Sumário

1. [`pinneapple_physics`](#1-pinneapple_physics)
2. [`pinneapple_presets`](#2-pinneapple_presets)
3. [`pinneapple_problemdesign`](#3-pinneapple_problemdesign)
4. [`pinneapple_science`](#4-pinneapple_science)
5. [`pinneapple_condense`](#5-pinneapple_condense)
6. [`pinneapple_query`](#6-pinneapple_query)

## 1. `pinneapple_physics`

### Objetivo

`pinneapple_physics` é o pacote onde um problema físico é definido e resolvido. Ele junta a especificação de EDPs (`pde_environment`: `ProblemSpec`, condições de contorno e iniciais, presets como Navier-Stokes, calor, Burgers, elasticidade, modelos RANS e uma base de famílias de EDP), o compilador de PINN (`pinn_solver`), o compilador SymPy para autograd (`symbolic_pde`) e modelos de forma fechada (`closed_form`). O `__init__` reexporta esses nomes e mantém apelidos antigos (`environment`, `pinn`, `symbolic`) por compatibilidade.

O caminho principal é: montar um `ProblemSpec` (à mão, com `define_problem` ou com `get_preset`), compilar as perdas com `compile_physics`/`compile_problem` e treinar com `solve_pde`, um laço Adam próprio que amostra pontos de colocação a cada passo. `pipeline(task)` junta preset, modelo do registro de `pinneapple_neural` e `solve_pde` numa chamada e devolve um `PhysicsPipeline` com `.predict()`. Condições com `selector_type="tag"` exigem geometria real (por exemplo `STLDomainBatchBuilder` de `pinneapple_design`); sem ela, `solve_pde` levanta `TagConditionsUnresolved` antes de treinar. Os pontos de entrada `solve`/`compare` (de `solving.py`) e `Experiment` também são reexportados: é por aqui que `pp.solve(problem, "analytic")` usa as soluções de `closed_form`.

Além do núcleo PINN, o pacote traz camadas que julgam previsões pela física, sem solução de referência: `darcy_checks` e `maxwell_checks` (verificações usadas pelo comando `pinneapple check` e por relatórios de confiança APPROVED/REVIEW/REJECT), `physics_audit` (premissas de cada modelo, números adimensionais, monitor de validade, atribuição de falhas, recomendações e roteamento) e `multiscale` (filtros LES, fechamentos, resíduos multiescala e seleção de modelos sensível à escala). `non_hermitian` cobre dímeros PT-simétricos e pontos excepcionais; `weather` é um caso de uso de previsão global com ERA5 do WeatherBench2.

Esta parte do mapa cobre o `__init__` do pacote e os submódulos `closed_form`, `darcy_checks`, `maxwell_checks`, `multiscale`, `non_hermitian`, `physics_audit`, `pinn_solver` e `weather`. Os demais submódulos (`pde_environment`, `symbolic_pde`, `solving`, `ensemble`, etc.) são descritos nas outras partes.

Esta parte cobre quatro submódulos de `pinneapple_physics`. `astro` é um observatório em NumPy próprio (escalas de tempo, efeméride JPL, posições aparentes, fenômenos, satélites, pequenos corpos, estrelas, lentes gravitacionais, fichas bilíngues) com modelos de IA conferidos contra referências independentes. `blackhole` simula acreção em torno de um buraco negro e treina um previsor de 'tempo do buraco negro' com horizonte de confiança. `pde_environment` define a linguagem de problemas do framework (`ProblemSpec`, condições, `ProblemBuilder`) e o catálogo de presets registrados usados por `get_preset` e pelos solvers. `symbolic_pde` compila EDPs escritas em SymPy ou em texto para resíduos PyTorch e condições de contorno, e liga sistemas de texto a um `PINNProblem`.

Esta parte cobre a camada de uso direto do `pinneapple_physics`: a descrição única de um problema (`PhysicalProblem`), a porta de entrada `pp.solve`/`pp.compare` com registro de métodos, `pp.compile` (reuso de derivadas no resíduo), `Experiment` reprodutível com métricas padronizadas e envio a MLflow/W&B, ensembles adaptativos de modelos físicos com aprendizado online, geração de scripts FDM e FEniCS independentes, ponte para o DeepXDE, problemas de artigos (PINNsFormer, X-TFC) e módulos de domínio: térmica de placa de circuito (PCB Hotspot), elasticidade axissimétrica, desgaste de Archard e projeção conservativa.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 166 | 43.959 | 169 | 763 | 1.075 (105 arquivos) | 52 |

Pasta de testes dedicada: `tests/pinneapple_physics` (48 funções de teste).

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `advection_diffusion_1d` | Banco de teste 1D periódico de advecção-difusão com solução exata espectral, esquemas de diferenças finitas e resíduo da EDP. | 1 | 1 | 13 |
| `astro` | Astronomia posicional e astrofísica do observatório PINNeAPPle, validada contra NOVAS, Horizons e skyfield, com substitutos neurais e fichas para o público. | 24 | 216 | 71 |
| `blackhole` | Simulação hidrodinâmica de acreção em buraco negro, dataset, previsor U-Net com horizonte de confiança, avaliação de habilidade, traçado de raios e visualização 3D. | 12 | 60 | 19 |
| `closed_form` | Modelos de engenharia em forma fechada (aletas, vigas e fadiga, difusão em membrana, Helmholtz, dissipadores, PCB, escoamento em tubos, bocal de foguete, Burgers exato, etc.) para verdade de referência rápida sem resolver EDP. | 19 | 91 | 103 |
| `codegen` | Gera arquivos Python independentes (FDM em NumPy ou FEniCS/dolfinx) a partir de um `ProblemSpec`. | 3 | 4 | 13 |
| `compile_api` | `pp.compile`: prepara um problema para resolver mais rápido reaproveitando derivadas dentro de cada avaliação da perda. | 1 | 3 | 5 |
| `conservation` | Projeção exata que força a integral ponderada da saída de um modelo a um valor conservado. | 1 | 3 | 6 |
| `darcy_checks` | Verificações físicas sem referência de campos de Darcy 2-D (resíduo de volumes finitos, limites certificados de erro, balanço de fluxo, contorno, envelope de coeficientes) e decisão de implantação. | 1 | 31 | 26 |
| `deepxde_bridge` | Traduz um `ProblemSpec` em objetos reais do DeepXDE e treina com ele, como backend alternativo. | 2 | 4 | 9 |
| `ensemble` | Ensemble adaptativo de modelos físicos (PINN, operadores, GNN, solvers) escolhidos ou combinados online pelo erro observado. | 1 | 9 | 68 |
| `ensemble_viz` | Gera um GIF animado de uma execução do ensemble adaptativo. | 1 | 1 | 3 |
| `experiment` | `Experiment`: execução reprodutível de um método num problema, com métricas e proveniência. | 1 | 3 | 0 |
| `maxwell_checks` | Verificações das equações de Maxwell e das leis de balanço (Gauss, Faraday, Ampère-Maxwell, continuidade, Poynting, PEC, reciprocidade, passividade) em campos de qualquer origem. | 1 | 19 | 51 |
| `metrics` | Métricas de erro padrão da biblioteca (`pp.metrics`), por campo e agregadas. | 1 | 11 | 36 |
| `multiscale` | Física multiescala para ensembles adaptativos: filtros LES, solver de vorticidade 2-D, fechamentos de subfiltro, resíduos multiescala, consistência entre escalas e roteamento sensível à escala. | 11 | 76 | 45 |
| `non_hermitian` | Dímeros e cadeias PT-simétricos, pontos excepcionais e a verificação simbólica de leis de sintonia do EP contra um teorema de impossibilidade. | 4 | 19 | 8 |
| `online_learning` | Algoritmos de aprendizado online com especialistas (Fixed-Share, AdaHedge, conformal adaptativo) usados pelos métodos adaptativos. | 1 | 4 | 0 |
| `paper_problems` | Problemas de teste do PINNsFormer em grade (x, t) com solução analítica. | 1 | 7 | 5 |
| `paper_problems_xtfc` | Problemas de teste do artigo X-TFC com equação, domínio, solução exata e configurações relatadas. | 1 | 4 | 10 |
| `pcb` | Modelo térmico de placa de circuito montada (motor do app PCB Hotspot): solver de volumes finitos, what-if e calibração. | 3 | 10 | 16 |
| `pde_environment` | Especificação de problemas (`ProblemSpec`, condições, `ProblemBuilder`), catálogo de presets por domínio, resíduos de turbulência e base de famílias de EDP. | 28 | 154 | 154 |
| `physical_problem` | `PhysicalProblem`: descrição única de um problema físico com parâmetros, unidades, quantidades de interesse e fingerprint. | 1 | 3 | 12 |
| `physics_audit` | Auditoria física de modelos: premissas e condições de validade, números adimensionais, monitor de validade sem verdade de referência, atribuição de falhas, recomendações, grafo de conhecimento e roteamento. | 8 | 65 | 37 |
| `pinn_solver` | Compilador de PINN: traduz um `ProblemSpec` em funções de perda, mais DoMINO (decomposição de domínio), PINN estocástico, fábrica SymPy legada e leitura de dados UPD. | 19 | 41 | 121 |
| `solid` | Pós-processamento de mecânica dos sólidos por autograd (deformações, tensões, resíduo de equilíbrio, von Mises). | 2 | 6 | 8 |
| `solving` | `pp.solve` e `pp.compare`: ponto de entrada único para resolver um problema por qualquer método registrado e comparar métodos. | 1 | 12 | 71 |
| `symbolic_pde` | Compilador de EDPs simbólicas (SymPy e texto) para resíduos PyTorch, com condições de contorno rígidas e suaves e ponte para PINN. | 6 | 24 | 17 |
| `tracking` | Adaptador que envia um `ExperimentResult` ao MLflow ou ao Weights & Biases. | 1 | 3 | 3 |
| `tribology` | Desgaste por deslizamento de uma barra contra contraface pela lei de Archard. | 2 | 3 | 20 |
| `weather` | Caso de uso de previsão do tempo global: dados ERA5 do WeatherBench2, U-Net esférica, treino por rollout, métricas de habilidade, eventos extremos e animações. | 7 | 38 | 8 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_physics`

Ponto de entrada do pacote: reexporta `pde_environment`, `pinn_solver`, `symbolic_pde`, `closed_form`, `solve`/`compare` e `Experiment`, mantém apelidos antigos e define os auxiliares `define_problem`, `compile_physics`, `solve_pde` e `pipeline`.

<small>`pinneapple_physics/__init__.py` · 604 linhas · testes: 206</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TagConditionsUnresolved` | classe | `()` | Erro (`ValueError`) levantado por `solve_pde` quando há condições `selector_type="tag"` sem pontos e máscaras fornecidos pelo chamador, para não treinar sem essas condições. |
| `PhysicsPipeline` | classe | `(model, spec, history)` | Resultado de `pipeline`: guarda modelo treinado, `ProblemSpec` e histórico, e oferece `.predict(x)` que devolve um array numpy. |
| `compile_physics` | função | `(spec, **kwargs)` | Atalho para `compile_problem(spec, **kwargs)`: compila o `ProblemSpec` em perdas PINN ponderadas. |
| `identify` | função | `(description)` | Atalho para `identify_pde`: identifica a família de EDP a partir de uma descrição em linguagem natural. |
| `define_problem` | função | `(preset=None, **spec_kwargs) -> 'ProblemSpec'` | Monta um `ProblemSpec`: a partir de um preset nomeado (sobrescrevendo atributos) ou diretamente dos argumentos. |
| `solve_pde` | função | `(spec, model, *, epochs=5000, device='cpu', lr=0.001, n_collocation=2048, seed=0, weights…)` | Treina um modelo num `ProblemSpec` com um laço Adam próprio, amostrando colocação e pontos de condição (`all`/`callable`) a cada passo; aceita pontos fixos `x_col` e máscaras de tags. |
| `pipeline` | função | `(task, *, architecture='modified_mlp', hidden_dim=64, n_layers=4, epochs=2000, device='cp…)` | Uma chamada para tarefa nomeada: `get_preset` + `ModelRegistry.build` + `solve_pde`, devolvendo um `PhysicsPipeline`. |

#### `pinneapple_physics.advection_diffusion_1d`

Advecção-difusão periódica 1D em [0, 2π) com solução exata por modos de Fourier, esquemas upwind e Lax-Wendroff e resíduo da EDP de qualquer campo previsto; serve de banco de teste para ensembles.

<small>`pinneapple_physics/advection_diffusion_1d.py` · 80 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AdvectionDiffusion1D` | classe | `(nx, nt, t_end)` | Banco de teste: sorteia casos, dá a solução exata, roda esquemas FD e calcula o resíduo da EDP de um campo previsto. |

#### `pinneapple_physics.astro`

Pacote de astronomia posicional e astrofísica do observatório PINNeAPPle, em NumPy próprio e validado contra USNO NOVAS, JPL Horizons e skyfield; reexporta `Observer`, `Time`, `apparent` e afins.

<small>`pinneapple_physics/astro/__init__.py` · 54 linhas · testes: 71</small>

#### `pinneapple_physics.astro.astrometry`

Posições aparentes do Sol, Lua, planetas e estrelas de catálogo para um observador na Terra (tempo de luz, deflexão, aberração), mais refração, massa de ar e extinção. [sem teste direto]

<small>`pinneapple_physics/astro/astrometry.py` · 194 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Observer` | classe | `(lat, lon, elevation_m, pressure_mbar, temperature_c, name)` | Um lugar na Terra: latitude/longitude geodésicas, altitude e o tempo (pressão, temperatura) usado pela refração; dá a posição em ITRS e GCRS. |
| `Apparent` | classe | `(body, ra, dec, ra_astrometric, dec_astrometric, distance_km, light_time_s, range_rate_km…)` | Resultado de `apparent`: RA/Dec aparentes e astrométricas, distância, tempo de luz, taxa de distância, altitude/azimute e ângulo horário. |
| `refraction_deg` | função | `(alt_true_deg, pressure_mbar=1010.0, temperature_c=10.0)` | Refração atmosférica (graus) para uma altitude verdadeira, fórmula de Saemundsson com correção de pressão e temperatura; zero abaixo de -1 grau. |
| `airmass` | função | `(alt_apparent_deg)` | Massa de ar relativa (Kasten & Young 1989), finita até o horizonte e infinita abaixo dele. |
| `extinction_mag` | função | `(alt_apparent_deg, k_v=0.25)` | Extinção atmosférica em magnitudes (banda V) para um coeficiente `k_v` por massa de ar. |
| `apparent` | função | `(body, t, observer=None, eph=None, *, deflection=True, refraction=True) -> Apparent` | Posição aparente de um corpo do Sistema Solar num instante (escalar ou vetor), geocêntrica se não houver observador. |
| `star_apparent` | função | `(ra_j2000, dec_j2000, t, observer=None, *, pm_ra_mas=0.0, pm_dec_mas=0.0, parallax_mas=0.…)` | Posições aparentes de estrelas de catálogo (ICRS J2000) num instante, com movimento próprio e paralaxe. |

#### `pinneapple_physics.astro.conditions`

Condições de observação a partir de previsão numérica aberta (Open-Meteo, CC BY 4.0): ensemble de nebulosidade, probabilidade de céu limpo, calibração e modo de céu 'perfeito'/'real'. Depende de rede.

<small>`pinneapple_physics/astro/conditions.py` · 126 linhas · testes: 25</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `fetch_ensemble` | função | `(lat, lon, model='icon_seamless', hours=48, timeout=15.0)` | Baixa da API de ensemble da Open-Meteo as variáveis horárias (nuvens, umidade, vento) de um modelo para um lugar. |
| `members` | função | `(resp, var) -> np.ndarray` | Matriz (membros, horas) de uma variável do ensemble, com a rodada de controle e cada membro. |
| `summarize` | função | `(resp, clear_threshold=30.0) -> dict` | Resume o ensemble: probabilidade de céu limpo por hora (nuvens abaixo de um limiar), umidade e vento em altitude. |
| `calibrate` | função | `(p_forecast, observed_clear, bins=10) -> dict` | Confiabilidade de probabilidades passadas de céu limpo contra o observado, com mapa de correção por faixa de probabilidade. |
| `fetch_cloud_forecast` | função | `(lat, lon, days=3, timeout=8.0) -> dict` | Previsão horária de cobertura de nuvens para os próximos dias pela API da Open-Meteo; levanta erro sem rede. |
| `cloud_at` | função | `(forecast, iso_utc) -> float \| None` | Cobertura de nuvens num instante, interpolada linearmente na previsão horária; None fora dela. |
| `sky_view` | função | `(mode, forecast, iso_utc) -> dict` | O que o céu deixa ver numa sessão: modo 'perfect' ignora o tempo, 'real' usa a previsão e esconde a vista quando nublado. |

#### `pinneapple_physics.astro.conjunction`

Triagem de conjunções: aproximação mínima no movimento relativo de Clohessy-Wiltshire, probabilidade de colisão 2D e propagação de dois corpos com J2.

<small>`pinneapple_physics/astro/conjunction.py` · 160 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ClosestApproach` | classe | `(t_tca_s, position_km, velocity_km_s, miss_distance_km, relative_speed_km_s, perpendicula…)` | Resultado da aproximação mínima: instante, posição e velocidade relativas, distância de passagem e velocidade relativa. |
| `cw_mean_motion` | função | `(a_km, mu_km3_s2=398600.4418) -> float` | Movimento médio n = sqrt(mu/a^3) da órbita circular de referência. |
| `cw_closest_approach` | função | `(n, state0, duration_s, n_samples=2000) -> ClosestApproach` | Aproximação mínima num intervalo para um estado relativo de Clohessy-Wiltshire em torno de uma órbita de referência. |
| `collision_probability_2d` | função | `(xm, zm, sigma_x, sigma_z, r_c) -> float` | Probabilidade de uma gaussiana 2D (média e desvios dados) cair dentro de um disco de raio `r_c`. |
| `collision_probability_centered` | função | `(r_c, sigma) -> float` | Forma fechada (CDF de Rayleigh) da probabilidade de colisão para gaussiana isotrópica centrada. |
| `screen_cw_conjunction` | função | `(n, state0, *, duration_s=None, n_samples=2000, hard_body_radius_km=0.01, sigma_km=1.0) -…)` | Aproximação mínima mais probabilidade de colisão para um encontro CW (um período de referência por padrão). |
| `two_body_j2_rhs` | função | `(mu, J2, Re)` | Lado direito f(t, estado) do movimento de dois corpos com aceleração de J2. |
| `propagate_two_body_j2` | função | `(state0, t_eval, *, mu=398600.4418, J2=0.00108262668, Re=6378.137, rtol=1e-11, atol=1e-08…)` | Propaga estados de dois corpos com J2 a partir de um estado inicial, com integrador DOP853. |

#### `pinneapple_physics.astro.decay`

Decaimento orbital de satélites em órbita baixa por arrasto atmosférico: integração de referência, envelope Monte Carlo, um PINN de decaimento com verificações e comparação com SGP4. Usa NRLMSIS (`pymsis`) quando disponível.

<small>`pinneapple_physics/astro/decay.py` · 258 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DensityTable` | classe | `(h_km=None, **kw)` | Log-densidade atmosférica numa grade de altitudes, interpolada em NumPy e PyTorch para avaliar modelos rápido. |
| `exponential_density` | função | `(h_km)` | Densidade atmosférica por um modelo exponencial em faixas de altitude (tabela interna), usada sem NRLMSIS. |
| `msis_available` | função | `() -> bool` | Diz se o pacote opcional `pymsis` (NRLMSIS) pode ser importado. |
| `orbit_mean_density` | função | `(h_km, date='2026-10-13T00:00', f107=150.0, f107a=150.0, ap=15.0, inclination_deg=51.6, n…)` | Densidade NRLMSIS 2.0 média ao longo da órbita (latitudes e horas locais) em cada altitude. |
| `ballistic_from_bstar` | função | `(bstar_per_er) -> float` | Coeficiente balístico B = C_D A / m implícito num B* do SGP4. |
| `reference_decay` | função | `(h0_km, B, days, table, n=200, reentry_km=150.0)` | Altitude versus tempo por integração Runge-Kutta adaptativa da EDO de decaimento. |
| `envelope` | função | `(h0_km, B, days, *, f107=150.0, f107a=None, ap=15.0, date='2026-10-13T00:00', inclination…)` | Faixa Monte Carlo do decaimento sobre entradas incertas (fluxo solar, atividade geomagnética, coeficiente balístico): mediana e percentis 5 e 95. |
| `train_decay_pinn` | função | `(h0_km=415.0, B=0.005, days=60.0, *, f107=150.0, ap=15.0, date='2026-10-13T00:00', inclin…)` | Treina o PINN de decaimento e devolve relatório com previsão, referência, envelope, verificações e veredito. |
| `sgp4_decay` | função | `(line1, line2, days=30.0, step_min=10.0)` | Altitude média versus tempo prevista pelo SGP4 a partir de um TLE (média por órbita). |
| `compare_with_sgp4` | função | `(line1, line2, days=30.0, f107=70.0, ap=8.0, B=ISS_B) -> dict` | Coloca lado a lado o decaimento deste modelo (NRLMSIS e B físico) e o previsto pelo SGP4 do mesmo TLE. |

#### `pinneapple_physics.astro.ephemeris`

Efeméride do Sistema Solar sobre um kernel JPL DE (domínio público) lido por `.spk`; localiza o kernel por variável de ambiente ou pelo pacote `skyfield_data`.

<small>`pinneapple_physics/astro/ephemeris.py` · 77 linhas · testes: 71</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Ephemeris` | classe | `(path=None)` | Efeméride sobre um kernel SPK: estados baricêntricos e relativos (km, km/s) dos corpos por data TDB. |
| `find_kernel` | função | `() -> str` | Acha o arquivo do kernel DE: variável `PINNEAPPLE_DE_KERNEL` ou o `de421.bsp` do pacote `skyfield_data`. |
| `default_ephemeris` | função | `(path=None) -> Ephemeris` | Cria uma `Ephemeris` com o kernel padrão (ou o caminho dado). |

#### `pinneapple_physics.astro.facts`

Fichas de objetos do céu para o público geral, em inglês e português: números com fonte, curiosidades, história, missões, quiz, busca e 'como ver hoje'.

<small>`pinneapple_physics/astro/facts.py` · 746 linhas · testes: 30</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `sources` | função | `() -> dict` | Todas as fontes que um número ou curiosidade pode citar (título, editor, URL, licença). |
| `ids` | função | `() -> dict` | Todos os objetos com ficha e seu tipo (planetas, estrelas, Messier, constelações, satélites). |
| `resolve` | função | `(ref) -> str` | Converte um id ou referência do observatório no id da ficha; levanta KeyError se não houver ficha. |
| `names` | função | `(oid) -> dict` | Nomes de exibição de uma ficha em inglês e português. |
| `numbers` | função | `(oid) -> dict` | Números-chave de uma ficha, cada um com valor, unidade e fonte. |
| `comparison` | função | `(key, number, lang='en') -> str \| None` | Comparação cotidiana calculada a partir de um número (fonte: calculado). |
| `quiz` | função | `(oid, lang='en', seed=0, n=3) -> list[dict]` | Quiz curto de múltipla escolha gerado a partir dos números da ficha. |
| `card` | função | `(ref, lang='en') -> dict` | Ficha completa de um objeto no idioma pedido: nome, resumo, números, curiosidades com fonte, história, missões, quiz e fontes. |
| `list_cards` | função | `(kind=None, lang='en') -> list[dict]` | Lista as fichas (id, tipo, nome), opcionalmente de um tipo. |
| `search` | função | `(query, lang=None, limit=12) -> list[dict]` | Busca fichas pelo nome em inglês ou português, id ou designação, ignorando acentos e caixa. |
| `did_you_know` | função | `(lang='en', seed=None) -> dict` | Uma curiosidade aleatória das fichas escritas à mão, para dica de abertura. |
| `validate` | função | `() -> list[str]` | Lista problemas do conjunto de fichas (fonte por número, dois idiomas, número de curiosidades, história, fontes conhecidas). |
| `position` | função | `(ref)` | RA/Dec J2000 de um objeto fixo (estrela, Messier, rótulo de constelação); None para corpos do Sistema Solar e satélites. |
| `tonight` | função | `(ref, observer, t, hours=24.0, step_min=10.0, eph=None) -> dict \| None` | Como ver um objeto nas próximas horas: altitude atual, nascer, culminação, ocaso e melhor altura com céu escuro. |

#### `pinneapple_physics.astro.frames`

Rotações de referencial: viés, precessão (IAU 2006), nutação (IAU 1980, termos acima de 5 mas), posição geodésica WGS84, coordenadas horizontais e eclípticas.

<small>`pinneapple_physics/astro/frames.py` · 202 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `fundamental_args` | função | `(T)` | Argumentos de Delaunay (D, M, M', F, Omega) para séculos julianos TT desde J2000. |
| `nutation` | função | `(jd_tt)` | Nutação em longitude e obliquidade (rad). |
| `mean_obliquity` | função | `(jd_tt)` | Obliquidade média da eclíptica (IAU 2006) numa data TT, em radianos. |
| `rot_x` | função | `(a)` | Matriz de rotação em torno do eixo x (vetorizada no ângulo). |
| `rot_y` | função | `(a)` | Matriz de rotação em torno do eixo y (vetorizada no ângulo). |
| `rot_z` | função | `(a)` | Matriz de rotação em torno do eixo z (vetorizada no ângulo). |
| `precession_matrix` | função | `(jd_tt)` | Matriz de precessão J2000 média para equador/equinócio médio da data (IAU 2006). |
| `nutation_matrix` | função | `(jd_tt)` | Matriz do equador médio da data para o verdadeiro. |
| `npb_matrix` | função | `(jd_tt)` | Matriz combinada GCRS para equador e equinócio verdadeiros da data (N P B). |
| `ecliptic_matrix` | função | `(jd_tt=None)` | Matriz GCRS para eclíptica (da data, ou a eclíptica média J2000 se a data for None). |
| `geodetic_to_itrs` | função | `(lat_deg, lon_deg, height_m=0.0)` | Posição do observador (km) no referencial terrestre a partir de latitude, longitude e altitude. |
| `itrs_to_geodetic` | função | `(r)` | Inversa de `geodetic_to_itrs` por iteração de Bowring. |
| `radec` | função | `(v)` | Vetores para RA, Dec (graus) e norma. |
| `from_radec` | função | `(ra_deg, dec_deg)` | Vetor unitário a partir de RA e Dec em graus. |
| `equatorial_to_horizontal` | função | `(ha_rad, dec_rad, lat_rad)` | Ângulo horário, declinação e latitude para altitude e azimute. |
| `horizontal_to_equatorial` | função | `(alt_deg, az_deg, lat_deg)` | Altitude e azimute para ângulo horário e declinação. |
| `angular_separation` | função | `(ra1, dec1, ra2, dec2)` | Separação angular de grande círculo (fórmula de Vincenty). |
| `position_angle` | função | `(ra1, dec1, ra2, dec2)` | Ângulo de posição (norte por leste) do ponto 2 visto do ponto 1. |

#### `pinneapple_physics.astro.journey`

Geometria de uma 'viagem' do observador até um planeta, lua, o Sol ou uma estrela: posições heliocêntricas, distância, tempos de viagem, magnitude e cena 3D.

<small>`pinneapple_physics/astro/journey.py` · 193 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `galactic` | função | `(v)` | Coordenadas galácticas (l, b) de um vetor ICRS. |
| `body_position` | função | `(body, t, eph=None) -> np.ndarray` | Posição heliocêntrica ICRS de um corpo do Sistema Solar (geométrica, sem tempo de luz). |
| `observer_position` | função | `(observer, t, eph=None) -> np.ndarray` | Posição heliocêntrica ICRS de um lugar na Terra (Terra pela efeméride mais o local em GCRS). |
| `star_position` | função | `(ra_deg, dec_deg, parallax_mas) -> np.ndarray` | Posição heliocêntrica de uma estrela a partir da direção J2000 e da paralaxe. |
| `travel_times` | função | `(distance_km) -> dict` | Segundos para percorrer uma distância nas velocidades da tabela `SPEEDS`. |
| `apparent_magnitude_star` | função | `(abs_mag, distance_km)` | Magnitude aparente m = M + 5 log10(d / 10 pc). |
| `angular_diameter_deg` | função | `(radius_km, distance_km)` | Diâmetro angular (graus) de um corpo de raio dado a uma distância. |
| `plan` | função | `(observer, target, t, standoff=None, n_profile=60, eph=None) -> dict` | Monta a viagem do observador até um alvo numa data: distância, tempo de luz, perfil de magnitude e tamanho. |
| `scene` | função | `(t, observer=None, max_parallax_err=0.2, eph=None) -> dict` | Tudo o que a viagem 3D desenha: planetas, Sol e Lua com raios, e as estrelas brilhantes com paralaxe utilizável. |

#### `pinneapple_physics.astro.lensing`

Demonstrações de lente gravitacional no céu real: um buraco negro de Schwarzschild na frente das estrelas, renderizado com o traçador de geodésicas de `blackhole.raytrace`, mais fórmulas fechadas (deflexão, raio de Einstein, sombra vs EHT).

<small>`pinneapple_physics/astro/lensing.py` · 366 linhas · testes: 32</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `deflection_angle_rad` | função | `(mass_kg, impact_parameter_m)` | Deflexão da luz em campo fraco, 4GM/(c^2 b). |
| `einstein_radius_rad` | função | `(mass_kg, d_lens_m, d_source_m)` | Raio de Einstein (rad) para massa e distâncias da lente e da fonte. |
| `shadow_diameter_rad` | função | `(mass_kg, distance_m)` | Diâmetro angular da sombra de Schwarzschild, 2 sqrt(27) GM/(c^2 D). |
| `eht_comparison` | função | `() -> list[dict]` | Compara o diâmetro de sombra previsto com o anel observado pelo EHT para cada objeto da tabela interna, em sigmas. |
| `sun_deflection_arcsec` | função | `(elongation_deg)` | Deflexão da luz estelar pelo Sol a uma elongação dada (Eddington 1919). |
| `star_splat` | função | `(v, mag_limit=6.5, sigma_deg=0.1, mag_white=1.5)` | Largura gaussiana e brilho de pico para desenhar uma estrela de magnitude dada. |
| `catalog_sky` | função | `(width=4096, height=2048, mag_limit=6.5, milky_way=0.32, star_sigma_deg=0.1) -> np.ndarray` | Imagem RGB equiretangular do catálogo de estrelas brilhantes sobre o brilho da Via Láctea. |
| `tone_map` | função | `(img, exposure=1.6, gamma=0.75) -> np.ndarray` | Mapeia radiância linear do céu para valores de exibição em [0, 1] com ombro suave e gama. |
| `bright_stars_in_field` | função | `(ra_deg, dec_deg, radius_deg, mag_limit=3.5) -> int` | Conta estrelas do catálogo mais brilhantes que um limite dentro de um raio. |
| `lensed_sky` | função | `(ra_deg, dec_deg, fov_deg=30.0, width=480, height=270, r_cam=30.0, sky=None, supersample=…)` | Renderiza o céu real em torno de (ra, dec) visto através de um buraco negro de Schwarzschild. |
| `trace_thin_disc` | função | `(width=640, height=360, inclination_deg=80.0, r_cam=60.0, fov_deg=32.0, max_crossings=3, …)` | Traçado de raios exato para trás num disco equatorial fino em torno de um buraco negro de Schwarzschild. |
| `disc_image` | função | `(tr, r_in=6.0, r_out=20.0, doppler=True, t_peak=3900.0, exposure=2.4, gamma=0.6, opacity=…)` | Cor e intensidade do disco para os raios traçados, com Doppler e desvio gravitacional opcionais. |
| `accretion_view` | função | `(ra_deg, dec_deg, width=640, height=360, inclination_deg=86.0, r_cam=60.0, fov_deg=42.0, …)` | Buraco negro com disco de acreção fino na frente do céu real, com geodésicas exatas, Doppler e bloom. |
| `lensing_view` | função | `(ra_deg, dec_deg, fov_deg=60.0, width=480, height=270, r_cam=120.0, sky=None, disc='auto'…)` | Vista pronta para exibição de um buraco negro na frente do céu, com ou sem disco ('auto'). |

#### `pinneapple_physics.astro.nbody`

Integradores clássicos de N corpos para Sol-planetas(-Lua), com diagnósticos de conservação de energia e momento angular. [sem teste direto]

<small>`pinneapple_physics/astro/nbody.py` · 116 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `initial_state` | função | `(jd_tdb, bodies=SOLAR_SYSTEM, eph=None)` | Posições, velocidades baricêntricas e GM dos corpos numa data, lidos da efeméride. |
| `accelerations` | função | `(r, mu)` | Acelerações gravitacionais mútuas de N corpos. |
| `energy` | função | `(r, v, mu)` | Energia total (por unidade de G) do sistema de N corpos. |
| `angular_momentum` | função | `(r, v, mu)` | Momento angular total do sistema, ponderado por GM. |
| `integrate` | função | `(r0, v0, mu, t_end_s, dt_s, method='yoshida4', n_out=200)` | Integra de t=0 a `t_end_s` (Yoshida-4 por padrão) e devolve trajetórias e históricos de erro de energia e momento angular. |

#### `pinneapple_physics.astro.neural_propagator`

Propagador aprendido para asteroides próximos da Terra: uma rede corrige o desvio da órbita kepleriana (no referencial R/T/N) para muitos corpos, com relatório de confiança.

<small>`pinneapple_physics/astro/neural_propagator.py` · 240 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NeuralPropagator` | classe | `(net=None, xm=None, xs=None, ys=None, info=None)` | Rede que prevê a correção à órbita kepleriana de um asteroide; treina, prevê, salva e carrega. |
| `sample_population` | função | `(n, seed=0)` | Sorteia elementos osculadores parecidos com os de NEOs. |
| `kepler_positions` | função | `(body, jd)` | Posições e velocidades keplerianas (ICRF) de um `SmallBody` nas datas pedidas. |
| `features` | função | `(body, tau)` | Entradas da rede: elementos, ângulos em seno/cosseno, tempo normalizado e anomalia média kepleriana. |
| `rtn_basis` | função | `(rk, vk)` | Vetores unitários radial, transversal e normal da órbita kepleriana. |
| `make_dataset` | função | `(n_bodies=160, n_t=60, seed=0, cache=None)` | Monta entradas e alvos: o desvio da órbita kepleriana no referencial R/T/N. |
| `tisserand` | função | `(r_km, v_km_s, a_jup_au=5.2026)` | Parâmetro de Tisserand em relação a Júpiter a partir de estados heliocêntricos. |
| `evaluate` | função | `(model, bodies, n_t=60)` | Erros do propagador aprendido e de Kepler sozinho contra o integrador completo, por corpo, com tempo de execução. |
| `trust_report` | função | `(folder=MODEL_DIR, real=None) -> dict` | Veredito da rede distribuída a partir da avaliação em corpos não vistos (precisa superar Kepler por margem fixa). |

#### `pinneapple_physics.astro.observing`

Sessão de observação: visibilidade de um alvo na noite, nascer/trânsito/ocaso, brilho do céu (Krisciunas & Schaefer), óptica do telescópio, slews GoTo, rotação de campo e recortes de survey HiPS.

<small>`pinneapple_physics/astro/observing.py` · 356 linhas · testes: 21</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Telescope` | classe | `(name, aperture_mm, focal_mm, kind, central_obstruction)` | Telescópio: nome, abertura, focal, tipo e obstrução central. |
| `Eyepiece` | classe | `(name, focal_mm, afov_deg)` | Ocular: nome, focal e campo aparente. |
| `Camera` | classe | `(name, width_mm, height_mm, pixel_um)` | Câmera: nome, tamanho do sensor e tamanho do pixel. |
| `target_altaz` | função | `(target, t, observer, refraction=True, eph=None)` | Altitude, azimute e RA/Dec da data de um corpo ou de um par (ra, dec) J2000. |
| `rise_transit_set` | função | `(target, observer, jd_start, days=1.0, eph=None) -> dict` | Nascer, trânsito superior e ocaso de corpos, estrelas e objetos de céu profundo num intervalo. |
| `extinction` | função | `(alt_apparent_deg, k=None) -> dict` | Extinção por banda numa altitude aparente (k vezes massa de ar de Kasten-Young). |
| `nl_to_mag` | função | `(b_nl)` | Brilho superficial de nanolamberts para magnitude V por segundo de arco quadrado. |
| `mag_to_nl` | função | `(m)` | Inversa de `nl_to_mag`: magnitude por arcsec^2 para nanolamberts. |
| `moon_brightness_nl` | função | `(moon_phase_angle_deg, moon_sep_deg, moon_alt_deg, target_alt_deg, k=0.172)` | Brilho do céu acrescentado pelo luar no alvo (Krisciunas & Schaefer 1991). |
| `dark_sky_nl` | função | `(target_alt_deg, k=0.172, b_zenith_nl=79.0)` | Brilho natural do céu noturno no alvo (KS91). |
| `twilight_mag` | função | `(sun_alt_deg)` | Brilho aproximado do zênite no crepúsculo a partir da altitude do Sol (ajuste linear). |
| `sky_brightness` | função | `(sun_alt_deg, moon_phase_angle_deg, moon_sep_deg, moon_alt_deg, target_alt_deg, k=0.172, …)` | Brilho do céu no alvo: céu escuro, luar e crepúsculo somados. |
| `visibility` | função | `(target, observer, t, hours=24.0, step_min=5.0, min_alt=30.0, eph=None) -> dict` | A noite de um alvo: amostras de altitude, massa de ar, Sol, Lua, brilho do céu, eventos e a melhor janela. |
| `optics` | função | `(telescope, eyepiece=None, camera=None, naked_eye_limit=6.0, wavelength_nm=550.0) -> dict` | O que um telescópio mostra: aumento, campo, pupila de saída, limites de resolução e magnitude limite. |
| `axis_time` | função | `(delta_deg, rate_deg_s, accel_deg_s2) -> float` | Tempo para mover um eixo com perfil de velocidade trapezoidal (ou triangular). |
| `hour_angle_dec` | função | `(alt_deg, az_deg, lat_deg)` | Ângulo horário e declinação a partir de altitude e azimute. |
| `slew` | função | `(start_altaz, target_altaz, lat_deg, mount='altaz', rate_deg_s=4.0, accel_deg_s2=2.0, set…)` | Simula um slew GoTo de montagem alt-az ou equatorial, com flip de meridiano e recusa abaixo da altitude mínima. |
| `parallactic_angle` | função | `(lat_deg, ha_deg, dec_deg)` | Ângulo paralático no alvo. |
| `field_rotation_rate` | função | `(lat_deg, alt_deg, az_deg)` | Taxa de rotação de campo de uma montagem alt-az. |
| `survey_url` | função | `(ra_deg, dec_deg, fov_deg, width=512, height=512, rotation_deg=0.0, survey=SURVEY) -> str` | URL de um recorte JPEG de um survey HiPS (CDS hips2fits) em torno de (ra, dec). |
| `survey_cutout` | função | `(ra_deg, dec_deg, fov_deg, width=512, height=512, cache_dir=None, timeout=8.0, survey=SUR…)` | Bytes JPEG do recorte, com cache em disco; levanta OSError sem rede e sem cache. |

#### `pinneapple_physics.astro.orbits`

Órbitas de dois corpos: elementos osculadores a partir do vetor de estado, equação de Kepler, propagação e órbitas heliocêntricas dos planetas pela efeméride JPL.

<small>`pinneapple_physics/astro/orbits.py` · 121 linhas · testes: 27</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `solve_kepler` | função | `(M, e, tol=1e-14, maxiter=50)` | Anomalia excêntrica a partir da anomalia média (Newton) para órbitas elípticas. |
| `elements_from_state` | função | `(r, v, mu=GM_SUN) -> dict` | Elementos keplerianos osculadores a partir de posição e velocidade. |
| `state_from_elements` | função | `(el, mu=GM_SUN, dt_s=0.0)` | Posição e velocidade após um intervalo de movimento kepleriano puro. |
| `heliocentric_state` | função | `(body, jd_tdb, frame='ecliptic', eph=None)` | Posição e velocidade heliocêntricas de um corpo, na eclíptica J2000 ou em eixos ICRF. |
| `planet_elements` | função | `(body, jd_tdb, eph=None) -> dict` | Elementos heliocêntricos osculadores de um planeta (geocêntricos para a Lua). |
| `orbit_polyline` | função | `(body, jd_tdb, n=256, eph=None)` | Uma volta da órbita osculadora em UA, para desenhar. |
| `light_time_and_doppler` | função | `(range_km, range_rate_km_s, rest_wavelength_nm=656.281)` | Tempo de luz de ida e desvio Doppler relativístico de uma linha espectral. |

#### `pinneapple_physics.astro.phenomena`

Fenômenos de almanaque: nascer/trânsito/ocaso, crepúsculos, fase da Lua, magnitudes, elongações, oposições, equinócios, eclipses e trânsitos de Mercúrio e Vênus. [sem teste direto]

<small>`pinneapple_physics/astro/phenomena.py` · 435 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `find_roots` | função | `(f, jd0, jd1, step, tol=0.5 * SECOND)` | Todas as trocas de sinal de uma função de data num intervalo, refinadas até a tolerância. |
| `find_minima` | função | `(f, jd0, jd1, step, tol=1.0 * SECOND)` | Mínimos locais de uma função de data, refinados por seção áurea. |
| `horizon_altitude` | função | `(body, distance_km=None) -> float` | Altitude geométrica do centro do corpo no nascer/ocaso (refração padrão mais semidiâmetro). |
| `rise_transit_set` | função | `(body, observer, jd_start, days=1.0, eph=None)` | Nascer, trânsito e ocaso de um corpo do Sistema Solar num intervalo. |
| `twilight_events` | função | `(observer, jd_start, days=1.0, eph=None)` | Nascer/pôr do Sol e início/fim dos crepúsculos civil, náutico e astronômico. |
| `sky_state` | função | `(sun_alt_deg) -> str` | Classifica o céu pela altitude do Sol: dia, crepúsculo civil, náutico, astronômico ou noite. |
| `phase_angle` | função | `(body, t, eph=None)` | Ângulo Sol-corpo-Terra e distância Sol-corpo. |
| `illuminated_fraction` | função | `(body, t, eph=None)` | Fração iluminada do disco a partir do ângulo de fase. |
| `moon_phase` | função | `(t, eph=None) -> dict` | Fração iluminada, ângulo de fase, elongação, longitude de fase e ângulo do limbo brilhante da Lua. |
| `magnitude` | função | `(body, t, eph=None)` | Magnitude visual do Sol, Lua e planetas (Saturno sem o termo dos anéis). |
| `planet_events` | função | `(body, jd0, jd1, eph=None)` | Oposições, conjunções e, para Mercúrio e Vênus, maiores elongações num intervalo. |
| `lunar_phase_events` | função | `(jd0, jd1, eph=None)` | Datas de lua nova, quarto crescente, cheia e quarto minguante. |
| `seasons` | função | `(jd0, jd1, eph=None)` | Equinócios e solstícios num intervalo. |
| `lunar_eclipses` | função | `(jd0, jd1, eph=None)` | Eclipses lunares com tipo e magnitudes umbral e penumbral. |
| `solar_eclipses` | função | `(jd0, jd1, observer=None, eph=None)` | Eclipses solares (critério geocêntrico) com tipo aproximado e circunstâncias locais opcionais. |
| `planetary_transits` | função | `(jd0, jd1, eph=None)` | Trânsitos de Mercúrio e Vênus pelo disco do Sol (geocêntricos). |

#### `pinneapple_physics.astro.satellites`

Satélites terrestres a partir de TLEs públicos (CelesTrak): propagação SGP4 (pacote `sgp4`), TEME para referencial terrestre, ângulos de visada, sombra da Terra, magnitude e passagens visíveis. [sem teste direto]

<small>`pinneapple_physics/astro/satellites.py` · 246 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Satellite` | classe | `(name, line1, line2)` | Satélite definido por um TLE; dá magnitude padrão e estados TEME e ITRS via SGP4. |
| `gmst82` | função | `(t)` | GMST pela expressão IAU 1982 usada pelo referencial TEME do SGP4. |
| `parse_tle` | função | `(text) -> list[Satellite]` | Lê texto de TLEs (com ou sem linha de nome) e devolve uma lista de `Satellite`. |
| `fetch_tles` | função | `(group='stations', cache_dir=None, max_age_hours=6.0, timeout=15.0) -> tuple[list[Satelli…)` | Baixa os TLEs de um grupo do CelesTrak com cache em disco; informa erro quando não há rede nem cache. |
| `sun_itrs` | função | `(t, eph=None)` | Posição geocêntrica do Sol no referencial terrestre. |
| `look` | função | `(sat, observer, t, eph=None) -> dict` | Altitude, azimute, distância, ponto subsatélite, iluminação e magnitude, vetorizado no tempo. |
| `magnitude` | função | `(std_mag, range_km, phase_rad, sunlit=1.0)` | Brilho de esfera difusa em função da distância e do ângulo de fase. |
| `find_passes` | função | `(sat, observer, t0, days=1.0, min_alt=10.0, step_s=20.0, eph=None) -> list[dict]` | Passagens acima do horizonte num intervalo, com visibilidade e magnitude máxima. |

#### `pinneapple_physics.astro.smallbodies`

Asteroides e cometas: elementos do JPL Small-Body Database, propagador de alta precisão no campo do Sol, planetas e Lua (DE421, com termo relativístico), aproximações à Terra e validação.

<small>`pinneapple_physics/astro/smallbodies.py` · 222 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SmallBody` | classe | `(name, epoch_jd, a_au, e, i_deg, node_deg, peri_deg, M_deg, kind, neo, pha, H, diameter_k…)` | Asteroide ou cometa com elementos orbitais, classificação (NEO, PHA), magnitude H e dados de aproximação. |
| `Trajectory` | classe | `(body, jd0, jd1, eph=None, rtol=1e-11, gr=True)` | Uma integração DOP853 adaptativa com saída densa; posições em qualquer instante custam só uma interpolação. |
| `load_body` | função | `(key) -> SmallBody` | Carrega um corpo empacotado a partir do JSON do JPL SBDB. |
| `available_bodies` | função | `()` | Chaves dos corpos empacotados, NEOs primeiro. |
| `propagate` | função | `(body, jd_out, eph=None, rtol=1e-11, gr=True)` | Posições e velocidades heliocêntricas ICRF nas datas TDB pedidas. |
| `close_approaches` | função | `(body, jd0, jd1, target='earth', max_dist_au=0.2, step_days=0.25, eph=None, traj=None)` | Mínimos de distância a um alvo (Terra por padrão), refinados por seção áurea. |
| `horizons_vectors` | função | `(path=None)` | Lê uma tabela VECTORS do JPL Horizons (CSV). |
| `cad_reference` | função | `(body, year)` | Registro de aproximação do JPL para um ano, com a faixa de 3 sigma. |
| `validation_summary` | função | `(eph=None) -> dict` | Propagador contra referências independentes: Ceres vs Horizons e a passagem de Apophis em 2029 vs a tabela do JPL. |

#### `pinneapple_physics.astro.spaceweather`

Clima espacial para observadores: previsão de Kp com incerteza calibrada (rede de quantis conformalizada) e probabilidade de ver aurora.

<small>`pinneapple_physics/astro/spaceweather.py` · 193 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `load_history` | função | `()` | Carrega o histórico empacotado de Kp e F10.7. |
| `geomagnetic_latitude` | função | `(lat_deg, lon_deg)` | Latitude geomagnética de dipolo centrado (IGRF-13 2020). |
| `oval_boundary` | função | `(kp)` | Latitude geomagnética do limite equatorial da aurora visível para um Kp (tabela NOAA SWPC). |
| `kp_needed` | função | `(maglat_deg, horizon_margin_deg=0.0)` | Kp necessário para ver aurora numa latitude geomagnética. |
| `build_xy` | função | `(kp, f107, horizon)` | Monta entradas e alvos de Kp para um horizonte de previsão. |
| `train_forecaster` | função | `(horizon=1, seed=0, epochs=300)` | Treina a rede de quantis de Kp, conformalizada e avaliada nos dois últimos meses. |
| `forecast` | função | `(model, kp_hist, f107_now)` | Previsão (baixo, mediana, alto) de Kp a partir do histórico recente. |
| `aurora_probability` | função | `(lo, med, hi, maglat_deg)` | Probabilidade de Kp atingir o necessário, a partir dos quantis da previsão. |
| `fetch_live_kp` | função | `(timeout=10.0)` | Baixa os últimos dias de Kp planetário da NOAA SWPC; levanta erro sem rede. |

#### `pinneapple_physics.astro.spk`

Leitor pequeno de kernels JPL SPK (arquivos DAF, segmentos tipo 2 e 3 de Chebyshev), mapeado em memória.

<small>`pinneapple_physics/astro/spk.py` · 124 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Segment` | classe | `(target, center, frame, kind, start_et, end_et, init, intlen, rsize, n, coeffs)` | Segmento de um kernel SPK (coeficientes de Chebyshev) que avalia estado num tempo. |
| `SPK` | classe | `(path)` | Kernel SPK mapeado em memória; encadeia segmentos por centros comuns para dar o estado alvo-centro. |

#### `pinneapple_physics.astro.stars`

Catálogo de estrelas brilhantes (Yale BSC5) e astrofísica estelar: distâncias, magnitude absoluta, temperatura pela cor, correção bolométrica, luminosidade, raio, espectros de Planck e cores de corpo negro.

<small>`pinneapple_physics/astro/stars.py` · 198 linhas · testes: 23</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `StarCatalog` | classe | `(hr, name, desig, con, ra, dec, pmra, pmdec, v, bv, sp, plx, plx_dyn, rv)` | Catálogo de estrelas em arrays: HR, nome, designação, posição, movimento próprio, magnitude, cor, tipo espectral, paralaxe. |
| `load_bright_stars` | função | `() -> StarCatalog` | Carrega o catálogo BSC5 empacotado como `StarCatalog`. |
| `load_constellations` | função | `() -> list` | Figuras das constelações da IAU com chave, nome latino, nomes por idioma e ponto de rótulo. |
| `constellation_names` | função | `(lang=None) -> dict` | Nomes das constelações por chave em latim (en) e português do Brasil (pt). |
| `planck_lambda` | função | `(wavelength_m, temperature_k)` | Radiância espectral de Planck B_lambda. |
| `wien_peak_m` | função | `(temperature_k)` | Comprimento de onda de pico de Wien (m) para uma temperatura. |
| `blackbody_rgb` | função | `(temperature_k)` | Cor sRGB de um corpo negro (reusa o ajuste do renderizador de buraco negro). |
| `blackbody_spectrum` | função | `(temperature_k, n=300, lam_min_nm=100.0, lam_max_nm=3000.0)` | Espectro de corpo negro com pico, fluxo total e cor RGB. |
| `teff_from_bv` | função | `(bv)` | Temperatura efetiva a partir de B-V (Ballesteros 2012). |
| `bolometric_correction` | função | `(teff)` | Correção bolométrica BC_V de corpo negro, com zero no Sol. |
| `star_properties` | função | `(cat, i, extinction_av=0.0) -> dict` | Distância, magnitude absoluta, temperatura, luminosidade e raio de uma estrela do catálogo. |
| `hr_diagram` | função | `(cat, max_rel_parallax_error=None, min_parallax_mas=5.0)` | Pontos (B-V, M_V) do diagrama HR para estrelas com paralaxe utilizável. |

#### `pinneapple_physics.astro.stars_ml`

Aprendizado de máquina sobre o catálogo de estrelas brilhantes (mistura gaussiana e k-NN no diagrama HR), conferido contra a classificação espectral do próprio catálogo.

<small>`pinneapple_physics/astro/stars_ml.py` · 154 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `luminosity_group` | função | `(sp)` | Extrai a classe de luminosidade de um tipo espectral (None se não houver). |
| `spectral_class` | função | `(sp)` | Índice da classe espectral OBAFGKM de um tipo espectral. |
| `dataset` | função | `(cat=None, min_parallax_mas=5.0)` | Dados do catálogo com paralaxe boa: B-V, M_V, grupo de luminosidade e classe espectral. |
| `gmm_fit` | função | `(X, k=3, iters=200, seed=0)` | Mistura gaussiana por EM com covariâncias completas. |
| `cross_validate` | função | `(X, y, k=15, folds=5, seed=0)` | Validação cruzada de um classificador k-NN sobre os dados padronizados. |
| `analyse` | função | `(seed=0) -> dict` | Roda a análise: agrupamentos GMM no diagrama HR nomeados pela classe de luminosidade majoritária e sua pureza. |

#### `pinneapple_physics.astro.surrogate`

Substituto neural informado por física (PINN) para propagação de órbitas, conferido contra o integrador clássico Yoshida-4, com limiares de confiança.

<small>`pinneapple_physics/astro/surrogate.py` · 268 linhas · testes: 32</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `reference_trajectory` | função | `(body, jd_tdb, days, n=400, eph=None)` | Posições e velocidades heliocêntricas de referência pela integração de N corpos Yoshida-4. |
| `train_orbit_pinn` | função | `(body='mars', jd_tdb=2461327.0, days=365.0, **kwargs) -> dict` | Treina o PINN substituto de órbita (com trava para uso em threads) e devolve o relatório. |

#### `pinneapple_physics.astro.timescales`

Escalas de tempo para astronomia posicional: UTC, TAI, TT, TDB, UT1 e tempo sideral; a classe `Time` guarda as datas julianas derivadas.

<small>`pinneapple_physics/astro/timescales.py` · 216 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Time` | classe | `(utc, dut1)` | Instante (ou vetor de instantes) em UTC com as escalas derivadas como datas julianas. |
| `tai_minus_utc` | função | `(jd_utc)` | TAI-UTC em segundos (NaN antes de 1972). |
| `delta_t_model` | função | `(year)` | TT - UT1 pelos polinômios de Espenak & Meeus (2006); usado antes de 1972 e como reserva. |
| `dut1` | função | `(jd_utc)` | UT1-UTC numa data, de uma tabela IERS finals2000A se houver (zero sem tabela). |
| `tdb_minus_tt` | função | `(jd_tt)` | TDB - TT pela aproximação periódica de 2 termos. |
| `jd_from_datetime` | função | `(dt) -> float` | Data juliana a partir de um `datetime` (UTC se sem fuso). |
| `datetime_from_jd` | função | `(jd) -> datetime` | `datetime` UTC a partir de uma data juliana. |
| `julian_centuries` | função | `(jd)` | Séculos julianos desde J2000. |
| `era` | função | `(jd_ut1)` | Ângulo de rotação da Terra (IAU 2000). |
| `gmst_rad` | função | `(t)` | Tempo sideral médio de Greenwich (IAU 2006). |
| `gast_rad` | função | `(t)` | Tempo sideral aparente de Greenwich (GMST mais equação dos equinócios). |
| `gast_hours` | função | `(t)` | Tempo sideral aparente de Greenwich em horas. |

#### `pinneapple_physics.astro.validation`

Verificações de precisão do observatório contra referências independentes (NOVAS, Horizons, almanaque USNO, eclipses NASA, passagens da ISS, skyfield), devolvidas como linhas de dados.

<small>`pinneapple_physics/astro/validation.py` · 198 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check_novas` | função | `()` | Compara posições aparentes com casos gravados do USNO NOVAS. |
| `check_horizons` | função | `()` | Compara posições com casos gravados do JPL Horizons. |
| `check_usno_almanac` | função | `()` | Compara nascer/pôr do Sol, fases e estações com tabelas do USNO. |
| `check_eclipses` | função | `()` | Compara horários de eclipses e trânsitos com o catálogo da NASA. |
| `check_iss_passes` | função | `()` | Compara passagens previstas da ISS com uma referência gravada. |
| `check_skyfield_live` | função | `(n=24, seed=1)` | Compara alt/az de todos os corpos com o skyfield em épocas e observadores aleatórios (opcional). |
| `check_integrators` | função | `()` | Confere conservação de energia e momento angular do integrador de N corpos e a diferença para o DE421. |
| `run_all` | função | `(include_skyfield=True) -> list[dict]` | Roda todas as verificações e devolve as linhas de resultado. |

#### `pinneapple_physics.blackhole`

Acreção em buraco negro: simulações hidrodinâmicas e previsão do 'tempo do buraco negro' com horizonte de confiança, reproduzindo e estendendo Duarte, Nemmen & Navarro (2022); reexporta `AccretionFlow`, `RIAFConfig`, `bondi_pw`, `torus_state`.

<small>`pinneapple_physics/blackhole/__init__.py` · 20 linhas · testes: 19</small>

#### `pinneapple_physics.blackhole.capture`

Grava uma cena Twin3D em quadros PNG/GIF com Chromium headless (Playwright, dependência opcional), para docs e posts. [sem teste direto]

<small>`pinneapple_physics/blackhole/capture.py` · 79 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `capture_twin` | função | `(folder, steps, out_dir=None, *, field='', cmap='turbo', view='', zoom=1.0, size=(1280, 7…)` | Tira screenshots da cena exportada numa pasta em cada passo de tempo e devolve os caminhos PNG. |

#### `pinneapple_physics.blackhole.convergence`

Estudo de resolução das simulações de acreção: remapeamento conservativo entre grades, estatísticas por janela de tempo e ordem observada, para saber se os dados de treino do previsor estão convergidos.

<small>`pinneapple_physics/blackhole/convergence.py` · 262 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `window_mask` | função | `(grid) -> np.ndarray` | Células de uma grade cujo centro cai na janela física de previsão. |
| `cell_volumes` | função | `(grid) -> np.ndarray` | Volume 2 pi r^2 dr d(cos theta) de cada célula. |
| `remap` | função | `(a, faces_f, faces_c) -> np.ndarray` | Remapeamento conservativo (ponderado por volume) de médias de célula entre duas grades esféricas do mesmo domínio. |
| `remap_r` | função | `(a, rf_f, rf_c) -> np.ndarray` | Remapeamento de um perfil radial por sobreposição em log r. |
| `psd_slope` | função | `(x, dt, f_lo, f_hi) -> float` | Inclinação log-log do espectro de Welch entre duas frequências. |
| `run_statistics` | função | `(folder, t0, t1, log_rho_range=LOG_RHO_RANGE) -> dict` | Estatísticas de uma rodada numa janela de tempo: séries de escalares e médias temporais de perfis e campos. |
| `resolution_study` | função | `(runs, *, t0=2000.0, t1=10000.0, log=print) -> dict` | Estudo de resolução sobre rodadas em várias grades e sementes, com razão de refinamento constante. |

#### `pinneapple_physics.blackhole.dataset`

Dataset do 'tempo do buraco negro': quadros de densidade de trajetórias simuladas inteiras, quantizados em uint16 e empacotados em npz determinístico para o model zoo.

<small>`pinneapple_physics/blackhole/dataset.py` · 170 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `encode` | função | `(log10_rho) -> np.ndarray` | Log10 da densidade para uint16; levanta erro fora da faixa de quantização. |
| `decode` | função | `(q) -> np.ndarray` | Coluna uint16 de volta para log10 da densidade (float32). |
| `deterministic_npz` | função | `(arrays) -> bytes` | npz com datas e ordem fixas no zip, para o SHA-256 não mudar entre builds iguais. |
| `build_arrays` | função | `(runs, conv=None, *, splits=SPLITS, t0=T0, t1=T1) -> dict[str, np.ndarray]` | Monta os arrays do dataset do zoo a partir das pastas de simulação e do relatório de convergência. |
| `write_npz` | função | `(arrays, path) -> Path` | Grava todos os splits num único arquivo npz. |
| `write_split_files` | função | `(arrays, folder) -> dict[str, Path]` | Grava um npz por split, para baixar só o split usado e manter arquivos pequenos. |
| `load_trajectories` | função | `(split, path=None) -> dict[int, dict]` | Carrega as trajetórias inteiras de um split, por semente, com quadros em ordem de tempo. |
| `window_volumes` | função | `(path=None) -> np.ndarray` | Volumes das células da janela de previsão. |

#### `pinneapple_physics.blackhole.figures`

Figuras do estudo: habilidade contra tempo de antecedência e estudo de resolução, com matplotlib. [sem teste direto]

<small>`pinneapple_physics/blackhole/figures.py` · 121 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_skill` | função | `(report, path, model=None, title='') -> str` | Figura de erro (MAE) contra antecedência para modelo e baselines, e da habilidade contra persistência com intervalo e horizonte. |
| `plot_convergence` | função | `(report, path) -> str` | Figura de perfis médios de densidade e momento angular por grade, e da taxa média de acreção com intervalo. |

#### `pinneapple_physics.blackhole.forecast`

Previsão do 'tempo do buraco negro': a U-Net de Duarte et al. (2022), treino com perda de rollout, ruído e penalidade de massa, rollouts, horizonte de confiança e envelope de massa.

<small>`pinneapple_physics/blackhole/forecast.py` · 473 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DensityCodec` | classe | `(log_min, log_max, r_cells, theta_trim)` | Normalização logarítmica e recorte compartilhados entre treino e inferência (salvos com o checkpoint). |
| `DuarteUNet` | classe | `(filters=32, frames=5, residual=False, padding='zeros')` | A U-Net de Duarte et al. (2022) que prevê o próximo bloco de quadros de densidade. |
| `TrainConfig` | classe | `(filters, frames, residual, padding, epochs, batch_size, lr, alpha, clip, loss, rollout_b…)` | Configuração de treino: arquitetura, épocas, lote, taxa, perda, blocos de rollout, ruído, pesos de massa, bf16, semente e tempo máximo. |
| `MassEnvelope` | classe | `(rate_max, margin)` | Limite aprendido de quão rápido a massa na janela de previsão pode mudar; aponta violações. |
| `MassProjection` | classe | `(codec, cell_volume, envelope, frame_dt)` | Projeta quadros previstos no orçamento de massa visto no treino, reescalando quadros que ganham ou perdem massa demais. |
| `make_blocks` | função | `(x, k=5, stride=1, out_blocks=1) -> tuple[np.ndarray, np.ndarray]` | Transforma quadros (T, H, W) em pares entrada/alvo de blocos de k quadros, com alvos para rollout de vários passos. |
| `duarte_loss` | função | `(pred, target, alpha=8.0)` | MAE na janela mais alfa vezes MAE nos pixels de alta densidade. |
| `duarte_regional_loss` | função | `(pred, target, regions=None)` | MSE na janela mais MAE ponderado nas quatro regiões do artigo. |
| `train_forecaster` | função | `(X, Y, Xv, Yv, cfg, codec, out_dir, *, log=print, cell_volume=None, init=None) -> dict[st…)` | Treina com Adam e a perda de Duarte, guardando os pesos com melhor perda de validação. |
| `train_on_trajectories` | função | `(train, val, cfg, out_dir, *, cell_volume=None, val_blocks=1, val_stride=1, init=None, lo…)` | Treina a partir de trajetórias inteiras, sem blocos cruzando trajetórias e com validação em trajetórias separadas. |
| `evaluate_loss` | função | `(model, X, Y, cfg, batch=32) -> float` | Perda da previsão desenrolada sobre todos os blocos alvo. |
| `save_forecaster` | função | `(path, model, cfg, codec, meta) -> None` | Salva pesos, configuração, codec e metadados num checkpoint torch. |
| `load_forecaster` | função | `(path) -> tuple[DuarteUNet, DensityCodec, dict]` | Carrega um checkpoint e devolve modelo em modo eval, codec e conteúdo do checkpoint. |
| `rollout` | função | `(model, x0, n_blocks, *, clip=False, mass_projection=None) -> np.ndarray` | Previsão iterativa: cada bloco previsto volta como entrada, opcionalmente com projeção de massa. |
| `lead_time_scores` | função | `(pred, truth, last_input, mean_state, weights=None) -> dict[str, np.ndarray]` | Métricas por antecedência: MAE do modelo, da persistência e da média temporal, correlação de anomalia e de tendência. |
| `trust_horizon` | função | `(scores, frame_dt, acc_min=0.6, block=1) -> dict[str, float]` | Antecedência até a qual a previsão é útil (correlação de anomalia acima do limiar e erro abaixo das referências). |
| `window_mass` | função | `(rho, cell_volume) -> np.ndarray` | Massa na janela para quadros de densidade e volumes de célula. |

#### `pinneapple_physics.blackhole.hydro`

Hidrodinâmica viscosa de acreção em torno de um buraco negro de Schwarzschild (potencial de Paczynski-Wiita), axissimétrica em (r, theta) com v_phi, em PyTorch; inclui toro de equilíbrio e solução de Bondi. [sem teste direto]

<small>`pinneapple_physics/blackhole/hydro.py` · 506 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RIAFConfig` | classe | `(nr, ntheta, r_in, r_out, gamma, alpha, viscosity, cfl, rho_floor, p_floor, v_max, t_cap,…)` | Configuração da simulação: grade, raios, gama, viscosidade alfa, CFL, pisos, contorno externo, backend, parâmetros do toro e perturbação. |
| `AccretionFlow` | classe | `(cfg)` | Fluxo de acreção viscoso 2.5D (axissimétrico com v_phi); conversões primitivas/conservadas, passo de tempo, balanços e `run` que salva snapshots. |
| `torus_state` | função | `(flow) -> np.ndarray` | Estado inicial de toro em equilíbrio com momento angular em lei de potência, imerso numa atmosfera. |
| `bondi_pw` | função | `(r, *, cs_inf, gamma=1.4, rho_inf=1.0)` | Solução exata de acreção transônica de Bondi no potencial de Paczynski-Wiita, por busca de raiz. |

#### `pinneapple_physics.blackhole.hydro_numba`

Kernels Numba para `AccretionFlow` quando `RIAFConfig(backend="numba")`; requer o pacote `numba`. [sem teste direto]

<small>`pinneapple_physics/blackhole/hydro_numba.py` · 321 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NumbaStepper` | classe | `(flow)` | Guarda a geometria de um `AccretionFlow` e avança o estado com os kernels Numba. |
| `rhs` | função | `(W, r, th, rf, thf, area_r, area_t, vol, inv_r, geo_t, Rc, gamma, alpha, visc, dU, fluxes)` | Kernel Numba: variação dU nas células internas e fluxos de massa e momento angular nos contornos. |
| `prim_to_cons` | função | `(W, Rc, gamma, U)` | Kernel Numba: primitivas internas para variáveis conservadas. |
| `cons_to_prim` | função | `(U, W, Rc, r, gamma, rho_floor, p_floor, v_max, t_cap)` | Kernel Numba: conservadas para primitivas, com pisos e limites; devolve quantas células foram limitadas. |
| `boundaries` | função | `(W, outer, has_outer)` | Kernel Numba que preenche as células fantasmas: cópia nos contornos radiais (sem entrada de massa onde não há contorno externo dado) e reflexão nos polos com troca de sinal de v_theta. |
| `time_step` | função | `(W, r, dr, dth, gamma, alpha, visc, cfl)` | Kernel Numba que calcula o passo de tempo pelos limites hidrodinâmico e viscoso (CFL). |

#### `pinneapple_physics.blackhole.raytrace`

Traçado de raios relativístico de um fluxo de acreção em torno de buraco negro de Schwarzschild (vista estilo 'Interstellar'), com campo de estrelas, bloom e renderização com Doppler.

<small>`pinneapple_physics/blackhole/raytrace.py` · 316 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RayPaths` | classe | `(ray, ir, ith, ds, kdotv, lapse, captured, escape_dir, shape)` | Onde cada raio cruza a região emissora, em ordem câmera-longe, com dados para emissão e captura. |
| `Camera` | classe | `(width, height, r_cam, inclination_deg, fov_deg, roll_deg, dphi, ds_in, ds_out, max_phi)` | Câmera em torno do buraco negro: resolução, distância, inclinação, campo de visão; traça os raios. |
| `blackbody_rgb` | função | `(T) -> np.ndarray` | Cor sRGB aproximada de um corpo negro (ajuste de Tanner Helland). |
| `star_field` | função | `(width=4096, height=2048, n=9000, seed=3) -> np.ndarray` | Mapa de estrelas equiretangular determinístico com faixa tênue tipo Via Láctea. |
| `bloom` | função | `(img, strength=0.35, sigma=6.0) -> np.ndarray` | Brilho cinematográfico: soma uma cópia borrada das partes brilhantes. |
| `render` | função | `(paths, rho, temperature, velocity=None, *, doppler=True, kappa=0.0, exposure=1.0, scale=…)` | Renderiza um quadro a partir dos caminhos dos raios, densidade, temperatura e velocidade (Doppler opcional). |

#### `pinneapple_physics.blackhole.runs`

Gera, retoma e lê as simulações de acreção usadas pelo previsor (issue #399).

<small>`pinneapple_physics/blackhole/runs.py` · 89 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `load_run` | função | `(folder) -> dict` | Lê uma rodada: tempos, quadros, taxa de acreção, grade e configuração. |
| `simulate_run` | função | `(cfg, folder, *, t_end=10000.0, every=10.0, chunk=50, name=None, log=print) -> str` | Roda (ou retoma) uma simulação numa pasta, salvando em blocos com checkpoint. |

#### `pinneapple_physics.blackhole.skill`

Habilidade do previsor contra persistência, climatologia e DMD, com intervalos bootstrap e verificações físicas de massa e fluxo.

<small>`pinneapple_physics/blackhole/skill.py` · 178 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DMDBaseline` | classe | `(trajectories, rank)` | Baseline DMD de posto r (via `pinneapple_neural`), iterada a partir de um quadro, mantendo congelada a parte fora do subespaço. |
| `window_mass` | função | `(x, codec, vol) -> np.ndarray` | Massa na janela para quadros normalizados. |
| `implied_inflow` | função | `(x, codec, vol, dt, shells) -> np.ndarray` | Fluxo líquido de massa para dentro pelas cascas, inferido da variação da massa contida entre quadros. |
| `evaluate_skill` | função | `(models, train, val, test, *, frame_dt, cell_volume, blocks=20, start_every=5, block_M=10…)` | Avalia modelos contra persistência, climatologia e DMD em vários postos, por antecedência, com verificações físicas. |

#### `pinneapple_physics.blackhole.twin`

Vista 3D de gêmeo digital de um fluxo de acreção axissimétrico com o PINNeAPPle-Twin3D: painéis em corte e esferas.

<small>`pinneapple_physics/blackhole/twin.py` · 120 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CutawayMesh` | classe | `(r_faces, theta_faces, *, r_view, phi_cut_deg=270.0, n_phi=72, offset=(0.0, 0.0, 0.0))` | Geometria de um painel em corte e o mapa dos campos de célula (nr, ntheta) para seus vértices. |
| `sphere` | função | `(radius, center=(0, 0, 0), n=24) -> tuple[np.ndarray, np.ndarray]` | Malha de esfera (vértices e faces) de raio e centro dados. |
| `accretion_scene` | função | `(grid, panels, times, *, field='log10_density', unit='', r_view=60.0, gap=0.35, title='Bl…)` | Cena Twin3D com um painel em corte por campo, dispostos lado a lado, ao longo do tempo. |

#### `pinneapple_physics.closed_form`

Subpacote de modelos físicos em forma fechada (sem EDP), portados de ferramentas de produtos para serem compartilhados e testados; o `__init__` importa só parte dos módulos (não importa `eddy_coil`, `electromagnetics`, `heat_exchanger`, `pipe_insulation`, `pump_npsh`, `thin_airfoil`).

<small>`pinneapple_physics/closed_form/__init__.py` · 73 linhas · testes: 103</small>

#### `pinneapple_physics.closed_form.burgers`

Solução exata da equação de Burgers viscosa (Cole-Hopf) e invíscida (características) com condição inicial `-sin(pi x)` em [-1, 1].

<small>`pinneapple_physics/closed_form/burgers.py` · 73 linhas · testes: 80</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `burgers_sine_exact` | função | `(x, t, nu, n_nodes=120) -> np.ndarray` | Calcula `u(x, t)` da Burgers viscosa com condição inicial senoidal por quadratura (Cole-Hopf). |
| `burgers_sine_inviscid_exact` | função | `(x, t) -> np.ndarray` | Calcula a solução de entropia da Burgers invíscida com condição inicial senoidal. |

#### `pinneapple_physics.closed_form.cantilever_fatigue`

Tensão de flexão de viga em balanço e vida em fadiga pela relação de Basquin (equações de livro-texto, Shigley e Dowling).

<small>`pinneapple_physics/closed_form/cantilever_fatigue.py` · 73 linhas · testes: 32</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `stress_from_moment_pa` | função | `(width_m, thickness_m, moment_n_m) -> float` | Tensão `sigma = M c / I` numa seção retangular sólida; base comum das outras funções de tensão. |
| `cantilever_root_bending_stress_pa` | função | `(width_m, thickness_m, load_n, length_m) -> float` | Tensão de flexão na raiz engastada da viga em balanço, onde o momento é máximo (`M = F L`). |
| `cantilever_bending_stress_at_position_pa` | função | `(width_m, thickness_m, load_n, length_m, x_from_root_m) -> float` | Perfil de tensão de flexão ao longo da mesma viga, numa posição medida a partir da raiz. |
| `basquin_fatigue_life_cycles` | função | `(stress_amplitude_pa, fatigue_strength_coeff_pa, fatigue_strength_exponent) -> float` | Ciclos até a falha pela equação de Basquin, dada a amplitude de tensão e as constantes de fadiga do material. |
| `basquin_allowable_stress_pa` | função | `(target_life_cycles, fatigue_strength_coeff_pa, fatigue_strength_exponent) -> float` | Basquin invertida: maior amplitude de tensão que ainda atinge a vida alvo em ciclos. |

#### `pinneapple_physics.closed_form.compressible`

Escoamento compressível 1-D de gás caloricamente perfeito: razões isentrópicas, choque normal, pressão de Pitot e `Cp_max` newtoniano modificado.

<small>`pinneapple_physics/closed_form/compressible.py` · 78 linhas · testes: 15</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `isentropic_ratios` | função | `(M, gamma=1.4) -> dict[str, np.ndarray]` | Razões de estagnação/estática (T, p, rho) e razão de áreas `A/A*` num número de Mach (NACA 1135). |
| `normal_shock` | função | `(M, gamma=1.4) -> dict[str, np.ndarray]` | Saltos através de um choque normal para Mach a montante `M >= 1` (NACA 1135). |
| `pitot_pressure_ratio` | função | `(M, gamma=1.4)` | Razão entre a pressão de Pitot e a pressão estática do escoamento livre. |
| `cp_max_newtonian` | função | `(M, gamma=1.4)` | Coeficiente de pressão no ponto de estagnação da teoria newtoniana modificada (Lees 1955). |

#### `pinneapple_physics.closed_form.eddy_coil`

Impedância de uma bobina de correntes parasitas com núcleo de ar sobre uma placa condutora (Dodd & Deeds, 1968).

<small>`pinneapple_physics/closed_form/eddy_coil.py` · 69 linhas · testes: 27</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `skin_depth` | função | `(f, sigma, mu_r=1.0) -> float` | Profundidade de penetração `1 / sqrt(pi f mu sigma)`. |
| `coil_impedance` | função | `(f, *, r1, r2, l1, l2, turns, sigma, thickness=math.inf) -> dict` | Impedância da bobina no ar, sua variação sobre a placa, impedância normalizada e profundidade de penetração. |

#### `pinneapple_physics.closed_form.electromagnetics`

Soluções fechadas dos presets eletromagnéticos, registradas como solução `"analytic"` de `pp.solve`.

<small>`pinneapple_physics/closed_form/electromagnetics.py` · 125 linhas · testes: 41</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `two_dielectric_capacitor` | função | `(y, V, d, d1, eps1, eps2)` | Potencial num capacitor de placas paralelas com dois dielétricos (Griffiths, ex. 4.7). |
| `wire_az` | função | `(x, y, a, J, nu=1.0)` | Potencial vetor `A_z` de um condutor redondo com densidade de corrente uniforme (lei de Ampère). |
| `tm_cavity_mode` | função | `(x, y, t, m, n, Lx, Ly, eps=1.0, mu=1.0)` | Onda estacionária TM_mn numa cavidade retangular PEC (campos Ez, Hx, Hy). |
| `skin_effect_slab` | função | `(x, delta, A0=1.0)` | Fasor do campo que se difunde num semiespaço condutor (efeito pelicular). |
| `cylinder_tm_total` | função | `(x, y, k0, radius, eps_r)` | Campo `Ez` total de uma onda plana espalhada por um cilindro dielétrico (série de funções cilíndricas, via `pinneapple_simulation`). |

#### `pinneapple_physics.closed_form.fin_array_conduction`

Modelo fechado de condução e convecção em conjunto de aletas (Incropera), generalizado para qualquer arranjo de superfícies estendidas.

<small>`pinneapple_physics/closed_form/fin_array_conduction.py` · 125 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `fin_array_profile_params` | função | `(k_material_w_mk, h_conv_w_m2k, n_fins, fin_thickness_m, fin_length_m, transverse_extent_…)` | Calcula uma vez as grandezas intermediárias do modelo de aletas, compartilhadas pelo valor escalar e pelo perfil espacial. |
| `fin_array_base_temperature_c` | função | `(params, t_ambient_c, power_w) -> float` | Temperatura na base do conjunto de aletas (verdade de referência escalar). |
| `fin_array_temperature_at_position_c` | função | `(params, t_ambient_c, x_from_root_m, corrected_length_m=None) -> float` | Perfil de temperatura ao longo da aleta pela mesma equação 1-D, com correção de ponta adiabática. |

#### `pinneapple_physics.closed_form.heat`

Condução transiente 1-D numa placa (série de Fourier) e aleta de pino com ponta convectiva (Incropera).

<small>`pinneapple_physics/closed_form/heat.py` · 79 linhas · testes: 17</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `slab_transient` | função | `(x, t, L, alpha, T_left, T_right, T0, n_terms=None) -> np.ndarray` | Temperatura `T(x, t)` numa placa com temperaturas impostas nas faces e temperatura inicial uniforme. |
| `pin_fin_theta` | função | `(xi, h, k, D, L)` | Excesso de temperatura adimensional ao longo de uma aleta de pino com ponta convectiva. |
| `pin_fin_heat` | função | `(h, k, D, L, T_base, T_inf) -> float` | Calor dissipado [W] por uma aleta de pino com ponta convectiva. |

#### `pinneapple_physics.closed_form.heat_exchanger`

Dimensionamento de trocadores de calor: efetividade-NTU, fator de correção da LMTD e coeficientes do lado dos tubos e do casco.

<small>`pinneapple_physics/closed_form/heat_exchanger.py` · 155 linhas · testes: 41</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `effectiveness` | função | `(ntu, cr, arrangement='counterflow', n_shell=1) -> float` | Efetividade `epsilon(NTU, C_r)` para contracorrente, paralelo, casco-tubo ou fluxo cruzado. |
| `ntu_from_effectiveness` | função | `(eps, cr, arrangement='counterflow') -> float` | Inverte `effectiveness` por bisseção para obter o NTU. |
| `lmtd_correction_1_2` | função | `(t_hot_in, t_hot_out, t_cold_in, t_cold_out) -> float` | Fator F de correção da LMTD de um trocador 1-2 (Bowman et al., 1940). |
| `gnielinski` | função | `(re, pr) -> dict[str, float]` | Número de Nusselt do lado dos tubos (Gnielinski; laminar desenvolvido abaixo de Re = 2300). |
| `hausen_mean_nusselt` | função | `(re, pr, d_over_l) -> float` | Nusselt médio laminar com entrada térmica num tubo de parede isotérmica (Hausen, 1943). |
| `mean_nusselt_from_temperatures` | função | `(re, pr, d_over_l, t_wall, t_in, t_out) -> float` | Nusselt médio de um tubo isotérmico a partir das temperaturas de entrada e saída (balanço de energia). |
| `kern_shell` | função | `(m_dot, mu, k, cp, *, shell_id, baffle_spacing, tube_od, pitch, layout='triangular') -> d…)` | Coeficiente do lado do casco pelo método de Kern (área de fluxo cruzado, diâmetro equivalente, Re, Nu, h). |
| `one_two_exchanger_ode` | função | `(ntu, cr, n=4001, profiles=False)` | Efetividade de um trocador 1-2 integrando as três equações de energia por diferenças finitas (modelo de Nagle), sem a fórmula fechada. |

#### `pinneapple_physics.closed_form.helmholtz_resonator`

Equação de sintonia de ressonador de Helmholtz (porta de caixa acústica bass-reflex).

<small>`pinneapple_physics/closed_form/helmholtz_resonator.py` · 50 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `helmholtz_resonant_frequency_hz` | função | `(port_radius_m, port_length_m, cabinet_volume_l, end_correction_factor=1.7) -> float` | Frequência de ressonância para uma porta de raio e comprimento dados num volume de caixa, com fator de correção de extremidade. |
| `port_volume_m3` | função | `(port_radius_m, port_length_m) -> float` | Volume de ar da porta, usado como restrição ou objetivo de projeto. |

#### `pinneapple_physics.closed_form.membrane_diffusion`

Transporte difusivo controlado por membrana (primeira lei de Fick), como em adesivos transdérmicos de reservatório.

<small>`pinneapple_physics/closed_form/membrane_diffusion.py` · 148 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `steady_state_release_rate_mg_day` | função | `(diffusion_coeff_m2_s, partition_coeff, reservoir_conc_kg_m3, area_m2, membrane_thickness…)` | Taxa de liberação em regime permanente (mg/dia) através da membrana limitante. |
| `depletion_time_days` | função | `(diffusion_coeff_m2_s, partition_coeff, membrane_thickness_m, reservoir_thickness_m) -> f…)` | Tempo de esgotamento do reservatório; depende só das espessuras e da permeabilidade, não da área nem da concentração. |
| `concentration_at_membrane_depth` | função | `(reservoir_conc_kg_m3, partition_coeff, membrane_thickness_m, z_from_reservoir_m) -> float` | Perfil linear de concentração dentro da membrana em regime permanente. |
| `post_depletion_decay_rate_constant_per_day` | função | `(diffusion_coeff_m2_s, partition_coeff, membrane_thickness_m, reservoir_thickness_m) -> f…)` | Constante de decaimento de primeira ordem depois que o excesso de fármaco do reservatório acaba. |
| `simulate_release_timeseries` | função | `(diffusion_coeff_m2_s, partition_coeff, reservoir_conc_kg_m3, area_m2, membrane_thickness…)` | Série temporal de liberação e esgotamento combinando as fórmulas do módulo. |

#### `pinneapple_physics.closed_form.particle_drag`

Coeficiente de arrasto de uma esfera rígida em escoamento uniforme por correlações empíricas.

<small>`pinneapple_physics/closed_form/particle_drag.py` · 32 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `clift_sphere_cd` | função | `(re) -> float` | Arrasto da esfera por Clift, Grace & Weber (1978), faixa 20 < Re <= 260 (avisa fora dela). |
| `schiller_naumann_cd` | função | `(re) -> float` | Arrasto da esfera por Schiller & Naumann (1933), Re < 800 (avisa acima). |

#### `pinneapple_physics.closed_form.pcb_thermal`

Blocos térmicos de placas de circuito: condutividade efetiva do empilhamento, vias térmicas, convecção e radiação nas faces, e dados de encapsulamentos para o modelo de dois resistores JEDEC.

<small>`pinneapple_physics/closed_form/pcb_thermal.py` · 179 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Layer` | classe | `(name, kind, thickness_m, coverage)` | Camada do empilhamento (cobre ou dielétrico) com espessura e cobertura de cobre; dá `k_xy` e `k_z` pela regra da fração de área. |
| `Environment` | classe | `(t_ambient_c, air_velocity_m_s, orientation, emissivity, extra)` | Condições do ambiente da placa: temperatura, velocidade do ar (0 = convecção natural), orientação e emissividade. |
| `stackup` | função | `(n_copper=4, thickness_mm=1.6, outer_oz=1.0, inner_oz=1.0, outer_coverage=0.3, inner_cove…)` | Monta um empilhamento simétrico de camadas de sinal, planos e dielétrico. |
| `effective_conductivity` | função | `(layers) -> Dict[str, float]` | Condutividade efetiva da placa: regra paralela no plano e regra em série na espessura. |
| `via_copper_fraction` | função | `(n_vias, drill_mm, plating_um, area_m2) -> float` | Fração de área de cobre dos barris de vias metalizadas (não preenchidas) sob uma área. |
| `convection_coefficients` | função | `(width_m, depth_m, t_surface_c, env) -> Dict[str, float]` | Coeficientes h das faces superior e inferior (convecção natural ou forçada mais radiação linearizada). |

#### `pinneapple_physics.closed_form.pipe_flow`

Escoamento desenvolvido em tubos e canais: fatores de atrito de Darcy, perda de carga e altura da primeira célula para um y+ alvo.

<small>`pinneapple_physics/closed_form/pipe_flow.py` · 169 linhas · testes: 29</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `laminar` | função | `(Re) -> float` | Fator de atrito laminar `64 / Re` (avisa para Re >= 2300). |
| `blasius` | função | `(Re) -> float` | Correlação de Blasius para tubo liso (avisa fora de 4e3 a 1e5). |
| `colebrook` | função | `(Re, rel_rough=0.0, tol=1e-14, max_iter=100) -> float` | Fator de atrito de Darcy pela equação implícita de Colebrook, resolvida por iteração. |
| `haaland` | função | `(Re, rel_rough=0.0) -> float` | Aproximação explícita de Haaland para a equação de Colebrook. |
| `dean_number` | função | `(Re, D, Rc) -> float` | Número de Dean de um tubo curvo. |
| `curved_pipe_ratio` | função | `(De) -> float` | Razão entre o atrito laminar de tubo curvo e de tubo reto (White, 1929). |
| `friction_factor` | função | `(Re, rel_rough=0.0) -> float` | Escolhe `laminar` abaixo de Re = 2300 e `colebrook` acima (sem aviso na faixa de transição). |
| `darcy_pressure_drop` | função | `(f, L, D, rho, U) -> float` | Perda de pressão de Darcy-Weisbach num duto. |
| `parallel_plates_pressure_drop` | função | `(mu, L, h, U) -> float` | Perda de pressão laminar entre placas paralelas infinitas (Poiseuille plano). |
| `first_cell_height` | função | `(y_plus, Re, D, U, nu, f=None, rel_rough=0.0) -> float` | Altura da primeira célula junto à parede para atingir um y+ desejado. |

#### `pinneapple_physics.closed_form.pipe_insulation`

Perda de calor em tubo isolado e espessura econômica de isolamento (método ISO 12241 / VDI 2055).

<small>`pinneapple_physics/closed_form/pipe_insulation.py` · 151 linhas · testes: 27</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `air` | função | `(t_c) -> dict[str, float]` | Propriedades do ar (nu, k, Pr, beta) interpoladas de tabela numa temperatura. |
| `surface_coefficient` | função | `(ts, ta, d, eps, wind=0.0) -> dict[str, float]` | Coeficiente externo de um cilindro horizontal no ar: radiação mais convecção mista. |
| `heat_loss` | função | `(d_pipe, thickness, t_in, t_amb, *, k0, k1=0.0, eps=0.9, wind=0.0) -> dict[str, float]` | Perda de calor por metro, temperatura de superfície e coeficientes de um tubo isolado ou nu. |
| `radial_fd` | função | `(d_pipe, thickness, t_in, t_surface, *, k0, k1=0.0, n=400) -> dict[str, np.ndarray]` | Perfil de temperatura no isolamento por volumes finitos radiais com k(T) e iterações de Picard. |
| `crf` | função | `(rate, years) -> float` | Fator de recuperação de capital. |
| `economic_thickness` | função | `(d_pipe, t_in, t_amb, thicknesses, *, k0, k1, eps, wind, hours, energy_price, efficiency,…)` | Custo anual (energia + isolamento anualizado) por espessura e escolha da espessura econômica. |

#### `pinneapple_physics.closed_form.plate_fin_heatsink`

Modelo térmico e hidráulico de dissipador de aletas planas com correlações publicadas (Teertstra, Bar-Cohen, Lee, Muzychka); os resultados trazem avisos quando uma correlação sai da faixa.

<small>`pinneapple_physics/closed_form/plate_fin_heatsink.py` · 309 linhas · testes: 18</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HeatSinkGeometry` | classe | `(base_width, base_depth, base_thickness, n_fins, fin_thickness, fin_height, material)` | Geometria do dissipador em metros (base, número, espessura e altura das aletas, material), com `fin_gap` e `mass_kg`. |
| `OperatingPoint` | classe | `(power_w, t_ambient_c, air_velocity_m_s, source_width, source_depth, tim_resistance_k_m2_…)` | Ponto de operação: potência, temperatura ambiente, velocidade do ar, tamanho da fonte e resistência da interface térmica. |
| `air_properties` | função | `(t_c, p_pa=101325.0) -> Dict[str, float]` | Propriedades do ar a 1 atm interpoladas da tabela de Incropera, com densidade de gás ideal. |
| `geometry_errors` | função | `(g, op) -> List[str]` | Lista os erros de geometria ou de ponto de operação (material desconhecido, dimensões inválidas). |
| `forced_convection` | função | `(g, v_approach, air) -> Dict[str, float]` | Convecção forçada em canais de placas paralelas (Teertstra) e perda de pressão. |
| `natural_convection` | função | `(g, dt_k, air) -> Dict[str, float]` | Convecção natural entre placas verticais isotérmicas (Bar-Cohen & Rohsenow). |
| `spreading_resistance` | função | `(k, tb, source_area, plate_area, r_conv) -> Dict[str, float]` | Resistência de espalhamento da base (Lee et al., 1995), incluindo condução 1-D pela base. |
| `evaluate` | função | `(g, op, max_iter=60, h_scale=1.0, t_root_guess_c=None) -> Dict[str, object]` | Resolve a rede de resistências térmicas e devolve temperatura da fonte, resistências e avisos. |

#### `pinneapple_physics.closed_form.pump_npsh`

Bomba centrífuga num sistema de sucção e recalque: ponto de operação, NPSH disponível contra requerido e margem de cavitação.

<small>`pinneapple_physics/closed_form/pump_npsh.py` · 94 linhas · testes: 27</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `iapws97_psat` | função | `(t_c) -> float` | Pressão de saturação da água pela IAPWS-IF97. |
| `water_density` | função | `(t_c) -> dict[str, float]` | Densidade e viscosidade da água interpoladas de tabela numa temperatura. |
| `atmosphere` | função | `(z_m) -> float` | Pressão atmosférica ISO 2533 numa altitude (troposfera). |
| `pipe_loss` | função | `(q, d, length, k_minor, rough, rho, mu) -> dict[str, float]` | Perda de carga de um tubo com perdas localizadas numa vazão. |
| `fit_quadratic` | função | `(q, y) -> np.ndarray` | Ajuste por mínimos quadrados de uma curva quadrática na vazão a pontos de catálogo. |
| `duty_point` | função | `(head_coef, h_static, system_loss, q_max) -> float` | Vazão onde a altura da bomba iguala a do sistema (bisseção). |
| `npsh_available` | função | `(p_surface, t_c, z_suction, h_loss) -> dict[str, float]` | NPSH disponível no flange de sucção. |

#### `pinneapple_physics.closed_form.rocket_nozzle`

Bocal de foguete ideal: escoamento isentrópico 1-D de gás caloricamente perfeito num bocal convergente-divergente (Sutton & Biblarz).

<small>`pinneapple_physics/closed_form/rocket_nozzle.py` · 114 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `area_ratio` | função | `(M, gamma) -> float` | Razão de áreas `A/A*` num número de Mach. |
| `mach_from_area_ratio` | função | `(ratio, gamma, *, supersonic=True, tol=1e-12, max_iter=300) -> float` | Inverte `A/A*` por bisseção, no ramo supersônico (padrão) ou subsônico. |
| `pressure_ratio` | função | `(M, gamma) -> float` | Razão entre pressão estática e de estagnação num Mach. |
| `temperature_ratio` | função | `(M, gamma) -> float` | Razão entre temperatura estática e de estagnação num Mach. |
| `choked_mass_flow` | função | `(p0, throat_area, gamma, R_gas, T0) -> float` | Vazão mássica numa garganta bloqueada. |
| `characteristic_velocity` | função | `(gamma, R_gas, T0) -> float` | Velocidade característica c*, que só depende do gás e da temperatura da câmara. |
| `thrust_coefficient` | função | `(p_exit, p0, p_atm, ratio, gamma) -> float` | Coeficiente de empuxo (termo de quantidade de movimento mais termo de pressão). |
| `specific_impulse` | função | `(cf, c_star, g0=G0) -> float` | Impulso específico `Isp = C_F c* / g0`. |
| `nozzle_performance` | função | `(*, gamma, R_gas, T_chamber, p_chamber, area_ratio_exit, p_atm=101325.0) -> dict` | Estado de saída e desempenho de um bocal ideal expandido até uma razão de áreas. |

#### `pinneapple_physics.closed_form.thin_airfoil`

Aerodinâmica 2-D fechada: sustentação de linha média NACA de 4 dígitos pela teoria de aerofólio fino e arrasto de perfil por correlação.

<small>`pinneapple_physics/closed_form/thin_airfoil.py` · 61 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `naca4_camber_slope` | função | `(x, m, p) -> np.ndarray` | Inclinação `dz/dx` da linha média NACA de 4 dígitos. |
| `thin_airfoil_cl` | função | `(m, p, alpha_deg, n_quad=4001) -> float` | Coeficiente de sustentação pela teoria de aerofólio fino, por quadratura trapezoidal. |
| `turbulent_flat_plate_cf` | função | `(reynolds) -> float` | Coeficiente médio de atrito de placa plana totalmente turbulenta. |
| `hoerner_profile_cd` | função | `(t_c, reynolds) -> float` | Arrasto de perfil de uma seção aerodinâmica com fator de forma de Hoerner. |

#### `pinneapple_physics.codegen`

Pacote que reexporta os geradores de scripts FDM e FEniCS; os scripts gerados expõem `solve(params) -> dict` e rodam sem importar a biblioteca. [sem teste direto do `__init__`]

<small>`pinneapple_physics/codegen/__init__.py` · 51 linhas · **sem teste direto**</small>

#### `pinneapple_physics.codegen.fdm_script_generator`

Gera um script NumPy de diferenças finitas explícitas para três formas curadas (calor 1D, Burgers 1D, Poisson/Laplace 2D); formas fora disso levantam `NotImplementedError`.

<small>`pinneapple_physics/codegen/fdm_script_generator.py` · 548 linhas · testes: 13 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PDEParser` | classe | `()` | Classifica um `ProblemSpec` (ou objeto equivalente) num dos três modelos FDM suportados, levantando `NotImplementedError` fora deles. |
| `FDMScriptGenerator` | classe | `(parser=None) -> None` | Renderiza o texto de um script FDM independente e executável (`generate`) ou grava em disco (`write`). |

#### `pinneapple_physics.codegen.fenics_script_generator`

Gera scripts UFL/dolfinx para difusão linear, Stokes (Taylor-Hood) ou elasticidade linear com contato por penalidade, escolhidos por uma heurística de 3 vias.

<small>`pinneapple_physics/codegen/fenics_script_generator.py` · 614 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FEniCSScriptGenerator` | classe | `()` | Renderiza scripts dolfinx independentes para difusão, Stokes ou elasticidade conforme a forma física do problema. |
| `classify_physics_shape` | função | `(spec) -> str` | Heurística que devolve "diffusion", "stokes" ou "elasticity" a partir de `pde.kind` e dos campos; o padrão é difusão. |

#### `pinneapple_physics.compile_api`

`pp.compile`: memoriza derivadas de primeira ordem dentro de uma avaliação da perda, para cada campo ser derivado uma vez por passo; as perdas são as mesmas. Fusão de operadores e planejamento de memória não são feitos.

<small>`pinneapple_physics/compile_api.py` · 137 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Benchmark` | classe | `(problem, epochs, repeats, baseline_s, compiled_s, speedup, max_param_diff, max_param_abs…)` | Resultado de `CompiledProblem.benchmark`: tempos base e compilado, speedup, diferença dos pesos e chamadas de derivada economizadas. |
| `CompiledProblem` | classe | `(problem, optimize='physics') -> None` | Problema com as otimizações decididas; seus `solve` e `loss_fn` usam o cache de derivadas. |
| `compile` | função | `(problem, optimize='physics', **preset_kwargs) -> CompiledProblem` | Prepara um `PhysicalProblem`, `ProblemSpec` ou nome de preset para resolver mais rápido. |

#### `pinneapple_physics.conservation`

Conservação como restrição dura: projeta a saída de um modelo (aditiva ou multiplicativamente) para que a integral ponderada iguale um total prescrito.

<small>`pinneapple_physics/conservation.py` · 83 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ConservationProjection` | classe | `(model, weights, mode='additive', target_fn=None)` | Módulo torch que aplica um modelo e projeta a saída para conservar exatamente o total dado por `target_fn`. |
| `integral` | função | `(field, weights, dims=1) -> Tensor` | Soma ponderada nas últimas `dims` dimensões de um campo. |
| `project_integral` | função | `(field, weights, target, mode='additive', dims=None, eps=1e-30) -> Tensor` | Corrige um campo (modo aditivo ou multiplicativo) para que sua integral ponderada iguale o alvo. |

#### `pinneapple_physics.darcy_checks`

Verificações físicas de campos de Darcy 2-D (`-div(a grad u) = f`, `u = 0` no contorno) sem solução de referência, na mesma discretização do solver de referência; alimenta `pinneapple check` (`physics = "darcy"`) e o relatório de confiança.

<small>`pinneapple_physics/darcy_checks.py` · 857 linhas · testes: 26</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CoefficientEnvelope` | classe | `(a_min, a_max, ranges, rel_tol)` | Envelope dos coeficientes de treino (faixa de `a` e de estatísticas do campo) para dizer se uma entrada está fora da distribuição. |
| `DarcyCheck` | classe | `(name, kind, values, value, limit, passed, critical, detail)` | Uma verificação sobre um lote: valores por amostra, agregado, limite, aprovado/reprovado e se é crítica. |
| `UseProfile` | classe | `(name, use, limits, reasons)` | Perfil de uso de uma previsão com os limites físicos que esse uso exige e o motivo de cada um. |
| `darcy_operator_apply` | função | `(u, a) -> np.ndarray` | Aplica o operador de volumes finitos de `-div(a grad u)` a `u`. |
| `darcy_residual` | função | `(u, a, f=1.0) -> np.ndarray` | Resíduo por célula `f - A_a u`; zero para a solução do próprio solver. |
| `residual_error_bound` | função | `(u, a, f=1.0) -> np.ndarray` | Limite certificado do erro relativo na norma de energia, só a partir do resíduo. |
| `residual_l2_bound` | função | `(u, a, f=1.0) -> np.ndarray` | Limite certificado do erro relativo em L2, só a partir do resíduo. |
| `boundary_flux_balance` | função | `(u, a, f=1.0) -> dict[str, np.ndarray]` | Fluxo líquido pelo contorno comparado com a integral da fonte (conservação discreta). |
| `boundary_values` | função | `(u) -> np.ndarray` | Valores de `u` nos quatro lados, extrapolados das duas células vizinhas. |
| `coefficient_stats` | função | `(a) -> dict[str, np.ndarray]` | Estatísticas por amostra de um campo de coeficientes, usadas na verificação de aplicabilidade. |
| `darcy_checks` | função | `(a, u, f=1.0, *, reference=None, envelope=None, tolerances=None, quantile=0.9, physics_ki…)` | Roda todas as verificações físicas (e, opcionalmente, envelope e comparação com referência) num lote de previsões. |
| `physics_metrics` | função | `(a, u, f=1.0) -> dict[str, np.ndarray]` | Valores por amostra das métricas físicas (`energy_bound`, `l2_bound`, `bc`, `flux`, `positivity`). |
| `calibrate_tolerances` | função | `(a, u, f=1.0, *, quantile=0.99, floor=None) -> dict[str, float]` | Calcula limites a partir das próprias previsões do modelo em entradas de validação dentro da distribuição. |
| `profile_verdicts` | função | `(a, u, f=1.0, *, profiles=None, quantile=0.9) -> dict[str, Any]` | Para cada perfil de uso, compara o quantil de cada métrica com o limite e decide APPROVED ou REJECT. |
| `sample_checks` | função | `(a, u, f=1.0, *, tolerances=None, envelope=None) -> list[dict[str, Any]]` | Verificações de uma única previsão no formato que `pinneapple check` usa. |
| `sample_verdict` | função | `(checks) -> str` | PASS se todas as verificações passam, senão FAIL (status de saída de `pinneapple check`). |
| `estimated_error` | função | `(u, a, f=1.0) -> np.ndarray` | Mapa do erro de `u` estimado só pelo resíduo. |
| `flux_magnitude` | função | `(u, a) -> np.ndarray` | Módulo do fluxo `\|a grad u\|` por célula a partir dos fluxos de face do solver. |
| `estimated_flux_error` | função | `(u, a, f=1.0) -> np.ndarray` | Mapa do erro estimado do fluxo, relativo ao maior fluxo. |
| `region_masks` | função | `(n, wall_cells=2, interior_from=8) -> dict[str, np.ndarray]` | Máscaras booleanas de região de parede e de interior numa grade n x n. |
| `regional_errors` | função | `(u, a, f=1.0, *, wall_cells=2, interior_from=8) -> dict[str, np.ndarray]` | Erro relativo estimado sem referência nas regiões de parede e interior. |
| `use_verdicts` | função | `(a, u, f=1.0, *, envelope=None, wall_cells=2, interior_from=8) -> dict[str, dict[str, Any…)` | Para uma previsão, quais usos ela atende e por quê, só pelo resíduo e pelo envelope. |
| `trust_report` | função | `(checks, *, model='', kinds=None)` | Decisão de implantação (`TrustReport`: APPROVED / REVIEW / REJECT) de uma lista de `DarcyCheck`. |
| `darcy_trust_report` | função | `(a, u, f=1.0, *, model='', kinds=…, **kwargs)` | Roda `darcy_checks` e devolve `(TrustReport, checks)`. |
| `save_darcy_npz` | função | `(path, a, u, **extra)` | Grava um campo de Darcy no formato de grade que `pinneapple check` lê. |
| `check_config_entry` | função | `(name, path, *, f=1.0, tolerances=None, envelope=None, coefficient='a', solution='u') -> …)` | Gera uma tabela `[[result]]` de `pinneapple-check.toml` para uma previsão de Darcy salva. |
| `nodal_darcy_residual` | função | `(u, k, f=1.0, dx=1.0, *, cropped=1, cross_denominator=None) -> np.ndarray` | Resíduo da forma nodal (diferenças finitas, layout `Darcy2D` do PhysicsNeMo) nos nós com vizinhos conhecidos. |
| `nodal_boundary_values` | função | `(u, *, cropped=1) -> np.ndarray` | Valores de contorno extrapolados até os nós fantasmas nulos, no layout nodal. |
| `nodal_flux_balance` | função | `(u, k, f=1.0, dx=1.0, *, cropped=1) -> np.ndarray` | Desequilíbrio relativo de um balanço de volume de controle em torno dos nós armazenados. |
| `solve_nodal_darcy` | função | `(k, f=1.0, dx=1.0, *, cross_denominator=None) -> np.ndarray` | Resolve diretamente a forma nodal; serve de referência para `nodal_darcy_residual`. |
| `check_nodal_darcy_batch` | função | `(k, u, f=1.0, dx=1.0, *, cropped=1, alt_cross_denominators=None, tolerances=None, floor_f…)` | Verifica um lote de pares de treino nodais `(k, u)` antes de alguém treinar com eles. |

#### `pinneapple_physics.deepxde_bridge`

Pacote da ponte DeepXDE ↔ PINNeAPPle; reexporta `solve_with_deepxde` e utilitários do `adapter`.

<small>`pinneapple_physics/deepxde_bridge/__init__.py` · 60 linhas · testes: 9</small>

#### `pinneapple_physics.deepxde_bridge.adapter`

Traduz `ProblemSpec` em geometria, dados, rede `FNN` e `dde.Model` reais do DeepXDE e chama o treino do DeepXDE; só `burgers` e `poisson` são traduzidos, o resto levanta `NotImplementedError`. DeepXDE é dependência opcional. [sem teste direto]

<small>`pinneapple_physics/deepxde_bridge/adapter.py` · 468 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DeepXDESolveResult` | classe | `(deepxde_model, spec, final_loss, train_time_s)` | Resultado de `solve_with_deepxde`: o modelo DeepXDE, o spec, a perda final e o tempo de treino. |
| `require_deepxde` | função | `() -> Any` | Importa `deepxde` ou levanta erro claro se não estiver instalado. |
| `is_deepxde_available` | função | `() -> bool` | Diz se o `deepxde` pode ser importado. |
| `solve_with_deepxde` | função | `(spec, *, n_domain=2000, n_boundary=200, n_initial=100, layers=(50, 50, 50, 50), epochs=1…)` | Traduz o `spec` num problema DeepXDE e o treina de verdade com os hiperparâmetros dados. |

#### `pinneapple_physics.ensemble`

Ensemble adaptativo de especialistas físicos: pesos atualizados online por erro contra referência, resíduo físico e custo, nos modos `select` ou `combine`, com intervalos conformais e detecção de suspeitos.

<small>`pinneapple_physics/ensemble.py` · 636 linhas · testes: 68</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Expert` | classe | `(name, predict, cost, kind)` | Um especialista do ensemble: nome, função `predict(query)`, custo por chamada e tipo. |
| `EnsembleRun` | classe | `(names, errors, ensemble_errors, weights, active, coverage, costs, predictions, suspect)` | Histórico de uma execução do ensemble (erros, pesos, ativo, cobertura, custos, suspeitos) com `summary` e `switches`. |
| `PhysicsEnsemble` | classe | `(experts, mode='combine', etas=(0.5, 2.0, 8.0), alphas=(0.0, 0.01, 0.05), point_weights=N…)` | Seleção ou combinação online de modelos físicos com pesos Fixed-Share/AdaHedge, resíduo físico, custo e intervalos conformais. |
| `relative_l2` | função | `(pred, ref, weights=None) -> float` | Erro L2 relativo, com pesos por ponto opcionais. |
| `from_callable` | função | `(name, fn, cost=None, kind='numerical') -> Expert` | Cria um `Expert` a partir de uma função qualquer. |
| `from_solution` | função | `(name, solution, points, cost=None, shape=None) -> Expert` | Cria um `Expert` a partir de um resultado de `pp.solve`, avaliado nos pontos de cada consulta. |
| `from_torch` | função | `(name, model, to_input, from_output=None, cost=None, kind='neural') -> Expert` | Cria um `Expert` a partir de um `torch.nn.Module` (FNO, DeepONet, MeshGraphNet, PINN) com adaptadores de entrada e saída. |
| `residual_baseline` | função | `(experts, domain_queries, residual_fn) -> dict[str, float]` | Resíduo físico mediano de cada especialista em casos do seu próprio domínio, referência para `PhysicsEnsemble(residual_baseline=...)`. |
| `fit_static_weights` | função | `(predictions, references, n_folds=5, point_weights=None, seed=0) -> dict[str, Any]` | Ensemble offline: pesos convexos que minimizam o erro quadrático num conjunto de validação, com validação cruzada. |

#### `pinneapple_physics.ensemble_viz`

Anima com matplotlib uma `EnsembleRun`: previsão vs. referência caso a caso, modelo escolhido e evolução dos pesos, gravando um GIF.

<small>`pinneapple_physics/ensemble_viz.py` · 131 linhas · testes: 3</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `animate_ensemble` | função | `(run, references, path='ensemble.gif', *, x=None, times=None, frames_per_case=4, fps=10, …)` | Grava um GIF da execução do ensemble e devolve o caminho. |

#### `pinneapple_physics.experiment`

`Experiment` fixa as sementes (Python, NumPy, PyTorch determinístico), resolve, pontua contra uma referência e grava registro JSON com fingerprint, versões e máquina. [sem teste direto pelo inventário; testado via `pp.Experiment`]

<small>`pinneapple_physics/experiment.py` · 174 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ExperimentResult` | classe | `(config, problem_fingerprint, solution, metrics, wall_time_s, started_at, environment, n_…)` | Registro de uma execução: config, fingerprint, solução, métricas, tempo, ambiente; salva e carrega em JSON. |
| `Experiment` | classe | `(problem, method, options, seed, reference, reference_options, n_eval, metric_names, name…)` | Método aplicado a um problema com semente, opções e referência opcional; `run` executa de forma reprodutível. |
| `run_all` | função | `(experiments) -> Dict[str, ExperimentResult]` | Roda vários experimentos e devolve os resultados por nome. |

#### `pinneapple_physics.maxwell_checks`

Verificações das equações de Maxwell sobre campos numa grade (de PINN, FNO, solver ou medição), sem referência: resíduos relativos com teste em `2h`, balanço de Poynting, contorno PEC, reciprocidade e passividade; alimenta `pinneapple check` (`physics = "maxwell"`).

<small>`pinneapple_physics/maxwell_checks.py` · 549 linhas · testes: 51</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MaxwellCheck` | classe | `(name, key, value, limit, passed, critical, detail, kind)` | Resultado de uma verificação: nome, chave, valor, limite, aprovado, criticidade e detalhe. |
| `FieldSet` | classe | `(grid, fields, *, eps=EPS0, mu=MU0, sigma=0.0, omega=None, interior=1)` | Campos numa grade tensorial com derivadas por diferenças centrais, máscara de interior e versão engrossada. |
| `relative_residual` | função | `(terms, mask=None, floor=0.0) -> float` | Resíduo relativo `\|\|soma dos termos\|\| / soma das normas`; NaN quando a lei não restringe o campo. |
| `gauss_electric` | função | `(grid, fields, *, eps=EPS0, tol=None, **kw) -> MaxwellCheck` | Verifica a lei de Gauss elétrica `div D - rho`. |
| `gauss_magnetic` | função | `(grid, fields, *, mu=MU0, tol=None, **kw) -> MaxwellCheck` | Verifica a lei de Gauss magnética `div B`. |
| `faraday` | função | `(grid, fields, *, mu=MU0, omega=None, tol=None, **kw) -> MaxwellCheck` | Verifica a lei de Faraday `curl E + dB/dt`. |
| `ampere_maxwell` | função | `(grid, fields, *, eps=EPS0, mu=MU0, sigma=0.0, omega=None, tol=None, **kw) -> MaxwellCheck` | Verifica a lei de Ampère-Maxwell `curl H - J - dD/dt`. |
| `continuity` | função | `(grid, fields, *, omega=None, tol=None, **kw) -> MaxwellCheck` | Verifica a equação de continuidade da carga `d rho/dt + div J`. |
| `poynting_balance` | função | `(grid, fields, *, eps=EPS0, mu=MU0, sigma=0.0, omega=None, tol=None) -> MaxwellCheck` | Verifica o teorema de Poynting sobre a caixa da grade. |
| `tangential_continuity` | função | `(F1, F2, normal, *, surface_current=None, tol=None, name='E') -> MaxwellCheck` | Verifica `n x (F1 - F2) = K` em pontos de interface. |
| `pec_boundary` | função | `(grid, fields, faces, *, tol=None) -> MaxwellCheck` | Verifica `n x E = 0` nas faces declaradas PEC. |
| `reciprocity` | função | `(S, *, tol=None) -> MaxwellCheck` | Verifica `S = S^T` de uma rede recíproca. |
| `passivity` | função | `(S, *, lossless=False, tol=None) -> MaxwellCheck` | Verifica passividade (maior valor singular de S no máximo 1) ou unitariedade se sem perdas. |
| `maxwell_checks` | função | `(grid, fields, *, eps=EPS0, mu=MU0, sigma=0.0, omega=None, laws=None, pec_faces=(), refer…)` | Roda todas as verificações que os campos fornecidos permitem e, opcionalmente, compara com uma referência. |
| `maxwell_trust_report` | função | `(grid, fields, *, model='', **kwargs)` | Roda `maxwell_checks` e devolve `(TrustReport, checks)` com a decisão de implantação. |
| `sample_checks` | função | `(grid, fields, **kwargs) -> list[dict[str, Any]]` | Verificações no formato que `pinneapple check` usa. |
| `sample_verdict` | função | `(checks) -> str` | PASS ou FAIL a partir da lista de verificações. |
| `save_maxwell_npz` | função | `(path, grid, fields, **extra)` | Grava eixos e componentes de campo (fasores como pares `_re`/`_im`) no formato lido por `pinneapple check`. |
| `check_config_entry` | função | `(name, path, *, parameters=None, tolerances=None) -> str` | Gera um bloco `[[result]]` de `pinneapple-check.toml` para um campo salvo. |

#### `pinneapple_physics.metrics`

`pp.metrics`: L2, L2 relativo, RMSE, MAE, erro máximo, L∞ relativo e R² por campo, mais resumo, métrica agregada para rankings e erro de rollout temporal.

<small>`pinneapple_physics/metrics.py` · 182 linhas · testes: 36</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `l2` | função | `(pred, true)` | Norma euclidiana do erro por campo. |
| `relative_l2` | função | `(pred, true)` | Erro L2 relativo por campo (`nan` quando a referência é nula). |
| `rmse` | função | `(pred, true)` | Raiz do erro quadrático médio por campo. |
| `mae` | função | `(pred, true)` | Erro absoluto médio por campo. |
| `max_abs` | função | `(pred, true)` | Maior erro absoluto por campo (norma L∞ do erro). |
| `relative_linf` | função | `(pred, true)` | Erro máximo relativo ao máximo da referência, por campo. |
| `r2` | função | `(pred, true)` | Coeficiente de determinação por campo. |
| `per_field` | função | `(metric, pred, true, fields) -> dict[str, float]` | Aplica uma métrica e devolve `{campo: valor}`. |
| `summary` | função | `(pred, true, fields, names=…) -> dict[str, dict[str, float]]` | Várias métricas de uma vez no formato `{métrica: {campo: valor}}` das tabelas de benchmark. |
| `pooled` | função | `(pred, true) -> dict[str, float]` | Um número por métrica sobre todos os campos juntos, para rankings. |
| `rollout_relative_l2` | função | `(pred_seq, true_seq, *, aggregate='steps') -> float` | Erro L2 relativo de um rollout temporal (eixo 0 é o tempo), agregado por passos ou no total. |

#### `pinneapple_physics.multiscale`

Subpacote de física multiescala para ensembles adaptativos (filtros, fechamentos, resíduos, consistência, seleção por escala); numpy puro, exceto `learned` (PyTorch).

<small>`pinneapple_physics/multiscale/__init__.py` · 90 linhas · testes: 29</small>

#### `pinneapple_physics.multiscale.closures`

Fechamentos de subfiltro para LES 2-D em forma de vorticidade (Smagorinsky, dinâmico, gradiente) e testes a priori, a posteriori e de saúde sem rótulo; a classe base interna levanta `NotImplementedError` em `nu_t`.

<small>`pinneapple_physics/multiscale/closures.py` · 381 linhas · testes: 29 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Smagorinsky` | classe | `(cs, delta, name)` | Fechamento de Smagorinsky `nu_t = (cs delta)^2 \|S\|`. |
| `DynamicSmagorinsky` | classe | `(delta, test_ratio, test_kind, name, last_c)` | Smagorinsky dinâmico com coeficiente calculado por filtro de teste (identidade de Germano, média no domínio). |
| `GradientModel` | classe | `(delta, coef, name)` | Modelo de gradiente (Clark) para a tensão de subfiltro. |
| `NoModel` | classe | `(name)` | Sem modelo de subfiltro (DNS sub-resolvida). |
| `AprioriReport` | classe | `(closure, filter, stress_correlation, tendency_correlation, transfer_true, transfer_model…)` | Resultado do teste a priori: correlações de tensão e tendência, transferência de energia e retroespalhamento. |
| `AposterioriReport` | classe | `(closure, times, spectrum_error, field_error, dissipation_les, dissipation_ref, stable, t…)` | Resultado do teste a posteriori: erros de espectro e campo, dissipação, estabilidade e crescimento de enstrofia. |
| `ClosureHealth` | classe | `(score, ok, reasons, enstrophy_ratio, tail_ratio, germano)` | Pontuação de saúde de um fechamento sem rótulos, com motivos e indicadores. |
| `strain` | função | `(w, solver) -> dict[str, np.ndarray]` | Gradientes de velocidade, componentes de deformação e `\|S\|` a partir da vorticidade. |
| `stress_tendency` | função | `(a, b, solver) -> np.ndarray` | Tendência de vorticidade `-curl(div tau)` de uma tensão desviadora. |
| `true_subfilter` | função | `(w_fine, filt, n_coarse=None, length=2 * np.pi) -> dict[str, np.ndarray]` | Termos exatos de subfiltro de um campo resolvido para um filtro dado. |
| `apriori_test` | função | `(w_fine, filt, closure, n_coarse, length=2 * np.pi) -> AprioriReport` | Teste a priori: compara tensão e tendência do fechamento com os termos exatos de subfiltro. |
| `spectrum_error` | função | `(w, w_ref, kmax, length=2 * np.pi, kmin=1) -> float` | Diferença média de `log10 E(k)` entre dois espectros numa faixa de números de onda. |
| `aposteriori_test` | função | `(les, w0_bar, ref_bar, times, dt=0.005, t_long=None) -> AposterioriReport` | Teste a posteriori: roda a LES com o fechamento e compara com a referência filtrada ao longo do tempo. |
| `closure_health` | função | `(w, w0, kc, length=2 * np.pi, *, forced=False, max_enstrophy_growth=1.05, tail_band=(0.05…)` | Saúde de uma previsão LES sem rótulos (crescimento de enstrofia, acúmulo espectral, erro de Germano). |
| `germano_error` | função | `(w, closure, solver, test_ratio=2.0, test_kind='sharp', delta=None) -> float` | Erro do fechamento na identidade de Germano entre o filtro da grade e um filtro de teste. |

#### `pinneapple_physics.multiscale.comparison`

Compara ensembles num fluxo de consultas multiescala: seleção cega à escala contra sensível à escala, com e sem verificações e modelos que ligam escalas; gera tabelas e figuras.

<small>`pinneapple_physics/multiscale/comparison.py` · 439 linhas · testes: 29</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EnsembleConfig` | classe | `(aware, router, checks, experts, error_tolerance, cost_weight)` | Configuração de um ensemble: sensível à escala ou não, roteador, verificações, especialistas, tolerância e peso de custo. |
| `evaluate_experts` | função | `(path, queries, initial_states, experts, checks=None, n_out=None) -> dict[str, np.ndarray]` | Roda cada especialista em cada consulta uma vez e guarda saídas e verificações (falhas viram NaN). |
| `adapt_output` | função | `(raw, source, target, aware, length=2 * np.pi) -> np.ndarray` | Adapta uma saída filtrada num filtro de origem ao filtro pedido, de forma cega ou sensível à escala. |
| `raw_residual` | função | `(pred, nu, dt, length=2 * np.pi) -> float` | Resíduo relativo da equação de vorticidade na escala da grade. |
| `closed_multiscale_residual` | função | `(pred, nu, dt, filt, h, length=2 * np.pi) -> float` | Resíduo da equação filtrada em duas larguras, com a tendência de subfiltro estimada pelo modelo de gradiente. |
| `run_ensembles` | função | `(queries, references, table, profiles, configs, *, dt, h, length=2 * np.pi, cost_weight=0…)` | Roda cada configuração no mesmo fluxo de consultas, reaproveitando as saídas já avaliadas. |
| `summarise_runs` | função | `(runs, profiles, configs, widths, *, wrong_factor=1.5, wrong_margin=0.03, solver=None, or…)` | Resume por configuração o erro médio, erro por largura, escolhas erradas e custo. |
| `closure_flag_table` | função | `(tables, targets, names, kc, length=2 * np.pi) -> list[dict[str, Any]]` | Tabela de como as verificações sem rótulo marcaram cada especialista e o erro quando marcado ou não. |
| `markdown_tables` | função | `(rows, widths, flags=(), sc_rows=(), bench_rows=()) -> str` | Gera as tabelas em Markdown dos resumos e estudos. |
| `summary_figure` | função | `(rows, widths, path, sc_rows=(), colors=…, unit='h') -> Path` | Figura de quatro painéis com erro por largura, erro contra custo, escolhas erradas e estudo de condicionamento. |
| `frontier_figure` | função | `(points, path, baselines=(), colors=…, title=…) -> Path` | Figura de fronteira custo-erro por configuração. |

#### `pinneapple_physics.multiscale.consistency`

Consistência entre escalas: uma previsão fina e uma grossa do mesmo estado devem concordar depois de levadas à mesma escala. [sem teste direto]

<small>`pinneapple_physics/multiscale/consistency.py` · 114 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ConsistencyReport` | classe | `(passed, field_error, spectrum_error, integrals, failed, tolerance, reasons)` | Resultado da verificação de consistência: aprovado, erros de campo e espectro, integrais e motivos. |
| `cross_scale_consistency` | função | `(fine, coarse, coarse_filter, fine_filter=None, *, length=2 * np.pi, tol=0.05, integrals=…)` | Filtra e restringe a previsão fina para a escala da grossa e compara campo, espectro e integrais. |

#### `pinneapple_physics.multiscale.filters`

Filtros espaciais (caixa, gaussiano, espectral) em grades periódicas com funções de transferência exatas, restrição, prolongamento, refiltragem e erro de comutação.

<small>`pinneapple_physics/multiscale/filters.py` · 271 linhas · testes: 29</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Filter` | classe | `(kind, delta)` | Filtro de largura `delta` e tipo dado; `delta = 0` é a identidade. |
| `transfer` | função | `(kind, k, delta) -> np.ndarray` | Função de transferência 1-D de um filtro. |
| `wavenumbers` | função | `(n, length=2 * np.pi) -> np.ndarray` | Números de onda angulares de uma grade periódica, na ordem da FFT. |
| `apply_filter` | função | `(u, filt, length=2 * np.pi, axes=None) -> np.ndarray` | Filtra um campo periódico nos eixos dados. |
| `restrict` | função | `(u, n, axes=None) -> np.ndarray` | Restrição de Fourier para uma grade periódica mais grossa, preservando a média. |
| `prolong` | função | `(u, n, axes=None) -> np.ndarray` | Interpolação de Fourier (preenchimento com zeros) para uma grade mais fina. |
| `post_filter_transfer` | função | `(source, target, k, max_gain=2.0) -> np.ndarray` | Transferência do pós-filtro que leva de um filtro de origem a um alvo, com ganho limitado. |
| `realizability_mismatch` | função | `(source, target, kmax, *, slope=-2.0, n=4096, max_gain=2.0, kmin=1.0) -> float` | Fração do filtro alvo que não pode ser obtida por pós-filtragem de um campo já filtrado. |
| `refilter` | função | `(u, source, target, length=2 * np.pi, axes=None, max_gain=2.0) -> np.ndarray` | Converte um campo filtrado num filtro para outro filtro, até onde for possível. |
| `filter_variable_width` | função | `(u, kind, delta, length=2 * np.pi, axes=None, decimals=10) -> np.ndarray` | Filtro cuja largura varia no espaço. |
| `spectral_derivative` | função | `(u, axis, length=2 * np.pi, order=1) -> np.ndarray` | Derivada espectral exata do interpolante trigonométrico. |
| `commutation_error` | função | `(u, kind, delta, axis=-1, length=2 * np.pi, axes=None) -> np.ndarray` | Erro de comutação entre filtrar e derivar (não nulo para largura variável). |
| `low_modes` | função | `(kmax) -> list[tuple[int, int]]` | Lista dos modos 2-D de baixa frequência num corte quadrado, um de cada par `k, -k`. |
| `fourier_basis` | função | `(modes, n, length=2 * np.pi) -> np.ndarray` | Base real de cossenos e senos para os modos dados numa grade n x n. |
| `fourier_coefficients` | função | `(w, modes, length=2 * np.pi) -> np.ndarray` | Coeficientes de um campo periódico na base de Fourier. |

#### `pinneapple_physics.multiscale.learned`

Modelos aprendidos que ligam escalas, em PyTorch: FNO condicionado pela largura do filtro, LES diferenciável, fechamento CNN treinado a posteriori e PINN de Fourier-Galerkin.

<small>`pinneapple_physics/multiscale/learned.py` · 501 linhas · testes: 29</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ScaleConditionedFNO2d` | classe | `(in_channels=1, out_channels=1, width=32, modes=12, layers=4, conditioned=True, delta_ref…)` | FNO periódico cujas camadas são moduladas pela largura do filtro; `conditioned=False` dá a versão cega à escala. |
| `TorchVorticityLES2D` | classe | `(n, length=2 * math.pi, dtype=torch.float64)` | Solver pseudo-espectral 2-D de vorticidade diferenciável (RK4 com fator integrante) com fechamento opcional. |
| `LearnedClosure` | classe | `(hidden=16, layers=3, kernel=5)` | Fechamento CNN periódico que prevê o fluxo de vorticidade de subfiltro. |
| `FourierGalerkinPINN` | classe | `(kmax=4, hidden=256, scale=5.0, t_ref=0.3, length=2 * math.pi)` | Rede informada pela física para as grandes escalas da vorticidade 2-D, treinada só no resíduo de Galerkin truncado, sem dados. |
| `torch_transfer` | função | `(kind, k, delta) -> torch.Tensor` | Função de transferência de filtro em torch. |
| `torch_filter` | função | `(x, kind, delta, length=2 * math.pi) -> torch.Tensor` | Filtra um tensor (B, C, n, n) com largura escalar ou por amostra. |
| `torch_refilter` | função | `(x, kind, d_from, d_to, length=2 * math.pi, max_gain=2.0) -> torch.Tensor` | Leva um tensor filtrado numa largura para outra, por amostra. |
| `filtering_consistency_loss` | função | `(model, x, d_small, d_large, kind='gaussian', length=2 * math.pi) -> torch.Tensor` | Perda de consistência: filtrar a saída na largura pequena deve dar a saída na largura grande. |
| `fit_scale_conditioned` | função | `(model, inputs, targets, kind='gaussian', steps=2000, batch=16, lr=0.002, consistency_wei…)` | Treina um modelo condicionado por escala em várias larguras, com perda relativa mais consistência. |
| `train_closure_aposteriori` | função | `(closure, solver, w0, nu, reference, dt, save_steps, steps=300, batch=8, lr=0.001, seed=0…)` | Treina um fechamento a posteriori, rolando a LES e comparando com a referência filtrada e seus espectros. |
| `as_numpy_closure` | função | `(closure, n, length=2 * math.pi)` | Embrulha um fechamento aprendido para uso em `VorticityNS2D` (numpy, sem gradiente). |
| `periodic_cnn2d` | função | `(in_channels=2, out_channels=2, width=32, dilations=(1, 2, 4, 8, 4, 2, 1)) -> nn.Sequenti…)` | CNN 2-D dilatada com preenchimento circular e GELU. |
| `torus_graph` | função | `(n, offsets=(1, 2, 4)) -> tuple[torch.Tensor, torch.Tensor]` | Arestas de um grafo de grade periódica com vizinhos em várias distâncias. |
| `fit_supervised` | função | `(forward, params, X, Y, steps, seed=0, lr=0.002, batch=16, log='') -> list[float]` | Treino supervisionado com Adam e agenda cosseno no erro quadrático relativo. |

#### `pinneapple_physics.multiscale.ns2d`

Navier-Stokes 2-D incompressível em forma de vorticidade no quadrado periódico, para DNS e LES, com espectro de energia e resíduo da equação de vorticidade.

<small>`pinneapple_physics/multiscale/ns2d.py` · 218 linhas · testes: 45</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `VorticityNS2D` | classe | `(n, nu, length, forcing, drag, closure, dealias)` | Solver pseudo-espectral 2-D de vorticidade com forçamento, arrasto linear e fechamento opcional. |
| `velocity_from_vorticity` | função | `(w, length=2 * np.pi) -> tuple[np.ndarray, np.ndarray]` | Velocidade `(u, v)` de um campo de vorticidade periódico. |
| `energy_spectrum` | função | `(w, length=2 * np.pi, kmax=None) -> tuple[np.ndarray, np.ndarray]` | Espectro de energia cinética por cascas a partir da vorticidade. |
| `random_vorticity` | função | `(rng, n, k0=4.0, energy=0.5, length=2 * np.pi, batch=None) -> np.ndarray` | Vorticidade aleatória com espectro de pico em `k0` e energia total dada. |
| `exact_advection` | função | `(w, length=2 * np.pi) -> np.ndarray` | Termo `div(u w)` sem aliasing, calculado numa grade 3/2 mais fina. |
| `vorticity_residual` | função | `(w_prev, w_next, dt, nu, length=2 * np.pi, forcing=None, drag=0.0, closure_tendency=None)…)` | Resíduo da equação de vorticidade entre dois instantâneos (trapezoidal no tempo). |

#### `pinneapple_physics.multiscale.residuals`

Resíduos físicos avaliados em várias larguras de filtro, separados em partes de subfiltro, comutação e modelo. [sem teste direto]

<small>`pinneapple_physics/multiscale/residuals.py` · 120 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MultiscaleResidual` | classe | `(filters, raw, subfilter, commutation, model)` | Curva de resíduo por largura de filtro, com exportação para dicionário e Markdown. |
| `multiscale_residuals` | função | `(state, residual_fn, filters, *, length=2 * np.pi, subfilter_fn=None, commutation_fn=None…)` | Calcula a curva de resíduo de qualquer equação a partir de funções de resíduo, subfiltro e comutação. |
| `ns2d_multiscale_residuals` | função | `(w_prev, w_next, dt, nu, filters, *, length=2 * np.pi, forcing=None, drag=0.0, subfilter=…)` | Resíduo multiescala da equação de vorticidade 2-D entre dois instantâneos. |

#### `pinneapple_physics.multiscale.scales`

Seleção de modelos sensível à escala: cada modelo declara as escalas que resolve e cada consulta as que precisa. [sem teste direto]

<small>`pinneapple_physics/multiscale/scales.py` · 245 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ScaleProfile` | classe | `(name, filter, grid_spacing, closure, re_range, time_step, fidelity, cost, notes)` | Escalas que um modelo representa: filtro da saída, espaçamento da grade, fechamento, faixa de Reynolds, passo de tempo, fidelidade e custo. |
| `ScaleQuery` | classe | `(filter, re, time_scale, kmax)` | Escalas que uma consulta pede: filtro da saída, Reynolds, escala de tempo e número de onda máximo. |
| `ScaleCheck` | classe | `(name, score, covered, mismatch, reasons)` | Resultado da compatibilidade de escala de um modelo com uma consulta. |
| `ScaleRoute` | classe | `(chosen, weights, checks, awake, rung, escalated, reasons)` | Decisão de roteamento: modelo escolhido, pesos, verificações, especialistas ativos e escalonamento. |
| `ScaleAwareRouter` | classe | `(profiles, query_fn=None, *, strength=1.0, floor=0.0001, abstain_below=0.05, ladder=True,…)` | Roteador que pondera especialistas pela compatibilidade de escala, com escada de fidelidade (surrogate, LES, solver resolvido). |
| `scale_compatibility` | função | `(profile, query, *, mismatch_tol=0.25, re_band=0.15, slope=-2.0) -> ScaleCheck` | Diz se um perfil pode responder a uma consulta e quão bem suas premissas de filtragem combinam. |

#### `pinneapple_physics.multiscale.turbulence`

Turbulência 2-D em decaimento como benchmark multiescala: dados de referência, conjuntos filtrados, fluxos de consultas e benchmark de fechamentos.

<small>`pinneapple_physics/multiscale/turbulence.py` · 295 linhas · testes: 29</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DecayingTurbulence2D` | classe | `(length, n_out, t_out, spinup, k0, energy, re_bands, widths, width_probabilities, kinds, …)` | Configuração do benchmark (padrões do experimento em `docs/core_concepts/multiscale_ensembles.md`) com geração de dados e consultas. |
| `benchmark_closures` | função | `(bench, data, closures, cases_per_band=8, times=(0.1, 0.2, 0.3), t_long=2.0, dt=0.005) ->…)` | Avalia fechamentos em casos separados de cada faixa de Reynolds na grade LES. |
| `scale_conditioning_study` | função | `(bench, data, idx, conditioned, fixed, blind=None, widths=(1, 2, 4, 6, 8), kind='gaussian…)` | Erros em cada largura de filtro para modelos condicionados, fixos e cegos. |

#### `pinneapple_physics.multiscale.viz`

Animação de uma rodada de ensemble sensível à escala (escala pedida, modelo escolhido, espectros, pesos). [sem teste direto]

<small>`pinneapple_physics/multiscale/viz.py` · 111 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `animate_multiscale_ensemble` | função | `(references, predictions, active, names, weights, widths, labels, cutoffs, path='multisca…)` | Gera um GIF com um quadro por consulta da rodada do ensemble. |

#### `pinneapple_physics.non_hermitian`

Física não hermitiana: dímeros e cadeias PT-simétricos, pontos excepcionais (EPs) e se eles podem ser sintonizados; numpy, sympy para a parte simbólica e matplotlib para as figuras.

<small>`pinneapple_physics/non_hermitian/__init__.py` · 29 linhas · testes: 8</small>

#### `pinneapple_physics.non_hermitian.ep_tuning`

Verifica simbolicamente se um parâmetro de controle move o EP de um dímero PT-simétrico segundo uma lei desejada, contra o teorema de impossibilidade (registro Zenodo citado no código), e mostra três rotas para contorná-lo.

<small>`pinneapple_physics/non_hermitian/ep_tuning.py` · 226 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EPTuningReport` | classe | `(ep_law, target, residual, exact, degree, perturbation_class, assumptions, theorem_applie…)` | Relatório da análise: lei do EP, alvo, resíduo, se é exata, classe da perturbação, premissas e rotas. |
| `classify_coupling` | função | `(b_ab, b_ba=None, lam=None) -> str` | Classifica o acoplamento como constante, só de fase, recíproco polinomial, recíproco não polinomial ou não recíproco. |
| `analyze_ep_tuning` | função | `(b_ab, b_ba=None, target=None, lam=None, dimension=2) -> EPTuningReport` | Checagem simbólica de uma lei de sintonia do EP contra o alvo e o teorema. |
| `coupling_classes` | função | `() -> list[tuple[str, str, Any, Any]]` | As quatro classes de acoplamento do artigo como leis sympy prontas para `analyze_ep_tuning`. |
| `route_a_sqrt_coupling` | função | `(J, G) -> Callable[[float], complex]` | Rota A: acoplamento `sqrt(J^2 + lam J G)`, que dá exatamente a lei linear em `gamma_EP^2`. |
| `route_c_asymmetric` | função | `(J, G)` | Rota C: acoplamento não recíproco que dá a lei linear, mas quebra `H12 = conj(H21)`. |
| `feedback_coupling` | função | `(J, G, lam, b0=None, iters=60, tol=1e-14) -> tuple[float, int]` | Rota B: acoplamento atualizado por realimentação cujo ponto fixo é o acoplamento da rota A. |
| `theorem_profile` | função | `()` | O teorema como `ModelProfile` de `physics_audit`, para auditar se ele restringe um projeto. |
| `design_features` | função | `(H, polynomial) -> dict[str, float]` | Características de um Hamiltoniano de dispositivo para a auditoria de `theorem_profile`. |

#### `pinneapple_physics.non_hermitian.pt_dimer`

Dímeros e cadeias PT-simétricos: Hamiltoniano, espectro, EPs (fórmula fechada e numérico) e coalescência de autovetores. [sem teste direto]

<small>`pinneapple_physics/non_hermitian/pt_dimer.py` · 185 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PTDimer` | classe | `(coupling, coupling_ba, onsite)` | Sistema de dois sítios com ganho e perda, acoplamento constante ou função do parâmetro de controle, recíproco ou não. |
| `PTChain` | classe | `(n, b, b_back)` | Cadeia de N sítios com ganho no primeiro, perda no último e acoplamento entre vizinhos. |
| `pt_violation` | função | `(H) -> float` | Medida de violação da simetria PT de um Hamiltoniano (0 se simétrico). |
| `eigenvector_overlap` | função | `(H) -> float` | Maior sobreposição normalizada entre autovetores à direita (1 num EP). |
| `petermann_factor` | função | `(H) -> np.ndarray` | Fator de Petermann de cada modo (1 para matriz hermitiana, diverge num EP). |

#### `pinneapple_physics.non_hermitian.viz`

Figuras matplotlib para dímeros PT e sintonia de EP.

<small>`pinneapple_physics/non_hermitian/viz.py` · 121 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_spectrum` | função | `(ax, dimer, lam=0.0, gamma_max=None, n=400)` | Partes real e imaginária dos autovalores contra gamma, com o EP e a sobreposição de autovetores. |
| `plot_ep_laws` | função | `(ax, J=1.0, G=0.5, lam=None)` | `gamma_EP^2(lam)` de cada classe de acoplamento contra o alvo linear. |
| `plot_coupling_plane` | função | `(ax, J=1.0, G=0.5, lam=None)` | Trajetória do acoplamento no plano complexo; `gamma_EP` é a distância à origem. |
| `plot_chain_threshold` | função | `(ax, sizes=(2, 3, 4, 5), b_values=None)` | Limiar de quebra PT de cadeias de N sítios contra `\|b\|^2`. |
| `ep_tuning_figure` | função | `(path, J=1.0, G=0.5) -> str` | Monta e grava a figura 2x2 da sintonia de EP e devolve o caminho. |

#### `pinneapple_physics.online_learning`

Fixed-Share, AdaHedge, inferência conformal adaptativa e grade de Fixed-Share ponderada por AdaHedge, compartilhados por previsão de séries e ensembles físicos; decisões só com perdas já observadas. [sem teste direto]

<small>`pinneapple_physics/online_learning.py` · 126 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FixedShare` | classe | `(n, eta, alpha)` | Média exponencialmente ponderada com fixed share, que acompanha a melhor sequência de especialistas. |
| `AdaHedge` | classe | `(k)` | Hedge sem parâmetros que ajusta a taxa de aprendizado sozinho. |
| `AdaptiveConformal` | classe | `(level=0.9, gamma=0.01, window=500)` | Inferência conformal adaptativa sobre resíduos absolutos, mantendo a cobertura sob mudança de distribuição. |
| `FixedShareGrid` | classe | `(n, etas=(0.5, 2.0, 8.0), alphas=(0.0, 0.01, 0.05))` | Grade de agregadores Fixed-Share (eta, alpha) ponderada por AdaHedge. |

#### `pinneapple_physics.paper_problems`

Problemas do PINNsFormer (convecção, reação, onda) como o código oficial os define: domínio, grade de treino e teste e solução analítica.

<small>`pinneapple_physics/paper_problems.py` · 114 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GridCase` | classe | `(name, equation, x_range, t_range, exact, n_train, n_test, bc, lbfgs_steps, coef, source)` | Um problema de teste em grade tensorial (x, t) com referência analítica e grades de treino e teste. |
| `tensor_grid` | função | `(x_range, t_range, nx, nt) -> dict[str, np.ndarray]` | Grade `meshgrid(x, t)` achatada como no `get_data` oficial do PINNsFormer. |
| `time_sequence` | função | `(points, k=5, dt=0.0001) -> np.ndarray` | Pseudo-sequência do PINNsFormer: repete cada (x, t) k vezes com t + i·dt. |
| `convection_exact` | função | `(x, t, beta=50.0)` | Solução exata da convecção u = sin(x − βt). |
| `reaction_exact` | função | `(x, t, rho=5.0)` | Solução logística exata da equação de reação. |
| `wave_exact` | função | `(x, t, c2=4.0)` | Solução exata da equação da onda 1D do PINNsFormer. |
| `pinnsformer_case` | função | `(name) -> GridCase` | Devolve o `GridCase` do PINNsFormer pelo nome. |

#### `pinneapple_physics.paper_problems_xtfc`

Problemas do artigo X-TFC (Schiassi et al.) com resíduo, solução exata e configurações relatadas; não contém resultados do artigo.

<small>`pinneapple_physics/paper_problems_xtfc.py` · 214 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `XTFCCase` | classe | `(name, problem, exact, n_train, n_neurons, activation, weight_range, paper_iterations, pa…)` | Um problema do X-TFC: problema, solução exata, pontos de treino, neurônios, ativação e configurações do artigo. |
| `p6_rhs` | função | `(X)` | Termo fonte do problema 6 do X-TFC. |
| `p7_rhs` | função | `(X)` | Termo fonte do problema 7 do X-TFC. |
| `xtfc_case` | função | `(name) -> XTFCCase` | Devolve o `XTFCCase` pelo nome. |

#### `pinneapple_physics.pcb`

Reexporta `Board`, `Component`, `solve`, `evaluate`, `what_if`, `calibrate` do modelo térmico de PCB.

<small>`pinneapple_physics/pcb/__init__.py` · 16 linhas · testes: 10</small>

#### `pinneapple_physics.pcb.engine`

Motor do PCB Hotspot: avaliação com faixa de incerteza do coeficiente de convecção, estudos what-if, escopo do modelo e calibração por Ensemble Kalman Inversion; sem surrogate aprendido.

<small>`pinneapple_physics/pcb/engine.py` · 392 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `model_scope` | função | `(comps, env) -> Dict[str, Any]` | Lista o que o modelo cobre, para que lado cada simplificação erra e o que fazer, para aparecer em todo relatório. |
| `evaluate` | função | `(board, comps, env, *, quick=False, n_cells=60) -> Dict[str, Any]` | Avalia a placa: solve rápido em grade grossa ou completo com faixa de incerteza e verificações. |
| `field_3d` | função | `(board, out) -> Dict[str, Any]` | Monta o campo de temperatura 3D da placa para visualização a partir da saída do solver. |
| `what_if` | função | `(board, comps, env, n_cells=48) -> Dict[str, Any]` | Resolve de novo com correções comuns e mostra o efeito no componente crítico e nas junções. |
| `calibrate` | função | `(board, comps, env, measurements, *, noise_std_c=1.0, n_ensemble=16, n_iterations=8, seed…)` | Ajusta fatores de resfriamento superficial e espalhamento no plano a temperaturas medidas por Ensemble Kalman Inversion. |

#### `pinneapple_physics.pcb.solver`

Modelo térmico 3D de volumes finitos de uma PCB em camadas com componentes JEDEC de dois resistores, vias térmicas, convecção+radiação iterada e contato com chassi; resolve com CG precondicionado por AMG.

<small>`pinneapple_physics/pcb/solver.py` · 341 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Component` | classe | `(name, x_mm, y_mm, power_w, package, side, tj_max_c, theta_jb, theta_jc, w_mm, d_mm, h_mm…)` | Componente montado: posição, potência, encapsulamento, lado, resistências térmicas, dimensões e vias. |
| `Board` | classe | `(width_mm, depth_mm, n_copper, thickness_mm, outer_oz, inner_oz, outer_coverage, inner_co…)` | Placa: dimensões, camadas de cobre, espessura, cobertura de cobre e contato com chassi nas bordas. |
| `validate` | função | `(board, comps) -> List[str]` | Lista problemas de entrada da placa e dos componentes. |
| `overlaps` | função | `(comps) -> List[str]` | Pares de componentes do mesmo lado cujas pegadas se sobrepõem. |
| `solve` | função | `(board, comps, env, *, n_cells=60, h_override=None, k_xy_scale=1.0, h_scale=1.0, max_iter…)` | Temperaturas em regime permanente da placa, das junções e balanço de energia. |

#### `pinneapple_physics.pde_environment`

Ponto de entrada do ambiente de problemas: reexporta `ProblemSpec`, `PDETermSpec`, as condições de contorno, `ProblemBuilder`, o registro de presets, presets acadêmicos/CFD/industriais, os resíduos de turbulência e a base de famílias de EDP. O bloco de presets estruturais é importado dentro de `try/except ImportError`. [sem teste direto]

<small>`pinneapple_physics/pde_environment/__init__.py` · 105 linhas · **sem teste direto**</small>

#### `pinneapple_physics.pde_environment.builder`

`ProblemBuilder`: API fluente para montar um `ProblemSpec` (domínio, campos, EDP, condições, amostragem, solver) sem mexer nos dataclasses, e registrá-lo como preset.

<small>`pinneapple_physics/pde_environment/builder.py` · 638 linhas · testes: 26</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ProblemBuilder` | classe | `(name) -> None` | Construtor fluente de `ProblemSpec`: encadeia domínio, coordenadas, campos, EDP, parâmetros, condições, dados, amostragem e solver, e termina em `build()`. |

#### `pinneapple_physics.pde_environment.capabilities`

Base de conhecimento de famílias de EDP: mapeia descrições em texto livre para tipos (`kind`) de EDP e parâmetros sugeridos de `ProblemSpec`, por contagem simples de palavras-chave; usada pelo agente de design de problemas. [sem teste direto]

<small>`pinneapple_physics/pde_environment/capabilities.py` · 314 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PDEFamily` | classe | `(kind, aliases, description, default_params, coords_hint, fields_hint, tags)` | Descritor de uma família de EDP conhecida: tipo, apelidos, descrição, parâmetros padrão, dicas de coordenadas e campos, tags. |
| `list_pde_families` | função | `() -> List[str]` | Lista ordenada dos tipos canônicos de EDP. |
| `get_pde_family` | função | `(kind_or_alias) -> Optional[PDEFamily]` | Busca uma `PDEFamily` por tipo ou apelido (sem diferenciar caixa). |
| `identify_pde` | função | `(description) -> List[Tuple[str, float]]` | Pontua as famílias conhecidas contra uma descrição em texto livre por sobreposição de palavras-chave. |
| `suggest_problem_spec` | função | `(description) -> Dict[str, Any]` | Sugere argumentos de construção de `ProblemSpec` a partir de uma descrição em texto livre. |

#### `pinneapple_physics.pde_environment.condition_sampling`

Amostragem de pontos para condições cuja região é dada por uma função seletora, e normais externas da caixa do domínio.

<small>`pinneapple_physics/pde_environment/condition_sampling.py` · 68 linhas · testes: 15</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `sample_condition_points` | função | `(select, domain_bounds, coords, n, rng) -> np.ndarray` | Até n pontos do domínio aceitos por uma função seletora (máscara booleana). |
| `box_face_normals` | função | `(points, domain_bounds, coords, time_coord='t') -> Tuple[np.ndarray, np.ndarray]` | Normais unitárias externas da caixa do domínio nos pontos (zero na coordenada de tempo). |

#### `pinneapple_physics.pde_environment.conditions`

`ConditionSpec`, a especificação genérica de restrição (Dirichlet, Neumann, Robin, interface, condição inicial, dados), e as fábricas que a constroem.

<small>`pinneapple_physics/pde_environment/conditions.py` · 495 linhas · testes: 26</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ConditionSpec` | classe | `(name, kind, fields, selector_type, selector, value_fn, weight, order, deriv_coord, inter…)` | Especificação genérica de restrição: tipo, campos, seletor, função de valor, peso e dados extras (ordem, coeficientes de interface, tração). |
| `DirichletBC` | função | `(name, fields=None, selector_type='all', selector=None, value_fn=None, weight=1.0) -> Con…)` | Fábrica de condição de Dirichlet (valor imposto). |
| `NeumannBC` | função | `(name, fields=None, selector_type='all', selector=None, value_fn=None, weight=1.0, order=…)` | Fábrica de condição de Neumann (derivada normal, ou derivada repetida numa coordenada quando `order` > 1). |
| `RobinBC` | função | `(name, fields=None, selector_type='all', selector=None, value_fn=None, weight=1.0) -> Con…)` | Fábrica de condição de Robin (combinação de valor e derivada normal). |
| `InterfaceBC` | função | `(name, fields, selector_type='all', selector=None, value_fn=None, weight=1.0, k_a=1.0, k_…)` | Fábrica de condição de interface entre dois subdomínios (continuidade de valor e de fluxo com k_a, k_b). |
| `InitialCondition` | função | `(name, fields=None, selector_type='all', selector=None, value_fn=None, weight=1.0) -> Con…)` | Fábrica de condição inicial. |
| `DataConstraint` | função | `(name, fields=None, selector_type='all', selector=None, value_fn=None, weight=1.0) -> Con…)` | Fábrica de restrição de dados (perda supervisionada em pontos dados). |

#### `pinneapple_physics.pde_environment.environment_typing`

Apelidos de tipo `CoordNames` e `FieldNames` (tuplas de strings) usados pelas specs.

<small>`pinneapple_physics/pde_environment/environment_typing.py` · 6 linhas · testes: 26</small>

#### `pinneapple_physics.pde_environment.presets`

Subpacote de presets de problemas; reexporta o registro (`get_preset`, `list_presets`, `register_preset`) e o preset de terramecânica.

<small>`pinneapple_physics/pde_environment/presets/__init__.py` · 14 linhas · testes: 21</small>

#### `pinneapple_physics.pde_environment.presets._curve_geometry`

Geometria paramétrica de curvas e polígonos para os presets com tags cujo domínio não é caixa nem cilindro: pá de cascata em arco circular, silhueta tipo Ahmed body e aerofólio NACA simétrico. Módulo privado. [sem teste direto]

<small>`pinneapple_physics/pde_environment/presets/_curve_geometry.py` · 227 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `circular_arc_cascade_blade_points` | função | `(n, rng, *, flow_angle_in_deg, flow_angle_out_deg, chord, x_le, y_le) -> np.ndarray` | Amostra n pontos no contorno de uma pá de cascata em arco circular definida pelos ângulos de entrada/saída e pela corda. |
| `ahmed_body_polygon` | função | `(car_length, car_height, *, slant_angle_deg=_AHMED_DEFAULT_SLANT_ANGLE_DEG, n_arc=24, x0=…)` | Polígono fechado da silhueta de carro inspirada no Ahmed body. |
| `naca4_symmetric_polygon` | função | `(chord, thickness_ratio, *, n=80, x0=0.0, y0=0.0) -> np.ndarray` | Polígono fechado de um aerofólio NACA de 4 dígitos simétrico. |
| `polygon_perimeter_sample` | função | `(vertices, n, rng) -> np.ndarray` | Amostra pontos uniformes por comprimento de arco ao longo de um polígono fechado. |
| `polygon_contains` | função | `(vertices, X) -> np.ndarray` | Teste vetorizado de ponto dentro de polígono por lançamento de raio. |

#### `pinneapple_physics.pde_environment.presets._utils`

Auxiliar interno dos presets: constantes de Lamé a partir de E e nu. Fora da API pública. [sem teste direto]

<small>`pinneapple_physics/pde_environment/presets/_utils.py` · 10 linhas · **sem teste direto**</small>

#### `pinneapple_physics.pde_environment.presets.academics`

Presets acadêmicos registrados: Laplace 2D, Poisson 2D (fonte via `ctx['source_fn']`) e Burgers 1D.

<small>`pinneapple_physics/pde_environment/presets/academics.py` · 113 linhas · testes: 37</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `laplace_2d_default` | função | `() -> ProblemSpec` | Preset de Laplace 2D com Dirichlet na fronteira marcada por tag. |
| `poisson_2d_default` | função | `() -> ProblemSpec` | Preset de Poisson 2D; a fonte f(x,y) vem de `ctx['source_fn']`. |
| `burgers_1d_default` | função | `(nu=0.01) -> ProblemSpec` | Preset de Burgers 1D viscoso com viscosidade `nu`. |

#### `pinneapple_physics.pde_environment.presets.astroml_bridge`

Invólucro fino sobre o `astroML` (dependência opcional): periodogramas de Lomb-Scargle, deconvolução extrema, Bayesian Blocks, histograma de Knuth, ajustes de reta com erros, datasets e verificações sintéticas. Os periodogramas do astroML são marcados como deprecated (avisos silenciados).

<small>`pinneapple_physics/pde_environment/presets/astroml_bridge.py` · 576 linhas · testes: 12 · palavras-chave no código (heurística; o texto acima diz o status real): shim ou obsoleto</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Periodogram` | classe | `(freq, power, method, n_terms, n_obs, false_alarm_levels, meta)` | Periodograma numa grade de frequências, com pico, melhor período, nível de falso alarme e conversão para amostra. |
| `XDResult` | classe | `(weights, means, covariances, log_likelihood, model)` | Resultado da deconvolução extrema: mistura gaussiana da distribuição sem erro; permite amostrar. |
| `LineFit` | classe | `(slope, intercept, covariance, chi2, dof, method)` | Reta ajustada com inclinação, intercepto, covariância, chi2 e graus de liberdade. |
| `astroml_available` | função | `() -> bool` | Diz se o `astroML` pode ser importado. |
| `frequency_grid` | função | `(t, period_range=None, f_min=None, f_max=None, oversample=10.0, n_max=200000) -> np.ndarr…)` | Grade uniforme de frequências para o periodograma de amostras irregulares. |
| `lomb_scargle_periodogram` | função | `(t, y, dy, freq=None, *, period_range=None, oversample=10.0, generalized=True, significan…)` | Periodograma de Lomb-Scargle generalizado (média flutuante) do astroML. |
| `multiterm_periodogram` | função | `(t, y, dy, freq=None, *, n_terms=3, period_range=None, oversample=10.0, time_unit='d') ->…)` | Periodograma multi-harmônico do astroML. |
| `bootstrap_peak_distribution` | função | `(t, y, dy, freq, n_bootstraps=100, random_state=0, generalized=True) -> np.ndarray` | Pico mais alto do periodograma em reamostragens bootstrap dos dados. |
| `best_period` | função | `(t, y, dy, *, period_range=None, n_terms=1, oversample=10.0, refine=True) -> float` | Período do pico mais alto do periodograma. |
| `light_curve_sample` | função | `(t, y, dy, *, band='', time_unit='d', value_unit='mag', **provenance)` | Curva de luz irregular como `PhysicalSample` de domínio de pontos (não precisa de astroML). |
| `xd_fit` | função | `(X, Xerr, n_components, *, max_iter=200, tol=1e-05, random_state=0) -> XDResult` | Deconvolução extrema (`XDGMM` do astroML) de dados com covariância de erro por ponto. |
| `bayesian_block_edges` | função | `(t, x=None, sigma=None, *, fitness='events', p0=0.05, gamma=None) -> np.ndarray` | Bordas de bins por Bayesian Blocks (Scargle et al. 2013). |
| `knuth_histogram` | função | `(data) -> tuple[np.ndarray, np.ndarray, float]` | Histograma com largura de bin ótima bayesiana de Knuth. |
| `fit_line_with_errors` | função | `(x, y, dy) -> LineFit` | Mínimos quadrados ponderados com erro só em y. |
| `fit_line_tls` | função | `(x, y, dx, dy, initial=None) -> LineFit` | Ajuste de reta com erros nos dois eixos (TLS, Hogg, Bovy & Lang 2010). |
| `load_dataset` | função | `(name, *, data_home=None, download_if_missing=False, **kwargs)` | Chama um fetcher de `astroML.datasets` por nome curto; não baixa nada por padrão. |
| `check_period_recovery` | função | `(seed=0, period=0.613, n=150, noise=0.15) -> dict` | Verificação sintética: período recuperado de uma senoide irregular contra o verdadeiro. |
| `check_xd_means` | função | `(seed=0, n=600) -> dict` | Verificação sintética: erro das médias da deconvolução extrema em duas gaussianas 2D. |
| `check_block_step` | função | `(seed=0, t_step=6.3, p0=0.001) -> dict` | Verificação sintética: borda de Bayesian Blocks contra um degrau numa série ruidosa. |
| `check_line_fits` | função | `(seed=0) -> dict` | Verificação sintética: erro relativo da inclinação dos ajustes WLS e TLS. |
| `validate` | função | `(cases=None, log=print, **kw) -> list[dict]` | Roda as verificações sintéticas e registra uma tabela; precisa de astroML, sem rede. |

#### `pinneapple_physics.pde_environment.presets.astrophysics`

Presets de astrofísica e sistemas espaciais (Kepler, Clohessy-Wiltshire, J2, atitude, Lane-Emden, NFW, Sod, CR3BP, deflexão de luz, disco de Shakura-Sunyaev), cada um com solução de referência em `meta`, verificada com sympy segundo o docstring.

<small>`pinneapple_physics/pde_environment/presets/astrophysics.py` · 1.486 linhas · testes: 72</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `kepler_exact_state` | função | `(t, mu, a, e) -> Dict[str, np.ndarray]` | Trajetória kepleriana exata de dois corpos nos tempos dados (perigeu em t=0). |
| `kepler_two_body_orbit` | função | `(mu=398600.4418, a=8000.0, e=0.15) -> ProblemSpec` | Preset de órbita kepleriana de dois corpos em formulação cartesiana plana. |
| `cw_exact_velocity` | função | `(t, n, x0, y0, z0, vx0, vy0, vz0) -> Dict[str, np.ndarray]` | Velocidade relativa de Clohessy-Wiltshire em forma fechada. |
| `space_debris_cw_relative_motion` | função | `(n=0.0011, x0=1.0, y0=0.0, z0=0.2, vx0=0.0, vy0=-0.0015, vz0=0.0008) -> ProblemSpec` | Preset das equações de Clohessy-Wiltshire para movimento relativo de detritos e operações de proximidade. |
| `satellite_j2_perturbation` | função | `(mu=398600.4418, J2=0.00108262668, Re=6378.137, a=7000.0, e=0.001, inclination_deg=98.7) …)` | Preset de órbita LEO com perturbação J2. |
| `spacecraft_attitude_euler_rotation` | função | `(I1=100.0, I3=150.0, w1_0=0.05, w3_0=0.5) -> ProblemSpec` | Preset de dinâmica de atitude sem torque de corpo rígido axissimétrico. |
| `lane_emden_exact_theta` | função | `(xi, n) -> np.ndarray` | Solução fechada de Lane-Emden para n em {0, 1, 5}; levanta erro para outros n. |
| `lane_emden_polytrope` | função | `(n=1.0, xi_min=0.001, xi_max=3.0) -> ProblemSpec` | Preset da equação de Lane-Emden para estrela politrópica. |
| `nfw_potential_exact` | função | `(r, G, rho_s, rs) -> np.ndarray` | Potencial gravitacional NFW em forma fechada. |
| `nfw_source_fn` | função | `(G, rho_s, rs)` | Função fonte 4 pi G rho_NFW para o tipo de EDP 'poisson'. |
| `nfw_dark_matter_potential` | função | `(G=1.0, rho_s=1.0, rs=1.0, r_max=10.0) -> ProblemSpec` | Preset de Poisson para o potencial de um halo de matéria escura NFW. |
| `sod_exact_solution` | função | `(x, t, gamma=1.4, rho_l=1.0, u_l=0.0, p_l=1.0, rho_r=0.125, u_r=0.0, p_r=0.1, x0=0.5) -> …)` | Solução de Riemann exata do tubo de choque de Sod (algoritmo de Toro). |
| `sod_shock_tube_astro` | função | `(gamma=1.4, x0=0.5, t_end=0.2) -> ProblemSpec` | Preset do tubo de choque de Sod 1D (Euler compressível). |
| `cr3bp_omega_gradient` | função | `(x, y, mu)` | Gradiente do potencial efetivo do problema restrito de três corpos no referencial sinódico. |
| `cr3bp_collinear_lagrange_point` | função | `(mu, x0, tol=1e-14, max_iter=100) -> float` | Ponto de Lagrange colinear por Newton-Raphson 1D a partir de um chute no ramo certo. |
| `cr3bp_lagrange_points` | função | `(mu) -> Dict[str, tuple]` | Os cinco pontos de Lagrange do CR3BP plano para uma razão de massa. |
| `cr3bp_planar_synodic` | função | `(mu=0.012150585609624, dx0=0.0001, dy0=0.0, n_periods=4.0) -> ProblemSpec` | Preset do CR3BP plano Terra-Lua no referencial sinódico, adimensional. |
| `schwarzschild_light_bending_u_exact` | função | `(phi, m, b) -> np.ndarray` | Solução perturbativa de primeira ordem da geodésica nula de Schwarzschild, u(phi). |
| `schwarzschild_deflection_angle_weak_field` | função | `(GM, c, b) -> float` | Deflexão da luz em campo fraco, 4GM/(c^2 b). |
| `schwarzschild_light_bending_weak_field` | função | `(GM=1.32712440018e+20, c=299792458.0, b=695700000.0) -> ProblemSpec` | Preset da equação da órbita do fóton u(phi) no limite de campo fraco. |
| `shakura_sunyaev_flux_exact` | função | `(r, GM, Mdot, R_in) -> np.ndarray` | Perfil radial de fluxo do disco alfa de Shakura-Sunyaev. |
| `shakura_sunyaev_teff_exact` | função | `(r, GM, Mdot, R_in, sigma_SB=5.670374419e-08) -> np.ndarray` | Perfil de temperatura efetiva do disco de Shakura-Sunyaev. |
| `shakura_sunyaev_accretion_disk` | função | `(M_bh_solar=10.0, Mdot_edd_frac=0.1, eta=0.1, r_out_factor=10000.0) -> ProblemSpec` | Preset do disco de acreção fino e opticamente espesso de Shakura-Sunyaev. |

#### `pinneapple_physics.pde_environment.presets.astropy_bridge`

Unidades e transformações de coordenadas com `astropy`: constantes astronômicas em SI com procedência, quantidades com unidade, conversão para SI e transformação GCRS/ITRS.

<small>`pinneapple_physics/pde_environment/presets/astropy_bridge.py` · 296 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `astro_constant_si` | função | `(name) -> float` | Valor em SI de uma constante da tabela `ASTRO_CONSTANTS`. |
| `constant_provenance` | função | `(name) -> str` | Referência bibliográfica do astropy para uma constante. |
| `with_units` | função | `(value, unit_str=None) -> 'u.Quantity'` | Anexa unidades astropy a um número ou analisa uma string de unidade. |
| `to_si` | função | `(quantity, target_unit=None) -> float` | Converte uma Quantity (ou string como "5 Msun") em float SI ou na unidade pedida. |
| `transform_coordinates` | função | `(x, y, z, obstime, frame_in='gcrs', frame_out='itrs', vx=None, vy=None, vz=None, unit='km…)` | Transforma posição (e velocidade) cartesiana entre GCRS e ITRS com `astropy.coordinates`. |

#### `pinneapple_physics.pde_environment.presets.cfd`

Presets de CFD: Navier-Stokes incompressível 2D e 3D, cavidade 3D com tampa móvel, canal 3D de Poiseuille e tubo de Hagen-Poiseuille. [sem teste direto]

<small>`pinneapple_physics/pde_environment/presets/cfd.py` · 459 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ns_incompressible_2d_default` | função | `(Re=100.0, Umax=1.0) -> ProblemSpec` | Preset de Navier-Stokes incompressível 2D transiente (u, v, p) com número de Reynolds dado. |
| `ns_incompressible_3d_default` | função | `(Re=100.0, Umax=1.0) -> ProblemSpec` | Preset de Navier-Stokes incompressível 3D transiente (u, v, w, p). |
| `lid_driven_cavity_3d` | função | `(Re=100.0, size=1.0, lid_velocity=1.0) -> ProblemSpec` | Preset da cavidade 3D com tampa móvel. |
| `channel_flow_3d` | função | `(Re=100.0, length=2.0, height=1.0, width=1.0, Umax=1.0) -> ProblemSpec` | Preset de escoamento de Poiseuille num canal retangular 3D. |
| `pipe_flow_3d` | função | `(Re=100.0, radius=0.5, length=2.0, Umax=1.0) -> ProblemSpec` | Preset de escoamento de Hagen-Poiseuille num tubo cilíndrico. |

#### `pinneapple_physics.pde_environment.presets.electromagnetics`

Presets eletromagnéticos com solução fechada registrada em `closed_form.electromagnetics` (via `pp.solve(problem, "analytic")`): capacitor de dois dielétricos, fio, cavidade TM, efeito pelicular e espalhamento por cilindro. [sem teste direto]

<small>`pinneapple_physics/pde_environment/presets/electromagnetics.py` · 184 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `em_two_dielectric_capacitor` | função | `(V=1.0, eps1=1.0, eps2=4.0, d1=0.5, h=0.02) -> ProblemSpec` | Preset de capacitor com dois dielétricos na forma integral (`electrostatics_integral`). |
| `em_two_dielectric_capacitor_differential` | função | `(V=1.0, eps1=1.0, eps2=4.0, d1=0.5) -> ProblemSpec` | Preset do mesmo capacitor na forma diferencial (`electrostatics`). |
| `em_wire_az` | função | `(a=0.25, J=1.0, nu=1.0) -> ProblemSpec` | Preset de magnetostática do potencial A_z de um fio com corrente uniforme. |
| `em_tm_cavity` | função | `(m=1, n=1, periods=0.5) -> ProblemSpec` | Preset de modo TM numa cavidade (`maxwell_tm`). |
| `em_tm_cavity_integral` | função | `(m=1, n=1, periods=0.5, h=0.02) -> ProblemSpec` | Preset do modo TM na forma integral (`maxwell_tm_integral`). |
| `em_skin_effect` | função | `(delta=0.2, omega=1.0, nu=1.0, length=1.0) -> ProblemSpec` | Preset 1D de efeito pelicular numa placa condutora. |
| `em_cylinder_scattering` | função | `(k0=2.0 * math.pi, radius=0.25, eps_cyl=4.0, eps_cyl_i=0.0, half_width=1.0) -> ProblemSpec` | Preset de espalhamento TM por cilindro dielétrico. |

#### `pinneapple_physics.pde_environment.presets.engineering`

Presets de engenharia: bocal e carcaça de foguete, asa, aerodinâmica de carro, freio, suspensão, dissipador de CPU, PCB, ventoinha, forno industrial, refratário, combustão e data center.

<small>`pinneapple_physics/pde_environment/presets/engineering.py` · 1.169 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `rocket_nozzle_cfd` | função | `(gamma=1.4, R_gas=287.0, T_inlet=3500.0, p_inlet=10000000.0, throat_radius=0.1, exit_radi…)` | Preset de escoamento compressível axissimétrico num bocal convergente-divergente. |
| `rocket_structural` | função | `(E=200000000000.0, nu=0.3, p_internal=10000000.0, T_inner=800.0, T_outer=293.0, alpha_T=1…)` | Preset da carcaça do motor sob pressão interna e gradiente térmico. |
| `aircraft_wing_aerodynamics` | função | `(Re=5000000.0, Ma=0.3, alpha_deg=5.0, chord=1.0, rho_inf=1.225, U_inf=102.0, nu_air=1.5e-…)` | Preset 2D de aerodinâmica de aerofólio tipo NACA em baixo Mach. |
| `aircraft_wing_structural` | função | `(E=70000000000.0, nu=0.33, lift_load=50000.0, span=5.0, thickness=0.1) -> ProblemSpec` | Preset de tensão plana da longarina de asa sob carga de sustentação. |
| `car_external_aero` | função | `(U_inf=33.3, rho=1.225, nu_air=1.5e-05, car_length=4.5, car_height=1.4) -> ProblemSpec` | Preset 2D de aerodinâmica externa de um carro simplificado. |
| `car_brake_thermal` | função | `(k_disc=55.0, rho_disc=7100.0, cp_disc=500.0, q_friction=2000000.0, T_ambient=293.0, h_co…)` | Preset de condução transiente num disco de freio em frenagem de emergência. |
| `car_suspension_fatigue` | função | `(E=210000000000.0, nu=0.3, F_vertical=8000.0, F_lateral=3000.0, arm_length=0.35, arm_widt…)` | Preset de tensão plana para fadiga de braço de suspensão. |
| `cpu_heatsink_thermal` | função | `(k_aluminium=205.0, k_copper=400.0, q_cpu=200.0, T_ambient=293.0, h_fin=50.0, die_area=0.…)` | Preset 3D de condução/convecção permanente num dissipador de CPU. |
| `pcb_thermal` | função | `(k_pcb=0.3, k_z_pcb=0.25, q_components=None, T_ambient=308.0, h_natural=10.0, board_lengt…)` | Preset 2D de espalhamento de calor numa PCB com várias fontes. |
| `fan_cooler_cfd` | função | `(rpm=2000.0, blade_radius=0.05, hub_radius=0.01, rho=1.225, nu_air=1.5e-05, n_blades=7) -…)` | Preset 2D de escoamento radial numa ventoinha de PC. |
| `industrial_furnace_thermal` | função | `(T_hot_gas=1600.0, T_ambient=300.0, k_refractory=1.5, k_insulation=0.1, h_hot_gas=50.0, h…)` | Preset 3D de transferência de calor permanente na parede de um forno industrial. |
| `refractory_lining` | função | `(layers=None, T_hot=1700.0, T_cold=350.0, total_thickness=0.5) -> ProblemSpec` | Preset de parede refratária multicamada com condutividades diferentes. |
| `furnace_combustion_zone` | função | `(T_flame=2200.0, T_wall=1600.0, rho_flue=0.24, nu_flue=0.0002, U_inlet=5.0, furnace_lengt…)` | Preset simplificado de zona de combustão com liberação volumétrica de calor. |
| `datacenter_airflow_2d` | função | `(U_cold_aisle=2.5, T_cold_air=291.0, T_hot_max=318.0, Q_rack=20000.0, rack_height=2.0, ra…)` | Preset 2D de resfriamento de fileira de racks (corredor frio/quente). |
| `datacenter_server_thermal` | função | `(Q_cpu=250.0, Q_gpu=400.0, Q_ram=20.0, T_inlet_air=295.0, U_fan=3.0, h_forced=120.0, k_bo…)` | Preset 2D de placa de servidor com condução e convecção forçada. |
| `datacenter_cfd_3d` | função | `(room_length=20.0, room_width=10.0, room_height=3.0, n_racks=20, Q_per_rack=15000.0, T_su…)` | Preset 3D simplificado de sala de data center para prever pontos quentes. |

#### `pinneapple_physics.pde_environment.presets.industry`

Presets industriais 3D/2D genéricos: condução de calor permanente e transiente, elasticidade linear, Darcy, Helmholtz acústico, onda ultrassônica e reação-difusão. [sem teste direto]

<small>`pinneapple_physics/pde_environment/presets/industry.py` · 347 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `steady_heat_conduction_3d_default` | função | `() -> ProblemSpec` | Preset de condução de calor 3D permanente (tipo 'poisson'). |
| `transient_heat_3d_default` | função | `() -> ProblemSpec` | Preset de equação do calor 3D transiente. |
| `linear_elasticity_3d_default` | função | `() -> ProblemSpec` | Preset de elasticidade linear 3D com constantes de Lamé unitárias. |
| `darcy_pressure_only_3d_default` | função | `() -> ProblemSpec` | Preset de Darcy 3D só com pressão. |
| `helmholtz_acoustics_3d_default` | função | `() -> ProblemSpec` | Preset de Helmholtz 3D para acústica. |
| `wave_ultrasound_3d_default` | função | `() -> ProblemSpec` | Preset de equação da onda 3D (ultrassom). |
| `reaction_diffusion_2d_default` | função | `() -> ProblemSpec` | Preset de reação-difusão 2D montado como equação do calor; o termo de reação deve vir de `ctx['source_fn']`. |

#### `pinneapple_physics.pde_environment.presets.multidisciplinary`

Presets multidisciplinares: água rasa atmosférica, giro oceânico de Stommel, fônons, fratura por campo de fase, Black-Scholes, Heston, farmacocinética, difusão de fármaco, SIR e dinâmica de opinião. [sem teste direto]

<small>`pinneapple_physics/pde_environment/presets/multidisciplinary.py` · 1.100 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `climate_atmosphere_2d` | função | `(g=9.81, f=0.0001, H0=1000.0, U_ref=10.0, lon_min=0.0, lon_max=360.0, lat_min=-90.0, lat_…)` | Preset de água rasa 2D para dinâmica atmosférica. |
| `climate_ocean_gyre` | função | `(beta=2e-11, tau0=0.1, r=1e-07, rho0=1025.0, H=500.0, L=1000000.0, W=500000.0) -> Problem…)` | Preset de giro oceânico pelo modelo de Stommel no plano beta. |
| `crystal_phonon` | função | `(k=150.0, tau=1e-12, vg=3000.0, T_hot=320.0, T_cold=280.0, L_domain=1e-06) -> ProblemSpec` | Preset 1D de transporte de fônons (Boltzmann simplificado). |
| `material_fracture_2d` | função | `(E=210000000000.0, nu=0.3, Gc=2700.0, l0=0.005, Lx=1.0, Ly=1.0) -> ProblemSpec` | Preset 2D de fratura por campo de fase (Bourdin-Francfort-Marigo). |
| `black_scholes_1d` | função | `(sigma=0.2, r=0.05, K=100.0, T=1.0) -> ProblemSpec` | Preset da EDP de Black-Scholes para opção de compra europeia. |
| `heston_pde_2d` | função | `(kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, r=0.05, K=100.0, T=1.0, S_max=300.0, v_max…)` | Preset da EDP de Heston (volatilidade estocástica) 2D. |
| `pk_two_compartment` | função | `(k12=0.5, k21=0.3, kel=0.2, V1=10.0, dose=500.0, t_end=24.0) -> ProblemSpec` | Preset de farmacocinética de dois compartimentos (sistema de EDOs). |
| `drug_diffusion_tissue` | função | `(D=1e-10, lam=5e-05, C0=1.0, Lx=0.01, Ly=0.005, t_end=3600.0) -> ProblemSpec` | Preset 2D de difusão de fármaco com metabolismo de primeira ordem. |
| `sir_epidemic` | função | `(beta=0.3, gamma=0.1, N=1000000.0, I0=100.0, t_end=200.0) -> ProblemSpec` | Preset do modelo epidêmico SIR (sistema de EDOs). |
| `opinion_dynamics_2d` | função | `(D=0.01, alpha=1.0, Lx=1.0, Ly=1.0, t_end=10.0) -> ProblemSpec` | Preset 2D de dinâmica de opinião (Hegselmann-Krause contínuo). |

#### `pinneapple_physics.pde_environment.presets.registry`

Registro que mapeia nomes de presets para funções fábrica, com decorador de registro, criação por nome e listagem.

<small>`pinneapple_physics/pde_environment/presets/registry.py` · 106 linhas · testes: 119</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `register_preset` | função | `(name)` | Decorador que registra uma função fábrica de preset por nome. |
| `get_preset` | função | `(name, **kwargs) -> ProblemSpec` | Cria um preset pelo nome, repassando argumentos. |
| `list_presets` | função | `() -> List[str]` | Lista ordenada de todos os presets registrados. |

#### `pinneapple_physics.pde_environment.presets.solid_mechanics`

Presets de mecânica dos sólidos: elasticidade axissimétrica, cilindro de parede grossa de Lamé (com solução analítica) e acoplamento roscado TC50 (corpos BOX e PIN, carga combinada). [sem teste direto]

<small>`pinneapple_physics/pde_environment/presets/solid_mechanics.py` · 669 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `axisymmetric_linear_elasticity_2d_default` | função | `(r_min=10.0, r_max=50.0, z_min=0.0, z_max=100.0, E=210000000000.0, nu=0.3, p_inner=0.0, p…)` | Preset de elasticidade linear axissimétrica em (r, z). |
| `thick_walled_cylinder_lame_default` | função | `(a=20.0, b=60.0, L=100.0, p_a=100000000.0, p_b=0.0, E=210000000000.0, nu=0.3) -> ProblemS…)` | Preset do cilindro de parede grossa sob pressão (problema de Lamé). |
| `lame_analytical` | função | `(r, a, b, p_a, p_b, E, nu) -> dict` | Solução analítica de Lamé para o cilindro de parede grossa. |
| `threaded_coupling_tc50_box_default` | função | `(clearance=0.1, thread_height=0.8, offset=1.0, E=210000000000.0, nu=0.3, traction_top=100…)` | Preset 2D axissimétrico do corpo BOX de um acoplamento roscado TC50. |
| `threaded_coupling_tc50_rotating_default` | função | `(body='BOX', E=210000000000.0, nu=0.3, p_inner=20000000.0, F_axial=500000.0, T_torque=400…)` | Preset do acoplamento TC50 sob carga combinada (pressão, força axial, torque, rotação). |
| `threaded_coupling_tc50_pin_default` | função | `(thread_height=0.8, E=210000000000.0, nu=0.3, traction_top=100000000.0) -> ProblemSpec` | Preset do corpo PIN do acoplamento TC50 (mesma física do BOX, outra geometria). |

#### `pinneapple_physics.pde_environment.presets.structural`

Presets estruturais: estado plano de tensão e deformação, von Mises, elasticidade 3D, acoplamento rotativo sob torção e termoelasticidade 2D. [sem teste direto]

<small>`pinneapple_physics/pde_environment/presets/structural.py` · 429 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plane_stress_2d_default` | função | `(E=210000000000.0, nu=0.3, load_x=0.0, load_y=-1000.0) -> ProblemSpec` | Preset de elasticidade linear em estado plano de tensão. |
| `plane_strain_2d_default` | função | `(E=210000000000.0, nu=0.3) -> ProblemSpec` | Preset de elasticidade linear em estado plano de deformação. |
| `von_mises_2d_default` | função | `(E=210000000000.0, nu=0.3) -> ProblemSpec` | Preset de tensão plana com a tensão de von Mises como campo extra. |
| `linear_elasticity_3d` | função | `(E=210000000000.0, nu=0.3) -> ProblemSpec` | Preset de elasticidade linear 3D para suportes e estruturas gerais. |
| `rotary_coupling_torsion_default` | função | `(E=210000000000.0, nu=0.3, r_inner=0.05, r_outer=0.08, torque=10000.0, axial_force=500000…)` | Preset de acoplamento rotativo sob torção e tração axial combinadas. |
| `thermoelasticity_2d_default` | função | `(E=70000000000.0, nu=0.33, alpha_T=2.3e-05, dT=100.0) -> ProblemSpec` | Preset 2D de termoelasticidade (expansão térmica por variação de temperatura). |

#### `pinneapple_physics.pde_environment.presets.tag_geometry`

Geometria real para presets com condições `selector_type="tag"`: só para presets cujo domínio é caixa ou cilindro descrito pelos próprios parâmetros; os demais continuam levantando `TagConditionsUnresolved` de propósito.

<small>`pinneapple_physics/pde_environment/presets/tag_geometry.py` · 816 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_tag_batch` | função | `(name, spec, *, n_col=20000, n_bc_per_face=3000, seed=7, user_ctx=None)` | Monta um lote com geometria real para um preset com fixture; KeyError se não houver fixture. |
| `tag_solve_kwargs` | função | `(name, spec, *, n_bc_per_face=1000, seed=7) -> Dict[str, Any]` | Argumentos extras de `solve_pde` que um preset precisa: vazio sem tags, a geometria da fixture, ou KeyError. |
| `solve_pde_kwargs_from_batch` | função | `(batch) -> Dict[str, Any]` | Tira `x_col` do lote e converte arrays NumPy em tensores torch para passar a `solve_pde`. |
| `per_tag_losses` | função | `(spec, model, batch, *, w_pde=1.0, w_bc=10.0) -> dict[str, float]` | Cada termo de perda de `compile_problem(spec)` num lote (total, EDP e um por condição), como floats. |
| `preset_name` | função | `(spec) -> str` | Chave de registro de onde a spec veio (`problem_id` ou `meta['preset']`), ou vazio. |
| `has_default_geometry` | função | `(spec) -> bool` | Diz se as condições com tag da spec podem ser satisfeitas pela própria fixture. |
| `default_geometry` | função | `(spec, *, n_bc_per_face=1000, seed=7) -> Dict[str, Any]` | Argumentos de `solve_pde` que satisfazem as tags com a caixa ou cilindro do preset, ou vazio. |
| `default_geometry_report` | função | `(spec) -> str` | Uma linha dizendo qual geometria um preset usa para suas fronteiras com tag. |

#### `pinneapple_physics.pde_environment.presets.terramechanics`

Resíduos físicos e preset do substituto PINN de Bekker-Wong (interação roda-solo) em terramecânica. [sem teste direto]

<small>`pinneapple_physics/pde_environment/presets/terramechanics.py` · 200 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TerramechanicsResiduals` | classe | `(c_Pa=1400.0, phi_deg=30.0, R_m=0.125, b_m=0.06, n_phys=256, R_factor=0.06, slip_range=(0…)` | Resíduos físicos chamáveis do substituto PINN de Bekker-Wong (parâmetros do solo e da roda, faixas de escorregamento e afundamento). |
| `bekker_wong_surrogate_2d` | função | `(c_Pa=1400.0, phi_deg=30.0, R_m=0.125, b_m=0.06) -> ProblemSpec` | Devolve o `ProblemSpec` do substituto PINN 2D de Bekker-Wong. |

#### `pinneapple_physics.pde_environment.presets.turbomachinery`

Presets de turbomáquinas: análise de linha média de compressor axial multiestágio, cascata 2D de Euler compressível e estágio 3D em coordenadas cilíndricas.

<small>`pinneapple_physics/pde_environment/presets/turbomachinery.py` · 480 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `axial_compressor_meanline` | função | `(num_stages=5, pressure_ratio=3.0, mass_flow_rate=4.37, rpm=10000.0, inlet_total_pressure…)` | Preset 1D de linha média termodinâmica de compressor axial multiestágio. |
| `axial_compressor_cascade_2d` | função | `(inlet_mach=0.5, flow_angle_in_deg=50.0, flow_angle_out_deg=20.0, chord=0.1, pitch_to_cho…)` | Preset 2D de Euler compressível numa cascata de pás de compressor. |
| `axial_compressor_stage_3d` | função | `(rpm=10000.0, pressure_ratio_stage=1.4, mass_flow_rate=4.37, hub_radius=0.1, tip_radius=0…)` | Preset 3D de um estágio de compressor em coordenadas cilíndricas. |

#### `pinneapple_physics.pde_environment.scales`

`ScaleSpec`: parâmetros mínimos e opcionais de adimensionalização (L, U, alpha).

<small>`pinneapple_physics/pde_environment/scales.py` · 18 linhas · testes: 69</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ScaleSpec` | classe | `(L, U, alpha)` | Parâmetros opcionais de escala: comprimento, velocidade e difusividade característicos. |

#### `pinneapple_physics.pde_environment.spec`

Dataclasses centrais `PDETermSpec` (descritor da EDP para o compilador) e `ProblemSpec` (especificação completa do problema).

<small>`pinneapple_physics/pde_environment/spec.py` · 51 linhas · testes: 154</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PDETermSpec` | classe | `(kind, fields, coords, params, meta)` | Descritor da EDP usado pelo compilador: tipo, campos, coordenadas, parâmetros e metadados. |
| `ProblemSpec` | classe | `(name, dim, coords, fields, pde, conditions, sample_defaults, scales, field_ranges, refer…)` | Especificação completa do problema: dimensão, coordenadas, campos, EDP, condições, amostragem, escalas, limites do domínio, solver e metadados. |

#### `pinneapple_physics.pde_environment.turbulence_presets`

Resíduos de fechamento de turbulência RANS/LES para treino de PINN: k-omega SST (2D/3D), Spalart-Allmaras e WALE (LES 3D), mais seleção por nome.

<small>`pinneapple_physics/pde_environment/turbulence_presets.py` · 992 linhas · testes: 49</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `KOmegaSSTResiduals` | classe | `(nu=1e-05, rho=1.0, consts=None) -> None` | Resíduos do modelo k-omega SST (RANS) para PINN, em 2D ou 3D conforme a entrada, com viscosidade turbulenta e resíduos de canal. |
| `SpalartAllmarasResiduals` | classe | `(nu=1e-05) -> None` | Resíduos do modelo de uma equação de Spalart-Allmaras. |
| `WALEResiduals` | classe | `(nu=1e-05, cw=0.325, delta=1.0, spatial_indices=(0, 1, 2), t_index=3) -> None` | Fechamento de subgrade WALE (Nicoud & Ducros 1999) como resíduo LES algébrico 3D. |
| `get_rans_preset` | função | `(model_name='k-omega-sst', **kwargs)` | Cria um objeto de resíduo de turbulência pelo nome; ValueError para nome desconhecido. |

#### `pinneapple_physics.pde_environment.turbulence_selector`

Seletor unificado de modelo de turbulência: só despacho e validação sobre os resíduos PINN de `turbulence_presets` e o parâmetro Smagorinsky `Cs` dos solvers LBM; não implementa física nova.

<small>`pinneapple_physics/pde_environment/turbulence_selector.py` · 171 linhas · testes: 26</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TurbulenceModel` | classe | `()` | Enum dos fechamentos disponíveis (laminar, k-omega SST, WALE, Smagorinsky). |
| `get_turbulence_closure` | função | `(model, dim=3, solver_family='pinn', **kwargs) -> Optional[Any]` | Devolve o objeto de fechamento certo (PINN) ou o `Cs` (LBM) para a família de solver; ValueError para combinação não suportada. |

#### `pinneapple_physics.physical_problem`

`PhysicalProblem` reúne geometria, EDP, condições, parâmetros com unidade e papel, quantidades de interesse e fingerprint; adapta de/para `ProblemSpec`, `PhysicsCase` e o `ProblemSpec` do problem design.

<small>`pinneapple_physics/physical_problem.py` · 534 linhas · testes: 12</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Parameter` | classe | `(name, value, unit, role, bounds, distribution, description)` | Parâmetro físico nomeado com valor, unidade, papel (fixo, projeto, incerto, desconhecido), limites e distribuição. |
| `Quantity` | classe | `(name, unit, field, description)` | Quantidade de interesse que o usuário quer da solução, com unidade e campo de origem. |
| `PhysicalProblem` | classe | `(name, coords, fields, pde, conditions, domain_bounds, geometry, geometry_kind, parameter…)` | Problema físico completo (geometria, EDP, condições, parâmetros, solver, intenção) com validação, serialização, fingerprint e adaptadores. |

#### `pinneapple_physics.physics_audit`

Auditoria física: qual modelo usar, quando confiar nele e quando sua formulação física deve ser revista; `diagnose` junta avaliação e diagnóstico num caso. Numpy puro, sem torch.

<small>`pinneapple_physics/physics_audit/__init__.py` · 171 linhas · testes: 37</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Diagnosis` | classe | `(model, features, audit, validity, attribution, recommendations, unrepresented)` | Resultado de `diagnose`: auditoria de premissas, validade, atribuição, recomendações e mecanismos não representados, com relatório legível. |
| `diagnose` | função | `(profile, features, prediction=None, query=None, monitor=None, error=None, error_toleranc…)` | Audita as premissas do perfil para o caso, roda o monitor na previsão, atribui a causa da falha e recomenda ações. |

#### `pinneapple_physics.physics_audit.assumptions`

Premissas físicas de um modelo, as condições em que cada uma vale e um auditor que as checa caso a caso. [sem teste direto; usado via `physics_audit`]

<small>`pinneapple_physics/physics_audit/assumptions.py` · 420 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Condition` | classe | `(feature, lo, hi, band, log)` | Condição `lo <= feature <= hi` com borda suave; classifica como válida, marginal ou violada e dá uma pontuação. |
| `Assumption` | classe | `(name, description, conditions, breaks_into, missing_mechanism)` | Simplificação em que um modelo se apoia, com o fenômeno que aparece quando ela falha e o mecanismo que falta. |
| `AssumptionCheck` | classe | `(name, status, score, evidence, breaks_into, missing_mechanism, features)` | Resultado da verificação de uma premissa num caso. |
| `Envelope` | classe | `(ranges, samples, sample_features, band)` | Onde um modelo foi treinado ou validado: faixas por característica e, opcionalmente, amostras para distância ao vizinho mais próximo. |
| `ModelProfile` | classe | `(name, family, equations, assumptions, mechanisms, envelope, cost, notes)` | O que a auditoria sabe de um modelo: família, equações, premissas, mecanismos, envelope e custo. |
| `AssumptionAudit` | classe | `(model, status, compatibility, checks, envelope, triggered, missing_mechanisms)` | Resultado da auditoria de um modelo num caso: status, compatibilidade, verificações e mecanismos ausentes. |
| `laminar` | função | `(re_crit=2300.0, feature='Re', geometry='pipe') -> Assumption` | Premissa de escoamento laminar abaixo do Reynolds de transição. |
| `turbulent` | função | `(re_min=4000.0, re_max=math.inf, feature='Re') -> Assumption` | Premissa de escoamento totalmente turbulento acima de um Reynolds mínimo. |
| `hydraulically_smooth` | função | `(eps_plus_max=5.0) -> Assumption` | Premissa de parede hidraulicamente lisa (Reynolds de rugosidade baixo). |
| `incompressible` | função | `(ma_max=0.3) -> Assumption` | Premissa de incompressibilidade por limite no número de Mach. |
| `creeping_flow` | função | `(re_max=1.0) -> Assumption` | Premissa de escoamento de Stokes (Reynolds muito baixo). |
| `continuum` | função | `(kn_max=0.01) -> Assumption` | Premissa de meio contínuo por limite no número de Knudsen. |
| `boussinesq` | função | `(max_beta_dt=0.1) -> Assumption` | Premissa de Boussinesq por limite em `beta dT`. |
| `constant_properties` | função | `(max_dt_over_t=0.1) -> Assumption` | Premissa de propriedades constantes por limite na variação relativa de temperatura. |
| `small_strain` | função | `(max_strain=0.01) -> Assumption` | Premissa de pequenas deformações. |
| `linear_elastic` | função | `(max_stress_ratio=1.0) -> Assumption` | Premissa de material linear elástico por limite na razão de tensões. |
| `lumped_capacitance` | função | `(bi_max=0.1) -> Assumption` | Premissa de capacitância concentrada por limite no número de Biot. |
| `forced_convection` | função | `(ri_max=0.1) -> Assumption` | Premissa de convecção forçada por limite no número de Richardson. |
| `diffusion_dominated` | função | `(pe_max=1.0) -> Assumption` | Premissa de transporte dominado por difusão (Péclet baixo). |
| `advection_dominated` | função | `(pe_min=10.0) -> Assumption` | Premissa de transporte dominado por advecção (Péclet alto). |
| `attached_flow` | função | `(alpha_stall_deg=12.0) -> Assumption` | Premissa de escoamento colado abaixo do ângulo de estol. |
| `shallow_water` | função | `(max_aspect=0.05) -> Assumption` | Premissa de águas rasas por limite na razão de aspecto. |
| `get_assumption` | função | `(a) -> Assumption` | Devolve uma `Assumption` a partir de um nome do catálogo ou do próprio objeto. |
| `audit_assumptions` | função | `(profile, features) -> AssumptionAudit` | Checa todas as premissas e o envelope de um perfil para as características de um caso. |

#### `pinneapple_physics.physics_audit.attribution`

Atribuição de falhas físicas: a causa provável de uma previsão errada ou suspeita, com evidências e confiança. [sem teste direto; usado via `physics_audit`]

<small>`pinneapple_physics/physics_audit/attribution.py` · 246 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Cause` | classe | `(code, title, probability, score, evidence)` | Uma causa candidata com código, título, probabilidade, pontuação e evidências. |
| `Attribution` | classe | `(failed, causes, confidence, evidence_available, error, notes)` | Resultado da atribuição: se falhou, causas ordenadas, confiança e evidências disponíveis. |
| `attribute_failure` | função | `(*, validity=None, audit=None, error=None, error_tolerance=0.05, residual_baseline=None, …)` | Pondera as evidências de um caso (validade, auditoria, erro, resolução, medição, dispersão) e estima a causa. |

#### `pinneapple_physics.physics_audit.improvement`

Melhoria guiada pela física: quando nenhum modelo basta, qual experimento ou mudança vale mais. [sem teste direto; usado via `physics_audit`]

<small>`pinneapple_physics/physics_audit/improvement.py` · 179 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Recommendation` | classe | `(action_id, action, cause, priority, gain, cost, rationale, details, cases)` | Ação recomendada com causa, prioridade, ganho, custo e justificativa. |
| `design_points` | função | `(features, spread=0.15, n=5, seed=0, keys=None) -> list[dict[str, float]]` | Pontos de projeto de alta fidelidade em torno de um caso, por hipercubo latino. |
| `recommend` | função | `(attribution, audit=None, graph=None, features=None, model=None, min_probability=0.1) -> …)` | Lista ações ordenadas para um caso, nomeando o mecanismo que falta quando há grafo de conhecimento. |
| `improvement_plan` | função | `(cases, top=None) -> list[Recommendation]` | Agrega recomendações de muitos casos, somando prioridades da mesma ação. |

#### `pinneapple_physics.physics_audit.knowledge`

Grafo de conhecimento físico: fenômenos, equações, premissas, parâmetros, mecanismos e famílias de modelos e suas relações. [sem teste direto; usado via `physics_audit`]

<small>`pinneapple_physics/physics_audit/knowledge.py` · 328 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Node` | classe | `(id, kind, label, data)` | Nó do grafo com id, tipo, rótulo e dados. |
| `Edge` | classe | `(src, rel, dst, data)` | Aresta do grafo com origem, relação, destino e dados. |
| `PhysicsKnowledgeGraph` | classe | `()` | Grafo de conhecimento com consultas sobre mecanismos, premissas, fenômenos disparados e mecanismos não representados. |
| `default_graph` | função | `(models=()) -> PhysicsKnowledgeGraph` | Catálogo padrão de premissas, fenômenos e equações, mais os modelos dados. |

#### `pinneapple_physics.physics_audit.monitor`

Monitor de validade física: se uma previsão ainda é plausível sem verdade de referência (resíduo, conservação, limites, contorno, invariantes, extrapolação). [sem teste direto; usado via `physics_audit`]

<small>`pinneapple_physics/physics_audit/monitor.py` · 260 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CheckResult` | classe | `(name, kind, value, tolerance, passed, critical, detail)` | Resultado de uma verificação de validade: valor, tolerância, aprovado e criticidade. |
| `ValidityCheck` | classe | `(name, kind, fn, tolerance, critical, describe)` | Verificação `fn(prediction, query) -> valor` que passa quando o valor não excede a tolerância. |
| `ValidityReport` | classe | `(status, results, alarms, index)` | Relatório de uma previsão: status, resultados, alarmes e índice. |
| `PhysicsValidityMonitor` | classe | `(checks, persistence=3, drift_ratio=3.0, drift_alpha=0.3, baseline_cases=10, baselines=No…)` | Roda um conjunto de verificações em cada previsão e guarda o histórico, com persistência e detecção de deriva. |
| `residual_check` | função | `(residual_fn, tolerance, name='pde_residual', critical=False) -> ValidityCheck` | Cria uma verificação sobre a norma do resíduo da EDP. |
| `conservation_check` | função | `(total_fn, target_fn, rel_tol=0.01, name='conservation', critical=False, abs_tol=0.0) -> …)` | Cria uma verificação de desequilíbrio relativo de uma grandeza conservada. |
| `bounds_check` | função | `(lo=None, hi=None, name='bounds', select=None, max_fraction=0.0, critical=True) -> Validi…)` | Cria uma verificação da fração de valores fora de `[lo, hi]` (crítica por padrão). |
| `boundary_check` | função | `(predict_boundary, boundary_value, rel_tol=0.01, name='boundary_condition', critical=Fals…)` | Cria uma verificação do desvio RMS relativo entre a previsão no contorno e a condição de contorno. |
| `constitutive_check` | função | `(fn, tolerance, name='constitutive', critical=False) -> ValidityCheck` | Cria uma verificação de qualquer relação constitutiva que a previsão deve satisfazer. |
| `invariant_check` | função | `(invariant_fn, expected, rel_tol=0.01, name='invariant', critical=False, abs_tol=0.0) -> …)` | Cria uma verificação da variação relativa de um invariante. |
| `envelope_check` | função | `(envelope, features_fn, min_score=0.5, name='training_envelope') -> ValidityCheck` | Cria uma verificação de extrapolação a partir da pontuação do envelope de treino. |

#### `pinneapple_physics.physics_audit.regime`

Números adimensionais que dizem em que física um caso está (Re, Ma, Pe, Kn, Gr, Pr, Ra, Ri, Bi, Fr, St, Da, rugosidade).

<small>`pinneapple_physics/physics_audit/regime.py` · 178 linhas · testes: 31</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `reynolds` | função | `(velocity, length, nu=None, *, rho=None, mu=None) -> float` | Número de Reynolds. |
| `mach` | função | `(velocity, sound_speed=None, *, gamma=1.4, R=287.05, T=None) -> float` | Número de Mach, com velocidade do som dada ou de gás ideal. |
| `peclet` | função | `(velocity, length, diffusivity) -> float` | Número de Péclet. |
| `knudsen` | função | `(mean_free_path, length) -> float` | Número de Knudsen. |
| `grashof` | função | `(g, beta, dT, length, nu) -> float` | Número de Grashof. |
| `prandtl` | função | `(nu, alpha) -> float` | Número de Prandtl. |
| `rayleigh` | função | `(g, beta, dT, length, nu, alpha) -> float` | Número de Rayleigh. |
| `richardson` | função | `(g, beta, dT, length, velocity) -> float` | Número de Richardson. |
| `biot` | função | `(h, length, k) -> float` | Número de Biot. |
| `froude` | função | `(velocity, length, g=9.81) -> float` | Número de Froude. |
| `strouhal` | função | `(frequency, length, velocity) -> float` | Número de Strouhal. |
| `dominant_frequency` | função | `(signal, dt=1.0, discard=0.5, window=True, refine=True, pad=1) -> float` | Frequência do pico espectral mais forte de um sinal amostrado (por exemplo desprendimento de vórtices). |
| `damkohler` | função | `(reaction_rate, length, velocity) -> float` | Número de Damköhler. |
| `roughness_reynolds` | função | `(re, eps_d, f=None) -> float` | Reynolds de rugosidade `eps+`, com atrito estimado por Haaland se não for dado. |
| `regime_features` | função | `(*, velocity=None, length=None, nu=None, sound_speed=None, diffusivity=None, alpha=None, …)` | Calcula todos os números adimensionais cujas entradas foram dadas, num dicionário de características. |

#### `pinneapple_physics.physics_audit.routing`

Roteamento de modelos ciente da física: pondera cada modelo pelo quanto o caso atual cabe nas suas premissas, não só pelos erros passados. [sem teste direto; usado via `physics_audit`]

<small>`pinneapple_physics/physics_audit/routing.py` · 142 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RouteDecision` | classe | `(chosen, weights, compatibility, awake, escalate, audits, reasons)` | Decisão de roteamento: modelo escolhido, pesos, compatibilidades, modelos ativos, escalonamento e motivos. |
| `PhysicsAwareRouter` | classe | `(profiles, features_fn=None, strength=1.0, floor=0.0001, abstain_below=0.05, escalate_bel…)` | Roteador que usa a compatibilidade física como prior dos pesos do ensemble, com especialistas adormecidos e escalonamento ao solver. |

#### `pinneapple_physics.pinn_solver`

Subpacote do compilador de PINN; reexporta `compile_problem`, pesos de perda, DoMINO e utilidades de PINN estocástico.

<small>`pinneapple_physics/pinn_solver/__init__.py` · 28 linhas · testes: 1</small>

#### `pinneapple_physics.pinn_solver.compiler`

Reexporta `compile_problem`, `LossWeights`, `AdaptiveWeights`, `SingleBatchDataset` e `dict_collate`.

<small>`pinneapple_physics/pinn_solver/compiler/__init__.py` · 11 linhas · testes: 5</small>

#### `pinneapple_physics.pinn_solver.compiler.autograd_ops`

Operadores diferenciais por autograd (gradiente, jacobiano, divergente, laplaciano, derivada temporal, derivada normal) com cache opcional de derivadas.

<small>`pinneapple_physics/pinn_solver/compiler/autograd_ops.py` · 153 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `derivative_cache` | função | `() -> Iterator[Dict[str, int]]` | Gerenciador de contexto que memoiza primeiras derivadas dentro de uma avaliação de perda e conta acertos e falhas. |
| `ensure_tensor` | função | `(y)` | Extrai um tensor da saída do modelo (tensor puro ou objeto com atributo de saída). |
| `grad` | função | `(y, x) -> torch.Tensor` | Gradiente de um campo escalar em relação às entradas, com grafo para derivadas de ordem maior. |
| `jacobian` | função | `(Y, x) -> torch.Tensor` | Jacobiano de um campo vetorial em relação às entradas. |
| `divergence` | função | `(V, x, coord_indices=None) -> torch.Tensor` | Divergente de um campo vetorial. |
| `laplacian` | função | `(y, x, coord_indices=None) -> torch.Tensor` | Laplaciano (soma das segundas derivadas não mistas). |
| `time_derivative` | função | `(y, x, t_index) -> torch.Tensor` | Derivada em relação à coordenada de tempo. |
| `norm_dot_grad` | função | `(y, x, normals) -> torch.Tensor` | Derivada normal `n . grad y`. |
| `mse` | função | `(a, b) -> torch.Tensor` | Erro quadrático médio entre dois tensores. |

#### `pinneapple_physics.pinn_solver.compiler.compile`

O compilador central: transforma um `ProblemSpec` numa função de perda por termo, com resíduos para dezenas de tipos de EDP (Laplace, Poisson, Navier-Stokes, elasticidade, Maxwell, Black-Scholes, órbitas, etc.) e perdas de condições Dirichlet, Neumann, Robin, interface, inicial e dados; tipo desconhecido levanta `ValueError`.

<small>`pinneapple_physics/pinn_solver/compiler/compile.py` · 2.604 linhas · testes: 121</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `compile_problem` | função | `(spec, *, weights=None, cache_derivatives=False)` | Compila o `spec` em `loss_fn(model, y_hat, batch) -> {nome: perda}`; `cache_derivatives=True` compartilha derivadas entre termos. |

#### `pinneapple_physics.pinn_solver.compiler.dataset`

Dataset de lote único e função de collate para lotes PINN em dicionário. [sem teste direto]

<small>`pinneapple_physics/pinn_solver/compiler/dataset.py` · 89 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SingleBatchDataset` | classe | `(batch)` | Embrulha um lote pronto (por exemplo de `STLDomainBatchBuilder`) num `Dataset`, ciclando pontos de contorno, inicial e dados. |
| `dict_collate` | função | `(batch) -> Dict[str, Any]` | Junta uma lista de lotes PINN em dicionário (usa `pinneapple_data.collate` se disponível). |

#### `pinneapple_physics.pinn_solver.compiler.em_kinds`

Tipos eletromagnéticos do compilador: equações de Maxwell em forma diferencial (eletrostática, magnetostática, correntes parasitas, TM, 3-D, frequência) e integral (volumes de controle). [sem teste direto]

<small>`pinneapple_physics/pinn_solver/compiler/em_kinds.py` · 261 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `em_residuals` | função | `(kind, *, fields, xcol, coords, params, ctx, model, field_names, split_fields) -> list[to…)` | Devolve a lista de resíduos nos pontos de colocação para um tipo eletromagnético. |

#### `pinneapple_physics.pinn_solver.compiler.loss`

Pesos de perda: `LossWeights` fixos lidos por `compile_problem` e `AdaptiveWeights` autonormalizados para laços de treino próprios.

<small>`pinneapple_physics/pinn_solver/compiler/loss.py` · 107 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LossWeights` | classe | `(w_pde, w_bc, w_ic, w_data)` | Pesos fixos dos termos de EDP, contorno, inicial e dados. |
| `AdaptiveWeights` | classe | `(base_weights, momentum, eps, max_ratio, enabled, _ema)` | Pesos adaptativos por nome de termo, a partir de média móvel exponencial defasada da perda bruta, limitados por `max_ratio`. |

#### `pinneapple_physics.pinn_solver.domino`

DoMINO: decomposição de domínio para PINN, com um PINN por subdomínio retangular sobreposto e perda de continuidade na interface. [sem teste direto]

<small>`pinneapple_physics/pinn_solver/domino.py` · 385 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Subdomain` | classe | `(bounds, overlap=0.1)` | Subdomínio retangular com sobreposição e testes de pertinência e interface. |
| `SubdomainPINN` | classe | `(subdomain, model)` | PINN de um único subdomínio. |
| `DoMINO` | classe | `(subdomains, model_factory, interface_weight=10.0)` | Treinador de PINN por decomposição de domínio: particiona, combina as saídas e penaliza descontinuidade na interface. |

#### `pinneapple_physics.pinn_solver.factory`

Fábrica PINN legada baseada em SymPy; reexporta `NeuralNetwork`, `PINN`, `PINNFactory`, `PINNProblemSpec` e indica `VanillaPINN` de `pinneapple_neural` como modelo preferido.

<small>`pinneapple_physics/pinn_solver/factory/__init__.py` · 23 linhas · testes: 7</small>

#### `pinneapple_physics.pinn_solver.factory.autodiff`

Cálculo e cache de derivadas por autograd exigidas pelos átomos `Derivative` do SymPy. [sem teste direto]

<small>`pinneapple_physics/pinn_solver/factory/autodiff.py` · 166 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DerivativeKey` | classe | `(base, wrt)` | Chave estável para guardar derivadas em cache. |
| `DerivativeComputer` | classe | `(independent_vars, dependent_vars, device, coord_scales=None) -> None` | Calcula e guarda as derivadas por autograd pedidas pelas expressões SymPy. |
| `ensure_requires_grad` | função | `(inputs) -> None` | Liga `requires_grad=True` em todos os tensores de entrada. |

#### `pinneapple_physics.pinn_solver.factory.pinn_factory`

Fábrica de PINN: compila equações simbólicas e monta uma perda unificada; `NeuralNetwork` e `PINN` são shims de compatibilidade sobre `VanillaPINN`. [shim parcial]

<small>`pinneapple_physics/pinn_solver/factory/pinn_factory.py` · 412 linhas · testes: 1</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NeuralNetwork` | classe | `(num_inputs, num_outputs, num_layers, num_neurons, activation=nn.Tanh())` | [shim] MLP compatível com a API antiga `(num_inputs, num_outputs, num_layers, num_neurons)`, sobre `VanillaPINN`. |
| `PINN` | classe | `(neural_network, inverse_params_names=None, initial_guesses=None, dtype=torch.float32)` | [shim] Rede mais parâmetros inversos treináveis, mantida só por compatibilidade de API. |
| `PINNProblemSpec` | classe | `(pde_residuals, independent_vars, dependent_vars, conditions, inverse_params, loss_weight…)` | Especificação PINN independente de modelo: resíduos em string, variáveis, condições, parâmetros inversos, pesos e sinais exógenos. |
| `PINNFactory` | classe | `(spec)` | Compila as equações SymPy em resíduos torch e gera uma função de perda unificada. |

#### `pinneapple_physics.pinn_solver.factory.signals`

Provedores de sinais exógenos (dados de sensor, forçamento) para `PINNFactory`. [sem teste direto]

<small>`pinneapple_physics/pinn_solver/factory/signals.py` · 62 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TabulatedSignal` | classe | `(x, values, *, input_index=0, extrapolate='clamp')` | Interpolação linear por partes de um sinal amostrado, avaliada em torch, com extrapolação constante ou linear. |

#### `pinneapple_physics.pinn_solver.factory.sympy_backend`

Compilador de equações SymPy para lambdas torch via `lambdify`; o gradiente não atravessa a avaliação simbólica (para isso o código indica `SymbolicPDE`). [sem teste direto]

<small>`pinneapple_physics/pinn_solver/factory/sympy_backend.py` · 243 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CompiledEquation` | classe | `(expr, derivatives, call, args_order, exogenous, runtime_constants)` | Equação compilada: expressão, derivadas necessárias, função chamável e ordem dos argumentos. |
| `SympyTorchCompiler` | classe | `(independent_vars, dependent_vars, inverse_params=None, extra_namespace=None)` | Compila strings de equações em lambdas torch usando SymPy. |
| `SympyBackend` | classe | `(*, extra_namespace=None) -> None` | [shim] Embrulho de compatibilidade que só devolve um `SympyTorchCompiler`. |

#### `pinneapple_physics.pinn_solver.io`

Reexporta mapeamentos e o leitor de dados UPD; `UPDItem`, `ConditionSpec`, `SamplingSpec` e `Batch` vêm de `pinneapple_data.upd_types`. [sem teste direto]

<small>`pinneapple_physics/pinn_solver/io/__init__.py` · 28 linhas · **sem teste direto**</small>

#### `pinneapple_physics.pinn_solver.io.mappings`

Mapeamentos entre coordenadas e variáveis UPD e as variáveis independentes e dependentes do PINN, com transformações (segundos, graus para radianos, normalização). [sem teste direto]

<small>`pinneapple_physics/pinn_solver/io/mappings.py` · 197 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CoordMapping` | classe | `(ind_vars, coord_sources, coord_transform, normalize_to_unit, _deriv_scale)` | Mapeia coordenadas UPD para variáveis independentes do PINN, com transformação e escala de derivadas. |
| `VarMapping` | classe | `(dep_vars, upd_to_dep, enforce_present)` | Mapeia nomes de variáveis UPD para variáveis dependentes do PINN. |
| `PINNMapping` | classe | `(coord, var)` | Mapeamento completo de coordenadas e variáveis. |
| `build_default_mapping_atmosphere` | função | `(dep_vars, ind_vars=None, use_lev=False, time_transform='seconds_since_start', angle_tran…)` | Mapeamento padrão para fragmentos de reanálise atmosférica UPD (tempo, lat, lon e, opcionalmente, nível). |

#### `pinneapple_physics.pinn_solver.io.upd_dataset`

Leitor de dataset UPD (Zarr + JSON, via xarray) que produz amostras prontas para PINN. [sem teste direto]

<small>`pinneapple_physics/pinn_solver/io/upd_dataset.py` · 253 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `UPDDataset` | classe | `(item, mapping, device='cpu', dtype=torch.float32)` | Lê UPD e gera pontos de colocação, pontos de condição e pares supervisionados opcionais. |

#### `pinneapple_physics.pinn_solver.schemas`

Pacote vazio que contém o adaptador de schema. [sem teste direto]

<small>`pinneapple_physics/pinn_solver/schemas/__init__.py` · 1 linhas · **sem teste direto**</small>

#### `pinneapple_physics.pinn_solver.schemas.schema_adapter`

Adaptador de Physical Schema (UPD) para `PINNProblemSpec` e especificações de amostragem; o próprio docstring descreve premissas de MVP. [sem teste direto]

<small>`pinneapple_physics/pinn_solver/schemas/schema_adapter.py` · 325 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SchemaAdapterOptions` | classe | `(default_loss_weights, default_condition_weight, build_sampling_from_conditions)` | Opções da conversão de schema: pesos padrão de perda, peso de condição e se cria amostragem a partir das condições. |
| `SchemaAdapter` | classe | `(mapping, options=None) -> None` | Converte um schema físico em `PINNProblemSpec` (para `PINNFactory`) e, opcionalmente, em especificações de amostragem para `UPDDataset`. |

#### `pinneapple_physics.pinn_solver.stochastic`

Treino de PINN estocástico condicionado por código latente: a rede aprende uma distribuição de saídas físicas e casa média e covariância do ensemble com estatísticas reais.

<small>`pinneapple_physics/pinn_solver/stochastic.py` · 378 linhas · testes: 15</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LatentConditionedModel` | classe | `(base, latent_dim) -> None` | Embrulha um modelo para aceitar também um código latente `xi`, concatenado à entrada. |
| `CorrelatedLatentField` | classe | `(bounds, coord_order, spatial_axes, periodic_axes, latent_dim, grid_res=(64, 24, 32), len…)` | Campo latente espacialmente correlacionado, alternativa ao sorteio independente por ponto. |
| `sample_latent` | função | `(n, latent_dim, device=None, dtype=torch.float32) -> torch.Tensor` | Sorteia códigos latentes `N(0, I)`. |
| `ensemble_forward` | função | `(model, x, latent_dim, n_samples, latent_sampler=None) -> torch.Tensor` | Avalia várias amostras latentes em cada ponto numa única passagem em lote. |
| `mean_covariance_loss` | função | `(model, x, latent_dim, n_samples, mean_target, cov_target=None, cov_index_pairs=None, fie…)` | Perda que casa a média e, opcionalmente, a covariância Monte Carlo do ensemble latente com estatísticas fornecidas. |

#### `pinneapple_physics.solid`

Reexporta as funções de elasticidade axissimétrica. [sem teste direto do `__init__`]

<small>`pinneapple_physics/solid/__init__.py` · 13 linhas · **sem teste direto**</small>

#### `pinneapple_physics.solid.axisymmetric`

Elasticidade linear axissimétrica em (r, z) por autograd: deformações, tensões, resíduo de equilíbrio e von Mises, com torção opcional.

<small>`pinneapple_physics/solid/axisymmetric.py` · 122 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `lame_parameters` | função | `(E, nu) -> tuple[float, float]` | Calcula (λ, μ) a partir de E e ν. |
| `strains_rz` | função | `(u, x, *, r_min=None, create_graph=True) -> dict[str, torch.Tensor]` | Deformações pequenas em (r, z) por autograd, com componentes de torção quando há `u_theta`. |
| `stresses_rz` | função | `(u, x, E, nu, *, r_min=None, create_graph=True) -> dict[str, torch.Tensor]` | Tensões pela lei de Hooke a partir das deformações. |
| `equilibrium_residual_rz` | função | `(u, x, E, nu, *, r_min=None) -> torch.Tensor` | Resíduo pontual das equações de equilíbrio axissimétricas. |
| `von_mises_from_components` | função | `(s_rr, s_zz, s_tt, s_rz=0.0, s_rt=0.0, s_tz=0.0) -> Any` | Tensão de von Mises a partir das componentes de tensão. |
| `von_mises_rz` | função | `(u, x, E, nu, *, r_min=None, create_graph=False) -> torch.Tensor` | Tensão de von Mises de um campo de deslocamento `u`. |

#### `pinneapple_physics.solving`

`pp.solve`/`pp.compare` com registro de métodos (`pinn`, `fem`, `analytic`, `reference`, `exact`, `external`) e de soluções analíticas; toda `Solution` tem `predict` e `metadata` uniformes.

<small>`pinneapple_physics/solving.py` · 594 linhas · testes: 71</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MethodNotAvailable` | classe | `()` | Exceção: o método pedido não pode resolver este problema. |
| `Solution` | classe | `(problem, method, predict_fn, wall_time_s, history, info, model, grid, kind)` | Resultado de `solve`: função `predict`, tempo, histórico, modelo, grade e tipo do método, com `metadata` uniforme. |
| `Comparison` | classe | `(problem, reference, n_points, rows)` | Tabela de `compare`: métricas de cada método contra a referência nos mesmos pontos, com `best`. |
| `as_problem` | função | `(problem, **preset_kwargs) -> PhysicalProblem` | Converte preset, `ProblemSpec` ou `PhysicalProblem` em `PhysicalProblem`. |
| `register_method` | função | `(name, kind='custom')` | Decorador que registra um método de solução sob um nome e um tipo. |
| `list_methods` | função | `(kind=None) -> List[str]` | Nomes dos métodos registrados, opcionalmente filtrados por tipo. |
| `list_kinds` | função | `() -> Dict[str, List[str]]` | Métodos agrupados por tipo. |
| `register_analytic` | função | `(name, matcher)` | Decorador que registra uma solução fechada para os problemas em que `matcher` é verdadeiro. |
| `list_analytic` | função | `() -> List[str]` | Nomes das soluções analíticas registradas. |
| `analytic_for` | função | `(problem) -> Optional[str]` | Nome da solução fechada conhecida para o problema, ou `None`. |
| `solve` | função | `(problem, method='pinn', *, preset_kwargs=None, **options) -> Solution` | Resolve o problema com o método pedido (nome registrado ou callable). |
| `compare` | função | `(problem, methods, *, reference='reference', options=None, points=None, n_points=4096, se…)` | Resolve ou reaproveita cada método e pontua contra a referência nos mesmos pontos. |

#### `pinneapple_physics.symbolic_pde`

Compilador simbólico de EDP: transforma resíduos SymPy em funções PyTorch com autograd e oferece condições de contorno rígidas e suaves; reexporta `SymbolicPDE`, `HardBC` etc. [sem teste direto do __init__]

<small>`pinneapple_physics/symbolic_pde/__init__.py` · 52 linhas · **sem teste direto**</small>

#### `pinneapple_physics.symbolic_pde.bc`

Imposição de condições de contorno: rígidas por construção (`HardBC` com função distância, `PeriodicBC`, `MultiPeriodicBC`) e suaves por penalidade (`DirichletBC`, `NeumannBC`). [sem teste direto]

<small>`pinneapple_physics/symbolic_pde/bc.py` · 447 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HardBC` | classe | `(distance_fn, bc_value_fn) -> None` | Condição de contorno rígida por função distância: u = g + phi * rede, satisfeita exatamente; embrulha o modelo. |
| `PeriodicBC` | classe | `(period, dim=0) -> None` | Periodicidade exata numa coordenada por mudança de variável (seno/cosseno). |
| `MultiPeriodicBC` | classe | `(periods) -> None` | Periodicidade exata em dois ou mais eixos de uma vez. |
| `DirichletBC` | classe | `(value_fn, boundary_fn=None, weight=1.0) -> None` | Condição de Dirichlet suave por penalidade MSE. |
| `NeumannBC` | classe | `(flux_fn, normal_fn, boundary_fn=None, weight=1.0) -> None` | Condição de Neumann suave pela derivada normal via autograd. |

#### `pinneapple_physics.symbolic_pde.compiler`

`SymbolicPDE`: percorre a árvore SymPy e avalia o resíduo com `torch.autograd.grad` (gradientes fluem pela expressão), ou em grade com diferenças finitas/espectral. Levanta NotImplementedError para tipos SymPy não suportados e para coordenada solta no modo grade.

<small>`pinneapple_physics/symbolic_pde/compiler.py` · 771 linhas · testes: 5 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SymbolicPDE` | classe | `(expr, coord_syms, field_syms, param_syms=None, grad_method='autograd', grid=None) -> None` | Resíduo de EDP simbólico compilado para função PyTorch (autograd ou grade). |
| `pde_from_sympy` | função | `(expr, coords, fields, params=None, grad_method='autograd', grid=None) -> SymbolicPDE` | Cria um `SymbolicPDE` a partir de uma expressão de resíduo SymPy. |
| `auto_residual` | função | `(model, coords_tensor, derivative_fns) -> Dict[str, torch.Tensor]` | Calcula valores do campo e derivadas nomeadas via autograd. |

#### `pinneapple_physics.symbolic_pde.gradient_backends`

Backends de derivada plugáveis para `SymbolicPDE` em dados de grade: mapeia coordenadas para eixos e compõe derivadas repetidas e mistas usando `fd_derivative`/`spectral_derivative` do PINO.

<small>`pinneapple_physics/symbolic_pde/gradient_backends.py` · 130 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GridAxisSpec` | classe | `(coord_names, dims, L)` | Liga nomes de coordenadas SymPy a eixos e comprimentos de um tensor de campo em grade. |
| `grid_derivative` | função | `(field, wrt, *, grid, method) -> torch.Tensor` | Derivada parcial de ordem qualquer de um campo em grade por diferenças finitas ou método espectral. |

#### `pinneapple_physics.symbolic_pde.pinn`

Converte um sistema de equações analisado por `.text` num `PINNProblem` de `pinneapple_analysis`, com resíduo por autograd e coeficientes desconhecidos treináveis (PINN inverso).

<small>`pinneapple_physics/symbolic_pde/pinn.py` · 194 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PINNWithParameters` | classe | `()` | Placeholder substituído na primeira utilização pela classe torch que junta a rede e os coeficientes treináveis do problema inverso. [placeholder] |
| `pinn_problem` | função | `(system, bounds, *, coefficients=None, unknowns=None, width=32, depth=4, exact=None, name…)` | Monta `(PINNProblem, modelo)` para um sistema numa caixa, com coeficientes desconhecidos opcionais e solução exata opcional. |
| `has_field` | função | `(expr) -> bool` | Diz se a expressão contém algum campo (função não definida do SymPy). |

#### `pinneapple_physics.symbolic_pde.text`

Equações como texto, analisadas com segurança para SymPy via `ast` (sem `eval`/`sympify`): atalhos `u_t`, `u_xx`, `d(...)`, `lap(...)`, sistemas com condições, classificação de família e compilação sem `lambdify`.

<small>`pinneapple_physics/symbolic_pde/text.py` · 486 linhas · testes: 17</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EquationSyntaxError` | classe | `()` | Erro de texto que não é uma equação aceita; a mensagem diz o quê e onde. |
| `Context` | classe | `(coords, fields, time)` | Contexto de análise: coordenadas, campos e coordenada de tempo. |
| `Equation` | classe | `(text, lhs, rhs, coefficients, kind, on)` | Uma equação analisada: texto, lados esquerdo e direito, coeficientes, tipo e onde vale; dá o resíduo e as derivadas. |
| `EquationSystem` | classe | `(ctx, pdes, conditions, coefficients, unknowns)` | Sistema de EDPs e condições, com coeficientes e incógnitas; lista problemas, substitui valores e classifica. |
| `parse_expression` | função | `(text, ctx) -> tuple[sp.Expr, set[str]]` | Analisa um texto sem '=' como expressão SymPy e devolve os coeficientes usados. |
| `parse_equation` | função | `(text, ctx, *, kind='pde', on='') -> Equation` | Analisa um texto 'lhs = rhs' como `Equation`; uma EDP precisa envolver um campo. |
| `parse_system` | função | `(pdes, conditions=None, *, coords, fields, coefficients=None, unknowns=None, time='t') ->…)` | Analisa textos de EDP e condições num `EquationSystem`; o erro nomeia o primeiro texto inválido. |
| `classify` | função | `(system) -> dict[str, Any]` | Família de EDP de um sistema de um campo (calor, onda, Poisson, Burgers etc. ou 'custom'), ordens e linearidade. |
| `derivative_key` | função | `(dv) -> str` | Chave de ambiente de uma derivada de campo (ex.: `u__xx`). |
| `derivative_keys` | função | `(expr) -> dict[str, tuple[str, str]]` | Mapa chave -> (campo, letras) de todas as derivadas de campo na expressão. |
| `compile_expr` | função | `(expr, backend='numpy')` | Função env -> valor construída como árvore de closures (sem eval nem lambdify). |

#### `pinneapple_physics.tracking`

Envia parâmetros, métricas, tempo e o registro de um `ExperimentResult` ao MLflow, W&B ou a um callable; os trackers são dependências opcionais.

<small>`pinneapple_physics/tracking.py` · 82 linhas · testes: 3</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `flat_metrics` | função | `(result) -> Dict[str, float]` | Achata as métricas de um resultado em chaves `<métrica>/<campo>`. |
| `flat_params` | função | `(result) -> Dict[str, str]` | Achata a config de um resultado em parâmetros texto. |
| `log_result` | função | `(result, tracker, **kw) -> Any` | Envia o resultado ao tracker ("mlflow", "wandb" ou callable) e devolve o id da execução. |

#### `pinneapple_physics.tribology`

Reexporta o modelo de desgaste de Archard.

<small>`pinneapple_physics/tribology/__init__.py` · 5 linhas · testes: 20</small>

#### `pinneapple_physics.tribology.archard`

Desgaste de uma barra abaulada contra contraface rígida: lei de Archard local com pressão de camada de Winkler redistribuída a cada passo, do running-in ao desgaste estacionário. [sem teste direto pelo inventário]

<small>`pinneapple_physics/tribology/archard.py` · 136 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `WearMaterial` | classe | `(name, k, hardness_MPa, E_GPa)` | Material de desgaste com coeficiente k, dureza e módulo; `K` dá a taxa específica de desgaste. |
| `BarWear` | classe | `(material, load_N, length_mm, width_mm, crown_radius_mm, layer_mm, nx)` | Barra abaulada pressionada com carga F; `run` integra o desgaste ao longo da distância de deslizamento. |
| `winkler_parabolic_contact` | função | `(F_N, width_mm, R_mm, k_w) -> dict[str, float]` | Solução fechada do contato de folga parabólica numa camada de Winkler. |

#### `pinneapple_physics.weather`

Subpacote de previsão global do tempo; o `__init__` está vazio (os módulos são importados diretamente). [sem teste direto]

<small>`pinneapple_physics/weather/__init__.py` · 1 linhas · **sem teste direto**</small>

#### `pinneapple_physics.weather.data`

Baixa ERA5 do WeatherBench2 (bucket público, grade 64 x 32) para um array local normalizado e calcula a climatologia para escores de anomalia. [sem teste direto]

<small>`pinneapple_physics/weather/data.py` · 176 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Era5Store` | classe | `(path)` | Leitura preguiçosa (memory map) do ERA5 normalizado gravado por `download`. |
| `channel_names` | função | `(levels=LEVELS) -> list[str]` | Nomes dos canais: variáveis de altitude em cada nível mais as de superfície. |
| `download` | função | `(dest, years=(1990, 2022), levels=LEVELS, url=ERA5_64x32, log=print, normalization=None) …)` | Baixa e normaliza o ERA5 para `state.npy`, `constants.npy`, `times.npy` e `meta.json`. |
| `climatology` | função | `(store, url=CLIMATOLOGY_64x32) -> np.ndarray` | Climatologia normalizada do WeatherBench2, guardada em cache ao lado do store. |
| `climatology_at` | função | `(clim, time) -> np.ndarray` | Seleciona a climatologia do dia do ano e da hora de um instante. |

#### `pinneapple_physics.weather.evaluate`

Habilidade de previsões globais por antecedência (RMSE e ACC ponderados por área) e horizonte útil de cada previsão.

<small>`pinneapple_physics/weather/evaluate.py` · 154 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PublishedForecasts` | classe | `(store, names=None)` | Previsões publicadas no WeatherBench2 na grade 64 x 32, no layout de canais do store (só inicializações 00 e 12 UTC). |
| `rmse` | função | `(f, o, w) -> np.ndarray` | RMSE ponderado por área em latitude e longitude. |
| `acc` | função | `(f, o, clim, w) -> np.ndarray` | Coeficiente de correlação de anomalias ponderado por área. |
| `first_crossing` | função | `(leads_h, values, threshold, below=True) -> float` | Antecedência interpolada em que uma curva cruza um limiar pela primeira vez. |
| `horizons` | função | `(leads_h, acc_curve, rmse_curve, clim_rmse) -> dict` | Horizonte útil (ACC acima de 0,6) e horizonte sem habilidade (RMSE acima do da climatologia). |
| `score_forecasts` | função | `(store, clim, forecasts, init_idx, steps, variables=SCORED) -> dict` | Pontua várias previsões (RMSE e ACC por variável e antecedência) a partir de inicializações dadas. |
| `persistence` | função | `(store)` | Previsão de referência por persistência do estado inicial. |
| `climatology_forecast` | função | `(store, clim)` | Previsão de referência pela climatologia. |

#### `pinneapple_physics.weather.events`

Estudos de caso de tempo extremo e os estados ERA5 em torno deles, do store de treino ou do ERA5 ARCO em resolução completa médio na grade do modelo.

<small>`pinneapple_physics/weather/events.py` · 156 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Event` | classe | `(key, title, init, peak, var, region, center, what, score, vmin, vmax, extra_inits)` | Descrição de um evento extremo: inicialização, pico, variável, região, centro do globo e forma de resumir. |
| `by_key` | função | `(key) -> Event` | Busca um evento do catálogo pela chave. |
| `coarsen` | função | `(field, src_lat, src_lon, lat, lon) -> np.ndarray` | Média ponderada por área de um campo fino em cada célula do modelo. |
| `arco_states` | função | `(store, times, log=print) -> np.ndarray` | Estados normalizados na grade do store lidos do ERA5 ARCO, com precipitação de 6 h somada. |
| `event_states` | função | `(store, event, leads, cache_dir=None, log=print) -> tuple[np.ndarray, np.ndarray]` | Instantes e estados normalizados em torno de um evento, do store ou do ARCO. |
| `region_mask` | função | `(lat, lon, box) -> np.ndarray` | Máscara booleana de uma caixa lat/lon, tratando a passagem por 360 graus. |

#### `pinneapple_physics.weather.model`

Rede de previsão global numa grade lat-lon: U-Net com convoluções esféricas que prevê a variação de 6 horas do estado.

<small>`pinneapple_physics/weather/model.py` · 122 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SConv` | classe | `(cin, cout, k=3)` | Convolução esférica: preenchimento periódico em longitude e através dos polos antes da convolução. |
| `Block` | classe | `(cin, cout)` | Bloco residual com duas convoluções esféricas 3x3, group norm e GELU. |
| `SphereUNet` | classe | `(n_state, n_extra, widths=(128, 192, 256, 384), blocks=2, patch=1, refine=64)` | U-Net esférica, com opção de dobrar células em canais (`patch`) e camada de suavização em resolução total. |
| `sphere_pad` | função | `(x, p) -> torch.Tensor` | Preenche um tensor (B, C, lat, lon): circular em longitude e através dos polos em latitude. |
| `toa_insolation` | função | `(times, lat, lon) -> np.ndarray` | Cosseno do ângulo zenital solar cortado em 0 (insolação normalizada no topo da atmosfera). |
| `static_fields` | função | `(lat, lon, constants) -> np.ndarray` | Campos fixos de entrada: constantes (máscara terra-mar, orografia) mais seno e cosseno de latitude e longitude. |

#### `pinneapple_physics.weather.train`

Treino da rede de previsão: um passo de 6 horas primeiro, depois rollouts mais longos, com MSE ponderado por área e por canal. [sem teste direto]

<small>`pinneapple_physics/weather/train.py` · 198 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Forecaster` | classe | `(store, model)` | Rede treinada com o necessário para fazer rollout a partir de um instante ERA5, salvar e carregar. |
| `train` | função | `(store_path, out, *, train_years=(1990, 2017), val_years=(2018, 2019), widths=(192, 256, …)` | Treina por um tempo de parede dividido por uma agenda de comprimentos de rollout e guarda o melhor checkpoint na validação. |

#### `pinneapple_physics.weather.viz`

Visualizações de previsão contra verdade: globos lado a lado, antecedência correndo e faixa de habilidade (ACC), em GIF e figuras.

<small>`pinneapple_physics/weather/viz.py` · 399 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `display_units` | função | `(var, x) -> np.ndarray` | Converte variáveis para unidades de exibição (°C, decâmetros, hPa, mm). |
| `coastlines` | função | `() -> list[np.ndarray]` | Carrega as linhas de costa do arquivo `coastline_110m.npz` do pacote. |
| `orthographic` | função | `(field, lat, lon, center=(-15.0, -50.0), size=420)` | Imagem do campo no hemisfério visível centrado num ponto (projeção ortográfica). |
| `project_lines` | função | `(lines, center, size=420)` | Projeta linhas (costas) na mesma projeção ortográfica. |
| `regional` | função | `(field, lat, lon, box, size=(360, 260))` | Imagem equirretangular interpolada de uma região. |
| `regional_lines` | função | `(lines, box, size=(360, 260))` | Projeta linhas na imagem regional. |
| `forecast_gif` | função | `(path, var, truth, forecast, lat, lon, times, *, skill_leads_h, skill_acc, horizon, cente…)` | GIF animado com ERA5 e previsão lado a lado por antecedência e a faixa de habilidade. |
| `fill_gaps` | função | `(x, y) -> np.ndarray` | Interpolação linear sobre NaNs (modelos publicados a cada 12 h num eixo de 6 h). |
| `skill_figure` | função | `(path, leads_h, curves, horizons_by_model, var, metric='acc', dpi=110)` | Figura de habilidade por antecedência de cada modelo com o horizonte útil marcado. |
| `upsample_sphere` | função | `(field, lat, lon, factor=4)` | Reamostragem cúbica de um campo lat/lon, periódica em longitude, só para exibição. |
| `earth2_gif` | função | `(path, var, truth, forecast, lat, lon, times, *, lead_hours, horizon, title, center=(20.0…)` | Animação em estilo de globos iluminados com rotação, quadros interpolados e barra de antecedência colorida pelo horizonte. |

### Exemplos de uso reais do repositório

**Treinar um preset com `solve_pde` (pontos de tag vindos de geometria)** — `examples/pde_environment/07_tagged_presets_real_geometry.py`

```python
    batch = build_tag_batch(name, spec, n_col=4000, n_bc_per_face=500, seed=1)
    print("tag_masks sizes:", {k: int(v.sum()) for k, v in batch["ctx"]["tag_masks"].items()})

    before = per_tag_losses(spec, model, batch)
    print("per-tag losses BEFORE training:", before)

    kwargs = solve_pde_kwargs_from_batch(batch)
    result = pp.solve_pde(spec, model, epochs=epochs, n_collocation=4000, seed=1, lr=1e-3, **kwargs)
    losses = result["history"]["loss"]
```

**Sintonia de ponto excepcional e auditoria das premissas do teorema** — `examples/non_hermitian/ep_tuning.py`

```python
for _key, title, b, b_ba in coupling_classes():
    print(f"\n== {title}")
    print(analyze_ep_tuning(b, b_ba))

print("\n== numerical check (J = 1, G = 0.5)")
d_a = PTDimer(route_a_sqrt_coupling(1.0, 0.5))
ab, ba = route_c_asymmetric(1.0, 0.5)
d_c = PTDimer(ab, ba)
d_l = PTDimer(lambda x: 1.0 + 0.25 * x)
```

**Verificações de Darcy sem referência sobre previsões de outro framework** — `examples/vs_physicsnemo/07_physicsnemo_trains_pinneapple_checks/run.py`

```python
            ref = dc.darcy_checks(data["a"], u, DARCY["f"], reference=data["u"], envelope=envelope)
            drift = dc.darcy_checks(data["a"], u, DARCY["f"], tolerances=calibrated[m], physics_kind="drift")
```

**EDP simbólica com condição de contorno rígida** — `examples/pinn_solver/03_symbolic_pde_hard_bc.py`

```python
    hard_bc = HardBC(distance_fn=phi, bc_value_fn=g_bc)

    # ---- Base network (input: 2, output: 1) ----
    base_net = MLP(in_dim=2, out_dim=1, width=width, depth=depth).to(device=device, dtype=dtype)

    # ---- Wrapped model: satisfies u=0 on boundary by construction ----
    model = hard_bc.wrap_model(base_net)

    # ---- Residual function from SymPy expression ----
    residual_fn = pde.to_residual_fn(model)
```

**Simular uma rodada de acreção** — `examples/black_hole_weather/simulate.py`

```python
    cfg = RIAFConfig(nr=a.nr, ntheta=a.ntheta, alpha=a.alpha, viscosity=a.viscosity, torus_a=a.torus_a, seed=a.seed,
                     perturbation_kind=a.perturbation_kind, backend=a.backend, cfl=a.cfl)
    simulate_run(cfg, os.path.join(a.out, a.name), t_end=a.t_end, every=a.every, chunk=a.chunk, name=a.name,
                 log=lambda m: print(m, flush=True))
```

**Equação em texto (docstring de symbolic_pde.text)** — `pinneapple_physics/symbolic_pde/text.py`

```python
    sys_ = parse_system(["u_t = alpha*lap(u)"], [{"kind": "dirichlet", "on": "x_min", "equation": "u = 0"}],
                        coords=["x", "t"], fields=["u"], coefficients={"alpha": 0.1})
    sys_.classify()["family"]       # "heat"
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 114 de 166.
- Funções de teste em arquivos que importam algum módulo do pacote: 1.075.

Itens do catálogo de métodos (`pinneapple_catalog/method_status.json`) cujo código está neste pacote:

| id | método | status | testes contra referência | outros testes |
|---|---|---|---:|---:|
| E55 | S-N curve + Goodman + Miner | validated | 1 | 14 |
| E58 | Fin array conduction | tested | 0 | 4 |
| E63 | Helmholtz resonator | tested | 0 | 4 |
| E64 | Membrane diffusion | tested | 0 | 5 |
| E65 | Cantilever fatigue | validated | 1 | 4 |
| S38 | Maxwell physics checks of predicted fields (Gauss, Faraday, Ampere-Maxwell, continuity, Poynting, interfaces, reciprocity, passivity) | validated | 4 | 2 |
| T10 | Domain decomposition: XPINN, DoMINO/D3M | tested | 0 | 2 |

### Limitações conhecidas

- `solve_pde` não deriva regiões de condições `selector_type="tag"` a partir de `domain_bounds`: sem pontos e máscaras vindos de geometria real, levanta `TagConditionsUnresolved`.
- `compile_problem` só aceita os tipos de EDP implementados em `compile.py`/`em_kinds.py`; um tipo desconhecido levanta `ValueError("Unsupported PDE kind")`. Nos tipos eletromagnéticos diferenciais, saltos dos mapas de material não são derivados (interfaces só são impostas pelos tipos integrais).
- A fábrica SymPy (`pinn_solver.factory`) é caminho legado: `NeuralNetwork`, `PINN` e `SympyBackend` são shims de compatibilidade, e `SympyTorchCompiler` usa `lambdify`, opaco ao autograd.
- Vários módulos não têm teste direto: `pinn_solver.domino`, `pinn_solver.io`, `pinn_solver.schemas`, `factory.autodiff/signals/sympy_backend`, `compiler.dataset/em_kinds`, `multiscale.consistency/residuals/scales/viz`, `non_hermitian.pt_dimer`, `weather.data/train` e os submódulos de `physics_audit` (testados só via o `__init__`).
- Dependências opcionais: `multiscale.learned`, `pinn_solver` e `weather` precisam de PyTorch; `weather` e `pinn_solver.io` usam xarray/Zarr e acesso aos buckets públicos do Google Cloud; as figuras usam matplotlib (e scipy em `weather.viz`); `non_hermitian.ep_tuning` usa sympy.
- `weather` trabalha na grade grossa 64 x 32 do WeatherBench2; as previsões publicadas para comparação existem só nas inicializações 00 e 12 UTC.
- As correlações de `closed_form` valem nas faixas citadas nos docstrings e emitem avisos fora delas (por exemplo `particle_drag`, `pipe_flow`, `plate_fin_heatsink`); o `__init__` de `closed_form` não importa todos os módulos.
- As classes de fechamento de `multiscale.closures` herdam de uma base interna cujo `nu_t` levanta `NotImplementedError`; `schema_adapter` declara premissas de MVP no próprio docstring.
- Dependências opcionais: `sgp4` (satélites e comparação de decaimento), `pymsis` (NRLMSIS), `skyfield`/`skyfield_data` (kernel DE421, tabela IERS e validação), `numba` (backend rápido da hidrodinâmica), `playwright` (captura), `matplotlib` (figuras), `astropy` e `astroML` (bridges).
- Várias funções de `astro` usam a rede (Open-Meteo, CelesTrak, NOAA SWPC, CDS hips2fits) e levantam erro sem conexão quando não há cache.
- Módulos sem teste direto: `astro.astrometry`, `astro.nbody`, `astro.phenomena`, `astro.satellites`, `blackhole.hydro`, `blackhole.hydro_numba`, `blackhole.capture`, `blackhole.figures` e vários presets (`cfd`, `industry`, `multidisciplinary`, `solid_mechanics`, `structural`, `electromagnetics`, `terramechanics`).
- `pde_environment/__init__.py` importa `drill_pipe_torsion_default` de `presets.structural`, mas o módulo define `rotary_coupling_torsion_default`; o `try/except ImportError` engole a falha e nenhum preset estrutural é reexportado pelo pacote.
- `symbolic_pde.compiler` levanta NotImplementedError para tipos de expressão SymPy não suportados e para coordenada solta em expressões avaliadas no modo grade.
- `tag_geometry` só fornece geometria para presets com domínio de caixa ou cilindro; os outros presets com condições por tag continuam levantando `TagConditionsUnresolved` de propósito.
- `astroml_bridge` está marcado como deprecated no inventário: os periodogramas do astroML delegam ao `astropy` e emitem avisos de depreciação, que o módulo silencia.
- Aproximações declaradas nos docstrings: magnitude de Saturno sem o termo dos anéis, nutação truncada (IAU 1980, termos acima de 5 mas), brilho do crepúsculo por ajuste linear, correção bolométrica de corpo negro; `capabilities.identify_pde` usa só contagem de palavras-chave.
- `codegen.fdm_script_generator` cobre só calor 1D, Burgers 1D e Poisson/Laplace 2D; outras formas levantam `NotImplementedError`. O gerador FEniCS escolhe por heurística e cai em difusão por padrão; os scripts FEniCS precisam do dolfinx instalado para rodar.
- `deepxde_bridge` traduz apenas `burgers` e `poisson`; outros tipos de EDP e condições não suportadas levantam `NotImplementedError`. DeepXDE é dependência opcional e o `adapter` não tem teste direto.
- `pp.compile` só reaproveita derivadas de primeira ordem; fusão de operadores, batching de kernel e planejamento de memória não são feitos (dito no docstring).
- O método `reference` de `pp.solve` levanta `MethodNotAvailable` se o solver clássico não devolve campo em grade sobre todas as coordenadas.
- `tracking` depende de MLflow ou W&B instalados e configurados; são importados só quando usados.
- O modelo de PCB assume placa em ar livre, coeficientes de convecção com faixa de ±20 % assumida e valores típicos de encapsulamento quando faltam θJB/θJC; `model_scope` lista essas simplificações.
- Sem teste direto pelo inventário: `experiment`, `online_learning`, `deepxde_bridge.adapter`, `tribology.archard`, `codegen/__init__`, `solid/__init__` (vários são testados indiretamente via os pacotes que os reexportam).

## 2. `pinneapple_presets`

### Objetivo

`pinneapple_presets` reúne presets industriais: problemas de engenharia já montados, resolvidos e validados com o PINNeAPPle. Cada preset é uma subclasse de `Preset` com uma ficha (entradas em SI com fonte do valor padrão e domínio de validade, saídas), um solver, checagens que alimentam um trust report, uma decisão (OK / NOT OK), referências citáveis para validação e, quando cabe, otimização, inversão, variantes de projeto e cena 3D. Os casos cobrem aerodinâmica (perfil a Re 4e6), eletromagnetismo (correntes parasitas, motores IPM e SPM), térmica (dissipador, PCB, isolamento, casco-tubo), estruturas (vaso de pressão ASME, passarela de aço) e fluidos (golpe de aríete, rede de água, bomba e NPSH).

O uso é `pp.presets.load(nome).run(**params)`, a CLI `python -m pinneapple_presets` ou o nó `preset.run` do Flow Studio. Uma corrida fora do domínio de validade é sinalizada. `surrogate` constrói um processo gaussiano ARD por saída a partir de um DOE com o solver real e mede o erro em corridas retidas; `run(fidelity="surrogate")` usa esse modelo.

`report` gera a pasta de cada preset em `examples/industrial_presets/` (resultados, figuras, README, manifesto do Lab) e `registry.write_index` grava o `pinneapple_presets/index.json` que o Hub lê. Os presets usam solvers e modelos de outros pacotes (por exemplo o substituto do zoo para o perfil, `pinneapple_core.jsonable` no índice, a cena de estúdio em `viz3d`).

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 21 | 5.247 | 22 | 31 | 50 (3 arquivos) | 16 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `__main__` | CLI do pacote. | 1 | 1 | 27 |
| `cases` | Os presets concretos (um módulo por problema de engenharia). | 14 | 20 | 4 |
| `core` | Classes base: `Param`, `Output`, `Reference`, `Result`, `Preset`. | 1 | 10 | 0 |
| `registry` | Registro de presets e índice do Hub. | 1 | 6 | 0 |
| `report` | Geração da documentação e dos resultados de cada preset. | 1 | 4 | 27 |
| `surrogate` | Substituto por processo gaussiano ARD. | 1 | 6 | 46 |
| `viz3d` | Primitivas e imagens 3D dos resultados. | 1 | 6 | 0 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_presets`

Presets industriais: problemas de engenharia já montados, resolvidos e validados com o PINNeAPPle, acessíveis por `pp.presets.names()`/`load()`; reexporta o registro e as classes base.

<small>`pinneapple_presets/__init__.py` · 19 linhas · testes: 27</small>

#### `pinneapple_presets.__main__`

CLI `python -m pinneapple_presets` com os comandos list, card, run (opcionalmente `--surrogate`), validate, surrogate, report e index.

<small>`pinneapple_presets/__main__.py` · 83 linhas · testes: 27</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `main` | função | `(argv=None) -> int` | Interpreta os argumentos da CLI e executa o comando pedido, devolvendo o código de saída. |

#### `pinneapple_presets.cases`

Pacote vazio que agrupa os presets concretos. [sem teste direto]

<small>`pinneapple_presets/cases/__init__.py` · 1 linhas · **sem teste direto**</small>

#### `pinneapple_presets.cases.airfoil`

Perfil de asa/pá a Re 4e6: sustentação, arrasto e momento a partir do substituto verificado do zoo treinado com OpenFOAM RANS, com teoria de perfil fino como checagem.

<small>`pinneapple_presets/cases/airfoil.py` · 290 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AirfoilSection` | classe | `()` | Preset do perfil de asa: modelo do zoo, forma NACA, solução, checagens, decisão, validação, otimização, inversão, variantes, figuras e cena. |
| `PresetCache` | classe | `()` | Cache em atributo de classe do modelo do zoo carregado, para não recarregá-lo a cada corrida. |
| `thin_airfoil_zero_lift` | função | `(code) -> float` | Ângulo de sustentação nula de uma linha de arqueamento NACA 4 dígitos pela teoria de perfil fino (quadratura numérica). |

#### `pinneapple_presets.cases.eddy_current`

Inspeção por correntes parasitas de chapa aeronáutica: impedância da sonda, profundidade de penetração e sensibilidade ao afinamento, com checagem por diferenças finitas. [sem teste direto]

<small>`pinneapple_presets/cases/eddy_current.py` · 308 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EddyCurrentInspection` | classe | `()` | Preset de inspeção por correntes parasitas, com solução analítica e checagem por diferenças finitas. |

#### `pinneapple_presets.cases.footbridge`

Passarela de aço: resistência à multidão, flecha, flambagem lateral-torcional e vibração de pedestres (EN 1990 A2). [sem teste direto]

<small>`pinneapple_presets/cases/footbridge.py` · 375 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SteelFootbridge` | classe | `()` | Preset da passarela de aço com checagens de resistência, flecha, flambagem e conforto. |
| `section_class_ok` | função | `(h, b, tf, tw, fy) -> tuple` | Verifica os limites de classe 3 da EN 1993-1-1 Tabela 5.2 para alma e mesa da seção. |

#### `pinneapple_presets.cases.heatsink`

Dissipador de aletas planas em ar forçado canalizado para módulo de potência: rede de resistências mais base em volumes finitos 3D. [sem teste direto]

<small>`pinneapple_presets/cases/heatsink.py` · 361 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PlateFinHeatsink` | classe | `()` | Preset do dissipador de aletas com rede de resistências e base em volumes finitos 3D. |

#### `pinneapple_presets.cases.insulation`

Isolamento de linha de vapor: perda de calor, temperatura de superfície e espessura econômica (ISO 12241, VDI 2055). [sem teste direto]

<small>`pinneapple_presets/cases/insulation.py` · 279 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SteamLineInsulation` | classe | `()` | Preset do isolamento de linha de vapor com espessura econômica. |

#### `pinneapple_presets.cases.ipm_traction`

Motor de tração de ímãs internos (geometria do Toyota Prius 2004): torque, ondulação, MTPA e tensão em velocidade por elementos finitos. [sem teste direto]

<small>`pinneapple_presets/cases/ipm_traction.py` · 252 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `IPMTraction` | classe | `()` | Preset do motor de tração IPM resolvido por FEA. |
| `machine` | função | `(n_gap) -> IPMMachine` | Monta a máquina `IPMMachine` da geometria de referência com `n_gap` elementos no entreferro. |

#### `pinneapple_presets.cases.pcb`

Temperaturas de junção numa placa de potência (estágio DC-DC/acionamento) por volumes finitos 3D do empilhamento. [sem teste direto]

<small>`pinneapple_presets/cases/pcb.py` · 266 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PowerPCB` | classe | `()` | Preset da placa de potência com temperaturas de junção por volumes finitos 3D. |

#### `pinneapple_presets.cases.pipe_network`

Rede de distribuição de água: pressões, vazões e dimensionamento de menor custo pelo algoritmo do gradiente global. [sem teste direto]

<small>`pinneapple_presets/cases/pipe_network.py` · 289 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `WaterNetwork` | classe | `()` | Preset da rede de água com algoritmo do gradiente global e dimensionamento de menor custo. |

#### `pinneapple_presets.cases.pressure_vessel`

Casco e tampos de vaso de pressão cilíndrico pela ASME VIII Div. 1 (UG-27, UG-32), conferidos por Lamé e FEM 3D. [sem teste direto]

<small>`pinneapple_presets/cases/pressure_vessel.py` · 346 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PressureVessel` | classe | `()` | Preset do vaso de pressão pela ASME VIII com checagem por Lamé e FEM de anel. |
| `ring_mesh` | função | `(ri, ro, h, nr, nt, nz) -> HexMesh` | Malha hexaédrica de 8 nós de um quarto de anel. |
| `lame` | função | `(p, ri, ro, r, E, nu) -> dict` | Tensões de Lamé num raio e deslocamento radial em deformação plana. |

#### `pinneapple_presets.cases.pump_npsh`

Bomba centrífuga aspirando de um poço: ponto de operação, NPSH disponível contra requerido e margem contra cavitação. [sem teste direto]

<small>`pinneapple_presets/cases/pump_npsh.py` · 276 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PumpNPSH` | classe | `()` | Preset da bomba centrífuga com NPSH e margem de cavitação. |

#### `pinneapple_presets.cases.shell_tube`

Resfriador casco-tubo de água em circuito fechado (TEMA E, 1 passe no casco, 2 nos tubos) avaliado por ε-NTU. [sem teste direto]

<small>`pinneapple_presets/cases/shell_tube.py` · 304 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ShellTubeCooler` | classe | `()` | Preset do resfriador casco-tubo avaliado por ε-NTU. |

#### `pinneapple_presets.cases.spm_motor`

Servo motor de ímãs superficiais: torque, ondulação, cogging e força contra-eletromotriz por FEM magnetostático 2D não linear. [sem teste direto]

<small>`pinneapple_presets/cases/spm_motor.py` · 292 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SPMMotor` | classe | `()` | Preset do servo motor de ímãs superficiais por FEM magnetostático 2D. |

#### `pinneapple_presets.cases.water_hammer`

Golpe de aríete no fechamento de válvula no fim de uma adutora de aço, pelo método das características, com teoria de coluna rígida como checagem. [sem teste direto]

<small>`pinneapple_presets/cases/water_hammer.py` · 369 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `WaterHammer` | classe | `()` | Preset do golpe de aríete pelo método das características, com estudo de convergência. |
| `rigid_column_max_head` | função | `(*, length, diameter, friction_factor, q0, h_reservoir, h_downstream, t_close, closure_ex…)` | Carga máxima na válvula para o mesmo fechamento pela teoria de coluna rígida. |

#### `pinneapple_presets.core`

Base dos presets: parâmetros com domínio de validade e fonte, saídas, referências citáveis, especificação de substituto, `Result` com trust report e decisão, e a classe `Preset` (cujo `solve` levanta NotImplementedError até ser implementado). [sem teste direto]

<small>`pinneapple_presets/core.py` · 342 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Param` | classe | `(name, unit, default, source, lo, hi, help, choices)` | Uma entrada do preset: unidade, padrão com fonte, domínio de validade `lo..hi`, ajuda e escolhas. |
| `Output` | classe | `(name, unit, help)` | Uma saída do preset com unidade e ajuda. |
| `Reference` | classe | `(name, params, output, value, unit, source, kind, tol)` | Valor de referência citável de uma saída (analítico, norma, benchmark, experimento ou solver independente) com tolerância. |
| `SurrogateSpec` | classe | `(inputs, outputs, n_train, n_test, bounds, log, log_outputs)` | Quais entradas e saídas o substituto cobre e o tamanho do DOE. |
| `Result` | classe | `(preset, params, outputs, trust, decision, fidelity, warnings, meta, fields)` | Saídas de uma corrida com trust report, decisão, fidelidade, avisos e proveniência. |
| `Preset` | classe | `(name, title, sector, problem, question, params, outputs, validity, pipeline, library, su…)` | Classe base: atributos da ficha e os métodos `solve`, `checks`, `decide`, `references` (e opcionalmente `scene`, `convergence`, `figures`) a implementar. |
| `check` | função | `(name, ok, value=None, limit=None, source='', reason='', critical=False) -> Check` | Cria uma checagem do trust report a partir de um passa/falha (None = não rodou). |
| `environment` | função | `() -> dict` | Versões e hardware da corrida, gravados com cada resultado. |
| `lhs` | função | `(n, bounds, seed=0, log=()) -> np.ndarray` | Amostra por hipercubo latino de `n` pontos numa caixa, com eixos opcionais em escala log. |
| `call_quiet` | função | `(fn, *a, **kw)` | Chama uma função suprimindo avisos. |

#### `pinneapple_presets.registry`

Registro dos presets embutidos (importados sob demanda) e de usuário, e o índice `pinneapple_presets/index.json` que o Hub lê. [sem teste direto]

<small>`pinneapple_presets/registry.py` · 97 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `register` | função | `(cls) -> type` | Decorador que adiciona um preset de usuário com nome único. |
| `names` | função | `() -> list[str]` | Nomes de todos os presets (embutidos e registrados), em ordem. |
| `load` | função | `(name) -> Preset` | Instancia o preset pelo nome, sugerindo nomes próximos se não existir. |
| `list_presets` | função | `(sector=None) -> list[dict]` | Nome, título, setor e capacidades de cada preset, com filtro de setor. |
| `index` | função | `() -> dict` | Tudo que o Hub mostra: ficha, resultados medidos e caminhos de figuras de cada preset. |
| `write_index` | função | `(path=INDEX_PATH) -> str` | Grava o índice JSON do Hub. |

#### `pinneapple_presets.report`

Resolve, valida e documenta um preset em `examples/industrial_presets/<name>/`: resultados, figuras, imagens 3D, substituto, inversão, otimização, variantes, README e manifesto do Lab.

<small>`pinneapple_presets/report.py` · 322 linhas · testes: 27</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `write_report` | função | `(preset, out_dir=None, *, surrogate='load', three_d=True, lab=True) -> dict` | Gera a pasta de documentação de um preset, com substituto construído, carregado ou pulado. |
| `compare_variants` | função | `(p, variants, res, root, three_d=True) -> dict` | Roda todas as variantes de projeto e gera tabela, gráfico de barras e galeria 3D. |
| `readme` | função | `(p, s) -> str` | Texto do README de um preset (ficha + resultados medidos). |
| `write_catalogue` | função | `(root=None) -> str` | Gera `examples/industrial_presets/README.md` com uma linha por preset a partir dos resumos gravados. |

#### `pinneapple_presets.surrogate`

Substituto de um preset: DOE por hipercubo latino rodado com o solver real, um processo gaussiano ARD por saída, erro medido em corridas retidas e a caixa de treino como domínio de validade.

<small>`pinneapple_presets/surrogate.py` · 232 linhas · testes: 46</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ARDGP` | classe | `()` | Processo gaussiano com kernel exponencial quadrático e um comprimento de escala por entrada, ajustado por máxima verossimilhança marginal. |
| `PresetSurrogate` | classe | `(meta, models, test=None)` | Substituto treinado de um preset: um GP ARD por saída, com predição e detecção de fora da caixa. |
| `surrogate_path` | função | `(name) -> str` | Caminho onde o substituto de um preset é gravado. |
| `load_surrogate` | função | `(name) -> PresetSurrogate \| None` | Carrega o substituto gravado de um preset, ou None. |
| `build_surrogate` | função | `(preset, *, seed=0, save=True, tol=0.05, n_train=None, n_test=None) -> PresetSurrogate` | Roda o DOE com o solver real, ajusta, mede o erro retido e grava o modelo. |
| `serve_spec` | função | `(name) -> dict \| None` | O que um servidor de modelos precisa para expor o substituto: entradas com limites e unidades, saídas, erro e chamada Python. |

#### `pinneapple_presets.viz3d`

Imagens 3D de resultados de presets na cena de estúdio da biblioteca: primitivas de malha (caixa, tubo, extrusão) e gravação de imagens e visualizador HTML. [sem teste direto]

<small>`pinneapple_presets/viz3d.py` · 144 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `box` | função | `(lo, hi, n=(1, 1, 1))` | Caixa fechada alinhada aos eixos como malha (V, F). |
| `tube` | função | `(path, radius, n_around=16)` | Tubo aberto ao longo de uma polilinha, com o índice do ponto de cada vértice. |
| `extrude` | função | `(section, x)` | Prisma de um polígono 2D extrudado ao longo de estações. |
| `merge` | função | `(parts)` | Concatena várias malhas numa só. |
| `stills` | função | `(scene, out_dir, name, modes=(), size=(1100, 700), views=None, frame=None) -> dict[str, s…)` | Grava o visualizador HTML autocontido e imagens estáticas de uma cena. |
| `field_on` | função | `(idx, values) -> np.ndarray` | Valores por vértice a partir de valores por estação. |

### Exemplos de uso reais do repositório

**Uso de um preset (docstring do pacote)** — `pinneapple_presets/__init__.py`

```python
import pinneapple as pp
pp.presets.names()                                   # catalogue
p = pp.presets.load("water_hammer_valve_closure")
p.card()                                             # data sheet: inputs (SI, defaults with sources), outputs
r = p.run(t_close=4.0)                               # solver; fidelity="surrogate" for the trained surrogate
print(r)                                             # outputs, decision, trust report
p.validate()                                         # reference cases with the measured error
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 5 de 21.
- Funções de teste em arquivos que importam algum módulo do pacote: 50.

### Limitações conhecidas

- `Preset.solve` levanta NotImplementedError na classe base; todo preset novo precisa implementar `solve`, `checks`, `decide` e `references`.
- O domínio `lo..hi` de cada parâmetro é o domínio de validade do solver e das correlações; corridas fora dele são sinalizadas, não extrapoladas com garantia.
- O substituto só vale dentro da caixa de treino e tem erro retido medido por preset; no catálogo gravado, o golpe de aríete aparece com 18 % de erro hold-out máximo, e vários presets não têm substituto porque a forma fechada já é instantânea.
- O perfil de asa usa um substituto do zoo de OpenFOAM RANS como solver; o catálogo registra que o OpenFOAM não estava instalado para conferir os ótimos.
- Os módulos de casos, exceto `airfoil`, e `core`, `registry` e `viz3d` não têm teste que os importe diretamente; são exercitados pelos testes da CLI e de `report`/`surrogate`.

## 3. `pinneapple_problemdesign`

### Objetivo

`pinneapple_problemdesign` ajuda a transformar uma descrição de problema em linguagem natural numa especificação estruturada e num plano de solução com o PINNeAPPle. O núcleo é o `DesignAgent`, que conversa com um LLM por estágios, preenche um `ProblemSpec` (dados, física, geometria, validação, implantação, restrições) e aplica uma política de não invenção: o que o usuário não informou vira `Gap` (pergunta) ou suposição marcada, nunca um valor inventado. O resultado é um `DesignReport` (Markdown ou JSON) com um plano FNO-first, PINN-first ou CFD-first.

A partir do spec, `codegen.build_pinneapple_spec` gera um `PinneappleSpec` com tipo de EDP, campos, modelo e código de pipeline; `method_selection` sugere método numérico e fechamento de turbulência com base no regime de escoamento (usando `pinneapple_analysis` e o registro de solvers de `pinneapple_simulation`); `knowledge.physics_knowledge` liga fenômenos a equações e presets de `pinneapple_physics`; e `preset_authoring` deixa um LLM propor uma EDP nova, aceitando-a só se passar numa checagem por solução manufaturada.

A parte agêntica compõe peças de outros pacotes sem alterá-las: `UnifiedPhysicsAgent` liga a elicitação ao `PhysicsOrchestrator` de `pinneapple_worldmodel`, e `AutonomousResearchAgent` roda iterações com memória de experimentos (`pinneapple_registry`) e validação por guardrail (`pinneapple_llm`). Tudo isso passa pela `ApprovalGate` de `autonomy`, que exige aprovação humana para ações consequentes (custo, ferramenta licenciada, efeitos fora do sandbox) em qualquer nível de autonomia. Os provedores de LLM incluídos são Gemini e um gateway HTTP (Cosmos); qualquer objeto com `generate` serve.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 28 | 5.642 | 40 | 21 | 123 (9 arquivos) | 19 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `agent` | Agente de elicitação `DesignAgent`. | 1 | 1 | 0 |
| `autonomy` | Trava de aprovação humana para ações consequentes. | 1 | 5 | 41 |
| `cli` | CLI interativa do agente com Gemini. | 1 | 1 | 0 |
| `codegen` | Gera `PinneappleSpec` e código a partir do `ProblemSpec`. | 1 | 1 | 0 |
| `elicitation` | Estágios, perguntas de lacuna e validadores da elicitação. | 4 | 3 | 0 |
| `extractor` | Extração JSON de requisitos via LLM. | 1 | 1 | 0 |
| `knowledge` | Construtores de plano, base de fenômenos físicos e mapa de capacidades. | 4 | 7 | 22 |
| `merge` | Mescla de dados extraídos no spec. | 1 | 1 | 0 |
| `method_selection` | Recomendação de método numérico e turbulência pelo regime. | 1 | 3 | 11 |
| `policy` | Política de não invenção. | 1 | 1 | 0 |
| `preset_authoring` | Rascunho de preset por LLM verificado por solução manufaturada. | 1 | 5 | 15 |
| `protocol` | Protocolo de LLM e provedor Gemini. | 1 | 4 | 20 |
| `providers` | Provedores de LLM adicionais (Cosmos). | 2 | 3 | 0 |
| `renderers` | Relatórios em Markdown e JSON. | 3 | 2 | 0 |
| `research_loop` | Agente de pesquisa autônomo sob a trava de aprovação. | 1 | 3 | 9 |
| `schema` | Dataclasses do problema, plano e relatório. | 1 | 16 | 70 |
| `state` | Estado da conversa de design. | 1 | 2 | 0 |
| `unified_agent` | Elicitação + `PhysicsOrchestrator` num laço único. | 1 | 2 | 0 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_problemdesign`

Reexporta o agente de design, o esquema do problema, o estado, o protocolo de LLM, o gerador de código e a trava de autonomia; `UnifiedPhysicsAgent` e `AutonomousResearchAgent` só são exportados se as dependências opcionais (`pinneapple_worldmodel`, `pinneapple_registry`) importarem.

<small>`pinneapple_problemdesign/__init__.py` · 62 linhas · testes: 58</small>

#### `pinneapple_problemdesign.agent`

`DesignAgent`: elicitação interativa do problema via LLM, com extração estruturada em JSON, mescla segura no `ProblemSpec` e política de não invenção (o que é desconhecido vira `Gap`). [sem teste direto]

<small>`pinneapple_problemdesign/agent.py` · 194 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DesignAgent` | classe | `(llm, use_orchestrator_bridge=True)` | Agente que conduz a elicitação por estágios: ingere mensagens, extrai campos, gera perguntas de lacuna e, ao fim, o relatório. |

#### `pinneapple_problemdesign.autonomy`

Trava de segurança do laço de pesquisa autônomo: classifica ações consequentes (custo, ferramenta licenciada, efeito fora do sandbox) e exige aprovação humana nelas em qualquer nível de autonomia, falhando fechado.

<small>`pinneapple_problemdesign/autonomy.py` · 481 linhas · testes: 41</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AutonomyLevel` | classe | `()` | Nível de autonomia do laço (`SUPERVISED`, `SEMI_AUTONOMOUS`, `FULL_AUTONOMOUS`). |
| `ConsequentialActionClassifier` | classe | `(*, name_patterns=None, category_patterns=None, tag_patterns=None, exceptions=None) -> No…)` | Classificador explícito (por padrões de nome, categoria e tag, sem ML) que decide se uma chamada de ferramenta é consequente. |
| `ApprovalRequest` | classe | `(tool_name, tool_args, reason, autonomy_level, is_consequential)` | Tudo que o aprovador humano precisa para decidir uma ação travada. |
| `ApprovalDecision` | classe | `(approved, reason, requires_human)` | Resultado de `ApprovalGate.check`: aprovado, motivo e se exige humano. |
| `ApprovalGate` | classe | `(autonomy_level, approver_fn=None, iteration_report_every=5, confidence_pause_threshold=N…)` | A trava: decide por chamada de ferramenta se pode prosseguir, pede aprovação quando necessário e falha fechado sem aprovador. |

#### `pinneapple_problemdesign.cli`

CLI interativa mínima do agente de design usando `GeminiProvider`. [sem teste direto]

<small>`pinneapple_problemdesign/cli.py` · 39 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `main` | função | `()` | Laço interativo no terminal com o `DesignAgent`. |

#### `pinneapple_problemdesign.codegen`

Traduz um `ProblemSpec` elicitado num `PinneappleSpec` concreto (tipo de EDP, coordenadas, campos, modelo, treino) com código de pipeline gerado; o código gerado contém TODOs para valores e DataLoader não resolvidos. [sem teste direto]

<small>`pinneapple_problemdesign/codegen.py` · 374 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_pinneapple_spec` | função | `(design_spec) -> PinneappleSpec` | Traduz um `ProblemSpec` num `PinneappleSpec` com objetos e código do PINNeAPPle. |

#### `pinneapple_problemdesign.elicitation`

Subpacote de elicitação: reexporta `STAGES_ORDER`, `Stage`, `build_stage_gaps` e `validate_and_suggest`. [sem teste direto]

<small>`pinneapple_problemdesign/elicitation/__init__.py` · 6 linhas · **sem teste direto**</small>

#### `pinneapple_problemdesign.elicitation.questions`

Perguntas de lacuna (`Gap`) específicas de cada estágio da elicitação. [sem teste direto]

<small>`pinneapple_problemdesign/elicitation/questions.py` · 101 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_stage_gaps` | função | `(stage, spec_dict) -> List[Gap]` | Lista as lacunas abertas de um estágio dado o spec atual. |

#### `pinneapple_problemdesign.elicitation.stages`

Definição dos estágios da elicitação (intake, entradas/saídas e tempo, dados, ...) e da ordem `STAGES_ORDER`. [sem teste direto]

<small>`pinneapple_problemdesign/elicitation/stages.py` · 76 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Stage` | classe | `(name, description, required_fields)` | Um estágio de elicitação: nome, descrição e campos obrigatórios. |

#### `pinneapple_problemdesign.elicitation.validators`

Validação de um `ProblemSpec`: avisos e suposições sugeridas. [sem teste direto]

<small>`pinneapple_problemdesign/elicitation/validators.py` · 42 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `validate_and_suggest` | função | `(spec) -> Tuple[List[str], List[Assumption]]` | Devolve avisos e suposições sugeridas para um `ProblemSpec`. |

#### `pinneapple_problemdesign.extractor`

Extração em JSON de requisitos a partir do texto do usuário via LLM. [sem teste direto]

<small>`pinneapple_problemdesign/extractor.py` · 140 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `extract_from_text` | função | `(llm, user_text) -> Tuple[Dict[str, Any], Dict[str, Any]]` | Pede ao LLM um JSON estruturado a partir do texto do usuário e devolve o dicionário extraído e metadados. |

#### `pinneapple_problemdesign.knowledge`

Subpacote de conhecimento: reexporta `PINNEAPPLE_CAPABILITIES` e os construtores de plano. [sem teste direto]

<small>`pinneapple_problemdesign/knowledge/__init__.py` · 10 linhas · **sem teste direto**</small>

#### `pinneapple_problemdesign.knowledge.mapping`

Construtor de plano: escolhe entre plano FNO-first, PINN-first ou CFD-first pelo `task_type`, e lista (só leitura) ferramentas registradas no `PhysicsOrchestrator` do `pinneapple_worldmodel` quando disponível.

<small>`pinneapple_problemdesign/knowledge/mapping.py` · 483 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `available_orchestrator_tools` | função | `(spec) -> List[Dict[str, object]]` | Consulta só de leitura das ferramentas registradas no `PhysicsOrchestrator`. |
| `build_plan` | função | `(spec, gaps, use_orchestrator_bridge=True) -> Plan` | Escolhe o construtor de plano pelo `task_type` do spec. |
| `build_plan_pinn_first` | função | `(spec, gaps) -> Plan` | Plano para solução de EDP e problema inverso via `PINNFactory`. |
| `build_plan_fno_first` | função | `(spec, gaps) -> Plan` | Plano FNO-first para previsão e aprendizado de operadores. |
| `build_plan_cfd_first` | função | `(spec, gaps) -> Plan` | Plano para aerodinâmica externa/CFD com o solver Lattice-Boltzmann, presets RANS/LES e pós-processamento de vórtices. |

#### `pinneapple_problemdesign.knowledge.physics_knowledge`

Base de conhecimento fenômeno -> equação governante -> parâmetros típicos -> suposições, cada entrada ligada a um preset real.

<small>`pinneapple_problemdesign/knowledge/physics_knowledge.py` · 615 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhenomenonEntry` | classe | `(phenomenon, governing_equation, typical_parameters, assumptions, preset_module, preset_f…)` | Um registro de conhecimento físico rastreável até um preset real. |
| `lookup_phenomenon` | função | `(name_or_keyword) -> List[PhenomenonEntry]` | Busca entradas da base por palavra-chave, sem diferenciar maiúsculas. |

#### `pinneapple_problemdesign.knowledge.pinneapple_capabilities`

Dicionário `PINNEAPPLE_CAPABILITIES` com os módulos do PINNeAPPle por área (treino, modelos, séries temporais, CFD...) usado no texto dos planos. [sem teste direto]

<small>`pinneapple_problemdesign/knowledge/pinneapple_capabilities.py` · 41 linhas · **sem teste direto**</small>

#### `pinneapple_problemdesign.merge`

Mescla profunda de um dicionário extraído dentro de um `ProblemSpec` existente. [sem teste direto]

<small>`pinneapple_problemdesign/merge.py` · 33 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `merge_into_spec` | função | `(spec, partial) -> None` | Mescla um dicionário parcial no `ProblemSpec`, no lugar. |

#### `pinneapple_problemdesign.method_selection`

Recomenda método numérico e fechamento de turbulência a partir do regime de escoamento calculado, com uma tabela de decisão explícita e heurística (o docstring avisa que não é sistema especialista validado); `recommend_method_from_spec` hoje sempre devolve None porque o `ProblemSpec` não guarda os números necessários.

<small>`pinneapple_problemdesign/method_selection.py` · 346 linhas · testes: 11 · palavras-chave no código (heurística; o texto acima diz o status real): placeholder/stub citado no docstring</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MethodRecommendation` | classe | `(numerical_method, turbulence_model, flow_regime, rationale, confidence)` | Par recomendado (método numérico, modelo de turbulência) com regime, justificativa e confiança. |
| `recommend_method` | função | `(velocity, length_scale, kinematic_viscosity, *, geometry='internal_pipe', compressible=F…)` | Recomenda solver e fechamento de turbulência a partir de velocidade, escala e viscosidade. |
| `recommend_method_from_spec` | função | `(spec) -> Optional[MethodRecommendation]` | Extrai os parâmetros do `ProblemSpec` e chama `recommend_method`, ou devolve None se faltarem. |

#### `pinneapple_problemdesign.policy`

Política de não invenção: aplica suposições e lacunas sugeridas ao estado do design. [sem teste direto]

<small>`pinneapple_problemdesign/policy.py` · 67 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `apply_non_invention_policy` | função | `(state, *, unknown_fields, assumptions_suggested, gaps_suggested) -> None` | Registra campos desconhecidos como lacunas e suposições no estado, sem inventar valores. |

#### `pinneapple_problemdesign.preset_authoring`

Rascunho de preset assistido por LLM: o LLM propõe uma EDP, ela é convertida em sympy e só é marcada verificada se passar num teste de solução manufaturada automático; não produz presets confiáveis para produção.

<small>`pinneapple_problemdesign/preset_authoring.py` · 707 linhas · testes: 15</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PresetDraftError` | classe | `(stage, message, *, raw='') -> None` | Erro quando a proposta do LLM não vira uma EDP real e checável. |
| `ManufacturedSolutionCheck` | classe | `(exact_solution_used, residual_at_exact_solution, tolerance, passed, n_points, notes)` | Resultado de uma checagem por solução manufaturada: solução usada, resíduo, tolerância e se passou. |
| `DraftedPreset` | classe | `(phenomenon_description, proposed_equation_latex_or_sympy, sympy_pde, proposed_bcs, propo…)` | Preset proposto pelo LLM com a EDP em sympy, condições, resultado da verificação e citações. |
| `run_manufactured_solution_check` | função | `(sympy_pde, coord_syms, field_syms, *, param_syms=None, params=None, tolerance=0.0001, n_…)` | Roda uma checagem de autoconsistência por solução manufaturada sobre uma EDP sympy. |
| `draft_preset` | função | `(phenomenon_description, *, llm_provider='anthropic', model=None, use_knowledge_base=True…)` | Pede a um LLM (padrão `anthropic`) uma EDP para um fenômeno e a verifica por solução manufaturada. |

#### `pinneapple_problemdesign.protocol`

Protocolo `LLMProvider`, mensagens/respostas e o `GeminiProvider` (SDK Google Generative AI).

<small>`pinneapple_problemdesign/protocol.py` · 100 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LLMMessage` | classe | `(role, content)` | Mensagem de LLM com papel e conteúdo. |
| `LLMResponse` | classe | `(text, raw)` | Resposta de LLM com texto e objeto bruto. |
| `LLMProvider` | classe | `()` | Interface única de provedor de LLM (`generate`). |
| `GeminiProvider` | classe | `(model='gemini-1.5-pro')` | Provedor Gemini via SDK Google Generative AI. |

#### `pinneapple_problemdesign.providers`

Define outra interface `LLMProvider` (Protocol) para os provedores do pacote. [sem teste direto]

<small>`pinneapple_problemdesign/providers/__init__.py` · 47 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LLMProvider` | classe | `()` | Interface de provedor (Protocol) usada pelo pacote, duplicada em relação a `protocol.LLMProvider`. |

#### `pinneapple_problemdesign.providers.cosmos_provider`

Provedor LLM que chama um gateway HTTP (Cosmos) via `requests`. [sem teste direto]

<small>`pinneapple_problemdesign/providers/cosmos_provider.py` · 187 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CosmosProviderConfig` | classe | `(base_url, api_key, timeout_s, route)` | Configuração do provedor Cosmos: URL, chave, timeout e rota. |
| `CosmosProvider` | classe | `(base_url=None, *, api_key=None, timeout_s=None, route=None, session=None)` | Provedor LLM que chama um gateway HTTP Cosmos. |

#### `pinneapple_problemdesign.renderers`

Subpacote de renderizadores do `DesignReport` (Markdown e JSON). [sem teste direto]

<small>`pinneapple_problemdesign/renderers/__init__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_problemdesign.renderers.report_json`

Renderiza um `DesignReport` em JSON. [sem teste direto]

<small>`pinneapple_problemdesign/renderers/report_json.py` · 35 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `render_json_report` | função | `(report, *, indent=2) -> str` | Serializa o `DesignReport` em JSON. |

#### `pinneapple_problemdesign.renderers.report_md`

Renderiza um `DesignReport` em Markdown com avisos. [sem teste direto]

<small>`pinneapple_problemdesign/renderers/report_md.py` · 133 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `render_markdown_report` | função | `(report, warnings) -> str` | Gera o relatório Markdown do design com os avisos. |

#### `pinneapple_problemdesign.research_loop`

`AutonomousResearchAgent`: ciclo de pesquisa autônomo sobre o `UnifiedPhysicsAgent`, com `ApprovalGate` em cada iteração e ferramenta, memória de experimentos (`pinneapple_registry`) e validação por confiança física e guardrail.

<small>`pinneapple_problemdesign/research_loop.py` · 423 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `IterationRecord` | classe | `(iteration, problem_description, approval, agent_result, guardrail_report, confidence, no…)` | O que aconteceu numa iteração do laço (aprovação, resultado, guardrail, confiança). |
| `ResearchLoopResult` | classe | `(status, iterations, final_confidence, final_result, memory_record_ids, prior_lessons)` | Resultado completo de `AutonomousResearchAgent.run`: status, iterações, confiança final e memória. |
| `AutonomousResearchAgent` | classe | `(llm, *, autonomy_level=AutonomyLevel.SUPERVISED, approver_fn=None, memory=None, physics_…)` | Agente de pesquisa autônomo que itera até a confiança alvo ou o limite de iterações, sob a trava de aprovação. |

#### `pinneapple_problemdesign.schema`

Esquema do design de problema: `ProblemSpec` e suas partes (dados, física, geometria, validação, implantação, restrições), `Gap`, `Plan`, `PinneappleSpec` e `DesignReport`.

<small>`pinneapple_problemdesign/schema.py` · 290 linhas · testes: 70</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Assumption` | classe | `(text, confidence, needs_confirmation)` | Suposição com confiança e se precisa de confirmação. |
| `Risk` | classe | `(text, level, mitigation)` | Risco com nível e mitigação. |
| `DataSpec` | classe | `(sources, format, sampling, variables_observed, target_variables, known_quality_issues, m…)` | Especificação dos dados: fontes, formato, amostragem, variáveis, qualidade e divisão. |
| `PhysicsSpec` | classe | `(governing_equations, boundary_conditions, initial_conditions, constraints, parameters_kn…)` | Especificação física: equações, condições, restrições, parâmetros, unidades, turbulência e método numérico. |
| `GeometrySpec` | classe | `(domain, representation, sensors, coordinate_system, cad_format, voxel_resolution, aoa_sw…)` | Especificação da geometria: domínio, representação, sensores, CAD e varredura de ângulo de ataque. |
| `ValidationSpec` | classe | `(primary_metrics, acceptance_criteria, robustness_tests, ood_scenarios)` | Especificação de validação: métricas, critérios de aceitação, robustez e cenários fora da distribuição. |
| `DeploymentSpec` | classe | `(environment, latency_budget_ms, update_policy, monitoring)` | Especificação de implantação: ambiente, latência, atualização e monitoramento. |
| `ConstraintsSpec` | classe | `(hardware, max_training_time, interpretability, compliance)` | Restrições: hardware, tempo de treino, interpretabilidade e conformidade. |
| `ProblemSpec` | classe | `(title, goal, task_type, inputs, outputs, horizon, input_window, frequency, domain_contex…)` | Especificação completa do problema (objetivo, tipo de tarefa, entradas/saídas e todas as partes acima). |
| `Gap` | classe | `(id, question, severity, rationale, how_to_obtain, resolved)` | Lacuna de informação com pergunta, severidade, justificativa e como obter. |
| `PlanStep` | classe | `(title, why, actions, pinneapple_modules, exit_criteria)` | Passo de plano: título, motivo, ações, módulos PINNeAPPle e critério de saída. |
| `Plan` | classe | `(recommended_approach, alternatives, steps, go_no_go)` | Plano: abordagem recomendada, alternativas, passos e critérios go/no-go. |
| `PinneappleSpec` | classe | `(pde_kind, pde_confidence, coords, fields, pde_params, environment_kwargs, model_name, mo…)` | Objetos concretos da API do PINNeAPPle derivados do spec (EDP, campos, modelo, treino, código). |
| `DesignReport` | classe | `(spec, gaps, plan, pinneapple_spec, created_at)` | Relatório final do design: spec, lacunas, plano e `PinneappleSpec`. |
| `uses_pinn_approach` | função | `(spec) -> bool` | Diz se o spec deve ser resolvido com um pipeline PINN de resíduo de EDP. |
| `uses_cfd_approach` | função | `(spec) -> bool` | Diz se o spec é um problema de aerodinâmica externa/CFD que recebe o plano CFD-first. |

#### `pinneapple_problemdesign.state`

Estado da conversa de design (`DesignState`, `Turn`). [sem teste direto]

<small>`pinneapple_problemdesign/state.py` · 33 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Turn` | classe | `(role, content)` | Um turno da conversa (papel, conteúdo). |
| `DesignState` | classe | `(spec, gaps, plan, history, stage, done)` | Estado do design: spec, lacunas, plano, histórico, estágio e se terminou. |

#### `pinneapple_problemdesign.unified_agent`

Compõe `DesignAgent` (elicitação) e `PhysicsOrchestrator` do `pinneapple_worldmodel` (execução de ferramentas) num laço único, sem mexer nos internos de nenhum dos dois. [sem teste direto]

<small>`pinneapple_problemdesign/unified_agent.py` · 572 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `UnifiedAgentResult` | classe | `(status, design_report, problem_statement, execution_result, design_state, gaps, question…)` | Resultado agrupado de `UnifiedPhysicsAgent.run`. |
| `UnifiedPhysicsAgent` | classe | `(llm, *, use_orchestrator_bridge=True, orchestrator=None, orchestrator_verbose=True) -> N…)` | Compõe um `DesignAgent` com um `PhysicsOrchestrator`: elicita o problema, monta o `ProblemStatement` e executa as ferramentas. |

### Exemplos de uso reais do repositório

**Relatório de design sem LLM** — `examples/problem_designer/03_offline_spec_to_report.py`

```python
    # 2) Run validators (warnings + safe assumptions)
    warnings, assumptions = validate_and_suggest(spec)
    spec.assumptions.extend(assumptions)

    # 3) Build plan and report
    gaps = []  # if you already know gaps, add them here; otherwise keep empty.
    plan = build_plan_fno_first(spec, gaps)
    report = DesignReport(spec=spec, gaps=gaps, plan=plan)

    md = render_markdown_report(report, warnings)
    js = render_json_report(report)
```

**CLI interativa do agente** — `pinneapple_problemdesign/cli.py`

```python
def main():
    llm = GeminiProvider()
    agent = DesignAgent(llm=llm)
    state = agent.start()

    print("Pinneapple Design CLI (type 'exit' to quit)\n")

    while True:
        user = input("You: ").strip()
        if user.lower() in ("exit", "quit"):
            break

        agent.ingest_user_message(state, user)
        out = agent.step(state)
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 9 de 28.
- Funções de teste em arquivos que importam algum módulo do pacote: 123.

### Limitações conhecidas

- O agente depende de um LLM: `GeminiProvider` precisa do SDK Google Generative AI, `CosmosProvider` de um gateway HTTP com `requests`, `draft_preset` usa `anthropic` por padrão.
- `UnifiedPhysicsAgent` e `AutonomousResearchAgent` dependem de `pinneapple_worldmodel` (e `pinneapple_registry`); se não importarem, o `__init__` simplesmente não os exporta.
- `method_selection` é uma heurística explícita, não um sistema especialista validado (docstring); `recommend_method_from_spec` devolve None para todo spec atual porque o `ProblemSpec` não tem campos numéricos de velocidade, escala e viscosidade.
- `preset_authoring` só rascunha presets: o docstring diz que não produz nada confiável para produção, e descoberta automática de física está fora de escopo.
- O código gerado por `codegen` contém TODOs (valores físicos não resolvidos, DataLoader, tensor de teste) que o usuário precisa preencher.
- Há duas interfaces `LLMProvider` (em `protocol` e em `providers`).
- Muitos módulos não têm teste direto: `agent`, `codegen`, `unified_agent`, `elicitation.*`, `extractor`, `policy`, `merge`, `providers.*`, `renderers.*`, `state`, `cli`.

## 4. `pinneapple_science`

### Objetivo

O `pinneapple_science` (exposto como `pp.science`) organiza experimentos por área da ciência, em nível acadêmico (didático, conferido contra valor de livro-texto) e de pesquisa (configurável, com estudo de convergência e incerteza). Cada experimento declara área, nível, tipo (`theory`, uma derivação numérica conferida contra forma fechada ou valor publicado, ou `practice`, uma bancada virtual com instrumentos ruidosos e ajuste com incerteza) e as dimensões em que roda: 0-D (concentrado), 1-D, 2-D, 3-D e 3-D + tempo.

Todo resultado (`core.Result`) carrega comparações com referências citadas, checagens físicas e um relatório de confiança (`pinneapple_analysis.trust.TrustReport`) com decisão final. O pacote também traz o atlas das áreas (`atlas`), um arquivo de dados versionado (`data/atlas.json`) que marca cada subárea como disponível, parcial ou planejada e aponta para o código que já a cobre em outros pacotes: um experimento de `pp.science`, um módulo da biblioteca, um preset de problema ou um experimento do Lab. O atlas aponta para o código existente e não o copia.

Hoje os experimentos implementados estão no subpacote `em` (eletromagnetismo), construído sobre `pp.em` (`pinneapple_simulation.electromagnetics`) e `pp.maxwell` (`pinneapple_physics.maxwell_checks`): eletrostática e magnetostática, indução e circuitos, ondas e óptica, bancadas virtuais e estudos de pesquisa (FDTD 3-D configurável, incerteza por Monte Carlo, problema inverso e surrogate). Blocos numéricos comuns ficam em `numerics`, instrumentos virtuais e ajustes em `instruments`, e convenções de figura em `plots`.

O módulo `report` liga o pacote ao `pinneapple_lab`: roda cada experimento em todas as dimensões, grava os resultados em `examples/science/`, escreve o manifesto `experiment.json`, registra o item no Lab e carimba as figuras com o selo da curadoria. A linha de comando é `python -m pinneapple_science atlas | list | run | report | table`.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 16 | 4.253 | 33 | 49 | 19 (1 arquivos) | 10 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `__main__` | Linha de comando `python -m pinneapple_science`: atlas, lista, execução, relatório e tabela. | 1 | 1 | 0 |
| `atlas` | Atlas das áreas da ciência: status por subárea e ponteiros para o código que as cobre, com validação. | 1 | 4 | 0 |
| `core` | Forma comum de um experimento científico: `Experiment`, `Result`, `Reference` e `Comparison`. | 1 | 7 | 19 |
| `em` | Experimentos de eletromagnetismo: estática, indução, ondas, bancadas e estudos de pesquisa. | 7 | 34 | 19 |
| `instruments` | Instrumentos virtuais de laboratório com ruído, erro de calibração e resolução, e os ajustes com incerteza GUM. | 1 | 10 | 19 |
| `numerics` | Blocos numéricos: malhas tensoriais graduadas, solver de volumes finitos para potencial, energia de campo e autovalores de Helmholtz. | 1 | 8 | 19 |
| `plots` | Convenções de figura do pacote (paleta, mapas de cor, eixos discretos). | 1 | 7 | 0 |
| `registry` | Registro dos experimentos por nome, com filtros e execução. | 1 | 5 | 0 |
| `report` | Resultados versionados em `examples/science/`, manifestos e selos do Lab, e o catálogo com o atlas. | 1 | 6 | 0 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_science`

Ponto de entrada (`pp.science`): reexporta `atlas`, `experiments`, `run` e os tipos centrais.

<small>`pinneapple_science/__init__.py` · 25 linhas · testes: 19</small>

#### `pinneapple_science.__main__`

Linha de comando `python -m pinneapple_science` com os subcomandos `atlas`, `list`, `run`, `report` e `table`. [sem teste direto]

<small>`pinneapple_science/__main__.py` · 73 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `main` | função | `(argv=None) -> int` | Monta o `argparse` dos subcomandos e despacha para atlas, registro e relatório. |

#### `pinneapple_science.atlas`

Atlas das áreas: carrega `data/atlas.json` (esquema `pinneapple.science.atlas/1`) e valida que cada ponteiro aponta para código, preset ou experimento existente. [sem teste direto; coberto por `tests/test_science.py` via o pacote]

<small>`pinneapple_science/atlas.py` · 273 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Area` | classe | `(id, title, branch, status, description, subareas)` | Uma área do atlas: id, título, ramo, status, descrição e subáreas com seus ponteiros. |
| `Atlas` | classe | `(schema, version, statuses, dims, areas)` | O atlas inteiro: busca de área, linhas de tabela, contagens por status e saída em Markdown ou HTML. |
| `load` | função | `(path=ATLAS_PATH) -> Atlas` | Carrega o atlas a partir do arquivo de dados versionado. |
| `validate` | função | `(atlas=None, *, check_presets=True, check_lab=True) -> list[str]` | Lista inconsistências do atlas: status desconhecido, módulos inexistentes, presets ou experimentos não registrados. |

#### `pinneapple_science.core`

Forma comum de um experimento: o que ele declara (área, nível, tipo, dimensões) e o que toda execução devolve (comparações, checagens, figuras e decisão de confiança). `Experiment._run` da base levanta `NotImplementedError`.

<small>`pinneapple_science/core.py` · 348 linhas · testes: 19 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Reference` | classe | `(value, source, kind, unit)` | Valor de referência e sua origem (forma fechada, valor publicado, CODATA, limite assintótico, conservação...). |
| `Comparison` | classe | `(quantity, computed, reference, rel_error, tol, passed, kind, source, unit, how, abs_erro…)` | Comparação de uma grandeza calculada com a referência: erros, tolerância, se passou e fonte. |
| `Result` | classe | `(experiment, title, area, topic, level, kind, dim, params, values, comparisons, checks, f…)` | O que uma execução devolve: valores, comparações, checagens, figuras, filmes, visualizador, dados e relatório de confiança; salva em pasta. |
| `Experiment` | classe | `(name, title, area, topic, level, kind, dims, description, references, params, quick)` | Classe base: subclasses declaram os atributos e implementam `_run(res, dim, p)`; `run` e `run_all` executam e montam a confiança. |
| `jsonable` | função | `(v) -> Any` | Converte para tipos JSON simples (numpy para números e listas; NaN e inf para None). |
| `observed_order` | função | `(errors, ratio=2.0) -> list[float]` | Ordens de convergência observadas de uma escada de refinamento. |
| `central_derivative` | função | `(f, x, h) -> float` | Derivada por diferença central de uma função escalar. |

#### `pinneapple_science.em`

Subpacote de experimentos de eletromagnetismo, construído sobre `pp.em` e `pp.maxwell`. [sem teste direto]

<small>`pinneapple_science/em/__init__.py` · 3 linhas · **sem teste direto**</small>

#### `pinneapple_science.em.bench`

Bancadas virtuais (prática): medir μ0 com um solenoide e uma sonda Hall, e uma capacitância pela descarga RC no osciloscópio, cada uma com orçamento de incerteza GUM. [sem teste direto]

<small>`pinneapple_science/em/bench.py` · 220 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Mu0Solenoid` | classe | `()` | Experimento `em.bench_mu0_solenoid`: mede μ0 a partir de B(I) no centro de um solenoide com instrumentos virtuais. |
| `RCDischarge` | classe | `()` | Experimento `em.bench_rc_discharge`: mede uma capacitância pela descarga de um circuito RC num osciloscópio virtual. |

#### `pinneapple_science.em.fdtd_study`

Estudo FDTD 3-D + tempo configurável por dicionário (fontes, materiais, escada de refinamento), com potência irradiada por transformada campo próximo-campo distante, snapshots e checagens de Maxwell.

<small>`pinneapple_science/em/fdtd_study.py` · 199 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FDTDConfig` | classe | `(f0, size, cells_per_wavelength, npml, bandwidth, periods, sources, materials)` | Configuração do estudo FDTD: frequência, tamanho em comprimentos de onda, resolução, CPML, fontes e materiais. |
| `FDTDRun` | classe | `(config, dx, n, power, current_moment, ntff, frames, frame_times, volume, volume_grid, sl…)` | Resultado do estudo: potência em f0, momento de corrente, quadros de E_z, campos de volume e diagrama de radiação (`pattern`). |
| `run_fdtd` | função | `(cfg, *, snapshots=24, volume_half=3, volume_steps=24, maxwell_block=True, snap_window=(2…)` | Roda o estudo configurado e devolve potência, snapshots dos planos centrais e campos em volume. |
| `ladder_values` | função | `(values) -> list[float]` | Converte uma sequência de valores da escada de refinamento em floats. |

#### `pinneapple_science.em.induction`

Indução e circuitos (acadêmico): lei de Faraday com um ímã caindo por uma espira (3-D + tempo) e circuitos RC, RL e RLC (0-D). [sem teste direto]

<small>`pinneapple_science/em/induction.py` · 271 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FaradayInduction` | classe | `()` | Experimento `em.faraday_induction`: calcula fluxo e FEM de um ímã cilíndrico caindo pela espira e compara com a referência. |
| `Circuits` | classe | `()` | Experimento `em.circuits_rc_rl_rlc`: integra as respostas ao degrau de RC, RL e RLC e confere constantes de tempo e frequências. |
| `magnet_rings` | função | `(a, length, Br, n_rings, n_seg, z_center=0.0)` | Discretiza um ímã cilíndrico em anéis poligonais da corrente superficial ligada K = Br/μ0. |
| `loop_flux` | função | `(segs, I_ring, R, n_path=64) -> float` | Fluxo através de uma espira circular pela circulação do potencial vetor (teorema de Stokes). |

#### `pinneapple_science.em.research`

Eletromagnetismo em nível de pesquisa: FDTD 3-D com convergência e ordem observada, incerteza por Monte Carlo, permissividade por problema inverso e surrogate com erro de validação. [sem teste direto]

<small>`pinneapple_science/em/research.py` · 548 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FDTDStudy3D` | classe | `()` | Experimento `em.research_fdtd_3d`: estudo FDTD configurável com convergência de malha (Richardson e GCI). |
| `WaveguideUQ` | classe | `()` | Experimento `em.research_uq_waveguide`: propaga por Monte Carlo as tolerâncias de fabricação até as frequências de corte de um guia. |
| `InversePermittivity` | classe | `()` | Experimento `em.research_inverse_permittivity`: recupera permissividade e perda de uma placa a partir do espectro de reflexão. |
| `HallSurrogate` | classe | `()` | Experimento `em.research_surrogate_hall`: processo gaussiano treinado em soluções de Hall, com erro em dados retidos. |
| `slice_viewer` | função | `(run, path, title='FDTD: E_z in 3-D + time') -> str` | Gera uma página three.js de arquivo único com os três planos centrais e E_z animado no tempo. |

#### `pinneapple_science.em.statics`

Eletro e magnetostática (acadêmico): Coulomb e superposição, lei de Gauss, capacitor de placas com efeito de borda, Biot-Savart, lei de Ampère e efeito Hall, cada um com escada de refinamento contra a fórmula de livro.

<small>`pinneapple_science/em/statics.py` · 901 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CoulombSuperposition` | classe | `()` | Experimento `em.coulomb_superposition`: anel e linhas de carga por superposição de cargas pontuais contra a fórmula exata. |
| `GaussLaw` | classe | `()` | Experimento `em.gauss_law`: fluxo de E por quadratura numa superfície fechada contra a carga envolvida sobre ε0. |
| `ParallelPlateCapacitor` | classe | `()` | Experimento `em.parallel_plate_capacitor`: capacitância pela energia de uma solução de volumes finitos, contra Palmer e Kirchhoff. |
| `BiotSavart` | classe | `()` | Experimento `em.biot_savart`: integral de Biot-Savart numérica para fio, espira e solenoide. |
| `AmpereLaw` | classe | `()` | Experimento `em.ampere_law`: circulação de B em caminhos fechados contra μ0 vezes a corrente envolvida. |
| `HallEffect` | classe | `()` | Experimento `em.hall_effect`: potencial 2-D numa barra Hall com tensor de condutividade de Drude. |
| `ring_potential` | função | `(points, Q, R) -> np.ndarray` | Potencial exato de um anel uniformemente carregado (integral elíptica, Jackson). |
| `palmer_strip_capacitance` | função | `(w, d, eps_r=1.0) -> float` | Capacitância por unidade de comprimento de duas fitas finas com correção de borda de Palmer (1937), assintótica para w >> d. |
| `kirchhoff_disc_capacitance` | função | `(a, d, eps_r=1.0) -> float` | Capacitância de dois discos coaxiais finos pela fórmula de Kirchhoff (1877), assintótica para d << a. |
| `strip_capacitor_2d` | função | `(w, d, h, L=40.0, growth=1.08)` | Capacitância de duas fitas finas por volumes finitos num quarto de domínio graduado. |
| `disc_capacitor` | função | `(a, d, h, *, cartesian, L=12.0)` | Capacitância de dois discos coaxiais, em solução axissimétrica ou 3-D cartesiana num oitavo do domínio. |
| `hall_bar` | função | `(L, W, beta, n, V=1.0, nx=None)` | Potencial numa barra Hall com contatos nas pontas e bordas isolantes, para um parâmetro de Hall dado. |

#### `pinneapple_science.em.waves`

Ondas e óptica (acadêmico): radiação de dipolo, onda plana em meio com perdas, Fresnel e Snell, modos de guia de onda, ressonância de cavidade e espalhamento de Mie, via FDTD ou autovalores contra formas fechadas. [sem teste direto]

<small>`pinneapple_science/em/waves.py` · 599 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DipoleRadiation` | classe | `()` | Experimento `em.dipole_radiation`: campos e fluxo de Poynting de um dipolo hertziano, em 3-D e 3-D + tempo. |
| `LossyPlaneWave` | classe | `()` | Experimento `em.lossy_plane_wave`: mede atenuação e fase de um pulso num dielétrico condutor por FDTD 1-D. |
| `FresnelSnell` | classe | `()` | Experimento `em.fresnel_snell`: coeficientes de reflexão s e p e lei de Snell por FDTD 2-D. |
| `WaveguideModes` | classe | `()` | Experimento `em.waveguide_modes`: frequências de corte TE e TM de um guia WR-90 por autovalores de Helmholtz. |
| `CavityResonance` | classe | `()` | Experimento `em.cavity_resonance`: ressonâncias de uma cavidade PEC pelo espectro de uma sonda em FDTD. |
| `MieScattering` | classe | `()` | Experimento `em.mie_scattering`: eficiências de espalhamento e extinção de uma esfera pela série de Mie contra valores publicados. |
| `dipole_cartesian` | função | `(X, Y, Z, Il, omega)` | Campos fasoriais E e H em componentes cartesianas de um dipolo hertziano na origem. |

#### `pinneapple_science.instruments`

Instrumentos virtuais (erro de ganho e offset sorteados uma vez, ruído por leitura, resolução) e ajustes lineares e não lineares com incerteza GUM tipo A e tipo B.

<small>`pinneapple_science/instruments.py` · 194 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Instrument` | classe | `(name, unit, noise, gain_u, offset_u, resolution, seed, _rng, _gain, _offset)` | Instrumento de leitura com ruído, incerteza de ganho e de offset e resolução conhecidos só pela especificação. |
| `Oscilloscope` | classe | `(rate, duration, v_range, bits, noise, gain_u, timebase_u, seed)` | Osciloscópio digital virtual: amostra um sinal com ruído, erro de ganho, ADC de `bits` e erro de base de tempo. |
| `LinearFit` | classe | `(slope, intercept, slope_u, intercept_u, cov, chi2, dof, residuals)` | Resultado de ajuste linear: inclinação, intercepto, incertezas, covariância, qui-quadrado e resíduos. |
| `CurveFit` | classe | `(params, errors, cov, chi2, dof, residuals)` | Resultado de ajuste não linear: parâmetros, erros, covariância, qui-quadrado e resíduos. |
| `FieldProbe` | função | `(**kw) -> Instrument` | Cria um `Instrument` que representa uma sonda Hall de campo magnético (tesla). |
| `Ammeter` | função | `(**kw) -> Instrument` | Cria um `Instrument` amperímetro (unidade A). |
| `Voltmeter` | função | `(**kw) -> Instrument` | Cria um `Instrument` voltímetro (unidade V). |
| `linear_fit` | função | `(x, y, sigma_y=None, *, through_origin=False, scale_by_chi2=True) -> LinearFit` | Mínimos quadrados ponderados y = a x + b, com incertezas da covariância escaladas pelo qui-quadrado reduzido. |
| `curve_fit` | função | `(model, x, y, p0, sigma_y=None) -> CurveFit` | Mínimos quadrados não lineares (scipy `curve_fit`) com a mesma regra de incerteza. |
| `combine` | função | `(*relative_uncertainties) -> float` | Soma quadrática de incertezas relativas independentes (GUM). |

#### `pinneapple_science.numerics`

Blocos numéricos compartilhados: malhas tensoriais graduadas, volumes finitos para `div(eps grad phi) = -rho` em 1-D, 2-D e 3-D, energia de campo, autovalores de Helmholtz, picos espectrais e escala de GP.

<small>`pinneapple_science/numerics.py` · 244 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TensorGrid` | classe | `(axes)` | Coordenadas dos nós por eixo (1, 2 ou 3 eixos), com malha e larguras duais. |
| `graded_axis` | função | `(core, h, length, growth=1.12, start=0.0) -> np.ndarray` | Gera nós com passo fixo numa região central e crescimento geométrico até o fim do domínio. |
| `solve_potential` | função | `(grid, fixed, values, *, eps=None, rho=None, tol=1e-10, direct=None, mean='harmonic') -> …)` | Resolve `-div(eps grad phi) = rho` com nós de potencial fixo e faces de simetria. |
| `field_energy` | função | `(grid, phi, eps=None, mean='harmonic') -> float` | Energia discreta do campo 1/2 Σ ε \|∇φ\|² dV com os fluxos de face do esquema. |
| `helmholtz_eigs` | função | `(nx, ny, a, b, kind, n=6, inside=None, modes=False)` | Menores números de onda de corte de uma seção de guia (TM ou TE) por autovalores de Helmholtz. |
| `spectral_peaks` | função | `(signal, dt, f_max, *, n_peaks=8, min_rel=0.001, pad=8) -> np.ndarray` | Frequências dos picos mais fortes do espectro de um sinal, com janela de Hann, zero-padding e interpolação parabólica. |
| `refine_ladder` | função | `(base, levels, ratio=2) -> Sequence[int]` | Escada de refinamento `base * ratio**i`. |
| `gp_loo_scale` | função | `(gp) -> float` | Fator de escala do desvio preditivo de um processo gaussiano a partir dos resíduos leave-one-out. |

#### `pinneapple_science.plots`

Convenções de figura: paleta fixa (calculado e referência), mapas sequencial e divergente, eixos discretos; nada é salvo aqui. [sem teste direto]

<small>`pinneapple_science/plots.py` · 83 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `style` | função | `(ax) -> None` | Aplica o estilo do pacote a um eixo matplotlib. |
| `figure` | função | `(ncols=1, nrows=1, *, width=4.2, height=3.3, three_d=(), title='')` | Cria figura e lista plana de eixos, com eixos 3-D nos índices pedidos. |
| `badge` | função | `(ax, text, ok=True, loc='lower right') -> None` | Caixa de texto com um resultado medido, com moldura indicando passou ou falhou. |
| `close` | função | `(fig) -> None` | Fecha uma figura matplotlib. |
| `log_ticks_ok` | função | `(values) -> bool` | Diz se todos os valores são positivos (válidos para eixo log). |
| `to_rgb` | função | `(fig) -> np.ndarray` | Converte a figura num array RGB, para quadros de animação. |
| `integer_ticks` | função | `(ax, values) -> None` | Põe rótulos inteiros simples num eixo x logarítmico. |

#### `pinneapple_science.registry`

Registro dos experimentos por nome; os módulos são importados no primeiro uso. [sem teste direto]

<small>`pinneapple_science/registry.py` · 68 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `register` | função | `(cls) -> type[Experiment]` | Decorador de classe que adiciona um experimento ao catálogo. |
| `get` | função | `(name) -> Experiment` | Instancia o experimento pelo nome, com erro listando os conhecidos. |
| `names` | função | `() -> list[str]` | Lista os nomes dos experimentos registrados. |
| `experiments` | função | `(area=None, *, level=None, kind=None, dim=None) -> list[dict[str, Any]]` | Cartões dos experimentos que casam com os filtros (área, dimensão etc.). |
| `run` | função | `(name, dim=None, *, quick=False, **params) -> Result` | Roda um experimento numa dimensão (a primeira por padrão) com parâmetros sobre os padrões. |

#### `pinneapple_science.report`

Resultados versionados: uma pasta por experimento em `examples/science/` com todas as dimensões, manifesto do Lab, selo e README, mais o catálogo e o atlas. [sem teste direto]

<small>`pinneapple_science/report.py` · 204 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `folder_of` | função | `(name, root=None) -> str` | Caminho da pasta de resultados de um experimento. |
| `write` | função | `(name, *, root=None, quick=False, register=True, seal=True) -> dict[str, Any]` | Roda um experimento em todas as dimensões, salva, escreve README e manifesto, registra no Lab e carimba as figuras. |
| `readme` | função | `(name, results, sealed, item) -> str` | Gera o README de um experimento a partir dos resultados. |
| `table` | função | `(root=None) -> list[dict[str, Any]]` | Tabela experimento x dimensão x grandeza x referência x erro medido a partir dos `result.json` versionados. |
| `write_catalogue` | função | `(root=None, register=True) -> str` | Escreve `examples/science/README.md`, `ATLAS.md`, `atlas.html` e o manifesto da pasta. |
| `write_all` | função | `(*, root=None, quick=False, register=True) -> list[dict[str, Any]]` | Roda `write` para todos os experimentos e depois escreve o catálogo. |

### Exemplos de uso reais do repositório

**Atlas, filtros e execução de um experimento** — `docs/guides/science.md`

```python
import pinneapple as pp

pp.science.atlas()                                     # every area and subarea, with its status and pointers
pp.science.experiments(area="electromagnetism", dim="3D+t")
r = pp.science.run("em.faraday_induction")             # a magnet falling through a loop, 3-D in time
print(r)                                               # computed vs reference, physics checks, trust decision
r.save("out/faraday")                                  # result.json, figures, the animation (MP4 and GIF)
```

**Estudo FDTD 3-D configurado por dicionário** — `docs/guides/science.md`

```python
from pinneapple_science.em.fdtd_study import FDTDConfig, run_fdtd

cfg = FDTDConfig.from_dict({"f0": 1e9, "size": 2.4, "cells_per_wavelength": 14,
                            "sources": [{"kind": "dipole", "component": "Ez", "at": [0, 0, 0]}],
                            "materials": [{"shape": "box", "center": [0.5, 0, 0], "size": [0.2, 0.6, 0.6],
                                           "eps_r": 6.0, "sigma": 0.01}]})
run = run_fdtd(cfg, snapshots=24)            # run.power (W at f0), run.pattern(theta), run.slices, run.volume
```

**Todo experimento rápido passa nas suas referências** — `tests/test_science.py`

```python
@pytest.mark.parametrize("name", FAST)
def test_experiment_passes_its_references(name):
    e = S.get(name)
    for d in e.dims:
        r = e.run(d, quick=True)
        assert r.comparisons, f"{name} [{d}] compares nothing"
        bad = [c.quantity for c in r.comparisons if not c.passed] + [k["name"] for k in r.checks if k["passed"] is False]
        assert not bad, f"{name} [{d}]: {bad}"
        assert r.decision == "APPROVED"
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 6 de 16.
- Funções de teste em arquivos que importam algum módulo do pacote: 19.

### Limitações conhecidas

- Só a área de eletromagnetismo (`em`) tem experimentos implementados; as demais áreas aparecem no atlas como parciais ou planejadas, com ponteiros para código de outros pacotes.
- As fórmulas de Palmer e de Kirchhoff usadas como referência do capacitor são assintóticas; o guia `docs/guides/science.md` diz que a borda da placa fina limita a ordem dos volumes finitos a cerca de 1.
- O dipolo ao lado de uma esfera converge em primeira ordem (esfera em escada), segundo o guia; os estudos de pesquisa reportam GCI e incerteza em vez de esconder esse erro.
- `Experiment._run` da classe base levanta `NotImplementedError`; cada experimento precisa implementá-lo.
- A maioria dos módulos não tem teste direto; `tests/test_science.py` cobre o pacote pelo registro e roda os experimentos rápidos em modo `quick`.
- `report.write` depende do `pinneapple_lab` (manifesto, registro e selo) e escreve em `examples/science/`; ajustes não lineares usam scipy e as figuras usam matplotlib.

## 5. `pinneapple_condense`

### Objetivo

`pinneapple_condense` implementa a "Adaptive Physics Condensation" (o `__init__` diz "work in progress"): resolver problemas físicos grandes feitos de blocos repetidos sem montar e resolver a malha fina inteira. O caso de referência é um encapsulamento de chip sintético com várias camadas (substrato, bumps C4, interposer com TSVs, microbumps, die, TIM, tampa de cobre), montado a partir de uma célula unitária repetida. Os materiais são valores representativos, não dados de um produto real.

A ideia central é a condensação estática (complemento de Schur): o interior de cada bloco é eliminado uma vez por tipo distinto de bloco, e o problema global fica só nas interfaces, que ainda podem ser reduzidas a poucos modos por face (POD ou modos de restrição). Sobre isso o pacote constrói uma hierarquia que reaproveita subárvores idênticas (`hierarchy`), um estimador de erro com refinamento por face, uma correção exata por PCG, uma versão transiente Craig-Bampton (`transient`), operadores de bloco aprendidos com verificação e fallback exato (`learned`), a homogeneização Keff como linha de base (`homogenize`), um método de "prior + patches de exceção" (`exceptions`) e uma versão algébrica que condensa qualquer matriz esparsa (`algebraic`).

O pacote também tem as soluções de referência (volumes finitos de condução em `fv`, escoamento 2D em grade MAC em `flow`), discretizações sem malha RBF-FD/GFDM (`meshless`, `meshless2d`) e os estudos medidos: `benchmark` gera os números de `docs/research/physics_condensation.md` e `falsify` faz os testes pré-registrados de `docs/research/condensation_falsification.md`; `figures` desenha os gráficos.

Liga-se ao resto do repositório por `query`, que fornece os métodos (cache, limite rigoroso, superposição, condensado, referência) do motor de consultas físicas `pinneapple_query.Engine`; pelos nós `condense.*` e `query` de `pinneapple_flow`; e por `pinneapple_analysis.validation`. Os exemplos `examples/physics_condensation` e `examples/condensation_falsification` rodam os estudos.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 18 | 6.006 | 29 | 115 | 27 (3 arquivos) | 3 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `algebraic` | Condensação algébrica exata de qualquer operador esparso montado, particionado em blocos por posição, com faces reduzidas por modos POD. | 1 | 6 | 6 |
| `benchmark` | Estudos medidos do benchmark de condensação (convergência, modos, refinamento, escala, aprendido, transiente), cada um devolvendo números em JSON. | 1 | 14 | 0 |
| `exceptions` | Método "prior + patches de exceção": uma solução barata (Keff ou coluna 1D) corrigida exatamente só nos blocos de maior resíduo. | 1 | 6 | 14 |
| `falsify` | Testes de falsificação pré-registrados do método (térmicos T1-T10, CFD F0-F3, sem malha M1-M4, patches E1). | 1 | 21 | 6 |
| `figures` | Figuras matplotlib do benchmark a partir do JSON de `benchmark`. | 1 | 8 | 0 |
| `flow` | Escoamento 2D incompressível e calor em grade escalonada (MAC) por volumes finitos: Stokes/Oseen, Picard para Navier-Stokes e equação escalar. | 1 | 9 | 6 |
| `fv` | Volumes finitos centrados na célula para condução estacionária e transiente numa caixa de voxels, e o solver de referência. | 1 | 8 | 21 |
| `hierarchy` | Condensação hierárquica aninhada com memoização de subárvores idênticas. | 1 | 2 | 14 |
| `homogenize` | Homogeneização por condutividade efetiva (Keff), a linha de base que a condensação precisa superar. | 1 | 7 | 14 |
| `learned` | Operador de bloco reduzido aprendido (POD + regressão GP) com indicadores de erro e fallback para condensação exata. | 1 | 10 | 14 |
| `meshless` | Solução sem malha (RBF-FD/GFDM) do encapsulamento a partir da geometria implícita, com pares de nós nas interfaces. | 1 | 9 | 14 |
| `meshless2d` | Discretizações 2D sem malha dos casos de CFD da falsificação (advecção-difusão, Stokes, condução híbrida FV/RBF-FD). | 1 | 9 | 6 |
| `package` | Encapsulamento de chip multicamada sintético feito de um bloco repetido: camadas, variantes, mapas de potência. | 1 | 7 | 27 |
| `query` | Métodos de consulta física sobre o encapsulamento para `pinneapple_query.Engine` (cache, limite, superposição, condensado, referência). | 1 | 8 | 7 |
| `query_benchmark` | Demonstração medida do motor de consultas sobre o encapsulamento contra a referência fina. | 1 | 1 | 0 |
| `substructure` | Condensação estática de blocos repetidos, interfaces reduzidas, recuperação sob demanda, estimativa de erro e refinamento. | 1 | 15 | 21 |
| `transient` | Condução transiente com redução Craig-Bampton por bloco e Euler implícito, mais a referência fina. | 1 | 4 | 14 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_condense`

Pacote da Adaptive Physics Condensation; o `__init__` só tem o docstring "work in progress" e não reexporta nada. [experimental]

<small>`pinneapple_condense/__init__.py` · 2 linhas · testes: 27</small>

#### `pinneapple_condense.algebraic`

Aplica o método de `substructure` a qualquer matriz esparsa (simétrica ou não, ponto de sela ou não): separadores pelo grafo de acoplamento, blocos comparados por conteúdo e faces reduzidas por modos.

<small>`pinneapple_condense/algebraic.py` · 299 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Partition` | classe | `(block, sep, interiors, adjacent, shape, size, origin, coords, fieldid)` | Partição das incógnitas: bloco de cada incógnita, máscara de separador e, por bloco, interiores e separadores adjacentes em ordem canônica. |
| `Condensation` | classe | `(A, part, timings)` | Condensação algébrica exata de `A x = b` sobre uma `Partition`, fatorando uma vez cada bloco de conteúdo distinto. |
| `Reduced` | classe | `(cond, fc, modes)` | Problema do separador reduzido por Galerkin sobre modos de face (exato quando cada face mantém todos os modos). |
| `partition` | função | `(A, coords, block_size, origin=(0.0, 0.0), fieldid=None, primal=None, h=None) -> Partition` | Divide as incógnitas em blocos de lado `block_size` e define os separadores pelo grafo de acoplamento, com `primal` opcional. |
| `faces` | função | `(part, A) -> dict` | Agrupa os separadores pelo conjunto de blocos que acoplam: faces entre dois blocos com posições canônicas; cantos e incógnitas primais ficam completos. |
| `face_modes` | função | `(snapshots, fc, m) -> dict` | Modos POD por tipo de face a partir de snapshots: a constante mais os `m - 1` principais vetores singulares. |

#### `pinneapple_condense.benchmark`

Estudos medidos do benchmark de condensação; cada função devolve números JSON e pode rodar num processo novo (`isolated`) para medir memória de pico. [sem teste]

<small>`pinneapple_condense/benchmark.py` · 396 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `peak_rss_gb` | função | `() -> float` | Memória residente de pico do processo atual, em GB. |
| `isolated` | função | `(func, timeout=3600, module=…, **kwargs) -> dict` | Roda `<module>.<func>(**kwargs)` num processo Python novo e devolve o resultado mais o pico de memória desse processo. |
| `mesh_convergence` | função | `(shape=(8, 8), levels=((4, 1), (8, 2), (16, 4)), seed=1) -> dict` | Soluções de referência em resoluções crescentes, ordem observada, extrapolação de Richardson e erro da resolução padrão. |
| `accuracy_vs_modes` | função | `(shape=(12, 12), ms=(1, 2, 4, 6, 8, 12, 16, 24), seed=1, modes='pod') -> dict` | Compara o condensado com a referência (Tmax, campo de junção, calores de interface, balanço de energia, estimativa de erro) para vários números de modos `m`. |
| `refinement_study` | função | `(shape=(12, 12), m0=2, steps=6, seed=1, hot=True) -> dict` | Refinamento de faces guiado pelo erro (Doerfler) contra aumentar `m` uniformemente, com hotspot menor que o passo. |
| `correction_study` | função | `(nb=32, ms=(6, 12), seed=0, power_scale=4.0, rtol=1e-09, every=10) -> dict` | Correção exata (`Condensed.correct`) num caso grande com hotspot: iterações até a solução fina e erro de Tmax ao longo delas. |
| `homogenization_study` | função | `(shape=(12, 12), seed=1) -> dict` | O que a homogeneização Keff apaga em três casos: potência suave, hotspot menor que o passo e bumps com vazios. |
| `scaling_point` | função | `(nb, m=12, reference=True, flat=True, hier=True, seed=2) -> dict` | Um tamanho de encapsulamento `nb x nb`: tempo e resultado de referência, condensação plana e hierarquia. |
| `configurations` | função | `(nb=64, m=12, n_configs=20, seed=0, check=2) -> dict` | Constrói uma vez e resolve vários mapas de potência novos, para separar custo de construção e custo por mapa. |
| `learned_study` | função | `(nb=32, m=12, n_train=48, seed=1, defects=4, exact_reference=True, tol_T=0.01) -> dict` | Operadores de bloco aprendidos com verificação contra a condensação exata por bloco, com defeitos. |
| `transient_study` | função | `(shape=(6, 6), dt=0.0002, steps=150, mk=((8, 8), (16, 16), (32, 32)), seed=1) -> dict` | Transiente Craig-Bampton contra a referência fina, com potência alternando entre dois mapas. |
| `root_compressibility` | função | `(nb=64, m=12, tol=1e-08) -> dict` | Posto numérico dos blocos fora da diagonal do complemento de Schur raiz: quanto uma representação H-matriz economizaria. |
| `meshless_study` | função | `(nb=2, hfs=(6, 8, 10, 12), methods=('gfdm', 'rbf_fd'), interfaces=('pairs', 'naive'), ls_…)` | Trilha 1b: nuvens RBF-FD e GFDM contra a referência FV para vários espaçamentos, tratamentos de interface e colocação por mínimos quadrados. |
| `exceptions_study` | função | `(shape=(12, 12), seed=1, budgets=…) -> dict` | Trilha 1c: prior Keff com patches escolhidos por resíduo contra patches aleatórios no mesmo orçamento, e o prior de coluna 1D. |

#### `pinneapple_condense.exceptions`

Trilha 1c: um campo prior barato (Keff ou coluna 1D) e solução exata só nos patches de blocos onde o resíduo do prior é grande, com correção final de energia.

<small>`pinneapple_condense/exceptions.py` · 190 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `prior` | função | `(pkg, kind='keff') -> tuple[np.ndarray, float]` | Calcula o campo prior nas células finas e o seu custo em segundos. |
| `block_indicator` | função | `(pkg, sysm, T, f) -> np.ndarray` | Norma do resíduo `K T - f` sobre as células de cada bloco. |
| `correct` | função | `(pkg, T0, blocks, *, halo=1, blend='core', sysm=None, f=None) -> tuple[np.ndarray, dict]` | Resolve o problema fino nos patches em volta dos blocos escolhidos, com `T0` como dado de Dirichlet fora. |
| `energy_correct` | função | `(pkg, T, sysm) -> tuple[np.ndarray, float]` | Soma `alpha * phi` (resposta a um mapa uniforme) para que o fluxo de saída Robin iguale a potência. |
| `run` | função | `(pkg, *, budget=0.1, select='residual', halo=1, blend='core', prior_kind='keff', energy=T…)` | O método inteiro para um orçamento (fração de blocos escolhidos): campo e custos. |
| `pareto` | função | `(pkg, budgets=…, *, halos=(0, 1), selects=('residual', 'random'), prior_kind='keff', seed…)` | Erro contra fração de células resolvidas exatamente, por regra de seleção e halo, com a referência. |

#### `pinneapple_condense.falsify`

Estudos de falsificação pré-registrados do método de condensação; cada função é um teste que devolve números JSON e deve rodar num processo novo.

<small>`pinneapple_condense/falsify.py` · 989 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `t1_reference` | função | `(nb, seed=2, m=12, rtol=1e-10, fine=True, maxiter=2000) -> dict` | Temperatura de junção de pico de referência: solução fina por AMG e condensação exata por PCG com o histórico de convergência. |
| `t1_point` | função | `(nb, m, seed=2) -> dict` | Hierarquia memoizada com `m` modos: temperatura de pico e tempos de construção e solução. |
| `f0_validation` | função | `(poiseuille=(16, 32, 64), cavity=(32, 64, 128), nusselt=(16, 32, 64)) -> dict` | Valida o solver de escoamento contra Poiseuille plano, a cavidade com Re = 100 e o número de Nusselt de placas paralelas. |
| `f2_stokes` | função | `(n_pins=16, cells=16, pin=8, ms=(1, 2, 4, 8, 16), n_snap=24, seed=0) -> dict` | F2/F2r: Stokes num arranjo de pinos, condensação algébrica exata contra solução direta global e depois faces reduzidas por snapshots. |
| `f1_advection` | função | `(n_pins=8, cells=16, pin=8, pes=(0.1, 1, 10, 100, 1000), ms=(1, 2, 4, 8, 16), n_snap=24, …)` | F1/F1r: advecção-difusão estacionária no arranjo de pinos com varredura do número de Péclet, condensação exata do operador não simétrico e faces reduzidas. |
| `f3_cht` | função | `(n_pins=(8, 4), cells=16, pin=8, res=(1, 10, 50, 100), ms=(2, 4, 8, 16), n_snap=16, tol=1…)` | F3/F3r: placa fria com aletas, Navier-Stokes por Picard com cada iterado condensado, depois a energia conjugada e faces reduzidas. |
| `thermal_case` | função | `(pkg, m=12, rtol=1e-10) -> dict` | Um encapsulamento: referência FV, condensação com `m` modos (erro de Tmax, efetividade da estimativa) e condensação exata por PCG. |
| `t2_contrast` | função | `(nb=32, ratios=(1000.0, 10000.0, 100000.0), m=12, seed=2) -> dict` | T2: aumenta o contraste de condutividade cobre/underfill para várias razões. |
| `t3_anisotropy` | função | `(nb=32, factors=(100.0, 0.01), m=12, seed=2) -> dict` | T3: TIM e interposer com razão de condutividade no plano / através da espessura. |
| `t8_adversarial` | função | `(nb=16, m=12, seed=2) -> dict` | T8: mapas de potência diferentes do conjunto de snapshots, com a mesma potência total. |
| `t9_thin` | função | `(nb=16, m=12, seed=2) -> dict` | T9: camadas finas (TIM e die mais finos). |
| `t4_no_repetition` | função | `(nb=32, m=12, seed=1, n_train=48) -> dict` | T4: todos os blocos diferentes; caminho exato (cada bloco condensado mais correção PCG) contra o caminho aprendido. |
| `t5_variants` | função | `(nb=64, Vs=(1, 4, 16, 64, 256), m=12, seed=0) -> dict` | T5: `V` variantes discretas em posições aleatórias; construção da hierarquia contra a condensação plana. |
| `t6_t7_nonlinear` | função | `(nb=16, case='k', seed=2, power_scale=2.0) -> dict` | T6 (condutividade k(T) de silício e cobre) ou T7 (resistência de contato R_c(T) entre die e TIM). |
| `t10_pitch` | função | `(nb=16, pitches=(4e-05, 2e-05), ms=(2, 4, 8, 12, 16, 24), n_snap=24, seed=2) -> dict` | T10: colunas de blocos alternando o passo, por condensação algébrica, contra o passo uniforme. |
| `m1_advection` | função | `(pins=4, densities=(8, 16, 32), truth_cells=64, pes=(1, 10, 100), methods=('rbf_fd', 'gfd…)` | M1: advecção-difusão sem malha em nuvens de pontos contra FV com o mesmo número de incógnitas. |
| `m2_stokes` | função | `(pins=4, densities=(8, 16, 32), truth_cells=64, methods=('rbf_fd', 'gfdm'), seed=0) -> di…)` | M2: Stokes colocado sem malha nos pontos de fluido; queda de pressão contra FV. |
| `m3_hybrid` | função | `(pins=2, densities=(8, 16, 32), truth_cells=64, k_pin=400.0, k_mat=1.0, R_c=0.001, stenci…)` | M3: híbrido FV nas interfaces + sem malha no resto contra FV em toda parte, com erro separado por região e sensibilidade ao estêncil. |
| `m4_meshless_condensed` | função | `(pins=4, cells=16, seed=0) -> dict` | M4: matrizes sem malha condensadas por `algebraic` (condução híbrida e advecção-difusão). |
| `exception_patches` | função | `(A, b, x0, x_true, block, nbshape, budgets, select='residual', halo=1, seed=0)` | Versão genérica de prior + patches de exceção para qualquer sistema `A x = b`, por resíduo ou aleatório. |
| `e1_cfd` | função | `(n_pins=8, fine=16, coarse=4, budgets=…, Pe_b=100.0, Re=50.0, seed=0) -> dict` | E1: patches por resíduo contra aleatórios na advecção-difusão e no campo de energia da placa fria, com prior numa grade mais grossa. |

#### `pinneapple_condense.figures`

Figuras matplotlib do benchmark de condensação, lidas do JSON de `benchmark`. [sem teste]

<small>`pinneapple_condense/figures.py` · 214 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `scaling` | função | `(points, path) -> str` | Tempo e memória de pico contra o número de blocos (dois painéis). |
| `accuracy` | função | `(res, path) -> str` | Gráfico dos erros do condensado contra o número de modos de interface. |
| `refinement` | função | `(res, path) -> str` | Gráfico do erro contra graus de liberdade de interface, refinamento guiado contra uniforme. |
| `keff_maps` | função | `(maps, path) -> str` | Mapas da temperatura de junção fina, das médias homogeneizadas por bloco e da diferença. |
| `transient` | função | `(res, path) -> str` | Gráfico do transiente Craig-Bampton contra a referência. |
| `meshless_convergence` | função | `(res, path, mesh_error=None) -> str` | Erro de Tmax e de balanço de energia contra número de nós, por método e tratamento de interface. |
| `meshless_fields` | função | `(res, method, path) -> str` | Nuvem de pontos numa fatia vertical colorida pela temperatura e o mapa de erro da junção. |
| `pareto` | função | `(res, path) -> str` | Erro contra fração de células resolvidas exatamente (patches de exceção), por regra e halo. |

#### `pinneapple_condense.flow`

Escoamento 2D incompressível e calor em grade MAC por volumes finitos, montado como um sistema esparso: referência e operador dos testes de falsificação de CFD.

<small>`pinneapple_condense/flow.py` · 330 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Grid` | classe | `(nx, ny, h, solid, walls, closed, lid, index)` | Grade MAC: tamanho, espaçamento, células sólidas, tipo de paredes, cavidade fechada/tampa móvel e a numeração das incógnitas. |
| `ScalarIndex` | classe | `(nx, ny)` | Numeração das incógnitas escalares por célula da grade. |
| `assemble_oseen` | função | `(g, nu, inlet=None, adv=None, force=(0.0, 0.0)) -> tuple[sp.csr_matrix, np.ndarray]` | Monta o sistema de Oseen (Stokes quando `adv` é None) com perfil de entrada, velocidade advectiva e força de corpo. |
| `split` | função | `(g, x) -> tuple[np.ndarray, np.ndarray, np.ndarray]` | Separa o vetor solução em `u`, `v` e `p` (NaN nas células sólidas). |
| `solve` | função | `(A, b) -> np.ndarray` | Solução direta esparsa (`spsolve`) de `A x = b`. |
| `picard` | função | `(g, nu, inlet=None, tol=1e-08, maxiter=200, relax=1.0, linear_solver=None, force=(0.0, 0.…)` | Navier-Stokes por iteração de Picard (Oseen) a partir de Stokes, com histórico da variação relativa e tempos por iteração. |
| `assemble_scalar` | função | `(g, u, v, k, rhoc, q, T_in=0.0, scheme='hybrid') -> tuple[sp.csr_matrix, np.ndarray]` | Monta `rho c (u . grad T) - div(k grad T) = q` em todas as células, esquema híbrido ou upwind. |
| `scalar_coords` | função | `(g) -> np.ndarray` | Coordenadas dos centros das células da grade. |
| `pin_array` | função | `(n_pins, cells, pin, h, inlet_cells=0, outlet_cells=0) -> np.ndarray` | Máscara sólida de um arranjo de pinos quadrados, com colunas vazias opcionais antes e depois. |

#### `pinneapple_condense.fv`

Volumes finitos centrados na célula para condução estacionária e transiente numa caixa de voxels (contato, Robin, portas laterais), e o solver de referência.

<small>`pinneapple_condense/fv.py` · 174 linhas · testes: 21</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `System` | classe | `(K, shape, ports, g_top, g_bottom, Gz, C)` | Operador de condução montado: `K`, forma, blocos de portas, condutâncias Robin de topo e base, condutâncias verticais e capacidade `C`. |
| `SolveInfo` | classe | `(method, seconds, iterations, residual)` | Informação de uma solução: método, segundos, iterações e resíduo. |
| `assemble` | função | `(k, dz, R, dx, h_top, h_bottom, *, ports=False, rho_c=None, dy=None) -> System` | Monta o operador de condução de uma caixa de voxels (condutividade, espessuras, resistências de contato, Robin, portas e capacidade opcionais). |
| `assemble_package` | função | `(pkg, *, transient=False) -> System` | O encapsulamento inteiro como uma caixa adiabática nos lados: o modelo de referência. |
| `source_vector` | função | `(pkg, power=None) -> np.ndarray` | Vetor de fonte de calor (W) de cada célula: o mapa de potência na linha ativa. |
| `factor` | função | `(K)` | LU esparsa de uma matriz simétrica (grau mínimo em `K + K^T`, pivoteamento simétrico). |
| `solve` | função | `(K, f, *, method='auto', rtol=1e-11, maxiter=4000) -> tuple[np.ndarray, SolveInfo]` | Resolve `K T = f`: direto (SuperLU) até 6e4 incógnitas, senão gradientes conjugados com multigrid algébrico (`pyamg`) ou Jacobi. |
| `solve_package` | função | `(pkg, power=None, **kw) -> tuple[np.ndarray, SolveInfo]` | Elevação de temperatura de referência de cada célula do encapsulamento. |

#### `pinneapple_condense.hierarchy`

Condensação hierárquica: divide o arranjo de blocos ao meio recursivamente, funde filhos eliminando faces compartilhadas e memoiza operadores de subárvores idênticas.

<small>`pinneapple_condense/hierarchy.py` · 356 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Hierarchy` | classe | `(pkg, m=6, *, modes='pod', keep_root=True)` | Condensação aninhada de um encapsulamento com `m` modos por face: preparação, passagem para cima das cargas, solução, observáveis e recuperação de um bloco. |
| `estimate` | função | `(h, sol, chunk=4096) -> np.ndarray` | Indicador de erro por face de uma solução hierárquica, como em `Condensed.estimate`. |

#### `pinneapple_condense.homogenize`

Homogeneização Keff: cada camada com inclusões vira uma camada anisotrópica (regra paralela em z, limite de Hashin-Shtrikman no plano); é a linha de base que apaga o que ocorre dentro do bloco.

<small>`pinneapple_condense/homogenize.py` · 121 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `effective_layer` | função | `(k_m, k_i, phi) -> tuple[float, float]` | Condutividades efetivas `(k_xy, k_z)` de uma camada com fração `phi` de prismas alinhados. |
| `effective_properties` | função | `(stack, variant=BASE) -> dict[str, np.ndarray]` | Propriedades efetivas por linha z de um bloco: `kxy`, `kz`, `rho_c`, `dz` e resistência de contato. |
| `homogenized_system` | função | `(pkg, cells=1, transient=False)` | Sistema do encapsulamento homogeneizado com `cells x cells` células laterais por bloco. |
| `coarse_power` | função | `(pkg, cells=1, power=None) -> np.ndarray` | Mapa de potência agregado na grade grossa homogeneizada. |
| `solve_homogenized` | função | `(pkg, cells=1, power=None)` | Campo de temperatura grosso e informação da solução. |
| `homogenized_observables` | função | `(pkg, T, sysm, cells=1) -> dict[str, np.ndarray]` | Por bloco: temperatura média de junção, calor die/TIM, calor na camada de microbumps e estimativa do calor por bump. |
| `compare` | função | `(pkg, ref, hom) -> dict[str, float]` | Mede o que a homogeneização apaga contra os observáveis de referência (pico, médias, espalhamento dentro do bloco, calores). |

#### `pinneapple_condense.learned`

Operador de bloco reduzido aprendido em função de parâmetros (POD + `ParametricPOD` com GP), nunca usado às cegas: blocos incertos ou fora da caixa de treino são condensados exatamente e uma auditoria verifica o resultado.

<small>`pinneapple_condense/learned.py` · 310 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LearnedBlocks` | classe | `(stack, m, bounds, to_variant, n_train, regressor, seed, timings)` | Mapa aprendido `theta -> operador reduzido do bloco`; `fit()` condensa exatamente amostras Latin hypercube e treina a regressão. |
| `AdaptiveResult` | classe | `(obs, tb, exact_blocks, checked, failed, rounds, timings, flagged_std, flagged_ood, estim…)` | Resultado de `solve_adaptive`: observáveis, blocos exatos, verificados e reprovados, rodadas, tempos e estimativa. |
| `default_variant` | função | `(theta) -> Variant` | Converte três parâmetros (microbump, contato die/TIM e TIM/tampa, interposer) numa `Variant`. |
| `reduced_block` | função | `(stack, variant, Phi) -> dict[str, np.ndarray]` | Operador reduzido exato de um bloco sobre os modos de interface `Phi` (uma fatoração, sem cache). |
| `assemble_blocks` | função | `(Sr, dofs, n) -> sp.csc_matrix` | Monta a matriz global de interface a partir dos operadores reduzidos por bloco e seus graus de liberdade. |
| `solve_blocks` | função | `(ops, pb, dofs, n)` | Monta, fatora e resolve o problema de interface; devolve valores de porta por bloco e tempos. |
| `block_observables` | função | `(ops, tb, pb, n2) -> dict[str, np.ndarray]` | Observáveis por bloco (temperaturas de junção e fluxos) a partir dos valores de porta e das potências. |
| `solve_adaptive` | função | `(learned, theta, power, *, tol_std=0.05, tol_T=0.01, n_hot=8, n_audit=16, max_rounds=5, s…)` | Condensação aprendida com o laço de verificação: previsão, substituição por exato onde incerto, solução, auditoria e novas rodadas. |
| `solve_exact` | função | `(learned, theta, power) -> tuple[dict, float]` | Todos os blocos condensados exatamente: a referência do caminho aprendido. |
| `variant_field` | função | `(shape, seed=0, *, defects=0, defect_k=0.05, bounds=None) -> np.ndarray` | Parâmetros de bloco aleatórios e suaves dentro dos limites, com `defects` microbumps quase abertos fora da caixa de treino. |

#### `pinneapple_condense.meshless`

Trilha 1b: solução sem malha do encapsulamento direto da geometria implícita (nuvem de pontos, pares de nós com continuidade de fluxo e lei de contato nas interfaces, estênceis RBF-FD ou GFDM).

<small>`pinneapple_condense/meshless.py` · 637 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Cloud` | classe | `(X, region, kind, normal, partner, area, k, R, seconds, info)` | Nuvem de pontos: posições, região, tipo, normal, parceiro de interface, área, condutividade e resistência. |
| `Geometry` | classe | `(pkg)` | Descrição implícita do encapsulamento: material em cada ponto, camada, bloco, inclusões e fontes. |
| `MeshlessSolution` | classe | `(cloud, T, A, b, timings, stencils)` | Solução sem malha: nuvem, temperaturas, sistema montado, tempos e estênceis. |
| `point_cloud` | função | `(pkg, *, h_xy=None, per_layer=3, h_max=2.5e-05, per_feature=4, band_h=2.0, gap=0.3, jitte…)` | Gera os nós a partir da geometria implícita (retículo com jitter por região, pares em interfaces, densidade adaptativa opcional). |
| `region_stencils` | função | `(X, region, Q, q_region, n, box, reach, normal=None)` | Estênceis dos `n` vizinhos mais próximos da mesma região, com lados do encapsulamento como planos espelho. |
| `assemble` | função | `(cloud, pkg, *, method='rbf_fd', degree=2, n=None, interface='pairs', oversample=False)` | Monta o sistema de colocação (`A`, `b`, estênceis) para o método e tratamento de interface escolhidos. |
| `solve` | função | `(pkg, cloud=None, *, method='rbf_fd', degree=2, n=None, interface='pairs', solver='auto',…)` | Nuvem, montagem e solução (LU esparsa, ou GMRES com ILU acima de 2e5 nós), com estimativa de condicionamento opcional. |
| `evaluate` | função | `(sol, pkg, P, *, degree=None) -> np.ndarray` | Temperatura em pontos quaisquer, interpolada com estênceis da região de cada ponto. |
| `observables` | função | `(sol, pkg) -> dict` | Temperaturas de junção, calor die/TIM, fluxos de saída e balanço de energia da solução sem malha. |

#### `pinneapple_condense.meshless2d`

Discretizações 2D sem malha (RBF-FD/GFDM) dos casos de CFD da falsificação, em nuvens de um arranjo de pinos quadrados construídas para dar matrizes idênticas em blocos idênticos.

<small>`pinneapple_condense/meshless2d.py` · 339 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `pin_sdf` | função | `(X, pins, pin_frac, offset=(1.0, 0.0))` | Distância com sinal ao pino quadrado mais próximo (negativa dentro). |
| `cloud` | função | `(pins, h, pin_frac=0.5, jitter=0.25, inlet_blocks=1, outlet_blocks=1, seed=0, fluid_only=…)` | Pontos de retículo com jitter por célula unitária, iguais em todas as células, opcionalmente só fluido e com pontos de parede. |
| `bilinear_mac` | função | `(g, u, v, X)` | Velocidade de uma solução MAC de `flow` interpolada em pontos. |
| `snap_to_box` | função | `(X, box, h)` | Move para o contorno os pontos a menos de h/2 dele. |
| `knn` | função | `(X, Q, n, tree=None)` | Os `n` vizinhos mais próximos de cada ponto, com desempate determinístico pelo deslocamento. |
| `advection_diffusion` | função | `(X, box, h, vel, alpha, q, *, method='rbf_fd', n=15, degree=2)` | Monta `vel . grad T - alpha lap T = q` por colocação, com `T = 0` na entrada e gradiente normal zero no resto. |
| `stokes` | função | `(X, wall, box, h, nu=1.0, *, method='rbf_fd', n=20, degree=2)` | Monta Stokes colocado (u, v, p) nos pontos de fluido, com não deslizamento nas paredes dos pinos. |
| `hybrid_conduction` | função | `(pins, cells, pin_frac, k_pin, k_mat, R_c, *, band=2, point_h=None, jitter=0.25, n=12, de…)` | Condução com resistência de contato: FV numa faixa em volta das interfaces e RBF-FD no resto, acoplados por interpolação. |
| `fv_conduction` | função | `(pins, cells, pin_frac, k_pin, k_mat, R_c, q_pin=1.0, h_top=1.0)` | O mesmo problema de condução só com volumes finitos (referência e linha de base). |

#### `pinneapple_condense.package`

Encapsulamento de chip multicamada sintético feito de um bloco repetido (um C4, um TSV, um microbump), com variantes de bloco e mapas de potência; materiais representativos, não de um produto real.

<small>`pinneapple_condense/package.py` · 253 linhas · testes: 27</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Layer` | classe | `(name, thickness, nz, matrix, inclusion, side)` | Uma camada da pilha: nome, espessura, número de células em z, material da matriz, material e lado da inclusão. |
| `Variant` | classe | `(inclusion_k, removed, contact)` | O que muda num bloco sem mudar a geometria: fator de condutividade da inclusão, inclusões removidas, fator de contato. |
| `Stack` | classe | `(layers, pitch, n, contact, h_top, h_bottom, active, materials, anisotropy)` | A pilha de camadas com passo do bloco, células por bloco, resistências de contato, coeficientes Robin, camada ativa e materiais. |
| `Package` | classe | `(stack, layout, variants, power)` | Arranjo lateral de blocos sobre uma pilha, com layout de variantes e potência por célula ativa. |
| `reference_stack` | função | `(n=8, vertical=1) -> Stack` | A pilha do benchmark (passo de 40 um, cerca de 1 mm de altura, 8 x 8 células por lado de bloco por padrão). |
| `hotspot_power` | função | `(stack, shape, total, *, hotspots=3, background=0.4, width=0.08, seed=0, centres=None) ->…)` | Mapa de potência não uniforme: fundo uniforme mais hotspots gaussianos. |
| `uniform_package` | função | `(shape, stack=None, *, total_power=None, power_density=1000000.0, seed=0, hotspots=3) -> …)` | Encapsulamento de blocos idênticos com mapa de potência com hotspots. |

#### `pinneapple_condense.query`

Trilha 2: métodos de consulta física sobre o encapsulamento para `pinneapple_query.Engine` (temperatura em ponto, pico em região, calor por feature, R_ja), do mais barato ao mais caro.

<small>`pinneapple_condense/query.py` · 411 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `State` | classe | `(power, obs, method, uncertainty, seconds, solver=None, sol=None, field=None, obs_unc=Non…)` | Configuração resolvida: observáveis por bloco, método de origem, incerteza e o necessário para ler pontos internos. |
| `PackageModel` | classe | `(pkg, m=8, reference_cells=2000000)` | Maquinário comum dos métodos: o encapsulamento, os estados guardados, as hierarquias e a extração de quantidades. |
| `Cached` | classe | `(model)` | Método `cached`: lê um estado já resolvido com as mesmas condições, sem física nova. |
| `Bound` | classe | `(model)` | Método `bound`: intervalo rigoroso pela monotonicidade da M-matriz de condução. |
| `Superposition` | classe | `(model, max_states=64)` | Método `superposition`: combinação linear de estados resolvidos por mínimos quadrados nos mapas de potência, com resíduo limitado. |
| `Condensed` | classe | `(model, expected_s=1.0)` | Método `condensed`: nova solução com a hierarquia memoizada em `m`, incerteza \|T_m - T_2m\|. |
| `Reference` | classe | `(model)` | Método `reference`: solução fina por volumes finitos, o último recurso. |
| `package_engine` | função | `(pkg, m=12, reference_cells=2000000, calibrate=True) -> tuple[Engine, PackageModel]` | Monta um motor de consultas sobre um encapsulamento com esses métodos, calibrando opcionalmente o método condensado. |

#### `pinneapple_condense.query_benchmark`

Demonstração medida do motor de consultas sobre o encapsulamento: milhares de consultas comparadas com a referência fina. [sem teste]

<small>`pinneapple_condense/query_benchmark.py` · 125 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `query_demo` | função | `(nb=48, m=12, n_points=2000, n_interior=100, seed=0, reference=True) -> dict` | Roda o conjunto de consultas e reporta latência por método, método escolhido, erro e se os intervalos rigorosos contêm a referência. |

#### `pinneapple_condense.substructure`

Condensação estática dos blocos repetidos (complemento de Schur por bloco distinto), interfaces reduzidas a poucos modos, recuperação sob demanda, estimativa de erro e refinamento.

<small>`pinneapple_condense/substructure.py` · 661 linhas · testes: 21</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BlockModel` | classe | `(stack, variant, S, W, Y, Z, ports, Om, obs_names, seconds, O_p, O_t)` | Forma condensada de um bloco distinto: complemento de Schur, mapas de carga e de recuperação e observáveis. |
| `CondensedState` | classe | `(t, pb, seconds)` | Configuração condensada resolvida: graus de liberdade de interface e potências dos blocos. |
| `Condensed` | classe | `(pkg, m=6, *, refined_x=None, refined_y=None, modes='pod', shift=0.0, face_m=None)` | O encapsulamento condensado: operadores reaproveitados, `m` modos por face, sistema global fatorado uma vez, com estimativa, correção e refinamento. |
| `block_functionals` | função | `(stack, variant, sysb) -> tuple[sp.csr_matrix, list[str]]` | Linhas dos observáveis lineares de um bloco: temperaturas de junção e calores verticais em die/TIM, microbump, TSV, C4 e saída Robin. |
| `block_model` | função | `(stack, variant=BASE, *, cache=True, shift=0.0) -> BlockModel` | Condensa um bloco (com cache por pilha, variante e `shift`). |
| `clear_cache` | função | `() -> None` | Limpa o cache de modelos de bloco. |
| `face_weights` | função | `(stack) -> np.ndarray` | Área (m²) de cada nó de porta de uma face. |
| `trace_snapshots` | função | `(stack, patch=4, shifts=(0.0,), *, global_size=16, global_maps=24) -> dict[str, np.ndarra…)` | Traços de interface usados para construir os modos de face (com cache por pilha). |
| `layered_modes` | função | `(stack, orient, m, shifts=(0.0,), *, groups='layer') -> np.ndarray` | `m` modos de face que nunca ligam duas camadas: uma constante por camada e modos POD por camada. |
| `interface_modes` | função | `(bm, orient, m, method='pod', shifts=(0.0,)) -> np.ndarray` | `m` modos ortonormais (ponderados por área) de uma face compartilhada, o primeiro constante; POD por padrão. |
| `reference_observables` | função | `(pkg, T) -> dict[str, np.ndarray]` | Os mesmos observáveis de `Condensed.observables` calculados a partir de um campo fino completo. |
| `energy_norm` | função | `(K, e) -> float` | Norma de energia `sqrt(e^T K e)` de um vetor de erro. |
| `errors` | função | `(obs, ref, total_power) -> dict[str, float]` | Erros dos observáveis contra uma referência: Tmax, temperaturas de junção, calores relativos e balanço de energia. |
| `tmax_cells` | função | `(T, active_k) -> tuple[float, float]` | Máximo sobre todas as células e sobre a linha ativa (iguais pelo princípio do máximo discreto quando a potência é não negativa). |
| `iterate_refinement` | função | `(pkg, m, tol, *, fraction=0.5, max_steps=6, power=None)` | Resolve, estima e refina faces até a estimativa relativa de energia ficar abaixo de `tol`. |

#### `pinneapple_condense.transient`

Condução transiente: redução Craig-Bampton de cada bloco (modos de restrição mais modos de interface fixa), Euler implícito com passo fixo, e a referência fina.

<small>`pinneapple_condense/transient.py` · 199 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CBBlock` | classe | `(K, M, W, G, Ot, Op, lam)` | Matrizes Craig-Bampton de um bloco (rigidez, massa, carga, acoplamentos, observáveis e autovalores). |
| `CBTransient` | classe | `(pkg, m=8, k=8, shifts=TRANSIENT_SHIFTS)` | Modelo Craig-Bampton do encapsulamento com `m` modos de interface e `k` modos internos por bloco. |
| `cb_block` | função | `(pkg_stack, variant, Phi_f, k) -> CBBlock` | Redução Craig-Bampton de um bloco sobre os modos de interface `Phi_f` e `k` modos de interface fixa. |
| `reference_transient` | função | `(pkg, powers, dt, every=1) -> dict` | Euler implícito na grade fina com o mesmo passo (fatoração única ou AMG-CG para grades grandes). |

### Exemplos de uso reais do repositório

**Condensação exata igual à referência fina** — `tests/test_condense.py`

```python
def test_exact_condensation_equals_reference(case):
    pkg, T, ref = case
    c = Condensed(pkg, m=10 ** 6).prepare()
    st = c.solve()
    assert np.abs(c.field(st) - T).max() < 1e-7 * T.max()
    e = errors(c.observables(st), ref, pkg.power.sum())
    assert e["err_Tmax"] < 1e-7 and e["err_q_ubump"] < 1e-7 and abs(e["energy_balance"]) < 1e-8
    assert c.timings["distinct_blocks"] == 3                     # three variants, twelve blocks: three factorisations
```

**Estudos do benchmark rodados por nome** — `examples/physics_condensation/run_benchmark.py`

```python
from pinneapple_condense import benchmark as B
from pinneapple_condense import figures as F

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")

FULL = {
    "mesh": ("mesh_convergence", {}),
    "accuracy": ("accuracy_vs_modes", {}),
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 15 de 18.
- Funções de teste em arquivos que importam algum módulo do pacote: 27.

### Limitações conhecidas

- O próprio pacote se declara "work in progress"; o `__init__` não reexporta nada e cada submódulo é importado direto.
- O caso de teste é um encapsulamento de chip sintético: geometria simplificada (inclusões quadradas, um C4 por bloco, tampa sem balanço) e materiais representativos, não de um produto real.
- A reutilização de blocos depende de blocos idênticos; quando todos diferem, é preciso o caminho aprendido (`learned`) ou condensar cada bloco (mais caro).
- Os estudos de `benchmark` e `falsify` são pesados (o exemplo cita cerca de uma hora em 4 núcleos e ~12 GB para o maior caso) e são pensados para rodar em processo isolado.
- `fv.solve` usa `pyamg` para problemas grandes se estiver instalado; sem ele cai para gradientes conjugados com Jacobi.
- `flow` e `meshless2d` são 2D e servem como referência e operador dos testes de falsificação, não como solver de CFD geral.
- `benchmark`, `figures` e `query_benchmark` não têm teste direto.

## 6. `pinneapple_query`

### Objetivo

`pinneapple_query` é o motor genérico de consultas físicas (Physics Query Engine). A ideia: pedir uma grandeza física (por exemplo `T`) num lugar físico estável (por exemplo `die@0.3,0.7`), sob certas condições (por exemplo um mapa de potência), com uma precisão-alvo. A resposta traz o valor, a unidade, a incerteza, o domínio de validade, a proveniência e o método que a produziu, e diz se o valor foi *lido* de um estado já resolvido (`read`), *previsto* para condições não resolvidas (`predicted`) ou obtido por uma nova solução (`solved`).

O problema que resolve é escolher, a cada pergunta, o caminho mais barato que ainda cumpre a precisão pedida. Um `Planner` pergunta a cada `Method` registrado por uma `Offer` (se pode responder, a que custo esperado, com que limite de erro, dentro de que validade) e escolhe a oferta válida mais barata cujo limite atende o alvo; se nenhuma atende, fica com a de menor limite de erro; um método de último recurso (`fallback`, tipicamente uma solução completa) só entra quando ninguém oferece. O `Engine` guarda um log de cada resposta (método, tipo, latência) para auditoria e aceita ganchos (`hooks`).

O pacote é pequeno e só tem a parte genérica: `core` (consulta, ofertas, planejador, respostas) e `refs` (referências físicas independentes de malha, como `ubump[12,40]`, `interface:die/tim` ou `lid:top`). Os métodos de uma física concreta ficam junto dela: para o encapsulamento de chips, `pinneapple_condense.query.package_engine` registra os métodos `cached`, `superposition`, `bound`, `condensed` e `reference`. No resto da plataforma, o nó `query.package` de `pinneapple_flow` expõe isso como um passo de fluxo, e `examples/physics_query/run_demo.py` é a demonstração medida (resultado gravado em `examples/physics_query/results/query_demo.json`). É uma linha de pesquisa ("track 2" no CHANGELOG), não uma API estável de `pp.*`.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 3 | 252 | 7 | 2 | 7 (1 arquivos) | 2 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `core` | Consulta (`Query`), oferta (`Offer`), resposta (`Answer`), interface de método (`Method`), planejador (`Planner`) e motor (`Engine`). | 1 | 7 | 0 |
| `refs` | Referências físicas estáveis (`PhysicalRef`) e o parser das suas formas em texto (`parse_ref`). | 1 | 2 | 0 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_query`

Fachada do pacote: reexporta `Engine`, `Planner`, `Method`, `Offer`, `Answer`, `Query`, `condition_key`, `PhysicalRef` e `parse_ref`; o docstring mostra o uso com `pinneapple_condense.query.package_engine`.

<small>`pinneapple_query/__init__.py` · 17 linhas · testes: 7</small>

#### `pinneapple_query.core`

Motor de consultas: o `Planner` coleta ofertas dos métodos e escolhe a mais barata que cumpre a precisão; o `Engine` executa, mede o tempo, registra as ofertas e grava um log de auditoria. `Method` é só interface (levanta NotImplementedError); [sem teste direto] — coberto por `tests/test_query.py` via `pinneapple_condense.query`.

<small>`pinneapple_query/core.py` · 155 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Query` | classe | `(quantity, where, time, conditions, accuracy, rigorous)` | Uma consulta: grandeza, referência física (`where`), tempo, condições, precisão absoluta desejada e se só valem métodos com limite rigoroso; `key` é o hash das condições. |
| `Offer` | classe | `(method, cost_s, bound, valid, rigorous, why)` | Oferta de um método para uma consulta: custo esperado em segundos, limite de erro esperado, se é válida, se o limite é rigoroso e o porquê. |
| `Answer` | classe | `(value, unit, uncertainty, kind, method, validity, provenance, seconds, offers, rigorous,…)` | Resposta: valor, unidade, incerteza, tipo (`read`, `predicted` ou `solved`), método, validade, proveniência, tempo gasto, ofertas consideradas e, se rigorosa, o intervalo. |
| `Method` | classe | `()` | Interface de um modo de responder: `offer` decide barato se pode responder e `answer` calcula; ambos levantam NotImplementedError na classe base. |
| `Planner` | classe | `(methods, fallback=None)` | Escolhe o método: a oferta válida mais barata que atende a precisão; senão a de menor limite de erro; o `fallback` só quando nenhum método oferece (sem fallback, levanta LookupError). |
| `Engine` | classe | `(planner, units=None, default_accuracy=None)` | Ponto de entrada `query(quantity, where, ...)`: aplica a precisão padrão por grandeza, consulta o `Planner`, mede a latência, anexa as ofertas, chama os ganchos e grava o log de auditoria. |
| `condition_key` | função | `(conditions) -> str` | Hash SHA-256 estável (16 caracteres) das condições de uma consulta; arrays entram pelos seus bytes. |

#### `pinneapple_query.refs`

Referências físicas independentes de malha (componente + coordenada fracionária local, índice de bloco, faixa de blocos, interface ou face) e o parser das formas de texto. [sem teste direto] — `tests/test_query.py` testa `parse_ref` pela fachada do pacote.

<small>`pinneapple_query/refs.py` · 80 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicalRef` | classe | `(component, local, index, span, kind)` | Referência física imutável: componente, coordenadas locais fracionárias, índice de feature repetida, faixa de blocos e tipo (`point`, `feature`, `region`, `interface` ou `face:...`); `str()` volta à forma de texto. |
| `parse_ref` | função | `(where) -> PhysicalRef` | Converte textos como `die@0.31,0.72`, `ubump[12,40]`, `interface:die/tim[3:9,0:4]`, `region:die[...]` ou `lid:top` em `PhysicalRef`; rejeita coordenadas locais fora de [0, 1]. |

### Exemplos de uso reais do repositório

**Referências físicas e seus tipos** — `tests/test_query.py`

```python
def test_physical_references():
    r = parse_ref("die@0.25,0.5")
    assert (r.component, r.local, r.kind) == ("die", (0.25, 0.5), "point")
    assert parse_ref("ubump[3,4]").index == (3, 4) and parse_ref("ubump[3,4]").kind == "feature"
    i = parse_ref("interface:die/tim[0:2,1:3]")
    assert i.kind == "interface" and i.span == ((0, 2), (1, 3)) and i.component == "die/tim"
    assert parse_ref("lid:top").kind == "face:top"
    assert str(parse_ref("die@0.25,0.5,1")) == "die@0.25,0.5,1"
    with pytest.raises(ValueError):
        parse_ref("die@1.5,0.2")                                   # local coordinates are fractions
```

**Resolver uma vez, depois ler do estado resolvido** — `tests/test_query.py`

```python
def test_solved_then_read_with_physical_points(engine):
    pkg, eng, model = engine
    a = eng.query("Tj_max", "region:die", accuracy=1e-9)          # nothing solved: the reference solves
    assert a.kind == "solved" and a.method == "reference"
    b = eng.query("T", "die@0.4,0.6")
    c = eng.query("T", "die@0.4,0.6,0.5")
    assert b.kind == c.kind == "read" and b.method == "cached"
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 1 de 3.
- Funções de teste em arquivos que importam algum módulo do pacote: 7.

### Limitações conhecidas

- O pacote só traz a parte genérica: sem métodos concretos ele não responde nada. Hoje os únicos métodos estão em `pinneapple_condense.query` (encapsulamento de chips).
- `Method.offer` e `Method.answer` levantam NotImplementedError: cada física precisa implementar sua própria subclasse.
- Os módulos `core` e `refs` não têm teste que os importe diretamente; a cobertura vem de `tests/test_query.py`, que passa pelos métodos de `pinneapple_condense`.
- As formas de referência (`die@...`, `ubump[...]`, `tsv[...]`, `c4[...]`, `lid:top`) foram pensadas para encapsulamento de chips; outra geometria precisa definir seus próprios nomes de componente.
- A escolha do `Planner` confia nos custos e limites de erro que cada método declara na oferta; o motor não verifica se esses limites são verdadeiros.
- Sem precisão-alvo (nem na consulta nem em `default_accuracy`), o planejador escolhe a oferta de menor limite de erro, que pode ser a mais cara.
- É linha de pesquisa ("track 2" no CHANGELOG), sem entrada em `pp.*`.
