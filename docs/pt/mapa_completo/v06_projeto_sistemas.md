# Volume 6 — Projeto e sistemas

> Parte do [Mapa completo do PINNeAPPle](../mapa_completo.md). Gerado por `docs/pt/_mapa/gerar.py` a partir do inventário do commit `e53cdcfb04`.

Geometria, otimização de projeto, projeto a partir de intenção, sistemas dinâmicos, séries temporais, co-simulação e gêmeos digitais.

## Sumário

1. [`pinneapple_design`](#1-pinneapple_design)
2. [`pinneapple_systems`](#2-pinneapple_systems)
3. [`pinneapple_twin3d`](#3-pinneapple_twin3d)

## 1. `pinneapple_design`

### Objetivo

`pinneapple_design` reúne construção de geometria e otimização de projeto guiada por física. Esta parte cobre o `__init__` do pacote e os submódulos `geometry`, `bricks` e `thermal`.

`geometry` (antigo `pinneapple_geom`) dá a base geométrica de todo o PINNeAPPle: SDFs 2D/3D, CSG, domínios físicos prontos (canal, cavidade, tubo, L, anel, junção T), malhadores 2D/3D com gmsh, limpeza e reparo de malhas, curvatura discreta, voxelização, importação STL/STEP/IGES/meshio/CadQuery, amostradores de pontos de colocação (`MeshCollocator`, construtores de lote por tags usados pelas presets de `pinneapple_physics.pde_environment`) e CAD paramétrico de veículos e peças para CFD e render.

`bricks` é um estudo completo com blocos de montar: catálogo e massas, estática por programação linear, FEA em voxels, replay de corpo rígido com o NVIDIA Newton, metas (balanço, ponte, torre), busca evolutiva NSGA-II ou guiada por surrogate GP, validação contra fórmulas fechadas e saídas visuais (LDraw, glTF, Blender, GIF/MP4).

`thermal` traz o motor do app HeatSink Sizer (`apps/heatsink_sizer` reexporta estes módulos), com rede de fórmulas fechadas, base em volumes finitos e surrogates MLP só para triagem, e o estudo `board_cooling`, um agente (via `pinneapple_autolab`) que lê um pedido, amostra, aprende um surrogate POD+GP, otimiza, verifica numa malha fina e entrega relatório de confiança.

O `__init__` reexporta `geometry` e `design_optimizer` e oferece `build_domain`, `sample_domain` e `optimize`. O pacote se apoia em `pinneapple_physics` (fórmulas fechadas, PCB), `pinneapple_simulation` (Newton, `solid_fem`, LBM), `pinneapple_neural` (treinador, ROM, GP) e `pinneapple_tools` (visualização).

Esta parte cobre cinco submódulos de `pinneapple_design`. `aero` faz o projeto aerodinâmico completo (perfis CST, casos OpenFOAM, surrogates MLP e MeshGraphNet, aeronave leve e avião comercial, busca NSGA-II e verificação por CFD). `design_optimizer` é o núcleo genérico de otimização de projeto: problemas com variáveis, objetivos, restrições e avaliadores, métodos comparáveis sob o mesmo orçamento, verificação em alta fidelidade e presets. `geometry_library` é a biblioteca de geometrias 3D paramétricas exposta como `pp.geometries`. `intent` transforma uma meta em linguagem natural (`pp.design`) num pacote de projeto verificado, usando `design_optimizer` e a biblioteca de geometrias. `surrogate_inverse` traz cinco casos didáticos de projeto inverso por surrogate sobre fórmulas de `pinneapple_physics.closed_form`, servidos por apps.

Esta parte cobre três submódulos do `pinneapple_design`. `fitmesh` leva de uma descrição em texto a CAD verificado, malhas planejadas para o propósito e um relatório único, com ajuda opcional de modelo de linguagem. `qualitative` faz uma prévia física qualitativa de geometrias e mudanças de peças antes da simulação, e depois a confere quantitativamente. `topology` faz otimização topológica por densidade (SIMP, filtro, projeção de Heaviside, OC e MMA) em problemas estruturais, térmicos, de Stokes e de rotor de máquina elétrica.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 196 | 48.153 | 269 | 740 | 744 (54 arquivos) | 86 |

Pasta de testes dedicada: `tests/pinneapple_design` (88 funções de teste).

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `aero` | Projeto aerodinâmico de ponta a ponta: perfis CST, malha O-grid e casos OpenFOAM, surrogates (MLP e MeshGraphNet), desempenho da aeronave leve e do avião comercial, busca NSGA-II e verificação por CFD. | 23 | 111 | 72 |
| `bricks` | Blocos de montar: geometria, estática, dinâmica, FEA, metas, busca automática de projetos e visualizações. | 14 | 108 | 24 |
| `design_optimizer` | Otimização de projeto guiada por física: espaço de parâmetros, objetivos e restrições, otimizadores (gradiente, bayesiano, evolutivo), estudos de projeto com orçamento de avaliações, verificação em alta fidelidade e presets de dissipador e perfil. | 18 | 98 | 179 |
| `fitmesh` | Geometria e malha adequadas ao propósito: lê o problema, sugere geometrias de referência, gera CAD checado, planeja e gera malhas e escreve um relatório. | 12 | 38 | 21 |
| `geometry` | Base geométrica: SDFs, CSG, domínios físicos, malhas, E/S de CAD, operações de malha, amostradores e CAD paramétrico de veículos. | 56 | 255 | 195 |
| `geometry_library` | Biblioteca de geometrias 3D paramétricas e realistas (aeronaves, veículos, navios, turbinas, prédios, anatomia) com registro, construtores, checagens de malha, exportadores (STL, OBJ, glTF, USD, STEP), renderizador por software e arquivos pré-gerados no Hugging Face Hub. | 15 | 91 | 42 |
| `intent` | Projeto a partir da intenção: uma meta em linguagem natural vira especificação com unidades, plano de física, otimização multifidelidade, verificação com solver completo, robustez e um pacote de projeto com relatório, geometria e arquivos de checagem. | 19 | 84 | 24 |
| `qualitative` | Prévia qualitativa de como cada geometria ou mudança de peça afeta o resultado, por modelos físicos baratos, com checagem quantitativa. | 5 | 25 | 13 |
| `surrogate_inverse` | Projeto inverso guiado por surrogate em física de forma fechada: amostra, treina uma MLP pequena, desce o gradiente no design pela rede congelada e reconfere o resultado contra a física; cinco casos resolvidos. | 8 | 104 | 94 |
| `thermal` | Projeto térmico: motor e dimensionador de dissipadores, surrogates e o estudo de resfriamento de placa. | 19 | 75 | 46 |
| `topology` | Otimização topológica baseada em densidade com filtros, projeção, OC e MMA. | 6 | 17 | 9 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_design`

Ponto de entrada do pacote de projeto: reexporta `geometry` e `design_optimizer` (com os apelidos antigos `geom` e `design_opt`) e oferece três atalhos de integração.

<small>`pinneapple_design/__init__.py` · 179 linhas · testes: 34</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_domain` | função | `(name, dim=2, **kwargs)` | Devolve um domínio físico pelo nome: chama `get_domain_3d` se `dim == 3` e `get_domain` caso contrário. |
| `sample_domain` | função | `(domain, n_interior=4096, n_boundary=512)` | Chama `domain.sample_interior` e `domain.sample_boundary` e devolve os pontos de colocação internos e de contorno; os domínios de `gen.domains`/`domains3d` não têm `sample_boundary` (só `sample_boundary_region`), então só funciona com objetos que tenham esse método, como as formas de `csg`. |
| `optimize` | função | `(param_space, surrogate, objective, *, constraints=None, method='bayesian', n_trials=100,…)` | Atalho de otimização de projeto: embrulha o modelo em `PhysicsSurrogate` se preciso, monta `DesignOptConfig` e roda `DesignOptLoop`. |

#### `pinneapple_design.aero`

Pacote de projeto aerodinâmico: formas de perfil, malhas O-grid, casos OpenFOAM, desempenho de aeronave e busca guiada por surrogate.

<small>`pinneapple_design/aero/__init__.py` · 2 linhas · testes: 17</small>

#### `pinneapple_design.aero.aircraft`

Converte a polar de uma seção em números de aeronave leve monomotor a hélice (classe Cessna 172 por padrão): velocidade máxima, CO2 por 100 km, velocidade de estol e checagem de requisitos.

<small>`pinneapple_design/aero/aircraft.py` · 154 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Aircraft` | classe | `(mass, wing_area, aspect_ratio, oswald, cd0_rest, power_kw, prop_efficiency, bsfc, cruise…)` | Dataclass com os dados da aeronave leve (massa, área e alongamento da asa, Oswald, CD0 do resto, potência, hélice, consumo, cruzeiro), com peso e corda derivados. |
| `Requirements` | classe | `(max_stall_speed, min_thickness, max_abs_cm, max_cruise_cl_ratio)` | Requisitos do design: velocidade de estol máxima, espessura mínima, \|Cm\| máximo e razão máxima entre CL de cruzeiro e CLmax. |
| `Polar` | classe | `(alpha, cl, cd, cm)` | Polar de seção a partir de coeficientes em poucos ângulos: cl(alpha) e cd, cm como funções de cl na parte colada. |
| `isa_density` | função | `(h) -> float` | Densidade do ar na atmosfera padrão (ISA, troposfera) na altitude `h`. |
| `performance` | função | `(polar, ac=Aircraft(), req=Requirements()) -> Dict[str, Any]` | Calcula o desempenho da aeronave a partir da polar: velocidade máxima, consumo e CO2 por 100 km, velocidade de estol e grandezas de cruzeiro. |
| `evaluate` | função | `(shape_props, polar, ac=Aircraft(), req=Requirements()) -> Dict[str, Any]` | Desempenho mais o requisito geométrico; o design é viável quando todas as checagens passam. |

#### `pinneapple_design.aero.aircraft3d`

Avalia a aeronave inteira em 3D: vortex lattice (asa + cauda, trimada), arrasto de seção faixa a faixa pelo surrogate de CFD, arrasto e pesos por build-up de Raymer, CG, ponto neutro e estol pela seção crítica.

<small>`pinneapple_design/aero/aircraft3d.py` · 295 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Requirements3D` | classe | `(max_stall_speed, min_thickness, min_static_margin, max_static_margin, max_tail_incidence…)` | Requisitos do design 3D: estol, espessura, margem estática mínima e máxima, incidência da cauda, estação de início do estol, envergadura e razão de CL de cruzeiro. |
| `Aircraft3D` | classe | `(x, wing_polar, tail_polar, ac=Aircraft(), req=Requirements3D(), base_weights=None, nc=4,…)` | Avalia um design 3D a partir das polares de seção (do surrogate de CFD): condição de voo, arrasto, estol e avaliação completa. |
| `airframe_from` | função | `(x, base=None) -> Airframe` | Monta um `Airframe` a partir do vetor de design (6 pesos CST + parâmetros de planta). |
| `baseline_x` | função | `() -> np.ndarray` | Vetor de design de referência: perfil NACA 2412 mais a planta de base. |
| `wing_weight` | função | `(af, Wdg, q, tc, nz=5.7, Wfw=70 * 3.785 * 0.72) -> float` | Massa da asa (kg) pela fórmula de Raymer para aviação geral, a partir da massa bruta de projeto, pressão dinâmica e combustível na asa. |
| `htail_weight` | função | `(af, Wdg, q, nz=5.7) -> float` | Massa da cauda horizontal pela fórmula de Raymer (aviação geral). |
| `vtail_weight` | função | `(af, Wdg, q, nz=5.7) -> float` | Massa da cauda vertical pela fórmula de Raymer (aviação geral). |
| `fuselage_wetted` | função | `(af)` | Área molhada da fuselagem e seu fator de forma, integrando o perímetro das seções. |
| `fuselage_cd0` | função | `(af, V, rho, mu=1.75e-05, wet=None) -> float` | CD0 da fuselagem (atrito turbulento x fator de forma x área molhada sobre a área da asa). |
| `vtail_cd0` | função | `(af, V, rho, mu=1.75e-05, vt=None) -> float` | CD0 da cauda vertical por build-up de componente. |

#### `pinneapple_design.aero.airframe`

Geometria paramétrica 3D de aeronave leve (fuselagem, asa com o perfil otimizado, caudas por coeficiente de volume, hélice, trem de pouso) com exportação para glTF binário, OpenUSD e STL.

<small>`pinneapple_design/aero/airframe.py` · 511 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Part` | classe | `(name, vertices, faces, material, group, aero)` | Uma peça da malha da aeronave: nome, vértices, faces, material, grupo (a hélice gira) e se entra na forma limpa de CFD. |
| `Airframe` | classe | `(length, width, height, wing_area, aspect_ratio, taper, sweep_le, dihedral, twist, incide…)` | Aeronave leve paramétrica (fuselagem, asa, caudas por coeficiente de volume) que gera suas peças com `build` e expõe envergadura, corda média e dados das caudas. |
| `vertex_normals` | função | `(V, F) -> np.ndarray` | Normais por vértice somando as normais das faces vizinhas. |
| `to_glb` | função | `(parts, scalars=None, fields=None) -> bytes` | Escreve glTF 2.0 binário com materiais PBR e, opcionalmente, campos por vértice (ex.: pressão de superfície). |
| `to_usda` | função | `(parts, title='aircraft') -> str` | Escreve OpenUSD em texto, uma Mesh por peça com materiais UsdPreviewSurface (Omniverse, usdview). |
| `to_stl` | função | `(parts, solid='aircraft') -> bytes` | STL binário das peças dadas, uma superfície fechada por peça. |

#### `pinneapple_design.aero.airliner`

Geometria paramétrica de avião comercial de corredor único (classe A320/737) com a mesma interface de `Airframe`, para usar o mesmo pipeline de vortex lattice e exportação. [sem teste]

<small>`pinneapple_design/aero/airliner.py` · 310 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Airliner` | classe | `(length, width, height, wing_area, aspect_ratio, taper, sweep_le, dihedral, twist, incide…)` | Avião comercial paramétrico (fuselagem circular com janelas e pintura, asa enflechada, turbofans, caudas) com a mesma interface de `Airframe`. |
| `sweep_le_from_qc` | função | `(sweep_qc, ar, taper) -> float` | Converte o enflechamento a 1/4 da corda em enflechamento do bordo de ataque, dados alongamento e afilamento. |

#### `pinneapple_design.aero.airliner3d`

Física do avião comercial (classe A320): cruzeiro transônico com Prandtl-Glauert, arrasto de onda (Korn + regra de Lock), alcance de Breguet com MTOW iterado, e lado de baixa velocidade. [sem teste]

<small>`pinneapple_design/aero/airliner3d.py` · 340 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Mission` | classe | `(passengers, mass_per_pax, range_km, cruise_altitude, tsfc, reserve, oew_rest, high_lift_…)` | Missão de projeto do avião comercial: passageiros, massa por passageiro, alcance, altitude de cruzeiro, consumo específico do turbofan e reserva. |
| `RequirementsAL` | classe | `(max_span, max_vref_kt, buffet_margin, min_static_margin, max_static_margin, max_stall_st…)` | Requisitos do avião comercial: envergadura máxima (código de aeródromo), velocidade de aproximação, margem de buffet, margem estática e estação de estol. |
| `AirlinerModel` | classe | `(x, wing_polar, tail_polar, mission=Mission(), req=RequirementsAL(), nc=4, ns_wing=12, ns…)` | Avalia um design de avião comercial: faixas da asa, arrasto de onda, cruzeiro, arrasto parasita, estol, volume de combustível e avaliação completa. |
| `isa` | função | `(h)` | Atmosfera padrão até a estratosfera baixa: densidade, velocidade do som e viscosidade (Sutherland) na altitude `h`. |
| `airliner_from` | função | `(x) -> Airliner` | Monta um `Airliner` a partir do vetor de design (seção CST + planta + Mach + posição da asa). |
| `baseline_x_al` | função | `() -> np.ndarray` | Vetor de design de referência do avião comercial (NACA 2412 + planta de base). |
| `wing_weight_transport` | função | `(af, mtow, tc, nz=3.75) -> float` | Massa da asa pela fórmula de Raymer para aviões de transporte, com um fator de correção do módulo. |

#### `pinneapple_design.aero.buildup`

Arrasto de sustentação nula (CD0) de uma aeronave da biblioteca de geometrias a partir da malha: atrito turbulento de placa plana por componente, fatores de forma de Raymer, interferência e excrescências. [sem teste]

<small>`pinneapple_design/aero/buildup.py` · 57 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `sutherland_mu` | função | `(T) -> float` | Viscosidade dinâmica do ar pela lei de Sutherland, em Pa.s. |
| `friction_drag` | função | `(geometry, ac, *, rho, V, T, mach, components=None, interference=1.08) -> dict[str, Any]` | CD0 (sobre a área da asa) dos corpos de uma `Geometry` da biblioteca cujos nomes casam com os componentes dados. |

#### `pinneapple_design.aero.case`

Monta, roda e lê um caso OpenFOAM 2D (simpleFoam, k-omega SST resolvido até a parede) de um perfil em um ângulo de ataque; também empacota o caso em zip pronto para rodar.

<small>`pinneapple_design/aero/case.py` · 302 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `write_case` | função | `(case, shape, alpha_deg, reynolds=4000000.0, iterations=1200, tu=0.001, grid=GridSpec(), …)` | Escreve o diretório do caso OpenFOAM (polyMesh, condições de contorno, controle) para uma forma, ângulo e Reynolds. |
| `read_internal` | função | `(path) -> np.ndarray` | Lê o internalField de um arquivo de campo ASCII do OpenFOAM como array (n,) ou (n, 3). |
| `wall_geometry` | função | `(P2) -> Dict[str, np.ndarray]` | Por face de parede: centro, comprimento, normal para dentro do corpo e distância do primeiro centro de célula. |
| `forces` | função | `(P2, p_wall, U_first, nu, alpha_deg) -> Dict[str, Any]` | Cl, Cd, Cm (a 1/4 da corda) e as distribuições de Cp e Cf a partir das células junto à parede. |
| `time_dirs` | função | `(case)` | Lista os diretórios de tempo numéricos de um caso OpenFOAM, em ordem. |
| `read_coefficients` | função | `(case, P2, nu, alpha_deg, ni) -> Dict[str, Any]` | Coeficientes em cada escrita guardada (checagem de convergência) e o resultado completo na última. |
| `run_case` | função | `(case, shape, alpha_deg, reynolds=4000000.0, iterations=1200, grid=GridSpec(), timeout=18…)` | Escreve e roda o simpleFoam e devolve coeficientes, veredito de convergência e tempo. |
| `case_zip` | função | `(shape, alpha_deg, reynolds=4000000.0, iterations=1200, grid=None) -> bytes` | Caso OpenFOAM pronto para rodar (com script de coeficientes e `Allrun`) como bytes de zip. |

#### `pinneapple_design.aero.case3d`

Caso OpenFOAM 3D da aeronave inteira (meio modelo com plano de simetria): STL do `Airframe`, snappyHexMesh, simpleFoam com funções de parede e leitura de forças, Cp, Cf e linhas de corrente. [sem teste]

<small>`pinneapple_design/aero/case3d.py` · 197 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `write_case3d` | função | `(case, af, alpha_deg=2.0, speed=55.0, nu=1.5e-05, iterations=800, level_wing=(5, 6), proc…)` | Escreve o caso 3D: STLs dos corpos do `Airframe`, snappyHexMesh e configuração do simpleFoam. |
| `run_case3d` | função | `(case, procs=4, log=print) -> Dict[str, float]` | Roda o caso 3D em paralelo (delegando a `run_case`). |
| `read_result3d` | função | `(case, info, af, n_lines=34) -> Dict[str, Any]` | Lê forças (pressão + atrito), Cp e Cf por face, linhas de corrente, esteira e fatias no plano de simetria. |
| `surface_scalars` | função | `(af, parts, surface, key='cp') -> Dict[str, np.ndarray]` | Passa um campo de superfície do CFD (cp ou cf) para os vértices da malha visual, por interpolação de faces vizinhas e espelhamento. |

#### `pinneapple_design.aero.dataset`

Conjunto de treino OpenFOAM dos surrogates de perfil: plano de rodadas (hipercubo latino de formas x ângulos + polares NACA), execução de uma rodada e regra do valor estacionário.

<small>`pinneapple_design/aero/dataset.py` · 92 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plan` | função | `(n_shapes, n_alpha, seed=7) -> list[dict[str, Any]]` | Lista de rodadas: formas válidas por hipercubo latino x ângulos em faixas, mais as polares NACA de referência; determinística pela semente. |
| `run_job` | função | `(job, out) -> tuple[str, Any, Any]` | Roda um job no OpenFOAM e grava `out/runs/<id>.json` (e `.npz` com campos de célula quando rodou). |
| `steady_values` | função | `(runs, max_osc=MAX_CL_OSC) -> list[dict[str, Any]]` | Cl, Cd, Cm estacionários de cada rodada; rodadas instáveis são médias das últimas escritas ou descartadas se oscilam demais. |

#### `pinneapple_design.aero.field_dataset`

Grava o dataset de campos do zoo (`airfoil_rans_cst_re4e6_fields`): campos de célula p, Ux, Uy, nut de cada rodada da campanha OpenFOAM, em arquivos `.npz` por split.

<small>`pinneapple_design/aero/field_dataset.py` · 98 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `write_field_dataset` | função | `(run_dir, out, name=…, shard_rows=SHARD_ROWS) -> dict` | Escreve o dataset de campos do zoo (train, test, naca_reference em `.npz` e `meta.json`) a partir dos arquivos de rodada da campanha. |

#### `pinneapple_design.aero.geometry`

Formas de perfil CST (Kulfan) com 6 pesos de espessura e arqueamento, ajuste de NACA 4 dígitos, propriedades derivadas, validade, hipercubo latino e alças de edição.

<small>`pinneapple_design/aero/geometry.py` · 153 linhas · testes: 72</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Loop` | classe | `(xy, n_te)` | Laço fechado de parede para a O-grid: base do bordo de fuga, superfície inferior e superior. |
| `cst_surface` | função | `(w, x, te) -> np.ndarray` | Avalia uma superfície CST (função de classe x Bernstein + espessura do bordo de fuga) nas estações x. |
| `surfaces` | função | `(shape, x) -> tuple[np.ndarray, np.ndarray]` | y das superfícies superior e inferior nas estações x (corda 1). |
| `fit_cst` | função | `(xu, yu, xl, yl) -> np.ndarray` | Pesos CST por mínimos quadrados para coordenadas dadas de extradorso e intradorso. |
| `naca4` | função | `(code, n=200) -> np.ndarray` | Ajuste CST de um perfil NACA 4 dígitos, ex.: `naca4("2412")`. |
| `properties` | função | `(shape) -> dict[str, float]` | Espessura e arqueamento máximos e suas posições, raio do bordo de ataque e área da seção. |
| `valid` | função | `(shape) -> tuple[bool, str]` | Testa se a forma é válida (superfícies não se cruzam, curvatura limitada) e devolve o motivo quando não é. |
| `outline` | função | `(shape, n=80) -> dict[str, list]` | Contorno do perfil em espaçamento cosseno, como dicionário de listas (para o navegador). |
| `latin_hypercube` | função | `(n, bounds, rng) -> np.ndarray` | Amostra `n` pontos por hipercubo latino dentro dos limites dados. |
| `wall_loop` | função | `(shape, n_surf=80, n_te=4) -> Loop` | Constrói o `Loop` de parede da forma, com pontos agrupados no bordo de ataque e de fuga. |
| `handles` | função | `(shape, xh=HANDLE_X) -> tuple[np.ndarray, np.ndarray]` | y das superfícies superior e inferior nas estações de alça (três por superfície). |
| `shape_from_handles` | função | `(y_upper, y_lower, xh=HANDLE_X) -> np.ndarray` | Forma CST cujas superfícies passam pelas alturas de alça dadas. |
| `out_of_bounds` | função | `(shape) -> list` | Nomes dos pesos CST fora da caixa de treino dos surrogates (`BOUNDS`). |

#### `pinneapple_design.aero.geometry_onnx`

Exporta a geometria CST como grafos ONNX float64 (geometria e conversão de alças para forma) que reproduzem as funções NumPy, para uso no navegador.

<small>`pinneapple_design/aero/geometry_onnx.py` · 188 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `torch_geometry` | função | `(n_outline=N_OUTLINE)` | Módulo torch que leva a forma (n, 6) ao dicionário de saídas de geometria em float64. |
| `torch_from_handles` | função | `()` | Módulo torch que leva alturas de alça (n, 6) à forma CST (n, 6), como `shape_from_handles`. |
| `export_onnx` | função | `(directory, *, presets=None) -> dict[str, Any]` | Grava `cst_geometry.onnx`, `cst_from_handles.onnx` e `cst_geometry.json` no diretório e devolve o JSON. |
| `reference` | função | `(shapes) -> dict[str, np.ndarray]` | Valores NumPy que os grafos ONNX devem reproduzir (usado nos testes). |

#### `pinneapple_design.aero.mesh`

Gera a O-grid estruturada em volta do perfil, com a mesma topologia para todo design, e escreve direto o polyMesh do OpenFOAM.

<small>`pinneapple_design/aero/mesh.py` · 182 linhas · testes: 21</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GridSpec` | classe | `(n_surf, n_te, nj, first, radius, span)` | Parâmetros da O-grid: células ao longo da superfície e do bordo de fuga, camadas até o campo distante, altura da primeira célula e raio externo. |
| `grid_points` | função | `(shape, g=GridSpec()) -> np.ndarray` | Pontos da grade (nj+1, ni, 2), com j = 0 na parede. |
| `topology` | função | `(g=GridSpec()) -> Dict[str, np.ndarray]` | Faces da grade 2D na ordem do OpenFOAM, como índices (i, j) dos pontos. |
| `write_polymesh` | função | `(shape, case, g=GridSpec()) -> np.ndarray` | Escreve `constant/polyMesh` para a forma e devolve os pontos. |
| `cell_centres` | função | `(P2) -> np.ndarray` | Centros das células (nj, ni, 2) a partir dos pontos da grade. |

#### `pinneapple_design.aero.optimize`

Busca de projeto NSGA-II sobre forma do perfil + área de asa no surrogate, com três objetivos (velocidade máxima, CO2, velocidade de estol), viabilidade e nível de confiança por design.

<small>`pinneapple_design/aero/optimize.py` · 183 linhas · testes: 23</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Engine` | classe | `(bundle)` | Junta os surrogates treinados e a avaliação da aeronave: carrega o bundle, prevê coeficientes (MLP e GNN), avalia e faz a busca NSGA-II. |
| `baseline` | função | `(engine, aircraft, req, name='NACA 2412') -> Dict[str, Any]` | Avalia o design de referência (um perfil NACA com a área de asa da aeronave). |
| `summarize` | função | `(rec) -> Dict[str, Any]` | Forma compacta de um design enviada ao navegador para a nuvem de designs. |
| `infeasibility_reasons` | função | `(designs) -> dict[str, int]` | Conta, por requisito violado, por que os designs de uma busca ficaram fora do conjunto de Pareto. |

#### `pinneapple_design.aero.optimize3d`

Busca NSGA-II da aeronave inteira (perfil + planta + posição da asa), avaliada com vortex lattice, surrogate de seção e build-up de componentes.

<small>`pinneapple_design/aero/optimize3d.py` · 107 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Engine3D` | classe | `(engine)` | Motor da busca da aeronave inteira: pesos de base, avaliação 3D de um lote e busca NSGA-II. |
| `summarize3d` | função | `(r) -> Dict[str, Any]` | Forma compacta de um registro da busca 3D (velocidades, CO2, viabilidade, margem estática, etc.). |

#### `pinneapple_design.aero.optimize_airliner`

Busca NSGA-II do avião comercial: seção da asa, planta, Mach de cruzeiro e posição da asa; objetivos CO2 por passageiro-km, Mach e velocidade de aproximação. [sem teste]

<small>`pinneapple_design/aero/optimize_airliner.py` · 102 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EngineAL` | classe | `(engine)` | Motor da busca do avião comercial: avaliação de um lote de designs com a missão e busca NSGA-II. |
| `summarize_al` | função | `(r) -> Dict[str, Any]` | Forma compacta de um registro da busca do avião comercial (CO2 por passageiro-km, Mach, Vref, MTOW, etc.). |

#### `pinneapple_design.aero.panel`

Método de painéis de Hess-Smith (escoamento potencial incompressível) em volta de um perfil CST: a referência invíscida do benchmark de perfil.

<small>`pinneapple_design/aero/panel.py` · 128 linhas · testes: 31</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PanelResult` | classe | `(nodes, sigma, gamma, alpha_deg, x_mid, y_mid, cp, cl_kutta_joukowski, cl_pressure, cd_pr…)` | Resultado do método de painéis: nós, intensidades de fonte, vórtice, pontos médios, com cálculo de velocidade em pontos. |
| `airfoil_nodes` | função | `(shape, n_per_side=100, close_te=False) -> np.ndarray` | Nós dos painéis em sentido horário, agrupados nas pontas; opcionalmente fecha o bordo de fuga rombudo. |
| `panel_solve` | função | `(shape=None, alpha_deg=0.0, *, nodes=None, n_per_side=100, close_te=False) -> PanelResult` | Resolve o escoamento potencial em volta de um perfil CST (ou de nós dados) num ângulo de ataque. |

#### `pinneapple_design.aero.render_blender`

Renderização fotorrealista de um `.glb` de aeronave com Blender Cycles, com opção de colorir por Cp/Cf e desenhar linhas de corrente; exige `bpy`. [sem teste]

<small>`pinneapple_design/aero/render_blender.py` · 250 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `render` | função | `(glb, out, view='hero', samples=96, size=(1600, 900), cp=False, lines_npz=None, sun_eleva…)` | Renderiza um `.glb` com Blender Cycles numa vista, com opção de campo Cp/Cf, modo estúdio e linhas de corrente. |
| `main` | função | `(argv=None)` | Ponto de entrada de linha de comando do render (argparse). |

#### `pinneapple_design.aero.surrogate`

Surrogates do escoamento no perfil: uma MeshGraphNet nas células da O-grid (campos + Cl, Cd, Cm) e um ensemble de MLPs cuja dispersão é a incerteza.

<small>`pinneapple_design/aero/surrogate.py` · 134 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Layout` | classe | `(grid, i_step, max_dist, j_sel, i_sel, edge_index)` | Subconjunto fixo de células da O-grid perto do perfil usado como grafo, com as arestas multiescala. |
| `AeroGNN` | classe | `(node_in, edge_in=3, hidden=48, mp=8, n_fields=4)` | MeshGraphNet que prevê os campos p, Ux, Uy, log(1+nut/nu) por nó e, por uma leitura agregada na parede, Cl, Cd, Cm. |
| `MLPEnsemble` | classe | `(k=5, width=96)` | K MLPs pequenas em (forma, alpha) -> (Cl, log Cd, Cm); a dispersão entre elas é a incerteza. |
| `node_features` | função | `(lay, shape, alpha, pos=None) -> np.ndarray` | Atributos de nó do grafo: posição, log da distância à parede, nível, flag de parede, direção do escoamento e pesos da forma. |
| `edge_features` | função | `(lay, pos) -> np.ndarray` | Atributos de aresta: direção unitária e log do comprimento. |
| `coef_targets` | função | `(cl, cd, cm)` | Empilha (Cl, log Cd, Cm) como alvos de treino. |
| `coef_from_targets` | função | `(t)` | Inverte `coef_targets`: devolve Cl, Cd, Cm. |

#### `pinneapple_design.aero.train`

Treina os dois surrogates de perfil com as rodadas OpenFOAM e mede o erro em formas nunca vistas (split por forma) e nas polares NACA.

<small>`pinneapple_design/aero/train.py` · 334 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `load_runs` | função | `(dirs, max_osc=MAX_CL_OSC) -> list[dict[str, Any]]` | Carrega os registros de rodada utilizáveis de cada diretório, com coeficientes estacionários e o caminho dos campos. |
| `split` | função | `(recs, frac=0.15, seed=3)` | Divide em (treino, teste, referência) separando formas inteiras para teste. |
| `coefs` | função | `(recs) -> np.ndarray` | Array (n, 3) de (Cl, Cd, Cm) dos registros. |
| `metrics` | função | `(true, pred) -> dict[str, dict[str, float]]` | Erros médio e máximo e R^2 por coeficiente; para Cd também erro relativo médio e percentil 90. |
| `train_mlp` | função | `(tr, k=5, width=96, epochs=6000, seed=0)` | Treina `k` MLPs em reamostragens bootstrap (Adam + agenda cosseno) e devolve modelo e normalização. |
| `predict_mlp` | função | `(model, norm, recs)` | Média e dispersão do ensemble para (Cl, Cd, Cm). |
| `graph_data` | função | `(lay, recs)` | Monta atributos de nó e aresta e alvos de campo e coeficiente de cada rodada a partir dos `.npz`. |
| `train_gnn` | função | `(tr, te, minutes=60.0, seed=0, batch=4, log=print, *, epochs=None, hidden=48, mp=8, layou…)` | Treina a rede de grafos por um orçamento de tempo e/ou épocas, com taxa de aprendizado em cosseno. |
| `predict_gnn` | função | `(model, norm, lay, recs)` | Prevê (Cl, Cd, Cm) com a GNN e calcula o RMSE de cada campo contra as rodadas. |
| `train_surrogates` | função | `(recs, gnn_minutes=60.0, gnn_from=None, log=print, *, mlp_k=5, mlp_width=96, mlp_epochs=6…)` | Divide os dados, treina os dois surrogates e mede; devolve `(bundle, metrics)`. |

#### `pinneapple_design.aero.verify`

Confere a busca com OpenFOAM: escolhe designs ao longo da frente de Pareto, roda suas polares e compara surrogate x CFD; as rodadas servem para aprendizado ativo. [sem teste]

<small>`pinneapple_design/aero/verify.py` · 85 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `pick` | função | `(front, n) -> list[int]` | Escolhe `n` designs espaçados ao longo da frente, sempre com as duas pontas. |
| `cfd_eval` | função | `(runs, shape, area, ac, req) -> dict[str, Any]` | Desempenho e checagens da aeronave a partir da polar OpenFOAM de uma forma com dada área de asa. |
| `verification_jobs` | função | `(designs, chosen, round_)` | Gera os jobs OpenFOAM (um por design e ângulo) no formato do dataset. |
| `compare_design` | função | `(engine, did, d, runs, ac, req) -> dict[str, Any]` | Compara surrogate e OpenFOAM para um design da busca. |

#### `pinneapple_design.aero.vlm`

Vortex lattice (ferraduras) para asa + cauda horizontal de um `Airframe`; a solução linear dá sustentação, momento, carregamento, arrasto induzido, trimagem e ponto neutro.

<small>`pinneapple_design/aero/vlm.py` · 257 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Lattice` | classe | `(P1, P2, C, nrm0, surf, strip, chord, yc, dy, strip_surf, tail_rot)` | Malha de ferraduras: pontos dos vórtices ligados, pontos de controle, normais, superfície (asa ou cauda) e faixa. |
| `VLMSolution` | classe | `(L, G, s_ref, c_ref, b_ref, x_ref, CL, Cm, CLw, cl_strip, Q)` | Solução do VLM linear em alpha e incidência da cauda: coeficientes, trimagem, cl por faixa e ponto neutro. |
| `build_lattice` | função | `(af, nc=5, ns_wing=14, ns_tail=6, tail=True) -> Lattice` | Monta o `Lattice` da asa e da cauda de um `Airframe` (enflechamento, diedro, torção, arqueamento). |
| `horseshoe_induced` | função | `(P, L, legs=True)` | Velocidade induzida pelas ferraduras do lattice em pontos dados (perna ligada e pernas livres). |
| `solve` | função | `(af, x_ref, nc=5, ns_wing=14, ns_tail=6, tail=True) -> VLMSolution` | Resolve o VLM para o `Airframe` e devolve um `VLMSolution`. |
| `streamlines` | função | `(sol, alpha, it, seeds, steps=220, h=0.08, core=0.08) -> np.ndarray` | Linhas de corrente do escoamento potencial (corrente livre + ferraduras com núcleo dessingularizado). |

#### `pinneapple_design.bricks`

Subpacote de blocos de montar (tipo LEGO): geometria, estática, dinâmica, FEA, metas de projeto e um laço de projeto que se modifica sozinho; reexporta as classes principais.

<small>`pinneapple_design/bricks/__init__.py` · 67 linhas · testes: 24</small>

#### `pinneapple_design.bricks.assembly`

Montagens de blocos numa grade de pinos: posicionamento, conectividade pino-tubo e regras de validade (sem sobreposição, apoio).

<small>`pinneapple_design/bricks/assembly.py` · 315 linhas · testes: 24</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Brick` | classe | `(part, x, y, z, rot, colour)` | Um bloco colocado na grade (peça, posição x/y/z, rotação, cor), com pegada, células ocupadas, caixa e centro de massa em mm. |
| `Ground` | classe | `(kind, regions)` | O chão sob a montagem: mesa lisa (com atrito) ou placa-base com pinos, opcionalmente só em regiões retangulares. |
| `Connection` | classe | `(lower, upper, z, rect, studs)` | Contato entre dois blocos (ou bloco e chão) numa altura, com o retângulo e os pinos que se encaixam; `clutch` dá a capacidade de tração. |
| `Assembly` | classe | `(bricks=None, ground=None, name='')` | Lista de blocos sobre o chão, com adição/remoção, mapa de ocupação, conexões, grafo, componentes, validação, massa e centro de massa. |
| `colour_rgb` | função | `(name) -> tuple[float, float, float]` | Converte o nome de uma cor da paleta em RGB. |
| `stack` | função | `(part_name, n, *, offsets=None, colours=('red',), ground=None) -> Assembly` | Monta `n` cópias de uma peça empilhadas, cada uma deslocada em x por `offsets[k]` pinos. |

#### `pinneapple_design.bricks.closed_form`

Fórmulas fechadas de referência (tombamento, empilhamento harmônico, balanço preso por pinos, viga biapoiada) usadas para validar `statics`, `dynamics` e `fea`.

<small>`pinneapple_design/bricks/closed_form.py` · 77 linhas · testes: 24</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `harmonic` | função | `(n) -> float` | Número harmônico H_n. |
| `harmonic_overhang` | função | `(n, length) -> float` | Balanço máximo de `n` blocos soltos iguais, um por nível: (L/2)·H_n. |
| `harmonic_stack` | função | `(n, length, edge=0.0) -> np.ndarray` | Posições x da borda esquerda de cada bloco da pilha harmônica sobre uma mesa que termina em `edge`. |
| `tip_over_offset` | função | `(length) -> float` | Maior balanço de um bloco uniforme sozinho na borda da mesa: metade do comprimento. |
| `stud_cantilever_tension` | função | `(weight, d, studs_x, x1) -> float` | Tração igual por pino que segura uma peça em balanço, girando em torno da borda do retalho de contato. |
| `beam_reactions` | função | `(span, w, P) -> float` | Reação em cada apoio de uma viga biapoiada simétrica. |
| `beam_moment` | função | `(x, span, w, P) -> float` | Momento fletor numa posição da viga biapoiada com carga central. |
| `beam_midspan_deflection` | função | `(span, E, inertia, P=0.0, w=0.0) -> float` | Flecha no meio do vão de uma viga biapoiada (Euler-Bernoulli). |
| `tower_tip_force` | função | `(weight, half_base, height) -> float` | Força horizontal na altura `height` que tomba um bloco rígido: F = W·b/h. |

#### `pinneapple_design.bricks.dynamics`

Reprodução de corpo rígido de uma montagem com o NVIDIA Newton (`pinneapple_simulation.external_solvers.newton_physics`): blocos presos viram um corpo, juntas quebradas são soltas. Depende do Newton estar instalado.

<small>`pinneapple_design/bricks/dynamics.py` · 239 linhas · testes: 24</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Replay` | classe | `(t, pos, quat, broken, max_displacement_mm, moved, solver, wall_time_s, max_tilt_deg)` | Resultado de uma simulação Newton: trajetórias, juntas quebradas, deslocamento e inclinação máximos; `collapsed` diz se a montagem caiu. |
| `table_boxes` | função | `(asm, margin_mm=80.0) -> list[tuple[float, float, float, float]]` | Regiões do chão recortadas em volta da montagem, em mm. |
| `clusters` | função | `(asm, broken=()) -> tuple[list[list[int]], set[int]]` | Grupos de blocos unidos por juntas intactas e quais deles estão soldados a uma placa-base. |
| `rigid_scene` | função | `(asm, broken=(), joint=None, dead_loads=())` | Monta a `RigidScene` do Newton: um corpo rígido por grupo de blocos, cargas descendentes como pesos mortos. |
| `replay` | função | `(asm, broken=(), *, duration=0.8, dt=0.001, solver='xpbd', iterations=16, frames=40, load…)` | Simula a montagem solta em repouso, com as juntas `broken` liberadas e as cargas aplicadas. |
| `harmonic_threshold` | função | `(n=4, length_mm=63.8, *, factors=…, duration=0.8, mu=0.4, tol_mm=1.0, solver='xpbd') -> l…)` | Testa no Newton a pilha harmônica com os deslocamentos escalados por fatores e compara com a fórmula fechada (fica de pé para f <= 1). |
| `cross_check` | função | `(goal, candidates, *, max_n=30, duration=0.4, tol_mm=2.0, dt=0.00025, iterations=32, seed…)` | Confere se o Newton concorda com o veredito da estática para uma amostra de candidatos. |
| `friction_creep` | função | `(asm, loads, *, dts=(0.001, 0.00025), duration=0.6, iterations=32) -> list[dict]` | Mede o deslocamento de uma montagem que deveria ficar parada sob carga lateral, para vários passos de tempo (artefato do atrito XPBD). |

#### `pinneapple_design.bricks.fea`

Elementos finitos lineares elásticos (Hex8 com modos incompatíveis, `solid_fem` de `pinneapple_simulation`) sobre a montagem voxelizada.

<small>`pinneapple_design/bricks/fea.py` · 127 linhas · testes: 24</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `VoxelModel` | classe | `(mesh, element_brick, element_volume)` | Malha hexaédrica da montagem com o bloco e o volume de cada elemento. |
| `FEAResult` | classe | `(fem, model, brick_displacement, brick_von_mises, max_displacement)` | Resultado da FEA: solução, modelo, deslocamento e tensão de von Mises por bloco, deslocamento máximo. |
| `voxel_mesh` | função | `(asm, per_stud=2) -> VoxelModel` | Gera a malha hexaédrica das células ocupadas, `per_stud` elementos por passo de pino. |
| `solve` | função | `(asm, loads=(), *, E=E_ABS_MPA, nu=NU_ABS, per_stud=2, gravity=True, supports='ground', f…)` | Resolve elasticidade estática linear da montagem sob gravidade e cargas pontuais (ABS por padrão). |

#### `pinneapple_design.bricks.goals`

Metas de projeto para o laço automático (balanço, ponte, torre): cada uma avalia uma montagem com a estática e explica por que falha. A classe base `Goal` levanta NotImplementedError nos métodos a sobrescrever.

<small>`pinneapple_design/bricks/goals.py` · 285 linhas · testes: 24 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Evaluation` | classe | `(score, feasible, violation, n_bricks, status, explanation, metrics, critical)` | Avaliação de uma montagem: pontuação, viabilidade, violação numa escala comum, status, explicação e métricas. |
| `Goal` | classe | `(joint=None)` | Classe base das metas: define chão, semente, cargas, avaliação e objetivos para o NSGA-II; `ground`, `seed` e `score_and_check` levantam NotImplementedError. |
| `CantileverGoal` | classe | `(max_bricks=10, joint=None)` | Meta de alcançar o mais longe possível além da borda de uma mesa com no máximo `max_bricks` peças, sob gravidade. |
| `BridgeGoal` | classe | `(gap=12, load_n=1.0, max_bricks=24, joint=None)` | Meta de vencer um vão entre duas mesas carregando `load_n` newtons no meio do tabuleiro, com o menor número de peças. |
| `TowerGoal` | classe | `(max_bricks=16, push_n=0.01, base=6, joint=None)` | Meta da torre mais alta numa base dada que resiste a um empurrão no topo, sobre mesa lisa (pode tombar ou quebrar). |
| `reference_cantilever` | função | `(n, length_mm=63.8) -> dict[str, float]` | Alcances de referência em fórmula fechada para `n` peças: bloco único, empilhamento harmônico e o limite do centro de massa. |

#### `pinneapple_design.bricks.ldraw`

Intercâmbio LDraw: escreve e lê arquivos .ldr/.mpd e carrega formas reais de peças da biblioteca LDraw (baixada sob demanda).

<small>`pinneapple_design/bricks/ldraw.py` · 281 linhas · testes: 24</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ReadReport` | classe | `(assembly, skipped)` | Resultado da leitura de um arquivo LDraw: a montagem e a lista de linhas ignoradas. |
| `PartsLibrary` | classe | `(root)` | Resolve e triangula arquivos .dat da biblioteca LDraw (sem impor o enrolamento BFC). |
| `to_ldr` | função | `(asm, name='model.ldr', *, steps=True, author='PINNeAPPle') -> str` | Texto LDraw da montagem, com um `0 STEP` por camada se `steps`. |
| `write_ldr` | função | `(asm, path, **kw) -> str` | Grava o texto de `to_ldr` num arquivo. |
| `write_mpd` | função | `(models, path) -> str` | Grava várias montagens num documento multi-peça (.mpd). |
| `from_ldr` | função | `(text, *, ground=None, tol=0.5) -> ReadReport` | Lê texto .ldr/.mpd: peças do catálogo alinhadas à grade viram blocos; o resto vai para `skipped`. |
| `read_ldr` | função | `(path, **kw) -> ReadReport` | Lê um arquivo .ldr/.mpd do disco com `from_ldr`. |
| `default_cache` | função | `() -> str` | Pasta padrão de cache da biblioteca de peças LDraw. |
| `fetch_parts_library` | função | `(cache_dir=None, urls=LIBRARY_URLS, timeout=120.0) -> str` | Baixa e descompacta a biblioteca LDraw uma vez; levanta RuntimeError se nenhuma URL responder. |

#### `pinneapple_design.bricks.mesh`

Malhas de triângulos de blocos e montagens e exportação para glTF (cor por bloco), STL e USD.

<small>`pinneapple_design/bricks/mesh.py` · 119 linhas · testes: 24</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `box_mesh` | função | `(lo, hi) -> tuple[np.ndarray, np.ndarray]` | Caixa fechada com normais para fora e quatro vértices por face. |
| `cylinder_mesh` | função | `(cx, cy, z0, r, h, n=16) -> tuple[np.ndarray, np.ndarray]` | Cilindro fechado com lateral suave e tampas planas. |
| `merge` | função | `(meshes) -> tuple[np.ndarray, np.ndarray]` | Junta várias malhas (vértices, faces) numa só. |
| `brick_mesh` | função | `(b, *, studs=True, segments=16, local=False) -> tuple[np.ndarray, np.ndarray]` | Malha de um bloco colocado, em mm, com ou sem pinos. |
| `assembly_meshes` | função | `(asm, **kw)` | Lista `(vértices_mm, faces, srgb)` de cada bloco da montagem. |
| `studio_scene` | função | `(asm, title='brick build', fields=None)` | Monta a montagem como uma `Scene` do estúdio de visualização (metros, z para cima), com um material por cor. |
| `write_gltf` | função | `(asm, path, title='brick build') -> str` | Grava um glTF binário (.glb) com um nó por bloco e sua cor. |
| `write_usd` | função | `(asm, path, title='brick build') -> str` | Grava a montagem em USD. |
| `write_stl` | função | `(asm, path, title='brick build') -> str` | Grava um STL binário da montagem inteira em mm. |

#### `pinneapple_design.bricks.parts`

Dimensões nominais dos blocos, catálogo de peças, modelo de massa, cores e o modelo de junta por pino (`JointModel`).

<small>`pinneapple_design/bricks/parts.py` · 215 linhas · testes: 24</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PartType` | classe | `(name, nx, ny, h, ldraw, mass, mass_source, smooth)` | Tipo de peça retangular: `nx` x `ny` pinos, `h` placas de altura (3 = bloco, 1 = placa), com massa e código LDraw. |
| `Colour` | classe | `(name, ldraw, rgb)` | Uma cor da paleta, com código LDraw e RGB. |
| `JointModel` | classe | `(clutch_n, clutch_band, mu, shear_n, edge_inset_mm, source, notes)` | Resistência de uma junta pino-tubo e atrito de contato usados pela estática; o padrão de `clutch_n` é 0,98 N com faixa assumida de 0,5 a 1,5 vez. |
| `fit_mass_model` | função | `() -> np.ndarray` | Ajusta por mínimos quadrados um modelo de casca `m = a·pinos + b·perímetro·altura` às massas publicadas. |
| `estimate_mass` | função | `(nx, ny, h) -> float` | Estima a massa de uma peça com o modelo de casca. |
| `mass_model_residuals` | função | `() -> dict[str, float]` | Erro relativo do modelo de casca em cada massa publicada. |
| `part` | função | `(name) -> PartType` | Busca uma peça do catálogo pelo nome ou pelo arquivo LDraw. |
| `smooth_block` | função | `(nx, ny, h, mass=None) -> PartType` | Cria um bloco liso, sem pinos (sem encaixe), para testes de empilhamento solto. |
| `colour` | função | `(name_or_code) -> Colour` | Busca uma cor pelo nome ou código. |

#### `pinneapple_design.bricks.report`

Resumos de um experimento com blocos: sensibilidade ao modelo de junta, gráfico de comparação de estratégias e relatório em Markdown.

<small>`pinneapple_design/bricks/report.py` · 191 linhas · testes: 24</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `sensitivity` | função | `(goal, asm, factors=None, shear_caps=(None, 1.0, 0.3)) -> list[dict]` | Reavalia uma montagem com o encaixe escalado na faixa assumida e com cisalhamento por pino limitado. |
| `plot_comparison` | função | `(summaries, path, *, ylabels=None, display=None) -> str` | Gráfico da melhor pontuação viável contra o número de avaliações, por meta e por estratégia (mediana entre sementes). |
| `write_report` | função | `(path, *, validation, comparisons, best, sensitivities, references=None, files=None, join…)` | Escreve o relatório Markdown: checagens fechadas, comparação de estratégias, melhores montagens, sensibilidade e limites. |

#### `pinneapple_design.bricks.search`

O laço que se modifica: muta uma montagem, simula cada candidato e guarda o que a física aceita (evolução NSGA-II ou busca guiada por GP).

<small>`pinneapple_design/bricks/search.py` · 371 linhas · testes: 24</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Candidate` | classe | `(asm, ev, edit, generation, parent, index)` | Um candidato avaliado: montagem, avaliação, edição aplicada, geração e pai. |
| `SearchResult` | classe | `(strategy, goal, seed, budget, history, generations, best_curve, wall_time_s, meta)` | Resultado de uma busca: histórico, gerações, curva do melhor, tempo e utilitários por geração. |
| `mutate` | função | `(goal, asm, rng, last=None, edit=None) -> tuple[Assembly, str]` | Aplica uma edição aleatória (ou a pedida) e devolve a nova montagem e o nome da edição. |
| `repair` | função | `(goal, asm) -> Assembly` | Remove peças sobrepostas ou flutuando para que a maioria das edições dê um candidato montável. |
| `evolve` | função | `(goal, budget=200, *, seed=0, mu=10, lam=10, progress=None) -> SearchResult` | Evolução (mu + lambda) com seleção NSGA-II e orçamento de avaliações físicas. |
| `surrogate_search` | função | `(goal, budget=200, *, seed=0, mu=10, lam=10, pool=8, kappa=1.0, warmup=20, progress=None)…)` | Mesmo laço, mas um surrogate GP ranqueia `pool * lam` filhos por média + kappa·desvio e só os melhores são simulados. |
| `compare` | função | `(goal_factory, *, budget=200, seeds=(0, 1, 2, 3, 4), strategies=('evolution', 'surrogate'…)` | Roda cada estratégia com cada semente no mesmo orçamento e resume mediana e faixa. |

#### `pinneapple_design.bricks.statics`

Equilíbrio estático de uma montagem por programação linear: forças de contato, utilização das juntas, falhas progressivas, tombamento e capacidade de carga.

<small>`pinneapple_design/bricks/statics.py` · 589 linhas · testes: 24</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Body` | classe | `(lo, hi, mass, com, label)` | Corpo rígido em caixa (limites, massa, centro de massa). |
| `Interface` | classe | `(lower, upper, z, rect, studs, label)` | Retalho de contato entre dois corpos (ou corpo e chão) numa altura, com pinos. |
| `Load` | classe | `(body, point, force, label)` | Carga pontual aplicada num corpo. |
| `Structure` | classe | `(bodies, interfaces, loads, gravity)` | Estrutura para a análise: corpos, interfaces, cargas e gravidade. |
| `StaticsResult` | classe | `(status, utilisation, interface_utilisation, interface_compression, interface_tension, bo…)` | Resultado da estática: status, utilização das juntas, forças, ponto crítico, explicação e tombamento. |
| `FailureRound` | classe | `(result, broken, new)` | Uma rodada da falha progressiva: resultado e juntas quebradas. |
| `from_assembly` | função | `(asm, loads=(), broken=()) -> Structure` | Converte uma `Assembly` em `Structure` (caixas pelas células). |
| `analyse` | função | `(structure, joint=None) -> StaticsResult` | Acha o equilíbrio sob gravidade e cargas com a menor utilização de pino máxima. |
| `capacity` | função | `(structure, joint=None, lam_max=1000000.0) -> float` | Maior fator sobre as cargas que a estrutura ainda carrega com todos os pinos dentro do limite. |
| `convex_hull` | função | `(points) -> np.ndarray` | Fecho convexo anti-horário de pontos 2D (cadeia monótona de Andrew). |
| `signed_distance_inside` | função | `(poly, p) -> float` | Distância de um ponto à borda de um polígono convexo (positiva dentro). |
| `centre_of_pressure` | função | `(s, members, z0) -> np.ndarray \| None` | Onde a resultante dos pesos e cargas de um grupo cruza um plano horizontal. |
| `tipping_report` | função | `(s, joint=None) -> dict \| None` | Pior polígono de apoio deixado pelo centro de pressão, tratando as juntas como inquebráveis. |
| `analyse_assembly` | função | `(asm, loads=(), joint=None, broken=()) -> StaticsResult` | Roda `analyse` numa montagem com gravidade, cargas e juntas liberadas. |
| `progressive_failure` | função | `(asm, loads=(), joint=None, max_rounds=50) -> list[FailureRound]` | Solta as juntas mais usadas, rodada a rodada, até a montagem ficar de pé ou virar mecanismo. |
| `top_load` | função | `(asm, force, *, at=None, label='load') -> Load` | Cria uma carga pontual na face de cima da peça mais alta em um ponto. |
| `single_stud_torsion` | função | `(asm, loads=(), joint=None, spread_mm=1.5) -> dict` | Verifica se um veredito depende de juntas de pino único sem rigidez à torção, resolvendo de novo com três pontos de contato. |

#### `pinneapple_design.bricks.validation`

Compara a física dos blocos com as fórmulas fechadas e junta os resultados num relatório de confiança.

<small>`pinneapple_design/bricks/validation.py` · 146 linhas · testes: 24</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `harmonic_limit_lp` | função | `(n=6, L=63.8, tol=0.0001, joint=None) -> float` | Maior fator dos deslocamentos harmônicos que a LP ainda acha estável (por bissecção). |
| `validate` | função | `(*, newton=None, fea=True, joint=None) -> dict` | Roda todas as checagens fechadas (estática, FEA e, se instalado, Newton). |

#### `pinneapple_design.bricks.visuals`

Imagens do laço: linha do tempo da evolução, visualizador web twin3d, GIF/MP4, renders no Blender e instruções de montagem.

<small>`pinneapple_design/bricks/visuals.py` · 508 linhas · testes: 24</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Item` | classe | `(brick, pos, quat, value)` | Um bloco num quadro: posição, orientação e valor de cor. |
| `Marker` | classe | `(size, pos, quat)` | Marcador (por exemplo, junta quebrada) num quadro. |
| `Frame` | classe | `(items, caption, short, kind, generation, markers, scenery)` | Um quadro da animação: itens, legenda, tipo, geração, marcadores e cenário. |
| `build_frame` | função | `(asm, values=None, **kw) -> Frame` | Cria um quadro da montagem em repouso, opcionalmente colorido pela utilização. |
| `scenery` | função | `(asms, thickness=80.0, margin=48.0) -> list[tuple[np.ndarray, np.ndarray]]` | Blocos de mesa sob as montagens para o cenário. |
| `utilisation` | função | `(asm, loads=(), joint=None) -> np.ndarray` | Pior utilização de pino de cada bloco (0 = sem tração, 1 = no limite). |
| `replay_frames` | função | `(asm, *, loads=(), joint=None, frames=14, duration=0.6, caption='', short='', generation=…)` | Falha progressiva seguida de replay Newton, um quadro por amostra, juntas quebradas em vermelho. |
| `evolution_timeline` | função | `(result, goal, *, generations=None, hold=3, replay=True, frames_per_replay=14, duration=0…)` | Quadros de uma `SearchResult`: o melhor de cada geração e, opcionalmente, um rejeitado desabando. |
| `frame_bounds` | função | `(frames, margin=6.0) -> tuple[np.ndarray, np.ndarray]` | Caixa que envolve todos os quadros. |
| `timeline_scene` | função | `(frames, title='self-modifying brick build', source='')` | Cena twin3d com um passo de tempo por quadro, para exportar como visualizador web. |
| `frame_geometry` | função | `(frame, *, colour_by='auto', ghost=None)` | Quadro como geometria para o renderizador de `geometry_library.render`. |
| `render_frames` | função | `(frames, *, size=(960, 540), azimuth=250.0, elevation=20.0, colour_by='auto', ssaa=1, lab…)` | Renderiza por software os quadros em RGB com uma câmera fixa. |
| `write_gif` | função | `(path, images, fps=8.0) -> str` | Grava as imagens como GIF. |
| `write_mp4` | função | `(path, images, fps=8.0) -> str` | Grava as imagens como MP4 H.264 com imageio-ffmpeg ou o programa `ffmpeg`. |
| `blender_available` | função | `() -> bool` | Diz se o módulo `bpy` do Blender está disponível. |
| `render_blender` | função | `(asm, out, *, values=None, view='iso', samples=64, size=(1280, 720), title=None, tables=N…)` | Render Cycles da montagem nas cores dos blocos ou colorida por utilização. |
| `turntable_blender` | função | `(asm, folder, *, frames=24, elevation=0.45, samples=32, size=(960, 540), gif=None, mp4=No…)` | Renders Cycles em volta da montagem, com GIF/MP4 opcionais. |
| `building_steps` | função | `(asm) -> list[list[int]]` | Índices dos blocos por passo de montagem, uma camada por passo. |
| `write_instructions` | função | `(asm, folder, *, title='Building instructions', size=(800, 520)) -> str` | Gera imagens por passo, lista de peças, arquivo LDraw e `instructions.md`. |

#### `pinneapple_design.design_optimizer`

Ponto de entrada do otimizador de projeto: reexporta o laço fechado (espaço de parâmetros, surrogate, otimizador, refinamento por PINN, Pareto) e a camada de estudos de projeto.

<small>`pinneapple_design/design_optimizer/__init__.py` · 116 linhas · testes: 36</small>

#### `pinneapple_design.design_optimizer.adjoint`

Otimização de forma por adjunto contínuo com autograd do PyTorch: parametrização de forma (FFD, Bézier, NACA), solver adjunto que calcula dJ/d(forma) e um objetivo de arrasto.

<small>`pinneapple_design/design_optimizer/adjoint.py` · 549 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ShapeParametrization` | classe | `(control_points, device='cpu') -> None` | Parametrização de forma para otimização por gradiente (pontos de controle FFD, curvas de Bézier ou família NACA), com deformação de malha. |
| `ContinuousAdjointSolver` | classe | `(primal_model, pde_residual_fn, objective_fn) -> None` | Solver adjunto por diferenciação automática: resolve a equação adjunta, calcula a sensibilidade de forma e roda a otimização. |
| `DragAdjointObjective` | classe | `(surface_pts, nu, alpha=0.0) -> None` | Objetivo de força de arrasto (pressão + viscoso) aproximado como média ponderada sobre pontos de superfície. |

#### `pinneapple_design.design_optimizer.airfoil_drag`

Estudo resolvido: reduz o arrasto de perfil de uma seção NACA 4 dígitos mantendo a sustentação, com MLP surrogate treinada na verdade de forma fechada (perfil fino + Hoerner) e o `DesignOptLoop`.

<small>`pinneapple_design/design_optimizer/airfoil_drag.py` · 175 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `section_aero` | função | `(m, t_c, p=CAMBER_POSITION, alpha_deg=ANGLE_OF_ATTACK_DEG, reynolds=REYNOLDS) -> dict[str…)` | (Cl, Cd) de um design pela verdade de forma fechada. |
| `airfoil_outline` | função | `(m, t_c, p=CAMBER_POSITION, n_pts=160) -> dict[str, list[float]]` | Pontos de superfície da seção em escala real. |
| `train_surrogate` | função | `(seed, n_train=N_TRAIN_SAMPLES, n_val=N_VAL_SAMPLES, epochs=SURROGATE_EPOCHS) -> dict[str…)` | Treina a MLP (m, t/c) -> (Cl, Cd) e valida num lote com outra semente. |
| `run_design_optimization` | função | `(n_iterations=150, seed=None, surrogate_epochs=SURROGATE_EPOCHS) -> dict[str, Any]` | Treina o surrogate, roda o laço fechado e devolve um resultado JSON com base, final, trajetória e erros. |

#### `pinneapple_design.design_optimizer.compare`

Compara métodos de otimização no mesmo problema, mesmo orçamento e mesmas sementes, usando hipervolume normalizado como métrica de convergência. [sem teste]

<small>`pinneapple_design/design_optimizer/compare.py` · 158 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MethodComparison` | classe | `(problem, results, ideal, nadir, budget, rows, curves)` | Resultado de uma comparação de métodos: tabela, união das avaliações, frentes, vencedor e exportação para dicionário. |
| `compare_methods` | função | `(problem, methods, *, budget=300, seeds=(0,), tol=0.01, options=None, progress=None) -> M…)` | Roda cada método uma vez por semente com o mesmo orçamento e tabula o resultado. |

#### `pinneapple_design.design_optimizer.constraints`

Restrições de projeto como penalidades diferenciáveis: caixa, conservação de massa (divergência), razão entre parâmetros, suavidade (variação total) e um conjunto que as agrega.

<small>`pinneapple_design/design_optimizer/constraints.py` · 254 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ConstraintBase` | classe | `()` | Base abstrata de restrição, com `penalty` e `satisfied`. |
| `BoxConstraint` | classe | `(theta_min, theta_max, weight=100.0) -> None` | Restrição de limites `theta_min <= theta <= theta_max` com barreira ReLU unilateral. |
| `MassConservationConstraint` | classe | `(weight=10.0, coords=None, n_neighbors=None) -> None` | Penaliza o quadrado médio da divergência estimada do campo de velocidade. |
| `GeometricConstraint` | classe | `(min_val=0.1, max_val=10.0, param_pairs=None, weight=50.0) -> None` | Restringe razões entre pares de parâmetros a um intervalo. |
| `ManufacturabilityConstraint` | classe | `(smoothness_weight=1.0) -> None` | Penaliza variação de alta frequência do vetor de projeto pela variação total. |
| `ConstraintSet` | classe | `(constraints)` | Agrega várias restrições: soma de penalidades e checagem de todas. |

#### `pinneapple_design.design_optimizer.evaluators`

Avaliadores que ligam modelos e solvers a um `DesignProblem`: modelo do `pp.zoo`, módulo torch diferenciável e comando externo via CSV. [sem teste]

<small>`pinneapple_design/design_optimizer/evaluators.py` · 168 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ZooEvaluator` | classe | `(name, space, fixed=None, derived=None)` | Usa um modelo do `pp.zoo` como avaliador (fidelidade surrogate, não diferenciável por este caminho); designs fora da faixa de validade ficam inválidos. |
| `TorchModuleEvaluator` | classe | `(module, outputs, *, input_transform=None, output_transform=None, name='torch-module', fi…)` | Usa um módulo torch como avaliador diferenciável, com transformações de entrada e saída opcionais. |
| `CommandEvaluator` | classe | `(command, space, outputs, *, timeout=600.0, name='external solver', fidelity='reference',…)` | Roda um solver externo por lote: o comando lê um CSV de designs e escreve um CSV de saídas, sem shell, em diretório temporário. |
| `from_model_ref` | função | `(ref, space, *, fixed=None, outputs=None) -> Evaluator` | Cria o avaliador certo a partir de um `ModelRef` ou dicionário com `uri` (zoo, TorchScript, etc.). |

#### `pinneapple_design.design_optimizer.methods`

Métodos de otimização sobre um `DesignProblem` sob o mesmo orçamento: random (hipercubo latino), NSGA-II próprio e do pymoo, SciPy DE e SLSQP, bayesiano (ParEGO) e gradiente por autograd. [sem teste]

<small>`pinneapple_design/design_optimizer/methods.py` · 501 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OptimizationResult` | classe | `(problem, method, seed, budget, evaluations, wall_s, status, message, notes)` | Todas as avaliações de uma rodada de um método, com custo, frente de Pareto, melhor design e curva de hipervolume. |
| `normalise_front` | função | `(F, ideal, nadir) -> np.ndarray` | Normaliza objetivos entre os pontos ideal e nadir. |
| `weight_vectors` | função | `(n_obj, n, seed=0) -> np.ndarray` | Gera `n` vetores de peso no simplex (espaçados para dois objetivos, Dirichlet com semente acima disso). |
| `available_methods` | função | `(problem=None) -> dict[str, dict[str, Any]]` | Lista cada método com sua família e se pode rodar aqui (pymoo instalado, avaliador diferenciável). |
| `run` | função | `(problem, method='nsga2', *, budget=400, seed=0, initial=None, progress=None, **options) …)` | Roda um método no problema com no máximo `budget` avaliações; métodos impossíveis voltam com status skipped. |

#### `pinneapple_design.design_optimizer.objective`

Objetivos de projeto sobre um campo previsto: proxies de arrasto, eficiência térmica, tensão e peso, e soma ponderada.

<small>`pinneapple_design/design_optimizer/objective.py` · 138 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ObjectiveBase` | classe | `()` | Classe base abstrata para objetivos de projeto. |
| `DragObjective` | classe | `(pressure_weight=1.0, viscous_weight=0.1) -> None` | Proxy de arrasto: média de \|u\|^2 + peso x p^2 sobre o campo (sem normais de superfície). |
| `ThermalEfficiencyObjective` | classe | `(heat_field_idx=0, pressure_drop_weight=0.5) -> None` | Eficiência térmica (fluxo de calor sobre queda de pressão proxy), devolvida negativa para minimizar. |
| `StructuralObjective` | classe | `()` | Proxy de tensão de von Mises pela média dos quadrados do campo. |
| `WeightMinimizationObjective` | classe | `()` | Proxy de volume pela norma L2 do vetor de parâmetros. |
| `CompositeObjective` | classe | `(terms=None) -> None` | Soma ponderada de vários objetivos. |

#### `pinneapple_design.design_optimizer.optimizer`

Espaço de parâmetros em caixa e os três otimizadores do laço: gradiente (Adam/L-BFGS), bayesiano com processo gaussiano e evolutivo (CMA-ES ou AG de reserva).

<small>`pinneapple_design/design_optimizer/optimizer.py` · 666 linhas · testes: 179</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ParamSpace` | classe | `(bounds, x0)` | Espaço de parâmetros limitado por caixa, com recorte aos limites. |
| `OptState` | classe | `(step, best_x, best_y, last_x, last_y)` | Retrato do estado do otimizador em uma iteração. |
| `DesignOptimizerConfig` | classe | `(method, n_iters, lr, grad_optimizer, grad_clip, n_initial_random, acquisition, xi, kappa…)` | Configuração única dos back-ends (método gradient/bayesian/evolutionary, iterações, taxa de aprendizado, etc.). |
| `GradientDesignOptimizer` | classe | `(theta_init, cfg) -> None` | Otimizador por gradiente que aplica um passo Adam ou L-BFGS por chamada de `step`. |
| `BayesianDesignOptimizer` | classe | `(cfg, seed=0) -> None` | Otimizador bayesiano sequencial com processo gaussiano (scikit-learn ou GP NumPy de reserva). |
| `EvolutionaryDesignOptimizer` | classe | `(cfg, seed=0, multi_objective=False) -> None` | Otimizador populacional: CMA-ES quando `cma` está instalado, senão um AG (mu+lambda) com torneio, cruzamento e mutação. |

#### `pinneapple_design.design_optimizer.pareto`

Utilidades multiobjetivo: dominância, máscara de Pareto, ordenação não dominada e distância de aglomeração do NSGA-II, representantes da frente e contêiner `ParetoFront`.

<small>`pinneapple_design/design_optimizer/pareto.py` · 341 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ParetoFront` | classe | `(objectives, params, mask)` | Contêiner de uma frente de Pareto: objetivos, parâmetros e máscara, com filtro, hipervolume e gráfico. |
| `pareto_dominates` | função | `(a, b) -> bool` | True se a solução `a` domina `b` (minimização). |
| `compute_pareto_front` | função | `(objectives) -> np.ndarray` | Máscara booleana das soluções Pareto-ótimas. |
| `fast_non_dominated_sort` | função | `(objectives) -> list[np.ndarray]` | Divide as soluções em frentes sucessivas (NSGA-II). |
| `crowding_distance` | função | `(objectives) -> np.ndarray` | Distância de aglomeração do NSGA-II dentro de uma frente. |
| `representatives` | função | `(objectives, senses, names, ids=None, balanced='balanced', weights=None) -> dict[str, Any]` | Designs representativos de uma frente: o melhor em cada objetivo e um compromisso equilibrado. |

#### `pinneapple_design.design_optimizer.pipeline`

Laço fechado de otimização guiada por física (`DesignOptLoop`): avalia o surrogate, otimiza com o back-end escolhido, refina por PINN e monta a frente de Pareto.

<small>`pinneapple_design/design_optimizer/pipeline.py` · 502 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DesignOptConfig` | classe | `(n_iterations, optimizer_cfg, refine_top_k, save_history, verbose, convergence_tol, seed,…)` | Configuração do `DesignOptLoop`: iterações, configuração do otimizador interno, refinamento top-k e outros. |
| `DesignOptResult` | classe | `(best_theta, best_objective, history_theta, history_objectives, history_all_objectives, p…)` | Resultado completo de um `DesignOptLoop`: melhor design, histórico, frente de Pareto, com gráfico de convergência e resumo. |
| `DesignOptLoop` | classe | `(param_space, surrogate, objective, *, constraints=None, refinement=None, cfg=None) -> No…)` | Orquestra o laço fechado: parametriza, avalia com o surrogate, otimiza, refina por PINN e monta a frente de Pareto. |

#### `pinneapple_design.design_optimizer.presets`

Problemas prontos: dissipador de aletas planas (fidelidades surrogate, forma fechada e FVM) e perfil CST (ensemble do zoo), com bases de comparação e geração de corpos/arquivos STL e STEP.

<small>`pinneapple_design/design_optimizer/presets.py` · 466 linhas · testes: 31 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HeatsinkOperating` | classe | `(power_w, t_ambient_c, air_velocity_m_s, source_width_mm, source_depth_mm, tim_k_mm2_w, m…)` | Ponto de operação do dissipador: potência, temperatura ambiente, velocidade do ar, tamanho da fonte, resistência de interface. |
| `HeatsinkLimits` | classe | `(t_source_max_c, pressure_drop_max_pa, fin_stress_max_mpa, fin_gap_min_mm, fin_aspect_max)` | Limites do dissipador: temperatura da fonte, queda de pressão, tensão na aleta e limites de extrusão. |
| `CostModel` | classe | `(material_per_kg, material_yield, per_fin, fixed, currency)` | Modelo de custo unitário com valores assumidos (preço do material x massa / rendimento + custo por aleta + fixo). |
| `HeatsinkEvaluator` | classe | `(fidelity='surrogate', operating=None, cost=None)` | Avaliador do dissipador: geometria em forma fechada mais o modelo térmico/hidráulico da fidelidade escolhida. |
| `AirfoilEvaluator` | classe | `()` | Avaliador do perfil: ensemble `airfoil_cst_rans_mlp` (média e dispersão de Cl, Cd, Cm) mais espessuras em forma fechada. |
| `heatsink_problem` | função | `(fidelity='surrogate', operating=None, limits=None, cost=None, variables=None, objectives…)` | Monta o problema de dimensionamento do dissipador de aletas planas na fidelidade pedida. |
| `heatsink_baseline` | função | `() -> dict[str, float]` | Design convencional de partida do dissipador para comparação. |
| `airfoil_problem` | função | `(cl_min=0.6, cm_min=-0.12, aft_min=0.012, alpha_range=(0.0, 8.0)) -> DesignProblem` | Problema do perfil CST: minimizar Cd e maximizar altura da longarina com restrições de Cl, Cm e espessura traseira. |
| `airfoil_baseline` | função | `() -> dict[str, float]` | Design de referência do perfil: NACA 2412 (ajuste CST) a 4 graus. |
| `preset_info` | função | `(name) -> dict[str, Any]` | O que um formulário precisa para montar um preset: variáveis, padrões, objetivos possíveis, base e referência. |
| `plate_fin_bodies` | função | `(design) -> list[tuple[str, np.ndarray, np.ndarray]]` | Malhas fechadas (metros) da placa base e de cada aleta do dissipador. |
| `airfoil_bodies` | função | `(design, span=0.2, chord=1.0, n=81) -> list[tuple[str, np.ndarray, np.ndarray]]` | Segmento de asa extrudado e fechado da seção CST. |
| `preset_bodies` | função | `(preset, design) -> list[tuple[str, np.ndarray, np.ndarray]]` | Corpos sólidos de um design de preset (heatsink ou airfoil). |
| `design_file` | função | `(bodies, fmt, name='design') -> bytes` | Bytes de STL ou STEP do design, escritos pelos exportadores da biblioteca de geometrias. |

#### `pinneapple_design.design_optimizer.refinement`

Refina previsões do surrogate continuando o treino de uma PINN com perda de dados + resíduo de EDP.

<small>`pinneapple_design/design_optimizer/refinement.py` · 225 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RefinementConfig` | classe | `(n_epochs, lr, pde_weight, bc_weight, data_weight, n_collocation, verbose)` | Configuração do refinamento por PINN: épocas, taxa de aprendizado e pesos das perdas. |
| `RefinementResult` | classe | `(theta, u_refined, loss_history, improvement_ratio)` | Resultado de um refinamento: parâmetros, saída refinada, histórico de perda e razão de melhora. |
| `PINNRefinement` | classe | `(pinn_model, pde_residual_fn, domain=None, cfg=None) -> None` | Refina a previsão do surrogate treinando uma PINN com perda de dados + resíduo de EDP, para um ou para os k melhores designs. |

#### `pinneapple_design.design_optimizer.report`

Leva estudos de projeto para dentro e fora dos contratos de dados compartilhados (`ParameterSpace`, `StudyResult`, `ModelRef`) e treina surrogates GP a partir de tabelas do usuário. [sem teste]

<small>`pinneapple_design/design_optimizer/report.py` · 344 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SurrogateFit` | classe | `(space, gp, outputs, metrics, n_train, n_test)` | Surrogate GP treinado numa tabela, com seu erro em dados separados. |
| `to_parameter_space` | função | `(space, name, description='', budget=None)` | Converte um espaço de projeto em `pinneapple_data.contracts.ParameterSpace`. |
| `designs_frame` | função | `(problem, ev, idx=None, status=None)` | Tabela pandas com um design por linha: variáveis, saídas, viabilidade, fidelidade e status de verificação. |
| `study_result` | função | `(problem, result, *, study_id, app, title='', candidates=(), verification=None, compariso…)` | Empacota uma otimização ou comparação de métodos como `StudyResult`, registrando só as checagens que rodaram. |
| `surrogate_from_table` | função | `(df, inputs, outputs, *, test_fraction=0.2, seed=0, integer=(), bounds=None) -> Surrogate…)` | Ajusta um `GPSurrogate` por saída a partir das linhas de uma tabela e mede o erro em linhas separadas. |
| `model_ref_for_surrogate` | função | `(fit, name, uri, dataset_name='')` | `ModelRef` de um surrogate treinado em tabela, com faixa de validade e métricas. |
| `table_evaluator` | função | `(space, df, outputs, *, tol=1e-06, name='your solver results') -> FunctionEvaluator` | Avaliador de referência que responde a partir de uma tabela de resultados de solver, casando linhas pelos valores das variáveis. |
| `verify_front` | função | `(problem, ev, front, n, reference=None)` | Escolhe candidatos ao longo da frente e os reavalia com a referência, quando existe. |
| `write_design_files` | função | `(folder, problem, ev, candidates, status=None, *, labels=None) -> list[str]` | Grava os arquivos do estudo numa pasta: avaliações, candidatos com status, variáveis para rodar no solver e o `ParameterSpace`. |
| `chart_points` | função | `(problem, ev, front=(), *, limit=2000, labels=None) -> list[dict[str, Any]]` | Seleciona e afina pontos para um gráfico de trade-off, mantendo sempre a frente. |
| `verify_uploaded_results` | função | `(space, candidates, results, objectives=(), constraints=(), *, tol=1e-06)` | Verifica candidatos exportados com resultados de solver rodados fora, casando linhas por valores das variáveis. |

#### `pinneapple_design.design_optimizer.spec`

Monta problemas de projeto a partir de specs JSON simples (fonte preset, zoo ou tabela), como os enviados por um app ou API. [sem teste]

<small>`pinneapple_design/design_optimizer/spec.py` · 205 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SpecError` | classe | `()` | Erro de spec que não vira problema, com mensagem legível. |
| `objectives_from` | função | `(items) -> list[Objective]` | Converte a lista de objetivos da spec em `Objective`; exige ao menos um. |
| `constraints_from` | função | `(items) -> list[Constraint]` | Converte a lista de restrições da spec em `Constraint` com limites inferior/superior. |
| `zoo_problem` | função | `(model, objectives, constraints=(), *, variables=None, fixed=None) -> DesignProblem` | Problema sobre um modelo do `pp.zoo`: cada entrada não fixa vira variável limitada pela faixa de validade. |
| `problem_from_spec` | função | `(spec, table=None) -> tuple[DesignProblem, Any]` | Monta `(problem, fit)` a partir de uma spec; `fit` só existe para fonte tabela. |
| `reference_for_spec` | função | `(spec) -> Evaluator \| None` | Referência de alta fidelidade da spec: o motor FVM para o dissipador, None nos outros casos. |

#### `pinneapple_design.design_optimizer.study`

Descrição do problema de projeto usada por todos os métodos: variáveis, espaço, objetivos, restrições, avaliador, registro das avaliações, relatório de viabilidade e hipervolume.

<small>`pinneapple_design/design_optimizer/study.py` · 494 linhas · testes: 49 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Variable` | classe | `(name, lower, upper, kind, unit, description)` | Uma variável de projeto em `[lower, upper]`; `kind="integer"` arredonda. |
| `DesignSpace` | classe | `(variables)` | Lista ordenada de variáveis: limites, reparo, mapeamento ao cubo unitário, amostragem e conversão de/para `ParameterSpace`. |
| `Objective` | classe | `(name, sense, unit, label)` | Minimizar ou maximizar uma saída nomeada do avaliador. |
| `Constraint` | classe | `(name, upper, lower, unit, kind, label)` | Restrição `lower <= saída <= upper` com violação normalizada, margem e tipo (física, fabricação, validade). |
| `Evaluator` | classe | `(name, fidelity, outputs, differentiable, relaxable, description)` | Base de avaliador: mapeia designs X (n, d) em saídas nomeadas, com fidelidade e flag de diferenciável; `evaluate` é abstrato. |
| `FunctionEvaluator` | classe | `(fn, outputs, *, name='function', fidelity='model', torch_fn=None, description='', relaxa…)` | Avaliador a partir de uma função vetorizada (e, opcionalmente, de sua versão torch). |
| `Evaluations` | classe | `(X, Y, F, G, feasible, relaxed, fidelity, wall_s, tag)` | Todos os designs avaliados num estudo, em ordem: matriz de objetivos, violações, viabilidade e frente de Pareto. |
| `DesignProblem` | classe | `(name, space, objectives, constraints, evaluator, description, notes)` | Problema de otimização: variáveis, objetivos, restrições e avaliador, com registro das avaliações. |
| `constants_of` | função | `(ps) -> dict[str, Any]` | Parâmetros constantes de um `ParameterSpace` ou dicionário. |
| `feasibility_report` | função | `(problem, ev, idx, active_tol=0.05) -> list[dict[str, Any]]` | Por design: valor, margem e status (violada, ativa, inativa) de cada restrição. |
| `hypervolume` | função | `(F, ref, n_mc=20000, seed=0) -> float` | Hipervolume dominado pelas linhas de F até `ref`: exato para 1 e 2 objetivos, Monte Carlo acima. |

#### `pinneapple_design.design_optimizer.surrogate`

Envoltório `PhysicsSurrogate` que dá uma interface uniforme (previsão, jacobiano, treino, salvar/carregar) a qualquer `nn.Module` que mapeia parâmetros em campo.

<small>`pinneapple_design/design_optimizer/surrogate.py` · 268 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SurrogateConfig` | classe | `(model_name, in_dim, out_channels, hidden_dim, n_layers, device, checkpoint_path)` | Configuração do `PhysicsSurrogate`: modelo (nome registrado ou MLP), dimensões e camadas. |
| `PhysicsSurrogate` | classe | `(model, cfg=None) -> None` | Envoltório fino de um `nn.Module` theta -> u com previsão, jacobiano, treino em dataset e checkpoint. |

#### `pinneapple_design.design_optimizer.verify`

Confirmação em alta fidelidade dos designs otimizados e otimização assistida por surrogate GP com aprendizado ativo. [sem teste]

<small>`pinneapple_design/design_optimizer/verify.py` · 254 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `VerificationReport` | classe | `(rows, errors, reference, wall_s, evaluations)` | Relatório de verificação: linhas por design, erros por saída, dados da referência e tempo gasto. |
| `GPSurrogate` | classe | `(problem, outputs, validity=None, name='gp-surrogate', safe=None, kappa=2.0)` | Processo gaussiano por saída (Matern 5/2, ARD, scikit-learn) em entradas no cubo unitário; saídas positivas em escala log. |
| `ActiveLearningResult` | classe | `(problem, reference_evals, rounds, final, verification, wall_s, notes)` | Resultado da otimização assistida: avaliações de referência, rodadas, resultado final e verificação. |
| `select_candidates` | função | `(ev, front, n) -> list[int]` | Escolhe `n` designs espaçados ao longo de uma frente, com as duas pontas. |
| `verify_candidates` | função | `(problem, ev, idx, reference, reference_problem=None) -> VerificationReport` | Reavalia designs escolhidos com a referência e compara cada saída (erro relativo, viabilidade, status). |
| `surrogate_assisted` | função | `(problem, reference, *, outputs=None, method='nsga2', n_initial=24, rounds=3, n_verify=6,…)` | Otimiza num GP da referência e o melhora onde o otimizador procura (aprendizado ativo). |

#### `pinneapple_design.fitmesh`

Reexporta a sessão e as etapas do fluxo fitmesh: leitura do problema, catálogo, sugestões, CAD, plano e geração de malha, relatório.

<small>`pinneapple_design/fitmesh/__init__.py` · 26 linhas · testes: 21</small>

#### `pinneapple_design.fitmesh.api`

API HTTP (FastAPI) sobre `FitMeshSession`: uma sessão por estudo, uma chamada por etapa, mais arquivos e relatório.

<small>`pinneapple_design/fitmesh/api.py` · 146 linhas · testes: 12</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `create_app` | função | `(root='fitmesh_runs')` | Cria o app FastAPI das sessões fitmesh numa pasta raiz. |

#### `pinneapple_design.fitmesh.cad`

Constrói as variantes em CadQuery, exporta STEP e STL e verifica cada uma (limites, BRep válido, volume exato, STL fechado, caixa, comparação com referências publicadas). [sem teste direto pelo inventário]

<small>`pinneapple_design/fitmesh/cad.py` · 115 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CadVariant` | classe | `(index, params, step, stl, checks, comparison, volume, area, bbox)` | Uma variante CAD construída: parâmetros, arquivos STEP/STL, checagens, comparação, volume, área e caixa, com `verdict`. |
| `build_variant` | função | `(tpl, params, out_dir, index, accuracy='design') -> CadVariant` | Constrói uma variante a partir do modelo e dos parâmetros, exporta e roda as checagens. |
| `compare_with_references` | função | `(tpl, params) -> list[dict[str, Any]]` | Diferença relativa de cada dimensão compartilhada com cada objeto de referência publicado. |

#### `pinneapple_design.fitmesh.catalog`

Catálogo de modelos de geometria paramétricos (asa NACA, corpo de Ahmed, cilindro, esfera, curva de tubo, degrau, placa com furo, viga, dissipador) com parâmetros, limites e referências publicadas. [sem teste direto pelo inventário]

<small>`pinneapple_design/fitmesh/catalog.py` · 460 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Param` | classe | `(name, value, unit, typical, limits, role, sensitivity)` | Parâmetro de modelo com valor de referência, unidade, faixa típica, limites, papel e sensibilidade. |
| `Reference` | classe | `(citation, what, params)` | Objeto de referência publicado: citação, o que foi medido e suas dimensões. |
| `Template` | classe | `(key, title, physics, keywords, description, params, references, quantities, build, volum…)` | Modelo de geometria paramétrico com física, palavras-chave, parâmetros, referências, construtor CadQuery, volume exato e comprimentos característicos. |
| `naca4_points` | função | `(m, p, t, chord, n=80) -> np.ndarray` | Pontos de uma seção NACA 4 dígitos fechada com espaçamento cosseno. |
| `polygon_area` | função | `(pts) -> float` | Área de um polígono 2D. |

#### `pinneapple_design.fitmesh.context`

Lê uma descrição livre (inglês ou português) em fatos com fonte (física, fluido, velocidade, tamanho, propósito) e gera as perguntas do que falta; calcula a altura da primeira célula para um y+.

<small>`pinneapple_design/fitmesh/context.py` · 208 linhas · testes: 12</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Fact` | classe | `(value, source)` | Um fato do problema com valor e fonte (texto, resposta, padrão). |
| `ProblemContext` | classe | `(text, facts)` | Fatos lidos do problema, com física, precisão, propriedades do fluido, Reynolds e perguntas pendentes. |
| `parse` | função | `(text, llm=None) -> ProblemContext` | Extrai fatos de uma descrição livre em inglês ou português, opcionalmente com um LLM. |
| `wall_unit_height` | função | `(y_plus, speed, length, nu, re, internal=False) -> dict[str, float]` | Altura da primeira célula para um y+ alvo pelo atrito de placa plana. |

#### `pinneapple_design.fitmesh.layers`

Dimensiona camadas prismáticas como pilha geométrica, resolvendo a quantidade que falta. [sem teste direto pelo inventário]

<small>`pinneapple_design/fitmesh/layers.py` · 81 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `layer_total` | função | `(first_height, expansion, n_layers) -> float` | Espessura total de n camadas em progressão geométrica. |
| `size_layers` | função | `(first_height, expansion=None, n_layers=None, total_thickness=None) -> dict[str, float]` | Resolve a pilha de camadas para a quantidade que falta (razão, número ou espessura total). |

#### `pinneapple_design.fitmesh.llm`

Ajuda opcional de modelo de linguagem (Ollama local, Anthropic ou OpenAI) em cada etapa; tudo que volta é validado, e código CadQuery livre roda em processo isolado com lista de imports e timeout.

<small>`pinneapple_design/fitmesh/llm.py` · 251 linhas · testes: 12</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LLMAssist` | classe | `(provider, model, api_key, conversation_store, call, provider_kwargs)` | Escolha do modelo de linguagem (provedor, modelo, chave ou função `call` própria). |
| `read_problem` | função | `(llm, text) -> dict[str, Any]` | Pede ao LLM os fatos de um texto livre. |
| `rank_geometries` | função | `(llm, ctx) -> dict[str, Any]` | Pede ao LLM para escolher modelos do catálogo real (chaves desconhecidas são descartadas). |
| `propose_parameters` | função | `(llm, tpl, ctx, base) -> list[dict[str, Any]]` | Pede ao LLM parâmetros e faixas, cortados nos limites do modelo. |
| `free_geometry` | função | `(llm, description) -> dict[str, Any]` | Pede ao LLM um script CadQuery quando nenhum modelo serve, checado estaticamente. |
| `check_code` | função | `(code) -> list[str]` | Checagens estáticas do código: sintaxe, imports permitidos, sem dunder ou E/S, presença de `build()`. |
| `run_free_geometry` | função | `(code, params, out_dir, tol, timeout=120) -> dict[str, Any]` | Roda o CadQuery escrito pelo modelo num interpretador isolado (`python -I`) com timeout. |
| `review_mesh_plan` | função | `(llm, plan, ctx) -> MeshPlan` | Pede ao LLM sugestões sobre o plano de malha; só alguns campos podem mudar. |

#### `pinneapple_design.fitmesh.mesh_gen`

Gera as malhas planejadas com Gmsh (sólidos e seções 2D) ou OpenFOAM blockMesh+snappyHexMesh (escoamento 3D) e mede a qualidade via `pinneapple_data.cae`. [sem teste direto pelo inventário]

<small>`pinneapple_design/fitmesh/mesh_gen.py` · 407 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MeshResult` | classe | `(variant, level, tool, path, files, cells, seconds, quality, layers, error)` | Resultado de uma malha: ferramenta, arquivos, células, tempo, qualidade, camadas e erro, com `verdict`. |
| `generate` | função | `(tpl, var, plan, out_dir, level=1.0, procs=1, max_cells=3000000.0, log=print) -> MeshResu…)` | Gera a malha de uma variante num nível de refinamento do plano. |
| `quality` | função | `(path, target) -> dict[str, Any]` | Mede a qualidade de uma malha contra o alvo. |

#### `pinneapple_design.fitmesh.mesh_plan`

Recomenda uma malha adequada ao propósito (triagem, projeto, verificação): método, tamanhos, camadas por y+, domínio, estimativa de células e custo e família para GCI. [sem teste direto pelo inventário]

<small>`pinneapple_design/fitmesh/mesh_plan.py` · 264 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MeshPlan` | classe | `(kind, tool, elements, purpose, sizes, layers, domain, estimated_cells, estimated_cost, f…)` | Plano de malha: tipo, ferramenta, elementos, tamanhos, camadas, domínio, estimativas, família e justificativa. |
| `mesh_kind` | função | `(tpl, ctx) -> str` | Escolhe o tipo de malha (sólido, escoamento 3D, seção 2D) para o modelo e o contexto. |
| `plan_mesh` | função | `(tpl, params, ctx, budget_cells=5000000.0) -> MeshPlan` | Monta o plano de malha adequado ao propósito dentro do orçamento de células. |

#### `pinneapple_design.fitmesh.report`

Calcula o índice de convergência de malha (GCI) e monta o relatório final JSON+HTML com veredito por etapa. [sem teste direto pelo inventário]

<small>`pinneapple_design/fitmesh/report.py` · 226 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `gci` | função | `(values, sizes, safety=None) -> dict[str, Any]` | Índice de convergência de malha a partir de três malhas (Roache; Celik et al. 2008). |
| `build_report` | função | `(session) -> dict[str, Any]` | Reúne todas as etapas de uma sessão num relatório com veredito por etapa. |
| `render_html` | função | `(rep) -> str` | Renderiza o relatório em HTML. |
| `write_report` | função | `(rep, stem) -> dict[str, str]` | Grava o relatório em JSON e HTML. |

#### `pinneapple_design.fitmesh.session`

`FitMeshSession`: o estudo interativo passo a passo, da descrição ao relatório. [sem teste direto pelo inventário]

<small>`pinneapple_design/fitmesh/session.py` · 295 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FitMeshSession` | classe | `(text, out_dir, llm=None, log=print)` | Estudo interativo: perguntas, escolha de geometria e parâmetros, CAD, plano, malhas, resultados e relatório. |

#### `pinneapple_design.fitmesh.suggest`

Ranqueia modelos de geometria para o problema e propõe parâmetros a variar, faixas e amostras por hipercubo latino.

<small>`pinneapple_design/fitmesh/suggest.py` · 94 linhas · testes: 12</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `suggest_geometries` | função | `(ctx, top=3) -> list[dict[str, Any]]` | Modelos ranqueados por física e palavras-chave, com o motivo e as referências. |
| `scale_to_context` | função | `(tpl, ctx) -> dict[str, float]` | Valores de referência escalados para o tamanho dado no problema. |
| `suggest_parameters` | função | `(tpl, ctx, base=None) -> list[dict[str, Any]]` | Cada parâmetro com recomendação de variar ou manter, faixa proposta e motivo. |
| `sample` | função | `(tpl, base, ranges, n, seed=0, include_base=True) -> list[dict[str, float]]` | Amostras por hipercubo latino nas faixas escolhidas, com a base primeiro. |

#### `pinneapple_design.geometry`

Subpacote de geometria (antigo `pinneapple_geom`): reexporta `GeometrySpec`, `GeometryAsset`, `MeshData`, SDFs, CSG, domínios físicos 2D/3D, malhadores, RBF e `MeshCollocator`; o construtor STL é importado de forma tolerante (vira `None` se faltar dependência).

<small>`pinneapple_design/geometry/__init__.py` · 228 linhas · testes: 1</small>

#### `pinneapple_design.geometry.bodies`

Corpos fechados triangulados para escoamento externo: corpo de Ahmed, esfera, cilindro, caixa e anel.

<small>`pinneapple_design/geometry/bodies.py` · 141 linhas · testes: 43</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ahmed_body` | função | `(slant_deg=25.0, scale=1.0, clearance=0.05, n_nose=10) -> Mesh` | Gera o corpo de Ahmed com a traseira inclinada em `slant_deg`, a uma folga acima do chão. |
| `sphere` | função | `(radius=0.5, centre=(0.0, 0.0, 0.0), n=48) -> Mesh` | Esfera UV. |
| `cylinder` | função | `(radius=0.5, length=2.0, axis='y', n=48) -> Mesh` | Cilindro fechado centrado na origem, no eixo pedido. |
| `box` | função | `(size=(1.0, 1.0, 1.0), centre=(0.0, 0.0, 0.0)) -> Mesh` | Caixa alinhada aos eixos. |
| `annulus` | função | `(r_inner, r_outer, height, n=72, z0=0.0) -> Mesh` | Anel fechado (tubo com tampas planas) em torno do eixo z. |

#### `pinneapple_design.geometry.builders`

Reexporta o construtor de lotes a partir de STL e os construtores analíticos. [sem teste]

<small>`pinneapple_design/geometry/builders/__init__.py` · 15 linhas · **sem teste direto**</small>

#### `pinneapple_design.geometry.builders.analytic_domain_batch_builder`

Gera lotes de treino (colocação e contorno com tags) sem malha, para domínios que são caixas, cilindros, anéis ou caixas com curvas embutidas, sem recentralizar as coordenadas. [sem teste]

<small>`pinneapple_design/geometry/builders/analytic_domain_batch_builder.py` · 709 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `sample_box_tag_batch` | função | `(spec, tag_faces, *, n_col=20000, n_bc_per_face=3000, seed=7, user_ctx=None) -> Dict[str,…)` | Amostra um lote numa caixa (de qualquer dimensão) descrita por `spec.domain_bounds`, com pontos de contorno por face com tag. |
| `sample_cylinder_tag_batch` | função | `(spec, *, axis_coord, cross_coords, radius, tag_faces, n_col=20000, n_bc_per_face=3000, s…)` | Amostra um lote num cilindro de seção circular extrudado ao longo de um eixo (ex.: `pipe_flow_3d`). |
| `sample_axisymmetric_wall_tag_batch` | função | `(spec, *, axis_coord, radial_coord, r_wall_fn, dr_wall_fn, tag_faces, n_col=20000, n_bc_p…)` | Amostra um domínio axissimétrico (r, z) cujo raio de parede varia com z (ex.: bocal cônico). |
| `sample_annulus_tag_batch` | função | `(spec, tag_faces, *, cross_coords, inner_radius, outer_radius, n_col=20000, n_bc_per_face…)` | Amostra uma seção anular, com tags para o círculo interno e o externo. |
| `sample_box_with_curve_tag_batch` | função | `(spec, *, shape_coords, box_tag_faces=None, curve_tags=None, inside_body_fn=None, n_col=2…)` | Amostra uma caixa 2D com corpos ou curvas embutidos (ex.: perfil de aerofólio) que não são faces da caixa. |

#### `pinneapple_design.geometry.builders.stl_domain_batch_builder`

Constrói lotes de treino PINN (colocação, contorno, inicial, dados) a partir de um STL via trimesh, com teste de dentro/fora configurável e tags heurísticas de entrada/saída/parede; recentraliza a malha. [sem teste]

<small>`pinneapple_design/geometry/builders/stl_domain_batch_builder.py` · 463 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `VoxelInsideConfig` | classe | `(pitch, target_resolution, fill)` | Configuração do teste de dentro/fora por voxelização. |
| `SDFConfig` | classe | `(mode, sdf_fn, inside_if_leq)` | Configuração do teste de dentro/fora por SDF (proximidade da malha ou função dada). |
| `TagHeuristics` | classe | `(enabled, inlet_outlet_axis, inlet_is_min, outlet_is_min, plane_eps_frac, require_normal_…)` | Regras heurísticas para marcar pontos de contorno como entrada, saída, parede ou fixo. |
| `STLDomainBatchConfig` | classe | `(n_col, n_bc, n_ic, n_data, normalize_unit_box, inside_mode, max_rejection_rounds, oversa…)` | Configuração completa do lote STL: quantidades de pontos, modo de dentro/fora, dispositivo e tags. |
| `STLDomainBatchBuilder` | classe | `(cfg)` | Carrega o STL, normaliza e amostra o lote de treino com `build`. |

#### `pinneapple_design.geometry.core`

Reexporta `GeometrySpec`, `GeometryAsset`, `MeshData` e as fábricas de ativos.

<small>`pinneapple_design/geometry/core/__init__.py` · 12 linhas · **sem teste direto**</small>

#### `pinneapple_design.geometry.core.geometry`

Define a especificação declarativa de geometria e o contêiner unificado de ativo geométrico com transformações e grupos de contorno. [sem teste direto]

<small>`pinneapple_design/geometry/core/geometry.py` · 422 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GeometrySpec` | classe | `(kind, name, path, params)` | Especificação declarativa (tipo, nome, caminho, parâmetros) com validação e impressão digital. |
| `GeometryAsset` | classe | `(mesh, bounds, units, boundary_groups, meta)` | Contêiner de geometria com malha, limites, unidades, grupos de contorno e transformações. |

#### `pinneapple_design.geometry.core.mesh`

Contêiner leve de malha de triângulos (só numpy) com transformações, normais, áreas e apoio a amostragem para PINN.

<small>`pinneapple_design/geometry/core/mesh.py` · 566 linhas · testes: 174</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MeshData` | classe | `(vertices, faces, normals)` | Malha de triângulos (vértices, faces, normais) com limites, transformações, normais e áreas. |

#### `pinneapple_design.geometry.core.registry`

Registro de construtores de ativos e a fábrica `build_geometry_asset` (arquivo ou primitiva). [sem teste direto]

<small>`pinneapple_design/geometry/core/registry.py` · 288 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_geometry_asset` | função | `(spec, *, options=None) -> GeometryAsset` | Constrói um `GeometryAsset` a partir de um `GeometrySpec` ou dicionário. |
| `load_geometry_asset` | função | `(geom, *, options=None) -> GeometryAsset` | Atalho: aceita caminho, `GeometryAsset` ou dicionário e devolve um `GeometryAsset`. |

#### `pinneapple_design.geometry.csg`

Geometria sólida construtiva 2D por SDF para domínios de colocação: formas básicas, união, interseção e diferença com amostragem por rejeição. A base `SDFShape` levanta NotImplementedError nos métodos abstratos.

<small>`pinneapple_design/geometry/csg.py` · 570 linhas · testes: 195 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SDFShape` | classe | `(bounds_min, bounds_max)` | Classe base de formas 2D por SDF, com `contains`, `on_boundary`, amostragem e operadores `+`, `*`, `-`. |
| `CSGRectangle` | classe | `(x_min, y_min, x_max, y_max) -> None` | Retângulo alinhado aos eixos. |
| `CSGCircle` | classe | `(center_x, center_y, radius) -> None` | Círculo 2D. |
| `CSGEllipse` | classe | `(center_x, center_y, a, b) -> None` | Elipse 2D alinhada aos eixos. |
| `CSGPolygon` | classe | `(vertices) -> None` | Polígono qualquer dado por vértices ordenados. |
| `CSGUnion` | classe | `()` | União booleana de duas formas: min das SDFs. |
| `CSGIntersection` | classe | `()` | Interseção booleana de duas formas: max das SDFs. |
| `CSGDifference` | classe | `()` | Diferença A menos B: max(sdf_a, -sdf_b). |
| `lshape` | função | `(width1, height1, width2, height2) -> CSGDifference` | Domínio em L. |
| `annulus` | função | `(cx, cy, r_inner, r_outer) -> CSGDifference` | Domínio anular (círculo externo menos interno). |
| `channel_with_hole` | função | `(length, height, hole_cx, hole_cy, hole_r) -> CSGDifference` | Canal retangular com obstáculo circular. |
| `t_junction` | função | `(h_length, h_height, v_x_start, v_width, v_height) -> CSGUnion` | Junção em T: canal horizontal mais ramo vertical. |

#### `pinneapple_design.geometry.gen`

Reexporta `build_primitive` e as funções de CadQuery.

<small>`pinneapple_design/geometry/gen/__init__.py` · 17 linhas · testes: 9</small>

#### `pinneapple_design.geometry.gen.airfoil`

Gera a superfície de um aerofólio NACA de 4 dígitos como tensor de pontos. [sem teste direto]

<small>`pinneapple_design/geometry/gen/airfoil.py` · 81 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `naca_parametric` | função | `(m=0.0, p=0.0, t_c=0.12, n_pts=100) -> torch.Tensor` | Gera pontos da superfície de um NACA 4 dígitos a partir de curvatura, posição e espessura. |

#### `pinneapple_design.geometry.gen.cadquery_advanced`

Modelos CadQuery paramétricos avançados (curva de tubo com flange, tê, redutor, serpentina, ventilador, engrenagem, dissipador de pinos) e reproduções dos exemplos clássicos da documentação CadQuery. Exige CadQuery.

<small>`pinneapple_design/geometry/gen/cadquery_advanced.py` · 292 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `pipe_bend_flanged` | função | `(p)` | Curva de tubo com raio e ângulo dados e flange parafusada em cada ponta. |
| `pipe_tee` | função | `(p)` | Tê igual: tubo principal em X e ramo em +Z. |
| `concentric_reducer` | função | `(p)` | Redutor cônico oco de D1 para D2. |
| `helical_coil` | função | `(p)` | Tubo sólido varrido ao longo de uma hélice. |
| `axial_fan` | função | `(p)` | Cubo com pás de perfil NACA com torção, em padrão polar. |
| `involute_tooth_profile` | função | `(module, teeth, pressure_deg=20.0, n=12) -> List[Tuple[float, float]]` | Contorno fechado de uma engrenagem de dentes retos com flancos evolventes. |
| `spur_gear` | função | `(p)` | Engrenagem de dentes retos extrudada a partir do perfil evolvente. |
| `pin_fin_heatsink` | função | `(p)` | Dissipador com base e matriz de pinos. |
| `occ_bottle` | função | `(p)` | Garrafa clássica do OCC (exemplo da documentação CadQuery). |
| `pillow_block` | função | `(p)` | Mancal paramétrico (exemplo da documentação CadQuery). |
| `lego_brick` | função | `(p)` | Bloco de montar com pinos em cima e tubos embaixo (exemplo da documentação CadQuery). |
| `register_advanced_templates` | função | `(reg)` | Registra todos esses construtores num `CadQueryRegistry` com seus esquemas de parâmetros. |

#### `pinneapple_design.geometry.gen.cadquery_gen`

Importação de malhas via CadQuery, especificação paramétrica de CAD, registro de construtores e exportação STEP/STL. CadQuery é dependência opcional.

<small>`pinneapple_design/geometry/gen/cadquery_gen.py` · 404 linhas · testes: 18</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CadQueryTessellationOptions` | classe | `(linear_deflection, angular_deflection)` | Controles de tesselação (desvios linear e angular). |
| `ParametricCadSpec` | classe | `(builder_id, params, schema, meta)` | Especificação declarativa de CAD: id do construtor, parâmetros e esquema. |
| `CadQueryBuildOptions` | classe | `(tessellation, heal, center_to_origin, normalize_to_unit_box)` | Opções de construção: tesselação, reparo, centralização e normalização. |
| `CadQueryRegistry` | classe | `(builders, schemas)` | Registro de construtores CAD paramétricos e seus esquemas. |
| `cadquery_available` | função | `() -> bool` | Diz se o CadQuery está instalado. |
| `cadquery_to_trimesh` | função | `(cq_obj, *, options=None)` | Converte um objeto CadQuery em memória num `trimesh.Trimesh`. |
| `build_mesh_from_step` | função | `(step_path, *, options=None) -> MeshData` | Lê um STEP com CadQuery e devolve `MeshData`. |
| `build_mesh_from_cadquery_object` | função | `(cq_obj, *, options=None) -> MeshData` | Converte um objeto CadQuery em `MeshData`. |
| `build_mesh_from_parametric_spec` | função | `(spec, registry, *, options=None) -> MeshData` | Constrói `MeshData` a partir de um `ParametricCadSpec` usando o registro. |
| `register_default_templates` | função | `(reg) -> CadQueryRegistry` | Registra alguns modelos de exemplo, como prova de conceito. |
| `export_cadquery` | função | `(cq_obj, out_path, *, fmt=None) -> Path` | Exporta um objeto CadQuery para STEP/STL ou outro formato suportado. |

#### `pinneapple_design.geometry.gen.car2d`

Perfil lateral 2D paramétrico de carro (tipo corpo de Ahmed, sem rodas) e sua rasterização em máscara/SDF.

<small>`pinneapple_design/geometry/gen/car2d.py` · 92 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CarProfile` | classe | `(slant_deg, windshield_deg, hood, clearance, diffuser_deg, nose, height, slant_length)` | Perfil de carro com ângulos de traseira, para-brisa e difusor; dá contorno, altura frontal, máscara e SDF. |
| `sample_designs` | função | `(n, seed=0) -> list[dict[str, float]]` | Amostra projetos por hipercubo latino em `DESIGN_SPACE`. |

#### `pinneapple_design.geometry.gen.domains`

Domínios físicos 2D prontos para treinar PINNs (canal, canal com obstáculo, cavidade, L, anel, vários obstáculos, junção T, SDF livre), com regiões de contorno nomeadas e lote PINN. A base levanta NotImplementedError nos métodos abstratos.

<small>`pinneapple_design/geometry/gen/domains.py` · 754 linhas · testes: 151 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BoundaryRegion` | classe | `(name, kind, conditions)` | Região de contorno nomeada com tipo e condições. |
| `PhysicsDomain2D` | classe | `(bounds_min, bounds_max, boundary_regions)` | Base dos domínios 2D: SDF, amostragem interna e por região, lote PINN e manifesto de condições. |
| `ChannelDomain2D` | classe | `(length=2.0, height=1.0, origin=(0.0, 0.0), inlet_velocity=1.0)` | Canal retangular com entrada à esquerda e saída à direita. |
| `ChannelWithObstacleDomain2D` | classe | `(length=2.0, height=1.0, origin=(0.0, 0.0), obstacle_center=(0.5, 0.5), obstacle_radius=0…)` | Canal retangular com obstáculo circular. |
| `LidDrivenCavityDomain2D` | classe | `(size=1.0, origin=(0.0, 0.0), lid_velocity=1.0)` | Cavidade quadrada com tampa móvel. |
| `LShapeDomain2D` | classe | `(size=2.0, origin=(-1.0, -1.0))` | Domínio em L para problemas de calor/Laplace. |
| `AnnularDomain2D` | classe | `(center=(0.0, 0.0), r_inner=0.2, r_outer=1.0, inner_condition=None, outer_condition=None)` | Domínio anular entre dois círculos. |
| `MultiObstacleDomain2D` | classe | `(length=3.0, height=1.0, origin=(0.0, 0.0), obstacles=None, inlet_velocity=1.0)` | Canal com vários obstáculos circulares. |
| `TJunctionDomain2D` | classe | `(main_length=3.0, main_height=0.5, branch_width=0.5, branch_length=1.0, origin=(0.0, 0.0)…)` | Junção em T: canal principal horizontal e ramo vertical. |
| `SDFDomain2D` | classe | `(sdf_fn, *, bounds_min, bounds_max, boundary_samplers=None, boundary_conditions=None)` | Domínio flexível a partir de qualquer função SDF e amostradores por região. |
| `get_domain` | função | `(name, **kwargs) -> PhysicsDomain2D` | Instancia um domínio 2D pelo nome do registro (ex.: `channel_2d`). |
| `list_domains` | função | `() -> List[str]` | Lista os nomes dos domínios 2D registrados. |

#### `pinneapple_design.geometry.gen.domains3d`

Domínios físicos 3D prontos para PINNs (cavidade, duto, tubo), com regiões de contorno e grade estruturada. Os alvos de contorno de `get_pinn_batch` são zeros de marcação; a base levanta NotImplementedError.

<small>`pinneapple_design/geometry/gen/domains3d.py` · 597 linhas · testes: 151 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsDomain3D` | classe | `(bounds_min, bounds_max, boundary_regions)` | Base dos domínios 3D: SDF, amostragem, lote PINN e grade estruturada. |
| `LidDrivenCavityDomain3D` | classe | `(size=1.0, origin=(0.0, 0.0, 0.0), lid_velocity=1.0, lid_axis='x')` | Cubo com tampa móvel na face de cima. |
| `ChannelDomain3D` | classe | `(length=2.0, height=1.0, width=1.0, origin=(0.0, 0.0, 0.0), inlet_velocity=1.0, flow_prof…)` | Duto retangular com perfil de entrada uniforme ou outro. |
| `PipeFlowDomain3D` | classe | `(radius=0.5, length=2.0, origin=(0.0, 0.0, 0.0), inlet_velocity=1.0)` | Tubo de seção circular com escoamento em x. |
| `get_domain_3d` | função | `(name, **kwargs) -> PhysicsDomain3D` | Instancia um domínio 3D pelo nome do registro. |
| `list_domains_3d` | função | `() -> List[str]` | Lista os nomes dos domínios 3D registrados. |

#### `pinneapple_design.geometry.gen.genetic_sdf`

Geometria implícita 'genética' por SDF em torch: um genoma de parâmetros define a forma (hoje só união de círculos). [sem teste]

<small>`pinneapple_design/geometry/gen/genetic_sdf.py` · 110 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GeneticSDF` | classe | `(sdf, genome, bounds_min, bounds_max, dim)` | Domínio SDF chamável com genoma explícito e limites. |
| `sdf_union` | função | `(a, b) -> torch.Tensor` | União de SDFs (mínimo) em torch. |
| `sdf_intersection` | função | `(a, b) -> torch.Tensor` | Interseção de SDFs (máximo) em torch. |
| `make_union_circles_sdf` | função | `(*, n_circles=3, seed=0, bounds_min=(0.0, 0.0), bounds_max=(1.0, 1.0), r_min=0.12, r_max=…)` | Cria um domínio aleatório como união de círculos, opcionalmente cortado pela caixa. |

#### `pinneapple_design.geometry.gen.primitives`

Construtores de malhas primitivas via trimesh e geometrias arquitetadas (tecido, trança), com fábrica genérica e booleanas. `project_points_to_surface` é marcado como placeholder.

<small>`pinneapple_design/geometry/gen/primitives.py` · 753 linhas · testes: 66 · palavras-chave no código (heurística; o texto acima diz o status real): placeholder/stub citado no docstring</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MeshBuilder` | classe | `()` | Tipo (protocolo) de um construtor de malha registrado. |
| `register_builder` | função | `(*names)` | Decorador que registra um construtor sob um ou mais nomes. |
| `list_builders` | função | `() -> Tuple[str, ...]` | Lista todos os nomes de construtores registrados, com apelidos. |
| `build_mesh` | função | `(name, **params) -> MeshData` | Fábrica geral: chama o construtor, aplica transformações e booleana opcional (cai para concatenação se não houver motor booleano). |
| `build_primitive` | função | `(name, **params) -> MeshData` | Constrói uma primitiva pelo nome e devolve `MeshData`. |
| `project_points_to_surface` | função | `(points, surface, *, method='nearest') -> np.ndarray` | [placeholder] Projeta pontos na superfície por ponto mais próximo (trimesh ou KD-tree); o docstring diz que deveria levantar erro, mas devolve `(pontos, distâncias)`. |

#### `pinneapple_design.geometry.gen.sdf_shapes`

Biblioteca de SDFs 2D e 3D em numpy: formas básicas, booleanas (também suaves), transformações e a classe `SDF` orientada a objeto.

<small>`pinneapple_design/geometry/gen/sdf_shapes.py` · 629 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SDF` | classe | `(fn, *args, **kwargs)` | Envoltório orientado a objeto de uma função SDF, com transformações e amostragem. |
| `sdf2d_circle` | função | `(p, center, radius) -> np.ndarray` | SDF de círculo 2D. |
| `sdf2d_rectangle` | função | `(p, center, half_extents) -> np.ndarray` | SDF de retângulo alinhado aos eixos. |
| `sdf2d_rounded_rectangle` | função | `(p, center, half_extents, radius) -> np.ndarray` | SDF de retângulo com cantos arredondados. |
| `sdf2d_ellipse` | função | `(p, center, semi_axes, iterations=4) -> np.ndarray` | SDF de elipse por refinamento de Newton. |
| `sdf2d_capsule` | função | `(p, a, b, radius) -> np.ndarray` | SDF de cápsula 2D entre dois pontos. |
| `sdf2d_line_segment` | função | `(p, a, b, thickness=0.0) -> np.ndarray` | SDF de um segmento, com espessura opcional. |
| `sdf2d_triangle` | função | `(p, v0, v1, v2) -> np.ndarray` | SDF de triângulo 2D. |
| `sdf2d_annulus` | função | `(p, center, r_inner, r_outer) -> np.ndarray` | SDF de anel 2D. |
| `sdf2d_sector` | função | `(p, center, radius, angle_start, angle_end) -> np.ndarray` | SDF de setor circular. |
| `sdf2d_convex_polygon` | função | `(p, vertices) -> np.ndarray` | SDF de polígono convexo. |
| `sdf3d_sphere` | função | `(p, center, radius) -> np.ndarray` | SDF de esfera. |
| `sdf3d_box` | função | `(p, center, half_extents) -> np.ndarray` | SDF de caixa alinhada aos eixos. |
| `sdf3d_cylinder` | função | `(p, center, radius, height, axis=2) -> np.ndarray` | SDF de cilindro finito num eixo. |
| `sdf3d_torus` | função | `(p, center, R, r, axis=2) -> np.ndarray` | SDF de toro. |
| `sdf3d_capsule` | função | `(p, a, b, radius) -> np.ndarray` | SDF de cápsula 3D. |
| `sdf3d_rounded_box` | função | `(p, center, half_extents, radius) -> np.ndarray` | SDF de caixa com arestas arredondadas. |
| `sdf3d_cone` | função | `(p, apex, height, radius) -> np.ndarray` | SDF de cone com base fechada. |
| `sdf_union` | função | `(a, b) -> np.ndarray` | União: min(a, b). |
| `sdf_intersection` | função | `(a, b) -> np.ndarray` | Interseção: max(a, b). |
| `sdf_difference` | função | `(a, b) -> np.ndarray` | Diferença: max(a, -b). |
| `sdf_smooth_union` | função | `(a, b, k=0.1) -> np.ndarray` | União suave com raio de mistura `k`. |
| `sdf_smooth_intersection` | função | `(a, b, k=0.1) -> np.ndarray` | Interseção suave. |
| `sdf_smooth_difference` | função | `(a, b, k=0.1) -> np.ndarray` | Diferença suave. |
| `sdf_translate` | função | `(sdf_fn, offset) -> Callable[[np.ndarray], np.ndarray]` | Translada uma SDF. |
| `sdf_scale` | função | `(sdf_fn, factor) -> Callable[[np.ndarray], np.ndarray]` | Escala uniformemente uma SDF. |
| `sdf_rotate_2d` | função | `(sdf_fn, angle_rad) -> Callable[[np.ndarray], np.ndarray]` | Gira uma SDF 2D. |
| `sdf_mirror_x` | função | `(sdf_fn) -> Callable[[np.ndarray], np.ndarray]` | Espelha uma SDF trocando o sinal de x. |
| `sdf_mirror_y` | função | `(sdf_fn) -> Callable[[np.ndarray], np.ndarray]` | Espelha uma SDF trocando o sinal de y. |
| `sdf_onion` | função | `(sdf_fn, thickness) -> Callable[[np.ndarray], np.ndarray]` | Cria uma casca de espessura dada. |
| `sdf_elongate_2d` | função | `(sdf_fn, dx=0.0, dy=0.0) -> Callable[[np.ndarray], np.ndarray]` | Alonga uma forma 2D em x e/ou y. |
| `sdf_repeat_2d` | função | `(sdf_fn, period, repetitions=None) -> Callable[[np.ndarray], np.ndarray]` | Repete uma SDF 2D com período dado. |
| `circle` | função | `(center=(0.0, 0.0), radius=0.5) -> SDF` | Atalho: `SDF` de círculo. |
| `rectangle` | função | `(center=(0.0, 0.0), half_extents=(0.5, 0.5)) -> SDF` | Atalho: `SDF` de retângulo. |
| `ellipse` | função | `(center=(0.0, 0.0), semi_axes=(0.5, 0.3)) -> SDF` | Atalho: `SDF` de elipse. |
| `annulus` | função | `(center=(0.0, 0.0), r_inner=0.2, r_outer=0.5) -> SDF` | Atalho: `SDF` de anel. |
| `capsule2d` | função | `(a=(0.0, -0.5), b=(0.0, 0.5), radius=0.2) -> SDF` | Atalho: `SDF` de cápsula 2D. |
| `sphere3d` | função | `(center=(0.0, 0.0, 0.0), radius=0.5) -> SDF` | Atalho: `SDF` de esfera. |
| `box3d` | função | `(center=(0.0, 0.0, 0.0), half_extents=(0.5, 0.5, 0.5)) -> SDF` | Atalho: `SDF` de caixa 3D. |
| `cylinder3d` | função | `(center=(0.0, 0.0, 0.0), radius=0.5, height=1.0, axis=2) -> SDF` | Atalho: `SDF` de cilindro. |
| `torus3d` | função | `(center=(0.0, 0.0, 0.0), R=0.5, r=0.15, axis=2) -> SDF` | Atalho: `SDF` de toro. |

#### `pinneapple_design.geometry.gen.threaded_connection`

Malhas 2D axissimétricas (r, z) de conexões rosqueadas rotativas (perfil API, TC50/TC46/TC38) geradas com gmsh. [sem teste]

<small>`pinneapple_design/geometry/gen/threaded_connection.py` · 442 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ThreadProfile` | classe | `(name, tpi, taper, h_factor, clearance_factor)` | Parâmetros de uma forma de rosca (fios por polegada, conicidade, altura, folga). |
| `ConnectionSpec` | classe | `(name, r_bore, r_outer, z_thread_start, z_thread_end, z_shoulder_top, thread, body, n_thr…)` | Especificação de um corpo (BOX ou PIN) da conexão: raios, posições e rosca. |
| `ThreadedConnectionMesher` | classe | `(spec, lc=1.0, lc_thread=None)` | Gera a malha 2D axissimétrica com gmsh e amostra pontos. |
| `TC50Geometry` | classe | `(body='BOX', lc=0.8, lc_thread=None)` | Conexão TC50 (diâmetro grande). |
| `TC46Geometry` | classe | `(body='BOX', lc=0.8, lc_thread=None)` | Conexão TC46 (diâmetro médio). |
| `TC38Geometry` | classe | `(body='BOX', lc=0.8, lc_thread=None)` | Conexão TC38 (padrão). |
| `build_thread_polygon` | função | `(spec, n_threads=None) -> List[Tuple[float, float]]` | Monta o polígono (r, z) da superfície rosqueada. |
| `get_connection_geometry` | função | `(connection, body='BOX', lc=0.8) -> ThreadedConnectionMesher` | Fábrica: devolve o malhador de uma conexão pelo nome. |

#### `pinneapple_design.geometry.io`

Ajudantes de E/S com imports tolerantes (STEP, IGES, CadQuery); `build_parametric_part` não existe em `cadquery_bridge`, então sempre vira `None`. [sem teste]

<small>`pinneapple_design/geometry/io/__init__.py` · 34 linhas · **sem teste direto**</small>

#### `pinneapple_design.geometry.io.cadquery_bridge`

Converte um sólido CadQuery em malha por tesselação e embrulha como `PhysicalSample`. [sem teste]

<small>`pinneapple_design/geometry/io/cadquery_bridge.py` · 39 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `cadquery_solid_to_upd` | função | `(solid, *, tess_linear_deflection=0.2, tess_angular_deflection=0.3, units='mm', meta=None…)` | Tessela um sólido CadQuery e devolve um `PhysicalSample` com vértices e faces. |

#### `pinneapple_design.geometry.io.iges`

Importa arquivos IGES em `MeshData` via gmsh, com cache.

<small>`pinneapple_design/geometry/io/iges.py` · 157 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `IgesImportConfig` | classe | `(kind, mesh_size, algorithm_2d, algorithm_3d, curvature_refine, optimize, heal_shapes)` | Configuração do import IGES (tipo, tamanho de malha, algoritmos, reparo). |
| `iges_to_mesh` | função | `(iges_path, *, cfg=None, cache_dir=None) -> MeshData` | Converte um IGES em `MeshData` via gmsh. |

#### `pinneapple_design.geometry.io.meshio_bridge`

Carrega e salva malhas com meshio (VTK/VTU/MSH/XDMF) e empacota como `PhysicalSample`. [sem teste]

<small>`pinneapple_design/geometry/io/meshio_bridge.py` · 149 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `load_meshio` | função | `(path, *, require_triangles=True) -> MeshData` | Lê uma malha com meshio e devolve `MeshData` de triângulos. |
| `save_meshio` | função | `(mesh, path, *, file_format=None, point_data=None, cell_data=None) -> None` | Salva `MeshData` com meshio. |
| `meshio_to_upd` | função | `(path)` | Lê malha e dados de ponto e empacota como amostra UPD. |

#### `pinneapple_design.geometry.io.sdf_meshing`

Utilitários para malhar geometrias implícitas 2D: pontos de contorno, contornos por marching squares e malha triangular via gmsh. [sem teste]

<small>`pinneapple_design/geometry/io/sdf_meshing.py` · 172 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SDFGrid2D` | classe | `(bounds_min, bounds_max, resolution)` | Grade 2D (limites e resolução) para avaliar uma SDF. |
| `sample_boundary_points_sdf2d` | função | `(sdf, *, grid=None, n=2048, rng=None, band=0.01) -> np.ndarray` | Amostra pontos de contorno por rejeição numa faixa fina em torno do zero. |
| `marching_squares_contours` | função | `(sdf, *, grid=None, level=0.0) -> list[np.ndarray]` | Extrai polilinhas do nível `sdf = level`. |
| `sdf2d_to_tri_mesh` | função | `(sdf, *, grid=None, mesh_size=0.03, cache_path=None) -> MeshData` | Gera malha triangular 2D do conjunto de nível zero com gmsh. |

#### `pinneapple_design.geometry.io.step`

Importa arquivos STEP em `MeshData` via gmsh, com cache. [sem teste]

<small>`pinneapple_design/geometry/io/step.py` · 132 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `StepImportConfig` | classe | `(kind, mesh_size, algorithm_2d, algorithm_3d, curvature_refine, optimize)` | Configuração do import STEP. |
| `step_to_mesh` | função | `(step_path, *, cfg=None, cache_dir=None) -> MeshData` | Converte um STEP em `MeshData` via gmsh. |

#### `pinneapple_design.geometry.io.stl`

Carregador de STL via trimesh que devolve `MeshData`. [sem teste]

<small>`pinneapple_design/geometry/io/stl.py` · 27 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `load_stl` | função | `(path, *, repair=True, compute_normals=True) -> MeshData` | Lê um STL, opcionalmente repara e calcula normais. |

#### `pinneapple_design.geometry.io.trimesh_bridge`

Adaptador entre `trimesh.Trimesh` e `MeshData` (carga, reparo e conversão).

<small>`pinneapple_design/geometry/io/trimesh_bridge.py` · 111 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TrimeshBridge` | classe | `()` | Adaptador que carrega arquivos via trimesh e converte para `MeshData`. |

#### `pinneapple_design.geometry.mesh`

Reexporta os malhadores 2D e 3D. [sem teste]

<small>`pinneapple_design/geometry/mesh/__init__.py` · 26 linhas · **sem teste direto**</small>

#### `pinneapple_design.geometry.mesh.mesher_2d`

Geração de malhas 2D: retângulo estruturado, malha de SDF via gmsh, triangulação de polígono e relatório de qualidade. [sem teste direto]

<small>`pinneapple_design/geometry/mesh/mesher_2d.py` · 436 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Mesh2D` | classe | `(vertices, triangles, boundary_edges, boundary_points, meta)` | Malha 2D de triângulos com estatísticas de qualidade e amostragem interna. |
| `mesh_rectangle_structured` | função | `(x_range=(0.0, 1.0), y_range=(0.0, 1.0), nx=20, ny=20) -> Mesh2D` | Malha triangular estruturada de um retângulo. |
| `mesh_sdf_2d` | função | `(sdf_fn, bounds_min, bounds_max, *, mesh_size=0.05, mesh_size_boundary=None, boundary_n_p…)` | Malha 2D de qualidade dentro de um domínio SDF usando gmsh. |
| `mesh_polygon_2d` | função | `(vertices, *, mesh_size=None, holes=None) -> Optional['Mesh2D']` | Triangula um polígono com Delaunay do scipy ou com a biblioteca `triangle` se houver. |
| `mesh_quality_report` | função | `(mesh) -> Dict[str, Any]` | Relatório de qualidade de uma malha 2D. |

#### `pinneapple_design.geometry.mesh.mesher_3d`

Malhas tetraédricas a partir de superfícies ou SDFs, nuvens de pontos para física e reconstrução de superfície. [sem teste direto]

<small>`pinneapple_design/geometry/mesh/mesher_3d.py` · 839 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Mesh3D` | classe | `(vertices, tets, surface_faces, boundary_groups, meta)` | Malha tetraédrica com faces de superfície e grupos de contorno. |
| `PhysicsPointCloud3D` | classe | `(interior, boundary, boundary_normals, near_wall, near_wall_dist, boundary_groups, meta)` | Pontos internos, de contorno e perto da parede para treino físico. |
| `mesh_from_surface` | função | `(surface, *, mesh_size=0.0, mesh_size_boundary=0.0, boundary_layer=False, bl_thickness=0.…)` | Gera malha de volume a partir de uma superfície fechada (gmsh), com camada-limite opcional. |
| `mesh_from_sdf_3d` | função | `(sdf_fn, bounds_min, bounds_max, *, resolution=64, mesh_size=0.0, clean=True, volume=True…)` | Extrai a superfície de uma SDF 3D por marching cubes e malha o volume. |
| `sample_physics_points` | função | `(mesh, *, n_interior=10000, n_boundary=2000, n_near_wall=0, near_wall_thickness=0.0, curv…)` | Gera pontos de colocação prontos para física a partir de uma malha 3D. |
| `surface_from_pointcloud` | função | `(points, *, normals=None, method='poisson', depth=8, alpha=0.0, density_quantile=0.05) ->…)` | Reconstrói uma superfície de uma nuvem de pontos (Poisson via open3d ou alpha shape via Delaunay). |

#### `pinneapple_design.geometry.mesh_collocator`

Amostrador de pontos de colocação 3D para PINN a partir de uma malha (interior, contorno com normais, suporte a corpo oco).

<small>`pinneapple_design/geometry/mesh_collocator.py` · 388 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MeshCollocatorConfig` | classe | `(n_interior, n_boundary, domain, boundary_mode, hollow, seed, max_rejection_tries)` | Configuração do `MeshCollocator` (quantidades, domínio, modo de contorno, oco, semente). |
| `CollocationBatch3D` | classe | `(X_interior, X_boundary, normals_boundary, boundary_groups)` | Lote de colocação 3D: pontos internos, de contorno, normais e grupos. |
| `MeshCollocator` | classe | `(mesh, config=None, *, inner_mesh=None)` | Amostra pontos internos e de contorno de uma malha 3D, como arrays ou tensores. |

#### `pinneapple_design.geometry.ops`

Reexporta as operações de malha: reparo, simplificação, remalha, limpeza, atributos e voxelização. [sem teste]

<small>`pinneapple_design/geometry/ops/__init__.py` · 53 linhas · **sem teste direto**</small>

#### `pinneapple_design.geometry.ops.clean`

Pipeline de limpeza de malhas 3D (fusão de vértices, degenerados, buracos, auto-interseção, suavização, decimação), com pymeshlab/open3d opcionais. [sem teste]

<small>`pinneapple_design/geometry/ops/clean.py` · 602 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CleanMeshConfig` | classe | `(merge_vertices, merge_tol, remove_degenerate, remove_duplicates, remove_unreferenced, fi…)` | Configuração completa do pipeline `clean_mesh`. |
| `mesh_quality_report_3d` | função | `(mesh) -> Dict[str, Any]` | Relatório de qualidade de uma malha de superfície 3D. |
| `smooth_mesh` | função | `(mesh, *, method='taubin', iterations=5, lam=0.5, mu=-0.53) -> MeshData` | Suavização Laplaciana ou de Taubin. |
| `remove_outlier_vertices` | função | `(mesh, *, nb_neighbors=20, std_ratio=2.0) -> MeshData` | Remove vértices atípicos por estatística de vizinhos. |
| `keep_largest_component` | função | `(mesh) -> MeshData` | Mantém só o maior componente conexo. |
| `decimate_mesh` | função | `(mesh, *, target_faces=None, ratio=0.5) -> MeshData` | Reduz o número de faces da malha. |
| `remove_self_intersections` | função | `(mesh) -> MeshData` | Remove faces auto-intersectantes com pymeshlab. |
| `clean_mesh` | função | `(mesh, *, config=None, verbose=False) -> MeshData` | Roda o pipeline completo de limpeza. |

#### `pinneapple_design.geometry.ops.curvature`

Geometria diferencial discreta em malhas de triângulos: áreas mistas, Laplaciano cotangente, Laplace-Beltrami e curvaturas.

<small>`pinneapple_design/geometry/ops/curvature.py` · 188 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `boundary_vertices` | função | `(mesh_or_vertices, faces=None) -> np.ndarray` | Máscara dos vértices em arestas de borda. |
| `euler_characteristic` | função | `(mesh_or_vertices, faces=None) -> int` | Característica de Euler V - E + F. |
| `mixed_areas` | função | `(mesh_or_vertices, faces=None) -> np.ndarray` | Área mista de Voronoi por vértice. |
| `cotan_laplacian` | função | `(mesh_or_vertices, faces=None) -> sparse.csr_matrix` | Matriz Laplaciana cotangente simétrica. |
| `laplace_beltrami` | função | `(f, mesh_or_vertices, faces=None) -> np.ndarray` | Laplace-Beltrami pontual de um campo nos vértices. |
| `gaussian_curvature` | função | `(mesh_or_vertices, faces=None, *, boundary='geodesic') -> np.ndarray` | Curvatura gaussiana por defeito angular. |
| `mean_curvature` | função | `(mesh_or_vertices, faces=None, *, signed=True, boundary='nan') -> np.ndarray` | Curvatura média por vértice a partir de Δx = -2Hn. |
| `principal_curvatures` | função | `(mesh_or_vertices, faces=None) -> Tuple[np.ndarray, np.ndarray]` | Curvaturas principais k1 e k2 a partir de H e K. |

#### `pinneapple_design.geometry.ops.features`

Normais de face e vértice, áreas e um indicador simples de curvatura. [sem teste]

<small>`pinneapple_design/geometry/ops/features.py` · 88 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `compute_face_normals` | função | `(mesh) -> np.ndarray` | Normais das faces. |
| `compute_vertex_normals` | função | `(mesh) -> np.ndarray` | Normais dos vértices ponderadas pela área. |
| `compute_face_areas` | função | `(mesh) -> np.ndarray` | Áreas das faces. |
| `compute_curvature_proxy` | função | `(mesh) -> np.ndarray` | Indicador leve de curvatura por face. |

#### `pinneapple_design.geometry.ops.lbm_bridge`

Voxeliza uma malha de superfície (STL/trimesh) numa máscara booleana 3D de obstáculo para o solver LBM, com ângulo de ataque.

<small>`pinneapple_design/geometry/ops/lbm_bridge.py` · 249 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `mesh_to_obstacle_mask` | função | `(mesh, bounds, resolution, *, aoa_deg=0.0, rotation_axis='y', oracle_supersample=2, devic…)` | Gera a máscara sólida 3D do LBM a partir de uma malha, com rotação por ângulo de ataque. |

#### `pinneapple_design.geometry.ops.meshfree`

Utilitários sem malha: interpolação por funções de base radial e SDF implícita reconstruída de nuvem de pontos.

<small>`pinneapple_design/geometry/ops/meshfree.py` · 397 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RBFInterpolator` | classe | `(kernel='gaussian', eps=1.0, reg=1e-10) -> None` | Interpolação de dados espalhados por RBF, com gradiente. |
| `ImplicitSurfaceRBF` | classe | `(kernel='gaussian', eps=2.0, offset=0.01, reg=1e-08) -> None` | Reconstrói uma SDF aproximada de pontos com normais e amostra dentro/contorno. |

#### `pinneapple_design.geometry.ops.occ_features`

Operações paramétricas diretas do OpenCASCADE (filete, chanfro, casca) sobre sólidos CadQuery/OCC.

<small>`pinneapple_design/geometry/ops/occ_features.py` · 206 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `fillet_edges` | função | `(solid, radius, edge_selector=None)` | Aplica filete de raio constante nas arestas de um sólido. |
| `chamfer_edges` | função | `(solid, distance, edge_selector=None)` | Aplica chanfro nas arestas de um sólido. |
| `shell_solid` | função | `(solid, thickness, faces_to_remove=None)` | Esvazia um sólido numa casca de espessura dada. |

#### `pinneapple_design.geometry.ops.pointcloud`

Utilitários de nuvem de pontos: amostrar de malha e de SDF 2D. [sem teste]

<small>`pinneapple_design/geometry/ops/pointcloud.py` · 80 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PointCloud` | classe | `(points, normals, features)` | Nuvem de pontos com normais e atributos. |
| `mesh_to_pointcloud` | função | `(mesh, *, n_surface=4096, seed=0, device=None, dtype=torch.float32, include_normals=True)…)` | Amostra uma nuvem de pontos da superfície de uma malha. |
| `sample_pointcloud_from_mesh` | função | `(points, cells, *, n=8192, seed=0) -> np.ndarray` | Atalho usado pelo webapp para amostrar pontos de uma malha. |
| `sdf2d_to_pointcloud` | função | `(sdf, *, grid=None, n_boundary=4096, band=0.01, seed=0, device=None, dtype=torch.float32)…)` | Amostra pontos de contorno de uma SDF 2D. |

#### `pinneapple_design.geometry.ops.remesh`

Remalha de superfície isotrópica via pymeshlab (melhor esforço). [sem teste]

<small>`pinneapple_design/geometry/ops/remesh.py` · 58 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `remesh_surface` | função | `(mesh, *, max_edge_length=None, target_edge_length=None) -> MeshData` | Remalha a superfície com comprimento de aresta alvo. |

#### `pinneapple_design.geometry.ops.repair`

Reparo de malha por melhor esforço com trimesh (duplicados, degenerados, normais, buracos).

<small>`pinneapple_design/geometry/ops/repair.py` · 61 linhas · testes: 1</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `repair_mesh` | função | `(mesh, *, remove_duplicates=True, remove_degenerate=True, fix_normals=True, fill_holes=Fa…)` | Repara a malha com trimesh. |

#### `pinneapple_design.geometry.ops.simplify`

Simplificação de malha por trimesh ou decimação quádrica do Open3D.

<small>`pinneapple_design/geometry/ops/simplify.py` · 187 linhas · testes: 1</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `simplify_mesh` | função | `(mesh, *, target_faces=10000, backend='auto', on_missing='error') -> MeshData` | Simplifica a malha até `target_faces`, com escolha de backend. |

#### `pinneapple_design.geometry.ops.voxelize`

Grades de voxels e amostragem sem malha (domínio, SDF, nuvem de pontos) para gerar pontos de colocação. [sem teste]

<small>`pinneapple_design/geometry/ops/voxelize.py` · 284 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `VoxelGrid` | classe | `(data, origin, voxel_size, dim, meta)` | Grade regular de voxels 2D/3D com conversão mundo-voxel e centros ocupados. |
| `voxelize_domain` | função | `(bounds, resolution, device=torch.device('cpu')) -> VoxelGrid` | Cria uma grade totalmente ocupada para um domínio retangular. |
| `voxelize_sdf` | função | `(sdf_fn, bounds, resolution, *, threshold=0.0, device=torch.device('cpu')) -> VoxelGrid` | Voxeliza uma SDF: voxel ocupado onde sdf <= limiar. |
| `voxelize_pointcloud` | função | `(points, bounds, resolution, *, values=None, aggregate='mean', device=torch.device('cpu')…)` | Rasteriza uma nuvem de pontos numa grade de voxels. |
| `voxelgrid_to_collocation` | função | `(grid, threshold=0.5, include_values=False)` | Extrai os centros dos voxels ocupados como pontos de colocação. |
| `sample_voxelgrid` | função | `(grid, n_samples, strategy='random', threshold=0.5) -> torch.Tensor` | Amostra pontos de colocação de uma grade de voxels. |

#### `pinneapple_design.geometry.optimize.loop`

Envoltório amigável para UI que usa `EvolutionaryDesignOptimizer` (CMA-ES/GA) com parâmetros em dicionário. [sem teste]

<small>`pinneapple_design/geometry/optimize/loop.py` · 124 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GeometryOptimizer` | classe | `(space, *, seed=0, sigma0=0.2)` | Otimizador de geometria com interface ask/tell/run sobre `EvolutionaryDesignOptimizer`. |

#### `pinneapple_design.geometry.retrieval`

Busca de peças parecidas: descritor de forma D2 sem ML e índice por similaridade de cosseno, também sobre embeddings pré-calculados.

<small>`pinneapple_design/geometry/retrieval.py` · 122 linhas · testes: 3</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GeometryIndex` | classe | `(ids, vectors, captions)` | Índice de similaridade de cosseno (força bruta) sobre vetores de peças, com busca por legenda. |
| `shape_descriptor` | função | `(mesh, n_pairs=20000, bins=48, seed=0) -> np.ndarray` | Histograma D2 mais extensões principais, normalizado; invariante a rotação, translação e escala. |
| `load_caption_index` | função | `(parquet_paths, id_column='abc_id', limit=None) -> GeometryIndex` | Monta um índice a partir de shards parquet com `abc_id`, `caption` e `embedding`. |

#### `pinneapple_design.geometry.sample`

Reexporta os amostradores baricêntricos, de caixa, de superfície e de partículas.

<small>`pinneapple_design/geometry/sample/__init__.py` · 31 linhas · testes: 3</small>

#### `pinneapple_design.geometry.sample.barycentric`

Amostragem e interpolação baricêntrica em triângulos. [sem teste]

<small>`pinneapple_design/geometry/sample/barycentric.py` · 66 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `sample_barycentric_uv` | função | `(n, rng=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]` | Amostra pesos baricêntricos uniformes num triângulo. |
| `sample_points_on_triangles` | função | `(tri_v0, tri_v1, tri_v2, n, rng=None) -> np.ndarray` | Amostra pontos uniformes num triângulo. |
| `interpolate_on_triangles` | função | `(values_v0, values_v1, values_v2, w0, w1, w2) -> np.ndarray` | Interpola valores por vértice com pesos baricêntricos. |

#### `pinneapple_design.geometry.sample.grids`

Amostragem uniforme e por hipercubo latino em caixas. [sem teste]

<small>`pinneapple_design/geometry/sample/grids.py` · 63 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `sample_uniform_box` | função | `(bounds_min, bounds_max, n, rng=None) -> np.ndarray` | Amostras uniformes numa caixa. |
| `sample_latin_hypercube_box` | função | `(bounds_min, bounds_max, n, rng=None) -> np.ndarray` | Amostras por hipercubo latino numa caixa. |

#### `pinneapple_design.geometry.sample.marching_squares`

Extrai o contorno de uma SDF 2D em torch por marching squares e amostra pontos dele. [sem teste]

<small>`pinneapple_design/geometry/sample/marching_squares.py` · 86 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `marching_squares_boundary` | função | `(sdf, *, bounds_min=(0.0, 0.0), bounds_max=(1.0, 1.0), resolution=256, device=None, dtype…)` | Devolve os vértices da polilinha de contorno. |
| `sample_boundary_points` | função | `(sdf, n, *, bounds_min=(0.0, 0.0), bounds_max=(1.0, 1.0), resolution=256, device=None, dt…)` | Amostra `n` pontos de contorno a partir do contorno extraído. |

#### `pinneapple_design.geometry.sample.particles`

Amostradores de partículas para SPH e conjuntos baseados em partículas. [sem teste]

<small>`pinneapple_design/geometry/sample/particles.py` · 51 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `sample_box_particles` | função | `(*, lo, hi, spacing, jitter=0.0, device=None, dtype=torch.float32) -> torch.Tensor` | Grade regular de partículas numa caixa, com tremor opcional. |
| `sample_circle_particles` | função | `(*, center, radius, spacing, jitter=0.0, device=None, dtype=torch.float32) -> torch.Tensor` | Partículas 2D dentro de um círculo. |

#### `pinneapple_design.geometry.sample.points`

Amostragem de pontos na superfície de malhas de triângulos (por área ou por pesos). [sem teste]

<small>`pinneapple_design/geometry/sample/points.py` · 108 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `sample_surface_points` | função | `(mesh, n, *, rng=None, return_normals=True, return_face_id=True)` | Amostra pontos uniformes na superfície, com normais e índice da face. |
| `sample_surface_points_weighted` | função | `(mesh, n, *, face_weights, rng=None, return_normals=True, return_face_id=True)` | Amostra pontos com pesos por face dados pelo usuário. |

#### `pinneapple_design.geometry.trough`

Seção longitudinal de uma calha de metal fundido como malha 2D do OpenFOAM (blockMesh): fundo inclinado, vertedouro, jato entrando por cima e ar acima.

<small>`pinneapple_design/geometry/trough.py` · 115 linhas · testes: 12</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TroughSection` | classe | `(length, slope_deg, weir_height, top, inlet_x, inlet_width, cell, thickness)` | Geometria da calha; gera o `blockMeshDict`, a gravidade inclinada, o nível do banho e grava a malha. |

#### `pinneapple_design.geometry.vehicle_cad`

CAD paramétrico de veículos tipo rover: primitivas de malha fechada, peças com material e massa, propriedades de massa e montagem em corpos para simulação multicorpo.

<small>`pinneapple_design/geometry/vehicle_cad.py` · 715 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CADPart` | classe | `(name, body, vertices, faces, material, mass, group)` | Peça com malha, corpo, material, massa e grupo; calcula propriedades de massa. |
| `VehicleCAD` | classe | `(title, parts, frames, notes)` | Conjunto de peças agrupadas em corpos, com propriedades de massa, poses, cena e exportação. |
| `orient_outward` | função | `(mesh) -> Mesh` | Inverte o enrolamento de uma malha fechada com volume negativo. |
| `signed_volume` | função | `(V, F) -> float` | Volume com sinal de uma malha fechada. |
| `box` | função | `(size, center=(0, 0, 0), R=None) -> Mesh` | Caixa como malha, com rotação opcional. |
| `beveled_box` | função | `(size, center=(0, 0, 0), bevel=0.02, n=3, R=None) -> Mesh` | Caixa com arestas arredondadas. |
| `cylinder` | função | `(p0, p1, radius, n=32) -> Mesh` | Cilindro (ou tronco de cone) fechado entre dois pontos. |
| `tube_shell` | função | `(axis_p0, axis_p1, r_in, r_out, n=48) -> Mesh` | Tubo anular fechado (aro de roda). |
| `revolve_closed` | função | `(profile, axis_origin=(0, 0, 0), axis=(0, 0, 1), n=48) -> Mesh` | Sólido de revolução de um perfil fechado. |
| `paraboloid_dish` | função | `(diameter, depth, thickness, center, axis, n=40, m=8) -> Mesh` | Refletor de antena parabólica como casca. |
| `loft` | função | `(rings, cap='fan') -> Mesh` | Sólido fechado passando por anéis de pontos. |
| `offset_polygon` | função | `(poly, d) -> np.ndarray` | Desloca um polígono convexo para dentro ou para fora. |
| `chamfered_rect` | função | `(sx, sy, c=0.0) -> np.ndarray` | Retângulo com cantos chanfrados a 45 graus. |
| `extrude_polygon` | função | `(poly, z0, z1, chamfer_top=0.0, chamfer_bottom=0.0, R=None, t=(0, 0, 0)) -> Mesh` | Prisma de um polígono convexo com chanfros em cima e embaixo. |
| `chamfered_box` | função | `(size, center=(0, 0, 0), edge=0.0, top=0.0, bottom=0.0, R=None) -> Mesh` | Caixa com chanfros planos a 45 graus. |
| `box_beam` | função | `(p0, p1, section0, section1=None, up=(0, 0, 1), chamfer=0.0) -> Mesh` | Viga de seção caixão reta, com afunilamento e chanfro opcionais. |
| `ladder_rail` | função | `(p0, p1, width, rail=(0.03, 0.04), rungs=4, side=(0, 1, 0), rung=(0.02, 0.02)) -> Mesh` | Trilho em escada: dois trilhos paralelos ligados por travessas. |
| `helix_sweep` | função | `(origin, axis, r_in, r_out, thickness, pitch, length, n_per_turn=24, phase=0.0) -> Mesh` | Hélice de seção retangular (rosca, broca). |
| `mli_blanket` | função | `(width, height, thickness=0.012, wrinkle=0.004, n=(28, 14), seed=0) -> Mesh` | Manta de isolamento multicamada com face externa enrugada. |
| `rotate` | função | `(mesh, R, about=(0, 0, 0)) -> Mesh` | Gira uma malha. |
| `translate` | função | `(mesh, t) -> Mesh` | Translada uma malha. |
| `rot` | função | `(axis, angle_rad)` | Matriz de rotação em torno de um eixo. |
| `merge` | função | `(meshes) -> Mesh` | Junta várias malhas. |
| `mesh_mass_properties` | função | `(V, F, mass) -> tuple[np.ndarray, np.ndarray, float]` | Centro de massa, tensor de inércia e volume de uma malha fechada de densidade uniforme. |
| `combine_mass` | função | `(items) -> tuple[float, np.ndarray, np.ndarray]` | Combina massa, centro de massa e inércia de vários itens. |
| `rover_wheel` | função | `(radius, width, *, rim_thickness=0.004, grousers=0, grouser_height=0.0, grouser_thickness…)` | Roda rígida de rover (aro, garras, raios, cubo) como lista de peças. |
| `solar_panel` | função | `(width, height, thickness=0.025, cells=(6, 4), gap=0.012, frame=0.03)` | Painel solar com moldura, substrato e células. |
| `radiator` | função | `(width, depth, thickness=0.02, fins=12, fin_height=0.03)` | Placa radiadora com aletas. |
| `camera_head` | função | `(width, depth, height, baseline, lens_radius=0.022, lights=2)` | Cabeça de câmera com par estéreo e barras de LED. |
| `actuator` | função | `(p0, p1, radius, cap_radius=None, cap_length=None, flange=0.012, housing='white_paint', t…)` | Atuador rotativo com carcaça, flanges e tampa do motor. |
| `pan_tilt_head` | função | `(width=0.36, depth=0.15, height=0.12, baseline=0.24, lens_radius=0.02, lights=6, aft_came…)` | Cabeça de mastro com pan-tilt, câmeras e LEDs. |
| `drill_assembly` | função | `(rail_height=1.3, rail_width=0.16, rail_offset=0.075, auger_radius=0.0125, flight_radius=…)` | Broca planetária com trilho de implantação, cabeça e trado. |

#### `pinneapple_design.geometry.vehicles3d`

Veículos 3D paramétricos para CFD e render: carro de passeio, carro de fórmula e lançador de dois estágios.

<small>`pinneapple_design/geometry/vehicles3d.py` · 687 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RoadCar` | classe | `(length, width, roof_height, clearance, hood_height, deck_height, windshield_deg, slant_d…)` | Sedã/cupê paramétrico (fastback ou notchback), com rodas, corpos para CFD e área frontal. |
| `LaunchVehicle` | classe | `(length, diameter, fairing_length, fairing_diameter, interstage_x, fin_root, fin_tip, fin…)` | Lançador de dois estágios com coifa ogiva e quatro aletas; inclui estimativa de Barrowman. |
| `FormulaCar` | classe | `(wheelbase, front_axle_x, ride_height, front_track, rear_track, wheel_radius, front_tyre_…)` | Carro de roda aberta paramétrico (não de nenhuma equipe), com asas, assoalho, difusor e rodas. |
| `road` | função | `(length=9.0, width=6.5, x0=-2.2) -> Part` | Plano de asfalto em z = 0 (só para desenho). |
| `naca4` | função | `(m=0.06, p=0.4, t=0.1, n=40) -> np.ndarray` | Perfil NACA 4 dígitos fechado com bordo de fuga finito. |

#### `pinneapple_design.geometry_library`

Fachada da biblioteca de geometrias (exposta como `pp.geometries`): listar, construir, exportar, renderizar miniaturas, carregar arquivos pré-gerados e exportar a galeria web.

<small>`pinneapple_design/geometry_library/__init__.py` · 96 linhas · testes: 34</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `list` | função | `(category=None) -> builtins.list[str]` | Nomes das geometrias da biblioteca, opcionalmente filtrados por categoria. |
| `categories` | função | `() -> dict[str, builtins.list[str]]` | Dicionário categoria -> nomes das geometrias. |
| `export` | função | `(name, path, fmt=None, lod='light', **params) -> str` | Constrói a geometria com os parâmetros dados e grava no formato pedido (ou pela extensão). |
| `render` | função | `(name, path=None, lod='light', size=(960, 640), **kw)` | Renderiza a miniatura de uma geometria com parâmetros padrão (câmera e parâmetros físicos opcionais). |
| `thumbnail` | função | `(name)` | Caminho da miniatura PNG versionada, renderizando-a se não existir. |
| `export_web` | função | `(directory, **kw) -> dict[str, Any]` | Gera o índice da galeria e as miniaturas para o navegador. |

#### `pinneapple_design.geometry_library.core`

Registro das entradas, especificação de parâmetros com faixas e fontes, o corpo fechado `Body` e o objeto `Geometry` devolvido por `build`.

<small>`pinneapple_design/geometry_library/core.py` · 321 linhas · testes: 21</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Param` | classe | `(name, default, unit, range, doc, choices)` | Um parâmetro físico: padrão, unidade, faixa admissível e o que controla. |
| `Source` | classe | `(what, ref, estimate)` | De onde vem um número do mundo real; `estimate=True` marca valor escolhido por julgamento, sem fonte publicada. |
| `Body` | classe | `(name, vertices, faces, materials, group, aero, closed)` | Uma superfície fechada (estanque, normais para fora) ou uma folha aberta só para desenho, com material por corpo ou por face. |
| `GeometrySpec` | classe | `(name, title, category, description, builder, params, dimensions, frame, sources, physics…)` | Especificação de uma entrada: nome, título, categoria, construtor, parâmetros, dimensões, fontes e usos físicos. |
| `Geometry` | classe | `(name, bodies, params, lod, spec)` | Geometria construída: corpos fechados com materiais, em metros, com malha, dimensões, volume, checagens, exportação e renderização. |
| `register` | função | `(spec) -> GeometrySpec` | Adiciona uma `GeometrySpec` ao registro, recusando nomes repetidos e chaves de dimensão inválidas. |
| `get_spec` | função | `(name) -> GeometrySpec` | Devolve a especificação registrada de um nome (carrega as entradas se preciso). |
| `params_hash` | função | `(name, params, lod) -> str` | Hash curto (SHA-256) de nome, nível de detalhe e parâmetros. |
| `build` | função | `(name, lod='light', **params) -> Geometry` | Constrói uma geometria da biblioteca no nível de detalhe `light` ou `fine` com os parâmetros físicos dados. |
| `names` | função | `() -> list[str]` | Lista ordenada dos nomes registrados. |
| `info` | função | `(name) -> dict[str, Any]` | Informações de uma entrada: dimensões, parâmetros com faixas, fontes e usos. |
| `summary_table` | função | `() -> str` | Tabela Markdown da biblioteca (nome, título, tamanho, usos). |

#### `pinneapple_design.geometry_library.entries`

Pacote de entradas: cada módulo dentro dele registra suas geometrias ao ser importado (descoberta automática). [sem teste]

<small>`pinneapple_design/geometry_library/entries/__init__.py` · 7 linhas · **sem teste direto**</small>

#### `pinneapple_design.geometry_library.entries._adapt`

Adaptador que transforma as listas de peças dos modelos anteriores (`Part` de aero, `CADPart` de veículo) em corpos fechados da biblioteca, soldando peças por material. [sem teste]

<small>`pinneapple_design/geometry_library/entries/_adapt.py` · 103 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `bodies_from_items` | função | `(items, prefix='') -> list[Body]` | Agrupa peças por grupo, solda vértices e separa em cascas conexas, cada uma virando um `Body` com material por face. |
| `base_name` | função | `(name, material) -> str` | Nome da pele de onde veio uma peça dividida por material (ex.: `fuselage_window` -> `fuselage`). |
| `drop_degenerate` | função | `(bodies) -> list[Body]` | Remove faces degeneradas (vértices repetidos) dos corpos. |
| `close_sheets` | função | `(bodies, thickness=0.004) -> list[Body]` | Dá espessura às folhas abertas dos modelos antigos para que todo corpo seja uma casca fechada. |

#### `pinneapple_design.geometry_library.entries.aircraft_new`

Registra três aeronaves movidas a rotor a partir de dados publicados: turboélice regional classe ATR 72-600, helicóptero leve classe H145 e drone quadricóptero classe F450. [sem teste]

<small>`pinneapple_design/geometry_library/entries/aircraft_new.py` · 880 linhas · **sem teste direto**</small>

#### `pinneapple_design.geometry_library.entries.industrial_bio`

Registra geometrias industriais e biomédicas de referência: dissipador de pinos, rotor centrífugo fechado, prédio alto padrão CAARC e arco da aorta torácica. [sem teste]

<small>`pinneapple_design/geometry_library/entries/industrial_bio.py` · 656 linhas · **sem teste direto**</small>

#### `pinneapple_design.geometry_library.entries.legacy`

Registra as primeiras entradas, embrulhando modelos já existentes no repositório: avião comercial A320, carro de passeio, carro de Fórmula, lançador de dois estágios e o rover NASA VIPER. [sem teste]

<small>`pinneapple_design/geometry_library/entries/legacy.py` · 288 linhas · **sem teste direto**</small>

#### `pinneapple_design.geometry_library.entries.marine_rail_space`

Registra entradas marinhas, ferroviárias e espaciais: navio KCS em escala real, casco Wigley analítico, trem de alta velocidade classe ICE 3 e lançador classe Falcon 9. [sem teste]

<small>`pinneapple_design/geometry_library/entries/marine_rail_space.py` · 890 linhas · **sem teste direto**</small>

#### `pinneapple_design.geometry_library.entries.wind_turbine`

Registra a turbina eólica de referência NREL 5 MW (Jonkman et al. 2009): pás pela tabela publicada de corda/torção/perfil, cubo, nacele e torre. [sem teste]

<small>`pinneapple_design/geometry_library/entries/wind_turbine.py` · 160 linhas · **sem teste direto**</small>

#### `pinneapple_design.geometry_library.exporters`

Escritores de arquivo determinísticos: STL binário, OBJ + MTL, glTF binário, OpenUSD ASCII e STEP AP214 (B-rep facetado), mais leitura de STL e checagem do STEP via OpenCASCADE.

<small>`pinneapple_design/geometry_library/exporters.py` · 180 linhas · testes: 21</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `normalise_format` | função | `(fmt) -> str` | Normaliza o nome do formato (aceita aliases como usd e stp) e valida. |
| `stl_bytes` | função | `(V, F, header) -> bytes` | STL binário de uma malha (V, F) com cabeçalho de 80 bytes. |
| `to_stl` | função | `(geom, aero_only=False) -> bytes` | Bytes STL de uma `Geometry` (opcionalmente só os corpos aerodinâmicos). |
| `to_obj` | função | `(geom) -> bytes` | Texto OBJ de uma `Geometry`, um grupo por superfície com referência ao MTL. |
| `to_mtl` | função | `(geom) -> bytes` | Arquivo MTL com as cores dos materiais usados. |
| `to_glb` | função | `(geom) -> bytes` | Bytes glTF binário da cena da geometria com a tabela de materiais. |
| `to_usda` | função | `(geom) -> bytes` | Texto OpenUSD da cena da geometria. |
| `to_step` | função | `(geom, aero_only=False) -> bytes` | Arquivo STEP AP214 com um `FACETED_BREP` por corpo fechado (folhas abertas são ignoradas). |
| `export` | função | `(geom, path, fmt=None) -> str` | Grava a geometria num caminho no formato dado ou deduzido da extensão. |
| `read_stl` | função | `(path)` | Lê STL binário como triângulos (m, 3, 3). |
| `step_check` | função | `(path) -> dict` | Relê um STEP com OpenCASCADE (`cadquery`): número de sólidos e volume total. |

#### `pinneapple_design.geometry_library.kit`

Blocos de construção de superfícies fechadas para as entradas: lofts, sólidos de revolução, seções NACA e asas/pás, caixas arredondadas, cilindros, tubos, extrusões e rotações.

<small>`pinneapple_design/geometry_library/kit.py` · 481 linhas · testes: 21</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `rot` | função | `(axis, angle_deg) -> np.ndarray` | Matriz de rotação em torno de um eixo por um ângulo em graus (Rodrigues). |
| `tf` | função | `(mesh, R=None, t=(0, 0, 0), about=(0, 0, 0), scale=1.0) -> Mesh` | Aplica escala, rotação e translação a uma malha. |
| `mirror` | função | `(mesh, axis=1) -> Mesh` | Espelha uma malha num eixo, invertendo a ordem das faces. |
| `with_mirror` | função | `(mesh, axis=1) -> list[Mesh]` | Lista com a malha e sua cópia espelhada. |
| `merge` | função | `(meshes) -> Mesh` | Junta várias malhas numa só. |
| `ear_clip_fast` | função | `(P2) -> np.ndarray` | Triangulação por ear clipping de polígono simples anti-horário, com testes vetorizados. |
| `loft` | função | `(rings, start='cap', end='cap') -> Mesh` | Superfície fechada por anéis, com pontas planas (cap) ou em polo. |
| `revolve` | função | `(x, r, n=48, axis='x', center=(0.0, 0.0, 0.0), phase=0.0) -> Mesh` | Sólido de revolução de um meridiano r(x) em torno de um eixo. |
| `superellipse` | função | `(hw, hh, n, e=2.0, zc=0.0, yc=0.0, top_e=None) -> np.ndarray` | Anel de pontos de uma superelipse (com expoente separado para a metade de cima, opcional). |
| `body_of_sections` | função | `(xs, hw, zb, zt, n=48, e=2.4, e_top=None) -> Mesh` | Corpo tipo fuselagem com seções superelípticas ao longo de x. |
| `naca4` | função | `(m=0.0, p=0.4, t=0.12, n=40, closed_te=True) -> np.ndarray` | Contorno NACA 4 dígitos em sentido anti-horário, com opção de bordo de fuga fechado. |
| `thick_te` | função | `(sec, te) -> np.ndarray` | Abre o bordo de fuga até uma espessura finita. |
| `section_ring` | função | `(sec2d, chord, twist_deg, le, frame=None, pivot=0.25) -> np.ndarray` | Posiciona uma seção de corda unitária em 3D com torção e bordo de ataque dados. |
| `wing` | função | `(stations, n=36, frame=None, tip='cap') -> Mesh` | Loft de asa ou pá por estações (bordo de ataque, corda, torção, espessura, arqueamento), fechado nas pontas. |
| `blend_sections` | função | `(a, b, w) -> np.ndarray` | Interpola linearmente duas seções com peso `w`. |
| `round_section` | função | `(n) -> np.ndarray` | Círculo unitário no formato de seção, com o mesmo número de pontos de `naca4`. |
| `box` | função | `(size, center=(0, 0, 0), R=None) -> Mesh` | Caixa fechada de tamanho dado, com rotação e centro opcionais. |
| `rounded_box` | função | `(size, center=(0, 0, 0), radius=0.05, n_side=4, R=None) -> Mesh` | Caixa com arestas e cantos arredondados (soma de Minkowski de caixa e esfera). |
| `cylinder` | função | `(p0, p1, radius, n=32, r1=None) -> Mesh` | Cilindro (ou tronco de cone) fechado entre dois pontos. |
| `ellipsoid` | função | `(center, radii, nu=24, nv=14, R=None) -> Mesh` | Elipsoide fechado com semieixos dados. |
| `tube` | função | `(path, radius, n=16, radii=None) -> Mesh` | Tubo fechado varrido ao longo de uma polilinha com referenciais de transporte paralelo. |
| `tube_loop` | função | `(path, radius, n=16) -> Mesh` | Tubo fechado ao longo de uma polilinha fechada (anéis, aros). |
| `torus` | função | `(center, R_major, r_minor, nu=48, nv=16, axis=(0, 0, 1)) -> Mesh` | Toro fechado com raios maior e menor. |
| `ear_clip` | função | `(poly) -> np.ndarray` | Triangula um polígono simples por ear clipping, em qualquer sentido. |
| `extrude` | função | `(poly2d, z0, z1, plane='xy') -> Mesh` | Prisma: polígono extrudado num eixo com tampas trianguladas. |
| `sweep_profile` | função | `(profile2d, path, up=(0, 0, 1)) -> Mesh` | Perfil 2D fechado varrido ao longo de um caminho, com tampas (trilhos, alças, nervuras). |
| `is_closed_volume` | função | `(mesh) -> bool` | True se a malha encerra volume positivo. |

#### `pinneapple_design.geometry_library.meshops`

Operações de malha em NumPy/SciPy: soldar vértices, orientar normais, estanqueidade, volume, autointerseção, solidificar folhas e união booleana (via `manifold3d`).

<small>`pinneapple_design/geometry_library/meshops.py` · 413 linhas · testes: 42</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `concat` | função | `(meshes) -> Mesh` | Concatena malhas reindexando as faces. |
| `compact` | função | `(V, F) -> Mesh` | Remove vértices não referenciados. |
| `weld` | função | `(V, F, tol=1e-09) -> Mesh` | Funde vértices coincidentes por tolerância relativa e remove faces colapsadas. |
| `area` | função | `(V, F) -> float` | Área total da malha. |
| `volume` | função | `(V, F) -> float` | Volume encerrado com sinal (positivo para normais para fora). |
| `face_normals` | função | `(V, F, unit=True) -> np.ndarray` | Normais das faces, unitárias ou não. |
| `edge_stats` | função | `(V, F) -> dict[str, int]` | Conta arestas de borda, não manifold e com orientação inconsistente. |
| `components` | função | `(F, n_vertices) -> np.ndarray` | Rótulo de componente conexo de cada face. |
| `orient_outward` | função | `(V, F) -> Mesh` | Inverte cada componente fechado com volume negativo. |
| `quality` | função | `(V, F, closed=True) -> dict[str, float]` | Resumo de qualidade da malha: contagens de arestas, estanqueidade, volume e área. |
| `self_intersections` | função | `(V, F, max_pairs=20000000, return_pairs=False)` | Conta pares de triângulos não adjacentes que se cruzam, com busca por k-d trees. |
| `union` | função | `(meshes) -> Mesh` | União booleana de malhas fechadas com `manifold3d`. |
| `crease_normals` | função | `(V, F, angle_deg=38.0) -> np.ndarray` | Normais por canto que mantêm arestas vivas acima de um ângulo. |
| `solidify` | função | `(V, F, thickness)` | Transforma uma folha aberta numa casca fina fechada. |
| `fix_winding` | função | `(V, F) -> Mesh` | Torna a orientação das faces consistente em cada componente e orienta para fora. |
| `refan_concave_caps` | função | `(V, F, materials=None, min_valence=5)` | Retriangula tampas planas em leque que se dobram sobre si (ex.: tampa de perfil arqueado). |

#### `pinneapple_design.geometry_library.render`

Renderizador por software (NumPy + Pillow, sem GPU) para miniaturas, folha de contatos e GIF giratório, com sombras, oclusão ambiente e supersampling. [sem teste]

<small>`pinneapple_design/geometry_library/render.py` · 362 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `rasterize` | função | `(sx, sy, depth, F, W, H, persp=True, max_frag=3000000)` | Rasteriza triângulos num z-buffer com id de triângulo e coordenadas baricêntricas por pixel. |
| `render_geometry` | função | `(geom, path=None, size=(960, 640), azimuth=None, elevation=None, ssaa=2, background='stud…)` | Renderiza uma `Geometry` e devolve uma imagem PIL (e grava se `path` for dado). |
| `contact_sheet` | função | `(geoms, path, cols=4, tile=(480, 330), title='') -> str` | Grade de miniaturas com nomes, em PNG. |
| `turntable` | função | `(geoms, path, frames=24, cols=4, tile=(240, 165), duration_ms=90) -> str` | GIF animado com todas as geometrias girando numa grade. |

#### `pinneapple_design.geometry_library.store`

Arquivos pré-gerados: manifesto com SHA-256, cópias locais, cache e Hugging Face Hub; `load` busca nessa ordem e reconstrói pelo código se o arquivo faltar.

<small>`pinneapple_design/geometry_library/store.py` · 138 linhas · testes: 21</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `file_name` | função | `(name, lod, fmt) -> str` | Nome do arquivo pré-gerado: `<nome>.<lod>.<formato>`. |
| `thumbnail_path` | função | `(name) -> Path` | Caminho da miniatura PNG de uma entrada. |
| `sha256` | função | `(path) -> str` | SHA-256 de um arquivo, lido em blocos. |
| `read_manifest` | função | `() -> dict[str, Any]` | Lê o manifesto de arquivos (ou devolve um vazio). |
| `write_manifest` | função | `(m) -> None` | Grava o manifesto de arquivos em JSON. |
| `cache_dir` | função | `() -> Path` | Diretório de cache das geometrias (`PINNEAPPLE_GEOMETRY_CACHE` ou `~/.cache/pinneapple/geometries`). |
| `offline` | função | `() -> bool` | True se o modo offline estiver ligado por variável de ambiente. |
| `hub_url` | função | `(name, fname) -> str` | URL do arquivo no dataset do Hugging Face Hub. |
| `load` | função | `(name, format='stl', lod='light', *, allow_build=True, allow_download=True) -> Path` | Caminho do arquivo pré-gerado de uma entrada, buscando repositório, cache e Hub, com conferência de hash. |

#### `pinneapple_design.geometry_library.web`

Gera o índice JSON da galeria e miniaturas JPEG de cada entrada para o navegador.

<small>`pinneapple_design/geometry_library/web.py` · 75 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `index` | função | `() -> dict[str, Any]` | Índice da galeria com o que cada cartão precisa (título, dimensões, arquivo, fontes, parâmetros, código). |
| `export_web` | função | `(directory, *, thumb_size=(360, 240)) -> dict[str, Any]` | Grava `index.json` e `thumbs/<nome>.jpg` no diretório e devolve o índice. |

#### `pinneapple_design.intent`

Fachada do projeto a partir da intenção (`pp.design`): reexporta os estágios formular, planejar, otimizar, verificar e avaliar, e os objetos tipados de cada um.

<small>`pinneapple_design/intent/__init__.py` · 73 linhas · testes: 24</small>

#### `pinneapple_design.intent.api`

Função `design(text)`: de uma frase a um pacote de projeto verificado, a uma lista de perguntas (`NeedsInput`) ou a uma recusa (`OutOfScope`). [sem teste]

<small>`pinneapple_design/intent/api.py` · 193 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NeedsInput` | classe | `(spec)` | Pedido dentro do escopo mas com valor crítico faltando: traz as perguntas a responder. |
| `design` | função | `(text, *, answers=None, llm=None, template=None, seed=0, out_dir=None, verify=True, n_scr…)` | Projeta a partir de uma meta em linguagem natural; devolve `DesignPackage`, `NeedsInput` ou `OutOfScope`. |

#### `pinneapple_design.intent.cli`

Linha de comando `pinneapple design "<meta>"` para rodar o pipeline a partir do shell, com respostas a perguntas e opção de pular a verificação. [sem teste]

<small>`pinneapple_design/intent/cli.py` · 64 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `main` | função | `(argv=None) -> int` | Ponto de entrada da CLI `pinneapple design` (argparse). |

#### `pinneapple_design.intent.figures`

Figuras do pacote de projeto em matplotlib (Agg): frente da otimização e figuras por template (loop, suporte, dissipador, asa), cada uma dizendo de qual modelo vêm os números. [sem teste]

<small>`pinneapple_design/intent/figures.py` · 300 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `pareto_figure` | função | `(ctx, x, y, xlabel, ylabel, out_dir) -> str \| None` | Gráfico de todas as avaliações do modelo reduzido (viáveis e inviáveis) com o design escolhido. |
| `loop_figures` | função | `(t, spec, d, ctx, out_dir) -> dict[str, str]` | Figuras do template do loop (pista, perfis de velocidade e força normal). |
| `loop_gif` | função | `(t, d, traj, out_dir, *, n_frames=48, size=(720, 480)) -> str` | GIF do carro dando a volta, a partir da trajetória do Newton. |
| `bracket_figures` | função | `(t, spec, d, ctx, out_dir) -> dict[str, str]` | Figuras do template do suporte (otimização e resultado do CalculiX quando rodou). |
| `heatsink_figures` | função | `(t, spec, d, ctx, out_dir) -> dict[str, str]` | Figuras do template do dissipador (otimização e verificação). |
| `wing_figures` | função | `(t, spec, d, ctx, out_dir) -> dict[str, str]` | Figuras do template da asa (comparação linha sustentadora x vortex lattice). |

#### `pinneapple_design.intent.materials`

Materiais com propriedades citadas (cada número com o documento de origem) para os templates; valores típicos de datasheet, não admissíveis de norma.

<small>`pinneapple_design/intent/materials.py` · 128 linhas · testes: 21</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Prop` | classe | `(value, unit, source, note)` | Uma propriedade de material: valor, unidade, fonte e nota. |
| `Material` | classe | `(key, label, category, props, processes, notes)` | Um material com categoria, propriedades citadas e processos, com `has` e `cite`. |
| `heat_sink_material` | função | `(key) -> Material` | Material do próprio modelo de dissipador (condutividade e densidade como a física usa). |
| `get` | função | `(key) -> Material` | Devolve um material pela chave ou levanta `KeyError`. |
| `candidates` | função | `(*needs, category=None) -> list[Material]` | Materiais que têm valor citado para todas as propriedades pedidas. |
| `find` | função | `(text) -> list[str]` | Chaves de materiais citados num texto livre. |

#### `pinneapple_design.intent.package`

`DesignPackage`: tudo para construir e checar o design e a gravação em pasta (design.json, relatório HTML e PDF, geometria STL/STEP/glTF e pasta `check/`). [sem teste]

<small>`pinneapple_design/intent/package.py` · 400 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DesignPackage` | classe | `(spec, plan, optimisation, candidate, design, robustness, alternatives, verification, ass…)` | Pacote de projeto completo (spec, plano, otimização, candidato, robustez, verificação, avaliação) com veredito, resumo e gravação em pasta. |
| `make_geometry` | função | `(name, title, bodies)` | Cria uma `Geometry` da biblioteca a partir de corpos, para exportar STL/STEP/glTF. |
| `write_html` | função | `(pkg, path) -> str` | Escreve o relatório HTML autocontido do pacote. |
| `write_pdf` | função | `(pkg, path) -> str` | Escreve um PDF curto com veredito, design, verificação e figuras. |

#### `pinneapple_design.intent.parse`

Converte intenção em `RequirementSpec`: escolhe o template por palavras-chave, lê números com unidades e preenche os slots; um LLM pode ser plugado via `IntentLLM`, mas o parser determinístico não precisa de rede.

<small>`pinneapple_design/intent/parse.py` · 416 linhas · testes: 21</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OutOfScope` | classe | `(text, kind, reason, needs, closest)` | O pedido não pode ser projetado aqui: diz o motivo e o que seria preciso. |
| `Mention` | classe | `(value, unit, start, end, quote)` | Um número achado no texto com a unidade, posição e trecho citado. |
| `IntentLLM` | classe | `()` | Protocolo de LLM plugável: qualquer objeto com `extract(text, schema) -> dict`. |
| `CallLLM` | classe | `(provider='anthropic', model=None, **kw)` | Adaptador de `pinneapple_llm.call_llm` (Anthropic, OpenAI, Ollama) para `IntentLLM`. |
| `register` | função | `(template) -> ProblemTemplate` | Adiciona uma instância de template ao registro usado por `formulate`. |
| `templates` | função | `() -> dict[str, ProblemTemplate]` | Dicionário dos templates registrados (carrega os embutidos). |
| `get_template` | função | `(name) -> ProblemTemplate` | Devolve um template pelo nome ou levanta `KeyError`. |
| `find_quantities` | função | `(text) -> list[Mention]` | Acha cada número do texto com a unidade que o segue; ignora números colados em letras. |
| `llm_schema` | função | `(tpls) -> dict[str, Any]` | Esquema JSON enviado ao LLM: templates possíveis e seus slots. |
| `select_template` | função | `(text, tpls=None) -> list[tuple[str, float]]` | Ordena os templates por pontuação de palavras-chave no texto. |
| `out_of_scope` | função | `(text) -> tuple[str, str] \| None` | Detecta pedidos fora de escopo ou recusados por palavras da lista `OUT_OF_SCOPE`. |
| `formulate` | função | `(text, *, llm=None, answers=None, template=None) -> RequirementSpec \| OutOfScope` | Transforma um pedido em `RequirementSpec` (ou `OutOfScope`), aceitando respostas a perguntas e template forçado. |

#### `pinneapple_design.intent.physics`

Subpacote com a física própria dos templates (suporte em L e pista de loop).

<small>`pinneapple_design/intent/physics/__init__.py` · 1 linhas · testes: 21</small>

#### `pinneapple_design.intent.physics.bracket`

Suporte em L sob carga na ponta: teoria de viga, modelo de elementos finitos hexaédricos próprio e modelo CalculiX (gmsh) com o filete e os furos de parafuso.

<small>`pinneapple_design/intent/physics/bracket.py` · 181 linhas · testes: 21</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `beam` | função | `(P, b, w, t, E)` | Viga em balanço sob carga na ponta: tensão de flexão na raiz e deflexão na ponta (vetorizado). |
| `l_mesh` | função | `(a, b, w, t, n_t=2, h=None, n_w=4, fillet=0.0)` | Malha estruturada de hexaedros do L (filete em escada), com nós engastados e carregados. |
| `hex_model` | função | `(P, a, b, w, t, E, nu, n_t=2, fillet=0.0)` | Resolve o L com `SolidFEM` (C3D8I): deflexão na ponta e tensão nominal na raiz. |
| `geo_shelf_bracket` | função | `(a, b, w, t, fillet, hole) -> str` | Script gmsh (OpenCASCADE) do L com canto interno filetado e dois furos de parafuso. |
| `gmsh_exe` | função | `() -> str \| None` | Caminho do executável `gmsh`, ou None se não achado. |
| `export_cad` | função | `(geo, workdir) -> dict[str, str]` | Gera STEP (B-rep exato) e STL a partir do script gmsh. |
| `calculix_model` | função | `(P, a, b, w, t, fillet, hole, E, nu, rho, size, workdir)` | Malha com gmsh (C3D10), aplica engaste e carga e resolve com CalculiX; devolve pico de von Mises, deflexão e reação. |
| `section_stress` | função | `(m, r, b, w, t, z_frac=0.45)` | Tensão média na face externa do braço numa seção, e o valor de viga M c / I no mesmo ponto. |
| `mesh_sizes` | função | `(t) -> tuple[float, float]` | Dois tamanhos de tetraedro quadrático (cerca de dois e três elementos na espessura). |

#### `pinneapple_design.intent.physics.loop_track`

Pista de loop para um carrinho: curva da pista (com clotoides), modelo 1-D do carro ao longo dela, forma fechada e a cena de corpo rígido no NVIDIA Newton.

<small>`pinneapple_design/intent/physics/loop_track.py` · 390 linhas · testes: 21</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LoopTrack` | classe | `(radius, clothoid, run_in, run_out, width, pitch)` | Pista em cinco trechos (lançamento, clotoide, arco, clotoide, saída) com curvatura, altura, amostragem e pegada. |
| `Run` | classe | `(completes, v_top, v_exit, n_min, n_max, top_margin, energy_lost, s_fail, profile)` | Resultado do modelo 1-D para um lote de designs, com os perfis do primeiro. |
| `NewtonLoopResult` | classe | `(completes, v_top, max_gap, s_reached, t, pos, vel, s, wall_s, solver, dt, meta)` | Resultado da simulação Newton: se completou, velocidade no topo, maior folga e trajetória. |
| `along_track` | função | `(radius, clothoid, v0, mass, *, crr=0.02, cd=0.6, area=0.0006, com_height=0.008, wheel_in…)` | Integra d(v^2)/ds ao longo da pista para um lote de designs; falha se o carro para ou perde contato. |
| `closed_form` | função | `(radius, clothoid, v0, mass, *, crr=0.02, cd=0.6, area=0.0006, com_height=0.008, wheel_in…)` | Estimativa de triagem: energia até o topo com perdas, condição no topo e carga de pico. |
| `track_meshes` | função | `(track, thickness=0.003, rail=0.008, n=600)` | Malhas fechadas da tira e dos dois trilhos laterais da pista. |
| `car_mesh` | função | `(length, width, height, wheel_r)` | Carrinho simples (corpo arredondado + quatro rodas) para desenhos e arquivos. |
| `newton_scene` | função | `(track, mass, v0, *, radius=0.006, mu=0.0, thickness=0.003, rail=0.008, n=900)` | Monta a cena de corpo rígido: pista como malha estática e carro como esfera lançada em +x. |
| `newton_check` | função | `(track, mass, v0, *, dt=2.5e-05, duration=None, iterations=12, record_every=None, radius=…)` | Roda o carro no loop no NVIDIA Newton e mede se fica na pista, velocidade no topo e folga máxima. |

#### `pinneapple_design.intent.pipeline`

Estágios RequirementSpec -> PhysicsPlan -> otimização multifidelidade (triagem em forma fechada + NSGA-II ou GP assistido) -> verificação com solver completo, robustez por Monte Carlo e veredito. [sem teste]

<small>`pinneapple_design/intent/pipeline.py` · 450 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsPlan` | classe | `(template, title, physics, assumptions, variables, objectives, failure_modes, fidelities,…)` | Plano de física de um template: física, hipóteses, variáveis, objetivos, modelos por fidelidade e modos de falha. |
| `CallLog` | classe | `(calls, wall)` | Registro de chamadas de modelo por fidelidade: designs avaliados e tempo. |
| `Candidate` | classe | `(x, design, outputs, closed_form, rank)` | Um design candidato com saídas do modelo reduzido e da forma fechada. |
| `OptimisationResult` | classe | `(candidates, method, n_screen, n_feasible_screen, n_reduced, log, study, notes)` | Resultado da otimização: candidatos, método, contagens de triagem e registro de chamadas. |
| `Robustness` | classe | `(p_success, ci, n, per_mode, margin_p05, inputs, sensitivity, level, calibration)` | Probabilidade de sucesso com intervalo de confiança, por modo de falha, e margens no percentil 5. |
| `Verification` | classe | `(ran, label, reason, result, comparison)` | Resultado da verificação com o solver completo e comparação com modelos reduzido e fechado. |
| `Assessment` | classe | `(trust, verdict, probability, ci, assumptions, reasons)` | Avaliação final: relatório de confiança, veredito, probabilidade, hipóteses e motivos. |
| `plan` | função | `(spec, template) -> PhysicsPlan` | Monta o `PhysicsPlan` do template para uma spec sem perguntas abertas. |
| `decode` | função | `(vs, x) -> dict[str, Any]` | Converte um vetor de design em dicionário de variáveis decodificadas. |
| `optimise` | função | `(spec, template, *, seed=0, n_screen=2000, budget=600, n_candidates=3, slow_budget=48, pr…)` | Faz triagem, otimiza e devolve até `n_candidates` designs viáveis. |
| `robustness` | função | `(spec, template, cand, *, n=2000, seed=0, level=REDUCED, log=None, calibration=None, opti…)` | Probabilidade de todas as margens >= 0 sob as entradas incertas, por Monte Carlo. |
| `verify` | função | `(spec, template, cand, *, log=None, workdir=None, options=None) -> Verification` | Roda o solver completo no candidato (se instalado) e compara com os modelos reduzido e fechado. |
| `assess` | função | `(spec, template, cand, rob, ver, *, agree_tol=0.1) -> Assessment` | Combina verificação e robustez num `TrustReport` e num veredito. |

#### `pinneapple_design.intent.spec`

Objetos de requisito tipados: `Quantity` em SI com proveniência (declarado, respondido, padrão, faltando), `Slot`, `Question` e `RequirementSpec`. [sem teste]

<small>`pinneapple_design/intent/spec.py` · 241 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SpecError` | classe | `()` | Valor inaceitável: unidade desconhecida, dimensão errada ou fora da faixa do template. |
| `Quantity` | classe | `(name, value, unit, provenance, source, quote)` | Um número em SI (ou uma escolha) com proveniência, fonte e trecho citado. |
| `Slot` | classe | `(name, label, unit, display_unit, keywords, cues, critical, default, default_source, boun…)` | Um valor que um template precisa ou pode usar: unidade SI, faixa, palavras-chave, padrão e criticidade. |
| `Question` | classe | `(slot, text, unit, choices, why)` | Uma pergunta de acompanhamento para um slot, com unidade, escolhas e motivo. |
| `ObjectiveChoice` | classe | `(name, sense, label, keywords)` | Um objetivo que o template pode otimizar (saída e sentido min/max). |
| `RequirementSpec` | classe | `(text, template, values, objectives, questions, issues, unsupported, parser, match_score)` | Especificação de requisitos: texto, template, valores, objetivos, perguntas e problemas encontrados. |
| `to_si` | função | `(value, unit) -> float` | Converte um valor da unidade dada para SI. |
| `from_si` | função | `(value, unit) -> float` | Converte um valor em SI para a unidade dada. |
| `same_dimension` | função | `(a, b) -> bool` | True se duas unidades têm a mesma dimensão física. |

#### `pinneapple_design.intent.template`

Classe base `ProblemTemplate` e seus tipos (fidelidade, modo de falha, variável, incerteza, resultado completo): como um tipo de pedido se liga a física, variáveis, modelos e falhas. [sem teste]

<small>`pinneapple_design/intent/template.py` · 218 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Fidelity` | classe | `(level, label, method, modules, executables, check)` | Um nível de modelo (closed_form, reduced, full) com rótulo, método e dependências; sabe se está disponível. |
| `FailureMode` | classe | `(name, label, description, critical)` | Um modo de falha: saída que guarda a margem (>= 0 não ocorre), rótulo e se é crítico. |
| `DesignVariable` | classe | `(name, lower, upper, unit, label, integer, choices)` | Variável de projeto com limites, unidade e opção inteira (escolhas discretas como índice). |
| `Uncertain` | classe | `(name, nominal, spread, dist, unit, source)` | Entrada incerta do estudo de robustez: nominal e meia-largura uniforme ou desvio-padrão normal. |
| `FullResult` | classe | `(outputs, checks, level, label, wall_s, artifacts, notes, compare, compare_options, compa…)` | O que o solver de alta fidelidade mediu para um design. |
| `ProblemTemplate` | classe | `(name, title, domain, keywords, physics, assumptions, slots, objectives, failure_modes, f…)` | Classe base dos templates; subclasses definem atributos e implementam `evaluate` e os ganchos do pacote (alguns levantam `NotImplementedError` na base). |

#### `pinneapple_design.intent.templates`

Templates embutidos; `builtin()` devolve loop do carrinho, suporte, dissipador e asa de planador. [sem teste]

<small>`pinneapple_design/intent/templates/__init__.py` · 12 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `builtin` | função | `() -> list` | Lista das instâncias dos templates embutidos. |

#### `pinneapple_design.intent.templates.bracket`

Template de suporte em L (cantoneira) que segura uma carga: espessura, largura, perna de fixação e material; modelos viga, FE hexaédrico e CalculiX. [sem teste]

<small>`pinneapple_design/intent/templates/bracket.py` · 346 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BracketTemplate` | classe | `()` | Template de suporte em L: variáveis, avaliação nas três fidelidades, verificação CalculiX, incertezas e itens do pacote. |

#### `pinneapple_design.intent.templates.example_spring`

Exemplo didático de template: mola helicoidal de compressão (Shigley), usado no guia `docs/guides/design_from_intent.md`; não é registrado por padrão.

<small>`pinneapple_design/intent/templates/example_spring.py` · 91 linhas · testes: 21</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CoilSpringTemplate` | classe | `()` | Template de exemplo de mola helicoidal (tensão com fator de Wahl, rigidez, fio de piano). |

#### `pinneapple_design.intent.templates.glider_wing`

Template de asa de planador modelo: envergadura, alongamento, afilamento e seção para menor taxa de afundamento, com linha sustentadora elíptica e vortex lattice. [sem teste]

<small>`pinneapple_design/intent/templates/glider_wing.py` · 297 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GliderWingTemplate` | classe | `()` | Template de asa de planador modelo, avaliado por linha sustentadora elíptica e vortex lattice grosso e fino. |
| `profile_cd0` | função | `(re, tc)` | Coeficiente de arrasto de perfil da asa por atrito de placa plana vezes fator de forma. |

#### `pinneapple_design.intent.templates.heatsink`

Template de dissipador de aletas planas (passivo ou com ventilador), reaproveitando a física de forma fechada e o FVM 3-D do dimensionador de dissipadores. [sem teste]

<small>`pinneapple_design/intent/templates/heatsink.py` · 327 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HeatSinkTemplate` | classe | `()` | Template de dissipador de aletas planas, com forma fechada e FVM 3-D como verificação. |

#### `pinneapple_design.intent.templates.loop_car`

Template do carrinho no loop: raio, entrada em clotoide, velocidade de lançamento, massa e tira da pista; modelos balanço de energia, EDO 1-D e NVIDIA Newton. [sem teste]

<small>`pinneapple_design/intent/templates/loop_car.py` · 469 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LoopCarTemplate` | classe | `()` | Template do carrinho no loop, com envelope de atrito e arrasto, EDO 1-D e verificação no Newton. |

#### `pinneapple_design.qualitative`

Reexporta `preview`, `part_sensitivity`, `Assembly` e os modelos qualitativos.

<small>`pinneapple_design/qualitative/__init__.py` · 36 linhas · testes: 13</small>

#### `pinneapple_design.qualitative.descriptors`

Descritores geométricos de corpos triangulados fechados (volume, área molhada, área frontal, esbeltez, perfil de seções) e `Assembly` de peças nomeadas. [sem teste direto pelo inventário]

<small>`pinneapple_design/qualitative/descriptors.py` · 259 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Descriptors` | classe | `(volume, wetted_area, centroid, extents, length, frontal_area, equivalent_diameter, slend…)` | Descritores geométricos de um corpo (volume, áreas, centroide, extensões, esbeltez, esfericidade e outros). |
| `Assembly` | classe | `(parts)` | Peças nomeadas de uma geometria que podem ser escaladas, movidas, removidas ou trocadas. |
| `as_mesh` | função | `(geom) -> tuple[np.ndarray, np.ndarray]` | Obtém `(vértices, faces)` de tupla, objeto tipo trimesh ou `Assembly`. |
| `extrude` | função | `(polygon, depth=1.0, axis='y') -> tuple[np.ndarray, np.ndarray]` | Prisma fechado a partir de um polígono 2D, para tratar seções 2D como corpos. |
| `face_geometry` | função | `(V, F)` | Área, normal unitária e centroide de cada face. |
| `describe` | função | `(geom, direction=(1.0, 0.0, 0.0)) -> Descriptors` | Calcula os `Descriptors` de um corpo fechado para uma direção de escoamento. |
| `section_profile` | função | `(geom, axis=0, n_slices=40, n_grid=48)` | Propriedades de seção ao longo de um eixo por voxelização. |

#### `pinneapple_design.qualitative.models`

Modelos físicos baratos avaliados face a face: escoamento externo (Newtoniano modificado e atrito), resfriamento convectivo, viga engastada e fórmulas do usuário; a classe base levanta `NotImplementedError` em `evaluate` (interface). [sem teste direto pelo inventário]

<small>`pinneapple_design/qualitative/models.py` · 501 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Quantity` | classe | `(name, unit, better, pt, en)` | Quantidade de um modelo qualitativo com unidade, sentido de melhora e nomes pt/en. |
| `Evaluation` | classe | `(quantities, terms, by_part, face_field, field_label, notes, caveats, mesh, extra)` | Resultado de um modelo numa geometria: quantidades, termos, contribuição por peça, campo na superfície e ressalvas. |
| `QualitativeModel` | classe | `(**conditions)` | Classe base dos modelos qualitativos com condições padrão; `evaluate` é abstrato. |
| `ExternalFlow` | classe | `()` | Arrasto e sustentação estimados por pressão Newtoniana modificada, separação e atrito de placa plana. |
| `ConvectiveCooling` | classe | `()` | Estimativa do calor dissipado por convecção de um corpo. |
| `Cantilever` | classe | `()` | Estimativa da deflexão de viga engastada com carga na ponta. |
| `ScalingModel` | classe | `(formulas, better=None, direction=(1.0, 0.0, 0.0), **conditions)` | Modelo com fórmulas do usuário em função dos descritores geométricos. |
| `sheltered_faces` | função | `(V, F, normals, centroids, direction, grid=96, tol=0.05) -> np.ndarray` | Marca faces de barlavento escondidas atrás de outras superfícies vistas de montante. |

#### `pinneapple_design.qualitative.preview`

Compara geometrias ou mudanças de peça para um objetivo, com direção, mecanismo, confiança e mapa na superfície; `quantify` roda a conta exata só onde vale e confere a leitura qualitativa.

<small>`pinneapple_design/qualitative/preview.py` · 523 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Objective` | classe | `(quantity, sense, model)` | Objetivo: quantidade, sentido (min/max) e modelo. |
| `VariantPreview` | classe | `(name, evaluation, change, mechanisms, parts, confidence, reasons, good)` | Prévia de uma variante: avaliação, mudança, mecanismos, peças, confiança e se é boa. |
| `QualitativePreview` | classe | `(model, objective, baseline, variants, geometries, lang)` | Comparação qualitativa de geometrias com ranking, texto, figura e `quantify`. |
| `QuantitativeCheck` | classe | `(preview, values, seconds)` | Confere a prévia contra valores quantitativos: concordância de direção e correlação de ranking. |
| `PartSensitivity` | classe | `(objective, model, base_value, rows, lang)` | Sensibilidade do objetivo a crescer cada peça em cada eixo. |
| `parse_objective` | função | `(objective, model=None) -> Objective` | Interpreta objetivos como "minimizar arrasto" ou `("tip_deflection", "min")`. |
| `preview` | função | `(geometries, objective, conditions=None, baseline=None, model=None, lang='pt') -> Qualita…)` | Compara geometrias para um objetivo e devolve a `QualitativePreview`. |
| `part_sensitivity` | função | `(assembly, objective, conditions=None, step=0.2, model=None, lang='pt', axes='xyz', about…)` | Mede quanto o objetivo muda ao crescer cada peça de um `Assembly`. |

#### `pinneapple_design.qualitative.quantitative`

Contrapartes quantitativas: elasticidade 3D por FEM em voxels com hexaedros de modos incompatíveis de Wilson, para checar prévias de viga engastada.

<small>`pinneapple_design/qualitative/quantitative.py` · 132 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `hex8_incompatible_stiffness` | função | `(size, E, nu) -> np.ndarray` | Rigidez 24×24 de um tijolo com modos incompatíveis de Wilson condensados. |
| `voxel_fem_cantilever` | função | `(geom, E=70000000000.0, nu=0.3, load=1000.0, axis=None, load_axis=2, clamp='min', n_slice…)` | Deflexão na ponta, rigidez e tensão de von Mises de uma viga engastada voxelizada por FEM 3D. |

#### `pinneapple_design.surrogate_inverse`

Projeto inverso por surrogate com cinco casos (dissipador, disco de freio, adesivo transdérmico, braço de suspensão, duto bass-reflex) que compartilham o mesmo método e são servidos por apps em `apps/`.

<small>`pinneapple_design/surrogate_inverse/__init__.py` · 38 linhas · testes: 91</small>

#### `pinneapple_design.surrogate_inverse.brake_rotor`

Caso do disco de freio ventilado: as aletas radiais entre os anéis tratadas como arranjo de aletas (ferro fundido) sob carga contínua, com Monte Carlo das tolerâncias de usinagem.

<small>`pinneapple_design/surrogate_inverse/brake_rotor.py` · 269 linhas · testes: 94</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RotorDesign` | classe | `(vane_count, vane_gap_m, vane_thickness_m)` | Design do disco: número, vão e espessura das aletas radiais, com espaçamento derivado. |
| `OptimizationResult` | classe | `(design, surrogate_predicted_temp_c)` | Resultado da otimização do disco: design e temperatura prevista. |
| `profile_params` | função | `(design, power_w, h_conv) -> dict[str, float]` | Grandezas da fórmula com elevação na raiz da aleta e comprimento radial corrigido. |
| `rotor_temperature_c` | função | `(design, power_w, h_conv) -> float` | Verdade: temperatura do anel de atrito sob calor contínuo. |
| `thermal_resistance_k_w` | função | `(design, h_conv) -> float` | Resistência térmica do disco (elevação por watt). |
| `vane_temperature_at_radial_position_c` | função | `(design, power_w, h_conv, r_fraction) -> float` | Perfil de temperatura ao longo da aleta, da borda interna (mais quente) à externa. |
| `thermal_field` | função | `(design, power_w, h_conv) -> dict[str, float]` | Parâmetros para colorir as aletas pelo perfil radial. |
| `constraint_violations` | função | `(design) -> dict[str, float]` | Violações com sinal (espaçamento mínimo, limites de vão e espessura). |
| `design_to_dict` | função | `(design) -> dict` | Dicionário do design com contagem arredondada e espaçamento. |
| `generate_dataset` | função | `(n_samples, seed=0) -> tuple[torch.Tensor, torch.Tensor]` | Designs aleatórios com perfis de resfriamento, rotulados pela fórmula. |
| `train_surrogate` | função | `(n_samples=6000, epochs=1500, seed=0) -> SurrogateFit` | Treina a MLP (saída em log) no dataset do disco. |
| `predict_temperature_c` | função | `(surrogate, design, power_w, h_conv) -> float` | Temperatura do disco prevista pelo surrogate. |
| `optimize_design` | função | `(surrogate, power_w, h_conv, steps=400, lr=0.05, penalty_weight=100.0) -> OptimizationRes…)` | Minimiza a temperatura prevista sobre (N, vão, t), como no caso do dissipador. |
| `verify_design` | função | `(design, power_w, h_conv, surrogate_predicted_temp_c) -> VerificationCard` | Reconfere o design com a fórmula e as regras de projeto. |
| `monte_carlo_uncertainty` | função | `(design, power_w, h_conv, safety_threshold_c, n_samples=200, seed=None, thickness_toleran…)` | Monte Carlo das tolerâncias de espessura e vão pela fórmula, com veredito no percentil 95. |
| `mesh` | função | `(design) -> meshes.Mesh` | Malha dos dois anéis e das aletas, em milímetros. |
| `export_stl_bytes` | função | `(design) -> bytes` | Bytes STL do disco ventilado. |

#### `pinneapple_design.surrogate_inverse.core`

Maquinário comum: MLP surrogate com entradas normalizadas, treino, congelamento, descida Adam nas variáveis de projeto e cartão de verificação que reconfere contra a verdade de forma fechada.

<small>`pinneapple_design/surrogate_inverse/core.py` · 267 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MLPSurrogate` | classe | `(bounds, hidden=48, n_layers=3, log_output=False)` | MLP `tanh` com entradas mapeadas dos limites para [-1, 1] e saída padronizada (opcionalmente em log). |
| `SurrogateFit` | classe | `(model, val_mean_rel_err, val_p95_rel_err, val_max_rel_err, n_train, n_val, epochs, secon…)` | Surrogate treinado com seus erros relativos em linhas separadas. |
| `CheckResult` | classe | `(name, verdict, detail, value, limit)` | Resultado de uma checagem: nome, veredito (PASS, WARNING, FAIL), detalhe, valor e limite. |
| `VerificationCard` | classe | `(checks, values, covered, not_covered)` | Design final reconferido contra a verdade de forma fechada, com veredito geral, cobertura explícita e `TrustReport`. |
| `fit_surrogate` | função | `(X, y, bounds, *, hidden=48, n_layers=3, epochs=1500, lr=0.003, seed=0, log_output=False,…)` | Treina a MLP com Adam em lote cheio e decaimento cosseno, separando as últimas linhas para validação. |
| `freeze` | função | `(model) -> nn.Module` | Congela os pesos do surrogate para que só o design se mova. |
| `descend` | função | `(loss_fn, x0, *, bounds, steps=400, lr=0.01, reparam='clamp', project=None) -> dict[str, …)` | Adam nas variáveis de projeto para minimizar uma perda, com reparametrização por clamp ou sigmoide. |
| `banded` | função | `(name, error, pass_limit, warn_limit, *, pass_text, warn_text, fail_text) -> CheckResult` | Veredito por faixas: PASS até `pass_limit`, WARNING até `warn_limit`, FAIL acima. |
| `relative_error` | função | `(predicted, reference, floor) -> float` | Erro relativo \|previsto - referência\| / max(\|referência\|, piso). |
| `constraint_checks` | função | `(violations, ok_text=…, skip=()) -> list[CheckResult]` | Uma checagem por violação com sinal (<= 0 satisfeita). |

#### `pinneapple_design.surrogate_inverse.heatsink_fins`

Caso do dissipador de CPU com aletas retas em base fixa de alumínio: fórmula 1-D de arranjo de aletas como verdade, surrogate na resistência térmica e checagem por volumes finitos 2-D numa célula de aleta.

<small>`pinneapple_design/surrogate_inverse/heatsink_fins.py` · 264 linhas · testes: 91</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HeatsinkDesign` | classe | `(fin_count, fin_height_m, fin_thickness_m)` | Design do dissipador: número, altura e espessura das aletas, com espaçamento derivado. |
| `OptimizationResult` | classe | `(design, surrogate_predicted_temp_c, steps_run)` | Resultado da otimização do dissipador: design, temperatura prevista e passos. |
| `profile_params` | função | `(design, power_w, h_conv) -> dict[str, float]` | Grandezas intermediárias da fórmula de aletas (m, comprimento corrigido, resistências, elevação na raiz). |
| `junction_temperature_c` | função | `(design, power_w, h_conv) -> float` | Verdade: temperatura da base (junção), usada nos rótulos e na verificação. |
| `thermal_resistance_k_w` | função | `(design, h_conv) -> float` | Resistência térmica (T_j - T_amb) / P, independente da potência neste modelo. |
| `fin_temperature_at_height_c` | função | `(design, power_w, h_conv, z_above_root_m) -> float` | Perfil 1-D de temperatura ao longo da altura da aleta. |
| `thermal_field` | função | `(design, power_w, h_conv) -> dict[str, float]` | O que um visualizador precisa para colorir as aletas pelo perfil. |
| `constraint_violations` | função | `(design) -> dict[str, float]` | Violações com sinal das regras de projeto (<= 0 satisfeita). |
| `design_to_dict` | função | `(design) -> dict` | Dicionário do design com contagem arredondada e espaçamento. |
| `generate_dataset` | função | `(n_samples, seed=0) -> tuple[torch.Tensor, torch.Tensor]` | Designs uniformes rotulados com a resistência térmica da fórmula. |
| `train_surrogate` | função | `(n_samples=6000, epochs=1500, seed=0) -> SurrogateFit` | Treina a MLP em log R. |
| `predict_temperature_c` | função | `(surrogate, design, power_w, h_conv) -> float` | Temperatura de junção prevista pelo surrogate. |
| `optimize_design` | função | `(surrogate, power_w, h_conv, steps=400, lr=0.05, penalty_weight=100.0) -> OptimizationRes…)` | Minimiza a temperatura prevista sobre (N, H, t) a partir da base. |
| `verify_design` | função | `(design, power_w, h_conv, surrogate_predicted_temp_c) -> VerificationCard` | Reroda a fórmula no design final: erro do surrogate, regras de projeto e limite de plausibilidade. |
| `fin_cell_check` | função | `(design, power_w, h_conv) -> dict` | Compara a fórmula com condução por volumes finitos 2-D num passo de aleta. |
| `mesh` | função | `(design) -> meshes.Mesh` | Malha da placa base com as aletas, em milímetros. |
| `export_stl_bytes` | função | `(design) -> bytes` | Bytes STL do dissipador. |

#### `pinneapple_design.surrogate_inverse.meshes`

Malhas triangulares das cinco peças em milímetros (para visualizador ou STL), como corpos fechados concatenados, sem união booleana.

<small>`pinneapple_design/surrogate_inverse/meshes.py` · 93 linhas · testes: 88</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `merge` | função | `(parts) -> Mesh` | Concatena malhas reindexando as faces. |
| `bbox_size` | função | `(mesh) -> tuple[float, float, float]` | Tamanho da caixa envolvente (x, y, z) da malha. |
| `to_stl` | função | `(mesh, name) -> bytes` | STL binário com cabeçalho iniciado pelo nome da peça. |
| `fin_array` | função | `(base_w_mm, base_d_mm, base_t_mm, n_fins, fin_h_mm, fin_t_mm) -> Mesh` | Placa base com aletas retas distribuídas na largura. |
| `vented_rotor` | função | `(r_mean_mm, radial_len_mm, ring_t_mm, gap_mm, n_vanes, vane_t_mm, n=96) -> Mesh` | Dois anéis de atrito unidos por aletas radiais no vão de ventilação. |
| `layered_disc` | função | `(radius_mm, layer_thicknesses_mm, n=96) -> Mesh` | Discos coaxiais empilhados (ex.: reservatório sob membrana). |
| `rectangular_beam` | função | `(width_mm, thickness_mm, length_mm) -> Mesh` | Caixa simples representando a viga. |
| `flanged_tube` | função | `(radius_mm, length_mm, flange_radius_mm, flange_t_mm, n=64) -> Mesh` | Duto cilíndrico com disco de flange na base. |

#### `pinneapple_design.surrogate_inverse.reflex_port`

Caso do duto bass-reflex: raio e comprimento que sintonizam a caixa numa frequência (ressonador de Helmholtz), com varredura do trade-off volume x precisão.

<small>`pinneapple_design/surrogate_inverse/reflex_port.py` · 232 linhas · testes: 88</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PortDesign` | classe | `(radius_m, length_m)` | Design do duto: raio e comprimento. |
| `OptimizationResult` | classe | `(design, surrogate_predicted_freq_hz)` | Resultado da otimização do duto. |
| `FrontierPoint` | classe | `(volume_weight, design, surrogate_predicted_freq_hz, real_achieved_freq_hz, real_freq_err…)` | Um ponto da varredura volume x precisão. |
| `resonant_frequency_hz` | função | `(design, cabinet_volume_l) -> float` | Verdade: frequência de ressonância de Helmholtz. |
| `acoustic_field` | função | `(design, cabinet_volume_l) -> dict[str, float]` | Um valor por design (o modelo concentrado não tem campo): a sintonia dentro da faixa alvo. |
| `constraint_violations` | função | `(design, cabinet_volume_l) -> dict[str, float]` | Violações com sinal das regras do duto. |
| `design_to_dict` | função | `(design) -> dict` | Dicionário do design do duto. |
| `generate_dataset` | função | `(n_samples, seed=0) -> tuple[torch.Tensor, torch.Tensor]` | Designs aleatórios rotulados com a frequência de Helmholtz. |
| `train_surrogate` | função | `(n_samples=6000, epochs=1500, seed=0) -> SurrogateFit` | Treina a MLP que prevê a frequência. |
| `predict_frequency_hz` | função | `(surrogate, design, cabinet_volume_l) -> float` | Frequência prevista pelo surrogate. |
| `optimize_design` | função | `(surrogate, cabinet_volume_l, target_freq_hz, steps=400, lr=0.05, penalty_weight=100.0) -…)` | Minimiza o erro relativo de frequência ao quadrado mais penalidade de volume do duto. |
| `sweep_pareto_frontier` | função | `(surrogate, cabinet_volume_l, target_freq_hz, n_points=8, steps=400, lr=0.05, penalty_wei…)` | Varre o peso lambda e monta o trade-off volume x precisão. |
| `verify_design` | função | `(design, cabinet_volume_l, target_freq_hz, surrogate_predicted_freq_hz) -> VerificationCa…)` | Reconfere o design com a fórmula de Helmholtz e as regras. |
| `mesh` | função | `(design) -> meshes.Mesh` | Malha do duto com flange, em milímetros. |
| `export_stl_bytes` | função | `(design) -> bytes` | Bytes STL do duto. |

#### `pinneapple_design.surrogate_inverse.suspension_arm`

Caso do braço de suspensão como viga em balanço retangular: menor largura x espessura que atinge uma vida à fadiga (lei de Basquin), com sensibilidades locais.

<small>`pinneapple_design/surrogate_inverse/suspension_arm.py` · 255 linhas · testes: 88</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ArmDesign` | classe | `(width_m, thickness_m)` | Design do braço: largura e espessura da seção retangular. |
| `OptimizationResult` | classe | `(design, surrogate_predicted_stress_pa)` | Resultado da otimização do braço. |
| `SensitivityResult` | classe | `(width_m, thickness_m, load_n, life_cycles, mass_kg, d_life_d_width, d_life_d_thickness, …)` | Sensibilidades locais (elasticidades) de vida e massa em relação a largura e espessura. |
| `bending_stress_pa` | função | `(design, load_n) -> float` | Verdade: tensão de flexão na raiz engastada. |
| `bending_stress_at_position_pa` | função | `(design, load_n, x_from_root_m) -> float` | Tensão de flexão ao longo do braço, máxima na raiz e nula na ponta. |
| `fatigue_life_cycles` | função | `(design, load_n) -> float` | Vida à fadiga em ciclos pela lei de Basquin. |
| `allowable_stress_for_life_pa` | função | `(target_life_cycles) -> float` | Tensão admissível para atingir a vida alvo. |
| `stress_field` | função | `(design, load_n) -> dict[str, float]` | Dados para colorir o braço pela tensão ao longo do comprimento. |
| `constraint_violations` | função | `(design, load_n, target_life_cycles) -> dict[str, float]` | Violações com sinal das regras do braço. |
| `design_to_dict` | função | `(design) -> dict` | Dicionário do design do braço. |
| `generate_dataset` | função | `(n_samples, seed=0) -> tuple[torch.Tensor, torch.Tensor]` | Designs aleatórios rotulados com a tensão de flexão. |
| `train_surrogate` | função | `(n_samples=6000, epochs=1500, seed=0) -> SurrogateFit` | Treina a MLP em log da tensão. |
| `predict_stress_pa` | função | `(surrogate, design, load_n) -> float` | Tensão prevista pelo surrogate. |
| `optimize_design` | função | `(surrogate, load_n, target_life_cycles, steps=500, lr=0.0006, penalty_weight=5e-11) -> Op…)` | Minimiza w t mais penalidade quando a tensão prevista passa da admissível. |
| `verify_design` | função | `(design, load_n, target_life_cycles, surrogate_predicted_stress_pa) -> VerificationCard` | Reconfere o design com a forma fechada e as regras. |
| `compute_sensitivity` | função | `(width_m, thickness_m, load_n, step_m=H_M) -> SensitivityResult` | Derivadas de vida e massa por diferenças centrais na forma fechada (nunca no surrogate). |
| `sensitivity_to_dict` | função | `(r) -> dict` | Dicionário das sensibilidades com um resumo em texto. |
| `mesh` | função | `(design) -> meshes.Mesh` | Malha do braço como caixa simples em milímetros (sem filetes nem furos). |
| `export_stl_bytes` | função | `(design) -> bytes` | Bytes STL do braço. |

#### `pinneapple_design.surrogate_inverse.transdermal_patch`

Caso do adesivo de liberação controlada por membrana (lei de Fick): área, espessura da membrana e concentração para uma taxa alvo e tempo de uso, com a menor área.

<small>`pinneapple_design/surrogate_inverse/transdermal_patch.py` · 249 linhas · testes: 88</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PatchDesign` | classe | `(area_m2, membrane_thickness_m, reservoir_conc_kg_m3)` | Design do adesivo: área, espessura da membrana e concentração do reservatório. |
| `OptimizationResult` | classe | `(design, surrogate_predicted_rate_mg_day)` | Resultado da otimização do adesivo. |
| `release_rate_mg_day` | função | `(design) -> float` | Verdade: taxa de liberação estacionária. |
| `depletion_time_days` | função | `(design) -> float` | Tempo até esgotar o reservatório. |
| `reservoir_mass_mg` | função | `(design) -> float` | Massa de fármaco no reservatório. |
| `concentration_at_membrane_depth` | função | `(design, z_from_reservoir_m) -> float` | Perfil linear estacionário de concentração na membrana. |
| `decay_rate_constant_per_day` | função | `(design) -> float` | Constante de primeira ordem da cauda após o esgotamento. |
| `simulate_release_timeseries` | função | `(design, duration_days=None, n_points=100) -> list[dict[str, float]]` | Série temporal de taxa, liberação acumulada e reservatório restante. |
| `concentration_field` | função | `(design) -> dict[str, float]` | Dados para colorir a membrana pelo perfil de concentração. |
| `constraint_violations` | função | `(design, target_wear_days) -> dict[str, float]` | Violações com sinal (taxa, tempo de uso, limites). |
| `design_to_dict` | função | `(design) -> dict` | Dicionário do design do adesivo. |
| `generate_dataset` | função | `(n_samples, seed=0) -> tuple[torch.Tensor, torch.Tensor]` | Designs aleatórios rotulados com a taxa de liberação. |
| `train_surrogate` | função | `(n_samples=6000, epochs=1500, seed=0) -> SurrogateFit` | Treina a MLP em log da taxa. |
| `predict_rate_mg_day` | função | `(surrogate, design) -> float` | Taxa prevista pelo surrogate. |
| `optimize_design` | função | `(surrogate, target_rate_mg_day, target_wear_days, steps=1200, lr=0.15, weight_rate=80.0, …)` | Minimiza área relativa mais erros de taxa e de tempo de uso num espaço limitado por sigmoide. |
| `verify_design` | função | `(design, target_rate_mg_day, target_wear_days, surrogate_predicted_rate_mg_day) -> Verifi…)` | Reconfere o design com a lei de Fick e as regras. |
| `mesh` | função | `(design) -> meshes.Mesh` | Malha do disco do reservatório com a membrana, em milímetros. |
| `export_stl_bytes` | função | `(design) -> bytes` | Bytes STL do adesivo. |

#### `pinneapple_design.thermal`

Subpacote de projeto térmico: dimensionamento de dissipadores (física + triagem por surrogate) e o estudo de resfriamento de placa.

<small>`pinneapple_design/thermal/__init__.py` · 2 linhas · testes: 3</small>

#### `pinneapple_design.thermal.board_cooling`

'Engenheiro que se auto-otimiza': resfria uma peça superaquecida numa placa com dissipador e ventiladores; reexporta `run_mission` e as etapas do estudo.

<small>`pinneapple_design/thermal/board_cooling/__init__.py` · 29 linhas · testes: 10</small>

#### `pinneapple_design.thermal.board_cooling.agent`

Agente do estudo de resfriamento: pedido em linguagem natural, plano (modelo fixo ou LLM via `pinneapple_autolab`), ferramentas controladas e evidências. [sem teste direto]

<small>`pinneapple_design/thermal/board_cooling/agent.py` · 197 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `t_read_mission` | função | `(state, text) -> dict[str, Any]` | Ferramenta que lê o pedido de resfriamento e guarda a missão no estado. |
| `t_baseline` | função | `(state) -> dict[str, Any]` | Ferramenta que resolve a placa nua em ar parado como ponto de partida. |
| `t_screen_natural` | função | `(state, gaps_mm=(4.0, 6.0, 8.0, 11.0)) -> dict[str, Any]` | Testa em ar parado o maior dissipador que cabe, para alguns espaçamentos de aleta; se nem o melhor atende, conclui que precisa de ar forçado. |
| `t_sample` | função | `(state, n=64, seed=0, workers=4) -> dict[str, Any]` | Ferramenta que roda soluções de referência em projetos amostrados. |
| `t_learn` | função | `(state, test_fraction=0.2, seed=0, max_error_k=3.0, min_coverage=0.9, add=32, max_n=320, …)` | Ajusta o surrogate e mede numa amostra separada; enquanto erro ou cobertura não atendem, adiciona soluções de referência até um limite. |
| `t_optimize` | função | `(state, rounds=5, k_sigma=2.0, seed=0) -> dict[str, Any]` | Ferramenta que roda o laço aprender-otimizar-verificar. |
| `t_verify` | função | `(state) -> dict[str, Any]` | Ferramenta que verifica o projeto escolhido numa malha fina. |
| `t_deliver` | função | `(state, out_dir='out', render=True, samples=24) -> dict[str, Any]` | Ferramenta que gera figuras, cena 3D, renders e relatório. |
| `toolbox` | função | `() -> Toolbox` | Monta o `Toolbox` com as ferramentas acima. |
| `template_plan` | função | `(text, *, n=64, out_dir='out', render=True, workers=4) -> TemplatePlanner` | Plano determinístico (sem LLM) com a sequência padrão de etapas. |
| `run_mission` | função | `(text, *, out_dir='out', planner='template', n=64, render=True, workers=4, llm=None, prov…)` | Planeja e roda o estudo; com `planner='llm'` cai para o plano fixo se o modelo não responder ou o plano for inválido. |

#### `pinneapple_design.thermal.board_cooling.learn`

Aproximação rápida da referência acoplada: temperaturas de junção e mapa de temperatura da placa (POD + processo gaussiano), com avaliação em amostra separada. [sem teste direto]

<small>`pinneapple_design/thermal/board_cooling/learn.py` · 107 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CoolingSurrogate` | classe | `(t_air_c, modes, tj_, map_, fit_seconds, n_train)` | Surrogate das temperaturas de junção e do mapa da placa, com ajuste, previsão e posto do mapa. |
| `errors` | função | `(pred_tj, pred_map, Y, maps) -> dict[str, Any]` | Calcula os erros de previsão de temperaturas e mapas contra a referência. |
| `holdout` | função | `(data, t_air_c, *, test_fraction=0.2, seed=0, modes=16) -> dict[str, Any]` | Treina numa parte das soluções e mede no resto: erros, cobertura da faixa de ±2 desvios e tempo de previsão. |

#### `pinneapple_design.thermal.board_cooling.mission`

A missão de resfriamento: placa, peça quente, limites, espaço disponível, orçamento de ventilação e espaço de projeto; inclui leitura determinística do pedido. [sem teste direto]

<small>`pinneapple_design/thermal/board_cooling/mission.py` · 212 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CoolingDesign` | classe | `(base_width_mm, base_depth_mm, base_thickness_mm, n_fins, fin_thickness_mm, height_mm, ai…)` | Um projeto: dissipador sobre a peça quente mais a velocidade do ar dos ventiladores (0 = convecção natural). |
| `DesignSpace` | classe | `(lo, hi, min_gap_mm, min_fin_height_mm)` | Limites das variáveis de projeto e regras geométricas (espaçamento e altura de aleta). |
| `CoolingMission` | classe | `(board, components, hot, t_ambient_c, tj_limit_c, max_width_mm, max_depth_mm, max_height_…)` | Placa, componentes, peça quente, temperaturas, limites de espaço e de potência de ventilação. |
| `default_mission` | função | `(power_w=40.0) -> CoolingMission` | Missão padrão: placa controladora com um SoC de potência `power_w`, reguladores e memória. |
| `parse_mission` | função | `(text) -> CoolingMission` | Lê um pedido em português ou inglês; os números citados substituem os padrões e `provenance` diz de onde veio cada campo. |

#### `pinneapple_design.thermal.board_cooling.optimize`

Busca com restrições no surrogate, verificação na referência e o laço entre eles. [sem teste direto]

<small>`pinneapple_design/thermal/board_cooling/optimize.py` · 141 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `constraint_violation` | função | `(mission, X, tj_upper, fan_w) -> np.ndarray` | Soma das violações de restrição (temperatura, ventilação, espaçamento), cada uma escalada. |
| `optimize` | função | `(mission, sur, *, k_sigma=2.0, seed=0, maxiter=120, popsize=24) -> dict[str, Any]` | Acha o projeto de menor massa no surrogate por evolução diferencial, com margem pessimista `k_sigma`. |
| `design_loop` | função | `(mission, data, *, rounds=5, k_sigma=2.0, seed=0, log=None) -> dict[str, Any]` | Laço aprender → otimizar → verificar na referência → adicionar o ponto, até confirmar. |

#### `pinneapple_design.thermal.board_cooling.report`

Entregáveis do estudo de resfriamento: figuras, cena e visualizador 3D, renders no Blender (opcional), relatório e relatório de confiança. [sem teste direto]

<small>`pinneapple_design/thermal/board_cooling/report.py` · 377 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `figure_maps` | função | `(state, path) -> str` | Figura com os mapas de temperatura da placa. |
| `figure_evolution` | função | `(state, path) -> str` | Figura da temperatura máxima ao longo do estudo: base, ar parado, otimizador e ótimos verificados. |
| `figure_holdout` | função | `(state, path) -> str` | Figura do surrogate contra a referência na amostra separada e do erro do mapa no projeto verificado. |
| `impact_gif` | função | `(state, path, hold=6) -> str` | Animação 2D do mapa da placa da placa nua ao projeto verificado. |
| `render_3d` | função | `(state, out_dir, *, render=True, samples=24, size=(1280, 720)) -> dict[str, Any]` | glTF/USD do projeto, visualizador da linha do tempo e, com Blender, imagens e animação. |
| `write_report` | função | `(state, out_dir, files, extra) -> str` | Escreve o relatório do estudo. |
| `deliver` | função | `(state, out_dir, *, render=True, samples=24) -> dict[str, Any]` | Gera todos os entregáveis numa pasta. |

#### `pinneapple_design.thermal.board_cooling.scene`

Cena 3D da placa resfriada: gabinete, ventiladores, mapa de temperatura, peças, dissipador e ar aquecido. [sem teste direto]

<small>`pinneapple_design/thermal/board_cooling/scene.py` · 323 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `fan_positions` | função | `(mission) -> tuple[list[tuple[float, float, float]], float]` | Centros e raio dos três ventiladores na parede de entrada. |
| `heat_sink_mesh` | função | `(mission, design, n_max, field3d=None)` | Vértices, faces e temperatura por vértice do dissipador. |
| `air_paths` | função | `(mission, result, n_x=18, heights=(4.0, 12.0, 22.0))` | Caminhos do ar e temperatura média de cada faixa por balanço de energia. |
| `board_scene` | função | `(mission, result, *, title='', paths=True)` | Um estado como cena estática com o campo de temperatura. |
| `timeline_scene` | função | `(mission, states, *, title='', t_range=T_RANGE)` | Os estados como quadros de uma animação, com painel de temperatura, massa e potência de ventilação. |

#### `pinneapple_design.thermal.board_cooling.simulate`

Simulação de referência da placa resfriada: modelo 3D da placa acoplado ao modelo do dissipador.

<small>`pinneapple_design/thermal/board_cooling/simulate.py` · 183 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `map_axes` | função | `(board) -> tuple[np.ndarray, np.ndarray]` | Centros (y, x) das células do mapa em mm. |
| `sink_mass_and_air` | função | `(mission, d) -> dict[str, float]` | Massa, perda de carga e potência do ar do dissipador por fórmula fechada. |
| `simulate` | função | `(mission, design, *, fidelity='train', pessimistic=True, tol=0.0001, max_iter=12) -> dict…)` | Temperaturas de junção, mapa da placa e checagens de um projeto (None = placa nua em ar parado). |
| `response` | função | `(mission, result) -> np.ndarray` | Vetor de respostas que o surrogate aprende: temperaturas de junção nominais e pessimistas. |
| `simulate_many` | função | `(mission, X, *, fidelity='train', workers=4) -> dict[str, np.ndarray]` | Soluções de referência de vários projetos em paralelo. |

#### `pinneapple_design.thermal.board_cooling.verify`

Verificação independente do projeto escolhido numa malha fina e o relatório de confiança.

<small>`pinneapple_design/thermal/board_cooling/verify.py` · 109 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `verify` | função | `(mission, design, *, coarse=None, prediction=None, holdout=None, train_X=None, test_X=Non…)` | Resolve o projeto na malha fina e monta o relatório de confiança com previsão, amostra separada e cobertura. |

#### `pinneapple_design.thermal.fin_array_speed`

Mede quanto um surrogate treinado é mais rápido que uma solução numérica e quão longe estão, numa varredura de dissipadores de aletas curados.

<small>`pinneapple_design/thermal/fin_array_speed.py` · 160 linhas · testes: 3</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `design_by_name` | função | `(name) -> dict[str, Any]` | Busca um dos projetos de `CURATED_DESIGNS` (KeyError se não existir). |
| `design_of` | função | `(spec) -> hf.HeatsinkDesign` | Converte uma especificação em `HeatsinkDesign`. |
| `train_surrogate` | função | `(n_samples=6000, epochs=1500, seed=0)` | Treina o surrogate de `heatsink_fins` e devolve o modelo e o erro relativo de validação. |
| `time_surrogate` | função | `(model, d, power_w, h_conv, reps=200) -> dict[str, float]` | Tempo médio de uma passada do surrogate e a temperatura prevista. |
| `reference_solve` | função | `(d, power_w, h_conv, refine=True) -> dict[str, Any]` | Solução de volumes finitos de um passo de aleta (`fin_cell`), cronometrada. |
| `measure_design` | função | `(model, spec, timing_reps=200, refine=True) -> dict[str, Any]` | Fórmula fechada, surrogate e referência cronometrados para um projeto. |
| `summarize` | função | `(designs) -> dict[str, Any]` | Resume as medições de todos os projetos. |
| `run_sweep` | função | `(model, progress_cb=None, timing_reps=200, refine=True) -> dict[str, Any]` | Mede todos os projetos de `CURATED_DESIGNS`. |
| `thermal_field` | função | `(name='baseline') -> dict[str, Any]` | Campo térmico de um projeto mais a temperatura da base. |

#### `pinneapple_design.thermal.fin_cell`

Condução em volumes finitos num passo de aleta de um dissipador de aletas retas: checagem independente da fórmula 1D de `pinneapple_physics.closed_form.fin_array_conduction`.

<small>`pinneapple_design/thermal/fin_cell.py` · 147 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FinCellResult` | classe | `(t_junction_c, t_fin_root_c, t_fin_tip_c, energy_balance_rel_error, n_cells, grid_change_…)` | Resultado: temperaturas de junção, raiz e ponta da aleta, erro de balanço de energia e mudança com a malha. |
| `solve_fin_cell` | função | `(*, n_fins, fin_thickness_m, fin_height_m, fin_spacing_m, depth_m, base_thickness_m, k_w_…)` | Resolve a condução permanente num passo de aleta (`power_w` é a potência do conjunto todo). |

#### `pinneapple_design.thermal.heatsink`

Dimensionamento de dissipadores de aletas planas (motor do app HeatSink Sizer); reexporta o motor, o dimensionador e os surrogates.

<small>`pinneapple_design/thermal/heatsink/__init__.py` · 18 linhas · **sem teste direto**</small>

#### `pinneapple_design.thermal.heatsink.engine`

Motor físico do HeatSink Sizer: avalia um projeto com a rede de fórmulas fechadas e a base resolvida em volumes finitos 3D, com faixa pessimista/otimista de convecção.

<small>`pinneapple_design/thermal/heatsink/engine.py` · 318 linhas · testes: 46</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DesignInput` | classe | `(base_width_mm, base_depth_mm, base_thickness_mm, n_fins, fin_thickness_mm, fin_height_mm…)` | Entrada de geometria do dissipador (base, aletas, material). |
| `OperatingInput` | classe | `(power_w, t_ambient_c, air_velocity_m_s, source_width_mm, source_depth_mm, source_x_mm, s…)` | Condições de operação (potência, ambiente, velocidade do ar, fonte, TIM, limite). |
| `base_theta` | função | `(g, op, r_conv, n=40) -> Dict[str, Any]` | Campo de temperatura da base para uma resistência de convecção. |
| `physics_resistance` | função | `(design, op, n=36) -> Dict[str, float]` | Resistência base-ambiente (sem TIM) pela física resolvida, mais perda de carga: o alvo do surrogate. |
| `field_3d` | função | `(g, op, fvm, nominal, t_amb, max_cells=64, n_z=14) -> Dict[str, Any]` | Temperaturas na superfície do dissipador inteiro para o visualizador 3D. |
| `model_scope` | função | `(mode) -> Dict[str, Any]` | Descreve o que o modelo cobre e para que lado erra cada simplificação; vai em todo relatório. |
| `evaluate_design` | função | `(design, op, *, grid=48, map_size=40) -> Dict[str, Any]` | Avalia um projeto completo (nominal, pessimista e otimista) e devolve resultado e mapas. |

#### `pinneapple_design.thermal.heatsink.field`

Solução 3D em volumes finitos da condução na placa-base do dissipador.

<small>`pinneapple_design/thermal/heatsink/field.py` · 111 linhas · testes: 18</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `solve_base_field` | função | `(*, base_width, base_depth, base_thickness, k, power_w, r_fins_k_w, source_width, source_…)` | Resolve o campo de temperatura da base com a fonte de calor e a resistência das aletas. |

#### `pinneapple_design.thermal.heatsink.monotone`

Treina o surrogate de dissipador com restrições de monotonicidade (modelo do zoo `heatsink_natural_mlp`). [sem teste direto]

<small>`pinneapple_design/thermal/heatsink/monotone.py` · 163 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `random_designs` | função | `(mode, n, rng) -> np.ndarray` | Linhas de atributos de projetos aleatórios no intervalo, amostradas log-uniformemente. |
| `split_indices` | função | `(n) -> dict[str, np.ndarray]` | Divisão do app em teste, calibração, validação e treino. |
| `train_monotone` | função | `(mode='natural', *, seed=0, epochs=600, lbfgs_rounds=4, lbfgs_steps=200, n_random=20000, …)` | Treina o surrogate com penalidade de monotonicidade e ajuste L-BFGS e devolve o pacote a salvar. |

#### `pinneapple_design.thermal.heatsink.sizer`

Dimensionamento: requisitos entram, projetos verificados saem; triagem por surrogate (limite superior conformal) ou por fórmula fechada, depois verificação física.

<small>`pinneapple_design/thermal/heatsink/sizer.py` · 251 linhas · testes: 18</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SizingRequest` | classe | `(power_w, t_limit_c, t_ambient_c, air_velocity_m_s, source_width_mm, source_depth_mm, tim…)` | Pedido de dimensionamento: potência, limites, ar, fonte, envelope, materiais e processo. |
| `size` | função | `(req, surrogates) -> Dict[str, Any]` | Amostra candidatos, faz a triagem, arredonda e verifica os melhores com o motor físico. |

#### `pinneapple_design.thermal.heatsink.surrogate`

Surrogate neural (MLP) do motor físico, usado só para triagem; um modelo por modo (forçado/natural), treinado e salvo em `ARTIFACTS`. [sem teste direto]

<small>`pinneapple_design/thermal/heatsink/surrogate.py` · 334 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Surrogate` | classe | `(mode, model, x_mean, x_std, y_mean, y_std, conformal_log_q, report)` | Modelo treinado com normalização e quantis conformais; prevê, salva e carrega. |
| `n_fins_for` | função | `(width_mm, gap_mm, t_mm)` | Número de aletas que cabem numa largura com espaçamento e espessura dados. |
| `features` | função | `(mode, W, D, tb, N, t, H, k, sw, sd, Ta, Q, V=None) -> np.ndarray` | Mapa vetorizado de atributos do surrogate. |
| `reference_solve` | função | `(params, mode='forced') -> dict \| None` | Roda o motor físico num projeto dado nas colunas do conjunto de dados. |
| `reference_solve_batch` | função | `(X, names, output='r_base_k_w', mode='forced') -> np.ndarray` | `reference_solve` em cada linha; NaN onde o motor rejeita o projeto. |
| `generate_dataset` | função | `(mode, n, seed=0, workers=4) -> List[Dict[str, float]]` | Gera o conjunto de dados de treino com o motor físico em paralelo. |
| `rows_to_xy` | função | `(mode, rows) -> Tuple[np.ndarray, np.ndarray]` | Converte linhas do conjunto em matrizes X, Y. |
| `train` | função | `(mode, rows, seed=0, epochs=400) -> Surrogate` | Treina o surrogate de um modo. |
| `load_surrogates` | função | `() -> Dict[str, Surrogate]` | Carrega os surrogates salvos que existirem em `ARTIFACTS`. |
| `main` | função | `(n_forced=6000, n_natural=5000, workers=4, reuse=False) -> None` | Gera os dados, treina e grava os surrogates e o relatório de confiança. |

#### `pinneapple_design.thermal.heatsink.trust`

Relatório de confiança dos surrogates de dissipador (erros, convergência, cobertura). [sem teste direto]

<small>`pinneapple_design/thermal/heatsink/trust.py` · 202 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `surrogate_report` | função | `(sur, *, X, Y, split, history, train_seconds, n_rows) -> Dict` | Monta o relatório de confiança de um surrogate a partir dos dados e do histórico. |

#### `pinneapple_design.topology`

Reexporta problemas, filtros, otimizadores e o driver de otimização topológica.

<small>`pinneapple_design/topology/__init__.py` · 19 linhas · testes: 9</small>

#### `pinneapple_design.topology.driver`

Laço de otimização: variáveis → simetria → filtro de densidade → projeção de Heaviside → problema, com adjunto, continuação de beta e checagem por diferenças finitas. [sem teste direto pelo inventário]

<small>`pinneapple_design/topology/driver.py` · 127 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DesignChain` | classe | `(n_el, filt=None, symmetry=None, beta=0.0, eta=0.5)` | Cadeia `x_phys = H_beta(W (S x))` e seu adjunto. |
| `TopOptResult` | classe | `(x, x_phys, objective, history)` | Resultado: densidades de projeto e físicas, objetivo e histórico. |
| `optimize` | função | `(problem, volfrac, filt=None, optimizer='oc', max_iter=200, tol=0.01, x0=None, symmetry=N…)` | Roda a otimização topológica com fração de volume, filtro, OC ou MMA, simetria e continuação de beta. |
| `finite_difference_check` | função | `(fun, x, idx, h=1e-06) -> np.ndarray` | Erro relativo do gradiente adjunto contra diferenças centrais em entradas escolhidas. |

#### `pinneapple_design.topology.filters`

Filtro linear de densidade, mapa de simetria e projeção de Heaviside com derivadas. [sem teste direto pelo inventário]

<small>`pinneapple_design/topology/filters.py` · 115 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DensityFilter` | classe | `(centroids, r_min, volumes=None)` | Filtro linear de densidade com raio `r_min`, com construtor para grades e adjunto. |
| `SymmetryMap` | classe | `(S)` | Mapa 0/1 de poucas variáveis de projeto para densidades de elemento (identidade, rotacional). |
| `heaviside` | função | `(x, beta, eta=0.5) -> np.ndarray` | Projeção de Heaviside suave com nitidez beta e limiar eta. |
| `heaviside_derivative` | função | `(x, beta, eta=0.5) -> np.ndarray` | Derivada da projeção de Heaviside. |

#### `pinneapple_design.topology.magnetic`

Otimiza o ferro do rotor para torque no modelo 2D A_z de `pinneapple_simulation.electromagnetics`, com sensibilidade por adjunto em cada ângulo.

<small>`pinneapple_design/topology/magnetic.py` · 79 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RotorTorqueProblem` | classe | `(machine, thetas, I_peak, gamma, objective='mean', T_ref=1.0)` | Problema de ferro do rotor: torque médio ou ripple sobre ângulos, com restrição de torque médio. |

#### `pinneapple_design.topology.optimizers`

Regras de atualização: critério de otimalidade (OC) e método das assíntotas móveis (MMA). [sem teste direto pelo inventário]

<small>`pinneapple_design/topology/optimizers.py` · 117 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MMA` | classe | `(n, m, xmin=0.0, xmax=1.0, move=0.5, asyinit=0.5, asyincr=1.2, asydecr=0.7, c=1000.0, d=1…)` | Método das assíntotas móveis para problemas com restrições de desigualdade e limites. |
| `oc_update` | função | `(x, dfdx, dvdx, volume_target, volume_of, move=0.2, eta=0.5, xmin=0.0, xmax=1.0)` | Um passo do critério de otimalidade com bisseção no multiplicador de volume. |

#### `pinneapple_design.topology.problems`

Problemas em grades de elementos Q4: compliance estrutural (MBB, viga engastada), compliance térmica e dissipação de Stokes com penalização de Brinkman. [sem teste direto pelo inventário]

<small>`pinneapple_design/topology/problems.py` · 264 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Q4Grid` | classe | `(nelx, nely, h=1.0)` | Grade estruturada de elementos Q4 quadrados com numeração e montagem. |
| `Compliance2D` | classe | `(nelx, nely, fixed_dofs, loads, penal=3.0, E0=1.0, Emin=1e-09, nu=0.3)` | Compliance estrutural em estado plano de tensão com SIMP; construtores `mbb` e `cantilever`. |
| `ThermalCompliance2D` | classe | `(nelx=60, nely=60, sink_fraction=0.1, q=1.0, penal=3.0, kmin=0.001)` | Placa com fonte de calor uniforme e sumidouro numa parte da borda esquerda; minimiza a compliance térmica. |
| `StokesDissipation2D` | classe | `(nelx, nely, inlets, outlets, mu=1.0, alpha_max=None, alpha_min=None, q=0.1, penalty=1000…)` | Escoamento de Stokes com penalização de Brinkman minimizando a dissipação. |
| `elasticity_ke` | função | `(nu=0.3, E=1.0, h=1.0, thickness=1.0) -> np.ndarray` | Rigidez 8×8 de um elemento Q4 em estado plano de tensão. |
| `conduction_ke` | função | `(h=1.0) -> np.ndarray` | Matriz de condução 4×4 de um elemento Q4. |

### Exemplos de uso reais do repositório

**Vortex lattice do avião comercial da biblioteca de geometrias** — `examples/geometry_library/03_airliner_vlm.py`

```python
g = pp.geometries.build("airliner_a320")
p = g.params
ac = Airliner(length=p["length"], wing_area=p["wing_area"], aspect_ratio=p["aspect_ratio"], taper=p["taper"],
              sweep_le=p["sweep_le"], dihedral=p["dihedral"], twist=p["twist"], fan_diameter=p["fan_diameter"],
              engine_station=p["engine_station"], sharklets=p["sharklets"])
x_ref = ac.mac_le_x + 0.25 * ac.mac
sol = solve(ac, x_ref)
print(g)
```

**Preset do dissipador no estudo de projeto (forma fechada)** — `tests/test_design_studies.py`

```python
def test_heatsink_geometry_matches_library_closed_forms():
    from pinneapple_design.thermal.heatsink.engine import DesignInput
    from pinneapple_physics.closed_form.cantilever_fatigue import stress_from_moment_pa

    p = presets.heatsink_problem("closed_form")
    b = presets.heatsink_baseline()
    ev = p.evaluate(p.space.from_dicts([b]))
    d = DesignInput(**b)
    assert ev.Y["mass_g"][0] == pytest.approx(d.geometry().mass_kg() * 1e3, rel=1e-12)
```

**Surrogate do surrogate_inverse com erro em dados separados** — `tests/pinneapple_design/test_surrogate_inverse.py`

```python
def test_fit_surrogate_learns_a_power_law_and_reports_held_out_error():
    rng = np.random.default_rng(0)
    X = torch.tensor(rng.uniform([1.0, 1.0], [10.0, 5.0], size=(2000, 2)), dtype=torch.float32)
    y = X[:, 0] ** 2 / X[:, 1]
    fit = core.fit_surrogate(X, y, [(1.0, 10.0), (1.0, 5.0)], epochs=1000, log_output=True, hidden=32, n_layers=2)
    assert fit.n_val == 200 and fit.n_train == 1800
    assert fit.val_mean_rel_err < 0.03 and fit.val_p95_rel_err >= fit.val_mean_rel_err
    with torch.no_grad():
        assert float(fit.model(torch.tensor([[4.0, 2.0]]))[0]) == pytest.approx(8.0, rel=0.05)
```

**Montar e amostrar um domínio** — `docs/core_concepts/geometry_domain.md`

```python
from pinneapple_design import build_domain, sample_domain

domain = build_domain("lid_driven_cavity_2d")   # dim=2 by default
x_int, x_bnd = sample_domain(domain, n_interior=4096, n_boundary=512)
```

**Laço de projeto com blocos de montar** — `examples/bricks/self_modifying_bricks.py`

```python
from pinneapple_design.bricks import (
    BridgeGoal,
    CantileverGoal,
    TowerGoal,
    compare,
    dynamics,
    ldraw,
    mesh,
    reference_cantilever,
    report,
    statics,
    validation,
    visuals,
)
```

**Estudo de resfriamento de placa** — `examples/autolab/self_optimizing_engineer.py`

```python
from pinneapple_design.thermal.board_cooling import DEFAULT_TEXT, run_mission
...
        run = run_mission(a.text, out_dir=a.out, planner=a.planner, n=n, render=render, workers=a.workers)
        st = run.state
        m = st["read_mission"]["mission"]
        v, h = st["verify"], st["learn"]["holdout"]
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 110 de 196.
- Funções de teste em arquivos que importam algum módulo do pacote: 744.

Itens do catálogo de métodos (`pinneapple_catalog/method_status.json`) cujo código está neste pacote:

| id | método | status | testes contra referência | outros testes |
|---|---|---|---:|---:|
| S39 | Density-based topology optimisation (SIMP, filter, projection, OC, MMA; compliance, thermal, Stokes-Brinkman, machine rotor) | validated | 1 | 2 |

### Limitações conhecidas

- `aero.case`, `aero.case3d`, `aero.dataset` e `aero.verify` precisam do OpenFOAM instalado para rodar; `render_blender` precisa do `bpy` (Blender).
- Os modelos de aeronave são de ordem reduzida (polar elíptica, build-up de Raymer, vortex lattice com esteira plana sem roll-up); o docstring de `vlm.streamlines` avisa que a esteira não mostra o enrolamento.
- Dependências opcionais mudam o que roda: `pymoo-nsga2` exige pymoo, o otimizador evolutivo usa `cma` se houver, o bayesiano e o `GPSurrogate` usam scikit-learn; métodos impossíveis voltam com status `skipped`.
- `Evaluator.torch` e os avaliadores não diferenciáveis levantam `NotImplementedError` (métodos por gradiente só funcionam com avaliadores diferenciáveis, como a fidelidade surrogate do dissipador).
- `meshops.union` exige `manifold3d`; `exporters.step_check` exige `cadquery` (OpenCASCADE); o carregamento do Hub depende de rede (há modo offline e reconstrução pelo código).
- A verificação completa do `intent` depende de programas externos: CalculiX e gmsh para o suporte, NVIDIA Newton para o loop; sem eles o relatório de confiança diz que a checagem completa não rodou.
- No template da asa de planador, a fidelidade 'full' é um vortex lattice fino; o docstring diz que um modelo RANS seria o próximo passo e não roda aqui.
- Os cinco casos usam fórmulas de livro texto com hipóteses declaradas; o docstring do pacote diz que nenhum foi validado contra medições.
- `sample_domain` chama `domain.sample_boundary`, mas `PhysicsDomain2D`/`PhysicsDomain3D` só têm `sample_boundary_region`; com os domínios de `build_domain` a chamada deve falhar (o exemplo da documentação usa exatamente esse caminho).
- `geometry.io.__init__` tenta importar `build_parametric_part` de `cadquery_bridge`, que não define esse nome; o valor fica sempre `None`.
- `gen.primitives.project_points_to_surface` é marcado como placeholder; o docstring diz que deveria levantar erro, mas devolve uma tupla `(pontos, distâncias)` em vez de um array.
- Muitas partes dependem de pacotes opcionais: trimesh, meshio, gmsh, CadQuery/OCC, pymeshlab, open3d, NVIDIA Newton (`bricks.dynamics`), Blender/`bpy` e ffmpeg (renders), `pinneapple_autolab` (agente). Os `__init__` usam imports tolerantes que viram `None` em silêncio.
- Vários módulos não têm teste direto: os construtores de lote (`builders`), `core.geometry`, `core.registry`, `io.*` (exceto IGES e trimesh), `mesh.mesher_2d`/`mesher_3d`, `ops.clean`, `ops.voxelize`, `sample.*`, `threaded_connection`, `genetic_sdf` e quase todo `thermal.board_cooling` e `thermal.heatsink.surrogate`/`monotone`/`trust`.
- `bricks.parts.JointModel` usa encaixe de 0,98 N por pino com faixa assumida (0,5 a 1,5 vez); os resultados da estática dependem dessa hipótese, e `bricks.report.sensitivity` existe para medi-la. `dynamics.friction_creep` documenta um artefato do atrito XPBD.
- `domains3d.get_pinn_batch` devolve alvos de contorno zerados (marcação), não valores físicos; o `STLDomainBatchBuilder` recentraliza a malha, o que quebra funções de valor que assumem coordenadas literais (motivo de existir `analytic_domain_batch_builder`).
- Os surrogates de `thermal.heatsink` só servem para triagem: `load_surrogates` só carrega o que existir em `ARTIFACTS` (`apps/heatsink_sizer/artifacts` ou `PINNEAPPLE_HEATSINK_ARTIFACTS`), e todo projeto recomendado é reverificado pelo motor físico.
- `fitmesh` depende de ferramentas externas opcionais: CadQuery e trimesh para CAD, Gmsh e OpenFOAM (blockMesh, snappyHexMesh, checkMesh) para malhas, FastAPI para a API.
- O docstring de `mesh_gen` avisa que camadas do snappyHexMesh podem colapsar perto de arestas vivas ou folgas finas.
- Geometria livre escrita por LLM roda em processo isolado com lista de imports e timeout, e fica marcada como sem referência publicada.
- Os modelos qualitativos são estimativas baratas (Newtoniano modificado, correlações), feitas para direção e ordem de grandeza; a área projetada é exata só para corpos convexos (limite superior nos demais).
- A otimização topológica usa grades estruturadas de Q4 em 2D; `oc` só trata a restrição de volume, outras restrições exigem `mma`.
- Muitos módulos não têm teste direto pelo inventário: em `fitmesh` (`cad`, `catalog`, `layers`, `mesh_gen`, `mesh_plan`, `report`, `session`), em `qualitative` (`descriptors`, `models`) e em `topology` (`driver`, `filters`, `optimizers`, `problems`); são exercitados via os `__init__` em `tests/test_fitmesh.py`, `tests/test_qualitative_preview.py` e nos testes de topologia.

## 2. `pinneapple_systems`

### Objetivo

`pinneapple_systems` reúne as ferramentas para sistemas físicos com vários componentes que evoluem no tempo. O docstring do pacote lista cinco submódulos: `time_series` (previsão de sinais físicos, antigo `pinneapple_timeseries`), `cosimulation` (co-simulação diferenciável em grafo, antigo `pinneapple_cosim`), `digital_twin` (gêmeo digital com streams de sensores, filtros de Kalman e detecção de anomalias), `component_modeling` (controle, MPC, incerteza, resíduos de PDE e empacotamento ONNX) e `process_components` (modelos de operações unitárias de planta). Há também o módulo `excitation`, com sinais de excitação de atuador.

O problema que o pacote resolve é juntar, num só lugar, modelos aprendidos e modelos físicos de partes diferentes de um sistema. A co-simulação liga nós (redes PyTorch, PINNs com resíduo físico, funções analíticas, caixas-pretas, modelos do `ModelRegistry`, PDEs em SymPy, previsores de séries) por portas nomeadas, executa o grafo no tempo resolvendo laços algébricos e treina tudo junto por retropropagação no tempo. A parte de séries temporais vai do preparo dos dados (imputação, outliers, reamostragem, EDA, auditoria estatística) a modelos (baselines, ML tabular, LSTM/GRU, N-BEATS, TCN, TFT, FNO, decomposição FFT/HHT, ARIMA, modelos fundacionais Chronos e TimesFM), validação temporal sem vazamento, métricas e visualização.

Dentro de `time_series` há dois laboratórios de estudo maiores. O `forecast_lab` define uma interface comum e um protocolo rígido (teste fixado antes, tuning em walk-forward com purga, testes de Diebold-Mariano com correção de comparações múltiplas) para comparar baselines, modelos físicos, modelos de dados e híbridos em séries quase aleatórias; ele usa `synthetic` e `predictability`. O subpacote `enso` aplica esse protocolo à previsão do El Niño, com dados abertos baixados e verificados por hash, osciladores conceituais, um modelo acoplado intermediário e PINNs.

O pacote se apoia no resto do repositório: usa `pinneapple_neural` (arquiteturas, `ModelRegistry`, `Trainer`, `TrainConfig`, `WeightScheduler`, `InversePINN`), `pinneapple_physics` (compilação de `ProblemSpec` em perda física), `pinneapple_simulation.numerical_solvers` (decomposições clássicas), `pinneapple_analysis` (regressão por quantis) e os renderizadores Blender (`pinneapple_blender`, `pinneapple_tools`). A raiz oferece os atalhos `forecast` e `cosim_from_models`, e mantém os aliases antigos `timeseries` e `cosim`.

Esta parte de `pinneapple_systems` cobre quatro submódulos voltados a sistemas em operação. `digital_twin` é o runtime genérico de gêmeo digital (fluxos de sensores, filtros de Kalman, anomalias, prognóstico, padrões industriais) mais a camada de comissionamento `product` (contrato, preflight, backtest, evidência, política de deriva) e duas demos completas (`pipe_loop`, o gêmeo que sabe quando errou, e `recession`, o gêmeo de rio com dados do USGS). `operations` reúne uma planta de referência (tanque aquecido) e três ferramentas sobre ela (runtime de gêmeo com EnKF, detecção física de anomalias, MPC contra PID), usadas pelos apps `twin_runtime`, `anomaly_detector`, `control_lab` e `predictive_twin`. `process_components` traz modelos físicos de operações unitárias independentes (gás real via CoolProp, turbomáquinas, válvulas, trocadores, tubulações, vigas, cinética, golpe de aríete). `sports` aplica física a dados públicos de futebol e compara com baselines só de dados.

Esta parte cobre dois submódulos do `pinneapple_systems`. `component_library` é um registro de componentes de planta nomeados (tubo, trocador de calor, tanque, célula de bateria, bomba, válvula). Cada componente junta uma arquitetura já existente do `ModelRegistry` com uma restrição física opcional (`Physics`), e os componentes são agrupados em toolboxes versionadas e instaláveis. `component_modeling` traz ferramentas que funcionam com qualquer modelo diferenciável, seja qual for a física que ele representa: controle PID e MPC, incerteza (SWAG, ensemble profundo, MC-Dropout), resíduos físicos genéricos por autograd e empacotamento ONNX para borda com runtime sem torch.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 182 | 37.262 | 301 | 515 | 579 (40 arquivos) | 78 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `component_library` | Registro de componentes de planta nomeados, com restrição física, benchmark por tipo e toolboxes instaláveis. | 7 | 15 | 36 |
| `component_modeling` | Controle, incerteza, resíduos físicos e exportação para borda para qualquer modelo diferenciável. | 8 | 12 | 165 |
| `cosimulation` | Motor de co-simulação diferenciável em grafo: nós com portas, conexões, motor por passos de tempo com laços algébricos, gravador, perdas e treinador por BPTT. | 10 | 23 | 151 |
| `digital_twin` | Runtime de gêmeo digital (fluxos, assimilação, anomalias, RUL, reconstrução, padrões AAS/DTDL/NGSI-LD/Ditto/OPC UA), camada de comissionamento `product` e as demos `pipe_loop` e `recession`. | 33 | 129 | 172 |
| `excitation` | Sinais de excitação de atuador em NumPy para identificação de sistemas e treino de surrogates. | 1 | 12 | 6 |
| `operations` | Planta de tanque aquecido e ferramentas de operação sobre ela: runtime de gêmeo EnKF, detecção física de anomalias, controle preditivo, ROM com entradas, serviço ao vivo e contratos de dados. | 10 | 73 | 25 |
| `process_components` | Modelos físicos independentes de operações unitárias para gêmeos de processo (gás real, compressores, válvulas, trocadores, tubos, vigas, reações, fadiga, golpe de aríete). | 17 | 112 | 171 |
| `sports` | Física aplicada a dados públicos de futebol (voo da bola, controle de campo, cinemática, xG), com estudos contra modelos só de dados. | 12 | 87 | 16 |
| `time_series` | Previsão de séries temporais de ponta a ponta: preparo, EDA, modelos, decomposição, validação, métricas, previsibilidade, previsão adaptativa e os estudos `forecast_lab` e `enso`. | 83 | 351 | 151 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_systems`

Raiz do pacote de sistemas multi-componente: importa os submódulos (`time_series`, `cosimulation`, `digital_twin`, `component_modeling`, `process_components`), reexporta seus nomes públicos, cria aliases antigos (`timeseries`, `cosim`) e define os atalhos `forecast` e `cosim_from_models`. [sem teste direto]

<small>`pinneapple_systems/__init__.py` · 280 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `cosim_from_models` | função | `(models_dict, connections) -> 'CoSimGraph'` | Atalho que embrulha cada `nn.Module` de um dicionário num `PINNeAPPleModelNode` com portas fixas `input`/`output` e liga as conexões dadas como pares de strings. |
| `forecast` | função | `(data, horizon, *, model='auto', **kwargs)` | Atalho de previsão: escolhe uma classe por nome (`lstm`, `gru`, `nbeats`, `tft`, `tcn`, `xgboost`; `auto` vira `lstm`), chama `fit(data)` e `predict(horizon)`. |

#### `pinneapple_systems.component_library`

Reexporta `ComponentRegistry`, `ComponentModel`, `Physics`, `Toolbox`, `ToolboxRegistry` e o benchmark, e registra os seis componentes de referência ao ser importado.

<small>`pinneapple_systems/component_library/__init__.py` · 44 linhas · testes: 36</small>

#### `pinneapple_systems.component_library.base`

`ComponentModel`: um `BaseModel` que envolve uma arquitetura registrada, conhece sua `Physics` e oferece `fit` e `evaluate` (RMSE, MAPE, R²) em uma chamada. [sem teste direto pelo inventário]

<small>`pinneapple_systems/component_library/base.py` · 173 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ComponentModel` | classe | `(*, architecture, architecture_kwargs=None, physics=None)` | Componente de planta nomeado: arquitetura existente + `Physics` opcional, com `fit` e `evaluate`. |

#### `pinneapple_systems.component_library.benchmark`

Treina todas as arquiteturas registradas sob um `component_type` e as ranqueia pela perda final de validação. A marca 'placeholder' do inventário é falso positivo: o docstring diz 'not a display placeholder'. [sem teste direto pelo inventário]

<small>`pinneapple_systems/component_library/benchmark.py` · 77 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): placeholder/stub citado no docstring</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ComponentBenchmarkResult` | classe | `(component_type, ranked)` | Ranking das arquiteturas de um tipo de componente, com `best`. |
| `benchmark_component_type` | função | `(component_type, train_loader, val_loader=None, *, epochs=50, lr=0.001, eval_coords=None,…)` | Treina cada arquitetura de um `component_type` e ranqueia pela perda de validação. |

#### `pinneapple_systems.component_library.components`

Seis componentes de referência (tubo, trocador de calor, casco de tanque, célula de bateria, bomba, válvula), cada um com uma combinação diferente de física e arquitetura (`vanilla_pinn` ou `modified_mlp`); a válvula é só por dados. [sem teste direto pelo inventário]

<small>`pinneapple_systems/component_library/components.py` · 126 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PipeVanillaPINN` | classe | `(in_dim=3, out_dim=3, hidden=(128, 128, 128, 128), **kwargs)` | Componente tubo: VanillaPINN com restrição de escoamento incompressível. |
| `HeatExchangerVanillaPINN` | classe | `(in_dim=2, out_dim=1, hidden=(64, 64, 64), **kwargs)` | Componente trocador de calor: VanillaPINN com resíduo de condução de calor. |
| `TankShellVanillaPINN` | classe | `(in_dim=2, out_dim=2, hidden=(64, 64, 64), **kwargs)` | Componente casco de tanque: VanillaPINN com resíduo de elasticidade linear. |
| `BatteryCellVanillaPINN` | classe | `(in_dim=2, out_dim=1, hidden=(64, 64, 64), **kwargs)` | Componente célula de bateria: VanillaPINN com resíduo de difusão de espécies. |
| `PumpModifiedMLP` | classe | `(in_dim=3, out_dim=3, hidden_dim=128, n_layers=6, **kwargs)` | Componente bomba: ModifiedMLP com restrição de escoamento incompressível. |
| `ValveModifiedMLP` | classe | `(in_dim=2, out_dim=1, hidden_dim=64, n_layers=3, **kwargs)` | Componente válvula: ModifiedMLP só com dados, sem restrição física. |

#### `pinneapple_systems.component_library.physics`

`Physics`: liga um componente à sua restrição, reduzida a uma `loss_fn` no formato do `Trainer`, com parâmetros guardados para inspeção. [sem teste direto pelo inventário]

<small>`pinneapple_systems/component_library/physics.py` · 88 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Physics` | classe | `(name, loss_fn, params)` | Restrição física nomeada reduzida a uma `loss_fn`, com parâmetros para inspeção. |

#### `pinneapple_systems.component_library.registry`

`ComponentRegistry` guarda componentes por `component_type`, permitindo várias arquiteturas concorrentes no mesmo tipo, e os constrói por nome. [sem teste direto pelo inventário]

<small>`pinneapple_systems/component_library/registry.py` · 122 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ComponentSpec` | classe | `(component_type, name, cls, physics, description, tags)` | Registro de um componente: tipo, nome, classe, física, descrição e tags. |
| `ComponentRegistry` | classe | `(_REGISTRY)` | Registro central de componentes: registra, lista por tipo, dá a spec e constrói. |
| `register_component` | função | `(*, component_type, name, physics=None, description='', tags=None)` | Decorador de módulo equivalente a `ComponentRegistry.register`. |

#### `pinneapple_systems.component_library.toolbox`

`Toolbox`: pacote nomeado e versionado de tipos de componentes com dependências; `ToolboxRegistry` registra e mantém o conjunto instalado. Registra a 'Fluid Mechanics Toolbox' e a 'Process Equipment Toolbox'. [sem teste direto pelo inventário]

<small>`pinneapple_systems/component_library/toolbox.py` · 110 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Toolbox` | classe | `(name, version, component_types, description, depends_on)` | Pacote nomeado e versionado de tipos de componentes, com dependências e lista de componentes. |
| `ToolboxRegistry` | classe | `(_REGISTRY, _INSTALLED)` | Registro de toolboxes com instalar, desinstalar e consultar as instaladas. |

#### `pinneapple_systems.component_modeling`

Reexporta controle, MPC, SWAG, ensemble, MC-Dropout, resíduos físicos e exportação para borda; o docstring explica que registro, classe base e geração de malha ficam em outros pacotes. [sem teste direto do `__init__`]

<small>`pinneapple_systems/component_modeling/__init__.py` · 60 linhas · **sem teste direto**</small>

#### `pinneapple_systems.component_modeling.bayesian`

`SWAGApproximation`: posterior Gaussiana diagonal sobre os pesos via SWAG, a partir de snapshots ao longo do treino, com previsão por amostragem de pesos.

<small>`pinneapple_systems/component_modeling/bayesian.py` · 127 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SWAGApproximation` | classe | `(model)` | Posterior aproximada SWAG (diagonal) sobre os pesos de um modelo, com `fit` e `predict_with_uncertainty`. |

#### `pinneapple_systems.component_modeling.control`

Controlador PID discreto com limites de saída e um executor de malha fechada para planta diferenciável ou caixa-preta, com ruído de processo e medição e observador opcional.

<small>`pinneapple_systems/component_modeling/control.py` · 113 linhas · testes: 165</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PIDController` | classe | `(kp, ki, kd, setpoint, output_limits=None, bias=0.0)` | PID discreto padrão com bias e limites de saída. |
| `run_closed_loop` | função | `(plant, *, setpoint, kp=1.0, ki=0.1, kd=0.0, steps=50, dt=0.1, process_noise=0.0, measure…)` | Fecha uma malha de controle PID em torno de uma planta e devolve os históricos. |

#### `pinneapple_systems.component_modeling.edge`

Exporta um modelo para ONNX e compacta com `manifest.json` num zip; `EdgeRuntime` roda esse pacote só com onnxruntime e NumPy.

<small>`pinneapple_systems/component_modeling/edge.py` · 96 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EdgeRuntime` | classe | `(zip_path)` | Runtime mínimo de borda que carrega o zip exportado e roda inferência ONNX sem torch. |
| `export_edge_package` | função | `(model, output_dir, *, name, sample_input, opset_version=17, extra_manifest=None) -> Dict…)` | Exporta o modelo para ONNX e compacta com um manifesto num pacote de borda. |

#### `pinneapple_systems.component_modeling.ensemble`

`DeepEnsemble`: N cópias da mesma arquitetura com inicializações diferentes, treinadas separadamente; a média e o desvio entre membros dão a incerteza epistêmica.

<small>`pinneapple_systems/component_modeling/ensemble.py` · 84 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DeepEnsemble` | classe | `(model_factory, n_members=5, base_seed=0)` | Ensemble profundo com `fit` e `predict` devolvendo média e desvio entre membros. |

#### `pinneapple_systems.component_modeling.mc_dropout`

Incerteza por MC-Dropout: várias passagens com dropout ativo, reduzidas a média e desvio; erro se o modelo não tem dropout. [sem teste direto]

<small>`pinneapple_systems/component_modeling/mc_dropout.py` · 41 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `mc_dropout_uncertainty` | função | `(model, x, n_samples=50) -> Tuple[torch.Tensor, torch.Tensor]` | Média e desvio de várias passagens com dropout ativo. |

#### `pinneapple_systems.component_modeling.mpc`

MPC de horizonte deslizante baseado em gradiente em torno de um componente diferenciável. [sem teste direto pelo inventário]

<small>`pinneapple_systems/component_modeling/mpc.py` · 119 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `run_mpc` | função | `(plant, *, in_dim, setpoint, control_input_index=0, observed_output_index=0, horizon=10, …)` | MPC de horizonte deslizante que otimiza a entrada de controle por gradiente a cada passo. |

#### `pinneapple_systems.component_modeling.physics_residuals`

Resíduos de EDP fechados por autograd para qualquer modelo coordenadas→campos: continuidade incompressível, condução de calor, elasticidade linear e difusão de espécies; saídas insuficientes levantam `NotImplementedError`.

<small>`pinneapple_systems/component_modeling/physics_residuals.py` · 126 linhas · testes: 8 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `incompressible_continuity_residual` | função | `(model, coords) -> torch.Tensor` | Resíduo de div(u) = 0 para as primeiras saídas de velocidade. |
| `heat_conduction_residual` | função | `(model, coords, *, k=50.0, Q=0.0) -> torch.Tensor` | Resíduo estacionário k·ΔT + Q. |
| `linear_elasticity_residual` | função | `(model, coords, *, E=210000000000.0, nu=0.3) -> torch.Tensor` | Resíduo de equilíbrio de Navier-Lamé sem força de corpo. |
| `species_diffusion_residual` | função | `(model, coords, *, D=1e-05, k=0.0) -> torch.Tensor` | Resíduo estacionário de difusão-reação D·ΔC − k·C. |

#### `pinneapple_systems.cosimulation`

Motor nativo de co-simulação diferenciável: nós com portas nomeadas, conexões, grafo com análise de topologia, motor por passos de tempo, gravador de trajetórias, perdas e treinador; o `__init__` só reexporta. [sem teste direto do `__init__`]

<small>`pinneapple_systems/cosimulation/__init__.py` · 86 linhas · **sem teste direto**</small>

#### `pinneapple_systems.cosimulation.adapters`

Nós de co-simulação que ligam o motor ao resto do PINNeAPPle: problema físico compilado (`ProblemSpec`), modelos do `ModelRegistry`, PDE simbólica em SymPy e modelos de séries temporais.

<small>`pinneapple_systems/cosimulation/adapters.py` · 432 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PINNeProblemNode` | classe | `(name, model, spec, compiled_fn, coord_ports, field_ports, extra_input_ports=None, physic…)` | Nó apoiado num `ProblemSpec` do PINNeAPPle e na perda física compilada por `compile_problem`; separa portas de coordenadas e de campos e expõe `physics_loss`. |
| `PINNeAPPleModelNode` | classe | `(name, model, input_ports, output_ports, batch_port_map=None) -> None` | Embrulha qualquer `BaseModel` (ou `nn.Module`) do `ModelRegistry` como nó, mapeando portas de entrada e saída para o batch do modelo; `from_registry` constrói pelo nome. |
| `SymbolicPDENode` | classe | `(name, model, symbolic_pde, coord_ports, field_ports, physics_weight=1.0) -> None` | Nó cujo resíduo físico é uma expressão SymPy convertida em resíduo com autograd sobre as portas de coordenadas e campos. |
| `TimeSeriesCoSimNode` | classe | `(name, model, input_len, horizon, extra_input_ports=None, extra_output_ports=None) -> None` | Embrulha um `TSModelBase` (LSTM, TFT, FFT+LSTM...) como nó: recebe uma janela de `input_len` e devolve a previsão de `horizon` passos. |

#### `pinneapple_systems.cosimulation.connection`

Define `Connection`, a aresta imutável `no.porta -> no.porta` do grafo de co-simulação, com parser de strings. [sem teste direto]

<small>`pinneapple_systems/cosimulation/connection.py` · 81 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Connection` | classe | `(src_node, src_port, dst_node, dst_port)` | Aresta dirigida e imutável entre a porta de saída de um nó e a porta de entrada de outro; `parse` lê strings `no.porta`. |

#### `pinneapple_systems.cosimulation.engine`

`CoSimEngine`: executa o grafo passo a passo no tempo, na ordem das componentes fortemente conexas, iterando laços algébricos com Gauss-Seidel ou Jacobi e preservando o grafo do autograd.

<small>`pinneapple_systems/cosimulation/engine.py` · 257 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CoSimEngine` | classe | `(graph, recorder=None, loop_solver='gauss_seidel', max_iter=50, tol=1e-06, retain_graph=F…)` | Motor de co-simulação por passos de tempo sobre um `CoSimGraph`: inicializa portas, executa cada passo (com iteração de laço até `tol` ou `max_iter`) e grava no `TrajectoryRecorder`. |

#### `pinneapple_systems.cosimulation.graph`

`CoSimGraph`: registro de nós e conexões com validação de portas, ordem de execução em cache e detecção de ciclos e laços algébricos.

<small>`pinneapple_systems/cosimulation/graph.py` · 187 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CoSimGraph` | classe | `() -> None` | Registro de nós e conexões com cache de topologia: ordem de execução, ciclos, laços algébricos, conexões de entrada/saída e parâmetros treináveis. |

#### `pinneapple_systems.cosimulation.losses`

Perdas compostas para treinar grafos de co-simulação: ajuste a dados, resíduos físicos dos nós PINN e consistência entre portas conectadas.

<small>`pinneapple_systems/cosimulation/losses.py` · 225 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DataLoss` | classe | `(weight=1.0) -> None` | Erro quadrático médio entre valores previstos nas portas e observações. |
| `PhysicsLoss` | classe | `(weight=1.0) -> None` | Soma dos resíduos físicos (`physics_loss`) de todos os nós PINN do grafo. |
| `CouplingLoss` | classe | `(weight=0.1) -> None` | Penaliza inconsistências entre os valores das portas ligadas por uma conexão. |
| `CoSimLoss` | classe | `(data_weight=1.0, physics_weight=1.0, coupling_weight=0.1) -> None` | Perda composta e ponderada: dados + física + acoplamento. |

#### `pinneapple_systems.cosimulation.node`

Abstrações de nó: base abstrata `CoSimNode` e nós concretos para `nn.Module`, função Python/NumPy, PINN com resíduo físico e caixa-preta sem gradiente.

<small>`pinneapple_systems/cosimulation/node.py` · 291 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CoSimNode` | classe | `(name, input_ports, output_ports) -> None` | Classe base abstrata de todo nó: nome, portas de entrada e saída, `step`, `reset`, `physics_loss` e `parameters`. |
| `TorchNode` | classe | `(name, model, input_ports, output_ports) -> None` | Nó que embrulha um `nn.Module`; os gradientes passam por `step()`. |
| `AnalyticalNode` | classe | `(name, fn, input_ports, output_ports) -> None` | Nó que embrulha uma função Python/NumPy pura; as saídas saem desligadas do autograd. |
| `PINNNode` | classe | `(name, model, input_ports, output_ports, physics_fn, physics_weight=1.0) -> None` | `TorchNode` com uma função de resíduo físico e peso próprio, usada pela `PhysicsLoss`. |
| `BlackBoxNode` | classe | `(name, fn, input_ports, output_ports) -> None` | Nó opaco (por exemplo um simulador externo) totalmente desligado do autograd. |
| `split_output_ports` | função | `(out, ports) -> Dict[str, torch.Tensor]` | Divide a última dimensão de um tensor de saída igualmente entre as portas dadas. |

#### `pinneapple_systems.cosimulation.recorder`

Gravação de trajetórias no tempo por (nó, porta) durante uma execução de co-simulação, com acesso como listas, arrays ou tensores.

<small>`pinneapple_systems/cosimulation/recorder.py` · 177 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Trajectory` | classe | `(node, port) -> None` | Série temporal de um único sinal (nó, porta), com tempos e valores como arrays ou tensor. |
| `TrajectoryRecorder` | classe | `() -> None` | Grava saídas de portas com carimbo de tempo para vários nós; permite escolher o que observar e resumir as trajetórias. |

#### `pinneapple_systems.cosimulation.trainer`

`CoSimTrainer`: treina por BPTT (desenrolando `n_unroll` passos) todos os parâmetros do grafo com `TrainConfig`, `WeightScheduler`, parada antecipada e checkpoint de `pinneapple_neural`.

<small>`pinneapple_systems/cosimulation/trainer.py` · 331 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CoSimTrainer` | classe | `(graph, engine, criterion=None, weight_scheduler_cfg=None, early_stopping=None, checkpoin…)` | Treina um grafo de co-simulação diferenciável com a pilha de treino do PINNeAPPle (BPTT, balanceamento de pesos, parada antecipada, checkpoint) e guarda o histórico. |

#### `pinneapple_systems.cosimulation.utils`

Algoritmos de grafo para a topologia da co-simulação: Tarjan (componentes fortemente conexas), ordem por condensação e ordenação topológica. [sem teste direto]

<small>`pinneapple_systems/cosimulation/utils.py` · 180 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `tarjan_scc` | função | `(adj) -> List[List[str]]` | Encontra as componentes fortemente conexas de um grafo de adjacência com o algoritmo de Tarjan. |
| `condensation_order` | função | `(adj) -> Tuple[List[List[str]], List[List[str]]]` | Calcula a ordem de execução pela condensação das componentes fortemente conexas e devolve também os laços. |
| `topological_sort` | função | `(adj) -> List[str]` | Ordenação topológica para grafos acíclicos (DAG). |

#### `pinneapple_systems.digital_twin`

Runtime de gêmeo digital sobre modelos substitutos: reexporta `DigitalTwin`, estados, fluxos de sensores, filtros de Kalman, detectores de anomalia, pipeline de tempo real, reconstrução de sinais, condicionamento de telemetria e prognóstico (RUL).

<small>`pinneapple_systems/digital_twin/__init__.py` · 121 linhas · testes: 15</small>

#### `pinneapple_systems.digital_twin.assimilation`

Subpacote de assimilação de dados: só reexporta `ExtendedKalmanFilter` e `EnsembleKalmanFilter`. [sem teste direto]

<small>`pinneapple_systems/digital_twin/assimilation/__init__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_systems.digital_twin.assimilation.kalman`

[shim] Shim de compatibilidade: a implementação real de `ExtendedKalmanFilter`/`EnsembleKalmanFilter` mudou para `pinneapple_analysis.state_estimation.kalman`; aqui só se reexportam os nomes para os chamadores antigos.

<small>`pinneapple_systems/digital_twin/assimilation/kalman.py` · 20 linhas · testes: 157 · palavras-chave no código (heurística; o texto acima diz o status real): shim ou obsoleto</small>

#### `pinneapple_systems.digital_twin.conditioning`

Condicionamento causal de telemetria, leitura a leitura: checagem de faixa, rejeição de outliers (Hampel causal com predição de um passo), limite de taxa, filtro passa-baixa e preenchimento de lacunas, com flags do que foi feito. [sem teste direto] — `tests/test_telemetry_conditioning.py` usa via `pinneapple_systems.digital_twin`.

<small>`pinneapple_systems/digital_twin/conditioning.py` · 195 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SignalProfile` | classe | `(kind, tau_s, hampel_window, hampel_k, max_consecutive_outliers, max_rate, valid_range, m…)` | Configuração de como condicionar um tipo de sinal (constante de tempo, janela e limiar de Hampel, taxa máxima, faixa válida, lacuna máxima); há presets em `PROFILES`. |
| `ConditionedSample` | classe | `(t, raw, value, flags)` | Uma leitura condicionada: tempo, valor bruto, valor final e flags; `ok` diz se o valor é finito e não velho. |
| `StreamingConditioner` | classe | `(profile='generic', **overrides)` | Condiciona um sinal, uma leitura por vez, de forma causal (custo O(janela) por leitura); `update` processa, `fill` preenche lacunas. |
| `TelemetryConditioner` | classe | `(profiles=None, default='generic')` | Conjunto de condicionadores por sensor e campo para um gêmeo inteiro; `condition_observation` limpa uma `Observation`. |

#### `pinneapple_systems.digital_twin.io`

Subpacote de entrada/saída: reexporta `Sensor`, `SensorRegistry` e os adaptadores de fluxo de `stream`.

<small>`pinneapple_systems/digital_twin/io/__init__.py` · 26 linhas · testes: 8</small>

#### `pinneapple_systems.digital_twin.io.mqtt_broker`

Broker MQTT 3.1.1 embutido e pequeno (QoS 0 e 1, mensagens retidas, curingas `+`/`#`, usuário/senha opcional) para demos, testes e borda de um ativo; sem persistência, sem QoS 2, sem will e sem TLS.

<small>`pinneapple_systems/digital_twin/io/mqtt_broker.py` · 263 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MQTTBroker` | classe | `(host='127.0.0.1', port=1883, username=None, password=None, max_packet=256 * 1024)` | Broker MQTT embutido que roda numa thread com laço de eventos próprio; `start`/`stop`, publica e entrega a assinantes. |
| `topic_matches` | função | `(filt, topic) -> bool` | Testa se um tópico casa com um filtro MQTT (`+` um nível, `#` o resto; tópicos `$` exigem primeiro nível explícito). |

#### `pinneapple_systems.digital_twin.io.sensors`

Cadastro de sensores do gêmeo: posição, calibração (offset e escala), ruído e campos; cada `Sensor` converte uma leitura bruta em `Observation` calibrada.

<small>`pinneapple_systems/digital_twin/io/sensors.py` · 151 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Sensor` | classe | `(sensor_id, coords, field_names, calibration_offset, calibration_scale, noise_std, transf…)` | Um sensor físico: id, coordenadas, campos, calibração, ruído e transformação; `read` gera uma `Observation` calibrada. |
| `SensorRegistry` | classe | `() -> None` | Cadastro de todos os sensores do gêmeo: adiciona, remove, lista ativos, lê em lote e monta o vetor de observações. |

#### `pinneapple_systems.digital_twin.io.stream`

Adaptadores de fluxo de dados em tempo real (arquivo JSONL/CSV/Parquet, MQTT, HTTP, Kafka, OPC UA, Modbus TCP e um fluxo sintético) que emitem `Observation` numa fila consumida pelo `DigitalTwin`; MQTT, Kafka, OPC UA e Modbus pedem bibliotecas opcionais.

<small>`pinneapple_systems/digital_twin/io/stream.py` · 576 linhas · testes: 172</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BaseStream` | classe | `(sensor_id, field_names) -> None` | Classe base abstrata dos adaptadores de fluxo: liga a fila de saída e define `start`/`stop`. |
| `FileWatchStream` | classe | `(path, sensor_id, field_names, *, poll_interval=1.0, format='jsonl') -> None` | Consulta periodicamente um arquivo JSON-lines (ou CSV/Parquet, com pandas) e emite as linhas novas. |
| `MQTTStream` | classe | `(broker, topic, sensor_id, field_names, *, port=1883, keepalive=60, username=None, passwo…)` | Assina um tópico MQTT e emite observações (requer `paho-mqtt`). |
| `HTTPPollStream` | classe | `(url, sensor_id, field_names, *, poll_interval=5.0, headers=None, transform=None) -> None` | Consulta um endpoint REST por GET em intervalos e emite observações. |
| `KafkaStream` | classe | `(bootstrap_servers, topic, sensor_id, field_names, *, group_id='pinneapple_dt', auto_offs…)` | Lê mensagens de um tópico Apache Kafka (requer `kafka-python`). |
| `OPCUAStream` | classe | `(server_url, node_ids, sensor_id, field_names, *, poll_interval=1.0) -> None` | Consulta tags de um servidor OPC UA em intervalos (requer `asyncua`). |
| `ModbusStream` | classe | `(host, port, register_map, sensor_id, field_names, *, poll_interval=1.0, device_id=1) -> …)` | Consulta registradores holding de um servidor Modbus TCP (requer `pymodbus`). |
| `MockStream` | classe | `(sensor_id, field_names, generator_fn, *, tick_interval=0.1, coords=None) -> None` | Fluxo sintético a partir de uma função geradora, para testes e simulação. |

#### `pinneapple_systems.digital_twin.io.usgs`

Cliente da API pública "Instantaneous Values" do USGS (vazão, nível de rios dos EUA), com cache curto em memória; falhas levantam `USGSDataError`, nada é preenchido. Os dados são baixados em tempo de execução.

<small>`pinneapple_systems/digital_twin/io/usgs.py` · 100 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `USGSDataError` | classe | `()` | Erro quando o serviço do USGS não responde ou não devolve dados utilizáveis. |
| `Series` | classe | `(site_id, site_name, parameter_cd, unit, timestamps, values, fetched_at, from_cache)` | Série temporal baixada de um posto: id e nome do posto, parâmetro, unidade, tempos, valores e se veio do cache. |
| `monitoring_url` | função | `(site_id) -> str` | URL da página pública de um posto do USGS, para o leitor conferir os dados. |
| `parse_iv` | função | `(payload, site_id, parameter_cd) -> Series` | Extrai a primeira série de um JSON da API IV, pulando leituras malformadas e ordenando por tempo. |
| `fetch_iv` | função | `(site_id, parameter_cd=DISCHARGE, period='P5D', timeout_s=10.0, cache_ttl_s=90.0) -> Seri…)` | Baixa um parâmetro de um posto num período ISO 8601 recente, com timeout e cache em memória. |

#### `pinneapple_systems.digital_twin.monitoring`

Subpacote de monitoramento: reexporta os detectores de anomalia e o `AnomalyMonitor`. [sem teste direto]

<small>`pinneapple_systems/digital_twin/monitoring/__init__.py` · 17 linhas · **sem teste direto**</small>

#### `pinneapple_systems.digital_twin.monitoring.anomaly`

Detecção de anomalias que compara previsões com observações: limiar fixo, z-score móvel, distância de Mahalanobis e checagem dos resíduos de inovação do filtro, combinados num monitor composto.

<small>`pinneapple_systems/digital_twin/monitoring/anomaly.py` · 415 linhas · testes: 166</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AnomalyEvent` | classe | `(timestamp, sensor_id, field_name, observed, predicted, score, detector, metadata)` | Registro de uma anomalia: tempo, sensor, campo, valor observado e previsto, escore e detector. |
| `ThresholdDetector` | classe | `(thresholds, rel_thresholds=None, eps=1e-08) -> None` | Detector simples por limiar absoluto (e opcionalmente relativo) por campo. |
| `ZScoreDetector` | classe | `(z_threshold=3.0, window_size=100, min_samples=10) -> None` | Detector por z-score numa janela móvel de referência. |
| `MahalanobisDetector` | classe | `(threshold=3.0, field_order=None) -> None` | Detector multivariado por distância de Mahalanobis, com covariância atualizável. |
| `ResidualDetector` | classe | `(z_threshold=3.0, window_size=100, min_samples=10) -> None` | Checa os resíduos de inovação do filtro de assimilação com z-score móvel. |
| `AnomalyMonitor` | classe | `() -> None` | Monitor composto que roda vários detectores em sequência e guarda os eventos recentes. |

#### `pinneapple_systems.digital_twin.pipe_loop`

Demo 3 do Autonomous Physics Lab: gêmeo de um circuito de água bombeada e aquecida (cotovelo e junção em T) que distingue processo mudado, sensor errado e modelo fora do domínio; reexporta `run_mission`, `run_benchmark`, a planta e o diagnóstico.

<small>`pinneapple_systems/digital_twin/pipe_loop/__init__.py` · 31 linhas · testes: 7</small>

#### `pinneapple_systems.digital_twin.pipe_loop.agent`

Missão do gêmeo como ferramentas controladas de `pinneapple_autolab` (montar o modelo rápido, calibrar alarmes, monitorar, pedir nova simulação, benchmark, relatório de confiança, entrega), com planejador determinístico ou LLM. [sem teste direto] — `tests/test_autolab_twin.py` usa via pacote.

<small>`pinneapple_systems/digital_twin/pipe_loop/agent.py` · 217 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `t_read_mission` | função | `(state, text) -> dict[str, Any]` | Ferramenta que lê o pedido em texto, monta o `Loop` e marca quais hipóteses (processo, sensor, domínio, benchmark) foram pedidas. |
| `t_build_twin` | função | `(state, n_train=120, n_test=40, seed=0) -> dict[str, Any]` | Ferramenta que ajusta o modelo rápido (`Surrogate`) e mede o erro em pontos de operação separados (hold-out) e a cobertura de 2 desvios. |
| `t_calibrate` | função | `(state, n_runs=12) -> dict[str, Any]` | Ferramenta que calibra os limiares do `Diagnoser` em corridas saudáveis. |
| `t_monitor` | função | `(state) -> dict[str, Any]` | Ferramenta que roda e diagnostica os casos de vitrine (`SHOWCASE`) e conta os acertos. |
| `t_request_simulation` | função | `(state, n_new=40, widen=0.25) -> dict[str, Any]` | Para cada caso diagnosticado fora do domínio, roda a referência, compara com as leituras, amplia o envelope, adiciona soluções novas, reajusta o modelo e diagnostica de novo. |
| `t_benchmark` | função | `(state, n_per_kind=20, n_calibration=12, workers=4) -> dict[str, Any]` | Ferramenta que roda o benchmark controlado e devolve acurácia, número de casos e casos com alarme falso. |
| `t_trust` | função | `(state) -> dict[str, Any]` | Ferramenta que monta o relatório de confiança e sua decisão. |
| `t_deliver` | função | `(state, out_dir='out', render=True, samples=16) -> dict[str, Any]` | Ferramenta que gera figuras, cena 3-D, visualizador, renders e o relatório. |
| `toolbox` | função | `() -> Toolbox` | Monta o `Toolbox` `pipe_twin` com as oito ferramentas e seus resumos. |
| `template_plan` | função | `(text, *, out_dir='out', render=True, n_per_kind=20, workers=4) -> TemplatePlanner` | Plano determinístico (`TemplatePlanner`) com a ordem padrão das ferramentas. |
| `run_mission` | função | `(text=DEFAULT_TEXT, *, out_dir='out', planner='template', render=True, n_per_kind=20, wor…)` | Executa a missão inteira com planejador `template` ou `llm` e devolve o `Run` com o estado de cada passo. |

#### `pinneapple_systems.digital_twin.pipe_loop.benchmark`

Casos controlados com resposta conhecida (normal, processo, sensor, domínio) e a matriz de confusão do diagnóstico, com escores de localização, alarmes falsos e atrasos. [sem teste direto]

<small>`pinneapple_systems/digital_twin/pipe_loop/benchmark.py` · 123 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `make_cases` | função | `(n_per_kind=20, seed=0) -> list[Scenario]` | Sorteia cenários controlados das quatro classes (normal, processo, sensor, domínio) com semente. |
| `run_benchmark` | função | `(loop=None, *, n_per_kind=20, n_calibration=12, seed=0, n_train=120, workers=4, log=None)…)` | Ajusta o modelo rápido, calibra em corridas saudáveis, diagnostica cada caso e devolve matriz de confusão, escores de isolamento, alarmes falsos, atrasos e tempos. |

#### `pinneapple_systems.digital_twin.pipe_loop.diagnose`

Como o gêmeo sabe que errou: resíduos físicos estruturados, modelo rápido POD + processo gaussiano, EKF de perda extra por tubo e domínio de validade, combinados num diagnóstico por janela. [sem teste direto]

<small>`pinneapple_systems/digital_twin/pipe_loop/diagnose.py` · 313 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Residuals` | classe | `(loop)` | Resíduos físicos de um vetor de leituras (massa no T, Darcy por tubo, bomba, válvulas, energia), em unidades de sensor, com seu desvio de ruído e jacobiano. |
| `Surrogate` | classe | `(n_train, seed, model, lo, hi)` | Modelo rápido POD + processo gaussiano das dez leituras sobre `NORMAL_BOX`, com desvio preditivo e excesso fora do envelope. |
| `LossTracker` | classe | `(loop, q_walk=0.02, k0_sigma=0.5)` | EKF do coeficiente de perda extra de cada tubo (passeio aleatório), observado pela queda de pressão medida dada a vazão medida. |
| `Diagnoser` | classe | `(loop, surrogate, window, thresholds, ood_threshold)` | Transforma evidência em janela numa hipótese (normal, processo, sensor, domínio) com alvo; limiares calibrados em corridas saudáveis. |

#### `pinneapple_systems.digital_twin.pipe_loop.plant`

A planta e seu simulador de referência quase estacionário: bomba (leis de afinidade), Darcy-Weisbach/Colebrook, perdas localizadas, balanço de energia, dez sensores com ruído e falhas injetadas (obstrução, deriva, ruído).

<small>`pinneapple_systems/digital_twin/pipe_loop/plant.py` · 258 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Pipe` | classe | `(length, diameter, roughness, k_minor, ua_w_k)` | Um tubo: comprimento, diâmetro, rugosidade, perda localizada e UA; calcula área, perda de carga e vazão a partir da perda. |
| `Loop` | classe | `(pipe_A, pipe_B, pipe_C, h0_m, k_pump, kv0_B, kv0_C, z_B, z_C, t_room_c, p_atm_head_m, wa…)` | Geometria e máquinas do circuito (três tubos, bomba, válvulas, cotas, ambiente) em SI. |
| `Faults` | classe | `(k_extra, bias, noise_gain)` | O que está errado num instante: perda extra por tubo, viés e ganho de ruído por sensor (tudo zero = saudável). |
| `Scenario` | classe | `(name, kind, seed, duration_s, dt_s, onset_s, ramp_s, target, magnitude, mode)` | Caso de teste controlado: histórico de operação e, opcionalmente, uma falha com início e rampa. |
| `water` | função | `(t_c) -> tuple[float, float]` | Densidade e viscosidade dinâmica da água numa temperatura em °C (tabela de White). |
| `steady_state` | função | `(loop, u, k_extra=None) -> dict[str, float]` | Estado verdadeiro da planta para entradas dadas: vazões, cargas, pressões e temperaturas. |
| `readings` | função | `(state) -> np.ndarray` | Valores verdadeiros dos sensores em unidades de sensor (bar, m3/h, °C) na ordem `SENSORS`. |
| `quasi_steady_valid` | função | `(loop, stroke_s, factor=10.0) -> bool` | Diz se o modelo quase estacionário vale para um curso de válvula (curso ≥ fator × 2L/a). |
| `operating_history` | função | `(sc) -> tuple[np.ndarray, dict[str, np.ndarray]]` | Movimentos lentos do operador dentro de `NORMAL_BOX` (soma de senos); num cenário de domínio leva uma entrada para fora da caixa. |
| `run_plant` | função | `(loop, sc) -> dict[str, Any]` | Simula um cenário: entradas, valores verdadeiros, leituras com ruído e a falha aplicada a cada passo. |
| `with_loop` | função | `(loop, **kw) -> Loop` | Cópia de um `Loop` com campos trocados. |

#### `pinneapple_systems.digital_twin.pipe_loop.report`

Entregáveis da demo: figuras de sala de controle, matriz de confusão, relatório de confiança, GIF, cena 3-D (visualizador web, glTF, USD, renders Blender opcionais) e relatório em texto. [sem teste direto]

<small>`pinneapple_systems/digital_twin/pipe_loop/report.py` · 341 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `trust_report` | função | `(state) -> TrustReport` | Monta o relatório de confiança (`TrustReport`) com checagens e decisão a partir do estado das ferramentas. |
| `control_room` | função | `(case, path, *, title='', cursor=None) -> str` | Figura de sala de controle de um caso: leituras, previsões e evidência ao longo do tempo (PNG). |
| `figure_confusion` | função | `(b, path) -> str` | Figura da matriz de confusão do benchmark. |
| `figure_request` | função | `(rs, path) -> str \| None` | Figura do pedido de nova simulação (antes e depois de ampliar o modelo), ou `None` se não houve. |
| `impact_gif` | função | `(case, path, n=14) -> str` | GIF 2-D do impacto de um caso ao longo do tempo (usa imageio). |
| `render_3d` | função | `(case, out_dir, *, render, samples, size=(1280, 720)) -> dict[str, Any]` | Gera visualizador web, glTF e USD da cena; renders Blender só se `render` e se `bpy` estiver instalado. |
| `deliver` | função | `(state, out_dir, *, render=True, samples=16) -> dict[str, Any]` | Gera todos os entregáveis (figuras, GIF, 3-D, `trust_report.json`, relatório) numa pasta. |
| `write_report` | função | `(state, out_dir, files, env) -> str` | Escreve o relatório em texto com os números das ferramentas e o ambiente da execução. |

#### `pinneapple_systems.digital_twin.pipe_loop.scene`

Cena 3-D esquemática do circuito, quadro a quadro: tubos coloridos pela pressão de referência, linhas de corrente do cotovelo por lei de potência 1/7 e sensores coloridos pelo veredito. [sem teste direto]

<small>`pinneapple_systems/digital_twin/pipe_loop/scene.py` · 181 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `tube` | função | `(path, r=R_PIPE, n_theta=20, n_seg=40)` | Malha triangulada de um tubo ao longo de uma polilinha, com a fração de comprimento de cada vértice. |
| `elbow_streamlines` | função | `(v_bulk, n_r=3, n_t=8)` | Linhas pelo cotovelo e trechos vizinhos em alguns raios, com a velocidade da lei de potência 1/7. |
| `loop_scene` | função | `(run, diagnosis, *, n_frames=24, title='')` | Cena animada de uma corrida monitorada: pressões nas paredes, veredito nos sensores e HUD com a evidência. |

#### `pinneapple_systems.digital_twin.product`

Camada de comissionamento do gêmeo como produto: contrato (`TwinSpec`), contrato do modelo, estimador EnKF de estado latente, preflight + backtest, evidência em arquivo e política de deriva; reexporta tudo.

<small>`pinneapple_systems/digital_twin/product/__init__.py` · 83 linhas · testes: 24</small>

#### `pinneapple_systems.digital_twin.product.assimilation`

Estimador do gêmeo: EnKF da plataforma sobre o estado latente que o modelo não conhece (incrustação, espessura, viés), com o próprio modelo como operador de observação e banda de 95 % pela dispersão do ensemble. [sem teste direto] — coberto via `pinneapple_systems.digital_twin.product`.

<small>`pinneapple_systems/digital_twin/product/assimilation.py` · 418 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TwinEstimate` | classe | `(theta, theta_std, innovation, n_obs)` | Um instante de análise: estado latente, sua dispersão, inovação e número de observações. |
| `LatentStateTwin` | classe | `(model, spec, *, n_ens=64, inflation=1.02, seed=0) -> None` | Estima o estado latente de um ativo a partir da spec e do modelo; dá banda e média nos sensores de hold-out, tendência e projeção de cruzamento de limite. |

#### `pinneapple_systems.digital_twin.product.commission`

Os dois portões antes de o gêmeo rodar: `preflight` (checagens baratas do contrato e do histórico) e `backtest` (replay do histórico medido contra sensores nunca assimilados).

<small>`pinneapple_systems/digital_twin/product/commission.py` · 588 linhas · testes: 24</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `read_timestamp` | função | `(frame) -> float` | Lê o carimbo de tempo de um frame, sob qualquer das chaves convencionais. |
| `read_readings` | função | `(frame, spec) -> dict[str, float]` | Valores de sensores presentes num frame, ignorando chaves que não são sensores. |
| `preflight` | função | `(spec, history) -> CheckReport` | Checa se o gêmeo pode ser comissionado (unidades, observabilidade, canal travado, cobertura da faixa) antes de gastar um backtest. |
| `backtest` | função | `(spec, model, history, *, n_ens=64, inflation=1.02, seed=0, warmup_steps=0, calibrate_fra…)` | Repassa o histórico pelo gêmeo e mede contra os sensores de hold-out (erro, viés, cobertura da banda, antecedência de alarme). |

#### `pinneapple_systems.digital_twin.product.evidence`

Contêineres da evidência: itens e relatório de checklist, métricas por campo com sua fonte, relatório de rastreamento e o arquivo `TwinEvidence` com a lista do que não foi verificado; não calcula física. [sem teste direto]

<small>`pinneapple_systems/digital_twin/product/evidence.py` · 284 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CheckItem` | classe | `(check_id, status, title, detail, why, fix, source)` | Uma linha do checklist: id, status, título, detalhe, porquê, correção e fonte. |
| `CheckReport` | classe | `(title, items)` | Checklist ordenado com veredito: qualquer `FAIL` bloqueia o comissionamento. |
| `FieldMetrics` | classe | `(field_name, unit, n, rmse, mae, bias, p95_abs, coverage, span, window_s, coverage_raw, b…)` | Concordância entre gêmeo e realidade num campo, em observações nunca assimiladas, com o número de comparações e a janela. |
| `TrackingReport` | classe | `(metrics, gates, breaches, alarm_lead_s, forecast_error_s, final_latent, n_steps)` | Métricas por campo, veredito dos portões, violações e antecedência dos alarmes. |
| `TwinEvidence` | classe | `(asset_id, spec_digest, model_id, created_unix, preflight, tracking, decisions, unverifie…)` | Tudo o que um gêmeo comissionado produz num arquivo, ligado ao digest da spec, incluindo o que não foi verificado. |

#### `pinneapple_systems.digital_twin.product.model`

Contrato do modelo `model(X, theta) -> {campo: array}` e o adaptador `BiasModel` para encaixar substitutos que ignoram `theta`. [sem teste direto]

<small>`pinneapple_systems/digital_twin/product/model.py` · 155 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TwinModelError` | classe | `()` | Erro quando o modelo não cumpre o contrato do gêmeo. |
| `BiasModel` | classe | `(model, field_names) -> None` | Envolve um `model(X)` simples para cumprir o contrato, usando `theta` como viés de saída. |
| `as_array_dict` | função | `(out, field_names) -> dict[str, np.ndarray]` | Normaliza a saída de um modelo para `{campo: array (n,) float32}`. |
| `ensure_twin_model` | função | `(model, field_names, n_latent)` | Devolve um chamável que cumpre o contrato, envolvendo o modelo quando preciso. |

#### `pinneapple_systems.digital_twin.product.policy`

Política de deriva: consistência da banda de 95 % do gêmeo numa janela móvel convertida em modo NORMAL → WATCH → ALARM → RECALIBRATE com histerese; nunca re-treina sozinho. [sem teste direto]

<small>`pinneapple_systems/digital_twin/product/policy.py` · 166 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TwinMode` | classe | `()` | Enum dos modos do gêmeo: NORMAL, WATCH, ALARM, RECALIBRATE. |
| `DriftPolicy` | classe | `(window, target_coverage, hard_break, max_sustained, recalibrate_after, hysteresis)` | Limiares da escada de consistência (janela, cobertura alvo, quebra forte, sustentação, recalibração, histerese). |
| `DriftMonitor` | classe | `(policy, _window, _mode, _off_run, _alarm_run, _ok_run, decisions)` | Consistência móvel da banda do gêmeo convertida em modo com histerese; registra as decisões. |

#### `pinneapple_systems.digital_twin.product.runner`

Superfície do produto: `commission` (preflight + backtest + evidência), `run_live`, `describe` (padrões industriais) e `export_scene` (cena `pinneapple_twin3d`). [sem teste direto] — coberto via pacote `product`.

<small>`pinneapple_systems/digital_twin/product/runner.py` · 346 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RunResult` | classe | `(n_steps, n_observed, n_holdout, final_mode, coverage, latent, modes_seen, decisions)` | O que aconteceu na execução: passos, observados, hold-out, modo final, cobertura, estado latente e decisões. |
| `commission` | função | `(spec, model, history, *, n_ens=64, inflation=1.02, seed=0, warmup_steps=0, calibrate_fra…)` | Roda preflight e backtest e devolve o arquivo de evidência; determinístico para uma semente. |
| `run_live` | função | `(spec, model, frames, *, n_ens=64, inflation=1.02, seed=0, policy=None, evidence=None, th…)` | Assimila uma sequência de frames, monitora a consistência e registra o que o gêmeo decidiu. |
| `describe` | função | `(spec, latent, fields=None)` | Descreve o gêmeo para os padrões industriais de `standards` (AAS, DTDL, NGSI-LD, Ditto, OPC UA). |
| `export_scene` | função | `(spec, frames, folder, *, mesh=None, field_on_mesh=None, times=None) -> str` | Escreve uma cena `pinneapple_twin3d` do gêmeo: sensores com séries medidas e envelopes (malha opcional via trimesh). |

#### `pinneapple_systems.digital_twin.product.spec`

O contrato do gêmeo (`TwinSpec`): campos, sensores, modelo, portões de aceitação e um digest; carrega e grava YAML (PyYAML opcional). [sem teste direto]

<small>`pinneapple_systems/digital_twin/product/spec.py` · 438 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SpecError` | classe | `()` | Erro quando a spec não é utilizável (afirmação faltando, contradição, campo inobservável). |
| `FieldSpec` | classe | `(name, unit, kind, lo, hi, alarm_above, alarm_below)` | Uma grandeza que o gêmeo estima: nome, unidade, tipo, faixa e limites de alarme. |
| `SensorSpec` | classe | `(sensor_id, unit, field, lo, hi, at, max_gap_s, noise_sigma, holdout, driver)` | Uma medição disponível ao gêmeo, com envelope, ruído, se é hold-out e se é driver. |
| `ModelSpec` | classe | `(model_id, version, accuracy_floor, validity)` | O modelo por trás do gêmeo, sua versão, a precisão que promete e a validade. |
| `AcceptanceGates` | classe | `(max_rmse_frac, max_bias_frac, min_coverage, min_alarm_lead_s)` | Limiares que decidem se o gêmeo entra em operação (erro e viés relativos, cobertura mínima, antecedência mínima de alarme). |
| `TwinSpec` | classe | `(asset_id, asset_name, asset_type, fields, sensors, model, asset_unit, coord_names, sampl…)` | Definição completa e checável de um gêmeo; tem digest, leitura/escrita em dict e YAML e consultas sobre sensores e campos. |

#### `pinneapple_systems.digital_twin.prognostics`

Manutenção preditiva: estimativa de vida útil remanescente (RUL) por tendência linear ou exponencial de um indicador de saúde, com intervalo de confiança e aviso de extrapolação pouco confiável.

<small>`pinneapple_systems/digital_twin/prognostics.py` · 628 linhas · testes: 15</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HealthIndicatorHistory` | classe | `(timestamps, values, source)` | Série temporal escalar de um indicador de saúde/degradação, montada de arrays ou de um `DigitalTwin`. |
| `RULEstimate` | classe | `(estimated_rul, rul_confidence_interval, failure_time_estimate, degradation_model_used, f…)` | Resultado da estimativa de RUL: vida remanescente, intervalo de confiança, tempo de falha, modelo usado, qualidade do ajuste e se a extrapolação é confiável. |
| `estimate_rul` | função | `(health_indicator_series, timestamps=None, failure_threshold=None, model='linear', confid…)` | Ajusta uma tendência linear ou exponencial ao indicador de saúde e extrapola até o limiar de falha. |
| `estimate_rul_from_twin` | função | `(twin, failure_threshold, *, detector=None, sensor_id=None, field_name=None, n=None, mode…)` | Atalho que estima a RUL direto de um `DigitalTwin` vivo (escores de anomalia ou um campo observado). |

#### `pinneapple_systems.digital_twin.realtime_pipeline`

Pipeline de inferência em tempo real que impõe tempo de inferência ≤ período dos dados, com políticas `warn`, `skip`, `throttle` ou `raise` e estatísticas de latência. [sem teste]

<small>`pinneapple_systems/digital_twin/realtime_pipeline.py` · 470 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LatencyBudgetExceeded` | classe | `()` | Erro levantado quando a inferência passa do período dos dados e a política é `raise`. |
| `LatencyStats` | classe | `(n_frames, n_skipped, n_budget_violations, mean_inference_ms, max_inference_ms, budget_ms…)` | Estatísticas de latência: frames, descartes, violações, média e máximo de inferência, orçamento e utilização. |
| `RealtimePipelineConfig` | classe | `(data_period, latency_policy, buffer_maxlen, latency_alpha, throttle_min_points, throttle…)` | Configuração do pipeline: período, política de latência, buffer, suavização, throttling, dispositivo e AMP. |
| `RealtimePipeline` | classe | `(model, data_period, field_names, coord_names=None, *, config=None, on_state=None, on_lat…)` | Pipeline que envolve um modelo substituto, recebe leituras por `push` e mantém o último estado dentro do orçamento de latência. |

#### `pinneapple_systems.digital_twin.recession`

Gêmeo de rio por recessão de vazão de base: calibra Q0 e k no passado recente de um posto, prevê o presente e sinaliza um desvio sustentado (veredito `DIVERGENCE_DETECTED`, `MATCHES_FORECAST` ou `NO_CLEAN_RECESSION`).

<small>`pinneapple_systems/digital_twin/recession.py` · 195 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `recession` | função | `(q0, k_per_hour, hours) -> float \| np.ndarray` | Curva de recessão Q0·exp(-k·t), com t em horas. |
| `fit_recession` | função | `(hours, values) -> dict[str, float]` | Ajusta Q0 e k por mínimos quadrados em ln Q e devolve R² e média/desvio dos resíduos. |
| `forecast` | função | `(calibration, hours) -> np.ndarray` | Avalia a curva calibrada em novas horas. |
| `detect_divergence` | função | `(hours, actual, predicted, residual_mean, residual_std, n_sigma=N_SIGMA, sustained_minute…)` | Acha a primeira excursão sustentada do resíduo além de n desvios da calibração. |
| `run_twin` | função | `(timestamps, values, *, stage_timestamps=None, stage_values=None, calibration_hours=CALIB…)` | Pipeline completo numa série: separa calibração e comparação, calibra, prevê, compara e dá o veredito; levanta ValueError se a série for curta. |
| `synthetic_recession` | função | `(hours=96.0, step_minutes=15.0, q0=2000.0, k_per_hour=0.004, noise=0.005, bump_at_hour=No…)` | Registro sintético de posto (recessão exponencial com ruído e degrau opcional) para testes e demos offline. |

#### `pinneapple_systems.digital_twin.signal_reconstruction`

Reconstrução de sinais limpos a partir das observações do gêmeo: espectral (FFT), SSA, wavelets, interpolação e via modelo físico. [sem teste]

<small>`pinneapple_systems/digital_twin/signal_reconstruction.py` · 587 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ReconstructionResult` | classe | `(sensor_id, field_name, method, timestamps, signal, raw_timestamps, raw_values, quality, …)` | Saída de uma reconstrução: sensor, campo, método, tempos e sinal reconstruídos, dados brutos e métricas de qualidade. |
| `SignalReconstructor` | classe | `(twin_or_observations, *, default_method='ssa', default_n_out=512) -> None` | Atalho que reconstrói sinais a partir do histórico de observações de um `DigitalTwin` (ou de uma lista de `Observation`). |
| `reconstruct_spectral` | função | `(timestamps, values, *, n_harmonics=10, out_times=None, min_points=8) -> Tuple[np.ndarray…)` | Reconstrução por FFT: ajusta os harmônicos dominantes e avalia numa grade densa. |
| `reconstruct_ssa` | função | `(timestamps, values, *, window=0, n_components=5, out_times=None) -> Tuple[np.ndarray, np…)` | Reconstrução por SSA: decompõe e mantém os componentes dominantes (supressão de ruído). |
| `reconstruct_wavelet` | função | `(timestamps, values, *, wavelet='db4', level=4, threshold_mode='soft', threshold_sigma_mu…)` | Remoção de ruído por limiar suave de wavelets (requer PyWavelets). |
| `reconstruct_interpolate` | função | `(timestamps, values, *, kind='cubic', out_times=None, n_out=512) -> Tuple[np.ndarray, np.…)` | Preenchimento de lacunas e reamostragem por interpolação SciPy (linear, cúbica, PCHIP etc.). |
| `reconstruct_model_based` | função | `(timestamps, values, *, model, sensor_coord, coord_names, field_name, field_index=0, out_…)` | Usa um modelo físico treinado (PyTorch) para reconstruir a série num ponto do espaço consultando `model(x, y, t)`. |

#### `pinneapple_systems.digital_twin.standards`

Interoperabilidade: descreve o gêmeo uma vez e publica em AAS (BaSyx), DTDL v3, NGSI-LD, Eclipse Ditto e um servidor OPC UA (asyncua).

<small>`pinneapple_systems/digital_twin/standards.py` · 322 linhas · testes: 31</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TwinProperty` | classe | `(name, value, unit, kind, description, writable)` | Uma propriedade do gêmeo: nome, valor, unidade, tipo (parâmetro, telemetria, estado...), descrição e se é gravável. |
| `TwinModel` | classe | `(name, family, status, compatibility, violated, probable_cause)` | Um modelo por trás do gêmeo e sua validade física no ponto de operação atual (pode vir de uma auditoria). |
| `TwinDescription` | classe | `(id, name, asset_type, description, manufacturer, properties, models, relationships)` | Descrição neutra de um gêmeo (id, nome, tipo, propriedades, modelos, relações) usada por todos os exportadores. |
| `OpcUaTwinServer` | classe | `(desc, endpoint=…, namespace='urn:pinneapple:twins')` | Serve o gêmeo por OPC UA (asyncua) numa thread: um objeto por gêmeo, uma variável por propriedade; `update` grava valores. |
| `to_aas` | função | `(desc, as_json=True)` | Monta um Asset Administration Shell (metamodelo v3) com o SDK Eclipse BaSyx e devolve o JSON ou o objeto. |
| `to_dtdl` | função | `(desc, version=1) -> dict[str, Any]` | Gera uma Interface DTDL v3 com Properties, Telemetry e Relationships. |
| `to_ngsi_ld` | função | `(desc, context=…, observed_at=None) -> dict[str, Any]` | Gera uma entidade NGSI-LD com Property e Relationship. |
| `to_ditto_thing` | função | `(desc, namespace='org.pinneapple') -> dict[str, Any]` | Gera um Thing do Eclipse Ditto com atributos e features. |

#### `pinneapple_systems.digital_twin.state`

Estruturas de estado: `Observation` (leitura bruta de um sensor) e `SystemState` (estimativa fundida com covariância, resíduos e histórico). [sem teste direto]

<small>`pinneapple_systems/digital_twin/state.py` · 137 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Observation` | classe | `(timestamp, sensor_id, values, coords, metadata)` | Uma observação bruta de um sensor: tempo, id, valores, coordenadas e metadados. |
| `SystemState` | classe | `(timestamp, fields, coords, covariance, residuals, metadata, history)` | Estimativa corrente do sistema: campos, coordenadas, covariância, resíduos e histórico; converte de/para vetor plano. |

#### `pinneapple_systems.digital_twin.twin`

Classe central `DigitalTwin`: envolve um modelo treinado, consome fluxos de sensores, faz assimilação (nenhuma, EKF ou EnKF), detecta anomalias e mantém o estado vivo; PyTorch é opcional.

<small>`pinneapple_systems/digital_twin/twin.py` · 650 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DigitalTwinConfig` | classe | `(device, batch_size, use_amp, update_interval, obs_queue_maxsize, max_history, assimilati…)` | Configuração do `DigitalTwin`: dispositivo, lote, AMP, intervalo de atualização, fila, histórico, assimilação e parâmetros de anomalia. |
| `DigitalTwin` | classe | `(model, field_names, coord_names=None, *, config=None, assimilation_filter=None) -> None` | Gêmeo digital que funde um modelo substituto treinado com dados de sensores vivos; gerencia fluxos, laço de atualização, callbacks, predição e histórico. |
| `build_digital_twin` | função | `(model, field_names, *, coord_names=None, device='cpu', update_interval=0.5, anomaly_z_th…)` | Fábrica de conveniência para `DigitalTwin` com as opções mais comuns. |

#### `pinneapple_systems.excitation`

Sinais de excitação de atuador em NumPy (degrau, rampa, senoide, escada aleatória, PRBS, chirp, dente de serra, passeio aleatório suavizado, pulsos gaussianos, AM de dois tons) para identificação de sistemas e treino de surrogates; separa famílias de treino e famílias reservadas (`TRAIN_FAMILIES`, `HELDOUT_FAMILIES`).

<small>`pinneapple_systems/excitation.py` · 152 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `shape_step` | função | `(tau, t0=0.3, **_)` | Degrau unitário em `tau = t0`. |
| `shape_ramp` | função | `(tau, t1=0.8, **_)` | Rampa linear de 0 em `tau = 0` até 1 em `tau = t1`, depois constante. |
| `shape_sinusoid` | função | `(tau, freq=2.0, phase=0.0, **_)` | Senoide `0.5 + 0.5 sin(2 pi freq tau + phase)`. |
| `shape_piecewise_constant` | função | `(tau, rng=None, n_seg=5, **_)` | Escada aleatória com `n_seg` níveis uniformes entre bordas aleatórias ordenadas. |
| `shape_prbs` | função | `(tau, rng=None, dwell=0.08, **_)` | Sequência binária pseudoaleatória (PRBS) em {0, 1}, trocando de nível só nos instantes do relógio `dwell`. |
| `shape_chirp` | função | `(tau, f0=0.5, f1=6.0, **_)` | Chirp linear com frequência instantânea de `f0` a `f1` ao longo de `tau`. |
| `shape_sawtooth` | função | `(tau, freq=3.0, **_)` | Dente de serra crescente `(freq * tau) mod 1`. |
| `shape_smoothed_random_walk` | função | `(tau, rng=None, **_)` | Passeio aleatório gaussiano suavizado por média móvel e reescalado para [0, 1]. |
| `shape_gaussian_bumps` | função | `(tau, rng=None, n_bumps=4, width=0.05, **_)` | Soma de `n_bumps` gaussianas com centros e amplitudes aleatórios, normalizada pelo pico. |
| `shape_am_two_tone` | função | `(tau, f_carrier=5.0, f_mod=0.7, **_)` | Modulação em amplitude: produto de uma portadora e de uma moduladora senoidais. |
| `rate_limit` | função | `(series, tau_steps) -> np.ndarray` | Filtro passa-baixa de primeira ordem que limita a taxa de variação do sinal (constante de tempo em amostras). |
| `actuator_series` | função | `(shape, n_steps, *, lo=0.0, hi=1.0, seed=0, rate_limit_steps=0, **kwargs) -> np.ndarray` | Amostra uma forma (pelo nome em `SHAPES` ou callable) em `n_steps` pontos, escala para `[lo, hi]` e opcionalmente aplica `rate_limit`. |

#### `pinneapple_systems.operations`

Operações: rodar modelos físicos validados e substitutos ao lado de dados de planta. Uma planta comum (tanque aquecido) e três ferramentas sobre ela: runtime de gêmeo, detecção física de anomalias e controle preditivo, todas falando `pinneapple_data.contracts`.

<small>`pinneapple_systems/operations/__init__.py` · 94 linhas · testes: 24</small>

#### `pinneapple_systems.operations.anomaly`

Detecção física de anomalias em séries de sensores: quatro famílias (estatística robusta, AR + CUSUM, balanços de conservação, resíduos de modelo físico) calibradas numa janela declarada normal, fundidas em eventos com severidade e explicações, mais um benchmark rotulado de falhas.

<small>`pinneapple_systems/operations/anomaly.py` · 573 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BalanceRule` | classe | `(name, accumulation, inflow, outflow, units, description)` | Lei de conservação sobre colunas: d(acumulação)/dt = entrada - saída, com unidades e descrição. |
| `DetectorConfig` | classe | `(normal_until, sensors, methods, balances, constants, window, threshold, ar_order, cusum_…)` | O que rodar e quão estrito ser: fim da janela normal, sensores, métodos, balanços, constantes, janela, limiar, ordem AR, parâmetros CUSUM. |
| `Channel` | classe | `(method, name, variables, residual, centre, spread, units, description, score, threshold)` | Uma saída de detector: série de resíduos, centro e dispersão da janela normal e escore normalizado. |
| `Event` | classe | `(start, end, i0, i1, severity, peak_ratio, methods, channels, variables, explanations, ev…)` | Evento de anomalia fundido: intervalo, severidade, razão de pico, métodos, canais, variáveis, explicações e evidência. |
| `DetectionResult` | classe | `(t, channels, flags, events, config, normal_samples)` | Resultado de `detect`: tempos, canais, flags, eventos, configuração e amostras normais. |
| `LabelledInterval` | classe | `(kind, start, end, target)` | Intervalo anormal rotulado `[start, end)` lido dos dados, com a mesma interface de um `Fault` da planta. |
| `tank_balance_rules` | função | `() -> list[BalanceRule]` | Balanços de massa e energia do tanque aquecido como `BalanceRule`. |
| `tank_constants` | função | `(p=None) -> dict[str, float]` | Constantes do tanque (área, densidade, cp, UA) usadas nas fórmulas dos balanços. |
| `windowed_balance` | função | `(t, acc, net_in, w) -> np.ndarray` | Resíduo de balanço numa janela: variação da acumulação dividida pelo tempo menos a média trapezoidal do líquido de entrada. |
| `severity_of` | função | `(ratio) -> str` | Severidade pela razão pico/limiar: baixa, média, alta ou crítica. |
| `detect` | função | `(data, config, model=None, model_states=(), model_inputs=()) -> DetectionResult` | Roda os detectores configurados sobre uma tabela (DataFrame ou dict com `t`) e devolve canais, flags e eventos. |
| `benchmark_faults` | função | `(t0=7200.0) -> list[Fault]` | As falhas do benchmark depois de um período inicial sem falha, com magnitudes pequenas de propósito. |
| `fault_benchmark` | função | `(seed=0, t0=7200.0, dt=10.0) -> tuple[dict[str, np.ndarray], list[Fault]]` | Uma corrida rotulada: operação normal inicial e depois as falhas do benchmark separadas por períodos normais. |
| `intervals_from_labels` | função | `(t, labels) -> list[LabelledInterval]` | Converte uma coluna de rótulos em intervalos contíguos de anomalia. |
| `score_events` | função | `(result, faults, grace=600.0, evaluate_from=None) -> dict` | Escores ponto a ponto e por evento contra os intervalos verdadeiros de falha. |
| `run_benchmark` | função | `(seeds=(101, 102, 103), methods_sets=None, window=12, threshold=5.0) -> dict` | Pontua cada família de detectores sozinha e todas juntas em corridas rotuladas separadas, sem ajustar limiares na parte com falha. |

#### `pinneapple_systems.operations.contracts`

Converte modelos e estudos de operações para `pinneapple_data.contracts`: modelos viram `ModelRef` (com métricas de hold-out e SHA-256) e estudos de gêmeo, anomalia e controle viram `StudyResult` com checagens do que rodou.

<small>`pinneapple_systems/operations/contracts.py` · 192 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `save_model` | função | `(model, path, name, *, app='', dataset=None, trust_decision='UNCHECKED') -> ModelRef` | Grava um modelo de gêmeo ou `ControlledOpInf` em JSON e devolve seu `ModelRef` (também gravado ao lado). |
| `load_model` | função | `(ref, root='.')` | Abre o modelo apontado por um `ModelRef`, conferindo o SHA-256. |
| `check` | função | `(name, passed, value=None, threshold=None, evidence='', detail='', warn=False) -> CheckRe…)` | Cria um `CheckRecord`; `passed=None` significa que a checagem não rodou (NOT_RUN). |
| `new_id` | função | `(prefix) -> str` | Gera um id curto com prefixo e parte aleatória (uuid). |
| `twin_study` | função | `(runtime, *, app='twin', model_ref=None, title='') -> StudyResult` | Converte a execução de um gêmeo num `StudyResult` (estimativas, alertas, checagens de validade, revalidações). |
| `anomaly_study` | função | `(result, *, app='anomaly', benchmark=None, title='') -> StudyResult` | Converte uma detecção (e opcionalmente um benchmark) num `StudyResult`. |
| `control_study` | função | `(comparison, *, app='control', title='') -> StudyResult` | Converte uma comparação de controladores num `StudyResult`, uma linha por controlador. |

#### `pinneapple_systems.operations.control`

Controle do tanque aquecido: MPC linear e não linear contra PID e regras com histerese, em malha fechada na mesma planta simulada, só em modo simulação/recomendação (nada é enviado à planta); CasADi é opcional.

<small>`pinneapple_systems/operations/control.py` · 658 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ControlSpec` | classe | `(dt, horizon, setpoints, h_bounds, T_max, q_in_bounds, Q_bounds, dq_max, dQ_max, h_scale,…)` | Objetivos, restrições e limites dos atuadores do controle (passo, horizonte, setpoints, limites, pesos). |
| `ControlScenario` | classe | `(duration, x0, T_in_steps, T_amb, faults, noise, seed)` | Lado da planta num estudo em malha fechada: duração, estado inicial, perturbações, falhas e ruído, comum a todos os controladores. |
| `PIDBaseline` | classe | `(spec, params=None, kp_h=0.02, ki_h=0.0001, kp_T=60000.0, ki_T=400.0)` | Baseline PID: malha de nível na bomba e de temperatura no aquecedor, com feed-forward e anti-windup. |
| `RuleBasedBaseline` | classe | `(spec, band_h=0.05, band_T=0.5, q_low=0.004, q_high=0.01, Q_on=600000.0)` | Baseline por regras com histerese: bomba em duas vazões e aquecedor liga/desliga. |
| `LinearMPC` | classe | `()` | MPC com linearização sucessiva: um QP no horizonte resolvido com SLSQP. |
| `NonlinearMPC` | classe | `(spec, params=None, model=None, backend='auto')` | MPC não linear nos balanços do tanque (CasADi/IPOPT se disponível, senão SciPy) ou sobre qualquer modelo com `step`. |
| `make_estimator` | função | `(spec, params=None, seed=0) -> TwinRuntime` | Estimador de estado lido pelos MPCs: gêmeo EnKF do tanque com vazamento não medido e eficiência do aquecedor. |
| `run_closed_loop` | função | `(controller, spec, scenario, params=None) -> dict` | Simula um controlador num cenário e devolve trajetórias, planos (MPC) e métricas de desempenho. |
| `performance` | função | `(tr, spec) -> dict` | Métricas de uma trajetória: erros integrados, violações de restrição, energia, curso dos atuadores e tempo de solução. |
| `compare_controllers` | função | `(spec=None, scenario=None, params=None, controllers=…, surrogate=None) -> dict` | Roda os controladores nomeados (`pid`, `rules`, `lmpc`, `nmpc`, `smpc`) no mesmo cenário. |
| `recommend` | função | `(estimate, reading, spec=None, params=None, t=0.0, method='nmpc') -> dict` | Propõe as próximas ações para uma pessoa revisar, a partir da estimativa e da leitura atual; nada é enviado. |

#### `pinneapple_systems.operations.exchanger`

Segunda planta de referência: trocador água/água em contracorrente com incrustação, usado pela camada de comissionamento de `digital_twin.product`; gera um histórico claramente sintético. [sem teste direto] — usado em `tests/test_digital_twin_product.py` via pacote.

<small>`pinneapple_systems/operations/exchanger.py` · 158 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OperatingPoint` | classe | `(t_hi, t_ci, m_h, m_c)` | Ponto de operação do trocador: temperaturas de entrada e vazões quente e fria. |
| `PlantConfig` | classe | `(ua_clean, flow_sensitivity, tau_fouling_s, rf_final)` | O trocador como construído, incluindo o erro de modelo com que o gêmeo tem de conviver. |
| `exchanger_model` | função | `(X, theta, *, ua_clean) -> dict[str, np.ndarray]` | `model(X, theta)` dado ao gêmeo: temperaturas de saída a partir do ponto de operação, com a resistência de incrustação em `theta`. |
| `plant_outlets` | função | `(op, rf, cfg) -> tuple[float, float]` | A planta real: o modelo mais um UA dependente da vazão que o modelo não conhece. |
| `fouling_at` | função | `(t_s, cfg) -> float` | Incrustação assintótica no tempo: rápida no início, depois patamar. |
| `generate_history` | função | `(hours=72.0, sampling_s=300.0, *, seed=7, cfg=None, op=None, t_start=1767225600.0) -> lis…)` | Histórico sintético e determinístico da planta com incrustação, oscilação de vazão e ruído de instrumento. |

#### `pinneapple_systems.operations.live`

[experimental no OPC UA] Roda um `TwinRuntime` como serviço: fonte simulada via MQTT (broker embutido ou externo), MQTT externo, pushes REST ou um poller OPC UA experimental; também faz replay sobre uma tabela gravada.

<small>`pinneapple_systems/operations/live.py` · 251 linhas · testes: 11 · palavras-chave no código (heurística; o texto acima diz o status real): experimental</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LiveTwin` | classe | `(twin, *, source='simulated', topic=None, broker_host='127.0.0.1', broker_port=0, embedde…)` | Gêmeo vivo de um ativo: liga fonte (simulada, MQTT, REST ou OPC UA experimental), consumidor e armazenamento; permite injetar falhas e tirar snapshot. |
| `make_twin` | função | `(model, *, asset='tank-1', store=None, rules=None, sigma_obs=None, sigma_process=None, se…)` | Cria um `TwinRuntime` do tanque aquecido com os níveis de ruído dos seus sensores. |
| `replay` | função | `(table, model, *, asset='tank-1', store=None, rules=None, forecast_steps=60, revalidate=T…)` | Roda um gêmeo sobre uma tabela gravada e devolve histórico, alertas, previsão final e, se pedido, re-pontuação do modelo. |
| `demo_record` | função | `(duration=7200.0, dt=10.0, seed=4, faults=None) -> dict` | Corrida gravada do tanque simulado com um vazamento na segunda metade (exemplo offline do app de gêmeo). |
| `default_model` | função | `(kind='lumped', dt=10.0)` | Modelo padrão do gêmeo: só o `LumpedTankModel`; outro tipo levanta ValueError. |

#### `pinneapple_systems.operations.plant`

Tanque aquecido de buffer: planta concentrada (0-D) com balanços de massa e energia, sensores com ruído, falhas e resíduos de balanço calculados de sinais medidos.

<small>`pinneapple_systems/operations/plant.py` · 290 linhas · testes: 25</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TankParams` | classe | `(area, k_out, ua, rho, cp, h_min, h_max, q_in_max, q_max_heater)` | Parâmetros físicos do tanque aquecido (SI, temperaturas em °C). |
| `Fault` | classe | `(kind, start, end, magnitude, target)` | Falha ativa em `[start, end)` segundos; `target` indica o sensor para falhas de sensor. |
| `Scenario` | classe | `(duration, dt, x0, T_in, T_amb, q_in, Q)` | Cenário de operação: como entradas e perturbações evoluem no tempo. |
| `SensorNoise` | classe | `(sigma)` | Desvio-padrão do ruído branco de cada sensor. |
| `TankSimulator` | classe | `(params=None, x0=(1.2, 25.0), faults=(), noise=None, seed=0)` | Simulador passo a passo do tanque com sensores e falhas, para malha aberta, malha fechada ou fonte de demo. |
| `tank_rhs` | função | `(x, u, d, p, eta=1.0, valve=1.0, q_leak=0.0) -> np.ndarray` | Derivadas [dh/dt, dT/dt] do tanque para estado, entradas e perturbações (vetorizado). |
| `tank_step` | função | `(x, u, d, p, dt, substeps=4, **fault_kw) -> np.ndarray` | Avança o tanque `dt` segundos com RK4 clássico. |
| `steady_state` | função | `(p, q_in, Q, T_in, T_amb) -> np.ndarray` | Equilíbrio [h, T] para entradas constantes (forma fechada). |
| `inputs_for_steady_state` | função | `(p, h, T, T_in, T_amb) -> np.ndarray` | Entradas [q_in, Q] que mantêm o tanque num [h, T] dado (forma fechada). |
| `measure` | função | `(truth, t, faults, noise, rng, frozen) -> dict[str, float]` | Leituras de um passo: valor verdadeiro + ruído, depois falhas de sensor. |
| `simulate` | função | `(scenario, params=None, faults=(), noise=None, seed=0) -> dict[str, np.ndarray]` | Corrida em malha aberta de um cenário; devolve sensores medidos, estados verdadeiros e a falha ativa. |
| `demo_scenario` | função | `(duration=3600.0, dt=10.0, seed=0) -> Scenario` | Perfil realista de turno: degraus de vazão, aquecedor por agenda e temperatura de entrada variando devagar; determinístico por semente. |
| `mass_balance_residual` | função | `(t, h, q_in, q_out, area) -> np.ndarray` | Resíduo do balanço de massa a partir de sinais medidos; um vazamento o torna igual a menos a vazão vazada. |
| `energy_balance_residual` | função | `(t, h, T, q_in, q_out, Q, T_in, T_amb, p) -> np.ndarray` | Resíduo do balanço de energia em forma conservativa, em W. |
| `with_params` | função | `(p, **changes) -> TankParams` | Cópia de `TankParams` com campos trocados. |

#### `pinneapple_systems.operations.rom`

Operator inference com entradas: modelo reduzido polinomial em tempo discreto ajustado por regressão ridge, validado em trajetórias inteiras separadas; serializa em JSON e tem passo CasADi opcional.

<small>`pinneapple_systems/operations/rom.py` · 178 linhas · testes: 11</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ControlledOpInf` | classe | `(state_names, input_names, dt, W, x_mean, x_scale, v_mean, v_scale, quadratic, bilinear, …)` | Modelo reduzido ajustado (matrizes, escalas, caixa de treino, validação); `step`, `rollout`, `casadi_step` e JSON. |
| `fit_controlled_opinf` | função | `(trajectories, state_names, input_names, dt, *, holdout=1, ridge=1e-06, quadratic=True, b…)` | Ajusta o modelo em todas menos as últimas trajetórias e valida nelas (erros de um passo e de rollout). |
| `residual_sigma` | função | `(model, trajectories) -> dict[str, float]` | Desvio dos resíduos de um passo por estado: o ruído de processo que um filtro com este modelo deve assumir. |

#### `pinneapple_systems.operations.series`

Cadeia ponta a ponta num ativo (Campaign Planner → corridas → Surrogate Studio → incerteza → otimizador → gêmeo), cada passagem um arquivo de contrato; diz para cada passo se o app existe em `apps/` ou rodou como substituto de biblioteca.

<small>`pinneapple_systems/operations/series.py` · 339 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `operating_space` | função | `() -> ParameterSpace` | `ParameterSpace` das entradas de operação do tanque excitadas pela campanha. |
| `app_available` | função | `(pattern, apps_dir) -> str \| None` | Nome do diretório `apps/` de um app da série, se existir neste checkout. |
| `run_series_flow` | função | `(workdir, *, seed=0, n_runs=12, steps_per_run=12, hold=300.0, dt=20.0, T_min=28.0, q_min=…)` | Roda a cadeia inteira e grava cada contrato em `workdir`; devolve um resumo em JSON. |

#### `pinneapple_systems.operations.twin`

Runtime de gêmeo digital: EnKF com parâmetros no estado, previsões com bandas 5-95 %, alertas por regras e teste de consistência (NIS), monitoramento de validade e revalidação, histórico em SQLite ou Postgres (psycopg opcional).

<small>`pinneapple_systems/operations/twin.py` · 602 linhas · testes: 25</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LumpedTankModel` | classe | `(params=None, dt=10.0, estimate=('q_leak', 'eta'))` | Balanços de massa e energia do tanque como modelo de gêmeo, opcionalmente com parâmetros (vazamento, eficiência) no estado. |
| `SurrogateModel` | classe | `(rom)` | Um `ControlledOpInf` usado como modelo de gêmeo (sem parâmetros no estado). |
| `AlertRules` | classe | `(limits, param_limits, param_sigmas, nis_probability, nis_consecutive, forecast_steps, dr…)` | Limiares de alerta: limites de estado e de parâmetro, teste NIS, horizonte de previsão, deriva e revisão. |
| `Alert` | classe | `(t, kind, severity, message, data, cleared)` | Um alerta: tempo, tipo, severidade, mensagem, dados e se foi limpo. |
| `TwinStore` | classe | `(url=':memory:')` | Histórico de leituras, estimativas, alertas e revalidações (SQLite por padrão, Postgres opcional). |
| `TwinRuntime` | classe | `(model, obs_names, sigma_obs, sigma_process, *, x0=None, sigma0=None, n_ens=48, inflation…)` | Assimila leituras uma a uma no modelo e mantém estado, previsões, alertas e validade do gêmeo. |
| `model_from_dict` | função | `(d)` | Reconstrói um modelo de gêmeo do JSON (`lumped_tank` ou `controlled_opinf`). |

#### `pinneapple_systems.process_components`

Modelos físicos de operações unitárias para gêmeos de processo: gás real, turbomáquinas, válvulas, trocadores, tubulações, vigas, cinética de reação, fluidos não newtonianos, tensões, stick-slip, fadiga e um motor genérico de equações; módulos independentes entre si.

<small>`pinneapple_systems/process_components/__init__.py` · 272 linhas · testes: 58</small>

#### `pinneapple_systems.process_components.beam_nonlinear_fem`

Viga Euler-Bernoulli geometricamente não linear (Von Karman), estática, por Newton-Raphson em passos de carga, reutilizando o motor transiente de `pinneapple_simulation` no limite estático (massa e amortecimento zero).

<small>`pinneapple_systems/process_components/beam_nonlinear_fem.py` · 172 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `StaticNonlinearBeamResult` | classe | `(z, u_m, w_m, theta_rad, bending_moment, bending_stress, load_step_deflections, iteration…)` | Resultado da viga não linear: deslocamentos, rotação, momento, tensão, deflexão por passo de carga, iterações e passos não convergidos. |
| `solve_nonlinear_beam_static` | função | `(boundary, L_m, A_m2, I_m4, E_Pa, fiber_distance_m, axial_dist_load_N_per_m=0.0, transver…)` | Resolve a viga Von Karman estática por Newton-Raphson em passos de carga e recupera a tensão de flexão na fibra externa. |

#### `pinneapple_systems.process_components.beam_statics`

Soluções fechadas de viga Euler-Bernoulli linear estática para casos padrão de balanço e biapoiada (cargas uniforme, parcial e triangular), mais propriedades de seção retangular e Von Mises.

<small>`pinneapple_systems/process_components/beam_statics.py` · 196 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BeamResult` | classe | `(x, deflection, slope, bending_moment, shear_force)` | Resultado da viga linear: posições, deflexão, inclinação, momento fletor e cortante. |
| `rectangular_section_properties` | função | `(width_m, height_m) -> tuple[float, float]` | Momento de inércia I e primeiro momento Q de uma seção retangular. |
| `von_mises_stress_rectangular_section` | função | `(bending_moment, shear_force, I, Q, width_m, height_m) -> np.ndarray` | Tensão de Von Mises numa seção retangular a partir de momento e cortante. |
| `solve_beam` | função | `(case, x, L, E, I, q0, load_extent=None) -> BeamResult` | Despacha para a solução fechada do caso de viga pedido (com `load_extent` nos casos de carga parcial). |

#### `pinneapple_systems.process_components.control_valve`

Dimensionamento de válvula de controle pela IEC 60534-2-1 (compressível com fator de expansão e bloqueio, e incompressível simples), característica inerente, vazamento de sede e atraso do atuador. [sem teste direto] — testado via `tests/test_process_components.py` pelo pacote.

<small>`pinneapple_systems/process_components/control_valve.py` · 115 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ValveSpec` | classe | `(name, Cv_max_us, xT, characteristic, equal_percentage_R, actuator_time_constant_s, leaka…)` | Especificação da válvula: Cv máximo, xT, característica, atuador e vazamento de sede. |
| `ValveFlowResult` | classe | `(mass_flow_kg_s, choked, x, Y, Cv_effective_us)` | Resultado de vazão compressível: vazão mássica, se bloqueou, x, Y e Cv efetivo. |
| `installed_cv` | função | `(travel_fraction, spec) -> float` | Cv numa fração de curso pela característica inerente (sem vazamento). |
| `effective_cv` | função | `(travel_fraction, spec, leakage_cv_us_override=None) -> float` | Cv que de fato passa vazão, incluindo o piso de vazamento de sede (pode sobrescrever o vazamento). |
| `compressible_mass_flow` | função | `(Cv_us, P1_Pa, P2_Pa, rho1_kg_m3, k_isentropic, xT) -> ValveFlowResult` | Vazão mássica compressível pela IEC 60534-2-1 com fator de expansão e bloqueio, em SI. |
| `incompressible_mass_flow` | função | `(Cv_us, P1_Pa, P2_Pa, rho_kg_m3) -> float` | Vazão de líquido simples w = Cv_SI·√(ΔP·ρ), sem modelo de flashing/cavitação. |
| `actuator_response_rhs` | função | `(travel_actual, travel_command, spec) -> float` | Derivada do curso real para o atraso de primeira ordem do atuador. |

#### `pinneapple_systems.process_components.curved_path_geometry`

Gerador de trajetória curva "build-and-hold" (vertical, ganho de inclinação com curvatura constante, trecho mantido) com profundidade vertical e afastamento horizontal; só em um plano (2D). A marca de placeholder do inventário vem da frase "não implementado aqui" sobre o caso 3D.

<small>`pinneapple_systems/process_components/curved_path_geometry.py` · 113 linhas · testes: 7 · palavras-chave no código (heurística; o texto acima diz o status real): placeholder/stub citado no docstring</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CurvedPathProfile` | classe | `(s_m, inclination_deg, tvd_m, hd_m, n_bins, vertical_length_m, total_length_m, bin_width_…)` | Perfil discretizado da trajetória: comprimento medido, inclinação, profundidade vertical, afastamento e larguras. |
| `inclination_at_depth` | função | `(s_m, vertical_length_m, build_length_m, final_inclination_deg) -> np.ndarray` | Inclinação em cada profundidade medida: vertical, linear no ganho e mantida depois. |
| `build_and_hold_profile` | função | `(total_length_m, build_length_m, final_inclination_deg, hold_length_m, bin_width_m) -> Cu…)` | Discretiza a trajetória build-and-hold e integra profundidade vertical e afastamento horizontal. |
| `circular_arc_tvd_hd` | função | `(s_within_build_m, build_length_m, final_inclination_deg) -> tuple` | Profundidade e afastamento em forma fechada no trecho de ganho (arco circular), para conferir a integração. |

#### `pinneapple_systems.process_components.explicit_equation_system`

Motor genérico e isolado de "modelo como equações nomeadas": avalia expressões com AST seguro, ordena por dependência (detecta ciclos) e calibra parâmetros limitados por mínimos quadrados.

<small>`pinneapple_systems/process_components/explicit_equation_system.py` · 393 linhas · testes: 171</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ExplicitEquationError` | classe | `()` | Erro do motor de equações explícitas. |
| `Definition` | classe | `(name, lhs, rhs, rhs_symbols)` | Uma equação já analisada: nome, símbolo do lado esquerdo, expressão e símbolos usados. |
| `AnalysisResult` | classe | `(valid, errors, evaluation_order)` | Resultado da validação: se é válido, erros e ordem de avaliação. |
| `ParameterSpec` | classe | `(symbol, value, fit, min, max)` | Um parâmetro: símbolo, valor, se deve ser ajustado e limites. |
| `CalibrationResult` | classe | `(fitted_parameters, start, bounds, n_evaluations, final_scaled_residual_ssr, termination_…)` | Resultado da calibração: parâmetros ajustados, ponto inicial, limites, avaliações, resíduo final e mensagem. |
| `safe_eval` | função | `(expr, env) -> Any` | Avalia uma expressão por AST permitindo só operações e funções de uma lista branca. |
| `build_definitions` | função | `(inputs, parameters, intermediates, outputs, equations) -> tuple[list[Definition], list[s…)` | Valida e monta as definições a partir de dicts `{name, equation}`, acumulando todos os erros. |
| `evaluation_order` | função | `(definitions) -> list[Definition]` | Ordenação topológica das definições com detecção de ciclos. |
| `evaluate` | função | `(input_values, parameter_values, order) -> dict[str, Any]` | Avalia todas as definições na ordem dada e devolve o ambiente completo de símbolos. |
| `analyze` | função | `(inputs, parameters, intermediates, outputs, equations) -> AnalysisResult` | Validação a seco: monta e ordena o sistema e reporta erros, sem calcular números. |
| `calibrate` | função | `(input_columns, parameters, order, measured_rows, target_symbols) -> CalibrationResult` | Calibração não linear limitada (Trust Region Reflective) dos parâmetros com `fit=True` contra dados medidos, reavaliando o sistema completo. |

#### `pinneapple_systems.process_components.fatigue_analysis`

Vida em fadiga por curva S-N (Wöhler), correção de tensão média de Goodman e dano acumulado pela regra de Miner.

<small>`pinneapple_systems/process_components/fatigue_analysis.py` · 108 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MinerDamageResult` | classe | `(N_f, damage_this_period, cumulative_damage, remaining_life_fraction)` | Resultado de Miner: vida N_f, dano do período, dano acumulado e fração de vida restante. |
| `sn_curve_cycles_to_failure` | função | `(sigma_a, A, m) -> ArrayLike` | Ciclos até a falha pela curva S-N para uma amplitude totalmente reversa. |
| `goodman_equivalent_amplitude` | função | `(sigma_alt, sigma_mean, S_ut) -> ArrayLike` | Amplitude equivalente totalmente reversa pela linha de Goodman modificada. |
| `goodman_safety_ratio` | função | `(sigma_alt, sigma_mean, S_e, S_ut) -> ArrayLike` | Razão de segurança de Goodman (< 1 dentro da região segura). |
| `miners_rule_damage` | função | `(sigma_alt, sigma_mean, n_cycles, A_sn, m_sn, S_ut, prior_damage_fraction=0.0) -> MinerDa…)` | Vida S-N corrigida por Goodman, dano do período e dano acumulado de Miner (limitado a [0, 1]). |

#### `pinneapple_systems.process_components.heat_exchanger`

Trocador de calor por efetividade-NTU (contracorrente e paralelo, formas fechadas) e resposta transiente de capacitância concentrada. [sem teste direto] — testado via pacote em `tests/test_process_components.py`.

<small>`pinneapple_systems/process_components/heat_exchanger.py` · 113 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HeatExchangerSpec` | classe | `(name, UA_design_W_K, design_dP_Pa, design_m_dot_hot_kg_s, design_rho_hot_kg_m3, thermal_…)` | Especificação do trocador: UA de projeto, queda de pressão e vazão de projeto, massa térmica, capacidade do lado frio e arranjo. |
| `HeatExchangerResult` | classe | `(Q_W, T_hot_out_K, T_cold_out_K, effectiveness, NTU, dP_Pa, UA_effective_W_K)` | Resultado: calor trocado, temperaturas de saída, efetividade, NTU, queda de pressão e UA efetivo. |
| `steady_state` | função | `(spec, m_dot_hot_kg_s, T_hot_in_K, cp_hot_J_kgK, T_cold_in_K, rho_hot_kg_m3, UA_multiplie…)` | Solução estacionária efetividade-NTU, com multiplicador de UA (incrustação). |
| `transient_rhs` | função | `(T_hot_out_current_K, spec, m_dot_hot_kg_s, T_hot_in_K, cp_hot_J_kgK, T_cold_in_K, rho_ho…)` | Derivada da temperatura de saída quente, relaxando para o valor estacionário com a constante de tempo da massa térmica. |

#### `pinneapple_systems.process_components.non_newtonian_pipe_flow`

Reologia Herschel-Bulkley, Reynolds generalizado e fator de atrito de Metzner-Reed, e integração 1D da queda de pressão num conduto inclinado. A marca de placeholder do inventário vem da frase "não implementado aqui" sobre a correlação de Dodge-Metzner.

<small>`pinneapple_systems/process_components/non_newtonian_pipe_flow.py` · 155 linhas · testes: 10 · palavras-chave no código (heurística; o texto acima diz o status real): placeholder/stub citado no docstring</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PressureProfile` | classe | `(s_m, tvd_m, P_Pa, rho_eq_kg_m3, rho_kg_m3, friction_factor, reynolds_number)` | Perfil de pressão ao longo do conduto: posição, profundidade, pressão, densidade equivalente, atrito e Reynolds. |
| `herschel_bulkley_stress` | função | `(tau_y, K, n, gamma_dot) -> ArrayLike` | Tensão de Herschel-Bulkley τ = τ_y + K·γ̇^n. |
| `effective_viscosity` | função | `(tau_y, K, n, gamma_dot) -> ArrayLike` | Viscosidade efetiva τ/γ̇. |
| `generalized_reynolds_number` | função | `(rho, v, D, K, n) -> ArrayLike` | Reynolds generalizado de Metzner-Reed. |
| `metzner_reed_friction_factor` | função | `(Re_gen) -> ArrayLike` | Fator de atrito laminar (16/Re) ou turbulento (Blasius/Metzner-Reed). |
| `pressure_gradient` | função | `(rho, v, D_h, f, inclination_rad=np.pi / 2, g=9.81) -> ArrayLike` | Gradiente de pressão: termo hidrostático mais atrito. |
| `integrate_pressure_profile` | função | `(length_m, v_m_s, D_h_m, K_Pa_sn, n_flow_index, P_inlet_Pa, rho_profile=1000.0, inclinati…)` | Integra a pressão ao longo do conduto por Euler explícito, com densidade e inclinação constantes ou funções da posição. |

#### `pinneapple_systems.process_components.pipe_network_1d`

Escoamento 1D de gás real num único segmento de tubo: modo estacionário rápido (marcha célula a célula) e modo transiente de volumes finitos (momento quase estacionário, continuidade transiente), atrito de Colebrook-White.

<small>`pinneapple_systems/process_components/pipe_network_1d.py` · 290 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PipeSpec` | classe | `(name, length_m, diameter_m, roughness_m, elevation_change_m, ambient_temperature_K, wall…)` | Especificação do tubo: comprimento, diâmetro, rugosidade, desnível, troca de calor com o ambiente, células e multiplicador de atrito. |
| `SteadyProfilePoint` | classe | `(x_m, P_Pa, T_K, rho_kg_m3, G_kg_m2s, mach)` | Um ponto do perfil estacionário: posição, pressão, temperatura, densidade, fluxo mássico e Mach. |
| `PipeState` | classe | `(P_Pa, T_K)` | Estado transiente do tubo: pressões e temperaturas por célula. |
| `TransientPipe` | classe | `(spec, gas)` | Solver transiente de volumes finitos de um tubo de gás real, inicializável pelo perfil estacionário. |
| `colebrook_white_f` | função | `(Re, relative_roughness) -> float` | Fator de atrito de Darcy: laminar abaixo de Re 2300, senão Colebrook-White por Newton. |
| `face_reynolds_number` | função | `(G_kg_m2s, D_m, viscosity_Pa_s) -> float` | Número de Reynolds numa face a partir do fluxo mássico. |
| `rapid_steady_state_profile` | função | `(spec, gas, m_dot_kg_s, P_in_Pa, T_in_K) -> list[SteadyProfilePoint]` | Perfil estacionário marchando célula a célula a partir da entrada, sem integração no tempo. |

#### `pinneapple_systems.process_components.pipe_stress_mechanics`

Tensões combinadas num tubo esbelto (Lamé, torção, flexão por curvatura, Von Mises triaxial), flambagem de Euler, flambagem de haste confinada (Paslay-Dawson), amplificação P-Delta e ciclo de flexão rotativa.

<small>`pinneapple_systems/process_components/pipe_stress_mechanics.py` · 209 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ConstrainedRodBucklingResult` | classe | `(F_critical_sinusoidal, F_critical_helical, mode)` | Cargas críticas senoidal e helicoidal de Paslay-Dawson e o modo. |
| `RotatingBendingCycleResult` | classe | `(theta_rad, sigma_von_mises, sigma_alternating, sigma_mean)` | Envelope de Von Mises numa rotação e as tensões alternada e média para fadiga. |
| `lame_hoop_stress_outer` | função | `(p_internal, r_outer, r_inner) -> ArrayLike` | Tensão circunferencial de Lamé na parede externa. |
| `lame_hoop_stress_inner` | função | `(p_internal, r_outer, r_inner) -> ArrayLike` | Tensão circunferencial de Lamé na parede interna (máxima). |
| `torsional_shear_stress` | função | `(torque, r_outer, J) -> ArrayLike` | Cisalhamento de torção máximo T·c/J na fibra externa. |
| `bending_stress_from_curvature` | função | `(curvature, E, r_outer) -> ArrayLike` | Tensão de flexão na fibra externa a partir da curvatura. |
| `von_mises_triaxial` | função | `(sigma_axial, sigma_hoop, sigma_radial=0.0, tau=0.0) -> ArrayLike` | Von Mises triaxial com cisalhamento no plano axial-circunferencial. |
| `euler_critical_buckling_load` | função | `(E, I, length, end_condition_factor=1.0) -> float` | Carga crítica de Euler com fator de condição de extremidade. |
| `constrained_rod_buckling_load` | função | `(E, I, lateral_load_per_length, radial_clearance) -> ConstrainedRodBucklingResult` | Cargas críticas de Paslay-Dawson para haste confinada numa folga cilíndrica. |
| `classify_buckling_mode` | função | `(compressive_force, result) -> ArrayLike` | Classifica o modo: 0 reto, 1 senoidal, 2 helicoidal. |
| `beam_column_moment_amplification_factor` | função | `(compressive_force, E, I, length, end_condition_factor=1.0, max_ratio=0.98) -> ArrayLike` | Fator de amplificação P-Delta 1/(1 - P/P_cr), com P/P_cr limitado. |
| `rotating_bending_stress_cycle` | função | `(sigma_axial, sigma_bending_amplitude, sigma_hoop=0.0, tau=0.0, n_angles=36) -> RotatingB…)` | Envelope de Von Mises numa rotação completa sob flexão rotativa e tensões alternada e média. |

#### `pinneapple_systems.process_components.polytropic_path`

Integração do caminho politrópico de gás real (compressão ou expansão) consistente com ASME PTC 10, por integração direta passo a passo. [sem teste direto] — testado via pacote em `tests/test_process_components.py`.

<small>`pinneapple_systems/process_components/polytropic_path.py` · 196 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PolytropicPathResult` | classe | `(inlet, outlet, polytropic_work_J_kg, polytropic_efficiency, isentropic_work_J_kg, isentr…)` | Resultado do caminho politrópico: estados de entrada e saída, trabalhos, eficiências, razão de pressão e expoente. |
| `solve_path_from_pressure_ratio` | função | `(inlet, gas, P2_Pa, polytropic_efficiency) -> PolytropicPathResult` | Integra o caminho politrópico até uma pressão de saída conhecida (compressão ou expansão). |
| `solve_path_from_work` | função | `(inlet, gas, polytropic_work_J_kg, polytropic_efficiency, *, max_pressure_ratio=6.0) -> P…)` | Integra até atingir um trabalho politrópico alvo e resolve a pressão de saída. |

#### `pinneapple_systems.process_components.reaction_kinetics`

Motor genérico de redes de reação por ação de massas (constantes Arrhenius, polinomiais ou catalíticas, especiação por pH), integração rígida e transporte 1D advecção-dispersão-reação em grade periódica.

<small>`pinneapple_systems/process_components/reaction_kinetics.py` · 257 linhas · testes: 164</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Reaction` | classe | `(name, rate_fn, stoichiometry)` | Uma reação: nome, função de taxa e estequiometria. |
| `ReactionNetwork` | classe | `(species, reactions)` | Rede de espécies e reações; `rhs` dá dC/dt. |
| `IntegrationResult` | classe | `(t, C, success)` | Resultado da integração: tempos, concentrações e sucesso. |
| `AdvectionDispersionReactionSolver` | classe | `(network, n_grid, length_m, velocity_m_s, dispersion_m2_s)` | Transporte 1D advecção-dispersão-reação numa grade periódica, com reações de uma `ReactionNetwork`. |
| `mass_action_rate` | função | `(k, orders) -> RateFn` | Constrói a taxa de ação de massas k·∏C^ordem, com k constante ou função de T, pH ou concentrações. |
| `arrhenius_rate_constant` | função | `(A, Ea_over_R) -> Callable[..., float]` | Constante de Arrhenius k(T); levanta erro se chamada sem temperatura. |
| `quadratic_in_T` | função | `(a, b, c) -> Callable[[float], float]` | Função quadrática em T (por exemplo pKa(T) ajustado). |
| `linear_combination_rate_constant` | função | `(terms) -> Callable[..., float]` | Constante efetiva como combinação linear de concentrações de catalisadores. |
| `acid_fraction` | função | `(pH, pKa) -> float` | Fração protonada de um par ácido-base monoprótico (Henderson-Hasselbalch). |
| `base_fraction` | função | `(pH, pKa) -> float` | Fração desprotonada, igual a 1 - `acid_fraction`. |
| `diprotic_fractions` | função | `(pH, pKa1, pKa2) -> tuple[float, float, float]` | Frações de Bjerrum de um sistema diprótico. |
| `integrate_network` | função | `(network, C0, t_eval, T_K=None, pH=None, *, method='Radau', rtol=1e-09, atol=1e-14) -> In…)` | Integra a rede (Radau por padrão), mantendo concentrações não negativas e levantando RuntimeError se o integrador falhar. |

#### `pinneapple_systems.process_components.real_gas_eos`

Propriedades de misturas de gás real por equação de estado de Helmholtz (GERG-2008 via backend HEOS do CoolProp; REFPROP opcional), envelope de validade, condições padrão e derivadas por diferença central fora de grafos de autograd.

<small>`pinneapple_systems/process_components/real_gas_eos.py` · 259 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OutOfEnvelopeError` | classe | `()` | Erro quando um estado cai fora do envelope de validade declarado. |
| `GasComposition` | classe | `(components, mole_fractions)` | Mistura de composição fixa (componentes e frações molares que somam 1). |
| `ValidityEnvelope` | classe | `(P_min_Pa, P_max_Pa, T_min_K, T_max_K)` | Limites (P, T) opcionais; fora deles o estado levanta `OutOfEnvelopeError` em vez de extrapolar. |
| `GasState` | classe | `(P_Pa, T_K, rho_kg_m3, Z, h_J_kg, s_J_kgK, cp_J_kgK, cv_J_kgK, k_isentropic, speed_of_sou…)` | Estado termodinâmico completo em SI (P, T, ρ, Z, h, s, cp, cv, k, som, viscosidade, massa molar). |
| `StandardConditions` | classe | `(P_Pa, T_K)` | Condições padrão de referência (presets ISO 13443 e 60 °F americano). |
| `state_from_PT` | função | `(gas, P_Pa, T_K, *, backend='HEOS', specify_phase='gas', envelope=None) -> GasState` | Estado completo a partir de pressão e temperatura (CoolProp). |
| `state_from_Ph` | função | `(gas, P_Pa, h_J_kg, *, backend='HEOS', specify_phase='gas', envelope=None) -> GasState` | Estado completo a partir de pressão e entalpia (flash na EOS). |
| `state_from_Ps` | função | `(gas, P_Pa, s_J_kgK, *, backend='HEOS', specify_phase='gas', envelope=None) -> GasState` | Estado completo a partir de pressão e entropia (ponto isentrópico). |
| `standard_volumetric_flow_to_mass_flow` | função | `(q_std_m3_s, gas, std) -> float` | Converte vazão volumétrica em condições padrão para vazão mássica. |
| `mass_flow_to_standard_volumetric_flow` | função | `(m_dot_kg_s, gas, std) -> float` | Converte vazão mássica para vazão volumétrica em condições padrão. |
| `central_difference` | função | `(gas, P_Pa, T_K, attr, wrt, *, rel_step=0.0001) -> float` | Derivada de uma propriedade por diferença central, para usar fora de grafos de autograd. |

#### `pinneapple_systems.process_components.similarity_map`

Utilitários de mapas adimensionais de turbomáquinas: coeficientes de vazão e altura, Mach de ponta, Reynolds da máquina, margens de surge e choke e velocidade necessária para uma altura.

<small>`pinneapple_systems/process_components/similarity_map.py` · 194 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MapCoefficients` | classe | `(name, D2_m, n_stages, phi_ref_bep, psi_ref_bep, psi_shape_a, psi0, eta_p_max, eta_p_shap…)` | Coeficientes de um mapa adimensional de turbomáquina (diâmetro, estágios, curvas de ψ e η, surge, choke, correções). |
| `MapEvaluation` | classe | `(phi, psi, eta_p, Mu2, Re, surge_margin_frac, choke_margin_frac)` | Avaliação do mapa num ponto: φ, ψ, η, Mach de ponta, Reynolds e margens de surge e choke. |
| `derive_psi_shape_a` | função | `(psi0, phi_ref_bep, psi_ref_bep) -> float` | Curvatura `a` de ψ(φ) = ψ0 - a·φ² que passa pelo ponto de melhor eficiência. |
| `make_map` | função | `(name, D2_m, phi_ref_bep, psi_ref_bep, psi0, eta_p_max, eta_p_shape_b, phi_surge, phi_cho…)` | Monta `MapCoefficients` derivando a curvatura da curva de altura. |
| `flow_coefficient` | função | `(Q1_m3_s, N_rpm, coeffs) -> float` | Coeficiente de vazão φ = Q1/(N·D2³). |
| `tip_speed_m_s` | função | `(N_rpm, coeffs) -> float` | Velocidade de ponta do impelidor. |
| `tip_mach_number` | função | `(N_rpm, coeffs, speed_of_sound_m_s) -> float` | Mach de ponta U2/a1. |
| `machine_reynolds_number` | função | `(N_rpm, coeffs, kinematic_viscosity_m2_s) -> float` | Reynolds da máquina U2·D2/ν1. |
| `evaluate_map` | função | `(coeffs, Q1_m3_s, N_rpm, speed_of_sound_m_s, kinematic_viscosity_m2_s, Re_ref=None) -> Ma…)` | Avalia o mapa numa vazão e rotação, com correções de Mach e Reynolds e margens. |
| `polytropic_head_from_psi` | função | `(psi, N_rpm, coeffs) -> float` | Altura politrópica total = estágios·ψ·U2². |
| `required_speed_for_head` | função | `(target_head_J_kg, Q1_m3_s, coeffs, speed_of_sound_m_s, kinematic_viscosity_m2_s, *, N_mi…)` | Rotação que produz uma altura alvo a vazão fixa, por bisseção. |

#### `pinneapple_systems.process_components.torsional_stickslip`

Simulação por diferenças finitas da onda torcional 1D num eixo flexível acionado numa ponta e com atrito de Stribeck na outra (mecanismo de stick-slip), com índice de stick-slip.

<small>`pinneapple_systems/process_components/torsional_stickslip.py` · 179 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `StickSlipResult` | classe | `(t_s, z_m, omega_field_rad_s, end_omega_ts_rad_s, surface_torque_ts_N_m, stick_slip_index)` | Resultado: tempos, posições, campo de velocidade angular, velocidade da ponta, torque na superfície e índice de stick-slip. |
| `stribeck_friction_torque` | função | `(omega, T_static, T_kinetic, omega_breakout) -> float` | Torque de atrito de Stribeck (enfraquece com a velocidade). |
| `simulate_torsional_stickslip` | função | `(length_m, G_Pa, J_m4, rho_kg_m3, omega_set_rad_s, T_static_N_m, T_kinetic_N_m, omega_bre…)` | Resolve a onda torcional 1D por diferenças finitas explícitas, da ponta acionada à ponta com atrito e inércia concentrada. |

#### `pinneapple_systems.process_components.water_hammer`

Golpe de aríete numa adutora bombeada: ponto de operação estacionário e transiente de fechamento de válvula pelo método das características, mais fórmulas de Joukowsky e período de onda.

<small>`pinneapple_systems/process_components/water_hammer.py` · 191 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Pipeline` | classe | `(length, diameter, wave_speed, roughness, z_in, z_out, nu, rho)` | Adutora: comprimento, diâmetro, celeridade, rugosidade, cotas, viscosidade e densidade. |
| `PumpCurve` | classe | `(shutoff_head, k, sump_head)` | Curva da bomba de rotação fixa H = H0 - k·Q\|Q\| com carga de sucção. |
| `OperatingPoint` | classe | `(flow, velocity, reynolds, friction_factor, pump_head, valve_head, friction_loss, valve_c…)` | Ponto de operação estacionário: vazão, velocidade, Reynolds, atrito, cargas, Cv da válvula e tanque. |
| `TransientResult` | classe | `(t, x, head, flow, tau, dt, courant, op, info)` | Resultado do transiente MOC: tempos, posições, carga, vazão, abertura, passo, Courant e informações. |
| `operating_point` | função | `(line, pump, *, valve_k, tank_head=0.0) -> OperatingPoint` | Vazão onde a curva da bomba encontra atrito (Colebrook) mais perda da válvula aberta. |
| `linear_closure` | função | `(closure_time, start=0.0)` | Lei de fechamento τ(t): 1 até o início, depois linear até 0. |
| `moc_valve_closure` | função | `(line, pump, op, *, closure_time, t_end, n_reaches=40, friction=True, closure_start=0.0, …)` | Transiente de fechamento de válvula pelo método das características a partir do ponto de operação. |
| `joukowsky_head_rise` | função | `(wave_speed, dv) -> float` | Sobrepressão de Joukowsky a·ΔV/g. |
| `wave_period` | função | `(length, wave_speed) -> float` | Período da oscilação 4L/a. |

#### `pinneapple_systems.sports`

Física aplicada a esportes sobre dados públicos de rastreamento e eventos; por enquanto só futebol. [sem teste direto]

<small>`pinneapple_systems/sports/__init__.py` · 2 linhas · **sem teste direto**</small>

#### `pinneapple_systems.sports.football`

Futebol: análises com física sobre dados públicos (voo da bola, controle de campo, cinemática de jogadores, gols esperados); ver `docs/guides/football_physics.md`.

<small>`pinneapple_systems/sports/football/__init__.py` · 3 linhas · testes: 16</small>

#### `pinneapple_systems.sports.football.ball`

Física do voo da bola (gravidade, arrasto quadrático com crise de arrasto, sustentação de Magnus), problema inverso a partir de posições rastreadas, identificabilidade por Cramér-Rao e modelo de bola rolando para passes rasteiros.

<small>`pinneapple_systems/sports/football/ball.py` · 464 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Ball` | classe | `(mass, diameter)` | Bola: massa e diâmetro; área, constante de arrasto e Reynolds. |
| `FlightFit` | classe | `(model, x0, v0, cd, magnus, g, t0, params, names, cov, rmse, n, success, extra)` | Resultado de `fit_flight`: parâmetros, erros-padrão de Laplace, RMSE, BIC e previsão. |
| `Flight` | classe | `(match_id, t, xyz, detected)` | Um voo aéreo recortado de um rastreamento: partida, tempos, posições e detecção. |
| `cd_crisis` | função | `(re, cd_sub=0.45, cd_super=0.18, re_crit=260000.0, width=25000.0)` | Curva suave de crise de arrasto entre Cd subcrítico e supercrítico. |
| `simulate` | função | `(x0, v0, t, *, cd=0.0, magnus=(0.0, 0.0, 0.0), g=G, ball=BALL, cd_fn=None, dt=0.005) -> n…)` | Posições de uma bola lançada com arrasto, Magnus e gravidade (RK4). |
| `fit_flight` | função | `(t, xyz, model='drag', *, sigma=None, ball=BALL, g=G, observe=(0, 1, 2), cd_prior=None, l…)` | Ajusta um modelo de voo a posições rastreadas (pula linhas NaN), com perda robusta. |
| `predict` | função | `(fit, t, ball=BALL) -> np.ndarray` | Posições de um voo ajustado em novos tempos. |
| `kinematic_extrapolation` | função | `(t, xyz, t_new, degree=2) -> np.ndarray` | Baseline só de dados: polinômio por eixo extrapolado. |
| `extract_flights` | função | `(track, *, match_id=None, z_on=0.6, min_apex=2.5, min_duration=0.8, max_duration=4.5, max…)` | Recorta voos aéreos de um rastreamento de bola por critérios de altura, duração e velocidade. |
| `fisher_identifiability` | função | `(x0, v0, t, *, cd=0.25, magnus=(0.0, 0.0, 0.0), sigma=0.3, observe=(0, 1, 2), ball=BALL, …)` | Desvios de Cramér-Rao dos parâmetros do voo para um ruído de observação. |
| `rolling_fit` | função | `(t, xy, *, ball=BALL) -> dict[str, float]` | Ajusta resistência ao rolamento e arrasto a um passe rasteiro em 2-D. |
| `rolling_travel_time` | função | `(distance, v0, decel, cd=0.0, ball=BALL)` | Tempo para uma bola rolando cobrir uma distância (infinito se parar antes). |
| `travel_time` | função | `(distance, *, speed=None, v0=None, decel=None)` | Tempo de viagem da bola: velocidade constante ou bola rolando. |

#### `pinneapple_systems.sports.football.ball_study`

Estudos do voo da bola: recuperação em voos simulados, identificabilidade 3-D vs 2-D, auditoria de gravidade em voos reais e previsão hold-out contra baselines de dados. [sem teste direto]

<small>`pinneapple_systems/sports/football/ball_study.py` · 235 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `synthetic_flights` | função | `(n=40, *, noise=0.2, fps=10.0, seed=0, g=B.G) -> list[tuple[B.Flight, dict]]` | Chutes simulados com arrasto, rotação e ruído, com a verdade de cada um. |
| `synthetic_recovery` | função | `(n=40, *, noise=0.2, seed=0) -> dict[str, Any]` | Ajusta modelos aos voos sintéticos e mede erro e cobertura dos intervalos de 95 %. |
| `identifiability` | função | `(speed=22.0, elevation_deg=25.0, *, fps=25.0, sigma=0.2, duration=1.6) -> dict[str, Any]` | Compara os desvios de Cramér-Rao de um passe visto em 3-D e em 2-D. |
| `flights_from_skillcorner` | função | `(match_ids, root=None, **kw) -> list[B.Flight]` | Voos aéreos de partidas SkillCorner. |
| `gravity_audit` | função | `(flights, *, max_rmse=0.6, workers=1) -> dict[str, Any]` | Ajusta a gravidade livre em cada voo para checar a consistência física dos dados. |
| `holdout_prediction` | função | `(train, test, *, frac=0.5, train_fracs=(0.3, 0.5, 0.7), seed=0, workers=1) -> dict[str, A…)` | Observa o início de cada voo de teste e prevê o resto; erros por modelo. |
| `summarise` | função | `(res) -> list[dict[str, Any]]` | Tabela por modelo de `holdout_prediction`, melhor aterrissagem primeiro. |

#### `pinneapple_systems.sports.football.data`

Dados públicos de futebol (Metrica, SkillCorner, StatsBomb) baixados em tempo de execução, em cache fora do git, com SHA-256 de cada arquivo num manifesto.

<small>`pinneapple_systems/sports/football/data.py` · 282 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MetricaMatch` | classe | `(game, t, period, players, names, ball, events, fps, gk)` | Uma partida de amostra Metrica em metros (jogadores, bola, eventos, fps, goleiros). |
| `cache_dir` | função | `(root=None) -> Path` | Pasta de cache do futebol (criada se faltar). |
| `manifest` | função | `(root=None) -> dict[str, dict[str, Any]]` | Manifesto URL → caminho, SHA-256 e bytes de cada arquivo baixado. |
| `fetch` | função | `(url, rel, root=None, *, timeout=120.0) -> Path` | Baixa uma URL uma vez para o cache e registra o SHA-256. |
| `metrica_match` | função | `(game, root=None) -> MetricaMatch` | Carrega a partida 1 ou 2 da amostra Metrica (baixa uma vez). |
| `skillcorner_matches` | função | `(root=None) -> list[dict]` | Lista das partidas abertas do SkillCorner (id, times, data). |
| `skillcorner_ball` | função | `(match_id, root=None) -> dict[str, np.ndarray]` | Rastreamento da bola de uma partida SkillCorner (NaN quando falta). |
| `statsbomb_shots` | função | `(seasons=None, root=None, *, max_matches=None) -> list[dict]` | Todos os chutes com freeze frame 360 nas temporadas StatsBomb pedidas. |

#### `pinneapple_systems.sports.football.kin_study`

Estudos de cinemática: suavização com verdade conhecida, incerteza de potência metabólica em dados reais e previsão de trajetória com e sem limites físicos. [sem teste direto]

<small>`pinneapple_systems/sports/football/kin_study.py` · 175 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `synthetic_runs` | função | `(n=20, T=60.0, fps=25.0, *, seed=0, v_cap=9.0, a_cap=6.0, tau=1.0, sigma_a=3.0)` | Trajetórias verdadeiras de jogadores por aceleração Ornstein-Uhlenbeck limitada. |
| `synthetic_smoothing` | função | `(noise=(0.05, 0.2), fps=(25.0, 10.0), *, n=20, seed=0) -> list[dict[str, Any]]` | RMSE de velocidade, aceleração e potência metabólica por método de suavização contra a verdade. |
| `match_load` | função | `(match, *, team='Home', max_frames=None) -> dict[str, Any]` | Potência metabólica de um time numa partida Metrica por jogador e método, e plausibilidade. |
| `prediction_samples` | função | `(match, horizon, *, every=1.0, history=1.0) -> dict[str, np.ndarray]` | Amostras para previsão de posição num horizonte, com estados de Kalman causal. |
| `prediction_study` | função | `(train, test, horizon, *, sizes=(100, 300, 1000, 3000, None), seeds=(0, 1, 2)) -> dict[st…)` | Erro final de deslocamento de cada preditor treinado em subconjuntos de tamanhos diferentes. |

#### `pinneapple_systems.sports.football.kinematics`

Cinemática de jogadores: suavização Kalman-RTS com ruído ajustado por verossimilhança, plausibilidade física, potência metabólica (modelo de inclinação equivalente) e previsão de curto prazo com e sem limites físicos.

<small>`pinneapple_systems/sports/football/kinematics.py` · 300 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MLPredictor` | classe | `(residual_of, project, max_iter, seed, _models)` | Preditor de deslocamento só de dados: árvores com gradient boosting (scikit-learn) sobre `history_features`. |
| `kalman_smooth` | função | `(xy, dt, *, sigma_pos=0.1, jerk=30.0, causal=False, max_gap=1.0) -> dict[str, np.ndarray]` | Posição, velocidade e aceleração suavizadas por Kalman-RTS (ou filtro causal). |
| `fit_noise` | função | `(xy, dt, sigmas=(0.01, 0.02, 0.05, 0.1, 0.2), jerks=(5.0, 10.0, 20.0, 40.0, 80.0))` | Ruído de medição e densidade de jerk por máxima verossimilhança marginal (busca em grade). |
| `finite_differences` | função | `(xy, dt) -> dict[str, np.ndarray]` | Diferenças centrais das posições brutas (baseline sem suavização). |
| `savgol_derivatives` | função | `(xy, dt, window=0.5, order=2) -> dict[str, np.ndarray]` | Posição, velocidade e aceleração por Savitzky-Golay. |
| `plausibility` | função | `(vel, acc, *, v_max=V_MAX, a_max=A_MAX) -> dict[str, float]` | Fração de amostras acima dos limites humanos de velocidade ou aceleração. |
| `minetti_cost` | função | `(i)` | Custo energético de correr numa inclinação (Minetti et al. 2002). |
| `metabolic_power` | função | `(speed, accel_fwd, *, terrain=1.0) -> np.ndarray` | Potência metabólica em W/kg a partir de velocidade e aceleração (inclinação equivalente). |
| `forward_acceleration` | função | `(vel, acc) -> np.ndarray` | Componente da aceleração ao longo da velocidade. |
| `reachable_radius` | função | `(speed0, horizon, *, v_max=V_MAX, a_max=A_MAX)` | Maior distância alcançável num horizonte (limite físico de qualquer previsão). |
| `predict_constant_velocity` | função | `(pos, vel, horizon)` | Posição após o horizonte a velocidade constante. |
| `predict_limited` | função | `(pos, vel, acc, horizon, *, v_max=V_MAX, a_max=A_MAX, tau=0.5, dt=0.04)` | Previsão física: aceleração limitada que decai e velocidade limitada. |
| `project_reachable` | função | `(pos, vel, pred, horizon, *, v_max=V_MAX, a_max=A_MAX)` | Projeta previsões no conjunto alcançável. |
| `history_features` | função | `(pos_hist, ball_rel=None) -> np.ndarray` | Features para o preditor de ML: posições passadas relativas e, opcionalmente, a bola. |
| `displacement_errors` | função | `(pred, true, pos=None, vel=None, horizon=None) -> dict[str, float]` | Erro final médio e percentil 90, e fração de previsões fora do conjunto alcançável. |

#### `pinneapple_systems.sports.football.metrics`

Escores de previsões binárias probabilísticas: log-loss, Brier, AUC, curvas de confiabilidade, ECE e bootstrap pareado de diferenças.

<small>`pinneapple_systems/sports/football/metrics.py` · 78 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `log_loss` | função | `(p, y) -> float` | Log-loss de probabilidades binárias. |
| `brier` | função | `(p, y) -> float` | Escore de Brier. |
| `auc` | função | `(p, y) -> float` | AUC ROC pela fórmula de soma de postos (Mann-Whitney). |
| `reliability` | função | `(p, y, bins=10, strategy='quantile') -> dict[str, np.ndarray]` | Probabilidade média prevista, frequência observada e contagem por faixa. |
| `ece` | função | `(p, y, bins=10) -> float` | Erro de calibração esperado com faixas por quantil. |
| `score` | função | `(p, y) -> dict[str, float]` | Todos os escores de um modelo. |
| `paired_bootstrap` | função | `(p_a, p_b, y, metric='log_loss', n=2000, seed=0) -> dict[str, Any]` | Diferença de uma métrica entre dois modelos nos mesmos eventos com intervalo bootstrap de 95 %. |

#### `pinneapple_systems.sports.football.pc_study`

Estudo de resultado de passes: controle de campo físico contra classificadores só de dados, com o mesmo conjunto de treino, e eficiência de dados. [sem teste direto]

<small>`pinneapple_systems/sports/football/pc_study.py` · 131 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `physics_features` | função | `(ps, p) -> np.ndarray` | Features físicas de um passe: probabilidade de controle e tempos de interceptação menos tempo da bola. |
| `fit_models` | função | `(train, *, seed=0, which=…) -> dict[str, Any]` | Ajusta os modelos (parâmetros físicos ou estimadores) no conjunto de treino. |
| `predict_models` | função | `(models, ps) -> dict[str, np.ndarray]` | Probabilidades previstas por cada modelo ajustado num conjunto de passes. |
| `compare` | função | `(train, test, *, seed=0) -> dict[str, Any]` | Escores no teste, bootstrap pareado de log-loss contra o melhor modelo só de dados e parâmetros físicos ajustados. |
| `data_efficiency` | função | `(train, test, sizes=(25, 50, 100, 200, 400, None), seeds=(0, 1, 2), which=…) -> dict[str,…)` | Log-loss de teste de cada modelo treinado em subconjuntos aleatórios. |
| `with_params` | função | `(p, **kw) -> PC.PCParams` | Cópia de `PCParams` com campos trocados. |

#### `pinneapple_systems.sports.football.pitch_control`

Controle de campo: probabilidade de o time atacante ganhar a bola num ponto, a partir da cinemática dos jogadores e do tempo de viagem da bola; extrai passes do Metrica e ajusta parâmetros por máxima verossimilhança.

<small>`pinneapple_systems/sports/football/pitch_control.py` · 275 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PCParams` | classe | `(tti, v_max, a_max, reaction, sigma, lam, kappa_def, gk_factor, ball, ball_speed, roll, d…)` | Parâmetros do controle de campo (velocidade e aceleração máximas, reação, incerteza, taxa de controle, goleiro, bola, horizonte). |
| `PassSet` | classe | `(att_pos, att_vel, def_pos, def_vel, att_gk, def_gk, ball, target, y, duration, meta)` | Conjunto de passes com o estado no chute (posições e velocidades dos 22 jogadores, bola, alvo, rótulo). |
| `time_to_intercept` | função | `(pos, vel, target, p) -> np.ndarray` | Tempo para jogadores chegarem a um alvo dada posição e velocidade. |
| `control_probability` | função | `(att_pos, att_vel, def_pos, def_vel, ball_pos, target, p=DEFAULT, att_gk=None, def_gk=Non…)` | Probabilidade de o time atacante controlar a bola no alvo (vetorizada). |
| `passes_from_metrica` | função | `(match, *, sigma_pos=0.1, jerk=30.0) -> PassSet` | Passes completados e interceptados de uma partida Metrica com estados de Kalman causal. |
| `fit_ball_travel` | função | `(ps, *, min_len=5.0) -> dict[str, Any]` | Ajusta o tempo de viagem da bola (rolando e velocidade constante) às durações observadas. |
| `fit_params` | função | `(ps, base, names=…, *, max_iter=200) -> tuple[PCParams, float]` | Ajuste por máxima verossimilhança de parâmetros nomeados (Nelder-Mead em log). |
| `geometric_features` | função | `(ps) -> tuple[np.ndarray, list[str]]` | Features só de dados de um passe (distâncias, posições, contagens de defensores). |
| `pitch_surface` | função | `(att_pos, att_vel, def_pos, def_vel, ball_pos, p=DEFAULT, *, nx=53, ny=34, pitch=(105.0, …)` | Probabilidade de controle numa grade do campo para um quadro (para figuras). |

#### `pinneapple_systems.sports.football.plots`

Figuras estáticas (matplotlib, PNG) dos estudos de futebol: campo, superfície de controle, voos 3-D, auditoria de gravidade, confiabilidade, eficiência de dados e erros de suavização. [sem teste]

<small>`pinneapple_systems/sports/football/plots.py` · 224 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `color_of` | função | `(name) -> str` | Cor fixa de um nome de modelo (cinza se desconhecido). |
| `draw_pitch` | função | `(ax, length=105.0, width=68.0, color='#b9b7b0')` | Desenha as linhas do campo em metros. |
| `pitch_control_figure` | função | `(xs, ys, q, att_pos, att_vel, def_pos, def_vel, ball, target, path, title='')` | Figura da superfície de controle com jogadores, bola e alvo. |
| `flights_3d` | função | `(flights, path, title='')` | Vista 3-D de voos observados e ajustados sobre o campo. |
| `gravity_audit_figure` | função | `(audit, synthetic_g, path)` | Figura da auditoria de gravidade. |
| `reliability_figure` | função | `(rel, path, title='', models=None, lim=1.0)` | Curvas de confiabilidade por modelo. |
| `efficiency_figure` | função | `(rows, metric, path, title='', ylabel=None, models=None, flat=None)` | Métrica de teste contra tamanho do treino por modelo, com faixa min-max. |
| `smoothing_figure` | função | `(rows, path, metric='mp_rmse', ylabel='metabolic power RMSE [W/kg]')` | Erros de suavização por método. |
| `holdout_figure` | função | `(res, path, models=None)` | Box plot dos erros de aterrissagem por modelo. |

#### `pinneapple_systems.sports.football.xg`

Gols esperados com e sem física: baseline de geometria do chute contra features de tempo de interceptação calculadas do freeze frame 360 do StatsBomb.

<small>`pinneapple_systems/sports/football/xg.py` · 239 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `XGParams` | classe | `(v_shot, v_gk, reach_gk, reaction_gk, v_def, reach_def, reaction_def, n_mouth, n_path)` | Parâmetros físicos do xG (velocidade do chute, do goleiro e dos defensores, alcance, reação, discretização). |
| `shots_arrays` | função | `(shots) -> dict[str, Any]` | Converte a tabela de chutes em arrays (posições em metros, rótulo, freeze frame). |
| `standard_features` | função | `(A) -> tuple[np.ndarray, list[str]]` | Features padrão de xG: distância, ângulo e contexto do chute. |
| `physics_features` | função | `(A, p=None, *, chunk=200) -> tuple[np.ndarray, list[str]]` | Features físicas: fração do gol aberta, margem do goleiro e bloqueadores. |
| `freeze_geometry_features` | função | `(A) -> tuple[np.ndarray, list[str]]` | Features geométricas do freeze frame sem física. |
| `select_params` | função | `(A_train, grid_v_shot=(12.0, 16.0, 20.0, 25.0, 30.0), grid_v_gk=(3.0, 5.0, 7.0), folds=5)…)` | Escolhe velocidades do chute e do goleiro por validação cruzada agrupada no treino. |
| `compare` | função | `(A_train, A_test, *, params=None, seed=0) -> dict[str, Any]` | Escores de teste dos modelos padrão, físicos e combinados, com o xG do StatsBomb como referência externa. |
| `data_efficiency` | função | `(A_train, A_test, params, sizes=(100, 200, 400, 800, None), seeds=(0, 1, 2, 3, 4))` | Log-loss de teste dos modelos padrão e padrão+física em subconjuntos do treino. |

#### `pinneapple_systems.time_series`

Previsão de séries temporais de sinais físicos (antigo `pinneapple_timeseries`): o `__init__` reexporta specs, datamodule, catálogo, validação, auditoria, features, baselines, previsão adaptativa, wrappers clássicos, preparação, EDA, modelos, decomposição, tuning e visualização.

<small>`pinneapple_systems/time_series/__init__.py` · 108 linhas · testes: 90</small>

#### `pinneapple_systems.time_series.adapters`

Pacote vazio que agrupa adaptadores de forma. [sem teste]

<small>`pinneapple_systems/time_series/adapters/__init__.py` · 1 linhas · **sem teste direto**</small>

#### `pinneapple_systems.time_series.adapters.shape`

Adaptador `LastStepToHorizon`, que repete uma saída de um passo ao longo do horizonte. [sem teste]

<small>`pinneapple_systems/time_series/adapters/shape.py` · 14 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LastStepToHorizon` | classe | `(model, horizon)` | Módulo que repete a saída de um passo do modelo interno `horizon` vezes, para obter a forma (B, H, C). |

#### `pinneapple_systems.time_series.adaptive`

Previsão adaptativa online: especialistas baratos (suavização exponencial, Holt, Holt-Winters, Theta, média móvel, AR ridge, regressor sobre defasagens) combinados ou alternados por pesos Fixed-Share calculados só com erros fora da amostra (prequencial), com intervalos conformais.

<small>`pinneapple_systems/time_series/adaptive.py` · 447 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SimpleExpSmoothing` | classe | `(alpha, level_)` | Especialista de suavização exponencial simples (previsão plana no nível). |
| `HoltDamped` | classe | `(alpha, beta, phi, level_, trend_)` | Especialista de Holt com tendência amortecida por `phi`. |
| `HoltWintersAdditive` | classe | `(season_length, alpha, beta, gamma, level_, trend_, seasonal_, n_)` | Especialista Holt-Winters aditivo com sazonalidade de comprimento `season_length`. |
| `Theta` | classe | `(alpha, level_, slope_, n_)` | Especialista Theta: suavização exponencial com deriva igual à metade da inclinação da regressão. |
| `MovingAverage` | classe | `(window, value_)` | Especialista que prevê a média das últimas `window` observações. |
| `RidgeAR` | classe | `(p, window, ridge, coef_, tail_)` | Especialista autorregressivo de ordem p com intercepto e regularização ridge, previsão recursiva. |
| `LagRegressorExpert` | classe | `(estimator, n_lags=12, refit_every=24, window=500)` | Transforma qualquer regressor tabular estilo scikit-learn em especialista sobre defasagens, com reajuste periódico. |
| `AdaptiveRun` | classe | `(origins, forecast, expert_forecasts, lower, upper, weights, active, expert_names, y)` | Resultado de `AdaptiveForecaster.run`: previsões, previsões por especialista, intervalos, pesos e especialista ativo por origem; calcula erros, cobertura e trocas. |
| `AdaptiveForecaster` | classe | `(experts, horizon=1, mode='combine', etas=(0.5, 2.0, 8.0), alphas=(0.0, 0.01, 0.05), loss…)` | Seleção e combinação online de previsores por pesos exponenciais Fixed-Share por horizonte, com intervalos conformais adaptativos. |
| `default_experts` | função | `(season_length=1) -> dict[str, Any]` | Conjunto diverso e barato de especialistas, com variantes de hiperparâmetros como especialistas separados. |

#### `pinneapple_systems.time_series.audit`

Pacote de auditoria estatística de séries; reexporta `AuditReport`, `AuditSection` e `TSAuditor`. [sem teste direto]

<small>`pinneapple_systems/time_series/audit/__init__.py` · 4 linhas · **sem teste direto**</small>

#### `pinneapple_systems.time_series.audit.report`

Contêineres serializáveis (`AuditReport`, `AuditSection`) para os resultados dos diagnósticos. [sem teste]

<small>`pinneapple_systems/time_series/audit/report.py` · 28 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AuditSection` | classe | `(name, results)` | Seção nomeada de um relatório de auditoria, com dicionário de resultados. |
| `AuditReport` | classe | `(meta, sections)` | Contêiner serializável dos diagnósticos de uma série, organizado por seções. |

#### `pinneapple_systems.time_series.audit.tests`

Testes diagnósticos de séries (ADF, KPSS, Phillips-Perron, ARCH, ACF/PACF, Chow, pontos de mudança via `ruptures`) que devolvem dicionários com `ok`/erro em vez de levantar exceção; depende de `statsmodels`. [sem teste]

<small>`pinneapple_systems/time_series/audit/tests.py` · 315 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TSAuditor` | classe | `(nlags, arch_lags)` | Roda a bateria de diagnósticos (estacionariedade, heterocedasticidade, autocorrelação, quebras de regime) e devolve um `AuditReport`. |
| `adf_test` | função | `(y, *, autolag='AIC', min_n=20) -> Dict[str, Any]` | Teste de raiz unitária Augmented Dickey-Fuller via `statsmodels`. |
| `kpss_test` | função | `(y, *, regression='c', nlags='auto', min_n=20) -> Dict[str, Any]` | Teste de estacionariedade KPSS via `statsmodels`. |
| `pp_test` | função | `(y, min_n=20) -> Dict[str, Any]` | Teste de Phillips-Perron; devolve erro explicativo se a versão do `statsmodels` não tiver a função. |
| `arch_test` | função | `(y, *, lags=12, min_n=30) -> Dict[str, Any]` | Teste ARCH de heterocedasticidade condicional. |
| `acf_pacf` | função | `(y, *, nlags=40, min_n=50) -> Dict[str, Any]` | Calcula autocorrelação e autocorrelação parcial até `nlags`. |
| `chow_test_mean_shift` | função | `(y, break_idx, min_n=20) -> Dict[str, Any]` | Teste de Chow simples para mudança de média num índice de quebra. |
| `changepoints_ruptures` | função | `(y, *, model='l2', max_breaks=5, min_n=50) -> Dict[str, Any]` | Detecção aproximada de múltiplos pontos de mudança com `ruptures`, se instalado. |

#### `pinneapple_systems.time_series.base`

Tipos base: `TSOutput` (saída padrão com `y_hat`) e `TSModelBase` (`nn.Module` cujo `forward` levanta `NotImplementedError`). [sem teste direto]

<small>`pinneapple_systems/time_series/base.py` · 26 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TSOutput` | classe | `(y_hat, extras)` | Saída padrão dos modelos de séries: `y_hat` e um dicionário `extras`. |
| `TSModelBase` | classe | `()` | Base opcional (`nn.Module`) para modelos de séries; seu `forward` levanta `NotImplementedError`. |

#### `pinneapple_systems.time_series.baselines`

Reexporta as baselines ingênuas (`NaiveForecaster`, `SeasonalNaiveForecaster`, `DriftForecaster`). [sem teste direto]

<small>`pinneapple_systems/time_series/baselines/__init__.py` · 2 linhas · **sem teste direto**</small>

#### `pinneapple_systems.time_series.baselines.naive`

Baselines sem aprendizado: último valor, sazonal ingênuo e deriva linear, com interface `fit`/`predict`. [sem teste direto]

<small>`pinneapple_systems/time_series/baselines/naive.py` · 71 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NaiveForecaster` | classe | `(last_value_)` | Prevê o último valor observado para todos os passos futuros. |
| `SeasonalNaiveForecaster` | classe | `(season_length, history_)` | Repete o valor da mesma estação no último ciclo. |
| `DriftForecaster` | classe | `(first_, last_, n_)` | Extrapola linearmente do primeiro ao último ponto de treino. |

#### `pinneapple_systems.time_series.classical_wrappers`

Wrappers reais dos pacotes estatísticos clássicos: ARIMA/SARIMA de `statsmodels` e `pmdarima.auto_arima`, com intervalos de previsão.

<small>`pinneapple_systems/time_series/classical_wrappers.py` · 244 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `StatsmodelsARIMAForecaster` | classe | `(order, seasonal_order, trend, enforce_stationarity, enforce_invertibility, fit_kwargs)` | Previsor ARIMA/SARIMA real sobre `statsmodels` SARIMAX, com intervalos e resumo. |
| `AutoARIMAForecaster` | classe | `(seasonal, m, max_p, max_q, max_d, max_P, max_Q, max_D, stepwise, suppress_warnings, auto…)` | Previsor sobre `pmdarima.auto_arima`, que escolhe a ordem automaticamente. |

#### `pinneapple_systems.time_series.csv_loader`

Carrega um CSV de série temporal em tensores de treino/validação/teste (com preenchimento, normalização por canal e divisão cronológica) e gera um CSV sintético. [sem teste]

<small>`pinneapple_systems/time_series/csv_loader.py` · 322 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TimeSeriesScaler` | classe | `() -> None` | Normalizador z-score por canal: ajusta no treino e aplica a todas as partições. |
| `load_timeseries_csv` | função | `(path, *, target_cols=None, feature_cols=None, time_col=None, fill_method='ffill', normal…)` | Lê um CSV e devolve tensores de treino, validação e teste prontos para o PINNeAPPle. |
| `generate_synthetic_csv` | função | `(path, T=1000, seed=42, freq='D') -> Path` | Gera e salva um CSV sintético multivariado de série temporal. |

#### `pinneapple_systems.time_series.datamodule`

`TSDataModule`: monta o dataset em janelas e loaders com divisões determinísticas e ordenadas no tempo (por índices ou holdout sequencial).

<small>`pinneapple_systems/time_series/datamodule.py` · 80 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TSDataModule` | classe | `(series, spec, batch_size, val_ratio, num_workers, pin_memory)` | Monta o dataset em janelas e os DataLoaders de treino/validação em ordem temporal. |

#### `pinneapple_systems.time_series.datasets`

Pacote vazio que agrupa os datasets. [sem teste]

<small>`pinneapple_systems/time_series/datasets/__init__.py` · 1 linhas · **sem teste direto**</small>

#### `pinneapple_systems.time_series.datasets.public`

Séries reais de licença aberta baixadas de um commit fixo e verificadas por SHA-256 (câmbio USD/EUR diário, manchas solares anuais via `statsmodels`, agregados diários de vídeos em alta do YouTube EUA).

<small>`pinneapple_systems/time_series/datasets/public.py` · 183 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PublicSeries` | classe | `(name, time, values, source)` | Série pública carregada: nome, tempos, valores e fonte. |
| `YouTubeDaily` | classe | `(dates, n_entries, total_views, total_likes, category_counts, source)` | Agregados diários dos vídeos em alta: data, número de entradas, visualizações, curtidas e contagem por categoria. |
| `fetch_pinned` | função | `(key, cache_dir=None, *, local=None) -> bytes` | Bytes de um arquivo fixado (de arquivo local, cache ou rede), com SHA-256 verificado. |
| `load_usd_eur_daily` | função | `(cache_dir=None) -> PublicSeries` | Carrega a taxa USD por EUR em dias úteis (dias sem cotação são descartados). |
| `load_sunspots_yearly` | função | `(cache_dir=None) -> PublicSeries` | Carrega a série anual de manchas solares distribuída com `statsmodels`. |
| `log_returns` | função | `(values) -> np.ndarray` | Calcula os retornos logarítmicos de uma série de valores. |
| `youtube_daily_aggregates` | função | `(csv_text) -> YouTubeDaily` | Agrega as linhas de um CSV de vídeos em alta para uma linha por dia. |
| `load_youtube_trending_us` | função | `(cache_dir=None) -> YouTubeDaily` | Baixa (fixado e verificado) e agrega o CSV de vídeos em alta dos EUA. |

#### `pinneapple_systems.time_series.datasets.windowed`

`WindowedTimeSeriesDataset`: corta uma série (T, F) em pares (janela de entrada, horizonte) com passo e deslocamento. [sem teste direto]

<small>`pinneapple_systems/time_series/datasets/windowed.py` · 41 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `WindowedTimeSeriesDataset` | classe | `(series, spec)` | Dataset PyTorch de pares (janela de entrada, horizonte) cortados de uma série (T, F). |

#### `pinneapple_systems.time_series.decomposition`

Reexporta os previsores por decomposição (FFT, FFT+NN, HHT+NN, FFT+LSTM, HHT+LSTM) e tenta reexportar solvers clássicos de decomposição de `pinneapple_simulation.numerical_solvers` (FFT, wavelet, HHT, EEMD, CEEMDAN, VMD, SSA, SST, STL). [sem teste direto]

<small>`pinneapple_systems/time_series/decomposition/__init__.py` · 53 linhas · **sem teste direto**</small>

#### `pinneapple_systems.time_series.decomposition.fft_forecaster`

Previsor só com FFT: ajusta as K frequências dominantes (com remoção de tendência opcional) e extrapola as senoides.

<small>`pinneapple_systems/time_series/decomposition/fft_forecaster.py` · 86 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FFTForecaster` | classe | `(n_harmonics=10, detrend=True)` | Ajusta as K componentes de frequência dominantes via FFT e as extrapola. |

#### `pinneapple_systems.time_series.decomposition.fft_lstm`

Híbrido FFT + LSTM multivariado: um `FFTForecaster` por canal captura a parte periódica e uma LSTM prevê o resíduo; `FFTLSTMPipeline` orquestra os dois. [sem teste]

<small>`pinneapple_systems/time_series/decomposition/fft_lstm.py` · 335 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FFTDecomposer` | classe | `(n_harmonics=10, detrend=True, auto_harmonics=True, auto_threshold=0.05) -> None` | Ajusta um `FFTForecaster` por canal e calcula resíduos e extrapolações. |
| `ResidualLSTMConfig` | classe | `(input_len, horizon, n_features, n_targets, hidden_size, num_layers, dropout, layer_norm)` | Hiperparâmetros do `ResidualLSTMForecaster`. |
| `ResidualLSTMForecaster` | classe | `(cfg=None, **kw)` | LSTM treinada sobre os resíduos da FFT. |
| `FFTLSTMPipeline` | classe | `(*, n_harmonics=10, detrend=True, auto_harmonics=True, auto_threshold=0.05, input_len=64,…)` | Orquestra decomposição FFT, treino da LSTM nos resíduos e recomposição da previsão. |

#### `pinneapple_systems.time_series.decomposition.fft_nn`

Previsor em dois estágios: FFT para a parte periódica global e uma rede neural treinada no resíduo.

<small>`pinneapple_systems/time_series/decomposition/fft_nn.py` · 100 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FFTNNForecaster` | classe | `(n_harmonics=10, detrend=True, input_len=32, horizon=16, nn_model=None, nn_epochs=50, nn_…)` | Previsor em dois estágios: FFT para a parte periódica e rede neural no resíduo. |

#### `pinneapple_systems.time_series.decomposition.hht_lstm`

Híbrido Hilbert-Huang (EMD) + LSTM multivariado: IMFs dominantes por canal são extrapoladas e a LSTM prevê o resíduo; precisa de `PyEMD`. [sem teste]

<small>`pinneapple_systems/time_series/decomposition/hht_lstm.py` · 341 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HHTDecomposer` | classe | `(n_imfs=6, dominant_energy_ratio=0.8, detrend=True) -> None` | Ajusta um decompositor EMD por canal, calcula resíduos e extrapola as IMFs dominantes. |
| `HHTLSTMPipeline` | classe | `(n_imfs=6, dominant_energy_ratio=0.8, detrend=True, input_len=64, horizon=16, n_features=…)` | Orquestra decomposição HHT, LSTM nos resíduos e recomposição. |

#### `pinneapple_systems.time_series.decomposition.hht_nn`

Previsor em dois estágios: EMD separa IMFs, as dominantes são extrapoladas e uma rede neural prevê o resto; precisa de `PyEMD`.

<small>`pinneapple_systems/time_series/decomposition/hht_nn.py` · 172 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HHTNNForecaster` | classe | `(n_imfs=5, input_len=32, horizon=16, nn_model=None, nn_epochs=50, nn_lr=0.001)` | Previsor em dois estágios: IMFs dominantes da EMD extrapoladas e rede neural no restante. |

#### `pinneapple_systems.time_series.eda`

Reexporta as dez ferramentas gráficas de análise exploratória de `eda.plots`. [sem teste]

<small>`pinneapple_systems/time_series/eda/__init__.py` · 26 linhas · **sem teste direto**</small>

#### `pinneapple_systems.time_series.eda.plots`

Dez gráficos de EDA em matplotlib: tendência, rastreio de setpoint, estatísticas móveis, resposta ao degrau, espectro, ACF/PACF, testes de estacionariedade, pontos de mudança, correlação cruzada e matriz RGA. [sem teste]

<small>`pinneapple_systems/time_series/eda/plots.py` · 463 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_trend` | função | `(y, *, window=24, title='Trend Analysis', figsize=(12, 6), show=False) -> Figure` | Série bruta com média móvel e faixa de desvio-padrão móvel. |
| `track_setpoint` | função | `(actual, setpoint, *, tolerance=0.05, title='Setpoint Tracking', figsize=(12, 5), show=Fa…)` | Valor real contra setpoint, com sombreamento do desvio e faixa de tolerância. |
| `rolling_statistics` | função | `(y, windows=(8, 24, 96), *, stats=('mean', 'std'), title='Rolling Statistics', figsize=(1…)` | Sobreposição de estatísticas móveis em várias janelas. |
| `step_response` | função | `(y, step_idx=None, *, title='Step Response', figsize=(12, 5), show=False) -> Figure` | Mostra a resposta a um degrau (detectado ou dado) e estima ganho, tempo de subida e outras métricas. |
| `power_spectrum` | função | `(y, *, fs=1.0, n_top=5, title='FFT Power Spectrum', figsize=(12, 5), show=False) -> Figure` | Espectro de potência da FFT em escala log, anotando as frequências dominantes. |
| `plot_acf_pacf` | função | `(y, nlags=40, *, alpha=0.05, title='ACF / PACF', figsize=(12, 5), show=False) -> Figure` | ACF e PACF lado a lado com bandas de confiança. |
| `stationarity_report` | função | `(y, *, title='Stationarity Tests', figsize=(10, 4), show=False) -> Figure` | Roda ADF, KPSS e PP e mostra uma tabela-resumo. |
| `changepoint_plot` | função | `(y, n_bkps=3, *, model='rbf', title='Change Point Detection', figsize=(12, 5), show=False…)` | Mostra pontos de mudança detectados com `ruptures`. |
| `cross_correlation` | função | `(df, cols=None, max_lag=20, *, title='Cross-Correlation', figsize=(12, 6), show=False) ->…)` | Mapa de calor da correlação cruzada máxima e da defasagem do pico entre pares de colunas. |
| `rga_matrix` | função | `(K, mv_labels=None, cv_labels=None, *, title='Relative Gain Array (RGA)', figsize=(8, 6),…)` | Calcula e mostra a Relative Gain Array (Bristol) de uma matriz de ganhos. |

#### `pinneapple_systems.time_series.enso`

Pacote do estudo de previsão do ENSO (El Niño): dados, modelos físicos, previsores informados por física e por dados, e avaliação; o `__init__` só tem o docstring.

<small>`pinneapple_systems/time_series/enso/__init__.py` · 2 linhas · testes: 22</small>

#### `pinneapple_systems.time_series.enso.data`

Baixa observações abertas do ENSO (OISST, ERA5 do WeatherBench 2, perfis WOD, conteúdo de calor NCEI, Niño 1+2 do `statsmodels`), reduz para produtos mensais do Pacífico tropical e grava em cache com o SHA-256 de cada arquivo-fonte num manifesto; precisa de rede na primeira execução.

<small>`pinneapple_systems/time_series/enso/data.py` · 674 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Manifest` | classe | `(path, expected=None)` | Manifesto URL -> {sha256, bytes} de cada arquivo-fonte de um produto; um manifesto anterior fixa os hashes esperados. |
| `Gridded` | classe | `(time, lat, lon, values, source)` | Campos mensais (nt, ny, nx) numa grade regular lat/lon, com seleção de região. |
| `manifests` | função | `(cache_dir=None) -> dict[str, Path]` | Lista os manifestos já gravados no cache, por produto. |
| `fetch` | função | `(url, *, timeout=300, retries=4) -> bytes` | Baixa os bytes de uma URL (respeitando o proxy), com novas tentativas e espera exponencial. |
| `month_range` | função | `(start, end) -> list[str]` | Lista de strings 'YYYY-MM' entre duas datas, inclusive. |
| `oisst_monthly` | função | `(start='1981-09', end=None, *, lat=(-30.0, 30.0), lon=(100.0, 300.0), res=1.0, cache_dir=…)` | Médias mensais de TSM do OISST v2.1 numa grade e região dadas. |
| `box_mean` | função | `(g, var, box) -> np.ndarray` | Média ponderada por cos(latitude) de uma variável dentro de uma caixa. |
| `nino_indices` | função | `(g, var='sst') -> dict[str, np.ndarray]` | Médias das caixas Niño 3.4, 3, 4 e 1+2 (não anomalias). |
| `coarsen` | função | `(g, ky, kx) -> Gridded` | Médias em blocos de ky x kx células, tolerante a NaN. |
| `anomalies` | função | `(x, time, base) -> tuple[np.ndarray, np.ndarray]` | Anomalias em relação à climatologia mensal de um período base. |
| `d20_from_profile` | função | `(z, T, *, iso=20.0, max_gap=60.0) -> float` | Profundidade em que um perfil de temperatura cruza a isoterma de 20 °C (D20), por interpolação linear. |
| `fetch_to_file` | função | `(url, path, *, timeout=1800, retries=4, chunk=1 << 22) -> str` | Grava uma URL em arquivo por streaming e devolve o SHA-256. |
| `wod_d20_casts` | função | `(years, kinds=('mrb',), *, lat=(-8.0, 8.0), lon=(120.0, 280.0), cache_dir=None, manifest=…)` | D20 de cada perfil com controle de qualidade do World Ocean Database dentro da caixa. |
| `d20_monthly` | função | `(casts, months, *, lat=(-5.0, 5.0), lon_edges=tuple(range(130, 281, 10)), min_casts=1, da…)` | D20 média mensal por faixa de longitude a partir dos perfis. |
| `d20_climatology` | função | `(casts, lon_edges, lat_edges, base) -> np.ndarray` | Climatologia da D20 por célula (lat, lon) no período base. |
| `era5_monthly` | função | `(variables=…, *, cache_dir=None, manifest=None, workers=8, derived=True, log=print) -> Gr…)` | Médias mensais de campos ERA5 na grade do WeatherBench 2, com variáveis derivadas. |
| `wind_stress` | função | `(u, v, rho_air=1.2, cd=0.0013)` | Tensão do vento pela fórmula bulk com coeficiente de arrasto constante. |
| `soi_from_mslp` | função | `(g, var='mean_sea_level_pressure', base=None) -> np.ndarray` | Índice de Oscilação Sul (Taiti menos Darwin) a partir da pressão ao nível do mar. |
| `ncei_heat_content` | função | `(box=(-5.0, 5.0, 120.0, 280.0), *, cache_dir=None, manifest=None) -> dict[str, Any]` | Soma da anomalia de conteúdo de calor 0-700 m do NCEI na caixa, em médias trimestrais. |
| `statsmodels_nino12` | função | `() -> dict[str, Any]` | Série mensal Niño 1+2 distribuída com `statsmodels`. |

#### `pinneapple_systems.time_series.enso.dataset`

Monta o conjunto de dados do estudo a partir das fontes em cache (índices mensais, mapas de anomalia de TSM, profundidade D20 preenchida por EOF, tensão do vento) e o converte em `SeriesData` do `forecast_lab`.

<small>`pinneapple_systems/time_series/enso/dataset.py` · 226 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EnsoDataset` | classe | `(months, indices, sst_maps, grid, d20_band, d20_eq, d20_lon, d20_observed, taux, base, cl…)` | Conjunto de dados do estudo: meses, índices, mapas de anomalia de TSM, D20, tensão do vento, período base e checagens. |
| `eof_fill` | função | `(X, base_rows, n_eofs=4, ridge=0.1) -> np.ndarray` | Preenche NaNs de cada linha com as EOFs dominantes por mínimos quadrados regularizados. |
| `build_dataset` | função | `(start='1982-01', end='2024-12', base=('1982-01', '2011-12'), grid=None, *, cache_dir=Non…)` | Constrói o `EnsoDataset` a partir das fontes em cache para um intervalo de meses. |
| `to_series` | função | `(ds, columns=COLUMNS, name='enso')` | Converte o `EnsoDataset` em `SeriesData` com as colunas pedidas e os mapas em `meta`. |

#### `pinneapple_systems.time_series.enso.evaluation`

Escores específicos do ENSO sobre os do `forecast_lab`: correlação de anomalia por antecedência, skill por mês alvo (barreira de primavera), verificação categórica El Niño/neutro/La Niña e hindcast com validação cruzada em blocos.

<small>`pinneapple_systems/time_series/enso/evaluation.py` · 96 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `acc` | função | `(mean, y) -> list[float]` | Correlação de anomalia (Pearson sobre as origens) por coluna de horizonte. |
| `by_target_month` | função | `(mean, y, origins, month_of_year, horizons, min_n=6) -> dict[str, Any]` | ACC e RMSE por (mês alvo, horizonte). |
| `category` | função | `(x, threshold=0.5) -> np.ndarray` | Classifica em La Niña (-1), neutro (0) ou El Niño (+1) por um limiar. |
| `categorical` | função | `(mean, y, threshold=0.5) -> dict[str, Any]` | Tabela de contingência 3 x 3 e taxas de acerto por categoria e horizonte. |
| `blocked_hindcast` | função | `(make, data, horizons, *, block=48, first=36, last=None, val_frac=0.15) -> dict[str, Any]` | Previsões fora da amostra para cada origem com validação cruzada em blocos. |

#### `pinneapple_systems.time_series.enso.forecasters`

Previsores do ENSO com a interface `Forecaster` do `forecast_lab`: oscilador de recarga, modelo linear inverso (LIM), CNN sobre mapas, modelo intermediário acoplado com EnKF e ensemble; `ColumnSubset._fit` levanta `NotImplementedError` porque `fit` é sobrescrito.

<small>`pinneapple_systems/time_series/enso/forecasters.py` · 683 linhas · testes: 22 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ColumnSubset` | classe | `(inner, cols, name=None, family=None)` | Embrulha um previsor para que ele veja só algumas colunas. |
| `RechargeOscillatorForecaster` | classe | `(h_col='wwv', seasonal=(), multiplicative=False, stochastic=True, n_members=200, sub=4, s…)` | Previsor pelo oscilador de recarga de Jin sobre (T, h), com ensemble estocástico opcional. |
| `LIMForecaster` | classe | `(n_eofs=10, extra_cols=('wwv',), tau0=1, name=None)` | Modelo linear inverso sobre as PCs dos mapas de TSM e índices extras. |
| `MapCNNForecaster` | classe | `(n_lags=3, pool=2, scalar_cols=('wwv', 'h_west', 'h_east'), channels=(16, 32), hidden=64,…)` | CNN sobre os últimos mapas mensais de anomalia de TSM mais escalares. |
| `IntermediateModelForecaster` | classe | `(init='nudged', n_members=1, enkf_members=24, coupling=1.0, eps_a=0.1, c=2.7, r_days=900.…)` | Previsor pelo modelo intermediário acoplado ajustado no treino, iniciado por nudging ou EnKF. |
| `EnsembleForecaster` | classe | `(members, n_draw=100, seed=0, name=None, family=None)` | Mistura de pesos iguais de membros ajustados juntos. |
| `box_index` | função | `(maps, grid, box=(-5.0, 5.0, 190.0, 240.0)) -> np.ndarray` | Média ponderada por cos(lat) de mapas numa caixa (padrão Niño 3.4). |
| `ro_rhs` | função | `(x, R=-0.05, F1=0.02, F2=1.5, eps=0.03)` | Campo vetorial do oscilador de recarga com coeficientes constantes, para os previsores de EDO. |
| `fit_intermediate` | função | `(meta, end, *, eps_a=0.1, c=2.7, r_days=900.0, coupling=1.0, window=4, ridge=0.001, L_T=6…)` | Ajusta o modelo acoplado intermediário às linhas de treino do estudo. |

#### `pinneapple_systems.time_series.enso.intermediate`

Modelo acoplado intermediário simplificado do Pacífico tropical, no espírito de Zebiak & Cane (1987): oceano de águas rasas em gravidade reduzida, atmosfera de Gill e equação de TSM, com testes de verificação (onda de Kelvin, Rossby, Gill analítico, convergência).

<small>`pinneapple_systems/time_series/enso/intermediate.py` · 607 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OceanGrid` | classe | `(lon0, lon1, dlon, lat_max, dlat)` | Grade C do oceano: h e T nos centros, u e v nas faces. |
| `ShallowWaterOcean` | classe | `(grid=None, c=2.7, H=150.0, r_days=900.0, sponge_deg=5.0, sponge_days=5.0, dtype='float64…)` | Oceano linear de águas rasas em gravidade reduzida no plano beta equatorial. |
| `GillAtmosphere` | classe | `(eps=0.1, c_a=60.0, n_modes=24, dlon=2.5, lon_offset=0.0)` | Resposta de onda longa de Gill (1980) a um aquecimento. |
| `CoupledParams` | classe | `(eps_T, gamma, kappa, mu, mu_y, L_T, noise_tau, noise_months, noise_lon, extra)` | Parâmetros do modelo acoplado (coeficientes de TSM por longitude, ruído). |
| `CoupledModel` | classe | `(params, ocean=None, atmosphere=None, dt_hours=8.0, atm_every=3, closure=None, atm_lat=40…)` | Modelo acoplado linear oceano-atmosfera de anomalias sobre a `OceanGrid`. |
| `gill_analytic` | função | `(x, y, L, eps, period) -> dict[str, np.ndarray]` | Resposta de Gill em forma fechada, usada como referência. |
| `sst_tendency` | função | `(T, h, taux_c, eps_T, gamma, kappa, S)` | Tendência de TSM a partir de T, h e tensão do vento. |
| `kelvin_wave_test` | função | `(grid=None, c=2.7, dt_hours=6.0, days=40.0, sigma_km=1500.0) -> dict` | Teste: solta uma onda de Kelvin equatorial e mede sua velocidade. |
| `rossby_wave_test` | função | `(grid=None, c=2.7, dt_hours=6.0, days=150.0, sigma_km=2500.0) -> dict` | Teste: solta uma onda de Rossby longa n = 1 e mede sua propagação. |
| `gill_test` | função | `(nlons=(72, 144, 288, 576), eps=0.1, L=2.0, n_modes=12) -> dict` | Teste: compara a resposta numérica de Gill com a analítica em várias resoluções. |
| `convergence_study` | função | `(factors=(1, 2, 4), c=2.7) -> dict` | Erro da velocidade da onda de Kelvin sob refinamento conjunto de dx, dy e dt. |
| `forced_ocean_run` | função | `(ocean, taux, *, steps_per_month=91, tauy=None, state=None) -> tuple[np.ndarray, dict]` | Integra só o oceano sob tensão do vento mensal prescrita. |

#### `pinneapple_systems.time_series.enso.oscillators`

Modelos conceituais do ENSO: oscilador de recarga (com coeficientes sazonais, ajuste e simulação estocástica) e oscilador com atraso (raízes por Lambert-W, método dos passos).

<small>`pinneapple_systems/time_series/enso/oscillators.py` · 264 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ROParams` | classe | `(coef, sigma_T, sigma_h, B, info)` | Coeficientes do oscilador de recarga por mês (harmônicos anuais) e amplitudes de ruído. |
| `ro_eigen` | função | `(R, F1, F2, eps) -> dict[str, float]` | Autovalores, taxa de crescimento e período do oscilador de recarga com coeficientes constantes. |
| `ro_coefficients` | função | `(p, month) -> dict[str, np.ndarray]` | Coeficientes do oscilador em meses do calendário (fracionários). |
| `ro_tendency` | função | `(p, X, month) -> np.ndarray` | Tendência determinística (dT/dt, dh/dt) de estados num mês dado. |
| `simulate_recharge` | função | `(p, x0, n_months, *, month0=0.0, sub=10, n_members=1, stochastic=True, seed=0, method='rk…)` | Integra o oscilador de recarga, determinístico ou estocástico, com vários membros. |
| `fit_recharge` | função | `(T, h, month=None, *, seasonal=(), multiplicative=False, lead=1, sub=4) -> ROParams` | Ajusta o oscilador de recarga a séries mensais T e h. |
| `do_rhs` | função | `(x, xt, a=1.0, b=0.75, c=1.0)` | Lado direito do oscilador com atraso de Battisti-Hirst / Suarez-Schopf. |
| `do_roots` | função | `(a, b, tau, branches=(0, -1, 1)) -> list[complex]` | Raízes da equação característica linear do oscilador com atraso via Lambert-W. |
| `simulate_delayed` | função | `(a, b, c, tau, history, n_steps, dt, method='rk4') -> np.ndarray` | Integra o oscilador com atraso pelo método dos passos. |
| `oscillation_stats` | função | `(x, dt, *, skip=0) -> dict[str, float]` | Período e taxa de crescimento de uma série oscilante. |

#### `pinneapple_systems.time_series.enso.pinn`

Redes informadas por física para o ENSO, treinadas com `InversePINN` de `pinneapple_neural` (pesos e parâmetros físicos desconhecidos otimizados juntos): identificação do oscilador de recarga, previsão por extrapolação de PINN e PINN de ondas equatoriais.

<small>`pinneapple_systems/time_series/enso/pinn.py` · 626 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FourierMLP` | classe | `(in_dim, out_dim, hidden=64, layers=3, n_freq=8, out_scale=None, freq_max=None)` | MLP sobre features de Fourier das entradas. |
| `GroupedMLP` | classe | `(groups, in_dim, out_dim, hidden=32, layers=2, n_freq=4, freq_max=2.0, out_scale=None, se…)` | G MLPs de Fourier independentes avaliadas em lote. |
| `PINNRechargeForecaster` | classe | `(h_col='wwv', seasonal=(), closure=False, stochastic=False, n_members=200, epochs=4000, s…)` | Previsor que integra o oscilador de recarga identificado por PINN no treino. |
| `PINNExtrapolationForecaster` | classe | `(h_col='wwv', window=24, epochs=3000, n_col=512, hidden=32, chunk=256, seed=0, name=None)` | Para cada origem, ajusta uma PINN aos últimos meses e às equações do oscilador e extrapola. |
| `EquatorialWavePINN` | classe | `(lon_obs, h_obs, T_obs, lon_f, X0, X2, tau_eq, *, lon_w=120.0, lon_e=280.0, H=150.0, sst=…)` | PINN para o oceano equatorial de ondas longas em dois modos mais uma equação de TSM. |
| `identify_recharge` | função | `(T, h, month=None, *, seasonal=(), closure=False, epochs=4000, n_col=2048, hidden=64, lay…)` | Identificação por PINN dos parâmetros do oscilador de recarga em séries mensais (T, h). |
| `project_forcing` | função | `(taux, lat, c=2.7, H=150.0, rho=1025.0) -> tuple[np.ndarray, np.ndarray]` | Projeções da forçante do vento nas funções de Hermite de ordem 0 e 2. |

#### `pinneapple_systems.time_series.enso.pipeline`

Etapas do estudo ENSO que devolvem resultados serializáveis em JSON (checagens dos solvers, escores, painéis de casos, skill contra referência, barreira de primavera, retrato de fase, estudo PINN de águas rasas, quadros para render, tabelas Markdown).

<small>`pinneapple_systems/time_series/enso/pipeline.py` · 419 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `solver_checks` | função | `() -> dict[str, Any]` | Roda os testes de Kelvin, Rossby, convergência e Gill e compara o oscilador de recarga simulado com a teoria. |
| `enso_scores` | função | `(merged, month_of_year, threshold=0.5) -> dict[str, Any]` | ACC por antecedência, skill por mês alvo e escores categóricos de cada modelo. |
| `case_panels` | função | `(parts, merged, data, *, families=…, events=None, months_before=18, leads=(1, 3, 6, 9, 12…)` | Painéis de plumas para as origens dos estudos de caso, por família de modelo. |
| `skill_against` | função | `(merged, reference='climatology', alpha=0.05) -> dict[str, Any]` | Skill de RMSE contra uma referência e testes de Diebold-Mariano com ajuste de Holm. |
| `spring_barrier_hindcasts` | função | `(data, prot, makers, *, block=48) -> dict[str, Any]` | Hindcasts em blocos com purga no período de desenvolvimento, estratificados por mês alvo. |
| `phase_portrait` | função | `(data, prot, *, seasonal=('R', 'F1', 'F2', 'eps'), seed=0, onsets=…, n_path=48) -> dict[s…)` | Retrato de fase (h, T) observado e trajetórias dos osciladores ajustados. |
| `shallow_water_pinn_study` | função | `(ds, window=('1996-01', '1999-12'), *, epochs=6000, n_col=4096, pde_weight=100.0, spin_mo…)` | Estudo da PINN de ondas equatoriais em dados sintéticos e observados. |
| `slab_frames` | função | `(ds, oisst, casts, start, end, *, spin_months=36, lon=(121.0, 279.0), lat=(-19.5, 19.5)) …)` | Quadros mensais (TSM e termoclina) para o render 3-D. |
| `markdown_tables` | função | `(res, leads=(1, 3, 6, 9, 12, 18, 24)) -> dict[str, str]` | Tabelas Markdown do estudo a partir dos resultados gravados. |

#### `pinneapple_systems.time_series.enso.plots`

Figuras do estudo ENSO em matplotlib no estilo do `forecast_lab`: plumas, skill por antecedência, barreira de primavera, retrato de fase, Hovmöller e validação dos solvers. [sem teste]

<small>`pinneapple_systems/time_series/enso/plots.py` · 217 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_plumes` | função | `(panels, path, *, ncols=3) -> str` | Plumas de distribuição de previsão a partir de uma origem por painel. |
| `plot_skill_vs_lead` | função | `(curves, path, *, metrics=…, labels=None) -> str` | Um painel por métrica com as curvas de skill contra antecedência. |
| `plot_spring_barrier` | função | `(maps, path, *, ncols=3, key='acc') -> str` | Mapas de calor do ACC por mês alvo e antecedência, um por modelo. |
| `plot_phase_portrait` | função | `(traj, path) -> str` | Retratos de fase (h, T) com setas de direção. |
| `plot_hovmoller` | função | `(panels, path, *, ncols=3) -> str` | Diagramas de Hovmöller (longitude x tempo). |
| `plot_validation` | função | `(kelvin, gill, conv, ro, path) -> str` | Figura das checagens dos solvers (Kelvin, Gill, convergência, oscilador). |

#### `pinneapple_systems.time_series.enso.render`

Renders 3-D do Pacífico tropical durante um evento ENSO com o renderizador Blender Cycles da biblioteca (`pinneapple_tools.visualization.studio.render`) e codificação de vídeo com ffmpeg.

<small>`pinneapple_systems/time_series/enso/render.py` · 131 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `slab_scene` | função | `(lon, lat, sst, d20, *, d20_anom=None, band_deg=2.0, z_per_100m=1.0, x_per_deg=0.06, y_pe…)` | Cena 3-D de uma fatia do oceano com TSM na superfície e termoclina em profundidade. |
| `render_frames` | função | `(frames, out_dir, *, sst_range=(22.0, 31.0), size=(960, 540), samples=24, view=(-0.4, -1.…)` | Renderiza uma sequência de quadros mensais em PNG. |
| `encode_video` | função | `(paths, out, fps=8, crf=30) -> str` | Codifica quadros PNG em vídeo MP4 H.264 com ffmpeg. |

#### `pinneapple_systems.time_series.enso.study`

O estudo de previsão do ENSO: protocolo, zoo de modelos e uma chamada por modelo (para rodar em processos paralelos e juntar com `merge_zoo`), além de hindcasts de eventos. [sem teste direto]

<small>`pinneapple_systems/time_series/enso/study.py` · 173 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `protocol` | função | `(months, test_start='2012-01', horizons=HORIZONS, min_history=36) -> Protocol` | Cria o `Protocol` do estudo ENSO (início do teste, horizontes, histórico mínimo). |
| `zoo` | função | `(quick=False) -> list[ModelSpec]` | Lista de `ModelSpec` do zoo; `quick` reduz épocas e membros para testes rápidos. |
| `spec` | função | `(name, quick=False) -> ModelSpec` | Devolve o `ModelSpec` de um modelo do zoo pelo nome. |
| `run_model` | função | `(name, data, prot, *, budget=FP.BUDGET, quick=False, case_origins=None, log=print, seed=0…)` | Faz tuning, ajuste e avaliação de um modelo do zoo. |
| `event_hindcast` | função | `(name, data, event_start, origins, *, quick=False, horizons=HORIZONS, min_history=36, val…)` | Reajusta um modelo e prevê a partir de origens próximas a um evento. |

#### `pinneapple_systems.time_series.features`

Reexporta o gerador de features (`TSFeatureEngineer`, defasagens, estatísticas móveis, Fourier). [sem teste direto]

<small>`pinneapple_systems/time_series/features/__init__.py` · 2 linhas · **sem teste direto**</small>

#### `pinneapple_systems.time_series.features.engineering`

Engenharia de features para ML tabular: defasagens, estatísticas móveis, harmônicos de Fourier, diferenças e janelas deslizantes (X, y). [sem teste direto]

<small>`pinneapple_systems/time_series/features/engineering.py` · 110 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TSFeatureEngineer` | classe | `(lags, rolling_windows, rolling_stats_set, fourier_periods, fourier_K)` | Wrapper que gera um dicionário de features (defasagens, estatísticas móveis, Fourier). |
| `make_lags` | função | `(y, lags) -> Dict[str, np.ndarray]` | Cria features de defasagem de uma série 1D. |
| `rolling_stats` | função | `(y, windows, stats=('mean', 'std', 'min', 'max')) -> Dict[str, np.ndarray]` | Calcula estatísticas móveis da série em várias janelas. |
| `fourier_features` | função | `(t, periods, K=1) -> Dict[str, np.ndarray]` | Gera features de sazonalidade seno/cosseno para cada período. |
| `rate_of_change` | função | `(y, orders=(1, 2)) -> Dict[str, np.ndarray]` | Diferenças de primeira e maior ordem (velocidade, aceleração). |
| `window_features` | função | `(y, input_len, horizon) -> tuple` | Monta arrays de janelas deslizantes (X, y_alvo) para ML supervisionado. |

#### `pinneapple_systems.time_series.forecast_lab`

Laboratório para prever séries quase aleatórias sem se enganar: uma interface comum para baselines, previsores por dados, dados + física, físicos e híbridos, e o protocolo que os compara (tuning em walk-forward com purga, período de teste intocado, escores pontuais e probabilísticos, testes de Diebold-Mariano com controle de comparações múltiplas).

<small>`pinneapple_systems/time_series/forecast_lab/__init__.py` · 44 linhas · testes: 50</small>

#### `pinneapple_systems.time_series.forecast_lab.base`

Interface comum dos previsores do `forecast_lab`: `SeriesData`, `Forecast` (média com desvio gaussiano ou amostras) e a base `Forecaster`, cujo `_predict_mean` levanta `NotImplementedError` se não for implementado. [sem teste direto]

<small>`pinneapple_systems/time_series/forecast_lab/base.py` · 168 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SeriesData` | classe | `(obs, target, dt, name, meta)` | Série observada (T, d), coluna alvo, passo de amostragem e metadados livres. |
| `Forecast` | classe | `(mean, std, samples)` | Distribuição preditiva em (origem, horizonte): média e desvio gaussiano ou amostras. |
| `Forecaster` | classe | `()` | Classe base dos previsores: subclasses implementam `_fit` e `_predict_mean` (ou sobrescrevem `predict`). |
| `train_end` | função | `(train_origins, horizons) -> int` | Índice final exclusivo dos dados que um ajuste nessas origens pode usar. |
| `forecast_origins` | função | `(lo, hi, horizons, *, stride=1, min_history=0) -> np.ndarray` | Origens válidas de previsão dentro de um intervalo, com passo e histórico mínimo. |
| `targets_at` | função | `(data, origins, horizons) -> np.ndarray` | Valores alvo observados em cada origem e horizonte. |

#### `pinneapple_systems.time_series.forecast_lab.baselines`

Baselines que todo previsor precisa vencer: persistência, climatologia, AR(p), ARIMA, suavização exponencial, ingênuo sazonal e climatologia alinhada pela fase. [sem teste direto]

<small>`pinneapple_systems/time_series/forecast_lab/baselines.py` · 278 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Persistence` | classe | `()` | Previsão de passeio aleatório: o último valor. |
| `Climatology` | classe | `()` | Média do treino, com dispersão igual ao seu desvio-padrão. |
| `ARForecaster` | classe | `(max_order=20, criterion='aic', difference=False)` | AR(p) por mínimos quadrados com ordem por AIC. |
| `ARIMAForecaster` | classe | `(p=(0, 1, 2), d=(0, 1), q=(0, 1), max_fit_length=2000)` | ARIMA via SARIMAX do `statsmodels`, ordem escolhida por AIC numa grade pequena. |
| `ExponentialSmoothing` | classe | `()` | Suavização exponencial simples com alpha que minimiza o erro de um passo. |
| `SeasonalNaive` | classe | `(period=None)` | Valor de um período (ou do menor número de períodos) antes do alvo. |
| `PhaseClimatology` | classe | `(period=None, fit_cycles=1.0)` | Ciclo médio alinhado à fase de cada origem. |
| `dominant_period` | função | `(y, min_period=2.0, max_frac=0.25) -> float` | Período do maior pico do periodograma. |

#### `pinneapple_systems.time_series.forecast_lab.hybrid`

Previsores híbridos: modelo físico mais algo aprendido (correção do resíduo por GBM ou rede, e equação diferencial universal com termo neural). [sem teste direto]

<small>`pinneapple_systems/time_series/forecast_lab/hybrid.py` · 181 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ResidualCorrectionForecaster` | classe | `(physical, corrector='gbm', n_lags=8, lag_step=1, max_iter=200, learning_rate=0.05, max_l…)` | Previsão física mais um corretor (GBM ou rede) treinado no resíduo. |
| `UDEForecaster` | classe | `(physical, hidden=64, layers=2, window=4, epochs=200, patience=20, lr=0.001, weight_decay…)` | Equação diferencial universal: física conhecida mais um termo neural, treinada através de RK4. |

#### `pinneapple_systems.time_series.forecast_lab.learned`

Previsores diretos por dados sobre janelas defasadas: gradient boosting e redes neurais (MLP, LSTM, TCN, transformer), com features físicas opcionais e, para o MLP, perda física. [sem teste direto]

<small>`pinneapple_systems/time_series/forecast_lab/learned.py` · 355 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LagFeatures` | classe | `(n_lags=8, lag_step=1, cols=None, extra=None)` | Janelas defasadas das colunas observadas, com features extras opcionais. |
| `GBMForecaster` | classe | `(n_lags=8, lag_step=1, max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_l…)` | Previsor direto com gradient boosting por histogramas do scikit-learn. |
| `NeuralForecaster` | classe | `(arch='mlp', n_lags=16, lag_step=1, hidden=64, layers=2, dropout=0.0, weight_decay=0.0, l…)` | Previsor neural direto multi-horizonte (MLP, LSTM, TCN ou transformer). |
| `PhysicsLossMLP` | classe | `(physical, lam=0.1, **kw)` | MLP que prevê todo o estado em cada passo e é penalizado por violar o modelo físico. |
| `train_torch_regressor` | função | `(build, X_tr, Y_tr, X_va=None, Y_va=None, *, epochs=100, lr=0.001, weight_decay=0.0, batc…)` | Treina um regressor torch com o `Trainer` de `pinneapple_neural`. |

#### `pinneapple_systems.time_series.forecast_lab.physical`

Previsores físicos: equações governantes com parâmetros estimados só no treino (EDO, EDO com atraso, espaço de estados linear, deriva de EDE, EnKF, SINDy) e features físicas para modelos de dados.

<small>`pinneapple_systems/time_series/forecast_lab/physical.py` · 564 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ODEForecaster` | classe | `(rhs, params, fit_params=None, sub=5, fit_steps=2, max_fit_origins=1500, divergence_facto…)` | Previsor por EDO `dx/dt = rhs(x)` com parâmetros ajustados no treino. |
| `DelayODEForecaster` | classe | `(rhs, params, tau, fit_params=None, dt_int=0.1, fit_steps=2, max_fit_origins=1500, diverg…)` | Previsor por EDO com atraso, integrada por RK4. |
| `LinearSSMForecaster` | classe | `(kind='ou', max_fit_length=4000, name=None, family=None)` | Modelo linear gaussiano de espaço de estados (por exemplo Ornstein-Uhlenbeck) com filtro de Kalman. |
| `SDEDriftForecaster` | classe | `(degree=3, n_samples=200, sub=10, closure=None, seed=0, name=None, family=None)` | EDE com deriva polinomial ajustada aos incrementos, com fechamento neural opcional. |
| `EnKFForecaster` | classe | `(physical, n_ens=50, inflation=1.02, q_frac=0.1, spinup=200, seed=0, name=None, family=No…)` | Filtro de Kalman por ensemble estocástico sobre um modelo físico. |
| `SINDyForecaster` | classe | `(threshold=0.05, l2=0.0001, library='poly', sub=5, divergence_factor=5.0, name=None, fami…)` | Equações descobertas por SINDy na trajetória de treino, integradas para prever. |
| `PhysicsFeatures` | classe | `(physical)` | Features de entrada para modelos de dados calculadas por um previsor físico. |

#### `pinneapple_systems.time_series.forecast_lab.pipeline`

Uma chamada por caso do estudo de séries quase aleatórias e as tabelas do guia, montadas a partir dos resultados gravados.

<small>`pinneapple_systems/time_series/forecast_lab/pipeline.py` · 270 linhas · testes: 28</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `fan_panels` | função | `(zoo, data, title, n_origins=2, families=('physical', 'hybrid', 'data')) -> list[dict]` | Dados de gráfico em leque para os representantes de cada família. |
| `surprise_record` | função | `(zoo, data, horizon_index=0) -> dict[str, Any] \| None` | Erro absoluto por origem de teste do melhor modelo e da melhor baseline. |
| `run_case` | função | `(name, *, n=6000, budget=BUDGET, seed=0, cache_dir=None, n_surrogates=99, log=print) -> d…)` | Roda um caso completo: previsibilidade, zoo de modelos, gêmeos treinados em surrogates e leques. |
| `run_overfitting` | função | `(name, *, n=6000, seed=0, cache_dir=None, schemes=…, log=print) -> dict[str, Any]` | Roda o estudo de sobreajuste de um caso com vários esquemas de validação. |
| `lyapunov_study` | função | `(dt_ints=(0.1, 0.05), total_times=(10000.0, 50000.0, 200000.0), seeds=(0, 1, 2), log=prin…)` | Convergência da estimativa de Benettin do expoente de Lyapunov do Mackey-Glass. |
| `memorise_vs_learn` | função | `(name, variants=…, *, horizon_index=0, window=200, n=6000, seed=0, cache_dir=None) -> dic…)` | Compara duas variantes de treino nas últimas origens de treino e nas primeiras de teste. |
| `markdown_tables` | função | `(results, overfit=()) -> dict[str, str]` | Tabelas Markdown do guia a partir dos resultados gravados. |

#### `pinneapple_systems.time_series.forecast_lab.plots`

Figuras estáticas (matplotlib, PNG) de um estudo de previsão: skill por horizonte, sobreajuste, testes com surrogates, PIT, leques, memorizar contra aprender e calendário de surpresas.

<small>`pinneapple_systems/time_series/forecast_lab/plots.py` · 268 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_skill_vs_horizon` | função | `(results, path, *, lyapunov=None, ncols=4, floor=-0.5, titles=None, reference='best_basel…)` | Um painel por série com o skill contra o horizonte de cada família. |
| `plot_overfitting` | função | `(studies, path) -> str` | Erros de treino, validação e teste de cada variante de treino. |
| `plot_surrogate_tests` | função | `(reports, path, *, ncols=4) -> str` | Erro de previsão não linear da série contra seus surrogates. |
| `plot_pit` | função | `(entries, path, *, ncols=4) -> str` | Histogramas de PIT. |
| `plot_fan_charts` | função | `(panels, path, *, ncols=2) -> str` | Distribuições de previsão no período de teste nos horizontes avaliados. |
| `plot_memorise_vs_learn` | função | `(records, path) -> str` | Verdade e previsões das duas variantes de `memorise_vs_learn`. |
| `calendar_grid` | função | `(dates, values) -> dict` | Organiza valores diários numa grade semana x dia da semana. |
| `plot_surprise_calendar` | função | `(records, path, *, vmax=1.0) -> str` | Mapas de calor em calendário dos erros do modelo e da baseline. |

#### `pinneapple_systems.time_series.forecast_lab.presets`

Os zoos de séries e de modelos do estudo de previsão quase aleatória (`examples/near_random_forecasting`).

<small>`pinneapple_systems/time_series/forecast_lab/presets.py` · 345 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Case` | classe | `(name, data, protocol, specs, truth, title)` | Caso do estudo: dados, protocolo, specs dos modelos, verdade e título. |
| `lorenz63_missing_xz_rhs` | função | `(x, sigma=10.0, rho=28.0, beta=8.0 / 3.0)` | Lorenz-63 sem o termo -x z: modelo físico errado de propósito. |
| `baseline_specs` | função | `(arima=True) -> list[ModelSpec]` | Lista de `ModelSpec` das baselines (ARIMA opcional). |
| `data_specs` | função | `(lag_steps=(1,), lags=(8, 16), seq_lags=32, epochs=60, patience=8, max_train=3000, cols=N…)` | Specs de GBM, MLP, LSTM, TCN e transformer sobre janelas defasadas. |
| `case` | função | `(name, *, n=6000, seed=0, cache_dir=None) -> Case` | Constrói um caso do estudo pelo nome. |
| `overfitting_variants` | função | `(c, physics=False) -> dict[str, Any]` | Variantes de treino de um MLP (e de GBM), de muito sobreajustada a regularizada. |

#### `pinneapple_systems.time_series.forecast_lab.renders`

Renders 3-D de um estudo de previsão via Blender Cycles (`pinneapple_blender.cycles`): paisagem de skill, cones de previsão, atratores e fibras de ensemble.

<small>`pinneapple_systems/time_series/forecast_lab/renders.py` · 314 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `landscape_tiles` | função | `(results, *, floor=-0.5, titles=None) -> list[dict[str, Any]]` | Um ladrilho por caso com o skill de cada família por horizonte. |
| `render_landscape` | função | `(tiles, path, *, ncols=4, resolution=(1600, 1000), samples=64, height_scale=1.2) -> str` | Desenha os ladrilhos como superfícies 3-D de skill. |
| `dense_forecast` | função | `(data, protocol, make, max_horizon, *, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict[s…)` | Ajusta e prevê todas as origens de teste em todos os horizontes, com quantis e cobertura. |
| `render_cones` | função | `(lanes, path, *, resolution=(1280, 720), samples=64, frames_dir=None, n_frames=0, history…)` | Desenha cones de previsão (faixas 50 % e 90 %) por raia. |
| `embedding_views` | função | `(x, *, delay=None, n_points=3000, seed=0) -> dict[str, Any]` | Imersões por atraso da série e de seus surrogates. |
| `render_attractors` | função | `(views, path, *, resolution=(1280, 480), samples=64, titles=…) -> str` | Renderiza as imersões por atraso como atratores 3-D. |
| `ensemble_fibres` | função | `(step, x0, *, n_members=50, rel_perturbation=0.01, n_steps=300, scale=None, seed=0, compo…)` | Integra membros perturbados de um estado e devolve suas trajetórias e o espalhamento. |
| `render_ensembles` | função | `(panels, path, *, resolution=(1280, 900), samples=48) -> str` | Desenha as fibras de ensemble com uma parede em um tempo de Lyapunov. |

#### `pinneapple_systems.time_series.forecast_lab.study`

Protocolo de avaliação: fixa primeiro o período de teste, faz tuning em dobras walk-forward com purga no resto, ajusta uma vez, avalia uma vez no teste intocado e testa as diferenças.

<small>`pinneapple_systems/time_series/forecast_lab/study.py` · 450 linhas · testes: 50</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Protocol` | classe | `(horizons, test_frac, val_frac, n_folds, fold_frac, inner_val_frac, min_history, test_str…)` | Divisão de uma série em teste, validação e dobras de tuning, com horizontes e histórico mínimo. |
| `ModelSpec` | classe | `(name, family, factory, grid)` | Um previsor do zoo: fábrica, família e grade de configurações. |
| `tune` | função | `(spec, data, protocol, *, budget=None, seed=0) -> dict[str, Any]` | Avalia configurações nas dobras walk-forward com purga e escolhe a melhor. |
| `evaluate` | função | `(model, data, origins, *, levels=LEVELS, keep=False) -> dict[str, Any]` | Escores pontuais e probabilísticos por horizonte nas origens dadas. |
| `skill_horizon` | função | `(horizons, skill, p_values=None, alpha=0.05) -> int \| None` | Maior horizonte em que o skill é positivo (e significativo, se houver p-valores). |
| `run_zoo` | função | `(data, specs, protocol, *, budget=None, keep_forecasts=(), keep_models=False, log=print, …)` | Faz tuning, ajuste e avaliação de todo o zoo e testa as diferenças. |
| `merge_zoo` | função | `(parts) -> dict[str, Any]` | Junta resultados de `run_zoo` rodados em partes. |
| `overfitting_study` | função | `(data, protocol, variants, *, schemes=('purged_walk_forward',), schemes_for=None, log=pri…)` | Compara erros de variantes de treino sob vários esquemas de validação. |
| `surrogate_trained` | função | `(make, data, protocol, *, kind='iaaft', seed=0) -> dict[str, Any]` | Skill de um modelo treinado em surrogates da série (gêmeo sem estrutura). |

#### `pinneapple_systems.time_series.foundation_models`

Wrappers de modelos fundacionais pré-treinados reais (Amazon Chronos e Google TimesFM) baixados do HuggingFace Hub, para previsão zero-shot; dependem de `chronos-forecasting` e `timesfm`.

<small>`pinneapple_systems/time_series/foundation_models.py` · 379 linhas · testes: 14 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ChronosForecaster` | classe | `(model_id='amazon/chronos-t5-tiny', device_map='cpu', dtype=None, **pipeline_kwargs)` | Wrapper do Chronos pré-treinado da Amazon (pacote `chronos-forecasting`), com previsão por quantis e amostras. |
| `TimesFMForecaster` | classe | `(model_id=…, max_context=512, max_horizon=128, torch_compile=False, **forecast_config_kwa…)` | Wrapper do TimesFM pré-treinado do Google (pacote `timesfm`), com contexto e horizonte máximos. |

#### `pinneapple_systems.time_series.losses`

Perda `mse_forecast` no formato de dicionário (`supervised`, `total`) usado pelos trainers. [sem teste]

<small>`pinneapple_systems/time_series/losses.py` · 9 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `mse_forecast` | função | `(model, y_hat, batch)` | Perda MSE entre `y_hat` e o alvo do batch, no formato de dicionário dos trainers. |

#### `pinneapple_systems.time_series.metrics`

Métricas de previsão em torch (MAE, MSE, RMSE, MAPE, sMAPE, R²) em duas assinaturas: `(y_hat, batch)` e `(y_hat, y)`. [sem teste direto]

<small>`pinneapple_systems/time_series/metrics.py` · 104 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `mae` | função | `(y_hat, batch)` | Erro absoluto médio a partir de `(y_hat, batch)`. |
| `mse` | função | `(y_hat, batch)` | Erro quadrático médio a partir de `(y_hat, batch)`. |
| `rmse` | função | `(y_hat, batch)` | Raiz do erro quadrático médio a partir de `(y_hat, batch)`. |
| `mape` | função | `(y_hat, batch)` | Erro percentual absoluto médio (%) a partir de `(y_hat, batch)`. |
| `smape` | função | `(y_hat, batch)` | Erro percentual absoluto simétrico (%) a partir de `(y_hat, batch)`. |
| `r2` | função | `(y_hat, batch)` | Coeficiente de determinação R² a partir de `(y_hat, batch)`. |
| `mae_fn` | função | `(y_hat, y) -> torch.Tensor` | MAE entre dois tensores. |
| `mse_fn` | função | `(y_hat, y) -> torch.Tensor` | MSE entre dois tensores. |
| `rmse_fn` | função | `(y_hat, y) -> torch.Tensor` | RMSE entre dois tensores. |
| `mape_fn` | função | `(y_hat, y) -> torch.Tensor` | MAPE (%) entre dois tensores. |
| `smape_fn` | função | `(y_hat, y) -> torch.Tensor` | sMAPE (%) entre dois tensores. |
| `r2_fn` | função | `(y_hat, y) -> torch.Tensor` | R² entre dois tensores. |
| `default_ts_metrics` | função | `()` | Dicionário padrão das métricas `mae`, `mse`, `rmse`, `mape`, `smape` e `r2`. |

#### `pinneapple_systems.time_series.metrics_ext`

Reexporta métricas pontuais, probabilísticas e de comparação de previsões.

<small>`pinneapple_systems/time_series/metrics_ext/__init__.py` · 28 linhas · testes: 20</small>

#### `pinneapple_systems.time_series.metrics_ext.comparison`

Comparação de previsões: skill score, teste de Diebold-Mariano (com correção HLN) e correções de múltiplas comparações (Holm, Benjamini-Hochberg).

<small>`pinneapple_systems/time_series/metrics_ext/comparison.py` · 127 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DMResult` | classe | `(statistic, p_value, p_value_better, mean_loss_difference, n, max_lag)` | Resultado do teste de Diebold-Mariano: estatística, p-valores, diferença média de perda, n e defasagem. |
| `skill_score` | função | `(score, reference) -> np.ndarray` | Skill `1 - score / referência`, elemento a elemento. |
| `diebold_mariano` | função | `(e1, e2, *, h=1, loss='se', max_lag=None, hln=True) -> DMResult` | Teste de Diebold-Mariano de igualdade de perda esperada entre duas séries de erros. |
| `holm` | função | `(p_values)` | P-valores ajustados pelo método step-down de Holm. |
| `benjamini_hochberg` | função | `(p_values)` | P-valores ajustados por Benjamini-Hochberg (q-valores). |

#### `pinneapple_systems.time_series.metrics_ext.point`

Métricas pontuais em NumPy (MAE, RMSE, MAPE, sMAPE, MASE, MAE por horizonte) e o agregador `PointMetrics`.

<small>`pinneapple_systems/time_series/metrics_ext/point.py` · 81 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PointMetrics` | classe | `(season_length)` | Calcula de uma vez as métricas pontuais, incluindo MASE com `season_length`. |
| `mae` | função | `(y_pred, y_true) -> float` | Erro absoluto médio em NumPy. |
| `rmse` | função | `(y_pred, y_true) -> float` | Raiz do erro quadrático médio em NumPy. |
| `mape` | função | `(y_pred, y_true, eps=1e-08) -> float` | Erro percentual absoluto médio em NumPy. |
| `smape` | função | `(y_pred, y_true, eps=1e-08) -> float` | Erro percentual absoluto simétrico em NumPy. |
| `mase` | função | `(y_pred, y_true, y_train, season_length=1, eps=1e-08) -> float` | Erro absoluto escalado (MASE) pelo erro do ingênuo sazonal no treino. |
| `multi_horizon_mae` | função | `(y_pred, y_true) -> Dict[str, float]` | MAE global e por passo do horizonte. |

#### `pinneapple_systems.time_series.metrics_ext.probabilistic`

Métricas probabilísticas: pinball, cobertura, largura de intervalo, PIT gaussiano e de ensemble, histograma de PIT e `ProbabilisticMetrics`.

<small>`pinneapple_systems/time_series/metrics_ext/probabilistic.py` · 109 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ProbabilisticMetrics` | classe | `(quantiles)` | Calcula de uma vez as métricas probabilísticas para uma lista de quantis. |
| `pinball_loss` | função | `(y_pred_q, y_true, q) -> float` | Perda pinball de um quantil. |
| `coverage` | função | `(y_lo, y_hi, y_true) -> float` | Fração das observações dentro do intervalo [lo, hi]. |
| `mean_interval_width` | função | `(y_lo, y_hi) -> float` | Largura média dos intervalos. |
| `pit_gaussian` | função | `(mu, sigma, y) -> np.ndarray` | Valores PIT para previsões gaussianas (média e desvio). |
| `pit_ensemble` | função | `(samples, y, rng=None) -> np.ndarray` | PIT aleatorizado de um ensemble (membros no eixo 0). |
| `pit_histogram` | função | `(pit, n_bins=10) -> Dict[str, object]` | Frequências relativas do PIT em classes iguais e seu desvio absoluto médio da uniforme. |
| `gaussian_interval_coverage` | função | `(mu, sigma, y, level=0.9) -> float` | Cobertura empírica do intervalo gaussiano de nível dado. |
| `ensemble_interval_coverage` | função | `(samples, y, level=0.9) -> float` | Cobertura empírica do intervalo de quantis de um ensemble. |

#### `pinneapple_systems.time_series.models`

Reexporta os modelos de previsão (clássicos de ML, LSTM/GRU, N-BEATS, TCN, TFT); o `FNOForecaster` é importado só se disponível. [sem teste direto]

<small>`pinneapple_systems/time_series/models/__init__.py` · 29 linhas · **sem teste direto**</small>

#### `pinneapple_systems.time_series.models.classical`

Previsores de ML tabular com interface `fit(X, y)`/`predict(X)`: XGBoost, LightGBM, RandomForest, CatBoost, GPR e MLP do scikit-learn; as bibliotecas são importadas sob demanda.

<small>`pinneapple_systems/time_series/models/classical.py` · 315 linhas · testes: 151 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ForecastModel` | classe | `()` | Base abstrata dos previsores tabulares (`fit`, `predict`, `predict_with_uncertainty`, `score`); o `fit` base levanta `NotImplementedError`. |
| `XGBoostForecaster` | classe | `(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, …)` | Regressor multi-saída XGBoost como previsor. |
| `LightGBMForecaster` | classe | `(n_estimators=300, num_leaves=31, learning_rate=0.05, subsample=0.8, **kwargs)` | Previsor sobre LightGBM. |
| `RandomForestForecaster` | classe | `(n_estimators=200, max_depth=None, min_samples_leaf=1, **kwargs)` | Previsor RandomForest do scikit-learn, com incerteza pela dispersão das árvores. |
| `CatBoostForecaster` | classe | `(iterations=300, depth=6, learning_rate=0.05, **kwargs)` | Previsor sobre CatBoost. |
| `GPRForecaster` | classe | `(kernel=None, alpha=0.0001, normalize_y=True, **kwargs)` | Regressão por processo gaussiano, naturalmente probabilística, indicada para poucos dados. |
| `MLPForecaster` | classe | `(hidden_layer_sizes=(128, 64), activation='relu', max_iter=1000, learning_rate_init=0.001…)` | Previsor MLP do scikit-learn (`MLPRegressor`). |

#### `pinneapple_systems.time_series.models.fno_forecaster`

Previsor FNO-1D multi-horizonte sobre `FourierNeuralOperator` de `pinneapple_neural`, com histórico e preenchimento com zeros. [sem teste direto]

<small>`pinneapple_systems/time_series/models/fno_forecaster.py` · 73 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FNOForecastConfig` | classe | `(input_len, horizon, n_features, n_targets, width, modes, layers)` | Configuração do `FNOForecaster` (comprimentos, features, largura, modos, camadas). |
| `FNOForecaster` | classe | `(cfg)` | Previsor de séries com FNO-1D: entrada (B, L_in, F), saída (B, H, C). |

#### `pinneapple_systems.time_series.models.nbeats`

N-BEATS (Oreshkin et al., 2019) com pilhas de tendência, sazonalidade e genérica.

<small>`pinneapple_systems/time_series/models/nbeats.py` · 127 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NBeatsConfig` | classe | `(input_len, horizon, n_features, stack_types, n_blocks, n_layers, layer_width, degree_of_…)` | Configuração do N-BEATS (pilhas, blocos, camadas, grau do polinômio, harmônicos). |
| `NBeats` | classe | `(cfg=None, **kw)` | Rede N-BEATS com pilhas de tendência, sazonalidade e genérica que devolve `TSOutput`. |

#### `pinneapple_systems.time_series.models.recurrent`

Previsores LSTM e GRU em PyTorch que saem do último estado oculto para todo o horizonte.

<small>`pinneapple_systems/time_series/models/recurrent.py` · 64 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RecurrentConfig` | classe | `(input_len, horizon, n_features, n_targets, hidden_size, num_layers, dropout, bidirection…)` | Configuração dos previsores LSTM/GRU. |
| `LSTMForecaster` | classe | `(cfg=None, **kw)` | Previsor multi-passo com LSTM (é um `nn.Module`, sem `fit`/`predict`). |
| `GRUForecaster` | classe | `(cfg=None, **kw)` | Previsor multi-passo com GRU (é um `nn.Module`, sem `fit`/`predict`). |

#### `pinneapple_systems.time_series.models.tcn`

Rede convolucional temporal (TCN) com dilatações exponenciais.

<small>`pinneapple_systems/time_series/models/tcn.py` · 74 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TCNConfig` | classe | `(input_len, horizon, n_features, n_targets, n_channels, n_layers, kernel_size, dropout)` | Configuração da TCN. |
| `TCNForecaster` | classe | `(cfg=None, **kw)` | TCN com dilatações exponenciais que devolve `TSOutput`. |

#### `pinneapple_systems.time_series.models.tft`

Temporal Fusion Transformer (Lim et al., 2021) em versão simplificada.

<small>`pinneapple_systems/time_series/models/tft.py` · 197 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TFTConfig` | classe | `(input_len, horizon, n_features, n_targets, hidden_size, num_heads, num_lstm_layers, drop…)` | Configuração do TFT. |
| `TFTForecaster` | classe | `(cfg=None, **kw)` | Temporal Fusion Transformer simplificado que devolve `TSOutput`. |

#### `pinneapple_systems.time_series.predictability`

Medidas de previsibilidade para calcular antes de ajustar qualquer modelo: AR, entropias, expoente de Lyapunov (Rosenstein), surrogates (embaralhado, fase aleatória, IAAFT), erro de previsão não linear e relatório consolidado.

<small>`pinneapple_systems/time_series/predictability.py` · 350 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ARFit` | classe | `(order, intercept, coef, sigma, r2_one_step, criterion)` | Resultado do ajuste AR: ordem, coeficientes, sigma, R² de um passo e critério. |
| `SurrogateTest` | classe | `(statistic, kind, observed, surrogates, p_value, alternative, reject_at_0_05)` | Resultado de um teste com surrogates: estatística observada, valores nos surrogates e p-valor. |
| `fit_ar` | função | `(x, max_order=20, criterion='aic') -> ARFit` | AR(p) por mínimos quadrados com ordem escolhida por AIC/BIC na mesma amostra. |
| `permutation_entropy` | função | `(x, order=5, delay=1, normalize=True) -> float` | Entropia de Shannon dos padrões ordinais. |
| `sample_entropy` | função | `(x, m=2, r=None) -> float` | Entropia amostral `-log(A/B)`. |
| `delay_embedding` | função | `(x, dim, delay) -> np.ndarray` | Imersão por atraso de uma série em dimensão e atraso dados. |
| `first_acf_zero` | função | `(x, max_lag=None) -> int` | Primeira defasagem em que a autocorrelação cai abaixo de 1/e (regra para o atraso). |
| `rosenstein_lyapunov` | função | `(x, *, dim=5, delay=None, dt=1.0, min_tsep=None, n_steps=40, fit=(5, 30), max_points=3000…)` | Maior expoente de Lyapunov estimado pelo método de Rosenstein. |
| `shuffle_surrogate` | função | `(x, rng) -> np.ndarray` | Surrogate por permutação: mantém a distribuição e destrói a estrutura temporal. |
| `phase_randomized_surrogate` | função | `(x, rng) -> np.ndarray` | Surrogate de Fourier: mesmo espectro, fases aleatórias. |
| `iaaft_surrogate` | função | `(x, rng, n_iter=200, tol=1e-08) -> np.ndarray` | Surrogate IAAFT (amplitude e espectro preservados iterativamente). |
| `nonlinear_prediction_error` | função | `(x, *, dim=4, delay=1, horizon=1, k=10, train_frac=0.5) -> float` | Erro fora da amostra de um previsor local por vizinhos mais próximos. |
| `time_reversal_asymmetry` | função | `(x, lag=1) -> float` | Terceiro momento normalizado dos incrementos; zero em média para processos reversíveis. |
| `surrogate_test` | função | `(x, statistic, *, kind='iaaft', n_surrogates=99, alternative='less', seed=0, name=None) -…)` | Teste de posto de uma estatística contra seus valores em surrogates. |
| `baseline_errors` | função | `(x, horizons, train_frac=0.8) -> dict[str, dict[int, float]]` | RMSE da persistência e da climatologia por horizonte. |
| `predictability_report` | função | `(x, *, horizons=(1, 2, 4, 8, 16, 32), dt=1.0, max_ar_order=20, pe_order=5, embed_dim=4, n…)` | Relatório consolidado de previsibilidade (AR, entropias, Lyapunov, surrogates, baselines). |

#### `pinneapple_systems.time_series.preparation`

Reexporta o pipeline de preparação (imputação, outliers, reamostragem). [sem teste direto]

<small>`pinneapple_systems/time_series/preparation/__init__.py` · 6 linhas · **sem teste direto**</small>

#### `pinneapple_systems.time_series.preparation.imputer`

`TimeSeriesImputer`: trata valores ausentes com estratégias de interpolação em camadas, respeitando um tamanho máximo de lacuna.

<small>`pinneapple_systems/time_series/preparation/imputer.py` · 126 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TimeSeriesImputer` | classe | `(strategy='layered', max_gap=None, verbose=False)` | Imputador em camadas de valores ausentes de séries temporais. |

#### `pinneapple_systems.time_series.preparation.outliers`

`OutlierDetector`: detecta outliers (IQR móvel, z-score etc.) e os trata por interpolação, winsorização ou outra opção.

<small>`pinneapple_systems/time_series/preparation/outliers.py` · 155 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OutlierDetector` | classe | `(method='rolling_iqr', window=20, iqr_factor=1.5, z_threshold=3.0, treatment='interpolate…)` | Detector e tratador de outliers em séries temporais. |

#### `pinneapple_systems.time_series.preparation.resampler`

`TimeSeriesResampler`: sobe ou desce a frequência de DataFrames numéricos indexados por tempo.

<small>`pinneapple_systems/time_series/preparation/resampler.py` · 122 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TimeSeriesResampler` | classe | `(target_freq, time_col=None, agg_method='mean', upsample_method='linear', numeric_only=Tr…)` | Reamostra um DataFrame indexado por tempo para a frequência alvo. |

#### `pinneapple_systems.time_series.problem`

`ForecastProblemSpec`: definição formal e imutável do problema de previsão (colunas, horizonte, objetivo, métricas, quantis) a ser feita antes da modelagem. [sem teste direto]

<small>`pinneapple_systems/time_series/problem.py` · 54 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ForecastProblemSpec` | classe | `(freq, time_col, target_cols, feature_cols, exog_past_cols, exog_future_cols, input_len, …)` | Definição formal do problema de previsão; `to_timeseries_spec` gera o `TimeSeriesSpec` de janelamento. |

#### `pinneapple_systems.time_series.registry`

Catálogo de modelos de séries (`TSModelCatalog`) sobre o `ModelRegistry` e registro idempotente do previsor padrão `ts_fno`.

<small>`pinneapple_systems/time_series/registry.py` · 82 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TSModelCatalog` | classe | `(families, default_name)` | Catálogo que lista e constrói modelos de séries no `ModelRegistry` e monta o FNO padrão. |

#### `pinneapple_systems.time_series.solar_cycle`

Envelope do ciclo solar a partir do número diário de manchas (SILSO): EMD/EEMD, a IMF mais próxima de 11 anos, sua amplitude de Hilbert e ajuste por ELM com restrição de periodicidade (TFC), estendido além dos dados.

<small>`pinneapple_systems/time_series/solar_cycle.py` · 274 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Decomposition` | classe | `(imfs, method, backend, trials, noise_width)` | Resultado da decomposição: IMFs, método, backend, tentativas e ruído. |
| `HilbertResult` | classe | `(amplitude, phase, instantaneous_frequency)` | Amplitude, fase e frequência instantâneas de uma IMF. |
| `PeriodicFit` | classe | `(rmse, fitted, forecast_t, forecast, hidden_units, backend, endpoint_error)` | Resultado do ajuste ELM periódico: RMSE, curva ajustada, previsão e erro no ponto final. |
| `parse_silso_csv` | função | `(source)` | Converte um CSV diário do SILSO em DataFrame com índice de datas. |
| `fetch_silso_daily` | função | `(url=SILSO_DAILY_URL, timeout=30.0)` | Baixa e lê a série diária atual do SILSO (rede; falha se não conseguir). |
| `synthetic_daily` | função | `(start='2005-01-01', end='2025-06-30', *, period_years=11.0, peak=180.0, seed=0)` | Série diária sintética com a forma da do SILSO, para testes e demos offline. |
| `preprocess` | função | `(df)` | Normaliza tempo para [0, 1] e valores por min-max. |
| `decompose` | função | `(y, method='emd', *, max_imf=6, trials=16, noise_width=0.05, seed=0, backend='auto') -> D…)` | Decomposição EMD (determinística) ou EEMD (com ruído). |
| `imf_statistics` | função | `(imfs, dt_days=1.0) -> list[dict[str, Any]]` | Estatísticas por IMF, incluindo período dominante em dias e anos. |
| `closest_to_schwabe_cycle` | função | `(stats) -> tuple[dict \| None, str \| None]` | Escolhe a IMF de período mais próximo de 11 anos e avisa se estiver longe. |
| `hilbert_analysis` | função | `(imf, t) -> HilbertResult` | Amplitude, fase e frequência instantâneas de uma IMF pela transformada de Hilbert. |
| `periodic_elm_fit` | função | `(t, y, *, hidden_units=50, forecast_horizon_frac=1.0, n_forecast_points=None, seed=0, bac…)` | Ajuste por mínimos quadrados de uma ELM com base que impõe f(0) = f(1), e extrapolação. |
| `run_pipeline` | função | `(df, *, decomposition_method='emd', max_imf=6, eemd_trials=16, eemd_noise_width=0.05, imf…)` | Executa o pipeline completo: preparo, decomposição, escolha da IMF, Hilbert e ajuste ELM. |

#### `pinneapple_systems.time_series.spec`

`TimeSeriesSpec`: comprimento de entrada, horizonte, passo e deslocamento do alvo.

<small>`pinneapple_systems/time_series/spec.py` · 13 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TimeSeriesSpec` | classe | `(input_len, horizon, stride, target_offset)` | Especificação de janela: `input_len`, `horizon`, `stride` e `target_offset`. |

#### `pinneapple_systems.time_series.synthetic`

Séries sintéticas com verdade conhecida para estudos de previsão: Lorenz-63, Lorenz-96, Mackey-Glass, Ornstein-Uhlenbeck, oscilador estocástico, poço duplo, ruído branco e passeio aleatório, com expoentes de Lyapunov e steppers.

<small>`pinneapple_systems/time_series/synthetic.py` · 364 linhas · testes: 50</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SyntheticSeries` | classe | `(name, t, state, obs, dt, target, observed, params, noise_std, noise_level, lyapunov, kin…)` | Série sintética com sua verdade: estado, observação, parâmetros, ruído e expoente de Lyapunov. |
| `rk4_step` | função | `(f, x, dt)` | Um passo de Runge-Kutta clássico (NumPy ou torch, em lote). |
| `lorenz63_rhs` | função | `(x, sigma=10.0, rho=28.0, beta=8.0 / 3.0)` | Lado direito das equações de Lorenz-63. |
| `lorenz96_rhs` | função | `(x, F=8.0)` | Lado direito das equações de Lorenz-96. |
| `mackey_glass_rhs` | função | `(x, x_tau, beta=0.2, gamma=0.1, n=10.0)` | Lado direito da equação com atraso de Mackey-Glass. |
| `double_well_drift` | função | `(x, a=1.0, b=1.0)` | Deriva de uma partícula no potencial quártico de poço duplo. |
| `mackey_glass_step` | função | `(hist, dt, beta=0.2, gamma=0.1, n=10.0) -> np.ndarray` | Avança históricos de Mackey-Glass por um passo RK4 (método dos passos). |
| `largest_lyapunov_benettin` | função | `(step, x0, *, step_time, n_steps=20000, transient=2000, d0=1e-08, seed=0) -> float` | Maior expoente de Lyapunov pelo método de Benettin. |
| `mackey_glass_lyapunov` | função | `(*, dt_int=0.1, total_time=50000.0, transient_time=2000.0, tau=17.0, beta=0.2, gamma=0.1,…)` | Maior expoente de Lyapunov do Mackey-Glass. |
| `lorenz63` | função | `(n=6000, *, dt=0.05, sub=5, noise=0.1, seed=0, sigma=10.0, rho=28.0, beta=8.0 / 3.0, obse…)` | Gera uma série de Lorenz-63 com ruído de observação. |
| `lorenz96` | função | `(n=6000, *, K=40, F=8.0, dt=0.05, sub=5, noise=0.1, seed=0, lyapunov_steps=20000) -> Synt…)` | Gera uma série de Lorenz-96 com `K` variáveis. |
| `mackey_glass` | função | `(n=6000, *, dt=3.0, dt_int=0.1, tau=17.0, beta=0.2, gamma=0.1, n_exp=10.0, noise=0.1, see…)` | Gera uma série de Mackey-Glass. |
| `ornstein_uhlenbeck` | função | `(n=6000, *, dt=0.1, theta=1.0, sigma=1.0, noise=0.1, seed=0) -> SyntheticSeries` | Gera um processo de Ornstein-Uhlenbeck por amostragem exata. |
| `oscillator_transition` | função | `(omega, zeta, sigma, dt)` | Discretização exata de um oscilador amortecido com ruído branco. |
| `stochastic_oscillator` | função | `(n=6000, *, dt=0.25, omega=1.0, zeta=0.05, sigma=0.3, noise=0.1, seed=0) -> SyntheticSeri…)` | Gera um oscilador amortecido forçado por ruído branco. |
| `double_well` | função | `(n=6000, *, dt=0.1, sigma=0.7, sub=100, a=1.0, b=1.0, noise=0.1, seed=0) -> SyntheticSeri…)` | Gera dinâmica de Langevin num poço duplo (Euler-Maruyama). |
| `white_noise` | função | `(n=6000, *, seed=0) -> SyntheticSeries` | Gera ruído branco (sem nada a prever). |
| `random_walk` | função | `(n=6000, *, seed=0) -> SyntheticSeries` | Gera um passeio aleatório. |
| `ode_stepper` | função | `(rhs, dt, sub=1) -> Callable[[np.ndarray], np.ndarray]` | Mapa que avança um lote de estados de uma EDO por `dt` com RK4. |
| `mackey_glass_stepper` | função | `(dt_int=0.1, sub=10, **params) -> Callable[[np.ndarray], np.ndarray]` | Mapa que avança históricos de Mackey-Glass. |
| `linear_sde_stepper` | função | `(Phi, Q, seed=0) -> Callable[[np.ndarray], np.ndarray]` | Mapa `x -> Phi x + w` para uma EDE linear discretizada. |

#### `pinneapple_systems.time_series.tuning`

Reexporta os afinadores Optuna e `temporal_split`. [sem teste direto]

<small>`pinneapple_systems/time_series/tuning/__init__.py` · 4 linhas · **sem teste direto**</small>

#### `pinneapple_systems.time_series.tuning.optuna_tuner`

Busca de hiperparâmetros com Optuna para os previsores clássicos e neurais, com divisão temporal em três partes.

<small>`pinneapple_systems/time_series/tuning/optuna_tuner.py` · 265 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ClassicalTuner` | classe | `(model_type='xgboost', n_trials=30, direction='minimize', metric='mae', timeout=None)` | Busca de hiperparâmetros com Optuna para previsores de ML clássicos. |
| `NeuralTuner` | classe | `(model_type='lstm', n_trials=20, epochs_per_trial=20, lr=0.001, batch_size=64, input_len=…)` | Busca de hiperparâmetros com Optuna para previsores neurais (LSTM etc.). |
| `temporal_split` | função | `(X, y, val_frac=0.15, test_frac=0.15)` | Divisão cronológica em treino, validação e teste. |

#### `pinneapple_systems.time_series.uncertainty`

Reexporta `QuantileConfig`, `QuantileHead` e `QuantileLoss`. [shim] [sem teste]

<small>`pinneapple_systems/time_series/uncertainty/__init__.py` · 2 linhas · **sem teste direto**</small>

#### `pinneapple_systems.time_series.uncertainty.quantile`

[shim] Reexporta a regressão por quantis de `pinneapple_analysis.uncertainty.quantile` para usuários antigos de `pinneapple_timeseries`. [sem teste]

<small>`pinneapple_systems/time_series/uncertainty/quantile.py` · 8 linhas · **sem teste direto**</small>

#### `pinneapple_systems.time_series.validation`

Reexporta os divisores de validação e o executor de backtest.

<small>`pinneapple_systems/time_series/validation/__init__.py` · 16 linhas · testes: 20</small>

#### `pinneapple_systems.time_series.validation.backtest`

`BacktestRunner`: treina e avalia um modelo em cada dobra de validação temporal e agrega as métricas.

<small>`pinneapple_systems/time_series/validation/backtest.py` · 111 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BacktestConfig` | classe | `(reset_model_each_fold, shuffle_train, val_batch_size)` | Configuração do backtest (reiniciar modelo por dobra, embaralhar treino, batch de validação). |
| `BacktestResult` | classe | `(folds, agg)` | Resultado do backtest: métricas por dobra e agregadas. |
| `BacktestRunner` | classe | `(*, trainer_factory, model_factory, datamodule, splits, train_cfg, preprocess_factory=Non…)` | Treina e avalia um modelo em cada dobra temporal. |

#### `pinneapple_systems.time_series.validation.splitters`

Divisores de validação temporal: janela expansiva, janela móvel, walk-forward e k-fold com purga/embargo, e k-fold embaralhado (mantido só para medir o otimismo).

<small>`pinneapple_systems/time_series/validation/splitters.py` · 163 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Split` | classe | `(fold, train_idx, val_idx, test_idx)` | Uma dobra de backtest em índices do dataset em janelas. |
| `ExpandingWindowSplitter` | classe | `(initial_train_size, val_size, step_size, gap, max_folds)` | Walk-forward com janela de treino crescente. |
| `RollingWindowSplitter` | classe | `(train_size, val_size, step_size, gap, max_folds)` | Walk-forward com janela de treino de tamanho fixo. |
| `PurgedWalkForwardSplitter` | classe | `(n_folds, val_size, purge, embargo, window, min_train)` | Walk-forward sobre origens de previsão com purga e embargo contra vazamento. |
| `PurgedKFoldSplitter` | classe | `(n_folds, purge, embargo)` | K dobras contíguas com purga antes e embargo depois da validação. |
| `ShuffledKFoldSplitter` | classe | `(n_folds, seed)` | K-fold embaralhado comum, que ignora o tempo; mantido só para medir o otimismo. |

#### `pinneapple_systems.time_series.viz`

Reexporta os gráficos de previsão e a animação. [sem teste]

<small>`pinneapple_systems/time_series/viz/__init__.py` · 18 linhas · **sem teste direto**</small>

#### `pinneapple_systems.time_series.viz.animation`

Animação de previsão móvel com bandas de incerteza e exportação opcional para MP4. [sem teste]

<small>`pinneapple_systems/time_series/viz/animation.py` · 114 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `animate_rolling_forecast` | função | `(y_true, y_pred_seq, y_std_seq=None, horizon=16, context_len=64, interval=80, sigmas=(1.0…)` | Anima a previsão móvel com bandas de incerteza; exporta MP4 opcionalmente. |

#### `pinneapple_systems.time_series.viz.forecast`

Gráficos de previsão: previsão móvel com bandas, horizonte com contexto, paridade, resíduos e comparação de backtests. [sem teste]

<small>`pinneapple_systems/time_series/viz/forecast.py` · 209 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_rolling_forecast` | função | `(y_true, y_pred, y_std=None, sigmas=(1.0, 2.0, 3.0), title='Rolling Forecast', xlabel='Ti…)` | Previsão móvel com bandas de 1, 2 e 3 sigmas. |
| `plot_forecast_horizon` | função | `(y_context, y_true, y_pred, y_std=None, sigmas=(1.0, 2.0), title='Forecast', figsize=(12,…)` | Contexto, verdade futura e previsão com bandas. |
| `plot_parity` | função | `(y_true, y_pred, title='Parity Plot', figsize=(5, 5), ax=None)` | Dispersão real contra previsto com diagonal de referência. |
| `plot_residuals` | função | `(y_true, y_pred, title='Residual Analysis', figsize=(14, 4))` | Três painéis de resíduos: no tempo, histograma e Q-Q. |
| `plot_backtest` | função | `(y_true, predictions, title='Backtest Comparison', figsize=(14, 5))` | Sobrepõe previsões de vários modelos à verdade. |

### Exemplos de uso reais do repositório

**Previsão adaptativa com especialistas padrão** — `examples/time_series/adaptive_forecasting.py`

```python
def main(out="adaptive_forecasting.png"):
    y = make_series()
    run = AdaptiveForecaster(default_experts(season_length=12), horizon=3).run(y, start=60)
    err = run.errors(h=1)
    for name in sorted(err, key=lambda k: err[k]["mae"]):
        print(f"{name:26s} MAE {err[name]['mae']:.3f}")
    print("interval coverage (90 % target):", round(run.coverage(1), 3))
    print("leading model changes (first 10):", run.switches()[:10])
```

**Execução de um grafo de co-simulação com gravador** — `examples/pinneapple_cosim/01_spring_mass_pinn.py`

```python
engine_eval = CoSimEngine(
    graph_eval, recorder=recorder, loop_solver="gauss_seidel", max_iter=1
)
engine_eval.reset()
engine_eval.initialize_ports(_INIT_PORTS)

with torch.no_grad():
    engine_eval.run(T=T_END, dt=DT)

traj_x1 = recorder.get("mass1", "x1")
traj_v1 = recorder.get("mass1", "v1")
```

**Sinal PRBS de dois níveis** — `tests/test_excitation.py`

```python
def test_prbs_two_levels_and_switches_only_on_clock():
    n, dwell, lo, hi = 1000, 0.05, -2.0, 3.0
    u = actuator_series("prbs", n, lo=lo, hi=hi, seed=7, dwell=dwell)
    assert set(np.unique(u)) == {lo, hi}
    hold = int(round(dwell * n))
    switches = np.flatnonzero(np.diff(u)) + 1           # first sample of each new level
    assert switches.size > 0 and np.all(switches % hold == 0)
```

**Gêmeo que sabe quando errou (missão completa)** — `examples/autolab_digital_twin/twin_knows_when_wrong.py`

```python
        run = run_mission(a.text, out_dir=a.out, planner=a.planner, render=render, n_per_kind=n, workers=a.workers)
        st = run.state
        b = st["benchmark"]["benchmark"]
        ex.log_metric("cause_accuracy", b["accuracy"], "fraction", f"{b['n_cases']} controlled cases")
        ex.log_metric("cases_with_false_alarm", b["cases_with_false_alarm"], "cases",
                      "alarm before the fault, or any alarm on a healthy case")
```

**Comissionar o gêmeo do trocador de calor** — `tests/test_digital_twin_product.py`

```python
    spec = TS.from_yaml(APP / "spec_hx201.yaml")
    hist = generate_history(hours=36.0, sampling_s=600.0)

    def model(X, theta):
        return exchanger_model(X, theta, ua_clean=10_200.0)

    ev = commission(spec, model, hist, n_ens=48, seed=0, warmup_steps=10)
    assert ev.commissioned, ev.render()
    assert ev.tracking.alarm_lead_s.get("T_ho", 0) > 0
```

**Gêmeo ao vivo do tanque via MQTT embutido** — `tests/test_operations_plant_twin.py`

```python
    lt = LiveTwin(make_twin(LumpedTankModel()), speed=1000.0).start()
    try:
        deadline = time.time() + 10
        while lt.twin.n_updates < 30 and time.time() < deadline:
            time.sleep(0.1)
        assert lt.twin.n_updates >= 30 and lt.broker is not None and lt.broker.n_published >= 30
        f = lt.inject_fault("leak", 5e-4)
        assert f["kind"] == "leak" and lt.sim.faults
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 104 de 182.
- Funções de teste em arquivos que importam algum módulo do pacote: 579.

Itens do catálogo de métodos (`pinneapple_catalog/method_status.json`) cujo código está neste pacote:

| id | método | status | testes contra referência | outros testes |
|---|---|---|---:|---:|
| E22 | Euler-Bernoulli beam | validated | 1 | 11 |
| E23 | von Karman beam / plate | validated | 2 | 6 |
| E44 | Reaction network / Arrhenius / mass action | tested | 0 | 10 |
| E47 | Colebrook-White | tested | 0 | 8 |
| E49 | Herschel-Bulkley + Metzner-Reed (non-Newtonian pipe flow) | tested | 0 | 4 |
| E51 | Lame thick-walled cylinder | tested | 0 | 3 |
| E52 | von Mises equivalent stress | tested | 0 | 11 |
| E53 | Euler buckling + constrained rod buckling | validated | 1 | 6 |
| E54 | Beam-column moment amplification factor | tested | 0 | 3 |
| E55 | S-N curve + Goodman + Miner | validated | 1 | 14 |
| E56 | Torsional stick-slip with Stribeck friction | tested | 0 | 6 |
| E57 | Heat exchanger (effectiveness-NTU) | tested | 0 | 8 |
| E59 | Control valve (Cv, compressible / incompressible) | tested | 0 | 1 |
| E60 | Polytropic compression path | tested | 0 | 1 |
| E61 | Real gas equation of state (CoolProp) | tested | 0 | 2 |
| E62 | Compressor similarity map | validated | 2 | 0 |
| S20 | Co-simulation (graph + engine, algebraic loops by Tarjan SCC) | tested | 0 | 11 |
| S21 | Digital twin state estimation: KF/EKF/UKF/EnKF, prognostics, signal reconstruction | tested | 2 | 39 |
| S23 | Explicit equation-system solver (evaluation order) | tested | 0 | 1 |
| S25 | Causal telemetry conditioning (Hampel one-step, rate limit, first-order filter, gaps) | validated | 1 | 6 |
| T13 | Uncertainty: heteroscedastic, B-PINN, MC Dropout, deep ensembles | tested | 0 | 20 |

### Limitações conhecidas

- O atalho `forecast` da raiz chama `cls(horizon=...)`, `fit(data)` e `predict(horizon)`, mas `LSTMForecaster`, `GRUForecaster`, `NBeats`, `TCNForecaster` e `TFTForecaster` são `nn.Module` sem `fit`/`predict`, e o padrão `auto` vira `lstm`; o atalho não tem teste e não parece funcionar como descrito.
- `cosim_from_models` cria todos os nós com portas fixas `input` e `output`; conexões como `fluid.T` do exemplo no docstring falham na validação de portas.
- Muitas dependências são opcionais e importadas sob demanda: `statsmodels`, `pmdarima`, `ruptures`, `PyEMD`, `xgboost`, `lightgbm`, `catboost`, `optuna`, `sympy`, `chronos-forecasting`, `timesfm`, Blender (`pinneapple_blender`) e ffmpeg; sem elas, as funções correspondentes falham ou devolvem erro.
- Os módulos `enso.data`, `datasets.public`, `solar_cycle.fetch_silso_daily` e `foundation_models` precisam de rede para baixar dados ou pesos; os dados do SILSO têm licença não comercial (CC BY-NC 4.0).
- `NotImplementedError`: `TSModelBase.forward`, `ForecastModel.fit`, `Forecaster._predict_mean` (base), `ColumnSubset._fit` (não usado) e `ChronosForecaster.predict_samples` para pipelines Chronos sem amostragem.
- `uncertainty` e `uncertainty.quantile` são shims que reexportam de `pinneapple_analysis`; o docstring de `foundation_models` cita um `time_series.classical_ts` que não existe no pacote.
- Vários módulos não têm teste direto: `cosimulation.connection`, `cosimulation.utils`, `audit`, `csv_loader`, `eda`, `viz`, `features`, `baselines.naive`, `fft_lstm`, `hht_lstm`, `metrics`, `enso.plots`, `enso.study` e partes do `forecast_lab` (`base`, `baselines`, `hybrid`, `learned`).
- `registry` importa `fno_forecaster` sem `try/except`, então importar `time_series` exige `pinneapple_neural` com o FNO, embora `models/__init__` trate esse import como opcional.
- `digital_twin.assimilation.kalman` é só um shim: os filtros EKF/EnKF vivem em `pinneapple_analysis.state_estimation.kalman`.
- Muitas partes dependem de bibliotecas opcionais: PyTorch (`DigitalTwin`, reconstrução por modelo), paho-mqtt, kafka-python, asyncua, pymodbus (fluxos), PyWavelets, PyYAML (spec), BaSyx (AAS), trimesh, CasADi (MPC/ROM), psycopg (Postgres), Blender `bpy` (renders) e CoolProp (todo `real_gas_eos` e o que depende dele).
- O broker MQTT embutido não tem persistência, QoS 2, will nem TLS; o próprio docstring manda usar um broker de produção diante de equipamento real. A fonte OPC UA de `operations.live` é marcada como experimental.
- `operations.control` funciona só em simulação/recomendação: `recommend` não envia nada à planta. As plantas de `operations` (tanque, trocador) e de `pipe_loop` são simuladas; o histórico do trocador é declaradamente sintético.
- Vários módulos não têm teste que os importe diretamente: `realtime_pipeline`, `signal_reconstruction`, `state`, os módulos internos de `product` e de `pipe_loop`, `ball_study`, `kin_study`, `pc_study` e `plots` (alguns são cobertos pelos testes do pacote pai).
- Escopos declarados nos docstrings: `pipe_network_1d` cobre um único segmento de tubo; o transporte de `reaction_kinetics` só aceita grade periódica; `curved_path_geometry` é só 2D (o caso 3D não está implementado); `recession` é um só reservatório exponencial, sem chuva nem roteamento.
- A política de deriva de `product` detecta que o gêmeo errou, não o porquê, e nunca re-treina sozinha; a spec não valida a física, só que existe uma afirmação falsificável.
- Futebol: os dados (Metrica, SkillCorner, StatsBomb) são baixados em tempo de execução e têm termos de uso próprios (StatsBomb não comercial). O veredito gravado em `docs/guides/football_physics.md` registra que a física não ajudou no xG nem no ponto de aterrissagem contra árvores treinadas.
- `physics_residuals` levanta `NotImplementedError` quando o modelo não tem saídas suficientes para o resíduo pedido; os resíduos são estacionários e simplificados (sem força de corpo, decaimento de primeira ordem).
- `SWAGApproximation` usa só a variância diagonal, não a parte de baixo posto do SWAG completo.
- `edge` exige onnx/onnxruntime; `EdgeRuntime` não importa torch nem pinneapple.
- Os seis componentes de referência são exemplos de arquitetura e física, sem dados reais; o exemplo usa alvos sintéticos aleatórios.
- O conjunto de toolboxes instaladas vive só na memória do processo (`ToolboxRegistry`).
- Sem teste direto pelo inventário: todos os módulos de `component_library` exceto o `__init__`, e `component_modeling` `mc_dropout`, `mpc` e o `__init__`.

## 3. `pinneapple_twin3d`

### Objetivo

`pinneapple_twin3d` é a camada de visualização 3D de gêmeos digitais do PINNeAPPle. Você monta uma `Scene` em Python (partes de malha, campos por vértice ao longo do tempo, movimentos, visibilidade, propriedades físicas e sensores com séries e envelopes de alarme) e exporta para uma pasta estática: `scene.json`, `geometry.glb`, `fields.bin` e o visualizador web embutido (pasta `viewer/`). `serve` abre essa pasta por HTTP.

O pacote resolve o problema de mostrar resultados de simulação e de monitoramento sobre a geometria real, com o tempo como eixo. Há várias entradas de geometria: primitivas (`primitives`), a biblioteca de geometrias de `pinneapple_design` (`library`), casos OpenFOAM com valores por face (`openfoam`), scans limpos e cenas MuJoCo (`scan`). Além do glTF, `usd` exporta OpenUSD com campos como time samples e física `UsdPhysics`, para Omniverse e Isaac Sim; a cena também vira cena de `pp.viz` (`to_studio`) e pode ser renderizada no Blender.

O subpacote `wear` é um gerador completo de gêmeos de desgaste de revestimento (refratário, liner, tubo): especificação do vaso por zonas de revolução, dataset de espessura, previsão linear de vida restante com intervalo e confiança, backtest, otimização de paradas de manutenção (apoio à decisão), calibração a partir de registros e campanhas sintéticas claramente rotuladas. Segundo o docstring do pacote, o visualizador fica na biblioteca para ser reaproveitado pelo PINNeAPPle-CFD, pelos apps e por `pinneapple_systems.digital_twin`.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 19 | 2.820 | 19 | 67 | 74 (7 arquivos) | 4 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `demo` | Cena de demonstração de curva de tubo com desgaste ilustrativo. | 1 | 1 | 6 |
| `gallery` | Galeria de peças variadas construídas com primitivas, com campos sintéticos. | 1 | 11 | 16 |
| `library` | Alias da biblioteca de geometrias de `pinneapple_design`. | 1 | 0 | 0 |
| `openfoam` | Leitura de casos OpenFOAM para cenas com valores por face. | 1 | 6 | 6 |
| `primitives` | Primitivas de malha (revolução, varredura, loft, asa, caixa, cilindro). | 1 | 12 | 16 |
| `scan` | Limpeza de malhas de scan e importação de cenas MuJoCo. | 1 | 4 | 4 |
| `scene` | Modelo da cena e exportação para glTF + campos + visualizador. | 1 | 5 | 9 |
| `serve` | Servidor HTTP da pasta exportada. | 1 | 1 | 6 |
| `usd` | Exportação OpenUSD com campos no tempo e física. | 1 | 1 | 5 |
| `wear` | Gerador de gêmeos de desgaste: modelo, previsão, otimização, calibração e presets. | 9 | 45 | 30 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_twin3d`

Fachada do pacote: reexporta `Scene` e `serve` para montar uma cena de gêmeo digital 3D em Python e abri-la no visualizador web embutido.

<small>`pinneapple_twin3d/__init__.py` · 26 linhas · testes: 34</small>

#### `pinneapple_twin3d.demo`

Cena de demonstração: curva de tubo de 90 graus com desgaste por erosão crescendo no tempo e dois sensores; o campo é ilustrativo, não resultado de CFD.

<small>`pinneapple_twin3d/demo.py` · 68 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `pipe_bend_scene` | função | `(years=(0, 1, 2, 3, 4, 5)) -> Scene` | Monta a cena de demonstração da curva de tubo com campo de desgaste ilustrativo por ano e dois sensores. |

#### `pinneapple_twin3d.gallery`

Galeria de gêmeos 3D de peças muito diferentes (canal de corrida, tanque, oleoduto, avião, foguete, drone, rover lunar, turbina eólica, satélite) montados com primitivas e campos sintéticos.

<small>`pinneapple_twin3d/gallery.py` · 438 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `get` | função | `(key) -> Scene` | Retorna a cena da galeria pelo nome da função registrada; levanta KeyError para nome desconhecido. |
| `runner_channel` | função | `() -> Scene` | Canal de corrida de alto-forno: calha em U revestida de refratário do furo de gusa ao skimmer. |
| `mixing_tank` | função | `() -> Scene` | Tanque agitado com fundo abaulado, chicanas e impulsor de dois estágios; mistura de traçador no tempo. |
| `oil_pipeline` | função | `() -> Scene` | Oleoduto com duas curvas e uma depressão; perda de parede por corrosão/erosão ao longo dos anos. |
| `airliner` | função | `() -> Scene` | Avião de corredor único (fuselagem, asa enflechada, cauda, dois motores); temperatura de pele e carga na raiz da asa num voo. |
| `rocket` | função | `() -> Scene` | Lançador com nariz ogival, tanques, bocal e aletas; aquecimento aerodinâmico e ablação do bocal na subida. |
| `drone` | função | `() -> Scene` | Quadricóptero; aquecimento de motores e bateria durante um voo. |
| `lunar_rover` | função | `() -> Scene` | Rover lunar de seis rodas num ciclo dia-noite lunar: temperatura do chassi e desgaste das rodas. |
| `wind_turbine` | função | `() -> Scene` | Turbina de três pás; erosão de bordo de ataque e fadiga ao longo dos anos. |
| `satellite` | função | `() -> Scene` | Satélite pequeno em órbita baixa; ciclos térmicos entre sol e eclipse. |
| `export_all` | função | `(folder) -> str` | Exporta todas as peças da galeria em subpastas e grava um `index.html` com links para elas. |

#### `pinneapple_twin3d.library`

[shim] Alias de `pinneapple_design.geometry_library`: expõe a biblioteca de geometrias do lado do gêmeo. [sem teste]

<small>`pinneapple_twin3d/library.py` · 8 linhas · **sem teste direto**</small>

#### `pinneapple_twin3d.openfoam`

Lê um caso OpenFOAM (pontos, faces, boundary, diretórios de tempo) e monta uma cena com os patches de parede como superfícies e valores por face ao longo do tempo, como mapas de erosão.

<small>`pinneapple_twin3d/openfoam.py` · 215 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `read_points` | função | `(case) -> np.ndarray` | Lê o arquivo `constant/polyMesh/points` do caso como array (N, 3). |
| `read_faces` | função | `(case) -> List[np.ndarray]` | Lê `polyMesh/faces` como lista de arrays de índices de vértices. |
| `read_boundary` | função | `(case) -> Dict[str, Dict[str, object]]` | Lê `polyMesh/boundary` e devolve tipo, face inicial e número de faces de cada patch. |
| `time_directories` | função | `(case) -> List[Tuple[float, str]]` | Lista os diretórios de tempo numéricos do caso, ordenados, como (tempo, nome). |
| `read_boundary_field` | função | `(path, patch, n_faces) -> Optional[np.ndarray]` | Lê os valores por face de um patch num arquivo volScalarField/volVectorField (vetores viram módulo). |
| `scene_from_case` | função | `(case, *, title=None, patches=None, face_values=None, units=None, boundary_fields=(), tim…)` | Monta uma cena a partir de um caso OpenFOAM, com patches de parede como partes e campos vindos de valores por face ou de campos de contorno. |

#### `pinneapple_twin3d.primitives`

Primitivas de malha para montar peças sem CAD: revolução, varredura, loft, perfil de aerofólio, asa, caixa, cilindro, transformações, junção e checagem de qualidade.

<small>`pinneapple_twin3d/primitives.py` · 183 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `revolve` | função | `(profile, n_theta=48, *, cap=False) -> Mesh` | Superfície de revolução de um perfil (r, y) em torno do eixo y, com tampas opcionais. |
| `sweep` | função | `(section, path, *, closed=True, up=(0, 1, 0)) -> Mesh` | Extruda uma seção 2D ao longo de um caminho 3D, aberta ou fechada. |
| `airfoil` | função | `(chord=1.0, thickness=0.12, n=24) -> np.ndarray` | Contorno fechado de aerofólio simétrico NACA 00xx. |
| `loft` | função | `(rings) -> Mesh` | Malha de pele através de anéis com o mesmo número de pontos. |
| `wing` | função | `(span, root_chord, tip_chord, *, sweep_deg=0.0, dihedral_deg=0.0, twist_deg=0.0, thicknes…)` | Meia asa ou pá com enflechamento, diedro, torção e afilamento. |
| `box` | função | `(size, center=(0, 0, 0)) -> Mesh` | Caixa com tamanho e centro dados. |
| `cylinder` | função | `(p0, p1, radius, n_theta=32, *, cap=True) -> Mesh` | Cilindro (ou tronco de cone) entre dois pontos. |
| `rotation` | função | `(axis, angle_deg) -> np.ndarray` | Matriz de rotação 3x3 em torno de um eixo, ângulo em graus. |
| `transform` | função | `(mesh, *, R=None, t=(0, 0, 0), scale=1.0) -> Mesh` | Aplica rotação, translação e escala a uma malha. |
| `mirror_z` | função | `(mesh) -> Mesh` | Espelha a malha no plano z=0, corrigindo a orientação das faces. |
| `merge` | função | `(meshes) -> Mesh` | Junta várias malhas numa só, reindexando as faces. |
| `mesh_quality` | função | `(mesh) -> dict` | Números de sanidade da malha (coordenadas finitas, índices válidos, fração de triângulos degenerados, limites) usados nos testes. |

#### `pinneapple_twin3d.scan`

Geometria de gêmeo a partir de escaneamentos: limpa malhas de scan (vértices duplicados, fragmentos, buracos, alinhamento do piso, metros) e converte cenas MuJoCo MJCF em partes; exige `trimesh`.

<small>`pinneapple_twin3d/scan.py` · 199 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ScanReport` | classe | `(vertices_in, faces_in, vertices_out, faces_out, components_dropped, holes_filled, watert…)` | Relatório da limpeza do scan: contagens antes/depois, fragmentos removidos, buracos preenchidos, estanqueidade e eixo vertical. |
| `clean_scan` | função | `(path_or_mesh, *, unit='m', min_component_fraction=0.01, fill_holes=True, target_faces=No…)` | Limpa uma malha de scan e devolve (malha trimesh em metros, `ScanReport`). |
| `add_scan` | função | `(scene, name, path_or_mesh, *, group='scan', **kw) -> ScanReport` | Limpa um scan e o adiciona como parte de uma cena, devolvendo o relatório. |
| `scene_from_mjcf` | função | `(path, *, title=None, include_primitives=True) -> Scene` | Cria uma parte por geom de uma cena MuJoCo MJCF, em coordenadas do mundo. |

#### `pinneapple_twin3d.scene`

Núcleo do pacote: modelo da cena (partes, campos no tempo, movimentos, visibilidade, física, sensores) e exportação para `scene.json` + `geometry.glb` + `fields.bin` + visualizador, com USD opcional.

<small>`pinneapple_twin3d/scene.py` · 360 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Part` | classe | `(name, vertices, faces, group, color, fields, motion, visibility, physics)` | Uma parte da cena: vértices, faces, grupo, cor, campos, movimento, visibilidade e propriedades físicas. |
| `PhysicsMaterial` | classe | `(name, static_friction, dynamic_friction, restitution, density)` | Material de contato (atrito estático e dinâmico, restituição, densidade) no estilo `UsdPhysics`. |
| `PhysicsProps` | classe | `(rigid_body, collider, kinematic, mass, density, center_of_mass, approximation, material,…)` | Como a parte entra numa simulação física: corpo rígido, colisor, cinemático, massa, material e velocidades. |
| `Sensor` | classe | `(id, position, label, unit, quantity, series, envelope)` | Sensor com id, posição, unidade, grandeza, série temporal e envelope de alarme. |
| `Scene` | classe | `(title, *, length_unit='m', times=None, time_unit='s', source='')` | Cena de gêmeo digital: adiciona partes, campos, movimentos, física e sensores e exporta para pasta (com visualizador, USD opcional, render Blender). |

#### `pinneapple_twin3d.serve`

Serve por HTTP a pasta exportada (o visualizador precisa de HTTP, não `file://`).

<small>`pinneapple_twin3d/serve.py` · 26 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `serve` | função | `(folder, port=8765, *, host='127.0.0.1', block=True)` | Serve a pasta exportada em `http://host:port/`; com `block=False` devolve o servidor rodando. |

#### `pinneapple_twin3d.usd`

Exporta a cena como OpenUSD texto (`.usda`) com campos por vértice como time samples, sensores como esferas e propriedades `UsdPhysics`, para Omniverse/Isaac Sim/usdview.

<small>`pinneapple_twin3d/usd.py` · 209 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `export_usd` | função | `(scene, path) -> str` | Grava a cena em `.usda` e devolve o caminho. |

#### `pinneapple_twin3d.wear`

Gerador de gêmeos de desgaste (refratário, revestimento, tubo) a partir de uma especificação: spec -> dataset -> previsão -> otimização -> cena + relatório.

<small>`pinneapple_twin3d/wear/__init__.py` · 21 linhas · testes: 30</small>

#### `pinneapple_twin3d.wear.build`

Converte um `WearDataset` em cena 3D, contrato do hub (`wear_dataset.json`) e relatório de previsão/otimização (`report.json`). [sem teste direto]

<small>`pinneapple_twin3d/wear/build.py` · 95 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `remaining_life_fields` | função | `(ds, cap, **kw) -> dict` | Vida restante por célula em cada leitura, limitada a `cap`. |
| `scene_from_wear` | função | `(ds, *, life_cap=None, **fit_kw) -> Scene` | Monta a cena 3D do dataset de desgaste, com campos de desgaste, fração e vida restante por célula. |
| `report` | função | `(ds, cfg=None, **fit_kw) -> dict` | Relatório que mantém separados o observado, o previsto e o recomendado. |
| `export_wear_twin` | função | `(ds, folder, *, cfg=None, usd=False, **fit_kw) -> str` | Grava cena 3D, `wear_dataset.json` e `report.json` numa pasta. |

#### `pinneapple_twin3d.wear.calibrate`

Calibra as hipóteses do otimizador (horas de parada, custos, fração de recuperação) a partir de registros de paradas e do próprio dataset, informando a fonte e o erro padrão de cada parâmetro.

<small>`pinneapple_twin3d/wear/calibrate.py` · 138 linhas · testes: 30</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `StopRecord` | classe | `(zones, duration_h, cost, x)` | Registro de uma parada de manutenção: zonas atendidas, duração, custo e posição x. |
| `ParamReport` | classe | `(name, value, source, n, se, note)` | Relatório de um parâmetro calibrado: valor, fonte, tamanho de amostra, erro padrão e nota. |
| `Calibration` | classe | `(cfg, report)` | Resultado da calibração: `OptimizerConfig` ajustado e relatório por parâmetro. |
| `estimate_recovery` | função | `(ds, repair_drop=0.08, min_cells=20)` | Mediana da fração da espessura útil recuperada por célula reparada, e quantas células foram usadas. |
| `calibrate` | função | `(stops=(), ds=None, base=None) -> Calibration` | Ajusta tempos e custos de parada por mínimos quadrados e a recuperação a partir do dataset, mantendo o padrão quando o ajuste não é confiável. |

#### `pinneapple_twin3d.wear.forecast`

Previsão de vida restante por célula por regressão linear da espessura residual numa janela recente, com intervalo t de Student de 90 % e índice de confiança; inclui histórico e backtest.

<small>`pinneapple_twin3d/wear/forecast.py` · 185 linhas · testes: 30</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CellFit` | classe | `(remaining, remaining_lo, remaining_hi, slope, r2, n, confidence)` | Ajuste de uma célula: vida restante com limites, inclinação, R2, número de pontos e confiança. |
| `ZoneForecast` | classe | `(zone, worst_fraction, n_over, min_remaining, rate, usable, critical_cell, confidence, r2…)` | Previsão de uma zona: pior fração de desgaste, células acima do limite, menor vida restante, taxa, célula crítica e confiança. |
| `t90` | função | `(df)` | Quantil t de Student para intervalo bilateral de 90 % por graus de liberdade, interpolado de uma tabela. |
| `confidence_index` | função | `(n, r2)` | Índice de confiança 0-3 (n/a, baixa, média, alta) a partir do número de pontos e do R2. |
| `fit_zone` | função | `(ds, zone, s, *, win_x=DEFAULT_WIN_X, min_pts=8, repair_drop=0.08) -> CellFit` | Ajuste linear por célula de uma zona na leitura `s`, parando no último reparo detectado. |
| `forecast` | função | `(ds, s=None, **kw) -> Dict[str, ZoneForecast]` | Previsão por zona na leitura indicada (por padrão a última). |
| `forecast_history` | função | `(ds, **kw) -> List[dict]` | Menor vida restante prevista recalculada em cada leitura, mostrando como a previsão evoluiu. |
| `backtest` | função | `(ds, **kw) -> dict` | Compara previsões passadas de quando a primeira célula atinge o limite com o que de fato aconteceu. |
| `status` | função | `(fraction) -> str` | Classifica a fração de desgaste em `normal`, `attention`, `alert` ou `critical`. |

#### `pinneapple_twin3d.wear.geometry`

Malhas de células para superfícies de revolução: amostragem do perfil, uma quad por célula, mapeamento célula->vértice e centros das células.

<small>`pinneapple_twin3d/wear/geometry.py` · 62 linhas · testes: 30</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `sample_profile` | função | `(profile, n_rows) -> np.ndarray` | Pontos igualmente espaçados em comprimento de arco ao longo do perfil (r, y). |
| `cell_mesh` | função | `(zone) -> Tuple[np.ndarray, np.ndarray]` | Vértices e faces de uma zona, uma quad (dois triângulos) por célula. |
| `cell_to_vertex` | função | `(values) -> np.ndarray` | Converte valores por célula em valores por vértice na ordem de `cell_mesh`. |
| `cell_centers` | função | `(zone) -> np.ndarray` | Posição no mundo do centro de cada célula, usada para posicionar sensores e pontos quentes. |

#### `pinneapple_twin3d.wear.model`

Modelo de dados do gêmeo de desgaste: `ZoneSpec` (perfil de revolução cortado em células), `VesselSpec` e `WearDataset` (desgaste por zona no tempo, com contrato JSON e leitura de CSV). [sem teste direto]

<small>`pinneapple_twin3d/wear/model.py` · 165 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ZoneSpec` | classe | `(name, label, profile, n_rows, n_sectors, e0, emin, center, theta0, group)` | Zona de revolução: perfil, linhas e setores, espessura nova e mínima, centro e grupo; `usable` dá a espessura útil. |
| `VesselSpec` | classe | `(key, title, zones, x_name, wear_unit, notes)` | Especificação do vaso: chave, título, zonas, nome da variável x e unidade de desgaste. |
| `WearDataset` | classe | `(spec, x, wear, origin, labels, note)` | Desgaste em mm por zona com forma (T, linhas, setores), NaN sem leitura; converte de/para o contrato JSON do hub e lê CSV. |

#### `pinneapple_twin3d.wear.optimize`

Agendamento de manutenção (gunning/reparo) sobre a previsão: simula políticas reativa, calendário fixo e preditiva agrupada e recomenda a mais barata sem violação; apoio à decisão apenas.

<small>`pinneapple_twin3d/wear/optimize.py` · 166 linhas · testes: 30</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OptimizerConfig` | classe | `(horizon, heats_per_day, recovery_frac, setup_hours, hours_per_zone, cost_per_hour, cost_…)` | Parâmetros do otimizador: horizonte, recuperação, horas e custos de parada, custo de violação, margem de segurança e janelas. |
| `ZoneState` | classe | `(zone, deadline, life_full, ext)` | Estado de uma zona para o otimizador: prazo até o limite, vida completa e extensão por intervenção. |
| `Policy` | classe | `(id, name, schedule, stops, zones_served, downtime_hours, breach, cost, window)` | Uma política simulada: agenda de paradas, zonas atendidas, horas paradas, violação e custo. |
| `OptimizationResult` | classe | `(cfg, zones, policies, recommended, baseline, notes)` | Resultado: zonas, políticas simuladas, política recomendada, linha de base e notas. |
| `prepare` | função | `(zones, cfg) -> List[ZoneState]` | Converte previsões de zona em `ZoneState` usando a configuração. |
| `simulate` | função | `(zs, stops, cfg) -> dict` | Simula uma agenda de paradas e devolve custo, horas paradas e violação. |
| `fixed_policy` | função | `(zs, cfg) -> List[dict]` | Gera a agenda de paradas em calendário fixo. |
| `predictive_policy` | função | `(zs, cfg, window) -> List[dict]` | Gera paradas preditivas agrupando zonas cujos prazos caem dentro de uma janela. |
| `optimize` | função | `(forecasts, cfg=None) -> OptimizationResult` | Simula as políticas reativa, fixa e preditiva e recomenda a mais barata sem violação (ou com menor violação). |
| `explain` | função | `(res) -> List[dict]` | Explica em texto cada parada da política recomendada (zonas, prazo e margem). |

#### `pinneapple_twin3d.wear.presets`

Especificações indicativas dos casos de referência (panela de aço, panela de gusa, conversor BOF, desgaseificador RH) e de um reator genérico de redução-fusão.

<small>`pinneapple_twin3d/wear/presets.py` · 176 linhas · testes: 30</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `get` | função | `(key)` | Retorna (spec, pontos quentes) do preset pelo nome da função; levanta KeyError se não existir. |
| `fill_level` | função | `(rin, y0, y1, rho, mass, n=300) -> float` | Nível do banho para uma massa dada num vaso de raio interno variável, por integração em troncos. |
| `steel_ladle` | função | `()` | Preset indicativo de panela de aço. |
| `pig_iron_ladle` | função | `()` | Preset indicativo de panela de gusa. |
| `bof_converter` | função | `()` | Preset indicativo de conversor BOF. |
| `rh_degasser` | função | `()` | Preset indicativo de desgaseificador RH. |
| `smelting_reduction_reactor` | função | `()` | Preset genérico de reator de redução-fusão: cuba, zona de ventaneiras, cadinho e soleira. |

#### `pinneapple_twin3d.wear.synthetic`

Campanhas de desgaste sintéticas (pontos quentes + lei de crescimento + reparos + ruído), sempre rotuladas SYNTHETIC; não são modelo físico. [sem teste direto]

<small>`pinneapple_twin3d/wear/synthetic.py` · 80 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HotSpot` | classe | `(zone, s, theta_deg, ds, dtheta_deg, amp)` | Ponto quente de desgaste numa zona: posição relativa, ângulo, larguras e amplitude. |
| `base_pattern` | função | `(spec, hotspots, base) -> Dict[str, np.ndarray]` | Padrão espacial de desgaste por zona a partir dos pontos quentes e de um nível base. |
| `synthetic_campaign` | função | `(spec, hotspots, *, n_readings=41, x_max=120.0, growth_power=1.0, base=0.2, noise=0.01, s…)` | Gera um `WearDataset` sintético com crescimento, reparos e ruído de medição. |

### Exemplos de uso reais do repositório

**Gêmeo de desgaste para uma peça nova** — `examples/use_cases/refractory_wear_twin/custom_part.py`

```python
def build(out: str) -> str:
    spec = tank_spec()
    hot = (HotSpot("wall", 0.7, None, 0.1, amp=0.6),          # a band all around at 70 % of the height
           HotSpot("floor", 0.4, 90, 0.3, 35, amp=0.7))      # an impact pad at 90 deg, 40 % of the radius
    ds = synthetic_campaign(spec, hot, n_readings=25, x_max=80)
    print({k: round(v.min_remaining, 1) for k, v in forecast(ds).items()}, "->", optimize(forecast(ds)).recommended.name)
    return export_wear_twin(ds, out)
```

**Exportar e servir uma cena** — `tests/test_twin3d.py`

```python
sc.export(str(tmp_path))
httpd = serve(str(tmp_path), port=0, block=False)
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 15 de 19.
- Funções de teste em arquivos que importam algum módulo do pacote: 74.

### Limitações conhecidas

- A previsão de desgaste é extrapolação linear: não antecipa mudança de regime; a confiança só avalia a qualidade do ajuste e o intervalo de 90 % é de ajuste, não de predição.
- O otimizador de manutenção é apoio à decisão apenas; custos de violação e margem de segurança são entradas de negócio, nunca inferidas.
- Os presets usam geometria indicativa, não desenhos de uma planta específica; as campanhas sintéticas e o campo da demo são ilustrativos, não modelos físicos.
- `scan` exige `trimesh`; `render_blender` exige `bpy`.
- O exemplo no docstring de `pinneapple_twin3d.wear` usa `presets.get("bof")`, mas a chave registrada é `bof_converter` (o nome da função), então aquele trecho levantaria KeyError.
- Sem teste direto: `library`, `wear.build`, `wear.model` e `wear.synthetic` (são exercitados indiretamente pelos testes de `wear`).
- O visualizador precisa ser servido por HTTP; abrir o `index.html` por `file://` não carrega os dados.
