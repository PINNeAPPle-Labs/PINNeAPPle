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
| `fitmesh` |  | 12 | 38 | 21 |
| `geometry` | Base geométrica: SDFs, CSG, domínios físicos, malhas, E/S de CAD, operações de malha, amostradores e CAD paramétrico de veículos. | 56 | 255 | 195 |
| `geometry_library` | Biblioteca de geometrias 3D paramétricas e realistas (aeronaves, veículos, navios, turbinas, prédios, anatomia) com registro, construtores, checagens de malha, exportadores (STL, OBJ, glTF, USD, STEP), renderizador por software e arquivos pré-gerados no Hugging Face Hub. | 15 | 91 | 42 |
| `intent` | Projeto a partir da intenção: uma meta em linguagem natural vira especificação com unidades, plano de física, otimização multifidelidade, verificação com solver completo, robustez e um pacote de projeto com relatório, geometria e arquivos de checagem. | 19 | 84 | 24 |
| `qualitative` |  | 5 | 25 | 13 |
| `surrogate_inverse` |  | 8 | 104 | 94 |
| `thermal` | Projeto térmico: motor e dimensionador de dissipadores, surrogates e o estudo de resfriamento de placa. | 19 | 75 | 46 |
| `topology` |  | 6 | 17 | 9 |

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

Fit-for-purpose geometry and mesh: from a problem description to checked CAD, planned meshes and one report.

<small>`pinneapple_design/fitmesh/__init__.py` · 26 linhas · testes: 21</small>

#### `pinneapple_design.fitmesh.api`

HTTP API over :class:`FitMeshSession`: one session per study, one call per step, the files and the report.

<small>`pinneapple_design/fitmesh/api.py` · 146 linhas · testes: 12</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `create_app` | função | `(root='fitmesh_runs')` |  |

#### `pinneapple_design.fitmesh.cad`

Build the CAD variants, export STEP and STL, and check each one before it is meshed.

<small>`pinneapple_design/fitmesh/cad.py` · 115 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CadVariant` | classe | `(index, params, step, stl, checks, comparison, volume, area, bbox)` |  |
| `build_variant` | função | `(tpl, params, out_dir, index, accuracy='design') -> CadVariant` |  |
| `compare_with_references` | função | `(tpl, params) -> list[dict[str, Any]]` | Relative difference of each shared dimension with each published reference object. |

#### `pinneapple_design.fitmesh.catalog`

Parametric geometry templates with the references that make them checkable.

<small>`pinneapple_design/fitmesh/catalog.py` · 460 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Param` | classe | `(name, value, unit, typical, limits, role, sensitivity)` |  |
| `Reference` | classe | `(citation, what, params)` |  |
| `Template` | classe | `(key, title, physics, keywords, description, params, references, quantities, build, volum…)` |  |
| `naca4_points` | função | `(m, p, t, chord, n=80) -> np.ndarray` | Closed NACA 4-digit section (x, z), trailing edge closed (a4 = -0.1036), cosine spacing. |
| `polygon_area` | função | `(pts) -> float` |  |

#### `pinneapple_design.fitmesh.context`

Read a problem description into the facts the geometry and mesh choices depend on, and ask for what is missing.

<small>`pinneapple_design/fitmesh/context.py` · 208 linhas · testes: 12</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Fact` | classe | `(value, source)` |  |
| `ProblemContext` | classe | `(text, facts)` |  |
| `parse` | função | `(text, llm=None) -> ProblemContext` | Facts from a free-text problem description (English or Portuguese). |
| `wall_unit_height` | função | `(y_plus, speed, length, nu, re, internal=False) -> dict[str, float]` | First-cell height for a target y+ from flat-plate skin friction at the reference length. |

#### `pinneapple_design.fitmesh.layers`

Prism (inflation) layer sizing: a geometric stack of wall layers.

<small>`pinneapple_design/fitmesh/layers.py` · 81 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `layer_total` | função | `(first_height, expansion, n_layers) -> float` | Total thickness ``h1 (r^n - 1) / (r - 1)`` of ``n`` layers (``n h1`` when ``r = 1``). |
| `size_layers` | função | `(first_height, expansion=None, n_layers=None, total_thickness=None) -> dict[str, float]` | Solve the geometric layer stack for the one missing quantity among ``expansion``, ``n_layers`` and ``total_thickness`` (exactly one must be ``None``). |

#### `pinneapple_design.fitmesh.llm`

Optional language-model help for every step, with a local model if you like (Ollama via ``pinneapple_llm``).

<small>`pinneapple_design/fitmesh/llm.py` · 251 linhas · testes: 12</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LLMAssist` | classe | `(provider, model, api_key, conversation_store, call, provider_kwargs)` | Which model to ask. ``provider``: "ollama" (local), "anthropic" or "openai"; ``call`` overrides everything (a function ``(prompt, system) -> str`` returning JSON), which is how tests and custom backends plug in. |
| `read_problem` | função | `(llm, text) -> dict[str, Any]` |  |
| `rank_geometries` | função | `(llm, ctx) -> dict[str, Any]` |  |
| `propose_parameters` | função | `(llm, tpl, ctx, base) -> list[dict[str, Any]]` |  |
| `free_geometry` | função | `(llm, description) -> dict[str, Any]` | A CadQuery script from the model, checked statically; run it with :func:`run_free_geometry`. |
| `check_code` | função | `(code) -> list[str]` | Static checks before anything runs: syntax, imports on the allow-list, no dunder or I/O names, build(). |
| `run_free_geometry` | função | `(code, params, out_dir, tol, timeout=120) -> dict[str, Any]` | Run model-written CadQuery in a separate isolated interpreter (``python -I``), with a timeout. |
| `review_mesh_plan` | função | `(llm, plan, ctx) -> MeshPlan` |  |

#### `pinneapple_design.fitmesh.mesh_gen`

Generate the planned meshes and measure their quality.

<small>`pinneapple_design/fitmesh/mesh_gen.py` · 407 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MeshResult` | classe | `(variant, level, tool, path, files, cells, seconds, quality, layers, error)` |  |
| `generate` | função | `(tpl, var, plan, out_dir, level=1.0, procs=1, max_cells=3000000.0, log=print) -> MeshResu…)` | Mesh one CAD variant at ``level`` x the planned sizes (1.0 = the plan; 1.5, 2.25 = the coarser family). |
| `quality` | função | `(path, target) -> dict[str, Any]` |  |

#### `pinneapple_design.fitmesh.mesh_plan`

Recommend a mesh that is fit for the purpose: fine enough for the quantities that matter, no finer.

<small>`pinneapple_design/fitmesh/mesh_plan.py` · 264 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MeshPlan` | classe | `(kind, tool, elements, purpose, sizes, layers, domain, estimated_cells, estimated_cost, f…)` |  |
| `mesh_kind` | função | `(tpl, ctx) -> str` |  |
| `plan_mesh` | função | `(tpl, params, ctx, budget_cells=5000000.0) -> MeshPlan` |  |

#### `pinneapple_design.fitmesh.report`

One final trust report for the whole study, plus the grid-convergence calculation it relies on.

<small>`pinneapple_design/fitmesh/report.py` · 226 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `gci` | função | `(values, sizes, safety=None) -> dict[str, Any]` | Grid-convergence index from three meshes (Roache 1994; procedure of Celik et al. 2008, J. Fluids Eng. 130). |
| `build_report` | função | `(session) -> dict[str, Any]` | Collect every stage of a :class:`FitMeshSession` into one report with a verdict per stage. |
| `render_html` | função | `(rep) -> str` |  |
| `write_report` | função | `(rep, stem) -> dict[str, str]` |  |

#### `pinneapple_design.fitmesh.session`

The interactive study: describe the problem, answer the questions, choose, and get one final report.

<small>`pinneapple_design/fitmesh/session.py` · 295 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FitMeshSession` | classe | `(text, out_dir, llm=None, log=print)` |  |

#### `pinneapple_design.fitmesh.suggest`

Rank geometry templates for a problem and propose the parameters to vary, their ranges and the samples.

<small>`pinneapple_design/fitmesh/suggest.py` · 94 linhas · testes: 12</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `suggest_geometries` | função | `(ctx, top=3) -> list[dict[str, Any]]` | Templates ranked by physics match and keyword hits, each with why it was picked and its references. |
| `scale_to_context` | função | `(tpl, ctx) -> dict[str, float]` | Reference values, scaled so the reference length matches the size given in the problem (if any). |
| `suggest_parameters` | função | `(tpl, ctx, base=None) -> list[dict[str, Any]]` | Every parameter with a vary / keep recommendation, a proposed range and the reason. |
| `sample` | função | `(tpl, base, ranges, n, seed=0, include_base=True) -> list[dict[str, float]]` | Latin hypercube over the selected ranges (scipy.stats.qmc); integer parameters are rounded. |

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

Qualitative physics preview of geometries and geometry changes, before the quantitative analysis.

<small>`pinneapple_design/qualitative/__init__.py` · 36 linhas · testes: 13</small>

#### `pinneapple_design.qualitative.descriptors`

Geometric descriptors of closed triangulated bodies and assemblies of named parts.

<small>`pinneapple_design/qualitative/descriptors.py` · 259 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Descriptors` | classe | `(volume, wetted_area, centroid, extents, length, frontal_area, equivalent_diameter, slend…)` | Geometric descriptors of one body (SI units). ``direction`` is the flow / load reference direction. |
| `Assembly` | classe | `(parts)` | Named parts of one geometry. Parts are treated as touching, not overlapping (areas add). |
| `as_mesh` | função | `(geom) -> tuple[np.ndarray, np.ndarray]` | ``(vertices, faces)`` from a tuple, a trimesh-like object (``.vertices``/``.faces``) or an ``Assembly``. |
| `extrude` | função | `(polygon, depth=1.0, axis='y') -> tuple[np.ndarray, np.ndarray]` | Closed prism from a counter-clockwise 2-D polygon (n, 2) in the (x, z) plane, extruded along ``axis`` by ``depth`` (centred), so 2-D sections can be compared with the same models. |
| `face_geometry` | função | `(V, F)` | Per-face area, unit normal and centroid. |
| `describe` | função | `(geom, direction=(1.0, 0.0, 0.0)) -> Descriptors` | Descriptors of a closed, outward-oriented triangulated body. The projected area is ``0.5 * sum \|n . d\| A`` (exact for convex bodies, an upper bound otherwise). |
| `section_profile` | função | `(geom, axis=0, n_slices=40, n_grid=48)` | Section properties along ``axis`` from a voxelization by ray parity: slice positions ``x`` (centres), area ``A(x)``, centroid of each section in the two other coordinates, second moments ``I(x)`` about the section's centroidal axes (``I_a``: about the first remaining axis, i.e. bending in the … |

#### `pinneapple_design.qualitative.models`

Cheap physical models that turn a geometry into first estimates, the mechanisms behind them and a surface map.

<small>`pinneapple_design/qualitative/models.py` · 501 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Quantity` | classe | `(name, unit, better, pt, en)` |  |
| `Evaluation` | classe | `(quantities, terms, by_part, face_field, field_label, notes, caveats, mesh, extra)` | Result of one model on one geometry. |
| `QualitativeModel` | classe | `(**conditions)` |  |
| `ExternalFlow` | classe | `()` |  |
| `ConvectiveCooling` | classe | `()` |  |
| `Cantilever` | classe | `()` |  |
| `ScalingModel` | classe | `(formulas, better=None, direction=(1.0, 0.0, 0.0), **conditions)` | User formulas of the geometric descriptors, e.g. ``ScalingModel({"conduction_resistance": lambda d, c: d.length / (c["k"] * d.frontal_area)}, better={"conduction_resistance": "min"}, k=200)``. The mechanisms are the descriptors: each one's share of a change comes from swapping it alone (in log … |
| `sheltered_faces` | função | `(V, F, normals, centroids, direction, grid=96, tol=0.05) -> np.ndarray` | Windward faces hidden behind other surfaces seen from upstream (in the wake of something in front of them): the windward triangles are rasterized on the plane normal to the flow, keeping the most upstream depth per cell; a windward face whose cell has a surface more than ``tol`` x body length … |

#### `pinneapple_design.qualitative.preview`

Qualitative preview: how each geometry, or each change to a part, should affect the result, before any simulation; then the quantitative check.

<small>`pinneapple_design/qualitative/preview.py` · 523 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Objective` | classe | `(quantity, sense, model)` |  |
| `VariantPreview` | classe | `(name, evaluation, change, mechanisms, parts, confidence, reasons, good)` |  |
| `QualitativePreview` | classe | `(model, objective, baseline, variants, geometries, lang)` |  |
| `QuantitativeCheck` | classe | `(preview, values, seconds)` |  |
| `PartSensitivity` | classe | `(objective, model, base_value, rows, lang)` |  |
| `parse_objective` | função | `(objective, model=None) -> Objective` | ``"minimizar arrasto"``, ``"maximize heat rejection"``, ``("tip_deflection", "min")`` or an ``Objective``. Downforce means lift with sense "min". |
| `preview` | função | `(geometries, objective, conditions=None, baseline=None, model=None, lang='pt') -> Qualita…)` | Qualitative comparison of ``geometries`` (name -> mesh tuple, trimesh-like object or ``Assembly``) for an ``objective`` such as "minimizar arrasto", "maximize heat rejection", "aumentar rigidez por massa" or ``("tip_deflection", "min")``. ``conditions`` are the model's operating conditions (speed, … |
| `part_sensitivity` | função | `(assembly, objective, conditions=None, step=0.2, model=None, lang='pt', axes='xyz', about…)` | For each part: grow it by ``step`` along each axis (about its minimum corner by default, i.e. away from where it is usually attached) and remove it; the change of the objective ranks where to act first. |

#### `pinneapple_design.qualitative.quantitative`

Quantitative counterparts of the qualitative models, for ``QualitativePreview.quantify``.

<small>`pinneapple_design/qualitative/quantitative.py` · 132 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `hex8_incompatible_stiffness` | função | `(size, E, nu) -> np.ndarray` | 24 x 24 stiffness of a rectangular brick ``size = (hx, hy, hz)`` with Wilson's incompatible modes condensed. |
| `voxel_fem_cantilever` | função | `(geom, E=70000000000.0, nu=0.3, load=1000.0, axis=None, load_axis=2, clamp='min', n_slice…)` | Tip deflection, stiffness, maximum von Mises stress and mass of a body clamped at one end of ``axis`` (default the longest) with ``load`` (N) along ``load_axis`` spread over the other end face. ``cells_across``: voxels along the larger section dimension (the other is scaled to keep the voxels near … |

#### `pinneapple_design.surrogate_inverse`

Surrogate-driven inverse design on closed-form physics, with the answer re-checked against the physics.

<small>`pinneapple_design/surrogate_inverse/__init__.py` · 38 linhas · testes: 91</small>

#### `pinneapple_design.surrogate_inverse.brake_rotor`

Vented brake rotor: the radial vanes between the two friction rings, as a fin array, under a sustained load.

<small>`pinneapple_design/surrogate_inverse/brake_rotor.py` · 269 linhas · testes: 94</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RotorDesign` | classe | `(vane_count, vane_gap_m, vane_thickness_m)` |  |
| `OptimizationResult` | classe | `(design, surrogate_predicted_temp_c)` |  |
| `profile_params` | função | `(design, power_w, h_conv) -> dict[str, float]` | Formula quantities, plus ``theta_vane_base_c`` (alias of the fin-root rise) and ``lc_radial_m`` (corrected length along the vane's radial extent, for the radial colour profile). |
| `rotor_temperature_c` | função | `(design, power_w, h_conv) -> float` | Ground truth: friction-ring temperature under a sustained heat input. |
| `thermal_resistance_k_w` | função | `(design, h_conv) -> float` |  |
| `vane_temperature_at_radial_position_c` | função | `(design, power_w, h_conv, r_fraction) -> float` | The fin profile (same ``m`` and root rise) along the vane's radial extent: hottest at the inner edge (``r_fraction = 0``), coolest at the outer edge. Used to colour the vanes; the scalar model integrates over the gap direction, this profile is a display of the same fin parameters along the radius. |
| `thermal_field` | função | `(design, power_w, h_conv) -> dict[str, float]` |  |
| `constraint_violations` | função | `(design) -> dict[str, float]` |  |
| `design_to_dict` | função | `(design) -> dict` |  |
| `generate_dataset` | função | `(n_samples, seed=0) -> tuple[torch.Tensor, torch.Tensor]` |  |
| `train_surrogate` | função | `(n_samples=6000, epochs=1500, seed=0) -> SurrogateFit` |  |
| `predict_temperature_c` | função | `(surrogate, design, power_w, h_conv) -> float` |  |
| `optimize_design` | função | `(surrogate, power_w, h_conv, steps=400, lr=0.05, penalty_weight=100.0) -> OptimizationRes…)` | Minimise the surrogate's rotor temperature over (N, gap, t): same loss form (``log R`` plus the relative spacing shortfall), sigmoid-bounded descent and integer rounding as :func:`.heatsink_fins.optimize_design`. |
| `verify_design` | função | `(design, power_w, h_conv, surrogate_predicted_temp_c) -> VerificationCard` |  |
| `monte_carlo_uncertainty` | função | `(design, power_w, h_conv, safety_threshold_c, n_samples=200, seed=None, thickness_toleran…)` | Vane thickness and gap drawn uniformly within their tolerance bands (vane count is a part count and stays fixed), each sample through the formula (never the surrogate). Verdict on the 95th percentile ``p95``: FRAGILE if above the threshold, MARGINAL within 5 % below it, ROBUST otherwise. … |
| `mesh` | função | `(design) -> meshes.Mesh` | Two friction rings and the radial vanes, in millimetres. |
| `export_stl_bytes` | função | `(design) -> bytes` |  |

#### `pinneapple_design.surrogate_inverse.core`

The machinery shared by every case: a small MLP surrogate, gradient descent on the design through the frozen surrogate, and a verification card that re-checks the answer against the ground truth.

<small>`pinneapple_design/surrogate_inverse/core.py` · 267 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MLPSurrogate` | classe | `(bounds, hidden=48, n_layers=3, log_output=False)` | ``tanh`` MLP with its inputs mapped from their bounds to [-1, 1] and its output standardised. |
| `SurrogateFit` | classe | `(model, val_mean_rel_err, val_p95_rel_err, val_max_rel_err, n_train, n_val, epochs, secon…)` | A trained surrogate with its held-out errors (relative, on rows it never trained on). |
| `CheckResult` | classe | `(name, verdict, detail, value, limit)` |  |
| `VerificationCard` | classe | `(checks, values, covered, not_covered)` | The final design re-checked against the ground truth (never the surrogate grading itself). |
| `fit_surrogate` | função | `(X, y, bounds, *, hidden=48, n_layers=3, epochs=1500, lr=0.003, seed=0, log_output=False,…)` | Full-batch Adam with a cosine learning-rate decay on ``(X, y)``; the last ``val_fraction`` of the rows (the data are expected in random order) is held out and scored as ``\|pred - y\| / max(\|y\|, rel_floor)``. |
| `freeze` | função | `(model) -> nn.Module` | Stop gradients into the weights: from here on only the design moves. |
| `descend` | função | `(loss_fn, x0, *, bounds, steps=400, lr=0.01, reparam='clamp', project=None) -> dict[str, …)` | Adam on the design variables ``x0`` to minimise ``loss_fn(values)``. |
| `banded` | função | `(name, error, pass_limit, warn_limit, *, pass_text, warn_text, fail_text) -> CheckResult` | PASS when ``error <= pass_limit``, WARNING up to ``warn_limit``, FAIL above. |
| `relative_error` | função | `(predicted, reference, floor) -> float` |  |
| `constraint_checks` | função | `(violations, ok_text=…, skip=()) -> list[CheckResult]` | One check per signed violation (``<= 0`` is satisfied). |

#### `pinneapple_design.surrogate_inverse.heatsink_fins`

Straight-fin CPU heat sink: fin count, height and thickness on a fixed 80 x 80 mm aluminium base.

<small>`pinneapple_design/surrogate_inverse/heatsink_fins.py` · 264 linhas · testes: 91</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HeatsinkDesign` | classe | `(fin_count, fin_height_m, fin_thickness_m)` |  |
| `OptimizationResult` | classe | `(design, surrogate_predicted_temp_c, steps_run)` |  |
| `profile_params` | função | `(design, power_w, h_conv) -> dict[str, float]` | The fin-array formula's intermediate quantities (``m``, ``lc_m``, resistances, fin-root rise). |
| `junction_temperature_c` | função | `(design, power_w, h_conv) -> float` | Ground truth: base (junction) temperature. Used for the training labels and for the verification. |
| `thermal_resistance_k_w` | função | `(design, h_conv) -> float` | ``(T_j - T_amb) / P``: independent of the power in this model. |
| `fin_temperature_at_height_c` | função | `(design, power_w, h_conv, z_above_root_m) -> float` | The 1-D fin profile ``theta(z) = theta_root cosh(m (Lc - z)) / cosh(m Lc)`` of the same formula. |
| `thermal_field` | função | `(design, power_w, h_conv) -> dict[str, float]` | What a viewer needs to colour the fins with the fin profile (evaluated per vertex). |
| `constraint_violations` | função | `(design) -> dict[str, float]` | Signed violations (``<= 0`` satisfied). |
| `design_to_dict` | função | `(design) -> dict` |  |
| `generate_dataset` | função | `(n_samples, seed=0) -> tuple[torch.Tensor, torch.Tensor]` | Uniform designs, ``h`` spread around the fan profiles; label = thermal resistance from the formula. |
| `train_surrogate` | função | `(n_samples=6000, epochs=1500, seed=0) -> SurrogateFit` | MLP on ``log R``; ``fit.val_mean_rel_err`` is the held-out error on ``R``, i.e. on the temperature rise. |
| `predict_temperature_c` | função | `(surrogate, design, power_w, h_conv) -> float` |  |
| `optimize_design` | função | `(surrogate, power_w, h_conv, steps=400, lr=0.05, penalty_weight=100.0) -> OptimizationRes…)` | Minimise the surrogate's junction temperature over (N, H, t) from the baseline. |
| `verify_design` | função | `(design, power_w, h_conv, surrogate_predicted_temp_c) -> VerificationCard` | Re-run the formula on the final design: surrogate error on the temperature rise (PASS <= 5 %, WARNING <= 15 %), every design rule, and a plausibility bound on the temperature. |
| `fin_cell_check` | função | `(design, power_w, h_conv) -> dict` | The formula against 2-D finite-volume conduction in one fin pitch (same h, same power per fin). |
| `mesh` | função | `(design) -> meshes.Mesh` | Base plate plus fins, in millimetres. |
| `export_stl_bytes` | função | `(design) -> bytes` |  |

#### `pinneapple_design.surrogate_inverse.meshes`

Triangle meshes of the five designs, in millimetres, for a 3-D viewer or an STL download.

<small>`pinneapple_design/surrogate_inverse/meshes.py` · 93 linhas · testes: 88</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `merge` | função | `(parts) -> Mesh` | Concatenate meshes (faces re-indexed). |
| `bbox_size` | função | `(mesh) -> tuple[float, float, float]` |  |
| `to_stl` | função | `(mesh, name) -> bytes` | Binary STL (the header starts with the part name, never with ``solid``). |
| `fin_array` | função | `(base_w_mm, base_d_mm, base_t_mm, n_fins, fin_h_mm, fin_t_mm) -> Mesh` | Base plate (centred on the origin, top face at ``z = base_t / 2``) with ``n_fins`` straight fins spread evenly across its width, first and last flush with the edges. |
| `vented_rotor` | função | `(r_mean_mm, radial_len_mm, ring_t_mm, gap_mm, n_vanes, vane_t_mm, n=96) -> Mesh` | Two annular friction rings (z = 0 to ``ring_t`` and above the vent gap) joined by ``n_vanes`` radial vane blocks in the gap, around the z axis. |
| `layered_disc` | função | `(radius_mm, layer_thicknesses_mm, n=96) -> Mesh` | Stacked coaxial discs (z up from 0), e.g. a drug reservoir under a rate-controlling membrane. |
| `rectangular_beam` | função | `(width_mm, thickness_mm, length_mm) -> Mesh` | A plain box: x = width, y = thickness (the bending axis), z = length, centred on the origin (12 triangles). |
| `flanged_tube` | função | `(radius_mm, length_mm, flange_radius_mm, flange_t_mm, n=64) -> Mesh` | A port: a cylinder of ``radius`` x ``length`` along z from 0, with a flange disc at its base. |

#### `pinneapple_design.surrogate_inverse.reflex_port`

Bass-reflex port: radius and length that tune a cabinet to a target frequency, and the trade-off between a small port and an accurate tuning.

<small>`pinneapple_design/surrogate_inverse/reflex_port.py` · 232 linhas · testes: 88</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PortDesign` | classe | `(radius_m, length_m)` |  |
| `OptimizationResult` | classe | `(design, surrogate_predicted_freq_hz)` |  |
| `FrontierPoint` | classe | `(volume_weight, design, surrogate_predicted_freq_hz, real_achieved_freq_hz, real_freq_err…)` |  |
| `resonant_frequency_hz` | função | `(design, cabinet_volume_l) -> float` | Ground truth (training labels and verification). |
| `acoustic_field` | função | `(design, cabinet_volume_l) -> dict[str, float]` | One value per design (the lumped model has no field along the port): the tuning within the target range. |
| `constraint_violations` | função | `(design, cabinet_volume_l) -> dict[str, float]` |  |
| `design_to_dict` | função | `(design) -> dict` |  |
| `generate_dataset` | função | `(n_samples, seed=0) -> tuple[torch.Tensor, torch.Tensor]` |  |
| `train_surrogate` | função | `(n_samples=6000, epochs=1500, seed=0) -> SurrogateFit` |  |
| `predict_frequency_hz` | função | `(surrogate, design, cabinet_volume_l) -> float` |  |
| `optimize_design` | função | `(surrogate, cabinet_volume_l, target_freq_hz, steps=400, lr=0.05, penalty_weight=100.0) -…)` | Minimise ``((f_surrogate - f_target) / f_target)^2`` plus a penalty on the port-volume rule, from the baseline, in the sigmoid-bounded space of :func:`.core.descend` (radius and length bounded, comparable step sizes). |
| `sweep_pareto_frontier` | função | `(surrogate, cabinet_volume_l, target_freq_hz, n_points=8, steps=400, lr=0.05, penalty_wei…)` | Volume-vs-accuracy trade-off for ``lambda`` in ``linspace(0.05, 0.95, n_points)``. |
| `verify_design` | função | `(design, cabinet_volume_l, target_freq_hz, surrogate_predicted_freq_hz) -> VerificationCa…)` |  |
| `mesh` | função | `(design) -> meshes.Mesh` | Port tube with its flange, in millimetres. |
| `export_stl_bytes` | função | `(design) -> bytes` |  |

#### `pinneapple_design.surrogate_inverse.suspension_arm`

Suspension arm as a rectangular cantilever: the lightest width x thickness that still reaches a fatigue life.

<small>`pinneapple_design/surrogate_inverse/suspension_arm.py` · 255 linhas · testes: 88</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ArmDesign` | classe | `(width_m, thickness_m)` |  |
| `OptimizationResult` | classe | `(design, surrogate_predicted_stress_pa)` |  |
| `SensitivityResult` | classe | `(width_m, thickness_m, load_n, life_cycles, mass_kg, d_life_d_width, d_life_d_thickness, …)` |  |
| `bending_stress_pa` | função | `(design, load_n) -> float` | Ground truth: bending stress at the clamped root, where the moment is largest. |
| `bending_stress_at_position_pa` | função | `(design, load_n, x_from_root_m) -> float` | ``sigma(x) = 6 F (L - x) / (w t^2)``: largest at the root, zero at the loaded tip. |
| `fatigue_life_cycles` | função | `(design, load_n) -> float` |  |
| `allowable_stress_for_life_pa` | função | `(target_life_cycles) -> float` |  |
| `stress_field` | função | `(design, load_n) -> dict[str, float]` |  |
| `constraint_violations` | função | `(design, load_n, target_life_cycles) -> dict[str, float]` |  |
| `design_to_dict` | função | `(design) -> dict` |  |
| `generate_dataset` | função | `(n_samples, seed=0) -> tuple[torch.Tensor, torch.Tensor]` |  |
| `train_surrogate` | função | `(n_samples=6000, epochs=1500, seed=0) -> SurrogateFit` | MLP on ``log sigma`` (the stress spans about 30 MPa to 6 GPa over the space). |
| `predict_stress_pa` | função | `(surrogate, design, load_n) -> float` |  |
| `optimize_design` | função | `(surrogate, load_n, target_life_cycles, steps=500, lr=0.0006, penalty_weight=5e-11) -> Op…)` | Minimise ``w t`` (mass at fixed length and density) plus ``penalty_weight * max(0, sigma - sigma_allow)^2`` through the surrogate's stress, from the baseline, clamped to the bounds. |
| `verify_design` | função | `(design, load_n, target_life_cycles, surrogate_predicted_stress_pa) -> VerificationCard` |  |
| `compute_sensitivity` | função | `(width_m, thickness_m, load_n, step_m=H_M) -> SensitivityResult` | Derivatives of life and mass at the design point by central differences on the closed form (never the surrogate), with a step of ``step_m`` metres. |
| `sensitivity_to_dict` | função | `(r) -> dict` |  |
| `mesh` | função | `(design) -> meshes.Mesh` | The beam as a plain box in millimetres (x width, y thickness, z length): the model has no fillets or holes. |
| `export_stl_bytes` | função | `(design) -> bytes` |  |

#### `pinneapple_design.surrogate_inverse.transdermal_patch`

Membrane-controlled release patch: area, membrane thickness and reservoir concentration for a target release rate and wear time, with the smallest area.

<small>`pinneapple_design/surrogate_inverse/transdermal_patch.py` · 249 linhas · testes: 88</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PatchDesign` | classe | `(area_m2, membrane_thickness_m, reservoir_conc_kg_m3)` |  |
| `OptimizationResult` | classe | `(design, surrogate_predicted_rate_mg_day)` |  |
| `release_rate_mg_day` | função | `(design) -> float` | Ground truth: steady release rate (training labels and verification). |
| `depletion_time_days` | função | `(design) -> float` |  |
| `reservoir_mass_mg` | função | `(design) -> float` |  |
| `concentration_at_membrane_depth` | função | `(design, z_from_reservoir_m) -> float` | Linear steady profile ``C(z) = K C_res (1 - z / L)`` (sink at the outer face). |
| `decay_rate_constant_per_day` | função | `(design) -> float` | First-order constant of the tail after depletion, ``k = D K / (L L_res) = 1 / t_dep`` (area cancels). |
| `simulate_release_timeseries` | função | `(design, duration_days=None, n_points=100) -> list[dict[str, float]]` | Rate, cumulative release and nominal reservoir left over time (default: 1.5 x the depletion time). |
| `concentration_field` | função | `(design) -> dict[str, float]` |  |
| `constraint_violations` | função | `(design, target_wear_days) -> dict[str, float]` |  |
| `design_to_dict` | função | `(design) -> dict` |  |
| `generate_dataset` | função | `(n_samples, seed=0) -> tuple[torch.Tensor, torch.Tensor]` |  |
| `train_surrogate` | função | `(n_samples=6000, epochs=1500, seed=0) -> SurrogateFit` | MLP on ``log rate`` (the rate spans about 0.5 to 2600 mg/day over the design space). |
| `predict_rate_mg_day` | função | `(surrogate, design) -> float` |  |
| `optimize_design` | função | `(surrogate, target_rate_mg_day, target_wear_days, steps=1200, lr=0.15, weight_rate=80.0, …)` | Minimise ``A / A_max + w_r (rel. rate error)^2 + w_w (rel. wear shortfall)^2`` in a sigmoid-bounded space (area, membrane and concentration differ by six orders of magnitude; in that space one learning rate moves all three). The wear term uses the closed-form depletion time and aims 3 % above the … |
| `verify_design` | função | `(design, target_rate_mg_day, target_wear_days, surrogate_predicted_rate_mg_day) -> Verifi…)` |  |
| `mesh` | função | `(design) -> meshes.Mesh` | Reservoir disc with the membrane on top, in millimetres (radius from the area). |
| `export_stl_bytes` | função | `(design) -> bytes` |  |

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

Density-based topology optimization: SIMP, density filter, Heaviside projection, OC and MMA.

<small>`pinneapple_design/topology/__init__.py` · 19 linhas · testes: 9</small>

#### `pinneapple_design.topology.driver`

Optimization loop: design variables -> (symmetry) -> density filter -> Heaviside projection -> problem.

<small>`pinneapple_design/topology/driver.py` · 127 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DesignChain` | classe | `(n_el, filt=None, symmetry=None, beta=0.0, eta=0.5)` | ``x_phys = H_beta(W (S x))`` and its adjoint (chain rule). |
| `TopOptResult` | classe | `(x, x_phys, objective, history)` |  |
| `optimize` | função | `(problem, volfrac, filt=None, optimizer='oc', max_iter=200, tol=0.01, x0=None, symmetry=N…)` | Run density-based topology optimization of ``problem`` (see module docstring). |
| `finite_difference_check` | função | `(fun, x, idx, h=1e-06) -> np.ndarray` | Relative errors ``\|g_adj - g_fd\| / max(\|g_fd\|, tiny)`` of the gradient of ``fun`` (returning ``(value, gradient)``) at the entries ``idx``, central differences with step ``h``. |

#### `pinneapple_design.topology.filters`

Density filter and Heaviside projection for density-based topology optimization.

<small>`pinneapple_design/topology/filters.py` · 115 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DensityFilter` | classe | `(centroids, r_min, volumes=None)` | Linear density filter ``x_tilde = W x`` with ``W = diag(1/(H v)) H diag(v)``. |
| `SymmetryMap` | classe | `(S)` | Element densities from fewer design variables: ``x_elem = S x_var`` (a 0/1 sparse matrix). |
| `heaviside` | função | `(x, beta, eta=0.5) -> np.ndarray` |  |
| `heaviside_derivative` | função | `(x, beta, eta=0.5) -> np.ndarray` |  |

#### `pinneapple_design.topology.magnetic`

Topology optimization of rotor iron for torque, on the 2D A_z machine model.

<small>`pinneapple_design/topology/magnetic.py` · 79 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RotorTorqueProblem` | classe | `(machine, thetas, I_peak, gamma, objective='mean', T_ref=1.0)` |  |

#### `pinneapple_design.topology.optimizers`

Design-update rules: optimality criteria (OC) and the method of moving asymptotes (MMA).

<small>`pinneapple_design/topology/optimizers.py` · 117 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MMA` | classe | `(n, m, xmin=0.0, xmax=1.0, move=0.5, asyinit=0.5, asyincr=1.2, asydecr=0.7, c=1000.0, d=1…)` | Method of moving asymptotes for ``min f0(x)  s.t.  f_i(x) <= 0,  xmin <= x <= xmax``. |
| `oc_update` | função | `(x, dfdx, dvdx, volume_target, volume_of, move=0.2, eta=0.5, xmin=0.0, xmax=1.0)` | One OC step: ``x_new = clip(x (max(-df/dx, 0) / (lambda dv/dx))^eta)`` with ``volume_of(x_new) = volume_target`` found by bisection on ``lambda``. ``volume_of`` maps design variables to the volume. |

#### `pinneapple_design.topology.problems`

Density-based topology-optimization problems on structured grids of square bilinear (Q4) elements.

<small>`pinneapple_design/topology/problems.py` · 264 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Q4Grid` | classe | `(nelx, nely, h=1.0)` |  |
| `Compliance2D` | classe | `(nelx, nely, fixed_dofs, loads, penal=3.0, E0=1.0, Emin=1e-09, nu=0.3)` |  |
| `ThermalCompliance2D` | classe | `(nelx=60, nely=60, sink_fraction=0.1, q=1.0, penal=3.0, kmin=0.001)` | Square plate, uniform heat source ``q`` per unit area, ``T = 0`` on the middle ``sink_fraction`` of the left edge, insulated elsewhere. |
| `StokesDissipation2D` | classe | `(nelx, nely, inlets, outlets, mu=1.0, alpha_max=None, alpha_min=None, q=0.1, penalty=1000…)` | Velocity-driven Stokes flow with Brinkman penalization on a ``Lx x 1`` box. |
| `elasticity_ke` | função | `(nu=0.3, E=1.0, h=1.0, thickness=1.0) -> np.ndarray` | 8x8 plane-stress stiffness of a square Q4 element, dofs (ux, uy) per node. |
| `conduction_ke` | função | `(h=1.0) -> np.ndarray` | 4x4 conduction matrix (unit conductivity) of a square Q4 element. |

### Exemplos de uso reais do repositório

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
- Os surrogates de perfil valem só dentro da caixa de treino dos pesos CST (`BOUNDS`); `out_of_bounds` existe para sinalizar isso.
- Vários módulos de `aero` não têm teste direto: `airliner`, `airliner3d`, `buildup`, `case3d`, `optimize_airliner`, `render_blender`, `verify`.
- Dependências opcionais mudam o que roda: `pymoo-nsga2` exige pymoo, o otimizador evolutivo usa `cma` se houver, o bayesiano e o `GPSurrogate` usam scikit-learn; métodos impossíveis voltam com status `skipped`.
- `Evaluator.torch` e os avaliadores não diferenciáveis levantam `NotImplementedError` (métodos por gradiente só funcionam com avaliadores diferenciáveis, como a fidelidade surrogate do dissipador).
- Os objetivos de `objective.py` (arrasto, eficiência térmica, tensão, peso) são proxies sobre o campo previsto, não grandezas físicas exatas.
- O preset de perfil não traz solver de referência: os candidatos ficam como 'só surrogate' até serem verificados com OpenFOAM fora do pacote; o modelo de custo do dissipador é uma suposição do usuário, não dado de mercado.
- `meshops.union` exige `manifold3d`; `exporters.step_check` exige `cadquery` (OpenCASCADE); o carregamento do Hub depende de rede (há modo offline e reconstrução pelo código).
- As dimensões das entradas vêm de dados publicados, mas algumas são estimativas: `Source(estimate=True)` marca valores escolhidos por julgamento.
- O STEP gerado é um B-rep facetado (triângulos planos), não superfícies NURBS exatas.
- Os módulos de `entries` e o renderizador por software não têm teste direto.
- A verificação completa depende de programas externos: CalculiX e gmsh para o suporte, NVIDIA Newton para o loop; sem eles o veredito fica como não verificado.
- Só há quatro templates embutidos (loop, suporte, dissipador, asa de planador); pedidos fora deles voltam como `OutOfScope`. A mola é só exemplo e não é registrada.
- No template da asa de planador, a fidelidade 'full' é um vortex lattice fino; o docstring diz que um modelo RANS seria o próximo passo e não roda aqui.
- As propriedades de materiais são valores típicos de datasheet, não admissíveis de norma, como o próprio módulo avisa.
- O LLM é opcional (via `IntentLLM`/`CallLLM`); o parser determinístico não adivinha valores e transforma slots críticos ausentes em perguntas.
- A maior parte dos módulos de `intent` (api, cli, pipeline, package, spec, template, templates) não tem teste direto pela contagem do inventário.
- `sample_domain` chama `domain.sample_boundary`, mas `PhysicsDomain2D`/`PhysicsDomain3D` só têm `sample_boundary_region`; com os domínios de `build_domain` a chamada deve falhar (o exemplo da documentação usa exatamente esse caminho).
- `geometry.io.__init__` tenta importar `build_parametric_part` de `cadquery_bridge`, que não define esse nome; o valor fica sempre `None`.
- `gen.primitives.project_points_to_surface` é marcado como placeholder; o docstring diz que deveria levantar erro, mas devolve uma tupla `(pontos, distâncias)` em vez de um array.
- Muitas partes dependem de pacotes opcionais: trimesh, meshio, gmsh, CadQuery/OCC, pymeshlab, open3d, NVIDIA Newton (`bricks.dynamics`), Blender/`bpy` e ffmpeg (renders), `pinneapple_autolab` (agente). Os `__init__` usam imports tolerantes que viram `None` em silêncio.
- Vários módulos não têm teste direto: os construtores de lote (`builders`), `core.geometry`, `core.registry`, `io.*` (exceto IGES e trimesh), `mesh.mesher_2d`/`mesher_3d`, `ops.clean`, `ops.voxelize`, `sample.*`, `threaded_connection`, `genetic_sdf` e quase todo `thermal.board_cooling` e `thermal.heatsink.surrogate`/`monotone`/`trust`.
- `bricks.parts.JointModel` usa encaixe de 0,98 N por pino com faixa assumida (0,5 a 1,5 vez); os resultados da estática dependem dessa hipótese, e `bricks.report.sensitivity` existe para medi-la. `dynamics.friction_creep` documenta um artefato do atrito XPBD.
- `domains3d.get_pinn_batch` devolve alvos de contorno zerados (marcação), não valores físicos; o `STLDomainBatchBuilder` recentraliza a malha, o que quebra funções de valor que assumem coordenadas literais (motivo de existir `analytic_domain_batch_builder`).
- Os surrogates de `thermal.heatsink` só servem para triagem: `load_surrogates` só carrega o que existir em `ARTIFACTS` (`apps/heatsink_sizer/artifacts` ou `PINNEAPPLE_HEATSINK_ARTIFACTS`), e todo projeto recomendado é reverificado pelo motor físico.

## 2. `pinneapple_systems`

### Objetivo

_Sem texto de objetivo._

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 182 | 37.262 | 301 | 515 | 579 (40 arquivos) | 78 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `component_library` |  | 7 | 15 | 36 |
| `component_modeling` |  | 8 | 12 | 165 |
| `cosimulation` |  | 10 | 23 | 151 |
| `digital_twin` |  | 33 | 129 | 172 |
| `excitation` |  | 1 | 12 | 6 |
| `operations` |  | 10 | 73 | 25 |
| `process_components` |  | 17 | 112 | 171 |
| `sports` |  | 12 | 87 | 16 |
| `time_series` |  | 83 | 351 | 151 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_systems`

pinneapple_systems — Multi-component physical systems (time series, co-simulation, digital twins).

<small>`pinneapple_systems/__init__.py` · 280 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `cosim_from_models` | função | `(models_dict, connections) -> 'CoSimGraph'` | Build a CoSimGraph from a dict of named nn.Module instances. |
| `forecast` | função | `(data, horizon, *, model='auto', **kwargs)` | Quick time-series forecast with auto model selection. |

#### `pinneapple_systems.component_library`

pinneapple_systems.component_library — the piece ``pinneapple_systems.component_modeling``'s own docstring says pinneapple deliberately did NOT build: a registry of NAMED PLANT COMPONENTS (keyed by ``component_type``, e.g. "pipe"/"pump"/"valve"), each pairing an existing …

<small>`pinneapple_systems/component_library/__init__.py` · 44 linhas · testes: 36</small>

#### `pinneapple_systems.component_library.base`

pinneapple_systems.component_library.base — ``ComponentModel``: a ``BaseModel`` that additionally knows its physical constraint (a ``Physics``) and offers single-call ``.fit()``/``.evaluate()`` convenience, closing the two real gaps found against a plain ``BaseModel``/``PINNBase``:

<small>`pinneapple_systems/component_library/base.py` · 173 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ComponentModel` | classe | `(*, architecture, architecture_kwargs=None, physics=None)` | A named plant component: an existing architecture + an optional ``Physics`` constraint. |

#### `pinneapple_systems.component_library.benchmark`

pinneapple_systems.component_library.benchmark — train every architecture registered under one ``component_type`` and rank by final validation loss (real numbers from ``ComponentModel.fit()``, not a display placeholder).

<small>`pinneapple_systems/component_library/benchmark.py` · 77 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): placeholder/stub citado no docstring</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ComponentBenchmarkResult` | classe | `(component_type, ranked)` |  |
| `benchmark_component_type` | função | `(component_type, train_loader, val_loader=None, *, epochs=50, lr=0.001, eval_coords=None,…)` | Trains every architecture registered for ``component_type`` on the same data and ranks them by final validation loss (ascending — lower is better). If ``eval_coords``/``eval_targets`` are given, also attaches RMSE/MAPE/R² per architecture via ``ComponentModel.evaluate()``. |

#### `pinneapple_systems.component_library.components`

pinneapple_systems.component_library.components — six reference plant components. Deliberately six, not padded to ten: each pairs a genuinely different (physics shortcut, backbone architecture, component_type) combination rather than near-duplicating the others.

<small>`pinneapple_systems/component_library/components.py` · 126 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PipeVanillaPINN` | classe | `(in_dim=3, out_dim=3, hidden=(128, 128, 128, 128), **kwargs)` |  |
| `HeatExchangerVanillaPINN` | classe | `(in_dim=2, out_dim=1, hidden=(64, 64, 64), **kwargs)` |  |
| `TankShellVanillaPINN` | classe | `(in_dim=2, out_dim=2, hidden=(64, 64, 64), **kwargs)` |  |
| `BatteryCellVanillaPINN` | classe | `(in_dim=2, out_dim=1, hidden=(64, 64, 64), **kwargs)` |  |
| `PumpModifiedMLP` | classe | `(in_dim=3, out_dim=3, hidden_dim=128, n_layers=6, **kwargs)` |  |
| `ValveModifiedMLP` | classe | `(in_dim=2, out_dim=1, hidden_dim=64, n_layers=3, **kwargs)` |  |

#### `pinneapple_systems.component_library.physics`

pinneapple_systems.component_library.physics — ``Physics``: the thin attachment layer between a named plant component and the constraint it must satisfy.

<small>`pinneapple_systems/component_library/physics.py` · 88 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Physics` | classe | `(name, loss_fn, params)` | A named physical constraint, reduced to a single ``Trainer``-shaped loss_fn. ``params`` is kept alongside for introspection (e.g. shown on a Toolbox/Component detail view) even though it is already baked into ``loss_fn`` via closure. |

#### `pinneapple_systems.component_library.registry`

pinneapple_systems.component_library.registry — ``ComponentRegistry``: a registry of NAMED PLANT COMPONENTS keyed by ``component_type`` (e.g. "pipe", "pump", "valve") — several competing architectures may register under the same ``component_type``, mirroring how …

<small>`pinneapple_systems/component_library/registry.py` · 122 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ComponentSpec` | classe | `(component_type, name, cls, physics, description, tags)` |  |
| `ComponentRegistry` | classe | `(_REGISTRY)` | Central registry for named plant components. |
| `register_component` | função | `(*, component_type, name, physics=None, description='', tags=None)` | Module-level alias for ``ComponentRegistry.register`` — reads better as a bare decorator: ``@register_component(component_type=..., name=...)``. |

#### `pinneapple_systems.component_library.toolbox`

pinneapple_systems.component_library.toolbox — ``Toolbox``: a named, versioned, installable bundle of ``component_type``s (e.g. "Fluid Mechanics Toolbox v1.0" bundling pipe+pump+valve).

<small>`pinneapple_systems/component_library/toolbox.py` · 110 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Toolbox` | classe | `(name, version, component_types, description, depends_on)` |  |
| `ToolboxRegistry` | classe | `(_REGISTRY, _INSTALLED)` | Registry of :class:`Toolbox` bundles, plus a tiny installed-set. |

#### `pinneapple_systems.component_modeling`

pinneapple_systems.component_modeling — control, optimization, and uncertainty-quantification tooling that operates on ANY differentiable component model, independent of what that component represents physically.

<small>`pinneapple_systems/component_modeling/__init__.py` · 60 linhas · **sem teste direto**</small>

#### `pinneapple_systems.component_modeling.bayesian`

pinneapple_systems.component_modeling.bayesian — SWAGApproximation: a real, architecture-agnostic approximate Bayesian posterior over a model's weights, via SWAG (Stochastic Weight Averaging Gaussian — Maddox et al. 2019, "A Simple Baseline for Bayesian Uncertainty in Deep Learning").

<small>`pinneapple_systems/component_modeling/bayesian.py` · 127 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SWAGApproximation` | classe | `(model)` |  |

#### `pinneapple_systems.component_modeling.control`

pinneapple_systems.component_modeling.control — PID feedback control for an arbitrary differentiable or black-box component.

<small>`pinneapple_systems/component_modeling/control.py` · 113 linhas · testes: 165</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PIDController` | classe | `(kp, ki, kd, setpoint, output_limits=None, bias=0.0)` | Standard discrete-time PID: u = bias + Kp*e + Ki*integral(e) + Kd*de/dt, e = setpoint - measurement. |
| `run_closed_loop` | função | `(plant, *, setpoint, kp=1.0, ki=0.1, kd=0.0, steps=50, dt=0.1, process_noise=0.0, measure…)` | Closes a discrete-time control loop around ``plant``. |

#### `pinneapple_systems.component_modeling.edge`

pinneapple_systems.component_modeling.edge — package a trained model as a self-contained edge-deployment bundle (ONNX graph + manifest.json zipped together), and a minimal runtime that can execute that bundle with onnxruntime + numpy alone (no torch/pinneapple import at all).

<small>`pinneapple_systems/component_modeling/edge.py` · 96 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EdgeRuntime` | classe | `(zip_path)` | Minimal edge-device inference runtime — onnxruntime + numpy only, no torch/pinneapple import at all. Loads an ``export_edge_package()`` zip and runs real ONNX inference. |
| `export_edge_package` | função | `(model, output_dir, *, name, sample_input, opset_version=17, extra_manifest=None) -> Dict…)` | Exports `model` to ONNX and zips it with a manifest.json into a self-contained edge deployment package. Returns {"manifest": {...}, "onnx_path": ..., "zip_path": ...}. |

#### `pinneapple_systems.component_modeling.ensemble`

pinneapple_systems.component_modeling.ensemble — DeepEnsemble: N independently-initialized copies of the same architecture, trained separately, combined at inference time.

<small>`pinneapple_systems/component_modeling/ensemble.py` · 84 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DeepEnsemble` | classe | `(model_factory, n_members=5, base_seed=0)` |  |

#### `pinneapple_systems.component_modeling.mc_dropout`

pinneapple_systems.component_modeling.mc_dropout — MC-Dropout epistemic uncertainty: n_samples stochastic forward passes with dropout kept active, reduced to a (mean, std) estimate.

<small>`pinneapple_systems/component_modeling/mc_dropout.py` · 41 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `mc_dropout_uncertainty` | função | `(model, x, n_samples=50) -> Tuple[torch.Tensor, torch.Tensor]` | Returns (mean, std) over `n_samples` stochastic forward passes with dropout active. Raises ValueError if `model` has no `nn.Dropout` submodule (nothing would actually be stochastic). |

#### `pinneapple_systems.component_modeling.mpc`

pinneapple_systems.component_modeling.mpc — Model Predictive Control (receding horizon) around an arbitrary differentiable component.

<small>`pinneapple_systems/component_modeling/mpc.py` · 119 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `run_mpc` | função | `(plant, *, in_dim, setpoint, control_input_index=0, observed_output_index=0, horizon=10, …)` | Receding-horizon MPC: at each of ``steps`` outer steps, optimize a length-``horizon`` control sequence (Adam, ``opt_iters`` iterations, gradients flowing through ``plant`` itself) minimizing:: |

#### `pinneapple_systems.component_modeling.physics_residuals`

pinneapple_systems.component_modeling.physics_residuals — generic, closed-form PDE residuals for a handful of standard physical constraints, computed via autograd against any differentiable model's coordinate -> field(s) map.

<small>`pinneapple_systems/component_modeling/physics_residuals.py` · 126 linhas · testes: 8 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `incompressible_continuity_residual` | função | `(model, coords) -> torch.Tensor` | Mass-conservation (div(u) = 0) residual for a velocity-field output. Assumes the model's first `dim` output columns are velocity components aligned with the first `dim` coordinate axes (u, v, [w]). |
| `heat_conduction_residual` | função | `(model, coords, *, k=50.0, Q=0.0) -> torch.Tensor` | Steady-state heat diffusion residual: k*laplacian(T) + Q, T = model's first output column, laplacian summed over every axis of `coords` (works for any spatial dimension). |
| `linear_elasticity_residual` | função | `(model, coords, *, E=210000000000.0, nu=0.3) -> torch.Tensor` | Navier-Lame equilibrium (isotropic linear elasticity, no body force): (lambda+mu)*grad(div(u)) + mu*laplacian(u) = 0. Assumes the model's first `dim` output columns are displacement components [u, v, (w)]. Works in 2D or 3D. |
| `species_diffusion_residual` | função | `(model, coords, *, D=1e-05, k=0.0) -> torch.Tensor` | Simplified steady-state diffusion-reaction (Fick's second law at steady state, first-order decay): D*laplacian(C) - k*C = 0. Assumes the model's first output column is concentration. A generic Fickian diffusion-reaction constraint — not tied to any specific transport system. |

#### `pinneapple_systems.cosimulation`

PINNeAPPle native co-simulation engine.

<small>`pinneapple_systems/cosimulation/__init__.py` · 86 linhas · **sem teste direto**</small>

#### `pinneapple_systems.cosimulation.adapters`

Adapters connecting pinneapple_cosim to PINNeAPPle's model and problem ecosystems.

<small>`pinneapple_systems/cosimulation/adapters.py` · 432 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PINNeProblemNode` | classe | `(name, model, spec, compiled_fn, coord_ports, field_ports, extra_input_ports=None, physic…)` | Co-sim node backed by a PINNeAPPle ProblemSpec + compiled physics loss. |
| `PINNeAPPleModelNode` | classe | `(name, model, input_ports, output_ports, batch_port_map=None) -> None` | Wraps any ``BaseModel`` from ModelRegistry as a co-sim node. |
| `SymbolicPDENode` | classe | `(name, model, symbolic_pde, coord_ports, field_ports, physics_weight=1.0) -> None` | Co-sim node whose physics residual is defined by a SymPy expression. |
| `TimeSeriesCoSimNode` | classe | `(name, model, input_len, horizon, extra_input_ports=None, extra_output_ports=None) -> None` | Wraps a PINNeAPPle ``TSModelBase`` as a co-simulation node. |

#### `pinneapple_systems.cosimulation.connection`

Connection dataclass for co-simulation graphs.

<small>`pinneapple_systems/cosimulation/connection.py` · 81 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Connection` | classe | `(src_node, src_port, dst_node, dst_port)` | Immutable directed edge between two node ports. |

#### `pinneapple_systems.cosimulation.engine`

Co-simulation engine: time-stepped execution with algebraic-loop iteration.

<small>`pinneapple_systems/cosimulation/engine.py` · 257 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CoSimEngine` | classe | `(graph, recorder=None, loop_solver='gauss_seidel', max_iter=50, tol=1e-06, retain_graph=F…)` | Time-stepped co-simulation engine over a ``CoSimGraph``. |

#### `pinneapple_systems.cosimulation.graph`

Co-simulation graph: node registry, connection management, topology analysis.

<small>`pinneapple_systems/cosimulation/graph.py` · 187 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CoSimGraph` | classe | `() -> None` | Registry of nodes and connections with lazy topology caching. |

#### `pinneapple_systems.cosimulation.losses`

Composite loss functions for training differentiable co-simulation graphs.

<small>`pinneapple_systems/cosimulation/losses.py` · 225 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DataLoss` | classe | `(weight=1.0) -> None` | MSE between predicted port values and observations. |
| `PhysicsLoss` | classe | `(weight=1.0) -> None` | Sum of physics residuals from all PINNNodes in the graph. |
| `CouplingLoss` | classe | `(weight=0.1) -> None` | Penalise interface inconsistencies between connected ports. |
| `CoSimLoss` | classe | `(data_weight=1.0, physics_weight=1.0, coupling_weight=0.1) -> None` | Composite loss: data fidelity + physics residuals + coupling consistency. |

#### `pinneapple_systems.cosimulation.node`

Co-simulation node abstractions.

<small>`pinneapple_systems/cosimulation/node.py` · 291 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CoSimNode` | classe | `(name, input_ports, output_ports) -> None` | Abstract base class for all co-simulation nodes. |
| `TorchNode` | classe | `(name, model, input_ports, output_ports) -> None` | Wraps a ``nn.Module``; gradients flow through ``step()``. |
| `AnalyticalNode` | classe | `(name, fn, input_ports, output_ports) -> None` | Wraps a pure Python/NumPy callable.  No autograd — outputs are detached. |
| `PINNNode` | classe | `(name, model, input_ports, output_ports, physics_fn, physics_weight=1.0) -> None` | ``TorchNode`` extended with a physics-residual loss. |
| `BlackBoxNode` | classe | `(name, fn, input_ports, output_ports) -> None` | Opaque callable node — fully detached from autograd. |
| `split_output_ports` | função | `(out, ports) -> Dict[str, torch.Tensor]` | Split *out*'s last dimension evenly across *ports*. |

#### `pinneapple_systems.cosimulation.recorder`

Trajectory recording for co-simulation runs.

<small>`pinneapple_systems/cosimulation/recorder.py` · 177 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Trajectory` | classe | `(node, port) -> None` | Time series for a single (node, port) signal. |
| `TrajectoryRecorder` | classe | `() -> None` | Records time-stamped port outputs from multiple nodes. |

#### `pinneapple_systems.cosimulation.trainer`

CoSimTrainer — trains co-simulation graphs using PINNeAPPle's training stack.

<small>`pinneapple_systems/cosimulation/trainer.py` · 331 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CoSimTrainer` | classe | `(graph, engine, criterion=None, weight_scheduler_cfg=None, early_stopping=None, checkpoin…)` | Trains a differentiable co-simulation graph with PINNeAPPle's training stack. |

#### `pinneapple_systems.cosimulation.utils`

Graph algorithms for co-simulation topology analysis.

<small>`pinneapple_systems/cosimulation/utils.py` · 180 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `tarjan_scc` | função | `(adj) -> List[List[str]]` | Find all Strongly Connected Components using Tarjan's algorithm. |
| `condensation_order` | função | `(adj) -> Tuple[List[List[str]], List[List[str]]]` | Compute execution order via SCC condensation. |
| `topological_sort` | função | `(adj) -> List[str]` | Topological sort for pure DAGs (no cycles). |

#### `pinneapple_systems.digital_twin`

pinneapple_digital_twin — Digital twins built on top of pinneapple surrogate models.

<small>`pinneapple_systems/digital_twin/__init__.py` · 121 linhas · testes: 15</small>

#### `pinneapple_systems.digital_twin.assimilation`

Data assimilation methods for digital twins.

<small>`pinneapple_systems/digital_twin/assimilation/__init__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_systems.digital_twin.assimilation.kalman`

``pinneapple_systems.digital_twin.assimilation.kalman`` compatibility shim -- the real ``ExtendedKalmanFilter`` / ``EnsembleKalmanFilter`` implementation moved to ``pinneapple_analysis.state_estimation.kalman`` (a capability audit found no general-purpose, standalone data-assimilation module usable …

<small>`pinneapple_systems/digital_twin/assimilation/kalman.py` · 20 linhas · testes: 157 · palavras-chave no código (heurística; o texto acima diz o status real): shim ou obsoleto</small>

#### `pinneapple_systems.digital_twin.conditioning`

Causal, sample-by-sample telemetry conditioning for production digital twins.

<small>`pinneapple_systems/digital_twin/conditioning.py` · 195 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SignalProfile` | classe | `(kind, tau_s, hampel_window, hampel_k, max_consecutive_outliers, max_rate, valid_range, m…)` | How to condition one kind of signal. Use a preset (``PROFILES``) or build your own. |
| `ConditionedSample` | classe | `(t, raw, value, flags)` |  |
| `StreamingConditioner` | classe | `(profile='generic', **overrides)` | Condition one signal, one reading at a time (causal; O(window) per reading). |
| `TelemetryConditioner` | classe | `(profiles=None, default='generic')` | Per-sensor, per-field conditioners for a whole twin. |

#### `pinneapple_systems.digital_twin.io`

I/O adapters for digital twin data streams.

<small>`pinneapple_systems/digital_twin/io/__init__.py` · 26 linhas · testes: 8</small>

#### `pinneapple_systems.digital_twin.io.mqtt_broker`

A small embedded MQTT 3.1.1 broker for demos, tests and single-asset edge deployments.

<small>`pinneapple_systems/digital_twin/io/mqtt_broker.py` · 263 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MQTTBroker` | classe | `(host='127.0.0.1', port=1883, username=None, password=None, max_packet=256 * 1024)` | Run with :meth:`start` (background thread with its own event loop) and :meth:`stop`. |
| `topic_matches` | função | `(filt, topic) -> bool` | MQTT topic-filter matching (``+`` one level, ``#`` the rest; ``$`` topics need an explicit first level). |

#### `pinneapple_systems.digital_twin.io.sensors`

Sensor management for digital twins.

<small>`pinneapple_systems/digital_twin/io/sensors.py` · 151 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Sensor` | classe | `(sensor_id, coords, field_names, calibration_offset, calibration_scale, noise_std, transf…)` | Represents a single physical sensor. |
| `SensorRegistry` | classe | `() -> None` | Registry of all sensors in the digital twin. |

#### `pinneapple_systems.digital_twin.io.stream`

Real-time data stream adapters for digital twins.

<small>`pinneapple_systems/digital_twin/io/stream.py` · 576 linhas · testes: 172</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BaseStream` | classe | `(sensor_id, field_names) -> None` | Abstract base class for all data stream adapters. |
| `FileWatchStream` | classe | `(path, sensor_id, field_names, *, poll_interval=1.0, format='jsonl') -> None` | Polls a JSON-lines file (or CSV/Parquet) for new rows. |
| `MQTTStream` | classe | `(broker, topic, sensor_id, field_names, *, port=1883, keepalive=60, username=None, passwo…)` | Subscribes to an MQTT topic and emits Observations. |
| `HTTPPollStream` | classe | `(url, sensor_id, field_names, *, poll_interval=5.0, headers=None, transform=None) -> None` | Periodically polls a REST endpoint (GET) and emits Observations. |
| `KafkaStream` | classe | `(bootstrap_servers, topic, sensor_id, field_names, *, group_id='pinneapple_dt', auto_offs…)` | Reads messages from an Apache Kafka topic. |
| `OPCUAStream` | classe | `(server_url, node_ids, sensor_id, field_names, *, poll_interval=1.0) -> None` | Polls an OPC-UA server for real-time tag values. |
| `ModbusStream` | classe | `(host, port, register_map, sensor_id, field_names, *, poll_interval=1.0, device_id=1) -> …)` | Polls a Modbus TCP server's holding registers for real-time values. |
| `MockStream` | classe | `(sensor_id, field_names, generator_fn, *, tick_interval=0.1, coords=None) -> None` | Synthetic data stream for testing. |

#### `pinneapple_systems.digital_twin.io.usgs`

Client of the USGS Water Services "Instantaneous Values" API (public, no key): river discharge, gage height and other parameters of US stream gauges every 5-15 minutes.

<small>`pinneapple_systems/digital_twin/io/usgs.py` · 100 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `USGSDataError` | classe | `()` | The USGS service could not be reached or returned no usable data. |
| `Series` | classe | `(site_id, site_name, parameter_cd, unit, timestamps, values, fetched_at, from_cache)` |  |
| `monitoring_url` | função | `(site_id) -> str` | Public USGS page of a gauge, for a reader to check the data. |
| `parse_iv` | função | `(payload, site_id, parameter_cd) -> Series` | The first time series of an IV JSON payload, malformed readings skipped, sorted by time. |
| `fetch_iv` | função | `(site_id, parameter_cd=DISCHARGE, period='P5D', timeout_s=10.0, cache_ttl_s=90.0) -> Seri…)` | Download one parameter of one gauge over the trailing ``period`` (ISO 8601 duration, e.g. ``"P5D"``). |

#### `pinneapple_systems.digital_twin.monitoring`

Monitoring and anomaly detection for digital twins.

<small>`pinneapple_systems/digital_twin/monitoring/__init__.py` · 17 linhas · **sem teste direto**</small>

#### `pinneapple_systems.digital_twin.monitoring.anomaly`

Anomaly detection for digital twins.

<small>`pinneapple_systems/digital_twin/monitoring/anomaly.py` · 415 linhas · testes: 166</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AnomalyEvent` | classe | `(timestamp, sensor_id, field_name, observed, predicted, score, detector, metadata)` | Records a detected anomaly. |
| `ThresholdDetector` | classe | `(thresholds, rel_thresholds=None, eps=1e-08) -> None` | Simple threshold-based anomaly detector. |
| `ZScoreDetector` | classe | `(z_threshold=3.0, window_size=100, min_samples=10) -> None` | Rolling z-score anomaly detector. |
| `MahalanobisDetector` | classe | `(threshold=3.0, field_order=None) -> None` | Multivariate Mahalanobis distance anomaly detector. |
| `ResidualDetector` | classe | `(z_threshold=3.0, window_size=100, min_samples=10) -> None` | Checks the innovation residuals produced by the assimilation filter. |
| `AnomalyMonitor` | classe | `() -> None` | Composite anomaly monitor that runs multiple detectors in sequence. |

#### `pinneapple_systems.digital_twin.pipe_loop`

The digital twin that knows when it is wrong (Autonomous Physics Lab, demo 3): a pumped, heated water loop with an elbow and a T junction, a fast model of its sensors, and a diagnosis that tells a changed process from a wrong sensor from a model outside its domain -- with physics residuals, a …

<small>`pinneapple_systems/digital_twin/pipe_loop/__init__.py` · 31 linhas · testes: 7</small>

#### `pinneapple_systems.digital_twin.pipe_loop.agent`

The digital twin that knows when it is wrong: a monitoring mission -> plan -> controlled tools -> evidence.

<small>`pinneapple_systems/digital_twin/pipe_loop/agent.py` · 217 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `t_read_mission` | função | `(state, text) -> dict[str, Any]` |  |
| `t_build_twin` | função | `(state, n_train=120, n_test=40, seed=0) -> dict[str, Any]` |  |
| `t_calibrate` | função | `(state, n_runs=12) -> dict[str, Any]` |  |
| `t_monitor` | função | `(state) -> dict[str, Any]` |  |
| `t_request_simulation` | função | `(state, n_new=40, widen=0.25) -> dict[str, Any]` | For every showcase case diagnosed as outside the domain: run the reference at the operating point, compare it with the readings (does the plant still obey the physics?), extend the fast model's envelope by ``widen`` of the box beyond the input that left it, add ``n_new`` reference solves there, … |
| `t_benchmark` | função | `(state, n_per_kind=20, n_calibration=12, workers=4) -> dict[str, Any]` |  |
| `t_trust` | função | `(state) -> dict[str, Any]` |  |
| `t_deliver` | função | `(state, out_dir='out', render=True, samples=16) -> dict[str, Any]` |  |
| `toolbox` | função | `() -> Toolbox` |  |
| `template_plan` | função | `(text, *, out_dir='out', render=True, n_per_kind=20, workers=4) -> TemplatePlanner` |  |
| `run_mission` | função | `(text=DEFAULT_TEXT, *, out_dir='out', planner='template', render=True, n_per_kind=20, wor…)` |  |

#### `pinneapple_systems.digital_twin.pipe_loop.benchmark`

Controlled test cases with a known answer, and the confusion matrix of the diagnosis.

<small>`pinneapple_systems/digital_twin/pipe_loop/benchmark.py` · 123 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `make_cases` | função | `(n_per_kind=20, seed=0) -> list[Scenario]` |  |
| `run_benchmark` | função | `(loop=None, *, n_per_kind=20, n_calibration=12, seed=0, n_train=120, workers=4, log=None)…)` | Fit the twin's fast model, calibrate the diagnoser on healthy runs, diagnose every controlled case and return the confusion matrix, isolation scores, false alarms, delays and timings. |

#### `pinneapple_systems.digital_twin.pipe_loop.diagnose`

How the twin knows it is wrong: physics residuals, a fast model, a Kalman filter and its domain of validity.

<small>`pinneapple_systems/digital_twin/pipe_loop/diagnose.py` · 313 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Residuals` | classe | `(loop)` | Physics residuals of one reading vector, in sensor units, and their noise standard deviation (propagated from SIGMA by finite differences). |
| `Surrogate` | classe | `(n_train, seed, model, lo, hi)` | POD + Gaussian-process model of the ten readings over NORMAL_BOX (the twin's fast model). |
| `LossTracker` | classe | `(loop, q_walk=0.02, k0_sigma=0.5)` | EKF on the extra loss coefficient of each pipe (random walk), observed through the measured pressure drop of each pipe given its measured flow. |
| `Diagnoser` | classe | `(loop, surrogate, window, thresholds, ood_threshold)` | Windowed evidence -> hypothesis. ``window`` steps; thresholds from :meth:`calibrate` (healthy runs). |

#### `pinneapple_systems.digital_twin.pipe_loop.plant`

The plant: a pumped, heated water loop with an elbow and a T junction, and its reference simulator.

<small>`pinneapple_systems/digital_twin/pipe_loop/plant.py` · 258 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Pipe` | classe | `(length, diameter, roughness, k_minor, ua_w_k)` |  |
| `Loop` | classe | `(pipe_A, pipe_B, pipe_C, h0_m, k_pump, kv0_B, kv0_C, z_B, z_C, t_room_c, p_atm_head_m, wa…)` | Geometry and machines of the loop (SI). |
| `Faults` | classe | `(k_extra, bias, noise_gain)` | What is wrong at one instant (all zero: healthy). |
| `Scenario` | classe | `(name, kind, seed, duration_s, dt_s, onset_s, ramp_s, target, magnitude, mode)` | A controlled test case: an operating history and (optionally) one fault with its onset and ramp. |
| `water` | função | `(t_c) -> tuple[float, float]` | (density kg/m3, dynamic viscosity Pa s) at ``t_c`` °C. |
| `steady_state` | função | `(loop, u, k_extra=None) -> dict[str, float]` | True plant state for inputs ``u`` (keys of INPUTS): flows, heads, pressures and temperatures. Water properties at the mean of the inlet and heater-outlet temperatures (two passes). |
| `readings` | função | `(state) -> np.ndarray` | True sensor values in sensor units (bar, m3/h, °C), order SENSORS. |
| `quasi_steady_valid` | função | `(loop, stroke_s, factor=10.0) -> bool` | The quasi-steady model holds when a valve moves over many acoustic round trips: stroke >= factor 2L/a. |
| `operating_history` | função | `(sc) -> tuple[np.ndarray, dict[str, np.ndarray]]` | Slow, smooth operator moves inside NORMAL_BOX (sum of a few low-frequency sines per input, seeded); a domain scenario then drives its ``target`` input to ``magnitude`` (outside the box) from the onset. |
| `run_plant` | função | `(loop, sc) -> dict[str, Any]` | Simulate one scenario: inputs, true sensor values, noisy readings and the fault applied at each step. |
| `with_loop` | função | `(loop, **kw) -> Loop` |  |

#### `pinneapple_systems.digital_twin.pipe_loop.report`

Deliverables of the pipe-loop twin: control-room figures, confusion matrix, trust report, 3-D scene and report.

<small>`pinneapple_systems/digital_twin/pipe_loop/report.py` · 341 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `trust_report` | função | `(state) -> TrustReport` |  |
| `control_room` | função | `(case, path, *, title='', cursor=None) -> str` |  |
| `figure_confusion` | função | `(b, path) -> str` |  |
| `figure_request` | função | `(rs, path) -> str \| None` |  |
| `impact_gif` | função | `(case, path, n=14) -> str` |  |
| `render_3d` | função | `(case, out_dir, *, render, samples, size=(1280, 720)) -> dict[str, Any]` |  |
| `deliver` | função | `(state, out_dir, *, render=True, samples=16) -> dict[str, Any]` |  |
| `write_report` | função | `(state, out_dir, files, env) -> str` |  |

#### `pinneapple_systems.digital_twin.pipe_loop.scene`

3-D control-room scene of the loop: pipes coloured by pressure, a transparent elbow with its velocity profile, the T junction, valves, tanks, the pump and the sensors coloured by the twin's verdict, frame by frame.

<small>`pinneapple_systems/digital_twin/pipe_loop/scene.py` · 181 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `tube` | função | `(path, r=R_PIPE, n_theta=20, n_seg=40)` | Triangulated tube along a polyline (resampled to ``n_seg`` rings); returns vertices, faces and the arc-length fraction 0..1 of every vertex. |
| `elbow_streamlines` | função | `(v_bulk, n_r=3, n_t=8)` | Lines through the elbow and the straight runs next to it at a few radii, valued by the 1/7 power-law speed. |
| `loop_scene` | função | `(run, diagnosis, *, n_frames=24, title='')` | Animated scene of one monitored run: pressures on the walls, verdict on the sensors, HUD with the evidence. |

#### `pinneapple_systems.digital_twin.product`

The closed digital-twin product: spec, commissioning, live run, evidence, drift policy.

<small>`pinneapple_systems/digital_twin/product/__init__.py` · 83 linhas · testes: 24</small>

#### `pinneapple_systems.digital_twin.product.assimilation`

The twin's estimator: an ensemble Kalman filter over the state the model cannot know.

<small>`pinneapple_systems/digital_twin/product/assimilation.py` · 418 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TwinEstimate` | classe | `(theta, theta_std, innovation, n_obs)` | One analysis instant: the latent state, its spread, and the innovation. |
| `LatentStateTwin` | classe | `(model, spec, *, n_ens=64, inflation=1.02, seed=0) -> None` | Estimates the latent state of one asset from a spec and the model behind it. |

#### `pinneapple_systems.digital_twin.product.commission`

Commissioning: the two gates a twin passes before it is allowed to run.

<small>`pinneapple_systems/digital_twin/product/commission.py` · 588 linhas · testes: 24</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `read_timestamp` | função | `(frame) -> float` | The frame's timestamp, under any of the conventional keys. |
| `read_readings` | função | `(frame, spec) -> dict[str, float]` | The sensor values present in a frame, ignoring keys that are not sensors. |
| `preflight` | função | `(spec, history) -> CheckReport` | Check that this twin can be commissioned at all, before spending a backtest on it. |
| `backtest` | função | `(spec, model, history, *, n_ens=64, inflation=1.02, seed=0, warmup_steps=0, calibrate_fra…)` | Replay the history through the twin and measure it against the held-out sensors. |

#### `pinneapple_systems.digital_twin.product.evidence`

The evidence a twin has to produce to be trusted, and the shapes it is filed in.

<small>`pinneapple_systems/digital_twin/product/evidence.py` · 284 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CheckItem` | classe | `(check_id, status, title, detail, why, fix, source)` | One line of a preflight checklist. |
| `CheckReport` | classe | `(title, items)` | An ordered checklist with a verdict: any ``FAIL`` blocks commissioning. |
| `FieldMetrics` | classe | `(field_name, unit, n, rmse, mae, bias, p95_abs, coverage, span, window_s, coverage_raw, b…)` | Agreement between the twin and reality, on observations it never assimilated. |
| `TrackingReport` | classe | `(metrics, gates, breaches, alarm_lead_s, forecast_error_s, final_latent, n_steps)` | Per-field tracking metrics plus the gate verdict and the alarm lead times. |
| `TwinEvidence` | classe | `(asset_id, spec_digest, model_id, created_unix, preflight, tracking, decisions, unverifie…)` | Everything a commissioned twin produces, in one file, keyed by the spec digest. |

#### `pinneapple_systems.digital_twin.product.model`

What a model must accept to be run as a twin, and an adapter so existing surrogates fit.

<small>`pinneapple_systems/digital_twin/product/model.py` · 155 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TwinModelError` | classe | `()` | The model does not honour the twin contract. |
| `BiasModel` | classe | `(model, field_names) -> None` | Wrap a plain ``model(X)`` so it satisfies the twin contract with an output bias. |
| `as_array_dict` | função | `(out, field_names) -> dict[str, np.ndarray]` | Normalise a model output to ``{field: (n,) float32}``. |
| `ensure_twin_model` | função | `(model, field_names, n_latent)` | Return a callable that honours the contract, wrapping ``model`` when it does not. |

#### `pinneapple_systems.digital_twin.product.policy`

What the twin does when it stops agreeing with the plant.

<small>`pinneapple_systems/digital_twin/product/policy.py` · 166 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TwinMode` | classe | `()` |  |
| `DriftPolicy` | classe | `(window, target_coverage, hard_break, max_sustained, recalibrate_after, hysteresis)` | Thresholds for the consistency ladder. |
| `DriftMonitor` | classe | `(policy, _window, _mode, _off_run, _alarm_run, _ok_run, decisions)` | Rolling consistency of the twin's band, converted into a mode with hysteresis. |

#### `pinneapple_systems.digital_twin.product.runner`

The product surface: one call to commission a twin, one to run it, one to file the evidence.

<small>`pinneapple_systems/digital_twin/product/runner.py` · 346 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RunResult` | classe | `(n_steps, n_observed, n_holdout, final_mode, coverage, latent, modes_seen, decisions)` | What happened while the twin ran. |
| `commission` | função | `(spec, model, history, *, n_ens=64, inflation=1.02, seed=0, warmup_steps=0, calibrate_fra…)` | Run both gates and return the evidence file. |
| `run_live` | função | `(spec, model, frames, *, n_ens=64, inflation=1.02, seed=0, policy=None, evidence=None, th…)` | Assimilate a run of frames, monitor consistency, and record what the twin decided. |
| `describe` | função | `(spec, latent, fields=None)` | Describe the twin for the industry standards the platform already speaks. |
| `export_scene` | função | `(spec, frames, folder, *, mesh=None, field_on_mesh=None, times=None) -> str` | Write a ``pinneapple_twin3d`` scene for this twin: sensors with measured series and envelopes. |

#### `pinneapple_systems.digital_twin.product.spec`

The twin contract: what a digital twin is asked to deliver, signed before it runs.

<small>`pinneapple_systems/digital_twin/product/spec.py` · 438 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SpecError` | classe | `()` | The spec is not usable: a missing claim, a contradiction, an unobservable field. |
| `FieldSpec` | classe | `(name, unit, kind, lo, hi, alarm_above, alarm_below)` | One quantity the twin estimates. |
| `SensorSpec` | classe | `(sensor_id, unit, field, lo, hi, at, max_gap_s, noise_sigma, holdout, driver)` | A measurement available to the twin, with the envelope that makes it checkable. |
| `ModelSpec` | classe | `(model_id, version, accuracy_floor, validity)` | The model behind the twin, and the accuracy it claims. |
| `AcceptanceGates` | classe | `(max_rmse_frac, max_bias_frac, min_coverage, min_alarm_lead_s)` | The thresholds that decide whether a twin goes live. |
| `TwinSpec` | classe | `(asset_id, asset_name, asset_type, fields, sensors, model, asset_unit, coord_names, sampl…)` | A complete, checkable definition of one digital twin. |

#### `pinneapple_systems.digital_twin.prognostics`

Predictive maintenance: trend-based Remaining Useful Life (RUL) estimation.

<small>`pinneapple_systems/digital_twin/prognostics.py` · 628 linhas · testes: 15</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HealthIndicatorHistory` | classe | `(timestamps, values, source)` | A scalar health/degradation-indicator time series. |
| `RULEstimate` | classe | `(estimated_rul, rul_confidence_interval, failure_time_estimate, degradation_model_used, f…)` | Result of a trend-based Remaining Useful Life estimation. |
| `estimate_rul` | função | `(health_indicator_series, timestamps=None, failure_threshold=None, model='linear', confid…)` | Trend-based Remaining Useful Life (RUL) estimation. |
| `estimate_rul_from_twin` | função | `(twin, failure_threshold, *, detector=None, sensor_id=None, field_name=None, n=None, mode…)` | Convenience wrapper: estimate RUL directly from a live `DigitalTwin`. |

#### `pinneapple_systems.digital_twin.realtime_pipeline`

Real-time inference pipeline for digital twins.

<small>`pinneapple_systems/digital_twin/realtime_pipeline.py` · 470 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LatencyBudgetExceeded` | classe | `()` | Raised when inference time exceeds data_period and policy='raise'. |
| `LatencyStats` | classe | `(n_frames, n_skipped, n_budget_violations, mean_inference_ms, max_inference_ms, budget_ms…)` | Rolling latency statistics. |
| `RealtimePipelineConfig` | classe | `(data_period, latency_policy, buffer_maxlen, latency_alpha, throttle_min_points, throttle…)` | Configuration for RealtimePipeline. |
| `RealtimePipeline` | classe | `(model, data_period, field_names, coord_names=None, *, config=None, on_state=None, on_lat…)` | Real-time inference pipeline wrapping a surrogate model. |

#### `pinneapple_systems.digital_twin.recession`

A river digital twin from baseflow recession: calibrate on the recent past of a gauge, forecast the present, flag a sustained departure.

<small>`pinneapple_systems/digital_twin/recession.py` · 195 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `recession` | função | `(q0, k_per_hour, hours) -> float \| np.ndarray` | Q0 exp(-k t), t in hours since t0. |
| `fit_recession` | função | `(hours, values) -> dict[str, float]` | Least-squares Q0, k on the log-linearised recession; ``hours`` is re-based to its first sample (``t0_hours``). |
| `forecast` | função | `(calibration, hours) -> np.ndarray` | The calibrated curve at ``hours`` (same axis as the fit). |
| `detect_divergence` | função | `(hours, actual, predicted, residual_mean, residual_std, n_sigma=N_SIGMA, sustained_minute…)` | First sustained excursion of ``actual - predicted`` beyond ``residual_mean +- n_sigma * residual_std``. |
| `run_twin` | função | `(timestamps, values, *, stage_timestamps=None, stage_values=None, calibration_hours=CALIB…)` | Calibrate, forecast and compare one discharge series (and, optionally, fit the gage height the same way for the water level now vs forecast). Raises ``ValueError`` when the series is too short to calibrate. |
| `synthetic_recession` | função | `(hours=96.0, step_minutes=15.0, q0=2000.0, k_per_hour=0.004, noise=0.005, bump_at_hour=No…)` | A synthetic gauge record for tests and offline demos: an exponential recession with multiplicative noise and an optional step increase ``bump`` (fraction) from ``bump_at_hour`` on. Timestamps start 2026-01-01 UTC. |

#### `pinneapple_systems.digital_twin.signal_reconstruction`

Signal reconstruction from incoming data streams.

<small>`pinneapple_systems/digital_twin/signal_reconstruction.py` · 587 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ReconstructionResult` | classe | `(sensor_id, field_name, method, timestamps, signal, raw_timestamps, raw_values, quality, …)` | Output of a signal reconstruction pass. |
| `SignalReconstructor` | classe | `(twin_or_observations, *, default_method='ssa', default_n_out=512) -> None` | Convenience wrapper that reconstructs signals from the observation history held by a live ``DigitalTwin`` (or from any list of ``Observation`` objects). |
| `reconstruct_spectral` | função | `(timestamps, values, *, n_harmonics=10, out_times=None, min_points=8) -> Tuple[np.ndarray…)` | FFT-based reconstruction: fit the dominant harmonics and evaluate on a dense grid (or ``out_times`` if provided). |
| `reconstruct_ssa` | função | `(timestamps, values, *, window=0, n_components=5, out_times=None) -> Tuple[np.ndarray, np…)` | SSA-based reconstruction: decompose the signal and reconstruct from the dominant components (noise suppression). |
| `reconstruct_wavelet` | função | `(timestamps, values, *, wavelet='db4', level=4, threshold_mode='soft', threshold_sigma_mu…)` | Wavelet soft-thresholding denoising + reconstruction. |
| `reconstruct_interpolate` | função | `(timestamps, values, *, kind='cubic', out_times=None, n_out=512) -> Tuple[np.ndarray, np.…)` | Scipy interpolation-based reconstruction (gap filling / upsampling). |
| `reconstruct_model_based` | função | `(timestamps, values, *, model, sensor_coord, coord_names, field_name, field_index=0, out_…)` | Use a trained physics model to reconstruct the time series at a given spatial location by querying model(x, y, t). |

#### `pinneapple_systems.digital_twin.standards`

Digital-twin interoperability: describe a PINNeAPPle twin once and publish it in the open standards industry platforms read.

<small>`pinneapple_systems/digital_twin/standards.py` · 322 linhas · testes: 31</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TwinProperty` | classe | `(name, value, unit, kind, description, writable)` |  |
| `TwinModel` | classe | `(name, family, status, compatibility, violated, probable_cause)` | A model behind the twin and its physical validity on the current operating point. |
| `TwinDescription` | classe | `(id, name, asset_type, description, manufacturer, properties, models, relationships)` |  |
| `OpcUaTwinServer` | classe | `(desc, endpoint=…, namespace='urn:pinneapple:twins')` | Serve a twin over OPC UA (asyncua) in a background thread: one object per twin with a variable per property (and the validity status of each model). :meth:`update` writes new values; OPC UA clients read or subscribe. |
| `to_aas` | função | `(desc, as_json=True)` | Asset Administration Shell (metamodel v3) with submodels Identification, TechnicalData (parameters), OperationalData (telemetry and estimated state) and SimulationModels (models and physical validity), built with the Eclipse BaSyx Python SDK. Returns the JSON text (``as_json``) or the BaSyx object … |
| `to_dtdl` | função | `(desc, version=1) -> dict[str, Any]` | A DTDL v3 Interface: parameters and estimated state as Properties, measurements as Telemetry, model validity as Properties, links to other twins as Relationships. |
| `to_ngsi_ld` | função | `(desc, context=…, observed_at=None) -> dict[str, Any]` | An NGSI-LD entity: every property as a ``Property`` (unit as ``unitText``, measurements with ``observedAt``), model validity as properties, links as ``Relationship``s. |
| `to_ditto_thing` | função | `(desc, namespace='org.pinneapple') -> dict[str, Any]` | An Eclipse Ditto Thing: identification and parameters as attributes; telemetry, state and model validity as features. |

#### `pinneapple_systems.digital_twin.state`

System state representation for digital twins.

<small>`pinneapple_systems/digital_twin/state.py` · 137 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Observation` | classe | `(timestamp, sensor_id, values, coords, metadata)` | A single raw observation from a sensor stream. |
| `SystemState` | classe | `(timestamp, fields, coords, covariance, residuals, metadata, history)` | Current estimate of the physical system state. |

#### `pinneapple_systems.digital_twin.twin`

Core DigitalTwin class.

<small>`pinneapple_systems/digital_twin/twin.py` · 650 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DigitalTwinConfig` | classe | `(device, batch_size, use_amp, update_interval, obs_queue_maxsize, max_history, assimilati…)` | Configuration for the DigitalTwin. |
| `DigitalTwin` | classe | `(model, field_names, coord_names=None, *, config=None, assimilation_filter=None) -> None` | Digital twin that fuses a trained surrogate model with live sensor data. |
| `build_digital_twin` | função | `(model, field_names, *, coord_names=None, device='cpu', update_interval=0.5, anomaly_z_th…)` | Convenience factory for DigitalTwin. |

#### `pinneapple_systems.excitation`

Actuator excitation signals for system identification and surrogate training (numpy).

<small>`pinneapple_systems/excitation.py` · 152 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `shape_step` | função | `(tau, t0=0.3, **_)` | Unit step at ``tau = t0``. |
| `shape_ramp` | função | `(tau, t1=0.8, **_)` | Linear ramp from 0 at ``tau = 0`` to 1 at ``tau = t1``, then held. |
| `shape_sinusoid` | função | `(tau, freq=2.0, phase=0.0, **_)` | ``0.5 + 0.5 sin(2 pi freq tau + phase)`` (``freq`` in cycles per unit ``tau``). |
| `shape_piecewise_constant` | função | `(tau, rng=None, n_seg=5, **_)` | Random staircase: ``n_seg`` levels ~ U(0, 1) between sorted random edges in ``[0.05, 0.95]``. |
| `shape_prbs` | função | `(tau, rng=None, dwell=0.08, **_)` | Pseudo-random binary sequence in ``{0, 1}``. |
| `shape_chirp` | função | `(tau, f0=0.5, f1=6.0, **_)` | Linear chirp ``0.5 + 0.5 sin(phi)``, instantaneous frequency ``f0 + (f1 - f0) tau`` (cycles per unit tau). |
| `shape_sawtooth` | função | `(tau, freq=3.0, **_)` | Rising sawtooth ``(freq * tau) mod 1``. |
| `shape_smoothed_random_walk` | função | `(tau, rng=None, **_)` | Gaussian random walk smoothed by a 15-sample moving average, rescaled to ``[0, 1]``. |
| `shape_gaussian_bumps` | função | `(tau, rng=None, n_bumps=4, width=0.05, **_)` | Sum of ``n_bumps`` Gaussians (centres ~ U(0.1, 0.9), amplitudes ~ U(0.5, 1), std ``width``), peak-normalised. |
| `shape_am_two_tone` | função | `(tau, f_carrier=5.0, f_mod=0.7, **_)` | Amplitude modulation: ``(0.5 + 0.5 sin 2 pi f_carrier tau) (0.5 + 0.5 sin 2 pi f_mod tau)``. |
| `rate_limit` | função | `(series, tau_steps) -> np.ndarray` | First-order low-pass ``y[i] = y[i-1] + (x[i] - y[i-1]) / tau_steps`` (time constant ``tau_steps`` samples). |
| `actuator_series` | função | `(shape, n_steps, *, lo=0.0, hi=1.0, seed=0, rate_limit_steps=0, **kwargs) -> np.ndarray` | Sample one excitation on ``n_steps`` uniform points of ``tau`` in ``[0, 1]``, scaled to ``[lo, hi]``. |

#### `pinneapple_systems.operations`

Operations: run validated physical models and surrogates next to live plant data.

<small>`pinneapple_systems/operations/__init__.py` · 94 linhas · testes: 24</small>

#### `pinneapple_systems.operations.anomaly`

Physics anomaly detection on sensor time series: statistics, time-series models, conservation balances and physical-model residuals, fused into events with severities and verifiable explanations.

<small>`pinneapple_systems/operations/anomaly.py` · 573 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BalanceRule` | classe | `(name, accumulation, inflow, outflow, units, description)` | A conservation law over the columns: ``d(accumulation)/dt = inflow - outflow``. |
| `DetectorConfig` | classe | `(normal_until, sensors, methods, balances, constants, window, threshold, ar_order, cusum_…)` | What to run and how strict to be. ``normal_until`` (time, same units as ``t``) ends the normal window. |
| `Channel` | classe | `(method, name, variables, residual, centre, spread, units, description, score, threshold)` | One detector output: a residual series, its normal-window centre/spread and score = (r - centre) / spread. |
| `Event` | classe | `(start, end, i0, i1, severity, peak_ratio, methods, channels, variables, explanations, ev…)` |  |
| `DetectionResult` | classe | `(t, channels, flags, events, config, normal_samples)` |  |
| `LabelledInterval` | classe | `(kind, start, end, target)` | A labelled abnormal interval ``[start, end)`` read from data (same interface as a plant ``Fault``). |
| `tank_balance_rules` | função | `() -> list[BalanceRule]` | Mass and energy balances of the heated tank (:mod:`pinneapple_systems.operations.plant`). |
| `tank_constants` | função | `(p=None) -> dict[str, float]` |  |
| `windowed_balance` | função | `(t, acc, net_in, w) -> np.ndarray` | ``[acc(t_k) - acc(t_{k-w})] / (t_k - t_{k-w}) - (trapezoid mean of net_in over the same interval)``. |
| `severity_of` | função | `(ratio) -> str` | Severity from the peak score / threshold ratio: < 2 low, < 4 medium, < 8 high, else critical. |
| `detect` | função | `(data, config, model=None, model_states=(), model_inputs=()) -> DetectionResult` | Run the configured detectors on ``data`` (DataFrame or dict of columns with ``t``). |
| `benchmark_faults` | função | `(t0=7200.0) -> list[Fault]` | The faults of :func:`fault_benchmark`, after a fault-free first ``t0`` seconds (the calibration window). Magnitudes are small on purpose (a few percent of the nominal flows and powers). |
| `fault_benchmark` | função | `(seed=0, t0=7200.0, dt=10.0) -> tuple[dict[str, np.ndarray], list[Fault]]` | One labelled run: ``t0`` s of normal operation, then the six faults of :func:`benchmark_faults` separated by normal periods. Different ``seed`` = different operating profile and noise (a held-out run). |
| `intervals_from_labels` | função | `(t, labels) -> list[LabelledInterval]` | Contiguous runs of a label column: empty, ``0``, ``normal`` or ``false`` mean normal, anything else names the anomaly. Each run ends at the next sample's time. |
| `score_events` | função | `(result, faults, grace=600.0, evaluate_from=None) -> dict` | Point-wise and event-wise scores against the true fault intervals. |
| `run_benchmark` | função | `(seeds=(101, 102, 103), methods_sets=None, window=12, threshold=5.0) -> dict` | Score each detector family alone and all together on held-out labelled runs (one per seed). The calibration window of every run is its own fault-free first two hours; thresholds are never tuned on the faulty part. Returns per-method aggregate scores (sums of TP/FP/FN over runs) and per-run details. |

#### `pinneapple_systems.operations.contracts`

The operations tools speak ``pinneapple_data.contracts``: models travel as ``ModelRef`` (with their held-out validation metrics and validity ranges) and every twin, anomaly or control study ends as a ``StudyResult`` whose checks record what actually ran.

<small>`pinneapple_systems/operations/contracts.py` · 192 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `save_model` | função | `(model, path, name, *, app='', dataset=None, trust_decision='UNCHECKED') -> ModelRef` | Write ``model`` (a twin model or a :class:`ControlledOpInf`) as JSON at ``path`` and return its ``ModelRef`` (saved next to it as ``<path>.modelref.json``). Held-out metrics come from the model's own validation record; a model without one carries no metrics. |
| `load_model` | função | `(ref, root='.')` | Open the model a ``ModelRef`` points to (a local JSON file, checked against its SHA-256). |
| `check` | função | `(name, passed, value=None, threshold=None, evidence='', detail='', warn=False) -> CheckRe…)` | A ``CheckRecord``: ``passed=None`` means it did not run (NOT_RUN). |
| `new_id` | função | `(prefix) -> str` |  |
| `twin_study` | função | `(runtime, *, app='twin', model_ref=None, title='') -> StudyResult` | The twin's run as a ``StudyResult``: estimates table, alerts, validity checks and revalidations. |
| `anomaly_study` | função | `(result, *, app='anomaly', benchmark=None, title='') -> StudyResult` | A detection run (and optionally a benchmark) as a ``StudyResult``. |
| `control_study` | função | `(comparison, *, app='control', title='') -> StudyResult` | A controller comparison as a ``StudyResult`` (one row per controller). |

#### `pinneapple_systems.operations.control`

Physics-based control of the heated tank: model predictive control against PID and rule-based baselines, in closed loop on the same simulated plant, in simulation / recommendation mode only.

<small>`pinneapple_systems/operations/control.py` · 658 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ControlSpec` | classe | `(dt, horizon, setpoints, h_bounds, T_max, q_in_bounds, Q_bounds, dq_max, dQ_max, h_scale,…)` | Objectives, constraints and actuator limits. |
| `ControlScenario` | classe | `(duration, x0, T_in_steps, T_amb, faults, noise, seed)` | The plant side of a closed-loop study: duration, disturbances and faults (shared by every controller). |
| `PIDBaseline` | classe | `(spec, params=None, kp_h=0.02, ki_h=0.0001, kp_T=60000.0, ki_T=400.0)` | Level loop on the pump, temperature loop on the heater, both with feed-forward bias and anti-windup. |
| `RuleBasedBaseline` | classe | `(spec, band_h=0.05, band_T=0.5, q_low=0.004, q_high=0.01, Q_on=600000.0)` | Hysteresis rules: pump at a high rate below ``h_ref - band_h`` and a low rate above ``h_ref + band_h``; heater at ``Q_on`` below ``T_ref - band_T`` and off above ``T_ref + band_T``. |
| `LinearMPC` | classe | `()` | Successively linearised MPC: a QP over the horizon with SLSQP. |
| `NonlinearMPC` | classe | `(spec, params=None, model=None, backend='auto')` | Nonlinear MPC on the tank balances (CasADi/IPOPT if available) or on any ``model`` (SciPy). |
| `make_estimator` | função | `(spec, params=None, seed=0) -> TwinRuntime` | The state estimator the MPCs read: an EnKF twin of the tank with unaccounted outflow and heater efficiency. |
| `run_closed_loop` | função | `(controller, spec, scenario, params=None) -> dict` | Simulate ``controller`` on ``scenario``. Returns the trajectories, the controller's plans (every 10th step, MPC only) and :func:`performance` metrics. |
| `performance` | função | `(tr, spec) -> dict` | Integrated absolute tracking errors (true states), constraint violations, energy, actuator travel and solve time of a closed-loop trajectory. |
| `compare_controllers` | função | `(spec=None, scenario=None, params=None, controllers=…, surrogate=None) -> dict` | Run the named controllers on the same scenario. Names: ``pid``, ``rules``, ``lmpc``, ``nmpc`` and ``smpc`` (nonlinear MPC on ``surrogate``, a model with ``step(x, v)``). |
| `recommend` | função | `(estimate, reading, spec=None, params=None, t=0.0, method='nmpc') -> dict` | Proposed next actions for a person to review. ``estimate``: state (``h``, ``T`` and optionally ``q_leak``, ``eta``); ``reading``: current ``q_in``, ``Q``, ``T_in``, ``T_amb``. Nothing is sent anywhere. |

#### `pinneapple_systems.operations.exchanger`

A counterflow water/water heat exchanger with fouling — the plant a twin is built for.

<small>`pinneapple_systems/operations/exchanger.py` · 158 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OperatingPoint` | classe | `(t_hi, t_ci, m_h, m_c)` | Where the exchanger is being run. |
| `PlantConfig` | classe | `(ua_clean, flow_sensitivity, tau_fouling_s, rf_final)` | The as-built exchanger, including the model error the twin has to live with. |
| `exchanger_model` | função | `(X, theta, *, ua_clean) -> dict[str, np.ndarray]` | ``model(X, theta)`` for the twin: outlet temperatures from the operating point. |
| `plant_outlets` | função | `(op, rf, cfg) -> tuple[float, float]` | The real exchanger: the model plus a flow-dependent UA the model does not know. |
| `fouling_at` | função | `(t_s, cfg) -> float` | Asymptotic fouling build-up: fast at first, then a plateau. |
| `generate_history` | função | `(hours=72.0, sampling_s=300.0, *, seed=7, cfg=None, op=None, t_start=1767225600.0) -> lis…)` | A plant history with fouling, flow wobble and instrument noise. |

#### `pinneapple_systems.operations.live`

Run a :class:`~pinneapple_systems.operations.twin.TwinRuntime` as a service: on a live stream (MQTT, REST pushes, an experimental OPC UA poller) or over a historical table.

<small>`pinneapple_systems/operations/live.py` · 251 linhas · testes: 11 · palavras-chave no código (heurística; o texto acima diz o status real): experimental</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LiveTwin` | classe | `(twin, *, source='simulated', topic=None, broker_host='127.0.0.1', broker_port=0, embedde…)` | One asset's live twin. ``source``: ``"simulated"`` (built-in plant over MQTT), ``"mqtt"`` (readings from an external publisher on ``topic``), ``"rest"`` (only :meth:`push`) or ``"opcua"`` (experimental). |
| `make_twin` | função | `(model, *, asset='tank-1', store=None, rules=None, sigma_obs=None, sigma_process=None, se…)` | A twin of the heated tank with the noise levels of its sensors (override for another asset). |
| `replay` | função | `(table, model, *, asset='tank-1', store=None, rules=None, forecast_steps=60, revalidate=T…)` | Run a twin over a recorded table (DataFrame or dict of columns with ``t`` and the model's inputs and observed states). Returns the twin and a JSON-ready result with the history, alerts, final forecast and, when ``revalidate``, an open-loop re-score of the model on the last part of the record. |
| `demo_record` | função | `(duration=7200.0, dt=10.0, seed=4, faults=None) -> dict` | A recorded run of the simulated tank with a 0.5 L/s leak over its second half (the twin app's offline example). |
| `default_model` | função | `(kind='lumped', dt=10.0)` |  |

#### `pinneapple_systems.operations.plant`

Heated buffer tank: a lumped (0-D) plant with a mass and an energy balance, its sensors and its faults.

<small>`pinneapple_systems/operations/plant.py` · 290 linhas · testes: 25</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TankParams` | classe | `(area, k_out, ua, rho, cp, h_min, h_max, q_in_max, q_max_heater)` | Physical parameters of the heated tank (SI units, temperatures in degC). |
| `Fault` | classe | `(kind, start, end, magnitude, target)` | A fault active on ``[start, end)`` seconds. ``target`` names the sensor for sensor faults. |
| `Scenario` | classe | `(duration, dt, x0, T_in, T_amb, q_in, Q)` | Operating scenario: how the inputs and disturbances evolve. Callables take the time [s] and return a value. |
| `SensorNoise` | classe | `(sigma)` | Standard deviation of each sensor's white measurement noise (SI units). |
| `TankSimulator` | classe | `(params=None, x0=(1.2, 25.0), faults=(), noise=None, seed=0)` | Step-by-step simulator of the heated tank with sensors and faults. Used open loop (scenario inputs), closed loop (a controller sets the inputs each step) or as the live source of a digital twin demo. |
| `tank_rhs` | função | `(x, u, d, p, eta=1.0, valve=1.0, q_leak=0.0) -> np.ndarray` | Time derivative ``[dh/dt, dT/dt]`` at state ``x = [h, T]``, inputs ``u = [q_in, Q]`` and disturbances ``d = [T_in, T_amb]``. Works on arrays of states with the leading axis being the state component. |
| `tank_step` | função | `(x, u, d, p, dt, substeps=4, **fault_kw) -> np.ndarray` | Advance the tank ``dt`` seconds with classical RK4 (inputs held constant over the step). |
| `steady_state` | função | `(p, q_in, Q, T_in, T_amb) -> np.ndarray` | Equilibrium ``[h, T]`` for constant inputs (closed form). |
| `inputs_for_steady_state` | função | `(p, h, T, T_in, T_amb) -> np.ndarray` | Inputs ``[q_in, Q]`` that hold the tank at ``[h, T]`` (closed form). |
| `measure` | função | `(truth, t, faults, noise, rng, frozen) -> dict[str, float]` | Sensor readings for one time step: true value + noise, then sensor faults. |
| `simulate` | função | `(scenario, params=None, faults=(), noise=None, seed=0) -> dict[str, np.ndarray]` | Open-loop run of ``scenario`` (its ``q_in`` and ``Q`` must be set). Returns columns: ``t``, the measured sensors, ``true_<state>``, ``true_q_leak``, ``true_eta`` and ``fault`` (name of the active fault or ""). |
| `demo_scenario` | função | `(duration=3600.0, dt=10.0, seed=0) -> Scenario` | A realistic day-shift profile: pump flow steps between production rates, heater follows a schedule, inlet water temperature wanders slowly. Deterministic for a given ``seed``. |
| `mass_balance_residual` | função | `(t, h, q_in, q_out, area) -> np.ndarray` | ``A dh/dt - (q_in - q_out)`` [m3/s] from measured signals (central differences in time, inputs averaged over each interval). Zero for a fault-free tank up to noise; a leak makes it equal to minus the leak flow. |
| `energy_balance_residual` | função | `(t, h, T, q_in, q_out, Q, T_in, T_amb, p) -> np.ndarray` | Energy balance in conservative form [W]:: |
| `with_params` | função | `(p, **changes) -> TankParams` |  |

#### `pinneapple_systems.operations.rom`

Operator inference with inputs: a discrete-time polynomial reduced model fitted by ridge regression.

<small>`pinneapple_systems/operations/rom.py` · 178 linhas · testes: 11</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ControlledOpInf` | classe | `(state_names, input_names, dt, W, x_mean, x_scale, v_mean, v_scale, quadratic, bilinear, …)` | A fitted model; build it with :func:`fit_controlled_opinf` or :meth:`from_dict`. |
| `fit_controlled_opinf` | função | `(trajectories, state_names, input_names, dt, *, holdout=1, ridge=1e-06, quadratic=True, b…)` | Fit on all but the last ``holdout`` trajectories and validate on those. |
| `residual_sigma` | função | `(model, trajectories) -> dict[str, float]` | One-step residual standard deviation per state on ``trajectories`` (use held-out ones): the process noise a filter running this model should assume. |

#### `pinneapple_systems.operations.series`

The days 13-31 series end to end on one asset: Campaign Planner -> (runs) -> Surrogate Studio -> Uncertainty -> Optimizer -> Digital Twin, every hand-over a ``pinneapple_data.contracts`` file.

<small>`pinneapple_systems/operations/series.py` · 339 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `operating_space` | função | `() -> ParameterSpace` |  |
| `app_available` | função | `(pattern, apps_dir) -> str \| None` | Name of the ``apps/`` directory of a series app (matched by its whole name), if this checkout has it. |
| `run_series_flow` | função | `(workdir, *, seed=0, n_runs=12, steps_per_run=12, hold=300.0, dt=20.0, T_min=28.0, q_min=…)` | Run the chain and write every contract under ``workdir``. Returns a JSON-ready summary. |

#### `pinneapple_systems.operations.twin`

Digital twin runtime: run a validated model next to live data, estimate its state, forecast, alert and watch the model's own validity.

<small>`pinneapple_systems/operations/twin.py` · 602 linhas · testes: 25</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LumpedTankModel` | classe | `(params=None, dt=10.0, estimate=('q_leak', 'eta'))` | The heated tank's mass and energy balances as a twin model, optionally with parameters in the state. |
| `SurrogateModel` | classe | `(rom)` | A :class:`ControlledOpInf` reduced model as a twin model (no parameters in the state). |
| `AlertRules` | classe | `(limits, param_limits, param_sigmas, nis_probability, nis_consecutive, forecast_steps, dr…)` | Alert thresholds. ``limits`` and ``param_limits`` map a state name to ``(low, high)`` (``None`` = open). |
| `Alert` | classe | `(t, kind, severity, message, data, cleared)` |  |
| `TwinStore` | classe | `(url=':memory:')` | History of readings, estimates, alerts and revalidations. |
| `TwinRuntime` | classe | `(model, obs_names, sigma_obs, sigma_process, *, x0=None, sigma0=None, n_ens=48, inflation…)` | Assimilate readings one at a time into ``model`` and keep the twin's state, alerts and validity. |
| `model_from_dict` | função | `(d)` | Rebuild a twin model from its JSON form (``kind`` = ``lumped_tank`` or ``controlled_opinf``). |

#### `pinneapple_systems.process_components`

pinneapple_systems.process_components — physical unit-operation models for process/plant digital twins: real-gas thermodynamic properties, turbomachinery (polytropic compression/expansion + nondimensional similarity maps), control valves, heat exchangers, and 1D pipe networks.

<small>`pinneapple_systems/process_components/__init__.py` · 272 linhas · testes: 58</small>

#### `pinneapple_systems.process_components.beam_nonlinear_fem`

pinneapple_systems.process_components.beam_nonlinear_fem -- static, geometrically nonlinear (Von Karman) Euler-Bernoulli beam FEM via load-stepped Newton-Raphson.

<small>`pinneapple_systems/process_components/beam_nonlinear_fem.py` · 172 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `StaticNonlinearBeamResult` | classe | `(z, u_m, w_m, theta_rad, bending_moment, bending_stress, load_step_deflections, iteration…)` |  |
| `solve_nonlinear_beam_static` | função | `(boundary, L_m, A_m2, I_m4, E_Pa, fiber_distance_m, axial_dist_load_N_per_m=0.0, transver…)` | Load-stepped Newton-Raphson static solve of the geometrically nonlinear (Von Karman) Euler-Bernoulli beam, at the static limit of `NonlinearBeamNewmarkFEM` (see module docstring). `fiber_distance_m` is the distance from the neutral axis to the outer fiber used for bending-stress recovery (height/2 … |

#### `pinneapple_systems.process_components.beam_statics`

pinneapple_systems.process_components.beam_statics -- closed-form static, linear-elastic Euler-Bernoulli beam solutions for the standard cantilever and simply-supported load cases.

<small>`pinneapple_systems/process_components/beam_statics.py` · 196 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BeamResult` | classe | `(x, deflection, slope, bending_moment, shear_force)` |  |
| `rectangular_section_properties` | função | `(width_m, height_m) -> tuple[float, float]` | Returns (I, Q) for a rectangular cross-section: second moment of area I = b*h^3/12, first moment of area at the neutral axis Q = b*h^2/8 (used for shear stress). |
| `von_mises_stress_rectangular_section` | função | `(bending_moment, shear_force, I, Q, width_m, height_m) -> np.ndarray` |  |
| `solve_beam` | função | `(case, x, L, E, I, q0, load_extent=None) -> BeamResult` | Dispatches to the closed-form solution for `case`. `load_extent` (the partial-load length, measured from the load's own reference end per case) is required for the two partial-uniform-load cases and ignored otherwise. |

#### `pinneapple_systems.process_components.control_valve`

pinneapple_systems.process_components.control_valve — compressible and incompressible control-valve flow sizing per IEC 60534-2-1 (harmonized with ANSI/ISA-75.01.01), plus actuator response and closed-seat leakage.

<small>`pinneapple_systems/process_components/control_valve.py` · 115 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ValveSpec` | classe | `(name, Cv_max_us, xT, characteristic, equal_percentage_R, actuator_time_constant_s, leaka…)` |  |
| `ValveFlowResult` | classe | `(mass_flow_kg_s, choked, x, Y, Cv_effective_us)` |  |
| `installed_cv` | função | `(travel_fraction, spec) -> float` | Cv at a given travel fraction (0=closed, 1=full open), per the inherent characteristic -- excludes leakage, see `effective_cv`. |
| `effective_cv` | função | `(travel_fraction, spec, leakage_cv_us_override=None) -> float` | Cv actually passing flow, including the closed-seat leakage floor -- pass `leakage_cv_us_override` to represent a TRUE (e.g. degraded) leakage different from the spec's nameplate value. |
| `compressible_mass_flow` | função | `(Cv_us, P1_Pa, P2_Pa, rho1_kg_m3, k_isentropic, xT) -> ValveFlowResult` |  |
| `incompressible_mass_flow` | função | `(Cv_us, P1_Pa, P2_Pa, rho_kg_m3) -> float` | Liquid-service sizing (no expansion factor / choking model beyond simple flashing checks are out of scope here) -- w = Cv_SI * sqrt(dP * rho). |
| `actuator_response_rhs` | função | `(travel_actual, travel_command, spec) -> float` | d(travel_actual)/dt for the first-order actuator lag. |

#### `pinneapple_systems.process_components.curved_path_geometry`

pinneapple_systems.process_components.curved_path_geometry -- the "build-and-hold" curved-conduit path generator: a straight (vertical) section, a constant-curvature build section transitioning to a target inclination, and a straight tangent/hold section at that inclination.

<small>`pinneapple_systems/process_components/curved_path_geometry.py` · 113 linhas · testes: 7 · palavras-chave no código (heurística; o texto acima diz o status real): placeholder/stub citado no docstring</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CurvedPathProfile` | classe | `(s_m, inclination_deg, tvd_m, hd_m, n_bins, vertical_length_m, total_length_m, bin_width_…)` |  |
| `inclination_at_depth` | função | `(s_m, vertical_length_m, build_length_m, final_inclination_deg) -> np.ndarray` | Inclination [deg, from horizontal] at each measured depth `s_m`: 90 up to `vertical_length_m`, linear (constant curvature) through the following `build_length_m`, held at `final_inclination_deg` after. |
| `build_and_hold_profile` | função | `(total_length_m, build_length_m, final_inclination_deg, hold_length_m, bin_width_m) -> Cu…)` | Discretize the build-and-hold path into `bin_width_m`-wide bins (midpoint-evaluated) and integrate TVD/HD along it. `vertical_length_m = total_length_m - build_length_m - hold_length_m` is derived, not an independent input, so the three lengths are always consistent. |
| `circular_arc_tvd_hd` | função | `(s_within_build_m, build_length_m, final_inclination_deg) -> tuple` | Closed-form TVD/HD accumulated over the build section only, for a circular arc of constant curvature `kappa = delta_theta_rad / build_length_m` (delta_theta = 90 - final_inclination_deg, in rad): `tvd(s) = sin(kappa*s)/kappa`, `hd(s) = (1 - cos(kappa*s))/kappa`. Independent closed-form cross-check … |

#### `pinneapple_systems.process_components.explicit_equation_system`

pinneapple_systems.process_components.explicit_equation_system -- a generic, sandboxed "define your model as named equations" engine: given a set of scalar/array symbols (inputs, parameters, computed intermediates, and outputs) and a set of equations relating them, safely parse and evaluate the …

<small>`pinneapple_systems/process_components/explicit_equation_system.py` · 393 linhas · testes: 171</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ExplicitEquationError` | classe | `()` |  |
| `Definition` | classe | `(name, lhs, rhs, rhs_symbols)` |  |
| `AnalysisResult` | classe | `(valid, errors, evaluation_order)` |  |
| `ParameterSpec` | classe | `(symbol, value, fit, min, max)` |  |
| `CalibrationResult` | classe | `(fitted_parameters, start, bounds, n_evaluations, final_scaled_residual_ssr, termination_…)` |  |
| `safe_eval` | função | `(expr, env) -> Any` |  |
| `build_definitions` | função | `(inputs, parameters, intermediates, outputs, equations) -> tuple[list[Definition], list[s…)` | Validates and builds Definitions from raw {"name","equation"} dicts. Returns (definitions, errors) -- errors accumulate rather than failing on the first one, so a caller can report everything wrong with a submitted model at once. |
| `evaluation_order` | função | `(definitions) -> list[Definition]` | Topological sort by LHS symbol, DFS with cycle detection. |
| `evaluate` | função | `(input_values, parameter_values, order) -> dict[str, Any]` | Evaluates every definition in the given (topological) order, returning the full symbol -> value environment (leaves + every computed symbol). |
| `analyze` | função | `(inputs, parameters, intermediates, outputs, equations) -> AnalysisResult` | Pure validation/dry-run -- builds and orders the system, reports errors, does NOT evaluate any numbers. |
| `calibrate` | função | `(input_columns, parameters, order, measured_rows, target_symbols) -> CalibrationResult` | Bounded nonlinear least-squares calibration (Trust Region Reflective) of every parameter with `fit=True` against `measured_rows`'s target-symbol columns, re-evaluating the FULL equation system (not a linearization) at every trial parameter vector -- see module docstring's NUMERICAL METHOD. |

#### `pinneapple_systems.process_components.fatigue_analysis`

pinneapple_systems.process_components.fatigue_analysis -- S-N (Wohler) curve fatigue life, Goodman mean-stress correction, and Miner's rule cumulative damage accumulation for cyclic structural loading.

<small>`pinneapple_systems/process_components/fatigue_analysis.py` · 108 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MinerDamageResult` | classe | `(N_f, damage_this_period, cumulative_damage, remaining_life_fraction)` |  |
| `sn_curve_cycles_to_failure` | função | `(sigma_a, A, m) -> ArrayLike` | N_f = 10**(A - m*log10(sigma_a)) for a fully-reversed amplitude `sigma_a` (floored at 1.0 to keep log10 finite). |
| `goodman_equivalent_amplitude` | função | `(sigma_alt, sigma_mean, S_ut) -> ArrayLike` | Equivalent fully-reversed amplitude per the modified-Goodman line (see module docstring): sigma_alt / (1 - sigma_mean/S_ut). A mean stress approaching (or exceeding) S_ut drives this to infinity/ negative, correctly signaling the Goodman boundary has already been crossed by static overload alone. |
| `goodman_safety_ratio` | função | `(sigma_alt, sigma_mean, S_e, S_ut) -> ArrayLike` | sigma_alt/S_e + sigma_mean/S_ut -- the Goodman-line safety ratio itself (< 1 inside the safe region, independent of any S-N life estimate); `S_e` is the endurance limit / target-life fully-reversed strength the diagram is drawn against. |
| `miners_rule_damage` | função | `(sigma_alt, sigma_mean, n_cycles, A_sn, m_sn, S_ut, prior_damage_fraction=0.0) -> MinerDa…)` | Goodman-corrected S-N life `N_f`, this-period damage `n_cycles/N_f`, and Miner's-rule cumulative damage (clipped to [0, 1]) combining it with `prior_damage_fraction`. |

#### `pinneapple_systems.process_components.heat_exchanger`

pinneapple_systems.process_components.heat_exchanger -- effectiveness-NTU heat exchanger model (counter-flow and parallel-flow closed-form effectiveness), plus a lumped-capacitance transient response.

<small>`pinneapple_systems/process_components/heat_exchanger.py` · 113 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HeatExchangerSpec` | classe | `(name, UA_design_W_K, design_dP_Pa, design_m_dot_hot_kg_s, design_rho_hot_kg_m3, thermal_…)` |  |
| `HeatExchangerResult` | classe | `(Q_W, T_hot_out_K, T_cold_out_K, effectiveness, NTU, dP_Pa, UA_effective_W_K)` |  |
| `steady_state` | função | `(spec, m_dot_hot_kg_s, T_hot_in_K, cp_hot_J_kgK, T_cold_in_K, rho_hot_kg_m3, UA_multiplie…)` |  |
| `transient_rhs` | função | `(T_hot_out_current_K, spec, m_dot_hot_kg_s, T_hot_in_K, cp_hot_J_kgK, T_cold_in_K, rho_ho…)` | d(T_hot_out)/dt -- relaxes toward the instantaneous steady-state effectiveness-NTU outlet temperature with a time constant set by thermal_mass_J_K / (m_dot * cp). |

#### `pinneapple_systems.process_components.non_newtonian_pipe_flow`

pinneapple_systems.process_components.non_newtonian_pipe_flow -- Herschel-Bulkley rheology, the generalized (Metzner-Reed) Reynolds number and friction factor for power-law/yield-stress fluids, and steady-state 1D pressure-drop integration along an arbitrarily inclined conduit (straight pipe, …

<small>`pinneapple_systems/process_components/non_newtonian_pipe_flow.py` · 155 linhas · testes: 10 · palavras-chave no código (heurística; o texto acima diz o status real): placeholder/stub citado no docstring</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PressureProfile` | classe | `(s_m, tvd_m, P_Pa, rho_eq_kg_m3, rho_kg_m3, friction_factor, reynolds_number)` |  |
| `herschel_bulkley_stress` | função | `(tau_y, K, n, gamma_dot) -> ArrayLike` | tau = tau_y + K*gamma_dot**n. `gamma_dot` is floored at 1e-9 s^-1. |
| `effective_viscosity` | função | `(tau_y, K, n, gamma_dot) -> ArrayLike` | mu_eff = tau/gamma_dot = tau_y/gamma_dot + K*gamma_dot**(n-1). |
| `generalized_reynolds_number` | função | `(rho, v, D, K, n) -> ArrayLike` | Re_gen = rho*\|v\|**(2-n)*D**n / (K*8**(n-1)); laminar if Re_gen < 2100. |
| `metzner_reed_friction_factor` | função | `(Re_gen) -> ArrayLike` | f = 16/Re_gen (laminar, Re_gen < 2100) or 0.0791*Re_gen**-0.25 (turbulent, Blasius/Metzner-Reed). |
| `pressure_gradient` | função | `(rho, v, D_h, f, inclination_rad=np.pi / 2, g=9.81) -> ArrayLike` | dP/ds = rho*g*sin(inclination) + f*rho*v**2/(2*D_h). `inclination_rad` is measured from horizontal (pi/2 = vertical, the default). |
| `integrate_pressure_profile` | função | `(length_m, v_m_s, D_h_m, K_Pa_sn, n_flow_index, P_inlet_Pa, rho_profile=1000.0, inclinati…)` | Steady-state 1D pressure integration along path length `s` in [0, length_m], by explicit forward Euler on `dP/ds` from `pressure_gradient`. `rho_profile`/`inclination_deg_profile` may be a constant or a callable `f(s_m) -> value`. `tvd_profile(s_m)` supplies true vertical depth for the … |

#### `pinneapple_systems.process_components.pipe_network_1d`

pinneapple_systems.process_components.pipe_network_1d — 1D real-gas pipe flow: a rapid steady-state scenario mode and a transient finite-volume mode, covering a single pipe segment.

<small>`pinneapple_systems/process_components/pipe_network_1d.py` · 290 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PipeSpec` | classe | `(name, length_m, diameter_m, roughness_m, elevation_change_m, ambient_temperature_K, wall…)` |  |
| `SteadyProfilePoint` | classe | `(x_m, P_Pa, T_K, rho_kg_m3, G_kg_m2s, mach)` |  |
| `PipeState` | classe | `(P_Pa, T_K)` |  |
| `TransientPipe` | classe | `(spec, gas)` | Finite-volume transient solver -- see module docstring's DISCRETIZATION NOTE for why `initialize_from_steady_state` averages node values into cell-center values rather than reusing them directly. |
| `colebrook_white_f` | função | `(Re, relative_roughness) -> float` | Darcy friction factor -- laminar closed form below Re=2300, else Colebrook-White, solved via Newton iteration on 1/sqrt(f). |
| `face_reynolds_number` | função | `(G_kg_m2s, D_m, viscosity_Pa_s) -> float` |  |
| `rapid_steady_state_profile` | função | `(spec, gas, m_dot_kg_s, P_in_Pa, T_in_K) -> list[SteadyProfilePoint]` | Marches downstream cell-by-cell from the known inlet, solving each face's momentum closure explicitly (no time integration). |

#### `pinneapple_systems.process_components.pipe_stress_mechanics`

pinneapple_systems.process_components.pipe_stress_mechanics -- combined-load stress analysis for a slender tube (axial + torsional + bending + internal/external pressure), Euler column buckling, and constrained-rod (Paslay-Dawson) buckling of a slender member confined within a surrounding …

<small>`pinneapple_systems/process_components/pipe_stress_mechanics.py` · 209 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ConstrainedRodBucklingResult` | classe | `(F_critical_sinusoidal, F_critical_helical, mode)` |  |
| `RotatingBendingCycleResult` | classe | `(theta_rad, sigma_von_mises, sigma_alternating, sigma_mean)` |  |
| `lame_hoop_stress_outer` | função | `(p_internal, r_outer, r_inner) -> ArrayLike` | sigma_theta at r=r_outer: 2*p_i*ri**2 / (ro**2-ri**2) (the general Lame hoop-stress formula's `(1 + ro**2/r**2)` factor evaluates to exactly 2 at r=r_outer -- see module docstring). |
| `lame_hoop_stress_inner` | função | `(p_internal, r_outer, r_inner) -> ArrayLike` | sigma_theta at r=r_inner (the maximum): p_i*(ri**2+ro**2) / (ro**2-ri**2). |
| `torsional_shear_stress` | função | `(torque, r_outer, J) -> ArrayLike` | tau = T*c/J, c = r_outer (maximum shear, at the outer fiber). |
| `bending_stress_from_curvature` | função | `(curvature, E, r_outer) -> ArrayLike` | sigma_bend = E*kappa*r_outer (outer-fiber bending stress for a circular cross-section bent to curvature `kappa` [1/length]). |
| `von_mises_triaxial` | função | `(sigma_axial, sigma_hoop, sigma_radial=0.0, tau=0.0) -> ArrayLike` | Full triaxial Von Mises equivalent stress for three orthogonal principal-direction normal stresses plus a shear stress acting in the (axial, hoop) plane. See module docstring for the derivation and its biaxial (`sigma_radial=0`) reduction. |
| `euler_critical_buckling_load` | função | `(E, I, length, end_condition_factor=1.0) -> float` | P_cr = pi**2*E*I / (K*L)**2. `end_condition_factor` K: 0.5 fixed- fixed, 0.7 fixed-pinned, 1.0 pinned-pinned (default), 2.0 fixed-free (cantilever). |
| `constrained_rod_buckling_load` | função | `(E, I, lateral_load_per_length, radial_clearance) -> ConstrainedRodBucklingResult` | Paslay-Dawson sinusoidal/helical critical compressive loads for a slender rod confined within a surrounding cylindrical clearance (see module docstring). `lateral_load_per_length` is the distributed load normal to the rod axis (e.g. buoyed self-weight component); returns "not_applicable" mode with … |
| `classify_buckling_mode` | função | `(compressive_force, result) -> ArrayLike` | 0 = straight, 1 = sinusoidal, 2 = helical, per Paslay-Dawson critical loads. |
| `beam_column_moment_amplification_factor` | função | `(compressive_force, E, I, length, end_condition_factor=1.0, max_ratio=0.98) -> ArrayLike` | AF = 1 / (1 - P/P_cr), the standard beam-column second-order (P-Delta) moment-amplification approximation (see module docstring). `P/P_cr` is clipped to `max_ratio` before dividing, since the approximation (and physically, the member) diverges as the axial load approaches the critical load -- the … |
| `rotating_bending_stress_cycle` | função | `(sigma_axial, sigma_bending_amplitude, sigma_hoop=0.0, tau=0.0, n_angles=36) -> RotatingB…)` | Von Mises stress envelope over one full rotation of a shaft/rod under constant axial + hoop + torsional stress plus a bending stress that varies as `sigma_bending_amplitude*cos(theta)` (see module docstring), and the alternating/mean stress a fatigue analysis needs from that envelope. |

#### `pinneapple_systems.process_components.polytropic_path`

pinneapple_systems.process_components.polytropic_path — real-gas polytropic compression/expansion path integration for turbomachinery, consistent with ASME PTC 10 ("Performance Test Code on Compressors and Exhausters").

<small>`pinneapple_systems/process_components/polytropic_path.py` · 196 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PolytropicPathResult` | classe | `(inlet, outlet, polytropic_work_J_kg, polytropic_efficiency, isentropic_work_J_kg, isentr…)` |  |
| `solve_path_from_pressure_ratio` | função | `(inlet, gas, P2_Pa, polytropic_efficiency) -> PolytropicPathResult` | Integrate equation (1) from the inlet state to a known discharge/ exit pressure P2 -- works for compression (P2 > P1) or expansion (P2 < P1). |
| `solve_path_from_work` | função | `(inlet, gas, polytropic_work_J_kg, polytropic_efficiency, *, max_pressure_ratio=6.0) -> P…)` | Integrate equation (1) until the accumulated work integral reaches a target (a map's polytropic head, in energy-per-mass units) -- solves for the discharge/exit pressure self-consistently. Positive `polytropic_work_J_kg` compresses; negative expands. |

#### `pinneapple_systems.process_components.reaction_kinetics`

pinneapple_systems.process_components.reaction_kinetics -- a generic multi-species mass-action reaction-network engine, plus a generic 1D advection-dispersion-reaction transport solver built on top of it.

<small>`pinneapple_systems/process_components/reaction_kinetics.py` · 257 linhas · testes: 164</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Reaction` | classe | `(name, rate_fn, stoichiometry)` |  |
| `ReactionNetwork` | classe | `(species, reactions)` |  |
| `IntegrationResult` | classe | `(t, C, success)` |  |
| `AdvectionDispersionReactionSolver` | classe | `(network, n_grid, length_m, velocity_m_s, dispersion_m2_s)` | 1D transport (equation 3) on a periodic grid, driven by a ReactionNetwork for R_s(C) -- see module docstring's TRANSPORT section for the discretization and the periodic-only scope note. |
| `mass_action_rate` | função | `(k, orders) -> RateFn` | Builds equation (2): rho(C) = k * prod(C[s]^order[s]). `k` may be a plain float or a callable(T_K=.., pH=.., C=..) -> float for a temperature/pH/catalyst-dependent rate constant (see `arrhenius_rate_constant`, `linear_combination_rate_constant`). |
| `arrhenius_rate_constant` | função | `(A, Ea_over_R) -> Callable[..., float]` | k(T_K) = A * exp(-Ea_over_R / T_K). Raises if called without a temperature (a temperature-dependent rate constant used where no T is supplied is a caller bug, not something to silently default). |
| `quadratic_in_T` | função | `(a, b, c) -> Callable[[float], float]` | f(T_K) = a*T_K^2 + b*T_K + c -- e.g. for a polynomial-fit pKa(T) or equilibrium constant, when a literature source gives that form instead of an Arrhenius one. |
| `linear_combination_rate_constant` | função | `(terms) -> Callable[..., float]` | k_eff(C) = sum(coefficient_i * C[species_i]) -- for a rate constant that is itself a linear combination of catalyst-species concentrations (e.g. an acid/base/buffer-catalyzed reaction). |
| `acid_fraction` | função | `(pH, pKa) -> float` | Henderson-Hasselbalch: fraction of a monoprotic acid-base pair present in the PROTONATED (acid) form. |
| `base_fraction` | função | `(pH, pKa) -> float` | Fraction present in the DEPROTONATED (base/conjugate-base) form -- exactly 1 - acid_fraction(pH, pKa). |
| `diprotic_fractions` | função | `(pH, pKa1, pKa2) -> tuple[float, float, float]` | Bjerrum fractions (alpha_H2A, alpha_HA, alpha_A) for a diprotic acid-base system (H2A / HA- / A2-), generalizing the carbonate system (H2CO3/HCO3-/CO3^2-) to any two-pKa diprotic pair. |
| `integrate_network` | função | `(network, C0, t_eval, T_K=None, pH=None, *, method='Radau', rtol=1e-09, atol=1e-14) -> In…)` | Integrates dC/dt = network.rhs(C) from C0 over t_eval, clamping concentrations non-negative before every RHS evaluation and raising RuntimeError (never silently returning a failed/partial solve) if the integrator does not report success. |

#### `pinneapple_systems.process_components.real_gas_eos`

pinneapple_systems.process_components.real_gas_eos — multi-component real-gas mixture properties via a Helmholtz-energy equation of state.

<small>`pinneapple_systems/process_components/real_gas_eos.py` · 259 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OutOfEnvelopeError` | classe | `()` | Raised when a requested state falls outside a caller-declared validity envelope. |
| `GasComposition` | classe | `(components, mole_fractions)` | A fixed-composition mixture, mole fractions summing to 1. |
| `ValidityEnvelope` | classe | `(P_min_Pa, P_max_Pa, T_min_K, T_max_K)` | Optional, caller-declared (P, T) bounds -- states outside it raise OutOfEnvelopeError instead of silently extrapolating. Pass None to `state_from_*`'s `envelope` argument to skip this check entirely. |
| `GasState` | classe | `(P_Pa, T_K, rho_kg_m3, Z, h_J_kg, s_J_kgK, cp_J_kgK, cv_J_kgK, k_isentropic, speed_of_sou…)` | A fully-resolved thermodynamic state point, SI units throughout. |
| `StandardConditions` | classe | `(P_Pa, T_K)` |  |
| `state_from_PT` | função | `(gas, P_Pa, T_K, *, backend='HEOS', specify_phase='gas', envelope=None) -> GasState` |  |
| `state_from_Ph` | função | `(gas, P_Pa, h_J_kg, *, backend='HEOS', specify_phase='gas', envelope=None) -> GasState` | Pressure + specific enthalpy -> full state (temperature recovered by EOS flash). Used wherever an energy balance produces h directly. |
| `state_from_Ps` | função | `(gas, P_Pa, s_J_kgK, *, backend='HEOS', specify_phase='gas', envelope=None) -> GasState` | Pressure + specific entropy -> full state (an isentropic endpoint). |
| `standard_volumetric_flow_to_mass_flow` | função | `(q_std_m3_s, gas, std) -> float` |  |
| `mass_flow_to_standard_volumetric_flow` | função | `(m_dot_kg_s, gas, std) -> float` |  |
| `central_difference` | função | `(gas, P_Pa, T_K, attr, wrt, *, rel_step=0.0001) -> float` | d(attr)/d(wrt) via central differences -- for use OUTSIDE an autograd graph (CoolProp's compiled solve is not autodiff-transparent; any training loop that needs a gradient through this EOS should call this instead of tracing through `state_from_*` directly). |

#### `pinneapple_systems.process_components.similarity_map`

pinneapple_systems.process_components.similarity_map — nondimensional turbomachinery performance-map utilities (flow/head coefficients, tip Mach and machine Reynolds corrections, surge/choke margins).

<small>`pinneapple_systems/process_components/similarity_map.py` · 194 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MapCoefficients` | classe | `(name, D2_m, n_stages, phi_ref_bep, psi_ref_bep, psi_shape_a, psi0, eta_p_max, eta_p_shap…)` |  |
| `MapEvaluation` | classe | `(phi, psi, eta_p, Mu2, Re, surge_margin_frac, choke_margin_frac)` |  |
| `derive_psi_shape_a` | função | `(psi0, phi_ref_bep, psi_ref_bep) -> float` | Curvature 'a' in psi(phi) = psi0 - a*phi^2 that makes the curve pass exactly through (phi_ref_bep, psi_ref_bep). |
| `make_map` | função | `(name, D2_m, phi_ref_bep, psi_ref_bep, psi0, eta_p_max, eta_p_shape_b, phi_surge, phi_cho…)` |  |
| `flow_coefficient` | função | `(Q1_m3_s, N_rpm, coeffs) -> float` |  |
| `tip_speed_m_s` | função | `(N_rpm, coeffs) -> float` |  |
| `tip_mach_number` | função | `(N_rpm, coeffs, speed_of_sound_m_s) -> float` |  |
| `machine_reynolds_number` | função | `(N_rpm, coeffs, kinematic_viscosity_m2_s) -> float` |  |
| `evaluate_map` | função | `(coeffs, Q1_m3_s, N_rpm, speed_of_sound_m_s, kinematic_viscosity_m2_s, Re_ref=None) -> Ma…)` |  |
| `polytropic_head_from_psi` | função | `(psi, N_rpm, coeffs) -> float` | Total machine polytropic head (J/kg) = n_stages * psi * U2^2. |
| `required_speed_for_head` | função | `(target_head_J_kg, Q1_m3_s, coeffs, speed_of_sound_m_s, kinematic_viscosity_m2_s, *, N_mi…)` | Solves for the shaft speed producing target_head_J_kg at a fixed actual inlet volumetric flow, via bisection on the monotonic head(N) relation at fixed Q1. |

#### `pinneapple_systems.process_components.torsional_stickslip`

pinneapple_systems.process_components.torsional_stickslip -- 1D torsional-wave finite-difference simulation of a compliant rotating shaft, driven at one end at a prescribed angular velocity and loaded at the other end by Stribeck (velocity-weakening) friction acting on a lumped end inertia -- the …

<small>`pinneapple_systems/process_components/torsional_stickslip.py` · 179 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `StickSlipResult` | classe | `(t_s, z_m, omega_field_rad_s, end_omega_ts_rad_s, surface_torque_ts_N_m, stick_slip_index)` |  |
| `stribeck_friction_torque` | função | `(omega, T_static, T_kinetic, omega_breakout) -> float` | T_friction = sign(omega)*[T_kinetic + (T_static-T_kinetic)*exp(-\|omega\|/omega_breakout)]. |
| `simulate_torsional_stickslip` | função | `(length_m, G_Pa, J_m4, rho_kg_m3, omega_set_rad_s, T_static_N_m, T_kinetic_N_m, omega_bre…)` | Explicit finite-difference solve of the 1D torsional wave equation described in the module docstring, from a driven end (node 0, held at `omega_set_rad_s`) to a friction-loaded end (node -1, lumped inertia `end_inertia_kg_m2` under Stribeck friction). Uses a fixed uniform shaft cross-section … |

#### `pinneapple_systems.process_components.water_hammer`

Water hammer in a pumped pipeline: steady operating point and the valve-closure transient by the method of characteristics (MOC).

<small>`pinneapple_systems/process_components/water_hammer.py` · 191 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Pipeline` | classe | `(length, diameter, wave_speed, roughness, z_in, z_out, nu, rho)` |  |
| `PumpCurve` | classe | `(shutoff_head, k, sump_head)` | ``H = shutoff_head - k Q\|Q\|`` (fixed speed); ``sump_head``: suction-side head [m]. |
| `OperatingPoint` | classe | `(flow, velocity, reynolds, friction_factor, pump_head, valve_head, friction_loss, valve_c…)` |  |
| `TransientResult` | classe | `(t, x, head, flow, tau, dt, courant, op, info)` |  |
| `operating_point` | função | `(line, pump, *, valve_k, tank_head=0.0) -> OperatingPoint` | Flow where the pump curve meets friction (Colebrook) + the open valve's loss ``valve_k * V^2 / (2 g)``. |
| `linear_closure` | função | `(closure_time, start=0.0)` | ``tau(t)``: 1 until ``start``, linearly to 0 over ``closure_time`` (0 = instantaneous). |
| `moc_valve_closure` | função | `(line, pump, op, *, closure_time, t_end, n_reaches=40, friction=True, closure_start=0.0, …)` | Valve closure transient from the steady operating point ``op`` (pump upstream, valve + tank downstream). |
| `joukowsky_head_rise` | função | `(wave_speed, dv) -> float` | ``a dV / g`` [m]: head rise of an instantaneous velocity change ``dv`` (Joukowsky). |
| `wave_period` | função | `(length, wave_speed) -> float` | ``4 L / a`` [s]: period of the pressure oscillation after a closure (pipe from a reservoir). |

#### `pinneapple_systems.sports`

Sports physics: physics-informed analytics on public sports tracking and event data (football first).

<small>`pinneapple_systems/sports/__init__.py` · 2 linhas · **sem teste direto**</small>

#### `pinneapple_systems.sports.football`

Football: physics-informed analytics on public tracking and event data (ball flight, pitch control, player kinematics, expected goals). See docs/guides/football_physics.md.

<small>`pinneapple_systems/sports/football/__init__.py` · 3 linhas · testes: 16</small>

#### `pinneapple_systems.sports.football.ball`

Ball flight physics: gravity, quadratic drag and Magnus lift, the inverse problem from tracked positions, and a rolling-ball model for ground passes.

<small>`pinneapple_systems/sports/football/ball.py` · 464 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Ball` | classe | `(mass, diameter)` |  |
| `FlightFit` | classe | `(model, x0, v0, cd, magnus, g, t0, params, names, cov, rmse, n, success, extra)` | Result of :func:`fit_flight`: parameters, their Laplace (Gauss-Newton) standard errors and fit quality. |
| `Flight` | classe | `(match_id, t, xyz, detected)` | An aerial ball flight cut from a tracking record (times in s, positions in m). |
| `cd_crisis` | função | `(re, cd_sub=0.45, cd_super=0.18, re_crit=260000.0, width=25000.0)` | A smooth drag-crisis curve: ``cd_sub`` below ``re_crit``, ``cd_super`` above (logistic of width ``width``). |
| `simulate` | função | `(x0, v0, t, *, cd=0.0, magnus=(0.0, 0.0, 0.0), g=G, ball=BALL, cd_fn=None, dt=0.005) -> n…)` | Positions ``(len(t), 3)`` of a ball launched at ``x0`` with velocity ``v0`` at ``t[0]`` (RK4, step ``dt``). |
| `fit_flight` | função | `(t, xyz, model='drag', *, sigma=None, ball=BALL, g=G, observe=(0, 1, 2), cd_prior=None, l…)` | Fit a flight model to tracked positions ``xyz`` (``(n, 3)``; NaN rows are skipped) at times ``t``. |
| `predict` | função | `(fit, t, ball=BALL) -> np.ndarray` | Positions of a fitted flight at times ``t``. |
| `kinematic_extrapolation` | função | `(t, xyz, t_new, degree=2) -> np.ndarray` | Data-only baseline: a polynomial per axis fitted to the observed positions, evaluated at ``t_new``. |
| `extract_flights` | função | `(track, *, match_id=None, z_on=0.6, min_apex=2.5, min_duration=0.8, max_duration=4.5, max…)` | Aerial flights in a ball track (``frame, x, y, z, detected``, e.g. :func:`data.skillcorner_ball`). |
| `fisher_identifiability` | função | `(x0, v0, t, *, cd=0.25, magnus=(0.0, 0.0, 0.0), sigma=0.3, observe=(0, 1, 2), ball=BALL, …)` | Cramer-Rao standard deviations of the flight parameters for positions observed with noise ``sigma`` (m). |
| `rolling_fit` | função | `(t, xy, *, ball=BALL) -> dict[str, float]` | Fit ``dv/dt = -mu g - k_d v^2`` (rolling resistance plus drag) to a ground pass observed in 2-D. |
| `rolling_travel_time` | função | `(distance, v0, decel, cd=0.0, ball=BALL)` | Time for a rolling ball launched at ``v0`` to cover ``distance`` (inf if it stops first). |
| `travel_time` | função | `(distance, *, speed=None, v0=None, decel=None)` | Ball travel time to a target: constant ``speed`` (the usual pitch-control assumption) or a rolling ball. |

#### `pinneapple_systems.sports.football.ball_study`

Ball-flight studies: does flight physics help on real tracking data, and when is the inverse problem posed?

<small>`pinneapple_systems/sports/football/ball_study.py` · 235 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `synthetic_flights` | função | `(n=40, *, noise=0.2, fps=10.0, seed=0, g=B.G) -> list[tuple[B.Flight, dict]]` | Simulated kicks (speed 15-30 m/s, elevation 15-40 deg, drag-crisis C_D, random spin) observed at ``fps`` with Gaussian noise ``noise`` (m) on each coordinate. Returns ``(Flight, truth)`` pairs. |
| `synthetic_recovery` | função | `(n=40, *, noise=0.2, seed=0) -> dict[str, Any]` | Fit ``drag`` and ``drag_free_g`` to :func:`synthetic_flights`: error of launch speed, C_D and g, and the coverage of the Laplace 95 % intervals (should be close to 0.95 when the model is right). |
| `identifiability` | função | `(speed=22.0, elevation_deg=25.0, *, fps=25.0, sigma=0.2, duration=1.6) -> dict[str, Any]` | Cramer-Rao standard deviations for one lofted pass seen in 3-D versus 2-D (ball height not observed). |
| `flights_from_skillcorner` | função | `(match_ids, root=None, **kw) -> list[B.Flight]` | Aerial flights of SkillCorner matches (:func:`ball.extract_flights` on :func:`data.skillcorner_ball`). |
| `gravity_audit` | função | `(flights, *, max_rmse=0.6, workers=1) -> dict[str, Any]` | Gravity fitted freely to each flight (vacuum: quadratic in time; drag: ODE with free ``g``). |
| `holdout_prediction` | função | `(train, test, *, frac=0.5, train_fracs=(0.3, 0.5, 0.7), seed=0, workers=1) -> dict[str, A…)` | Observe the first ``frac`` of each test flight, predict the remaining positions; errors per model. |
| `summarise` | função | `(res) -> list[dict[str, Any]]` | Rows ``model, rmse_xy, rmse_z, landing_err_median`` of :func:`holdout_prediction`, best landing first. |

#### `pinneapple_systems.sports.football.data`

Public football data, downloaded at run time, cached outside git with the SHA-256 of every file.

<small>`pinneapple_systems/sports/football/data.py` · 282 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MetricaMatch` | classe | `(game, t, period, players, names, ball, events, fps, gk)` | One Metrica sample match in metres (origin at the centre spot, x along the long side). |
| `cache_dir` | função | `(root=None) -> Path` | The football cache folder (created). |
| `manifest` | função | `(root=None) -> dict[str, dict[str, Any]]` | URL -> {"path", "sha256", "bytes"} of every file fetched into the cache. |
| `fetch` | função | `(url, rel, root=None, *, timeout=120.0) -> Path` | Download ``url`` once to ``<cache>/rel`` and record its SHA-256; return the local path. |
| `metrica_match` | função | `(game, root=None) -> MetricaMatch` | Metrica sample game 1 or 2: tracking of both teams, ball and events in metres (downloaded once, ~65 MB). |
| `skillcorner_matches` | função | `(root=None) -> list[dict]` | The ten SkillCorner open-data matches (``id``, teams, date). |
| `skillcorner_ball` | função | `(match_id, root=None) -> dict[str, np.ndarray]` | Ball track of a SkillCorner match at 10 Hz: ``frame, t, period, x, y, z, detected`` (NaN when missing). |
| `statsbomb_shots` | função | `(seasons=None, root=None, *, max_matches=None) -> list[dict]` | Every open-play and set-piece shot with a 360 freeze frame in the given StatsBomb seasons. |

#### `pinneapple_systems.sports.football.kin_study`

Player-kinematics studies: smoothing with known truth, processing uncertainty of metabolic power on real tracking, and trajectory prediction with and without physical limits at equal data.

<small>`pinneapple_systems/sports/football/kin_study.py` · 175 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `synthetic_runs` | função | `(n=20, T=60.0, fps=25.0, *, seed=0, v_cap=9.0, a_cap=6.0, tau=1.0, sigma_a=3.0)` | True positions, velocities and accelerations of ``n`` players: Ornstein-Uhlenbeck acceleration (time constant ``tau``, stationary std ``sigma_a``) clipped at ``a_cap``, speed clipped at ``v_cap``. |
| `synthetic_smoothing` | função | `(noise=(0.05, 0.2), fps=(25.0, 10.0), *, n=20, seed=0) -> list[dict[str, Any]]` | RMSE of speed, acceleration and metabolic power per smoothing method against the known truth. |
| `match_load` | função | `(match, *, team='Home', max_frames=None) -> dict[str, Any]` | Metabolic power of one team of a Metrica match per player and per method, and plausibility per method (the first ``max_frames`` frames only, if given). ``gated_mean_mp_per_player`` drops the samples above the human limits of :mod:`.kinematics` (tracking jumps), which otherwise dominate a match mean. |
| `prediction_samples` | função | `(match, horizon, *, every=1.0, history=1.0) -> dict[str, np.ndarray]` | Samples ``(features, pos, vel, acc, target)`` for every outfield player every ``every`` s with a complete past ``history`` and a tracked position ``horizon`` s later. States are causal Kalman estimates. |
| `prediction_study` | função | `(train, test, horizon, *, sizes=(100, 300, 1000, 3000, None), seeds=(0, 1, 2)) -> dict[st…)` | Final displacement error on ``test`` of each predictor trained on random subsets of ``train``. |

#### `pinneapple_systems.sports.football.kinematics`

Player kinematics from tracking: smoothing with uncertainty, physical plausibility, metabolic power and short-horizon trajectory prediction with and without physical limits.

<small>`pinneapple_systems/sports/football/kinematics.py` · 300 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MLPredictor` | classe | `(residual_of, project, max_iter, seed, _models)` | Data-only displacement predictor: gradient-boosted trees (scikit-learn) on :func:`history_features`. |
| `kalman_smooth` | função | `(xy, dt, *, sigma_pos=0.1, jerk=30.0, causal=False, max_gap=1.0) -> dict[str, np.ndarray]` | RTS-smoothed position, velocity and acceleration of tracks ``xy`` of shape ``(T, ..., 2)``. |
| `fit_noise` | função | `(xy, dt, sigmas=(0.01, 0.02, 0.05, 0.1, 0.2), jerks=(5.0, 10.0, 20.0, 40.0, 80.0))` | Measurement noise and jerk density by maximum marginal likelihood of :func:`kalman_smooth` (grid search on the innovations): the data, not a guess, set how much the smoother trusts the samples. |
| `finite_differences` | função | `(xy, dt) -> dict[str, np.ndarray]` | Central differences of raw positions: ``vel``, ``acc`` (the data-only baseline with no smoothing). |
| `savgol_derivatives` | função | `(xy, dt, window=0.5, order=2) -> dict[str, np.ndarray]` | Savitzky-Golay smoothed position, velocity and acceleration (``window`` in seconds; NaN filled linearly). |
| `plausibility` | função | `(vel, acc, *, v_max=V_MAX, a_max=A_MAX) -> dict[str, float]` | Share of finite samples whose speed exceeds ``v_max`` or whose acceleration magnitude exceeds ``a_max``. |
| `minetti_cost` | função | `(i)` | Energy cost of running on slope ``i`` (Minetti et al. 2002), J/(kg m); ``i`` clipped to [-0.45, 0.45]. |
| `metabolic_power` | função | `(speed, accel_fwd, *, terrain=1.0) -> np.ndarray` | Metabolic power in W/kg from speed (m/s) and forward acceleration (m/s^2), equivalent-slope model. |
| `forward_acceleration` | função | `(vel, acc) -> np.ndarray` | Component of the acceleration along the velocity (tangential), the input of :func:`metabolic_power`. |
| `reachable_radius` | função | `(speed0, horizon, *, v_max=V_MAX, a_max=A_MAX)` | Largest distance a player moving at ``speed0`` can cover in ``horizon`` s (accelerate at ``a_max`` up to ``v_max``): the physical bound on any displacement prediction. |
| `predict_constant_velocity` | função | `(pos, vel, horizon)` | Position after ``horizon`` s at constant velocity. |
| `predict_limited` | função | `(pos, vel, acc, horizon, *, v_max=V_MAX, a_max=A_MAX, tau=0.5, dt=0.04)` | Physics prediction: the current acceleration (capped at ``a_max``) decays with time constant ``tau`` and the speed is capped at ``v_max`` (players cannot keep accelerating; a simple first-order motor model). |
| `project_reachable` | função | `(pos, vel, pred, horizon, *, v_max=V_MAX, a_max=A_MAX)` | Project predictions onto the reachable set: the disc around the coasting point ``pos + vel * horizon`` of radius ``a_max horizon^2 / 2`` intersected with the disc of radius :func:`reachable_radius` around ``pos``. |
| `history_features` | função | `(pos_hist, ball_rel=None) -> np.ndarray` | Features for the ML predictor: past positions relative to the current one (``(n, k, 2)`` -> ``(n, 2k)``), optionally the ball position relative to the player. |
| `displacement_errors` | função | `(pred, true, pos=None, vel=None, horizon=None) -> dict[str, float]` | Mean and 90th-percentile final displacement error (m); with ``pos, vel, horizon`` also the share of predictions outside the physically reachable set. |

#### `pinneapple_systems.sports.football.metrics`

Scores for probabilistic binary predictions: log-loss, Brier, ROC AUC, calibration (reliability bins, ECE) and bootstrap intervals of score differences between two models on the same events.

<small>`pinneapple_systems/sports/football/metrics.py` · 78 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `log_loss` | função | `(p, y) -> float` |  |
| `brier` | função | `(p, y) -> float` |  |
| `auc` | função | `(p, y) -> float` | ROC AUC by the rank-sum (Mann-Whitney) formula, ties averaged. |
| `reliability` | função | `(p, y, bins=10, strategy='quantile') -> dict[str, np.ndarray]` | Mean predicted probability, observed frequency and count per bin (``quantile`` or ``uniform`` bins). |
| `ece` | função | `(p, y, bins=10) -> float` | Expected calibration error with quantile bins. |
| `score` | função | `(p, y) -> dict[str, float]` | All scores of one model. |
| `paired_bootstrap` | função | `(p_a, p_b, y, metric='log_loss', n=2000, seed=0) -> dict[str, Any]` | Difference ``metric(a) - metric(b)`` on the same events with a 95 % bootstrap interval (negative: a better for losses). |

#### `pinneapple_systems.sports.football.pc_study`

Pass outcome: physics-based pitch control against data-only classifiers at equal data.

<small>`pinneapple_systems/sports/football/pc_study.py` · 131 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `physics_features` | função | `(ps, p) -> np.ndarray` | Pitch-control probability, earliest attacker and defender intercept times minus the ball travel time. |
| `fit_models` | função | `(train, *, seed=0, which=…) -> dict[str, Any]` | Fitted models (parameters or estimators) on ``train``. |
| `predict_models` | função | `(models, ps) -> dict[str, np.ndarray]` |  |
| `compare` | função | `(train, test, *, seed=0) -> dict[str, Any]` | Scores of every model on ``test`` after fitting on ``train``, paired bootstrap of log-loss differences against the best data-only model, and the fitted physical parameters. |
| `data_efficiency` | função | `(train, test, sizes=(25, 50, 100, 200, 400, None), seeds=(0, 1, 2), which=…) -> dict[str,…)` | Test log-loss of each model trained on random subsets of ``train`` (``None`` = all passes). |
| `with_params` | função | `(p, **kw) -> PC.PCParams` |  |

#### `pinneapple_systems.sports.football.pitch_control`

Pitch control: who wins the ball at a target point, from player kinematics and ball travel time.

<small>`pinneapple_systems/sports/football/pitch_control.py` · 275 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PCParams` | classe | `(tti, v_max, a_max, reaction, sigma, lam, kappa_def, gk_factor, ball, ball_speed, roll, d…)` |  |
| `PassSet` | classe | `(att_pos, att_vel, def_pos, def_vel, att_gk, def_gk, ball, target, y, duration, meta)` | Passes with the state at the kick: ``(n, 11, 2)`` attacker/defender positions and velocities (causal Kalman estimates), ball position, target (pass end), label (1 = completed to a team-mate, 0 = intercepted). |
| `time_to_intercept` | função | `(pos, vel, target, p) -> np.ndarray` | Time for players at ``pos`` with velocity ``vel`` (``(..., 2)``) to reach ``target`` (broadcastable). |
| `control_probability` | função | `(att_pos, att_vel, def_pos, def_vel, ball_pos, target, p=DEFAULT, att_gk=None, def_gk=Non…)` | Probability that the attacking team controls the ball at ``target`` (vectorised over events). |
| `passes_from_metrica` | função | `(match, *, sigma_pos=0.1, jerk=30.0) -> PassSet` | Completed (``PASS``) and intercepted (``BALL LOST`` with an interception subtype) passes of a :class:`data.MetricaMatch`, with player states from the causal Kalman filter of :mod:`.kinematics`. |
| `fit_ball_travel` | função | `(ps, *, min_len=5.0) -> dict[str, Any]` | Fit the rolling-ball travel time ``t(d)`` (kick speed ``v0 = a + b d``, constant deceleration) and a constant speed to the observed durations of the completed passes; returns both fits and their RMSE (s). |
| `fit_params` | função | `(ps, base, names=…, *, max_iter=200) -> tuple[PCParams, float]` | Maximum-likelihood fit of the named parameters on the pass outcomes (Nelder-Mead on log-parameters). |
| `geometric_features` | função | `(ps) -> tuple[np.ndarray, list[str]]` | Data-only features of a pass (no kinematics, no physics): lengths, positions, distances of the nearest attackers and defenders to the target and to the pass line, defender counts near the target. |
| `pitch_surface` | função | `(att_pos, att_vel, def_pos, def_vel, ball_pos, p=DEFAULT, *, nx=53, ny=34, pitch=(105.0, …)` | Attacking-team control probability on an ``nx x ny`` grid for one frame (for figures). |

#### `pinneapple_systems.sports.football.plots`

Figures of the football studies (matplotlib, static PNG): pitch, pitch-control surface, ball flights in 3-D, gravity audit, reliability curves, data-efficiency curves, smoothing errors.

<small>`pinneapple_systems/sports/football/plots.py` · 224 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `color_of` | função | `(name) -> str` | The fixed colour of a model name (grey for names outside :data:`MODEL_COLORS`). |
| `draw_pitch` | função | `(ax, length=105.0, width=68.0, color='#b9b7b0')` | Pitch lines in metres, origin at the centre spot. |
| `pitch_control_figure` | função | `(xs, ys, q, att_pos, att_vel, def_pos, def_vel, ball, target, path, title='')` |  |
| `flights_3d` | função | `(flights, path, title='')` | ``(observed xyz, fitted xyz or None, label)`` per flight, in a 3-D view over the pitch plane. |
| `gravity_audit_figure` | função | `(audit, synthetic_g, path)` |  |
| `reliability_figure` | função | `(rel, path, title='', models=None, lim=1.0)` |  |
| `efficiency_figure` | função | `(rows, metric, path, title='', ylabel=None, models=None, flat=None)` | Mean test ``metric`` against training-set size per model (seeds averaged, min-max band). |
| `smoothing_figure` | função | `(rows, path, metric='mp_rmse', ylabel='metabolic power RMSE [W/kg]')` |  |
| `holdout_figure` | função | `(res, path, models=None)` | Distribution of landing errors per model (box plot). |

#### `pinneapple_systems.sports.football.xg`

Expected goals with and without physics: a standard shot-geometry baseline against intercept-time features computed from the StatsBomb 360 freeze frame.

<small>`pinneapple_systems/sports/football/xg.py` · 239 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `XGParams` | classe | `(v_shot, v_gk, reach_gk, reaction_gk, v_def, reach_def, reaction_def, n_mouth, n_path)` |  |
| `shots_arrays` | função | `(shots) -> dict[str, Any]` | Shot table to arrays: positions in metres, label, freeze frame padded to ``(n, 22, 4)`` (x, y, mate, gk). |
| `standard_features` | função | `(A) -> tuple[np.ndarray, list[str]]` |  |
| `physics_features` | função | `(A, p=None, *, chunk=200) -> tuple[np.ndarray, list[str]]` |  |
| `freeze_geometry_features` | função | `(A) -> tuple[np.ndarray, list[str]]` | Data-only freeze-frame features (no physics): opponents inside the shot triangle, keeper distance to the goal centre and his lateral offset from the shot line, nearest opponent distance to the shooter. |
| `select_params` | função | `(A_train, grid_v_shot=(12.0, 16.0, 20.0, 25.0, 30.0), grid_v_gk=(3.0, 5.0, 7.0), folds=5)…)` | Choose ``v_shot`` and ``v_gk`` by grouped cross-validated log-loss of a logistic model on the standard plus physics features, on the training shots only. |
| `compare` | função | `(A_train, A_test, *, params=None, seed=0) -> dict[str, Any]` | Test scores of: standard logistic / GBM, physics-only logistic, standard+physics logistic / GBM, standard plus data-only freeze-frame geometry (with and without physics); StatsBomb's own xG on the same shots as an external reference (a proprietary model trained on far more shots). |
| `data_efficiency` | função | `(A_train, A_test, params, sizes=(100, 200, 400, 800, None), seeds=(0, 1, 2, 3, 4))` | Test log-loss of the standard and standard+physics logistic models trained on random subsets. |

#### `pinneapple_systems.time_series`

_Módulo sem docstring e sem texto._

<small>`pinneapple_systems/time_series/__init__.py` · 108 linhas · testes: 90</small>

#### `pinneapple_systems.time_series.adapters`

_Módulo sem docstring e sem texto._

<small>`pinneapple_systems/time_series/adapters/__init__.py` · 1 linhas · **sem teste direto**</small>

#### `pinneapple_systems.time_series.adapters.shape`

Shape adapters for time series models (e.g. LastStepToHorizon).

<small>`pinneapple_systems/time_series/adapters/shape.py` · 14 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LastStepToHorizon` | classe | `(model, horizon)` |  |

#### `pinneapple_systems.time_series.adaptive`

Adaptive forecasting: switch between and combine forecasting models online, using only past out-of-sample errors.

<small>`pinneapple_systems/time_series/adaptive.py` · 447 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SimpleExpSmoothing` | classe | `(alpha, level_)` |  |
| `HoltDamped` | classe | `(alpha, beta, phi, level_, trend_)` | Holt's linear trend with damping phi (Gardner & McKenzie 1985). |
| `HoltWintersAdditive` | classe | `(season_length, alpha, beta, gamma, level_, trend_, seasonal_, n_)` |  |
| `Theta` | classe | `(alpha, level_, slope_, n_)` | Theta method as SES with drift equal to half the regression slope (Assimakopoulos & Nikolopoulos 2000; Hyndman & Billah 2003, Int. J. Forecasting 19:287). |
| `MovingAverage` | classe | `(window, value_)` |  |
| `RidgeAR` | classe | `(p, window, ridge, coef_, tail_)` | Autoregression of order p with an intercept, ridge-regularised, fitted on the last ``window`` points; multi-step forecasts are recursive. |
| `LagRegressorExpert` | classe | `(estimator, n_lags=12, refit_every=24, window=500)` | Any tabular regressor (scikit-learn style ``fit(X, y)`` / ``predict(X)``, including the ML forecasters of ``pinneapple_systems.time_series.models``) as an expert: lagged values as features, recursive multi-step forecasts. The estimator is refitted every ``refit_every`` calls (on the last ``window`` … |
| `AdaptiveRun` | classe | `(origins, forecast, expert_forecasts, lower, upper, weights, active, expert_names, y)` | Result of :meth:`AdaptiveForecaster.run`: one row per forecast origin t (forecasts for t+1..t+H). |
| `AdaptiveForecaster` | classe | `(experts, horizon=1, mode='combine', etas=(0.5, 2.0, 8.0), alphas=(0.0, 0.01, 0.05), loss…)` | Online selection and combination of forecasting models. |
| `default_experts` | função | `(season_length=1) -> dict[str, Any]` | A diverse, cheap pool. Hyper-parameter variants are separate experts, so choosing a smoothing constant is part of the online selection instead of an in-sample fit. |

#### `pinneapple_systems.time_series.audit`

_Módulo sem docstring e sem texto._

<small>`pinneapple_systems/time_series/audit/__init__.py` · 4 linhas · **sem teste direto**</small>

#### `pinneapple_systems.time_series.audit.report`

_Módulo sem docstring e sem texto._

<small>`pinneapple_systems/time_series/audit/report.py` · 28 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AuditSection` | classe | `(name, results)` |  |
| `AuditReport` | classe | `(meta, sections)` | Serializable container for time-series diagnostic outputs. |

#### `pinneapple_systems.time_series.audit.tests`

_Módulo sem docstring e sem texto._

<small>`pinneapple_systems/time_series/audit/tests.py` · 315 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TSAuditor` | classe | `(nlags, arch_lags)` |  |
| `adf_test` | função | `(y, *, autolag='AIC', min_n=20) -> Dict[str, Any]` |  |
| `kpss_test` | função | `(y, *, regression='c', nlags='auto', min_n=20) -> Dict[str, Any]` |  |
| `pp_test` | função | `(y, min_n=20) -> Dict[str, Any]` | Phillips–Perron test. |
| `arch_test` | função | `(y, *, lags=12, min_n=30) -> Dict[str, Any]` |  |
| `acf_pacf` | função | `(y, *, nlags=40, min_n=50) -> Dict[str, Any]` |  |
| `chow_test_mean_shift` | função | `(y, break_idx, min_n=20) -> Dict[str, Any]` | Simple Chow test for a mean shift using an intercept-only model. |
| `changepoints_ruptures` | função | `(y, *, model='l2', max_breaks=5, min_n=50) -> Dict[str, Any]` | Multi-break approximation via `ruptures` (if installed). |

#### `pinneapple_systems.time_series.base`

Base TSOutput and TSModelBase for time series models.

<small>`pinneapple_systems/time_series/base.py` · 26 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TSOutput` | classe | `(y_hat, extras)` | Standard output for time series model predictions. |
| `TSModelBase` | classe | `()` | Base opcional para modelos TS. |

#### `pinneapple_systems.time_series.baselines`

_Módulo sem docstring e sem texto._

<small>`pinneapple_systems/time_series/baselines/__init__.py` · 2 linhas · **sem teste direto**</small>

#### `pinneapple_systems.time_series.baselines.naive`

_Módulo sem docstring e sem texto._

<small>`pinneapple_systems/time_series/baselines/naive.py` · 71 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NaiveForecaster` | classe | `(last_value_)` | Predicts the last observed value for all future steps. |
| `SeasonalNaiveForecaster` | classe | `(season_length, history_)` | Repeats the value from the same season in the last cycle. |
| `DriftForecaster` | classe | `(first_, last_, n_)` | Linearly extrapolates from the first to the last training point. |

#### `pinneapple_systems.time_series.classical_wrappers`

Real wrappers around the classical statistical forecasting libraries.

<small>`pinneapple_systems/time_series/classical_wrappers.py` · 244 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `StatsmodelsARIMAForecaster` | classe | `(order, seasonal_order, trend, enforce_stationarity, enforce_invertibility, fit_kwargs)` | Real ARIMA/SARIMA forecaster wrapping `statsmodels`. |
| `AutoARIMAForecaster` | classe | `(seasonal, m, max_p, max_q, max_d, max_P, max_Q, max_D, stepwise, suppress_warnings, auto…)` | Real forecaster wrapping `pmdarima.auto_arima`. |

#### `pinneapple_systems.time_series.csv_loader`

CSV → tensor loader for time series datasets.

<small>`pinneapple_systems/time_series/csv_loader.py` · 322 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TimeSeriesScaler` | classe | `() -> None` | Channel-wise z-score scaler: fit on training data, apply to all splits. |
| `load_timeseries_csv` | função | `(path, *, target_cols=None, feature_cols=None, time_col=None, fill_method='ffill', normal…)` | Load a CSV file and return train / val / test tensors ready for PINNeAPPle. |
| `generate_synthetic_csv` | função | `(path, T=1000, seed=42, freq='D') -> Path` | Generate a synthetic multivariate time series CSV and save it to *path*. |

#### `pinneapple_systems.time_series.datamodule`

Time series DataModule with deterministic, time-ordered splits and index-based loaders.

<small>`pinneapple_systems/time_series/datamodule.py` · 80 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TSDataModule` | classe | `(series, spec, batch_size, val_ratio, num_workers, pin_memory)` |  |

#### `pinneapple_systems.time_series.datasets`

_Módulo sem docstring e sem texto._

<small>`pinneapple_systems/time_series/datasets/__init__.py` · 1 linhas · **sem teste direto**</small>

#### `pinneapple_systems.time_series.datasets.public`

Openly licensed real series, downloaded from a pinned commit and checked against a SHA-256.

<small>`pinneapple_systems/time_series/datasets/public.py` · 183 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PublicSeries` | classe | `(name, time, values, source)` |  |
| `YouTubeDaily` | classe | `(dates, n_entries, total_views, total_likes, category_counts, source)` | One row per trending day: the date, the number of trending entries, their total views and likes, and the number of entries per YouTube category id. |
| `fetch_pinned` | função | `(key, cache_dir=None, *, local=None) -> bytes` | Bytes of the pinned file ``key`` (from ``local``, the cache, or the network), checksum verified. |
| `load_usd_eur_daily` | função | `(cache_dir=None) -> PublicSeries` | U.S. dollars per euro on business days (days without a quote are dropped). |
| `load_sunspots_yearly` | função | `(cache_dir=None) -> PublicSeries` |  |
| `log_returns` | função | `(values) -> np.ndarray` |  |
| `youtube_daily_aggregates` | função | `(csv_text) -> YouTubeDaily` | Aggregate the rows of a trending-videos CSV (columns ``trending_date`` as yy.dd.mm, ``category_id``, ``views``, ``likes``) to one row per trending day, in date order. |
| `load_youtube_trending_us` | função | `(cache_dir=None) -> YouTubeDaily` |  |

#### `pinneapple_systems.time_series.datasets.windowed`

Windowed time series dataset for input/horizon pairs.

<small>`pinneapple_systems/time_series/datasets/windowed.py` · 41 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `WindowedTimeSeriesDataset` | classe | `(series, spec)` | series: (T, F) x: (L_in, F) y: (H, C)   (aqui C = F por padrão) |

#### `pinneapple_systems.time_series.decomposition`

_Módulo sem docstring e sem texto._

<small>`pinneapple_systems/time_series/decomposition/__init__.py` · 53 linhas · **sem teste direto**</small>

#### `pinneapple_systems.time_series.decomposition.fft_forecaster`

Pure FFT forecaster: fits dominant sinusoids and extrapolates.

<small>`pinneapple_systems/time_series/decomposition/fft_forecaster.py` · 86 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FFTForecaster` | classe | `(n_harmonics=10, detrend=True)` | Fits a signal using its top-K dominant frequency components (via FFT) and extrapolates them forward. |

#### `pinneapple_systems.time_series.decomposition.fft_lstm`

Hybrid FFT decomposition + LSTM residual forecaster (multivariate-ready).

<small>`pinneapple_systems/time_series/decomposition/fft_lstm.py` · 335 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FFTDecomposer` | classe | `(n_harmonics=10, detrend=True, auto_harmonics=True, auto_threshold=0.05) -> None` | Fits one FFTForecaster per channel; computes residuals and extrapolations. |
| `ResidualLSTMConfig` | classe | `(input_len, horizon, n_features, n_targets, hidden_size, num_layers, dropout, layer_norm)` | Hyperparameters for ResidualLSTMForecaster. |
| `ResidualLSTMForecaster` | classe | `(cfg=None, **kw)` | LSTM forecaster trained on FFT residuals. |
| `FFTLSTMPipeline` | classe | `(*, n_harmonics=10, detrend=True, auto_harmonics=True, auto_threshold=0.05, input_len=64,…)` | Orchestrates the two-stage hybrid FFT + LSTM pipeline. |

#### `pinneapple_systems.time_series.decomposition.fft_nn`

FFT + Neural Network on residual decomposition forecaster.

<small>`pinneapple_systems/time_series/decomposition/fft_nn.py` · 100 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FFTNNForecaster` | classe | `(n_harmonics=10, detrend=True, input_len=32, horizon=16, nn_model=None, nn_epochs=50, nn_…)` | Two-stage forecaster: 1. FFT component captures global periodicity / trend. 2. A neural network (default: MLP) models the residual. |

#### `pinneapple_systems.time_series.decomposition.hht_lstm`

Hilbert-Huang Transform (EMD-based) + LSTM residual forecaster (multivariate-ready).

<small>`pinneapple_systems/time_series/decomposition/hht_lstm.py` · 341 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HHTDecomposer` | classe | `(n_imfs=6, dominant_energy_ratio=0.8, detrend=True) -> None` | Fits one EMD decomposer per channel; computes residuals and extrapolations. |
| `HHTLSTMPipeline` | classe | `(n_imfs=6, dominant_energy_ratio=0.8, detrend=True, input_len=64, horizon=16, n_features=…)` | Orchestrates HHTDecomposer + LSTM residual forecasting. |

#### `pinneapple_systems.time_series.decomposition.hht_nn`

Hilbert-Huang Transform + Neural Network on residual decomposition forecaster.

<small>`pinneapple_systems/time_series/decomposition/hht_nn.py` · 172 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HHTNNForecaster` | classe | `(n_imfs=5, input_len=32, horizon=16, nn_model=None, nn_epochs=50, nn_lr=0.001)` | Two-stage forecaster: 1. EMD (Empirical Mode Decomposition) decomposes the signal into IMFs. Dominant IMFs are extrapolated via Hilbert instantaneous frequency. 2. A neural network models the residual (sum of discarded IMFs + trend). |

#### `pinneapple_systems.time_series.eda`

_Módulo sem docstring e sem texto._

<small>`pinneapple_systems/time_series/eda/__init__.py` · 26 linhas · **sem teste direto**</small>

#### `pinneapple_systems.time_series.eda.plots`

10 EDA tools for time series analysis.

<small>`pinneapple_systems/time_series/eda/plots.py` · 463 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_trend` | função | `(y, *, window=24, title='Trend Analysis', figsize=(12, 6), show=False) -> Figure` | Raw series + rolling mean + rolling std band. |
| `track_setpoint` | função | `(actual, setpoint, *, tolerance=0.05, title='Setpoint Tracking', figsize=(12, 5), show=Fa…)` | Actual vs setpoint with deviation shading and tolerance band. |
| `rolling_statistics` | função | `(y, windows=(8, 24, 96), *, stats=('mean', 'std'), title='Rolling Statistics', figsize=(1…)` | Multi-window rolling statistics overlay. |
| `step_response` | função | `(y, step_idx=None, *, title='Step Response', figsize=(12, 5), show=False) -> Figure` | Visualize system response to a detected or specified step. Estimates: steady-state gain, rise time (10%→90%), settling time (±2%). |
| `power_spectrum` | função | `(y, *, fs=1.0, n_top=5, title='FFT Power Spectrum', figsize=(12, 5), show=False) -> Figure` | Log-scale FFT power spectrum with top-N dominant frequencies annotated. |
| `plot_acf_pacf` | função | `(y, nlags=40, *, alpha=0.05, title='ACF / PACF', figsize=(12, 5), show=False) -> Figure` | Side-by-side ACF and PACF with confidence bounds. |
| `stationarity_report` | função | `(y, *, title='Stationarity Tests', figsize=(10, 4), show=False) -> Figure` | Runs ADF, KPSS, and PP tests and renders a summary table. |
| `changepoint_plot` | função | `(y, n_bkps=3, *, model='rbf', title='Change Point Detection', figsize=(12, 5), show=False…)` | Visualize change points detected via the ruptures library. |
| `cross_correlation` | função | `(df, cols=None, max_lag=20, *, title='Cross-Correlation', figsize=(12, 6), show=False) ->…)` | Heatmap of peak cross-correlation values and lag at peak between all column pairs. |
| `rga_matrix` | função | `(K, mv_labels=None, cv_labels=None, *, title='Relative Gain Array (RGA)', figsize=(8, 6),…)` | Compute and visualize the Bristol Relative Gain Array. |

#### `pinneapple_systems.time_series.enso`

ENSO prediction: data, physical models, physics-informed and data-driven forecasters, evaluation.

<small>`pinneapple_systems/time_series/enso/__init__.py` · 2 linhas · testes: 22</small>

#### `pinneapple_systems.time_series.enso.data`

Openly licensed ENSO observations, downloaded once, reduced to monthly tropical-Pacific products and cached with the SHA-256 of every source file.

<small>`pinneapple_systems/time_series/enso/data.py` · 674 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Manifest` | classe | `(path, expected=None)` | URL -> {sha256, bytes} of every source file of a product. ``expected`` (an earlier manifest) pins hashes. |
| `Gridded` | classe | `(time, lat, lon, values, source)` | Monthly fields (nt, ny, nx) on a regular lat/lon grid (cell centres, lon in 0-360). |
| `manifests` | função | `(cache_dir=None) -> dict[str, Path]` | The manifests written so far in the cache: product -> path of ``manifest_<product>.json``. |
| `fetch` | função | `(url, *, timeout=300, retries=4) -> bytes` | Bytes at ``url`` (through the environment's proxy settings), retried with exponential backoff. |
| `month_range` | função | `(start, end) -> list[str]` | 'YYYY-MM' strings from ``start`` to ``end`` inclusive. |
| `oisst_monthly` | função | `(start='1981-09', end=None, *, lat=(-30.0, 30.0), lon=(100.0, 300.0), res=1.0, cache_dir=…)` | Monthly mean OISST v2.1 SST [deg C] in ``res``-degree cells (a multiple of 0.25) over ``lat`` x ``lon``. |
| `box_mean` | função | `(g, var, box) -> np.ndarray` | cos(latitude)-weighted mean of ``g.values[var]`` over cells whose centres lie in ``box``. |
| `nino_indices` | função | `(g, var='sst') -> dict[str, np.ndarray]` | Nino 3.4, 3, 4 and 1+2 box means (not anomalies) of ``g``. |
| `coarsen` | função | `(g, ky, kx) -> Gridded` | Block means of ky x kx cells (NaN-aware; trailing cells that do not fill a block are dropped). |
| `anomalies` | função | `(x, time, base) -> tuple[np.ndarray, np.ndarray]` | Anomalies from the calendar-month climatology of the period ``base`` = ('YYYY-MM', 'YYYY-MM') inclusive. |
| `d20_from_profile` | função | `(z, T, *, iso=20.0, max_gap=60.0) -> float` | Depth [m] where the profile first crosses ``iso`` going down (linear interpolation between the bracketing levels), or NaN when the profile does not start above ``iso`` within the top 30 m, does not reach below it, or the bracketing levels are more than ``max_gap`` m apart. |
| `fetch_to_file` | função | `(url, path, *, timeout=1800, retries=4, chunk=1 << 22) -> str` | Stream ``url`` to ``path`` and return its SHA-256 (large files never sit in memory). |
| `wod_d20_casts` | função | `(years, kinds=('mrb',), *, lat=(-8.0, 8.0), lon=(120.0, 280.0), cache_dir=None, manifest=…)` | D20 of every quality-controlled WOD cast (flags 0) in the box, from the yearly files ``wod_<kind>_<year>.nc`` (``kind`` in mrb = moored buoys, xbt, pfl = profiling floats, ctd, ...). |
| `d20_monthly` | função | `(casts, months, *, lat=(-5.0, 5.0), lon_edges=tuple(range(130, 281, 10)), min_casts=1, da…)` | Monthly mean D20 [m] per longitude bin (``len(lon_edges) - 1`` bins) for casts with latitude in ``lat``; NaN where a bin has fewer than ``min_casts`` casts. Each cast counts once (a mooring reporting daily weighs as much as 30 XBT drops); ``daily_per_platform`` is kept for the record (WOD moored … |
| `d20_climatology` | função | `(casts, lon_edges, lat_edges, base) -> np.ndarray` | Mean D20 [m] of the casts in each (lat, lon) bin over the months of ``base`` (each calendar month weighted equally); NaN where a bin has no cast. Shape (len(lat_edges) - 1, len(lon_edges) - 1). |
| `era5_monthly` | função | `(variables=…, *, cache_dir=None, manifest=None, workers=8, derived=True, log=print) -> Gr…)` | Monthly means of 6-hourly ERA5 fields on the 64 x 32 WeatherBench 2 grid, 1959-01 to 2022-12, global. |
| `wind_stress` | função | `(u, v, rho_air=1.2, cd=0.0013)` | Bulk wind stress tau = rho_air C_D \|U\| U [N m-2] from 10 m wind components [m/s] (constant C_D; Large & Pond 1981 give 1.1e-3 to 1.5e-3 for 4-11 m/s). |
| `soi_from_mslp` | função | `(g, var='mean_sea_level_pressure', base=None) -> np.ndarray` | Tahiti (17.5S, 149.6W) minus Darwin (12.4S, 130.9E) sea-level pressure, each standardised per calendar month over ``base`` (default: the whole record), the difference standardised again (the CPC/BoM construction) -- taken at the nearest cells of ``g``, so on a coarse grid it is a proxy of the … |
| `ncei_heat_content` | função | `(box=(-5.0, 5.0, 120.0, 280.0), *, cache_dir=None, manifest=None) -> dict[str, Any]` | Sum of the NCEI 0-700 m heat-content anomaly [1e18 J] over the 1-deg cells in ``box``, 3-month means (JFM, AMJ, JAS, OND) 1955-2020; ``mid_month`` is the centre month 'YYYY-MM' of each season. |
| `statsmodels_nino12` | função | `() -> dict[str, Any]` | Monthly Nino 1+2 SST [deg C] 1950-2010 as shipped with statsmodels (NOAA CPC); time as 'YYYY-MM'. |

#### `pinneapple_systems.time_series.enso.dataset`

The ENSO study data set: monthly indices and maps built from the cached sources of :mod:`.data`.

<small>`pinneapple_systems/time_series/enso/dataset.py` · 226 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EnsoDataset` | classe | `(months, indices, sst_maps, grid, d20_band, d20_eq, d20_lon, d20_observed, taux, base, cl…)` |  |
| `eof_fill` | função | `(X, base_rows, n_eofs=4, ridge=0.1) -> np.ndarray` | Fill NaNs of each row of X (nt, nb) from its observed entries: least-squares amplitudes of the leading ``n_eofs`` EOFs (computed on the complete rows among ``base_rows``), ridge-regularised in units of the EOF variances. Observed entries are kept as they are. Rows with fewer observed entries than … |
| `build_dataset` | função | `(start='1982-01', end='2024-12', base=('1982-01', '2011-12'), grid=None, *, cache_dir=Non…)` |  |
| `to_series` | função | `(ds, columns=COLUMNS, name='enso')` | :class:`~pinneapple_systems.time_series.forecast_lab.SeriesData` with ``columns`` (target = first) and the maps in ``meta`` (``month_of_year``, ``sst_maps``, ``d20_band``, ``d20_eq``, ``d20_lon``, ``taux``, ``grid``, ``months``). |

#### `pinneapple_systems.time_series.enso.evaluation`

ENSO-specific scores on top of the forecast_lab ones: anomaly correlation per lead, skill stratified by target calendar month (the spring predictability barrier), categorical El Nino / neutral / La Nina verification, and a blocked cross-validated hindcast that gives every year an out-of-sample …

<small>`pinneapple_systems/time_series/enso/evaluation.py` · 96 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `acc` | função | `(mean, y) -> list[float]` | Anomaly correlation (Pearson, over origins) per horizon column. |
| `by_target_month` | função | `(mean, y, origins, month_of_year, horizons, min_n=6) -> dict[str, Any]` | ACC and RMSE for each (target calendar month, horizon); NaN where fewer than ``min_n`` forecasts. |
| `category` | função | `(x, threshold=0.5) -> np.ndarray` | -1 La Nina (x <= -threshold), 0 neutral, +1 El Nino (x >= threshold). |
| `categorical` | função | `(mean, y, threshold=0.5) -> dict[str, Any]` | Per horizon: 3 x 3 contingency table (rows observed La Nina / neutral / El Nino, columns forecast), hit rate (probability of detection) of each category, false-alarm ratio of the warm and cold categories, accuracy, and the Heidke skill score (accuracy relative to random forecasts with the same … |
| `blocked_hindcast` | função | `(make, data, horizons, *, block=48, first=36, last=None, val_frac=0.15) -> dict[str, Any]` | Out-of-sample forecasts for every origin in [first, last]: the origins are cut into blocks of ``block`` months; each block is forecast by a model fitted on all other origins whose targets end more than max(h) before the block or start after it (purged on both sides), with its last ``val_frac`` (in … |

#### `pinneapple_systems.time_series.enso.forecasters`

ENSO forecasters with the :class:`~pinneapple_systems.time_series.forecast_lab.Forecaster` interface, so the forecast_lab protocol (purged walk-forward tuning, untouched test period, DM tests) applies to them unchanged.

<small>`pinneapple_systems/time_series/enso/forecasters.py` · 683 linhas · testes: 22 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ColumnSubset` | classe | `(inner, cols, name=None, family=None)` | Wrap ``inner`` so that it sees only ``cols`` (names in ``data.meta["columns"]``; the target must be one). |
| `RechargeOscillatorForecaster` | classe | `(h_col='wwv', seasonal=(), multiplicative=False, stochastic=True, n_members=200, sub=4, s…)` | Jin's recharge oscillator on (T, h) = (``target`` column, ``h_col``), fitted on the training segment. |
| `LIMForecaster` | classe | `(n_eofs=10, extra_cols=('wwv',), tau0=1, name=None)` | Linear inverse model dx/dt = L x + noise on x = (leading ``n_eofs`` PCs of the SST anomaly maps, the ``extra_cols`` indices), each standardised; L = log(C(tau0) C(0)^-1) / tau0 from the training segment; the target is read out by least squares on x. Forecast mean a . exp(L h) x(t), variance a . … |
| `MapCNNForecaster` | classe | `(n_lags=3, pool=2, scalar_cols=('wwv', 'h_west', 'h_east'), channels=(16, 32), hidden=64,…)` | CNN on the last ``n_lags`` monthly SST anomaly maps (coarsened by ``pool``) plus scalars (``scalar_cols`` at the origin and the sine / cosine of the origin month); outputs the standardised change of the target at every horizon (persistence = zero output). ``n_members`` seeds are averaged. |
| `IntermediateModelForecaster` | classe | `(init='nudged', n_members=1, enkf_members=24, coupling=1.0, eps_a=0.1, c=2.7, r_days=900.…)` | The coupled intermediate model fitted on the training segment (:func:`fit_intermediate`). |
| `EnsembleForecaster` | classe | `(members, n_draw=100, seed=0, name=None, family=None)` | Equal-weight mixture of fitted-together members: mean = mean of the members' means; samples = ``n_draw`` draws from each member's distribution (its samples, or its Gaussian). |
| `box_index` | função | `(maps, grid, box=(-5.0, 5.0, 190.0, 240.0)) -> np.ndarray` | cos-latitude weighted box mean of maps (..., ny, nx) on ``grid`` (default the Nino 3.4 box). |
| `ro_rhs` | função | `(x, R=-0.05, F1=0.02, F2=1.5, eps=0.03)` | Constant-coefficient recharge oscillator vector field on (T, h) for the forecast_lab ODE forecasters. |
| `fit_intermediate` | função | `(meta, end, *, eps_a=0.1, c=2.7, r_days=900.0, coupling=1.0, window=4, ridge=0.001, L_T=6…)` | Fit the coupled model to the rows [0, end) of the study data (``meta`` of :func:`.dataset.to_series`). |

#### `pinneapple_systems.time_series.enso.intermediate`

An intermediate coupled model of the tropical Pacific in the spirit of Zebiak & Cane (1987, Mon. Wea. Rev. 115:2262), simplified, and the checks that every component is solved correctly.

<small>`pinneapple_systems/time_series/enso/intermediate.py` · 607 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OceanGrid` | classe | `(lon0, lon1, dlon, lat_max, dlat)` | C-grid: h and T at cell centres (ny, nx); u on the nx + 1 west/east faces; v on the ny + 1 south/north faces. |
| `ShallowWaterOcean` | classe | `(grid=None, c=2.7, H=150.0, r_days=900.0, sponge_deg=5.0, sponge_days=5.0, dtype='float64…)` | Linear reduced-gravity shallow water on the equatorial beta-plane (dimensional, SI units). |
| `GillAtmosphere` | classe | `(eps=0.1, c_a=60.0, n_modes=24, dlon=2.5, lon_offset=0.0)` | Gill (1980) long-wave response in Matsuno units (y in L_a = sqrt(c_a / beta), x in L_a, Q per unit heating). |
| `CoupledParams` | classe | `(eps_T, gamma, kappa, mu, mu_y, L_T, noise_tau, noise_months, noise_lon, extra)` | Parameters of :class:`CoupledModel` (SST coefficients per ocean-grid longitude, in months). |
| `CoupledModel` | classe | `(params, ocean=None, atmosphere=None, dt_hours=8.0, atm_every=3, closure=None, atm_lat=40…)` | Linear coupled ocean-atmosphere anomaly model on :class:`OceanGrid`; time in months for the SST equation and seconds inside the ocean step. ``closure(T, h)`` (optional) adds a learned SST tendency [K/month] -- the hybrid models train it. |
| `gill_analytic` | função | `(x, y, L, eps, period) -> dict[str, np.ndarray]` | Closed-form long-wave Gill response on a periodic circle of length ``period`` (Matsuno units) to Q = F(x) exp(-y^2 / 2) with F = cos(pi x / (2 L)) for \|x\| < L, else 0. Kelvin part q0 = -int G_K(x - s) F(s) ds, Rossby part q2 = -1/2 int G_R(s - x) F(s) ds with the periodic Green's functions G(d) = … |
| `sst_tendency` | função | `(T, h, taux_c, eps_T, gamma, kappa, S)` | dT/dt [K/month] = -eps_T T + gamma S h + kappa S tau_x (T, h, tau_x at cell centres; gamma, kappa per longitude; S the meridional structure (ny, 1)). |
| `kelvin_wave_test` | função | `(grid=None, c=2.7, dt_hours=6.0, days=40.0, sigma_km=1500.0) -> dict` | Release an equatorial Kelvin wave (h = exp(-y^2 / 2 L^2 - (x - x0)^2 / 2 s^2), u = g' h / c, v = 0) in the western basin without damping; its speed is the slope of the tracked equatorial h peak. Theory: c; basin crossing time = basin width / c. |
| `rossby_wave_test` | função | `(grid=None, c=2.7, dt_hours=6.0, days=150.0, sigma_km=2500.0) -> dict` | Release a long n = 1 Rossby wave (Matsuno structure u ~ (2 y^2 - 3), h ~ (2 y^2 + 1) times exp(-y^2 / 2) in units of L = sqrt(c / beta)) in the eastern basin; its h centroid moves west at c / 3 in the long-wave limit (finite width makes it slightly slower). |
| `gill_test` | função | `(nlons=(72, 144, 288, 576), eps=0.1, L=2.0, n_modes=12) -> dict` | Numerical Gill response to Q = cos(pi x / 2L) exp(-y^2 / 2) on \|x\| < L against :func:`gill_analytic`, for several longitude resolutions (relative L2 error of u, v, p over the band \|y\| < 3). |
| `convergence_study` | função | `(factors=(1, 2, 4), c=2.7) -> dict` | Kelvin-wave speed error under joint refinement of dx, dy and dt (factor f: dlon = 5/f deg, dlat = 2/f deg, dt = 12/f h). |
| `forced_ocean_run` | função | `(ocean, taux, *, steps_per_month=91, tauy=None, state=None) -> tuple[np.ndarray, dict]` | Integrate the ocean alone under prescribed monthly stress ``taux`` (months, ny, nx + 1) [N m-2] (and ``tauy`` at v points); returns the monthly mean h (months, ny, nx) and the final state. NaN stress months are zero. |

#### `pinneapple_systems.time_series.enso.oscillators`

Conceptual ENSO models: the recharge oscillator and the delayed oscillator.

<small>`pinneapple_systems/time_series/enso/oscillators.py` · 264 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ROParams` | classe | `(coef, sigma_T, sigma_h, B, info)` | Recharge-oscillator coefficients per month: ``coef[name] = (c0, c1, c2)`` (annual harmonics; c1 = c2 = 0 for a constant coefficient), noise amplitudes per sqrt(month), multiplicative noise B [1/K]. |
| `ro_eigen` | função | `(R, F1, F2, eps) -> dict[str, float]` | Eigenvalues of the constant-coefficient recharge oscillator: growth rate [1/month], period [months] (inf when the eigenvalues are real) and the two eigenvalues. |
| `ro_coefficients` | função | `(p, month) -> dict[str, np.ndarray]` | Coefficients at (fractional) calendar month(s) ``month`` (0 = mid-January... any real number, period 12). |
| `ro_tendency` | função | `(p, X, month) -> np.ndarray` | Deterministic tendency (dT/dt, dh/dt) of states X (..., 2) at calendar month(s) ``month``. |
| `simulate_recharge` | função | `(p, x0, n_months, *, month0=0.0, sub=10, n_members=1, stochastic=True, seed=0, method='rk…)` | Integrate from x0 ((2,) or (n, 2) states) for ``n_months``; returns (n_members, n, n_months + 1, 2) samples at month ends. Deterministic part: RK4 (``method="rk4"``) or Euler; noise: Euler-Maruyama with ``sub`` steps per month (Ito; the multiplicative factor is clipped at zero). ``month_offsets`` … |
| `fit_recharge` | função | `(T, h, month=None, *, seasonal=(), multiplicative=False, lead=1, sub=4) -> ROParams` | Recharge oscillator fitted to monthly series T, h (pairs with a NaN are skipped). |
| `do_rhs` | função | `(x, xt, a=1.0, b=0.75, c=1.0)` | Battisti-Hirst / Suarez-Schopf delayed oscillator dT/dt = a T - b T(t - tau) - c T^3. |
| `do_roots` | função | `(a, b, tau, branches=(0, -1, 1)) -> list[complex]` | Roots of the linear characteristic equation lambda = a - b exp(-lambda tau) on Lambert-W ``branches``, sorted by decreasing real part. |
| `simulate_delayed` | função | `(a, b, c, tau, history, n_steps, dt, method='rk4') -> np.ndarray` | Method of steps on a uniform grid (``tau`` a multiple of ``dt``): ``history`` holds the last tau / dt + 1 values (oldest first); returns the n_steps new values. RK4 uses the mean of the delayed values at the half step (second order in the delayed argument) unless the grid is refined. |
| `oscillation_stats` | função | `(x, dt, *, skip=0) -> dict[str, float]` | Period (mean spacing of upward zero crossings, linearly interpolated) and growth rate (slope of log \|peak\| over successive extrema) of a linear oscillation sampled every ``dt``. |

#### `pinneapple_systems.time_series.enso.pinn`

Physics-informed networks for ENSO, trained with :class:`pinneapple_neural.trainer.inverse_pinn.InversePINN` (network weights and unknown physical parameters optimised together; data misfit plus ODE/PDE residual at random collocation points, adaptive weighting).

<small>`pinneapple_systems/time_series/enso/pinn.py` · 626 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FourierMLP` | classe | `(in_dim, out_dim, hidden=64, layers=3, n_freq=8, out_scale=None, freq_max=None)` | MLP on [x, sin(2 pi k x_j), cos(2 pi k x_j)] (k = 1..n_freq for each input j, inputs scaled to [0, 1]), tanh hidden layers; ``out_scale`` multiplies the outputs. |
| `GroupedMLP` | classe | `(groups, in_dim, out_dim, hidden=32, layers=2, n_freq=4, freq_max=2.0, out_scale=None, se…)` | G independent Fourier MLPs evaluated on the same inputs: (B, in) -> (B, G * out) (group-major). |
| `PINNRechargeForecaster` | classe | `(h_col='wwv', seasonal=(), closure=False, stochastic=False, n_members=200, epochs=4000, s…)` | Forecasts by integrating the recharge oscillator identified by :func:`identify_recharge` on the training segment (``closure``: with the learned thermocline-feedback network). ``stochastic``: ensemble with the noise of the one-month residuals; otherwise deterministic with a calibrated Gaussian … |
| `PINNExtrapolationForecaster` | classe | `(h_col='wwv', window=24, epochs=3000, n_col=512, hidden=32, chunk=256, seed=0, name=None)` | For each origin t: a PINN X(tau), tau = (s - t) / window in [-1, max(h) / window], fitted to the observed (T, h) of the last ``window`` months and to the recharge-oscillator residual (coefficients fitted by least squares on the training segment, held fixed) on the whole interval; the forecast at t … |
| `EquatorialWavePINN` | classe | `(lon_obs, h_obs, T_obs, lon_f, X0, X2, tau_eq, *, lon_w=120.0, lon_e=280.0, H=150.0, sst=…)` | PINN for the two-mode long-wave equatorial ocean plus an equatorial SST equation (see the module docstring). |
| `identify_recharge` | função | `(T, h, month=None, *, seasonal=(), closure=False, epochs=4000, n_col=2048, hidden=64, lay…)` | PINN identification of the recharge oscillator on monthly series (T, h) of length n. |
| `project_forcing` | função | `(taux, lat, c=2.7, H=150.0, rho=1025.0) -> tuple[np.ndarray, np.ndarray]` | Projections X0, X2 of X = tau_x / (rho H) [m s-2] (..., nlat, nx) on phi_0 = exp(-y^2/2) and phi_2 = (4 y^2 - 2) exp(-y^2/2), y = lat / L with L = sqrt(c / beta) (norms sqrt(pi) and 8 sqrt(pi)). |

#### `pinneapple_systems.time_series.enso.pipeline`

Study-level steps of the ENSO study, each returning JSON-able results (the example script only calls these).

<small>`pinneapple_systems/time_series/enso/pipeline.py` · 419 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `solver_checks` | função | `() -> dict[str, Any]` |  |
| `enso_scores` | função | `(merged, month_of_year, threshold=0.5) -> dict[str, Any]` | ACC per lead, target-month stratified ACC / RMSE and categorical scores of every model in ``merged``. |
| `case_panels` | função | `(parts, merged, data, *, families=…, events=None, months_before=18, leads=(1, 3, 6, 9, 12…)` | Plume panels: for every case-study origin, the representative of each family (from ``merged``) with its quantiles at the evaluated leads. ``parts[name]["_cases"]`` holds forecasts from origins after the development period; ``events[label][name]`` (from :func:`.study.event_hindcast`) those of … |
| `skill_against` | função | `(merged, reference='climatology', alpha=0.05) -> dict[str, Any]` | RMSE skill 1 - RMSE / RMSE(reference) on the test period for every model of ``merged`` (with its kept forecasts), and one-sided Diebold-Mariano tests (squared errors, HLN, long-run variance over the h-step overlap) of each non-baseline model against the reference, Holm-adjusted across those models … |
| `spring_barrier_hindcasts` | função | `(data, prot, makers, *, block=48) -> dict[str, Any]` | Blocked, purged cross-validated hindcasts over the development period for each maker; returns, per model, the target-month stratified ACC / RMSE and the ACC per lead. |
| `phase_portrait` | função | `(data, prot, *, seasonal=('R', 'F1', 'F2', 'eps'), seed=0, onsets=…, n_path=48) -> dict[s…)` | Observed (h, T) over the record (and its 3-month running mean), the deterministic paths of the fitted recharge oscillators (constant and seasonal) for ``n_path`` months from the observed state at each El Nino onset in ``onsets``, and one stochastic realisation of each with the record's length; … |
| `shallow_water_pinn_study` | função | `(ds, window=('1996-01', '1999-12'), *, epochs=6000, n_col=4096, pde_weight=100.0, spin_mo…)` | The equatorial-wave PINN (:class:`.pinn.EquatorialWavePINN`) on (a) synthetic equatorial h from the 2-D shallow-water model forced by ERA5 stress (known c = 2.7 m/s, r = 1/900 d, closed west wall) and (b) observed 2S-2N D20 and SST anomalies, both over ``window``; plus the Hovmoller fields … |
| `slab_frames` | função | `(ds, oisst, casts, start, end, *, spin_months=36, lon=(121.0, 279.0), lat=(-19.5, 19.5)) …)` | Monthly frames for :func:`.render.render_frames`: full SST [deg C] from OISST (1 deg) and thermocline depth = the D20 climatology of the casts (2 deg lat x 10 deg lon bins, base period, smoothed and interpolated to the SST grid) plus the anomaly of the ERA5-forced 2-D ocean model, scaled to D20 by … |
| `markdown_tables` | função | `(res, leads=(1, 3, 6, 9, 12, 18, 24)) -> dict[str, str]` | Markdown tables of the study from the stored results (``res``: the study JSON). |

#### `pinneapple_systems.time_series.enso.plots`

Figures of the ENSO study, in the style of :mod:`pinneapple_systems.time_series.forecast_lab.plots` (same family colours, grid and spines): forecast plumes, skill against lead, the spring-barrier heatmap, the recharge-oscillator phase portrait and Hovmoller diagrams.

<small>`pinneapple_systems/time_series/enso/plots.py` · 217 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_plumes` | função | `(panels, path, *, ncols=3) -> str` | Forecast distributions at the evaluated leads from one origin per panel, one family per colour (median marker, 50 % thick and 90 % thin interval bars, slightly offset per family), observed Nino 3.4 in black. Each panel: ``title``, ``t_obs``/``y_obs`` (months as fractional years), ``origin_t``, and … |
| `plot_skill_vs_lead` | função | `(curves, path, *, metrics=…, labels=None) -> str` | One panel per metric: each curve (``family``, ``horizons``, values, optional ``significant`` mask) against lead; baselines dashed grey. |
| `plot_spring_barrier` | função | `(maps, path, *, ncols=3, key='acc') -> str` | Heatmaps of ``key`` (ACC) by target month (rows) and lead (columns), one per model; cells with too few forecasts are blank, the count range is in each title. |
| `plot_phase_portrait` | função | `(traj, path) -> str` | (h, T) phase portraits: ``traj[name] = {"h", "T", "family" or "color", "lw", "alpha"}``; arrows show the direction of time on the first (observed) curve. |
| `plot_hovmoller` | função | `(panels, path, *, ncols=3) -> str` | Hovmoller diagrams (longitude x time): each panel ``title``, ``lon``, ``t`` (fractional years), ``z`` (nt, nlon), ``vmax``, ``units``. |
| `plot_validation` | função | `(kelvin, gill, conv, ro, path) -> str` | Solver checks: Kelvin-wave speed error under refinement, Gill error against resolution, recharge-oscillator period and growth (numerical vs analytic) against the time step. |

#### `pinneapple_systems.time_series.enso.render`

3-D renders of the tropical Pacific during an ENSO event with the library's Blender Cycles renderer (:func:`pinneapple_tools.visualization.studio.render`).

<small>`pinneapple_systems/time_series/enso/render.py` · 131 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `slab_scene` | função | `(lon, lat, sst, d20, *, d20_anom=None, band_deg=2.0, z_per_100m=1.0, x_per_deg=0.06, y_pe…)` | Scene of SST [deg C] (ny, nx) and thermocline depth ``d20`` [m, positive down] (ny, nx) on the same (lat, lon) grid (lon 0-360). The sea surface north of ``band_deg`` N and the equatorial band (``band_deg`` S to N) carry ``sst``; the band's southern wall drops from the surface to the thermocline … |
| `render_frames` | função | `(frames, out_dir, *, sst_range=(22.0, 31.0), size=(960, 540), samples=24, view=(-0.4, -1.…)` | Render ``frames`` (dicts with ``lon``, ``lat``, ``sst``, ``d20``, optionally ``d20_anom``, and a ``label``) to PNG files in ``out_dir``; ``interpolate`` > 1 inserts linearly interpolated frames between consecutive months. Needs ``bpy`` (``pip install bpy``) or a run inside Blender. |
| `encode_video` | função | `(paths, out, fps=8, crf=30) -> str` | H.264 MP4 (yuv420p, even frame size) from PNG frames with ffmpeg. |

#### `pinneapple_systems.time_series.enso.study`

The ENSO forecasting study: the protocol, the model zoo, and one call per model (so the zoo runs in parallel processes and is merged by :func:`~pinneapple_systems.time_series.forecast_lab.study.merge_zoo`).

<small>`pinneapple_systems/time_series/enso/study.py` · 173 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `protocol` | função | `(months, test_start='2012-01', horizons=HORIZONS, min_history=36) -> Protocol` |  |
| `zoo` | função | `(quick=False) -> list[ModelSpec]` | The model zoo. ``quick``: tiny epochs / members for smoke runs. |
| `spec` | função | `(name, quick=False) -> ModelSpec` |  |
| `run_model` | função | `(name, data, prot, *, budget=FP.BUDGET, quick=False, case_origins=None, log=print, seed=0…)` | Tune, fit and score one spec under ``prot`` (forecasts kept for :func:`merge_zoo`), plus its forecasts from the case-study origins that lie after the development period (``_cases``: label -> origin -> Forecast). |
| `event_hindcast` | função | `(name, data, event_start, origins, *, quick=False, horizons=HORIZONS, min_history=36, val…)` | Refit ``name`` (its first grid configuration) on data whose labels end before ``event_start`` and forecast from ``origins`` (for an event inside the development period, e.g. 1997/98, so no forecast uses a model that saw it). Returns the forecasts and the fitted model's description. |

#### `pinneapple_systems.time_series.features`

_Módulo sem docstring e sem texto._

<small>`pinneapple_systems/time_series/features/__init__.py` · 2 linhas · **sem teste direto**</small>

#### `pinneapple_systems.time_series.features.engineering`

_Módulo sem docstring e sem texto._

<small>`pinneapple_systems/time_series/features/engineering.py` · 110 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TSFeatureEngineer` | classe | `(lags, rolling_windows, rolling_stats_set, fourier_periods, fourier_K)` | Convenience wrapper to generate a feature dictionary. |
| `make_lags` | função | `(y, lags) -> Dict[str, np.ndarray]` | Create lag features for a 1D target series. |
| `rolling_stats` | função | `(y, windows, stats=('mean', 'std', 'min', 'max')) -> Dict[str, np.ndarray]` | Compute rolling statistics over the target series. |
| `fourier_features` | função | `(t, periods, K=1) -> Dict[str, np.ndarray]` | Fourier seasonality features: sin/cos harmonics for each period. |
| `rate_of_change` | função | `(y, orders=(1, 2)) -> Dict[str, np.ndarray]` | First and higher-order differences (velocity, acceleration). |
| `window_features` | função | `(y, input_len, horizon) -> tuple` | Build sliding-window (X, y_target) arrays for supervised ML. Returns: X      : (n_windows, input_len) y_out  : (n_windows, horizon) |

#### `pinneapple_systems.time_series.forecast_lab`

Forecasting near-random series without fooling yourself: one interface for baselines, data-driven, data + physics, physical and hybrid forecasters, and the protocol that compares them (purged walk-forward tuning, an untouched test period, point and probabilistic scores, Diebold-Mariano tests with …

<small>`pinneapple_systems/time_series/forecast_lab/__init__.py` · 44 linhas · testes: 50</small>

#### `pinneapple_systems.time_series.forecast_lab.base`

Common interface of the forecasters in :mod:`pinneapple_systems.time_series.forecast_lab`.

<small>`pinneapple_systems/time_series/forecast_lab/base.py` · 168 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SeriesData` | classe | `(obs, target, dt, name, meta)` | Observed series (T, d), the forecast target column, the sampling step and free-form metadata. |
| `Forecast` | classe | `(mean, std, samples)` | Predictive distribution at (origin, horizon): a mean, and either a Gaussian ``std`` or ``samples`` (members, origins, horizons). |
| `Forecaster` | classe | `()` | Base class: subclasses implement :meth:`_fit` and :meth:`_predict_mean` (or override :meth:`predict`). |
| `train_end` | função | `(train_origins, horizons) -> int` | Exclusive end index of the data that a fit on these origins may use. |
| `forecast_origins` | função | `(lo, hi, horizons, *, stride=1, min_history=0) -> np.ndarray` | Origins t with lo <= t, t >= min_history - 1 and t + max(h) < hi, every ``stride`` samples. |
| `targets_at` | função | `(data, origins, horizons) -> np.ndarray` |  |

#### `pinneapple_systems.time_series.forecast_lab.baselines`

Baselines every other forecaster must beat: persistence, climatology, AR(p), ARIMA, exponential smoothing, and for (near-)periodic series the seasonal-naive forecast and a phase-aligned mean cycle.

<small>`pinneapple_systems/time_series/forecast_lab/baselines.py` · 278 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Persistence` | classe | `()` | y_{t+h} = y_t (the random-walk forecast). |
| `Climatology` | classe | `()` | y_{t+h} = mean of the training segment, spread = its standard deviation. |
| `ARForecaster` | classe | `(max_order=20, criterion='aic', difference=False)` | Least-squares AR(p) on the target, order by AIC up to ``max_order``, iterated to each horizon. |
| `ARIMAForecaster` | classe | `(p=(0, 1, 2), d=(0, 1), q=(0, 1), max_fit_length=2000)` | ARIMA(p, d, q) (statsmodels SARIMAX), order by AIC over a small grid on the training segment; forecasts from every origin with the fitted parameters fixed, through the Kalman filter state at that origin (so a forecast uses data up to its origin only). The Gaussian predictive variance comes from the … |
| `ExponentialSmoothing` | classe | `()` | Simple exponential smoothing l_t = alpha y_t + (1 - alpha) l_{t-1}, flat forecast l_t; alpha minimises the one-step squared error on the training segment. |
| `SeasonalNaive` | classe | `(period=None)` | y_{t+h} = y_{t+h-kP} with k = ceil(h / P): the value one period (or the fewest whole periods) before the target. ``period`` in samples; by default the dominant period of the training segment (rounded to an integer number of samples). |
| `PhaseClimatology` | classe | `(period=None, fit_cycles=1.0)` | The mean cycle, aligned to the phase of each origin. |
| `dominant_period` | função | `(y, min_period=2.0, max_frac=0.25) -> float` | Period (in samples) of the highest periodogram peak of ``y`` after removing the mean and a linear trend, searched between ``min_period`` samples and ``max_frac`` of the series length. |

#### `pinneapple_systems.time_series.forecast_lab.hybrid`

Hybrid forecasters: a physical model plus something learned.

<small>`pinneapple_systems/time_series/forecast_lab/hybrid.py` · 181 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ResidualCorrectionForecaster` | classe | `(physical, corrector='gbm', n_lags=8, lag_step=1, max_iter=200, learning_rate=0.05, max_l…)` | physical forecast + corrector(lags, physical forecast). ``corrector`` is ``"gbm"`` or ``"mlp"``. |
| `UDEForecaster` | classe | `(physical, hidden=64, layers=2, window=4, epochs=200, patience=20, lr=0.001, weight_decay…)` | dx/dt = f_known(x; theta_hat) + s * g_NN((x - mu) / sd), trained through RK4 over ``window`` samples. |

#### `pinneapple_systems.time_series.forecast_lab.learned`

Data-driven direct forecasters on lagged windows: gradient boosting and neural networks (MLP, LSTM, TCN, transformer), with optional physics-derived input features and, for the MLP, a physics residual penalty.

<small>`pinneapple_systems/time_series/forecast_lab/learned.py` · 355 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LagFeatures` | classe | `(n_lags=8, lag_step=1, cols=None, extra=None)` | Lagged windows of the observed columns, plus optional extra features ``extra(data, origins) -> (n, k)``. |
| `GBMForecaster` | classe | `(n_lags=8, lag_step=1, max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_l…)` | Histogram gradient boosting (scikit-learn), one regressor per horizon, on lag features. |
| `NeuralForecaster` | classe | `(arch='mlp', n_lags=16, lag_step=1, hidden=64, layers=2, dropout=0.0, weight_decay=0.0, l…)` | Direct multi-horizon neural forecaster: ``arch`` in {"mlp", "lstm", "tcn", "transformer"}. |
| `PhysicsLossMLP` | classe | `(physical, lam=0.1, **kw)` | MLP that predicts the whole observed state at every step 1..max(h) and is penalised for violating a known vector field: loss = MSE + ``lam`` * mean \|(X_{k+1} - X_k) / dt - f((X_{k+1} + X_k) / 2)\|^2 / scale^2 along the predicted path (implicit-midpoint residual, second order in dt; X_0 is the … |
| `train_torch_regressor` | função | `(build, X_tr, Y_tr, X_va=None, Y_va=None, *, epochs=100, lr=0.001, weight_decay=0.0, batc…)` | Train ``build()`` on (X, Y) with :class:`pinneapple_neural.trainer.Trainer`. |

#### `pinneapple_systems.time_series.forecast_lab.physical`

Physical forecasters: the governing equations with parameters estimated from the training data only.

<small>`pinneapple_systems/time_series/forecast_lab/physical.py` · 564 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ODEForecaster` | classe | `(rhs, params, fit_params=None, sub=5, fit_steps=2, max_fit_origins=1500, divergence_facto…)` | dx/dt = rhs(x, **params) with every state variable observed (the columns of ``data.obs``). |
| `DelayODEForecaster` | classe | `(rhs, params, tau, fit_params=None, dt_int=0.1, fit_steps=2, max_fit_origins=1500, diverg…)` | dx/dt = rhs(x, x(t - tau), **params) for a scalar series sampled every dt; integrated with RK4 at ``dt_int`` (dt and tau must be multiples of it), the history taken from the observations by linear interpolation. |
| `LinearSSMForecaster` | classe | `(kind='ou', max_fit_length=4000, name=None, family=None)` | Linear Gaussian state-space model of a scalar observation y = mu + x_0 + noise. |
| `SDEDriftForecaster` | classe | `(degree=3, n_samples=200, sub=10, closure=None, seed=0, name=None, family=None)` | dx = f(x) dt + sigma dW for a scalar series, f(x) = sum_k c_k x^k (k <= ``degree``) fitted to the increments, optionally plus a neural closure g(x) fitted to what the polynomial leaves (``closure=dict(hidden=.., epochs=..)``). Forecast: ``n_samples`` Euler-Maruyama paths with ``sub`` substeps per … |
| `EnKFForecaster` | classe | `(physical, n_ens=50, inflation=1.02, q_frac=0.1, spinup=200, seed=0, name=None, family=No…)` | Stochastic EnKF over the observations with a fitted :class:`ODEForecaster` as the model. |
| `SINDyForecaster` | classe | `(threshold=0.05, l2=0.0001, library='poly', sub=5, divergence_factor=5.0, name=None, fami…)` | Equations discovered by SINDy (STLSQ, polynomial library of degree 2) on the training trajectory of a fully observed state; forecasts integrate them (RK4, ``sub`` substeps per sample) from the observed state. |
| `PhysicsFeatures` | classe | `(physical)` | Input features for a data-driven model computed by a physical forecaster (fitted on the training origins when the data model is fitted): its ``physics_features(data, origins)`` -- the fitted vector field or drift at the origin state, or the filtered state and energy of a linear model. |

#### `pinneapple_systems.time_series.forecast_lab.pipeline`

One call per study case, and the tables of the guide, built from the stored results.

<small>`pinneapple_systems/time_series/forecast_lab/pipeline.py` · 270 linhas · testes: 28</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `fan_panels` | função | `(zoo, data, title, n_origins=2, families=('physical', 'hybrid', 'data')) -> list[dict]` | Fan-chart data for the representatives of ``families`` at ``n_origins`` test origins (spread over the test period). |
| `surprise_record` | função | `(zoo, data, horizon_index=0) -> dict[str, Any] \| None` | Per test origin: the absolute error at one horizon of the best non-baseline model (by validation score) and of the best baseline at that horizon (by validation RMSE), with the target dates when ``data.meta["dates"]`` exists. Feeds :func:`~.plots.plot_surprise_calendar`. |
| `run_case` | função | `(name, *, n=6000, budget=BUDGET, seed=0, cache_dir=None, n_surrogates=99, log=print) -> d…)` |  |
| `run_overfitting` | função | `(name, *, n=6000, seed=0, cache_dir=None, schemes=…, log=print) -> dict[str, Any]` |  |
| `lyapunov_study` | função | `(dt_ints=(0.1, 0.05), total_times=(10000.0, 50000.0, 200000.0), seeds=(0, 1, 2), log=prin…)` | Convergence of the Benettin estimate of the Mackey-Glass (tau = 17) largest Lyapunov exponent with the integration step, the length of the run and the seed (initial history and perturbation direction). |
| `memorise_vs_learn` | função | `(name, variants=…, *, horizon_index=0, window=200, n=6000, seed=0, cache_dir=None) -> dic…)` | Refit two training variants of :func:`~.presets.overfitting_variants` on the final split of case ``name`` and return, for one horizon, the truth and both forecasts over the last ``window`` training origins and the first ``window`` test origins, with each variant's RMSE on the whole training and … |
| `markdown_tables` | função | `(results, overfit=()) -> dict[str, str]` | Markdown tables: predictability, skill horizon per family, scores of the representatives, the zoo (wall time, parameters), surrogate-trained twins, and the overfitting study. |

#### `pinneapple_systems.time_series.forecast_lab.plots`

Figures of a forecasting study (matplotlib, static PNG).

<small>`pinneapple_systems/time_series/forecast_lab/plots.py` · 268 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_skill_vs_horizon` | função | `(results, path, *, lyapunov=None, ncols=4, floor=-0.5, titles=None, reference='best_basel…)` | One panel per series: RMSE skill of each family's representative (chosen on the validation block) against the horizon, relative to ``reference`` -- ``"best_baseline"`` (at each horizon the baseline with the lowest validation RMSE) or ``"persistence"``; filled markers where the Holm-adjusted … |
| `plot_overfitting` | função | `(studies, path) -> str` | Per series: train, validation and test error of every training variant (bars), and the cross-validation estimates of each scheme (markers) for the variants where they were computed. |
| `plot_surrogate_tests` | função | `(reports, path, *, ncols=4) -> str` | Nonlinear prediction error of each series (vertical line) against its shuffled and IAAFT surrogates. |
| `plot_pit` | função | `(entries, path, *, ncols=4) -> str` | PIT histograms. ``entries``: dicts with ``title``, ``freq`` (relative frequencies) and ``deviation``. |
| `plot_fan_charts` | função | `(panels, path, *, ncols=2) -> str` | Forecast distributions on the test period, drawn only at the evaluated horizons: a marker at the median and error bars for the central 50 % (thick) and 90 % (thin) intervals, with no line joining the horizons (the forecasts are direct, made for those horizons only). Each panel: ``title``, … |
| `plot_memorise_vs_learn` | função | `(records, path) -> str` | For each record of :func:`~.pipeline.memorise_vs_learn`: truth and the two variants' forecasts over the end of the training period (left) and the start of the untouched test period (right). One shared legend; each panel title carries the variants' RMSE on the whole training or test set. |
| `calendar_grid` | função | `(dates, values) -> dict` | Arrange daily ``values`` on a week (column) x weekday (row) grid; days without a value are NaN. Returns the grid (7, n_weeks), the Monday of each week and the day range. |
| `plot_surprise_calendar` | função | `(records, path, *, vmax=1.0) -> str` | Calendar heatmaps, one per record (``title``, ``dates``, ``abs_error_model``, ``abs_error_baseline``, ``model``, ``baseline``, ``horizon``): each day coloured by log2(\|error of the model\| / \|error of the best baseline\|) -- blue where the model was closer, red where the baseline was, grey midpoint … |

#### `pinneapple_systems.time_series.forecast_lab.presets`

The series and model zoos of the near-random forecasting study (``examples/near_random_forecasting``).

<small>`pinneapple_systems/time_series/forecast_lab/presets.py` · 345 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Case` | classe | `(name, data, protocol, specs, truth, title)` |  |
| `lorenz63_missing_xz_rhs` | função | `(x, sigma=10.0, rho=28.0, beta=8.0 / 3.0)` | Lorenz-63 without the -x z term of dy/dt: a deliberately misspecified physical model. |
| `baseline_specs` | função | `(arima=True) -> list[ModelSpec]` |  |
| `data_specs` | função | `(lag_steps=(1,), lags=(8, 16), seq_lags=32, epochs=60, patience=8, max_train=3000, cols=N…)` | GBM, MLP, LSTM, TCN and transformer on lagged windows, each with a grid sampled to ``BUDGET`` configs. |
| `case` | função | `(name, *, n=6000, seed=0, cache_dir=None) -> Case` | Build the study case ``name`` (one of :data:`CASES`); ``n`` is the length of the synthetic series. |
| `overfitting_variants` | função | `(c, physics=False) -> dict[str, Any]` | Training variants of an MLP (and of gradient boosting) for :func:`~.study.overfitting_study`: from a deliberately overfit network (wide, deep, long window, no early stopping, no weight decay, 200 epochs) to small, regularised, early-stopped and ensembled ones. ``physics`` adds the physics-loss MLP … |

#### `pinneapple_systems.time_series.forecast_lab.renders`

3-D renders of a forecasting study (Blender Cycles through :mod:`pinneapple_blender.cycles`).

<small>`pinneapple_systems/time_series/forecast_lab/renders.py` · 314 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `landscape_tiles` | função | `(results, *, floor=-0.5, titles=None) -> list[dict[str, Any]]` | One tile per case result (from ``pipeline.run_case``): the families present, the horizon axis (log10 of Lyapunov times when the series is chaotic, else log10 of samples) and the skill against the best baseline of each family's representative at every horizon, clipped below at ``floor``. |
| `render_landscape` | função | `(tiles, path, *, ncols=4, resolution=(1600, 1000), samples=64, height_scale=1.2) -> str` | Draw the tiles as surfaces (x: horizon, y: family, z: skill) on a grid of ``ncols`` columns over a dark studio floor, each with a translucent sea at skill 0 bounded by a glowing waterline. The families share one labelled axis per row (a tile with a single family is a narrow strip on that family's … |
| `dense_forecast` | função | `(data, protocol, make, max_horizon, *, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict[s…)` | Fit ``make()`` on the protocol's final split with horizons 1..``max_horizon`` (every step) and forecast every test origin. Returns the quantiles at all test origins and the empirical coverage of the 50 % and 90 % bands per horizon (measured on the test period). |
| `render_cones` | função | `(lanes, path, *, resolution=(1280, 720), samples=64, frames_dir=None, n_frames=0, history…)` | Draw one lane per entry (``title``, ``dense`` from :func:`dense_forecast`, ``origin_pos`` index into its test origins, ``std`` for scaling), stacked vertically: the observed history as an emissive tube, the 90 % and 50 % bands as translucent ribbons, the median as a thin tube and what happened as a … |
| `embedding_views` | função | `(x, *, delay=None, n_points=3000, seed=0) -> dict[str, Any]` | Delay embeddings (dimension 3) of ``x``, of an IAAFT surrogate and of a shuffled surrogate, each standardised, plus the nonlinear prediction error (h = 1, NN predictor) of the three series. |
| `render_attractors` | função | `(views, path, *, resolution=(1280, 480), samples=64, titles=…) -> str` |  |
| `ensemble_fibres` | função | `(step, x0, *, n_members=50, rel_perturbation=0.01, n_steps=300, scale=None, seed=0, compo…)` | Integrate ``n_members`` copies of a state, each perturbed by ``rel_perturbation`` x ``scale`` (per component, default 1), with ``step(states) -> states`` acting on a batch (m, d). Returns the two chosen components over time for every member and the ensemble spread (std across members of the first … |
| `render_ensembles` | função | `(panels, path, *, resolution=(1280, 900), samples=48) -> str` | One lane per panel (``title``, ``fibres`` from :func:`ensemble_fibres`, ``times`` (n,), ``lt_time`` or None, ``family`` colour key), stacked vertically: members as thin tubes with time along x, the first component as height and the second as depth, and a translucent wall at one Lyapunov time. |

#### `pinneapple_systems.time_series.forecast_lab.study`

The evaluation protocol: fix the test period first, tune on purged walk-forward folds of the rest, fit once, score once on the untouched test period, and test the differences.

<small>`pinneapple_systems/time_series/forecast_lab/study.py` · 450 linhas · testes: 50</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Protocol` | classe | `(horizons, test_frac, val_frac, n_folds, fold_frac, inner_val_frac, min_history, test_str…)` | Split of a series of length ``n`` into development (train + validation) and an untouched test period. |
| `ModelSpec` | classe | `(name, family, factory, grid)` | A forecaster in the zoo: ``factory(**config)`` builds an unfitted model; ``grid`` lists the configurations to try (one entry means no tuning). |
| `tune` | função | `(spec, data, protocol, *, budget=None, seed=0) -> dict[str, Any]` | Score every configuration (or ``budget`` of them, drawn at random) by purged walk-forward CV. |
| `evaluate` | função | `(model, data, origins, *, levels=LEVELS, keep=False) -> dict[str, Any]` | Point and probabilistic scores per horizon on ``origins``. |
| `skill_horizon` | função | `(horizons, skill, p_values=None, alpha=0.05) -> int \| None` | Largest horizon h such that the skill is positive (and, with ``p_values``, significant) at every horizon up to h; None when there is no skill at the first horizon. |
| `run_zoo` | função | `(data, specs, protocol, *, budget=None, keep_forecasts=(), keep_models=False, log=print, …)` | Tune, fit and score every model of the zoo on one series. Returns a JSON-able dict (``_forecasts`` holds the test forecasts of ``keep_forecasts`` for figures, ``_models`` the fitted models when ``keep_models``; both are dropped by ``to_jsonable(drop_private=True)``). Zoos run in parts (e.g. one … |
| `merge_zoo` | função | `(parts) -> dict[str, Any]` | Combine :func:`run_zoo` results of the same series and protocol run in parts (each with its models' test forecasts kept): the models and forecasts are pooled and the comparisons (skill, DM tests against persistence and the best baseline, Holm and BH adjustments, family representatives) are … |
| `overfitting_study` | função | `(data, protocol, variants, *, schemes=('purged_walk_forward',), schemes_for=None, log=pri…)` | For each training variant: its error on its training origins, on the validation block and on the untouched test period, the gap test - train, and the cross-validation estimate under each scheme with its optimism (test - CV). The error is the mean over horizons of RMSE / RMSE of persistence *on the … |
| `surrogate_trained` | função | `(make, data, protocol, *, kind='iaaft', seed=0) -> dict[str, Any]` | Skill (vs persistence, mean over horizons and per horizon) on the real test period of a univariate model trained on the real development data and of the same model trained on a surrogate of it. |

#### `pinneapple_systems.time_series.foundation_models`

Pretrained time-series *foundation model* forecasters.

<small>`pinneapple_systems/time_series/foundation_models.py` · 379 linhas · testes: 14 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ChronosForecaster` | classe | `(model_id='amazon/chronos-t5-tiny', device_map='cpu', dtype=None, **pipeline_kwargs)` | Wraps Amazon's real, pretrained Chronos forecasting pipeline. |
| `TimesFMForecaster` | classe | `(model_id=…, max_context=512, max_horizon=128, torch_compile=False, **forecast_config_kwa…)` | Wraps Google's real, pretrained TimesFM forecasting model. |

#### `pinneapple_systems.time_series.losses`

Time series forecast loss functions.

<small>`pinneapple_systems/time_series/losses.py` · 9 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `mse_forecast` | função | `(model, y_hat, batch)` |  |

#### `pinneapple_systems.time_series.metrics`

Time series metrics for forecasting evaluation.

<small>`pinneapple_systems/time_series/metrics.py` · 104 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `mae` | função | `(y_hat, batch)` | Mean Absolute Error. |
| `mse` | função | `(y_hat, batch)` | Mean Squared Error. |
| `rmse` | função | `(y_hat, batch)` | Root Mean Squared Error. |
| `mape` | função | `(y_hat, batch)` | Mean Absolute Percentage Error (%). |
| `smape` | função | `(y_hat, batch)` | Symmetric Mean Absolute Percentage Error (%).  Bounded in [0, 200]. |
| `r2` | função | `(y_hat, batch)` | Coefficient of determination R².  R²=1 → perfect; R²<0 → worse than mean. |
| `mae_fn` | função | `(y_hat, y) -> torch.Tensor` |  |
| `mse_fn` | função | `(y_hat, y) -> torch.Tensor` |  |
| `rmse_fn` | função | `(y_hat, y) -> torch.Tensor` |  |
| `mape_fn` | função | `(y_hat, y) -> torch.Tensor` | Mean Absolute Percentage Error (%). |
| `smape_fn` | função | `(y_hat, y) -> torch.Tensor` |  |
| `r2_fn` | função | `(y_hat, y) -> torch.Tensor` |  |
| `default_ts_metrics` | função | `()` |  |

#### `pinneapple_systems.time_series.metrics_ext`

_Módulo sem docstring e sem texto._

<small>`pinneapple_systems/time_series/metrics_ext/__init__.py` · 28 linhas · testes: 20</small>

#### `pinneapple_systems.time_series.metrics_ext.comparison`

Comparing forecasts: skill scores, the Diebold-Mariano test and multiple-comparison control.

<small>`pinneapple_systems/time_series/metrics_ext/comparison.py` · 127 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DMResult` | classe | `(statistic, p_value, p_value_better, mean_loss_difference, n, max_lag)` |  |
| `skill_score` | função | `(score, reference) -> np.ndarray` | 1 - score / reference, elementwise (for errors and other negatively oriented scores). |
| `diebold_mariano` | função | `(e1, e2, *, h=1, loss='se', max_lag=None, hln=True) -> DMResult` | Test E[L(e1)] = E[L(e2)] for two error series on the same targets (in time order). |
| `holm` | função | `(p_values)` | Holm step-down adjusted p-values (same keys / order as given); NaNs are left out of the family. |
| `benjamini_hochberg` | função | `(p_values)` | Benjamini-Hochberg adjusted p-values (q-values); NaNs are left out of the family. |

#### `pinneapple_systems.time_series.metrics_ext.point`

_Módulo sem docstring e sem texto._

<small>`pinneapple_systems/time_series/metrics_ext/point.py` · 81 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PointMetrics` | classe | `(season_length)` |  |
| `mae` | função | `(y_pred, y_true) -> float` |  |
| `rmse` | função | `(y_pred, y_true) -> float` |  |
| `mape` | função | `(y_pred, y_true, eps=1e-08) -> float` |  |
| `smape` | função | `(y_pred, y_true, eps=1e-08) -> float` |  |
| `mase` | função | `(y_pred, y_true, y_train, season_length=1, eps=1e-08) -> float` | Mean Absolute Scaled Error (MASE). |
| `multi_horizon_mae` | função | `(y_pred, y_true) -> Dict[str, float]` | Overall MAE and MAE per horizon step for (N,H) or (N,H,C). |

#### `pinneapple_systems.time_series.metrics_ext.probabilistic`

_Módulo sem docstring e sem texto._

<small>`pinneapple_systems/time_series/metrics_ext/probabilistic.py` · 109 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ProbabilisticMetrics` | classe | `(quantiles)` |  |
| `pinball_loss` | função | `(y_pred_q, y_true, q) -> float` |  |
| `coverage` | função | `(y_lo, y_hi, y_true) -> float` |  |
| `mean_interval_width` | função | `(y_lo, y_hi) -> float` |  |
| `pit_gaussian` | função | `(mu, sigma, y) -> np.ndarray` |  |
| `pit_ensemble` | função | `(samples, y, rng=None) -> np.ndarray` | Randomised PIT of an ensemble (members on axis 0): ties and the discreteness of the empirical CDF are broken uniformly at random, so a calibrated ensemble gives exactly uniform values. |
| `pit_histogram` | função | `(pit, n_bins=10) -> Dict[str, object]` | Relative frequencies of the PIT in ``n_bins`` equal bins and their mean absolute deviation from 1 / n_bins (0 for a perfectly flat histogram), divided by 1 / n_bins. |
| `gaussian_interval_coverage` | função | `(mu, sigma, y, level=0.9) -> float` |  |
| `ensemble_interval_coverage` | função | `(samples, y, level=0.9) -> float` |  |

#### `pinneapple_systems.time_series.models`

_Módulo sem docstring e sem texto._

<small>`pinneapple_systems/time_series/models/__init__.py` · 29 linhas · **sem teste direto**</small>

#### `pinneapple_systems.time_series.models.classical`

Classical ML forecasting models: XGBoost, LightGBM, RandomForest, CatBoost, GPR.

<small>`pinneapple_systems/time_series/models/classical.py` · 315 linhas · testes: 151 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ForecastModel` | classe | `()` |  |
| `XGBoostForecaster` | classe | `(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, …)` | XGBoost multi-output regressor (wraps MultiOutputRegressor for horizon>1). |
| `LightGBMForecaster` | classe | `(n_estimators=300, num_leaves=31, learning_rate=0.05, subsample=0.8, **kwargs)` |  |
| `RandomForestForecaster` | classe | `(n_estimators=200, max_depth=None, min_samples_leaf=1, **kwargs)` |  |
| `CatBoostForecaster` | classe | `(iterations=300, depth=6, learning_rate=0.05, **kwargs)` |  |
| `GPRForecaster` | classe | `(kernel=None, alpha=0.0001, normalize_y=True, **kwargs)` | Gaussian Process Regressor — natively probabilistic. Best for small datasets (< ~5000 samples). Multi-output: one GP per output dimension. |
| `MLPForecaster` | classe | `(hidden_layer_sizes=(128, 64), activation='relu', max_iter=1000, learning_rate_init=0.001…)` |  |

#### `pinneapple_systems.time_series.models.fno_forecaster`

FNO-1D forecaster for time series multi-horizon prediction.

<small>`pinneapple_systems/time_series/models/fno_forecaster.py` · 73 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FNOForecastConfig` | classe | `(input_len, horizon, n_features, n_targets, width, modes, layers)` |  |
| `FNOForecaster` | classe | `(cfg)` | Forecaster TS padrão usando FNO-1D. Convenção: x: (B, L_in, F) y_hat: (B, H, C) |

#### `pinneapple_systems.time_series.models.nbeats`

N-BEATS: Neural Basis Expansion Analysis for Time Series (Oreshkin et al., 2019).

<small>`pinneapple_systems/time_series/models/nbeats.py` · 127 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NBeatsConfig` | classe | `(input_len, horizon, n_features, stack_types, n_blocks, n_layers, layer_width, degree_of_…)` |  |
| `NBeats` | classe | `(cfg=None, **kw)` | N-BEATS with trend + seasonality + generic stacks. Input  : (B, L, F) Output : TSOutput with y_hat (B, H, 1) |

#### `pinneapple_systems.time_series.models.recurrent`

LSTM and GRU forecasters (PyTorch, GPU-ready).

<small>`pinneapple_systems/time_series/models/recurrent.py` · 64 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RecurrentConfig` | classe | `(input_len, horizon, n_features, n_targets, hidden_size, num_layers, dropout, bidirection…)` |  |
| `LSTMForecaster` | classe | `(cfg=None, **kw)` | LSTM-based multi-step forecaster. |
| `GRUForecaster` | classe | `(cfg=None, **kw)` | GRU-based multi-step forecaster. |

#### `pinneapple_systems.time_series.models.tcn`

Temporal Convolutional Network (TCN) forecaster.

<small>`pinneapple_systems/time_series/models/tcn.py` · 74 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TCNConfig` | classe | `(input_len, horizon, n_features, n_targets, n_channels, n_layers, kernel_size, dropout)` |  |
| `TCNForecaster` | classe | `(cfg=None, **kw)` | TCN with exponentially increasing dilations (1,2,4,8,...). Input  : (B, L, F) Output : TSOutput with y_hat (B, H, n_targets) |

#### `pinneapple_systems.time_series.models.tft`

Temporal Fusion Transformer (Lim et al., 2021) — simplified, GPU-ready.

<small>`pinneapple_systems/time_series/models/tft.py` · 197 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TFTConfig` | classe | `(input_len, horizon, n_features, n_targets, hidden_size, num_heads, num_lstm_layers, drop…)` |  |
| `TFTForecaster` | classe | `(cfg=None, **kw)` | Temporal Fusion Transformer. Input  : (B, L, F) Output : TSOutput with y_hat (B, H, n_targets) |

#### `pinneapple_systems.time_series.predictability`

Is there anything to predict? Measures of predictability to compute before fitting any forecaster.

<small>`pinneapple_systems/time_series/predictability.py` · 350 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ARFit` | classe | `(order, intercept, coef, sigma, r2_one_step, criterion)` |  |
| `SurrogateTest` | classe | `(statistic, kind, observed, surrogates, p_value, alternative, reject_at_0_05)` |  |
| `fit_ar` | função | `(x, max_order=20, criterion='aic') -> ARFit` | Least-squares AR(p), p = 0..max_order, all fitted on the same sample (the first ``max_order`` values are dropped for every p) so the information criteria are comparable. |
| `permutation_entropy` | função | `(x, order=5, delay=1, normalize=True) -> float` | Shannon entropy of the ordinal patterns of length ``order`` (ties broken by position). |
| `sample_entropy` | função | `(x, m=2, r=None) -> float` | -log(A / B): B = template pairs of length m within tolerance r (Chebyshev), A = the same for m + 1; self matches excluded and the same N - m templates used for both lengths. ``r`` defaults to 0.2 std(x). |
| `delay_embedding` | função | `(x, dim, delay) -> np.ndarray` |  |
| `first_acf_zero` | função | `(x, max_lag=None) -> int` | First lag at which the autocorrelation drops below 1/e (a common embedding-delay rule). |
| `rosenstein_lyapunov` | função | `(x, *, dim=5, delay=None, dt=1.0, min_tsep=None, n_steps=40, fit=(5, 30), max_points=3000…)` | Largest Lyapunov exponent per unit time from a scalar series. |
| `shuffle_surrogate` | função | `(x, rng) -> np.ndarray` | Random permutation: keeps the amplitude distribution, destroys all temporal structure. |
| `phase_randomized_surrogate` | função | `(x, rng) -> np.ndarray` | Fourier-transform surrogate: same power spectrum, random phases (a linear Gaussian process). |
| `iaaft_surrogate` | função | `(x, rng, n_iter=200, tol=1e-08) -> np.ndarray` | Iterative amplitude-adjusted Fourier transform surrogate (Schreiber & Schmitz 1996): the same amplitude distribution as ``x`` exactly and nearly the same power spectrum. |
| `nonlinear_prediction_error` | função | `(x, *, dim=4, delay=1, horizon=1, k=10, train_frac=0.5) -> float` | Out-of-sample error of a local-constant (nearest-neighbour) predictor, divided by std(x). |
| `time_reversal_asymmetry` | função | `(x, lag=1) -> float` | Normalised third moment of the lag-``lag`` increments; 0 in expectation for any time-reversible process (every linear Gaussian process is). |
| `surrogate_test` | função | `(x, statistic, *, kind='iaaft', n_surrogates=99, alternative='less', seed=0, name=None) -…)` | Rank test of ``statistic(x)`` against ``n_surrogates`` surrogates of the given ``kind``. |
| `baseline_errors` | função | `(x, horizons, train_frac=0.8) -> dict[str, dict[int, float]]` | RMSE of persistence (x_{t+h} = x_t) and climatology (x_{t+h} = mean of the first ``train_frac``) on the last 1 - ``train_frac`` of the series, per horizon, both divided by std(x). |
| `predictability_report` | função | `(x, *, horizons=(1, 2, 4, 8, 16, 32), dt=1.0, max_ar_order=20, pe_order=5, embed_dim=4, n…)` | Predictability of a scalar series before any modelling. |

#### `pinneapple_systems.time_series.preparation`

_Módulo sem docstring e sem texto._

<small>`pinneapple_systems/time_series/preparation/__init__.py` · 6 linhas · **sem teste direto**</small>

#### `pinneapple_systems.time_series.preparation.imputer`

Missing value treatment with layered interpolation strategies.

<small>`pinneapple_systems/time_series/preparation/imputer.py` · 126 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TimeSeriesImputer` | classe | `(strategy='layered', max_gap=None, verbose=False)` | Layered missing-value imputer for time series. |

#### `pinneapple_systems.time_series.preparation.outliers`

Outlier detection and treatment for time series.

<small>`pinneapple_systems/time_series/preparation/outliers.py` · 155 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OutlierDetector` | classe | `(method='rolling_iqr', window=20, iqr_factor=1.5, z_threshold=3.0, treatment='interpolate…)` | Outlier detection (Rolling IQR / Z-Score) with 6 treatment options. |

#### `pinneapple_systems.time_series.preparation.resampler`

Automatic upsampling / downsampling for numeric time-indexed DataFrames.

<small>`pinneapple_systems/time_series/preparation/resampler.py` · 122 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TimeSeriesResampler` | classe | `(target_freq, time_col=None, agg_method='mean', upsample_method='linear', numeric_only=Tr…)` | Resample a time-indexed DataFrame to a target frequency. |

#### `pinneapple_systems.time_series.problem`

_Módulo sem docstring e sem texto._

<small>`pinneapple_systems/time_series/problem.py` · 54 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ForecastProblemSpec` | classe | `(freq, time_col, target_cols, feature_cols, exog_past_cols, exog_future_cols, input_len, …)` | Formal forecasting problem definition. |

#### `pinneapple_systems.time_series.registry`

Time series model catalog and default FNO forecaster registration.

<small>`pinneapple_systems/time_series/registry.py` · 82 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TSModelCatalog` | classe | `(families, default_name)` |  |

#### `pinneapple_systems.time_series.solar_cycle`

Solar-cycle envelope from the daily sunspot number: (E)EMD, the IMF nearest 11 years, its Hilbert amplitude, and a fit of that amplitude by an extreme learning machine (ELM) with the end-point equality f(0) = f(1) built into the basis (a constrained expression of the Theory of Functional …

<small>`pinneapple_systems/time_series/solar_cycle.py` · 274 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Decomposition` | classe | `(imfs, method, backend, trials, noise_width)` |  |
| `HilbertResult` | classe | `(amplitude, phase, instantaneous_frequency)` |  |
| `PeriodicFit` | classe | `(rmse, fitted, forecast_t, forecast, hidden_units, backend, endpoint_error)` |  |
| `parse_silso_csv` | função | `(source)` | A SILSO daily CSV (path, file-like or bytes; ';'-separated, no header, -1 = missing) to a DataFrame with a DatetimeIndex and the non-missing ``Sunspot`` values. |
| `fetch_silso_daily` | função | `(url=SILSO_DAILY_URL, timeout=30.0)` | Download and parse the current SILSO daily series (network; raises on failure, never substitutes data). |
| `synthetic_daily` | função | `(start='2005-01-01', end='2025-06-30', *, period_years=11.0, peak=180.0, seed=0)` | A SILSO-shaped synthetic daily series: a sin^2 cycle of ``period_years`` (minima 3.9 years after ``start``, as 2008.9 is after 2005), ~27-day rotational modulation and multiplicative noise, clipped at 0. Not solar data; for tests and offline demos. |
| `preprocess` | função | `(df)` | ``(t, y_scaled, (y_min, y_max))``: t = linspace(0, 1, N), y min-max scaled to [0, 1]. |
| `decompose` | função | `(y, method='emd', *, max_imf=6, trials=16, noise_width=0.05, seed=0, backend='auto') -> D…)` | EMD (deterministic) or EEMD (noise-assisted ensemble of ``trials`` runs) of a 1-D series. |
| `imf_statistics` | função | `(imfs, dt_days=1.0) -> list[dict[str, Any]]` | Per IMF: dominant period (days and years, from the median positive Hilbert instantaneous frequency), mean instantaneous amplitude and share of the variance of the summed signal. |
| `closest_to_schwabe_cycle` | função | `(stats) -> tuple[dict \| None, str \| None]` | The IMF whose period is nearest 11 years, and a warning when it is more than 40 % off. |
| `hilbert_analysis` | função | `(imf, t) -> HilbertResult` | Instantaneous amplitude, unwrapped phase and frequency of one IMF. |
| `periodic_elm_fit` | função | `(t, y, *, hidden_units=50, forecast_horizon_frac=1.0, n_forecast_points=None, seed=0, bac…)` | Least-squares ELM fit of ``y(t)`` on t in [0, 1] with f(0) = f(1) exact, extended to [1, 1 + horizon]. |
| `run_pipeline` | função | `(df, *, decomposition_method='emd', max_imf=6, eemd_trials=16, eemd_noise_width=0.05, imf…)` | Steps 1-4 of the module docstring on a daily DataFrame (``Sunspot`` column, DatetimeIndex). |

#### `pinneapple_systems.time_series.spec`

Time series specification (input_len, horizon, stride, target_offset).

<small>`pinneapple_systems/time_series/spec.py` · 13 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TimeSeriesSpec` | classe | `(input_len, horizon, stride, target_offset)` | Specification for time series input length, horizon, stride, and target offset. |

#### `pinneapple_systems.time_series.synthetic`

Synthetic time series with known ground truth, for forecasting studies.

<small>`pinneapple_systems/time_series/synthetic.py` · 364 linhas · testes: 50</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SyntheticSeries` | classe | `(name, t, state, obs, dt, target, observed, params, noise_std, noise_level, lyapunov, kin…)` | A synthetic series and its ground truth. |
| `rk4_step` | função | `(f, x, dt)` | One classical Runge-Kutta step of ``dx/dt = f(x)`` (NumPy or torch, batched over leading axes). |
| `lorenz63_rhs` | função | `(x, sigma=10.0, rho=28.0, beta=8.0 / 3.0)` |  |
| `lorenz96_rhs` | função | `(x, F=8.0)` |  |
| `mackey_glass_rhs` | função | `(x, x_tau, beta=0.2, gamma=0.1, n=10.0)` |  |
| `double_well_drift` | função | `(x, a=1.0, b=1.0)` | Drift of a particle in the quartic potential V(x) = -a x^2 / 2 + b x^4 / 4 (minima at +-sqrt(a / b)). |
| `mackey_glass_step` | função | `(hist, dt, beta=0.2, gamma=0.1, n=10.0) -> np.ndarray` | Advance Mackey-Glass histories by one RK4 step of size ``dt`` (method of steps). |
| `largest_lyapunov_benettin` | função | `(step, x0, *, step_time, n_steps=20000, transient=2000, d0=1e-08, seed=0) -> float` | Largest Lyapunov exponent (per unit time) of a deterministic map ``x -> step(x)`` by two-trajectory renormalisation (Benettin, Galgani, Giorgilli & Strelcyn 1980, Meccanica 15:9). |
| `mackey_glass_lyapunov` | função | `(*, dt_int=0.1, total_time=50000.0, transient_time=2000.0, tau=17.0, beta=0.2, gamma=0.1,…)` | Largest Lyapunov exponent (per unit time, natural log) of the Mackey-Glass delay equation. |
| `lorenz63` | função | `(n=6000, *, dt=0.05, sub=5, noise=0.1, seed=0, sigma=10.0, rho=28.0, beta=8.0 / 3.0, obse…)` | Lorenz-63 sampled every ``dt`` (RK4 with ``sub`` substeps); observation noise ``noise`` x std per column. |
| `lorenz96` | função | `(n=6000, *, K=40, F=8.0, dt=0.05, sub=5, noise=0.1, seed=0, lyapunov_steps=20000) -> Synt…)` | Lorenz-96 with ``K`` variables, every variable observed, target variable 0. |
| `mackey_glass` | função | `(n=6000, *, dt=3.0, dt_int=0.1, tau=17.0, beta=0.2, gamma=0.1, n_exp=10.0, noise=0.1, see…)` | Mackey-Glass sampled every ``dt``, integrated with RK4 at ``dt_int`` (tau must be a multiple of dt_int). |
| `ornstein_uhlenbeck` | função | `(n=6000, *, dt=0.1, theta=1.0, sigma=1.0, noise=0.1, seed=0) -> SyntheticSeries` | Exact sampling: x_{k+1} = e^{-theta dt} x_k + sqrt(sigma^2 (1 - e^{-2 theta dt}) / (2 theta)) eps. |
| `oscillator_transition` | função | `(omega, zeta, sigma, dt)` | Exact discretisation of d[x, v] = A [x, v] dt + [0, sigma] dW with A = [[0, 1], [-omega^2, -2 zeta omega]]. |
| `stochastic_oscillator` | função | `(n=6000, *, dt=0.25, omega=1.0, zeta=0.05, sigma=0.3, noise=0.1, seed=0) -> SyntheticSeri…)` | Damped oscillator driven by white noise, position observed with noise (velocity hidden). |
| `double_well` | função | `(n=6000, *, dt=0.1, sigma=0.7, sub=100, a=1.0, b=1.0, noise=0.1, seed=0) -> SyntheticSeri…)` | Double-well Langevin dynamics (Euler-Maruyama with ``sub`` substeps per sample): two regimes at +-1. |
| `white_noise` | função | `(n=6000, *, seed=0) -> SyntheticSeries` |  |
| `random_walk` | função | `(n=6000, *, seed=0) -> SyntheticSeries` |  |
| `ode_stepper` | função | `(rhs, dt, sub=1) -> Callable[[np.ndarray], np.ndarray]` | A map advancing a batch of states (m, d) of dx/dt = rhs(x) by ``dt`` (``sub`` RK4 substeps). |
| `mackey_glass_stepper` | função | `(dt_int=0.1, sub=10, **params) -> Callable[[np.ndarray], np.ndarray]` | A map advancing Mackey-Glass histories (m, tau / dt_int + 1) by ``sub`` RK4 steps of ``dt_int``. |
| `linear_sde_stepper` | função | `(Phi, Q, seed=0) -> Callable[[np.ndarray], np.ndarray]` | A map x -> Phi x + w for a batch of states, with one noise draw w ~ N(0, Q) per call shared by the whole batch: members that differ only in their initial state (sensitivity to initial conditions, same forcing). |

#### `pinneapple_systems.time_series.tuning`

_Módulo sem docstring e sem texto._

<small>`pinneapple_systems/time_series/tuning/__init__.py` · 4 linhas · **sem teste direto**</small>

#### `pinneapple_systems.time_series.tuning.optuna_tuner`

Optuna-based HPO for all timeseries forecasters (neural + classical).

<small>`pinneapple_systems/time_series/tuning/optuna_tuner.py` · 265 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ClassicalTuner` | classe | `(model_type='xgboost', n_trials=30, direction='minimize', metric='mae', timeout=None)` | Optuna HPO wrapper for classical ML forecasters. |
| `NeuralTuner` | classe | `(model_type='lstm', n_trials=20, epochs_per_trial=20, lr=0.001, batch_size=64, input_len=…)` | Optuna HPO for PyTorch forecasters. |
| `temporal_split` | função | `(X, y, val_frac=0.15, test_frac=0.15)` | Chronological 3-way split → (X_tr, y_tr, X_val, y_val, X_te, y_te). |

#### `pinneapple_systems.time_series.uncertainty`

_Módulo sem docstring e sem texto._

<small>`pinneapple_systems/time_series/uncertainty/__init__.py` · 2 linhas · **sem teste direto**</small>

#### `pinneapple_systems.time_series.uncertainty.quantile`

_Módulo sem docstring e sem texto._

<small>`pinneapple_systems/time_series/uncertainty/quantile.py` · 8 linhas · **sem teste direto**</small>

#### `pinneapple_systems.time_series.validation`

_Módulo sem docstring e sem texto._

<small>`pinneapple_systems/time_series/validation/__init__.py` · 16 linhas · testes: 20</small>

#### `pinneapple_systems.time_series.validation.backtest`

_Módulo sem docstring e sem texto._

<small>`pinneapple_systems/time_series/validation/backtest.py` · 111 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BacktestConfig` | classe | `(reset_model_each_fold, shuffle_train, val_batch_size)` | Backtesting settings. |
| `BacktestResult` | classe | `(folds, agg)` |  |
| `BacktestRunner` | classe | `(*, trainer_factory, model_factory, datamodule, splits, train_cfg, preprocess_factory=Non…)` | Train/evaluate a model across a sequence of time-series splits. |

#### `pinneapple_systems.time_series.validation.splitters`

_Módulo sem docstring e sem texto._

<small>`pinneapple_systems/time_series/validation/splitters.py` · 163 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Split` | classe | `(fold, train_idx, val_idx, test_idx)` | A single backtesting fold expressed as indices of the windowed dataset. |
| `ExpandingWindowSplitter` | classe | `(initial_train_size, val_size, step_size, gap, max_folds)` | Walk-forward validation with an expanding training window. |
| `RollingWindowSplitter` | classe | `(train_size, val_size, step_size, gap, max_folds)` | Walk-forward validation with a fixed-size rolling training window. |
| `PurgedWalkForwardSplitter` | classe | `(n_folds, val_size, purge, embargo, window, min_train)` | Walk-forward (rolling-origin) validation over forecast origins with purge and embargo. |
| `PurgedKFoldSplitter` | classe | `(n_folds, purge, embargo)` | K contiguous folds; training on both sides of the validation fold with purge before and embargo after. |
| `ShuffledKFoldSplitter` | classe | `(n_folds, seed)` | Ordinary shuffled k-fold, which ignores time: kept to measure how optimistic it is on serial data. |

#### `pinneapple_systems.time_series.viz`

_Módulo sem docstring e sem texto._

<small>`pinneapple_systems/time_series/viz/__init__.py` · 18 linhas · **sem teste direto**</small>

#### `pinneapple_systems.time_series.viz.animation`

Animated rolling forecast visualization with optional MP4 export.

<small>`pinneapple_systems/time_series/viz/animation.py` · 114 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `animate_rolling_forecast` | função | `(y_true, y_pred_seq, y_std_seq=None, horizon=16, context_len=64, interval=80, sigmas=(1.0…)` | Create an animated rolling forecast. |

#### `pinneapple_systems.time_series.viz.forecast`

Forecast visualization: rolling predictions, confidence bands, parity plots.

<small>`pinneapple_systems/time_series/viz/forecast.py` · 209 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_rolling_forecast` | função | `(y_true, y_pred, y_std=None, sigmas=(1.0, 2.0, 3.0), title='Rolling Forecast', xlabel='Ti…)` | Plot actual vs. predicted with ±1σ/2σ/3σ confidence bands. |
| `plot_forecast_horizon` | função | `(y_context, y_true, y_pred, y_std=None, sigmas=(1.0, 2.0), title='Forecast', figsize=(12,…)` | Plot context + future ground truth + forecast with confidence bands. |
| `plot_parity` | função | `(y_true, y_pred, title='Parity Plot', figsize=(5, 5), ax=None)` | Scatter plot of actual vs. predicted with diagonal reference line. |
| `plot_residuals` | função | `(y_true, y_pred, title='Residual Analysis', figsize=(14, 4))` | Three-panel: residuals vs time, histogram, Q-Q plot. |
| `plot_backtest` | função | `(y_true, predictions, title='Backtest Comparison', figsize=(14, 5))` | Overlay multiple model predictions on ground truth. |

### Exemplos de uso reais do repositório

_Nenhum exemplo encontrado no repositório para este pacote._

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

- _Nenhuma registrada._

## 3. `pinneapple_twin3d`

### Objetivo

_Sem texto de objetivo._

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 19 | 2.820 | 19 | 67 | 74 (7 arquivos) | 4 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `demo` |  | 1 | 1 | 6 |
| `gallery` |  | 1 | 11 | 16 |
| `library` |  | 1 | 0 | 0 |
| `openfoam` |  | 1 | 6 | 6 |
| `primitives` |  | 1 | 12 | 16 |
| `scan` |  | 1 | 4 | 4 |
| `scene` |  | 1 | 5 | 9 |
| `serve` |  | 1 | 1 | 6 |
| `usd` |  | 1 | 1 | 5 |
| `wear` |  | 9 | 45 | 30 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_twin3d`

3D digital-twin visualization: build a scene in Python, open it in the bundled web viewer.

<small>`pinneapple_twin3d/__init__.py` · 26 linhas · testes: 34</small>

#### `pinneapple_twin3d.demo`

Demo scene: a 90-degree pipe bend with erosion wear growing over time, plus two sensors.

<small>`pinneapple_twin3d/demo.py` · 68 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `pipe_bend_scene` | função | `(years=(0, 1, 2, 3, 4, 5)) -> Scene` |  |

#### `pinneapple_twin3d.gallery`

Gallery: 3D twins of very different parts built from primitives, to exercise the generator beyond liners.

<small>`pinneapple_twin3d/gallery.py` · 438 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `get` | função | `(key) -> Scene` |  |
| `runner_channel` | função | `() -> Scene` | Blast-furnace cast-house runner: refractory-lined U trough from the tap hole to the skimmer. |
| `mixing_tank` | função | `() -> Scene` | Agitated tank: dished bottom, baffles, two-stage pitched-blade impeller; tracer blending over time. |
| `oil_pipeline` | função | `() -> Scene` | Buried-style crude line with two bends and a sag: wall loss from corrosion/erosion over 20 years. |
| `airliner` | função | `() -> Scene` | Narrow-body airliner: fuselage, swept wing, tail, two engines; skin temperature and wing-root load over a flight. |
| `rocket` | função | `() -> Scene` | Launch vehicle: ogive nose, tank stack, bell nozzle, four fins; aero heating and nozzle ablation during ascent. |
| `drone` | função | `() -> Scene` | Quadcopter: body, arms, motors, rotor discs, battery; motor and battery heating over a 20-minute flight. |
| `lunar_rover` | função | `() -> Scene` | Six-wheel lunar rover over one lunar day-night cycle: chassis temperature and regolith wheel wear. |
| `wind_turbine` | função | `() -> Scene` | Three-blade turbine: tapered tower, nacelle, twisted blades; leading-edge erosion and fatigue over 20 years. |
| `satellite` | função | `() -> Scene` | Small satellite in LEO: bus, two solar wings, dish; thermal cycling through sunlight and eclipse. |
| `export_all` | função | `(folder) -> str` | Export every gallery part to ``folder/<key>/`` and write ``folder/index.html`` linking them. |

#### `pinneapple_twin3d.library`

The geometry library from the twin side: ``pinneapple_twin3d.library.build("viper_rover").scene()``.

<small>`pinneapple_twin3d/library.py` · 8 linhas · **sem teste direto**</small>

#### `pinneapple_twin3d.openfoam`

OpenFOAM case -> Twin3D scene: wall patches as surfaces, per-face values over time.

<small>`pinneapple_twin3d/openfoam.py` · 215 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `read_points` | função | `(case) -> np.ndarray` |  |
| `read_faces` | função | `(case) -> List[np.ndarray]` |  |
| `read_boundary` | função | `(case) -> Dict[str, Dict[str, object]]` | ``{patch: {"type": str, "startFace": int, "nFaces": int}}`` in file order. |
| `time_directories` | função | `(case) -> List[Tuple[float, str]]` |  |
| `read_boundary_field` | função | `(path, patch, n_faces) -> Optional[np.ndarray]` | Per-face values of ``patch`` in a volScalarField/volVectorField file (vectors -> magnitude). |
| `scene_from_case` | função | `(case, *, title=None, patches=None, face_values=None, units=None, boundary_fields=(), tim…)` | Build a Twin3D scene from an OpenFOAM case. |

#### `pinneapple_twin3d.primitives`

Mesh primitives for building twin parts without a CAD file: revolve, sweep, loft, box, cylinder.

<small>`pinneapple_twin3d/primitives.py` · 183 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `revolve` | função | `(profile, n_theta=48, *, cap=False) -> Mesh` | Surface of revolution of ``(r, y)`` points around the y axis (``cap`` closes the ends with a fan). |
| `sweep` | função | `(section, path, *, closed=True, up=(0, 1, 0)) -> Mesh` | Extrude a 2D section ``(u, v)`` along a 3D path. ``u`` runs along the frame normal (closest to ``up``), ``v`` along the binormal. ``closed=False`` leaves an open profile (e.g. a U-channel). |
| `airfoil` | função | `(chord=1.0, thickness=0.12, n=24) -> np.ndarray` | Closed symmetric NACA 00xx outline, ``(2n, 2)``: x along the chord from the leading edge, y thickness. |
| `loft` | função | `(rings) -> Mesh` | Skin through rings of equal point count ``(K, 3)`` (closed rings): wings, blades, fuselage sections. |
| `wing` | função | `(span, root_chord, tip_chord, *, sweep_deg=0.0, dihedral_deg=0.0, twist_deg=0.0, thicknes…)` | Half wing / blade along +z from the root (z=0): chord along x (leading edge at -x), thickness along y. |
| `box` | função | `(size, center=(0, 0, 0)) -> Mesh` |  |
| `cylinder` | função | `(p0, p1, radius, n_theta=32, *, cap=True) -> Mesh` | Cylinder between two points (``radius`` may be a pair for a frustum). |
| `rotation` | função | `(axis, angle_deg) -> np.ndarray` |  |
| `transform` | função | `(mesh, *, R=None, t=(0, 0, 0), scale=1.0) -> Mesh` |  |
| `mirror_z` | função | `(mesh) -> Mesh` |  |
| `merge` | função | `(meshes) -> Mesh` |  |
| `mesh_quality` | função | `(mesh) -> dict` | Sanity numbers used by the tests: finite coordinates, indices in range, degenerate-triangle share, bounds. |

#### `pinneapple_twin3d.scan`

Twin geometry from scans: cleaned scan meshes and MuJoCo scenes (e.g. LiteReality output).

<small>`pinneapple_twin3d/scan.py` · 199 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ScanReport` | classe | `(vertices_in, faces_in, vertices_out, faces_out, components_dropped, holes_filled, watert…)` |  |
| `clean_scan` | função | `(path_or_mesh, *, unit='m', min_component_fraction=0.01, fill_holes=True, target_faces=No…)` | Clean a scan mesh. Returns ``(trimesh.Trimesh, ScanReport)`` in metres. |
| `add_scan` | função | `(scene, name, path_or_mesh, *, group='scan', **kw) -> ScanReport` |  |
| `scene_from_mjcf` | função | `(path, *, title=None, include_primitives=True) -> Scene` | One Twin3D part per geom of a MuJoCo scene, in world coordinates. |

#### `pinneapple_twin3d.scene`

Build and export a 3D digital-twin scene: parts, fields over time, sensors.

<small>`pinneapple_twin3d/scene.py` · 360 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Part` | classe | `(name, vertices, faces, group, color, fields, motion, visibility, physics)` |  |
| `PhysicsMaterial` | classe | `(name, static_friction, dynamic_friction, restitution, density)` | Contact material (UsdPhysics ``PhysicsMaterialAPI``). |
| `PhysicsProps` | classe | `(rigid_body, collider, kinematic, mass, density, center_of_mass, approximation, material,…)` | How a part takes part in a physics simulation (OpenUSD ``UsdPhysics``). |
| `Sensor` | classe | `(id, position, label, unit, quantity, series, envelope)` |  |
| `Scene` | classe | `(title, *, length_unit='m', times=None, time_unit='s', source='')` | A digital-twin scene. Add parts, fields and sensors, then ``export`` to a folder. |

#### `pinneapple_twin3d.serve`

Serve an exported scene folder over HTTP (the viewer needs HTTP, not file://, to fetch its data).

<small>`pinneapple_twin3d/serve.py` · 26 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `serve` | função | `(folder, port=8765, *, host='127.0.0.1', block=True)` | Serve ``folder`` at ``http://host:port/``. With ``block=False`` returns the running server. |

#### `pinneapple_twin3d.usd`

Export a Twin3D scene as OpenUSD (``.usda`` text), for Omniverse / Isaac Sim / usdview.

<small>`pinneapple_twin3d/usd.py` · 209 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `export_usd` | função | `(scene, path) -> str` |  |

#### `pinneapple_twin3d.wear`

Generator of 3D wear (refractory / liner / pipe) digital twins from a spec.

<small>`pinneapple_twin3d/wear/__init__.py` · 21 linhas · testes: 30</small>

#### `pinneapple_twin3d.wear.build`

WearDataset -> Scene (3D viewer) + hub contract + forecast/optimizer report.

<small>`pinneapple_twin3d/wear/build.py` · 95 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `remaining_life_fields` | função | `(ds, cap, **kw) -> dict` | Per-cell remaining life at every reading, capped at ``cap`` (no trend -> cap; reading 0 -> cap). |
| `scene_from_wear` | função | `(ds, *, life_cap=None, **fit_kw) -> Scene` |  |
| `report` | função | `(ds, cfg=None, **fit_kw) -> dict` | Observed / predicted / recommended, kept apart on purpose. |
| `export_wear_twin` | função | `(ds, folder, *, cfg=None, usd=False, **fit_kw) -> str` | Write the 3D viewer scene, ``wear_dataset.json`` (hub contract) and ``report.json`` to ``folder``. |

#### `pinneapple_twin3d.wear.calibrate`

Calibrate the optimizer's assumptions from the plant's own records.

<small>`pinneapple_twin3d/wear/calibrate.py` · 138 linhas · testes: 30</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `StopRecord` | classe | `(zones, duration_h, cost, x)` |  |
| `ParamReport` | classe | `(name, value, source, n, se, note)` |  |
| `Calibration` | classe | `(cfg, report)` |  |
| `estimate_recovery` | função | `(ds, repair_drop=0.08, min_cells=20)` | Median fraction of the usable thickness restored per repaired cell, and the number of cells used. |
| `calibrate` | função | `(stops=(), ds=None, base=None) -> Calibration` |  |

#### `pinneapple_twin3d.wear.forecast`

Remaining-life forecast per cell: the model of the reference twins, plus confidence.

<small>`pinneapple_twin3d/wear/forecast.py` · 185 linhas · testes: 30</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CellFit` | classe | `(remaining, remaining_lo, remaining_hi, slope, r2, n, confidence)` |  |
| `ZoneForecast` | classe | `(zone, worst_fraction, n_over, min_remaining, rate, usable, critical_cell, confidence, r2…)` |  |
| `t90` | função | `(df)` | Student-t 0.95 quantile (90 % two-sided interval) for ``df`` degrees of freedom (array ok). |
| `confidence_index` | função | `(n, r2)` |  |
| `fit_zone` | função | `(ds, zone, s, *, win_x=DEFAULT_WIN_X, min_pts=8, repair_drop=0.08) -> CellFit` |  |
| `forecast` | função | `(ds, s=None, **kw) -> Dict[str, ZoneForecast]` |  |
| `forecast_history` | função | `(ds, **kw) -> List[dict]` | Minimum predicted remaining life recomputed at every reading (how the forecast evolved). |
| `backtest` | função | `(ds, **kw) -> dict` | Compare past predictions of 'x when the first cell hits the limit' with what happened. |
| `status` | função | `(fraction) -> str` |  |

#### `pinneapple_twin3d.wear.geometry`

Cell meshes for surfaces of revolution.

<small>`pinneapple_twin3d/wear/geometry.py` · 62 linhas · testes: 30</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `sample_profile` | função | `(profile, n_rows) -> np.ndarray` | (n_rows + 1, 2) points (r, y) equally spaced in arc length along the polyline. |
| `cell_mesh` | função | `(zone) -> Tuple[np.ndarray, np.ndarray]` | Vertices (N*M*4, 3) and faces (N*M*2, 3) of the zone, one quad per cell. |
| `cell_to_vertex` | função | `(values) -> np.ndarray` | Per-cell values (..., N, M) -> per-vertex (..., N*M*4) following ``cell_mesh``'s ordering. |
| `cell_centers` | função | `(zone) -> np.ndarray` | (N, M, 3) world position of each cell centre (used to place sensors / hotspots). |

#### `pinneapple_twin3d.wear.model`

Data model of a wear twin: vessel spec (geometry + zones) and the wear dataset.

<small>`pinneapple_twin3d/wear/model.py` · 165 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ZoneSpec` | classe | `(name, label, profile, n_rows, n_sectors, e0, emin, center, theta0, group)` |  |
| `VesselSpec` | classe | `(key, title, zones, x_name, wear_unit, notes)` |  |
| `WearDataset` | classe | `(spec, x, wear, origin, labels, note)` | Wear depth [mm] per zone: ``wear[zone]`` has shape (T, rows, sectors); NaN = no reading. |

#### `pinneapple_twin3d.wear.optimize`

Maintenance (gunning / repair) scheduling over the forecast. Decision support only.

<small>`pinneapple_twin3d/wear/optimize.py` · 166 linhas · testes: 30</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OptimizerConfig` | classe | `(horizon, heats_per_day, recovery_frac, setup_hours, hours_per_zone, cost_per_hour, cost_…)` |  |
| `ZoneState` | classe | `(zone, deadline, life_full, ext)` |  |
| `Policy` | classe | `(id, name, schedule, stops, zones_served, downtime_hours, breach, cost, window)` |  |
| `OptimizationResult` | classe | `(cfg, zones, policies, recommended, baseline, notes)` |  |
| `prepare` | função | `(zones, cfg) -> List[ZoneState]` |  |
| `simulate` | função | `(zs, stops, cfg) -> dict` |  |
| `fixed_policy` | função | `(zs, cfg) -> List[dict]` |  |
| `predictive_policy` | função | `(zs, cfg, window) -> List[dict]` |  |
| `optimize` | função | `(forecasts, cfg=None) -> OptimizationResult` |  |
| `explain` | função | `(res) -> List[dict]` |  |

#### `pinneapple_twin3d.wear.presets`

The four reference use cases as specs: steel ladle, pig-iron ladle, BOF converter, RH degasser.

<small>`pinneapple_twin3d/wear/presets.py` · 176 linhas · testes: 30</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `get` | função | `(key)` |  |
| `fill_level` | função | `(rin, y0, y1, rho, mass, n=300) -> float` | Bath level for ``mass`` [t] in a vessel of inner radius ``rin(y)`` (frustum integration). |
| `steel_ladle` | função | `()` |  |
| `pig_iron_ladle` | função | `()` |  |
| `bof_converter` | função | `()` |  |
| `rh_degasser` | função | `()` |  |
| `smelting_reduction_reactor` | função | `()` | Generic smelting-reduction shaft reactor: shaft, tuyere zone, crucible, hearth. |

#### `pinneapple_twin3d.wear.synthetic`

Synthetic wear campaigns: explicit hot spots + growth law + repairs. Always labelled SYNTHETIC.

<small>`pinneapple_twin3d/wear/synthetic.py` · 80 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HotSpot` | classe | `(zone, s, theta_deg, ds, dtheta_deg, amp)` |  |
| `base_pattern` | função | `(spec, hotspots, base) -> Dict[str, np.ndarray]` |  |
| `synthetic_campaign` | função | `(spec, hotspots, *, n_readings=41, x_max=120.0, growth_power=1.0, base=0.2, noise=0.01, s…)` | ``noise`` is the sigma of additive measurement noise as a fraction of the usable thickness. Wear grows as ``(x/x_max)**growth_power`` towards ``pattern * usable``; at each ``repairs`` x the wear of ``repair_zones`` drops by ``repair_gain`` of its value (gunning), then resumes. |

### Exemplos de uso reais do repositório

_Nenhum exemplo encontrado no repositório para este pacote._

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 15 de 19.
- Funções de teste em arquivos que importam algum módulo do pacote: 74.

### Limitações conhecidas

- _Nenhuma registrada._
