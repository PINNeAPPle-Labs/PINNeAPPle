# Volume 10 — Glossário e apêndice numérico

> Parte do [Mapa completo do PINNeAPPle](../mapa_completo.md). Gerado por `docs/pt/_mapa/gerar.py` a partir do inventário do commit `e53cdcfb04`.

Glossário em português e os números do inventário.

## 1. Glossário

| termo | significado |
|---|---|
| **Adjunto** | Método que calcula o gradiente de um objetivo em relação a muitos parâmetros com custo de uma simulação extra. |
| **Assimilação de dados** | Combinar um modelo com medições ao longo do tempo (por exemplo, filtro de Kalman, EnKF). |
| **Autolab** | Laboratório autônomo: planeja e executa missões com ferramentas controladas, compara surrogates e escolhe a próxima simulação. |
| **Co-simulação** | Acoplar simuladores diferentes que trocam dados a cada passo de tempo (por exemplo, via FMI/FMU). |
| **Condensação (Schur)** | Eliminar os graus de liberdade internos de um bloco, ficando só com a interface; reaproveitado entre blocos idênticos. |
| **Condição de contorno (BC)** | Valor ou fluxo imposto na fronteira do domínio. Dirichlet fixa o valor; Neumann fixa a derivada normal; Robin combina os dois. |
| **Condição inicial (IC)** | Estado do sistema no instante inicial de um problema transiente. |
| **Craig-Bampton** | Redução de modelos para dinâmica: modos de interface fixa mais modos de restrição. |
| **Credibilidade** | Nível atribuído por pp.validate a cada fator verificado, com decisão final pass / warn / fail. |
| **DeepONet** | Operador neural com uma rede 'branch' (entrada funcional) e uma 'trunk' (coordenadas). |
| **EKI** | Ensemble Kalman Inversion: método de inversão sem gradiente baseado em ensembles. |
| **Extra (pyproject)** | Grupo opcional de dependências instalado com pip install "pinneapple[extra]". |
| **FDM / FEM / FVM** | Diferenças finitas, elementos finitos e volumes finitos: famílias clássicas de discretização de equações diferenciais. |
| **Flow** | Grafo de nós tipados (JSON pinneapple-flow/1) que encadeia chamadas da biblioteca; executado por pinneapple_flow. |
| **FMI / FMU** | Functional Mock-up Interface: padrão para trocar modelos de simulação empacotados (Functional Mock-up Units). |
| **FNO** | Fourier Neural Operator: operador neural que mistura informação no espaço de Fourier. |
| **Frente de Pareto** | Conjunto de soluções em que nenhum objetivo melhora sem piorar outro. |
| **GNN / MeshGraphNet** | Rede de grafos que opera sobre os nós e arestas de uma malha. |
| **Guardrail físico** | Verificação que bloqueia propostas (por exemplo, de um LLM) que violam a física ou o escopo. |
| **Gêmeo digital (digital twin)** | Modelo de um ativo real que é atualizado por medições e usado para monitorar, prever ou decidir. |
| **Hard BC** | Condição de contorno imposta pela própria forma da rede (por construção), e não por um termo de perda. |
| **Hub / zoo** | Repositório de modelos treinados e verificados, datasets e biblioteca pesquisável de ferramentas e exemplos. |
| **Lab / selo** | pinneapple_lab registra experimentos com manifesto, métricas, verificações e uma nota (A-F); o selo resume essa avaliação. |
| **LBM** | Lattice Boltzmann Method: método de fluidos baseado em distribuições de partículas numa rede regular. |
| **MCP** | Model Context Protocol: protocolo para expor ferramentas a agentes de IA; pinneapple_mcp é o servidor do PINNeAPPle. |
| **NSGA-II** | Algoritmo genético multiobjetivo que encontra uma frente de Pareto. |
| **OOD (fora do domínio)** | Entrada fora da região onde o modelo foi treinado ou validado; os resultados ali não são confiáveis. |
| **OpenUSD / Omniverse** | Formato de cena 3-D da Pixar e plataforma da NVIDIA que o usa; o PINNeAPPle exporta campos e decisões de confiança para ele. |
| **Operador neural** | Rede que aprende um mapa entre funções (por exemplo, condição de contorno → campo solução). Exemplos: FNO, DeepONet, AFNO. |
| **Otimização bayesiana** | Otimização com um modelo probabilístico (em geral processo gaussiano) que escolhe o próximo ponto a avaliar. |
| **PhysicalProblem** | Classe única do PINNeAPPle que descreve um problema físico para todas as partes da biblioteca. |
| **PhysicsNeMo** | Framework da NVIDIA para IA física; o PINNeAPPle confere surrogates feitos com ele. |
| **PINN** | Physics-Informed Neural Network: rede neural treinada para satisfazer a equação diferencial (resíduo na perda), além de dados e condições de contorno. |
| **Ponto de colocação (collocation point)** | Ponto do domínio onde o resíduo da equação é avaliado durante o treino de uma PINN. |
| **Predição conformal** | Método que produz intervalos com cobertura garantida a partir de um conjunto de calibração. |
| **Preset** | Problema pronto, com nome, equações, domínio e parâmetros padrão; pode ser carregado e resolvido diretamente. |
| **Problema inverso** | Estimar parâmetros ou entradas desconhecidas a partir de observações da saída. |
| **ProblemSpec** | Definição executável de uma EDP (coordenadas, campos, termo da EDP, contornos, domínio) consumida pelo compilador de PINN. |
| **Referência independente** | Valor de comparação que não vem do próprio código testado: forma fechada, solução manufaturada ou valor publicado/tabelado. |
| **Resíduo** | Quanto a solução deixa de satisfazer a equação num ponto. Resíduo zero = equação satisfeita. |
| **ROM** | Reduced-Order Model: modelo de ordem reduzida (por exemplo, POD) que aproxima o modelo completo com poucos graus de liberdade. |
| **SDF / CSG** | Signed Distance Function (distância com sinal até a fronteira) e Constructive Solid Geometry (união, interseção e diferença de sólidos). |
| **Shim de compatibilidade** | Pacote que só reexporta outro, mantido para que imports antigos funcionem. |
| **Solution** | Objeto devolvido por pp.solve: valores, pontos, método e metadados (info). |
| **Solução manufaturada** | Solução escolhida de antemão, inserida na equação para gerar o termo fonte; serve para verificar a ordem de convergência de um solver. |
| **SPH** | Smoothed Particle Hydrodynamics: método sem malha baseado em partículas. |
| **Surrogate (modelo substituto)** | Modelo rápido, em geral aprendido, que aproxima um simulador caro. |
| **Teste de Diebold-Mariano (DM)** | Teste estatístico que compara a precisão de duas previsões. |
| **Trust card** | Cartão de confiança: conjunto de verificações físicas de um campo (resíduo, conservação, contornos, referência, domínio) com decisão APPROVED / REVIEW / REJECT. |
| **UQ** | Uncertainty Quantification: quantificação de incerteza (ensembles, conformal, propagação de tolerâncias, Monte Carlo). |
| **validated / tested / untested** | Status do catálogo de métodos: validated = algum teste compara contra referência independente; tested = há testes, mas nenhum contra referência; untested = sem testes. |
| **Walk-forward purgado** | Protocolo de validação de séries temporais que treina no passado, testa no futuro e remove amostras vizinhas para evitar vazamento. |
| **X-TFC / TFC** | Theory of Functional Connections: expressões que satisfazem as restrições exatamente; X-TFC combina isso com redes de camada única (extreme learning machine). |

## 2. Apêndice: números do inventário

| item | valor |
|---|---:|
| pacotes `pinneapple*` | 41 |
| módulos Python nesses pacotes | 2.142 |
| linhas de código nesses módulos | 494.775 |
| classes públicas | 2.757 |
| funções públicas de nível de módulo | 6.130 |
| métodos públicos de classes | 6.220 |
| módulos importados por pelo menos um arquivo de teste | 1.073 |
| arquivos de teste (`tests/**/test*.py`) | 513 |
| funções de teste (`test*`, inclusive em classes `Test*`) | 4.611 |
| nomes expostos em `pp.*` | 297 |
| extras do `pyproject.toml` | 54 |
| apps em `apps/` | 64 |
| entradas em `examples/` | 88 |
| entradas em `scripts/` | 24 |
| arquivos Markdown em `docs/` | 190 |
| workflows de CI | 12 |
| métodos no catálogo | 201 |

Status dos métodos do catálogo (regra: validated = a test exercising the item compares against an independent reference (closed form, manufactured solution, published/tabulated value); arquivo gerado em 2026-10-17):

| status | métodos |
|---|---:|
| tested | 47 |
| untested | 42 |
| validated | 112 |

### 2.1 Por pacote

| pacote | módulos | linhas | classes | funções | testes* | módulos sem teste |
|---|---:|---:|---:|---:|---:|---:|
| `pinneapple` | 3 | 866 | 0 | 2 | 431 | 2 |
| `pinneapple_adaptation` | 13 | 2.946 | 10 | 9 | 154 | 2 |
| `pinneapple_analysis` | 159 | 42.004 | 249 | 450 | 939 | 56 |
| `pinneapple_app` | 18 | 3.672 | 29 | 53 | 48 | 11 |
| `pinneapple_arena` | 54 | 21.853 | 69 | 243 | 99 | 39 |
| `pinneapple_autolab` | 16 | 3.596 | 38 | 39 | 37 | 10 |
| `pinneapple_blender` | 5 | 616 | 1 | 7 | 15 | 2 |
| `pinneapple_catalog` | 3 | 1.024 | 2 | 8 | 16 | 1 |
| `pinneapple_check` | 23 | 4.547 | 22 | 71 | 144 | 11 |
| `pinneapple_condense` | 18 | 6.006 | 29 | 115 | 27 | 3 |
| `pinneapple_core` | 34 | 8.747 | 71 | 92 | 233 | 15 |
| `pinneapple_data` | 115 | 29.019 | 129 | 256 | 507 | 70 |
| `pinneapple_decision` | 12 | 2.499 | 37 | 20 | 44 | 9 |
| `pinneapple_design` | 196 | 48.153 | 269 | 740 | 744 | 86 |
| `pinneapple_flow` | 89 | 18.086 | 23 | 381 | 263 | 58 |
| `pinneapple_hub` | 31 | 7.260 | 22 | 170 | 183 | 8 |
| `pinneapple_lab` | 42 | 11.775 | 45 | 119 | 96 | 27 |
| `pinneapple_llm` | 49 | 10.750 | 66 | 99 | 242 | 28 |
| `pinneapple_mcp` | 8 | 1.541 | 2 | 31 | 30 | 3 |
| `pinneapple_models` | 2 | 74 | 0 | 1 | 1 | 1 |
| `pinneapple_neural` | 198 | 36.217 | 334 | 150 | 438 | 152 |
| `pinneapple_orchestration` | 16 | 591 | 1 | 11 | 6 | 14 |
| `pinneapple_pdb` | 7 | 1.929 | 9 | 14 | 100 | 5 |
| `pinneapple_perception` | 7 | 1.037 | 6 | 19 | 29 | 2 |
| `pinneapple_physics` | 166 | 43.959 | 169 | 763 | 1.075 | 52 |
| `pinneapple_presets` | 21 | 5.247 | 22 | 31 | 50 | 16 |
| `pinneapple_problemdesign` | 28 | 5.642 | 40 | 21 | 123 | 19 |
| `pinneapple_quantum` | 16 | 2.636 | 17 | 24 | 0 | 16 |
| `pinneapple_query` | 3 | 252 | 7 | 2 | 7 | 2 |
| `pinneapple_reason` | 18 | 3.757 | 17 | 60 | 17 | 8 |
| `pinneapple_registry` | 10 | 1.699 | 10 | 3 | 92 | 3 |
| `pinneapple_science` | 16 | 4.253 | 33 | 49 | 19 | 10 |
| `pinneapple_security` | 20 | 3.398 | 33 | 93 | 105 | 0 |
| `pinneapple_simulation` | 298 | 66.085 | 364 | 931 | 1.119 | 113 |
| `pinneapple_solvers` | 2 | 71 | 0 | 0 | 1 | 1 |
| `pinneapple_systems` | 182 | 37.262 | 301 | 515 | 579 | 78 |
| `pinneapple_tools` | 177 | 34.350 | 155 | 384 | 465 | 107 |
| `pinneapple_train` | 4 | 109 | 0 | 0 | 1 | 1 |
| `pinneapple_twin3d` | 19 | 2.820 | 19 | 67 | 74 | 4 |
| `pinneapple_veriphysics` | 13 | 3.411 | 17 | 38 | 98 | 1 |
| `pinneapple_worldmodel` | 31 | 15.016 | 90 | 49 | 94 | 23 |

\* funções de teste em arquivos que importam algum módulo do pacote (um teste pode contar em mais de um pacote).

### 2.2 Extras do `pyproject.toml`

Projeto `pinneapple` versão `0.7.0`, Python `>=3.10`. Dependências obrigatórias: `numpy>=1.24`, `scipy>=1.10`, `pandas>=2.0`, `pyyaml>=6.0`, `tqdm>=4.66`, `typing-extensions>=4.8`, `pydantic>=2.0`, `matplotlib>=3.8`, `torch>=2.2`, `sympy>=1.12`, `xarray`, `zarr`, `tomli>=2.0; python_version < '3.11'`.

| extra | pacotes |
|---|---|
| `cfd` | manifold3d>=2.4, huggingface_hub>=0.20, cadquery>=2.4, gmsh>=4.11, trimesh>=4.0, pyvista>=0.43, plotly>=5.18, meshio>=5.3, h5py>=3.8 |
| `enterprise` | fastapi>=0.111, uvicorn[standard]>=0.29, cryptography>=41, argon2-cffi>=23.1, meshio>=5.3, h5py>=3.8, huggingface_hub>=0.24, pyarrow>=14, safetensors>=0.4, mcp>=1.10 |
| `mcp` | mcp>=1.10 |
| `security` | cryptography>=41, argon2-cffi>=23.1 |
| `check` | meshio>=5.3, h5py>=3.8, cryptography>=41 |
| `sports` | scikit-learn>=1.3 |
| `em` | miepython>=3.0 |
| `veriphysics` | reportlab>=4.0 |
| `viz3d` | rerun-sdk>=0.20, plotly>=5.18, trame>=3.0, trame-vtk>=2.5, trame-vuetify>=2.4 |
| `twin` | basyx-python-sdk>=1.0, asyncua>=1.1 |
| `geometry` | manifold3d>=2.4, huggingface_hub>=0.20 |
| `fitmesh` | cadquery>=2.4, gmsh>=4.11, trimesh>=4.0 |
| `newton` | newton>=1.6 |
| `sky` | sgp4>=2.22, pymsis>=0.10, skyfield-data>=6.0, skyfield>=1.48 |
| `astroml` | astroML>=1.0 |
| `fmi` | fmpy>=0.3.20, pythonfmu>=0.6 |
| `tracking` | mlflow>=2.9, wandb>=0.16 |
| `geom` | trimesh>=4.0.0, meshio>=5.3.0, gmsh>=4.11 |
| `geom_simplify` | fast-simplification>=0.1.0, open3d>=0.18.0 |
| `cad` | cadquery>=2.4.0 |
| `solvers` | numba>=0.59 |
| `process` | CoolProp>=6.6 |
| `astro` | astropy>=6.0 |
| `jax_cfd` | jax-cfd>=0.2.0 |
| `fenics` | fenics-dolfinx>=0.7.0, mpi4py>=3.1, petsc4py>=3.20 |
| `mujoco` | mujoco>=3.0 |
| `mujoco_mjx` | mujoco>=3.0, jax[cuda12]>=0.4.25 |
| `genesis` | genesis-world>=0.2.0 |
| `noether` | emmiai-noether>=2026.4.0 |
| `pinn` | sympy>=1.12 |
| `uq` | scipy>=1.10 |
| `serve` | fastapi>=0.111, uvicorn[standard]>=0.29 |
| `export` | onnx>=1.16, onnxruntime>=1.18 |
| `hub` | huggingface_hub>=0.24, pyarrow>=14, safetensors>=0.4, onnx>=1.16, onnxruntime>=1.18 |
| `flow` | sqlalchemy>=2.0, sympy>=1.12, fastapi>=0.111, uvicorn[standard]>=0.29, httpx>=0.27, pillow>=10 |
| `llm` | anthropic>=0.34, openai>=1.40 |
| `orchestration` | prefect>=2.19 |
| `tuning` | optuna>=3.6 |
| `workbench` | fastapi>=0.110, uvicorn>=0.29, requests>=2.31 |
| `finetune` | peft>=0.12, datasets>=2.20, transformers>=4.44 |
| `vlm_curation` | transformers>=4.45, accelerate>=0.30 |
| `digital_twin` | paho-mqtt>=2.0, kafka-python>=2.0, asyncua>=1.1, pymodbus>=3.6 |
| `timeseries` | statsmodels>=0.14, pmdarima>=2.0, scikit-learn>=1.4, PyWavelets>=1.6 |
| `synth` | sympy>=1.12, trimesh>=4.0.0, meshio>=5.3.0 |
| `earthdata` | requests>=2.31, earthaccess>=0.10.0 |
| `research` | requests>=2.31, pdfminer.six>=20231228 |
| `retrieval` | sentence-transformers>=3.0 |
| `all` | cryptography>=41, trimesh>=4.0.0, meshio>=5.3.0, gmsh>=4.11, fast-simplification>=0.1.0, cadquery>=2.4.0, numba>=0.59, sympy>=1.12, fastapi>=0.111, uvicorn[standard]>=0.29, onnx>=1.16, onnxruntime>=1.18, paho-mqtt>=2.0, kafka-python>=2.0, sentence-transformers>=3.0, statsmodels>=0.14, requests>=2.31, mujoco>=3.0, fmpy>=0.3.20, genesis-world>=0.2.0, emmiai-noether>=2026.4.0, huggingface_hub>=0.24, anthropic>=0.34, openai>=1.40, transformers>=4.45, accelerate>=0.30 |
| `viz` | matplotlib>=3.8, scipy>=1.10 |
| `viz_interactive` | plotly>=5.18, kaleido>=0.2 |
| `pyvista` | pyvista>=0.43, matplotlib>=3.8, imageio>=2.34 |
| `cost` | psutil>=5.9 |
| `surrogate` | scikit-learn>=1.4, pyarrow>=14, h5py>=3.8, onnx>=1.15, onnxruntime>=1.17 |
| `dev` | ruff>=0.5, pytest>=8.0, pytest-cov>=4.1, mypy>=1.8, build>=1.2, twine>=5.0, ipykernel>=6.0, httpx>=0.27, jupyter>=1.0 |

### 2.3 Catálogo completo de métodos

| id | método | status | testes contra referência | outros testes | código |
|---|---|---|---:|---:|---|
| E1 | Laplace | validated | 7 | 28 |  |
| E2 | Poisson | validated | 15 | 24 |  |
| E3 | Helmholtz | validated | 2 | 1 |  |
| E4 | Heat (steady anisotropic / transient) | validated | 4 | 13 |  |
| E5 | Wave | validated | 1 | 0 |  |
| E6 | Advection-diffusion / convection | validated | 1 | 0 |  |
| E7 | Burgers (viscous / inviscid) | validated | 10 | 50 |  |
| E8 | Reaction-diffusion 1D/2D (Gray-Scott) | validated | 2 | 1 |  |
| E9 | Fisher-KPP | validated | 1 | 3 |  |
| E10 | Incompressible Navier-Stokes 2D/3D | tested | 0 | 13 |  |
| E11 | Navier-Stokes in a rotating frame | tested | 0 | 2 |  |
| E12 | Stokes | validated | 1 | 1 |  |
| E13 | Brinkman | validated | 1 | 0 |  |
| E14 | Darcy | tested | 0 | 12 |  |
| E15 | Buckley-Leverett two-phase | validated | 1 | 0 |  |
| E16 | Compressible Euler 1D/2D/axisymmetric/rotating 3D | validated | 3 | 1 |  |
| E17 | Shallow water 1D/2D | validated | 1 | 6 |  |
| E18 | Stommel gyre | validated | 2 | 0 |  |
| E19 | Plane-stress / plane-strain linear elasticity | validated | 2 | 5 |  |
| E20 | Axisymmetric elasticity (+ torsion) / Lame | validated | 1 | 4 |  |
| E21 | Neo-Hookean hyperelasticity | validated | 1 | 0 |  |
| E22 | Euler-Bernoulli beam | validated | 1 | 11 | pinneapple_systems/process_components/beam_statics.py |
| E23 | von Karman beam / plate | validated | 2 | 6 | pinneapple_systems/process_components/beam_nonlinear_fem.py |
| E24 | Thermoelasticity | validated | 1 | 0 |  |
| E25 | Phase-field fracture | untested | 0 | 0 |  |
| E26 | Biot poroelasticity | validated | 1 | 0 |  |
| E27 | Maxwell TE | validated | 1 | 0 |  |
| E28 | Bekker-Wong terramechanics | validated | 1 | 2 |  |
| E29 | Compressor meanline 1D | validated | 2 | 3 |  |
| E30 | Phonon Boltzmann transport 1D (gray) | validated | 3 | 1 |  |
| E31 | Black-Scholes | validated | 1 | 2 |  |
| E32 | Heston | validated | 2 | 1 |  |
| E33 | Opinion dynamics (continuum Hegselmann-Krause) | validated | 1 | 0 |  |
| E34 | Kepler two-body orbit | validated | 1 | 6 |  |
| E35 | Circular restricted three-body problem (planar, synodic) | validated | 3 | 3 |  |
| E36 | J2 perturbation | validated | 1 | 1 |  |
| E37 | Clohessy-Wiltshire relative motion | validated | 1 | 1 |  |
| E38 | Euler rigid-body attitude | validated | 1 | 1 |  |
| E39 | Lane-Emden polytrope | validated | 5 | 19 |  |
| E40 | Shakura-Sunyaev accretion disk | validated | 6 | 4 |  |
| E41 | Schwarzschild weak-field light bending | validated | 2 | 3 |  |
| E42 | SIR epidemic | validated | 1 | 0 |  |
| E43 | Pharmacokinetics two-compartment | validated | 1 | 0 |  |
| E44 | Reaction network / Arrhenius / mass action | tested | 0 | 10 | pinneapple_systems/process_components/reaction_kinetics.py |
| E45 | Hagen-Poiseuille | tested | 0 | 2 | PINNeAPPle-CFD/pinneapple_cfd/etapas/e02_modelo_matematico/correlacoes.py |
| E46 | Blasius friction factor | validated | 2 | 10 | PINNeAPPle-CFD/.../correlacoes.py |
| E47 | Colebrook-White | tested | 0 | 8 | PINNeAPPle-CFD/.../correlacoes.py, pinneapple_systems/process_components/pipe_network_1d.py |
| E48 | Darcy-Weisbach pressure drop | untested | 0 | 0 | PINNeAPPle-CFD/.../correlacoes.py |
| E49 | Herschel-Bulkley + Metzner-Reed (non-Newtonian pipe flow) | tested | 0 | 4 | pinneapple_systems/process_components/non_newtonian_pipe_flow.py |
| E50 | DNV-RP-O501 erosion (straight pipe and bend) | untested | 0 | 0 | PINNeAPPle-CFD/pinneapple_cfd/etapas/e02_modelo_matematico/erosao.py |
| E51 | Lame thick-walled cylinder | tested | 0 | 3 | pinneapple_systems/process_components/pipe_stress_mechanics.py |
| E52 | von Mises equivalent stress | tested | 0 | 11 | pinneapple_systems/process_components/pipe_stress_mechanics.py, pinneapple_systems/process_components/beam_statics.py |
| E53 | Euler buckling + constrained rod buckling | validated | 1 | 6 | pinneapple_systems/process_components/pipe_stress_mechanics.py |
| E54 | Beam-column moment amplification factor | tested | 0 | 3 | pinneapple_systems/process_components/pipe_stress_mechanics.py |
| E55 | S-N curve + Goodman + Miner | validated | 1 | 14 | pinneapple_systems/process_components/fatigue_analysis.py, pinneapple_physics/closed_form/cantilever_fatigue.py |
| E56 | Torsional stick-slip with Stribeck friction | tested | 0 | 6 | pinneapple_systems/process_components/torsional_stickslip.py |
| E57 | Heat exchanger (effectiveness-NTU) | tested | 0 | 8 | pinneapple_systems/process_components/heat_exchanger.py |
| E58 | Fin array conduction | tested | 0 | 4 | pinneapple_physics/closed_form/fin_array_conduction.py |
| E59 | Control valve (Cv, compressible / incompressible) | tested | 0 | 1 | pinneapple_systems/process_components/control_valve.py |
| E60 | Polytropic compression path | tested | 0 | 1 | pinneapple_systems/process_components/polytropic_path.py |
| E61 | Real gas equation of state (CoolProp) | tested | 0 | 2 | pinneapple_systems/process_components/real_gas_eos.py |
| E62 | Compressor similarity map | validated | 2 | 0 | pinneapple_systems/process_components/similarity_map.py |
| E63 | Helmholtz resonator | tested | 0 | 4 | pinneapple_physics/closed_form/helmholtz_resonator.py |
| E64 | Membrane diffusion | tested | 0 | 5 | pinneapple_physics/closed_form/membrane_diffusion.py |
| E65 | Cantilever fatigue | validated | 1 | 4 | pinneapple_physics/closed_form/cantilever_fatigue.py |
| E66 | Electrostatics (Gauss's law with permittivity), differential and integral form | validated | 1 | 0 |  |
| E67 | Magnetostatics in the vector potential A_z | validated | 1 | 0 |  |
| E68 | Magneto-quasistatic eddy currents (time-harmonic A_z) | validated | 1 | 0 |  |
| E69 | Maxwell TMz time domain, differential and integral (Ampere-Maxwell circulation, Faraday) form | validated | 1 | 0 |  |
| E70 | Maxwell's curl equations in 3-D with optional Gauss constraints | validated | 1 | 0 |  |
| E71 | Frequency-domain TMz wave equation with complex permittivity | validated | 1 | 0 |  |
| P9 | PINNeAPPle-CFD v1: pipes + particle erosion (CFD-04 pilot) | untested | 0 | 0 | PINNeAPPle-CFD/pinneapple_cfd/ |
| P1.1 |  | validated | 9 | 43 |  |
| P1.2 |  | tested | 0 | 4 |  |
| P1.3 |  | untested | 0 | 0 |  |
| P1.4 |  | untested | 0 | 0 |  |
| P1.5 |  | validated | 6 | 15 |  |
| P1.6 |  | untested | 0 | 0 |  |
| P1.7 |  | tested | 0 | 7 |  |
| P1.8 |  | untested | 0 | 0 |  |
| P1.9 |  | validated | 4 | 15 |  |
| P2.1 |  | untested | 0 | 0 |  |
| P2.2 |  | tested | 0 | 5 |  |
| P2.3 |  | tested | 0 | 1 |  |
| P2.4 |  | untested | 0 | 0 |  |
| P2.5 |  | untested | 0 | 0 |  |
| P2.6 |  | validated | 1 | 0 |  |
| P2.7 |  | untested | 0 | 0 |  |
| P2.8 |  | tested | 1 | 0 |  |
| P2.9 |  | tested | 0 | 1 |  |
| P3.1 |  | untested | 0 | 0 |  |
| P3.2 |  | untested | 0 | 0 |  |
| P3.3 |  | validated | 1 | 3 |  |
| P3.4 |  | untested | 0 | 0 |  |
| P3.5 |  | untested | 0 | 0 |  |
| P3.6 |  | untested | 0 | 0 |  |
| P3.7 |  | untested | 0 | 0 |  |
| P4.1 |  | untested | 0 | 0 |  |
| P4.2 |  | validated | 1 | 2 |  |
| P4.3 |  | validated | 1 | 0 |  |
| P4.4 |  | tested | 0 | 1 |  |
| P5.1 |  | validated | 2 | 6 |  |
| P5.2 |  | untested | 0 | 0 |  |
| P5.3 |  | untested | 0 | 0 |  |
| P5.4 |  | validated | 1 | 0 |  |
| P5.5 |  | untested | 0 | 0 |  |
| P5.6 |  | validated | 1 | 2 |  |
| P5.7 |  | validated | 1 | 5 |  |
| P5.8 |  | validated | 1 | 0 |  |
| P6.1 |  | validated | 1 | 2 |  |
| P6.2 |  | untested | 0 | 0 |  |
| P6.3 |  | untested | 0 | 0 |  |
| P6.4 |  | validated | 1 | 0 |  |
| P6.5 |  | untested | 0 | 0 |  |
| P6.6 |  | tested | 2 | 1 |  |
| P6.7 |  | validated | 1 | 0 |  |
| P6.8 |  | validated | 1 | 0 |  |
| P6.9 |  | untested | 0 | 0 |  |
| P7.1 |  | validated | 3 | 2 |  |
| P7.2 |  | validated | 1 | 2 |  |
| P7.3 |  | validated | 2 | 8 |  |
| P7.4 |  | validated | 1 | 1 |  |
| P7.5 |  | validated | 1 | 1 |  |
| P7.6 |  | validated | 1 | 2 |  |
| P7.7 |  | validated | 1 | 3 |  |
| P7.8 |  | untested | 0 | 0 |  |
| P7.9 |  | validated | 1 | 1 |  |
| P8.1 |  | validated | 1 | 2 |  |
| P1.10 |  | validated | 2 | 1 |  |
| P1.11 |  | tested | 0 | 1 |  |
| P1.12 |  | untested | 0 | 0 |  |
| P1.13 |  | untested | 0 | 0 |  |
| P2.10 |  | untested | 0 | 0 |  |
| P2.11 |  | untested | 0 | 0 |  |
| P2.12 |  | untested | 0 | 0 |  |
| P2.13 |  | untested | 0 | 0 |  |
| P7.10 |  | validated | 1 | 1 |  |
| S1 | Finite differences 1D/2D/3D | validated | 2 | 11 | pinneapple_simulation/numerical_solvers/fdm.py, pinneapple_simulation/numerical_solvers/fdm3d.py |
| S2 | Finite elements (linear) + nonlinear beam FEM | validated | 6 | 15 | pinneapple_simulation/numerical_solvers/fem.py, pinneapple_simulation/numerical_solvers/nonlinear_beam_fem.py |
| S3 | Finite volumes | validated | 2 | 2 | pinneapple_simulation/numerical_solvers/fvm.py |
| S4 | Spectral (Fourier pseudo-spectral) | validated | 3 | 2 | pinneapple_simulation/numerical_solvers/spectral.py |
| S5 | Lattice Boltzmann D2Q9 / D3Q19 (+ Smagorinsky LES) | validated | 2 | 21 | pinneapple_simulation/numerical_solvers/lbm.py |
| S6 | SPH / ISPH / DFSPH | untested | 0 | 0 | pinneapple_simulation/numerical_solvers/sph.py, pinneapple_simulation/numerical_solvers/isph.py, pinneapple_simulation/numerical_solvers/dfsph.py |
| S7 | Meshfree RBF collocation (Kansa) | validated | 1 | 1 | pinneapple_simulation/numerical_solvers/meshfree.py |
| S8 | X-TFC (IVP, PDE, subdomains) | validated | 4 | 8 | pinneapple_simulation/numerical_solvers/xtfc_ivp.py, pinneapple_simulation/numerical_solvers/xtfc_pde.py, pinneapple_simulation/numerical_solvers/xtfc_subdomain.py |
| S9 | Discrete-time PINN / implicit Runge-Kutta Gauss-Legendre | tested | 0 | 2 | pinneapple_simulation/numerical_solvers/discrete_time_pinn.py, pinneapple_simulation/numerical_solvers/irk_gauss_legendre.py |
| S10 | Stiff kinetics ODE | validated | 1 | 0 | pinneapple_simulation/numerical_solvers/stiff_kinetics_ode.py |
| S11 | TVD advection | validated | 1 | 0 | pinneapple_simulation/numerical_solvers/tvd_advection.py |
| S12 | Immersed boundary (FDM) | validated | 1 | 1 | pinneapple_simulation/numerical_solvers/immersed_boundary_fdm.py |
| S13 | Beam BVP FDM / 3D elasticity FDM | validated | 2 | 0 | pinneapple_simulation/numerical_solvers/beam_bvp_fdm.py, pinneapple_simulation/numerical_solvers/elasticity3d_fdm.py |
| S14 | Eddy current FDM | validated | 3 | 3 | pinneapple_simulation/numerical_solvers/eddy_current_fdm.py |
| S15 | Bekker-Wong terramechanics | validated | 1 | 19 | pinneapple_simulation/numerical_solvers/bekker_wong.py, pinneapple_simulation/particle_dynamics/terramechanics.py |
| S16 | N-body (REBOUND) | validated | 1 | 2 | pinneapple_simulation/numerical_solvers/nbody_rebound.py |
| S17 | JAX-CFD | tested | 0 | 3 | pinneapple_simulation/numerical_solvers/jax_cfd_solver.py |
| S18 | Particle dynamics: rigid body, MPM, molecular dynamics | tested | 0 | 13 | pinneapple_simulation/particle_dynamics/ |
| S19 | Signal decomposition: CEEMDAN, EEMD, VMD, HHT, SSA, SST, STL, FFT, wavelet | tested | 0 | 1 |  |
| S20 | Co-simulation (graph + engine, algebraic loops by Tarjan SCC) | tested | 0 | 11 | pinneapple_systems/cosimulation/ |
| S21 | Digital twin state estimation: KF/EKF/UKF/EnKF, prognostics, signal reconstruction | tested | 2 | 39 | pinneapple_systems/digital_twin/, pinneapple_neural/architectures/classical_ts/ |
| S22 | 1D process components (pipe network, transient pipe, heat exchanger, valve, reaction, polytropic path, compressor map) | tested | 0 | 3 |  |
| S23 | Explicit equation-system solver (evaluation order) | tested | 0 | 1 | pinneapple_systems/process_components/explicit_equation_system.py |
| S24 | External solvers: OpenFOAM, FEniCS, ANSYS/CFD formats, MATLAB, Modelica/FMI, MuJoCo, Genesis, TurboDesigner | validated | 1 | 80 | pinneapple_simulation/external_solvers/ |
| S25 | Causal telemetry conditioning (Hampel one-step, rate limit, first-order filter, gaps) | validated | 1 | 6 | pinneapple_systems/digital_twin/conditioning.py |
| S26 | 2D shallow-water finite volumes (MUSCL + HLL, wet/dry, walls, gates, virtual sensors, health checks) | validated | 1 | 3 | pinneapple_simulation/numerical_solvers/shallow_water_fv.py |
| S27 | ETDRK4 exponential integrator for stiff semilinear PDEs (contour phi-functions, blow-up guard) | validated | 5 | 1 | pinneapple_simulation/numerical_solvers/etdrk4.py |
| S28 | Stable fluids 2D (semi-Lagrangian advection, spectral projection, vorticity confinement) | validated | 2 | 4 | pinneapple_simulation/numerical_solvers/stable_fluids.py |
| S29 | 2-D A_z magnetostatics, P1 FEM with Newton (nonlinear B-H, magnets, (anti-)periodic constraints) | validated | 1 | 6 | pinneapple_simulation/electromagnetics/fem.py, pinneapple_simulation/electromagnetics/mesh.py, pinneapple_simulation/electromagnetics/materials.py |
| S30 | Time-harmonic 2-D eddy currents (A_z, solid conductors with imposed net currents) | validated | 1 | 1 | pinneapple_simulation/electromagnetics/eddy.py |
| S31 | Surface-PM machine model: Maxwell-stress and co-energy torque, cogging, back-EMF, losses | validated | 1 | 8 | pinneapple_simulation/electromagnetics/machine.py, pinneapple_simulation/electromagnetics/postprocess.py |
| S32 | 2-D electrostatics, P1 FEM (capacitance from field energy) | validated | 2 | 0 | pinneapple_simulation/electromagnetics/electrostatics.py |
| S33 | Biot-Savart integration of straight filament segments, loop and solenoid fields | validated | 3 | 0 | pinneapple_simulation/electromagnetics/statics.py |
| S34 | Mie series for dielectric and perfectly conducting spheres | validated | 2 | 0 | pinneapple_simulation/electromagnetics/mie.py |
| S35 | Fresnel coefficients and characteristic matrices of planar multilayers | validated | 2 | 0 | pinneapple_simulation/electromagnetics/waves.py |
| S36 | Jiles-Atherton hysteresis model (RK4 along an applied field path) | validated | 1 | 0 | pinneapple_simulation/electromagnetics/hysteresis.py |
| S37 | Dispersive media (Drude, Lorentz, Debye) and their auxiliary-differential-equation coefficients | validated | 1 | 0 | pinneapple_simulation/electromagnetics/media.py |
| S38 | Maxwell physics checks of predicted fields (Gauss, Faraday, Ampere-Maxwell, continuity, Poynting, interfaces, reciprocity, passivity) | validated | 4 | 2 | pinneapple_physics/maxwell_checks.py |
| S39 | Density-based topology optimisation (SIMP, filter, projection, OC, MMA; compliance, thermal, Stokes-Brinkman, machine rotor) | validated | 1 | 2 | pinneapple_design/topology/ |
| S40 | Yee FDTD in 1-D/2-D/3-D (CPML, Courant check, Drude/Lorentz/Debye ADE, PEC/PMC/periodic/Bloch, TF/SF plane waves, DFT monitors, sub-cell averaging) | validated | 4 | 6 | pinneapple_simulation/electromagnetics/fdtd.py |
| S41 | Near-to-far-field transform on a closed box (FDTD radiation patterns) | validated | 1 | 0 | pinneapple_simulation/electromagnetics/fdtd.py |
| T1 | Base trainer: AMP, gradient accumulation, auto batch size, CUDA graphs, profiler | tested | 0 | 9 | pinneapple_neural/trainer/trainer.py, pinneapple_neural/trainer/hpc.py |
| T2 | Distributed: DDP, FSDP, SLURM, torchrun, PowerSGD | untested | 0 | 0 | pinneapple_neural/trainer/distributed.py, pinneapple_neural/trainer/hpc.py |
| T3 | Causal training | untested | 0 | 0 | pinneapple_neural/trainer/causal.py |
| T4 | Time marching | untested | 0 | 0 | pinneapple_neural/trainer/time_marching.py |
| T5 | Two phases Adam -> L-BFGS + L-BFGS finetune | tested | 0 | 6 | pinneapple_neural/trainer/two_phase.py, pinneapple_neural/trainer/lbfgs_finetune.py |
| T6 | Multi-restart | untested | 0 | 0 | pinneapple_neural/trainer/multistart.py |
| T7 | Adaptive residual-based collocation | untested | 0 | 0 | pinneapple_neural/trainer/collocation.py |
| T8 | Loss balancing: ReLoBRaLo, SoftAdapt, PCGrad, Augmented Lagrangian, Inverse Dirichlet, LR Annealing, AutoBalancer | validated | 1 | 6 | pinneapple_neural/trainer/loss_balancer.py |
| T9 | Weight schedulers: SA-PINN, GradNorm, LossRatio, NTK | tested | 0 | 2 | pinneapple_neural/trainer/weight_scheduler.py |
| T10 | Domain decomposition: XPINN, DoMINO/D3M | tested | 0 | 2 | pinneapple_neural/architectures/pinns/xpinn.py, pinneapple_physics/pinn_solver/domino.py |
| T11 | Inverse PINN (learnable physical parameters) | tested | 0 | 3 | pinneapple_neural/trainer/inverse_pinn.py |
| T12 | Gray-box + symbolic regression (PySR) | untested | 0 | 0 | pinneapple_neural/trainer/graybox.py |
| T13 | Uncertainty: heteroscedastic, B-PINN, MC Dropout, deep ensembles | tested | 0 | 20 | pinneapple_neural/trainer/heteroscedastic.py, pinneapple_neural/architectures/pinns/bpinn.py, pinneapple_systems/component_modeling/ |
| T14 | SINDy | tested | 0 | 2 | pinneapple_neural/architectures/rom/sindy.py |
| T15 | Adaptive hyperparameter sweep / architecture search (arena NAS) | untested | 0 | 0 | pinneapple_neural/trainer/adaptive_sweep.py, pinneapple_arena/nas.py |
| T16 | Noether trainer (research only) | tested | 0 | 1 | pinneapple_neural/trainer/noether_trainer.py |
| T17 | Post-training advisor / audit | untested | 0 | 0 | pinneapple_neural/trainer/advisor.py, pinneapple_neural/trainer/audit.py |
| T18 | Meta-learning: MAML, Reptile | tested | 0 | 6 | pinneapple_adaptation/meta_learning/ |
| T19 | Transfer learning (freeze, adapters, parametric) | tested | 0 | 3 | pinneapple_adaptation/transfer_learning/ |
| T20 | Self-scaled quasi-Newton: BFGS / SSBFGS / SSBroyden (new, 2026-09-24) | validated | 1 | 3 | pinneapple_neural/trainer/self_scaled_qn.py |
| T21 | Autoresearch loop (fixed-budget trials, keep/revert, random or LLM proposer) | tested | 0 | 2 | pinneapple_arena/autoresearch.py |
| T22 | Large Physics Model blocks (Fourier encoding, multi-scale neighbourhoods, geometry code, ensemble UQ, OOD score, fine-tuning) | tested | 0 | 1 | pinneapple_neural/lpm.py |
| T23 | MultiAdam: per-loss-term Adam moments for multi-term PINN losses | validated | 1 | 3 | pinneapple_neural/trainer/multiadam.py |

### 2.4 Como os números foram obtidos

- Tudo vem de `docs/pt/_mapa/inventario.py`, que lê o código com `ast` (nada é importado ou executado).
- *Público* = nome de nível de módulo que não começa com `_`. Métodos de classes são contados à parte.
- *Testes de um módulo* = funções `test*` em arquivos de `tests/` que importam aquele módulo (`import`, `from … import`, `importlib.import_module`, `pytest.importorskip`). É uma medida de alcance, não de cobertura de linhas.
- O status de validação dos métodos é o de `pinneapple_catalog/method_status.json`, gerado por `scripts/build_method_status.py`.
- Itens ligados a clientes foram excluídos do inventário e do texto.
