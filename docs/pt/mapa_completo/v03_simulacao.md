# Volume 3 — Simulação numérica

> Parte do [Mapa completo do PINNeAPPle](../mapa_completo.md). Gerado por `docs/pt/_mapa/gerar.py` a partir do inventário do commit `e53cdcfb04`.

Solvers clássicos (FDM, FEM, FVM, espectral, SPH, LBM), pontes para solvers externos, eletromagnetismo, voo, multicorpos, veículos e rovers.

## Sumário

1. [`pinneapple_simulation`](#1-pinneapple_simulation)

## 1. `pinneapple_simulation`

### Objetivo

Esta parte cobre três submódulos de `pinneapple_simulation`. `accelerated_foam` acelera as soluções de pressão do OpenFOAM trocando só o chute inicial do solver linear (plugin C++ `pinneWarmStart`, projeção/POD/rede aprendida em Python) e compara com o OpenFOAM puro na mesma tolerância. `electromagnetics` é uma biblioteca de eletromagnetismo em SI: referências em forma fechada, FEM 2-D em A_z, eletrostática, correntes parasitas, FDTD, ondas, Mie, máquinas elétricas (SPM e IPM, acionamento, térmica), antenas Vivaldi com solver externo, células periódicas e compact ranges por óptica física, com suítes de verificação. `geophysics` traz blocos do sistema terrestre verificados contra soluções conhecidas: águas rasas na esfera, equação de Richards e balanço de energia de duas camadas.

Esta parte cobre quatro submódulos de `pinneapple_simulation`. `external_solvers` liga o PINNeAPPle a programas de simulação de fora (OpenFOAM, CalculiX, OpenRadioss, ANSYS Fluent, FEniCS, MATLAB, Modelica/FMI, MuJoCo, Genesis, NVIDIA Newton, FluidX3D, GeoTaichi, TurboDesigner) e lê formatos abertos de CFD/FEM; quase tudo termina num `PhysicalSample` (UPD) para treino e checagem. `flight` é um simulador de voo 6-DOF com aerodinâmica de vortex lattice, trim, piloto automático, decolagem e turbulência. `meshless` compara métodos sem malha (Kansa, RBF-FD, PINN) com FD e FEM no mesmo problema 2-D. `pipelines` encadeia FreeCAD, malha e OpenFOAM num fluxo reproduzível, com validação contra correlações publicadas, varredura e surrogate.

Esta parte cobre a física de corpos e partículas do `pinneapple_simulation`: dinâmica multicorpo própria (`multibody`), simuladores de partículas em PyTorch (`particle_dynamics`: corpos rígidos, MPM, GeoMPM, SPH, dinâmica molecular), robôs no NVIDIA Newton com checagens independentes (`robot_sim`), terramecânica de rodas planetárias (`terramechanics`) e o rover completo em terreno deformável (`vehicle`, com o VIPER). O fio comum é sempre ter uma referência independente para checar o resultado: modelo analítico, balanço de energia, dados publicados ou substituto PINN comparado com o modelo de origem.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 298 | 66.085 | 364 | 931 | 1.119 (109 arquivos) | 113 |

Pasta de testes dedicada: `tests/pinneapple_simulation` (16 funções de teste).

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `accelerated_foam` | Acelera solves de pressão do OpenFOAM com chutes iniciais melhores (plugin C++ e ferramentas Python), mantendo o solver e o teste de parada do OpenFOAM. | 9 | 46 | 20 |
| `electromagnetics` | Eletromagnetismo em SI: formas fechadas, FEM 2-D, FDTD, ondas, máquinas elétricas, antenas, estruturas periódicas e refletores, com verificação. | 56 | 345 | 88 |
| `external_solvers` | Pontes para solvers e ferramentas externas (execução, montagem de casos, leitura de resultados) e leitores de formatos CFD/FEM, com saída em `PhysicalSample`. | 66 | 192 | 151 |
| `flight` | Dinâmica de voo 6-DOF: aeronave rígida com vortex lattice, motores, trem de pouso, trim, controle, decolagem, turbulência e cenas 3-D. | 17 | 108 | 19 |
| `geophysics` | Modelos do sistema terrestre: águas rasas na esfera, equação de Richards no solo e balanço de energia climático de duas camadas. | 4 | 19 | 11 |
| `meshless` | Métodos sem malha e com malha resolvendo os mesmos problemas 2-D, e o estudo de refinamento que os compara. | 8 | 19 | 22 |
| `multibody` | Árvore multicorpo de base flutuante com juntas de revolução e utilitários de braço de base fixa. | 3 | 18 | 45 |
| `numerical_solvers` |  | 73 | 295 | 206 |
| `particle_dynamics` | Simuladores de partículas diferenciáveis em PyTorch: corpos rígidos, MPM, GeoMPM, SPH e dinâmica molecular. | 8 | 21 | 151 |
| `pipelines` | Pipelines de ponta a ponta que encadeiam CAD (FreeCAD), malhador e solver (OpenFOAM) num fluxo checado e reproduzível. | 14 | 59 | 14 |
| `robot_sim` | Robôs simulados no NVIDIA Newton (com backend Isaac Sim) e checados contra o modelo multicorpo próprio. | 6 | 17 | 39 |
| `rover` |  | 17 | 88 | 21 |
| `terramechanics` | Contato roda-solo para rovers: solos, Wong-Reece, terreno deformável, bancada e substitutos PINN. | 8 | 38 | 34 |
| `vehicle` | Veículo planetário com rodas (VIPER) do spec ao CAD, simulação, dashboard e animação. | 8 | 28 | 38 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_simulation`

pinneapple_simulation — Physics simulation back-ends and data generation.

<small>`pinneapple_simulation/__init__.py` · 163 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `simulate` | função | `(scenario, **kwargs) -> Any` | Run a named physics scenario and return its output. |
| `generate_data` | função | `(scenario, n_samples=10, **kwargs) -> list` | Generate a list of simulation outputs for training data. |

#### `pinneapple_simulation.accelerated_foam`

Pacote que acelera as soluções de pressão do OpenFOAM trocando só o chute inicial do solver linear (plugin C++ `pinneWarmStart` + ferramentas Python); reexporta `LduSystem`, `ProjectionWarmStart`, `PODWarmStart`, `galerkin` e os leitores de sistemas.

<small>`pinneapple_simulation/accelerated_foam/__init__.py` · 41 linhas · testes: 15</small>

#### `pinneapple_simulation.accelerated_foam.benchmark`

As comparações por trás de `docs/core_concepts/openfoam_acceleration.md`: OpenFOAM puro contra `pinneWarmStart` (mesmo solver interno e tolerâncias) e o benchmark offline dos sistemas extraídos. [sem teste]

<small>`pinneapple_simulation/accelerated_foam/benchmark.py` · 387 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `openfoam_comparison` | função | `(work_dir, lib, n=128, end_time=0.5, inners=('PCG', 'GAMG'), tolerance=1e-06, rel_tol=0.0…)` | Roda a cavidade com tampa móvel (`icoFoam`) com solves de pressão puros e com chute aquecido, e devolve uma linha por execução (iterações, tempos, diferenças de campo). |
| `solver_entries` | função | `(fv_solution, field_p) -> dict` | Lê as entradas `chave valor;` do bloco `solvers { <field_p> { ... } }` de um texto `fvSolution`. |
| `set_solver_entries` | função | `(fv_solution, field_p, entries) -> str` | Substitui o corpo do bloco do solver de `field_p` em um texto `fvSolution`, sem mexer nas outras entradas. |
| `prepare_tutorial` | função | `(tutorial_dir, case_dir, p_solver, field_p, libs=(), setup=('blockMesh',), end_time=None,…)` | Copia um tutorial instalado do OpenFOAM, define o solver de pressão, força escrita ASCII com 17 dígitos e roda os comandos de preparo (`blockMesh` etc.). |
| `read_solver_dict` | função | `(tutorial_dir, field_p) -> dict` | Lê os controles do solver de `field_p` definidos pelo próprio tutorial (como strings). |
| `tutorial_comparison` | função | `(tutorial_dir, work_dir, lib, application, field_p, compare_fields=('U',), setup=('blockM…)` | Roda um tutorial instalado de três jeitos: solver original, o mesmo solver envolvido por `pinneWarmStart` e uma referência com tolerância muito apertada. |
| `offline_benchmark` | função | `(systems, guesses, solver_specs, tolerance=1e-06, rel_tol=0.0) -> list[dict]` | Resolve de novo, em ordem, os sistemas de uma execução com cada par (chute inicial, solver) no teste de parada do OpenFOAM. |
| `default_solver_specs` | função | `(include_amg=True, include_torch=True) -> dict` | Dicionário padrão de solvers do benchmark offline: PCG-DIC e PCG-Jacobi em NumPy, mais AMG (pyamg) e CG em PyTorch quando instalados. |
| `default_guesses` | função | `(n_snapshots=8) -> dict` | Dicionário padrão de chutes iniciais: o campo anterior (padrão do OpenFOAM) e a projeção nas últimas `n_snapshots` soluções. |
| `markdown_table` | função | `(rows, columns=None) -> str` | Formata uma lista de linhas (dicts) como tabela Markdown para a documentação. |

#### `pinneapple_simulation.accelerated_foam.build`

Compila a biblioteca de solver linear `pinneWarmStart` (fonte C++ em `plugin/`) contra um OpenFOAM instalado e guarda o `.so` em cache em `~/.cache/pinneapple/accelerated_foam`.

<small>`pinneapple_simulation/accelerated_foam/build.py` · 155 linhas · testes: 15</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `include_dirs` | função | `(openfoam_src=None, work=None) -> list[Path]` | Diretórios de cabeçalhos do OpenFOAM (de `$FOAM_SRC` ou de uma árvore de fontes); levanta `FileNotFoundError` se não houver nenhum. |
| `compile_flags` | função | `(label_size=None, precision=None, api=None) -> list[str]` | Flags de compilação C++ conforme tamanho de label, precisão e versão da API do OpenFOAM (lidas das variáveis `WM_*`/`FOAM_API` quando não informadas). |
| `build_plugin` | função | `(openfoam_src=None, out_dir=None, force=False, cxx=None) -> Path` | Compila `libpinneWarmStart.so` e devolve o caminho (usa o cache, a menos que `force`). |
| `find_plugin` | função | `(openfoam_src=None) -> Path \| None` | Devolve `$PINNEAPPLE_WARMSTART_LIB` se definido, senão uma compilação em cache; `None` se não der para compilar. |

#### `pinneapple_simulation.accelerated_foam.case`

Casos pequenos do OpenFOAM escritos do zero (cavidade com tampa, `icoFoam`), um executor de comandos e um parser de log para os benchmarks de solução de pressão.

<small>`pinneapple_simulation/accelerated_foam/case.py` · 395 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FieldSolves` | classe | `(n_solves, iterations, per_solve, initial_residuals, final_residuals)` | Resumo dos solves lineares de um campo lidos de um log (número de solves, iterações, resíduos inicial e final). |
| `RunResult` | classe | `(case_dir, wall_time, execution_time, solves, log)` | Resultado de uma execução: pasta do caso, tempo de parede, ExecutionTime, solves por campo e o texto do log. |
| `find_bashrc` | função | `(bashrc=None) -> str \| None` | Localiza o script de ambiente do OpenFOAM (`bashrc` dado, `$FOAM_BASHRC` ou um local usual). |
| `openfoam_available` | função | `(solver='icoFoam') -> bool` | Diz se o solver indicado do OpenFOAM pode ser executado (no PATH ou via script de ambiente). |
| `plain_solver` | função | `(kind='PCG', tolerance=1e-06, rel_tol=0.0, **extra) -> dict` | Dicionário do solver de pressão do próprio OpenFOAM: `PCG` (pré-condicionador DIC) ou `GAMG`. |
| `warm_solver` | função | `(inner='PCG', tolerance=1e-06, rel_tol=0.0, guess='projection', n_snapshots=8, **extra) -…)` | Dicionário do wrapper `pinneWarmStart` em volta do mesmo solver interno e das mesmas tolerâncias de `plain_solver`. |
| `write_cavity` | função | `(case_dir, n=64, end_time=0.25, nu=0.01, lid=1.0, courant=0.5, p_solver=None, p_final_sol…)` | Escreve uma cavidade 2-D com tampa móvel (quadrado unitário, `n` x `n` células) para o `icoFoam`. |
| `run_command` | função | `(case_dir, command, bashrc=None, log_name=None, env=None) -> tuple[str, float]` | Executa um comando do OpenFOAM na pasta do caso e devolve (texto do log, segundos de parede); levanta erro em caso de falha. |
| `parse_log` | função | `(log) -> tuple[dict[str, FieldSolves], float]` | Extrai de um log de solver os solves lineares por campo e o ExecutionTime final. |
| `run_case` | função | `(case_dir, solver='icoFoam', bashrc=None, mesh=True, env=None) -> RunResult` | Roda `blockMesh` (opcional) e o solver, e devolve o log interpretado como `RunResult`. |
| `time_dirs` | função | `(case_dir) -> list[str]` | Lista as pastas de tempo (nomes numéricos) de um caso. |
| `read_field` | função | `(case_dir, name, time_name=None) -> np.ndarray` | Lê o campo interno `name` em um instante (padrão: o último) como array NumPy. |
| `field_difference` | função | `(case_a, case_b, name, time_name=None) -> float` | Diferença L2 relativa de um campo entre duas execuções no mesmo instante. |

#### `pinneapple_simulation.accelerated_foam.guess_server`

Serve chutes iniciais calculados em Python a um solver OpenFOAM em execução (`pinneWarmStart` com `guess external`) por dois named pipes.

<small>`pinneapple_simulation/accelerated_foam/guess_server.py` · 138 linhas · testes: 15</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GuessServer` | classe | `(pipe_dir, guessers)` | Servidor em thread de fundo que responde pedidos de chute em `pipe_dir/request` usando um objeto chutador por campo; usado como context manager. |

#### `pinneapple_simulation.accelerated_foam.ldu`

Lê os sistemas `lduMatrix` gravados por `pinneWarmStart` e os converte para matrizes esparsas SciPy/PyTorch, com o resíduo normalizado do OpenFOAM.

<small>`pinneapple_simulation/accelerated_foam/ldu.py` · 197 linhas · testes: 15</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LduSystem` | classe | `(lower_addr, upper_addr, diag, upper, source, psi0, psi, lower, meta, _csr)` | Um sistema linear `A x = b` no formato em que o OpenFOAM o entrega ao solver (diag, upper, lower, endereçamento), com produto, conversão para CSR e para PyTorch. |
| `from_csr` | função | `(A, b) -> LduSystem` | Cria um `LduSystem` a partir de uma matriz esparsa simétrica, para testes e casos sintéticos. |
| `norm_factor` | função | `(system, x, Ax=None) -> float` | Fator de normalização do resíduo usado pelo OpenFOAM. |
| `normalised_residual` | função | `(system, x, nf=None) -> float` | Resíduo normalizado `sum\|b - A x\| / normFactor`, o mesmo que o OpenFOAM imprime e compara com `tolerance`. |
| `read_system` | função | `(directory) -> LduSystem` | Lê um sistema gravado por `pinneWarmStart` (`meta.txt` e `system.bin`). |
| `read_dump` | função | `(dump_dir, field_name='p') -> list[LduSystem]` | Lê todos os sistemas de um campo em uma pasta de dump, na ordem dos solves. |

#### `pinneapple_simulation.accelerated_foam.learned`

Correção aprendida do chute inicial: uma pequena rede em grafo (PyTorch) sobre o grafo da matriz, com salvaguarda que mantém a projeção se a rede piorar o resíduo. [experimental] O docstring avisa que o efeito esperado é modesto.

<small>`pinneapple_simulation/accelerated_foam/learned.py` · 162 linhas · testes: 15</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GraphGuessNet` | classe | `(hidden=32, layers=4, seed=0)` | Rede de passagem de mensagens sobre o grafo normalizado da matriz, que prevê a correção por célula. |
| `LearnedWarmStart` | classe | `(n_snapshots=8, hidden=32, layers=4, seed=0)` | Chute por projeção mais correção aprendida (com salvaguarda), com a mesma interface `fit`/`guess`/`update` dos outros warm starts. |
| `projection_inputs` | função | `(systems, n_snapshots=8) -> list[np.ndarray]` | Calcula o chute por projeção de cada sistema de uma execução a partir das próprias soluções convergidas (dados de treino). |

#### `pinneapple_simulation.accelerated_foam.solvers`

Solvers de Krylov (PCG em NumPy e CG em PyTorch) para os sistemas de pressão extraídos, com o mesmo teste de parada do OpenFOAM, para comparar contagens de iterações com os logs.

<small>`pinneapple_simulation/accelerated_foam/solvers.py` · 230 linhas · testes: 15</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SolveResult` | classe | `(x, iterations, initial_residual, final_residual, wall_time, converged, history, setup_ti…)` | Resultado de um solve: solução, iterações, resíduos inicial e final, tempo, convergência, histórico e tempo de preparo. |
| `jacobi` | função | `(system)` | Pré-condicionador de Jacobi (divisão pela diagonal). |
| `dic` | função | `(system)` | Pré-condicionador DIC do OpenFOAM (Cholesky incompleto diagonal), aplicado por duas soluções triangulares esparsas. |
| `amg` | função | `(system, **kwargs)` | Pré-condicionador AMG por agregação suavizada do pyamg (dependência opcional), aplicado a `-A`. |
| `pcg` | função | `(system, x0=None, preconditioner='dic', tolerance=1e-06, rel_tol=0.0, max_iter=1000, min_…)` | Gradiente conjugado pré-condicionado em NumPy com o algoritmo e o teste de parada do PCG do OpenFOAM. |
| `torch_pcg` | função | `(system, x0=None, tolerance=1e-06, rel_tol=0.0, max_iter=1000, device=None, dtype='float6…)` | CG com pré-condicionador de Jacobi em PyTorch (GPU quando houver), em float64 ou precisão mista. |

#### `pinneapple_simulation.accelerated_foam.warm_start`

Chutes iniciais para uma sequência de sistemas de pressão: projeção de Galerkin nas últimas soluções e em uma base POD de execuções anteriores, nunca piores que o campo atual na norma de energia.

<small>`pinneapple_simulation/accelerated_foam/warm_start.py` · 121 linhas · testes: 15</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ProjectionWarmStart` | classe | `(n_snapshots=8)` | Projeção de Galerkin de cada novo sistema no espaço das últimas `n_snapshots` soluções e do campo atual (o mesmo algoritmo do plugin C++). |
| `PODWarmStart` | classe | `(energy=0.99999, max_rank=32, n_recent=0)` | Projeção de Galerkin em uma base POD de soluções guardadas (outras execuções ou passos anteriores) mais o campo atual. |
| `galerkin` | função | `(system, basis, drop=1e-12) -> np.ndarray \| None` | Resolve `(V^T A V) c = V^T b` e devolve `x = V c`, descartando colunas dependentes; `None` se a base estiver vazia. |

#### `pinneapple_simulation.electromagnetics`

Pacote de eletromagnetismo em SI: referências em forma fechada, materiais, solvers FEM 2-D em A_z, FDTD, ondas e máquinas elétricas; reexporta as classes e funções principais dos submódulos.

<small>`pinneapple_simulation/electromagnetics/__init__.py` · 67 linhas · testes: 88</small>

#### `pinneapple_simulation.electromagnetics.analytic`

Referências em forma fechada para verificar os solvers 2-D em A_z: fio, coaxial, cilindro e anel magnetizados, efeito pelicular e solução por subdomínios em camadas radiais.

<small>`pinneapple_simulation/electromagnetics/analytic.py` · 226 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Layer` | classe | `(r_in, r_out, mu_r, Br_cos, Br_sin, J_cos, J_sin)` | Uma coroa circular `r_in < r < r_out` com permeabilidade relativa, remanência radial e densidade de corrente de um harmônico. |
| `RadialLayers` | classe | `(layers, p)` | Solução exata de um harmônico `p >= 1` em camadas concêntricas (potencial, campo polar e torque). |
| `wire_field` | função | `(r, current, radius, R_outer)` | A_z e \|B\| de um fio redondo com corrente uniforme dentro de um círculo com A = 0. |
| `coax_field` | função | `(r, current, a, b, c)` | \|B\| de um cabo coaxial (condutor interno +I, externo -I). |
| `magnetized_cylinder` | função | `(x, y, Br, a, R, alpha=0.0)` | A_z de um cilindro uniformemente magnetizado no ar com A = 0 em um raio externo; devolve (A, B interno). |
| `magnetized_ring` | função | `(x, y, Br, a, b, r0, R, alpha=0.0)` | A_z de um anel uniformemente magnetizado entre dois círculos com A = 0, por superposição de cilindros. |
| `skin_effect_wire` | função | `(r, current, radius, sigma, omega, mu=MU0)` | Densidade de corrente fasorial e impedância por metro de um fio redondo com efeito pelicular (funções de Bessel). |
| `round_conductor_eddy_loss_low_freq` | função | `(B_peak, omega, sigma, a) -> float` | Perda média por correntes parasitas por metro em um condutor redondo sob campo transversal uniforme, no limite de baixa frequência. |

#### `pinneapple_simulation.electromagnetics.antenna`

Antenas impressas end-fire (Vivaldi antipodal) com solver FEM externo (EMerge): limites físicos, checagens, executor, surrogates, laço de projeto e visualização 3-D.

<small>`pinneapple_simulation/electromagnetics/antenna/__init__.py` · 30 linhas · testes: 31</small>

#### `pinneapple_simulation.electromagnetics.antenna.ava`

Modelo paramétrico aberto de antena Vivaldi antipodal (AVA) para o solver FEM EMerge, escrito do zero, com CLI; o EMerge é dependência opcional.

<small>`pinneapple_simulation/electromagnetics/antenna/ava.py` · 386 linhas · testes: 31</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `InvalidGeometry` | classe | `()` | Exceção para parâmetros que não descrevem uma antena construível (metal sobreposto, fendas atravessando o braço etc.). |
| `AVAParams` | classe | `(Lb, W, Wa, R, xs, Wg, Lo, nslot, Ls, ws, ps, dir_a, dir_b, dir_x, res, er, tand, h, wf, …)` | Parâmetros de projeto da AVA em milímetros (placa, flare, fendas, diretor, substrato, malha). |
| `Layout` | classe | `(params, top, bottom, director, wf, min_metal)` | Polígonos condutores (topo, fundo, diretor) e dimensões derivadas de um projeto. |
| `microstrip_width` | função | `(z0, er, h) -> float` | Largura de uma microfita de impedância dada (Pozar eq. 3.197). |
| `layout` | função | `(p) -> Layout` | Gera os polígonos condutores de um projeto; levanta `InvalidGeometry` se não for construível. |
| `use_superlu` | função | `() -> None` | Força o solver direto SuperLU do EMerge. |
| `nearfield_planes` | função | `(p) -> dict[str, dict[str, np.ndarray]]` | Os dois planos de corte de campo próximo do projeto (plano da placa e plano do eixo). |
| `plane_points` | função | `(pl) -> np.ndarray` | Pontos `(nv, nu, 3)` de um plano de `nearfield_planes`. |
| `simulate` | função | `(params, freqs_ghz, out_dir=None, *, superlu=True, log='WARNING') -> list[dict[str, Any]]` | Monta o projeto no EMerge, resolve cada frequência e devolve uma linha por frequência. |
| `main` | função | `(argv=None) -> int` | Ponto de entrada da CLI que simula um projeto e grava `result.json`. |

#### `pinneapple_simulation.electromagnetics.antenna.bounds`

Limites físicos de diretividade de antenas end-fire (Hansen-Woodyard e ordinária) e de abertura broadside, comprimento mínimo e frequência de viabilidade.

<small>`pinneapple_simulation/electromagnetics/antenna/bounds.py` · 87 linhas · testes: 23</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `to_dbi` | função | `(x)` | Converte ganho linear para dBi. |
| `from_dbi` | função | `(g)` | Converte dBi para ganho linear. |
| `wavelength` | função | `(f_hz)` | Comprimento de onda no vácuo em metros. |
| `endfire_directivity` | função | `(length_m, f_hz, kind='hw')` | Diretividade máxima (linear) de uma antena end-fire de comprimento dado. |
| `endfire_directivity_dbi` | função | `(length_m, f_hz, kind='hw')` | O mesmo limite de diretividade end-fire em dBi. |
| `aperture_gain` | função | `(area_m2, f_hz, efficiency=1.0)` | Ganho de abertura broadside (linear). |
| `aperture_gain_dbi` | função | `(area_m2, f_hz, efficiency=1.0)` | Ganho de abertura broadside em dBi. |
| `min_length` | função | `(gain_dbi, f_hz, kind='hw')` | Menor comprimento end-fire cujo limite atinge um ganho alvo. |
| `feasibility_frequency` | função | `(gain_dbi, length_m, kind='hw')` | Menor frequência em que uma antena end-fire de comprimento dado pode atingir o ganho alvo. |

#### `pinneapple_simulation.electromagnetics.antenna.checks`

Checagens de consistência de um resultado de antena (ganho realizado, eficiência, pico vs. boresight, limite end-fire, S11 interpolado, fechamento de potência) como `Check`/`TrustReport` de `pinneapple_analysis.trust`.

<small>`pinneapple_simulation/electromagnetics/antenna/checks.py` · 172 linhas · testes: 23</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check_realized_gain` | função | `(g_dbi, g_real_dbi, s11_db, *, tol_db=0.1) -> Check` | Confere o ganho realizado contra ganho e S11. |
| `check_efficiency_ratio` | função | `(g_dbi, d_dbi, eff, *, tol_db=0.1) -> Check` | Confere a eficiência de radiação como G/D. |
| `check_efficiency_le_one` | função | `(eff, *, tol=0.01) -> Check` | Checagem crítica: eficiência de radiação no máximo 1. |
| `check_peak_ge_boresight` | função | `(peak_dbi, bore_dbi, *, tol_db=0.05) -> Check` | Confere que o ganho de pico não é menor que o de boresight. |
| `check_directivity_limit` | função | `(d_dbi, length_m, f_hz, *, margin_db=0.5, kind='hw') -> Check` | Confere a diretividade abaixo do limite end-fire para o comprimento. |
| `check_s11_resolved` | função | `(n_solved, *, interpolated, min_points=5) -> Check` | Exige um número mínimo de frequências resolvidas quando o S11 vem de interpolação racional. |
| `check_power_closure` | função | `(p_pattern, p_surface, *, rtol=0.1) -> Check` | Compara a potência do padrão de campo distante com o fluxo de Poynting no campo próximo. |
| `antenna_checks` | função | `(row, *, length_m=None, n_solved=None, interpolated=False, min_points=5, tol_db=0.1, marg…)` | Aplica todas as checagens a uma linha da tabela e devolve um `TrustReport`. |

#### `pinneapple_simulation.electromagnetics.antenna.design`

Laço de projeto da antena: DOE (LHS/Sobol), NSGA-II nos surrogates, aprendizado ativo verificado pelo solver, correção de viés multifidelidade e cartão de confiança por previsão.

<small>`pinneapple_simulation/electromagnetics/antenna/design.py` · 326 linhas · testes: 23</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OptimizationOutcome` | classe | `(designs, Gmin, S11max, area_cm2, n_evals, wall_s, method, notes)` | Resultado da otimização: projetos, ganho mínimo, S11 máximo, área, avaliações, tempo e método. |
| `ALRound` | classe | `(round, n_train_designs, new_tags, rmse_prospective, mean_std, coverage95)` | Registro de uma rodada de aprendizado ativo (tamanho do treino, novos projetos, erro prospectivo, cobertura). |
| `ava_transform` | função | `(sample, *, fixed=None) -> dict[str, Any]` | Converte uma amostra crua do espaço de projeto em parâmetros da AVA (liga/desliga diretor e fendas). |
| `is_buildable` | função | `(params) -> bool` | Diz se os parâmetros geram um layout de AVA válido. |
| `training_box` | função | `(designs, keys=('Lb', 'W', 'Wa'), ranges=None) -> dict[str, tuple[float, float]]` | Estreita as faixas de busca para o intervalo coberto pelos projetos de treino. |
| `doe` | função | `(n, ranges=None, *, method='lhs', seed=0, transform=ava_transform, valid=is_buildable, ma…)` | Gera `n` projetos válidos de uma amostra space-filling, contando os sorteados e inválidos. |
| `band_metrics` | função | `(models, params, freqs, *, gain='Greal_bore_dBi', s11='S11_dB') -> dict[str, np.ndarray]` | Ganho mínimo previsto, seu desvio e S11 máximo na banda para cada projeto. |
| `optimize` | função | `(models, freqs, *, ranges=None, fixed=None, budget=600, seed=0, method='nsga2', s11_max=-…)` | Busca multiobjetivo (NSGA-II) nos surrogates de ganho e S11. |
| `active_learning` | função | `(simulate, table, freqs, *, target='Greal_bore_dBi', rounds=2, per_round=2, ranges=None, …)` | Aprendizado ativo guiado pela variância, com verificação pelo simulador. |
| `multi_fidelity` | função | `(coarse, fine, target) -> dict[str, Any]` | Viés grosso-para-fino de uma grandeza por frequência, validado deixando um projeto fino de fora. |
| `prediction_card` | função | `(model, params, f_ghz, train, *, target='Greal_bore_dBi', std_max_db=1.0, lodo_rmse=None,…)` | Cartão de confiança (`TrustReport`) de uma previsão do surrogate. |

#### `pinneapple_simulation.electromagnetics.antenna.package`

Lê um pacote de estudo externo (tabela de resultados + script opcional) de uma pasta local, sem copiar nada para o repositório, e reexecuta um de seus projetos.

<small>`pinneapple_simulation/electromagnetics/antenna/package.py` · 83 linhas · testes: 23</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Package` | classe | `(root, table_path, table, script, argv, flags, length_col)` | Pacote de estudo carregado: pasta, tabela, script, argumentos e flags; cria o simulador e descreve o conteúdo. |
| `load_package` | função | `(root) -> Package` | Carrega o pacote de uma pasta (usando `package.json` se houver). |

#### `pinneapple_simulation.electromagnetics.antenna.plots`

Figuras matplotlib de um estudo de antena end-fire (diretividade vs. L/lambda com limites, RMSE LODO, paridade, layout). [sem teste]

<small>`pinneapple_simulation/electromagnetics/antenna/plots.py` · 87 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `directivity_vs_length` | função | `(table, ax=None, *, column='D_peak_dBi', target_dbi=10.0, length_col='Lb', label=None)` | Plota uma coluna contra L/lambda com os limites Hansen-Woodyard e ordinário e o alvo. |
| `lodo_rmse_bars` | função | `(results, ax=None, *, title='leave-one-design-out RMSE')` | Barras de RMSE leave-one-design-out por modelo. |
| `parity_plot` | função | `(result, model, ax=None)` | Gráfico de paridade (previsto vs. simulado) de um modelo. |
| `layout_plot` | função | `(lay, ax=None, *, colors=…)` | Desenha os condutores de topo, fundo e diretor de um `Layout`. |

#### `pinneapple_simulation.electromagnetics.antenna.runner`

Executa um script de simulador paramétrico externo, um processo por amostra (e por frequência), com timeout, memória de pico e fallback para SuperLU, e lê/grava as tabelas de resultados.

<small>`pinneapple_simulation/electromagnetics/antenna/runner.py` · 309 linhas · testes: 34</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RunRecord` | classe | `(tag, freqs, status, returncode, wall_s, max_rss_mb, rows, message, retried)` | Registro de um processo: projeto, frequências, status, código de retorno, tempo, memória e linhas. |
| `CLISimulator` | classe | `(argv, workdir, *, per_frequency=True, timeout=3600.0, params_flag='--params', freq_flag=…)` | Executor que chama um script de linha de comando por projeto (e por frequência) e coleta resultados. |
| `ResultTable` | classe | `(rows)` | Tabela de linhas (projeto, frequência) com acesso por coluna, tag, frequência e filtros. |
| `ava_simulator` | função | `(workdir, **kw) -> CLISimulator` | Um `CLISimulator` sobre o modelo AVA aberto (SuperLU, uma frequência por processo). |
| `read_table` | função | `(path) -> ResultTable` | Lê uma tabela de resultados (CSV, TSV ou separada por espaços). |
| `write_table` | função | `(table, path) -> str` | Grava uma `ResultTable` em arquivo. |

#### `pinneapple_simulation.electromagnetics.antenna.scene3d`

Vistas 3-D de uma Vivaldi antipodal a partir dos campos resolvidos, sobre a cena studio de `pinneapple_tools.visualization.studio` (glTF/USD, visualizador three.js, Blender).

<small>`pinneapple_simulation/electromagnetics/antenna/scene3d.py` · 333 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FarField` | classe | `(f_ghz, theta, phi, U, P_rad, P_acc)` | Campo distante em grade (theta, phi) com potências irradiada e aceita; dá diretividade e ganho em dBi. |
| `antenna_surfaces` | função | `(params, *, copper_mm=COPPER_MM)` | Substrato e cobre do projeto como superfícies `Surface` do studio. |
| `read_farfield` | função | `(path) -> FarField` | Lê um arquivo de campo distante gravado por `ava`. |
| `farfield_metrics` | função | `(ff) -> dict[str, float]` | Diretividade e ganho de pico e de boresight em dBi. |
| `balloon` | função | `(ff, centre, radius, *, g_max=None, dyn_db=30.0, quantity='gain')` | Malha de 'balão' de radiação colorida por ganho e diretividade. |
| `read_nearfield` | função | `(path) -> dict[str, dict[str, np.ndarray]]` | Lê um arquivo de campo próximo gravado por `ava`. |
| `plane_surface` | função | `(points, name, group='cuts')` | Superfície do studio a partir de uma grade estruturada de pontos. |
| `solved_files` | função | `(run_dir) -> dict[float, dict[str, str]]` | Encontra os arquivos de campo distante (e próximo) por frequência numa pasta de execução. |
| `frequency_scene` | função | `(params, files, *, rows=(), dyn_db=30.0, title='') -> tuple[Any, dict]` | Cena com antena, balão de ganho e cortes de \|E\| sobre as frequências resolvidas. |
| `phase_scene` | função | `(params, nf_path, *, n_phase=24, component=0, title='')` | Animação dos planos de corte ao longo de um período em uma frequência. |
| `build_3d` | função | `(params, run_dir, out_dir, *, rows=(), name='design', render=True, size=(960, 540), sampl…)` | Gera todas as saídas 3-D de um projeto a partir dos arquivos de suas execuções. |

#### `pinneapple_simulation.electromagnetics.antenna.study`

Estudos completos de uma antena end-fire como chamadas de biblioteca (usadas pelo exemplo e pelo Flow Studio): limites, estudo do modelo aberto, reprodução de pacote externo e renderização 3-D.

<small>`pinneapple_simulation/electromagnetics/antenna/study.py` · 228 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `limits_report` | função | `(target_dbi=10.0, f_min_ghz=2.0, f_max_ghz=4.0, n=5, lengths_mm=(150.0, 206.0, 260.0)) ->…)` | Comprimentos end-fire mínimos para um ganho alvo na banda e frequência de viabilidade de comprimentos dados (sem solver). |
| `check_summary` | função | `(table, *, n_solved=None, interpolated=False, margin_db=0.5) -> dict[str, Any]` | Contagens de resultados das checagens e de quantas vezes cada uma falhou. |
| `validate_targets` | função | `(table, targets=TARGETS) -> dict[str, sg.ValidationResult]` | Validação leave-one-design-out dos surrogates padrão para cada alvo presente na tabela. |
| `study_figure` | função | `(table, vals, path, *, title='', target_dbi=10.0, column='D_peak_dBi') -> str` | Figura de diretividade vs. L/lambda com limites e RMSE LODO por modelo. |
| `open_study` | função | `(out, *, freqs=(2.0, 2.5, 3.0, 3.5, 4.0), n_designs=16, n_fine=4, coarse_res=0.25, fine_r…)` | Estudo completo do modelo aberto (precisa do EMerge): DOE, execuções, checagens, surrogates, otimização, aprendizado ativo e relatórios. |
| `package_study` | função | `(path, out, *, rerun=None, rerun_freqs=1, interpolated=False, timeout=3600.0, target_dbi=…)` | Reproduz um pacote externo: checagens, limites, LODO, figuras e reexecução de um projeto. |
| `render_3d` | função | `(designs, out, *, freqs=(2.0, 2.5, 3.0, 3.5, 4.0), res=0.25, nf_step=2.0, timeout=3600.0,…)` | Resolve projetos nomeados com campo próximo e gera suas saídas 3-D. |

#### `pinneapple_simulation.electromagnetics.antenna.surrogates`

Surrogates de uma grandeza de antena sobre (projeto, frequência) — média por frequência, Hansen-Woodyard + offset, regressão linear física e GP — e validação honesta (LODO, holdout); a classe base levanta `NotImplementedError` nos métodos abstratos.

<small>`pinneapple_simulation/electromagnetics/antenna/surrogates.py` · 301 linhas · testes: 23 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FrequencyMean` | classe | `()` | Linha de base: média dos alvos de treino em cada frequência. |
| `HWOffset` | classe | `()` | Limite de Hansen-Woodyard mais um offset ajustado em dB. |
| `LinearPhysics` | classe | `(ridge=0.001)` | Regressão ridge sobre as features físicas padronizadas. |
| `GPModel` | classe | `(prior='hw', restarts=4, seed=0, features=None)` | Processo gaussiano sobre as features físicas, com o limite HW (ou constante) como média a priori. |
| `ValidationResult` | classe | `(target, scheme, rows, failed)` | Previsões fora da amostra de cada modelo, com erros, RMSE, cobertura de 95% e melhor modelo. |
| `physical_features` | função | `(params, f_ghz) -> np.ndarray` | Features físicas de cada (projeto, frequência): log(L/lambda), larguras relativas, fendas e diretor. |
| `columns` | função | `(*names)` | Cria uma função de features que usa os parâmetros de projeto nomeados diretamente. |
| `default_models` | função | `() -> dict[str, Callable[[], _Model]]` | Dicionário dos quatro modelos padrão (média, HW + offset, linear, GP). |
| `lodo` | função | `(table, target, models=None) -> ValidationResult` | Validação leave-one-design-out. |
| `holdout` | função | `(table, target, test_tags, models=None) -> ValidationResult` | Treina sem os projetos de teste e prevê esses projetos. |

#### `pinneapple_simulation.electromagnetics.constants`

Constantes físicas do eletromagnetismo em SI, valores CODATA 2022 (NIST), com `mu0` medido após a redefinição do SI de 2019.

<small>`pinneapple_simulation/electromagnetics/constants.py` · 29 linhas · testes: 36</small>

#### `pinneapple_simulation.electromagnetics.eddy`

Correntes parasitas harmônicas no tempo em 2-D (A_z) com condutores sólidos e correntes líquidas impostas, meios lineares, montagem P1 e solver direto. [sem teste direto]

<small>`pinneapple_simulation/electromagnetics/eddy.py` · 93 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TimeHarmonicSolution` | classe | `(A, u, J_nodal, loss, geometry)` | Resultado do solve harmônico: A complexo, tensões por metro dos condutores, J nodal, perda e geometria. |
| `solve_time_harmonic` | função | `(mesh, nu, sigma, omega, conductors, currents, J_source=None, dirichlet=None, dirichlet_v…)` | Resolve A complexo com condutores sólidos (máscaras de elementos) e seus fasores de corrente líquida. |

#### `pinneapple_simulation.electromagnetics.electrostatics`

Eletrostática 2-D em triângulos P1 (`-div(eps grad phi) = rho`) com eletrodos a potencial fixo, reaproveitando o operador do solver magnetostático; só geometria planar.

<small>`pinneapple_simulation/electromagnetics/electrostatics.py` · 61 linhas · testes: 26</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ElectrostaticSolution` | classe | `(phi, E, eps, energy, geometry)` | Resultado eletrostático: potencial, campo E, permissividade, energia e geometria; `D` dá o deslocamento elétrico. |
| `solve_electrostatic` | função | `(mesh, eps_r, electrodes=None, rho=None, dirichlet=None, dirichlet_values=None) -> Electr…)` | Calcula o potencial para permissividades por elemento, densidade de carga e eletrodos `{nome: (nós, potencial)}`. |
| `capacitance` | função | `(sol, V) -> float` | Capacitância por metro `C = 2 W / V^2` de um eletrodo a `V` com os outros aterrados. |

#### `pinneapple_simulation.electromagnetics.fdtd`

Solver FDTD das equações de Maxwell na grade de Yee em 1-D, 2-D e 3-D (um só código), com contornos PEC/PMC/periódico/Bloch/CPML, fontes TF/SF, monitores DFT e transformada campo próximo-distante.

<small>`pinneapple_simulation/electromagnetics/fdtd.py` · 912 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `YeeGrid` | classe | `(shape, spacing, courant, dt, allow_unstable)` | Grade de Yee: número de nós por eixo, espaçamentos e passo de tempo (fração do limite de Courant). |
| `Probe` | classe | `(component, index, values, times)` | Sonda que registra um componente de campo em um nó ao longo do tempo. |
| `DFTMonitor` | classe | `(components, freqs, region, data)` | Monitor que acumula a DFT de componentes de campo numa região durante a simulação. |
| `PlaneWaveTFSF` | classe | `(axis, polarization, waveform, box)` | Caixa campo total / campo espalhado que injeta uma onda plana com polarização e forma de onda dadas. |
| `FDTD` | classe | `(grid, boundaries=None, *, npml=10, pml_order=3.0, pml_kappa_max=1.0, pml_alpha_max=None,…)` | Solver leapfrog de Yee: materiais, fontes, correntes, sondas, monitores, ondas planas, avanço no tempo, energia e snapshots. |
| `NearToFarField` | classe | `(sim, freqs, box)` | Transformada campo próximo-distante em uma caixa fechada de planos de nós (grades 3-D). |
| `fill_fraction` | função | `(grid, inside, oversample=8) -> np.ndarray` | Fração de cada célula de nó que fica dentro de uma forma dada, por sobreamostragem. |
| `courant_limit` | função | `(spacing, dims=None) -> float` | Maior passo de tempo estável da grade de Yee nos eixos ativos. |
| `numerical_dispersion` | função | `(omega, dt, spacing, direction, n=1.0) -> float` | Número de onda numérico do esquema de Yee em uma direção e frequência (relação de dispersão resolvida por bisseção/Newton). |
| `gaussian_pulse` | função | `(t0, tau) -> Callable[[float], float]` | Forma de onda de pulso gaussiano. |
| `modulated_gaussian` | função | `(f0, bandwidth, t0=None) -> Callable[[float], float]` | Forma de onda de gaussiana modulada por seno (média zero, sem DC). |
| `ricker` | função | `(fp, t0=None) -> Callable[[float], float]` | Forma de onda de Ricker com frequência de pico dada. |
| `sinusoid` | função | `(f0, ramp_periods=3.0) -> Callable[[float], float]` | Seno ligado com rampa suave ao longo de alguns períodos. |
| `poynting_flux_dft` | função | `(mon, axis, spacing) -> np.ndarray` | Potência média através de um plano monitor calculada dos fasores DFT de E e H. |

#### `pinneapple_simulation.electromagnetics.fem`

Magnetostática 2-D no potencial vetor A_z com triângulos P1: montagem, restrições de Dirichlet e (anti)periódicas e solver de Newton com busca linear no funcional de energia, para ferro não linear e ímãs.

<small>`pinneapple_simulation/electromagnetics/fem.py` · 244 linhas · testes: 36</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Geometry` | classe | `(area, gx, gy, rows, cols, n)` | Áreas e gradientes baricêntricos por elemento de uma malha P1, com índices de montagem esparsa e operadores de rigidez, carga e rotacional. |
| `MagnetostaticSolution` | classe | `(A, B, nu, energy_functional, iterations, residual, geometry, tangent, P, info)` | Resultado magnetostático: A, B, relutividade, funcional de energia, iterações, resíduo e dados da tangente; `coenergy` dá a coenergia. |
| `flux_density` | função | `(mesh, A) -> np.ndarray` | Densidade de fluxo B por elemento a partir do A_z nodal. |
| `constraint_map` | função | `(n, dirichlet=None, dirichlet_values=None, periodic=None)` | Mapa esparso `A = P a + A_D` que aplica nós de Dirichlet e nós escravos periódicos/antiperiódicos. |
| `solve_magnetostatic` | função | `(mesh, nu, J=None, Br=None, nonlinear=(), dirichlet=None, dirichlet_values=None, periodic…)` | Resolve o problema magnetostático em A_z com correntes, remanência, materiais não lineares e restrições. |

#### `pinneapple_simulation.electromagnetics.hysteresis`

Magnetização não linear: curva de saturação de Fröhlich-Kennelly (compatível com o solver de Newton) e modelo de histerese de Jiles-Atherton.

<small>`pinneapple_simulation/electromagnetics/hysteresis.py` · 198 linhas · testes: 26</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FrohlichKennelly` | classe | `(alpha, beta)` | Curva B-H de Fröhlich-Kennelly em forma fechada, com a mesma interface de `BHCurve` (h, nu, dnu, energia). |
| `HysteresisLoop` | classe | `(H, M)` | Laço de histerese amostrado (H, M) com B, perda por ciclo, coercividade e remanência. |
| `JilesAtherton` | classe | `(Ms, a, k, c, alpha)` | Parâmetros do modelo de Jiles-Atherton, com curva anisterética, dM/dH, traçado de histórico de H e laço maior. |
| `langevin` | função | `(x)` | Função de Langevin `coth x - 1/x`, com série perto de zero. |

#### `pinneapple_simulation.electromagnetics.machine`

Máquina de fluxo radial com ímãs de superfície (ou rotor de relutância) em FEM 2-D A_z sobre a malha polar com banda móvel: torque, cogging, perdas e sensibilidades para otimização topológica.

<small>`pinneapple_simulation/electromagnetics/machine.py` · 473 linhas · testes: 47</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ScaledCurve` | classe | `(curve, w)` | Relutividade interpolada por densidade entre ar e ferro com pesos por elemento (para otimização topológica do ferro). |
| `MachineSolution` | classe | `(theta, currents, mesh, sol, J)` | Solução de uma posição do rotor: ângulo, correntes, malha, solução magnetostática e J. |
| `SPMMachine` | classe | `(pole_pairs, slots, r_rotor, magnet_thickness, magnet_arc, magnetization, Br, mu_r_magnet…)` | Máquina com ímãs de superfície ou rotor de relutância dentro de um estator ranhurado: malha, enrolamento, solve, torque e gradientes de projeto. |
| `belt_winding` | função | `(slots, pole_pairs, phases=3) -> list[tuple[int, int]]` | Enrolamento de camada única em faixas de fase de 60 graus: (fase, sinal) por ranhura. |

#### `pinneapple_simulation.electromagnetics.machines`

Máquinas elétricas: seções transversais paramétricas, malhas conformes, FEA 2-D não linear e pós-processamento; reexporta geometria, malhador, `IPMMachine` e dados de aço/ímã.

<small>`pinneapple_simulation/electromagnetics/machines/__init__.py` · 27 linhas · testes: 48</small>

#### `pinneapple_simulation.electromagnetics.machines.design`

Projeto do rotor da máquina IPM em V com o núcleo de `pinneapple_design.design_optimizer`: FEA como avaliador, GP com aprendizado ativo, NSGA-II, reverificação por FEA e comparação com um GA só com FEA (precisa de torch).

<small>`pinneapple_simulation/electromagnetics/machines/design.py` · 175 linhas · testes: 2</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `rotor_evaluator` | função | `(**kwargs)` | Cria um `IPMRotorEvaluator` (classe construída no primeiro uso, pois o núcleo de projeto importa torch). |
| `rotor_design_problem` | função | `(evaluator=None, ripple_max=0.3)` | Problema de projeto: maximizar torque, minimizar massa de ímã, com ripple abaixo de um limite. |
| `hybrid_vs_fea_only` | função | `(problem, *, n_initial=24, rounds=3, n_verify=8, ga_budget=400, surrogate_budget=1500, se…)` | Roda o estudo híbrido com surrogate e o estudo só com FEA no mesmo problema e os compara com qualidade igual. |

#### `pinneapple_simulation.electromagnetics.machines.drive`

Modelo de acionamento de máquina síncrona: regime dq, MTPA e enfraquecimento de campo com limites do inversor, mapa de eficiência e energia em ciclo de condução (WLTC).

<small>`pinneapple_simulation/electromagnetics/machines/drive.py` · 484 linhas · testes: 11</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LinearDQ` | classe | `(pole_pairs, psi_m, L_d, L_q, core_loss_coeffs)` | Máquina com indutâncias constantes, com MTPA e torque máximo em forma fechada para verificar o otimizador. |
| `FluxMap` | classe | `(pole_pairs, i_d, i_q, psi_d, psi_q, T, loss_i_d, loss_i_q, S_iron, magnet_per_f2, loss_c…)` | Máquina tabelada da FEA: fluxos dq, torque e coeficientes de perda numa grade (i_d, i_q), com leitura/gravação em npz. |
| `Drive` | classe | `(machine, R, I_max, V_max, n_d, L_end)` | Máquina alimentada por inversor com limites de corrente e tensão, resistência e indutância de cabeça de bobina; calcula pontos de operação e mapa de eficiência. |
| `Vehicle` | classe | `(mass, c_rr, cd_a, wheel_radius, gear_ratio, gear_efficiency, rotating_mass_factor, air_d…)` | Veículo longitudinal (massa, rolamento, arrasto, roda, relação de marcha) que converte velocidade em demanda do motor. |
| `v_max_svpwm` | função | `(v_dc) -> float` | Maior amplitude de tensão de fase do SVPWM na faixa linear. |
| `copper_resistance` | função | `(series_turns, conductor_area, stack_length, end_length, temperature=20.0, rho20=1.724e-0…)` | Resistência CC de fase a partir de espiras, área do condutor, comprimentos e temperatura. |
| `prius_phase_resistance` | função | `(temperature=100.0) -> tuple[float, dict]` | Resistência de fase do motor do Prius 2004 a partir dos dados de enrolamento do pyleecan (comprimento da cabeça de bobina estimado). |
| `wltc_class3b` | função | `() -> tuple[np.ndarray, np.ndarray]` | Ciclo WLTC classe 3b: tempo e velocidade do veículo. |
| `cycle_energy` | função | `(drive, vehicle, t, v, envelope_speeds=25) -> dict` | Energia do lado da bateria ao longo de um perfil de velocidade (motorização, regeneração e freio de atrito). |

#### `pinneapple_simulation.electromagnetics.machines.endwinding`

Cabeças de bobina fora da seção 2-D: caminhos 3-D, resistência e indutância de dispersão (fórmula de Neumann), no ar e com espelho de ferro.

<small>`pinneapple_simulation/electromagnetics/machines/endwinding.py` · 77 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `coil_end_path` | função | `(r_c, phi_a, phi_b, overhang, n_arc=48) -> np.ndarray` | Polilinha 3-D de uma cabeça de bobina entre dois ângulos. |
| `end_winding_paths` | função | `(layout, slots, coil_pitch, r_c, overhang) -> list[tuple[int, np.ndarray]]` | Caminhos (fase, polilinha) de todas as cabeças de bobina de um enrolamento de camada única. |
| `end_winding_inductance` | função | `(geometry, layout, turns_per_coil, coil_pitch, overhang, pieces_per_metre=300.0) -> dict` | Indutâncias própria e mútua de fase das cabeças de bobina e a dispersão dq, no ar e com espelho. |

#### `pinneapple_simulation.electromagnetics.machines.export`

Exporta malhas e campos de máquinas para VTK (`.vtu`, ParaView).

<small>`pinneapple_simulation/electromagnetics/machines/export.py` · 58 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `write_vtu` | função | `(path, points, cells, cell_data=None, point_data=None) -> Path` | Grava uma grade não estruturada VTK de triângulos ou prismas com dados de célula e de ponto. |
| `solution_to_vtu` | função | `(path, ms) -> Path` | Grava uma solução 2-D de máquina (regiões, B, \|B\|, A_z) em `.vtu`. |

#### `pinneapple_simulation.electromagnetics.machines.geometry`

Seções transversais paramétricas de máquinas de fluxo radial: rotor com ímãs internos em V (parametrização `HoleM50` do pyleecan), ranhuras de fundo redondo e o motor do Prius 2004.

<small>`pinneapple_simulation/electromagnetics/machines/geometry.py` · 320 linhas · testes: 18</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Polygon` | classe | `(name, xy, magnetization)` | Polígono fechado nomeado (com magnetização opcional). |
| `SectorGeometry` | classe | `(r_in, r_out, span, fill, polygons, arc_nodes_in, arc_nodes_out)` | Setor anular com polígonos sobre uma região de fundo, base do malhador. |
| `VRotor` | classe | `(poles, r_out, r_in, H0, H1, H2, H3, W0, W1, W2, W3, W4, magnet_segments)` | Rotor IPM com um V de dois ímãs por polo. |
| `RoundBottomSlots` | classe | `(slots, r_in, r_out, W0, H0, H1, W1, W2, H2, R1)` | Lâmina do estator com ranhuras abertas de fundo redondo. |
| `MachineGeometry` | classe | `(rotor, stator, length, stacking_factor)` | Rotor + estator + entreferro, com comprimento e fator de empilhamento. |
| `polygon_area` | função | `(xy) -> float` | Área com sinal de um polígono fechado. |
| `arc` | função | `(center, radius, a0, a1, h, even=False) -> np.ndarray` | Pontos de um arco circular com espaçamento máximo dado. |
| `prius_2004` | função | `() -> MachineGeometry` | Geometria do motor de tração do Toyota Prius 2004 com as dimensões do arquivo do pyleecan. |
| `geometry_errors` | função | `(sector, tol=1e-09) -> list[str]` | Motivos pelos quais um setor não é uma seção válida (lista vazia se válido). |

#### `pinneapple_simulation.electromagnetics.machines.getdp`

Verificação cruzada independente da FEA da máquina com gmsh e GetDP, chamados como executáveis externos (2-D e meia pilha 3-D para dispersão de extremidade); só funciona com `gmsh` e `getdp` no PATH.

<small>`pinneapple_simulation/electromagnetics/machines/getdp.py` · 469 linhas · testes: 3</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GmshGetDPError` | classe | `()` | Erro levantado quando o gmsh ou o GetDP falham. |
| `available` | função | `() -> bool` | Diz se `gmsh` e `getdp` estão no PATH. |
| `write_geo` | função | `(path, geometry, theta=0.0, n_gap=960, h_max=None, grading=0.2, h=0.001) -> dict` | Escreve a geometria gmsh da máquina inteira num ângulo do rotor. |
| `write_pro` | função | `(path, geometry, info, steel, Br, mu_r, layout, turns_per_slot, currents, parallel_paths=…)` | Escreve o problema GetDP (A_z magnetostático) para as regiões da geometria. |
| `solve` | função | `(geometry, theta, currents, steel, Br, mu_r, layout, turns_per_slot, n_gap=960, workdir=N…)` | Malha com gmsh e resolve com GetDP; devolve torque, fluxos concatenados, elementos e tempos. |
| `write_geo_3d` | função | `(path, geometry, n_gap, layers_stack, end_air, layers_air) -> dict` | Extruda a seção em prismas sobre meia pilha, com extensão de ar além da face. |
| `write_pro_3d` | função | `(path, info, steel, Br, mu_r, stacking_factor, pole_half_angle, r_gap, with_end_air) -> N…)` | Escreve o problema GetDP 3-D em potencial escalar magnético com ferro não linear por Picard. |
| `end_leakage_3d` | função | `(geometry, steel, Br, mu_r, n_gap=240, layers_stack=6, end_air=0.04, layers_air=6, workdi…)` | Razão de fluxo no entreferro com e sem a região de ar além da face (efeito de extremidade 3-D). |

#### `pinneapple_simulation.electromagnetics.machines.interop`

Interoperabilidade dos modelos de máquina: pyleecan, FEMM (script Lua), MATLAB, DXF, ElmerFEM e os contratos de dados do PINNeAPPle; FEMM e Elmer não são executados aqui.

<small>`pinneapple_simulation/electromagnetics/machines/interop.py` · 357 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `from_pyleecan` | função | `(machine) -> tuple[MachineGeometry, dict]` | Lê um arquivo `MachineIPMSM` do pyleecan em `MachineGeometry` e dados de enrolamento. |
| `to_pyleecan` | função | `(geometry) -> dict` | Escreve os parâmetros nos nomes de campo do pyleecan. |
| `cross_section_polygons` | função | `(geometry, h=0.001) -> list[tuple[str, np.ndarray, float \| None]]` | Todos os polígonos da seção completa com região e ângulo de magnetização. |
| `femm_lua` | função | `(geometry, steel, magnet, layout, turns_per_slot, currents=(0.0, 0.0, 0.0), magnet_temper…)` | Gera um script Lua do FEMM 4.2 para a máquina completa. |
| `read_femm_lua_polygons` | função | `(script) -> list[np.ndarray]` | Relê os polígonos traçados por um script de `femm_lua` (usado para checá-lo). |
| `to_mat` | função | `(path, **arrays) -> Path` | Grava arrays nomeados num arquivo `.mat` do MATLAB. |
| `from_mat` | função | `(path) -> dict` | Lê um arquivo `.mat` do MATLAB como dict. |
| `to_dxf` | função | `(path, geometry, h=0.001, unit_scale=1000.0) -> Path` | Grava a seção como polilinhas fechadas em DXF R12, uma camada por tipo de região. |
| `read_dxf_polylines` | função | `(path, unit_scale=1000.0) -> dict[str, list[np.ndarray]]` | Lê as polilinhas fechadas de um DXF R12 por camada. |
| `rotor_parameter_space` | função | `(name='v-ipm-rotor')` | `ParameterSpace` das variáveis de projeto do rotor. |
| `fluxmap_dataset` | função | `(fm, name='ipm-flux-map')` | Converte um mapa de fluxo em tabela pandas com seu `DatasetSpec`. |
| `elmer_case` | função | `(directory, geometry, steel, magnet, layout, turns_per_slot, currents=(0.0, 0.0, 0.0), th…)` | Escreve um caso ElmerFEM (geometria gmsh e `case.sif`) da máquina; não é rodado aqui. |

#### `pinneapple_simulation.electromagnetics.machines.mesher`

Malhas de Delaunay conformes de setores da máquina e sua montagem numa malha giratória com banda móvel no entreferro. [sem teste direto]

<small>`pinneapple_simulation/electromagnetics/machines/mesher.py` · 573 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SectorMesh` | classe | `(points, tris, region, polygon, arc_in, arc_out, side0, side1)` | Malha de um setor: pontos, triângulos, regiões e nós dos arcos e lados; `min_angle` dá a qualidade. |
| `MachineMesh` | classe | `(geometry, n_gap=960, h_max=None, grading=0.2, smooth=3, mirror=True)` | Malha da máquina completa, copiando setores de polo e de ranhura e unindo-os pela banda móvel. |
| `mesh_sector` | função | `(geom, size, n_in, n_out, smooth=3, mirror=False, max_rounds=60) -> SectorMesh` | Malha um setor com tamanho de elemento função do raio, respeitando todas as arestas dos polígonos. |
| `replicate_sector` | função | `(sm, copies, span) -> Mesh2D` | Monta o círculo completo a partir de cópias giradas de uma malha de setor. |

#### `pinneapple_simulation.electromagnetics.machines.pmsm`

Máquina síncrona de ímãs internos (IPM): FEA 2-D A_z não linear e pós-processamento (torque, cogging, fluxo, transformada de Park, L_d/L_q, MTPA, perdas). [sem teste direto]

<small>`pinneapple_simulation/electromagnetics/machines/pmsm.py` · 361 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Winding` | classe | `(turns_per_slot, parallel_paths, layout)` | Enrolamento de camada única: espiras por ranhura, caminhos paralelos e layout (fase, sinal). |
| `IPMMachine` | classe | `(geometry, steel, magnet, magnet_temperature, winding, n_gap, h_max, grading, tol)` | Modelo FEA da máquina IPM (geometria, aço, ímã, temperatura, enrolamento, malha) com solve, torque, fluxos e grandezas dq. |

#### `pinneapple_simulation.electromagnetics.machines.robustness`

Inclinação em degraus (multi-fatia 2-D), tolerâncias de fabricação e vista 3-D do campo de uma máquina IPM.

<small>`pinneapple_simulation/electromagnetics/machines/robustness.py` · 132 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `step_offsets` | função | `(steps, step_angle) -> np.ndarray` | Ângulos dos segmentos do rotor centrados em zero. |
| `step_skew_factor` | função | `(h, steps, step_angle, period)` | Fator do skew em degraus sobre um harmônico. |
| `skewed_no_load` | função | `(machine, thetas, steps, step_angle) -> dict` | Torque de cogging e fluxos concatenados do rotor com skew. |
| `skewed_torque` | função | `(machine, i_d, i_q, thetas, steps, step_angle) -> np.ndarray` | Torque do rotor com skew com correntes fixas no referencial dq. |
| `tolerance_model` | função | `(base, I_peak=250.0, gamma_deg=50.0, n_gap=288, n_positions=6, variables=TOLERANCE_VARIAB…)` | Função que leva amostras de tolerâncias a torque médio e ripple via FEA (NaN para geometrias inválidas). |
| `skewed_field_3d` | função | `(machine, theta, currents, steps, step_angle, layers_per_step=1) -> dict` | Malha de prismas da pilha com o campo 2-D de cada segmento. |

#### `pinneapple_simulation.electromagnetics.machines.steel`

Dados de catálogo de aço elétrico (M400-50A) e ímã (NdFeB do pyleecan) com fontes, curva de pilha e ajuste de perdas no ferro. [sem teste direto]

<small>`pinneapple_simulation/electromagnetics/machines/steel.py` · 191 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SteelGrade` | classe | `(name, H, B, loss_f, loss_B, loss_W_per_kg, thickness, resistivity, density, source)` | Grau de aço: curva B-H, tabela de perdas, espessura, resistividade, densidade e fonte. |
| `MagnetGrade` | classe | `(name, Br20, mu_r, alpha_Br, resistivity, density, source)` | Grau de ímã: Br a 20 °C, mu_r, coeficiente de temperatura, resistividade e densidade; dá Br(T) e Hc. |
| `LossFit` | classe | `(model, rms_rel_error, max_rel_error, fitted)` | Resultado do ajuste de perdas: modelo, erros relativos e parâmetros ajustados. |
| `TableIronLoss` | classe | `(grade)` | Perda específica interpolada da tabela de catálogo (mesma interface de `IronLossModel`). |
| `stacked` | função | `(curve, kf, n=200) -> BHCurve` | Curva B-H de uma pilha de lâminas com fator de empilhamento. |
| `fit_iron_loss` | função | `(grade, classical_from_resistivity=True, B_max=1.8, f_max=2500.0) -> LossFit` | Ajuste por mínimos quadrados do modelo de perdas à tabela do grau. |

#### `pinneapple_simulation.electromagnetics.machines.thermal`

Rede térmica concentrada de uma máquina de fluxo radial e checagem de desmagnetização dos ímãs; os dados térmicos padrão são valores típicos declarados como hipóteses.

<small>`pinneapple_simulation/electromagnetics/machines/thermal.py` · 283 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ThermalNetwork` | classe | `(names, C, links, boundaries)` | Rede genérica de nós isotérmicos com capacidades, condutâncias e contornos; regime permanente e transiente. |
| `MachineThermalData` | classe | `(k_lam, k_winding, k_cu, copper_fill, h_gap, h_end, jacket_GA, coolant_T, k_magnet_contac…)` | Dados de material e contorno da rede da máquina (valores típicos de livro, declarados como hipóteses). |
| `intrinsic_coercivity` | função | `(Hcj20, temperature, beta=-0.0055) -> float` | Coercividade intrínseca do ímã em função da temperatura. |
| `cylinder_conductance` | função | `(k, r1, r2, length, fraction=1.0) -> float` | Condutância de condução radial através de uma casca cilíndrica. |
| `machine_network` | função | `(geometry, data=None, end_length=0.16) -> ThermalNetwork` | Rede de sete nós de uma máquina com camisa de água. |
| `reverse_field` | função | `(B, angle, Br, mu_r) -> np.ndarray` | Campo oposto à magnetização em elementos de ímã lineares. |
| `demagnetisation_margin` | função | `(machine, ms, Hcj, margin_ratio=0.9) -> dict` | Pior campo reverso nos ímãs de uma solução e margem até Hcj. |
| `demagnetisation_refinement` | função | `(geometry, magnet_temperature, currents_d, Hcj, gradings=(0.2, 0.1, 0.05, 0.025), n_gap=9…)` | Estudo de malha da margem de desmagnetização com refino crescente. |
| `magnet_temperature_sweep` | função | `(geometry, temperatures, I_peak=250.0, n_gap=960, n_positions=12, n_flux=24, magnet=None)…)` | Desempenho da máquina em função da temperatura do ímã. |
| `iron_split` | função | `(m, I_peak=250.0, gamma_deg=50.3, rpm=1500.0, steps=24) -> dict` | Frações da perda no ferro na coroa, nos dentes e no rotor num ponto de operação. |

#### `pinneapple_simulation.electromagnetics.machines.verification`

Verificação da FEA de máquinas: formas fechadas, consistência interna de torque/fluxo/indutância e comparação com a curva torque-ângulo publicada do Prius 2004 (Yang et al. 2013).

<small>`pinneapple_simulation/electromagnetics/machines/verification.py` · 365 linhas · testes: 30</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `case` | função | `(name)` | Decorador que registra um caso de verificação de máquinas. |
| `run` | função | `(names=None, quick=False) -> list[CaseResult]` | Executa os casos de verificação de máquinas e devolve `CaseResult`. |
| `prius_torque_curve` | função | `(n_gap=960, I_peak=250.0, phi0_deg=YANG_2013_PHI0_DEG, n_positions=12) -> dict` | Torque médio (tensor de Maxwell e dq) do modelo do Prius em função do ângulo de corrente. |
| `yang_comparison` | função | `(curve) -> dict` | Erros ponto a ponto contra Yang et al. 2013 e o deslocamento de ângulo que minimiza o erro RMS. |

#### `pinneapple_simulation.electromagnetics.materials`

Materiais magnéticos para magnetostática 2-D: meios lineares, ímãs permanentes e curvas B-H não lineares (PCHIP e Brauer), sem histerese.

<small>`pinneapple_simulation/electromagnetics/materials.py` · 147 linhas · testes: 26</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LinearMaterial` | classe | `(mu_r, sigma)` | Meio linear isotrópico com permeabilidade relativa e condutividade. |
| `Magnet` | classe | `(Br, mu_r, direction, sigma)` | Ímã permanente linear `B = mu0 mu_r H + Br`. |
| `BHCurve` | classe | `(B, H)` | Curva B-H de valor único a partir de pontos, cúbica monótona em H(B), com inclinação do vácuo acima do último ponto. |
| `BrauerCurve` | classe | `(k1, k2, k3)` | Curva de Brauer `nu(s) = k1 exp(k2 s) + k3` com energia e derivada em forma fechada. |
| `illustrative_steel` | função | `(mu_r_init=4000.0, Js=2.0, n=60) -> BHCurve` | Curva de aço saturante suave gerada por fórmula (ilustrativa, não é dado de catálogo). |

#### `pinneapple_simulation.electromagnetics.media`

Meios elétricos e magnéticos para problemas de ondas: lineares, anisotrópicos e dispersivos (Drude, Lorentz, Debye), na convenção `exp(+j omega t)`.

<small>`pinneapple_simulation/electromagnetics/media.py` · 212 linhas · testes: 36</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Drude` | classe | `(omega_p, gamma)` | Polo de Drude (elétrons livres: metais e plasmas). |
| `Lorentz` | classe | `(delta_eps, omega0, delta)` | Polo de Lorentz (ressonância ligada). |
| `Debye` | classe | `(delta_eps, tau)` | Polo de Debye (relaxação). |
| `Medium` | classe | `(eps_inf, mu_r, sigma, sigma_m, poles)` | Meio linear isotrópico com eps_inf, mu_r, condutividades elétrica e magnética e polos dispersivos; dá eps_r, índice, k e impedância. |
| `Anisotropic` | classe | `(eps_r, mu_r, sigma)` | Meio linear anisotrópico com tensores eps_r e mu_r 3x3, rotação, reciprocidade e índices. |
| `drude_metal` | função | `(omega_p, gamma=0.0, eps_inf=1.0) -> Medium` | Atalho que cria um `Medium` de metal de Drude. |
| `is_passive` | função | `(medium, omegas, tol=1e-12) -> bool` | Testa passividade (`Im(eps_r) <= 0`) nas frequências dadas. |
| `rotation` | função | `(axis, angle) -> np.ndarray` | Matriz de rotação em torno de um eixo (fórmula de Rodrigues). |

#### `pinneapple_simulation.electromagnetics.mesh`

Malhas de triângulos para magnetostática 2-D: malhador polar por anéis com banda móvel para rotação, malha retangular estruturada e importador do gmsh (opcional).

<small>`pinneapple_simulation/electromagnetics/mesh.py` · 301 linhas · testes: 26</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Mesh2D` | classe | `(points, tris, region, region_names, layer, boundary, rings)` | Malha P1 de triângulos com um nome de região por elemento, máscaras, centroides e áreas. |
| `Ring` | classe | `(r, n, offset, rotating)` | Círculo de `n` nós igualmente espaçados num raio (com deslocamento angular e flag de rotação). |
| `zipper` | função | `(ia, ang_a, ib, ang_b, closed=True) -> np.ndarray` | Triangula a faixa entre dois anéis de nós quaisquer. |
| `polar_mesh` | função | `(rings, classify, theta=0.0, span=TWO_PI, region_names=None) -> Mesh2D` | Malha o disco (ou setor) entre anéis concêntricos, classificando regiões. |
| `rect_mesh` | função | `(x, y, classify=None) -> Mesh2D` | Malha triangular estruturada de um retângulo com diagonais alternadas. |
| `mesh_from_gmsh` | função | `(dim=2, model=None) -> Mesh2D` | Converte o modelo gmsh atual (já malhado) em `Mesh2D`. |
| `boundary_nodes` | função | `(tris) -> np.ndarray` | Nós que estão em arestas pertencentes a um só triângulo. |

#### `pinneapple_simulation.electromagnetics.mie`

Espalhamento de Mie de uma onda plana por uma esfera homogênea (e por uma esfera condutora perfeita), série de Bohren & Huffman.

<small>`pinneapple_simulation/electromagnetics/mie.py` · 127 linhas · testes: 26</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MieResult` | classe | `(x, m, an, bn, qext, qsca, qback, g)` | Resultado de Mie: parâmetros, coeficientes a_n/b_n e eficiências de extinção, espalhamento, retroespalhamento e g. |
| `mie_coefficients` | função | `(m_bh, x, n_max=None)` | Coeficientes a_n, b_n na convenção de Bohren & Huffman. |
| `mie` | função | `(m, x, n_max=None) -> MieResult` | Eficiências de uma esfera homogênea com índice na convenção do pacote. |
| `mie_pec` | função | `(x, n_max=None) -> MieResult` | Eficiências de uma esfera condutora perfeita. |
| `rayleigh_qsca` | função | `(m, x) -> float` | Limite de esfera pequena (Rayleigh) da eficiência de espalhamento. |

#### `pinneapple_simulation.electromagnetics.periodic`

Estruturas periódicas (arranjos infinitos, FSS) em execuções de célula unitária Floquet e as campanhas construídas sobre elas.

<small>`pinneapple_simulation/electromagnetics/periodic/__init__.py` · 15 linhas · testes: 5</small>

#### `pinneapple_simulation.electromagnetics.periodic.benchmark`

Campanhas de células unitárias e benchmark com orçamento igual de um surrogate de operador (FNO) contra otimização bayesiana no solver.

<small>`pinneapple_simulation/electromagnetics/periodic/benchmark.py` · 212 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `cell_simulator` | função | `(workdir, scans, *, phi=0.0, timeout=3600.0, log=None, resume=False) -> CLISimulator` | `CLISimulator` sobre `tcda` com todas as frequências e ângulos de varredura num processo por célula. |
| `campaign` | função | `(n, freqs, scans, workdir, *, ranges=None, fixed=None, method='lhs', seed=0, tag='c', log…)` | Gera e resolve `n` células de uma amostragem do espaço de parâmetros. |
| `to_arrays` | função | `(table, names, freqs, scans, column='S11_dB') -> tuple[np.ndarray, np.ndarray, list[str]]` | Converte a tabela em arrays X (parâmetros) e Y (frequência x varredura). |
| `worst_case` | função | `(Y) -> np.ndarray` | Pior reflexão ativa na banda e na varredura por célula. |
| `fit_fno` | função | `(X, Y, *, wall_s=60.0, seed=0, val_frac=0.15, width=24, threads=1)` | Treina um FNO (de `pinneapple_analysis.surrogate`) dos parâmetros ao mapa completo por tempo limitado. |
| `heldout` | função | `(predict, X, Y) -> dict[str, float]` | Erros do modelo em células nunca vistas. |
| `equal_budget` | função | `(solve, ranges, budget, *, n_verify=4, seed=0, fno_wall_s=60.0, n_candidates=4000, log=pr…)` | Compara as duas estratégias com o mesmo número de chamadas ao solver. |
| `solver_from` | função | `(sim, names, freqs, scans, fixed=None)` | Cria o callable `solve` de `equal_budget` a partir de um `cell_simulator`. |

#### `pinneapple_simulation.electromagnetics.periodic.tcda`

Célula unitária genérica de arranjo de dipolos fortemente acoplados (TCDA) para execuções Floquet no EMerge, escrita do zero, com CLI.

<small>`pinneapple_simulation/electromagnetics/periodic/tcda.py` · 165 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `InvalidCell` | classe | `()` | Exceção para parâmetros que não descrevem uma célula construível. |
| `TCDACell` | classe | `(a, h, w, g, s, t_s, d_s, er_s, z_port, res, air)` | Parâmetros da célula TCDA (lado, altura, dipolo, gap, superstrato, porta, malha) com checagem. |
| `simulate` | função | `(cell, freqs_ghz, scans_deg=((0.0, 0.0),), *, log='WARNING') -> list[dict]` | Reflexão ativa da célula em cada frequência e ângulo de varredura. |
| `main` | função | `(argv=None) -> int` | Ponto de entrada da CLI que simula uma célula e grava o resultado. |

#### `pinneapple_simulation.electromagnetics.postprocess`

Pós-processamento de soluções A_z: torque por tensor de Maxwell (Arkkio) e por coenergia, fluxo concatenado, FEM induzida, perdas no ferro e perdas por correntes parasitas nos ímãs.

<small>`pinneapple_simulation/electromagnetics/postprocess.py` · 285 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CoilSide` | classe | `(region, phase, turns, sign)` | Lado de bobina: região, fase, espiras e orientação. |
| `IronLossModel` | classe | `(k_h, alpha, beta, k_c, k_e, density)` | Modelo de perda no ferro por kg (histerese, clássica e excesso) com densidade. |
| `torque_maxwell_stress` | função | `(mesh, B, r1, r2, length=1.0, elements=None) -> float` | Torque de Arkkio sobre tudo dentro de uma coroa de ar. |
| `torque_maxwell_stress_gradient` | função | `(mesh, geometry, r1, r2, elements, length=1.0) -> np.ndarray` | O torque de Arkkio como forma quadrática do potencial nodal, com valor e gradiente (para adjuntos). |
| `torque_coenergy` | função | `(coenergy_at, theta, delta, length=1.0, stencil='central') -> float` | Torque por trabalho virtual (derivada da coenergia em relação ao ângulo). |
| `region_integral_A` | função | `(geometry, tris, A, mask) -> float` | Integral de A sobre os elementos de uma máscara. |
| `flux_linkage` | função | `(mesh, geometry, A, coil_sides, n_phases, length=1.0, parallel_paths=1) -> np.ndarray` | Fluxo concatenado por fase a partir dos lados de bobina. |
| `back_emf` | função | `(psi, theta, omega_mech) -> np.ndarray` | FEM induzida por derivada espectral do fluxo concatenado em um período. |
| `classical_eddy_coefficient` | função | `(sigma, thickness, density) -> float` | Coeficiente clássico de perdas parasitas de lâminas finas. |
| `element_flux_density_history` | função | `(B_steps, thetas, rotor_mask) -> np.ndarray` | Empilha B por passo, levando os elementos do rotor para o referencial do rotor. |
| `flux_density_ellipses` | função | `(B_hist, mask, n_harmonics=None)` | Semieixos da elipse descrita por cada harmônico de B por elemento. |
| `iron_loss` | função | `(B_hist, f_fundamental, model, area, mask, length=1.0, n_harmonics=None)` | Perda no ferro (total e densidade por elemento) a partir do histórico de B em um período. |
| `iron_loss_components` | função | `(B_hist, area, mask, alpha, beta, density, length=1.0, n_harmonics=None) -> np.ndarray` | Somas independentes da frequência que dão a perda no ferro em qualquer velocidade a partir de um período de solução. |
| `magnet_eddy_loss_resistance_limited` | função | `(mesh, geometry, A_hist, period, sigma, pieces, length=1.0)` | Perda média por correntes parasitas em peças de ímã, limitada por resistência. |

#### `pinneapple_simulation.electromagnetics.reflector`

Antenas refletoras e compact ranges por óptica física: geometria, correntes PO, campo próximo e métricas de zona quieta.

<small>`pinneapple_simulation/electromagnetics/reflector/__init__.py` · 18 linhas · testes: 6</small>

#### `pinneapple_simulation.electromagnetics.reflector.analytic`

Formas fechadas que verificam o código de óptica física: campo no eixo de um disco PEC e taper de abertura por óptica geométrica.

<small>`pinneapple_simulation/electromagnetics/reflector/analytic.py` · 55 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `disk_on_axis` | função | `(z, radius, k, E0=1.0) -> np.ndarray` | E_x exato no eixo de um disco PEC sob onda plana normal (corrente PO, integral exata). |
| `go_aperture_taper_db` | função | `(x, y, focal, q, feed_axis) -> np.ndarray` | Amplitude relativa por óptica geométrica na abertura de um paraboloide alimentado por um feed `cos^q`. |

#### `pinneapple_simulation.electromagnetics.reflector.compact_range`

Compact range de refletor offset único: feed, refletor, zona quieta, avaliação por PO e varreduras de parâmetros; os valores padrão são um exemplo genérico.

<small>`pinneapple_simulation/electromagnetics/reflector/compact_range.py` · 153 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CompactRange` | classe | `(freq_hz, reflector, feed_gain_dbi, pol, qz_z, qz_diameter, qz_centre_y, feed_offset, mes…)` | Projeto de compact range: frequência, refletor, ganho e polarização do feed, plano e diâmetro da zona quieta, parâmetros de malha. |
| `scaled` | função | `(cr, side=None, roll=None, feed_gain_dbi=None) -> CompactRange` | Cópia do range com outro tamanho de seção, raio de borda enrolada ou ganho de feed, mantendo a folga abaixo da seção. |
| `qz_field` | função | `(cr, points_xy=None) -> dict[str, Any]` | Campo PO no plano da zona quieta. |
| `evaluate` | função | `(cr, *, amp_tol=0.5, phase_tol=5.0) -> dict[str, Any]` | Métricas da zona quieta do range com tamanho de malha e custo. |
| `sweep` | função | `(base, gains, sides, rolls=(None,), *, log=None)` | Executa todas as combinações de ganho, tamanho e raio de borda e devolve uma `ResultTable`. |
| `go_check` | função | `(cr) -> dict[str, float]` | Compara a amplitude PO na zona quieta com o taper de óptica geométrica. |

#### `pinneapple_simulation.electromagnetics.reflector.geometry`

Refletores paraboloides offset com bordas enroladas opcionais, como malhas de triângulos para a óptica física.

<small>`pinneapple_simulation/electromagnetics/reflector/geometry.py` · 172 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OffsetReflector` | classe | `(focal, half_x, half_y, centre_y, n, roll, end_angle, blend, roll_aspect)` | Seção de paraboloide offset (contorno superelipse) com borda enrolada opcional; dá o foco e o centro. |
| `ReflectorMesh` | classe | `(vertices, faces, centres, normals, areas, roll)` | Malha do refletor: vértices, faces, centros, normais, áreas e marca de borda enrolada. |
| `superellipse_radius` | função | `(phi, a, b, n)` | Raio de uma superelipse em função do ângulo. |
| `mesh_reflector` | função | `(r, h) -> ReflectorMesh` | Malha triangular do refletor com arestas de tamanho aproximado `h`. |
| `flat_disk` | função | `(radius, h) -> ReflectorMesh` | Disco plano malhado, para validar a PO contra a forma fechada. |

#### `pinneapple_simulation.electromagnetics.reflector.po`

Óptica física sobre superfície PEC triangulada: campos de feed `cos^q` e de onda plana, correntes PO e campo irradiado exato com a função de Green diádica completa.

<small>`pinneapple_simulation/electromagnetics/reflector/po.py` · 118 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `wavenumber` | função | `(f_hz) -> float` | Número de onda no vácuo para uma frequência. |
| `q_from_gain` | função | `(gain_dbi) -> float` | Expoente `q` de um feed `cos^q` a partir da diretividade em dBi. |
| `gain_from_q` | função | `(q) -> float` | Diretividade em dBi de um feed `cos^q`. |
| `cosq_feed` | função | `(points, k, *, position, axis, q, pol=(0.0, 1.0, 0.0))` | Campos E e H de um feed `cos^q` em pontos dados. |
| `plane_wave` | função | `(points, k, direction, pol, amplitude=1.0)` | Campos E e H de uma onda plana. |
| `po_currents` | função | `(normals, centres, H_inc, source=None)` | Correntes PO `J = 2 n x H` nos triângulos iluminados. |
| `radiate` | função | `(centres, areas, J, obs, k, chunk=2000000)` | Campo E irradiado pelas correntes nos pontos de observação, perto ou longe. |

#### `pinneapple_simulation.electromagnetics.reflector.qz`

Métricas de zona quieta de um campo colimado: variação pico a pico, taper, ripple, ripple de fase e diâmetro utilizável.

<small>`pinneapple_simulation/electromagnetics/reflector/qz.py` · 69 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `disk_grid` | função | `(diameter, step, centre=(0.0, 0.0)) -> np.ndarray` | Pontos de uma grade quadrada dentro de um disco. |
| `qz_metrics` | função | `(xy, E_co, diameter, *, centre=(0.0, 0.0), E_x=None, amp_tol=0.5, phase_tol=5.0) -> dict[…)` | Métricas do campo copolar complexo sobre o disco da zona quieta. |

#### `pinneapple_simulation.electromagnetics.reflector.study`

Estudos de compact range: validação do código PO, varredura (ganho do feed x tamanho da seção), surrogate GP com erro fora da amostra e mapa de trade-off da qualidade da zona quieta.

<small>`pinneapple_simulation/electromagnetics/reflector/study.py` · 212 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `validate` | função | `(base, *, disk_radius_wl=3.0, mesh_densities=(8, 12, 16), rolls_wl=(3.0, 6.0), log=print)…)` | As três checagens de validação (disco PO, PO vs. GO, convergência de malha) na frequência do range. |
| `fit_surrogates` | função | `(table, metrics=METRICS, *, test_frac=0.25, seed=0)` | Ajusta um GP por métrica e raio de borda, com dois erros fora da amostra. |
| `tradeoff_map` | função | `(table, models, path, *, roll, qz_diameter, amp_tol=0.5, phase_tol=5.0, mark=None, metric…)` | Mapas previstos da métrica e do diâmetro utilizável sobre (ganho, tamanho), com a região viável. |
| `qz_figure` | função | `(cr, path, *, length_unit=1.0, unit_name='m', title='') -> dict[str, Any]` | Figura da amplitude e fase copolar na zona quieta, com cortes; devolve as métricas. |
| `qz_study` | função | `(base, gains, sides, rolls, out, *, mark=None, side_unit=1.0, side_label='section side [m…)` | Varredura, surrogates e um mapa de trade-off por raio de borda numa pasta de saída. |

#### `pinneapple_simulation.electromagnetics.reflector.study_report`

Estudo de compact range que se planeja sozinho: do problema às checagens e varreduras necessárias (por limiares de números adimensionais), à execução com cache em JSON e à folha de dados HTML/PDF.

<small>`pinneapple_simulation/electromagnetics/reflector/study_report.py` · 1.018 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Problem` | classe | `(freq_hz, focal, section, offset, roll, feed_gain_dbi, qz_diameter, qz_z, n, pols, band_h…)` | Projeto de compact range e o que ele deve entregar; aceita leitura de texto livre com unidades (`from_text`). |
| `PlannedStep` | classe | `(key, title, reason, required)` | Passo planejado do estudo: chave, título, motivo e se é obrigatório. |
| `from_text` | função | `(text, **defaults) -> Problem` | Atalho para `Problem.from_text`. |
| `plan` | função | `(problem) -> list[PlannedStep]` | Lista os passos que o problema exige e por quê. |
| `field_on_points` | função | `(cr, pts, *, mesh_J=None) -> np.ndarray` | Campo PO em pontos absolutos. |
| `qz_metrics_at` | função | `(cr, z=None, *, amp_tol=None, phase_tol=None) -> dict[str, float]` | Métricas de disco no plano da zona quieta numa posição z. |
| `cut_metrics_at` | função | `(cr, z=None, *, half=None, points=None, cuts=('horizontal', 'vertical')) -> dict[str, dic…)` | Métricas pico a pico e polarização cruzada em cortes pelo centro da zona quieta. |
| `main_roll` | função | `(p) -> float` | Raio de borda enrolada principal do problema. |
| `run` | função | `(p, out='cr_study', *, steps=None, parallel=0, log=print, reuse=True) -> dict[str, Any]` | Executa o plano (ou os passos nomeados) e guarda os números em `study.json`. |
| `report` | função | `(p, study, out='report', *, pdf_name=None) -> dict[str, str \| None]` | Renderiza o estudo como folha de dados técnica (HTML, e PDF quando há navegador). |

#### `pinneapple_simulation.electromagnetics.reflector.study_report_figures`

Figuras do relatório de compact range (geometria, iluminação, validação, zona quieta, varreduras), desenhadas a partir dos números guardados. [sem teste direto]

<small>`pinneapple_simulation/electromagnetics/reflector/study_report_figures.py` · 431 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_figures` | função | `(p, study, out) -> dict[str, str]` | Gera todas as figuras do relatório como `{chave: caminho png}`. |

#### `pinneapple_simulation.electromagnetics.reflector.study_report_text`

Texto da folha de dados do estudo de compact range: cada frase é calculada a partir dos números, com regras fixas e sem modelo de linguagem. [sem teste direto]

<small>`pinneapple_simulation/electromagnetics/reflector/study_report_text.py` · 578 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_datasheet` | função | `(p, study, figs) -> Datasheet` | Monta o `Datasheet` (de `pinneapple_tools.reporting.datasheet`) a partir do problema, do estudo e das figuras. |

#### `pinneapple_simulation.electromagnetics.statics`

Eletrostática e magnetostática em forma fechada (capacitores, fios, espiras, solenoides, indutâncias) e integração de Biot-Savart de correntes filamentares, vetorizadas.

<small>`pinneapple_simulation/electromagnetics/statics.py` · 293 linhas · testes: 50</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `parallel_plate_capacitance` | função | `(area, gap, eps_r=1.0) -> float` | Capacitância de placas paralelas sem efeito de borda. |
| `coax_capacitance` | função | `(a, b, eps_r=1.0) -> float` | Capacitância por metro de uma linha coaxial. |
| `coax_inductance` | função | `(a, b, mu_r=1.0) -> float` | Indutância externa por metro de uma linha coaxial. |
| `spherical_capacitance` | função | `(a, b=math.inf, eps_r=1.0) -> float` | Capacitância de esferas concêntricas (ou esfera isolada). |
| `point_charge_field` | função | `(points, q, at=(0.0, 0.0, 0.0), eps_r=1.0)` | Potencial e campo de uma carga pontual. |
| `line_charge_field` | função | `(points, lam, eps_r=1.0)` | Campo E de uma linha de carga infinita no eixo z. |
| `infinite_wire_B` | função | `(points, current)` | Campo B de um fio reto infinito no eixo z. |
| `polyline_segments` | função | `(vertices) -> tuple[np.ndarray, np.ndarray]` | Segmentos (início, fim) de uma polilinha aberta. |
| `circle_segments` | função | `(radius, n=360, center=(0.0, 0.0, 0.0), z=0.0)` | Círculo aproximado por segmentos retos. |
| `helix_segments` | função | `(radius, length, turns, n_per_turn=72, z0=None)` | Bobina helicoidal ao longo de z em segmentos retos. |
| `biot_savart` | função | `(points, starts, ends, current=1.0, chunk=4096)` | Campo B exato de segmentos filamentares finitos (Biot-Savart). |
| `vector_potential` | função | `(points, starts, ends, current=1.0, chunk=4096)` | Potencial vetor A exato de segmentos filamentares (gauge de Coulomb). |
| `flux_through_path` | função | `(path_starts, path_ends, starts, ends, current=1.0, gauss=4) -> float` | Fluxo magnético através de um caminho poligonal fechado pela circulação de A (Stokes). |
| `loop_on_axis_B` | função | `(z, radius, current)` | B axial de uma espira circular. |
| `loop_field` | função | `(points, radius, current)` | B de uma espira circular em qualquer ponto fora do fio, com integrais elípticas completas. |
| `finite_solenoid_on_axis_B` | função | `(z, radius, length, n_turns, current)` | B axial de um solenoide finito bem enrolado. |
| `helmholtz_coil_center_B` | função | `(radius, current, turns=1) -> float` | Campo no centro de um par de Helmholtz. |
| `mutual_inductance_coaxial_loops` | função | `(r1, r2, d) -> float` | Indutância mútua de duas espiras coaxiais (fórmula de Maxwell). |
| `loop_self_inductance` | função | `(R, a, internal=True) -> float` | Autoindutância de uma espira de fio fino. |
| `filament_inductance` | função | `(paths, wire_radius, pieces_per_metre=2000.0, gauss=6, mirror_z=None) -> np.ndarray` | Matriz de indutâncias de condutores finos ao longo de polilinhas (fórmula de Neumann). |

#### `pinneapple_simulation.electromagnetics.units`

Dimensões SI das grandezas eletromagnéticas e verificação dimensional das equações de Maxwell e de unidades declaradas pelo usuário.

<small>`pinneapple_simulation/electromagnetics/units.py` · 145 linhas · testes: 26</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `UnitProblem` | classe | `(where, message)` | Um problema de unidade encontrado: onde e mensagem. |
| `dimension` | função | `(unit) -> Dim` | Vetor de dimensão de uma expressão de unidade. |
| `term_dimension` | função | `(term, quantities=QUANTITIES) -> Dim` | Dimensão de um termo (produto de grandezas e operadores). |
| `check_equations` | função | `(equations=EQUATIONS, quantities=QUANTITIES) -> list[UnitProblem]` | Verifica que todos os termos de cada lei têm a mesma dimensão e devolve as violações. |
| `check_units` | função | `(declared) -> list[UnitProblem]` | Compara unidades declaradas de grandezas nomeadas com a tabela de referência. |

#### `pinneapple_simulation.electromagnetics.verification`

Suíte de verificação do eletromagnetismo: cada caso compara um solver com um resultado independente (forma fechada, valor publicado, outra biblioteca, limite) e mede erro e ordem de convergência; gera a tabela de `docs/guides/electromagnetics.md`.

<small>`pinneapple_simulation/electromagnetics/verification.py` · 893 linhas · testes: 12</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CaseResult` | classe | `(name, what, reference, error, tolerance, orders, expected_order, detail, seconds)` | Resultado de um caso de verificação: erro, tolerância, ordens observadas e esperada, detalhes e tempo; `passed` e `row`. |
| `case` | função | `(name)` | Decorador que registra uma função como caso de verificação. |
| `observed_orders` | função | `(errors, ratio=2.0) -> list[float]` | Ordens de convergência observadas entre erros sucessivos de um refinamento. |
| `run` | função | `(names=None, quick=False) -> list[CaseResult]` | Executa os casos registrados (todos ou os nomeados, opcionalmente em modo rápido). |
| `to_markdown` | função | `(results) -> str` | Renderiza os resultados como tabela Markdown. |
| `write_markdown_block` | função | `(path, results, marker='verification-table') -> None` | Substitui, num arquivo Markdown, o texto entre marcadores pela tabela dos resultados. |

#### `pinneapple_simulation.electromagnetics.waves`

Eletromagnetismo de ondas em forma fechada (convenção `exp(+j omega t)`): Fresnel, multicamadas, efeito pelicular, guias e cavidades, dipolos, espalhamento por cilindro e plásmons de superfície.

<small>`pinneapple_simulation/electromagnetics/waves.py` · 372 linhas · testes: 51</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MultilayerResult` | classe | `(r, t, R, T)` | Coeficientes de reflexão e transmissão (amplitude e potência) de uma pilha de camadas. |
| `free_space_wavelength` | função | `(f) -> float` | Comprimento de onda no vácuo para uma frequência. |
| `fresnel` | função | `(n1, n2, theta_i)` | Coeficientes de Fresnel (rs, rp, ts, tp) em uma interface plana. |
| `brewster_angle` | função | `(n1, n2) -> float` | Ângulo de Brewster. |
| `critical_angle` | função | `(n1, n2) -> float` | Ângulo crítico da reflexão interna total. |
| `multilayer` | função | `(n_layers, d_layers, wavelength, theta_i=0.0, pol='s', n_in=1.0, n_out=1.0) -> Multilayer…)` | Reflexão e transmissão de uma pilha de camadas planas pelo método das matrizes características. |
| `skin_depth` | função | `(omega, sigma, mu_r=1.0, eps_r=1.0, exact=False)` | Profundidade pelicular (forma de bom condutor ou exata). |
| `surface_impedance` | função | `(omega, sigma, mu_r=1.0)` | Impedância de superfície de um bom condutor. |
| `rect_waveguide_cutoff` | função | `(m, n, a, b, eps_r=1.0, mu_r=1.0) -> float` | Frequência de corte de um modo TE/TM de guia retangular. |
| `rect_waveguide_modes` | função | `(a, b, n_modes=10, eps_r=1.0, max_index=8)` | Modos mais baixos de um guia retangular ordenados por corte. |
| `rect_te_mn_fields` | função | `(x, y, m, n, a, b)` | Perfil transversal (não normalizado) do modo TE_mn de um guia retangular. |
| `circular_waveguide_cutoff` | função | `(kind, m, n, radius, eps_r=1.0) -> float` | Frequência de corte de um modo TE/TM de guia circular. |
| `circular_waveguide_modes` | função | `(radius, n_modes=10, eps_r=1.0, max_m=6, max_n=4)` | Modos mais baixos de um guia circular ordenados por corte. |
| `propagation_constant` | função | `(f, f_cutoff, eps_r=1.0, mu_r=1.0)` | Constante de propagação acima do corte (ou evanescente abaixo). |
| `rect_cavity_resonance` | função | `(m, n, p, a, b, d, eps_r=1.0, mu_r=1.0) -> float` | Ressonância de um modo de cavidade retangular PEC. |
| `rect_cavity_modes` | função | `(a, b, d, n_modes=10, eps_r=1.0, max_index=6)` | Ressonâncias mais baixas de uma cavidade retangular com multiplicidade. |
| `cylindrical_cavity_resonance` | função | `(kind, m, n, p, radius, length, eps_r=1.0) -> float` | Ressonância TE/TM de uma cavidade cilíndrica fechada. |
| `rect_cavity_te101_q` | função | `(a, b, d, sigma, eps_r=1.0) -> float` | Fator Q por perdas no condutor do modo TE101. |
| `hertzian_dipole_fields` | função | `(r, theta, current_moment, omega)` | Campos completos de um dipolo hertziano. |
| `hertzian_dipole_power` | função | `(current_moment, omega) -> float` | Potência irradiada média de um dipolo hertziano. |
| `hertzian_dipole_radiation_resistance` | função | `(length, wavelength) -> float` | Resistência de radiação de um dipolo curto. |
| `dipole_pattern` | função | `(theta, length_wavelengths=0.5)` | Padrão de campo distante de um dipolo fino alimentado no centro. |
| `cylinder_tm_coefficients` | função | `(ka, eps_r, n_max=None)` | Coeficientes do espalhamento TM_z por um cilindro dielétrico ou PEC. |
| `cylinder_scattering_tm` | função | `(x, y, k0, radius, eps_r, E0=1.0, total=True)` | Campo Ez fasorial do espalhamento de uma onda plana por um cilindro circular. |
| `half_wave_dipole_radiation_resistance` | função | `(eta=ETA0) -> float` | Resistência de radiação do dipolo de meia onda. |
| `half_wave_dipole_directivity` | função | `() -> float` | Diretividade do dipolo de meia onda. |
| `spp_wavenumber` | função | `(omega, eps_metal, eps_dielectric=1.0)` | Número de onda de um plásmon polariton de superfície numa interface plana. |
| `plasma_dispersion_omega` | função | `(k, omega_p)` | Relação de dispersão de ondas transversais num plasma sem colisões. |

#### `pinneapple_simulation.external_solvers`

Ponto de entrada das integrações externas: reexporta OpenFOAM e, com imports opcionais (`try/except`), os leitores CFD, MATLAB, Modelica, FEniCS, MuJoCo, Genesis e Newton, para o pacote importar mesmo sem as ferramentas instaladas.

<small>`pinneapple_simulation/external_solvers/__init__.py` · 226 linhas · testes: 16</small>

#### `pinneapple_simulation.external_solvers.ansys`

Pacote do acoplamento com ANSYS Fluent em modo batch (journal `.jou`); reexporta `journal_builder` e `runner`. [sem teste]

<small>`pinneapple_simulation/external_solvers/ansys/__init__.py` · 23 linhas · **sem teste direto**</small>

#### `pinneapple_simulation.external_solvers.ansys.journal_builder`

Gera o texto de um journal TUI do Fluent (ler caso/malha, condições de contorno simples, inicializar, iterar, gravar) a partir de `FluentCaseConfig`. O docstring separa comandos de alta e média confiança e lista o que não foi implementado; nunca foi rodado contra um Fluent real.

<small>`pinneapple_simulation/external_solvers/ansys/journal_builder.py` · 188 linhas · testes: 15 · palavras-chave no código (heurística; o texto acima diz o status real): placeholder/stub citado no docstring</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FluentCaseConfig` | classe | `(output_case_file, case_file, mesh_file, steady, iterations, time_steps, max_iterations_p…)` | Configuração de uma rodada batch do Fluent: caso ou malha de entrada, regime permanente/transiente, iterações e condições de velocity-inlet/pressure-outlet. |
| `build_journal` | função | `(config) -> str` | Gera o texto do journal TUI do Fluent para uma `FluentCaseConfig`, na ordem ler, contorno, inicializar, resolver, gravar e sair. |

#### `pinneapple_simulation.external_solvers.ansys.runner`

Lança o executável `fluent` real com o journal gerado, guarda logs e entrega o resultado ao leitor `fluent_mesh_to_upd`. Só os caminhos de geração de journal e de 'executável não encontrado' são testados (não há Fluent licenciado no ambiente).

<small>`pinneapple_simulation/external_solvers/ansys/runner.py` · 182 linhas · testes: 15</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FluentRunConfig` | classe | `(executable, precision, n_procs, work_dir, timeout_s, extra_args)` | Como invocar o executável do Fluent (nome, precisão, número de processos, pasta, timeout), sem dizer o que simular. |
| `FluentRunResult` | classe | `(success, upd_result, exit_code, wall_time_s, stdout_log_path, stderr_log_path, journal_p…)` | Resultado de uma rodada do Fluent: sucesso, `PhysicalSample` lido, código de saída, tempo, caminhos de log e journal, e erro de leitura. |
| `run_fluent_case` | função | `(case_config, run_config=None) -> FluentRunResult` | Monta o journal, lança o Fluent em batch e empacota o resultado com `fluent_mesh_to_upd`; falha explicitamente se o executável não existir. |

#### `pinneapple_simulation.external_solvers.calculix`

Ponte para o CalculiX (FEM estrutural implícito): decks `.inp`, execução do `ccx`, leitura de `.frd` e um `ccx` virtual.

<small>`pinneapple_simulation/external_solvers/calculix/__init__.py` · 23 linhas · testes: 14</small>

#### `pinneapple_simulation.external_solvers.calculix.frd`

Lê e escreve arquivos de resultado `.frd` do CalculiX (ASCII) e lê deslocamentos nodais de `.dat`.

<small>`pinneapple_simulation/external_solvers/calculix/frd.py` · 135 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FrdResults` | classe | `(node_ids, coords, steps)` | Resultados lidos de um `.frd`: ids e coordenadas dos nós e os passos com seus campos (acesso por `field`). |
| `read_frd` | função | `(path) -> FrdResults` | Lê um `.frd` ASCII do CalculiX em `FrdResults`. |
| `write_frd` | função | `(path, node_ids, coords, elements=None, element_type='C3D8', fields=None, step_time=1.0, …)` | Escreve um `.frd` ASCII com nós, elementos e blocos `DISP`/`STRESS`. |
| `read_dat_displacements` | função | `(path) -> Dict[int, np.ndarray]` | Lê deslocamentos nodais de um arquivo `.dat` (saída `*NODE PRINT`). |

#### `pinneapple_simulation.external_solvers.calculix.inp`

Contêiner simples de modelo CalculiX, escritor e leitor de decks `.inp`, e uma malha hexaédrica de viga em balanço. [sem teste direto]

<small>`pinneapple_simulation/external_solvers/calculix/inp.py` · 205 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CalculixModel` | classe | `(nodes, elements, element_type, nsets, material, E, nu, density, boundary, cloads, title)` | Modelo CalculiX mínimo: nós, elementos, conjuntos, material elástico, apoios e cargas pontuais. |
| `write_inp` | função | `(model, path, *, job_outputs=('U',), stress_outputs=('S',)) -> str` | Escreve um `CalculixModel` como deck `.inp` com pedidos de saída de deslocamento e tensão. |
| `read_inp` | função | `(path) -> CalculixModel` | Lê malha, material, apoios e cargas pontuais de um deck `.inp`, seguindo `*INCLUDE`. |
| `expand_target` | função | `(model, target) -> List[int]` | Devolve os ids de nós por trás de um alvo de `*BOUNDARY`/`*CLOAD` (id de nó ou nome de NSET). |
| `cantilever_hex_mesh` | função | `(L=1.0, b=0.1, h=0.1, nx=20, ny=2, nz=2, element_type='C3D20R') -> CalculixModel` | Cria uma viga em balanço de hexaedros ao longo de x, engastada em x = 0, com elementos de 8 ou 20 nós. |

#### `pinneapple_simulation.external_solvers.calculix.runner`

Executa `ccx` com o binário nativo se estiver no PATH, senão com uma imagem Docker construída localmente a partir de um `Dockerfile` (Debian `calculix-ccx`).

<small>`pinneapple_simulation/external_solvers/calculix/runner.py` · 100 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CcxRun` | classe | `(returncode, stdout, frd, dat)` | Resultado de uma execução do `ccx`: código de retorno, saída e caminhos do `.frd` e `.dat`. |
| `docker_available` | função | `() -> bool` | Diz se o Docker responde (`docker info`). |
| `ensure_image` | função | `(image=IMAGE, attempts=3) -> str` | Constrói a imagem Docker do CalculiX se faltar, tentando registries alternativos a cada falha. |
| `ccx_backend` | função | `() -> str \| None` | Diz qual backend está disponível: 'native', 'docker' ou None. |
| `run_ccx` | função | `(workdir, job, *, threads=1, timeout=3600.0, backend=None) -> CcxRun` | Roda `ccx -i <job>` numa pasta que contém `<job>.inp`, pelo backend escolhido. |

#### `pinneapple_simulation.external_solvers.calculix.study`

Estudos CalculiX em poucas linhas: malha (hexaedros, tetraedros gmsh ou deck existente), conjuntos por geometria, um passo de análise (estático, frequência, flambagem, calor), execução e resultados em arrays.

<small>`pinneapple_simulation/external_solvers/calculix/study.py` · 503 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Material` | classe | `(name, E, nu, density, conductivity, expansion, plastic)` | Material do estudo (E, nu, densidade, condutividade, expansão, plasticidade) e seu trecho de deck. |
| `FEModel` | classe | `(nodes, elements, element, material, title)` | Modelo de elementos finitos do estudo: malha (caixa, hexmesh, `.inp` ou gmsh), seleção de nós e faces por geometria, e o deck. |
| `Static` | classe | `(fix, loads, pressure, gravity, nlgeom, increments, prescribed)` | Passo estático linear ou não linear: apoios, forças, pressões, gravidade e deslocamentos prescritos. |
| `Frequency` | classe | `(n_modes, fix)` | Passo de frequências naturais e modos de vibração (requer densidade). |
| `Buckle` | classe | `(n_modes, fix, loads, pressure)` | Passo de flambagem linear: fatores de carga das cargas de referência e formas de flambagem. |
| `Heat` | classe | `(temperature, film, flux, initial)` | Passo de condução de calor permanente com temperaturas, convecção (film) e fluxos. |
| `Result` | classe | `(model, u, stress, strain, reactions, peeq, temperature, frequencies, buckling_factors, m…)` | Resultados de um estudo: deslocamentos, tensões, deformações, reações, temperaturas, frequências e modos, com von Mises e deslocamento máximo. |
| `solve` | função | `(model, step, workdir, *, threads=2, keep=True) -> Result` | Escreve `workdir/job.inp`, roda o `ccx` e lê os resultados num `Result`. |
| `geo_plate_with_hole` | função | `(L, W, t, d) -> str` | Gera a geometria (texto para gmsh) de uma placa com furo central. |
| `geo_l_bracket` | função | `(a=0.1, b=0.08, w=0.04, t=0.01, hole=0.012, fillet=0.01) -> str` | Gera a geometria (texto para gmsh) de uma cantoneira em L com filete e furo de parafuso. |

#### `pinneapple_simulation.external_solvers.calculix.virtual_ccx`

Um `ccx` substituto (`pinneapple-ccx`): mesma linha de comando, lê o `.inp` e escreve o `.frd` com campos previstos por um modelo (PINN/surrogate), recusando pontos fora do envelope de validade ou caindo no `ccx` real com `--fallback`.

<small>`pinneapple_simulation/external_solvers/calculix/virtual_ccx.py` · 117 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OutOfEnvelope` | classe | `()` | Erro levantado quando o caso pedido sai do envelope de validade do modelo preditor. |
| `check_envelope` | função | `(model, coords, envelope, tol=1e-06) -> None` | Confere caixa da geometria, módulo E e cargas contra o envelope declarado pelo preditor e levanta `OutOfEnvelope` se sair. |
| `predict_to_frd` | função | `(inp_path, predictor, frd_path=None) -> str` | Lê um `.inp`, prevê os campos nodais com o preditor e escreve o `.frd` correspondente. |
| `load_predictor` | função | `(spec) -> Callable` | Carrega um preditor a partir de uma string `pacote.modulo:atributo`. |
| `main` | função | `(argv=None) -> int` | Linha de comando `pinneapple-ccx -i job [--predictor ...] [--fallback]`, espelhando `ccx -i job`. |

#### `pinneapple_simulation.external_solvers.campaign_cases`

Escreve os arquivos de uma campanha de simulação planejada: tabelas de casos, um runner Python, casos OpenFOAM (aerofólio CST ou cópias de um caso base) e decks CalculiX, mais contratos e README; usado pelo app Campaign Planner.

<small>`pinneapple_simulation/external_solvers/campaign_cases.py` · 349 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `write_case_tables` | função | `(cases, out_dir, space=None, stem='cases', formats=('csv', 'parquet', 'json')) -> list[Pa…)` | Grava a tabela de casos em CSV, Parquet e/ou JSON, com o espaço de parâmetros ao lado. |
| `write_python_runner` | função | `(out_dir, parameters, outputs=()) -> Path` | Gera `run_cases.py`, que percorre os casos e chama a função `simulate` do usuário (o modelo gerado levanta `NotImplementedError` até ser preenchido). |
| `airfoil_space` | função | `(budget=40) -> ParameterSpace` | Espaço de parâmetros da campanha de aerofólio: pesos CST de espessura e arqueamento e ângulo de ataque. |
| `airfoil_feasible` | função | `(cases) -> np.ndarray` | Marca as formas de aerofólio válidas (superfícies que não se cruzam e permanecem suaves). |
| `airfoil_existing` | função | `()` | Devolve como DataFrame as rodadas OpenFOAM já existentes no dataset do zoo `airfoil_rans_cst_re4e6`. |
| `write_openfoam_airfoil` | função | `(cases, out_dir, reynolds=AIRFOIL_REYNOLDS, iterations=1200, limit=None) -> list[Path]` | Escreve um caso simpleFoam por linha (aerofólio CST) com `params.json`, `coefficients.py` e `Allrun`. |
| `write_openfoam_from_base` | função | `(base, cases, out_dir, edits, derived=None, solver='simpleFoam', procs=1, pre=('checkMesh…)` | Copia um caso OpenFOAM base aplicando edições de dicionário por caso e grava o `plan.json` do runner de campanhas. |
| `cantilever_space` | função | `(budget=30) -> ParameterSpace` | Espaço de parâmetros de uma viga em balanço com carga na ponta (dimensões, material, carga), esbelta para valer a teoria de vigas. |
| `write_calculix_cantilever` | função | `(cases, out_dir, elements_per_height=2) -> list[Path]` | Escreve um `job.inp` CalculiX por caso da viga em balanço, com a deflexão de Euler-Bernoulli em `params.json`. |
| `write_calculix_from_template` | função | `(template, cases, out_dir, name='job.inp') -> list[Path]` | Preenche os marcadores `{parametro}` de um deck CalculiX para cada caso. |
| `write_campaign` | função | `(plan, out_dir, *, outputs=(), units=None, openfoam=None, calculix=None, openfoam_base=No…)` | Grava um `CampaignPlan` inteiro: tabelas, runner, casos OpenFOAM/CalculiX que se aplicam, contratos e README. |

#### `pinneapple_simulation.external_solvers.cfd_formats`

Pacote de leitores de formatos abertos CFD/FEM além do OpenFOAM: CGNS, Exodus II, malha Fluent (só nós), ponte Abaqus `.inp`/`.odb` e exportações do STAR-CCM+. [sem teste direto do `__init__`]

<small>`pinneapple_simulation/external_solvers/cfd_formats/__init__.py` · 48 linhas · **sem teste direto**</small>

#### `pinneapple_simulation.external_solvers.cfd_formats._abaqus_odb_export_script`

Script que roda dentro do Python do próprio Abaqus (`abaqus python ...`) para exportar campos de um `.odb` para `.npz`; nunca é importado pelo código normal. [sem teste]

<small>`pinneapple_simulation/external_solvers/cfd_formats/_abaqus_odb_export_script.py` · 87 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `main` | função | `()` | Lê os argumentos JSON, abre o `.odb` com a API do Abaqus e grava os campos pedidos num `.npz`. |

#### `pinneapple_simulation.external_solvers.cfd_formats.abaqus_reader`

Lê nós e elementos de um `.inp` do Abaqus e faz a ponte para resultados `.odb` chamando o Abaqus instalado como subprocesso; empacota tudo como `PhysicalSample`.

<small>`pinneapple_simulation/external_solvers/cfd_formats/abaqus_reader.py` · 320 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `read_abaqus_inp_mesh` | função | `(path)` | Extrai os blocos `*NODE` e `*ELEMENT` de um `.inp` do Abaqus. |
| `abaqus_inp_to_upd` | função | `(path)` | Empacota a malha de um `.inp` (sem campos de resultado) como `PhysicalSample`. |
| `export_odb_fields` | função | `(odb_path, out_npz_path, *, step_name=None, frame_index=-1, field_outputs=('U', 'S'), aba…)` | Exporta campos de um `.odb` para `.npz` rodando o Python do Abaqus instalado. |
| `load_exported_odb_npz` | função | `(npz_path)` | Carrega o `.npz` exportado de um `.odb` como `PhysicalSample`. |

#### `pinneapple_simulation.external_solvers.cfd_formats.cgns_reader`

Lê coordenadas e campos de solução de arquivos CGNS/HDF5 e os empacota como `PhysicalSample` com domínio de malha.

<small>`pinneapple_simulation/external_solvers/cfd_formats/cgns_reader.py` · 186 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `read_cgns_mesh_and_fields` | função | `(path, *, fields=None, zone_index=0)` | Lê coordenadas de nós e campos de solução de um arquivo CGNS/HDF5. |
| `cgns_to_upd` | função | `(path, *, fields=None, zone_index=0)` | Lê um CGNS/HDF5 e empacota como `PhysicalSample` com `geometry.nodes`. |

#### `pinneapple_simulation.external_solvers.cfd_formats.convert`

Despacho por formato de arquivos CFD/FEM/CAD: nuvem de pontos portátil de qualquer formato suportado, superfície triangulada dos formatos com conectividade, e reparo/simplificação na conversão.

<small>`pinneapple_simulation/external_solvers/cfd_formats/convert.py` · 142 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `detect_format` | função | `(filename) -> str` | Devolve o nome do formato pelo sufixo do arquivo, ou `ValueError` listando os suportados. |
| `read_point_bundle` | função | `(path, fmt=None) -> dict[str, Any]` | Lê um formato volumétrico como dicionário de coordenadas, campos, proveniência e formato. |
| `tet_boundary_faces` | função | `(tets) -> np.ndarray` | Devolve os triângulos usados por um só tetraedro, ou seja, a fronteira de uma malha tetraédrica. |
| `read_surface_mesh` | função | `(path, fmt=None)` | Lê uma superfície triangulada (`MeshData`) de STL, OBJ, PLY, STEP ou `.inp` do Abaqus. |
| `convert_surface` | função | `(path, out_path, *, fmt=None, repair=False, simplify_target_faces=None) -> dict[str, Any]` | Lê uma superfície, opcionalmente repara e simplifica, e grava em STL, OBJ ou PLY. |

#### `pinneapple_simulation.external_solvers.cfd_formats.exodus_reader`

Lê coordenadas e campos nodais de Exodus II clássico (NetCDF-3) via `scipy.io.netcdf_file`; arquivos NetCDF-4/HDF5 levantam `NotImplementedError` com a explicação, e a conectividade de elementos não é lida.

<small>`pinneapple_simulation/external_solvers/cfd_formats/exodus_reader.py` · 177 linhas · testes: 5 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `read_exodus` | função | `(path, *, fields=None, time_index=-1)` | Lê coordenadas e campos nodais de um Exodus II em NetCDF clássico. |
| `exodus_to_upd` | função | `(path, *, fields=None, time_index=-1)` | Lê um Exodus II e empacota como `PhysicalSample` com domínio de malha. |

#### `pinneapple_simulation.external_solvers.cfd_formats.fluent_mesh_reader`

Lê só as coordenadas de nós (seção 10, ASCII) de malhas `.msh` do Fluent/Gambit; células, faces, zonas e seções binárias levantam `NotImplementedError` de propósito.

<small>`pinneapple_simulation/external_solvers/cfd_formats/fluent_mesh_reader.py` · 162 linhas · testes: 5 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `read_fluent_mesh_nodes` | função | `(path)` | Lê as coordenadas de nós das seções 10 de um `.msh` ASCII do Fluent/Gambit. |
| `fluent_mesh_to_upd` | função | `(path)` | Empacota os nós de um `.msh` do Fluent como `PhysicalSample` sem campos. |

#### `pinneapple_simulation.external_solvers.cfd_formats.starccm_reader`

Lê exportações do STAR-CCM+: CGNS (reaproveita `cgns_reader`) e tabelas texto/CSV (parser próprio). O formato nativo `.sim`/`.ccm` não é suportado, por ser binário proprietário.

<small>`pinneapple_simulation/external_solvers/cfd_formats/starccm_reader.py` · 243 linhas · testes: 9 · palavras-chave no código (heurística; o texto acima diz o status real): placeholder/stub citado no docstring</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `read_starccm_cgns` | função | `(path, *, fields=None, zone_index=0)` | Lê uma exportação CGNS do STAR-CCM+, delegando a `read_cgns_mesh_and_fields`. |
| `starccm_cgns_to_upd` | função | `(path, *, fields=None, zone_index=0)` | Empacota uma exportação CGNS do STAR-CCM+ como `PhysicalSample`, delegando a `cgns_to_upd`. |
| `read_starccm_table` | função | `(path, *, coord_names=('X', 'Y', 'Z'))` | Lê uma tabela texto/CSV exportada do STAR-CCM+, com cabeçalhos que podem ter unidades. |
| `starccm_table_to_upd` | função | `(path, *, coord_names=('X', 'Y', 'Z'))` | Empacota uma tabela exportada do STAR-CCM+ como `PhysicalSample` com as coordenadas como nós. |

#### `pinneapple_simulation.external_solvers.fenics`

Pacote da integração FEniCS; reexporta a camada de workflow de `fenics.solver`.

<small>`pinneapple_simulation/external_solvers/fenics/__init__.py` · 30 linhas · **sem teste direto**</small>

#### `pinneapple_simulation.external_solvers.fenics.solver`

Camada de workflow sobre `FEniCSBridge` (em `pinneapple_simulation.numerical_solvers.fenics_bridge`): configuração, solve com empacotamento em `PhysicalSample` e varreduras de parâmetros. Requer FEniCS instalado.

<small>`pinneapple_simulation/external_solvers/fenics/solver.py` · 180 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FEniCSConfig` | classe | `(pde, domain, bcs, params, solver_opts)` | Configuração de um solve FEniCS: identificador da EDP, domínio, condições de contorno, parâmetros e opções do solver. |
| `FEniCSWorkflow` | classe | `(base_config) -> None` | Atalho para solves FEniCS repetidos sobre uma configuração base, com `solve` e `sweep` em grade de parâmetros. |
| `solve_and_package` | função | `(config)` | Roda um solve via `FEniCSBridge` e devolve um `PhysicalSample`. |
| `dof_to_upd` | função | `(dof_vector, mesh_coords, field_name='u')` | Empacota um vetor de DOFs do FEniCS e as coordenadas da malha como `PhysicalSample`. |

#### `pinneapple_simulation.external_solvers.fluidx3d`

Ponte para o FluidX3D (LBM em GPU, OpenCL): escreve o `main_setup()` de um túnel de vento no checkout do usuário, compila e roda com `make.sh` e lê forças e campos `.vtk`. O PINNeAPPle não distribui código do FluidX3D, e o uso segue a licença dele (proíbe uso comercial e militar).

<small>`pinneapple_simulation/external_solvers/fluidx3d.py` · 225 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `WindTunnelSetup` | classe | `(length_m, velocity_m_s, nu_m2_s, rho_kg_m3, stl, box, memory_mb, body_cells, lbm_velocit…)` | Descrição de um túnel de vento no FluidX3D: corpo STL (ou esfera), velocidade, viscosidade, caixa, memória, tempo e exportações. |
| `fluidx3d_home` | função | `() -> str \| None` | Devolve a pasta do checkout do FluidX3D (`FLUIDX3D_HOME`) ou None. |
| `fluidx3d_available` | função | `() -> bool` | Diz se há um checkout configurado e um driver OpenCL instalado. |
| `write_setup` | função | `(setup, home=None) -> str` | Substitui o `main_setup()` de `src/setup.cpp` pelo gerado, guardando backup do original. |
| `enable_features` | função | `(features, home=None) -> str` | Ativa `#define`s em `src/defines.hpp` (com backup) e desliga `BENCHMARK`. |
| `run_fluidx3d` | função | `(setup=None, home=None, device=None, timeout=24 * 3600, log='log.pinneapple') -> subproce…)` | Escreve setup e defines, compila e roda com `make.sh`, gravando a saída num log. |
| `parse_forces` | função | `(text) -> np.ndarray` | Extrai as linhas (t, Fx, Fy, Fz) impressas pelo setup gerado. |
| `read_results` | função | `(home=None, log='log.pinneapple', export_dir=None) -> dict` | Lê o histórico de forças do log e o último campo exportado de cada grandeza. |
| `results_to_samples` | função | `(results, source='fluidx3d') -> list` | Converte os últimos campos exportados numa lista de `PhysicalSample` em grade regular. |

#### `pinneapple_simulation.external_solvers.genesis`

Pacote da integração com Genesis AI (simulação multifísica para robótica); reexporta configuração, construtor de cena, runner e adaptadores. [sem teste]

<small>`pinneapple_simulation/external_solvers/genesis/__init__.py` · 70 linhas · **sem teste direto**</small>

#### `pinneapple_simulation.external_solvers.genesis.adapter`

Converte dicionários de trajetória do Genesis em `PhysicalSample` (UPD). [sem teste]

<small>`pinneapple_simulation/external_solvers/genesis/adapter.py` · 96 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `genesis_traj_to_upd` | função | `(traj, time_array=None, field_map=None, meta_extra=None, dt=None)` | Converte um dicionário de trajetória do Genesis num `PhysicalSample`. |
| `genesis_trajs_to_upd` | função | `(trajs, time_array=None, field_map=None, meta_extra=None, dt=None) -> list` | Converte uma lista de trajetórias do Genesis em `PhysicalSample`s. |

#### `pinneapple_simulation.external_solvers.genesis.runner`

Executa o laço `scene.step()` de uma cena Genesis e coleta estados (corpo rígido, partículas MPM/SPH ou função própria) por passo. [sem teste]

<small>`pinneapple_simulation/external_solvers/genesis/runner.py` · 125 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GenesisRunner` | classe | `(config, state_fn=None, controller_fn=None) -> None` | Executa uma cena Genesis e coleta observações de trajetória por passo de tempo. |

#### `pinneapple_simulation.external_solvers.genesis.scene_builder`

Configuração de cena Genesis (backend, passo, ambientes paralelos, entidades, opções de corpo rígido/MPM/SPH) e a função que a materializa num `gs.Scene`. [sem teste]

<small>`pinneapple_simulation/external_solvers/genesis/scene_builder.py` · 180 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EntitySpec` | classe | `(morph, material, surface, vis_mode, name)` | Especificação de uma entidade da cena Genesis (morph, material, superfície, nome). |
| `GenesisConfig` | classe | `(backend, precision, dt, substeps, n_steps, n_envs, env_spacing, gravity, requires_grad, …)` | Configuração completa de uma cena Genesis: backend, precisão, passo, ambientes, gravidade, entidades e opções por física. |
| `build_scene` | função | `(config)` | Materializa uma `GenesisConfig` num `gs.Scene`. |

#### `pinneapple_simulation.external_solvers.geotaichi`

Ponte para o GeoTaichi (MPM, DEM, FEM, IGA; licença GPL-3.0), rodado como programa separado: executa um script e lê os quadros de partículas `.npz`, que também servem para o solver `geo_mpm` do próprio PINNeAPPle.

<small>`pinneapple_simulation/external_solvers/geotaichi.py` · 89 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `geotaichi_home` | função | `() -> str \| None` | Devolve a pasta do checkout GeoTaichi (`GEOTAICHI_HOME`) ou None. |
| `geotaichi_available` | função | `() -> bool` | Diz se há um checkout do GeoTaichi configurado e se o Taichi é importável pelo interpretador que vai rodá-lo. |
| `run_geotaichi` | função | `(script, workdir, args=(), python=None, timeout=24 * 3600, log='log.geotaichi') -> subpro…)` | Roda um script do GeoTaichi num processo próprio dentro de `workdir`, com log. |
| `read_geotaichi` | função | `(path, kind='MPM', allow_pickle=False) -> list[dict]` | Lê os quadros de partículas de uma rodada; entradas pickled são puladas, salvo `allow_pickle=True`. |
| `frames_to_samples` | função | `(frames, source='geotaichi', fields=None, times=None) -> list` | Converte cada quadro em um `PhysicalSample`: posições como nuvem de pontos e os demais arrays como campos. |

#### `pinneapple_simulation.external_solvers.matlab`

Pacote da integração MATLAB: Engine API, runner por subprocesso e E/S de `.mat`.

<small>`pinneapple_simulation/external_solvers/matlab/__init__.py` · 27 linhas · **sem teste direto**</small>

#### `pinneapple_simulation.external_solvers.matlab.engine`

Invólucro fino sobre `matlab.engine.MatlabEngine` (chamar funções, rodar scripts, avaliar, trocar variáveis). Requer a Engine API do MATLAB.

<small>`pinneapple_simulation/external_solvers/matlab/engine.py` · 88 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MATLABEngine` | classe | `(start_options=None) -> None` | Invólucro da Engine do MATLAB com `run_function`, `run_script`, `eval`, `get`, `put` e `quit`. |

#### `pinneapple_simulation.external_solvers.matlab.field_reader`

Converte a saída `.mat` de um solver MATLAB em `PhysicalSample`. [sem teste]

<small>`pinneapple_simulation/external_solvers/matlab/field_reader.py` · 61 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `mat_to_upd` | função | `(mat_path, *, field_keys=None, coord_keys=None, units=None)` | Carrega um `.mat` e empacota seus arrays como `PhysicalSample`. |

#### `pinneapple_simulation.external_solvers.matlab.mat_io`

Lê e grava arquivos `.mat` com `scipy.io`. [sem teste]

<small>`pinneapple_simulation/external_solvers/matlab/mat_io.py` · 34 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `load_mat` | função | `(path) -> Dict[str, np.ndarray]` | Carrega um `.mat` como dicionário de arrays numpy. |
| `save_mat` | função | `(path, arrays) -> None` | Grava um dicionário de arrays (ou escalares) num `.mat`. |

#### `pinneapple_simulation.external_solvers.matlab.runner`

Roda scripts e funções MATLAB por subprocesso em modo `-batch`, trocando dados por `.mat`. [sem teste]

<small>`pinneapple_simulation/external_solvers/matlab/runner.py` · 89 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `run_matlab_script` | função | `(script_path, *, matlab_executable='matlab', timeout=300, extra_flags=None) -> subprocess…)` | Roda um script MATLAB por subprocesso em modo `-batch`. |
| `run_matlab_function` | função | `(func_name, args, out_mat_path, *, matlab_executable='matlab', timeout=300) -> None` | Chama uma função MATLAB por um script-invólucro gerado e grava a saída num `.mat`. |

#### `pinneapple_simulation.external_solvers.modelica`

Pacote da integração Modelica: simulação de FMUs (fmpy), OpenModelica (OMPython ou `omc` direto), inspeção de FMUs, compatibilidade FMI e varreduras.

<small>`pinneapple_simulation/external_solvers/modelica/__init__.py` · 35 linhas · testes: 5</small>

#### `pinneapple_simulation.external_solvers.modelica.fmi_compat`

Roda FMUs por `simulate_fmu` em cada interface que oferecem (Co-Simulation, Model Exchange) e monta a tabela de compatibilidade pedida pela lista de ferramentas FMI.

<small>`pinneapple_simulation/external_solvers/modelica/fmi_compat.py` · 122 linhas · testes: 2</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CompatRow` | classe | `(fmu, fmi_version, interface, exporter, platform, passed, detail)` | Uma linha da tabela de compatibilidade FMI: FMU, versão, interface, exportador, plataforma, se passou e detalhe. |
| `platform_tag` | função | `() -> str` | Devolve a plataforma binária FMI desta máquina (por exemplo `x86_64-linux`). |
| `check_fmu` | função | `(fmu, fmi_type, exporter='', stop_time=None, tol=None) -> CompatRow` | Simula uma FMU numa interface e informa se funcionou. |
| `check_fmus` | função | `(fmus, exporter='', interfaces=…) -> list[CompatRow]` | Aplica `check_fmu` a cada FMU e a cada interface que ela implementa. |
| `markdown_table` | função | `(rows) -> str` | Monta tabelas Markdown por ferramenta exportadora e versão FMI, com células 'CS, ME'. |

#### `pinneapple_simulation.external_solvers.modelica.fmi_runner`

Simula uma FMU com fmpy e devolve as saídas como arrays ou como `PhysicalSample`.

<small>`pinneapple_simulation/external_solvers/modelica/fmi_runner.py` · 107 linhas · testes: 2</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FMUSimConfig` | classe | `(start_time, stop_time, step_size, output_interval, parameters, output_variables, fmi_typ…)` | Configuração de uma simulação de FMU: tempos, passo, intervalo de saída, parâmetros, variáveis e tipo FMI. |
| `simulate_fmu` | função | `(fmu_path, config) -> Dict[str, np.ndarray]` | Simula uma FMU com fmpy e devolve as variáveis de saída como arrays. |
| `fmu_to_upd` | função | `(fmu_path, config)` | Simula uma FMU e empacota o resultado como `PhysicalSample`. |

#### `pinneapple_simulation.external_solvers.modelica.fmu_inspect`

Inspeção estática de um `.fmu` antes de usá-lo: versão FMI, interfaces, binários por plataforma, símbolos exportados de ELF64 contra os exigidos pelo padrão e consistência do modelDescription.

<small>`pinneapple_simulation/external_solvers/modelica/fmu_inspect.py` · 233 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BinaryCheck` | classe | `(platform, path, format, exported, missing)` | Resultado da checagem de um binário da FMU: plataforma, caminho, formato, funções exportadas e faltantes. |
| `FMUReport` | classe | `(path, fmi_version, model_name, interfaces, binaries, problems, n_variables)` | Relatório de inspeção de uma FMU: versão, modelo, interfaces, binários, problemas e número de variáveis. |
| `elf_exported_functions` | função | `(data) -> List[str]` | Lista as funções globais exportadas na tabela `.dynsym` de um objeto ELF64 little-endian. |
| `inspect_fmu` | função | `(path) -> FMUReport` | Inspeciona estaticamente um `.fmu` sem carregar nem executar código. |

#### `pinneapple_simulation.external_solvers.modelica.om_runner`

Carrega e simula um modelo Modelica com OpenModelica via OMPython e empacota o resultado como `PhysicalSample`. [sem teste]

<small>`pinneapple_simulation/external_solvers/modelica/om_runner.py` · 211 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OMSimConfig` | classe | `(model_name, start_time, stop_time, number_of_intervals, tolerance, parameters, output_va…)` | Configuração de uma simulação OpenModelica: modelo, tempos, intervalos, tolerância, parâmetros e saídas. |
| `simulate_openmodelica` | função | `(model_file, config, *, working_dir=None) -> Dict[str, np.ndarray]` | Carrega e simula um modelo Modelica com OpenModelica e devolve arrays por variável. |
| `om_to_upd` | função | `(model_file, config, **sim_kwargs)` | Simula com OpenModelica e devolve um `PhysicalSample`. |

#### `pinneapple_simulation.external_solvers.modelica.omc`

Usa o compilador `omc` sem OMPython, local ou pela imagem Docker oficial, para gerar FMUs e simular arquivos `.mo`.

<small>`pinneapple_simulation/external_solvers/modelica/omc.py` · 140 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `omc_available` | função | `() -> str \| None` | Diz se o `omc` está disponível: 'local', 'docker' ou None. |
| `run_omc_script` | função | `(script, workdir, timeout=1800.0) -> str` | Roda um script `.mos` do `omc` numa pasta e devolve a saída, com `RuntimeError` em caso de erro. |
| `build_fmu` | função | `(models, model_name, out_dir, fmi_version='2.0', fmu_type='cs', libraries=()) -> Path` | Compila um modelo de arquivos `.mo` numa FMU. |
| `simulate_mo` | função | `(models, model_name, stop_time=1.0, start_time=0.0, number_of_intervals=500, tolerance=1e…)` | Simula um modelo com `omc` e devolve `{variavel: array}` incluindo `time`. |

#### `pinneapple_simulation.external_solvers.modelica.result_reader`

Converte dicionários de resultado Modelica/FMI em `PhysicalSample`. [sem teste]

<small>`pinneapple_simulation/external_solvers/modelica/result_reader.py` · 53 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `modelica_result_to_upd` | função | `(result, *, time_key='time', field_keys=None, units=None, source='modelica')` | Empacota um dicionário de resultado Modelica como `PhysicalSample`. |

#### `pinneapple_simulation.external_solvers.modelica.sweep`

Varreduras de parâmetros de uma FMU por hipercubo latino, gerando dados de treino e teste para surrogates de um modelo Modelica.

<small>`pinneapple_simulation/external_solvers/modelica/sweep.py` · 86 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FMUDataset` | classe | `(names, params, time, outputs, bounds)` | Dataset de uma varredura de FMU: nomes, parâmetros, tempo, saídas e limites, com `split`, `save` e `load`. |
| `latin_hypercube` | função | `(n, d, seed=0) -> np.ndarray` | Gera n pontos em [0, 1]^d por hipercubo latino. |
| `fmu_sweep` | função | `(fmu, ranges, n, stop_time, output_interval, outputs, seed=0, fixed=None, start_time=0.0,…)` | Simula n amostras de hipercubo latino dos intervalos de parâmetros numa grade de saída comum. |

#### `pinneapple_simulation.external_solvers.mujoco`

Pacote da integração MuJoCo: carregamento de modelo, runner de trajetórias, rollouts paralelos MJX e adaptadores UPD. [sem teste]

<small>`pinneapple_simulation/external_solvers/mujoco/__init__.py` · 53 linhas · **sem teste direto**</small>

#### `pinneapple_simulation.external_solvers.mujoco.adapter`

Converte dicionários de trajetória MuJoCo em `PhysicalSample`. [sem teste]

<small>`pinneapple_simulation/external_solvers/mujoco/adapter.py` · 78 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `trajectory_to_upd` | função | `(traj, field_map=None, meta_extra=None)` | Converte um dicionário de trajetória MuJoCo num `PhysicalSample`. |
| `trajectories_to_upd` | função | `(trajs, field_map=None, meta_extra=None) -> list` | Converte uma lista de trajetórias MuJoCo em `PhysicalSample`s. |

#### `pinneapple_simulation.external_solvers.mujoco.loader`

Configuração MuJoCo e funções para carregar o `MjModel` (arquivo ou XML) e alocar o `MjData`. [sem teste]

<small>`pinneapple_simulation/external_solvers/mujoco/loader.py` · 92 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MuJoCoConfig` | classe | `(model_path, model_xml, assets, dt, n_steps, n_substeps, gravity, extra_opts)` | Configuração de uma simulação MuJoCo: modelo (arquivo ou XML), passo, número de passos, gravidade e opções. |
| `load_model` | função | `(config)` | Carrega o `MjModel` a partir da configuração. |
| `make_data` | função | `(model)` | Aloca um `MjData` novo para o modelo. |

#### `pinneapple_simulation.external_solvers.mujoco.mjx_backend`

Backend MJX (JAX/GPU) que roda muitos ambientes MuJoCo em paralelo com `jax.vmap` para gerar datasets. Requer `mujoco` e `jax`. [sem teste]

<small>`pinneapple_simulation/external_solvers/mujoco/mjx_backend.py` · 118 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MJXParallelRunner` | classe | `(config, n_envs=256, device=None) -> None` | Roda N ambientes MuJoCo independentes em paralelo com MJX e `jax.vmap`, com saída em arrays ou UPD. |

#### `pinneapple_simulation.external_solvers.mujoco.runner`

Roda um modelo MuJoCo com controlador opcional e coleta a trajetória (qpos, qvel etc.). [sem teste]

<small>`pinneapple_simulation/external_solvers/mujoco/runner.py` · 137 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MuJoCoRunner` | classe | `(config, controller=None) -> None` | Roda um modelo MuJoCo com controlador opcional e coleta trajetórias. |

#### `pinneapple_simulation.external_solvers.newton_physics`

Pacote do backend de corpos rígidos NVIDIA Newton (XPBD, semi-implícito, Featherstone, MuJoCo-Warp) com checagens analíticas.

<small>`pinneapple_simulation/external_solvers/newton_physics/__init__.py` · 28 linhas · testes: 8</small>

#### `pinneapple_simulation.external_solvers.newton_physics.runner`

Integra uma `RigidScene` com um solver do Newton e grava a trajetória; velocidades iniciais com solvers de coordenadas de junta levantam `NotImplementedError`. [sem teste direto]

<small>`pinneapple_simulation/external_solvers/newton_physics/runner.py` · 182 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RigidTrajectory` | classe | `(names, t, pos, quat, vel, omega, mass, inertia, com, gravity, solver, dt, wall_time_s, m…)` | Estados gravados de uma simulação de corpos rígidos (posições, quatérnios, velocidades, massas, inércias) com busca de corpo por nome. |
| `simulate` | função | `(scene, duration, dt=0.001, solver='xpbd', iterations=8, record_every=10, device='cpu', l…)` | Integra a cena por um tempo com o solver escolhido e grava a trajetória a cada `record_every` passos. |

#### `pinneapple_simulation.external_solvers.newton_physics.scene`

Descrição de cena de corpos rígidos (corpos, formas, juntas, chão) em Python puro, que só importa Newton/Warp ao chamar `build`.

<small>`pinneapple_simulation/external_solvers/newton_physics/scene.py` · 280 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Shape` | classe | `(kind, size, vertices, faces, boxes)` | Uma forma de colisão de um corpo no referencial do corpo (esfera, caixa, cápsula, cilindro, malha ou composta). |
| `Body` | classe | `(name, shape, pos, quat, vel, omega, density, mass, mu, restitution)` | Um corpo rígido da cena: forma, pose, velocidades iniciais, densidade ou massa, atrito e restituição. |
| `Joint` | classe | `(kind, parent, child, anchor, axis, damping)` | Uma junta entre dois corpos (revoluta ou fixa) com âncora, eixo e amortecimento. |
| `RigidScene` | classe | `(gravity, bodies, joints, ground_mu, ground_restitution)` | Cena de corpos, juntas e chão opcional, montada num modelo Newton só quando pedida. |

#### `pinneapple_simulation.external_solvers.newton_physics.validation`

Checa um solver do Newton contra resultados fechados: queda livre, atrito, pêndulo, quique, giro livre e contato em repouso. [sem teste direto]

<small>`pinneapple_simulation/external_solvers/newton_physics/validation.py` · 109 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `free_fall` | função | `(solver='xpbd', dt=0.001, **kw) -> dict` | Checa queda livre de uma esfera contra z(t) = z0 - g t²/2. |
| `sliding_friction` | função | `(solver='xpbd', dt=0.001, mu=0.4, v0=2.0, **kw) -> dict` | Checa a distância de parada de uma caixa com atrito de Coulomb contra v0²/(2 mu g). |
| `pendulum_period` | função | `(solver='xpbd', dt=0.001, length=1.0, amp_deg=5.0, **kw) -> dict` | Checa o período de um pêndulo em pequena amplitude contra a fórmula fechada. |
| `bounce` | função | `(solver='xpbd', dt=0.0001, e=0.5, **kw) -> dict` | Checa a altura do primeiro quique contra e² vezes a altura de queda. |
| `torque_free_spin` | função | `(solver='xpbd', dt=0.001, **kw) -> dict` | Checa a conservação do momento angular de uma caixa girando sem gravidade. |
| `resting_contact` | função | `(solver='xpbd', dt=0.001, **kw) -> dict` | Checa que uma caixa em repouso no chão não afunda nem escorrega. |
| `validate` | função | `(solver='xpbd', cases=None, log=print, **kw) -> list[dict]` | Roda as checagens analíticas para um solver e devolve e registra uma tabela. |

#### `pinneapple_simulation.external_solvers.openfoam`

Pacote OpenFOAM: montagem de casos, execução, amostragem, leitura binária, empacotamento UPD, arquivos `.splash` e campanhas. [sem teste direto do `__init__`]

<small>`pinneapple_simulation/external_solvers/openfoam/__init__.py` · 8 linhas · **sem teste direto**</small>

#### `pinneapple_simulation.external_solvers.openfoam.binary_reader`

Primitivas para ler arquivos FoamFile em formato binário (cabeçalho, listas, pontos, faces, internalField); formato ASCII ou tipos de lista não suportados levantam `NotImplementedError`. [sem teste direto]

<small>`pinneapple_simulation/external_solvers/openfoam/binary_reader.py` · 254 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FoamFileHeader` | classe | `(klass, fmt, body_start)` | Cabeçalho de um FoamFile: classe, formato e posição onde começa o corpo. |
| `read_header` | função | `(data) -> FoamFileHeader` | Localiza o fim do bloco `FoamFile {...}` e lê suas chaves. |
| `read_binary_block` | função | `(data, pos, n_components, dtype=_SCALAR_DTYPE) -> Tuple[np.ndarray, int]` | Lê um bloco binário `<count>(<bytes>)` a partir de uma posição. |
| `read_label_list` | função | `(data, path_hint='') -> np.ndarray` | Lê uma lista binária de rótulos (inteiros) de um FoamFile. |
| `read_vector_field_points` | função | `(data, path_hint='') -> np.ndarray` | Lê uma lista binária de pontos/vetores de um FoamFile. |
| `read_face_compact_list` | função | `(data, path_hint='') -> Tuple[np.ndarray, np.ndarray]` | Lê uma `faceCompactList` (offsets e índices de pontos). |
| `read_internal_field` | função | `(data, path_hint='') -> Tuple[np.ndarray, bool, Optional[int]]` | Lê o `internalField` de um campo volumétrico, uniforme ou não. |

#### `pinneapple_simulation.external_solvers.openfoam.campaign`

Campanhas de parâmetros de rodadas OpenFOAM divididas entre várias máquinas: copia o caso base, aplica edições, roda com MPI, amostra linhas e grava um JSON por rodada, com trava por arquivo e publicação opcional por git.

<small>`pinneapple_simulation/external_solvers/openfoam/campaign.py` · 686 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CampaignRun` | classe | `(id, params, mesh, worker, group)` | Uma rodada da campanha: id, parâmetros, malha, worker e grupo. |
| `Campaign` | classe | `(base, runs, name, solver, procs, end_time, pre, derived, edits, area_patch, setup, sampl…)` | Plano de campanha (caso base, rodadas, solver, edições, amostragem, resultados) com leitura/escrita JSON, preparação e execução por worker. |
| `GitPublisher` | classe | `(name=None, email=None, remote='origin', tries=6)` | Faz commit e push de um arquivo de resultado no branch atual, com rebase sobre o que outros workers enviaram. |
| `evaluate` | função | `(expr, names) -> float \| list` | Avalia uma expressão aritmética restrita sobre nomes e funções de `math`. |
| `format_template` | função | `(template, values) -> str` | Formata um template restrito a `{nome}` e `{nome:spec}` numéricos, sem acesso a atributos. |
| `resolve` | função | `(params, derived=None) -> dict` | Devolve os parâmetros mais os valores derivados, na ordem dada. |
| `get_entry` | função | `(text, key) -> str` | Lê o valor de uma chave `a/b/c` num texto de dicionário OpenFOAM. |
| `set_entry` | função | `(text, key, value) -> str` | Define uma chave `a/b/c` num texto de dicionário OpenFOAM, criando níveis faltantes. |
| `edit_file` | função | `(path, key, value, root=None) -> None` | Define uma chave num arquivo de dicionário OpenFOAM, exigindo que o arquivo esteja dentro de `root` se dado. |
| `reset_nonuniform` | função | `(case, n_cells) -> list[str]` | Após troca de malha, reinicia campos de `0/` com número errado de valores para o valor mais comum. |
| `n_cells` | função | `(case) -> int` | Devolve o número de células da malha de um caso. |
| `area_patch` | função | `(case, patch, area, centre, normal, from_patches=None, rest='wall', plane_tol=0.03, bashr…)` | Reconstrói um patch de contorno com as faces de um plano mais próximas do centro até atingir a área pedida. |
| `write_decompose` | função | `(case, n, method='auto', bashrc=None) -> str` | Escreve o `decomposeParDict` para n processos (scotch se houver, senão `simple`). |
| `parse_solver_log` | função | `(text, every=200) -> dict` | Lê do log a saúde de uma rodada transiente: último tempo, Courant, limites de fração de fase, término e primeiro erro fatal. |
| `worker_commands` | função | `(plan, n, procs=4, publish_git=True) -> list[str]` | Gera uma linha de shell por máquina para rodar cada fatia da campanha. |
| `main` | função | `(argv=None) -> None` | Linha de comando do runner de campanhas (`--worker`, `--shard`, `--procs`, `--publish-git`, `--commands`). |

#### `pinneapple_simulation.external_solvers.openfoam.case_builder`

Copia um caso-modelo OpenFOAM e escreve `0/T` e `transportProperties` de um cenário (problema térmico com obstáculo). [sem teste]

<small>`pinneapple_simulation/external_solvers/openfoam/case_builder.py` · 105 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OpenFOAMCaseTemplate` | classe | `(template_dir, obstacle_stl)` | Caso-modelo OpenFOAM (pasta e STL do obstáculo) com validação dos arquivos exigidos. |
| `stage_case_for_scenario` | função | `(*, tpl, out_case_dir, scenario) -> None` | Copia o caso-modelo para a pasta de saída e escreve `0/T` e `transportProperties` do cenário. |

#### `pinneapple_simulation.external_solvers.openfoam.export_bundle`

Monta uma pasta-pacote copiando geometria, specs, pontos derivados e sensores amostrados; os nomes de arquivo estão fixos para um projeto de canal com cilindro. [sem teste]

<small>`pinneapple_simulation/external_solvers/openfoam/export_bundle.py` · 38 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `export_bundle` | função | `(*, project_dir, bundle_dir) -> None` | Copia geometria, specs, pontos derivados e sensores de um projeto para uma pasta-pacote. |

#### `pinneapple_simulation.external_solvers.openfoam.field_reader`

Lê o internalField de campos OpenFOAM (por padrão `p` e `U`) e empacota como `PhysicalSample`. [sem teste direto]

<small>`pinneapple_simulation/external_solvers/openfoam/field_reader.py` · 173 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `openfoam_case_to_upd` | função | `(case_dir, *, time=None, fields=('p', 'U'))` | Lê o internalField dos campos pedidos num tempo e empacota como `PhysicalSample`. |

#### `pinneapple_simulation.external_solvers.openfoam.mesh_reader`

Reconstrói centros e tamanhos de célula a partir de um polyMesh binário, no contrato de malha dos dataloaders de `pinneapple_data`. [sem teste direto]

<small>`pinneapple_simulation/external_solvers/openfoam/mesh_reader.py` · 125 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MeshGeometry` | classe | `(nodes, cell_size, cell_delta, bounds_min, bounds_max, n_points, n_faces, n_internal_face…)` | Geometria mínima de malha (centros, tamanhos de célula, limites, contagens) no contrato dos dataloaders. |
| `load_mesh` | função | `(source, prefix='constant/polyMesh/') -> MeshGeometry` | Constrói centros e tamanhos de célula a partir de um polyMesh binário (pasta ou arquivo). |

#### `pinneapple_simulation.external_solvers.openfoam.runner`

Executa um caso OpenFOAM via bash (blockMesh, snappyHexMesh opcional, solver serial ou MPI, comandos extras). [sem teste]

<small>`pinneapple_simulation/external_solvers/openfoam/runner.py` · 43 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OpenFOAMRunConfig` | classe | `(foam_bashrc, solver, n_procs, use_snappy, extra_cmds)` | Como rodar um caso OpenFOAM: bashrc, solver, número de processos, snappyHexMesh e comandos extras. |
| `run_openfoam_case` | função | `(case_dir, cfg) -> None` | Roda malha, solver (serial ou MPI) e comandos extras num caso OpenFOAM. |

#### `pinneapple_simulation.external_solvers.openfoam.sampling`

Escreve um `sampleDict` de nuvem de pontos, roda `postProcess -func sample` e lê os `.raw` resultantes em DataFrame. [sem teste]

<small>`pinneapple_simulation/external_solvers/openfoam/sampling.py` · 99 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `write_sample_dict_cloud` | função | `(case_dir, *, set_name, points, fields) -> Path` | Escreve `system/sampleDict` para amostragem em nuvem nos pontos dados. |
| `run_sampling` | função | `(case_dir, foam_bashrc, *, time=None) -> None` | Roda `postProcess -func sample` (último tempo por padrão). |
| `read_sampled_scalar_field` | função | `(case_dir, *, set_name, field) -> pd.DataFrame` | Lê um campo escalar amostrado (`x y z valor`) em DataFrame. |

#### `pinneapple_simulation.external_solvers.openfoam.splash_packer`

Empacota um caso OpenFOAM terminado num arquivo `.splash` (zip + `splash-manifest.json` com tamanho e sha256 por arquivo), lê o manifesto e verifica a integridade.

<small>`pinneapple_simulation/external_solvers/openfoam/splash_packer.py` · 138 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `pack_splash_archive` | função | `(case_dir, out_path, *, case_name=None, summary=None, generated_by=…, splash_version='1.0…)` | Compacta um caso OpenFOAM terminado num `.splash` com manifesto de tamanhos e hashes. |
| `read_splash_manifest` | função | `(splash_path) -> Dict[str, Any]` | Lê só o `splash-manifest.json` de um `.splash`. |
| `verify_splash_archive` | função | `(splash_path) -> Iterable[str]` | Lista os arquivos cujo tamanho ou sha256 não bate com o manifesto (vazio se íntegro). |

#### `pinneapple_simulation.external_solvers.openradioss`

Ponte para o OpenRadioss (FEA explícito): edição de deck para DoE, execução via Docker e leitores de resultados ANIM->VTK.

<small>`pinneapple_simulation/external_solvers/openradioss/__init__.py` · 18 linhas · testes: 2</small>

#### `pinneapple_simulation.external_solvers.openradioss.deck`

Edita decks OpenRadioss: lê e troca espessuras de `/PROP/SHELL`, escreve o engine com saída ANIM e prepara a pasta de um caso. [sem teste]

<small>`pinneapple_simulation/external_solvers/openradioss/deck.py` · 124 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `read_shell_thickness` | função | `(deck_text, prop_id) -> float` | Lê a espessura `Thick` de um `/PROP/SHELL/<id>`. |
| `set_shell_thickness` | função | `(deck_text, thickness_by_prop) -> str` | Devolve uma cópia do deck com as espessuras trocadas por id de propriedade. |
| `write_engine_anim` | função | `(path, run_name, t_end, anim_dt, *, extra_anim=('/ANIM/VECT/DISP',), header='', tail='') …)` | Escreve um arquivo engine que pede saída ANIM em vez de H3D. |
| `stage_case` | função | `(template_starter, engine_file, out_dir, thickness_by_prop, *, extra_files=None) -> Path` | Cria a pasta de um caso com o starter de espessura alterada e o engine original. |

#### `pinneapple_simulation.external_solvers.openradioss.log_reader`

Lê do log do engine os erros de balanço de energia (`ERR=...%`).

<small>`pinneapple_simulation/external_solvers/openradioss/log_reader.py` · 38 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `read_energy_errors` | função | `(case_dir_or_log) -> np.ndarray` | Devolve cada `ERR=...%` impresso pelo engine, em ordem. |
| `read_energy_error` | função | `(case_dir_or_log) -> float` | Devolve o maior \|ERR\| de balanço de energia do log, ou `nan`. |

#### `pinneapple_simulation.external_solvers.openradioss.runner`

Roda starter, engine e conversão ANIM->VTK do OpenRadioss via Docker, um caso ou vários em paralelo limitado. [sem teste]

<small>`pinneapple_simulation/external_solvers/openradioss/runner.py` · 111 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OpenRadiossDockerConfig` | classe | `(openradioss_dir, image, arch, threads, docker_bin, extra_path)` | Configuração da execução via Docker: pasta do OpenRadioss, imagem, arquitetura, threads e binário do docker. |
| `run_openradioss_case` | função | `(case_dir, run_name, cfg, *, starter=None, engine=None) -> Path` | Roda um caso preparado (starter, engine e conversão) e levanta erro se o solver falhar. |
| `run_openradioss_cases` | função | `(case_dirs, run_name, cfg, *, max_parallel=4, on_done=None) -> List[Path]` | Roda vários casos com paralelismo limitado, um solver de uma thread por caso. |
| `docker_available` | função | `(cfg) -> bool` | Diz se o Docker configurado responde. |

#### `pinneapple_simulation.external_solvers.openradioss.vtk_reader`

Lê resultados OpenRadioss convertidos por `anim_to_vtk` (VTK legado ASCII) e séries de deslocamento. [sem teste]

<small>`pinneapple_simulation/external_solvers/openradioss/vtk_reader.py` · 73 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RadiossMesh` | classe | `(points, node_ids, cells, cell_types, part_ids, time)` | Malha lida de um VTK do OpenRadioss: pontos, ids de nós, células, tipos, partes e tempo. |
| `read_vtk_mesh` | função | `(path) -> RadiossMesh` | Lê um arquivo VTK legado ASCII convertido por `anim_to_vtk`. |
| `read_case_displacements` | função | `(frames_dir) -> Tuple[np.ndarray, np.ndarray]` | Devolve tempos e deslocamentos (T, N, 3) a partir de arquivos `disp_XXX.txt`/`time_XXX.txt`. |

#### `pinneapple_simulation.external_solvers.turbodesigner`

Pacote da ponte TurboDesigner (análise de linha média de compressor axial). [sem teste]

<small>`pinneapple_simulation/external_solvers/turbodesigner/__init__.py` · 43 linhas · **sem teste direto**</small>

#### `pinneapple_simulation.external_solvers.turbodesigner.bridge`

Envolve o pacote `turbodesigner` para obter campos por estação de um compressor axial como fonte de dados para PINN; sem o pacote, usa um cálculo analítico simplificado (`fallback_if_missing=True`). [sem teste]

<small>`pinneapple_simulation/external_solvers/turbodesigner/bridge.py` · 400 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TurboDesignerConfig` | classe | `(pressure_ratio, num_stages, mass_flow_rate, rpm, inlet_total_pressure, inlet_total_tempe…)` | Parâmetros de um compressor axial no TurboDesigner (razão de pressão, estágios, vazão, rotação, condições de entrada). |
| `TurboDesignerWorkflow` | classe | `(base_config) -> None` | Atalho para solves repetidos do TurboDesigner sobre uma configuração base, com `solve`, `solve_upd` e `sweep`. |
| `run_turbodesigner` | função | `(cfg, *, fallback_if_missing=True) -> Dict[str, np.ndarray]` | Roda a análise de linha média e devolve arrays por estação; sem o pacote, usa o cálculo analítico simplificado. |
| `turbodesigner_to_upd` | função | `(cfg, **run_kwargs)` | Roda o TurboDesigner e empacota o resultado como `PhysicalSample`. |

#### `pinneapple_simulation.flight`

Dinâmica de voo: aeronave rígida 6-DOF com aerodinâmica de vortex lattice, empuxo, gravidade e trem de pouso, com trim, derivadas de estabilidade, piloto/piloto automático, decolagem, turbulência e cenas 3-D.

<small>`pinneapple_simulation/flight/__init__.py` · 59 linhas · testes: 9</small>

#### `pinneapple_simulation.flight.aero`

Aerodinâmica quase estacionária da aeronave inteira a partir do vortex lattice de `pinneapple_design.aero.vlm`, com superfícies de controle, deriva vertical e correções de fuselagem (Raymer). [sem teste direto]

<small>`pinneapple_simulation/flight/aero.py` · 239 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ControlSurface` | classe | `(name, surface, eta_in, eta_out, cf_c, antisymmetric, tau)` | Painéis do lattice movidos por um comando (asa ou cauda, faixa de envergadura, razão de corda, simétrico ou antissimétrico). |
| `Fin` | classe | `(area, height, ac, rudder_cf_c, sweep_deg, stall_deg)` | Deriva vertical modelada como uma linha sustentadora: área, altura, centro aerodinâmico, leme e enflechamento. |
| `AeroModel` | classe | `(af, cg, cd0, fin, controls, cma_fus, cnb_fus, cd_gear, cd_flap_per_rad, nc, ns_wing, ns_…)` | Modelo aerodinâmico da aeronave, construído a partir do airframe, que dá forças e momentos no CG em eixos do corpo. |
| `flap_effectiveness` | função | `(cf_c, correction=0.85) -> float` | Eficácia de um flap simples pela teoria de perfil fino, vezes uma correção empírica. |
| `helmbold_lift_slope` | função | `(aspect_ratio, sweep_half_chord_deg=0.0, eta=0.95) -> float` | Inclinação da curva de sustentação de uma asa finita pela forma de Helmbold/DATCOM (Raymer). |
| `fuselage_munk_factor` | função | `(wing_qc_fraction) -> float` | Fator K_fus de Raymer em função da posição da asa ao longo da fuselagem. |
| `fuselage_moments` | função | `(length, max_width, volume, wing_area, mac, span, depth, wing_qc_fraction)` | Contribuições da fuselagem a Cm_alpha e Cn_beta pelas equações de Raymer. |

#### `pinneapple_simulation.flight.aircraft`

A aeronave como corpo rígido 6-DOF: propriedades de massa, aerodinâmica, motores, trem de pouso e controles, com a avaliação de forças e a derivada do estado.

<small>`pinneapple_simulation/flight/aircraft.py` · 155 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Controls` | classe | `(elevator, aileron, rudder, flap, throttle, steer, brake, gear_down)` | Posições dos comandos (profundor, aileron, leme, flap, manete, direção, freio, trem) com convenção de sinais. |
| `Aircraft` | classe | `(name, af, aero, mass, cg, inertia, engines, gear, cabin, tail_point, wingtip, clmax_take…)` | A aeronave completa: airframe, aerodinâmica, massa, CG, inércia, motores, trem, pontos de cabine e limites de sustentação, com avaliação de forças. |

#### `pinneapple_simulation.flight.catalog`

Quatro aeronaves prontas (light_ga, regional_turboprop, a320, motor_glider) montadas com a geometria paramétrica e os modelos de peso e arrasto da biblioteca; números publicados e premissas ficam em `Aircraft.info`, nada ajustado a ensaio em voo.

<small>`pinneapple_simulation/flight/catalog.py` · 383 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ExplicitAirframe` | classe | `(wing_z, htail_spec, vtail_spec)` | Um `Airframe` com altura da asa e caudas dadas explicitamente (por exemplo asa alta com cauda em T). |
| `profile_cd0` | função | `(re, tc) -> float` | Arrasto de perfil por atrito de placa plana turbulenta vezes fator de forma de espessura. |
| `light_ga` | função | `() -> Aircraft` | Constrói a aeronave leve monomotor a pistão (classe Cessna 172). |
| `regional_turboprop` | função | `() -> Aircraft` | Constrói o turboélice regional bimotor de asa alta (classe ATR 72-600). |
| `a320` | função | `() -> Aircraft` | Constrói o jato bimotor de corredor único (classe A320). |
| `motor_glider` | função | `() -> Aircraft` | Constrói o motoplanador de grande alongamento. |
| `build` | função | `(name) -> Aircraft` | Constrói uma aeronave do catálogo pelo nome. |
| `names` | função | `() -> list[str]` | Lista os nomes das aeronaves do catálogo. |

#### `pinneapple_simulation.flight.control`

Modelos de piloto e piloto automático: atitude de arfagem, altitude, velocidade, asas niveladas/proa, amortecedor de guinada e o cronograma de decolagem.

<small>`pinneapple_simulation/flight/control.py` · 221 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LoopDesign` | classe | `(omega_pitch, zeta_pitch, omega_roll, zeta_roll, omega_yaw, zeta_yaw, pitch_integral, q_f…)` | Parâmetros de projeto das malhas de controle (frequências e amortecimentos de arfagem, rolamento e guinada). |
| `InnerLoops` | classe | `(ac, derivs, design=None)` | Malhas internas de atitude de uma aeronave, que dão incrementos de profundor, aileron e leme. |
| `StickFixed` | classe | `(controls)` | Controlador que mantém os comandos congelados nos valores de trim. |
| `Autopilot` | classe | `(ac, trim, derivs, design=None, altitude=None, heading=None, speed=None, omega_h=0.25, ta…)` | Piloto automático com manutenção de altitude, asas niveladas/proa, amortecedor de guinada e manete automática. |
| `TakeoffSchedule` | classe | `(V_R, theta_rotation, rotation_rate, V_climb, theta_max, theta_min, runway_heading, flap,…)` | Cronograma de decolagem: V_R, atitude e taxa de rotação, velocidade de subida, limites de arfagem, proa da pista, flap e manete. |
| `TakeoffPilot` | classe | `(ac, sched, derivs, design=None)` | Piloto de decolagem: manete total e flap, mantém o eixo da pista no solo, roda em V_R e sobe na velocidade alvo. |
| `wrap` | função | `(a) -> float` | Normaliza um ângulo para o intervalo (-pi, pi]. |

#### `pinneapple_simulation.flight.encounter`

Encontros com turbulência e rajadas em voo e estatísticas de fator de carga, com comparação à fórmula de Pratt-Walker. [sem teste direto]

<small>`pinneapple_simulation/flight/encounter.py` · 140 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `gust_sensitivity` | função | `(ac, V, rho, cl_alpha) -> float` | Sensibilidade dn/dw_rajada de uma asa fixa em escoamento quase estacionário. |
| `stall_angle` | função | `(ac, trim_alpha, trim_CL, cl_alpha) -> float` | Ângulo de ataque em que a sustentação linear atinge o CLmax limpo da aeronave. |
| `encounter_metrics` | função | `(ac, hist, t_skip=5.0, alpha_stall=None) -> dict[str, Any]` | Estatísticas de um encontro após um tempo inicial, opcionalmente checando o tempo acima do estol. |
| `turbulence_encounter` | função | `(ac, V, altitude, spec=None, wind=None, duration=60.0, dt=0.01, controller='autopilot', f…)` | Voa uma aeronave por turbulência durante um tempo, com piloto automático ou comandos fixos. |
| `discrete_gust` | função | `(ac, V, altitude, U, H=None, duration=None, dt=0.005)` | Resposta com comandos fixos a uma rajada vertical 1-cosseno, devolvendo histórico e pico de dn. |
| `pratt_check` | função | `(ac, V, altitude, U_de) -> dict[str, float]` | Compara o pico de dn simulado de uma rajada 1-cosseno com a fórmula de Pratt-Walker. |

#### `pinneapple_simulation.flight.figures`

Figuras estáticas (PNG, matplotlib) dos resultados: espectros de turbulência, trajetórias de decolagem, vento cruzado, fatores de carga e checagem de sensibilidade a rajada.

<small>`pinneapple_simulation/flight/figures.py` · 212 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `psd_figure` | função | `(curves, path) -> str` | Figura com um painel por espectro: medido contra o alvo. |
| `takeoff_figure` | função | `(runs, metrics, path) -> str` | Figura de decolagem: altura da roda e velocidade contra distância, arfagem contra tempo. |
| `lateral_figure` | função | `(runs, path, title='Crosswind take-off') -> str` | Figura de desvio lateral, ângulo de rolamento e leme contra tempo (vento cruzado). |
| `encounter_traces_figure` | função | `(runs, path, window=20.0, t0=10.0, notes=None) -> str` | Pequenos múltiplos de incremento de fator de carga e rolamento, uma linha por aeronave. |
| `encounter_bars_figure` | função | `(table, path) -> str` | Barras de RMS do fator de carga no CG e na cabine por aeronave. |
| `gust_sensitivity_figure` | função | `(table, pratt, path) -> str` | Figura que compara RMS medido com a previsão quase estacionária e o pico de rajada com Pratt-Walker. |
| `table_markdown` | função | `(rows, cols, digits=None) -> str` | Monta uma tabela Markdown a partir de linhas em dicionário. |

#### `pinneapple_simulation.flight.gear`

Contato do trem de pouso com pista plana: pernas mola-amortecedor com rolamento, frenagem e força lateral do pneu. [sem teste direto]

<small>`pinneapple_simulation/flight/gear.py` · 96 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GearLeg` | classe | `(name, contact, stiffness, damping, mu_roll, mu_brake, mu_side, slip_ref, steerable, brak…)` | Uma perna do trem de pouso: ponto de contato, rigidez, amortecimento, coeficientes de atrito, direção e freio. |
| `size_gear` | função | `(legs, mass, cg, static_compression=0.25, zeta=0.6, g=9.80665) -> list[GearLeg]` | Dimensiona mola e amortecedor de cada perna para uma compressão estática dada sob sua parcela do peso. |
| `gear_forces` | função | `(legs, x, C, cg, steer=0.0, brake=0.0, ground_z=0.0)` | Forças, momentos e cargas normais de todas as pernas para um estado da aeronave. |

#### `pinneapple_simulation.flight.mass`

Massa, centro de gravidade e tensor de inércia a partir de uma lista de componentes (ponto, barra, placa, cilindro). [sem teste direto]

<small>`pinneapple_simulation/flight/mass.py` · 83 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Component` | classe | `(name, mass, center, inertia)` | Um componente de massa com centro e inércia própria. |
| `point` | função | `(name, mass, center) -> Component` | Cria um componente de massa pontual. |
| `rod` | função | `(name, mass, center, length, axis=0) -> Component` | Cria um componente barra fina ao longo de um eixo. |
| `plate` | função | `(name, mass, center, span, chord, taper=1.0) -> Component` | Cria uma placa fina no plano x-y (asa, estabilizador), opcionalmente afilada. |
| `fin_plate` | função | `(name, mass, center, height, chord) -> Component` | Cria uma placa fina no plano x-z (deriva vertical). |
| `cylinder` | função | `(name, mass, center, length, radius, shell=False) -> Component` | Cria um cilindro ao longo de x (fuselagem, nacele), maciço ou casca fina. |
| `mass_properties` | função | `(components) -> tuple[float, np.ndarray, np.ndarray]` | Calcula massa total, CG e inércia em torno do CG. |
| `balance` | função | `(components, x_target, movable) -> list[Component]` | Desloca um componente ao longo de x para pôr o CG numa posição alvo. |

#### `pinneapple_simulation.flight.propulsion`

Motores: empuxo de turbofan e de hélice (pistão ou turboélice) contra velocidade e altitude, e a direção do empuxo. [sem teste direto]

<small>`pinneapple_simulation/flight/propulsion.py` · 74 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Turbofan` | classe | `(static_thrust, position, cant_deg, name)` | Turbofan de alto bypass com queda de empuxo instalado com velocidade e altitude. |
| `Propeller` | classe | `(power, diameter, efficiency, figure_of_merit, kind, position, cant_deg, name)` | Hélice movida por pistão ou turboélice: potência com a altitude, empuxo estático e empuxo em voo. |
| `thrust_direction_body` | função | `(cant_deg) -> np.ndarray` | Vetor unitário do empuxo em eixos do corpo para um ângulo de inclinação do motor. |

#### `pinneapple_simulation.flight.rigid_body`

Equações 6-DOF de corpo rígido em eixos do corpo com quatérnio de atitude, conversões de ângulos e passo RK4.

<small>`pinneapple_simulation/flight/rigid_body.py` · 116 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `quat_multiply` | função | `(a, b) -> np.ndarray` | Produto de dois quatérnios. |
| `quat_to_dcm` | função | `(q) -> np.ndarray` | Matriz de rotação corpo -> NED a partir do quatérnio. |
| `euler_to_quat` | função | `(phi, theta, psi) -> np.ndarray` | Converte rolamento, arfagem e guinada (3-2-1) em quatérnio. |
| `quat_to_euler` | função | `(q) -> tuple[float, float, float]` | Converte quatérnio em ângulos de Euler, tratando o gimbal lock. |
| `state_vector` | função | `(pos=(0, 0, 0), vel_body=(0, 0, 0), euler=(0, 0, 0), omega=(0, 0, 0)) -> np.ndarray` | Monta o vetor de estado de 13 componentes (posição, velocidade, quatérnio, rotação). |
| `rigid_body_derivative` | função | `(x, F_b, M_b, mass, inertia, inertia_inv=None, k_norm=1.0) -> np.ndarray` | Derivada temporal do estado de 13 componentes para força e momento totais no corpo. |
| `gravity_body` | função | `(q, mass, g=G0) -> np.ndarray` | Força peso em eixos do corpo. |
| `rk4_step` | função | `(f, t, x, dt) -> np.ndarray` | Um passo de Runge-Kutta clássico com renormalização do quatérnio. |
| `kinetic_energy` | função | `(x, mass, inertia) -> float` | Energia cinética de translação e rotação do estado. |
| `angular_momentum_ned` | função | `(x, inertia) -> np.ndarray` | Momento angular em torno do CG em eixos NED. |

#### `pinneapple_simulation.flight.simulate`

Integração no tempo de uma `Aircraft` com controlador e campo de vento, gravando o histórico de voo, e checagem de energia do integrador.

<small>`pinneapple_simulation/flight/simulate.py` · 165 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FlightHistory` | classe | `(aircraft, t, x, euler, airspeed, alpha, beta, CL, nz, ny, f_spec, thrust, gear_loads, co…)` | Histórico gravado de um voo (estados, ângulos, velocidades, fatores de carga, empuxo, cargas do trem, comandos, vento, eventos) com resumo. |
| `simulate` | função | `(ac, x0, controller, t_end, dt=0.01, wind=None, record_every=1, record_gamma=False, stop=…)` | Integra a aeronave no tempo com um controlador e um vento opcional e devolve o `FlightHistory`. |
| `energy_check` | função | `(ac, x0, t_end=5.0, dt=0.01) -> dict[str, float]` | Checa conservação de energia e momento do integrador com aerodinâmica, motores e trem desligados. |
| `rms` | função | `(a) -> float` | Valor RMS de um array. |

#### `pinneapple_simulation.flight.takeoff`

Decolagem: corrida, rotação em V_R, saída do chão e subida até a barreira de 35 ft, com a referência fechada de Raymer.

<small>`pinneapple_simulation/flight/takeoff.py` · 184 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `default_schedule` | função | `(ac, kind='transport') -> TakeoffSchedule` | Cronograma de decolagem padrão a partir da velocidade de estol com flap de decolagem. |
| `takeoff_derivatives` | função | `(ac, V) -> dict[str, float]` | Derivadas de controle em torno da subida com flap de decolagem, para projetar os ganhos do piloto. |
| `simulate_takeoff` | função | `(ac, schedule=None, wind=None, t_max=90.0, dt=0.01, h_stop=400 * FT, design=None, record_…)` | Simula a decolagem completa da aeronave até uma altura de parada. |
| `takeoff_metrics` | função | `(ac, hist) -> dict[str, Any]` | Extrai do histórico as métricas de decolagem (distâncias, velocidades, tempos). |
| `raymer_takeoff` | função | `(ac, V_LOF=None, mu=0.02, t_rotation=3.0, h_obstacle=SCREEN, theta_ground=0.0) -> dict[st…)` | Distância de decolagem pela forma fechada de Raymer com o empuxo, sustentação e arrasto desta aeronave. |
| `raymer_variants` | função | `(ac, metrics, kind='light') -> dict[str, dict[str, float]]` | As duas referências fechadas de Raymer citadas nos resultados de decolagem. |

#### `pinneapple_simulation.flight.trim`

Trim em voo reto, equilíbrio no solo, derivadas de estabilidade e modos dinâmicos (autovalores da linearização) de uma `Aircraft`.

<small>`pinneapple_simulation/flight/trim.py` · 190 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TrimResult` | classe | `(V, altitude, gamma, alpha, theta, elevator, throttle, x, controls, residual, CL, feasibl…)` | Resultado de trim: velocidade, altitude, ângulos, profundor, manete, estado, resíduo e viabilidade. |
| `trim` | função | `(ac, V, altitude, gamma=0.0, flap=0.0, gear_down=False, psi=0.0) -> TrimResult` | Acha o trim de voo reto e nivelado (ou em subida) com asas niveladas. |
| `ground_equilibrium` | função | `(ac, flap=0.0, throttle=0.0, psi=0.0, x_pos=0.0, y_pos=0.0) -> np.ndarray` | Estado de repouso sobre o trem: altura e arfagem com força vertical e momento nulos. |
| `stability_derivatives` | função | `(ac, tr, d_angle=math.radians(0.5), d_rate=0.01, d_ctl=math.radians(1.0)) -> dict[str, fl…)` | Derivadas aerodinâmicas em torno do trim por diferenças finitas. |
| `modes` | função | `(ac, tr, eps=1e-05) -> dict[str, object]` | Autovalores da linearização de 12 estados em torno do trim, com comandos fixos. |
| `classify_modes` | função | `(eigs, tol=1e-06) -> dict[str, dict[str, float]]` | Nomeia os pares complexos de autovalores por frequência, amortecimento e período. |

#### `pinneapple_simulation.flight.turbulence`

Turbulência atmosférica e rajadas (MIL-F-8785C / MIL-HDBK-1797): especificações, espectros, filtros de Dryden, campos congelados 1-D e 3-D (von Karman), vento com cisalhamento e rajada 1-cosseno. `WindField.velocity` é interface abstrata (`NotImplementedError`).

<small>`pinneapple_simulation/flight/turbulence.py` · 476 linhas · testes: 19 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TurbulenceSpec` | classe | `(sigma_u, sigma_v, sigma_w, L_u, L_v, L_w, model, altitude, severity)` | Intensidades RMS e comprimentos de escala das três componentes da turbulência e a forma espectral. |
| `WindField` | classe | `()` | Interface de campo de vento: `velocity(t, pontos)` em NED; soma com `+`. |
| `SumWind` | classe | `(fields)` | Soma de vários campos de vento. |
| `ConstantWind` | classe | `(ned)` | Vento uniforme e constante. |
| `ShearWind` | classe | `(w20, from_deg, z0, ground_z)` | Vento médio com perfil logarítmico MIL-F-8785C e direção meteorológica. |
| `FrozenLine` | classe | `(spec, length, dx, heading_deg, origin, seed, series)` | Uma realização congelada das três componentes ao longo de uma linha reta. |
| `DrydenFilter` | classe | `(spec, V, dt, seed=0)` | Filtros de forma de Dryden (MIL-HDBK-1797) excitados por ruído branco, para um ponto voando a velocidade V. |
| `FrozenField3D` | classe | `(spec, n_shells, n_dirs, k_min, k_max, dx_min, heading_deg, seed)` | Turbulência 3-D isotrópica e sem divergência de von Karman por modos de Fourier aleatórios. |
| `DiscreteGust` | classe | `(amplitude, length, start, shape, direction, heading_deg, origin)` | Rajada discreta 1-cosseno (ou rampa) encontrada quando a aeronave passa um ponto do trajeto. |
| `turbulence_spec` | função | `(altitude, severity='moderate', model='von_karman', *, w20=None, probability=None) -> Tur…)` | Intensidades e escalas da turbulência numa altitude para uma severidade e modelo. |
| `psd` | função | `(spec, Omega, component='w') -> np.ndarray` | PSD espacial unilateral de uma componente numa frequência espacial. |
| `psd_temporal` | função | `(spec, omega, V, component='w') -> np.ndarray` | PSD temporal unilateral vista a uma velocidade V (campo congelado). |
| `welch_psd` | função | `(x, d, nseg=16)` | PSD unilateral de um sinal amostrado pelo método de Welch. |
| `path_axes` | função | `(heading_deg) -> np.ndarray` | Vetores unitários u, v, w do trajeto em NED para uma proa. |
| `von_karman_energy` | função | `(k, L, sigma=1.0)` | Espectro de energia isotrópico de von Karman normalizado. |
| `isotropic_line_spectra` | função | `(k1, L, sigma=1.0)` | Espectros longitudinal e transversal 1-D do campo isotrópico, por quadratura. |
| `pratt_gust_load` | função | `(wing_loading, cl_alpha, mac, U_de, V_e, rho0=1.225, rho=1.225, g=9.80665)` | Incremento de fator de carga de rajada pela fórmula de Pratt-Walker. |

#### `pinneapple_simulation.flight.verification`

Verificação dos blocos da dinâmica de voo como números citáveis: turbulência, trim, conservação e lattice. [sem teste direto]

<small>`pinneapple_simulation/flight/verification.py` · 142 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `band_ratios` | função | `(W, P, target, lo, hi, nbins=8)` | Razões medido/alvo de PSD médias por banda de frequência. |
| `turbulence_checks` | função | `(seed=3) -> dict[str, Any]` | Checa variância e PSD de cada gerador de turbulência contra o alvo. |
| `trim_checks` | função | `(names=None, hold=10.0) -> list[dict[str, Any]]` | Checa resíduo de trim e deriva de velocidade, altitude e fator de carga com comandos fixos em cada aeronave. |
| `conservation_check` | função | `() -> dict[str, Any]` | Checa erro de energia e momento angular do integrador num corpo girando sob gravidade. |
| `lattice_check` | função | `() -> list[dict[str, Any]]` | Compara a sustentação do modelo com lattice no laço com o vortex lattice estacionário. |
| `run` | função | `() -> dict[str, Any]` | Roda todas as verificações e devolve um dicionário. |

#### `pinneapple_simulation.flight.visuals`

Cenas 3-D de um voo (aeronave na trajetória, pista, rajadas como setas, carga do lattice e HUD) como `Scene` de `pinneapple_tools.visualization.studio`.

<small>`pinneapple_simulation/flight/visuals.py` · 236 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `body_vertices` | função | `(ac, V_airframe) -> np.ndarray` | Converte pontos em eixos do airframe para o referencial do corpo em torno do CG. |
| `aircraft_surfaces` | função | `(ac, group='aircraft', parts=None)` | Geometria visual da aeronave como superfícies do studio no referencial do corpo. |
| `lattice_surface` | função | `(ac, group='aircraft', lift=0.6)` | Painéis do vortex lattice como uma folha acima da superfície, para colorir pela carga. |
| `runway_surfaces` | função | `(length=3000.0, width=45.0, x0=-300.0, grass=1500.0)` | Superfícies da pista com marcas e grama ao redor. |
| `frame_indices` | função | `(hist, n_frames) -> np.ndarray` | Índices do histórico usados como quadros da animação. |
| `flight_scene` | função | `(ac, hist, *, n_frames=240, wind=None, runway=True, runway_length=None, gust_grid=7, gust…)` | Cena 3-D animada de um voo, com pista, rajadas, carga e HUD opcionais. |
| `gust_wisps` | função | `(ac, hist, wind, n_lat=5, n_vert=3, spacing=None, exaggeration=0.6, ahead=0.15)` | Linhas estáticas que mostram o campo de rajada congelado ao longo do trajeto. |
| `scene_summary` | função | `(sc) -> dict[str, Any]` | Resumo em dicionário de uma cena gerada. |

#### `pinneapple_simulation.geophysics`

Blocos do sistema terrestre verificados contra soluções exatas ou testes padrão: águas rasas na esfera, equação de Richards e modelo de balanço de energia de duas camadas.

<small>`pinneapple_simulation/geophysics/__init__.py` · 40 linhas · testes: 11</small>

#### `pinneapple_simulation.geophysics.energy_balance`

Modelo de balanço de energia de duas camadas do aquecimento médio global (Held et al. 2010; Geoffroy et al. 2013), com integração exata, resposta ao degrau, ECS/TCR e ajuste de parâmetros. [sem teste direto]

<small>`pinneapple_simulation/geophysics/energy_balance.py` · 136 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TwoLayerEBM` | classe | `(lam, gamma, C, C0, eps)` | Modelo de duas camadas (oceano superior e profundo) com simulação, modos, resposta ao degrau, ECS e TCR. |
| `fit_two_layer` | função | `(T_obs, forcing, dt=1.0, fit_eps=False, guess=None, sigma=None) -> dict[str, object]` | Ajuste por mínimos quadrados dos parâmetros a uma série de aquecimento forçada. |

#### `pinneapple_simulation.geophysics.richards`

Fluxo não saturado no solo: equação de Richards 1-D (forma mista) com os modelos de retenção e condutividade clássicos. [sem teste direto]

<small>`pinneapple_simulation/geophysics/richards.py` · 316 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `VanGenuchten` | classe | `(theta_r, theta_s, alpha, n, Ks, l)` | Solo de van Genuchten-Mualem: saturação efetiva, umidade, condutividade e capacidade. |
| `BrooksCorey` | classe | `(theta_r, theta_s, h_b, lam, Ks)` | Solo de Brooks-Corey. |
| `ClappHornberger` | classe | `(theta_s, psi_s, b, Ks, theta_r)` | Solo de Clapp-Hornberger (mesmas leis de potência de Campbell; `Campbell` é um alias). |
| `Gardner` | classe | `(theta_r, theta_s, alpha, Ks)` | Solo de Gardner (condutividade exponencial), que tem perfil estacionário exato. |
| `Boundary` | classe | `(kind, value)` | Condição de contorno: carga, fluxo ou drenagem livre, com valor constante ou função do tempo. |
| `RichardsResult` | classe | `(z, t, h, theta, flux_top, flux_bottom, mass_balance_ratio, n_steps, n_picard)` | Resultado: perfis de carga e umidade, fluxos no topo e no fundo, balanço de massa e contagens de passos. |
| `solve_richards` | função | `(soil, depth, n_cells, h0, t_end, top, bottom, dt0=1.0, dt_min=1e-06, dt_max=None, t_out=…)` | Resolve uma coluna de solo no tempo com passo adaptativo e iteração de Picard. |
| `gardner_steady_infiltration` | função | `(soil, z_above_table, q0) -> np.ndarray` | Perfil de carga estacionário exato acima do lençol para o solo de Gardner. |

#### `pinneapple_simulation.geophysics.sphere_shallow_water`

Equações de águas rasas na esfera em rotação: solver espectral (vorticidade-divergência, grade gaussiana) e os casos de teste de Williamson et al. (1992).

<small>`pinneapple_simulation/geophysics/sphere_shallow_water.py` · 298 linhas · testes: 11</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SphericalHarmonics` | classe | `(trunc, nlat=None, nlon=None, radius=EARTH['a'])` | Transformadas espectrais escalares e vetoriais com truncamento triangular numa grade gaussiana. |
| `SphereShallowWater` | classe | `(sh, zeta, delta, phi, f, phi_s, omega, g, hyperdiffusion, t, history)` | Estado e integrador do modelo de águas rasas (vorticidade, divergência, geopotencial), com diagnósticos. |
| `legendre_tables` | função | `(mu, M, N) -> tuple[np.ndarray, np.ndarray]` | Tabelas das funções de Legendre associadas normalizadas e suas derivadas. |
| `stable_dt` | função | `(trunc, gh_max, u_max, radius=EARTH['a'], safety=0.7) -> float` | Passo de tempo RK4 explícito estável para a truncagem e velocidades dadas. |
| `williamson_case2` | função | `(sh, alpha=0.0, gh0=29400.0, **kw) -> SphereShallowWater` | Caso 2 de Williamson: escoamento zonal geostrófico estacionário. |
| `williamson_case5` | função | `(sh, **kw) -> SphereShallowWater` | Caso 5 de Williamson: escoamento zonal sobre montanha cônica. |
| `williamson_case6` | função | `(sh, R=4, omega_w=7.848e-06, K=7.848e-06, h0=8000.0, **kw) -> SphereShallowWater` | Caso 6 de Williamson: onda de Rossby-Haurwitz. |
| `rossby_haurwitz_speed` | função | `(R=4, omega_w=7.848e-06, omega=EARTH['omega']) -> float` | Velocidade de fase angular da onda de Rossby-Haurwitz no modelo barotrópico não divergente. |
| `height_errors` | função | `(sh, h, h_ref) -> dict[str, float]` | Erros normalizados l1, l2 e l_inf de Williamson et al. |

#### `pinneapple_simulation.meshless`

Meshless and mesh-based solvers of the same 2-D problems, and the study that compares them.

<small>`pinneapple_simulation/meshless/__init__.py` · 23 linhas · testes: 14</small>

#### `pinneapple_simulation.meshless.contracts`

A meshless study as the apps' ``StudyResult`` contract (``pinneapple_data.contracts``): one table row per method and spacing, the convergence and the winners as metrics, and checks that carry the value they were decided on.

<small>`pinneapple_simulation/meshless/contracts.py` · 50 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `meshless_checks` | função | `(res) -> list` | Observed order against the expected one where the solution is smooth enough to expect it, and stability. |
| `study_result` | função | `(res, *, study_id, title, limits=(), app='meshless')` |  |

#### `pinneapple_simulation.meshless.domains`

2-D domains and node sets for the meshless study.

<small>`pinneapple_simulation/meshless/domains.py` · 218 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NodeSet` | classe | `(X, boundary, h)` |  |
| `Domain2D` | classe | `(name, sdf, pieces, lo, hi, description, cloud, eval_points)` |  |
| `polygon_domain` | função | `(vertices, name='polygon', holes=()) -> Domain2D` | A simple polygon (vertices in order), optionally with circular holes ``((cx, cy), r)``. |
| `domain` | função | `(name) -> Domain2D` | A preset: ``square``, ``lshape``, ``disk``, ``annulus``, ``plate_hole``. |
| `point_cloud_domain` | função | `(X, boundary, *, eval_fraction=0.1, seed=0, name='point cloud') -> Domain2D` | An uploaded point cloud: nodes ``X`` (N, 2) with a boundary flag. ``eval_fraction`` of the interior points are held out as evaluation points (not used by any method). |

#### `pinneapple_simulation.meshless.methods`

Five ways to solve the same 2-D problem on the same domain.

<small>`pinneapple_simulation/meshless/methods.py` · 427 linhas · testes: 2</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Discretization` | classe | `(method, nodes, evaluate, setup_s, L, K, M, kansa, info, options)` |  |
| `default_eps` | função | `(domain) -> float` | Shape parameter of the multiquadric: 3 / diameter (a fixed, not node-scaled, value: spectral convergence until the condition number stops it). |
| `rbf_fd_weights` | função | `(X, Q, idx, op='lap', degree=4) -> np.ndarray` | RBF-FD weights (q, n) of the stencils ``idx`` (q, n) (rows of ``X``) for the operator ``op`` at the points ``Q``: ``lap`` (the Laplacian) or ``eval`` (interpolation). PHS r³ + all monomials up to ``degree``, in local coordinates centred at the point and scaled by the stencil radius. |
| `fd_nodes` | função | `(domain, h)` | Grid nodes inside the domain and the points where grid lines cross the boundary. |
| `discretize` | função | `(method, domain, nodes, **options) -> Discretization` | Build the operator of ``method`` on ``nodes`` (FD builds its own grid nodes at the same spacing). |
| `solve` | função | `(problem, disc, *, seed=0) -> dict[str, Any]` | Solve on the discretization; returns ``u`` (nodal values, or the network for the PINN) and the costs. |

#### `pinneapple_simulation.meshless.pinn`

PINN on the meshless node set: the interior nodes are the collocation points, the boundary nodes carry the Dirichlet condition, and training stops at a wall-clock budget (``pinneapple_analysis.cost.train_within_budget``), so the PINN is compared with the other methods at a stated cost. …

<small>`pinneapple_simulation/meshless/pinn.py` · 89 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `train_pinn` | função | `(problem, disc, *, budget_s=5.0, width=32, depth=3, lr=0.002, bc_weight=10.0, n_times=8, …)` |  |

#### `pinneapple_simulation.meshless.problems`

Linear 2-D problems with Dirichlet conditions, defined by expressions in ``x``, ``y`` (and ``t``).

<small>`pinneapple_simulation/meshless/problems.py` · 147 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Problem` | classe | `(kind, exact, source, boundary, initial, k, alpha, t_end, n_steps)` |  |

#### `pinneapple_simulation.meshless.stencils`

Stencil weights in any dimension for meshless discretisations: RBF-FD and GFDM (weighted least squares).

<small>`pinneapple_simulation/meshless/stencils.py` · 105 linhas · testes: 2</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `monomials` | função | `(dim, degree) -> list[tuple[int, ...]]` | Exponent tuples of every monomial of total degree <= ``degree`` in ``dim`` variables (constant first). |
| `stencil_weights` | função | `(X, Q, idx, ops=('lap',), *, method='rbf_fd', degree=2, batch=2000) -> dict[str, np.ndarr…)` | Weights ``{op: (q, n)}`` (see the module docstring). ``X``: (N, dim) nodes; ``Q``: (q, dim) query points; ``idx``: (q, n) stencil node indices. |

#### `pinneapple_simulation.meshless.study`

Refinement study of meshless and mesh-based methods on one problem and one domain.

<small>`pinneapple_simulation/meshless/study.py` · 258 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MeshlessStudy` | classe | `(problem, domain, rows, convergence, eps_sweep, wins, notes, reference, fields, skipped, …)` |  |
| `meshless_study` | função | `(problem, dom, h=None, *, methods=…, options=None, pinn_levels='finest', n_eval=2000, tol…)` | Run every method at every spacing in ``h`` (default: three halvings from diameter / 10). ``options``: per method, e.g. ``{"kansa": {"eps": 2.0}, "rbf_fd": {"degree": 3}, "pinn": {"budget_s": 5}}``. ``pinn_levels``: ``finest`` (default, the PINN trains once) or ``all``. |

#### `pinneapple_simulation.multibody`

Dinâmica de corpos rígidos em coordenadas reduzidas: uma base flutuante com árvores de juntas de revolução; reexporta a árvore e os utilitários de braço.

<small>`pinneapple_simulation/multibody/__init__.py` · 26 linhas · testes: 45</small>

#### `pinneapple_simulation.multibody.arm`

Braços seriais de base fixa sobre `MultibodyTree`: dinâmica inversa e direta, referência analítica do braço plano de dois elos, trajetórias quínticas, balanço de energia e checagem de tensão na raiz dos elos.

<small>`pinneapple_simulation/multibody/arm.py` · 161 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `fixed_base_inverse_dynamics` | função | `(tree, q, qd, qdd) -> np.ndarray` | Calcula os torques de junta ao longo de uma trajetória (q, qd, qdd) com a base fixa na origem. |
| `fixed_base_forward` | função | `(tree, q0, qd0, tau_fn, t) -> tuple[np.ndarray, np.ndarray]` | Integra (RK4) os ângulos e velocidades de junta sob torques dados por `tau_fn(t, q, qd)`. |
| `planar_two_link_torques` | função | `(m1, m2, l1, lc1, lc2, I1, I2, g, q, qd, qdd) -> np.ndarray` | Torques em forma fechada de um braço plano de dois elos num plano vertical, usados como referência independente. |
| `quintic` | função | `(q0, q1, T, t) -> tuple[np.ndarray, np.ndarray, np.ndarray]` | Gera uma trajetória quíntica repouso-a-repouso de `q0` a `q1` em `T` segundos (posição, velocidade, aceleração). |
| `energy_balance` | função | `(tree, t, q, qd, tau) -> dict` | Compara o trabalho dos torques (trapézio) com a variação de energia cinética + potencial e devolve o desvio relativo. |
| `link_world_points` | função | `(tree, q, points) -> dict[str, np.ndarray]` | Posições no mundo de pontos fixos em cada corpo, para cada configuração. |
| `lumped_link` | função | `(name, length, mass_per_m, point_masses=(), width=0.05)` | Cria um `Body` de elo reto: viga uniforme mais massas pontuais (motor, carga), com massa, centro de massa e inércia (eixos paralelos). |
| `link_root_checks` | função | `(tau, section, *, yield_strength, fatigue_strength=None, safety=3.0, names=None) -> tuple…)` | Tensão de flexão na raiz de cada elo a partir dos torques de junta, checada contra escoamento/segurança e, opcionalmente, fadiga. |

#### `pinneapple_simulation.multibody.tree`

Árvore multicorpo de base flutuante com juntas de revolução: cinemática, jacobianos, matriz de massa, forças de viés, limites de junta por mola de penalidade e passo de tempo linearmente implícito. [sem teste]

<small>`pinneapple_simulation/multibody/tree.py` · 332 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Body` | classe | `(name, mass, com, inertia, parent_joint)` | Corpo rígido com massa, centro de massa e tensor de inércia no referencial do corpo. |
| `RevoluteJoint` | classe | `(name, parent, child, origin, axis, R0, limits, damping)` | Junta de revolução entre pai e filho, com origem, eixo, rotação inicial, limites (mola de penalidade) e amortecimento. |
| `Kinematics` | classe | `(R, p, c, axis, o, Jv, Jw, omega, vel)` | Poses e velocidades no mundo de todos os corpos para um estado, com jacobianos. |
| `MultibodyTree` | classe | `(gravity=(0.0, 0.0, -1.62))` | Árvore multicorpo (o primeiro corpo é a base flutuante) com cinemática, matriz de massa, forças de viés, gravidade, passo de tempo e energia. |
| `cross` | função | `(a, b)` | Produto vetorial de arrays (..., 3), mais barato que `np.cross` para arrays pequenos. |
| `skew` | função | `(a)` | Matriz antissimétrica de um vetor 3D. |
| `quat_to_matrix` | função | `(q)` | Converte quatérnio em matriz de rotação. |
| `axis_angle` | função | `(axis, angle)` | Matriz de rotação a partir de eixo e ângulo. |
| `quat_mul` | função | `(a, b)` | Produto de dois quatérnios. |
| `quat_from_matrix` | função | `(R)` | Converte matriz de rotação em quatérnio. |

#### `pinneapple_simulation.numerical_solvers`

_Módulo sem docstring e sem texto._

<small>`pinneapple_simulation/numerical_solvers/__init__.py` · 73 linhas · testes: 111</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `openfoam_available` | função | `() -> bool` | Return True if the OpenFOAM bridge is importable. |
| `fenics_available` | função | `() -> bool` | Return True if the FEniCS bridge is importable. |

#### `pinneapple_simulation.numerical_solvers._bc`

Read Dirichlet conditions of a ProblemSpec for grid / node-based solvers.

<small>`pinneapple_simulation/numerical_solvers/_bc.py` · 82 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `dirichlet_conditions` | função | `(spec) -> List` |  |
| `select` | função | `(cond, pts, edges=None, boundary=None) -> torch.Tensor` | Indices of ``pts`` the condition applies to. |
| `values` | função | `(cond, pts, field=0) -> torch.Tensor` | Dirichlet values at ``pts`` for one field (column ``field`` of a multi-field value_fn). |
| `boundary_rect` | função | `(nodes, edges) -> torch.Tensor` |  |

#### `pinneapple_simulation.numerical_solvers.adapters`

_Módulo sem docstring e sem texto._

<small>`pinneapple_simulation/numerical_solvers/adapters/__init__.py` · 20 linhas · **sem teste direto**</small>

#### `pinneapple_simulation.numerical_solvers.adapters.upd_mesh`

UPD mesh adapter for solver inputs (vertices, faces, topology).

<small>`pinneapple_simulation/numerical_solvers/adapters/upd_mesh.py` · 163 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `mesh_to_fvm_topology` | função | `(mesh, *, device=None, dtype=None) -> Dict[str, torch.Tensor]` | Build a minimal FVM topology from a surface mesh (triangles/quads). |
| `mesh_to_fem_assembly_inputs` | função | `(mesh, *, device=None, dtype=None) -> Dict[str, torch.Tensor]` | Minimal FEM assembly inputs (MVP): - nodes (N,3) - elements (E,k) connectivity The actual K,f assembly is handled by user-provided assemble_fn (in FEMSolver). |

#### `pinneapple_simulation.numerical_solvers.adapters.upd_signal`

UPD signal/sample adapter for solver field extraction and stacking.

<small>`pinneapple_simulation/numerical_solvers/adapters/upd_signal.py` · 184 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `list_field_vars` | função | `(sample) -> List[str]` | Tries to list available physical field variables from a PhysicalSample-like object. |
| `list_signal_axes` | função | `(sample, var) -> List[str]` | Best-effort axis naming for a field. If sample has sample.axes[var] or sample.dims[var], use it. Fallback: ["dim0","dim1",...] |
| `extract_1d_signal` | função | `(sample, *, var, axis=-1, reduce='mean', index=None, device=None, dtype=None) -> Tuple[to…)` | Extract a 1D signal from a (possibly multi-dim) physical field. |
| `to_signal_batch` | função | `(samples, *, var, axis=-1, reduce='mean', index=None, device=None, dtype=None) -> Tuple[t…)` | Convert a list of samples into a padded batch tensor (B,Tmax). |

#### `pinneapple_simulation.numerical_solvers.base`

Base SolverOutput and SolverBase for numerical solvers.

<small>`pinneapple_simulation/numerical_solvers/base.py` · 34 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SolverOutput` | classe | `(result, losses, extras)` | Standard output for solvers / numerical methods. result: - main output tensor (spectra, modes, field solution, etc.) extras: - metadata (freqs, imfs, diagnostics, residuals) |
| `SolverBase` | classe | `()` | Base class for numerical solvers. |

#### `pinneapple_simulation.numerical_solvers.beam_bvp_fdm`

Euler-Bernoulli beam deflection: 4th-order finite-difference solve of the beam BVP

<small>`pinneapple_simulation/numerical_solvers/beam_bvp_fdm.py` · 296 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SectionProps` | classe | `(I_m4, c_m, Q_m3, t_m, A_m2)` | Cross-section properties required for stress calculations. |
| `BeamBVPFDMSolver` | classe | `(nx=200, bc_type=BC_SIMPLY_SUPPORTED)` | Thin `SolverBase`/registry wrapper. `solve_euler_bernoulli_beam_1d` is the primary entry point and can be used directly without this wrapper. |
| `compute_stresses` | função | `(moment, shear, section)` | Bending stress (extreme fibre), shear stress (neutral axis), von Mises. |
| `second_derivative` | função | `(f, dz) -> np.ndarray` | Second derivative of `f`, second-order accurate everywhere including the two boundary points (3-point central stencil in the interior, a 3-point one-sided stencil at each edge). |
| `solve_euler_bernoulli_beam_1d` | função | `(L_m=10.0, E_Pa=200000000000.0, I_m4=8.33e-06, q_N_per_m=1000.0, nx=200, bc_type=BC_SIMPL…)` | Solve the 1D Euler-Bernoulli beam BVP EI*w''''=q under a uniform distributed load q_N_per_m, for one of the three BC_* boundary condition sets. `section` (default: an 0.10x0.10 m rectangular section) supplies the cross-section geometry used for the returned stresses -- pass a `SectionProps` built … |

#### `pinneapple_simulation.numerical_solvers.beam_fe`

Euler-Bernoulli beam by Hermite finite elements: static deflection and natural modes.

<small>`pinneapple_simulation/numerical_solvers/beam_fe.py` · 74 linhas · testes: 27</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `beam_matrices` | função | `(L, EI, m, n_el)` | Global stiffness, consistent mass and node positions of a uniform beam. |
| `beam_static` | função | `(L, EI, q, n_el=40, ends=('pinned', 'pinned')) -> dict` | Deflection under a uniform load ``q`` (N/m, positive down) with consistent nodal loads. |
| `beam_modes` | função | `(L, EI, m, n_el=40, n_modes=4, ends=('pinned', 'pinned')) -> dict` | Natural frequencies (Hz) and mass-normalised mode shapes (deflection at the nodes). |

#### `pinneapple_simulation.numerical_solvers.bekker_wong`

Bekker-Wong numerical solver for rigid-wheel / deformable-soil interaction.

<small>`pinneapple_simulation/numerical_solvers/bekker_wong.py` · 347 linhas · testes: 50</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BekkerWongSolver` | classe | `(soil=None, wheel=None, rear_angle_ratio=1.0 / 3.0)` | Numerical Bekker-Wong solver. |
| `bekker_wong_forces_torch` | função | `(slip, z, *, R, b, c, tan_phi, K, k_c, k_phi, n, a0=0.4, a1=0.15, rear_angle_ratio=1.0 / …)` | Bekker-Wong (F_x, F_z, M_y) for batches, piecewise Gauss-Legendre on the three smooth sub-intervals [theta_r, min(0,theta_m)], [.., theta_m], [theta_m, theta_f]. |

#### `pinneapple_simulation.numerical_solvers.ceemdan`

CEEMDAN: Complete Ensemble Empirical Mode Decomposition with Adaptive Noise.

<small>`pinneapple_simulation/numerical_solvers/ceemdan.py` · 103 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CEEMDANSolver` | classe | `(n_ensembles=50, noise_std=0.2, max_imfs=10, seed=None)` |  |
| `ceemdan_1d` | função | `(x, *, n_ensembles=50, noise_std=0.2, max_imfs=10, seed=None) -> np.ndarray` | Approximate CEEMDAN. Returns imfs (K,T). |

#### `pinneapple_simulation.numerical_solvers.cfd_pipeline`

Full CAD-to-CFD pipeline: STEP/STL → mesh → Navier-Stokes solve.

<small>`pinneapple_simulation/numerical_solvers/cfd_pipeline.py` · 954 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CFDMesh` | classe | `(nodes, elements, boundary_tags) -> None` | Unstructured mesh for CFD computations. |
| `NSFlowSolver` | classe | `(mesh, nu=0.001, rho=1.0, solver='direct') -> None` | Mesh-native incompressible Navier-Stokes solver. |
| `CADToCFDPipeline` | classe | `(nu=0.001, rho=1.0) -> None` | Full pipeline: CAD file → mesh → NS solve → PINN training data. |

#### `pinneapple_simulation.numerical_solvers.compressible_fv`

Compressible Navier-Stokes / Euler finite-volume solver (2-D and 3-D Cartesian), PyTorch.

<small>`pinneapple_simulation/numerical_solvers/compressible_fv.py` · 299 linhas · testes: 15</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CompressibleConfig` | classe | `(shape, lengths, gamma, Pr, mu_ref, omega, cfl, limiter, bc, inflow, p_out, device, dtype…)` |  |
| `CompressibleFV` | classe | `(cfg)` |  |
| `sod_exact` | função | `(x, t, left=(1.0, 0.0, 1.0), right=(0.125, 0.0, 0.1), gamma=1.4, x0=0.5)` | Exact solution of the 1-D Riemann problem (Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed., 2009, ch. 4); returns rho, u, p on x for ``left``/``right`` = (rho, u, p). |

#### `pinneapple_simulation.numerical_solvers.cylinder_wake`

Confined cylinder wake with the D2Q9 lattice-Boltzmann solver: one run, its verification and the zoo dataset ``cylinder_wake_lbm``.

<small>`pinneapple_simulation/numerical_solvers/cylinder_wake.py` · 178 linhas · testes: 3</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `probe_strouhal` | função | `(signal, dt) -> float` | Strouhal number of a probe record sampled every ``dt`` convective units: strongest peak of the Hann-windowed, 4x zero-padded spectrum with Gaussian peak interpolation (a fraction of a bin). |
| `simulate` | função | `(Re, D=SETUP['D'], height=SETUP['height'], length=SETUP['length'], u_in=SETUP['u_in'], co…)` | One run. Returns the setup, the convective times ``t`` and vorticity frames ``vorticity`` (n, nx, ny) of the last ``keep_time`` convective units (``None``: the second half of the run), the probe ``probe_uy`` (/U) over the same times, ``wake_amplitude`` (its rms), ``amplitude_growth`` (probe rms of … |
| `verify` | função | `(Ds=(20, 30, 40), Re=100.0, channel_Re=2.0) -> dict` | Developed channel profile (plane Poiseuille; at ``channel_Re`` on D the entrance length is ~2.5 D, so the profile is developed 9 D downstream) on the dataset grid and a 2x coarser one, and the resolution study of the wake at ``Re``. At the dataset's Reynolds numbers the 12 D channel is far shorter … |
| `generate` | função | `(out, workers=1, name='cylinder_wake_lbm') -> dict` | Run :data:`PLAN` and write ``out/<name>/<split>[-NN].npz`` (one row per snapshot) and ``meta.json``. |

#### `pinneapple_simulation.numerical_solvers.dfsph`

DFSPH: Divergence-Free SPH (Bender & Koschier 2015 / 2017).

<small>`pinneapple_simulation/numerical_solvers/dfsph.py` · 317 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DFSPHSolver` | classe | `(h=0.04, mass=1.0, rho0=1000.0, mu=0.02, gravity=None, boundary_lo=None, boundary_hi=None…)` | Divergence-Free SPH (DFSPH) solver. |

#### `pinneapple_simulation.numerical_solvers.discrete_time_pinn`

Discrete-time physics-informed neural network (PINN), per Raissi, Perdikaris & Karniadakis, "Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations" (arXiv:1711.10566), Section 2 ("Discrete Time Models"), specialized here to the pure …

<small>`pinneapple_simulation/numerical_solvers/discrete_time_pinn.py` · 408 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DiscreteTimeRKNet` | classe | `(q, hidden_layers=(50, 50, 50, 50), activation='tanh')` | x -> (q+1) discrete-time IRK network (Raissi et al. Part II, Section 2). Column i<q is the i-th RK stage prediction U_i(x); column q is the final-time prediction U_{q+1}(x). Plain tanh MLP, matching the original paper's architecture choice. |
| `DiscreteTimeResult` | classe | `(model, A, b, c, q, dt, x_span, loss_history)` |  |
| `DiscreteTimePINNSolver` | classe | `(pde='burgers_1d', q=32, hidden_layers=(30, 30, 30), activation='tanh', n_boundary=20, bc…)` | Thin `SolverBase`/registry wrapper around `train_discrete_time`. The functional API (`train_discrete_time`, `DiscreteTimeRKNet`, `stage_derivatives`, `discrete_time_rk_residual`) is the primary entry point. |
| `stage_derivatives` | função | `(model, x)` | U(x), U_x(x), U_xx(x), each (N, q+1), via forward-mode AD. |
| `discrete_time_rk_residual` | função | `(model, x, u_n_data, dt, A, b, nonlinear_op) -> 'tuple[torch.Tensor, torch.Tensor]'` | Stage and final-time IRK consistency residuals (module docstring point 4). Returns (res_stage, res_final), shapes (N, q) and (N, 1). |
| `train_discrete_time` | função | `(pde, x_n, u_n, dt, q=32, x_span=None, hidden_layers=(30, 30, 30), activation='tanh', par…)` | Train a `DiscreteTimeRKNet` for one implicit-RK step of `pde` in {"burgers_1d", "allen_cahn_1d"}, from known data (x_n, u_n) at t_n to t_n + dt, via L-BFGS (module docstring point 5). |

#### `pinneapple_simulation.numerical_solvers.eddy_current_fdm`

2D axisymmetric magnetostatic / eddy-current FDM solver.

<small>`pinneapple_simulation/numerical_solvers/eddy_current_fdm.py` · 240 linhas · testes: 74</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EddyCurrentFDMSolver` | classe | `(nr=60, nz=120)` | Thin `SolverBase`/registry wrapper. The functional API (`solve_axisymmetric_eddy_current`, `build_axisymmetric_helmholtz_system`, `annular_current_source`) is the primary entry point and can be used directly without this wrapper. |
| `build_axisymmetric_helmholtz_system` | função | `(r, z, k2, rhs_2d)` | Assemble the sparse complex system M @ a = b for the axisymmetric operator |
| `annular_current_source` | função | `(r, z, r_center, z_center, width_z, width_r, n_turns, current_amp) -> np.ndarray` | Uniform azimuthal current density J_theta (A/m^2) in a rectangular annular band centered at (r_center, z_center) -- the standard lumped-coil approximation for an axisymmetric current loop / solenoid turn, used to drive the eddy-current or magnetostatic equation above. The band is ``\|z - z_center\| … |
| `solve_axisymmetric_eddy_current` | função | `(r, z, omega, mu, sigma, source_current_density) -> np.ndarray` | Solve for the azimuthal vector potential A_theta(r,z) at angular frequency `omega` (rad/s), given material maps mu(r,z) [H/m] and sigma(r,z) [S/m] and a source current density J_source(r,z) [A/m^2], all as (len(r), len(z)) arrays matching the r,z grids. `omega=0` solves the magnetostatic limit (no … |
| `eddy_current_density` | função | `(A, omega, sigma) -> np.ndarray` | Induced eddy-current density J_theta = -i*omega*sigma*A, given the solved vector potential A and the same sigma(r,z) map used to solve it. |
| `axial_flux_density` | função | `(A, r, z) -> np.ndarray` | B_z(r,z) = (1/r) d(r*A)/dr = A/r + dA/dr, computed via central differences (the axisymmetric curl relating A_theta to the axial field component). Complex A gives the complex phasor B_z (in-phase and quadrature parts); A = B0 r / 2 gives B_z = B0, on the axis too. |

#### `pinneapple_simulation.numerical_solvers.eemd`

EEMD: Ensemble Empirical Mode Decomposition.

<small>`pinneapple_simulation/numerical_solvers/eemd.py` · 167 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EEMDSolver` | classe | `(n_ensembles=50, noise_std=0.2, max_imfs=10, seed=None)` |  |
| `emd_1d` | função | `(x, *, max_imfs=10, max_sift=50, stop_thresh=0.05) -> List[np.ndarray]` | Very small EMD implementation for 1D signals. |
| `eemd_1d` | função | `(x, *, n_ensembles=50, noise_std=0.2, max_imfs=10, seed=None) -> np.ndarray` | EEMD for 1D: returns imfs array (K, T). |

#### `pinneapple_simulation.numerical_solvers.elasticity3d_fdm`

Static 3D linear (isotropic, small-strain) elasticity via finite differences: solves the Navier-Cauchy equilibrium equations

<small>`pinneapple_simulation/numerical_solvers/elasticity3d_fdm.py` · 182 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): placeholder/stub citado no docstring</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Elasticity3DFDMSolver` | classe | `(n_iter=20000, omega=1.0, tol=1e-10)` | Thin `SolverBase`/registry wrapper. The functional API (`solve_static_elasticity_3d`, `lame_parameters`) is the primary entry point and can be used directly without this wrapper. |
| `lame_parameters` | função | `(E, nu) -> Tuple[float, float]` | Lame's first parameter (lambda) and shear modulus (mu) from Young's modulus E and Poisson's ratio nu. |
| `solve_static_elasticity_3d` | função | `(lam, mu, body_force, u_bc, dx, dy, dz, n_iter=20000, omega=1.0, tol=1e-10) -> dict` | Solve the static Navier-Cauchy equations on a structured grid via damped-Jacobi relaxation. |

#### `pinneapple_simulation.numerical_solvers.etdrk4`

ETDRK4: fourth-order exponential time differencing Runge-Kutta for stiff semilinear problems ``du/dt = L u + N(u, t)`` with a *diagonal* linear part.

<small>`pinneapple_simulation/numerical_solvers/etdrk4.py` · 151 linhas · testes: 17</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SimulationBlowUp` | classe | `(message, *, step, t)` | The state became non-finite or exceeded the magnitude limit. |
| `ETDRK4Result` | classe | `(t, u, steps, blew_up, meta)` |  |
| `ETDRK4` | classe | `(linear, nonlinear, dt, *, contour_points=32, radius=1.0)` | Fixed-step ETDRK4 integrator. |
| `check_finite` | função | `(u, *, limit=1e+37, step=-1, t=float('nan')) -> None` | Raise :class:`SimulationBlowUp` if ``u`` has a NaN/inf or ``\|u\| > limit``. |
| `fourier_wavenumbers` | função | `(n, length=2 * np.pi, *, real=False) -> np.ndarray` | Angular wavenumbers ``k`` of an ``n``-point periodic grid of the given length, in FFT order (``np.fft.rfft`` order when ``real``). |

#### `pinneapple_simulation.numerical_solvers.external_flow`

External flow around any body in OpenFOAM, in one call: STL -> snappyHexMesh -> simpleFoam (k-omega SST, wall functions) -> forces, skin pressure and friction, streamlines and slice planes -> a 3D scene to render or browse.

<small>`pinneapple_simulation/numerical_solvers/external_flow.py` · 728 linhas · testes: 40</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CaseFields` | classe | `(case, time=None)` | The reconstructed mesh and the latest U, p of a case, with the helpers the results need: wall forces and skin coefficients on given patches, a velocity interpolator, streamline tracing, plane sampling. |
| `FlowResult` | classe | `(coefficients, surface, bodies, lines, line_speed, slices, info, geometry)` |  |
| `ExternalFlow` | classe | `(bodies, speed, alpha, beta, nu, rho, ref_area, ref_length, half_model, ground, ground_mo…)` | Steady incompressible flow around ``bodies`` (dict name -> (vertices, faces), an STL path, or a studio Scene). |
| `flow_directions` | função | `(alpha_deg, beta_deg=0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]` | Unit drag (along the flow), lift (up, normal to it in the x-z plane) and side directions. |
| `write_flow_fields` | função | `(case, speed, alpha_deg, beta_deg, nu, iterations, procs, walls, symmetry, ground=None, g…)` | 0/ fields, physical properties, controlDict, schemes and solvers of a steady external-flow case. ``walls`` is the patch group of the bodies; ``ground``: name of a road patch (moving with ``ground_velocity``); ``turbulence``: kOmegaSST (wall functions) or laminar (no k, omega, nut fields). |
| `snappy_dict` | função | `(geometry, features, surfaces, regions, location) -> str` | snappyHexMeshDict body (castellate + snap, no layers: wall functions). |
| `run_case` | função | `(case, procs=4, log=print, all_times=False) -> Dict[str, float]` | blockMesh, surfaceFeatureExtract, snappyHexMesh, checkMesh and simpleFoam in parallel, then reconstruct (the latest time, or every written time with ``all_times``). Returns the seconds per step; each step's log is in ``case/log.<step>``. |
| `latest_time` | função | `(case) -> str` |  |
| `openfoam_available` | função | `() -> bool` | True when the OpenFOAM tools this module calls can be found. |

#### `pinneapple_simulation.numerical_solvers.fdm`

Finite Difference Method (FDM) — problem-agnostic numerical solver.

<small>`pinneapple_simulation/numerical_solvers/fdm.py` · 919 linhas · testes: 3</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FDMSolver` | classe | `(nx=64, ny=64, nt=500, iters=8000, omega=1.5, theta=0.5, tol=1e-09)` | Problem-agnostic FDM solver. |

#### `pinneapple_simulation.numerical_solvers.fdm3d`

3D Finite Difference Method (FDM) solvers.

<small>`pinneapple_simulation/numerical_solvers/fdm3d.py` · 911 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SolverOutput3D` | classe | `(u, coords, meta)` | Output from a 3D FDM solver. |
| `HeatConfig3D` | classe | `(nx, ny, nz, nt, lx, ly, lz, dt, alpha, ic_fn, source_fn)` | Configuration for 3D heat conduction solver. |
| `HeatConduction3D` | classe | `(cfg=None)` | 3D heat equation solver using explicit Euler + central differences. |
| `NavierStokesConfig3D` | classe | `(nx, ny, nz, nt, lx, ly, lz, dt, nu, rho, pressure_iters, ic_fn, force_fn)` | Configuration for 3D incompressible Navier-Stokes solver. |
| `NavierStokes3D` | classe | `(cfg=None)` | 3D incompressible Navier-Stokes via fractional-step (projection) method. |
| `ElasticWaveConfig3D` | classe | `(nx, ny, nz, nt, lx, ly, lz, dt, c, ic_u_fn, ic_du_fn, source_fn)` | Configuration for 3D elastic wave solver. |
| `ElasticWave3D` | classe | `(cfg=None)` | 3D elastic (acoustic) wave equation solver via Störmer-Verlet integration. |
| `LidDrivenCavityConfig3D` | classe | `(nx, ny, nz, nt, size, dt, nu, rho, lid_velocity, lid_axis, pressure_iters)` | Configuration for the 3D lid-driven cavity solver. |
| `LidDrivenCavitySolver3D` | classe | `(cfg=None)` | 3D Lid-Driven Cavity flow solver. |
| `ChannelFlowConfig3D` | classe | `(nx, ny, nz, nt, lx, ly, lz, dt, nu, rho, dp_dx, pressure_iters)` | Configuration for the 3D pressure-driven channel flow solver. |
| `ChannelFlowSolver3D` | classe | `(cfg=None)` | 3D pressure-driven channel (Poiseuille-like) flow solver. |

#### `pinneapple_simulation.numerical_solvers.fem`

Finite Element Method — problem-agnostic P1/Q1 solver.

<small>`pinneapple_simulation/numerical_solvers/fem.py` · 368 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FEMSolver` | classe | `(nx=32, ny=32, solver='cg', tol=1e-08, max_iter=4000)` | Problem-agnostic FEM solver. |

#### `pinneapple_simulation.numerical_solvers.fenics_bridge`

FEniCS/dolfinx bridge for pinneapple.

<small>`pinneapple_simulation/numerical_solvers/fenics_bridge.py` · 647 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FEnicsBridge` | classe | `(mesh_nx=32, mesh_ny=32, element_degree=1, solver_backend='dolfinx') -> None` | Bridge that formulates and solves FEM problems using FEniCS/dolfinx. |

#### `pinneapple_simulation.numerical_solvers.fft`

FFT solver for spectrum computation with rFFT/FFT support.

<small>`pinneapple_simulation/numerical_solvers/fft.py` · 85 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FFTSolver` | classe | `(*, dims=None, real_input=True, output='magnitude_phase', norm=None)` | FFT Solver (PyTorch backend). |

#### `pinneapple_simulation.numerical_solvers.free_surface`

Free-surface flow of a liquid and a gas in OpenFOAM (interFoam, volume of fluid), in one object: an initial liquid level, inlets with a velocity vector, open boundaries, k-omega SST, and turbulence sampled on lines and averaged in time. Built for melt runners, launders, spillways and channels.

<small>`pinneapple_simulation/numerical_solvers/free_surface.py` · 626 linhas · testes: 12</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Phase` | classe | `(name, rho, nu)` |  |
| `SampleLine` | classe | `(name, start, end, n)` |  |
| `FreeSurfaceFlow` | classe | `(regions, roles, mesh, inlet_velocity, level, liquid, gas, sigma, gravity, p_ref, turbule…)` |  |
| `LiveLineSampler` | classe | `(case, lines, fields, keep_every=None, end_time=float('inf'), out_name='lines', poll_s=3.…)` | Samples lines from a running case as each time is written, then optionally deletes that time. |
| `split_surface` | função | `(V, F, boxes, rest='walls') -> dict[str, tuple[np.ndarray, np.ndarray]]` | Split one closed surface into named regions: a triangle goes to the first box that holds its centroid, otherwise to ``rest``. Feed the result to ``FreeSurfaceFlow(regions=...)``. |
| `cell_centres` | função | `(poly_dir) -> np.ndarray` | Cell centres (mean of face centres) from an ASCII constant/polyMesh. |
| `write_profile` | função | `(case, tname, idx, res, fields, out_name='lines') -> None` |  |
| `sample_lines` | função | `(case, lines, fields) -> int` | Sample every written time of a finished case (serial or decomposed) onto ``lines``. Returns the count. |
| `read_profiles` | função | `(case, names=None, t_start=0.0, t_end=float('inf'), out_name='lines') -> dict` | {line: {"s": distance along the line, "xyz", "times", <field>: (n_times, n_points), "mean_<field>"}}. |
| `time_average` | função | `(profile, field, window=None, origin='min') -> tuple[np.ndarray, np.ndarray]` | Time-mean of ``field`` along a line against the distance from the line's first sampled point (``origin="min"``), optionally cut to ``window`` = (s0, s1). |

#### `pinneapple_simulation.numerical_solvers.fvm`

Finite Volume Method — problem-agnostic cell-centered structured solver.

<small>`pinneapple_simulation/numerical_solvers/fvm.py` · 345 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FVMSolver` | classe | `(nx=64, ny=64, nt=200, dt=None, safety=0.4)` | Problem-agnostic FVM solver on a structured cell-centred 2-D grid. |

#### `pinneapple_simulation.numerical_solvers.gated_xtfc`

Gated X-TFC: X-TFC with Gaussian RBF features whose centers and widths follow a soft domain split, for 1D second-order linear boundary value problems with sharp layers (forward and inverse).

<small>`pinneapple_simulation/numerical_solvers/gated_xtfc.py` · 309 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LinearBVP1D` | classe | `(a, b, c, f, u_left, u_right, nu, exact)` | ``a u'' + b u' + c u = f`` on [0, 1], ``u(0) = u_left``, ``u(1) = u_right``. |
| `GatedLayout` | classe | `(centers, widths, collocation, splits, eps_b)` |  |
| `GatedSolution` | classe | `(problem, layout, coef, solve_seconds)` |  |
| `InverseFit` | classe | `(nu, splits, eps_scales, log_evidence, eta, solution, history, seconds)` |  |
| `convection_diffusion` | função | `(nu) -> LinearBVP1D` | ``u' - nu u'' = 0``, ``u(0) = 0``, ``u(1) = 1``; ``u = (e^{x/nu} - 1)/(e^{1/nu} - 1)`` (paper eq. 1-2). |
| `twin_boundary_layer` | função | `(nu) -> LinearBVP1D` | ``2(2x - 1) u' - nu u'' + 4u = 0``, ``u(0) = u(1) = 1``; ``u = exp(-2x(1 - x)/nu)`` (paper Sec. 4.4). |
| `gated_layout` | função | `(splits, eps_scales, nu, n_centers, n_colloc=None, k=1.5) -> GatedLayout` | Centers, widths and collocation points of the soft split (per block ``n_centers`` centers and ``n_colloc`` collocation points). With one split this is the paper's two-block layout; with several, each split blends the widths of its two neighbouring blocks by its own logistic gate. |
| `solve_forward` | função | `(problem, layout, lam=1e-12) -> GatedSolution` | ``argmin (1/N) \|\|R c + r0\|\|^2 + lam \|\|c\|\|^2`` at the layout's collocation points (stacked least squares). |
| `validation_energy` | função | `(sol, x_val) -> float` | Mean squared strong-form residual on the validation points. |
| `search_gates_nested` | função | `(problem, x_val, *, n_centers, n_colloc=None, k=1.5, lam=1e-12, xs_bounds=(0.8, 0.999), e…)` | One split: outer bounded Brent search over ``x_s``, inner over ``eps_scale``, minimising the validation residual energy (the paper's nested ``fminbnd``). Returns the best solution and the search record. |
| `log_evidence` | função | `(Phi, y, *, eta0=1e-07, eta_bounds=(1e-12, 0.01), max_iter=50, damping=0.5) -> dict` | Bayesian linear model ``y = Phi c + e``, ``e ~ N(0, I)`` (whitened), ``c ~ N(0, eta^-1 I)``: the prior precision ``eta`` by the evidence fixed point ``gamma = sum k/(k + eta)``, ``eta <- gamma / \|\|m\|\|^2`` (damped in log space, clipped), then the log-evidence, posterior mean and the … |
| `fit_inverse` | função | `(make_problem, x_data, y_data, *, sigma, beta_pde, n_centers, n_colloc, k=1.5, log10_nu_b…)` | Infer ``nu`` (and the gate) by maximising the evidence of the stacked whitened data + physics system over ``(log10 nu, x_s, eps_scale)`` with the library's Bayesian optimiser (GP + expected improvement). |
| `search_gates_bo` | função | `(problem, x_val, *, split_bounds, eps_bounds=(10.0, 100.0), n_centers, n_colloc=None, k=1…)` | Several splits: minimise the validation residual energy (log10) over all splits and their eps_scales with the library's Bayesian optimiser (the paper's choice for the multi-split twin boundary layer). |

#### `pinneapple_simulation.numerical_solvers.heat_fv_reference`

Independent reference solutions for the 2D and 3D cases: cell-centred finite volumes, sparse direct solve.

<small>`pinneapple_simulation/numerical_solvers/heat_fv_reference.py` · 123 linhas · testes: 3</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `footprint` | função | `(x, y, xc, yc, half, width)` | Smooth indicator of a square device footprint (1 inside, 0 outside, tanh edges of ``width``). |
| `footprint_area` | função | `(xc, yc, half, width, lx, ly, n=2000) -> float` | Area (m2) of the smooth ``footprint`` over the plate [0, lx] x [0, ly] (all lengths in m). |
| `plate_2d` | função | `(h, *, lx, ly, t, k, power, t_air, device, nx=200, ny=120)` | Temperature (°C) at the cell centres of an (ny, nx) grid; returns (x, y, T). |
| `block_3d` | função | `(h, *, lx, ly, lz, k, power, t_air, device, nx=40, ny=40, nz=12)` | Temperature (°C) at the cell centres of an (nz, ny, nx) grid; returns (x, y, z, T). |
| `interp_grid` | função | `(xs, ys, field, px, py)` | Bilinear interpolation of a cell-centred 2D field at points (px, py), clamped to the grid. |
| `convergence` | função | `(fn, coarse, fine, **kw) -> float` | Largest temperature difference (K) between two grids, compared at the coarse cell centres. |

#### `pinneapple_simulation.numerical_solvers.hilbert_huang`

Hilbert-Huang transform (EMD + Hilbert) solver for time series.

<small>`pinneapple_simulation/numerical_solvers/hilbert_huang.py` · 225 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HilbertHuangSolver` | classe | `(*, max_imfs=8, max_sift=50, stop_mean_tol=0.001, dt=1.0)` | Hilbert–Huang Transform (HHT) MVP for 1D signals. |

#### `pinneapple_simulation.numerical_solvers.immersed_boundary_fdm`

Immersed-boundary-method (IBM) finite-difference solver for incompressible Navier-Stokes on an arbitrary 3D geometry.

<small>`pinneapple_simulation/numerical_solvers/immersed_boundary_fdm.py` · 964 linhas · testes: 2</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ImmersedBoundaryFDMSolver` | classe | `(flow_type='internal', **solver_kwargs)` | Thin `SolverBase`/registry wrapper. The functional API (`solve_ibm_internal_flow`, `solve_ibm_external_flow`, `solve_ibm_navier_stokes`) is the primary entry point and can be used directly without this wrapper. |
| `field_divergence` | função | `(u, v, w, dx, dy, dz) -> np.ndarray` | Central-difference divergence of a velocity field -- exposed standalone for verifying the near-zero-divergence property of a projected field. |
| `interpolate_to_points` | função | `(grid_axes, field, query_points, fill_value=0.0) -> np.ndarray` | Trilinear interpolation of a (nx,ny,nz) grid field onto arbitrary query points, via `scipy.interpolate.RegularGridInterpolator`. |
| `solve_ibm_internal_flow` | função | `(wall_points, bbox, *, Re=100.0, U_in=1.0, rho=1.0, nx=32, ny=32, nz=32, n_iter=500, n_fr…)` | IBM/FDM incompressible Navier-Stokes on an enclosed domain whose bounding box IS the fluid domain. |
| `solve_ibm_external_flow` | função | `(wall_points, bbox, *, Re=100.0, U_in=1.0, rho=1200.0, nx=32, ny=32, nz=32, n_iter=1000, …)` | IBM/FDM incompressible Navier-Stokes for external (free-stream) flow around an arbitrary solid immersed body. |
| `solve_ibm_navier_stokes` | função | `(wall_points, bbox, flow_type='internal', **kwargs) -> Dict[str, Any]` | Dispatch to `solve_ibm_internal_flow` (`flow_type="internal"`) or `solve_ibm_external_flow` (`flow_type="external"`). All other keyword arguments are forwarded to the selected solver. |

#### `pinneapple_simulation.numerical_solvers.internal_flow`

Internal flow in pipes, bends and static mixers in OpenFOAM, in one call.

<small>`pinneapple_simulation/numerical_solvers/internal_flow.py` · 817 linhas · testes: 35</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Route` | classe | `(D, segments)` | A pipe centreline: straight runs and circular bends, lengths and radii in diameters. Starts at the origin along +x; bends turn in the plane given by ``plane_deg`` (0: towards +y, 90: towards +z). |
| `InternalResult` | classe | `(route, info, sections, profiles, slices, lines, line_speed, wall, timings)` |  |
| `InternalFlow` | classe | `(route, Re, nu, turbulent, mixer, mixer_start_D, scalar, inlet, n_core, n_radial, wall_gr…)` | A pipe ``route`` at Reynolds number ``Re`` (bulk velocity and diameter). ``turbulent``: None picks laminar below Re 2300. ``mixer``: number of Kenics elements starting at ``mixer_start_D`` (straight routes). ``scalar``: carry a passive scalar entering on the upper half of the inlet (c = 1 where the … |
| `kenics_elements` | função | `(route_D, n_elements, start_D, length_D=1.5, thickness_D=0.06, width_D=0.98, nu=10, ns=48…)` | Kenics static-mixer elements in a straight pipe along +x starting at ``start_D``: plates twisted by 180 deg, alternately left- and right-handed, each turned 90 deg from the previous one. A closed triangulated surface. |
| `oblock_mesh_dict` | função | `(route, n_core=8, n_radial=10, wall_grading=0.25, axial_cell_D=0.2, core=0.36, convert=1.…)` | blockMeshDict of the O-grid pipe swept along ``route``: five blocks per station interval (a square core of half-diagonal ``core`` x radius, four curved blocks), arcs on the wall, patches inlet / outlet / wall. ``wall_grading``: wall cell / core-side cell in the radial direction. |
| `pipe_surface` | função | `(st, r, nth=64, sub=4) -> tuple[np.ndarray, np.ndarray]` | Triangulated wall of a swept circular section (stations from ``Route.stations``), subdividing straight runs. |
| `ito_bend_loss` | função | `(Re, radius_ratio, angle_deg=90.0) -> float` | Total loss coefficient of a smooth pipe bend in turbulent flow (Ito 1960, J. Basic Eng. 82: 131-143), with ``radius_ratio`` = bend radius / pipe radius, valid for Re (r/R)^2 > 91; includes friction over the bend and the excess loss recovered downstream. |
| `colebrook` | função | `(Re, rel_roughness=0.0) -> float` | Darcy friction factor of fully developed turbulent pipe flow (Colebrook-White): :func:`pinneapple_physics.closed_form.pipe_flow.colebrook`. |

#### `pinneapple_simulation.numerical_solvers.irk_gauss_legendre`

Butcher tableaus for the q-stage Gauss-Legendre implicit Runge-Kutta (IRK) family -- a general-purpose, arbitrarily-high-order (order 2q), A-stable and B-stable time integrator for any ODE-IVP system dy/dt = f(t,y), stiff or non-stiff. Pure NumPy/SciPy, no model/framework dependency, so it can be …

<small>`pinneapple_simulation/numerical_solvers/irk_gauss_legendre.py` · 236 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_gauss_legendre_tableau` | função | `(q) -> Tuple[np.ndarray, np.ndarray, np.ndarray]` | Build the q-stage Gauss-Legendre IRK tableau (A, b, c) on [0,1]. |
| `validate_tableau` | função | `(A, b, c, q, tol=1e-09) -> dict` | Hard-assert the order conditions that certify a genuine 2q-order collocation method (C(q) and B(2q) jointly). Raises AssertionError with diagnostic residuals on failure; returns a dict of residuals on success. |
| `get_irk_tableau` | função | `(q, cache_dir=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]` | Build-or-load-cached, then validate, the q-stage Gauss-Legendre IRK tableau. Raises loudly (AssertionError/RuntimeError) rather than returning a silently-wrong tableau. |
| `step_irk` | função | `(rhs_fn, t, y, h, A, b, c, jac_fn=None, newton_tol=1e-10, newton_max_iter=25) -> np.ndarr…)` | Advance y(t) -> y(t+h) by one fully-implicit Runge-Kutta step using tableau (A, b, c) (e.g. from `get_irk_tableau`). Solves the q coupled stage equations |

#### `pinneapple_simulation.numerical_solvers.isph`

ISPH: Incompressible SPH via pressure-projection (Cummins & Rudman 1999).

<small>`pinneapple_simulation/numerical_solvers/isph.py` · 264 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ISPHSolver` | classe | `(h=0.04, mass=1.0, rho0=1000.0, mu=0.02, gravity=None, boundary_lo=None, boundary_hi=None…)` | Incompressible SPH (ISPH) using prediction-correction pressure projection. |

#### `pinneapple_simulation.numerical_solvers.jax_cfd_solver`

JAX-CFD adapter — real differentiable Navier-Stokes via Google's jax-cfd.

<small>`pinneapple_simulation/numerical_solvers/jax_cfd_solver.py` · 358 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `JAXCFDSolver` | classe | `(nx=64, ny=64, viscosity=0.01, density=1.0, domain=None, max_velocity=1.0, cfl_safety=0.5…)` | 2-D incompressible Navier-Stokes solver backed by ``jax_cfd.base``. |
| `jax_cfd_available` | função | `() -> bool` | Return ``True`` if both ``jax`` and ``jax_cfd`` can be imported. |
| `taylor_green_velocity_fns` | função | `() -> Tuple[Callable, Callable]` | 2-D Taylor-Green vortex velocity functions ``u(x,y), v(x,y)``. |
| `shear_layer_velocity_fns` | função | `(sigma=15.0, width=0.05) -> Tuple[Callable, Callable]` | 2-D double shear-layer velocity functions ``u(x,y), v(x,y)``. |

#### `pinneapple_simulation.numerical_solvers.kuramoto_sivashinsky`

Kuramoto-Sivashinsky in one dimension, ``u_t + (u^2 / 2)_x + u_xx + u_xxxx = 0`` on a periodic interval, with the ETDRK4 scheme of :mod:`.etdrk4` (Kassam & Trefethen 2005 set-up, 2/3 dealiasing).

<small>`pinneapple_simulation/numerical_solvers/kuramoto_sivashinsky.py` · 60 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `KuramotoSivashinsky1D` | classe | `(n, length, dt)` |  |

#### `pinneapple_simulation.numerical_solvers.lbm`

Lattice Boltzmann Method — D2Q9 BGK + D3Q19 BGK.

<small>`pinneapple_simulation/numerical_solvers/lbm.py` · 739 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LBMSolver` | classe | `(nx=128, ny=64, Re=100.0, u_in=0.05, rho_out=1.0, obstacle_mask=None, Cs=0.0)` | D2Q9 Lattice Boltzmann solver. |
| `LBMSolver3D` | classe | `(nx=32, ny=32, nz=32, Re=100.0, u_in=0.05, obstacle_mask=None, Cs=0.0)` | D3Q19 Lattice Boltzmann solver. |
| `cylinder_mask` | função | `(nx, ny, cx, cy, r) -> torch.Tensor` | Create a circular obstacle mask (True = solid). cx, cy, r in lattice units. |
| `rectangle_mask` | função | `(nx, ny, x0, x1, y0, y1) -> torch.Tensor` | Rectangular obstacle mask. |
| `airfoil_naca_mask` | função | `(nx, ny, chord, aoa_deg=0.0, naca='0012') -> torch.Tensor` | Approximate NACA 4-digit airfoil mask centred in the domain. chord: chord length in lattice units. |
| `naca4_mask` | função | `(nx, ny, naca='4412', chord=80.0, aoa_deg=0.0, x_le=None, y_le=None, n_pts=400) -> torch.…)` | NACA 4-digit airfoil mask with camber (True = solid), aerodynamic convention. |
| `airfoil_naca_mask_3d` | função | `(nx, ny, nz, chord, aoa_deg=0.0, naca='0012', z0=0, z1=None) -> torch.Tensor` | 3-D obstacle mask: the 2-D NACA airfoil cross-section (airfoil_naca_mask) extruded uniformly along the z (span) axis from z0 to z1 (exclusive). |

#### `pinneapple_simulation.numerical_solvers.lbm_tunnel`

3-D lattice Boltzmann wind tunnel in PyTorch, with the memory and boundary techniques FluidX3D made popular.

<small>`pinneapple_simulation/numerical_solvers/lbm_tunnel.py` · 479 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Units` | classe | `(m, s, kg)` | Conversion between SI and lattice units, fixed by a length, a velocity and a density given in both systems (lattice length in cells, lattice velocity in cells/step, lattice density usually 1). |
| `WindTunnelLBM` | classe | `(shape, nu, u_in=(0.0, 0.0, 0.0), velocity_set_name='D3Q19', collision='srt', smagorinsky…)` | 3-D lattice Boltzmann flow in a box (lattice units: cell = 1, step = 1, rho0 = 1). |
| `velocity_set` | função | `(name='D3Q19')` | (c [q,3] int, w [q], opposite index [q]) for D3Q19 or D3Q27. |
| `sphere` | função | `(shape, center, radius) -> torch.Tensor` |  |
| `box` | função | `(shape, lo, hi) -> torch.Tensor` |  |
| `voxelize_stl` | função | `(mesh, shape, center, size, axis=0) -> torch.Tensor` | Solid mask of a closed surface (path or trimesh.Trimesh), scaled so its extent along ``axis`` is ``size`` cells and centred at ``center`` (cells). Inside test by ray parity (trimesh). |
| `write_vtk` | função | `(path, data, origin=(0.0, 0.0, 0.0), spacing=1.0) -> None` | Legacy VTK, BINARY big-endian STRUCTURED_POINTS. ``data`` is (nx, ny, nz) or (3, nx, ny, nz). |
| `read_vtk` | função | `(path) -> dict` | Legacy VTK structured points (binary big-endian or ASCII), as written by FluidX3D's VTK export and by :func:`write_vtk`. Returns {"data": (nx,ny,nz) or (k,nx,ny,nz), "origin", "spacing", "dimensions", "name"}. |
| `drag_sphere_schiller_naumann` | função | `(re) -> float` | Drag coefficient of a sphere, Schiller & Naumann (1933), valid up to Re ~ 800. |
| `stokes_sphere_field` | função | `(shape, center, radius, u_inf)` | Stokes flow past a fixed sphere (uniform stream u_inf), velocity (3, nx, ny, nz); zero inside. |

#### `pinneapple_simulation.numerical_solvers.meshfree`

Meshfree solvers: RBF collocation and Moving Least Squares (MLS).

<small>`pinneapple_simulation/numerical_solvers/meshfree.py` · 377 linhas · testes: 8 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RBFCollocationSolver` | classe | `(rbf='gaussian', eps=3.0, coeff=1.0, reg=1e-08)` | Meshfree RBF collocation solver (Kansa method). |
| `MLSApproximation` | classe | `(degree=1, support_radius=0.2, weight='quartic')` | Meshfree Moving Least Squares approximation. |

#### `pinneapple_simulation.numerical_solvers.nbody_rebound`

REBOUND N-body gravitational integrator.

<small>`pinneapple_simulation/numerical_solvers/nbody_rebound.py` · 270 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NBodySolver` | classe | `(bodies, integrator='ias15', G=1.0, dt=None, move_to_com=True)` | Thin, real wrapper around `rebound.Simulation`. |

#### `pinneapple_simulation.numerical_solvers.neighbors`

Neighbor search utilities (grid hashing) for particle methods.

<small>`pinneapple_simulation/numerical_solvers/neighbors.py` · 113 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NeighborList` | classe | `(j, indptr)` | Compressed neighbor list representation. |
| `build_neighbor_list` | função | `(pos, *, h, max_neighbors=64) -> NeighborList` | Build neighbor list for 2D/3D particles using uniform grid. |

#### `pinneapple_simulation.numerical_solvers.nonlinear_beam_fem`

Spectral-element FEM for a geometrically nonlinear (Von Karman) transient Euler-Bernoulli beam, time-integrated with the Newmark-beta method and a full Newton-Raphson iteration at every timestep to resolve the nonlinear residual.

<small>`pinneapple_simulation/numerical_solvers/nonlinear_beam_fem.py` · 843 linhas · testes: 2</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NonlinearBeamNewmarkFEM` | classe | `(num_elements, nodes_per_element, length, area, moment_of_inertia, E, density, damping_co…)` | Spectral-element / Newmark-beta engine for the transient geometrically nonlinear (Von Karman) Euler-Bernoulli beam described in the module docstring. Construct once per problem (mesh + BCs are fixed at construction), then call `run()` to march through time, or `natural_frequencies_hz()` for the … |
| `NonlinearBeamFEMSolver` | classe | `(num_elements=50, nodes_per_element=NODES_PER_ELEMENT)` | Thin `SolverBase`/registry wrapper. `solve_nonlinear_beam_transient` and `NonlinearBeamNewmarkFEM` are the primary entry points and can be used directly without this wrapper. |
| `gauss_lobatto_legendre_points` | função | `(n_points) -> np.ndarray` | Gauss-Lobatto-Legendre points on [-1, 1] for `n_points` nodes (endpoints included, interior points at the roots of P'_{n-1}). |
| `generate_1d_mesh` | função | `(num_elements, nodes_per_element, dofpn, length)` | Uniform 1D mesh of `num_elements` elements, each with `nodes_per_element` nodes placed at Gauss-Lobatto-Legendre points (so nodes_per_element=2 is a plain linear mesh; higher values place spectral interior nodes). |
| `gauss_legendre_quadrature` | função | `(n_points)` | Gauss-Legendre quadrature points and weights on [-1, 1]. |
| `lagrange_shape_functions_1d` | função | `(nodes_per_element, n_quad_points)` | Lagrange shape functions and their first derivative (built on GLL nodes), evaluated at `n_quad_points` Gauss-Legendre quadrature points. |
| `hermite_shape_functions_1d` | função | `(n_quad_points, h)` | Cubic Hermite shape functions (value + slope DOF at each of the two element endpoints) and their first/second derivatives, evaluated at `n_quad_points` Gauss-Legendre quadrature points on the element of length `h`. Fixed at two endpoints regardless of mesh order -- see `NODES_PER_ELEMENT`. |
| `neumann_boundary_time_series` | função | `(num_elements, nodes_per_element, dofpn, conn, boundary_n, timesteps, delta_t, f_drive)` | Pre-evaluate the Neumann ("N") boundary loads at every timestep, scattered onto the (element, local-dof) layout used during assembly. A load value is const + sin_amp*sin(2*pi*f_drive*t) + cos_amp*cos(2*pi*f_drive*t). |
| `dirichlet_boundary_values` | função | `(num_elements, nodes_per_element, dofpn, conn, boundary_d)` | Pre-scatter the Dirichlet ("D") boundary values onto the (element, local-dof) layout; np.nan marks an unconstrained (element, local-dof). |
| `apply_neumann_bc` | função | `(i, t, load_vec_elem, boundary_n_series, ratio)` | Add the pre-evaluated Neumann load (scaled by the current load-step ratio) into an element's residual/load vector. |
| `apply_dirichlet_bc` | função | `(i, newton_iter, stiff_mat_elem, load_vec_elem, boundary_d_values)` | Enforce Dirichlet constraints on one element's tangent matrix and residual by row/column elimination (constrained DOFs decouple from the rest of the system; their prescribed value is only injected on the first Newton iteration of a load step, since later iterations solve for the *correction* to an … |
| `cantilever_frequency_roots` | função | `(n) -> np.ndarray` | First `n` positive roots of cos(x)cosh(x) = -1, the transcendental frequency equation for a clamped-free (cantilever) Euler-Bernoulli beam -- solved by bracketed root-finding since no closed form exists. Useful as an independent closed-form check for `natural_frequencies_hz` when the caller's … |
| `solve_nonlinear_beam_transient` | função | `(boundary, L_m, A_m2, I_m4, E_Pa, rho_kg_m3, damping_coeff=0.0, axial_dist_load_N_per_m=0…)` | Solve the transient geometrically nonlinear (Von Karman) Euler- Bernoulli beam FEM problem described in the module docstring, for a caller-supplied `boundary` dict (Dirichlet + Neumann DOF specification -- see module docstring for its shape). `newmark_beta`/`newmark_gamma` use the standard … |
| `acceleration_from_displacement` | função | `(w, dt) -> np.ndarray` | Convenience/diagnostic: second time-derivative of a displacement time-history via a proper 3-point second-derivative stencil (reuses `beam_bvp_fdm.second_derivative`, which is exact for a quadratic displacement history and doesn't compound `np.gradient`'s first-order edge truncation the way … |

#### `pinneapple_simulation.numerical_solvers.ns3d_spectral`

Pseudo-spectral solver for the 3-D incompressible Navier-Stokes equations on the periodic box [0, 2*pi)^3 (rotational form, 2/3-rule dealiasing, classical RK4, exact integrating factor for viscosity).

<small>`pinneapple_simulation/numerical_solvers/ns3d_spectral.py` · 103 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SpectralNS3D` | classe | `(N, nu, dt)` |  |
| `taylor_green_ic` | função | `(N)` |  |

#### `pinneapple_simulation.numerical_solvers.openfoam_bridge`

OpenFOAM bridge for pinneapple.

<small>`pinneapple_simulation/numerical_solvers/openfoam_bridge.py` · 751 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OpenFOAMBridge` | classe | `(solver='simpleFoam', n_cores=1, n_iterations=500, openfoam_bin=None) -> None` | Bridge that generates OpenFOAM cases, runs the solver, and extracts fields. |
| `generate_case` | função | `(problem_spec, case_dir, mesh_cfg=None) -> Path` | Generate an OpenFOAM case directory from *problem_spec*. |
| `run_openfoam` | função | `(case_dir, solver='simpleFoam', n_cores=1, n_iter=500) -> 'subprocess.CompletedProcess[st…)` | Run an OpenFOAM solver inside *case_dir*. |
| `extract_fields` | função | `(case_dir, time='latestTime', field_names=None) -> Dict[str, 'Any']` | Extract field arrays from a finished OpenFOAM run. |
| `openfoam_to_dataset` | função | `(case_dir, field_names=None, time='latestTime') -> Dict[str, 'Any']` | High-level helper: extract fields and return as a flat dataset dict. |

#### `pinneapple_simulation.numerical_solvers.operator_benchmarks`

Reference solvers and input distributions of the standard operator-learning benchmarks.

<small>`pinneapple_simulation/numerical_solvers/operator_benchmarks.py` · 277 linhas · testes: 132</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FourierInterpolant` | classe | `(values, *, max_modes=None, dtype=None)` | Evaluate a periodic grid field (period 1 in every axis, nodes ``j / n``) at arbitrary points, with autograd. |
| `periodic_grf` | função | `(n_samples, shape, *, alpha, tau, sigma=None, seed=None) -> np.ndarray` | Zero-mean Gaussian random fields on the periodic unit box with covariance ``sigma^2 (-lap + tau^2)^-alpha``. |
| `threshold_coefficient` | função | `(field, high=12.0, low=3.0) -> np.ndarray` | Piecewise-constant coefficient ``high`` where ``field >= 0`` and ``low`` elsewhere (FNO Darcy inputs). |
| `spectral_downsample` | função | `(u, factor, axes) -> np.ndarray` | Point values on every ``factor``-th node of a periodic grid (exact for band-limited fields). |
| `burgers_1d` | função | `(u0, nu, t_end, *, dt=0.0001, n_save=1, dealias=True) -> tuple[np.ndarray, np.ndarray]` | Viscous Burgers on the periodic interval [0, 1). |
| `li_forcing` | função | `(n) -> np.ndarray` | ``f(x, y) = 0.1 (sin(2 pi (x + y)) + cos(2 pi (x + y)))`` on the ``n x n`` periodic grid (Li et al. 2021). |
| `navier_stokes_2d_vorticity` | função | `(w0, nu, t_end, *, forcing=None, dt=0.001, n_save=1, dealias=True) -> tuple[np.ndarray, n…)` | 2-D incompressible Navier-Stokes, vorticity form, periodic unit square, grid ``x_i = i / n``. |
| `advection_diffusion_2d` | função | `(u0, velocity, kappa, dt, n_steps) -> np.ndarray` | ``u_t + c . grad u = kappa lap u`` on the periodic unit square, exact in time for the resolved modes: each Fourier mode is multiplied by ``exp((-i k . c - kappa \|k\|^2) dt)`` per step. ``u0``: ``(n, n)`` or ``(batch, n, n)`` (axis 0 is x). Returns ``(n_steps + 1, ...)`` states, the first one ``u0``. |
| `darcy_2d` | função | `(a, f=1.0, *, axis_weights=(1.0, 1.0)) -> np.ndarray` | ``-div(a grad u) = f`` on the unit square, ``u = 0`` on the boundary; ``a`` (and ``f``) at the ``n x n`` cell centres ``((i + 1/2) h, (j + 1/2) h)``, ``h = 1/n``. Returns ``u`` at the cell centres. |
| `darcy_matrix` | função | `(a, axis_weights=(1.0, 1.0)) -> sp.csc_matrix` | The ``n^2 x n^2`` finite-volume matrix of :func:`darcy_2d` (``A u = f`` with ``u`` raveled in C order). It is symmetric positive definite, and ``A - min(a) * darcy_matrix(ones)`` is positive semi-definite (every face coefficient is a harmonic mean, never below ``min a``). |

#### `pinneapple_simulation.numerical_solvers.operator_datasets`

Operator-learning benchmark datasets: 1-D Burgers, 2-D Darcy, 2-D Navier-Stokes (vorticity).

<small>`pinneapple_simulation/numerical_solvers/operator_datasets.py` · 309 linhas · testes: 17</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `burgers_inputs` | função | `(n_samples, n, seed) -> np.ndarray` |  |
| `fourier_pad` | função | `(u, n_new, axes) -> np.ndarray` | Band-limited (trigonometric) interpolation of periodic grid data onto ``n_new`` points along each of ``axes``; exact for fields whose Fourier content is below the coarse Nyquist mode (the Nyquist mode is dropped). |
| `solve_burgers` | função | `(u0, dt=None) -> np.ndarray` |  |
| `darcy_inputs` | função | `(n_samples, n, seed, *, tau=None, high=None, low=None) -> np.ndarray` | Coefficients of the Darcy benchmark; ``tau`` (inverse correlation length), ``high`` and ``low`` override the dataset's values to draw shifted inputs (other length scale, other contrast) from the same family. |
| `block_average` | função | `(x, factor) -> np.ndarray` | Cell averages on a grid ``factor`` times coarser over the last two axes (``(..., n, n)``, ``n % factor == 0``): the coarse-grid version of a cell-averaged field (``u`` of the dataset) or of a coefficient. |
| `solve_darcy` | função | `(a, refine=None) -> np.ndarray` | Solve on the grid refined ``refine`` times (the piecewise-constant coefficient is represented exactly) and return the cell averages of u on the coefficient's own grid. |
| `ns_inputs` | função | `(n_samples, n, seed) -> np.ndarray` |  |
| `solve_ns` | função | `(w0, n_solve=None, dt=None) -> np.ndarray` | Solve on ``n_solve``^2 (inputs given on a coarser grid are Fourier-padded) and return the point values on the stored ``n_store``^2 grid at t = 0, 1, ..., t_end. |
| `rel_l2` | função | `(a, b) -> np.ndarray` | Per-sample relative L2 error of ``a`` against reference ``b`` (all axes but the first). |
| `write_npz` | função | `(path, arrays) -> None` | Deflated npz with fixed zip timestamps (the SHA-256 depends only on the data). |
| `darcy_set` | função | `(n_samples, seed, *, n=None, tau=None, high=None, low=None, workers=1) -> tuple[np.ndarra…)` | ``(a, u)`` pairs of the Darcy benchmark drawn and solved exactly as the zoo dataset ``darcy_2d_li2021`` (same solver, refinement and source), optionally from a shifted input distribution: ``tau`` (correlation length ~ 1 / tau; the dataset uses 3), ``high`` / ``low`` (the two coefficient values; 12 … |
| `generate` | função | `(out, workers=1, only=None) -> None` | Generate the three datasets (or ``only`` one) into ``out/<name>/`` (resumable for Navier-Stokes). |
| `convergence` | função | `(out, workers=1, n_samples=4, only=None) -> dict` | Resolution and time-step study on held-aside inputs (seed + 7): each setting is compared with a finer reference at the stored nodes, all saved times; the dataset setting must be converged to well below the error level operator-learning methods reach on these problems (FNO: 3e-3 to 1e-2 rel. L2). … |

#### `pinneapple_simulation.numerical_solvers.particles`

Lagrangian particles in a liquid: soft-sphere DEM contacts, drag, buoyancy and turbulent dispersion, in a prescribed carrier flow (one-way coupling). Fast enough for tens of thousands of particles on a laptop, and meant for process pictures and trends (suspension, mixing, settling), not for …

<small>`pinneapple_simulation/numerical_solvers/particles.py` · 284 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Liquid` | classe | `(rho, mu)` |  |
| `StirredTank` | classe | `(R, H, D_imp, C, swirl, loop, turbulence, T_L, blades, pitch_deg, down_pumping)` | Flat-bottom cylindrical tank (axis z, bottom at z = 0, liquid up to H) with a centred impeller of diameter ``D_imp`` at clearance ``C``. ``swirl`` and ``loop``: swirl and circulation speeds as fractions of the tip speed; ``turbulence``: velocity fluctuation as a fraction of the tip speed; ``T_L``: … |
| `DEM` | classe | `(x, d, rho_p, liq=WATER, e=0.4, t_c=None, dt=0.0005, g=9.81, seed=0)` | Equal spheres (diameter ``d``) with linear spring-dashpot normal contacts (restitution ``e``, collision time ``t_c``), drag towards the liquid velocity, buoyancy-reduced gravity, and contacts with the tank (wall, bottom, surface). Verlet neighbour lists with a skin. |
| `drag_factor` | função | `(Re) -> np.ndarray` | Schiller-Naumann correction to Stokes drag (Re < 1000). |
| `terminal_velocity` | função | `(d, rho_p, liq=WATER, g=9.81) -> float` | Settling velocity of an isolated sphere with Schiller-Naumann drag (fixed point). |
| `packed_bed` | função | `(tank, n, d, seed=0) -> np.ndarray` | Initial positions: a loose cubic lattice filling the tank from the bottom (it settles in the first second). |
| `suspend` | função | `(tank, n=20000, d=0.003, rho_p=1200.0, rpm=lambda t: min(150.0, 6.0 * t), t_end=30.0, dt=…)` | Suspension of an initially settled bed as the impeller speeds up. Returns frames (positions, speed), times, rpm, the fraction of particles in the top third, the fraction off the bottom and the largest contact overlap. |
| `zwietering_njs` | função | `(tank, d, rho_p, mass_ratio, liq=WATER, S=6.0, g=9.81) -> float` | Just-suspended impeller speed (rpm) of Zwietering (1958): N_js = S nu^0.1 (g drho/rho_L)^0.45 X^0.13 d^0.2 / D^0.85, X in mass percent of solids to liquid. ``S`` depends on the impeller and the geometry (about 4-8 for pitched-blade turbines). |

#### `pinneapple_simulation.numerical_solvers.pipe_network`

Steady flow in a looped pipe network (water distribution) by the global gradient algorithm.

<small>`pinneapple_simulation/numerical_solvers/pipe_network.py` · 184 linhas · testes: 27</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Network` | classe | `(junctions, reservoirs, pipes, headloss, coords)` | Nodes by name: ``elevation`` (m), ``demand`` (m3/s, positive = drawn off) for junctions, ``head`` (m) for fixed-head nodes. Pipes: (name, from, to, length m, diameter m, roughness) with roughness = Hazen-Williams C or Darcy-Weisbach absolute roughness in m, per ``headloss`` ("HW" or "DW"). |
| `solve_network` | função | `(net, tol=1e-10, max_iter=100) -> dict` | Heads, pressures (head - elevation), flows, velocities and head losses; ``iterations``, the continuity residual per junction and the energy residual per pipe (both checks of the converged solution). |
| `two_loop_network` | função | `(diameters_in=(18, 10, 16, 4, 16, 10, 10, 1), headloss='HW', roughness=None) -> Network` | The two-loop network of Alperovits & Shamir (1977), "Design of optimal water distribution systems", Water Resources Research 13(6): one reservoir at 210 m, six junctions, eight 1000 m pipes, Hazen-Williams C = 130. Default diameters (inches): the least-cost design reported by Savic & Walters (1997) … |
| `optimize_diameters` | função | `(net, sizes_m, unit_cost, p_min, *, pop=100, generations=400, seed=0, penalty=1000000.0, …)` | Least-cost discrete diameters with every junction pressure >= ``p_min`` (m): a genetic algorithm (integer genes, tournament selection, uniform crossover, creep mutation, elitism) with a penalty per metre of pressure deficit; every candidate is evaluated with :func:`solve_network`. Returns the best … |

#### `pinneapple_simulation.numerical_solvers.plane_stress_fem`

Linear plane-stress finite elements on any triangle mesh (constant-strain triangles), for parts whose geometry comes from elsewhere (a B-rep face, a CAD tessellation).

<small>`pinneapple_simulation/numerical_solvers/plane_stress_fem.py` · 137 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PlaneStressSolution` | classe | `(V, F, u, stress, area, reaction)` |  |
| `heywood_kt_net` | função | `(d_over_w) -> float` | Net-section stress concentration of a central hole in a strip under tension, ``2 + (1 - d/W)^3``. |
| `boundary_edges` | função | `(F) -> np.ndarray` | Edges ``(m, 2)`` that belong to one triangle only. |
| `solve_plane_stress` | função | `(V, F, *, E, nu, thickness, fixed, traction=(), body_force=(0.0, 0.0)) -> PlaneStressSolu…)` | ``V (n, 2)`` nodes [m], ``F (m, 3)`` triangles; ``fixed = {"x": nodes, "y": nodes}`` (zero displacement); ``traction``: ``[(edges (k, 2), (tx, ty) [Pa]), ...]`` on boundary edges; ``body_force`` [N/m^3]. |
| `patch_test` | função | `(V, F, *, E=1.0, nu=0.3) -> float` | Impose the displacement of a uniform stress state on the boundary nodes and return the largest relative deviation of the element stresses from it (exact elements give round-off). |

#### `pinneapple_simulation.numerical_solvers.problem_runner`

Generate PINN training datasets from ProblemSpec definitions.

<small>`pinneapple_simulation/numerical_solvers/problem_runner.py` · 570 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `generate_pinn_dataset` | função | `(problem_spec, *, n_col=None, n_bc=None, n_ic=None, n_data=0, seed=0, solver_cfg_override…)` | Generate a PINN training dataset from a ProblemSpec. |

#### `pinneapple_simulation.numerical_solvers.reaction_diffusion`

One-dimensional reaction-diffusion on a periodic interval, ``u_t = D u_xx + R(u)``, with the ETDRK4 scheme of :mod:`.etdrk4` (stiff diffusion exact, reaction explicit, 2/3 dealiasing).

<small>`pinneapple_simulation/numerical_solvers/reaction_diffusion.py` · 65 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ReactionDiffusion1D` | classe | `(reaction, diffusion, n, length, dt)` |  |

#### `pinneapple_simulation.numerical_solvers.registry`

Solver registry.

<small>`pinneapple_simulation/numerical_solvers/registry.py` · 137 linhas · testes: 206</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SolverSpec` | classe | `(name, family, cls, description, tags)` |  |
| `SolverRegistry` | classe | `(_REG)` |  |
| `SolverCatalog` | classe | `(registry)` | Convenience wrapper (dict-like catalog) around SolverRegistry. |
| `register_all` | função | `() -> Dict[str, str]` | Import solver modules so their @register decorators execute. |

#### `pinneapple_simulation.numerical_solvers.shallow_water_fv`

2D shallow-water finite-volume solver for free-surface water: walls, gates, virtual sensors, live health checks, CSV export and comparison with experiments.

<small>`pinneapple_simulation/numerical_solvers/shallow_water_fv.py` · 352 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Sensor` | classe | `(name, kind, x, y, z, faces, values)` |  |
| `ShallowWater2D` | classe | `(nx, ny, dx, dy=None, *, g=G, rho=RHO, cfl=0.45, second_order=True)` | Water on an ``nx x ny`` grid of cell size ``dx x dy`` (domain boundary: solid walls). |
| `ritter` | função | `(x, t, h0, x0=0.0, g=G)` | Dam break on a dry bed (Ritter 1892): depth and velocity. |
| `stoker` | função | `(x, t, h0, h1, x0=0.0, g=G)` | Dam break on a wet bed h1 < h0 (Stoker 1957): rarefaction + constant state + bore. |

#### `pinneapple_simulation.numerical_solvers.solid_fem`

3-D linear elasticity with 8-node hexahedra and incompatible modes (Wilson-Taylor; the element Abaqus calls C3D8I), so slender parts in bending do not lock. Structured box meshes, clamps, supports, nodal and pressure loads, nodal stresses (extrapolated from the Gauss points and averaged) and von …

<small>`pinneapple_simulation/numerical_solvers/solid_fem.py` · 415 linhas · testes: 47</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HexMesh` | classe | `(nodes, elements, shape, size)` |  |
| `FEMResult` | classe | `(mesh, u, stress, reactions, info)` |  |
| `SolidFEM` | classe | `(mesh, E, nu, incompatible, _fixed, _f)` |  |
| `box_mesh` | função | `(L, W, H, nx, ny, nz, origin=(0.0, 0.0, 0.0)) -> HexMesh` | Structured hexahedral mesh of a box [0, L] x [-W/2, W/2] x [-H/2, H/2] (shifted by ``origin``). |
| `elasticity_matrix` | função | `(E, nu) -> np.ndarray` |  |
| `von_mises` | função | `(s) -> np.ndarray` | von Mises stress of Voigt stresses (..., 6). |
| `fea_figure` | função | `(res, field='von_mises', scale=None, bands=12, title='', unit='MPa', factor=1e-06, view=(…)` | The deformed boundary mesh coloured by ``field`` in discrete bands with black element edges and a post-processor legend (S, Mises), the undeformed outline in grey. ``scale``: deformation scale factor (default: the largest displacement drawn as 8 % of the part's length). Any nodal ``values`` (a … |

#### `pinneapple_simulation.numerical_solvers.spectral`

Spectral solvers (MVP).

<small>`pinneapple_simulation/numerical_solvers/spectral.py` · 49 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SpectralSolver` | classe | `()` |  |
| `poisson_periodic_fft` | função | `(f, *, Lx=1.0, Ly=1.0, eps=1e-12) -> torch.Tensor` | Solve ∇²u = f with periodic BC on 2D domain via FFT. |

#### `pinneapple_simulation.numerical_solvers.sph`

SPH: Weakly-compressible Smoothed Particle Hydrodynamics (WCSPH) MVP.

<small>`pinneapple_simulation/numerical_solvers/sph.py` · 167 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SPHState` | classe | `(pos, vel, rho, p)` |  |
| `SPHSolver` | classe | `(h=0.04, mass=1.0, rho0=1000.0, c0=20.0, mu=0.02, gravity=None, boundary_lo=None, boundar…)` |  |

#### `pinneapple_simulation.numerical_solvers.sph_boundaries`

Boundary handling helpers for SPH (MVP).

<small>`pinneapple_simulation/numerical_solvers/sph_boundaries.py` · 88 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BoxBoundary` | classe | `(lo, hi)` | Axis-aligned box boundary [min,max] in each dimension. |
| `wall_repulsion_force` | função | `(pos, vel, boundary, *, k=50.0, d0=0.02) -> torch.Tensor` |  |
| `reflect_box` | função | `(pos, vel, lo, hi, *, restitution=0.0) -> Tuple[torch.Tensor, torch.Tensor]` | Hard boundary reflection for an axis-aligned box. |

#### `pinneapple_simulation.numerical_solvers.ssa`

SSA: Singular Spectrum Analysis for 1D time series.

<small>`pinneapple_simulation/numerical_solvers/ssa.py` · 73 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SSASolver` | classe | `(L=64, r=10)` |  |
| `ssa_decompose` | função | `(x, L=64, r=10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]` | Return reconstructed components (r,T), singular values, and U. |

#### `pinneapple_simulation.numerical_solvers.sst`

SST: Synchrosqueezed transform (MVP).

<small>`pinneapple_simulation/numerical_solvers/sst.py` · 89 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SSTSolver` | classe | `(fs=1.0, nperseg=128, noverlap=None, nfft=None)` |  |
| `sst_stft_1d` | função | `(x, *, fs=1.0, nperseg=128, noverlap=None, nfft=None, eps=1e-08) -> Tuple[np.ndarray, np.…)` | Return (SST_like, freqs, times). |

#### `pinneapple_simulation.numerical_solvers.stable_fluids`

Stable Fluids: unconditionally stable 2-D incompressible flow on a periodic grid.

<small>`pinneapple_simulation/numerical_solvers/stable_fluids.py` · 198 linhas · testes: 17</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `StableFluidsState` | classe | `(velocity, scalars, t)` |  |
| `StableFluids2D` | classe | `(nx, ny, *, length=(2 * np.pi, 2 * np.pi), viscosity=0.0, confinement=0.0, scalar_dissipa…)` | Periodic box ``[0, Lx) x [0, Ly)`` with ``nx x ny`` cell-centred samples. |
| `stable_time_step` | função | `(max_velocity, dx, *, courant=0.5, viscosity=0.0, ndim=2, implicit_diffusion=True) -> flo…)` | Advective CFL step ``courant * dx / max_velocity``, capped by the explicit-diffusion limit ``dx^2 / (2 ndim nu)`` unless diffusion is implicit (as in :class:`StableFluids2D`). |

#### `pinneapple_simulation.numerical_solvers.stiff_kinetics_ode`

Stiff-ODE convenience wrapper for coupled reaction-network / chemical- or biological-kinetics systems: dy_i/dt = rate_fn(t, y)[i], y_i(t0) = y0[i].

<small>`pinneapple_simulation/numerical_solvers/stiff_kinetics_ode.py` · 100 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `solve_stiff_reaction_network` | função | `(rate_fn, y0, t_span, t_eval=None, jac_fn=None, method='Radau', rtol=1e-09, atol=1e-12, c…)` | Integrate dy/dt = rate_fn(t, y), y(t0) = y0, over t_span = (t0, tf) with a stiff-appropriate implicit method. |

#### `pinneapple_simulation.numerical_solvers.stl`

STL: Seasonal-Trend decomposition using Loess.

<small>`pinneapple_simulation/numerical_solvers/stl.py` · 40 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `STLSolver` | classe | `(period=24, robust=True)` |  |

#### `pinneapple_simulation.numerical_solvers.symbolic_fd`

Finite differences for any parsed equation system on a box: steady (Newton) or method of lines (BDF).

<small>`pinneapple_simulation/numerical_solvers/symbolic_fd.py` · 498 linhas · testes: 17</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FDError` | classe | `()` | The system cannot be solved by this finite-difference scheme (and why). |
| `FDResult` | classe | `(axes, fields, t, info)` |  |
| `FDOperator` | classe | `()` | A parsed system compiled on a grid: the right-hand sides, the face conditions and the initial state. Arrays may carry leading axes (an ensemble); coefficients named in ``keep`` stay symbolic and are read from the environment (``extra``), so an ensemble of coefficient values can be stepped at once. |
| `compile_system` | função | `(system, bounds, *, n=64, coefficients=None, keep=()) -> FDOperator` | Compile ``system`` for finite differences (see :func:`solve_fd`, which uses it). |
| `initial_state` | função | `(op, t0) -> dict[tuple[str, bool], np.ndarray]` | Initial values (and ``f_t`` for second order) of every field on the grid, faces set by their conditions. |
| `solve_fd` | função | `(system, bounds, *, n=64, n_save=41, coefficients=None, relax=20000.0, rtol=1e-06, atol=1…)` | Solve ``system`` on the box ``bounds`` (one interval per coordinate, ``t`` included when time-dependent) with ``n`` nodes per spatial axis. Returns :class:`FDResult`; FDError when the system is outside the scheme. |

#### `pinneapple_simulation.numerical_solvers.thermal_channel_2d`

Transient 2-D heated-channel CFD: incompressible Navier-Stokes + energy, with time-varying actuators (inlet velocity U_in(t), heater power Q_h(t)).

<small>`pinneapple_simulation/numerical_solvers/thermal_channel_2d.py` · 231 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ChannelConfig` | classe | `(nx, ny, L, Re, Pr, heater, dt, gamma_safety, record_every)` |  |
| `ThermalChannel2D` | classe | `(cfg=None)` |  |

#### `pinneapple_simulation.numerical_solvers.thermal_channel_datasets`

Heated-channel CFD dataset (zoo ``thermal_channel_cfd``): verification and actuated trajectories.

<small>`pinneapple_simulation/numerical_solvers/thermal_channel_datasets.py` · 181 linhas · testes: 2</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `verify` | função | `() -> dict` | Poiseuille (3 grids), fully developed Nusselt number (2 grids) and the three-grid study of T_out, T_max, dp. |
| `make_actuation` | função | `(family, seed)` | (U spec, Q spec) of one trajectory: ``random_steps_ramps`` (training family), ``sinusoid_chirp`` and ``edge_steps`` (unseen families). |
| `eval_act` | função | `(spec, t, which) -> float` |  |
| `simulate` | função | `(args) -> dict` | One trajectory: warm-up, then the recorded horizon (fields every ``REC``). |
| `generate` | função | `(out, workers=1, name='thermal_channel_cfd') -> dict` | Run :data:`PLAN` and write ``out/<name>/<split>[-NN].npz`` and ``meta.json``. |

#### `pinneapple_simulation.numerical_solvers.tvd_advection`

Second-order TVD (Total Variation Diminishing) upwind scheme for 1D scalar advection, with a van Leer flux limiter.

<small>`pinneapple_simulation/numerical_solvers/tvd_advection.py` · 69 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `van_leer_limiter` | função | `(r) -> np.ndarray` | van Leer TVD flux limiter: Phi(r) = (r + \|r\|) / (1 + \|r\|). r is the ratio of consecutive solution gradients (upstream/downstream); Phi(r)=0 for r<=0 (a local extremum -- falls back to pure upwind) and Phi(r) -> 1 as r -> infinity (smooth region -- full 2nd-order correction). |
| `tvd_advection_rhs` | função | `(f, v, dz) -> np.ndarray` | Second-order TVD-upwind estimate of df/dt = -v*df/dz on a uniform grid of spacing dz, given the field f (shape (N,)) and a constant velocity v (v>0: flow toward increasing index; v<0: toward decreasing index). Domain-boundary interface fluxes use plain first-order upwind (no downstream neighbor … |

#### `pinneapple_simulation.numerical_solvers.vmd`

VMD: Variational Mode Decomposition.

<small>`pinneapple_simulation/numerical_solvers/vmd.py` · 143 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `VMDSolver` | classe | `(K=4, alpha=2000.0, tau=0.0, DC=False, init='uniform', tol=1e-07, max_iter=500)` |  |
| `vmd_1d` | função | `(x, *, K=4, alpha=2000.0, tau=0.0, DC=False, init='uniform', tol=1e-07, max_iter=500) -> …)` | Returns (modes[K,T], center_freqs[K]). |

#### `pinneapple_simulation.numerical_solvers.water_hammer_moc`

Water hammer in a reservoir - pipe - valve system by the method of characteristics (MOC).

<small>`pinneapple_simulation/numerical_solvers/water_hammer_moc.py` · 130 linhas · testes: 17</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `wave_speed` | função | `(K, rho, D, e, E, c1=1.0) -> float` | Pressure-wave speed in a liquid-filled elastic pipe (Korteweg; Wylie & Streeter eq. 2.29): ``a = sqrt((K/rho) / (1 + c1 K D / (E e)))``. ``c1 = 1 - nu^2`` for a pipe anchored against axial movement. |
| `joukowsky_head` | função | `(a, dv) -> float` | Head rise of an instantaneous velocity change: ``dH = a dV / g`` (Joukowsky 1898). |
| `michaud_head` | função | `(a, dv, length, t_close) -> float` | Michaud's (1878) estimate for a closure slower than ``2L/a``: ``dH = 2 L dV / (g t_c)``; equals Joukowsky for ``t_c <= 2L/a``. |
| `valve_closure_moc` | função | `(*, length, diameter, wave_speed, friction_factor, q0, h_reservoir, h_downstream=0.0, t_c…)` | Transient heads and flows after the valve at the end of the pipe starts closing at t = 0. |

#### `pinneapple_simulation.numerical_solvers.wavelet`

Wavelet-based decomposition/denoising features.

<small>`pinneapple_simulation/numerical_solvers/wavelet.py` · 63 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `WaveletSolver` | classe | `(wavelet='db4', level=4, mode='periodization')` |  |
| `wavedec_features` | função | `(x, wavelet='db4', level=4, mode='periodization') -> Tuple[np.ndarray, Dict[str, Any]]` | Return concatenated detail coefficients as features. |

#### `pinneapple_simulation.numerical_solvers.xtfc_general`

General X-TFC on a box: any number of dimensions and fields, value and derivative constraints on faces, linear or non-linear residuals.

<small>`pinneapple_simulation/numerical_solvers/xtfc_general.py` · 335 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Constraint` | classe | `(dim, at, order)` | ``d^order u / dx_dim^order`` at ``x_dim = at`` is prescribed (by the problem's constraint function). |
| `XTFCProblem` | classe | `(bounds, fields, constraint_fn, residual, derivatives, linear)` | A differential problem on ``bounds`` for X-TFC. |
| `XTFCResult` | classe | `(beta, iterations, converged, step_norms, lstsq_seconds, total_seconds, n_neurons, _probl…)` |  |
| `activation_derivative` | função | `(name, z, n) -> np.ndarray` | ``d^n act / dz^n`` at ``z`` (``name`` in ``tanh``, ``logistic``). |
| `switching_functions` | função | `(constraints) -> np.ndarray` | Coefficients (monomial basis, row k = s_k) of the switching functions of constraints on one dimension. |
| `solve_xtfc` | função | `(problem, points, *, n_neurons, activation='tanh', weight_range=(-1.0, 1.0), seed=0, max_…)` | Solve ``problem`` on the training ``points`` (N, ndim) by X-TFC. |
| `grid` | função | `(bounds, n, kind='uniform') -> np.ndarray` | Tensor grid (N, ndim) of ``n`` points per dimension, ``uniform`` or ``cgl`` (Chebyshev-Gauss-Lobatto), boundary included. |

#### `pinneapple_simulation.numerical_solvers.xtfc_ivp`

X-TFC (eXtreme Theory of Functional Connections) solver for systems of ODEs posed as initial-value problems:

<small>`pinneapple_simulation/numerical_solvers/xtfc_ivp.py` · 438 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `XTFCIVPSolver` | classe | `(n_basis=50, n_collocation=100, activation='tanh', max_iter=50, tol=1e-10, seed=0, colloc…)` | Thin `SolverBase`/registry wrapper around `solve_xtfc_ode` for callers that want the uniform `SolverOutput` interface. The functional API (`solve_xtfc_ode`, `xtfc_predict`) is the primary entry point and can be used directly without this wrapper. |
| `chebyshev_gauss_lobatto` | função | `(n, lo=-1.0, hi=1.0) -> Array` | N Chebyshev-Gauss-Lobatto points on [lo, hi], ascending order. Standard collocation choice for spectral/TFC methods -- clusters points quadratically near both endpoints, which both improves the conditioning of the least-squares system (vs. uniform spacing, which is offered as a fallback via … |
| `xtfc_predict` | função | `(t_query, w, b, beta, y0, activation, t0, tf) -> Array` | Evaluate the trained TFC constrained expression at arbitrary query times -- a pure function of the persisted (w, b, beta, y0, activation, t0, tf), so a solved model can be re-evaluated later from just those small arrays without re-running solve_xtfc_ode. |
| `solve_xtfc_ode` | função | `(rhs_fn, y0, t_span, n_basis=50, n_collocation=100, activation='tanh', max_iter=50, tol=1…)` | Solve dy_i/dt = rhs_fn(t, y)[:, i], y_i(t0) = y0[i], for t in t_span, via X-TFC (see module docstring for the full method). |

#### `pinneapple_simulation.numerical_solvers.xtfc_pde`

X-TFC (eXtreme Theory of Functional Connections) solver for 1D-space(+time) PDEs posed on a rectangle x in [x0, x1], t in [t0, tf], with one initial condition (at t=t0) and two Dirichlet boundary conditions (at x=x0, x=x1):

<small>`pinneapple_simulation/numerical_solvers/xtfc_pde.py` · 584 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `XTFCPDESolver` | classe | `(pde='burgers_1d', n_basis=60, n_collocation_x=25, n_collocation_t=25, activation='tanh',…)` | Thin `SolverBase`/registry wrapper around `solve_xtfc_pde` for callers that want the uniform `SolverOutput` interface. The functional API (`solve_xtfc_pde`, `xtfc_pde_predict`) is the primary entry point. |
| `xtfc_pde_predict` | função | `(x_query, t_query, wx, wt, b, beta, pde, activation, x_span, t_span) -> Array` | Evaluate the trained 2D TFC constrained expression at arbitrary (paired) query points -- a pure function of the persisted (wx, wt, b, beta, pde, activation, x_span, t_span), mirroring `xtfc_predict` in `xtfc_ivp.py`. |
| `solve_xtfc_pde` | função | `(pde, x_span=None, t_span=None, n_basis=60, n_collocation_x=25, n_collocation_t=25, activ…)` | Solve `pde` in {"heat_1d", "burgers_1d"} via X-TFC (see module docstring for the full method). |

#### `pinneapple_simulation.numerical_solvers.xtfc_subdomain`

Subdomain-chained X-TFC for extremely stiff ODE-IVP systems (target stiffness ratio -- max|eigenvalue(Jacobian)| / min|eigenvalue(Jacobian)| -- roughly 1e10 to 1e14), plus a generic "extra scalar parameter" input.

<small>`pinneapple_simulation/numerical_solvers/xtfc_subdomain.py` · 395 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `XTFCSubdomainSolver` | classe | `(n_subdomains=12, first_width=None, spacing='log', n_basis=30, n_collocation=40, activati…)` | Thin `SolverBase`/registry wrapper around `solve_xtfc_subdomains`. The functional API (`solve_xtfc_subdomains`, `log_subdomain_boundaries`) is the primary entry point. |
| `log_subdomain_boundaries` | função | `(t0, tf, n_subdomains, first_width=None) -> Array` | n_subdomains+1 boundaries [t0, ..., tf] with logarithmically GROWING widths -- tiny near t0 (to resolve a fast initial transient spanning many decades), geometrically larger towards tf. |
| `solve_xtfc_subdomains` | função | `(rhs_fn, y0, t_span, n_subdomains=12, first_width=None, spacing='log', n_basis=30, n_coll…)` | Solve a (possibly extremely stiff) dy_i/dt = rhs_fn(t, y)[:, i], y_i(t0) = y0[i] IVP over log-spaced subdomains of t_span, chaining continuity across subdomain boundaries (see module docstring). |

#### `pinneapple_simulation.particle_dynamics`

Simulação dinâmica diferenciável em PyTorch puro: corpos rígidos, MPM (MLS-MPM) e SPH; reexporta também os parâmetros de solo/roda de Bekker-Wong.

<small>`pinneapple_simulation/particle_dynamics/__init__.py` · 61 linhas · **sem teste direto**</small>

#### `pinneapple_simulation.particle_dynamics._utils`

Utilitário privado que monta o vetor de gravidade (padrão -9,81 no eixo y) usado pelos simuladores de partículas. [sem teste]

<small>`pinneapple_simulation/particle_dynamics/_utils.py` · 25 linhas · **sem teste direto**</small>

#### `pinneapple_simulation.particle_dynamics.geo_mpm`

MPM para geomecânica (colapso granular, deslizamentos, rompimento de barragem, sólidos moles): MLS-MPM com APIC em unidades físicas, materiais areia (Drucker-Prager), fluido e elástico, com quadros no formato do GeoTaichi.

<small>`pinneapple_simulation/particle_dynamics/geo_mpm.py` · 355 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Sand` | classe | `(friction_deg, E, nu, rho)` | Material areia: plasticidade de Drucker-Prager sem coesão, com ângulo de atrito, E, nu e densidade. |
| `Fluid` | classe | `(K, rho, gamma)` | Material fluido fracamente compressível, pressão de J (tipo Tait). |
| `Elastic` | classe | `(E, nu, rho)` | Material sólido hiperelástico corrotacional fixo. |
| `GeoMPM` | classe | `(domain, dx, gravity=None, floor='friction', walls='slip', wall_friction=0.5, cfl=0.3, fl…)` | Simulador MLS-MPM 2D/3D em unidades físicas com corpos, paredes (slip, sticky ou atrito), passo estável por CFL e gravação de quadros. |
| `box` | função | `(lo, hi)` | Corpo retangular (ou caixa): devolve um teste de pertencimento para `add_body`. |
| `write_particle_frame` | função | `(out, i, fr) -> str` | Grava um quadro de partículas em `.npz` no layout do GeoTaichi. |
| `read_particle_frames` | função | `(out, kind='MPM', allow_pickle=False) -> list[dict]` | Lê os quadros de uma corrida deste módulo ou do GeoTaichi (MPM ou DEM); `allow_pickle` só para arquivos confiáveis. |
| `runout` | função | `(position, axis=0, floor_band=None, quantile=0.995) -> float` | Posição da frente de um depósito ao longo de um eixo, por um quantil alto robusto a partículas soltas. |

#### `pinneapple_simulation.particle_dynamics.molecular_dynamics`

Dinâmica molecular NVE com potencial de Lennard-Jones par a par e integração velocity-Verlet, em PyTorch.

<small>`pinneapple_simulation/particle_dynamics/molecular_dynamics.py` · 274 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MDState` | classe | `(pos, vel)` | Contêiner de posições e velocidades de um sistema de N partículas. |
| `MolecularDynamicsSimulator` | classe | `(epsilon=1.0, sigma=1.0, mass=1.0, cutoff=None, box_size=None, dim=3) -> None` | Simulador mínimo NVE de Lennard-Jones com velocity-Verlet, raio de corte e caixa periódica opcionais. |
| `lj_potential` | função | `(dist, epsilon, sigma) -> torch.Tensor` | Potencial de par de Lennard-Jones U(r) = 4ε((σ/r)^12 - (σ/r)^6). |
| `lj_force_magnitude` | função | `(dist, epsilon, sigma) -> torch.Tensor` | Módulo analítico da força radial de Lennard-Jones, F(r) = -dU/dr. |

#### `pinneapple_simulation.particle_dynamics.mpm`

Método do Ponto Material (MLS-MPM) diferenciável para sólidos elásticos, fluidos viscosos e neve com plasticidade de estiramento crítico, com etapas P2G, atualização da grade e G2P.

<small>`pinneapple_simulation/particle_dynamics/mpm.py` · 481 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MPMState` | classe | `(pos, vel=None, F=None, C=None, Jp=None) -> None` | Contêiner do estado das partículas MPM: posição, velocidade, gradiente de deformação F, matriz afim C e Jp. |
| `MPMSimulator` | classe | `(grid_resolution=64, dt=0.0001, material='elastic', dim=2, E=10000.0, nu=0.2, rho=1.0, gr…)` | Simulador MPM diferenciável (`nn.Module`) com materiais elástico, fluido ou neve, em 2D/3D. |

#### `pinneapple_simulation.particle_dynamics.particles`

Simulação por partículas: SPH fracamente compressível (equação de Tait, viscosidade laminar) sobre uma classe base com busca de vizinhos O(N²); a base levanta NotImplementedError em `forward`, e o DEM citado no docstring não está implementado aqui.

<small>`pinneapple_simulation/particle_dynamics/particles.py` · 334 linhas · testes: 151 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ParticleSystem` | classe | `(n_particles, dim=2, device='cpu') -> None` | Base de sistemas de partículas com busca de vizinhos por força bruta O(N²); `forward` levanta NotImplementedError. |
| `SPHParticles` | classe | `(n_particles, smoothing_length, rho0=1000.0, c0=10.0, nu=0.0001, dim=2, gamma=7.0, gravit…)` | SPH fracamente compressível: densidade por soma de kernel, pressão de Tait, viscosidade e gravidade. |

#### `pinneapple_simulation.particle_dynamics.rigid_body`

Dinâmica de corpos rígidos diferenciável em 2D/3D com integração Euler simplética, para aprendizado de robôs e acoplamento com PINNs.

<small>`pinneapple_simulation/particle_dynamics/rigid_body.py` · 306 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RigidBodyState` | classe | `(n_bodies=1, dim=2, device='cpu', dtype=torch.float32) -> None` | Contêiner mutável do estado (posição, orientação, velocidades) de vários corpos rígidos. |
| `RigidBody` | classe | `(mass, inertia, dim=2) -> None` | Corpo rígido único com massa, inércia e integrador simplético. |
| `RigidBodySystem` | classe | `(bodies, gravity=(0.0, -9.81, 0.0)) -> None` | Sistema de vários corpos rígidos com gravidade opcional. |

#### `pinneapple_simulation.particle_dynamics.terramechanics`

Dataclasses de parâmetros de solo e roda para a interação roda-solo de Bekker-Wong.

<small>`pinneapple_simulation/particle_dynamics/terramechanics.py` · 48 linhas · testes: 28</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SoilParams` | classe | `(c, phi_deg, K, k_c, k_phi, n, rho, g, a0, a1)` | Parâmetros de solo de Bekker-Wong (coesão, atrito, módulos de afundamento, densidade). |
| `WheelParams` | classe | `(R, b, n_wheels, mass_rover)` | Geometria de roda rígida e massa do rover, com peso por roda. |

#### `pinneapple_simulation.pipelines`

End-to-end pipelines that chain external tools (CAD, mesher, solver) into one checked, reproducible run.

<small>`pinneapple_simulation/pipelines/__init__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_simulation.pipelines.freecad_openfoam`

Headless, reproducible pipelines from a parametric FreeCAD model to a checked OpenFOAM result.

<small>`pinneapple_simulation/pipelines/freecad_openfoam/__init__.py` · 29 linhas · testes: 14</small>

#### `pinneapple_simulation.pipelines.freecad_openfoam.__main__`

``python -m pinneapple_simulation.pipelines.freecad_openfoam``:

<small>`pinneapple_simulation/pipelines/freecad_openfoam/__main__.py` · 46 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `main` | função | `(argv=None) -> int` |  |

#### `pinneapple_simulation.pipelines.freecad_openfoam.cad`

The FreeCAD step: a parametric document in, named boundary patches out.

<small>`pinneapple_simulation/pipelines/freecad_openfoam/cad.py` · 365 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PatchSpec` | classe | `(name, where, kind)` | One boundary patch: the rule that selects its faces and its kind (``inlet``, ``outlet``, ``wall``, ``symmetry``); a plain string is a rule, the kind then follows the name (inlet*, outlet*, else wall). |
| `Face` | classe | `(index, type, area, centre, points, triangles)` |  |
| `CadGeometry` | classe | `(faces, info)` | Faces of one solid in document units (mm), with what the backend reported (volume, validity, files). |
| `FreeCADBackend` | classe | `(freecadcmd=None, timeout_s=600.0)` | FreeCAD through ``freecadcmd`` (headless). ``timeout_s`` per job. |
| `FakeFreeCAD` | classe | `()` | The templates of :mod:`.models` built in NumPy: the same faces (types, centres, outward triangles) and spreadsheet parameters as the FreeCAD documents, without FreeCAD. Documents are JSON: ``{"fake_freecad": template, "parameters": {...}, "solid": name}``. |
| `rule_eval` | função | `(expr, names, functions=None) -> Any` | Value of a patch rule: numbers, names, arithmetic, comparisons (chained), ``and or not`` and a few ``math`` functions (plus ``functions``); anything else raises ``ValueError``. |
| `assign_patches` | função | `(geom, patches, parameters=None) -> dict[str, list[int]]` | Face indices of each patch: the first rule a face satisfies takes it; ``rest`` takes what is left. Raises ``ValueError`` when a face is left without a patch or a patch gets no face. |
| `write_stl` | função | `(path, geom, assignment, scale=0.001) -> Path` | ASCII STL with one ``solid <patch>`` per patch (snappyHexMesh turns each into a region), points times ``scale`` (mm -> m by default), merged across faces so shared edges stay shared. |
| `find_freecadcmd` | função | `() -> str \| None` | ``PINNEAPPLE_FREECADCMD``, the automation toolchain's FreeCAD (``/opt/freecad`` AppImage), or PATH. |
| `get_backend` | função | `(name='auto')` | ``"freecad"``, ``"fake"``, or ``"auto"`` (FreeCAD when ``freecadcmd`` is found, else the fake). |

#### `pinneapple_simulation.pipelines.freecad_openfoam.case`

OpenFOAM case templates for the meshed geometry: steady ``simpleFoam`` or transient ``pimpleFoam``, incompressible, laminar or k-omega SST, with an optional passive scalar (a temperature at low heating, or a concentration).

<small>`pinneapple_simulation/pipelines/freecad_openfoam/case.py` · 221 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ScalarSpec` | classe | `(name, D, inlet, walls, initial)` | A passive scalar: ``D`` its diffusivity (m²/s; nu / Pr for a temperature), ``inlet`` its value on the inlets, ``walls`` patch -> fixed value or ``"zeroGradient"`` (walls not named: zero gradient). |
| `Physics` | classe | `(velocity, nu, rho, turbulence, solver, iterations, residual_tol, end_time, dt, max_coura…)` | Incompressible flow: ``velocity`` (m/s, uniform on the inlets), ``nu`` (m²/s), ``rho`` (kg/m³, for forces and pressures in Pa), ``turbulence`` laminar \| kOmegaSST, ``solver`` simpleFoam \| pimpleFoam. Steady: ``iterations`` and ``residual_tol`` (residualControl). Transient: ``end_time``, ``dt``, … |
| `write_internal_case` | função | `(case, kinds, inlet_normals, physics, procs=1, size=None) -> dict[str, Any]` | 0.orig fields, constant/ and system/ dictionaries and ``Allrun`` of an internal-flow case. |

#### `pinneapple_simulation.pipelines.freecad_openfoam.freecad_scripts.fc_job`

_Módulo sem docstring e sem texto._

<small>`pinneapple_simulation/pipelines/freecad_openfoam/freecad_scripts/fc_job.py` · 175 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `sheet` | função | `(doc, parameters)` |  |
| `cylinder` | função | `(doc, name, radius, height, placement)` |  |
| `build_pipe_bend` | função | `(doc, p)` | Straight inlet leg along +x ending at the origin, a bend of radius Rc (centre line) through `angle` degrees about the z axis, a straight outlet leg; diameter D. |
| `build_heated_pipe` | função | `(doc, p)` | Straight pipe along +x of diameter D: an unheated entry length L_entry from x = 0, then a heated length L_heated (two cylinders, so the wall has one face per section). |
| `build_spheroid` | função | `(doc, p)` | Spheroid centred at the origin, length L along x (the flow direction), diameter D across (L = D: sphere). |
| `set_parameters` | função | `(doc, parameters)` |  |
| `faces_of` | função | `(shape, tol)` |  |
| `main` | função | `()` |  |

#### `pinneapple_simulation.pipelines.freecad_openfoam.mesh`

Meshing the exported geometry: blockMesh background + snappyHexMesh (always available), or cfMesh's ``cartesianMesh`` when it is installed (``mesher="auto"`` picks it then).

<small>`pinneapple_simulation/pipelines/freecad_openfoam/mesh.py` · 193 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MeshSpec` | classe | `(cells_across, length, level, mesher, feature_angle, max_non_ortho)` | ``cells_across``: cells across ``length`` (a parameter name, mm, or a number in mm) near the walls; ``level``: snappyHexMesh surface level (the background is ``level`` times coarser by 2); ``mesher``: snappy, cfmesh or auto. ``max_non_ortho`` is the snappy quality limit. |
| `has_cfmesh` | função | `(bashrc=None) -> bool` |  |
| `split_procs` | função | `(procs, size) -> tuple[int, int, int]` | Split ``procs`` into (nx, ny, nz) along the longest sides of the box first. |
| `decompose_dict` | função | `(case, procs, size) -> None` | hierarchical decomposition (scotch / metis can be stubs in some builds). |
| `allmesh_script` | função | `(procs, mesher='snappy') -> str` |  |
| `allrun_script` | função | `(solver, procs, scalar=False) -> str` | Copy ``0.orig`` to ``0`` and run ``solver`` (decomposed with MPI when ``procs > 1``, then the latest time reconstructed); the solver log is ``log.<solver>`` in the case. ``scalar``: then 20 steady iterations of ``scalarTransportFoam`` on the converged flow (log ``log.scalarTransport``). |
| `write_internal_mesh` | função | `(case, stl, patches, inside, cell_size, spec, procs=1, bashrc=None) -> dict[str, Any]` | Dictionaries and ``Allmesh`` of an internal-flow mesh of the closed multi-solid ``stl`` (metres). |

#### `pinneapple_simulation.pipelines.freecad_openfoam.models`

Parametric FreeCAD templates the pipeline ships, with their default patch rules and a NumPy twin for tests.

<small>`pinneapple_simulation/pipelines/freecad_openfoam/models.py` · 196 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Model` | classe | `(name, title, flow, solid, parameters, patches, fake)` |  |
| `build_document` | função | `(template, output, parameters=None, backend='auto') -> dict[str, Any]` | Write the parametric document of ``template`` (defaults overridden by ``parameters``) with FreeCAD, or a fake document with :class:`~.cad.FakeFreeCAD` (``backend="fake"``, or ``"auto"`` without FreeCAD). |

#### `pinneapple_simulation.pipelines.freecad_openfoam.pipeline`

A parametric FreeCAD model to a checked OpenFOAM result, as one reproducible workflow.

<small>`pinneapple_simulation/pipelines/freecad_openfoam/pipeline.py` · 231 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PipelineSpec` | classe | `(id, title, template, document, solid, flow, parameters, inputs, patches, physics, mesh, …)` | One FreeCAD -> OpenFOAM study. |
| `evaluate` | função | `(expr, names) -> Any` | A number, or the value of an expression (see :func:`.cad.rule_eval`) with the reference functions. |
| `resolve_physics` | função | `(physics, inputs) -> dict[str, Any]` | Physics values with expressions evaluated in order (a later key may use an earlier one, e.g. ``velocity = Re * nu / (D * 1e-3)``); nested dicts (the scalar) too. |
| `run_pipeline` | função | `(spec, folder, inputs=None, approve=True, approver='pipeline')` | Run (or resume) the study in ``folder`` with ``inputs`` (document parameters and other inputs). ``approve``: every step is approved (the solver is ``expensive``; a resumed step overwrites its own earlier files); without it the run stops at the first step that asks. Returns the automation … |

#### `pinneapple_simulation.pipelines.freecad_openfoam.post`

Post-processing of a finished internal-flow case: what the pipeline reports and checks.

<small>`pinneapple_simulation/pipelines/freecad_openfoam/post.py` · 322 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Section` | classe | `(name, origin, normal, radius)` | A disc of ``radius`` (m) centred at ``origin`` (m) with unit ``normal`` (the positive flow direction). |
| `CaseResult` | classe | `(case, scalar=None)` | Fields of the latest time of a case on its mesh: cell centres, volumes, U, p and the scalar, the boundary patches and their face values, with nearest-cell sampling. |
| `section_average` | função | `(res, sec) -> dict[str, float]` | Area-averaged p (kinematic), flux and bulk scalar over the part of the disc inside the fluid. |
| `wall_loads` | função | `(case, walls, nu, rho) -> dict[str, Any]` | Pressure and viscous force (N) on the wall patches and y+ of their first cells, from the written fields: :meth:`~pinneapple_simulation.numerical_solvers.external_flow.CaseFields.wall` (laminar sublayer, log law above y+ = 11.25), y+ = d u_tau / nu with d the wall distance of the first cell centre. |
| `read_case` | função | `(case, kinds, rho, sections=(), scalar=None, nu=1e-06) -> dict[str, Any]` | Every measured quantity of a finished case (see the module docstring); pressures in Pa. |
| `plane_figure` | função | `(res, path, origin, normal, title, rho, scalar_label=None, n=360) -> Path` | Speed, pressure (and the scalar) sampled on a plane through the domain; outside the fluid is blank. |
| `mesh_figure` | função | `(res, path, origin, normal, title) -> Path` | The cells cut by a plane, each drawn as a square of its size at its centre (snappyHexMesh cells are cubes away from the walls), coloured by size. |
| `geometry_figure` | função | `(geom, assignment, path, title) -> Path` | The exported solid with its faces coloured by patch (3-D, matplotlib). |

#### `pinneapple_simulation.pipelines.freecad_openfoam.report`

Validation of a study: the same case on successively finer meshes, Richardson extrapolation (Roache's GCI, :func:`pinneapple_analysis.verification.convergence.richardson_extrapolate`), the error of every mesh and of the extrapolated value against the reference, a table and figures.

<small>`pinneapple_simulation/pipelines/freecad_openfoam/report.py` · 154 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `mesh_study` | função | `(factory, resolutions, folder, reference, inputs=None, h_of=None, **kw) -> dict[str, Any]` | Run ``factory(resolution, **kw)`` for each resolution (coarse to fine) in ``folder/r<resolution>`` (resumed when already run) and extrapolate the measured value of reference ``reference``. ``h_of``: the cell size of a resolution (default 1 / resolution: cells across a length; for snappy surface … |
| `mesh_study_figure` | função | `(studies, path, labels=None) -> Path` | One panel per mesh study: measured value against cell size, the reference and its tolerance band, the Richardson-extrapolated value at h = 0. |
| `sweep_figure` | função | `(rows, inputs, output, study, path, unit='') -> Path` | The sweep (``output`` over the first two inputs) and the surrogate's predictions against the solver on the training and the held-out cases. |
| `validation_markdown` | função | `(studies) -> str` | The measured validation table (one row per mesh, then the extrapolated value). |

#### `pinneapple_simulation.pipelines.freecad_openfoam.steps`

The three pipeline actions that are not plain OpenFOAM commands, as functions of their parameters and a step folder. The automation adapters (``freecadcmd.export``, ``openfoam.prepare``, ``openfoam.post`` in :mod:`pinneapple_tools.automation.adapters.pipeline`) wrap them; each returns ``{"outputs", …

<small>`pinneapple_simulation/pipelines/freecad_openfoam/steps.py` · 340 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `point_inside` | função | `(V, F, p) -> bool` | Whether ``p`` is inside the closed triangle mesh (parity of crossings along three skew rays, majority). |
| `auto_procs` | função | `(cells, cores=None, per_proc=20000.0) -> int` | Processes for a mesh of about ``cells`` cells: one per ``per_proc`` cells, at most the cores here. |
| `cad_step` | função | `(params, step_dir) -> dict[str, Any]` | Set the parameters, recompute, export, assign patches, write ``geometry.stl`` (m) and ``geometry.png``. |
| `prepare_step` | função | `(params, step_dir) -> dict[str, Any]` | Mesh dictionaries + case + ``Allmesh`` / ``Allrun`` in ``step_dir/case``. |
| `post_step` | função | `(params, step_dir) -> dict[str, Any]` | Measured quantities, figures, derived quantities and reference errors of a finished case. |

#### `pinneapple_simulation.pipelines.freecad_openfoam.studies`

The worked, validated studies of the pipeline, as :class:`~.pipeline.PipelineSpec` factories (generic geometry, no client data). Each compares a measured quantity with a closed form or a published correlation (:mod:`pinneapple_physics.closed_form`); the measured numbers are in …

<small>`pinneapple_simulation/pipelines/freecad_openfoam/studies.py` · 116 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `pipe_bend` | função | `(cells_across=16, backend='auto', procs='auto', bend_window=True) -> PipelineSpec` | Laminar flow through a pipe bend; inputs: the FreeCAD parameters (D, Rc, angle, L_in, L_out) and Re. ``bend_window=False`` drops the sections inside the bend and the White (1929) reference, which need a bend of at least 165° (for sweeps over the angle). |
| `heated_pipe` | função | `(cells_across=16, backend='auto', procs='auto') -> PipelineSpec` | Laminar thermal entry at constant wall temperature (passive scalar T, inlet 0, heated wall 1). |
| `sphere` | função | `(surface_level=4, backend='auto', procs='auto') -> PipelineSpec` | Sphere (the spheroid template with L = D) in a uniform stream at low Reynolds number. |

#### `pinneapple_simulation.pipelines.freecad_openfoam.sweep`

Parameter sweeps / designs of experiments over the inputs of a :class:`.pipeline.PipelineSpec`, and a surrogate of the results measured on held-out cases.

<small>`pinneapple_simulation/pipelines/freecad_openfoam/sweep.py` · 152 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Sweep` | classe | `(spec, design, folder, ids)` | ``spec`` run at every point of ``design`` (list of input dicts) under ``folder``. |
| `doe` | função | `(bounds, n, method='lhs', seed=0, fixed=None) -> list[dict[str, float]]` | ``n`` points in the box ``bounds`` (name -> (low, high)): ``lhs`` (Latin hypercube), ``sobol`` or ``grid`` (``n`` levels per input, n**d points); ``fixed`` values are added to every point. |
| `case_id` | função | `(i, inputs) -> str` |  |
| `fit_surrogate` | função | `(rows, inputs, outputs, *, test_frac=0.25, val_frac=0.15, seed=0, budget=5.0, methods='au…)` | A surrogate of ``outputs`` from ``inputs`` on the finished cases: :func:`pinneapple_analysis.surrogate. run_study` with a random held-out test set (no distribution-shift split: sweeps are small). Returns the study; ``study.ranking("test")`` holds the held-out errors, ``study.model.predict`` the … |

#### `pinneapple_simulation.robot_sim`

Robot Sim do PINNeAPPle: robôs no NVIDIA Newton com as checagens do PINNeAPPle (torques contra um modelo multicorpo independente, energia, convergência em dt, cargas estruturais) e um backend Isaac Sim com a mesma interface.

<small>`pinneapple_simulation/robot_sim/__init__.py` · 18 linhas · testes: 39</small>

#### `pinneapple_simulation.robot_sim.arms`

Braços prontos para o robot sim (o `TwoLinkArm` de tubos de alumínio), montados com os dados de material e modelos de elo do PINNeAPPle, para que as mesmas massas vão ao Newton (URDF), ao Isaac Sim (UsdPhysics) e à árvore multicorpo própria. [sem teste direto]

<small>`pinneapple_simulation/robot_sim/arms.py` · 74 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TwoLinkArm` | classe | `(l1, l2, tube, wall, material, motor, payload, shoulder_z, base_mass, effort_limits, grav…)` | Braço plano de dois elos no plano x-z com tubos quadrados de alumínio, motor e carga; gera o spec para Newton/Isaac e a árvore multicorpo. |

#### `pinneapple_simulation.robot_sim.checks`

Checagem em uma chamada de um movimento de pegar-e-colocar de um braço no robot sim, usada pelo nó `lib.robot.arm_check` do Flow Studio.

<small>`pinneapple_simulation/robot_sim/checks.py` · 50 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `arm_pick_and_place` | função | `(payload=5.0, duration=0.8, q_start=(-0.6, 1.2), q_end=(0.9, -0.4), dt=0.0005, kp=(350.0,…)` | Simula e checa um movimento de pegar-e-colocar do braço (torques contra modelo independente, energia, cargas na raiz dos elos). |

#### `pinneapple_simulation.robot_sim.learn`

Pequeno laço de aprendizado em CPU (método da entropia cruzada, CEM) contra rollouts do `RobotSim`.

<small>`pinneapple_simulation/robot_sim/learn.py` · 48 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CEMResult` | classe | `(best, best_cost, history, evaluations)` | Resultado do CEM: melhores parâmetros, custo, histórico e número de avaliações. |
| `cem` | função | `(objective, mean, std, *, iterations=6, population=8, elite=0.25, lower=None, upper=None,…)` | Minimiza `objective(params)` pelo método da entropia cruzada, com limites opcionais. |

#### `pinneapple_simulation.robot_sim.sim`

Simulação de robôs com NVIDIA Newton (`SolverFeatherstone`, CPU ou GPU via Warp), gravando torques, energia, poses e IMU; o `IsaacSimBackend` só escreve um script e levanta NotImplementedError/RuntimeError fora do Isaac Sim. [sem teste direto]

<small>`pinneapple_simulation/robot_sim/sim.py` · 275 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RobotRun` | classe | `(t, q, qd, tau, body_pos, body_quat, body_vel, body_omega, body_names, joint_names, masse…)` | Registro de uma corrida (tempo, q, qd, torques, poses e velocidades dos corpos, saturação) com energia, trabalho do atuador e IMU. |
| `RobotSim` | classe | `(builder_fn, *, gravity=9.81, ground=True, device='cpu', name='robot')` | Modelo Newton de um robô (base fixa ou flutuante) construído de URDF, MJCF, USD ou spec, com `run` e controladores. |
| `IsaacSimBackend` | classe | `(usd_path, robot_prim)` | Mesma interface `run` para o Isaac Sim; aqui só escreve o script autônomo e levanta erro fora do Isaac Sim. |
| `newton_available` | função | `() -> bool` | Diz se o NVIDIA Newton (e Warp) pode ser importado. |
| `torque_trajectory` | função | `(fn)` | Controlador em malha aberta com torques `fn(t)`. |
| `pd_tracking` | função | `(reference, kp, kd, feedforward=None)` | Controlador PD de rastreamento com feedforward opcional. |
| `convergence_order` | função | `(errors, dts) -> float` | Ordem observada de precisão: inclinação de log(erro) contra log(dt) por mínimos quadrados. |

#### `pinneapple_simulation.robot_sim.spec`

Robô como dados simples (elos, juntas, acionamentos), exportável como URDF e como UsdPhysics para Omniverse/Isaac Sim. [sem teste direto]

<small>`pinneapple_simulation/robot_sim/spec.py` · 108 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Link` | classe | `(name, mass, size, position, com, inertia, box_offset, color, collide, shape)` | Elo do robô: massa, tamanho, posição, centro de massa, inércia, forma e colisão. |
| `Joint` | classe | `(name, parent, child, anchor, axis, limits_deg, drive, stiffness, damping, max_force, max…)` | Junta do robô: pai, filho, âncora, eixo, limites, acionamento e limites de força/velocidade. |
| `ArticulationSpec` | classe | `(name, links, joints, fixed_base, gravity, ground, info)` | Articulação completa (elos, juntas, base fixa, gravidade, chão) como dados simples. |
| `box_inertia` | função | `(mass, size) -> np.ndarray` | Inércia principal de uma caixa sólida em torno do centro. |
| `cylinder_inertia` | função | `(mass, radius, length) -> np.ndarray` | Inércia principal de um cilindro sólido com eixo em y. |
| `spec_to_urdf` | função | `(spec) -> str` | Escreve o URDF do spec: inerciais, colisores e visuais caixa/cilindro, juntas de revolução com limites. |

#### `pinneapple_simulation.rover`

Six-wheel rocker-bogie rover on lunar terrain with Bekker-Wong wheel-soil forces.

<small>`pinneapple_simulation/rover/__init__.py` · 31 linhas · testes: 21</small>

#### `pinneapple_simulation.rover.autolab`

PINNeAPPle Autonomous Physics Lab, rover demo: *the rover that learns the planet*.

<small>`pinneapple_simulation/rover/autolab/__init__.py` · 38 linhas · testes: 8</small>

#### `pinneapple_simulation.rover.autolab.estimation`

What the rover measures, and what it infers about the soil from it.

<small>`pinneapple_simulation/rover/autolab/estimation.py` · 281 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Sensors` | classe | `(sinkage_sd, slip_sd, torque_rel, torque_abs, load_rel, seed)` | Noise levels (one standard deviation) of the rover's measurements; stated assumptions. |
| `UnitBelief` | classe | `(base, mean, cov, n_obs, history, chi2_dof, scale)` |  |
| `SoilBelief` | classe | `(prior_soils, wheel, g=1.62, prior_sd=PRIOR_SD, obs_sd_sinkage=0.008, obs_sd_torque=(2.0,…)` | Gaussian beliefs over the soil parameters of each mapped unit (see the module docstring). |
| `soil_theta` | função | `(soil) -> np.ndarray` |  |
| `theta_soil` | função | `(base, theta) -> Soil` |  |
| `measure` | função | `(run, sensors=None, wheels=None, every=4, min_speed=0.02, max_slip=0.7, slip_window=0.3, …)` | Noisy measurements from a vehicle run (front wheels by default), one row per wheel and kept sample. |

#### `pinneapple_simulation.rover.autolab.mission`

A mission in plain language -> the rover's lab loop, through a fixed set of tools.

<small>`pinneapple_simulation/rover/autolab/mission.py` · 393 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MissionSpec` | classe | `(text, goal, start, slip_limit, risk_max, objective, speed, max_replans, parser, quotes, …)` |  |
| `RuleParser` | classe | `()` | Deterministic parser with the ``extract(text, schema)`` interface of an LLM adapter (English and Portuguese keywords). |
| `StepLog` | classe | `(stage, tool, summary, seconds)` |  |
| `MissionResult` | classe | `(spec, routes, runs, belief, planners, measurements, comparison, report, log, retreat, ou…)` |  |
| `RoverMission` | classe | `(site, text, *, llm=None, spec=None, dt=0.004, terrain_cell=0.03, planner_cell=0.5, n_sam…)` | The rover lab loop on ``site`` for one request (see the module docstring). |
| `parse_mission` | função | `(text, site, llm=None) -> MissionSpec` | Request -> :class:`MissionSpec`. Values an LLM returns are kept only when their quote is in the text; missing values take the site's start / goal or the defaults (listed in ``defaults_used``). |
| `traction_monitor` | função | `(slip_limit, window_s=2.0, output_dt=0.05, front=None, t_min=2.0)` | Stop rule for :meth:`VehicleSimulator.run`: the mean slip of the (front) wheels over the last ``window_s`` seconds above ``slip_limit`` ends the run as "stalled". Samples before ``t_min`` seconds of driving (the start-up ramp) are not monitored. ``output_dt`` is not used (kept for compatibility): … |
| `retreat_point` | função | `(run, site, back=2.0) -> tuple` | Where the rover backs out to after a stall: ``back`` metres before the first point where a front wheel entered the unit it stalled on, on its own track (the reversal itself is not simulated). |
| `to_json` | função | `(obj)` | JSON-safe copy of mission outputs (arrays to lists, numbers rounded). |

#### `pinneapple_simulation.rover.autolab.planner`

Risk-aware route planning on a site from what the rover believes about its soils.

<small>`pinneapple_simulation/rover/autolab/planner.py` · 227 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `UnitTables` | classe | `(grades, slip, energy, mean_slip, mean_energy)` |  |
| `Route` | classe | `(waypoints, cells, length_m, energy_J, max_risk, mean_risk, max_slip, profile, belief_lab…)` |  |
| `RiskPlanner` | classe | `(site, belief, wheel, mass, slip_limit=0.4, g=1.62, n_wheels=4, cell=0.5, n_samples=24, r…)` | Plans on ``site`` with ``belief``; ``mass`` [kg], ``wheel``, ``n_wheels`` and ``g`` describe the rover. |
| `densify` | função | `(P, step)` |  |
| `simplify` | função | `(P, tol, accept=None)` | Douglas-Peucker simplification of a polyline (tolerance ``tol`` metres); ``accept(a, b)`` can veto a shortcut (e.g. one that crosses risky ground), which is then split further. |

#### `pinneapple_simulation.rover.autolab.render`

Blender Cycles renders of a rover mission (``pip install bpy``): the VIPER model of the repository (``data/geometries/viper_rover/viper_rover.light.glb``) on the site, the terrain with the ruts the multibody runs left, coloured by the learned traverse risk or by soil, and both routes.

<small>`pinneapple_simulation/rover/autolab/render.py` · 322 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `available` | função | `() -> bool` |  |
| `import_viper` | função | `(sc, path=VIPER_GLB)` | Import the VIPER glb under an empty (origin on the ground under the chassis, x forward, z up). |
| `pose_at` | função | `(run, k)` | (location, rotation matrix) of the rover at output sample ``k``: contact-point centroid and chassis axes. |
| `place` | função | `(root, run, k)` |  |
| `terrain_grid` | função | `(ter, x_range, y_range, step)` |  |
| `terrain_colors` | função | `(site, X, Y, planner=None, mode='risk', shade_with=None)` | Soil albedo (``mode="soil"``) or the learned worst-heading risk blended over it (``"risk"``). |
| `build_terrain` | função | `(site, runs, cell=0.03, upto=None)` | A fresh deformable terrain with the ruts of ``runs`` replayed (all of each run, or up to ``upto[i]`` s). |
| `route_lines` | função | `(runs, routes, learned_planner=None)` | Caption lines of a render: what each route was planned with and what the multibody run measured. |
| `render_overview` | função | `(site, runs, routes, planner, out_png, *, size=(1600, 900), samples=48, step=0.06, lines=…)` |  |
| `render_hero` | função | `(site, runs, out_png, *, size=(1600, 900), samples=96, window=6.0, cell=0.02, grains=2000…)` | Close-up of the stall of ``runs[0]`` (studio background, hard shadow, sand grains on the berms). |
| `render_traverse` | função | `(site, runs, out_dir, *, size=(800, 450), samples=16, every_s=2.5, step=0.06, fps=12, gif…)` | Frames of ``runs`` one after the other (chase camera, ruts growing), then ``traverse.mp4`` and ``traverse.gif``. Returns (frame paths, mp4, gif). |

#### `pinneapple_simulation.rover.autolab.report`

Report of a rover mission: ``summary.json`` (every number), ``trust_report.md``, ``report.md`` and the figures.

<small>`pinneapple_simulation/rover/autolab/report.py` · 206 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `run_info` | função | `() -> dict` | Versions and hardware of this run. |
| `mission_numbers` | função | `(mission, res) -> dict` | The measured and predicted numbers of a mission, labelled by where they come from. |
| `write_report` | função | `(mission, res, out_dir, figures=True, info=None) -> dict` |  |
| `report_markdown` | função | `(n) -> str` |  |

#### `pinneapple_simulation.rover.autolab.site`

A synthetic planetary site whose truth is known: relief, soils, and what an orbiter would have said about them.

<small>`pinneapple_simulation/rover/autolab/site.py` · 190 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Mound` | classe | `(x, y, sx, sy, height)` | Gaussian hill ``height * exp(-((x-x0)^2/sx^2 + (y-y0)^2/sy^2) / 2)`` [m]. |
| `Patch` | classe | `(unit, x, y, a, b, angle_deg)` | An ellipse of one soil unit (rotated by ``angle_deg``); later patches cover earlier ones. |
| `SoilUnit` | classe | `(name, truth, orbital, orbital_note, color)` | A mapped unit: its name, its true soil (hidden) and the soil the orbital map assigns to it. |
| `SyntheticSite` | classe | `(x_range, y_range, units, patches, mounds, tilt_deg, tilt_dir_deg, start, goal, descripti…)` |  |
| `soft_sand` | função | `() -> Soil` | The hidden soft sand of the demo site: GRC-1 with a looser state (see :data:`SOFT_SAND_NOTE`). |
| `demo_site` | função | `() -> SyntheticSite` | A 40 m x 24 m site of compact regolith (GRC-1) cut by a north-south ridge (3.6 m high, flanks up to about 20 deg). The ridge's pass (about 14 deg) is filled with soft sand that the orbital map labels as compact regolith; its northern part is firm gravelly ground (JSC-1A). A second soft-sand patch … |

#### `pinneapple_simulation.rover.autolab.verify`

Trust report of a rover mission: did it work, does the physics hold, and can the learned soil be believed.

<small>`pinneapple_simulation/rover/autolab/verify.py` · 182 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `monitor_slip` | função | `(run, front, window_s=2.0, t_min=2.0)` | The traction monitor's quantity: front-wheel mean slip averaged over the last ``window_s`` seconds (per output sample, once ``window_s`` seconds have been monitored from ``t_min`` on, exactly as :func:`~.mission.traction_monitor` decides; nan before). |
| `timestep_check` | função | `(mission, route, seconds=6.0) -> dict` |  |
| `verify_mission` | função | `(mission, routes, runs, planners, belief, meas_all, comparison, do_timestep=True)` |  |

#### `pinneapple_simulation.rover.autolab.visuals`

Figures of a rover mission (matplotlib): site and risk maps with both routes, the soil belief converging, the wheel models against the measurements, and the traverses. Every curve is computed from the mission's own runs.

<small>`pinneapple_simulation/rover/autolab/visuals.py` · 194 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_site_map` | função | `(site, planners, routes, runs, path, slip_limit=0.4)` | Truth soils with relief and routes; the risk the rover saw before (orbital prior) and after learning. |
| `plot_learning` | função | `(belief, unit, truth, path, title='')` | Mean +- 2 sd of each soil parameter of ``unit`` after every filter update, with the prior and the truth. |
| `plot_model_comparison` | função | `(comparison, path)` | Sinkage and torque against slip: measured (multibody + sensor noise) vs analytic (orbital prior, learned) vs the zoo surrogate, per unit the rover drove on. |
| `plot_traverse` | função | `(runs, site, path, slip_limit=0.4, front=(0, 1))` | Per run vs distance: the traction monitor's slip, front-wheel sinkage, cumulative drive energy, elevation with the soil unit under the rover. |
| `plot_route_risk` | função | `(planners, routes, path, slip_limit=0.4, risk_max=0.05)` | Predicted slip and risk along every route, under the orbital prior and under the learned belief. |

#### `pinneapple_simulation.rover.autolab.wheel_models`

Wheel-soil laws at the operating point a rover needs, and the three ways to get them.

<small>`pinneapple_simulation/rover/autolab/wheel_models.py` · 128 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GradeTable` | classe | `(grades, slip, sinkage, torque, energy_per_m, max_dp_over_w)` | Operating points of a quasi-static rover on one soil, per grade [deg] (arrays over ``grades``). |
| `grade_table` | função | `(soil, wheel, mass, g=1.62, n_wheels=4, grades=GRADES, slips=SLIPS, n_gl=12) -> GradeTable` | Slip, sinkage, torque and energy per metre a ``mass`` kg rover on ``n_wheels`` needs on each grade. |
| `surrogate_steady_state` | função | `(predict, load, slip, z_max=0.2, iters=50)` | Sinkage with ``Fz(slip, z) = load`` for a ``predict(slip, z) -> (N, 3)`` contact model (bisection), and the forces there. Returns dict like :meth:`WongReeceModel.steady_state`. |
| `compare_contact_models` | função | `(measured, soil_believed, wheel, surrogate=None, g=1.62, in_range=None) -> dict` | Score the analytic law (believed soil) and a surrogate against measured operating points. |
| `max_climbable_grade` | função | `(table, slip_limit) -> float` | Largest grade of the table the rover climbs at a slip at or below ``slip_limit`` [deg] (nan if none). |

#### `pinneapple_simulation.rover.checks`

Verification of the rover model against closed forms, the standalone solver and its own balance laws.

<small>`pinneapple_simulation/rover/checks.py` · 149 linhas · testes: 12</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `single_wheel_check` | função | `(state, rover, soils, rear_angle_ratio=1.0 / 3.0) -> dict` |  |
| `pressure_only` | função | `(soil) -> SoilParams` | The soil's pressure-sinkage law with shear switched off (c = 0, K very large) and the stress peak at the bottom of the wheel (a0 = a1 = 0); use with ``rear_angle_ratio=0`` for front-only contact. |
| `bekker_rigid_wheel_sinkage` | função | `(W, b, D, k, n) -> float` | Bekker's closed form for the static sinkage of a rigid wheel: z = [3 W / ((3 - n) b k sqrt(D))]^(2/(2n+1)). |
| `static_sinkage_check` | função | `(rover=None, soil=None) -> dict` |  |
| `balance_check` | função | `(state, rover) -> dict` | Net force (including gravity, the lateral reaction, the tow load and the d'Alembert force) and net moment about the centre of mass, recomputed from the stored wheel forces; scaled by m g and m g L. |
| `surrogate_vs_numerical` | função | `(states, surrogate, numerical, soils) -> dict` | Errors of ``surrogate`` against ``numerical`` at every loaded (slip, sinkage) of the given states. |
| `dp_grade_estimate` | função | `(rover, soil, s_limit, g=1.62, rear_angle_ratio=1.0 / 3.0, tol=0.01) -> dict` | Grade where six equally loaded wheels at slip ``s_limit`` give drawbar pull = m g sin(a) (bisection). |

#### `pinneapple_simulation.rover.contact`

Wheel-soil force laws for the rover: (slip, sinkage, soil) -> (F_x, F_z, M_y) per wheel.

<small>`pinneapple_simulation/rover/contact.py` · 167 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BekkerWongContact` | classe | `(wheel_radius, wheel_width, rear_angle_ratio=1.0 / 3.0, method='gauss', n_gl=48)` | Bekker-Wong forces for N wheels at once (``soils[k]`` selected per wheel with ``soil_idx``). |
| `RangeLog` | classe | `(n_calls, n_out, slip_out, sinkage_out, soil_mismatch, worst)` | Counts of surrogate inputs inside / outside the training box. |
| `SurrogateContact` | classe | `(predict, slip_range, sinkage_range, soil, wheel_radius, wheel_width)` | Wrap ``predict(slip, z) -> (N, 3)`` trained on ``soil`` for a wheel (R, b) on slip x sinkage box ranges. |

#### `pinneapple_simulation.rover.dynamics`

Rocker-bogie rover on lunar terrain with Bekker-Wong wheel forces.

<small>`pinneapple_simulation/rover/dynamics.py` · 380 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `StepState` | classe | `(t, pose, v, omega, s_cmd, wheel_centre, contact_point, normal, tangent, sinkage, slip, F…)` | Converged state of one step (arrays per wheel in WHEEL_NAMES order). |
| `RoverSimulator` | classe | `(rover, terrain, contact=None, g=LUNAR_G, s_limit=0.95, conv_tol=1e-06)` | Quasi-static rocker-bogie rover with Bekker-Wong wheels on a :class:`LunarTerrain`. |
| `Trajectory` | classe | `(states, rover, mode, failed_at, terrain)` |  |

#### `pinneapple_simulation.rover.geometry`

Parametric six-wheel rocker-bogie rover: parameters, kinematics and meshes (glTF / STL export).

<small>`pinneapple_simulation/rover/geometry.py` · 237 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RoverParams` | classe | `(wheel_radius, wheel_width, n_grousers, grouser_height, track, wheelbase, pivot_height, b…)` | Generic small lunar rover (every value is a generic choice; see the module docstring). |
| `Pose` | classe | `(x, y, z, yaw, pitch, roll, rocker, bogie_l, bogie_r, extra)` | Rover configuration: body origin (rocker pivot axis centre), attitude and suspension angles [m, rad]. |
| `rot_x` | função | `(a)` |  |
| `rot_y` | função | `(a)` |  |
| `rot_z` | função | `(a)` |  |
| `body_rotation` | função | `(yaw, pitch, roll) -> np.ndarray` | Body-to-world rotation. ``pitch`` > 0 is nose up, ``roll`` > 0 is left side up [rad]. |
| `quat_from_matrix` | função | `(R) -> np.ndarray` | (w, x, y, z) of a rotation matrix. |
| `link_frames` | função | `(p, pose) -> dict` | World positions of the pivots and wheel centres plus link rotations. |
| `rover_meshes` | função | `(p) -> dict` | Meshes in their own link frames: ``body`` (body frame), ``rocker_L/R`` (rocker frame, origin at the pivot), ``bogie_L/R`` (bogie frame, origin at the bogie pivot) and ``wheel`` (wheel frame, axle along y). |
| `assembled_mesh` | função | `(p, pose, wheel_spin=None)` | All parts in world coordinates: list of (name, vertices, faces). |
| `export_rover` | função | `(p, path, pose=None) -> str` | Write the rover (level, on its wheels) as .glb or .stl with the studio writer. |

#### `pinneapple_simulation.rover.terrain`

Lunar terrain for rover simulation: a height function with a planar slope, craters and rocks, plus soil regions.

<small>`pinneapple_simulation/rover/terrain.py` · 207 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Crater` | classe | `(x, y, diameter, rim_ratio, depth_ratio, crest_width)` |  |
| `Rock` | classe | `(x, y, radius, height)` |  |
| `SoilRegion` | classe | `(soil, kind, params, name)` | Soil inside a circle (``kind="circle"``: centre, radius) or an axis-aligned box (``"box"``: x0, x1, y0, y1). |
| `LunarTerrain` | classe | `(slope_deg, slope_dir_deg, ramp, craters, rocks, soil, regions)` | Height function and soil map. |
| `random_rocks` | função | `(rng, n, x_range, y_range, radius_range, height_ratio_range, keep_out=None) -> list[Rock]` | ``n`` rocks with uniform positions and radii (no published size-frequency law is used); ``keep_out(x, y)`` returns True where no rock may be placed (e.g. the rover path). |
| `random_craters` | função | `(rng, n, x_range, y_range, diameter_range, depth_ratio_range=(0.05, 0.15), rim_ratio=0.01…)` | ``n`` small degraded craters (scenery): positions uniform, diameters from a power law ``N(>D) ~ D^-slope`` truncated to ``diameter_range``, depth/diameter uniform in ``depth_ratio_range`` (below Pike's 0.2 for fresh craters, since small lunar craters are mostly degraded). ``keep_out(x, y, D)`` … |

#### `pinneapple_simulation.rover.viz`

Plots, animations and the 3D twin view of a rover :class:`~.dynamics.Trajectory`.

<small>`pinneapple_simulation/rover/viz.py` · 225 linhas · testes: 12</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_wheel_series` | função | `(traj, path, title='', x='distance') -> str` |  |
| `plot_rover_series` | função | `(trajs, path, title='') -> str` |  |
| `animate` | função | `(traj, path_gif, *, color_by='sinkage', path_mp4=None, frames=80, window=(1.7, 1.1), fps=…)` | 3D animation following the rover over the terrain (``window``: half sizes of the terrain patch shown around it [m]); wheels coloured by ``color_by`` ("sinkage" [mm] or "slip"). |
| `twin_scene` | função | `(traj, title='Lunar rover', margin=1.5, grid=(160, 100))` | A ``pinneapple_twin3d.Scene`` of the run (y up for the viewer): terrain coloured by height, links and wheels moving with the trajectory, wheel fields ``sinkage`` [mm] and ``slip`` [-] per step. |

#### `pinneapple_simulation.terramechanics`

Terramecânica de rodas de rovers planetários: solos com proveniência, modelo de roda de Wong-Reece e extensões, terreno deformável com sulcos, bancada de roda única com dados publicados e substitutos PINN do contato.

<small>`pinneapple_simulation/terramechanics/__init__.py` · 37 linhas · testes: 34</small>

#### `pinneapple_simulation.terramechanics.mobility`

Triagem de mobilidade de roda com a PINN `bekker_wong_surrogate_2d`: treina, acha o afundamento que sustenta a carga, lê tração na barra contra escorregamento e rechecagem de cada restrição física.

<small>`pinneapple_simulation/terramechanics/mobility.py` · 159 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `contact_entry_angle` | função | `(R_m, sinkage_m) -> float` | Ângulo de entrada θ1 = arccos(1 - z/R) de uma roda rígida afundada z. |
| `mohr_coulomb_traction_limit` | função | `(c_Pa, phi_deg, R_m, b_m, sinkage_m, Fz_N) -> float` | Maior tração na barra permitida pela resistência ao cisalhamento do solo, c A + Fz tan(φ). |
| `train_wheel_surrogate` | função | `(*, c_Pa, phi_deg, R_m, b_m, epochs=2000, n_collocation=1024, architecture='modified_mlp'…)` | Treina a PINN `bekker_wong_surrogate_2d` para um solo e uma roda. |
| `analyze_wheel_surrogate` | função | `(model, spec, *, c_Pa, phi_deg, R_m, b_m, Fz_N, slip_min=0.0, slip_max=0.7, n_slip=15, n_…)` | Tração contra escorregamento no afundamento que sustenta `Fz_N`, com checagem de cada restrição física. |
| `mobility_analysis` | função | `(*, c_Pa=1400.0, phi_deg=30.0, R_m=0.125, b_m=0.06, Fz_N=50.0, slip_min=0.0, slip_max=0.7…)` | Treina, checa as salvaguardas e analisa a mobilidade da roda em uma chamada. |

#### `pinneapple_simulation.terramechanics.soils`

Banco de solos para modelos roda-solo (Bekker, Janosi-Hanamoto, Wong-Reece, escorregamento-afundamento, descarregamento) em que cada parâmetro registra a fonte ('secondary' ou 'estimate'). [sem teste direto]

<small>`pinneapple_simulation/terramechanics/soils.py` · 222 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Soil` | classe | `(name, c, phi_deg, K, k_c, k_phi, n0, rho, n1, a0, a1, c3, Ky, k_u0, A_u, description, so…)` | Parâmetros de um solo com proveniência por parâmetro (`sources`). |
| `register_soil` | função | `(soil) -> Soil` | Registra um solo no banco. |
| `get_soil` | função | `(name) -> Soil` | Busca um solo pelo nome. |
| `list_soils` | função | `() -> list[str]` | Lista os nomes dos solos registrados. |

#### `pinneapple_simulation.terramechanics.surrogate`

Substituto PINN de um modelo de contato roda-solo, (escorregamento, afundamento) -> (tração, carga normal, torque), treinado com termos físicos unilaterais e conjunto de teste separado, salvo como TorchScript + JSON.

<small>`pinneapple_simulation/terramechanics/surrogate.py` · 292 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ContactSurrogate` | classe | `(net, x_lo, x_hi, y_lo, y_hi, slip_range, sinkage_range, meta)` | Rede treinada com sua normalização e caixa de treino; prevê forças, diz se está no domínio e salva. |
| `PhysicsMismatchWarning` | classe | `()` | Aviso de que o substituto foi treinado com uma física diferente do modelo com que é usado. |
| `ZooContactSurrogate` | classe | `(zoo_model)` | Adapta um modelo de contato do model zoo (`pp.zoo.load`) à interface de `ContactSurrogate`. |
| `load_surrogate` | função | `(stem) -> ContactSurrogate` | Carrega um `ContactSurrogate` salvo (TorchScript + JSON). |
| `contact_net` | função | `(hidden=96, depth=4, n_fourier=16, scale=2.0, seed=0)` | Rede MLP residual com features de Fourier aleatórias e tanh, 2 entradas -> 3 saídas. |
| `train_contact_surrogate` | função | `(model, slip_range=(0.0, 0.8), sinkage_range=(0.002, 0.06), *, n_grid=40, n_lhs=2000, n_t…)` | Treina o substituto em grade + hipercubo latino com termos físicos e devolve o relatório de erro no teste separado. |
| `compare_on_states` | função | `(sur, model, slip, z) -> dict` | Compara substituto e referência em estados dados e conta os que estão fora da caixa de treino. |

#### `pinneapple_simulation.terramechanics.terrain`

Terreno deformável para simulação roda-solo: memória plástica de afundamento (sulcos, multipassagem), marcas de garras, mapas de tensão de contato e rochas rígidas.

<small>`pinneapple_simulation/terramechanics/terrain.py` · 341 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RigidRock` | classe | `(x, y, a, b, c, burial, yaw, z0, seed)` | Rocha rígida elipsoidal parcialmente enterrada, com altura, penetração e malha. |
| `DeformableTerrain` | classe | `(base, x_range, y_range, cell=0.01, rebound=None, relief=0.0, relief_seed=0, berm=0.3, wa…)` | Grade de memória de afundamento sobre uma superfície base, com sulcos, bermas, marcas de garras e rochas. |
| `fractal_relief` | função | `(shape, cell, amplitude, seed=0, beta=2.2, min_wavelength=0.05) -> np.ndarray` | Superfície aleatória gaussiana com espectro em lei de potência e RMS dado. |

#### `pinneapple_simulation.terramechanics.testbed`

Bancada de roda única: tração, torque e afundamento contra escorregamento a carga fixa, comparação com medições publicadas embarcadas e calibração por mínimos quadrados com pontos retidos.

<small>`pinneapple_simulation/terramechanics/testbed.py` · 253 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TestbedData` | classe | `(dataset, wheel_kind, wheel, soil, load, slip, drawbar_pull, sinkage, torque, source)` | Pontos medidos de uma roda num solo a uma carga (tração, afundamento, torque, fonte). |
| `ValidationReport` | classe | `(data, model, curve, errors)` | Relatório de validação: dados, modelo, curva e erros. |
| `list_datasets` | função | `() -> list[tuple[str, str]]` | Lista os conjuntos embarcados (dataset, tipo de roda). |
| `load_dataset` | função | `(dataset, wheel, lambda_grouser=1.0) -> TestbedData` | Carrega os pontos embarcados ('ding2011', 'hurrell2025', 'wanghan2016') para roda lisa ou com garras. |
| `run_testbed` | função | `(model, load, slips) -> dict` | Tração, torque, afundamento e DP/W do modelo contra escorregamento a carga fixa. |
| `validate` | função | `(data, soil=None, g=9.81, **model_kw) -> ValidationReport` | Roda a bancada nos escorregamentos medidos e relata os erros contra a medição. |
| `holdout_split` | função | `(data, every=2, offset=0) -> np.ndarray` | Máscara de treino que separa pontos para ajuste e pontos retidos. |
| `calibrate` | função | `(data, params=('n0', 'n1', 'K'), bounds=None, g=9.81, train=None) -> dict` | Ajuste por mínimos quadrados de parâmetros do solo à tração e ao afundamento medidos. |
| `calibrate_holdout` | função | `(data, params=('n0', 'n1', 'K'), every=2, offset=0, g=9.81) -> dict` | Calibração de escorregamento-afundamento com pontos retidos para medir o erro fora do ajuste. |
| `plot_validation` | função | `(reports, path, title='', calibrated=None) -> str` | Gráfico de tração/carga e afundamento contra escorregamento: pontos medidos e curvas do modelo. |

#### `pinneapple_simulation.terramechanics.wheel`

Geometria de roda rígida para terramecânica: raio, largura, garras (retas ou chevron), massa e inércia de giro. [sem teste direto]

<small>`pinneapple_simulation/terramechanics/wheel.py` · 68 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Grousers` | classe | `(count, height, thickness, shape, chevron_angle_deg, shear_radius_factor)` | Garras (lugs) de uma roda rígida: número, altura, espessura, forma e ângulo chevron. |
| `Wheel` | classe | `(radius, width, grousers, mass, spin_inertia)` | Roda rígida: raio, largura, garras, massa e inércia de giro. |

#### `pinneapple_simulation.terramechanics.wong_reece`

Modelo de Wong-Reece roda rígida / solo deformável com escorregamento-afundamento, garras, multipassagem, bulldozing e força lateral. [sem teste direto]

<small>`pinneapple_simulation/terramechanics/wong_reece.py` · 335 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ContactResult` | classe | `(Fx, Fy, Fz, My, theta_f, theta_r, z_eff, bulldozing, j_exit, theta, sigma, tau_x, tau_y)` | Forças e momento por ponto de operação, com as tensões no arco quando pedidas. |
| `WongReeceModel` | classe | `(soil, wheel, g=1.62, bulldozing='lateral', n_gl=24, slip_sinkage='drive')` | Roda de Wong-Reece num solo: tensões normal e cisalhante, forças, regime permanente e configuração física registrável. |
| `bearing_capacity_factors` | função | `(phi_deg) -> tuple[np.ndarray, np.ndarray]` | Fatores de capacidade de carga de Terzaghi (N_c, N_γ) nas formas fechadas de Wong (2008). |
| `bulldozing_resistance` | função | `(soil, depth, width, g)` | Resistência de bulldozing de uma cunha de solo empurrada por uma face. |
| `slip_ratio` | função | `(v_long, omega, radius, v_eps=0.001)` | Escorregamento longitudinal (tração ou frenagem) a partir da velocidade do centro e da rotação da roda. |
| `slip_angle` | função | `(v_long, v_lat, v_eps=0.001)` | Ângulo de deriva de uma roda a partir das componentes da velocidade do centro. |
| `physics_mismatch` | função | `(recorded, current, rtol=1e-09) -> list[str]` | Lista as chaves em que uma configuração física gravada difere da atual. |

#### `pinneapple_simulation.vehicle`

Veículos planetários com rodas: descrição (chassi + módulos de canto) com CAD, modelo multicorpo construído a partir dela e simulador em terreno deformável com rodas de Wong-Reece.

<small>`pinneapple_simulation/vehicle/__init__.py` · 28 linhas · testes: 38</small>

#### `pinneapple_simulation.vehicle.build`

Constrói o modelo multicorpo de um `VehicleSpec`: chassi como base flutuante e, por canto, braço de suspensão, módulo de direção e roda, com massas e inércias tiradas do CAD. [sem teste direto]

<small>`pinneapple_simulation/vehicle/build.py` · 60 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `VehicleModel` | classe | `(spec, tree, susp, steer, wheel, wheel_body, origins)` | Modelo multicorpo de um veículo: spec, árvore, índices de juntas e corpos, com transformações para o CAD. |
| `build_vehicle` | função | `(spec, gravity=1.62) -> VehicleModel` | Constrói o `VehicleModel` com corpos nomeados como no CAD (`chassis`, `arm_<c>`, `steer_<c>`, `wheel_<c>`). |

#### `pinneapple_simulation.vehicle.dashboard`

Dashboard centrado em animação de uma corrida do veículo (sobre `pinneapple_tools.visualization.dashboard`): curvas da bancada, escorregamento por roda, PINN contra Wong-Reece e cena 3D. [sem teste direto]

<small>`pinneapple_simulation/vehicle/dashboard.py` · 240 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `render_frames` | função | `(run, terrain, n=150, *, field=None, clim=None, size=(1280, 720), camera=None, preset='lu…)` | Renderiza `n` quadros distribuídos pela corrida com o renderizador planetário. |
| `vehicle_dashboard` | função | `(run, frames, ks, *, model=None, surrogate=None, title=None, subtitle='', field_label=Non…)` | Monta o dashboard da corrida com gráficos, tiles, painel de parâmetros e animação. |
| `vehicle_scene3d` | função | `(run, ks, terrain=None, cell=0.03, preset='lunar_low_sun') -> dict` | Payload 3D ao vivo: corpos do CAD, suas poses e o terreno final com sulcos. |

#### `pinneapple_simulation.vehicle.scenarios`

Sítios lunares e estudos prontos: terreno plano ou inclinado, cratera com rochas e varredura de rampas em processos paralelos.

<small>`pinneapple_simulation/vehicle/scenarios.py` · 87 linhas · testes: 17</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `flat_site` | função | `(x_range=(-3.0, 12.0), y_range=(-3.0, 3.0), cell=0.01, relief=0.005, seed=2, slope_deg=0.…)` | Regolito plano (ou com inclinação uniforme) com relevo aleatório fino. |
| `crater_site` | função | `(crater=(5.0, 0.5, 4.0), depth_ratio=0.08, rim_ratio=0.015, rock=(2.0, 0.65, 0.16, 0.13, …)` | Cratera degradada com borda e uma rocha sobre regolito. |
| `grade_sweep` | função | `(factory, soil='GRC-1', grades=(0, 5, 10, 15, 20), duration=12.0, speed=0.1, workers=4, c…)` | Sobe rampas uniformes em linha reta em processos paralelos e resume cada caso. |

#### `pinneapple_simulation.vehicle.sim`

Simulação no tempo de um veículo com rodas em terreno deformável: dinâmica multicorpo mais rodas de Wong-Reece, rochas por penalidade, atuadores e seguimento de rota. [sem teste direto]

<small>`pinneapple_simulation/vehicle/sim.py` · 571 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `VehicleRun` | classe | `(model, t, q, v, wheel, joints, chassis, stresses, terrain_events, meta)` | Amostras gravadas de uma corrida (estado, rodas, juntas, chassi, tensões) com distância, energia, resumo, CSV e reprodução do terreno. |
| `VehicleSimulator` | classe | `(model, terrain, soil, g=1.62, soil_damping=0.7, rock_stiffness=300000.0, rock_damping=40…)` | Simulador de veículo em terreno deformável com rodas de Wong-Reece, rochas, atuadores e seguimento de rota. |

#### `pinneapple_simulation.vehicle.spec`

Descrição de veículo compartilhada por CAD, modelo multicorpo e simulador: chassi e módulos de canto (suspensão -> direção -> roda motriz) com limites de junta e de atuador. [sem teste direto]

<small>`pinneapple_simulation/vehicle/spec.py` · 81 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Actuator` | classe | `(torque_limit, speed_limit, kp, ki, kd)` | Limites de torque e velocidade e ganhos de um servo de junta. |
| `CornerModule` | classe | `(name, susp_pivot, susp_axis, steer_origin, steer_axis, wheel_centre, wheel_axis, susp_li…)` | Um canto do veículo: pivô e eixo da suspensão, direção, centro e eixo da roda, limites e atuadores. |
| `VehicleSpec` | classe | `(name, wheel, corners, total_mass, max_speed, sources, cad)` | Descrição completa do veículo: roda, cantos, massa total, velocidade máxima, fontes e CAD. |

#### `pinneapple_simulation.vehicle.viper`

NASA VIPER em escala real a partir de números públicos (fontes em `VIPER_SOURCES`, estimativas marcadas), com CAD paramétrico próprio.

<small>`pinneapple_simulation/vehicle/viper.py` · 323 linhas · testes: 32</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `viper` | função | `(wheel=None, total_mass=450.0) -> VehicleSpec` | `VehicleSpec` do VIPER na altura nominal em terreno plano. |
| `viper_cad` | função | `(spec) -> VehicleCAD` | CAD paramétrico do VIPER (chassi, braços, direção, rodas) com massas. |

#### `pinneapple_simulation.vehicle.viz`

Imagens e animações de um `VehicleRun`: CAD posado pelo estado multicorpo, terreno com sulcos reproduzido a cada instante, tensões de contato em cores, GIF/MP4 e gráficos por roda.

<small>`pinneapple_simulation/vehicle/viz.py` · 352 linhas · testes: 11</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `posed_parts` | função | `(model, q, *, skip=())` | Malhas no mundo de cada peça do CAD para uma posição generalizada `q`. |
| `terrain_at` | função | `(run, terrain, t_end, cache=None) -> DeformableTerrain` | Cópia do terreno com os sulcos e marcas da corrida até `t_end`. |
| `frame_view` | função | `(run, terrain, k, *, field=None, clim=None, half_window=(3.2, 2.4), far_margin=14.0, far_…)` | Monta a vista de superfície de um quadro: veículo posado, terreno próximo e campo distante. |
| `far_surface` | função | `(ter, X, Y, relief=0.03, feather=1.0)` | Altura do solo no campo distante, misturando o terreno deformável com a superfície base. |
| `follow_camera` | função | `(run, k, offset=(-4.4, -5.0, 2.1), look=(0.5, 0.0, 0.75), smooth=15)` | Câmera que segue o veículo com rumo suavizado. |
| `render_frame` | função | `(run, terrain, k, path, *, field=None, clim=None, camera=None, size=(1600, 900), title=No…)` | Renderiza um quadro da corrida em imagem. |
| `animate` | função | `(run, terrain, gif, *, mp4=None, field=None, clim=None, frames=120, fps=15, size=(1280, 7…)` | GIF (e MP4 com ffmpeg) da corrida. |
| `write_gif` | função | `(path, imgs, fps, size=None) -> str` | Grava um GIF animado de quadros RGB, com redução opcional. |
| `beauty_view` | função | `(spec, terrain=None, preset='lunar_south_pole', earth=False, horizon=4000.0, sun_azimuth_…)` | Vista de superfície usada por `beauty`. |
| `beauty` | função | `(spec, path, *, terrain=None, camera=BEAUTY_CAMERA, size=(1600, 900), title='', preset='l…)` | Imagem do veículo na pose de referência sobre um trecho de terreno. |
| `plot_run` | função | `(run, path, title='')` | Gráficos por roda contra distância (afundamento, escorregamento, carga, tração, torque, força lateral) e arfagem/rolagem do chassi. |
| `plot_testbed` | função | `(curves, path, title='')` | Curvas de roda única contra escorregamento. |
| `chassis_rotation` | função | `(run, k)` | Matriz de rotação do chassi na amostra `k`. |

### Exemplos de uso reais do repositório

**Warm start por projeção economiza iterações do PCG** — `tests/test_accelerated_foam.py`

```python
    systems = _sequence()
    ws = warm_start.ProjectionWarmStart(6)
    x_prev = np.zeros(systems[0].n_cells)
    it_plain = it_warm = 0
    for s in systems:
        A = s.to_csr()
        xs = spsolve(A.tocsc(), s.source)
        g = ws.guess(s, x_prev)

        def enorm(e, A=A):
            return float(-(e @ (A @ e)))  # A is negative definite

        assert enorm(g - xs) <= enorm(x_prev - xs) * (1 + 1e-9) + 1e-12
        it_plain += solvers.pcg(s, x_prev, "dic", 1e-8).iterations
        r = solvers.pcg(s, g, "dic", 1e-8)
```

**Máquina SPM: cogging e torque por dois métodos** — `examples/motor_magnetics/01_spm_cogging_and_torque.py`

```python
m = SPMMachine(n_theta=360, slot_opening=2 / 15, slot_width=0.4)  # edges on the 1-degree grid
I_RATED = 30.0  # A peak, 20 turns per slot
summary = {"machine": {"pole_pairs": m.pole_pairs, "slots": m.slots, "length_m": m.length,
                       "airgap_m": m.airgap, "Br_T": m.Br, "n_theta": m.n_theta}}

# --- cogging torque over one period -------------------------------------------------------------
period = m.cogging_period()
th_cog = np.linspace(0.0, period, 16, endpoint=False)
T_cog = m.cogging(th_cog)
check = [math.radians(d) for d in (1.3, 4.1, 9.7)]
cog_pairs = [(math.degrees(t), m.torque(m.solve(t)), m.torque_coenergy(t, (0.0, 0.0, 0.0))) for t in check]
summary["cogging"] = {"period_deg": math.degrees(period), "peak_to_peak_Nm": float(np.ptp(T_cog)),
                      "maxwell_vs_coenergy_Nm": cog_pairs}
```

**Águas rasas na esfera (caso 6 de Williamson)** — `examples/geophysics/01_sphere_and_assimilation.py`

```python
def main(days: float = 4.0, out: str = "sphere_and_assimilation.png") -> dict:
    sh = SphericalHarmonics(42)
    sw = williamson_case6(sh)
    sw.run(days * 86400, 600.0, diagnostics_every=86400)
    d0, d1 = sw.history[0], sw.history[-1]
    drift = {k: (d1[k] - d0[k]) / d0[k] for k in ("mass", "energy", "potential_enstrophy")}
```

**Colapso de coluna granular com GeoMPM** — `examples/geomechanics/01_granular_column_collapse.py`

```python
from pinneapple_simulation.particle_dynamics.geo_mpm import GeoMPM, Sand, box, runout
...
    sim = GeoMPM(domain=(L0 * (2 + 2.5 * a), a * L0 + 0.1), dx=0.01, floor="sticky", walls="slip")
    sim.add_body(box((0, 0), (L0, a * L0)), Sand(friction_deg=PHI))
...
    sim.run(t_end=0.6 + 0.4 * a, save_every=0.05, out=run_dir)
```

**Braço de dois elos no Robot Sim (Newton)** — `examples/nvidia/03_robotsim_pick_and_place/run.py`

```python
    a = TwoLinkArm()
...
    sim = RobotSim.from_spec(a.spec(), ground=False)
...
            run = sim.run(DURATION, dts[-1], q0=q0, controller=pd_tracking(ref, kp, kd, feedforward=feed),
```

**VIPER: CAD e imagem** — `examples/use_cases/terramechanics/lunar_rover_3d.py`

```python
    spec = viper()
...
    spec.cad.export(str(media / "viper.glb"))
    spec.cad.export(str(OUT / "viper.stl"))
    viz.beauty(spec, str(media / "viper_beauty.png"), camera=((4.3, -3.9, 1.75), (0.1, 0.0, 0.95)),
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 185 de 298.
- Funções de teste em arquivos que importam algum módulo do pacote: 1.119.

Itens do catálogo de métodos (`pinneapple_catalog/method_status.json`) cujo código está neste pacote:

| id | método | status | testes contra referência | outros testes |
|---|---|---|---:|---:|
| S1 | Finite differences 1D/2D/3D | validated | 2 | 11 |
| S2 | Finite elements (linear) + nonlinear beam FEM | validated | 6 | 15 |
| S3 | Finite volumes | validated | 2 | 2 |
| S4 | Spectral (Fourier pseudo-spectral) | validated | 3 | 2 |
| S5 | Lattice Boltzmann D2Q9 / D3Q19 (+ Smagorinsky LES) | validated | 2 | 21 |
| S6 | SPH / ISPH / DFSPH | untested | 0 | 0 |
| S7 | Meshfree RBF collocation (Kansa) | validated | 1 | 1 |
| S8 | X-TFC (IVP, PDE, subdomains) | validated | 4 | 8 |
| S9 | Discrete-time PINN / implicit Runge-Kutta Gauss-Legendre | tested | 0 | 2 |
| S10 | Stiff kinetics ODE | validated | 1 | 0 |
| S11 | TVD advection | validated | 1 | 0 |
| S12 | Immersed boundary (FDM) | validated | 1 | 1 |
| S13 | Beam BVP FDM / 3D elasticity FDM | validated | 2 | 0 |
| S14 | Eddy current FDM | validated | 3 | 3 |
| S15 | Bekker-Wong terramechanics | validated | 1 | 19 |
| S16 | N-body (REBOUND) | validated | 1 | 2 |
| S17 | JAX-CFD | tested | 0 | 3 |
| S18 | Particle dynamics: rigid body, MPM, molecular dynamics | tested | 0 | 13 |
| S24 | External solvers: OpenFOAM, FEniCS, ANSYS/CFD formats, MATLAB, Modelica/FMI, MuJoCo, Genesis, TurboDesigner | validated | 1 | 80 |
| S26 | 2D shallow-water finite volumes (MUSCL + HLL, wet/dry, walls, gates, virtual sensors, health checks) | validated | 1 | 3 |
| S27 | ETDRK4 exponential integrator for stiff semilinear PDEs (contour phi-functions, blow-up guard) | validated | 5 | 1 |
| S28 | Stable fluids 2D (semi-Lagrangian advection, spectral projection, vorticity confinement) | validated | 2 | 4 |
| S29 | 2-D A_z magnetostatics, P1 FEM with Newton (nonlinear B-H, magnets, (anti-)periodic constraints) | validated | 1 | 6 |
| S30 | Time-harmonic 2-D eddy currents (A_z, solid conductors with imposed net currents) | validated | 1 | 1 |
| S31 | Surface-PM machine model: Maxwell-stress and co-energy torque, cogging, back-EMF, losses | validated | 1 | 8 |
| S32 | 2-D electrostatics, P1 FEM (capacitance from field energy) | validated | 2 | 0 |
| S33 | Biot-Savart integration of straight filament segments, loop and solenoid fields | validated | 3 | 0 |
| S34 | Mie series for dielectric and perfectly conducting spheres | validated | 2 | 0 |
| S35 | Fresnel coefficients and characteristic matrices of planar multilayers | validated | 2 | 0 |
| S36 | Jiles-Atherton hysteresis model (RK4 along an applied field path) | validated | 1 | 0 |
| S37 | Dispersive media (Drude, Lorentz, Debye) and their auxiliary-differential-equation coefficients | validated | 1 | 0 |
| S40 | Yee FDTD in 1-D/2-D/3-D (CPML, Courant check, Drude/Lorentz/Debye ADE, PEC/PMC/periodic/Bloch, TF/SF plane waves, DFT monitors, sub-cell averaging) | validated | 4 | 6 |
| S41 | Near-to-far-field transform on a closed box (FDTD radiation patterns) | validated | 1 | 0 |

### Limitações conhecidas

- `accelerated_foam` precisa de um OpenFOAM instalado e de um compilador C++ para o plugin `pinneWarmStart`; numa instalação só binária é preciso a árvore de fontes da mesma versão.
- `accelerated_foam.learned` é experimental: o próprio docstring diz que uma rede local só remove resíduo de alta frequência e que o efeito esperado é modesto; pyamg e PyTorch são opcionais.
- Antenas e células periódicas (`antenna.ava`, `periodic.tcda`, `antenna.study.open_study`) dependem do solver externo EMerge (`pip install emerge`); a classe base de `antenna.surrogates` levanta `NotImplementedError` nos métodos abstratos.
- A verificação cruzada `machines.getdp` exige `gmsh` e `getdp` no PATH; FEMM e ElmerFEM só têm os arquivos de entrada gerados (`interop.femm_lua`, `interop.elmer_case`), não são executados.
- `machines.design` importa torch (via `pinneapple_design.design_optimizer`); `mesh.mesh_from_gmsh` precisa do gmsh; figuras precisam de matplotlib.
- Dados de aço e ímã (`machines.steel`) e as dimensões do Prius 2004 vêm do pyleecan; `materials.illustrative_steel` e os dados térmicos padrão de `machines.thermal` são ilustrativos/hipóteses, não dados de um fornecedor.
- `electrostatics` é só planar (sem peso axisimétrico); `eddy` assume meios lineares.
- Vários módulos não têm teste direto: `accelerated_foam.benchmark`, `antenna.plots`, `eddy`, `machines.mesher`, `machines.pmsm`, `machines.steel`, `reflector.study_report_figures`, `reflector.study_report_text`, `geophysics.energy_balance` e `geophysics.richards`.
- `robot_sim` depende do NVIDIA Newton/Warp (`newton_available`); o `IsaacSimBackend` só escreve um script e levanta erro fora de uma máquina com Isaac Sim e GPU RTX.
- `ParticleSystem.forward` levanta NotImplementedError (classe base); o docstring de `particles` cita DEM, mas só o SPH está implementado.
- Vários módulos centrais não têm teste direto: `multibody.tree`, `terramechanics.wong_reece`, `terramechanics.soils`, `vehicle.sim`, `vehicle.build`, `vehicle.spec`, `vehicle.dashboard`, `robot_sim.sim`, `robot_sim.spec`.
- Os solos em `terramechanics.soils` trazem valores 'secondary' (não conferidos com a tabela original) ou 'estimate' (estimativa de engenharia), segundo o próprio docstring.
- O VIPER é um modelo paramétrico a partir de números públicos; o docstring avisa que os PDFs da NASA não puderam ser abertos e que alguns valores são estimativas.
- `read_particle_frames` com `allow_pickle=True` só deve ser usado com arquivos confiáveis (GeoTaichi grava objetos em pickle).
- Os substitutos PINN de contato só valem dentro da caixa de treino; fora dela `compare_on_states` e `in_range` sinalizam, e `PhysicsMismatchWarning` avisa física diferente.
