# Volume 1 — Núcleo, porta de entrada e catálogo

> Parte do [Mapa completo do PINNeAPPle](../mapa_completo.md). Gerado por `docs/pt/_mapa/gerar.py` a partir do inventário do commit `e53cdcfb04`.

A porta de entrada `pp.*`, os primitivos do núcleo, o catálogo de métodos, o registro de artefatos e os shims de compatibilidade.

## Sumário

1. [`pinneapple`](#1-pinneapple)
2. [`pinneapple_core`](#2-pinneapple_core)
3. [`pinneapple_catalog`](#3-pinneapple_catalog)
4. [`pinneapple_registry`](#4-pinneapple_registry)
5. [`pinneapple_orchestration`](#5-pinneapple_orchestration)
6. [`pinneapple_models`](#6-pinneapple_models)
7. [`pinneapple_train`](#7-pinneapple_train)
8. [`pinneapple_solvers`](#8-pinneapple_solvers)

## 1. `pinneapple`

### Objetivo

_Sem texto de objetivo._

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 3 | 866 | 0 | 2 | 431 (39 arquivos) | 2 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `check` |  | 2 | 0 | 0 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple`

pinneapple — Physics-Informed Neural Networks for Engineering and Research.

<small>`pinneapple/__init__.py` · 854 linhas · testes: 431</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `quickstart` | função | `(problem_id='burgers_1d', **problem_kwargs)` | Interactive quickstart that summarises a problem preset and its fields. |
| `info` | função | `()` | Print library version, device info, and available presets/models. |

#### `pinneapple.check`

``pinneapple.check``: the physics linter (see :mod:`pinneapple_check`). ``python -m pinneapple.check [paths]``.

<small>`pinneapple/check/__init__.py` · 4 linhas · **sem teste direto**</small>

#### `pinneapple.check.__main__`

``python -m pinneapple.check [paths]`` is ``pinneapple check [paths]``.

<small>`pinneapple/check/__main__.py` · 8 linhas · **sem teste direto**</small>

### Exemplos de uso reais do repositório

_Nenhum exemplo encontrado no repositório para este pacote._

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 1 de 3.
- Funções de teste em arquivos que importam algum módulo do pacote: 431.

### Limitações conhecidas

- _Nenhuma registrada._

## 2. `pinneapple_core`

### Objetivo

O `pinneapple_core` reúne as primitivas compartilhadas da biblioteca: `Domain` (a região do problema), `Geometry` (um domínio com fronteiras nomeadas e um gerador de malha), `Mesh` (malha simplicial 1D/2D/3D com cálculo P1 exato) e `Field`/`FunctionField` (valores de uma grandeza numa grade, malha, nuvem de pontos ou numa função torch, como uma PINN). Todo campo responde `gradient()`, `divergence()`, `interpolate(x)` e `integrate()`, seja qual for a discretização. O módulo `operators` oferece uma função por operador (`grad`, `div`, `curl`, `laplacian`, `jacobian`, `hessian`, `integrate`, `flux`) que funciona igual em todas essas representações.

Sobre essas primitivas, o pacote traz as peças de treino e otimização usadas pelo resto do projeto: `pp.func` (transformações de `torch.func` com `wrt` por nome e diferenciação implícita através de solvers), `pp.loss` (termos de perda funcionais e um `Balancer` com estratégias como GradNorm, NTK e ReLoBRaLo), `pp.transforms` (transformações invertíveis com proveniência, inclusive adimensionalização de um `PhysicalProblem`), `data` (`PhysicsDataset`, `DataLoader` e amostradores de colocação, fronteira, malha, trajetória, adaptativo e ativo), `optim` (`PhysicsOptimizer` com SLSQP, penalidade, Lagrangiano aumentado, evolução diferencial etc.), `module` (`PhysicsModule`, `Sequential`, `SolverModule` e `Hybrid` para compor redes e solvers) e `fem` (elementos finitos P1 diferenciáveis em torch, para gradientes de forma).

O pacote `pinneapple` (o `pp.*`) reexporta boa parte disso de forma preguiçosa: `pp.loss`, `pp.func`, `pp.transforms`, `pp.PhysicsOptimizer`, `pp.PhysicsModule`, `pp.grad`/`pp.div`/`pp.laplacian`, `pp.use_backend` e o `PhysicsBackend` (mesma API sobre torch e jax). O `__init__` carrega os submódulos que dependem de torch só no primeiro uso.

Três subpacotes atendem o resto do ecossistema. `onboarding` é a primeira hora do usuário: `pp.doctor` (checa a instalação por perfil), `pp.recommend` (frase em inglês ou português vira preset, método e snippet, por pontuação de palavras-chave), `card`/catálogo de presets de `pinneapple_physics` e `pp.new` (gera um projeto independente). `service` dá às apps em `apps/*` e ao servidor MCP fila de jobs em SQLite, chaves de API, medição de uso e uploads, só com a biblioteca padrão. `workflows` orquestra simulações como um DAG de passos (adaptadores `command`, `python`, `openfoam`, `calculix`; executores `local`, `docker`, `slurm`), com retomada, cache por hash de conteúdo e geração de datasets a partir das execuções.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 34 | 8.747 | 71 | 92 | 233 (20 arquivos) | 15 |

Pasta de testes dedicada: `tests/pinneapple_core` (132 funções de teste).

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `backend` | `PhysicsBackend`: uma API de arrays, derivadas e solves sobre torch, jax ou outro motor registrado. | 1 | 7 | 5 |
| `data` | API de dados física: `PhysicsDataset`, `DataLoader` e amostradores (colocação, fronteira, malha, trajetória, adaptativo, ativo). | 1 | 10 | 25 |
| `domain` | `Domain`: caixa alinhada aos eixos, opcionalmente cortada por uma função distância com sinal. | 1 | 1 | 0 |
| `fem` | Elementos finitos P1 densos e diferenciáveis em torch (rigidez, massa, Poisson) para gradientes de forma. | 1 | 5 | 25 |
| `field` | `Field`, `Grid`, `PointCloud` e `FunctionField`: valores numa grade, malha, nuvem ou função, com gradiente, integral e interpolação. | 1 | 4 | 0 |
| `func` | `pp.func`: `grad`/`jacobian`/`hessian`/`vmap` com `wrt` por nome, e diferenciação implícita (`implicit_solve`). | 1 | 8 | 10 |
| `geometry` | `Geometry`: um `Domain` com fronteiras nomeadas e um gerador de malha. | 1 | 1 | 0 |
| `jsonable` | Conversão de resultados para JSON seguro (NumPy para Python, NaN/Inf para `None`, arredondamento). | 1 | 2 | 8 |
| `loss` | `pp.loss`: termos de perda funcionais e o `Balancer` de pesos escolhido por nome. | 1 | 11 | 22 |
| `mesh` | `Mesh`: malhas simpliciais (intervalos, triângulos, tetraedros) com cálculo P1. | 1 | 1 | 0 |
| `module` | `PhysicsModule` e composição de redes e solvers (`Sequential`, `SolverModule`, `Hybrid`). | 1 | 5 | 7 |
| `onboarding` | A primeira hora: `doctor`, `recommend`, catálogo de presets (`card`) e `new_project`, com CLI. | 6 | 32 | 34 |
| `operators` | Operadores físicos (`grad`, `div`, `curl`, `laplacian`, `jacobian`, `hessian`, `integrate`, `flux`) para todas as representações. | 1 | 8 | 0 |
| `optim` | `PhysicsOptimizer`: parâmetros, objetivo e restrições escritos uma vez e minimizados por várias famílias de otimizador. | 1 | 6 | 8 |
| `service` | Peças de runtime para servidores de apps (fila de jobs, chaves de API, medição de uso, uploads), só com a biblioteca padrão. | 6 | 18 | 30 |
| `transforms` | `pp.transforms`: transformações invertíveis e componíveis para tabelas, datasets e problemas, com proveniência. | 1 | 10 | 13 |
| `workflows` | Workflows de simulação como DAG de passos, com executores e adaptadores plugáveis, execuções rastreáveis e datasets a partir das execuções. | 7 | 34 | 64 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_core`

Ponto de entrada do pacote: exporta `Domain`, `Geometry`, `Mesh`, `Field`, `FunctionField`, `Grid`, `PointCloud` e os operadores, e carrega `func`, `fem`, `loss`, `optim`, `data`, `transforms` etc. preguiçosamente no primeiro acesso.

<small>`pinneapple_core/__init__.py` · 49 linhas · testes: 79</small>

#### `pinneapple_core.backend`

Define `PhysicsBackend`, interface única de arrays, derivadas e solves, com implementações para torch e jax (float64 por padrão) e um registro para novos motores; a classe base levanta `NotImplementedError` nos métodos abstratos.

<small>`pinneapple_core/backend.py` · 233 linhas · testes: 5 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsBackend` | classe | `(name, xp)` | Interface base que todo motor implementa (arrays, `solve`, `integrate`, `grad`, `jacobian`, `vmap`, `jit`); os métodos abstratos levantam `NotImplementedError`. |
| `TorchPhysicsBackend` | classe | `() -> None` | Implementação de `PhysicsBackend` sobre torch. |
| `JaxPhysicsBackend` | classe | `() -> None` | Implementação de `PhysicsBackend` sobre jax, com `jax_enable_x64` ligado para casar com os outros motores. |
| `register_backend` | função | `(name, cls) -> None` | Registra uma classe de backend sob um nome para `set_backend`/`use_backend`; instanciada sob demanda, uma vez. |
| `list_backends` | função | `() -> list` | Lista os nomes de backends registrados. |
| `get_physics_backend` | função | `(name=None) -> PhysicsBackend` | Devolve o objeto backend pelo nome ou o backend ativo. |
| `use_backend` | função | `(name)` | Gerenciador de contexto que roda um bloco num backend e restaura o anterior. |

#### `pinneapple_core.data`

API de dados para PINN, operador neural e problema inverso: um `DataLoader` recebe datasets e amostradores e gera um `Batch` por passo, marcando `requires_grad` nos pontos de colocação.

<small>`pinneapple_core/data.py` · 432 linhas · testes: 25 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Batch` | classe | `(groups, step, meta)` | Dados de um passo de treino, um grupo por fonte (`batch["col"]["x"]`), com atalhos `x`/`y` quando há uma só fonte. |
| `PhysicsDataset` | classe | `(inputs, targets=None, *, coords=None, fields=None, domain=None, name='', provenance=None…)` | Dataset rotulado `inputs[i] -> targets[i]` com nomes de coordenadas e campos, domínio e proveniência; suporta `split`. |
| `Sampler` | classe | `(coords)` | Base dos amostradores que geram pontos sob demanda; subclasses implementam `sample` (a base levanta `NotImplementedError`). |
| `CollocationSampler` | classe | `(domain, strategy='uniform', seed=0) -> None` | Pontos interiores para o resíduo da PDE, com estratégia `uniform`, `lhs` ou `sobol`. |
| `BoundarySampler` | classe | `(domain, part=None, seed=0) -> None` | Pontos de fronteira com normais unitárias externas, na fronteira toda ou numa parte nomeada de uma `Geometry`. |
| `MeshSampler` | classe | `(mesh, mode='nodes', seed=0) -> None` | Pontos ligados a uma malha: nós, centros de células ou pontos aleatórios dentro das células. |
| `TrajectorySampler` | classe | `(states, n_in=1, n_out=1, stride=1, seed=0) -> None` | Janelas de trajetórias `(B, T, d)`: `n_in` estados como entrada e os `n_out` seguintes como alvo. |
| `AdaptiveSampler` | classe | `(base, residual_fn, pool_size=20000, exponent=1.0, uniform_fraction=0.2, seed=0) -> None` | Amostragem adaptativa pelo resíduo: sorteia de um conjunto de candidatos com probabilidade crescente no resíduo, mais uma fração uniforme. |
| `ActiveSampler` | classe | `(base, score_fn, n_candidates=2000) -> None` | Seleção por aquisição: sorteia candidatos e devolve os de maior `score_fn`. |
| `DataLoader` | classe | `(sources, batch_size=256, steps=None, shuffle=True, drop_last=False, seed=0, physics_awar…)` | Gera lotes de datasets e amostradores; com `physics_aware` liga `requires_grad`, ajusta dtype/dispositivo e anexa nomes de coordenadas. |

#### `pinneapple_core.domain`

`Domain` como caixa em qualquer dimensão ou caixa cortada por SDF (formas CSG 2D de `pinneapple_design`): pertinência, volume e amostragem de interior e fronteira; fronteira de domínios não-caixa só em 2D (`NotImplementedError`). [sem teste direto]

<small>`pinneapple_core/domain.py` · 190 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Domain` | classe | `(lo, hi, *, sdf=None, shape=None) -> None` | Região de R^d: caixa ou caixa cortada por SDF, com pertinência, volume e amostragem de interior e fronteira. |

#### `pinneapple_core.fem`

Elementos finitos P1 em torch com matrizes densas, de modo que a solução é diferenciável em relação às coordenadas dos vértices; pensado para malhas de até alguns milhares de nós.

<small>`pinneapple_core/fem.py` · 110 linhas · testes: 25</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `cell_geometry` | função | `(points, cells) -> tuple` | Volumes e gradientes baricêntricos de cada célula, diferenciáveis nas coordenadas dos pontos. |
| `stiffness` | função | `(points, cells, kappa=1.0) -> Tensor` | Matriz de rigidez densa `(N, N)` com coeficiente `kappa`. |
| `mass` | função | `(points, cells) -> Tensor` | Matriz de massa consistente densa `(N, N)`. |
| `integrate_p1` | função | `(points, cells, values) -> Tensor` | Integral do interpolante P1 de valores nodais, diferenciável nos pontos e nos valores. |
| `solve_poisson` | função | `(points, cells, source, dirichlet_nodes, dirichlet_values=0.0, kappa=1.0) -> Tensor` | Resolve `-div(kappa grad u) = source` com Dirichlet em nós dados, diferenciável pela solução linear. |

#### `pinneapple_core.field`

`Field` liga valores ao suporte (`Grid`, `Mesh` ou `PointCloud`) e oferece gradiente, divergência, interpolação e integral; `FunctionField` dá a mesma interface para uma função torch via autograd. [sem teste direto]

<small>`pinneapple_core/field.py` · 407 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Grid` | classe | `(axes) -> None` | Grade produto tensorial definida por eixos 1D estritamente crescentes. |
| `PointCloud` | classe | `(points, *, domain=None, weights=None, k_neighbors=None) -> None` | Pontos dispersos com pesos de quadratura opcionais e vizinhos para derivadas. |
| `Field` | classe | `(values, support, name=None) -> None` | Valores de uma grandeza numa `Grid`, `Mesh` ou `PointCloud`, com gradiente, divergência, interpolação e integral. |
| `FunctionField` | classe | `(fn, domain=None, dim=None) -> None` | Campo definido por uma função torch (expressão analítica ou rede, como numa PINN), com derivadas por autograd. |

#### `pinneapple_core.func`

Transformações de função (`grad`, `jacrev`, `jacfwd`, `jacobian`, `hessian`, `vmap`) sobre `torch.func` com `wrt` por nome ou posição, e `implicit_solve`, que diferencia através de um solver por uma solução adjunta.

<small>`pinneapple_core/func.py` · 180 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `grad` | função | `(fn, wrt=0, *, has_aux=False) -> Callable` | Gradiente de uma função escalar em relação ao(s) argumento(s) `wrt`, por nome ou posição. |
| `jacrev` | função | `(fn, wrt=0, *, has_aux=False) -> Callable` | Jacobiana por modo reverso (barata com poucas saídas). |
| `jacfwd` | função | `(fn, wrt=0, *, has_aux=False) -> Callable` | Jacobiana por modo direto (barata com poucas entradas). |
| `jacobian` | função | `(fn, wrt=0, *, mode='auto') -> Callable` | Jacobiana com `mode` `rev`, `fwd` ou `auto`. |
| `hessian` | função | `(fn, wrt=0) -> Callable` | Hessiana de uma função escalar em relação a `wrt`. |
| `vmap` | função | `(fn, in_dims=0, out_dims=0, **kwargs) -> Callable` | Vetoriza uma função sobre uma dimensão de lote (`torch.func.vmap`). |
| `newton_solve` | função | `(residual, x0, *params, tol=1e-10, max_iter=50) -> Tensor` | Iteração de Newton amortecida para `residual(x, *params) = 0` com Jacobiana densa. |
| `implicit_solve` | função | `(residual, x0, *params, solver=None, tol=1e-10, max_iter=50) -> Tensor` | Resolve `residual(x, *params) = 0` e diferencia pela solução com o teorema da função implícita (uma solução adjunta no backward). |

#### `pinneapple_core.geometry`

`Geometry` acrescenta fronteiras nomeadas (`x_min`, `x_max`, ...) e geração de malha a um `Domain`; malhar domínios não-caixa só funciona em 2D (`NotImplementedError`). [sem teste direto]

<small>`pinneapple_core/geometry.py` · 114 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Geometry` | classe | `(domain, boundaries=None) -> None` | Um `Domain` com partes de fronteira nomeadas e um método `mesh`. |

#### `pinneapple_core.jsonable`

Implementação única da conversão de resultados para JSON usada por apps, servidor MCP, lab e ferramentas de dados: escalares e arrays NumPy viram Python, NaN/Inf viram `None`.

<small>`pinneapple_core/jsonable.py` · 62 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `round_sig` | função | `(x, digits=6, small=0.001) -> float` | Arredonda por casas decimais para valores grandes e por dígitos significativos para valores pequenos. |
| `to_jsonable` | função | `(obj, *, digits=None, nonfinite=None, drop_private=False, unknown='keep') -> Any` | Devolve uma cópia de um objeto que `json.dumps` aceita, com opção de arredondar e tratar não-finitos. |

#### `pinneapple_core.loss`

Termos de perda como funções simples (`pde`, `boundary`, `supervised`, `conservation`, `energy`, `symmetry`, `inverse`), soma ponderada (`combine`) e o `Balancer` de pesos dinâmicos escolhido por nome.

<small>`pinneapple_core/loss.py` · 266 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Balancer` | classe | `(strategy='fixed', names=None, model=None, weights=None, update_every=100, *, schedule=No…)` | Ponderação de perdas escolhida por nome de estratégia (fixa, aprendida, GradNorm, NTK, ReLoBRaLo, SoftAdapt, currículo etc.). |
| `causal_weights` | função | `(residual, t, n_chunks=32, epsilon=1.0) -> Tensor` | Pesos causais por ponto, que decaem com o resíduo acumulado de fatias de tempo anteriores. |
| `pde` | função | `(residual, *, weights=None, t=None, causal_epsilon=None, n_chunks=32) -> Tensor` | Resíduo quadrático médio, com pesos opcionais e versão causal no tempo. |
| `boundary` | função | `(pred, target=0.0) -> Tensor` | Erro quadrático médio em relação aos valores de contorno ou inicial. |
| `supervised` | função | `(pred, target, kind='mse', *, delta=1.0) -> Tensor` | Ajuste a dados com `mse`, `mae`, `huber` ou `relative_l2`. |
| `conservation` | função | `(quantity, target=0.0, scale=None) -> Tensor` | Desvio quadrático de uma grandeza conservada em relação ao alvo, normalizado por escala. |
| `energy` | função | `(e, reference=None, mode='conserve') -> Tensor` | Penalidade sobre a energia ao longo do tempo: conservada ou não crescente. |
| `symmetry` | função | `(model, x, transform, output_transform=None) -> Tensor` | Penalidade por quebrar uma simetria: `mean((f(T x) - S f(x))^2)`. |
| `inverse` | função | `(pred, observed, params=None, prior=None, reg=0.0) -> Tensor` | Desajuste às observações mais regularização dos parâmetros desconhecidos em direção a um prior. |
| `combine` | função | `(terms, weights=None) -> Tensor` | Soma ponderada de termos nomeados; peso para termo ausente levanta `KeyError`. |
| `list_strategies` | função | `() -> List[str]` | Lista os nomes de estratégias aceitos pelo `Balancer`. |

#### `pinneapple_core.mesh`

`Mesh` com pontos `(N, d)` e células `(M, d+1)`; campos nodais P1, integral exata para dados P1 e gradiente exato para dados lineares. [sem teste direto]

<small>`pinneapple_core/mesh.py` · 291 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Mesh` | classe | `(points, cells) -> None` | Malha simplicial não estruturada em R^d (d = 1, 2 ou 3) com construtores estruturados e cálculo P1. |

#### `pinneapple_core.module`

`PhysicsModule` é o `nn.Module` da física: redes, solvers e híbridos se compõem com `Sequential`, `SolverModule` e `Hybrid`, e têm um `fit` simples.

<small>`pinneapple_core/module.py` · 234 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsModule` | classe | `(kind)` | Classe base de modelos e solvers com `kind`, `describe()`, `num_parameters()` e um `fit` simples. |
| `Lambda` | classe | `(fn) -> None` | Envolve uma função simples sem parâmetros como componente. |
| `Sequential` | classe | `(*parts) -> None` | Encadeia componentes (módulos, solvers ou funções), a saída de um é a entrada do próximo. |
| `SolverModule` | classe | `(fn, params=None, trainable=(), name=None) -> None` | Um solver como componente, com parâmetros treináveis opcionais; `from_method` usa um backend de `pp.solve` (não diferenciável). |
| `Hybrid` | classe | `(solver, corrector, mode='residual', corrector_input=None, detach_solver=False) -> None` | Solver mais correção neural (por exemplo modo `residual`), treinável através do solver. |

#### `pinneapple_core.onboarding`

Subpacote da primeira hora com a biblioteca: reexporta `doctor`, `recommend`, `card` e `new_project`.

<small>`pinneapple_core/onboarding/__init__.py` · 38 linhas · testes: 34</small>

#### `pinneapple_core.onboarding.catalog`

Catálogo de presets: completa o registro de `pinneapple_physics.pde_environment.presets` com descrição, tags, método que roda hoje e um snippet executável; a busca é por palavras-chave (inglês ou português), não um modelo.

<small>`pinneapple_core/onboarding/catalog.py` · 817 linhas · testes: 34</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Card` | classe | `(name, domain, description, tags, coordinates, fields, spatial_dim, transient, domain_bou…)` | Descrição de um preset: domínio, coordenadas, campos, condições, solver, método, snippet, presets relacionados e disponibilidade. |
| `best_method` | função | `(name) -> str` | O método mais barato que roda hoje para o preset: analítico, senão solver de referência, senão rede treinada. |
| `needs_geometry_tags` | função | `(name) -> bool` | Diz se o preset liga condições de contorno a tags de geometria nomeadas. |
| `tag_names` | função | `(name) -> Tuple[str, ...]` | Tags de geometria citadas pelas condições do preset. |
| `default_geometry_note` | função | `(name) -> str` | Nota sobre qual geometria canônica o preset fornece para as fronteiras com tag, ou string vazia. |
| `trainable` | função | `(name) -> bool` | Diz se dá para treinar uma rede neste preset só pelo nome. |
| `probe` | função | `(name) -> Tuple[bool, str]` | Tenta uma vez o método mais barato do preset e informa se funcionou, com o motivo. |
| `runnable` | função | `(name) -> bool` | Diz se o preset produz um campo nesta máquina agora, sem geometria extra nem solver a instalar. |
| `working_method` | função | `(name) -> str` | O método que funciona aqui para o preset (o sondado após fallback, senão o preferido). |
| `solver_available` | função | `(solver) -> bool` | Diz se esta máquina consegue rodar o solver de referência nomeado. |
| `solver_requirement` | função | `(solver) -> str` | Comando de instalação de um solver de referência ausente, ou string vazia. |
| `snippet` | função | `(name, method=None, *, train=False) -> str` | Menor programa correto para um preset, executável como impresso. |
| `standalone_snippet` | função | `(name) -> str` | Programa que resolve a mesma física do preset descrita do zero. |
| `card` | função | `(name, *, with_related=True, check=True) -> Card` | Monta o `Card` de um preset; nome desconhecido levanta `KeyError` com sugestões próximas. |
| `describe` | função | `(name) -> Dict[str, Any]` | O `card` de um preset como dicionário pronto para JSON. |
| `catalogue` | função | `() -> List[Card]` | Todos os presets registrados como cards, ordenados por domínio e nome. |
| `domains` | função | `() -> Dict[str, List[str]]` | Nomes de presets agrupados por domínio. |
| `format_card` | função | `(c, width=78) -> str` | Formata um card como texto simples, sem cor nem caixas. |
| `search` | função | `(text, top=5, *, min_score=0.0) -> List[Dict[str, Any]]` | Ordena presets contra um texto livre (inglês ou português) por pontuação de palavras-chave. |

#### `pinneapple_core.onboarding.cli`

Comandos `pinneapple doctor | recommend | card | new`, todos offline e com saída de texto simples.

<small>`pinneapple_core/onboarding/cli.py` · 153 linhas · testes: 34</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `doctor_main` | função | `(argv) -> int` | Implementa o subcomando `pinneapple doctor`. |
| `recommend_main` | função | `(argv) -> int` | Implementa o subcomando `pinneapple recommend`. |
| `card_main` | função | `(argv) -> int` | Implementa o subcomando `pinneapple card`. |
| `new_main` | função | `(argv) -> int` | Implementa o subcomando `pinneapple new`. |
| `main` | função | `(argv=None) -> int` | Despacha `pinneapple <doctor\|recommend\|card\|new> ...`. |

#### `pinneapple_core.onboarding.doctor`

`pp.doctor`: verifica a instalação contra um perfil (`start`, `cfd`, `research`, `enterprise`, `all`) e imprime um comando de correção por falha.

<small>`pinneapple_core/onboarding/doctor.py` · 319 linhas · testes: 34</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Check` | classe | `(name, ok, detail, fix, required)` | Um item verificado, com resultado (`None` se não pôde rodar), detalhe, correção e se é obrigatório. |
| `DoctorReport` | classe | `(profile, checks)` | Relatório do `doctor` para um perfil, com a lista de checagens e as falhas. |
| `profiles` | função | `() -> Dict[str, str]` | Perfis aceitos pelo `doctor` e a frase que descreve cada um. |
| `doctor` | função | `(profile='start', *, verbose=True) -> DoctorReport` | Verifica a instalação contra um perfil e, com `verbose`, imprime o relatório. |

#### `pinneapple_core.onboarding.newproject`

`pp.new` / `pinneapple new`: escreve um projeto independente deste repositório, com modelos `plain` (só YAML), `research` e `cfd`.

<small>`pinneapple_core/onboarding/newproject.py` · 842 linhas · testes: 34</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plan` | função | `(name, template='plain', preset='burgers_1d') -> Dict[str, Any]` | Mostra o que `new_project` escreveria, sem escrever (ensaio). |
| `new_project` | função | `(name, *, dir=None, template='plain', preset='burgers_1d', verbose=True) -> str` | Escreve um projeto independente a partir de um template e preset e devolve a pasta. |

#### `pinneapple_core.onboarding.recommend`

`pp.recommend`: transforma uma frase em preset, método e modelo, com o motivo e os termos que casaram; é pontuação por palavras-chave e pode errar. [sem teste direto]

<small>`pinneapple_core/onboarding/recommend.py` · 299 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Recommendation` | classe | `(query, preset, method, model, confidence, reason, snippet, matched, alternatives, next_s…)` | Resultado do `recommend`: preset, método, modelo, confiança, motivo, snippet, termos casados, alternativas e próximos passos. |
| `recommend` | função | `(text, *, top=3) -> Recommendation` | Transforma a descrição de um problema em preset, método e modelo, com os motivos. |

#### `pinneapple_core.operators`

Operadores físicos com a mesma chamada em campo contínuo (autograd), grade, malha simplicial e nuvem de pontos; em `Field` devolvem `Field`, em funções devolvem tensores com grafo. [sem teste direto]

<small>`pinneapple_core/operators.py` · 275 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `grad` | função | `(f, x=None)` | Gradiente: `(N, d)` para escalar, `(N, c, d)` para `c` componentes. |
| `div` | função | `(f, x=None)` | Divergência de um campo vetorial. |
| `jacobian` | função | `(f, x=None)` | Jacobiana `(N, c, d)` de um campo. |
| `curl` | função | `(f, x=None)` | Rotacional: vetor em 3D, escalar (componente z) em 2D. |
| `hessian` | função | `(f, x=None)` | Hessiana `(N, d, d)` de um campo escalar. |
| `laplacian` | função | `(f, x=None)` | Soma das segundas derivadas, por componente em campos vetoriais. |
| `integrate` | função | `(f, **kwargs)` | Integral sobre o suporte (trapézio na grade, P1 exato na malha, quadratura ou Monte Carlo na nuvem). |
| `flux` | função | `(f, where=None, *, domain=None, n=20000, seed=0)` | Fluxo de um campo vetorial pela fronteira, `integral f . n dS`. |

#### `pinneapple_core.optim`

`PhysicsOptimizer`: o problema (parâmetros com limites, objetivo, restrições de igualdade e desigualdade) é escrito uma vez e `minimize(method)` escolhe a família (SLSQP, adjunto, penalidade, Lagrangiano aumentado, evolução diferencial etc.).

<small>`pinneapple_core/optim.py` · 395 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Constraint` | classe | `(fn, kind, name)` | Restrição `fn(x) <= 0` (`ineq`) ou `fn(x) == 0` (`eq`). |
| `OptimizeResult` | classe | `(x, params, fun, violation, feasible, n_evals, method, family, success, message, wall_tim…)` | Resultado de `minimize`: ponto, objetivo, violação, viabilidade, avaliações, método, tempo e histórico. |
| `PhysicsOptimizer` | classe | `(parameters, objective, constraints=(), *, backend='torch', argument='vector', tol=0.001)…)` | Parâmetros com limites, objetivo e restrições, minimizados pelo método escolhido em `minimize`. |
| `ineq` | função | `(fn, name='') -> Constraint` | Cria uma restrição de desigualdade `fn(x) <= 0`. |
| `eq` | função | `(fn, name='') -> Constraint` | Cria uma restrição de igualdade `fn(x) == 0`. |
| `list_methods` | função | `() -> List[str]` | Lista os métodos aceitos por `PhysicsOptimizer.minimize`. |

#### `pinneapple_core.service`

Peças de runtime para servidores (apps, servidor MCP, serviços próprios) só com a biblioteca padrão; as rotas FastAPI ficam em `apps/_shared/appkit.py`.

<small>`pinneapple_core/service/__init__.py` · 45 linhas · testes: 30</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `data_dir` | função | `(app, prefix=None, environ=None) -> Path` | Diretório de estado de uma app: `<PREFIX>_DATA_DIR`, senão `$PINNEAPPLE_APP_DATA/<app>`, senão `/data/<app>`. |

#### `pinneapple_core.service.apikeys`

Chaves de API criadas uma vez (a chave em texto só aparece na criação), guardadas como SHA-256 e revogáveis; clientes enviam `X-API-Key`. [sem teste direto]

<small>`pinneapple_core/service/apikeys.py` · 84 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `APIKeyStore` | classe | `(db_path, app='')` | Store SQLite de chaves de API: criar, listar, verificar e revogar. |
| `hash_key` | função | `(key) -> str` | SHA-256 de uma chave, usado para guardar e procurar. |

#### `pinneapple_core.service.db`

Utilitários SQLite dos stores de serviço: uma conexão por thread, WAL e transações `BEGIN IMMEDIATE`. [sem teste direto]

<small>`pinneapple_core/service/db.py` · 47 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LocalDB` | classe | `(db_path, schema='')` | Uma conexão SQLite por thread para o mesmo arquivo, com esquema aplicado. |
| `connect` | função | `(db_path) -> sqlite3.Connection` | Abre conexão em autocommit com WAL e timeout de espera. |
| `transaction` | função | `(conn)` | Gerenciador de contexto de transação `BEGIN IMMEDIATE` com rollback em erro. |

#### `pinneapple_core.service.jobs`

Fila de jobs persistente em SQLite com workers em thread ou processo, progresso, logs, cancelamento cooperativo, novas tentativas e retomada após reinício. [sem teste direto]

<small>`pinneapple_core/service/jobs.py` · 510 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `JobCancelled` | classe | `()` | Levantada dentro de um job por `ctx.progress`/`ctx.check_cancel` após pedido de cancelamento. |
| `JobTimeout` | classe | `()` | Levantada dentro de um job em thread que passou do limite de tempo. |
| `JobContext` | classe | `(db_path, job_id, params, workdir, attempt, deadline=None)` | O que um job em execução vê: id, parâmetros, diretório de trabalho, progresso e log. |
| `JobQueue` | classe | `(db_path, *, app='', max_concurrent=2, executor='thread', workdir=None, poll_interval=0.2…)` | Fila persistente em SQLite com registro de funções, `submit`, `wait`, cancelamento, novas tentativas e limite de concorrência entre workers. |

#### `pinneapple_core.service.metering`

Medição de uso por app, principal e unidade, com limites gratuitos por mês (UTC) vindos do código e do ambiente; só mede, não cobra. [sem teste direto]

<small>`pinneapple_core/service/metering.py` · 110 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `UsageLimitExceeded` | classe | `(unit, used, amount, limit, period)` | Erro de uso acima do limite gratuito do período para uma unidade. |
| `UsageMeter` | classe | `(db_path, app, limits=None)` | Registra e verifica uso por principal e unidade (`consume`, `usage`, `totals`). |
| `current_period` | função | `(ts=None) -> str` | Período corrente em UTC no formato ano-mês. |
| `limits_from_env` | função | `(prefix, defaults, environ=None) -> dict[str, float \| None]` | Limites gratuitos por unidade, com padrões sobrescritos por `<PREFIX>_FREE_<UNIT>`. |

#### `pinneapple_core.service.uploads`

Armazena uploads com limite de tamanho, sufixos permitidos, nomes seguros e SHA-256 em `<root>/<id>/`. [sem teste direto]

<small>`pinneapple_core/service/uploads.py` · 106 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `UploadTooLarge` | classe | `()` | Erro de upload acima do tamanho máximo. |
| `UploadRejected` | classe | `()` | Erro de upload recusado (por exemplo sufixo não permitido). |
| `UploadStore` | classe | `(root, max_bytes=50 * 2 ** 20, allowed_suffixes=None)` | Guarda uploads em disco com metadados, e permite ler, listar, apagar e podar por idade. |
| `safe_name` | função | `(name) -> str` | Reduz um nome de arquivo ao nome base com caracteres seguros, nunca vazio nem oculto. |

#### `pinneapple_core.transforms`

Transformações invertíveis e componíveis (`>>`) sobre tabelas, `PhysicsDataset` e, para as de escala, `PhysicalProblem`, registrando proveniência a cada `apply`; as não-escalares levantam `NotImplementedError` em nível de problema.

<small>`pinneapple_core/transforms.py` · 636 linhas · testes: 13 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NotInvertible` | classe | `()` | Erro de transformação (ou passo de composição) sem inversa. |
| `Transform` | classe | `()` | Base das transformações: `forward`/`inverse` em tabelas e `params()` para proveniência. |
| `Compose` | classe | `(steps) -> None` | Sequência de transformações; `inverse` roda na ordem reversa. |
| `Scale` | classe | `(scale, shift=None) -> None` | Escala afim de colunas nomeadas; num `PhysicalProblem` reescala coordenadas, domínio e parâmetros. |
| `Nondimensionalize` | classe | `(problem=None, *, length=None, time=None, field=None, scale=None, shift=None) -> None` | Leva um problema a variáveis adimensionais e permite trazer predições de volta às unidades originais. |
| `Coordinate` | classe | `(kind, inputs, outputs, center=None) -> None` | Mudança de coordenadas em colunas: polar, cilíndrica, esférica ou log. |
| `Symmetry` | classe | `(kind, coords, vectors=(), *, axis=None, angle=0.0, center=None) -> None` | Operação de simetria (reflexão, rotação, com campos vetoriais) para aumento de dados ou teste de equivariância. |
| `Periodic` | classe | `(coord, period, harmonics=1, lo=0.0) -> None` | Troca uma coordenada periódica por `cos`/`sin` da fase, tornando a rede periódica por construção. |
| `FourierFeatures` | classe | `(coords, n_features=16, scale=1.0, seed=0, prefix='ff') -> None` | Features de Fourier aleatórias de algumas colunas. |
| `list_pde_rules` | função | `() -> List[str]` | Tipos de PDE cujos parâmetros `Scale`/`Nondimensionalize` sabem reescalar. |

#### `pinneapple_core.workflows`

Orquestra ferramentas existentes (solvers, Docker, Slurm) como DAG de passos; reexporta `Workflow`, `Step`, `WorkflowRunner`, executores, adaptadores, templates e `dataset_from_run`.

<small>`pinneapple_core/workflows/__init__.py` · 79 linhas · testes: 64</small>

#### `pinneapple_core.workflows.adapters`

Adaptadores por `kind` de passo (`command`, `python`, `openfoam`, `calculix`): preparam o diretório, rodam comandos pelo executor e devolvem saídas em JSON.

<small>`pinneapple_core/workflows/adapters.py` · 285 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `StepFailed` | classe | `()` | Erro de passo que rodou e falhou (saída não zero, erro do solver, saída ausente). |
| `StepContext` | classe | `(run_id, step, workdir, executor, upstream, log, should_stop, attempt)` | Contexto de um passo: id da execução, passo, diretório, executor, saídas a montante, log e tentativa. |
| `StepCancelled` | classe | `()` | Erro de passo cancelado. |
| `Adapter` | classe | `(kind)` | Base abstrata de um tipo de passo (`kind`) com verificação de disponibilidade e `run`. |
| `CommandAdapter` | classe | `()` | Passo `command`: roda `argv` com `{param}` preenchidos e lê `outputs.json` se existir. |
| `PythonAdapter` | classe | `()` | Passo `python`: chama `module:fn` no processo (só executor local) ou roda um script com o executor. |
| `OpenFOAMAdapter` | classe | `()` | Passo `openfoam`: copia um caso, aplica edições, roda blockMesh e o solver e lê a saúde do log. |
| `CalculiXAdapter` | classe | `()` | Passo `calculix`: copia um `.inp` e roda `ccx`. |
| `resolve_function` | função | `(ref) -> Callable` | Importa e devolve a função de uma referência `module:function`. |
| `register_adapter` | função | `(adapter) -> None` | Registra um adaptador para tornar executáveis passos do seu `kind`. |
| `get_adapter` | função | `(kind) -> Adapter` | Devolve o adaptador registrado para um `kind`. |

#### `pinneapple_core.workflows.dag`

Definição do workflow como DAG (`Step`, `Workflow`): validação de dependências e ciclos, ordem, JSON e expansão por tabela de casos com preenchimento de `{param}`. [sem teste direto]

<small>`pinneapple_core/workflows/dag.py` · 208 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): placeholder/stub citado no docstring</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `WorkflowError` | classe | `()` | Erro de workflow inválido (passo desconhecido, id duplicado, ciclo). |
| `Step` | classe | `(id, kind, params, needs, executor, executor_options, retries, timeout, inputs, group)` | Um passo do DAG: id, `kind`, parâmetros, dependências, executor, tentativas, timeout e arquivos de entrada. |
| `Workflow` | classe | `(name, steps, fail_fast, max_parallel, metadata)` | Conjunto de passos com validação, ordem topológica, `fail_fast`, paralelismo máximo e salvar/carregar JSON. |
| `fill` | função | `(obj, values) -> Any` | Preenche `{name}` em strings, recursivamente; uma string que é só um placeholder recebe o valor tal qual. |
| `workflow_from_cases` | função | `(name, cases, template, *, id_column=None, final=(), **kw) -> Workflow` | Copia os passos de um template uma vez por caso de uma tabela, com ids `<caso>.<passo>`. |

#### `pinneapple_core.workflows.datasets`

Leva uma tabela de casos a um workflow e as execuções terminadas a uma tabela de treino com `DatasetSpec` (marcado `partial` se faltam casos). [sem teste direto]

<small>`pinneapple_core/workflows/datasets.py` · 115 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `read_cases` | função | `(path)` | Lê uma tabela de casos (CSV, Parquet, HDF5, JSON) e o contrato anexado, se houver. |
| `id_column_of` | função | `(df, contract=None) -> str \| None` | Descobre a coluna de id dos casos pelo contrato ou por nomes comuns. |
| `dataset_from_run` | função | `(runner, run_id, cases, *, output_step, path, id_column=None, outputs=None, skeleton=None…)` | Grava a tabela de dataset de uma execução (parâmetros + saídas numéricas do passo) e devolve o `DatasetSpec`. |

#### `pinneapple_core.workflows.engine`

`WorkflowRunner`: executa workflows com estado e log por passo em SQLite, novas tentativas, retomada e pulo de passos já feitos por hash de conteúdo. [sem teste direto]

<small>`pinneapple_core/workflows/engine.py` · 367 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `WorkflowRunner` | classe | `(root, *, cache=True, input_root=None)` | Executa e retoma workflows, guarda estado, logs e cache por hash em SQLite sob uma raiz. |
| `sha256_file` | função | `(path) -> str` | SHA-256 do conteúdo de um arquivo. |
| `step_hash` | função | `(step, upstream_hashes, root=None) -> str` | Hash de conteúdo de um passo (tipo, parâmetros, opções, arquivos de entrada, hashes a montante). |

#### `pinneapple_core.workflows.executors`

Executores plugáveis que rodam o comando de um passo: `local` (subprocesso), `docker` (sem rede por padrão) e `slurm` (script sbatch, só escrito por padrão com `dry_run=True`). [sem teste direto]

<small>`pinneapple_core/workflows/executors.py` · 236 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ExecResult` | classe | `(returncode, seconds, command, dry_run, timed_out, cancelled, details)` | Resultado da execução de um comando: código de saída, tempo, comando, ensaio, timeout, cancelamento. |
| `Executor` | classe | `(**options)` | Base de executores: rodam um comando de um passo; subclasses definem `name` e `run`. |
| `LocalExecutor` | classe | `()` | Executor por subprocesso local, com timeout e parada sob pedido. |
| `DockerExecutor` | classe | `()` | Executor `docker run --rm` com o diretório do passo em /work, sem rede por padrão e com limites. |
| `SlurmExecutor` | classe | `()` | Executor que escreve um script sbatch por passo (submete só fora do modo de ensaio). |
| `register_executor` | função | `(name, cls) -> None` | Registra uma classe de executor sob um nome usável em `executor=`. |
| `get_executor` | função | `(name, **options) -> Executor` | Instancia o executor registrado com as opções dadas. |

#### `pinneapple_core.workflows.templates`

Templates de workflow seguros para servidor público (`slab_heat`, `beam_calculix`, `cavity_openfoam`), que validam cada caso contra o espaço de parâmetros.

<small>`pinneapple_core/workflows/templates.py` · 213 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Template` | classe | `(id, title, description, space, steps, output_step, final, outputs, solver)` | Template de workflow: espaço de parâmetros, passos por caso, passo de saída e solver necessário. |
| `slab_heat_solve` | função | `(kw, ctx) -> dict` | Perfil de temperatura da placa em `t_end` pela série exata; escreve `profile.csv`. |
| `collect_outputs` | função | `(kw, ctx) -> dict` | Resumo sobre todos os passos a montante (contagem, mínimo e máximo de cada saída numérica); escreve `summary.csv`. |
| `cantilever_deck` | função | `(kw, ctx) -> dict` | Gera o `job.inp` de uma viga em balanço (hexaedros C3D8I) e a deflexão de ponta de Euler-Bernoulli. |
| `cantilever_compare` | função | `(kw, ctx) -> dict` | Compara a deflexão máxima do CalculiX com a de Euler-Bernoulli e devolve a razão. |

### Exemplos de uso reais do repositório

**PINN de Poisson sobre as primitivas** — `examples/core_primitives/01_poisson_pinn.py`

```python
from pinneapple_core import Field, FunctionField, Geometry

torch.manual_seed(0)
geom = Geometry.box([0, 0], [1, 1])
net = torch.nn.Sequential(
    torch.nn.Linear(2, 48), torch.nn.Tanh(),
    torch.nn.Linear(48, 48), torch.nn.Tanh(),
    torch.nn.Linear(48, 1),
)
u = FunctionField(net, geom.domain)

source = lambda x: 2 * np.pi**2 * torch.sin(np.pi * x[:, 0]) * torch.sin(np.pi * x[:, 1])
x_int = torch.tensor(geom.sample_interior(2000, seed=0), dtype=torch.float32)
x_bnd = torch.tensor(geom.sample_boundary(400, seed=1), dtype=torch.float32)
```

**DataLoader com amostradores e pp.loss** — `examples/data_api/01_pinn_operator_inverse.py`

```python
geom = Geometry.box([0.0], [1.0])
net = mlp(1, 1)
loader = DataLoader({"col": CollocationSampler(geom, "lhs"), "bc": BoundarySampler(geom)}, batch_size={"col": 256, "bc": 2}, steps=1500)
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
for batch in loader:
    x = batch["col"]["x"]                                              # already requires_grad
    r = -second(net(x), x) - PI**2 * torch.sin(PI * x)
    loss = L.combine({"pde": L.pde(r), "bc": L.boundary(net(batch["bc"]["x"]))}, {"bc": 10.0})
    opt.zero_grad(); loss.backward(); opt.step()
```

**Workflow com template e dataset** — `docs/guides/workflow_runner.md`

```python
from pinneapple_core.workflows import Step, Workflow, WorkflowRunner, TEMPLATES, dataset_from_run

runner = WorkflowRunner("work/runner")
t = TEMPLATES["slab_heat"]
cases = t.example_cases(20)                       # or read_cases("campaign.parquet")
run = runner.run(t.workflow(cases, max_parallel=4))
runner.run(run_id=run["id"])                      # resume (no-op when everything finished)
spec, path = dataset_from_run(runner, run["id"], cases, output_step="solve", path="slab.parquet")
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 19 de 34.
- Funções de teste em arquivos que importam algum módulo do pacote: 233.

### Limitações conhecidas

- `fem` usa matrizes densas: o docstring diz que serve para malhas de até alguns milhares de nós (casos de referência, protótipos de otimização de forma, testes).
- Domínios e geometrias não-caixa só são suportados em 2D (formas CSG de `pinneapple_design`); medir ou malhar a fronteira em outras dimensões levanta `NotImplementedError`.
- `PhysicsBackend` base e `Sampler` base levantam `NotImplementedError`; o backend jax exige jax instalado e liga float64 globalmente.
- Em `transforms`, só `Scale` e `Nondimensionalize` têm forma em nível de problema; as demais levantam `NotImplementedError` ao serem aplicadas a um `PhysicalProblem`.
- `recommend` e `search` do onboarding são pontuação por palavras-chave sobre um catálogo curado, não um modelo; podem errar e dizem quais termos casaram.
- Os adaptadores `openfoam` e `calculix` e os templates `beam_calculix`/`cavity_openfoam` precisam de OpenFOAM ou `ccx` instalados; o executor `slurm` só escreve o script por padrão (`dry_run=True`).
- `service` mede uso mas não cobra; o cancelamento de jobs em modo thread é cooperativo (só vale quando o job chama `ctx.progress`/`ctx.check_cancel`).
- Vários módulos não têm teste que os importe diretamente (`domain`, `field`, `mesh`, `geometry`, `operators`, `onboarding.recommend`, os stores de `service` e partes de `workflows`); são exercitados pelos testes que importam o pacote ou o subpacote.

## 3. `pinneapple_catalog`

### Objetivo

`pinneapple_catalog` é o catálogo do que o PINNeAPPle conhece, sempre com a fonte. Ele tem duas partes. `resources` lista recursos públicos de Physics AI (datasets, conjuntos de CAD/geometria, modelos pré-treinados e benchmarks hospedados), cada um com licença conferida, classificação de uso e, quando houver, o repositório no Hugging Face. `methods` lista cada solver, método de treino, equação física e problema (preset) da biblioteca, com o caminho do código, as referências bibliográficas e o status de validação.

O problema que o pacote resolve é de rastreabilidade e de licença. Um componente que quer um recurso público pergunta primeiro ao catálogo: `require_allowed` aplica a mesma trava de licença de `pinneapple_neural._licencas` e recusa recursos sem uso comercial (ou com licença não verificada) quando `PINNEAPPLE_COMMERCIAL_MODE=1`. `fetch` baixa o recurso do Hugging Face Hub só depois dessa trava.

No lado dos métodos, o status de validação não é escrito à mão. O script `scripts/build_method_status.py` varre `tests/` e grava `pinneapple_catalog/method_status.json`. Um item é `validated` quando algum teste o compara com uma referência independente (forma fechada, valor publicado ou solução manufaturada), `tested` quando há testes sem essa comparação, e `untested` caso contrário. `validated_methods()` devolve só os validados; a página pública `docs/validation_status.md` é conferida contra essa varredura por teste. Os caminhos de código apontam para os outros pacotes (`pinneapple_simulation`, `pinneapple_neural`, `pinneapple_physics`, `pinneapple_systems`) e os problemas apontam para os presets de `pinneapple_physics/pde_environment/presets/`.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 3 | 1.024 | 2 | 8 | 16 (3 arquivos) | 1 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `methods` | Inventário de solvers, métodos de treino, equações e problemas, com código, referências e status de validação lido de `method_status.json`. | 1 | 5 | 8 |
| `resources` | Catálogo de recursos públicos com licença conferida, filtros, trava de licença e download pelo Hugging Face Hub. | 1 | 5 | 0 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_catalog`

Raiz do pacote: reexporta a API de `resources` (`Resource`, `list_resources`, `get_resource`, `require_allowed`, `fetch`); `methods` é importado à parte.

<small>`pinneapple_catalog/__init__.py` · 11 linhas · testes: 8</small>

#### `pinneapple_catalog.methods`

Lista fixa de `Method` (IDs S = solver, T = treino, E = equação, P = problema) com caminho do código, referências citadas no código e referências acrescentadas, e funções para consultar o status de validação.

<small>`pinneapple_catalog/methods.py` · 569 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Method` | classe | `(id, name, code, probes, refs_in_code, refs_added, equations)` | Registro imutável de um item do catálogo: id, nome, caminhos de código, termos de busca nos testes (`probes`), referências e, para problemas, as equações usadas; expõe `category` e `references`. |
| `get_method` | função | `(method_id) -> Method` | Devolve o `Method` de um id ou levanta `KeyError` se o id não existe. |
| `list_methods` | função | `(category=None) -> List[Method]` | Lista todos os métodos, ou só os de uma categoria (`solver`, `training`, `equation`, `problem`). |
| `method_status` | função | `() -> Dict[str, Dict]` | Lê `method_status.json` e devolve, por id, o status e os testes de referência e outros testes; devolve dict vazio se o arquivo não existe. |
| `validated_methods` | função | `(category=None) -> List[Method]` | Lista oficial: só os métodos cujo status na última varredura é `validated`. |

#### `pinneapple_catalog.resources`

Lista fixa de `Resource` com licença, uso comercial (`yes`/`no`/`unverified`), classificação (`use_now`, `product`, `reference`, `skip`), filtros, trava de licença e download. [sem teste direto pelo inventário; é exercitado via `tests/test_catalog_resources.py`, que importa pela raiz]

<small>`pinneapple_catalog/resources.py` · 444 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Resource` | classe | `(id, name, kind, domain, summary, license, commercial, url, rating, paper, size, hf_repo,…)` | Registro imutável de um recurso público (tipo, domínio, licença, uso comercial, URL, classificação, artigo, tamanho, repositório HF, notas, tags); `research_only` é verdadeiro sempre que o uso comercial não é `yes`. |
| `list_resources` | função | `(kind=None, domain=None, rating=None, commercial_only=False) -> List[Resource]` | Filtra os recursos por tipo, domínio e classificação; `commercial_only` mantém só os de uso comercial `yes`. |
| `get_resource` | função | `(resource_id) -> Resource` | Devolve o `Resource` de um id ou levanta `KeyError` listando os ids conhecidos. |
| `require_allowed` | função | `(resource_id) -> Resource` | Trava de licença: avisa para recursos só de pesquisa e os recusa em modo comercial, via `require_research_use` de `pinneapple_neural._licencas`. |
| `fetch` | função | `(resource_id, *, allow_patterns=None, local_dir=None, token=None, revision=None) -> str` | Passa pela trava de licença e baixa o recurso com `huggingface_hub.snapshot_download`, devolvendo o caminho local; aceita `allow_patterns` para baixar só parte. |

### Exemplos de uso reais do repositório

**Recursos sem licença comercial contam como só pesquisa** — `tests/test_catalog_resources.py`

```python
from pinneapple_catalog import fetch, get_resource, list_resources, require_allowed
from pinneapple_neural._licencas import ResearchOnlyError, ResearchOnlyWarning


def test_unknown_or_missing_license_counts_as_research_only():
    assert get_resource("pressnet").research_only  # no LICENSE file
    assert get_resource("drivaernet").research_only  # CC BY-NC
    assert not get_resource("airfrans").research_only  # ODbL allows commercial use


def test_commercial_only_filter_excludes_research_only():
    assert all(not r.research_only for r in list_resources(commercial_only=True))
```

**Status de validação cobre todo o catálogo de métodos** — `tests/test_catalog_methods.py`

```python
def test_status_covers_every_item_and_validated_is_a_subset():
    status = method_status()
    assert {m.id for m in list_methods()} <= set(status)
    for m in validated_methods():
        assert status[m.id]["status"] == "validated"
        assert status[m.id]["reference_tests"], m.id
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 2 de 3.
- Funções de teste em arquivos que importam algum módulo do pacote: 16.

### Limitações conhecidas

- As listas de recursos e de métodos são fixas no código; incluir um item novo exige editar `resources.py` ou `methods.py`.
- O status de validação depende de rodar `scripts/build_method_status.py`; se `method_status.json` estiver desatualizado, `validated_methods()` também estará. Sem o arquivo, `method_status()` devolve vazio.
- A detecção de testes é por presença de strings (`probes`) nos arquivos de teste, não por execução.
- `fetch` exige o extra `hub` (`huggingface_hub`) e só funciona para recursos com `hf_repo`; os demais levantam `ValueError` apontando a URL de origem.
- Licença não verificada (`unverified`) é tratada como não comercial; em `PINNEAPPLE_COMMERCIAL_MODE=1` esses recursos são recusados.
- Alguns caminhos de código e docstrings citam o repositório irmão `PINNeAPPle-CFD`, que pode não estar presente; o teste pula esses caminhos nesse caso.
- `pinneapple_catalog.methods` não é reexportado pela raiz do pacote; é preciso importar `pinneapple_catalog.methods` diretamente.

## 4. `pinneapple_registry`

### Objetivo

_Sem texto de objetivo._

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 10 | 1.699 | 10 | 3 | 92 (10 arquivos) | 3 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `dataset_store` |  | 1 | 1 | 0 |
| `experiment_memory` |  | 1 | 2 | 19 |
| `experiment_store` |  | 1 | 1 | 10 |
| `gateway` |  | 1 | 2 | 10 |
| `gateway_api` |  | 1 | 2 | 10 |
| `gateway_client` |  | 1 | 2 | 10 |
| `model_store` |  | 1 | 1 | 13 |
| `problem_store` |  | 1 | 1 | 0 |
| `registry` |  | 1 | 1 | 0 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_registry`

pinneapple_registry — a local, self-hosted artifact registry: versioned model storage, dataset storage, experiment tracking, and problem-spec history, plus Triton Inference Server export.

<small>`pinneapple_registry/__init__.py` · 45 linhas · testes: 50</small>

#### `pinneapple_registry.dataset_store`

pinneapple_registry.dataset_store — ``DatasetStore``: local, per- problem/scenario storage for simulation datasets (coords + field arrays), with auto-computed summary statistics.

<small>`pinneapple_registry/dataset_store.py` · 99 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DatasetStore` | classe | `(root)` |  |

#### `pinneapple_registry.experiment_memory`

pinneapple_registry.experiment_memory -- ``ExperimentMemory``: persistent cross-run memory for an (separately built, not part of this repo) autonomous research loop.

<small>`pinneapple_registry/experiment_memory.py` · 252 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ExperimentRecord` | classe | `(problem_description, physics_domain, approach_summary, outcome, confidence_score, metric…)` | One logged attempt at solving a physics problem. |
| `ExperimentMemory` | classe | `(store=None, *, db_path=None)` | Persistent memory of past experiment attempts, backed by the real ``ExperimentStore``. |

#### `pinneapple_registry.experiment_store`

pinneapple_registry.experiment_store — ``ExperimentStore``: a SQLite- backed run tracker. Closes the one gap ``pinneapple_hub`` genuinely has no answer for: it publishes finished models to the Hugging Face Hub, but nothing in the repo durably records *runs* (start/metrics-per-epoch/finish) …

<small>`pinneapple_registry/experiment_store.py` · 144 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ExperimentStore` | classe | `(db_path)` |  |

#### `pinneapple_registry.gateway`

``ModelGateway``: publish physical models as versioned endpoints, run them, and record every run.

<small>`pinneapple_registry/gateway.py` · 457 linhas · testes: 10 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ModelGateway` | classe | `(root, *, cache_size=8)` |  |
| `ood_verdict` | função | `(flag, policy='warn') -> str` | What ``predict`` does with one assessed row under ``policy``: ``"in range"`` (predicted), ``"warn"`` (predicted and flagged as an extrapolation) or ``"refuse"`` (no prediction). |

#### `pinneapple_registry.gateway_api`

The gateway's REST shape (FastAPI), shared by every app that serves models: ``/v1/models/...``.

<small>`pinneapple_registry/gateway_api.py` · 144 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `count_rows` | função | `(inputs) -> int` | Rows a request asks for, before the model is loaded. |
| `gateway_router` | função | `(gateway, *, prefix='/v1', on_rows=None, caller=None, max_rows=10000, tags=('models',))` |  |

#### `pinneapple_registry.gateway_client`

``GatewayClient``: call any server that exposes the gateway API (``/v1/models/...``) from Python.

<small>`pinneapple_registry/gateway_client.py` · 83 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GatewayError` | classe | `(status, message)` |  |
| `GatewayClient` | classe | `(base_url, *, api_key=None, timeout=30.0, headers=None)` |  |

#### `pinneapple_registry.model_store`

pinneapple_registry.model_store — ``ModelStore``: a local, filesystem- backed, versioned store for trained models, paralleling (not replacing) ``pinneapple_hub`` — see the package docstring in ``pinneapple_registry/__init__.py`` for why both exist.

<small>`pinneapple_registry/model_store.py` · 246 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ModelStore` | classe | `(root)` |  |

#### `pinneapple_registry.problem_store`

pinneapple_registry.problem_store — ``ProblemStore``: versioned problem definitions with history.

<small>`pinneapple_registry/problem_store.py` · 98 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ProblemStore` | classe | `(root)` |  |

#### `pinneapple_registry.registry`

pinneapple_registry.registry — ``ArtifactRegistry``: unifies ``ModelStore``/``DatasetStore``/``ExperimentStore``/``ProblemStore`` under one root, plus a Triton Inference Server model-repository generator.

<small>`pinneapple_registry/registry.py` · 131 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ArtifactRegistry` | classe | `(root)` | One root directory, four sub-stores, one experiment-tracking DB:: |

### Exemplos de uso reais do repositório

_Nenhum exemplo encontrado no repositório para este pacote._

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 7 de 10.
- Funções de teste em arquivos que importam algum módulo do pacote: 92.

### Limitações conhecidas

- _Nenhuma registrada._

## 5. `pinneapple_orchestration`

### Objetivo

`pinneapple_orchestration` é uma camada opcional de pipelines no estilo Prefect. Ela não traz lógica nova: só encadeia o que já existe em outros pacotes. Os dados vão para o `DatasetStore` e os modelos para o registro de artefatos de `pinneapple_registry`. Os componentes treináveis vêm de `pinneapple_systems.component_library` e o gêmeo digital de `pinneapple_systems.digital_twin`.

O pacote é pequeno de propósito: são cinco flows. `flow_data_generation` roda N vezes uma função de amostragem dada pelo usuário e grava cada cenário. `flow_training` monta um componente registrado, treina e registra a execução num `ExperimentStore`. `flow_benchmark` ranqueia todas as arquiteturas de um `component_type`. `flow_registry_publish` promove um modelo de estágio (`development` para `staging`/`production`), sempre com confirmação explícita. `flow_digital_twin_prep` monta um `DigitalTwin` a partir de um modelo do registro, só se ele estiver no estágio exigido.

Cada flow chama uma task fina em `tasks/`, e cada task delega ao pacote dono da lógica (por exemplo `ComponentModel.fit`, que usa o `Trainer` comum). Todo flow roda dentro de `flow_run_manifest`, que grava um JSON por execução com parâmetros, SHA do git, log das tasks e artefatos.

O Prefect não é obrigatório. Com o extra `pinneapple[orchestration]` instalado, os decoradores `flow`/`task` são os do Prefect (retries, agendamento, UI). Sem ele, `_prefect_compat` troca os decoradores por funções que não fazem nada, e cada flow vira uma função Python comum, chamável e testável.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 16 | 591 | 1 | 11 | 6 (1 arquivos) | 14 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `_prefect_compat` | Shim que usa os decoradores `flow`/`task` do Prefect quando instalado e, senão, decoradores que devolvem a função sem mudança. | 1 | 0 | 0 |
| `common` | Utilitários comuns aos flows; hoje só o manifesto de proveniência de cada execução. | 2 | 2 | 0 |
| `flows` | Os cinco flows públicos (geração de dados, treino, benchmark, publicação no registro, preparo de gêmeo digital). | 6 | 5 | 6 |
| `tasks` | As tasks finas por trás de cada flow, que delegam a `pinneapple_registry` e `pinneapple_systems`. | 6 | 5 | 0 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_orchestration`

Ponto de entrada do pacote: reexporta `PREFECT_AVAILABLE`, os subpacotes `tasks` e `flows`, `flow_run_manifest` e `FlowRunManifest`.

<small>`pinneapple_orchestration/__init__.py` · 44 linhas · testes: 6</small>

#### `pinneapple_orchestration._prefect_compat`

[shim] Importa `flow` e `task` do Prefect se existir; senão define decoradores de passagem que aceitam `@flow` e `@flow(...)` e ignoram os kwargs. Define `PREFECT_AVAILABLE`. [sem teste direto]

<small>`pinneapple_orchestration/_prefect_compat.py` · 36 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): shim ou obsoleto</small>

#### `pinneapple_orchestration.common`

Subpacote que reexporta `flow_run_manifest` e `FlowRunManifest`. [sem teste direto]

<small>`pinneapple_orchestration/common/__init__.py` · 6 linhas · **sem teste direto**</small>

#### `pinneapple_orchestration.common.provenance`

Gerenciador de contexto que grava `<manifest_dir>/<flow_name>/<run_id>.json` ao fim de cada flow (sucesso ou falha), com parâmetros, SHA do git, log de tasks, artefatos e duração. [sem teste direto]

<small>`pinneapple_orchestration/common/provenance.py` · 95 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FlowRunManifest` | classe | `(flow_name, run_id, parameters, git_sha, started_at, finished_at, status, task_log, artif…)` | Registro de uma execução de flow: nome, id, parâmetros, SHA do git, horários, status, log de tasks, artefatos e erro. |
| `flow_run_manifest` | função | `(flow_name, parameters=None, *, manifest_dir='data/manifests', repo_dir=None) -> Iterator…)` | Gerenciador de contexto que cria um `FlowRunManifest`, marca sucesso ou falha e grava o JSON no fim. |

#### `pinneapple_orchestration.flows`

Subpacote que reexporta os cinco flows.

<small>`pinneapple_orchestration/flows/__init__.py` · 16 linhas · testes: 6</small>

#### `pinneapple_orchestration.flows.flow_01_data_generation`

Flow que roda `sample_fn` N vezes e grava cada resultado como cenário num `DatasetStore`. [sem teste direto; coberto por `tests/test_orchestration.py` via `flows`]

<small>`pinneapple_orchestration/flows/flow_01_data_generation.py` · 31 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `flow_data_generation` | função | `(dataset_store, problem_id, sample_fn, n_scenarios, *, manifest_dir='data/manifests') -> …)` | Gera `n_scenarios` cenários com `sample_fn`, grava no `DatasetStore` e devolve os ids e o `run_id`. |

#### `pinneapple_orchestration.flows.flow_02_training`

Flow que monta e treina um componente registrado e registra a execução num `ExperimentStore` opcional.

<small>`pinneapple_orchestration/flows/flow_02_training.py` · 36 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `flow_training` | função | `(component_name, train_loader, val_loader=None, *, experiment_store=None, epochs=100, lr=…)` | Treina o componente `component_name` com os loaders dados e devolve o modelo, o resultado do `fit` e o `run_id`. |

#### `pinneapple_orchestration.flows.flow_03_benchmark`

Flow que ranqueia todas as arquiteturas registradas para um `component_type`.

<small>`pinneapple_orchestration/flows/flow_03_benchmark.py` · 29 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `flow_benchmark` | função | `(component_type, train_loader, val_loader=None, *, epochs=50, lr=0.001, manifest_dir='dat…)` | Ranqueia as arquiteturas de um `component_type` e guarda o ranking no manifesto. |

#### `pinneapple_orchestration.flows.flow_04_registry_publish`

Flow terminal que salva um modelo no registro e o promove de estágio só com `confirm=True`.

<small>`pinneapple_orchestration/flows/flow_04_registry_publish.py` · 32 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `flow_registry_publish` | função | `(registry, problem_id, model, *, metadata=None, stage='staging', confirm=False, manifest_…)` | Publica um modelo no registro e devolve a versão; levanta `PermissionError` se o estágio não é `development` e `confirm` não é `True`. |

#### `pinneapple_orchestration.flows.flow_05_digital_twin_prep`

Flow que monta um `DigitalTwin` a partir de um modelo do registro, exigindo por padrão o estágio `production`.

<small>`pinneapple_orchestration/flows/flow_05_digital_twin_prep.py` · 36 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `flow_digital_twin_prep` | função | `(registry, problem_id, field_names, coord_names=None, *, version=None, require_stage='pro…)` | Devolve um `DigitalTwin` montado com o modelo do registro, checando o estágio exigido. |

#### `pinneapple_orchestration.tasks`

Subpacote que reexporta as cinco tasks. [sem teste direto]

<small>`pinneapple_orchestration/tasks/__init__.py` · 16 linhas · **sem teste direto**</small>

#### `pinneapple_orchestration.tasks.benchmark_tasks`

Task que só chama `pinneapple_systems.component_library.benchmark_component_type`; não há lógica de benchmark aqui. [sem teste direto]

<small>`pinneapple_orchestration/tasks/benchmark_tasks.py` · 25 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `benchmark_component_type_task` | função | `(component_type, train_loader, val_loader=None, *, epochs=50, lr=0.001) -> ComponentBench…)` | Repassa a chamada para `benchmark_component_type` e devolve um `ComponentBenchmarkResult`. |

#### `pinneapple_orchestration.tasks.data_generation_tasks`

Task genérica de geração de dados: roda um thunk `sample_fn() -> (coords, fields)` dado pelo usuário e salva cada execução no `DatasetStore`. [sem teste direto]

<small>`pinneapple_orchestration/tasks/data_generation_tasks.py` · 43 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `generate_scenarios_task` | função | `(dataset_store, problem_id, sample_fn, n_scenarios, *, scenario_prefix='scenario') -> Lis…)` | Roda `sample_fn()` `n_scenarios` vezes, salva cada resultado como cenário `<prefixo>_NNNN` e devolve os ids. |

#### `pinneapple_orchestration.tasks.digital_twin_tasks`

Task que carrega um modelo do registro e monta um `DigitalTwin`, recusando com `PermissionError` modelos fora do estágio exigido. [sem teste direto]

<small>`pinneapple_orchestration/tasks/digital_twin_tasks.py` · 48 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `prepare_digital_twin_task` | função | `(registry, problem_id, field_names, coord_names=None, *, version=None, require_stage='pro…)` | Escolhe a versão (a mais recente por padrão), confere o estágio, carrega o modelo e devolve um `DigitalTwin`. |

#### `pinneapple_orchestration.tasks.registry_tasks`

Task que salva o modelo em `development` e só promove para outro estágio com `confirm=True`; um `trust_report` REJECT bloqueia a promoção salvo `override_reason`. [sem teste direto]

<small>`pinneapple_orchestration/tasks/registry_tasks.py` · 47 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `publish_model_task` | função | `(registry, problem_id, model, *, metadata=None, stage='staging', confirm=False, trust_rep…)` | Salva o modelo em `development` e promove para `stage` só com `confirm=True`; devolve a versão. |

#### `pinneapple_orchestration.tasks.training_tasks`

Task fina que monta um componente via `ComponentRegistry`, chama `ComponentModel.fit()` e registra métricas num `ExperimentStore`. [sem teste direto]

<small>`pinneapple_orchestration/tasks/training_tasks.py` · 51 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `train_component_task` | função | `(component_name, train_loader, val_loader=None, *, experiment_store=None, epochs=100, lr=…)` | Monta o componente, treina com `fit`, registra `best_val` no `ExperimentStore` (se dado) e devolve modelo, resultado e `run_id`. |

### Exemplos de uso reais do repositório

**Gerar cenários com uma função de amostragem** — `tests/test_orchestration.py`

```python
def test_flow_data_generation(tmp_path):
    reg = _registry()

    def sample_fn():
        return np.random.rand(20, 2).astype(np.float32), {"u": np.random.rand(20, 1).astype(np.float32)}

    result = flow_data_generation(reg.datasets, "pipe_flow", sample_fn, 3, manifest_dir=str(tmp_path / "manifests"))
    assert len(result["scenarios"]) == 3
    assert reg.datasets.list_scenarios("pipe_flow") == sorted(result["scenarios"])
```

**Promoção de estágio exige confirm=True** — `tests/test_orchestration.py`

```python
    try:
        flow_registry_publish(reg, "valve_demo", model, stage="production", confirm=False,
                               manifest_dir=str(tmp_path / "manifests"))
        assert False, "expected PermissionError"
    except PermissionError:
        pass

    result = flow_registry_publish(reg, "valve_demo", model, stage="production", confirm=True,
                                    manifest_dir=str(tmp_path / "manifests"))
    assert reg.models.metadata("valve_demo", result["version"])["stage"] == "production"
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 2 de 16.
- Funções de teste em arquivos que importam algum módulo do pacote: 6.

### Limitações conhecidas

- Prefect é opcional (extra `pinneapple[orchestration]`); sem ele os flows rodam como funções comuns, sem retries, agendamento nem UI.
- Sem Prefect, os kwargs dos decoradores (por exemplo `retries`) são ignorados em silêncio.
- `flow_run_manifest` grava por padrão em `data/manifests` relativo ao diretório atual e chama `git rev-parse HEAD`; fora de um repositório git o SHA fica `None`.
- A geração de dados não configura solvers sozinha: o usuário precisa passar um thunk `sample_fn` sem argumentos.
- `flow_registry_publish` não repassa `trust_report` nem `override_reason` para `publish_model_task`; para usar esse portão é preciso chamar a task direto.
- Os testes (`tests/test_orchestration.py`) importam só `flows`; tasks, `common` e `_prefect_compat` não têm teste direto.
- Depende de `pinneapple_registry` e `pinneapple_systems` (importados no topo dos módulos de flows e tasks).

## 6. `pinneapple_models`

### Objetivo

`pinneapple_models` é um shim de compatibilidade. Ele não implementa nenhum modelo: só reexporta, de `pinneapple_neural.architectures`, o registro (`ModelRegistry`), o catálogo (`ModelCatalog`), a classe base `BaseModel` e algumas arquiteturas (`VanillaPINN`, `SIREN`, `ModifiedMLP`, `FourierNeuralOperator`, `FNO2d`, `DeepONet`, `MeshGraphNet`, `GraphBatch`, `GraphOutput`).

O problema que resolve é manter código antigo funcionando. Exemplos e scripts escritos com `from pinneapple_models import ...` continuam rodando depois que as arquiteturas foram para `pinneapple_neural`. O próprio docstring diz que código novo deve importar direto de `pinneapple_neural.architectures`.

Além dos nomes planos, o pacote cria pseudo-namespaces antigos (`pinns`, `neural_operators`, `graphnn`), um alias `catalog = ModelCatalog`, uma função `registry()` e tenta importar `register_all` do catálogo de `pinneapple_neural` (se falhar, vira uma função vazia). O submódulo `pinneapple_models.registry` existe só para que o caminho `pinneapple_models.registry.ModelRegistry` funcione, como em `pinneapple_train` e `pinneapple_solvers`.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 2 | 74 | 0 | 1 | 1 (1 arquivos) | 1 |

Pasta de testes dedicada: `tests/pinneapple_models` (1 funções de teste).

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `registry` | Submódulo de compatibilidade que reexporta `ModelRegistry` de `pinneapple_neural.architectures.registry`. | 1 | 0 | 1 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_models`

[shim] Reexporta registro, catálogo e arquiteturas de `pinneapple_neural.architectures` para código legado; marcado como obsoleto (use `pinneapple_neural.architectures`). [sem teste direto]

<small>`pinneapple_models/__init__.py` · 64 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): shim ou obsoleto</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `registry` | função | `()` | Função legada que devolve a classe `ModelRegistry` de `pinneapple_neural` (não cria registro novo). |

#### `pinneapple_models.registry`

[shim] Expõe `ModelRegistry` no caminho antigo `pinneapple_models.registry`, criado porque `tests/pinneapple_models/test_registry.py` não coletava sem ele.

<small>`pinneapple_models/registry.py` · 10 linhas · testes: 1</small>

### Exemplos de uso reais do repositório

**Teste de fumaça do registro pelo caminho antigo** — `tests/pinneapple_models/test_registry.py`

```python
def test_models_registry_smoke():
    from pinneapple_models.registry import ModelRegistry

    try:
        from pinneapple_models.register_all import register_all
        register_all()
    except Exception:
        pass

    names = ModelRegistry.list()
    assert isinstance(names, list)
```

**Tour do registro via imports legados** — `examples/architectures/00_registry_tour.py`

```python
from pinneapple_models import ModelCatalog
from pinneapple_models import register_all
from pinneapple_models import ModelRegistry


def _short(s: str, n: int = 90) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> None:
    register_all()

    print("\n=== Families ===")
    print(ModelRegistry.families())
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 1 de 2.
- Funções de teste em arquivos que importam algum módulo do pacote: 1.

### Limitações conhecidas

- É só um shim: todo o comportamento vem de `pinneapple_neural.architectures`; qualquer mudança lá afeta este pacote.
- Marcado como obsoleto no docstring; código novo deve importar de `pinneapple_neural.architectures`.
- Se `pinneapple_neural.architectures._catalog.register_all` não puder ser importado, `register_all` vira uma função que não faz nada, em silêncio.
- `XTFC` no namespace `pinns` fica `None` se o import de `pinneapple_neural.architectures.pinns.xtfc` falhar.
- O teste existente tenta importar `pinneapple_models.register_all` como módulo, que não existe; o erro é engolido por `try/except`.
- O `__init__` não tem teste direto; só o submódulo `registry` é coberto, por um teste de fumaça.

## 7. `pinneapple_train`

### Objetivo

`pinneapple_train` é um shim de compatibilidade. Ele não tem código próprio: só reexporta o treinador que hoje mora em `pinneapple_neural.trainer`. Existe para que código antigo que faz `from pinneapple_train import Trainer, TrainConfig` (ou importa `pinneapple_train.trainer`, `.losses`, `.metrics`) continue funcionando.

O `__init__` reexporta `Trainer`, `TrainConfig`, `EarlyStopping`, `ModelCheckpoint`, `Metrics`, `Metric`, `MetricBundle` e `default_metrics`. Nomes opcionais (`TimeMarchingTrainer`, `RunLogger`, `Checkpoint`, `save_checkpoint`, `SweepConfig`, `run_parallel_sweep`, `AdaptiveSweepConfig`, `run_adaptive_sweep`) vêm dentro de `try/except` e viram `None` se o import falhar.

Os submódulos `trainer.py`, `losses.py` e `metrics.py` foram criados porque exemplos (`examples/time_series/*`, `examples/trainer/01_audited_training.py`), ferramentas de benchmark e a base de conhecimento de `pinneapple_problemdesign` importavam esses caminhos, que antes não existiam. Código novo deve importar direto de `pinneapple_neural.trainer`.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 4 | 109 | 0 | 0 | 1 (1 arquivos) | 1 |

Pasta de testes dedicada: `tests/pinneapple_train` (1 funções de teste).

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `losses` | Reexporta `SupervisedLoss`, `PhysicsLossHook`, `CombinedLoss` e `build_loss` de `pinneapple_neural.trainer.losses`. | 1 | 0 | 1 |
| `metrics` | Reexporta as métricas (`MSE`, `MAE`, `RMSE`, `R2`, `RelL2`, `MaxError`, `default_metrics` etc.) de `pinneapple_neural.trainer.metrics`. | 1 | 0 | 1 |
| `trainer` | Reexporta `Trainer` e `TrainConfig` de `pinneapple_neural.trainer.trainer`. | 1 | 0 | 1 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_train`

[shim] Pacote de compatibilidade (marcado como obsoleto) que reexporta o treinador, callbacks e métricas de `pinneapple_neural.trainer`; nomes opcionais viram `None` se o import falhar. [sem teste direto do `__init__`]

<small>`pinneapple_train/__init__.py` · 65 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): shim ou obsoleto</small>

#### `pinneapple_train.losses`

[shim] Reexporta as perdas (`SupervisedLoss`, `PhysicsLossHook`, `CombinedLoss`, `build_loss`) de `pinneapple_neural.trainer.losses`.

<small>`pinneapple_train/losses.py` · 8 linhas · testes: 1</small>

#### `pinneapple_train.metrics`

[shim] Reexporta as métricas de regressão e `default_metrics` de `pinneapple_neural.trainer.metrics`.

<small>`pinneapple_train/metrics.py` · 14 linhas · testes: 1</small>

#### `pinneapple_train.trainer`

[shim] Reexporta `Trainer` e `TrainConfig` de `pinneapple_neural.trainer.trainer`, para o caminho `pinneapple_train.trainer` que exemplos e ferramentas já usavam.

<small>`pinneapple_train/trainer.py` · 22 linhas · testes: 1</small>

### Exemplos de uso reais do repositório

**Teste mínimo do treinador pelo caminho antigo** — `tests/pinneapple_train/test_trainer_minimal.py`

```python
from pinneapple_train.trainer import Trainer, TrainConfig
from pinneapple_train.losses import CombinedLoss, SupervisedLoss
from pinneapple_train.metrics import default_metrics

...
    combined = CombinedLoss(supervised=SupervisedLoss("mse"), physics=None)

    def loss_fn(model_, y_hat, batch):
        return combined(model_, y_hat, batch)

    tr = Trainer(model=model, loss_fn=loss_fn, metrics=default_metrics())
```

**Exemplo de séries temporais que usa o shim** — `examples/time_series/01_FNO.py`

```python
from pinneapple_train.trainer import Trainer, TrainConfig
from pinneapple_train.losses import CombinedLoss, SupervisedLoss
from pinneapple_train.metrics import default_metrics
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 3 de 4.
- Funções de teste em arquivos que importam algum módulo do pacote: 1.

### Limitações conhecidas

- É só um shim: todo o comportamento está em `pinneapple_neural.trainer`; o pacote é marcado como obsoleto.
- Vários nomes (`TimeMarchingTrainer`, `RunLogger`, `Checkpoint`, `SweepConfig`, `run_parallel_sweep`, `AdaptiveSweepConfig` etc.) são importados com `except Exception` e viram `None` em silêncio se o módulo de origem falhar.
- O `__init__` não tem teste direto; só os submódulos são exercitados por `tests/pinneapple_train/test_trainer_minimal.py`.
- Os submódulos reexportam só uma parte de `pinneapple_neural.trainer`; nomes fora das listas `__all__` não estão disponíveis por este caminho.

## 8. `pinneapple_solvers`

### Objetivo

`pinneapple_solvers` é um shim de compatibilidade. Não tem código próprio: reexporta os solvers numéricos que hoje vivem em `pinneapple_simulation.numerical_solvers`, para que código antigo com `from pinneapple_solvers import ...` continue funcionando.

O `__init__` traz `SolverBase`, `SolverOutput`, `SolverRegistry` e `FDMSolver` diretamente. `FEMSolver`, `FVMSolver`, `LBMSolver`, `SpectralSolver`, `SPHSolver` e `FFTSolver` vêm dentro de `try/except` e viram `None` se o import falhar. `registry` é apenas um apelido para a classe `SolverRegistry`.

O submódulo `fft.py` foi criado porque `tests/pinneapple_solvers/test_fft.py` importava `pinneapple_solvers.fft`, que não existia. O comentário no `__init__` registra outro bug corrigido: antes ele importava um nome inexistente (`FFTProcessor`), e o erro era engolido pelo `except`. Código novo deve importar de `pinneapple_simulation.numerical_solvers`.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 2 | 71 | 0 | 0 | 1 (1 arquivos) | 1 |

Pasta de testes dedicada: `tests/pinneapple_solvers` (4 funções de teste).

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `fft` | Reexporta `FFTSolver` de `pinneapple_simulation.numerical_solvers.fft`. | 1 | 0 | 1 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_solvers`

[shim] Pacote de compatibilidade (obsoleto) que reexporta os solvers de `pinneapple_simulation.numerical_solvers`; os solvers opcionais viram `None` se o import falhar. [sem teste direto do `__init__`]

<small>`pinneapple_solvers/__init__.py` · 61 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): shim ou obsoleto</small>

#### `pinneapple_solvers.fft`

[shim] Reexporta `FFTSolver` de `pinneapple_simulation.numerical_solvers.fft`, para o caminho usado em `tests/pinneapple_solvers/test_fft.py`.

<small>`pinneapple_solvers/fft.py` · 10 linhas · testes: 1</small>

### Exemplos de uso reais do repositório

**Teste de fumaça do FFTSolver pelo caminho antigo** — `tests/pinneapple_solvers/test_fft.py`

```python
import torch
from pinneapple_solvers.fft import FFTSolver

def test_fft_solver_smoke():
    solver = FFTSolver()
    x = torch.randn(4, 128)
    out = solver(x)
    assert hasattr(out, "result")
    assert out.result.shape[0] == 4
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 1 de 2.
- Funções de teste em arquivos que importam algum módulo do pacote: 1.

### Limitações conhecidas

- É só um shim obsoleto: todo o comportamento está em `pinneapple_simulation.numerical_solvers`.
- `FEMSolver`, `FVMSolver`, `LBMSolver`, `SpectralSolver`, `SPHSolver` e `FFTSolver` são importados com `except Exception` e viram `None` em silêncio se o módulo de origem falhar.
- READMEs de exemplos (`examples/use_cases/TEMPLATE/README.md`, `examples/use_cases/drill_pipe_surrogate/README.md`) ainda citam `FEnicsBridge`, `OpenFOAMBridge` e `pinneapple_solvers.openfoam_bridge`, que este shim não exporta.
- O `__init__` não tem teste direto; `tests/pinneapple_solvers/` testa `pinneapple_solvers.fft` e o `FDMSolver` pelo caminho novo.
