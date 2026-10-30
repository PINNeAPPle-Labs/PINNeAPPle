# Volume 4 — Aprendizado de máquina e dados

> Parte do [Mapa completo do PINNeAPPle](../mapa_completo.md). Gerado por `docs/pt/_mapa/gerar.py` a partir do inventário do commit `e53cdcfb04`.

Arquiteturas e treino de redes (PINN, operadores neurais, grafos), dados e amostragem, adaptação, modelos de mundo, percepção, computação quântica e a arena de benchmarks.

## Sumário

1. [`pinneapple_neural`](#1-pinneapple_neural)
2. [`pinneapple_data`](#2-pinneapple_data)
3. [`pinneapple_pdb`](#3-pinneapple_pdb)
4. [`pinneapple_adaptation`](#4-pinneapple_adaptation)
5. [`pinneapple_arena`](#5-pinneapple_arena)
6. [`pinneapple_worldmodel`](#6-pinneapple_worldmodel)
7. [`pinneapple_perception`](#7-pinneapple_perception)
8. [`pinneapple_quantum`](#8-pinneapple_quantum)

## 1. `pinneapple_neural`

### Objetivo

`pinneapple_neural` é o pacote de redes neurais do PINNeAPPle. Ele junta três antigos pacotes: `architectures` (antes `pinneapple_models`) com o zoológico de modelos, `trainer` (antes `pinneapple_train`) com o laço de treino e utilitários de HPC, e `predictor` (antes `pinneapple_inference`) com inferência e gráficos. Os nomes antigos `models`, `train` e `inference` continuam como apelidos.

O problema que resolve é dar uma porta única para construir, treinar e usar modelos de física com IA. As arquiteturas cobrem PINNs (vanilla, inverso, bayesiano, X-TFC, XPINN, difusão informada por física), operadores neurais (FNO 1D/2D/3D, DeepONet, U-NO, Transolver, PINO), redes em grafo (MeshGraphNet e dinâmica em malha), modelos de tempo contínuo, séries temporais, computação de reservatório, autoencoders e modelos de ordem reduzida (POD, DMD, OpInf, SINDy e o estudo `rom_study`). Cada família tem um catálogo próprio e se registra no `ModelRegistry` global quando `architectures` é importado; `ModelRegistry.build(nome, **kwargs)` normaliza nomes de argumentos entre famílias.

Os atalhos da raiz (`build_model`, `train_model`, `predict`, `train_and_predict`) encadeiam registro, `Trainer` e inferência em lote. Em volta disso há peças mais novas: `callbacks` leva os checks de `pp.validate` para dentro do treino em Lightning, PhysicsNeMo e Hugging Face; `physicsnemo_bridge` liga modelos treinados no PhysicsNeMo às ferramentas de UQ, gêmeo digital e checagem do PINNeAPPle; `lpm` traz blocos de um modelo para um espaço de projeto inteiro; `workflows` oferece fluxos prontos (FNO3d em volumes densos, versão paramétrica e surrogate híbrido com pós-modelo físico).

O pacote é usado pelo resto da biblioteca: a fachada `pp.*` e os solvers montam modelos pelo registro, `pinneapple_systems.component_library` e `pinneapple_orchestration` treinam componentes com o mesmo `Trainer`, e os resultados de ROM viram contratos de `pinneapple_data.contracts` para os apps. Integrações com licença restrita (Noether/Emmi AI) são só para pesquisa e bloqueadas em modo comercial por `_licencas`.

O submódulo `pinneapple_neural.trainer` reúne a maquinaria de treino do PINNeAPPle. O centro é o `Trainer` genérico (perdas, métricas, callbacks, checkpoint, auditoria, AMP e DDP), cercado por estratégias específicas de PINN: treino em duas fases, time-marching, treino causal, PINN inverso e gray-box, colocação adaptativa (RAR), balanceamento automático de pesos de perda (`loss_balancer`, `weight_scheduler`), otimizadores especiais (MultiAdam, quase-Newton auto-escalado, L-BFGS multi-rodada), reinícios múltiplos, incerteza heteroscedástica, restrições de monotonicidade e utilitários de paralelismo/HPC (DDP, FSDP, SLURM).

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 198 | 36.217 | 334 | 150 | 438 (49 arquivos) | 152 |

Pasta de testes dedicada: `tests/pinneapple_neural` (36 funções de teste).

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `_licencas` | Guardas de licença: avisa quando um componente só para pesquisa é usado e o bloqueia com `PINNEAPPLE_COMMERCIAL_MODE=1`. | 1 | 4 | 20 |
| `architectures` | Zoológico de arquiteturas por família, com catálogos, adaptadores de batch e o `ModelRegistry` global. | 147 | 270 | 64 |
| `callbacks` | Checks físicos de `pp.validate` como métricas de treino em Lightning, PhysicsNeMo e Hugging Face. | 1 | 3 | 4 |
| `lpm` | Blocos de um 'Large Physics Model': codificação espacial, código de geometria, ensemble e sinal fora-da-distribuição. | 1 | 6 | 5 |
| `physicsnemo_bridge` | Cola entre PhysicsNeMo e PINNeAPPle: UQ, gêmeo digital, active learning com retreino e operadores em grade. | 4 | 15 | 19 |
| `predictor` | Inferência pós-treino em lote e em grade, e gráficos de campos, erros e escoamentos. | 4 | 27 | 0 |
| `trainer` | Treinadores, perdas, métricas, balanceadores de perda, otimizadores e utilitários de paralelismo/HPC para treinar PINNs e surrogates. | 34 | 132 | 151 |
| `utils` | Helpers `as_tensor` (tensor de qualquer saída de modelo) e `fit_supervised` (laço Adam mínimo). | 1 | 2 | 5 |
| `workflows` | Fluxos prontos de treino e previsão: FNO3d em volumes densos (simples e paramétrico) e surrogate híbrido com pós-modelo físico. | 4 | 21 | 10 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_neural`

Raiz do pacote: importa `architectures`, `trainer` e `predictor`, mantém os apelidos antigos `models`, `train` e `inference`, reexporta os símbolos principais e define os atalhos `build_model`, `train_model`, `predict` e `train_and_predict`.

<small>`pinneapple_neural/__init__.py` · 256 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_model` | função | `(name, **kwargs)` | Atalho para `ModelRegistry.build(name, **kwargs)`. |
| `train_model` | função | `(model, loss_fn, train_loader, val_loader=None, *, epochs=5000, device='cpu', **cfg_kwarg…)` | Atalho que monta um `TrainConfig` e chama `Trainer.fit`; usa o loader de treino como validação se nenhum for dado. |
| `predict` | função | `(model, x, *, device='cpu', batch_size=4096)` | Move modelo e entrada para o dispositivo e roda `batched_inference`. |
| `train_and_predict` | função | `(model, loss_fn, train_loader, x_test, *, val_loader=None, epochs=5000, device='cpu')` | Treina com `train_model` e prevê em `x_test`; devolve um dicionário com os dois resultados. |

#### `pinneapple_neural._licencas`

Guardas de licença para integrações de terceiros só para pesquisa (hoje, Noether/Emmi AI): avisa com `ResearchOnlyWarning` e bloqueia com `ResearchOnlyError` quando `PINNEAPPLE_COMMERCIAL_MODE=1`.

<small>`pinneapple_neural/_licencas.py` · 44 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ResearchOnlyWarning` | classe | `()` | Aviso emitido quando um componente só para pesquisa é instanciado. |
| `ResearchOnlyError` | classe | `()` | Erro levantado ao pedir um componente só para pesquisa em modo comercial. |
| `commercial_mode` | função | `() -> bool` | Diz se a variável `PINNEAPPLE_COMMERCIAL_MODE` vale `1`. |
| `require_research_use` | função | `(component, notice=NOETHER_NOTICE) -> None` | Levanta `ResearchOnlyError` em modo comercial; fora dele, emite `ResearchOnlyWarning` com o aviso de licença. |

#### `pinneapple_neural.architectures`

Subpacote de arquiteturas (antigo `pinneapple_models`): reexporta `BaseModel`, `ModelRegistry`, `ModelCatalog`, `select_adapter`, SIREN, ModifiedMLP, HashGridMLP, AFNO e MeshGraphNet, e chama `register_all()` ao ser importado para popular o registro global.

<small>`pinneapple_neural/architectures/__init__.py` · 43 linhas · testes: 28</small>

#### `pinneapple_neural.architectures._registry_bridge`

Função auxiliar que copia um dicionário `nome -> classe` de uma família para o `ModelRegistry` global, pulando apelidos da mesma classe. [sem teste direto]

<small>`pinneapple_neural/architectures/_registry_bridge.py` · 39 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `register_family_registry` | função | `(family_registry, *, family, description_getter=None, tags_getter=None, capabilities_gett…)` | Registra cada classe de um dicionário de família no `ModelRegistry`, com descrição, tags e capacidades opcionais. |

#### `pinneapple_neural.architectures.adapters`

Adaptadores que chamam qualquer modelo a partir de um batch em dicionário, sem refatorar cada implementação; usados pela Arena via `select_adapter`. [sem teste direto]

<small>`pinneapple_neural/architectures/adapters/__init__.py` · 27 linhas · **sem teste direto**</small>

#### `pinneapple_neural.architectures.adapters.ae`

Adaptador para autoencoders: lê `batch['x']` como entrada a reconstruir. [sem teste direto]

<small>`pinneapple_neural/architectures/adapters/ae.py` · 28 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AutoEncoderAdapter` | classe | `()` | Adaptador que chama autoencoders com `batch['x']`. |

#### `pinneapple_neural.architectures.adapters.base`

Define o protocolo `ModelAdapter`, um adaptador padrão (usa `forward_batch` ou `model(x)`) e `select_adapter`, que testa os adaptadores em ordem. [sem teste direto]

<small>`pinneapple_neural/architectures/adapters/base.py` · 63 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ModelAdapter` | classe | `()` | Protocolo de adaptador: `can_handle(spec)` e `forward_batch(model, batch)`. |
| `select_adapter` | função | `(spec) -> ModelAdapter` | Devolve o primeiro adaptador que aceita o `ModelSpec`, ou o adaptador padrão. |

#### `pinneapple_neural.architectures.adapters.conv`

Adaptador para modelos de grade (Conv1D/2D/3D, AFNO, UNO em modo grade) que lê `batch['u_grid']`. [sem teste direto]

<small>`pinneapple_neural/architectures/adapters/conv.py` · 31 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ConvAdapter` | classe | `()` | Adaptador para modelos de grade que lê `u_grid`. |

#### `pinneapple_neural.architectures.adapters.dynamics`

Adaptador para modelos de evolução no tempo (Neural ODE, Latent ODE, ODE-RNN, CDE, SDE, Symplectic RNN) que lê `x0` e `t`. [sem teste direto]

<small>`pinneapple_neural/architectures/adapters/dynamics.py` · 32 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DynamicsAdapter` | classe | `()` | Adaptador para modelos de dinâmica que chama `forward(x0, t)`. |

#### `pinneapple_neural.architectures.adapters.gnn`

Adaptador para modelos em grafo/malha: passa `batch['graph']` ou os tensores `x`/`edge_index`. [sem teste direto]

<small>`pinneapple_neural/architectures/adapters/gnn.py` · 39 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GNNAdapter` | classe | `()` | Adaptador para modelos em grafo ou malha. |

#### `pinneapple_neural.architectures.adapters.operators`

Adaptador para operadores neurais (DeepONet, FNO-1D, GNO, UNO, PINO) que escolhe as chaves do batch conforme o tipo. [sem teste direto]

<small>`pinneapple_neural/architectures/adapters/operators.py` · 49 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OperatorAdapter` | classe | `()` | Adaptador para operadores neurais que monta a chamada certa para DeepONet, FNO, GNO, UNO ou PINO. |

#### `pinneapple_neural.architectures.adapters.pinn`

Adaptador para PINNs pontuais: usa `batch['x']` ou `batch['x_col']`, ou delega a `forward_batch`. [sem teste direto]

<small>`pinneapple_neural/architectures/adapters/pinn.py` · 37 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PINNAdapter` | classe | `()` | Adaptador para PINNs pontuais (entrada `x` ou `x_col`). |

#### `pinneapple_neural.architectures.adapters.ts`

Adaptador para modelos de série temporal: lê o histórico `x` (B, T, D) e chaves opcionais como `x_mark` e `y0`. [sem teste direto]

<small>`pinneapple_neural/architectures/adapters/ts.py` · 44 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TimeSeriesAdapter` | classe | `()` | Adaptador para modelos de série temporal (histórico `x` e chaves opcionais). |

#### `pinneapple_neural.architectures.afno`

Adaptive Fourier Neural Operator (Guibas et al.) para campos 2D em grade `(B, H, W, C)`: mistura de tokens bloco-diagonal no espaço de Fourier mais MLP de canais. [sem teste direto]

<small>`pinneapple_neural/architectures/afno.py` · 264 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AFNOLayer` | classe | `(hidden_dim, n_modes_h=12, n_modes_w=12, mlp_ratio=4.0, dropout=0.0, num_blocks=8, hidden…)` | Uma camada AFNO: mistura bloco-diagonal no espaço de Fourier com limiar de esparsidade mais MLP de canais. |
| `AFNO` | classe | `(in_channels, out_channels, hidden_dim=64, n_layers=4, n_modes_h=12, n_modes_w=12, mlp_ra…)` | Operador AFNO completo para campos 2D `(B, H, W, C)`: projeção de entrada, camadas AFNO e projeção de saída. |

#### `pinneapple_neural.architectures.autoencoders`

Família de autoencoders; reexporta `AutoencoderCatalog`, `AEBase` e `AEOutput`. [sem teste direto]

<small>`pinneapple_neural/architectures/autoencoders/__init__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_neural.architectures.autoencoders.ae_2d`

Autoencoder convolucional 2D para imagens e campos 2D, com saída redimensionada para (H, W) por interpolação. [sem teste direto]

<small>`pinneapple_neural/architectures/autoencoders/ae_2d.py` · 133 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Autoencoder2D` | classe | `(in_channels, latent_dim, img_size, base_channels=32, norm='group', gn_groups=8, output_a…)` | Autoencoder Conv2D para campos 2D, com latente vetorial. |

#### `pinneapple_neural.architectures.autoencoders.ae_rom_hybrid`

Autoencoder denso com dinâmica linear contínua no latente (dz/dt = K z + c, passo de Euler), perdas de previsão, rollout e estabilidade. [sem teste direto]

<small>`pinneapple_neural/architectures/autoencoders/ae_rom_hybrid.py` · 320 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AEROMHybrid` | classe | `(input_dim, latent_dim, hidden=(512, 256), activation='gelu', rom_weight=1.0, pred_weight…)` | Autoencoder com dinâmica latente linear (K, c), ajuste por mínimos quadrados e penalidade de estabilidade. |

#### `pinneapple_neural.architectures.autoencoders.base`

Contrato comum dos autoencoders: `encode`, `decode`, `forward` e `loss_from_parts`; os dois primeiros levantam `NotImplementedError` na base (classe abstrata).

<small>`pinneapple_neural/architectures/autoencoders/base.py` · 51 linhas · testes: 5 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AEOutput` | classe | `(x_hat, z, losses, extras)` | Saída padronizada dos autoencoders: `x_hat`, `z`, perdas e extras. |
| `AEBase` | classe | `()` | Classe base abstrata dos autoencoders (encode/decode/forward/perdas). |

#### `pinneapple_neural.architectures.autoencoders.dense_ae`

Autoencoder MLP simples para vetores, com decoder espelhado. [sem teste direto]

<small>`pinneapple_neural/architectures/autoencoders/dense_ae.py` · 54 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DenseAutoencoder` | classe | `(input_dim, latent_dim, hidden=(512, 256), activation='gelu')` | Autoencoder MLP para vetores. |

#### `pinneapple_neural.architectures.autoencoders.kae`

Autoencoder estilo MMD-WAE: penalidade MMD (kernel RBF, sem viés) no latente para casar com um prior simples. [sem teste direto]

<small>`pinneapple_neural/architectures/autoencoders/kae.py` · 177 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `KAEAutoencoder` | classe | `(input_dim, latent_dim, hidden=(512, 256), activation='gelu', mmd_weight=1.0, mmd_sigma=1…)` | Autoencoder com penalidade MMD no latente (um ou vários kernels RBF; banda pela mediana se `mmd_sigma <= 0`). |

#### `pinneapple_neural.architectures.autoencoders.koopman_pi_ae`

Autoencoder de Koopman: operador linear treinável no latente, opção de gerador contínuo (expm(A*dt)), perdas de latente, previsão e rollout. [sem teste direto]

<small>`pinneapple_neural/architectures/autoencoders/koopman_pi_ae.py` · 331 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsInformedKoopmanAutoencoder` | classe | `(input_dim, latent_dim, hidden=(512, 256), activation='gelu', koopman_weight=1.0, *, pred…)` | Autoencoder de Koopman com operador linear no latente e perdas de consistência e previsão. |

#### `pinneapple_neural.architectures.autoencoders.registry`

Registro da família de autoencoders (com apelidos) e `AutoencoderCatalog`; registra no `ModelRegistry` com `input_kind='autoencoder'`. [sem teste direto]

<small>`pinneapple_neural/architectures/autoencoders/registry.py` · 70 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AutoencoderCatalog` | classe | `(registry)` | Catálogo da família: `list`, `get` e `build` por nome. |
| `register_into_global` | função | `() -> None` | Registra os autoencoders no `ModelRegistry` global. |

#### `pinneapple_neural.architectures.autoencoders.vae`

Autoencoder variacional MLP (beta-VAE) para vetores, com log-variância limitada. [sem teste direto]

<small>`pinneapple_neural/architectures/autoencoders/vae.py` · 122 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `VariationalAutoencoder` | classe | `(input_dim, latent_dim, hidden=(512, 256), beta=1.0, activation='gelu', clamp_logvar=(-10…)` | VAE MLP com reparametrização e peso `beta` no termo KL. |

#### `pinneapple_neural.architectures.base`

Contrato mínimo dos modelos: `ModelOutput` e `BaseModel` com `forward` (abstrato, levanta `NotImplementedError`), `forward_batch` padrão e checkpoints.

<small>`pinneapple_neural/architectures/base.py` · 93 linhas · testes: 5 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ModelOutput` | classe | `(y, losses, extras)` | Saída padrão de modelo: `y`, perdas e extras. |
| `BaseModel` | classe | `(family, name)` | Classe base de todos os modelos: `forward` abstrato, `forward_batch` padrão e salvar/carregar checkpoint. |

#### `pinneapple_neural.architectures.benchmarks`

Família de modelos de referência para benchmarks (MLPs genéricos). [sem teste direto]

<small>`pinneapple_neural/architectures/benchmarks/__init__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_neural.architectures.benchmarks.generic_pinn_models`

Modelos-base genéricos para comparar arquiteturas: MLP, MLP com Fourier features, SIREN, MLP residual e linear. [sem teste direto]

<small>`pinneapple_neural/architectures/benchmarks/generic_pinn_models.py` · 122 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FourierFeatures` | classe | `(in_dim, num_f=64, scale=10.0)` | Camada de features aleatórias de Fourier com escala fixa. |
| `GenericMLP` | classe | `(in_dim=3, out_dim=1, width=128, depth=4, act='tanh')` | MLP genérico de referência (tanh por padrão). |
| `GenericFourierMLP` | classe | `(in_dim=3, out_dim=1, num_f=64, scale=10.0, width=128, depth=4)` | MLP genérico com Fourier features na entrada. |
| `Sine` | classe | `(w0=30.0)` | Ativação seno com frequência `w0`. |
| `GenericSIREN` | classe | `(in_dim=3, out_dim=1, width=128, depth=4, w0=30.0)` | SIREN genérica de referência. |
| `GenericResMLP` | classe | `(in_dim=3, out_dim=1, width=128, depth=6)` | MLP residual genérico de referência. |
| `GenericLinear` | classe | `(in_dim=3, out_dim=1)` | Modelo linear de referência. |

#### `pinneapple_neural.architectures.benchmarks.registry`

Registra os modelos de benchmark no `ModelRegistry` como `bench_mlp`, `bench_fourier_mlp`, `bench_siren`, `bench_res_mlp` e `bench_linear`. [sem teste direto]

<small>`pinneapple_neural/architectures/benchmarks/registry.py` · 32 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `register_into_global` | função | `() -> None` | Registra os cinco modelos de benchmark no `ModelRegistry` global. |

#### `pinneapple_neural.architectures.catalog`

`ModelCatalog`: agrega os catálogos de todas as famílias num único objeto. [sem teste direto]

<small>`pinneapple_neural/architectures/catalog.py` · 33 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ModelCatalog` | classe | `(autoencoders, pinns, transformers, recurrent, convolutions, neural_operators, continuous…)` | Dataclass com um catálogo por família (autoencoders, pinns, transformers, recurrent, convolutions, neural_operators, continuous, graphnn, reservoir_computing, classical_ts, rom, physics_aware). |

#### `pinneapple_neural.architectures.classical_ts`

Família de modelos clássicos de série temporal e filtragem. [sem teste direto]

<small>`pinneapple_neural/architectures/classical_ts/__init__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_neural.architectures.classical_ts.arima`

ARIMA(p,d,q) implementado sem dependências externas, ajustado por regressão. [sem teste direto]

<small>`pinneapple_neural/architectures/classical_ts/arima.py` · 250 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ARIMA` | classe | `(dim=1, p=3, d=0, q=0, l2=1e-06, use_bias=True)` | Modelo ARIMA(p,d,q) com `fit`, `forecast` e `forward`, sem bibliotecas externas. |

#### `pinneapple_neural.architectures.classical_ts.base`

Saída `ClassicalTSOutput` e base `ClassicalTSBase` com `mse` para a família. [sem teste direto]

<small>`pinneapple_neural/architectures/classical_ts/base.py` · 34 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ClassicalTSOutput` | classe | `(y, losses, extras)` | Saída padrão dos modelos clássicos de série temporal. |
| `ClassicalTSBase` | classe | `()` | Base da família com o helper `mse`. |

#### `pinneapple_neural.architectures.classical_ts.ekf`

Filtro de Kalman estendido em lote, com funções de transição/observação e jacobianos dados pelo usuário. [sem teste direto]

<small>`pinneapple_neural/architectures/classical_ts/ekf.py` · 125 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ExtendedKalmanFilter` | classe | `(f, h, Q, R, F_jac, H_jac)` | Filtro de Kalman estendido com jacobianos `F_jac` e `H_jac`. |

#### `pinneapple_neural.architectures.classical_ts.enkf`

Filtro de Kalman por ensemble (EnKF). [sem teste direto]

<small>`pinneapple_neural/architectures/classical_ts/enkf.py` · 122 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EnsembleKalmanFilter` | classe | `(f, h, Q, R, *, ensemble_size=64)` | Filtro de Kalman por ensemble com `ensemble_size` membros. |

#### `pinneapple_neural.architectures.classical_ts.kalman`

Filtro de Kalman linear em lote, com controle `B` opcional. [sem teste direto]

<small>`pinneapple_neural/architectures/classical_ts/kalman.py` · 130 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `KalmanFilter` | classe | `(A, H, Q, R, *, B=None)` | Filtro de Kalman linear em lote. |

#### `pinneapple_neural.architectures.classical_ts.registry`

Registro da família `classical_ts` e `ClassicalTSCatalog`. [sem teste direto]

<small>`pinneapple_neural/architectures/classical_ts/registry.py` · 67 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ClassicalTSCatalog` | classe | `(registry)` | Catálogo da família `classical_ts`. |
| `register_into_global` | função | `() -> None` | Registra a família `classical_ts` no `ModelRegistry` global. |

#### `pinneapple_neural.architectures.classical_ts.tcn`

Rede convolucional temporal (TCN) para regressão sequência a sequência. [sem teste direto]

<small>`pinneapple_neural/architectures/classical_ts/tcn.py` · 89 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TCN` | classe | `(in_dim, out_dim, *, channels=(64, 64, 64), kernel=3, dropout=0.1)` | Rede convolucional temporal para regressão sequência a sequência. |

#### `pinneapple_neural.architectures.classical_ts.ukf`

Filtro de Kalman unscented (pontos sigma) em lote, com jitter crescente na fatoração de Cholesky. [sem teste direto]

<small>`pinneapple_neural/architectures/classical_ts/ukf.py` · 254 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `UnscentedKalmanFilter` | classe | `(f, h, Q, R, *, alpha=0.001, beta=2.0, kappa=0.0, jitter=1e-06, jitter_growth=10.0, max_c…)` | Filtro de Kalman unscented em lote. |

#### `pinneapple_neural.architectures.classical_ts.var`

Modelo VAR(p) ajustado por regressão ridge em forma fechada. [sem teste direto]

<small>`pinneapple_neural/architectures/classical_ts/var.py` · 165 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `VAR` | classe | `(dim, p=1, l2=1e-06, use_bias=True)` | Modelo VAR(p) ajustado por ridge em forma fechada. |

#### `pinneapple_neural.architectures.continuous`

Família de modelos de tempo contínuo e probabilísticos (ODE/SDE/CDE neurais, hamiltonianos, GP neural). [sem teste direto]

<small>`pinneapple_neural/architectures/continuous/__init__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_neural.architectures.continuous.base`

Saída `ContOutput` e base `ContinuousModelBase` com `mse` e `gaussian_nll`. [sem teste direto]

<small>`pinneapple_neural/architectures/continuous/base.py` · 38 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ContOutput` | classe | `(y, losses, extras)` | Saída padrão dos modelos de tempo contínuo. |
| `ContinuousModelBase` | classe | `()` | Base da família com `mse` e `gaussian_nll`. |

#### `pinneapple_neural.architectures.continuous.bayesian_rnn`

RNN bayesiana aproximada por MC Dropout, com dropout travado ao longo do tempo. [sem teste direto]

<small>`pinneapple_neural/architectures/continuous/bayesian_rnn.py` · 200 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LockedDropout` | classe | `(p=0.0)` | Dropout com a mesma máscara em todos os passos de tempo. |
| `BayesianRNN` | classe | `(input_dim, out_dim, hidden_dim=128, num_layers=2, dropout=0.1, cell='gru', min_logvar=-1…)` | RNN (GRU/LSTM) com média e log-variância; `predict_mc` faz amostragem por MC Dropout. |

#### `pinneapple_neural.architectures.continuous.deep_state_space`

Modelo de espaço de estados profundo variacional (filtragem), com NLL gaussiana e KL entre gaussianas diagonais. [sem teste direto]

<small>`pinneapple_neural/architectures/continuous/deep_state_space.py` · 214 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ContOutput` | classe | `(y, losses, extras)` | Cópia local da saída padrão (`y`, perdas, extras) usada pelo modelo de espaço de estados. |
| `DeepStateSpaceModel` | classe | `(input_dim, out_dim, latent_dim=32, hidden_dim=128, mlp_depth=2, min_logvar=-10.0, max_lo…)` | Modelo de espaço de estados profundo variacional treinado por ELBO (NLL + `beta_kl` * KL). |
| `gaussian_nll_diag` | função | `(mu, logvar, y) -> torch.Tensor` | NLL de uma gaussiana diagonal. |
| `kl_diag_gaussians` | função | `(mu_q, logvar_q, mu_p, logvar_p) -> torch.Tensor` | Divergência KL entre duas gaussianas diagonais. |

#### `pinneapple_neural.architectures.continuous.hamiltonian`

Hamiltonian Neural Network: aprende H(q,p) e deriva a dinâmica por autograd. [sem teste direto]

<small>`pinneapple_neural/architectures/continuous/hamiltonian.py` · 69 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HamiltonianNeuralNetwork` | classe | `(dim_q, hidden=128, num_layers=3)` | Rede que aprende H(q,p) e devolve dq/dt = dH/dp, dp/dt = -dH/dq. |

#### `pinneapple_neural.architectures.continuous.latent_ode`

Latent ODE: encoder GRU-ODE para tempos irregulares, dinâmica latente por ODE neural e decoder gaussiano. [sem teste direto]

<small>`pinneapple_neural/architectures/continuous/latent_ode.py` · 249 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LatentODE` | classe | `(obs_dim, latent_dim=32, hidden=128, *, ode_method='dopri5', enc_method='dopri5', min_log…)` | Latent ODE com `forward` (reconstrução e NLL) e `forecast`. |

#### `pinneapple_neural.architectures.continuous.neural_cde`

Equação diferencial controlada neural (CDE) sem depender de `torchcde`. [sem teste direto]

<small>`pinneapple_neural/architectures/continuous/neural_cde.py` · 156 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NeuralCDE` | classe | `(input_dim, hidden_dim, out_dim, *, f_hidden=128, g_hidden=128, num_layers=2, activation=…)` | CDE neural com solver Euler por padrão. |

#### `pinneapple_neural.architectures.continuous.neural_gp`

Processo gaussiano neural: features aleatórias de Fourier e cabeça linear bayesiana com posterior condicionável. [sem teste direto]

<small>`pinneapple_neural/architectures/continuous/neural_gp.py` · 248 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NeuralGaussianProcess` | classe | `(in_dim, out_dim, *, embed_dim=64, mlp_hidden=128, mlp_layers=2, rff_features=512, rff_le…)` | GP neural com features RFF e cabeça linear bayesiana; `condition` atualiza o posterior. |

#### `pinneapple_neural.architectures.continuous.neural_ode`

Neural ODE simples integrada numa grade de tempos com Euler ou RK4. [sem teste direto]

<small>`pinneapple_neural/architectures/continuous/neural_ode.py` · 91 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NeuralODE` | classe | `(state_dim, hidden=128, num_layers=3, method='rk4')` | ODE neural dy/dt = f(t, y) integrada com Euler ou RK4. |

#### `pinneapple_neural.architectures.continuous.neural_sde`

SDE neural integrada por Euler–Maruyama, com difusão diagonal ou completa e vários caminhos. [sem teste direto]

<small>`pinneapple_neural/architectures/continuous/neural_sde.py` · 264 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NeuralSDE` | classe | `(state_dim, *, hidden=128, num_layers=3, diffusion='diag', activation='tanh', sigma_eps=1…)` | SDE neural com deriva e difusão aprendidas, simulada por Euler–Maruyama. |

#### `pinneapple_neural.architectures.continuous.ode_rnn`

ODE-RNN: integra o estado oculto entre observações e atualiza com `GRUCell`. [sem teste direto]

<small>`pinneapple_neural/architectures/continuous/ode_rnn.py` · 54 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ODERNN` | classe | `(input_dim, hidden_dim=128)` | ODE-RNN para séries com observações irregulares. |

#### `pinneapple_neural.architectures.continuous.registry`

Registro da família `continuous` e `ContinuousCatalog`. [sem teste direto]

<small>`pinneapple_neural/architectures/continuous/registry.py` · 80 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ContinuousCatalog` | classe | `(registry)` | Catálogo da família `continuous`. |
| `register_into_global` | função | `() -> None` | Registra a família `continuous` no `ModelRegistry` global. |

#### `pinneapple_neural.architectures.continuous.symplectic_ode`

Symplectic ODE-Net: hamiltoniano aprendido com passo simplético e rollout. [sem teste direto]

<small>`pinneapple_neural/architectures/continuous/symplectic_ode.py` · 147 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SymplecticODENet` | classe | `(dim_q, hidden=128)` | Rede com hamiltoniano aprendido, campo vetorial, passo simplético e `rollout`. |

#### `pinneapple_neural.architectures.continuous.symplectic_rnn`

RNN simplética com atualização tipo leapfrog (energias cinética e potencial separadas). [sem teste direto]

<small>`pinneapple_neural/architectures/continuous/symplectic_rnn.py` · 128 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SymplecticRNN` | classe | `(dim_q, *, hidden=128, num_layers=2, activation='tanh')` | RNN simplética com gradientes de T(p) e V(q) aprendidos. |

#### `pinneapple_neural.architectures.convolutions`

Família de modelos convolucionais 1D/2D/3D. [sem teste direto]

<small>`pinneapple_neural/architectures/convolutions/__init__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_neural.architectures.convolutions.base`

Saída `ConvOutput` e base `ConvModelBase` com `mse`. [sem teste direto]

<small>`pinneapple_neural/architectures/convolutions/base.py` · 36 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ConvOutput` | classe | `(y, losses, extras)` | Saída padrão dos modelos convolucionais. |
| `ConvModelBase` | classe | `()` | Base da família convolucional com `mse`. |

#### `pinneapple_neural.architectures.convolutions.conv1d`

Modelo Conv1D flexível: previsão causal, remoção de ruído, aprendizado de operador e surrogate de EDP 1D. [sem teste direto]

<small>`pinneapple_neural/architectures/convolutions/conv1d.py` · 265 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Conv1DModel` | classe | `(in_channels, out_channels, *, mode='denoise', cond_channels=0, hidden_channels=64, num_b…)` | Modelo Conv1D com modos `denoise`, causal, operador e surrogate de EDP 1D. |

#### `pinneapple_neural.architectures.convolutions.conv2d`

Rede Conv2D residual (estilo ResNet) para regressão de campos. [sem teste direto]

<small>`pinneapple_neural/architectures/convolutions/conv2d.py` · 148 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Conv2DModel` | classe | `(in_channels, out_channels, hidden_channels=32, num_blocks=4, kernel_size=3, dropout=0.0,…)` | CNN 2D residual para regressão. |

#### `pinneapple_neural.architectures.convolutions.conv3d`

Rede Conv3D residual para regressão de volumes, com skip global opcional. [sem teste direto]

<small>`pinneapple_neural/architectures/convolutions/conv3d.py` · 174 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ResidualBlock3D` | classe | `(channels, kernel_size=3, dropout=0.0, norm_groups=8) -> None` | Bloco residual 3D (Conv, GroupNorm, GELU, Dropout); a soma residual fica a cargo de quem chama. |
| `Conv3DModel` | classe | `(in_channels, out_channels, hidden_channels=32, num_blocks=4, kernel_size=3, dropout=0.0,…)` | CNN 3D residual para regressão de volumes. |

#### `pinneapple_neural.architectures.convolutions.registry`

Registro da família `convolutions` e `ConvolutionCatalog`. [sem teste direto]

<small>`pinneapple_neural/architectures/convolutions/registry.py` · 60 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ConvolutionCatalog` | classe | `(registry)` | Catálogo da família `convolutions`. |
| `register_into_global` | função | `() -> None` | Registra a família `convolutions` no `ModelRegistry` global. |

#### `pinneapple_neural.architectures.graphnn`

Família de redes em grafo; reexporta `GraphModelBase`, `GraphBatch`, `GraphOutput`, `MeshGraphNet`, `MeshDynamicsMGN`, `MeshGraph`, `Normalizer` e `GraphCatalog`. [sem teste direto]

<small>`pinneapple_neural/architectures/graphnn/__init__.py` · 8 linhas · **sem teste direto**</small>

#### `pinneapple_neural.architectures.graphnn.base`

Formato mínimo de batch de grafo (`GraphBatch`, sem dependências externas), saída `GraphOutput` e base `GraphModelBase`.

<small>`pinneapple_neural/architectures/graphnn/base.py` · 54 linhas · testes: 15</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GraphBatch` | classe | `(x, edge_index, pos, edge_attr, mask)` | Batch de grafo com `x`, `edge_index` e, opcionalmente, `pos`, `edge_attr` e `mask`. |
| `GraphOutput` | classe | `(y, losses, extras)` | Saída padrão das GNNs: `y` por nó, perdas e extras. |
| `GraphModelBase` | classe | `()` | Base das redes em grafo com `mse`. |

#### `pinneapple_neural.architectures.graphnn.equivariant_gnn`

GNN E(n)-equivariante no estilo EGNN, que pode atualizar as posições dos nós.

<small>`pinneapple_neural/architectures/graphnn/equivariant_gnn.py` · 229 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EquivariantGNN` | classe | `(node_dim, pos_dim, out_dim, *, hidden=128, layers=4, edge_dim=0, dropout=0.0, update_pos…)` | GNN E(n)-equivariante (EGNN) com atualização opcional das posições. |

#### `pinneapple_neural.architectures.graphnn.gnn`

Rede de passagem de mensagens (MPNN) simples para previsão por nó.

<small>`pinneapple_neural/architectures/graphnn/gnn.py` · 123 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GraphNeuralNetwork` | classe | `(node_dim, out_dim, *, edge_dim=0, hidden=128, layers=4, dropout=0.0)` | MPNN simples com features de aresta opcionais. |

#### `pinneapple_neural.architectures.graphnn.gnn_ode`

ODE neural em grafo: o campo vetorial é uma GNN, com features de Fourier do tempo opcionais; os métodos `dopri5`/`adams` exigem `torchdiffeq`. [sem teste direto]

<small>`pinneapple_neural/architectures/graphnn/gnn_ode.py` · 212 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FourierTimeFeatures` | classe | `(n_frequencies, *, max_frequency=10.0)` | Features [sin, cos](2π f t) do tempo escalar. |
| `GraphNeuralODE` | classe | `(node_dim, *, edge_dim=0, hidden=128, layers=3, method='rk4', time_features=0, time_max_f…)` | ODE neural em grafo integrada com RK4 por padrão ou com `torchdiffeq` (`dopri5`, `adams`, adjunto opcional). |

#### `pinneapple_neural.architectures.graphnn.graphcast`

Rede inspirada no GraphCast (encoder, processador em malha, decoder) para malha fixa ou grafos variáveis. [sem teste direto]

<small>`pinneapple_neural/architectures/graphnn/graphcast.py` · 289 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GraphCast` | classe | `(node_in, out_dim, *, hidden=256, processor_blocks=8, pos_dim=0, edge_dim=0, dropout=0.0)` | Modelo inspirado no GraphCast com blocos de processamento em malha. |

#### `pinneapple_neural.architectures.graphnn.mesh_graph_net`

MeshGraphNet (Pfaff et al., ICLR 2021) encoder-processador-decoder com um único tipo de aresta (malha fixa); o conjunto de arestas no espaço do mundo, usado para contato, não está implementado. O inventário marca 'placeholder', mas o código é uma implementação real.

<small>`pinneapple_neural/architectures/graphnn/mesh_graph_net.py` · 375 linhas · testes: 15 · palavras-chave no código (heurística; o texto acima diz o status real): placeholder/stub citado no docstring</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MeshGraphNet` | classe | `(node_in_dim, out_dim, *, edge_in_dim=0, pos_dim=0, hidden_dim=128, n_layers=2, n_message…)` | GNN encoder-processador-decoder para simulação em malha não estruturada com conectividade fixa. |

#### `pinneapple_neural.architectures.graphnn.mgn_dynamics`

Dinâmica transiente sobre `MeshGraphNet`: malha estática, normalizadores, passo autorregressivo de (velocidade, pressão), treino de um passo e avaliação de rollout.

<small>`pinneapple_neural/architectures/graphnn/mgn_dynamics.py` · 270 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MeshGraph` | classe | `(edge_index, edge_attr, node_type)` | Uma malha estática: topologia, features brutas das arestas e tipo de cada nó; `from_mesh` monta a partir de triângulos. |
| `Normalizer` | classe | `(dim, eps=1e-08) -> None` | Padronizador por feature com estatísticas fixas guardadas como buffers. |
| `MeshDynamicsMGN` | classe | `(vel_dim=2, num_node_types=4, edge_dim=3, out_p=True, **mgn_kwargs) -> None` | MeshGraphNet autorregressivo para (velocidade, pressão) em malha transiente: `step`, `loss` e `rollout`. |
| `triangles_to_edges` | função | `(cells) -> torch.Tensor` | Gera `edge_index` bidirecional e sem duplicatas a partir de triângulos. |
| `edge_features` | função | `(pos, edge_index) -> torch.Tensor` | Deslocamento relativo entre nós da aresta e sua norma. |
| `train_dynamics` | função | `(model, data, *, steps, lr=0.0001, lr_final=1e-06, noise_std=0.02, device='cpu', log_ever…)` | Treino de um passo em pares (trajetória, tempo) aleatórios, com Adam, ruído de entrada e decaimento exponencial da taxa de aprendizado. |
| `eval_rollout` | função | `(model, data, *, n_steps, device='cpu', normal_type=0) -> dict[str, float]` | RMSE da velocidade nos nós NORMAL, em rollout livre e em um passo, média sobre trajetórias. |
| `frozen_ic_rmse` | função | `(data, *, n_steps, normal_type=0) -> float` | RMSE de referência quando a velocidade fica congelada no valor inicial. |

#### `pinneapple_neural.architectures.graphnn.registry`

Registro da família `graphnn` e `GraphCatalog`. [sem teste direto]

<small>`pinneapple_neural/architectures/graphnn/registry.py` · 64 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GraphCatalog` | classe | `(registry)` | Catálogo da família `graphnn`. |
| `register_into_global` | função | `() -> None` | Registra a família `graphnn` no `ModelRegistry` global. |

#### `pinneapple_neural.architectures.graphnn.spatiotemporal_gnn`

GNN espaço-temporal: GNN em cada instante seguida de uma GRU por nó ao longo do tempo. [sem teste direto]

<small>`pinneapple_neural/architectures/graphnn/spatiotemporal_gnn.py` · 117 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SpatiotemporalGNN` | classe | `(node_in, out_dim, *, edge_dim=0, spatial_hidden=128, spatial_layers=3, temporal_hidden=1…)` | GNN espacial em cada instante, GRU por nó ao longo do tempo e cabeça por nó. |

#### `pinneapple_neural.architectures.graphnn.utils`

Utilitários de scatter-add e de achatamento de vários grafos num grafo único com deslocamento de índices. [sem teste direto]

<small>`pinneapple_neural/architectures/graphnn/utils.py` · 84 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `scatter_add` | função | `(src, index, dim_size) -> torch.Tensor` | Scatter-add para grafos compartilhados no batch. |
| `scatter_add_flat` | função | `(src, index, dim_size) -> torch.Tensor` | Scatter-add para um grafo achatado (batch de grafos variáveis). |
| `flatten_graph_batch` | função | `(x_list, edge_index_list, pos_list=None, edge_attr_list=None)` | Junta grafos por amostra num único grafo, deslocando os índices dos nós. |

#### `pinneapple_neural.architectures.group_b_registry`

Registro do 'grupo B' (SIREN, ModifiedMLP, HashGridMLP, AFNO) com apelidos e capacidades; MeshGraphNet fica na família `graphnn`. [sem teste direto]

<small>`pinneapple_neural/architectures/group_b_registry.py` · 86 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GroupBCatalog` | classe | `(registry)` | Catálogo do grupo B (SIREN, ModifiedMLP, HashGridMLP, AFNO). |
| `register_into_global` | função | `() -> None` | Registra o grupo B no `ModelRegistry` com a família `group_b`. |

#### `pinneapple_neural.architectures.hash_grid`

Codificação multirresolução por tabela hash (estilo Instant-NGP) e MLP que a usa para campos 3D. [sem teste direto]

<small>`pinneapple_neural/architectures/hash_grid.py` · 219 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HashGridEncoding` | classe | `(in_dim=3, n_levels=16, n_features_per_level=2, log2_hashmap_size=19, base_resolution=16,…)` | Codificação hash multirresolução com interpolação entre os cantos do voxel em cada nível. |
| `HashGridMLP` | classe | `(in_dim=3, out_dim=1, hidden_dim=64, n_hidden=2, n_levels=8, n_features_per_level=2, log2…)` | MLP pequeno sobre a codificação hash, para representar campos 3D. |

#### `pinneapple_neural.architectures.instantiate`

Normaliza nomes de kwargs entre famílias (`in_dim`, `input_dim`, `in_channels`...), traduz `hidden_dim`/`n_layers` para os nomes de cada classe e descarta argumentos não aceitos. [sem teste direto, mas usado por `ModelRegistry.build`]

<small>`pinneapple_neural/architectures/instantiate.py` · 173 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `InstantiateReport` | classe | `(original, normalized, dropped, used_aliases)` | Diagnóstico da normalização: kwargs originais, normalizados, descartados e apelidos usados. |
| `normalize_kwargs` | função | `(kwargs) -> Tuple[Dict[str, Any], Dict[str, str]]` | Converte apelidos de kwargs para os nomes canônicos. |
| `filter_supported_kwargs` | função | `(model_cls, kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]` | Mantém só os kwargs aceitos por `model_cls.__init__` (nenhum é descartado se ele aceita `**kwargs`). |
| `instantiate` | função | `(model_cls, kwargs=None, *, strict=False) -> Tuple[BaseModel, InstantiateReport]` | Instancia a classe com kwargs normalizados e devolve o modelo e um `InstantiateReport`; com `strict=True` falha se algo for descartado. |

#### `pinneapple_neural.architectures.modified_mlp`

MLP modificado (Wang et al.): embedding de Fourier aleatório e conexões tipo highway U/V. [sem teste direto]

<small>`pinneapple_neural/architectures/modified_mlp.py` · 136 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FourierFeatureEmbedding` | classe | `(in_dim, n_fourier=64, sigma=1.0, trainable=False)` | Embedding [cos, sin](2π B x) com matriz `B` aleatória ou treinável. |
| `ModifiedMLP` | classe | `(in_dim, out_dim, hidden_dim=128, n_layers=6, n_fourier=32, sigma=1.0, activation=nn.Tanh…)` | MLP com Fourier features e portas U/V tipo highway em cada camada. |

#### `pinneapple_neural.architectures.neural_operators`

Família de operadores neurais; reexporta FNO, FNO2d, MLPFNOSurrogate, Transolver e as classes Noether (que ficam `None` se `emmiai-noether` não estiver instalado).

<small>`pinneapple_neural/architectures/neural_operators/__init__.py` · 51 linhas · testes: 2</small>

#### `pinneapple_neural.architectures.neural_operators.base`

Saída `OperatorOutput` e base `NeuralOperatorBase` com `mse`.

<small>`pinneapple_neural/architectures/neural_operators/base.py` · 40 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OperatorOutput` | classe | `(y, losses, extras)` | Saída padrão dos operadores neurais. |
| `NeuralOperatorBase` | classe | `()` | Base dos operadores neurais com `mse`. |

#### `pinneapple_neural.architectures.neural_operators.deeponet`

DeepONet não empilhado (Lu et al. 2021): rede branch e rede trunk, um conjunto de coeficientes por canal de saída.

<small>`pinneapple_neural/architectures/neural_operators/deeponet.py` · 79 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DeepONet` | classe | `(branch_dim, trunk_dim, out_dim, hidden=128, modes=64, depth=1, trunk_activation=False)` | DeepONet branch-trunk não empilhado, com um conjunto de coeficientes por canal de saída. |
| `mlp` | função | `(in_dim, hidden, out_dim, depth=1, act=nn.GELU, last_act=False) -> nn.Sequential` | Monta um MLP a partir de uma lista de larguras ou de `depth` camadas de largura fixa. |

#### `pinneapple_neural.architectures.neural_operators.fno`

Fourier Neural Operators 1D, 2D e 3D e um híbrido FNO + decoder MLP para avaliar em pontos espalhados.

<small>`pinneapple_neural/architectures/neural_operators/fno.py` · 530 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FourierNeuralOperator` | classe | `(in_channels, out_channels, width=64, modes=16, layers=4, *, use_grid=False)` | FNO 1D. |
| `FNO2d` | classe | `(in_channels, out_channels, width=32, modes1=12, modes2=12, layers=4, *, use_grid=True)` | FNO 2D para campos em grade, com canais de coordenadas opcionais. |
| `FNO3d` | classe | `(in_channels, out_channels, width=32, modes1=8, modes2=8, modes3=8, layers=4, *, use_grid…)` | FNO 3D para campos 3D ou 2D + tempo. |
| `MLPFNOSurrogate` | classe | `(in_channels, out_channels, fno_width=32, fno_modes=12, fno_layers=4, mlp_hidden=128, mlp…)` | Backbone FNO com decoder MLP para avaliar em pontos espalhados. |

#### `pinneapple_neural.architectures.neural_operators.fno_export`

Exportação ONNX de FNOs: troca as convoluções espectrais 2D por versões em aritmética real e confere a saída com `onnxruntime`.

<small>`pinneapple_neural/architectures/neural_operators/fno_export.py` · 121 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RealSpectralConv2d` | classe | `(conv, H, W)` | Convolução espectral 2D reescrita em aritmética real para uma grade H x W fixa (exportável para ONNX). |
| `to_exportable` | função | `(model, grid) -> nn.Module` | Copia o modelo trocando cada convolução espectral 2D pela versão real. |
| `export_onnx` | função | `(model, example, path, *, grid=None, opset=17, atol=0.0001) -> float` | Exporta para ONNX com eixo de batch dinâmico, compara com `onnxruntime` e devolve a maior diferença absoluta (falha acima de `atol`). |

#### `pinneapple_neural.architectures.neural_operators.gno`

Galerkin Neural Operator em malha: aprende uma base Φ(x), projeta com pesos de quadratura, aplica o operador nos coeficientes e reconstrói. [sem teste direto]

<small>`pinneapple_neural/architectures/neural_operators/gno.py` · 146 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GalerkinNeuralOperator` | classe | `(in_dim, out_dim, basis_dim=64, basis_hidden=128, op_hidden=128, op_depth=2, orthonormali…)` | Operador de Galerkin com base aprendida e projeção por quadratura. |

#### `pinneapple_neural.architectures.neural_operators.ms_deeponet`

DeepONet multiescala: um trunk por escala com coordenadas esticadas (ideia do MscaleDNN).

<small>`pinneapple_neural/architectures/neural_operators/ms_deeponet.py` · 69 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MultiScaleDeepONet` | classe | `(branch_dim, trunk_dim, out_dim, *, hidden=128, scales=(32, 64, 128), scale_factors=None,…)` | DeepONet com um trunk por escala espacial e branch comum. |

#### `pinneapple_neural.architectures.neural_operators.noether_bridge`

[experimental; só pesquisa] Envolve os modelos de `emmiai-noether` (UPT, AB-UPT, Transformer, Transolver e variantes aero) como `NeuralOperatorBase`; cada instância passa por `require_research_use` e é recusada em modo comercial.

<small>`pinneapple_neural/architectures/neural_operators/noether_bridge.py` · 296 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NoetherModelConfig` | classe | `(model_name, config_kwargs, device)` | Config mínima repassada sem mudanças às classes de config do Noether. |
| `NoetherUPT` | classe | `(**config_kwargs)` | Wrapper do Universal Physics Transformer do Noether (só pesquisa). |
| `NoetherABUPT` | classe | `(**config_kwargs)` | Wrapper do AB-UPT do Noether (só pesquisa; pesos CC BY-NC). |
| `NoetherTransformer` | classe | `(**config_kwargs)` | Wrapper do Transformer simples do Noether (só pesquisa). |
| `NoetherTransolver` | classe | `(**config_kwargs)` | Wrapper do Transolver do Noether (só pesquisa). |
| `NoetherAeroUPT` | classe | `(**config_kwargs)` | Wrapper do UPT com roteamento superfície/volume para aerodinâmica (só pesquisa). |
| `NoetherAeroABUPT` | classe | `(**config_kwargs)` | Wrapper do AB-UPT com roteamento superfície/volume (só pesquisa). |
| `NoetherAeroTransformer` | classe | `(**config_kwargs)` | Wrapper do Transformer para CFD aerodinâmico (só pesquisa). |
| `NoetherAeroTransolver` | classe | `(**config_kwargs)` | Wrapper do Transolver para CFD aerodinâmico (só pesquisa). |

#### `pinneapple_neural.architectures.neural_operators.pino`

PINO: wrapper que soma perdas físicas a um operador, mais derivadas espectrais/diferenças finitas em grade uniforme e uma perda de resíduo genérica. [sem teste direto]

<small>`pinneapple_neural/architectures/neural_operators/pino.py` · 232 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsInformedNeuralOperator` | classe | `(operator, *, lambda_pde=1.0, lambda_bc=1.0, lambda_ic=1.0, lambda_physics=1.0, prefix='p…)` | Wrapper PINO: roda o operador e soma as perdas físicas (EDP, contorno, inicial) pesadas. |
| `GridSpec` | classe | `(L, dims)` | Descreve a grade uniforme: tamanho físico por eixo e quais dimensões de `u` são eixos. |
| `ResidualSpec` | classe | `(residual, orders, method)` | Especifica o resíduo da EDP, as ordens de derivada e o método (espectral ou diferenças finitas). |
| `pino_physics_fn` | função | `(u, *, grid, spec, params=None, bc_mask=None, bc_value=None, ic_mask=None, ic_value=None,…)` | Calcula derivadas na grade, avalia o resíduo e devolve as perdas `pde`, `bc`, `ic` e `physics_total`. |
| `spectral_derivative` | função | `(u, *, dim, L, order=1) -> torch.Tensor` | Derivada de ordem `order` ao longo de um eixo via FFT (domínio periódico). |
| `fd_derivative` | função | `(u, *, dim, dx, order=1) -> torch.Tensor` | Derivada por diferenças finitas ao longo de um eixo. |
| `compute_derivatives` | função | `(u, *, grid, orders=(1, 2), method='spectral') -> Dict[Tuple[int, int], torch.Tensor]` | Dicionário de derivadas por (eixo, ordem), pelo método espectral ou de diferenças finitas. |

#### `pinneapple_neural.architectures.neural_operators.registry`

Registro da família `neural_operators` com capacidades por chave (tipo de entrada e chaves esperadas); `transolver` fica reservado para o bridge Noether.

<small>`pinneapple_neural/architectures/neural_operators/registry.py` · 123 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NeuralOperatorCatalog` | classe | `(registry)` | Catálogo da família `neural_operators`. |
| `register_into_global` | função | `() -> None` | Registra os operadores no `ModelRegistry` com tipo de entrada e chaves esperadas por modelo. |

#### `pinneapple_neural.architectures.neural_operators.transolver`

Transolver nativo em PyTorch (Physics-Attention por fatias) para nuvens de pontos, e uma versão leve `TransolverLite`. [sem teste direto]

<small>`pinneapple_neural/architectures/neural_operators/transolver.py` · 145 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsAttention` | classe | `(dim, heads=8, slices=32, dropout=0.0)` | Atenção física multi-cabeça: agrupa pontos em fatias, atende entre fatias e devolve aos pontos. |
| `TransolverBlock` | classe | `(dim, heads, slices, mlp_ratio=2, dropout=0.0)` | Bloco Transolver: Physics-Attention mais MLP com normalização. |
| `Transolver` | classe | `(in_dim, out_dim, dim=128, depth=4, heads=8, slices=32, mlp_ratio=2, dropout=0.0)` | Operador em nuvem de pontos (B, N, in_dim) -> (B, N, out_dim). |
| `TransolverLite` | classe | `(in_dim=5, out_dim=101, d_model=64, n_heads=4, n_slices=8, n_blocks=3, dropout=0.1)` | Versão leve: projeção linear, alguns blocos de atenção por fatias e cabeça linear por nó. |

#### `pinneapple_neural.architectures.neural_operators.uno`

U-NO (operador em forma de U) para grades 1D/2D/3D e para pontos irregulares via kNN, unificado em `UniversalUNO`. [sem teste direto]

<small>`pinneapple_neural/architectures/neural_operators/uno.py` · 494 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SpectralConvND` | classe | `(dim, in_channels, out_channels, modes)` | Convolução espectral 1D/2D/3D (rfftn) nos modos de baixa frequência. |
| `OperatorBlockND` | classe | `(dim, width, modes)` | Bloco de operador espectral com caminho local. |
| `DownBlockND` | classe | `(dim, in_ch, out_ch, modes, depth)` | Bloco de descida (redução de resolução) do U-NO em grade. |
| `UpBlockND` | classe | `(dim, in_ch, skip_ch, out_ch, modes, depth)` | Bloco de subida com skip do U-NO em grade. |
| `UNONdGrid` | classe | `(dim, in_channels, out_channels, width=64, levels=3, depth_per_level=2, modes=None, add_c…)` | U-NO para grades regulares (B, Cin, *espaço) -> (B, Cout, *espaço). |
| `PointKernelOperator` | classe | `(dim, in_ch, out_ch, hidden=64, k=16)` | Operador contínuo em pontos irregulares por passagem de mensagens kNN com kernel aprendido. |
| `PointOperatorBlock` | classe | `(dim, width, k=16)` | Bloco residual de operador em pontos. |
| `UNONdPoints` | classe | `(dim, in_channels, out_channels, width=128, levels=3, depth_per_level=2, k=16, pool_ratio…)` | U-NO para pontos irregulares com pirâmide multiescala por subamostragem. |
| `UniversalUNO` | classe | `(dim, in_channels, out_channels, *, mesh_mode='auto', width_grid=64, levels_grid=3, depth…)` | U-NO unificado que escolhe o modo grade ou pontos (`mesh_mode='auto'`). |

#### `pinneapple_neural.architectures.physics_aware`

Família de redes com estrutura física (PANN, SPN). [sem teste direto]

<small>`pinneapple_neural/architectures/physics_aware/__init__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_neural.architectures.physics_aware.base`

Saída `PhysicsAwareOutput` e base `PhysicsAwareBase` com `mse`. [sem teste direto]

<small>`pinneapple_neural/architectures/physics_aware/base.py` · 25 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsAwareOutput` | classe | `(y, losses, extras)` | Saída padrão da família `physics_aware`. |
| `PhysicsAwareBase` | classe | `()` | Base da família com `mse`. |

#### `pinneapple_neural.architectures.physics_aware.pann`

Envolve um backbone e soma uma perda física opcional, com componentes prefixados por `physics/`. [sem teste direto]

<small>`pinneapple_neural/architectures/physics_aware/pann.py` · 135 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsAwareNeuralNetwork` | classe | `(backbone, *, physics_loss_fn=None, physics_weight=1.0, physics_components_prefix='physic…)` | Backbone com função de perda física opcional e peso `physics_weight`. |

#### `pinneapple_neural.architectures.physics_aware.registry`

Registro da família `physics_aware` e `PhysicsAwareCatalog`. [sem teste direto]

<small>`pinneapple_neural/architectures/physics_aware/registry.py` · 47 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsAwareCatalog` | classe | `(registry)` | Catálogo da família `physics_aware`. |
| `register_into_global` | função | `() -> None` | Registra a família `physics_aware` no `ModelRegistry` global. |

#### `pinneapple_neural.architectures.physics_aware.spn`

Rede que aplica um projetor à saída do backbone para preservar uma estrutura (por exemplo, uma restrição). [sem teste direto]

<small>`pinneapple_neural/architectures/physics_aware/spn.py` · 53 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `StructurePreservingNetwork` | classe | `(backbone, *, projector)` | Backbone seguido de um `projector` que impõe uma estrutura à saída. |

#### `pinneapple_neural.architectures.pinns`

Família de PINNs; reexporta Vanilla, Inverse, PIELM, PINN-LSTM, PINNsFormer, VPINN, XPINN, XTFC, PINN bayesiano e difusão informada por física. [sem teste direto do `__init__`]

<small>`pinneapple_neural/architectures/pinns/__init__.py` · 57 linhas · **sem teste direto**</small>

#### `pinneapple_neural.architectures.pinns.base`

Saída `PINNOutput` e base `PINNBase` com gradientes por autograd, derivada normal, `physics_loss` e exportação TorchScript/ONNX.

<small>`pinneapple_neural/architectures/pinns/base.py` · 261 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PINNOutput` | classe | `(y, losses, extras)` | Saída padrão dos PINNs: `y`, perdas e extras. |
| `PINNBase` | classe | `()` | Base dos PINNs com gradiente, derivada normal, `predict`, `physics_loss` e exportação TorchScript/ONNX. |

#### `pinneapple_neural.architectures.pinns.bpinn`

PINN bayesiano por inferência variacional de campo médio (camadas lineares bayesianas, ELBO, previsão por ensemble). [sem teste direto]

<small>`pinneapple_neural/architectures/pinns/bpinn.py` · 389 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BayesianLinear` | classe | `(in_features, out_features, *, prior_std=1.0, init_rho=-3.0, bias=True)` | Camada linear com pesos gaussianos (média e rho), amostragem e termo KL. |
| `BPINNConfig` | classe | `(in_dim, out_dim, hidden, activation, prior_std, init_rho, noise_pde, noise_obs, kl_weigh…)` | Configuração do `BayesianPINN` (arquitetura, prior, ruídos, peso do KL, parâmetros inversos). |
| `BayesianPINN` | classe | `(in_dim=2, out_dim=1, hidden=(64, 64, 64), activation='tanh', *, config=None, prior_std=1…)` | PINN bayesiano por VI de campo médio, com `elbo_loss` e previsão por ensemble (média e desvio). |

#### `pinneapple_neural.architectures.pinns.inverse`

PINN inverso do catálogo, com parâmetros físicos treináveis e gancho de compatibilidade com `PINNFactory`. [sem teste direto]

<small>`pinneapple_neural/architectures/pinns/inverse.py` · 118 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `InversePINN` | classe | `(in_dim, out_dim, hidden=(128, 128, 128, 128), activation='tanh', inverse_params=None, in…)` | PINN com parâmetros físicos desconhecidos treináveis; `as_factory_model` o adapta ao `PINNFactory`. |

#### `pinneapple_neural.architectures.pinns.physics_diffusion`

Modelo de difusão informado por física: escalonadores VP e EDM, denoiser MLP, perda de score matching e guias por resíduo de EDP ou dados. As bases `NoiseSchedulerBase` e `GuidanceBase` levantam `NotImplementedError` (são abstratas). [sem teste direto]

<small>`pinneapple_neural/architectures/pinns/physics_diffusion.py` · 787 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NoiseSchedulerBase` | classe | `()` | Protocolo de escalonador de ruído; os métodos levantam `NotImplementedError`. |
| `VPNoiseScheduler` | classe | `(T=1000, beta_min=0.0001, beta_max=0.02)` | Escalonador VP-DDPM com beta linear. |
| `EDMNoiseScheduler` | classe | `(sigma_min=0.002, sigma_max=80.0, rho=7.0)` | Escalonador EDM (Karras et al.) com sigmas entre `sigma_min` e `sigma_max`. |
| `SinusoidalEmbedding` | classe | `(dim)` | Embedding senoidal do nível de ruído ou passo de tempo. |
| `FiLMLayer` | classe | `(feat_dim, cond_dim)` | Modulação FiLM das ativações condicionada ao nível de ruído. |
| `MLPDenoiser` | classe | `(x_dim, t_emb_dim=64, hidden=(256, 256, 256), activation='gelu', predict_x0=False)` | Denoiser MLP condicionado no tempo para campos 1D. |
| `DSMLoss` | classe | `(denoiser, scheduler, loss_weight_fn=None)` | Perda de denoising score matching. |
| `GuidanceBase` | classe | `()` | Protocolo de guia tipo DPS; o gradiente levanta `NotImplementedError` na base. |
| `PDEResidualGuidance` | classe | `(residual_fn, weight=0.1)` | Guia que empurra a amostra para reduzir o resíduo de uma EDP. |
| `DataConsistencyGuidance` | classe | `(mask, y_obs, weight=0.5)` | Guia de consistência com observações mascaradas (assimilação de dados, problemas inversos). |
| `ComposedGuidance` | classe | `(*guidances)` | Soma de várias guias. |
| `PIDiffConfig` | classe | `(x_dim, t_emb_dim, hidden, activation, scheduler_type, T, beta_min, beta_max, sigma_min, …)` | Configuração do `PhysicsInformedDiffusion`. |
| `PhysicsInformedDiffusion` | classe | `(x_dim=64, hidden=(256, 256, 256), activation='gelu', *, denoiser=None, scheduler=None, c…)` | Modelo de difusão com amostragem, ensemble, perda de treino e previsão guiada por física. |

#### `pinneapple_neural.architectures.pinns.pielm`

Extreme learning machine informada por física: camada oculta aleatória fixa e saída ajustada por ridge. [sem teste direto]

<small>`pinneapple_neural/architectures/pinns/pielm.py` · 141 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PIELM` | classe | `(in_dim, out_dim, hidden_dim=1024, activation='tanh', freeze_random=True)` | ELM informada por física: camada oculta aleatória e `fit_ridge` para a saída. |
| `PIELMFactoryAdapter` | classe | `(pielm, *, inverse_params_names=None, initial_guesses=None, dtype=torch.float32)` | Adapta um `PIELM` à interface do `PINNFactory`, com ajuste ridge a partir das entradas. |

#### `pinneapple_neural.architectures.pinns.pinn_lstm`

Híbrido PINN-LSTM que reorganiza os pontos em sequências e aplica um LSTM. [sem teste direto]

<small>`pinneapple_neural/architectures/pinns/pinn_lstm.py` · 75 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PINNLSTM` | classe | `(in_dim, out_dim, *, seq_len, hidden_dim=128, num_layers=2, dropout=0.0, inverse_params_n…)` | PINN que passa pontos agrupados em sequências de `seq_len` por um LSTM. |

#### `pinneapple_neural.architectures.pinns.pinnsformer`

PINNsFormer: PINN baseado em Transformer que trata pontos como sequência.

<small>`pinneapple_neural/architectures/pinns/pinnsformer.py` · 212 linhas · testes: 1</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PINNsFormer` | classe | `(in_dim, out_dim, *, seq_len=32, d_model=128, nhead=4, num_layers=4, dim_feedforward=256,…)` | PINN Transformer com embedding posicional e máscaras configuráveis. |

#### `pinneapple_neural.architectures.pinns.registry`

Registro da família `pinns` e `PINNCatalog`. [sem teste direto]

<small>`pinneapple_neural/architectures/pinns/registry.py` · 78 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PINNCatalog` | classe | `(registry)` | Catálogo da família `pinns`. |
| `register_into_global` | função | `() -> None` | Registra a família `pinns` no `ModelRegistry` global. |

#### `pinneapple_neural.architectures.pinns.vanilla`

PINN MLP básico (forma forte), com parâmetros inversos opcionais exigidos pela perda do `PINNFactory`.

<small>`pinneapple_neural/architectures/pinns/vanilla.py` · 94 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `VanillaPINN` | classe | `(in_dim, out_dim, hidden=(128, 128, 128, 128), activation='tanh', *, inverse_params_names…)` | PINN MLP com ativação configurável e parâmetros inversos opcionais. |

#### `pinneapple_neural.architectures.pinns.vpinn`

PINN variacional (forma fraca) e um adaptador de perda para o `PINNFactory`. [sem teste direto]

<small>`pinneapple_neural/architectures/pinns/vpinn.py` · 119 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `VPINN` | classe | `(in_dim, out_dim, hidden=(128, 128, 128), activation='tanh', *, inverse_params_names=None…)` | PINN variacional (forma fraca). |
| `pinn_factory_adapter` | função | `(loss_fn)` | Converte uma função de perda em `physics_fn` no formato esperado pelo `PINNFactory`. |

#### `pinneapple_neural.architectures.pinns.xpinn`

XPINN: uma sub-rede por subdomínio, com perdas de interface (valor e fluxo). [sem teste direto]

<small>`pinneapple_neural/architectures/pinns/xpinn.py` · 126 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SubnetWrapper` | classe | `(in_dim, out_dim, hidden=(128, 128, 128, 128), activation='tanh')` | Sub-rede de um subdomínio (um `VanillaPINN` envolvido). |
| `XPINN` | classe | `(n_subdomains, in_dim, out_dim, hidden=(128, 128, 128, 128), activation='tanh', interface…)` | PINN com decomposição de domínio: uma sub-rede por subdomínio e perdas de interface. |

#### `pinneapple_neural.architectures.pinns.xtfc`

X-TFC: features aleatórias com saída ajustada por ridge e expressões restritas (biblioteca `tfc`, opcional) que satisfazem contorno/condição inicial exatamente.

<small>`pinneapple_neural/architectures/pinns/xtfc.py` · 681 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TFCConstrainedExpression` | classe | `(tfc_obj, dim=1)` | Envolve a biblioteca `tfc` para montar expressões restritas 1D ou nD que cumprem exatamente as condições. |
| `XTFCConfig` | classe | `(in_dim, out_dim, rf_dim, activation, freeze_random, rf_kind, rff_sigma, use_bias, init_s…)` | Configuração do `XTFC` (features aleatórias, regularização, uso de TFC). |
| `XTFC` | classe | `(in_dim, out_dim, rf_dim=2048, activation=None, freeze_random=True, g_fn=None, B_fn=None,…)` | PINN X-TFC: features aleatórias congeladas, saída por ridge e forma restrita g + B·N. |
| `XTFCFactoryModel` | classe | `(xtfc, inverse_params_names=None, initial_guesses=None, dtype=torch.float32)` | Wrapper com `forward(x) -> Tensor` compatível com o `PINNFactory` e o `Trainer`. |
| `tfc_available` | função | `() -> bool` | Diz se a biblioteca `tfc` está instalada. |
| `build_xtfc` | função | `(in_dim, out_dim, *, rf_dim=2048, activation=None, freeze_random=True, g_fn=None, B_fn=No…)` | Construtor de conveniência do `XTFC`, com TFC opcional. |

#### `pinneapple_neural.architectures.recurrent`

Família de modelos recorrentes de previsão (GRU, LSTM, Seq2Seq). [sem teste direto]

<small>`pinneapple_neural/architectures/recurrent/__init__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_neural.architectures.recurrent.base`

Saída `RNNOutput` e base `RecurrentModelBase` com `mse`. [sem teste direto]

<small>`pinneapple_neural/architectures/recurrent/base.py` · 36 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RNNOutput` | classe | `(y, losses, extras)` | Saída padrão dos modelos recorrentes. |
| `RecurrentModelBase` | classe | `()` | Base dos modelos recorrentes com `mse`. |

#### `pinneapple_neural.architectures.recurrent.gru`

Previsores GRU e GRU bidirecional com horizonte fixo, pooling e embedding de tempo opcionais. [sem teste direto]

<small>`pinneapple_neural/architectures/recurrent/gru.py` · 199 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GRUModel` | classe | `(in_dim, out_dim, horizon, hidden_dim=128, num_layers=2, dropout=0.0, *, pool='last', tim…)` | Previsor GRU para `horizon` passos. |
| `BiGRUModel` | classe | `(in_dim, out_dim, horizon, hidden_dim=128, num_layers=2, dropout=0.0, *, pool='last', tim…)` | Previsor GRU bidirecional. |

#### `pinneapple_neural.architectures.recurrent.lstm`

Previsores LSTM e LSTM bidirecional com horizonte fixo. [sem teste direto]

<small>`pinneapple_neural/architectures/recurrent/lstm.py` · 201 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LSTMModel` | classe | `(in_dim, out_dim, horizon, hidden_dim=128, num_layers=2, dropout=0.0, *, pool='last', tim…)` | Previsor LSTM para `horizon` passos. |
| `BiLSTMModel` | classe | `(in_dim, out_dim, horizon, hidden_dim=128, num_layers=2, dropout=0.0, *, pool='last', tim…)` | Previsor LSTM bidirecional. |

#### `pinneapple_neural.architectures.recurrent.registry`

Registro da família `recurrent` e `RecurrentCatalog`. [sem teste direto]

<small>`pinneapple_neural/architectures/recurrent/registry.py` · 57 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RecurrentCatalog` | classe | `(registry)` | Catálogo da família `recurrent`. |
| `register_into_global` | função | `() -> None` | Registra a família `recurrent` no `ModelRegistry` global. |

#### `pinneapple_neural.architectures.recurrent.seq2seq`

Previsor Seq2Seq (encoder-decoder GRU ou LSTM) com covariáveis futuras opcionais. [sem teste direto]

<small>`pinneapple_neural/architectures/recurrent/seq2seq.py` · 215 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Seq2SeqRNN` | classe | `(in_dim, out_dim, horizon, hidden_dim=128, num_layers=2, dropout=0.0, cell='gru', future_…)` | Previsor encoder-decoder recorrente. |

#### `pinneapple_neural.architectures.register_all`

Popula o `ModelRegistry` importando o registro de cada família; idempotente, ignora famílias com dependência faltando (log em debug) e só avisa em colisão de nomes. [sem teste direto]

<small>`pinneapple_neural/architectures/register_all.py` · 44 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `register_all` | função | `() -> None` | Chama `register_into_global` de cada família uma única vez. |

#### `pinneapple_neural.architectures.registry`

`ModelSpec` e `ModelRegistry`: registro central por nome, com família, tags, tipo de entrada e chaves esperadas; `build` passa pela normalização de kwargs de `instantiate`.

<small>`pinneapple_neural/architectures/registry.py` · 101 linhas · testes: 64</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ModelSpec` | classe | `(name, family, cls, description, tags, input_kind, supports_physics_loss, expects, predic…)` | Ficha de um modelo registrado: nome, família, classe, descrição, tags, tipo de entrada, suporte a perda física, chaves esperadas e campos previstos. |
| `ModelRegistry` | classe | `(_REGISTRY)` | Registro central por nome: `register` (decorador), `list`, `families`, `spec`, `build` e `build_with_report`. |

#### `pinneapple_neural.architectures.reservoir_computing`

Família de computação de reservatório e modelos de features aleatórias com saída por ridge (ELM, ESN, RBF, Koopman/EDMD). [sem teste direto]

<small>`pinneapple_neural/architectures/reservoir_computing/__init__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_neural.architectures.reservoir_computing.base`

Saída `RCOutput` e base `RCBase` com `mse` e solução ridge. [sem teste direto]

<small>`pinneapple_neural/architectures/reservoir_computing/base.py` · 48 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RCOutput` | classe | `(y, losses, extras)` | Saída padrão dos modelos de reservatório. |
| `RCBase` | classe | `()` | Base com `mse` e `ridge_solve`. |

#### `pinneapple_neural.architectures.reservoir_computing.elm`

Extreme Learning Machine: camada oculta aleatória e saída por ridge em forma fechada. [sem teste direto]

<small>`pinneapple_neural/architectures/reservoir_computing/elm.py` · 88 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ExtremeLearningMachine` | classe | `(in_dim, out_dim, hidden_dim=1024, activation='tanh', l2=1e-06, freeze_random=True)` | ELM com `fit` por ridge e `forward`. |

#### `pinneapple_neural.architectures.reservoir_computing.esn`

Echo State Network com reservatório vazado (leak), saída por ridge e previsão autorregressiva. [sem teste direto]

<small>`pinneapple_neural/architectures/reservoir_computing/esn.py` · 340 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EchoStateNetwork` | classe | `(in_dim, out_dim, reservoir_dim=1024, spectral_radius=0.9, leak=1.0, input_scale=1.0, l2=…)` | ESN com `fit` por ridge e `predict_autoregressive`. |

#### `pinneapple_neural.architectures.reservoir_computing.esn_rc`

Variante de ESN com leitura aumentada por termos polinomiais do estado. [sem teste direto]

<small>`pinneapple_neural/architectures/reservoir_computing/esn_rc.py` · 113 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ESNRC` | classe | `(in_dim, out_dim, reservoir_dim=1024, *, spectral_radius=0.9, leak=1.0, input_scale=1.0, …)` | ESN com leitura polinomial do estado. |

#### `pinneapple_neural.architectures.reservoir_computing.hybrid_rbf`

Rede RBF híbrida: y = [Φ(x), x, 1] W, resolvida num único sistema ridge. [sem teste direto]

<small>`pinneapple_neural/architectures/reservoir_computing/hybrid_rbf.py` · 158 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HybridRBFNetwork` | classe | `(in_dim, out_dim, num_centers=512, sigma=None, l2=1e-06, learn_centers=False, use_linear=…)` | Rede RBF com termos linear e bias, resolvida por ridge. |

#### `pinneapple_neural.architectures.reservoir_computing.koopman`

Operador de Koopman estilo EDMD num espaço elevado (RFF ou polinomial), ajustado por ridge. [sem teste direto]

<small>`pinneapple_neural/architectures/reservoir_computing/koopman.py` · 155 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `KoopmanOperator` | classe | `(in_dim, *, lift_dim=1024, lift='rff', poly_degree=2, rff_lengthscale=1.0, l2=1e-06, use_…)` | Operador de Koopman EDMD: eleva x, ajusta K por ridge e faz rollout. |

#### `pinneapple_neural.architectures.reservoir_computing.rbf`

Rede RBF gaussiana com leitura por ridge e opção de ajuste por gradiente. [sem teste direto]

<small>`pinneapple_neural/architectures/reservoir_computing/rbf.py` · 263 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RBFNetwork` | classe | `(in_dim, out_dim, num_centers=512, sigma=None, l2=1e-06, learn_centers=False, *, normaliz…)` | Rede RBF gaussiana com `fit` (ridge) e `fit_gd` (gradiente). |
| `sigma_is_default` | função | `(log_sigma, eps=1e-06) -> bool` | Diz se a largura `log_sigma` ainda está no valor padrão (sem ajuste). |

#### `pinneapple_neural.architectures.reservoir_computing.registry`

Registro da família `reservoir_computing` e `ReservoirCatalog`. [sem teste direto]

<small>`pinneapple_neural/architectures/reservoir_computing/registry.py` · 70 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ReservoirCatalog` | classe | `(registry)` | Catálogo da família `reservoir_computing`. |
| `register_into_global` | função | `() -> None` | Registra a família `reservoir_computing` no `ModelRegistry` global. |

#### `pinneapple_neural.architectures.rom`

Família de modelos de ordem reduzida (POD, DMD, HAVOK, OpInf, SINDy, Koopman, ROM neural, POD paramétrico) e o estudo `rom_study`.

<small>`pinneapple_neural/architectures/rom/__init__.py` · 39 linhas · testes: 25</small>

#### `pinneapple_neural.architectures.rom.base`

Saída `ROMOutput` e base `ROMBase` com `mse`. [sem teste direto]

<small>`pinneapple_neural/architectures/rom/base.py` · 31 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ROMOutput` | classe | `(y, losses, extras)` | Saída padrão dos ROMs. |
| `ROMBase` | classe | `()` | Base dos ROMs com `mse`. |

#### `pinneapple_neural.architectures.rom.contracts`

Converte resultados do estudo de ROM nos contratos abertos de `pinneapple_data.contracts` (`ModelRef`, `StudyResult`), para outros apps usarem o ROM. [sem teste direto]

<small>`pinneapple_neural/architectures/rom/contracts.py` · 120 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `model_ref` | função | `(rm, *, name, uri, payload=None, row=None, split='test', dataset_hash='', dataset_name=''…)` | Monta o `ModelRef` de um `ReducedModel`: entradas com faixa de treino, saídas, métricas e SHA-256 do arquivo. |
| `rom_checks` | função | `(res, tolerance) -> list` | Lista os checks do estudo com valores: melhor ROM contra o baseline, contra a tolerância e divergência dos passos no tempo. |
| `study_result` | função | `(res, *, study_id, title, tolerance, models=(), artifacts=(), limits=(), app='rom', datas…)` | Monta o `StudyResult` do estudo: uma linha por (método, posto), o melhor modelo e os checks. |

#### `pinneapple_neural.architectures.rom.deep_uq_rom`

[placeholder] Esqueleto ('MVP scaffold' no docstring) de preditor latente que devolve média e log-variância. [sem teste direto]

<small>`pinneapple_neural/architectures/rom/deep_uq_rom.py` · 61 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DeepUQROM` | classe | `(r, hidden=256, layers=2, dropout=0.0)` | Preditor latente pequeno que devolve média e log-variância (esqueleto). |

#### `pinneapple_neural.architectures.rom.dmd`

Decomposição em modos dinâmicos truncada (em lote, estabilizada, com análise espectral) e DMD com controle (DMDc, NumPy). [sem teste direto]

<small>`pinneapple_neural/architectures/rom/dmd.py` · 346 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DynamicModeDecomposition` | classe | `(r=64, center='global', l2=0.0)` | DMD truncada de posto r com autovalores, frequências e rollout. |
| `DMDc` | classe | `(affine=False)` | DMD com controle (Proctor, Brunton & Kutz 2016) em NumPy. |

#### `pinneapple_neural.architectures.rom.havok`

HAVOK: embedding de Hankel, SVD e dinâmica linear por DMD no subespaço. [sem teste direto]

<small>`pinneapple_neural/architectures/rom/havok.py` · 225 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HAVOK` | classe | `(delays=50, r=64, center=True, svd_eps=1e-12, decode_mode='last')` | Modelo HAVOK com `fit` e `forward` (rollout no subespaço). |

#### `pinneapple_neural.architectures.rom.koopman`

Operador de Koopman treinável (completo ou em blocos) e autoencoder de Koopman profundo. [sem teste direto]

<small>`pinneapple_neural/architectures/rom/koopman.py` · 269 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `KoopmanOperator` | classe | `(k_dim, structure='full', num_blocks=4)` | Matriz de Koopman treinável, completa ou bloco-diagonal. |
| `KoopmanAutoencoder` | classe | `(in_dim, k_dim=128, hidden=128, n_layers=3, activation='tanh', k_structure='full', num_bl…)` | Autoencoder de Koopman profundo com perdas de reconstrução, previsão e linearidade. |

#### `pinneapple_neural.architectures.rom.neural_rom`

ROM totalmente neural: encoder não linear, dinâmica latente (MLP) e decoder. [sem teste direto]

<small>`pinneapple_neural/architectures/rom/neural_rom.py` · 237 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NeuralROM` | classe | `(in_dim, r=32, hidden_enc=128, n_layers_enc=3, dynamics='mlp', hidden_dyn=128, n_layers_d…)` | ROM neural com encoder, dinâmica latente, decoder e rollout. |

#### `pinneapple_neural.architectures.rom.opinf`

Operator Inference para dinâmica latente (termos linear, quadrático e bias, com regularização), em tempo discreto ou contínuo. [sem teste direto]

<small>`pinneapple_neural/architectures/rom/opinf.py` · 387 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OperatorInference` | classe | `(r, *, use_quadratic=True, use_bias=True, l2_linear=1e-06, l2_quad=1e-06, l2_bias=0.0, ce…)` | Operator Inference com operadores A, H e b, ajuste discreto ou contínuo e rollout. |

#### `pinneapple_neural.architectures.rom.parametric`

ROM paramétrico: parâmetros de projeto -> campo completo sem rodar o solver (base POD + regressão dos coeficientes) e amostragem por hipercubo latino. [sem teste direto]

<small>`pinneapple_neural/architectures/rom/parametric.py` · 198 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ParametricPOD` | classe | `(r, energy, regressor, kernel, smoothing, normalize, seed, mean_, basis_, sv_)` | Base POD com regressão dos coeficientes nos parâmetros (RBF ou GP), com predição, erro de projeção e serialização em arrays. |
| `latin_hypercube` | função | `(n, bounds, seed=0) -> tuple[np.ndarray, list[str]]` | `n` pontos de hipercubo latino numa caixa de parâmetros; devolve pontos e nomes. |

#### `pinneapple_neural.architectures.rom.pod`

POD por SVD completa ou randomizada, com ajuste incremental, para compressão. [sem teste direto]

<small>`pinneapple_neural/architectures/rom/pod.py` · 252 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `POD` | classe | `(r=64, center=True, energy=None, randomised=False, random_seed=0)` | POD por SVD completa ou randomizada, com `encode`, `decode` e erro de reconstrução. |

#### `pinneapple_neural.architectures.rom.registry`

Registro da família `rom` e `ROMCatalog`. [sem teste direto]

<small>`pinneapple_neural/architectures/rom/registry.py` · 82 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ROMCatalog` | classe | `(registry)` | Catálogo da família `rom`. |
| `register_into_global` | função | `() -> None` | Registra a família `rom` no `ModelRegistry` global. |

#### `pinneapple_neural.architectures.rom.rom_hybrid`

ROM híbrido: POD, Operator Inference no latente e corretor neural do erro de truncamento. [sem teste direto]

<small>`pinneapple_neural/architectures/rom/rom_hybrid.py` · 177 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ROMHybrid` | classe | `(field_dim, r=64, l2=1e-06, use_corrector=True, corrector_hidden=64)` | ROM híbrido POD + OpInf + corretor MLP. |

#### `pinneapple_neural.architectures.rom.sindy`

SINDy para dinâmica latente: biblioteca de funções e regressão esparsa (STLSQ). [sem teste direto]

<small>`pinneapple_neural/architectures/rom/sindy.py` · 323 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SINDy` | classe | `(r, *, library='poly', poly_degree=2, threshold=0.05, n_iter=10, l2=0.0001, solver='stlsq…)` | SINDy no latente com biblioteca polinomial, limiar de esparsidade e integrador RK4. |

#### `pinneapple_neural.architectures.rom.steppers`

Funções de passo `step(u) -> u_next` de ROMs lineares (POD+DMD, POD+OpInf) no formato que `pp.validate` audita em rollout. [sem teste direto]

<small>`pinneapple_neural/architectures/rom/steppers.py` · 48 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `pod_dmd_stepper` | função | `(snapshots, r=8)` | Monta o passo POD + DMD exata e devolve também o maior módulo de autovalor (acima de 1, o ROM amplifica). |
| `pod_opinf_stepper` | função | `(train, energy) -> tuple[Callable[[np.ndarray], np.ndarray], int]` | Monta o passo POD + OpInf linear e devolve o posto escolhido pela energia. |

#### `pinneapple_neural.architectures.rom.study`

Estudo de ROM: ajusta vários métodos e postos em snapshots de treino e mede erro, tempo e ganho de velocidade em snapshots separados. Exportação ONNX levanta `NotImplementedError` para métodos e opções sem suporte.

<small>`pinneapple_neural/architectures/rom/study.py` · 787 linhas · testes: 11 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Split` | classe | `(train, test, how)` | Índices de treino e teste e o modo de separação. |
| `ReducedModel` | classe | `(method, mode, rank, arrays, config, fit_s, _cache)` | ROM ajustado com `predict`, manifesto, forma portátil `.npz` e exportação ONNX quando suportada. |
| `ROMStudyResult` | classe | `(mode, split, singular_values, rows, projection, skipped, budget_s, hf_seconds, metric, m…)` | Resultado do estudo: linhas por método e posto, valores singulares, erro de projeção, melhor modelo e frente de Pareto. |
| `default_ranks` | função | `(n_train, n_values, sv=None, max_rank=64) -> list[int]` | Postos padrão: potências de 2 até o posto dos dados, mais os postos que guardam 99 %, 99,9 % e 99,99 % da energia. |
| `split_snapshots` | função | `(snaps, *, how='auto', test_fraction=0.2, seed=0, test_idx=None) -> Split` | Separa snapshots de teste (previsão, intercalado ou aleatório para dados paramétricos). |
| `field_scales` | função | `(snaps, train) -> np.ndarray` | Uma escala por valor: o RMS de cada campo nos snapshots de treino. |
| `fit_reduced_model` | função | `(snaps, train, method, rank=None, *, budget_s=5.0, seed=0, mode=None, scale='auto', log_p…)` | Ajusta um ROM de um método e posto nos snapshots de treino. |
| `rom_study` | função | `(snaps, *, methods=None, ranks=None, split='auto', test_fraction=0.2, test_idx=None, budg…)` | Ajusta todos os métodos em todos os postos e pontua cada um nos snapshots separados. |

#### `pinneapple_neural.architectures.rom.toy_systems`

Sistemas lineares sintéticos para demos e testes de ROM. [sem teste direto]

<small>`pinneapple_neural/architectures/rom/toy_systems.py` · 27 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `make_linear_system` | função | `(D, seed=0) -> torch.Tensor` | Operador linear aleatório contrativo (maior valor singular 0,95). |
| `generate_sequences` | função | `(B, T, D, A, noise=0.01, seed=123) -> torch.Tensor` | Gera trajetórias ruidosas de x_{k+1} = A x_k, determinísticas para uma `seed`. |

#### `pinneapple_neural.architectures.rom.trusted`

Surrogate latente que diz quando não confiar: limite de erro conformal, teste de envelope e resíduo físico por consulta, convertidos num `TrustReport`.

<small>`pinneapple_neural/architectures/rom/trusted.py` · 192 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TrustedLatentSurrogate` | classe | `(rom, spread, members, subsample, neighbours, alpha, alpha_envelope, alpha_residual, resi…)` | POD latente com limite de erro, p-valor de envelope e resíduo físico por consulta; `trust_report` gera o relatório de confiança. |

#### `pinneapple_neural.architectures.rom.views`

Visão de um snapshot separado na malha: campo de referência, previsão, diferença, regiões de contorno e carga, pontos de maior erro e um texto curto.

<small>`pinneapple_neural/architectures/rom/views.py` · 122 linhas · testes: 11</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `zones` | função | `(snaps, points, grow=0.1) -> dict[str, np.ndarray]` | Regiões de comparação do erro: conjuntos nomeados de nós crescidos (zona de engaste, de carga) e o resto do corpo. |
| `reconstruction_view` | função | `(snaps, model, index, *, reported=None, grow=0.1, max_faces=None, n_hotspots=3, full_labe…)` | Visão de um snapshot contra a previsão de um `ReducedModel`. |
| `prediction_view` | função | `(snaps, index, pred, *, label='Prediction', note='', reported=None, grow=0.1, max_faces=N…)` | Visão de um snapshot contra uma previsão qualquer, com marcadores, regiões e pontos de maior erro. |

#### `pinneapple_neural.architectures.siren`

SIREN (Sitzmann et al. 2020): MLP com ativação seno e inicialização própria, para campos de alta frequência. [sem teste direto]

<small>`pinneapple_neural/architectures/siren.py` · 135 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SineLayer` | classe | `(in_features, out_features, bias=True, is_first=False, omega_0=30.0)` | Camada senoidal com inicialização de pesos de Sitzmann et al. |
| `SIREN` | classe | `(in_dim, out_dim, hidden_dim=256, n_layers=5, omega_0=30.0, outermost_linear=True, final_…)` | Rede SIREN com `omega_0` configurável e última camada linear por padrão. |

#### `pinneapple_neural.architectures.transformers`

Família de Transformers para séries temporais (Autoformer, FEDformer, Informer, TFT, TimesNet, Transformer). [sem teste direto]

<small>`pinneapple_neural/architectures/transformers/__init__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_neural.architectures.transformers.autoformer`

Autoformer mínimo: decomposição tendência/sazonal e auto-correlação via FFT. [sem teste direto]

<small>`pinneapple_neural/architectures/transformers/autoformer.py` · 367 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SeriesDecomp` | classe | `(kernel_size=25)` | Decomposição por média móvel: sazonal + tendência. |
| `AutoCorrelation` | classe | `(topk_factor=1.0, dropout=0.0)` | Auto-correlação via FFT com agregação por atraso top-k. |
| `AutoCorrelationLayer` | classe | `(d_model, nhead, topk_factor=1.0, dropout=0.0)` | Camada multi-cabeça de auto-correlação com projeções. |
| `EncoderLayer` | classe | `(d_model, nhead, d_ff, kernel_size, dropout, topk_factor)` | Camada de encoder do Autoformer. |
| `DecoderLayer` | classe | `(d_model, nhead, d_ff, kernel_size, dropout, topk_factor)` | Camada de decoder do Autoformer que acumula a tendência. |
| `Autoformer` | classe | `(in_dim, out_dim, horizon, d_model=128, nhead=4, num_layers=2, kernel_size=25, dropout=0.…)` | Modelo Autoformer completo para previsão de `horizon` passos. |

#### `pinneapple_neural.architectures.transformers.base`

Saída `TSOutput` e base `TimeSeriesModelBase`, cujo `forward` levanta `NotImplementedError` (abstrata). [sem teste direto]

<small>`pinneapple_neural/architectures/transformers/base.py` · 48 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TSOutput` | classe | `(y, losses, extras)` | Saída padrão dos Transformers de série temporal. |
| `TimeSeriesModelBase` | classe | `()` | Base abstrata de previsão de sequências com `mse`. |

#### `pinneapple_neural.architectures.transformers.fedformer`

Modelo rápido inspirado no FEDformer: seleção de frequências top-k e MLP. Não é o FEDformer completo. [sem teste direto]

<small>`pinneapple_neural/architectures/transformers/fedformer.py` · 89 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FEDformer` | classe | `(in_dim, out_dim, horizon, d_model=128, topk_freq=32)` | Modelo rápido inspirado no FEDformer (frequências top-k e MLP). |

#### `pinneapple_neural.architectures.transformers.informer`

Encoder Transformer com destilação opcional, inspirado no Informer (usa atenção padrão, não ProbSparse). [sem teste direto]

<small>`pinneapple_neural/architectures/transformers/informer.py` · 132 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SinusoidalPositionalEncoding` | classe | `(d_model, max_len=10000)` | Codificação posicional senoidal. |
| `Informer` | classe | `(in_dim, out_dim, horizon, d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropo…)` | Encoder Transformer com destilação opcional e cabeça de previsão. |

#### `pinneapple_neural.architectures.transformers.registry`

Registro da família `transformers` e `TransformerCatalog`. [sem teste direto]

<small>`pinneapple_neural/architectures/transformers/registry.py` · 61 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TransformerCatalog` | classe | `(registry)` | Catálogo da família `transformers`. |
| `register_into_global` | função | `() -> None` | Registra a família `transformers` no `ModelRegistry` global. |

#### `pinneapple_neural.architectures.transformers.tft`

Versão leve do Temporal Fusion Transformer: seleção de variáveis por grupo, encoder causal, decoder com consultas futuras e portas GRN/GLU. [sem teste direto]

<small>`pinneapple_neural/architectures/transformers/tft.py` · 254 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GRN` | classe | `(d, hidden, dropout=0.0)` | Gated Residual Network simplificada. |
| `GLU` | classe | `(d)` | Gated Linear Unit para portas de skip. |
| `GroupVSN` | classe | `(d_model, num_groups, hidden, dropout=0.0)` | Rede de seleção de variáveis por grupo. |
| `TemporalFusionTransformer` | classe | `(in_dim, out_dim, horizon, d_model=128, nhead=4, num_enc_layers=2, num_dec_layers=2, drop…)` | TFT simplificado para previsão multi-passo. |

#### `pinneapple_neural.architectures.transformers.timesnet`

Previsor no estilo TimesNet: períodos por FFT, mapas 2D e blocos inception. [sem teste direto]

<small>`pinneapple_neural/architectures/transformers/timesnet.py` · 199 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TimesNet` | classe | `(in_dim, out_dim, horizon, d_model=128, num_blocks=2, top_k_periods=3, head_window=64, dr…)` | Previsor no estilo TimesNet. |

#### `pinneapple_neural.architectures.transformers.transformer`

Encoder Transformer causal para previsão autorregressiva. [sem teste direto]

<small>`pinneapple_neural/architectures/transformers/transformer.py` · 113 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `VanillaTransformer` | classe | `(in_dim, out_dim, d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1, p…)` | Encoder Transformer causal com codificação posicional aprendida ou senoidal. |

#### `pinneapple_neural.callbacks`

Roda os checks de `pp.validate` durante o treino em PyTorch Lightning, PhysicsNeMo e Hugging Face: registra métricas a cada N passos, avisa (ou para) quando uma métrica física piora enquanto a perda cai e grava relatório e trust card no fim.

<small>`pinneapple_neural/callbacks.py` · 274 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsWarning` | classe | `()` | Aviso: uma métrica física piorou em vários checks seguidos enquanto a perda de treino melhorava. |
| `PhysicsMonitor` | classe | `(problem, data=None, *, reference=None, query=None, level='quick', every=100, patience=3,…)` | Núcleo sem framework: roda os checks de `pp.validate` a cada `every` passos, acompanha as métricas e avisa ou para após `patience` pioras. |
| `PhysicsNeMoHook` | classe | `(monitor)` | Gancho para um laço de treino PhysicsNeMo, chamado após cada passo do otimizador. |

#### `pinneapple_neural.lpm`

Blocos de um 'Large Physics Model': um modelo para um espaço de projeto inteiro, com codificação de Fourier, vizinhança multiescala, código de geometria (DeepSets), cabeças em ensemble para incerteza e sinal fora-da-distribuição.

<small>`pinneapple_neural/lpm.py` · 212 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FourierEncoding` | classe | `(n_freq=6, include_input=True)` | Codificação posicional [x, sin(2^k π x), cos(2^k π x)] por coordenada. |
| `GeometryEncoder` | classe | `(d, width=64, code=32)` | Encoder DeepSets: MLP por ponto em (posição, normal) e pooling média + máximo para um código de geometria. |
| `Design` | classe | `(points, normals, query, params, target)` | Uma simulação do espaço de projeto: geometria, parâmetros de operação, pontos de consulta e campo-alvo; `ellipse` e `square` criam exemplos. |
| `LargePhysicsModel` | classe | `(dim, n_fields, n_params=0, radii=(0.1, 0.4, 1.2), n_freq=6, width=128, code=32, n_heads=…)` | Modelo que combina codificação local, código de geometria e parâmetros num tronco com várias cabeças; `predict` dá média e dispersão e `ood` mede distância aos projetos de treino. |
| `multiscale_neighbourhood` | função | `(query, points, normals, radii)` | Para cada ponto de consulta e raio: deslocamento médio aos vizinhos, normal média e ocupação. |
| `fit` | função | `(model, designs, *, epochs=2000, lr=0.002, freeze_geometry=False, seed=0) -> list[float]` | Treina ou faz fine-tuning (com `freeze_geometry=True`) numa lista de projetos, cada cabeça num reamostra bootstrap. |

#### `pinneapple_neural.physicsnemo_bridge`

Integração PhysicsNeMo <-> PINNeAPPle, promovida de `examples/vs_physicsnemo/` e `examples/pinneapple_and_physicsnemo/`: adaptadores (UQ, gêmeo digital), laço de active learning + retreino e operadores de campo em grade.

<small>`pinneapple_neural/physicsnemo_bridge/__init__.py` · 86 linhas · testes: 19</small>

#### `pinneapple_neural.physicsnemo_bridge.active_learning_bridge`

Laço 'active learning seleciona -> retreina -> seleciona' (padrão 2 do README de exemplos), com a seleção delegada a `ResidualBasedAL` de `pinneapple_data`.

<small>`pinneapple_neural/physicsnemo_bridge/active_learning_bridge.py` · 226 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RetrainConfig` | classe | `(n_rounds, n_epochs_per_round, batch_size, lr, selection_mode, device, verbose)` | Configuração da parte de retreino do laço (rodadas, épocas por rodada, batch, taxa, modo de seleção). |
| `ALRoundResult` | classe | `(round_index, n_points_added, n_points_total, mean_train_loss, train_losses)` | Registro de uma rodada: pontos adicionados, total e perdas de treino. |
| `ALRetrainResult` | classe | `(rounds, x_collocation, final_loss)` | Resultado do laço completo: rodadas, pontos de colocação finais e perda final. |
| `active_learning_retrain_loop` | função | `(model, loss_fn, residual_fn, bounds, *, x_initial=None, al_config=None, retrain_config=N…)` | Alterna retreino do modelo nos pontos atuais e seleção de novos pontos por resíduo. |

#### `pinneapple_neural.physicsnemo_bridge.adapters`

Adaptadores para usar um modelo treinado no PhysicsNeMo em PINNeAPPle: UQ por MC Dropout e `DigitalTwin`; o import de `physicsnemo` é opcional e feito só onde é preciso. [sem teste direto]

<small>`pinneapple_neural/physicsnemo_bridge/adapters.py` · 254 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsNemoModelAdapter` | classe | `(model, field_names) -> None` | Envolve um modelo para que `forward` devolva um único tensor (junta saídas em dicionário). |
| `is_physicsnemo_available` | função | `() -> bool` | Diz se `physicsnemo` (ou o nome antigo `modulus`) pode ser importado. |
| `require_physicsnemo_model` | função | `(model) -> Any` | Confere que o modelo é compatível com PhysicsNeMo; exige o pacote instalado. |
| `wrap_for_uq` | função | `(model, field_names=None, *, mc_dropout_config=None) -> MCDropoutWrapper` | Envolve um modelo treinado num `MCDropoutWrapper` para quantificar incerteza. |
| `build_digital_twin_for_model` | função | `(model, field_names, coord_names=None, *, config=None) -> DigitalTwin` | Envolve um modelo treinado num `DigitalTwin`. |

#### `pinneapple_neural.physicsnemo_bridge.field_operators`

Constrói FNO e MeshGraphNet do próprio PhysicsNeMo numa grade n x n, treina sob um `ComputeBudget` e expõe como preditor NumPy `predict(a) -> u` para os checks de PINNeAPPle.

<small>`pinneapple_neural/physicsnemo_bridge/field_operators.py` · 241 linhas · testes: 3</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GridFieldPredictor` | classe | `(module, config, name)` | Modelo PhysicsNeMo treinado numa grade n x n exposto como preditor NumPy, com normalização, salvar e carregar. |
| `TrainingResult` | classe | `(run, epochs, final_loss, history)` | O que `train_grid_operator` gastou e alcançou (épocas, perda final, histórico). |
| `grid_graph` | função | `(n) -> tuple[np.ndarray, np.ndarray]` | Grafo de uma grade n x n centrada nas células: `edge_index` bidirecional e `edge_attr` (dx, dy, \|d\|). |
| `physicsnemo_fno` | função | `(n, *, latent_channels=32, num_fno_layers=4, num_fno_modes=12, decoder_layers=1, decoder_…)` | Cria um FNO 2D do PhysicsNeMo (coeficiente -> solução) como `GridFieldPredictor`. |
| `physicsnemo_meshgraphnet` | função | `(n, *, processor_size=15, hidden_dim=64, aggregation='sum', seed=0) -> GridFieldPredictor` | Cria um MeshGraphNet do PhysicsNeMo sobre o grafo da grade. |
| `train_grid_operator` | função | `(pred, a, u, budget, *, batch_size=8, lr=0.001, weight_decay=1e-05, seed=0, evaluate=None…)` | Treina o preditor em pares (a, u) até gastar o `ComputeBudget`, com perda L2 relativa por amostra. |

#### `pinneapple_neural.predictor`

Inferência pós-treino (antigo `pinneapple_inference`): reexporta inferência em lote e em grade, gráficos de campos, erro, perda, comparação, fluxo 3D e otimização de projeto. [sem teste direto]

<small>`pinneapple_neural/predictor/__init__.py` · 69 linhas · **sem teste direto**</small>

#### `pinneapple_neural.predictor.infer`

Inferência em lote para evitar falta de memória, inferência com NumPy e avaliação em grades 1D+tempo e 2D, com o contêiner `InferenceResult`. [sem teste direto]

<small>`pinneapple_neural/predictor/infer.py` · 190 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `InferenceResult` | classe | `(coords, fields, grid_shape, model_id, metadata)` | Contêiner da inferência: coordenadas, campos, forma da grade, id do modelo e metadados. |
| `batched_inference` | função | `(model, x, *, batch_size=4096, device=None, use_amp=False, amp_dtype=torch.float16) -> to…)` | Roda o modelo em lotes sobre um tensor grande, com AMP opcional. |
| `infer` | função | `(model, x, *, device='cpu', field_names=None) -> np.ndarray` | Passo direto com entrada NumPy e saída NumPy. |
| `infer_on_grid_1d` | função | `(model, x_range, t_range, *, nx=200, nt=200, device='cpu', field_names=None, coord_names=…)` | Avalia o modelo numa grade retangular x-t. |
| `infer_on_grid_2d` | função | `(model, x_range, y_range, *, nx=100, ny=100, device='cpu', field_names=None, coord_names=…)` | Avalia o modelo numa grade retangular x-y. |

#### `pinneapple_neural.predictor.postprocess`

Visualização 3D de campos: linhas de corrente por RK4, isosuperfícies (marching cubes, exige `scikit-image`), fatias de volume e a classe `FlowVisualizer`. [sem teste direto]

<small>`pinneapple_neural/predictor/postprocess.py` · 762 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FlowVisualizer` | classe | `(model, coord_ranges, device='cpu') -> None` | Classe de alto nível com linhas de corrente, contornos de pressão, módulo de velocidade e vorticidade a partir de um modelo. |
| `compute_streamlines` | função | `(u_field, v_field, x_grid, y_grid, seed_points=None, n_seeds=20, max_length=1.0, dt=0.01)…)` | Linhas de corrente 2D por integração RK4. |
| `compute_isosurface` | função | `(field_3d, x, y, z, level=0.0) -> Tuple[np.ndarray, np.ndarray]` | Isosuperfície de um campo 3D por marching cubes (exige `scikit-image`). |
| `plot_streamlines_2d_from_model` | função | `(model, x_range=(0.0, 1.0), y_range=(0.0, 1.0), n_grid=50, n_seeds=30, figsize=(10, 6), t…)` | Desenha linhas de corrente do campo de velocidade previsto por um modelo. |
| `plot_isosurface_3d` | função | `(model, x_range=(0.0, 1.0), y_range=(0.0, 1.0), z_range=(0.0, 1.0), n_grid=30, level=0.0,…)` | Desenha a isosuperfície 3D de um campo escalar previsto. |
| `plot_volume_slice` | função | `(model, axis='z', slice_val=0.5, x_range=(0.0, 1.0), y_range=(0.0, 1.0), z_range=(0.0, 1.…)` | Desenha uma fatia 2D de um campo 3D previsto. |

#### `pinneapple_neural.predictor.visualize`

Gráficos com matplotlib dos resultados de inferência: campos 1D/2D, mapas de erro, curvas de perda, comparação de modelos, fatias de velocidade e vorticidade, perfil de linha central, convergência e frente de Pareto. [sem teste direto]

<small>`pinneapple_neural/predictor/visualize.py` · 1.103 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_field_1d` | função | `(result, field_name, *, ax=None, title=None, cmap='RdBu_r', save_path=None, show=False) -…)` | Mapa de cores x vs t de um campo 1D+tempo. |
| `plot_field_2d` | função | `(result, field_name, *, ax=None, title=None, cmap='viridis', save_path=None, show=False) …)` | Mapa de cores x vs y de um campo 2D. |
| `plot_error_map_1d` | função | `(result, reference, field_name, *, ax=None, title=None, save_path=None, show=False) -> Any` | Erro absoluto contra uma referência numa grade x-t. |
| `plot_error_map_2d` | função | `(result, reference, field_name, *, ax=None, title=None, save_path=None, show=False) -> Any` | Erro absoluto contra uma referência numa grade 2D. |
| `plot_loss_curve` | função | `(history, *, keys=None, ax=None, title='Training loss', log_scale=True, save_path=None, s…)` | Curvas de perda de treino e validação a partir do histórico. |
| `plot_model_comparison_1d` | função | `(results, field_name, *, reference=None, title=None, cmap='RdBu_r', save_path=None, show=…)` | Comparação lado a lado de vários modelos num campo 1D+tempo. |
| `plot_model_comparison_2d` | função | `(results, field_name, *, reference=None, title=None, cmap='viridis', save_path=None, show…)` | Comparação lado a lado de vários modelos num campo 2D. |
| `render_visualizations` | função | `(viz_cfgs, *, model_results, problem_spec=None, reference_result=None, out_dir='.') -> Di…)` | Gera os gráficos pedidos numa lista de configuração (YAML) e devolve os caminhos dos arquivos. |
| `plot_velocity_slice` | função | `(u, x, y, z, *, slice_axis='z', slice_idx=None, component=0, title=None, cmap='RdBu_r', a…)` | Fatia 2D de um componente da velocidade num campo 3D. |
| `plot_velocity_magnitude_slice` | função | `(u, x, y, z, *, slice_axis='z', slice_idx=None, title=None, cmap='viridis', ax=None, save…)` | Fatia 2D do módulo da velocidade. |
| `plot_streamlines_2d` | função | `(u, v, x, y, *, density=1.5, title=None, background_field=None, cmap='viridis', ax=None, …)` | Linhas de corrente de um campo de velocidade 2D com fundo opcional. |
| `plot_centerline_velocity` | função | `(u_centerline, coord, *, reference_coord=None, reference_u=None, reference_label='Ghia et…)` | Perfil de velocidade na linha central com referência da literatura opcional. |
| `plot_vorticity_slice` | função | `(u, v, x, y, z, *, slice_axis='z', slice_idx=None, title=None, cmap='RdBu_r', ax=None, sa…)` | Vorticidade ω_z = dv/dx - du/dy numa fatia 2D. |
| `plot_internal_flow_summary` | função | `(solver_output, *, time_step=-1, slice_axis='z', title='Internal Flow Summary', save_path…)` | Figura de vários painéis para escoamento interno 3D (cavidade, canal). |
| `plot_design_opt_convergence` | função | `(history_objectives, *, title=…, xlabel='Iteration', ylabel='Objective', log_scale=False,…)` | Curva de convergência do objetivo ao longo das iterações de otimização. |
| `plot_pareto_front_2d` | função | `(objectives, pareto_mask, *, xlabel='Objective 1', ylabel='Objective 2', title='Pareto Fr…)` | Dispersão com os pontos Pareto-ótimos destacados. |

#### `pinneapple_neural.trainer`

Fachada do subpacote: reexporta treinadores (`Trainer`, `TwoPhaseTrainer`, `TimeMarchingTrainer`, `CausalPINNTrainer`, `DDPPINNTrainer`), balanceadores de perda, otimizadores e utilitários de HPC num só lugar.

<small>`pinneapple_neural/trainer/__init__.py` · 191 linhas · testes: 1</small>

#### `pinneapple_neural.trainer.adaptive_sweep`

Busca adaptativa de hiperparâmetros: usa TPE do Optuna (com poda opcional) quando o pacote está instalado e cai para busca aleatória sem ele. [sem teste]

<small>`pinneapple_neural/trainer/adaptive_sweep.py` · 178 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AdaptiveSweepConfig` | classe | `(param_space, n_trials, direction, seed, pruning)` | Configuração da busca adaptativa: espaço de parâmetros, número de tentativas, direção (minimizar/maximizar), semente e poda. |
| `run_adaptive_sweep` | função | `(trial_fn, cfg) -> List[Dict[str, Any]]` | Executa a busca adaptativa (TPE do Optuna ou busca aleatória como fallback) e devolve a lista de resultados por tentativa. |

#### `pinneapple_neural.trainer.advisor`

Diagnóstico de treino: lê históricos de perda e métricas finais, detecta sinais (platô, divergência, oscilação, desequilíbrio BC/física etc.) e devolve sugestões ordenadas com patches de configuração. [sem teste]

<small>`pinneapple_neural/trainer/advisor.py` · 378 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Suggestion` | classe | `(code, priority, description, config_patch, rationale)` | Uma recomendação acionável: código, prioridade, descrição, patch de configuração e justificativa. |
| `DiagnosticReport` | classe | `(model_name, signals, suggestions, metrics_summary, diagnosis_text)` | Saída completa do diagnóstico de uma execução: sinais detectados, sugestões, resumo de métricas e texto de diagnóstico. |
| `TrainingAdvisor` | classe | `(plateau_rel_std_thresh=0.005, divergence_rel_thresh=-0.05, slow_start_thresh=0.05, high_…)` | Analisa histórico de perda e métricas finais com limiares configuráveis e produz sugestões ordenadas; `apply_suggestions` aplica os patches a uma configuração. |

#### `pinneapple_neural.trainer.audit`

Auditabilidade: semente global, modo determinístico, impressão digital do ambiente e um logger JSONL por execução. [sem teste]

<small>`pinneapple_neural/trainer/audit.py` · 76 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RunLogger` | classe | `(out_dir, run_name, flush_every)` | Logger JSONL por execução que também salva a configuração e artefatos no diretório da execução. |
| `set_seed` | função | `(seed) -> None` | Fixa a semente de `random`, NumPy e PyTorch (incluindo CUDA). |
| `set_deterministic` | função | `(deterministic=True) -> None` | Liga ou desliga o modo determinístico do PyTorch/cuDNN. |
| `env_fingerprint` | função | `() -> Dict[str, Any]` | Retorna um dict com versões de Python, plataforma, PyTorch e disponibilidade/versão de CUDA para registrar o ambiente. |

#### `pinneapple_neural.trainer.callbacks`

Callbacks simples de treino: parada antecipada (`EarlyStopping`) e salvamento do melhor modelo (`ModelCheckpoint`). [sem teste]

<small>`pinneapple_neural/trainer/callbacks.py` · 67 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EarlyStopping` | classe | `(monitor, patience, min_delta, mode, best, bad_epochs, stop)` | Para o treino quando a métrica monitorada não melhora por `patience` épocas, com `min_delta` e modo min/max. |
| `ModelCheckpoint` | classe | `(path, monitor, mode, best)` | Salva o `state_dict` do modelo (com extras e melhor valor) no caminho dado sempre que a métrica monitorada melhora. |

#### `pinneapple_neural.trainer.causal`

Treino causal para EDPs temporais (Wang et al. 2022): pesos exponencialmente decrescentes por fatia de tempo, conforme o resíduo acumulado nas fatias anteriores. [sem teste]

<small>`pinneapple_neural/trainer/causal.py` · 408 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CausalWeightScheduler` | classe | `(n_time_chunks=100, epsilon=1.0, update_every=100)` | Calcula pesos causais por fatia de tempo, w_k = exp(-epsilon * soma das perdas das fatias anteriores), e a perda ponderada e uma métrica de violação de causalidade. |
| `CausalPINNTrainer` | classe | `(model, epsilon=1.0, n_time_chunks=50, update_every=100)` | Treinador PINN que aplica os pesos causais ao resíduo da EDP junto com perdas de condição inicial/contorno; permite consultar e plotar os pesos. |

#### `pinneapple_neural.trainer.checkpoint`

Salva e carrega um checkpoint com estado do modelo, do otimizador, configuração, metadados e normalizadores via `torch.save`.

<small>`pinneapple_neural/trainer/checkpoint.py` · 40 linhas · testes: 3</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Checkpoint` | classe | `(model_state, optim_state, cfg, meta, normalizers)` | Dataclass com estado do modelo, estado do otimizador, configuração, metadados e normalizadores. |
| `save_checkpoint` | função | `(path, ckpt) -> None` | Grava um `Checkpoint` em disco com `torch.save`, criando o diretório se preciso. |
| `load_checkpoint` | função | `(path) -> Checkpoint` | Lê um arquivo salvo e reconstrói o `Checkpoint`. |

#### `pinneapple_neural.trainer.collocation`

Amostragem adaptativa de pontos de colocação (RAR): um conjunto ativo é reamostrado de um pool grande com densidade puxada para onde o resíduo da EDP é alto; inclui utilitários de escala por canal e filtragem de NaN. [sem teste]

<small>`pinneapple_neural/trainer/collocation.py` · 413 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AdaptiveCollocationConfig` | classe | `(n_active, n_pool, resample_every, residual_frac, residual_batch, replace)` | Configuração do RAR: pontos ativos, tamanho do pool, frequência de reamostragem, fração guiada pelo resíduo e tamanho do lote de avaliação. |
| `AdaptiveCollocationSampler` | classe | `(pool, config=None, residual_fn=None) -> None` | Mantém um pool de candidatos e um subconjunto ativo, reamostrado periodicamente com probabilidade proporcional ao resíduo ao quadrado misturada com uniforme. |
| `estimate_channel_scales` | função | `(Y, quantile=0.95, min_scale=1e-06, channel_names=None) -> List[float]` | Estima a escala de cada canal de saída como o quantil q de \|Y\|, para normalizar canais de magnitudes muito diferentes. |
| `filter_nan_rows` | função | `(*tensors, warn=True) -> Tuple[torch.Tensor, ...]` | Remove, de vários tensores ao mesmo tempo, as linhas onde algum tem NaN ou Inf. |

#### `pinneapple_neural.trainer.datamodule`

Monta DataLoaders de treino/validação/teste com split consistente (`pinneapple_data.splits`), adaptadores de item e pipeline de pré-processamento. [sem teste]

<small>`pinneapple_neural/trainer/datamodule.py` · 200 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ItemAdapter` | classe | `()` | Interface que converte itens do dataset em dicts de lote com `x` e, opcionalmente, `y`, `meta` e `physics`. |
| `FnAdapter` | classe | `(x_fn, y_fn, meta_fn=None)` | Adaptador por funções: `x_fn`, `y_fn` e `meta_fn` opcional aplicados a cada amostra (por exemplo `PhysicalSample`). |
| `AdaptedSequenceDataset` | classe | `(samples, adapter)` | Dataset que embrulha uma sequência Python e aplica o adaptador em `__getitem__`. |
| `DataModule` | classe | `(dataset, split, group_ids, batch_size, num_workers, pin_memory, persistent_workers, drop…)` | Constrói e guarda em cache os DataLoaders de treino/validação/teste com split e pré-processamento consistentes. |

#### `pinneapple_neural.trainer.distributed`

Treino multi-GPU/multinó com DistributedDataParallel do PyTorch, lançado por `torchrun` ou por `mp.spawn`. [sem teste]

<small>`pinneapple_neural/trainer/distributed.py` · 323 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DDPTrainerConfig` | classe | `(backend, world_size, master_addr, master_port, gradient_sync_every, find_unused_paramete…)` | Configuração do treino DDP: backend, número de processos, endereço/porta do mestre, clip de gradiente, AMP e frequência de log. |
| `DDPPINNTrainer` | classe | `(model, config=None)` | Treinador PINN multi-GPU com DDP: inicializa o grupo de processos, sincroniza gradientes e pode lançar workers via `mp.spawn`. |
| `is_distributed` | função | `() -> bool` | Diz se há um grupo de processos distribuído ativo. |
| `get_rank` | função | `() -> int` | Retorna o rank do processo atual (0 fora de modo distribuído). |
| `get_world_size` | função | `() -> int` | Retorna o número total de processos (1 fora de modo distribuído). |

#### `pinneapple_neural.trainer.graybox`

Treino PINN gray-box (estilo AI-Aristotle): rede principal aprende o estado e redes auxiliares pequenas aprendem um termo desconhecido da equação; opcionalmente destila esse termo em fórmula com PySR. [sem teste]

<small>`pinneapple_neural/trainer/graybox.py` · 280 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GrayBoxNet` | classe | `(n_in=1, n_out=1, width=32, depth=3)` | MLP pequena e rasa que aproxima um termo funcional desconhecido de uma EDP/EDO, mantida simples para facilitar a destilação simbólica. |
| `GrayBoxConfig` | classe | `(epochs_supervised, epochs_joint, lr_main, lr_graybox, lr_params, w_data, w_pde, pde_warm…)` | Configuração do treino gray-box: épocas das duas fases, taxas de aprendizado, pesos de dados e EDP, aquecimento da EDP, clip e EMA. |
| `train_graybox` | função | `(main_model, graybox_nets, known_residual_fn, obs_coords, obs_values, collocation_sampler…)` | Treino em duas fases: ajusta a rede principal só aos dados e depois treina rede principal, redes gray-box e parâmetros desconhecidos contra dados + resíduo da EDP. |
| `distill_to_symbolic` | função | `(graybox_net, coord_bounds, n_samples=2000, var_names=None, n_iterations=40) -> Dict[str,…)` | Amostra a rede gray-box no domínio e roda regressão simbólica PySR para obter uma expressão fechada; exige o pacote opcional `pysr`. |

#### `pinneapple_neural.trainer.heteroscedastic`

Quantificação de incerteza heteroscedástica: a rede prevê média e log-variância por ponto (perda NLL gaussiana) e, com MC-dropout, separa incerteza aleatória e epistêmica. [sem teste]

<small>`pinneapple_neural/trainer/heteroscedastic.py` · 136 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HeteroscedasticWrapper` | classe | `(base_network, n_out_physical)` | Envolve uma rede com saída 2*n para expor `forward_uq` (média e log-variância); `forward` devolve só a média. |
| `init_logvar_bias` | função | `(output_layer, n_out_physical, bias_value=-4.6) -> None` | Inicializa o viés das colunas de log-variância da última camada para começar com incerteza pequena. |
| `heteroscedastic_nll_loss` | função | `(pred_mean, pred_logvar, target) -> torch.Tensor` | Perda NLL gaussiana com log-variância limitada a [-10, 10]. |
| `compute_aleatoric_epistemic_uq` | função | `(model, coords, n_mc=50, forward_kwargs=None) -> Dict[str, torch.Tensor]` | Decompõe a incerteza preditiva em aleatória e epistêmica com amostras MC-dropout, pela lei da variância total. |

#### `pinneapple_neural.trainer.hpc`

Utilitários de HPC: FSDP, ZeRO-1, CUDA graphs, ganchos de compressão de gradiente (PowerSGD, Top-K), geração de comandos `torchrun` e scripts SLURM, profiler e busca de batch máximo. [sem teste]

<small>`pinneapple_neural/trainer/hpc.py` · 819 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FSDPConfig` | classe | `(sharding_strategy, mixed_precision, min_num_params, activation_checkpointing, cpu_offloa…)` | Configuração do FSDP: estratégia de sharding, precisão mista, tamanho mínimo de módulo, checkpointing de ativações e offload para CPU. |
| `CUDAGraphModule` | classe | `(module, warmup_steps=3, device='cuda') -> None` | Envolve um módulo para capturar e reexecutar um CUDA graph na inferência com formato de entrada fixo. |
| `TorchRunConfig` | classe | `(n_nodes, n_procs_per_node, master_addr, master_port, rdzv_backend, max_restarts)` | Configuração de um lançamento `torchrun` (nós, processos por nó, endereço, porta, rendezvous, reinícios). |
| `SLURMConfig` | classe | `(job_name, n_nodes, n_gpus_per_node, n_cpus_per_task, time_limit, partition, account, mem…)` | Configuração de um script batch SLURM (nome, nós, GPUs, CPUs, tempo, partição, conta, memória, módulos, ambiente conda). |
| `ProfilerConfig` | classe | `(trace_dir, record_shapes, profile_memory, with_flops, with_stack, schedule_wait, schedul…)` | Configuração do `PINNeAPPleProfiler`: diretório de traces, o que registrar e agenda de profiling. |
| `PINNeAPPleProfiler` | classe | `(config=None) -> None` | Context manager sobre `torch.profiler.profile` que registra tempo, memória e FLOPs por operador e exporta trace Chrome. |
| `AutoBatchSizeFinder` | classe | `(trial_fn, lo=1, hi=8192, verbose=True) -> None` | Busca binária do maior tamanho de lote que cabe na memória da GPU, usando uma função de teste fornecida. |
| `wrap_fsdp` | função | `(model, cfg=None, *, process_group=None) -> nn.Module` | Envolve um modelo com Fully Sharded Data Parallel; devolve o modelo sem mudança se FSDP não estiver disponível. |
| `wrap_zero_optimizer` | função | `(optimizer_class, params, **optimizer_kwargs) -> torch.optim.Optimizer` | Cria um otimizador com `ZeroRedundancyOptimizer` (ZeRO-1), que divide o estado do otimizador entre ranks. |
| `register_powersgd_hook` | função | `(model, *, matrix_approximation_rank=1, start_powersgd_iter=10, use_error_feedback=True) …)` | Registra no modelo DDP o gancho de compressão de gradiente PowerSGD (baixo posto). |
| `register_topk_hook` | função | `(model, *, compression_ratio=0.1) -> None` | Registra no modelo DDP um gancho que envia só a fração Top-K de gradientes de maior módulo. |
| `build_torchrun_cmd` | função | `(script, cfg, *, script_args='') -> str` | Gera a string de comando `torchrun` para um script de treino. |
| `build_slurm_script` | função | `(torchrun_cmd, slurm, *, output_dir='slurm_logs') -> str` | Gera um script batch SLURM completo que executa um comando `torchrun`. |

#### `pinneapple_neural.trainer.inference`

Inferência mínima: carrega pesos de um checkpoint num modelo e faz predição em modo `eval` sem gradiente. [sem teste]

<small>`pinneapple_neural/trainer/inference.py` · 22 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `load_checkpoint` | função | `(model, path, map_location='cpu') -> Dict[str, Any]` | Carrega os pesos da chave `model` de um arquivo `torch.save` no modelo e devolve o dict lido. |
| `predict` | função | `(model, x, device=None) -> torch.Tensor` | Roda o modelo em modo `eval`, sem gradiente, no dispositivo indicado. |

#### `pinneapple_neural.trainer.inverse_pinn`

PINN inverso: otimiza ao mesmo tempo os pesos da rede e parâmetros físicos escalares desconhecidos, a partir de sensores esparsos e de um resíduo de EDP/EDO fornecido pelo usuário.

<small>`pinneapple_neural/trainer/inverse_pinn.py` · 218 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `InverseParam` | classe | `(initial_value, log_space=False, name='param')` | Escalar treinável, opcionalmente em espaço log para manter o parâmetro positivo. |
| `InversePINN` | classe | `(model, inverse_params, residual_fn, sensor_points, sensor_values, domain_bounds, n_col=2…)` | Treina pesos da rede e parâmetros físicos desconhecidos contra resíduo nos pontos de colocação mais erro nos sensores, com parada antecipada. |

#### `pinneapple_neural.trainer.lbfgs_finetune`

Ajuste fino com L-BFGS em várias rodadas independentes, cada uma com um closure novo (novo mini-lote), em vez de uma fase longa única.

<small>`pinneapple_neural/trainer/lbfgs_finetune.py` · 112 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LBFGSFinetuneResult` | classe | `(round_losses, round_iters)` | Registro por rodada do L-BFGS multi-rodada (perdas e iterações), com a perda final. |
| `multi_round_lbfgs` | função | `(params, closure_builder, n_rounds, n_steps_per_round, lr=1.0, history_size=50, line_sear…)` | Executa N fases L-BFGS independentes, cada uma com um closure recém-construído, com callbacks opcionais ao fim de cada rodada. |

#### `pinneapple_neural.trainer.loss_balancer`

Balanceamento automático de pesos de perda para PINNs: ReLoBRaLo, SoftAdapt, Lagrangiano aumentado, Inverse Dirichlet, LR annealing, PCGrad, pesos conjuntos via softplus e um meta-agendador `AutoBalancer` que troca de método quando o treino estagna.

<small>`pinneapple_neural/trainer/loss_balancer.py` · 1.229 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ReLoBRaLo` | classe | `(loss_names, *, tau=0.1, alpha=0.999, update_every=1, clip_min=0.001, clip_max=1000.0, in…)` | Balanceamento ReLoBRaLo: pesos por softmax das razões de perda relativas, com memória aleatória (lookback) e suavização. |
| `SoftAdapt` | classe | `(loss_names, *, tau=0.1, beta=0.0, normalized=True, update_every=1, clip_min=0.001, clip_…)` | SoftAdapt: pesos por softmax das taxas de variação de cada perda, opcionalmente normalizados. |
| `AugmentedLagrangian` | classe | `(loss_names, *, pde_key='pde', rho=0.1, lambda_init=1.0, lambda_min=0.0, lambda_max=10000…)` | Trata a perda da EDP como restrição e atualiza multiplicadores de Lagrange com penalidade `rho`, após um aquecimento. |
| `InverseDirichlet` | classe | `(model, loss_names, *, shared_layer=None, update_every=50, alpha=0.9, clip_min=0.001, cli…)` | Pesos a partir das estatísticas (variância) dos gradientes de cada termo numa camada compartilhada. |
| `LRAnnealing` | classe | `(model, loss_names, *, reference='pde', update_every=100, alpha=0.1, clip_min=0.001, clip…)` | Algoritmo 1 de Wang, Teng & Perdikaris (2021): ajusta pesos pela razão entre o gradiente máximo do termo de referência e o gradiente médio de cada termo. |
| `PCGrad` | classe | `(model, loss_names, *, update_every=1, weights=None) -> None` | PCGrad: projeta gradientes conflitantes entre termos de perda antes de combiná-los. |
| `AutoBalancerConfig` | classe | `(method, check_every, health_window, plateau_rel_thresh, divergence_thresh, max_switches,…)` | Configuração do meta-agendador: método inicial, janela de saúde, limiares de platô e divergência, máximo de trocas e método de fallback. |
| `AutoBalancer` | classe | `(model, loss_names, *, config=None, pde_key='pde') -> None` | Meta-agendador que monitora a saúde do treino e troca de método de balanceamento quando há platô ou divergência. |
| `JointAdaptiveWeights` | classe | `(loss_names, *, initial_weights=None, sp_beta=1.0, max_raw_init=20.0, scale=None) -> None` | Pesos de perda treináveis junto com o modelo, parametrizados por softplus. |
| `AutoPDEWeightCalibrator` | classe | `(target_ratio=0.01, *, once=True, clip_min=1e-20, clip_max=1000000.0) -> None` | Calibra o peso da perda da EDP para atingir uma razão-alvo em relação às perdas de contorno/dados. |

#### `pinneapple_neural.trainer.losses`

Perdas supervisionadas e com física: perda supervisionada (MSE/MAE), gancho de perda física e combinação ponderada, com helper para a Arena. [sem teste]

<small>`pinneapple_neural/trainer/losses.py` · 148 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SupervisedLoss` | classe | `(kind)` | Perda supervisionada entre predição e `y`: MAE quando `kind` é `mae`, MSE nos demais casos. |
| `PhysicsLossHook` | classe | `(physics_loss_fn)` | Gancho que chama uma função de perda física do usuário com modelo, predição e lote. |
| `CombinedLoss` | classe | `(supervised, physics, w_supervised, w_physics)` | Soma ponderada da perda supervisionada e da física, devolvendo dict com `total` e componentes. |
| `build_loss` | função | `(*, problem_spec, model_capabilities=None, weights=None, supervised_kind='mse', physics_l…)` | Monta um `CombinedLoss` para a Arena, ligando a física só quando o problema e o modelo a suportam. |

#### `pinneapple_neural.trainer.metrics`

Métricas de avaliação (MSE, MAE, RMSE, R2, L2 relativo, erro máximo) e agregadores; as classes base `Metric` e `Metrics` levantam NotImplementedError por serem abstratas. [sem teste]

<small>`pinneapple_neural/trainer/metrics.py` · 109 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Metric` | classe | `(name)` | Classe base de métrica com `name`; `__call__` levanta NotImplementedError. |
| `MSE` | classe | `(name)` | Erro quadrático médio. |
| `MAE` | classe | `(name)` | Erro absoluto médio. |
| `RMSE` | classe | `(name)` | Raiz do erro quadrático médio. |
| `R2` | classe | `(name, eps)` | Coeficiente de determinação R2, com `eps` no denominador. |
| `RelL2` | classe | `(name, eps)` | Erro L2 relativo, \|\|y_hat - y\|\| / \|\|y\|\|, com `eps`. |
| `MaxError` | classe | `(name)` | Maior erro absoluto. |
| `Metrics` | classe | `()` | Interface base de conjunto de métricas; `compute` levanta NotImplementedError. |
| `MetricBundle` | classe | `(metrics)` | Aplica uma lista de `Metric` e devolve um dict nome -> valor. |
| `RegressionMetrics` | classe | `(eps)` | Conjunto fixo de métricas de regressão (MSE, MAE, RMSE, R2) num único `compute`. |
| `default_metrics` | função | `() -> List[Metric]` | Retorna a lista padrão de métricas. |
| `regression_metrics_bundle` | função | `() -> Metrics` | Retorna um objeto `Metrics` padrão para regressão. |

#### `pinneapple_neural.trainer.metrics_cfg`

Constrói um `MetricBundle` a partir de uma lista de nomes ou dicts vinda de configuração YAML. [sem teste]

<small>`pinneapple_neural/trainer/metrics_cfg.py` · 81 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_metrics_from_cfg` | função | `(cfg) -> Metrics` | Converte nomes como `mse`, `rmse`, `rel_l2`, `r2`, `mae`, `max_error` (ou dicts com `name`) num `MetricBundle`. |

#### `pinneapple_neural.trainer.monotonic`

Restrições de monotonicidade para surrogates de regressão: penalidade hinge sobre derivadas entrada-saída e medida da fração de violações.

<small>`pinneapple_neural/trainer/monotonic.py` · 55 linhas · testes: 2</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `monotonicity_penalty` | função | `(model, z, cols, signs, *, scale=None, margin=0.0, output=0, create_graph=True) -> torch.…)` | Penalidade hinge quadrática média que é zero quando cada derivada amostrada tem o sinal exigido com folga `margin`. |
| `monotone_violation_share` | função | `(model, z, cols, signs, output=0) -> list[float]` | Fração de linhas onde a derivada da saída em cada coluna restrita tem o sinal errado. |

#### `pinneapple_neural.trainer.multiadam`

Otimizador MultiAdam (Yao et al., ICML 2023): Adam com momentos separados por termo de perda, para que um termo muito grande não domine o passo.

<small>`pinneapple_neural/trainer/multiadam.py` · 103 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MultiAdam` | classe | `(params, lr=0.001, betas=(0.99, 0.99), eps=1e-08)` | Otimizador Adam com um par de momentos (m, v) por termo de perda; `step` recebe a lista de perdas e calcula os gradientes. |

#### `pinneapple_neural.trainer.multistart`

Treino com múltiplos reinícios (sementes diferentes), vigia de NaN e escolha do melhor modelo por uma função de score. [sem teste]

<small>`pinneapple_neural/trainer/multistart.py` · 363 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MultiRestartConfig` | classe | `(n_restarts, seed_base, nan_patience, max_failed, verbose, keep_all)` | Configuração dos reinícios: número, semente base, paciência de NaN, máximo de falhas, verbosidade e se guarda todos os resultados. |
| `RestartResult` | classe | `(model, history, score, restart_idx, seed, nan_detected)` | Resultado de um reinício: modelo, histórico, score, índice, semente e se houve NaN. |
| `MultiRestartResult` | classe | `(best, all_results, n_valid, n_nan)` | Resultado agregado: melhor reinício, todos os resultados (opcional) e contagens de válidos e com NaN. |
| `MultiRestartTrainer` | classe | `(model_factory, train_fn, *, config=None, score_fn=None, on_restart_end=None) -> None` | Roda uma função de treino várias vezes com sementes diferentes, descarta execuções com NaN e devolve a de menor score. |

#### `pinneapple_neural.trainer.noether_trainer`

Treinador para modelos Noether (UPT, AB-UPT, Transolver) com dados do PINNeAPPle, num loop PyTorch próprio ou delegando ao `HydraRunner` do Noether (dependência opcional).

<small>`pinneapple_neural/trainer/noether_trainer.py` · 344 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NoetherTrainConfig` | classe | `(epochs, batch_size, lr, weight_decay, field_weights, loss_fn, warmup_steps, grad_clip, d…)` | Configuração do treino Noether: épocas, lote, taxa de aprendizado, decaimento, pesos por campo, aquecimento, clip, rastreador e checkpoints. |
| `NoetherTrainResult` | classe | `(train_losses, val_losses, best_val_loss, best_epoch, checkpoint_path, extras)` | Resumo do treino Noether: perdas de treino/validação, melhor perda e época, caminho do checkpoint e extras. |
| `NoetherSurrogateTrainer` | classe | `(model, train_loader, val_loader=None, config=None) -> None` | Treina um modelo Noether (UPT, AB-UPT, Transolver) com DataLoaders do PINNeAPPle; `from_hydra_config` delega tudo ao HydraRunner do Noether. |

#### `pinneapple_neural.trainer.normalizers`

[shim] Reexporta `Normalizer`, `StandardScaler` e `MinMaxScaler` de `pinneapple_data.transforms`. [sem teste]

<small>`pinneapple_neural/trainer/normalizers.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_neural.trainer.parallel`

Paralelismo e otimização de GPU: escolha de dispositivo, AMP, prefetch CUDA, acumulação de gradiente, `torch.compile`, DataParallel, sweep em grade paralelo, checkpointing de gradiente e monitor de throughput. [sem teste]

<small>`pinneapple_neural/trainer/parallel.py` · 508 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AMPContext` | classe | `(device='cuda', dtype=torch.bfloat16, enabled=True) -> None` | Context manager de precisão mista (`torch.amp`), que vira no-op fora de CUDA. |
| `CUDAPrefetcher` | classe | `(loader, device='cuda') -> None` | Pré-carrega o próximo lote para a GPU num stream CUDA separado, sobrepondo cópia e computação. |
| `GradAccumConfig` | classe | `(accumulate_steps, clip_grad_norm)` | Configuração da acumulação de gradiente: número de passos e clip de norma. |
| `GradAccumTrainer` | classe | `(model, optimizer, cfg=None, amp_ctx=None) -> None` | Acumula gradientes de micro-lotes e dá o passo do otimizador a cada N micro-lotes. |
| `SweepConfig` | classe | `(param_grid, n_jobs, backend, timeout_per_trial)` | Configuração do sweep em grade: grade de parâmetros, número de jobs, backend (threads/processos) e timeout. |
| `ThroughputMonitor` | classe | `(device_index=0) -> None` | Mede amostras/s, memória da GPU e utilização (com `pynvml`, se houver) por época. |
| `best_device` | função | `(prefer_cuda=True, allow_mps=True) -> torch.device` | Retorna o melhor dispositivo disponível: CUDA, depois MPS (se permitido), depois CPU. |
| `count_gpus` | função | `() -> int` | Retorna o número de GPUs CUDA disponíveis. |
| `gpu_info` | função | `() -> List[Dict[str, Any]]` | Retorna um dict de informações por GPU disponível. |
| `maybe_compile` | função | `(model, *, mode='default', fullgraph=False, dynamic=False) -> nn.Module` | Aplica `torch.compile` se o PyTorch suportar; caso contrário devolve o modelo original. |
| `wrap_data_parallel` | função | `(model, device_ids=None) -> nn.Module` | Envolve o modelo em `DataParallel` quando há mais de uma GPU. |
| `unwrap_model` | função | `(model) -> nn.Module` | Remove o invólucro DataParallel/DDP e devolve o modelo cru. |
| `run_parallel_sweep` | função | `(trial_fn, cfg) -> List[Dict[str, Any]]` | Executa um sweep em grade em paralelo (threads ou processos) e devolve a lista de resultados. |
| `enable_gradient_checkpointing` | função | `(model) -> nn.Module` | Liga checkpointing de gradiente em modelos que o suportam (estilo HuggingFace) ou envolvendo camadas. |

#### `pinneapple_neural.trainer.preprocess`

Pipeline de pré-processamento de lotes com passos ajustados no fold de treino (normalização, preenchimento de NaN, winsorização, escala robusta, log, diferenciação, features de solver). [sem teste]

<small>`pinneapple_neural/trainer/preprocess.py` · 445 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PreprocessStep` | classe | `()` | Protocolo de passo de pré-processamento com `fit` e `apply`. |
| `PreprocessPipeline` | classe | `(steps)` | Sequência de passos ajustados nos lotes de treino e aplicados a cada lote, com `state_dict` para salvar. |
| `NormalizeStep` | classe | `(key, dim, store_key, enabled, scaler)` | Ajusta um `StandardScaler` numa chave do lote no treino e normaliza em `apply`. |
| `SolverFeatureStep` | classe | `(solver, mode, select_var_dim, reduce_fft_to, fit_noop)` | Calcula features com um solver (por exemplo FFT) a partir de `x` e as anexa ou substitui no lote. |
| `MissingValueStep` | classe | `(key, strategy, enabled)` | Preenche NaN em `x` por forward fill, backward fill ou zero. |
| `WinsorizeStep` | classe | `(key, q_low, q_high, enabled, low_, high_)` | Corta valores por quantis ajustados no fold de treino (winsorização). |
| `RobustScaleStep` | classe | `(key, eps, enabled, median_, iqr_)` | Escala robusta (x - mediana) / (IQR + eps), ajustada no fold de treino. |
| `LogStep` | classe | `(key, enabled, clamp_min)` | Transformação log opcional para sinais não negativos. |
| `DifferencingStep` | classe | `(key, time_dim, order, enabled, anchor_key)` | Diferenciação ao longo do tempo para estacionariedade; reduz o comprimento da sequência em `order`. |

#### `pinneapple_neural.trainer.self_scaled_qn`

Otimizadores quase-Newton auto-escalados (BFGS, SSBFGS, SSBroyden) com Hessiana inversa densa e busca de linha strong-Wolfe, conforme Urban et al. (JCP 2025).

<small>`pinneapple_neural/trainer/self_scaled_qn.py` · 186 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SelfScaledQuasiNewton` | classe | `(params, variant='ssbroyden', max_iter=20, tolerance_grad=1e-12, tolerance_change=1e-14, …)` | Otimizador BFGS/SSBFGS/SSBroyden com Hessiana inversa densa e busca strong-Wolfe, usado como `torch.optim.LBFGS` com closure. |
| `self_scaled_parameters` | função | `(s, y, g, alpha, Hy, n, variant)` | Calcula (tau, phi) de uma atualização auto-escalada, conforme as equações 11-23 do artigo. |

#### `pinneapple_neural.trainer.splits`

[shim] Reexporta `SplitSpec` e `split_indices` de `pinneapple_data.splits`. [sem teste]

<small>`pinneapple_neural/trainer/splits.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_neural.trainer.time_marching`

Time-marching sequencial (Wight & Zhao 2020): divide o tempo em janelas e treina uma rede por janela, usando a solução final de uma como condição inicial da próxima. [sem teste]

<small>`pinneapple_neural/trainer/time_marching.py` · 388 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TimeMarchingTrainer` | classe | `(model_factory, t_start, t_end, n_windows=10, epochs_per_window=2000, n_col=1000, ic_weig…)` | Treina uma rede por janela de tempo, congelando a predição no fim de cada janela como condição inicial da seguinte; avalia e plota o resultado completo. |

#### `pinneapple_neural.trainer.trainer`

`Trainer` unificado para PINNs e modelos supervisionados: perdas que retornam tensor ou dict com `total`, AMP, DDP opcional, callbacks, checkpoint e log de auditoria.

<small>`pinneapple_neural/trainer/trainer.py` · 397 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TrainConfig` | classe | `(epochs, lr, weight_decay, grad_clip, amp, device, seed, deterministic, log_dir, run_name…)` | Configuração do `Trainer`: épocas, taxa de aprendizado, clip, AMP, dispositivo, semente, determinismo, log, DDP e validação com física. |
| `Trainer` | classe | `(model, loss_fn, metrics=None, preprocess=None, early_stopping=None, checkpoint=None)` | Treinador genérico: `fit(train_loader, val_loader, cfg)` roda épocas com perdas tensor/dict, métricas, early stopping, checkpoint e log JSONL. |

#### `pinneapple_neural.trainer.two_phase`

Treinador em duas fases: primeiro ajuste supervisionado a dados de solver/medição, depois imposição de resíduos da EDP e condições de contorno. [sem teste]

<small>`pinneapple_neural/trainer/two_phase.py` · 414 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TwoPhaseConfig` | classe | `(phase1_epochs, phase1_lr, phase2_epochs, phase2_lr, weight_decay, grad_clip, device, amp…)` | Configuração das duas fases: épocas e taxas de cada fase, modo combinado e rampa do peso da física. |
| `TwoPhaseHistory` | classe | `(phase1_epochs, phase1_loss, phase2_epochs, phase2_loss, phase2_pde_loss, phase2_bc_loss)` | Histórico de perdas das duas fases (incluindo componentes EDP e contorno na fase 2). |
| `TwoPhaseTrainer` | classe | `(model, supervised_loss_fn, physics_loss_fn, config=None)` | Executa a fase supervisionada e depois a fase de física, com otimizadores e históricos separados. |
| `UnnormModel` | classe | `(model, Y_mean, Y_std)` | Envolve um modelo treinado em alvos normalizados para devolver predições na escala física (y * std + mean). |

#### `pinneapple_neural.trainer.viz`

[shim] `plot_history` mantido por compatibilidade: delega a `pinneapple_viz.pinn` quando disponível e cai para um gráfico matplotlib mínimo. [sem teste]

<small>`pinneapple_neural/trainer/viz.py` · 57 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): shim ou obsoleto</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_history` | função | `(history, keys=('train_total', 'val_total'), *, log_scale=True, show=True)` | Plota o histórico de perdas em escala log, via `pinneapple_viz.pinn` quando disponível. |

#### `pinneapple_neural.trainer.weight_scheduler`

Agendamento automático de pesos de perda (SA-PINN, GradNorm, razão de perdas, NTK) com interface única `WeightScheduler`, que também despacha para os métodos de `loss_balancer`.

<small>`pinneapple_neural/trainer/weight_scheduler.py` · 799 linhas · testes: 11</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `WeightSchedulerConfig` | classe | `(method, initial_weights, update_every, lr, alpha, clip_min, clip_max, ema_decay)` | Configuração do agendamento de pesos: método (de `self_adaptive` até `auto`), pesos iniciais, frequência, taxas, limites e decaimento EMA. |
| `SelfAdaptiveWeights` | classe | `(loss_names, config) -> None` | Pesos de perda aprendíveis (SA-PINN) atualizados por subida de gradiente, crescendo nos termos mais difíceis. |
| `GradNormBalancer` | classe | `(model, loss_names, shared_layer=None, config=None) -> None` | GradNorm: ajusta pesos para manter as normas de gradiente de cada termo numa camada compartilhada proporcionais à média. |
| `LossRatioBalancer` | classe | `(loss_names, target_ratios=None, config=None) -> None` | Mantém as razões entre perdas (via EMA) próximas de valores-alvo em relação à perda da EDP. |
| `NTKWeightBalancer` | classe | `(model, loss_names, n_proj=50, config=None) -> None` | Pesos inversamente proporcionais ao traço do NTK de cada termo, estimado por Hutchinson. |
| `WeightScheduler` | classe | `(model, loss_names, config=None) -> None` | Interface única: `step(losses)` aplica os pesos atuais, atualiza conforme o método e expõe parâmetros aprendíveis quando houver. |

#### `pinneapple_neural.utils`

Helpers comuns: `as_tensor` extrai o tensor de previsão de qualquer saída de modelo e `fit_supervised` é um laço Adam mínimo em pares (x, y).

<small>`pinneapple_neural/utils.py` · 92 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `as_tensor` | função | `(out, *, attrs=OUTPUT_TENSOR_ATTRS) -> torch.Tensor` | Extrai o tensor de previsão de um Tensor, dicionário de campos, tupla ou objeto de saída (`PINNOutput`, `ModelOutput` etc.). |
| `fit_supervised` | função | `(model, x, y, *, steps=1000, lr=0.001, batch=32, loss='mse', weight_decay=0.0, seed=0, fo…)` | Treina o modelo para mapear x em y com Adam em mini-batches aleatórios (perda MSE ou L2 relativa) e devolve o histórico. |

#### `pinneapple_neural.workflows`

Workflows prontos de treino e previsão sobre as arquiteturas: operador de volume denso (simples e paramétrico) e surrogate híbrido com pós-modelo físico. [sem teste direto do `__init__`]

<small>`pinneapple_neural/workflows/__init__.py` · 56 linhas · **sem teste direto**</small>

#### `pinneapple_neural.workflows.dense_volume_operator`

Workflow de treino/previsão de um `FNO3d` em série temporal real de volumes `(T, C, D, H, W)`, com o truque de pushforward para estabilizar o rollout autorregressivo.

<small>`pinneapple_neural/workflows/dense_volume_operator.py` · 248 linhas · testes: 3</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DenseVolumeOperatorConfig` | classe | `(in_channels, out_channels, width, modes, layers, epochs, lr, val_frac, batch_size, rollo…)` | Configuração do treino do `FNO3d` em volumes densos (largura, modos, camadas, épocas, validação, rollout, pushforward). |
| `train_dense_volume_operator` | função | `(frames, cfg, *, seed=0) -> Tuple[Checkpoint, List[Dict[str, Any]]]` | Treina um `FNO3d` numa série de volumes e devolve o checkpoint e o histórico. |
| `load_dense_volume_operator` | função | `(checkpoint, *, device='cpu') -> nn.Module` | Reconstrói o `FNO3d` a partir do checkpoint (também dos checkpoints antigos de mesmo formato). |
| `predict_single_step` | função | `(checkpoint, input_frame, *, device='cpu') -> np.ndarray` | Um passo de previsão em unidades reais, o regime em que a rede é mais precisa. |
| `rollout_dense_volume_operator` | função | `(checkpoint, initial_frame, n_steps, *, device='cpu') -> np.ndarray` | Rollout autorregressivo a partir de um quadro real; devolve n_steps + 1 quadros. |

#### `pinneapple_neural.workflows.hybrid_surrogate_physics`

Modelo híbrido: um surrogate aprende só as grandezas intermediárias e um pós-modelo físico explícito (nunca treinado, trocável sem retreino) calcula a grandeza derivada; a incerteza passa por Monte Carlo. Só NumPy (scikit-learn opcional).

<small>`pinneapple_neural/workflows/hybrid_surrogate_physics.py` · 517 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsPostModel` | classe | `(name, source)` | Pós-modelo físico explícito, com nome e fonte científica, aplicado às grandezas intermediárias. |
| `CallablePostModel` | classe | `(fn, name, source)` | Envolve uma função simples como `PhysicsPostModel`. |
| `SurrogatePrediction` | classe | `(mean, std, sampler)` | Distribuição preditiva do surrogate: média, desvio e amostrador opcional. |
| `Surrogate` | classe | `()` | Interface mínima de surrogate: `fit(X, Y)` e `predict(X)`. |
| `GaussianProcessSurrogate` | classe | `(*, n_restarts=2, random_state=0, use_sklearn=None)` | Um processo gaussiano por coluna de saída (scikit-learn se houver, senão um GP em NumPy). |
| `PCASurrogate` | classe | `(base, *, n_components=None, variance=None)` | Reduz as saídas por PCA, ajusta o surrogate base nos escores e volta ao espaço original. |
| `TransformedSurrogate` | classe | `(base, forward, inverse, name='transform')` | Ajusta o surrogate base em `forward(Y)` e desfaz com `inverse` na previsão. |
| `PerQuantitySurrogate` | classe | `(blocks)` | Um surrogate por bloco de colunas de saída. |
| `HybridPrediction` | classe | `(intermediates, derived_at_mean, derived_mean, derived_std, derived_lower, derived_upper,…)` | Saída do híbrido: intermediárias, grandeza derivada na média, média e desvio por Monte Carlo e intervalo. |
| `HybridSurrogatePhysics` | classe | `(surrogate, post_model, *, param_names=None)` | Compõe o surrogate com o pós-modelo; `with_post_model` troca a física sem retreinar. |

#### `pinneapple_neural.workflows.parametric_dense_volume_operator`

Extensão multicaso de `dense_volume_operator`: um `FNO3d` treinado em vários casos que diferem num parâmetro físico `theta`, usado como canal constante de entrada, com normalização global.

<small>`pinneapple_neural/workflows/parametric_dense_volume_operator.py` · 382 linhas · testes: 3</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ParametricCase` | classe | `(name, frames, theta)` | Um caso de treino: sequência de volumes e o parâmetro físico `theta` que a gerou. |
| `ParametricDenseVolumeOperatorConfig` | classe | `(state_channels, theta_dim, width, modes, layers, epochs, lr, val_frac, batch_size, rollo…)` | Configuração do treino paramétrico (canais de estado, dimensão de theta, escala log de theta, pushforward). |
| `train_parametric_dense_volume_operator` | função | `(cases, cfg, *, seed=0) -> Tuple[Checkpoint, List[Dict[str, Any]]]` | Treina um `FNO3d` condicionado em theta em todos os casos juntos. |
| `load_parametric_dense_volume_operator` | função | `(checkpoint, *, device='cpu') -> nn.Module` | Reconstrói o `FNO3d` paramétrico a partir do checkpoint. |
| `predict_single_step_parametric` | função | `(checkpoint, input_frame, theta, *, device='cpu') -> np.ndarray` | Um passo de previsão num `theta` dado, inclusive não visto no treino. |
| `rollout_parametric_dense_volume_operator` | função | `(checkpoint, initial_frame, theta, n_steps, *, device='cpu') -> np.ndarray` | Rollout autorregressivo num `theta` dado, recolocando o canal de theta a cada passo. |

### Exemplos de uso reais do repositório

**Montar um modelo pelo registro com tamanho genérico** — `tests/pinneapple_neural/test_registry_size_kwargs.py`

```python
@pytest.mark.parametrize("name", ["vanilla_pinn", "bench_mlp"])
def test_size_kwargs_are_honoured(name):
    model, report = ModelRegistry.build_with_report(name, in_dim=2, out_dim=1, hidden_dim=16, n_layers=2)
    assert not ({"hidden_dim", "n_layers"} & set(report.dropped)), report.dropped
    assert _linears(model) == [(2, 16), (16, 16), (16, 1)]            # two hidden layers of width 16
```

**Estudo de ROM em snapshots separados** — `tests/test_rom_study.py`

```python
def test_dmd_and_opinf_forecast_a_linear_wave_and_time_interpolation_cannot():
    r = rom_study(_wave(), ranks=[2, 4, 8], methods=["persistence", "pod_rbf", "dmd", "opinf"], timing_repeats=2)
    err = {(x["method"], x["rank"]): x["error_mean"] for x in r.rows}
    assert err[("persistence", None)] > 0.5                          # the waves move: the last snapshot is wrong
    assert err[("opinf", 4)] < 1e-3          # rank 4 holds the two waves (OpInf works on centred coefficients)
    assert err[("dmd", 8)] < 1e-3            # uncentred DMD: the mean is one more mode
    assert err[("dmd", 4)] > 0.1 and err[("opinf", 2)] > 0.1
```

**Checks físicos num laço de treino PhysicsNeMo** — `tests/test_training_callbacks.py`

```python

def physicsnemo_loop(darcy, w, mon, steps=STEPS):
    a, u, RL = darcy
    net = fno(u)
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    hook, u2, step = PhysicsNeMoHook(mon), float((u[:64] ** 2).mean()), 0
    while step < steps:
        for batch in loader(a, u):
            loss = loss_fn(net, batch, RL, w, u2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if hook(net, step, loss) or step == steps:
```

**Trainer com perda supervisionada + física** — `examples/end_to_end/03_pinn_physics_loss_training.py`

```python
combined = CombinedLoss(
    supervised=SupervisedLoss("mse"),
    physics=PhysicsLossHook(physics_loss_fn),
    w_supervised=1.0,
    w_physics=1.0,
)

def loss_fn(model, y_hat, batch):
    return combined(model, y_hat, batch)

trainer = Trainer(model=model, loss_fn=loss_fn, metrics=default_metrics())
```

**MultiAdam recebe os termos de perda no step** — `tests/test_multiadam.py`

```python
opt = MultiAdam([p])
opt.step([p.sum() ** 2, (p - 1).sum() ** 2])
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 46 de 198.
- Funções de teste em arquivos que importam algum módulo do pacote: 438.

Itens do catálogo de métodos (`pinneapple_catalog/method_status.json`) cujo código está neste pacote:

| id | método | status | testes contra referência | outros testes |
|---|---|---|---:|---:|
| S21 | Digital twin state estimation: KF/EKF/UKF/EnKF, prognostics, signal reconstruction | tested | 2 | 39 |
| T1 | Base trainer: AMP, gradient accumulation, auto batch size, CUDA graphs, profiler | tested | 0 | 9 |
| T2 | Distributed: DDP, FSDP, SLURM, torchrun, PowerSGD | untested | 0 | 0 |
| T3 | Causal training | untested | 0 | 0 |
| T4 | Time marching | untested | 0 | 0 |
| T5 | Two phases Adam -> L-BFGS + L-BFGS finetune | tested | 0 | 6 |
| T6 | Multi-restart | untested | 0 | 0 |
| T7 | Adaptive residual-based collocation | untested | 0 | 0 |
| T8 | Loss balancing: ReLoBRaLo, SoftAdapt, PCGrad, Augmented Lagrangian, Inverse Dirichlet, LR Annealing, AutoBalancer | validated | 1 | 6 |
| T9 | Weight schedulers: SA-PINN, GradNorm, LossRatio, NTK | tested | 0 | 2 |
| T10 | Domain decomposition: XPINN, DoMINO/D3M | tested | 0 | 2 |
| T11 | Inverse PINN (learnable physical parameters) | tested | 0 | 3 |
| T12 | Gray-box + symbolic regression (PySR) | untested | 0 | 0 |
| T13 | Uncertainty: heteroscedastic, B-PINN, MC Dropout, deep ensembles | tested | 0 | 20 |
| T14 | SINDy | tested | 0 | 2 |
| T15 | Adaptive hyperparameter sweep / architecture search (arena NAS) | untested | 0 | 0 |
| T16 | Noether trainer (research only) | tested | 0 | 1 |
| T17 | Post-training advisor / audit | untested | 0 | 0 |
| T20 | Self-scaled quasi-Newton: BFGS / SSBFGS / SSBroyden (new, 2026-09-24) | validated | 1 | 3 |
| T22 | Large Physics Model blocks (Fourier encoding, multi-scale neighbourhoods, geometry code, ensemble UQ, OOD score, fine-tuning) | tested | 0 | 1 |
| T23 | MultiAdam: per-loss-term Adam moments for multi-term PINN losses | validated | 1 | 3 |

### Limitações conhecidas

- Boa parte das arquiteturas não tem teste direto (testes=0 no inventário): famílias inteiras como `recurrent`, `transformers`, `reservoir_computing`, `classical_ts`, `continuous` e `convolutions`, além de `predictor`, `siren`, `hash_grid`, `afno` e `modified_mlp`.
- Muitos modelos se descrevem como 'MVP' no docstring (filtros de Kalman, ARIMA, Neural ODE/SDE/CDE, GNNs, Koopman); `rom.deep_uq_rom` é explicitamente um esqueleto ('scaffold').
- Dependências opcionais: `emmiai-noether` (bridge Noether), `physicsnemo` (field_operators e `require_physicsnemo_model`), `tfc` (X-TFC), `torchdiffeq` (métodos `dopri5`/`adams` de `GraphNeuralODE`), `onnxruntime` (`export_onnx`), `scikit-image` (isosuperfícies), `matplotlib` (gráficos), `lightning`/`transformers` (callbacks).
- O bridge Noether é só para pesquisa: emite `ResearchOnlyWarning` sempre e levanta `ResearchOnlyError` com `PINNEAPPLE_COMMERCIAL_MODE=1`; as classes viram `None` se o pacote não estiver instalado.
- `MeshGraphNet` implementa só o tipo de aresta da malha; o conjunto de arestas no espaço do mundo (contato) do artigo original não existe.
- `register_all` ignora em silêncio (log em debug) famílias cujo import falha, então um modelo pode simplesmente não aparecer no registro.
- A exportação ONNX de `ReducedModel` só cobre alguns métodos e opções e levanta `NotImplementedError` nos demais (por exemplo, RBF que não seja thin-plate spline).
- O adaptador padrão de `adapters.base` usa `batch.get('x') or batch.get('x_col')`, o que falha com tensores de mais de um elemento; na prática ele depende de o modelo ter `forward_batch`.
- Muitos módulos não têm teste direto (por exemplo `causal`, `time_marching`, `two_phase`, `hpc`, `distributed`, `parallel`, `collocation`, `multistart`, `graybox`, `heteroscedastic`).
- `Metric` e `Metrics` em `metrics.py` são bases abstratas e levantam NotImplementedError.
- Dependências opcionais: Optuna em `adaptive_sweep` (sem ele cai para busca aleatória), PySR em `distill_to_symbolic`, Noether em `noether_trainer`, matplotlib nos gráficos e `pynvml` no `ThroughputMonitor`.
- O docstring de `hpc.py` cita `CUDAGraphContext`, `PowerSGDCompressor` e `TopKGradCompressor`, mas o módulo expõe só `CUDAGraphModule` e as funções `register_powersgd_hook`/`register_topk_hook`; o docstring também fala em estender `pinneapple_train.parallel`.
- FSDP, ZeRO, CUDA graphs, DDP e ganchos de compressão exigem GPU e grupo de processos inicializado; fora disso há fallback ou nada acontece.
- `normalizers.py`, `splits.py` e `viz.py` são shims de compatibilidade que reexportam ou delegam para `pinneapple_data` e `pinneapple_viz`.

## 2. `pinneapple_data`

### Objetivo

`pinneapple_data` é a camada de dados do PINNeAPPle. Ela define o formato comum de dados físicos, o UPD (Unified Physical Data), cujo contêiner é `PhysicalSample`: estado (de preferência um `xarray.Dataset`), geometria, esquema das equações, domínio e proveniência. Em volta desse formato há gravação e leitura (`serialization`, `zarr_store` e variantes com cache, shards e prefetch), `DataLoader`s do PyTorch, splits, normalizadores, validação e geração de pontos de colocação e batches para treinar PINNs (`collocation`, `active_learning`, `adapters.pinn_batch_builders`, `upd_types`).

O pacote também concentra os datasets. `datasets` tem um registro de datasets prontos (física gerada analiticamente ou por solver, geometria, séries temporais), carregadores de dados reais da web e clientes de fontes externas; vários destes últimos caem em dados sintéticos quando a fonte não responde. Há ainda leitores de datasets públicos (DeepMind `meshgraphnets`), um amostrador de campos aleatórios gaussianos para benchmarks de operadores, geradores sintéticos (`synth`) e a auditoria física de datasets (`audit`), que verifica se os dados de fato resolvem a EDP declarada antes do treino.

Uma terceira parte é a camada de dados de engenharia usada pelos apps (`apps/`): leitura de malhas e resultados CAE de OpenFOAM, CalculiX/Abaqus, Gmsh, VTK e STL (`cae`), qualidade de malha, comparação de resultados, pré-voo de simulações (`preflight`), extração de metadados e convergência (`simulation_metadata`), saúde de dados de sensores (`data_health`), padronização de unidades e fontes (`unified`, `physical_units`), snapshots para modelos de ordem reduzida, linhagem de modelos (`lineage`), preenchimento de datasheets e formulários a partir de documentos (`formfill`, `udr1`) e otimização de processo só com dados. Apps como `simulation_preflight`, `mesh_quality`, `data_health` e `model_lineage` importam estes módulos diretamente.

Os contratos em `contracts` (`ParameterSpace`, `DatasetSpec`, `ModelRef`, `StudyResult`) e o design de experimentos em `doe` ligam os apps entre si: um desenho de experimentos vira campanha de simulação (`pinneapple_simulation.external_solvers.campaign_cases`), as execuções viram dataset e o dataset vira modelo substituto. O pacote se apoia em `pinneapple_design.geometry` para geometria, em `pinneapple_physics` para `ProblemSpec` e presets, e em `pinneapple_security.archives` para expandir uploads com segurança; `pinneapple_train` reexporta `splits` e `transforms`.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 115 | 29.019 | 129 | 256 | 507 (54 arquivos) | 70 |

Pasta de testes dedicada: `tests/pinneapple_data` (33 funções de teste).

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `active_learning` | Escolha adaptativa de pontos de colocação de PINN por resíduo, variância ou combinação. | 1 | 5 | 22 |
| `adapters` | Adaptadores de geometria, shards UPD, batches PINN e arquivos `.splash` para o formato do pacote. | 5 | 19 | 0 |
| `audit` | Auditoria física de datasets de treino: EDP satisfeita, vazamento, duplicatas, diversidade e restrições. | 1 | 10 | 49 |
| `augmentation` | Aumento de nuvens de pontos com ruído, rotação/escala e interpolação. | 1 | 3 | 0 |
| `cache` | Cache LRU em memória limitado por número de itens. | 1 | 2 | 0 |
| `cache_bytes` | Cache LRU em memória limitado por bytes. | 1 | 3 | 1 |
| `cae` | Leitura de malhas e resultados CAE (OpenFOAM, CalculiX/Abaqus, Gmsh, VTK, STL), qualidade de malha, comparação de resultados e exportação neutra. | 13 | 74 | 62 |
| `chunking` | Fatiamento de tensores grandes em pedaços. | 1 | 2 | 0 |
| `collate` | Collate e transferência de dispositivo de batches PINN e supervisionados. | 1 | 3 | 0 |
| `collocation` | Geração de pontos de colocação a partir de limites, `ProblemSpec`, domínios, malhas ou STL. | 1 | 2 | 7 |
| `contracts` | Contratos de dados abertos entre apps (`ParameterSpace`, `DatasetSpec`, `ModelRef`, `StudyResult`) e tabelas com sidecar. | 4 | 27 | 175 |
| `data_health` | Relatório de qualidade física de dados de sensores, ensaios e planta. | 1 | 6 | 37 |
| `dataloaders` | Construção de `DataLoader` do PyTorch para shards UPD e `PhysicalSample`. | 1 | 3 | 0 |
| `datasets` | Registro de datasets prontos (física, geometria, séries temporais, reais e externos) e downloads. | 9 | 28 | 10 |
| `device` | Pin de memória e troca de dispositivo de amostras. | 1 | 2 | 0 |
| `doe` | Design de experimentos e planejamento de campanhas de simulação. | 4 | 10 | 21 |
| `expressions` | Avaliador seguro de expressões de restrição, sem `eval`. | 1 | 5 | 22 |
| `formfill` | Preenchimento de formulários e datasheets de engenharia a partir de documentos, com fonte de cada valor. | 14 | 31 | 44 |
| `geometry_features` | Nuvem de pontos e grade SDF a partir de malha triangular. | 1 | 3 | 0 |
| `grf_sampler` | Campos aleatórios gaussianos 1D/2D para funções de entrada de benchmarks de operador. | 1 | 5 | 0 |
| `lineage` | Linhagem de modelos de engenharia (digital thread) com checagens e exportação PROV. | 3 | 4 | 13 |
| `noether_dataset` | Ponte entre datasets do Noether e `PhysicalSample`. | 1 | 3 | 5 |
| `parameter_sampling` | Amostragem de espaços de parâmetros (LHS, Sobol, Halton, Saltelli). | 1 | 2 | 5 |
| `physical_sample` | Contêiner `PhysicalSample` alinhado ao UPD. | 1 | 1 | 28 |
| `physical_units` | Unidades, conversão para SI e inferência de grandeza física, só com a biblioteca padrão. | 1 | 12 | 52 |
| `physics_case` | `PhysicsCase`: geometria + spec + solver + resultados referenciados juntos. | 1 | 2 | 47 |
| `prefetch_adaptive` | Streaming de store Zarr UPD com prefetch adaptativo. | 1 | 2 | 0 |
| `preflight` | Checagem de casos OpenFOAM e decks CalculiX antes de rodar o solver. | 4 | 11 | 13 |
| `process_optimizer` | Otimização de processo industrial só com dados (gradient boosting). | 1 | 7 | 16 |
| `serialization` | Gravação e leitura de `PhysicalSample` em `.pt`, HDF5 e Zarr. | 1 | 9 | 11 |
| `simulation_metadata` | Extração de metadados e avaliação de convergência de simulações OpenFOAM e CalculiX. | 5 | 25 | 25 |
| `snapshots` | Matrizes de snapshots para modelos de ordem reduzida. | 1 | 3 | 31 |
| `splits` | Divisão treino/validação/teste aleatória, temporal ou por grupo. | 1 | 2 | 0 |
| `stl_import` | Leitor de STL sem dependências. | 1 | 3 | 0 |
| `synth` | Geradores de dados sintéticos (EDP, distribuições, curvas, imagens, geometria). | 15 | 20 | 0 |
| `transforms` | Normalizadores (`StandardScaler`, `MinMaxScaler`). | 1 | 3 | 2 |
| `udr1` | Atalhos do formulário ASME U-DR-1 sobre `formfill`. | 1 | 3 | 36 |
| `unified` | Padronização de várias fontes de dados de engenharia num esquema físico único. | 1 | 9 | 24 |
| `upd_dataset` | Dataset UPD com validação e persistência. | 1 | 1 | 5 |
| `upd_types` | Tipos puros do pipeline UPD (shard, amostragem, batch). | 1 | 5 | 0 |
| `validators` | Validação de `PhysicalSample`. | 1 | 3 | 0 |
| `zarr_cached_store` | Store Zarr UPD com cache LRU por contagem. | 1 | 2 | 0 |
| `zarr_cached_store_bytes` | Store Zarr UPD com cache LRU por bytes. | 1 | 2 | 0 |
| `zarr_iterable` | `IterableDataset` sobre store Zarr UPD com divisão por worker. | 1 | 1 | 0 |
| `zarr_prefetch` | `IterableDataset` Zarr com prefetch em thread e transferência para GPU. | 1 | 2 | 0 |
| `zarr_shard_iterable` | `IterableDataset` balanceado sobre stores Zarr fatiados em shards. | 1 | 2 | 0 |
| `zarr_shards` | Escritor de stores Zarr UPD fatiados por chave. | 1 | 2 | 0 |
| `zarr_store` | Store Zarr colunar do UPD. | 1 | 1 | 1 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_data`

Raiz do pacote: importa, cada um dentro de `try/except` (falha silenciosa), STL, `PhysicalSample`, armazenamento Zarr, collate, colocação, active learning, normalizadores, tipos UPD, splits e construtores de batch; os `datasets` são importados sem proteção.

<small>`pinneapple_data/__init__.py` · 117 linhas · testes: 78</small>

#### `pinneapple_data.active_learning`

Active learning para escolher pontos de colocação de PINN: seleção por resíduo (RAR/RAD), por variância de um modelo com incerteza, combinação ponderada com diversidade, e um treinador que refina a colocação a cada `al_every` épocas.

<small>`pinneapple_data/active_learning.py` · 674 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ActiveLearningConfig` | classe | `(strategy, n_candidates, n_select, n_initial, n_iterations, exploitation_weight, seed)` | Configuração do active learning: estratégia, número de candidatos, de pontos escolhidos por rodada, iniciais, iterações, peso de exploração e semente. |
| `ResidualBasedAL` | classe | `(config, bounds) -> None` | Escolhe pontos de colocação onde o resíduo da EDP é maior, pegando os top-k (RAR) ou sorteando com peso proporcional ao resíduo (RAD). |
| `VarianceBasedAL` | classe | `(config, bounds) -> None` | Escolhe pontos onde a variância da saída do modelo é maior; exige um modelo com incerteza que exponha `variance_fn` (ensemble, rede bayesiana). |
| `CombinedAL` | classe | `(config, bounds, residual_weight=0.6, variance_weight=0.3, diversity_weight=0.1) -> None` | Pontua candidatos por soma ponderada de resíduo, variância e diversidade (distância aos pontos já escolhidos), todos normalizados. |
| `AdaptiveCollocationTrainer` | classe | `(model, physics_fn, sampler, al_config=None, optimizer=None, device='cpu') -> None` | Envolve um PINN e uma função física, mantém o próprio conjunto de colocação e acrescenta pontos escolhidos pela estratégia de active learning durante o treino. |

#### `pinneapple_data.adapters`

Reexporta os adaptadores de geometria, de shards UPD, de batch PINN e de arquivos `.splash`; os de geometria e UPD ficam em `try/except`. [sem teste]

<small>`pinneapple_data/adapters/__init__.py` · 63 linhas · **sem teste direto**</small>

#### `pinneapple_data.adapters.geom_adapter`

Carrega geometria (arquivo, STL, objeto) como `GeometryAsset` via `pinneapple_design.geometry` e a anexa a um `PhysicalSample`; o docstring ainda usa o nome antigo `pinneapple_geom`. [sem teste]

<small>`pinneapple_data/adapters/geom_adapter.py` · 286 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GeometryLoadOptions` | classe | `(scale, translate, rotate_euler_deg, repair, compute_normals, boundary_labels)` | Opções de carga de geometria: escala, translação, rotação em graus de Euler, reparo, cálculo de normais e rótulos de contorno. |
| `load_geometry_asset` | função | `(geom, *, options=None) -> Any` | Carrega ou monta um `GeometryAsset` a partir de caminho, STL ou especificação, delegando ao registro de geometria de `pinneapple_design`. |
| `stl_to_upd` | função | `(path, *, repair=True, compute_normals=True, units='m') -> Any` | Lê um STL com `trimesh` e devolve um `PhysicalSample` de domínio malha com `vertices` e `faces` como tensores e a proveniência do arquivo. |
| `attach_geometry` | função | `(sample, geom_asset) -> Any` | Anexa um `GeometryAsset` a uma amostra (dict ou objeto) e define o tipo de domínio como `mesh` se ainda não houver. |

#### `pinneapple_data.adapters.pinn_batch_builders`

Converte fontes diferentes (bundle do Arena, solver, medições reais) para o batch em dict usado pelo compilador PINN (`x_col`, `x_bc`, `y_bc`, `x_ic`, `y_ic`, `x_data`, `y_data`, `ctx`). [sem teste]

<small>`pinneapple_data/adapters/pinn_batch_builders.py` · 149 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PINNBatch` | classe | `(batch)` | Invólucro fino de um batch PINN em memória (o dict de tensores). |
| `build_from_bundle` | função | `(bundle, *, n_collocation=4096, n_boundary=2048, n_data=2048, device=None, dtype=torch.fl…)` | Monta um batch PINN a partir de um `BundleData` de `pinneapple_arena`, sorteando pontos de colocação, contorno e dados. |
| `build_from_solver` | função | `(problem_spec, geometry, solver_cfg, *, device=None, dtype=torch.float32) -> PINNBatch` | Monta um batch PINN chamando um `solver_fn` da configuração ou usando tensores já prontos nela. |
| `build_from_real_data` | função | `(adapter_cfg, *, device=None, dtype=torch.float32) -> PINNBatch` | Monta um batch PINN a partir de medições reais, via `loader_fn` ou tensores `x_data`/`y_data` dados diretamente. |

#### `pinneapple_data.adapters.splash_archive_adapter`

Lê um caso `.splash` (caso OpenFOAM zipado) para arrays NumPy: nuvem de pontos para PINN ou volumes densos em grade para operadores neurais como `FNO3d`; código promovido de um projeto derivado. [sem teste]

<small>`pinneapple_data/adapters/splash_archive_adapter.py` · 315 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SplashMesh` | classe | `(n_cells, n_points, n_faces, n_internal_faces, cell_centers, cell_size, cell_delta, bound…)` | Malha reconstruída de um `.splash`: contagens de células, pontos e faces, centros e tamanhos de célula e limites do domínio. |
| `FieldSnapshot` | classe | `(time, fields)` | Campos de um instante de tempo de um caso `.splash` (tempo + dict de arrays). |
| `SplashCase` | classe | `(path, mesh, time_dirs, transport, meta)` | Caso `.splash` aberto: malha, diretórios de tempo e propriedades; lê campos, instantâneos, nuvem de pontos e volume denso. |
| `load_mesh` | função | `(zf, prefix='constant/polyMesh/') -> SplashMesh` | Monta centros e tamanhos de célula a partir do polyMesh binário dentro do zip `.splash` aberto. |
| `save_mesh_cache` | função | `(mesh, path) -> None` | Grava um `SplashMesh` num `.npz` comprimido para reaproveitar depois. |
| `load_mesh_cache` | função | `(path) -> SplashMesh` | Lê de volta um `SplashMesh` gravado por `save_mesh_cache`. |
| `open_case` | função | `(path, mesh_cache_dir=None) -> SplashCase` | Abre um arquivo `.splash`, reconstruindo a malha ou lendo-a do cache. |
| `load_dense_volumes` | função | `(splash_path, field_components, nx, ny, nz, *, mesh_cache_dir=None) -> Tuple[np.ndarray, …)` | Lê todos os diretórios de tempo de um `.splash` como série temporal de volumes densos `nx×ny×nz`, para malha de um único bloco hexaédrico estruturado. |

#### `pinneapple_data.adapters.upd_adapter`

Carrega shards UPD (Zarr + JSON de metadados) como `xarray.Dataset` e os converte em `PhysicalSample` ou anexa o estado a uma amostra existente. [sem teste]

<small>`pinneapple_data/adapters/upd_adapter.py` · 288 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `load_upd_item` | função | `(upd) -> Tuple[xr.Dataset, Dict[str, Any]]` | Lê um shard UPD (par de caminhos, dict ou dataset pronto) e devolve o `xarray.Dataset` e os metadados. |
| `upd_to_physical_sample` | função | `(upd, *, schema_override=None, provenance_extra=None) -> Any` | Converte um shard UPD em `PhysicalSample`, com esquema opcional sobrescrito e proveniência extra. |
| `attach_upd_state` | função | `(sample, upd) -> Any` | Anexa estado, esquema, domínio e proveniência de um shard UPD a uma amostra existente (dict ou objeto). |

#### `pinneapple_data.audit`

Auditoria física de um dataset de treino só com NumPy/SciPy: verifica se os dados resolvem a própria EDP, vazamento entre splits, duplicatas, diversidade, restrições e unidades, e gera um cartão de auditoria.

<small>`pinneapple_data/audit.py` · 519 linhas · testes: 49</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AuditCheck` | classe | `(id, status, summary, value, threshold, evidence)` | Resultado de uma verificação da auditoria: id, status (passa/falha/não se aplica), resumo, valor, limite e evidência. |
| `AuditCard` | classe | `(dataset, splits, physics, checks, tolerances)` | Cartão de auditoria de um dataset com todas as verificações; dá a decisão geral e exporta para dict, JSON e Markdown. |
| `DatasetPhysics` | classe | `(name, columns, check, tolerances, units)` | Física por amostra de um dataset: função `check(sample)` que devolve erros relativos por nome, e as tolerâncias de cada um. |
| `darcy_consistency` | função | `(a, u, f=1.0) -> dict[str, float]` | Verifica uma amostra de Darcy na discretização de `operator_benchmarks.darcy_2d`: distância à solução do próprio problema discreto e erro nas condições de contorno. |
| `effective_sample_count` | função | `(x) -> float` | Razão de participação do espectro de componentes principais das amostras: aproxima quantas direções independentes os dados cobrem. |
| `near_duplicates` | função | `(x, near=0.001) -> dict[str, Any]` | Acha duplicatas exatas e quase duplicatas dentro de um conjunto, relativas à distância mediana ao vizinho mais próximo. |
| `leakage` | função | `(train, test, near=0.001) -> dict[str, Any]` | Para cada amostra de teste acha a de treino mais próxima e marca como cópia (vazamento) as que ficam perto demais. |
| `constraint_mask` | função | `(expr, cols) -> np.ndarray` | Avalia elemento a elemento onde uma relação declarada (como `"Cd > 0"` ou `"a == b * c"`) vale; serve também para checar predições. |
| `audit_dataset` | função | `(splits, *, name='dataset', physics=None, inputs=None, units=None, expected_units=None, c…)` | Audita splits `{split: {coluna: array}}` com física, restrições, diversidade, vazamento e unidades, e devolve um `AuditCard`. |
| `audit_zoo_dataset` | função | `(name, *, offline=None, **kw) -> AuditCard` | Audita uma entrada de `pp.datasets` usando a física e as restrições declaradas no seu catálogo. |

#### `pinneapple_data.augmentation`

Aumento de dados para nuvens de pontos `(coords, fields)`: ruído gaussiano, rotação/escala de coordenadas e novos pontos por interpolação, sem acoplamento a framework. [sem teste]

<small>`pinneapple_data/augmentation.py` · 139 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `augment_noise` | função | `(coords, fields, n_copies, std_fraction=0.05, seed=0) -> Dict[str, Any]` | Faz `n_copies` cópias do dataset somando ruído gaussiano proporcional ao desvio-padrão de cada campo; coordenadas ficam iguais. |
| `augment_transform` | função | `(coords, fields, n_copies, max_rotation_deg=15.0, scale_range=(0.9, 1.1), seed=0) -> Dict…)` | Faz `n_copies` cópias com rotação no plano dos dois primeiros eixos e escala isotrópica das coordenadas; os campos não são rotacionados. |
| `augment_interpolation` | função | `(coords, fields, n_new_points, method='linear', seed=0) -> Dict[str, Any]` | Gera pontos novos dentro do fecho convexo da nuvem e estima os campos por interpolação dispersa, com vizinho mais próximo onde a interpolação dá NaN. |

#### `pinneapple_data.cache`

Cache LRU em memória, seguro para threads, limitado por número de itens. [sem teste]

<small>`pinneapple_data/cache.py` · 120 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CacheStats` | classe | `(hits, misses, evictions)` | Contadores do `LRUCache`: acertos, falhas e despejos. |
| `LRUCache` | classe | `(max_items=256)` | Cache LRU seguro para threads limitado por `max_items`, com `get`, `put` e `clear`. |

#### `pinneapple_data.cache_bytes`

Cache LRU seguro para threads limitado por bytes, para tensores e amostras.

<small>`pinneapple_data/cache_bytes.py` · 179 linhas · testes: 1</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ByteCacheStats` | classe | `(hits, misses, evictions, bytes_in_use)` | Contadores do `ByteLRUCache`: acertos, falhas, despejos e bytes em uso. |
| `ByteLRUCache` | classe | `(max_bytes=512 * 1024 * 1024, weigher=default_weigher)` | Cache LRU que limita o total de bytes (`max_bytes`) em vez do número de itens, despejando os menos usados. |
| `default_weigher` | função | `(value) -> int` | Estima o tamanho em bytes de um valor: `nbytes` de tensor, soma para dicts e amostras, e um valor fixo de reserva para o resto. |

#### `pinneapple_data.cae`

Camada de dados CAE: lê malhas e resultados de OpenFOAM, CalculiX/Abaqus, Gmsh, VTK, STL e tabelas de pontos num único tipo `Mesh`, com qualidade de malha, Courant e distância à parede.

<small>`pinneapple_data/cae/__init__.py` · 14 linhas · testes: 44</small>

#### `pinneapple_data.cae.compare`

Compara dois resultados campo a campo (simulação, experimento, predição de IA ou referência analítica), interpolando o candidato nas posições da referência e calculando MAE, RMSE, erro máximo, viés, L2 relativo, NRMSE e R².

<small>`pinneapple_data/cae/compare.py` · 420 linhas · testes: 24</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Interpolator` | classe | `(src_pts, dst_pts, method='auto')` | Interpola um campo candidato em pontos de consulta: cópia direta se as posições coincidem, linear (Delaunay) dentro do domínio e vizinho mais próximo fora. |
| `locations` | função | `(m, field) -> Tuple[np.ndarray, str, Optional[np.ndarray]]` | Devolve onde um campo vive (centros de célula ou nós), o tipo de posição e os pesos (volumes de célula, se houver). |
| `cell_centres_volumes` | função | `(m) -> Tuple[np.ndarray, Optional[np.ndarray]]` | Calcula (e guarda em cache na malha) centros e volumes de célula pela geometria de volumes finitos. |
| `field_value` | função | `(m, f) -> np.ndarray` | Devolve os valores de um campo da malha, ou de uma componente dele quando o nome a indica. |
| `fields_of` | função | `(m) -> Dict[str, Dict[str, Any]]` | Lista os campos de uma malha com posição (ponto/célula), número de componentes, unidade e tamanho. |
| `match_fields` | função | `(ref, cand, mapping=None) -> List[Tuple[str, str]]` | Pareia campos de referência e candidato: mapeamento explícito, depois mesmo nome sem caixa, depois apelidos (U~velocity, p~pressure, T~temperature, DISP~displacement). |
| `tolerance_verdicts` | função | `(result, tolerance) -> Optional[str]` | Marca cada campo de um resultado de `compare` como `pass`/`fail` pelo erro L2 relativo contra uma tolerância e devolve o veredito geral. |
| `compare` | função | `(ref, cand, mapping=None, method='auto', fields=None) -> Dict[str, Any]` | Compara referência e candidato em todos os campos pareados e devolve métricas por campo e componente; levanta `ValueError` se não houver campo em comum. |
| `surface` | função | `(m, max_tris=300000) -> Optional[Dict[str, Any]]` | Extrai a superfície de contorno em triângulos, com a célula de cada um, para exibição; em casos 2D do OpenFOAM mantém só um lado `empty`. |
| `views` | função | `(ref, res, max_points=60000) -> Dict[str, Any]` | Monta o payload de exibição por campo: superfície colorida, gráfico de linha ou mapa de pontos 2D/3D. |

#### `pinneapple_data.cae.dataset`

Dataset físico neutro: normaliza qualquer resultado suportado em geometria, malha, coordenadas e campos com grandeza e unidade, e exporta para VTU, HDF5, Parquet, CSV, NPZ, JSON ou dataset PINNeAPPle (Zarr UPD).

<small>`pinneapple_data/cae/dataset.py` · 507 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `quantity_of` | função | `(name) -> Tuple[Optional[str], Optional[str]]` | Infere grandeza física e unidade a partir do nome de um campo, por padrões de nome. |
| `poly_cells` | função | `(m)` | Converte células OpenFOAM em células VTK (hexaedro, prisma, tetraedro, pirâmide) e as demais em poliedros VTK. |
| `field_scalar` | função | `(values) -> Tuple[Optional[np.ndarray], str]` | Reduz um campo a um escalar por ponto/célula: o próprio valor, a magnitude de um vetor ou a tensão de von Mises de um tensor simétrico. |
| `build_dataset` | função | `(m, *, unit_system='auto', to_si=False, rho=None, location='auto', files=None) -> Dict[st…)` | Normaliza uma `Mesh` no dataset neutro (dict com arrays e metadados), com sistema de unidades, conversão opcional para SI e escolha de posição. |
| `describe` | função | `(ds) -> Dict[str, Any]` | Descrição do dataset segura para JSON (o esquema mostrado ao usuário), sem arrays. |
| `export` | função | `(ds, fmt) -> Tuple[bytes, str, str]` | Serializa o dataset no formato pedido e devolve (conteúdo, tipo de mídia, extensão). |
| `write_vtu` | função | `(ds) -> bytes` | Escreve o dataset como VTK XML não estruturado (`.vtu`), incluindo poliedros quando a malha é OpenFOAM. |
| `write_hdf5` | função | `(ds) -> bytes` | Escreve o dataset em HDF5 com malha, campos e metadados (requer `h5py`). |
| `write_parquet` | função | `(ds) -> bytes` | Escreve o dataset como tabela Parquet com unidades nos metadados (requer `pandas` e `pyarrow`). |
| `write_csv` | função | `(ds) -> bytes` | Escreve o dataset como CSV com unidades no cabeçalho e uma linha de comentário com esquema e origem. |
| `write_npz` | função | `(ds) -> bytes` | Escreve pontos, centros de célula, campos e metadados num `.npz` comprimido. |
| `write_json` | função | `(ds, max_values=2000000) -> bytes` | Escreve a descrição em JSON e inclui os valores só se o total couber em `max_values`. |
| `write_pinneapple` | função | `(ds) -> bytes` | Escreve o dataset como um `PhysicalSample` num store Zarr UPD zipado, que se lê com `serialization.load_zarr`. |
| `preview` | função | `(ds, max_fields=6, max_tris=200000) -> Dict[str, Any]` | Payload de exibição: superfície de contorno colorida pelos primeiros campos reduzidos a escalar. |

#### `pinneapple_data.cae.foam`

Leitores OpenFOAM: polyMesh (pontos, faces, owner, neighbour, boundary) e campos de volume de um diretório de tempo, em ASCII ou binário, com ou sem gzip; casos decompostos (`processor*`) não são remontados e nada é executado.

<small>`pinneapple_data/cae/foam.py` · 372 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `header` | função | `(text) -> Dict[str, str]` | Lê o cabeçalho `FoamFile` de um arquivo OpenFOAM como dict de texto. |
| `read_points` | função | `(b) -> np.ndarray` | Lê o arquivo `points` do polyMesh (ASCII ou binário) como array (n, 3). |
| `read_labels` | função | `(b) -> np.ndarray` | Lê uma lista de rótulos inteiros (como `owner` ou `neighbour`) do polyMesh. |
| `read_faces` | função | `(b) -> Tuple[np.ndarray, np.ndarray]` | Lê o arquivo `faces` como lista plana de nós mais offsets (suporta `faceList` e `faceCompactList`). |
| `read_boundary` | função | `(b) -> List[Dict[str, Any]]` | Lê o arquivo `boundary` e devolve os patches com nome, tipo, número de faces e face inicial. |
| `read_polymesh` | função | `(fs, root='') -> Mesh` | Monta uma `Mesh` a partir de `constant/polyMesh` do upload; levanta `ValueError` se o polyMesh não estiver presente. |
| `dims_unit` | função | `(dims) -> Optional[str]` | Converte o vetor de dimensões OpenFOAM (`[0 2 -1 0 0 0 0]`) em texto de unidade. |
| `read_boundary_values` | função | `(b, mesh, width) -> Optional[np.ndarray]` | Valor em cada face de contorno lido do `boundaryField`: o `value` escrito, zero para `noSlip`, o valor da célula vizinha quando nada está escrito, NaN para patches `empty`. |
| `read_field` | função | `(b, n_cells) -> Tuple[np.ndarray, Optional[str], str]` | Lê o `internalField` de um campo de volume: valores por célula, unidade e classe. |
| `time_dirs` | função | `(fs, root='') -> List[str]` | Lista os diretórios de tempo numéricos do caso em ordem crescente. |
| `read_time` | função | `(fs, mesh, root='', time=None, only=None) -> Dict[str, Any]` | Carrega os campos de volume de um diretório de tempo (o último, por padrão) em `mesh.cell_data`. |
| `boundary_conditions` | função | `(fs, mesh, root='', time=None) -> Dict[str, Dict[str, str]]` | Devolve patch → campo → condição de contorno como escrita (por exemplo `fixedValue uniform (1 0 0)`, `noSlip`), por padrão do primeiro diretório de tempo. |

#### `pinneapple_data.cae.frd`

Leitor do arquivo de resultados CalculiX `.frd` (ASCII): nós, elementos e blocos de resultado (DISP, STRESS, NDTEMP, FLUX...); guarda o último incremento de cada resultado.

<small>`pinneapple_data/cae/frd.py` · 117 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `read_frd` | função | `(b) -> Mesh` | Lê um `.frd` ASCII e devolve uma `Mesh` com os resultados do último incremento em `point_data` e a lista de incrementos em `source`. |

#### `pinneapple_data.cae.geometry`

Geometria de volumes finitos de qualquer malha, calculada como o `primitiveMesh` do OpenFOAM (centros e vetores de área de face, centros e volumes de célula); converte malhas FE em faces/owner/neighbour. [sem teste direto]

<small>`pinneapple_data/cae/geometry.py` · 249 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FaceMesh` | classe | `(points, face_nodes, face_offsets, owner, neighbour, n_cells, patch_of_face, patch_names,…)` | Malha na forma de faces: nós de face, owner, neighbour, patches e tipos, faces não-manifold e blocos invertidos. |
| `signed_volume_sign` | função | `(btype, pts, conn) -> np.ndarray` | Sinal de uma estimativa simples do volume de cada elemento, por decomposição em tetraedros a partir do primeiro canto. |
| `oriented_blocks` | função | `(mesh) -> List[str]` | Inverte a ordem de nós de blocos inteiros cuja maioria dos elementos sai com volume negativo (convenções diferentes entre VTK, Gmsh e Abaqus) e devolve os nomes dos blocos invertidos. |
| `face_mesh` | função | `(mesh) -> FaceMesh` | Devolve o polyMesh OpenFOAM como está, ou converte uma malha FE de volume em faces (internas primeiro, depois contorno). |
| `face_geometry` | função | `(fm) -> Dict[str, np.ndarray]` | Calcula centros de face, vetores de área e os grupos de cantos por decomposição em triângulos. |
| `cell_geometry` | função | `(fm, fg) -> Dict[str, np.ndarray]` | Calcula centros e volumes de célula por decomposição em pirâmides a partir das faces. |
| `cell_adjacency` | função | `(fm) -> sp.csr_matrix` | Matriz esparsa de adjacência entre células que compartilham uma face interna. |
| `regions` | função | `(adj) -> np.ndarray` | Rótulo de componente conexa de cada célula a partir da matriz de adjacência. |
| `element_adjacency_by_nodes` | função | `(mesh) -> sp.csr_matrix` | Adjacência entre células que compartilham pelo menos um nó (para malhas de superfície e para a checagem de conectividade). |
| `solution_dirs` | função | `(fm, fg) -> Optional[np.ndarray]` | Indica em quais direções não há patch `empty` (casos OpenFOAM 2D/1D). |

#### `pinneapple_data.cae.inp`

Leitor de decks Abaqus/CalculiX `.inp`: blocos de palavra-chave com número de linha, nós, elementos em tipos VTK, conjuntos de nós e elementos e `*INCLUDE` resolvido a partir do upload.

<small>`pinneapple_data/cae/inp.py` · 296 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Block` | classe | `(keyword, params, data, line, file)` | Um bloco de palavra-chave do deck: palavra-chave, parâmetros, linhas de dados, linha e arquivo de origem. |
| `Deck` | classe | `(blocks, includes, missing_includes)` | Deck lido: lista de blocos, includes resolvidos e includes faltantes; `find` busca blocos por palavra-chave. |
| `vtk_type` | função | `(abq) -> Optional[str]` | Traduz um tipo de elemento Abaqus (como `C3D8`) para o tipo VTK correspondente, ou `None`. |
| `read_deck` | função | `(text, name='model.inp', files=None, _depth=0) -> Deck` | Lê o texto de um deck em blocos, guardando o número de linha e resolvendo `*INCLUDE` a partir dos arquivos dados. |
| `resolve_sets` | função | `(deck, kind) -> Dict[str, List[str]]` | Resolve conjuntos NSET ou ELSET (que podem conter outros conjuntos) em listas de rótulos de membros. |
| `deck_mesh` | função | `(deck) -> Mesh` | Converte um `Deck` numa `Mesh` com nós, blocos de elementos e conjuntos. |
| `deck_conditions` | função | `(deck) -> List[Dict[str, object]]` | Lista apoios e cargas do deck por nó: `*BOUNDARY` (graus de liberdade restritos) e `*CLOAD`/`*CFLUX` (soma de forças ou fluxos nodais), entre outros. |

#### `pinneapple_data.cae.io`

Lê qualquer upload (dict caminho → bytes) para uma `Mesh`: caso OpenFOAM, `.inp`, `.frd`, STL, Gmsh/VTK e outros formatos do `meshio`, e tabelas de pontos; `read_any` escolhe o leitor. [sem teste direto]

<small>`pinneapple_data/cae/io.py` · 296 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FormatError` | classe | `()` | Erro para formato não reconhecido ou dependência faltando (por exemplo `meshio`). |
| `kind_of` | função | `(fs) -> Tuple[str, Optional[str]]` | Detecta o tipo do upload (`openfoam`, `inp`, `frd`, `stl`, `meshio`, `points`) e o arquivo principal ou raiz do caso. |
| `read_any` | função | `(fs, fields=True, time=None) -> Mesh` | Lê o upload com o leitor certo, incluindo os campos do último (ou do pedido) diretório de tempo no caso OpenFOAM. |
| `read_stl` | função | `(b) -> Mesh` | Lê STL binário ou ASCII como `Mesh` de superfície de triângulos. |
| `read_meshio` | função | `(name, b) -> Mesh` | Lê um formato suportado pelo `meshio` (Gmsh, VTK/VTU e outros) via arquivo temporário; levanta `FormatError` sem `meshio`. |
| `read_points` | função | `(name, b) -> Mesh` | Lê uma tabela de pontos com colunas de campos (CSV/TXT, Parquet, NPZ, HDF5) como nuvem de pontos com `point_data`. |

#### `pinneapple_data.cae.metrics`

Métricas de escoamento por célula numa malha de volumes finitos: número de Courant e distância à parede mais próxima. [sem teste direto]

<small>`pinneapple_data/cae/metrics.py` · 109 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `courant_number` | função | `(face_area_vectors, face_owner, face_neighbour, U, cell_volumes, dt, where='auto') -> np.…)` | Número de Courant por célula na forma de volumes finitos, `0.5 dt Σ\|U_f·S_f\| / V_P`. |
| `wall_distance` | função | `(cell_centres, wall_face_centres, wall_face_normals=None) -> np.ndarray` | Distância de cada centro de célula ao centro de face de parede mais próximo (`cKDTree`), opcionalmente projetada na normal da face. |
| `mesh_courant_number` | função | `(mesh, U, dt, where='auto') -> np.ndarray` | Aplica `courant_number` a uma `Mesh`, calculando antes a geometria das faces e células. |
| `mesh_wall_distance` | função | `(mesh, patches=None, project=True) -> np.ndarray` | Aplica `wall_distance` a todas as células de uma `Mesh` em relação aos patches dados (padrão: todos os do tipo `wall`). |

#### `pinneapple_data.cae.model`

Contêiner comum de malha e resultado (`Mesh`) usado pelos leitores CAE, pela qualidade de malha, pelo comparador e pela exportação; aceita células FE em blocos ou polyMesh OpenFOAM.

<small>`pinneapple_data/cae/model.py` · 173 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CellBlock` | classe | `(type, conn, ids)` | Bloco de células de um único tipo VTK com conectividade de nós; dá versão linear, cantos e dimensão. |
| `PolyMesh` | classe | `(face_nodes, face_offsets, owner, neighbour, n_cells, patches)` | polyMesh OpenFOAM: faces (lista plana + offsets), owner/neighbour e patches de contorno. |
| `Mesh` | classe | `(points, blocks, poly, boundary_blocks, cell_sets, node_sets, point_data, cell_data, unit…)` | Malha com resultados: pontos, blocos FE ou polyMesh, conjuntos, campos em pontos e células, unidades, valores de contorno e origem. |

#### `pinneapple_data.cae.quality`

Qualidade de malha: métricas de volumes finitos como o `checkMesh` do OpenFOAM, métricas de forma para elementos FE, conectividade e checagens de superfície STL; cada métrica tem limite e fonte em `METRICS`.

<small>`pinneapple_data/cae/quality.py` · 306 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `fv_metrics` | função | `(mesh) -> Dict[str, Any]` | Calcula as métricas de volumes finitos (não-ortogonalidade, skewness, razão de aspecto, volumes, entre outras) a partir da geometria de faces e células. |
| `element_metrics` | função | `(mesh) -> Dict[str, np.ndarray]` | Calcula Jacobiano escalado, skewness e razão de arestas de cada célula da dimensão principal da malha. |
| `surface_checks` | função | `(mesh) -> Dict[str, Any]` | Checa uma superfície de triângulos/quadriláteros: arestas abertas e não-manifold, orientação, faces degeneradas e cascas. |

#### `pinneapple_data.cae.report`

Relatório de saúde da malha: checagens com veredito, estatísticas e histogramas, regiões com problema (aglomerados de células ruins) e o que fazer; também o payload da vista 3D.

<small>`pinneapple_data/cae/report.py` · 371 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build` | função | `(mesh, target='auto', max_regions=12) -> Dict[str, Any]` | Monta o relatório de saúde da malha com limites de CFD (`cfd`), de elementos finitos (`fea`) ou escolhidos automaticamente. |
| `view_payload` | função | `(mesh, rep, max_tris=250000) -> Dict[str, Any]` | Superfície de contorno com o valor da métrica da célula dona de cada face, mais as células das regiões com problema. |
| `worst_cells` | função | `(mesh, rep, n=40) -> List[Dict[str, Any]]` | As `n` piores células de cada métrica do relatório, respeitando o sentido (maior ou menor é pior). |

#### `pinneapple_data.cae.upload`

Expande um upload (arquivos soltos e zips) em `{caminho relativo: bytes}` com limites de tamanho, mantendo malhas e resultados e descartando lixo e pastas `processor*`.

<small>`pinneapple_data/cae/upload.py` · 45 linhas · testes: 62</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `expand` | função | `(files, max_unzipped=MAX_UNZIPPED) -> Dict[str, bytes]` | Expande arquivos e zips (via leitor seguro de `pinneapple_security.archives`) num dict caminho → bytes, respeitando `max_unzipped`. |

#### `pinneapple_data.chunking`

Especificação de fatiamento (`ChunkSpec`) e iteração de tensores em pedaços ao longo de um ou mais eixos, para processar arrays grandes em streaming. [sem teste]

<small>`pinneapple_data/chunking.py` · 112 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ChunkSpec` | classe | `(mode, chunk_size, dim, dims, chunk_sizes)` | Como fatiar um tensor: modo (`none`, `time`, `spatial`), tamanho do pedaço e eixo, ou vários eixos com tamanhos próprios. |
| `iter_chunks` | função | `(x, spec) -> Iterator[torch.Tensor]` | Itera sobre os pedaços de um tensor conforme um `ChunkSpec`. |

#### `pinneapple_data.collate`

Funções de collate: junta batches PINN (`collocation`, `conditions`, `data`, `meta`), move-os para um dispositivo e junta `PhysicalSample` para treino supervisionado. [sem teste]

<small>`pinneapple_data/collate.py` · 132 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `collate_pinn_batches` | função | `(batches) -> Dict[str, Any]` | Concatena uma lista de dicts de batch PINN num único dict (chaves `collocation`, `conditions`, `data`, `meta`). |
| `move_batch_to_device` | função | `(batch, device) -> Dict[str, Any]` | Move todos os tensores de um batch PINN para o dispositivo dado, sem mexer em `meta`. |
| `collate_upd_supervised` | função | `(samples) -> Dict[str, Any]` | Empilha `state['x']` e `state['y']` de uma lista de `PhysicalSample` em `{x, y, meta}` para treino supervisionado. |

#### `pinneapple_data.collocation`

Geração automática de pontos de colocação para PINN a partir de limites, `ProblemSpec`, domínios de `pinneapple_design.geometry`, malhas ou STL, com estratégias `uniform`, `lhs`, `sobol` e `adaptive` (por resíduo).

<small>`pinneapple_data/collocation.py` · 558 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CollocationConfig` | classe | `(n_col, n_bc, n_ic, n_data, strategy, seed, oversample_bc, include_time, t_range)` | Configuração da geração de colocação: estratégia e número de pontos interiores, de contorno e de condição inicial, entre outros. |
| `CollocationSampler` | classe | `(*, bounds, coord_names, sdf_fn=None, boundary_samplers=None, condition_value_fns=None, c…)` | Gerador único de pontos de colocação: construtores `from_bounds`, `from_problem_spec`, `from_domain`, `from_mesh`, `from_stl`, e `sample`/`sample_adaptive` que devolvem o dict de batch PINN. |

#### `pinneapple_data.contracts`

Contratos de dados abertos compartilhados pelos apps (`ParameterSpace`, `DatasetSpec`, `ModelRef`, `StudyResult`) e leitura/escrita de tabelas com sidecar `.contract.json`; o que um app exporta outro importa.

<small>`pinneapple_data/contracts/__init__.py` · 51 linhas · testes: 175</small>

#### `pinneapple_data.contracts.expressions`

Reexporta a avaliação segura de expressões de `pinneapple_data.expressions` (que só depende de NumPy). [shim] [sem teste]

<small>`pinneapple_data/contracts/expressions.py` · 12 linhas · **sem teste direto**</small>

#### `pinneapple_data.contracts.models`

Os quatro contratos como modelos pydantic com `kind` e `schema_version`: ida e volta em JSON/YAML, arquivos, JSON Schema e hash de conteúdo.

<small>`pinneapple_data/contracts/models.py` · 675 linhas · testes: 40</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Contract` | classe | `(schema_version, metadata)` | Base dos contratos: conversão para/de dict e JSON, salvar/carregar arquivo, JSON Schema e hash do conteúdo. |
| `Parameter` | classe | `(name, kind, low, high, values, value, unit, log, group, default, description)` | Um parâmetro de estudo: tipo, limites ou valores, unidade, escala log, grupo e padrão; converte entre coordenada unitária e valor. |
| `ParameterSpace` | classe | `(kind, name, description, parameters, constraints, budget)` | Parâmetros de um estudo e restrições entre eles; converte de/para o cubo unitário, testa viabilidade e lê/escreve tabelas. |
| `Column` | classe | `(name, role, unit, dtype, description)` | Coluna de um dataset: nome, papel (entrada, saída, id), unidade, tipo e descrição. |
| `FileRef` | classe | `(path, format, sha256, size, rows, description)` | Referência a um arquivo do dataset com formato, SHA-256, tamanho e número de linhas; `of` preenche a partir do disco. |
| `Provenance` | classe | `(app, study_id, source, run_ids, library_version, created, notes)` | De onde veio um contrato: app, id do estudo, fonte, ids de execução, versão da biblioteca, data e notas. |
| `Splits` | classe | `(train, val, test, seed, id_column)` | Divisão em splits: ids explícitos por split ou frações com semente. |
| `DatasetSpec` | classe | `(kind, name, version, description, columns, files, n_rows, parameter_space, splits, prove…)` | Tabela de casos ou amostras: colunas, arquivos, splits, espaço de parâmetros amostrado e proveniência; confere os hashes dos arquivos. |
| `Variable` | classe | `(name, unit, low, high, description)` | Variável de entrada ou saída de um modelo com unidade e faixa. |
| `DatasetRef` | classe | `(name, version, content_hash, uri)` | Referência ao dataset de treino de um modelo: nome, versão, hash e URI. |
| `ModelRef` | classe | `(kind, name, version, model_type, uri, sha256, framework, inputs, outputs, dataset, metri…)` | Onde está um modelo treinado e para que pode ser usado: entradas, saídas, dataset, métricas e faixa de validade (`in_range`). |
| `CheckRecord` | classe | `(name, status, value, threshold, evidence, detail)` | Registro de uma checagem; PASS/FAIL/WARN exigem valor ou evidência, para só contar como rodada se rodou. |
| `Table` | classe | `(columns, file)` | Tabela pequena embutida (colunas de mesmo tamanho) ou referência a arquivo; converte para/de DataFrame. |
| `StudyResult` | classe | `(kind, id, study_type, app, title, status, parameter_space, dataset, models, table, metri…)` | Resultado de um estudo (DOE, campanha, UQ, sensibilidade, otimização, calibração, validação): tabela, métricas, checagens e artefatos. |
| `from_dict` | função | `(data) -> Contract` | Constrói o contrato indicado por `data['kind']`. |
| `load` | função | `(path) -> Contract` | Lê um contrato de JSON ou YAML, qualquer que seja dos quatro. |
| `validate` | função | `(data) -> list[str]` | Lista os problemas de um contrato dado como dict, uma linha legível por problema (vazia quando válido). |
| `json_schemas` | função | `() -> dict[str, dict[str, Any]]` | Devolve o JSON Schema de cada contrato, por tipo. |
| `export_schemas` | função | `(directory) -> list[Path]` | Grava `<Kind>.schema.json` de cada contrato no diretório dado. |
| `sha256_file` | função | `(path, chunk=1 << 20) -> str` | Calcula o SHA-256 de um arquivo lendo em blocos. |

#### `pinneapple_data.contracts.tables`

Tabelas em disco com o contrato: CSV, Parquet e HDF5 com sidecar JSON; Parquet e HDF5 também guardam o contrato nos metadados. [sem teste direto]

<small>`pinneapple_data/contracts/tables.py` · 150 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `sidecar_path` | função | `(path) -> Path` | Caminho do sidecar `<arquivo>.contract.json` de uma tabela. |
| `table_format` | função | `(path) -> str` | Deduz o formato da tabela (`csv`, `parquet`, `hdf5`, `json`) pela extensão. |
| `write_sidecar` | função | `(path, contract) -> Path` | Grava o contrato em `<caminho>.contract.json`. |
| `read_sidecar` | função | `(path) -> Contract \| None` | Lê o contrato do sidecar, ou `None` se não existe. |
| `write_table` | função | `(df, path, contract=None) -> Path` | Grava um DataFrame (ou dict de colunas) como CSV, Parquet, HDF5 ou JSON, junto com o sidecar do contrato. |
| `read_table` | função | `(path)` | Lê uma tabela e seu contrato, preferindo o sidecar aos metadados embutidos. |
| `convert` | função | `(src, dst) -> Path` | Copia uma tabela para outro formato (pela extensão) levando o contrato junto. |

#### `pinneapple_data.data_health`

Relatório de saúde de dados de engenharia (sensores, ensaios, planta): lê tabelas em vários formatos e verifica unidades, lacunas, eixo de tempo, valores fisicamente impossíveis, códigos de erro, unidade incoerente e outros problemas, com sugestões.

<small>`pinneapple_data/data_health.py` · 998 linhas · testes: 37</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Issue` | classe | `(category, severity, title, detail, column, affected_fraction, count, spans, suggestion, …)` | Um problema encontrado: categoria, severidade, título, detalhe, coluna, fração afetada, trechos, sugestão e evidência. |
| `LoadInfo` | classe | `(format, rows, columns, delimiter, decimal, encoding, sheet, sheets, units_from_file, tru…)` | Como a tabela foi lida: formato, linhas, colunas, delimitador, separador decimal, codificação, planilha, unidades do arquivo e notas. |
| `load_table` | função | `(data, filename, sheet=None, max_rows=1000000) -> Tuple[pd.DataFrame, LoadInfo]` | Lê uma tabela enviada (CSV/TSV com detecção de delimitador e vírgula decimal, Excel, Parquet, JSON, HDF5) e devolve o DataFrame e um `LoadInfo`. |
| `detect_time_column` | função | `(df) -> Tuple[Optional[str], Optional[pd.Series], str]` | Procura a coluna de tempo pelo nome e pelo conteúdo e devolve o nome, a série convertida e uma explicação. |
| `analyze` | função | `(df, units=None, time_column=None, n_bins=160, series_points=600) -> Dict[str, Any]` | Gera o relatório de saúde de um DataFrame (pronto para JSON): unidades, completude, eixo de tempo, validade física, sinais travados e outros problemas; levanta `ValueError` para tabela vazia. |
| `example_chiller_plant` | função | `(seed=7, days=7, inject=True) -> Tuple[pd.DataFrame, Dict[str, Any]]` | Gera uma semana de dados de minuto em minuto de uma planta de água gelada com falhas injetadas e devolve a tabela e o gabarito das falhas. |

#### `pinneapple_data.dataloaders`

Monta `DataLoader` do PyTorch a partir de um shard UPD (Zarr + JSON) ou de um `PhysicalSample` em grade; amostras em malha ainda não são suportadas. [sem teste]

<small>`pinneapple_data/dataloaders.py` · 353 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DataLoaderSpec` | classe | `(batch_size, num_workers, shuffle, pin_memory, drop_last)` | Parâmetros do `DataLoader`: tamanho de batch, workers, shuffle, pin_memory, entre outros. |
| `build_upd_dataloader` | função | `(*, zarr_path, meta_path, mapping, sampling, loader=None, device='cpu', dtype=torch.float…)` | Cria um `DataLoader` sobre um shard UPD, envolvendo-o num `Dataset` do Torch conforme o `DataLoaderSpec`. |
| `build_physical_sample_dataloader` | função | `(sample, *, mapping, sampling, loader=None, device='cpu', dtype=torch.float32, length=100…)` | Cria um `DataLoader` a partir de um `PhysicalSample` em grade (xarray); levanta `ValueError` para casos não suportados, como malha. |

#### `pinneapple_data.datasets`

Datasets prontos (física, geometria, séries temporais) num registro único, com `load_dataset`, `list_datasets`, `dataset_info` e `dataset_ids`; os módulos `real_world` e `external_sources` são opcionais e carregados em `try/except`.

<small>`pinneapple_data/datasets/__init__.py` · 152 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `load_dataset` | função | `(dataset_id, **kwargs) -> Dict[str, np.ndarray]` | Carrega um dataset registrado pelo id e devolve um dict de arrays NumPy; kwargs vão para o gerador. |
| `list_datasets` | função | `(category=None, verbose=True) -> List[DatasetInfo]` | Lista (e imprime, se `verbose`) os datasets registrados, opcionalmente filtrando por categoria. |
| `dataset_info` | função | `(dataset_id) -> DatasetInfo` | Devolve os metadados (`DatasetInfo`) de um dataset. |
| `dataset_ids` | função | `(category=None) -> List[str]` | Devolve os ids registrados, opcionalmente por categoria. |

#### `pinneapple_data.datasets.deepmind_mgn`

Datasets `meshgraphnets` da DeepMind (Pfaff et al., ICLR 2021): download parcial de `meta.json` + prefixo do TFRecord, leitura de trajetórias e conversão de `cylinder_flow` para tensores; exige `tfrecord`.

<small>`pinneapple_data/datasets/deepmind_mgn.py` · 116 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `download_prefix` | função | `(dataset, split, dest, nbytes) -> Path` | Baixa `meta.json` e os primeiros `nbytes` do TFRecord de um split; como o formato é sequencial, o prefixo contém trajetórias inteiras. |
| `read_trajectories` | função | `(data_dir, split, max_traj) -> list[dict[str, np.ndarray]]` | Lê até `max_traj` trajetórias do TFRecord como dicts de arrays, com formas vindas de `meta.json`; descarta o último registro truncado. |
| `cylinder_flow_to_tensors` | função | `(traj, node_type_map=CYLINDER_TYPE_MAP) -> tuple[MeshGraph, torch.Tensor, torch.Tensor]` | Converte uma trajetória `cylinder_flow` em `(MeshGraph, velocidade (T,N,2), pressão (T,N,1))`, agrupando tipos de nó. |

#### `pinneapple_data.datasets.download`

Baixa uma URL uma vez para um cache local, com novas tentativas, só com a biblioteca padrão; escreve em arquivo temporário e renomeia ao terminar.

<small>`pinneapple_data/datasets/download.py` · 85 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `cache_path` | função | `(url, cache_dir) -> Path` | Caminho do arquivo de cache de uma URL: `<cache_dir>/<md5(url)><ext>`. |
| `fetch_cached` | função | `(url, cache_dir, *, tries=4, timeout=60, backoff=2.0) -> Path` | Devolve uma cópia local da URL, baixando só se ainda não estiver no cache. |

#### `pinneapple_data.datasets.external_sources`

Clientes de fontes externas (Materials Project, NOMAD, NASA, NOAA, Copernicus, Brazil Data Cube, CPTEC/INPE, ONS, NREL, DesignSafe, OpenFOAM, SU2, NASA POWER, Open-Meteo) que registram datasets; vários caem em dados sintéticos quando a fonte não está disponível.

<small>`pinneapple_data/datasets/external_sources.py` · 1.168 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MaterialsProjectClient` | classe | `(api_key=None) -> None` | Cliente da API do Materials Project (requer `mp-api` e chave `MP_API_KEY`); sem acesso, devolve dados sintéticos marcados com `_synthetic`. |
| `NOMADClient` | classe | `()` | Cliente da API REST pública do NOMAD (cálculos ab initio/DFT). |
| `NASAClient` | classe | `()` | Cliente do portal de dados abertos da NASA: exoplanetas e temperatura GISS. |
| `NOAAClient` | classe | `()` | Cliente NOAA NCEI/ERDDAP: temperatura da superfície do mar e anomalia de temperatura, com reserva sintética. |
| `CopernicusClient` | classe | `()` | Cliente Copernicus (CDS): tenta `cdsapi` e, sem ele, gera dados sintéticos tipo ERA5 (`fetch_era5_synthetic`). |
| `BrazilDataCubeClient` | classe | `()` | Cliente da API STAC do Brazil Data Cube: lista coleções; a série de NDVI é sintética. |
| `CPTECClient` | classe | `()` | Cliente CPTEC/INPE; a série de precipitação devolvida é sintética. |
| `ONSClient` | classe | `()` | Cliente de dados abertos do ONS; a carga elétrica devolvida é sintética, com sazonalidade diária e semanal. |
| `NRELClient` | classe | `(api_key=None) -> None` | Cliente NREL para recurso solar e eólico (dados reais exigem `NREL_API_KEY`; sem ela, sintéticos). |
| `DesignSafeClient` | classe | `()` | Cliente DesignSafe-CI para registros de terremoto/estrutural. |
| `OpenFOAMClient` | classe | `(openfoam_data_dir=None) -> None` | Devolve dados sintéticos com as curvas típicas de tutoriais OpenFOAM (arrasto de cilindro, perfil de canal, cavidade) ou lê casos reais se `openfoam_data_dir` for dado. |
| `SU2Client` | classe | `(su2_results_dir=None) -> None` | Devolve dados aerodinâmicos representativos de tutoriais SU2 (polar NACA 0012, Cp do RAE 2822) ou lê resultados reais de `su2_results_dir`. |
| `NASAPowerClient` | classe | `()` | Cliente da API diária NASA POWER (meteorologia e recurso solar, sem chave); erros de rede são levantados; converte para arrays e `PhysicalSample`. |
| `OpenMeteoClient` | classe | `()` | Cliente da API histórica Open-Meteo (reanálise ERA5 horária, sem chave); erros de rede são levantados. |

#### `pinneapple_data.datasets.geometry`

Datasets de geometria gerados analiticamente (pontos de contorno e interior): `naca0012`, `cylinder_2d`, `lshape_2d`, `channel_with_obstacle`. [sem teste direto]

<small>`pinneapple_data/datasets/geometry.py` · 269 linhas · **sem teste direto**</small>

#### `pinneapple_data.datasets.physics`

Datasets de física gerados analiticamente ou por método das linhas com SciPy (Burgers, calor, Poisson, onda, Kovasznay, Allen-Cahn, Helmholtz) e um solver pseudo-espectral de Burgers periódico para pares de operador.

<small>`pinneapple_data/datasets/physics.py` · 450 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `burgers_periodic_solve` | função | `(u0, T, nu, *, length=1.0, refine=None, cfl=0.5) -> np.ndarray` | Resolve Burgers viscoso periódico por pseudo-espectral de Fourier com regra 2/3 e Runge-Kutta de 4ª ordem com fator integrante. |
| `burgers_periodic_pairs` | função | `(n, nx=128, T=1.0, nu=0.01, seed=0, *, length=1.0, refine=None)` | Gera `n` pares aleatórios `(u(x,0), u(x,T))` de Burgers periódico para aprendizado de operadores, com condições iniciais somas de senos. |

#### `pinneapple_data.datasets.real_world`

Datasets reais carregados da web ou de bibliotecas opcionais (sunspots, CO2, ETT, Jena, scikit-learn, seaborn, statsmodels...), com reserva sintética quando a fonte falha; alguns itens são sintéticos por natureza. [sem teste]

<small>`pinneapple_data/datasets/real_world.py` · 1.665 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `load_real_dataset` | função | `(dataset_id) -> Dict[str, np.ndarray]` | Carrega um dataset real pelo id, baixando na primeira chamada e guardando em memória no processo. |

#### `pinneapple_data.datasets.registry`

Registro central (`DatasetRegistry`) dos datasets prontos, com metadados `DatasetInfo`. [sem teste direto]

<small>`pinneapple_data/datasets/registry.py` · 65 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DatasetInfo` | classe | `(id, name, category, description, fields, tags, license, reference)` | Metadados de um dataset: id, nome, categoria, descrição, campos, tags, licença e referência. |
| `DatasetRegistry` | classe | `(_store)` | Registro em nível de classe: `register`, `load`, `get_info`, `list_all` e `ids`. |

#### `pinneapple_data.datasets.timeseries`

Datasets de séries temporais gerados por EDOs ou funções sintéticas (Lorenz-63, massa-mola, Van der Pol, Rössler, pêndulo duplo, Lorenz-96, seno com ruído). [sem teste direto]

<small>`pinneapple_data/datasets/timeseries.py` · 355 linhas · **sem teste direto**</small>

#### `pinneapple_data.device`

Fixa em memória (pin) e move entre CPU e GPU todos os tensores de `fields` e `coords` de uma amostra tipo `PhysicalSample` (objeto ou dict). [sem teste]

<small>`pinneapple_data/device.py` · 148 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `pin_sample` | função | `(sample) -> Any` | Fixa em memória paginável (pinned) todos os tensores de `fields` e `coords` de uma amostra. |
| `to_device_sample` | função | `(sample, device, *, dtype=None, non_blocking=True) -> Any` | Move todos os tensores de `fields` e `coords` de uma amostra para o dispositivo dado. |

#### `pinneapple_data.doe`

Design de experimentos para campanhas de simulação: desenhos, cobertura, redundância, prioridade e plano de campanha, sobre os contratos `ParameterSpace`/`StudyResult`/`DatasetSpec` e `scipy.stats.qmc`.

<small>`pinneapple_data/doe/__init__.py` · 20 linhas · testes: 21</small>

#### `pinneapple_data.doe.campaign`

Planeja uma campanha de simulação: quais casos rodar, em que ordem, e o que acrescentam ao que já rodou; gera `StudyResult` e esqueleto de `DatasetSpec`. [sem teste direto]

<small>`pinneapple_data/doe/campaign.py` · 174 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CampaignPlan` | classe | `(space, design, cases, unit, coverage_planned, coverage_existing, coverage_combined, redu…)` | Plano de campanha: casos com prioridade, cobertura antes/depois, redundâncias removidas; exporta resumo, checagens, `StudyResult` e esqueleto de `DatasetSpec`. |
| `plan_campaign` | função | `(space, n=None, method='lhs', *, existing=None, existing_id_column=None, outputs=(), seed…)` | Desenha `n` casos, descarta os que repetem execuções existentes, ordena por prioridade e mede a cobertura do plano, do existente e de ambos. |

#### `pinneapple_data.doe.coverage`

Mede quão bem um conjunto de casos cobre o espaço de parâmetros (no cubo unitário), quais casos planejados repetem outros e em que ordem rodá-los.

<small>`pinneapple_data/doe/coverage.py` · 119 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `coverage` | função | `(unit, names=None, *, n_probe=2 ** 14, seed=0, bins_2d=None, levels=None) -> dict[str, An…)` | Métricas de cobertura de pontos em [0,1]^d: discrepância L2 centrada, distância mínima, distância média ao vizinho e outras. |
| `redundancy` | função | `(planned_unit, existing_unit=None, *, tol=None, existing_ids=None) -> dict[str, Any]` | Marca casos planejados que quase repetem um caso existente ou outro planejado, com o vizinho mais próximo e a distância. |
| `prioritize` | função | `(planned_unit, existing_unit=None) -> dict[str, Any]` | Ordem de execução por maximin guloso a partir do que já existe, para que qualquer prefixo da lista cubra o espaço de forma espalhada. |

#### `pinneapple_data.doe.designs`

Desenhos de experimentos sobre um `ParameterSpace`: `lhs`, `maximin`, `sobol`, `halton`, `random`, fatorial completo e fracionário, entre outros, filtrados por limites e restrições. [sem teste direto]

<small>`pinneapple_data/doe/designs.py` · 283 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Design` | classe | `(cases, unit, method, info)` | Casos de um desenho: valores (DataFrame com todos os parâmetros), coordenadas unitárias e como foram gerados. |
| `design` | função | `(space, n, method='lhs', *, seed=0, existing=None, outputs=(), levels=3, n_restarts=20, f…)` | Gera `n` casos do espaço pelo método escolhido, considerando casos já existentes quando dados. |
| `fractional_factorial_unit` | função | `(k, n) -> tuple[np.ndarray, dict]` | Desenho fatorial fracionário de dois níveis 2^(k-p) com pelo menos `n` execuções, em coordenadas unitárias. |
| `greedy_maximin` | função | `(points, n, fixed=None, weights_fixed=None) -> np.ndarray` | Escolhe `n` pontos um a um, cada um o mais distante dos já escolhidos e dos fixos (com pesos opcionais). |
| `min_distance` | função | `(unit) -> float` | Menor distância entre dois pontos de um conjunto (NaN se houver menos de dois). |

#### `pinneapple_data.expressions`

Avaliação segura de expressões de restrição (como `"chord - 2 * thickness >= 0.01"`): interpreta a árvore nó a nó, sem `eval`, só com aritmética, comparações, lógica, nomes de colunas e algumas funções matemáticas.

<small>`pinneapple_data/expressions.py` · 131 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ExpressionError` | classe | `()` | Erro para expressão com construção proibida ou nome que não é parâmetro. |
| `names_in` | função | `(expression) -> set[str]` | Nomes de parâmetros usados numa expressão (sem os nomes de função). |
| `evaluate` | função | `(expression, values, functions=None) -> np.ndarray` | Avalia a expressão com valores (escalares ou colunas) e devolve um array booleano ou numérico. |
| `check` | função | `(expression, names, functions=(), strings=False) -> None` | Levanta `ExpressionError` se a expressão sair da gramática permitida, dos nomes dados ou das funções permitidas. |
| `names_in_tree` | função | `(tree) -> set[str]` | Nomes de variáveis numa árvore AST, excluindo os nomes chamados como função. |

#### `pinneapple_data.formfill`

Preenche formulários de engenharia a partir dos documentos que têm os dados: lê datasheets e especificações, acha cada item com a fonte, relata conflitos e lacunas e preenche o PDF; modelo embutido ASME U-DR-1 e outros datasheets.

<small>`pinneapple_data/formfill/__init__.py` · 28 linhas · testes: 44</small>

#### `pinneapple_data.formfill.extract`

Lê documentos (texto e tabelas via pdfplumber, OCR opcional), extrai candidatos pelos padrões do `FormSpec` e compila uma decisão por item, guardando documento, página e trecho de origem. [sem teste direto]

<small>`pinneapple_data/formfill/extract.py` · 698 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Document` | classe | `(name, pages, tables, role, has_text, ocr_pages, ocr_conf, unread_pages)` | Documento lido: nome, texto por página, tabelas, papel, páginas por OCR e confiança, páginas não lidas. |
| `Candidate` | classe | `(key, value, raw, doc, page, snippet, method, confidence, unit, si, gauge, ocr)` | Valor candidato para um item: valor, texto bruto, documento, página, trecho, método, confiança, unidade e valor em SI. |
| `Decision` | classe | `(key, label, section, status, value, display, chosen, candidates, required, note)` | Decisão para um item: status, valor escolhido, candidatos, se é obrigatório e nota. |
| `Compilation` | classe | `(decisions, tables, documents, spec)` | Resultado da compilação: decisões, tabelas, documentos; dá lacunas, conflitos, respostas, resumo e o registro para outros formulários. |
| `read_document` | função | `(source, name='', role='', ocr='auto', max_ocr_pages=30) -> Document` | Lê um PDF (caminho ou bytes) ou texto puro; com `ocr='auto'` lê páginas escaneadas com Tesseract quando disponível. |
| `parse_value` | função | `(f, text) -> Optional[Dict[str, Any]]` | Interpreta um texto como resposta a um campo (quantidade, número, booleano, escolha...) ou devolve `None`. |
| `extract_rules` | função | `(doc, spec) -> List[Candidate]` | Gera candidatos para cada item do spec cujo rótulo aparece no documento (tabelas primeiro, depois linhas de texto), mais as linhas das tabelas do spec. |
| `check_answer` | função | `(f, value) -> Any` | Valida a resposta de um engenheiro para um campo; `ValueError` cita as opções aceitas. |
| `compile` | função | `(docs, candidates, spec, overrides=None) -> Compilation` | Uma decisão por item: documentos anteriores na lista vencem conflitos (que ainda são relatados) e respostas do engenheiro vencem tudo. |

#### `pinneapple_data.formfill.fill`

Escreve uma compilação na cópia do usuário de um PDF preenchível (via `pypdf`); itens em `spec.left_blank` (data, assinatura) nunca são preenchidos. [sem teste direto]

<small>`pinneapple_data/formfill/fill.py` · 177 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `pdf_widgets` | função | `(pdf) -> List[Dict[str, Any]]` | Lista os widgets de formulário de um PDF: nome qualificado, tipo, página, tooltip, estados de botão e grupo de rádio. |
| `form_values` | função | `(comp, spec=None) -> Tuple[Dict[str, str], Dict[str, str]]` | Calcula `(campos de texto, estados de botão)` para o PDF a partir das decisões e linhas de tabela. |
| `fill_pdf` | função | `(blank_pdf, comp, spec=None, extra_text=None) -> bytes` | Devolve o formulário preenchido como bytes de PDF. |
| `read_filled` | função | `(pdf) -> Dict[str, Any]` | Lê nome do campo → valor de um formulário preenchido, para checagem e ida e volta. |

#### `pinneapple_data.formfill.llm`

Extração opcional com um LLM local servido pelo Ollama: só aceita respostas cuja citação existe no texto do documento e contém o valor; o resto é rejeitado e listado. [sem teste direto]

<small>`pinneapple_data/formfill/llm.py` · 230 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OllamaError` | classe | `()` | Erro de comunicação ou resposta inválida do servidor Ollama. |
| `OllamaClient` | classe | `(url=None, model=None, timeout=600.0, num_ctx=16384)` | Cliente mínimo da API HTTP do Ollama (`/api/chat`, `/api/tags`), com chat em saída JSON estruturada. |
| `document_text` | função | `(doc, pages=None) -> str` | Texto do documento com marcação de página e tabelas como linhas `célula \| célula`. |
| `verify_quote` | função | `(doc, quote, page) -> Optional[int]` | Página onde a citação aparece no texto do documento (preferindo a página declarada), ou `None`. |
| `extract_with_llm` | função | `(doc, spec, client=None, max_chars=24000) -> Tuple[List[Candidate], List[Dict[str, Any]]]` | Extrai candidatos de um documento com o LLM e devolve os aceitos e as respostas rejeitadas com o motivo. |

#### `pinneapple_data.formfill.ocr`

OCR de páginas escaneadas com Tesseract: renderiza, corrige inclinação, apaga linhas de grade de tabela antes da leitura e recoloca cada palavra na célula; exige o programa `tesseract`. [sem teste direto]

<small>`pinneapple_data/formfill/ocr.py` · 241 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OcrPage` | classe | `(lines, tables, confidence, word_conf, angle)` | Página lida por OCR: linhas de texto, tabelas, confiança média e por palavra e ângulo de correção. |
| `ocr_available` | função | `() -> bool` | Diz se o executável do Tesseract está disponível. |
| `ocr_image` | função | `(img) -> OcrPage` | Faz OCR de uma imagem de página (PIL). |
| `ocr_pdf_page` | função | `(pdf_bytes, index, dpi=DPI) -> OcrPage` | Faz OCR da página `index` (base 0) de um PDF. |

#### `pinneapple_data.formfill.pipeline`

Uma chamada dos documentos ao formulário compilado (regras + LLM opcional) e utilitários de front-end: validar respostas do usuário e nome do arquivo de saída. [sem teste direto]

<small>`pinneapple_data/formfill/pipeline.py` · 66 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `compile_documents` | função | `(docs, spec, client=None, overrides=None, max_ocr_pages=None) -> Tuple[Compilation, List[…)` | Lê cada documento `(nome, dados, papel)`, extrai candidatos por regras (e LLM, se houver) e compila; falha do LLM não derruba a chamada. |
| `check_overrides` | função | `(spec, raw) -> Dict[str, Any]` | Valida as respostas do usuário contra o spec; chave desconhecida ou valor inválido levanta `ValueError`. |
| `output_filename` | função | `(comp, prefix='', day=None) -> str` | Nome seguro do arquivo preenchido: `<spec id>_<valor da chave>_<AAAAMMDD>.pdf`. |

#### `pinneapple_data.formfill.report`

Gera o datasheet compilado em PDF com `reportlab` (itens com valor e fonte, pendências primeiro), marcado como rascunho para revisão de engenharia. [sem teste direto]

<small>`pinneapple_data/formfill/report.py` · 166 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `render_datasheet` | função | `(comp, spec=None, title=None) -> bytes` | Renderiza uma `Compilation` como PDF de datasheet; exige `reportlab`. |

#### `pinneapple_data.formfill.spec`

Define o que um formulário pede (`FormSpec`, `Field`, `TableSpec`): rótulos, padrões de texto, tipo de campo, campo do PDF; specs vêm de modelos embutidos, JSON ou de qualquer PDF preenchível. [sem teste direto]

<small>`pinneapple_data/formfill/spec.py` · 259 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Field` | classe | `(key, label, section, kind, pdf, states, patterns, quantity, required, source_hint, optio…)` | Um item do formulário: chave, rótulo, seção, tipo (`text`, `quantity`, `number`, `bool`, `choice`, `multi`), campo do PDF, padrões e opções. |
| `TableSpec` | classe | `(key, label, columns, header, combine, defaults, pdf_rows, description)` | Item de linhas repetidas (lista de bocais, lista de linhas) achado como tabela cujo cabeçalho casa com os padrões e preenchido linha a linha no PDF. |
| `FormSpec` | classe | `(id, title, fields, sections, tables, description, notes_field, left_blank, filename_key,…)` | Especificação completa de um formulário: campos, seções, tabelas e metadados; salva/carrega JSON e pode ser gerada de um PDF preenchível (`from_pdf`). |

#### `pinneapple_data.formfill.specs`

Registro dos modelos de formulário embutidos (`asme_u-dr-1`, `psv`, `shell_tube`, `tank`, `pump`), carregados sob demanda. [sem teste direto]

<small>`pinneapple_data/formfill/specs/__init__.py` · 41 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `get_spec` | função | `(spec_id) -> FormSpec` | Devolve o `FormSpec` de um modelo embutido pelo id; `KeyError` lista os ids conhecidos. |
| `list_specs` | função | `() -> List[Dict[str, object]]` | Lista os modelos embutidos com título, norma, resumo, entradas, saída, número de itens, obrigatórios e tabelas. |

#### `pinneapple_data.formfill.specs.asme_udr1`

Modelo embutido do ASME BPVC Seção VIII Div. 1 Form U-DR-1: campos, padrões de texto e mapeamento para os campos do PDF preenchível (revisão 07/25). [sem teste direto]

<small>`pinneapple_data/formfill/specs/asme_udr1.py` · 326 linhas · **sem teste direto**</small>

#### `pinneapple_data.formfill.specs.psv`

Modelo embutido de datasheet de válvula de alívio de pressão, com os dados pedidos por API 520 Parte I e API 526; a saída é um datasheet gerado pela ferramenta. [sem teste direto]

<small>`pinneapple_data/formfill/specs/psv.py` · 115 linhas · **sem teste direto**</small>

#### `pinneapple_data.formfill.specs.pump`

Modelo embutido de datasheet de bomba centrífuga com os dados que a API 610 pede ao comprador. [sem teste direto]

<small>`pinneapple_data/formfill/specs/pump.py` · 110 linhas · **sem teste direto**</small>

#### `pinneapple_data.formfill.specs.shell_tube`

Modelo embutido de datasheet de trocador casco-tubo com os dados de uma folha TEMA e da API 660, lendo colunas de lado casco e lado tubo. [sem teste direto]

<small>`pinneapple_data/formfill/specs/shell_tube.py` · 110 linhas · **sem teste direto**</small>

#### `pinneapple_data.formfill.specs.tank`

Modelo embutido de datasheet de tanque de armazenamento atmosférico com os dados que a API 650 pede (Anexo L). [sem teste direto]

<small>`pinneapple_data/formfill/specs/tank.py` · 117 linhas · **sem teste direto**</small>

#### `pinneapple_data.geometry_features`

Features de geometria a partir de malha triangular: nuvem de pontos na superfície e grade grosseira de distância assinada (SDF), via `trimesh`. [sem teste]

<small>`pinneapple_data/geometry_features.py` · 86 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `sample_point_cloud` | função | `(vertices, faces, n_samples=2048) -> np.ndarray` | Amostra pontos uniformes (ponderados por área) na superfície da malha, limitando o número em malhas muito grosseiras. |
| `compute_sdf_grid` | função | `(vertices, faces, grid_resolution=16, bounds=None) -> Dict[str, Any]` | Avalia a distância assinada numa grade regular sobre a caixa da malha (ou limites dados). |
| `featurize_mesh` | função | `(vertices, faces, n_point_cloud_samples=2048, sdf_grid_resolution=16) -> Dict[str, Any]` | Nuvem de pontos + grade SDF numa chamada, relatando erros por etapa em vez de levantar. |

#### `pinneapple_data.grf_sampler`

Amostrador de campos aleatórios gaussianos (GRF) para funções de entrada de benchmarks de operador (como no DeepONet): 1D com uma única fatoração de Cholesky por lote e spline cúbica, 2D por síntese espectral com FFT. [sem teste]

<small>`pinneapple_data/grf_sampler.py` · 177 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GRFDraw` | classe | `(x_grid, u_grid, spline)` | Uma função amostrada u(x) em [x_min, x_max], consultável em qualquer ponto; `sensors` avalia nos sensores. |
| `GRFDraw2D` | classe | `(x_grid, y_grid, u_grid, interp)` | Um campo 2D amostrado u(x,y) em [0,1]×[0,1]; `sensors` avalia nos pontos dados. |
| `sample_grf_batch` | função | `(n_draws, x_min=0.0, x_max=1.0, n_grid=1000, length_scale=0.2, jitter=1e-10, seed=None) -…)` | Sorteia vários caminhos GRF 1D de média zero com covariância RBF, compartilhando uma fatoração de Cholesky numa grade fina. |
| `sample_grf_paths` | função | `(n_draws, x_min=0.0, x_max=1.0, n_grid=1000, length_scale=0.2, jitter=1e-10, seed=None) -…)` | Apelido de `sample_grf_batch`. |
| `sample_grf_2d_batch` | função | `(n_draws, nx=64, ny=64, length_scale=0.2, seed=None) -> List[GRFDraw2D]` | Sorteia vários campos GRF 2D de média zero e variância unitária por síntese espectral com FFT. |

#### `pinneapple_data.lineage`

Linhagem de modelos de engenharia (digital thread): artefatos e derivações, checagens, perguntas, exportação W3C PROV e detecção automática a partir dos arquivos de um projeto.

<small>`pinneapple_data/lineage/__init__.py` · 7 linhas · testes: 13</small>

#### `pinneapple_data.lineage.detect`

Detecta a linhagem de um projeto pelos arquivos: casos OpenFOAM, Gmsh `.geo`→`.msh`, decks e resultados CalculiX, datasets exportados (ligados por sha256), relatórios e model cards, mais um `lineage.json` declarado. [sem teste direto]

<small>`pinneapple_data/lineage/detect.py` · 304 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `detect_project` | função | `(fs, project='') -> Lineage` | Monta um `Lineage` a partir de um dict caminho → bytes, ligando artefatos pelos arquivos e hashes. |

#### `pinneapple_data.lineage.graph`

Grafo de linhagem: artefatos (CAD, malha, simulação, dataset, modelo, predição...) ligados por arestas 'derivado de', com checagens (referências quebradas, ciclos, arquivos alterados, derivações velhas) e exportação PROV-JSON e Markdown. [sem teste direto]

<small>`pinneapple_data/lineage/graph.py` · 365 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Artifact` | classe | `(id, kind, name, file, version, software, parameters, timestamp, hash, owner, origin, inp…)` | Um artefato da linhagem: id, tipo, arquivo, versão, software, parâmetros, hash, dono, entradas; mede completude e verifica o hash. |
| `Lineage` | classe | `(artifacts=(), project='')` | Grafo de artefatos: adiciona, lê/escreve JSON, navega a montante e a jusante, roda checagens e exporta relatório, PROV e Markdown. |
| `sha256` | função | `(b) -> str` | SHA-256 em hexadecimal de bytes. |

#### `pinneapple_data.noether_dataset`

Ponte de dados entre o framework Noether e o PINNeAPPle: usa datasets do Noether (como DrivAerML) como `Dataset` do PyTorch e converte listas de `PhysicalSample` para o formato `{campo: tensor}` do Noether.

<small>`pinneapple_data/noether_dataset.py` · 216 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NoetherDatasetBridge` | classe | `(noether_dataset, input_fields=None, target_fields=None, transform=None, device=None) -> …)` | Envolve um dataset do Noether (ou qualquer dataset que devolva `{campo: tensor}`) como `torch.utils.data.Dataset` comum, sem Hydra. |
| `PhysicalSampleDataset` | classe | `(samples, input_fields=None, output_fields=None, include_coords=False, transform=None) ->…)` | Dataset PyTorch em que cada `PhysicalSample` vira um item `{campo: tensor}`, com campos e coordenadas expostos. |
| `split_physical_samples` | função | `(samples, val_fraction=0.1, seed=42, **dataset_kwargs)` | Divide uma lista de `PhysicalSample` em datasets de treino e validação com fração e semente. |

#### `pinneapple_data.parameter_sampling`

Amostragem de espaços de parâmetros (constantes físicas, magnitudes de contorno, dimensões) por LHS, Sobol, Halton ou aleatória via `scipy.stats.qmc`, e conjunto tipo Saltelli para sensibilidade de Sobol sem SALib.

<small>`pinneapple_data/parameter_sampling.py` · 118 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `sample_parameters` | função | `(param_ranges, n_samples, method='lhs', seed=0) -> list[dict[str, float]]` | Sorteia `n_samples` pontos de um espaço `{nome: (min, max)}` pelo método escolhido (`lhs` por padrão) e devolve uma lista de dicts. |
| `saltelli_perturbation_sweep` | função | `(nominal, param_ranges, n_base=8, seed=0) -> list[dict[str, float]]` | Monta o conjunto de amostras no estilo Saltelli (matrizes A, B e trocas de coluna) para índices de Sobol de primeira ordem e total. |

#### `pinneapple_data.physical_sample`

Dataclass `PhysicalSample`, o contêiner padrão (alinhado ao UPD) de dados físicos: estado (xarray ou dict), geometria, esquema, domínio, proveniência e extras, com ponte para dict de treino.

<small>`pinneapple_data/physical_sample.py` · 241 linhas · testes: 28</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicalSample` | classe | `(state, geometry, schema, domain, provenance, extras)` | Amostra física unificada: estado, geometria, esquema de equações/condições, domínio e proveniência; diz o tipo de domínio, lista variáveis, resume e converte para dict de treino (`to_train_dict`). |

#### `pinneapple_data.physical_units`

Unidades e grandezas físicas só com a biblioteca padrão: interpreta unidades de cabeçalhos, converte para SI, infere a grandeza de uma coluna e guarda limites físicos rígidos e típicos por grandeza.

<small>`pinneapple_data/physical_units.py` · 436 linhas · testes: 52</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Unit` | classe | `(symbol, factor, dim, offset, gauge)` | Uma unidade com fator e deslocamento para SI, vetor de dimensão e indicação de pressão manométrica; converte para/de SI. |
| `UnitError` | classe | `()` | Erro de unidade desconhecida ou mal formada. |
| `Quantity` | classe | `(name, si_unit, dim, keywords, hard, typical, display)` | Uma grandeza física: nome, unidade SI, dimensão, palavras-chave, limites rígidos, faixa típica e unidade de exibição. |
| `QuantityGuess` | classe | `(quantity, confidence, reason, alternatives, is_difference)` | Palpite de grandeza para uma coluna: grandeza, confiança, motivo, alternativas e se é diferença. |
| `parse_unit` | função | `(text) -> Unit` | Interpreta uma expressão de unidade (`°C`, `kg/m^3`, `W/(m·K)`, `psig`...) como `Unit`; levanta `UnitError` para token desconhecido. |
| `try_parse_unit` | função | `(text) -> Optional[Unit]` | Como `parse_unit`, mas devolve `None` em vez de levantar erro. |
| `convert` | função | `(values, src, dst, difference=False)` | Converte valores (escalar ou array) de uma unidade para outra, com opção para diferenças (sem deslocamento). |
| `dim_str` | função | `(dim) -> str` | Texto legível de um vetor de dimensão (por exemplo `kg·m^-3`, ou `dimensionless`). |
| `split_header` | função | `(header) -> Tuple[str, Optional[str]]` | Separa o cabeçalho de coluna em nome e texto de unidade (`[°C]`, `(bar)`, `, m3/h`, `in kW`, sufixos como `_degC`). |
| `infer_quantity` | função | `(name, unit=None) -> QuantityGuess` | Infere a grandeza física de uma coluna combinando a dimensão da unidade com palavras-chave do nome. |
| `describe_unit` | função | `(u) -> Dict[str, object]` | Descrição em dict de uma unidade: símbolo, fator e deslocamento SI, dimensão e se é manométrica. |
| `si_unit_for` | função | `(quantity) -> str` | Unidade SI de uma grandeza registrada em `QUANTITIES`. |

#### `pinneapple_data.physics_case`

`PhysicsCase`: ponte mínima que referencia juntos geometria, `ProblemSpec`, escolha de solver e resultados de um caso físico, e compara resultados com um benchmark sem emitir veredito.

<small>`pinneapple_data/physics_case.py` · 405 linhas · testes: 47</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BenchmarkComparison` | classe | `(benchmark, x_vars, y_vars, n_reference_points, n_compared_points, rmse, relative_l2_erro…)` | Comparação numérica (RMSE, erro relativo etc.) produzida por `validate_against_benchmark`; o critério de aprovação fica em outro pacote. |
| `PhysicsCase` | classe | `(name, geometry, geometry_kind, physics, solver, solver_config, results, reference_benchm…)` | Um caso físico com campos opcionais (geometria, spec, solver, resultados, benchmark); valida contra benchmark e converte para/de dict. |

#### `pinneapple_data.prefetch_adaptive`

`IterableDataset` que lê amostras de um store Zarr UPD com cache e prefetch adaptativo: uma thread produtora ajusta quantas amostras mantém na fila. [sem teste]

<small>`pinneapple_data/prefetch_adaptive.py` · 281 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AdaptivePrefetchConfig` | classe | `(queue_max, min_target_fill, max_target_fill, target_fill_init, control_period_s, increas…)` | Configuração do prefetch adaptativo: capacidade máxima da fila e faixa e passos do alvo de preenchimento. |
| `AdaptivePrefetchZarrUPDIterable` | classe | `(root, *, fields=None, coords=None, dtype=None, start=0, end=None, stride=1, sample_ctor=…)` | Itera amostras de um store Zarr UPD em cache com uma fila limitada abastecida por thread que ajusta o alvo conforme o consumo. |

#### `pinneapple_data.preflight`

Pré-voo de simulação: checa um caso OpenFOAM ou um deck CalculiX/Abaqus antes de gastar horas de solver e devolve status PASS/WARNING/FAIL com achados que apontam arquivo e linha.

<small>`pinneapple_data/preflight/__init__.py` · 74 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `preflight` | função | `(fs) -> Dict[str, Any]` | Detecta se o upload é caso OpenFOAM ou deck `.inp` e roda as regras correspondentes, devolvendo o relatório. |
| `is_result_file` | função | `(path) -> bool` | Diz se um arquivo é resultado de solver (diretório de tempo diferente de `0`, resultado CalculiX ou VTK) e não entrada. |
| `drop_results` | função | `(fs) -> Dict[str, bytes]` | Remove do upload os arquivos de resultado de solver. |
| `attach_snippets` | função | `(report, fs, context=4, width=160) -> None` | Acrescenta a cada achado com arquivo e linha as linhas de código ao redor (`snippet`). |

#### `pinneapple_data.preflight.calculix`

Regras de pré-voo para decks CalculiX/Abaqus planos: malha, seções e materiais, propriedades por procedimento, consistência de unidades, apoios e cargas, condições iniciais e passos; regras FAIL confirmadas rodando CalculiX em decks quebrados. [sem teste direto]

<small>`pinneapple_data/preflight/calculix.py` · 450 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check` | função | `(fs, main) -> Report` | Lê o deck principal (com includes) e aplica todas as regras CalculiX, devolvendo um `Report`. |

#### `pinneapple_data.preflight.common`

Tipos `Finding` e `Report` e utilitários de texto compartilhados pelas regras OpenFOAM e CalculiX. [sem teste direto]

<small>`pinneapple_data/preflight/common.py` · 83 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Finding` | classe | `(section, status, title, detail, fix, file, line, entity, rule, evidence)` | Um achado do pré-voo: seção, status, título, detalhe, correção sugerida, arquivo, linha, entidade, regra e evidência. |
| `Report` | classe | `(solver, findings, info)` | Relatório de pré-voo: solver, lista de achados e informações; `add` acrescenta e `to_dict` calcula o status geral. |
| `line_of` | função | `(text, pattern, flags=re.M) -> Optional[int]` | Número da linha onde um padrão regex aparece no texto, ou `None`. |
| `num` | função | `(v) -> Optional[float]` | Último número de uma entrada OpenFOAM (`'nu [0 2 -1 0 0 0 0] 1e-05'` → `1e-05`). |
| `vec` | função | `(v) -> Optional[List[float]]` | Lê um vetor entre parênteses de uma entrada OpenFOAM como lista de floats, ou `None`. |

#### `pinneapple_data.preflight.openfoam`

Regras de pré-voo para casos OpenFOAM: malha, propriedades de transporte/termo, condições de contorno e iniciais, esquemas, solvers lineares e passo de tempo; cada regra cita o que o OpenFOAM faz quando violada. [sem teste direto]

<small>`pinneapple_data/preflight/openfoam.py` · 551 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check` | função | `(fs, root='') -> Report` | Lê os dicionários do caso e aplica as regras do solver indicado no `controlDict` (ou só regras genéricas se o solver não está no conjunto). |

#### `pinneapple_data.process_optimizer`

Otimização de processo industrial só com dados (sem modelo físico): limpa o histórico com o relatório de saúde, treina um `HistGradientBoostingRegressor` do scikit-learn que prevê o KPI a partir das alavancas e do contexto, valida em divisão temporal e sugere ajustes das alavancas.

<small>`pinneapple_data/process_optimizer.py` · 436 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `suggest_roles` | função | `(columns) -> Dict[str, Any]` | Sugere pelos nomes das colunas qual é o alvo, quais são alavancas e quais são contexto (para revisão humana). |
| `clean_for_modeling` | função | `(df, report) -> Tuple[pd.DataFrame, List[str]]` | Tabela numérica ordenada no tempo com os problemas apontados pelo relatório de saúde marcados como ausentes. |
| `time_series_cv` | função | `(X, y, n_folds=4, l2_grid=L2_GRID, seed=0) -> Dict[str, Any]` | Validação cruzada encadeada no tempo (cada dobra treina no passado e testa no bloco seguinte) para escolher a regularização L2. |
| `fit_and_optimize` | função | `(data, target, levers, context, goal='minimize', constraints=(), bounds=None, max_move=0.…)` | Treina na parte antiga do histórico, avalia e otimiza as alavancas na parte mais recente e devolve modelo, métricas e recomendações. |
| `plant_physics` | função | `(oat, rh, load, chws_sp, cws_sp, pump_pct)` | Comportamento 'verdadeiro' da planta do exemplo, oculto do modelo de ML. |
| `example_plant_operations` | função | `(seed=11, days=60) -> Tuple[pd.DataFrame, Dict[str, Any]]` | Gera dados de exemplo de uma planta de água gelada operada com hábitos manuais de setpoint. |
| `true_saving` | função | `(result) -> Dict[str, Any]` | No exemplo, mede a economia real das recomendações usando a física verdadeira da planta. |

#### `pinneapple_data.serialization`

Grava e lê listas de `PhysicalSample` em `.pt` (com gzip opcional), HDF5 e Zarr (via `UPDZarrStore`), compatível com o formato UPD novo e o legado (`fields`/`coords`/`meta`).

<small>`pinneapple_data/serialization.py` · 338 linhas · testes: 11</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `save_manifest` | função | `(path, manifest) -> None` | Grava um manifesto JSON, criando o diretório se preciso. |
| `load_manifest` | função | `(path) -> Dict[str, Any]` | Lê um manifesto JSON. |
| `save_pt` | função | `(samples, path, *, compress=False) -> None` | Grava amostras num arquivo PyTorch `.pt`, opcionalmente comprimido com gzip. |
| `load_pt` | função | `(path) -> List[PhysicalSample]` | Lê amostras de um `.pt`, detectando sozinho se está comprimido. |
| `save_hdf5` | função | `(samples, path) -> None` | Grava amostras em HDF5 como blobs serializados pelo torch, com atributos de manifesto (requer `h5py`); o nome aparece definido duas vezes no módulo e vale a segunda definição. |
| `save_hdf5` | função | `(samples, path) -> None` | Grava amostras em HDF5 como blobs serializados pelo torch, com atributos de manifesto (requer `h5py`); o nome aparece definido duas vezes no módulo e vale a segunda definição. |
| `load_hdf5` | função | `(path) -> List[PhysicalSample]` | Lê amostras de um HDF5 gravado por `save_hdf5` (requer `h5py`). |
| `save_zarr` | função | `(samples, root, *, compressor='default') -> None` | Grava amostras num store Zarr de diretório via `UPDZarrStore` (versão corrigida; antes usava uma classe inexistente). |
| `load_zarr` | função | `(root)` | Lê amostras de um store Zarr via `UPDZarrStore`. |

#### `pinneapple_data.simulation_metadata`

Extrai metadados de simulação (solver, malha, passo de tempo, condições de contorno, parâmetros e convergência) dos arquivos de um caso OpenFOAM, deck CalculiX/Abaqus ou tabela de resíduos.

<small>`pinneapple_data/simulation_metadata/__init__.py` · 346 linhas · testes: 25</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `read_upload` | função | `(files) -> dict[str, bytes]` | Transforma arquivos enviados (zips expandidos com limites) em dict caminho relativo → bytes. |
| `parse_residual_table` | função | `(text) -> dict[str, Any] \| None` | Lê uma tabela genérica de resíduos (exportada por outros solvers) achando o cabeçalho com a coluna de iteração; `None` se não reconhece. |
| `extract` | função | `(fs) -> dict[str, Any]` | Monta o registro de metadados a partir do upload, detectando OpenFOAM, `.inp` ou tabela de resíduos; levanta `ValueError` sem arquivos. |

#### `pinneapple_data.simulation_metadata.calculix`

Lê decks CalculiX/Abaqus (`.inp`) e arquivos de execução CalculiX (`.sta`, `.cvg`, `.dat`, log) e avalia a convergência. [sem teste direto]

<small>`pinneapple_data/simulation_metadata/calculix.py` · 323 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `parse_inp` | função | `(text, include=None) -> Dict[str, Any]` | Interpreta um deck em modelo, passos, materiais, condições de contorno e cargas, resolvendo `*INCLUDE` enviados. |
| `parse_sta` | função | `(text) -> List[Dict[str, Any]]` | Lê o arquivo `.sta` (incrementos por passo, tentativas, iterações, tempos). |
| `parse_cvg` | função | `(text) -> List[Dict[str, Any]]` | Lê o arquivo `.cvg` (iterações com elementos de contato e resíduos de força). |
| `parse_dat` | função | `(text) -> Dict[str, Any]` | Do `.dat`, maior deslocamento impresso (ou mín./máx. de outros resultados) no último tempo de cada conjunto. |
| `parse_ccx_log` | função | `(text) -> Dict[str, Any]` | Do log do CalculiX: versão, se terminou, erros, número de avisos, tempo total e threads. |
| `assess` | função | `(sta, cvg, deck, log) -> Dict[str, Any]` | Dá o status de uma execução CalculiX (`failed`, `completed`, `incomplete`...) com motivos e recomendações. |

#### `pinneapple_data.simulation_metadata.convergence`

Avalia convergência a partir de históricos de resíduos e devolve status (`converged`, `still_falling`, `stalled`, `diverged`, `crashed`...), motivos e o que fazer. [sem teste direto]

<small>`pinneapple_data/simulation_metadata/convergence.py` · 228 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `downsample` | função | `(x, series, k=1500) -> Dict[str, Any]` | Reduz séries de resíduos a no máximo `k` pontos (sempre incluindo o último) para exibição. |
| `assess_steady` | função | `(log, targets) -> Dict[str, Any]` | Avalia uma execução em regime permanente pelos resíduos contra as metas do `fvSolution`. |
| `assess_transient` | função | `(log, tolerances, max_co, algorithm, reached_end) -> Dict[str, Any]` | Avalia uma execução transiente: se chegou ao fim, falhas, Courant máximo e tolerâncias dos solvers lineares. |
| `assess_table` | função | `(iters, residuals, is_log10) -> Dict[str, Any]` | Avalia um histórico genérico de resíduos (exportações de Fluent, STAR-CCM+, SU2). |

#### `pinneapple_data.simulation_metadata.foam_dict`

Parser tolerante de dicionários OpenFOAM (`controlDict`, `fvSchemes`, `fvSolution`, `0/U`...) em dicts aninhados; diretivas e macros são registradas, nunca executadas.

<small>`pinneapple_data/simulation_metadata/foam_dict.py` · 176 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `parse` | função | `(text, max_list=2000) -> Dict[str, Any]` | Interpreta um arquivo FoamFile em dicts aninhados, resumindo listas longas como `<list of N>`. |
| `header` | função | `(text) -> Dict[str, str]` | Lê só o cabeçalho FoamFile (classe, objeto, formato, nota). |
| `scalar` | função | `(v) -> Optional[float]` | Último número de uma entrada (`'uniform 0'` → `0.0`). |
| `vector` | função | `(v) -> Optional[List[float]]` | Lê um vetor de uma entrada (`'uniform (10 0 0)'` → `[10, 0, 0]`). |
| `words` | função | `(v) -> List[str]` | Separa uma entrada em palavras (lista vazia para `None` ou dict). |
| `boundary` | função | `(text) -> List[Dict[str, Any]]` | Lê os patches de um arquivo `polyMesh/boundary`: nome, tipo, nFaces, startFace e grupos. |

#### `pinneapple_data.simulation_metadata.openfoam`

Extração de caso e log OpenFOAM para o registro de metadados: log do solver, raiz do caso, tabelas de `postProcessing` e resumo por seções.

<small>`pinneapple_data/simulation_metadata/openfoam.py` · 523 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `parse_log` | função | `(text) -> dict[str, Any]` | Interpreta um log de solver OpenFOAM (ou de blockMesh/checkMesh): passos, resíduos, Courant, erros. |
| `find_root` | função | `(fs) -> str \| None` | Acha a raiz do caso: o diretório com `system/controlDict` de caminho mais curto, ou `None`. |
| `parse_dat` | função | `(text) -> tuple[list[str], np.ndarray]` | Interpreta uma tabela de function object (`forceCoeffs.dat`, `probes`...) em nomes de coluna e array, expandindo vetores entre parênteses. |
| `read_dat` | função | `(path) -> tuple[list[str], np.ndarray]` | Aplica `parse_dat` ao arquivo do caminho dado. |
| `extract_case` | função | `(fs, root) -> dict[str, Any]` | Lê os arquivos do caso (controlDict, esquemas, solução, propriedades, malha, campos iniciais, logs) num registro bruto. |
| `summarise` | função | `(case) -> dict[str, Any]` | Transforma a extração bruta do caso nas seções do registro de metadados. |

#### `pinneapple_data.snapshots`

Matrizes de snapshots para modelos de ordem reduzida: vários resultados de um modelo (tempos ou pontos de parâmetro) na mesma malha empilhados em `X`, lidos de caso OpenFOAM, série de arquivos de malha ou arrays.

<small>`pinneapple_data/snapshots.py` · 413 linhas · testes: 31</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SnapshotSet` | classe | `(X, fields, times, params, param_names, locations, source, meta, cells, snapshot_points)` | `X` (n_snapshots, n_values) com o significado de linhas e colunas: tempos, parâmetros, campos, malha; seleciona, resume e exporta `.npz`. |
| `snapshots_from_arrays` | função | `(arrays, *, times=None, params=None, param_names=None, locations=None, where=None, source…)` | Monta um `SnapshotSet` a partir de arrays de campos com os snapshots no primeiro eixo, opcionalmente com malha. |
| `load_snapshots` | função | `(files, *, fields=None, param_columns=None) -> SnapshotSet` | Lê um upload (lista de `(nome, bytes)`) num `SnapshotSet`, conforme os formatos do docstring do módulo. |

#### `pinneapple_data.splits`

Estratégias de divisão treino/validação/teste (aleatória, temporal, por grupo); é a fonte única, `pinneapple_train.splits` reexporta daqui. [sem teste direto]

<small>`pinneapple_data/splits.py` · 104 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SplitSpec` | classe | `(method, train, val, test, seed)` | Estratégia de split: `random` (embaralha), `time` (mantém a ordem, teste no fim) ou `group` (sem vazamento entre grupos), com razões e semente. |
| `split_indices` | função | `(n, spec, *, group_ids=None) -> Dict[str, np.ndarray]` | Devolve os índices de cada split para `n` amostras segundo o `SplitSpec` (exige `group_ids` na estratégia por grupo). |

#### `pinneapple_data.stl_import`

Leitor de STL ASCII e binário sem dependências, com deduplicação de vértices, para ingestão em servidor. [sem teste direto]

<small>`pinneapple_data/stl_import.py` · 93 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `STLMesh` | classe | `(verts, faces)` | Malha STL lida: vértices `(V,3)` float32 e faces `(F,3)` int64. |
| `load_stl_bytes` | função | `(data, *, dedup_eps=1e-06) -> STLMesh` | Lê STL (binário ou ASCII) de bytes, deduplicando vértices com tolerância `dedup_eps`; levanta `ValueError` para dados curtos demais. |
| `load_stl` | função | `(path, *, dedup_eps=1e-06) -> STLMesh` | Lê STL de um arquivo em disco via `load_stl_bytes`. |

#### `pinneapple_data.synth`

Geradores de dados sintéticos (EDP/EDO, EDP simbólica com diferenças finitas, distribuições, ajuste de curvas, imagens, geometria com trimesh e CadQuery) com interface comum e catálogo por nome. [sem teste]

<small>`pinneapple_data/synth/__init__.py` · 38 linhas · **sem teste direto**</small>

#### `pinneapple_data.synth.base`

Tipos base dos geradores sintéticos: `SynthConfig`, `SynthOutput` e o protocolo `SynthGenerator`. [sem teste]

<small>`pinneapple_data/synth/base.py` · 66 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SynthConfig` | classe | `(seed, device, dtype)` | Configuração comum dos geradores: semente, dispositivo e tipo de dado. |
| `SynthOutput` | classe | `(samples, extras)` | Saída padrão de um gerador: lista de amostras e extras. |
| `SynthGenerator` | classe | `()` | Protocolo de gerador sintético: precisa implementar `generate` devolvendo `SynthOutput`. |

#### `pinneapple_data.synth.curvefit`

Gerador que ajusta um polinômio com regularização ridge a dados 1D e sintetiza novas amostras com ruído. [sem teste]

<small>`pinneapple_data/synth/curvefit.py` · 153 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CurveFitSynthGenerator` | classe | `(cfg=None)` | Ajusta uma tendência polinomial (ridge) a pares (x, y) e gera novos y em novos x. |

#### `pinneapple_data.synth.distributions`

Gerador que amostra entradas de distribuições simples: gaussiana, uniforme e mistura de gaussianas. [sem teste]

<small>`pinneapple_data/synth/distributions.py` · 154 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DistributionSynthGenerator` | classe | `(cfg=None)` | Amostra de distribuição gaussiana, uniforme ou mistura de gaussianas com parâmetros configuráveis. |

#### `pinneapple_data.synth.geometry`

Gerador de variantes de uma malha triangular existente (trimesh) por transformações aleatórias controladas. [sem teste]

<small>`pinneapple_data/synth/geometry.py` · 221 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GeometrySynthGenerator` | classe | `(cfg=None)` | Gera variantes de uma malha (escala, rotação, ruído etc.) que mantêm a forma geral. |

#### `pinneapple_data.synth.geometry_cadquery`

Síntese de geometria com CadQuery: variantes de um modelo paramétrico e variantes guiadas por um STL modelo. [sem teste]

<small>`pinneapple_data/synth/geometry_cadquery.py` · 532 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ParametricCadQuerySynthGenerator` | classe | `(cfg=None)` | Sorteia parâmetros, constrói o sólido com `template_fn` do CadQuery, tessela em malha e devolve amostras tipo `PhysicalSample`. |
| `STLTemplateSynthGenerator` | classe | `(cfg=None)` | Lê um STL/malha modelo, usa a caixa envolvente e, se houver CadQuery, gera primitivas (caixa, cilindro...) parecidas; senão, deforma a malha. |

#### `pinneapple_data.synth.images`

Gerador de tarefas de reconstrução de imagem e inpainting: entrada corrompida e reconstrução como amostras tipo `PhysicalSample`. [sem teste]

<small>`pinneapple_data/synth/images.py` · 262 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ImageReconstructionSynthGenerator` | classe | `(cfg=None)` | Gera pares imagem corrompida / reconstruída em modos como inpainting suave dentro de uma máscara. |

#### `pinneapple_data.synth.operators`

Operadores pequenos usados pelos geradores sintéticos: ruído gaussiano, máscara de bloco e normalização para [0, 1]. [sem teste]

<small>`pinneapple_data/synth/operators/__init__.py` · 6 linhas · **sem teste direto**</small>

#### `pinneapple_data.synth.operators.masks`

Gera máscara retangular aleatória numa grade 2D. [sem teste]

<small>`pinneapple_data/synth/operators/masks.py` · 38 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `random_block_mask` | função | `(H, W, frac=0.2, seed=42) -> torch.Tensor` | Gera uma máscara booleana com um único bloco retangular ocupando a fração `frac` da altura e largura, em posição sorteada com gerador determinístico. |

#### `pinneapple_data.synth.operators.noise`

Soma ruído gaussiano a tensores com gerador determinístico. [sem teste]

<small>`pinneapple_data/synth/operators/noise.py` · 28 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `add_gaussian_noise` | função | `(x, std=0.01, seed=42) -> torch.Tensor` | Soma ruído normal de média zero e desvio `std` a um tensor. |

#### `pinneapple_data.synth.operators.transforms`

Normalização de tensores para [0, 1]. [sem teste]

<small>`pinneapple_data/synth/operators/transforms.py` · 28 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `normalize_01` | função | `(x, eps=1e-08) -> torch.Tensor` | Reescala um tensor para [0, 1] pelo mínimo e máximo globais, com epsilon para tensor constante. |

#### `pinneapple_data.synth.pde`

Gerador sintético de EDP/EDO simples: calor 1D, advecção 1D e logística, com diferenças finitas. [sem teste]

<small>`pinneapple_data/synth/pde.py` · 254 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SimplePhysicalSample` | classe | `(fields, coords, meta)` | Contêiner mínimo com `fields`, `coords` e `meta`, para o módulo funcionar sem o `PhysicalSample` completo. |
| `PDESynthGenerator` | classe | `(cfg=None)` | Gera trajetórias de calor 1D, advecção 1D ou EDO logística a partir de domínio, grade, tempo e condição inicial. |

#### `pinneapple_data.synth.pde_symbolic`

Interpreta uma EDP escrita como texto (forma de resíduo, como `"u_t - alpha*u_xx"`) e gera trajetórias e resíduo com diferenças finitas. [sem teste]

<small>`pinneapple_data/synth/pde_symbolic.py` · 498 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SymbolicFDSynthGenerator` | classe | `(cfg=None)` | Converte uma equação simbólica em operador de diferenças finitas e gera um dataset sintético de trajetórias. |
| `make_fd_residual_fn` | função | `(equation, *, parameters=None, bc='periodic')` | Cria uma função que avalia o resíduo simbólico trocando `u_t`, `u_x`, `u_xx` por diferenças finitas. |

#### `pinneapple_data.synth.registry`

`SynthCatalog`: fábrica que constrói geradores sintéticos por nome a partir de um registro. [sem teste]

<small>`pinneapple_data/synth/registry.py` · 73 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SynthCatalog` | classe | `(registry)` | Registro nome → classe de gerador; `list` mostra os nomes e `build` instancia (levanta `KeyError` para nome desconhecido). |

#### `pinneapple_data.synth.sample_adapter`

Converte saídas dos geradores sintéticos em `PhysicalSample` quando possível, ou num substituto mínimo. [sem teste]

<small>`pinneapple_data/synth/sample_adapter.py` · 192 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicalSampleLike` | classe | `(fields, coords, meta)` | Protocolo mínimo de amostra: `fields`, `coords` e `meta`. |
| `has_pinnego_physical_sample` | função | `() -> bool` | Diz se `pinneapple_data.physical_sample.PhysicalSample` pode ser importado (o nome da função ainda usa o nome antigo do projeto). |
| `to_physical_sample` | função | `(sample_like, *, device=None, dtype=None) -> Any` | Conversão de melhor esforço para `PhysicalSample` (ou um dataclass substituto) a partir de objeto ou dict. |

#### `pinneapple_data.transforms`

Normalizadores para pré-processamento (fonte única; `pinneapple_train.normalizers` e `pinneapple_train.preprocess` reexportam daqui): base abstrata, `StandardScaler` e `MinMaxScaler`.

<small>`pinneapple_data/transforms.py` · 140 linhas · testes: 2 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Normalizer` | classe | `()` | Base abstrata de normalizadores com `fit`/`transform`/`inverse` (e apelidos `encode`/`decode`) e estado serializável; os métodos base levantam `NotImplementedError`. |
| `StandardScaler` | classe | `(eps, mean_, std_)` | Normaliza para média zero e variância unitária; aceita estilo de método de classe (`StandardScaler.fit(x)`) e de instância. |
| `MinMaxScaler` | classe | `(eps, min_, max_)` | Reescala cada feature para [0, 1] e desfaz a escala com `inverse`. |

#### `pinneapple_data.udr1`

Atalhos do ASME Form U-DR-1 sobre o motor genérico `formfill` (modelo `asme_u-dr-1`): extrair, compilar requisitos e preencher o formulário.

<small>`pinneapple_data/udr1/__init__.py` · 30 linhas · testes: 36</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `extract_rules` | função | `(doc) -> List[Candidate]` | Aplica `formfill.extract_rules` ao documento com o spec do U-DR-1. |
| `compile_requirements` | função | `(docs, candidates, overrides=None) -> Compilation` | Compila os candidatos com o spec do U-DR-1, aceitando respostas do engenheiro como `overrides`. |
| `fill_form` | função | `(blank_pdf, comp) -> bytes` | Preenche o PDF em branco do U-DR-1 com a compilação. |

#### `pinneapple_data.unified`

Padronizador de dados de engenharia: vários arquivos com convenções diferentes viram um esquema físico único (UPD), com mapeamento revisado por pessoa, conversão exata de unidades, fusos para UTC e alinhamento num eixo de tempo comum.

<small>`pinneapple_data/unified.py` · 584 linhas · testes: 24</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `canonical_name` | função | `(base, quantity) -> str` | Gera o nome canônico de uma coluna a partir do cabeçalho e da grandeza (por exemplo `'CHW Supply Temp'` → `chw_supply_temperature`). |
| `conversion_text` | função | `(src, dst, difference=False) -> str` | Texto legível de uma conversão de unidade, por exemplo `°F → °C: (x − 32) × 0.5556`. |
| `propose_mapping` | função | `(df, source, system='metric', time_column=None, units=None, tz=None) -> Dict[str, Any]` | Rascunha o mapeamento de uma fonte a partir dos cabeçalhos (coluna de tempo, grandeza, unidade, nome e unidade alvo), para revisão humana. |
| `make_recipe` | função | `(mappings, system='metric', resample='auto', layout='wide', agg='mean') -> Dict[str, Any]` | Monta a receita (`pinneapple.recipe/1`) a partir dos mapeamentos, com sistema alvo, reamostragem, layout e agregação. |
| `standardize` | função | `(sources, recipe) -> Dict[str, Any]` | Aplica a receita às fontes e devolve tabelas larga e longa, manifesto, notas por fonte e concordância entre fontes. |
| `write` | função | `(result, fmt='csv', layout='wide') -> Tuple[bytes, str, str]` | Serializa o resultado padronizado e devolve (bytes, tipo de mídia, nome de arquivo). |
| `to_physical_sample` | função | `(result)` | Converte a tabela larga num `PhysicalSample` (xarray com coordenada `time` e unidades nos atributos). |
| `example_sources` | função | `(seed=3) -> List[Tuple[str, bytes]]` | Três fontes de exemplo da mesma planta de água gelada (CSV de BMS, JSON de SCADA, Excel de bancada) com convenções diferentes. |
| `before_after_series` | função | `(result, k=500) -> Dict[str, Any]` | Por variável, valores brutos de cada fonte e valores padronizados no eixo UTC comum, reduzidos a no máximo `k` pontos. |

#### `pinneapple_data.upd_dataset`

`UPDDataset` endurecido: lista de amostras com versão de esquema no manifesto, validação e salvar/carregar em pt, zarr ou hdf5.

<small>`pinneapple_data/upd_dataset.py` · 192 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `UPDDataset` | classe | `(samples, manifest)` | Dataset de `PhysicalSample` com validação e persistência em `.pt`, Zarr ou HDF5. |

#### `pinneapple_data.upd_types`

Tipos de dados puros do pipeline UPD (shard, especificações de amostragem de condições, batch PINN) e conversão de `datetime64` para segundos; o docstring ainda cita o pacote antigo `pinneapple_pinn`. [sem teste]

<small>`pinneapple_data/upd_types.py` · 103 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `UPDItem` | classe | `(zarr_path, meta_path)` | Um item de shard UPD: caminho do store Zarr e do JSON de metadados; lê os metadados e abre o dataset. |
| `ConditionSpec` | classe | `(name, type, equation, n, options)` | Como amostrar uma condição PINN: `initial`, `boundary`, `interior` ou `slice`, com número de pontos e variáveis. |
| `SamplingSpec` | classe | `(n_collocation, conditions, n_data, replace, seed)` | Plano de amostragem de um shard UPD: pontos de colocação e lista de `ConditionSpec`. |
| `Batch` | classe | `(collocation, conditions, data)` | Batch de treino PINN: tensores de colocação, de cada condição e pares de dados supervisionados. |
| `seconds_since` | função | `(t, t0=None) -> np.ndarray` | Converte um array `datetime64` em segundos (float64) desde `t0` (padrão: o menor tempo). |

#### `pinneapple_data.validators`

Validação de `PhysicalSample`: unidades (política `strict`/`warn`/`off`), faixas, não negatividade e coordenadas monótonas. [sem teste]

<small>`pinneapple_data/validators.py` · 161 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ValidationIssue` | classe | `(level, message, field)` | Um problema de validação: nível (`error` ou `warning`), mensagem e campo. |
| `validate_physical_sample` | função | `(sample, *, units_policy='warn', required_units=None, ranges=None, non_negative=None, mon…)` | Valida uma amostra e devolve a lista de problemas encontrados. |
| `assert_valid_physical_sample` | função | `(*args, **kwargs) -> None` | Valida uma amostra e levanta `ValueError` se houver algum erro. |

#### `pinneapple_data.zarr_cached_store`

Envolve o `UPDZarrStore` com cache LRU por amostra e por campo, limitado por número de itens. [sem teste]

<small>`pinneapple_data/zarr_cached_store.py` · 231 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ZarrCacheConfig` | classe | `(max_samples, max_fields, enable_field_cache)` | Limites do cache por contagem: máximo de amostras e de leituras de campo, e se o cache de campo está ligado. |
| `CachedUPDZarrStore` | classe | `(root, *, cache=None, mode='r')` | Store Zarr UPD com cache LRU: `count`, `manifest`, `read_sample` e estatísticas do cache. |

#### `pinneapple_data.zarr_cached_store_bytes`

Envolve o `UPDZarrStore` com cache LRU limitado por bytes (`ByteLRUCache`) por amostra e por campo. [sem teste]

<small>`pinneapple_data/zarr_cached_store_bytes.py` · 226 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ZarrByteCacheConfig` | classe | `(max_sample_bytes, max_field_bytes, enable_field_cache)` | Limites do cache em bytes para amostras inteiras e para campos. |
| `CachedUPDZarrStoreBytes` | classe | `(root, *, cache=None, mode='r')` | Store Zarr UPD com cache limitado por bytes: `count`, `manifest`, `read_sample` e estatísticas. |

#### `pinneapple_data.zarr_iterable`

`IterableDataset` que lê `PhysicalSample` de um `UPDZarrStore`, abrindo o store em cada worker e dividindo o trabalho entre workers. [sem teste]

<small>`pinneapple_data/zarr_iterable.py` · 69 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ZarrUPDIterable` | classe | `(root, fields, coords, start, stop, step, shuffle, seed)` | Itera as amostras de um store Zarr UPD com divisão por worker do `DataLoader`, sem exigir `__len__`. |

#### `pinneapple_data.zarr_prefetch`

`IterableDataset` sobre store Zarr UPD com cache, thread produtora em segundo plano, memória fixada e transferência assíncrona para GPU. [sem teste]

<small>`pinneapple_data/zarr_prefetch.py` · 205 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PrefetchConfig` | classe | `(prefetch, queue_max, use_sample_cache, pin_memory, target_device, transfer_non_blocking)` | Controles do prefetch: tamanho da fila, pin_memory, dispositivo alvo e transferência não bloqueante. |
| `PrefetchZarrUPDIterable` | classe | `(root, *, fields=None, coords=None, dtype=None, start=0, end=None, stride=1, sample_ctor=…)` | Lê amostras do Zarr numa thread produtora e entrega em CPU ou já movidas para a GPU. |

#### `pinneapple_data.zarr_shard_iterable`

`IterableDataset` que lê o `index.json` de um store Zarr UPD fatiado em shards e distribui as partes entre os workers. [sem teste]

<small>`pinneapple_data/zarr_shard_iterable.py` · 162 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ShardAwareConfig` | classe | `(pin_memory, target_device, transfer_non_blocking, use_sample_cache)` | Configuração da iteração por shards: pin_memory, dispositivo, transferência não bloqueante e embaralhamento. |
| `ShardAwareZarrUPDIterable` | classe | `(root, *, fields=None, coords=None, dtype=None, sample_ctor=None, cache=None, cfg=None)` | Itera as partes de shard atribuídas a cada worker, uma de cada vez, a partir do `index.json`. |

#### `pinneapple_data.zarr_shards`

Escritor de stores Zarr UPD fatiados por chave (tempo, região, regime) em `shards/<chave>/part=NNNN/data.zarr` com `index.json`. [sem teste]

<small>`pinneapple_data/zarr_shards.py` · 154 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ShardSpec` | classe | `(key_fn, max_per_shard)` | Nomeação de shards: função `key_fn(sample)` que devolve a chave (por exemplo `time=2020-01`) e tamanho de parte. |
| `UPDZarrShardedWriter` | classe | `(root, shard_spec, *, shards_dirname='shards', index_filename='index.json')` | Escreve amostras em vários shards por chave e partes numeradas, e grava o índice. |

#### `pinneapple_data.zarr_store`

Store Zarr do UPD em layout colunar (cada campo empilhado entre amostras, coordenadas e manifesto opcionais) para persistir `PhysicalSample`.

<small>`pinneapple_data/zarr_store.py` · 396 linhas · testes: 1</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `UPDZarrStore` | classe | `(root, mode='r')` | Store Zarr UPD mínimo: escreve amostras, itera, conta, lista campos e coordenadas e lê o manifesto. |

### Exemplos de uso reais do repositório

**Criar um PhysicalSample em grade e convertê-lo para dict de treino** — `examples/data_pipeline/01_physical_sample_basics.py`

```python
from pinneapple_data.physical_sample import PhysicalSample

ds = xr.Dataset(
    data_vars=dict(
        T2M=(("t","x"), np.random.randn(24, 16).astype("float32")),
        U10M=(("t","x"), np.random.randn(24, 16).astype("float32")),
    ),
    coords=dict(t=np.arange(24), x=np.arange(16)),
)

sample = PhysicalSample(
    state=ds,
    domain={"type": "grid"},
    schema={"governing": {"type": "PDE", "name": "toy"}},
    provenance={"source": "demo"},
)
```

**Planejar uma campanha que não repete execuções existentes** — `tests/pinneapple_data/test_doe.py`

```python
    plan = plan_campaign(space, method="maximin", seed=1, existing=existing, existing_id_column="run_id")
    assert len(plan.cases) == 12 and list(plan.cases["priority"]) == list(range(1, 13))
    assert plan.cases["case_id"].iloc[0] == "case_001"
    s = plan.summary()
    assert s["coverage"]["combined"]["fill_distance"] < s["coverage"]["existing"]["fill_distance"]
    res = plan.study_result()
    assert isinstance(res, StudyResult) and res.passed and res.status == "planned"
    assert StudyResult.from_json(res.to_json()) == res
    skel = plan.dataset_skeleton(outputs=["y"], units={"y": "K"})
```

**Pré-voo de um caso OpenFOAM ou deck CalculiX enviado** — `tests/test_preflight.py`

```python
from pinneapple_data.cae.upload import expand
from pinneapple_data.preflight import preflight


def _run(name):
    with open(os.path.join(EX, name), "rb") as f:
        return preflight(expand([(name, f.read())]))
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 45 de 115.
- Funções de teste em arquivos que importam algum módulo do pacote: 507.

### Limitações conhecidas

- A raiz do pacote importa quase tudo dentro de `try/except Exception: pass`: se uma dependência opcional faltar (xarray, zarr, torch), o nome simplesmente não existe em `pinneapple_data`, sem aviso.
- Muitas dependências são opcionais e só exigidas pelo módulo que as usa: `trimesh` (geometria, `geometry_features`), `meshio`, `h5py`, `pyarrow`, `pydantic` (contratos), `pypdf`/`pdfplumber`/`reportlab` (formfill), o programa `tesseract` (OCR), `tfrecord` (DeepMind), `cadquery`, `mp-api`, `cdsapi`, e um servidor Ollama local para `formfill.llm`.
- Vários clientes de `datasets.external_sources` e carregadores de `datasets.real_world` devolvem dados sintéticos quando a fonte não está disponível (marcados com `_synthetic` ou `source: synthetic`); alguns datasets de `real_world` são sintéticos por definição. Não use esses dados como medições reais sem conferir a marcação.
- `dataloaders.build_physical_sample_dataloader` só aceita amostras em grade; amostras em malha ainda não são suportadas.
- `transforms.Normalizer` é base abstrata: `fit`, `transform` e `inverse` levantam `NotImplementedError` e precisam de subclasse.
- Há nomes antigos nos docstrings: `pinneapple_geom` (hoje `pinneapple_design.geometry`), `pinneapple_pinn` (citado em `upd_types` e `adapters.pinn_batch_builders`) e `pinnego` em `has_pinnego_physical_sample`. Em `serialization`, `save_hdf5` aparece definido duas vezes e vale a segunda definição. O exemplo de `formfill.pipeline` cita o id `api610_pump`, mas o registro usa `pump`.
- Muitos módulos não têm teste direto: adaptadores, caches de Zarr, iteráveis e prefetch, `synth` inteiro, `augmentation`, `geometry_features`, `grf_sampler`, `validators`, `upd_types` e vários submódulos de `cae`, `formfill`, `preflight`, `lineage` e `simulation_metadata` (estes são exercitados pelos testes dos pacotes-pai e dos apps).
- `cae.foam` não remonta casos decompostos (`processor*`), e `cae.upload` descarta essas pastas; diretivas OpenFOAM (`#include`, `#codeStream`) são registradas, não executadas.

## 3. `pinneapple_pdb`

### Objetivo

O `pinneapple_pdb` (physics database) tem duas partes independentes. A primeira é o `PhysicalDatasetBuilder`, que monta datasets físicos a partir de hubs de dados da Terra (NASA CMR via `earthaccess`): busca coleções e granules, escolhe variáveis e janela espaço-temporal, abre os dados com xarray, padroniza dimensões, valida unidades, faixas e monotonicidade, acrescenta variáveis derivadas (vorticidade e divergência), fatia em janelas de tempo e blocos espaciais com tags de regime e grava tudo em Zarr no formato UPD, com um catálogo e um esquema físico (equações, contornos, política de unidades) vindo de templates como `atmosphere_reanalysis_v1`.

A segunda parte é `benchmarks`, um catálogo pequeno e curado de datasets de referência resolvidos por nome (`get_benchmark("lane_emden_n1.5")`), no mesmo padrão `@register_x`/`get_x`/`list_x` do registro de presets de `pinneapple_physics`. Cada entrada é recalculada ao ser pedida (integração com SciPy ou solução fechada) e conferida contra um valor publicado ou uma solução exata; se a conferência falhar, a função levanta erro em vez de devolver um perfil não verificado. As entradas cobrem Lane-Emden (n=1.5 e n=3), sustentação de vórtice de Polhamus no ângulo de Concorde (teoria, não dado de voo), camada-limite de Blasius, condução transiente em parede plana com Bi=1 e três casos com as equações do PDEBench (advecção 1D, difusão-reação 1D, Darcy 2D) sem baixar as trajetórias do PDEBench.

O catálogo de benchmarks é usado pelo guardrail físico de `pinneapple_llm` e por exemplos como `examples/use_cases/concorde_high_aoa`; o builder é usado pelos exemplos de `examples/physics_db/`. O pacote não tem CLI: o `pyproject.toml` registra que o antigo script `pinneapple_pdb.cli:main` foi removido por não existir.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 7 | 1.929 | 9 | 14 | 100 (5 arquivos) | 5 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `benchmarks` | Catálogo curado, por nome, de datasets de referência (x, y) recalculados e verificados a cada uso. | 1 | 5 | 35 |
| `builder` | `PhysicalDatasetBuilder`: busca dados em hubs da Terra (earthaccess/CMR), valida, deriva, fatia e grava Zarr UPD com catálogo. | 1 | 5 | 0 |
| `derived` | Variáveis derivadas (vorticidade e divergência) a partir de U/V em grade lat/lon. | 1 | 3 | 0 |
| `shard` | Fatiamento por janelas de tempo e blocos espaciais, com tags de regime (trópicos, polar). | 1 | 6 | 0 |
| `templates` | Templates de esquema físico (por exemplo reanálise atmosférica). | 1 | 1 | 0 |
| `validate` | Padronização de dimensões e validação de unidades, faixas e monotonicidade. | 1 | 3 | 0 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_pdb`

Ponto de entrada: reexporta o builder e suas specs, `schema_templates` e a API do catálogo de benchmarks; não tem docstring de módulo.

<small>`pinneapple_pdb/__init__.py` · 36 linhas · testes: 65</small>

#### `pinneapple_pdb.benchmarks`

Catálogo de referências por nome com oito entradas registradas, cada uma recalculada e conferida contra valor publicado ou solução exata (levanta erro se divergir); o docstring ainda fala em "FIVE entries", desatualizado.

<small>`pinneapple_pdb/benchmarks.py` · 984 linhas · testes: 35</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BenchmarkEntry` | classe | `(name, description, reference_source, x_vars, y_vars, reference_x, reference_y, verificat…)` | Um dataset de referência nomeado: descrição, fonte, variáveis x/y, arrays de referência e nota de verificação. |
| `register_benchmark` | função | `(name)` | Decorador que registra uma fábrica sem argumentos de `BenchmarkEntry` sob um nome. |
| `get_benchmark` | função | `(name) -> BenchmarkEntry` | Resolve um benchmark pelo nome, construindo e verificando a entrada. |
| `list_benchmarks` | função | `() -> List[str]` | Lista ordenada dos nomes de benchmarks registrados. |
| `benchmark_catalog` | função | `() -> Dict[str, BenchmarkEntry]` | Dicionário nome -> `BenchmarkEntry` de todos os benchmarks, construídos e verificados de novo. |

#### `pinneapple_pdb.builder`

Construtor de datasets físicos a partir de NASA CMR/`earthaccess` (extra opcional `pinneapple[earthdata]`, importado sob demanda): busca, inspeção de granule, seleção, validação, derivadas, fatiamento e escrita Zarr UPD com catálogo; `build` exige login no Earthdata. [sem teste direto]

<small>`pinneapple_pdb/builder.py` · 539 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HubQuery` | classe | `(provider, short_name, keyword)` | Parâmetros de busca no hub: provedor, nome curto e palavra-chave. |
| `SpaceTime` | classe | `(time_start, time_end, bbox, stride, chunking)` | Extensão temporal e espacial (bbox) com passo e chunking. |
| `VariableSelection` | classe | `(include, exclude, packs)` | Seleção de variáveis por inclusão, exclusão e pacotes. |
| `PhysicalSchema` | classe | `(physical_system, governing_equations, ics, bcs, forcings, units_policy, regime_tags, val…)` | Metadados físicos: sistema, equações, CI, CC, forçantes, política de unidades, tags e validade. |
| `PhysicalDatasetBuilder` | classe | `(packs=None)` | Builder MVP: busca e inspeção de granules, sharding por tempo e blocos, derivadas e validações, saída Zarr com catálogo. |

#### `pinneapple_pdb.derived`

Calcula vorticidade e divergência de vento U/V numa grade lat/lon e aplica as derivadas pedidas numa spec. [sem teste direto]

<small>`pinneapple_pdb/derived.py` · 116 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DerivedSpec` | classe | `(derived, u_name, v_name)` | Quais variáveis derivadas acrescentar e os nomes de U e V. |
| `add_vorticity_divergence` | função | `(ds, u_name=None, v_name=None) -> xr.Dataset` | Acrescenta vorticidade e divergência calculadas de U/V em grade lat/lon. |
| `apply_derived` | função | `(ds, spec) -> xr.Dataset` | Aplica as derivadas de uma spec e devolve um novo dataset. |

#### `pinneapple_pdb.shard`

Gera janelas de tempo não sobrepostas e blocos lat/lon alinhados, recorta o dataset e marca tags `tropics`/`polar` pela latitude. [sem teste direto]

<small>`pinneapple_pdb/shard.py` · 72 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ShardSpec` | classe | `(time_window, tile_deg, add_regime_tags)` | Janela de tempo, tamanho de bloco em graus e se adiciona tags de regime. |
| `iter_time_windows` | função | `(time_values, window) -> Iterator[Tuple[pd.Timestamp, pd.Timestamp]]` | Gera pares (início, fim) de janelas de tempo não sobrepostas. |
| `subset_time` | função | `(ds, a, b) -> xr.Dataset` | Recorta o intervalo de tempo [a, b). |
| `iter_tiles` | função | `(lat_vals, lon_vals, dlat, dlon)` | Gera limites de blocos lat/lon alinhados à grade de passos dados. |
| `subset_tile` | função | `(ds, lat0, lat1, lon0, lon1) -> xr.Dataset` | Recorta um bloco espacial; supõe coordenadas ordenadas. |
| `regime_tags_for` | função | `(ds) -> list[str]` | Tags de regime (`tropics`, `polar`) pela extensão em latitude. |

#### `pinneapple_pdb.templates`

Devolve o template `atmosphere_reanalysis_v1` (equações primitivas hidrostáticas); a política pede SI mas nenhuma conversão automática é feita. [sem teste direto]

<small>`pinneapple_pdb/templates.py` · 28 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `schema_templates` | função | `() -> Dict[str, Dict[str, Any]]` | Mapa id de template -> dicionário compatível com `PhysicalSchema`. |

#### `pinneapple_pdb.validate`

Padroniza nomes e ordem de dimensões (time, lev, lat, lon) e devolve a lista de problemas de unidades, faixas e monotonicidade. [sem teste direto]

<small>`pinneapple_pdb/validate.py` · 154 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ValidationSpec` | classe | `(require_units, ranges, enforce_lev_monotonic, enforce_time_monotonic)` | Spec de validação: exigir unidades, faixas por variável e monotonicidade de nível e tempo. |
| `standardize_dims` | função | `(ds) -> xr.Dataset` | Padroniza nomes de dimensões e transpõe variáveis para (time, lev, lat, lon). |
| `validate_dataset` | função | `(ds, spec) -> List[str]` | Valida o dataset contra a spec e devolve as mensagens de problema (vazia se válido). |

### Exemplos de uso reais do repositório

**Inspecionar um granule de um hub da Terra** — `examples/physics_db/00_quickstart_inspect.py`

```python
    b = PhysicalDatasetBuilder()

    # 1) Search collections (keep keyword broad at first)
    collections = b.list_collections(keyword="MERRA", limit=10)
    print("\nTop matches:")
    for c in collections[:10]:
        print(f"- {c.get('short_name')} | provider={c.get('provider')} | {c.get('title','')[:80]}")

    # 2) Pick one (edit as needed)
    # Many users end up using MERRA-2 products; pick a short_name shown above.
    b.set_dataset(provider=collections[0].get("provider"), short_name=collections[0].get("short_name"))
```

**Usar um benchmark do catálogo como referência teórica** — `examples/use_cases/concorde_high_aoa/concorde_high_aoa_pipeline.py`

```python
    bench = get_benchmark("concorde_high_aoa")

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.tick_params(colors="white")

    ax.plot(
        bench.reference_x[:, 0], bench.reference_y[:, 0],
        "-", color="white", lw=1.5,
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 2 de 7.
- Funções de teste em arquivos que importam algum módulo do pacote: 100.

### Limitações conhecidas

- O builder depende do extra opcional `earthaccess` (`pip install pinneapple[earthdata]`), de acesso à rede e de login no Earthdata (`build` chama `login`); também usa xarray, pandas e um engine como `pydap`.
- Os módulos do builder (`builder`, `derived`, `shard`, `templates`, `validate`) não têm teste que os importe diretamente; os testes do pacote cobrem sobretudo `benchmarks`.
- O catálogo de benchmarks é pequeno e curado; as entradas PDEBench reproduzem só as equações e parâmetros de exemplo, não as trajetórias reais do PDEBench.
- `concorde_high_aoa` é a teoria de Polhamus avaliada na razão de aspecto aproximada do Concorde, não dado de túnel de vento ou de voo.
- O docstring de `benchmarks` diz "FIVE entries", mas há oito registradas; um dataset LES de canal citado no docstring não foi incluído por viver fora do repositório.
- A política de unidades do template pede SI, mas nenhuma conversão é feita: as unidades dos granules são gravadas como vieram (`converted: False`).
- Não há CLI: o script `pinneapple_pdb.cli:main` foi removido do `pyproject.toml` porque o módulo não existe.

## 4. `pinneapple_adaptation`

### Objetivo

`pinneapple_adaptation` reúne duas formas de reaproveitar um PINN já treinado em um problema físico parecido, sem treinar do zero. A primeira é transfer learning (`transfer_learning`, antigo `pinneapple_transfer`): pegar um modelo treinado, congelar parte das camadas e ajustar o resto no novo problema, por exemplo Burgers com outra viscosidade. A segunda é meta-learning (`meta_learning`, antigo `pinneapple_meta`): treinar uma inicialização compartilhada, com MAML ou Reptile, que se adapta a qualquer membro de uma família de EDPs em poucos passos de gradiente.

O pacote trabalha com qualquer `torch.nn.Module` e com funções de perda física no formato `(model, batch) -> perda`. Na prática, essas funções vêm do compilador de problemas de `pinneapple_physics` (`compile_problem` sobre um `ProblemSpec` de um preset, como `laplace_2d_default` ou `burgers_1d_default`), como mostra o teste `tests/test_adaptation_and_active_learning.py`. O catálogo de métodos (`pinneapple_catalog.methods`) lista o meta-learning como método de treino (item T18).

Na raiz há dois atalhos: `fine_tune`, que monta um `TransferConfig` e um `TransferTrainer`, chama `prepare()` e depois `finetune()`; e `adapt_model`, que escolhe entre meta-learning (quando o alvo é um dict com chave `support`) e transfer learning. Os nomes antigos `transfer` e `meta` continuam como apelidos dos submódulos, para compatibilidade.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 13 | 2.946 | 10 | 9 | 154 (2 arquivos) | 2 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `meta_learning` | Meta-learning (MAML e Reptile) sobre famílias paramétricas de EDPs: amostrador de tarefas, treinadores, configurações e um invólucro `MetaModel` para adaptar e prever. | 6 | 8 | 151 |
| `transfer_learning` | Transfer learning e fine-tuning de PINNs: congelamento de camadas, taxas de aprendizado por camada, descongelamento progressivo, interpolação de pesos entre variantes e adaptação de domínio por MMD. | 6 | 9 | 151 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_adaptation`

Raiz do pacote: reexporta as classes de `transfer_learning` e `meta_learning`, mantém os apelidos antigos `transfer`/`meta` e define os atalhos `fine_tune` e `adapt_model`.

<small>`pinneapple_adaptation/__init__.py` · 106 linhas · testes: 3</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `fine_tune` | função | `(source_model, target_physics_fn, target_data=None, *, strategy='finetune', epochs=500, l…)` | Atalho que cria `TransferConfig` + `TransferTrainer`, chama `prepare()` e `finetune()` e devolve o dict de resultado (modelo, histórico, métricas). |
| `adapt_model` | função | `(model, target, *, mode='auto', **kwargs)` | Adapta um modelo a uma tarefa nova: usa `meta_adapt` se o alvo for um dict com `support` (ou `mode='meta'`), senão `fine_tune`. |

#### `pinneapple_adaptation.meta_learning`

Reexporta configurações, amostrador e treinadores de meta-learning e oferece `meta_train` (constrói e treina) e `meta_adapt` (adapta a uma tarefa nova).

<small>`pinneapple_adaptation/meta_learning/__init__.py` · 113 linhas · testes: 3</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `meta_train` | função | `(model, sampler, *, algorithm='reptile', **cfg_kwargs)` | Constrói um `ReptileTrainer` (padrão) ou `MAMLTrainer`, executa `train()` e devolve o treinador já treinado. |
| `meta_adapt` | função | `(meta_trainer, task, n_steps=10)` | Chama `adapt` do treinador meta-treinado para uma tarefa nova e devolve uma cópia adaptada do modelo. |

#### `pinneapple_adaptation.meta_learning.config`

Dataclasses `MAMLConfig` e `ReptileConfig` com os hiperparâmetros dos laços interno e externo, validados em `__post_init__`.

<small>`pinneapple_adaptation/meta_learning/config.py` · 134 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MAMLConfig` | classe | `(n_inner_steps, inner_lr, outer_lr, n_tasks_per_batch, n_meta_epochs, first_order, device…)` | Hiperparâmetros do MAML: passos e taxa do laço interno, taxa externa, tarefas por lote, épocas, flag `first_order`, dispositivo, semente e checkpoint. |
| `ReptileConfig` | classe | `(n_inner_steps, inner_lr, outer_lr, n_tasks_per_batch, n_meta_epochs, device, seed, epsil…)` | Hiperparâmetros do Reptile, incluindo o passo de interpolação `epsilon` (validado no intervalo (0, 1]). |

#### `pinneapple_adaptation.meta_learning.maml`

Treinador MAML para famílias de EDPs, em versão de primeira ordem (padrão) ou de segunda ordem; usa `torch.func.functional_call` quando existe e cai para cópia profunda + SGD em PyTorch antigo.

<small>`pinneapple_adaptation/meta_learning/maml.py` · 480 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MAMLTrainer` | classe | `(model, config, task_sampler) -> None` | Treina um modelo com MAML sobre uma distribuição de tarefas de EDP, com `meta_update`, `train`, `adapt`, `save` e `load`. |

#### `pinneapple_adaptation.meta_learning.meta_model`

Invólucro `MetaModel` de um modelo meta-treinado: adapta a uma função física nova, faz predição e salva/carrega o estado.

<small>`pinneapple_adaptation/meta_learning/meta_model.py` · 191 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MetaModel` | classe | `(model, meta_type='reptile', device='cpu') -> None` | Invólucro de um modelo meta-treinado (rótulo `reptile` ou `maml`) para adaptar em poucos passos, prever e serializar. |

#### `pinneapple_adaptation.meta_learning.reptile`

Treinador Reptile: treina em cada tarefa separadamente e interpola os pesos compartilhados na direção dos pesos de cada tarefa, sem gradientes de segunda ordem.

<small>`pinneapple_adaptation/meta_learning/reptile.py` · 294 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ReptileTrainer` | classe | `(model, config, task_sampler) -> None` | Treinador Reptile com `meta_update`, `train`, `adapt` (devolve cópia adaptada), `save` e `load`. |

#### `pinneapple_adaptation.meta_learning.task_sampler`

`PDETaskSampler`: sorteia parâmetros de EDP em intervalos dados e monta tarefas com conjuntos de suporte e de consulta para o meta-treino.

<small>`pinneapple_adaptation/meta_learning/task_sampler.py` · 196 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PDETaskSampler` | classe | `(param_ranges, physics_fn_factory, data_factory=None, n_support=64, n_query=64, input_dim…)` | Gera tarefas de meta-treino: parâmetros sorteados uniformemente, função física da fábrica e pontos de suporte/consulta (de `data_factory` ou uniformes em [0,1]^d). |

#### `pinneapple_adaptation.transfer_learning`

Reexporta `TransferConfig`, `TransferTrainer`, utilitários de congelamento, `ParametricFamilyTransfer` e `PhysicsTransferAdapter`. [sem teste direto do `__init__`]

<small>`pinneapple_adaptation/transfer_learning/__init__.py` · 49 linhas · **sem teste direto**</small>

#### `pinneapple_adaptation.transfer_learning.adapter`

`PhysicsTransferAdapter`: combina a perda física do domínio alvo com uma regularização pela física de origem e uma perda MMD (kernel RBF) entre features dos dois domínios.

<small>`pinneapple_adaptation/transfer_learning/adapter.py` · 346 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsTransferAdapter` | classe | `(model, source_spec, target_spec, source_physics_fn=None, source_weight=0.1, mmd_bandwidt…)` | Adapta um PINN de um domínio físico de origem a um alvo relacionado, somando perda alvo, regularização da física de origem e perda MMD entre features. |

#### `pinneapple_adaptation.transfer_learning.config`

Dataclass `TransferConfig` com estratégia de transferência (`finetune`, `feature_extract`, `partial_freeze`, `progressive`), prefixos de congelamento, taxas de aprendizado e pesos de perda.

<small>`pinneapple_adaptation/transfer_learning/config.py` · 98 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TransferConfig` | classe | `(strategy, freeze_prefix, unfreeze_prefix, layer_lr_scale, base_lr, finetune_lr, epochs, …)` | Configuração de um fine-tuning: estratégia, prefixos a congelar/descongelar, escala de taxa por camada, taxas base e de ajuste, épocas, pesos de perda física/dados e warmup. |

#### `pinneapple_adaptation.transfer_learning.freeze`

Funções para congelar/descongelar parâmetros por prefixo de nome, montar grupos de otimizador com taxa de aprendizado por camada e contar parâmetros treináveis. [sem teste direto]

<small>`pinneapple_adaptation/transfer_learning/freeze.py` · 202 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `freeze_layers` | função | `(model, prefixes) -> int` | Congela (`requires_grad=False`) os parâmetros cujo nome começa por algum prefixo e devolve quantos foram congelados. |
| `unfreeze_layers` | função | `(model, prefixes) -> int` | Descongela os parâmetros cujo nome começa por algum prefixo e devolve quantos foram alterados. |
| `freeze_all_except` | função | `(model, trainable_prefixes) -> None` | Congela todos os parâmetros que não casam com os prefixos treináveis. |
| `layer_lr_groups` | função | `(model, base_lr, scale_dict) -> List[dict]` | Monta grupos de parâmetros para o otimizador com taxa de aprendizado por camada (taxa base vezes a escala do prefixo). |
| `count_trainable` | função | `(model) -> Dict[str, int]` | Conta parâmetros treináveis, congelados e totais de um modelo. |

#### `pinneapple_adaptation.transfer_learning.parametric`

`ParametricFamilyTransfer`: guarda variantes de um PINN ajustadas para valores diferentes de um parâmetro físico e interpola linearmente os pesos entre as duas mais próximas.

<small>`pinneapple_adaptation/transfer_learning/parametric.py` · 272 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ParametricFamilyTransfer` | classe | `(base_model, param_name) -> None` | Coleção de PINNs ajustados por valor de um parâmetro escalar: devolve a variante mais próxima, interpola pesos entre vizinhas e salva/carrega a família inteira. |

#### `pinneapple_adaptation.transfer_learning.trainer`

`TransferTrainer`: copia o modelo de origem, aplica a estratégia de congelamento (`prepare`) e roda o laço de fine-tuning, inclusive com cronograma de descongelamento progressivo.

<small>`pinneapple_adaptation/transfer_learning/trainer.py` · 465 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TransferTrainer` | classe | `(source_model, config) -> None` | Treinador de transferência: copia o modelo de origem, aplica a estratégia (`prepare`), ajusta no problema alvo (`finetune`) ou com descongelamento por época (`progressive_unfreeze`). |

### Exemplos de uso reais do repositório

**Fine-tuning de um PINN num preset de Laplace** — `tests/test_adaptation_and_active_learning.py`

```python
    from pinneapple_adaptation import fine_tune

    spec = laplace_2d_default()
    loss_fn = compile_problem(spec)
    bounds = spec.domain_bounds
    coords = list(spec.coords)
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))

    def target_physics_fn(m, batch):
        x_col = torch.as_tensor(
            np.stack([np.random.uniform(*bounds[c], size=256) for c in coords], axis=1).astype(np.float32)
        )
        x_col.requires_grad_(True)
        y = m(x_col)
        return loss_fn(m, y, _empty_batch(x_col, len(coords), 1))
```

**Meta-treino Reptile sobre Burgers com viscosidade variável** — `tests/test_adaptation_and_active_learning.py`

```python
    sampler = PDETaskSampler(param_ranges={"nu": (0.001, 0.05)},
                              physics_fn_factory=physics_fn_factory, input_dim=2, seed=0)
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))

    trainer = meta_train(model, sampler, algorithm="reptile",
                         n_meta_epochs=5, n_tasks_per_batch=2, n_inner_steps=3, seed=0)
    assert hasattr(trainer, "adapt"), "meta_train() should return a trainer with .adapt()"
    assert len(trainer._history) == 5, "meta_train() must have actually run .train(), not just constructed the trainer"
    assert all(np.isfinite(h["meta_loss"]) for h in trainer._history)

    task = sampler.sample_batch(1)[0]
    adapted = trainer.adapt(task, n_steps=5)
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 11 de 13.
- Funções de teste em arquivos que importam algum módulo do pacote: 154.

Itens do catálogo de métodos (`pinneapple_catalog/method_status.json`) cujo código está neste pacote:

| id | método | status | testes contra referência | outros testes |
|---|---|---|---:|---:|
| T18 | Meta-learning: MAML, Reptile | tested | 0 | 6 |
| T19 | Transfer learning (freeze, adapters, parametric) | tested | 0 | 3 |

### Limitações conhecidas

- O docstring da raiz cita `fast_adapt` no exemplo de uso, mas essa função não existe; o nome real é `meta_adapt`.
- `MAMLTrainer` e `ReptileTrainer` não usam o mesmo contrato de função física: MAML chama `physics_fn(forward_ou_modelo, x)` e Reptile chama `physics_fn(model, batch) -> (perda, dict)`. Uma mesma fábrica só serve aos dois se tratar as duas formas.
- No PyTorch sem `torch.func`, o MAML usa um caminho alternativo (cópia profunda + SGD) que só faz a aproximação de primeira ordem.
- Em `PhysicsTransferAdapter`, `source_spec` e `target_spec` só ficam guardados como referência; não entram no cálculo, a menos que se faça subclasse.
- `ParametricFamilyTransfer.interpolate_weights` não extrapola: fora do intervalo registrado, ou com menos de duas variantes, devolve a variante mais próxima.
- `fine_tune` e `meta_train` tinham bugs (não chamavam `prepare()` / `train()`); foram corrigidos e agora têm teste de regressão, mas o `__init__` de `transfer_learning` e o módulo `freeze` não têm teste direto.

## 5. `pinneapple_arena`

### Objetivo

`pinneapple_arena` é o laboratório de comparação de modelos do PINNeAPPle. A peça central é a classe `Arena`: ela lê uma configuração YAML/JSON (`ArenaConfig`) com um problema físico e uma lista de modelos, treina todos sob o mesmo problema, avalia contra a solução de referência e gera ranking, tabela de métricas e figuras. O problema que resolve é responder, com números do próprio código, qual arquitetura funciona melhor para uma EDP ou um conjunto de dados, sem que cada usuário escreva seu próprio laço de treino e avaliação.

O pacote se apoia no resto da biblioteca: os modelos vêm do `ModelRegistry` de `pinneapple_neural` (via `build_model`), as perdas físicas compiladas vêm de `pinneapple_physics` quando o problema tem `physics_preset`, os dados podem vir de `pinneapple_data` (`DatasetConfig`, `DatasetProblem`, `dataset_bench`), e os passos opcionais de problema inverso e de quantificação de incerteza chamam `pinneapple_analysis`. O pacote é acessível de forma preguiçosa como `pp.arena`, mas não é reexportado em `pinneapple/__init__`.

Além do fluxo padrão (treinar tudo e ordenar por erro relativo), há modos extras: `physics_aware_rank` marca modelos com resíduo de EDP fora do grupo sem reordenar o ranking; o modo de decisão (`decision_mode`) usa `pinneapple_decision` para escolher qual modelo treinar a seguir dentro de um orçamento; `nas` faz busca de arquitetura e hiperparâmetros com Optuna sobre o catálogo existente; `regimes` compara métodos de `pinneapple_analysis.surrogate` por regime de dados; `autoresearch` roda um laço automático de experimentos sobre um único `train.py`; e `custom_builder.define_problem` cria problemas novos a partir de uma EDP em texto.

Dois subpacotes servem aplicativos. `pointwise` contém o protocolo, o catálogo de problemas, os dados de referência, as perdas e os adaptadores do app `apps/pinn_arena` (regressão de campo ponto a ponto). `external` é um catálogo de bibliotecas de terceiros de ML para física (DeepXDE, PINA, NeuralOperator, MACE, etc.): cada entrada é só uma referência (repositório, licença, instalação), e os runners importam a biblioteca real na hora, rodam uma chamada mínima sobre dados sintéticos e informam com honestidade o status (`ok`, `not_installed`, `skipped_heavy`, `manual_install`, `failed`). Nada de código de terceiros é copiado.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 54 | 21.853 | 69 | 243 | 99 (12 arquivos) | 39 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `arena` | Classe `Arena` (orquestrador do benchmark) e as funções de ranking `rank_by_accuracy` e `physics_aware_rank`. | 1 | 3 | 8 |
| `autoresearch` | Laço automático de experimentos que edita só o bloco TUNABLE de um `train.py` e mantém as mudanças que melhoram a métrica. | 1 | 8 | 5 |
| `config` | Dataclasses da configuração da Arena (`ArenaConfig` e seções: problema, modelos, treino, saída, inverso, UQ, dataset, decisão). | 1 | 10 | 13 |
| `custom_builder` | API declarativa `define_problem` para criar problemas a partir de uma EDP escrita como texto. | 1 | 2 | 0 |
| `dataset_bench` | Presets de benchmark por dataset de `pinneapple_data` e a função de uma linha `benchmark_dataset`. | 1 | 3 | 0 |
| `dataset_problems` | `DatasetProblem`: um `ArenaProblem` alimentado por um dataset real de `pinneapple_data`. | 1 | 1 | 0 |
| `decision_mode` | Modo de decisão: o motor de `pinneapple_decision` escolhe qual modelo configurado treinar a seguir. | 1 | 8 | 16 |
| `external` | Catálogo de bibliotecas externas de ML para física, com runners de inferência e de fine-tune de demonstração. | 28 | 186 | 5 |
| `model_factory` | Constrói qualquer modelo suportado a partir de um `ModelConfig`, via `ModelRegistry`. | 1 | 5 | 10 |
| `nas` | Busca de arquitetura e hiperparâmetros (Optuna TPE ou busca aleatória) sobre o catálogo existente. | 1 | 4 | 5 |
| `pointwise` | Protocolo, catálogos, dados de referência, perdas e adaptadores do app `apps/pinn_arena`. | 11 | 35 | 16 |
| `problems` | Registro dos problemas físicos embutidos da Arena, com solução analítica e resíduos PINN. | 1 | 27 | 15 |
| `regimes` | Arena de regimes: compara métodos por regime de dados sob um protocolo comum. | 1 | 3 | 19 |
| `showcase` | Demonstração: quatro arquiteturas PINN no mesmo problema de Burgers, ranqueadas por acurácia e pela visão física. | 1 | 2 | 5 |
| `trainer` | Motor de treino unificado (PINN, supervisionado, operador em grade, grafo), avaliação, UQ e inverso. | 1 | 10 | 25 |
| `visualizer` | Figuras em tema escuro dos resultados da Arena (campos, curvas de perda, tabela, linhas de corrente, UQ). | 1 | 5 | 0 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_arena`

Pacote raiz: reexporta configuração, `Arena`, rankings, problemas, fábrica de modelos, treino, visualização, `define_problem`, `DatasetProblem`, presets de dataset, modo de decisão e NAS.

<small>`pinneapple_arena/__init__.py` · 157 linhas · testes: 52</small>

#### `pinneapple_arena.arena`

Orquestrador principal: prepara dados do problema ou dataset, treina e avalia cada modelo configurado, roda UQ/inverso opcionais, gera figuras e imprime o resumo.

<small>`pinneapple_arena/arena.py` · 628 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Arena` | classe | `(config, device=None)` | Executor do benchmark multimodelo; criado de YAML, JSON ou `ArenaConfig`, com `run`, `run_decision`, `physics_aware_summary`, `results` e `compare`. |
| `rank_by_accuracy` | função | `(train_results, eval_results, field_names) -> List[Tuple[str, float]]` | Ranking padrão: lista `(nome, erro relativo médio)` em ordem crescente, do melhor para o pior. |
| `physics_aware_rank` | função | `(train_results, eval_results, field_names, n_std=2.0, residual_threshold=None) -> Dict[st…)` | Visão opcional que mantém o ranking por acurácia e marca modelos PINN cujo resíduo de EDP é atípico no grupo (mediana + `n_std`·desvio) ou passa de `residual_threshold`. |

#### `pinneapple_arena.autoresearch`

Modo autoresearch (padrão do karpathy/autoresearch): um proponente edita o bloco TUNABLE de `train.py`, cada tentativa roda com orçamento de tempo, e só as mudanças que reduzem a métrica ficam; tudo vai para `results.tsv`.

<small>`pinneapple_arena/autoresearch.py` · 284 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Trial` | classe | `(index, metric, status, kept, description, seconds, changes)` | Registro de uma tentativa: índice, métrica, status (ok, timeout, crash, no_metric), se foi mantida, descrição, duração e mudanças. |
| `RandomSearchProposer` | classe | `(space, seed=0, max_changes=2)` | Proponente sem LLM e reprodutível: sorteia um ou dois parâmetros de um espaço discreto por tentativa. |
| `LLMProposer` | classe | `(provider='anthropic', model=None, llm_call=None, history_len=15, **llm_kwargs)` | Proponente que pede a um LLM (provedor de `pinneapple_llm` ou `llm_call` próprio) a próxima edição do bloco TUNABLE, dado `program.md` e o histórico. |
| `AutoResearch` | classe | `(workdir, proposer, budget_s=300.0, python=sys.executable)` | Laço de experimentos sobre um diretório de trabalho: propõe, aplica, roda, mantém ou reverte, e guarda o melhor script em `best_train.py`. |
| `read_tunables` | função | `(code) -> Dict[str, Any]` | Lê os valores `NOME = valor` do bloco `# >>> TUNABLE` ... `# <<< TUNABLE`; erro se o bloco não existe. |
| `apply_tunables` | função | `(code, changes) -> str` | Reescreve valores só dentro do bloco TUNABLE; nomes desconhecidos são rejeitados. |
| `run_trial` | função | `(workdir, budget_s, *, python=sys.executable, grace_s=30.0)` | Roda `train.py` uma vez num subprocesso com orçamento de tempo e devolve (métrica, status, fim do stdout, segundos). |
| `init_workdir` | função | `(path, *, overwrite=False) -> str` | Cria `program.md` e um `train.py` modelo (PINN de Poisson 1D com erro relativo contra sin(pi x)). |

#### `pinneapple_arena.config`

Configuração estruturada da Arena em dataclasses com `from_dict`, carregável de YAML/JSON e exportável para YAML.

<small>`pinneapple_arena/config.py` · 329 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NetworkConfig` | classe | `(hidden, activation, width, modes, modes2, layers, use_grid, hidden_dim, n_layers, n_mess…)` | Hiperparâmetros de arquitetura (camadas ocultas, ativação, modos FNO, passos de mensagem, `omega_0`, transformer etc.). |
| `TrainingConfig` | classe | `(epochs, lr, weight_decay, grad_clip, optimizer, scheduler, batch_size, seed)` | Hiperparâmetros de treino de um modelo: épocas, taxa de aprendizado, decaimento, clipping, otimizador, scheduler, lote e semente. |
| `ModelConfig` | classe | `(name, type, network, training)` | Especificação completa de um modelo do benchmark: nome, tipo no registro, rede e treino. |
| `ProblemConfig` | classe | `(name, params, physics_preset, grid_n, n_col, n_bc, n_train_supervised, n_mesh_nodes)` | Escolha do problema físico, parâmetros, preset físico opcional e tamanhos de amostragem (grade, colocação, contorno, nós de malha). |
| `OutputConfig` | classe | `(dir, prefix, save_figures, dpi, dark_theme, show)` | Configuração de saída: diretório, prefixo, salvar figuras, dpi, tema escuro e exibição. |
| `InverseConfig` | classe | `(enabled, params, n_obs, noise_std, method, n_iters, lambda_reg, sensor_locations)` | Configuração do problema inverso (parâmetros desconhecidos, observações, ruído, método, iterações, regularização, sensores). |
| `UQConfig` | classe | `(enabled, method, n_samples, dropout_rate, coverage)` | Configuração da quantificação de incerteza (método, amostras, dropout, cobertura). |
| `DatasetConfig` | classe | `(dataset_id, input_fields, output_fields, n_train, n_val, split_seed)` | Usa um dataset de `pinneapple_data` no lugar de um problema embutido (campos de entrada/saída, tamanhos de treino/validação, semente). |
| `DecisionConfig` | classe | `(enabled, budget, stop_when_passed, thresholds, evidence_path, problem)` | Ativa o modo de decisão: orçamento, parar ao passar, limiares, caminho de evidências e fatos do problema. |
| `ArenaConfig` | classe | `(problem, models, output, inverse, uq, dataset, decision)` | Configuração de topo que junta problema, modelos, saída, inverso, UQ, dataset e decisão; lê de dict, YAML ou JSON e grava YAML. |

#### `pinneapple_arena.custom_builder`

API de alto nível para problemas próprios: a EDP vem como texto (`u_xx`, `u_t`, ...), as condições de contorno como dicts, e perdas, amostragem e pipeline são gerados automaticamente. [sem teste]

<small>`pinneapple_arena/custom_builder.py` · 661 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EasyArenaProblem` | classe | `(name, description, coords, fields, pde_evaluator, bcs, params, analytical_fn, bc_weight)` | `ArenaProblem` gerado por `define_problem`, com resíduo montado a partir da EDP em texto e método `solve` que roda a Arena. |
| `define_problem` | função | `(name, coords, fields, pde, bcs, params=None, description='', analytical=None, bc_weight=…)` | Define um problema físico de forma declarativa (coordenadas, campos, EDP em texto, contornos, parâmetros, solução analítica opcional) e o registra. |

#### `pinneapple_arena.dataset_bench`

Tabela `DATASET_PRESETS` com configuração recomendada para cada dataset de `pinneapple_data` e a função `benchmark_dataset` que monta e roda a Arena a partir dela. [sem teste]

<small>`pinneapple_arena/dataset_bench.py` · 723 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `list_benchmarks` | função | `(category=None, verbose=True) -> List[str]` | Imprime e devolve os ids de presets disponíveis, com filtro opcional por categoria. |
| `get_benchmark_preset` | função | `(dataset_id) -> Dict[str, Any]` | Devolve o dict de preset de um `dataset_id`. |
| `benchmark_dataset` | função | `(dataset_id, *, models=None, epochs=None, mode=None, n_train=None, n_bc=None, grid_n=None…)` | Roda um benchmark completo da Arena sobre um dataset de `pinneapple_data`, usando o preset e aceitando substituições (modelos, épocas, modo, tamanhos). |

#### `pinneapple_arena.dataset_problems`

`DatasetProblem` converte datasets de `pinneapple_data` (EDP em grade, séries temporais, regressão, geometria) em problemas da Arena, inclusive o modo híbrido `pinn_data`. [sem teste]

<small>`pinneapple_arena/dataset_problems.py` · 520 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DatasetProblem` | classe | `(dataset_id, name, description, input_fields, output_fields, mode, dataset_params, pinn_r…)` | `ArenaProblem` alimentado por um dataset real; achata grades em `(N, in_dim)` e, no modo `pinn_data`, soma resíduo físico à perda de dados. |

#### `pinneapple_arena.decision_mode`

A Arena como laboratório de decisão: em vez de treinar tudo, o laço DECIDE/EXECUTA/VALIDA/APRENDE de `pinneapple_decision` escolhe o próximo modelo, valida `rel_l2` e resíduo e para ao passar ou ao fim do orçamento.

<small>`pinneapple_arena/decision_mode.py` · 320 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ArenaExecutor` | classe | `(arena)` | Executor do laço de decisão: treina e avalia o modelo escolhido com o próprio código da Arena. |
| `ArenaDecisionReport` | classe | `(budget, stop_reason, ran, not_trained, accepted, ranking, evidence, facts)` | Relatório do modo de decisão: orçamento, motivo da parada, modelos rodados e não treinados, aceito, ranking, evidências e fatos. |
| `catalog_family` | função | `(model_type) -> Optional[str]` | Mapeia um tipo de modelo da Arena para uma chave de `ARCHITECTURE_CATALOG`, ou `None`. |
| `arena_choice` | função | `(arena) -> Tuple[PhysicsChoice, Dict[str, Optional[str]]]` | Monta a `PhysicsChoice` "arena_model" (opções = nomes dos modelos configurados) e a família de cada um; exige ao menos dois modelos. |
| `arena_model_rules` | função | `(families)` | Regras da escolha `arena_model`: pontuação do `ModelSelector` da família, menos uma penalidade para famílias que já falharam. |
| `default_decider` | função | `(families) -> PhysicsDecider` | `PhysicsDecider` padrão com backend de regras (seletores existentes mais as regras de `arena_model`). |
| `arena_problem_facts` | função | `(arena, overrides=None) -> Dict[str, Any]` | Fatos que a Arena realmente conhece sobre o problema; o resto fica como desconhecido, e `overrides` têm prioridade. |
| `run_decision_mode` | função | `(arena, *, budget=3, decider=None, verifier=None, store=None, problem=None, stop_when_pas…)` | Roda o laço de decisão sobre os modelos configurados da Arena e devolve um `ArenaDecisionReport`. |

#### `pinneapple_arena.external`

Catálogo externo: referências a bibliotecas de terceiros de ML para física e a cola que as executa; cada runner importa a biblioteca real e informa um `ModelResult` com status honesto.

<small>`pinneapple_arena/external/__init__.py` · 68 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `licence_counts` | função | `() -> dict[str, int]` | Número de entradas do catálogo por licença ("none found" quando o repositório não tem arquivo de licença). |

#### `pinneapple_arena.external.__main__`

Linha de comando `python -m pinneapple_arena.external` para listar o catálogo e rodar inferência ou fine-tune de demonstração localmente, com saída JSON/CSV. [sem teste]

<small>`pinneapple_arena/external/__main__.py` · 134 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `cmd_list` | função | `(args) -> None` | Subcomando que lista os modelos do catálogo, com filtros por categoria e domínio. |
| `print_result` | função | `(r) -> None` | Imprime um `ModelResult` de forma legível no terminal. |
| `cmd_run` | função | `(args) -> None` | Subcomando que roda inferência ou fine-tune nos modelos selecionados e grava relatório JSON/CSV opcional. |
| `main` | função | `() -> None` | Ponto de entrada da linha de comando do catálogo externo. |

#### `pinneapple_arena.external.arch_meta`

Rótulos das famílias de arquitetura (campo `arch` de `ModelSpec`). [sem teste]

<small>`pinneapple_arena/external/arch_meta.py` · 16 linhas · **sem teste direto**</small>

#### `pinneapple_arena.external.benchmark`

Harness de benchmark/comparação do catálogo externo, chamável pela API para as telas Benchmark e Compare do frontend. [sem teste]

<small>`pinneapple_arena/external/benchmark.py` · 53 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `select_slugs` | função | `(slugs=None, category=None, domain=None) -> list[str]` | Seleciona slugs do catálogo por lista explícita, categoria ou domínio. |
| `run_benchmark` | função | `(slugs, include_heavy=False) -> dict` | Roda a inferência de demonstração em cada slug e agrega os resultados. |
| `run_compare` | função | `(slugs, include_heavy=False) -> dict` | Roda o benchmark dos slugs escolhidos e devolve as especificações junto com resultados e resumo, para comparação lado a lado. |

#### `pinneapple_arena.external.dataset_probe`

Ação "probe" do catálogo de datasets externos: carrega uma amostra real de um subconjunto e relata campos e formas, com o mesmo contrato de honestidade dos modelos. [sem teste]

<small>`pinneapple_arena/external/dataset_probe.py` · 123 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `local_commands` | função | `(spec, subset, split) -> str` | Comandos exatos para baixar e carregar localmente um subconjunto/split de um dataset externo. |
| `probe_dataset` | função | `(spec, subset, split, stream=False, dataset_root=None) -> ModelResult` | Carrega a amostra 0 de um subconjunto/split, de cópia local (`EXTERNAL_CATALOG_DATASET_ROOT`) ou por streaming, e devolve um `ModelResult`. |

#### `pinneapple_arena.external.datasets_data`

Fonte única do catálogo de datasets externos (contraparte de `models_data`). [sem teste]

<small>`pinneapple_arena/external/datasets_data.py` · 87 linhas · **sem teste direto**</small>

#### `pinneapple_arena.external.demo_datasets`

Geradores de dados sintéticos padronizados, um por forma de problema físico, reutilizados pelas demonstrações dos runners (cada `ModelSpec` aponta um via `dataset_fn`). [sem teste]

<small>`pinneapple_arena/external/demo_datasets.py` · 230 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ds_pde_1d` | função | `(n=64) -> dict` | Pontos de colocação 1D sintéticos para PINNs gerais. |
| `ds_burgers` | função | `(nx=64, nt=32) -> dict` | Grade (x, t) e condição inicial sintéticas de Burgers 1D. |
| `ds_grid_2d` | função | `(nx=16, ny=16) -> dict` | Grade 2D genérica para simuladores de campo. |
| `ds_molecular` | função | `(n_atoms=6) -> dict` | Sistema molecular pequeno (posições e espécies) para potenciais de ML. |
| `ds_weather_grid` | função | `(nlat=16, nlon=32, nlevels=4) -> dict` | Grade lat/lon/níveis sintética para modelos de tempo e clima. |
| `ds_em_grid` | função | `(nx=20, ny=20) -> dict` | Grade de permissividade relativa para solvers eletromagnéticos. |
| `ds_seismic` | função | `(nx=32, nz=32) -> dict` | Modelo de velocidade sísmica 2D para propagação de onda. |
| `ds_spin_lattice` | função | `(n_sites=6) -> dict` | Cadeia de spins pequena para estados quânticos neurais. |
| `ds_power_system` | função | `() -> dict` | Sistema de potência de 3 barras para modelos de rede elétrica. |
| `ds_traffic` | função | `(nx=50, nt=50) -> dict` | Grade de densidade de tráfego em (x, t). |
| `ds_epidemic` | função | `(n_days=60) -> dict` | Curva epidêmica SIR diária. |
| `ds_combustion` | função | `(n_species=5, n_t=50) -> dict` | Estado de combustão (espécies ao longo do tempo). |
| `ds_cosmology` | função | `(n_particles=200) -> dict` | Partículas N-corpos para modelos de cosmologia. |
| `ds_room` | função | `() -> dict` | Dimensões de uma sala retangular (ShoeBox) para acústica. |
| `ds_graph_mesh` | função | `(n_nodes=30) -> dict` | Grafo de malha em anel para redes de grafos. |
| `ds_rigidbody` | função | `(n=8) -> dict` | Estado de partículas livres para motores de corpo rígido. |
| `ds_image_field` | função | `(size=32) -> dict` | Campo escalar em imagem quadrada. |
| `ds_timeseries` | função | `(n=100) -> dict` | Série temporal genérica. |
| `ds_pedestrians` | função | `(n=20) -> dict` | Estado de uma multidão de pedestres (posições, velocidades, destinos). |
| `ds_airfoil` | função | `() -> dict` | Descrição de um aerofólio NACA para modelos aerodinâmicos. |
| `ds_option_pricing` | função | `(n=64) -> dict` | Grade de Black-Scholes para precificação de opções. |
| `ds_lorenz` | função | `(n_steps=2000, dt=0.01) -> dict` | Trajetória caótica do Lorenz-63, usada para descobrir (PySINDy) e integrar (Diffrax) a mesma EDO. |
| `ds_tle` | função | `() -> dict` | Conjunto TLE real e válido da ISS (época ilustrativa), copiado de um TLE público, para propagação orbital. |
| `ds_quantum_circuit` | função | `(n_qubits=2) -> dict` | Circuito quântico de par de Bell com poucos qubits. |
| `ds_crystal` | função | `() -> dict` | Célula unitária cúbica sintética em arrays simples (sem pymatgen/ASE); o runner monta a estrutura real se precisar. |

#### `pinneapple_arena.external.domain_meta`

Metadados por domínio físico: rótulo, glifo e descrição. [sem teste]

<small>`pinneapple_arena/external/domain_meta.py` · 56 linhas · **sem teste direto**</small>

#### `pinneapple_arena.external.finetune`

Runners de fine-tune/retreino, um por modelo instalável via pip: modo "finetune" (laço curto e real de gradiente em CPU) ou "reconfigure" (re-roda o solver com parâmetros do usuário, ou só confirma o import em frameworks que exigem checkpoint). [sem teste]

<small>`pinneapple_arena/external/finetune.py` · 1.013 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `finetune_deepxde` | função | `(params) -> ModelResult` | Retreina a FNN do DeepXDE do problema 1D de demonstração por algumas iterações Adam e registra a curva de perda. |
| `finetune_neuraloperator` | função | `(params) -> ModelResult` | Treina um FNO do NeuralOperator numa tarefa de brinquedo de aprender um operador filtro passa-baixa. |
| `finetune_pina` | função | `(params) -> ModelResult` | Treina a `FeedForward` do PINA numa regressão de brinquedo y=x^2. |
| `finetune_sciann` | função | `(params) -> ModelResult` | Treina um `Functional` do SciANN numa regressão de brinquedo y=x^2. |
| `finetune_jax_md` | função | `(params) -> ModelResult` | Roda passos de minimização de energia FIRE do JAX-MD em esferas macias. |
| `finetune_ceviche` | função | `(params) -> ModelResult` | Otimiza a grade de permissividade do Ceviche por subida de gradiente com autograd. |
| `finetune_netket` | função | `(params) -> ModelResult` | Roda iterações VMC do NetKet otimizando um estado quântico neural RBM. |
| `finetune_pinnicle` | função | `(params) -> ModelResult` | Retreina a FNN do PINNICLE (backend DeepXDE) como substituto de um treino real de geleira. |
| `finetune_pybamm` | função | `(params) -> ModelResult` | Re-roda a simulação de descarga SPM do PyBaMM com corrente e duração escolhidas. |
| `finetune_socialforce` | função | `(params) -> ModelResult` | Re-roda o simulador Social Force com o número de pedestres escolhido. |
| `finetune_phiflow` | função | `(params) -> ModelResult` | Monta uma grade do PhiFlow e roda passos de relaxação de brinquedo. |
| `finetune_jax_cfd` | função | `(params) -> ModelResult` | Monta uma grade do JAX-CFD e roda passos de difusão explícita. |
| `finetune_taichi` | função | `(params) -> ModelResult` | Roda lançamentos de um kernel de difusão num campo Taichi. |
| `finetune_warp` | função | `(params) -> ModelResult` | Roda lançamentos de kernel de integração Euler explícita sobre partículas no NVIDIA Warp. |
| `finetune_devito` | função | `(params) -> ModelResult` | Monta uma grade Devito e roda passos explícitos de diferenças finitas. |
| `finetune_deepwave` | função | `(params) -> ModelResult` | Propaga uma onda escalar com Deepwave num modelo de velocidade constante. |
| `finetune_pyroomacoustics` | função | `(params) -> ModelResult` | Reconstrói uma sala ShoeBox com dimensões escolhidas e recalcula a resposta ao impulso. |
| `finetune_ml4gw` | função | `(params) -> ModelResult` | Regera formas de onda IMRPhenomD do ml4gw para as massas escolhidas. |
| `finetune_reservoirflow` | função | `(params) -> ModelResult` | Reconstrói uma grade de reservatório do ReservoirFlow com a permeabilidade escolhida. |
| `finetune_photontorch` | função | `(params) -> ModelResult` | Reconstrói um `Waveguide` do Photontorch com comprimento e índice efetivo escolhidos. |
| `finetune_neuralfoil` | função | `(params) -> ModelResult` | Roda o NeuralFoil num aerofólio NACA com ângulo e Reynolds escolhidos. |
| `finetune_mujoco_mjx` | função | `(params) -> ModelResult` | Roda passos de física MJX num sistema de um corpo com junta livre (pesado: exige `include_heavy`). |
| `finetune_brax` | função | `(params) -> ModelResult` | Só confirma o import do Brax; a nota explica que um fine-tune real seria treinar uma política de RL. |
| `finetune_torax` | função | `(params) -> ModelResult` | Só confirma o import do TORAX; um run real exige um CONFIG de transporte completo. |
| `finetune_physicsnemo` | função | `(params) -> ModelResult` | Treina um FNO do PhysicsNeMo numa tarefa de brinquedo de operador de deslocamento 1D (pesado). |
| `finetune_mace` | função | `(params) -> ModelResult` | Para MACE: só faz o import real e devolve uma nota; o fine-tune real precisa de dados rotulados e da CLI/checkpoint da própria biblioteca (pesado). |
| `finetune_nequip` | função | `(params) -> ModelResult` | Para NequIP: só faz o import real e devolve uma nota; o fine-tune real precisa de dados rotulados e da CLI/checkpoint da própria biblioteca (pesado). |
| `finetune_deepmd_kit` | função | `(params) -> ModelResult` | Para DeePMD-kit: só faz o import real e devolve uma nota; o fine-tune real precisa de dados rotulados e da CLI/checkpoint da própria biblioteca (pesado). |
| `finetune_schnetpack` | função | `(params) -> ModelResult` | Para SchNetPack: só faz o import real e devolve uma nota; o fine-tune real precisa de dados rotulados e da CLI/checkpoint da própria biblioteca (pesado). |
| `finetune_neuralgcm` | função | `(params) -> ModelResult` | Para NeuralGCM: só faz o import real e devolve uma nota; o fine-tune real precisa de dados rotulados e da CLI/checkpoint da própria biblioteca (pesado). |
| `finetune_pysindy` | função | `(params) -> ModelResult` | Reajusta o SINDy do PySINDy com limiar de esparsidade escolhido na trajetória Lorenz-63. |
| `finetune_diffrax` | função | `(params) -> ModelResult` | Reintegra o sistema de Lorenz com o Diffrax para `rho` e tempo final escolhidos. |
| `finetune_tensorcircuit_ng` | função | `(params) -> ModelResult` | Roda descida de gradiente por diferenças finitas num circuito do TensorCircuit-NG. |
| `finetune_dsgp4` | função | `(params) -> ModelResult` | Repropaga o TLE da ISS com dSGP4 em instantes escolhidos. |
| `finetune_chgnet` | função | `(params) -> ModelResult` | Re-roda o CHGNet pré-treinado numa célula cúbica com elemento e parâmetro de rede escolhidos. |
| `run_finetune` | função | `(slug, params=None) -> ModelResult` | Despacha o fine-tune de um slug; sem rotina registrada, devolve `manual_install` (repositório sem pip) ou `finetune_not_supported` com o comando local. |

#### `pinneapple_arena.external.installer`

Utilitários de estado de instalação usados pelos runners e pela API. [sem teste]

<small>`pinneapple_arena/external/installer.py` · 31 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `is_installed` | função | `(module_name) -> bool` | Diz se um módulo existe sem importá-lo (import de bibliotecas pesadas tem efeitos colaterais). |
| `pip_install` | função | `(package, timeout=900) -> tuple[bool, str]` | `pip install` de melhor esforço, usado só quando o usuário pede explicitamente (`install=true`). |

#### `pinneapple_arena.external.local_command`

Monta o comando exato que o usuário roda na própria máquina quando o app não consegue executar um modelo (GPU pesada, repositório só git, ou fine-tune não ligado). [sem teste]

<small>`pinneapple_arena/external/local_command.py` · 37 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `local_command_for` | função | `(spec, action='infer', **params) -> str` | Linha de comando de `python -m pinneapple_arena.external` para inferência ou fine-tune, ou os passos de clone e build de um repositório de instalação manual. |
| `pip_install_command` | função | `(spec) -> str \| None` | Comando `pip install` de um `ModelSpec`, ou `None` se não há pacote pip. |

#### `pinneapple_arena.external.models_data`

Fonte única do catálogo de modelos externos (`MODELS`, `MODELS_BY_SLUG`), gerada uma vez a partir do catálogo HTML original e mantida à mão a partir daí. [sem teste]

<small>`pinneapple_arena/external/models_data.py` · 1.302 linhas · **sem teste direto**</small>

#### `pinneapple_arena.external.result`

Contrato de resultado de todo runner (inferência e fine-tune): `ModelResult`, construtores por status e utilitários. [sem teste]

<small>`pinneapple_arena/external/result.py` · 141 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ModelResult` | classe | `(name, slug, category, domains, repo, status, dataset_name, dataset_description, input_sh…)` | Resultado de um runner: identidade do modelo, status, dataset, formas de entrada/saída, resumo, métricas (inclui `loss_curve` no fine-tune), erro, nota e comando local. |
| `Timer` | classe | `()` | Gerenciador de contexto que mede o tempo decorrido, legível a qualquer momento, inclusive dentro de um `except`. |
| `summarize_array` | função | `(arr) -> dict` | Resumo de um array (média, desvio, mínimo, máximo, tamanho) para o campo `output_summary`. |
| `not_installed` | função | `(name, slug, category, domains, repo, pip_hint, dataset=None, local_command=None) -> Mode…)` | Constrói um `ModelResult` com status `not_installed` e a dica de pip. |
| `manual_install` | função | `(name, slug, category, domains, repo, instructions, dataset=None, local_command=None) -> …)` | Constrói um `ModelResult` com status `manual_install` e as instruções de clone/build. |
| `skipped_heavy` | função | `(name, slug, category, domains, repo, pip_hint, dataset=None, local_command=None) -> Mode…)` | Constrói um `ModelResult` com status `skipped_heavy` (só roda com `include_heavy=True`). |
| `failed` | função | `(name, slug, category, domains, repo, exc, duration, dataset=None) -> ModelResult` | Constrói um `ModelResult` com status `failed` e a exceção real. |
| `ok` | função | `(name, slug, category, domains, repo, dataset, duration, output_shape=None, output_summar…)` | Constrói um `ModelResult` com status `ok`, formas, resumo, nota e métricas da chamada real. |

#### `pinneapple_arena.external.runners`

Subpacote dos runners de inferência por área (arquivo `__init__` vazio). [sem teste]

<small>`pinneapple_arena/external/runners/__init__.py` · 1 linhas · **sem teste direto**</small>

#### `pinneapple_arena.external.runners.applied_domains`

Runners de domínios aplicados (acústica, epidemiologia, tráfego, energia elétrica, ondas gravitacionais, prognóstico, multidões, combustão, aerodinâmica, finanças); a maioria é código de artigo só com instalação manual. Cada função devolve um `ModelResult`. [sem teste]

<small>`pinneapple_arena/external/runners/applied_domains.py` · 236 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `infer_simvascular` | função | `(ds) -> ModelResult` | SimVascular / svFSI: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_pyroomacoustics` | função | `(ds) -> ModelResult` | Pyroomacoustics: monta uma sala ShoeBox com fonte e microfone e calcula a resposta ao impulso. |
| `infer_rirpinn` | função | `(ds) -> ModelResult` | RIRPINN: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_epipinn` | função | `(ds) -> ModelResult` | EpiPINN: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_pise` | função | `(ds) -> ModelResult` | PISE (Physics-Informed Traffic State Estimation): sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_pinn_ac_opf` | função | `(ds) -> ModelResult` | PINN for AC Optimal Power Flow: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_piml_converter` | função | `(ds) -> ModelResult` | PIML Converter: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_ml4gw` | função | `(ds) -> ModelResult` | ml4gw: gera formas de onda IMRPhenomD de binárias de buracos negros numa grade de frequência. |
| `infer_physics_informed_am` | função | `(ds) -> ModelResult` | Physics-Informed AM (hybrid thermal model): sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_pinn_swe_open` | função | `(ds) -> ModelResult` | PINN_SWE_open: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_pinn_corrosion_fatigue` | função | `(ds) -> ModelResult` | PINN for Corrosion-Fatigue Prognosis: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_pinn_wind_bearing` | função | `(ds) -> ModelResult` | PINN for Wind Turbine Main Bearing Fatigue: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_pinnicle` | função | `(ds) -> ModelResult` | PINNICLE: confirma o import com o backend DeepXDE e monta uma FNN pequena sobre dados sintéticos. |
| `infer_physnet_thermal` | função | `(ds) -> ModelResult` | PhysNet Thermal Models: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_neuralsolvers` | função | `(ds) -> ModelResult` | NeuralSolvers: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_pinn_multivp` | função | `(ds) -> ModelResult` | PINN-MULTIVP: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_socialforce` | função | `(ds) -> ModelResult` | Social Force: simula alguns passos de atualização de uma multidão de pedestres. |
| `infer_spdiff` | função | `(ds) -> ModelResult` | SPDiff: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_stiff_pinn` | função | `(ds) -> ModelResult` | Stiff-PINN: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_crnn` | função | `(ds) -> ModelResult` | CRNN (Chemical Reaction Neural Network): sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_neuralfoil` | função | `(ds) -> ModelResult` | NeuralFoil: roda a inferência aerodinâmica num aerofólio NACA 2412. |
| `infer_pinn_option_pricing` | função | `(ds) -> ModelResult` | PINN Option Pricing: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |

#### `pinneapple_arena.external.runners.dynamics_robotics_astro`

Runners de dinâmica não linear e identificação de sistemas, motores diferenciáveis de robótica e modelos astro (fluidos astrofísicos, órbitas, oceano). Cada função devolve um `ModelResult`. [sem teste]

<small>`pinneapple_arena/external/runners/dynamics_robotics_astro.py` · 210 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `infer_pysindy` | função | `(ds) -> ModelResult` | PySINDy: ajusta um modelo SINDy à trajetória Lorenz-63 e relata as equações descobertas. |
| `infer_diffrax` | função | `(ds) -> ModelResult` | Diffrax: integra a EDO de Lorenz-63 com o solver Dopri5. |
| `infer_genesis` | função | `(ds, include_heavy=False) -> ModelResult` | Genesis: só confirma que a biblioteca importa e devolve a versão; a nota explica o que um run real exigiria. Pesado: pulado sem `include_heavy=True`. |
| `infer_nimble_physics` | função | `(ds) -> ModelResult` | Nimble Physics: monta um mundo vazio com gravidade e lê o vetor de estado inicial. |
| `infer_dojo_jl` | função | `(ds) -> ModelResult` | Dojo.jl: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_astronomix` | função | `(ds) -> ModelResult` | astronomix (jf1uids): só confirma que a biblioteca importa e devolve a versão; a nota explica o que um run real exigiria. |
| `infer_dsgp4` | função | `(ds) -> ModelResult` | dSGP4: propaga o TLE da ISS em alguns instantes. |
| `infer_veros` | função | `(ds) -> ModelResult` | Veros: só confirma que a biblioteca importa e devolve a versão; a nota explica o que um run real exigiria. |
| `infer_diffpd` | função | `(ds) -> ModelResult` | DiffPD: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_difftaichi` | função | `(ds) -> ModelResult` | DiffTaichi: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_diffsph` | função | `(ds) -> ModelResult` | diffSPH: só confirma que a biblioteca importa e devolve a versão; a nota explica o que um run real exigiria. |

#### `pinneapple_arena.external.runners.em_photonics`

Runners de eletromagnetismo e fotônica (Ceviche, Photontorch, hPINN). Cada função devolve um `ModelResult`. [sem teste]

<small>`pinneapple_arena/external/runners/em_photonics.py` · 68 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `infer_ceviche` | função | `(ds) -> ModelResult` | Ceviche: resolve `fdfd_ez` com fonte pontual e devolve o campo Ez complexo. |
| `infer_photontorch` | função | `(ds) -> ModelResult` | Photontorch: monta um componente `Waveguide` simples. |
| `infer_hpinn` | função | `(ds) -> ModelResult` | hPINN: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |

#### `pinneapple_arena.external.runners.fluids_structural`

Runners de fluidos e estruturas (simuladores diferenciáveis e redes de grafos de malha); muitos são só instalação manual. Cada função devolve um `ModelResult`. [sem teste]

<small>`pinneapple_arena/external/runners/fluids_structural.py` · 163 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `infer_phiflow` | função | `(ds) -> ModelResult` | PhiFlow: monta uma `CenteredGrid` 2D e lê seus valores. |
| `infer_jax_cfd` | função | `(ds) -> ModelResult` | JAX-CFD: monta uma `Grid`; a nota avisa que um `step()` completo exige campo de velocidade inicial. |
| `infer_jax_fluids` | função | `(ds) -> ModelResult` | JAX-Fluids: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_deepflame` | função | `(ds) -> ModelResult` | DeepFlame: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_fluidx3d` | função | `(ds) -> ModelResult` | FluidX3D: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_lesnets` | função | `(ds) -> ModelResult` | LESnets: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_twophasepinn` | função | `(ds) -> ModelResult` | twophasePINN: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_meshgraphnets_pytorch` | função | `(ds) -> ModelResult` | MeshGraphNets (PyTorch port): sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_meshgraphnets_original` | função | `(ds) -> ModelResult` | MeshGraphNets (original, DeepMind): sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_gns` | função | `(ds) -> ModelResult` | GNS — Graph Network Simulator: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_graingnn` | função | `(ds) -> ModelResult` | GrainGNN: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_taichi` | função | `(ds) -> ModelResult` | Taichi: preenche um campo 2D com um kernel na CPU. |
| `infer_warp` | função | `(ds) -> ModelResult` | NVIDIA Warp: lança um kernel que soma 1 a um array na CPU. |
| `infer_dl4to` | função | `(ds) -> ModelResult` | DL4TO: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |

#### `pinneapple_arena.external.runners.geophysics_energy`

Runners de geofísica e energia (Devito, Deepwave, TORAX, PyBaMM, ReservoirFlow). Cada função devolve um `ModelResult`. [sem teste]

<small>`pinneapple_arena/external/runners/geophysics_energy.py` · 111 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `infer_devito` | função | `(ds) -> ModelResult` | Devito: monta `Grid` e `Function` e carrega o modelo de velocidade. |
| `infer_deepwave` | função | `(ds) -> ModelResult` | Deepwave: monta o tensor do modelo de velocidade pronto para propagação. |
| `infer_torax` | função | `(ds, include_heavy=False) -> ModelResult` | TORAX: só confirma que a biblioteca importa e devolve a versão; a nota explica o que um run real exigiria. Pesado: pulado sem `include_heavy=True`. |
| `infer_pybamm` | função | `(ds) -> ModelResult` | PyBaMM: monta e resolve uma descarga de Single Particle Model. |
| `infer_reservoirflow` | função | `(ds) -> ModelResult` | ReservoirFlow: monta uma grade 2D pequena de reservatório. |

#### `pinneapple_arena.external.runners.materials2_and_misc`

Segunda leva de potenciais de materiais e outros domínios (biomoléculas, circuitos quânticos, operadores neurais novos, esteira de turbinas, incêndios). Cada função devolve um `ModelResult`. [sem teste]

<small>`pinneapple_arena/external/runners/materials2_and_misc.py` · 202 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `infer_openfold` | função | `(ds) -> ModelResult` | OpenFold: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_tensorcircuit_ng` | função | `(ds) -> ModelResult` | TensorCircuit-NG: monta um circuito de preparação GHZ e calcula <Z0> e o estado. |
| `infer_transolver` | função | `(ds) -> ModelResult` | Transolver: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_u_no` | função | `(ds) -> ModelResult` | U-NO: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_torchmd` | função | `(ds) -> ModelResult` | TorchMD: só confirma que a biblioteca importa e devolve a versão; a nota explica o que um run real exigiria. |
| `infer_fairchem` | função | `(ds, include_heavy=False) -> ModelResult` | FAIRChem (Open Catalyst Project): só confirma que a biblioteca importa e devolve a versão; a nota explica o que um run real exigiria. Pesado: pulado sem `include_heavy=True`. |
| `infer_mattersim` | função | `(ds, include_heavy=False) -> ModelResult` | MatterSim: só confirma que a biblioteca importa e devolve a versão; a nota explica o que um run real exigiria. Pesado: pulado sem `include_heavy=True`. |
| `infer_chgnet` | função | `(ds) -> ModelResult` | CHGNet: carrega o checkpoint pré-treinado e prevê energia e forças de uma célula sintética. |
| `infer_orb_models` | função | `(ds) -> ModelResult` | Orb models: só confirma que a biblioteca importa e devolve a versão; a nota explica o que um run real exigiria. |
| `infer_grainlearning` | função | `(ds) -> ModelResult` | GrainLearning: só confirma que a biblioteca importa e devolve a versão; a nota explica o que um run real exigiria. |
| `infer_wakenet` | função | `(ds) -> ModelResult` | WakeNet: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_bpinn_wildfire` | função | `(ds) -> ModelResult` | BPINN-Wildfire: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |

#### `pinneapple_arena.external.runners.materials_md`

Runners de dinâmica molecular e potenciais interatômicos (JAX-MD, MACE, NequIP, Allegro, DeePMD-kit, SchNetPack). Cada função devolve um `ModelResult`. [sem teste]

<small>`pinneapple_arena/external/runners/materials_md.py` · 121 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `infer_jax_md` | função | `(ds) -> ModelResult` | JAX-MD: monta o par de funções de deslocamento em espaço livre e aplica `shift_fn`. |
| `infer_mace` | função | `(ds, include_heavy=False) -> ModelResult` | MACE: só confirma que a biblioteca importa e devolve a versão; a nota explica o que um run real exigiria. Pesado: pulado sem `include_heavy=True`. |
| `infer_nequip` | função | `(ds, include_heavy=False) -> ModelResult` | NequIP: só confirma que a biblioteca importa e devolve a versão; a nota explica o que um run real exigiria. Pesado: pulado sem `include_heavy=True`. |
| `infer_allegro` | função | `(ds) -> ModelResult` | Allegro: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_deepmd_kit` | função | `(ds, include_heavy=False) -> ModelResult` | DeePMD-kit: só confirma que a biblioteca importa e devolve a versão; a nota explica o que um run real exigiria. Pesado: pulado sem `include_heavy=True`. |
| `infer_schnetpack` | função | `(ds, include_heavy=False) -> ModelResult` | SchNetPack: monta um módulo de representação SchNet (pesado: exige `include_heavy`). |

#### `pinneapple_arena.external.runners.pinn_general`

Runners de bibliotecas gerais de PINN e operadores neurais (DeepXDE, PINA, PhysicsNeMo, SciANN, NeuralOperator, DeepONet, PINO). Cada função devolve um `ModelResult`. [sem teste]

<small>`pinneapple_arena/external/runners/pinn_general.py` · 167 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `infer_deepxde` | função | `(ds) -> ModelResult` | DeepXDE: monta um problema de Poisson 1D com FNN pequena, compila e faz `predict` (sem treino). |
| `infer_pina` | função | `(ds) -> ModelResult` | PINA: faz uma passada direta de uma `FeedForward` pequena. |
| `infer_physicsnemo` | função | `(ds, include_heavy=False) -> ModelResult` | NVIDIA PhysicsNeMo: só confirma que a biblioteca importa e devolve a versão; a nota explica o que um run real exigiria. Pesado: pulado sem `include_heavy=True`. |
| `infer_sciann` | função | `(ds) -> ModelResult` | SciANN: monta um `SciModel` pequeno e faz `predict`. |
| `infer_neuralpde_jl` | função | `(ds) -> ModelResult` | NeuralPDE.jl: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_pinns_torch` | função | `(ds) -> ModelResult` | PINNs-Torch: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_pinns_original` | função | `(ds) -> ModelResult` | PINNs (original reference code): sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_pinn_tfi_hss` | função | `(ds) -> ModelResult` | PINN_TFI-HSS: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_neuraloperator` | função | `(ds) -> ModelResult` | NeuralOperator: faz uma passada direta de um FNO 1D pequeno. |
| `infer_deeponet_original` | função | `(ds) -> ModelResult` | DeepONet (original): sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_deeponet_fno_benchmark` | função | `(ds) -> ModelResult` | DeepONet-FNO benchmark: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_pino` | função | `(ds) -> ModelResult` | PINO (Physics-Informed Neural Operator): sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |

#### `pinneapple_arena.external.runners.quantum_cosmology`

Runners de física quântica e cosmologia (NetKet, FermiNet, FlowPM/JaxPM, CAMELS). Cada função devolve um `ModelResult`. [sem teste]

<small>`pinneapple_arena/external/runners/quantum_cosmology.py` · 68 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `infer_netket` | função | `(ds) -> ModelResult` | NetKet: monta um estado quântico neural RBM com amostrador Metropolis e tira amostras MC. |
| `infer_ferminet` | função | `(ds) -> ModelResult` | FermiNet: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_flowpm_jaxpm` | função | `(ds) -> ModelResult` | FlowPM / JaxPM: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_camels` | função | `(ds) -> ModelResult` | CAMELS: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |

#### `pinneapple_arena.external.runners.registry`

Tabela de despacho que liga cada slug ao par (gerador de dados, função `infer_*`), usada pela API para inferência, benchmark e comparação. [sem teste]

<small>`pinneapple_arena/external/runners/registry.py` · 268 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `call_infer` | função | `(slug, include_heavy=False) -> ModelResult` | Monta o dataset de demonstração e roda a função `infer_*` real do slug, passando `include_heavy` só a quem aceita; resultados não-`ok` recebem o comando local. |
| `get_dataset_preview` | função | `(slug) -> dict` | Metadados e prévia JSON dos arrays do dataset de demonstração de um slug, sem rodar inferência. |

#### `pinneapple_arena.external.runners.rigidbody_robotics`

Runners de corpo rígido e robótica (Brax, MuJoCo MJX), ambos pesados. Cada função devolve um `ModelResult`. [sem teste]

<small>`pinneapple_arena/external/runners/rigidbody_robotics.py` · 65 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `infer_brax` | função | `(ds, include_heavy=False) -> ModelResult` | Brax: só confirma que a biblioteca importa e devolve a versão; a nota explica o que um run real exigiria. Pesado: pulado sem `include_heavy=True`. |
| `infer_mujoco_mjx` | função | `(ds, include_heavy=False) -> ModelResult` | MuJoCo MJX: monta um modelo de um corpo e roda um `mjx.forward()` (pesado: exige `include_heavy`). |

#### `pinneapple_arena.external.runners.weather_climate`

Runners de tempo e clima (GraphCast, Pangu-Weather, FourCastNet só manuais; NeuralGCM pesado). Cada função devolve um `ModelResult`. [sem teste]

<small>`pinneapple_arena/external/runners/weather_climate.py` · 62 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `infer_graphcast` | função | `(ds) -> ModelResult` | GraphCast: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_pangu_weather` | função | `(ds) -> ModelResult` | Pangu-Weather: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_fourcastnet` | função | `(ds) -> ModelResult` | FourCastNet: sem pacote pip utilizável aqui; só devolve `manual_install` com as instruções de instalação, sem executar nada. |
| `infer_neuralgcm` | função | `(ds, include_heavy=False) -> ModelResult` | NeuralGCM: só confirma que a biblioteca importa e devolve a versão; a nota explica o que um run real exigiria. Pesado: pulado sem `include_heavy=True`. |

#### `pinneapple_arena.external.schema`

Modelo de dados do catálogo externo: `ModelSpec`, `DatasetSpec` e metadados de domínio e família de arquitetura. [sem teste]

<small>`pinneapple_arena/external/schema.py` · 111 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ModelSpec` | classe | `(slug, name, org, arch, domains, desc, tags, repo, ref, pip_package, import_name, heavy, …)` | Uma entrada do catálogo de modelos: slug, nome, organização, família, domínios, repositório, pacote pip, nome de import, se é pesado ou só manual, gerador de dados, licença e citação. |
| `DomainMeta` | classe | `(key, label, glyph, desc)` | Metadados de um domínio físico: chave, rótulo, glifo e descrição. |
| `ArchMeta` | classe | `(key, label)` | Metadados de uma família de arquitetura: chave e rótulo. |
| `SubDataset` | classe | `(name, label, domains, dims, desc)` | Um subconjunto de um dataset externo: nome, rótulo, domínios, dimensões e descrição. |
| `DatasetSpec` | classe | `(slug, name, org, desc, tags, repo, ref, size, subsets, splits, pip_package, import_name,…)` | Uma entrada do catálogo de datasets externos: repositório, tamanho, subconjuntos, splits, pacote, streaming, comando de download e licença. |

#### `pinneapple_arena.external.status`

Checagem barata de estado de instalação, sem inferência, para o selo de cada cartão do catálogo. [sem teste]

<small>`pinneapple_arena/external/status.py` · 29 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `install_state` | função | `(spec) -> str` | Estado de instalação de um `ModelSpec` (`installed`, `not_installed`, `heavy` ou `manual_only`), sem importar nem rodar o modelo. |

#### `pinneapple_arena.model_factory`

Constrói qualquer modelo suportado pela Arena a partir de um `ModelConfig`, via `ModelRegistry.build()` de `pinneapple_neural`, com construtores fixos de reserva para os tipos mais comuns.

<small>`pinneapple_arena/model_factory.py` · 324 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `is_graph_model` | função | `(cfg) -> bool` | Diz se o tipo do modelo é de grafo (MeshGraphNet, GNN etc.). |
| `is_pinn_model` | função | `(cfg) -> bool` | Diz se o tipo do modelo é da família PINN (treinado por resíduo físico). |
| `is_inverse_model` | função | `(cfg) -> bool` | Diz se o tipo do modelo é de problema inverso. |
| `is_operator_model` | função | `(cfg) -> bool` | Diz se o tipo do modelo é um operador neural (FNO, DeepONet etc.). |
| `build_model` | função | `(cfg, in_dim=2, out_dim=1, edge_in_dim=0) -> nn.Module` | Instancia o modelo de um `ModelConfig` com as dimensões de entrada/saída (e de aresta para grafos). |

#### `pinneapple_arena.nas`

Busca conjunta de arquitetura, hiperparâmetros e taxa de aprendizado sobre o catálogo do `ModelRegistry`, com Optuna (TPE + poda pela mediana) ou busca aleatória; cada tentativa treina um modelo do zero, não é NAS diferenciável.

<small>`pinneapple_arena/nas.py` · 447 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ArchitectureCandidate` | classe | `(name, param_space, build_kwargs_fn)` | Uma arquitetura buscável: chave no `ModelRegistry`, seu espaço de hiperparâmetros e a função que monta os kwargs de construção. |
| `ArchitectureSearchSpace` | classe | `(candidates, lr_param_space)` | Arquiteturas candidatas, com o espaço de cada uma e uma faixa de taxa de aprendizado comum. |
| `NASResult` | classe | `(best_architecture, best_hyperparams, best_score, all_trials, study)` | Resultado da busca: melhor arquitetura, melhores hiperparâmetros, melhor pontuação, todas as tentativas e o `optuna.Study` bruto. |
| `search_architecture` | função | `(problem, search_space=None, n_trials=20, physics_aware=False, lam=1.0, epochs_per_trial=…)` | Roda a busca sobre um `ArenaProblem`, treinando cada tentativa com `train_pinn` e pontuando pelo erro (mais resíduo físico se `physics_aware`). |

#### `pinneapple_arena.pointwise`

Benchmarks de campo ponto a ponto do app `apps/pinn_arena`: coordenadas `(N, in_dim)` entram, campos `(N, out_dim)` saem; reúne catálogo de problemas, dados de referência, perdas, adaptadores, treino e leaderboard.

<small>`pinneapple_arena/pointwise/__init__.py` · 71 linhas · testes: 16</small>

#### `pinneapple_arena.pointwise.adapters`

Adaptadores que fazem arquiteturas de grade ou sequência aceitarem dados ponto a ponto, tratando cada ponto como amostra própria (`PointwiseAdapter`, `ClosedFormAdapter`, lista `NOT_APPLICABLE`).

<small>`pinneapple_arena/pointwise/adapters.py` · 181 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ArchitectureNotApplicable` | classe | `()` | Exceção: a arquitetura não pode ser treinada em dados coordenada→campo (por exemplo modelos não supervisionados de trajetória). |
| `adapt_model` | função | `(arch_id, model, in_dim, out_dim) -> Any` | Envolve o modelo para aceitar `(N, in_dim)` e devolver `(N, out_dim)`. |
| `check_applicable` | função | `(arch_id) -> None` | Levanta `ArchitectureNotApplicable` se a arquitetura está na lista de não aplicáveis. |

#### `pinneapple_arena.pointwise.architectures`

Catálogo `ARCHITECTURES_CATALOG` das arquiteturas do registro que o app pode escolher, com descrições; algumas entradas têm chave renomeada no registro. [sem teste]

<small>`pinneapple_arena/pointwise/architectures.py` · 1.722 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `registry_key` | função | `(arch_id) -> str` | Chave de `pp.build_model` de uma entrada do catálogo (trata os renomes `symm_*`→`noether_*` e `icosahedral_gnn`→`graphcast`). |
| `catalog_status` | função | `() -> dict[str, bool]` | `{id: registrado}`: se o registro do PINNeAPPle tem o modelo de cada entrada. |

#### `pinneapple_arena.pointwise.datasets`

Dados de referência dos problemas do app: solução analítica quando existe, senão solução numérica própria; alguns campos (colisão, aerofólio 3D, esteira de cilindro) são ilustrativos ou de ordem reduzida, e o metadado diz isso. [sem teste]

<small>`pinneapple_arena/pointwise/datasets.py` · 3.180 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DatasetGenerator` | classe | `(problem_id, parameters)` | Gera a solução de referência de um problema: `generate(n)` devolve `coords`, `solution` e `metadata` com a origem da solução. |

#### `pinneapple_arena.pointwise.geometries`

Catálogo `GEOMETRIES_CATALOG` de geometrias 3D prontas para experimentos 3D. [sem teste]

<small>`pinneapple_arena/pointwise/geometries.py` · 503 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `get_geometry_mesh_config` | função | `(geometry_id, resolution) -> dict` | Dict de configuração de malha de um preset de geometria numa resolução, compatível com os construtores de geometria. |

#### `pinneapple_arena.pointwise.leaderboard`

Ranking global de arquiteturas sobre muitos experimentos do app, com métricas médias e pontuações 0-100 por objetivo. [sem teste]

<small>`pinneapple_arena/pointwise/leaderboard.py` · 212 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `aggregate` | função | `(experiments, families=None) -> dict` | Agrega experimentos concluídos numa entrada por arquitetura, ordenada por L2 relativo médio; custo, memória e velocidade são relativos ao conjunto da chamada. |

#### `pinneapple_arena.pointwise.losses`

Perdas de resíduo de EDP dos problemas do app: primeiro resíduo nativo em autograd com os parâmetros do experimento, depois o preset compilado de `pinneapple_physics`, senão nenhuma (só dados). [sem teste]

<small>`pinneapple_arena/pointwise/losses.py` · 732 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsLossGenerator` | classe | `(problem_id, parameters)` | Compila a perda física (resíduo de EDP) de um problema e descreve qual fonte foi usada. |

#### `pinneapple_arena.pointwise.problems`

Catálogo `PROBLEMS_CATALOG` de problemas físicos do app, com equações, parâmetros, condições de contorno/iniciais e a chave de preset quando existe. [sem teste]

<small>`pinneapple_arena/pointwise/problems.py` · 1.875 linhas · **sem teste direto**</small>

#### `pinneapple_arena.pointwise.schema`

Descritores pydantic v2 dos catálogos do app: problemas, presets de geometria e arquiteturas. [sem teste]

<small>`pinneapple_arena/pointwise/schema.py` · 175 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsArea` | classe | `()` | Enum das áreas físicas dos problemas. |
| `Dimension` | classe | `()` | Enum da dimensão do problema (1D, 2D, 3D, com ou sem tempo). |
| `DifficultyLevel` | classe | `()` | Enum do nível de dificuldade do problema. |
| `ArchitectureFamily` | classe | `()` | Enum das famílias de arquitetura do catálogo. |
| `GeometryType` | classe | `()` | Enum dos tipos de geometria 3D. |
| `ProblemParameter` | classe | `(name, default, min_val, max_val, description, unit)` | Parâmetro escalar configurável de um problema: nome, padrão, limites, descrição e unidade. |
| `ColumnHints` | classe | `(coord_cols, output_cols, time_col, csv_header, notes)` | Layout esperado de colunas de um dataset do problema, para guiar uploads CSV/NPZ/JSON. |
| `PhysicsProblem` | classe | `(id, name, description, domain, dimension, equations, applications, initial_conditions, b…)` | Descritor completo de um problema físico do catálogo (equações, aplicações, condições, parâmetros, dificuldade, preset). |
| `GeometryPreset` | classe | `(id, name, description, geometry_type, dimensions, supported_problems, mesh_resolution_de…)` | Geometria 3D pré-montada disponível para experimentos 3D. |
| `MeshInfo` | classe | `(n_points, n_elements, bounds, quality_score)` | Estatísticas resumidas de uma malha gerada (pontos, elementos, limites, qualidade). |
| `ArchitectureInfo` | classe | `(id, name, family, description, detailed_description, supports_physics_loss, supports_3d,…)` | Metadados de uma arquitetura do catálogo (família, descrição, suporte a perda física e 3D, uso, referência). |

#### `pinneapple_arena.pointwise.tables`

Leitura de tabelas numéricas enviadas ao app (CSV/TSV/TXT com separador detectado, JSON de registros, NPZ).

<small>`pinneapple_arena/pointwise/tables.py` · 111 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `parse_numeric_table` | função | `(filename, content) -> tuple[list[str], np.ndarray, int]` | Devolve `(colunas, array float32, linhas puladas)` de uma tabela enviada, com formato pelo sufixo do arquivo. |

#### `pinneapple_arena.pointwise.training`

Protocolo de treino e pontuação de uma arquitetura no app: perda de dados mais resíduo de EDP balanceado em pontos de colocação, LR cosseno, parada antecipada na validação e nota no teste; nunca treina um substituto com o nome da arquitetura. [sem teste]

<small>`pinneapple_arena/pointwise/training.py` · 706 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ArchitectureUnavailable` | classe | `()` | Exceção: o registro não conseguiu construir a arquitetura para os dados deste problema; ela é reportada como falha. |
| `best_device` | função | `() -> str` | Melhor dispositivo disponível (cuda > mps > cpu) como string. |
| `device_info` | função | `() -> dict` | Dispositivo de cálculo e, para CUDA, nome, memória e versão. |
| `field_metrics` | função | `(y_pred, y_true) -> dict` | Métricas de erro da previsão contra a referência, agrupadas sobre pontos e campos (`l2_relative`, `l1`, etc.). |
| `prediction_quality` | função | `(y_pred, y_true, metrics) -> dict` | Heurística que detecta saídas degeneradas (constante, zero, NaN) e dá uma nota de qualidade; não é checagem física. |
| `problem_in_dim` | função | `(problem_id, parameters=None) -> int` | Número de coordenadas de entrada de um problema. |
| `problem_out_dim` | função | `(problem_id) -> int` | Número de campos de saída de um problema. |
| `instantiate_model` | função | `(arch_id, in_dim, out_dim) -> Any` | Constrói a arquitetura pelo registro (`pp.build_model`) para `(N, in_dim) -> (N, out_dim)`, com adaptadores quando preciso. |
| `model_forward` | função | `(model, X_t) -> Any` | Passada direta unificada que lida com os vários tipos de retorno dos modelos. |
| `train_torch_model` | função | `(model, X_train, y_train, X_val, y_val, X_col, physics_loss_fn, n_epochs, lr, device, pro…)` | Treina um modelo torch com perda de dados e resíduo físico opcional, parada antecipada e restauração dos melhores pesos. |
| `train_closed_form_model` | função | `(model, X_train, y_train, X_val, y_val, progress_callback=None) -> dict[str, Any]` | Treina por `fit()` um modelo de `ClosedFormAdapter` (ELM, RBF híbrida, leitura de ESN). |
| `run_inverse_solver` | função | `(model, problem_id, inverse_cfg, coords, solution, device) -> dict` | Envolve o modelo no `InverseProblemSolver` e chama `invert()` para recuperar parâmetros da EDP. |
| `predict` | função | `(model, X, device='cpu') -> tuple[np.ndarray, float]` | Previsão de um modelo treinado sem gradientes e o tempo em milissegundos. |
| `rank_models` | função | `(results) -> list[dict]` | Ordena resultados concluídos por L2 relativo (rank 1..n); os que falharam ficam depois, com rank `None`. |

#### `pinneapple_arena.problems`

Registro dos problemas físicos embutidos da Arena, cada um com solução analítica, resíduos PINN e dados supervisionados; a classe base `ArenaProblem` levanta `NotImplementedError` em `pinn_residuals` e `supervised_data`.

<small>`pinneapple_arena/problems.py` · 1.206 linhas · testes: 15 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ArenaProblem` | classe | `(name, description, domain, input_dim, output_dim, physics_preset)` | Classe base de problema: `analytical`, `pinn_residuals`, `supervised_data` (abstratos, `NotImplementedError`), `compiled_losses` e `to_problem_spec` via preset de `pinneapple_physics`. |
| `KovasznayNS` | classe | `()` | Escoamento de Kovasznay (Navier-Stokes 2D estacionário) com solução analítica, campos u, v, p. |
| `Stokes2D` | classe | `()` | Escoamento de Stokes 2D (Poiseuille) com solução analítica u = 1-y², v = 0, p = -2x. |
| `Heat1D` | classe | `()` | Equação do calor 1D com solução analítica exponencial-senoidal. |
| `Heat2D` | classe | `()` | Equação do calor 2D com solução analítica. |
| `ConvectionDiffusion1D` | classe | `()` | Convecção-difusão 1D estacionária com solução analítica de camada-limite. |
| `DarcyFlow2D` | classe | `()` | Darcy 2D (Poisson para a pressão) com solução analítica sin(πx)sin(πy). |
| `Laplace2D` | classe | `()` | Laplace 2D com solução analítica sin(πx)sinh(πy)/sinh(π). |
| `Poisson2D` | classe | `()` | Poisson 2D com solução analítica sin(πx)sin(πy). |
| `Helmholtz2D` | classe | `()` | Helmholtz 2D com solução fabricada sin(πx)sin(πy). |
| `Biharmonic2D` | classe | `()` | Equação biharmônica 2D (placa fina) com solução fabricada. |
| `AdvectionDiffusion1D` | classe | `()` | Advecção-difusão 1D transiente com solução analítica. |
| `Wave1D` | classe | `()` | Onda 1D com solução analítica sin(πx)cos(πct). |
| `KleinGordon1D` | classe | `()` | Klein-Gordon 1D linear com solução analítica cos(kx)cos(ωt). |
| `NLS1D` | classe | `()` | Schrödinger não linear 1D (sóliton brilhante), resolvido nas partes real e imaginária. |
| `Burgers1D` | classe | `()` | Burgers 1D viscoso com referência exata pela transformação de Cole-Hopf (quadratura de Gauss-Hermite). |
| `KdV1D` | classe | `()` | Korteweg-de Vries 1D com solução de sóliton. |
| `AllenCahn` | classe | `()` | Allen-Cahn 1D (campo de fase) com interface em tanh; a classe é definida duas vezes no arquivo e só a segunda definição é registrada. |
| `FisherKPP1D` | classe | `()` | Fisher-KPP 1D com onda viajante exata (o docstring a chama de Ablowitz-Zeidler). |
| `FitzHughNagumo1D` | classe | `()` | FitzHugh-Nagumo 1D (meio excitável, campos v e w). |
| `LinearElasticity2D` | classe | `()` | Elasticidade linear 2D em deformação plana com solução fabricada. |
| `BlackScholes1D` | classe | `()` | EDP de Black-Scholes 1D para opção de compra europeia, referência pela fórmula BS. |
| `AllenCahn` | classe | `()` | Allen-Cahn 1D (campo de fase) com interface em tanh; a classe é definida duas vezes no arquivo e só a segunda definição é registrada. |
| `get_problem` | função | `(name) -> ArenaProblem` | Devolve a instância registrada de um problema pelo nome; `KeyError` com a lista se não existe. |
| `register_problem` | função | `(problem) -> 'ArenaProblem'` | Registra uma instância própria de `ArenaProblem` para a Arena achá-la pelo nome. |
| `list_problems` | função | `() -> List[str]` | Nomes dos problemas registrados, em ordem alfabética. |
| `list_problems_by_domain` | função | `() -> Dict[str, List[str]]` | Problemas registrados agrupados pelo domínio físico. |

#### `pinneapple_arena.regimes`

Arena de regimes: treina cada método de `pinneapple_analysis.surrogate.methods` com mesmos dados, mesmo orçamento, mesma regra de parada e várias sementes, e mede por regime (interpolação, poucos dados, extrapolação, mudança de geometria, frentes íngremes) mais custo e violações físicas; regime impossível é marcado indisponível.

<small>`pinneapple_arena/regimes.py` · 387 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ArenaResult` | classe | `(problem, protocol, regimes, methods, rankings, overall, baselines, audit, created, hardw…)` | Resultado da arena de regimes: problema, protocolo, regimes, métodos, rankings, visão geral, baselines, auditoria, hardware e tempo; exporta dict e Markdown. |
| `arena_markdown` | função | `(r, title='') -> str` | Relatório comparativo em Markdown a partir de `ArenaResult.to_dict()` ou de um arquivo de resultado gravado. |
| `run_regime_arena` | função | `(data, *, methods='auto', budget=10.0, seeds=3, regimes='auto', few_data_fraction=0.25, s…)` | Roda a arena de regimes sobre os dados e devolve um `ArenaResult`. |

#### `pinneapple_arena.showcase`

Demonstração com quatro arquiteturas PINN contrastantes no mesmo problema `burgers_1d`, ranqueadas por acurácia e pela visão física da Arena; usa só peças da própria Arena.

<small>`pinneapple_arena/showcase.py` · 158 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `field_fit_slice` | função | `(xy_eval, y_ref, y_pred, grid_n) -> dict[str, Any]` | Referência e previsão na última linha de tempo da grade de avaliação (t = 1 para `burgers_1d`). |
| `run_benchmark` | função | `(problem_name=PROBLEM_NAME, architectures=None, seed=None, progress_cb=None, n_col=N_COL,…)` | Treina cada arquitetura nos mesmos pontos de colocação, avalia, ranqueia por acurácia e por `physics_aware_rank`, reportando progresso. |

#### `pinneapple_arena.trainer`

Motor de treino unificado: PINN (resíduo próprio ou perdas compiladas), supervisionado, operador em grade completa, grafo, mais avaliação, UQ e problema inverso via `pinneapple_analysis`.

<small>`pinneapple_arena/trainer.py` · 616 linhas · testes: 25</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TrainResult` | classe | `(name, model, train_losses, train_time, metrics, physics_residual, uq_result, inverse_res…)` | Resultado de treino de um modelo: modelo, perdas, tempo, métricas, resíduo físico, resultados de UQ e inverso, extras. |
| `train_pinn` | função | `(model, cfg, pinn_residuals_fn, xy_int, xy_bc, uv_bc, problem_params, device='cpu', log_i…)` | Treina um modelo PINN com resíduo de EDP em pontos internos e perda de contorno. |
| `train_supervised` | função | `(model, cfg, X_train, Y_train, device='cpu', log_interval=100) -> TrainResult` | Treina um modelo com MSE em pares (entrada, alvo). |
| `grid_layout` | função | `(x) -> Tuple[np.ndarray, int, int]` | Ordem das linhas de `x` que forma uma grade completa (y-major); `ValueError` se os pontos não formam grade. |
| `train_grid_operator` | função | `(model, cfg, X_grid, Y_grid, device='cpu', log_interval=100) -> TrainResult` | Treina um modelo tipo FNO sobre campos inteiros: cada passo vê a grade de treino completa e ordenada. |
| `train_graph` | função | `(model, cfg, node_feats, edge_index, edge_attr, node_targets, device='cpu', log_interval=…)` | Treina um modelo de grafo de forma supervisionada nos nós. |
| `evaluate_model` | função | `(result, cfg, xy_eval, Y_ref, field_names, device='cpu', node_positions=None, edge_index=…)` | Avalia um modelo treinado nos pontos de avaliação contra a referência e devolve métricas por campo. |
| `run_uq` | função | `(result, xy_eval, uq_cfg, device='cpu') -> Optional[Any]` | Roda quantificação de incerteza com `pinneapple_analysis.uq_predict`. |
| `run_inverse` | função | `(result, xy_eval, Y_ref, inv_cfg, device='cpu') -> Optional[Any]` | Roda o problema inverso com `pinneapple_analysis.invert`. |
| `load_pinneapple_dataset` | função | `(dataset_id, input_fields, output_fields, n_train=1000, n_val=200, split_seed=42)` | Carrega um dataset de `pinneapple_data` e divide em arrays de treino e validação. |

#### `pinneapple_arena.visualizer`

Figuras em tema escuro dos resultados da Arena, salvas como PNG no diretório de saída. [sem teste]

<small>`pinneapple_arena/visualizer.py` · 452 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_field_comparison` | função | `(eval_results, field_names, xy_eval, grid_n, problem_name, dark_theme=True, dpi=150, save…)` | Painel de campos: referência, cada modelo e o erro. |
| `plot_loss_curves` | função | `(train_results, dark_theme=True, dpi=150, save_path=None, show=False)` | Curvas de perda de treino de cada modelo. |
| `plot_metrics_table` | função | `(eval_results, field_names, train_results, dark_theme=True, dpi=150, save_path=None, show…)` | Tabela de métricas por modelo e campo, como figura. |
| `plot_streamlines` | função | `(eval_results, xy_eval, grid_n, u_field='u', v_field='v', field_names=None, problem_name=…)` | Linhas de corrente do campo de velocidade previsto por cada modelo. |
| `plot_uq` | função | `(uq_result, xy_eval, field_names, grid_n=None, title='UQ', dark_theme=True, dpi=150, save…)` | Média ± desvio de um `UQResult` de `pinneapple_analysis`. |

### Exemplos de uso reais do repositório

**Arena a partir de configuração programática** — `examples/arena_pipelines/04_arena_from_yaml.py`

```python
from pinneapple_arena import Arena, ArenaConfig

BUILTIN_CONFIGS = {
    "kovasznay_ns": {
        "problem": {
            "name": "kovasznay_ns",
            "params": {"re": 40.0},
            "grid_n": 40, "n_col": 2000, "n_bc": 500,
            "n_train_supervised": 200, "n_mesh_nodes": 600,
        },
        "models": [
            {"name": "VanillaPINN", "type": "vanilla_pinn",
             "network": {"hidden": [128, 128, 128, 128], "activation": "tanh"},
             "training": {"epochs": 2000, "lr": 5e-4, "grad_clip": 1.0}},
```

**Problema próprio com define_problem** — `pinneapple_arena/custom_builder.py`

```python
from pinneapple_arena import define_problem
import numpy as np

prob = define_problem(
    name="reaction_diffusion",
    description="eps·u'' − k·u = f  on [0,1],  u(0)=u(1)=0",
    coords={"x": (0.0, 1.0)},
    fields=["u"],
    pde="eps * u_xx - k * u + (eps * pi**2 + k) * sin(pi * x)",
    bcs=[
        {"type": "dirichlet", "at": "x_min", "field": "u", "value": 0.0},
        {"type": "dirichlet", "at": "x_max", "field": "u", "value": 0.0},
    ],
    params={"eps": 0.1, "k": 1.0},
    analytical=lambda x, eps=0.1, k=1.0: {"u": np.sin(np.pi * x)},
)
prob.solve(models=["VanillaPINN", "SIREN"], epochs=3000)
```

**Autoresearch com proponente aleatório** — `pinneapple_arena/autoresearch.py`

```python
from pinneapple_arena.autoresearch import AutoResearch, RandomSearchProposer, init_workdir
init_workdir("runs/poisson")  # program.md + train.py (1D Poisson PINN, rel-L2 vs sin(pi x))
AutoResearch("runs/poisson", RandomSearchProposer({"LR": [1e-3, 3e-3, 1e-2], "WIDTH": [16, 32, 64]}),
             budget_s=60).run(max_trials=20)
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 15 de 54.
- Funções de teste em arquivos que importam algum módulo do pacote: 99.

Itens do catálogo de métodos (`pinneapple_catalog/method_status.json`) cujo código está neste pacote:

| id | método | status | testes contra referência | outros testes |
|---|---|---|---:|---:|
| T15 | Adaptive hyperparameter sweep / architecture search (arena NAS) | untested | 0 | 0 |
| T21 | Autoresearch loop (fixed-budget trials, keep/revert, random or LLM proposer) | tested | 0 | 2 |

### Limitações conhecidas

- Muitos módulos não têm teste direto: `custom_builder`, `dataset_bench`, `dataset_problems`, `visualizer`, quase todo o subpacote `external` e a maior parte de `pointwise` (só `adapters` e `tables` têm).
- O catálogo `external` depende de bibliotecas opcionais de terceiros: sem elas o resultado é `not_installed`; entradas pesadas só rodam com `include_heavy=True`; muitos runners apenas devolvem `manual_install` (código de artigo só via git, Julia, C++) ou só confirmam o import, sem inferência real.
- Os runners de fine-tune do catálogo externo são demonstrações curtas em CPU sobre dados de brinquedo; no modo "reconfigure" não há gradiente, e para MACE, NequIP, DeePMD-kit, SchNetPack e NeuralGCM só há import real.
- Em `problems.py`, o docstring cita `lid_driven_cavity_ns`, mas esse problema não é registrado; a classe `AllenCahn` aparece duas vezes (a primeira com um alias `AreraProblem`) e só a segunda vale.
- A classe base `ArenaProblem` levanta `NotImplementedError` em `pinn_residuals` e `supervised_data`: problemas próprios precisam implementá-los (ou usar `define_problem`).
- `nas` não é NAS diferenciável: cada tentativa treina um modelo do zero, e sem Optuna cai para busca aleatória.
- Em `pointwise.datasets`, alguns campos de referência (colisão, aerofólio 3D, esteira de cilindro) são ilustrativos ou de ordem reduzida, não soluções das equações completas; `prediction_quality` é heurística, não checagem física.
- No `leaderboard`, as notas de custo, memória e velocidade são relativas às entradas da mesma chamada e dependem do hardware; o L2 médio mistura problemas diferentes se não for filtrado.

## 6. `pinneapple_worldmodel`

### Objetivo

O `pinneapple_worldmodel` tenta montar um "modelo de mundo" físico: uma rede que aprende `f(state_t, descritor) -> state_{t+1}`, onde o estado é um campo espacial (temperatura, velocidade, pressão) e o descritor resume a PDE, os parâmetros e o domínio. O fluxo principal é o `PhysicsAIPipeline`: gera dados de várias fontes (`PhysicsDatasetFactory`, com solvers clássicos, dados de PINN, PDE simbólica e colocação), treina um especialista por cenário (`SpecialistTrainer` -> `ModelZoo`), faz meta-aprendizado (`MetaLearner`, MAML/Reptile), monta um modelo de fundação (`WaMaModel` por média de pesos e `PhysicsFoundationModel`, um FNO com codificador de descritor e adaptadores LoRA) e avalia num `PhysicsBenchmark` com tarefas embutidas.

A base é simples e roda em CPU: cenários embutidos (`BUILTIN_SCENARIOS`: `burgers_1d`, `wave_1d`, `heat_2d`, `advection_2d`, `ns2d_cavity`, `heat_multiscale`), o `PhysicsSimulator` (tenta solvers de `pinneapple_simulation` e cai para diferenças finitas em PyTorch), o `WorldModelDataset` de pares (estado, próximo estado) e o `PhysicsWorldModel` (FNO residual condicionado a contexto). Há ainda um orquestrador por ferramentas (`PhysicsOrchestrator` + `PhysicsToolRegistry`) que escolhe, por regras fixas para cada tipo de problema (forward, inverse, design, forecast, uncertainty, discovery, digital_twin, world_model), uma cadeia de chamadas a outros pacotes.

Um segundo eixo é a fábrica de dados sintéticos para IA física (`SyntheticDataFactory`, "9 estágios"): cenário a partir de JSON ou de uma frase (`ScenarioGenerator`, extração por palavras-chave), solver (`PhysicsSolverBridge`), renderização em RGB/térmico/profundidade/IR (`PhysicsRenderer`), câmeras virtuais (`CameraSystem`), aleatorização para reduzir a diferença sim-real (`RealityRandomizer`), realce fotorrealista que nunca toca os campos físicos (`PhotorealisticEnhancer`), empacotamento em Zarr/vídeo/JSON (`DatasetPackager`) e datasets prontos para treino (`TrainingHooks`).

Os módulos mais novos e mais testados são `physical_world` (vídeo sintético de uma bola quicando com rótulos exatos, um world model pequeno treinável em CPU e checagens físicas que julgam qualquer vídeo, inclusive de um modelo de fundação como NVIDIA Cosmos via `CosmosAdapter`), `inspection_data` (inspeção térmica sintética com rótulo físico e cenas OpenUSD) e `robot_perception` (movimentos de braço simulados no Robot Sim/Newton, vistos por câmera aleatorizada). Eles são usados pelos exemplos em `examples/nvidia/`. O pacote se apoia em `pinneapple_simulation`, `pinneapple_physics`, `pinneapple_analysis`, `pinneapple_neural`, `pinneapple_adaptation`, `pinneapple_data` e `pinneapple_tools`, quase sempre com import opcional e fallback.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 31 | 15.016 | 90 | 49 | 94 (6 arquivos) | 23 |

Pasta de testes dedicada: `tests/pinneapple_worldmodel` (8 funções de teste).

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `benchmark` | `PhysicsBenchmark`: avalia um modelo em tarefas embutidas com trajetórias de referência, RMSE, L2 relativo e checagens de conservação. | 1 | 3 | 0 |
| `camera_system` | Câmeras virtuais (RGB, térmica, profundidade, IR) que projetam campos 2D em observações de sensor. | 1 | 3 | 0 |
| `curriculum` | Treino em estágios de complexidade crescente, avançando quando a perda de validação atinge a meta. | 1 | 3 | 0 |
| `dataset` | `WorldModelDataset` de pares (estado, contexto) -> próximo estado e o `DatasetBuilder` simulador -> dataset. | 1 | 3 | 20 |
| `dataset_factory` | `PhysicsDatasetFactory`: gera dados de várias fontes (solver, PINN, simbólico, colocação) num `DatasetCatalog`. | 1 | 4 | 0 |
| `dataset_packager` | Grava cada amostra na estrutura canônica de dataset de IA física (Zarr, vídeos, VTK, JSONs) com manifesto. | 1 | 4 | 0 |
| `generate_datasets` | Funções e CLI para gerar e carregar catálogos de datasets do world model. | 1 | 4 | 0 |
| `geometry` | Geometrias 2D/3D com CSG, regiões de contorno nomeadas e domínios prontos (cavidade, canal, cilindro, L, anel). | 1 | 17 | 0 |
| `inspection_data` | Dados sintéticos de inspeção térmica com rótulo físico, variações de cena em OpenUSD e checagem do rótulo. | 1 | 11 | 39 |
| `mega_model` | `PhysicsFoundationModel`: FNO com codificador de descritor por transformer e adaptadores LoRA. | 1 | 5 | 0 |
| `meta_learning` | Meta-aprendizado MAML/Reptile sobre tarefas amostradas de um catálogo. | 1 | 3 | 0 |
| `model` | `PhysicsWorldModel`: FNO residual condicionado a contexto, `(state_t, context) -> state_{t+1}`. | 1 | 2 | 20 |
| `model_zoo` | Registro de especialistas treinados com consulta, checkpoints e ensembles (média de saídas ou de pesos). | 1 | 4 | 0 |
| `orchestrator` | `PhysicsOrchestrator`: resolve um `ProblemStatement` encadeando ferramentas do registro por planos fixos por tipo de problema. | 1 | 3 | 20 |
| `photorealistic_enhancer` | Realce visual de quadros renderizados sem tocar dados físicos; backend padrão é identidade. | 1 | 3 | 0 |
| `physical_world` | Vídeo sintético com rótulos exatos, world model pequeno em CPU e checagens físicas de vídeo a partir dos pixels. | 1 | 12 | 39 |
| `physics_renderer` | Renderiza trajetórias de campos em quadros RGB, térmicos, de profundidade e IR, e em vídeo ou PNG. | 1 | 3 | 0 |
| `physics_solver_bridge` | Despacha um cenário para um backend de solver e empacota a saída como verdade de referência. | 1 | 4 | 0 |
| `physics_tools` | Inventário das capacidades dos pacotes como ferramentas chamáveis (`PhysicsToolRegistry`). | 1 | 2 | 35 |
| `pipeline` | `PhysicsAIPipeline`: ponto de entrada do fluxo dados -> especialistas -> meta -> fundação -> benchmark. | 1 | 3 | 0 |
| `reality_randomizer` | Aumentos de realismo (ruído, JPEG, iluminação, sombras, oclusões, blur, distorção, vinheta) só nas imagens. | 1 | 2 | 0 |
| `robot_perception` | Dados de percepção de robô: movimentos de braço simulados (Newton), linhas do tempo OpenUSD, câmera aleatorizada e checagens. | 1 | 13 | 39 |
| `scenario` | `PhysicsScenario` e o registro `BUILTIN_SCENARIOS` de problemas para treinar o world model. | 1 | 1 | 0 |
| `scenario_generator` | Converte JSON ou uma frase (por palavras-chave) num `ScenarioSpec` com geometria, contorno e sensores. | 1 | 2 | 0 |
| `simulator` | `PhysicsSimulator`: gera trajetórias de um cenário com solvers do projeto ou diferenças finitas embutidas. | 1 | 2 | 0 |
| `specialist_trainer` | Treina um `PhysicsWorldModel` por cenário e preenche o `ModelZoo`. | 1 | 2 | 0 |
| `synthetic_factory` | `SyntheticDataFactory`: encadeia os estágios de cenário a dataset multimodal, com atalhos por caso. | 1 | 7 | 0 |
| `train` | Entrada de treino com modos specialist, meta, foundation e pipeline (funções e CLI). | 1 | 5 | 0 |
| `trainer` | Laço de treino do world model com perda de rollout multi-passo e perda física opcional. | 1 | 3 | 0 |
| `training_hooks` | Datasets PyTorch sobre amostras empacotadas para codificador de vídeo, decodificador físico, PINN inversa e operador neural. | 1 | 6 | 0 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_worldmodel`

Ponto de entrada do pacote: reexporta o pipeline, o orquestrador, cenários, modelos e fábricas; o docstring descreve o pipeline de cinco estágios do modelo de fundação física.

<small>`pinneapple_worldmodel/__init__.py` · 409 linhas · testes: 11</small>

#### `pinneapple_worldmodel.benchmark`

Suíte de benchmark com tarefas embutidas (`BUILTIN_TASKS`, como `heat_2d_smooth`, `burgers_1d_shock`, `ns2d_lid_driven`) avaliadas por RMSE, L2 relativo, passo de divergência no rollout e checagens físicas. [sem teste direto]

<small>`pinneapple_worldmodel/benchmark.py` · 564 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BenchmarkTask` | classe | `(name, scenario_name, n_trajectories, rollout_steps, seed, physics_checks, metrics, descr…)` | Uma tarefa de benchmark: cenário, número de trajetórias, passos de rollout, semente, checagens físicas e métricas. |
| `BenchmarkResult` | classe | `(task_name, next_step_rmse, rollout_rmse, rel_l2, div_step, physics_scores, n_trajectorie…)` | Resultado de uma tarefa: RMSE de um passo e de rollout, L2 relativo, passo de divergência e escores físicos. |
| `PhysicsBenchmark` | classe | `(device='cpu', verbose=True) -> None` | Avalia modelos com `forward(state_t, descriptor)` nas tarefas do benchmark. |

#### `pinneapple_worldmodel.camera_system`

Estágio 5 da fábrica sintética: projeta campos 2D em observações de câmeras virtuais com pose, campo de visão, ruído e profundidade de bits. [sem teste direto]

<small>`pinneapple_worldmodel/camera_system.py` · 392 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CameraConfig` | classe | `(name, sensor_type, position, look_at, fov_deg, resolution, fps, near_clip, far_clip, noi…)` | Configuração de uma câmera virtual: tipo de sensor, pose, campo de visão, resolução, fps, ruído e bits. |
| `MultiCameraArray` | classe | `(cameras, sync_fps)` | Conjunto de câmeras que observam o mesmo campo simulado, sincronizadas. |
| `CameraSystem` | classe | `(array=None, domain_bounds=None) -> None` | Projeta campos físicos em observações no plano de cada câmera. |

#### `pinneapple_worldmodel.curriculum`

Currículo de treino por estágios de cenários (de calor 2D a Navier-Stokes) que avança ao atingir a perda-alvo de validação. [sem teste direto]

<small>`pinneapple_worldmodel/curriculum.py` · 283 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CurriculumStage` | classe | `(scenarios, n_samples, epochs, target_val_loss, lr, rollout_steps, description)` | Um estágio do currículo: cenários, amostras, épocas, perda-alvo, taxa de aprendizado e passos de rollout. |
| `CurriculumConfig` | classe | `(stages, model_config, device, batch_size, patience, save_dir, verbose)` | Configuração completa do currículo: estágios, modelo, dispositivo, lote, paciência e pasta de saída. |
| `PhysicsCurriculum` | classe | `(config) -> None` | Treina um world model passando pelos estágios do currículo. |

#### `pinneapple_worldmodel.dataset`

Transforma trajetórias simuladas em amostras `state_t`, `state_tp1`, `params`, `context` normalizadas; `DatasetBuilder` orquestra simulação e montagem.

<small>`pinneapple_worldmodel/dataset.py` · 382 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `WorldModelDataset` | classe | `(trajectories, *, horizon=1, normalize=True, transform=None) -> None` | Dataset de previsão de um ou vários passos construído de uma lista de trajetórias. |
| `DatasetConfig` | classe | `(scenarios, n_samples_per_scenario, horizon, normalize, validate_physics, save_dir, devic…)` | Configuração do `DatasetBuilder`: cenários, amostras por cenário, horizonte, normalização e validação física. |
| `DatasetBuilder` | classe | `(config) -> None` | Orquestra simulação e montagem de um `WorldModelDataset`. |

#### `pinneapple_worldmodel.dataset_factory`

Fábrica multi-fonte de dados (solver, PINN via `pinneapple_physics`, PDE simbólica, colocação de `pinneapple_data`) com catálogo por cenário e fonte; fontes indisponíveis são puladas. [sem teste direto]

<small>`pinneapple_worldmodel/dataset_factory.py` · 725 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FactoryConfig` | classe | `(sources, scenarios, n_samples_per_scenario, horizon, validate_physics, device, save_dir,…)` | Configuração da `PhysicsDatasetFactory`: fontes, cenários, amostras, horizonte, validação e pasta. |
| `DatasetEntry` | classe | `(name, dataset, source, scenario, physics_tags, n_samples, generation_time_s, metadata)` | Uma fatia do catálogo: dataset, fonte, cenário, tags físicas, número de amostras e tempo de geração. |
| `DatasetCatalog` | classe | `() -> None` | Registro de todos os datasets gerados, consultável por cenário e fonte. |
| `PhysicsDatasetFactory` | classe | `(config) -> None` | Gera datasets de treino combinando as fontes de dados disponíveis. |

#### `pinneapple_worldmodel.dataset_packager`

Estágio 8: grava cada amostra em `sample_XXXXXX/` com campos em Zarr, vídeos, `mesh.vtk` e JSONs de metadados, PDE, contorno e câmera, mais um manifesto. [sem teste direto]

<small>`pinneapple_worldmodel/dataset_packager.py` · 439 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PackagerConfig` | classe | `(output_dir, format, zarr_chunks, zarr_compressor, save_vtk, save_videos, sample_id_digit…)` | Como o empacotador escreve o dataset: formato, chunks e compressor Zarr, VTK, vídeos e sobrescrita. |
| `PackagedSample` | classe | `(sample_id, sample_dir, field_paths, video_paths, metadata_path)` | Metadados de uma amostra empacotada: id, pasta e caminhos de campos, vídeos e metadados. |
| `DatasetManifest` | classe | `(output_dir) -> None` | Índice de todas as amostras de um dataset de IA física. |
| `DatasetPackager` | classe | `(config=None) -> None` | Escreve uma amostra física na estrutura canônica do dataset. |

#### `pinneapple_worldmodel.generate_datasets`

Gera catálogos de trajetórias multi-fonte para treinar o world model, via funções ou `python -m pinneapple_worldmodel.generate_datasets`. [sem teste direto]

<small>`pinneapple_worldmodel/generate_datasets.py` · 441 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `generate` | função | `(scenarios=None, sources=None, n_samples=100, output_dir='./data/worldmodel', device='cpu…)` | Gera datasets multi-fonte e devolve um `DatasetCatalog`. |
| `generate_via_factory` | função | `(scenarios=None, sources=None, n_samples=100, output_dir='./data/worldmodel', device='cpu…)` | Gera datasets pela `PhysicsDatasetFactory` de alto nível. |
| `load_catalog` | função | `(path) -> DatasetCatalog` | Carrega do disco um `DatasetCatalog` salvo. |
| `main` | função | `() -> None` | Ponto de entrada da linha de comando de geração de datasets. |

#### `pinneapple_worldmodel.geometry`

Sistema de geometria componível (primitivas 2D/3D, CSG, regiões de contorno, domínios prontos) usado pela fábrica de dados e pelo treinador; o docstring cita `make_pipe_bend`, que não existe no módulo. [sem teste direto]

<small>`pinneapple_worldmodel/geometry.py` · 569 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GeometryBase` | classe | `()` | Base abstrata das geometrias (pertinência e amostragem de interior e fronteira). |
| `Rectangle` | classe | `(lo, hi) -> None` | Retângulo alinhado aos eixos. |
| `Circle` | classe | `(center, radius) -> None` | Círculo 2D preenchido. |
| `Polygon` | classe | `(vertices) -> None` | Polígono 2D preenchido definido por vértices ordenados. |
| `Box3D` | classe | `(lo, hi) -> None` | Caixa 3D alinhada aos eixos. |
| `Sphere` | classe | `(center, radius) -> None` | Esfera 3D preenchida. |
| `Union` | classe | `(a, b) -> None` | União CSG: pontos dentro de *a* ou de *b*. |
| `Intersection` | classe | `(a, b) -> None` | Interseção CSG: pontos dentro de *a* e de *b*. |
| `Difference` | classe | `(a, b) -> None` | Diferença CSG: pontos dentro de *a* e fora de *b*. |
| `BoundaryRegion` | classe | `(name, bc_type, value, sampler)` | Parte nomeada da fronteira com tipo e valor de condição de contorno e um amostrador. |
| `PhysicsDomain` | classe | `(geometry, boundaries, name, tags)` | Domínio completo: geometria, regiões de contorno, nome e tags. |
| `make_unit_square` | função | `(*, with_boundary_labels=True) -> PhysicsDomain` | Quadrado unitário com rótulos de contorno, domínio padrão de benchmark. |
| `make_cavity` | função | `(L=1.0) -> PhysicsDomain` | Cavidade com tampa móvel (`u=1` em `y=L`). |
| `make_channel` | função | `(length=4.0, height=1.0, *, inlet_velocity=1.0) -> PhysicsDomain` | Domínio de canal de Poiseuille com velocidade de entrada. |
| `make_channel_with_cylinder` | função | `(length=4.0, height=1.0, cyl_center=(1.0, 0.5), cyl_radius=0.1) -> PhysicsDomain` | Canal 2D com obstáculo circular. |
| `make_l_shaped` | função | `() -> PhysicsDomain` | Domínio em L (benchmark de concentração de tensão). |
| `make_annulus` | função | `(inner_r=0.3, outer_r=1.0, center=(0.0, 0.0)) -> PhysicsDomain` | Domínio anular para problemas de calor ou eletromagnetismo em geometria cilíndrica. |

#### `pinneapple_worldmodel.inspection_data`

Inspeção térmica sintética: placa aquecida com vazio subsuperficial resolvida por volumes finitos, variações de câmera gravadas como estágios OpenUSD e verificação de que o aumento manteve o rótulo físico; precisa de `pxr`.

<small>`pinneapple_worldmodel/inspection_data.py` · 224 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Variation` | classe | `(gain, offset, noise, blur, zoom, angle_deg, shift, glint)` | Parâmetros de uma variação da câmera sintética: ganho, offset, ruído, blur, zoom, ângulo, deslocamento e brilho especular. |
| `thermal_plate` | função | `(defect, n=48, *, k=1.0, k_void=0.15, q=1.0) -> dict` | Condução estacionária no quadrado unitário com disco de baixa condutividade; devolve campo, campo sem defeito e o rótulo. |
| `sample_variation` | função | `(rng, *, glint_p=0.2) -> Variation` | Sorteia uma `Variation`, com brilho especular com probabilidade `glint_p`. |
| `camera_transform` | função | `(var, H, W)` | Transformação placa -> pixel e sua inversa para a câmera da variação. |
| `write_variation_stage` | função | `(path, plate, var, *, cmap='gray') -> str` | Grava a placa e a variação como um estágio OpenUSD e devolve o caminho. |
| `thermal_image` | função | `(stage_path, var, H=64, W=64, seed=0) -> np.ndarray` | Imagem `(H, W)` da câmera de um estágio de variação. |
| `nominal_image` | função | `(stage_path, plate, var, H=64, W=64) -> np.ndarray` | Imagem que a mesma câmera veria na placa sem defeito, a referência do inspetor. |
| `label_check` | função | `(image, nominal, var, label_xy, *, tol=0.08) -> dict` | Verifica se a imagem ainda mostra a anomalia física onde o rótulo indica, pelo resíduo contra a imagem nominal. |
| `inspection_dataset` | função | `(n, seed, workdir, *, glint_p=0.25, H=64, W=64, keep_stages=0) -> dict` | Gera `n` amostras com vazio aleatório, campo térmico, variação em USD, rótulo em pixels e checagem do rótulo. |
| `inspection_sample` | função | `(x=0.4, y=0.6, r=0.08, seed=0, glint=False, usd_path='inspection_sample.usda') -> dict` | Uma amostra de inspeção térmica com vazio em posição dada, variação de câmera e checagem do rótulo. |
| `cleanup` | função | `(paths) -> None` | Remove os arquivos indicados que existirem. |

#### `pinneapple_worldmodel.mega_model`

Modelo de fundação generalista: FNO com condicionamento FiLM por um codificador transformer de descritor físico e adaptadores LoRA; suporta grades 1D e 2D, 3D ainda não. [sem teste direto]

<small>`pinneapple_worldmodel/mega_model.py` · 668 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FoundationConfig` | classe | `(n_modes, width, depth, context_dim, descriptor_vocab, n_heads, n_context_layers, lora_ra…)` | Hiperparâmetros do `PhysicsFoundationModel` (modos, largura, profundidade, contexto, cabeças, posto LoRA, incerteza aleatória). |
| `LoRALinear` | classe | `(in_features, out_features, rank=8) -> None` | Camada linear com adaptação de baixo posto para adaptação rápida de domínio. |
| `PhysicsDescriptorEncoder` | classe | `(context_dim=64, max_scalar_features=32, n_heads=4, n_layers=2, dropout=0.1) -> None` | Codifica um dicionário descritor da física num vetor de contexto de tamanho fixo (transformer). |
| `FoundationFNOBlock` | classe | `(width, n_modes, context_dim, lora_rank=8, dropout=0.1, spatial_dim=2) -> None` | Bloco FNO com injeção de contexto e LoRA opcional. |
| `PhysicsFoundationModel` | classe | `(config, *, n_fields=1, grid_shape=(64, 64)) -> None` | Modelo de fundação generalista para previsão de campos condicionada a descritor. |

#### `pinneapple_worldmodel.meta_learning`

Meta-aprendizado MAML/Reptile para adaptação rápida a novo domínio, com caminho opcional por `pinneapple_adaptation.meta_learning`; o docstring cita classes `MAMLWorldModel`/`ReptileWorldModel` que não existem no arquivo. [sem teste direto]

<small>`pinneapple_worldmodel/meta_learning.py` · 512 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MetaConfig` | classe | `(algorithm, n_meta_epochs, n_inner_steps, inner_lr, outer_lr, n_tasks_per_batch, n_suppor…)` | Configuração do `MetaLearner`: algoritmo, épocas, passos internos, taxas e tamanhos de suporte e consulta. |
| `TaskDistribution` | classe | `(catalog, n_support=32, n_query=32, device='cpu') -> None` | Sorteia tarefas (suporte, consulta) de um `DatasetCatalog`. |
| `MetaLearner` | classe | `(config, model_config=None) -> None` | Meta-aprendizado MAML/Reptile de um world model. |

#### `pinneapple_worldmodel.model`

Arquitetura `PhysicsWorldModel`: codifica o contexto, concatena ao estado, aplica camadas de Fourier e prevê o incremento sobre o estado atual (1D com `rfft`, 2D com `rfft2`).

<small>`pinneapple_worldmodel/model.py` · 320 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `WorldModelConfig` | classe | `(n_modes, width, depth, context_dim, embed_dim, activation, dropout, rollout_steps)` | Hiperparâmetros do `PhysicsWorldModel`. |
| `PhysicsWorldModel` | classe | `(config, *, n_fields, grid_shape) -> None` | World model FNO: `(state_t, context) -> state_{t+1}` com formulação residual. |

#### `pinneapple_worldmodel.model_zoo`

Registro de especialistas por cenário com consulta por nome, tag ou métrica, salvar/carregar e ensembles por média de saídas ou de pesos. [sem teste direto]

<small>`pinneapple_worldmodel/model_zoo.py` · 494 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ZooEntry` | classe | `(name, model, scenario, physics_tags, metrics, n_params, trained_at, checkpoint_path, met…)` | Entrada do zoo: modelo, cenário, tags, métricas, parâmetros, data de treino e checkpoint. |
| `EnsembleModel` | classe | `(models, weights=None, aggregation='mean') -> None` | Ensemble que combina as saídas de vários especialistas `PhysicsWorldModel`. |
| `WaMaModel` | classe | `()` | Modelo obtido pela média dos pesos de vários especialistas. |
| `ModelZoo` | classe | `(root_dir=None) -> None` | Registro de especialistas treinados com consulta, persistência e ensembles. |

#### `pinneapple_worldmodel.orchestrator`

Orquestrador orientado a objetivo: escolhe um plano fixo por `kind` do `ProblemStatement` e executa as ferramentas do `PhysicsToolRegistry`, com um `tool_gate` opcional que só permite ou bloqueia chamadas.

<small>`pinneapple_worldmodel/orchestrator.py` · 766 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ProblemStatement` | classe | `(kind, description, pde_hint, domain_hint, scenarios, params, observations, model, n_samp…)` | Descrição de alto nível de um problema físico: tipo, descrição, dicas de PDE e domínio, cenários, parâmetros e saídas. |
| `OrchestratorResult` | classe | `(kind, artifacts, plan, elapsed_s, logs)` | Saída de `PhysicsOrchestrator.solve`: artefatos, plano, tempo e logs. |
| `PhysicsOrchestrator` | classe | `(registry=None, verbose=True, tool_gate=None) -> None` | Resolve problemas físicos encadeando ferramentas do registro segundo o tipo do problema. |

#### `pinneapple_worldmodel.photorealistic_enhancer`

Estágio 7: realce de imagens que nunca lê campos físicos; o backend padrão `stub` devolve a entrada, `local_diffusion` precisa de `diffusers` e `cosmos` é [placeholder] ainda indisponível (cai para `stub`). [sem teste direto]

<small>`pinneapple_worldmodel/photorealistic_enhancer.py` · 283 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): placeholder/stub citado no docstring</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EnhancerConfig` | classe | `(backend, strength, style_prompt, negative_prompt, model_id, device, batch_size, apply_to…)` | Configuração do realce: backend, intensidade, prompts, modelo, dispositivo e sensores. |
| `PhotorealisticEnhancer` | classe | `(config=None) -> None` | Realça quadros renderizados sem tocar dados físicos, com fallback para o backend identidade. |
| `register_enhancer` | função | `(name, fn) -> None` | Registra uma função de realce personalizada sob um nome. |

#### `pinneapple_worldmodel.physical_world`

Vídeo sintético de bola quicando com física exata e aleatorização de domínio, um world model conv+GRU treinável em CPU, edições não-físicas e checagens físicas a partir dos pixels, além de um adaptador HTTP para NVIDIA Cosmos.

<small>`pinneapple_worldmodel/physical_world.py` · 489 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Video` | classe | `(frames, dt, px_per_m, labels, params)` | Uma cena: quadros `(T, H, W)`, passo de tempo, pixels por metro, rótulos e parâmetros. |
| `WorldModel` | classe | `(H=64, W=96, latent=32, seed=0)` | Codificador conv, dinâmica latente GRU e decodificador, treinado para prever o próximo quadro; pequeno para CPU. |
| `StateEncoder` | classe | `(H=64, W=96, seed=0)` | CNN pequena que lê o centro do objeto num quadro, treinada nos rótulos exatos. |
| `CosmosAdapter` | classe | `(base_url, api_key=None, timeout=600.0)` | Cliente HTTP de um NIM NVIDIA Cosmos: envia prompt (e vídeo opcional) e decodifica o vídeo devolvido em quadros. |
| `bouncing_ball_scene` | função | `(*, seed=0, T=48, fps=24.0, H=64, W=96, randomize=True, defect=False, e=None) -> Video` | Bola lançada quicando no chão, renderizada com aleatorização de domínio e rótulos exatos. |
| `dataset` | função | `(n, *, seed=0, **kw) -> list[Video]` | Gera `n` vídeos de `bouncing_ball_scene` com sementes sucessivas. |
| `track_object` | função | `(frames) -> dict[str, np.ndarray]` | Centro e área do objeto de maior contraste em cada quadro (NaN quando não há). |
| `physics_checks` | função | `(video, *, dt=None, px_per_m=None, g=G, g_rtol=0.1, area_rtol=0.25, max_speed=10.0) -> di…)` | Checagens físicas de um vídeo de bola a partir dos pixels (gravidade ajustada, área, velocidade), com veredito. |
| `video_check` | função | `(seed=0, edit='none') -> dict[str, Any]` | Gera um vídeo sintético, opcionalmente editado de forma não física, e o julga com `physics_checks`. |
| `corrupt` | função | `(video, kind, seed=0) -> Video` | Edições não físicas: `time_warp`, `teleport`, `fade`, `gravity_flip`. |
| `state_pairs` | função | `(videos) -> tuple[np.ndarray, np.ndarray]` | Pares `(s_t, s_t+1)` com posição e velocidade em metros tirados dos rótulos exatos. |
| `render_like` | função | `(video, x_m, y_m) -> np.ndarray` | Renderiza uma trajetória com a câmera e a aparência de um vídeo dado. |

#### `pinneapple_worldmodel.physics_renderer`

Estágio 4: converte campos em quadros RGB, térmicos, de profundidade e IR por colormaps e monta vídeos `.mp4` (PNG se não houver ffmpeg). [sem teste direto]

<small>`pinneapple_worldmodel/physics_renderer.py` · 444 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RendererConfig` | classe | `(resolution, fps, sensors, colormap_rgb, colormap_thermal, colormap_depth, colormap_ir, d…)` | Configuração de renderização: resolução, fps, sensores, colormaps, dpi e qualidade de vídeo. |
| `RenderResult` | classe | `(frames, video_paths, png_dirs, field_stats)` | Saída de `PhysicsRenderer.render()`: quadros, caminhos de vídeo e PNG e estatísticas de campo. |
| `PhysicsRenderer` | classe | `(config=None) -> None` | Renderiza trajetórias de campos em quadros e vídeos por sensor. |

#### `pinneapple_worldmodel.physics_solver_bridge`

Estágios 2 e 3: despacha o cenário para `builtin` (PyTorch), OpenFOAM, FEniCS, PINN ou FNO, com aviso e retorno ao `builtin` quando o backend falta, e empacota a saída em Zarr (ou NumPy). [sem teste direto]

<small>`pinneapple_worldmodel/physics_solver_bridge.py` · 837 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SolverOutput` | classe | `(fields, t_coords, x_coords, y_coords, z_coords, params, solver_name, metadata)` | Histórico multivariado de campos (u, v, w, p, T, C) com coordenadas, parâmetros e nome do solver. |
| `SolverBridgeConfig` | classe | `(solver, generate_temperature, generate_concentration, openfoam_case_dir, fenics_problem,…)` | Configuração do bridge: solver, campos extras, caso OpenFOAM, problema FEniCS, checkpoints PINN/FNO. |
| `PhysicsSolverBridge` | classe | `(config=None) -> None` | Despacha um cenário para o backend de solver adequado. |
| `GroundTruthPackager` | classe | `(output_dir, format='zarr', zarr_chunks=None, overwrite=True) -> None` | Organiza `SolverOutput` na estrutura canônica de dataset (Zarr ou NumPy). |

#### `pinneapple_worldmodel.physics_tools`

Cataloga capacidades dos pacotes como `PhysicsTool` com categoria, esquemas de entrada/saída e função, usadas pelo orquestrador; ferramentas cujo pacote falta ficam indisponíveis.

<small>`pinneapple_worldmodel/physics_tools.py` · 983 linhas · testes: 35</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsTool` | classe | `(name, category, description, input_schema, output_schema, module_path, fn, tags)` | Uma capacidade registrada: nome, categoria, descrição, esquemas, módulo, função e tags. |
| `PhysicsToolRegistry` | classe | `() -> None` | Catálogo das capacidades como ferramentas chamáveis, com busca por categoria, tag ou texto. |

#### `pinneapple_worldmodel.pipeline`

`PhysicsAIPipeline`: roda os cinco estágios (dados, especialistas, meta-aprendizado, fundação, benchmark) a partir de um `PhysicsAIConfig`. [sem teste direto]

<small>`pinneapple_worldmodel/pipeline.py` · 498 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsAIConfig` | classe | `(scenarios, sources, n_samples, device, save_dir, specialist_epochs, specialist_lr, speci…)` | Configuração completa do `PhysicsAIPipeline`. |
| `PhysicsAIPipelineResult` | classe | `(catalog, zoo, meta_model, mega_model, benchmark, elapsed_s)` | Saídas do pipeline: catálogo, zoo, meta-modelo, mega-modelo, benchmark e tempo. |
| `PhysicsAIPipeline` | classe | `(config) -> None` | Constrói o world model generalista rodando todos os estágios. |

#### `pinneapple_worldmodel.reality_randomizer`

Estágio 6: aplica aumentos de imagem com probabilidade e intensidade configuráveis, sem tocar os arrays físicos. [sem teste direto]

<small>`pinneapple_worldmodel/reality_randomizer.py` · 328 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RandomizerConfig` | classe | `(seed, p_sensor_noise, gaussian_std, poisson_scale, p_jpeg, jpeg_quality_lo, jpeg_quality…)` | Quais aumentos aplicar, com que probabilidade e intensidade. |
| `RealityRandomizer` | classe | `(config=None) -> None` | Aplica aumentos de diferença sim-real a sequências de imagens. |

#### `pinneapple_worldmodel.robot_perception`

Movimentos de braço de dois elos simulados no Robot Sim (NVIDIA Newton), gravados como linha do tempo OpenUSD, renderizados por câmera aleatorizada e rotulados com os ângulos exatos, com checagens de percepção.

<small>`pinneapple_worldmodel/robot_perception.py` · 283 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CameraVariation` | classe | `(zoom, offset, gain, noise, blur, albedo, background, occluder)` | Variação da câmera sintética: zoom, offset, ganho, ruído, blur, albedo, fundo e oclusor. |
| `ImageRegressor` | classe | `(H, W, n_out, seed=0)` | CNN pequena de uma imagem cinza para `n_out` números. |
| `forward_points` | função | `(arm, q) -> dict[str, np.ndarray]` | Cotovelo e ponto da ferramenta no plano x-z para ângulos de junta dados. |
| `arm_motions` | função | `(arm, n, seed, *, duration=0.8, dt=0.001, fps=30.0, kp=(350.0, 110.0), kd=(5.0, 2.5), q_b…)` | Gera `n` movimentos ponto a ponto aleatórios simulados no Newton, amostrados em `fps`. |
| `sample_camera` | função | `(rng, *, occluder_p=0.3) -> CameraVariation` | Sorteia uma `CameraVariation`, com oclusor com probabilidade `occluder_p`. |
| `write_motion_stage` | função | `(path, arm, motion, var, *, payload=None) -> str` | Grava o movimento como linha do tempo OpenUSD, com carga opcional presa à ferramenta. |
| `camera_frames` | função | `(path, arm, motion, var, H=72, W=72, seed=0) -> np.ndarray` | Quadros cinza `(T, H, W)` do estágio em cada instante registrado. |
| `visibility` | função | `(arm, motion, var, *, margin=0.03) -> np.ndarray` | Por quadro, se cotovelo e ferramenta estão na imagem e não atrás do oclusor. |
| `payload_track_check` | função | `(frames, arm, motion, var, *, tol_px=2.0, offset=(0.0, 0.0)) -> dict` | Verifica se cada quadro mostra a peça carregada onde a cinemática a coloca. |
| `arm_camera_sample` | função | `(payload=5.0, seed=0, occluder=False, usd_path='arm_camera_sample.usda') -> dict` | Uma amostra rotulada de percepção: movimento, estágio USD com câmera, quadros e checagens. |
| `angle_features` | função | `(q) -> np.ndarray` | Converte ângulos de junta em features `sin`/`cos`. |
| `feature_angles` | função | `(f) -> np.ndarray` | Inverso de `angle_features`: recupera os ângulos por `arctan2`. |
| `perception_checks` | função | `(q_pred, motion, arm, *, qd_max=12.0) -> dict` | Erros de uma sequência de ângulos prevista para um movimento, com limite de velocidade de junta. |

#### `pinneapple_worldmodel.scenario`

Define `PhysicsScenario` (PDE, grade, tempo, faixas de parâmetros, CI/CC, solver) e o registro de cenários embutidos. [sem teste direto]

<small>`pinneapple_worldmodel/scenario.py` · 253 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsScenario` | classe | `(name, pde_kind, grid_shape, t_span, n_steps, domain_bounds, param_ranges, ic_type, bc_ty…)` | Especificação completa de um problema físico para o world model. |

#### `pinneapple_worldmodel.scenario_generator`

Estágio 1: transforma JSON, nome ou frase (extração heurística por palavras-chave, por exemplo número de Reynolds) num `ScenarioSpec` completo. [sem teste direto]

<small>`pinneapple_worldmodel/scenario_generator.py` · 360 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ScenarioSpec` | classe | `(scenario, domain, geometry, fluid, boundary_conditions, initial_conditions, sensor_confi…)` | Especificação estendida: cenário, domínio, geometria, fluido, contorno, condições iniciais e sensores. |
| `ScenarioGenerator` | classe | `(default_grid_2d=(64, 64), default_n_steps=32, device='cpu') -> None` | Converte prompt, JSON, dicionário ou nome num `ScenarioSpec`. |

#### `pinneapple_worldmodel.simulator`

Gera trajetórias `(T+1, C, *grid)` tentando solvers de `pinneapple_simulation` (FDM, LBM, FEM) e caindo para solvers PyTorch embutidos de calor, Burgers, onda, advecção e NS2D. [sem teste direto]

<small>`pinneapple_worldmodel/simulator.py` · 630 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TrajectoryData` | classe | `(states, params, scenario_name, metadata)` | Uma trajetória: sequência de campos, parâmetros, cenário e metadados. |
| `PhysicsSimulator` | classe | `(scenario, *, device='cpu', verbose=False) -> None` | Gera trajetórias de um `PhysicsScenario` com o backend disponível. |

#### `pinneapple_worldmodel.specialist_trainer`

Treina um especialista por cenário com validação física opcional (`pinneapple_analysis.validation`) e transferência (`pinneapple_adaptation`); o docstring ainda cita pacotes com nomes antigos. [sem teste direto]

<small>`pinneapple_worldmodel/specialist_trainer.py` · 423 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SpecialistConfig` | classe | `(model, epochs, lr, batch_size, device, val_fraction, patience, grad_clip, rollout_steps,…)` | Configuração do treino de especialistas (épocas, taxa, lote, paciência, rollout, transferência, UQ). |
| `SpecialistTrainer` | classe | `(config, *, physics_loss_fn=None, pretrained_zoo=None) -> None` | Treina especialistas e preenche um `ModelZoo`. |

#### `pinneapple_worldmodel.synthetic_factory`

Fábrica de dados sintéticos de ponta a ponta (cenário, simulação, renderização, câmeras, aleatorização, realce, empacotamento, hooks de treino) com atalhos para tubo, cilindro, calor e mistura. [sem teste direto]

<small>`pinneapple_worldmodel/synthetic_factory.py` · 525 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SyntheticFactoryConfig` | classe | `(scenario_input, n_samples, output_dir, device, render_fps, sensors, render_resolution, a…)` | Configuração única de todos os estágios da fábrica sintética. |
| `FactoryResult` | classe | `(config, spec, samples, output_dir, elapsed_s)` | Saída de `SyntheticDataFactory.generate_dataset()`: configuração, spec, amostras, pasta e tempo. |
| `SyntheticDataFactory` | classe | `(config=None) -> None` | Fábrica de dados sintéticos de ponta a ponta para IA física. |
| `generate_pipe_flow` | função | `(n_samples=20, output_dir='./pipe_dataset', **kw) -> FactoryResult` | Gera um dataset de escoamento em tubo (NS2D, Re de 100 a 5000). |
| `generate_cylinder_flow` | função | `(n_samples=20, output_dir='./cylinder_dataset', **kw) -> FactoryResult` | Gera um dataset de escoamento ao redor de cilindro (esteira de von Kármán). |
| `generate_heat_transfer` | função | `(n_samples=20, output_dir='./heat_dataset', **kw) -> FactoryResult` | Gera um dataset de condução de calor 2D. |
| `generate_chemical_mixing` | função | `(n_samples=20, output_dir='./mixing_dataset', **kw) -> FactoryResult` | Gera um dataset de mistura química (advecção-difusão). |

#### `pinneapple_worldmodel.train`

Entrada de treino com quatro modos (specialist, meta, foundation, pipeline), como funções e CLI. [sem teste direto]

<small>`pinneapple_worldmodel/train.py` · 593 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `train_specialist` | função | `(catalog, output_dir, *, epochs=50, batch_size=32, lr=0.001, device='cpu', patience=10, n…)` | Treina um especialista por cenário do catálogo. |
| `train_meta` | função | `(catalog, output_dir, *, algorithm='reptile', n_meta_epochs=100, n_inner_steps=5, inner_l…)` | Meta-treina uma inicialização compartilhada sobre o catálogo. |
| `train_foundation` | função | `(catalog, output_dir, *, base_ckpt=None, epochs=30, batch_size=16, lr=0.0005, device='cpu…)` | Monta e ajusta o `PhysicsFoundationModel` com LoRA. |
| `run_pipeline` | função | `(scenarios=None, sources=None, n_samples=100, output_dir='./checkpoints/worldmodel', devi…)` | Roda o pipeline completo de cinco estágios. |
| `main` | função | `() -> None` | Ponto de entrada da linha de comando de treino. |

#### `pinneapple_worldmodel.trainer`

Laço de treino do `PhysicsWorldModel` com perda de rollout autoregressivo, perda física opcional e validação por RMSE e L2 relativo. [sem teste direto]

<small>`pinneapple_worldmodel/trainer.py` · 421 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `WorldModelLoss` | classe | `(rollout_steps=1, rollout_weight=0.5, physics_loss_fn=None, physics_weight=0.01) -> None` | Perda de rollout multi-passo com termo físico opcional. |
| `WorldModelTrainConfig` | classe | `(epochs, lr, weight_decay, batch_size, val_fraction, device, amp, grad_clip, rollout_step…)` | Hiperparâmetros do `WorldModelTrainer`. |
| `WorldModelTrainer` | classe | `(model, config, *, physics_loss_fn=None) -> None` | Treina um `PhysicsWorldModel` num `WorldModelDataset`. |

#### `pinneapple_worldmodel.training_hooks`

Estágio 9: datasets PyTorch sobre o dataset empacotado para quatro objetivos (codificador de vídeo, decodificador de campo, PINN inversa, operador neural); sem arquivo de embedding, o decodificador recebe embedding zero. [sem teste direto]

<small>`pinneapple_worldmodel/training_hooks.py` · 478 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsAIDataset` | classe | `(dataset_dir, fields=('u', 'v', 'p', 'T'), sensors=(), max_samples=None, device='cpu') ->…)` | Dataset base sobre as amostras empacotadas (campos e sensores escolhidos). |
| `CosmosEncoderDataset` | classe | `(dataset_dir, video_sensors=('rgb', 'thermal'), field_names=('u', 'v', 'p', 'T'), **kwarg…)` | Dataset para treinar um codificador vídeo -> embedding latente. |
| `PhysicsDecoderDataset` | classe | `(dataset_dir, embedding_dim=512, field_names=('u', 'v', 'p', 'T'), device='cpu', max_samp…)` | Dataset para treinar um decodificador embedding -> campos físicos (embedding zero se não houver arquivo). |
| `InversePINNDataset` | classe | `(dataset_dir, param_keys=None, video_sensor='rgb', **kwargs) -> None` | Dataset para PINN inversa: quadros de vídeo -> parâmetros físicos. |
| `NeuralOperatorDataset` | classe | `(dataset_dir, field_names=('u', 'v', 'p'), horizon=1, subsample_t=1, device='cpu', max_sa…)` | Dataset de pares (estado, próximo estado) para operadores neurais. |
| `TrainingHooks` | classe | `(dataset_dir, device='cpu') -> None` | Fábrica de conveniência para todos os tipos de dataset de treino. |

### Exemplos de uso reais do repositório

**Gerar dados e treinar especialistas** — `examples/worldmodel/01_generate_train_evaluate.py`

```python
    from pinneapple_worldmodel import train_specialist

    t0 = time.perf_counter()
    zoo = train_specialist(
        catalog      = catalog,
        output_dir   = Path(cfg["output_dir"]) / "checkpoints",
        epochs       = cfg["epochs"],
        batch_size   = cfg["batch_size"],
        lr           = cfg["lr"],
        device       = cfg["device"],
        patience     = max(cfg["epochs"] // 3, 5),
        n_modes      = cfg["n_modes"],
        width        = cfg["width"],
```

**Vídeo sintético julgado por checagens físicas** — `examples/nvidia/04_worldmodel_physics_video/run.py`

```python
        with info.step("synthetic videos (exact physics + randomization)"):
            train = dataset(n_train, seed=100, defect=False)
            test = [v for v in dataset(n_test, seed=5000)]
            test_def = dataset(max(2, n_test // 5), seed=7000, defect=True)
        with info.step("checks on ground truth and edits"):
            gt = [physics_checks(v) for v in test + test_def]
            bad = {k: [physics_checks(corrupt(v, k)) for v in test] for k in KINDS}
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 8 de 31.
- Funções de teste em arquivos que importam algum módulo do pacote: 94.

### Limitações conhecidas

- A maior parte dos módulos não tem teste que os importe diretamente; os testados são `physical_world`, `inspection_data`, `robot_perception`, `physics_tools`, `orchestrator`, `dataset` e `model`.
- Os docstrings citam pacotes com nomes antigos que não existem no repositório (`pinneapple_pinn`, `pinneapple_symbolic`, `pinneapple_validate`, `pinneapple_uq`, `pinneapple_transfer`, `pinneapple_meta`); o código usa `pinneapple_physics`, `pinneapple_analysis`, `pinneapple_adaptation` e `pinneapple_simulation`. Também citam `MAMLWorldModel`, `ReptileWorldModel` e `make_pipe_bend`, que não existem.
- `PhysicsFoundationModel` só suporta grades 1D e 2D (3D ainda não, segundo o docstring).
- `PhotorealisticEnhancer`: o backend padrão é identidade, `local_diffusion` exige `diffusers` e `cosmos` é placeholder ainda indisponível; backends ausentes caem para identidade com aviso.
- `PhysicsSolverBridge` e `PhysicsSimulator` caem para solvers PyTorch embutidos (com aviso) quando OpenFOAM, FEniCS ou checkpoints PINN/FNO não estão disponíveis.
- `ScenarioGenerator.from_prompt` e o `PhysicsOrchestrator` usam heurísticas: extração por palavras-chave e um plano fixo por tipo de problema, sem modelo de linguagem.
- `inspection_data` e `robot_perception` precisam de OpenUSD (`pxr`); `robot_perception` precisa do NVIDIA Newton; `CosmosAdapter` precisa de um endpoint NIM Cosmos e de `imageio` com ffmpeg.
- Em `training_hooks`, `PhysicsDecoderDataset` devolve um embedding zero como placeholder quando os embeddings não foram pré-calculados.

## 7. `pinneapple_perception`

### Objetivo

_Sem texto de objetivo._

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 7 | 1.037 | 6 | 19 | 29 (3 arquivos) | 2 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `audio_modal` |  | 1 | 1 | 0 |
| `change_detection` |  | 1 | 10 | 13 |
| `image_geometry` |  | 1 | 2 | 0 |
| `piv_tools` |  | 1 | 6 | 6 |
| `sentinel2` |  | 1 | 4 | 13 |
| `video_piv` |  | 1 | 2 | 6 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_perception`

pinneapple_perception — extract physics observations from images, video, and audio, for use as `DataConstraint` targets or boundary geometry anywhere else in PINNeAPPle.

<small>`pinneapple_perception/__init__.py` · 55 linhas · testes: 23 · palavras-chave no código (heurística; o texto acima diz o status real): experimental</small>

#### `pinneapple_perception.audio_modal`

Extract dominant vibration/acoustic frequencies from an audio recording (e.g. a struck beam, plate, or shell's impulse response) via FFT peak-picking with sub-bin parabolic frequency refinement. Depends only on numpy/scipy (already core PINNeAPPle dependencies).

<small>`pinneapple_perception/audio_modal.py` · 95 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `extract_dominant_frequencies` | função | `(waveform, sample_rate, n_peaks=5, min_freq=0.0, max_freq=None, window='hann') -> List[Tu…)` | Return the `n_peaks` most prominent frequency components of `waveform`, as a list of (frequency_Hz, amplitude) tuples sorted by descending amplitude. |

#### `pinneapple_perception.change_detection`

Change detection between two multispectral images of the same place: per-band mean absolute reflectance change and NDVI change against thresholds, a comparison figure and a self-contained HTML report.

<small>`pinneapple_perception/change_detection.py` · 288 linhas · testes: 13 · palavras-chave no código (heurística; o texto acima diz o status real): placeholder/stub citado no docstring</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BandChangeResult` | classe | `(band, mean_abs_diff_raw, mean_abs_diff_reflectance, threshold, exceeds_threshold)` |  |
| `NdviChangeResult` | classe | `(mean_abs_diff, threshold, exceeds_threshold)` |  |
| `PairAnalysis` | classe | `(date_a, date_b, band_results, ndvi_result, any_change, narrative_html)` |  |
| `band_difference` | função | `(img_a, img_b, band_index, band_name, threshold=DEFAULT_BAND_DIFF_THRESHOLD, reflectance_…)` | Real per-band mean-absolute-difference change check, same computation as the original notebook's `detecta_mudanca` (mean of \|img_b - img_a\| for one band, normalized, compared to a threshold), but with an explicit, physically-meaningful `reflectance_scale` (default the real Sentinel-2 SR scale, … |
| `compute_ndvi` | função | `(red, nir, eps=1e-06) -> np.ndarray` | Real NDVI: (NIR - RED) / (NIR + RED + eps). `red` and `nir` must both be REAL band arrays (Sentinel-2 B4 and B8 respectively) -- this function has no fallback that fabricates either band. Same formula as the original notebook's `calcula_ndvi`, applied to a real NIR input instead of the original's … |
| `ndvi_change` | função | `(ndvi_a, ndvi_b, threshold=DEFAULT_NDVI_THRESHOLD) -> NdviChangeResult` |  |
| `analyze_pair` | função | `(rgb_a, rgb_b, date_a, date_b, nir_a=None, nir_b=None, band_diff_threshold=DEFAULT_BAND_D…)` | Real per-pair change analysis: three band-difference checks (red/green/blue) always run; NDVI is computed and checked only if real `nir_a`/`nir_b` arrays are supplied (never fabricated). `date_a` should chronologically precede `date_b` -- the caller is responsible for ordering (this fixes a real, … |
| `rgb_preview_stretch` | função | `(img, normalizer=…) -> np.ndarray` | Purely a display contrast stretch (clip to [0,1] after dividing by a fixed constant) -- matches the original notebook's `img/3000` preview trick. This has NO physical-reflectance meaning (see this module's docstring); it is only ever used for the RGB preview panels, never for the quantitative … |
| `render_comparison_figure` | função | `(rgb_a, rgb_b, date_a, date_b, ndvi_a=None, ndvi_b=None) -> bytes` | Render the same-style comparison grid the original notebook produced (RGB pair + diff, per-band diff maps, and -- when real NDVI arrays are supplied -- an NDVI before/after/delta row), as real PNG bytes (matplotlib `Agg` backend, no display needed). |
| `render_html_report` | função | `(pairs, images_png, lon, lat, buffer_m, has_ndvi) -> str` |  |

#### `pinneapple_perception.image_geometry`

Extract boundary/geometry point clouds from an image -- e.g. a photo or scan of a physical part's cross-section -- for use as a domain boundary in `pinneapple_design.geometry` or as boundary-condition selector data. Depends only on numpy/scipy (already core PINNeAPPle dependencies).

<small>`pinneapple_perception/image_geometry.py` · 114 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `extract_boundary_points` | função | `(image, threshold=None, units_per_pixel=1.0, ordered=True) -> np.ndarray` | Extract the boundary of the foreground region(s) in `image` as a point cloud. |
| `estimate_bounding_circle` | função | `(points) -> Tuple[float, float, float]` | Least-squares circle fit (algebraic Kasa method) to a boundary point cloud -- a quick, standard way to validate/characterize a roughly-circular extracted boundary (e.g. a pipe or shaft cross-section). Returns (center_x, center_y, radius). |

#### `pinneapple_perception.piv_tools`

Around :mod:`.video_piv`: reading frames from a video or image files, the vector-grid size a setting gives, a summary of a PIV result, and window / search-margin advice from the expected displacement.

<small>`pinneapple_perception/piv_tools.py` · 128 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FFmpegMissing` | classe | `()` | ``ffmpeg`` or ``ffprobe`` is not on PATH. |
| `decode_video` | função | `(path) -> np.ndarray` | (T, H, W) float64 grayscale frames of the video at ``path`` (ffprobe for the size, ffmpeg for the pixels). |
| `decode_image_frames` | função | `(files) -> np.ndarray` | (T, H, W) grayscale frames from ``(file name, bytes)`` pairs, sorted by file name (use zero-padded names). |
| `piv_grid_count` | função | `(height, width, window_size, search_margin, step=None, *, margin=None) -> tuple[int, int]` | (rows, cols) of interrogation windows ``piv_velocity_field`` uses on an H x W frame. |
| `summarize_piv` | função | `(seq, frame_shape, window_size, search_margin, step=None) -> dict` | Counts and speeds of a ``piv_velocity_sequence`` result on frames of shape (T, H, W). |
| `piv_parameter_guide` | função | `(displacement_px, width_px, height_px) -> dict` | Window size, search margin and step for an expected displacement (px per frame), with warnings. |

#### `pinneapple_perception.sentinel2`

Sentinel-2 surface-reflectance images of a point from Google Earth Engine, as arrays for :mod:`pinneapple_perception.change_detection` (bands B4, B3, B2 and the real NIR band B8).

<small>`pinneapple_perception/sentinel2.py` · 136 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EarthEngineUnavailable` | classe | `()` | Raised when ee.Initialize() genuinely fails -- no credential, no project, or the Earth Engine API not enabled for the given project. Carries the real underlying exception's message plus the concrete remediation steps from this module's docstring, so a caller sees exactly what a real deployment … |
| `Sentinel2Frame` | classe | `(date, rgb, nir)` |  |
| `ee_initialize` | função | `(project=None) -> None` | Real ee.Initialize() call. Tries a service account credential if EE_SERVICE_ACCOUNT_EMAIL + EE_SERVICE_ACCOUNT_KEY_FILE are set (the real, non-interactive path for a server deployment); otherwise falls through to ee's default persistent-credential lookup (populated by a prior real `earthengine … |
| `fetch_sentinel2_sequence` | função | `(lon, lat, buffer_m=5000.0, start_date='2024-01-01', end_date='2025-07-27', max_cloud_pct…)` | Real fetch of the n_images most recent, least-cloudy Sentinel-2 SR images intersecting a buffered point, returned oldest-to-newest (chronological order -- the original notebook fetched newest-first and then had two of its own cells label "before/after" in opposite directions for the same pairs; … |

#### `pinneapple_perception.video_piv`

Extract a velocity field from a video (image pair/sequence) using cross-correlation Particle Image Velocimetry (PIV) -- the actual standard technique experimental fluid dynamicists use to turn a video of a seeded/textured flow into a velocity field, not a generic computer- vision optical-flow …

<small>`pinneapple_perception/video_piv.py` · 221 linhas · testes: 6 · palavras-chave no código (heurística; o texto acima diz o status real): experimental</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `piv_velocity_field` | função | `(frame_a, frame_b, window_size=32, search_margin=16, step=None, dt=1.0, units_per_pixel=1…)` | Cross-correlation PIV velocity field between two grayscale frames. |
| `piv_velocity_sequence` | função | `(frames, dt=1.0, **piv_kwargs) -> Dict[str, np.ndarray]` | Run `piv_velocity_field` on every consecutive pair in a video (`frames`: (T, H, W) array) and stack the results with a `t` column, ready for a 3D (x, y, t) `DataConstraint`. |

### Exemplos de uso reais do repositório

_Nenhum exemplo encontrado no repositório para este pacote._

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 5 de 7.
- Funções de teste em arquivos que importam algum módulo do pacote: 29.

### Limitações conhecidas

- _Nenhuma registrada._

## 8. `pinneapple_quantum`

### Objetivo

`pinneapple_quantum` (PINNeAPPle Quantum Module, versão 0.1.0) é uma extensão experimental de aprendizado de máquina informado por física híbrido clássico-quântico. Ele usa circuitos quânticos variacionais (VQC) como ansatz de função de onda, ou como camada dentro de uma rede clássica, e os treina com perdas da equação de Schrödinger.

Os fluxos previstos são: VQ-PINN (o circuito é ψ(x) e a perda é o quociente de Rayleigh, como no VQE), PINN híbrida (MLP clássica + circuito), aprendizado de Hamiltoniana a partir de energias medidas e kernel quântico para métodos de kernel clássicos. `pipeline.presets.solve()` imita a API `solve` do PINNeAPPle para esses casos.

A execução pode usar PennyLane, Qiskit/Aer ou um simulador clássico de vetor de estado em PyTorch. Se as bibliotecas quânticas não estiverem instaladas, o pacote cai automaticamente para o simulador clássico, então a API toda continua disponível. O pacote é isolado do resto do repositório: não importa outros pacotes `pinneapple_*` e nenhum teste o importa.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 16 | 2.636 | 17 | 24 | 0 (0 arquivos) | 16 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `backends` | Backends PennyLane, Qiskit e simulador clássico. | 2 | 7 | 0 |
| `circuits` | Configuração e construção de circuitos variacionais. | 2 | 4 | 0 |
| `data` | Estados, observáveis, datasets e amostradores de domínio. | 2 | 7 | 0 |
| `loss` | Perdas da equação de Schrödinger e potenciais. | 2 | 10 | 0 |
| `models` | `QuantumModel` e `HybridModel` como módulos PyTorch. | 3 | 2 | 0 |
| `pipeline` | Presets de alto nível e `solve()`. | 2 | 5 | 0 |
| `training` | Treinador híbrido e gradiente parameter-shift. | 2 | 4 | 0 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_quantum`

Fachada do Módulo Quântico (PQM): reexporta configuração de circuito, backends, modelos, perdas de Schrödinger, datasets, treinador e presets, e oferece os atalhos `build_circuit` e `backend_info`. [experimental] [sem teste]

<small>`pinneapple_quantum/__init__.py` · 210 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_circuit` | função | `(n_qubits=4, depth=2, ansatz='hardware_efficient', encoding='angle', n_observables=1, bac…)` | Atalho que monta um `QuantumModel` a partir de argumentos simples (qubits, profundidade, ansatz, codificação, backend). |
| `backend_info` | função | `(provider='pennylane') -> dict` | Devolve o dict de informações do backend pedido (ou do fallback resolvido). |

#### `pinneapple_quantum.backends`

Subpacote dos backends de execução quântica (reexporta `backend`). [sem teste]

<small>`pinneapple_quantum/backends/__init__.py` · 4 linhas · **sem teste direto**</small>

#### `pinneapple_quantum.backends.backend`

Adaptador de backend quântico: PennyLane, Qiskit/Aer ou simulador clássico de vetor de estado em PyTorch; `get_backend` cai para o clássico se a biblioteca não estiver instalada. A base `QuantumBackend` levanta NotImplementedError. [sem teste]

<small>`pinneapple_quantum/backends/backend.py` · 324 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BackendConfig` | classe | `(provider, device_name, diff_method, shots)` | Configuração do backend: provedor, nome do dispositivo, método de diferenciação e número de shots. |
| `QuantumBackend` | classe | `(config)` | Base abstrata de backend; `name` e `build_qnode` levantam NotImplementedError. |
| `PennyLaneBackend` | classe | `()` | Backend PennyLane (simuladores CPU/GPU e QPUs reais). |
| `QiskitBackend` | classe | `()` | Backend Qiskit/Aer via plugin PennyLane-Qiskit; sem ele, usa o backend clássico. |
| `ClassicalBackend` | classe | `()` | Simulador clássico de vetor de estado em PyTorch, sem dependências quânticas. |
| `ClassicalQNode` | classe | `(circuit_fn, n_qubits)` | Executa uma descrição de circuito gravada aplicando as portas como unitárias complexas a um vetor de estado. |
| `get_backend` | função | `(config=None) -> 'QuantumBackend'` | Cria o backend a partir da configuração, caindo para o clássico se PennyLane ou Qiskit não estiverem instalados. |

#### `pinneapple_quantum.circuits`

Subpacote de construção de circuitos (reexporta `base`). [sem teste]

<small>`pinneapple_quantum/circuits/__init__.py` · 8 linhas · **sem teste direto**</small>

#### `pinneapple_quantum.circuits.base`

Define o circuito variacional (ansatz e codificação) e o constrói como QNode do PennyLane ou como gravação de portas para o simulador clássico; inclui `ClassicalVQC`, uma aproximação por MLP. [sem teste]

<small>`pinneapple_quantum/circuits/base.py` · 300 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `QuantumCircuitConfig` | classe | `(n_qubits, depth, ansatz, encoding, n_observables)` | Especificação do circuito variacional: qubits, profundidade, ansatz (`hardware_efficient`, `strongly_entangling`, `efficient_su2`), codificação (`angle`, `amplitude`, `iqp`) e número de observáveis. |
| `ClassicalVQC` | classe | `(config)` | Aproximação por MLP de um circuito variacional, com saída em [-1, 1]; o próprio docstring diz que não reproduz a dinâmica das portas. |
| `build_pennylane_circuit` | função | `(config, backend)` | Constrói um QNode do PennyLane para a configuração dada. |
| `build_classical_circuit` | função | `(config, backend)` | Constrói o mesmo circuito gravando as portas num protocolo que o `ClassicalQNode` executa. |

#### `pinneapple_quantum.data`

Subpacote de dados quânticos (reexporta `quantum_dataset`). [sem teste]

<small>`pinneapple_quantum/data/__init__.py` · 20 linhas · **sem teste direto**</small>

#### `pinneapple_quantum.data.quantum_dataset`

Camada de dados quânticos: estados, observáveis, datasets de colocação e de cadeia de spins (Ising), amostradores de domínio 1D/2D e autoestados analíticos do oscilador harmônico. [sem teste]

<small>`pinneapple_quantum/data/quantum_dataset.py` · 290 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `QuantumState` | classe | `(coords, psi_real, psi_imag, meta)` | Estado quântico puro psi(x) sobre um domínio espacial, com partes real e imaginária e norma. |
| `QuantumObservable` | classe | `(expectation, variance, operator_names)` | Observável medido: valor esperado, variância e nomes dos operadores. |
| `QuantumCollocationDataset` | classe | `(x_col, x_bc, psi_exact=None, batch_size=256)` | Dataset de pontos de colocação e de contorno para treinar PINNs quânticas, com solução exata opcional. |
| `SpinChainDataset` | classe | `(n_spins=4, n_samples=1024, J=1.0, h=0.5, seed=42)` | Configurações aleatórias de spins ±1 com energia de Ising (acoplamento J e campo h). |
| `sample_domain_1d` | função | `(x_min=0.0, x_max=1.0, n_col=1000, n_bc=50, device='cpu', dtype=torch.float32) -> Tuple[t…)` | Amostra pontos de colocação e de contorno num intervalo 1D. |
| `sample_domain_2d` | função | `(x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0, n_col=4096, n_bc=200, device='cpu', dtype=…)` | Amostra pontos de colocação e de contorno num retângulo 2D. |
| `exact_eigenstates_1d_harmonic` | função | `(x, n_state=0, omega=1.0, hbar=1.0, mass=1.0) -> Tuple[torch.Tensor, float]` | Calcula o n-ésimo autoestado analítico do oscilador harmônico 1D e sua energia. |

#### `pinneapple_quantum.loss`

Subpacote de perdas físicas (reexporta `schrodinger`). [sem teste]

<small>`pinneapple_quantum/loss/__init__.py` · 26 linhas · **sem teste direto**</small>

#### `pinneapple_quantum.loss.schrodinger`

Perdas físicas para a equação de Schrödinger: potenciais prontos, energia variacional (quociente de Rayleigh), resíduo forte independente e dependente do tempo, normalização e valor esperado de Hamiltoniana matricial. [sem teste]

<small>`pinneapple_quantum/loss/schrodinger.py` · 344 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `harmonic_potential` | função | `(x, omega=1.0, mass=1.0) -> torch.Tensor` | Potencial do oscilador harmônico, V = ½ m ω² \|x\|². |
| `infinite_well_potential` | função | `(x, width=1.0) -> torch.Tensor` | Poço infinito: V = 0 no domínio, com as paredes impostas por condições de Dirichlet. |
| `coulomb_potential` | função | `(x, charge=1.0, eps=0.001) -> torch.Tensor` | Potencial coulombiano regularizado na origem (átomo hidrogenoide). |
| `double_well_potential` | função | `(x, a=1.0, b=0.0) -> torch.Tensor` | Poço duplo (x² − a)² + b x, simétrico ou assimétrico. |
| `get_potential` | função | `(name_or_fn) -> Callable` | Resolve um potencial pelo nome ou devolve a função se já for chamável. |
| `energy_loss` | função | `(model, x, potential='harmonic_oscillator', hbar=1.0, mass=1.0, **potential_kwargs) -> Tu…)` | Energia variacional (quociente de Rayleigh) da equação de Schrödinger independente do tempo; devolve energia e norma. |
| `pde_residual_loss` | função | `(model, x, eigenvalue, potential='harmonic_oscillator', hbar=1.0, mass=1.0, **potential_k…)` | Resíduo forte −(ħ²/2m)∇²ψ + Vψ − Eψ para um autovalor dado. |
| `time_dependent_loss` | função | `(model, x, t, potential='harmonic_oscillator', hbar=1.0, mass=1.0, **potential_kwargs) ->…)` | Resíduo da equação de Schrödinger dependente do tempo, com ψ complexo como duas saídas. |
| `normalization_loss` | função | `(model, x, domain_volume=1.0) -> torch.Tensor` | Restrição suave para ∫\|ψ\|² dx ≈ 1. |
| `hamiltonian_expectation` | função | `(model, x, H_matrix) -> torch.Tensor` | Valor esperado discreto ⟨ψ\|H\|ψ⟩ para Hamiltonianas em forma de matriz. |

#### `pinneapple_quantum.models`

Subpacote de modelos (reexporta `QuantumModel` e `HybridModel`). [sem teste]

<small>`pinneapple_quantum/models/__init__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_quantum.models.hybrid_model`

`HybridModel`: combina uma rede clássica e um `QuantumModel` num grafo diferenciável, em série (clássico primeiro ou quântico primeiro) ou em paralelo com fusão linear. [sem teste]

<small>`pinneapple_quantum/models/hybrid_model.py` · 112 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HybridModel` | classe | `(classical_net, quantum_model, mode='classical_first', out_dim=None)` | Modelo híbrido que compõe rede clássica e circuito quântico em modo `classical_first`, `quantum_first` ou `parallel`. |

#### `pinneapple_quantum.models.quantum_model`

`QuantumModel`: circuito variacional exposto como `nn.Module` do PyTorch, devolvendo valores esperados de Pauli-Z por amostra. [sem teste]

<small>`pinneapple_quantum/models/quantum_model.py` · 133 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `QuantumModel` | classe | `(circuit_config, backend_config=None, output_scale=1.0)` | Circuito variacional como `nn.Module`, com pesos treináveis e escala de saída. |

#### `pinneapple_quantum.pipeline`

Subpacote de presets de alto nível (reexporta `presets`). [sem teste]

<small>`pinneapple_quantum/pipeline/__init__.py` · 16 linhas · **sem teste direto**</small>

#### `pinneapple_quantum.pipeline.presets`

Presets prontos: VQ-PINN, PINN híbrido, aprendizado de Hamiltoniana, kernel quântico e um `solve()` que imita a API do PINNeAPPle. [sem teste]

<small>`pinneapple_quantum/pipeline/presets.py` · 490 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `vq_pinn` | função | `(equation='schrodinger_1d', potential='harmonic_oscillator', n_qubits=4, depth=3, ansatz=…)` | Monta e treina uma VQ-PINN para Schrödinger com o potencial escolhido e devolve resultados (energia estimada e exata, histórico, modelo). |
| `hybrid_pinn` | função | `(classical_in=1, classical_hidden=32, n_qubits=4, depth=2, potential='harmonic_oscillator…)` | Treina uma PINN híbrida: MLP clássica codifica a entrada para o circuito quântico. |
| `learn_hamiltonian` | função | `(data, labels, n_spins=4, epochs=2000, lr=0.001, device='cpu', seed=42) -> Dict[str, Any]` | Ajusta uma matriz Hamiltoniana simétrica tal que ⟨ψ\|H\|ψ⟩ aproxime as energias medidas. |
| `quantum_kernel` | função | `(x_train, x_test=None, n_qubits=4, encoding='angle', backend='pennylane') -> torch.Tensor` | Matriz de kernel a partir das expectativas de Pauli-Z do circuito (cosseno ao quadrado); no backend clássico usa um kernel RBF. |
| `solve` | função | `(problem='quantum', equation='schrodinger', potential='harmonic', method='vq_pinn', backe…)` | Ponto de entrada no estilo `solve()` do PINNeAPPle que despacha para `vq_pinn` ou `hybrid_pinn`. |

#### `pinneapple_quantum.training`

Subpacote de treino (reexporta `trainer`). [sem teste]

<small>`pinneapple_quantum/training/__init__.py` · 9 linhas · **sem teste direto**</small>

#### `pinneapple_quantum.training.trainer`

Treinador híbrido clássico-quântico com perda física, de contorno e de normalização; a opção de gradiente por parameter-shift não está implementada e cai para backprop com aviso. [sem teste]

<small>`pinneapple_quantum/training/trainer.py` · 345 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `QTrainerConfig` | classe | `(epochs, lr, optimizer, quantum_gradient, lambda_norm, lambda_bc, print_every, device)` | Configuração do treinador quântico: épocas, taxa, otimizador, modo de gradiente quântico e pesos de normalização e contorno. |
| `QTrainHistory` | classe | `(epochs, total_loss, pde_loss, energy, norm_sq)` | Histórico do treino: perda total, perda da EDP, energia e norma por época. |
| `QTrainer` | classe | `(model, physics_loss_fn, bc_loss_fn=None, config=None)` | Treinador híbrido que minimiza perda física + contorno + normalização sobre pontos de colocação. |
| `parameter_shift_gradient` | função | `(loss_fn, weights, shift=math.pi / 2) -> torch.Tensor` | Gradiente exato de `loss_fn` em relação a `weights` pela regra parameter-shift. |

### Exemplos de uso reais do repositório

**Quick start do docstring do pacote** — `pinneapple_quantum/__init__.py`

```python
>>> cfg = pqm.QuantumCircuitConfig(n_qubits=4, depth=3)
>>> model = pqm.QuantumModel(cfg)

>>> # Physics loss
>>> x = torch.linspace(-3, 3, 500).unsqueeze(-1)
>>> energy, norm = pqm.loss.energy_loss(model, x, "harmonic_oscillator")

>>> # One-line solve
>>> result = pqm.solve(
...     problem="quantum",
...     equation="schrodinger",
...     potential="harmonic",
...     method="vq_pinn",
...     n_qubits=4, depth=3, epochs=2000,
... )
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 0 de 16.
- Funções de teste em arquivos que importam algum módulo do pacote: 0.

### Limitações conhecidas

- Nenhum módulo do pacote tem teste.
- `QTrainerConfig.quantum_gradient='parameter_shift'` não está implementado no treino: o `QTrainer` emite aviso e usa backprop comum, porque a reconstrução anterior dava gradientes errados com o termo laplaciano de `energy_loss`.
- `quantum_kernel` não calcula a fidelidade exata |⟨φ(xi)|φ(xj)⟩|², e sim um proxy pelas expectativas de Pauli-Z; no backend clássico troca por um kernel RBF.
- `ClassicalVQC` é uma aproximação por MLP, não uma simulação fiel; a simulação fiel é o `ClassicalQNode` (vetor de estado).
- PennyLane e Qiskit/Aer são opcionais; sem eles tudo roda no simulador clássico, que avalia amostra por amostra.
- `QuantumBackend` é base abstrata (`name` e `build_qnode` levantam NotImplementedError).
