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

_Sem texto de objetivo._

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 198 | 36.217 | 334 | 150 | 438 (49 arquivos) | 152 |

Pasta de testes dedicada: `tests/pinneapple_neural` (36 funções de teste).

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `_licencas` |  | 1 | 4 | 20 |
| `architectures` |  | 147 | 270 | 64 |
| `callbacks` |  | 1 | 3 | 4 |
| `lpm` |  | 1 | 6 | 5 |
| `physicsnemo_bridge` |  | 4 | 15 | 19 |
| `predictor` |  | 4 | 27 | 0 |
| `trainer` |  | 34 | 132 | 151 |
| `utils` |  | 1 | 2 | 5 |
| `workflows` |  | 4 | 21 | 10 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_neural`

pinneapple_neural — Neural network architectures, training, and inference.

<small>`pinneapple_neural/__init__.py` · 256 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_model` | função | `(name, **kwargs)` | Build a model from the registry by name. |
| `train_model` | função | `(model, loss_fn, train_loader, val_loader=None, *, epochs=5000, device='cpu', **cfg_kwarg…)` | Train a model with the given loss function. |
| `predict` | função | `(model, x, *, device='cpu', batch_size=4096)` | Run batched inference on input ``x``. |
| `train_and_predict` | função | `(model, loss_fn, train_loader, x_test, *, val_loader=None, epochs=5000, device='cpu')` | Convenience: train then evaluate on test points. |

#### `pinneapple_neural._licencas`

License guards for optional third-party integrations.

<small>`pinneapple_neural/_licencas.py` · 44 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ResearchOnlyWarning` | classe | `()` | A research-only (non-production / non-commercial) component was used. |
| `ResearchOnlyError` | classe | `()` | A research-only component was requested in commercial mode. |
| `commercial_mode` | função | `() -> bool` |  |
| `require_research_use` | função | `(component, notice=NOETHER_NOTICE) -> None` | Block ``component`` in commercial mode; otherwise warn that it is research only. |

#### `pinneapple_neural.architectures`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/__init__.py` · 43 linhas · testes: 28</small>

#### `pinneapple_neural.architectures._registry_bridge`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/_registry_bridge.py` · 39 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `register_family_registry` | função | `(family_registry, *, family, description_getter=None, tags_getter=None, capabilities_gett…)` | Registers a dictionary-like family registry into the global ModelRegistry. |

#### `pinneapple_neural.architectures.adapters`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/adapters/__init__.py` · 27 linhas · **sem teste direto**</small>

#### `pinneapple_neural.architectures.adapters.ae`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/adapters/ae.py` · 28 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AutoEncoderAdapter` | classe | `()` | Adapter for autoencoders. |

#### `pinneapple_neural.architectures.adapters.base`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/adapters/base.py` · 63 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ModelAdapter` | classe | `()` | Protocol for model adapters. |
| `select_adapter` | função | `(spec) -> ModelAdapter` | Pick an adapter based on model spec. |

#### `pinneapple_neural.architectures.adapters.conv`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/adapters/conv.py` · 31 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ConvAdapter` | classe | `()` | Adapter for grid/raster models (Conv1D/2D/3D, AFNO, UNO-grid-mode). |

#### `pinneapple_neural.architectures.adapters.dynamics`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/adapters/dynamics.py` · 32 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DynamicsAdapter` | classe | `()` | Adapter for time-evolution / neural-ODE-style models. |

#### `pinneapple_neural.architectures.adapters.gnn`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/adapters/gnn.py` · 39 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GNNAdapter` | classe | `()` | Adapter for graph/mesh-based models. |

#### `pinneapple_neural.architectures.adapters.operators`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/adapters/operators.py` · 49 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OperatorAdapter` | classe | `()` | Forward adapter for Neural Operators. |

#### `pinneapple_neural.architectures.adapters.pinn`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/adapters/pinn.py` · 37 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PINNAdapter` | classe | `()` | Adapter for pointwise coordinate PINNs. |

#### `pinneapple_neural.architectures.adapters.ts`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/adapters/ts.py` · 44 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TimeSeriesAdapter` | classe | `()` | Adapter for time-series models. |

#### `pinneapple_neural.architectures.afno`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/afno.py` · 264 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AFNOLayer` | classe | `(hidden_dim, n_modes_h=12, n_modes_w=12, mlp_ratio=4.0, dropout=0.0, num_blocks=8, hidden…)` | Single AFNO layer: block-diagonal token mixing in Fourier space + channel MLP. |
| `AFNO` | classe | `(in_channels, out_channels, hidden_dim=64, n_layers=4, n_modes_h=12, n_modes_w=12, mlp_ra…)` | Adaptive Fourier Neural Operator for structured-grid PDEs. |

#### `pinneapple_neural.architectures.autoencoders`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/autoencoders/__init__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_neural.architectures.autoencoders.ae_2d`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/autoencoders/ae_2d.py` · 133 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Autoencoder2D` | classe | `(in_channels, latent_dim, img_size, base_channels=32, norm='group', gn_groups=8, output_a…)` | Conv2D autoencoder for images / 2D fields. |

#### `pinneapple_neural.architectures.autoencoders.ae_rom_hybrid`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/autoencoders/ae_rom_hybrid.py` · 320 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AEROMHybrid` | classe | `(input_dim, latent_dim, hidden=(512, 256), activation='gelu', rom_weight=1.0, pred_weight…)` | AE-ROM Hybrid (Koopman/latent linear dynamics) com: - AE denso (encoder/decoder) - Dinâmica contínua linear no latente: dz/dt = K z + c - Passo discreto via Euler: z_next_pred = z + dt*(K z + c) - Loss de previsão em x (decode do z_next_pred) - Rollout multi-step (latente e x) - Regularização de … |

#### `pinneapple_neural.architectures.autoencoders.base`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/autoencoders/base.py` · 51 linhas · testes: 5 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AEOutput` | classe | `(x_hat, z, losses, extras)` | Standardized output for all AEs. |
| `AEBase` | classe | `()` | Base class for all autoencoders in Pinneapple. |

#### `pinneapple_neural.architectures.autoencoders.dense_ae`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/autoencoders/dense_ae.py` · 54 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DenseAutoencoder` | classe | `(input_dim, latent_dim, hidden=(512, 256), activation='gelu')` | Plain MLP autoencoder for vector inputs. |

#### `pinneapple_neural.architectures.autoencoders.kae`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/autoencoders/kae.py` · 177 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `KAEAutoencoder` | classe | `(input_dim, latent_dim, hidden=(512, 256), activation='gelu', mmd_weight=1.0, mmd_sigma=1…)` | Practical MMD-WAE style Autoencoder: - Dense AE - MMD penalty on latent space to match a simple prior. |

#### `pinneapple_neural.architectures.autoencoders.koopman_pi_ae`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/autoencoders/koopman_pi_ae.py` · 331 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsInformedKoopmanAutoencoder` | classe | `(input_dim, latent_dim, hidden=(512, 256), activation='gelu', koopman_weight=1.0, *, pred…)` | Koopman AE (prático / literature-grade): - Dense AE - Operador linear treinável no latente (K, opcionalmente com bias b) - Suporte explícito a dt via gerador contínuo A (K_eff = expm(A*dt)) se desejado - Loss no latente: \|\|z_next - K_eff z\|\|^2 - Loss de previsão no espaço original: \|\|x_next - … |

#### `pinneapple_neural.architectures.autoencoders.registry`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/autoencoders/registry.py` · 70 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AutoencoderCatalog` | classe | `(registry)` | Group entrypoint: cat = AutoencoderCatalog() model = cat.build("vae", input_dim=..., latent_dim=...) |
| `register_into_global` | função | `() -> None` |  |

#### `pinneapple_neural.architectures.autoencoders.vae`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/autoencoders/vae.py` · 122 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `VariationalAutoencoder` | classe | `(input_dim, latent_dim, hidden=(512, 256), beta=1.0, activation='gelu', clamp_logvar=(-10…)` | MLP VAE for vector inputs. |

#### `pinneapple_neural.architectures.base`

BaseModel and ModelOutput for Pinneapple model family.

<small>`pinneapple_neural/architectures/base.py` · 93 linhas · testes: 5 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ModelOutput` | classe | `(y, losses, extras)` |  |
| `BaseModel` | classe | `(family, name)` | Unified base class for all Pinneapple models. |

#### `pinneapple_neural.architectures.benchmarks`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/benchmarks/__init__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_neural.architectures.benchmarks.generic_pinn_models`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/benchmarks/generic_pinn_models.py` · 122 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FourierFeatures` | classe | `(in_dim, num_f=64, scale=10.0)` |  |
| `GenericMLP` | classe | `(in_dim=3, out_dim=1, width=128, depth=4, act='tanh')` |  |
| `GenericFourierMLP` | classe | `(in_dim=3, out_dim=1, num_f=64, scale=10.0, width=128, depth=4)` |  |
| `Sine` | classe | `(w0=30.0)` |  |
| `GenericSIREN` | classe | `(in_dim=3, out_dim=1, width=128, depth=4, w0=30.0)` |  |
| `GenericResMLP` | classe | `(in_dim=3, out_dim=1, width=128, depth=6)` |  |
| `GenericLinear` | classe | `(in_dim=3, out_dim=1)` |  |

#### `pinneapple_neural.architectures.benchmarks.registry`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/benchmarks/registry.py` · 32 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `register_into_global` | função | `() -> None` | Register benchmark models into the global ModelRegistry. |

#### `pinneapple_neural.architectures.catalog`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/catalog.py` · 33 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ModelCatalog` | classe | `(autoencoders, pinns, transformers, recurrent, convolutions, neural_operators, continuous…)` |  |

#### `pinneapple_neural.architectures.classical_ts`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/classical_ts/__init__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_neural.architectures.classical_ts.arima`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/classical_ts/arima.py` · 250 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ARIMA` | classe | `(dim=1, p=3, d=0, q=0, l2=1e-06, use_bias=True)` | ARIMA(p,d,q) MVP without external deps: |

#### `pinneapple_neural.architectures.classical_ts.base`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/classical_ts/base.py` · 34 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ClassicalTSOutput` | classe | `(y, losses, extras)` | Output for classical time-series / filtering models. |
| `ClassicalTSBase` | classe | `()` |  |

#### `pinneapple_neural.architectures.classical_ts.ekf`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/classical_ts/ekf.py` · 125 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ExtendedKalmanFilter` | classe | `(f, h, Q, R, F_jac, H_jac)` | EKF (robust MVP): x_t = f(x_{t-1}, u_t) + w y_t = h(x_t) + v |

#### `pinneapple_neural.architectures.classical_ts.enkf`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/classical_ts/enkf.py` · 122 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EnsembleKalmanFilter` | classe | `(f, h, Q, R, *, ensemble_size=64)` | Ensemble Kalman Filter (EnKF) MVP: |

#### `pinneapple_neural.architectures.classical_ts.kalman`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/classical_ts/kalman.py` · 130 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `KalmanFilter` | classe | `(A, H, Q, R, *, B=None)` | Linear Kalman Filter (batch, robust): |

#### `pinneapple_neural.architectures.classical_ts.registry`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/classical_ts/registry.py` · 67 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ClassicalTSCatalog` | classe | `(registry)` |  |
| `register_into_global` | função | `() -> None` |  |

#### `pinneapple_neural.architectures.classical_ts.tcn`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/classical_ts/tcn.py` · 89 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TCN` | classe | `(in_dim, out_dim, *, channels=(64, 64, 64), kernel=3, dropout=0.1)` | TCN for sequence-to-sequence regression. |

#### `pinneapple_neural.architectures.classical_ts.ukf`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/classical_ts/ukf.py` · 254 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `UnscentedKalmanFilter` | classe | `(f, h, Q, R, *, alpha=0.001, beta=2.0, kappa=0.0, jitter=1e-06, jitter_growth=10.0, max_c…)` | UKF (sigma-point filter), batch-friendly. |

#### `pinneapple_neural.architectures.classical_ts.var`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/classical_ts/var.py` · 165 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `VAR` | classe | `(dim, p=1, l2=1e-06, use_bias=True)` | VAR(p) with ridge regression (closed form). |

#### `pinneapple_neural.architectures.continuous`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/continuous/__init__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_neural.architectures.continuous.base`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/continuous/base.py` · 38 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ContOutput` | classe | `(y, losses, extras)` | Standard output for continuous-time / probabilistic dynamics models. |
| `ContinuousModelBase` | classe | `()` |  |

#### `pinneapple_neural.architectures.continuous.bayesian_rnn`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/continuous/bayesian_rnn.py` · 200 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LockedDropout` | classe | `(p=0.0)` | Locked (aka variational) dropout for sequences: uses the same dropout mask across the time dimension T. |
| `BayesianRNN` | classe | `(input_dim, out_dim, hidden_dim=128, num_layers=2, dropout=0.1, cell='gru', min_logvar=-1…)` | Bayesian RNN (MVP via MC Dropout / variational dropout approximation). |

#### `pinneapple_neural.architectures.continuous.deep_state_space`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/continuous/deep_state_space.py` · 214 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ContOutput` | classe | `(y, losses, extras)` |  |
| `DeepStateSpaceModel` | classe | `(input_dim, out_dim, latent_dim=32, hidden_dim=128, mlp_depth=2, min_logvar=-10.0, max_lo…)` | Variational Deep State Space Model (DSSM / deep SSM) - MVP (filtering): |
| `gaussian_nll_diag` | função | `(mu, logvar, y) -> torch.Tensor` |  |
| `kl_diag_gaussians` | função | `(mu_q, logvar_q, mu_p, logvar_p) -> torch.Tensor` |  |

#### `pinneapple_neural.architectures.continuous.hamiltonian`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/continuous/hamiltonian.py` · 69 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HamiltonianNeuralNetwork` | classe | `(dim_q, hidden=128, num_layers=3)` | HNN MVP: - Learn Hamiltonian H(q,p) - Dynamics: dq/dt = dH/dp, dp/dt = -dH/dq |

#### `pinneapple_neural.architectures.continuous.latent_ode`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/continuous/latent_ode.py` · 249 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LatentODE` | classe | `(obs_dim, latent_dim=32, hidden=128, *, ode_method='dopri5', enc_method='dopri5', min_log…)` | Latent ODE (improved, literature-aligned): - Encoder: GRU-ODE recognition model (handles irregular t) - Latent dynamics: NeuralODE with adaptive solver (default dopri5) - Decoder: Gaussian likelihood (mu_x, log_sigma_x) -> NLL |

#### `pinneapple_neural.architectures.continuous.neural_cde`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/continuous/neural_cde.py` · 156 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NeuralCDE` | classe | `(input_dim, hidden_dim, out_dim, *, f_hidden=128, g_hidden=128, num_layers=2, activation=…)` | Neural CDE (literature-style MVP, no torchcde): |

#### `pinneapple_neural.architectures.continuous.neural_gp`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/continuous/neural_gp.py` · 248 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NeuralGaussianProcess` | classe | `(in_dim, out_dim, *, embed_dim=64, mlp_hidden=128, mlp_layers=2, rff_features=512, rff_le…)` | Neural Gaussian Process (RFF + Bayesian Linear head): |

#### `pinneapple_neural.architectures.continuous.neural_ode`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/continuous/neural_ode.py` · 91 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NeuralODE` | classe | `(state_dim, hidden=128, num_layers=3, method='rk4')` | Neural ODE MVP: dy/dt = f(t, y) where f is a neural net. Integrate over time grid with Euler or RK4. |

#### `pinneapple_neural.architectures.continuous.neural_sde`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/continuous/neural_sde.py` · 264 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NeuralSDE` | classe | `(state_dim, *, hidden=128, num_layers=3, diffusion='diag', activation='tanh', sigma_eps=1…)` | Neural SDE (Euler–Maruyama) for *solving* problems via simulation: |

#### `pinneapple_neural.architectures.continuous.ode_rnn`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/continuous/ode_rnn.py` · 54 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ODERNN` | classe | `(input_dim, hidden_dim=128)` | ODE-RNN MVP: - Integrate hidden state with Neural ODE between observation times - Update with GRUCell when observation arrives |

#### `pinneapple_neural.architectures.continuous.registry`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/continuous/registry.py` · 80 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ContinuousCatalog` | classe | `(registry)` |  |
| `register_into_global` | função | `() -> None` |  |

#### `pinneapple_neural.architectures.continuous.symplectic_ode`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/continuous/symplectic_ode.py` · 147 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SymplecticODENet` | classe | `(dim_q, hidden=128)` | Symplectic ODE-Net: |

#### `pinneapple_neural.architectures.continuous.symplectic_rnn`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/continuous/symplectic_rnn.py` · 128 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SymplecticRNN` | classe | `(dim_q, *, hidden=128, num_layers=2, activation='tanh')` | Symplectic RNN (MVP, leapfrog-style update): |

#### `pinneapple_neural.architectures.convolutions`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/convolutions/__init__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_neural.architectures.convolutions.base`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/convolutions/base.py` · 36 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ConvOutput` | classe | `(y, losses, extras)` | Standard output for convolutional models in Pinneapple. |
| `ConvModelBase` | classe | `()` | Base class for convolution models. |

#### `pinneapple_neural.architectures.convolutions.conv1d`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/convolutions/conv1d.py` · 265 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Conv1DModel` | classe | `(in_channels, out_channels, *, mode='denoise', cond_channels=0, hidden_channels=64, num_b…)` | Flexible 1D Conv model supporting: - causal forecasting - denoising - operator learning - PDE 1D surrogate |

#### `pinneapple_neural.architectures.convolutions.conv2d`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/convolutions/conv2d.py` · 148 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Conv2DModel` | classe | `(in_channels, out_channels, hidden_channels=32, num_blocks=4, kernel_size=3, dropout=0.0,…)` | Conv2D regression model (ResNet-like CNN). |

#### `pinneapple_neural.architectures.convolutions.conv3d`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/convolutions/conv3d.py` · 174 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ResidualBlock3D` | classe | `(channels, kernel_size=3, dropout=0.0, norm_groups=8) -> None` | A stable 3D residual block: Conv -> GroupNorm -> GELU -> Dropout -> Conv -> GroupNorm -> GELU Residual is applied outside (so the caller can scale it). |
| `Conv3DModel` | classe | `(in_channels, out_channels, hidden_channels=32, num_blocks=4, kernel_size=3, dropout=0.0,…)` | Conv3D regression model. |

#### `pinneapple_neural.architectures.convolutions.registry`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/convolutions/registry.py` · 60 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ConvolutionCatalog` | classe | `(registry)` |  |
| `register_into_global` | função | `() -> None` |  |

#### `pinneapple_neural.architectures.graphnn`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/graphnn/__init__.py` · 8 linhas · **sem teste direto**</small>

#### `pinneapple_neural.architectures.graphnn.base`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/graphnn/base.py` · 54 linhas · testes: 15</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GraphBatch` | classe | `(x, edge_index, pos, edge_attr, mask)` | Minimal graph batch format (no external deps). |
| `GraphOutput` | classe | `(y, losses, extras)` | Standard output for GNN family. y: - node-level: (B, N, out_dim) - graph-level: (B, out_dim) (not used in this MVP) |
| `GraphModelBase` | classe | `()` | Base for graph models. |

#### `pinneapple_neural.architectures.graphnn.equivariant_gnn`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/graphnn/equivariant_gnn.py` · 229 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EquivariantGNN` | classe | `(node_dim, pos_dim, out_dim, *, hidden=128, layers=4, edge_dim=0, dropout=0.0, update_pos…)` | EGNN-style MVP (E(n)-equivariant): |

#### `pinneapple_neural.architectures.graphnn.gnn`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/graphnn/gnn.py` · 123 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GraphNeuralNetwork` | classe | `(node_dim, out_dim, *, edge_dim=0, hidden=128, layers=4, dropout=0.0)` | Message Passing Neural Network (MVP). |

#### `pinneapple_neural.architectures.graphnn.gnn_ode`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/graphnn/gnn_ode.py` · 212 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FourierTimeFeatures` | classe | `(n_frequencies, *, max_frequency=10.0)` | Fourier features for scalar time t. Produces [sin(2π f_i t), cos(2π f_i t)] features. |
| `GraphNeuralODE` | classe | `(node_dim, *, edge_dim=0, hidden=128, layers=3, method='rk4', time_features=0, time_max_f…)` | Graph Neural ODE (MVP): |

#### `pinneapple_neural.architectures.graphnn.graphcast`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/graphnn/graphcast.py` · 289 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GraphCast` | classe | `(node_in, out_dim, *, hidden=256, processor_blocks=8, pos_dim=0, edge_dim=0, dropout=0.0)` | GraphCast-inspired MVP (fixed mesh + variable graph support). |

#### `pinneapple_neural.architectures.graphnn.mesh_graph_net`

MeshGraphNet — Encoder-Process-Decoder GNN for mesh-based simulation.

<small>`pinneapple_neural/architectures/graphnn/mesh_graph_net.py` · 375 linhas · testes: 15 · palavras-chave no código (heurística; o texto acima diz o status real): placeholder/stub citado no docstring</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MeshGraphNet` | classe | `(node_in_dim, out_dim, *, edge_in_dim=0, pos_dim=0, hidden_dim=128, n_layers=2, n_message…)` | Encoder-Process-Decoder GNN for unstructured mesh simulation. |

#### `pinneapple_neural.architectures.graphnn.mgn_dynamics`

Transient mesh dynamics on top of :class:`MeshGraphNet`.

<small>`pinneapple_neural/architectures/graphnn/mgn_dynamics.py` · 270 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MeshGraph` | classe | `(edge_index, edge_attr, node_type)` | One static mesh: topology, raw edge features and node types. |
| `Normalizer` | classe | `(dim, eps=1e-08) -> None` | Per-feature standardiser with fixed statistics (stored as buffers). |
| `MeshDynamicsMGN` | classe | `(vel_dim=2, num_node_types=4, edge_dim=3, out_p=True, **mgn_kwargs) -> None` | Autoregressive MeshGraphNet for fields ``(vel, p)`` on a transient mesh. |
| `triangles_to_edges` | função | `(cells) -> torch.Tensor` | Bidirectional, de-duplicated edge_index (2, E) from triangle ``cells`` (T, 3). |
| `edge_features` | função | `(pos, edge_index) -> torch.Tensor` | Relative displacement (dst - src) and its norm → (E, pos_dim + 1). |
| `train_dynamics` | função | `(model, data, *, steps, lr=0.0001, lr_final=1e-06, noise_std=0.02, device='cpu', log_ever…)` | One-step training on random ``(trajectory, time)`` pairs with Adam and exponential LR decay. |
| `eval_rollout` | função | `(model, data, *, n_steps, device='cpu', normal_type=0) -> dict[str, float]` | Velocity RMSE on NORMAL nodes: mean over trajectories of the ``n_steps`` free rollout and of one step. |
| `frozen_ic_rmse` | função | `(data, *, n_steps, normal_type=0) -> float` | Baseline for :func:`eval_rollout`: velocity RMSE on NORMAL nodes when the velocity stays at its t = 0 value over the same horizon, mean over trajectories. |

#### `pinneapple_neural.architectures.graphnn.registry`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/graphnn/registry.py` · 64 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GraphCatalog` | classe | `(registry)` |  |
| `register_into_global` | função | `() -> None` |  |

#### `pinneapple_neural.architectures.graphnn.spatiotemporal_gnn`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/graphnn/spatiotemporal_gnn.py` · 117 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SpatiotemporalGNN` | classe | `(node_in, out_dim, *, edge_dim=0, spatial_hidden=128, spatial_layers=3, temporal_hidden=1…)` | Spatiotemporal GNN (MVP): |

#### `pinneapple_neural.architectures.graphnn.utils`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/graphnn/utils.py` · 84 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `scatter_add` | função | `(src, index, dim_size) -> torch.Tensor` | Scatter-add with shared-graph batching. |
| `scatter_add_flat` | função | `(src, index, dim_size) -> torch.Tensor` | Flattened variable-graph batching. src: (E_total, F) index: (E_total,) destination indices [0..N_total-1] returns: (N_total, F) where out[index[e], :] += src[e, :] |
| `flatten_graph_batch` | função | `(x_list, edge_index_list, pos_list=None, edge_attr_list=None)` | Turns per-sample graphs into a single flattened graph with node index offsets. |

#### `pinneapple_neural.architectures.group_b_registry`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/group_b_registry.py` · 86 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GroupBCatalog` | classe | `(registry)` |  |
| `register_into_global` | função | `() -> None` |  |

#### `pinneapple_neural.architectures.hash_grid`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/hash_grid.py` · 219 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HashGridEncoding` | classe | `(in_dim=3, n_levels=16, n_features_per_level=2, log2_hashmap_size=19, base_resolution=16,…)` | Multi-resolution hash grid encoding (Instant-NGP style). |
| `HashGridMLP` | classe | `(in_dim=3, out_dim=1, hidden_dim=64, n_hidden=2, n_levels=8, n_features_per_level=2, log2…)` | MLP with multi-resolution hash grid encoding for fast 3-D field representation. |

#### `pinneapple_neural.architectures.instantiate`

Model instantiation helpers.

<small>`pinneapple_neural/architectures/instantiate.py` · 173 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `InstantiateReport` | classe | `(original, normalized, dropped, used_aliases)` | Diagnostics about how kwargs were normalized. |
| `normalize_kwargs` | função | `(kwargs) -> Tuple[Dict[str, Any], Dict[str, str]]` | Normalize user kwargs to a canonical set. |
| `filter_supported_kwargs` | função | `(model_cls, kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]` | Keep only kwargs accepted by ``model_cls.__init__``. |
| `instantiate` | função | `(model_cls, kwargs=None, *, strict=False) -> Tuple[BaseModel, InstantiateReport]` | Instantiate a model class with normalized kwargs. |

#### `pinneapple_neural.architectures.modified_mlp`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/modified_mlp.py` · 136 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FourierFeatureEmbedding` | classe | `(in_dim, n_fourier=64, sigma=1.0, trainable=False)` | Random Fourier feature embedding for coordinate inputs. |
| `ModifiedMLP` | classe | `(in_dim, out_dim, hidden_dim=128, n_layers=6, n_fourier=32, sigma=1.0, activation=nn.Tanh…)` | Modified MLP with Fourier feature embedding and highway U/V connections. |

#### `pinneapple_neural.architectures.neural_operators`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/neural_operators/__init__.py` · 51 linhas · testes: 2</small>

#### `pinneapple_neural.architectures.neural_operators.base`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/neural_operators/base.py` · 40 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OperatorOutput` | classe | `(y, losses, extras)` | Standard output for neural operators. |
| `NeuralOperatorBase` | classe | `()` | Base class for Neural Operators. |

#### `pinneapple_neural.architectures.neural_operators.deeponet`

DeepONet operator learning with branch-trunk architecture (Lu et al., Nat. Mach. Intell. 3, 2021).

<small>`pinneapple_neural/architectures/neural_operators/deeponet.py` · 79 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DeepONet` | classe | `(branch_dim, trunk_dim, out_dim, hidden=128, modes=64, depth=1, trunk_activation=False)` | Unstacked DeepONet, one set of branch coefficients per output channel: |
| `mlp` | função | `(in_dim, hidden, out_dim, depth=1, act=nn.GELU, last_act=False) -> nn.Sequential` | MLP with hidden widths ``hidden`` (a list) or ``depth`` layers of width ``hidden`` (an int). |

#### `pinneapple_neural.architectures.neural_operators.fno`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/neural_operators/fno.py` · 530 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FourierNeuralOperator` | classe | `(in_channels, out_channels, width=64, modes=16, layers=4, *, use_grid=False)` | FNO-1D (MVP) - extendable to 2D/3D by changing spectral conv + projections. |
| `FNO2d` | classe | `(in_channels, out_channels, width=32, modes1=12, modes2=12, layers=4, *, use_grid=True)` | FNO-2D: Fourier Neural Operator for 2D spatial fields. |
| `FNO3d` | classe | `(in_channels, out_channels, width=32, modes1=8, modes2=8, modes3=8, layers=4, *, use_grid…)` | FNO-3D: Fourier Neural Operator for 3D spatial (or 2D-space + time) fields -- the natural fit for a structured, periodic-in-2-of-3-axes domain like a periodic channel (matches the setup FNO was originally demonstrated on: Li et al. 2020, Navier-Stokes on a periodic torus). |
| `MLPFNOSurrogate` | classe | `(in_channels, out_channels, fno_width=32, fno_modes=12, fno_layers=4, mlp_hidden=128, mlp…)` | Hybrid FNO backbone + MLP decoder for scattered-point evaluation. |

#### `pinneapple_neural.architectures.neural_operators.fno_export`

ONNX export of Fourier neural operators.

<small>`pinneapple_neural/architectures/neural_operators/fno_export.py` · 121 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RealSpectralConv2d` | classe | `(conv, H, W)` | Real-arithmetic equivalent of :class:`~.fno._SpectralConv2d` on an ``H x W`` grid. |
| `to_exportable` | função | `(model, grid) -> nn.Module` | A copy of ``model`` with every 2-D spectral convolution in real arithmetic for an ``H x W`` grid. |
| `export_onnx` | função | `(model, example, path, *, grid=None, opset=17, atol=0.0001) -> float` | Export ``model`` (an FNO, or any module containing 2-D spectral convolutions) to ONNX at ``path`` with a dynamic batch axis, check onnxruntime against PyTorch on ``example`` and return the largest absolute difference (raises when it exceeds ``atol`` times the output scale). ``grid`` defaults to the … |

#### `pinneapple_neural.architectures.neural_operators.gno`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/neural_operators/gno.py` · 146 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GalerkinNeuralOperator` | classe | `(in_dim, out_dim, basis_dim=64, basis_hidden=128, op_hidden=128, op_depth=2, orthonormali…)` | Galerkin Neural Operator (mesh-based, 1D/2D/3D): - Learn a basis Phi(x) from coords (x in R^d, d can be 1,2,3) - Project u -> coefficients using quadrature/lumped-mass weights - Apply operator in coefficient space - Reconstruct y = Phi * c_out |

#### `pinneapple_neural.architectures.neural_operators.ms_deeponet`

Multi-scale DeepONet: one trunk per spatial scale, each seeing the coordinates stretched by its own factor (the MscaleDNN idea, Liu, Cai and Xu 2020), so high-frequency parts of the output are learned by trunks that see them as low frequencies. The branch gives the coefficients of all scales.

<small>`pinneapple_neural/architectures/neural_operators/ms_deeponet.py` · 69 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MultiScaleDeepONet` | classe | `(branch_dim, trunk_dim, out_dim, *, hidden=128, scales=(32, 64, 128), scale_factors=None,…)` | y(u, x) = sum_s < B_s(u), T_s(c_s x) > + bias |

#### `pinneapple_neural.architectures.neural_operators.noether_bridge`

Noether (Emmi AI) architecture bridge for PINNeAPPle.

<small>`pinneapple_neural/architectures/neural_operators/noether_bridge.py` · 296 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NoetherModelConfig` | classe | `(model_name, config_kwargs, device)` | Minimal config wrapper forwarded verbatim to noether's own config classes. |
| `NoetherUPT` | classe | `(**config_kwargs)` | Universal Physics Transformer (Noether/Emmi AI). |
| `NoetherABUPT` | classe | `(**config_kwargs)` | Anchored-Branched Universal Physics Transformer (Noether/Emmi AI). |
| `NoetherTransformer` | classe | `(**config_kwargs)` | Vanilla Transformer backbone from Noether. |
| `NoetherTransolver` | classe | `(**config_kwargs)` | Transolver / Transolver++ Physics-Attention backbone (Noether/Emmi AI). |
| `NoetherAeroUPT` | classe | `(**config_kwargs)` | AeroUPT — UPT with surface/volume domain routing for aerodynamics. |
| `NoetherAeroABUPT` | classe | `(**config_kwargs)` | AeroABUPT — AB-UPT with surface/volume domain routing. |
| `NoetherAeroTransformer` | classe | `(**config_kwargs)` | AeroTransformer — Transformer for CFD aerodynamics. |
| `NoetherAeroTransolver` | classe | `(**config_kwargs)` | AeroTransolver — Transolver for CFD aerodynamics. |

#### `pinneapple_neural.architectures.neural_operators.pino`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/neural_operators/pino.py` · 232 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsInformedNeuralOperator` | classe | `(operator, *, lambda_pde=1.0, lambda_bc=1.0, lambda_ic=1.0, lambda_physics=1.0, prefix='p…)` | Generic PINO wrapper. |
| `GridSpec` | classe | `(L, dims)` | Descreve grid uniforme (PINO clássico). L: tamanho físico por eixo (mesmo length de dims). dims: quais dims de u correspondem aos eixos (ex.: (-2, -1) para 2D). |
| `ResidualSpec` | classe | `(residual, orders, method)` | Spec genérico para PDE residual. residual(u=..., deriv=..., params=..., **extra) -> r onde deriv é o dict retornado por compute_derivatives. |
| `pino_physics_fn` | função | `(u, *, grid, spec, params=None, bc_mask=None, bc_value=None, ic_mask=None, ic_value=None,…)` | Generic PINO physics loss: - computes derivatives on grid (FFT/FD) - evaluates residual r - returns pde/bc/ic + physics_total |
| `spectral_derivative` | função | `(u, *, dim, L, order=1) -> torch.Tensor` |  |
| `fd_derivative` | função | `(u, *, dim, dx, order=1) -> torch.Tensor` |  |
| `compute_derivatives` | função | `(u, *, grid, orders=(1, 2), method='spectral') -> Dict[Tuple[int, int], torch.Tensor]` | Retorna derivadas por eixo e ordem: key = (axis_index, order) onde axis_index é o índice dentro de grid.dims/grid.L. |

#### `pinneapple_neural.architectures.neural_operators.registry`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/neural_operators/registry.py` · 123 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NeuralOperatorCatalog` | classe | `(registry)` |  |
| `register_into_global` | função | `() -> None` |  |

#### `pinneapple_neural.architectures.neural_operators.transolver`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/neural_operators/transolver.py` · 145 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsAttention` | classe | `(dim, heads=8, slices=32, dropout=0.0)` | Multi-head Physics-Attention (slice -> attend -> deslice). |
| `TransolverBlock` | classe | `(dim, heads, slices, mlp_ratio=2, dropout=0.0)` |  |
| `Transolver` | classe | `(in_dim, out_dim, dim=128, depth=4, heads=8, slices=32, mlp_ratio=2, dropout=0.0)` | Point-cloud operator: (B, N, in_dim) -> (B, N, out_dim). |
| `TransolverLite` | classe | `(in_dim=5, out_dim=101, d_model=64, n_heads=4, n_slices=8, n_blocks=3, dropout=0.1)` | Linear input projection -> ``n_blocks`` slice-attention blocks -> per-node linear head. |

#### `pinneapple_neural.architectures.neural_operators.uno`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/neural_operators/uno.py` · 494 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SpectralConvND` | classe | `(dim, in_channels, out_channels, modes)` | Spectral convolution generalized to 1D/2D/3D (rfftn). Applies a learned complex linear map on low-frequency modes. |
| `OperatorBlockND` | classe | `(dim, width, modes)` |  |
| `DownBlockND` | classe | `(dim, in_ch, out_ch, modes, depth)` |  |
| `UpBlockND` | classe | `(dim, in_ch, skip_ch, out_ch, modes, depth)` |  |
| `UNONdGrid` | classe | `(dim, in_channels, out_channels, width=64, levels=3, depth_per_level=2, modes=None, add_c…)` | U-shaped Neural Operator for regular grids (1D/2D/3D). Input:  (B, Cin, *spatial) Output: (B, Cout, *spatial) |
| `PointKernelOperator` | classe | `(dim, in_ch, out_ch, hidden=64, k=16)` | Continuous operator on irregular points using kNN message passing with a learned kernel. x:      (B, N, Cin) coords: (B, N, dim) out:    (B, N, Cout) |
| `PointOperatorBlock` | classe | `(dim, width, k=16)` |  |
| `UNONdPoints` | classe | `(dim, in_channels, out_channels, width=128, levels=3, depth_per_level=2, k=16, pool_ratio…)` | U-shaped operator for irregular meshes (points). This builds a multi-scale pyramid by subsampling points and using kNN operators. Input:  x      (B,N,Cin) coords (B,N,dim) Output: y      (B,N,Cout) |
| `UniversalUNO` | classe | `(dim, in_channels, out_channels, *, mesh_mode='auto', width_grid=64, levels_grid=3, depth…)` | Unified U-NO module: - grid mode: (B,C,*spatial) -> (B,Cout,*spatial) - points mode: (B,N,Cin) + coords(B,N,dim) -> (B,N,Cout) |

#### `pinneapple_neural.architectures.physics_aware`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/physics_aware/__init__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_neural.architectures.physics_aware.base`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/physics_aware/base.py` · 25 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsAwareOutput` | classe | `(y, losses, extras)` |  |
| `PhysicsAwareBase` | classe | `()` |  |

#### `pinneapple_neural.architectures.physics_aware.pann`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/physics_aware/pann.py` · 135 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsAwareNeuralNetwork` | classe | `(backbone, *, physics_loss_fn=None, physics_weight=1.0, physics_components_prefix='physic…)` | Physics-Aware Neural Network: |

#### `pinneapple_neural.architectures.physics_aware.registry`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/physics_aware/registry.py` · 47 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsAwareCatalog` | classe | `(registry)` |  |
| `register_into_global` | função | `() -> None` |  |

#### `pinneapple_neural.architectures.physics_aware.spn`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/physics_aware/spn.py` · 53 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `StructurePreservingNetwork` | classe | `(backbone, *, projector)` | Structure-Preserving Network (MVP): |

#### `pinneapple_neural.architectures.pinns`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/pinns/__init__.py` · 57 linhas · **sem teste direto**</small>

#### `pinneapple_neural.architectures.pinns.base`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/pinns/base.py` · 261 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PINNOutput` | classe | `(y, losses, extras)` | Standard output for all PINN-family models. |
| `PINNBase` | classe | `()` | Base class for all PINN-family models. |

#### `pinneapple_neural.architectures.pinns.bpinn`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/pinns/bpinn.py` · 389 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BayesianLinear` | classe | `(in_features, out_features, *, prior_std=1.0, init_rho=-3.0, bias=True)` | Linear layer whose weights are Gaussian random variables. |
| `BPINNConfig` | classe | `(in_dim, out_dim, hidden, activation, prior_std, init_rho, noise_pde, noise_obs, kl_weigh…)` | Configuration for BayesianPINN. |
| `BayesianPINN` | classe | `(in_dim=2, out_dim=1, hidden=(64, 64, 64), activation='tanh', *, config=None, prior_std=1…)` | Bayesian Physics-Informed Neural Network via mean-field VI. |

#### `pinneapple_neural.architectures.pinns.inverse`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/pinns/inverse.py` · 118 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `InversePINN` | classe | `(in_dim, out_dim, hidden=(128, 128, 128, 128), activation='tanh', inverse_params=None, in…)` | Catalog InversePINN (returns PINNOutput), with a compatibility hook for PINNFactory. |

#### `pinneapple_neural.architectures.pinns.physics_diffusion`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/pinns/physics_diffusion.py` · 787 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NoiseSchedulerBase` | classe | `()` | Protocol for noise schedulers. |
| `VPNoiseScheduler` | classe | `(T=1000, beta_min=0.0001, beta_max=0.02)` | Variance-preserving (VP-DDPM) linear beta schedule. |
| `EDMNoiseScheduler` | classe | `(sigma_min=0.002, sigma_max=80.0, rho=7.0)` | Elucidated Diffusion Model (EDM) schedule. |
| `SinusoidalEmbedding` | classe | `(dim)` | Sinusoidal positional embedding for noise level / timestep. |
| `FiLMLayer` | classe | `(feat_dim, cond_dim)` | Feature-wise Linear Modulation — conditions activations on noise level. |
| `MLPDenoiser` | classe | `(x_dim, t_emb_dim=64, hidden=(256, 256, 256), activation='gelu', predict_x0=False)` | Time-conditioned MLP denoiser for 1-D physics fields. |
| `DSMLoss` | classe | `(denoiser, scheduler, loss_weight_fn=None)` | Denoising score-matching loss. |
| `GuidanceBase` | classe | `()` | Protocol for DPS-style guidance. |
| `PDEResidualGuidance` | classe | `(residual_fn, weight=0.1)` | Physics-constrained guidance via PDE residual. |
| `DataConsistencyGuidance` | classe | `(mask, y_obs, weight=0.5)` | DPS guidance for data assimilation / inverse problems. |
| `ComposedGuidance` | classe | `(*guidances)` | Sum of multiple guidance terms. |
| `PIDiffConfig` | classe | `(x_dim, t_emb_dim, hidden, activation, scheduler_type, T, beta_min, beta_max, sigma_min, …)` |  |
| `PhysicsInformedDiffusion` | classe | `(x_dim=64, hidden=(256, 256, 256), activation='gelu', *, denoiser=None, scheduler=None, c…)` | Physics-Informed Diffusion Model. |

#### `pinneapple_neural.architectures.pinns.pielm`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/pinns/pielm.py` · 141 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PIELM` | classe | `(in_dim, out_dim, hidden_dim=1024, activation='tanh', freeze_random=True)` |  |
| `PIELMFactoryAdapter` | classe | `(pielm, *, inverse_params_names=None, initial_guesses=None, dtype=torch.float32)` |  |

#### `pinneapple_neural.architectures.pinns.pinn_lstm`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/pinns/pinn_lstm.py` · 75 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PINNLSTM` | classe | `(in_dim, out_dim, *, seq_len, hidden_dim=128, num_layers=2, dropout=0.0, inverse_params_n…)` |  |

#### `pinneapple_neural.architectures.pinns.pinnsformer`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/pinns/pinnsformer.py` · 212 linhas · testes: 1</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PINNsFormer` | classe | `(in_dim, out_dim, *, seq_len=32, d_model=128, nhead=4, num_layers=4, dim_feedforward=256,…)` | Transformer-based PINN. |

#### `pinneapple_neural.architectures.pinns.registry`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/pinns/registry.py` · 78 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PINNCatalog` | classe | `(registry)` |  |
| `register_into_global` | função | `() -> None` |  |

#### `pinneapple_neural.architectures.pinns.vanilla`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/pinns/vanilla.py` · 94 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `VanillaPINN` | classe | `(in_dim, out_dim, hidden=(128, 128, 128, 128), activation='tanh', *, inverse_params_names…)` |  |

#### `pinneapple_neural.architectures.pinns.vpinn`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/pinns/vpinn.py` · 119 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `VPINN` | classe | `(in_dim, out_dim, hidden=(128, 128, 128), activation='tanh', *, inverse_params_names=None…)` |  |
| `pinn_factory_adapter` | função | `(loss_fn)` |  |

#### `pinneapple_neural.architectures.pinns.xpinn`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/pinns/xpinn.py` · 126 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SubnetWrapper` | classe | `(in_dim, out_dim, hidden=(128, 128, 128, 128), activation='tanh')` |  |
| `XPINN` | classe | `(n_subdomains, in_dim, out_dim, hidden=(128, 128, 128, 128), activation='tanh', interface…)` |  |

#### `pinneapple_neural.architectures.pinns.xtfc`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/pinns/xtfc.py` · 681 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TFCConstrainedExpression` | classe | `(tfc_obj, dim=1)` | Wraps the `tfc` library to create constrained expressions for 1D or nD problems that exactly satisfy boundary/initial conditions. |
| `XTFCConfig` | classe | `(in_dim, out_dim, rf_dim, activation, freeze_random, rf_kind, rff_sigma, use_bias, init_s…)` |  |
| `XTFC` | classe | `(in_dim, out_dim, rf_dim=2048, activation=None, freeze_random=True, g_fn=None, B_fn=None,…)` | XtFC — eXtreme Theory of Functional Connections PINN. |
| `XTFCFactoryModel` | classe | `(xtfc, inverse_params_names=None, initial_guesses=None, dtype=torch.float32)` | Thin wrapper around XTFC that exposes a standard ``forward(x) -> Tensor`` interface compatible with PINNFactory and the Trainer. |
| `tfc_available` | função | `() -> bool` | Return True if the `tfc` library is installed. |
| `build_xtfc` | função | `(in_dim, out_dim, *, rf_dim=2048, activation=None, freeze_random=True, g_fn=None, B_fn=No…)` | Convenience constructor for XTFC. |

#### `pinneapple_neural.architectures.recurrent`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/recurrent/__init__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_neural.architectures.recurrent.base`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/recurrent/base.py` · 36 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RNNOutput` | classe | `(y, losses, extras)` | Standard output for recurrent models in Pinneapple. |
| `RecurrentModelBase` | classe | `()` | Base class for time-series recurrent models. |

#### `pinneapple_neural.architectures.recurrent.gru`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/recurrent/gru.py` · 199 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GRUModel` | classe | `(in_dim, out_dim, horizon, hidden_dim=128, num_layers=2, dropout=0.0, *, pool='last', tim…)` | GRU forecaster (baseline+). |
| `BiGRUModel` | classe | `(in_dim, out_dim, horizon, hidden_dim=128, num_layers=2, dropout=0.0, *, pool='last', tim…)` | Bidirectional GRU forecaster (baseline+). |

#### `pinneapple_neural.architectures.recurrent.lstm`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/recurrent/lstm.py` · 201 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LSTMModel` | classe | `(in_dim, out_dim, horizon, hidden_dim=128, num_layers=2, dropout=0.0, *, pool='last', tim…)` | LSTM forecaster (baseline+). |
| `BiLSTMModel` | classe | `(in_dim, out_dim, horizon, hidden_dim=128, num_layers=2, dropout=0.0, *, pool='last', tim…)` | Bidirectional LSTM forecaster (baseline+). |

#### `pinneapple_neural.architectures.recurrent.registry`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/recurrent/registry.py` · 57 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RecurrentCatalog` | classe | `(registry)` |  |
| `register_into_global` | função | `() -> None` |  |

#### `pinneapple_neural.architectures.recurrent.seq2seq`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/recurrent/seq2seq.py` · 215 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Seq2SeqRNN` | classe | `(in_dim, out_dim, horizon, hidden_dim=128, num_layers=2, dropout=0.0, cell='gru', future_…)` | Seq2Seq RNN forecaster. |

#### `pinneapple_neural.architectures.register_all`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/register_all.py` · 44 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `register_all` | função | `() -> None` | Populate ModelRegistry with every family. Idempotent — safe to call multiple times. |

#### `pinneapple_neural.architectures.registry`

ModelRegistry and ModelSpec for model registration and lookup.

<small>`pinneapple_neural/architectures/registry.py` · 101 linhas · testes: 64</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ModelSpec` | classe | `(name, family, cls, description, tags, input_kind, supports_physics_loss, expects, predic…)` |  |
| `ModelRegistry` | classe | `(_REGISTRY)` | Central registry for all Pinneapple models. |

#### `pinneapple_neural.architectures.reservoir_computing`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/reservoir_computing/__init__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_neural.architectures.reservoir_computing.base`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/reservoir_computing/base.py` · 48 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RCOutput` | classe | `(y, losses, extras)` | Standard output for reservoir computing models. |
| `RCBase` | classe | `()` |  |

#### `pinneapple_neural.architectures.reservoir_computing.elm`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/reservoir_computing/elm.py` · 88 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ExtremeLearningMachine` | classe | `(in_dim, out_dim, hidden_dim=1024, activation='tanh', l2=1e-06, freeze_random=True)` | Extreme Learning Machine (ELM): - Random hidden layer features: H = phi(X W + b) - Solve output weights with ridge regression (closed-form) |

#### `pinneapple_neural.architectures.reservoir_computing.esn`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/reservoir_computing/esn.py` · 340 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EchoStateNetwork` | classe | `(in_dim, out_dim, reservoir_dim=1024, spectral_radius=0.9, leak=1.0, input_scale=1.0, l2=…)` | Echo State Network (ESN): h_t = (1-leak)*h_{t-1} + leak*tanh(x_t W_in + h_{t-1} W) y_t = phi(h_t, x_t) W_out  (W_out trained by ridge) |

#### `pinneapple_neural.architectures.reservoir_computing.esn_rc`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/reservoir_computing/esn_rc.py` · 113 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ESNRC` | classe | `(in_dim, out_dim, reservoir_dim=1024, *, spectral_radius=0.9, leak=1.0, input_scale=1.0, …)` | ESN-RC (Reservoir Computing) variant: - Uses ESN reservoir but augments readout with nonlinear transforms (e.g., h, h^2, h^3) to approximate richer dynamics while keeping closed-form ridge training. |

#### `pinneapple_neural.architectures.reservoir_computing.hybrid_rbf`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/reservoir_computing/hybrid_rbf.py` · 158 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HybridRBFNetwork` | classe | `(in_dim, out_dim, num_centers=512, sigma=None, l2=1e-06, learn_centers=False, use_linear=…)` | Hybrid RBF Network: y = [Phi(x), x, 1] W where W is solved by ridge regression in a single linear system. |

#### `pinneapple_neural.architectures.reservoir_computing.koopman`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/reservoir_computing/koopman.py` · 155 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `KoopmanOperator` | classe | `(in_dim, *, lift_dim=1024, lift='rff', poly_degree=2, rff_lengthscale=1.0, l2=1e-06, use_…)` | Koopman Operator (EDMD-style MVP): |

#### `pinneapple_neural.architectures.reservoir_computing.rbf`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/reservoir_computing/rbf.py` · 263 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RBFNetwork` | classe | `(in_dim, out_dim, num_centers=512, sigma=None, l2=1e-06, learn_centers=False, *, normaliz…)` | RBF Network (Gaussian features) + ridge-regression readout. |
| `sigma_is_default` | função | `(log_sigma, eps=1e-06) -> bool` |  |

#### `pinneapple_neural.architectures.reservoir_computing.registry`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/reservoir_computing/registry.py` · 70 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ReservoirCatalog` | classe | `(registry)` |  |
| `register_into_global` | função | `() -> None` |  |

#### `pinneapple_neural.architectures.rom`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/rom/__init__.py` · 39 linhas · testes: 25</small>

#### `pinneapple_neural.architectures.rom.base`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/rom/base.py` · 31 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ROMOutput` | classe | `(y, losses, extras)` | Output for Reduced Order Models. |
| `ROMBase` | classe | `()` |  |

#### `pinneapple_neural.architectures.rom.contracts`

ROM study results as the apps' open contracts (``pinneapple_data.contracts``): a ``ModelRef`` per exported reduced model and a ``StudyResult`` for the study, so a ROM built in the ROM Lab is an input of the uncertainty, optimisation, twin or gateway apps.

<small>`pinneapple_neural/architectures/rom/contracts.py` · 120 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `model_ref` | função | `(rm, *, name, uri, payload=None, row=None, split='test', dataset_hash='', dataset_name=''…)` | ``ModelRef`` of a ``ReducedModel``: inputs with their training range, outputs (the fields), the held-out metrics of its study row and the SHA-256 of the file at ``uri`` (``payload``: its bytes). |
| `rom_checks` | função | `(res, tolerance) -> list` | Checks of a study, each with its value: the best ROM against its baseline (nearest snapshot or persistence), against the tolerance, and divergence of the time steppers. |
| `study_result` | função | `(res, *, study_id, title, tolerance, models=(), artifacts=(), limits=(), app='rom', datas…)` | ``StudyResult`` of a ROM study: one table row per (method, rank), the best model, checks with values. |

#### `pinneapple_neural.architectures.rom.deep_uq_rom`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/rom/deep_uq_rom.py` · 61 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DeepUQROM` | classe | `(r, hidden=256, layers=2, dropout=0.0)` | Deep UQ-ROM (MVP scaffold): - A small latent predictor that outputs mean and log-variance. |

#### `pinneapple_neural.architectures.rom.dmd`

Dynamic Mode Decomposition (DMD) — batch-capable, stabilised, with spectral analysis.

<small>`pinneapple_neural/architectures/rom/dmd.py` · 346 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DynamicModeDecomposition` | classe | `(r=64, center='global', l2=0.0)` | Truncated (rank-r) DMD in reduced space. |
| `DMDc` | classe | `(affine=False)` | Dynamic Mode Decomposition with control (Proctor, Brunton & Kutz 2016), NumPy. |

#### `pinneapple_neural.architectures.rom.havok`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/rom/havok.py` · 225 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HAVOK` | classe | `(delays=50, r=64, center=True, svd_eps=1e-12, decode_mode='last')` | HAVOK (MVP-ish, closer to literature): 1) build Hankel (delay embedding) from scalar or low-dim observations 2) center + SVD of Hankel and project to low-rank subspace (HAVOK core) 3) fit linear dynamics in that subspace via DMD (no forced stitching across trajectories) 4) rollout in subspace; … |

#### `pinneapple_neural.architectures.rom.koopman`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/rom/koopman.py` · 269 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `KoopmanOperator` | classe | `(k_dim, structure='full', num_blocks=4)` | Learnable Koopman operator K. |
| `KoopmanAutoencoder` | classe | `(in_dim, k_dim=128, hidden=128, n_layers=3, activation='tanh', k_structure='full', num_bl…)` | Deep Koopman autoencoder: learned lifting + linear dynamics. |

#### `pinneapple_neural.architectures.rom.neural_rom`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/rom/neural_rom.py` · 237 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NeuralROM` | classe | `(in_dim, r=32, hidden_enc=128, n_layers_enc=3, dynamics='mlp', hidden_dyn=128, n_layers_d…)` | Fully neural ROM: nonlinear encoder + latent dynamics + nonlinear decoder. |

#### `pinneapple_neural.architectures.rom.opinf`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/rom/opinf.py` · 387 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OperatorInference` | classe | `(r, *, use_quadratic=True, use_bias=True, l2_linear=1e-06, l2_quad=1e-06, l2_bias=0.0, ce…)` | Operator Inference (OpInf) for latent dynamics. |

#### `pinneapple_neural.architectures.rom.parametric`

Parametric reduced-order models: design parameters -> full field, without running the solver.

<small>`pinneapple_neural/architectures/rom/parametric.py` · 198 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ParametricPOD` | classe | `(r, energy, regressor, kernel, smoothing, normalize, seed, mean_, basis_, sv_)` | POD basis + coefficient regression. ``r``: largest number of modes; ``energy``: keep the modes holding this fraction of the snapshot variance (whichever is smaller). Parameters are scaled to [0, 1] on the training box. ``normalize``: split each snapshot into its amplitude (its L2 norm, regressed as … |
| `latin_hypercube` | função | `(n, bounds, seed=0) -> tuple[np.ndarray, list[str]]` | ``n`` points of a Latin hypercube in the box ``bounds`` (name -> (lo, hi)). Returns (points, names). |

#### `pinneapple_neural.architectures.rom.pod`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/rom/pod.py` · 252 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `POD` | classe | `(r=64, center=True, energy=None, randomised=False, random_seed=0)` | POD via full or randomised truncated SVD for ROM compression. |

#### `pinneapple_neural.architectures.rom.registry`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/rom/registry.py` · 82 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ROMCatalog` | classe | `(registry)` |  |
| `register_into_global` | função | `() -> None` |  |

#### `pinneapple_neural.architectures.rom.rom_hybrid`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/rom/rom_hybrid.py` · 177 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ROMHybrid` | classe | `(field_dim, r=64, l2=1e-06, use_corrector=True, corrector_hidden=64)` | Hybrid ROM combining three components: |

#### `pinneapple_neural.architectures.rom.sindy`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/rom/sindy.py` · 323 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SINDy` | classe | `(r, *, library='poly', poly_degree=2, threshold=0.05, n_iter=10, l2=0.0001, solver='stlsq…)` | SINDy for latent dynamics. |

#### `pinneapple_neural.architectures.rom.steppers`

One-step rollout functions of linear reduced-order models fitted on snapshot trajectories: ``step(u) -> u_next`` in the full space, the shape ``pp.validate(step, {"rollout": ...})`` audits (horizon, drift, spectrum, invariants).

<small>`pinneapple_neural/architectures/rom/steppers.py` · 48 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `pod_dmd_stepper` | função | `(snapshots, r=8)` | POD (rank ``r``, centred) of ``snapshots`` ``(n_t, *shape)`` and exact DMD of its coefficients. Returns ``(step, max_abs_eig)``: ``step(u)`` advances one snapshot interval in the full space; ``max_abs_eig`` is the largest DMD eigenvalue modulus (above 1: the ROM amplifies). |
| `pod_opinf_stepper` | função | `(train, energy) -> tuple[Callable[[np.ndarray], np.ndarray], int]` | POD of ``train`` ``(n_traj, n_t, n_x)`` keeping ``energy`` of the snapshot energy, and linear operator inference (no quadratic term, no bias) on the coefficient trajectories. Returns ``(step, rank)``. |

#### `pinneapple_neural.architectures.rom.study`

Reduced-order model study: which ROM, at which rank, for which error and speed, measured on held-out snapshots.

<small>`pinneapple_neural/architectures/rom/study.py` · 787 linhas · testes: 11 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Split` | classe | `(train, test, how)` |  |
| `ReducedModel` | classe | `(method, mode, rank, arrays, config, fit_s, _cache)` | A fitted ROM: ``predict`` and a portable ``.npz`` form (``to_bytes`` / ``from_bytes``). |
| `ROMStudyResult` | classe | `(mode, split, singular_values, rows, projection, skipped, budget_s, hf_seconds, metric, m…)` |  |
| `default_ranks` | função | `(n_train, n_values, sv=None, max_rank=64) -> list[int]` | 1, 2, 4, 8, ... up to the data's rank, plus the ranks holding 99 %, 99.9 % and 99.99 % of the energy. |
| `split_snapshots` | função | `(snaps, *, how='auto', test_fraction=0.2, seed=0, test_idx=None) -> Split` | Held-out snapshots. Time series: ``forecast`` (the last ``test_fraction`` of the times, default) or ``interleaved`` (every k-th time inside the window). Parametric: ``random`` (seeded). ``test_idx`` overrides. |
| `field_scales` | função | `(snaps, train) -> np.ndarray` | One positive scale per value: the RMS of each field over the training snapshots, so fields in different units (a displacement in mm and a stress in MPa) weigh the same in the POD. |
| `fit_reduced_model` | função | `(snaps, train, method, rank=None, *, budget_s=5.0, seed=0, mode=None, scale='auto', log_p…)` | Fit one ROM on the snapshots ``train`` (indices). ``rank``: number of POD modes (ignored by ``nearest`` and the parametric ``neural``). ``budget_s``: wall-clock budget of the neural surrogate. ``scale``: ``"auto"`` (each field divided by its training RMS when there are several fields), … |
| `rom_study` | função | `(snaps, *, methods=None, ranks=None, split='auto', test_fraction=0.2, test_idx=None, budg…)` | Fit every method at every rank on the training snapshots and score it on the held-out ones (see the module docstring). ``hf_seconds``: wall time of the full-order model per snapshot (default: ``snaps.meta``). ``scale``: with several fields (``"auto"``) each is divided by its training RMS before the … |

#### `pinneapple_neural.architectures.rom.toy_systems`

Synthetic linear dynamical systems for ROM demonstrations and tests (POD compression, DMD rollout).

<small>`pinneapple_neural/architectures/rom/toy_systems.py` · 27 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `make_linear_system` | função | `(D, seed=0) -> torch.Tensor` | A contractive random linear operator ``A`` of shape ``(D, D)``: a Gaussian matrix scaled by ``1 / sqrt(D)`` whose singular values are rescaled to a largest value of 0.95 (so ``x_{k+1} = A x_k`` decays). |
| `generate_sequences` | função | `(B, T, D, A, noise=0.01, seed=123) -> torch.Tensor` | ``B`` trajectories of ``T`` steps of ``x_{k+1} = A x_k + noise * N(0, I)`` from Gaussian initial states in ``D`` dimensions, shape ``(B, T, D)``; deterministic for a given ``seed``. |

#### `pinneapple_neural.architectures.rom.trusted`

Latent surrogate that says when not to trust it: error bounds, envelope check and physics residual per query.

<small>`pinneapple_neural/architectures/rom/trusted.py` · 192 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TrustedLatentSurrogate` | classe | `(rom, spread, members, subsample, neighbours, alpha, alpha_envelope, alpha_residual, resi…)` | POD latent surrogate + per-query error bound, envelope p-value and physics residual. |

#### `pinneapple_neural.architectures.rom.views`

A held-out snapshot on its body: the full-order field, a prediction (a ROM, a surrogate) and their difference on the mesh, with the deformed shape when the snapshots carry a displacement, the boundary-condition and load regions of the dataset, the error hotspots and a short text on where the error …

<small>`pinneapple_neural/architectures/rom/views.py` · 122 linhas · testes: 11</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `zones` | função | `(snaps, points, grow=0.1) -> dict[str, np.ndarray]` | The dataset's named node sets grown by ``grow`` times the body size ("clamp zone", "loaded tip zone") and the remaining nodes ("rest of the body"): the regions errors are compared over. |
| `reconstruction_view` | função | `(snaps, model, index, *, reported=None, grow=0.1, max_faces=None, n_hotspots=3, full_labe…)` | View of snapshot ``index`` of ``snaps`` against ``model`` (a ``ReducedModel``): :func:`prediction_view` of the model's prediction at the snapshot's time or parameters. |
| `prediction_view` | função | `(snaps, index, pred, *, label='Prediction', note='', reported=None, grow=0.1, max_faces=N…)` | View of snapshot ``index`` of ``snaps`` against ``pred``, a predicted snapshot row in the layout of ``snaps`` (a ROM, a surrogate). Named node sets of the dataset become markers (kind ``bc`` / ``load`` / ``region``) and, grown by ``grow`` times the body size, the regions the error is compared over … |

#### `pinneapple_neural.architectures.siren`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/siren.py` · 135 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SineLayer` | classe | `(in_features, out_features, bias=True, is_first=False, omega_0=30.0)` | Single sinusoidal layer with principled weight initialization. |
| `SIREN` | classe | `(in_dim, out_dim, hidden_dim=256, n_layers=5, omega_0=30.0, outermost_linear=True, final_…)` | Sinusoidal Representation Network for physics-informed learning. |

#### `pinneapple_neural.architectures.transformers`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/transformers/__init__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_neural.architectures.transformers.autoformer`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/transformers/autoformer.py` · 367 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SeriesDecomp` | classe | `(kernel_size=25)` | Moving-average decomposition: x = seasonal + trend. |
| `AutoCorrelation` | classe | `(topk_factor=1.0, dropout=0.0)` | Autoformer-style Auto-Correlation: - compute cross-correlation between Q and K via FFT - pick top-k delays - aggregate V by shifting (time-delay aggregation) |
| `AutoCorrelationLayer` | classe | `(d_model, nhead, topk_factor=1.0, dropout=0.0)` | Multi-head wrapper with projections like attention. |
| `EncoderLayer` | classe | `(d_model, nhead, d_ff, kernel_size, dropout, topk_factor)` |  |
| `DecoderLayer` | classe | `(d_model, nhead, d_ff, kernel_size, dropout, topk_factor)` |  |
| `Autoformer` | classe | `(in_dim, out_dim, horizon, d_model=128, nhead=4, num_layers=2, kernel_size=25, dropout=0.…)` | Autoformer (minimal faithful): - Decompose input into seasonal/trend - Encoder: AutoCorrelation + decomposition - Decoder: AutoCorrelation (self + cross) + decomposition, accumulates trend - Output: y = seasonal_out + trend_out (projected to out_dim) |

#### `pinneapple_neural.architectures.transformers.base`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/transformers/base.py` · 48 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TSOutput` | classe | `(y, losses, extras)` | Standard output for time-series transformers in Pinneapple. |
| `TimeSeriesModelBase` | classe | `()` | Base class for sequence forecasting models. |

#### `pinneapple_neural.architectures.transformers.fedformer`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/transformers/fedformer.py` · 89 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FEDformer` | classe | `(in_dim, out_dim, horizon, d_model=128, topk_freq=32)` | Fast baseline inspired by FEDformer ideas: - learn a 1D projection for spectral features (cheap multivariate proxy) - select top-k frequency bins (batch-consistent) - mix last-token embedding + spectral features with a small MLP - horizon embedding to avoid constant repeated outputs |

#### `pinneapple_neural.architectures.transformers.informer`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/transformers/informer.py` · 132 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SinusoidalPositionalEncoding` | classe | `(d_model, max_len=10000)` |  |
| `Informer` | classe | `(in_dim, out_dim, horizon, d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropo…)` | Efficient encoder-only baseline: - Transformer encoder (standard attention) - Optional distilling (Conv1d stride=2) on early layers only - Attention pooling to get sequence context - Forecast head uses context + (optional) known future covariates per horizon |

#### `pinneapple_neural.architectures.transformers.registry`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/transformers/registry.py` · 61 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TransformerCatalog` | classe | `(registry)` |  |
| `register_into_global` | função | `() -> None` |  |

#### `pinneapple_neural.architectures.transformers.tft`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/transformers/tft.py` · 254 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GRN` | classe | `(d, hidden, dropout=0.0)` | Gated Residual Network (lite): y = LayerNorm(x + sigmoid(Wg x) * Drop( W2 gelu(W1 x) )) |
| `GLU` | classe | `(d)` | Gated Linear Unit used for skip gating. |
| `GroupVSN` | classe | `(d_model, num_groups, hidden, dropout=0.0)` | Variable selection network (lite) over groups, not per-variable. |
| `TemporalFusionTransformer` | classe | `(in_dim, out_dim, horizon, d_model=128, nhead=4, num_enc_layers=2, num_dec_layers=2, drop…)` | TFT-lite (strong, honest): - Input projection - Grouped variable selection (past groups + optional future group) - Causal TransformerEncoder over past - TransformerDecoder with learnable future queries (multi-step) attending to past memory - GRN/GLU gating around blocks for stability and TFT-like … |

#### `pinneapple_neural.architectures.transformers.timesnet`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/transformers/timesnet.py` · 199 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TimesNet` | classe | `(in_dim, out_dim, horizon, d_model=128, num_blocks=2, top_k_periods=3, head_window=64, dr…)` | TimesNet-like forecasting: 1) FFT period discovery (Top-K) 2) Reshape into 2D maps per period 3) 2D inception blocks (multi-scale) 4) Weighted aggregation across periods 5) Flatten head for horizon prediction |

#### `pinneapple_neural.architectures.transformers.transformer`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/architectures/transformers/transformer.py` · 113 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `VanillaTransformer` | classe | `(in_dim, out_dim, d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1, p…)` | Causal Transformer encoder for autoregressive forecasting. |

#### `pinneapple_neural.callbacks`

Physics checks during training in PyTorch Lightning, PhysicsNeMo and Hugging Face (#704).

<small>`pinneapple_neural/callbacks.py` · 274 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsWarning` | classe | `()` | A physics metric worsened for several consecutive checks while the training loss improved. |
| `PhysicsMonitor` | classe | `(problem, data=None, *, reference=None, query=None, level='quick', every=100, patience=3,…)` | The checks of ``pp.validate(model, problem, data, reference, query, level)`` as training metrics. |
| `PhysicsNeMoHook` | classe | `(monitor)` | For a PhysicsNeMo training loop: ``hook(model, step, loss, log=...)`` after each optimiser step. |

#### `pinneapple_neural.lpm`

Large Physics Model (LPM) building blocks: one model for a whole design space.

<small>`pinneapple_neural/lpm.py` · 212 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FourierEncoding` | classe | `(n_freq=6, include_input=True)` | x -> [x, sin(2^k pi x), cos(2^k pi x)] for k = 0..n_freq-1 (per coordinate). |
| `GeometryEncoder` | classe | `(d, width=64, code=32)` | DeepSets: per-point MLP on (position, normal) then mean+max pooling -> geometry code. |
| `Design` | classe | `(points, normals, query, params, target)` | One simulation of the design space: its geometry, operating point and a target field. |
| `LargePhysicsModel` | classe | `(dim, n_fields, n_params=0, radii=(0.1, 0.4, 1.2), n_freq=6, width=128, code=32, n_heads=…)` |  |
| `multiscale_neighbourhood` | função | `(query, points, normals, radii)` | Per query point and radius: mean offset to neighbours, mean neighbour normal, occupancy. |
| `fit` | função | `(model, designs, *, epochs=2000, lr=0.002, freeze_geometry=False, seed=0) -> list[float]` | Train (or fine-tune with ``freeze_geometry=True``) on a list of designs. |

#### `pinneapple_neural.physicsnemo_bridge`

pinneapple_neural.physicsnemo_bridge -- PhysicsNeMo <-> PINNeAPPle integration glue.

<small>`pinneapple_neural/physicsnemo_bridge/__init__.py` · 86 linhas · testes: 19</small>

#### `pinneapple_neural.physicsnemo_bridge.active_learning_bridge`

PINNeAPPle active learning -> PhysicsNeMo retrain bridge.

<small>`pinneapple_neural/physicsnemo_bridge/active_learning_bridge.py` · 226 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RetrainConfig` | classe | `(n_rounds, n_epochs_per_round, batch_size, lr, selection_mode, device, verbose)` | Configuration for the "retrain" half of the AL <-> retrain loop. |
| `ALRoundResult` | classe | `(round_index, n_points_added, n_points_total, mean_train_loss, train_losses)` | Bookkeeping for a single AL-select + retrain round. |
| `ALRetrainResult` | classe | `(rounds, x_collocation, final_loss)` | Full result of :func:`active_learning_retrain_loop`. |
| `active_learning_retrain_loop` | função | `(model, loss_fn, residual_fn, bounds, *, x_initial=None, al_config=None, retrain_config=N…)` | Run the "AL selects -> retrains -> AL selects -> ..." loop (Padrao 2). |

#### `pinneapple_neural.physicsnemo_bridge.adapters`

PhysicsNeMo -> PINNeAPPle model adapters (UQ + Digital Twin).

<small>`pinneapple_neural/physicsnemo_bridge/adapters.py` · 254 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsNemoModelAdapter` | classe | `(model, field_names) -> None` | Wrap a trained model so its ``forward()`` output is one plain tensor. |
| `is_physicsnemo_available` | função | `() -> bool` | Return ``True`` if ``physicsnemo`` (or the legacy ``modulus`` name) can be imported. |
| `require_physicsnemo_model` | função | `(model) -> Any` | Validate that ``model`` is a physicsnemo-compatible trained model. |
| `wrap_for_uq` | função | `(model, field_names=None, *, mc_dropout_config=None) -> MCDropoutWrapper` | Wrap a trained model for MC-Dropout uncertainty quantification. |
| `build_digital_twin_for_model` | função | `(model, field_names, coord_names=None, *, config=None) -> DigitalTwin` | Wrap a trained model into a :class:`DigitalTwin`. |

#### `pinneapple_neural.physicsnemo_bridge.field_operators`

PhysicsNeMo field operators on a structured grid -> PINNeAPPle predictors (pattern 1: PhysicsNeMo trains, PINNeAPPle checks).

<small>`pinneapple_neural/physicsnemo_bridge/field_operators.py` · 241 linhas · testes: 3</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GridFieldPredictor` | classe | `(module, config, name)` | A trained PhysicsNeMo model on an ``n x n`` grid as a NumPy predictor ``predict(a) -> u``. |
| `TrainingResult` | classe | `(run, epochs, final_loss, history)` | What :func:`train_grid_operator` spent and reached. |
| `grid_graph` | função | `(n) -> tuple[np.ndarray, np.ndarray]` | Graph of an ``n x n`` cell-centred grid: ``edge_index (2, E)`` (both directions, node ``i * n + j``) and ``edge_attr (E, 3)`` = ``(dx, dy, \|d\|)`` of each edge in cell units. |
| `physicsnemo_fno` | função | `(n, *, latent_channels=32, num_fno_layers=4, num_fno_modes=12, decoder_layers=1, decoder_…)` | A 2-D ``physicsnemo.models.fno.FNO`` (coefficient -> solution) wrapped as a :class:`GridFieldPredictor`. |
| `physicsnemo_meshgraphnet` | função | `(n, *, processor_size=15, hidden_dim=64, aggregation='sum', seed=0) -> GridFieldPredictor` | A ``physicsnemo.models.meshgraphnet.MeshGraphNet`` on the graph of the ``n x n`` grid (:func:`grid_graph`). |
| `train_grid_operator` | função | `(pred, a, u, budget, *, batch_size=8, lr=0.001, weight_decay=1e-05, seed=0, evaluate=None…)` | Train ``pred`` on ``(a, u)`` pairs until ``budget`` (a ``ComputeBudget``) is spent. |

#### `pinneapple_neural.predictor`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/predictor/__init__.py` · 69 linhas · **sem teste direto**</small>

#### `pinneapple_neural.predictor.infer`

Inference utilities for trained PINN and neural network models.

<small>`pinneapple_neural/predictor/infer.py` · 190 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `InferenceResult` | classe | `(coords, fields, grid_shape, model_id, metadata)` | Container for model inference output. |
| `batched_inference` | função | `(model, x, *, batch_size=4096, device=None, use_amp=False, amp_dtype=torch.float16) -> to…)` | Run model inference over a large tensor in batches to avoid OOM. |
| `infer` | função | `(model, x, *, device='cpu', field_names=None) -> np.ndarray` | Run a forward pass on numpy input. |
| `infer_on_grid_1d` | função | `(model, x_range, t_range, *, nx=200, nt=200, device='cpu', field_names=None, coord_names=…)` | Inference on a 1D+time rectangular grid. |
| `infer_on_grid_2d` | função | `(model, x_range, y_range, *, nx=100, ny=100, device='cpu', field_names=None, coord_names=…)` | Inference on a 2D rectangular grid. |

#### `pinneapple_neural.predictor.postprocess`

Production-quality 3D field visualization: streamlines, isosurfaces, volume rendering.

<small>`pinneapple_neural/predictor/postprocess.py` · 762 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FlowVisualizer` | classe | `(model, coord_ranges, device='cpu') -> None` | High-level flow visualization class. |
| `compute_streamlines` | função | `(u_field, v_field, x_grid, y_grid, seed_points=None, n_seeds=20, max_length=1.0, dt=0.01)…)` | Compute 2D streamlines via RK4 integration. |
| `compute_isosurface` | função | `(field_3d, x, y, z, level=0.0) -> Tuple[np.ndarray, np.ndarray]` | Extract isosurface from 3D scalar field using marching cubes. |
| `plot_streamlines_2d_from_model` | função | `(model, x_range=(0.0, 1.0), y_range=(0.0, 1.0), n_grid=50, n_seeds=30, figsize=(10, 6), t…)` | Plot 2D streamlines of velocity field predicted by a PINN model. |
| `plot_isosurface_3d` | função | `(model, x_range=(0.0, 1.0), y_range=(0.0, 1.0), z_range=(0.0, 1.0), n_grid=30, level=0.0,…)` | Plot 3D isosurface of a scalar field predicted by a PINN model. |
| `plot_volume_slice` | função | `(model, axis='z', slice_val=0.5, x_range=(0.0, 1.0), y_range=(0.0, 1.0), z_range=(0.0, 1.…)` | Plot a 2D slice of a 3D field predicted by a PINN model. |

#### `pinneapple_neural.predictor.visualize`

Visualization utilities for PINN inference results.

<small>`pinneapple_neural/predictor/visualize.py` · 1.103 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_field_1d` | função | `(result, field_name, *, ax=None, title=None, cmap='RdBu_r', save_path=None, show=False) -…)` | 2D colormesh plot of a 1D+time field (x vs t). |
| `plot_field_2d` | função | `(result, field_name, *, ax=None, title=None, cmap='viridis', save_path=None, show=False) …)` | Colormesh plot of a 2D field (x vs y). |
| `plot_error_map_1d` | função | `(result, reference, field_name, *, ax=None, title=None, save_path=None, show=False) -> Any` | Plot absolute error \|prediction - reference\| on a 1D+time grid. |
| `plot_error_map_2d` | função | `(result, reference, field_name, *, ax=None, title=None, save_path=None, show=False) -> Any` | Plot absolute error for a 2D field. |
| `plot_loss_curve` | função | `(history, *, keys=None, ax=None, title='Training loss', log_scale=True, save_path=None, s…)` | Plot training/validation loss curves from a history list. |
| `plot_model_comparison_1d` | função | `(results, field_name, *, reference=None, title=None, cmap='RdBu_r', save_path=None, show=…)` | Side-by-side comparison of multiple models on a 1D+time field. |
| `plot_model_comparison_2d` | função | `(results, field_name, *, reference=None, title=None, cmap='viridis', save_path=None, show…)` | Side-by-side comparison for 2D fields. |
| `render_visualizations` | função | `(viz_cfgs, *, model_results, problem_spec=None, reference_result=None, out_dir='.') -> Di…)` | Dispatch visualization from YAML-driven config list. |
| `plot_velocity_slice` | função | `(u, x, y, z, *, slice_axis='z', slice_idx=None, component=0, title=None, cmap='RdBu_r', a…)` | 2D colormesh of a velocity slice through a 3D field. |
| `plot_velocity_magnitude_slice` | função | `(u, x, y, z, *, slice_axis='z', slice_idx=None, title=None, cmap='viridis', ax=None, save…)` | Plot \|u\| = sqrt(u^2 + v^2 + w^2) on a 2D slice. |
| `plot_streamlines_2d` | função | `(u, v, x, y, *, density=1.5, title=None, background_field=None, cmap='viridis', ax=None, …)` | Streamline plot for 2D velocity field. |
| `plot_centerline_velocity` | função | `(u_centerline, coord, *, reference_coord=None, reference_u=None, reference_label='Ghia et…)` | 1D centerline velocity profile with optional literature reference overlay. |
| `plot_vorticity_slice` | função | `(u, v, x, y, z, *, slice_axis='z', slice_idx=None, title=None, cmap='RdBu_r', ax=None, sa…)` | Plot z-component of vorticity (omega_z = dv/dx - du/dy) on a 2D slice. |
| `plot_internal_flow_summary` | função | `(solver_output, *, time_step=-1, slice_axis='z', title='Internal Flow Summary', save_path…)` | Multi-panel summary figure for 3D internal flow (LDC or channel). |
| `plot_design_opt_convergence` | função | `(history_objectives, *, title=…, xlabel='Iteration', ylabel='Objective', log_scale=False,…)` | Plot convergence curve of objective function over optimization iterations. |
| `plot_pareto_front_2d` | função | `(objectives, pareto_mask, *, xlabel='Objective 1', ylabel='Objective 2', title='Pareto Fr…)` | Scatter plot with Pareto-optimal points highlighted. |

#### `pinneapple_neural.trainer`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/trainer/__init__.py` · 191 linhas · testes: 1</small>

#### `pinneapple_neural.trainer.adaptive_sweep`

Automatic (adaptive) hyperparameter search.

<small>`pinneapple_neural/trainer/adaptive_sweep.py` · 178 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AdaptiveSweepConfig` | classe | `(param_space, n_trials, direction, seed, pruning)` |  |
| `run_adaptive_sweep` | função | `(trial_fn, cfg) -> List[Dict[str, Any]]` | Run an adaptive (Bayesian/TPE via Optuna, or random-search fallback) hyperparameter search. |

#### `pinneapple_neural.trainer.advisor`

_Módulo sem docstring e sem texto._

<small>`pinneapple_neural/trainer/advisor.py` · 378 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Suggestion` | classe | `(code, priority, description, config_patch, rationale)` | A single actionable improvement recommendation. |
| `DiagnosticReport` | classe | `(model_name, signals, suggestions, metrics_summary, diagnosis_text)` | Complete diagnostic output for one model run. |
| `TrainingAdvisor` | classe | `(plateau_rel_std_thresh=0.005, divergence_rel_thresh=-0.05, slow_start_thresh=0.05, high_…)` | Inspects training histories and final metrics to produce ranked Suggestion objects for improving the next training run. |

#### `pinneapple_neural.trainer.audit`

Auditability: seed, deterministic mode, RunLogger, env fingerprint.

<small>`pinneapple_neural/trainer/audit.py` · 76 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RunLogger` | classe | `(out_dir, run_name, flush_every)` | JSONL logger for industry-grade auditability. |
| `set_seed` | função | `(seed) -> None` |  |
| `set_deterministic` | função | `(deterministic=True) -> None` |  |
| `env_fingerprint` | função | `() -> Dict[str, Any]` |  |

#### `pinneapple_neural.trainer.callbacks`

Training callbacks: EarlyStopping and ModelCheckpoint.

<small>`pinneapple_neural/trainer/callbacks.py` · 67 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EarlyStopping` | classe | `(monitor, patience, min_delta, mode, best, bad_epochs, stop)` |  |
| `ModelCheckpoint` | classe | `(path, monitor, mode, best)` |  |

#### `pinneapple_neural.trainer.causal`

Causal training for temporal PDEs.

<small>`pinneapple_neural/trainer/causal.py` · 408 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CausalWeightScheduler` | classe | `(n_time_chunks=100, epsilon=1.0, update_every=100)` | Compute per-point causal weights for temporal PDE training. |
| `CausalPINNTrainer` | classe | `(model, epsilon=1.0, n_time_chunks=50, update_every=100)` | PINN trainer with causal weighting for temporal PDEs. |

#### `pinneapple_neural.trainer.checkpoint`

Checkpoint save/load for model, optimizer, config, and normalizers.

<small>`pinneapple_neural/trainer/checkpoint.py` · 40 linhas · testes: 3</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Checkpoint` | classe | `(model_state, optim_state, cfg, meta, normalizers)` |  |
| `save_checkpoint` | função | `(path, ckpt) -> None` |  |
| `load_checkpoint` | função | `(path) -> Checkpoint` |  |

#### `pinneapple_neural.trainer.collocation`

Adaptive collocation point sampling for PINN training.

<small>`pinneapple_neural/trainer/collocation.py` · 413 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AdaptiveCollocationConfig` | classe | `(n_active, n_pool, resample_every, residual_frac, residual_batch, replace)` | Configuration for ``AdaptiveCollocationSampler``. |
| `AdaptiveCollocationSampler` | classe | `(pool, config=None, residual_fn=None) -> None` | Residual-based Adaptive Refinement (RAR) for PINN collocation points. |
| `estimate_channel_scales` | função | `(Y, quantile=0.95, min_scale=1e-06, channel_names=None) -> List[float]` | Estimate per-channel output scales as the q-th quantile of \|Y_i\|. |
| `filter_nan_rows` | função | `(*tensors, warn=True) -> Tuple[torch.Tensor, ...]` | Remove rows where any tensor has NaN or Inf. |

#### `pinneapple_neural.trainer.datamodule`

DataModule, loaders, and item adapters for training.

<small>`pinneapple_neural/trainer/datamodule.py` · 200 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ItemAdapter` | classe | `()` | Adapter that standardizes how we convert dataset items into dict batches. |
| `FnAdapter` | classe | `(x_fn, y_fn, meta_fn=None)` | Convenience adapter for Pinneapple PhysicalSample-like objects (or any object). |
| `AdaptedSequenceDataset` | classe | `(samples, adapter)` | Wrap a Python sequence and apply an adapter in __getitem__. Useful for: list[PhysicalSample] or list[dict] etc. |
| `DataModule` | classe | `(dataset, split, group_ids, batch_size, num_workers, pin_memory, persistent_workers, drop…)` | Builds train/val/test DataLoaders with consistent split + preprocess. |

#### `pinneapple_neural.trainer.distributed`

Distributed Data Parallel (DDP) training for PINNeAPPle.

<small>`pinneapple_neural/trainer/distributed.py` · 323 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DDPTrainerConfig` | classe | `(backend, world_size, master_addr, master_port, gradient_sync_every, find_unused_paramete…)` | Configuration for DDP training. |
| `DDPPINNTrainer` | classe | `(model, config=None)` | Multi-GPU PINN trainer using PyTorch DDP. |
| `is_distributed` | função | `() -> bool` | Return True when running inside an active distributed process group. |
| `get_rank` | função | `() -> int` | Return the rank of the current process (0 when not distributed). |
| `get_world_size` | função | `() -> int` | Return the total number of processes (1 when not distributed). |

#### `pinneapple_neural.trainer.graybox`

Gray-box PINN training (AI-Aristotle style): a main network learns the state variables while one or more small auxiliary networks learn an unknown functional term in an otherwise-known governing equation.

<small>`pinneapple_neural/trainer/graybox.py` · 280 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GrayBoxNet` | classe | `(n_in=1, n_out=1, width=32, depth=3)` | Small network approximating an unknown functional term in a PDE/ODE. Kept shallow/narrow by design: if the term is later distilled into a closed-form expression (e.g. via symbolic regression), a simple network is much easier to distill accurately than a large one. |
| `GrayBoxConfig` | classe | `(epochs_supervised, epochs_joint, lr_main, lr_graybox, lr_params, w_data, w_pde, pde_warm…)` |  |
| `train_graybox` | função | `(main_model, graybox_nets, known_residual_fn, obs_coords, obs_values, collocation_sampler…)` | Two-phase gray-box training: Phase 1 fits `main_model` to (obs_coords, obs_values) alone; Phase 2 jointly trains `main_model`, the gray-box networks, and any unknown scalar parameters against a weighted sum of the data loss and `known_residual_fn`'s PDE residual, with the PDE weight auto-balanced … |
| `distill_to_symbolic` | função | `(graybox_net, coord_bounds, n_samples=2000, var_names=None, n_iterations=40) -> Dict[str,…)` | Sample a trained gray-box network on its input domain and run PySR symbolic regression to recover a closed-form analytical expression for the unknown term it learned. Requires the optional `pysr` package. |

#### `pinneapple_neural.trainer.heteroscedastic`

Heteroscedastic (input-dependent variance) uncertainty quantification.

<small>`pinneapple_neural/trainer/heteroscedastic.py` · 136 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HeteroscedasticWrapper` | classe | `(base_network, n_out_physical)` | Wraps a base network (output dim = 2 * n_out_physical) into a model exposing `.forward_uq(x) -> (mean, log_variance)`, each of shape (N, n_out_physical). `.forward(x)` returns the mean only, for drop-in compatibility with code expecting a single-prediction model. |
| `init_logvar_bias` | função | `(output_layer, n_out_physical, bias_value=-4.6) -> None` | Bias-initialize a network's final linear layer's log-variance output columns (indices [n_out_physical:]) toward a small initial predicted uncertainty (log(sigma^2) ~= bias_value, i.e. sigma ~= exp(bias_value/2) ~= 0.1 at the default). Call once, right after constructing the base network and before … |
| `heteroscedastic_nll_loss` | função | `(pred_mean, pred_logvar, target) -> torch.Tensor` | Gaussian negative log-likelihood: L = 0.5*(log(sigma^2) + (y-mu)^2/sigma^2), with log-variance clamped to [-10, 10] for numerical stability. |
| `compute_aleatoric_epistemic_uq` | função | `(model, coords, n_mc=50, forward_kwargs=None) -> Dict[str, torch.Tensor]` | Decompose predictive uncertainty via the law of total variance: |

#### `pinneapple_neural.trainer.hpc`

Advanced HPC utilities for large-scale PINN training.

<small>`pinneapple_neural/trainer/hpc.py` · 819 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FSDPConfig` | classe | `(sharding_strategy, mixed_precision, min_num_params, activation_checkpointing, cpu_offloa…)` | Configuration for FSDP wrapping. |
| `CUDAGraphModule` | classe | `(module, warmup_steps=3, device='cuda') -> None` | Wrap a module to use CUDA graph replay for static-shape inference. |
| `TorchRunConfig` | classe | `(n_nodes, n_procs_per_node, master_addr, master_port, rdzv_backend, max_restarts)` | Configuration for a ``torchrun`` launch. |
| `SLURMConfig` | classe | `(job_name, n_nodes, n_gpus_per_node, n_cpus_per_task, time_limit, partition, account, mem…)` | SLURM batch script configuration. |
| `ProfilerConfig` | classe | `(trace_dir, record_shapes, profile_memory, with_flops, with_stack, schedule_wait, schedul…)` | Configuration for PINNeAPPleProfiler. |
| `PINNeAPPleProfiler` | classe | `(config=None) -> None` | Context-manager profiler that wraps ``torch.profiler.profile``. |
| `AutoBatchSizeFinder` | classe | `(trial_fn, lo=1, hi=8192, verbose=True) -> None` | Find the maximum batch size that fits in GPU memory via binary search. |
| `wrap_fsdp` | função | `(model, cfg=None, *, process_group=None) -> nn.Module` | Wrap a model with Fully Sharded Data Parallel. |
| `wrap_zero_optimizer` | função | `(optimizer_class, params, **optimizer_kwargs) -> torch.optim.Optimizer` | Wrap an optimizer with ZeroRedundancyOptimizer (ZeRO-1 sharding). |
| `register_powersgd_hook` | função | `(model, *, matrix_approximation_rank=1, start_powersgd_iter=10, use_error_feedback=True) …)` | Register a PowerSGD gradient compression hook on a DDP model. |
| `register_topk_hook` | função | `(model, *, compression_ratio=0.1) -> None` | Register a Top-K gradient sparsification hook on a DDP model. |
| `build_torchrun_cmd` | função | `(script, cfg, *, script_args='') -> str` | Generate a ``torchrun`` launch command string. |
| `build_slurm_script` | função | `(torchrun_cmd, slurm, *, output_dir='slurm_logs') -> str` | Generate a complete SLURM batch script as a string. |

#### `pinneapple_neural.trainer.inference`

Inference utilities: load_checkpoint and predict.

<small>`pinneapple_neural/trainer/inference.py` · 22 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `load_checkpoint` | função | `(model, path, map_location='cpu') -> Dict[str, Any]` |  |
| `predict` | função | `(model, x, device=None) -> torch.Tensor` |  |

#### `pinneapple_neural.trainer.inverse_pinn`

Inverse PINN: jointly optimize network weights and unknown scalar physical parameters from sparse sensor observations plus a PDE/ODE residual.

<small>`pinneapple_neural/trainer/inverse_pinn.py` · 218 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `InverseParam` | classe | `(initial_value, log_space=False, name='param')` | Wraps a scalar as a learnable parameter. Supports a log-space transform for positivity-constrained parameters (the raw trainable value is log(param); `.value` exponentiates it, so it is always > 0 and unconstrained-Adam-friendly even near 0). |
| `InversePINN` | classe | `(model, inverse_params, residual_fn, sensor_points, sensor_values, domain_bounds, n_col=2…)` | Jointly optimizes network weights (theta) and unknown physical parameters (lambda) from sparse sensor observations plus a caller- supplied PDE/ODE residual (see module docstring). |

#### `pinneapple_neural.trainer.lbfgs_finetune`

Multi-round L-BFGS fine-tuning: resample-per-round quasi-Newton polish.

<small>`pinneapple_neural/trainer/lbfgs_finetune.py` · 112 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LBFGSFinetuneResult` | classe | `(round_losses, round_iters)` | Per-round bookkeeping from :func:`multi_round_lbfgs`. |
| `multi_round_lbfgs` | função | `(params, closure_builder, n_rounds, n_steps_per_round, lr=1.0, history_size=50, line_sear…)` | Run ``n_rounds`` independent :class:`torch.optim.LBFGS` phases, each ``max_iter=n_steps_per_round``, each against a freshly built closure (so a new mini-batch, if the caller's ``closure_builder`` resamples one) -- rather than one L-BFGS phase for the whole budget. |

#### `pinneapple_neural.trainer.loss_balancer`

Advanced automatic loss weight optimisation for PINNs.

<small>`pinneapple_neural/trainer/loss_balancer.py` · 1.229 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ReLoBRaLo` | classe | `(loss_names, *, tau=0.1, alpha=0.999, update_every=1, clip_min=0.001, clip_max=1000.0, in…)` | Relative Loss Balancing via Residuals and Losses. |
| `SoftAdapt` | classe | `(loss_names, *, tau=0.1, beta=0.0, normalized=True, update_every=1, clip_min=0.001, clip_…)` | SoftAdapt — Adaptive loss weighting via rates of change. |
| `AugmentedLagrangian` | classe | `(loss_names, *, pde_key='pde', rho=0.1, lambda_init=1.0, lambda_min=0.0, lambda_max=10000…)` | Augmented Lagrangian method for PINN loss balancing. |
| `InverseDirichlet` | classe | `(model, loss_names, *, shared_layer=None, update_every=50, alpha=0.9, clip_min=0.001, cli…)` | Inverse Dirichlet (gradient statistics) weighting. |
| `LRAnnealing` | classe | `(model, loss_names, *, reference='pde', update_every=100, alpha=0.1, clip_min=0.001, clip…)` | Learning-rate annealing for PINN loss weights (Algorithm 1 of Wang, Teng & Perdikaris, 2021 — "Understanding and mitigating gradient flow pathologies in physics-informed neural networks", SIAM J. Sci. Comput. 43(5):A3055– A3081, doi:10.1137/20M1318043). Not the NTK paper cited for … |
| `PCGrad` | classe | `(model, loss_names, *, update_every=1, weights=None) -> None` | PCGrad — Projecting Conflicting Gradients. |
| `AutoBalancerConfig` | classe | `(method, check_every, health_window, plateau_rel_thresh, divergence_thresh, max_switches,…)` | Configuration for the AutoBalancer meta-scheduler. |
| `AutoBalancer` | classe | `(model, loss_names, *, config=None, pde_key='pde') -> None` | Auto-balancing meta-scheduler for PINN loss weights. |
| `JointAdaptiveWeights` | classe | `(loss_names, *, initial_weights=None, sp_beta=1.0, max_raw_init=20.0, scale=None) -> None` | Jointly-trained adaptive loss weights via softplus parameterisation. |
| `AutoPDEWeightCalibrator` | classe | `(target_ratio=0.01, *, once=True, clip_min=1e-20, clip_max=1000000.0) -> None` | Automatically calibrates the PDE loss weight relative to BC/data losses. |

#### `pinneapple_neural.trainer.losses`

Loss functions for supervised and physics-aware training.

<small>`pinneapple_neural/trainer/losses.py` · 148 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SupervisedLoss` | classe | `(kind)` |  |
| `PhysicsLossHook` | classe | `(physics_loss_fn)` |  |
| `CombinedLoss` | classe | `(supervised, physics, w_supervised, w_physics)` |  |
| `build_loss` | função | `(*, problem_spec, model_capabilities=None, weights=None, supervised_kind='mse', physics_l…)` | Helper to build a CombinedLoss for Arena. |

#### `pinneapple_neural.trainer.metrics`

Metrics for evaluation (MSE, MAE, RMSE, R2, MetricBundle).

<small>`pinneapple_neural/trainer/metrics.py` · 109 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Metric` | classe | `(name)` |  |
| `MSE` | classe | `(name)` |  |
| `MAE` | classe | `(name)` |  |
| `RMSE` | classe | `(name)` |  |
| `R2` | classe | `(name, eps)` |  |
| `RelL2` | classe | `(name, eps)` |  |
| `MaxError` | classe | `(name)` |  |
| `Metrics` | classe | `()` |  |
| `MetricBundle` | classe | `(metrics)` |  |
| `RegressionMetrics` | classe | `(eps)` |  |
| `default_metrics` | função | `() -> List[Metric]` |  |
| `regression_metrics_bundle` | função | `() -> Metrics` |  |

#### `pinneapple_neural.trainer.metrics_cfg`

Build Metrics objects from YAML-driven configuration lists.

<small>`pinneapple_neural/trainer/metrics_cfg.py` · 81 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_metrics_from_cfg` | função | `(cfg) -> Metrics` | Build a MetricBundle from a list of metric names or config dicts. |

#### `pinneapple_neural.trainer.monotonic`

Monotonicity constraints for regression surrogates: a hinge penalty on input-output derivatives.

<small>`pinneapple_neural/trainer/monotonic.py` · 55 linhas · testes: 2</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `monotonicity_penalty` | função | `(model, z, cols, signs, *, scale=None, margin=0.0, output=0, create_graph=True) -> torch.…)` | Mean squared hinge on ``signs[j] * d out / d z[:, cols[j]] * scale[j] + margin`` (zero when every sampled derivative has the required sign with at least ``margin`` to spare). |
| `monotone_violation_share` | função | `(model, z, cols, signs, output=0) -> list[float]` | Share of the rows of ``z`` where ``d out / d z[:, col]`` has the wrong sign, per constrained column. |

#### `pinneapple_neural.trainer.multiadam`

MultiAdam: Adam with separate moment statistics per loss term.

<small>`pinneapple_neural/trainer/multiadam.py` · 103 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MultiAdam` | classe | `(params, lr=0.001, betas=(0.99, 0.99), eps=1e-08)` | Adam with one ``(m, v)`` pair per loss term (Yao et al., ICML 2023). |

#### `pinneapple_neural.trainer.multistart`

Multi-restart PINN training with NaN watchdog and best-model selection.

<small>`pinneapple_neural/trainer/multistart.py` · 363 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MultiRestartConfig` | classe | `(n_restarts, seed_base, nan_patience, max_failed, verbose, keep_all)` | Configuration for ``MultiRestartTrainer``. |
| `RestartResult` | classe | `(model, history, score, restart_idx, seed, nan_detected)` | Outcome of a single training restart. |
| `MultiRestartResult` | classe | `(best, all_results, n_valid, n_nan)` | Aggregated output of ``MultiRestartTrainer.fit()``. |
| `MultiRestartTrainer` | classe | `(model_factory, train_fn, *, config=None, score_fn=None, on_restart_end=None) -> None` | Runs a training function N times with different seeds, keeps the best. |

#### `pinneapple_neural.trainer.noether_trainer`

Noether surrogate trainer for PINNeAPPle.

<small>`pinneapple_neural/trainer/noether_trainer.py` · 344 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NoetherTrainConfig` | classe | `(epochs, batch_size, lr, weight_decay, field_weights, loss_fn, warmup_steps, grad_clip, d…)` | Training configuration for NoetherSurrogateTrainer. |
| `NoetherTrainResult` | classe | `(train_losses, val_losses, best_val_loss, best_epoch, checkpoint_path, extras)` | Summary returned by ``NoetherSurrogateTrainer.train()``. |
| `NoetherSurrogateTrainer` | classe | `(model, train_loader, val_loader=None, config=None) -> None` | Train a Noether (UPT / AB-UPT / Transolver …) model with PINNeAPPle data and report training results in a PINNeAPPle-compatible way. |

#### `pinneapple_neural.trainer.normalizers`

Normalizers — re-exported from pinneapple_data.transforms (single source of truth).

<small>`pinneapple_neural/trainer/normalizers.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_neural.trainer.parallel`

Parallelization and GPU optimization utilities for pinneapple training.

<small>`pinneapple_neural/trainer/parallel.py` · 508 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AMPContext` | classe | `(device='cuda', dtype=torch.bfloat16, enabled=True) -> None` | Context manager for automatic mixed precision training. |
| `CUDAPrefetcher` | classe | `(loader, device='cuda') -> None` | Asynchronous GPU prefetcher that overlaps CPU→GPU data transfer with GPU computation using CUDA streams. |
| `GradAccumConfig` | classe | `(accumulate_steps, clip_grad_norm)` | Configuration for gradient accumulation. |
| `GradAccumTrainer` | classe | `(model, optimizer, cfg=None, amp_ctx=None) -> None` | Mixin / wrapper that adds gradient accumulation to any training loop. |
| `SweepConfig` | classe | `(param_grid, n_jobs, backend, timeout_per_trial)` | Hyperparameter sweep configuration. |
| `ThroughputMonitor` | classe | `(device_index=0) -> None` | Training throughput and GPU health monitor. |
| `best_device` | função | `(prefer_cuda=True, allow_mps=True) -> torch.device` | Return the best available device: CUDA, then Apple MPS (unless ``allow_mps=False``), then CPU. |
| `count_gpus` | função | `() -> int` | Return the number of available CUDA GPUs. |
| `gpu_info` | função | `() -> List[Dict[str, Any]]` | Return info dict for each available GPU. |
| `maybe_compile` | função | `(model, *, mode='default', fullgraph=False, dynamic=False) -> nn.Module` | Wrap model with torch.compile() if supported (PyTorch >= 2.0). |
| `wrap_data_parallel` | função | `(model, device_ids=None) -> nn.Module` | Wrap model in DataParallel if multiple GPUs are available. |
| `unwrap_model` | função | `(model) -> nn.Module` | Unwrap DataParallel / DistributedDataParallel to get the raw model. |
| `run_parallel_sweep` | função | `(trial_fn, cfg) -> List[Dict[str, Any]]` | Run a hyperparameter sweep in parallel. |
| `enable_gradient_checkpointing` | função | `(model) -> nn.Module` | Enable gradient checkpointing on transformer / large MLP models. |

#### `pinneapple_neural.trainer.preprocess`

Preprocessing pipeline and preprocessing steps for training batches.

<small>`pinneapple_neural/trainer/preprocess.py` · 445 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PreprocessStep` | classe | `()` |  |
| `PreprocessPipeline` | classe | `(steps)` |  |
| `NormalizeStep` | classe | `(key, dim, store_key, enabled, scaler)` | Fits a StandardScaler on `key` from train batches, then normalizes in apply(). Works for typical (B,T,D) or (B,D) tensors. |
| `SolverFeatureStep` | classe | `(solver, mode, select_var_dim, reduce_fft_to, fit_noop)` |  |
| `MissingValueStep` | classe | `(key, strategy, enabled)` | Fill missing values (NaNs) in x. |
| `WinsorizeStep` | classe | `(key, q_low, q_high, enabled, low_, high_)` | Quantile clipping (winsorization) fit on the train fold. |
| `RobustScaleStep` | classe | `(key, eps, enabled, median_, iqr_)` | Robust scaling: (x - median) / (IQR + eps), fit on the train fold. |
| `LogStep` | classe | `(key, enabled, clamp_min)` | Optional log transform for non-negative signals (fold-safe). |
| `DifferencingStep` | classe | `(key, time_dim, order, enabled, anchor_key)` | Differencing along the time axis to improve stationarity. |

#### `pinneapple_neural.trainer.self_scaled_qn`

Self-scaled quasi-Newton optimizers: BFGS, SSBFGS and SSBroyden (dense inverse Hessian).

<small>`pinneapple_neural/trainer/self_scaled_qn.py` · 186 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SelfScaledQuasiNewton` | classe | `(params, variant='ssbroyden', max_iter=20, tolerance_grad=1e-12, tolerance_change=1e-14, …)` | Dense BFGS / SSBFGS / SSBroyden with strong-Wolfe line search (Urban et al., JCP 2025). |
| `self_scaled_parameters` | função | `(s, y, g, alpha, Hy, n, variant)` | Return ``(tau, phi)`` for one update (eqs. 11-23). Tensors are 1-D, ``g`` is the old gradient. |

#### `pinneapple_neural.trainer.splits`

Train/val/test split strategies — re-exported from pinneapple_data.splits.

<small>`pinneapple_neural/trainer/splits.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_neural.trainer.time_marching`

Sequential time-marching for stiff/long-horizon PDE problems.

<small>`pinneapple_neural/trainer/time_marching.py` · 388 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TimeMarchingTrainer` | classe | `(model_factory, t_start, t_end, n_windows=10, epochs_per_window=2000, n_col=1000, ic_weig…)` | Trains a PINN by marching sequentially through time windows. |

#### `pinneapple_neural.trainer.trainer`

Unified Trainer with physics-aware losses, callbacks, and audit logging.

<small>`pinneapple_neural/trainer/trainer.py` · 397 linhas · testes: 151</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TrainConfig` | classe | `(epochs, lr, weight_decay, grad_clip, amp, device, seed, deterministic, log_dir, run_name…)` |  |
| `Trainer` | classe | `(model, loss_fn, metrics=None, preprocess=None, early_stopping=None, checkpoint=None)` | Trainer for PINNs and general supervised models. |

#### `pinneapple_neural.trainer.two_phase`

Two-phase PINN trainer: supervised (Phase 1) → physics (Phase 2).

<small>`pinneapple_neural/trainer/two_phase.py` · 414 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TwoPhaseConfig` | classe | `(phase1_epochs, phase1_lr, phase2_epochs, phase2_lr, weight_decay, grad_clip, device, amp…)` |  |
| `TwoPhaseHistory` | classe | `(phase1_epochs, phase1_loss, phase2_epochs, phase2_loss, phase2_pde_loss, phase2_bc_loss)` |  |
| `TwoPhaseTrainer` | classe | `(model, supervised_loss_fn, physics_loss_fn, config=None)` | Auto-scheduled two-phase trainer for PINNs. |
| `UnnormModel` | classe | `(model, Y_mean, Y_std)` | Wraps a normalised model to output physical-scale predictions. |

#### `pinneapple_neural.trainer.viz`

Visualization utilities for training history.

<small>`pinneapple_neural/trainer/viz.py` · 57 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): shim ou obsoleto</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_history` | função | `(history, keys=('train_total', 'val_total'), *, log_scale=True, show=True)` | Plot training history. |

#### `pinneapple_neural.trainer.weight_scheduler`

Automatic loss weight scheduling for PINN training.

<small>`pinneapple_neural/trainer/weight_scheduler.py` · 799 linhas · testes: 11</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `WeightSchedulerConfig` | classe | `(method, initial_weights, update_every, lr, alpha, clip_min, clip_max, ema_decay)` | Configuration for automatic PINN loss weight scheduling. |
| `SelfAdaptiveWeights` | classe | `(loss_names, config) -> None` | Learnable loss weights trained by gradient *ascent*. |
| `GradNormBalancer` | classe | `(model, loss_names, shared_layer=None, config=None) -> None` | Gradient-norm-based weight balancing. |
| `LossRatioBalancer` | classe | `(loss_names, target_ratios=None, config=None) -> None` | Keep loss component ratios near target values using EMA smoothing. |
| `NTKWeightBalancer` | classe | `(model, loss_names, n_proj=50, config=None) -> None` | NTK-trace based weight balancing. |
| `WeightScheduler` | classe | `(model, loss_names, config=None) -> None` | Unified interface for automatic PINN loss weight scheduling. |

#### `pinneapple_neural.utils`

Small helpers shared by trainers, solvers and examples.

<small>`pinneapple_neural/utils.py` · 92 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `as_tensor` | função | `(out, *, attrs=OUTPUT_TENSOR_ATTRS) -> torch.Tensor` | Return the prediction Tensor of a model output. |
| `fit_supervised` | função | `(model, x, y, *, steps=1000, lr=0.001, batch=32, loss='mse', weight_decay=0.0, seed=0, fo…)` | Train ``model`` to map ``x`` to ``y`` (tensors or arrays, first axis = samples) with Adam on random mini-batches for ``steps`` steps; returns the loss history. ``loss``: ``"mse"`` or ``"rel_l2"`` (per-sample relative L2, for fields). ``forward(model, xb)`` overrides the call (default … |

#### `pinneapple_neural.workflows`

End-to-end train/predict workflows over PINNeAPPle's neural architectures -- reusable, product-facing entry points (as opposed to ``pinneapple_neural.architectures``/``trainer``, which expose the lower-level building blocks these workflows compose).

<small>`pinneapple_neural/workflows/__init__.py` · 56 linhas · **sem teste direto**</small>

#### `pinneapple_neural.workflows.dense_volume_operator`

A reusable train/predict workflow for a grid-based neural operator (:class:`~pinneapple_neural.architectures.neural_operators.fno.FNO3d`) trained on a DENSE-in-time real volume time series -- e.g. the output of :func:`pinneapple_data.adapters.load_dense_volumes` from a ``.splash`` OpenFOAM archive, …

<small>`pinneapple_neural/workflows/dense_volume_operator.py` · 248 linhas · testes: 3</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DenseVolumeOperatorConfig` | classe | `(in_channels, out_channels, width, modes, layers, epochs, lr, val_frac, batch_size, rollo…)` |  |
| `train_dense_volume_operator` | função | `(frames, cfg, *, seed=0) -> Tuple[Checkpoint, List[Dict[str, Any]]]` | Train an :class:`FNO3d` on a real dense volume time series. |
| `load_dense_volume_operator` | função | `(checkpoint, *, device='cpu') -> nn.Module` | Reconstruct the real ``FNO3d`` from a checkpoint produced by :func:`train_dense_volume_operator` (or the same-shaped legacy checkpoints ``train_fno3d.py`` already produced before this module existed -- both save the same ``meta`` keys). |
| `predict_single_step` | função | `(checkpoint, input_frame, *, device='cpu') -> np.ndarray` | One forward pass: ``input_frame`` (C, D, H, W), real (not normalized) units in, real units out. This is the regime the network is most accurate in -- see the module docstring. |
| `rollout_dense_volume_operator` | função | `(checkpoint, initial_frame, n_steps, *, device='cpu') -> np.ndarray` | Autoregressive rollout from a real starting frame. Returns ``(n_steps + 1, C, D, H, W)`` real-unit frames, index 0 = the given ``initial_frame`` unchanged, indices 1..n_steps = the model's own successive predictions fed back in as input. |

#### `pinneapple_neural.workflows.hybrid_surrogate_physics`

Hybrid model: a data-driven surrogate for intermediate quantities + an explicit physics post-model.

<small>`pinneapple_neural/workflows/hybrid_surrogate_physics.py` · 517 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsPostModel` | classe | `(name, source)` | An explicit physics model applied to the surrogate's intermediate quantities. |
| `CallablePostModel` | classe | `(fn, name, source)` | Wrap a plain function as a :class:`PhysicsPostModel`. |
| `SurrogatePrediction` | classe | `(mean, std, sampler)` | Predictive distribution of a surrogate over ``m`` flat outputs at ``n`` points. |
| `Surrogate` | classe | `()` | Minimal surrogate interface: ``fit(X, Y)`` with X (n, d), Y (n, m); ``predict(X)`` returns a :class:`SurrogatePrediction`, a ``(mean, std)`` tuple (std may be None) or a mean array. |
| `GaussianProcessSurrogate` | classe | `(*, n_restarts=2, random_state=0, use_sklearn=None)` | One independent Gaussian process per output column. |
| `PCASurrogate` | classe | `(base, *, n_components=None, variance=None)` | Reduce the outputs with PCA, fit ``base`` on the scores, map predictions back. |
| `TransformedSurrogate` | classe | `(base, forward, inverse, name='transform')` | Fit ``base`` on ``forward(Y)``; predictions are mapped back with ``inverse``. |
| `PerQuantitySurrogate` | classe | `(blocks)` | One surrogate per block of output columns (e.g. one per intermediate quantity). |
| `HybridPrediction` | classe | `(intermediates, derived_at_mean, derived_mean, derived_std, derived_lower, derived_upper,…)` | Output of :meth:`HybridSurrogatePhysics.predict`. |
| `HybridSurrogatePhysics` | classe | `(surrogate, post_model, *, param_names=None)` | Surrogate of intermediate quantities composed with an explicit :class:`PhysicsPostModel`. |

#### `pinneapple_neural.workflows.parametric_dense_volume_operator`

A joint, multi-case extension of :mod:`dense_volume_operator`: one :class:`~pinneapple_neural.architectures.neural_operators.fno.FNO3d` trained across SEVERAL real dense-in-time volume time series that share the same physical problem but differ in a scalar (or low-dim) physical parameter ``theta`` …

<small>`pinneapple_neural/workflows/parametric_dense_volume_operator.py` · 382 linhas · testes: 3</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ParametricCase` | classe | `(name, frames, theta)` | One real training case: a dense-in-time volume sequence plus the scalar (or low-dim) physical parameter(s) that produced it. |
| `ParametricDenseVolumeOperatorConfig` | classe | `(state_channels, theta_dim, width, modes, layers, epochs, lr, val_frac, batch_size, rollo…)` |  |
| `train_parametric_dense_volume_operator` | função | `(cases, cfg, *, seed=0) -> Tuple[Checkpoint, List[Dict[str, Any]]]` | Jointly train one theta-conditioned :class:`FNO3d` across ``cases``. |
| `load_parametric_dense_volume_operator` | função | `(checkpoint, *, device='cpu') -> nn.Module` |  |
| `predict_single_step_parametric` | função | `(checkpoint, input_frame, theta, *, device='cpu') -> np.ndarray` | One forward pass at a given (possibly never-trained-on) ``theta``. ``input_frame``: (state_c, D, H, W), real units in, real units out. |
| `rollout_parametric_dense_volume_operator` | função | `(checkpoint, initial_frame, theta, n_steps, *, device='cpu') -> np.ndarray` | Autoregressive rollout at a given ``theta``, re-attaching the (constant) theta channel after every step -- see module docstring ("theta is a condition, not a predicted state"). Returns ``(n_steps + 1, state_c, D, H, W)``, index 0 = ``initial_frame``. |

### Exemplos de uso reais do repositório

_Nenhum exemplo encontrado no repositório para este pacote._

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

- _Nenhuma registrada._

## 2. `pinneapple_data`

### Objetivo

_Sem texto de objetivo._

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 115 | 29.019 | 129 | 256 | 507 (54 arquivos) | 70 |

Pasta de testes dedicada: `tests/pinneapple_data` (33 funções de teste).

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `active_learning` |  | 1 | 5 | 22 |
| `adapters` |  | 5 | 19 | 0 |
| `audit` |  | 1 | 10 | 49 |
| `augmentation` |  | 1 | 3 | 0 |
| `cache` |  | 1 | 2 | 0 |
| `cache_bytes` |  | 1 | 3 | 1 |
| `cae` |  | 13 | 74 | 62 |
| `chunking` |  | 1 | 2 | 0 |
| `collate` |  | 1 | 3 | 0 |
| `collocation` |  | 1 | 2 | 7 |
| `contracts` |  | 4 | 27 | 175 |
| `data_health` |  | 1 | 6 | 37 |
| `dataloaders` |  | 1 | 3 | 0 |
| `datasets` |  | 9 | 28 | 10 |
| `device` |  | 1 | 2 | 0 |
| `doe` |  | 4 | 10 | 21 |
| `expressions` |  | 1 | 5 | 22 |
| `formfill` |  | 14 | 31 | 44 |
| `geometry_features` |  | 1 | 3 | 0 |
| `grf_sampler` |  | 1 | 5 | 0 |
| `lineage` |  | 3 | 4 | 13 |
| `noether_dataset` |  | 1 | 3 | 5 |
| `parameter_sampling` |  | 1 | 2 | 5 |
| `physical_sample` |  | 1 | 1 | 28 |
| `physical_units` |  | 1 | 12 | 52 |
| `physics_case` |  | 1 | 2 | 47 |
| `prefetch_adaptive` |  | 1 | 2 | 0 |
| `preflight` |  | 4 | 11 | 13 |
| `process_optimizer` |  | 1 | 7 | 16 |
| `serialization` |  | 1 | 9 | 11 |
| `simulation_metadata` |  | 5 | 25 | 25 |
| `snapshots` |  | 1 | 3 | 31 |
| `splits` |  | 1 | 2 | 0 |
| `stl_import` |  | 1 | 3 | 0 |
| `synth` |  | 15 | 20 | 0 |
| `transforms` |  | 1 | 3 | 2 |
| `udr1` |  | 1 | 3 | 36 |
| `unified` |  | 1 | 9 | 24 |
| `upd_dataset` |  | 1 | 1 | 5 |
| `upd_types` |  | 1 | 5 | 0 |
| `validators` |  | 1 | 3 | 0 |
| `zarr_cached_store` |  | 1 | 2 | 0 |
| `zarr_cached_store_bytes` |  | 1 | 2 | 0 |
| `zarr_iterable` |  | 1 | 1 | 0 |
| `zarr_prefetch` |  | 1 | 2 | 0 |
| `zarr_shard_iterable` |  | 1 | 2 | 0 |
| `zarr_shards` |  | 1 | 2 | 0 |
| `zarr_store` |  | 1 | 1 | 1 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_data`

_Módulo sem docstring e sem texto._

<small>`pinneapple_data/__init__.py` · 117 linhas · testes: 78</small>

#### `pinneapple_data.active_learning`

Active Learning for adaptive collocation in PINN training.

<small>`pinneapple_data/active_learning.py` · 674 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ActiveLearningConfig` | classe | `(strategy, n_candidates, n_select, n_initial, n_iterations, exploitation_weight, seed)` |  |
| `ResidualBasedAL` | classe | `(config, bounds) -> None` | Sample collocation points where PDE residual is highest (RAR/RAD). |
| `VarianceBasedAL` | classe | `(config, bounds) -> None` | Sample where model output variance is highest (requires a UQ model). |
| `CombinedAL` | classe | `(config, bounds, residual_weight=0.6, variance_weight=0.3, diversity_weight=0.1) -> None` | Weighted combination: exploit high residual + explore uncertain regions. |
| `AdaptiveCollocationTrainer` | classe | `(model, physics_fn, sampler, al_config=None, optimizer=None, device='cpu') -> None` | Wraps a PINN model with active-learning collocation updates. |

#### `pinneapple_data.adapters`

_Módulo sem docstring e sem texto._

<small>`pinneapple_data/adapters/__init__.py` · 63 linhas · **sem teste direto**</small>

#### `pinneapple_data.adapters.geom_adapter`

Geometry loading and attachment utilities for PhysicalSample via pinneapple_geom.

<small>`pinneapple_data/adapters/geom_adapter.py` · 286 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GeometryLoadOptions` | classe | `(scale, translate, rotate_euler_deg, repair, compute_normals, boundary_labels)` | Configuration container for geometry or mesh loading behavior. |
| `load_geometry_asset` | função | `(geom, *, options=None) -> Any` | Load or construct a GeometryAsset via pinneapple_geom. |
| `stl_to_upd` | função | `(path, *, repair=True, compute_normals=True, units='m') -> Any` | Load an STL and package it as a UPD-aligned PhysicalSample. |
| `attach_geometry` | função | `(sample, geom_asset) -> Any` | Attach a GeometryAsset to a PhysicalSample-like structure. |

#### `pinneapple_data.adapters.pinn_batch_builders`

PINN batch builders: convert various data sources into the unified dict-batch format.

<small>`pinneapple_data/adapters/pinn_batch_builders.py` · 149 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PINNBatch` | classe | `(batch)` | In-memory PINN-ready batch (thin wrapper around the dict). |
| `build_from_bundle` | função | `(bundle, *, n_collocation=4096, n_boundary=2048, n_data=2048, device=None, dtype=torch.fl…)` | Create a PINN-style batch from an Arena BundleData. |
| `build_from_solver` | função | `(problem_spec, geometry, solver_cfg, *, device=None, dtype=torch.float32) -> PINNBatch` | Build a PINN batch by running a solver or using pre-computed tensors. |
| `build_from_real_data` | função | `(adapter_cfg, *, device=None, dtype=torch.float32) -> PINNBatch` | Build a PINN batch from real measurement data. |

#### `pinneapple_data.adapters.splash_archive_adapter`

Loader for a Splash-CFD ``.splash`` case (a zipped OpenFOAM case directory) into plain numpy arrays -- both a flat point-cloud form (for PINN-style training) and a dense structured-grid volume time series (for grid-based neural operators like :class:`~pinneapple_neural. …

<small>`pinneapple_data/adapters/splash_archive_adapter.py` · 315 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SplashMesh` | classe | `(n_cells, n_points, n_faces, n_internal_faces, cell_centers, cell_size, cell_delta, bound…)` |  |
| `FieldSnapshot` | classe | `(time, fields)` |  |
| `SplashCase` | classe | `(path, mesh, time_dirs, transport, meta)` |  |
| `load_mesh` | função | `(zf, prefix='constant/polyMesh/') -> SplashMesh` | Build cell centers/sizes from the binary polyMesh inside a .splash zip. ``zf`` is an open ``zipfile.ZipFile``. |
| `save_mesh_cache` | função | `(mesh, path) -> None` |  |
| `load_mesh_cache` | função | `(path) -> SplashMesh` |  |
| `open_case` | função | `(path, mesh_cache_dir=None) -> SplashCase` | Open a ``.splash`` file, reconstructing (or loading a cached) mesh. |
| `load_dense_volumes` | função | `(splash_path, field_components, nx, ny, nz, *, mesh_cache_dir=None) -> Tuple[np.ndarray, …)` | Load every time directory of a ``.splash`` archive as a dense volume time series, for a structured single hex block mesh. |

#### `pinneapple_data.adapters.upd_adapter`

UPD shard loading and PhysicalSample conversion adapters.

<small>`pinneapple_data/adapters/upd_adapter.py` · 288 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `load_upd_item` | função | `(upd) -> Tuple[xr.Dataset, Dict[str, Any]]` | Load a UPD shard dataset and its associated metadata. |
| `upd_to_physical_sample` | função | `(upd, *, schema_override=None, provenance_extra=None) -> Any` | Convert a UPD shard into a PhysicalSample instance. |
| `attach_upd_state` | função | `(sample, upd) -> Any` | Attach UPD dataset state and metadata to an existing sample object. |

#### `pinneapple_data.audit`

Physics audit of a training dataset: does the data solve its PDE? Split leakage, near duplicates, diversity and units (#700). NumPy/SciPy only.

<small>`pinneapple_data/audit.py` · 519 linhas · testes: 49</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AuditCheck` | classe | `(id, status, summary, value, threshold, evidence)` |  |
| `AuditCard` | classe | `(dataset, splits, physics, checks, tolerances)` |  |
| `DatasetPhysics` | classe | `(name, columns, check, tolerances, units)` | Per-sample physics of a dataset. ``check(sample) -> {name: relative error}`` for one sample (a mapping of column -> array without the sample axis); ``tolerances`` maps each name to its limit (keys of :data:`DEFAULT_TOLERANCES` or numbers). |
| `darcy_consistency` | função | `(a, u, f=1.0) -> dict[str, float]` | Darcy sample checks in the discretisation of ``operator_benchmarks.darcy_2d`` (cell-centred, ``h = 1/n``): ``solve`` = relative L2 distance of ``u`` to the solution of its own discrete problem; ``bc`` = RMS difference of the boundary values (linear extrapolation of the first two cells to each face) … |
| `effective_sample_count` | função | `(x) -> float` | Participation ratio ``(sum l)^2 / sum l^2`` of the principal-component spectrum of the samples (rows): about the number of independent directions the samples span; 1 when every sample is a multiple of one pattern. |
| `near_duplicates` | função | `(x, near=0.001) -> dict[str, Any]` | Exact and near duplicates inside one set (rows), relative to the median nearest-neighbour distance. |
| `leakage` | função | `(train, test, near=0.001) -> dict[str, Any]` | Nearest training sample of each test sample; a test sample closer than ``near`` x the median training nearest-neighbour distance is a (near) copy. |
| `constraint_mask` | função | `(expr, cols) -> np.ndarray` | Where the declared relation ``expr`` (``"Cd > 0"``, ``"abs(u) <= 1"``, ``"a == b * c"`` to 1e-6 relative) holds, element-wise over ``cols``: the evaluator the ``constraints`` check uses, for checking predictions too. |
| `audit_dataset` | função | `(splits, *, name='dataset', physics=None, inputs=None, units=None, expected_units=None, c…)` | Audit ``splits`` (``{split: {column: array}}``). |
| `audit_zoo_dataset` | função | `(name, *, offline=None, **kw) -> AuditCard` | Audit a ``pp.datasets`` entry with the physics and constraints its catalog entry states. |

#### `pinneapple_data.augmentation`

Point-cloud dataset augmentation strategies for PDE/PINN training data.

<small>`pinneapple_data/augmentation.py` · 139 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `augment_noise` | função | `(coords, fields, n_copies, std_fraction=0.05, seed=0) -> Dict[str, Any]` | Add Gaussian noise (scaled to each field's own std) to `n_copies` tiled copies of the dataset. Coordinates are unchanged; only field values are perturbed. |
| `augment_transform` | função | `(coords, fields, n_copies, max_rotation_deg=15.0, scale_range=(0.9, 1.1), seed=0) -> Dict…)` | Random in-plane rotation (about the first two coordinate axes) plus isotropic scaling of the coordinate frame, `n_copies` times. Field values are treated as physical quantities sampled at each point, not coordinate-derived geometry — a rigid rotation/scale of the frame leaves field *values* … |
| `augment_interpolation` | função | `(coords, fields, n_new_points, method='linear', seed=0) -> Dict[str, Any]` | Generate `n_new_points` new samples inside the convex hull of the original point cloud, with field values estimated via scattered-data interpolation (falling back to nearest-neighbor wherever the requested method produces NaN, e.g. outside the interpolation method's support). |

#### `pinneapple_data.cache`

Thread-safe LRU cache for in-memory data with configurable capacity.

<small>`pinneapple_data/cache.py` · 120 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CacheStats` | classe | `(hits, misses, evictions)` | Statistics container for the LRUCache. |
| `LRUCache` | classe | `(max_items=256)` | Thread-safe LRU cache. |

#### `pinneapple_data.cache_bytes`

Byte-constrained LRU cache for memory-bounded caching of tensors and samples.

<small>`pinneapple_data/cache_bytes.py` · 179 linhas · testes: 1</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ByteCacheStats` | classe | `(hits, misses, evictions, bytes_in_use)` | Statistics container for the ByteLRUCache. |
| `ByteLRUCache` | classe | `(max_bytes=512 * 1024 * 1024, weigher=default_weigher)` | Thread-safe LRU cache constrained by max_bytes (not max items). |
| `default_weigher` | função | `(value) -> int` | Estimate memory usage in bytes. |

#### `pinneapple_data.cae`

CAE data layer: read meshes and results from OpenFOAM, CalculiX/Abaqus, Gmsh, VTK, STL and point tables into one ``Mesh`` type; mesh quality (checkMesh-equivalent finite-volume metrics, element shape metrics, surface checks); per-cell Courant number and wall distance.

<small>`pinneapple_data/cae/__init__.py` · 14 linhas · testes: 44</small>

#### `pinneapple_data.cae.compare`

Compare two results field by field: simulation vs simulation (different meshes or models), vs experiment (sensor tables), vs AI predictions (point tables or meshes), vs analytical or benchmark data.

<small>`pinneapple_data/cae/compare.py` · 420 linhas · testes: 24</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Interpolator` | classe | `(src_pts, dst_pts, method='auto')` | Values of a candidate field at query points: identical locations, linear (Delaunay), nearest outside. |
| `locations` | função | `(m, field) -> Tuple[np.ndarray, str, Optional[np.ndarray]]` | (points where the field lives, 'cell'\|'point', weights). |
| `cell_centres_volumes` | função | `(m) -> Tuple[np.ndarray, Optional[np.ndarray]]` |  |
| `field_value` | função | `(m, f) -> np.ndarray` |  |
| `fields_of` | função | `(m) -> Dict[str, Dict[str, Any]]` |  |
| `match_fields` | função | `(ref, cand, mapping=None) -> List[Tuple[str, str]]` | Pairs (reference field, candidate field): explicit mapping first, then same name (case-insensitive), then aliases (U ~ velocity, p ~ pressure, T ~ temperature, DISP ~ displacement, ...). |
| `tolerance_verdicts` | função | `(result, tolerance) -> Optional[str]` | Mark every field of a :func:`compare` result ``"pass"`` / ``"fail"`` by its relative L2 error against ``tolerance`` (a fraction, e.g. 0.05 for 5 %), in ``field["verdict"]``. Returns ``"PASS"`` when every judged field passes, ``"FAIL"`` otherwise, ``None`` when nothing was judged (no tolerance or no … |
| `compare` | função | `(ref, cand, mapping=None, method='auto', fields=None) -> Dict[str, Any]` |  |
| `surface` | função | `(m, max_tris=300000) -> Optional[Dict[str, Any]]` | Boundary surface for display: triangles with the cell each one belongs to. For 2D OpenFOAM cases (empty patches) only one of the two empty sides is kept. |
| `views` | função | `(ref, res, max_points=60000) -> Dict[str, Any]` | Per-field display payload: a surface coloured per cell (or per node), a line plot, or a 2D/3D point map. |

#### `pinneapple_data.cae.dataset`

Neutral physical dataset: any supported result (OpenFOAM, CalculiX, VTK/VTU, Gmsh, point tables) normalised into one representation (geometry, mesh, coordinates, fields with quantity and unit, metadata) and written to VTK (.vtu), HDF5, Parquet, CSV, NPZ, JSON or a PINNeAPPle dataset (UPD Zarr …

<small>`pinneapple_data/cae/dataset.py` · 507 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `quantity_of` | função | `(name) -> Tuple[Optional[str], Optional[str]]` |  |
| `poly_cells` | função | `(m)` | OpenFOAM cells as VTK cells: hexahedra, wedges, tetrahedra and pyramids are rebuilt from their faces; any other cell becomes a VTK polyhedron (list of faces). Returns (blocks [(type, conn)], polyhedra faces, cell order). |
| `field_scalar` | função | `(values) -> Tuple[Optional[np.ndarray], str]` | One scalar per point or cell: the value itself, the magnitude of a vector (up to 3 components) or the von Mises stress of a symmetric tensor in Voigt order ``xx yy zz xy yz zx``. Returns ``(scalar, kind)``; ``scalar`` is ``None`` for other shapes. |
| `build_dataset` | função | `(m, *, unit_system='auto', to_si=False, rho=None, location='auto', files=None) -> Dict[st…)` | Normalise a Mesh into the neutral dataset (a dict holding numpy arrays and metadata). |
| `describe` | função | `(ds) -> Dict[str, Any]` | JSON-safe description (the schema the user sees): no arrays. |
| `export` | função | `(ds, fmt) -> Tuple[bytes, str, str]` | (content, media type, file extension). |
| `write_vtu` | função | `(ds) -> bytes` |  |
| `write_hdf5` | função | `(ds) -> bytes` |  |
| `write_parquet` | função | `(ds) -> bytes` |  |
| `write_csv` | função | `(ds) -> bytes` |  |
| `write_npz` | função | `(ds) -> bytes` |  |
| `write_json` | função | `(ds, max_values=2000000) -> bytes` |  |
| `write_pinneapple` | função | `(ds) -> bytes` | A PINNeAPPle dataset: one PhysicalSample (state = coordinates and fields at the chosen location) written with the library's UPD Zarr store (pinneapple_data.serialization.save_zarr), zipped. Load it with ``load_zarr`` after unzipping; units, quantities and the source are in the sample's provenance. |
| `preview` | função | `(ds, max_fields=6, max_tris=200000) -> Dict[str, Any]` | Display payload of a dataset from :func:`build_dataset`: the boundary surface (always, so a bare mesh shows too) coloured by the first ``max_fields`` fields reduced to one scalar each (:func:`field_scalar`), rounded to 5 significant digits. Empty dict when the mesh has no surface. |

#### `pinneapple_data.cae.foam`

OpenFOAM readers: polyMesh (points, faces, owner, neighbour, boundary) and volume fields of a time directory.

<small>`pinneapple_data/cae/foam.py` · 372 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `header` | função | `(text) -> Dict[str, str]` |  |
| `read_points` | função | `(b) -> np.ndarray` |  |
| `read_labels` | função | `(b) -> np.ndarray` |  |
| `read_faces` | função | `(b) -> Tuple[np.ndarray, np.ndarray]` | Faces as (flat node list, offsets). |
| `read_boundary` | função | `(b) -> List[Dict[str, Any]]` |  |
| `read_polymesh` | função | `(fs, root='') -> Mesh` |  |
| `dims_unit` | função | `(dims) -> Optional[str]` |  |
| `read_boundary_values` | função | `(b, mesh, width) -> Optional[np.ndarray]` | Value on every boundary face (NaN for empty patches), from the boundaryField of a field file: the written 'value' (uniform or a list), zero for noSlip, the owner-cell value where nothing is written (zeroGradient). |
| `read_field` | função | `(b, n_cells) -> Tuple[np.ndarray, Optional[str], str]` | internalField of a vol*Field: (values per cell, unit, class). |
| `time_dirs` | função | `(fs, root='') -> List[str]` | Numeric time directories under the case root, ascending. |
| `read_time` | função | `(fs, mesh, root='', time=None, only=None) -> Dict[str, Any]` | Load the volume fields of one time directory (latest by default) into ``mesh.cell_data``. |
| `boundary_conditions` | função | `(fs, mesh, root='', time=None) -> Dict[str, Dict[str, str]]` | Patch -> field -> condition as written in the boundaryField of each field of one time directory (the first, which holds the initial and boundary conditions, by default): ``"fixedValue uniform (1 0 0)"``, ``"noSlip"``, ``"zeroGradient"``. Empty patches are left out. |

#### `pinneapple_data.cae.frd`

CalculiX result file (.frd, ASCII): nodes, elements and the result blocks (DISP, STRESS, NDTEMP, FLUX, ...).

<small>`pinneapple_data/cae/frd.py` · 117 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `read_frd` | função | `(b) -> Mesh` |  |

#### `pinneapple_data.cae.geometry`

Finite-volume geometry of any mesh, computed the way OpenFOAM does (primitiveMesh): face centres and area vectors by triangle decomposition about the face average point, cell centres and volumes by pyramid decomposition about the average of the face centres. FE volume meshes are turned into the …

<small>`pinneapple_data/cae/geometry.py` · 249 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FaceMesh` | classe | `(points, face_nodes, face_offsets, owner, neighbour, n_cells, patch_of_face, patch_names,…)` |  |
| `signed_volume_sign` | função | `(btype, pts, conn) -> np.ndarray` | Sign of a simple volume estimate of each element (tetra decomposition about its first corner). |
| `oriented_blocks` | função | `(mesh) -> List[str]` | Element node-order conventions differ (VTK, Gmsh, Abaqus wedges...). If most elements of a block come out with negative volume, the whole block uses the opposite convention: flip it, so only truly inverted elements stay negative. Returns the names of the flipped blocks. |
| `face_mesh` | função | `(mesh) -> FaceMesh` | OpenFOAM polyMesh as is; FE volume meshes converted (internal faces first, then boundary faces). |
| `face_geometry` | função | `(fm) -> Dict[str, np.ndarray]` | Face centres (nf,3), area vectors (nf,3), and the per-face corner groups used later. |
| `cell_geometry` | função | `(fm, fg) -> Dict[str, np.ndarray]` |  |
| `cell_adjacency` | função | `(fm) -> sp.csr_matrix` |  |
| `regions` | função | `(adj) -> np.ndarray` |  |
| `element_adjacency_by_nodes` | função | `(mesh) -> sp.csr_matrix` | Cells sharing at least one node (for surface/planar meshes and the node-connectivity check). |
| `solution_dirs` | função | `(fm, fg) -> Optional[np.ndarray]` | Directions with no 'empty' patch (OpenFOAM 2D/1D cases): boolean (3,). |

#### `pinneapple_data.cae.inp`

Abaqus / CalculiX input deck (.inp): keyword blocks, nodes, elements (to VTK types), node and element sets.

<small>`pinneapple_data/cae/inp.py` · 296 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Block` | classe | `(keyword, params, data, line, file)` |  |
| `Deck` | classe | `(blocks, includes, missing_includes)` |  |
| `vtk_type` | função | `(abq) -> Optional[str]` |  |
| `read_deck` | função | `(text, name='model.inp', files=None, _depth=0) -> Deck` |  |
| `resolve_sets` | função | `(deck, kind) -> Dict[str, List[str]]` | NSET or ELSET name (upper case) -> member labels; sets may contain other sets' names. |
| `deck_mesh` | função | `(deck) -> Mesh` |  |
| `deck_conditions` | função | `(deck) -> List[Dict[str, object]]` | The supports and loads of a deck as node ids (labels of ``*NODE``), one entry per keyword block and kind: ``*BOUNDARY`` (kind ``bc``: the constrained degrees of freedom and their values), ``*CLOAD`` / ``*CFLUX`` (kind ``load``: the sum of the nodal forces or fluxes per direction), ``*DLOAD`` / … |

#### `pinneapple_data.cae.io`

Read whatever was uploaded into a ``Mesh``: OpenFOAM case or polyMesh, Abaqus/CalculiX .inp, CalculiX .frd, STL, Gmsh .msh, VTK (.vtk/.vtu) and the other meshio formats, and point clouds with fields (CSV, HDF5, NPZ).

<small>`pinneapple_data/cae/io.py` · 296 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FormatError` | classe | `()` |  |
| `kind_of` | função | `(fs) -> Tuple[str, Optional[str]]` | (kind, main file or case root). Kinds: openfoam, inp, frd, stl, meshio, points. |
| `read_any` | função | `(fs, fields=True, time=None) -> Mesh` |  |
| `read_stl` | função | `(b) -> Mesh` |  |
| `read_meshio` | função | `(name, b) -> Mesh` |  |
| `read_points` | função | `(name, b) -> Mesh` | A table of points with field columns (CSV/TXT, Parquet, NPZ, HDF5): a point cloud mesh with point_data. |

#### `pinneapple_data.cae.metrics`

Per-cell flow metrics on a finite-volume mesh: Courant number and distance to the nearest wall.

<small>`pinneapple_data/cae/metrics.py` · 109 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `courant_number` | função | `(face_area_vectors, face_owner, face_neighbour, U, cell_volumes, dt, where='auto') -> np.…)` | Cell Courant number ``Co_P = 0.5 dt sum_f \|U_f . S_f\| / V_P`` over the faces f of cell P. |
| `wall_distance` | função | `(cell_centres, wall_face_centres, wall_face_normals=None) -> np.ndarray` | Distance from each cell centre to the nearest wall face centre (``scipy.spatial.cKDTree`` query). |
| `mesh_courant_number` | função | `(mesh, U, dt, where='auto') -> np.ndarray` | :func:`courant_number` on a :class:`Mesh` (cell or face velocity ``U``, see there). |
| `mesh_wall_distance` | função | `(mesh, patches=None, project=True) -> np.ndarray` | :func:`wall_distance` of every cell of a :class:`Mesh` to the faces of ``patches`` (default: every patch of type ``wall``). ``project`` uses the face normals (see :func:`wall_distance`). |

#### `pinneapple_data.cae.model`

The mesh/result container shared by the CAE readers, the mesh-quality checks, the comparator and the interop export.

<small>`pinneapple_data/cae/model.py` · 173 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CellBlock` | classe | `(type, conn, ids)` |  |
| `PolyMesh` | classe | `(face_nodes, face_offsets, owner, neighbour, n_cells, patches)` | OpenFOAM polyMesh: faces (flat node list + offsets), owner/neighbour, boundary patches. |
| `Mesh` | classe | `(points, blocks, poly, boundary_blocks, cell_sets, node_sets, point_data, cell_data, unit…)` |  |

#### `pinneapple_data.cae.quality`

Mesh quality: finite-volume metrics as OpenFOAM's checkMesh computes them, element shape metrics for finite-element meshes, connectivity, and surface (STL) checks; problem regions are clusters of bad cells.

<small>`pinneapple_data/cae/quality.py` · 306 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `fv_metrics` | função | `(mesh) -> Dict[str, Any]` |  |
| `element_metrics` | função | `(mesh) -> Dict[str, np.ndarray]` | Scaled Jacobian, skewness and edge ratio for every cell of the mesh's top dimension (global cell order). |
| `surface_checks` | função | `(mesh) -> Dict[str, Any]` | Triangle/quad surface (STL, shell mesh): open and non-manifold edges, orientation, degenerate faces, shells. |

#### `pinneapple_data.cae.report`

Mesh health report: checks with a verdict, statistics and histograms, problem regions (clusters of bad cells, with where they are) and what to do about each issue. Also the payload for a 3D view (boundary surface coloured by a metric, plus the bad cells).

<small>`pinneapple_data/cae/report.py` · 371 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build` | função | `(mesh, target='auto', max_regions=12) -> Dict[str, Any]` | target: 'cfd' (finite-volume thresholds lead), 'fea' (element-shape thresholds lead) or 'auto'. |
| `view_payload` | função | `(mesh, rep, max_tris=250000) -> Dict[str, Any]` | Boundary surface (triangles) with each face's owner-cell metric values, and the problem regions' cells. |
| `worst_cells` | função | `(mesh, rep, n=40) -> List[Dict[str, Any]]` | The ``n`` worst cells of every metric in a report from :func:`build`, worst first. |

#### `pinneapple_data.cae.upload`

Expand an upload (loose files and/or zip archives) into {relative path: bytes}, with limits.

<small>`pinneapple_data/cae/upload.py` · 45 linhas · testes: 62</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `expand` | função | `(files, max_unzipped=MAX_UNZIPPED) -> Dict[str, bytes]` |  |

#### `pinneapple_data.chunking`

Chunking specifications and tensor iteration for streaming large arrays.

<small>`pinneapple_data/chunking.py` · 112 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ChunkSpec` | classe | `(mode, chunk_size, dim, dims, chunk_sizes)` | Generic chunking specification for tensors. |
| `iter_chunks` | função | `(x, spec) -> Iterator[torch.Tensor]` | Iterate over chunks of a tensor according to a ChunkSpec. |

#### `pinneapple_data.collate`

Collate functions for PINN batches and UPD supervision batches.

<small>`pinneapple_data/collate.py` · 132 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `collate_pinn_batches` | função | `(batches) -> Dict[str, Any]` | Collate a list of PINN batch dicts into a single batch dict. |
| `move_batch_to_device` | função | `(batch, device) -> Dict[str, Any]` | Move all tensors in a collated PINN batch dict to *device*. Leaves ``meta`` unchanged. |
| `collate_upd_supervised` | função | `(samples) -> Dict[str, Any]` | Collate a list of PhysicalSample objects for supervised training. |

#### `pinneapple_data.collocation`

Automated collocation point generation for PINN training.

<small>`pinneapple_data/collocation.py` · 558 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CollocationConfig` | classe | `(n_col, n_bc, n_ic, n_data, strategy, seed, oversample_bc, include_time, t_range)` | Configuration for collocation point generation. |
| `CollocationSampler` | classe | `(*, bounds, coord_names, sdf_fn=None, boundary_samplers=None, condition_value_fns=None, c…)` | Unified collocation point generator for PINN training. |

#### `pinneapple_data.contracts`

Open data contracts shared by the PINNeAPPle apps: ``ParameterSpace``, ``DatasetSpec``, ``ModelRef``, ``StudyResult``.

<small>`pinneapple_data/contracts/__init__.py` · 51 linhas · testes: 175</small>

#### `pinneapple_data.contracts.expressions`

Safe evaluation of constraint expressions: re-exported from :mod:`pinneapple_data.expressions`, which needs only NumPy (``pinneapple check`` imports it without the contract models and their pydantic dependency).

<small>`pinneapple_data/contracts/expressions.py` · 12 linhas · **sem teste direto**</small>

#### `pinneapple_data.contracts.models`

The four data contracts shared by the PINNeAPPle apps and the library.

<small>`pinneapple_data/contracts/models.py` · 675 linhas · testes: 40</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Contract` | classe | `(schema_version, metadata)` | Base of the four contracts: JSON round trip, files, schema. |
| `Parameter` | classe | `(name, kind, low, high, values, value, unit, log, group, default, description)` |  |
| `ParameterSpace` | classe | `(kind, name, description, parameters, constraints, budget)` | The parameters of a study and the constraints between them. |
| `Column` | classe | `(name, role, unit, dtype, description)` |  |
| `FileRef` | classe | `(path, format, sha256, size, rows, description)` |  |
| `Provenance` | classe | `(app, study_id, source, run_ids, library_version, created, notes)` |  |
| `Splits` | classe | `(train, val, test, seed, id_column)` | Either explicit row ids per split or fractions with a seed (and optionally the column holding ids). |
| `DatasetSpec` | classe | `(kind, name, version, description, columns, files, n_rows, parameter_space, splits, prove…)` | A table of cases or samples: what the columns are, where the files are, how it is split. |
| `Variable` | classe | `(name, unit, low, high, description)` |  |
| `DatasetRef` | classe | `(name, version, content_hash, uri)` |  |
| `ModelRef` | classe | `(kind, name, version, model_type, uri, sha256, framework, inputs, outputs, dataset, metri…)` | Where a trained model is and what it may be used for. |
| `CheckRecord` | classe | `(name, status, value, threshold, evidence, detail)` | One check. PASS / FAIL / WARN need a record (value or evidence): a check is only reported as run if it ran. |
| `Table` | classe | `(columns, file)` | Small tables inline (columns of equal length); large ones as a file. |
| `StudyResult` | classe | `(kind, id, study_type, app, title, status, parameter_space, dataset, models, table, metri…)` | The outcome of a study, with what went in, what came out and which checks ran. |
| `from_dict` | função | `(data) -> Contract` | Build the contract named by ``data["kind"]``. |
| `load` | função | `(path) -> Contract` | Read a contract from JSON or YAML, whichever of the four it is. |
| `validate` | função | `(data) -> list[str]` | Problems with a contract given as a dict (empty list when valid), one readable line each. |
| `json_schemas` | função | `() -> dict[str, dict[str, Any]]` |  |
| `export_schemas` | função | `(directory) -> list[Path]` | Write ``<Kind>.schema.json`` for each contract into ``directory``. |
| `sha256_file` | função | `(path, chunk=1 << 20) -> str` |  |

#### `pinneapple_data.contracts.tables`

Tables on disk with their contract: CSV, Parquet and HDF5, each with a JSON sidecar.

<small>`pinneapple_data/contracts/tables.py` · 150 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `sidecar_path` | função | `(path) -> Path` |  |
| `table_format` | função | `(path) -> str` |  |
| `write_sidecar` | função | `(path, contract) -> Path` | Write ``<path>.contract.json``. |
| `read_sidecar` | função | `(path) -> Contract \| None` |  |
| `write_table` | função | `(df, path, contract=None) -> Path` | Write a DataFrame (or a dict of columns) as CSV / Parquet / HDF5 / JSON records, plus the sidecar. |
| `read_table` | função | `(path)` | Read a table and its contract: ``(DataFrame, contract or None)``. |
| `convert` | função | `(src, dst) -> Path` | Copy a table to another format (by suffix), carrying its contract along. |

#### `pinneapple_data.data_health`

Engineering data health: a physics-aware quality report for sensor / test / plant data.

<small>`pinneapple_data/data_health.py` · 998 linhas · testes: 37</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Issue` | classe | `(category, severity, title, detail, column, affected_fraction, count, spans, suggestion, …)` |  |
| `LoadInfo` | classe | `(format, rows, columns, delimiter, decimal, encoding, sheet, sheets, units_from_file, tru…)` |  |
| `load_table` | função | `(data, filename, sheet=None, max_rows=1000000) -> Tuple[pd.DataFrame, LoadInfo]` | Read an uploaded table. Supported: .csv .tsv .txt .xlsx .xlsm .xls .parquet .json .jsonl .h5 .hdf5 |
| `detect_time_column` | função | `(df) -> Tuple[Optional[str], Optional[pd.Series], str]` |  |
| `analyze` | função | `(df, units=None, time_column=None, n_bins=160, series_points=600) -> Dict[str, Any]` |  |
| `example_chiller_plant` | função | `(seed=7, days=7, inject=True) -> Tuple[pd.DataFrame, Dict[str, Any]]` | One week of 1-minute data from a chilled-water plant, with eight realistic faults injected. |

#### `pinneapple_data.dataloaders`

DataLoader builders for UPD shards and PhysicalSample-based PINN training.

<small>`pinneapple_data/dataloaders.py` · 353 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DataLoaderSpec` | classe | `(batch_size, num_workers, shuffle, pin_memory, drop_last)` | Configuration container for torch.utils.data.DataLoader parameters. |
| `build_upd_dataloader` | função | `(*, zarr_path, meta_path, mapping, sampling, loader=None, device='cpu', dtype=torch.float…)` | Construct a PyTorch DataLoader from a single UPD shard (Zarr + JSON). |
| `build_physical_sample_dataloader` | função | `(sample, *, mapping, sampling, loader=None, device='cpu', dtype=torch.float32, length=100…)` | Construct a PyTorch DataLoader from a PhysicalSample instance. |

#### `pinneapple_data.datasets`

Pre-built datasets for PINNeAPPle — physics, geometry, and time series.

<small>`pinneapple_data/datasets/__init__.py` · 152 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `load_dataset` | função | `(dataset_id, **kwargs) -> Dict[str, np.ndarray]` | Load a pre-built dataset by ID. |
| `list_datasets` | função | `(category=None, verbose=True) -> List[DatasetInfo]` | Print and return all registered datasets. |
| `dataset_info` | função | `(dataset_id) -> DatasetInfo` | Return metadata for a specific dataset. |
| `dataset_ids` | função | `(category=None) -> List[str]` | Return list of all registered dataset IDs. |

#### `pinneapple_data.datasets.deepmind_mgn`

DeepMind ``meshgraphnets`` datasets: partial download, TFRecord reader and conversion to tensors.

<small>`pinneapple_data/datasets/deepmind_mgn.py` · 116 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `download_prefix` | função | `(dataset, split, dest, nbytes) -> Path` | Download ``meta.json`` and the first ``nbytes`` of ``<split>.tfrecord`` of a DeepMind dataset into ``dest``. |
| `read_trajectories` | função | `(data_dir, split, max_traj) -> list[dict[str, np.ndarray]]` | Read up to ``max_traj`` trajectories of ``<data_dir>/<split>.tfrecord`` as dicts of numpy arrays. |
| `cylinder_flow_to_tensors` | função | `(traj, node_type_map=CYLINDER_TYPE_MAP) -> tuple[MeshGraph, torch.Tensor, torch.Tensor]` | ``cylinder_flow`` trajectory -> ``(MeshGraph, velocity (T, N, 2), pressure (T, N, 1))``. |

#### `pinneapple_data.datasets.download`

Download a URL once into a local cache, with retries (standard library only).

<small>`pinneapple_data/datasets/download.py` · 85 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `cache_path` | função | `(url, cache_dir) -> Path` | The file :func:`fetch_cached` uses for ``url``: ``<cache_dir>/<md5(url)><ext>``. |
| `fetch_cached` | função | `(url, cache_dir, *, tries=4, timeout=60, backoff=2.0) -> Path` | Return a local copy of ``url``, downloading it only if it is not cached yet. |

#### `pinneapple_data.datasets.external_sources`

External data source integrations for PINNeAPPle.

<small>`pinneapple_data/datasets/external_sources.py` · 1.168 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MaterialsProjectClient` | classe | `(api_key=None) -> None` | Client for the Materials Project API. |
| `NOMADClient` | classe | `()` | Client for the NOMAD Laboratory REST API (public, no key needed). |
| `NASAClient` | classe | `()` | Client for NASA Open Data Portal (data.nasa.gov). |
| `NOAAClient` | classe | `()` | Client for NOAA NCEI / ERDDAP APIs. |
| `CopernicusClient` | classe | `()` | Client for the Copernicus Data Space (CDS + CDSE). |
| `BrazilDataCubeClient` | classe | `()` | Client for the Brazil Data Cube STAC API. |
| `CPTECClient` | classe | `()` | Client for CPTEC/INPE weather and climate data. |
| `ONSClient` | classe | `()` | Client for ONS (Brazilian National Grid Operator) open data. |
| `NRELClient` | classe | `(api_key=None) -> None` | Client for NREL open data. |
| `DesignSafeClient` | classe | `()` | Client for DesignSafe-CI (earthquake / structural data). |
| `OpenFOAMClient` | classe | `(openfoam_data_dir=None) -> None` | Provides representative CFD data from OpenFOAM tutorial cases. |
| `SU2Client` | classe | `(su2_results_dir=None) -> None` | Provides aerodynamic data representative of SU2 tutorial cases. |
| `NASAPowerClient` | classe | `()` | Client for the NASA POWER daily point API (meteorology and solar resource, no API key). |
| `OpenMeteoClient` | classe | `()` | Client for the Open-Meteo Historical Weather API (ERA5-based reanalysis, hourly, no API key; CC BY 4.0). |

#### `pinneapple_data.datasets.geometry`

Pre-built geometry datasets for PINNeAPPle benchmarks.

<small>`pinneapple_data/datasets/geometry.py` · 269 linhas · **sem teste direto**</small>

#### `pinneapple_data.datasets.physics`

Pre-built physics datasets for PINNeAPPle benchmarks.

<small>`pinneapple_data/datasets/physics.py` · 450 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `burgers_periodic_solve` | função | `(u0, T, nu, *, length=1.0, refine=None, cfl=0.5) -> np.ndarray` | Solve periodic viscous Burgers ``u_t + (u^2/2)_x = nu u_xx`` from ``u0`` (``(..., nx)``) to time ``T``. |
| `burgers_periodic_pairs` | função | `(n, nx=128, T=1.0, nu=0.01, seed=0, *, length=1.0, refine=None)` | ``n`` random pairs ``(u(x, 0), u(x, T))`` of periodic viscous Burgers on ``[0, length)`` (operator learning). |

#### `pinneapple_data.datasets.real_world`

Real-world datasets loaded from the web or optional Python libraries.

<small>`pinneapple_data/datasets/real_world.py` · 1.665 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `load_real_dataset` | função | `(dataset_id) -> Dict[str, np.ndarray]` | Load a real-world dataset by ID. |

#### `pinneapple_data.datasets.registry`

Central registry for pre-built PINNeAPPle datasets.

<small>`pinneapple_data/datasets/registry.py` · 65 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DatasetInfo` | classe | `(id, name, category, description, fields, tags, license, reference)` |  |
| `DatasetRegistry` | classe | `(_store)` | Central registry for pre-built PINNeAPPle datasets. |

#### `pinneapple_data.datasets.timeseries`

Pre-built time-series datasets for PINNeAPPle benchmarks.

<small>`pinneapple_data/datasets/timeseries.py` · 355 linhas · **sem teste direto**</small>

#### `pinneapple_data.device`

Device utilities for pinning and moving PhysicalSample tensors between CPU and GPU.

<small>`pinneapple_data/device.py` · 148 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `pin_sample` | função | `(sample) -> Any` | Pin all tensor entries inside a PhysicalSample-like structure. |
| `to_device_sample` | função | `(sample, device, *, dtype=None, non_blocking=True) -> Any` | Move all tensor entries inside a PhysicalSample-like structure to a target device. |

#### `pinneapple_data.doe`

Design of experiments for simulation campaigns: designs, coverage, redundancy, priorities, campaign plans.

<small>`pinneapple_data/doe/__init__.py` · 20 linhas · testes: 21</small>

#### `pinneapple_data.doe.campaign`

Plan a simulation campaign: which cases to run, in which order, and what they add to what already ran.

<small>`pinneapple_data/doe/campaign.py` · 174 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CampaignPlan` | classe | `(space, design, cases, unit, coverage_planned, coverage_existing, coverage_combined, redu…)` |  |
| `plan_campaign` | função | `(space, n=None, method='lhs', *, existing=None, existing_id_column=None, outputs=(), seed…)` | Design ``n`` cases (default ``space.budget``), drop those repeating an existing run (``drop_repeats``), order the rest by priority and measure the coverage of the plan, of what exists, and of both together. Existing cases outside the space (bounds or constraints) are left out of the coverage and … |

#### `pinneapple_data.doe.coverage`

How well a set of cases covers a parameter space, which planned cases repeat what already ran, and in which order to run them.

<small>`pinneapple_data/doe/coverage.py` · 119 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `coverage` | função | `(unit, names=None, *, n_probe=2 ** 14, seed=0, bins_2d=None, levels=None) -> dict[str, An…)` | Space-coverage metrics of points in [0, 1]^d. |
| `redundancy` | função | `(planned_unit, existing_unit=None, *, tol=None, existing_ids=None) -> dict[str, Any]` | Which planned cases (nearly) repeat an existing case or another planned case. |
| `prioritize` | função | `(planned_unit, existing_unit=None) -> dict[str, Any]` | Run order: greedy maximin over the planned cases, starting from what already exists, so every prefix of the list covers the space as evenly as possible (stop the campaign early and the cases run are still spread out). ``score`` is each case's distance to everything before it (existing included): … |

#### `pinneapple_data.doe.designs`

Designs of experiments over a :class:`~pinneapple_data.contracts.ParameterSpace`.

<small>`pinneapple_data/doe/designs.py` · 283 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Design` | classe | `(cases, unit, method, info)` | The cases of a design: values (DataFrame with every parameter), unit coordinates and how they were made. |
| `design` | função | `(space, n, method='lhs', *, seed=0, existing=None, outputs=(), levels=3, n_restarts=20, f…)` | ``n`` cases of ``space`` by ``method`` (see the module docstring). ``existing`` is a DataFrame of cases already run (a column per parameter, plus ``outputs`` for adaptive infill). |
| `fractional_factorial_unit` | função | `(k, n) -> tuple[np.ndarray, dict]` | Two-level 2^(k-p) design with at least ``n`` runs (or 2^k if smaller): the first k-p factors are crossed in full, each further factor is the product of a distinct subset of them (largest subsets first, which keeps the resolution high). Returns unit coordinates (0 / 1) and the generators and … |
| `greedy_maximin` | função | `(points, n, fixed=None, weights_fixed=None) -> np.ndarray` | Indices of ``n`` of ``points`` picked one by one, each the farthest from those already picked and from ``fixed``. ``weights_fixed`` (one per fixed point, >= 1) favours the neighbourhoods to refine: a candidate's distance to the fixed points is multiplied by the weight of its nearest fixed point. |
| `min_distance` | função | `(unit) -> float` |  |

#### `pinneapple_data.expressions`

Safe evaluation of constraint expressions such as ``"chord - 2 * thickness >= 0.01"``.

<small>`pinneapple_data/expressions.py` · 131 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ExpressionError` | classe | `()` | The expression uses something that is not allowed, or a name that is not a parameter. |
| `names_in` | função | `(expression) -> set[str]` | Parameter names an expression refers to (function names excluded). |
| `evaluate` | função | `(expression, values, functions=None) -> np.ndarray` | Evaluate ``expression`` with ``values`` (name -> scalar or column) and return a boolean (or numeric) array. ``functions``: the names it may call (default :data:`FUNCTIONS`). |
| `check` | função | `(expression, names, functions=(), strings=False) -> None` | Raise :class:`ExpressionError` unless ``expression`` uses only the grammar above, the variables ``names`` and calls of ``functions`` (and no string literals unless ``strings``). |
| `names_in_tree` | função | `(tree) -> set[str]` |  |

#### `pinneapple_data.formfill`

Fill engineering forms from the documents that hold the data: read datasheets and specifications, find every item a form asks for with its source, report conflicts between documents and missing data, and fill the fillable PDF.

<small>`pinneapple_data/formfill/__init__.py` · 28 linhas · testes: 44</small>

#### `pinneapple_data.formfill.extract`

Read datasheets and specifications, find each answer a form asks for, and keep where every value came from.

<small>`pinneapple_data/formfill/extract.py` · 698 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Document` | classe | `(name, pages, tables, role, has_text, ocr_pages, ocr_conf, unread_pages)` |  |
| `Candidate` | classe | `(key, value, raw, doc, page, snippet, method, confidence, unit, si, gauge, ocr)` |  |
| `Decision` | classe | `(key, label, section, status, value, display, chosen, candidates, required, note)` |  |
| `Compilation` | classe | `(decisions, tables, documents, spec)` |  |
| `read_document` | função | `(source, name='', role='', ocr='auto', max_ocr_pages=30) -> Document` | A PDF (path or bytes) or plain text (``.txt``/``.md`` path, or bytes that are not a PDF). |
| `parse_value` | função | `(f, text) -> Optional[Dict[str, Any]]` | Interpret ``text`` as an answer to ``f``; ``None`` when it does not look like one. |
| `extract_rules` | função | `(doc, spec) -> List[Candidate]` | Candidates for every item of ``spec`` whose label appears in ``doc`` (tables first, then text lines), plus the rows of the spec's tables. |
| `check_answer` | função | `(f, value) -> Any` | An engineer's answer for ``f``, validated (``ValueError`` names the accepted options). |
| `compile` | função | `(docs, candidates, spec, overrides=None) -> Compilation` | One decision per item of ``spec``. Documents earlier in ``docs`` win conflicts (order = priority); the conflict is still reported with every source. ``overrides`` (an engineer's answers) win over everything and are marked as such. |

#### `pinneapple_data.formfill.fill`

Write a compilation into the user's copy of a fillable PDF form.

<small>`pinneapple_data/formfill/fill.py` · 177 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `pdf_widgets` | função | `(pdf) -> List[Dict[str, Any]]` | Every form widget of a PDF: qualified name, type (``/Tx``, ``/Btn``...), page, tooltip, button states and whether it belongs to a radio group. |
| `form_values` | função | `(comp, spec=None) -> Tuple[Dict[str, str], Dict[str, str]]` | ``(text fields, button states)`` for the PDF, from the decisions and the table rows. |
| `fill_pdf` | função | `(blank_pdf, comp, spec=None, extra_text=None) -> bytes` | The filled form as PDF bytes. ``blank_pdf`` is the user's copy of the fillable form ``spec`` describes. |
| `read_filled` | função | `(pdf) -> Dict[str, Any]` | Field name -> value of a filled form (for checks and round trips). |

#### `pinneapple_data.formfill.llm`

Optional extraction with a local LLM served by Ollama, for documents the label rules cannot read (free-form specifications, odd layouts, forms whose items have no usual label).

<small>`pinneapple_data/formfill/llm.py` · 230 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OllamaError` | classe | `()` |  |
| `OllamaClient` | classe | `(url=None, model=None, timeout=600.0, num_ctx=16384)` | Minimal client for the Ollama HTTP API (``/api/chat``, ``/api/tags``). |
| `document_text` | função | `(doc, pages=None) -> str` | Page-marked text of a document, its tables written as ``cell \| cell`` rows. |
| `verify_quote` | função | `(doc, quote, page) -> Optional[int]` | Page (1-based) where ``quote`` occurs in the document text, preferring the stated page; ``None`` if absent. |
| `extract_with_llm` | função | `(doc, spec, client=None, max_chars=24000) -> Tuple[List[Candidate], List[Dict[str, Any]]]` | ``(accepted candidates, rejected answers with the reason)`` for one document. |

#### `pinneapple_data.formfill.ocr`

OCR for scanned pages (pages without a text layer), with Tesseract.

<small>`pinneapple_data/formfill/ocr.py` · 241 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `OcrPage` | classe | `(lines, tables, confidence, word_conf, angle)` |  |
| `ocr_available` | função | `() -> bool` |  |
| `ocr_image` | função | `(img) -> OcrPage` | OCR one page image (PIL). |
| `ocr_pdf_page` | função | `(pdf_bytes, index, dpi=DPI) -> OcrPage` | OCR page ``index`` (0-based) of a PDF. |

#### `pinneapple_data.formfill.pipeline`

One call from source documents to a compiled form, and the helpers a front end needs around it.

<small>`pinneapple_data/formfill/pipeline.py` · 66 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `compile_documents` | função | `(docs, spec, client=None, overrides=None, max_ocr_pages=None) -> Tuple[Compilation, List[…)` | Read every ``(name, data, role)`` document, extract candidates with the rules of ``spec`` (and with the local LLM ``client`` too, when given) and compile them into one decision per form item. |
| `check_overrides` | função | `(spec, raw) -> Dict[str, Any]` | User answers ``{item key: value}`` checked against ``spec`` (:func:`check_answer`): empty values are dropped, an unknown key or an invalid value raises ``ValueError`` with a message for the user. |
| `output_filename` | função | `(comp, prefix='', day=None) -> str` | File name for the filled form: ``<spec id>_<value of spec.filename_key or "filled">_<YYYYMMDD>.pdf``, made of safe characters only. |

#### `pinneapple_data.formfill.report`

The compiled datasheet as a PDF (needs reportlab): every item of the spec with its value and source, the open items (missing required data, conflicts between documents) first, side-by-side columns where the spec has them (shell side | tube side), the table rows and the documents read.

<small>`pinneapple_data/formfill/report.py` · 166 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `render_datasheet` | função | `(comp, spec=None, title=None) -> bytes` |  |

#### `pinneapple_data.formfill.spec`

What a form asks for, how documents phrase each item, and where each answer goes in the fillable PDF.

<small>`pinneapple_data/formfill/spec.py` · 259 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Field` | classe | `(key, label, section, kind, pdf, states, patterns, quantity, required, source_hint, optio…)` |  |
| `TableSpec` | classe | `(key, label, columns, header, combine, defaults, pdf_rows, description)` | A repeated-row item (nozzle schedule, line list...): found as a document table whose header matches ``header`` (every regex must match some header cell), filled into ``pdf_rows`` (one dict column -> PDF field per row). |
| `FormSpec` | classe | `(id, title, fields, sections, tables, description, notes_field, left_blank, filename_key,…)` |  |

#### `pinneapple_data.formfill.specs`

Built-in form templates. Add one by writing a module that builds a ``FormSpec`` and registering it here, or ship it as JSON and load it with ``FormSpec.load``.

<small>`pinneapple_data/formfill/specs/__init__.py` · 41 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `get_spec` | função | `(spec_id) -> FormSpec` |  |
| `list_specs` | função | `() -> List[Dict[str, object]]` |  |

#### `pinneapple_data.formfill.specs.asme_udr1`

Built-in template: ASME BPVC Section VIII Division 1 Form U-DR-1, and where each answer goes in its fillable PDF.

<small>`pinneapple_data/formfill/specs/asme_udr1.py` · 326 linhas · **sem teste direto**</small>

#### `pinneapple_data.formfill.specs.psv`

Built-in template: pressure relief valve datasheet, with the data API 520 Part I (sizing) and API 526 (flanged steel valves) ask for. The output is a compiled datasheet generated by the tool, not a reproduction of a standard's form.

<small>`pinneapple_data/formfill/specs/psv.py` · 115 linhas · **sem teste direto**</small>

#### `pinneapple_data.formfill.specs.pump`

Built-in template: centrifugal pump datasheet, with the data API 610 asks the purchaser for. The output is a compiled datasheet generated by the tool, not a reproduction of the standard's data sheet.

<small>`pinneapple_data/formfill/specs/pump.py` · 110 linhas · **sem teste direto**</small>

#### `pinneapple_data.formfill.specs.shell_tube`

Built-in template: shell-and-tube heat exchanger datasheet, with the data a TEMA specification sheet and API 660 ask for. The output is a compiled datasheet generated by the tool, not a reproduction of a standard's form.

<small>`pinneapple_data/formfill/specs/shell_tube.py` · 110 linhas · **sem teste direto**</small>

#### `pinneapple_data.formfill.specs.tank`

Built-in template: atmospheric storage tank datasheet, with the data API 650 asks the purchaser for (Annex L lists them). The output is a compiled datasheet generated by the tool, not a reproduction of the standard's data sheet.

<small>`pinneapple_data/formfill/specs/tank.py` · 117 linhas · **sem teste direto**</small>

#### `pinneapple_data.geometry_features`

Geometry featurization: surface point-cloud sampling and a coarse signed-distance-field (SDF) grid from a triangle mesh — generic inputs (vertices/faces arrays, matching this package's own `stl_import.STLMesh` convention), no file-format or storage coupling. Useful as a preprocessing step for any …

<small>`pinneapple_data/geometry_features.py` · 86 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `sample_point_cloud` | função | `(vertices, faces, n_samples=2048) -> np.ndarray` | Sample `n_samples` points uniformly (area-weighted) from a mesh's surface. `n_samples` is silently capped at 3x the face count for a very coarse mesh, since sampling far more points than the mesh has resolution to support doesn't add real information. |
| `compute_sdf_grid` | função | `(vertices, faces, grid_resolution=16, bounds=None) -> Dict[str, Any]` | Evaluate a signed distance field on a regular grid spanning the mesh's bounding box (or a caller-supplied `bounds` = [[xmin,ymin,zmin], [xmax,ymax,zmax]], e.g. to featurize several meshes on one shared grid). |
| `featurize_mesh` | função | `(vertices, faces, n_point_cloud_samples=2048, sdf_grid_resolution=16) -> Dict[str, Any]` | Convenience wrapper: point cloud + SDF grid in one call, with per-stage errors reported rather than raised (a degenerate/non-watertight mesh may still yield a usable point cloud even if the SDF query fails). |

#### `pinneapple_data.grf_sampler`

Gaussian Random Field (GRF) input-function sampler, the standard way to draw random forcing/coefficient functions u(x) for operator-learning benchmarks (e.g. Lu, Jin & Karniadakis 2019, "DeepONet", arXiv:1910.03193, which draws every one of its input functions from a zero-mean GRF with an …

<small>`pinneapple_data/grf_sampler.py` · 177 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GRFDraw` | classe | `(x_grid, u_grid, spline)` | One sampled function u(x), x in [x_min, x_max]. |
| `GRFDraw2D` | classe | `(x_grid, y_grid, u_grid, interp)` | One sampled 2D field u(x,y), (x,y) in [0,1]x[0,1] (normalized). |
| `sample_grf_batch` | função | `(n_draws, x_min=0.0, x_max=1.0, n_grid=1000, length_scale=0.2, jitter=1e-10, seed=None) -…)` | Draw n_draws independent zero-mean GRF paths sharing ONE Cholesky factorization of the RBF covariance on a common fine grid. |
| `sample_grf_paths` | função | `(n_draws, x_min=0.0, x_max=1.0, n_grid=1000, length_scale=0.2, jitter=1e-10, seed=None) -…)` | Alias for sample_grf_batch (kept for readability at call sites that draw a single small batch rather than a full dataset-generation pass). |
| `sample_grf_2d_batch` | função | `(n_draws, nx=64, ny=64, length_scale=0.2, seed=None) -> List[GRFDraw2D]` | Draw n_draws independent zero-mean, unit-variance 2D GRF paths on the normalized [0,1]x[0,1] domain via FFT spectral synthesis (see module docstring above). One (nx,ny) white-noise draw + one 2D FFT + one inverse 2D FFT per sample — no shared factorization to amortize across draws the way … |

#### `pinneapple_data.lineage`

Engineering model lineage (digital thread): artifacts and derivations, checks, questions, W3C PROV export, and auto-detection of the lineage from a project's files.

<small>`pinneapple_data/lineage/__init__.py` · 7 linhas · testes: 13</small>

#### `pinneapple_data.lineage.detect`

Auto-detect a project's lineage from its files: OpenFOAM cases (blockMeshDict → mesh → run), Gmsh .geo → .msh, CalculiX decks and results, datasets exported by the Interoperability Hub (linked to their source files by sha256), Comparator / Mesh Quality / Preflight reports, PINNeAPPle model cards, …

<small>`pinneapple_data/lineage/detect.py` · 304 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `detect_project` | função | `(fs, project='') -> Lineage` |  |

#### `pinneapple_data.lineage.graph`

Engineering model lineage: artifacts (CAD, mesh, simulation, dataset, model, prediction, ...) linked by "derived from" edges into a digital thread, with the checks that make it trustworthy and the questions it answers.

<small>`pinneapple_data/lineage/graph.py` · 365 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Artifact` | classe | `(id, kind, name, file, version, software, parameters, timestamp, hash, owner, origin, inp…)` |  |
| `Lineage` | classe | `(artifacts=(), project='')` |  |
| `sha256` | função | `(b) -> str` |  |

#### `pinneapple_data.noether_dataset`

Noether ↔ PINNeAPPle data bridge.

<small>`pinneapple_data/noether_dataset.py` · 216 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NoetherDatasetBridge` | classe | `(noether_dataset, input_fields=None, target_fields=None, transform=None, device=None) -> …)` | Bridge a ``noether.data.base.Dataset`` to a plain PyTorch Dataset. |
| `PhysicalSampleDataset` | classe | `(samples, input_fields=None, output_fields=None, include_coords=False, transform=None) ->…)` | Convert a list of PINNeAPPle ``PhysicalSample`` (UPD) to a Noether-compatible ``{field: tensor}`` PyTorch Dataset. |
| `split_physical_samples` | função | `(samples, val_fraction=0.1, seed=42, **dataset_kwargs)` | Split a list of PhysicalSamples into train and validation datasets. |

#### `pinneapple_data.parameter_sampling`

Design-of-experiments parameter-space sampling.

<small>`pinneapple_data/parameter_sampling.py` · 118 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `sample_parameters` | função | `(param_ranges, n_samples, method='lhs', seed=0) -> list[dict[str, float]]` | Sample `n_samples` points from a bounded parameter space. |
| `saltelli_perturbation_sweep` | função | `(nominal, param_ranges, n_base=8, seed=0) -> list[dict[str, float]]` | Build a Saltelli-style sample set for first-order/total-order Sobol global sensitivity analysis, without requiring the SALib dependency. |

#### `pinneapple_data.physical_sample`

PhysicalSample dataclass and training bridge for UPD-aligned physical data.

<small>`pinneapple_data/physical_sample.py` · 241 linhas · testes: 28</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicalSample` | classe | `(state, geometry, schema, domain, provenance, extras)` | Unified Physical Sample (UPD-aligned) used across Pinneapple. |

#### `pinneapple_data.physical_units`

Physical units and quantities for engineering data (PINNeAPPle Physical Data Layer).

<small>`pinneapple_data/physical_units.py` · 436 linhas · testes: 52</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Unit` | classe | `(symbol, factor, dim, offset, gauge)` | A unit: ``SI value = value * factor + offset``. |
| `UnitError` | classe | `()` |  |
| `Quantity` | classe | `(name, si_unit, dim, keywords, hard, typical, display)` |  |
| `QuantityGuess` | classe | `(quantity, confidence, reason, alternatives, is_difference)` |  |
| `parse_unit` | função | `(text) -> Unit` | Parse a unit expression: ``°C``, ``kg/m^3``, ``W/(m·K)``, ``m³/h``, ``N*m``, ``psig``. |
| `try_parse_unit` | função | `(text) -> Optional[Unit]` |  |
| `convert` | função | `(values, src, dst, difference=False)` | Convert values (scalar or numpy array) from ``src`` to ``dst``. |
| `dim_str` | função | `(dim) -> str` |  |
| `split_header` | função | `(header) -> Tuple[str, Optional[str]]` | Split a column header into (name, unit text). ``"T_supply [°C]"`` -> ``("T_supply", "°C")``; also understands ``Pressure (bar)``, ``flow, m3/h``, ``power in kW`` and suffixes like ``T_inlet_degC``, ``pressure_bar`` or ``vibration_mm_s``. Returns ``(header, None)`` when no unit is found. A short … |
| `infer_quantity` | função | `(name, unit=None) -> QuantityGuess` | Infer the physical quantity of a column from its unit dimension and name keywords. |
| `describe_unit` | função | `(u) -> Dict[str, object]` |  |
| `si_unit_for` | função | `(quantity) -> str` |  |

#### `pinneapple_data.physics_case`

``PhysicsCase`` -- a minimal bridge between the three disjoint "partial canonical representations" of a physics problem already in this repo:

<small>`pinneapple_data/physics_case.py` · 405 linhas · testes: 47</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BenchmarkComparison` | classe | `(benchmark, x_vars, y_vars, n_reference_points, n_compared_points, rmse, relative_l2_erro…)` | Plain numeric comparison produced by ``PhysicsCase.validate_against_benchmark()``. |
| `PhysicsCase` | classe | `(name, geometry, geometry_kind, physics, solver, solver_config, results, reference_benchm…)` | One physics case: geometry + physics spec + solver choice + results, referenced together. |

#### `pinneapple_data.prefetch_adaptive`

Adaptive prefetch IterableDataset for Zarr-backed UPD stores with configurable queue control.

<small>`pinneapple_data/prefetch_adaptive.py` · 281 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AdaptivePrefetchConfig` | classe | `(queue_max, min_target_fill, max_target_fill, target_fill_init, control_period_s, increas…)` | Configuration for adaptive prefetching behavior. |
| `AdaptivePrefetchZarrUPDIterable` | classe | `(root, *, fields=None, coords=None, dtype=None, start=0, end=None, stride=1, sample_ctor=…)` | IterableDataset that streams samples from a cached Zarr-backed UPD store with adaptive prefetching. |

#### `pinneapple_data.preflight`

Simulation pre-flight: check an OpenFOAM case or a CalculiX/Abaqus deck before spending solver hours on it.

<small>`pinneapple_data/preflight/__init__.py` · 74 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `preflight` | função | `(fs) -> Dict[str, Any]` |  |
| `is_result_file` | função | `(path) -> bool` | A solver result rather than an input: inside an OpenFOAM time directory other than ``0`` (``0.5/U``, ``1e-3/p``) or a CalculiX / VTK result file. Not needed for a pre-flight; drop such files before checking. |
| `drop_results` | função | `(fs) -> Dict[str, bytes]` | ``fs`` without the solver result files (:func:`is_result_file`). |
| `attach_snippets` | função | `(report, fs, context=4, width=160) -> None` | Add to every finding that names a file and line the source lines around it, in place: ``finding["snippet"] = {"start": first line number, "lines": [...], "at": line}`` (``context`` lines on each side, each cut to ``width`` characters). File names are resolved against the case root of the report. |

#### `pinneapple_data.preflight.calculix`

Pre-flight rules for CalculiX (and flat Abaqus) input decks: mesh, sections and materials, the properties each procedure needs, unit consistency, supports and loads, initial conditions and step settings.

<small>`pinneapple_data/preflight/calculix.py` · 450 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check` | função | `(fs, main) -> Report` |  |

#### `pinneapple_data.preflight.common`

Findings and the pre-flight report shared by the OpenFOAM and CalculiX rule sets.

<small>`pinneapple_data/preflight/common.py` · 83 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Finding` | classe | `(section, status, title, detail, fix, file, line, entity, rule, evidence)` |  |
| `Report` | classe | `(solver, findings, info)` |  |
| `line_of` | função | `(text, pattern, flags=re.M) -> Optional[int]` |  |
| `num` | função | `(v) -> Optional[float]` | Last number in an OpenFOAM entry ('nu [0 2 -1 0 0 0 0] 1e-05' -> 1e-05, 'uniform 0' -> 0). |
| `vec` | função | `(v) -> Optional[List[float]]` |  |

#### `pinneapple_data.preflight.openfoam`

Pre-flight rules for OpenFOAM cases: mesh, transport/thermo properties, boundary and initial conditions, schemes, linear solvers and time stepping, checked before the solver runs.

<small>`pinneapple_data/preflight/openfoam.py` · 551 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check` | função | `(fs, root='') -> Report` |  |

#### `pinneapple_data.process_optimizer`

Data-driven process optimization for industrial plants (plain machine learning, no physics model).

<small>`pinneapple_data/process_optimizer.py` · 436 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `suggest_roles` | função | `(columns) -> Dict[str, Any]` | Guess target / levers / context from column names (to be reviewed by a person). |
| `clean_for_modeling` | função | `(df, report) -> Tuple[pd.DataFrame, List[str]]` | Time-sorted numeric table with the problems found by the health report set to missing. |
| `time_series_cv` | função | `(X, y, n_folds=4, l2_grid=L2_GRID, seed=0) -> Dict[str, Any]` | Forward-chaining cross-validation on the training period: fold k trains on the first k blocks and is scored on the next one (never on the past), for every L2 regularisation strength in ``l2_grid``. Picks the strength with the lowest mean validation MAE; the per-fold errors show how stable the model … |
| `fit_and_optimize` | função | `(data, target, levers, context, goal='minimize', constraints=(), bounds=None, max_move=0.…)` | Train on the older part of the record, evaluate and optimize on the newest part. The L2 regularisation of the KPI model is chosen by forward-chaining cross-validation inside the training part (``cv_folds``; 0 keeps the default) and the fold errors go in the report under ``model.cv``. |
| `plant_physics` | função | `(oat, rh, load, chws_sp, cws_sp, pump_pct)` | True plant behaviour used by the example (not visible to the ML model). |
| `example_plant_operations` | função | `(seed=11, days=60) -> Tuple[pd.DataFrame, Dict[str, Any]]` | 60 days of 15-minute data from a chilled-water plant run by operators with manual setpoint habits. |
| `true_saving` | função | `(result) -> Dict[str, Any]` | For the example: score the recommendations with the plant's true physics. |

#### `pinneapple_data.serialization`

Serialization utilities for PhysicalSample to/from PT, Zarr, and HDF5 formats.

<small>`pinneapple_data/serialization.py` · 338 linhas · testes: 11</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `save_manifest` | função | `(path, manifest) -> None` |  |
| `load_manifest` | função | `(path) -> Dict[str, Any]` |  |
| `save_pt` | função | `(samples, path, *, compress=False) -> None` | Save a sequence of PhysicalSamples to a PyTorch (.pt) file. |
| `load_pt` | função | `(path) -> List[PhysicalSample]` | Load PhysicalSamples from a PyTorch (.pt) file. |
| `save_hdf5` | função | `(samples, path) -> None` | Save PhysicalSamples to an HDF5 file. |
| `save_hdf5` | função | `(samples, path) -> None` |  |
| `load_hdf5` | função | `(path) -> List[PhysicalSample]` |  |
| `save_zarr` | função | `(samples, root, *, compressor='default') -> None` | Save PhysicalSamples to a Zarr directory store using UPDZarrStore. |
| `load_zarr` | função | `(root)` | Load PhysicalSamples from a Zarr directory store using UPDZarrStore. |

#### `pinneapple_data.simulation_metadata`

Simulation metadata extraction (PINNeAPPle): solver, mesh, time step, boundary conditions, parameters and convergence from the files a simulation leaves behind.

<small>`pinneapple_data/simulation_metadata/__init__.py` · 346 linhas · testes: 25</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `read_upload` | função | `(files) -> dict[str, bytes]` | Turn uploaded files (zips are expanded, with size limits) into {relative path: bytes}. |
| `parse_residual_table` | função | `(text) -> dict[str, Any] \| None` |  |
| `extract` | função | `(fs) -> dict[str, Any]` |  |

#### `pinneapple_data.simulation_metadata.calculix`

CalculiX / Abaqus input decks (.inp) and CalculiX run files (.sta, .cvg, .dat, solver output).

<small>`pinneapple_data/simulation_metadata/calculix.py` · 323 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `parse_inp` | função | `(text, include=None) -> Dict[str, Any]` | Parse an input deck into model / steps / materials / BCs / loads. ``include`` maps file names to the text of *INCLUDE'd files that were uploaded too. |
| `parse_sta` | função | `(text) -> List[Dict[str, Any]]` |  |
| `parse_cvg` | função | `(text) -> List[Dict[str, Any]]` |  |
| `parse_dat` | função | `(text) -> Dict[str, Any]` | Largest printed displacement (or min/max of other printed results) at the last time of each set. |
| `parse_ccx_log` | função | `(text) -> Dict[str, Any]` |  |
| `assess` | função | `(sta, cvg, deck, log) -> Dict[str, Any]` |  |

#### `pinneapple_data.simulation_metadata.convergence`

Convergence assessment from residual histories: a verdict, the reasons and what to do next.

<small>`pinneapple_data/simulation_metadata/convergence.py` · 228 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `downsample` | função | `(x, series, k=1500) -> Dict[str, Any]` |  |
| `assess_steady` | função | `(log, targets) -> Dict[str, Any]` |  |
| `assess_transient` | função | `(log, tolerances, max_co, algorithm, reached_end) -> Dict[str, Any]` |  |
| `assess_table` | função | `(iters, residuals, is_log10) -> Dict[str, Any]` | Generic residual history (Fluent / STAR-CCM+ / SU2 exports). |

#### `pinneapple_data.simulation_metadata.foam_dict`

A tolerant parser for OpenFOAM dictionary files (controlDict, fvSchemes, fvSolution, 0/U, ...).

<small>`pinneapple_data/simulation_metadata/foam_dict.py` · 176 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `parse` | função | `(text, max_list=2000) -> Dict[str, Any]` | Parse a FoamFile. Very long lists (mesh points, nonuniform fields) are summarised as ``"<list of N>"`` beyond ``max_list`` entries to keep the result small. |
| `header` | função | `(text) -> Dict[str, str]` | The FoamFile header (class, object, format, note) without parsing the body. |
| `scalar` | função | `(v) -> Optional[float]` | Last number in an entry: 'nu [0 2 -1 0 0 0 0] 1e-05' -> 1e-05; 'uniform 0' -> 0.0. |
| `vector` | função | `(v) -> Optional[List[float]]` | 'uniform (10 0 0)' -> [10, 0, 0]. |
| `words` | função | `(v) -> List[str]` |  |
| `boundary` | função | `(text) -> List[Dict[str, Any]]` | Patches of a polyMesh/boundary file: name, type, nFaces, startFace, inGroups. |

#### `pinneapple_data.simulation_metadata.openfoam`

OpenFOAM case + log extraction for the simulation metadata record.

<small>`pinneapple_data/simulation_metadata/openfoam.py` · 523 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `parse_log` | função | `(text) -> dict[str, Any]` | Parse an OpenFOAM solver log (or blockMesh / checkMesh log). |
| `find_root` | função | `(fs) -> str \| None` | Case root: the directory that holds system/controlDict (shortest such path). |
| `parse_dat` | função | `(text) -> tuple[list[str], np.ndarray]` | Parse an OpenFOAM function-object table (forceCoeffs.dat, forces.dat, probes, residuals.dat, ...). |
| `read_dat` | função | `(path) -> tuple[list[str], np.ndarray]` | :func:`parse_dat` of the file at ``path`` (an OpenFOAM ``postProcessing/<name>/<time>/<table>.dat``). |
| `extract_case` | função | `(fs, root) -> dict[str, Any]` |  |
| `summarise` | função | `(case) -> dict[str, Any]` | Turn the raw case extraction into the record sections. |

#### `pinneapple_data.snapshots`

Snapshot matrices for reduced-order models: many results of one model (time steps or parameter points) on one mesh, stacked into ``X`` with one row per snapshot.

<small>`pinneapple_data/snapshots.py` · 413 linhas · testes: 31</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SnapshotSet` | classe | `(X, fields, times, params, param_names, locations, source, meta, cells, snapshot_points)` | ``X`` (n_snapshots, n_values) plus what each row and column is. |
| `snapshots_from_arrays` | função | `(arrays, *, times=None, params=None, param_names=None, locations=None, where=None, source…)` | Build a set from field arrays with the snapshots on the first axis: (n, m) scalars or (n, m, c) vectors (more axes are flattened into m). ``cells``: (VTK type, connectivity) of the point mesh; ``snapshot_points``: (n, m, 3) node coordinates per snapshot. |
| `load_snapshots` | função | `(files, *, fields=None, param_columns=None) -> SnapshotSet` | Read an upload into a :class:`SnapshotSet` (see the module docstring for the formats). |

#### `pinneapple_data.splits`

Train/val/test split strategies (random, time, group).

<small>`pinneapple_data/splits.py` · 104 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SplitSpec` | classe | `(method, train, val, test, seed)` | Split strategies: - random: shuffle then split by ratios - time: keep order; split by ratios (train earliest, test latest) - group: ensure groups don't leak (requires group_ids) |
| `split_indices` | função | `(n, spec, *, group_ids=None) -> Dict[str, np.ndarray]` |  |

#### `pinneapple_data.stl_import`

STL import utilities (ASCII + binary).

<small>`pinneapple_data/stl_import.py` · 93 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `STLMesh` | classe | `(verts, faces)` |  |
| `load_stl_bytes` | função | `(data, *, dedup_eps=1e-06) -> STLMesh` |  |
| `load_stl` | função | `(path, *, dedup_eps=1e-06) -> STLMesh` |  |

#### `pinneapple_data.synth`

_Módulo sem docstring e sem texto._

<small>`pinneapple_data/synth/__init__.py` · 38 linhas · **sem teste direto**</small>

#### `pinneapple_data.synth.base`

Base SynthConfig, SynthOutput, and SynthGenerator protocol for synthetic data generation.

<small>`pinneapple_data/synth/base.py` · 66 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SynthConfig` | classe | `(seed, device, dtype)` | Configuration container for synthetic data generation. |
| `SynthOutput` | classe | `(samples, extras)` | Standard output structure for synthetic generators. |
| `SynthGenerator` | classe | `()` | Protocol defining the interface for synthetic data generators. |

#### `pinneapple_data.synth.curvefit`

Curve-fit synthetic generator for polynomial ridge regression and sample synthesis.

<small>`pinneapple_data/synth/curvefit.py` · 153 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CurveFitSynthGenerator` | classe | `(cfg=None)` | Synthetic generator that fits a polynomial trend to 1D data and synthesizes samples. |

#### `pinneapple_data.synth.distributions`

Distribution-based synthetic generator for Gaussian, uniform, and mixture-of-Gaussians sampling.

<small>`pinneapple_data/synth/distributions.py` · 154 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DistributionSynthGenerator` | classe | `(cfg=None)` | Synthetic generator for sampling inputs from simple probability distributions. |

#### `pinneapple_data.synth.geometry`

Geometry synthesis and mesh augmentation from trimesh-like inputs.

<small>`pinneapple_data/synth/geometry.py` · 221 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GeometrySynthGenerator` | classe | `(cfg=None)` | Geometry synthesis and augmentation from an existing triangle mesh. |

#### `pinneapple_data.synth.geometry_cadquery`

CadQuery-based geometry synthesis: parametric templates and STL-driven variants.

<small>`pinneapple_data/synth/geometry_cadquery.py` · 532 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ParametricCadQuerySynthGenerator` | classe | `(cfg=None)` | Generate geometry variants from a parametric CadQuery template. |
| `STLTemplateSynthGenerator` | classe | `(cfg=None)` | Synthetic geometry generator driven by an existing STL/mesh template. |

#### `pinneapple_data.synth.images`

Image reconstruction and inpainting synthetic generator.

<small>`pinneapple_data/synth/images.py` · 262 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ImageReconstructionSynthGenerator` | classe | `(cfg=None)` | Synthetic generator for image reconstruction tasks. |

#### `pinneapple_data.synth.operators`

_Módulo sem docstring e sem texto._

<small>`pinneapple_data/synth/operators/__init__.py` · 6 linhas · **sem teste direto**</small>

#### `pinneapple_data.synth.operators.masks`

Random block mask generation for 2D grids.

<small>`pinneapple_data/synth/operators/masks.py` · 38 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `random_block_mask` | função | `(H, W, frac=0.2, seed=42) -> torch.Tensor` | Generate a random rectangular block mask within a 2D grid. |

#### `pinneapple_data.synth.operators.noise`

Gaussian noise addition for synthetic data augmentation.

<small>`pinneapple_data/synth/operators/noise.py` · 28 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `add_gaussian_noise` | função | `(x, std=0.01, seed=42) -> torch.Tensor` | Add Gaussian noise to a tensor. |

#### `pinneapple_data.synth.operators.transforms`

Tensor normalization transforms for synth operators.

<small>`pinneapple_data/synth/operators/transforms.py` · 28 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `normalize_01` | função | `(x, eps=1e-08) -> torch.Tensor` | Normalize a tensor to the [0, 1] range. |

#### `pinneapple_data.synth.pde`

PDE/ODE synthetic generator for heat, advection, and logistic equations.

<small>`pinneapple_data/synth/pde.py` · 254 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SimplePhysicalSample` | classe | `(fields, coords, meta)` | MVP duck-typed PhysicalSample to keep synth module usable standalone. |
| `PDESynthGenerator` | classe | `(cfg=None)` | PDE/ODE synthetic generator. |

#### `pinneapple_data.synth.pde_symbolic`

Symbolic PDE parser and finite-difference-based synthetic trajectory generator.

<small>`pinneapple_data/synth/pde_symbolic.py` · 498 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SymbolicFDSynthGenerator` | classe | `(cfg=None)` | Symbolic equation -> automatic finite difference operator -> synthetic dataset. |
| `make_fd_residual_fn` | função | `(equation, *, parameters=None, bc='periodic')` | Create a residual evaluator using finite-difference approximations. |

#### `pinneapple_data.synth.registry`

SynthCatalog factory for building synthetic generators by name from a registry.

<small>`pinneapple_data/synth/registry.py` · 73 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SynthCatalog` | classe | `(registry)` | Factory-style catalog for synthetic data generators. |

#### `pinneapple_data.synth.sample_adapter`

PhysicalSample-like conversion and torchify utilities for synth module outputs.

<small>`pinneapple_data/synth/sample_adapter.py` · 192 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicalSampleLike` | classe | `(fields, coords, meta)` | Protocol defining the minimal interface of a PhysicalSample-like object. |
| `has_pinnego_physical_sample` | função | `() -> bool` | Check whether `pinneapple_data.physical_sample.PhysicalSample` is importable. |
| `to_physical_sample` | função | `(sample_like, *, device=None, dtype=None) -> Any` | Best-effort conversion: - if already a PhysicalSample -> return - if duck-typed {fields, coords, meta} -> convert - else if dict with keys -> interpret |

#### `pinneapple_data.transforms`

Data transforms and normalizers for preprocessing pipelines.

<small>`pinneapple_data/transforms.py` · 140 linhas · testes: 2 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Normalizer` | classe | `()` | Abstract base for all normalizers (fit / transform / inverse). |
| `StandardScaler` | classe | `(eps, mean_, std_)` | Zero-mean unit-variance scaler. |
| `MinMaxScaler` | classe | `(eps, min_, max_)` | Scales each feature to [0, 1]. |

#### `pinneapple_data.udr1`

ASME Form U-DR-1 shortcuts over the generic ``pinneapple_data.formfill`` engine (template ``asme_u-dr-1``).

<small>`pinneapple_data/udr1/__init__.py` · 30 linhas · testes: 36</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `extract_rules` | função | `(doc) -> List[Candidate]` |  |
| `compile_requirements` | função | `(docs, candidates, overrides=None) -> Compilation` |  |
| `fill_form` | função | `(blank_pdf, comp) -> bytes` |  |

#### `pinneapple_data.unified`

Engineering data standardizer: many files, many conventions -> one physical schema (PINNeAPPle UPD).

<small>`pinneapple_data/unified.py` · 584 linhas · testes: 24</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `canonical_name` | função | `(base, quantity) -> str` | 'CHW Supply Temp' + temperature -> 'chw_supply_temperature'; 'chws_temp' -> same. |
| `conversion_text` | função | `(src, dst, difference=False) -> str` | Human-readable conversion, e.g. '°F → °C: (x − 32) × 0.5556'. |
| `propose_mapping` | função | `(df, source, system='metric', time_column=None, units=None, tz=None) -> Dict[str, Any]` | Draft the mapping of one source from its headers (to be reviewed by a person). |
| `make_recipe` | função | `(mappings, system='metric', resample='auto', layout='wide', agg='mean') -> Dict[str, Any]` |  |
| `standardize` | função | `(sources, recipe) -> Dict[str, Any]` | Apply a recipe to sources [(file name, dataframe, raw bytes)]. Returns a dict with ``wide`` and ``long`` DataFrames, the ``manifest``, per-source notes and cross-source ``agreement``. |
| `write` | função | `(result, fmt='csv', layout='wide') -> Tuple[bytes, str, str]` | Serialize a result. Returns (bytes, media type, file name). |
| `to_physical_sample` | função | `(result)` | The wide table as a PINNeAPPle :class:`~pinneapple_data.physical_sample.PhysicalSample` (xarray Dataset on a ``time`` coordinate, units in attributes). |
| `example_sources` | função | `(seed=3) -> List[Tuple[str, bytes]]` | The same chilled-water plant recorded by a building management system (CSV, local time, °C, semicolon + decimal comma), a SCADA REST API (JSON, Unix ms, °F, gpm, tons) and a test rig (Excel, ISO UTC, K, L/s, W). |
| `before_after_series` | função | `(result, k=500) -> Dict[str, Any]` | Per variable of a :func:`standardize` result: the raw values of every source (in their original units) and the standardized values, on the common UTC time axis, downsampled to at most ``k`` evenly spaced points. |

#### `pinneapple_data.upd_dataset`

Hardened UPDDataset wrapper with validation, persistence, and save/load support.

<small>`pinneapple_data/upd_dataset.py` · 192 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `UPDDataset` | classe | `(samples, manifest)` | Hardened dataset wrapper. |

#### `pinneapple_data.upd_types`

Pure data-structure types for the UPD (Unified Physical Data) pipeline.

<small>`pinneapple_data/upd_types.py` · 103 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `UPDItem` | classe | `(zarr_path, meta_path)` | One UPD shard item: points to a Zarr store and a JSON metadata file. |
| `ConditionSpec` | classe | `(name, type, equation, n, options)` | Sampling definition for a single PINN condition. |
| `SamplingSpec` | classe | `(n_collocation, conditions, n_data, replace, seed)` | Overall sampling plan for a UPD shard. |
| `Batch` | classe | `(collocation, conditions, data)` | PINN-ready training batch. |
| `seconds_since` | função | `(t, t0=None) -> np.ndarray` | Convert a ``datetime64`` array to float seconds elapsed since *t0*. |

#### `pinneapple_data.validators`

PhysicalSample validation: units, ranges, non-negativity, and monotonic coordinate checks.

<small>`pinneapple_data/validators.py` · 161 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ValidationIssue` | classe | `(level, message, field)` | Single validation issue from validate_physical_sample. |
| `validate_physical_sample` | função | `(sample, *, units_policy='warn', required_units=None, ranges=None, non_negative=None, mon…)` | Validate a PhysicalSample and return a list of issues. |
| `assert_valid_physical_sample` | função | `(*args, **kwargs) -> None` | Validate a PhysicalSample and raise ValueError if any errors are found. |

#### `pinneapple_data.zarr_cached_store`

Cached wrapper around UPDZarrStore with LRU sample and field-level caching.

<small>`pinneapple_data/zarr_cached_store.py` · 231 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ZarrCacheConfig` | classe | `(max_samples, max_fields, enable_field_cache)` | Configuration for Zarr-backed cache limits. |
| `CachedUPDZarrStore` | classe | `(root, *, cache=None, mode='r')` | Cached wrapper around UPDZarrStore. |

#### `pinneapple_data.zarr_cached_store_bytes`

Byte-cached UPDZarrStore wrapper with configurable sample and field cache budgets.

<small>`pinneapple_data/zarr_cached_store_bytes.py` · 226 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ZarrByteCacheConfig` | classe | `(max_sample_bytes, max_field_bytes, enable_field_cache)` | Configuration for byte-bounded Zarr cache limits. |
| `CachedUPDZarrStoreBytes` | classe | `(root, *, cache=None, mode='r')` | Cached wrapper around UPDZarrStore using ByteLRUCache. |

#### `pinneapple_data.zarr_iterable`

IterableDataset for streaming PhysicalSamples from a UPDZarrStore with worker-aware sharding.

<small>`pinneapple_data/zarr_iterable.py` · 69 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ZarrUPDIterable` | classe | `(root, fields, coords, start, stop, step, shuffle, seed)` | IterableDataset over a UPDZarrStore. |

#### `pinneapple_data.zarr_prefetch`

Prefetch IterableDataset for Zarr-backed UPD stores with background producer and optional GPU transfer.

<small>`pinneapple_data/zarr_prefetch.py` · 205 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PrefetchConfig` | classe | `(prefetch, queue_max, use_sample_cache, pin_memory, target_device, transfer_non_blocking)` | Industrial prefetch controls. |
| `PrefetchZarrUPDIterable` | classe | `(root, *, fields=None, coords=None, dtype=None, start=0, end=None, stride=1, sample_ctor=…)` | Streaming dataset backed by Zarr with caching + background prefetch, plus pinned memory and async CPU->GPU transfer. |

#### `pinneapple_data.zarr_shard_iterable`

Shard-aware IterableDataset for balanced iteration over multi-shard Zarr UPD stores.

<small>`pinneapple_data/zarr_shard_iterable.py` · 162 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ShardAwareConfig` | classe | `(pin_memory, target_device, transfer_non_blocking, use_sample_cache)` | Configuration for shard-aware iteration behavior. |
| `ShardAwareZarrUPDIterable` | classe | `(root, *, fields=None, coords=None, dtype=None, sample_ctor=None, cache=None, cfg=None)` | Shard-aware iterator: - reads root/index.json - balances shard parts across DataLoader workers - iterates each assigned zarr store sequentially |

#### `pinneapple_data.zarr_shards`

Sharded Zarr writer for partitioning UPD samples by key with index-based discovery.

<small>`pinneapple_data/zarr_shards.py` · 154 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ShardSpec` | classe | `(key_fn, max_per_shard)` | Simple shard naming. |
| `UPDZarrShardedWriter` | classe | `(root, shard_spec, *, shards_dirname='shards', index_filename='index.json')` | Writes multiple shards: root/ shards/ <key>/part=0000/data.zarr <key>/part=0001/data.zarr index.json |

#### `pinneapple_data.zarr_store`

Zarr-backed UPD store for PhysicalSample persistence with columnar layout.

<small>`pinneapple_data/zarr_store.py` · 396 linhas · testes: 1</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `UPDZarrStore` | classe | `(root, mode='r')` | Minimal UPD Zarr store: |

### Exemplos de uso reais do repositório

_Nenhum exemplo encontrado no repositório para este pacote._

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 45 de 115.
- Funções de teste em arquivos que importam algum módulo do pacote: 507.

### Limitações conhecidas

- _Nenhuma registrada._

## 3. `pinneapple_pdb`

### Objetivo

_Sem texto de objetivo._

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 7 | 1.929 | 9 | 14 | 100 (5 arquivos) | 5 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `benchmarks` |  | 1 | 5 | 35 |
| `builder` |  | 1 | 5 | 0 |
| `derived` |  | 1 | 3 | 0 |
| `shard` |  | 1 | 6 | 0 |
| `templates` |  | 1 | 1 | 0 |
| `validate` |  | 1 | 3 | 0 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_pdb`

_Módulo sem docstring e sem texto._

<small>`pinneapple_pdb/__init__.py` · 36 linhas · testes: 65</small>

#### `pinneapple_pdb.benchmarks`

A small, curated, NAME-based catalog of real benchmark datasets.

<small>`pinneapple_pdb/benchmarks.py` · 984 linhas · testes: 35</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BenchmarkEntry` | classe | `(name, description, reference_source, x_vars, y_vars, reference_x, reference_y, verificat…)` | A single named, real reference dataset: enough to plug directly into ``pinneapple_llm.guardrail.PhysicsGuardrail._check_reference`` as ``(reference_x, reference_y)``. |
| `register_benchmark` | função | `(name)` | Decorator to register a zero-argument ``BenchmarkEntry`` factory under ``name`` -- mirrors ``pinneapple_physics.pde_environment.presets .registry.register_preset``'s decorator pattern. The factory is called fresh on every ``get_benchmark``/``benchmark_catalog`` lookup (not memoised), so the … |
| `get_benchmark` | função | `(name) -> BenchmarkEntry` | Resolve a benchmark dataset by name. |
| `list_benchmarks` | função | `() -> List[str]` | Return the sorted list of all registered benchmark dataset names. |
| `benchmark_catalog` | função | `() -> Dict[str, BenchmarkEntry]` | Return name -> ``BenchmarkEntry`` for every registered benchmark, built (and independently re-verified) fresh. |

#### `pinneapple_pdb.builder`

Builder for physical datasets from Earth data hubs (NASA CMR, earthaccess).

<small>`pinneapple_pdb/builder.py` · 539 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HubQuery` | classe | `(provider, short_name, keyword)` | Query parameters for Earth data hub search. |
| `SpaceTime` | classe | `(time_start, time_end, bbox, stride, chunking)` | Temporal and spatial extent plus stride/chunking options. |
| `VariableSelection` | classe | `(include, exclude, packs)` | Variable include/exclude/pack selection. |
| `PhysicalSchema` | classe | `(physical_system, governing_equations, ics, bcs, forcings, units_policy, regime_tags, val…)` | Physical system metadata: equations, BCs, ICs, units policy, regime tags. |
| `PhysicalDatasetBuilder` | classe | `(packs=None)` | MVP-1 builder: - time-window sharding (and optional spatial tiles) - derived: vorticity/divergence - validations: units/ranges/monotonic - CLI friendly |

#### `pinneapple_pdb.derived`

Derived variable computation (e.g. vorticity, divergence) for physical datasets.

<small>`pinneapple_pdb/derived.py` · 116 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DerivedSpec` | classe | `(derived, u_name, v_name)` | Specification for derived variables to add (e.g. vorticity, divergence). |
| `add_vorticity_divergence` | função | `(ds, u_name=None, v_name=None) -> xr.Dataset` | Add vorticity and divergence from U/V on lat/lon grid; returns new dataset. |
| `apply_derived` | função | `(ds, spec) -> xr.Dataset` | Apply derived variables from spec to dataset; returns new dataset. |

#### `pinneapple_pdb.shard`

Sharding and subsetting of physical datasets by time and spatial tiles.

<small>`pinneapple_pdb/shard.py` · 72 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ShardSpec` | classe | `(time_window, tile_deg, add_regime_tags)` | Specification for time-window and optional spatial tiling of datasets. |
| `iter_time_windows` | função | `(time_values, window) -> Iterator[Tuple[pd.Timestamp, pd.Timestamp]]` | Yield (start, end) time pairs for non-overlapping windows over time_values. |
| `subset_time` | função | `(ds, a, b) -> xr.Dataset` | Select time slice [a, b) from dataset. |
| `iter_tiles` | função | `(lat_vals, lon_vals, dlat, dlon)` | Yield (lat0, lat1, lon0, lon1) tile bounds aligned to dlat/dlon grid. |
| `subset_tile` | função | `(ds, lat0, lat1, lon0, lon1) -> xr.Dataset` | Select spatial tile [lat0,lat1] x [lon0,lon1]; assumes coords sorted. |
| `regime_tags_for` | função | `(ds) -> list[str]` | Derive regime tags (e.g. tropics, polar) from dataset lat extent. |

#### `pinneapple_pdb.templates`

Physical schema templates for common datasets (e.g. atmosphere reanalysis).

<small>`pinneapple_pdb/templates.py` · 28 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `schema_templates` | função | `() -> Dict[str, Dict[str, Any]]` | Return mapping of template_id -> PhysicalSchema-compatible dict. |

#### `pinneapple_pdb.validate`

Validation and standardization for physical datasets (UPD).

<small>`pinneapple_pdb/validate.py` · 154 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ValidationSpec` | classe | `(require_units, ranges, enforce_lev_monotonic, enforce_time_monotonic)` | Specification for dataset validation (units, ranges, monotonicity). |
| `standardize_dims` | função | `(ds) -> xr.Dataset` | Standardize dimension names and transpose data vars to canonical order (time, lev, lat, lon). |
| `validate_dataset` | função | `(ds, spec) -> List[str]` | Validate dataset against spec; returns list of issue messages (empty if valid). |

### Exemplos de uso reais do repositório

_Nenhum exemplo encontrado no repositório para este pacote._

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 2 de 7.
- Funções de teste em arquivos que importam algum módulo do pacote: 100.

### Limitações conhecidas

- _Nenhuma registrada._

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

_Sem texto de objetivo._

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 31 | 15.016 | 90 | 49 | 94 (6 arquivos) | 23 |

Pasta de testes dedicada: `tests/pinneapple_worldmodel` (8 funções de teste).

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `benchmark` |  | 1 | 3 | 0 |
| `camera_system` |  | 1 | 3 | 0 |
| `curriculum` |  | 1 | 3 | 0 |
| `dataset` |  | 1 | 3 | 20 |
| `dataset_factory` |  | 1 | 4 | 0 |
| `dataset_packager` |  | 1 | 4 | 0 |
| `generate_datasets` |  | 1 | 4 | 0 |
| `geometry` |  | 1 | 17 | 0 |
| `inspection_data` |  | 1 | 11 | 39 |
| `mega_model` |  | 1 | 5 | 0 |
| `meta_learning` |  | 1 | 3 | 0 |
| `model` |  | 1 | 2 | 20 |
| `model_zoo` |  | 1 | 4 | 0 |
| `orchestrator` |  | 1 | 3 | 20 |
| `photorealistic_enhancer` |  | 1 | 3 | 0 |
| `physical_world` |  | 1 | 12 | 39 |
| `physics_renderer` |  | 1 | 3 | 0 |
| `physics_solver_bridge` |  | 1 | 4 | 0 |
| `physics_tools` |  | 1 | 2 | 35 |
| `pipeline` |  | 1 | 3 | 0 |
| `reality_randomizer` |  | 1 | 2 | 0 |
| `robot_perception` |  | 1 | 13 | 39 |
| `scenario` |  | 1 | 1 | 0 |
| `scenario_generator` |  | 1 | 2 | 0 |
| `simulator` |  | 1 | 2 | 0 |
| `specialist_trainer` |  | 1 | 2 | 0 |
| `synthetic_factory` |  | 1 | 7 | 0 |
| `train` |  | 1 | 5 | 0 |
| `trainer` |  | 1 | 3 | 0 |
| `training_hooks` |  | 1 | 6 | 0 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_worldmodel`

pinneapple_worldmodel — Generalist Physics AI World Model.

<small>`pinneapple_worldmodel/__init__.py` · 409 linhas · testes: 11</small>

#### `pinneapple_worldmodel.benchmark`

Standard physics AI benchmark suite.

<small>`pinneapple_worldmodel/benchmark.py` · 564 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BenchmarkTask` | classe | `(name, scenario_name, n_trajectories, rollout_steps, seed, physics_checks, metrics, descr…)` | A single benchmark evaluation task. |
| `BenchmarkResult` | classe | `(task_name, next_step_rmse, rollout_rmse, rel_l2, div_step, physics_scores, n_trajectorie…)` | Results for one BenchmarkTask. |
| `PhysicsBenchmark` | classe | `(device='cpu', verbose=True) -> None` | Evaluate physics AI models on standard tasks. |

#### `pinneapple_worldmodel.camera_system`

Stage 5 — Camera System.

<small>`pinneapple_worldmodel/camera_system.py` · 392 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CameraConfig` | classe | `(name, sensor_type, position, look_at, fov_deg, resolution, fps, near_clip, far_clip, noi…)` | Single virtual camera / sensor configuration. |
| `MultiCameraArray` | classe | `(cameras, sync_fps)` | A set of cameras sharing the same simulation field. |
| `CameraSystem` | classe | `(array=None, domain_bounds=None) -> None` | Project physics fields to camera-plane observations. |

#### `pinneapple_worldmodel.curriculum`

Curriculum learning for physics world models.

<small>`pinneapple_worldmodel/curriculum.py` · 283 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CurriculumStage` | classe | `(scenarios, n_samples, epochs, target_val_loss, lr, rollout_steps, description)` | One stage in the curriculum. |
| `CurriculumConfig` | classe | `(stages, model_config, device, batch_size, patience, save_dir, verbose)` | Full curriculum configuration. |
| `PhysicsCurriculum` | classe | `(config) -> None` | Train a world model through a staged physics curriculum. |

#### `pinneapple_worldmodel.dataset`

World model dataset: (state_t, params) → state_{t+1} pairs.

<small>`pinneapple_worldmodel/dataset.py` · 382 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `WorldModelDataset` | classe | `(trajectories, *, horizon=1, normalize=True, transform=None) -> None` | One-step or multi-step prediction dataset from a list of trajectories. |
| `DatasetConfig` | classe | `(scenarios, n_samples_per_scenario, horizon, normalize, validate_physics, save_dir, devic…)` | Configuration for :class:`DatasetBuilder`. |
| `DatasetBuilder` | classe | `(config) -> None` | Orchestrates physics simulation → :class:`WorldModelDataset` assembly. |

#### `pinneapple_worldmodel.dataset_factory`

Multi-source physics dataset factory.

<small>`pinneapple_worldmodel/dataset_factory.py` · 725 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FactoryConfig` | classe | `(sources, scenarios, n_samples_per_scenario, horizon, validate_physics, device, save_dir,…)` | Configuration for :class:`PhysicsDatasetFactory`. |
| `DatasetEntry` | classe | `(name, dataset, source, scenario, physics_tags, n_samples, generation_time_s, metadata)` | One dataset slice from the catalog. |
| `DatasetCatalog` | classe | `() -> None` | Registry of all generated physics datasets. |
| `PhysicsDatasetFactory` | classe | `(config) -> None` | Generate multi-source physics training datasets. |

#### `pinneapple_worldmodel.dataset_packager`

Stage 8 — Dataset Packager.

<small>`pinneapple_worldmodel/dataset_packager.py` · 439 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PackagerConfig` | classe | `(output_dir, format, zarr_chunks, zarr_compressor, save_vtk, save_videos, sample_id_digit…)` | Controls how the packager writes the dataset. |
| `PackagedSample` | classe | `(sample_id, sample_dir, field_paths, video_paths, metadata_path)` | Metadata for a single packaged sample. |
| `DatasetManifest` | classe | `(output_dir) -> None` | Index of all samples in a Physical AI dataset. |
| `DatasetPackager` | classe | `(config=None) -> None` | Write one physics sample to the canonical dataset structure. |

#### `pinneapple_worldmodel.generate_datasets`

Dataset generation for Physics World Model training.

<small>`pinneapple_worldmodel/generate_datasets.py` · 441 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `generate` | função | `(scenarios=None, sources=None, n_samples=100, output_dir='./data/worldmodel', device='cpu…)` | Generate multi-source physics datasets and return a DatasetCatalog. |
| `generate_via_factory` | função | `(scenarios=None, sources=None, n_samples=100, output_dir='./data/worldmodel', device='cpu…)` | Generate datasets using the high-level PhysicsDatasetFactory. |
| `load_catalog` | função | `(path) -> DatasetCatalog` | Load a previously saved DatasetCatalog from disk. |
| `main` | função | `() -> None` |  |

#### `pinneapple_worldmodel.geometry`

Physical domain geometry for physics simulation and world model training.

<small>`pinneapple_worldmodel/geometry.py` · 569 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GeometryBase` | classe | `()` | Abstract base class for all geometry objects. |
| `Rectangle` | classe | `(lo, hi) -> None` | Axis-aligned rectangle ``[lo_x, hi_x] × [lo_y, hi_y]``. |
| `Circle` | classe | `(center, radius) -> None` | 2-D filled circle. |
| `Polygon` | classe | `(vertices) -> None` | 2-D filled polygon defined by ordered vertices. |
| `Box3D` | classe | `(lo, hi) -> None` | Axis-aligned 3-D box. |
| `Sphere` | classe | `(center, radius) -> None` | 3-D filled sphere. |
| `Union` | classe | `(a, b) -> None` | CSG union: points inside *a* OR *b*. |
| `Intersection` | classe | `(a, b) -> None` | CSG intersection: points inside *a* AND *b*. |
| `Difference` | classe | `(a, b) -> None` | CSG difference: points inside *a* but NOT inside *b*. |
| `BoundaryRegion` | classe | `(name, bc_type, value, sampler)` | A named subset of the domain boundary with its own BC type. |
| `PhysicsDomain` | classe | `(geometry, boundaries, name, tags)` | Complete domain specification: geometry + boundary regions + metadata. |
| `make_unit_square` | função | `(*, with_boundary_labels=True) -> PhysicsDomain` | Unit square [0,1]² — standard benchmark domain. |
| `make_cavity` | função | `(L=1.0) -> PhysicsDomain` | Lid-driven cavity: unit square with moving lid (y=L, u=1). |
| `make_channel` | função | `(length=4.0, height=1.0, *, inlet_velocity=1.0) -> PhysicsDomain` | Poiseuille channel flow domain. |
| `make_channel_with_cylinder` | função | `(length=4.0, height=1.0, cyl_center=(1.0, 0.5), cyl_radius=0.1) -> PhysicsDomain` | 2-D channel with circular obstacle — classic benchmark. |
| `make_l_shaped` | função | `() -> PhysicsDomain` | L-shaped domain (stress concentration benchmark). |
| `make_annulus` | função | `(inner_r=0.3, outer_r=1.0, center=(0.0, 0.0)) -> PhysicsDomain` | Annular domain — useful for heat/EM problems in cylindrical geometry. |

#### `pinneapple_worldmodel.inspection_data`

Synthetic thermal-inspection data with labels from physics, scene variations written as OpenUSD stages, and the check that an augmentation kept the label true.

<small>`pinneapple_worldmodel/inspection_data.py` · 224 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Variation` | classe | `(gain, offset, noise, blur, zoom, angle_deg, shift, glint)` |  |
| `thermal_plate` | função | `(defect, n=48, *, k=1.0, k_void=0.15, q=1.0) -> dict` | Steady conduction on the unit square with a low-conductivity disk ``defect = (x, y, r)``. Returns the field, the defect-free field, the coefficient and the label (the cell where the void raises the temperature most). |
| `sample_variation` | função | `(rng, *, glint_p=0.2) -> Variation` |  |
| `camera_transform` | função | `(var, H, W)` | Plate unit square (u, v) -> image pixel (col, row) and its inverse, for the variation's camera. |
| `write_variation_stage` | função | `(path, plate, var, *, cmap='gray') -> str` |  |
| `thermal_image` | função | `(stage_path, var, H=64, W=64, seed=0) -> np.ndarray` | The camera image ``(H, W)`` of a variation stage. |
| `nominal_image` | função | `(stage_path, plate, var, H=64, W=64) -> np.ndarray` | What the same camera would see on the defect-free plate (the inspector's reference): the nominal field through the variation's gain, offset, pose and blur, no glint, no noise. |
| `label_check` | função | `(image, nominal, var, label_xy, *, tol=0.08) -> dict` | Does the image still show the physical anomaly where the label says? The residual against the nominal image, smoothed (2 px), its maximum mapped back to the plate; ``tol`` in plate units (0.08 = about 4 cells). A glint that outshines the defect moves the maximum and fails; so does a label on … |
| `inspection_dataset` | função | `(n, seed, workdir, *, glint_p=0.25, H=64, W=64, keep_stages=0) -> dict` | ``n`` samples: a random subsurface void, its steady temperature field, a random camera variation written as a USD stage and rendered from it, the physical label in pixels, and the label check. Stages beyond the first ``keep_stages`` are deleted after rendering. |
| `inspection_sample` | função | `(x=0.4, y=0.6, r=0.08, seed=0, glint=False, usd_path='inspection_sample.usda') -> dict` | One synthetic thermal-inspection sample: the plate with a void at ``(x, y)``, radius ``r`` (plate units), a random camera variation (with a specular glint if asked) written as a USD stage at ``usd_path``, and the label check of its image. |
| `cleanup` | função | `(paths) -> None` |  |

#### `pinneapple_worldmodel.mega_model`

Physics Foundation Model — the generalist mega-model.

<small>`pinneapple_worldmodel/mega_model.py` · 668 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FoundationConfig` | classe | `(n_modes, width, depth, context_dim, descriptor_vocab, n_heads, n_context_layers, lora_ra…)` | Hyper-parameter configuration for :class:`PhysicsFoundationModel`. |
| `LoRALinear` | classe | `(in_features, out_features, rank=8) -> None` | Low-Rank Adaptation layer for fast domain adaptation. |
| `PhysicsDescriptorEncoder` | classe | `(context_dim=64, max_scalar_features=32, n_heads=4, n_layers=2, dropout=0.1) -> None` | Encode a physics descriptor dict into a fixed-size context vector. |
| `FoundationFNOBlock` | classe | `(width, n_modes, context_dim, lora_rank=8, dropout=0.1, spatial_dim=2) -> None` | One FNO block with context injection and optional LoRA. |
| `PhysicsFoundationModel` | classe | `(config, *, n_fields=1, grid_shape=(64, 64)) -> None` | Generalist physics AI foundation model. |

#### `pinneapple_worldmodel.meta_learning`

Meta-learning for the physics world model.

<small>`pinneapple_worldmodel/meta_learning.py` · 512 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MetaConfig` | classe | `(algorithm, n_meta_epochs, n_inner_steps, inner_lr, outer_lr, n_tasks_per_batch, n_suppor…)` | Configuration for :class:`MetaLearner`. |
| `TaskDistribution` | classe | `(catalog, n_support=32, n_query=32, device='cpu') -> None` | Sample physics tasks from a :class:`~.dataset_factory.DatasetCatalog`. |
| `MetaLearner` | classe | `(config, model_config=None) -> None` | MAML / Reptile meta-learning for physics world models. |

#### `pinneapple_worldmodel.model`

Physics World Model architecture.

<small>`pinneapple_worldmodel/model.py` · 320 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `WorldModelConfig` | classe | `(n_modes, width, depth, context_dim, embed_dim, activation, dropout, rollout_steps)` | Hyperparameters for :class:`PhysicsWorldModel`. |
| `PhysicsWorldModel` | classe | `(config, *, n_fields, grid_shape) -> None` | FNO-based physics world model: (state_t, context) → state_{t+1}. |

#### `pinneapple_worldmodel.model_zoo`

Specialist model registry and checkpoint management.

<small>`pinneapple_worldmodel/model_zoo.py` · 494 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ZooEntry` | classe | `(name, model, scenario, physics_tags, metrics, n_params, trained_at, checkpoint_path, met…)` | One entry in the :class:`ModelZoo`. |
| `EnsembleModel` | classe | `(models, weights=None, aggregation='mean') -> None` | Averaging ensemble of multiple PhysicsWorldModel specialists. |
| `WaMaModel` | classe | `()` | Model created by averaging the weights of multiple specialist models. |
| `ModelZoo` | classe | `(root_dir=None) -> None` | Registry of trained specialist physics models. |

#### `pinneapple_worldmodel.orchestrator`

General-purpose physics problem solver — tool-based orchestrator.

<small>`pinneapple_worldmodel/orchestrator.py` · 766 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ProblemStatement` | classe | `(kind, description, pde_hint, domain_hint, scenarios, params, observations, model, n_samp…)` | A high-level description of a physics problem to solve. |
| `OrchestratorResult` | classe | `(kind, artifacts, plan, elapsed_s, logs)` | Output of :class:`PhysicsOrchestrator.solve`. |
| `PhysicsOrchestrator` | classe | `(registry=None, verbose=True, tool_gate=None) -> None` | Solve arbitrary physics problems by chaining Pinneapple tools. |

#### `pinneapple_worldmodel.photorealistic_enhancer`

Stage 7 — Photorealistic Enhancer.

<small>`pinneapple_worldmodel/photorealistic_enhancer.py` · 283 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): placeholder/stub citado no docstring</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EnhancerConfig` | classe | `(backend, strength, style_prompt, negative_prompt, model_id, device, batch_size, apply_to…)` | Configuration for the photorealistic enhancer. |
| `PhotorealisticEnhancer` | classe | `(config=None) -> None` | Enhance visual realism of rendered frames WITHOUT touching physics data. |
| `register_enhancer` | função | `(name, fn) -> None` | Register a custom enhancement backend. |

#### `pinneapple_worldmodel.physical_world`

PINNeAPPle World Model: physics-labelled synthetic video, a world model that trains on a CPU, and the physics checks that decide whether any video (simulated, predicted or generated by a foundation model such as NVIDIA Cosmos) is physically consistent.

<small>`pinneapple_worldmodel/physical_world.py` · 489 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Video` | classe | `(frames, dt, px_per_m, labels, params)` |  |
| `WorldModel` | classe | `(H=64, W=96, latent=32, seed=0)` | Encoder (conv) -> latent ``z`` -> GRU dynamics -> decoder (deconv); trained to predict frame t+1 from frames <= t. Small enough to train on a CPU in minutes on 64 x 96 frames. |
| `StateEncoder` | classe | `(H=64, W=96, seed=0)` | A small CNN that reads the object's centre (pixels) from a frame, trained on the exact labels of the synthetic videos (the labels are free because the frames come from a simulation). |
| `CosmosAdapter` | classe | `(base_url, api_key=None, timeout=600.0)` | ``generate(prompt, frames=None, seed=0)`` on an NVIDIA Cosmos world foundation model NIM: ``POST {base}/infer`` with ``prompt``, ``video`` (base64 MP4 of ``frames`` when given), ``seed``, ``video_params``; the answer's ``b64_video`` is decoded to frames (needs ``imageio`` with an FFmpeg plugin). … |
| `bouncing_ball_scene` | função | `(*, seed=0, T=48, fps=24.0, H=64, W=96, randomize=True, defect=False, e=None) -> Video` | A thrown ball bouncing on the floor, rendered with domain randomization; exact labels. |
| `dataset` | função | `(n, *, seed=0, **kw) -> list[Video]` |  |
| `track_object` | função | `(frames) -> dict[str, np.ndarray]` | Centre ``(cx, cy)`` [px] and area [px] of the brightest-contrast object per frame (NaN when none is found). |
| `physics_checks` | função | `(video, *, dt=None, px_per_m=None, g=G, g_rtol=0.1, area_rtol=0.25, max_speed=10.0) -> di…)` | Physics checks of a ball video from its pixels; ``video`` is a :class:`Video` or frames (then give ``dt`` and ``px_per_m``). Returns ``{"checks": [...], "passed": bool, "track": ..., "g_fit": ...}``. |
| `video_check` | função | `(seed=0, edit='none') -> dict[str, Any]` | A synthetic bouncing-ball video (exact physics, randomized camera), optionally edited (``"time_warp"``, ``"teleport"``, ``"fade"``, ``"gravity_flip"``), judged by :func:`physics_checks` from its pixels alone. |
| `corrupt` | função | `(video, kind, seed=0) -> Video` | Non-physical edits: ``"time_warp"`` (motion played at a varying rate), ``"teleport"`` (frames swapped in the middle), ``"fade"`` (the object fades out), ``"gravity_flip"`` (the video played backwards in time after the first bounce, so the ball gains energy). |
| `state_pairs` | função | `(videos) -> tuple[np.ndarray, np.ndarray]` | ``(s_t, s_t+1)`` with ``s = (x, y, vx, vy)`` in metres from the exact labels (the training data of a learned state dynamics). |
| `render_like` | função | `(video, x_m, y_m) -> np.ndarray` | Frames of a trajectory ``(x_m, y_m)`` rendered with ``video``'s camera and appearance (its randomization parameters), noise from a fixed seed. |

#### `pinneapple_worldmodel.physics_renderer`

Stage 4 — Physics Renderer.

<small>`pinneapple_worldmodel/physics_renderer.py` · 444 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RendererConfig` | classe | `(resolution, fps, sensors, colormap_rgb, colormap_thermal, colormap_depth, colormap_ir, d…)` | Configuration for the PhysicsRenderer. |
| `RenderResult` | classe | `(frames, video_paths, png_dirs, field_stats)` | Output of PhysicsRenderer.render(). |
| `PhysicsRenderer` | classe | `(config=None) -> None` | Render physical field trajectories into video frames. |

#### `pinneapple_worldmodel.physics_solver_bridge`

Stage 2 — Physics Solver Layer  +  Stage 3 — Physical Ground Truth Dataset.

<small>`pinneapple_worldmodel/physics_solver_bridge.py` · 837 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SolverOutput` | classe | `(fields, t_coords, x_coords, y_coords, z_coords, params, solver_name, metadata)` | Multivariate physical field history from one simulation run. |
| `SolverBridgeConfig` | classe | `(solver, generate_temperature, generate_concentration, openfoam_case_dir, fenics_problem,…)` | Configuration for the PhysicsSolverBridge (Stage 2). |
| `PhysicsSolverBridge` | classe | `(config=None) -> None` | Dispatch a physics scenario to the appropriate solver backend. |
| `GroundTruthPackager` | classe | `(output_dir, format='zarr', zarr_chunks=None, overwrite=True) -> None` | Organise SolverOutput into the canonical Physical AI dataset structure. |

#### `pinneapple_worldmodel.physics_tools`

Pinneapple module capability inventory for tool-based orchestration.

<small>`pinneapple_worldmodel/physics_tools.py` · 983 linhas · testes: 35</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsTool` | classe | `(name, category, description, input_schema, output_schema, module_path, fn, tags)` | One registered physics capability. |
| `PhysicsToolRegistry` | classe | `() -> None` | Catalogue of all Pinneapple capabilities as callable tools. |

#### `pinneapple_worldmodel.pipeline`

Physics AI Pipeline — unified entry point.

<small>`pinneapple_worldmodel/pipeline.py` · 498 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsAIConfig` | classe | `(scenarios, sources, n_samples, device, save_dir, specialist_epochs, specialist_lr, speci…)` | Complete configuration for :class:`PhysicsAIPipeline`. |
| `PhysicsAIPipelineResult` | classe | `(catalog, zoo, meta_model, mega_model, benchmark, elapsed_s)` | All outputs of :class:`PhysicsAIPipeline.run`. |
| `PhysicsAIPipeline` | classe | `(config) -> None` | Build a generalist physics AI world model using all Pinneapple tools. |

#### `pinneapple_worldmodel.reality_randomizer`

Stage 6 — Reality Randomizer.

<small>`pinneapple_worldmodel/reality_randomizer.py` · 328 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RandomizerConfig` | classe | `(seed, p_sensor_noise, gaussian_std, poisson_scale, p_jpeg, jpeg_quality_lo, jpeg_quality…)` | Controls which augmentations are applied and their strength. |
| `RealityRandomizer` | classe | `(config=None) -> None` | Apply reality-gap augmentations to rendered image sequences. |

#### `pinneapple_worldmodel.robot_perception`

Robot perception data from the robot sim: arm motions simulated by PINNeAPPle Robot Sim (NVIDIA Newton), written as OpenUSD timelines with a camera, seen by a randomized synthetic camera, labelled with the simulator's exact joint angles, and checked for what a perception model of a robot must …

<small>`pinneapple_worldmodel/robot_perception.py` · 283 linhas · testes: 39</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CameraVariation` | classe | `(zoom, offset, gain, noise, blur, albedo, background, occluder)` |  |
| `ImageRegressor` | classe | `(H, W, n_out, seed=0)` | A small CNN from a grey image ``(H, W)`` to ``n_out`` numbers. |
| `forward_points` | função | `(arm, q) -> dict[str, np.ndarray]` | Elbow and tool point ``(T, 2)`` in the x-z plane for joint angles ``q (T, 2)`` (the simulator's convention: angles from the horizontal, positive turning x towards -z). |
| `arm_motions` | função | `(arm, n, seed, *, duration=0.8, dt=0.001, fps=30.0, kp=(350.0, 110.0), kd=(5.0, 2.5), q_b…)` | ``n`` random point-to-point moves simulated in Newton, sampled at ``fps`` (or the given ``moves``: ``[(q_start, q_end)]``). |
| `sample_camera` | função | `(rng, *, occluder_p=0.3) -> CameraVariation` |  |
| `write_motion_stage` | função | `(path, arm, motion, var, *, payload=None) -> str` | ``payload``: ``(V, F, grey)``, a mesh carried at the tool point (in the tool frame: x along the forearm). |
| `camera_frames` | função | `(path, arm, motion, var, H=72, W=72, seed=0) -> np.ndarray` | ``(T, H, W)`` grey frames of the stage at each recorded time. |
| `visibility` | função | `(arm, motion, var, *, margin=0.03) -> np.ndarray` | Per frame: are the elbow and the tool point inside the image and not behind the occluder? |
| `payload_track_check` | função | `(frames, arm, motion, var, *, tol_px=2.0, offset=(0.0, 0.0)) -> dict` | Does each frame show the carried part where the kinematics put it? The centroid of the pixels brighter than the links (the part is drawn white, the links grey: threshold half-way between the two after the camera's gamma and gain) against the tool point, plus the part's ``offset`` (x, z in the tool … |
| `arm_camera_sample` | função | `(payload=5.0, seed=0, occluder=False, usd_path='arm_camera_sample.usda') -> dict` | One labelled perception sample: a random arm move in the robot sim (Newton), its USD timeline with a side camera (random variation, an occluder if asked) at ``usd_path``, the camera frames, and the label checks (the forward kinematics against the simulator's link poses, the frames that show the … |
| `angle_features` | função | `(q) -> np.ndarray` |  |
| `feature_angles` | função | `(f) -> np.ndarray` |  |
| `perception_checks` | função | `(q_pred, motion, arm, *, qd_max=12.0) -> dict` | Errors of a predicted angle sequence for one motion, all frames visible. |

#### `pinneapple_worldmodel.scenario`

Physics scenario definitions for world model training.

<small>`pinneapple_worldmodel/scenario.py` · 253 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsScenario` | classe | `(name, pde_kind, grid_shape, t_span, n_steps, domain_bounds, param_ranges, ic_type, bc_ty…)` | Complete specification of a physics simulation problem. |

#### `pinneapple_worldmodel.scenario_generator`

Stage 1 — Scenario Generator.

<small>`pinneapple_worldmodel/scenario_generator.py` · 360 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ScenarioSpec` | classe | `(scenario, domain, geometry, fluid, boundary_conditions, initial_conditions, sensor_confi…)` | Extended scenario specification produced by the generator. |
| `ScenarioGenerator` | classe | `(default_grid_2d=(64, 64), default_n_steps=32, device='cpu') -> None` | Convert a user prompt or JSON dict into a :class:`ScenarioSpec`. |

#### `pinneapple_worldmodel.simulator`

Physics trajectory simulator.

<small>`pinneapple_worldmodel/simulator.py` · 630 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TrajectoryData` | classe | `(states, params, scenario_name, metadata)` | One physics trajectory: sequence of field snapshots + metadata. |
| `PhysicsSimulator` | classe | `(scenario, *, device='cpu', verbose=False) -> None` | Generate physics trajectories from a :class:`PhysicsScenario`. |

#### `pinneapple_worldmodel.specialist_trainer`

Specialist model training using Pinneapple's full training stack.

<small>`pinneapple_worldmodel/specialist_trainer.py` · 423 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SpecialistConfig` | classe | `(model, epochs, lr, batch_size, device, val_fraction, patience, grad_clip, rollout_steps,…)` | Training configuration for :class:`SpecialistTrainer`. |
| `SpecialistTrainer` | classe | `(config, *, physics_loss_fn=None, pretrained_zoo=None) -> None` | Train specialist models and populate a :class:`~.model_zoo.ModelZoo`. |

#### `pinneapple_worldmodel.synthetic_factory`

Physics Synthetic Data Factory for Physical AI.

<small>`pinneapple_worldmodel/synthetic_factory.py` · 525 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SyntheticFactoryConfig` | classe | `(scenario_input, n_samples, output_dir, device, render_fps, sensors, render_resolution, a…)` | Single configuration dataclass covering all 9 stages. |
| `FactoryResult` | classe | `(config, spec, samples, output_dir, elapsed_s)` | Output of SyntheticDataFactory.generate_dataset(). |
| `SyntheticDataFactory` | classe | `(config=None) -> None` | Physics Synthetic Data Factory for Physical AI. |
| `generate_pipe_flow` | função | `(n_samples=20, output_dir='./pipe_dataset', **kw) -> FactoryResult` | Generate a pipe flow dataset (NS2D, Re=100-5000). |
| `generate_cylinder_flow` | função | `(n_samples=20, output_dir='./cylinder_dataset', **kw) -> FactoryResult` | Generate flow-around-cylinder dataset (classic von Karman vortex shedding). |
| `generate_heat_transfer` | função | `(n_samples=20, output_dir='./heat_dataset', **kw) -> FactoryResult` | Generate 2D heat conduction dataset. |
| `generate_chemical_mixing` | função | `(n_samples=20, output_dir='./mixing_dataset', **kw) -> FactoryResult` | Generate chemical mixing / advection-diffusion dataset. |

#### `pinneapple_worldmodel.train`

Physics World Model training entry point.

<small>`pinneapple_worldmodel/train.py` · 593 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `train_specialist` | função | `(catalog, output_dir, *, epochs=50, batch_size=32, lr=0.001, device='cpu', patience=10, n…)` | Train one specialist model per scenario in the catalog. |
| `train_meta` | função | `(catalog, output_dir, *, algorithm='reptile', n_meta_epochs=100, n_inner_steps=5, inner_l…)` | Meta-train a shared initialisation over the full scenario catalog. |
| `train_foundation` | função | `(catalog, output_dir, *, base_ckpt=None, epochs=30, batch_size=16, lr=0.0005, device='cpu…)` | Build and fine-tune the PhysicsFoundationModel (mega-model with LoRA). |
| `run_pipeline` | função | `(scenarios=None, sources=None, n_samples=100, output_dir='./checkpoints/worldmodel', devi…)` | Run the complete 5-stage Physics AI pipeline. |
| `main` | função | `() -> None` |  |

#### `pinneapple_worldmodel.trainer`

World model training loop.

<small>`pinneapple_worldmodel/trainer.py` · 421 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `WorldModelLoss` | classe | `(rollout_steps=1, rollout_weight=0.5, physics_loss_fn=None, physics_weight=0.01) -> None` | Multi-step rollout loss for world model training. |
| `WorldModelTrainConfig` | classe | `(epochs, lr, weight_decay, batch_size, val_fraction, device, amp, grad_clip, rollout_step…)` | Training hyperparameters for :class:`WorldModelTrainer`. |
| `WorldModelTrainer` | classe | `(model, config, *, physics_loss_fn=None) -> None` | Train a :class:`PhysicsWorldModel` on a :class:`WorldModelDataset`. |

#### `pinneapple_worldmodel.training_hooks`

Stage 9 — Model Training Layer.

<small>`pinneapple_worldmodel/training_hooks.py` · 478 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicsAIDataset` | classe | `(dataset_dir, fields=('u', 'v', 'p', 'T'), sensors=(), max_samples=None, device='cpu') ->…)` | Base dataset over a packaged Physical AI dataset directory. |
| `CosmosEncoderDataset` | classe | `(dataset_dir, video_sensors=('rgb', 'thermal'), field_names=('u', 'v', 'p', 'T'), **kwarg…)` | Dataset for training a video → latent embedding encoder. |
| `PhysicsDecoderDataset` | classe | `(dataset_dir, embedding_dim=512, field_names=('u', 'v', 'p', 'T'), device='cpu', max_samp…)` | Dataset for training a latent embedding → physical field decoder. |
| `InversePINNDataset` | classe | `(dataset_dir, param_keys=None, video_sensor='rgb', **kwargs) -> None` | Dataset for inverse PINN training: video frames → physical parameters. |
| `NeuralOperatorDataset` | classe | `(dataset_dir, field_names=('u', 'v', 'p'), horizon=1, subsample_t=1, device='cpu', max_sa…)` | Dataset for training neural operators on (state_t, state_{t+1}) pairs. |
| `TrainingHooks` | classe | `(dataset_dir, device='cpu') -> None` | Convenience factory for all training dataset types. |

### Exemplos de uso reais do repositório

_Nenhum exemplo encontrado no repositório para este pacote._

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 8 de 31.
- Funções de teste em arquivos que importam algum módulo do pacote: 94.

### Limitações conhecidas

- _Nenhuma registrada._

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

_Sem texto de objetivo._

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 16 | 2.636 | 17 | 24 | 0 (0 arquivos) | 16 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `backends` |  | 2 | 7 | 0 |
| `circuits` |  | 2 | 4 | 0 |
| `data` |  | 2 | 7 | 0 |
| `loss` |  | 2 | 10 | 0 |
| `models` |  | 3 | 2 | 0 |
| `pipeline` |  | 2 | 5 | 0 |
| `training` |  | 2 | 4 | 0 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_quantum`

pinneapple_quantum — PINNeAPPle Quantum Module (PQM).

<small>`pinneapple_quantum/__init__.py` · 210 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_circuit` | função | `(n_qubits=4, depth=2, ansatz='hardware_efficient', encoding='angle', n_observables=1, bac…)` | Shortcut to build a :class:`QuantumModel` from flat arguments. |
| `backend_info` | função | `(provider='pennylane') -> dict` | Return info dict for the specified backend. |

#### `pinneapple_quantum.backends`

_Módulo sem docstring e sem texto._

<small>`pinneapple_quantum/backends/__init__.py` · 4 linhas · **sem teste direto**</small>

#### `pinneapple_quantum.backends.backend`

Quantum backend adapter — PennyLane, Qiskit, or classical simulator.

<small>`pinneapple_quantum/backends/backend.py` · 324 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BackendConfig` | classe | `(provider, device_name, diff_method, shots)` | Configuration for the quantum execution backend. |
| `QuantumBackend` | classe | `(config)` | Abstract base for quantum backends. |
| `PennyLaneBackend` | classe | `()` | PennyLane backend (supports CPU/GPU simulators and real QPUs). |
| `QiskitBackend` | classe | `()` | Qiskit / Aer backend. |
| `ClassicalBackend` | classe | `()` | Classical state-vector simulator. |
| `ClassicalQNode` | classe | `(circuit_fn, n_qubits)` | Callable that executes a quantum circuit description on the classical engine. |
| `get_backend` | função | `(config=None) -> 'QuantumBackend'` | Instantiate the correct backend from a config (or defaults). |

#### `pinneapple_quantum.circuits`

_Módulo sem docstring e sem texto._

<small>`pinneapple_quantum/circuits/__init__.py` · 8 linhas · **sem teste direto**</small>

#### `pinneapple_quantum.circuits.base`

Quantum circuit builder — ansatz and encoding definitions.

<small>`pinneapple_quantum/circuits/base.py` · 300 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `QuantumCircuitConfig` | classe | `(n_qubits, depth, ansatz, encoding, n_observables)` | Full specification for a variational quantum circuit (VQC). |
| `ClassicalVQC` | classe | `(config)` | Variational quantum circuit **approximated** entirely in PyTorch. |
| `build_pennylane_circuit` | função | `(config, backend)` | Construct a PennyLane QNode implementing the given circuit config. |
| `build_classical_circuit` | função | `(config, backend)` | Construct the same circuit as :func:`build_pennylane_circuit`, but recorded against a gate-recorder protocol (``recorder.rx/ry/rz/cnot/hadamard/cz/ measure_z``) that :class:`~pinneapple_quantum.backends.backend.ClassicalQNode` replays on an exact state-vector engine — used when no quantum package … |

#### `pinneapple_quantum.data`

_Módulo sem docstring e sem texto._

<small>`pinneapple_quantum/data/__init__.py` · 20 linhas · **sem teste direto**</small>

#### `pinneapple_quantum.data.quantum_dataset`

Quantum dataset layer — states, operators, measurements, and domain samplers.

<small>`pinneapple_quantum/data/quantum_dataset.py` · 290 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `QuantumState` | classe | `(coords, psi_real, psi_imag, meta)` | A pure quantum state ψ(x) over a spatial domain. |
| `QuantumObservable` | classe | `(expectation, variance, operator_names)` | A measured observable produced by a quantum circuit. |
| `QuantumCollocationDataset` | classe | `(x_col, x_bc, psi_exact=None, batch_size=256)` | Dataset of domain collocation points for quantum PINN training. |
| `SpinChainDataset` | classe | `(n_spins=4, n_samples=1024, J=1.0, h=0.5, seed=42)` | Dataset of spin-chain configurations and their energy labels. |
| `sample_domain_1d` | função | `(x_min=0.0, x_max=1.0, n_col=1000, n_bc=50, device='cpu', dtype=torch.float32) -> Tuple[t…)` | Sample collocation and boundary points for a 1D quantum domain. |
| `sample_domain_2d` | função | `(x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0, n_col=4096, n_bc=200, device='cpu', dtype=…)` | Sample collocation and boundary points for a 2D quantum domain. |
| `exact_eigenstates_1d_harmonic` | função | `(x, n_state=0, omega=1.0, hbar=1.0, mass=1.0) -> Tuple[torch.Tensor, float]` | Analytically compute the n-th eigenstate of the 1D harmonic oscillator. |

#### `pinneapple_quantum.loss`

_Módulo sem docstring e sem texto._

<small>`pinneapple_quantum/loss/__init__.py` · 26 linhas · **sem teste direto**</small>

#### `pinneapple_quantum.loss.schrodinger`

Physics losses for the Schrödinger equation and quantum Hamiltonians.

<small>`pinneapple_quantum/loss/schrodinger.py` · 344 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `harmonic_potential` | função | `(x, omega=1.0, mass=1.0) -> torch.Tensor` | V(x) = ½mω²‖x‖² — quantum harmonic oscillator. |
| `infinite_well_potential` | função | `(x, width=1.0) -> torch.Tensor` | V(x) = 0 inside (0, L), handled via Dirichlet BCs. Returns 0 in domain. |
| `coulomb_potential` | função | `(x, charge=1.0, eps=0.001) -> torch.Tensor` | V(r) = −e²/r — hydrogen-like atom potential (regularized at origin). |
| `double_well_potential` | função | `(x, a=1.0, b=0.0) -> torch.Tensor` | V(x) = (x² − a)² + bx — symmetric / asymmetric double well. |
| `get_potential` | função | `(name_or_fn) -> Callable` | Resolve a potential by name string or return it directly if callable. |
| `energy_loss` | função | `(model, x, potential='harmonic_oscillator', hbar=1.0, mass=1.0, **potential_kwargs) -> Tu…)` | Variational energy (Rayleigh quotient) for the time-independent Schrödinger equation. |
| `pde_residual_loss` | função | `(model, x, eigenvalue, potential='harmonic_oscillator', hbar=1.0, mass=1.0, **potential_k…)` | Strong-form PDE residual for the time-independent Schrödinger equation. |
| `time_dependent_loss` | função | `(model, x, t, potential='harmonic_oscillator', hbar=1.0, mass=1.0, **potential_kwargs) ->…)` | PDE residual for the time-dependent Schrödinger equation. |
| `normalization_loss` | função | `(model, x, domain_volume=1.0) -> torch.Tensor` | Soft constraint enforcing wavefunction normalization: ∫\|ψ\|² dx ≈ 1. |
| `hamiltonian_expectation` | função | `(model, x, H_matrix) -> torch.Tensor` | Discrete Hamiltonian expectation ⟨ψ\|H\|ψ⟩ for matrix Hamiltonians. |

#### `pinneapple_quantum.models`

_Módulo sem docstring e sem texto._

<small>`pinneapple_quantum/models/__init__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_quantum.models.hybrid_model`

HybridModel — classical neural network + variational quantum circuit.

<small>`pinneapple_quantum/models/hybrid_model.py` · 112 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HybridModel` | classe | `(classical_net, quantum_model, mode='classical_first', out_dim=None)` | Classical–quantum hybrid model that combines an ``nn.Module`` with a :class:`QuantumModel` into a single differentiable computational graph. |

#### `pinneapple_quantum.models.quantum_model`

QuantumModel — variational quantum circuit as a PyTorch nn.Module.

<small>`pinneapple_quantum/models/quantum_model.py` · 133 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `QuantumModel` | classe | `(circuit_config, backend_config=None, output_scale=1.0)` | A variational quantum circuit (VQC) exposed as a standard PyTorch ``nn.Module``. |

#### `pinneapple_quantum.pipeline`

_Módulo sem docstring e sem texto._

<small>`pinneapple_quantum/pipeline/__init__.py` · 16 linhas · **sem teste direto**</small>

#### `pinneapple_quantum.pipeline.presets`

High-level pipeline presets for common quantum ML workflows.

<small>`pinneapple_quantum/pipeline/presets.py` · 490 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `vq_pinn` | função | `(equation='schrodinger_1d', potential='harmonic_oscillator', n_qubits=4, depth=3, ansatz=…)` | Build and train a Variational Quantum PINN (VQ-PINN). |
| `hybrid_pinn` | função | `(classical_in=1, classical_hidden=32, n_qubits=4, depth=2, potential='harmonic_oscillator…)` | Hybrid PINN: classical MLP pre-encodes features → quantum circuit → output. |
| `learn_hamiltonian` | função | `(data, labels, n_spins=4, epochs=2000, lr=0.001, device='cpu', seed=42) -> Dict[str, Any]` | Learn a matrix Hamiltonian from energy measurement data. |
| `quantum_kernel` | função | `(x_train, x_test=None, n_qubits=4, encoding='angle', backend='pennylane') -> torch.Tensor` | Compute a quantum-feature-map kernel matrix K(xᵢ, xⱼ) for use with classical kernel methods (SVM, GP regression) in an elevated feature space. |
| `solve` | função | `(problem='quantum', equation='schrodinger', potential='harmonic', method='vq_pinn', backe…)` | High-level entry-point that mirrors the PINNeAPPle ``solve()`` API. |

#### `pinneapple_quantum.training`

_Módulo sem docstring e sem texto._

<small>`pinneapple_quantum/training/__init__.py` · 9 linhas · **sem teste direto**</small>

#### `pinneapple_quantum.training.trainer`

Quantum-aware hybrid trainer with parameter-shift support.

<small>`pinneapple_quantum/training/trainer.py` · 345 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `QTrainerConfig` | classe | `(epochs, lr, optimizer, quantum_gradient, lambda_norm, lambda_bc, print_every, device)` | Configuration for the quantum-aware trainer. |
| `QTrainHistory` | classe | `(epochs, total_loss, pde_loss, energy, norm_sq)` | Training history container. |
| `QTrainer` | classe | `(model, physics_loss_fn, bc_loss_fn=None, config=None)` | Hybrid classical-quantum trainer. |
| `parameter_shift_gradient` | função | `(loss_fn, weights, shift=math.pi / 2) -> torch.Tensor` | Compute exact gradient of ``loss_fn`` w.r.t. ``weights`` using the parameter-shift rule. |

### Exemplos de uso reais do repositório

_Nenhum exemplo encontrado no repositório para este pacote._

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 0 de 16.
- Funções de teste em arquivos que importam algum módulo do pacote: 0.

### Limitações conhecidas

- _Nenhuma registrada._
