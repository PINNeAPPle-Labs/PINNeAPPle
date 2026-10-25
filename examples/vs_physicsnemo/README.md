# PINNeAPPle vs PhysicsNeMo — Comparison & Integration Guide

> As linhas marcadas como medidas vêm do benchmark entre frameworks
> ([`benchmarks/vs_frameworks/REPORT.md`](../../benchmarks/vs_frameworks/REPORT.md)): só CPU (Intel Xeon, 1 thread por run),
> 3 a 5 seeds, nvidia-physicsnemo 2.2.2 e nvidia-physicsnemo.sym 2.4.0. Nada foi medido em GPU, que é onde o
> PhysicsNeMo é otimizado. As outras linhas descrevem funcionalidades e não foram medidas.

## Resumo executivo

| Pergunta | Ferramenta |
|---|---|
| Preciso de UQ / intervalos de confiança nas predições? | **PINNeAPPle** |
| Preciso treinar em multi-GPU / cluster com NCCL? | **PhysicsNeMo** |
| Preciso de digital twin com streams de sensores reais? | **PINNeAPPle** |
| Preciso de FNO/AFNO para operator learning em larga escala? | **PhysicsNeMo** |
| Preciso de meta-aprendizado para famílias paramétricas de PDEs? | **PINNeAPPle** |
| Preciso de MeshGraphNet para malhas não-estruturadas >200k nós? | **PhysicsNeMo** |
| Preciso de collocação adaptativa (RAR/RAD)? | **Ambos**: PINNeAPPle (RAR, RAD); PhysicsNeMo-Sym (amostragem por importância); DeepXDE também tem RAR |
| Preciso de FourCastNet / SFNO para previsão climática global? | **PhysicsNeMo** |
| Preciso de XPINN com decomposição de domínio? | **PINNeAPPle** |
| Preciso de assimilação de dados (EKF/EnKF)? | **PINNeAPPle** |
| Preciso de TensorRT / Triton para inferência em produção NVIDIA? | **PhysicsNeMo** |
| Preciso de presets para finanças, fármacos, sistemas sociais? | **PINNeAPPle** |
| Preciso de validação de consistência física (conservação, BCs)? | **PINNeAPPle** |
| Quero o melhor throughput bruto em CFD em hardware NVIDIA? | **PhysicsNeMo** |

---

## Filosofias diferentes

### PhysicsNeMo (ex-NVIDIA Modulus)
- **GPU-first**: built para tirar o máximo de hardware NVIDIA (cuDNN, Transformer Engine, TensorRT)
- **Escala**: de um GPU a clusters multi-nó via `torch.distributed` + NCCL
- **Foco**: surrogate de alto throughput para CFD e previsão do tempo
- **Deploy**: integrado com Omniverse / NVIDIA digital twin platform
- **Modelos pré-treinados**: FourCastNet, SFNO, GraphCastNet disponíveis como checkpoints

### PINNeAPPle
- **Pesquisa → Produção**: pipeline completo de geometria até digital twin deployed
- **Domínios amplos**: não só CFD — finanças, farmacologia, materiais, sistemas sociais
- **Inteligência sobre velocidade**: UQ, assimilação, meta-aprendizado, validação física
- **Facilidade de uso**: `pp.quickstart("burgers_1d")`, presets prontos, YAML pipeline
- **Open research**: implementações de referência limpas de SA-PINN, XPINN, XTFC, VPINN

---

## Tabela de features completa

| Feature | PINNeAPPle | PhysicsNeMo | Notas |
|---|---|---|---|
| **Arquiteturas PINN** | 8 variantes (ver catálogo abaixo) | PhysicsNeMo-Sym 2.4.0 (`physicsnemo/sym/models/`): fully connected, Fourier features, modified Fourier, multiscale Fourier, highway Fourier, SiReN, DGM, multiplicative filter, hash encoding, radial basis, fused MLP | Ambos têm várias; "mais variado" não foi medido. Com a mesma rede, pontos e perda (trilha A1) PINNeAPPle, DeepXDE e Sym dão os mesmos erros |
| **Operadores neurais** | FNO, DeepONet, GNO, PINO, UNO, MultiScale-DeepONet | FNO, AFNO, SFNO, FNO-2D/3D (o Sym tem também DeepONet, FNO, AFNO) | Medido só em CPU, 1 thread, Darcy 64×64, 12 épocas, mesmo bloco espectral (~2,36M parâmetros): o `FNO` do PhysicsNeMo levou 3,2× o tempo por época e 2,4–4,5× a latência de inferência do `FNO2d` do PINNeAPPle, com erro 1,4× maior. Velocidade em GPU e ranking na convergência não medidos ([trilha B](../../benchmarks/vs_frameworks/REPORT.md#b-fno-on-darcy-64-x-64-3-seeds-12-epochs-the-same-training-loop-and-data-for-the-three-models)) |
| **GNN** | GNN, EGNN, GNN-ODE, STGNN, GraphCast | MeshGraphNet, AeroGraphNet, X-MeshGraphNet, DoMINO | PhysicsNeMo especializado para CFD |
| **ROMs** | POD, DMD, HAVOK, OpInf, DeepUQROM | ❌ | PINNeAPPle exclusivo |
| **Neural ODEs** | NeuralODE, LatentODE, HNN, SymplecticODE, NeuralSDE | ❌ | PINNeAPPle exclusivo |
| **Autoencoders** | DenseAE, VAE, KAE, PI-Koopman-AE, AEROMHybrid | ❌ | PINNeAPPle exclusivo |
| **Reservoir Computing** | ELM, RBF, ESN, Koopman, HybridRBF | ❌ | PINNeAPPle exclusivo |
| **Séries Temporais** | LSTM, GRU, Seq2Seq, Informer, TFT, Autoformer, TimesNet | ❌ | PINNeAPPle exclusivo |
| **Clima / Tempo** | — | FourCastNet, SFNO, DLWP, GraphCastNet | PhysicsNeMo exclusivo |
| **Multi-GPU** | `torch.DataParallel` + `maybe_compile` | DDP completo + NCCL + multi-nó | PhysicsNeMo mais robusto |
| **UQ** | MC Dropout, Ensemble, Conformal | ❌ | PINNeAPPle exclusivo |
| **Digital Twin** | Streams MQTT/Kafka/HTTP/Mock, EKF/EnKF, anomalia | ❌ | PINNeAPPle exclusivo |
| **Meta-aprendizado** | MAML, Reptile, PDETaskSampler | ❌ | PINNeAPPle exclusivo |
| **Transfer Learning** | TransferTrainer, ParametricFamilyTransfer | ❌ | PINNeAPPle exclusivo |
| **Active Learning** | RAR, RAD, CombinedAL, AdaptiveCollocationTrainer | PhysicsNeMo-Sym: amostragem por importância nas restrições contínuas (`importance_measure`, `domain/constraint/continuous.py`) | Não é exclusivo; DeepXDE também tem RAR (exemplo `Burgers_RAR.py`). Não medido |
| **Ajuste de pesos** | SA-PINN, GradNorm, NTK, LossRatio | PhysicsNeMo-Sym: agregadores `grad_norm`, `res_norm`, `homoscedastic`, `lr_annealing`, `soft_adapt`, `relobralo` e opção NTK (`training.ntk`) | Não é exclusivo. Medido: o balanceamento adaptativo do `InversePINN` (hoje opcional, `weighting="adaptive"`) era a causa principal do seu erro de ν no problema inverso ([trilha C](../../benchmarks/vs_frameworks/REPORT.md#c-inverse-burgers-ν-from-2000-samples-with-1--noise-5-seeds-3000-iterations)) |
| **Validação física** | ConservationCheck, BoundaryCheck, SymmetryCheck | ❌ | PINNeAPPle exclusivo |
| **Problem design** | ProblemBuilder → ProblemSpec → Arena pipeline | Scripts Python | PINNeAPPle exclusivo |
| **Presets** | 35+ (CFD, estrutural, térmico, finanças, fármacos…) | CFD focado | PINNeAPPle mais amplo |
| **YAML pipeline** | `run_full_pipeline(config.yaml)` | Scripts Python | PINNeAPPle mais simples |
| **Export** | ONNX, TorchScript, checkpoint | ONNX, TensorRT, Triton | PhysicsNeMo melhor deploy NVIDIA |
| **Solvers** | FEniCS, OpenFOAM, FDM, FEM, SPH, LBM | OpenFOAM (básico) | PINNeAPPle mais completo |
| **Licença** | Open source (Apache 2.0) | Open source (Apache 2.0) | Ambos abertos |

---

## O que o benchmark mediu

Resumo de [`benchmarks/vs_frameworks/REPORT.md`](../../benchmarks/vs_frameworks/REPORT.md) (mediana [mín, máx] sobre seeds; CPU, 1 thread):

* **Mesma configuração (A1: mesma rede 4×50, pesos iniciais, pontos e perda, 2000 iterações).** PINNeAPPle, DeepXDE
  e PhysicsNeMo-Sym fazem a mesma conta: mesma perda inicial até 6 dígitos e erros iguais dentro da dispersão entre
  seeds. Custo por iteração: PINNeAPPle e DeepXDE a 4–11 % um do outro; Sym 1,4–1,9× mais caro em CPU.
* **Receita de cada framework (A2, 240 s de CPU).** Com os padrões de `pp.solve`, o PINNeAPPle teve o menor erro em
  Poisson (0,0013 contra 0,0134 do DeepXDE) e Helmholtz (0,0036 contra 0,0085), empatou em Burgers (0,020 contra
  0,017) e foi pior em Allen-Cahn (0,59 contra 0,49; todos falham). A receita oficial do Sym (rede 6×512) roda só
  70–94 passos em 240 s de CPU e falha nos quatro problemas; isso diz respeito à CPU, não à GPU.
* **FNO em Darcy (B, 12 épocas).** Ver a linha "Operadores neurais" acima.
* **Problema inverso (C, ν de Burgers a partir de 2000 amostras com 1 % de ruído, 3000 iterações).** O PINNeAPPle é
  claramente pior: o `InversePINN` erra ν por 12× (erro relativo 12,5 [11,0, 13,6]), contra 2,16 [1,90, 2,74] do
  DeepXDE e 1,32 [0,24, 1,83] do Sym (cujo ν é a média de um campo aprendido, negativa em 4 de 5 seeds). Depois das
  correções do `InversePINN` (parada antecipada sobre a perda não ponderada, sem fallback silencioso do resíduo) o
  resultado com os padrões é o mesmo: 12,5 [11,0, 13,7]. O diagnóstico aponta o balanceamento adaptativo PDE/dados
  como causa. Por isso o `InversePINN` passou a usar pesos fixos iguais por padrão (`weighting="equal"`; o esquema
  adaptativo continua como `weighting="adaptive"`). Rodada de novo com o novo padrão (5 seeds, 3000 iterações, mesmo
  harness): erro de ν 3,41 [2,99, 3,90] e erro do campo 0,112 [0,103, 0,133], contra 12,5 e 0,38 antes. Mais perto do
  DeepXDE (2,16; campo 0,079), mas ainda atrás: ν continua 4,4× grande demais.

## Estado da integração com PhysicsNeMo

* `pinneapple_neural.physicsnemo_bridge`: adaptadores para um modelo treinado no PhysicsNeMo (padrões 1 e 2 abaixo:
  UQ, digital twin, pontos de active learning), com testes próprios (`tests/test_physicsnemo_bridge.py`). Não foi
  exercitado pelo benchmark.
* `pinneapple_tools.nvidia.physicsnemo.MLPSurrogate`: surrogates tabulares `physicsnemo.models.mlp.FullyConnected`
  treinados e verificados pelo PINNeAPPle.
* `pinneapple_tools.benchmark_suite.backends.physicsnemo_sym`: não importava com o PhysicsNeMo-Sym 2.4.0 (importava
  `ModulusConfig`, que não existe mais), dizia que o pacote não estava instalado e não devolvia um preditor.
  Corrigido; hoje é verificado só por um teste de fumaça de 3 passos em CPU
  (`tests/pinneapple_tools/test_physicsnemo_sym_backend.py`). A acurácia não foi medida.
* Não há caminho que resolva um `pp.PhysicalProblem` com o solver do PhysicsNeMo-Sym: `pp.solve` não tem método
  PhysicsNeMo, e o benchmark precisou de um runner próprio para cada framework.
* O padrão 3 (MeshGraphNet validado pelo PINNeAPPle) existe só como script de exemplo
  (`06_combined_meshgraphnet_valid/`).

---

## Catálogo completo de modelos — PINNeAPPle (92 modelos)

### PINNs — Physics-Informed Neural Networks (8)

| Classe | Arquivo | Descrição |
|---|---|---|
| `VanillaPINN` | `pinns/vanilla.py` | PINN base — MLP com physics residual |
| `InversePINN` | `pinns/inverse.py` | Identifica parâmetros desconhecidos da PDE |
| `PIELM` | `pinns/pielm.py` | Physics-Informed ELM — treinamento linear rápido |
| `PINNLSTM` | `pinns/pinn_lstm.py` | PINN com backbone LSTM para séries temporais |
| `PINNsFormer` | `pinns/pinnsformer.py` | PINN com Transformer encoder |
| `VPINN` | `pinns/vpinn.py` | Variational PINN — formulação fraca da PDE |
| `XPINN` | `pinns/xpinn.py` | Extended PINN — decomposição de domínio em subdomínios |
| `XTFC` | `pinns/xtfc.py` | eXtreme Theory of Functional Connections — satisfação exata de BCs |

### Operadores Neurais (6)

| Classe | Arquivo | Descrição |
|---|---|---|
| `FourierNeuralOperator` / `FNO` | `neural_operators/fno.py` | Aprende operadores via camadas espectrais |
| `DeepONet` | `neural_operators/deeponet.py` | Branch + trunk network para operadores |
| `GalerkinNeuralOperator` / `GNO` | `neural_operators/gno.py` | Operador via atenção de Galerkin |
| `PhysicsInformedNeuralOperator` / `PINO` | `neural_operators/pino.py` | FNO com restrições de PDE |
| `UniversalUNO` | `neural_operators/uno.py` | U-Net + operador neural |
| `MultiScaleDeepONet` | `neural_operators/ms_deeponet.py` | DeepONet multi-escala |

### Graph Neural Networks (5)

| Classe | Arquivo | Descrição |
|---|---|---|
| `GraphNeuralNetwork` / `GNN` | `graphnn/gnn.py` | Message passing em malhas não-estruturadas |
| `EquivariantGNN` / `EGNN` | `graphnn/equivariant_gnn.py` | GNN com equivariância SO(3) |
| `GraphNeuralODE` | `graphnn/gnn_ode.py` | GNN com integrador de ODE contínuo |
| `SpatiotemporalGNN` / `STGNN` | `graphnn/spatiotemporal_gnn.py` | GNN para dados espaço-temporais |
| `GraphCast` | `graphnn/graphcast.py` | Previsão meteorológica estilo DeepMind |

### Modelos de Tempo Contínuo / Neural ODEs (11)

| Classe | Arquivo | Descrição |
|---|---|---|
| `NeuralODE` | `continuous/neural_ode.py` | Rede neural como campo de vetores |
| `LatentODE` | `continuous/latent_ode.py` | VAE + ODE no espaço latente |
| `ODERNN` | `continuous/ode_rnn.py` | RNN com transições ODE entre observações |
| `NeuralCDE` | `continuous/neural_cde.py` | Controlled differential equation para dados irregulares |
| `NeuralSDE` | `continuous/neural_sde.py` | SDE estocástica como modelo generativo |
| `HamiltonianNeuralNetwork` / `HNN` | `continuous/hamiltonian.py` | Aprende Hamiltoniano — conserva energia |
| `SymplecticODENet` | `continuous/symplectic_ode.py` | ODE com integradores simpléticos |
| `SymplecticRNN` | `continuous/symplectic_rnn.py` | RNN simpléctica para dinâmicas conservativas |
| `BayesianRNN` | `continuous/bayesian_rnn.py` | RNN com pesos bayesianos |
| `DeepStateSpaceModel` / `DSSM` | `continuous/deep_state_space.py` | SSM com encoder/decoder profundo |
| `NeuralGaussianProcess` / `NGP` | `continuous/neural_gp.py` | Processo Gaussiano com kernel neural |

### Recurrent / Sequence Models (5)

| Classe | Arquivo | Descrição |
|---|---|---|
| `LSTMModel` | `recurrent/lstm.py` | LSTM padrão para séries temporais |
| `BiLSTMModel` | `recurrent/lstm.py` | LSTM bidirecional |
| `GRUModel` | `recurrent/gru.py` | GRU — alternativa leve ao LSTM |
| `BiGRUModel` | `recurrent/gru.py` | GRU bidirecional |
| `Seq2SeqRNN` | `recurrent/seq2seq.py` | Encoder-decoder para previsão de sequências |

### Transformers para Séries Temporais (6)

| Classe | Arquivo | Descrição |
|---|---|---|
| `VanillaTransformer` | `transformers/transformer.py` | Transformer padrão para TS |
| `Informer` | `transformers/informer.py` | Atenção esparsa para sequências longas |
| `TemporalFusionTransformer` / `TFT` | `transformers/tft.py` | Multi-horizon forecasting com variáveis externas |
| `Autoformer` | `transformers/autoformer.py` | Decomposição sazonal + atenção autocorrelação |
| `FEDformer` | `transformers/fedformer.py` | Frequência + Transformer |
| `TimesNet` | `transformers/timesnet.py` | TS como imagem 2D |

### Autoencoders (6)

| Classe | Arquivo | Descrição |
|---|---|---|
| `DenseAutoencoder` | `autoencoders/dense_ae.py` | AE denso padrão |
| `Autoencoder2D` | `autoencoders/ae_2d.py` | AE convolucional 2D |
| `VariationalAutoencoder` / `VAE` | `autoencoders/vae.py` | VAE com reparametrização |
| `KAEAutoencoder` / `KAE` | `autoencoders/kae.py` | Koopman AE — dinâmicas lineares no espaço latente |
| `PhysicsInformedKoopmanAE` | `autoencoders/koopman_pi_ae.py` | Koopman AE com restrições físicas |
| `AEROMHybrid` | `autoencoders/ae_rom_hybrid.py` | AE + ROM para redução de ordem |

### Modelos de Ordem Reduzida — ROM (6)

| Classe | Arquivo | Descrição |
|---|---|---|
| `POD` | `rom/pod.py` | Proper Orthogonal Decomposition — SVD de snapshots |
| `DynamicModeDecomposition` / `DMD` | `rom/dmd.py` | DMD — modos e valores próprios de Koopman |
| `HAVOK` | `rom/havok.py` | Hankel Alternative View of Koopman |
| `OperatorInference` / `OpInf` | `rom/opinf.py` | ROM intrusivo por inferência de operadores |
| `ROMHybrid` | `rom/rom_hybrid.py` | ROM + rede neural para closure |
| `DeepUQROM` | `rom/deep_uq_rom.py` | ROM com quantificação de incerteza |

### Reservoir Computing (6)

| Classe | Arquivo | Descrição |
|---|---|---|
| `ExtremeLearningMachine` / `ELM` | `reservoir_computing/elm.py` | Rede com pesos fixos + regressão linear |
| `RBFNetwork` | `reservoir_computing/rbf.py` | Rede de funções de base radial |
| `HybridRBFNetwork` | `reservoir_computing/hybrid_rbf.py` | RBF + componente neural treinável |
| `EchoStateNetwork` / `ESN` | `reservoir_computing/esn.py` | Reservoir computing com reservatório fixo |
| `ESNRC` | `reservoir_computing/esn_rc.py` | ESN com readout classificador |
| `KoopmanOperator` | `reservoir_computing/koopman.py` | Operador de Koopman para dinâmicas não-lineares |

### Modelos Clássicos de Séries Temporais (7)

| Classe | Arquivo | Descrição |
|---|---|---|
| `VAR` | `classical_ts/var.py` | Vector Autoregression |
| `ARIMA` | `classical_ts/arima.py` | ARIMA com diferenciação |
| `KalmanFilter` | `classical_ts/kalman.py` | Filtro de Kalman linear |
| `ExtendedKalmanFilter` / `EKF` | `classical_ts/ekf.py` | Kalman não-linear (linearização local) |
| `UnscentedKalmanFilter` / `UKF` | `classical_ts/ukf.py` | Kalman não-linear (sigma points) |
| `EnsembleKalmanFilter` / `EnKF` | `classical_ts/enkf.py` | Kalman por ensemble Monte Carlo |
| `TCN` | `classical_ts/tcn.py` | Temporal Convolutional Network — receptive field explícito |

### Convolucionais (3)

| Classe | Arquivo | Descrição |
|---|---|---|
| `Conv1DModel` | `convolutions/conv1d.py` | CNN 1D para séries e campos 1D |
| `Conv2DModel` | `convolutions/conv2d.py` | CNN 2D para campos espaciais |
| `Conv3DModel` | `convolutions/conv3d.py` | CNN 3D para campos volumétricos |

### Physics-Aware (2)

| Classe | Arquivo | Descrição |
|---|---|---|
| `PhysicsAwareNeuralNetwork` / `PANN` | `physics_aware/pann.py` | Rede com termos físicos explícitos embutidos |
| `StructurePreservingNetwork` / `SPN` | `physics_aware/spn.py` | Preserva estrutura geométrica (Lie groups, etc.) |

### Baselines / Benchmarks (5)

| Classe | Arquivo | Descrição |
|---|---|---|
| `GenericMLP` | `benchmarks/generic_pinn_models.py` | MLP baseline |
| `GenericFourierMLP` | `benchmarks/generic_pinn_models.py` | MLP com encoding de Fourier |
| `GenericSIREN` | `benchmarks/generic_pinn_models.py` | SIREN — ativações sinusoidais |
| `GenericResMLP` | `benchmarks/generic_pinn_models.py` | MLP com skip connections |
| `GenericLinear` | `benchmarks/generic_pinn_models.py` | Baseline linear |

---

## Presets de problemas (35+)

### Acadêmicos / Didáticos
`laplace_2d` · `poisson_2d` · `burgers_1d`

### CFD / Navier-Stokes
`ns_incompressible_2d` · `ns_incompressible_3d`

### Industrial
`steady_heat_conduction_3d` · `transient_heat_3d` · `linear_elasticity_3d` · `darcy_pressure_only_3d` · `helmholtz_acoustics_3d` · `wave_ultrasound_3d` · `reaction_diffusion_2d`

### Estrutural
`plane_stress_2d` · `plane_strain_2d` · `von_mises_2d` · `linear_elasticity_3d` · `drill_pipe_torsion` · `thermoelasticity_2d`

### Engenharia (18)
`rocket_nozzle_cfd` · `rocket_structural` · `aircraft_wing_aerodynamics` · `aircraft_wing_structural` · `car_external_aero` · `car_brake_thermal` · `car_suspension_fatigue` · `cpu_heatsink_thermal` · `pcb_thermal` · `fan_cooler_cfd` · `industrial_furnace_thermal` · `refractory_lining` · `furnace_combustion_zone` · `datacenter_airflow_2d` · `datacenter_server_thermal` · `datacenter_cfd_3d` ·…

### Multidisciplinar
`climate_atmosphere_2d` · `climate_ocean_gyre` · `crystal_phonon` · `material_fracture_2d` · `black_scholes_1d` · `heston_pde_2d` · `pk_two_compartment` · `drug_diffusion_tissue` · `sir_epidemic` · `opinion_dynamics_2d`

```python
from pinneapple_physics.pde_environment import list_presets
print(list_presets())  # lista todos os presets disponíveis
```

---

## Padrões de integração

### Padrão 1 — PhysicsNeMo treina, PINNeAPPle opera
```
Dados CFD → PhysicsNeMo FNO (throughput, GPU) → checkpoint
         → PINNeAPPle carrega → MCDropoutWrapper (UQ)
         → DigitalTwin + MockStream (sensores)
         → Anomaly detection + EKF assimilation
         → REST API via pinneapple_serve
```
**Melhor para**: surrogate de alta fidelidade + digital twin em produção.

### Padrão 2 — PINNeAPPle seleciona pontos, PhysicsNeMo retreina
```
PINNeAPPle ResidualBasedAL → pontos hard de collocation
PhysicsNeMo retreina FNO nesses pontos (GPU rápido)
→ loop: AL seleciona → PhysicsNeMo retreina → AL seleciona...
```
**Melhor para**: reduzir número de simulações necessárias.

### Padrão 3 — PhysicsNeMo para geometria complexa, PINNeAPPle valida
```
Malha STL → PhysicsNeMo MeshGraphNet (inferência em malha não-estruturada)
          → PINNeAPPle PhysicsValidator (conservação de massa, BCs, simetria)
          → PINNeAPPle export_onnx() para deploy C++
```
**Melhor para**: validar que GNNs não predizem soluções não-físicas.

---

## Exemplos neste diretório

| # | Diretório | O que mostra | Ferramentas |
|---|---|---|---|
| 01 | `01_pinneapple_uq_digital_twin/` | UQ + Digital Twin em equação do calor | PINNeAPPle **only** |
| 02 | `02_pinneapple_meta_transfer/` | Meta-aprendizado Reptile + Transfer Learning em Burgers | PINNeAPPle **only** |
| 03 | `03_pinneapple_active_weight_sched/` | Active collocation RAD + SA-PINN em Laplace 2D | PINNeAPPle **only** |
| 04 | `04_physicsnemo_fno_operator/` | FNO operator learning + multi-GPU | PhysicsNeMo (+ fallback) |
| 05 | `05_combined_fno_digital_twin/` | PhysicsNeMo FNO surrogate → PINNeAPPle UQ + DT | **Combined** |
| 06 | `06_combined_meshgraphnet_valid/` | PhysicsNeMo GNN → PINNeAPPle validação + export | **Combined** |

---

## Instalação

```bash
# PINNeAPPle (este repositório)
pip install -e .

# PhysicsNeMo (opcional — exemplos têm fallback)
pip install nvidia-physicsnemo

# Dependências opcionais para os exemplos
pip install matplotlib scipy
```

## Como responder à pergunta "qual a diferença?"

> **PhysicsNeMo** é uma fábrica de surrogates de alto desempenho otimizada para hardware NVIDIA.
> **PINNeAPPle** é um pipeline completo de pesquisa à produção, com UQ, digital twins, meta-aprendizado
> e validação física — cobrindo domínios muito além de CFD.
>
> **Juntos**: o PhysicsNeMo é feito para treinar rápido em GPUs NVIDIA (não medido aqui); o PINNeAPPle adiciona
> incerteza quantificada, digital twin e validação física a um modelo treinado. As medições em CPU estão em
> [`benchmarks/vs_frameworks/REPORT.md`](../../benchmarks/vs_frameworks/REPORT.md).
