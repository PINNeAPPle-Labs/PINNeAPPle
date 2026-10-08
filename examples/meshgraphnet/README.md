# MeshGraphNet no PINNeAPPle

Implementação: `pinneapple_neural/architectures/graphnn/mesh_graph_net.py` (rede encoder–processor–decoder,
registrada como `mgn` / `meshgraphnet`) e `mgn_dynamics.py` (receita de treino transiente de Pfaff et al.:
normalização, ruído de treino, alvo Δv, rollout autoregressivo com nós de contorno fixados).

## Experimentos que rodam (`examples/meshgraphnet/`)

| Script | O que valida | Dados |
|---|---|---|
| `01_synthetic_diffusion.py` | MGN aprende difusão de calor em malhas Delaunay aleatórias e supera o baseline "estado congelado" em rollout em malhas inéditas | gerado na hora |
| `02_cylinder_flow_deepmind.py` | Escoamento transiente em torno de cilindro (DeepMind `cylinder_flow`, Pfaff et al. 2021), orçamento reduzido | download de um prefixo do TFRecord |

### Resultados medidos (CPU, máquina carregada)

| Experimento | Configuração | RMSE rollout | Baseline "estado inicial congelado" |
|---|---|---|---|
| 01 difusão sintética | 24 malhas de treino, 150 nós, 4 camadas MP, 2500 passos, rollout 20, 6 malhas de teste | 0,0297 | 0,1104 |
| 02 cylinder_flow | 11 trajetórias de treino, 64 ocultos, 6 camadas MP, 3000 passos, lr 1e-3→1e-5, ruído 0,003, rollout 50, **2** trajetórias de validação | 0,0711 (1 passo: 0,0265) | 0,0962 |

O sintético valida a implementação. O cylinder_flow é **só** uma prova de que o pipeline aprende com dados reais
(~26% melhor que o baseline, com poucos dados e 3000 passos); **não** reproduz o paper
(1000 trajetórias, 15 camadas, 128 ocultos, ~10⁶ passos, ruído 0,02). Hiperparâmetros foram escolhidos
para o orçamento reduzido: uma primeira tentativa com lr 1e-4→1e-6 e ruído 0,02 não convergiu em 1000 passos
(loss normalizada ~5) e foi interrompida.

## Literatura usada como referência de experimentos que funcionam

- Pfaff et al., ICLR 2021, *Learning Mesh-Based Simulation with Graph Networks* (arXiv 2010.03409): cylinder_flow,
  airfoil, flag, deforming_plate. Código/dados: github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets
- Sanchez-Gonzalez et al., ICML 2020, *Learning to Simulate Complex Physics with Graph Networks* (arXiv 2002.09405):
  Water/Sand/Goop (base do `lagrangian_mgn`).
- Exemplos do PhysicsNeMo (`examples/cfd/{vortex_shedding_mgn,stokes_mgn,lagrangian_mgn}`).

## Scripts de paridade com o PhysicsNeMo (criados, **não executados**) — `physicsnemo_parity/`

| PhysicsNeMo | PINNeAPPle | Dataset |
|---|---|---|
| `vortex_shedding_mgn` | `vortex_shedding_mgn.py` | DeepMind cylinder_flow completo (13,6 GB) |
| `stokes_mgn` | `stokes_mgn.py` | NGC `physicsnemo_datasets_stokes_flow` (FEniCS, .vtp) |
| `lagrangian_mgn` | `lagrangian_mgn.py` | DeepMind Learning-to-Simulate (Water etc.) |

Fora do escopo desta entrega: `vortex_shedding_mesh_reduced` (modelo reduzido + transformer) e
`stokes_mgn/pi_fine_tuning*.py` (fine-tuning por resíduo de PDE).

## CANTO (NVIDIA, arXiv 2609.36806)

Operador transformer *CAD-native*: tokeniza patches NURBS (pontos de controle, pesos, knots) e prediz
campos de superfície/volume em pontos de consulta arbitrários (AhmedML, WindsorML, DrivAerML, HiLiftAeroML).
Não é um GNN e não depende de malha; a página não cita código público. Relação com MGN: é a alternativa
sem malha para aerodinâmica externa. Integração sugerida: backbone em `neural_operators`/ponte opcional
(padrão `noether_bridge`), com o MGN como baseline de malha na mesma suíte de benchmark.
