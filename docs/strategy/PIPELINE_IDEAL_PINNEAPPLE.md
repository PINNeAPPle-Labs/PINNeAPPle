# O pipeline ideal do PINNeAPPle, adaptado do da Luminary

Escrito em 2026-11-28. A referência é `docs/strategy/LUMINARY_PIPELINE_VS_OURS.md` (o que eles fazem, item a item, com
as fontes). Aqui a pergunta é outra: **como ter o mesmo pipeline com as nossas tecnologias** — e o que a nossa stack
permite fazer que a deles não faz. Cada item traz o que já existe no repositório (com o módulo), o que falta, e o que
construir.

Três regras de adaptação, declaradas antes para valerem em todos os itens:

1. **Nada que exija capital deles.** Solver CFD GPU-native próprio, cluster de GPU e frota de CAD connectors são a
   aposta de investimento da Luminary, não a nossa. Onde o item depende disso, a adaptação é *usar* o que existe
   (OpenFOAM nativo, PhysicsNeMo, CAD via STEP) e *medir* o que se ganha.
2. **O que já está no repositório entra antes do que é novo.** Temos morfismo, troca de CAD, solvers verificados,
   registry de arquiteturas, cards e o VeriPhysics. Boa parte do pipeline deles existe aqui com outro nome e sem fio
   ligando as pontas.
3. **Cada estágio tem de rodar numa máquina de mesa**, com a nuvem como opção e não como pré-requisito. É o que faz um
   piloto custar horas em vez de semanas — e é o que a campanha dos 11 espaços de projeto provou.

## 1. Estratégia de amostragem do espaço de projeto

**Deles:** definem a estratégia de amostragem do espaço (geométrico *e* de operação) antes de simular qualquer coisa.

**Temos:** o espaço de projeto já é **dado, não script** — `benchmarks/lpm_recipes.json` diz geometrias, condições,
épocas e largura por família, e `draw()` (`pinneapple_neural/lpm_pipeline.py:280`) amostra com rejeição do que a
família recusa. O desenho do espaço de cada família está em `pinneapple_neural/lpm_families*.py` (faixas por
parâmetro). Para amostragem de verdade há `pinneapple_physics/active_design.py`.

**Falta:** a amostragem é **uniforme e fixa**; não há desenho experimental (LHS/Sobol) nem amostragem **ativa**.

**Construir:** `morph.space` como objeto versionado (ver §2) + `lpm_recipes` ganhando `sampler: {kind: lhs|sobol|active,
seed}`. E o laço que é upgrade direto sobre eles: **o OOD do card escolhe os próximos corpos**. Onde o modelo tem
score alto perto do limiar, simula-se mais denso; onde está confortável, não. Custo: dias. Ganho: o mesmo erro com
menos simulações — que é justamente o que falta quando a referência custa segundos por design.

## 2. Geometria por morfismo paramétrico com CAD

**Deles:** morfismo + integrações diretas (Onshape, nTop, Blender, SolidWorks) e formatos de intercâmbio.

**Temos:** `pinneapple_design/geometry/morph/` com `surface.py`, `rbf.py`, `modes.py`, `lattice.py`, `space.py` e
**`validity.py`** (a checagem que eles fazem à mão); `pinneapple_design/geometry/io/` com `step.py`, `iges.py`,
`cadquery_bridge.py`, `meshio_bridge.py`, `trimesh_bridge.py`, `sdf_meshing.py`; `pinneapple_design/geometry/brep/`;
e `pinneapple_blender` para o caminho Blender.

**Falta:** conector **direto** com Onshape/SolidWorks/nTop (hoje é STEP/CadQuery), e o morfismo não está ligado ao
pipeline LPM — cada família desenha seus parâmetros em código.

**Construir:** (a) `morph.space` como a receita geométrica de cada família — base CAD + modos + faixas + `validity` —
de modo que "amostrar 900 corpos" seja uma chamada, não código novo; (b) conector Onshape primeiro (tem API REST
pública e é o mesmo caminho que eles usam), depois SolidWorks; (c) o gate de validade roda **antes** de simular:
corpo aberto, autointersecção e volume negativo não entram no dataset. Custo: (a) dias, (b) 1–2 semanas por CAD.

## 3. Solver com healing, malha adaptativa e banco versionado

**Deles:** solver GPU-native, *geometry healing*, malha adaptativa, tudo anotado e versionado num data lake.

**Temos:** solvers verificados por identidade — `pinneapple_simulation/immersed` (FEM imerso com Nitsche/ghost penalty),
`tet_fem` (P2 conforme com gmsh), VLM + painel 2-D para asa, `fvm`; **OpenFOAM nativo no Mac** (cask, ~150× mais rápido
que Docker amd64); `manifest.json` com **SHA-256 de cada arquivo** e o solver/citação/hipóteses no manifesto.

**Falta:** healing automático; malha adaptativa; e o "banco": hoje o dataset é um diretório com manifesto — não há
catálogo que responda "quais datasets existem, com que solver, versão e hash".

**Construir:** (a) **catálogo** pequeno e honesto: SQLite/Parquet com `dataset_id`, família, solver + versão, nº de
designs, hash do manifesto, data e máquina — o card passa a referenciar `dataset_id` em vez de um caminho; (b) healing
mínimo com trimesh/CadQuery + gate (fechar buraco, remover casca, normalizar norma das normais) com o que reparou
registrado no manifesto; (c) AMR guiada pela solução (já está no backlog como item de refinamento adaptativo) —
começar pelo `immersed`, onde o erro é estimável por célula. Custo: (a) dias, (b) 1 semana, (c) semanas.

## 4. Medições físicas como âncoras de validação

**Deles:** túnel e sensores entram como âncoras, nunca como treino.

**Temos:** é exatamente o desenho do **VeriPhysics**: `validation_case.py`, `evidence_package.py`, `dossier.py`,
`ledger.py`. E o conceito de âncora já existe no produto (`pp.zoo.anchors`, família `PINNeAPPle Anchor`). Do lado de
aquisição, o `engineering-api` já tem conectores com agente de pull (`app/connectors`, tela `#/conectores`).

**Falta:** ingerir uma medição (CSV/planilha do cliente) e transformá-la em `ValidationCase`; e o card não tem um gate
"âncora".

**Construir:** `validation_case` a partir de medição + gate `anchor` no card, com a mesma regra do resto: a âncora
**não** entra no treino, entra como evidência e pode reprovar o modelo. É o item mais alinhado com o que já temos, e o
que mais falta para um piloto com cliente — o dado dele é o que dá valor ao modelo e a âncora é o que o faz confiável.

## 5. Cache de índices de vizinhança (o ganho de 50×)

**Deles:** descobriram que o pré-processamento custava >100× o treino por recalcular a relação entre pontos vizinhos a
cada passo; cachear deu **50× por época** e 8× no carregamento.

**Temos:** o **mesmo defeito**. `_multiscale` (`pinneapple_neural/lpm.py:79`) recalcula `torch.cdist(query, points)` em
todo forward — chamado em `lpm.py:313`, dentro de `_members_batch`. Os designs **não mudam** durante o fit.

**Falta:** o cache.

**Construir (feito):** no estágio 3, as features de vizinhança são calculadas **uma vez por design** e guardadas no
`.npz`; `_multiscale` lê do batch em vez de recalcular. Medido nos dados da campanha: **1,6× por época no
`pp_wing3d`** (960 pontos, batch 24) e **2,2× no `pp_fin`** (111 pontos), com o histórico de loss idêntico até 1e-6.
**Não** são os 50× deles: o gargalo deles era o carregamento famintos de GPU; o nosso `cdist` é só parte do forward.
O número é este, não a analogia.

## 6. PhysicsNeMo (DoMINO/GeoTransolver), arquitetura própria, templates, GPU e fine-tune

**Deles:** suportam DoMINO e GeoTransolver, mais a SMART; templates por aplicação; treino em cluster GPU com
paralelismo de dados e modelo; partem de pesos pré-treinados e fazem fine-tune.

**Temos:** `pinneapple_neural/physicsnemo_bridge/` (os padrões "PhysicsNeMo treina, PINNeAPPle opera/UQ" e
"PINNeAPPle seleciona por active learning, PhysicsNeMo retreina"), `pinneapple_neural/architectures/` com registry de
arquiteturas, e o nosso backbone de LPM (atenção ponto-a-superfície, Fourier, DeepSets, ensembles) — que já é da mesma
linhagem metodológica.

**Falta:** DoMINO/GeoTransolver de verdade (atenção: o `domino.py` que existe é **outra coisa** — PINN de decomposição
de domínio, o nome confunde), paralelismo, e o fluxo de fine-tune.

**Construir, em ordem de custo:**
- **fine-tune entre famílias** (dias): carregar os pesos de uma família irmã, congelar a parte geométrica
  (`freeze_geometry=True` já existe em `fit`) e continuar. As sete famílias prontas viram os pesos iniciais da 12ª —
  é a diferença deles que mais barato se fecha.
- **templates de hiperparâmetro como dados** (dias): o `lpm_recipes.json` cresce para carregar `model: {backbone, width,
  attn_layers, radii}` e cada família ganha o seu, em vez de constantes no código.
- **DoMINO pelo bridge** (semanas, depende de GPU): treinar com PhysicsNeMo onde houver GPU (nuvem pontual) e trazer os
  pesos para o nosso pipeline operar, comparar com o nosso backbone no mesmo dataset — a comparação é nossa, o
  benchmark deles é interno.
- **paralelismo** (só quando houver GPU): no MacBook o honesto é CPU multiprocess por família ou treino pontual em
  nuvem; MPS não tem DDP. Não fingir escala que a máquina não tem.

## 7. Conjunto de teste que evolui, registry e entrega

**Deles:** test suite que evolui; cada modelo vira entrada de registry (arquitetura + pesos + hiperparâmetros + versão
do dataset + características da distribuição); entrega por API, on-premise ou embarcada em CAD.

**Temos:** `pinneapple_models/registry.py` (registry de **arquiteturas**), `pinneapple_hub/verified_card.py` — um card
que **recusa** ser emitido quando nenhuma verificação computável rodou —, o `card.json` do LPM com split, baseline,
OOD e `trust.decision`, o `engineering-api` com rotas versionadas, e o **Studio** (viewer 3-D, apps, painel por
família) que é a nossa forma de "embarcado".

**Falta:** registry de **modelos treinados** com versão e distribuição; rota de inferência com OOD; teste que evolui.

**Construir:**
- **card v2** (dias): acrescentar `dataset_id`, `distribution` (média e quantis dos parâmetros de treino),
  `git_commit`, `sha256` do artefato. O resto já está lá.
- **registry de modelos** (dias): índice que lê os cards e responde "quais modelos existem, com que métrica, sobre que
  dataset, em que commit" — o `verified_card` já define a regra de recusa.
- **`/v1/lpm/predict`** (semanas): rota no `engineering-api` que devolve campo + OOD + veredito dos gates, não o campo
  sozinho.
- **test set que evolui**: cada família mantém os casos que já reprovaram (o `pp_airfoil` com 150 de 400 piores que o
  campo médio é um caso de teste, não uma vergonha) e o card diz se o modelo novo os passa.

## 8. Confiança — e onde o VeriPhysics nos põe à frente

**Deles:** score 0–1 calibrado a partir de um embedding do input, com tabela publicada de deslocamentos (0,99
in-distribution → 0,13 para geometria de carro), e derivação de ação.

**Temos:** duas coisas que a maioria não tem, e uma que falta.
- **OOD** (`lpm.py:347`): distância do código de geometria ao design de treino mais próximo, normalizada pelo quantil
  0,95 das distâncias de treino, mais checagem de faixa dos parâmetros. Mesma escolha deles (olhar o input), sem
  calibração.
- **Ensemble com bagging**: desvio entre cabeças por predição — incerteza que já sai em cada campo.
- **VeriPhysics** (`pinneapple_veriphysics/`): `surrogate_gates.py` com quatro gates numéricos — **accuracy**,
  **conservation** (o integral do campo conservado, na predição **e** no dado), **boundary** (desvio de cada condição de
  contorno declarada) e **residual** (a equação governante avaliada na predição por diferenças finitas); e
  `model_grade.py`, que audita um modelo **caixa-preta de qualquer framework** contra a equação que ele diz resolver.
  Mais `dossier.py`, `evidence_package.py`, `ledger.py`, `policy.py`, `decision.py`, `applicability.py`, `robustness.py`
  e `maturity.py`.

**Falta:** calibrar o score; e ligar o VeriPhysics ao card do LPM.

**Construir:** (a) calibrar o OOD com uma tabela de deslocamentos nossa (Mach fora, geometria fora, corpo de outra
família) e publicar a tabela como eles publicam — em dias, e transforma o número num semáforo que cliente entende;
(b) rodar `surrogate_gates` no campo predito e escrever os quatro números no card; (c) oferecer `model_grade` como
auditoria de modelo de terceiro (o cliente que já tem um surrogate e quer saber se ele obedece à equação).

## O que a nossa stack permite fazer que a deles não faz

1. **Verificação, não só confiança.** Eles entregam "0,87 de confiança". Nós podemos entregar isso **e** "o resíduo da
   equação na predição é 3 %, a conservação desvia 0,4 %, a condição de contorno erra 1,2 %". Um score diz o quanto o
   modelo se afastou do que viu; os gates dizem se a resposta **obedece à física** — e é a segunda pergunta que decide
   um review de engenharia. É o item que mais nos diferencia.
2. **Baseline em todo card.** O erro do campo constante ao lado do erro do modelo. É o que separa "aprendeu" de
   "rodou", e nenhum fornecedor publica isso.
3. **Split por geometria declarado e testado**, com a razão escrita no código — a fonte de vazamento mais comum nesse
   tipo de benchmark.
4. **`trust.decision` que nunca se auto-promove**: nenhum modelo nosso se declara verificado.
5. **Reprodutibilidade de mesa**: refazer uma família inteira, do dataset ao card, em 1–2 h numa máquina só.
6. **A camada de cima**: `pinneapple_intent` (texto → especificação de simulação), `pinneapple_advisor` (consultor de
   pipeline), `pinneapple_catalog.stages` (mapa por etapa). Eles entregam SDK e API; nós podemos entregar a pergunta
   em português virando caso verificado — que é o gargalo real da PME, não o treino.
7. **O Studio** como entrega embarcada: viewer 3-D com o campo, figuras por família, painel de campanha — sem CAD
   proprietário.
8. **Connectors**: puxar o dado do cliente (inclusive medição) em vez de pedir upload.

## O pipeline ideal, em oito estágios

```
0  Pergunta        pinneapple_intent / advisor        → spec + espaço de projeto declarado
1  Espaço          morph.space + lpm_recipes          → base CAD + modos + faixas + sampler (lhs/sobol/active)
2  Geometria       morph + io (STEP/Onshape) + validity → corpos válidos, com proveniência
3  Referência      immersed / tet_fem / VLM / OpenFOAM  → dataset + manifest com hash + catálogo + cache de vizinhos
4  Âncoras         connectors → validation_case         → medição do cliente como evidência (nunca treino)
5  Treino          LPM (+ physicsnemo_bridge)          → pesos, fine-tune de família irmã, templates por receita
6  Prova           split por geometria + baseline + OOD calibrado + VeriPhysics gates → card com veredito
7  Registro        card v2 + registry + /v1/lpm/predict → modelo versionado, com dataset e distribuição
8  Entrega         Studio / CAD / API                  → campo + confiança + gates, no lugar onde a decisão acontece
```

## Ordem de implementação

| fase | itens | o que fica verdadeiro no fim | custo |
|---|---|---|---|
| **0** | cache de vizinhança (§5), OOD calibrado (§8a) | a campanha de 11 famílias cai de ~34 h para uma noite, e o score vira semáforo | dias |
| **1** | VeriPhysics no card (§8b), fine-tune entre famílias (§6), card v2 + registry (§7) | todo modelo publica verificação, e a 12ª família parte de pesos | ~2 semanas |
| **2** | catálogo versionado + healing (§3), morfismo como receita (§2a), âncoras de medição (§4) | o dataset deixa de ser diretório e passa a ser ativo rastreável; medição do cliente entra como evidência | ~1 mês |
| **3** | conector Onshape (§2b), `/v1/lpm/predict` (§7), DoMINO/GeoTransolver por GPU (§6) | o pipeline deles, com a nossa prova em cima | quando houver cliente/GPU |

## O que não fazer

- **Solver CFD GPU-native próprio.** É a aposta de capital deles; o nosso diferencial está em cima do solver, não nele.
- **Registry, API pública e conectores CAD antes do primeiro piloto pago.** Custo de manutenção sem retorno.
- **Chamar os cinco estágios de adoção deles de "pipeline"** numa apresentação: são modelos de maturidade
  organizacional, não de treino, e a confusão é fácil de explorar contra nós.
- **Perseguir o número deles sem o dado deles**: 3 000 RANS de asa exigem o solver deles. O caminho honesto é
  fine-tune, cache e amostragem ativa — e publicar o baseline ao lado.
