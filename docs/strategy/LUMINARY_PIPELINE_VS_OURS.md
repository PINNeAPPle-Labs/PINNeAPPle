# O pipeline da Luminary e o nosso: o que eles fazem, o que fazemos, e onde está a diferença

Documento de comparação, escrito em 2026-11-28. Lado deles: o que a Luminary Cloud publica (links na §1). Lado nosso:
o que está no código, com arquivo e linha, e o que a campanha mediu até 28/11 (7 de 11 espaços de projeto com card).
Nada aqui é estimativa nossa sobre eles: onde não há publicação, está dito que não há.

## 1. Fontes

| Data | Artigo | O que dá para tirar |
|---|---|---|
| 2025-03-25 | [Open Sourcing a 50x Speed Up in Physics AI Model Training](https://luminary.ai/resources/open-sourcing-a-50x-speed-up-in-physics-ai-model-training/) | o gargalo real do treino deles era o pré-processamento: cache de índices de vizinhança, 50× por época, 8× no carregamento |
| 2025-07-17 | [Physics AI Factory: the end-to-end platform](https://luminary.ai/resources/luminarys-physics-ai-factory-the-end-to-end-platform-to-build-deploy-physics-ai-models/) | o pipeline completo deles, estágio por estágio — a fonte principal deste documento |
| 2026-04-08 | [Demystifying Physics AI: the five stages to scale](https://luminary.ai/resources/demystifying-physics-ai-the-five-stages-to-scale/) | os "cinco estágios" deles são de **adoção organizacional**, não de treino (ver §2.8) |
| 2026-07-28 | [Luminary-SMART](https://luminary.ai/resources/luminary-smart-the-new-state-of-the-art-model-architecture-for-large-physics-models/) | arquitetura própria, benchmark interno contra DoMINO e GeoTransolver |
| 2026-08-11 | [Quantifying the Unknown: OOD & UQ](https://luminary.ai/resources/quantifying-the-unknown-ood-detection-and-uncertainty-quantification-for-large-physics-models/) | o mecanismo de confiança deles, com tabela de shifts medidos |
| — | [Modelos e datasets SHIFT](https://luminary.ai/models/) | tamanho publicado de cada modelo: 700 a 5 000 simulações de alta fidelidade |

## 2. O pipeline deles

### 2.1 Dados: geração antes de tudo

O pipeline começa na **geração**, não no treino. Eles definem a estratégia de amostragem do espaço de projeto —
variações geométricas *e* condições de operação — e produzem as geometrias por **parametrização, morfismo de
superfície e troca de configuração**, com integrações diretas a CAD (Onshape, nTop, Blender, SolidWorks) além dos
formatos de intercâmbio. Daí:

- solver CFD **GPU-native próprio**, com *geometry healing* e *adaptive mesh*;
- orquestração massiva de casos, cada resultado **anotado automaticamente com os metadados de entrada**;
- ingestão de dados legados de solvers comerciais e open-source, quando existem;
- medições físicas (túnel de vento, sensores) **não** como treino, mas como **âncoras de validação**: são esparsas
  demais para treinar sozinhas.

Tudo cai num **data lake centralizado e versionado**, para rastreabilidade.

Escala publicada por modelo: SHIFT-SUV >1 000 DES transientes; SHIFT-Truck >700 DDES em malhas de 122 M células;
SHIFT-Crash ~5 000 simulações de impacto; SHIFT-Battery >1 300 CHT; SHIFT-Wing >3 000 RANS.

### 2.2 Pré-processamento: um cache que valeu 50×

O achado mais útil de todos, e o mais fácil de subestimar: **o preparo e o carregamento de dados consumiam mais de
100× o recurso do treino em si**. A causa era o cálculo das relações espaciais entre pontos vizinhos — que **não muda
durante o treino** e era recalculado a cada passo. Eles passaram a pré-processar uma vez, guardar os índices de
vizinhança e carregar em paralelo com múltiplos workers: **épocas até 50× mais rápidas**, carregamento 8× mais rápido,
um caso de 5 dias virou 2 horas. Contribuído de volta ao PhysicsNeMo.

### 2.3 Configuração e arquitetura

- **Arquiteturas suportadas**: convolucionais, grafos, recorrentes (séries temporais) e, como recomendação para
  aerodinâmica externa, **DoMINO** (PhysicsNeMo, point-cloud multi-escala com SDF e encodings posicionais) e
  **GeoTransolver**; mais a própria **Luminary-SMART**, com benchmark interno contra DoMINO e GeoTransolver.
- **Templates de hiperparâmetros** por aplicação, com controle total exposto; o *input pipeline* (dado → tensor) é
  gerenciado pela plataforma, sem I/O manual.
- **Pesos pré-treinados**: os SHIFT models inicializam novos treinos, e o *fine-tune* com dado do cliente parte daí em
  vez de partir do zero.

### 2.4 Treino

Clusters de GPU em nuvem, **paralelismo de dados e de modelo**, provisionamento e escalonamento abstraídos. Registro
em tempo real de curvas de loss, métricas de validação e distribuições de parâmetros; toda configuração de treino é
capturada em metadado estruturado para reprodutibilidade.

### 2.5 Validação

Conjuntos de teste que evoluem, incluindo medições reais e casos sintéticos, avaliados **durante e depois** do treino,
para interrogar o modelo para além da distribuição de treino antes de publicá-lo.

### 2.6 Registro e implantação

Cada modelo vira uma entrada de **Model Registry** com arquitetura, pesos, hiperparâmetros e metadados de treino —
**incluindo versão do dataset e características da distribuição**. Retreinar gera nova entrada versionada. A entrega é
por **API de inferência** ou integração própria/on-premise, e há inferência embarcada em CAD (nTop, Onshape) para
projeto inverso.

### 2.7 Confiança embutida: OOD e UQ

Eles avaliam o **input** em vez de rodar o modelo N vezes: projetam geometria e parâmetros num espaço de embedding e
aplicam uma métrica leve que devolve uma **confiança de 0 a 1 antes da inferência pesada**, com custo desprezível
(contra 50× do MC Dropout ou 5× de deep ensembles). Publicam a tabela no SHIFT-Wing:

| caso | deslocamento | confiança |
|---|---|---|
| baseline in-distribution | nenhum | 0,99 |
| Mach levemente fora | pequeno | 0,92 |
| Mach muito fora | grande | 0,82 |
| NASA CRM | geométrico leve (cauda extra) | 0,57 |
| geometria de SUV | geométrico massivo | 0,13 |

E conectam isso a **active learning** (rodar solver só onde o modelo não entende) e a **Bayesian optimization**, além de
usar o sinal em produção para decidir quando retreinar.

### 2.8 Os "cinco estágios" deles não são o pipeline

Vale registrar para não confundir: os cinco estágios do artigo de abril/2026 são de **adoção** — Education → Lab
Project → Pilot → Production → Scale — e medem quem usa o modelo e com que consequência, não como se treina. É um
modelo de maturidade organizacional; o pipeline técnico é o da §2.1–2.7.

## 3. O nosso pipeline hoje

### 3.1 Os cinco estágios, por espaço de projeto

`benchmarks/lpm_campaign.py` roda onze espaços de projeto, um por vez, em sequência, e cada um passa por cinco
estágios (`pinneapple_neural/lpm_pipeline.py`):

| estágio | o que faz | o que garante |
|---|---|---|
| **draw** | amostra o espaço de projeto a partir da receita (`benchmarks/lpm_recipes.json`) | o espaço é **dado**, versionado no repositório, não um script |
| **build** | roda a referência da família (analítica, FEM, VLM) | o alvo é exato ou convergido, com a identidade que o prova |
| **write** | grava um `.npz` por design + `manifest.json` com SHA-256 de cada arquivo | o dataset é identificável e reproduzível |
| **fit** | treina o LPM, com split por geometria, checkpoint e progresso | ver §3.2 |
| **card** | mede e publica: erro held-out, baseline, OOD, tolerância, `trust.decision` | o modelo diz o que não sabe |

### 3.2 O que o card mede — e é a parte mais defensável do que temos

- **Split por geometria, nunca por design**: dois pontos de operação do mesmo corpo nunca caem em lados opostos
  (`_split_by_geometry`). Sem isso o número de held-out mentiria.
- **Baseline de campo constante**: o card traz o erro de prever o campo médio. É o que separa "aprendeu" de "rodou" —
  foi assim que `pp_airfoil` ficou marcado como reprovado (0,710 contra 0,974 de baseline, p90 3,13).
- **Sondas OOD** fora da família, com o número de quantas foram levantadas.
- **Tolerância por família** e veredito explícito `within_tolerance`.
- **`trust.decision = not_verified_until_independent_review`** em todo card: nenhum modelo nosso se declara verificado.

### 3.3 Arquitetura e treino

`pinneapple_neural/lpm.py`: encoder de Fourier dos pontos de consulta, vizinhança multi-escala em três raios,
**geometria codificada por pooling invariante a permutação** (DeepSets) e condicionamento nos parâmetros de operação;
backbone MLP ou **atenção ponto-a-superfície** (`PointAttention`); **cabeças de ensemble com bagging** → média e desvio
por predição; OOD pela **distância do código de geometria** ao design de treino mais próximo, normalizada pelo quantil
0,95 das distâncias de treino, mais checagem de faixa dos parâmetros (`ood()`, lpm.py:347).

### 3.4 Operação

Uma máquina (MPS), um espaço de projeto por vez, trava em `<out>/.campaign.lock`, `--threads` para não tomar a máquina,
`--max-hours` que para **entre** famílias, checkpoint a cada 15 min (`fit.pt`) para sobreviver a uma morte silenciosa,
`runner.json`/`progress.json` para acompanhar, e um painel local (porta 8130) com as quatro figuras e o viewer 3-D do
Studio por família.

### 3.5 Números medidos (28/11/2026)

| família | held-out | baseline | veredito |
|---|---|---|---|
| `pp_airfoil_rans` (dado RANS de terceiro) | **0,055** | 0,959 | passa |
| `pp_nozzle` | **0,044** | 0,666 | passa |
| `pp_exchanger` | **0,065** | 0,444 | passa |
| `pp_bracket` | **0,189** | 0,919 | fora da tolerância (0,10) |
| `pp_fin` | **0,190** | 0,585 | fora da tolerância (0,10) |
| `pp_wing3d` | **0,361** (p90 0,462) | 0,953 | fora da tolerância (0,10) |
| `pp_airfoil` | **0,710** (p90 3,13) | 0,974 | reprovado; refit com 900 corpos em curso |

## 4. As diferenças

### 4.1 Escala de dados: ~10×

400–900 geometrias × 2–4 condições (1 200–3 600 designs) contra 700–5 000 **simulações de alta fidelidade** por
modelo deles, em malhas de até 122 M células. A nossa referência custa milissegundos a segundos por design; a deles,
horas de GPU. Não é um detalhe de orçamento: é a razão pela qual eles acertam o campo em domínios onde nós erramos a
cauda — e é o argumento a favor do refit de 900 corpos, não de mais épocas.

### 4.2 De onde vem o dado

Eles **possuem o solver** e o gerador de geometria (morfismo paramétrico + CAD), então podem fabricar o dataset que o
modelo precisa. Nós usamos referências que já temos — analítica, FEM imerso, VLM com painel 2-D, e um dataset RANS
público — o que dá verdade verificável por identidade, mas limita a física e a geometria ao que essas referências
cobrem. É a diferença mais estrutural das quatro.

### 4.3 Arquitetura

Eles têm arquitetura própria (SMART) medida num benchmark interno contra DoMINO e GeoTransolver, e oferecem várias
famílias. O nosso backbone é a pesquisa pública de 2017–2024 (atenção ponto-a-superfície, Fourier, DeepSets,
ensembles) implementada por nós, medida numa família analítica e em dez espaços de projeto. Estamos na mesma linhagem
metodológica, um ciclo de arquitetura atrás.

### 4.4 Transferência de aprendizado

Eles **partem de um modelo pré-treinado** (SHIFT) e fazem fine-tune com o dado do cliente. Nós treinamos **do zero** em
cada espaço de projeto. É provavelmente a diferença que mais barato se fecha do nosso lado: as sete famílias já
treinadas são os pesos de partida naturais para as próximas.

### 4.5 Confiança

Mesma ideia, execuções diferentes. Ambos olham o **input** (eles: embedding de geometria e parâmetros; nós: código de
geometria e faixa de parâmetros), o que é a escolha certa — e ambos evitam o custo de MC Dropout/ensembles no
inferência. A diferença é o **produto**: eles calibram para um score 0–1 com uma tabela publicada de deslocamentos
(0,99 → 0,13) e derivam ação (seguir, segunda olhada, parada dura); nós devolvemos uma **razão** (ex.: 1,20× o spread
de treino) com um limiar (`factor=1.5`) e uma lista de razões. Nosso número não é interpretável por quem não é do time,
e não dá para virar semáforo sem calibração.

### 4.6 Produto em volta

Eles têm registry versionado (arquitetura + pesos + hiperparâmetros + versão do dataset + características da
distribuição), API de inferência, conectores de CAD e laço de retreino. Nós temos o card e um viewer local. É a
diferença que menos importa **agora** — sem cliente recorrente, registry é custo — e a que mais importa quando o
primeiro piloto existir.

### 4.7 Uma lição que é nossa, e é a mesma deles

O gargalo de treino deles era **recalcular a relação entre pontos vizinhos a cada passo**; o cache valeu 50× por época.
O nosso `_multiscale` faz exatamente isso: `torch.cdist(query, points)` em todo forward (`lpm.py:81`, chamado em
`lpm.py:313`), com designs que **não mudam**. É o mesmo erro na mesma arquitetura de dados — mas **medido do nosso
lado dá 1,6× no wing e 2,2× no fin** (4 epochs, 400 designs, batch 24), não os 50× deles: o ganho deles vinha de
descongestionar o *carregamento* numa GPU faminta, e o nosso `cdist` é só uma parte do forward. Fica o cache, que é
grátis; não fica a expectativa de ordem de grandeza.

## 5. O que já está do nosso lado

- **Baseline no card.** Publicar o erro do campo constante junto com o do modelo é mais honesto do que a maioria dos
  números de fornecedor que circulam: "0,361" sozinho soa bem, e ao lado de 0,953 diz que ainda erra a cauda.
- **Split por geometria declarado e testado**, com a razão escrita (dois pontos de operação do mesmo corpo não
  atravessam o split). É a fonte de vazamento mais comum nesse tipo de benchmark.
- **Tolerância por família e veredito explícito**, com `trust.decision` que nunca se auto-promove.
- **Refazer uma família inteira em 1–2 h numa máquina de mesa**, com o dataset reconstruído do zero e o card refeito —
  nenhum pipeline de nuvem faz isso com essa facilidade.
- **Transparência do que falhou**: o `pp_airfoil` está no painel com 0,710 e 150 de 400 corpos piores que o campo médio,
  não escondido.

## 6. Ações, em ordem de custo/benefício

| # | ação | custo | ganho esperado |
|---|---|---|---|
| 1 | **Cachear a vizinhança por design** (pré-computar os índices de `_multiscale` uma vez por dataset, como eles fizeram) | dias | treino de ordem de grandeza mais rápido; as 11 famílias fecham numa noite |
| 2 | **Calibrar o OOD para 0–1** com uma tabela de deslocamentos como a deles (Mach fora, geometria fora, corpo de outra família) | ~1 dia por família | o número vira semáforo utilizável por cliente |
| 3 | **Partir de pesos pré-treinados** entre famílias (fine-tune de família irmã em vez de zero) | dias | menos dado por família nova — o caminho mais curto para a 12ª família |
| 4 | **Subir o dataset das famílias que reprovaram** (900 corpos no airfoil já em curso; wing depois) | ~10 h por família | ataca a cauda, que é o que reprova |
| 5 | **Registry mínimo** (versão + dataset + métricas por card, lendo o que já escrevemos) | dias | pré-requisito do primeiro piloto; não fazer antes de haver cliente |
| 6 | **Solver GPU próprio** | meses | não copiar: é a aposta de capital deles, não a nossa |

## 7. O que este comparativo não cobre

- Não há publicação do **cronograma** deles por modelo (quantas GPUs, quantas horas por SHIFT); os números de dados e de
  velocidade de treino são os únicos concretos.
- Não há publicação do **método de calibração** do score de confiança, só da tabela de resultados.
- Não avaliamos a **qualidade** dos SHIFT models de forma independente: os números deles são auto-reportados, como os
  nossos — a diferença é que os nossos vêm com baseline e com o que falhou.
