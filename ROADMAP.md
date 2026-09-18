# Roadmap

Catálogo único de iniciativas propostas para o PINNeAPPle e seu ecossistema
(`PINNeAPPle-arena`, `pinneapple-os`, `veriphysics`, `pinneapple-apps`,
`pinneapple_splash`, `reality2physics`, `physcurator`, `ge3`, `portfolio`).
Consolida duas fontes: o brainstorm de referências externas trazido nesta
sessão e o catálogo de 33 ideias já mantido em CoupleTasks (com Laís) —
sem repetir item já coberto em nenhuma das duas.

Cada entrada usa um destes status, herdados do vocabulário já usado no
catálogo do CoupleTasks e nos `ROADMAP.md` de `veriphysics`/`pinneapple-os`:

- **Já existe** — real, rodando, em algum repo do ecossistema.
- **Sobreposição parcial** — existe uma peça real que cobre parte do escopo.
- **Projeto novo** — ideia, nada construído ainda.
- **Peça em aberto** — a peça está faltando por definição (ex.: GenAItor).

> **Recomendação estratégica (herdada do CoupleTasks):** priorizar o
> **GenAItor** (camada fina de orquestração) sobre os repositórios já
> existentes (`reality2physics`, `PINNeAPPle-arena`, `ge3`, `portfolio`)
> antes de iniciar os domínios novos listados abaixo — ver [[chordiq_portfolio]]
> na memória: a maior parte da "Scientific Intelligence Platform" já existe
> espalhada, o que falta é o roteador que amarra tudo.

---

## 1. Scientific Intelligence Platform — GenAItor + PINNeAPPle + Blender

Núcleo estratégico: um roteador que interpreta um pedido em linguagem
natural, classifica domínio físico + capacidade desejada (simular,
otimizar, prever, descobrir, comparar, decidir) e despacha para o
repositório/motor certo, narrando o resultado.

### GenAItor — Camada de Orquestração
**Peça em aberto.** Agente/roteador central. É a única peça da visão que
ainda não existe de fato — o protótipo de 2024 encontrado em
`barrosyan/pinneaple/genaitor/` é só um gerador de system-prompts via
Flask/llama.cpp, sem lógica de roteamento real (ver memória
`chordiq_portfolio`, correção de 2026-09-11).

### PINNeAPPle Physics AI Arena / Model Benchmarking
**Já existe**, em expansão. Para um mesmo problema físico, treina/avalia
múltiplos modelos (PINN, FNO, DeepONet, GNN) e compara acurácia, resíduo
de PDE, tempo de treino/inferência, memória e generalização OOD — usado
pelo GenAItor para responder "qual modelo devo usar?". Vive em
`PINNeAPPle-arena` (backend + frontend + leaderboard); está recebendo o
catálogo externo de ~94 bibliotecas de terceiros (ver §4, "Noether
Surrogate Benchmark" e a integração `physics-based-references` em
andamento).

### Reality-to-Simulation Agent (Reality2Physics)
**Já existe** (MVP validado, repo `reality2physics`). Converte vídeo/
imagem/sensores em representações físicas estruturadas (campo físico,
parâmetro escalar, PDE) — validado em Navier-Stokes, calor e onda com uma
única arquitetura compartilhada.

### Physics Discovery / Equation Discovery
**Já existe**: `portfolio/pinneapple/inverse_sindy` — EKI + SINDy
redescobre o sistema de Lorenz (100% de acerto na estrutura). Dado dados
observacionais sem a equação explícita, descobre a PDE governante via
regressão simbólica/esparsa restrita por física. Este item é a semente de
uma vertente muito maior — geometria/manifold escondido, transições
ordem↔caos, invariantes desconhecidos, causalidade estrutural, eventos
extremos — consolidada em **§8, "Structure Discovery / Chaos-to-Law"**.

### Uncertainty Quantification Lab
**Sobreposição parcial.** Padronizar a saída de todo modelo Physics AI do
portfólio com intervalo de confiança, comparando calibração, detecção de
OOD e robustez entre PINN/FNO com e sem UQ. Backlog identificado em
`portfolio/pinneapple/missile_aero` ("UQ deferred"). Módulo relacionado:
`pinneapple_analysis.uncertainty`.

### Autonomous Scientific Experimentation
**Sobreposição parcial.** Agente de loop fechado que formula hipótese,
gera candidatos de design, simula, analisa e decide o próximo experimento
sem intervenção humana a cada iteração. Precedente parcial:
`portfolio/pinneapple/self_healing` (`TrainingAdvisor`, auto-retrain,
~20x redução de erro).

### PINNeAPPle Auto-PINN
**Projeto novo.** Dado apenas a formulação matemática de uma PDE (domínio,
condições de contorno/iniciais), gerar automaticamente arquitetura, função
de perda e treinamento do PINN correspondente. Módulo relacionado:
`pinneapple_problemdesign` (já existe um agente NLP→PDE parcial —
`DesignAgent`/`UnifiedPhysicsAgent` — este projeto fecha o ciclo até o
treino real).

### PINNeAPPle Surrogate Factory
**Projeto novo.** Pipeline industrializado: CAD paramétrico → DOE →
execução em lote do solver → dataset → treino de surrogate
(FNO/DeepONet) → validação → registro/deploy, sem intervenção manual.
Ver §3 "Autonomous DOE-CFD" para o blueprint técnico validado em escala
industrial (nTop/CoreWeave) que valida a viabilidade desta ideia.

### PINNeAPPle Model Zoo
**Sobreposição parcial** com `PINNeAPPle-arena` (falta a biblioteca
padronizada por PDE). Coleção padronizada de PDEs e operadores de
referência (Burgers, Navier-Stokes, calor, onda, Poisson, Allen-Cahn,
reação-difusão, Darcy, elasticidade), cada um com dataset, arquitetura,
benchmark e pesos publicados.

---

## 2. CAD generativo

### Spec-to-Solid (LLM → CAD paramétrico)
**Projeto novo.** Agente que recebe especificação em linguagem natural
(ou perfis/pontos, como aerofólios) e gera script CADQuery/Build123D;
quando a topologia falha, recorre à API do OnShape/SpaceClaim para
reparar a geometria. Saída validada automaticamente (sólido fechado,
pronto para malhar em CFD/FEM). Fonte: LinkedIn — Jaydeep Singh
(SpaceClaim scripting + CADQuery/Build123D/NumPy + Gemini). Módulo
relacionado: `pinneapple_design.geometry` (já tem SDF/CSG/mesh/NACA
airfoil — falta o passo LLM→script e o reparo de topologia).
Ver também §7, `pinneapple_llm.cad_draft` (usado pelo produto `Text2Part`
em `pinneapple-apps`) como ponto de partida real já validado.

### Reproduzir o comportamento do Astra em CAD com modelos 100% open-source
**Projeto novo, trazido por Yan em 2026-09-13 — e um contraste de design
explícito com o que já existe, não apenas uma extensão.** A pergunta
original: como reproduzir o comportamento positivo de agentes tipo Astra
(citado também em §7, `OpenV`) para geração de CAD, usando só modelos
abertos? Receita levantada na sessão, em três camadas:

1. **Backbone open-source** — um modelo especializado em código para
   gerar o script (`DeepSeek-Coder-V2` ou `Qwen2.5-Coder`), com um
   modelo generalista maior (`Llama 3.3`) como orquestrador de alto
   nível que decompõe o pedido em subtarefas.
2. **CAD programático como alvo de geração** — `CadQuery`/`Build123D`
   (Python, o mesmo par que a entrada "Spec-to-Solid" acima já usa) ou
   `FreeCAD` via API Python para geometria+restrições mais ricas;
   `OpenSCAD` como alternativa CSG mais simples.
3. **Harness com loop de auto-correção** — decompor o pedido → gerar o
   script → executar em sandbox → se falhar, devolver o log de erro real
   do interpretador ao LLM para correção → repetir → montar as peças via
   coordenadas/transformações relativas no fim.

**Isto é genuinamente diferente do que `pinneapple_llm.cad_draft` faz
hoje, não uma generalização direta dele — e essa diferença precisa ficar
explícita antes de qualquer implementação.** `cad_draft.py` deliberadamente
NUNCA deixa o LLM escrever código CAD livre: ele monta um *recipe* JSON
(nome de builder + parâmetros + operação booleana, recursivo) validado
contra o registro real de `pinneapple_design.geometry.gen.primitives`/
`cadquery_gen` ANTES de qualquer geometria ser executada — um nome de
builder alucinado é rejeitado, nunca chega a rodar. A receita "Astra-com-
modelo-aberto" acima é o oposto: o LLM escreve o script Python inteiro,
que só é validado DEPOIS de rodar (sucesso/erro de execução), com
correção via replay do log de erro. Isto é uma superfície de risco maior
(código arbitrário gerado por um modelo pequeno local, precisa de sandbox
de verdade — não apenas `subprocess`, isolamento real de processo/
filesystem) em troca de expressividade maior (features CAD que a receita
JSON de `cad_draft.py` não modela, ex.: fillets/chamfers paramétricos
encadeados, padrões/arrays de features, sketches 2D->3D complexos).

**Como isto se conecta ao `OpenV`/Astra já mapeado em §7**: o princípio
"o LLM nunca avalia seu próprio trabalho" do `OpenV` (Astra propõe,
ferramenta externa de CAD/cálculo produz evidência, comparação
determinística decide PASS/FAIL) é exatamente o guardrail que esta
receita precisa para ser segura com um modelo pequeno local — o loop de
auto-correção por log de erro do interpretador (Python: falha ou não
falha) já é uma forma primitiva disso, mas "código Python roda sem
exceção" é um teste muito mais fraco que "a peça satisfaz os requisitos
físicos/dimensionais reais" — o mesmo gap de rigor que o `veriphysics` já
resolve para simulação PINN (trust score + Decision Record, não só
"o código rodou") precisaria ser replicado aqui: um sólido gerado que
compila e exporta STL/STEP mas tem topologia errada ou dimensão fora de
especificação passaria no loop de auto-correção acima sem ser pego.

**Como construir isto de verdade, concretamente, sem duplicar
`cad_draft.py`**: um segundo modo, explicitamente opt-in e claramente
rotulado como maior risco, dentro de `pinneapple_llm` — não uma
substituição do modo checked-menu existente:
1. Novo módulo `pinneapple_llm.cad_script_agent` (nome provisório):
   aceita um backbone de código (`DeepSeek-Coder-V2`/`Qwen2.5-Coder` via
   Ollama, mesma camada `_dispatch.call_llm` já usada por `cad_draft.py`)
   e gera um script CadQuery/Build123D real, não um recipe JSON.
2. Sandbox de execução real (processo isolado, sem acesso a rede/
   filesystem fora de um diretório temporário) — este é o item que
   `cad_draft.py` nunca precisou construir, porque nunca executa código
   arbitrário; aqui é obrigatório antes de qualquer uso além de
   experimentação local.
3. Loop de correção: script falha → captura stderr real → devolve ao
   LLM com o script original + erro → nova tentativa, com um limite
   máximo de tentativas (o histórico deste org com modelos locais
   pequenos — ver `Helm`'s `code_agent.propose_code_change`, que já
   documentou um modelo `llama3.2:3b` reescrevendo um arquivo inteiro e
   apagando imports por engano — é um sinal real de que um modelo pequeno
   não segura um script CAD inteiro na cabeça de forma confiável; vale
   considerar edição incremental de um script existente em vez de
   reescrita completa a cada tentativa, mesmo padrão que o `Helm` adotou
   depois de ver o problema na prática).
4. Verificação pós-execução real, não só "rodou sem exceção": geometria
   fechada (watertight), dimensões dentro de tolerância da especificação
   original — reaproveitando `pinneapple_analysis.verification` (o mesmo
   pacote de guardrails que `veriphysics` já usa para PINN), não
   inventando um verificador novo.
5. Só depois de (1)-(4) existirem de verdade, comparar lado a lado contra
   `cad_draft.py` no mesmo conjunto de pedidos — decidir com números reais
   se o modo script-livre realmente resolve algo que o modo checked-menu
   não resolve, em vez de assumir que "mais expressivo" significa
   "melhor" sem medir.

**Pesquisa verificada em 2026-09-13 — a comunidade de pesquisa já
identificou o mesmo problema de rigor (4) acima, com abordagens
concretas.** Três referências lidas de verdade (fetch, não só título),
todas de 2026:
- **Embodied CAD** ("Solver-Grounded LLM Agents for Parametric B-Rep
  Assembly Modeling",
  [arXiv:2606.31252](https://arxiv.org/abs/2606.31252)) — em vez de
  gerar um script inteiro de uma vez, o agente escolhe ações de uma
  "biblioteca de skills CAD estratificada L0-L4", cada ação é executada
  contra um kernel geométrico exato que valida imediatamente se a
  feature/posicionamento/relação de montagem é válida como B-Rep
  paramétrico editável — o feedback do solver guia planejamento,
  reparo e aprendizado. Isto é essencialmente o item (4) desta entrada
  ("verificação pós-execução real") já implementado como parte do loop
  em vez de um passo separado no fim — vale desenhar o passo 3 acima
  (loop de correção) inspirado nesta ação-por-ação em vez de
  script-completo-por-tentativa. Nenhum repositório de código público
  foi encontrado (verificado).
- **ToolCAD** ("Exploring Tool-Using Large Language Models in
  Text-to-CAD Generation with Reinforcement Learning", ACL Findings
  2026) — treina LLMs abertos como agentes tool-using via RL com
  curriculum online, especificamente para fechar a lacuna de modelos
  abertos terem desempenho comparável a proprietários neste tipo de
  tarefa — relevante diretamente para a escolha de backbone
  (`DeepSeek-Coder-V2`/`Qwen2.5-Coder`) desta entrada: RL sobre o
  próprio loop de execução/correção, não apenas prompting, é uma
  direção real para melhorar a taxa de sucesso desses modelos pequenos.
- **Text2CAD-Bench** ([arXiv:2605.18430](https://arxiv.org/html/2605.18430v1))
  — benchmark dedicado para geração de CAD paramétrico via LLM,
  avaliando modelos como Text2CAD/Text2CADQuery/CADFusion. Antes de
  declarar o modo script-livre "melhor" no passo 5 acima, rodar contra
  este benchmark público dá um número comparável externamente, não só
  uma comparação interna contra `cad_draft.py`.

---

## 3. Infraestrutura de Physics AI / Scientific ML

### Correção: `solve_pde()` treinava sem nenhuma condição de contorno real para presets `tag`-based (2026-09-16)
**Já corrigido.** `solve_pde()` (`pinneapple_physics/__init__.py`) só
auto-amostrava condições `selector_type in ("all", "callable")`; condições
`selector_type="tag"` (32/65 presets registrados são 100% tag-based, mais
8/65 mistos — `pipe_flow_3d`, `industrial_furnace_thermal`,
`aircraft_wing_aerodynamics`, `lid_driven_cavity_3d`, `laplace_2d`,
`poisson_2d`, etc.) eram puladas **silenciosamente**, sem erro — o
resíduo de PDE ainda caía, o treino "funcionava", mas nenhuma condição de
contorno real era imposta. Qualquer produto do portfólio que chame
`solve_pde()` num desses presets sem montar `ctx["tag_masks"]`/`x_bc`/
`mask_<nome>` à mão (ex.: `veriphysics/orchestrator/pipeline.py`'s
`analyze()`, que não passa nenhum desses hoje) treinaria e devolveria um
resultado fisicamente inválido com aparência de sucesso — o oposto do
princípio anti-fabricação do produto.

Antes de tratar como bug, o mecanismo `tag_masks` em si foi verificado
rodando `examples/pde_environment/04_heat3d_stl_box.py` (STL real via
`STLDomainBatchBuilder`) e `03_ns2d_channel_tags.py` de ponta a ponta:
funciona corretamente com geometria real — `selector_type="tag"` é um
contrato real (exige mesh/STL), não um recurso quebrado. A correção:
`solve_pde()` agora levanta `TagConditionsUnresolved` (erro claro,
nomeando exatamente quais condições/tags faltam) em vez de treinar
silenciosamente sem elas. Detalhes completos (classificação programática
dos 65 presets, números de antes/depois da suite de testes, todos os
`solve_pde()` callers do repo auditados) em `docs/dev/AUDIT_REPORT.md` e
`docs/dev/ROADMAP_PHYSICS_AI_HUB.md`.

### Follow-up (2026-09-17): geometria analítica real para 23 dos 40 presets tag-based
**Já corrigido, parcialmente por natureza (não por falta de esforço).**
Dos 40 presets que o fix acima deixou inutilizáveis sem geometria real, 23
tinham domínio canônico (caixa/retângulo/cilindro) totalmente descrito
pelos próprios parâmetros do preset, com mapeamento tag→face inequívoco
(citado literalmente do docstring/comentário de cada preset em
`pinneapple_physics/pde_environment/presets/tag_geometry.py`). Para esses
23, um novo builder mesh-free (`pinneapple_design/geometry/builders/
analytic_domain_batch_builder.py`) amostra o domínio analiticamente (sem
precisar de STL, e sem o recentragem forçada do `STLDomainBatchBuilder`
que quebraria `value_fn`s que assumem coordenadas literais) e treina de
ponta a ponta via `solve_pde()` de verdade — confirmado com resíduo/loss
por tag real, distinto e não-degenerado para os 23 (ex.: `pipe_flow_3d`
convergiu de loss agregado 299.6 → 2.55 em 200 epochs, com os 3 tags
inlet/outlet/wall caindo individualmente e para valores distintos).

Os outros 17/40 permanecem documentados (não "consertados") porque a
geometria real é específica de cliente/domínio e não dedutível dos
parâmetros do preset (perfil de asa real, silhueta de carro, geometria
interna de forno/datacenter, zonas de componente numa PCB dadas só como
dict de potência sem coordenada, etc.) — inventar uma geometria genérica
pra esses seria fabricar confiança sobre um resultado fisicamente
inválido, exatamente o que o fix original existe para prevenir.
`TagConditionsUnresolved` continua disparando para os 17, sem alteração.

Um bug real e independente foi encontrado e corrigido no processo:
`STLDomainBatchBuilder._targets_from_conditions` só preenchia `y_bc` para
condições `kind="dirichlet"`, deixando alvos Neumann/Robin como NaN
silenciosamente (o `compile.py`'s `loss_fn` usa `y_bc` como alvo pra
QUALQUER tipo de condição uma vez que ele é fornecido, não só Dirichlet).
Corrigido nos dois builders.

Suite completa (`pytest tests/`, 1667 testes, antes vs. depois, mesmo
ambiente, mesmo commit-base `bfa19dbd`): passed 1289→1340 (+51), skipped
263→212 (-51), failed 75→75 e error 39→39 (conjunto de IDs
bit-a-bit idêntico antes/depois — zero regressão, zero teste
pré-existente consertado por acidente). Os 51 que viraram pass são
exatamente as combinações arquitetura×preset dos 23 presets consertados
em `test_cartesian_breadth.py`/`test_full_library_matrix.py`. Detalhes
completos, lista dos 23 consertados e dos 17 documentados como não
consertáveis sem fabricar geometria, em `docs/dev/AUDIT_REPORT.md`.

### Segundo follow-up (2026-09-17): 9 presets específicos dos 17 "não consertáveis", 3 estratégias pré-decididas
**6 de 9 fechados de verdade; 1 confirmado bloqueado por parâmetro
ausente (não chutado); 2 parciais.** Esta passada não escolheu a
abordagem — recebeu uma pronta pra cada caso e mediu se dava pra cumprir
sem fabricar dado que o preset não fornece:

- **Grupo 1 (geometria pública/padrão real)**: `rocket_nozzle_cfd`
  (bocal cônico convergente-divergente — ângulo de meio-cone DERIVADO de
  `throat_radius`/`exit_radius`/`nozzle_length`, já existentes no preset),
  `car_external_aero` (silhueta inspirada no Ahmed body — Ahmed, Ramm &
  Faltin 1984, SAE 840300 — usando as RAZÕES publicadas de folga/raio de
  nariz/rampa traseira aplicadas ao `car_length`/`car_height` já do
  preset) e `axial_compressor_cascade_2d` (pá em arco circular — Dixon &
  Hall, fórmula de raio de câmber `R=chord/(2 sin(θc/2))` a partir dos
  ângulos de escoamento já do preset) foram fechados de verdade — geometria
  real amostrada analiticamente, treino de ponta a ponta via `solve_pde()`,
  perdas por tag reais e distintas confirmadas (cascade converge de
  verdade: `387.4 → 3.43` em 30 epochs). `aircraft_wing_aerodynamics`
  **ficou bloqueado por investigação de verdade, não suposição**: o
  preset não expõe NENHUM parâmetro de espessura/código NACA — usar
  "NACA 0012" seria inventar o único número que falta.
- **Grupo 2 (bug de compilador)**: `aircraft_wing_structural` fechado.
  A causa real não era geometria (já inequívoca) — era `compile.py`
  avaliar `fvals[f]` pros campos de tração `tx`/`ty` antes mesmo de saber
  que eles não são campos de saída do modelo (`ux`/`uy`), gerando
  `KeyError`. Generalizado via novo `ConditionSpec.traction_map`
  (mapeamento explícito tração→campo de deslocamento), que deriva a
  tração do tensor de tensão (mesma fórmula usada pelo resíduo interno,
  incluindo o lambda* reduzido de plane stress) em vez de copiar o nome
  literalmente — verificado com um caso de cisalhamento puro fechado
  analiticamente (erro relativo ~7e-8) e rodando o preset real de ponta a
  ponta (perdas por tag distintas e não-degeneradas). Nenhuma condição
  pré-existente no repo é afetada (`traction_map=None` por padrão,
  36/36 de `test_manufactured_solutions.py` inalterado).
- **Grupo 3 (convenção física, decisão já tomada)**: `linear_elasticity_3d`
  e `plane_stress_2d` fechados — `"fixed"` = face mínima do eixo principal,
  `"load"` = face máxima (viga em balanço canônica), documentado
  explicitamente como **convenção escolhida por esta sessão**, não texto
  do preset (`CHOSEN_CONVENTIONS` em `tag_geometry.py`, mesmo padrão de
  transparência dos 23+1 anteriores). `car_brake_thermal` e
  `rocket_structural` são os 2 parciais: pra `car_brake_thermal`, a
  convenção `"friction"` = as duas faces planas do disco, `"cooling"` =
  raio externo, foi implementada e a geometria verificada de forma
  isolada (`sample_box_tag_batch` dá exatamente 1000/500 pontos nas tags
  certas) — mas treinar de ponta a ponta esbarra num SEGUNDO defeito,
  independente, de compilador: `"friction_surface"`/`"cooling_surface"`
  declaram campos `q_heat`/`h`/`T_ref` que não são campos de saída do
  modelo (só `T`) nem resolvíveis pelo mecanismo `traction_map` (feito
  pra tração elástica, não fluxo de calor/convecção) — o mesmo padrão
  aparece em `cpu_heatsink_thermal`, `pcb_thermal`,
  `industrial_furnace_thermal` e nos 3 presets `datacenter_*` (confirmado
  via grep), um gap sistêmico bem além do escopo autorizado aqui. Pra
  `rocket_structural`: o `domain_bounds` quadrado sólido foi
  corrigido para o anular real usando `inner_radius`/`outer_radius` —
  parâmetros que **já existiam** em `spec.meta` — geometria verificada
  (pontos interiores caem exatamente em `r∈[0.2,0.22]`) e 3 das 4
  condições treinam com perdas reais e distintas. Mas a 4ª
  (`inner_wall`, campo `p_normal`) esbarra num SEGUNDO defeito de
  compilador, independente e fora do escopo autorizado (pressão escalar
  precisa de `n^T·σ·n`, não de um componente de tração nomeado). Nos dois
  parciais, o fixture de geometria foi propositalmente NÃO conectado em
  `TAG_GEOMETRY_FIXTURES`/`build_tag_batch` (fica como infraestrutura
  testada e verificada, só isso) — `test_full_library_matrix.py` continua
  pulando os dois exatamente como antes (confirmado rodando de novo), zero
  mudança de status de teste por causa deles. Ambos ficam parcialmente
  fechados, com o motivo exato documentado pra cada um, e
  `TagConditionsUnresolved` continua disparando pros dois.

Suite completa (`pytest tests/`, 1667 testes, `571b66ba` limpo vs. este
commit, ambiente idêntico): passed 1341→1353 (+12), failed 74→75 (+1),
error 39→39, skipped 212→199 (-13), xfail 1→1. **Diff exato, não só
contagem** (o pytest desse ambiente não escreve a linha final de tally
em nenhuma das duas rodadas — confirmado não ser trava/concorrência,
`--collect-only` confirma os mesmos 1667 IDs na mesma ordem nos dois
commits, então o stream de caracteres `.FEsx` por teste foi mapeado
posição-a-posição de volta aos IDs reais): **14 mudaram de status, 13 são
skip→pass exatamente as combinações dos 6 presets consertados** (7
arquiteturas × `plane_stress_2d` em `test_cartesian_breadth.py` + 1 teste
por preset para os 6 em `test_full_library_matrix.py`) — bate certinho,
nada sobrando. **A 14ª é pass→fail e não tem nada a ver com este
trabalho**: `test_architecture_critique.py::test_real_llm_produces_a_valid_complete_response`
chama um Ollama real (`provider="ollama"`) e falhou porque o LLM
alucinou uma categoria fora do checklist fixo — reproduzido isoladamente,
causa raiz confirmada e desligada de tudo que essa sessão tocou. **Zero
regressão real** deste trabalho.
Tally atualizado contra os 40 originais: **29/40 (23+6) agora treinam de
ponta a ponta com geometria real**; 11/40 seguem documentados como não
consertáveis, cada um com motivo específico verificado (não suposto).
Detalhes completos em `docs/dev/AUDIT_REPORT.md`.

### Terceiro follow-up (2026-09-17): NACA 0012 padrão de literatura + 2 mecanismos novos de Neumann no compilador

Decisão do dono do produto (Yan), não escolha desta passada: 3 pendências
específicas da passada anterior. **As 3 fecharam de verdade.**

1. **`aircraft_wing_aerodynamics`**: adicionado `naca_thickness: float =
   0.12` (NACA 0012, perfil simétrico de 12% de espessura) como parâmetro
   explícito e sobrescrevível do preset — documentado em toda parte
   (docstring, `_curve_geometry.py`, `tag_geometry.py`) como **default de
   literatura escolhido pelo dono do produto**, não algo inerente ao
   preset original (que nunca teve parâmetro de espessura/código NACA,
   exatamente como a passada anterior confirmou por inspeção). Geometria
   real construída pela fórmula pública NACA 4 dígitos (`y_t =
   5t(0.2969√x − 0.1260x − 0.3516x² + 0.2843x³ − 0.1015x⁴)`), nova função
   `naca4_symmetric_polygon` reaproveitando os helpers genéricos
   `polygon_perimeter_sample`/`polygon_contains` já existentes (sem
   mudança neles). Verificado de ponta a ponta via `solve_pde()`: 4 tags
   com pontos reais e não-degenerados (`farfield_inlet`, `farfield_outlet`,
   `wake_outlet`, `airfoil`), perdas por tag reais e distintas no início
   (`pde=4.65, bc_farfield_inlet=5189.5, bc_farfield_outlet=2.12,
   bc_airfoil=0.088, bc_wake_outlet=3.81`), 2 das 4 tags convergem
   visivelmente em 20 epochs.
2. **`car_brake_thermal` (fluxo de calor/convecção)**: generalizado
   `compile.py` com `ConditionSpec.thermal_bc` (`kind="flux"` →
   `-k·dT/dn=q_heat`; `kind="convection"` → `-k·dT/dn=h·(T-T_ref)`),
   mecanismo genérico (não hardcoded pro brake) que também resolve o
   mesmo padrão `q_heat`/`h`/`T_ref` de `cpu_heatsink_thermal`,
   `pcb_thermal`, `industrial_furnace_thermal` e os 3 `datacenter_*` — só
   `car_brake_thermal` tinha geometria pronta pra usar (fixture da
   passada anterior, só não estava no dispatch table). Verificado com
   forma fechada (resíduo `~0` pro alvo exato, `~1e6`-`~1e8` pro errado,
   em 2 casos) e de ponta a ponta: `friction_surface`/`cooling_surface`
   com 800/400 pontos reais, perdas `bc_friction_surface=4.00e12`
   (consistente com `q_friction=2e6` ao quadrado, rede não-treinada),
   `bc_cooling_surface=5.51e8`. `test_full_library_matrix.py` agora treina
   3 epochs reais pra esse preset (antes pulava via
   `TagConditionsUnresolved`).
3. **`rocket_structural` (pressão normal)**: generalizado `compile.py`
   com `ConditionSpec.normal_stress_field` (contração escalar completa
   `n^T·σ·n = -p_internal`, caminho novo, diferente do `traction_map`
   componente-a-componente já existente) — precisou também estender
   `_elasticity_stress_tensor` pro `pde_kind` real do preset
   (`thermoelasticity_2d`, não uma das 3 elasticidades puras que
   `traction_map` já cobria), incluindo a correção de deformação térmica
   isotrópica (`eps_th=alpha_T·T·I`) que o resíduo interno já usa —
   verificado que essa correção é realmente usada (alvo sem ela dá perda
   `44.75` em vez de `~0`). Fixture anular da passada anterior
   (`inner_radius`/`outer_radius` reais de `spec.meta`) agora conectada
   via novo branch `"annulus"` em `build_tag_batch`. Verificado de ponta a
   ponta: 4 tags com 400 pontos cada, perdas reais
   (`bc_inner_wall=1.32e23`, `bc_outer_wall=0.056`, `bc_T_inner=639568`,
   `bc_T_outer=85690` — os 2 últimos batem quase exatamente com os
   valores já verificados na passada anterior pras mesmas 3 condições sem
   `inner_wall`, confirmando que a geometria não mudou, só o `inner_wall`
   virou um número real em vez de erro).

Suite completa (`pytest tests/`, `14e0a131` limpo vs. este commit, mesmo
ambiente, mesmos 1667 IDs coletados nos dois, mapeados posição-a-posição
como nas passadas anteriores): passed 1354→1357 (+3), failed 74→74,
error 39→39, skipped 199→196 (-3), xfail 1→1. **Exatamente 3 mudaram de
status, todos skip→pass, exatamente os 3 presets desta passada**
(`test_full_library_matrix.py::test_audit_breadth_preset_trains_a_few_steps`
pra `aircraft_wing_aerodynamics`, `car_brake_thermal`, `rocket_structural`)
— **zero mudança inesperada, zero regressão** (diferente da passada
anterior, que teve 1 flip não-relacionado de uma chamada real ao Ollama;
desta vez `FAILED`/`ERROR` ficaram idênticos bit-a-bit, 74/39 nos dois).

Tally atualizado contra os 40 originais: **32/40 (23+6+3) agora treinam
de ponta a ponta com geometria real**; 8/40 seguem documentados como não
consertáveis (geometria real de aleta/rack/hotspot/pá que esta passada
não foi autorizada a fabricar — 4 deles já têm o campo `thermal_bc`
pronto pro compilador, só falta geometria). Detalhes completos em
`docs/dev/AUDIT_REPORT.md`.

### PhysicsNeMo Backbone Swap
**Projeto novo.** Avaliar/trocar o NVIDIA PhysicsNeMo como backbone dos
surrogates CFD do ChordIQ (mixing-tank, cloramina), comparando contra a
abordagem PINNeAPPle atual. Já existe um precedente real de comparação
lado a lado: `portfolio/pinneapple/vs_physicsnemo` (PINNeAPPle 4.4%
rel-L2/70k params vs. PhysicsNeMo 1.0%/137k params, ver memória
`chordiq_portfolio`) — este item estende essa comparação para os
surrogates de produção, não só para o card de demonstração.

### Newton Multibody Data Forge
**Projeto novo.** Usar o NVIDIA Newton para acelerar simulações
multibody em GPU e gerar dados sintéticos de treino em escala para os
surrogates físicos existentes.

### Multi-Physics Sandbox
**Projeto novo.** Usar o simulador unificado do Genesis (rigid body + FEM
+ MPM + SPH/PBD, compilador cross-platform Quadrants para
CUDA/ROCm/Metal/**Vulkan**/x86/ARM64) para gerar dados sintéticos de
escoamento/partículas, complementando os dados OpenFOAM que já alimentam
os twins do ChordIQ. Fonte:
[Genesis-Embodied-AI/genesis-world](https://github.com/Genesis-Embodied-AI/genesis-world)
(29.9k stars, engine de física unificado para robótica/embodied AI —
verificado nesta sessão via `gh api`). Módulo relacionado:
`pinneapple_simulation.particle_dynamics` (MPM/SPH/rigid-body hoje em
PyTorch puro — Genesis é uma alternativa muito mais madura como backend
plugável) e `pinneapple_worldmodel` (ambientes para agentes). Conecta
também com o backend Vulkan do item "LatticePT" (§7) — mesma direção de
runtime GPU-nativo/multiplataforma.

### Noether Surrogate Benchmark
**Projeto novo.** Framework de transformers pré-construídos para
CFD/aerodinâmica (ex.: AB-UPT no DrivAerML); usar como benchmark antes de
investir em arquitetura própria. Fonte:
[Emmi-AI/noether](https://github.com/Emmi-AI/noether). Alimenta
diretamente o catálogo externo do `PINNeAPPle-arena` (§1).

### GPU Signal-Prep Kit
**Projeto novo.** Reaproveitar o padrão de pipeline GPU-first do
cuPhoton (CuPy/Numba-CUDA) como pré-processamento de sensores de planta
industrial antes de virar input de um surrogate. Fonte:
[nvidia/cuPhoton](https://github.com/nvidia/cuPhoton). Módulo relacionado:
`pinneapple_systems.digital_twin.io` (streams MQTT/Kafka/OPC-UA/Modbus).

### Scientific ML Curriculum Pipeline
**Projeto novo.** Usar a estrutura de 20 capítulos do livro "Data-Driven
Modeling and Scientific Computation" como esqueleto para pipelines de
Scientific ML de referência antes de treinar modelos próprios. Fonte:
[nathankutz/ScientificComputing](https://github.com/nathankutz/ScientificComputing).

### Agentic PDE Debugger
**Projeto novo.** Time multiagente (inspirado no paper ATHENA) que
escolhe o método numérico, implementa, detecta falhas de física
silenciosas (ex.: instabilidade Kelvin-Helmholtz/Rayleigh-Taylor) e
corrige sozinho; aplicável ao auto-tuning dos solvers CFD do roadmap de
cloramina. Fonte: arXiv — ATHENA. Módulo relacionado:
`pinneapple_analysis.verification` (guardrails, conservação) +
`pinneapple_llm.guardrail.PhysicsGuardrail` já fazem parte disso — este
projeto fecha o ciclo com correção automática, não só detecção.

### Geometry OOD guardrail no trust_gate -- CONCLUÍDO (2026-09-18)
**Item fechado**, originado de uma investigação cruzada real com o
PhysicsNeMo (`physicsnemo-notes/insights-to-import-into-pinneapple.md`,
seção 7a): o PhysicsNeMo tem um guardrail real de OOD geométrico
(`physicsnemo/experimental/guardrails/geometry/`, ~3561 linhas) que
sinaliza quando uma geometria de entrada foge da distribuição de formas
vista no treino -- extração de features de forma + modelo de densidade
(GMM/PCE) + classificação OK/WARN/REJECT. O sistema de confiança do
PINNeAPPle (`trust_gate.py`/`physics_confidence_score.py`/
`evidence_graph.py`) não tinha esse sinal especificamente --
`geometry_intelligence.py` resolve um problema adjacente mas diferente
(classificação semântica de região/BC, não detecção de anomalia de
forma). Implementado em
`pinneapple_analysis/verification/geometry_ood_guardrail.py`: 13
features geométricas reais (centróide, extensão de bbox, autovalores
PCA, área de superfície, volume de bbox, aspect ratio, proxy de
curvatura), reaproveitando `MeshData`/`compute_curvature_proxy` já
existentes, mais uma distância de Mahalanobis diagonal (não uma GMM/PCE
completa -- decisão de escopo explícita, documentada no docstring do
módulo: o catálogo de geometria atual do PINNeAPPle ainda não tem dados
suficientes por preset para justificar uma covariância completa) +
`scipy.stats.chi2.sf`, o mesmo recurso estatístico que
`TrustGate._ood_score` já usa para OOD no espaço de coordenadas.
Integrado como 5º componente de `PhysicsConfidenceScore`
(`N_POSSIBLE_COMPONENTS` 4→5); `evidence_graph.py` passou a consumir o
novo componente sem nenhuma mudança de código (itera genericamente sobre
`confidence.components`). Verificado de verdade: uma caixa
`(1.5, 1.2, 0.9)` dentro do catálogo de referência (48 malhas reais de
caixa/cilindro/canal via `build_mesh()`) pontua p=0.998 (OK); uma caixa
extrema `(500, 0.001, 0.001)` pontua p≈0 (REJECT), com
`aspect_ratio`/`pca_eigenvalue_1` corretamente apontados como os
motivos. Suite completa (`pytest tests/`, `69117744` limpo vs. este
commit, mesmo ambiente, extra opcional `trimesh`/`geom` instalado para os
dois lados): collected 1691→1711, passed 1375→1394, failed 78→79 (o
único delta: um teste que dependia de "4 componentes = cobertura total"
foi dividido em 2, ambos falhando pelo MESMO bug de ambiente
pré-existente, não relacionado — PyTorch deixando o device default em
"mps" a partir de um teste anterior da suite, o que já quebrava 2 outros
testes de calibração antes desta sessão também), error 40→40 idêntico,
skipped 197→197 idêntico. Diff completo dos IDs FAILED/ERROR mostra
exatamente essa 1 mudança esperada, nada mais. Ver `docs/dev/
AUDIT_REPORT.md` para a análise completa.

### Paper-to-Repro Benchmark Suite
**Projeto novo.** Pipeline que extrai método e resultados numéricos de
papers do arXiv e usa um agente tipo ATHENA para reproduzi-los, virando um
benchmark contínuo de "quão bem uma IA reproduz física publicada". Fonte:
arXiv — ATHENA + ideia própria. Precedente real direto:
`pinn_reproduction_results` (reprodução de Raissi et al. 2017 e Lu et al.
2019, incluindo bugs reais encontrados e corrigidos) — este projeto
generaliza esse esforço manual em um pipeline automatizado e contínuo.

### Corpus de referência PIELM/XTFC/TFC/OpInf (material recebido, não integrado)
**Peça em aberto.** `PINNeAPPle-Talk/resources/` (ver `MANIFEST.md`)
recebeu nesta sessão implementações reais em MATLAB de PIELM e X-TFC para
PDEs de advecção-difusão (`Advection-Diffusion_PIELM-&-XTFC/`,
`PDE_matlab/`, `Laura/`, `codes/`, `common/`), mais uma pasta de
referências dedicada a TFC (`References/TFC/`) e papers de Operator
Inference/POD (`OpInf_summary_2022{a,b}.pdf`, `POD-ROM.pdf`, em
`deeponet_papers_and_notebooks/`). Conecta diretamente com módulos já
reais: `pinneapple_neural/architectures/rom/{opinf,pod}.py` (existentes)
e o `TFC/ELM` já usado em produção pelo `HelioTFC` (`pinneapple-apps`).
Vale avaliar se o código MATLAB tem algo que valide/estenda o `OpInf`/
`POD` já implementados, antes de tratar como só leitura de referência.

### CrunchOptimizer/PINNs — SS-Quasi-Newton (SSBFGS/SSBroyden)
**Projeto novo.** Otimizadores quasi-Newton curvature-aware
(SSBFGS/SSBroyden), construídos sobre Optimistix (JAX), para treino de
PINN de alta precisão — vão além do que Adam/L-BFGS entregam. Fonte:
[CrunchOptimizer/PINNs](https://github.com/CrunchOptimizer/PINNs)
(verificado nesta sessão: "Curvature-Aware Optimization for
High-Precision Physics-Informed Neural Networks"). Conexão direta com
achado real já documentado em `pinn_reproduction_results/INSIGHTS.md`:
trocar Adam por L-BFGS já cortou o erro em 100-1000x nos problemas de
Burgers/Allen-Cahn/Schrödinger — SSBFGS/SSBroyden são o próximo passo
natural dessa mesma descoberta. Módulo relacionado:
`pinneapple_neural.trainer` + `pinneapple_tools.compute_backends` (já tem
backend JAX, onde o Optimistix vive nativamente).

### `grad_method` pluggável em `SymbolicPDE` (autograd/finite_difference/spectral) (2026-09-17)
**Já corrigido/implementado.** Insight importado do PhysicsNeMo
(`physicsnemo-notes/insights-to-import-into-pinneapple.md`, seções 2-3):
separar "o que é a PDE" (SymPy, já existia) de "como calcular a derivada
dado o formato do dado" (autograd vs. grade). `pinneapple_physics/
symbolic_pde/compiler.py::SymbolicPDE` ganhou `grad_method` (default
`"autograd"`, comportamento 100% preservado — não alterei uma linha do
código antigo, só adicionei um `to_grid_residual_fn` irmão) reaproveitando
de verdade a lógica de FD/espectral já real em `pinneapple_neural/
architectures/neural_operators/pino.py` (import lazy, sem custo pra quem
usa o default). **Correção de premissa relevante**: o doc de insights
afirmava que `SymbolicPDE` alimenta o catálogo de 65 presets — não
alimenta; o catálogo real passa por `pinn_solver/compiler/compile.py`
(dispatch por string `kind`, engine totalmente separada, não tocada aqui).
Validado comparando os 3 backends nos mesmos kinds reais (`laplace`,
`burgers`, `reaction_diffusion_2d`) — spectral bate com autograd a ~1e-13,
FD a ~1e-3..1e-4 (O(dx²), como esperado). Suite completa antes/depois:
1732→1737 testes (5 novos, todos passando), **zero mudança nos 75
failed/163 skipped/1 xfail pré-existentes** (nomes dos testes que falham
conferidos byte a byte, não só a contagem). Detalhes completos, incluindo
um bug real e não-relacionado de vazamento de `torch` default device
("mps") encontrado (e contornado, não corrigido — fora do escopo) durante
a rodada de testes, em `docs/dev/AUDIT_REPORT.md`.

---

## 4. Simulação em escala / CFD industrial

### Autonomous DOE-CFD
**Projeto novo**, com blueprint técnico validado em escala industrial.
Geometria paramétrica via signed distance fields (sem falha topológica ao
variar parâmetros) + burst de GPU em nuvem, para rodar milhares de
variantes de reator/tanque automaticamente. Aplicável à otimização do
misturador no roadmap de cloramina. Fonte: nTop / CoreWeave — NASA 2030
Grand Challenge in CFD (verificado nesta sessão: 280 GPUs NVIDIA RTX Pro
Blackwell, 2.400 geometrias × 5 ângulos de ataque = 12.000 simulações em
menos de 24h, zero falhas de geometria, usando LBM sobre grid cartesiano
fixo — meta que a NASA tinha projetado só para 2030, alcançada 4 anos
antes). É o blueprint concreto de como construir a "PINNeAPPle Surrogate
Factory" (§1) em escala real: geometria implícita (SDF) em vez de
b-rep é o que evita a quebra topológica que travaria um DOE massivo, e
LBM é o solver certo para isso porque opera sobre grid fixo sem malha.

### The Well Benchmark Adapter
**Projeto novo.** Usar o dataset físico de 15 TB da Polymathic AI ("The
Well") como pretraining/benchmark externo para testar se os surrogates do
ChordIQ generalizam fora do domínio de mistura/cloramina. Fonte:
[PolymathicAI/the_well](https://github.com/PolymathicAI/the_well)
(verificado nesta sessão: 16 datasets, 6.9 GB–5.1 TB cada, cobrindo
fluidos, MHD, supernovas, convecção, espalhamento acústico e sistemas
biológicos; instalável via PyPI/HuggingFace Hub, interface
`WellDataset` compatível com `torch.utils.data.DataLoader`). Vários
desses domínios já têm um preset equivalente no PINNeAPPle (Navier-
Stokes, onda/acústica, reação-difusão) — o adaptador concreto é
`pinneapple_data.adapters.well_adapter` (novo módulo, mesmo formato de
saída que `load_dense_volumes`, já usado por `pinneapple_splash`),
convertendo um `WellDataset` em tensores de campo no formato interno do
PINNeAPPle. Três usos concretos, em ordem de esforço: (1) benchmark de
generalização OOD para o FNO3d já treinado em
`PINNeAPPle-SplashCFD` (treinar em canal turbulento próprio, avaliar
contra o subconjunto de fluidos do Well, sem re-treinar); (2) nova
categoria "dataset" no catálogo externo do `PINNeAPPle-arena` (§1) — o
catálogo hoje só tipa *modelos* (`ok`/`not_installed`/...), Well é o
primeiro caso real de "dataset externo" e expõe esse gap de schema; (3)
corpus de pretraining para o `pinneapple_worldmodel` (Physics Foundation
Model generalista), o uso mais caro e mais alinhado à visão de longo
prazo desse módulo.

**Pesquisa verificada em 2026-09-13 — o uso (3) acima já tem um
resultado publicado que vale ler antes de implementar do zero.**
"Towards a Foundation Model for Partial Differential Equations Across
Physics Domains" ([arXiv:2511.21861](https://arxiv.org/abs/2511.21861),
lido de verdade via fetch, não só título) descreve o **PDE-FM**: um
backbone Mamba (state-space model) com tokenização espectral-espacial e
condicionamento físico, pré-treinado em 12 datasets 2D/3D **do próprio
The Well** cobrindo hidrodinâmica, sistemas radiativos, elasticidade e
astrofísica — SOTA em 6 dos domínios testados, com redução relativa de
46% no VRMSE médio contra baselines de neural operator anteriores. Não
há repositório de código público mencionado no paper (verificado — não
assumir que existe). Isto muda a ordem de prioridade sugerida acima: o
uso (3) (`pinneapple_worldmodel` como Physics Foundation Model) não
precisa começar do zero nem apenas do uso (1)/(2) — replicar/adaptar a
arquitetura do PDE-FM (Mamba + tokenização espectral, publicamente
descrita ainda que sem código) sobre o mesmo The Well é um ponto de
partida mais barato do que desenhar uma arquitetura de foundation model
própria. Módulo relacionado adicional:
[jNO](https://arxiv.org/pdf/2605.10159) (biblioteca JAX para treino de
neural operators/foundation models, verificado via arXiv) conecta
diretamente com o backend JAX que `pinneapple_tools.compute_backends`
já expõe, evitando reimplementar a parte de infraestrutura de treino em
JAX do zero.

---

## 5. World models / robótica

### Cosmos Synthetic Perception Pretrainer
**Projeto novo.** Usar os world models do NVIDIA Cosmos (modo Generator)
para gerar vídeo+ação sintéticos de processos industriais, pré-treinando
visão para monitoramento de planta/robótica. Fonte: NVIDIA/Cosmos. Nota:
o modelo de visão da família `nvidia/Cosmos-Reason2-2B` já está em uso
real no `physcurator` (curadoria de dados sintéticos antes do treino de
PINN) — este projeto é o mesmo ecossistema Cosmos aplicado à geração
(em vez de à curadoria).

### Low-Cost Autonomous Robot
**Projeto novo.** Plataforma robótica autônoma de baixo custo (Raspberry
Pi/ESP32 + sensores low-cost) para testes de campo e monitoramento.
Fonte: brainstorm interno — "Criar robô autônomo barato".

---

## 6. Novos domínios físicos (flagship demos)

### Space Debris Tracking & Collision Risk
**Sobreposição parcial**: `ge3` (RK4 3DOF/ISA-76/Barrowman). Trajetória
orbital com perturbações (drag atmosférico, J2, pressão de radiação
solar) + surrogate PINNeAPPle + estimativa de risco de colisão,
visualizado em 3D. Nota: `pinneapple-apps/satellite_conjunction_
screening` (SatScreen) já cobre boa parte disso como produto comercial
(CW relative motion, J2, Kepler, CR3BP + fórmula própria de Pc) — este
item é o flagship demo público, SatScreen é a versão paga.

### Airfoil / Engineering Design Optimization
**Já existe**: `portfolio/pinneapple/missile_aero` (Cp R² 0.99) +
`portfolio/01_automotive_aero`. Geometria → CFD dataset → surrogate
(FNO/DeepONet) → otimização de milhares de designs → melhor geometria
(lift/drag).

### Astrophysical Parameter Discovery
**Sobreposição parcial**: `ge3` (lightkurve/BLS) — falta a camada de UQ.
Inferir massa, raio, temperatura, metalicidade e idade estelar a partir
de curva de luz observada, com quantificação de incerteza.

### Satellite Thermal Digital Twin
**Projeto novo.** Modelo térmico de satélite (radiação solar/terrestre,
orientação, propriedades de material) aprendido por PINNeAPPle, com
recomendação de orientação que minimiza estresse térmico.

### Drilling Hydraulics Digital Twin / Smart Mud Pump
**Sobreposição parcial**: base já existe em
`~/Documents/GitHub/biaml/database/geometries` (BOP, desanders, chokes).
Estado de hidráulica de perfuração (ECD, pressão, vazão, temperatura) em
tempo real + otimização da bomba de lama para minimizar energia mantendo
ECD dentro do limite de segurança.

### Digital Twin Setorial (padrão replicável)
**Sobreposição parcial**: `portfolio/03_mixing_tank` e
`portfolio/02_wind_turbine`. Pipeline único reaproveitável —
sensores/CAD → PINNeAPPle → surrogate → predição → otimização — aplicado
a vários setores: bateria/EV (gestão térmica), wind farm (yaw, +8,7%
potência), HVAC predial (-21% energia), trocador de calor (+8%
recuperação), estrutural (manutenção preditiva), power grid, braço
robótico (MPC), fábrica (anomalia) e bombeamento industrial.

### Flood Prediction / Urban Hydrology
**Projeto novo.** Modelo físico de chuva + terreno (DEM) + hidrologia →
mapa previsto de profundidade de inundação ao longo do tempo, comparado a
interpolação tradicional.

### Climate Downscaling
**Projeto novo.** Modelo Physics AI que refina resolução de simulação
climática (25 km → 1 km) para temperatura, precipitação, vento e
umidade, comparado contra interpolação e ML puro.

### Ocean Physics AI Explorer
**Sobreposição parcial — peça real já existe.** Reconstrução de campos
oceânicos (temperatura/salinidade como traçador 2D) a partir de
observações esparsas, com um wrapper real (opcional) para a rede Argo de
boias reais via `argopy`, comparando interpolação determinística contra
um PINN físico e dois neural operators (FNO, DeepONet) contra uma
verdade-terreno numérica exata (solver FDM próprio, validado contra a
solução analítica fechada de difusão gaussiana), mais seleção ativa do
próximo ponto de observação via incerteza epistêmica
(`x* = argmax_x U(x)`, usando `pinneapple_analysis.uncertainty.
uq_predict` real). Implementado em `PINNeAPPle-Research/ocean/` (pacote
irmão do `research/` já existente nesse repo, ambos como camadas finas
sobre PINNeAPPle) — ver seu próprio `README.md`/`ROADMAP.md` para o que
já funciona (reconstrução sintética de ponta a ponta, testada) versus o
que falta (3D real, digital twin contínuo, ingestão Argo em streaming,
integração com AUV real ou simulado — a peça que fecharia o loop
"hipótese → explorar → observar → atualizar hipótese" descrito na
motivação original deste projeto).

### Materials Inverse Design
**Projeto novo.** Dado um requisito de propriedade alvo (ex.:
condutividade térmica), buscar composição/microestrutura candidata via
formulação física + PINNeAPPle + busca.

### Biomechanics AI
**Projeto novo.** Prever distribuição de pressão/deformação em uma
articulação a partir de geometria, propriedades de material e
carregamento, com Neural Operator substituindo FEM tradicional (17 min →
0,4 s por caso).

### Solar Farm / Solar Panel Optimization
**Projeto novo.** Prever irradiância, temperatura e cobertura de nuvens,
e otimizar orientação dos painéis, cronograma de limpeza e resfriamento
para maximizar energia e minimizar custo operacional.

### Fire Spread / Wildfire PINN
**Projeto novo**, com uma semente de código real. Existe um script
standalone `Fire_PINN_PlusLowDifusion.py` em
`PINNeAPPle-Talk/resources/deeponet_papers_and_notebooks/` (ver
`PINNeAPPle-Talk/resources/MANIFEST.md`) — um PINN para dinâmica de
propagação de fogo/baixa difusão, nunca integrado a este portfólio.
Avaliar se vale a pena portar como ponto de partida em vez de começar do
zero.

### Optimal Control / Nuclear & Radiative Transport (domínios não cobertos)
**Peça em aberto — nenhum repo do ecossistema cobre isso hoje.** Um
levantamento de referências recebido nesta sessão (`PINNeAPPle-Talk/
resources/References/`, ver MANIFEST) tem pastas inteiras dedicadas a
controle ótimo/HJB/GNC/controle adaptativo (`Optimal Control /`,
`Roberto-Suggestions/`) e a transporte nuclear/radiativo/equações de
cinética pontual (`Transport/{Radiative,Neutron,PKE}`) — nenhum dos dois
domínios existe em nenhum repo do PINNeAPPle-Labs hoje. Não é
necessariamente um novo produto, mas vale decidir deliberadamente se
algum dos dois merece entrar no catálogo de domínios físicos antes de
continuar acumulando referência sem repo correspondente.

---

## 7. GPU-native / verificação de hardware / plataformas comerciais (referências externas, fora do catálogo CoupleTasks)

### LatticePT Boltzmann Reactor: Process Studio
**Projeto novo.** App CFD+CHT (conjugate heat transfer) 100%
GPU-nativo em navegador: geometria paramétrica em OpenCascade → malha de
casca QUAD8 → volume fluido voxelizado por marching cubes → LBM+LES com
immersed boundary method para agitadores móveis → coeficientes de
convecção nas paredes molhadas pelo processo (correlações Kawase-Moo &
John Thomas) → lado da jaqueta via correlação de Gnielinski → acoplamento
explícito de condução através dos elementos de casca. Roda localmente no
Chrome, sem servidor, e é deployável via **Vulkan** em qualquer GPU
(AMD/Intel/NVIDIA/Apple). Fonte: post do LinkedIn (LATTICEPT, "Reactor
Lab"), construído com Anthropic Fable 5.1 em 4 semanas. Módulos
relacionados: `pinneapple_simulation.numerical_solvers` (já tem LBM),
`pinneapple_design.geometry` (SDF/CSG — falta OpenCascade paramétrico),
`pinneapple_tools.compute_backends` (PyTorch+JAX hoje — falta um backend
WebGPU/Vulkan para rodar no browser), `pinneapple_systems.digital_twin`
(agitador móvel + jaqueta térmica = twin clássico). Mesma direção do
"Multi-Physics Sandbox" (§3, Genesis) no eixo de runtime GPU-nativo
multiplataforma.

### OpenV — Verification-first hardware engineering
**Projeto novo.** Pipeline open-source onde um agente (Astra) propõe
requisitos/design de hardware, ferramentas externas de CAD/cálculo/solver
produzem evidência, e uma comparação determinística decide
PASS/FAIL/UNKNOWN — o LLM nunca avalia seu próprio trabalho. Inclui
invalidação automática de evidência dependente quando o design muda, e
histórico rastreável de cada experimento (hipótese → mudança → efeito
esperado → efeito real) via um modelo de engenharia persistente (Dalus
via MCP). Fonte: [sebastianvkl/OpenV](https://github.com/sebastianvkl/OpenV)
(verificado nesta sessão). Conexão direta: é o mesmo princípio do
`veriphysics` (execução + verificação + evidência, Decision Record,
trust score) aplicado a hardware/CAD em vez de PDEs — o mecanismo de
"invalidação automática de evidência dependente ao mudar o design" é
genuinamente novo e vale portar para `pinneapple_analysis.verification
.evidence_graph`/`provenance`.

### Luminary Cloud — Physics AI Stack (referência competitiva)
**Projeto novo** (pesquisa de benchmark, não integração de código).
Todo o conteúdo relevante em luminary.ai/resources: modelos "Physics AI"
treinados sobre simulação para acelerar design de engenharia
(aeroespacial, automotivo, defesa), arquitetura mesh-independent
("Luminary-SMART"), UQ/validação, cases como "Physics AI Cuts Aircraft
Design Costs By 80%" (verificado nesta sessão via fetch da página).
Uso recomendado: benchmark de posicionamento para `pinneapple-apps`
(especialmente `VerifiedPhysics`/`physics_verification_engine`) e para o
`PINNeAPPle-arena` — no espírito anti-fabricação já documentado no
`tool_recommendation.py`, qualquer claim tipo "-80% custo" deveria ser
tratada como claim de terceiro a verificar independentemente, não
absorvida como fato.

### Reality2Physics estendido — Digital Complex-Systems Benchmark
**Projeto novo** (o mais especulativo e mais amplo desta lista).
Generaliza o `reality2physics` (hoje: vídeo/sensor → campo físico +
parâmetro escalar, validado em 3 PDEs) para sistemas complexos onde a
"física" é a dinâmica interna de um organismo/sistema, seguindo sempre o
mesmo padrão: **Observar → Reconstruir → Inferir → Simular → Perturbar →
Validar**. Ponto de partida recomendado (mais tratável, conectoma
completo + músculos + circuitos já públicos via OpenWorm):
**C. elegans Digital Twin** — connectome (~300 neurônios) → grafo neural
→ modelo de neurônio → modelo de músculo → biomecânica do corpo →
ambiente → comportamento; validação = "o modelo reconstruído reproduz
locomoção real?". Extensões mapeadas na mesma sessão de brainstorm
(cada uma como capítulo futuro do mesmo benchmark, não itens separados):
- **Drosophila** via FlyWire/BANC/hemibrain (~140k neurônios, 50M+
  sinapses) — conectoma maduro, mas só estrutura; pesos sinápticos,
  neurotransmissores e corpo ainda precisam ser inferidos/calibrados.
- **MICrONS** (córtex visual de camundongo, Allen Institute) — o dataset
  mais forte por combinar estrutura *e* função nos mesmos neurônios
  (~1mm³, centenas de milhões de sinapses, dezenas de milhares de
  neurônios com atividade registrada).
- **Mind2Physics** — tratar estado psicológico como sistema dinâmico
  parcialmente observável (θ\* = argmin L(comportamento_sim,
  comportamento_real)) em vez de classificador — mesmo paradigma
  PINN/inverse-problem aplicado a séries temporais comportamentais.
- **PINNeAPPle Anomaly Lab** — mesmo pipeline aplicado a alegações
  paranormais/anômalas (casas "assombradas", UAP, EVP) como
  reconstrução física multi-sensor + teste cego + classificação em
  Explained/Unresolved/Reproducibly anomalous — valor científico está no
  rigor do protocolo, não na conclusão.
- **PINNeAPPle Football Dynamics** — tracking de 22 jogadores + bola como
  sistema dinâmico multiagente; simulação contrafactual ("e se o
  jogador X estivesse 3m à esquerda?"), papel emergente de jogador,
  time como rede/campo de influência.
Todas compartilham a mesma pergunta central: dado um sistema
parcialmente observável, consigo reconstruir a dinâmica que produz o
comportamento observado, e depois perturbar essa reconstrução para
prever algo que ainda não vi? Módulo relacionado: `pinneapple_worldmodel`
(agentes/ambientes) + `reality2physics` como base de pipeline.

---

## 8. Structure Discovery / Chaos-to-Law — nova vertente

Trazido por Yan nesta sessão como uma pergunta diferente da que o
portfólio normalmente faz. Em vez de "como eu preveja este sistema",
perguntar **"que estrutura mais simples explica este sistema"** —
geometria intrínseca, dinâmica reduzida, causalidade, invariantes,
regimes de caos/ordem, eventos extremos. O item "Physics Discovery /
Equation Discovery" (§1) já é uma instância real e validada disso; esta
seção generaliza a mesma pergunta para eixos que o portfólio ainda não
cobre, verificado diretamente no código desta sessão (não assumido).

Pipeline-alvo da vertente inteira:

```
sistema observado
   ├─ geometria  (manifold/TDA)
   ├─ dinâmica   (Koopman/DMD/SINDy)  ── já existe, ver abaixo
   └─ causalidade (GNN/NOTEARS/PCMCI)
        │
   estrutura latente
        │
   invariantes / leis / regimes
        │
   modelo físico → PINN / FNO / GNN (motor já existente do PINNeAPPle)
```

### Dynamics-to-Law: SINDy / Koopman / DMD / HAVOK / POD / OpInf
**Já existe**, o pilar mais maduro desta lista de longe.
`pinneapple_neural/architectures/rom/` já implementa `SINDy`,
`DynamicModeDecomposition`, `HAVOK` (Hankel-DMD via delay embedding),
`KoopmanAutoencoder` (uma segunda implementação vive em
`reservoir_computing/koopman.py`), `OperatorInference`, `POD`,
`NeuralROM`, `ROMHybrid` e `DeepUQROM`, todos catalogados via
`ROMCatalog` (`rom/registry.py`). O precedente citado em §1
(`inverse_sindy` redescobrindo Lorenz via EKI+SINDy, 100% de acerto
estrutural) é a validação ponta a ponta deste pilar — é literalmente o
"problema concreto 1" do brainstorm desta sessão (Lorenz → descoberta
automática da estrutura), só que já feito. **Falta**: um benchmark
dedicado comparando SINDy vs. Koopman vs. DMD/HAVOK na mesma bateria de
sistemas (ver "Bateria de validação" abaixo) com métrica de *acerto
estrutural*, não só erro numérico — encaixe natural em
`PINNeAPPle-arena` (§1), no mesmo espírito do benchmark PINN/FNO/DeepONet
que já existe lá.

### Symbolic regression livre (estilo PySR / AI Feynman)
**Peça em aberto.** `pinneapple_neural/trainer/graybox.py` já antecipa a
ideia no próprio docstring do `GrayBoxNet` ("if the term is later
distilled into a closed-form expression, e.g. via symbolic regression")
mas não implementa busca simbólica livre — hoje toda "descoberta de
equação" do portfólio é restrita a uma base de termos conhecida (SINDy)
ou a uma rede substituta (gray-box), nunca uma busca evolutiva por
expressão fechada como PySR/AI Feynman. Projeto novo:
`pinneapple_neural.architectures.symbolic` — wrapper sobre PySR com
verificação determinística de erro de ajuste antes de aceitar qualquer
expressão (mesmo princípio anti-fabricação do resto do portfólio), com
uma rota explícita para "distilar" um `GrayBoxNet` já treinado numa
expressão fechada. **Pesquisa verificada em 2026-09-13** (busca +
leitura real, não só título): PySR segue sendo a biblioteca de
referência (confirmado por um estudo comparativo recente que a aponta
como a mais adequada para recuperar equações de 9 processos dinâmicos,
incluindo dinâmica caótica e modelos epidêmicos), mas duas extensões
recentes valem avaliar antes de implementar o wrapper do zero: **ANN-
PySR** combina PySR com uma rede de atenção residual para identificar
PDEs a partir de dados esparsos e ruidosos, com speedup de quase duas
ordens de magnitude; e "Knowledge integration for physics-informed
symbolic regression using pre-trained large language models" (Nature
*Scientific Reports*, 2026,
[doi.org/10.1038/s41598-026-35327-6](https://www.nature.com/articles/s41598-026-35327-6))
usa um LLM para injetar conhecimento de domínio na busca simbólica —
conexão direta com `pinneapple_llm` (o mesmo `PhysicsGuardrail` já usado
em outros pontos do portfólio serviria de verificação determinística
sobre qualquer expressão que o LLM ajude a propor, mantendo o LLM fora
do papel de juiz).

### Geometria / manifold escondido
**Projeto novo — gap confirmado no código.** Nenhuma implementação de
UMAP, Isomap, Diffusion Maps ou autoencoder-como-manifold-discovery
encontrada no repo (`POD` é hoje a única redução de dimensionalidade, e é
linear). Diffusion Maps é o candidato mais interessante trazido nesta
sessão: descobre a geometria intrínseca de dados de alta dimensão sem
assumir linearidade — complementa, não substitui, o `POD`/`DMD` lineares
já existentes. Composição proposta: dados observados → alta dimensão →
Diffusion Maps/UMAP → coordenadas intrínsecas → alimentar como input do
`SINDy`/`KoopmanAutoencoder` já existentes (manifold discovery + SINDy +
PINN). Módulo relacionado: novo `pinneapple_neural.architectures.manifold`
ao lado de `rom/`. Referência já disponível: "Elements of Dimensionality
Reduction and Manifold Learning" (Ghojogh, Crowley, Karray, Ghodsi,
Springer 2023) está duplicado em `PINNeAPPle-Talk/resources/
deeponet_papers_and_notebooks/Papers e Documentos/` e em
`PINNeAPPle-Talk/resources/cfd_pde_neuralnets_notebooks/` (ver MANIFEST).
**Pesquisa verificada em 2026-09-13**: existe um candidato Python
pip-instalável real e pronto para fechar esta lacuna sem esperar por
`pinneapple_neural.architectures.manifold` inteiro —
[pyDiffMap](https://github.com/DiffusionMapsAcademics/pyDiffMap)
(`pip install pyDiffMap`, MIT, diffusion maps de largura de banda
variável + extensão out-of-sample) — verificado via GitHub/PyPI, não
apenas citado de memória. Conexão direta com um paper real também
verificado nesta sessão: "Nonlinear dimensionality reduction then and
now: AIMs for dissipative PDEs in the ML era"
([arXiv:2310.15816](https://arxiv.org/pdf/2310.15816)) aplica
exatamente esta técnica (Diffusion Maps) para reduzir a dimensionalidade
de PDEs dissipativas — o mesmo tipo de problema que `pinneapple_physics`
já resolve com PINN, dando um caminho de validação concreto (dataset
sintético de uma PDE dissipativa já suportada, comparar redução via
`pyDiffMap` contra `POD` linear existente) antes de generalizar para
dados observacionais arbitrários.

### Invariant Discovery Engine
**Projeto novo.** Hoje o portfólio só *impõe* invariantes já conhecidos
(ex.: divergente nulo em `reality2physics`, leis de conservação como
termo de perda nos PINNs) — nada *descobre* uma quantidade conservada
desconhecida a partir de trajetórias observadas. Ideia concreta desta
sessão: treinar uma rede pequena `I_θ(x)` penalizando `dI_θ/dt` ao longo
de trajetórias reais e, depois, tentar destilar `I_θ` numa expressão
simbólica via o item de symbolic regression acima — a mesma composição
"descobrir → destilar" do resto desta seção. Conecta com `veriphysics`:
um invariante descoberto e destilado é exatamente o tipo de claim que o
Decision Record/trust score do `veriphysics` deveria verificar
independentemente antes de ser aceito como real, não só o PINNeAPPle
produzindo o número.

### Transição ordem↔caos (Lyapunov, RQA, bifurcação, entropia)
**Projeto novo — gap confirmado no código.** Nenhuma métrica de caos
(maior expoente de Lyapunov, entropia de Kolmogorov-Sinai/permutação,
dimensão de correlação, recurrence plots/RQA, seções de Poincaré,
análise de bifurcação) encontrada no repo. É infraestrutura de
diagnóstico, não um modelo — barata de construir e reutilizável por
qualquer item acima (ex.: usar RQA para decidir automaticamente se um
sistema está no regime em que SINDy/Koopman conseguem generalizar, antes
de gastar treino de verdade). Módulo relacionado: novo
`pinneapple_analysis.chaos_metrics`. **Pesquisa verificada em
2026-09-13**: [nolds](https://github.com/CSchoel/nolds) (`pip install
nolds`, numpy puro, confirmado via PyPI/GitHub) já implementa maior
expoente de Lyapunov (algoritmos de Rosenstein e de Eckmann), expoente
de Hurst, entropia amostral, dimensão de correlação e DFA — cobre a
maior parte desta lista pronto para uso, sem precisar reimplementar os
algoritmos numéricos; `pinneapple_analysis.chaos_metrics` pode nascer
como wrapper fino sobre `nolds` mais o que faltar (seções de Poincaré,
análise de bifurcação, RQA), em vez de uma biblioteca do zero. Para
escala (séries muito longas ou lote de muitas trajetórias): "Chaoticus:
a parallel approach to the computation of chaos indicators"
([arXiv:2507.00622](https://arxiv.org/pdf/2507.00622)) é uma referência
recente para uma versão paralela/GPU, relevante só se `nolds` (CPU,
single-series) virar gargalo real.

### Causal discovery estrutural (GNN / NOTEARS / PCMCI)
**Projeto novo — gap confirmado no código.** Nenhum algoritmo de
descoberta causal (NOTEARS, PCMCI+, LiNGAM, Neural Relational Inference,
GNN causal) encontrado no repo. Pergunta central: dado `x_1(t), ...,
x_n(t)` de um sistema de alta dimensão, existe uma estrutura causal
esparsa por trás da bagunça estatística? Conecta com `pinneapple_worldmodel`
(agentes/ambientes, já citado em §7 para os digital twins biológicos) —
descoberta causal seria o passo que precede a construção de qualquer um
daqueles twins a partir de dados observacionais puros, em vez de assumir
a topologia do grafo a priori. **Pesquisa verificada em 2026-09-13**:
dois candidatos reais, ambos abertos e verificados via GitHub/PyPI/
arXiv, não apenas citados de memória —
[Tigramite](https://github.com/jakobrunge/tigramite) (`pip install
tigramite`, GPL-3, mantido por Jakob Runge, implementa PCMCI/PCMCIplus —
a referência estabelecida para causalidade em série temporal, incluindo
laggeds e contemporâneos) e
[Causal-TS](https://github.com/bloomberg/causal-ts) (Bloomberg, 2026,
`pip install`-ável, testado em Python 3.10–3.12, quatro algoritmos
próprios — CDNOTS/CDNOTS+/CEDAR/GRACE — mais wrappers para GES/Granger/
LASSO-VAR, teste de independência condicional acelerado por GPU via
PyTorch, e um pipeline de "regime discovery" para quebras estruturais em
séries não-estacionárias). Tigramite é o ponto de entrada mais maduro e
testado; Causal-TS é mais recente e endereça especificamente
não-estacionariedade — relevante porque sistemas físicos reais (troca de
regime, degradação de equipamento) raramente são estacionários, o mesmo
problema que já motiva o item "Transição ordem↔caos" acima.

### Extreme events / rare-event discovery
**Projeto novo.** Nenhuma infraestrutura de Extreme Value Theory,
rare-event/importance sampling ou large deviation theory encontrada.
Em vez de estudar o comportamento médio, estudar o que produz os eventos
raros (turbulência extrema, falha industrial, crash) — pergunta natural
para o mesmo domínio industrial que já motiva `pinneapple_systems.digital_twin`
e os produtos de monitoramento do `PINNeAPPle-apps`. Ponto de entrada
mais barato: aplicar EVT sobre os mesmos dados de sensor/SCADA que já
alimentam twins industriais existentes (ex. `shinagawa-ai-platform`,
hoje em `ChordIQ-tech` — o método é agnóstico a onde os dados moram)
antes de qualquer simulação nova de rare-event.

### Bateria de validação (extensão do PINNeAPPle-arena)
**Projeto novo**, extensão natural do `PINNeAPPle-arena` (§1) — antes de
tratar "Structure Discovery" como produto, validar contra ground truth
conhecida, no mesmo espírito anti-fabricação do `PINNeAPPle-Research`
(que só pontua contra problemas já resolvidos). Bateria mínima trazida
nesta sessão:
1. **Lorenz → recuperação da equação** — já validado via `inverse_sindy`
   (ver acima); vira o baseline "fácil" da bateria.
2. **Navier-Stokes turbulento → detecção automática de troca de regime**
   (POD + Koopman + Lyapunov + clustering, todos já existentes ou
   listados acima).
3. **Sistemas multiestáveis** — aprender bacias de atração automaticamente.
4. **Rare events** — caminho mais provável de um estado normal a um
   evento extremo.
5. **Invariant discovery cego** — dado só `x(t), y(t), z(t)`, redescobrir
   o que é conservado.
6. **Generalização entre regimes nunca vistos** (treinar em `Re_1, Re_2`,
   testar em `Re_3`) — testa se o método descobre a lei, não memoriza o
   regime.
7. **Universal Structure Discovery** (o item mais ambicioso): dado
   qualquer sinal (série temporal, imagem, grafo, simulação) sem dizer a
   matemática a priori, decidir sozinho qual das perguntas acima se
   aplica — e produzir uma explicação, não só uma previsão.

### AGI Evaluation Engine — generaliza a "Bateria de validação" em produto próprio
**Projeto novo, trazido por Yan em 2026-09-13.** Não é uma ideia nova
para este roadmap — é uma formalização do item "Bateria de validação"
acima como um produto/framework com identidade própria, em vez de um
apêndice do `PINNeAPPle-arena`. A tese central trazida na sessão: para
medir se um sistema tem capacidades de tipo-AGI (generalização,
descoberta de estrutura, planejamento, adaptação), **nenhuma parte do
pipeline de avaliação em si precisa de LLM** — só o candidato avaliado
pode, opcionalmente, ser um LLM. Isso separa "como construo um teste de
inteligência" de "qual modelo uso para tentar resolvê-lo", e é
literalmente o mesmo princípio anti-fabricação que `PINNeAPPle-Research`
já aplica a matemática (pontua contra problema JÁ resolvido, nunca
LLM-julgando-LLM) — aqui generalizado para PDEs, dinâmica e causalidade.

Arquitetura proposta (todas as peças reaproveitam módulos reais já
existentes, nenhuma reimplementação do zero):

```
Task Generator ──► World Generator ──► Environment ──► Agent Interface
                                                              │
                                                        Evaluator (sem LLM)
                                                              │
                                                    Capability Vector / AGI Profile
```

| Peça | O que faz | Módulo real a reaproveitar |
|---|---|---|
| **World Generator** | Gera mundos paramétricos com verdade-base conhecida (porque foi gerado) | `pinneapple_physics.pde_environment` (presets/BCs/ICs), `pinneapple_design.geometry` (SDF/CSG para geometria procedural), sistemas caóticos já usados em `inverse_sindy` (Lorenz, e por extensão Rössler/logistic map) |
| **Task Generator** | Famílias paramétricas de problema (causal, dinâmico, geométrico, compositivo), não perguntas de linguagem natural | Reaproveita a "Bateria de validação" acima como conjunto inicial de famílias — não recomeça do zero |
| **Environment** | Loop observação → ação → novo estado, para tarefas com intervenção (não só previsão passiva) | Candidato natural: `pinneapple_worldmodel` (ambientes/agentes já existentes na visão desse módulo) |
| **Agent Interface** | Qualquer candidato plugável — PINN, RL, busca, ou um LLM — nunca privilegiado na avaliação | `pinnaitor` (framework de agentes de propósito geral do próprio org) é um substrato candidato para construir agentes-candidato, não para julgar resultado |
| **Evaluator** | Escora contra verdade-base conhecida (a mesma que gerou o mundo), nunca um segundo LLM julgando o primeiro | Mesmo padrão determinístico de `PINNeAPPle-Research/research/benchmark.py` (overlap de conjunto real, sem chamada de LLM na pontuação) |
| **Capability Vector / AGI Profile** | Perfil multi-eixo (causal inference, generalização, adaptação, descoberta de invariante, ...) em vez de um único score de acurácia | Nova dimensão de leaderboard para `PINNeAPPle-arena` (§1) — nem toda tarefa vira um número de "quem ganhou", vira um radar de capacidades |

Eixos de tarefa que a bateria acima já cobre parcialmente e este produto
formaliza: descoberta de lei física (SINDy/Koopman, já real), geometria/
manifold escondido (gap, já mapeado acima), causalidade estrutural (gap,
já mapeado acima), transição ordem↔caos (gap, já mapeado acima),
generalização estrutural entre sistemas com a mesma lei mas superfície
diferente (ex.: treinar em oscilador mecânico, testar em circuito LC —
mesma EDO, objetos diferentes; nenhuma infraestrutura disso existe hoje,
gap novo), e inteligência composicional (compor duas leis físicas
aprendidas separadamente sem tê-las visto compostas antes; gap novo,
nenhuma implementação encontrada).

**Pesquisa verificada em 2026-09-13 — validação externa do princípio
central, não apenas inspiração.** O ARC Prize (arcprize.org) é a
confirmação mais forte disponível hoje de que "avaliação determinística,
sem LLM-juiz" é viável em escala real, não só uma boa intenção deste
roadmap: **ARC-AGI-2** usa "no smoothing, no rubric judgement, and no
LLM-as-judge intermediary" (verificado via Epoch AI/arcprize.org) — cada
tarefa é validada por múltiplos humanos resolvendo em ≤2 tentativas, o
mesmo espírito de "verdade-base conhecida" deste item, só que com
humanos no lugar do World Generator. Mais relevante ainda para a coluna
"Environment/Agent Interface" da tabela acima: **ARC-AGI-3** (2026,
[github.com/arodmor/arc-agi-3](https://github.com/arodmor/arc-agi-3),
confirmado via arcprize.org — "primeiro benchmark totalmente
interativo" da série) avalia agentes em quatro eixos — **Exploration,
Modeling, Goal-setting, Planning & Execution** — sem instruções, sem
regras declaradas, exigindo que o agente descubra sozinho como o
ambiente funciona; isto é quase literalmente o "Capability Vector" desta
seção já validado como benchmark real e competitivo (humanos resolvem
100%, LLMs de fronteira ficam abaixo de 1% quando usados diretamente,
segundo o próprio arcprize.org). Dois usos concretos: (1) os quatro
eixos do ARC-AGI-3 são um ponto de partida testado publicamente para
nomear as dimensões do Capability Vector, em vez de inventar uma
taxonomia do zero; (2) avaliar formalmente candidatos-agentes deste
projeto (PINN/RL/LLM) contra o ARC-AGI-3 real, antes ou em paralelo à
bateria física própria, dá um ponto de comparação externo e público —
"este agente físico-específico generaliza tão pouco quanto um LLM de
fronteira generaliza em ARC-AGI-3, ou melhor?" é uma pergunta que este
roadmap pode responder com números reais, não afirmação.

**Por que isto é maior que "mais um item da bateria"**: dar a isso uma
identidade própria (candidato a nome: `PINNeAPPle-AGI-Lab`, seguindo a
convenção `PINNeAPPle-Research`/`PINNeAPPle-arena` de repo-satélite thin
wrapper) permite medir explicitamente a diferença entre
`memorization ≠ pattern matching ≠ generalization ≠ world-model
discovery` — hoje nenhum repo do org faz essa distinção de forma
explícita; `PINNeAPPle-arena` mede acurácia/tempo/memória por
arquitetura, não "esse modelo descobriu a lei ou só decorou o regime
treinado". **Pré-requisito antes de qualquer código**: os gaps de
geometria/manifold, causalidade estrutural e transição ordem↔caos
listados acima nesta mesma seção §8 precisam existir primeiro — este
item é o produto que consome esses três pilares, não um substituto para
construí-los.

---

## 9. Referências externas trazidas por Yan em 2026-09-13 — LBM em GPU, geometria pública de aeronave elétrica, depth estimation, geotecnia

Quatro referências novas (a quinta, The Well, já existia em §4 e foi
expandida lá em vez de duplicada aqui). Mesma disciplina de §7: cada
uma só entra se tiver uma conexão explícita com um módulo real do
PINNeAPPle ou de um repo satélite — nunca "porque é uma tecnologia
interessante".

### FluidX3D — LBM em GPU/OpenCL como backend de CFD de altíssima performance
**Projeto novo.** Solver de Lattice Boltzmann Method (LBM) em C++17,
OpenCL cross-vendor (NVIDIA/AMD/Intel/Apple/ARM, não só CUDA), com
D2Q9/D3Q15/D3Q19/D3Q27, 55 bytes/célula (vs. ~344 bytes de abordagens
tradicionais — ~19 milhões de células por GB de VRAM), multi-GPU,
free-surface LBM (volume-of-fluid), simulação térmica, tracking de
partículas via immersed-boundary, e voxelização de malha STL acelerada
por GPU. Fonte: [ProjectPhysX/FluidX3D](https://github.com/ProjectPhysX/FluidX3D)
(verificado nesta sessão). Licença: **gratuita apenas para uso
não-comercial** — qualquer uso deste solver para gerar dados de treino
ou rodar inferência dentro de um produto pago (`PINNeAPPle-apps`,
`veriphysics`) precisa de revisão de licença própria antes, não pode
ser assumido compatível.

Conexão direta: `pinneapple_simulation.numerical_solvers` já implementa
LBM em Python/Numba puro — FluidX3D é uma implementação de referência
madura, multiplataforma, que resolve o mesmo problema em escala muito
maior. Módulo relacionado mais natural:
`pinneapple_simulation.external_solvers` (já faz bridge para
OpenFOAM/MATLAB/FMU/FEniCS via subprocesso/arquivo) — um
`fluidx3d_bridge` seguiria o mesmo padrão: escreve `.stl` de entrada,
invoca o binário FluidX3D compilado localmente, lê `.vtk`/`.png` de
saída e converte para os tensores de campo internos do PINNeAPPle.
Conecta também diretamente com o item já existente "Autonomous DOE-CFD"
(§4): o blueprint nTop/CoreWeave citado ali usa exatamente esta mesma
ideia (LBM sobre grid cartesiano fixo, sem malha, para rodar milhares de
geometrias sem falha topológica) — FluidX3D é o substrato open-source
concreto para validar essa tese antes de construir algo próprio.
Conecta ainda com `PINNeAPPle-arena` (§1): entraria no catálogo externo
como `manual_install` (não é `pip install`-ável — C++/OpenCL, precisa
compilar), mesmo contrato de honestidade já usado para outros repositórios
só-git do catálogo. E com `PINNeAPPle-SplashCFD`: hoje depende
inteiramente de dados OpenFOAM (`.splash`); FluidX3D roda um canal
turbulento em minutos numa única GPU em vez de horas de solver CFD
tradicional, um candidato real e muito mais barato para o item 4 do
próprio `ROADMAP.md` desse repo ("um catálogo de modelos populado com
múltiplos datasets").

### NASA X-57 Maxwell — geometria pública real de aeronave elétrica
**Projeto novo.** Recursos 3D públicos do X-57 Maxwell, avião
experimental 100% elétrico da NASA (missão Electrified Aircraft
Propulsion): `X-57.glb` (visualização/glTF) e `X-57.vsp3` (formato do
OpenVSP, ferramenta de projeto conceitual de aeronaves). Fonte:
[science.nasa.gov/3d-resources/x-57-maxwell](https://science.nasa.gov/3d-resources/x-57-maxwell/)
(verificado nesta sessão).

Conexão direta: `pinneapple_design.geometry` já tem SDF/CSG/mesh/NACA
airfoil mas nenhum importador de `.vsp3` (OpenVSP tem API Python própria
— `openvsp`/`degen_geom` — capaz de exportar STL/IGES a partir de um
`.vsp3`). Um `pinneapple_design.geometry.io.openvsp_bridge` traria uma
geometria pública, real, bem documentada (NASA publicou dados reais de
voo e túnel de vento do X-57) para o pipeline de geometria — exatamente
o tipo de ground truth externo que a disciplina anti-fabricação deste
ecossistema exige antes de publicar qualquer número de validação.
Estende diretamente o flagship demo já existente em §6, "Airfoil /
Engineering Design Optimization" (`missile_aero`, Cp R² 0.99 +
`01_automotive_aero`) — de um único aerofólio/míssil para uma aeronave
completa com propulsão elétrica distribuída (múltiplas nacelles ao
longo da asa), um caso multi-físico (aero + elétrico + térmico) que os
flagships atuais não cobrem. Conecta também com `PINNeAPPle-apps`: um
12º produto candidato, "ElectricAeroScreen", seguindo a mesma receita já
validada de `NozzleScreen`/`RoverMobility` (preset PINN +
cross-check analítico independente), usando o X-57 como caso de
validação público em vez de dados proprietários — mas só depois que o
importador `.vsp3` e um preset de referência existirem; não é um
produto que se constrói antes da geometria estar disponível no
pipeline.

**Pesquisa verificada em 2026-09-13 — encontrado um atalho real para o
`openvsp_bridge` acima, mais barato que escrever um bridge do zero.**
[OpenVSP MCP Server](https://github.com/Three-Little-Birds/openvsp-mcp)
(MIT, verificado via fetch direto do repositório) já expõe OpenVSP como
três ferramentas MCP: `openvsp.inspect` (lê IDs de componente e metadados
de geometria), `openvsp.modify` (edições paramétricas via `.vspscript`
reproduzível) e `openvsp.run_vspaero` (roda VSPAero e retorna
coeficientes aerodinâmicos), com export STL/OBJ e suporte a
STDIO/HTTP/Docker. Isto muda o caminho recomendado: em vez de construir
`pinneapple_design.geometry.io.openvsp_bridge` como um bridge de
subprocesso do zero, o caminho mais barato é rodar este servidor MCP
localmente e deixar um agente (Claude Code incluso) chamá-lo diretamente
para inspecionar/exportar a geometria do X-57 — só valeria migrar para
um bridge Python nativo dentro do PINNeAPPle depois de confirmar, via
este MCP, exatamente quais campos/exports o `.vsp3` do X-57 realmente
produz. Referência adicional para a mesma tarefa, caso o MCP não cubra
algo necessário: [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox)
(discussão pública sobre exportar aeronaves como STEP/STL,
verificado via GitHub) é uma biblioteca Python pura para projeto
conceitual de aeronaves, alternativa ao OpenVSP sem dependência de um
binário C++ externo.

### Marigold V2 — depth estimation monocular (encaixe direto num gap já mapeado)
**Projeto novo, mas não é uma ideia nova — fecha um item que já estava
no roadmap do `reality2physics`.** Marigold V2
(`huawei-bayerlab/marigold-v2`) é um Diffusion Transformer quantizado
(base Qwen-Image-Edit-2509) com adaptadores LoRA, reaproveitado como
"single-step dense predictor": estima profundidade, normais de
superfície e albedo a partir de uma única imagem RGB (até 2048²),
saída em `.npy` + PNG de visualização. Licença Apache 2.0, pesos no
Hugging Face (`huawei-bayerlab/marigold-v2-0`). Fonte:
[huawei-bayerlab/marigold-v2](https://github.com/huawei-bayerlab/marigold-v2)
(verificado nesta sessão).

Conexão direta e já documentada: o próprio README de `reality2physics`,
seção Roadmap, já lista "Adicionar modalidades: profundidade, térmico
real (câmera FLIR), áudio" e "Multi-task real (depth, segmentação,
reconstrução de superfície) via backbones tipo Depth Anything / SAM como
professores (distillation)" como itens em aberto. Marigold V2 é um
candidato concreto, aberto (Apache 2.0) e mais rico que um Depth-Anything
puro para esse papel de "professor": por ser baseado em difusão, permite
estimar incerteza via múltiplas amostras, e já produz normais de
superfície junto com profundidade — o que dá um termo de consistência
físico adicional (normal-de-superfície) para o encoder-decoder do
`reality2physics`, além do campo de profundidade em si. Módulo
relacionado no monorepo: `pinneapple_perception` (já extrai observações
físicas — campos de velocidade, geometria de contorno, frequências
modais — de imagem/vídeo/áudio, mas nada de profundidade hoje) — um novo
submódulo `pinneapple_perception.depth` seguiria o mesmo padrão "a
capacidade genérica vive no PINNeAPPle, a fiação específica do problema
vive no repo satélite" já usado por `physcurator`/`pinneapple_splash`.
Uso concreto mais barato: adicionar uma flag `--depth_teacher
marigold-v2` a `reality2physics/scripts/extract_frames_from_video.py`,
gravando um canal de profundidade junto ao optical flow (Farnebäck) já
usado como pseudo-rótulo no fine-tuning em vídeo real (cenário
`navier_stokes`).

### Geotecnia / Soil-Structure PINN — domínio físico não coberto (sinal de literatura, não uma implementação)
**Peça em aberto, tratada com o mesmo cuidado anti-fabricação do resto
deste roadmap.** Artigo em
`sciencedirect.com/science/article/pii/S0266352X26007251` — **não foi
possível ler o conteúdo real** (paywall, HTTP 403; o DOI ainda não está
indexado no Crossref nesta sessão, ou seja, é um artigo muito recente ou
"in press" de 2026). O que É verificado, e é a base real desta entrada:
o prefixo do PII (`S0266352X`) corresponde ao ISSN 0266-352X, da revista
*Computers and Geotechnics* (Elsevier), cujo "Guide for authors" hoje
convida explicitamente "innovative applications of physics-informed
AI/ML techniques" para problemas de engenharia geotécnica, e a mesma
revista já publicou, no mesmo volume 2026, pelo menos um artigo
próximo confirmado por busca (`S0266352X26000327`, "Utilizing
physics-informed neural network and geotechnical distance field for
solving three-dimensional nonlinear consolidation"). Tratar isto como
**sinal de uma linha de pesquisa ativa** (PINNs para consolidação de
solo, problemas inversos geotécnicos, encoding de geometria via distance
field para domínios solo-estrutura irregulares) — não como uma
afirmação sobre o que o artigo específico contém, o que seria fabricar.

**Pesquisa verificada em 2026-09-13 — o artigo original continua
ilegível, mas a linha de pesquisa em si agora tem lastro real, lido de
verdade, não apenas o sinal indireto de ISSN acima.** Três referências
confirmadas por busca e leitura de abstract real:
- "Physics-informed Deep Learning to Solve Three-dimensional Terzaghi
  Consolidation Equation: Forward and Inverse Problems"
  ([arXiv:2401.05439](https://arxiv.org/abs/2401.05439)) — PINN 3D para
  consolidação de Terzaghi, forward e inverso, >99% de acurácia contra
  método numérico tradicional.
- "Physics-informed neural networks for back-analysis and consolidation
  settlement prediction using field measurements" (*Acta Geotechnica*,
  2025/2026,
  [doi.org/10.1007/s11440-025-02888-1](https://link.springer.com/article/10.1007/s11440-025-02888-1))
  — PINN para prever recalque de consolidação sob carregamento em
  etapas, integrando dados reais de campo (não só sintéticos) — o tipo
  de validação contra medição real que a disciplina anti-fabricação
  deste ecossistema já exige em outros domínios.
- "A Critical Assessment of PINNs and Operator Learning for Geotechnical
  Engineering" ([arXiv:2512.24365](https://arxiv.org/html/2512.24365))
  — um review crítico (não promocional) especificamente sobre onde
  PINN/operator learning funcionam e onde falham em geotecnia; ponto de
  partida melhor do que qualquer paper individual para decidir SE vale
  abrir este domínio novo antes de comprometer arquitetura.

Isso não substitui o pré-requisito já declarado abaixo (ler
S0266352X26007251 de verdade antes de código) — mas remove a dependência
de um único artigo paywalled como única evidência da linha de pesquisa:
mesmo sem aquele artigo específico, "Geotechnical / Soil-Structure PINN"
já é um domínio ativo e publicamente verificável, com pelo menos um
review crítico disponível para avaliar viabilidade antes de investir.

Conexão: nenhum repo do PINNeAPPle-Labs cobre hoje geomecânica/solo —
o item mais próximo em §6, "Drilling Hydraulics Digital Twin", é
hidráulica de fluido (lama de perfuração), não mecânica de solo/rocha.
Candidato a novo domínio flagship: "Geotechnical / Soil-Structure PINN"
— consolidação não-linear (Terzaghi/Biot), estimação inversa de
parâmetros (permeabilidade, compressibilidade) a partir de dados de
campo/laboratório, e encoding de geometria irregular solo-estrutura via
distance field — este último item reaproveitaria diretamente a
biblioteca SDF que `pinneapple_design.geometry` já tem para outros
domínios, em vez de reimplementar. **Pré-requisito explícito antes de
qualquer código**: ler de fato o artigo (acesso institucional ou
preprint) para confirmar a contribuição real, em vez de comprometer
arquitetura com base só no sinal de journal/ISSN.

---

## Como este roadmap se conecta ao resto do ecossistema

- §1 (Scientific Intelligence Platform) é onde a maior parte do valor já
  construído (`reality2physics`, `PINNeAPPle-arena`, `ge3`, `portfolio`)
  já vive — a recomendação estratégica do CoupleTasks é amarrar isso via
  GenAItor antes de abrir os domínios novos de §6.
- §3 e §4 (infraestrutura/CFD em escala) alimentam diretamente a
  "PINNeAPPle Surrogate Factory" (§1) com peças reais (Genesis, nTop/
  CoreWeave, Noether) em vez de reinventar cada uma do zero.
- §7 é o bucket de referências externas trazidas fora do catálogo
  CoupleTasks nesta sessão — cada uma tem uma conexão explícita com um
  módulo real do PINNeAPPle ou com `veriphysics`/`pinneapple-apps`,
  nunca uma integração "porque é legal".
- §8 (Structure Discovery / Chaos-to-Law) generaliza o item "Physics
  Discovery / Equation Discovery" de §1: a metade "dinâmica" já existe de
  verdade (`rom/` — SINDy, Koopman, DMD, HAVOK, POD, OpInf), a metade
  "geometria/causalidade/regime/evento extremo" é gap confirmado no
  código, não suposição. A bateria de validação proposta em §8 é o
  candidato mais natural para o próximo ciclo de expansão do
  `PINNeAPPle-arena` (§1), e `reality2physics` (§1, §7) é a superfície de
  aplicação onde a "camada de descoberta de PDE" do seu próprio roadmap
  (`README.md`, seção Roadmap) deveria consumir este pilar em vez de
  reimplementá-lo.
- §9 é o bucket de referências externas trazidas em 2026-09-13 — mesma
  disciplina de §7 (conexão explícita com módulo real, nunca "porque é
  legal"), com um cuidado extra na última entrada (geotecnia): quando a
  fonte primária não pôde ser lida de verdade (paywall), o item registra
  isso explicitamente e trata o achado como sinal de literatura, não como
  fato sobre o conteúdo do artigo. A entrada de The Well foi expandida
  dentro de §4 (onde já existia) em vez de duplicada aqui.
- Além das cinco referências de §9, a mesma sessão de 2026-09-13 trouxe
  dois PROJETOS (não apenas links de terceiros) que também foram
  mapeados no lugar certo em vez de virarem uma seção à parte: o
  **AGI Evaluation Engine**, adicionado ao final de §8 (generaliza a
  "Bateria de validação" que já vivia lá em produto próprio), e o
  **harness de CAD com modelos open-source estilo Astra**, adicionado ao
  final de §2 (contrastado explicitamente com o `cad_draft.py` real que
  já existe, não tratado como extensão direta dele — os dois resolvem
  problemas diferentes com trade-offs de risco/expressividade opostos).

---

## 10. Estado da arte / para onde o campo está indo (Physical AI, Physics AI, Scientific ML) — mapeado em 2026-09-13

Diferente das seções anteriores, isto não é um catálogo de "projeto para
construir" — é um mapa de posicionamento: onde a indústria e a pesquisa
estão indo agora, verificado por pesquisa real (fetch direto quando
indicado; busca com trecho real citado quando não), e onde cada
tendência toca (ou não) algo real já existente no PINNeAPPle-Labs. Sem
isto, o roadmap corre o risco de otimizar peças que o campo já
resolveu em outro lugar, ou de não perceber uma peça que ficou
estrategicamente mais importante do que parecia há um ano.

### "Physical AI" (o termo, cunhado pela NVIDIA) — world models + robótica + simulação, não é sinônimo de "Physics AI"
**Verificado via fetch direto** da NVIDIA Newsroom: em 2026 a NVIDIA
declarou "Physical AI has arrived — every industrial company will become
a robotics company" (Jensen Huang), lançando **Cosmos Predict 2.5** e
**Cosmos Transfer 2.5** (world models abertos para geração de dados
sintéticos fisicamente plausíveis e avaliação de política de robô em
simulação), **Cosmos Reason 2** (VLA — vision-language-action — aberto),
**Isaac Lab-Arena** (benchmarking de política de robô em escala,
conectado a benchmarks de indústria como Libero/Robocasa) e **OSMO**
(orquestração cloud-native para geração de dados sintéticos/treino/
software-in-the-loop). Tudo integrado ao Hugging Face e ao framework
aberto LeRobot.

**Por que isto importa para o PINNeAPPle-Labs, especificamente**: "Physical
AI" (robótica + world models + ação) e "Physics AI" (PINNs/neural
operators/scientific ML, o que este org faz) são campos vizinhos, não o
mesmo campo — mas já se tocam em pontos reais do próprio roadmap: o
`pinneapple_worldmodel` (Physics Foundation Model) e o item "Cosmos
Synthetic Perception Pretrainer" (§5) já usam a família Cosmos
(`nvidia/Cosmos-Reason2-2B`, já em produção real no `physcurator`); o
"Multi-Physics Sandbox" (§3, Genesis) e o "LatticePT" (§7) já apontavam
para a mesma direção de simuladores GPU-nativos multiplataforma que a
NVIDIA está consolidando com Isaac Sim 6.0/Isaac Lab 3.0. Não é
recomendação para o org virar uma empresa de robótica — é sinal de que a
infraestrutura de simulação/world-model que a indústria está investindo
bilhões para construir (Isaac, Cosmos, Omniverse) é reaproveitável como
backend por baixo do que este org já faz (superfície física/científica,
não humanoides).

### Physics AI como categoria de mercado com capital real — mais de US$1B levantado por poucas empresas
**Verificado via fetch direto** da PhysicsX Newsroom e via busca (dados
adicionais não fetchados diretamente, tratados como achado de busca, não
verificação de primeira mão): **PhysicsX** levantou US$300M em Series C
em junho de 2026 (avaliação ~US$2,4B, liderada por Temasek, com NVIDIA e
Siemens entre os investidores) — o CEO descreve a tese como "model
architectures and GPU economics now mature enough to support physics AI
at production scale" e fala em construir "Large Physics Models" (modelos
físicos pré-treinados maiores, análogo direto ao `pinneapple_worldmodel`
deste org, em escala comercial). Concorrentes diretos, todos levantando
capital relevante no mesmo período (achado via busca): **Neural Concept**
(US$100M Series C, 2025), **Rescale** (US$115M Series D, 2025), e
**Luminary Cloud** (US$72M Series B, 2025 — já é a entrada "referência
competitiva" existente em §7 deste roadmap; esta pesquisa confirma que
não é um caso isolado, é uma categoria de mercado inteira em movimento).
Somados a `nTop`, `Monolith AI`, `BeyondMath` e `DIVE Solutions`, a
categoria "engineering simulation + physics AI" levantou quase US$1B
combinado.

**Conexão direta**: isto valida e generaliza a nota já existente em §7
sobre Luminary Cloud ("uso recomendado: benchmark de posicionamento para
`pinneapple-apps`... e para o `PINNeAPPle-arena`") — o mesmo tratamento
deveria se aplicar a PhysicsX/Neural Concept/Rescale: claims de
performance de terceiros (ex.: "segundos em vez de semanas") são claims
a verificar de forma independente, nunca absorvidas como fato, no mesmo
espírito anti-fabricação de `tool_recommendation.py`. Para
`VerifiedPhysics` e `PhysicsCopilot` especificamente, isto confirma que o
mercado já reconhece o mesmo diferencial que a tese destes produtos
aposta ("Physics AI + LLM sozinho não é moat, execução mecanicamente
verificada é") — PhysicsX, com capital 1000x maior, ainda não resolveu a
parte de verificação/trust do jeito que `veriphysics` já implementa
(Decision Record, trust score com coverage explícito); isto é uma janela
de diferenciação real, não apenas otimismo.

### Neural operators e scientific ML — a pesquisa está migrando de arquitetura única para "self-driving research loops"
**Achado via busca, trechos reais citados, não fetch completo do paper
primário**: surveys recentes (ex. "Physics-Informed Neural Networks and
Neural Operators for Parametric PDEs", submetido ao ICAIS 2025)
descrevem a área migrando de tuning manual de hiperparâmetro para loops
de pesquisa que iteram sozinhos sobre arquitetura e restrições físicas.
Avanços pontuais recentes incluem Spectral-boosted FNO (redução de 40%
no viés de frequência em modelagem sísmica 3D) e Sensitivity-Constrained
FNO (SC-FNO, ICLR 2025, integra análise de sensibilidade na inversão de
parâmetro). Conecta diretamente com `pinneapple_neural.architectures`
(FNO já implementado) e com o item "PDE Foundation Models" já adicionado
em §4 desta sessão (PDE-FM) — a direção "self-driving research loop" é
também, em espírito, o mesmo padrão do "Agentic PDE Debugger" (§3) e do
"PINNeAPPle Auto-PINN" (§1) já catalogados aqui, agora confirmados como
tendência de campo, não ideia isolada deste org.

### Differentiable simulation — NVIDIA Warp consolidando como padrão de interoperabilidade
**Achado via busca**: NVIDIA Warp (Python, diferenciável, interopera
nativamente com PyTorch, JAX, PhysicsNeMo e Omniverse) está se
consolidando como uma camada comum entre simulação física e treino de
ML — junto com o ecossistema JAX mais amplo (JAX-MD, já catalogado no
`PINNeAPPle-arena`'s external catalog) e um benchmark dedicado real,
"Mosaic: A Benchmark Suite for Differentiable Physics Solvers"
(arXiv:2606.27895). Conexão direta: `pinneapple_simulation.particle_dynamics`
hoje é PyTorch puro para MPM/SPH/rigid-body (o próprio `ROADMAP.md`
principal já nota isto ao discutir Genesis, §3) — Warp é um segundo
candidato real e mais leve que Genesis para o mesmo papel de backend
diferenciável GPU-nativo, e já conecta nativamente com o backend JAX que
`pinneapple_tools.compute_backends` já expõe, sem precisar de uma ponte
nova.

### AI for Science — descoberta autônoma já produz resultado real e verificável, não só promessa
**Achado via busca com trechos reais**: o GNoME da DeepMind já descobriu
2,2 milhões de estruturas cristalinas novas (52 mil condutores de
lítio-íon), com 736 dessas predições já sintetizadas por pesquisadores
externos — um resultado real, não um benchmark interno. O modelo de
clima da DeepMind previu a Furacão Melissa como categoria 5 com dias de
antecedência e estendeu a janela de aviso em ~3 dias sem perder acurácia
— ganho comparável a uma década de progresso meteorológico tradicional,
segundo a cobertura. Do lado de agentes autônomos de pesquisa: o "AI
Scientist" da Sakana AI (v2, busca em árvore agêntica) já produz papers
de ponta a ponta (ideia → experimento → escrita → revisão por pares),
existe desde 2026 uma revista dedicada a papers gerados por IA (JAIGP) e
uma conferência dedicada a avaliar contribuição científica de IA
(Agents4Science) — mas avaliações independentes (arXiv:2502.14297)
já documentaram falhas reais e específicas (revisão de literatura por
busca de palavra-chave simplista, avaliação de novidade pobre), não
apenas hype.

**Conexão direta e dupla**: (1) isto é validação externa forte do item
"Autonomous Scientific Experimentation" (§1, hoje "sobreposição
parcial") e do "Paper-to-Repro Benchmark Suite" (§3) já catalogados
aqui — a categoria inteira "agente autônomo de pesquisa científica" está
madura o suficiente para ter revista e conferência próprias em 2026, não
é mais especulativo; (2) as falhas documentadas do AI Scientist (revisão
de literatura rasa, avaliação de novidade fraca) são exatamente o tipo
de falha que a disciplina anti-fabricação deste ecossistema (scoring
determinístico contra ground truth conhecida, nunca LLM-julgando-LLM) foi
desenhada para evitar — o "AGI Evaluation Engine" (§8) e o
`PINNeAPPle-Research` (scoring contra problema já resolvido) já são,
por construção, mais rigorosos neste eixo específico do que o estado da
arte público em agentes de pesquisa autônomos, o que vale documentar
como diferencial real, não apenas coincidência de design.

### A economia de surrogates de IA para simulação física — um framework de decisão, não só uma tecnologia
**Trazido por Yan em 2026-09-13**, a partir de um post técnico externo
("The Economics of AI Surrogate Models for Physics Simulation", Javier
Jiménez, jun/2026) cujas duas referências acadêmicas centrais foram
verificadas nesta sessão (busca real, resumo confirmado, não só
título): "Breakeven complexity: A new perspective on neural partial
differential equation solvers" ([arXiv:2605.15399](https://arxiv.org/abs/2605.15399),
Zhang/Roberts/Marwah/Khodak, 2026) e "Fluid Intelligence: A Forward
Look on AI Foundation Models in Computational Fluid Dynamics"
([arXiv:2511.20455](https://arxiv.org/abs/2511.20455), Ashton/NVIDIA,
Brandstetter, Mishra, 2025).

O argumento central, resumido: um surrogate de IA (PINN/FNO/DeepONet/
GNN) só compensa financeiramente depois de aproximadamente **N**
consultas, onde **N é aproximadamente o próprio tamanho do dataset de
treino** (mais uma sobrecarga pequena de GPU medida em
"simulações-equivalentes") — abaixo disso, rodar o solver de verdade é
mais barato. O verdadeiro vilão econômico não é o custo de inferência
nem o de treino: é a **acurácia-alvo**, porque o tamanho de dataset
necessário escala como uma poder-lei íngreme (expoente tipicamente
entre 2 e 5) — reduzir o erro-alvo pela metade pode multiplicar o
dataset necessário por 4–32×. Fine-tuning a partir de um modelo
pré-treinado desloca o break-even de milhares de consultas para dezenas/
centenas — o mecanismo econômico exato por trás de qualquer estratégia
de "Physics Foundation Model" (ver PDE-FM, §4 acima). O paper
"Fluid Intelligence" (Ashton et al.) generaliza exatamente esse ponto
para CFD industrial: propõe a primeira lei de escala que incorpora
geração de dados E treino simultaneamente, distinguindo os regimes onde
cada um domina o custo total, e conclui que dados transientes de alta
fidelidade são a rota ótima para um foundation model de CFD — a mesma
conclusão prática, com lastro acadêmico independente do post original.

**Por que isto muda como o resto deste roadmap deveria ser lido, não só
mais um dado**: vários itens já catalogados aqui pressupõem implicitamente
que "surrogate mais rápido" é sempre desejável — este framework mostra
que a pergunta certa não é "o surrogate é mais rápido?" (quase sempre é),
mas "eu vou consultar este surrogate vezes suficientes para pagar o
dataset que o treinou?". Conexões diretas e concretas:
- **`PINNeAPPle-SplashCFD`, item 1 do seu próprio `ROADMAP.md`** (a
  trilha paramétrica/multi-caso ainda não construída) — antes de
  justificar o esforço de generalizar o FNO3d para uma família de
  parâmetros, este framework dá a pergunta certa a responder primeiro:
  quantas consultas reais este surrogate paramétrico vai receber ao
  longo da vida útil, comparado ao número de casos OpenFOAM necessários
  para treiná-lo bem?
- **`veriphysics`, `pinneapple_analysis.verification.solver_orchestration`**
  (decide PINN interno vs. solver clássico vs. ferramenta externa) —
  hoje a decisão é por adequação física/documentada, não por custo; um
  termo de break-even explícito (dataset já existente / consultas
  esperadas) seria uma dimensão real e nova de recomendação, não
  substituindo a lógica atual, complementando.
- **"PINNeAPPle Surrogate Factory" (§1)** — o pipeline CAD→DOE→solver→
  dataset→surrogate proposto lá deveria nascer com este cálculo de
  break-even como gate de decisão explícito (vale a pena rodar o DOE
  completo?), não como reflexão posterior.
- **`PINNeAPPle-apps` (`VerifiedPhysics`/`PhysicsCopilot`)** — o "Business
  case" que ambos os produtos já documentam poderia literalmente expor
  este cálculo como uma feature de venda real: um calculador de
  break-even embutido no produto ("baseado no seu volume esperado de
  consultas, um surrogate paga-se em X execuções") em vez de apenas
  alegar velocidade.
- **§10 acima (PDE-FM)** — este framework é o argumento econômico formal
  por trás da recomendação já feita de reaproveitar PDE-FM/Warp em vez
  de treinar do zero: fine-tuning desloca o break-even em 1–2 ordens de
  magnitude, o que é precisamente o que torna qualquer investimento em
  foundation model físico (deste org ou de terceiros) economicamente
  justificável mesmo para equipes sem campanhas massivas de design.
