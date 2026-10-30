# Volume 8 — Apps, exemplos, reproduções, scripts e CI

> Parte do [Mapa completo do PINNeAPPle](../mapa_completo.md). Gerado por `docs/pt/_mapa/gerar.py` a partir do inventário do commit `e53cdcfb04`.

O que cada app, exemplo, reprodução de paper, script, benchmark e workflow de CI faz. A coluna *testes* conta funções de teste em arquivos que citam o caminho.

## 1. Apps (`apps/`)

64 itens.

| item | o que faz | arquivos .py | testes |
|---|---|---:|---:|
| `apps/_shared` | Não é um app: é a cola (`appkit.py`) que todo app em `apps/` importa, com rotas HTTP comuns, login, fila de jobs, chaves de API, medição de uso e a vista 3-D de campos (`/shared/fieldview3d.js`). A lógica fica na biblioteca (`pinneapple_core.service`, `pinneapple_data.contracts`, `pinneapple_security`) e tem testes (por exemplo `tests/test_appkit_runtime.py`). Entra no `PYTHONPATH` de cada Dockerfile. | 1 | 23 |
| `apps/aero_optimizer` | Otimizador de projeto de avião de corredor único (classe A320/737): o NSGA-II busca 13 números (seção da asa em CST, área, alongamento, afilamento, enflechamento, torção, posição e Mach de cruzeiro) para menos CO₂ por passageiro-km, com checagem em OpenFOAM 3D e campos de escoamento. Usa sobretudo `pinneapple_design` (e `pinneapple_tools`). Roda com `cd apps/aero_optimizer && uvicorn aero_optimizer.api:app --port 8092`; nenhum teste cita a pasta. | 9 | 0 |
| `apps/anomaly_detector` | Detector de anomalias para engenheiros de planta: combina estatística robusta, modelo de séries temporais e balanços de massa e energia para dizer qual balanço quebrou, quanto e em quais sensores. A lógica é `pinneapple_systems.operations.anomaly` (com `pinneapple_data` e `pinneapple_core`). Roda com `uvicorn anomaly_detector.api:app --port 8111` (comando do Dockerfile) ou `docker run -p 8111:8111 anomaly-detector`. | 2 | 4 |
| `apps/benchmark_arena` | Arena de benchmarks de IA física: compara, sob um só protocolo, ML clássico, MLP, FNO, DeepONet, PINN e solvers numéricos em problemas públicos (campanha OpenFOAM de aerofólios do zoo, Burgers viscoso) e na tabela de runs do próprio usuário. Usa `pinneapple_arena.regimes.run_regime_arena`, `pinneapple_analysis` e `pinneapple_core`. Roda com `uvicorn benchmark_arena.api:app --port 8105` (Dockerfile); os resultados públicos se refazem com `python run_public_benchmarks.py --budget 10 --seeds 3`. | 3 | 5 |
| `apps/brakecool_ai` | Projeto inverso das aletas radiais de um disco de freio ventilado (número de aletas, folga e espessura) por um surrogate neural congelado, sempre reconferido contra a física, mais um Monte Carlo das tolerâncias de usinagem. A lógica está em `pinneapple_design.surrogate_inverse.brake_rotor`; a pasta só tem rotas e a página. Roda com `cd apps/brakecool_ai && PYTHONPATH=../..:../_shared uvicorn brakecool_ai.api:app --port 8161`. | 2 | 5 |
| `apps/calibration_lab` | Laboratório de calibração: ajusta parâmetros de um modelo a medições reais, dá intervalos só onde os dados os determinam e confere o ajuste em medições que ele não viu. Usa `pinneapple_analysis`. Roda com `cd apps/calibration_lab && uvicorn calibration_lab.api:app --port 8101` ou pela imagem Docker. | 2 | 4 |
| `apps/control_lab` | Laboratório de controle baseado em física: compara MPC linear (sobre os balanços de massa e energia) e outras estratégias com o que roda hoje (PID, liga/desliga). Funciona só em modo de simulação ou recomendação; não escreve em nenhum controlador real. A lógica é `pinneapple_systems.operations.control`; roda com `uvicorn control_lab.api:app --port 8112` (Dockerfile). | 2 | 4 |
| `apps/data_health` | Recebe um conjunto de dados (CSV, Excel, Parquet, JSON, HDF5), reconhece a grandeza física e a unidade de cada coluna e devolve um relatório de qualidade com nota, evidências e correções. Usa `pinneapple_data` e `pinneapple_core`. Roda com `cd apps/data_health && uvicorn data_health.api:app --port 8082`; nenhum teste cita a pasta. | 2 | 0 |
| `apps/data_standardizer` | Junta até 8 arquivos de fontes diferentes (unidades, fusos e convenções distintas) num só esquema físico em UTC e unidades coerentes, com uma receita JSON reaproveitável e checagem de concordância entre fontes. Usa `pinneapple_data`. Roda com `cd apps/data_standardizer && uvicorn data_standardizer.api:app --port 8083`; nenhum teste cita a pasta. | 2 | 0 |
| `apps/deploy` | Não é um app: são os arquivos para publicar vários apps numa só VM Linux, cada um no seu subdomínio, atrás de um Caddy com certificados Let's Encrypt (`docker-compose.yml`, variantes de pouca memória e de proxy existente, `Caddyfile`, `fail2ban`, `check_server.sh`). Sobe com `docker compose up -d`; a senha de login vem de `python -m pinneapple_security hash-password`. | 0 | 9 |
| `apps/design_optimization` | Caso fixo de demonstração: uma seção de asa da família NACA 2418 redesenhada ao vivo para menos arrasto com a mesma sustentação, com passos de Adam através de um surrogate MLP e cada iterado reavaliado pela fórmula fechada. Usa `pinneapple_design.design_optimizer`. Roda com `PYTHONPATH=../_shared uvicorn design_optimization.api:app --port 8181`. | 2 | 2 |
| `apps/design_requirements` | Form Compiler: lê folhas de dados e especificações (PDF, texto, páginas escaneadas via OCR) e compila um formulário ou folha de dados, com a origem de cada valor e a lista do que falta ou se contradiz. Usa `pinneapple_data`. Roda com `cd apps/design_requirements && UDR_FORM_PDF=/caminho/u-dr-1.pdf uvicorn design_requirements.api:app --port 8085`; nenhum teste cita a pasta. | 3 | 0 |
| `apps/engineering_optimizer` | Otimização caixa-preta de projetos dentro de limites físicos e de fabricação, sobre um modelo rápido, com o ótimo confirmado por um modelo de alta fidelidade. Usa `pinneapple_design` e `pinneapple_data`. Roda com `cd apps/engineering_optimizer && PYTHONPATH=../_shared uvicorn engineering_optimizer.api:app --port 8100`. | 2 | 3 |
| `apps/evidence_report` | Transforma saídas de experimentos e simulações (relatórios do Validation Lab, ledgers de evidência, `pinneapple check --json`, tabelas de métricas) num relatório técnico reprodutível; um teste só conta como aprovado se de fato rodou. Usa `pinneapple_veriphysics`. Roda com `cd apps/evidence_report && uvicorn evidence_report.api:app --port 8113`. | 2 | 5 |
| `apps/flow_studio` | Canvas de nós no estilo n8n para fluxos de engenharia: descreve-se o problema, recebe-se um ou mais fluxos, editam-se equações e métodos, roda-se e inspeciona-se cada nó (métricas, figuras, vista 3-D). Toda a lógica é `pinneapple_flow`; a pasta tem só o shell HTTP e o frontend React. Roda com `python -m pinneapple_flow studio` (porta 8765); os mesmos fluxos rodam sem GUI com `python -m pinneapple_flow run flow.json`. | 1 | 19 |
| `apps/flowtrace` | Recebe um vídeo ou uma sequência de imagens de um escoamento com partículas e devolve o campo de velocidade por PIV de correlação cruzada. Usa `pinneapple_perception.video_piv` e `pinneapple_perception.piv_tools`. Roda com `cd apps/flowtrace && uvicorn flowtrace.api:app --port 8145` (vídeos pedem `ffmpeg`). | 2 | 3 |
| `apps/governed_card` | Emite um model card só depois de uma checagem física do modelo: treina (ou carrega) o modelo de um preset, roda o `PhysicsGuardrail` e monta um `ModelCard` cujas métricas são as daquele run; um card inválido é recusado. Usa `pinneapple_hub.verified_card`, `pinneapple_hub.model_card` e `pinneapple_physics`. Roda com `cd apps/governed_card && uvicorn governed_card.api:app --port 8148`. | 2 | 3 |
| `apps/guiagent` | Agente que executa estudos de engenharia em software com interface gráfica (CAD, pré-processador, solver, pós) dentro do seu contêiner, a partir de uma frase ou de um workflow, e devolve os arquivos com a evidência de cada passo; há uma aprovação antes do solver. Usa `pinneapple_tools.automation` e `pinneapple_security`. Roda com `docker build -f apps/guiagent/Dockerfile -t guiagent .` e `docker run -p 8115:8115 guiagent`. | 2 | 2 |
| `apps/heatsink_inverse_ai` | Demo de projeto inverso por gradiente de um dissipador de aletas retas para CPU (número, altura e espessura das aletas numa base fixa de 80 x 80 mm) através de um surrogate neural congelado, reconferido contra a física. O próprio README diz que não dimensiona um dissipador real (para isso existe `heatsink_sizer`). Lógica em `pinneapple_design.surrogate_inverse.heatsink_fins`; roda com `PYTHONPATH=../..:../_shared uvicorn heatsink_inverse_ai.api:app --port 8160`. | 2 | 5 |
| `apps/heatsink_sizer` | Protótipo SaaS (dia 1 da série) que dimensiona dissipadores de placas e aletas a partir de correlações publicadas de convecção forçada e natural, perda de carga e uma solução 3-D da base, e devolve projetos fabricáveis verificados pela física. Usa `pinneapple_design` e `pinneapple_physics`. Roda com `uvicorn heatsink_sizer.api:app --port 8080` ou `docker run -p 8080:8080 heatsink-sizer`; nenhum teste cita a pasta. | 7 | 0 |
| `apps/helio_tfc` | Estima para onde vai o envelope do ciclo solar: EMD do número diário de manchas solares, amplitude de Hilbert do componente de cerca de 11 anos e um ajuste por máquina de aprendizado extremo com valores de ponta amarrados (Theory of Functional Connections). O README avisa que é um método, não uma previsão validada. Usa `pinneapple_systems.time_series.solar_cycle`; roda com `cd apps/helio_tfc && uvicorn helio_tfc.api:app --port 8149`. | 2 | 4 |
| `apps/hub` | Catálogo de ferramentas prontas, modelos treinados, geometrias, datasets, fluxos e exemplos: buscar, ver o run gravado, copiar código, baixar arquivos, abrir no Flow Studio e rodar um modelo do zoo. Tudo vem do índice gerado por `pinneapple_hub.library`; a pasta só tem rotas e página. Roda com `PYTHONPATH=../..:../_shared:. uvicorn hub.api:app --port 8116` dentro de `apps/hub`. | 2 | 14 |
| `apps/identifiability_lab` | Antes de estimar parâmetros, verifica se as medições conseguem determiná-los (por exemplo, numa aleta só h/k é identificável) e diz o que medir em vez disso. Usa `pinneapple_analysis`. Roda com `cd apps/identifiability_lab && uvicorn identifiability_lab.api:app --port 8108`. | 2 | 4 |
| `apps/interop_hub` | Converte resultados de simulação (OpenFOAM, CalculiX, VTK/VTU, Gmsh, STL, CSV/HDF5/NPZ) num dataset físico neutro (`pinneapple.physical_dataset/1`, com unidades e metadados) e dali para VTU, HDF5, Parquet, CSV, NPZ, JSON ou Zarr. Usa `pinneapple_data`. Roda com `cd apps/interop_hub && uvicorn interop_hub.api:app --port 8091`; nenhum teste cita a pasta. | 2 | 0 |
| `apps/inverse_heat` | Estima o coeficiente de convecção h a partir de poucos termopares com uma PINN montada pelo `PINNFactory` (h é um parâmetro treinável), em 1D, 2D e 3D, e confere cada resposta contra uma solução independente. Usa `pinneapple_physics` e `pinneapple_analysis`. Roda com `cd apps/inverse_heat && uvicorn inverse_heat.api:app --port 8086 --workers 1`; nenhum teste cita a pasta. | 7 | 0 |
| `apps/inverse_studio` | Um mesmo problema inverso (por exemplo o campo h(x) de uma aleta) resolvido por cinco métodos com os mesmos dados, cada um conferido. Reusa a física do Inverse Heat Lab via `pinneapple_analysis.inverse_problems`. Roda com `cd apps/inverse_studio && uvicorn inverse_studio.api:app --port 8104`. | 2 | 3 |
| `apps/mesh_quality` | Lê uma malha (OpenFOAM, Gmsh, VTK, STL, Abaqus/CalculiX e outros formatos meshio) e diz em segundos onde ela é ruim, quanto, e como corrigir, com veredito PASS/WARNING/FAIL e regiões problemáticas. Usa `pinneapple_data`. Roda com `cd apps/mesh_quality && uvicorn mesh_quality.api:app --port 8088`; nenhum teste cita a pasta. | 2 | 0 |
| `apps/meshbridge` | Recebe uma malha de CFD, FEM ou CAD (CGNS, Exodus II, Fluent ASCII, Abaqus `.inp` e outras) e devolve pontos e campos em NPZ ou uma superfície triangular reparada (STL/OBJ/PLY). Usa `pinneapple_simulation.external_solvers.cfd_formats.convert`. Roda com `cd apps/meshbridge && uvicorn meshbridge.api:app --port 8141`. | 2 | 4 |
| `apps/meshless_lab` | Mede, no problema do usuário (Poisson, Helmholtz ou calor em vários domínios ou numa nuvem de pontos), onde métodos sem malha, com malha e neurais ganham uns dos outros. Usa `pinneapple_simulation` e `pinneapple_core`. Roda com `cd apps/meshless_lab && uvicorn meshless_lab.api:app --port 8109`. | 2 | 5 |
| `apps/model_benchmarking` | Comparação fixa: quatro arquiteturas PINN (Vanilla, SIREN, PIELM, XTFC) treinadas ao vivo no mesmo Burgers viscoso `burgers_1d` e ranqueadas por precisão e por uma checagem do resíduo da EDP. Usa `pinneapple_arena`. Roda com `PYTHONPATH=../_shared uvicorn model_benchmarking.api:app --port 8182`; o estado fica em memória e se perde ao reiniciar. | 2 | 2 |
| `apps/model_gateway` | Publica modelos físicos (do zoo, ONNX, `model.zip` do Surrogate Studio ou remotos) atrás de endpoints versionados que checam cada consulta contra a faixa de validade, registram e medem o uso. Usa `pinneapple_registry.ModelGateway` e `pinneapple_security`. Roda com `uvicorn model_gateway.api:app --port 8114` (Dockerfile). | 2 | 6 |
| `apps/model_lineage` | Mostra como um resultado foi produzido: o fio digital do CAD à previsão, com arquivo, versão, software, parâmetros, data, sha256 e origem de cada artefato, como um grafo consultável. Usa `pinneapple_data`. Roda com `cd apps/model_lineage && uvicorn model_lineage.api:app --port 8090`; nenhum teste cita a pasta. | 2 | 0 |
| `apps/nozzle_screen` | Triagem rápida de bocais de foguete: estimativa ideal isentrópica 1-D (`pinneapple_physics.closed_form.rocket_nozzle`) e uma checagem cruzada por PINN que avisa quando as duas discordam (`pinneapple_analysis.verification.nozzle_crosscheck`). Roda com `cd apps/nozzle_screen && uvicorn nozzle_screen.api:app --port 8146`. | 2 | 2 |
| `apps/observatory` | O céu e o sistema solar agora, vistos de onde o usuário está, em 3-D: estrelas do Yale Bright Star Catalogue, constelações, Sol, Lua, planetas e satélites, com modo de apontar o celular. Os números vêm de `pinneapple_physics.astro`. Roda com `cd apps/observatory && uvicorn observatory.api:app --port 8094`; os testes de navegador ficam na própria pasta (`npx playwright test`, workflow `observatory.yml`), e nenhum teste Python cita a pasta. | 3 | 0 |
| `apps/orbitwatch` | Compara duas imagens de satélite do mesmo lugar e gera um relatório HTML com a mudança de refletância por banda e de NDVI. Usa `pinneapple_perception.change_detection` e, para buscar Sentinel-2 no Google Earth Engine, `pinneapple_perception.sentinel2`. Roda com `cd apps/orbitwatch && uvicorn orbitwatch.api:app --port 8150`. | 2 | 5 |
| `apps/parameter_discovery` | Problema-teste fixo: uma PINN recupera a difusividade escondida de uma barra (equação do calor) a partir de leituras ruidosas, comparando a estimativa com o limite de Cramér-Rao. Usa `pinneapple_analysis`. Roda com `PYTHONPATH=../_shared uvicorn parameter_discovery.api:app --port 8183`. | 2 | 2 |
| `apps/patchdose_ai` | Exercício de transferência de massa (o README diz que não é ferramenta médica nem de dosagem): acha o menor adesivo de liberação por membrana para uma taxa e um tempo de uso, por um surrogate neural reconferido contra a lei de Fick. Lógica em `pinneapple_design.surrogate_inverse.transdermal_patch`; roda com `PYTHONPATH=../..:../_shared uvicorn patchdose_ai.api:app --port 8162`. | 2 | 4 |
| `apps/pcb_hotspot` | Protótipo SaaS (dia 2 da série) de análise térmica de placas de circuito: temperaturas de junção, pontos quentes e o que os corrige, em cerca de um segundo. Usa `pinneapple_physics`. Roda com `uvicorn pcb_hotspot.api:app --port 8081` ou `docker run -p 8081:8081 pcb-hotspot`; nenhum teste cita a pasta. | 4 | 0 |
| `apps/physics_copilot` | Um problema de física em linguagem natural vira uma PINN treinada e o relatório recalculado do guardrail: análise dimensional, escolha de um preset registrado pelo LLM, treino e checagem do resíduo em pontos novos. Usa `pinneapple_analysis.verification.pipeline` (com `pinneapple_llm`). Roda com `cd apps/physics_copilot && uvicorn physics_copilot.api:app --port 8142`. | 2 | 3 |
| `apps/physics_design` | Otimiza projetos físicos paramétricos com objetivos e restrições explícitos e compara métodos por gradiente, evolutivos e bayesianos com o mesmo orçamento, mostrando quais candidatos o modelo de alta fidelidade confirma. Usa `pinneapple_design`. Roda com `cd apps/physics_design && PYTHONPATH=../_shared uvicorn physics_design.api:app --port 8106`. | 2 | 2 |
| `apps/pinn_arena` | Escolhe-se um problema físico e algumas arquiteturas PINNeAPPle, treina-se sob um protocolo, compara-se num conjunto separado e monta-se um leaderboard; uma segunda seção roda bibliotecas de terceiros pelo próprio pacote delas. Lógica em `pinneapple_arena` (backend FastAPI e frontend React). Roda com `cd apps/pinn_arena && PYTHONPATH=../_shared uvicorn pinn_arena.api:app --port 8130` (o frontend se compila antes com `npm run build`). | 21 | 25 |
| `apps/playground` | Site estático: desenhar um aerofólio, dimensionar um dissipador ou dirigir uma roda de rover lunar e ver um modelo do zoo responder no navegador (ONNX), com checagens físicas e veredito de confiança. Os modelos e checagens vêm de `pp.zoo.export_web` (`pinneapple_hub.zoo.web`) e `pinneapple_design.geometry_library.web`. Roda com `python apps/playground/build.py --serve`; testes JS com `npm test` e workflow `playground.yml`. | 1 | 2 |
| `apps/predictive_twin` | Gêmeo digital de um medidor real de rio (USGS 01646500, Potomac): baixa os últimos dias de vazão da API pública do USGS, calibra uma recessão exponencial, prevê as últimas 24 h e sinaliza quando o rio se afasta da previsão. Usa `pinneapple_systems`. Roda com `PYTHONPATH=../_shared uvicorn predictive_twin.api:app --port 8184` (precisa de internet). | 2 | 3 |
| `apps/ridetune_ai` | Acha a seção mais leve de um braço de suspensão (modelado como viga em balanço) para uma meta de vida em fadiga, por um surrogate neural da tensão reconferido contra a fórmula fechada, com sensibilidades locais. Lógica em `pinneapple_design.surrogate_inverse.suspension_arm`; roda com `PYTHONPATH=../..:../_shared uvicorn ridetune_ai.api:app --port 8163`. | 2 | 4 |
| `apps/rom_lab` | Constrói modelos de ordem reduzida (POD + RBF, POD + GPR e outros) a partir de snapshots de simulação e mostra o erro em dados separados, por posto. Usa `pinneapple_neural.architectures.rom`, `pinneapple_data`, `pinneapple_lab` e `pinneapple_core`. Roda com `cd apps/rom_lab && uvicorn rom_lab.api:app --port 8102`. | 3 | 4 |
| `apps/rover_mobility` | Triagem de mobilidade de roda rígida: força de tração contra escorregamento por uma PINN de Bekker-Wong, com cada restrição física reconferida isoladamente. Usa `pinneapple_simulation.terramechanics.mobility`. Roda com `cd apps/rover_mobility && uvicorn rover_mobility.api:app --port 8147`. | 2 | 2 |
| `apps/satscreen` | Triagem rápida de conjunção de satélites: aproximação mínima e estimativa de probabilidade de colisão no movimento relativo de Clohessy-Wiltshire, mais propagação de dois corpos com J2; não treina PINN por requisição. Usa `pinneapple_physics.astro.conjunction`. Roda com `cd apps/satscreen && uvicorn satscreen.api:app --port 8144`. | 2 | 3 |
| `apps/sensitivity_lab` | Mostra quais parâmetros controlam de fato o resultado e quais podem ser congelados, com análise global e incerteza da resposta. Usa `pinneapple_analysis`, `pinneapple_data` e `pinneapple_core`. Roda com `cd apps/sensitivity_lab && uvicorn sensitivity_lab.api:app --port 8099`. | 2 | 7 |
| `apps/simulation_campaign` | Planeja campanhas de simulação: a partir de um espaço de parâmetros, gera um projeto de experimentos (hipercubo latino e outros), a ordem dos runs e os arquivos de caso. Usa `pinneapple_data`, `pinneapple_simulation` e `pinneapple_security`. Roda com `cd apps/simulation_campaign && uvicorn simulation_campaign.api:app --port 8096`. | 2 | 8 |
| `apps/simulation_comparator` | Compara um resultado com uma referência campo a campo (MAE, RMSE, máximo, normalizados) e mostra onde diferem; aceita OpenFOAM, VTK, CalculiX, Gmsh e tabelas de pontos, inclusive previsões de IA. Usa `pinneapple_data`. Roda com `cd apps/simulation_comparator && uvicorn simulation_comparator.api:app --port 8089`; nenhum teste cita a pasta. | 2 | 0 |
| `apps/simulation_metadata` | Lê (sem executar) um caso OpenFOAM, CalculiX/Abaqus ou um histórico de resíduos e devolve um registro estruturado (`pinneapple.simmeta/1`) com solver, malha, condições de contorno, parâmetros e veredito de convergência. Usa `pinneapple_data`. Roda com `cd apps/simulation_metadata && uvicorn simulation_metadata.api:app --port 8084`; nenhum teste cita a pasta. | 2 | 0 |
| `apps/simulation_preflight` | Antes de gastar horas de solver, confere se um caso OpenFOAM ou um deck CalculiX/Abaqus está montado certo (malha, materiais, contornos, condições iniciais, solver) e dá PASS/WARNING/FAIL. Usa `pinneapple_data`. Roda com `cd apps/simulation_preflight && uvicorn simulation_preflight.api:app --port 8087`. | 2 | 5 |
| `apps/simulation_runner` | Executa campanhas de simulação como cadeias de passos (gerar entrada, malhar, resolver, pós-processar, checar), registra o que terminou e retoma de onde parou, com executores locais ou Docker isolado. Usa `pinneapple_core` e `pinneapple_tools`. Roda com `cd apps/simulation_runner && uvicorn simulation_runner.api:app --port 8110`. | 2 | 4 |
| `apps/soundshape_ai` | Acha raio e comprimento do duto de uma caixa bass-reflex para uma frequência de sintonia, por um surrogate neural reconferido contra a fórmula de Helmholtz, com a curva de compromisso entre duto pequeno e sintonia exata. Lógica em `pinneapple_design.surrogate_inverse.reflex_port`; roda com `PYTHONPATH=../..:../_shared uvicorn soundshape_ai.api:app --port 8164`. | 2 | 4 |
| `apps/surrogate_speed` | Demo fixa com sete dissipadores: compara fórmula fechada, previsão de um surrogate (cronometrada) e uma solução de volumes finitos com índice de convergência de malha. O README avisa que as razões de velocidade são contra outro solver de referência que o do projeto original. Usa `pinneapple_design`; roda com `PYTHONPATH=../_shared uvicorn surrogate_speed.api:app --port 8185`. | 2 | 2 |
| `apps/surrogate_studio` | Transforma resultados de simulação existentes num modelo preditivo rápido, medido em runs que não viu e servido num endpoint versionado com a faixa de validade imposta. Usa `pinneapple_analysis`, `pinneapple_registry` e `pinneapple_core`. Roda com `cd apps/surrogate_studio && uvicorn surrogate_studio.api:app --port 8097`. | 2 | 4 |
| `apps/text2part` | Descreve-se uma peça em linguagem natural e recebe-se uma malha STL ou um CAD paramétrico STEP com suas medidas; o LLM só escolhe uma receita de um menu fixo, nunca escreve código CAD. Usa `pinneapple_llm` (`draft_part`, `measure_part`). Roda com `cd apps/text2part && uvicorn text2part.api:app --port 8140` e precisa de um servidor Ollama. | 2 | 4 |
| `apps/twin_ops` | Exemplo de ponta a ponta (não é serviço web) de um gêmeo digital comissionado para um trocador casco-e-tubo com incrustação: contrato assinado, preflight que pode recusar, backtest separado, arquivo de evidência e escada de deriva. Usa `pinneapple_systems`. Roda com `python apps/twin_ops/demo.py`. | 1 | 24 |
| `apps/twin_runtime` | Põe um modelo físico validado (tanque aquecido concentrado) ou um surrogate ao lado de dados operacionais, com assimilação por EnKF, e diz quando parar de confiar nele. Usa `pinneapple_systems.operations`, `pinneapple_data` e `pinneapple_core`. Roda com `uvicorn twin_runtime.api:app --port 8103` (Dockerfile). | 2 | 6 |
| `apps/uncertainty_lab` | Mede quando uma previsão é confiável e quando o caso deve ir ao solver de alta fidelidade (fora dos dados, dados esparsos, erro maior que a margem ao limite). Usa `pinneapple_analysis`, `pinneapple_data` e `pinneapple_core`. Roda com `cd apps/uncertainty_lab && uvicorn uncertainty_lab.api:app --port 8098`. | 2 | 8 |
| `apps/validation_lab` | Verifica se um modelo ou resultado é consistente física e numericamente (equações, conservação, contornos, unidades, referência, faixa de treino) e grava o registro de cada checagem. Usa `pinneapple_veriphysics`, `pinneapple_check`, `pinneapple_data`, `pinneapple_neural` e `pinneapple_lab`. Roda com `cd apps/validation_lab && uvicorn validation_lab.api:app --port 8107`. | 3 | 5 |
| `apps/verification_gap` | Caso fixo de demonstração: um disco de freio que passa na checagem nominal e o que as tolerâncias de usinagem fazem com essa aprovação, por Monte Carlo sobre um modelo fechado. Usa `pinneapple_design` e `pinneapple_analysis`. Roda com `PYTHONPATH=../_shared uvicorn verification_gap.api:app --port 8180`. | 2 | 2 |
| `apps/verified_physics` | O LLM propõe a física e o PINNeAPPle executa e reconfere mecanicamente, com a evidência anexada; inclui arquiteturas candidatas, incerteza, intervenções causais e grafo de conhecimento. Usa `pinneapple_analysis.verification`. Roda com `cd apps/verified_physics && uvicorn verified_physics.api:app --port 8143`. | 2 | 5 |
| `apps/zoo_space` | Página estática (`index.html`, `app.js`, `space.css`) do demo do zoo publicada como Space no Hugging Face por `scripts/zoo/publish_to_hub.py` e pelo workflow `publish-zoo.yml`: mostra cada modelo do zoo numa varredura das entradas, com banda de incerteza e onde as checagens físicas avisam. Não tem README nem teste que cite a pasta. | 0 | 0 |

## 2. Exemplos (`examples/`)

88 itens.

| item | o que faz | arquivos .py | testes |
|---|---|---:|---:|
| `examples/README.md` | # Pinneapple Examples | 0 | 0 |
| `examples/architectures` | Pinneapple Models — Showcase Examples: These examples are **curated demos** focused on the full breadth of `pinneapple_models`: registry/catalog usage, neural operators, PINNs (including inverse parameters), ROMs, GNNs, and time-series transformers. | 12 | 0 |
| `examples/arena_pipelines` | 00_list_datasets.py — List and inspect all available PINNeAPPle datasets. | 10 | 0 |
| `examples/autolab` | The self-optimizing engineer (PINNeAPPle Autonomous Physics Lab, demo 2: thermal). | 1 | 0 |
| `examples/autolab_aero` | The AI Aerodynamicist (PINNeAPPle Autonomous Physics Lab, stage 1: the scientific cycle without an agent). | 1 | 0 |
| `examples/autolab_aero_agent` | The AI Aerodynamicist, stage 2: an agent reads the mission and runs the lab only through controlled tools. | 1 | 0 |
| `examples/autolab_digital_twin` | The digital twin that knows when it is wrong (PINNeAPPle Autonomous Physics Lab, demo 3). | 1 | 0 |
| `examples/autolab_rover` | PINNeAPPle Autonomous Physics Lab, demo 4: the rover that learns the planet. | 1 | 0 |
| `examples/benchmark` | An equal-budget benchmark in one call: PINN vs FNO vs DeepONet on 1-D Burgers. | 1 | 0 |
| `examples/benchmark_suite` | PINNeAPPle Arena — Examples: These examples are meant to **show the full potential of `pinneapple_arena`**: | 20 | 0 |
| `examples/black_hole_weather` | Black-hole weather: forecasting the gas around a black hole, and knowing for how long to trust it: This is a reproduction and extension of **Duarte, Nemmen & Navarro (2022)**, *Black hole weather forecasting with deep learning: a pilot study*: - MNRAS 512, 5848; [arXiv:2102.06242](https://arxiv.org/abs/2102.06242); - original code (MIT): [black-hole-group/DL_BH_fluids](https://github.com/black-hole-group/DL_BH_fluids). | 5 | 20 |
| `examples/bricks` | Self-modifying interlocking-brick builds: simulate, judge by physics, redesign, repeat. | 1 | 20 |
| `examples/calculix_cantilever` | Viga em balanço (CalculiX): Execução de referência do solver CalculiX (`ccx`) de uma viga em balanço de 1,0 × 0,1 × 0,1 (m). | 0 | 0 |
| `examples/calculix_pyvista_cantilever.py` | CalculiX cantilever -> PyVista image: von Mises stress on the deformed beam. | 1 | 0 |
| `examples/chain_of_whys` | Chain of whys: four projects, four scales, the same reasoning (pinneapple_reason). | 1 | 0 |
| `examples/condensation_falsification` | Condensation falsification campaign: the measured runs of docs/research/condensation_falsification.md. | 1 | 0 |
| `examples/core_primitives` | Poisson PINN on the core primitives (Domain, Geometry, FunctionField, Mesh, Field). | 2 | 0 |
| `examples/data_api` | A PINN, a neural operator and an inverse problem trained from the same loader API. | 1 | 0 |
| `examples/data_pipeline` | 03_prefetch_streaming_gpu.py | 8 | 0 |
| `examples/design_from_intent` | Design from intent: five requests through the same pipeline. | 1 | 0 |
| `examples/electrodynamics` | Electrodynamics & Electromagnetics Examples: Physics-informed neural network (PINN) solutions to classical electrostatics, magnetostatics and electromagnetics problems. | 17 | 20 |
| `examples/end_to_end` |  | 6 | 0 |
| `examples/enso_forecasting` | El Nino (ENSO) prediction: data-driven, physical, physics-informed and hybrid models under one honest protocol. | 1 | 20 |
| `examples/first_example` | Your first PINNeAPPle result in about a minute on a laptop CPU: solve Burgers' equation with a physics-informed network, check it against the exact solution and plot both. | 1 | 0 |
| `examples/flight_dynamics` | Verification of the flight-dynamics library: turbulence spectra and variances against MIL-F-8785C, trim of the four aircraft held for 10 s stick fixed, energy / angular-momentum conservation of the integrator, the lattice-in-the-loop lift against the steady vortex lattice. Writes … | 8 | 20 |
| `examples/flow_studio` | Write the example flows of Flow Studio into ``pinneapple_flow/flows/`` (the flows the GUI's "Open..." lists). | 3 | 0 |
| `examples/free_surface` | Free-surface flow (interFoam, VOF): `01_runner_slope_study.py`: a tapping jet falling into a melt runner held by a weir; the floor slope (+2°, 0°, -2°) changes the depth of metal under the jet. 2-D sections with `pp.cfd.FreeSurfaceFlow`, k sampled 5 cm above the floor while the solver runs and averaged in time. Needs OpenFOAM. Results: `docs/core_concepts/free_surface.md`. | 1 | 0 |
| `examples/freecad_openfoam` | FreeCAD -> OpenFOAM pipelines: three validated worked studies, a parameter sweep and a surrogate. | 1 | 0 |
| `examples/geomechanics` | Geomechanics with the material point method: `01_granular_column_collapse.py`: sand columns of aspect ratio 0.5 to 3 collapse on a rough floor; runout against Lube et al. (2005); frames written in GeoTaichi's layout and read back through the GeoTaichi bridge. About 6 minutes on one core. Results and limits: `docs/core_concepts/geomechanics_mpm.md`. | 1 | 0 |
| `examples/geometry` | pinneapple_geom examples: This folder focuses on geometry + meshing workflows that typically appear in Physics-AI / PINN pipelines: | 8 | 0 |
| `examples/geometry_library` | Wind on the CAARC standard tall building with OpenFOAM, through the generic STL path of ``pp.cfd``. | 3 | 0 |
| `examples/geophysics` | Earth-system blocks in one script (about one minute on a laptop CPU): | 1 | 0 |
| `examples/getting_started` | Getting Started Examples — Tier 1: Explorer: These examples are your **first step into Physics AI**. Each one: | 10 | 0 |
| `examples/handoff_cases` | Engineering hand-off cases: Two self-contained case packages for testing an engineering-simulation agent (or an engineer) from a cold start. Each folder holds the task (`brief.md`), the geometry, the simulation set-up and scripts, results that were really computed, and verified references; `<case>.zip` is the same folder as one archive. | 13 | 0 |
| `examples/hpo_experiments` | pinneapple_researcher examples: These examples are designed to show the full workflow: | 8 | 0 |
| `examples/industrial_presets` | Presets industriais: Problemas de engenharia já montados, resolvidos e validados com o PINNeAPPle. Cada pasta tem a ficha (entradas SI com a fonte de cada valor padrão), os resultados rodados aqui, a validação com o erro medido e as figuras. Regenere com `python -m pinneapple_presets report all`. | 0 | 0 |
| `examples/inspection` | pinneapple_analysis.inspection example 01: eddy-current NDE end to end. | 1 | 0 |
| `examples/latent` | Latent surrogate with a trust card: error, speed and calibration against the full solver. | 1 | 0 |
| `examples/lbm` | 3-D lattice Boltzmann wind tunnel: drag on a sphere against Stokes' law and the Schiller-Naumann correlation, in 32-bit and 16-bit storage of the distributions. | 1 | 0 |
| `examples/meshgraphnet` | MeshGraphNet in PINNeAPPle: Implementation: - `pinneapple_neural/architectures/graphnn/mesh_graph_net.py`: the encoder, processor and decoder network, registered as `mgn` / `meshgraphnet`. - `mgn_dynamics.py`: the transient training recipe of Pfaff et al. It covers normalization, training noise, the Δv target, and autoregressive rollout with boundary nodes held fixed. | 6 | 6 |
| `examples/model_zoo` | Model zoo quickstart: load a pre-trained model, predict, compare with the reference, validate, benchmark and fine-tune -- every step on the shipped, verified airfoil surrogate and its dataset. | 1 | 0 |
| `examples/modelica` | Modelica models as Physics AI data: Models: `models/DrainingTank.mo`, `models/LadleCooling.mo`. OpenModelica is used from the PATH or from Docker (`docker pull openmodelica/openmodelica:v1.24.4-minimal`). Results and discussion: `docs/core_concepts/modelica.md`. | 4 | 6 |
| `examples/molecular` | A protein on a gold surface: orient, build the Au(111) slab, run a short OpenMM MD and visualise it. | 1 | 0 |
| `examples/motor_magnetics` | Surface-PM machine (4 poles, 24 slots, nonlinear iron): cogging torque, torque vs current angle, back-EMF and losses with the 2D A_z FEM of pinneapple_simulation.electromagnetics. | 1 | 0 |
| `examples/near_random_forecasting` | Visual pack of the near-random forecasting study: four Blender Cycles renders and two 2-D figures. | 2 | 20 |
| `examples/non_hermitian` | Exceptional-point tuning in PT-symmetric dimers: the no-go theorem (Zenodo 23287622) checked symbolically and numerically, the three routes around it, the hypothesis map as a physics audit, and the figure. | 1 | 0 |
| `examples/numerical_solvers` | pinneapple_solvers examples: This folder contains runnable scripts showcasing the **numerical-solvers layer**: | 13 | 0 |
| `examples/nvidia` | PINNeAPPle with the NVIDIA stack: Every example puts PINNeAPPle at the centre (the problem, the reference solvers, the geometry checks, the physics checks and the trust decision) and joins it with a different combination of stacks. The glue lives in `pinneapple_tools.nvidia`; each example is a thin `run.py` that writes `results/architecture.png` (PINNeAPPle's role and each stack's role, and what was *not* run here), result figures, `summary.json` and `run_info.json` (versions, hardware, timings, which components ran). | 25 | 39 |
| `examples/openfoam_campaign` | OpenFOAM parameter campaign: One base case, six runs that change the jet speed and elevation, split between two machines. | 1 | 0 |
| `examples/pde_environment` | End-to-end demonstration for the astrophysics specialization: train a PINN on the `kepler_two_body_orbit` preset and compare the trained network's predicted trajectory against the exact (Kepler's-equation) solution. | 9 | 10 |
| `examples/physics_audit` | Physics audit on pipe friction: models judged by their physics, not only by their past errors. | 1 | 0 |
| `examples/physics_check` | A stand-in for a drag surrogate Cd(Re, aoa) fitted on wind-tunnel points (Re 1e5 .. 1e6, aoa 0 .. 10 deg). | 7 | 22 |
| `examples/physics_condensation` | Adaptive Physics Condensation, track 1: the measured benchmark of docs/research/physics_condensation.md. | 1 | 0 |
| `examples/physics_db` | Quickstart: discover a dataset via CMR/earthaccess and inspect variables. | 7 | 0 |
| `examples/physics_ensemble` | Side by side: the five-family ensemble reacting only to past errors (left) and with the current-case physics residual against the level each model has in its own training regime (right, ``residual_lookahead`` + ``residual_baseline``), on the same stream of cases. | 5 | 3 |
| `examples/physics_learning_verification` | Physics learning verification, track 3: the measured study of docs/research/physics_condensation.md (section 6). | 1 | 0 |
| `examples/physics_module` | Hybrid model: a coarse finite-element solve plus a neural correction, trained end to end. | 1 | 0 |
| `examples/physics_optimizer` | One constrained design problem through one interface, six optimizers. | 1 | 0 |
| `examples/physics_query` | Physics query engine, track 2: the measured demonstration of docs/research/physics_condensation.md (section 7). | 1 | 0 |
| `examples/pinn_solver` | pinneapple_physics.pinn_solver examples: This folder focuses on the **PINN-specific** layer of PINNeAPPle. | 10 | 0 |
| `examples/pinneapple_and_physicsnemo` | Wave equation 1D u_tt - (c^2 u_x)_x = 0 | 4 | 0 |
| `examples/pinneapple_benchmark` | 01_pde_comparison.py — PINNeAPPle PDE Architecture Benchmark | 2 | 0 |
| `examples/pinneapple_cosim` | 01_spring_mass_pinn.py — 2-DOF Coupled Spring-Mass System with Two PINN Models | 1 | 0 |
| `examples/problem_designer` | pinneapple_problemdesign examples: This folder showcases **problem elicitation + specification + planning**. | 5 | 22 |
| `examples/proof_graph` | Proof-graph experiment: does the no-go theorem for linear exceptional-point tuning (PT dimers, Zenodo 23287622) extend to PT tetramers whose bonds are modulated independently? An open question of the paper (Q2), attacked as a graph of assumptions, lemmas and a theorem, each with exact (sympy) and … | 1 | 0 |
| `examples/qualitative` | Qualitative preview of heat-sink variants, then the quantitative check. | 3 | 0 |
| `examples/quickstart` | 01 — First look: what is installed, and what is already solved for you. | 13 | 0 |
| `examples/reproductions` | Reproduce Gated X-TFC (Dwivedi, Schiassi, Sigovan, Sixou, arXiv:2510.01039v1) with the protocol frozen in TARGETS.md. | 3 | 0 |
| `examples/results_chassis` |  | 0 | 0 |
| `examples/rigid_body` | Rigid bodies with NVIDIA Newton: check the solver, then run a small scene. | 1 | 0 |
| `examples/science` | PINNeAPPle science: experiments: Every row is a quantity computed by an experiment of `pinneapple_science` (`pp.science`) and compared with its reference; the values come from the committed `result.json` files. Regenerate with `python -m pinneapple_science report all`. The map of every area (available, partial, planned) is [ATLAS.md](ATLAS.md) and [atlas.html](atlas.html). | 0 | 0 |
| `examples/scratch` | scratch: This directory holds working and experimental scripts that were used during development (benchmarking snippets, one-off result-generation and report-embedding utilities, etc.) but were never organized into the curated topic-folder examples (`examples/architectures/`, `examples/geometry/`, and so on). They may be less polished, less documented, or narrower in scope than the rest of `examples/`, and some assume they're run from the repository root with the project installed in editable mode — … | 6 | 0 |
| `examples/security` | Secure Physics AI pipeline, end to end, with pinneapple_security. | 2 | 0 |
| `examples/solver_api` | The same problem through one call: a classical FEM solve and a PINN, compared against the exact solution. | 1 | 0 |
| `examples/solvers` | Examples: time-series decomposition solvers. | 3 | 0 |
| `examples/sports` | Ball flight: drag, Magnus and gravity from tracked positions; where the inverse problem is posed, and whether the physics predicts real broadcast-tracking flights better than data-only extrapolation and boosted trees. | 5 | 0 |
| `examples/studio` | Any simulation result in 3D, with three calls: load, export, render. | 7 | 6 |
| `examples/systems` | Component registry + toolbox + physics-informed fit, end to end. | 1 | 0 |
| `examples/time_series` | PINNeAPPle Time Series Examples: This folder contains runnable scripts that demonstrate **forecasting workflows** supported by `pinneapple_timeseries`. | 10 | 0 |
| `examples/topology_optimization` | Half MBB beam and cantilever: SIMP compliance minimisation with pinneapple_design.topology. | 3 | 0 |
| `examples/trainer` | pinneapple_train example 02: DataModule + robust preprocessing. | 5 | 0 |
| `examples/use_cases` | Compact range quiet zone by physical optics: validation, feed gain x reflector size (x rolled edge) sweep, a surrogate with held-out error and the trade-off map. Generic example parameters (3 GHz, F = 1.5 m, a 1.2 m square section with a superellipse junction, a 0.6 m quiet zone at 3 m). | 34 | 17 |
| `examples/validation` | PINN, steady heat conduction: -lap T = q on the unit square, T = 0 on the walls (manufactured source, exact T). | 12 | 4 |
| `examples/validation_arena` | Run an external model (for instance the FNO or MeshGraphNet of a PhysicsNeMo experiment) through the hardened defence of the validation arena (#707), and compare each of its outputs with the reference solver. | 2 | 0 |
| `examples/visualizations` | Run all PINNeAPPle visualization examples. | 8 | 0 |
| `examples/vs_physicsnemo` | PINNeAPPle vs PhysicsNeMo — Comparison & Integration Guide: > As linhas marcadas como medidas vêm do benchmark entre frameworks > ([`benchmarks/vs_frameworks/REPORT.md`](../../benchmarks/vs_frameworks/REPORT.md)): só CPU (Intel Xeon, 1 thread por run), > 3 a 5 seeds, nvidia-physicsnemo 2.2.2 e nvidia-physicsnemo.sym 2.4.0. Nada foi medido em GPU, que é onde o > PhysicsNeMo é otimizado. As outras linhas descrevem funcionalidades e não foram medidas. | 7 | 22 |
| `examples/weather_forecasting` | Weather forecasting on ERA5: how many days a model is good for, and extreme events: A global forecast model trained with `pinneapple_physics.weather` on ERA5 (WeatherBench2, public, no account), scored by lead time against persistence, climatology and the published IFS HRES, Pangu-Weather, Keisler and NeuralGCM forecasts on the same 2020 cases, with the useful horizon (ACC >= 0.6) and the no-skill horizon (RMSE reaches climatology), a per-forecast horizon from an ensemble, and seven extreme events forecast against what happened. | 2 | 6 |
| `examples/worldmodel` | Physics World Model — end-to-end example. | 1 | 0 |

## 3. Reproduções de papers

### `examples/reproductions/xtfc_schiassi2021`

- **Paper:** E. Schiassi, C. Leake, M. De Florio, H. Johnston, R. Furfaro, D. Mortari, "Extreme Theory of Functional Connections: A Physics-Informed Neural Network Method for Solving Parametric Differential Equations", arXiv:2005.10632v1, 2020 (versão de periódico: Neurocomputing 457, 2021, que não foi lida)
- **Resultado gravado:** Os alvos foram congelados em `TARGETS.md` antes de qualquer run, e a regra de status é `pinneapple_analysis.reproduction.status`. Segundo `examples/reproductions/xtfc_schiassi2021/README.md` e `RESULTS.md`, 7 de 10 alvos foram reproduzidos, em escala completa e em CPU. Exemplos: ODE 1 com erro máximo mediano 4.4e-16 contra 6.66e-16 do paper; PDE 1 com 1.4e-13 contra 5.05e-13; PDE 7 (não linear 3D + t) com 2.4e-5 contra 2.98e-5. Não reproduzidos: ODE 2 (y1 mediano 0.91 contra 5.54e-10; a iteração de Gauss-Newton diverge na maioria das seeds) e PDE 6 (5.5e-6 contra 1.66e-8). Dados brutos em `results.json` e `baselines.json`.

### `examples/reproductions/gated_xtfc_dwivedi2025`

- **Paper:** V. Dwivedi, E. Schiassi, M. Sigovan, B. Sixou, "Gated X-TFC: Soft Domain Decomposition for Forward and Inverse Problems in Sharp-Gradient PDEs", arXiv:2510.01039v1, 2025
- **Resultado gravado:** Alvos congelados em `TARGETS.md` antes de rodar; tudo medido por `run_gated.py`. Segundo `examples/reproductions/gated_xtfc_dwivedi2025/README.md`, só 2 de 9 alvos foram reproduzidos (faixa de fator 2): o problema direto com ν = 1e-2 (6.7e-4 contra 4.95e-4 do paper) e a camada-limite dupla com ν = 1e-4 (5.0e-3 contra 2.97e-3). Falharam os casos diretos com ν ≤ 1e-3 (por exemplo 3.9e-2 contra 5.86e-4 em ν = 1e-4) e os quatro casos inversos (por exemplo 41 % de erro em ν̂ contra 10.3 % em ν = 0.1). O README atribui a diferença a detalhes que o paper deixa em aberto (λ de Tikhonov, tipo de solução, apêndice ausente). Dados brutos em `part_*.json`.

### `examples/black_hole_weather`

- **Paper:** Duarte, Nemmen & Navarro, "Black hole weather forecasting with deep learning: a pilot study", MNRAS 512, 5848, 2022 (arXiv:2102.06242)
- **Resultado gravado:** Reprodução e extensão (o README diz as duas coisas). Os dados PLUTO do paper não estavam acessíveis, então as simulações 2.5-D foram refeitas com `pinneapple_physics.blackhole`, e a U-Net do paper foi portada camada por camada (com 16 filtros em vez de 32). Segundo `examples/black_hole_weather/README.md`, a porta fiel não bate a persistência no espaçamento de quadros usado aqui (erro 0.038 contra 0.0137 da persistência em +1200 M), enquanto a variante residual, que é adição do PINNeAPPle, bate a persistência em todo o rollout (0.0076 em +1200 M, com correlação de tendência 0.61). Resumos por modelo ficam em `examples/black_hole_weather/results/`.

## 4. Scripts (`scripts/`)

24 itens.

| item | o que faz | arquivos .py | testes |
|---|---|---:|---:|
| `scripts/README.md` | # scripts/ — Runbook do MVP (Flow Obstacle 2D) | 0 | 0 |
| `scripts/app_helpers_baseline.txt` | # Existing duplicates of library helpers in apps/ and examples/ (scripts/check_app_helpers.py). | 0 | 4 |
| `scripts/arena` |  | 5 | 0 |
| `scripts/astro` | Build the compact star and constellation catalogues shipped in pinneapple_physics/astro/data. | 3 | 0 |
| `scripts/build_method_status.py` | Compute the validation status of every item in ``pinneapple_catalog.methods`` from the test suites. | 1 | 2 |
| `scripts/check_app_helpers.py` | Flag reusable logic that is written in apps/ or examples/ instead of being imported from the library. | 1 | 0 |
| `scripts/check_dist.py` | Check the built distributions: every wheel package is in both the sdist and the wheel, and imports when the sdist is installed in a clean virtual environment (roadmap I5, #92). | 1 | 0 |
| `scripts/check_example_imports.py` | Check that every ``pinneapple*`` import in examples/ and templates/ resolves. | 1 | 0 |
| `scripts/check_experiment_manifests.py` | CI gate: every examples/* folder has a valid experiment.json and every Lab run reads as a valid manifest. | 1 | 0 |
| `scripts/check_release_version.py` | Fail unless the git tag, pyproject.toml, CITATION.cff, pinneapple.__version__ and CHANGELOG.md agree on the release version. | 1 | 0 |
| `scripts/cosmos` | scripts/cosmos — Cosmos (LLM / World Foundation Model) integration: Esses scripts assumem que você tem um endpoint HTTP para Cosmos (ou gateway interno) que aceita JSON e retorna JSON. | 3 | 0 |
| `scripts/dev` |  | 1 | 0 |
| `scripts/example_imports_baseline.txt` | examples/architectures/50_timeseries_transformer_forecast_toy.py | 0 | 2 |
| `scripts/fault_injection_matrix.py` | Regenerate the detection matrix of the validation checks (#697): every check against the fault library on Darcy 2-D and Burgers 1-D. | 1 | 0 |
| `scripts/geometry_library.py` | Generate, verify and publish the geometry library files. | 1 | 21 |
| `scripts/issues` | Snapshot the open issues of the repository into the contribute page (docs/org/contribute/index.html). | 1 | 0 |
| `scripts/mvp` | scripts/mvp — scripts locais “cola-tudo” do MVP: Aqui ficam scripts para: - gerar/copiar JSONs do bundle (conditions/manifest) - validar bundle contra schema + presença de arquivos - rodar uma matriz de runs da Arena (native + physicsnemo) | 3 | 0 |
| `scripts/omniverse` | scripts/omniverse — executar dentro do Omniverse Kit: Esses scripts rodam DENTRO do Omniverse Code / Isaac Sim. | 2 | 0 |
| `scripts/publish_issues.sh` | #!/usr/bin/env bash | 0 | 0 |
| `scripts/reference` | Build the module reference at docs/org/reference/: every package, subpackage and module of the repository with its docstring and public API (read from the source with ``ast``, nothing is imported), plus the curated explanations and the examples run by ``run_examples.py`` (code, printed output, … | 3 | 0 |
| `scripts/run_colab_notebooks.py` | Execute the Colab notebooks headless, top to bottom, and check that the committed copies carry no outputs. | 1 | 0 |
| `scripts/update_imports.py` | Step 1: Update all Python import statements to use the new mega-module paths. | 1 | 0 |
| `scripts/validate_model_card.py` | Governance-as-code gate for the model hub (see ``ROADMAP_PHYSICS_AI_HUB.md``, P1.5): reject a model-card push with no computed ``validation_metrics`` or no ``reference_source`` citation, the same schema check ``ModelCard.validate()`` already implements -- this script is just the CI-callable/CLI … | 1 | 8 |
| `scripts/zoo` | Build the zoo's bundled assets, model cards and ``manifest.json`` from the committed sources. | 10 | 28 |

## 5. Benchmarks (`benchmarks/`)

7 itens.

| item | o que faz | arquivos .py | testes |
|---|---|---:|---:|
| `benchmarks/data_io_bench.py` |  | 1 | 0 |
| `benchmarks/lbm_vortex_shedding.py` | Vortex shedding with the D2Q9 LBM solver: Strouhal number of a cylinder and a NACA 4412. | 1 | 0 |
| `benchmarks/lpm_ellipse_design_space.py` | Large Physics Model on an exact design space: potential-flow Cp on ellipses. | 1 | 0 |
| `benchmarks/shard_balance_bench.py` |  | 1 | 0 |
| `benchmarks/ssqn_helmholtz.py` | Paper benchmark: 2D Helmholtz with BFGS vs SSBFGS vs SSBroyden vs Adam. | 1 | 0 |
| `benchmarks/ssqn_paper_benchmarks.py` | Paper benchmarks for the self-scaled quasi-Newton optimizers (BFGS / SSBFGS / SSBroyden vs Adam). | 1 | 0 |
| `benchmarks/vs_frameworks` | Cross-framework benchmark: runners, driver and report (see PROTOCOL.md). | 15 | 0 |

## 6. Integração contínua (`.github/workflows/`)

| workflow | nome | gatilhos | jobs | o que faz |
|---|---|---|---|---|
| `docs.yml` | Deploy Docs to GitHub Pages | push, workflow_dispatch, contents | deploy |  |
| `draft-pdf.yml` | Draft PDF | push, pull_request, workflow_dispatch | paper |  |
| `extras.yml` | Install extras | push, pull_request, workflow_dispatch | extras |  |
| `observatory.yml` | Observatory (browser tests) | push, pull_request, workflow_dispatch | browser |  |
| `physics-check.yml` | Physics check (pinneapple check) | push, pull_request, workflow_dispatch, contents | passing-fixtures, failing-fixtures, darcy-trust, unit-tests |  |
| `playground.yml` | Zoo web export and Playground (ONNX parity, JS runtime, browser smoke test) | push, pull_request, workflow_dispatch | onnx-parity, browser, docker |  |
| `publish-zoo.yml` | Publish zoo to the Hugging Face Hub | workflow_dispatch, push, contents, group | publish, geometries |  |
| `release.yml` | Release to PyPI | push | build, publish |  |
| `security.yml` | Security | pull_request, push, schedule, workflow_dispatch, contents (cron `17 5 * * 1`) | scan, tests, image |  |
| `tests.yml` | Tests (audit + core suite) | push, pull_request, workflow_dispatch | test, generated, first-example, colab-notebooks |  |
| `training-callbacks.yml` | Training callbacks (Lightning, PhysicsNeMo, Hugging Face) | push, pull_request, workflow_dispatch, contents | callbacks |  |
| `validation-gallery.yml` | Validation gallery | push, pull_request, workflow_dispatch, contents | gallery |  |

## 7. Documentação existente (`docs/`)

190 arquivos Markdown. Lista com o título de cada um:

| arquivo | título |
|---|---|
| `docs/SUMMARY.md` | SUMMARY |
| `docs/apps/anomaly_detector.md` | Physics Anomaly Detector (day 28) |
| `docs/apps/benchmark_arena.md` | Physics AI Benchmark Arena |
| `docs/apps/calibration_lab.md` | Model Calibration Lab (day 18) |
| `docs/apps/control_lab.md` | Physics-Based Control Lab (day 29) |
| `docs/apps/evidence_report.md` | Engineering Evidence Report (day 30) |
| `docs/apps/hub.md` | PINNeAPPle Hub |
| `docs/apps/identifiability_lab.md` | Parameter Identifiability Lab (day 25) |
| `docs/apps/inverse_studio.md` | Inverse Model Studio (day 21) |
| `docs/apps/meshless_lab.md` | Meshless Method Lab (day 26) |
| `docs/apps/model_gateway.md` | Physics AI Model Gateway |
| `docs/apps/rom_lab.md` | Reduced Order Model Lab (day 19) |
| `docs/apps/surrogate_studio.md` | Surrogate Studio |
| `docs/apps/twin_runtime.md` | Digital Twin Runtime (day 20) |
| `docs/apps/validation_lab.md` | Physics Validation Lab (day 24) |
| `docs/architecture/execution_model.md` | Execution Model |
| `docs/architecture/package_layers.md` | Package Layers |
| `docs/architecture/system_overview.md` | System Overview |
| `docs/archive/overview_en.md` | PINNeAPPle — Library Overview |
| `docs/archive/overview_pt.md` | PINNeAPPle — Visão Geral da Biblioteca |
| `docs/assets/apps/README.md` | App images (30 apps with PINNeAPPle, days 1–14) |
| `docs/assets/proof_graph/pt_chain_no_go_report.md` | No linear exceptional-point tuning in PT-symmetric tetramers |
| `docs/core_concepts/adaptive_forecasting.md` | Adaptive forecasting: switching and combining models online |
| `docs/core_concepts/astro.md` | Positional astronomy and astrophysics (`pinneapple_physics.astro`) |
| `docs/core_concepts/astroml.md` | astroML: periodograms, deconvolved densities and fits with errors |
| `docs/core_concepts/backend.md` | Backend |
| `docs/core_concepts/black_hole_weather.md` | Black-hole weather: forecasting accretion flows with a trust horizon |
| `docs/core_concepts/core_primitives.md` | Core primitives: Domain, Geometry, Mesh, Field |
| `docs/core_concepts/dashboard.md` | Animation-centred dashboards |
| `docs/core_concepts/earth_system.md` | Earth-system building blocks and data assimilation |
| `docs/core_concepts/fitmesh.md` | From a problem description to checked CAD, fit-for-purpose meshes and one report |
| `docs/core_concepts/free_surface.md` | Free-surface flow: runners, channels, launders |
| `docs/core_concepts/geomechanics_mpm.md` | Geomechanics with the material point method |
| `docs/core_concepts/geometry_domain.md` | Geometry & Domain |
| `docs/core_concepts/lab.md` | PINNeAPPle Lab: experiments at scale, and a database of results and datasets |
| `docs/core_concepts/lbm_wind_tunnel.md` | Lattice Boltzmann wind tunnel and the FluidX3D bridge |
| `docs/core_concepts/lunar_rover_terramechanics.md` | Lunar rover terramechanics in 3D |
| `docs/core_concepts/model.md` | Model |
| `docs/core_concepts/modelica.md` | Modelica models as Physics AI data |
| `docs/core_concepts/molecular_visualization.md` | Molecular visualisation: proteins on metal surfaces |
| `docs/core_concepts/motor_and_topopt_audit.md` | Audit: electric-motor magnetics and topology optimization |
| `docs/core_concepts/multiscale_ensembles.md` | Multiscale ensembles: scales, filters and closures |
| `docs/core_concepts/newton_rigid_body.md` | Rigid bodies with NVIDIA Newton |
| `docs/core_concepts/non_hermitian.md` | Non-Hermitian physics: PT-symmetric dimers and exceptional-point tuning |
| `docs/core_concepts/openfoam_acceleration.md` | Faster implicit pressure solves in OpenFOAM |
| `docs/core_concepts/openfoam_campaigns.md` | OpenFOAM parameter campaigns on several machines |
| `docs/core_concepts/overview.md` | Core Concepts |
| `docs/core_concepts/physical_sample.md` | PhysicalSample |
| `docs/core_concepts/physics_audit.md` | Physics audit: which model to use, when to trust it, when to revisit the physics |
| `docs/core_concepts/physics_ensembles.md` | Adaptive ensembles of physics models |
| `docs/core_concepts/physics_optimizer.md` | PhysicsOptimizer |
| `docs/core_concepts/pinn.md` | PINN / Physics |
| `docs/core_concepts/problem_definition.md` | ProblemDefinition |
| `docs/core_concepts/proof_graphs.md` | Proof graphs: mathematics with the same evidence discipline |
| `docs/core_concepts/qualitative_preview.md` | Qualitative preview: physical impact of geometry changes before the numbers |
| `docs/core_concepts/reduced_order_models.md` | Reduced-order models |
| `docs/core_concepts/researcher_benchmarking.md` | Researcher & Benchmarking |
| `docs/core_concepts/security.md` | Security for scientific and industrial Physics AI (`pinneapple_security`) |
| `docs/core_concepts/solver.md` | Solver |
| `docs/core_concepts/studio.md` | 3D studio and external flow |
| `docs/core_concepts/training_pipeline.md` | Training Pipeline |
| `docs/core_concepts/veriphysics.md` | Verification and evidence (`pinneapple_veriphysics`) |
| `docs/core_concepts/viewers_and_twin_standards.md` | Other 3-D viewers and digital-twin standards |
| `docs/dev/ANALISE_FEATURES_COMERCIAIS.md` | O que as ferramentas comerciais e os outros CFDs têm que o PINNeAPPle ainda não tem |
| `docs/dev/AUDIT_REPORT.md` | Evidenced audit report |
| `docs/dev/CATALOGO_METODOS.md` | Catálogo de métodos do PINNeAPPle |
| `docs/dev/LATENT_AND_CYBER_EVALUATION.md` | Latent-space methods and cybersecurity applications: evaluation and roadmap |
| `docs/dev/LIBRARY_CONSOLIDATION.md` | Library consolidation: reusable logic outside the library |
| `docs/dev/MARKET_SCAN_LOVABLE_PHYSICS_AI_2026-09-13.md` | Pesquisa: Physics AI / Simulação / CAE / Digital Twin em lovable.app |
| `docs/dev/PEDIDOS_2026-09-24.md` | Pedidos e insights de 2026-09-24 — registro e decisões |
| `docs/dev/REA_INTEGRATION.md` | REA no PINNeAPPle — integração, fluxo de harvest e mapa do ecossistema |
| `docs/dev/ROADMAP_PHYSICS_AI_HUB.md` | PINNeAPPle → "the Hugging Face of Physics AI": full roadmap |
| `docs/dev/issues/001-geometric-similarity-score.md` |  |
| `docs/dev/issues/002-active-simulation-loop.md` |  |
| `docs/dev/issues/003-raw-geometry-field-surrogate.md` |  |
| `docs/dev/issues/004-rom-fmu-export.md` |  |
| `docs/dev/issues/005-pinneapple-mcp-server.md` |  |
| `docs/dev/issues/006-topology-optimization-simp.md` |  |
| `docs/dev/issues/007-solution-adaptive-amr.md` |  |
| `docs/dev/issues/008-multi-fidelity-cokriging.md` |  |
| `docs/dev/issues/009-hybrid-rom-online-calibration.md` |  |
| `docs/dev/issues/010-pce-sobol.md` |  |
| `docs/dev/issues/011-geometry-encoders.md` |  |
| `docs/dev/issues/012-lbm-d3q27-cumulant-voxel.md` |  |
| `docs/dev/issues/013-ifno-incremental-modes.md` |  |
| `docs/dev/issues/014-multiadam.md` |  |
| `docs/dev/issues/015-mapdl-rst-reader-pyfluent.md` |  |
| `docs/dev/issues/016-browser-pde-preview.md` |  |
| `docs/dev/issues/017-physicsnemo-missing-models.md` |  |
| `docs/dev/issues/018-rea-native-ghidra.md` |  |
| `docs/dev/issues/019-adjoint-mma-bspline.md` |  |
| `docs/dev/issues/020-design-study-object.md` |  |
| `docs/dev/issues/021-rbf-mesh-morphing.md` |  |
| `docs/dev/issues/022-inference-cache-viewer.md` |  |
| `docs/dev/issues/023-the-well-adapter.md` |  |
| `docs/dev/issues/024-procedural-pde-datasets.md` |  |
| `docs/dev/issues/025-neural-corrected-stepper.md` |  |
| `docs/dev/issues/026-differentiable-linear-solves.md` |  |
| `docs/dev/issues/027-nektar-bridge.md` |  |
| `docs/dev/issues/028-jaxfluids-bridge.md` |  |
| `docs/dev/issues/029-paraview-catalyst-insitu.md` |  |
| `docs/dev/issues/030-nn-in-weak-forms.md` |  |
| `docs/dev/issues/031-warp-backend.md` |  |
| `docs/dev/issues/032-rbf-kriging-surrogates-outliers.md` |  |
| `docs/dev/issues/033-onshape-configuration-sampler.md` |  |
| `docs/dev/issues/034-rescale-job-connector.md` |  |
| `docs/dev/issues/035-comsol-mph-bridge.md` |  |
| `docs/dev/model_dataset_inventory.md` | Model and dataset inventory for the zoo (2026-10-13) |
| `docs/dev/pr/feat-rea-bridge.md` | Summary |
| `docs/getting_started/first_pinn.md` | First PINN |
| `docs/getting_started/installation.md` | Installation |
| `docs/getting_started/project_layout.md` | Project Layout |
| `docs/getting_started/quickstart.md` | Quickstart |
| `docs/getting_started/start_here.md` | Start here (docs) |
| `docs/guides/adding_an_experiment.md` | Adding an experiment / Como adicionar um experimento |
| `docs/guides/agent_workbench.md` | Agent Workbench: coding and engineering agents on a local model |
| `docs/guides/antenna_design.md` | Antenna design: an end-fire Vivaldi with an external FEM solver |
| `docs/guides/autonomous_lab.md` | Autonomous Physics Lab: an AI that manages evidence for engineering decisions |
| `docs/guides/autonomous_physics_lab.md` | Autonomous Physics Lab |
| `docs/guides/campaign_planner.md` | Simulation Campaign Planner: which simulations need to run |
| `docs/guides/chain_of_whys.md` | Chain of whys: why each element of a design exists, computed |
| `docs/guides/check_power.md` | Check power: how often each check catches a known defect |
| `docs/guides/check_power_matrix.md` | Check power: the detection matrix |
| `docs/guides/choosing_a_method.md` | Choosing a method: PINN, neural operator, graph network or the solver |
| `docs/guides/colab_notebooks.md` | Colab notebooks |
| `docs/guides/compact_range.md` | Compact range quiet zone: feed gain × reflector size |
| `docs/guides/dataset_audit.md` | Dataset physics audit: does the data solve its PDE? |
| `docs/guides/design_from_intent.md` | Design from intent |
| `docs/guides/design_report.md` | Design reports without an LLM: `Datasheet` and the self-planning study |
| `docs/guides/design_studies.md` | Design studies: one problem, many optimisers, verified candidates |
| `docs/guides/electromagnetics.md` | Electromagnetics and electrodynamics |
| `docs/guides/engineering_design_optimizer.md` | Engineering Design Optimizer (app, day 17) |
| `docs/guides/engineering_gui_agent.md` | Engineering GUI agent: run engineering software and keep the evidence |
| `docs/guides/enso_forecasting.md` | Predicting El Niño: physical, physics-informed, data-driven and hybrid models under one protocol |
| `docs/guides/evidence_ledger.md` | Evidence ledger: a test passed only if it ran |
| `docs/guides/flight_dynamics.md` | Flight dynamics, turbulence and take-off |
| `docs/guides/flow_schema.md` | The flow schema (`pinneapple-flow/1`) and how another app reuses it |
| `docs/guides/fmi_compatibility.md` | FMI import compatibility |
| `docs/guides/football_physics.md` | Football: where physics helps on public tracking data |
| `docs/guides/freecad_openfoam_pipelines.md` | FreeCAD -> OpenFOAM pipelines |
| `docs/guides/geometry_library.md` | Geometry library |
| `docs/guides/identifiability_calibration_inversion.md` | Identifiability, calibration and inverse problems |
| `docs/guides/industrial_presets.md` | Industrial presets |
| `docs/guides/interlocking_bricks.md` | Self-modifying brick builds |
| `docs/guides/mcp_server.md` | PINNeAPPle from any coding agent: the MCP server |
| `docs/guides/metamorphic_tests.md` | Metamorphic tests: checking a surrogate without ground truth |
| `docs/guides/model_zoo.md` | Pre-trained models and datasets (`pp.zoo`, `pp.datasets`) |
| `docs/guides/near_random_forecasting.md` | Forecasting near-random series without overfitting |
| `docs/guides/observatory.md` | The PINNeAPPle Observatory |
| `docs/guides/physics_checks_in_ci.md` | Physics checks in CI |
| `docs/guides/physics_design_optimizer.md` | Physics Design Optimizer (app, day 23) |
| `docs/guides/physics_residual_attack_detection.md` | Detect false data injection with physics residuals |
| `docs/guides/physicsnemo_trust.md` | PhysicsNeMo trains, PINNeAPPle says when to trust |
| `docs/guides/pinn_validation.md` | PINN validation: what the training loss does not say |
| `docs/guides/pinneapple_llm_quality_pipeline.md` | Quality pipeline for LLM-generated physics and twin artifacts |
| `docs/guides/playground.md` | Playground: physics surrogates in the browser |
| `docs/guides/pp_validate.md` | pp.validate: one call, one validation report |
| `docs/guides/recreate_a_wear_twin.md` | Recreate a 3D wear twin (refractory, liner, pipe wall) |
| `docs/guides/residual_error_bound.md` | From residual to error bound: how wrong is this prediction, without the true answer? |
| `docs/guides/rollout_validation.md` | Rollout validation: when an autoregressive surrogate stops being trustworthy |
| `docs/guides/rom_and_meshless_studies.md` | Reduced-order models and meshless methods: measured, not assumed |
| `docs/guides/rover_learns_the_planet.md` | The rover that learns the planet |
| `docs/guides/running_a_benchmark.md` | Running a benchmark |
| `docs/guides/scale_aware_ensembles.md` | Choose a model by the scale of the question |
| `docs/guides/science.md` | Science by area (`pp.science`) |
| `docs/guides/sensitivity_lab.md` | Engineering Sensitivity Lab |
| `docs/guides/spectral_fidelity.md` | Spectral fidelity: what relative L2 does not see |
| `docs/guides/surrogates_arena_gateway.md` | Surrogates from simulation results: study, regime arena, gateway |
| `docs/guides/topology_optimization.md` | Topology optimisation |
| `docs/guides/training_callbacks.md` | Physics checks during training (Lightning, PhysicsNeMo, Hugging Face) |
| `docs/guides/trusted_latent_surrogate.md` | A latent surrogate that says when not to trust it |
| `docs/guides/uncertainty_and_sensitivity.md` | Uncertainty and sensitivity studies on any model |
| `docs/guides/uncertainty_lab.md` | Engineering Uncertainty Lab |
| `docs/guides/unit_cell_benchmark.md` | Periodic unit cell: an FNO over frequency × scan, and an equal-budget benchmark against Bayesian optimisation |
| `docs/guides/validation_arena.md` | Red team / blue team for the validation checks |
| `docs/guides/validation_gallery.md` | Validation gallery: one worked example per model family and per physics |
| `docs/guides/validation_methods_reference.md` | Validation methods reference |
| `docs/guides/workflow_runner.md` | Simulation Workflow Runner: campaigns as traceable, resumable workflows |
| `docs/guides/zoo_catalogue.md` | Zoo catalogue |
| `docs/index.md` | PINNeAPPle |
| `docs/lab/coverage.md` | Experiments coverage |
| `docs/philosophy.md` | Philosophy |
| `docs/research/condensation_falsification.md` | Falsification campaign for physics condensation — pre-registration |
| `docs/research/physics_condensation.md` | Adaptive Physics Condensation: geometry-independent thermal models of chip packages |
| `docs/security/audit.md` | Security audit: the PINNeAPPle web apps and their deployment |
| `docs/security/runbook.md` | Operator runbook: PINNeAPPle apps |
| `docs/security/server.md` | Server hardening: the host that runs the PINNeAPPle apps |
| `docs/strategy/STRATEGY_PHYSICS_AI.md` | Physics AI Strategy — structure and first implementation |
| `docs/validation_status.md` | Validation status |
| `docs/why.md` | Why PINNeAPPle? |
