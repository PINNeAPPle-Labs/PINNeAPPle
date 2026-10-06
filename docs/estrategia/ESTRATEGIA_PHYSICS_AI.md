# Estratégia Physics AI — estrutura e primeira implementação

> **Status:** rascunho estruturado a partir da pesquisa do dono do projeto (2026-10-06). Os fatos de mercado citados na pesquisa (Siemens, NVIDIA, PhysicsX, Ansys, Altair, Cadence, Luminary, Dassault) **não foram verificados por este documento**. Tratar como hipótese de trabalho até conferência com fonte primária.
>
> **Limite do material recebido:** o texto da pesquisa chegou truncado (limite de 50 mil caracteres). As decisões finais que o dono pretende tomar não estão no trecho recebido; a seção 3 lista só as decisões que aparecem explicitamente no texto. Completar quando o restante estiver disponível.

---

## 1. Tese

Physics AI está deixando de ser uma pergunta de arquitetura ("qual rede resolve melhor uma PDE?") e virando infraestrutura de engenharia computacional. O valor comercial migra de "o modelo" para "o fluxo acelerado": explorar mais projetos, prever em segundos o que levava horas, operar um gêmeo digital, executar o workflow de engenharia.

Cadeia de evolução proposta pela pesquisa:

```
Solver → Surrogate → Physics AI → Design AI → Digital Twin → Engineering Agent → Autonomous Engineering
```

Posicionamento proposto: **camada aberta, vendor-neutral, de infraestrutura entre o modelo físico e a aplicação de engenharia.** O PINNeAPPle não substitui o solver (OpenFOAM, FEniCS, Ansys, Siemens); fica acima deles.

## 2. Camadas e onde o repositório está hoje

Verificado no código de `main` (2026-10-06) por inspeção de pastas e roadmap. "Existe" significa que há módulo e, quando indicado, teste. Não significa que está validado em produção.

| # | Prioridade (pesquisa) | Módulos existentes | Lacuna principal |
|---|---|---|---|
| 1 | Physics AI Core: abstração universal de `PhysicsModel` (PINN, operador, GNN, ROM, híbrido sob a mesma interface) | `pinneapple_neural`, `pinneapple_physics`, `pinneapple_models` | Não há uma interface única: o usuário escolhe a arquitetura. |
| 2 | Physics AI Compiler: `Problem → plano de pipeline` | `pinneapple_problemdesign`, `pinneapple_llm` (guardrail) | Compila problemas, mas não gera plano de dados, validação e deploy. |
| 3 | Surrogate Factory: CAD/simulação → dataset → modelo → validação → deploy | `ROADMAP.md` tem a seção "Surrogate Factory" e "Model Zoo"; `pinneapple_train`, `pinneapple_data` | Fluxo ponta a ponta não é uma abstração única ainda. |
| 4 | Physics Trust: validação + UQ + OOD + decisão de deploy | `pinneapple_analysis/trust/trust_gate.py` (OOD, resíduo, ensemble por predição); `pinneapple_analysis/verification/evidence_graph.py`; `pinneapple_hub/model_card.py` | **Decisão de deploy** juntando as checagens: **implementado nesta PR** (ver seção 4). |
| 5 | Physical Data Graph: linhagem problema → geometria → simulação → dataset → modelo → predição | `pinneapple_registry` (`problem_store`, `dataset_store`, `model_store`, `experiment_store`), `pinneapple_pdb` | Os registros existem separados; não há grafo de linhagem que os una. |
| 6 | PhysicsOps: registro, experimentos, deploy, monitoramento, drift | `pinneapple_registry`, `pinneapple_hub`, `pinneapple_orchestration` | Monitoramento de drift em produção não aparece como módulo próprio. |
| 7 | Engineering Agents: agente que executa o workflow via APIs | `pinneapple_llm`, `pinneapple_tools`, `pinneapple_problemdesign` | Agente que dispara o workflow completo (geometria → simulação → modelo → relatório) não existe. |
| 8 | Physics Foundation Models | `pinneapple_worldmodel` (descrito como generalista) | Sem demonstração multi-domínio nem comparação contra baselines. Programa de pesquisa, não prioridade imediata. |

Além das oito, a pesquisa destaca **Geometric AI** (CAD, malha, nuvem de pontos, SDF → modelo) e **Multi-fidelidade** (muitos dados baratos + poucos caros). Existem partes (`GNN`, `SDF`, `CSG` no repositório), mas não uma camada de geometria de primeira classe.

## 3. Decisões explícitas no texto recebido

1. O produto é o **workflow acelerado**, não a rede neural. Comunicação deve falar de decisões exploradas por unidade de tempo e custo, não de erro percentual do modelo.
2. Não competir como mais um solver nem mais uma plataforma CAE: ficar **acima** dos solvers e **vendor-neutral**.
3. **Validação e confiança são produto**, não detalhe: "se você não consegue validar, não deve implantar".
4. A arquitetura alvo é: Geometria + Dados + Conhecimento → Simulação → Modelos de física → Validação/UQ → Otimização → Gêmeo digital → Agente.
5. Mudar o posicionamento de "framework de PINNs/FNO/DeepONet" para "transforma simulações caras em modelos rápidos para exploração, otimização e operação".

## 4. Primeira implementação: relatório de confiança para deploy

Implementado em `pinneapple_analysis/trust/trust_report.py` (testes em `tests/test_trust_report.py`):

- `Check`: uma checagem com status `supports` / `contradicts` / `not_run` (mesmo vocabulário das provas do E11 do PINNeAPPle-CFD), criticidade e motivo.
- `TrustReport`: junta as checagens e devolve **APPROVED, REVIEW ou REJECT**, com score, cobertura, motivos e relatório em Markdown.
- `from_trust_score`: converte o `TrustScore` do `TrustGate` (OOD, resíduo PDE, ensemble) em checagens, reaproveitando o que já existe.

Regras de decisão (escolhas de projeto, revogáveis):
- falha em checagem **crítica** → REJECT;
- qualquer outra falha → REVIEW;
- cobertura abaixo de 60% → REVIEW;
- score (média ponderada das checagens executadas) abaixo de 0,80 → REVIEW;
- caso contrário → APPROVED.

**Pendências deste módulo:**
- os limites (0,8 de score, 60% de cobertura, 0,7 dos sub-scores) são provisórios e não foram calibrados com dados;
- ainda não há checagem de **conservação** ou de **condições de contorno** automática; hoje entram como booleanas fornecidas por quem chama;
- o relatório ainda não é gravado no `ModelCard` nem no registro (ligar ao `pinneapple_hub` e ao `pinneapple_registry`).

## 5. Próximos passos sugeridos (ordem)

1. Ligar o `TrustReport` ao `ModelCard` e ao registro: cada modelo publicado carrega sua decisão e seus motivos.
2. Checagens automáticas de conservação e de condições de contorno, a partir de `pinneapple_physics`.
3. Calibrar os limites com um conjunto de modelos reais (o PINNeAPPle-CFD tem surrogates com erro medido de 0,23% a 10%).
4. Physical Data Graph: unir `problem → geometry → simulation → dataset → model → validation` no registro existente.
5. Surrogate Factory como abstração única sobre os módulos já existentes.
6. Engineering Agent sobre as APIs do próprio PINNeAPPle, depois que 1–5 estiverem prontos.

Fora de ordem, por ora: Physics Foundation Models e world models (programa de pesquisa de longo prazo).
