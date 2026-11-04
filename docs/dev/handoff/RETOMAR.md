# Retomar o trabalho em outra conta (VS Code no seu PC)

Atualizado em 04/11/2026. Este arquivo é o ponto de partida para continuar sem acesso às sessões em nuvem, às conversas
ou aos contêineres. Tudo o que importa está no GitHub: nos repositórios, nos PRs abertos, nos branches `wip/*` e nas
issues listadas aqui.

## 1. Preparar o PC (uma vez)

1. Instale o VS Code, a extensão do Claude Code, git, Python 3.11 e, se quiser rodar CFD, o OpenFOAM (Linux ou WSL2).
2. Entre na outra conta do Claude, e conecte o GitHub com acesso à organização `PINNeAPPle-Labs`. Os repositórios
   privados exigem isso.
3. Clone os repositórios lado a lado:

   ```bash
   mkdir -p ~/pinneapple && cd ~/pinneapple
   git clone https://github.com/PINNeAPPle-Labs/PINNeAPPle
   git clone https://github.com/PINNeAPPle-Labs/pinneapple-engineering-api
   git clone https://github.com/PINNeAPPle-Labs/PINNeAPPle-CFD
   git clone https://github.com/PINNeAPPle-Labs/PINNeAPPle-apps
   cd PINNeAPPle && python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
   ```

4. Para rodar o app de engenharia, siga o `docs/RODAR_LOCALMENTE.md` do `pinneapple-engineering-api`.
5. Abra a pasta `PINNeAPPle` no VS Code. O Claude Code lê o `CLAUDE.md`, que aponta para o `AGENTS.md`. Na primeira
   mensagem, peça: "Leia docs/dev/handoff/RETOMAR.md e siga as regras dele".

## 2. Regras de trabalho (para o Claude seguir)

- Responder ao dono em português, com checklist do estado das tarefas em cada atualização.
- Criar a issue no GitHub antes de começar a resolver qualquer coisa.
- Nunca apresentar número que não foi medido. Dizer o que não rodou, por exemplo "não rodado: sem OpenFOAM aqui".
- Dado de cliente nunca vai para repositório público. Isso inclui nomes, geometria e resultados. O `PINNeAPPle` é público;
  `PINNeAPPle-CFD` e `pinneapple-engineering-api` são privados.
- Nunca enfraquecer um gate de CI, por exemplo o orçamento de 160 000 B por página. Nunca pular nem desativar teste
  para ficar verde.
- Features moram na biblioteca (`PINNeAPPle`); os apps só chamam a biblioteca.
- Arquivos gerados (tool_runs, flow_runs e snippets do Hub) se regeneram com a ferramenta do repositório, nunca à mão.
  Ver "Generated files" no `AGENTS.md`.
- Rodar `ruff` só nos arquivos alterados. Nunca usar `ruff --fix` num diretório.
- Matar processos pelo PID, nunca com `pkill -f`.
- Sem force push. Sem ID de modelo em commits ou PRs.
- Não continuar o protótipo de defesa antimíssil (#1153–#1158).
- **Merge** (o dono autorizou os merges em geral, mas pediu para revisar ele mesmo #1259 e #1260 antes):
  1. CI verde.
  2. `git merge-tree --write-tree origin/main <branch>` sem conflito.
  3. Revisão do diff e de segurança.
  4. Corpo do PR reescrito sem o rodapé "Generated with Claude Code".
  5. Squash merge com o SHA completo (40 caracteres) do head.
  6. Depois do merge, validar a main: `python -m pinneapple_hub check` e a cobertura do Studio.

## 3. PRs (estado em 04/11/2026)

**Mergeados em 04/11:** #1245, #1259, #1262 na biblioteca; #74 no app de engenharia. #1261, #1255 e
PINNeAPPle-CFD#141 entram quando a CI terminar; confira na página de PRs.

**Aberto para você terminar: #1260** (séries temporais #1197 e clima extremo #1198). CI verde em 19cf2813, mas
conflita com a `main` em arquivos gerados. Os dados (séries de Coles 2001 / `ismev`) foram aprovados pelo dono, com a
fonte em `pinneapple_analysis/extremes/data/SOURCES.md`. Passo a passo:

```bash
git fetch origin && git checkout -B fix1260 origin/claude/workflows-data-climate-em
git merge origin/main
# pinneapple_flow/nodes_ext/__init__.py: manter os dois lados dos imports, em ordem alfabética
# pinneapple_studio/tool_reasons.json: união das chaves dos dois lados (ordenadas)
git checkout origin/main -- data/studio/tool_runs/pinneapple_flow.json data/web/hub/flow_runs.json
python -m pinneapple_studio smoke --package pinneapple_flow
python -m pinneapple_hub.library flowruns extreme_climate time_series_prediction
python -m pinneapple_hub.library run --stale
python -m pinneapple_hub.library check            # tem que dizer "index ok"
python -m pinneapple_studio coverage | grep -c "problem:"   # tem que dar 0
pytest tests/test_time_series_workflow.py tests/test_extreme_climate_workflow.py tests/test_hub_library.py -q
git add -A data pinneapple_flow pinneapple_studio && git commit -m "Merge main; regenerate the generated data"
git push origin fix1260:claude/workflows-data-climate-em
```

(No zsh, cole sem as linhas que começam com `#`.) Depois a CI (~35 min) e o merge com squash.

Os mais antigos (#1031, #1033–#1035, #986, #731 e os do dependabot) estão parados desde outubro. O #731 é de
contribuidor externo e espera a decisão do dono: merge ou fechar.

## 4. O que se perdeu e tem de ser refeito a partir das issues

As sessões paralelas atingiram o limite semanal antes de enviar o trabalho local e foram arquivadas. Nenhum branch
`wip/*` chegou ao GitHub. Refazer a partir das issues, que continuam abertas com a especificação completa:

- #1226, lote 2 dos apps de outubro;
- #1194 (CFD/OpenFOAM) e #1195 (estrutural);
- #1202 e #1203 (braços robóticos, submarinos);
- #1196 (imagens de satélite) e #1204 (EM);
- no app de engenharia, #37 (e o backlog #34, #36, #33).

#1199 (terramecânica) e #1200 (rover) não se perderam: entraram no #1256, em 03/11.

A lista completa e conferida do que ficou aberto, com o passo a passo do physicsnemo, está em
`docs/pendencias/PENDENCIAS_YAN.md`, no branch
[`pendencias/yan`](https://github.com/PINNeAPPle-Labs/PINNeAPPle-CFD/tree/pendencias/yan) do PINNeAPPle-CFD (privado).

**Campanha do canal (PINNeAPPle-CFD):** D0 e AV02–AV05 prontos, com o relatório unificado, as animações e os CSVs no
PR #141. O AV06 foi parado e fica para o seu PC: passo a passo em PINNeAPPle-CFD#144
(`bash estudos/canal_jato_doe/ferramentas/refazer_rodada.sh AV06`).

## 5. Issues para continuar

- Registro da pausa: #1286, com cada PR, o que se perdeu e os passos para retomar.
- Animação 3D de campo sobre o contorno do CAD, na biblioteca: #1338.
- Pendência da biblioteca: #1287 (`verify.converged` em `sensitivity_step` e `uncertainty_step`).
- Workflows base: épico #1185 (#1186–#1207).
- Camada de dados de engenharia: épico #1210 (#1211–#1221). Implementação futura.
- Agentes especializados pré-definidos: épico #1247 (#1248–#1254). Só planejamento; não executar sem o dono pedir.
- Lacunas de mercado: épico #1263 (#1264–#1285). Só planejamento. Os pequenos acréscimos combinados para #1194,
  #435/#691, #114 e #80 estão descritos no #1263 e no #1286.
- Auditoria dos apps de outubro: épico #1226 (#1227–#1240).

## 6. O que só o dono pode fazer

A lista completa, que cita clientes, está no repositório privado: `PINNeAPPle-CFD`, branch `pendencias/yan`, arquivo
`docs/pendencias/PENDENCIAS_YAN.md`. Ela traz servidor, deploys, chaves, e-mails e decisões pendentes.

## 7. Histórico

- `docs/dev/handoff/session-2026-11-03.md`: o que foi feito e mergeado em 03/11.
- Os PRs mergeados, com as descrições, ficam no GitHub.
