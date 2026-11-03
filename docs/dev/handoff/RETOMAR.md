# Retomar o trabalho em outra conta (VS Code no seu PC)

Atualizado em 03/11/2026. Este arquivo é o ponto de partida para continuar sem acesso às sessões em nuvem, às conversas
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

## 3. PRs abertos (nenhum mergeado; esperando revisão do dono)

| Repo | PR | Assunto | Estado em 03/11 |
|---|---|---|---|
| PINNeAPPle | #1245 | apps de outubro na biblioteca, lote 1 de 3 | verde |
| PINNeAPPle | #1255 | workflow base: gêmeo digital (#1193) | verde |
| PINNeAPPle | #1259 | rodar um ModelDraft do formulate como job (#1246) | conflito resolvido (5405dca9); CI rodando de novo |
| PINNeAPPle | #1260 | séries temporais (#1197) e clima extremo (#1198) | **conflito com a main** em arquivo gerado; dados aprovados pelo dono, com a fonte registrada em SOURCES.md. Para resolver: mesclar a main, usar o arquivo gerado da main e regenerar com a ferramenta |
| PINNeAPPle | #1261 | bio, projeto de chip, projeto de satélite (#1205–#1207) | não revisado |
| PINNeAPPle | #1262 | drones (#1201) | não revisado |
| pinneapple-engineering-api | #74 | colunas medidas como entradas (#61); ferramentas de ajuste (#43) | não revisado |
| PINNeAPPle-CFD | #141 | canal AV05/AV06, CI com OpenFOAM, UX | rascunho; rodadas em andamento |

Os mais antigos (#1031, #1033–#1035, #986, #731 e os do dependabot) estão parados desde outubro. O #731 é de
contribuidor externo e espera a decisão do dono: merge ou fechar.

## 4. Trabalho que estava só nas sessões

Em 03/11, 19:18 UTC, cada sessão recebeu a ordem de enviar tudo para o GitHub: branches `wip/<tema>` e um arquivo
`docs/dev/handoff/wip-<tema>.md` explicando o que falta. Para achar:

```bash
git fetch origin && git branch -r | grep -E 'wip/|claude/'
git log origin/<branch> -1 -- docs/dev/handoff/
```

| Tema | Onde procurar |
|---|---|
| apps de outubro, lote 2 (#1226) | `wip/*` sobre #1245 |
| #1194 CFD/OpenFOAM e #1195 estrutural | `wip/*` sobre #1255 |
| braços robóticos (#1202), submarinos (#1203), terramecânica e rover (#1199, #1200) | `wip/*` ou branch do #1262 |
| imagens de satélite (#1196) e EM (#1204) | `wip/*` ou branch do #1260 |
| palavras-chave do planner e galeria industrial (#1244) | `wip/*` ou branch do #1259 |
| backlog do app de engenharia (#37, #50, #34, #36, #33) | `wip/*` no `pinneapple-engineering-api` |
| campanha do canal (D0, AV02–AV06) | PINNeAPPle-CFD, PR #141 e `docs/HANDOFF-campanha-anteparos.md` |

Se algum desses branches não aparecer, a sessão não conseguiu enviar a tempo. Nesse caso, o trabalho tem que ser
refeito a partir da issue.

## 5. Issues para continuar

- Registro da pausa: #1286, com cada PR, a ordem dos merges e os passos para retomar.
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
