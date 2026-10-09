# Contribuições em código aberto de terceiros: tudo o que é preciso para retomar

Esta pasta guarda **tudo** da campanha de 2026-10-09, para que nada dependa de arquivos locais.

| Arquivo / pasta | Conteúdo |
|---|---|
| `OSS_CAMPAIGN.md` | **Comece por aqui.** Estado dos PRs, regras de cada repositório, lista de pular/descartados, receitas de ambiente, fluxo por PR |
| `OSS_REPOS.md`, `oss_repos.csv` | Catálogo: 2.793 repositórios candidatos de 58 organizações (NVIDIA, NASA, NOAA, laboratórios do DOE, ESA, ECMWF, comunidades científicas...) |
| `tools/` | Scripts que geraram o catálogo e acham novos alvos (`fetch.py`, `enrich.py`, `build_doc.py`, `gfi.py`, `screen.py`, `screen2.py`) |
| `dados/` | Dados intermediários: `all_repos.json.gz` (11.393 repositórios), `candidates.json.gz` (2.793 com sinais de contribuição), `gfi_issues.json` (664 issues "good first issue/help wanted" sem responsável), `screen2.json` |
| `memoria/` | Cópia das notas de memória do assistente (campanha externa, progresso da série X, regra de não mencionar IA) |

## Restaurar e usar

```bash
cd PINNeAPPle/docs/dev/oss_contribuicoes
mkdir -p /tmp/oss && cd /tmp/oss
gunzip -c <repo>/docs/dev/oss_contribuicoes/dados/all_repos.json.gz > all_repos.json
gunzip -c <repo>/docs/dev/oss_contribuicoes/dados/candidates.json.gz > candidates.json
cp <repo>/docs/dev/oss_contribuicoes/dados/gfi_issues.json .
python3 <repo>/docs/dev/oss_contribuicoes/tools/screen2.py      # triagem de CLA/DCO/política de IA (usa candidates.json)
OSS_OUT=. python3 <repo>/docs/dev/oss_contribuicoes/tools/build_doc.py   # regenera OSS_REPOS.md e oss_repos.csv
```

Para refazer a coleta do zero: `tools/fetch.py` (lista as organizações), depois `tools/enrich.py` (GraphQL; ~30 minutos) e `tools/gfi.py` (busca de issues; respeitar o limite de ~30 chamadas por minuto). Precisa de `gh` autenticado como `barrosyan`.

## Restaurar a memória do assistente (outra sessão ou máquina)

Copiar os arquivos de `memoria/` para `~/.claude/projects/-Users-yanbarros-Documents-GitHub-pinneapple-labs/memory/` e acrescentar ao `MEMORY.md` de lá uma linha por arquivo (`- [Título](arquivo.md) — resumo`).

## Onde está o resto

- PRs abertos: tabela em `OSS_CAMPAIGN.md` (links diretos). As branches dos PRs vivem nos forks `github.com/barrosyan/<repo>`.
- Campanha de issues do próprio PINNeAPPle: `../HANDOFF_CAMPANHA_ISSUES.md`. Trabalho em andamento do X14 (`pp.distributed`): branch `feat/x14-distributed` deste repositório.
