# Campanha de contribuições em código aberto de terceiros

Documento de retomada, escrito em 2026-10-09. Complementa o catálogo `OSS_REPOS.md` / `oss_repos.csv` desta pasta.

## O que foi pedido

1. Levantar repositórios open source de ciência, tecnologia e pesquisa, inclusive de empresas e agências grandes (NVIDIA, PhysicsX, Luminary Cloud, NASA, NOAA, CIA etc.), e documentar a lista localmente.
2. Procurar issues abertas neles e resolvê-las, **checando as normas de contribuição de cada repositório**.
3. Regras do dono: commits, PRs e comentários **em inglês, em nome de `barrosyan` (Yan Barros), sem nenhuma referência a Claude ou IA**; apagar clones que não forem mais usados para não encher o computador; ritmo leve (o dono reclamou de uso de CPU); abrir PRs autorizado a seguir sem confirmar cada um, **mas parar e avisar** quando o repositório exigir CLA, DCO ou declaração de uso de IA.

## Catálogo (fase 1, concluída)

- 11.393 repositórios públicos em 58 organizações; 2.793 candidatos (não fork, não arquivado, push em 12 meses, licença, issues abertas) com sinais de "contribuível" (good first issue, help wanted, CONTRIBUTING, último merge).
- **PhysicsX** (empresa) não tem organização pública no GitHub (a conta `PhysicsX` é de outra pessoa). **CIA** não tem organização oficial com código (`cia-foundation` é projeto de TempleOS, sem relação). **Luminary Cloud** tem só 3 repositórios.
- Arquivos: `OSS_REPOS.md` (legível), `oss_repos.csv` (completo). Para regenerar, use `tools/` (ver abaixo).

## PRs enviados (estado em 2026-10-09)

| Repositório | PR | Issue | O que faz | Estado |
|---|---|---|---|---|
| NOAA-OWP/nwm-post-processing | [#179](https://github.com/NOAA-OWP/nwm-post-processing/pull/179) | #90 | erro de configuração informa o arquivo de origem | **mesclado** |
| sandialabs/WecOptTool | [#463](https://github.com/sandialabs/WecOptTool/pull/463) | #314 | documentação do `nsubsteps` (PR vai para a branch `dev`) | aberto |
| NCAR/DART | [#1195](https://github.com/NCAR/DART/pull/1195) | parte da #808 | typos e exemplos errados na doc do `obs_common_subset` | aberto |
| NOAA-OWP/DMOD | [#731](https://github.com/NOAA-OWP/DMOD/pull/731) | parte da #711 | testes assíncronos do dataservice para `IsolatedAsyncioTestCase` | aberto |
| NOAA-OWP/DMOD | [#732](https://github.com/NOAA-OWP/DMOD/pull/732) | parte da #711 | idem em externalrequests e `test_scheduler_client` | aberto |
| NOAA-OWP/hypy | [#40](https://github.com/NOAA-OWP/hypy/pull/40) | #33 | move `python/hypy/test` para `python/test` | aberto |
| NOAA-OWP/ras2fim | [#334](https://github.com/NOAA-OWP/ras2fim/pull/334) | #327 | `code_version` em `run_arguments.txt` (+ CHANGELOG) | aberto |
| sandialabs/sdynpy | [#29](https://github.com/sandialabs/sdynpy/pull/29) | itens 2 e 3 da #27 | módulo de cisalhamento `1+ν` e constante de torção do retângulo | aberto |
| nasa-jpl/bespokebpv7 | [#65](https://github.com/nasa-jpl/bespokebpv7/pull/65) | #52 | typos no README de exemplos | aberto |
| nasa-jpl/bespokebpv7 | [#66](https://github.com/nasa-jpl/bespokebpv7/pull/66) | itens 2 e 4 da #45 | listas de EID copiadas em vez de referenciadas | aberto |

Nenhum dos abertos tinha recebido revisão em 2026-10-09. O NOAA-OWP respondeu em poucas horas no #179.

**Primeira coisa ao retomar:** ver o estado de todos e responder a pedidos de ajuste:

```bash
for x in sandialabs/WecOptTool#463 NCAR/DART#1195 NOAA-OWP/DMOD#731 NOAA-OWP/DMOD#732 NOAA-OWP/hypy#40 \
         NOAA-OWP/ras2fim#334 sandialabs/sdynpy#29 nasa-jpl/bespokebpv7#65 nasa-jpl/bespokebpv7#66; do
  r=${x%#*}; n=${x#*#}
  gh pr view $n -R $r --json state,reviewDecision,comments,reviews --jq '"'"$x"': \(.state) comments=\(.comments|length) reviews=\(.reviews|length)"'
done
```

Para ajustar um PR: clonar `--depth 1` a branch do fork `barrosyan/<repo>` (nomes abaixo), editar, `git push`. Quando um PR for mesclado ou fechado, **apagar o clone** correspondente.

## Clones locais (nesta pasta)

| Pasta | Branch do PR | Observação |
|---|---|---|
| `DMOD/` | `test/711-async-tests-dataservice`, `test/711-async-tests-externalrequests-communication` | repositório padrão `master`; mais fatias possíveis (ver abaixo) |
| `hypy/` | `move-tests-out-of-package` | |
| `ras2fim/` | `dev-add-code-version-to-run-arguments` | PRs vão para a branch `dev` |
| `bespokebpv7/` | `fix/eid-setters-copy-lists` (a do #65 é `docs/fix-examples-readme-typos`) | |
| (apagados) | WecOptTool `docs/314-explain-nsubsteps`, DART `docs/808-obs-common-subset-fixes`, sdynpy `fix/rect-beam-shear-modulus-and-torsion-constant` | já estão no fork; reclonar só se pedirem ajuste |

Todos os clones têm `upstream` = repositório original e `origin` = fork `barrosyan/<repo>`; `git config user.name "Yan Barros"` e `user.email yanbarrosyan@gmail.com` já definidos em cada um.

## Regras específicas de cada repositório (descobertas)

- **NOAA-OWP/\*** (DMOD, hypy, ras2fim, nwm-post-processing): CONTRIBUTING simples; contribuições viram domínio público pela submissão. **Comentar na issue antes** ("I would like to take this"). ras2fim: PR para a branch `dev`, título `[Npt] PR: ...`, branch `dev-...`, entrada no `doc/CHANGELOG.md` com o número do PR (segundo commit), `black` com linha de 110. nwm-post-processing precisa de `brew install nco` para rodar os testes.
- **sandialabs/WecOptTool**: PR para a branch `dev`; título começa com `DOCUMENTATION:`/`BUG FIX:` etc.
- **NCAR/DART**: Guia em `docs.dart.ucar.edu/en/latest/guide/contributors-guide.html`; PRs pequenos e focados; template de PR com checklist.
- **nasa-jpl/bespokebpv7**: BSD-3, sem CLA; tem `AGENTS.md` (usam assistentes). `ruff` com `select = ALL`, `mypy` estrito: **colocar testes em arquivo existente** (os arquivos têm cabeçalho de copyright do Caltech; não criar arquivo novo com esse aviso). Rodar `ruff check`, `ruff format --check` e `mypy` antes do PR.
- **sandialabs/sdynpy**: sem guia de contribuição; testes em `tests/*_test.py` (pytest).

## Pular e avisar (exigem compromisso legal em nome do dono)

Nunca marcar "No" em declarações de uso de LLM, nunca aceitar CLA em nome do dono.

- **idaholab/MontePy**: template de PR exige declarar modelo/harness/esforço de LLM. **O dono decidiu abandonar esse repositório.**
- **ECMWF/anemoi-\***: abrir o PR afirma o Contributor License Agreement.
- **Unidata/MetPy** (e provavelmente siphon): CLA via bot.
- **sandialabs/OpenCSP**: contribuidor declara autorização do empregador.
- **NOAA-GFDL/\*** (`pace`, `fre-cli`): exigem humano responsável identificado para trabalho com IA e contato prévio.
- Outros com política de IA/CLA detectada: llnl/Surfactant, ansys/pyaedt e pyspeos, NVIDIA/cuml, google-deepmind/concordia, NVIDIA (CLAs).
- **nasa-jpl/tos2ca-anomaly-detection**: a manutenção ativa terminou em maio de 2026; issues são de design.

## Descartados e por quê (não refazer a triagem)

- NOAA-OWP/inundation-mapping (Docker/DevOps/changelog versionado; #1011 obsoleta), NOAA-OWP/t-route (C/Fortran/Cython, issues de 2021), DMOD #712 (decisão de arquitetura).
- precice/aste (C++; precisa de preCICE e VTK para compilar), esa/torchquad (CI exige 4 backends, só 2 rodam aqui), sandialabs/PEAT (#17: 1.237 achados de segurança; #28: exige dispositivos reais), SciML (Julia não instalado).
- nasa/OnAIR (issues perguntam "o que deveria acontecer?"), nasa/python_cmr#49 (alguém já abriu PR), ACCESS-NRI/meorg_client#68 (outra pessoa interessada).

## Próximas fatias possíveis

- **DMOD #711** (continuação): `python/lib/communication/dmod/test/test_websocket_interface.py` e `test_decorated_interface.py` falham na linha de base porque procuram o diretório `ssl` com certificados do projeto; só dá para validar com esse diretório montado. `python/services/requestservice/dmod/test/`.
- **sdynpy #27 item 1**: aguardar decisão dos mantenedores (se `ei1` é "em torno do eixo 1" ou "deflexão ao longo da direção 1"). Se decidirem corrigir a docstring, é um PR pequeno.
- **bespokebpv7 #45**: itens 1, 3, 5 e 6 ainda abertos (validação pós-unpack, tipagem de `dest_seq`, descrição do `pyproject.toml`, exportações).
- **NCAR/DART #808**: itens de código (mensagem de erro sobre tamanhos de ensemble diferentes) ficaram para a equipe do DART.

## Receita de ambiente (os ambientes virtuais foram apagados; recriar quando precisar)

Há Python 3.12 e `uv` no computador. Por repositório, em um diretório temporário:

```bash
uv venv --python 3.12 <venv> && uv pip install --python <venv>/bin/python -e . pytest
# DMOD: instalar sem dependências os pacotes locais (core communication scheduler modeldata redis access externalrequests),
#   mais Deprecated Faker aiohttp~=3.8 cryptography docker geopandas gitpython jsonschema minio pandas pydantic<2 pyogrio pyyaml redis shapely uri websockets python-dotenv,
#   e do GitHub: ngen-config, ngen-config-gen (noaa-owp/ngen-cal, subdiretórios python/ngen_conf e python/ngen_config_gen) e hypy (noaa-owp/hypy, subdiretório python).
# hypy: pandas hydrotools.nwis-client pytest (o teste test_nwislocation.py usa o serviço USGS ao vivo e é instável)
# bespokebpv7: uv pip install -e . pytest hypothesis ruff mypy
# ras2fim: não roda (Windows + HEC-RAS); a função alterada foi validada extraindo-a com ast e usando keepachangelog
```

## Descoberta de novos alvos (scripts em `tools/`)

Rodar em uma pasta de trabalho (os scripts gravam JSON no diretório atual); requer `gh` autenticado como `barrosyan`.

- `fetch.py` lista repositórios das organizações; `enrich.py` junta GraphQL (good first issue, help wanted, CONTRIBUTING, último merge); `build_doc.py` gera `OSS_REPOS.md` e `oss_repos.csv` (caminho de saída fixo nesta pasta).
- `gfi.py` coleta issues com `good first issue`/`help wanted` sem responsável; `screen2.py` procura CLA, DCO e política de IA em CONTRIBUTING, template de PR e `AGENTS.md`/copilot-instructions.
- Busca rápida de pequenos ajustes: `gh search issues "typo" --owner <org> --state open --no-assignee --match title` (trocar por "broken link", "docstring", "spelling").
- Limite de busca do GitHub: ~30 chamadas por minuto; esperar se aparecer "secondary rate limit".

## Fluxo por PR

1. Ler CONTRIBUTING, template de PR, `AGENTS.md`/política de IA e CLA. Se houver compromisso legal ou declaração de IA, **parar e avisar**.
2. Comentar na issue que vai assumir (o que for pedido pelo projeto).
3. Fork (`gh repo fork --clone=false`), clone raso, branch no padrão do projeto.
4. Mudança pequena com teste; **mostrar que o teste falha no código antigo** (`git show upstream/<branch>:<arquivo>`; `git stash` não serve depois de commitar).
5. Rodar o lint/format/tipos do projeto; o CI do projeto decide o resto.
6. Commit em inglês, sem menção a IA, sem frases de "I agree to the waiver" além do que o projeto pede (a submissão já vale).
7. Corpo do PR com: o que muda, como foi testado, o que NÃO foi feito e limitações. Usar `Closes #N` só se resolver tudo.
8. Registrar aqui e na memória; apagar o clone quando o PR fechar.
