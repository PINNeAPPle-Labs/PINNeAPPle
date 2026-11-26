- `pinneapple_data.registry`: **um registro só para os datasets públicos e os modelos pré-treinados** (#528), na
  frente do catálogo externo (`pinneapple_data.datasets.catalog`: licença, se ela foi verificada, citação, homepage,
  adaptador) e do zoo (`pinneapple_hub/zoo/manifest.json`: pesos, SHA-256, versão, repo do Hugging Face, model card).
  Nada é recopiado: cada campo é lido da fonte que é dona dele, então corrigir uma licença num lugar corrige o
  registro, a tabela de docs e o que fica gravado em cada run. `list(domain=, kind=, commercial_only=, readable=)`,
  `get`, `info`, `licence()` (o aviso mostrado **antes** do download, com a não comercial sinalizada e o acesso com
  cadastro barrado até `allow_gated=True`), `citation()`, `load(id, split=)` (imprime a licença antes e a citação
  depois; os checksums do zoo são os mesmos do manifesto) e `metadata()` -- o dicionário que `record(ctx, id)` grava
  no `card.json` do dataset do Lab, de onde a curadoria já lê licença e proveniência. Medido no repositório: **25
  datasets e 4 modelos**, 16 carregáveis pelo registro hoje (12 datasets e 4 modelos do zoo, verificados por SHA-256),
  13 entradas externas com licença e citação e 7 delas só por adaptador (caminho local) -- `load` levanta
  `NoReaderError` nomeando a chamada que funciona, em vez de fingir leitura; `airfrans` segue sem leitor e isso está
  escrito. `publish()` é declarado, não executado: devolve o plano (arquivos, checksums, carta, licença) e diz o que
  falta. Página gerada em `docs/core_concepts/datasets_and_models.md`; `pp.datasets.load` cai no registro quando o
  nome não é do zoo, então o erro passou a ser a licença e o adaptador em vez de `KeyError`.
