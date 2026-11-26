- Portão de dependências (#96): o job **Dependency floors** rodava em Python 3.10 — o piso declarado, que é onde um
  piso sofre mais pressão — e morria na primeira linha de `scripts/licensing/supply_chain.py`, porque `tomllib` só
  existe de 3.11 em diante. Resultado: o portão nunca auditou nada, em nenhum pull request que tocasse `pyproject.toml`,
  e o vermelho lia como dependência quebrada em vez de importador quebrado. O script passa a cair em `tomli` no
  3.10, como `check_dist.py` e `dependency_licenses.py` já faziam, e o step instala `tomli` e `packaging` antes de
  auditar. Verificado nos dois caminhos: no 3.11 normal e com `tomllib` bloqueado, caindo no `tomli` (mesma saída,
  mesmos 102 dependências declaradas, 97 com licença em registro).
