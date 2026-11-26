- Importação em Python 3.10 (#96): o piso declarado é 3.10 e `tomllib` é do 3.11 — três módulos importavam `tomllib` sem
  alternativa e **não carregavam no piso**: `pinneapple_studio/inventory.py` e `pinneapple_studio/requirements.py` (no
  topo do módulo, então o pacote inteiro caía), e `scripts/licensing/dependency_licenses.py`. Os três passam a cair em
  `tomli` — que já é dependência declarada para `python_version < '3.11'` — com o mesmo padrão que `pinneapple_check/config.py`
  e `scripts/check_dist.py` já usavam. Apareceu quando o job de pisos, consertado no fragmento anterior, conseguiu
  chegar à suíte rápida pela primeira vez.
