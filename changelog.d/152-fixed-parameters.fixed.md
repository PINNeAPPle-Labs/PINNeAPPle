- Fábrica de simulação, condição da issue #152: os parâmetros que o caso **fixa** (`Base.fixed` do `DesignStudy`)
  passam a chegar à seta CAD. Antes, `run_factory` entregava à geometria só as variáveis do projeto, e um caso que
  declara `fixed={"inlet_length": 0.5, "outlet_length": 1.0}` — os trechos de 10 D e 20 D que a linha do caso de
  erosão em curva precisa — recebia em silêncio os padrões da template (0,762 m e 0,381 m, ou seja, 15,2 D e 7,6 D em
  D = 0,05 m). O parâmetro fixo agora entra na geometria (e no hash de procedência da linha), continua fora das
  colunas do dataset e fora das entradas do solver, que segue recebendo as variáveis mais as condições do avaliador; um
  parâmetro fixo que a template não conhece recusa a linha com o motivo em vez de ser descartado. Efeito colateral
  coerente: linha recusada pela geometria passa a sair `rejected` com a razão, em vez de `kept` sem razão. Medido em
  `tests/test_autolab_factory.py` (3 testes: comprimento derivado, parâmetro desconhecido, hash de procedência).
