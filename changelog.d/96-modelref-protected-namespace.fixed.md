- Contratos de dados (#96): `ModelRef.model_type` colidia com o *protected namespace* `model_` do pydantic, e no piso
  declarado (`pydantic==2.0`, que é o que o job de pisos instala) isso não é aviso: é `NameError` na criação da classe,
  ou seja, **o pacote não importava no piso**. Passou despercebido porque o portão morria antes de chegar lá
  (`tomllib`, no mesmo job). O `_Model` base passa a declarar `protected_namespaces=()`, com o motivo escrito ao lado:
  `model_type` é um campo de domínio, e `extra="forbid"` continua recusando o que não foi declarado. Verificado no 3.11
  local: `tests/pinneapple_data/test_contracts.py` e `tests/test_deprecation_and_api.py` verdes.
