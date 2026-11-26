Três portões da `main` estavam vermelhos e não eram de uma frente só: `pinneapple_veriphysics/validated_domain.py`
entrou com quatro `zip()` sem `strict`, import fora de ordem e `typing.Mapping` no lugar de `collections.abc`
(#2452), e dois arquivos de teste — o da própria #2452 e o `tests/test_solver_contract.py` do contrato de solver
(#2470) — ficaram fora de qualquer shard. Os quatro `zip` agora declaram `strict=True`, porque os comprimentos já
são validados antes da chamada e truncar em silêncio era o pior dos dois mundos; os dois testes ganharam shard
(`solvers` e `veriphysics`), e a linha de base do lint desceu de 9480 para 9472.
