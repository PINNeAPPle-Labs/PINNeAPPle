- O shard `root` deixa de contar 94 erros de import como falha de teste (#625): `tests/test_geometry_library.py`
  constrói cada entrada da biblioteca pelo kernel de geometria, que é `manifold3d`, e o `manifold3d` vive nos extras
  `geometry`/`cfd`. Sem o extra o arquivo inteiro morria — no relatório do shard isso é indistinguível de uma
  biblioteca de geometria quebrada, e escondia as sete falhas de verdade que o resto do shard tinha. Agora pula com o
  motivo e a linha de instalação.
