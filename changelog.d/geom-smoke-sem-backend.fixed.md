O smoke test de `tests/pinneapple_geom/test_mesh_ops.py` falhava no shard bloqueante `geom` desde que o
`trimesh` entrou no extra `geom` do CI sem o decimador: `simplify_mesh` com o padrão `on_missing="error"`
recusa quando `fast-simplification` e `open3d` (extra `geom_simplify`) não estão instalados. O teste passa a
usar `on_missing="return_input"`, que é o contrato documentado para "não há decimador aqui", e dois testes
novos pinam a recusa com a mensagem que diz o que instalar e a simplificação de verdade quando o decimador
existe.
