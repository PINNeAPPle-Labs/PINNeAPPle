- O shard `flow` volta a passar (#625): três vermelhos, três causas distintas. `examples/data_readiness` entrou em
  `main` sem entrada em `pinneapple_flow/flows/examples/inventory.json` — agora tem a razão declarada (o portão de
  prontidão é o passo anterior ao nó que treina, não um nó). O nó `ugs.storage_season` ficou na seção recolhida
  `library` da paleta por falta de linha em `pinneapple_flow.palette.SECTION_OF` — vai para `solvers`, onde estão os
  que rodam simulador. E `tests/test_flow_wrappers_car3d.py` morria no import de `trimesh` (extra `geometry`), que é
  ausência de dependência opcional, não wrapper quebrado.
