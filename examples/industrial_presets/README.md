# Presets industriais

Problemas de engenharia já montados, resolvidos e validados com o PINNeAPPle. Cada pasta tem a ficha (entradas SI com a fonte de cada valor padrão), os resultados rodados aqui, a validação com o erro medido e as figuras. Regenere com `python -m pinneapple_presets report all`.

| preset | setor | decisão do caso padrão | validação (pior erro / tolerância) | surrogate (erro hold-out máx) | inversão (erro) | 3-D | otimização | variantes |
|---|---|---|---|---|---|---|---|---|
| [Dissipador de aletas planas com ventilação forçada (módulo de potência)](plate_fin_heatsink_forced/README.md) | Térmica / eletrônica de potência | OK | passa: 7.5 % / 8 % (5 casos) | 5.48 % | 0.26 % | sim | sim | 6 |
| [Temperatura de junção numa placa de potência (conversor DC-DC 4 camadas)](power_pcb_junction/README.md) | Térmica / eletrônica | NOT OK | passa: 0.75 % / 8 % (3 casos) | 2.16 % | 1.2 % | sim | sim | 6 |
| [Passarela de aço de 30 m: resistência, flecha, flambagem lateral e vibração de pedestres](steel_footbridge_pedestrian/README.md) | Estrutural / construção | NOT OK | passa: 4.2e-05 % / 0.2 % (4 casos) | 7.44 % | 2.6 % | sim | sim | 5 |
| [Rede de distribuição de água: pressões, vazões e dimensionamento de menor custo](water_distribution_network/README.md) | Fluidos / saneamento | OK | passa: 0.42 % / 1 % (6 casos) | não (o solver leva ~5 ms por rede; um surrogate não economiza tempo e só acrescentaria erro) | 0.044 % | sim | sim | 5 |
| [Golpe de aríete no fechamento de válvula de adutora de aço](water_hammer_valve_closure/README.md) | Fluidos / saneamento e óleo e gás | OK | passa: 0.012 % / 5 % (2 casos) | 18 % | 0.0015 % | sim | sim | 5 |

Uso: `pp.presets.load(nome).run(**params)`, `python -m pinneapple_presets run nome k=v`, nó `preset.run` no Flow Studio. Índice para o Hub: `pinneapple_presets/index.json`.
