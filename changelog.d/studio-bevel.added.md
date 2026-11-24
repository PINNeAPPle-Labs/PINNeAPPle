`pp.viz.bevel_scene` fecha o último item do nível cinematic de geometria: o chanfro de uma aresta que já existe. Não é
um builder ao lado do corpo — é o corpo com material cortado —, então solda as superfícies (a malha que volta do glTF
tem vértices partidos nas emendas), aplica o modificador bevel do Blender com limite de ângulo, e devolve uma `Scene`
que renderiza e exporta como qualquer outra. Recusa rodar quando há campo por vértice sem `drop_fields=True`: o bevel
reconstrói a malha e o campo não sobreviveria.
