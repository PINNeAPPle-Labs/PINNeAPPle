`pinneapple_tools.visualization.studio.hero(result, kind)` gera os ativos de destaque de um experimento numa chamada
(#920): still em 1920×1080 e 3840×2160, turntable em MP4 e o glTF que o viewer web lê, com um preset por física
(`external_aero`, `internal_flow`, `conjugate_heat`, `structural`, `electromagnetic`, e os apelidos `fluids`, `heat`,
`structures`, `em`). O preset escolhe o campo pela ordem de preferência, o palco, a câmera e o que mais entra na cena
(streamlines, plano de corte, parede translúcida). O retorno diz o que saiu de verdade: `backend` (cycles ou pyvista),
`fallback` com o motivo quando não é Cycles, `stills` por tamanho, `turntable` com frames, fps, backend e tempo por
frame, `gltf`, `viewer` e `field` com a faixa de cores. Sem Blender os stills saem pelo PyVista fora de tela com o mapa
de cores da casa e a mesma composição — declarado, nunca disfarçado; sem nenhum dos dois, nenhum still é escrito e o
motivo é devolvido, e o glTF sai do mesmo jeito porque geometria não precisa de renderizador. Deformação é exagero
declarado: `warp` traz o campo, o fator aplicado vértice a vértice, o deslocamento máximo medido e a frase que vai no
título da barra de cores, e a malha não sofre suavização nenhuma além do sombreamento. O turntable só vira MP4 se
`ffmpeg` estiver no PATH (os frames do PyVista custam ~0,06 s cada, então um clipe de 8 s a 24 fps leva segundos;
frames do Cycles custam dezenas de segundos cada e o tempo medido vai em `per_frame_s`). Também há
`hero_from_experiment(folder)`, que acha a geometria pelo manifesto e grava em `<folder>/report/hero`, onde
`experiment_report` a encontra sozinho.
