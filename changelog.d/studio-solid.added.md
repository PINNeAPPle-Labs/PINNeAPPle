`pp.viz.solid` traz o kernel que faltava para furar de verdade: primitivas fechadas (`solid_drum`, `solid_box`,
`cutter_cylinder`), a checagem de manifold (`is_closed`, `open_edges`) e `boolean` — diferença, união e interseção pelo
solver exato do Blender, com recusa explícita de casca aberta em qualquer lado. `Body.bore` corta quando há kernel e
corpo fechado e desenha quando não há, e a história registra qual dos dois aconteceu. `loft` e
`rounded_rect_profile` dão as formas orgânicas por seções (transição de duto, voluta); `Body.spec()` e
`manufacture.build()` escrevem e reproduzem a história como dado, uma folha de processo em JSON.
