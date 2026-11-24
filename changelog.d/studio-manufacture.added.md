`pp.viz.manufacture` monta um corpo pelo histórico de fabricação: `cast_blank` (parede com saída de molde, pele
as-cast e linha de partição), `machine_face`, `bore`, `bolt_pattern`, `weld` e `finish`, cada um deixando geometria e
uma linha em `body.history`, com `body.describe()` para a legenda. Três shaders novos no `pbr` — `as_cast`, `machined`
e `heat_tint` — fazem a pele fundida, a face usinada e a zona termicamente afetada se distinguirem na mesma luz. O furo
é desenhado, não cortado: um kernel de malha é o que falta para furar de verdade.
