- Limites numéricos do conector OpenFOAM, medidos e decididos (`pinneapple_data.preflight.limits`, issue #132): cada
  limite de `buoyantSimpleFoam`, `buoyantPimpleFoam` e `interIsoFoam` traz a decisão (aceito e imposto, contornado ou
  não suportado), a evidência medida e o caso mínimo que a demonstra; `decision(solver)`, `limits_for(solver)` e
  `unsupported()` leem o registro, e o documento `docs/guides/openfoam_numerical_limits.md` repete a tabela com os
  números. Rodado de verdade em OpenFOAM v2606: o caso de free surface com k-omega SST (o dam-break `interFoam/RAS`
  trocado de kEpsilon), que nunca tinha rodado, faz 71 passos até 0,2 s em 1,05 s de execução, fração de água entre
  -7,7e-05 e 1,0005 e Courant máximo 1,02 — é suportado, não apenas não testado, e custa três entradas de dicionário
  que o kEpsilon não precisava (`wallDist`, `div(rhoPhi,omega)` e o solver de omega), cada uma uma parada fatal se
  faltar. `buoyantSimpleFoam` sem `constant/g` e sem `pRefCell`/`pRefPoint` para na partida (as mensagens estão na
  evidência) e sem relaxamento não cai, mas estaciona em resíduo de p_rgh 0,39 contra 0,11 com 0,7 — o conector agora
  impõe as duas primeiras e reporta a terceira. `buoyantPimpleFoam` exige entrada de solver para `rho` e, sem
  `correctPhi`/`pcorr`, completa sem avisar com erro de continuidade local 12x maior no primeiro passo (1,6e-3 contra
  1,3e-4, medido): o preflight passa a exigir o solver e a avisar sobre a correção. Regras novas no preflight para os
  dois casos, com os dicionários como teste, e dois achados silenciosos registrados sem regra: `0/` restaurado depois
  do `setFields` roda até o fim com `alpha = 0` sem erro, e a turbulência do lado gasoso fica declarada não suportada.
  Medido em `tests/test_openfoam_limits.py` (13 testes, dos quais 4 rodam OpenFOAM de verdade e são pulados sem ele).
