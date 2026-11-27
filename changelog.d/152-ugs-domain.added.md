- Segundo domínio da fábrica (`pinneapple_autolab.factory.domains.ugs`) cujo solver é da própria biblioteca:
  `pinneapple_simulation.ugs` (gás real monofásico, fluxo de dois pontos, Euler implícito, Newton com jacobiano
  analítico). A cadeia CAD → DOE → solver → dataset → QC atravessa o solver e fecha em dataset assinado: a geometria
  é a malha do reservatório (`UGSGridGeometry`: 441 células, volume bruto e de poros, transmissibilidade
  mínima/máxima, índice de Peaceman, com as checagens de porosidade e transmissibilidade), o DOE é o maximin com
  cobertura, o solver é `UGSStorage` (uma estação de produção a fundo de poço fixo por design) devolvendo
  entregabilidade de pico [m³/d] e queda de pressão média [MPa], e o dataset carrega procedência por linha com a
  malha. O controle de qualidade ganhou a checagem **conservação** (crítica): o erro de balanço de massa medido pelo
  próprio simulador contra o limite de 1e-3 do caso Ahmed (configurável por `meta`), e `not_run` quando o solver não
  declara conservação. Medido em `tests/test_autolab_factory.py`: 7 designs, `done`/`APPROVED`, erro de balanço
  ~1e-16, GCI `not_run` (uma malha por design).
