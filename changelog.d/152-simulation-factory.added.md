- Fábrica de simulação genérica (`pinneapple_autolab.factory`): CAD → DOE → solver → dataset → controle de qualidade,
  no lugar do caminho que o produto de CFD tinha só para erosão em tubulação. As quatro pontas são interfaces, e cada
  implementação delega ao que a biblioteca já tem em vez de reimplementar: a geometria vem de
  `pinneapple_design.fitmesh` (limites de validade, volume exato, menor feição, comprimento de referência, bocais e
  ponto no fluido; o sólido exige o extra `cadquery` e, sem ele, as checagens que precisam de sólido ficam `not_run`
  com o motivo), o DOE do `DesignSpace.sample` de `run_study` e do `plan_doe` do `pinneapple_autolab.study` com suas
  métricas de cobertura, o solver é o `Simulator` do laboratório, e o dataset é o `StudyDataset`/`build_dataset` com
  `DatasetSpec`. O caso é o mesmo `DesignStudy` que `run_study` consome — variáveis, mapeamento de entradas, saídas com
  unidade, orçamento e semente —, então um caso não é descrito duas vezes. Controle de qualidade em vocabulário de
  `Check`: run que falhou, resíduos, malha e GCI por linha, e por dataset procedência, saídas, rejeitados e designs
  distintos; `REJECT` impede a escrita, e linha sem procedência (solver, versão, semente, hash dos dados, commit,
  malha) faz o writer recusar o dataset inteiro. O caso de erosão em curva (CFD-04) entra como domain com geometria e
  DOE reais e o solver declarado ausente com o motivo — o solver de erosão (OpenFOAM + DNV-RP-O501) está no
  PINNeAPPle-CFD, não nesta biblioteca. Medido em `tests/test_autolab_factory.py` (19 testes).
