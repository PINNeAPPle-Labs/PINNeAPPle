- Esquema `pinneapple-design-study/v1` e `run_study` (`pinneapple_analysis.design_study`): tipo de estudo
  (otimização, surrogate, PINN/operador, inverso, gêmeo), geometria ou sistema base com variáveis, avaliador com
  mapeamento de entradas e saídas com unidade, objetivos, limites absolutos ou relativos à linha de base, orçamento e
  verificador; `to_dict`/`from_dict` estritos, refusando saída sem unidade, campo desconhecido e outra versão.
  `run_study(spec, evaluator)` faz linha de base → DOE → surrogate → NSGA-II → verificação dos escolhidos e devolve
  frente de Pareto (cada membro marcado medido ou previsto), designs rejeitados com o motivo, veredito de confiança e
  domínio validado; o avaliador é injetado. A otimização busca contra os limites apertados pelo próprio erro do
  surrogate. Três estudos de exemplo (asa, dissipador, polar) com os números medidos nos testes.
