- `pinneapple_systems.control.optimal`: controle ótimo resolvido como equação diferencial em vez de tiro em
  torno de um simulador, em duas formas sobre a base X-TFC que a biblioteca já tem (features ELM fixas,
  colocação de Chebyshev-Gauss-Lobatto, derivadas analíticas da base).
  - **PoNN** (`pontryagin`): o Princípio do Máximo de Pontryagin vira problema de valor de contorno com
    expressão constrangida TFC — as condições de contorno valem para todo vetor de coeficientes, nunca como
    perda penalizada. Para problema linear-quadrático com deriva afim e custo quadrático o resíduo é afim nos
    coeficientes, então **uma** solução de mínimos quadrados é a solução inteira.
  - **BeNN** (`bellman`): resíduo de Hamilton-Jacobi-Bellman sobre uma rede de valor cuja condição terminal é
    uma expressão constrangida TFC, Gauss-Newton com Jacobiano analítico (`hjb_jacobian`), e a política sai do
    gradiente do valor, `u = -R^-1 B' grad V`, que atua em estados para os quais a rede nunca foi resolvida.
  - confiança: `hjb_residual` avaliado ao longo da trajetória que a política realmente voa diz se ela ainda
    está na região em que foi ajustada; `landing_demo` entrega o veículo ao MPC (`mpc_control`) quando cruza o
    limite.
  - Números medidos nos testes novos (`tests/test_control_optimal.py`): regulador do integrador duplo contra
    `scipy.linalg.solve_continuous_are` com 1,8e-16 em `P` e 0,0 em `K`; PoNN contra a Riccheta integrada para
    trás com 1,0e-4 de erro de custo e 4,5e-8 de resíduo; transferência de energia mínima com restrição terminal
    dura reproduzindo a forma fechada em `u(0)`, `u(T)` e custo 20,000750 contra 20,0; ganho da BeNN 0,17 % e
    0,89 % do `K(t0)` da Riccheta.
  - Na demo lunar (pouso bang-bang de 8,40 s, 28,61 m/s de combustível) o resíduo relativo é 0,409 na descida
    nominal e 17,81 depois de uma rajada de -20 m/s que tira o veículo do corredor; o hand-over dispara em
    4,00 s e leva o toque de +22,97 m (só a política) para +2,59 m.
  - Escopo honesto no docstring e em `docs/guides/pontryagin_bellman.md`: os casos 3-D não lineares dos
    artigos não estão implementados, nenhuma tabela publicada é reproduzida, e a rede de valor estacionária foi
    removida porque o resíduo de HJB é satisfeito exatamente por **todas** as raízes simétricas da equação de
    Riccheta — medido, ela converge para a raiz anti-estabilizante (`3,8e-16` de resíduo, ganho `[-0,62, +0,62]`
    contra `[1,00, 1,73]`).
