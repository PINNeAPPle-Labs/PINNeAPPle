# Viga em balanço (CalculiX)

Execução de referência do solver CalculiX (`ccx`) de uma viga em balanço de 1,0 × 0,1 × 0,1 (m).

- `beam.inp`: arquivo de entrada (malha e condições do caso).
- `beam.frd`: resultados (campos de deslocamento e tensão).
- `beam.cvg`: histórico de convergência.
- `beam.sta`: resumo do passo e do tempo.
- `beam.dat`, `beam.12d`, `spooles.out`: saídas do solver, vazias nesta execução.

Origem: pasta de trabalho local `pp-ccx-run` (execução de 2026-09-26). Não há script de geração; a execução foi feita com o `ccx` sobre `beam.inp`.
