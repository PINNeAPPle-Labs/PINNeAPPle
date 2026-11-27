- Executor de campanhas (`pinneapple_orchestration.campaign`, `pp.campaign.run`, issue #295): a camada de execução que
  faltava embaixo da fábrica de simulação, no lugar do laço que cada app reescrevia. Um caso é endereçado pelo
  conteúdo — SHA-256 da geometria (os bytes do arquivo, ou de toda a árvore de um diretório), das configurações do
  solver e das condições —, então campanha repetida, retomada ou em outra máquina não refaz o que já está feito; e o
  acerto é *provado*: o registro é relido, a própria chave e o hash das saídas são conferidos antes de pular o run, e
  registro corrompido é miss (refaz, não confia). Dentro de uma campanha o mesmo endereço roda uma vez: 2000 casos com
  25 conteúdos distintos executam 25 jobs e respondem 1975 linhas a partir deles (`source` diz qual é qual). A fila
  tem retries e timeout; o run que falha vira linha com trecho do log e classificação — `mesh`, `diverged`,
  `timeout`, `setup`, `crashed` — construída sobre `pinneapple_data.simulation_metadata.openfoam.parse_log`, e só as
  classes que podem passar na segunda tentativa são repetidas (`timeout`, `crashed`, `unknown`; malha errada e
  divergência são determinísticas). Cada run termina gravado antes de a fila seguir, então um processo morto no meio
  perde só o que estava executando. Backends: `LocalBackend` (padrão), `SlurmBackend` (`sbatch` + `squeue`, mesmo
  contrato de `outputs.json`) e `CloudBatchBackend`/`aws_batch_backend` (serviço de lote atrás de `submit`/`poll`;
  as chamadas da CLI da AWS são testadas, a resposta de uma conta real não — está dito no módulo e na issue).
  Contabilidade com linhagem: `campaign_report.json` com tempo de parede, horas de CPU (tempo x núcleos), taxa de
  falha por classe e a máquina onde rodou; `lineage.json` com a cadeia geometria → run → dataset via
  `pinneapple_data.lineage`. Medido em `tests/test_campaign_runner.py` (23 testes, com solver dublê, SLURM falso e um
  processo que morre no meio para provar a retomada).
