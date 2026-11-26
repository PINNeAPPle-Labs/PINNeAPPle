- Custo medido decide os recursos do job (`pinneapple_analysis.cost.resources`, issue #131): `classify` deriva o tipo
  de recurso do que foi medido, não de um rótulo — `gpu_preferred` quando o pico veio de um alocador CUDA ou a taxa
  de trabalho medida passa de 10 GFLOP/s, `io_bound` quando não há FLOPs contáveis e o pico fica abaixo de 1 MiB por
  segundo de execução (o tempo é espera de disco ou de subprocesso, e mais núcleos não encurtam), `cpu_heavy` a partir
  de 0,5 GFLOP/s ou de uma execução de mais de 1 s com pico acima de 64 MiB, e `cpu_light` no resto; as constantes
  estão nomeadas no módulo e cada fronteira tem teste. `recommend_cores` lê a curva de speedup medida
  (`t(1)/t(n)/n`), pede o maior número de núcleos que ainda usa ao menos 0,75 do que recebe, e reduz para o menor
  número que cumpre um prazo quando um é dado — sem extrapolar: a resposta é sempre uma contagem que foi medida, e um
  prazo fora do alcance medido volta com o motivo registrado. `plan`/`JobProfile` levam o pico multiplicado por
  `headroom` (1,25) para a memória por execução, separam a memória da campanha (`workers x por execução`) e marcam
  `memory_is_floor` quando o pico era limite inferior (RSS amostrado): um pedido abaixo de uma medida é falha
  garantida, um pedido a partir de uma amostra é um piso, e o perfil diz qual dos dois é. `JobProfileTable` gera a
  tabela a partir de um `CostLedger` e grava, junto, a máquina onde foi medido (sistema, arquitetura, CPUs, RAM,
  Python, torch, GPU) — o número não é lido como se valesse em outra máquina. Medido em
  `tests/pinneapple_analysis/test_cost_resources.py` (16 testes).
