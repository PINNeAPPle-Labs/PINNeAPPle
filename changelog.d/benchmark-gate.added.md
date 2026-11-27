O benchmark visual vira gate: `data/benchmark/baseline.json` guarda as quatro estatísticas de cada cena de referência
na configuração canônica (`GATE`), `pp.viz.benchmark.gate` aponta o que saiu da tolerância e o workflow
`benchmark.yml` roda isso em todo pull request que toca o studio. O tempo é registrado, não comparado: ele é da
máquina. Baseline novo se grava de propósito, com `--record-baseline`.
