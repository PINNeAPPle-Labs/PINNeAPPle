- Dívida de lint registrada e decrescente (#51): `python scripts/lint_debt.py --record` grava
  `docs/dev/lint_debt.json` com uma contagem por (arquivo, regra) — hoje **21 339 achados em 1 781 arquivos** — e
  `--check` reprova qualquer par que passe da contagem gravada. Contar por par, e não por total, é o que impede o
  gate de ser pago por troca: tirar cinco `UP006` de um módulo não compra cinco em outro. `--summary` mostra a dívida
  por regra (`UP006` 8 135, `UP045` 3 100, `UP035` 2 173, `I001` 1 783, `E402` 1 389) e diz quantos pares já estão
  abaixo do baseline, prontos para o próximo `--record` baixar o teto. O outro lado — barrar o novo — continua com o
  `.pre-commit-config.yaml`, que roda o ruff da configuração declarada. Junto: seis arquivos de teste que entraram
  hoje em `main` sem shard (`test_astro_yt_reader`, `test_campaign_runner`, `test_compare_solutions`,
  `test_openfoam_limits`, `test_openfoam_pyvista`, `test_viz_layer`) foram declarados, e `scripts/tier_shards.py
  --check` voltou a passar.
