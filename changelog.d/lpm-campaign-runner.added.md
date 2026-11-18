- `benchmarks/lpm_campaign.py`: every design space in one command — one dataset, one fit, one card, nine families,
  cheapest first. It **resumes** (a dataset whose manifest exists is not regenerated and a model whose card exists is
  not refitted; `--force` is the only way to change that), it records the traceback of a family that raised in
  `failures.json` and carries on with the rest, and it exits 1 at the end so a scheduler sees it. `--stage` splits the
  data from the fits, `--plan` prints the designs and a wall-clock estimate from measured per-design costs, `--scale
  smoke` runs the whole wiring in about four minutes, and `--detach LOG` re-runs the command with `os.setsid()`, which
  is what actually protects a long fit — `nohup` only ignores SIGHUP, and what kills it is the process group going
  away. `examples/lpm_five_models/experiment.json` now exists, so the manifest gate is satisfied for the example.
