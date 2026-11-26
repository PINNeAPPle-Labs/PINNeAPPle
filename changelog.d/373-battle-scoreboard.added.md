- **The battle scoreboard** (`pinneapple_arena.battle`, issue #373): two models on one stream of conditions, and the
  step at which each one leaves the domain it was validated on -- the question a leaderboard's mean error cannot
  answer. `run_all` trains an FNO and a PINN on the same window of a case (1-D heat, Burgers at nu = 0.01), same
  split, same optimiser, same seed, and scores **every** step of the stream: the relative L2 against the closed form,
  the PDE residual of the model's own field, the wall time per seed and the parameter count. The exit is written down
  twice because the two questions are different: `residual_alarm` is what the model can say about itself with no
  reference at all (its residual crossing a factor of the value it had at the edge of its window), and `error_exit` is
  the first step past the window where the error crossed the tolerance this library calls a reproduced field -- the
  column the board ranks by, because ranking by the self-calibrated one would put the better model first for the only
  reason that it had less room to be four times worse. The board also reports the step where the two fields are
  furthest apart and which one the reference agrees with there. `bundle_bench.run_and_predict` is the one addition to
  the benchmark: the same training, the same budget and the same seed as `run_model`, predicting the whole stream
  instead of the held-out block, because the exit is a step and it usually falls on the near side of the split.
  `python scripts/battle.py --run` publishes `benchmarks/battle/results.json` and generates the page from it;
  `--check` is the gate -- declaration, record, page and total order have to agree, so an inverted table cannot be
  published. The Burgers residual monitor carries the operator's own floor on the published grid (0.35 on 129 x 41,
  measured, falling with refinement): the module says so in the record rather than hiding it behind a tighter
  tolerance, and the criterion is calibrated so the exact solution -- and only the exact solution -- never trips it.
