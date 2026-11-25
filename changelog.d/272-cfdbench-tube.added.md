- `pinneapple_simulation.cfd_validation.reference.CFDBENCH_TUBE`: the tube flow's operating-parameter table of
  CFDBench (Luo, Chen and Zhang, 2023, arXiv:2310.05963, Table 3) transcribed as data with its citation — the
  baseline tube, the fifty inlet velocities, the 10 x 10 density/viscosity grid and the five diameters by ten
  length-to-diameter ratios, which is the 175 cases the paper counts. It is a table of *cases*, not of answers: the
  tube is a water-air two-phase flow and the paper publishes no physical solution for it.
- `pinneapple_simulation.cfd_validation.cfdbench.tube_against_table`: the tube archive held against that table, every
  published value looked for on disk and every value on disk looked for in the table. It finds one disagreement, and
  it is in the data: the `prop` family stores its density as the ten-value index `0..9` where the paper publishes
  `10 .. 1000 kg/m3`, so `Re = rho U L / mu` computed from `case.json` is a hundredfold short for that family. The
  other four parameter sets agree value for value. `results/cfdbench_tube.json` holds the record; the runner is
  `run_cases.py --only cfdbench-tube`.
- `examples/cfd/classical_cases/run_cases.py --only pinn-budget --budgets ...`: the training budget of the network
  swept on one command, one row per budget in `results/pinn_budget.json`, each run measured through `pinn.measure`
  with its own cosine schedule over its own budget so the rows differ only in `Config.steps`. Rows merge under a file
  lock, so several budgets can be trained at once and an interrupted one resumes from its own checkpoint.
