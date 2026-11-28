- `docs/guides/lpm_pipeline.md`: a section on what a killed campaign now costs. The fit checkpoints every fifteen
  minutes into `model/fit.pt` and the next start continues from the epoch after it, and three launchd jobs keep it
  alive — the campaign, a watcher that restarts it and then runs the 900-body airfoil refit, and a guard that restarts
  the other two until `~/lpm-campaign/.finished` exists.
