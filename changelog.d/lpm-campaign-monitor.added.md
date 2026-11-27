- `benchmarks/lpm_monitor.py`: the campaign on one screen. It serves a page (`http://127.0.0.1:8130/`, or `--state`
  for the same thing as JSON) with one row per design space — what it predicts and against which reference, the stage
  it is in (draw → build → write → fit → card), the progress of that stage with its ETA, the held-out error against the
  constant-field baseline — and the family that is running right now. It reads files only, so what it shows is what a
  run that died left behind, and it marks a progress file older than three minutes on a running stage as stale instead
  of drawing it as live. `--watch` on `benchmarks/run_lpm_campaign.sh` starts the campaign and opens the panel.
- `pinneapple_neural/lpm_pipeline.py`: `progress.json` while a stage runs. `generate_dataset(progress_every_s=...)`
  writes the designs done, the rate and the time still needed; `train(progress_every_s=...)` writes the same for
  epochs, with the loss. Both are off by default and write atomically, so a reader never sees half a record. This is
  what makes a four-hour fit visible at all: without it nothing reaches the disk between the first epoch and the card.
  `pinneapple_neural/lpm.py`'s `fit()` gained `on_epoch(epoch, loss)`, which costs one comparison per epoch when unused.
- `benchmarks/lpm_campaign.py`: one design space at a time, and it says so. The families already ran in sequence in one
  process; now a lock in the output directory keeps a second campaign from starting against the same `--out`, and
  `--threads` (default: half the cores) caps the torch/BLAS pools so the machine stays usable while it trains. A second
  and third way to end a night honestly: `--max-hours` stops between families (never inside one, where the work would
  be thrown away) and exits 0 with the rest resumable by re-running the same command. `runner.json` records which
  family is in flight and what the ones before it produced, which is what the panel reads.
