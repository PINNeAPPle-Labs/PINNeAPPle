- `benchmarks/lpm_monitor.py`: the campaign on one screen. It serves a page (`http://127.0.0.1:8130/`, or `--state`
  for the same thing as JSON) with one row per design space — what it predicts and against which reference, the stage
  it is in (draw → build → write → fit → card), the progress of that stage with its ETA, the held-out error against the
  constant-field baseline — and the family that is running right now. It reads files only, so what it shows is what a
  run that died left behind, and it marks a progress file older than three minutes on a running stage as stale instead
  of drawing it as live. `--watch` on `benchmarks/run_lpm_campaign.sh` starts the campaign and opens the panel, and
  `--detach LOG` serves it in a session of its own — a screen for a 34-hour run has the same problem the run does. The
  detach itself moved to `benchmarks/_daemon.py`, which both commands now share.
- `pinneapple_neural/lpm_pipeline.py`: `view.json` next to the card — the loss curve over the sampled epochs, the design
  space with the train/held-out split, and for the best, the median, the worst held-out body plus one training body:
  the geometry, the reference field and the prediction. Every sample names the design it came from, so a picture is a
  picture of that body and never a claim about the space.
- `benchmarks/lpm_figures.py`: the four figures — training loss, the design space, the bodies, reference against
  prediction — drawn from `view.json` with `pinneapple_tools.visualization` (the house theme), served by the panel at
  `/figure/<family>/<kind>.png` and rebuilt only when the view is newer than the PNG. Samples are the best, median and
  worst held-out body, never a hand-picked one.
- `benchmarks/lpm_monitor.py`: clicking a design space opens its figures, side by side with the held-out numbers; a
  family fitted before the view existed says so instead of drawing an empty chart.
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
