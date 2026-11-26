- **Structured run events and `pinneapple runs why`** (E10/#1481): each workflow run writes
  `runs/<id>/events.jsonl` — run start/finish with the DAG hash, the host and the plugin set; step start/finish with
  the attempt, the executor, wall time, exit code and peak RSS when the executor can measure it; the bounded tail of a
  failed step's output; retries and cancellations with their reason. `pinneapple runs show <id>` summarises a run and
  `pinneapple runs why <id>` gives the ordered evidence of a failure without re-running it; `--otlp URL` exports the
  same events as OTLP/HTTP JSON spans.
