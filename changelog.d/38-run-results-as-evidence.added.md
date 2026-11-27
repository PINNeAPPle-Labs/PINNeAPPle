- `pinneapple_core.run_record` — a run result becomes evidence (#38, roadmap B12). Every run writes a record that
  carries what ran (kind, id, parameters), on what (environment and commit), what it produced **by hash** (each output
  file with its sha256 and size), the numbers it publishes, and **the command that re-executes it** — as a list of
  arguments, so a tool can run it — plus a digest over all of it. Editing any field, a published metric included,
  breaks the digest, and `verify()` says so instead of raising: a missing output, a changed file, a digest that no
  longer matches, a record with no command. `index()`/`find()` answer the three questions a report asks (by id, by
  kind, by metric), and `python -m pinneapple_core.run_record verify <folder>` checks every record under a tree in one
  call.
- The integrations make it a property of the tree rather than an intention: the Benchmark writes one record per job
  under `<out>/records/<job>.record.json` (with the config, seed, budget and instance it ran, and the command that
  re-runs it), and each (problem, method) row of the report now names the records behind its numbers; the examples
  runner writes one record per script under `run_results/<category>/<name>/<name>.record.json`, and the report's
  every row names its own. Output paths in a record are relative to the record, so moving a run folder does not
  invalidate the evidence.
