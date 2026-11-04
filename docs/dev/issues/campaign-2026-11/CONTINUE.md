# CONTINUE — experiment campaign issues (PINNeAPPle)

Status: **complete.** 79 issues created in `PINNeAPPle-Labs/PINNeAPPle`; the series index is **#1288**, and its
body now carries the campaign table with every issue number by series.

## What was done

- All 79 bodies exist in this folder, one `ID.md` per entry of `manifest.tsv`, and pass
  `python3 build_manifest.py --check` with no problems.
- `created.tsv` maps `id <TAB> number <TAB> url` for all 79. Numbers run #1288–#1374.
- The index issue body was extended with `## Campaign index`: one table per series (A 1, B 26, C 12, D 39, E 1)
  plus three series notes.

## How to re-check

```bash
cd /Users/yanbarros/Documents/GitHub/pinneapple-labs/_issue-drafts
python3 build_manifest.py --check   # 0 bodies missing, no problems
```

## If the campaign is ever extended

1. Add the entry to `BACKLOG-A-B.md`, `BACKLOG-C.md` or `BACKLOG-D.md` in the `## ID | Title | labels: ...` form.
2. Rerun `build_manifest.py` so `manifest.tsv` picks it up.
3. Write `_issue-drafts/<ID>.md` following `SPEC.md` (the 17 `##` headings, English only, no attribution, no
   invented results, only the verified module inventory, real references only).
4. `python3 create_issues.py create <ID>` — it prepends `Part of #1288.` automatically.
5. Add the new issue to the campaign table in #1288 by regenerating it from `created.tsv`.

## Facts worth not re-deriving

- El Niño forecasting already exists in the repository (`pinneapple_systems/time_series/enso`,
  `examples/enso_forecasting/`, `docs/guides/enso_forecasting.md`) — issue A1 (#1289) is a follow-on
  (sealed out-of-sample forecast, calibration, regime-conditional skill), not a greenfield build.
- The Fiction-to-Reality protocol already exists as `pp.futures` (six evidence axes, two verdicts,
  `PLAUSIBILITY_CLASSES`) — issue D0 (#1288) adopts it rather than proposing a new framework.
- `SPEC.md` carries the verified module inventory; anything not in it must be listed under
  "What is missing (the work)", never invented.
- `pinneapple_quantum` is a variational quantum-ML package (ansatz, physics loss, classical/PennyLane/Qiskit
  backends), not a general circuit simulator: D19's work item is mid-circuit measurement plus density-matrix
  channels, and its body says so.
- The intent layer behind `pp.design` covers five templates (`bracket`, `heatsink`, `glider_wing`, `loop_car`,
  `example_spring`) with a deterministic unit parser and explicit `OutOfScope` / `NeedsInput` paths — D26's
  benchmark is written against exactly that scope.
