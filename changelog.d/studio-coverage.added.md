- Studio coverage gate (`python -m pinneapple_studio coverage`, in the `generated` job; the table is a page of the docs
  site). Every public symbol has a category. Every executable tool has a valid form and one of: a recorded smoke run
  of its current code, a stated reason, or a line in `pinneapple_studio/coverage_baseline.txt`, a baseline that only
  shrinks. Every example and reproduction runs from the Studio or says why.
  - Smoke runs (`pinneapple studio smoke`) run in disposable checkouts with their own `HOME`, no credentials and no
    network. A tool that writes files is recorded with the files, not excluded.
  - Example inputs are harvested from calls with literal arguments in the tests, examples and Hub snippets
    (`pinneapple studio inputs`). The scripts of examples without a flow are run and recorded
    (`pinneapple studio examples`).
  - The baseline curve is kept in `data/studio/baseline_history.json`, and a weekly workflow re-records the runs.
