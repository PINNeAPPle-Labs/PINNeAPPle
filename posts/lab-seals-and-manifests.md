A plot in a repository says nothing about whether the result behind it was ever checked. We wanted every result in PINNeAPPle to carry that answer itself.

What we built in the PINNeAPPle Lab:
- A to F quality seals. `python -m pinneapple_lab curate` grades every run from its own stored checks: F for a crash, divergence, a failed physics check or a contradicted reference; E for any other failing check; A needs an independent reference plus a baseline or physics check, reproducibility and documentation. Gates, not averages. No argument sets a letter by hand, and `stamp` puts the seal on figures, GIFs and MP4s.
- An `experiment.json` manifest for every example folder, Lab run, workflow run and `pp.solve` result, with a CI gate that fails when a manifest is missing, lists a file that does not exist or reports a metric without a value.
- An experiments page that grades each folder through its Lab items, and an animated showcase panel that only accepts A and B items.

Measured on the committed lab:
- 85 items from 212 runs: 19 A, 53 B, 0 C, 7 D, 1 E, 5 F.
- Re-curating those runs in a fresh store reproduced every letter and headline (the repo test passed), and the gate reported 0 problems over 68 example folders and 212 run folders.

The bad grades stay visible. One F is an EM cavity PINN with a relative L2 error of 1.33 against a 0.05 tolerance; its seal prints the failing check and the next step. The experiments page still marks 52 of 122 entries "not curated", each with the command that would curate it.

Guide: https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main/docs/guides/adding_an_experiment.md

#ScientificComputing #PhysicsInformedML #Reproducibility #Verification #ContinuousIntegration #OpenSource #PINNeAPPle

---
## Media

1. `media/lab-seals-and-manifests/lab_seals_and_manifests.mp4` (64.6 s, 1080x1350, H.264): title card with the items per letter; the six A-F seals (committed legend); an F seal stamped on the EM cavity PINN figure; a Playwright recording of the experiments page built locally (`python -m pinneapple_lab site`) with the grid, the F filter and an F item's problem and next steps, then an A item's trust card and metrics; the committed animated showcase panel; the CI gate card; end card with commands and the repo URL.

## Sources

All paths relative to the repo root (main at e716a065; PRs #835, #849, #851, #852, #836).

- 85 items, 19 A / 53 B / 0 C / 7 D / 1 E / 5 F -> `lab/curation.json` `items[*].tier` (counted); re-curated in my run (`--root` scratch store with `lab/runs`, `reindex` + `site`): same counts.
- 212 runs -> `lab/curation.json` `runs` (212 keys); `ls lab/runs/*/*/run.json` = 212; `reindex` printed "212 runs indexed".
- Every letter and headline reproduced -> my run of `tests/test_lab_grades.py::test_committed_lab_curation_is_reproduced` (passed), together with `tests/test_experiment_manifests.py` and `tests/test_experiments_site.py` (20 passed, 1 skipped: Chromium screenshot test).
- Gate 0 problems, 68 example folders, 212 run folders -> my run of `scripts/check_experiment_manifests.py` ("every experiment has a valid manifest"); 68 = `pinneapple_lab.manifest.example_dirs()`; run folders = the 212 above. Must-pass step: `.github/workflows/tests.yml` line 50-53.
- Tier rules (F/E/A gates, no tier by hand) -> `docs/core_concepts/lab.md` "Curation" table; `tests/test_lab_grades.py::test_the_tier_cannot_be_set_by_hand`.
- Manifest rules (missing manifest, missing file, metric without value) -> `docs/guides/adding_an_experiment.md` section 4; `pinneapple_lab/manifest.py` `check_repo`.
- Showcase accepts A and B only -> `docs/core_concepts/lab.md` "Showcase panel" ("a C to F item is never shown").
- EM cavity relative L2 1.33 vs 0.05 -> `lab/curation.json` `items["em_pinn/em_tm_cavity"].next_steps[0].measured` = {value 1.3295, threshold 0.05}; `why` "value 1.33, max 0.05".
- 122 entries, 52 not curated (and 18/46/0/3/1/2 per letter in the page bar, video only) -> experiments page built in my run from main e716a065 (`python -m pinneapple_lab --root <scratch> site`). The committed `docs/lab/coverage.md` (2026-10-18) says 119 experiments / 49 not curated; the difference is folders added since.
