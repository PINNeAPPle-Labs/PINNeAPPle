# Handoff: base workflows bio (#1205), chip design (#1206), satellite design (#1207)

Written 2026-11-03 for an agent with no context. Everything below was measured in the session that wrote it.

## Goal and links
- Epic #1185 (base workflows). Issues #1205 (bio), #1206 (chip design), #1207 (satellite design).
- Branch `claude/workflows-bio-chip-satellite`, PR **#1261** (open, not merged). The coordinating session merges;
  do not merge it yourself.
- PR state at handoff: all 16 CI checks green on head `4f23c89` (tests, generated, scan, notebooks, ...).

## What is done
Each workflow: library module with the verified model and its checks, a flow node with a planner capability (PT + EN),
a bundled flow JSON, tests, a CI step, Studio inputs/reasons, smoke runs, flow runs and Hub snippet runs.

| Issue | Library | Node / flow | Tests |
|---|---|---|---|
| #1205 bio | `pinneapple_physics/biology/workflow.py`, `standard_models.py` (+ SBML writer fix in `sbml.py`) | `bio.workflow` / `pinneapple_flow/flows/bio_workflow.json` | `tests/test_bio_workflow.py` (13) |
| #1206 chip | `pinneapple_design/chip/` (floorplan, thermal, power_grid, interconnect, workflow) | `chip.design` (+ `litho.opc`) / `chip_design.json` | `tests/test_chip_design_workflow.py` (10) |
| #1207 satellite | `pinneapple_design/satellite/` (budgets, workflow) | `satellite.design` / `satellite_design.json` | `tests/test_satellite_design_workflow.py` (9) |

Other changes:
- `pinneapple_physics/astro/decay.py` `_rhs`: an overshooting Runge-Kutta stage is evaluated at the surface. Before
  this, 3 of 195 long-decay cases raised `math domain error` (it failed CI once).
- `pinneapple_hub/library/categories.py`: `node:bio.` goes under `uq`; `node:chip.` and `node:satellite.` under `design`.
- `.github/workflows/tests.yml`: a step that runs the three test files (installs `pymsis`).
- `changelog.d/base-workflows-bio-chip-satellite.added.md`.

The PR body (#1261) lists every measured number. The main ones:
- SIR final size: error 7e-10 of N.
- PK AUC: matches F·n·D/CL to 7e-16.
- Hodgkin-Huxley: Hopf onset 9.775 vs 9.78 µA/cm² published; pulse threshold 6.919 µA/cm², all or none.
- SBML round trip: exact.
- Chip: 1-D die stack against its closed form 8.5e-15; IR mesh against the discrete-sine solution 2.2e-14; Elmore
  against the transient's first moment 1.5e-6.
- Satellite: SSO 98.188° at 700 km; eclipse against geometry 3e-6; design 7.51 kg; lifetime 9.9 y at F10.7 = 200.

## What is left
1. **#1207 acceptance item "a published CubeSat budget reproduced" is NOT done.**
   - The sources found (FramSat-1 NTNU `orbitntnu.com/framsat-1-doc/framsat1_budget.pdf`, RSP-03
     `rsp03.rymansat.com/assets/pdfs/RSP-03_LinkBudget.pdf`, NASA NTRS 20160001365) returned 403 through the cloud proxy.
   - The arXiv papers that could be fetched (1908.09501, 2211.15781) only give parameters and figures.
   - Next step, from a machine with normal internet:
     1. Open one of those budgets and take every input row (Tx power, gains, losses, range, G/T or T_sys, data rate,
        required Eb/N0) and the published outputs (EIRP, path loss, Eb/N0, margin).
     2. Add a test calling `pinneapple_design.satellite.budgets.link_budget` with those inputs and asserting the
        published outputs to the printed precision.
     3. Optionally add a `verify.published_budget` check in `satellite/workflow.py`.
     4. Then change "Part of #1207" to "Closes #1207" in the PR body.
2. Open question for the owner: the PR covers three issues, while the guidance was one or two per PR. Ask whether to
   split. Splitting would need new branches from main after merge; the content is independent per directory.
3. After the coordinator merges #1261, nothing else is pending on this branch.

## Validation commands (all run in the session, results as recorded)
Environment set-up used here: `python -m pip install torch` (download.pytorch.org was blocked by the proxy, so the
PyPI wheel was used), then `python -m pip install -e ".[dev,flow,hub,mcp,geometry]" -r requirements-docs.txt pymsis`.

```
pytest tests/test_bio_workflow.py tests/test_chip_design_workflow.py tests/test_satellite_design_workflow.py -q   # all pass (~3 min)
pytest tests/test_hub_library.py tests/test_hub_flow_link.py tests/test_flow_palette.py tests/test_flow_planner.py tests/test_flow_extensions.py tests/test_biology_sbml.py tests/test_biology_models.py -q   # pass, 2 pre-existing skips
python -m pinneapple_studio smoke --package pinneapple_design      # 23/23 ok (+1 after lint)
python -m pinneapple_studio smoke --package pinneapple_physics     # 11/11 ok
python -m pinneapple_studio smoke --package pinneapple_flow        # 3/3 ok
python -m pinneapple_studio coverage -o /tmp/cov.md               # 0 "problem:" lines
python -m pinneapple_hub.library flowruns bio_workflow chip_design satellite_design   # done: 49 s, 114 s, 10 s
python -m pinneapple_hub.library run --stale                       # ok
python -m pinneapple_hub.library check                             # index ok
python -m pinneapple_flow coverage -o /tmp/f.md ; python -m pinneapple_flow example-flows check   # ok
ruff check <changed .py files>                                     # clean
```
Run one workflow from Python:

```
python -c "from pinneapple_physics.biology.workflow import bio_workflow as w; print(w(model='hodgkin_huxley')['summary'])"
python -c "from pinneapple_design.chip import chip_design_workflow as w; print(w()['summary'])"
python -c "from pinneapple_design.satellite import satellite_design_workflow as w; print(w()['summary'])"
```

## Known problems and limits
- The satellite lifetime needs `pymsis`. Without it, every F10.7 scenario uses the exponential atmosphere and the
  check `lifetime.flux_scenarios` is inconclusive; the default-run test is skipped.
- Bio design coverage (Wilks, 32 draws) measured with 400 fresh draws: 0.895 for seed 0 (0.845 with 29 draws).
  The bound holds with 95 % confidence, not always. Seed 1 gave 0.99.
- Sobol with `n_sobol=32` is noisy: S1 can exceed ST within the bootstrap intervals.
- Chip: the lid and heat sink have the die footprint (no spreading), conductivities are constant, nets are Manhattan
  with no repeaters. The default floorplan is illustrative.
- Satellite: a single thermal node with a radiator fraction plus MLI. The wheel and torquer steps are representative,
  not a vendor catalogue. Disposal is judged at the mean flux.
- Pre-existing, not from this branch: `tests/test_studio.py::test_api_lists_shows_compares_and_runs_a_tool` (see
  `session-2026-11-03.md` §6).

## Merge order and updating from main
- #1261 is independent of the other open library PRs. When it conflicts, run `git fetch origin main && git merge origin/main`
  (a merge, never a rebase or force push).
- Conflicts seen so far, and how they were resolved:
  - `pinneapple_flow/nodes_ext/__init__.py`: keep both module names, in alphabetical order.
  - `pinneapple_studio/tool_reasons.json`: the union of the keys, then `json.dumps(indent=1, sort_keys=True)`.
  - `data/studio/tool_runs/*.json` and `data/web/hub/flow_runs.json`: take main's file (`git checkout --theirs`),
    then regenerate with the commands above (`smoke --package <pkg>`, `flowruns bio_workflow chip_design
    satellite_design`, `run --stale`). Never edit them by hand.

## Files outside the repo
None needed. The scratch scripts were throwaway drivers of the commands above. The downloaded arXiv PDFs are public
and can be re-fetched: `curl -L -o 1908.09501.pdf https://arxiv.org/pdf/1908.09501`.
