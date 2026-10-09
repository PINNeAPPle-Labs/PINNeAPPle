# TODO — PINNeAPPle, validation batch 4 / 0.6.2 release

Ordered by dependency, not importance — do them roughly top to bottom. Updated 2026-10-02.

## Blocking the 0.6.2 release (do these first) — regression CONFIRMED, fixes COMMITTED

- [x] Regression rerun confirmed: 80→68 failed, 786→798 passed, nothing newly broken (see
  CURRENT_STATE.md for the reasoning).
- [x] Committed the fixes (`c38c454b`) and this `.agent/` package (`cd4a2766`).
- [ ] Push the branch (`git push -u origin feat/validation-batch4` from `pp-release-061`), open a PR
  against `origin/main` (title referencing "validation batch 4"), body summarizing: +9 validated
  methods, 4 real solver/compiler bugs fixed, the MPS-leak investigation finding (worth mentioning
  even though not fixed), 2 test-infrastructure bugs fixed.
- [ ] Wait for CI (`.github/workflows/tests.yml`) — remember only
  `tests/test_manufactured_solutions.py` blocks the workflow; the rest runs with
  `continue-on-error: true`.
- [ ] Merge the PR.
- [ ] Fresh worktree from the merged `origin/main` (NOT `$TMPDIR`), `uv build`, install the wheel in
  a clean venv, smoke-import (`pinneapple.__version__ == "0.6.2"`, plus
  `pinneapple_simulation.numerical_solvers.{fem,meshfree,eddy_current_fdm}`).
- [ ] `uvx twine check dist/*`, then `uvx twine upload dist/*`.
- [ ] Confirm on pypi.org/project/pinneapple/0.6.2/, tag `v0.6.2`, push the tag.

## MeshGraphNet follow-ups (PR #232)

- [x] #234: full test sweep, ruff, reduced-example test, TFRecord reader test.
- [x] #237: example 06 on PINNeAPPle's MGN, English README with optional deps and `_out/` policy, CHANGELOG,
  link from `examples/README.md`.

## Other work from this session, not yet wrapped up

- [ ] **PK-PD rerun** (worktree `pp-pkpd-rerun`, branch `chore/pkpd-benchmark-rerun`): commit the
  `benchmarks/_out/ssqn_paper_benchmarks.json` diff (data only, no code change), push, small PR.
- [ ] **Shallow-water browser demo**: currently sitting uncommitted in THIS worktree
  (`examples/numerical_solvers/13_shallow_water_twin3d_browser_demo.py`) — move it to its own
  worktree/branch before committing, don't bundle it into the batch-4 PR (unrelated topic).
- [ ] **Tell the owner**: PINNeAPPle-CFD PR #17 (the erosion → Twin3D bridge) is done and verified
  but needs a MANUAL merge — an auto-mode policy denial ("Merge Without Review") blocked doing it
  unattended, and that denial must not be routed around via another tool/method.

## Method catalog (item 2 of the owner's standing request, ongoing across many sessions)

- [ ] Continue validation batches. Still untested (42) / tested-without-reference (53) as of this
  branch — regenerate `pinneapple_catalog/method_status.json` rather than trusting a cached number.
- [ ] `immersed_boundary_fdm.py`'s channel mode needs a real pressure-Poisson projection step to
  conserve mass (see SCIENTIFIC_CONTEXT.md §4) before S12 can be validated — nontrivial, not a quick fix.

## Explicitly out of scope (don't pick up casually)

- [ ] The MPS test-isolation leak (FAILED_APPROACHES.md #7) — real, worth fixing eventually, needs a
  dedicated investigation/bisection session.
- [ ] Reconciling this `.agent/` package with the other one in the main PINNeAPPle checkout — a
  decision for the project owner, not something to auto-merge.
- [ ] OpenRadioss virtual-solver bridge — someone else's uncommitted WIP in the main PINNeAPPle
  checkout; don't touch without knowing their design intent.

## Resolved during this session (was previously listed as blocked)

- [x] PINNeAPPle-CFD commit `435ad37` (PressNet/PLAID docs) — turned out to already be on `main`
  and pushed by the time this was checked; no action was needed.
- [x] PINNeAPPle-CFD's `erosao_3d` route → `pinneapple_twin3d.openfoam` integration — implemented
  (see "Other work" above); needs a manual merge, not more implementation work.
