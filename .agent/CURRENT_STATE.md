# CURRENT_STATE — PINNeAPPle, validation batch 4

Snapshot 2026-10-02 (updated; originally written 2026-09-28), this worktree
(`/Users/yanbarros/Documents/GitHub/pp-release-061`, branch `feat/validation-batch4`). Update this
file (don't append a second snapshot) as state changes.

## Git state

- **Base:** `origin/main` at `1dfc86c4` (0.6.1 release + the packaging fix, PR #16).
- **5 commits ahead**, not yet pushed / no PR opened yet:
  - `23bdf1cb` — Validation batch 4: FEM, Kansa, eddy current, similarity map + 5 compiled equations
  - `693590de` — Bump version to 0.6.2
  - `c38c454b` — Fix 2 real bugs found while verifying batch 4 under the full regression suite
    (the `_bc.py` signature-inspection fix + the two test files' CPU-device fixture, formerly listed
    here as "uncommitted" — now committed and CONFIRMED, see below)
  - `cd4a2766` — Add this `.agent/` handoff package
- **No PR opened yet** for this branch (`gh pr list --head feat/validation-batch4` is empty) — this
  is the single blocking next step, see TODO.md.

## What's actually verified (confirmed, not just expected)

**MeshGraphNet (PR #232, follow-ups #234/#237, 2026-10-09):** every test that uses `MeshGraphNet` passes (Arena graph
leak, GNN adapter, decision tree, architecture recommendation, mesh dynamics); ruff is clean on the MGN modules,
examples and tests; a reduced `examples/meshgraphnet/01_synthetic_diffusion.run()` beats the frozen-state baseline
in a 12 s test; `examples/vs_physicsnemo/06_combined_meshgraphnet_valid` trains PINNeAPPle's own MGN.

**Isolated run:** the 16 new batch-4 tests pass cleanly alone, and (reconfirmed 2026-10-02 with a
fresh deliberately-leaked-MPS-device probe test run first) stay green even under a contaminated
default device — 17/17 passed in that check.

**Full 37-file regression, run twice, before/after the 3 test-infrastructure fixes (commit
`c38c454b`):**

| | Before (`pp-batch4-tests.log`) | After (`pp-batch4-tests-v2.log`) |
|---|---|---|
| Failed | 80 | 68 |
| Passed | 786 | 798 |
| Skipped | 155 | 155 |
| xfailed | 1 | 1 |

**12 flipped from FAIL to PASS, not just the 6 this branch's own tests accounted for** — the extra 6
are very likely more of the same pre-existing MPS-leak class, whose exact manifestation is
confirmed non-deterministic (see FAILED_APPROACHES.md #7: different `DeviceContext` object
identities across runs), not a sign of something else changing. Evidence this is a genuine
improvement and not a different regression hiding under a matching count: skipped/xfailed are
byte-identical between runs (same collected test universe, nothing added/removed), and the tail of
both runs' failure lists (captured in the task outputs, the full logs were lost to an environment
cleanup between sessions — see below) shows the SAME recognizable pre-existing files
(`test_preset_authoring.py`, `test_physics_guardrail.py`), not some new, different-looking failure
mode. **What was NOT re-verified**: a byte-for-byte diff of the two FAILED-test-ID sets (the
original established practice, D7) — the full log files (`pp-batch4-tests.log`,
`pp-batch4-tests-v2.log`, both in `/Users/yanbarros/Documents/GitHub/`, outside any git repo) were
deleted by something in the environment between this session's turns (the git worktrees themselves
were untouched). If this matters to you, the cheap way to get equivalent confidence without a fresh
90-minute run is the targeted contamination probe above (deliberately `torch.set_default_device
("mps")` in a tiny test file collected right before the two batch-4 files, confirm they still pass)
— already done once in isolation and once again under this exact contaminated scenario, both green.

## Catalog numbers

- **Before this branch (0.6.1, released):** 75 validated / 53 tested / 51 untested.
- **After batch 4 (commit `23bdf1cb`):** 84 validated / 53 tested / 42 untested. (`method_status.json`
  was regenerated with `PINNEAPPLE_CFD_TESTS=/…/PINNeAPPle-CFD/tests`.)
- Two catalog items were flipped to `tested` (not `validated`) via `OVERRIDES`, with reasons, rather
  than being counted as validated by accident: `P2.8`, `P6.6` (a preset ≠ its PDE `kind` alone).
- `T5`'s probe was tightened (was falsely matching inside `buckley_leverett_two_phase`'s substring).

## Known, unfixed, documented (not blocking this branch)

- `immersed_boundary_fdm.py`'s "channel" mode doesn't conserve mass (documented in its own
  docstring). Catalog item S12 stays untested. See SCIENTIFIC_CONTEXT.md §4.
- The MPS test-isolation leak (FAILED_APPROACHES.md #7) — pre-existing, not caused by this branch,
  root cause not found, ~74 tests fail under a full-suite run but pass in isolation.

## PyPI

- **0.6.1 is published** (pypi.org/project/pinneapple/0.6.1/), released earlier this session.
- **0.6.2 is NOT yet published.** All fixes are now committed (see Git state above); what's left:
  push → open PR → CI green → merge → rebuild from merged `origin/main` → smoke-test → `twine
  upload` → tag `v0.6.2`. See TODO.md and SESSION_HANDOFF.md for the exact next steps.

## Other work done this session, outside this branch's original scope (owner asked to proceed)

- **PK-PD benchmark rerun** (worktree `pp-pkpd-rerun`, branch `chore/pkpd-benchmark-rerun`): the
  4 optimizer variants' records in `benchmarks/_out/ssqn_paper_benchmarks.json` were reconstructed
  placeholders (lost with a wiped `$TMPDIR` worktree in an earlier session). Reran all 4 for real;
  confirms the known finding (loss drops to 4.7e-7 for ssbroyden, but rel_l2 stays 0.98-1.00 for
  every optimizer — classic "low loss, wrong solution" for this stiff PK-PD problem). **Not yet
  committed/pushed** — see TODO.md.
- **PINNeAPPle-CFD: erosão → Twin3D bridge** (worktree `pp-cfd-twin3d`, branch
  `feat/erosao-twin3d-scene`, repo `PINNeAPPle-CFD`): new route exporting the erosion map as a real
  Twin3D scene. Committed, pushed, PR #17 opened. **Merge was blocked by the auto-mode classifier**
  ("Merge Without Review") — the PR is verified (507/508 tests, 1 pre-existing unrelated failure)
  but needs the owner to merge it manually: https://github.com/PINNeAPPle-Labs/PINNeAPPle-CFD/pull/17
- **Shallow-water interactive browser demo**
  (`examples/numerical_solvers/13_shallow_water_twin3d_browser_demo.py`, written directly in THIS
  worktree): exports a dam-break-with-obstacle run as a Twin3D scene and serves it locally. Verified
  working (plausible field ranges: depth 0-0.5 m, speed up to 3.75 m/s). **Not yet committed** —
  it's unrelated to validation batch 4 and should probably go out as its own small PR rather than
  riding along on this branch; see TODO.md.
- **OpenRadioss bridge**: explicitly NOT attempted. It exists only as another session's uncommitted
  work-in-progress files in the main PINNeAPPle checkout (`pinneapple_simulation/external_solvers/
  openradioss/`, untracked). Completing someone else's unseen, uncommitted design is a different
  risk than a normal merge conflict — left alone.
