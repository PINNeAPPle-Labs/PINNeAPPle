---
name: pinneapple-series-x-progress
description: Progress of the PINNeAPPle issues campaign (series X3-X13 done, X14 in progress), where the handoff doc lives, and how to resume
metadata:
  type: project
---

Campaign on the PINNeAPPle-Labs/PINNeAPPle issues: one at a time from the first open one, only unassigned or barrosyan's, commit+push+PR+merge+comment, all as barrosyan with no Claude/AI mention (see [[sem-mencao-claude]]).

**Done (merged, issues closed):** #187-#197 = X3-X13 via PRs #290, #291, #300, #301, #302, #303, #304, #307, #308, #309, #310. New package `pinneapple_core` (Field/Mesh/Domain/Geometry, operators, pp.func, fem, PhysicsBackend, PhysicsModule, pp.loss, PhysicsOptimizer, data API, transforms) plus `pp.compile` and the solve() contract (kinds, fem/external backends).

**In progress:** #198 (X14, pp.distributed). Branch `feat/x14-distributed` is pushed with a WIP commit and no PR. Tests are not validated (suite was interrupted because it was heavy; limit workers).

**Handoff doc (versioned):** `PINNeAPPle/docs/dev/HANDOFF_CAMPANHA_ISSUES.md` on the branch `feat/x14-distributed`. It has the per-issue workflow, environment gotchas (use `python3` + `PYTHONPATH=.`, pre-existing segfault in tests/test_breadth_six_packages.py), the table of what was done with honest caveats, and what remains.

**Remaining (2026-10-09):** 229 open issues: X15-X40 (#199-#227), data experiments (#228-#289, many need GPU/big data), old backlog #6-#186.

**Why:** the user wants steady, correct, one-at-a-time progress and a light load on their laptop. **How to apply:** continue from #198 (finish with light tests) then #199; follow the handoff doc.
