---
name: oss-campaign-2026-10
description: User-approved campaign (2026-10-09) to contribute to big open-source science repos: rules, catalog location, repos to skip, PRs opened
metadata:
  type: project
---

**Full handoff doc: `pinneapple-labs/forkes/OSS_CAMPAIGN.md` (PR table, repo-specific rules, skip list, env recipes, tools in `forkes/tools/`).** Status: 10 PRs sent, 1 merged (nwm-post-processing#179), 9 open and unreviewed as of 2026-10-09; the user asked to save everything and resume later.

On 2026-10-09 the user approved opening small PRs on third-party science OSS repos (NASA, NOAA, NVIDIA, SciML, ECMWF, DOE labs...) without confirming each one. Catalog: `pinneapple-labs/forkes/OSS_REPOS.md` + `oss_repos.csv` (2,793 candidate repos, built from the GitHub API). Clones live in `forkes/`, remote `upstream` = original, `origin` = barrosyan fork.

**Rules from the user:** commits/PRs/issue comments in English, authored as barrosyan (git user Yan Barros <yanbarrosyan@gmail.com>), no mention of Claude/AI. Read each repo's CONTRIBUTING first; comment on the issue before starting when asked. One PR at a time, keep the PC load low (user complained about CPU use; no long test runs, limit workers).

**Why skip some repos:** NOAA-GFDL repos (pace, fre-cli) require a known human in the loop for AI-assisted work and prior contact; skipped. If any repo demands AI disclosure or a CLA/DCO, stop and tell the user instead of hiding it. PhysicsX and CIA have no usable GitHub orgs.

**LLM disclosure repos (stop and ask the user, never tick 'No'):** idaholab/MontePy (PR template asks model/harness/effort; user decided on 2026-10-09 to DROP this repo; clone deleted), llnl/Surfactant, ansys/pyaedt, ansys/pyspeos, NVIDIA/cuml, google-deepmind/concordia have AI policy files. Screen with scratchpad screen2.py (CONTRIBUTING + PR template + copilot-instructions grep).

**How to apply:** pick repos with good-first-issue/help-wanted + CONTRIBUTING + recent merges (see catalog), claim the issue, small fix with a test, PR.

**PRs opened:** NOAA-OWP/nwm-post-processing#179 (issue #90, name source file in construction errors; needs `brew install nco`, installed); sandialabs/WecOptTool#463 (issue #314, docs on nsubsteps; PRs go to branch `dev`, title starts with `DOCUMENTATION:`); NCAR/DART#1195 (issue #808, obs_common_subset doc typos; sparse clone in forkes/DART); NOAA-OWP/DMOD#731 (issue #711 first slice: dataservice tests -> IsolatedAsyncioTestCase; default branch master; needs py3.12 venv via `uv` + ngen-config, ngen-config-gen, hypy from git; second slice NOAA-OWP/DMOD#732 externalrequests + test_scheduler_client; left alone: communication websocket/decorated tests (need project ssl dir, fail at baseline) and requestservice).

**Also skip (legal commitment on user's behalf):** ECMWF/anemoi* PR template says opening the PR affirms the Contributor License Agreement; nasa-jpl/tos2ca-anomaly-detection (maintenance ended May 2026, design-level issues).

**User asked to delete clones when no longer needed:** use `--depth 1`/sparse clones, rm -rf the clone after merge/close or when dropping a target. Old clones the_well, physicsnemo, jax-md, exponax were deleted on 2026-10-09 at the user's OK (they had no unpushed work); earlier PRs from them are on the forks.

**Looked at and dropped (do not redo):** NOAA-OWP/inundation-mapping (Docker/DevOps/changelog, #1011 obsolete); precice/aste (C++, needs preCICE+VTK to build, #191 --mesh with extension is the easy bug); esa/torchquad (AI-friendly, CLAUDE.md rule 11 says never mention AI in commits, but CI wants all four backends: #123 unequal N per dim is big); sandialabs/PEAT (#17 bandit = 1237 findings; #28 needs real devices); SciML (needs Julia, not installed).

**Clones removed to save disk (PRs still open, branches live on the fork):** DART (sparse clone ballooned to 621M) and WecOptTool (341M). If a maintainer asks for changes, re-clone with `--depth 1` from `barrosyan/<repo>` on the PR branch (WecOptTool: docs/314-explain-nsubsteps, DART: docs/808-obs-common-subset-fixes).

**More PRs:** NOAA-OWP/hypy#40 (issue #33, move tests to python/test; its test_nwislocation.py hits the live USGS NWIS service and is flaky/hangs). Skipped: nasa/python_cmr#49 (someone already opened a PR).
**Merged:** NOAA-OWP/nwm-post-processing#179 (merged by maintainers within hours; clone deleted). The NOAA-OWP repos (public-domain CONTRIBUTING, claim the issue first) respond fast.
**ras2fim:** NOAA-OWP/ras2fim#334 (issue #327, code_version in run_arguments.txt). ras2fim PRs go to `dev`, title `[Npt] PR: ...`, branch `dev-...`, CHANGELOG entry with PR number (second commit), black line length 110. Dropped as design-heavy/old: DMOD#712, t-route#596/#364, python_cmr#49 (taken).
**sdynpy:** sandialabs/sdynpy#29 (issue #27 items 2+3: shear modulus 1+nu, rectangle torsion constant; item 1 left to maintainers: ei1 is 'about axis 1', callers beamkm_2d/System.beam/frame_wing depend on current I1/I2). Verify claims against OLD code with `git show upstream/main:<file>` (git stash does nothing after committing).
**More legal-term skips:** Unidata/MetPy (CLA bot), sandialabs/OpenCSP (contributor must represent employer authorization). Wording: don't write 'I agree to the waiver' in PRs beyond what the repo asks; plain submission is enough.
**Discovery tip:** `gh search issues <typo|broken link|docstring|spelling> --owner <org> --state open --no-assignee --match title` finds small doc fixes; AI-written detailed bug issues (reproducer + suggested fix) are good targets but verify numerically first.
**bespokebpv7 (nasa-jpl, BSD-3, no CLA, repo has AGENTS.md = AI-friendly):** PR#65 (issue #52 README typos), PR#66 (issue #45 items 2+4: deep-copy EID lists, header typo). Its ruff config is `select=ALL` so new tests go in existing files (they carry the Caltech header; do not create new files with a copyright header). Venv: uv py3.12 + hypothesis + ruff + mypy; run ruff check/format --check + mypy before PR.
