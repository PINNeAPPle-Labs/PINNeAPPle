---
name: oss-contributions
description: "Where third-party OSS clones live (forkes/), how to commit/PR on them, and the open PRs as of 2026-09-25"
metadata:
  node_type: memory
  type: project
  originSessionId: 59de5785-7d2a-4fce-ae62-9157eb73fc66
  modified: 2026-09-25T12:33:31.530Z
---

Third-party repos the user contributes to live in `pinneapple-labs/forkes/` (not in the root). Root keeps only PINNeAPPle-Labs / barrosyan / ChordIQ repos. Each clone has `origin` = upstream and `fork` = github.com/barrosyan/<repo>.

Commits: author Yan Barros <yanbarrosyan@gmail.com>, no Co-Authored-By / "Generated with" / any AI mention in commits or PR bodies. PhysicsNeMo needs `git commit -s` (DCO). Opening PRs is public: confirm with the user first.

Open PRs (2026-09-25): NVIDIA/physicsnemo#2019 (issue #2001, the user's own), PolymathicAI/the_well#101 (#91), #102 (#77), jax-md/jax-md#425 (#351), #426 (#423), Ceyron/exponax#109 (#58), #110 (Discontinuities channels, no issue).

Parked on purpose: exponax #47/#68 (maintainer must choose formula vs docs), foamlib #892 (design in progress, relevant to [[pinneapple-cfd-project]]), the_well #97–#100 (data/maintainer decisions). Pre-existing failure: jax-md `test_npt_nose_hoover_lammps` float32 fails on main.

**Update 2026-10-09:** the clones the_well, physicsnemo, jax-md and exponax were deleted (no unpushed work, with the user's OK). The September PRs listed above live on the forks (barrosyan/*). For the current external-contribution campaign see [[oss-campaign-2026-10]] and `forkes/OSS_CAMPAIGN.md`.
