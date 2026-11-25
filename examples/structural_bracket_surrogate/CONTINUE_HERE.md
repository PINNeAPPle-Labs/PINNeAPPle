# Resume the Danilo structural bracket reproduction

Paused 2026-11-14. Working PR: https://github.com/PINNeAPPle-Labs/PINNeAPPle/pull/1722
Working branch: `feat/structural-bracket-surrogate-baseline`.
Companion API draft PR: https://github.com/PINNeAPPle-Labs/pinneapple-engineering-api/pull/312
Full cross-repository handoff: API `docs/REPRODUCTION_HANDOFF_2026-11-14.md`.

## Next work

1. https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1725 — run/fix tests and full CAD→FEM→DeepONet benchmark; record independent validation.
2. https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1724 — obtain original source assets, resolve architecture parameter discrepancy.
3. https://github.com/PINNeAPPle-Labs/pinneapple-engineering-api/issues/315 — runnable workers and artifact handoff.
4. https://github.com/PINNeAPPle-Labs/pinneapple-engineering-api/issues/316 — Studio 3D and VeriPhysics overlays.
5. https://github.com/PINNeAPPle-Labs/pinneapple-engineering-api/issues/317 — CI, dependency pin and release validation.

Source results are **not** reproduced yet. Do not merge without passing tests,
numerical validation and documented scientific fidelity. API currently pins
draft library SHA `6abe0862b19d646765e65afea7427a2f6d449158`
for staging integration only.
