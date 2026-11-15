# Rocket Landing 001 — SESSION HANDOFF (2026-11-14)

## Goal
Reproducible physically grounded autonomous vertical rocket landing experiment in PINNeAPPle and pinneapple-engineering-api, with interactive 3D simulation, trained surrogate, scenario sweeps, physical verification, and traceable results. This is a **research simulator**, not flight-ready GNC.

## User-confirmed baseline
- altitude=1500 m; mass=18000 kg; horizontal wind=8 m/s; engine nominal; requested architecture=hybrid.
- Additional **assumed, not Falcon 9 validated** parameters: x=0 m, vx=0 m/s, vz=-60 m/s, dry mass=13000 kg, fuel=5000 kg, max thrust=400000 N, Isp=300 s, g=9.80665 m/s², RK4 dt=0.02 s, seed=42.
- Planar 2D model includes (x,z,vx,vz,m), quadratic wind-relative drag and commanded thrust tilt, but **not rigid-body attitude or 6-DOF**.

## Historical local artifacts (from earlier steps in this chat)
A mounted local ZIP was verified at `/mnt/data/rocket_landing_001_phase4.zip` (16 entries, approx. 1.4 MB ZIP). Contains: `rocket_landing_001/{README.md,config.yaml,reference.py,landing.py,planar.py,train_surrogate.py,train_planar_surrogate.py,predict.py,surrogate_metrics.json,planar_surrogate_metrics.json,surrogate.joblib,planar_surrogate.joblib,tests/test_reference.py,tests/test_landing.py,tests/test_planar.py,tests/test_planar_surrogate.py}`. The ZIP is **not** yet attached to the GitHub branch; the GitHub connector accepts UTF-8 files, not binary model artifacts. Retrieve the original ZIP from the originating chat and port source/tests; regenerate models from pinned dependencies and seeds, or publish binary artifacts using an approved model registry with hashes. Do not unpickle untrusted joblib files.

## Historical measurements (reported by prior local execution, not rerun in GitHub CI)
- Phase 1: naive controller failed with ~670.45 m/s vertical touchdown speed.
- Phase 2 1D: touchdown |vz|=1.51 m/s, descent 34.17 s, fuel used 2254 kg, 9 tests passed.
- Phase 3 2D at 8 m/s wind: |vz|=1.50 m/s, horizontal error 3.94 m, |vx|=0.17 m/s, descent 34.19 s, fuel used 2195.19 kg, final mass 15804.81 kg, 13 tests passed.
- Phase 4: ExtraTrees trajectory-summary surrogate, training 240 trajectories seed 42, independent test 80 seed 2026; MAE x_touchdown=3.91 m, vx_touchdown=0.301 m/s, vz_touchdown=0.0032 m/s, fuel=24.39 kg, time=0.147 s; safe touchdown rate only 57.5% of tested scenarios; 17 tests reportedly passed. These numbers are **historical local results, not evidence of 6-DOF validity**. Surrogate is not yet a temporal physics-neural hybrid.

## GitHub changes actually made
- PINNeAPPle draft PR https://github.com/PINNeAPPle-Labs/PINNeAPPle/pull/1743, branch `feat/rocket-landing-experiment-001`; committed `examples/aerospace/rocket_landing_001/README.md` and `config.yaml`.
- Engineering API draft PR https://github.com/PINNeAPPle-Labs/pinneapple-engineering-api/pull/340, same branch name; committed `docs/experiments/rocket_landing_001.md` and `rocket_landing_001.yaml`.
- PINNeAPPle issues: #1739 port runnable 1D/2D code, surrogate, tests; #1740 6-DOF, controller, VeriPhysics; #1741 temporal hybrid surrogate with OOD/uncertainty; existing related #394 powered descent.
- Engineering API issues: #338 experiment registry/workers/artifacts; #339 3D interactive Studio playback/scenario comparisons.

## Resume execution order
1. Port exact source and tests from local ZIP into PINNeAPPle branch; pin requirements; run pytest and compare numerical metrics to tolerances. Keep model binaries out of Git until provenance and license/size reviewed.
2. Integrate actual solver and trained surrogate into library APIs and model registry, plus VeriPhysics physical/validity checks.
3. Wire Engineering API experiment schema, worker dispatch, cancellation, progress, results, structured array/VTK trajectories and provenance; end-to-end test.
4. Studio: real 3D visualization with time scrub, telemetry, controls for wind/mass/altitude/engine, side-by-side solver/surrogate and uncertainty. Never substitute an animation for computed trajectories.
5. Improve controller robustness, then 6-DOF dynamics with rigid-body attitude and engine dynamics; train hybrid temporal surrogate, validate on held-out regimes and failure cases, calibrated OOD with solver fallback.
6. Run CI for both PRs; report only actually executed results; request review and merge after acceptance.

## Scientific acceptance gates
Touchdown |vz| <= 2 m/s, |vx| <= 2 m/s, |x| <= 5 m for planar prototype; attitude <=3 degrees only once 6-DOF exists. Nonnegative propellant, thrust bounds, mass balance, temporal convergence, out-of-domain rejection/fallback, train/test leakage prevention, physical vs surrogate errors and latency, Monte Carlo success/failure distribution. Safe landing and numerical solver success must be distinct statuses.

## Current state
**Only documentation/configuration/contract scaffolds committed, not functional full integration.** The two PRs are drafts. Do not claim source, trained models, workers, UI or CI are already integrated. All pending work has corresponding issues. This file is the canonical handoff for resuming from a new chat.
