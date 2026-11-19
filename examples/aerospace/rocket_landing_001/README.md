# Rocket Landing 001 — reproducible experiment (prototype)

Initial conditions: altitude 1500 m, mass 18000 kg, horizontal wind 8 m/s, initial vertical speed -60 m/s, nominal engine. Units SI, seed 42.

The runnable local prototype is supplied as `rocket_landing_001_phase4.zip` in the originating ChatGPT session. This repository entry documents the experiment contract; the prototype's Python modules and serialized model are **not yet vendored here**. Do not claim full integration or 6-DOF verification.

## Baselines
- Vertical 1D reference: mass depletion, gravity, thrust, RK4 and touchdown event.
- Planar 2D: state (x,z,vx,vz,m), wind-relative quadratic drag, commanded thrust tilt (not rigid-body attitude dynamics).
- Trained ExtraTreesRegressor: trajectory summary only; features [x0,z0,vx0,vz0,m0,wind], targets [touchdown_x,touchdown_vx,touchdown_vz,fuel_used,time].
- Independent data seeds 42 (240 training cases) and 2026 (80 test cases), reported test MAE: x=3.91 m, vx=0.301 m/s, vz=0.0032 m/s, fuel=24.39 kg, time=0.147 s. These are **historical local measurements, not CI-verified on this branch**.

## Acceptance and limitations
- Touchdown criteria: |vz|<=2 m/s, |vx|<=2 m/s, |x|<=5 m; mass >= dry mass, thrust limits, integration refinement.
- Out-of-bounds predictions must reject/fallback to solver; training range checks alone do not establish calibrated OOD detection.
- Historical test safe-landing rate: 57.5%; this is not production-ready.
- 6-DOF dynamics, robust controller, temporal physics-neural surrogate, uncertainty calibration, VeriPhysics integration, Engineering API workers and Studio visualization remain open.

## Suggested implementation location
`examples/aerospace/rocket_landing_001/`; use experiment manifest, model registry, dataset lineage and VeriPhysics gates. Keep raw trajectories separate from serialized models. Never deserialize untrusted joblib files.
