# VerifiedPhysics

**The LLM proposes the physics; PINNeAPPle executes it and re-checks it mechanically, with the evidence attached.**

Imported from PINNeAPPle-apps (`physics_verification_engine`). All of its `core/` modules were already in the
library as `pinneapple_analysis.verification` (dimensional analysis, geometry and mesh intelligence, solver
orchestration, uncertainty and inverse problems, causal interventions and model discrepancy, the knowledge graph,
provenance); the orchestration loop `core/pipeline.py` is now `pinneapple_analysis.verification.pipeline`
(`analyze`, `rank_key`), and `run_intervention`, which imported it from the private app, uses the library copy.
The app's eight test files are the library's tests (`tests/pinneapple_analysis/test_verification_*.py`).

How it relates to the other apps: `apps/validation_lab` checks results and models you bring and keeps an evidence
ledger; `apps/evidence_report` packages evidence into reports; `apps/mesh_quality` is the full volume-mesh quality
service. VerifiedPhysics is the LLM-to-trained-PINN pipeline and the analysis modules that came with it.

| Method | Path | |
|---|---|---|
| POST | `/v1/analyze` (job) | description -> preset -> PINN(s) -> guardrail -> physics-aware pick; optional `quantify_uncertainty`, `reference_x`/`reference_y` (discrepancy) |
| POST | `/v1/intervention` (job) | re-run with one preset parameter changed, % changes, flagged unreliable when a run is untrustworthy |
| POST | `/v1/inverse_problem` (job) | steady 1-D heat conduction only (others answer with the scope error) |
| GET | `/v1/runs/{id}` | job status and result |
| GET / POST | `/v1/solver_families`, `/v1/recommend_solver` | which solver families run here; documented-scope match |
| GET | `/v1/knowledge/stats`, `/preset/{name}`, `/phenomenon/{name}`, `/verified_equations` | graph from the live preset registry and validation tests |
| POST | `/v1/classify_geometry`, `/v1/assess_mesh_quality` | STL / OBJ / PLY upload |

## Run

```bash
pip install -e . -r apps/verified_physics/requirements.txt
ollama serve & ollama pull llama3.2:3b           # for /v1/analyze and /v1/intervention
cd apps/verified_physics && uvicorn verified_physics.api:app --port 8143
```

`VPHYS_LLM_PROVIDER` / `VPHYS_LLM_MODEL`, `VPHYS_MAX_EPOCHS` (3000), `VPHYS_FREE_RUNS` (20), `VPHYS_FREE_CHECKS` (500);
see `apps/_shared/README.md`. Not in `apps/deploy` (the LLM is called at 127.0.0.1:11434 inside the process).

## Tests

- App: `pytest tests/test_app_verified_physics.py` -- 4 pass (knowledge graph, solver routes, mesh routes, the
  rejection path and the inverse-problem scope), 1 training test with `PINNEAPPLE_SLOW=1`.
- Library (the source app's tests, imports pointed at the library): `pytest tests/pinneapple_analysis/test_verification_*.py`
  -- 88 collected: 81 pass, 7 skip (4 need Ollama with `llama3.2:3b`, 3 are the ~90 s inverse-problem recovery,
  run with `PINNEAPPLE_SLOW=1`, which pass). Three knowledge-graph tests pinned the counts of the registry of their
  day (31/57 presets cited, burgers unverified); the registry has grown (72 presets, burgers now checked in
  `tests/test_pinn_validation.py`), so they assert lower bounds and the current burgers status. Two
  geometry tests need `manifold3d` for boolean cuts and skip without it.
- The source TODO's "2 failing + 5 errors in test_uncertainty_inverse.py" does not reproduce: all pass here.

## Known limitations

- **The drafted preset is not checked against the description** (see `apps/physics_copilot/README.md`).
- Every training route is a job (one at a time by default); a 64 x 4 PINN takes minutes on a shared CPU.
- Inverse problems: steady 1-D heat conduction only. Solver orchestration detects availability; it does not run
  FEniCS / OpenFOAM through this API. The knowledge-graph numbers are those of the current registry.
- Numbers quoted in the source README (e.g. k recovered within 1.9 %, discrepancy RMSE 0.1345 -> 0.0158, intervention
  percentages against a live Llama) are reported there and were not re-measured here, except through the tests above.
