# PhysicsCopilot

**A physics problem in plain language -> a trained PINN and the guardrail's re-computed report of it.**

Imported from PINNeAPPle-apps (`physics_copilot_api`). One job runs
`pinneapple_analysis.verification.pipeline.analyze`: dimensional analysis of the parameters you give, the LLM picks
and parametrises a registered preset (`pinneapple_llm.draft_problem`; an unknown preset or kwarg name is rejected),
a PINN trains on it (`pp.solve_pde`, with the library's boundary geometry for tagged presets) and
`PhysicsGuardrail` re-evaluates the residual on fresh points. The report lists every check with its value.

This is the minimal front of the same pipeline `apps/verified_physics` serves with candidate architectures,
uncertainty, interventions and the analysis modules; keep using that one when you need more than one solve.

## Run

```bash
pip install -e . -r apps/physics_copilot/requirements.txt
ollama serve & ollama pull llama3.2:3b
cd apps/physics_copilot && uvicorn physics_copilot.api:app --port 8142
curl -X POST localhost:8142/v1/solve -H 'Content-Type: application/json' \
     -d '{"description": "1D viscous Burgers equation with nu=0.01", "epochs": 500}'
curl localhost:8142/v1/jobs/<id>          # status, progress, result
```

`COPILOT_LLM_PROVIDER` / `COPILOT_LLM_MODEL` (default `ollama` / `llama3.2:3b`; hosted providers take their key
from `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`), `COPILOT_MAX_EPOCHS` (3000), `COPILOT_FREE_SOLVES` (20 per month),
`COPILOT_MAX_JOBS` (1). Not in `apps/deploy` (the LLM call goes to 127.0.0.1:11434 inside the process).

## Tests

`pytest tests/test_app_physics_copilot.py`: the rejection path with the LLM stubbed passes; the full stubbed solve
(10 epochs of a 64 x 4 network, about 3 minutes on a loaded 4-core machine) runs with `PINNEAPPLE_SLOW=1`; one
real-LLM test skips without an Ollama server. The pipeline itself: `tests/pinneapple_analysis/test_verification_pipeline.py`.

## Known limitations

- **The chosen preset is not checked against the description.** The guardrail re-checks the physics of the preset
  the LLM picked, never whether it is the right one: the source app's TODO records a heat-conduction request drafted
  as Burgers and reported trustworthy. Not reproduced here (no LLM), but nothing in the pipeline compares the two;
  each result carries `"not_checked"` saying so, and `preset` / `reasoning` must be read before using the result.
- A short training rarely passes the 1e-2 residual threshold; "not trustworthy" is then the correct report.
- The LLM drafts within PINNeAPPle's registered presets only; a problem none covers is rejected.
