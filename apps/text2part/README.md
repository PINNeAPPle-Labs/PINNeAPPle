# Text2Part

**Describe a part in plain language; get a mesh (STL) or a parametric CAD file (STEP) and its measured size.**

Imported from PINNeAPPle-apps (`text_to_cad`). The LLM never writes CAD code: `pinneapple_llm.draft_part` asks it
for a recipe from a fixed menu (box, cylinder, sphere, plane, channel combined by boolean CSG, or a registered
CadQuery template), checks every name against the real registry and retries a refused draft. The part is built
and measured (`pinneapple_llm.measure_part`: volume, extents, watertight) so you can compare it with what you asked
for. Related app: none does text-to-geometry; `apps/physics_design` and `pp.design` go from a goal to a verified
design for their own templates.

## Run

```bash
pip install -e . -r apps/text2part/requirements.txt
ollama serve & ollama pull llama3.2:3b          # local LLM (default); nothing leaves the machine
cd apps/text2part && uvicorn text2part.api:app --port 8140    # http://localhost:8140/docs
```

`T2P_LLM_PROVIDER=anthropic|openai` with `T2P_LLM_MODEL` uses a hosted model; the key comes from
`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` in the environment, never from a request. Logins: `T2P_USER` /
`T2P_PASSWORD_HASH`; free use `T2P_FREE_GENERATIONS` (20 per caller and month); see `apps/_shared/README.md`.
Docker: `docker build -f apps/text2part/Dockerfile -t text2part .` (the container must reach an Ollama server on
its own 127.0.0.1, so it is not in `apps/deploy`).

| Method | Path | |
|---|---|---|
| POST | `/v1/generate` | `{"description": "...", "tier": "quick" \| "engineering", "fmt": "stl" \| "step"}` -> the file; headers `X-Measurements`, `X-Recipe`, `X-Attempts-Used`, `X-Reasoning` (JSON) |
| GET | `/api/meta`, `/health` | LLM in use, boolean engines, limits |

## Tests

`pytest tests/test_app_text2part.py tests/test_llm_draft_helpers.py` (the LLM stubbed: 3 + 6 pass; one real-LLM
test skips without an Ollama server with `llama3.2:3b`).

## Known limitations

- **Compound parts came out with wrong dimensions in the source app** (washer: 526 % volume error, plate with hole:
  98 %; plain boxes and cylinders within 0.6 %), reported in its TODO and not reproduced here (no LLM in this
  environment). What was measured here: a correct washer recipe builds within 0.6 % of its volume (faceting) with
  `manifold3d` installed, and **82 % too large without a boolean engine**, because `build_mesh` then concatenates
  the parts with only a warning (plate with hole: +11 %). `manifold3d` is in this app's requirements and
  `/api/meta` lists the engines found. An LLM choosing wrong primitives or sizes is not caught: compare
  `X-Measurements` with your request.
- The `engineering` tier and STEP output need `cadquery` (not in the requirements; `pip install "pinneapple[cad]"`).
  A STEP of a quick-tier recipe is a triangulated shell, not a B-rep.
- A small local model often needs several attempts; generation is synchronous (seconds to a minute per call).
