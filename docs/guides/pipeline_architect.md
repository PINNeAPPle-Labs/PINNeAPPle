# Pipeline Architect (planning-only MVP)

Input: text description and structured answers. Output: versioned DAG, questions, candidate solver/model, validation gates, JSON and Mermaid.

No solver, CAD, meshing, model training or LLM calls are executed. All candidates require expert verification. Runtime estimates require user-provided benchmark measurements; never fabricate costs.

Core: `pinneapple_pipeline_architect.py` exposes `Problem`, `plan`, `to_json`, `to_mermaid`.

Engineering API: `app/pipeline_architect/router.py` is an optional FastAPI router; register it in the existing app and install the pinned PINNeAPPle dependency exposing the core module. The router is not mounted automatically.

Next increments: (1) map verified Capability Registry entries to step availability; (2) Ollama structured extraction with Pydantic validation and deterministic fallback; (3) question/answer state machine and user approvals; (4) solver benchmark-based hardware cost model; (5) React Flow editor, Mermaid/JSON export, plan history; (6) tests for physics coverage, domain/architecture selection and security.

Important: generated recommendations are proposals, not physically validated results. Do not equate a proposed VeriPhysics gate with a passed check.
