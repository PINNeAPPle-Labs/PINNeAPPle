# Global Physics AI Atlas — initial foundation

The Atlas is a **global** evidence-based scientific knowledge graph, not a catalog restricted to PINNeAPPle. Public scientific methods, equations, papers, datasets, repositories and benchmarks should be connected to corresponding PINNeAPPle capabilities, source files, tests, PRs, issues and experiment evidence.

## First implementation

The standalone `pinneapple_atlas` module provides typed capability and evidence records, evidence validation, deterministic comparison and JSON-ready graph nodes/edges. It intentionally does **not** claim automatic ingestion, frontend integration, live repository inventory, or scientific verification.

## Coverage rules

- Public knowledge and PINNeAPPle implementations are independent provenance domains.
- No local inventory means `not_assessed`, not missing or unsupported.
- Planned work is not implemented work.
- Implemented and validated claims require linked evidence; a URL alone is not sufficient to prove a test passed, so reviewers must assess the evidence.
- Comparison is by normalized capability key, with optional dimension tags. Future work must match dimensionality, geometry, PDE assumptions, boundary conditions, datasets, solver fidelity and validation protocols before asserting equivalence.
- Do not generate percentage coverage until a denominator, capability taxonomy and assessment methodology are established.

## Next phases

1. Define versioned scientific ontology (physics domains, PDEs, methods, data, benchmarks, validity and evidence types).
2. Add incremental public connectors (OpenAlex, Crossref, arXiv, GitHub, Hugging Face, Zenodo), honoring licenses, quotas and provenance.
3. Audit actual PINNeAPPle code, docs, tests, issues, PRs and execution results into a versioned inventory.
4. Add deduplication, DOI resolution, confidence, contradiction handling and review workflow.
5. Expose search, graph, comparison and gap APIs through Engineering API.
6. Build interactive Physics Graph and Coverage Explorer in Engineering Studio.
7. Integrate reproducible experiment planning, VeriPhysics evidence and contributor issue proposals.

## Run initial tests

`python -m pytest tests/test_atlas_core.py -q`
