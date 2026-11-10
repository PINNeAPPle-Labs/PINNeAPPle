# LLM code-training pipeline — durable handoff

**Canonical work:** [draft PR #1602](https://github.com/PINNeAPPle-Labs/PINNeAPPle/pull/1602) against `main`, branch `feat/llm-code-training-pipeline`.

**Purpose:** Create a native PINNeAPPle pipeline for generating supervised LLM training examples from PINNeAPPle Python source and reviewed experiment records, then fine-tune a local Hugging Face causal LM with LoRA. No hosted LLM API is required by these modules.

## Implemented in the PR (code committed, not validated end to end)

| File | Responsibility |
| --- | --- |
| `pinneapple_llm/code_training.py` | AST extraction, train/validation manifests and checksums, CLI: `generate`, `ingest-evidence`, `assemble`, `train`, `benchmark` |
| `pinneapple_llm/finetune.py` | Local LoRA supervised fine-tuning, response-only labels, masked padding, held-out validation metrics |
| `pinneapple_llm/verified_examples.py` | Import reviewed execution examples with exit-code, evidence hash, path and provenance gates |
| `pinneapple_llm/corpus.py` | Mix source-code and reviewed-execution records into reproducible, checksummed corpus |
| `pinneapple_llm/benchmark.py` | Score precomputed model outputs by exact match and API-token coverage (proxies only) |
| `pinneapple_llm/__init__.py` | Public exports |
| `tests/pinneapple_llm/test_code_training.py` | Source extraction, checksums, response-only labels, collator |
| `tests/pinneapple_llm/test_verified_examples.py` | Evidence acceptance, tampering, review gates |
| `tests/pinneapple_llm/test_corpus_benchmark.py` | Corpus and offline benchmark regression tests |
| `.github/workflows/llm-code-training.yml` | Focused PR CI |
| `docs/llm_code_training.md` | Usage, constraints, examples and limitations |

## Commands to reproduce after checkout of PR branch

```bash
python -m pip install -e ".[dev,finetune]"
python -m pytest -q tests/pinneapple_llm/test_code_training.py tests/pinneapple_llm/test_verified_examples.py tests/pinneapple_llm/test_corpus_benchmark.py
python -m pinneapple_llm.code_training generate --root . --output-dir artifacts/llm_code_dataset
# Once real executions have been reviewed and their evidence files hashed:
python -m pinneapple_llm.code_training ingest-evidence --input artifacts/reviewed_executions.jsonl --output artifacts/verified_training.jsonl --root .
python -m pinneapple_llm.code_training assemble --code-dataset-dir artifacts/llm_code_dataset --evidence-jsonl artifacts/verified_training.jsonl --output-dir artifacts/llm_mixed_dataset
# Requires accessible model weights and sufficient local hardware:
python -m pinneapple_llm.code_training train --dataset-dir artifacts/llm_mixed_dataset --base-model YOUR_ACCESSIBLE_HF_CAUSAL_MODEL --output-dir artifacts/llm_code_adapter
# Requires precomputed model predictions:
python -m pinneapple_llm.code_training benchmark --tasks artifacts/benchmark/tasks.jsonl --predictions artifacts/benchmark/adapter_predictions.jsonl --report artifacts/benchmark/adapter_report.json
```

## Outstanding work, tracked in GitHub

1. [#1606](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1606) — Run CI/tests/CLI, fix bugs, review and merge PR.
2. [#1607](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1607) — Real LoRA training, logs, artifacts and baseline comparison.
3. [#1608](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1608) — Independently recompute physics metrics with VeriPhysics.
4. [#1609](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1609) — Sandboxed executable Physics AI benchmarks and inference harness.
5. [#1610](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1610) — Security, licensing and semantic data-leakage audit.
6. [#1601](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1601) — Original umbrella issue.

## Honest state and limitations

- No successful GitHub Actions run, actual GPU LoRA fine-tune, or executed Physics AI benchmark has been confirmed in this conversation.
- The evidence importer checks integrity and human review, **not** independent numerical physics.
- Offline exact-match/API coverage scores are not physical validation.
- No model weights, trained adapter, reviewed execution artifacts or dataset corpus have been claimed as generated.
- The PR remains **draft** and **unmerged** intentionally. Work is preserved on the remote GitHub branch and in issues.
- Before merging, review code correctness, optional dependencies, train/validation leakage and security.
- Generated datasets may include sensitive or third-party code: review rights and redaction before distribution.

## Continuation instructions

Open PR #1602 and this file first. Avoid reimplementing already committed modules. Start with #1606, then #1607; keep #1608–#1610 open until independent scientific validation and security reviews are completed. If the PR branch changes, update this handoff document and the PR body with the actual status and evidence.
