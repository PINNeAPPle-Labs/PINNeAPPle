# Training a code-aware local LLM with PINNeAPPle

PINNeAPPle already provides local LoRA fine-tuning in
`pinneapple_llm.finetune`, based on recorded conversations.
This pipeline adds a source-code dataset builder inside the same package.

## Generate source-grounded instruction data

Run from the PINNeAPPle repository root:

```bash
python -m pinneapple_llm.code_training generate \
  --root . --output-dir artifacts/llm_code_dataset \
  --package pinneapple \
  --package pinneapple_core \
  --package pinneapple_physics \
  --package pinneapple_llm
```

It writes `train.jsonl`, `validation.jsonl`, and `manifest.json`.
No hosted LLM service or GPU is needed for this step. Samples contain
a public Python function's name, signature, docstring, source snippet, origin
path and source checksum. Splitting is deterministic and grouped by source file
to reduce train/validation overlap. The manifest records the Git revision,
row counts, settings and dataset SHA-256 checksums.

**Important:** source text and docstrings are evidence of what the code says,
not proof that the function runs or that its physics is valid. A separate
scientific verification corpus should use actual PINNeAPPle solver and
VeriPhysics execution artifacts rather than invented answers.

## Fine-tune an existing causal language model

```bash
pip install "pinneapple[finetune]"
python -m pinneapple_llm.code_training train \
  --dataset-dir artifacts/llm_code_dataset \
  --base-model YOUR_ACCESSIBLE_HF_CAUSAL_MODEL \
  --output-dir artifacts/llm_code_adapter
```

This validates dataset checksums and delegates to the existing
`pinneapple_llm.finetune.finetune_lora` backend using
Hugging Face Transformers and PEFT. The base model must be available
locally or through your Hugging Face credentials under its own license.
Training compute, hardware compatibility and model access are not provided
by the package. The train command does not upload data to any LLM API.

## Quality and known limitations

- Inspect train/validation outputs for secret material, privacy, third-party
  licensing, inaccurate docstrings and security-sensitive source before use.
  The built-in secret check is a limited pattern-based safeguard, not a full audit.
- Current code samples teach explanation of available APIs, not verified
  engineering problem solving or agent tool use.
- The existing LoRA backend trains on the train split but does not yet score
  the held-out validation split. Do not claim a benchmark score from its loss.
- The current backend also computes language-model loss over the prompt; a
  completion-only masking improvement is planned separately.
- The generator's file-level split is weaker than held-out tasks/projects.
  For scientific reasoning claims, benchmark held-out PDE families, physical
  invariants, API/tool usage and refusal behavior.
- Dataset artifacts should normally stay outside source control and must be
  reviewed before distribution. Public repository code and its dependencies
  may have different licensing/provenance constraints.

## Next steps

Add a verified execution-example generator, prompt/completion-only loss
masking, validation evaluation, vulnerability/secret audits and held-out
Physics AI benchmarks. Keep these gates separate from code extraction.
