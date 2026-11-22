# Lab run integrity receipts (issue #38 — partial)

The lab runner now writes `evidence-manifest.json` **after** `experiment.json`
and the run index are written. The manifest records relative file paths, sizes
and SHA-256 digests for run artifacts, excluding the manifest itself.

When a cached run has a manifest, the runner verifies it before reusing the
cached result. Missing or altered files invalidate the cache and trigger a
fresh run through the existing runner path. Legacy runs without a manifest
retain their prior cache behavior.

The receipt is **integrity evidence only**. It does not imply scientific
correctness, independent validation, reproducibility across platforms, or
authorization for engineering deployment.

Limitations: no cryptographic signing, external timestamping, trusted storage,
physics metric attestation, or independent reviewer approval. Manifest
generation has file-count and per-file size limits; oversized run outputs
require a separate artifact-store integration. No tests or GitHub Actions were
run for this PR.
