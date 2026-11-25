- **The secret scan is green again: the fake key a test wrote is allow-listed**: `gitleaks detect` (the whole history,
  on every push) found `generic-api-key` in `tests/pinneapple_llm/test_code_training.py:43`, the imaginary
  `sk-abcdefghijklmnopqrstuvwxyz1234` that `test_code_dataset_rejects_secrets` writes into a temporary package to prove
  the code-dataset generator refuses credentials. The test file has since left the tree, but its commit is in the
  history, so the finding could never disappear on its own: the `Security` workflow failed at that step, which skipped
  hadolint and Trivy and left the SBOM step without the images to read. The key is allow-listed in `.gitleaks.toml`
  beside the other placeholders; the scan of that commit reports no leaks.
