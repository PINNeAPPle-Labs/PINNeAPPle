- **The Security workflow parses again**: a step name in `.github/workflows/security.yml` held an unquoted
  `#96: ...`, which YAML reads as a mapping, so GitHub refused the whole file ("This run likely failed because of a
  workflow file issue") and none of the scans ran -- bandit, pip-audit, gitleaks, hadolint, Trivy, the SBOM and the
  dependency/licence audit were all silently absent from every push since. The name is quoted, and the file parses.
