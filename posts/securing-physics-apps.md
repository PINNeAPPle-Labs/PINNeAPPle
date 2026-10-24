Engineering web apps take uploads, formulas and solver settings from strangers, then spend CPU on them. That is a bigger attack surface than most physics code is written for.

We audited the 34 PINNeAPPle web apps and their Docker deployment against OWASP Top 10 and ASVS, with bandit, semgrep, pip-audit, gitleaks, hadolint and Trivy, then fixed what we found in three pull requests: the app layer, the deployment and the supply chain.

Practices that carry over to any engineering app:
- User formulas (boundary values, dataset constraints) go through a small AST interpreter, never eval.
- Uploaded archives are checked for member names, links, total size and compression ratio; request bodies are counted while streaming.
- Per-caller rate limits on solver and inference routes, job time and memory limits, login lockout, hashed passwords.
- A CSP built from each app's own inline scripts, no script 'unsafe-inline', and a cross-site POST check.
- Containers run as uid 10001 with a read-only root filesystem, all capabilities dropped, no-new-privileges and a pids limit.
- CI on every pull request and weekly: bandit, pip-audit, gitleaks, hadolint, Trivy on base images, CycloneDX SBOMs; Dependabot for pip, Actions and base images pinned by digest.

Measured:
- A 15-property behaviour probe went from 3/15 to 15/15.
- Trivy "runs as root" findings: 33 to 0; bandit high: 2 to 0; semgrep eval-detected: 4 to 0.
- 169 security tests pass in a local run (1 skipped for a missing optional library).

Trust: the fixes come with tests, and the audit records before and after numbers from the same tools and scope. Some lower-severity items stay open and are tracked there; hash-pinned requirements are not done yet.

https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main/docs/security/server.md

#AppSec #DevSecOps #Docker #SupplyChainSecurity #FastAPI #ScientificComputing #OpenSource

---
## Media

1. `media/securing-physics-apps/01_scope.png` — Physics apps take formulas, files and solver budgets from strangers: scope of the audit and the three fix PRs.
2. `media/securing-physics-apps/02_findings.png` — What the audit found and what was fixed: the critical and high-severity classes of issues.
3. `media/securing-physics-apps/03_app_layer.png` — App layer checklist: bounded inputs, bounded compute, browser headers, logins, errors.
4. `media/securing-physics-apps/04_compose.png` — Container hardening in one Compose anchor: non-root, read-only, no capabilities, pids/CPU limits.
5. `media/securing-physics-apps/05_ci.png` — The Security workflow: what fails the build, SBOMs, the weekly hardened image check, Dependabot.
6. `media/securing-physics-apps/06_measured.png` — Before and after with the same tools and scope, plus the security test run.

## Sources

- 34 apps; OWASP Top 10 (2021) and ASVS 4.0; tool list -> `docs/security/audit.md` (Scope paragraph; "Tool results" table)
- 1 critical (C1) and 5 high (H1–H5), all "Fixed" -> `docs/security/audit.md` "Findings, ranked" (Critical and High tables)
- behaviour probe 3 / 15 -> 15 / 15 -> `docs/security/audit.md` "After the changes (measured)", row **total**
- Trivy DS-0002 (image runs as root) 33 -> 0 -> `docs/security/audit.md` Tools table, row "trivy config"
- bandit high 2 -> 0 -> `docs/security/audit.md` Tools table, row "bandit (same scope)"
- semgrep eval-detected 4 -> 0 -> `docs/security/audit.md` Tools table, row "semgrep" ("eval-detected 4 → 0")
- Playground base image 7 critical -> 0 critical -> `docs/security/audit.md` Tools table, row "trivy, Playground base image"; M10
- uid 10001, CapEff 0, NoNewPrivs 1, read-only `/` -> `docs/security/audit.md` Tools table, last row; `apps/deploy/docker-compose.yml` lines 14–25 (`x-app-hardening`)
- pids_limit 512, cpus 2, tmpfs 1g defaults -> `apps/deploy/docker-compose.yml` `x-app-hardening`
- CI gates (bandit -lll -ii, pip-audit, gitleaks, hadolint --failure-threshold error, trivy CRITICAL --ignore-unfixed --exit-code 1, SBOMs, weekly image job) -> `.github/workflows/security.yml`
- Dependabot ecosystems pip, github-actions, docker, docker-compose; weekly -> `.github/dependabot.yml`
- Caddy pinned by digest -> `apps/deploy/docker-compose.yml` line 823, `apps/playground/Dockerfile` line 14
- 169 passed, 1 skipped -> local run on main (e716a065): `pytest tests/test_app_security.py tests/test_deploy_security.py tests/pinneapple_security/test_web_hardening.py` (skip: argon2 not installed). Wider CI selection (security.yml `tests` job): 239 passed, 2 skipped
- "hash-pinned requirements not done" -> `docs/security/audit.md` L7 ("Hash-pinned requirements were not generated")
