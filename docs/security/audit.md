# Security audit: the PINNeAPPle web apps and their deployment

Scope: the FastAPI apps under `apps/` (34 apps plus the static Playground and Zoo space), the shared glue
`apps/_shared/appkit.py` and the library code it runs (`pinneapple_core.service`, the parsers and solvers the apps
call), the Lab catalogue server (`pinneapple_lab/server.py`), and the deployment in `apps/deploy` (Docker Compose,
Caddy, the app Dockerfiles). Reference: OWASP Top 10 (2021) and ASVS 4.0 levels 1–2. Audit date: 2026-10-18,
commit `ee4b770b` (main).

Method: an inventory of every app's routes, uploads, process calls, deserialisation, path handling and outbound
requests, read from the code (summarised below), plus the tools in the next section. Every finding marked
**measured** was reproduced by a test or a scan; the others were read from the code and are marked as such.

## Tool results (baseline, before any change)

| Tool | Scope | Result |
|---|---|---|
| bandit 1.8 | `apps/`, `pinneapple_security/`, `pinneapple_core/service/` | 24 findings: **2 high** (B324 SHA-1, used for design IDs), 5 medium (3× B608 SQL built from a fixed list of `?` placeholders, B610, B301 pickle of a server-written cache), 17 low |
| semgrep 1.181 (semgrep-rules: python lang, fastapi, jwt, requests, cryptography, dockerfile) | apps + the evaluators they call | 241 findings: 52 error (**32× dockerfile missing-user**, 16 useless-inner-function, …), 116 warning (incl. **4× eval-detected**, 2× avoid-pickle, 2× SHA-1), 73 info |
| pip-audit 2.9 | union of `apps/*/requirements.txt`, resolved to current releases | **0** known vulnerabilities |
| gitleaks 8.21 | whole git history (502 commits) | 39 hits, **all false positives**: the placeholder `curl -u USER:PASSWORD` in three app UIs and `key="axial_compressor_stage_3d"` in `pinneapple_veriphysics/example_cases.py`. No committed secret found |
| hadolint 2.12 | 35 app Dockerfiles | 37: 33× DL3013 (unpinned pip), 3× DL3008 (unpinned apt), 1 info |
| trivy 0.69 config | `apps/` | **33 high: DS-0002, image runs as root** (all app Dockerfiles but the Simulation Runner and the GUI agent) |
| trivy 0.69 image | base images | `python:3.11-slim`: 0 critical, 46 high (2 with a fix); `caddy:2`: 0 critical, 4 high; **`caddy:2.8.4-alpine` (Playground): 7 critical, 86 high**, 92 % fixable |

Not run here: Trivy on the built app images. `download.pytorch.org` (the CPU torch wheel every app image installs)
is blocked from this environment and the CUDA wheel from PyPI exceeded the disk; the base images and the resolved
Python dependencies were scanned instead, and the `Security` workflow's weekly `image` job builds and scans one app
image in CI.

## Inventory

**Shared to every app** (`apps/_shared/appkit.py`, before this change):

- Login: optional HTTP Basic (`<PREFIX>_USER` / `<PREFIX>_PASSWORD`, plain text in the environment, constant-time
  compare); with neither set, every route is open. Any login is admin. No lockout, no rate limit.
- Caller identity (job and upload ownership, monthly quotas): API key, else login, else `ip:` + the **last
  `X-Forwarded-For` entry, believed from any peer**.
- Runtime (`install_runtime`): SQLite job queue (thread or process executor, optional time limit, no resource
  limits), API keys (random, stored as SHA-256, scopes stored but not enforced), usage metering, uploads (streamed
  size cap, name reduced to `[A-Za-z0-9._-]`, suffix allow-list, stored under the data volume, not the web root).
- Static files: `/vendor` (three.js), `/studio`, `/shared`, each app's `/static`. No security headers, no CORS
  middleware anywhere (browsers default to same-origin), no CSRF protection, no session cookies (Basic auth and
  `X-API-Key` only).
- Errors: FastAPI's default 500 has no trace; many routes return `f"{type(e).__name__}: {e}"` with 422; job logs
  carry the last 40 traceback lines to the job owner.

**Per app** (prefix · runtime · uploads · expensive work and its limit · process calls · deserialisation · outbound):

| App | Prefix | Uploads (default limit) | Expensive work, limit | Notable |
|---|---|---|---|---|
| aero_optimizer | ADO | none | optimisation, GNN inference: BusyLimiter `ADO_MAX_HEAVY`; mesh exports unthrottled | `torch.load(weights_only=False)` on the operator's `ADO_MODEL` path |
| anomaly_detector | ANOMALY | csv/parquet/json via runtime (50 MB) | analyses, benchmarks: job queue | model JSON stored by `Path(uri).name` |
| benchmark_arena | ARN | runtime (100 MB) | arena runs: job queue, budget cap | **dataset `constraints` evaluated with `eval`** |
| calibration_lab | CALIB | tables via runtime | job queue | outbound link only (`CALIB_IDENTIFIABILITY_URL`, env) |
| control_lab | CONTROL | json via runtime | job queue | — |
| data_health | EDH | table (50 MB) | BusyLimiter | — |
| data_standardizer | EDS | up to 8 files (80 MB) | BusyLimiter | — |
| design_requirements | UDR | PDFs/docs (40 MB, 8 files) | OCR (`UDR_OCR_MAX_PAGES`), local LLM via Ollama: BusyLimiter | LLM output fills form fields only, runs nothing |
| engineering_optimizer | EDO | runtime (20 MB) | job queue | external solvers only from the operator's `EDO_SOLVERS` (argv, no shell, timeout) |
| evidence_report | EVR | files (200 MB) | BusyLimiter | report HTML escaped; `urls` are recorded, not fetched |
| guiagent | GUIAGENT | workflow zip, self-hosted mode only (50 MB) | FreeCAD/OpenFOAM runs: job queue | zip checked per member; LLM picks among committed workflows |
| heatsink_sizer | HSS | none | BusyLimiter `HSS_MAX_SIZING` | — |
| identifiability_lab | IDENT | table (runtime cap) | job queue, row/parameter caps | — |
| interop_hub | IOP | files/zips (500 MB) | BusyLimiter | zips via `cae.upload.expand` (count and size caps, **no ratio, no name check**); pickle of a server-written cache (`sid.isalnum()`) |
| inverse_heat | IHL | none | PINN training threads, `IHL_MAX_JOBS` | jobs without owner (random 48-bit ids) |
| inverse_studio | INVS | table | job queue | — |
| mesh_quality | MQA | files/zips (200 MB) | BusyLimiter | zips via `expand` |
| meshless_lab | MESHLESS | point cloud (20 MB) | job queue, node and budget caps | **user formulas → `sympy.sympify` behind a string blacklist** |
| model_gateway | MDL | onnx/zip/json (100 MB) | inference, metered | **surrogate zip → `pickle.load`, "sha256-checked" against a card in the same zip** (admin only); remote refs: server fetches a user URL (admin only) |
| model_lineage | LIN | files/zips (300 MB) | BusyLimiter | zips via `expand` |
| observatory | OBS | none | BusyLimiter for the heavy routes; `/api/sky`, `/api/satellites` unthrottled | outbound to fixed hosts only; **unbounded result cache** |
| pcb_hotspot | PCB | none | BusyLimiter | — |
| physics_design | PDO | none | job queue, evaluation caps | — |
| rom_lab | ROM | npz/zip/vtu/frd (200 MB) | job queue | `np.load(allow_pickle=False)` throughout; resolved-path check on job files |
| sensitivity_lab | SEN | tables, onnx (50 MB) | job queue; **`/api/design` synchronous with unbounded `n_sobol`**; `?wait=` sleeps inside async handlers | ONNX parsed by onnxruntime |
| simulation_campaign | CAMPAIGN | tables, zip, inp, yaml (100 MB) | job queue, 5000 cases | **base-case zip `extractall` without size limit; OpenFOAM edit `file` joined unchecked; user `str.format` templates** |
| simulation_comparator | CMP | files (300 MB) | BusyLimiter | whole upload read before the size check |
| simulation_metadata | SMD | files (100 MB) | BusyLimiter | same |
| simulation_preflight | PFL | files (200 MB) | BusyLimiter | same; gzipped OpenFOAM files decompressed without a cap |
| simulation_runner | RUNNER | tables (20 MB) | job queue, 60 cases | `Popen(argv)` without shell; public mode runs 3 templates with range-checked numbers; `RUNNER_SELF_HOSTED=1` runs any workflow by design |
| surrogate_studio | SUR | tables (100 MB) | job queue, budget cap | pickled surrogate written and read by the server only |
| twin_runtime | TWIN | tables, model JSON | job queue | **live twins have no owner**; MQTT/OPC-UA/DB URLs from the environment only |
| uncertainty_lab | UQL | tables, onnx (50 MB) | job queue; `?wait=` in an async handler | — |
| validation_lab | VAL | files/zips (200 MB) | BusyLimiter | **boundary-value formulas evaluated with `eval`**; **config paths may point outside the upload**; `os.chdir` per run |
| playground, zoo_space | — | static | — | Playground CSP allows `'unsafe-inline'` and all of jsDelivr |
| Lab catalogue (`pinneapple_lab`) | LAB | none (read-only) | — | own Basic auth, plain password, no lockout |

**Deployment** (before): every image but two ran as root; no `read_only`, `cap_drop`, `no-new-privileges` or
`pids_limit` in Compose; `mem_limit` set; healthchecks in the Dockerfiles; only Caddy published ports (80/443),
the apps `expose` theirs; in existing-proxy mode the apps join the proxy network `pinneapple_physai_net` and the
project network. Caddy: HSTS without subdomains, nosniff, Referrer-Policy, no `Server` header; body limits on 29
of 36 sites; no access log; TLS defaults. Secrets in `.env` (git-ignored; `.env.example` holds placeholders
`demo` / `change-me`).

## Findings, ranked

Severity follows OWASP risk rating (impact × likelihood) for the default deployment. "App PR", "Deploy PR" and
"CI PR" are the pull requests that address them; tests are named so each claim can be re-run.

### Critical

| ID | Finding | OWASP / ASVS | Status |
|---|---|---|---|
| C1 | **Code execution through formula evaluation.** `pinneapple_data.audit` (benchmark arena dataset `constraints`) and `pinneapple_check.checks.results.evaluate_expr` (Validation Lab boundary values from an uploaded config) ran user strings with `eval(..., {"__builtins__": {}})`, which attribute access escapes (`().__class__.__base__.__subclasses__()`). Reachable without login when the app has none configured. **Measured**: before, `constraint_mask("().__class__.__base__.__subclasses__().__len__() > 0", ...)` returned `True` (the list of every loaded class was reached, the first step of the known escapes); after, it raises `ExpressionError`. The same for `evaluate_expr`. | A03 Injection · V5.2.4 | Fixed (App PR): both go through the existing AST interpreter `pinneapple_data.contracts.expressions`, never `eval`. Test `test_constraint_and_boundary_evaluators_use_the_interpreter` |

### High

| ID | Finding | OWASP / ASVS | Status |
|---|---|---|---|
| H1 | **Arbitrary file overwrite in the Campaign Planner**: `openfoam_edits[].file` is joined to the case directory without a check (`../`, absolute paths), then read, edited and written. **Measured** by `test_openfoam_edit_outside_the_case_refused`. | A01 · V12.3.1 | Fixed (App PR) |
| H2 | **Caller spoofing via `X-Forwarded-For`** when an app is reachable without the proxy (or from another container on the shared proxy network): the header chose the `ip:` principal, giving access to another anonymous caller's jobs and uploads and resetting quotas. **Measured** by `test_forwarded_for_trusted_only_from_the_proxy`. | A01 · V4.1.3 | Fixed (App PR): believed only from trusted proxies (loopback/private by default) |
| H3 | **Pickle code execution in the Model Gateway**: a surrogate `model.zip` is unpickled; its "sha256 check" compares the pickle with a hash from the same zip. Admin only. **Measured** by `test_pickled_surrogate_upload_off_by_default_and_scanned`. | A08 · V5.5.3 | Fixed (App PR): off unless `MDL_ALLOW_PICKLE_UPLOAD=1`, and then scanned for code-running globals first |
| H4 | **Containers run as root** (33 of 35 images; trivy DS-0002 ×33, semgrep missing-user ×32), with a writable root filesystem and all default capabilities. | A05 · V14.1 | Fixed (Deploy PR): uid 10001, read-only root, `cap_drop: ALL`, `no-new-privileges`, pids/CPU limits. **Measured** in a container (below) |
| H5 | **No brute-force protection and plain-text passwords**: unlimited Basic-auth attempts; passwords in plain text in the environment. | A07 · V2.2.1, V2.4.1 | Fixed (App PR): lockout per IP and user (429), scrypt/argon2id hashes (`<PREFIX>_PASSWORD_HASH`), audit events. Tests `test_login_lockout_after_repeated_failures`, `test_hashed_and_legacy_plain_passwords` |

### Medium

| ID | Finding | Status |
|---|---|---|
| M1 | Cross-site request forgery: browsers resend cached Basic credentials, so a page elsewhere can POST to an app (start jobs, delete uploads, revoke keys). A01 · V4.2.2 | Fixed (App PR): Fetch-Metadata/Origin check on state-changing requests, `test_cross_site_post_refused` |
| M2 | No security headers from the apps (CSP, frame-ancestors, Permissions-Policy). A05 · V14.4 | Fixed (App PR): CSP with hashes of each app's own inline scripts (no script `'unsafe-inline'`), checked against every app page by `test_app_pages_fit_their_csp` |
| M3 | Zip bombs and unbounded decompression: Campaign `extractall` without limits (and the case copied ~51 times), `cae.upload.expand` without a counted size or member-name check, gzip without a cap. A05 · V12.1.2 | Fixed (App PR): `pinneapple_security.archives` (names, links, counted sizes, ratio), `gunzip` cap |
| M4 | No per-caller rate limits on expensive routes (only per-worker concurrency). A04 · V11.1.4 | Fixed (App PR): token buckets per caller, 429 + `Retry-After` |
| M5 | SSRF: the gateway calls the URL of a remote ModelRef (admin only), including loopback and the cloud metadata address. **Measured**: before, publishing a ref to a service on 127.0.0.1 made the server send it a request; after, none. A10 · V12.6.1 | Fixed (App PR): `check_outbound_url` with `MDL_REMOTE_ALLOW_HOSTS` |
| M6 | Validation Lab: config result/reference paths may point outside the upload (server files into the report). A01 · V12.3.1 | Fixed (App PR), `test_validation_lab_config_confined_to_uploads` |
| M7 | Denial of service: `?wait=` sleeps inside async handlers (Sensitivity, Uncertainty); Sensitivity `/api/design` synchronous with unbounded `n_sobol`; Observatory cache unbounded; jobs without resource limits and orphaned grandchildren on kill. A04 | Fixed (App PR): threadpool, caps, bounded cache, opt-in `setrlimit`, process-group kill (`test_job_limits_and_no_traceback_in_job_log`) |
| M8 | User `str.format` templates in the Campaign Planner (attribute access, gigabyte padding); integer `**` in its formula evaluator. A03 | Fixed (App PR), `test_template_fields_restricted` |
| M9 | Meshless formulas reach `sympy.sympify` (which uses `eval`) behind a string blacklist. A03 | Fixed (App PR): grammar check first |
| M10 | Playground base image `caddy:2.8.4-alpine`: 7 critical, 86 high CVEs (trivy). A06 | Fixed (Deploy PR): `caddy:2-alpine` by digest, 0 critical / 4 high |
| M11 | Twin Runtime: live twins have no owner; anyone allowed in can stop, fault or feed another user's twin. A01 | **Open**: set a login or `TWIN_REQUIRE_API_KEY=1` on a shared server; ownership per twin needs a product decision |
| M12 | Comparator, Metadata, Preflight, Validation Lab, Twin models read a whole upload before their size check. A04 | Mitigated: request body counted and capped by the app (`<PREFIX>_MAX_BODY_MB`, 600 MB) and per site by Caddy (Deploy PR); a per-app streaming cap is open |

### Low

| ID | Finding | Status |
|---|---|---|
| L1 | Exception type and message in 422 answers; tracebacks in job logs (FastAPI's default 500 already had none). A09 · V7.4.1 | Partly fixed: JSON 500 with a reference id (`test_no_stack_trace_to_clients`); job tracebacks go to the server log unless `<PREFIX>_JOB_TRACEBACKS=1`. Input-error messages stay (they explain the input) |
| L2 | API key scopes are stored but not enforced; any login is admin. A01 | Partly: `<PREFIX>_ADMIN_USERS` (App PR). Scopes open |
| L3 | Validation Lab `os.chdir` races between concurrent runs. | Fixed (App PR): serialised |
| L4 | Inverse Heat jobs have no owner (random 48-bit ids). | Accepted (no uploads, demo results) |
| L5 | A ModelRef `zoo:<name>` bypasses the app's zoo allow-list in Sensitivity/Uncertainty (weights load with `weights_only=True`). | Open |
| L6 | HDF5 inputs read with `h5py` `[()]`: compression bombs and external links not checked. | Open (needs a test with crafted files) |
| L7 | Unpinned Python dependencies (DL3013 ×33) and base images by tag. A06 | Partly (CI PR): Dependabot, weekly pip-audit and Trivy, Caddy by digest. Hash-pinned requirements were not generated: the CPU torch index is unreachable from the audit environment |
| L8 | `RUNNER_SELF_HOSTED=1` runs any workflow (commands, Docker, sbatch) by design. | Documented: only behind its own login |
| L9 | bandit B324: SHA-1 for design IDs in `apps/aero_optimizer/tools`. | Fixed (CI PR): `usedforsecurity=False` (same IDs) |
| L10 | LLM use (Design Requirements, GUI agent): model output fills fields or picks one of the committed workflows; nothing it writes is executed. Prompt injection can at most pick another committed workflow. A03 (LLM01) | Accepted |
| L11 | Lab catalogue: plain-text password, no lockout. | Fixed (App PR): `LAB_PASSWORD_HASH`, lockout |

**Checked and not exploitable**: the gateway's `zoo:` reference with a path (`../../etc/passwd`): `pp.zoo.load`
only accepts registered names (**measured**: `KeyError`). Uploads: names are reduced to a safe basename and stored
under the data volume by a random id (`test_upload_name_traversal_and_suffix`). ROM Lab and Physics Design job
files: allow-listed or resolved-path checked. `np.load` everywhere with `allow_pickle=False`. No committed secrets
(gitleaks, whole history). Not applicable: session cookies (none are set; Basic auth and API keys only).

**Not implemented**: OIDC/OAuth for browser logins. Recommended route: an identity-aware proxy in front of Caddy's
sites (`forward_auth` to oauth2-proxy or Authelia), keeping `X-API-Key` for API clients.

## After the changes (measured)

The same behaviour probe (15 properties, FastAPI TestClient against `appkit` and the real Campaign Planner and
Model Gateway apps, plus the library evaluators) was run unchanged on `main` and on the changed tree:

| Property | main | changed |
|---|---|---|
| security headers (CSP with frame-ancestors, HSTS, nosniff) | fail | pass |
| no stack trace or exception text in a 500 | pass (FastAPI default) | pass (+ reference id) |
| no CORS grant to a foreign origin | pass | pass |
| oversized request body → 413 (1 MB limit, 3 MB body) | fail (200) | pass |
| upload name `../../x.csv` neutralised | pass | pass |
| rate limit → 429 (5/min, 8 POSTs) | fail (8× 200) | pass (5× 200, 3× 429) |
| lockout after repeated failed logins, also for the right password | fail (401 then 200) | pass (429) |
| hashed password (`PASSWORD_HASH`) | fail (unsupported) | pass |
| cross-site POST refused | fail (200) | pass (403) |
| direct client cannot choose its identity with `X-Forwarded-For` | fail | pass |
| dataset constraint cannot reach Python internals | fail | pass |
| boundary-value formula cannot reach Python internals | fail | pass |
| Campaign edit cannot write outside the case | fail (file modified) | pass (job refused, file untouched) |
| gateway does not call an internal address | fail (1 request) | pass (0) |
| CAE upload refuses `..` member names | fail | pass |
| **total** | **3 / 15** | **15 / 15** |

Tools:

| Check | Before | After |
|---|---|---|
| bandit (same scope) | 2 high, 5 medium, 17 low | **0 high**, 5 medium, 19 low |
| semgrep (same rules, same scope) | 52 error / 116 warning / 73 info | 21 error / 113 warning / 72 info (missing-user 32 → 0, eval-detected 4 → 0) |
| trivy config | 33 high (DS-0002) | **0** |
| hadolint | 36 warnings, 1 info | 36 warnings, 1 info (unpinned pip/apt: L7) |
| gitleaks (history, with `.gitleaks.toml` allow-listing the 3 placeholders) | 39 false positives | 0 |
| trivy, Playground base image | 7 critical / 86 high | 0 critical / 4 high |
| A hardened container (uid, capabilities, writes) | root, all default capabilities, writable root | uid 10001, `CapEff 0`, `NoNewPrivs 1`, `/` and `/srv` read-only, `/data` and `/tmp` writable |

The remaining semgrep errors are code-quality rules (`useless-inner-function` for FastAPI route closures,
`identical-is-comparison` in two scripts), not security findings; the medium bandit findings are B608 on SQL built
only from `?` placeholders, B610 and the server-written interop cache (L-level, reviewed above).
