# apps/_shared: what every app gets

`appkit.py` is the glue every app under `apps/` imports (`apps/_shared` is on `PYTHONPATH` in each Dockerfile).
The logic behind it lives in the library, with tests:

| Piece | Library | Tests |
|---|---|---|
| Job queue, API keys, usage metering, uploads | `pinneapple_core.service` | `tests/pinneapple_core/test_service.py` |
| Data contracts (`ParameterSpace`, `DatasetSpec`, `ModelRef`, `StudyResult`) | `pinneapple_data.contracts` | `tests/pinneapple_data/test_contracts.py` |
| The HTTP routes below | `apps/_shared/appkit.py` | `tests/test_appkit_runtime.py` |
| 3-D field view (`/shared/fieldview3d.js`: fields, errors, deformed shape, markers) | `pp.viz.field_view`, `pinneapple_analysis.validation.error_map` | `tests/test_field_view.py`, `tests/js/studio_core.test.mjs` |

No new dependencies: SQLite and threads/processes from the standard library, plus FastAPI and
`python-multipart`, which every app already lists. Parquet needs `pyarrow` and HDF5 needs `h5py`.

## Minimal app

```python
from fastapi import FastAPI, Request
from appkit import install, install_runtime

app = FastAPI(title="Simulation Campaign Planner")
install(app, prefix="CAMPAIGN")                        # optional login + /shared, /vendor, /studio static files
rt = install_runtime(app, prefix="CAMPAIGN", name="campaign",
                     units={"cases_planned": 500, "campaigns": 20},   # metered units, free per month
                     upload_suffixes={".csv", ".parquet", ".json", ".h5"})


def plan(params: dict, ctx) -> dict:                   # runs in the queue: thread (default) or process
    ctx.progress(0.1, "sampling")                      # also raises JobCancelled after a cancel request
    ctx.log("LHS with 40 points")
    (ctx.workdir / "cases.csv").write_text("...")      # per-job directory under the data volume
    return {"n_cases": 40}                             # JSON-able result, stored in SQLite


rt.jobs.register("plan", plan)


@app.post("/api/campaigns")
def create(request: Request, body: dict):
    # checks the free limit (429 when exceeded), queues the job owned by the caller, counts the units
    return rt.submit(request, "plan", body, units={"campaigns": 1, "cases_planned": body["n"]})
```

The UI polls `GET /api/jobs/{id}` (status, progress, message, result) and `GET /api/jobs/{id}/logs?since=<seq>`.

## Routes added by `install_runtime`

| Route | Who | What |
|---|---|---|
| `GET /api/runtime` | anyone allowed in | max concurrent jobs, job kinds, free limits, upload limits |
| `GET /api/jobs?status=&kind=&limit=` | caller (admin: all) | the caller's jobs |
| `GET /api/jobs/{id}` | owner, admin | status, progress, message, result, error, attempts |
| `GET /api/jobs/{id}/logs?since=` | owner, admin | log lines after sequence number `since` |
| `POST /api/jobs/{id}/cancel` | owner, admin | queued: cancelled at once; running: stop requested |
| `POST /api/jobs/{id}/retry` | owner, admin | put a failed or cancelled job back in the queue |
| `GET /api/usage` | caller (admin: + totals) | use this month per unit, limit, remaining |
| `GET/POST /api/keys`, `DELETE /api/keys/{id}` | admin | list, create (`{"name": ...}`; the key is shown once), revoke |
| `POST /api/uploads` (multipart `file`), `GET /api/uploads`, `DELETE /api/uploads/{id}` | caller | uploads |

**Who is the caller** (the "principal" use is counted against and jobs belong to): `key:<id>` for a valid
`X-API-Key`, `user:<login>` for the app login, otherwise `ip:<address>` (behind Caddy, the address Caddy saw; `X-Forwarded-For` is believed only from a trusted proxy).
A job or upload of another principal answers 404. **Admin** is the app login (`<PREFIX>_USER`) or the
`X-Admin-Token` header matching `<PREFIX>_ADMIN_TOKEN`; with neither configured, nobody can manage keys.

## Environment

All optional; `<PREFIX>` is the app's prefix (`CAMPAIGN`, `RUNNER`, ...).

| Variable | Default | Meaning |
|---|---|---|
| `<PREFIX>_USER`, `<PREFIX>_PASSWORD` | unset | HTTP Basic login on everything but `/health`; a valid `X-API-Key` also passes (prefer `<PREFIX>_PASSWORD_HASH`, below) |
| `<PREFIX>_ADMIN_TOKEN` | unset | `X-Admin-Token` for key management when there is no login |
| `<PREFIX>_REQUIRE_API_KEY` | `0` | `1`: every `/api/*` call needs a key (or the login) |
| `<PREFIX>_DATA_DIR` | `/data/<app>` if `/data` is writable, else `~/.pinneapple/apps/<app>` | SQLite, job directories, uploads. `PINNEAPPLE_APP_DATA` sets the parent for all apps |
| `<PREFIX>_MAX_JOBS` | `max_jobs` argument (2) | jobs running at once, over all server workers |
| `<PREFIX>_JOB_EXECUTOR` | `thread` | `process`: each job in its own interpreter (cancel and time limit kill it; the function must be module-level) |
| `<PREFIX>_JOB_MAX_SECONDS` | unset | time limit per job (thread jobs stop at their next `ctx.progress`) |
| `<PREFIX>_FREE_<UNIT>` | the `units` default | free use per principal and calendar month (UTC); `unlimited` or `-1` lifts it |
| `<PREFIX>_UPLOAD_MAX_MB` | `upload_max_mb` argument (50) | upload size limit (413 above it) |
| `<PREFIX>_JOB_MAX_MEMORY_MB`, `<PREFIX>_JOB_MAX_CPU_S` | unset | process executor: address-space and CPU-time limit per job (`setrlimit`); a job and every process it starts are killed together |
| `<PREFIX>_JOB_TRACEBACKS` | `0` | `1`: tracebacks of failed jobs go into the job log the caller reads (otherwise only into the server log) |

## Security settings

`install()` wires `pinneapple_security.web` into every app; the code and its tests are in the library
(`tests/pinneapple_security/test_web_hardening.py`, `tests/test_app_security.py`). All optional:

| Variable | Default | Meaning |
|---|---|---|
| `<PREFIX>_PASSWORD_HASH` | unset | hash of the login password (`python -m pinneapple_security hash-password`: scrypt, or argon2id with `argon2-cffi`; bcrypt `$2b$` hashes from `caddy hash-password` are verified when `bcrypt` is installed). Takes precedence over the plain `<PREFIX>_PASSWORD`, which still works and logs a warning |
| `<PREFIX>_USERS_FILE` | unset | more logins: `user:hash` lines |
| `<PREFIX>_ADMIN_USERS` | every login (only `<PREFIX>_USER` once a users file is set) | comma-separated logins that may manage API keys and publish models |
| `<PREFIX>_LOGIN_MAX_FAILURES`, `<PREFIX>_LOGIN_WINDOW_S`, `<PREFIX>_LOGIN_LOCK_S` | 10, 900, 900 | after this many failed logins in the window, the client IP and the user name get 429 for the lock time (also with the right password) |
| `<PREFIX>_RATE_LIMIT` | `120/min` | POST/PUT/PATCH/DELETE under `/api` and `/v1` (solver runs, inference, uploads) per caller (`key:`, `user:` or `ip:`); `off` disables |
| `<PREFIX>_RATE_LIMIT_READ` | `1200/min` | every request per caller |
| `<PREFIX>_MAX_BODY_MB` | `max_body_mb` argument, else 600 | request body limit, counted while streaming (413) |
| `<PREFIX>_CORS_ORIGINS` | unset (no CORS headers: same origin only) | comma-separated allow-list of origins; `<PREFIX>_CORS_CREDENTIALS=1` allows credentials and then refuses `*` |
| `<PREFIX>_ALLOWED_ORIGINS` | unset | other origins whose pages may POST to this app (all other cross-site POST/PUT/DELETE from browsers get 403) |
| `<PREFIX>_TRUSTED_PROXIES` / `PINNEAPPLE_TRUSTED_PROXIES` | loopback and private ranges | peers whose `X-Forwarded-For` is believed (the Docker network Caddy is on) |
| `<PREFIX>_CSP`, `<PREFIX>_HSTS` | built from the app's pages / one year | override the Content-Security-Policy or Strict-Transport-Security value (`""` drops HSTS) |
| `<PREFIX>_AUDIT_LOG`, `<PREFIX>_AUDIT_KEY` | `audit.jsonl` in the data directory | hash-chained trail of logins, lockouts, refused requests, API key and admin events; with a key the chain is an HMAC chain. `off`: log lines only |

Every response carries a Content-Security-Policy (scripts only from the app itself plus its own inline scripts by
hash, no `'unsafe-inline'` for scripts, `frame-ancestors 'none'`), HSTS, `X-Content-Type-Options: nosniff`,
`X-Frame-Options: DENY`, `Referrer-Policy` and `Permissions-Policy`. An unhandled error answers
`{"detail": "internal error", "reference": "<id>"}`; the traceback goes to the server log under that id. Auth
events are one JSON line each on stderr (`docker logs`), which `apps/deploy/fail2ban` reads.

## Deploying

Give the service a volume for its state, so jobs, keys and usage survive a rebuild:

```yaml
  campaign:
    environment:
      CAMPAIGN_MAX_JOBS: ${CAMPAIGN_MAX_JOBS:-2}
      CAMPAIGN_FREE_CASES_PLANNED: ${CAMPAIGN_FREE_CASES_PLANNED:-}
      CAMPAIGN_ADMIN_TOKEN: ${CAMPAIGN_ADMIN_TOKEN:-}
    volumes:
      - campaign_data:/data
    mem_limit: ${CAMPAIGN_MEM_LIMIT:-1g}
```

The queue is safe with `uvicorn --workers 2`: jobs are claimed atomically in SQLite, the concurrency limit holds
across workers, and a job whose worker died (restart, crash, OOM kill) is found by its stale heartbeat (30 s) and
queued again, at most `max_retries + 1` extra times, then marked failed.

## Data contracts

Every app reads and writes `pinneapple_data.contracts` so the apps chain: a design of experiments
(`StudyResult` + `DatasetSpec` skeleton) from the Campaign Planner is the input of the Workflow Runner, whose
finished runs become a `DatasetSpec` for the Surrogate Studio, whose `ModelRef` goes into the uncertainty,
sensitivity and optimisation apps.

```python
from pinneapple_data.contracts import DatasetSpec, ParameterSpace, StudyResult, load, read_table, write_table

space = ParameterSpace.load("fin.space.json")          # or .yaml; ParameterSpace.from_table("params.csv")
cases = space.from_unit(unit_points)                   # DataFrame, one column per parameter
cases = cases[space.feasible(cases)]                   # bounds + constraints
write_table(cases, "cases.parquet", space)             # + cases.parquet.contract.json (also embedded)
df, contract = read_table("cases.parquet")
spec = DatasetSpec.from_table("cases.parquet", outputs=["Rth"], ids=["case_id"])
```

JSON Schemas for other languages are in `pinneapple_data/contracts/schemas/` (draft 2020-12). Rules the models
enforce: unknown fields are errors (extras go in `metadata`); `ModelRef.metrics` are held-out and finite; a
`CheckRecord` with PASS/FAIL/WARN needs a value or evidence; `StudyResult.passed` is false when no check ran.

## Limits

- Metering counts use; it does not bill. Anonymous use is counted per IP address, which a shared NAT shares
  and a changing address escapes: set `<PREFIX>_REQUIRE_API_KEY=1` where that matters.
- Thread jobs cannot be killed: cancel and the time limit take effect at the job's next `ctx.progress` /
  `ctx.check_cancel`. Use the process executor for code that does not report progress.
- One SQLite file per app on a local volume; not for a network file system or many hosts.
