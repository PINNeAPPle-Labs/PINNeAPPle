# PINNeAPPle Arena

**Port:** 8130 · **Prefix:** `PINN_ARENA` · **Service:** `pinnarena` (apps/deploy)

Pick a physics problem and a few PINNeAPPle architectures, train them under one protocol, compare them on a held-out
split, and rank them on a leaderboard built from the experiments run on this server. A second section browses a
catalog of third-party physics-ML libraries and runs each one through its own package, reporting what really ran.

Imported from the private `PINNeAPPle-Labs/PINNeAPPle-arena` repository (FastAPI backend + React frontend); see
"What changed in the import" below.

## What it does

**PINNeAPPle's own architectures** (logic: `pinneapple_arena.pointwise`)

- **Problems** (`/problems`): 34 problems in 7 domains (fluids, solids, electromagnetism, quantum, heat, waves,
  nonlinear dynamics), each with its equations, parameters and a reference dataset from an analytic or numerical
  solution (`DatasetGenerator`; the metadata says which). Some reference fields are reduced-order or illustrative
  (crash, 3-D airfoil, cylinder wake): their metadata says so.
- **Experiments**: one or more architectures (catalog of 93 registry entries) trained on the reference samples, with
  a PDE residual on separate collocation points where the problem has one (16 of the 34), cosine LR, early stopping
  on a validation split and the best weights restored; scored on a held-out split (relative L2, RMSE, MAE, max error,
  R², MAPE) with a prediction-quality heuristic (constant, zero, NaN or below-the-mean outputs are flagged). Trained
  weights can be downloaded (`state_dict`). Your own table (CSV / TSV / JSON / NPZ, with a column mapping) can replace
  the generated data.
- **Leaderboard** (`/leaderboard`): mean relative L2 per architecture over the completed experiments (filters by
  problem, domain, dimension, difficulty), win rate, convergence rate (relative L2 < 0.1), and 0-100 scores for
  accuracy (absolute) and training cost, memory and inference speed (relative to the other entries: hardware
  dependent). The database starts empty: no stored results come with the app.
- Optional **GitHub sign-in**: experiments of signed-in users are kept under their account; guests' experiments are
  visible to their own session.

**External catalog** (`/external-catalog`, logic: `pinneapple_arena.external`)

- 95 third-party libraries (DeepXDE, PINA, NeuralOperator, PhiFlow, JAX-MD, MACE, NetKet, PyBaMM, Brax, ...) in 41
  physical domains, each a reference: name, organisation, repository, licence (read from the repository's licence
  file or PyPI metadata on 2026-10-24, with the source recorded), citation and install instructions. No third-party
  code is in this repository; the glue runner of each library imports it at run time.
- Run inference / fine-tune / reconfigure on a small synthetic input, compare 2-6 libraries, benchmark a slice of the
  catalog. Every result has a status: `ok` (the library ran), `not_installed`, `skipped_heavy` (GPU / heavy, opt in
  with `include_heavy`), `manual_install` (no pip package), `failed` (with the exception), `finetune_not_supported`.
  Non-`ok` results carry the exact command to get a real result elsewhere (`python -m pinneapple_arena.external
  --model <slug> --action infer`). On a fresh install almost every library reports `not_installed`,
  `skipped_heavy` or `manual_install`: that is the expected state.
- **Datasets**: The Well (16 physical datasets, 24 loader ids); a probe loads one real sample through its own loader.
- Licences in the catalog: MIT 38, Apache-2.0 25, no licence file found 12, GPL-3.0 6, BSD-3-Clause 4,
  CC-BY-NC-SA-4.0 3, LGPL-2.1 2, BSD-2-Clause 2, AGPL-3.0 1, LGPL-3.0 1, FluidX3D's custom licence 1
  (`pinneapple_arena.external.licence_counts()`). Non-commercial code or weights (flagged on the cards):
  DeepONet (original), DeepONet/FNO benchmark, ReservoirFlow, FluidX3D, GraphCast weights, Pangu-Weather weights.

## How pinn_arena differs from `apps/benchmark_arena` and `pinneapple_arena`

| | `apps/pinn_arena` (this app) | `apps/benchmark_arena` | `pinneapple_arena` (library, before this app) |
|---|---|---|---|
| Question | which PINNeAPPle architecture fits this field best, problem by problem | which method family works best per data regime | run configured PINN benchmarks from YAML |
| Candidates | any registry architecture (`pp.build_model`) | `pinneapple_analysis.surrogate.methods` (ridge, GP, GBT, MLP, FNO, DeepONet, PINN, solvers) | registry models via `model_factory` |
| Data | coordinates → field of 34 problems, or your table | parametric tables / fields, regimes (few data, extrapolation, geometry change, fronts) | `ArenaProblem` classes with their residuals |
| Budget | epochs per run (not equal compute) | equal wall-clock budget, several seeds | epochs per model |
| Extra | leaderboard across experiments, GitHub sign-in, external catalog | split audit, seed spread, StudyResult | UQ, inverse, NAS, decision mode |

Overlap: both arenas rank PINNeAPPle models on physics data, and both libraries hold problem catalogs (the
`ArenaProblem` classes of `pinneapple_arena.problems` and the 34 UI problems of `pinneapple_arena.pointwise.problems`
share Burgers, heat, wave, Klein-Gordon, KdV, Poisson, Laplace, Helmholtz, Darcy, elasticity). They are not merged:
the pointwise problems carry UI metadata and reference generators the app needs, the `ArenaProblem`s carry their
own residuals for the YAML runner. benchmark_arena's equal-budget, multi-seed protocol is the one to use for claims
about method families; pinn_arena's leaderboard is a per-server record of single runs.

## Run

```bash
pip install -e . -r apps/pinn_arena/requirements.txt
(cd apps/pinn_arena/frontend && npm ci && npm run build && rm -rf ../pinn_arena/static && cp -r dist ../pinn_arena/static)
cd apps/pinn_arena && PYTHONPATH=../_shared uvicorn pinn_arena.api:app --port 8130   # http://localhost:8130, /docs
```

Without the frontend build the API runs alone (`/` says so). Frontend development: `npm run dev` in
`apps/pinn_arena/frontend` (Vite on :5173, proxies `/api` to :8130).
Docker: `docker build -f apps/pinn_arena/Dockerfile -t pinn-arena .` from the repository root (a Node stage builds
the frontend from `package-lock.json`).

Sanity check of every architecture (random data, building and training only):
`python apps/pinn_arena/scripts/check_all_models.py --epochs 1 --skip-3d`.

## Environment

All optional. App login and hardening come from `apps/_shared/appkit.py` (`apps/_shared/README.md`).

| Variable | Default | Meaning |
|---|---|---|
| `PINN_ARENA_USER`, `PINN_ARENA_PASSWORD` / `PINN_ARENA_PASSWORD_HASH` | unset | HTTP Basic login on everything but `/health` (private deployment) |
| `PINN_ARENA_DATA_DIR` | `/data/pinn_arena` or `~/.pinneapple/apps/pinn_arena` | SQLite database (`arena.sqlite`), uploaded datasets, trained weights |
| `PINN_ARENA_MAX_EPOCHS`, `_MAX_ARCHITECTURES`, `_MAX_COLLOCATION` | 5000, 8, 20000 | limits of one experiment (422 above) |
| `PINN_ARENA_MAX_RUNNING` | 2 | experiments training at once (429 above) |
| `PINN_ARENA_FREE_EXPERIMENTS` | 100 | experiments per caller and month (`unlimited` lifts it) |
| `PINN_ARENA_ADMIN_TOKEN`, `PINN_ARENA_RATE_LIMIT`, `PINN_ARENA_CORS_ORIGINS`, ... | | appkit settings |
| `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET` | unset (sign-in off) | the GitHub OAuth app (callback `<server>/api/auth/github/callback`) |
| `GITHUB_CALLBACK_URL` | `<this server>/api/auth/github/callback` | override behind a proxy that rewrites hosts |
| `ARENA_JWT_SECRET` | random per process | HS256 key of the session tokens; set it or every restart signs everyone out |
| `ARENA_FRONTEND_URL` | `/` | where the browser goes after sign-in (only for a separately hosted frontend) |
| `EXTERNAL_CATALOG_DATASET_ROOT` | unset | local copy of The Well for dataset probes |
| `EXTERNAL_CATALOG_ALLOW_SERVER_SIDE_INSTALL` | `false` | allow `POST /api/external-catalog/models/{slug}/install` (and then only for the admin) |
| `EXTERNAL_CATALOG_SYNC_WAIT_SECONDS`, `EXTERNAL_CATALOG_MAX_BENCHMARK_MODELS`, `EXTERNAL_CATALOG_PUBLIC_BASE_URL` | 20, 40, request URL | external catalog job wait, benchmark size, URL in the API snippets |

The session token travels in `X-Arena-Token`, so GitHub sign-in works behind the Basic login too.

## Verified here (2026-10-25, CPU, PyTorch 2.14)

- Source repository tests, ported: `pytest tests/test_app_pinn_arena.py` 25 passed (the 21 integration tests
  of the source, two merged into others, plus 6 for the port's changes); `pytest tests/test_arena_pointwise.py tests/test_arena_external.py` (adapter tests ported
  from `test_model_adapters.py`, plus new tests); frontend `npx vitest run` 20 passed, `npm run build` ok.
- `check_all_models.py --epochs 1 --skip-3d`: **62 PASS, 114 FAIL, 10 SKIP** of 186 (93 architectures × 2 problems).
  The source README's "176 PASS, 0 FAIL, 10 SKIP" is reproduced by the source code here too, but 108 of those 176
  were a fallback MLP trained under the architecture's name (54 architectures `pp.build_model` can not build for
  pointwise data, including the 9 entries renamed in the arena's catalog). The port reports those as failures:
  46 constructors need inputs a pointwise problem does not have (sequences, graphs, a backbone, filters), 8 renamed
  Noether wrappers need the optional `emmiai-noether` package, 2 models (`koopman_rom`, `neural_rom`) return a
  non-pointwise shape, `gno` fails on its input shape; 5 trajectory models are not applicable (SKIP).
- Started the app (`uvicorn`, Basic login set) and hit `/health` (200 without login), `/api/*` (401 without, 200
  with), the SPA routes and its CSP. The Docker image was not built here (no Docker daemon).

## Limits

- Training and test points are random samples of the same reference field: the score is interpolation accuracy, not
  extrapolation or generalisation to other parameters.
- Architectures are run with registry defaults (64 hidden, 4 layers); no per-architecture tuning; one seed; the budget
  is epochs, not time. Rankings between close errors are not settled.
- The leaderboard pools different problems in its mean relative L2; filter to one problem to compare like with like.
- The PDE residual is the arena's native one (with the experiment's parameters) or, failing that, the PINNeAPPle
  preset's; 18 of the 34 problems train on data only. Boundary and initial conditions enter only through the data.
- Parameter identification (inverse mode) is not wired to `pinneapple_analysis.inverse_problems` yet: an inverse
  experiment trains the forward models and reports that no parameters were identified.
- The quality levels (excellent / good / fair / poor) are a heuristic of R² and relative L2, not a physics check;
  no trust card is computed per run.
- Experiments train in the server process (background tasks); a restart loses running ones. The external catalog's
  job store is in memory. One uvicorn worker.

## What changed in the import

- Libraries: the problems, reference data, losses, adapters, training, metrics, leaderboard and upload parsing moved to
  `pinneapple_arena.pointwise`; the external catalog (references, runners, fine-tune, benchmark, probe) to
  `pinneapple_arena.external`. The app keeps routes, schemas, the database, auth and the UI.
- Fixed: no stand-in MLP under another architecture's name; catalog ids renamed in the arena build from their
  registry key; models with a non-pointwise output shape are refused instead of trained on a broadcast loss; six
  problems declared an output width different from their data (they could not be evaluated); the preset PDE losses
  were called with an outdated signature and skipped silently (`physics_loss_enabled` now means the residual ran);
  collocation points were drawn in `[0, 1]^d` whatever the problem's domain; an unreadable uploaded dataset no longer
  falls back to generated data; an unknown problem no longer gets a synthetic field; a cancelled experiment no longer
  flips to completed.
- Removed: the unauthenticated `POST /api/experiments/reset`; the `.pt` upload (`torch.load(weights_only=False)` on an
  unauthenticated upload, and the uploaded model was never trained); the stored GitHub access token; Google Fonts
  (blocked by the CSP); `slowapi` (appkit's rate limits and metering instead); the materialized leaderboard table.
- Not brought: `docker-compose.yml`, `Caddyfile`, `setup.sh`, `setup.ps1`, `dev.ps1`, `backend/entrypoint.sh`,
  `frontend/Dockerfile` + `nginx.conf` (folded into this Dockerfile and apps/deploy), `fix_db.py` and
  `seed_leaderboard.py` (one-off migration and seeding of the private server database), `backend/tests/results/*.txt`
  (a stale result log), `.claude/` (local agent settings), `TODO.md`.
