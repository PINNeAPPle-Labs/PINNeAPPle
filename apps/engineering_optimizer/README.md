# Engineering Design Optimizer

**Better designs within physical and manufacturing limits, found by black-box optimisation on a fast model and
confirmed by a high-fidelity one.** Day 17 of the PINNeAPPle app series; live at `https://optimize.pinneapple.org`.

![Engineering Design Optimizer](../../docs/assets/apps/d17-optimize-result.jpg)

## The pain

Design teams size parts by hand-iterating a spreadsheet or a solver: a few variants, one at a time, and the
constraints (temperature, stress, pressure drop, what the supplier can extrude) checked at the end. Optimisers exist,
but wiring one to a model, keeping the constraints honest and checking that the "optimum" holds in the real solver is
a project in itself. Surrogate optima that the reference model rejects are the usual failure.

## Input

- **Design variables** (bounds, integer or continuous): edited in the form, or a `ParameterSpace` contract.
- **Objectives** (minimise or maximise) and **constraints** (upper or lower limits; physical, manufacturing or model
  validity) on the evaluation model's outputs.
- **Evaluation model**, one of
  - the **example**: an extruded plate-fin heat sink in ducted air, searched with the zoo surrogate
    `heatsink_forced_mlp` or the closed-form resistance network, verified with the 3-D finite-volume engine;
  - **your data**: a CSV / Parquet / JSON table of designs you already evaluated (a `DatasetSpec` sidecar is read):
    a Gaussian-process surrogate is fitted per output and its error on held-out rows is reported;
  - a **zoo model** (`pp.zoo`, the same `ModelRef` URI `zoo:<name>` the Benchmark Arena and the Surrogate Studio
    publish): its inputs become the variables, bounded by the model's validity range;
  - an **external solver** the server operator configured (`EDO_SOLVERS`), used as the reference.

## Output

- **Optimised candidates** spread along the Pareto front, each with its verification status: *verified* (the
  reference ran and the design meets every limit there), *rejected by reference*, *reference failed* or *only
  surrogate*.
- **Pareto front and trade-off plots** for any two objectives or constrained outputs, with the limits drawn.
- **Feasibility report**: per candidate, each constraint's value, margin and status (active within 5 % of the limit,
  inactive, violated).
- **High-fidelity confirmation**: surrogate vs reference value at every candidate and the relative error.
- **Convergence**: hypervolume of the feasible designs found against the number of evaluations.
- **Files**: `evaluations.csv` (every design), `candidates.csv`, `candidates_to_run.csv` (to run in your solver),
  `parameter_space.json` (ParameterSpace), `study.json` (StudyResult with the checks that ran), `model_ref.json`
  (ModelRef of a surrogate trained on your data, with held-out metrics).
- **Verify later**: run `candidates_to_run.csv` in your solver, upload the results, get the same verification table.

## Methods

All in `pinneapple_design.design_optimizer` (guide: [docs/guides/design_studies.md](../../docs/guides/design_studies.md)):
NSGA-II (the library's, with constrained domination and the crowded tournament), NSGA-II from pymoo, SciPy
differential evolution on augmented Tchebycheff scalarisations, Gaussian-process Bayesian optimisation (ParEGO) and a
Latin-hypercube baseline, all under the same evaluation budget. Candidates are picked evenly along the front;
`verify_candidates` re-evaluates them; `surrogate_assisted` runs active learning (fit a GP on reference runs, optimise
on it with each constraint at mean ± 2 standard deviations, verify, add, repeat).

## Measured

The example as shipped (NSGA-II, 600 evaluations of the zoo surrogate, 5 candidates verified with the FVM engine;
the run behind the screenshot): best mass 82.4 g (baseline design 486 g, -83 %), best unit cost 2.36 (baseline 6.15,
-62 %, with the placeholder cost model), all 5 candidates verified, largest surrogate error at the candidates 1.45 %
(pressure drop), 0.7 s in total. The optimisers are tested on an analytic two-objective problem with a known Pareto
set (`tests/test_design_studies.py`).

## Integrations

SciPy and pymoo (optional dependency) optimisers; your own models through a table of evaluated designs or a zoo
`ModelRef`; external solvers through the CSV-in / CSV-out command contract of `CommandEvaluator` (configured by the
operator) or offline through `candidates_to_run.csv`; the shared contracts (`ParameterSpace`, `ModelRef`,
`StudyResult`) so a Campaign Planner design, a Surrogate Studio model or an Arena verdict feeds straight in and the
result feeds the next app.

## Run

```bash
pip install -e . -r apps/engineering_optimizer/requirements.txt       # from the repository root
cd apps/engineering_optimizer && PYTHONPATH=../_shared uvicorn engineering_optimizer.api:app --port 8100
```

Docker: `docker build -f apps/engineering_optimizer/Dockerfile -t engineering-optimizer .` from the repository root;
deployment with the other apps in [`apps/deploy`](../deploy) (service `optimize`, port 8100).

| Variable | Default | Meaning |
|---|---|---|
| `EDO_USER`, `EDO_PASSWORD` | unset | HTTP Basic login (`/health` stays open); an `X-API-Key` also passes |
| `EDO_MAX_JOBS` | 2 | optimisations running at once; more wait in the queue |
| `EDO_FREE_EVALUATIONS`, `EDO_FREE_CAMPAIGNS`, `EDO_FREE_REFERENCE_RUNS` | 200000, 100, 500 | free use per caller and month (429 above) |
| `EDO_SOLVERS` | unset | JSON `{name: {command, outputs, timeout}}` of solvers users may pick as the reference |
| `EDO_ADMIN_TOKEN`, `EDO_REQUIRE_API_KEY`, `EDO_DATA_DIR` | | API keys and state (see `apps/_shared/README.md`) |

## API

`POST /api/runs` (`spec`, `method`, `budget`, `seed`, `n_verify`, `verify_with`, `mode`, `upload_id`) queues a run and
returns the job; `GET /api/jobs/{id}` gives progress, logs and the result; `GET /api/runs/{id}/files/{name}` the
files; `POST /api/runs/{id}/verify` checks uploaded solver results. `POST /api/uploads`, `POST /api/table/inspect`,
`GET /api/zoo`, `GET /api/meta`. Interactive docs at `/docs`.

## Monetization

Evaluations (search-model calls), optimisation campaigns (runs) and reference / cloud solver runs are metered per
caller; free monthly limits, API keys for automation, and operator-hosted solvers as the paid tier.

## Limits

- Candidates are optimal **for the search model**. Only those re-evaluated by the reference carry "verified"; with no
  reference they are labelled "only surrogate".
- Heat-sink example: ducted laminar/developing forced convection, no radiation, fin stress as root bending of a
  cantilever fin under a uniform lateral load. The cost model is a placeholder to replace with your numbers.
- Surrogate from your data: a GP per output, valid inside the range of your data only (the variables cannot leave it);
  a few hundred rows at most keep it fast.
- Bayesian optimisation is limited to 150 evaluations (it refits a GP at every step). Categorical variables are not
  optimised: fix them and compare runs.
- No user-supplied commands or model files are executed: external solvers are the operator's list.
