# Engineering Sensitivity Lab

**Which parameters really control the result, and which can you freeze?**

Day 16 of the PINNeAPPle 30-app program (sensitivity.pinneapple.org, port 8099).

## Pain

Calibrations, optimisations and test campaigns often carry dozens of parameters, and most of them barely matter. Teams
usually find out which ones by hand, one-at-a-time, around one point. That misses interactions and says nothing about
how sure the answer is.

## Input

- **A model.** One of:
  - a built-in benchmark with exact indices (Ishigami, Sobol' G);
  - a `pp.zoo` model (`heatsink_forced_mlp`, `airfoil_cst_rans_mlp`);
  - an uploaded **ONNX** graph (no user Python runs).

  The model interface is `pinneapple_analysis.study_models.StudyModel`, so the shared `ModelRef` contract maps onto it.
- **Runs done elsewhere** (OpenFOAM, CalculiX, the Campaign Planner / Runner): download the Saltelli design as CSV,
  run it in your solver, then upload `run_id` and the responses.
- **A dataset** of experiments or past simulations, for first-order indices from the data alone.
- **Parameters with ranges or distributions** (uniform, normal, triangular, lognormal, truncated normal). Every other
  input is held at a fixed value.
- **The responses** to study.
- **The budget.** The model-evaluation count is shown before anything runs.

## Output

- **Importance ranking.** Sobol first-order and total indices per response, with bootstrap 95 % intervals.
- **Interactions.** Total minus first order per parameter, and second-order indices `S_ij` as a heatmap.
- **Morris screening.** μ* (importance) against σ (nonlinearity or interactions), with an interval on μ*.
- **Effect plots.** Main effects (partial dependence) and ICE curves for every parameter. ICE curves that are not
  parallel mean the parameter interacts with others.
- **Local derivatives** at the nominal point:
  - by **automatic differentiation** when the model is a torch model (the zoo models), otherwise by central
    differences;
  - each derivative also scaled by its range width, and as an elasticity.
- **Parameters you can freeze, with the evidence.**
  - A parameter is frozen only when the *upper* confidence bound of its total index is below the threshold for every
    response.
  - Below the threshold with a wider interval is reported as *inconclusive*.
  - The recommendation is then checked directly: the model is re-run with those parameters fixed, and the variance
    ratio and pointwise change are reported.
- **Given data.** First-order indices with intervals, plus a warning with the largest input correlation when the
  inputs are not independent.
- **Downloads.** The full study as JSON (`study_result`), and the indices as CSV.

## The examples (offline, seconds)

| Example | What it shows |
|---|---|
| Ishigami function | Estimated S1 and ST against the exact indices: largest error 0.003. All six exact values fall inside the 95 % intervals. x3 has no main effect but a 0.24 interaction with x1. |
| Heat-sink design (zoo model, 10 parameters, 2 responses) | Fin count dominates the thermal resistance (ST 0.54); fin count and air velocity dominate the pressure drop. Ambient temperature and power can be frozen (ST < 2e-4, upper bound < 0.01). Re-running with them fixed changes the output variance by less than 0.1 %. Derivatives come from autodiff. |
| From data only (3,900 heat-sink simulations) | The first-order indices add up to 1.09 because the dataset's inputs are correlated (r = 0.66). The app says so instead of presenting them as a variance decomposition. |

## Methods and validation

The app composes library calls only:

- `pp.sensitivity.sensitivity_study` and `study_cost`;
- `saltelli_design` and `sobol_from_runs`;
- `first_order_given_data`.

The Sobol estimators are implemented in the library, because SALib is not a dependency:

- first order: Saltelli et al. (2010);
- total: Jansen (1999);
- second order: Saltelli (2002);
- intervals: percentile bootstrap.

Morris follows Morris (1991) with μ* from Campolongo et al. (2007). The library tests are in
`tests/test_sensitivity.py`:

- exact Ishigami and Sobol' G indices;
- Morris on a linear model;
- autodiff against the analytic gradient;
- given-data indices;
- freeze check on an inert input;
- runs done elsewhere returned in any order.

The API tests are in `tests/test_sensitivity_lab_app.py`.

## Limits

- **Indices depend on the input distributions you give.** Widen a range and that parameter matters more.
- **Designed studies sample inputs independently.** Given-data indices of correlated inputs can add up to more than 1
  and credit one input with another's effect; the app warns. Shapley effects are planned.
- **Small samples.** At N = 256-512 the bootstrap intervals can be narrower than the true error, as seen on Ishigami
  at N = 512. Use N ≥ 1024 for decisions.
- **Indices are those of the model.** A surrogate can carry small spurious dependencies or miss real ones outside its
  data. Check the model first, for example in the Engineering Uncertainty Lab.
- **Runs done elsewhere.** Every run of the design is needed, and a missing run is named. The freeze recommendation
  is not re-checked there, because the model is not available to the app.
- **Execution.** Studies run as jobs in the shared queue (`SEN_MAX_JOBS` at once), capped by `SEN_MAX_EVALS` and
  `SEN_MAX_PARAMS`; integer parameters of a `ParameterSpace` are sampled as continuous values, categorical and
  log-scaled ones are refused unless a distribution is given in its `metadata`.

## Integrations

- Python models through `StudyModel`.
- Any ONNX export.
- **OpenFOAM and CalculiX**, through the design and results round trip: the Campaign Planner / Runner apps can
  execute `sobol_design.csv` and return `run_id` plus responses.
- Experimental or historical data as CSV, Parquet or JSON.
- **The shared contracts** (`pinneapple_data.contracts`): a `ParameterSpace` for the parameters (`parameter_space`;
  distributions other than uniform in its `metadata["distributions"]`), a `ModelRef` for the model (`model_ref`,
  `zoo:<name>` or an ONNX graph uploaded with it), and a `StudyResult` (`study_type: sensitivity`, with the checks
  that ran) in every result. `/api/model/...` and `/api/design` also return the parameters as a `ParameterSpace`.

## Run

```bash
pip install -e . -r apps/sensitivity_lab/requirements.txt          # from the repository root
cd apps/sensitivity_lab && uvicorn sensitivity_lab.api:app --port 8099
```

Environment variables (the shared runtime ones are in [`apps/_shared/README.md`](../_shared/README.md)):

- `SEN_USER` / `SEN_PASSWORD`: HTTP Basic login; `SEN_ADMIN_TOKEN` for API-key management without a login.
- `SEN_MAX_JOBS`: studies running at once (default 2); `SEN_MAX_HEAVY`: concurrent synchronous examples.
- `SEN_FREE_STUDIES`, `SEN_FREE_EVALUATIONS`, `SEN_FREE_PARAMETERS`: free use per caller and month (defaults 200,
  5,000,000 and 2,000; `unlimited` lifts a limit).
- `SEN_MAX_EVALS`: model evaluations per study (default 400000).
- `SEN_MAX_PARAMS`: varied parameters (default 30).
- `SEN_MAX_MB`: largest upload in MB (default 50).

Docker: `docker build -f apps/sensitivity_lab/Dockerfile -t sensitivity-lab .`. To deploy it with the other apps, see
[`apps/deploy`](../deploy) (service `sensitivity`).

## API

| Endpoint | What it does |
|---|---|
| `POST /api/study` | Model study as a job: multipart `config` JSON, plus `model` (.onnx) or `model_upload`; `?wait=<s>` |
| `GET /api/jobs/{id}` | Job status, progress and result (shared runtime; also `/logs`, `/cancel`, `/retry`) |
| `POST /api/cost` | Evaluations a study would take |
| `POST /api/design` | Saltelli design CSV; its first line is the spec |
| `POST /api/runs` | Indices from solver results, as a job: `spec` + `runs` table |
| `POST /api/given-data` | First-order indices from data, as a job: `config` + `data` table (or `data_upload`) |
| `POST /api/uploads`, `GET /api/usage`, `/api/keys` | Shared runtime: uploads, metered use, API keys |
| `GET /api/example/{ishigami\|heatsink\|dataset}` | Runs an example (`/config` returns its configuration) |
| `GET /api/model/{builtin\|zoo}/{name}` | Inputs, units and a default parameter space |

Interactive documentation is at `/docs`.

## Monetization

Billed on **parameters** (varied parameters per study) and **computational evaluations** (model evaluations or solver
runs; the cost is shown before running). Both are metered per caller and month by the shared runtime (`GET
/api/usage`, units `studies`, `evaluations`, `parameters`), with a free tier (200 studies, 5 million evaluations,
2,000 parameters a month by default) and HTTP 429 past it; the examples are free.
