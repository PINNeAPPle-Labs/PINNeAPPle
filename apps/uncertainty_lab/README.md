# Engineering Uncertainty Lab

**Know when a prediction is reliable, and when the case must go to the high-fidelity solver.**

Day 15 of the PINNeAPPle 30-app program (uncertainty.pinneapple.org, port 8098).

## Pain

A surrogate quoted to three digits looks accurate. It can still drive a bad decision when the case lies outside the
data the model learned from, when the data there are sparse, or when the model's error is larger than the margin to a
limit. Most teams do not measure any of this before using the number.

## Input

- **A model.** It can be a `pp.zoo` model with held-out reference data (`heatsink_forced_mlp`, `airfoil_cst_rans_mlp`)
  or an uploaded **ONNX** graph: one `(n, d)` float input, and no Python code from the user runs. The model interface
  is `pinneapple_analysis.study_models.StudyModel`, so the shared `ModelRef` contract maps onto it.
- **Reference data.** By default the zoo model's own splits are used. You can upload CSV, Parquet or JSON with the
  inputs and the true output. An optional `split` column (`cal` / `test` / `train`) controls the split; without it the
  rows are split half and half at random.
- **Cases to screen.** Pasted as CSV or uploaded as a table.
- **Uncertain variables** with distributions (normal, uniform, triangular, lognormal, truncated normal) around a
  nominal operating point.
- **Decision limits** (lower and/or upper) and the **tolerance** a decision can accept, in output units or as a
  percent of the prediction.

## Output

- **Prediction intervals** (split conformal) at 80, 90 or 95 %. The width can be constant, relative to the prediction,
  or scaled by the model's ensemble spread.
  - Coverage is **measured on a held-out split** that the calibration never saw, with its 95 % Wilson interval.
  - A reliability curve shows measured against nominal coverage at five levels.
  - Coverage is broken down by predicted-value bin and by applicability gate, which shows where the marginal
    guarantee hides under-coverage.
  - When the model has an ensemble, the coverage of its raw `mean ± z·std` interval is reported for comparison.
- **Applicability alerts** for each case:
  - *outside*: outside the validity range of the model card or the data;
  - *sparse*: inside the range, but the 5th nearest reference point is farther away than for 99 % of the reference
    points themselves;
  - *inside*: neither of the above.
- **Routing rule.** A case goes to the high-fidelity solver when any of these holds:
  - it is outside or sparse;
  - its interval is wider than the tolerance;
  - its interval contains a limit, so pass/fail is undetermined.

  Each case lists the reasons. For the heat-sink model the library's physics engine
  (`pinneapple_design.thermal.heatsink.surrogate.reference_solve`) runs the routed cases, or all of them. This shows
  whether the rule sent the right cases.
- **Uncertainty propagation** by Monte Carlo, Latin hypercube or Sobol QMC. It reports:
  - percentiles and a histogram;
  - P(meeting the limits), both for the prediction alone and with the model's own error (the whole interval within
    the limits);
  - the share of propagated samples that leave the checked region;
  - polynomial chaos (degree 3, with its leave-one-out Q²) and the total Sobol index of each uncertain input, when the
    expansion has at most 200 terms.
- **Downloads.** The full study as JSON (`study_result`), and the screened cases as CSV.

## The examples (offline, a few seconds)

| Example | What it shows |
|---|---|
| Heat-sink surrogate vs its physics engine | 90 % relative intervals measured on 900 held-out designs: 91.0 % coverage (95 % CI 89.0–92.7 %). 11 designs are screened against R ≤ 0.40 K/W and every one is also solved by the engine. The only design whose engine value falls outside its interval is the corner of the design space, which the gate flags as *sparse* and routes to the solver. The designs beyond the range (8 m/s fan, k = 120 W/mK) are routed even though the surrogate happens to be close there; the rule is deliberately conservative. |
| Airfoil lift with an ensemble spread | The model's own ensemble interval (mean ± 1.64 std) covers 75 % of the 60 held-out OpenFOAM runs at a 90 % nominal level. The conformal interval built on the same spread covers 88.3 % (CI 77.8–94.2 %; small test set). |

## Methods and validation

The app composes library calls only: `pp.uncertainty.uncertainty_study`, which runs `conformal_study`,
`ApplicabilityDomain`, `route_cases`, `propagate_space` and `polynomial_chaos`. The tests are
`tests/test_uncertainty_study.py` (the library) and `tests/test_uncertainty_lab_app.py` (this API).

- The conformal quantile is the finite-sample order statistic `ceil((n+1)(1-α))` (Angelopoulos & Bates 2022). The
  library implements it once, in `ConformalPredictor`.
- Propagation reproduces the Ishigami mean and variance. Polynomial chaos reproduces polynomials exactly and reports
  its leave-one-out error.
- The gate is tested on a dataset with a hole inside its range.

## Limits

- Conformal coverage is **marginal**. It holds on average for cases like the calibration data, not case by case.
  Read the per-group and per-gate tables, and route sparse or outside cases to the solver.
- With few calibration points the interval is close to the largest residual seen, and the measured coverage has a
  wide confidence interval (the airfoil example: 36 + 60 runs).
- The gate looks at inputs only (range and k-NN density in standardised units). A point can be dense in inputs and
  still be in a different physical regime, such as stall. Encode regime limits as decision limits.
- Propagation assumes independent inputs. The propagated spread is that of the prediction; the model's error enters
  only in "with model error".
- A high-fidelity solver is wired for the heat-sink model only. For other models the routed cases are listed for your
  own solver (OpenFOAM, CalculiX). The runner app (day 27) will close that loop.
- Studies run as jobs in the shared queue (`UQL_MAX_JOBS` at once), within the size caps below. A `ParameterSpace`
  given as the uncertain inputs is uniform on its bounds unless its `metadata["distributions"]` says otherwise;
  categorical and log-scaled parameters are refused.

## Integrations

- The zoo models and datasets (`pp.zoo`, `pp.datasets`).
- Any tool that exports ONNX (PyTorch, scikit-learn via skl2onnx, Keras, MATLAB).
- Reference data from any solver or test rig as CSV, Parquet or JSON. Parquet files from the Interoperability Hub
  work directly.
- **The shared contracts** (`pinneapple_data.contracts`): a `ModelRef` for the model (`model_ref`: `zoo:<name>`, or an
  ONNX graph uploaded with it; `GET /api/model/{name}` returns one), a `ParameterSpace` for the uncertain inputs
  (`parameter_space`), and a `StudyResult` (`study_type: uncertainty`, with the checks that ran: held-out coverage,
  coverage per group, high-fidelity agreement, chaos Q²) in every result, for the other apps.

## Run

```bash
pip install -e . -r apps/uncertainty_lab/requirements.txt          # from the repository root
cd apps/uncertainty_lab && uvicorn uncertainty_lab.api:app --port 8098
```

Environment variables (the shared runtime ones are in [`apps/_shared/README.md`](../_shared/README.md)):

- `UQL_USER` / `UQL_PASSWORD`: HTTP Basic login; `UQL_ADMIN_TOKEN` for API-key management without a login.
- `UQL_MAX_JOBS`: studies running at once (default 2); `UQL_MAX_HEAVY`: concurrent synchronous examples.
- `UQL_FREE_STUDIES`, `UQL_FREE_SCENARIOS`, `UQL_FREE_PROPAGATION_SAMPLES`, `UQL_FREE_HF_RUNS`: free use per caller
  and month (defaults 200, 20,000, 2,000,000 and 500; `unlimited` lifts a limit).
- `UQL_MAX_MB`: largest upload in MB (default 50).
- `UQL_MAX_CASES`: cases screened per study (default 2000).
- `UQL_MAX_PROPAGATION`: Monte-Carlo samples per study (default 20000).
- `UQL_MAX_HF`: high-fidelity runs per study (default 40).

Docker: `docker build -f apps/uncertainty_lab/Dockerfile -t uncertainty-lab .`. To deploy it with the other apps, see
[`apps/deploy`](../deploy) (service `uncertainty`).

## API

- `POST /api/study`: runs a study as a job. Multipart `config` (JSON) plus the optional files `model` (.onnx),
  `reference` and `cases` (or `*_upload` ids from `/api/uploads`); `?wait=<s>` waits for the job.
- `GET /api/jobs/{id}`: status, progress and result (shared runtime; also `/logs`, `/cancel`, `/retry`, `/api/usage`).
- `GET /api/example/{heatsink|airfoil}`: runs an example; `/config` returns its configuration.
- `GET /api/model/{name}`: inputs, units, validity range and nominal point.
- `GET /api/template/{name}.csv`: a reference-data template.

Interactive documentation is at `/docs`. In Python it is one call:

```python
import pinneapple as pp
from pinneapple_analysis.study_models import StudyModel

model = StudyModel.from_zoo("heatsink_forced_mlp")
X_cal, Y_cal = model.reference("cal")
X_test, Y_test = model.reference("test")
out = pp.uncertainty.uncertainty_study(model, "r_base_k_w", X_cal, Y_cal[:, 0], X_test, Y_test[:, 0],
                                       scale="relative", X_ref=model.reference("train")[0], bounds=model.bounds)
```

## Monetization

Billed on three things:

- **Evaluations**: the model evaluations and high-fidelity runs of each study, reported in `usage`.
- **Scenarios**: the cases screened.
- **Study size**: propagation samples, and uploads above the free caps.

Metered per caller and month by the shared runtime (`GET /api/usage`, units `studies`, `scenarios`,
`propagation_samples`, `hf_runs`), with a free tier and HTTP 429 past it; the examples are free.
