# Simulation Campaign Planner

**Which simulations really need to run, in which order, and the case files to run them.**

Day 13 of the PINNeAPPle app program (campaign.pinneapple.org). This is the first of the apps that run studies,
after twelve that prepare, check and connect engineering data.

**Pain.** Simulation campaigns are often planned by hand: a grid that explodes with the number of parameters, a few
"interesting" combinations, and reruns of cases that already exist under another name. Compute goes into redundant or
unsystematic runs. The holes in the parameter space only show up later, when a surrogate model extrapolates into them.

| You give it | You get |
|---|---|
| A **parameter space**: geometry, materials, boundary conditions and operating point. Each parameter has bounds, units and an optional log scale; integers, categories and constants are allowed, plus constraints between parameters. | A **design of experiments**: Latin hypercube, maximin Latin hypercube, Sobol, Halton, full factorial, two-level fractional factorial (with generators and resolution), or **sequential / adaptive infill** around the runs you already have |
| A **simulation budget** (number of cases) | **Space coverage**: centred L2 discrepancy, minimum and mean nearest-neighbour distance, fill distance (the biggest hole), and the 1-D and 2-D projections. Measured for the plan, for your existing runs, and for both together. |
| Optionally the **runs you already have** (CSV / Parquet / JSON / HDF5, one column per parameter, plus outputs) | **Redundancy**: planned cases that repeat a run already done are flagged and dropped. Near-duplicate pairs inside your data are listed, and so are runs outside the space. |
| | **Priorities**: a run order in which every prefix covers the space as evenly as possible (greedy maximin from the existing runs), with the distance each case adds |
| | **Case files**: OpenFOAM case directories and CalculiX `.inp` decks, plus a resumable Python runner (`run_cases.py`), CSV / Parquet / JSON tables, and the OpenFOAM campaign-runner `plan.json` |
| | **Contracts** for the next apps: `space.json` (ParameterSpace), `study.json` (StudyResult with coverage, checks and limits), and `dataset.skeleton.json`. The skeleton is a DatasetSpec with the inputs and the outputs to come, for the Surrogate Studio and the Workflow Runner. |

## Presets ("Try with example", offline, seconds)

- **Airfoil polars with OpenFOAM.** The space has 6 CST weights (thickness and camber, the bounds of
  `pinneapple_design.aero.geometry`) and the angle of attack (−2° to 14°), at Re = 4·10⁶. The existing runs are the
  **436 OpenFOAM runs** of the zoo dataset `airfoil_rans_cst_re4e6`. Infill places 24 new cases where those runs are
  sparsest, each a simpleFoam k-ω SST case written by the same library builder that produced the dataset. In those
  436 runs the planner finds 2 near-duplicate pairs (shapes s016 and s081, each run twice at angles 0.18° and 0.32° apart). It also
  finds 18 runs outside the CST bounds: the NACA 4412 and 4415 polars, whose camber weight M1 = 0.142 exceeds 0.14.
- **Cantilever stiffness with CalculiX.** Length, width, height, material (S355 steel, 6061-T6 aluminium, Ti-6Al-4V),
  and a log-scaled tip load, with the slenderness constraint `length >= 8 * height`. Each case gets a C3D8I deck
  and its Euler-Bernoulli tip deflection for checking.
- **Your own space.** Edit the table or load one: one row per parameter, with columns `name, kind, low, high,
  values (a|b|c), unit, log, group` ([example](examples/parameters_cantilever.csv)). For OpenFOAM, upload a
  zipped base case and the dictionary edits per case (`0/U | boundaryField/inlet/value | uniform ({U} 0 0)`). For
  CalculiX, upload an `.inp` with `{parameter}` placeholders.

## Validation

| What | Result |
|---|---|
| Fractional factorial generators | Minimum-aberration designs of the literature: 2^(7−3) and 2^(8−4) of resolution IV, 2^(5−1) of resolution V, 2^(10−5) of resolution IV. All have orthogonal, balanced main effects (`tests/pinneapple_data/test_doe.py`). |
| Space filling | A Latin hypercube has one case in every stratum of every parameter. Maximin beats LHS and random sampling on minimum distance, and Sobol beats random sampling on discrepancy (tested). |
| CalculiX decks | Generated cantilever decks run with **CalculiX 2.21**. Tip deflection was 0.991, 0.994 and 0.990 × F L³/3EI on the three cases run. C3D8I with 2 elements through the height sits about 1 % below beam theory. (`tests/pinneapple_simulation/test_campaign_cases.py` runs the check when `ccx` is installed.) |
| OpenFOAM base-case mode | Dictionary edits per case and a `plan.json` that `openfoam.campaign.Campaign.from_json` reads back (tested on the cavity example case) |
| HTTP | `tests/test_app_campaign.py`: example jobs, uploads, repeat detection, infill, downloads, metering, invalid requests |

## Methods

`pinneapple_data.doe` builds every design in the unit hypercube of the free parameters. The ParameterSpace maps
it to values: log scales, integer rounding and category cells. Bounds and constraint expressions are then
enforced, with a safe evaluator rather than `eval`. When constraints remove points, more candidates are drawn and a
feasible subset is kept, so the Latin-hypercube stratification becomes approximate; the 1-D coverage shows by how much.
Maximin is the best of 20 random Latin hypercubes by minimum distance. Infill is greedy maximin over a Sobol
candidate pool from the existing runs. With outputs, the distance is weighted up where the outputs vary most
between neighbouring runs. Repeats are cases within a tolerance of an existing run (default 1 % of the unit-cube
diagonal). The case files come from `pinneapple_simulation.external_solvers.campaign_cases`.

## Integrations

OpenFOAM: airfoil cases from `pinneapple_design.aero.case`, or your base case with `openfoam.campaign` edits plus
its multi-machine runner plan. CalculiX: `calculix.study.FEModel`, or your `.inp` template. Python: `run_cases.py`
skips the cases already done. CSV, Parquet, JSON and HDF5 tables carry a `.contract.json` sidecar. The output feeds
the Workflow Runner (day 27, which runs the cases) and the Surrogate Studio (day 14, which trains on the finished
DatasetSpec).

## Run

```bash
pip install -e . -r apps/simulation_campaign/requirements.txt          # from the repository root
cd apps/simulation_campaign && uvicorn simulation_campaign.api:app --port 8096
```

The environment variables are listed below. The job queue, API keys, usage and uploads come from
[`apps/_shared`](../_shared/README.md).

| Variable | Default | Meaning |
|---|---|---|
| `CAMPAIGN_USER` / `CAMPAIGN_PASSWORD` | unset | HTTP Basic login on everything except `/health` |
| `CAMPAIGN_MAX_JOBS` | 2 | campaigns planned at once |
| `CAMPAIGN_FREE_CAMPAIGNS` | 50 | free campaigns per principal and month |
| `CAMPAIGN_FREE_CASES_PLANNED` | 5000 | free planned cases per principal and month |
| `CAMPAIGN_MAX_CASES` | 5000 | largest campaign |
| `CAMPAIGN_MAX_CASE_DIRS` | 50 | solver case directories per campaign; the tables always hold every case |
| `CAMPAIGN_ADMIN_TOKEN`, `CAMPAIGN_REQUIRE_API_KEY`, `CAMPAIGN_DATA_DIR`, `CAMPAIGN_UPLOAD_MAX_MB` | | see `apps/_shared/README.md` |

Docker: `docker build -f apps/simulation_campaign/Dockerfile -t simulation-campaign .`. To deploy it with the other
apps, use [`apps/deploy`](../deploy) (service `campaign`, volume `campaign_data`).

## API

| Route | What |
|---|---|
| `POST /api/campaigns` | Start a campaign (JSON: `preset` or `space`, `n`, `method`, `seed`, `levels`, `tol`, `existing_upload`, `existing_id_column`, `outputs`, `openfoam`, `calculix`, ...). Returns a job. |
| `GET /api/jobs/{id}` | Status, progress, and the result when done: summary, checks, cases, projection data |
| `GET /api/campaigns/{id}/download/{file}` | `campaign.zip`, `cases.csv`, `cases.parquet`, `cases.json`, `study.json`, `dataset.skeleton.json`, `space.json`, `run_cases.py`, `plan.json`, `dropped_repeats.csv` |
| `POST /api/example/{airfoil,cantilever}` | The presets |
| `POST /api/uploads`, `POST /api/existing/inspect`, `POST /api/space/from-table` | Upload runs or a parameter table |
| `GET /api/usage`, `/api/keys`, `/api/jobs` | Runtime (see `apps/_shared/README.md`) |

Library:

```python
from pinneapple_data.doe import plan_campaign
from pinneapple_simulation.external_solvers.campaign_cases import write_campaign

plan = plan_campaign(space, n=40, method="infill", existing=runs, existing_id_column="run_id")
write_campaign(plan, "campaign/", outputs=["Cl", "Cd"])
```

**Monetization:** campaigns and number of cases planned. Both are metered per API key, login or IP address
(`/api/usage`), with monthly free limits set by `CAMPAIGN_FREE_CAMPAIGNS` and `CAMPAIGN_FREE_CASES_PLANNED`.

## Limits

- Coverage is measured in the unit cube of the parameters, not in the response. A design that fills the cube can
  still miss a narrow feature of the outputs, such as a stall angle or a buckling threshold.
- Adaptive infill is a heuristic: it weights by how much the existing outputs vary between neighbouring runs. It is
  not a model-based error estimate. Uncertainty-driven infill belongs to the Surrogate Studio.
- A repeat is a case within the tolerance of an existing run in parameter space. Runs that differ only in solver
  settings or mesh look the same unless those are parameters.
- The airfoil and cantilever builders cover their own parameters only. Other OpenFOAM or CalculiX models use the
  base-case or template modes.
- Constraints are expressions over the parameters. Geometric validity that needs code (as for the airfoils) is
  available through the library's `feasible=` argument, not through the UI.
- The planner writes the cases; it does not run them (that is the Workflow Runner) and it does not check solver
  settings (that is Simulation Preflight).
