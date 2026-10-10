# PINNeAPPle Lab: experiments at scale, and a database of results and datasets

`pinneapple_lab` runs experiments reproducibly and stores everything they produce in one place. The goal is a growing
database of physics experiments that can be used as demonstrations and use cases, and as training data for
future PINNeAPPle models.

## The idea

An **experiment** is a class with parameters and a `run(ctx)` method. A **run** is one execution with one set of
parameters. Its id is a hash of the experiment, its version and the parameters, so:

- the same run is never computed twice; an existing result is returned as `cached:...`;
- changing the experiment's code meaning means bumping `version`, which keeps the old results apart.

Through the `RunContext`, an experiment records:

| what | how | where |
|---|---|---|
| inputs and outputs | `ctx.input(name, value)`, `ctx.output(name, value)`: arrays → `.npy`, dicts / numbers → `.json`, files → copied | `inputs/`, `outputs/` |
| metrics | `ctx.metric(name, value, step=None)`; final values plus a curve log | `metrics.json`, `metrics_log.jsonl` |
| validation | `ctx.check(name, passed)` or `ctx.check(name, value=..., reference=..., rtol=..., max=...)`; a failed check ends the run as `failed_validation` | `validation.json` |
| status and stages | `with ctx.stage("train"):`; statuses `running`, `completed`, `failed_validation`, `failed` (with traceback) | `run.json`, `progress.json` |
| figures and animations | `ctx.figure(name, fig)`, `ctx.image(name, array)`, `ctx.gif(name, frames)` | `figures/` |
| datasets | `ds = ctx.dataset(name, units=...)`, then `ds.add(**sample)`; samples are written in shards, with a card (schema, shapes, value ranges, units, provenance, checksum) | `datasets/<name>/` |
| provenance | parameters, git commit, Python and package versions, timings | `run.json` |

Runs live in `lab/runs/<experiment>/<run_id>/` and are indexed in `lab/index.sqlite`. Set the location with `--root`
or `$PINNEAPPLE_LAB`.

## Use

```python
from pinneapple_lab import Experiment, register, run, sweep, LabStore

@register
class MyExperiment(Experiment):
    name = "my_experiment"
    version = "1"
    params = {"Re": 100.0, "seed": 0}
    space = {"Re": ("log", 10.0, 1000.0)}           # for random / Latin-hypercube sweeps

    def run(self, ctx):
        with ctx.stage("simulate"):
            field, cd = simulate(ctx.params["Re"])
        ctx.metric("drag_coefficient", cd)
        ctx.check("drag_positive", value=cd, min=0.0)
        ctx.dataset("fields").add(u=field, Re=ctx.params["Re"])
        ctx.figure("field", plot(field))

run("my_experiment", {"Re": 200.0})
sweep("my_experiment", grid={"Re": [50, 100, 200]}, n_jobs=3)
sweep("my_experiment", samples=100)                 # Latin hypercube over `space`

store = LabStore()
store.table("my_experiment")                        # pandas: params and metrics per run
store.export_dataset("my_experiment", "fields", "train.npz")   # completed runs only
store.catalog()                                     # lab/CATALOG.md
```

From the command line:

```bash
python -m pinneapple_lab list
python -m pinneapple_lab sweep cylinder_lbm -g Re=20,40,60,80,120,160,200 -j 4
python -m pinneapple_lab sweep oscillator -n 200
python -m pinneapple_lab status
python -m pinneapple_lab report
python -m pinneapple_lab export cylinder_lbm vorticity data/cylinder_vorticity.npz
```

An experiment defined outside the package is addressed as `package.module:ClassName`.

## Built-in experiments

| name | what | checks | dataset |
|---|---|---|---|
| `oscillator` | damped oscillator: RK4, Euler and symplectic Euler against the exact solution | error within the method's order | trajectories |
| `heat_xtfc` | 1D heat equation by X-TFC against the exact solution | relative L2 below 1e-3 | fields |
| `bondi_accretion` | black-hole hydro started from exact Bondi accretion | accretion rate and density steady within 2 % | radial profiles |
| `accretion_flow` | torus accreting onto a Schwarzschild hole | mass and angular-momentum budgets | density movies (forecasting) |
| `cylinder_lbm` | lattice-Boltzmann flow past a cylinder | Strouhal number in its physical range; steady below the onset | vorticity images labelled by regime (vision) |
| `kepler_law`, `pendulum_video`, `lorenz_discovery`, `oscillator_discovery` | physical laws discovered from data (planet ephemerides, a pendulum video, a chaotic trajectory) | recovered exponents and coefficients against the known law | orbits, video frames, discovered laws |
| `repo_results` | results already produced by repository scripts, re-validated: Burgers PINN, fin inverse PINN (2D, 3D), LBM Strouhal, MeshGraphNet, delta-wing polar, black-hole forecast skill | exact solution, true h, published Strouhal, solver baselines | fields, polars, skill curves |
| `example_script` | any script of `examples/` (examples and use cases) | exits cleanly within its timeout | artifacts (arrays the script wrote) |
| `benchmark_case` | the landing-page cases of the public PINNeAPPle-Benchmark and PINNeAPPle-Climate repositories: soil-temperature twin, bumper crash surrogate, terramechanics, heated-channel twin, pedestrian dead reckoning, supersonic shock train, SST forecasts | each headline claim against its baseline (model vs persistence, improvement vs reproduction, ...) | arrays behind the paper figures, sweeps, lead skill |
| `car_lbm` | a parametric 2-D car body (six design parameters, `pinneapple_design.geometry.gen.car2d`) in a lattice-Boltzmann wind tunnel with moving road | finite fields, converged drag, drag in the bluff-body range | geometry + mean flow + coefficients; wake vorticity |
| `car_surrogate` | Physics AI on the `car_lbm` database: FNO geometry → mean flow (with and without a mass-conservation loss), MLP ensemble design → drag/lift, then a low-drag design found by the surrogate and verified by LBM | errors on unseen designs vs baselines; surrogate optimum confirmed by a new simulation | predictions |
| `bar_wear` / `bar_wear_ranking` | sliding wear of a crowned bar end on a flat counterface, one run per material (Archard & Hirst data: mild steel, brass, tool steel, stellite, PTFE, stainless steel, polyethylene, tungsten carbide); Archard wear with contact pressure on an elastic layer (`pinneapple_physics.tribology`) | worn volume = K F s, initial contact vs closed form, steady rate vs flat punch, running-in distance; ranking follows K = k/H | wear history (profiles, pressures), ranking |
| `pipe_flow` | internal flow in OpenFOAM on a structured O-grid swept along the pipe (`pinneapple_simulation.numerical_solvers.internal_flow`): laminar pipe, turbulent 90° bend, Kenics static mixer with a passive scalar | f vs 64/Re and Colebrook, bend loss vs Ito, Dean vortices, scalar balance, mixer vs empty pipe | section averages, cross-section grids, wall shear |
| `solid_fem` | 3-D linear elasticity with hexahedra with incompatible modes (`pinneapple_simulation.numerical_solvers.solid_fem`): cantilever, simply supported beam, bar in torsion; post-processor figures (deformed mesh, von Mises bands), loading GIF, Blender render, 3-D viewer | deflection and bending stress vs beam theory, twist rate and shear vs Saint-Venant, CalculiX on the same mesh, equilibrium | mesh, nodal displacements and stresses |
| `rom_study` | reduced-order models on the lab's solvers: POD / DMD / Operator Inference on a cylinder wake (LBM), parametric POD-RBF / POD-GPR on 3-D FEM cantilevers | POD mode pair, DMD frequency vs probe spectrum, forecasts vs persistence; unseen designs vs nearest design | wake snapshots and modes, parametric FEM designs |
| `particle_suspension` | solids suspension in a stirred tank: soft-sphere DEM with drag, buoyancy and turbulent dispersion in an analytic divergence-free stirred-tank flow; process video (particles by speed, turning impeller, live charts) | settling velocity vs Schiller-Naumann, divergence-free flow, particles in the tank, contact overlap, just-suspended speed vs Zwietering | particle positions and speeds over time |
| `calculix_case` | CalculiX studies built in Python (`pp.fea`): plate with a hole and L bracket meshed by gmsh (C3D10), cantilever modes, column buckling, cooling fin | Kt vs Heywood, bending vs M c / I, frequencies vs Euler-Bernoulli, buckling vs Euler, fin vs 1-D solution, mesh convergence, equilibrium | mesh and nodal results |

`repo_results` and `benchmark_case` let the long runs (an hour of LBM, 30 minutes of 3D PINN) enter the database without being repeated:
it reads what the script wrote, checks it against the reference again, and snapshots the generating script with
the run.

## Curation: which experiments are good enough

`python -m pinneapple_lab curate` (also run by `report`) grades every run and experiment and writes
`lab/curation.json` and `lab/CURATION.md`; the catalogue page shows the tier of each run, a quality filter and, in
each run's sheet, the dimensions, the readiness for each use and what to do next. It uses the vocabulary of
`pinneapple_veriphysics.applicability`: a dimension without evidence is *not run*, never a neutral pass, and evidence
is *verified* (a check measured it in this run), *inferred* (metrics, files, names) or *unsupported*.

| dimension | what counts | kind |
|---|---|---|
| validation | fraction of the run's checks that pass | quantitative |
| reference | checks against an independent truth: exact solution, experiment, published value | quantitative |
| baseline | checks that the method beats a simpler one (persistence, linear, nearest design, the original) | quantitative |
| physics | conservation, symmetry, monotonicity, a physical range or law the method was not trained on | quantitative |
| generalization | checks on unseen inputs (held-out designs, out-of-distribution cases, beyond the observed window) | quantitative |
| uncertainty | an uncertainty estimate is reported (ensemble spread, sigma, intervals) | quantitative |
| reproducibility | code snapshot, git commit, parameters, saved diff; agreement of repeats with other seeds | quantitative |
| data | datasets with documented cards | quantitative |
| assets | figures, movies, photoreal renders, a 3-D viewer | qualitative |
| documentation | description, docs pages that cite the experiment, papers / references | qualitative |
| review | a human review: novelty, clarity, visual appeal (1-5), a one-line story, approvals, limitations | qualitative |

A check says what it is evidence of with `ctx.check(..., kind="reference" | "baseline" | "physics" |
"generalization" | "sanity")`; left out, the kind is inferred from the name and the arguments.

Tiers are gates, not averages, so a strong figure never hides a failed check:

- **A flagship**: all checks pass; a reference check and a baseline or physics check pass; reproducible (code
  snapshot, commit); at least three figures or a movie / render / 3-D view; documented; for machine-learning
  experiments (surrogates, forecasters, neural operators) a generalization check passes. Ready to become a product,
  a paper or marketing once a human review approves it.
- **B solid**: all checks pass, at least one independent check (reference, baseline or physics), reproducible.
  Demos, use cases, training data.
- **C exploratory**: ran, but the evidence is thin (no checks or only sanity checks).
- **D not usable**: failed validation or crashed.

An experiment takes the tier of its best run, one level lower when fewer than half of its runs are usable. The score
(0-100) is the weighted mean of the dimensions that ran, shown with its coverage, like the Veriphysics trust score.
Readiness lists what each use still needs: *product* (generalization, uncertainty, a reference or physics check,
reproducibility), *paper* (reference, baseline or physics, reproducibility, novelty review), *marketing* (strong
visuals, a validated headline number, a story), *training data* (a documented dataset of validated runs).

The catalogue opens on **What can we present?**: for each use (product, publication, marketing, training data) the
items that are ready or approved, and the closest candidates with their distance in effort points (S = 1, M = 3,
L = 8) and what they miss. Below it, the **roadmap** lists every item (an experiment, or one case of it when the
experiment declares `case_param`) with its limitations — declared by the experiment (`Experiment.limitations`, a list
or a dict per case), by the reviewer, or detected (failed runs, inferred-only evidence, low coverage) — and the path
forward: the actions that close each gap, specific to the kind of experiment (a persistence baseline for a forecast,
a grid-convergence study for CFD, held-out designs for a surrogate, ...), with their effort and what each unlocks.

**Reports** for one item or a filtered set, as HTML, Markdown or PDF: a summary table (tier and distance to each use)
and, per item, the story, description, key figures, readiness, headline metrics, every validation check with its
kind, the evidence dimensions, limitations, path forward, references and the command that reproduces it.

```bash
python -m pinneapple_lab brief --item kepler_law -o kepler.html
python -m pinneapple_lab brief --tier A,B --use paper --limit 10 -o paper_candidates.pdf
python -m pinneapple_lab brief --ready marketing --format md -o marketing.md
python -m pinneapple_lab brief --experiment benchmark_case -o landing_cases.pdf
```

The catalogue has the same reports one click away (an item, a use's ready and closest items, or everything the
current filter shows); served by `python -m pinneapple_lab serve` they also download from
`/report.pdf|.html|.md?items=...&tier=...&use=...&ready=...`. The **Showcase** at the top of the catalogue gathers
the most striking figures of the database (photoreal renders, general-relativistic views, 3-D twins, movies).

Human review is the qualitative half and is never filled in automatically:

```bash
python -m pinneapple_lab review kepler_law --reviewer "name" --novelty 4 --clarity 5 --visual-appeal 3 \
    --story "Kepler's third law and the Sun/Jupiter mass ratio recovered from planetary data" --approve paper
python -m pinneapple_lab curate
```

## Examples and use cases

Every script under `examples/` (224 today, use cases included) is also a lab experiment, `example_script`. A run
executes the script from the repository root as a user would, then keeps what it produced anywhere in the checkout:
figures, JSON files (outputs, and their numbers as metrics), the `name: value` lines it printed (metrics prefixed
`stdout.`), arrays from `.npy` / `.npz` / `.csv` (the `artifacts` dataset), the console output, and the script with
its neighbouring Python files as the code that ran. Data files it overwrote in the checkout are restored, and
untracked files it left in the working tree are moved into the run, so the checkout stays clean. A script that needs
a GPU, a download or a missing optional dependency fails and is recorded, so the catalogue doubles as a health
report of the examples.

```bash
python -m pinneapple_lab examples                                   # list them, grouped by folder
python -m pinneapple_lab examples --run --match use_cases --timeout 1800
python -m pinneapple_lab run example_script -p script=examples/getting_started/03_heat_diffusion_1d.py
```

Scripts run one at a time, and the checkout should not be edited while they run: a run attributes every file that
changes during it to the script (sources and docs are never restored, but they are recorded as produced).

## Serving the catalogue

`python -m pinneapple_lab serve --host 0.0.0.0 --port 8093` serves the lab as a web app (standard library only):
the catalogue page, live from the database (new runs appear on reload), every run's files (`/files/...`), a JSON API
(`/api/catalog`, `/api/status`, `/api/runs`, `/api/runs/<id>`, `/api/datasets`) and each dataset over all validated
runs as one `.npz` (`/download/<experiment>/<dataset>.npz`). `LAB_USER` / `LAB_PASSWORD` turn on HTTP Basic login
(except `/health`). `pinneapple_lab/deploy/Dockerfile` packages it, and `apps/deploy` runs it as the `lab` service
behind Caddy next to the other apps. On a fresh checkout the run records are present but the dataset shards are not:
sweeping again recomputes exactly the runs whose shards are missing.

## Versioning the database

The run metadata, metrics, validation, figures and dataset cards are small and versioned in git under `lab/`. Arrays,
dataset shards and the SQLite index are not versioned (see `.gitignore`). Rebuild the index with
`python -m pinneapple_lab reindex`. `python -m pinneapple_lab report` also writes `lab/DATASETS.md`: every dataset
with its schema (shapes, ranges, units), its sample count over validated runs, and the sweep and export commands
that rebuild it (runs are cached by parameters, so the sweep only computes what is missing). Large datasets are exported (`export`) and published separately, for example to
the Hugging Face Hub with `pinneapple_hub`.
