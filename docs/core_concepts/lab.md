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

`repo_results` lets the long runs (an hour of LBM, 30 minutes of 3D PINN) enter the database without being repeated:
it reads what the script wrote, checks it against the reference again, and snapshots the generating script with
the run.

## Versioning the database

The run metadata, metrics, validation, figures and dataset cards are small and versioned in git under `lab/`. Arrays,
dataset shards and the SQLite index are not versioned (see `.gitignore`). Rebuild the index with
`python -m pinneapple_lab reindex`. `python -m pinneapple_lab report` also writes `lab/DATASETS.md`: every dataset
with its schema (shapes, ranges, units), its sample count over validated runs, and the sweep and export commands
that rebuild it (runs are cached by parameters, so the sweep only computes what is missing). Large datasets are exported (`export`) and published separately, for example to
the Hugging Face Hub with `pinneapple_hub`.
