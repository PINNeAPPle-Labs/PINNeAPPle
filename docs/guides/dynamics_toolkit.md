# Dynamics toolkit: the shared foundation

`pinneapple_analysis.dynamics` is where the mathematical toolkit for dynamical systems is being built. It will have
five frameworks:

- first-passage and hitting times;
- dimensional analysis and Buckingham pi;
- Fisher information and experimental design;
- critical slowing down and early-warning indicators;
- Koopman and spectral analysis.

Each framework is added in its own pull request, together with its analytic benchmarks. **None of them is in the
package yet.** This page documents the parts they share and will be extended as each framework lands.

The parts they share:

| Part | Module | What it is |
|---|---|---|
| Method contract | `_contract` | `MethodCard` describes a method before it runs. `DynamicsResult` is what every method returns. |
| Backends | `_backends` | `capabilities()`, `require()` and `to_numpy()`. NumPy and SciPy are the core; PyTorch and JAX are optional. |
| Trajectories | `trajectories` | `Trajectory`, `TrajectoryEnsemble`, and adapters that build them from data files, simulations and trained models. |

## The method contract

Every method in the toolkit exposes the same ten items. Each item lives in one place:

| Item | Where |
|---|---|
| 1. mathematical inputs | `MethodCard.inputs` |
| 2. explicit assumptions | `MethodCard.assumptions`, then `DynamicsResult.assumptions` with `checked` set |
| 3. configuration | `MethodCard.config`, then `DynamicsResult.config` |
| 4. execution | the method's own call |
| 5. structured results | `DynamicsResult.estimate` |
| 6. uncertainty | `DynamicsResult.uncertainty`, where each entry names its kind |
| 7. numerical diagnostics | `DynamicsResult.diagnostics` (`Diagnostic` objects) and `DynamicsResult.convergence` |
| 8. validation | the method's benchmark against a reference, in `tests/` |
| 9. reproducibility metadata | `DynamicsResult.reproducibility`: seed, input hash, library versions, wall time |
| 10. failure and applicability conditions | `MethodCard.failure_modes`, then `DynamicsResult.applicable` and `.failures` |

### Assumptions

An `Assumption` has three possible states:

- `checked=True`: verified on these inputs. The `evidence` field says how.
- `checked=False`: tested on these inputs and found violated.
- `checked=None`: the inputs cannot test it, so the caller is responsible for it.

The two lists show which applies:

- `DynamicsResult.unchecked_assumptions` lists the ones the caller must vouch for.
- `DynamicsResult.violated_assumptions` lists the ones that failed.

### When a result is `ok`

`DynamicsResult.ok` is true only when all three hold:

- the method applies;
- no diagnostic has the status `fail`;
- no assumption was found violated.

A result that does not apply still carries its diagnostics, so you can see why it does not apply.

### Roles

`MethodCard.role` says how a method is meant to be used. Not everything is forced into an `nn.Module`.

| Role | Meaning |
|---|---|
| `primitive` | a function of numbers |
| `trajectory_operator` | acts on trajectories or ensembles |
| `differentiable_module` | can sit inside a training graph |
| `validation` | judges another result |
| `experimental_design` | decides what to measure |

### Exporting a result

`DynamicsResult.to_dict()` gives plain JSON:

- arrays become lists;
- NaN and infinity become `null`;
- complex numbers become `{"re", "im"}`.

`DynamicsResult.to_study_result()` builds the apps' shared contract, `pinneapple_data.contracts.StudyResult`:

| From the result | In the `StudyResult` |
|---|---|
| diagnostic with status `ok` / `warn` / `fail` / `info` | check with status `PASS` / `WARN` / `FAIL` / `N/A` |
| assumption that was checked | `PASS` or `FAIL` check |
| unchecked assumptions and `limits` | `limits` of the study |
| scalar entries of the estimate and of the uncertainty | metrics |

```python
import numpy as np
from pinneapple_analysis import dynamics as dyn

rep = dyn.reproducibility("demo", {"window": 50}, [np.arange(10.0)], seed=0)
r = dyn.DynamicsResult(
    "demo", {"tau": 2.0}, uncertainty={"std": 0.1, "kind": "bootstrap"},
    assumptions=[dyn.Assumption("markov", "the process is Markov")],      # not testable here: checked=None
    diagnostics=[dyn.Diagnostic("residual", 1e-9, "ok", threshold=1e-6)],
    reproducibility=rep)
assert r.ok and [a.name for a in r.unchecked_assumptions] == ["markov"]
study = r.to_study_result(study_type="uncertainty")       # needs pydantic
```

### Random numbers

`make_rng(seed)` returns a NumPy generator together with its seed. With `seed=None` it draws a fresh seed and returns
it, so an unseeded run still records how to repeat it.

## Backends

`capabilities()` reports which versions of NumPy, SciPy, SymPy, PyTorch and JAX are installed. It looks for each
package without importing it.

`require("jax", purpose)` imports a backend. If the backend is missing, it raises `BackendUnavailable`, and the
message gives the install command.

`to_numpy(x)` converts a PyTorch tensor (it detaches it and moves it to the CPU) or a JAX array. It does not
import either library.

The modules of `pinneapple_analysis.dynamics` never import PyTorch or JAX at module level, and a test enforces
this. However, the parent package `pinneapple_analysis` does import PyTorch when it loads, because its uncertainty
and validation sub-packages use it. PyTorch is also a core dependency of `pinneapple`. So in this repository,
"optional" means two things:

- no method of the toolkit needs PyTorch or JAX unless you ask for an autodiff path;
- importing `pinneapple_analysis` still requires PyTorch to be installed.

## Trajectories

A `Trajectory` holds a state `x(t_i)` in `R^d`:

| Field | Meaning |
|---|---|
| `t` | sample times, strictly increasing and finite; the grid may be irregular |
| `x` | values, shape `(n, d)`; `NaN` marks a missing value, per component |
| `names` | one name per component |
| `units` | unit text per component, and `"t"` for time. A missing entry means "not stated", not "dimensionless". |
| `observed_until` | end of the observation window, `>= t[-1]`. An event not seen by then is right-censored, not "never". |
| `source` | where the numbers came from: `{"kind": ...}`, with `kind` one of `experimental`, `simulation`, `analytic`, `pinn`, `fno`, `deeponet`, `model`, `stochastic`, `array` |

A trajectory is immutable. These operations return a new one:

| Operation | What it does |
|---|---|
| `window(t0, t1)` | keeps the samples between `t0` and `t1` |
| `select(names)` | keeps the named components |
| `dropna(how)` | drops samples with missing values |
| `resample(t_new, method)` | interpolates per component (`"linear"` or `"previous"`) from that component's own samples. Outside a component's data it gives `NaN` (no extrapolation), and the result is marked `source["resampled"]`: interpolated values are not measurements. |

`sampling()` reports the step statistics. Its `irregularity` value is `dt_max / dt_min`.

`TrajectoryEnsemble` holds several trajectories of the same variables, for example Monte Carlo paths, repeated
experiments or a parameter sweep. Its members may use different grids and windows. `stack()` needs a common grid.
`TrajectoryEnsemble.from_paths(t, paths)` builds an ensemble from an array of shape `(m, n)` or `(m, n, d)`.

### Adapters

| Source | Adapter | Notes |
|---|---|---|
| arrays (NumPy, PyTorch, JAX) | `from_array(t, x, names, kind=...)` | |
| table or DataFrame | `from_table(table, time, columns)` | Reads units from the headers (`"T [K]"`, `"p (bar)"`) with `pinneapple_data.physical_units.split_header`. Empty cells and `NA` become missing values. Rows without a time stamp are dropped and counted. |
| CSV file | `from_csv(path, time, columns)` | Same rules as `from_table`. A row with the wrong number of fields is refused. |
| `pp.solve` result | `from_solution(solution, t, at={"x": ...})` | Samples one point of space over time. The `kind` comes from the method: `pinn`, `analytic` or `simulation`. These values are predictions, not measurements, so check them with `pp.trust_card`. |
| trained network or callable | `from_model(fn, t, names, kind="pinn")` | `fn` gets `t` as `(n,)`, or as `(n, 1)` with `column=True`. A PyTorch module must be wrapped by the caller; the adapter does not guess the device or the dtype. |
| PDE snapshots, or an FNO / DeepONet rollout on its grid | `from_snapshots(t, U, kind="fno", coords=...)` | `U` has shape `(n_t, *spatial)` and is flattened into a state vector. `Trajectory.snapshots()` restores the grid. |

```python
import numpy as np
from pinneapple_analysis import dynamics as dyn

t = np.array([0.0, 0.1, 0.3, 0.35, 1.0])
tr = dyn.from_array(t, np.column_stack([np.sin(t), [1, np.nan, 3, 4, 5]]), ("s", "p"),
                    units={"t": "s", "p": "Pa"}, observed_until=2.0, kind="experimental")
print(tr, tr.sampling(), tr.censored)       # irregular, 1 missing, censored at t = 2 s
g = tr.resample(np.linspace(0, 1, 11))      # interpolated inside the data only; marked as resampled
```

## Known limits of the foundation

- `from_solution` samples a single point of space. To get a field over time, sample a grid with `from_snapshots`.
- No adapter estimates the error of the model it reads. That is the job of the trust layer (`pp.trust_card`) and of
  each framework's own diagnostics.
- `units` are kept as text and are not converted. Dimensional reasoning on them arrives with the Buckingham pi
  framework.
