# Modelica models as Physics AI data

`pinneapple_simulation.external_solvers.modelica` takes a Modelica model, written as a `.mo` file, to training
data and checked surrogates:

1. OpenModelica compiles the model into an FMU (FMI 2.0, Co-Simulation or Model Exchange), or simulates it
   directly.
2. fmpy runs the FMU over a parameter sweep.
3. The runs become a dataset, split by run, so the test runs are parameter combinations the model never saw.
4. The surrogates are trained on that dataset and checked against the physics.

| Function | What it does |
|---|---|
| `omc.build_fmu(models, name, out_dir, fmi_version="2.0", fmu_type="cs")` | `.mo` file(s) to an `.fmu`, via `buildModelFMU` |
| `omc.simulate_mo(models, name, stop_time, parameters=...)` | Simulates with omc and returns `{variable: array}` including `time`. Parameters are applied with `-override`, so nothing is recompiled |
| `omc.omc_available()` | Returns `"local"` (omc on the PATH), `"docker"` (the image `openmodelica/openmodelica:v1.24.4-minimal`, or `PINNEAPPLE_OMC_IMAGE`) or None |
| `sweep.fmu_sweep(fmu, ranges, n, stop_time, output_interval, outputs, step_size=...)` | Latin-hypercube sweep on one output grid. Returns an `FMUDataset` (`params`, `time`, `outputs`, `split`, `save` / `load`) |
| `simulate_fmu`, `fmu_to_upd` | One FMU run, as arrays or as a `PhysicalSample` |
| `fmi_compat.check_fmus(fmus, exporter)` | Run FMUs in every interface they offer (CS, ME) and tabulate what passed; see [FMI import compatibility](../guides/fmi_compatibility.md) |

OpenModelica does not need to be installed. If Docker is available, the helpers run omc from the official image
and mount only the working directory.

```python
from pinneapple_simulation.external_solvers.modelica.omc import build_fmu
from pinneapple_simulation.external_solvers.modelica.sweep import fmu_sweep

fmu = build_fmu("LadleCooling.mo", "LadleCooling", "build/")
ds = fmu_sweep(fmu, {"k_lin": (1.5, 4.0), "d_lin": (0.15, 0.35)}, n=80, stop_time=14400,
               output_interval=120, outputs=["T_melt", "T_shell"], step_size=10.0)
train, test = ds.split(16)
```

**Step size of Co-Simulation FMUs.** A Co-Simulation FMU from OpenModelica integrates with explicit Euler on the
communication step. fmpy uses the output interval as that step, so a coarse output grid can make the FMU
inaccurate or unstable. `fmu_sweep(step_size=...)` simulates on the fine step and interpolates onto the output
grid. The tests include a case that fails without it.

## Examples (`examples/modelica/`)

The models are in `examples/modelica/models/`:
- `DrainingTank.mo`: a tank with an inflow, emptying through an orifice by Torricelli's law;
- `LadleCooling.mo`: a hot-metal ladle as three lumped masses (melt, refractory lining, steel shell). The model
  includes conduction through the lining, radiation from the slag-covered surface, and convection plus radiation
  from the shell.

| Example | Result |
|---|---|
| `01_modelica_to_fmu.py` | FMU and omc agree within 0.13 K on the melt and 0.76 K on the shell over 4 h (fixed-step FMU against DASSL). The run is packed as a `PhysicalSample` |
| `02_ladle_surrogate.py` | MLP over 5 inputs and time, trained on 64 FMU runs and tested on 16 unseen ones: see below |
| `03_tank_inverse_pinn.py` | Discharge coefficient identified from 21 noisy level readings by a PINN: see below |

### Surrogate of the ladle model

![Ladle surrogate on unseen runs](../assets/modelica/ladle_surrogate.png)

The sweep has 80 Latin-hypercube runs over five inputs: lining conductivity and thickness, surface emissivity,
tapping temperature and shell convection. The trajectories last 4 h. On the 16 held-out runs:

| | Neural surrogate | Nearest training run |
|---|---|---|
| Melt temperature RMSE | 0.81 K (max 4.0 K) | 19.1 K |
| Shell temperature RMSE | 1.23 K (max 4.6 K) | 27.7 K |
| Time for the melt to reach 1350 °C, mean error | 0.6 min | 19.0 min |
| Steps where the predicted melt warms up | 0 of 1920 | |

The nearest-run baseline shows how much the sweep alone gives. The physical check is that the melt never warms up
in the predictions.

### Discharge coefficient by a PINN

![Tank inverse problem](../assets/modelica/tank_inverse_pinn.png)

**Setup.** The FMU runs with the true Cd = 0.62. The readings are 21 levels over the first 300 s, with 5 mm noise.
The PINN fits the readings and the mass balance A dh/dt = q_in − Cd a √(2gh) at 200 collocation points over
0–900 s, with Cd as a trainable parameter.

**Results.**

| | Cd | Level error over 300–900 s (no readings) |
|---|---|---|
| PINN | 0.627 (1.2 % off) | 2.9 mm |
| Finite differences on the readings, averaged | 0.628 ± 0.020 | |
| Network fitted to the readings only | | 57 mm |

On Cd alone, the PINN is no better than averaged finite differences. What it adds is a level model that follows the
mass balance where there are no readings.

Tests: `tests/test_modelica_omc.py` and `tests/test_fmi_runner.py`. The omc tests check the Torricelli
equilibrium, the parameter override, the sweep and the Latin-hypercube strata. They skip when neither omc nor the
Docker image is present.
