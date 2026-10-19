# Inverse Model Studio (day 21)

**From measurements back to the unknown parameters and fields: one problem, five methods, the same data, checked.**
Live at <https://inverse-studio.pinneapple.org> (port 8104). This is a separate app from the
[Inverse Heat Lab](https://inverse.pinneapple.org) (`apps/inverse_heat`), which is unchanged. The studio reuses
that lab's physics (the fin and plate physics-informed networks and the finite-volume references) through
`pinneapple_analysis.inverse_problems`, and links to it.

![The h(x) field of a fan-cooled fin, reconstructed by four methods from twelve thermocouples](../../docs/assets/apps/d21-inverse-studio-field.jpg)

## Pain

Inverse problems are solved one way, with whatever tool is at hand, and the answer is rarely checked:

- Was a PINN needed, or would least squares do?
- Does the adjoint solution agree with the Bayesian one?
- Is the reconstructed field right where nobody measured?
- Can these sensors determine the unknowns at all?

Each method usually lives in a different script with different data handling, so they are never compared on the same
data.

## Input

- **Measurements**: typed, or uploaded as CSV, Excel, Parquet, JSON or a sensor export, with column mapping.
  **Held-out measurements** are never fitted, and they rank the methods.
- **A physical model and its geometry**, in one of three forms:
  - Built in: pin fin (analytic), fin with zone-wise h(x) (finite volumes, adjoint), 2D heat-spreader plate (finite
    volumes), lumped-capacitance cooling, Arrhenius kinetics.
  - Your formula.
  - A table of your simulated runs.
- **The unknowns**: parameters (h, k, heat capacity, kinetic constants), boundary conditions (the wall temperature),
  or the zones of a field (h(x)). Each comes with bounds and log scale, or is loaded from a `ParameterSpace`.
- **The methods to compare**, the Bayesian prior width, and the field smoothness.

## Output

- **Estimates from every method side by side**, each with its 95 % interval where the method gives one.
- **The identifiability diagnosis** (day-25 core). A non-identifiable unknown gets no interval, except the Bayesian
  one, which is labelled as set by the prior.
- **Reconstructed physical fields** per method: the temperature profile, the h(x) field, and the plate temperature map
  (the PINN's own field too). On the examples, each is shown against the known truth with its error.
- **Residuals** at the measurements per method, **held-out RMSE**, and the ranking. Ties to 3 significant digits go to
  the cheaper method.
- **Truth recovery** on the examples, and **cost**: model runs and time per method. **Convergence** histories.
- **Downloads**: the report (JSON), a `StudyResult` (`study_type="inverse"`), and CSV files of the estimates and the 1D
  fields.

## Methods

One formulation (`InverseProblem`) and one call (`compare_methods`) in `pinneapple_analysis.inverse_problems`:

| Method | How | Uncertainty |
|---|---|---|
| Classical least squares | trust-region reflective in scaled parameters; Jacobian from the model's exact sensitivities, autograd or central differences | Gauss-Newton 95 % |
| Adjoint gradient | L-BFGS-B with the gradient from one adjoint solve of the discretised model (`A^T λ = G^T w`), whatever the number of unknowns | Gauss-Newton 95 % |
| Automatic differentiation | L-BFGS-B with the gradient by PyTorch autograd through the model | Gauss-Newton 95 % |
| Bayesian regularisation | MAP with a Gaussian prior and a first-difference smoothness prior on field zones (Tikhonov); Laplace covariance and adaptive-Metropolis posterior for fast models | posterior 95 % |
| Physics-informed network | the Inverse Heat Lab's PINN learns the field and h together from the PDE and the readings (fin, plate) | none from one run |

The adjoint gradient and the finite-volume sensitivities are checked against finite differences (relative error
below 1e-6). The finite-volume fin is checked against the analytic fin (within 0.005 K) in `tests/test_inverse_core.py`.

## What the examples show

Measured on this server's examples (synthetic measurements from a known truth plus Gaussian noise, fixed seeds):

- **Fin with a fan, h(x) in 4 zones** (12 thermocouples ±0.15 °C; truth 90 / 45 / 20 / 30 W/m²K):
  - Least squares, adjoint and autodiff give the same h1..h4 (91.1 / 44.0 / 23.6 / 27.3). They differ in cost: 11
    model runs (exact sensitivities), 38 (adjoint) and 47 (autodiff).
  - The Bayesian method with smoothness 0.5 gives 91.2 / 43.8 / 23.9 / 27.2, with narrower intervals.
  - All four contain the truth inside their 95 % intervals.
  - Held-out RMSE: 0.173 °C (0.177 for the Bayesian method).
  - The temperature field is within 0.15 °C of the truth everywhere. The h(x) field is within 3.6 W/m²K.
- **Unknown wall temperature** (5 thermocouples on the outer fin): T_base 108.7 [107.1, 110.3] °C for a true 110, and
  h 33.9 [32.4, 35.4] for a true 35, by all four methods.
- **Inverse Heat Lab pin fin** (h only): least squares 24.97 [24.15, 25.82]. The PINN gives 24.88 in 1500 steps
  (25 s), with no interval. Held-out RMSE 0.405 vs 0.410 °C.
- **Inverse Heat Lab plate** (2D):
  - Least squares on the finite-volume model gives 14.96 [14.84, 15.08] for a true 15.
  - The PINN, in 1500 steps (96 s), gives 14.0: it misses the truth by 7 %. Its own temperature field still puts the
    hot spot at 78.99 °C against 79.14 °C true.
  - The page shows both.

## Integrations

- **REST API** at `/docs`:
  - `POST /api/solve` queues a problem; `GET /api/jobs/{id}` returns its status and progress (with the PINN's live h)
    and the result.
  - `POST /api/methods` lists the methods that apply.
  - `GET /api/presets/{id}` returns a complete, runnable request.
- **Shared job queue**: API keys (`X-API-Key`), usage per caller, and uploads.
- **Contracts**: `ParameterSpace` in, `StudyResult` out (`pinneapple_data.contracts`).
- **The same in Python**: `inv.compare_methods(inv.InverseProblem(model, obs, params, validation), methods)`.

## Limits

- **Model form**: every method fits the same model. If the model misses a physical effect, they can all agree and all
  be wrong. Read the held-out error and the residuals.
- **The adjoint** needs a discretised model with an adjoint solver: the finite-volume fin, today. **Autodiff** needs a
  PyTorch model. **The PINN** needs the fin or plate formulation, with h as the only unknown. The page says why a
  method does not apply.
- **Intervals**: Gauss-Newton intervals assume local linearity. The Bayesian interval depends on the prior. A single
  PINN run gives none.
- **No 3D**: the Inverse Heat Lab's 3D block (about 30 minutes of training) is not run here; see it in the lab.
- **Your own code is not run on the server.** Upload a table of solver runs, or use the library call.
- **Size**: at most 16 unknowns and 5000 measurements per problem.

## Monetization

Charged per inverse problem, by unknowns and runs. The metered units, per caller and month, are `problems`,
`method_runs` (one per method and problem) and `pinn_runs` (network trainings, the costly ones), with free limits from
`INVS_FREE_*`. A team plan raises the limits and adds API keys. Engineering services (an adjoint for a customer's
solver, a field reconstruction study) build on the same core.

## Run it

```bash
pip install -e . -r apps/inverse_studio/requirements.txt
cd apps/inverse_studio && uvicorn inverse_studio.api:app --port 8104
# or: docker build -f apps/inverse_studio/Dockerfile -t inverse-studio . && docker run -p 8104:8104 inverse-studio
```

| Variable | Default | Meaning |
|---|---|---|
| `INVS_USER` / `INVS_PASSWORD` | unset | HTTP Basic login on everything but `/health` |
| `INVS_MAX_JOBS` | 2 | problems solving at once over all workers (a PINN uses a core for 25 to 95 s) |
| `INVS_FREE_PROBLEMS`, `INVS_FREE_METHOD_RUNS`, `INVS_FREE_PINN_RUNS` | 50, 250, 10 | free use per caller and month (`unlimited` lifts it) |
| `INVS_ADMIN_TOKEN` | unset | `X-Admin-Token` to manage API keys without the login |
| `INVS_HEAT_LAB_URL` | https://inverse.pinneapple.org | link to the Inverse Heat Lab |
| `INVS_DATA_DIR` | `/data/inverse_studio` | SQLite with jobs, keys and usage; uploads |

Tests: `pytest tests/test_app_inverse_studio.py`.
