# Parameter Discovery

**Can a physics-informed network recover a hidden material property from a few noisy readings and the PDE alone?**

One fixed test problem (dimensionless): a bar with u(0, x) = sin πx and both ends at 0, governed by
u_t = α u_xx. The diffusivity α is hidden (α = 0.15). 256 readings at t = 0.5, drawn from the exact solution
exp(−π² α t) sin πx with Gaussian noise σ = 0.02, are the only data. A PINN starts from a guess 3.3× too high
(α = 0.5) and learns the field and α together (2,500 Adam steps, one loss: PDE residual + initial/boundary conditions
+ data misfit). Next to the estimate, the Fisher information of those readings gives the Cramér-Rao bound on the
standard error of any unbiased estimate of α.

Ported from the `poc_parameter_discovery` case study of PINNeAPPle-apps.

## What it computes

All of it is `pinneapple_analysis.inverse_problems.heat_diffusivity.run_inverse_discovery`, built on
`pinneapple_physics.pinn_solver.factory` (`PINNProblemSpec(inverse_params=["alpha"])`, `PINN(...,
inverse_params_names=["alpha"])`, the call sequence of `examples/pinn_solver/04_factory_inverse_parameter_heat_1d.py`)
and `pinneapple_analysis.inverse_problems.sensitivity` (`LocalSensitivity`, `IdentifiabilityAnalyzer`). The app runs
it in a background thread and serves the page.

## Results

Measured here, full settings (2,500 steps, 256 readings), single-threaded torch on a shared container:

| Seed | α recovered | error | field rel. L2 at t = 0.5 | Cramér-Rao SE | wall time |
|---|---|---|---|---|---|
| 0 | 0.14669 | 2.2 % | 6.9·10⁻³ | 7.2·10⁻⁴ | 257 s |
| 1 | 0.14818 | 1.2 % | 4.2·10⁻² | 7.8·10⁻⁴ | 267 s |

The source project reported one run at α = 0.14692 (2.06 %), of the same order. Its README also said the error was
"well inside" the Cramér-Rao bound; that is not what the numbers say: the errors here (0.0033 and 0.0018) are
2.3-4.6 times the bound, i.e. the PINN estimate is not efficient, as expected of an estimate that also carries
network and optimisation error. With several torch threads competing on a loaded machine a run took over 100 min;
keep `OMP_NUM_THREADS=1` (the Dockerfile sets it).

## Run it

```bash
# from the repository root
pip install -e . fastapi "uvicorn[standard]"
cd apps/parameter_discovery
PYTHONPATH=../_shared uvicorn parameter_discovery.api:app --port 8183      # open http://localhost:8183
```

A run starts at startup. `GET /api/discovery/status` (progress: step, loss, current α; the result when done),
`POST /api/discovery/run` (409 while running), `GET /health`.

Environment: `PDISC_USER` / `PDISC_PASSWORD` (or `PDISC_PASSWORD_HASH`), `PDISC_AUTOSTART=0`, `PDISC_EPOCHS`
(default 2500), `PDISC_OBSERVATIONS` (default 256).

## Limitations

- Synthetic readings from the exact solution of the same model the PINN assumes: there is no model error, so this
  shows the mechanism, not a calibration of a real bar.
- The Cramér-Rao bound is for an unbiased estimator of the exact model at those sensor positions; the PINN estimate
  also carries network and optimisation error and is not guaranteed to reach the bound.
- Unseeded runs vary (network initialisation, collocation points, noise).
