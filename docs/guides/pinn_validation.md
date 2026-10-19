# PINN validation: what the training loss does not say

A PINN's training loss is measured on its own collocation points and mixes the PDE residual with the boundary and
initial terms. A low loss is compatible with a near-constant field, a boundary condition given up for a smaller
residual, or a solution that never propagated from the initial condition. `pinneapple_analysis.validation.pinn`
evaluates the trained network on points it never saw and names the failure mode.

```python
from pinneapple_analysis.validation.pinn import burgers_1d, validate_pinn, PINNMonitor, train_pinn

rep = validate_pinn(net, burgers_1d(), train_points=X_collocation)
print(rep.summary())              # one line per check
rep.failure_modes                 # e.g. ["boundary condition not satisfied"], most specific first
rep.evidence["residual"]["map"]   # residual on the dense grid

import pinneapple as pp           # the same checks in the validation report
pp.validate(net, {"pinn": "burgers_1d"}, data={"train_points": X_collocation})
```

Any problem is a `PINNProblem`: bounds, a residual written with `d(u, X, i, order)` (autograd), samplers for the
boundary and initial conditions, and optionally the exact solution, the operator and forcing (`L[u] = f`) and the time
axis. Built in: `burgers_1d` (exact by Cole–Hopf), `poisson_2d` (manufactured) and `kovasznay` (steady
Navier–Stokes; pressure compared up to a constant).

## Checks

| ID | What | Fails when |
|---|---|---|
| `pinn.residual` | residual on a dense grid, fresh random points and adaptive points (around the largest residuals of a large sample), relative to the problem's residual scale | RMS > 0.05 or max > 0.3 |
| `pinn.trivial` | spread and gradient energy of the field against the exact solution's (or the problem scale) | below 0.2 of the reference |
| `pinn.causality` | residual by time; with an exact solution, error by time | residual resolved in the first quarter of the interval and over 10× larger in the last; or an error that grows above 2 % while the residual passes (compare Wang, Sankaran & Perdikaris 2022) |
| `pinn.boundary`, `pinn.initial` | conditions at fresh points, relative to the solution scale | above 1 %; when the residual passes, the summary says the condition was traded off against the residual |
| `pinn.spectrum` | high-band energy against the exact solution, or of `L[u]` against `f` | off by more than 25 % |
| `pinn.exact` | relative L2 error against the exact solution | above 2 % |

`PINNMonitor(problem, every=100, warmup=..., stop_on=(...))` runs the same checks during training (`monitor(model,
epoch)`) and sets `stop` once a failure mode persists for `patience` evaluations after the warm-up. Every network is
near-trivial at initialisation, so the warm-up matters. `train_pinn(problem, ...)` is a plain reference trainer
(Adam then L-BFGS, soft constraints) used by the tests.

## Results (from `tests/test_pinn_validation.py`, CPU)

The Cole–Hopf reference agrees with the pseudo-spectral Burgers solver to 1e-6. The exact Poisson and Kovasznay
fields pass every check with a residual below 1e-8.

| PINN | Diagnosis |
|---|---|
| Burgers (ν = 0.05/π), Adam + L-BFGS | passes; relative L2 error 7e-3 |
| Poisson (manufactured), Adam + L-BFGS | passes, with and without the exact solution; fresh / training residual ≈ 1 |
| either, stopped after 20 Adam steps | **near-trivial solution** (spread 0.06× and 0.006× of the exact solution's) |
| Poisson trained with boundary weight 0 | **boundary condition not satisfied**: residual passes (1.3e-3), boundary error 1.4× the solution scale |
| the converged Burgers PINN frozen after t = 0.3 | **propagation failure in time**: late/early residual 23 |
| Poisson, boundary weight 0, monitored | training stopped at step 300 on the boundary failure |

## Limits (honest results)

- The residual alone misses a near-trivial Burgers network: a nearly constant field almost solves Burgers, so its
  relative residual (2.6e-2) passes. The trivial check catches it.
- Without an exact solution, the spectrum check (`L[u]` vs `f`) cannot see a boundary error: the error is harmonic
  and has zero Laplacian. Only the boundary check sees it.
- The causality test was checked on a constructed failure (a network frozen in time), not on a PINN that failed to
  propagate on its own. With a smaller viscosity a real propagation failure is expected but was not reproduced
  within the CPU budget.
- The tolerances are relative to the problem's `scale` and `residual_scale`. For a new problem, set these from the
  magnitude of the solution and of the PDE terms.
