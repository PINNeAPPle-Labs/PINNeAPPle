# Earth-system building blocks and data assimilation

Four verified pieces for climate, weather, hydrology and digital-twin work. Each is checked against an exact solution
or a standard test case (`tests/test_geophysics.py`, `tests/test_data_assimilation_conservation.py`). Shortcuts:
`pp.geophysics` and `pp.assimilation`.

## Shallow water on the sphere (`pinneapple_simulation.geophysics.sphere_shallow_water`)

Spectral-transform model in vorticity-divergence form on a Gaussian grid (triangular truncation T_N, RK4, optional
del^4 hyper-diffusion), with the test suite of Williamson et al. (1992, J. Comput. Phys. 102:211):

```python
from pinneapple_simulation.geophysics import SphericalHarmonics, williamson_case6, stable_dt

sh = SphericalHarmonics(42)                      # T42: 64 x 128 Gaussian grid
model = williamson_case6(sh)                      # Rossby-Haurwitz wave, wavenumber 4
model.run(14 * 86400, dt=600.0, diagnostics_every=86400)
fields = model.fields()                           # u, v, h, zeta, delta, hs on the grid
model.history                                     # mass, total energy, potential enstrophy per day
```

| Case | What it checks | Result at T42 |
|---|---|---|
| 2, steady geostrophic flow (alpha = 0, pi/4, ~pi/2) | exact steady state, flow over the poles | normalised l2 height error ~1e-13 after 5 days |
| 5, flow over a mountain | topography, nonlinear flow | stable for 15 days; mass exact, energy drift 2e-8 |
| 6, Rossby-Haurwitz wave | long-time stability, wave propagation | eastward at ~11 deg/day (barotropic theory 12.2); energy drift 2e-8 in 14 days |

Mass is conserved to round-off by construction (the divergence of the mass flux is analysed directly).

## Richards equation (`pinneapple_simulation.geophysics.richards`)

1-D unsaturated flow in mixed form with the modified Picard iteration of Celia et al. (1990), mass-conservative.
Soils: van Genuchten-Mualem, Brooks-Corey, Clapp-Hornberger / Campbell, Gardner. Boundary conditions: head, flux,
free drainage; optional sink (root uptake).

```python
from pinneapple_simulation.geophysics import CELIA_1990_SOIL, Boundary, solve_richards

r = solve_richards(CELIA_1990_SOIL, depth=100.0, n_cells=100, h0=-1000.0, t_end=86400.0,
                   top=Boundary("head", -75.0), bottom=Boundary("free_drainage"))
r.h[-1], r.theta[-1], r.mass_balance_ratio        # profiles after one day; ratio = 1.000000
```

Verified against the exact steady infiltration profile of the Gardner soil (second-order convergence).

## Two-layer energy-balance model (`pinneapple_simulation.geophysics.energy_balance`)

Upper ocean and deep ocean (Held et al. 2010; Geoffroy et al. 2013), with efficacy of deep-ocean heat uptake:
exact integration of annual forcing, closed-form step response, ECS and TCR, and `fit_two_layer` for estimating
the parameters from a warming series and its forcing. An abrupt-forcing run identifies every parameter. A historical
ramp identifies the transient response (TCR) but not lambda and the deep-ocean capacity separately; the fit returns
their correlation (about -1) so this is visible.

## 4D-Var through autograd (`pinneapple_analysis.data_assimilation`)

Strong-constraint 4D-Var where the model is any PyTorch function (a solver written in torch, a neural surrogate or a
hybrid), so the adjoint comes from autograd. Control variable B^{-1/2}(x0 - xb), L-BFGS, observation operators and
covariances per observation time, `gradient_check` (the adjoint test) and a Lorenz-96 identical-twin experiment.

```python
from pinneapple_analysis.data_assimilation import Observation, Var4D, lorenz96_step

var = Var4D(model=lorenz96_step, B_sqrt=1.0)
result = var.analyse(x_background, [Observation(step=k, y=y_k, H=lambda x: x[::2], R=1.0) for k, y_k in obs])
result.x0, result.trajectory
```

## Exact conservation for surrogates (`pinneapple_physics.conservation`)

`project_integral` corrects a predicted field so that its weighted integral (mass, energy, water) equals a target
exactly: additive (smallest correction), multiplicative (keeps the pattern) or positive (clip then rescale, the
global mass fixer). `ConservationProjection(model, weights)` wraps a network so that a rollout keeps the total of its
input; it stays differentiable and can be used during training.

These methods are implemented from the cited papers.
