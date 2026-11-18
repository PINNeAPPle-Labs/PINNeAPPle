# Steady heat conduction through a wall, cooled by convection

**Contract** `d245688bdffbaec2da721335b0939b3c` · mode `physics_ai` · family `heat_wall` · decision **APPROVED**

## What was asked

> a 12 mm steel wall with k = 200 W/mK, 300 degC inside and air at 25 degC outside with h = 12 W/m2K

## Quantity of interest

- **cold-face temperature**: `572.953 K` — from the convergence rung

## Every value the study used

| quantity | value | where it came from |
|---|---|---|
| thermal conductivity | 200 W/(m*K) | stated in the request: "k = 200 W/m*K" |
| convection coefficient | 12 W/(m2*K) | stated in the request: "h = 12 W/m^2*K" |
| hot-face temperature | 573.1 K | stated in the request: "300 degC" |
| ambient temperature | 298.1 K | stated in the request: "25 degC" |
| wall thickness | 0.012 m | stated in the request: "12 mm" |
| wall width | 0.04 m | default: 40 mm, the width of the two-dimensional model; the closed form does not depend on it |

## Equations

- `d/dx (k dT/dx) = 0   (one-dimensional steady conduction, no source)`
- `q'' = h (T_cold - T_inf)   (convection on the far face)`
- `T_cold = T_inf + q'' / h,  q'' = (T_hot - T_inf) / (L/k + 1/h)   (series resistance)`

## Assumptions

- Steady state and no volumetric source.
- Conduction across the wall is one-dimensional: the wall is long and wide against its thickness and its edges are insulated, so the two-dimensional solve should agree with the closed form.
- The conductivity is constant (temperature-independent), and the surface is grey and diffuse with a single convection coefficient over the whole face.

## The ladder

| rung | what it is | ran | time |
|---|---|---|---|
| analytic | series-resistance closed form | yes | 0.00 s |
| numerical | immersed finite cell, two-dimensional | yes | 0.00 s |
| convergence | the same solver on two grids | yes | 0.01 s |
| surrogate | a surrogate trained on the solver | yes | 0.00 s |

## Checks

| check | result | value | limit |
|---|---|---|---|
| cold_face_temperature: the solver against the series-resistance closed form | pass | 2e-06 | 0.05 |
| cold_face_temperature: change under refinement | pass | 1e-06 | 0.02 |
| the surrogate on designs held out of its fit | pass | 5e-06 | 0.05 |
| the closed form returns a finite far-face temperature | pass | 572.9521424574307 | finite |
| the solution is finite | pass | 572.9530731608912 | finite |
| energy balance of the discrete system | pass | -9.076522710021778e-12 | 1e-08 |
| the two-dimensional solve is within the family's tolerance of the closed form | pass | 2e-06 | 0.05 |
| the solution is finite | pass | 572.9535609760062 | finite |
| energy balance of the discrete system | pass | -2.2578846515966655e-11 | 1e-08 |
| the two-dimensional solve is within the family's tolerance of the closed form | pass | 2e-06 | 0.05 |
| the solution is finite | pass | 572.9530731608912 | finite |
| energy balance of the discrete system | pass | -9.076522710021778e-12 | 1e-08 |
| the two-dimensional solve is within the family's tolerance of the closed form | pass | 2e-06 | 0.05 |
| the surrogate was fitted and scored on held-out designs | pass | 2 | >= 1 |

## What this study does not cover

- No radiation: a hot face above roughly 400 K in still air loses a share of its heat by radiation that this study does not model.
- No contact resistance, no layered wall, no transient: the answer is the steady state of one homogeneous slab.
- The convection coefficient is an input, not a prediction: this study does not compute h from a flow.

## References

- Incropera & DeWitt, Fundamentals of Heat and Mass Transfer, ch. 3 (one-dimensional steady conduction, thermal resistance network).

## Parts of the request that were not used

- '2*K' was not used: no value of that kind in the Steady heat conduction through a wall, cooled by convection template

---

Every number above comes from a closed form, a solver call, or a value the contract records with its source. To reproduce the study without an agent: `python -m pinneapple_intent replay /Users/yanbarros/Documents/GitHub/wt-intent/examples/intent_to_study/_out/heat-wall-surrogate`.
