# Pressure drop of a laminar flow along a circular pipe

**Contract** `9ed6a06b186165046067807576f70c7b` · mode `simulation` · family `laminar_pipe` · decision **APPROVED**

## What was asked

> a 20 mm pipe, 3 m long, carrying 0.5 L/min of water, viscosity 1 mPa*s

## Quantity of interest

- **pressure drop**: `6.36634 Pa` — from the convergence rung

## Every value the study used

| quantity | value | where it came from |
|---|---|---|
| pipe diameter | 0.02 m | stated in the request: "20 mm pipe" |
| dynamic viscosity | 0.001 Pa*s | stated in the request: "1 mPa*s" |
| volumetric flow rate | 8.333e-06 m^3/s | stated in the request: "0.5 L/min" |
| pipe length | 3 m | stated in the request: "3 m long" |
| density | 998.2 kg/m^3 | default: water at 20 °C (998.2 kg/m3); change it for another fluid |

## Equations

- `mu (1/r) d/dr (r du/dr) = dp/dx   (steady, fully developed, axisymmetric momentum balance)`
- `u(R) = 0 and du/dr(0) = 0   (no-slip at the wall, symmetry on the axis)`
- `dp = 8 mu L Q / (pi R^4)   (Hagen-Poiseuille, the closed form the solver is checked against)`
- `Re = rho u D / mu, laminar when Re < 2300`

## Assumptions

- Steady, fully developed, laminar and incompressible flow.
- A Newtonian fluid of constant viscosity, in a straight circular pipe of constant section.
- The entrance length and the local losses (bends, fittings, a valve) are not part of this study: it prices the straight run only, so a real circuit needs those added.
- No roughness: at the Reynolds numbers this study admits, a laminar profile does not feel it.

## The ladder

| rung | what it is | ran | time |
|---|---|---|---|
| analytic | Hagen-Poiseuille closed form | yes | 0.00 s |
| numerical | finite differences on the radial momentum balance | yes | 0.00 s |
| convergence | the same solver on two grids | yes | 0.00 s |

## Checks

| check | result | value | limit |
|---|---|---|---|
| pressure_drop: the solver against the Hagen-Poiseuille closed form | pass | 2.3e-05 | 0.05 |
| pressure_drop: change under refinement | pass | 0.006621 | 0.02 |
| the flow is laminar, where the closed form applies | pass | 529.6 | 2300.0 |
| the discrete profile is positive everywhere | pass | 3.9269002307059145e-06 | > 0 |
| the finite-difference solve is within the family's tolerance of the closed form | pass | 2.3e-05 | 0.05 |

## What this study does not cover

- Turbulent flow is outside the family: above the Reynolds number this study reports, the answer it gives is the answer to a different problem.
- No heat transfer, no compressibility, no non-Newtonian rheology, no transient start-up.
- The pipe is a straight line: elbows, tees, valves, and the developing region near the inlet are not priced.

## References

- Hagen, G. (1839) and Poiseuille, J. L. (1840): the fourth-power law for laminar pipe flow.
- White, F. M., Viscous Fluid Flow, ch. 3 (the Hagen-Poiseuille solution and f = 64/Re).

---

Every number above comes from a closed form, a solver call, or a value the contract records with its source. To reproduce the study without an agent: `python -m pinneapple_intent replay /Users/yanbarros/Documents/GitHub/wt-intent/examples/intent_to_study/_out/pipe-laminar`.
