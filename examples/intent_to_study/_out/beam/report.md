# Tip deflection of a cantilever beam under an end load

**Contract** `1954c4276b241d775b5025ac082e50d9` · mode `simulation` · family `cantilever_beam` · decision **APPROVED**

## What was asked

> a cantilever beam 0.5 m long with E = 210 GPa, 20 mm deep, carrying 500 N at the tip

## Quantity of interest

- **tip deflection**: `0.000135243 m` — from the convergence rung

## Every value the study used

| quantity | value | where it came from |
|---|---|---|
| beam length | 0.5 m | stated in the request: "0.5 m long" |
| Young's modulus | 2.1e+11 Pa | stated in the request: "E = 210 GPa" |
| tip load | 500 N | stated in the request: "500 N" |
| section depth in the plane of bending | 0.02 m | stated in the request: "20 mm deep" |
| out-of-plane depth | 1 m | default: 1 m, the unit depth of the two-dimensional model; the load is entered as a total over it and divided by it |
| Poisson's ratio | 0.3 | default: 0.3, the usual value for steel and aluminium alloys |

## Equations

- `E' I d^2 w/dx^2 = M(x),  E' = E / (1 - nu^2)   (Euler-Bernoulli, plane strain)`
- `w(0) = 0, w'(0) = 0   (clamped root)`
- `delta = P L^3 / (3 E' I),  I = h^3 / 12   (tip load, one metre of depth)`
- `sigma_max = P L (h/2) / I   (outer fibre, bending only)`

## Assumptions

- Small deflection, linear elasticity, and a beam slender enough for Euler-Bernoulli: the closed form has no shear deformation, which is why the family reports the slenderness L/h next to the answer.
- The two-dimensional solver is plane strain, so the closed form compares against E / (1 - nu^2).
- The root is fully clamped (both displacement components fixed) and the section is rectangular, homogeneous and isotropic.

## The ladder

| rung | what it is | ran | time |
|---|---|---|---|
| analytic | Euler-Bernoulli closed form (plane strain) | yes | 0.00 s |
| numerical | immersed finite cell, plane-strain linear elasticity | yes | 0.44 s |
| convergence | the same solver on two grids | yes | 0.49 s |

## Checks

| check | result | value | limit |
|---|---|---|---|
| tip_deflection: the solver against the Euler-Bernoulli closed form (plane strain) | pass | 0.001279 | 0.05 |
| tip_deflection: change under refinement | pass | 0.005368 | 0.02 |
| the beam is slender enough for Euler-Bernoulli | pass | 25.0 | 10.0 |
| the solution is finite | pass | 0.00013524349229549122 | finite |
| the root reaction balances the applied load | pass | 499.99998455397963 | -500.0 |
| the solver is within the family's tolerance of the closed form | pass | 0.001279 | 0.05 |

## What this study does not cover

- No plasticity, no buckling, no contact and no fatigue: the answer is the elastic response to one load.
- Deflections above roughly a tenth of the span are outside the linear theory this compares with.
- The stress reported is the bending stress of the closed form, not the peak the solver finds at the clamp, where a real fillet and a real contact pressure change the number.

## References

- Timoshenko, Strength of Materials, part I, ch. 5 (cantilever with an end load, and the shear correction this closed form omits).
- Boresi & Chong, Elasticity in Engineering Mechanics (plane strain replaces E with E / (1 - nu^2)).

## Parts of the request that were not used

- '20 mm' was not used: no value of that kind in the Tip deflection of a cantilever beam under an end load template

---

Every number above comes from a closed form, a solver call, or a value the contract records with its source. To reproduce the study without an agent: `python -m pinneapple_intent replay /Users/yanbarros/Documents/GitHub/wt-intent/examples/intent_to_study/_out/beam`.
