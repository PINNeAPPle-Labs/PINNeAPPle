# Derivative trees across physics

In thermodynamics one fundamental potential in its natural variables — the Helmholtz energy `A(T, V)`, the Gibbs
energy `G(T, p)` — generates every property by differentiation. Organised by order, the derivatives form a tree:
depth 1 gives the "forces" (entropy, pressure), depth 2 the response functions (heat capacities, compressibility,
expansion, speed of sound), and so on. Legendre transforms move between potentials; Maxwell relations tie the
branches together. Because every property comes from one function, the properties are consistent by construction.

The same structure appears across physics, and `pp.potential` (`pinneapple_physics.potential`) treats it once:

| Domain | Potential | Natural variables | Depth 1 | Depth 2 | Adapter |
|---|---|---|---|---|---|
| Thermodynamics, EOS | `A(T, V)`, `a(T, rho)`, `G(T, p)` | T, V (or rho), p | S, p, U, H, G | c_v, c_p, kappa_T, alpha_p, w | `thermo` |
| Hyperelasticity | `W(F)` | deformation gradient | P = dW/dF, Cauchy stress | tangent d2W/dFdF | `hyperelastic` |
| Electric machines | co-energy `W'(i, theta)` | currents, rotor angle | flux linkages, torque | differential inductances | `machines` |
| Analytical mechanics | `L(q, qd)`, `H(q, p)` | coordinates, velocities / momenta | momenta, generalised forces, energy | mass matrix, coupling | `mechanics` |

## A potential and its tree

```python
import numpy as np
from pinneapple_physics.potential import thermo, tree, maxwell_relations

A = thermo.van_der_waals(a=0.3640, b=4.267e-5, c=2.5, molar_mass=0.0440098)   # CO2 constants, SI, per mole
t = tree(A, depth=2, at={"T": np.array([280.0, 320.0]), "V": np.array([2e-4, 4e-4])})
print(t)                                    # every node: depth, symbol, unit, value
print(t["p"], t["cp"], t["w"])              # named quantities and responses
print(t.symbolic_agreement())               # autodiff against SymPy derivatives, per node
print(maxwell_relations(A)[0].text)         # (dS/dV)_T = (dp/dT)_V
```

A potential is any differentiable function written in torch (`Potential(fn, {"T": "K", "V": "m3/mol"}, names=...,
responses=...)`) or a SymPy expression (`Potential.from_sympy`). Variables can be tensors: for `W(F)` the tree holds
`P` with shape `(3, 3)` and the tangent with shape `(3, 3, 3, 3)` per point.

## Legendre transforms and where they fail

`legendre(potential, swap={"V": "p"}, conjugate_sign=-1)` gives `G(T, p) = A + p V` (`thermo.gibbs(A)` is the same
call with the Gibbs names), and `mechanics.hamiltonian(L)` gives `H(q, p) = p qd - L`. The transform is one-to-one only
where the potential is convex (or concave) in the swapped variables. With `bounds=` the convexity is checked on a grid
and reported: for van der Waals below its critical temperature, `A` is not convex in `V` between the two spinodals,
and `ConvexityReport.intervals("V")` gives that interval at each temperature — the test suite checks it against the
closed-form spinodal.

```python
from pinneapple_physics.potential import convexity
rep = convexity(A, "V", {"T": (250.0, 330.0), "V": (6e-5, 6e-4)}, n_grid=81)
print(rep)                                  # convex in V: no (share of stable grid points)
print(rep.intervals("V")[0])                # ({'T': 250.0}, [(V_lo, V_hi)]): the unstable interval
```

## Is a model or a table consistent with one potential?

`consistency_check(model, family, at=...)` evaluates the identities a potential satisfies — the Maxwell relation,
`c_v = T (ds/dT)_v`, the energy equation, the Mayer relation, the speed of sound — on anything that predicts the
properties separately: a multi-output surrogate, a lookup table, or measurements on a grid (`{"axes": ..., "values":
...}`, differentiated with `np.gradient`). Each identity is reported with its relative violation (median, 95th
percentile, maximum). It is to thermodynamic consistency what the conservation checks of the trust layer are to
conservation laws.

## Known answers

`tests/test_potential.py` (must pass in CI):

| Case | Reference | Measured |
|---|---|---|
| Ideal gas: p, S, c_v, c_p, kappa_T, alpha_p, w | closed forms | 1e-13 relative |
| G(T, p) by symbolic and numeric Legendre transform | each other, to third order; `A + pV` | 1e-9 (relative to the node scale) |
| Van der Waals: p, (dp/dV)_T, c_p - c_v, critical point, spinodal | closed forms | 1e-11; spinodal within one grid step |
| Peng-Robinson, CO2, 4 states: p, (dp/dT)_V, residual c_v | CoolProp `PR` backend (optional) | 1e-15 to 2e-13 relative |
| Neo-Hookean: P and the 3x3x3x3 tangent | Bonet & Wood (2008) closed forms | 1e-9 relative; major symmetry 1e-14 |
| Double pendulum: accelerations, energy, Hamiltonian | closed-form equations of motion; T + V | 1e-11; RK4 drift 6.9e-8 over 5 s |
| Surface-PM machine (linear iron): torque from the co-energy potential | Maxwell-stress (Arkkio) torque of the same FEA | 16.854 vs 16.837 N m (0.10 %); flux linkages to 1e-14 Wb |

## Measured study: a potential-based surrogate against a direct one (CO2 near the critical point)

`surrogate.PotentialSurrogate` learns the Helmholtz energy `a(T, rho)`, and every property is a derivative of it.
Its MLP and input-convex (`convex=True`, convex in `v`) variants were compared with `surrogate.DirectSurrogate`,
which has one output per property, on mixed CoolProp data for CO2 (Span-Wagner): each training sample carries one
of `p, s, u, c_v, c_p, w`. The data, loss, optimiser and steps were the same for every model
(`examples/potential_co2`, full tables in `results/report.md`).

The table gives the RMS error divided by the standard deviation of the property, averaged over the six properties,
as the mean and 95 % t-interval over seeds. The training region is T 305-335 K, rho 100-800 kg/m3.

| configuration (seeds) | direct | potential (MLP) | potential (ICNN) |
|---|---|---|---|
| all six properties, 3000 steps (5): training region | 0.10 [0.07, 0.13] | 1.61 [1.60, 1.62] | 1.24 [0.79, 1.69] |
| the same, extrapolated to T 335-365 K | 0.25 [0.20, 0.30] | 1.97 [1.91, 2.02] | 3.24 [1.43, 5.05] |
| the same, extrapolated to rho 800-950 kg/m3 | 0.66 [0.62, 0.71] | 3.20 [3.19, 3.21] | 2.69 [2.27, 3.10] |
| no c_p in training, 3000 steps (5): training region | 0.22 [0.21, 0.23] | 1.57 [-0.89, 4.02] | 3.49 [0.72, 6.27] |
| all six properties, 9000 steps (2): training region | 0.04 [-0.04, 0.11] | 1.51 [-1.08, 4.10] | 1.30 [-7.03, 9.62] |
| Maxwell relation, 95th percentile of the violation (first row) | 0.41 | 2.8e-5 | 1.1e-6 |

**The potential approach lost on accuracy.** The direct surrogate was more accurate in every region of both 5-seed
configurations, and no region favoured the potential. What the potential bought was consistency: its identities held
to between 1e-7 and 1e-4, where the direct surrogate's were off by 0.3 to 14.

Two failure modes showed up:

- With c_p in the mixed data, the MLP potential collapsed to near-zero pressure. Near the Widom line the c_p peak
  needs a nearly flat isotherm, and fitting it pulled the potential away from p.
- The input-convex network keeps `(dp/drho)_T > 0`, but nothing keeps `c_v > 0`, and in the hotter region it was
  not.

Training took 4 to 10 times longer per step, because the loss differentiates twice.

Not measured:

- tuning for each model;
- longer budgets with five seeds;
- subcritical and two-phase states, other fluids and noisy data.


## Flow nodes

`potential.eos_tree`, `potential.hyperelastic`, `potential.double_pendulum`, `potential.machine_coenergy` and
`potential.co2_surrogates` (one seed at a reduced budget; the example flow `potential_co2/surrogate_study`).

## Not covered

- Mooney-Rivlin is tested at small strain only; the hyperelastic potentials are not yet wired into the FEM code.
- The machine co-energy map has no angle-current cross terms, so it does not give `dpsi/dtheta` (the reciprocity
  with `dT/di`).
- Legendre transforms of tensor variables (swap their components instead).
- Mixtures (`G(T, p, n_i)`), plasticity and damage potentials: the classes take them, but no adapter or known-answer
  test is provided.
