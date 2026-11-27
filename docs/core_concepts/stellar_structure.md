# Stellar structure: from Lane-Emden to a stellar model (`pinneapple_physics.astro.stellar`)

Issue #389. The four equations of stellar structure written as residuals a physics-informed network can consume, an
analytic microphysics declared with its valid range, a classical solver that converges (and a measured account of
what it cannot reproduce), and the invariants the result is checked against.

MESA is the declared reference of the issue and is **not measured here**: it is not installed in this worktree and
there is no network. `stellar.MESA_PLAN` records exactly what is missing, the exact command and inlist that would
produce the comparison, and `compare_with_mesa` is the loader for a MESA `LOGS/profile*.data` table, tested against a
synthetic MESA-shaped table. The substitute reference used in this page is the published Sun (IAU 2015 for L, R and
T_eff; Christensen-Dalsgaard et al. 1996 and Basu & Antia 1997 for the interior values).

```python
from pinneapple_physics.astro import stellar

model = stellar.stellar_model(1.0, 0.02)          # ~3 s, cached per (mass, composition, kappa_scale)
model["radius_rsun"], model["luminosity_lsun"], model["t_center"]
model["solver"]["photospheric_pressure_ratio"]    # the closure, measured (see the last section)
stellar.solar_calibration()                       # the same model against the published Sun, as data
```

## The four equations as residuals

In the mass coordinate `m` (`x = m/M`), with `grad = min(grad_rad, grad_ad)` (Schwarzschild):

| Equation | Residual |
|---|---|
| continuity | `dr/dm - 1/(4 pi r^2 rho)` |
| hydrostatic equilibrium | `dP/dm + G m/(4 pi r^4)` |
| energy transport | `dT/dm + G m T grad /(4 pi r^4 P)` |
| energy generation | `dL/dm - eps` |

`stellar.structure_residuals(m, r, P, T, L, dr_dm=..., dP_dm=..., dT_dm=..., dL_dm=..., composition=...)` returns all
four, scaled by `R_sun/M_sun`, `P_REF/M_sun`, `T_REF/M_sun` and `EPS_REF` so a single loss weight is meaningful, plus
the microphysics that produced them (`rho`, `kappa`, `epsilon`, `grad`, `grad_rad`, `grad_ad`, `convective`). A PINN
evaluates the derivatives by autograd and passes them in; the test re-feeds the classical model's own profiles and
finds every residual small in the interior 20–95 % band.

## Declared microphysics, with its range

**Equation of state.** Ideal gas plus radiation, `P = rho k T/(mu m_u) + a T^4/3`, `1/mu = 2X + 3Y/4 + Z/2`
(fully ionised). For Z = 0.02 and the enrichment law `Y = 0.2465 + 1.5 Z`: X = 0.7035, mu = 0.6156, mu_e = 1.1718.

* Valid while the gas is fully ionised: `T >~ 2e5 K` for hydrogen and helium. It has **no ionisation**, and that is
  the approximation this model pays for (see the last section).
* `rho = (P - aT^4/3) mu m_u/(k T)`, floored at a positive value so a Newton iteration sees a large residual instead
  of a NaN.

**Opacity.** Analytic Rosseland mean, four terms in SI (`kappa` in m²/kg):

| Term | Form | Range |
|---|---|---|
| electron scattering | `0.02 (1+X)` | exact (Thomson), every temperature; the whole opacity above ~5e6 K |
| bound-free + free-free | `K_scale (Z/0.02)(1+X) [K_BF + K_FF] rho T^-3.5`, tapered by `exp(-(1e5/T)^6 - (T/T_max)^2)` | 1e5–1e7 K |
| H^- | `K_HM (Z/0.02) rho^0.5 T^9 exp(-(T/1e4)^3)` | 3e3–1e4 K |

`K_BF + K_FF` is fixed by the opacity bump at the base of the solar envelope, `kappa(200 kg/m^3, 2e6 K) = 12 m^2/kg`
(120 cm²/g), and clipped to `[0.02(1+X), 1e3]` m²/kg. Measured: `kappa(200, 2e6) = 12.0` m²/kg;
`kappa(1.5e5, 1.57e7) = 0.65` m²/kg at the solar centre, against the physical ~0.034 m²/kg — the Kramers law is
**19× too high there** even with the `T_max = 1e7 K` taper, because the taper is applied at too high a scale. This is
a defect of the declared fit, not of the solver; it changes the interior gradient but not the conclusion below.

**Nuclear energy.** The Kippenhahn & Weigert (1990, Sec. 18) analytic pp-chain and CNO-cycle fits, converted to SI:

| Check | Measured | Reference |
|---|---|---|
| `eps_pp(1.5e5 kg/m^3, 1.5e7 K, X = 0.70)` | 3.41e-3 W/kg | 3.41e-3 W/kg from the same fit restated in cgs |
| `eps_pp(1e4 kg/m^3, 1.5e7 K, X = 0.70)` | 2.27e-4 W/kg | 2.4e-4 W/kg |
| `d ln eps_pp/d ln T` at 1.5e7 K | 3.89 | ~4 (pp) |
| pp/CNO crossover, Z = 0.02 | 1.8e7 K | 1.3–1.5 M_sun main-sequence turn |

The 2.4e-4 W/kg figure is the same rate at `rho = 1e4 kg/m^3`; at the solar central density it is 3.4e-3 W/kg, and
`L_sun = int eps dm` closes to 1e-4 relative on this model, which is the check that matters.

**Transport.** Radiative diffusion plus convection in the adiabatic limit: `grad = grad_ad` wherever
`grad_rad > grad_ad`. There is no mixing-length theory and no `alpha_MLT`; the convective zones are isentropic. This is
the second declared simplification and it is the one that costs the radius.

## Lane-Emden, the polytropic anchor

`stellar.lane_emden(n)` integrates `theta'' + (2/xi) theta' + theta^n = 0` to the first zero crossing.

| n | xi_1 measured | Reference (Hansen, Kawaler & Trimble Table 4.1) | closed form |
|---|---|---|---|
| 1 | 3.141593 | pi | `sin(xi)/xi`, max abs error 4e-8 |
| 1.5 | 3.653754 | 3.65375 | — |
| 3 | 6.896849 | 6.89685 | — |

## The converged model: measured, and against the Sun

`stellar_model(1.0, 0.02)` on this machine, cold cache: **3.3 s** (the previous implementation of the same call took
**3108 s**, 52 minutes, because the Newton it ran walked into central pairs that made the integrator grind for 45 s
each; see "what changed in the solver" below).

| Quantity | This model | Sun | ratio |
|---|---|---|---|
| Radius | 0.626 R_sun | 1 R_sun | 0.63 |
| Luminosity | 0.187 L_sun | 1 L_sun | 0.19 |
| Central temperature | 1.20e7 K | 1.57e7 K | 0.76 |
| Central density | 3.4e4 kg/m^3 | 1.5e5 kg/m^3 | 0.23 |
| Central pressure | 5.5e15 Pa | 2.3e16 Pa | 0.24 |
| Effective temperature | 4795 K | 5772 K | 0.83 |
| Mass reached by the integration | 1.000 M_sun | 1 M_sun | 1.00 |
| Convective mass fraction | 0.96 | 0.02 | — |

**The model does not reproduce the radius, and this page does not claim it does.** `solar_calibration()` returns
`verdict["status"] = "not_calibrated"` and lists the quantities it fails on; `trust_checks()` carries the same
failures as data, so a caller that trusts a card rather than a number sees them.

### The invariants that are actually measured

| Check | Measured | What it means |
|---|---|---|
| `|L_surface - int eps dm| / L_surface` | 1.2e-4 | the energy-generation equation closes against the boundary |
| worst zone `|dL/dm - eps| / max eps` | 4.7e-3 | pointwise, not just in the integral |
| `(2U + Omega + 4 pi R^3 P_s)/(2U)` | 2.0e-5 | scalar virial theorem: hydrostatic equilibrium as a whole |
| `eta = -Omega/U` | 2.000 | the n = 3 polytrope (Eddington standard model) value is 2 |
| structure residuals on the model's own profiles | small in 20–95 % of the mass | the model is a solution of the four equations, not a fit |

### Why the radius does not come out, and what the four candidate defects say

The four things worth checking were checked, measured:

| Candidate | Verdict |
|---|---|
| pp rate against the standard value | **correct**: 3.41e-3 W/kg at the solar centre, identical to the K&W fit restated in cgs; `L = int eps dm` closes to 1e-4 |
| `mu` of the composition | **correct**: 0.6156 for X = 0.7035, Z = 0.02 |
| luminosity integral closing `L(m=M) = L_surface` | **correct**: 1.2e-4 relative |
| opacity at the solar central density | **wrong**: 0.65 m²/kg where the plasma gives ~0.034, because the high-temperature taper is applied at `T_max = 1e7 K` instead of ~5e6 K |

None of these is what blocks the radius. Fixing the taper moves the structure (and was measured: it changes `L` by a
factor ~2 and leaves `R` at 0.4–0.5), and the radius still does not land. The defect is the **equation of state**:

* A stellar model's radius is set by the entropy of its convective envelope, and that entropy is set by where the
  envelope can put `T = T_eff`. For the model's own pressure range (`P_c ~ 5e15 Pa` down to the photosphere) the
  required gradient is `grad = ln(T_eff/T_c)/ln(P_phot/P_c) = 0.248`.
* A fully ionised ideal gas has `grad_ad = 0.4`, and the envelope is convective, so the model uses 0.4 — a 60 % too
  steep adiabat. The envelope is therefore too dense, the star too compact, and `T = T_eff` is reached at
  `P ~ 1e10 P_tau` instead of at `P_tau = 2GM/(3 kappa R^2)`.
* Measured confirmation: replacing `grad_ad` by 0.30 takes the photospheric pressure mismatch from `e^25` (7e10) to
  `4e2`, in the direction predicted. The missing physics is the H and He ionisation, which lowers the adiabatic
  gradient through exactly that range (and adds the H/He bound-free opacity bumps that set the solar envelope).

That is what a finished version needs, and it is a larger change than this issue: a Saha equation of state and its
`grad_ad`, and a better envelope opacity. Until then the honest statement is: **the equations are right, the solver
converges, the central values are close, and the radius is not reproducible by a fully ionised EOS.**

## What changed in the solver

| | Before | After |
|---|---|---|
| `stellar_model(1.0)` | 3108 s | 3.3 s |
| Mechanism of the cost | a damped Newton on the two central values with a line search that accepted steps into the `T_c -> 0` corner, where each trial integration ground through 1e6–3.5e6 right-hand-side evaluations (8–45 s each) | a coarse grid in `T_c`, a bisection in `P_c` for the mass (which is monotone there), and a runaway guard `r < 30 R_sun` that bounds every trial |
| Closure | over-determined in practice, and it diverged | lexicographic: the mass is met first, the photospheric mismatch is then minimised and *reported* |

Two more things were fixed on the way, both of which made the module unusable rather than slow: the mass-coordinate
integration cannot resolve a photosphere at `Delta m/M ~ 1e-8` unless the trial is bounded (hence the guard), and
`trust_checks` read a solver key that did not exist.

## Reproduced versus declared

**Reproduced in this repository, measured:** the four residuals; the Lane-Emden surfaces for n = 1, 1.5, 3; the pp
rate and its temperature exponent; the pp/CNO crossover; `mu`; the Schwarzschild switch; energy balance; the scalar
virial theorem; the central values within the bands in the table above; the runtime.

**Declared and not measured here:** MESA and any comparison against it (`MESA_PLAN`); the opacity table (the fit is
analytic, and it is shown above to disagree with the plasma at the solar centre); mixing-length theory (absent); mass
loss, rotation, magnetic fields (absent); the ionisation physics whose absence is the reason the radius is wrong.
