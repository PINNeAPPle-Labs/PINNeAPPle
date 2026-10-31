# Biology-informed modelling

Well-mixed biological models (SBML or written in code), identifiability before any fit, parameter inference with
physics-informed networks (SBINN) and with classical estimators plus uncertainty, term discovery in reaction-diffusion
(BINNs), nonlinear pharmacokinetics, excitable media and activation maps, and pulsatile aortic flow with Windkessel
outlets in OpenFOAM. Every number below was measured by the example that prints it; the paper reproductions follow
the protocol of `pinneapple_analysis.reproduction` (targets quoted in `examples/reproductions/<paper>/TARGETS.md`,
committed before any run).

## SBML models as ODE systems

```python
import numpy as np
from pinneapple_physics.biology import biomodel, read_sbml

rep = biomodel(12)                                     # Elowitz & Leibler repressilator, shipped (CC0)
traj = rep.simulate(np.linspace(0, 200, 401))          # SciPy BDF with the analytic Jacobian
print(rep.states, traj["PX"].max())
again = read_sbml(rep.to_sbml())                       # SBML L3V2 export (rate rules) and back
```

`read_sbml` needs no libsbml (with it installed, `validate=True` runs its consistency checks). It reads compartments,
species in amount or concentration units, reactions with local parameters, assignment / rate rules, initial
assignments, function definitions and MathML, and refuses what an ODE system cannot represent faithfully (events,
algebraic rules, delays, variable compartments, fast reactions) with `SBMLUnsupported`.

| Check (tests/test_biology_sbml.py) | Result |
|---|---|
| Six BioModels entries (5, 10, 12, 21, 28, 35) vs libRoadRunner 2.11 (CVODE, rtol 1e-10) | max relative L2 error 5e-6 (our rtol 1e-8), 3e-9 (rtol 1e-11) |
| Export to SBML L3V2, re-import, simulate | identical trajectories; libRoadRunner on the export agrees to the same 1e-6 |
| Hill terms `x**n` at `x = 0` | Jacobian finite (powers combined before lambdifying) |

An `ODESystem` also gives forward sensitivities (`sensitivities`), a `reaction_kinetics_network` problem for the PINN
compiler (`to_problem_spec`), and a `ForwardModel` for the identifiability and calibration tools (`forward_model`).
Inputs with jumps (doses, meals) are detected as breakpoints and the integrator restarts there.

## Identifiability before fitting

```python
from pinneapple_analysis.inverse_problems.identifiability import analyze_identifiability
from pinneapple_analysis.inverse_problems.ode_model import ODEForwardModel

model = ODEForwardModel(system, ["G"], ["E", "tp", "C2"])   # observables, parameters (or "<state>_0")
report = analyze_identifiability(model, observations)       # Fisher rank, Cramer-Rao bounds, profiles, design
print(report.summary)
```

`examples/biology/identifiability_ultradian.py` reproduces the identifiability part of the SBINN chapter (Daneker et
al., arXiv:2202.01723) on the ultradian glucose-insulin model, with the Fisher matrix built from exact forward
sensitivities:

| Target (TARGETS.md) | Paper | Measured | Status |
|---|---|---|---|
| T3: Fisher eigenvalues, 1 % noise on G | "only one eigenvalue close to 0", eigenvector "C2 ... dominant" | smallest / largest eigenvalue 1e-15; null vector C2 = 0.9996, all others < 0.022 | reproduced |
| T2: scaling invariance U0, Um, C3 (Eq 7) | invariance | null direction, lambda_min / lambda_max = 1e-16, equal weights | reproduced |
| T2: Rm with Ip(0), Ii(0) (Eq 7) | invariance (of the rational form, Eq 6) | in the original system the null direction is (Rm, Ip0, Ii0, C4, C5), ratio 1e-16 | reproduced, in its original-system form |
| T2: alpha with C5 | "does not hold in the original system" | ratio 0.10 (not null) | reproduced |

The structural test is numerical and local (Fisher rank at one point), not the symbolic test of the chapter.

## Nonlinear PK/PD and PBPK

`pinneapple_physics.biology.models`: `pk_one_compartment_mm` (oral dosing, Michaelis-Menten elimination, effect
compartment, sigmoid Emax), `minimal_pbpk` (flow-limited, well-stirred liver, renal clearance; placeholder physiology
to replace for real use), `ultradian_glucose_insulin`. `examples/biology/pkpd_inference.py`:

- design before data: a low-dose study (C far below Km) identifies V and ka only; Vmax and Km are practically
  non-identifiable and the analysis names the combination the data fix (Km / Vmax). Adding a high dose at 24 h makes
  all four identifiable;
- inference on that design over 20 noise draws (adaptive Metropolis, 3000 steps): the 95 % credible intervals
  contain the true value in 18, 19, 20 and 19 of 20 draws (Vmax, Km, V, ka), and the posterior medians are within
  4 % to 6 % of the truth (median over draws); the PD fit with the PK fixed recovers E0, Emax, ke0 within 6 % to 13 %
  and EC50 within 29 %;
- PBPK mass balance closes to 1e-15 and the area under the arterial plasma curve matches dose / (CL_ws + CLr) to
  0.2 % (the rest is the truncated tail).

## Excitable media and activation maps

`pinneapple_physics.biology.excitable`: FitzHugh-Nagumo 1-D (method of lines; second-order convergence, 3e-7 between
401 and 801 points; now the reference of the arena problem `fitzhugh_nagumo_1d`, whose previous approximate formula
was 64 % off), Aliev-Panfilov 2-D with heterogeneous diffusivity and stimuli, activation times and conduction
velocity (planar-wave speed 1.383 at dx = 0.25 and 1.393 at dx = 0.125, dimensionless).

`examples/biology/cardiac_activation_mapping.py`: a scar in a 50 x 50 sheet, 64 random measurement sites, the eikonal
PINN of `pinneapple_analysis.inverse_problems.activation_mapping` against linear interpolation of the same sites:

| Map from 64 sites | RMSE activation time | median conduction-velocity error |
|---|---|---|
| linear interpolation | 0.91 | 3.3 % |
| eikonal PINN (20,000 steps) | 0.46 | 2.5 % (from T), 2.4 % (V network) |

Sahli Costabal et al. 2020 (Front Phys) could not be read from this session (frontiersin.org refused), so this is a
reimplementation of the approach, not a reproduction of its numbers.

## Pulsatile aorta with Windkessel outlets

`pinneapple_simulation.hemodynamics`: `Windkessel` (exact update for piecewise-constant flow; matches its analytic
impedance to 5e-4 in amplitude), `split_rcr`, `half_sine_inflow`; `aorta.setup_case` meshes the library's
`aortic_arch` (inlet and five outlets found as the flat caps of the lumen) with snappyHexMesh and writes a laminar
pimpleFoam case; `aorta.CoupledRun` couples the outlets to their Windkessels window by window (OpenFOAM v1912 has no
Windkessel boundary condition). The coupling is explicit, so the window must be short against the inertia of the
vessel: `explicit_coupling_gain(wk, vessel_inertance(length, diameter), dt)` is `g = Rp dt / L`, and an error in the
outlet flow comes back multiplied by `1 - g` one window later (oscillating for g > 1, divergent for g > 2). On the
library's arch a 10 ms window (g of 1.4 to 4.2 with 5 cm branches) made the outlet pressures flip sign and grow to
about +-2300 mmHg in 0.12 s; the default is now 1 ms (g of 0.14 to 0.42). See `examples/biology/aorta_windkessel.py`
for the run and its checks. Measured (`examples/biology/results/aorta_windkessel.json`, reduced-scale: 2 of 4
cycles, 57k cells, no wall layers): cycle-mean outlet flow fractions within 0.43 % of the 0-D resistance split, mass
imbalance 1.7e-8 of the mean inflow, inlet pressure 141.5 / 72.2 mmHg in the second cycle. The Windkessels are not
periodic after two cycles (cycle-mean inlet pressure 106.4 then 100.3 mmHg, against 93 mmHg from the 0-D model), so
the pressures are a transient, not the periodic state.

## Paper reproductions

- `examples/reproductions/binns_lagergren2020`: BINNs (Lagergren et al. 2020), scratch-assay data of Jin et al. 2016.
- `examples/reproductions/sbinn_daneker2022`: SBINN on the ultradian model, targets from the 2022 chapter (the 2020
  PLoS paper was not accessible).

Their `README.md` files hold the measured tables and statuses.
