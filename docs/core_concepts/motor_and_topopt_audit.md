# Audit: electric-motor magnetics and topology optimization

Status: merged with the electromagnetics package (user guides: [electromagnetics](../guides/electromagnetics.md),
[topology optimisation](../guides/topology_optimization.md)). Numbers below come from runs of the code (tests in
`tests/test_motor_magnetics.py`, `tests/test_topology_optimization.py`, examples in
`examples/motor_magnetics/` and `examples/topology_optimization/`).

## What existed before this branch

| Capability | State before |
|---|---|
| 2D magnetostatics in A_z | Only a PINN example of an infinite wire (`examples/electrodynamics/04_magnetostatics_wire.py`); unverified for machines |
| Axisymmetric eddy current (FDM) | Exists and verified (`numerical_solvers/eddy_current_fdm.py`, manufactured-solution test) |
| Nonlinear B-H | Only a scalar-potential FDM for flux-leakage inspection (`pinneapple_analysis/inspection/magnetic_flux_leakage.py`) |
| Magnets, coils, rotation, torque (MST / co-energy), flux linkage, back-EMF, cogging, iron/magnet losses | Missing |
| Generic P1 FEM | `pinneapple_core/fem.py` (dense, torch, Poisson); no magnetics |
| SIMP, density filter, Heaviside projection, OC, MMA, compliance, thermal, Borrvall-Petersson, magnetic topology optimization | Missing (only shape/parametric optimizers in `pinneapple_design/design_optimizer`) |

## Added (status: exists and verified unless noted)

`pinneapple_simulation/electromagnetics/`: polar ring mesher with moving band, gmsh importer,
P1 A_z solver with Newton (nonlinear B-H), magnets, coils, Dirichlet and (anti-)periodic BCs,
Arkkio Maxwell-stress torque and co-energy torque, flux linkage, back-EMF, cogging,
Steinmetz/Bertotti iron loss, magnet eddy loss (resistance limited) and a time-harmonic eddy
solver with net-current constraints, SPM/reluctance machine model, adjoint torque sensitivities.

`pinneapple_design/topology/`: density filter, Heaviside projection, OC, MMA (written from
Svanberg 1987, dual solved with L-BFGS-B), compliance (MBB, cantilever), thermal compliance,
Stokes/Brinkman double pipe, reluctance-rotor torque problem.

Verification (measured):
- Wire: A error 1.3e-3 (second order); magnetized cylinder: inner B within 0.14 %; anti-periodic half model identical to full.
- Nonlinear iron ring around a wire: B vs B-H curve within 0.3 % (three current levels).
- Slotless SPM vs exact subdomain solution: torque within 0.05 % (Maxwell stress) and 0.02 % (co-energy) at n=360, rotor off-grid.
- Skin effect in round wire: R and X within 0.1 % of the Bessel solution; low-frequency eddy loss within 0.09 %.
- Slotted SPM (4 poles, 24 slots): on-load torque 16.80 vs 16.82 N m (Maxwell vs co-energy); cogging within ~0.01 N m of a ~0.37 N m peak; cogging exactly periodic.
- Adjoint torque sensitivities vs finite differences: rel. error < 1e-5 (linear and nonlinear iron).
- MMA vs SLSQP on Svanberg's 5-variable problem: same optimum (1.33995636).
- MBB 60x20: OC 233.7, MMA 233.5; double pipe: two pipes for 1x1, merged channel for 1.5x1 (qualitative Borrvall-Petersson result).

## Open points

- Cogging: the original disagreement between Maxwell stress and co-energy came from the
  moving-band elements (shape changes with angle). Fixed by excluding the band from the Arkkio
  annulus and differencing co-energy at whole grid steps; remaining difference ~3 % of the peak
  at n_theta=360 and mesh-pattern dependence of a few % when the slot pitch has an odd number of steps.
- Optimized reluctance rotor: Maxwell 46.2 vs co-energy 47.1 N m (1.9 %) on the thin-featured
  design; nonlinear re-evaluation drops mean torque from 38.1 (linear) to 13.4 N m (saturation),
  so the optimization should use nonlinear iron.
- Done since: user guides, CHANGELOG, CI step; the Newton solver's stall at a 1e-7 residual (Armijo test below
  round-off) is fixed.
- Not done: TEAM benchmark references, transient eddy currents, geometry-parameter surrogate.
