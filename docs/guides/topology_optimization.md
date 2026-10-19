# Topology optimisation

`pinneapple_design.topology` optimises the material layout on a fixed grid of elements (density method, SIMP):
each element carries a density `rho` in `[0, 1]`, a filter and a Heaviside projection turn the design variables
into densities, and an optimiser (optimality criteria or MMA) updates them from adjoint sensitivities.

| Piece | API |
|---|---|
| Density filter (cone weights, any radius), symmetry maps | `DensityFilter.grid(nelx, nely, r_min)`, `SymmetryMap` |
| Heaviside projection with continuation | `heaviside`, `heaviside_derivative`, `DesignChain(n, filter, beta=...)` |
| Optimisers | `oc_update` (optimality criteria), `MMA` (method of moving asymptotes, Svanberg 1987; dual solved with L-BFGS-B) |
| Driver and gradient check | `optimize(problem, volume_fraction, filter, "oc" / "mma", ...)`, `finite_difference_check` |
| Structural compliance on Q4 elements (MBB beam, cantilever) | `Compliance2D.mbb`, `Compliance2D.cantilever` |
| Thermal compliance | `ThermalCompliance2D` |
| Stokes flow with Brinkman penalisation (Borrvall-Petersson) | `StokesDissipation2D.double_pipe` |
| Rotor iron of an electric machine for torque | `pinneapple_design.topology.magnetic`, on `pinneapple_simulation.electromagnetics` ([guide](electromagnetics.md)) |

## What is verified

`tests/test_topology_optimization.py` asserts:

* sensitivities through filter and projection against finite differences for compliance, thermal compliance and
  Stokes dissipation;
* MMA against SciPy's SLSQP on the five-variable cantilever problem of Svanberg (1987): same optimum to 1e-8
  relative (1.33995636);
* OC and MMA reach the same MBB compliance, and the MBB result is a truss attached to the load and supports;
* the double-pipe problem gives two pipes in a 1 x 1 domain and one merged channel in a 1.5 x 1 domain, the
  qualitative result of Borrvall & Petersson (2003);
* the reluctance-rotor optimisation increases torque, and the torque-ripple objective's gradient.

Runs of the examples (2026-10-17, CPU):

| Example | Result |
|---|---|
| `01_mbb_beam.py` (60 x 20, volume 0.5) | compliance 233.71 (OC, 146 iterations), 233.49 (MMA, 58 iterations); with projection 190.05, gray fraction 2.3 %; gradient check 1.0e-6 |
| `02_heat_and_stokes_flow.py` | thermal compliance 4.18e8 (uniform) -> 2.49e7; double pipe dissipation 136.4 -> 25.7 (1 x 1) and 179.3 -> 26.8 (1.5 x 1) |
| `03_reluctance_rotor.py` | 1410 design variables; mean torque 38.10 N m with linear iron, 13.44 N m when the same rotor is re-evaluated with nonlinear iron; Maxwell stress 46.21 vs co-energy 47.07 N m at 5 degrees |

## Limits

* The rotor optimisation uses linear iron; the nonlinear re-evaluation shows that a thin-featured rotor saturates,
  so the design should be optimised with the nonlinear B-H curve (the adjoint supports it; the example does not).
* On thin-featured designs Maxwell-stress and co-energy torque differ by about 2 % (46.21 vs 47.07 N m above),
  more than on smooth geometries; refine the band and air gap before trusting single-angle torques.
* Grids are structured and 2-D.

## References

* K. Svanberg, The method of moving asymptotes, *Int. J. Numer. Meth. Eng.* 24 (1987) 359-373.
* M. P. Bendsoe and O. Sigmund, *Topology Optimization: Theory, Methods and Applications*, Springer, 2003.
* T. Borrvall and J. Petersson, Topology optimization of fluids in Stokes flow, *Int. J. Numer. Meth. Fluids* 41
  (2003) 77-107.
