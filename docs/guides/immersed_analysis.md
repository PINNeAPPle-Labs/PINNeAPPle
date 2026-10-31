# Immersed analysis on raw geometry

`pinneapple_simulation.immersed` solves heat conduction and linear elasticity directly on a part given as a signed
distance (SDF), a voxel mask or an STL surface, without generating a conforming mesh. The part sits in a uniform
background grid of Q1 or Q2 elements; the cells it cuts are integrated by an adaptive octree; boundary conditions
on the curved surface are imposed weakly (Nitsche for temperatures and displacements, surface quadrature for fluxes,
convection, tractions and forces). This is the finite cell / CutFEM family (Parvizian, Düster & Rank 2007; Burman
et al. 2015; Schillinger & Ruess 2015). The measured verification and the comparison against conforming FEM are in
[the research report](../research/immersed_analysis.md).

## When to use it, and when not

Measured, not assumed (report, "Verdict"):

| Use immersed when | Use conforming FEM (gmsh + `tet_fem`, or CalculiX) when |
|---|---|
| the input is voxels, an implicit SDF, or an STL that does not mesh (holes, flipped facets: the winding number still gives a robust inside test) | the part meshes cleanly: P2 was 1.2–1.8× faster on a smooth sphere, 20× on a bracket and over 50× on a heat sink at equal accuracy |
| the geometry changes topology across a sweep (no remeshing step to fail) | the part has thin walls in bending (Q1 locks; Q2 needs 4× the unknowns of P2) |
| the fields are smooth and unknowns matter more than wall time (Q2 used half the unknowns of P2 on a smooth sphere) | sharp edges carry a change of boundary condition, or convection on many sharp edges (the edges are chamfered at the octree-leaf scale: 1.3 % of the convective area on the heat sink) |

## Steady heat

```python
import numpy as np
from pinneapple_simulation import immersed as im

part = im.cylinder((0, 0, 0), 0.02, 0.1) - im.cylinder((0, 0, 0), 0.008, 0.12)    # CSG of exact SDFs
disc = im.Discretization(part, h=0.004, order=2)                                   # grid + cut-cell quadrature
bore = lambda X, n: np.hypot(X[:, 0], X[:, 1]) < 0.01                              # select by point (and normal)
sol = im.solve_heat(disc, conductivity=200.0, source=0.0,
                    bcs=[im.Dirichlet(80.0, where=bore), im.Convection(h=25.0, T_inf=20.0)])
print(sol.max(), sol.balance["imbalance"], sol.info["dofs"])                      # energy balance closes to round-off
```

- Geometry: `im.ball`, `im.box`, `im.cylinder`, `im.torus`, `im.annulus` and `|`, `&`, `-`; any vectorised SDF with
  `im.SDFGeometry(fn, (lo, hi))`; `im.VoxelGeometry(mask, origin, spacing)`; `im.STLGeometry.from_file(path)`.
- Conditions: `Dirichlet(value)`, `HeatFlux(q)` (into the body), `HeatInput(power)` (a total power spread over the
  selected area), `Convection(h, T_inf)`. `where(X)` or `where(X, n)` selects the surface points; the first matching
  condition wins; unselected surface is adiabatic.
- Options: `order` 1 or 2; `depth` (octree levels, default 2 for Q1, 3 for Q2, 2 for STL); `stabilization`
  `"ghost"` (default, bounded condition number for any cut) or `"fcm"`; `condition=True` reports the condition
  number.

## Linear elasticity

```python
from pinneapple_simulation import immersed as im

disc = im.Discretization(im.box((0, 0, 0), (0.1, 0.02, 0.02)), h=0.004, order=2)
clamp = lambda X, n: (X[:, 0] < 1e-3) & (n[:, 0] < -0.5)
tip = lambda X, n: (X[:, 0] > 0.099) & (n[:, 0] > 0.5)
sol = im.solve_elasticity(disc, E=210e9, nu=0.3,
                          bcs=[im.Displacement(0.0, where=clamp), im.Force((0, 0, -100.0), where=tip)])
print(sol.max_displacement(), sol.balance["imbalance"])                           # equilibrium check
```

`Displacement(value, components=(2,))` restricts a support to some directions; `Traction`, `Pressure` and `Force`
(an exact total load) are the loads. Stresses and von Mises at any points: `sol.stress(X)`, `sol.von_mises(X)`.

## Transient heat

```python
from pinneapple_simulation.immersed import verification as V
from pinneapple_simulation.immersed.transient import solve_transient_heat

c = V.sphere_cooling()                                                             # series solution available
disc = im.Discretization(c["geometry"], 0.0625, order=1)
s = solve_transient_heat(disc, c["k"], c["rho_c"], 0.0, c["bcs"], T0=c["T_i"], dt=1e-3, n_steps=50, theta=0.5)
```

`theta` 1 is backward Euler, 1/2 Crank–Nicolson; `source(X, t)` and `bcs(t)` may depend on time.

## Incompressible flow (Stokes and Navier-Stokes)

`immersed.flow` solves steady Stokes and Navier-Stokes around a body given as an SDF, a 2-D polygon or an STL
surface, on a Cartesian grid that covers the flow box exactly (the box faces take prescribed velocity, wall, slip or
do-nothing outflow) and that the body cuts. Q2 velocity / Q1 pressure, no-slip on the body by Nitsche, ghost penalty
on the velocity and pressure across the faces of cut cells, Newton from the Stokes solution. The force on the body
comes from the variational (reaction) form, with the Nitsche surface flux next to it.

```python
from pinneapple_simulation.immersed import flow
from pinneapple_simulation.cfd_direct.cases import SCHAFER_TUREK as c, cylinder_polygon

body = flow.PolygonGeometry(cylinder_polygon(512))                 # or im.STLGeometry.from_file("body.stl") in 3-D
fd = flow.FlowDiscretization(body, c.box, h=c.D / 8)
bcs = {"x-": flow.Velocity(lambda X: np.column_stack([c.inflow(X[:, 1]), 0 * X[:, 1]])),
       "y-": "wall", "y+": "wall", "x+": "outflow"}
sol = flow.solve_navier_stokes(fd, c.nu, bcs, u_ref=c.U_mean)
Cd, Cl = c.coefficients(sol.force()["variational"])               # 5.64, 0.003 (benchmark 5.580, 0.0106)
```

Known answers (`tests/test_immersed_flow.py`): Poiseuille flow between walls that cut the grid to 1e-13; a
manufactured solution around a disk at the optimal rates (L2 > 2.8, H1 > 1.8, Stokes and Navier-Stokes); a condition
number that stays at 2.2e6-2.3e6 when the wall cuts a cell at fluid fractions 0.5 to 1e-4. How it compares with
body-fitted FEM and with lattice Boltzmann on the same polygon or STL, in time and accuracy:
[docs/research/cfd_direct_from_stl.md](../research/cfd_direct_from_stl.md).

## Checks that come with every result

- heat: `sol.balance` (heat generated against heat leaving through each condition, Nitsche's consistent flux);
- elasticity: `sol.balance` (applied loads against reactions);
- verification against exact solutions: `immersed.verification` (manufactured solutions on curved domains,
  sphere / cylinder / Lamé sphere / sphere cooling in closed form, `conditioning()` for small cuts);
- against conforming FEM on the same part: `immersed.compare` (SDF parts) and `immersed.parts.study` (STL parts).

## In a flow

The Flow node `immersed.heat` runs steady heat with uniform generation and convection on a built-in solid or an STL
file, and returns the peak temperature, the energy balance, the cost by phase and a mid-plane figure.
