# A body of revolution: the axisymmetric element, and the cylinder that proves it

`deck.geometry_2d` writes an axisymmetric deck — `*NODE` with two coordinates `(r, z)` and one `*ELEMENT` block of
`CAX8` (8-node quadratic) or `CAX4` — and `mesh.QuadGrid` builds the mesh that feeds it. Issue
[#142](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/142) is why: the deck writer only produced `C3D8`
bricks, so an axisymmetric problem had no element.

## Why a brick block cannot stand in for it

Two ways of faking the axisymmetric model both fail, and both fail quietly enough to be worth writing down:

* **A polar block of bricks.** A ring meshed in `(r, theta)` reproduces the geometry, but the axis `r = 0` degenerates:
  every node there collapses to one point and the elements next to it have a zero-length edge, so the solver refuses
  the Jacobian.
* **A staircase footprint.** Cutting a circular loaded area out of a rectilinear block changes the area, and an
  axisymmetric closed form is usually *about* that area (the total load, the settlement).

The element solves it by being a ring: the mesh is two-dimensional, and CalculiX integrates the revolution itself.

## What the mesh gives you

```python
from pinneapple_simulation.external_solvers.calculix.contact.mesh import QuadGrid, axis
from pinneapple_simulation.external_solvers.calculix.contact import deck as D

grid = QuadGrid(axis(10.0, 20.0, 24, grading=1.6, cluster="low"), [0.0, 1.0])   # r from 10 to 20 mm, one cell in z
D.geometry_2d(deck, grid.nodes, zip(grid.elem_ids, grid.cells), "CIL")
grid.faces("r", 10.0)        # (element, face) on the bore — selected by the plane it lies in, never by face number
grid.nodes_on("r", 10.0)     # the node set the pressure and the reaction are read from
grid.ring_area("r", 10.0)    # 2 pi r h: what a pressure of p puts on that surface, revolved once
```

`QuadGrid` numbers the corners first, then the mid-side nodes along `r`, then along `z`, and each mid-side node is
shared between the two elements that own the edge — a conforming quadratic mesh, which is what `CAX8` needs.

## The case that proves it: Lamé's thick cylinder

`cilindro_lame` is a thick cylinder under internal pressure. It is the right validation precisely because **all three**
principal stresses and the bore displacement have exact solutions in plane strain:

| quantity | closed form |
|---|---|
| `sigma_r(r)` | `A + B/r^2` |
| `sigma_theta(r)` | `A - B/r^2` |
| `sigma_z(r)` | `2 nu A` |
| `u_r(r)` | `r [A (1 - nu - 2 nu^2) - (B/r^2)(1 + nu)] / E` |

with `A = p a^2/(b^2 - a^2)` and `B = -A b^2`. The two end faces are held flat, which is exactly `eps_z = 0`; an open
tube (`sigma_z = 0`) is a different closed form and comparing it here would compare two different problems.

Measured in CalculiX 2.23 (`tests/test_calculix_axisym.py`, `a = 10`, `b = 20`, `p = 100 MPa`, steel):

| check | measured | closed form |
|---|---|---|
| `sigma_r` at the bore | -99,92 MPa | -100,0 |
| `sigma_theta` at the bore | 166,60 MPa | 166,67 |
| `sigma_z` at the bore | 20,00 MPa | 20,00 |
| `sigma_r` at mid-wall | -24,08 MPa | -24,17 |
| `sigma_theta` at mid-wall | 90,83 MPa | 90,84 |
| `sigma_z` at mid-wall | 20,03 MPa | 20,00 |
| `u_r` of the bore | 0,009078 mm | 0,009079 |

The component mapping is the one to remember: in an axisymmetric deck `SXX` is **radial**, `SYY` is **axial** and
`SZZ` is **circumferential**. Reading `SZZ` as the axial stress is a deck that converges and is wrong.

## The force convention, measured and not assumed

An axisymmetric model reports **totals over the revolution**, so a reaction is not the 2-D number on the section. What
the `.dat` prints is not the ring total either: measured over four runs that vary the radius (`a = 10` and `20`), the
height (`1` and `2 mm`) and the pressure (`100` and `200 MPa`), the printed axial total is **exactly `1/180`** of
`sigma_z * pi (b^2 - a^2)` — the same factor every time, which is why the case checks the *ratio* rather than a force.

The reason for `180` is not established here: a sector of 2 degrees (`360/2`) and a per-radian total divided by
`2*pi` differ by a factor of about 28,6 and the measurements rule that out, but they cannot tell a 2-degree sector
from another convention of the same size. Until it is settled, the ratio is a **regression check**: a solver whose
convention changes is caught, and nothing is built on the absolute number.

## What this does not do

* A body that reaches its own axis (`r = 0`) is refused, deliberately: the elements there are degenerate and a solid
  of revolution about its axis is a different problem with a different strain measure.
* It is a writer and a mesh, not a case generator: a new axisymmetric case is a `QuadGrid` plus the boundary
  conditions of that problem, as `cilindro_lame` is.
* The Roark case 8 (a rigid cylindrical die on a half-space) still lives in the product, not here: it needs this
  element through a `PINNEAPPLE_COMMIT` bump, and the case stays in `app/prepomax` until then.
