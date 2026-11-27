- **The axisymmetric element, and the cylinder that proves it** (#142): `deck.geometry_2d` writes a two-coordinate
  deck (`*NODE` with `(r, z)`, `*ELEMENT, TYPE=CAX8`) and `mesh.QuadGrid` builds a conforming quadratic mesh in the
  `(r, z)` plane with the mid-side nodes shared between neighbours, faces selected by the plane they lie in and
  `ring_area` for the area a pressure actually acts on once revolved. The case is Lamé's thick cylinder under internal
  pressure, where all three stresses and the bore displacement are closed-form: measured in CalculiX 2.23 the three
  stresses at the bore and at mid-wall close inside 0,5 % (`sigma_theta` 166,60 MPa against 166,67; `sigma_z` 20,00
  against 20,00) and `u_r` of the bore is 0,009078 mm against 0,009079 mm. The solver's force convention had to be
  measured rather than assumed: over four runs that vary radius, height and pressure, the axial reaction the `.dat`
  prints is exactly `1/180` of the ring total, so the case checks that ratio — a solver that changes the convention is
  caught — and the reason for the 180 is recorded as open.
