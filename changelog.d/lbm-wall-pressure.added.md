- `WindTunnelLBM` (lattice Boltzmann) now returns wall and surface pressure: `wall_pressure()` (the kinematic pressure
  of every solid cell that faces fluid, with its centroid and outward normal), `surface_pressure(points)` (trilinear
  sampling anywhere in the box), `pressure()` and `run_averaged()` (a time-averaged field for a flow that sheds, with
  the averaged wall values); `sample_field` interpolates any 3-D field at lattice points without extrapolating. This
  is the extraction a surface-pressure dataset needs, and it is tested against the Stokes pressure difference across a
  sphere.
