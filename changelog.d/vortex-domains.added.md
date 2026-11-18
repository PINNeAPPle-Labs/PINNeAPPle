- **VVD, a mesh-free 2-D Navier-Stokes solver**: `pinneapple_simulation.vortex_domains` carries the vorticity of an
  incompressible flow on Lagrangian vortex domains (regularised Biot-Savart, viscosity as a drift down the vorticity
  gradient, Lighthill's flux at the wall) over rigid polygonal bodies that can move, hinge and deploy, for any
  Reynolds number. `VVD2D` + `VVDConfig` run it; `cylinder_case` / `naca0012_spoiler_case` build the two reference
  problems (the second is a NACA 0012 with a spoiler that rises to 90 degrees at Re = 4e4); `run_study` reduces a run
  to Cl / Cd / Cm, a Strouhal number and the physics checks the solver can be held to (impermeability, no slip,
  domain displacement, bounded circulation, resolution, and the drag and shedding against their references);
  `grid_field` / `surface_pressure` / `streamlines` give the fields. `python -m pinneapple_simulation.vortex_domains`
  prints what the checkout reproduces, and the guide is `docs/guides/vortex_domains.md`.
