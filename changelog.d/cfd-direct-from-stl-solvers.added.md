- CFD directly on polygon / STL geometry, and its body-fitted reference (campaign
  `docs/research/cfd_direct_from_stl.md`).
  - `pinneapple_simulation.immersed.flow`: steady Stokes and Navier-Stokes on a Cartesian grid cut by the body (SDF,
    2-D `PolygonGeometry`, STL): Q2/Q1, Nitsche no-slip, ghost penalty on velocity and pressure, Newton, variational
    force. Poiseuille between cut walls to 1e-13, optimal rates on a manufactured solution, condition number
    independent of the cut.
  - `pinneapple_simulation.numerical_solvers.taylor_hood`: P2/P1 Taylor-Hood Navier-Stokes with scikit-fem on gmsh
    meshes (`read_gmsh`, prisms split conformingly), reaction forces; `numerical_solvers.sparse_direct`: one direct
    solve (PARDISO when installed, else SuperLU) shared by both.
  - `generate_mesh`: a `polygon` geometry (`vertices`, meshed as one curve) and `enclosure` (mesh a box minus the
    body: primitives, polygons, STEP / IGES by boolean cut, closed STL surfaces as the inner shell).
  - `pinneapple_simulation.cfd_direct`: the campaign's cases (Schäfer-Turek 2D-1, sphere with Clift-Grace-Weber,
    Armaly's step), runners for the three methods and the pre-registered verdict rule.
  - `immersed.Discretization` takes a given `grid` and an `n_facet` rule.
