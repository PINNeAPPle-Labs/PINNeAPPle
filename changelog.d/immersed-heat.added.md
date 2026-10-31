- `pinneapple_simulation.immersed`: mesh-free analysis on raw geometry (issue #966, part 1). Immersed / finite cell
  solver on a uniform Q1 or Q2 background grid; geometry from any signed distance (and CSG), a voxel mask (distance
  transforms) or an STL (generalised winding number with a dipole tree; pseudonormal signs near watertight
  surfaces); adaptive octree cut-cell quadrature with simplex clipping at the leaves, compressed per cell by moment
  fitting; small-cut stabilisation by ghost penalty or the finite cell method's fictitious material, with the
  condition number reported; Nitsche Dirichlet conditions and surface quadrature for heat flux and convection;
  steady heat with an energy balance. `immersed.verification`: manufactured solutions on curved domains with
  measured L2 / H1 rates, sphere and cylinder with convection in closed form, a small-cut conditioning study.
