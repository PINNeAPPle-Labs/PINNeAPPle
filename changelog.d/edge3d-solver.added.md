- `pinneapple_simulation.electromagnetics.fem3d` (issue #929, first step): 3-D magnetostatics and time-harmonic eddy
  currents on first-order Nedelec edge elements (tetrahedral `TetMesh`, structured `box_mesh`, gmsh input), nonlinear
  B-H by Newton, tree gauge or ungauged conjugate gradients with discretely solenoidal sources, energy, co-energy and
  `source_work` (`dW'/di = psi`); GetDP 3.2 as the reference on the same meshes (`fem3d.getdp3d`). Verified on a
  permeable sphere in a uniform field (mean-field error 3.9e-2 to 2.2e-3, 6 500 to 494 000 unknowns), agreement with
  GetDP to 1e-13, a manufactured solution (order 1.00) and a conducting slab (loss within 1.5 %);
  `examples/electrodynamics/19_edge3d_verification.py`, Flow node `em.edge3d_verification`. TEAM benchmarks not yet
  compared.
