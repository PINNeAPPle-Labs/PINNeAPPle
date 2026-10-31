- `pinneapple_simulation.hemodynamics.aorta.CoupledRun`: **Behavior:** the default coupling window is 1 ms instead
  of 10 ms, which was unstable on the library's aortic arch (outlet pressures flipping sign and growing every window);
  `explicit_coupling_gain` and `vessel_inertance` give the stability number `Rp dt / L` of a window.
