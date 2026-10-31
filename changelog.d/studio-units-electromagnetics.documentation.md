- Output units for `pinneapple_simulation.electromagnetics` beyond the tools the examples use (#1020): the statics closed
  forms (F, F/m, H, H/m, T), waveguide and cavity cut-offs and resonances (Hz), wavelength, Brewster and critical
  angles, the FDTD Courant limit and numerical wavenumber, the 2-D magnetostatic solution (Wb/m, T, m/H, J/m), the
  TEAM 7 coil current density, magnet eddy losses, Mie efficiencies, feed gain (dBi) and the periodic-array worst case
  (dB). The package is SI (`constants`), so each unit holds for lengths in m; values in the units of the caller's
  inputs (microstrip width, polygon area) stay empty.
