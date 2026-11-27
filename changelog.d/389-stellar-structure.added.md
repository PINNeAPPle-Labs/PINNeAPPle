- `pinneapple_physics.astro.stellar`: stellar structure from the four equations in mass coordinate, from Lane-Emden to
  a full model (issue #389). The four structure equations as residuals a physics-informed network can consume
  (`structure_residuals`, scaled in solar units, with the local microphysics returned alongside); `lane_emden(n)`
  against the closed form and the tabulated surfaces (xi_1 = 3.141593 / 3.653754 / 6.896849 for n = 1 / 1.5 / 3); a
  declared analytic equation of state (ideal gas plus radiation, mu = 0.6156 for X = 0.7035) and opacity (electron
  scattering, Kramers, H^-) with their valid ranges; the analytic pp-chain and CNO-cycle fits (3.41e-3 W/kg at the
  solar centre, crossover at 1.8e7 K); the Schwarzschild criterion; and `stellar_model(mass, metallicity)`, which
  solves the photospheric closure and returns the whole model with its trust checks. One solar model now takes 3.3 s
  instead of 3108 s (bounded trials, and a bisection for the mass instead of the Newton that used to walk into the
  cold corner). `solar_calibration()` reports, as data, that the model reproduces the central temperature to 24 % and
  the luminosity to a factor 5 but **not the radius** (0.63 R_sun), and the docs page
  (`docs/core_concepts/stellar_structure.md`) measures why: a fully ionised EOS puts `grad_ad` at 0.4 where a
  solar-composition envelope needs 0.248. The comparison against MESA is declared and not run -- MESA is not installed
  here and there is no network -- with the exact plan, command and inlist in `stellar.MESA_PLAN` and the profile
  loader written and tested against a MESA-shaped table.
