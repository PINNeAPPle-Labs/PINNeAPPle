- Output units for `pinneapple_design`: the closed-form ground truths of `surrogate_inverse` (heat sink, brake rotor,
  reflex port, suspension arm, transdermal patch), the aero helpers (ISA density, Raymer weights in kg, drag
  coefficients, sweep in degrees, Sutherland viscosity), the panel solution, the surrogate fit and sensitivity records,
  and the board-cooling mission and fin-cell results. Written in numpy `Returns` / `Attributes`; each unit is taken
  from the code, the names or the docs, `-` for dimensionless, and left empty when it follows the caller's input.
