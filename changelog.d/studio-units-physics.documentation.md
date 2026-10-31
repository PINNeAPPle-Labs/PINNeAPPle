- Output units for the closed-form tools of `pinneapple_physics` (pipe flow, pumps, rocket nozzles, heat and heat
  exchangers, fins, membranes, resonators, drag, airfoils, fatigue, eddy-current skin depth) and for the
  `pinneapple_physics` tools the gallery's examples use (the Aliev-Panfilov and FitzHugh-Nagumo runs are
  dimensionless, the weather scores' ACC is `-`, the first crossing in hours). The units are written in numpy
  `Returns` / `Attributes` sections, each taken from the code, the argument names (`_pa`, `_m3`, `_hz`) or the
  docs. Outputs whose unit follows the caller's input (the capital recovery factor, a fitted polynomial, the Lame
  parameters) stay empty. Closed-form quantitative outputs with a unit: 53 of 55 (6 before).
