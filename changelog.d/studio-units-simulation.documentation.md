- Output units for the `pinneapple_simulation` tools the gallery's examples use, written in the code (numpy
  `Returns` / `Attributes`, `name : type [unit]`) and read by the Studio. Each unit is taken from the code or
  its docs, and dimensionless outputs are marked `-`. Outputs whose unit follows the caller's input (grids, mesh
  points, positions) stay empty. Of these tools' quantitative outputs, 16 of 29 now carry a unit (0 before). The
  outputs reader (`pinneapple_studio.outputs`) also splits numpy's grouped names (`x, y : float [m]`) into one
  output per name, sharing type, unit and description.
