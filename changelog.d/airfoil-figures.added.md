- `pinneapple_autolab.domains.airfoil_figures` draws the figures of the airfoil trust study, one per thing a stage
  measured: the design plan, the grid-convergence study, the POD spectrum, the held-out parity, the physics gates, the
  surface-pressure trace, the input sensitivity, the evidence map, the before and after of an evidence extension and
  the calibration curve. Each function takes the numbers the stage already wrote and returns the PNG path; the stage
  that owns the number calls it, and the engineering report calls the same function again, so a figure beside a
  verdict is the figure the run produced instead of a second implementation of it.

  It also draws the study's one non-matplotlib picture, `surface_pressure_3d`: the section a campaign solved, extruded
  over a picture span and coloured by the wall pressure the solver returned, through
  `pinneapple_tools.visualization.studio` (Blender Cycles). It returns `None` where there is no Blender, so a study
  that uses it keeps running. `wall_pressure` reads the wall out of a lattice field (the wall samples are every
  `1 + len(OFFSETS)`-th value, not the first `2 * len(STATIONS)`).
