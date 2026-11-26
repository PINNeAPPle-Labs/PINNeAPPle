- `pinneapple_autolab.domains.airfoil_figures` draws the figures of the airfoil trust study, one per thing a stage
  measured: the design plan, the grid-convergence study, the POD spectrum, the held-out parity, the physics gates, the
  surface-pressure trace, the input sensitivity, the evidence map, the before and after of an evidence extension and
  the calibration curve. Each function takes the numbers the stage already wrote and returns the PNG path; the stage
  that owns the number calls it, and the engineering report calls the same function again, so a figure beside a
  verdict is the figure the run produced instead of a second implementation of it.
