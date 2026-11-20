`airfrans_benchmark.curve_projection`: the slope field a chord-curve readout would produce, computed from the one the
network already produces — the curve is estimated on the wall nodes (mean per bin, per surface, in the chord frame the
freestream defines) and then evaluated at every node, so it reaches the boundary-layer band where the wall layer's weight
is non-zero rather than the surface where that weight is zero. `WallAwareNet.forward(..., slope_override=...)` is what
lets a caller impose it. The measurement it exists for: binning the *true* slope into 33 cells per surface reproduces it
to 12.1 %, so the shape is capturable by a short curve — but projecting the trained readout onto it takes the holdout
wall shear from 22.05 % to 78.7 % while moving the drag R² from +0.449 to +0.875. The force coefficient is a chord-scale
integral and the wall shear is a local derivative; smoothing serves one and destroys the other.
