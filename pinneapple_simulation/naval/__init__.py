"""Naval engineering models built on :mod:`pinneapple_systems.control`.

* :mod:`.submarine`: depth-keeping of a submerged submarine. A compressible hull in compressible water makes the
  depth equilibrium statically unstable. The module covers the trim tank and pump limits, and the two command
  architectures (target tank mass vs pump flow) with their closed-loop eigenvalues. It also covers pole-placement
  tuning and robustness.
* :mod:`.hydrostatics`: initial transverse stability of a surface ship: GM, free-surface correction, righting arm
  and roll period.
"""
from .hydrostatics import (
    box_barge,
    free_surface_correction,
    initial_stability,
    metacentric_height,
    righting_arm,
    roll_period,
)
from .submarine import (
    ILLUSTRATIVE,
    ILLUSTRATIVE_GAINS,
    SubmarineDepth,
    SubmarineParams,
    architectures,
    depth_study,
)

__all__ = ["SubmarineParams", "ILLUSTRATIVE", "ILLUSTRATIVE_GAINS", "SubmarineDepth", "architectures", "depth_study",
           "box_barge", "free_surface_correction", "metacentric_height", "righting_arm", "roll_period",
           "initial_stability"]
