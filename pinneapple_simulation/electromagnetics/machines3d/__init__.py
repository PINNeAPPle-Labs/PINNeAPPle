"""3-D electric machines on the edge-element solver of :mod:`pinneapple_simulation.electromagnetics.fem3d`.

* :mod:`.extrude` -- conforming extrusion of a 2-D triangle mesh into tetrahedra.
* :mod:`.spm` -- :class:`SPM3D`: the surface-PM machine extruded along the stack, magnet skew (step or
  continuous), end regions, Maxwell-stress torque, multi-slice 2-D on the same mesh.
"""
from .extrude import extrude
from .spm import SPM3D, continuous_skew, no_skew, step_skew

__all__ = ["extrude", "SPM3D", "continuous_skew", "no_skew", "step_skew"]
