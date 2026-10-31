"""3-D electric machines on the edge-element solver of :mod:`pinneapple_simulation.electromagnetics.fem3d`.

* :mod:`.extrude` -- conforming extrusion of a 2-D triangle mesh into tetrahedra.
* :mod:`.spm` -- :class:`SPM3D`: the surface-PM machine extruded along the stack, magnet skew (step or
  continuous), end regions, Maxwell-stress torque, multi-slice 2-D on the same mesh.
* :mod:`.skew`, :mod:`.endeffects` -- skew study; end effects against L/D (search-coil flux linkage).
* :mod:`.magnet_loss` -- eddy loss of a segmented magnet, 3-D against 2-D and the closed form.
* :mod:`.afpm` -- :class:`AFPM3D`: axial-flux PM machine (single or double rotor) in 3-D, quasi-3D on the same grid.
"""
from .afpm import AFPM3D, AFPMMachine
from .extrude import extrude
from .spm import SPM3D, continuous_skew, no_skew, step_skew

__all__ = ["extrude", "SPM3D", "continuous_skew", "no_skew", "step_skew", "AFPM3D", "AFPMMachine"]
