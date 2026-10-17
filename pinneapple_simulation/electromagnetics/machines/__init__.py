"""Electric machines: parametric cross-sections, conforming meshes, nonlinear 2-D FEA and post-processing.

* :mod:`.geometry` -- V-shaped interior-magnet rotors, round-bottom slots, the 2004 Toyota Prius motor.
* :mod:`.mesher` -- conforming Delaunay sector meshes copied around the circle, moving band in the air gap.
* :mod:`.steel` -- catalogue steel (M400-50A B-H and loss table), magnet grades, stacking factor, loss fits.
* :mod:`.pmsm` -- :class:`IPMMachine`: torque (Maxwell stress and co-energy), cogging, ripple, flux linkage,
  back-EMF, dq flux linkages, frozen-permeability L_d / L_q, MTPA, iron and magnet losses.

The surface-PM and reluctance model on the polar mesher is :class:`pinneapple_simulation.electromagnetics.machine.SPMMachine`.
"""
from .geometry import MachineGeometry, Polygon, RoundBottomSlots, SectorGeometry, VRotor, prius_2004
from .mesher import MachineMesh, SectorMesh, mesh_sector
from .pmsm import IPMMachine, Winding
from .steel import (
           M400_50A,
           MAGNET_PRIUS,
           MagnetGrade,
           SteelGrade,
           TableIronLoss,
           fit_iron_loss,
           stacked,
)

__all__ = ["MachineGeometry", "Polygon", "RoundBottomSlots", "SectorGeometry", "VRotor", "prius_2004",
           "MachineMesh", "SectorMesh", "mesh_sector", "IPMMachine", "Winding", "M400_50A", "MAGNET_PRIUS",
           "MagnetGrade", "SteelGrade", "TableIronLoss", "fit_iron_loss", "stacked"]
