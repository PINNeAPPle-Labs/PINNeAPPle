"""Suspension corner: the double wishbone's rigid and elastokinematic behaviour, its structural model, and the
benchmark built on both.

    from pinneapple_simulation.suspension import BASELINE, corner, metrics, sweep
    metrics(corner(BASELINE))["camber_gain_deg_per_25mm"]

Modules: :mod:`.hardpoints` (the design vector and the corner it describes), :mod:`.kinematics` (the pose at a given
wheel travel, and camber, toe, roll centre, scrub radius, motion ratio), :mod:`.structure` (the wishbones as CAD,
meshed and run in CalculiX), :mod:`.elastokinematics` (bush and arm compliance at the wheel) and :mod:`.simulator`
(the corner as one simulator, which is what the study calls).
"""

from .hardpoints import BASELINE, BOUNDS, PARAMETERS, Corner, corner, invalid
from .kinematics import Pose, camber_deg, metrics, solve_pose, solve_pose_angles, sweep, toe_deg

__all__ = ["BASELINE", "BOUNDS", "PARAMETERS", "Corner", "corner", "invalid", "Pose", "metrics", "sweep",
           "solve_pose", "solve_pose_angles", "camber_deg", "toe_deg"]
