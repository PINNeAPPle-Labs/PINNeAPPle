"""Suspension design as one problem: a double-wishbone corner, its hardpoints, its arm structure and its bushes.

The three models of vehicle engineering meet at a suspension corner and cannot be designed one after the other: the
**CAD** places the hardpoints and the arm sections, the **multibody** moves the linkage through wheel travel and reads
the wheel plane (camber, toe, caster, scrub, motion ratio, roll centre), and the **structural** model sizes the arm for
the loads that linkage delivers. On top of them sits the *elastokinematic* question — how much the bushes let the wheel
steer and camber under load — which is what separates a car that tracks straight from one that wanders.

Modules:

* :mod:`.hardpoints` — :class:`SuspensionSpec` (hardpoints, bushings, spring, tube, tyre) and a reference corner;
* :mod:`.kinematics` — the closed-loop multibody solve through wheel travel, and the curves it produces;
* :mod:`.loads` — the load cases and the force each arm carries;
* :mod:`.structure` — the arm as a steel tube: bending, Euler buckling and a 3-D finite-element check;
* :mod:`.calculix` — the same leg in an external solver (gmsh builds the B-rep, ``ccx`` solves it): the third model;
* :mod:`.compliance` — the bushes' elastic modes and the steer/camber the wheel gains under load;
* :mod:`.design` — the loop that searches the lightest corner meeting all of it;
* :mod:`.geometry` — the corner as CAD, from the same numbers;
* :mod:`.study` — the Studio's two tools, with their files, checks and stated limits;
* :mod:`.verification` — the cases whose answer is known independently of the code, and the error each one measures.

    from pinneapple_design.suspension import front_corner, kinematic_sweep, wheel_compliance, suspension_study
    spec = front_corner()
    kinematic_sweep(spec).summary()["bump_steer_deg_per_50mm"]
    wheel_compliance(spec)["toe_deg_per_kn"]
    suspension_study(out_dir="runs/corner")["checks"]
"""
from __future__ import annotations

from .compliance import ComplianceMode, compliance_modes, mode_responses, wheel_compliance
from .design import (
    BASELINE_DESIGN,
    BOUNDS,
    CONSTRAINTS,
    Design,
    DesignResult,
    evaluate,
    suspension_design,
)
from .geometry import corner_cad
from .hardpoints import Bushing, SuspensionSpec, front_corner
from .kinematics import (
    Pose,
    Sweep,
    kinematic_sweep,
    pose,
    pose_from_angles,
    roll_centre_height_m,
    wishbone_instant_centre,
)
from .loads import LOAD_CASES, arm_loads, corner_load_cases, worst_case
from .structure import TubeSection, arm_check, arm_mass_kg, fem_lower_arm, fem_tube, tube_mesh
from .study import suspension_study, suspension_verify

__all__ = [
    "Bushing", "SuspensionSpec", "front_corner",
    "Pose", "Sweep", "pose", "pose_from_angles", "kinematic_sweep", "roll_centre_height_m",
    "wishbone_instant_centre",
    "LOAD_CASES", "corner_load_cases", "arm_loads", "worst_case",
    "TubeSection", "arm_mass_kg", "arm_check", "tube_mesh", "fem_tube", "fem_lower_arm",
    "ComplianceMode", "compliance_modes", "mode_responses", "wheel_compliance",
    "Design", "DesignResult", "evaluate", "suspension_design", "CONSTRAINTS", "BOUNDS", "BASELINE_DESIGN",
    "corner_cad", "suspension_study", "suspension_verify",
]
