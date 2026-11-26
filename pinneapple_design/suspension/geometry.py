"""The corner as CAD: tubes, the upright, the wheel and the damper, in the assembly frame of the reference pose.

The geometry is parametric on the hardpoints and the tube section, so the design loop's winner *is* a solid: the same
numbers that went into the stress check draw the arms. The scene is what the Studio shows in 3-D next to the field
view, and the scenes' masses come from the meshes (:func:`pinneapple_design.geometry.vehicle_cad.mesh_mass_properties`),
not from a typed number.

    from pinneapple_design.suspension import front_corner, corner_cad
    cad = corner_cad(front_corner())
    cad.export("corner.glb")
"""
from __future__ import annotations

import numpy as np

from ..geometry.vehicle_cad import VehicleCAD, box_beam, cylinder, merge, tube_shell
from .hardpoints import SuspensionSpec
from .kinematics import Pose
from .structure import TubeSection, arm_mass_kg

__all__ = ["corner_cad"]

TYRE_WIDTH_M = 0.235
UPRIGHT_SECTION_M = (0.075, 0.045)
DAMPER_RADIUS_M = 0.023
HUB_RADIUS_M = 0.085


def corner_cad(spec: SuspensionSpec, p: Pose | None = None) -> VehicleCAD:
    """The corner as a CAD model: the two A-arms as tubes, the upright, the wheel and the damper, each its own body.

    The arms are drawn with the section the spec carries (the design's tube); the boss at each ball joint is a short
    cylinder at the tube's diameter, which is also where the masses are concentrated — the mass the optimiser reports
    is the mesh's own, computed from these solids.

    Returns
    -------
    cad : VehicleCAD
        Bodies ``lca``, ``uca``, ``upright``, ``wheel`` and ``damper``.
    """
    section = TubeSection(spec.tube_od_m, spec.tube_wall_m)
    cad = VehicleCAD(f"{spec.name}: design pose")
    mass = arm_mass_kg(spec, section)
    r_out, r_in = section.r_out, section.r_in
    # the two A-arms: one tube per leg plus the ball-joint boss
    for body, pivots, outer, share in (
            ("lca", (spec.lca_front, spec.lca_rear), spec.lca_outer, 0.62),
            ("uca", (spec.uca_front, spec.uca_rear), spec.uca_outer, 0.38)):
        for i, pivot in enumerate(pivots):
            cad.add(f"{body}_leg_{i}", body, tube_shell(pivot, outer, r_in, r_out, n=24), "steel", mass * share / 4.0)
        cad.add(f"{body}_boss", body, cylinder(outer - (outer - pivots[0]) * 0.12, outer, r_out * 1.6, n=24),
                "steel", mass * share / 2.0)
        cad.add(f"{body}_bushing_{0}", body, cylinder(pivots[0] - np.array([0.012, 0, 0]),
                                                      pivots[0] + np.array([0.012, 0, 0]), r_out * 1.35, n=20),
                "anodized_black", 0.12)
        cad.add(f"{body}_bushing_{1}", body, cylinder(pivots[1] - np.array([0.012, 0, 0]),
                                                      pivots[1] + np.array([0.012, 0, 0]), r_out * 1.35, n=20),
                "anodized_black", 0.12)
    # the upright: the kingpin strut, the hub and the steering arm
    cad.add("upright", "upright", box_beam(spec.lca_outer, spec.uca_outer, UPRIGHT_SECTION_M, chamfer=0.006),
            "aluminium", 2.6)
    hub0 = spec.wheel_centre - spec.wheel_axis * 0.06
    hub1 = spec.wheel_centre + spec.wheel_axis * 0.02
    cad.add("upright_hub", "upright", cylinder(hub0, hub1, HUB_RADIUS_M * 0.45, n=28), "aluminium", 1.1)
    cad.add("upright_arm", "upright", box_beam(spec.lca_outer, spec.tierod_outer, (0.05, 0.028), chamfer=0.004),
            "aluminium", 0.9)
    # the wheel: a tyre about the axle, with the rim inside it
    axis = spec.wheel_axis
    cad.add("tyre", "wheel", cylinder(spec.wheel_centre - axis * TYRE_WIDTH_M / 2,
                                      spec.wheel_centre + axis * TYRE_WIDTH_M / 2, spec.tyre_radius_m, n=56),
            "plastic", 9.5)
    cad.add("rim", "wheel", cylinder(spec.wheel_centre - axis * TYRE_WIDTH_M / 2.6,
                                     spec.wheel_centre + axis * TYRE_WIDTH_M / 2.6, spec.tyre_radius_m * 0.68, n=48),
            "wheel_metal", 7.5)
    # the damper: body and rod
    mid = 0.5 * (spec.spring_lower + spec.spring_upper)
    cad.add("damper_body", "damper", cylinder(spec.spring_lower, mid, DAMPER_RADIUS_M, n=24), "metal", 1.8)
    cad.add("damper_rod", "damper", cylinder(mid, spec.spring_upper, DAMPER_RADIUS_M * 0.45, n=20), "steel", 0.7)
    cad.notes["source"] = ("parametric model built here from the hardpoints and the tube section; the masses are the "
                           "meshes' own, from their volumes and the materials' densities")
    if p is not None:                                  # posed: the bodies carry the linkage's transform
        raise NotImplementedError("posing the CAD is done with VehicleCAD.posed by the caller")
    return cad


def merged_arm_mesh(spec: SuspensionSpec):
    """The lower arm's two legs as one mesh (for a render or an axis convention check)."""
    return merge([tube_shell(spec.lca_front, spec.lca_outer, 0.5 * spec.tube_od_m - spec.tube_wall_m,
                             0.5 * spec.tube_od_m, n=24),
                  tube_shell(spec.lca_rear, spec.lca_outer, 0.5 * spec.tube_od_m - spec.tube_wall_m,
                             0.5 * spec.tube_od_m, n=24)])
