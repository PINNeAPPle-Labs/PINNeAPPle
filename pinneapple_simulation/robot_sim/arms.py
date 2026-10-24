"""Ready-made arms for the robot sim, built from PINNeAPPle's material data and link models so the same masses go to
Newton (URDF), Omniverse / Isaac Sim (UsdPhysics) and PINNeAPPle's own multibody tree."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spec import ArticulationSpec, Joint, Link

__all__ = ["TwoLinkArm"]


@dataclass
class TwoLinkArm:
    """A planar two-link arm in the x-z plane, joints about +y (USD / URDF convention), square aluminium tubes.

    ``motor``: point mass at the elbow; ``payload``: point mass at the tool point. Angles are from the horizontal,
    the elbow relative to the upper arm; positive angles turn x towards -z (about +y)."""
    l1: float = 0.6
    l2: float = 0.5
    tube: float = 0.05              # outer width [m]
    wall: float = 0.003             # wall [m]
    material: str = "al6061_t6"
    motor: float = 1.5              # kg at the elbow
    payload: float = 5.0            # kg at the tool point
    shoulder_z: float = 0.4
    base_mass: float = 20.0
    effort_limits: tuple = (120.0, 60.0)
    gravity: float = 9.81

    def material_props(self):
        from pinneapple_design.intent import materials
        m = materials.get(self.material)
        return {k: m.props[k].value for k in ("density", "youngs_modulus", "yield_strength")}

    def section(self):
        from pinneapple_simulation.numerical_solvers.beam_bvp_fdm import SectionProps
        return SectionProps.hollow_rectangular(self.tube, self.tube, self.wall)

    def bodies(self):
        """The two links as :class:`~pinneapple_simulation.multibody.tree.Body` (lumped tube + point masses)."""
        from pinneapple_simulation.multibody.arm import lumped_link
        mpm = self.material_props()["density"] * self.section().A_m2
        return (lumped_link("link1", self.l1, mpm, [(self.l1, self.motor)], self.tube),
                lumped_link("link2", self.l2, mpm, [(self.l2, self.payload)], self.tube))

    def tree(self):
        """PINNeAPPle's multibody tree of the arm (the independent reference for the simulator). Its joints turn
        about -y, so its angles and torques are the negatives of the simulator's."""
        from pinneapple_simulation.multibody.tree import Body, MultibodyTree
        b1, b2 = self.bodies()
        t = MultibodyTree(gravity=(0.0, 0.0, -self.gravity))
        t.add_body(Body("base", self.base_mass, [0, 0, 0], np.eye(3) * 0.5))
        t.add_revolute("shoulder", "base", b1, [0, 0, self.shoulder_z], [0, -1, 0])
        t.add_revolute("elbow", "link1", b2, [self.l1, 0, 0], [0, -1, 0])
        return t

    def spec(self, colors=None, drive: str = "effort") -> ArticulationSpec:
        b1, b2 = self.bodies()
        c = colors or [(0.25, 0.25, 0.3), (0.75, 0.75, 0.78), (0.75, 0.75, 0.78)]
        z = self.shoulder_z
        return ArticulationSpec("Arm", [
            Link("base", self.base_mass, (0.25, 0.25, z - 0.06), (0, 0, 0), com=(0, 0, z / 2),
                 inertia=(0.5, 0.5, 0.5), box_offset=(0, 0, (z - 0.06) / 2), color=c[0]),
            Link("link1", b1.mass, (self.l1, self.tube, self.tube), (0, 0, z), com=tuple(b1.com),
                 inertia=tuple(np.diag(b1.inertia)), box_offset=(self.l1 / 2, 0, 0), color=c[1]),
            Link("link2", b2.mass, (self.l2, self.tube, self.tube), (self.l1, 0, z), com=tuple(b2.com),
                 inertia=tuple(np.diag(b2.inertia)), box_offset=(self.l2 / 2, 0, 0), color=c[2])],
            [Joint("shoulder", "base", "link1", (0, 0, z), "Y", (-170, 170), drive=drive,
                   max_force=self.effort_limits[0]),
             Joint("elbow", "link1", "link2", (self.l1, 0, z), "Y", (-170, 170), drive=drive,
                   max_force=self.effort_limits[1])], gravity=self.gravity)
