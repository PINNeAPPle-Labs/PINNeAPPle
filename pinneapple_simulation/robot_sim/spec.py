"""A robot as plain data (links, joints, drives), written out as URDF for any simulator and as UsdPhysics for
Omniverse / Isaac Sim (``pinneapple_tools.nvidia.isaac.author_articulation``)."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from xml.sax.saxutils import escape

import numpy as np

__all__ = ["Link", "Joint", "ArticulationSpec", "box_inertia", "cylinder_inertia", "spec_to_urdf"]


def box_inertia(mass: float, size) -> np.ndarray:
    """Principal inertia ``(Ixx, Iyy, Izz)`` of a solid box ``size = (a, b, c)`` about its centre."""
    a, b, c = map(float, size)
    return mass / 12.0 * np.array([b * b + c * c, a * a + c * c, a * a + b * b])


def cylinder_inertia(mass: float, radius: float, length: float) -> np.ndarray:
    """Principal inertia ``(Ixx, Iyy, Izz)`` of a solid cylinder with its axis along y."""
    ia = 0.5 * mass * radius ** 2
    it = mass * (3 * radius ** 2 + length ** 2) / 12.0
    return np.array([it, ia, it])


@dataclass
class Link:
    name: str
    mass: float                                  # kg
    size: Sequence[float]                        # box (x, y, z) [m]; cylinder: (diameter, length) along y
    position: Sequence[float]                    # world position of the link frame at the authored pose [m]
    com: Sequence[float] = (0.0, 0.0, 0.0)       # centre of mass in the link frame [m]
    inertia: Sequence[float] | None = None       # principal inertia about the com [kg m^2]; default: the shape's
    box_offset: Sequence[float] = (0.0, 0.0, 0.0)  # shape centre in the link frame [m]
    color: Sequence[float] = (0.7, 0.7, 0.72)
    collide: bool = True
    shape: str = "box"                           # "box" | "cylinder"

    def principal_inertia(self) -> np.ndarray:
        if self.inertia is not None:
            return np.asarray(self.inertia, float)
        if self.shape == "cylinder":
            return cylinder_inertia(self.mass, self.size[0] / 2, self.size[1])
        return box_inertia(self.mass, self.size)


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    anchor: Sequence[float]                      # world position of the joint at the authored pose [m]
    axis: str = "Y"                              # "X" | "Y" | "Z" (positive axis; no authored rotations)
    limits_deg: tuple[float, float] | None = None
    drive: str = "position"                      # "position" | "effort" | "velocity"
    stiffness: float = 0.0                       # N m / deg (USD angular drives are per degree)
    damping: float = 0.0                         # N m s / deg
    max_force: float = 1e6                       # N m (effort limit)
    max_velocity: float = 1e3                    # rad/s


@dataclass
class ArticulationSpec:
    name: str
    links: list[Link]
    joints: list[Joint]
    fixed_base: bool = True
    gravity: float = 9.81
    ground: bool = True
    info: dict[str, Any] = field(default_factory=dict)

    def joint_names(self) -> list[str]:
        return [j.name for j in self.joints]


def spec_to_urdf(spec: ArticulationSpec) -> str:
    """URDF of the spec: inertials, box / cylinder colliders and visuals, revolute joints with limits."""
    pos = {L.name: np.asarray(L.position, float) for L in spec.links}
    f = lambda v: " ".join(f"{float(x):.9g}" for x in v)                        # noqa: E731
    out = [f'<robot name="{escape(spec.name)}">']
    for L in spec.links:
        inr = L.principal_inertia()
        if L.shape == "cylinder":                       # URDF cylinders are along z: turn them onto y
            geo = f'<cylinder radius="{L.size[0] / 2:.9g}" length="{L.size[1]:.9g}"/>'
            rpy = "1.5707963267948966 0 0"
        else:
            geo, rpy = f'<box size="{f(L.size)}"/>', "0 0 0"
        g = f'<origin xyz="{f(L.box_offset)}" rpy="{rpy}"/><geometry>{geo}</geometry>'
        out.append(f' <link name="{escape(L.name)}"><inertial><origin xyz="{f(L.com)}"/><mass value="{L.mass:.9g}"/>'
                   f'<inertia ixx="{inr[0]:.9g}" iyy="{inr[1]:.9g}" izz="{inr[2]:.9g}" ixy="0" ixz="0" iyz="0"/>'
                   f'</inertial><visual>{g}</visual>' + (f"<collision>{g}</collision>" if L.collide else "")
                   + "</link>")
    for J in spec.joints:
        ax = {"X": "1 0 0", "Y": "0 1 0", "Z": "0 0 1"}[J.axis.upper()]
        lo, hi = (np.radians(J.limits_deg) if J.limits_deg is not None else (-np.pi, np.pi))
        org = np.asarray(J.anchor, float) - pos[J.parent]
        child_off = pos[J.child] - np.asarray(J.anchor, float)
        if np.linalg.norm(child_off) > 1e-12:
            raise ValueError(f"joint {J.name}: the child link frame must sit on the joint anchor for URDF")
        out.append(f' <joint name="{escape(J.name)}" type="revolute"><parent link="{escape(J.parent)}"/>'
                   f'<child link="{escape(J.child)}"/><origin xyz="{f(org)}"/><axis xyz="{ax}"/>'
                   f'<limit lower="{lo:.9g}" upper="{hi:.9g}" effort="{J.max_force:.9g}" '
                   f'velocity="{J.max_velocity:.9g}"/></joint>')
    out.append("</robot>")
    return "\n".join(out) + "\n"
