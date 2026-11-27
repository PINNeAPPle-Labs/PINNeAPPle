"""Elastokinematics: what the wheel does when the forces are on, because the arms and the bushes are not rigid.

The rigid kinematics of :mod:`.kinematics` says where the wheel points when everything holds its shape. A tyre
sideways force of 3 kN bends the wishbones and squeezes the bushes, and the upright -- still rigid, still held by two
ball joints and a tie rod -- turns a little: *compliance steer*, and with it a slice of oversteer the designer has to
know about. That is the number this module computes, and the way it is computed is the FEM-MBD coupling of the
benchmark:

1. **Each ball-joint connection is three linear springs in series**: the arm's own flexibility at the ball joint
   (a 3 x 3 compliance from CalculiX, three unit loads, :func:`pinneapple_simulation.suspension.structure.response`),
   plus the rubber of the bush pair (:func:`bush_compliance`). In series: ``C = C_arm + C_bush``,
   ``K = C^-1``.
2. **The upright is a rigid body on three such springs** -- at the lower ball joint, the upper ball joint and the outer
   tie-rod end. Its 6 x 6 stiffness about the wheel centre is the standard rigid-body-on-springs assembly
   ``K = sum_i J_i^T K_i J_i`` with ``J_i = [[p_i - c]_x, I]`` (the map from a body twist to the displacement of
   point ``i``). This is the same assembly a multibody code builds for a bushing-mounted body; here the legs are the
   two wishbones and the tie rod instead of a Stewart platform.
3. **The wheel loads are applied at the contact patch**, and the response there gives the flanking numbers: lateral
   compliance (mm/kN), longitudinal compliance, compliance steer (deg/kN) and compliance camber (deg/kN).

Verification in :mod:`pinneapple_simulation.suspension.verification`: with infinitely stiff joints every number goes
to zero (the rigid case is the limit), the 6 x 6 matrix is symmetric positive definite, and a symmetric three-spring
arrangement reproduces the hand calculation of a rigid body on three springs.

Sign conventions follow :mod:`.kinematics`: compliance steer positive is toe-in under the load named, compliance
camber negative is the top of the wheel leaning inboard.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .hardpoints import Corner

__all__ = ["BUSH", "bush_compliance", "body_stiffness", "elastokinematics", "WheelLoads"]

#: Radial stiffness of one wishbone bush, N/m. Rubber wishbone bushes are 1-5 kN/mm radially; the pair at a wishbone's
#: two pivots acts in parallel, which is how :func:`bush_compliance` counts them.
BUSH = {"wishbone_n_per_m": 2.0e6, "tie_rod_n_per_m": 2.0e7}


def bush_compliance(n_bushes: int, k_n_per_m: float) -> np.ndarray:
    """3 x 3 compliance of ``n_bushes`` identical bushes in parallel, isotropic (radial = axial here: a first-order
    model of a rubber bush, whose axial stiffness is 3-10x its radial one -- documented, not hidden)."""
    return np.eye(3) / (n_bushes * k_n_per_m)


@dataclass
class WheelLoads:
    """The tyre forces at the contact patch, newtons, right-hand corner: forward, outboard, up."""
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0


def _cross(p: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -p[2], p[1]], [p[2], 0.0, -p[0]], [-p[1], p[0], 0.0]])


def _jacobian(p: np.ndarray, c: np.ndarray) -> np.ndarray:
    """3 x 6 map from a twist (rotation about ``c``, translation of ``c``) to the displacement of point ``p``."""
    return np.hstack([_cross(p - c), np.eye(3)])


def body_stiffness(c: Corner, joints: dict[str, np.ndarray], origin: np.ndarray | None = None) -> np.ndarray:
    """6 x 6 stiffness of the upright about ``origin`` (the wheel centre by default), from the joint stiffnesses."""
    c0 = np.asarray(origin if origin is not None else c.wheel_centre, float)
    K = np.zeros((6, 6))
    for name, Kj in joints.items():
        J = _jacobian(np.asarray(_joint_point(c, name), float), c0)
        K += J.T @ Kj @ J
    return K


def _joint_point(c: Corner, name: str) -> np.ndarray:
    return {"lbj": c.lbj, "ubj": c.ubj, "tie_outer": c.tie_outer}[name]


def elastokinematics(c: Corner, compliance: dict[str, np.ndarray], *,
                     bush: dict[str, float] | None = None) -> dict[str, float]:
    """Compliance steer, compliance camber and the contact-patch compliances of one corner.

    ``compliance``: ``{"lower": 3 x 3, "upper": 3 x 3}`` ball-joint compliance of each wishbone in m/N, as
    :func:`pinneapple_simulation.suspension.structure.response` returns it. The tie rod is stiff (a steel rod and two
    ball joints): its compliance is the rubber of its inner joint only.
    """
    bush = bush or BUSH
    joints = {}
    for name, key, n_bush in (("lbj", "lower", 2), ("ubj", "upper", 2), ("tie_outer", "tie", 1)):
        C_arm = np.asarray(compliance[key], float) if key in compliance else np.zeros((3, 3))
        k_bush = bush["wishbone_n_per_m"] if key != "tie" else bush["tie_rod_n_per_m"]
        C = C_arm + bush_compliance(n_bush, k_bush)
        joints[name] = np.linalg.inv(C)

    c0 = c.wheel_centre
    K = body_stiffness(c, joints, c0)
    Cw = np.linalg.inv(K)                                        # 6 x 6 compliance of the body about the wheel centre
    contact = c0 + np.array([0.0, 0.0, -c.wheel_radius])          # contact patch, straight below the wheel centre
    Jc = _jacobian(contact, c0)
    C_contact = Jc @ Cw @ Jc.T                                    # 3 x 3 translational compliance at the patch

    out: dict[str, float] = {
        "lateral_compliance_mm_per_kN": float(C_contact[1, 1]) * 1e6,
        "longitudinal_compliance_mm_per_kN": float(C_contact[0, 0]) * 1e6,
        "vertical_compliance_mm_per_kN": float(C_contact[2, 2]) * 1e6,
        "body_stiffness_min_eig": float(np.linalg.eigvalsh(0.5 * (K + K.T)).min()),
    }
    # the toe and camber a unit tyre force turns into: the rotation part of the body twist, in the sign convention of
    # kinematics (toe = -yaw, camber = -roll), degrees per kN
    def twist(load: np.ndarray) -> np.ndarray:
        return Cw @ (Jc.T @ np.asarray(load, float))

    lateral = twist((0.0, 1000.0, 0.0))
    braking = twist((-1000.0, 0.0, 0.0))                          # -x: the force a braking tyre puts on the arm
    out["compliance_steer_deg_per_kN_lateral"] = -math.degrees(lateral[5])
    out["compliance_camber_deg_per_kN_lateral"] = -math.degrees(lateral[3])
    out["compliance_toe_deg_per_kN_braking"] = -math.degrees(braking[5])
    return out
