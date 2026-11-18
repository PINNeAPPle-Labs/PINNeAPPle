"""The loads the corner hands to the arms.

A suspension arm is sized by a handful of load cases that bracket a car's life: static, a bump or a kerb strike, hard
cornering, braking, and a combination. Each case is a force at the contact patch; the upright turns it into a force at
each ball joint, and the arms turn those into stress.

The upright is a three-force body: the wheel's force at the contact patch, the upper arm's and the lower arm's. The
upper arm is taken as a two-force member — its line runs from its inner pivots through its ball joint, so its reaction
has a known direction and one unknown magnitude, which a moment balance about the lower ball joint gives (the
projection of the balance on the direction that force can produce). Whatever force and moment the upper arm cannot
take, the lower arm does: it is the member that carries bending as well as axial load, which is what a wishbone is
for, and its root moment is what the sizing is about.

The load magnitudes are the usual static design envelope of a road car, quoted per unit of the corner's static load
``m g`` — not a measured duty cycle and not a fatigue spectrum:

===============  =========================  ==================================================
case             ``(Fx, Fy, Fz) / (m g)``   what it stands for
===============  =========================  ==================================================
static           ``(0, 0, 1)``             the car at rest
bump 3 g         ``(0, 0, 3)``             a pothole or a kerb, the wheel's own mass included
cornering 1.1 g  ``(0, 1.1, 1)``           the outer wheel at the limit of road-tyre grip
braking 1.2 g    ``(-1.2, 0, 1)``          the front axle under full braking
kerb + corner    ``(-0.6, 1.2, 2.5)``      the combined case, which usually sizes the arm
===============  =========================  ==================================================

Not in the model: the arm's own inertia, friction in the ball joints, the spring's reaction on the lower arm (the damper
carries most of the vertical load once the wheel moves, so the case is conservative — the arm is checked as if the
spring did nothing), gyroscopic and brake-torque effects, and the tyre's own compliance.
"""
from __future__ import annotations

import numpy as np

from .hardpoints import GRAVITY, SuspensionSpec
from .kinematics import Pose

__all__ = ["corner_load_cases", "arm_loads", "LOAD_CASES", "worst_case"]

#: name -> (Fx, Fy, Fz) per unit of the corner's static load, at the contact patch; ``x`` forward
LOAD_CASES: dict[str, tuple[float, float, float]] = {
    "static": (0.0, 0.0, 1.0),
    "bump 3 g": (0.0, 0.0, 3.0),
    "cornering 1.1 g": (0.0, 1.1, 1.0),
    "braking 1.2 g": (-1.2, 0.0, 1.0),
    "kerb strike + cornering": (-0.6, 1.2, 2.5),
}


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n else v


def corner_load_cases(spec: SuspensionSpec, cases: dict[str, tuple[float, float, float]] | None = None
                      ) -> dict[str, np.ndarray]:
    """The wheel forces at the contact patch [N] of each case, scaled by the corner's static load ``m g``.

    Returns
    -------
    cases : dict
        ``{case: array([Fx, Fy, Fz])}`` at the contact patch, ``x`` forward, ``y`` left, ``z`` up.
    """
    static = spec.corner_mass_kg * GRAVITY
    return {k: np.asarray(v, float) * static for k, v in (cases or LOAD_CASES).items()}


def arm_loads(spec: SuspensionSpec, p: Pose, wheel_force) -> dict[str, float]:
    """Turn a wheel force at the contact patch into the force each arm and the moment the lower arm carries.

    The upper arm is a two-force member along the line from its inner pivots to its ball joint; the moment balance
    about the lower ball joint gives its magnitude (projected on the direction that force can produce, so the residual
    moment is the lower arm's). The lower ball joint then takes the rest of the force.

    Returns
    -------
    loads : dict
        ``upper_axial_n`` (positive in compression), ``lower_force_n`` (the ball joint's resultant),
        ``lower_axial_n`` (its component along the lower arm's leg), ``lower_moment_nm`` (root moment: the ball joint
        force about the inner pivots plus the residual moment the upper arm cannot take), ``leg_length_m`` and
        ``residual_force_n`` (the part of the balance the two-force model leaves to the other members).
    """
    F = np.asarray(wheel_force, float).reshape(3)
    lower, upper = spec.lca_outer, spec.uca_outer
    cp = p.contact_patch
    moment_at_lower = np.cross(cp - lower, F)                     # where the wheel's force pushes, about the LBJ
    u = _unit(upper - spec.uca_inner_mid)                         # the upper arm's line
    arm = np.cross(upper - lower, u)                              # moment per unit of the upper arm's force
    if float(np.dot(arm, arm)) < 1e-18:
        raise ValueError("the upper arm's line passes through the lower ball joint: no moment can be balanced")
    f_upper = -float(np.dot(moment_at_lower, arm)) / float(np.dot(arm, arm))
    F_upper = f_upper * u
    residual = moment_at_lower + np.cross(upper - lower, F_upper)  # the moment left to the lower arm
    F_lower = -(F + F_upper)
    leg = _unit(lower - spec.lca_inner_mid)
    axial_lower = float(np.dot(F_lower, leg))
    # the arm is a beam from the pivot line, through the spring seat, to the ball joint. The spring balances the wheel
    # load about the pivot line (that is what the pivot axis being free means), so the arm's bending is the ball joint's
    # force about the spring seat — the section a wishbone actually fails at — and not a cantilever at the pivots.
    seat = p.spring_lower
    bending = np.cross(lower - seat, F_lower) + residual
    return {"upper_axial_n": float(f_upper), "lower_force_n": float(np.linalg.norm(F_lower)),
            "lower_axial_n": float(axial_lower), "lower_moment_nm": float(np.linalg.norm(bending)),
            "spring_lever_m": float(np.linalg.norm(lower - seat)),
            "residual_force_n": float(np.linalg.norm(F - F_upper - (-F_lower))),
            "leg_length_m": float(max(np.linalg.norm(lower - spec.lca_front),
                                      np.linalg.norm(lower - spec.lca_rear))),
            "wheel_force_n": float(np.linalg.norm(F))}


def worst_case(spec: SuspensionSpec, p: Pose, section=None, cases: dict[str, np.ndarray] | None = None) -> dict:
    """The case that sizes the lower arm: the largest outer-fibre stress of the set, with its name and its numbers."""
    from .structure import arm_check

    best = None
    for name, force in (cases or corner_load_cases(spec)).items():
        chk = arm_check(spec, p, force, section) | {"case": name}
        if best is None or chk["sigma_vm_pa"] > best["sigma_vm_pa"]:
            best = chk
    return best
