"""The elastokinematic half: what the bushes let the wheel do under load.

The kinematics of :mod:`pinneapple_design.suspension.kinematics` assumes rigid bushings. Real ones are rubber: a
lateral force at the contact patch shears the lower and upper arm bushes, the arm moves, and the wheel steers and
cambles — *elastokinematics*, the part of a suspension that decides how a car feels on a straight road and how much its
rear axle steers under braking.

The model is a first-order one, and it is a real one rather than a fitted one. Six **compliance modes** span what the
bushes allow: each arm translating fore-aft and laterally on its bushes, and the tie rod's inner end doing the same.
Each mode has the stiffness of the bushes it stretches — two radial for a lateral translation, two axial for a fore-aft
one — and a unit amplitude of it is fed through the same closed-loop solver the kinematics uses, which gives how much
the wheel steers, cambles and moves per metre of that mode:

    Q_i = F . d(cp)/dq_i + M_z d(toe)/dq_i          (virtual work of the wheel force on mode i)
    q   = K^-1 Q                                    (the bushes' elastic equilibrium)
    d(toe) = sum_i q_i d(toe)/dq_i                  (and the same for camber)

The wheel is held at its ride height while the load is applied, which is what a compliance rig does. What the model
leaves out: the tyres' own compliance, the bushes' nonlinearity, rate and temperature, friction in the ball joints, and
the cross-coupling of a bush loaded in two directions at once (the modes are treated as independent).

    from pinneapple_design.suspension import front_corner, wheel_compliance
    wheel_compliance(front_corner(), lateral_n=3000.0)["toe_deg_per_kn"]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hardpoints import SuspensionSpec
from .kinematics import pose

__all__ = ["ComplianceMode", "compliance_modes", "wheel_compliance", "mode_responses"]

#: displacement used to differentiate the linkage with respect to a mode [m] — 0.1 mm, well inside the linear range
STEP_M = 1e-4


@dataclass(frozen=True)
class ComplianceMode:
    """One direction the bushes let a body move: which body, which way, and the stiffness that resists it."""
    name: str
    body: str                                # "lower", "upper" or "tierod" (the solver's displacement groups)
    direction: tuple[float, float, float]
    stiffness_n_per_m: float
    what: str

    def offsets(self, magnitude_m: float = STEP_M) -> dict[str, np.ndarray]:
        return {self.body: np.asarray(self.direction, float) * magnitude_m}


def compliance_modes(spec: SuspensionSpec) -> list[ComplianceMode]:
    """The six modes, with the stiffness of the bushes they stretch: two per arm plus the tie rod's."""
    return [
        ComplianceMode("lower arm, lateral", "lower", (0.0, 1.0, 0.0),
                       2.0 * spec.bushing_lca.radial_n_per_m, "half-track and camber under a lateral force"),
        ComplianceMode("lower arm, fore-aft", "lower", (1.0, 0.0, 0.0),
                       2.0 * spec.bushing_lca.axial_n_per_m, "caster and toe under braking"),
        ComplianceMode("upper arm, lateral", "upper", (0.0, 1.0, 0.0),
                       2.0 * spec.bushing_uca.radial_n_per_m, "camber under a lateral force"),
        ComplianceMode("upper arm, fore-aft", "upper", (1.0, 0.0, 0.0),
                       2.0 * spec.bushing_uca.axial_n_per_m, "caster under braking"),
        ComplianceMode("tie rod, lateral", "tierod", (0.0, 1.0, 0.0),
                       spec.bushing_tierod.radial_n_per_m, "steer under a lateral force"),
        ComplianceMode("tie rod, fore-aft", "tierod", (1.0, 0.0, 0.0),
                       spec.bushing_tierod.axial_n_per_m, "steer under braking"),
    ]


def mode_responses(spec: SuspensionSpec, modes: list[ComplianceMode] | None = None) -> dict[str, dict]:
    """Per metre of each mode: how much the wheel steers, cambles, and moves at the contact patch.

    Returns
    -------
    responses : dict
        ``{mode: {"d_toe_rad", "d_camber_rad", "d_cp": (3,), "stiffness_n_per_m"}}``: angles and contact-patch motion
        per metre of the mode's amplitude.
    """
    modes = modes or compliance_modes(spec)
    base = pose(spec, 0.0)
    out = {}
    for m in modes:
        moved = pose(spec, 0.0, offsets=m.offsets())
        out[m.name] = {
            "d_toe_rad": float(np.radians(moved.toe_deg - base.toe_deg)) / STEP_M,
            "d_camber_rad": float(np.radians(moved.camber_deg - base.camber_deg)) / STEP_M,
            "d_cp": (moved.contact_patch - base.contact_patch) / STEP_M,
            "stiffness_n_per_m": m.stiffness_n_per_m,
            "what": m.what,
        }
    return out


def _solve(force: np.ndarray, aligning_nm: float, resp: dict) -> dict[str, float]:
    """The elastic equilibrium of the modes under a wheel force and an aligning torque."""
    names = list(resp)
    k = np.array([resp[n]["stiffness_n_per_m"] for n in names])
    q = np.array([float(np.dot(force, resp[n]["d_cp"])) / k[i] for i, n in enumerate(names)])
    toe = sum(q[i] * resp[n]["d_toe_rad"] for i, n in enumerate(names))
    camber = sum(q[i] * resp[n]["d_camber_rad"] for i, n in enumerate(names))
    half_track = sum(q[i] * resp[n]["d_cp"][1] for i, n in enumerate(names))
    if aligning_nm:                     # the torque acts about the kingpin and the tie rod carries it
        q_toe = np.array([aligning_nm * resp[n]["d_toe_rad"] / k[i] for i, n in enumerate(names)])
        toe += sum(q_toe[i] * resp[n]["d_toe_rad"] for i, n in enumerate(names))
    return {"toe_rad": float(toe), "camber_rad": float(camber), "half_track_m": float(half_track),
            "mode_amplitudes_m": {n: float(q[i]) for i, n in enumerate(names)}}


def wheel_compliance(spec: SuspensionSpec, lateral_n: float = 3000.0, longitudinal_n: float = 2000.0,
                     aligning_nm: float = 150.0) -> dict:
    """The wheel's elastic response to a lateral force, a longitudinal force and an aligning torque.

    Returns
    -------
    out : dict
        ``toe_deg_per_kn`` and ``camber_deg_per_kn`` (lateral force), ``toe_deg_per_kn_braking`` and
        ``camber_deg_per_kn_braking`` (longitudinal), ``toe_deg_per_nm`` (aligning torque), ``half_track_mm_per_kn``,
        the absolute angles at the loads given, ``share`` (each mode's part of the lateral steer) and ``bush_stiffness``.
    """
    resp = mode_responses(spec)
    lat = _solve(np.array([0.0, lateral_n, 0.0]), 0.0, resp)
    brk = _solve(np.array([-longitudinal_n, 0.0, 0.0]), 0.0, resp)
    ali = _solve(np.zeros(3), aligning_nm, resp)
    toe_lat = float(np.degrees(lat["toe_rad"]))
    toe_brk = float(np.degrees(brk["toe_rad"]))
    names = list(resp)
    share = {n: (lat["mode_amplitudes_m"][n] * resp[n]["d_toe_rad"] / lat["toe_rad"]
                 if abs(lat["toe_rad"]) > 1e-18 else 0.0) for n in names}
    return {"toe_deg_per_kn": toe_lat / (lateral_n / 1e3),
            "camber_deg_per_kn": float(np.degrees(lat["camber_rad"])) / (lateral_n / 1e3),
            "half_track_mm_per_kn": lat["half_track_m"] * 1e3 / (lateral_n / 1e3),
            "toe_deg_per_kn_braking": toe_brk / (longitudinal_n / 1e3),
            "camber_deg_per_kn_braking": float(np.degrees(brk["camber_rad"])) / (longitudinal_n / 1e3),
            "toe_deg_per_nm": float(np.degrees(ali["toe_rad"])) / aligning_nm if aligning_nm else float("nan"),
            "toe_deg_lateral": toe_lat, "toe_deg_braking": toe_brk,
            "camber_deg_lateral": float(np.degrees(lat["camber_rad"])),
            "half_track_mm_lateral": lat["half_track_m"] * 1e3,
            "share": share, "bush_stiffness_n_per_m": {n: resp[n]["stiffness_n_per_m"] for n in names},
            "lateral_n": float(lateral_n), "longitudinal_n": float(longitudinal_n), "aligning_nm": float(aligning_nm)}
