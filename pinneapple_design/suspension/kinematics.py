"""The multibody part: the double wishbone as a closed kinematic loop, solved for wheel travel.

The linkage has one degree of freedom once the rack is fixed, and three unknowns get it there. The lower arm turns
about the line through its two inner pivots by :math:`\\theta`; the upper arm, about its own line, by :math:`\\phi`;
the upright is the rigid body that carries the upper ball joint, the steering arm and the wheel, and it spins about
the kingpin by :math:`\\psi`. Three scalar equations close the loop:

1. the kingpin length is the upright's own (``|U - L| = |U0 - L0|``), which ties :math:`\\phi` to :math:`\\theta`;
2. the tie rod keeps its length (``|T - tierod_inner| = |T0 - tierod_inner|``), which ties :math:`\\psi` to the arm
   angles — this is where bump steer comes from;
3. the wheel centre sits at the travelled height (``W.z = W0.z + travel``), which fixes :math:`\\theta`.

Newton on the 3-vector with a numeric Jacobian, seeded by the reference pose and continued along the travel sweep, so
a 5 mm step lands in two or three iterations and the sweep of a full bump/rebound costs milliseconds. No constraint
library, no mass matrix: this is the *kinematic* half of the multibody loop (the dynamic half is
:mod:`pinneapple_simulation.multibody`).

What is reported, per travel position:

* **camber** [deg], SAE sign: negative is the top of the wheel leaning inboard;
* **toe** [deg], positive is toe-in on the modelled (left) corner — the front of the wheel pointing at the car's
  centreline;
* **caster** and **KPI** [deg]: the kingpin's lean in the side and front views;
* **scrub radius** and **mechanical trail** [m]: where the kingpin axis meets the ground, relative to the contact
  patch;
* **half-track** [m] and its change, **motion ratio** and **wheel rate** [N/m], and the **roll centre height** [m] of
  the axle (front-view instant centre, line to the contact patch, meeting the centreline).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .hardpoints import SuspensionSpec

__all__ = ["Pose", "pose", "pose_from_angles", "kinematic_sweep", "Sweep", "roll_centre_height_m",
           "wishbone_instant_centre"]

#: the largest change of the three unknowns a Newton step may take at once [rad] — a trust region, not a tolerance
MAX_STEP_RAD = 0.25


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n else v


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    a = _unit(np.asarray(axis, float))
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _rotate_about(p: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues: ``p`` turned by ``angle`` about the line through ``origin`` along ``axis``."""
    return origin + _axis_angle(axis, angle) @ (p - origin)


def _min_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The shortest rotation taking the direction ``a`` onto the direction ``b``."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    s = float(np.linalg.norm(v))
    if s < 1e-14:                                   # parallel (or antiparallel: a half turn about any perpendicular)
        if c > 0:
            return np.eye(3)
        other = np.cross(a, np.array([1.0, 0.0, 0.0]))
        return _axis_angle(other if np.linalg.norm(other) > 1e-9 else np.array([0.0, 0.0, 1.0]), math.pi)
    return _axis_angle(v, math.atan2(s, c))


def _circle(origin: np.ndarray, axis: np.ndarray, point: np.ndarray):
    """The circle a ball joint traces about a pivot axis: ``(centre, e1, e2, radius)``, ``e1`` through ``point``."""
    v = point - origin
    along = float(v @ axis)
    perp = v - along * axis
    radius = float(np.linalg.norm(perp))
    e1 = perp / radius
    return origin + along * axis, e1, np.cross(axis, e1), radius


def _newton(f, u0: np.ndarray, tol: float = 1e-13, iters: int = 60, h: float = 1e-8):
    """Newton on a small residual vector: central-difference Jacobian and a halving line search on its norm."""
    u = np.asarray(u0, float).copy()
    r = np.atleast_1d(f(u))
    for it in range(1, iters + 1):
        n = float(np.linalg.norm(r))
        if n < tol:
            return u, n, it
        J = np.empty((len(r), len(u)))
        for j in range(len(u)):
            du = np.zeros_like(u)
            du[j] = h
            J[:, j] = (np.atleast_1d(f(u + du)) - np.atleast_1d(f(u - du))) / (2.0 * h)
        try:
            step = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:                                   # pragma: no cover - degenerate geometry
            step = np.linalg.lstsq(J, -r, rcond=None)[0]
        a = 1.0
        for _ in range(40):
            r_new = np.atleast_1d(f(u + a * step))
            if float(np.linalg.norm(r_new)) < n:
                u, r = u + a * step, r_new
                break
            a *= 0.5
        else:                                                           # pragma: no cover - stuck at the solution
            break
    return u, float(np.linalg.norm(np.atleast_1d(r))), it


def _upright_transform(g: dict, lower: np.ndarray, upper: np.ndarray, spin: float):
    """The rigid transform of the upright that takes the reference ball joints to ``lower``/``upper`` and spins it by
    ``spin`` about the (current) kingpin."""
    a0 = _unit(g["uca_outer"] - g["lca_outer"])
    a = _unit(upper - lower)
    axis = np.cross(a0, a)
    s, c = float(np.linalg.norm(axis)), float(np.dot(a0, a))
    R1 = _axis_angle(axis, np.arctan2(s, c)) if s > 1e-14 else np.eye(3)
    R = _axis_angle(a, spin) @ R1
    return R, lower - R @ g["lca_outer"]


@dataclass(frozen=True)
class Pose:
    """The linkage at one wheel-travel position: the solved angles, the transformed points and the wheel plane."""
    spec: SuspensionSpec
    travel_m: float
    theta: float                     # lower arm rotation about its inner axis [rad]
    phi: float                       # upper arm rotation [rad]
    psi: float                       # upright spin about the kingpin [rad]
    lower_ball: np.ndarray
    upper_ball: np.ndarray
    tierod_outer: np.ndarray
    wheel_centre: np.ndarray
    spring_lower: np.ndarray
    R: np.ndarray
    translation: np.ndarray

    # -- the wheel plane ---------------------------------------------------------------------------------------
    @property
    def spin_axis(self) -> np.ndarray:
        """The wheel's axle direction (unit), carried by the upright."""
        return _unit(self.R @ self.spec.wheel_axis)

    @property
    def forward(self) -> np.ndarray:
        """The wheel plane's forward direction: the upright's ``x`` axis with the axle component removed."""
        f = self.R @ np.array([1.0, 0.0, 0.0])
        return _unit(f - np.dot(f, self.spin_axis) * self.spin_axis)

    @property
    def kingpin(self) -> np.ndarray:
        return _unit(self.upper_ball - self.lower_ball)

    @property
    def camber_deg(self) -> float:
        """SAE camber: negative when the top of the wheel leans toward the car's centreline."""
        return float(-np.degrees(np.arcsin(np.clip(self.spin_axis[2], -1.0, 1.0))))

    @property
    def toe_deg(self) -> float:
        """Positive is toe-in on the left corner: the front of the wheel pointing at the centreline (``-y``)."""
        f = self.forward
        return float(np.degrees(np.arctan2(-f[1], f[0])))

    @property
    def caster_deg(self) -> float:
        """Positive when the top of the kingpin leans rearward."""
        a = self.kingpin
        return float(np.degrees(np.arctan2(-a[0], a[2])))

    @property
    def kpi_deg(self) -> float:
        """Kingpin inclination: positive when the top of the kingpin leans inboard."""
        a = self.kingpin
        return float(np.degrees(np.arctan2(-a[1], a[2])))

    @property
    def wheel_up(self) -> np.ndarray:
        """The wheel plane's "up": the world vertical projected on the plane (unit)."""
        up = np.array([0.0, 0.0, 1.0])
        return _unit(up - np.dot(up, self.spin_axis) * self.spin_axis)

    @property
    def contact_patch(self) -> np.ndarray:
        """The point of the tyre nearest the ground: the wheel centre less the radius along the wheel's up."""
        return self.wheel_centre - self.spec.tyre_radius_m * self.wheel_up

    @property
    def kingpin_at_ground(self) -> np.ndarray:
        a = self.kingpin
        if abs(a[2]) < 1e-9:
            return self.lower_ball.copy()
        return self.lower_ball - self.lower_ball[2] / a[2] * a

    @property
    def scrub_radius_m(self) -> float:
        """Lateral distance at the ground from the contact patch's centre to the kingpin axis; positive when the
        kingpin meets the ground inboard of the tyre centre."""
        return float(self.contact_patch[1] - self.kingpin_at_ground[1])

    @property
    def trail_m(self) -> float:
        """Longitudinal distance from the contact patch to the kingpin axis at the ground; positive when the axis
        meets the ground ahead of the contact patch (stabilising)."""
        return float(self.kingpin_at_ground[0] - self.contact_patch[0])

    @property
    def half_track_m(self) -> float:
        return float(self.wheel_centre[1])

    @property
    def spring_length_m(self) -> float:
        return float(np.linalg.norm(self.spec.spring_upper - self.spring_lower))

    def point(self, p: np.ndarray) -> np.ndarray:
        """A reference-frame point carried by the upright (steering arm, wheel centre, brake caliper)."""
        return self.R @ np.asarray(p, float) + self.translation


def _residual(g: dict, travel_m: float, x, offsets: tuple) -> np.ndarray:
    """The three closure residuals [m] at the unknowns ``x`` — written with scalars, not arrays: the solver calls it a
    few hundred times per sweep and the numpy overhead of 3-vectors is what dominated the loop's cost."""
    theta, phi, spin = x
    dl, du, dt = offsets

    lower = _rot_at(g["lca_front"], g["lca_outer"], g["lca_axis"], theta, dl)
    upper = _rot_at(g["uca_front"], g["uca_outer"], g["uca_axis"], phi, du)
    a0 = g["a0"]
    dx, dy, dz = upper[0] - lower[0], upper[1] - lower[1], upper[2] - lower[2]
    n = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
    ax, ay, az = dx / n, dy / n, dz / n
    # R1: the rotation that takes the reference kingpin direction a0 to the current one
    cx = a0[1] * az - a0[2] * ay
    cy = a0[2] * ax - a0[0] * az
    cz = a0[0] * ay - a0[1] * ax
    s = math.sqrt(cx * cx + cy * cy + cz * cz)
    c = a0[0] * ax + a0[1] * ay + a0[2] * az
    if s > 1e-14:
        R1 = _matrix((cx, cy, cz), math.atan2(s, c))
    else:
        R1 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    R = _matmul(_matrix((ax, ay, az), spin), R1)
    Rlca = _matvec(R, g["lca_outer"])
    T = (lower[0] - Rlca[0], lower[1] - Rlca[1], lower[2] - Rlca[2])
    tierod = _matvec_add(R, g["tierod_outer"], T)
    wheel = _matvec_add(R, g["wheel_centre"], T)
    tx, ty, tz = tierod[0] - (g["tierod_inner"][0] + dt[0]), tierod[1] - (g["tierod_inner"][1] + dt[1]), \
        tierod[2] - (g["tierod_inner"][2] + dt[2])
    return np.array([n - g["kingpin"],
                     math.sqrt(tx * tx + ty * ty + tz * tz) - g["tierod_length"],
                     wheel[2] - (g["wheel_centre"][2] + travel_m)])


def _rot_at(origin, point, axis, angle, offset) -> tuple:
    """Rodrigues about the line through ``origin`` along ``axis``, plus a rigid offset [m]."""
    R = _matrix(axis, angle)
    r = _matvec(R, (point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]))
    return (origin[0] + r[0] + offset[0], origin[1] + r[1] + offset[1], origin[2] + r[2] + offset[2])


def _matrix(axis, angle) -> tuple:
    """Rodrigues' rotation matrix (axis need not be unit) as a 3x3 tuple."""
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if n:
        ax, ay, az = ax / n, ay / n, az / n
    s, c = math.sin(angle), math.cos(angle)
    t = 1.0 - c
    return ((c + ax * ax * t, ax * ay * t - az * s, ax * az * t + ay * s),
            (ay * ax * t + az * s, c + ay * ay * t, ay * az * t - ax * s),
            (az * ax * t - ay * s, az * ay * t + ax * s, c + az * az * t))


def _matvec(R, v) -> tuple:
    return (R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
            R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
            R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2])


def _matvec_add(R, v, t) -> tuple:
    m = _matvec(R, v)
    return (m[0] + t[0], m[1] + t[1], m[2] + t[2])


def _matmul(A, B) -> tuple:
    return tuple(tuple(A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j] for j in range(3))
                 for i in range(3))


def _geometry(spec: SuspensionSpec) -> dict:
    """The constants of a solve, read from the spec once (its properties recompute their unit vectors every call)."""
    a0 = _unit(spec.uca_outer - spec.lca_outer)
    return {"lca_outer": spec.lca_outer, "lca_front": spec.lca_front, "lca_axis": spec.lca_axis,
            "uca_outer": spec.uca_outer, "uca_front": spec.uca_front, "uca_axis": spec.uca_axis,
            "tierod_outer": spec.tierod_outer, "tierod_inner": spec.tierod_inner,
            "wheel_centre": spec.wheel_centre, "kingpin": spec.kingpin_length_m,
            "tierod_length": spec.tierod_length_m, "a0": a0}


def _offset(offsets: dict | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three bushing displacements of a solve: the lower arm's (with its ball joint), the upper arm's, the tie
    rod's inner end."""
    o = offsets or {}
    z = np.zeros(3)
    return (np.asarray(o.get("lower", z), float), np.asarray(o.get("upper", z), float),
            np.asarray(o.get("tierod", z), float))


def pose(spec: SuspensionSpec, travel_m: float = 0.0, seed=None, tol: float = 1e-9, iters: int = 40,
         offsets: dict | None = None) -> Pose:
    """The linkage at ``travel_m`` metres of wheel travel (up is positive), solved from ``seed`` (the reference pose
    by default: pass the previous position's angles to continue a sweep).

    ``offsets`` displaces the bodies away from their rigid bushing positions — ``{"lower": (3,), "upper": (3,),
    "tierod": (3,)}`` in metres — which is how :mod:`pinneapple_design.suspension.compliance` asks what the bushes give
    when they deflect. The upright's transform is measured from the *nominal* hardpoints, so a displaced arm really
    moves the wheel instead of carrying its own reference with it.
    """
    g = _geometry(spec)
    off = _offset(offsets)
    dl, du, dt = off
    x = np.zeros(3) if seed is None else np.asarray(seed, float).copy()
    if abs(travel_m) < 1e-15 and not np.any(dl) and not np.any(du) and not np.any(dt):
        x = np.zeros(3)
    for _ in range(iters):
        r = _residual(g, travel_m, x, off)
        if np.max(np.abs(r)) < tol:
            break
        J = np.empty((3, 3))
        h = 1e-7 * max(1.0, float(np.max(np.abs(x))))
        for k in range(3):
            dx = np.zeros(3)
            dx[k] = h
            J[:, k] = (_residual(g, travel_m, x + dx, off) - r) / h
        try:
            step = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:                      # a singular pose is off the linkage's range
            raise ValueError(f"the linkage has no solution near travel {travel_m * 1e3:.0f} mm") from None
        if float(np.max(np.abs(step))) < 1e-14:            # a nanometre of travel: the closure is at the noise floor
            break
        # trust region: the spin about the kingpin is a poorly scaled unknown, and an undamped Newton step from the
        # reference pose can throw it hundreds of radians away from any solution
        limit = float(np.max(np.abs(step)))
        if limit > MAX_STEP_RAD:
            step = step * (MAX_STEP_RAD / limit)
        x = x + step
        if not np.all(np.isfinite(x)):
            raise ValueError(f"the linkage diverged at travel {travel_m * 1e3:.0f} mm")
    else:
        raise ValueError(f"the linkage did not close at travel {travel_m * 1e3:.0f} mm "
                         f"(residual {np.max(np.abs(_residual(g, travel_m, x, off))):.2e})")
    theta, phi, spin = x
    lower = _rotate_about(spec.lca_outer, spec.lca_front, spec.lca_axis, theta) + dl
    upper = _rotate_about(spec.uca_outer, spec.uca_front, spec.uca_axis, phi) + du
    R, T = _upright_transform(g, lower, upper, spin)
    return Pose(spec=spec, travel_m=float(travel_m), theta=float(theta), phi=float(phi), psi=float(spin),
                lower_ball=lower, upper_ball=upper, tierod_outer=R @ spec.tierod_outer + T,
                wheel_centre=R @ spec.wheel_centre + T, spring_lower=_rotate_about(
                    spec.spring_lower, spec.lca_front, spec.lca_axis, theta) + dl, R=R, translation=T)


def pose_from_angles(spec: SuspensionSpec, travel_m: float = 0.0, seed=None, tol: float = 1e-13,
                     iters: int = 60) -> Pose:
    """The same pose from three **joint angles** instead of the three closure unknowns: one angle along each wishbone's
    circle, one spin of the upright about the kingpin.

    A different unknown set and a different decomposition from :func:`pose` -- the circle a ball joint traces is
    parameterised directly and the upright is placed by its rigid lengths, where :func:`pose` keeps three scalars of
    a closure residual. The two agreeing is therefore evidence about the kinematics, not a tautology, and that is what
    :mod:`pinneapple_design.suspension.verification` measures.

    ``seed``: the three angles of the neighbouring travel (:func:`pose_from_angles` of the step before), which is what
    makes a sweep cheap. Bushing displacements are not accepted here: they are the compliance model's question, and
    that one goes through :func:`pose`.
    """
    c_low = _circle(spec.lca_front, spec.lca_axis, spec.lca_outer)
    c_up = _circle(spec.uca_front, spec.uca_axis, spec.uca_outer)
    kingpin0 = spec.uca_outer - spec.lca_outer
    d0 = float(np.linalg.norm(kingpin0))
    kp0 = kingpin0 / d0
    tierod_length = spec.tierod_length_m

    def ball(circle, angle: float) -> np.ndarray:
        centre, e1, e2, radius = circle
        return centre + radius * (math.cos(angle) * e1 + math.sin(angle) * e2)

    def build(u: np.ndarray) -> Pose:
        lower = ball(c_low, u[0])
        upper = ball(c_up, u[1])
        R = _axis_angle(upper - lower, u[2]) @ _min_rotation(kp0, upper - lower)
        t = lower - R @ spec.lca_outer
        return Pose(spec=spec, travel_m=float(travel_m), theta=float(u[0]), phi=float(u[1]), psi=float(u[2]),
                    lower_ball=lower, upper_ball=upper, tierod_outer=R @ spec.tierod_outer + t,
                    wheel_centre=R @ spec.wheel_centre + t,
                    spring_lower=_rotate_about(spec.spring_lower, spec.lca_front, spec.lca_axis, float(u[0])),
                    R=R, translation=t)

    def residuals(u: np.ndarray) -> np.ndarray:
        p = build(u)
        return np.array([np.linalg.norm(p.upper_ball - p.lower_ball) - d0,
                         np.linalg.norm(p.tierod_outer - spec.tierod_inner) - tierod_length,
                         float(p.wheel_centre[2] - spec.wheel_centre[2]) - float(travel_m)], float)

    u0 = np.zeros(3) if seed is None else np.asarray(seed, float)
    u, _, _ = _newton(residuals, u0, tol=tol, iters=iters)
    return build(u)


def wishbone_instant_centre(p: Pose) -> np.ndarray:
    """The front-view instant centre of the wishbones (``y``, ``z``): where the lines through each arm meet."""
    spec = p.spec
    a = np.array([spec.lca_outer[1] - spec.lca_inner_mid[1], spec.lca_outer[2] - spec.lca_inner_mid[2]])
    b = np.array([spec.uca_outer[1] - spec.uca_inner_mid[1], spec.uca_outer[2] - spec.uca_inner_mid[2]])
    A = np.array([[a[1], -a[0]], [b[1], -b[0]]])
    rhs = np.array([a[1] * spec.lca_outer[1] - a[0] * spec.lca_outer[2],
                    b[1] * spec.uca_outer[1] - b[0] * spec.uca_outer[2]])
    if abs(np.linalg.det(A)) < 1e-12:
        return np.array([0.0, spec.lca_outer[2]])          # parallel arms: the centre is at infinity, reported flat
    y, z = np.linalg.solve(A, rhs)
    return np.array([y, z])


def roll_centre_height_m(p: Pose) -> float:
    """The axle's roll centre: the line from the contact patch through the front-view instant centre, where it meets
    the car's centreline (``y = 0``)."""
    ic = wishbone_instant_centre(p)
    cp = p.contact_patch
    dy = ic[0] - cp[1]
    if abs(dy) < 1e-9:
        return float(cp[2])
    t = (0.0 - cp[1]) / dy
    return float(cp[2] + t * (ic[1] - cp[2]))


@dataclass(frozen=True)
class Sweep:
    """A wheel-travel sweep, with the curves the design is judged on."""
    travel_mm: np.ndarray
    camber_deg: np.ndarray
    toe_deg: np.ndarray
    caster_deg: np.ndarray
    kpi_deg: np.ndarray
    scrub_m: np.ndarray
    trail_m: np.ndarray
    half_track_m: np.ndarray
    motion_ratio: np.ndarray
    wheel_rate_n_per_m: np.ndarray
    ride_rate_n_per_m: np.ndarray
    roll_centre_m: np.ndarray
    spring_length_m: np.ndarray
    residual_m: float

    def at(self, travel_mm: float) -> dict[str, float]:
        i = int(np.argmin(np.abs(self.travel_mm - travel_mm)))
        return {k: float(getattr(self, k)[i]) for k in
                ("camber_deg", "toe_deg", "caster_deg", "kpi_deg", "scrub_m", "trail_m", "half_track_m",
                 "motion_ratio", "wheel_rate_n_per_m", "ride_rate_n_per_m", "roll_centre_m", "spring_length_m")}

    def _slope(self, y: np.ndarray, half_span_mm: float = 10.0) -> float:
        """``dy / d(travel)`` per metre, as the secant over ``+/- half_span_mm`` about the design height.

        A fixed span, interpolated on the sweep's own grid, so the loop (a coarse sweep) and the report (a fine one)
        measure the same thing: a slope that changed with the grid would make the constraint a discretisation."""
        grid = self.travel_mm
        y0 = float(np.interp(-half_span_mm, grid, y))
        y1 = float(np.interp(+half_span_mm, grid, y))
        return (y1 - y0) / (2.0 * half_span_mm * 1e-3)

    def summary(self) -> dict[str, float]:
        """The numbers a suspension is specified by, all read off the sweep (no separate closed form)."""
        i0 = int(np.argmin(np.abs(self.travel_mm)))
        return {
            "static_camber_deg": float(self.camber_deg[i0]),
            "static_toe_deg": float(self.toe_deg[i0]),
            "static_caster_deg": float(self.caster_deg[i0]),
            "static_kpi_deg": float(self.kpi_deg[i0]),
            "static_scrub_radius_mm": float(self.scrub_m[i0] * 1e3),
            "static_trail_mm": float(self.trail_m[i0] * 1e3),
            "camber_gain_deg_per_50mm": self._slope(self.camber_deg) * 0.050,
            "bump_steer_deg_per_50mm": self._slope(self.toe_deg) * 0.050,
            "caster_change_deg_per_50mm": self._slope(self.caster_deg) * 0.050,
            "half_track_change_mm_per_50mm": self._slope(self.half_track_m) * 0.050 * 1e3,
            "motion_ratio": float(self.motion_ratio[i0]),
            "wheel_rate_n_per_mm": float(self.wheel_rate_n_per_m[i0] / 1e3),
            "ride_rate_n_per_mm": float(self.ride_rate_n_per_m[i0] / 1e3),
            "roll_centre_height_mm": float(self.roll_centre_m[i0] * 1e3),
            "rebound_to_bump_ratio": float(abs(self.travel_mm.max() / self.travel_mm.min())),
            "kinematic_closure_residual_m": self.residual_m,
        }


def kinematic_sweep(spec: SuspensionSpec, travel_mm=(-60.0, 60.0, 5.0)) -> Sweep:
    """Solve the linkage through a wheel-travel range (``(from, to, step)`` mm, bump positive)."""
    lo, hi, step = travel_mm
    travels = np.arange(lo, hi + 0.5 * step, step)
    # continuation outwards from the reference pose: each point starts from the solution a step nearer zero, which is
    # what keeps the Newton solve on the branch of the linkage the design lives on
    order = sorted(range(len(travels)), key=lambda i: (abs(travels[i]) > 1e-9, abs(travels[i])))
    solved: dict[int, Pose] = {}
    seed = None
    for i in order:
        p = pose(spec, float(travels[i]) * 1e-3, seed=seed)
        solved[i] = p
        seed = (p.theta, p.phi, p.psi) if abs(travels[i]) < 1e-9 else seed
    poses = [solved[i] for i in range(len(travels))]
    # the sweep's own derivative: motion ratio from the spring length, wheel rate from it and the spring
    travel = travels * 1e-3
    spring = np.array([p.spring_length_m for p in poses])
    mr = np.abs(np.gradient(spring, travel))
    wheel_rate = spec.spring_rate_n_per_m * mr ** 2
    ride_rate = wheel_rate * spec.tyre_rate_n_per_m / (wheel_rate + spec.tyre_rate_n_per_m)
    worst = max(float(np.max(np.abs(_residual(_geometry(spec), p.travel_m, (p.theta, p.phi, p.psi),
                                                       _offset(None))))) for p in poses)
    return Sweep(
        travel_mm=travels,
        camber_deg=np.array([p.camber_deg for p in poses]),
        toe_deg=np.array([p.toe_deg for p in poses]),
        caster_deg=np.array([p.caster_deg for p in poses]),
        kpi_deg=np.array([p.kpi_deg for p in poses]),
        scrub_m=np.array([p.scrub_radius_m for p in poses]),
        trail_m=np.array([p.trail_m for p in poses]),
        half_track_m=np.array([p.half_track_m for p in poses]),
        motion_ratio=mr,
        wheel_rate_n_per_m=wheel_rate,
        ride_rate_n_per_m=ride_rate,
        roll_centre_m=np.array([roll_centre_height_m(p) for p in poses]),
        spring_length_m=spring,
        residual_m=worst,
    )
