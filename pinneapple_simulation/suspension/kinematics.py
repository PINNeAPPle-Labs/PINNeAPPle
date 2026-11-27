"""Rigid kinematics of the double wishbone: the pose of the upright at a given wheel travel, and every alignment
number that follows from it.

The mechanism. The lower and upper wishbones swing about fixed chassis axes, so their ball joints trace circles; the
tie rod holds its length; and the upright is one rigid body carrying both ball joints, the outer tie-rod end and the
wheel centre. Six degrees of freedom, five constraints -- lower circle (2), upper circle (2), tie-rod length (1) --
leaves exactly one: the wheel travel. Prescribing it (:func:`solve_pose`) fixes the corner.

Two independent solvers compute that pose, and they are expected to agree:

* :func:`solve_pose` -- Newton on the five constraint residuals, unknown the pose of the upright (a lateral shift
  and a rotation vector, with the vertical travel prescribed);
* :func:`solve_pose_angles` -- Newton on three joint angles instead (one per wishbone, one spin about the kingpin),
  with the rigid-body length between the ball joints, the tie-rod length and the travel as the three equations.

Both are checked in :mod:`pinneapple_simulation.suspension.verification` against cases with a closed-form answer: a
parallelogram linkage (equal, parallel wishbones) has zero camber gain, a tie rod pinned at the lower ball joint has
zero bump steer, and bump and rebound are antisymmetric about the static pose.

Sign conventions, for the right-hand corner (``+y`` outboard): camber is negative when the top of the wheel leans
inboard (SAE); toe is positive when the wheel points inboard (toe-in); the roll centre height is measured from the
ground plane; the scrub radius is positive when the steering axis meets the ground inboard of the tyre's contact
centre.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .hardpoints import Corner, corner

__all__ = ["Pose", "solve_pose", "solve_pose_angles", "metrics", "sweep", "camber_deg", "toe_deg"]

_TOL = 1e-12          # m, the constraint residual a pose has to reach
_MAXITER = 80


# ------------------------------------------------------------------------------------------------ rotations
def _rodrigues(w: np.ndarray) -> np.ndarray:
    """Rotation matrix of a rotation vector (axis * angle)."""
    th = float(np.linalg.norm(w))
    if th < 1e-15:
        return np.eye(3)
    k = w / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def _rot_about(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, float)
    return _rodrigues(axis / np.linalg.norm(axis) * angle)


def _log_so3(R: np.ndarray) -> np.ndarray:
    """The rotation vector of a rotation matrix (inverse of :func:`_rodrigues`)."""
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    th = math.acos(c)
    if th < 1e-14:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return w * th


def _min_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The shortest rotation taking unit vector ``a`` onto unit vector ``b``."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    s = float(np.linalg.norm(v))
    if s < 1e-14:
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-9:
            axis = np.cross(a, np.array([0.0, 0.0, 1.0]))
        return _rot_about(axis, math.pi)
    return _rot_about(v / s, math.atan2(s, c))


def _newton(f, u0: np.ndarray, tol: float = _TOL, maxiter: int = _MAXITER, h: float = 1e-8):
    """Newton with a central-difference Jacobian and a halving line search on the residual norm."""
    u = np.asarray(u0, float).copy()
    r = f(u)
    it = 0
    for it in range(1, maxiter + 1):
        n = float(np.linalg.norm(r))
        if n < tol:
            return u, n, it
        J = np.empty((len(np.atleast_1d(r)), len(u)))
        for j in range(len(u)):
            du = np.zeros_like(u)
            du[j] = h
            J[:, j] = (f(u + du) - f(u - du)) / (2 * h)
        try:
            step = np.linalg.solve(J, -np.atleast_1d(r))
        except np.linalg.LinAlgError:                                   # pragma: no cover - degenerate geometry
            step = np.linalg.lstsq(J, -np.atleast_1d(r), rcond=None)[0]
        a, accepted = 1.0, False
        for _ in range(40):
            r_new = np.atleast_1d(f(u + a * step))
            if np.linalg.norm(r_new) < n:
                u, r, accepted = u + a * step, r_new, True
                break
            a *= 0.5
        if not accepted:                                                # pragma: no cover - stuck at the solution
            break
    return u, float(np.linalg.norm(np.atleast_1d(r))), it


# ------------------------------------------------------------------------------------------------ circles
def _circle(q: np.ndarray, a: np.ndarray, p0: np.ndarray):
    """The circle a body point traces about the axis (``q``, ``a``): (centre, e1, e2, radius)."""
    v = p0 - q
    s = float(v @ a)
    perp = v - s * a
    rho = float(np.linalg.norm(perp))
    e1 = perp / rho
    return q + s * a, e1, np.cross(a, e1), rho


def _arm_point(q: np.ndarray, a: np.ndarray, p0: np.ndarray, ball0: np.ndarray, ball: np.ndarray) -> np.ndarray:
    """Where a point of a wishbone ends up: the arm turns about its pivot axis by the angle its ball joint turned."""
    w0 = ball0 - q
    w1 = ball - q
    u0 = w0 - (w0 @ a) * a
    u1 = w1 - (w1 @ a) * a
    ang = math.atan2(float(np.cross(u0, u1) @ a), float(u0 @ u1))
    v = p0 - q
    return q + (v @ a) * a + _rot_about(a, ang) @ (v - (v @ a) * a)


# ------------------------------------------------------------------------------------------------ the pose
@dataclass(frozen=True)
class Pose:
    """The upright's pose at one wheel travel: ``X = wc0 + t + R (X0 - wc0)`` for every point of the body."""
    travel: float
    R: np.ndarray
    t: np.ndarray
    points: dict[str, np.ndarray]
    damper_lower: np.ndarray
    residual: float = 0.0
    iterations: int = 0
    solver: str = "constraints"

    def camber_deg(self) -> float:
        return camber_deg(self.R)

    def toe_deg(self) -> float:
        return toe_deg(self.R)

    @property
    def wheel_centre(self) -> np.ndarray:
        return self.points["wheel_centre"]

    def damper_length(self, upper: np.ndarray) -> float:
        return float(np.linalg.norm(upper - self.damper_lower))

    def kingpin(self) -> tuple[np.ndarray, np.ndarray]:
        """(foot point, unit direction) of the steering axis, from the lower to the upper ball joint."""
        d = self.points["ubj"] - self.points["lbj"]
        return self.points["lbj"], d / np.linalg.norm(d)

    def constrain(self) -> np.ndarray:
        """The Newton unknown vector of this pose, to warm-start the next travel (continuation)."""
        return np.r_[self.t[0], self.t[1], _log_so3(self.R)]


def camber_deg(R: np.ndarray) -> float:
    """Camber of the right-hand wheel: the upright's up direction against vertical, negative when it leans inboard."""
    up = R @ np.array([0.0, 0.0, 1.0])
    return math.degrees(math.asin(float(np.clip(up[1] / np.linalg.norm(up), -1.0, 1.0))))


def toe_deg(R: np.ndarray) -> float:
    """Toe of the right-hand wheel: its forward direction against the vehicle's x axis, positive for toe-in."""
    fwd = R @ np.array([1.0, 0.0, 0.0])
    return -math.degrees(math.atan2(float(fwd[1]), float(fwd[0])))


def _pose_from(c: Corner, R: np.ndarray, t: np.ndarray, travel: float) -> Pose:
    wc0 = c.wheel_centre
    pts = {k: wc0 + t + R @ (v - wc0) for k, v in c.points().items()}
    q, a = c.lca_axis
    dmp = _arm_point(q, a, c.damper_lower, c.lbj, pts["lbj"])
    return Pose(travel, R, t, pts, dmp)


def _residuals(c: Corner, travel: float):
    wc0 = c.wheel_centre
    pts = c.points()
    radii = {}
    for ball, (q, a) in (("lbj", c.lca_axis), ("ubj", c.uca_axis)):
        v = pts[ball] - q
        radii[ball] = (np.linalg.norm(v - (v @ a) * a), float(v @ a))
    lt = float(np.linalg.norm(pts["tie_outer"] - c.tie_inner))

    def f(u: np.ndarray) -> np.ndarray:
        R = _rodrigues(u[2:5])
        t = np.array([u[0], u[1], travel])
        P = {k: wc0 + t + R @ (v - wc0) for k, v in pts.items()}
        out = []
        for ball, (q, a) in (("lbj", c.lca_axis), ("ubj", c.uca_axis)):
            d = P[ball] - q
            perp = d - (d @ a) * a
            out.append(np.linalg.norm(perp) - radii[ball][0])
            out.append(d @ a - radii[ball][1])
        out.append(np.linalg.norm(P["tie_outer"] - c.tie_inner) - lt)
        return np.array(out, float)
    return f


def solve_pose(c: Corner, travel: float, guess=None) -> Pose:
    """The upright's pose with the wheel centre displaced by ``travel`` metres (positive = bump).

    Unknown: the lateral shift of the wheel centre and the two free rotations of the upright -- five unknowns for the
    five constraints. ``guess``: the pose of the neighbouring travel (:meth:`Pose.constrain`), which is what makes a
    sweep cheap: each step starts from the solution of the one before it.
    """
    u0 = np.zeros(5) if guess is None else np.asarray(guess, float)
    u, res, it = _newton(_residuals(c, travel), u0)
    R = _rodrigues(u[2:5])
    t = np.array([u[0], u[1], travel])
    p = _pose_from(c, R, t, travel)
    return Pose(travel, R, t, p.points, p.damper_lower, res, it, "constraints")


def solve_pose_angles(c: Corner, travel: float, guess=None) -> Pose:
    """The same pose from three joint angles: one on each wishbone's circle, one spin about the kingpin.

    Equations: the rigid distance between the ball joints, the tie-rod length, and the wheel travel. A different
    unknown set and a different decomposition from :func:`solve_pose` -- which is why the two agreeing is evidence
    and not a tautology.
    """
    wc0 = c.wheel_centre
    pts = c.points()
    cL = _circle(*c.lca_axis, pts["lbj"])
    cU = _circle(*c.uca_axis, pts["ubj"])
    d0 = float(np.linalg.norm(pts["ubj"] - pts["lbj"]))
    kp0 = (pts["ubj"] - pts["lbj"]) / d0
    lt = float(np.linalg.norm(pts["tie_outer"] - c.tie_inner))

    def ball(circ, phi: float) -> np.ndarray:
        centre, e1, e2, rho = circ
        return centre + rho * (math.cos(phi) * e1 + math.sin(phi) * e2)

    def pose(u: np.ndarray) -> Pose:
        lbj, ubj = ball(cL, u[0]), ball(cU, u[1])
        kp = ubj - lbj
        R = _rot_about(kp, u[2]) @ _min_rotation(kp0, kp)
        t = lbj - wc0 - R @ (pts["lbj"] - wc0)
        return _pose_from(c, R, t, float(t[2]))

    def f(u: np.ndarray) -> np.ndarray:
        lbj, ubj = ball(cL, u[0]), ball(cU, u[1])
        p = pose(u)
        return np.array([np.linalg.norm(ubj - lbj) - d0,      # the two ball joints are d0 apart on one rigid upright
                         np.linalg.norm(p.points["tie_outer"] - c.tie_inner) - lt,
                         float(p.points["wheel_centre"][2] - wc0[2]) - travel])
    u, res, it = _newton(f, np.zeros(3) if guess is None else np.asarray(guess, float))
    p = pose(u)
    return Pose(p.travel, p.R, p.t, p.points, p.damper_lower, res, it, "angles")


# ------------------------------------------------------------------------------------------------ metrics
def _front_view_centre(low_in: np.ndarray, low_ball: np.ndarray, up_in: np.ndarray,
                       up_ball: np.ndarray) -> tuple[float, float] | None:
    """Instantaneous centre in the y-z plane of the two wishbones (None when they are parallel)."""
    p1, p2 = np.array([low_in[1], low_in[2]]), np.array([low_ball[1], low_ball[2]])
    p3, p4 = np.array([up_in[1], up_in[2]]), np.array([up_ball[1], up_ball[2]])
    d1, d2 = p2 - p1, p4 - p3
    den = float(d1[0] * d2[1] - d1[1] * d2[0])
    if abs(den) < 1e-12:
        return None
    s = float((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / den
    return float(p1[0] + s * d1[0]), float(p1[1] + s * d1[1])


def metrics(c: Corner) -> dict[str, float]:
    """Every number the study reports for one design: static alignment, travel gains, the geometry that names the
    handling behaviour, and the motion ratio the spring sees.

    ``camber_gain_deg_per_25mm`` and ``bump_steer_deg_per_25mm`` are measured between the static pose and 25 mm of
    bump -- the convention of a wishbone design sheet. The roll centre is the front-view construction: the contact
    patch through the wishbones' instantaneous centre, meet the vehicle centreline.
    """
    static = solve_pose(c, 0.0)
    bump = solve_pose(c, 0.025, static.constrain())
    rebound = solve_pose(c, -0.025, static.constrain())
    up = static.points

    kp = up["ubj"] - up["lbj"]
    kpi = math.degrees(math.atan2(-float(kp[1]), float(kp[2])))
    caster = math.degrees(math.atan2(-float(kp[0]), float(kp[2])))
    ground = up["lbj"] - kp * (float(up["lbj"][2]) / float(kp[2]))
    scrub = (float(static.wheel_centre[1]) - float(ground[1])) * 1e3
    y_wc = float(static.wheel_centre[1])

    ic = _front_view_centre(0.5 * (c.lca_front + c.lca_rear), up["lbj"],
                            0.5 * (c.uca_front + c.uca_rear), up["ubj"])
    if ic is None or abs(ic[0] - y_wc) < 1e-9:
        roll_centre = 0.0
    else:
        roll_centre = ic[1] * (0.0 - y_wc) / (ic[0] - y_wc) * 1e3

    h = 1e-4
    dp = solve_pose(c, h, static.constrain()).damper_length(c.damper_upper)
    dm = solve_pose(c, -h, static.constrain()).damper_length(c.damper_upper)
    mr = abs(dp - dm) / (2 * h)

    return {
        "camber_static_deg": static.camber_deg(),
        "toe_static_deg": static.toe_deg(),
        "kpi_deg": kpi,
        "caster_deg": caster,
        "scrub_radius_mm": scrub,
        "roll_centre_mm": roll_centre,
        "motion_ratio": mr,
        "wheel_rate_n_per_mm": c.spring_rate * mr * mr,
        "track_change_mm": (float(bump.wheel_centre[1]) - y_wc) * 1e3,
        "camber_gain_deg_per_25mm": bump.camber_deg() - static.camber_deg(),
        "bump_steer_deg_per_25mm": bump.toe_deg() - static.toe_deg(),
        "camber_gain_rebound_deg_per_25mm": static.camber_deg() - rebound.camber_deg(),
        "bump_steer_rebound_deg_per_25mm": static.toe_deg() - rebound.toe_deg(),
        "damper_length_mm": static.damper_length(c.damper_upper) * 1e3,
        "residual_m": max(static.residual, bump.residual, rebound.residual),
    }


def sweep(c: Corner, travel_mm) -> dict[str, np.ndarray]:
    """Camber, toe, track and damper length along a travel sweep -- the curves a design sheet is drawn from."""
    out: dict[str, list[float]] = {k: [] for k in ("travel_mm", "camber_deg", "toe_deg", "track_mm", "damper_mm")}
    guess = None
    for tr in travel_mm:
        p = solve_pose(c, float(tr) / 1e3, guess)
        guess = p.constrain()
        out["travel_mm"].append(float(tr))
        out["camber_deg"].append(p.camber_deg())
        out["toe_deg"].append(p.toe_deg())
        out["track_mm"].append(float(p.wheel_centre[1]) * 1e3)
        out["damper_mm"].append(p.damper_length(c.damper_upper) * 1e3)
    return {k: np.array(v) for k, v in out.items()}


def corner_metrics(design) -> dict[str, float]:
    """``metrics`` of a design vector (dict) rather than of its :class:`Corner`."""
    return metrics(corner(design))
