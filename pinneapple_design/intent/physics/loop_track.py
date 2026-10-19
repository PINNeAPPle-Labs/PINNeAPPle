"""Loop-the-loop track for a small car: the track curve, a 1-D model of the car along it, the closed form, and the
Newton rigid-body scene of the same track.

The track lies in the x-z plane (z up), the car runs in +x and turns counter-clockwise through the loop. It is made
of five pieces: a launch run, an entry clothoid (curvature rising linearly from 0 to 1/R), a circular arc of radius R,
an exit clothoid (1/R back to 0) and a run-out. With clothoid length 0 the loop is a circle. In 3-D the loop is a
helix: the exit is shifted sideways by ``pitch`` (a smoothstep, no sideways kink) so that it clears the entry, the way
real toy tracks are built.

Three models of the same car on the same track, from cheap to expensive:

``closed_form``
    Energy balance with the loss terms evaluated at a mean speed, the top-of-loop condition
    ``v_top^2 >= g R`` (normal force >= 0) and the peak load at the end of the entry clothoid.
``along_track``
    The car as a point on its centre-of-mass path (the track offset by the CoM height ``h``), with the normal force
    ``N = m (v^2 kappa + g cos theta)``, rolling resistance ``C_rr N``, aerodynamic drag ``0.5 rho C_d A v^2`` and the
    rotating wheels as an added inertia. ``d(v^2)/ds`` is integrated with fourth-order Runge-Kutta, vectorised over
    many designs.
``newton_check``
    The car as a rigid sphere of the wheel radius in contact with the 3-D track mesh (strip and side rails, the
    helical exit), integrated by NVIDIA Newton (XPBD). It checks the track geometry, the contact and the loop
    dynamics in 3-D; Newton has no air, and its contact dissipates a few per cent of the energy on the curved mesh,
    so the run is judged against the worst-case envelope of the 1-D model rather than for equality.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

G0 = 9.80665     # standard gravity [m/s^2] (3rd CGPM, 1901)


# ============================================================================================ track curve
@dataclass(frozen=True)
class LoopTrack:
    radius: float                 # radius of the circular arc at the top [m]
    clothoid: float = 0.0         # length of each clothoid (entry and exit) [m]; 0 = circular loop
    run_in: float = 0.3           # straight before the loop [m]
    run_out: float = 0.3          # straight after the loop [m]
    width: float = 0.04           # running surface width [m]
    pitch: float = 0.06           # sideways shift of the exit relative to the entry [m]

    def __post_init__(self):
        if self.radius <= 0 or self.clothoid < 0 or self.run_in < 0 or self.run_out < 0:
            raise ValueError("radius must be > 0 and lengths >= 0")
        if self.turn_clothoid() >= math.pi:
            raise ValueError("the clothoids turn the track by more than 360 degrees: shorten them or enlarge R")

    def turn_clothoid(self) -> float:
        """Heading change of one clothoid [rad] (curvature 0 -> 1/R over its length)."""
        return 0.5 * self.clothoid / self.radius

    def arc_length(self) -> float:
        return self.radius * (2 * math.pi - 2 * self.turn_clothoid())

    def pieces(self) -> list[tuple[str, float]]:
        return [("run_in", self.run_in), ("entry", self.clothoid), ("arc", self.arc_length()),
                ("exit", self.clothoid), ("run_out", self.run_out)]

    def length(self) -> float:
        return float(sum(L for _, L in self.pieces()))

    def curvature(self, s: np.ndarray) -> np.ndarray:
        """Signed curvature [1/m] at arc length ``s`` of the running surface."""
        s = np.asarray(s, float)
        k0, Lc = 1.0 / self.radius, self.clothoid
        a = self.run_in
        b = a + Lc
        c = b + self.arc_length()
        d = c + Lc
        out = np.zeros_like(s)
        if Lc > 0:
            out = np.where((s >= a) & (s < b), k0 * (s - a) / Lc, out)
            out = np.where((s >= c) & (s < d), k0 * (d - s) / Lc, out)
        return np.where((s >= b) & (s < c), k0, out)

    def loop_span(self) -> tuple[float, float]:
        """Arc-length interval of the loop proper (entry clothoid start to exit clothoid end)."""
        return self.run_in, self.run_in + 2 * self.clothoid + self.arc_length()

    def sample(self, n: int = 2000) -> dict[str, np.ndarray]:
        """The running surface at ``n`` points: ``s, x, y, z, theta`` (heading from +x, rad), ``kappa``,
        ``nx, nz`` (unit normal pointing to the car's side). Integrated with the trapezoid rule on a fine grid."""
        L = self.length()
        s = np.linspace(0.0, L, n)
        # heading is the integral of curvature: exact piecewise (linear / constant curvature)
        k = self.curvature(s)
        theta = np.concatenate([[0.0], np.cumsum(0.5 * (k[1:] + k[:-1]) * np.diff(s))])
        cx, cz = np.cos(theta), np.sin(theta)
        x = np.concatenate([[0.0], np.cumsum(0.5 * (cx[1:] + cx[:-1]) * np.diff(s))])
        z = np.concatenate([[0.0], np.cumsum(0.5 * (cz[1:] + cz[:-1]) * np.diff(s))])
        s0, s1 = self.loop_span()
        u = np.clip((s - s0) / max(s1 - s0, 1e-12), 0.0, 1.0)
        y = self.pitch * u * u * (3 - 2 * u)          # smoothstep: no sideways kink where the loop starts and ends
        return {"s": s, "x": x, "y": y, "z": z, "theta": theta, "kappa": k, "nx": -np.sin(theta),
                "nz": np.cos(theta)}

    def height(self) -> float:
        p = self.sample(1500)
        return float(p["z"].max())

    def footprint(self) -> float:
        """Length along x of the whole track [m]."""
        p = self.sample(1500)
        return float(p["x"].max() - p["x"].min())


# ============================================================================================ 1-D model
@dataclass
class Run:
    """Result of the 1-D model for a batch of designs (arrays of shape ``(n_designs,)``) plus the profiles of the
    first design."""
    completes: np.ndarray          # True when the car keeps contact and speed through the loop and reaches the end
    v_top: np.ndarray              # speed at the highest point [m/s]
    v_exit: np.ndarray             # speed at the end of the run-out [m/s] (0 if it did not get there)
    n_min: np.ndarray              # minimum of N / (m g) over the loop (< 0: contact lost)
    n_max: np.ndarray              # peak N / (m g) (g-load on the car, contact load on the track)
    top_margin: np.ndarray         # v_top^2 / (g R_com) - 1 (>= 0 needed: closed-form condition at the top)
    energy_lost: np.ndarray        # fraction of the launch kinetic energy dissipated up to the exit (or the stop)
    s_fail: np.ndarray             # arc length where contact or speed was lost [m] (nan if it completes)
    profile: dict = field(default_factory=dict)


def _broadcast(*arrs):
    arrs = [np.atleast_1d(np.asarray(a, float)) for a in arrs]
    n = max(len(a) for a in arrs)
    return [np.broadcast_to(a, (n,)).copy() for a in arrs]


def along_track(radius, clothoid, v0, mass, *, crr=0.02, cd=0.6, area=6e-4, com_height=0.008, wheel_inertia=0.0,
                rho=1.204, g=G0, run_in=0.3, run_out=0.3, n=1200) -> Run:
    """Integrate ``d(v^2)/ds`` along the centre-of-mass path for a batch of designs (all arguments broadcast).

    ``v0`` is the speed at the start of the launch run. The car fails when ``v^2`` reaches 0 (it stops and rolls
    back) or when the normal force becomes negative (it leaves the track). Losses: ``C_rr max(N, 0)`` and
    ``0.5 rho C_d A v^2``; ``wheel_inertia`` adds ``sum I/r^2`` to the inertia (not to the weight).
    """
    R, Lc, v0, m, crr, cd, A, h, mw = _broadcast(radius, clothoid, v0, mass, crr, cd, area, com_height,
                                                 wheel_inertia)
    nd = len(R)
    k0 = 1.0 / R
    th_c = 0.5 * Lc / R
    L_arc = R * (2 * np.pi - 2 * th_c)
    L = run_in + 2 * Lc + L_arc + run_out
    # surface arc length grid per design, shape (nd, n)
    u = np.linspace(0.0, 1.0, n)
    s = L[:, None] * u[None, :]
    a, b = run_in, run_in + Lc
    c, d = b + L_arc, b + L_arc + Lc
    a = np.broadcast_to(np.asarray(a, float), (nd,))[:, None]
    b, c, d = b[:, None], c[:, None], d[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        kin = np.where(Lc[:, None] > 0, k0[:, None] * (s - a) / Lc[:, None], 0.0)
        kout = np.where(Lc[:, None] > 0, k0[:, None] * (d - s) / Lc[:, None], 0.0)
    kap = np.where((s >= a) & (s < b), kin, 0.0)
    kap = np.where((s >= b) & (s < c), k0[:, None], kap)
    kap = np.where((s >= c) & (s < d), kout, kap)
    ds = np.diff(s, axis=1)
    theta = np.concatenate([np.zeros((nd, 1)), np.cumsum(0.5 * (kap[:, 1:] + kap[:, :-1]) * ds, axis=1)], axis=1)
    # centre-of-mass path: offset by h along the inner normal; ds_c = (1 - h kappa) ds, kappa_c = kappa / (1 - h kappa)
    fac = 1.0 - h[:, None] * kap
    if np.any(fac <= 0):
        raise ValueError("centre of mass higher than the loop radius")
    kap_c = kap / fac
    dsc = 0.5 * (fac[:, 1:] + fac[:, :-1]) * ds
    m_eff = m + mw
    q = 0.5 * rho * cd * A

    def rhs(E, th, kc):
        N = m * (E * kc + g * np.cos(th))
        return (-m * g * np.sin(th) - crr * np.maximum(N, 0.0) - q * E) * 2.0 / m_eff, N

    E = v0 ** 2 + 0.0
    alive = np.ones(nd, bool)
    s_fail = np.full(nd, np.nan)
    n_min = np.full(nd, np.inf)
    n_max = np.full(nd, -np.inf)
    zc = np.zeros(nd)
    z_top = np.full(nd, -np.inf)
    v_top = np.zeros(nd)
    keep = nd == 1
    prof_E, prof_N, prof_z = ([E[0]], [], [0.0]) if keep else ([], [], [])
    s0, s1 = run_in, run_in + 2 * Lc + L_arc
    for j in range(n - 1):
        h_ = dsc[:, j]
        thm = 0.5 * (theta[:, j] + theta[:, j + 1])
        km = 0.5 * (kap_c[:, j] + kap_c[:, j + 1])
        k1, N1 = rhs(E, theta[:, j], kap_c[:, j])
        k2, _ = rhs(E + 0.5 * h_ * k1, thm, km)
        k3, _ = rhs(E + 0.5 * h_ * k2, thm, km)
        k4, _ = rhs(E + h_ * k3, theta[:, j + 1], kap_c[:, j + 1])
        En = E + h_ / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        Nn = m * (En * kap_c[:, j + 1] + g * np.cos(theta[:, j + 1]))
        in_loop = (s[:, j + 1] >= s0) & (s[:, j + 1] <= s1[:])
        nn = Nn / (m * g)
        fail_now = alive & ((En <= 0) | (in_loop & (nn < 0)))
        s_fail = np.where(fail_now, s[:, j + 1], s_fail)
        alive &= ~fail_now
        n_min = np.where(alive & in_loop, np.minimum(n_min, nn), n_min)
        n_min = np.where(fail_now, np.minimum(n_min, nn), n_min)
        n_max = np.where(alive, np.maximum(n_max, nn), n_max)
        zc = zc + 0.5 * (np.sin(theta[:, j]) + np.sin(theta[:, j + 1])) * dsc[:, j]
        top = alive & (zc > z_top)
        z_top = np.where(top, zc, z_top)
        v_top = np.where(top, np.sqrt(np.maximum(En, 0)), v_top)
        E = np.where(alive, En, E)
        if keep:
            prof_E.append(float(En[0]))
            prof_N.append(float(nn[0]))
            prof_z.append(float(zc[0]))
    R_com = R - h
    top_margin = v_top ** 2 / (g * R_com) - 1.0
    v_exit = np.where(alive, np.sqrt(np.maximum(E, 0)), 0.0)
    lost = 1.0 - np.where(alive, (m_eff * E + 0.0) / (m_eff * v0 ** 2), 0.0)
    prof = {}
    if keep:
        prof = {"s": s[0], "v": np.sqrt(np.maximum(prof_E, 0)), "n": np.array([prof_N[0]] + prof_N),
                "z": np.array(prof_z), "theta": theta[0]}
    return Run(alive, v_top, v_exit, n_min, n_max, top_margin, lost, s_fail, prof)


def closed_form(radius, clothoid, v0, mass, *, crr=0.02, cd=0.6, area=6e-4, com_height=0.008, wheel_inertia=0.0,
                rho=1.204, g=G0, run_in=0.3) -> dict[str, np.ndarray]:
    """Screening estimate (vectorised): energy to the top with the losses at the mean speed, the top condition and
    the peak load.

    The top of the loop is at height ``z_top`` of the centre-of-mass path (``2 R_c`` for a circular loop; the
    clothoids raise it, computed from their Fresnel integrals). Losses from the start to the top: rolling
    ``C_rr m g`` on the run-in plus ``C_rr m v_m^2 theta_top`` in the turn (``int N ds ~ m v^2 dtheta`` with the
    gravity terms cancelling over the half loop), and drag ``q v_m^2 s_top`` with ``v_m^2`` the mean of the launch
    and top values (one fixed-point pass).
    """
    R, Lc, v0, m, crr, cd, A, h, mw = _broadcast(radius, clothoid, v0, mass, crr, cd, area, com_height,
                                                 wheel_inertia)
    th_c = 0.5 * Lc / R
    # height of the arc's top: entry clothoid rise + arc rise from heading th_c to pi
    z_cl, x_cl = _clothoid_rise(R, Lc)
    z_top_surface = z_cl + R * (1 + np.cos(th_c))
    R_c = R - h
    z_top = z_top_surface - 2 * h        # CoM: +h at the start (normal up), -h at the top (normal down)
    s_top = run_in + Lc + R * (np.pi - th_c)
    m_eff = m + mw
    q = 0.5 * rho * cd * A
    v_top2 = v0 ** 2 - 2 * m * g * z_top / m_eff
    for _ in range(2):
        vm2 = np.maximum(0.5 * (v0 ** 2 + np.maximum(v_top2, 0)), 0)
        loss = crr * m * g * run_in + crr * m * vm2 * np.pi + q * vm2 * s_top
        v_top2 = v0 ** 2 - (2 * m * g * z_top + 2 * loss) / m_eff
    top_margin = v_top2 / (g * R_c) - 1.0
    # peak load: at the end of the entry clothoid (heading th_c, curvature 1/R), speed from the energy balance
    z_e = z_cl + h * (1 - np.cos(th_c))
    loss_e = crr * m * g * (run_in + Lc) + q * v0 ** 2 * (run_in + Lc)
    v_e2 = np.maximum(v0 ** 2 - (2 * m * g * z_e + 2 * loss_e) / m_eff, 0)
    n_peak = np.maximum(v_e2 / (g * R_c) + np.cos(th_c), 1.0)
    return {"v_top": np.sqrt(np.maximum(v_top2, 0)), "top_margin": top_margin, "n_max": n_peak,
            "completes": top_margin >= 0, "z_top": z_top_surface}


def _clothoid_rise(R, Lc, n: int = 64):
    """Height and length gained along a clothoid of length ``Lc`` that ends with curvature 1/R (Fresnel integrals
    by Gauss-Legendre quadrature)."""
    R, Lc = np.broadcast_arrays(np.asarray(R, float), np.asarray(Lc, float))
    xg, wg = np.polynomial.legendre.leggauss(n)
    t = 0.5 * (xg + 1)                               # s / Lc in (0, 1)
    th = 0.5 * Lc[..., None] / R[..., None] * t ** 2
    z = 0.5 * Lc * np.sum(wg * np.sin(th), axis=-1)
    x = 0.5 * Lc * np.sum(wg * np.cos(th), axis=-1)
    return z, x


# ============================================================================================ 3-D geometry
def _frames(track: LoopTrack, n: int):
    p = track.sample(n)
    P = np.c_[p["x"], p["y"], p["z"]]
    T = np.c_[np.cos(p["theta"]), np.gradient(p["y"], p["s"]), np.sin(p["theta"])]
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    Nn = np.c_[p["nx"], np.zeros(len(P)), p["nz"]]
    Nn -= (Nn * T).sum(1, keepdims=True) * T
    Nn /= np.linalg.norm(Nn, axis=1, keepdims=True)
    B = np.cross(T, Nn)
    return P, T, Nn, B


def _sweep(P, Nn, B, rect) -> tuple[np.ndarray, np.ndarray]:
    """Sweep a rectangle ``(b0, b1, n0, n1)`` (offsets along the binormal and the normal) along the path."""
    from pinneapple_design.geometry_library.kit import loft
    b0, b1, n0, n1 = rect
    corners = [(b0, n0), (b1, n0), (b1, n1), (b0, n1)]
    rings = np.stack([P + bb * B + nn * Nn for bb, nn in corners], axis=1)
    return loft(rings)


def track_meshes(track: LoopTrack, thickness: float = 0.003, rail: float = 0.008, n: int = 600):
    """Closed meshes of the running strip and the two side rails, in metres: ``{"strip": (V, F), "rail_l": ...}``.
    The strip lies below the running surface (normal offset ``-thickness``..0)."""
    P, T, Nn, B = _frames(track, n)
    w = track.width / 2
    t2 = thickness
    return {"strip": _sweep(P, Nn, B, (-w - t2, w + t2, -t2, 0.0)),
            "rail_l": _sweep(P, Nn, B, (w, w + t2, 0.0, rail)),
            "rail_r": _sweep(P, Nn, B, (-w - t2, -w, 0.0, rail))}


def car_mesh(length: float, width: float, height: float, wheel_r: float):
    """A simple car for drawings and files: rounded body plus four wheels, resting on z = 0, nose along +x."""
    from pinneapple_design.geometry_library import kit
    body = kit.rounded_box((length, width * 0.8, height - wheel_r), center=(0, 0, wheel_r + (height - wheel_r) / 2),
                           radius=min(length, width, height) * 0.15)
    wheels = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            c = np.array([sx * 0.32 * length, sy * width * 0.42, wheel_r])
            wheels.append(kit.cylinder(c - [0, 0.08 * width, 0], c + [0, 0.08 * width, 0], wheel_r, n=24))
    return body, wheels


# ============================================================================================ Newton
@dataclass
class NewtonLoopResult:
    completes: bool               # reached the end of the run-out without leaving the track
    v_top: float                  # speed at the highest point of the loop [m/s]
    max_gap: float                # largest gap between the car and the running surface inside the loop [m]
    s_reached: float              # arc length reached along the track [m]
    t: np.ndarray
    pos: np.ndarray               # (n_frames, 3) car centre
    vel: np.ndarray
    s: np.ndarray                 # arc length of the nearest track point for each frame [m]
    wall_s: float
    solver: str
    dt: float
    meta: dict = field(default_factory=dict)


def newton_scene(track: LoopTrack, mass: float, v0: float, *, radius: float = 0.006, mu: float = 0.0,
                 thickness: float = 0.003, rail: float = 0.008, n: int = 900):
    """The track (one static triangle mesh: strip + rails) and the car as a sphere of the wheel radius resting on the
    running surface, launched along +x, as a :class:`~pinneapple_simulation.external_solvers.newton_physics.
    RigidScene`."""
    from pinneapple_design.geometry_library.meshops import concat
    from pinneapple_simulation.external_solvers.newton_physics import RigidScene

    V, F = concat(list(track_meshes(track, thickness=thickness, rail=rail, n=n).values()))
    sc = RigidScene(gravity=G0)
    tid = sc.mesh("track", V, F, mu=mu, density=1000.0)
    sc.fixed(-1, tid, anchor=(0.0, 0.0, 0.0))
    sc.sphere("car", radius, pos=(0.05, 0.0, radius + 1e-4), vel=(v0, 0.0, 0.0), mass=mass, mu=mu)
    return sc


def newton_check(track: LoopTrack, mass: float, v0: float, *, dt: float = 2.5e-5, duration: float | None = None,
                 iterations: int = 12, record_every: int | None = None, radius: float = 0.006,
                 mu: float = 0.0) -> NewtonLoopResult:
    """Run the car through the loop in NVIDIA Newton (XPBD contact on the track mesh) and measure whether it stays
    on the track and reaches the end, the speed at the top and the largest gap to the running surface in the loop."""
    from pinneapple_simulation.external_solvers.newton_physics import simulate

    sc = newton_scene(track, mass, v0, radius=radius, mu=mu)
    if duration is None:
        duration = 1.6 * track.length() / max(v0 * 0.6, 0.3)
    rec = record_every or max(1, int(round(2e-3 / dt)))
    traj = simulate(sc, duration, dt=dt, solver="xpbd", iterations=iterations, record_every=rec)
    k = traj.index("car")
    pos, vel = traj.pos[:, k], traj.vel[:, k]
    p = track.sample(4000)
    P = np.c_[p["x"], p["y"], p["z"]]
    Nrm = np.c_[p["nx"], np.zeros(len(P)), p["nz"]]
    idx, s_tr, gaps = 0, [], []
    for x in pos:
        lo, hi = max(idx - 40, 0), min(idx + 400, len(P))
        j = lo + int(np.argmin(np.linalg.norm(P[lo:hi] - x, axis=1)))
        idx = max(idx, j)
        s_tr.append(p["s"][j])
        gaps.append(float((x - P[j]) @ Nrm[j]) - radius)
    s_tr, gaps = np.asarray(s_tr), np.asarray(gaps)
    s0, s1 = track.loop_span()
    end = p["s"][-1]
    on_track = s_tr < end - 0.02                      # frames before the car leaves the end of the run-out
    in_loop = (s_tr > s0) & (s_tr < s1)
    max_gap = float(np.abs(gaps[in_loop]).max()) if in_loop.any() else float("nan")
    left = bool((np.abs(gaps[on_track]) > 2 * radius).any())
    reached = float(s_tr.max())
    completes = (reached >= end - 0.05) and not left
    v_top = float(np.linalg.norm(vel[in_loop][np.argmax(pos[in_loop, 2])])) if in_loop.any() else 0.0
    return NewtonLoopResult(completes, v_top, max_gap, reached, traj.t, pos, vel, s_tr, traj.wall_time_s, "xpbd", dt,
                            {"newton_version": traj.meta.get("newton_version"), "iterations": iterations,
                             "left_track": left, "mu": mu})
