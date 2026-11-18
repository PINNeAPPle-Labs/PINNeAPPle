"""A two-dimensional compact range: offset parabolic cylinder with rolled edges, ``cos^q`` line feed, physical
optics and the exact integral-equation solution of the same problem, on the same contour, with the same
quiet-zone numbers.

Why a plane model: the question these experiments ask is a question about the *difference* between physical optics
and an exact surface-current solution -- how large it is, whether it can be learned, and whether spending a few
exact solves inside an optimisation pays. That question is answered by a solver that is exact, cheap and
validatable. In two dimensions the same physics (a parabolic section, a rolled edge, an offset feed, a quiet zone
in the near field of the section) is solved by a single-layer integral equation on a contour of a few hundred
unknowns, in milliseconds, and validated against the closed-form series of a perfectly conducting circle
(:func:`validate_circle`). A three-dimensional direct solve of the production geometry is a different class of
tool: the measured cost of the operator alone at ``lambda/8`` is reported by
``examples/use_cases/antennas/direct_cost_probe.py`` and is three orders of magnitude away from PO.

Conventions: ``E = E_z z_hat`` with ``exp(+j omega t)`` (the package convention); the contour lies in the y-z plane,
the paraboloid is ``z = y^2 / (4F)`` with vertex at the origin and focus ``(0, F)``; a perfectly conducting surface
is the Dirichlet condition ``E_z = 0`` on the contour (the 2-D analogue of the 3-D ``J = 2 n x H_inc``).

Everything is per radian in 2-D, so the metrics are *relative* (taper, ripple, phase ripple, usable width): the
absolute level of a 2-D feed has no 3-D meaning and nothing here quotes it.

* :class:`Section` -- the reflector contour: an arc of the parabola between two aperture coordinates, continued
  past each end by the blended rolled edge of :func:`~.geometry._roll_profile`, sampled at a target spacing;
* :class:`Feed` -- the incident field of a directive line source at the focus: ``cos^q`` amplitude about an axis,
  ``H0^(2)(kR)`` phase, and its normal derivative (what the PO density is);
* :func:`po_metrics` / :func:`direct_metrics` -- quiet-zone metrics of the PO density and of the exact one;
* :func:`near_field_map` -- the near field both densities produce on a grid, and where they disagree;
* :func:`metrics_1d` -- the 1-D quiet-zone numbers: taper, ripple, phase ripple and usable width;
* :func:`mom_solve` -- the exact current by point-matched pulse expansion of the single-layer equation, with the
  self term integrated analytically over the log singularity;
* :func:`validate_circle` -- the exact series of a PEC circle against :func:`mom_solve`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.special import hankel2, jv

C0 = 299_792_458.0
ETA0 = 376.730313412
EULER = 0.5772156649015329

__all__ = ["Section", "Feed", "Grid", "metrics_1d", "mom_solve", "po_density", "radiate_2d", "incident",
           "po_metrics", "direct_metrics", "near_field_map", "validate_circle", "EDGE_FAMILY", "Design",
           "design_space"]

#: the 18 plausible rolled-edge profiles of the 2 GHz study: roll radius factor x blend x depth of the roll
EDGE_FAMILY = tuple((f, b, a) for f in (0.6, 1.0, 1.4) for b in (0.4, 0.7) for a in (0.7, 1.0, 1.4))


def _h0(x):
    """``H0^(2)`` of a real argument, elementwise."""
    x = np.asarray(x, float)
    return hankel2(0, x + 0j)


def _h1(x):
    x = np.asarray(x, float)
    return hankel2(1, x + 0j)


# ------------------------------------------------------------------------------------------------ geometry
def _edge_profile(theta0: float, kappa_p: float, roll: float, end_angle: float, blend: float, n_pts: int,
                  aspect: float = 1.0):
    """The rolled edge in the meridian plane of :func:`.geometry._roll_profile`: ``(du, dw)`` from the junction,
    its arc length, and the tangent angle at the last point. ``aspect`` stretches depth against width, the same
    parameter the 3-D geometry uses."""
    kr = -1.0 / roll
    end = np.deg2rad(end_angle)
    if roll <= 0 or theta0 <= end:
        return np.zeros((1, 2)), 0.0, theta0
    L_b = blend * roll * (theta0 - end)
    ds = roll * 2e-3
    pts, th, s = [np.zeros(2)], theta0, 0.0
    while th > end and s < 50 * roll:
        t = min(s / L_b, 1.0) if L_b > 0 else 1.0
        w = np.sin(0.5 * np.pi * t) ** 2
        k = (1 - w) * kappa_p + w * kr
        pts.append(pts[-1] + ds * np.array([np.cos(th), np.sin(th)]))
        th += k * ds
        s += ds
    P = np.array(pts)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if arc[-1] <= 0:
        return np.zeros((1, 2)), 0.0, theta0
    a = np.linspace(0, arc[-1], n_pts)
    du = np.interp(a, arc, P[:, 0])
    dw = np.interp(a, arc, P[:, 1])
    t0 = np.tan(theta0)
    dw = du * t0 + aspect * (dw - du * t0)
    return np.c_[du, dw], float(arc[-1]), float(th)


@dataclass
class Section:
    """The reflector contour (y, z): the parabola ``z = y^2/(4F)`` between ``y_low`` and ``y_high``, with a rolled
    edge at each end of radius ``roll``, blend fraction ``blend`` and depth ``aspect`` (the 3-D parameters)."""
    focal: float = 0.6
    y_low: float = 0.15
    y_high: float = 1.05
    roll: float = 0.15
    blend: float = 0.5
    aspect: float = 1.0
    end_angle: float = -90.0

    def contour(self, h: float) -> Grid:
        """The contour sampled at about ``h``, running from the lower rolled-edge tip to the upper one: points,
        segment centres, tangents, normals toward the illuminated (concave) side, and segment lengths."""
        F = self.focal
        ys = np.linspace(self.y_low, self.y_high, max(8, int(np.ceil((self.y_high - self.y_low) / h)) + 1))
        par = np.c_[ys, ys ** 2 / (4 * F)]
        n_roll = max(2, int(np.ceil(4 * self.roll / h)))
        tips = []
        for end, outward in ((0, -1.0), (-1, +1.0)):
            y0 = par[end, 0]
            dz = y0 / (2 * F)
            th0 = np.arctan(dz)
            kp = (1 / (2 * F)) / (1 + dz ** 2) ** 1.5
            prof, _, _ = _edge_profile(th0, kp, self.roll, self.end_angle, self.blend, n_roll, self.aspect)
            pts = par[end] + np.c_[outward * prof[:, 0], prof[:, 1]]
            tips.append(pts[1:][::-1] if end == 0 else pts[1:])
        P = np.concatenate([tips[0], par, tips[1]])
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        mid = 0.5 * (P[1:] + P[:-1])
        t = np.diff(P, axis=0) / seg[:, None]
        n = np.stack([t[:, 1], -t[:, 0]], 1)
        focus = np.array([0.0, F])
        flip = np.einsum("ij,ij->i", n, focus - mid) < 0
        n[flip] *= -1
        return Grid(P, mid, t, n, seg)


@dataclass
class Grid:
    """The sampled contour: ``points`` (n+1, 2), segment ``mid`` (n, 2), ``tangent``, ``normal`` (n, 2) toward the
    illuminated side and ``length`` (n,)."""
    points: np.ndarray
    mid: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    length: np.ndarray


# ---------------------------------------------------------------------------------------------------- feed
@dataclass
class Feed:
    """A directive line source at ``position`` pointing along ``axis``: amplitude ``cos^q psi`` inside the forward
    half space, ``H0^(2)`` phase. ``q`` comes from a feed gain the same way the 3-D code does
    (:func:`~.po.q_from_gain`) so the two share the parameter."""
    q: float = 4.42
    position: tuple[float, float] = (0.0, 0.6)
    axis: tuple[float, float] | None = None
    amp: float = 1.0

    def field(self, pts: np.ndarray) -> np.ndarray:
        P = np.asarray(pts, float)
        d = P - np.asarray(self.position, float)
        R = np.linalg.norm(d, axis=1)
        a = self._axis()
        cpsi = (d @ a) / R
        psi = np.arccos(np.clip(cpsi, -1, 1))
        patt = np.maximum(np.cos(psi), 0.0) ** self.q
        return self.amp * patt * _h0(self._k * R)

    def _axis(self):
        if self.axis is None:
            return np.array([0.0, -1.0])                  # from the feed down to the section
        v = np.asarray(self.axis, float)
        return v / np.linalg.norm(v)

    _k: float = 2 * np.pi

    def with_k(self, k: float) -> Feed:
        return replace(self, _k=k)

    def normal_derivative(self, pts: np.ndarray, normals: np.ndarray, h: float = 1e-6) -> np.ndarray:
        """``d/dn`` of the incident field, by a central difference along the normal (the amplitude and the phase
        are smooth away from the source)."""
        P = np.asarray(pts, float)
        N = np.asarray(normals, float)
        return (self.field(P + h * N) - self.field(P - h * N)) / (2 * h)


# ----------------------------------------------------------------------------------------------- operators
def radiate_2d(mid: np.ndarray, length: np.ndarray, rho: np.ndarray, obs: np.ndarray, k: float,
               *, chunk: int = 4_000_000) -> np.ndarray:
    """The field of a line density ``rho`` on the segments ``mid`` of ``length``: ``sum_i (j/4) H0^(2)(kR) rho_i
    l_i`` -- the 2-D Green's function of the package's ``exp(+j omega t)`` convention."""
    M = np.atleast_2d(np.asarray(obs, float))
    out = np.zeros(len(M), complex)
    step = max(1, chunk // max(len(mid), 1))
    w = np.asarray(rho) * np.asarray(length)
    for i in range(0, len(M), step):
        R = np.linalg.norm(M[i:i + step, None, :] - mid[None, :, :], axis=2)
        out[i:i + step] = _h0(k * np.maximum(R, 1e-300)) @ w
    return 0.25j * out


def _self_integral(k: float, length: np.ndarray) -> np.ndarray:
    """``integral over a segment of H0^(2)(k|s|) ds`` across the log singularity.

    The small-argument form ``H0^(2)(z) ~ 1 - j (2/pi)[ln(z/2) + gamma]`` is subtracted, integrated in closed form
    (``integral_0^a ln(k s / 2) ds = a (ln(k a / 2) - 1)``), and the smooth remainder is integrated with a
    Gauss-Legendre rule. Accurate for every ``k L``, including the ``k L ~ 0.5`` of a ``lambda/12`` mesh.
    """
    from numpy.polynomial.legendre import leggauss
    L = np.atleast_1d(np.asarray(length, float))
    xs, ws = leggauss(48)
    half = L / 2
    s = xs[None, :] * half[:, None]                      # (n, ng) in [-L/2, L/2]
    z = k * s
    smooth = np.zeros(len(L), complex)
    for j in range(len(L)):
        zj = z[j]
        rem = hankel2(0, np.abs(zj) + 0j) - (1 - 1j * (2 / np.pi) * (np.log(np.abs(zj) / 2 + 1e-300) + EULER))
        smooth[j] = half[j] * np.sum(ws * rem)
    a = np.maximum(half, 1e-300)
    log_part = L * (1 - 1j * (2 / np.pi) * (EULER + np.log(np.maximum(k * a / 2, 1e-300)) - 1))
    return smooth + log_part


def mom_solve(grid: Grid, feed: Feed, k: float) -> dict:
    """Solve ``integral_S (j/4) H0^(2)(kR) rho dl' = -u_inc`` on the contour, point matched at segment midpoints,
    pulse expansion. Returns the density and the cost."""
    t0 = time.perf_counter()
    n = len(grid.mid)
    A = np.zeros((n, n), complex)
    feed = feed.with_k(k)
    u_inc = feed.field(grid.mid)
    step = max(1, 2_000_000 // max(n, 1))
    for i in range(0, n, step):
        R = np.linalg.norm(grid.mid[i:i + step, None, :] - grid.mid[None, :, :], axis=2)
        A[i:i + step] = 0.25j * _h0(k * np.maximum(R, 1e-300)) * grid.length[None, :]
    diag = 0.25j * _self_integral(k, grid.length)
    np.fill_diagonal(A, diag)
    rho = np.linalg.solve(A, -u_inc)
    return {"rho": rho, "seconds": time.perf_counter() - t0, "n": n,
            "condition": float(np.linalg.cond(A)) if n <= 1200 else float("nan")}


def po_density(grid: Grid, feed: Feed, k: float) -> np.ndarray:
    """The physical-optics density: ``2 du_inc/dn`` on the segments that face the feed, zero in shadow -- the 2-D
    counterpart of ``J = 2 n x H_inc``."""
    feed = feed.with_k(k)
    lit = np.einsum("ij,ij->i", grid.normal, np.asarray(feed.position, float) - grid.mid) > 0
    rho = 2 * feed.normal_derivative(grid.mid, grid.normal, h=1e-4 / k)
    return np.where(lit, rho, 0.0)


def incident(obs: np.ndarray, feed: Feed, k: float) -> np.ndarray:
    return feed.with_k(k).field(np.atleast_2d(obs))


# -------------------------------------------------------------------------------------------------- metrics
def metrics_1d(s: np.ndarray, E: np.ndarray, span: float, *, amp_tol: float = 0.5,
               phase_tol: float = 5.0) -> dict[str, float]:
    """Quiet-zone numbers along a cut: ``s`` positions from the zone centre, ``E`` complex along it.

    ``taper_dB`` the range of a quadratic fit of the amplitude (the smooth illumination taper), ``ripple_pp_dB``
    the peak to peak of the residual about it, the same for the phase, and ``usable_width`` the largest centred
    segment where the amplitude stays within ``+- amp_tol`` and the phase within ``+- phase_tol``.
    """
    s = np.asarray(s, float)
    E = np.asarray(E)
    c = int(np.argmin(np.abs(s)))
    A = 20 * np.log10(np.abs(E) / np.abs(E[c]))
    P = np.rad2deg(np.unwrap(np.angle(E / E[c])))
    P -= P[c]
    m = np.abs(s) <= span / 2 * (1 + 1e-9)
    V = np.stack([np.ones(m.sum()), s[m], s[m] ** 2], 1)
    fa = V @ np.linalg.lstsq(V, A[m], rcond=None)[0]
    fp = V @ np.linalg.lstsq(V, P[m], rcond=None)[0]
    order = np.argsort(np.abs(s))
    As, Ps, ss = A[order], P[order], s[order]
    amax = np.maximum.accumulate(As) - np.minimum.accumulate(As)
    pmax = np.maximum.accumulate(Ps) - np.minimum.accumulate(Ps)
    ok = (amax <= 2 * amp_tol) & (pmax <= 2 * phase_tol)
    bad = np.flatnonzero(~ok)
    return {"amp_pp_dB": float(np.ptp(A[m])), "phase_pp_deg": float(np.ptp(P[m])),
            "taper_dB": float(np.ptp(fa)), "ripple_pp_dB": float(np.ptp(A[m] - fa)),
            "phase_ripple_pp_deg": float(np.ptp(P[m] - fp)),
            "usable_width": float(2 * (abs(ss[bad[0]]) if len(bad) else abs(ss[-1]))),
            "usable_limited_by_region": bool(len(bad) == 0), "n_points": int(m.sum())}


# ----------------------------------------------------------------------------------------------- the range
@dataclass
class Range2D:
    """A two-dimensional compact range: section, feed, frequency, quiet-zone plane and width."""
    section: Section = field(default_factory=Section)
    q: float = 4.42
    freq_hz: float = 3e9
    qz_z: float = 1.6
    qz_width: float = 0.5
    qz_centre_y: float | None = None          # default: the centre of the section
    qz_points: int = 241
    per_wavelength: float = 12.0
    feed_amp: float = 1.0
    roll_reference: float = 0.15

    @property
    def wavelength(self) -> float:
        return C0 / self.freq_hz

    @property
    def k(self) -> float:
        return 2 * np.pi / self.wavelength

    @property
    def grid(self) -> Grid:
        return self.section.contour(self.wavelength / self.per_wavelength)

    def feed(self) -> Feed:
        sec = self.section
        y_c = 0.5 * (sec.y_low + sec.y_high)
        c = np.array([y_c, y_c ** 2 / (4 * sec.focal)])
        pos = np.array([0.0, sec.focal])
        return Feed(q=self.q, position=tuple(pos), axis=tuple(c - pos), amp=self.feed_amp)

    def qz_line(self) -> np.ndarray:
        """The points of the quiet-zone line, ``(qz_points, 2)`` in absolute ``(y, z)``, centred on
        ``qz_centre_y`` (the section centre by default)."""
        s = np.linspace(-self.qz_width / 2, self.qz_width / 2, self.qz_points)
        y_c = 0.5 * (self.section.y_low + self.section.y_high) if self.qz_centre_y is None else self.qz_centre_y
        return np.c_[s + y_c, np.full_like(s, self.qz_z)]


def _qz_metrics(rng: Range2D, E_qz: np.ndarray, s: np.ndarray, **kw) -> dict:
    m = metrics_1d(s, E_qz, rng.qz_width, **kw)
    m.update({"freq_GHz": rng.freq_hz / 1e9, "qz_width": rng.qz_width, "qz_z": rng.qz_z,
              "y_low": rng.section.y_low, "y_high": rng.section.y_high, "roll": rng.section.roll,
              "blend": rng.section.blend, "aspect": rng.section.aspect, "q": rng.q})
    return m


def po_metrics(rng: Range2D, *, grid: Grid | None = None, **kw) -> dict:
    """Quiet-zone numbers of the physical-optics density."""
    t0 = time.perf_counter()
    g = grid or rng.grid
    rho = po_density(g, rng.feed(), rng.k)
    s, obs = _qz(rng)
    E = incident(obs, rng.feed(), rng.k) + radiate_2d(g.mid, g.length, rho, obs, rng.k)
    m = _qz_metrics(rng, E, s, **kw)
    m.update({"n_segments": int(len(g.mid)), "seconds": round(time.perf_counter() - t0, 4)})
    return m


def direct_metrics(rng: Range2D, *, grid: Grid | None = None, **kw) -> dict:
    """Quiet-zone numbers of the exact (moment-method) density, and how far it is from PO."""
    t0 = time.perf_counter()
    g = grid or rng.grid
    sol = mom_solve(g, rng.feed(), rng.k)
    s, obs = _qz(rng)
    E = incident(obs, rng.feed(), rng.k) + radiate_2d(g.mid, g.length, sol["rho"], obs, rng.k)
    m = _qz_metrics(rng, E, s, **kw)
    rho_po = po_density(g, rng.feed(), rng.k)
    m.update({"n_segments": int(len(g.mid)), "seconds": round(time.perf_counter() - t0, 4),
              "mom_seconds": round(sol["seconds"], 4), "condition": sol["condition"],
              "po_distance": float(np.linalg.norm((sol["rho"] - rho_po) * g.length) /
                                   max(np.linalg.norm(rho_po * g.length), 1e-300))})
    return m


def _qz(rng: Range2D):
    obs = rng.qz_line()
    return obs[:, 0] - obs[:, 0].mean(), obs


def near_field_map(rng: Range2D, *, n_y: int = 300, n_z: int = 280, pad: float = 0.05,
                   floor_db: float = -20.0, ceiling_db: float = 10.0, grid: Grid | None = None,
                   solution: dict | None = None) -> dict:
    """``|E|`` on a ``(y, z)`` grid from the PO density and from the exact one, and where the two part company.

    The grid spans the aperture plus ``pad`` (m) in ``y`` and ``0 .. qz_z + 0.25`` m in ``z``; the quiet-zone plane
    sits inside it. Fields are returned as ``(n_y, n_z)`` arrays indexed ``[y, z]`` (transpose to plot with ``z`` up).

    Returns a dict:

    * ``y``, ``z`` -- the grid axes (m);
    * ``po_db``, ``direct_db`` -- ``20 log10 |E|`` re the larger of the two maxima, so both maps share one scale;
    * ``diff_db`` -- ``20 log10(|E_po - E_direct| / |E_direct|)``, clipped to ``[floor_db, ceiling_db]``: where the
      two solvers disagree, in dB of the local field (0 dB = the error is as large as the field);
    * ``diff_db_percentiles`` -- the 50th, 90th and 99th percentile of ``diff_db`` over the whole grid, keyed
      ``"50"``, ``"90"``, ``"99"`` so the dict writes to JSON as it stands;
    * ``qz_s``, ``cut_po_db``, ``cut_direct_db`` -- the quiet-zone line cut (``qz_s`` centred), each solver re its
      own maximum; ``qz_z``, ``qz_width`` -- where that line is;
    * ``reference`` (the amplitude of ``po_db``/``direct_db``), ``n_segments``, ``seconds``.

    ``grid`` and ``solution`` reuse an already built mesh and exact current (the sweep solves each design once).
    """
    t0 = time.perf_counter()
    g = grid or rng.grid
    feed = rng.feed()
    y = np.linspace(g.mid[:, 0].min() - pad, g.mid[:, 0].max() + pad, n_y)
    z = np.linspace(0.0, rng.qz_z + 0.25, n_z)
    pts = np.stack(np.meshgrid(y, z, indexing="ij"), axis=-1).reshape(-1, 2)

    rho_po = po_density(g, feed, rng.k)
    rho_di = (solution or mom_solve(g, feed, rng.k))["rho"]
    e_po = incident(pts, feed, rng.k) + radiate_2d(g.mid, g.length, rho_po, pts, rng.k)
    e_di = incident(pts, feed, rng.k) + radiate_2d(g.mid, g.length, rho_di, pts, rng.k)

    a_po, a_di = np.abs(e_po), np.abs(e_di)
    ref = max(a_di.max(), a_po.max(), 1e-300)
    with np.errstate(divide="ignore"):
        rel_db = 20 * np.log10(np.maximum(np.abs(e_po - e_di) / np.maximum(a_di, 1e-300), 1e-300))
        shape = (n_y, n_z)
        d_db = np.clip(rel_db, floor_db, ceiling_db).reshape(shape)

    s, obs = _qz(rng)
    cut_po = np.abs(incident(obs, feed, rng.k) + radiate_2d(g.mid, g.length, rho_po, obs, rng.k))
    cut_di = np.abs(incident(obs, feed, rng.k) + radiate_2d(g.mid, g.length, rho_di, obs, rng.k))
    return {
        "y": y, "z": z,
        "po_db": (20 * np.log10(np.maximum(a_po, 1e-300) / ref)).reshape(shape),
        "direct_db": (20 * np.log10(np.maximum(a_di, 1e-300) / ref)).reshape(shape),
        "diff_db": d_db,
        "diff_db_percentiles": {str(p): float(np.percentile(rel_db, p)) for p in (50, 90, 99)},
        "qz_s": s, "qz_z": rng.qz_z, "qz_width": rng.qz_width,
        "cut_po_db": 20 * np.log10(np.maximum(cut_po, 1e-300) / max(cut_po.max(), 1e-300)),
        "cut_direct_db": 20 * np.log10(np.maximum(cut_di, 1e-300) / max(cut_di.max(), 1e-300)),
        "reference": float(ref), "n_segments": int(len(g.mid)),
        "seconds": round(time.perf_counter() - t0, 4),
    }


# ----------------------------------------------------------------------------------------------- validation
def validate_circle(*, ka: float = 6.0, per_wavelength: float = 12.0, source_distance: float = 6.0) -> dict:
    """The exact current of a PEC circle in a cylindrical wave against :func:`mom_solve`.

    The incident field of a line source at distance ``d`` from the centre is ``sum_n H_n^(2)(k d) J_n(k r)
    e^{j n phi}`` (Graf's addition theorem); the scattered field is then ``sum_n a_n H_n^(2)(k r) e^{j n phi}``
    with ``a_n = -H_n^(2)(k d) J_n(k a) / H_n^(2)(k a)``. Both the direct solve and PO are compared on the same
    contour, in dB of the field at a point on the shadow side and on the lit side.
    """
    lam = 1.0
    k = 2 * np.pi / lam
    a = ka / k
    d = source_distance * a
    h = lam / per_wavelength
    n = max(24, int(np.ceil(2 * np.pi * a / h)))
    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.c_[a * np.cos(phi), a * np.sin(phi)]
    pts = np.r_[pts, pts[:1]]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    mid = 0.5 * (pts[1:] + pts[:-1])
    t = np.diff(pts, axis=0) / seg[:, None]
    nrm = -np.stack([t[:, 1], -t[:, 0]], 1)                          # outward
    g = Grid(pts, mid, t, nrm, seg)
    feed = Feed(q=0.0, position=(d, 0.0), axis=(0.0, 1.0), amp=1.0)  # monopole line source; pattern off
    feed = replace(feed, q=0.0)

    class _Mono(Feed):
        def field(self, p):
            R = np.linalg.norm(np.asarray(p, float) - np.asarray(self.position, float), axis=1)
            return self.amp * _h0(self._k * R)

    mono = _Mono(q=0.0, position=(d, 0.0), axis=(0.0, 1.0), amp=1.0, _k=k)
    sol = mom_solve(g, mono, k)
    obs = np.array([[-2.5 * a, 0.0], [2.5 * a, 0.0], [0.0, 2.5 * a], [0.0, -2.5 * a]])
    E_mom = mono.field(obs) + radiate_2d(g.mid, g.length, sol["rho"], obs, k)
    E_exact = np.array([_circle_exact(r, k, a, d) for r in obs])
    # PO on the lit side
    lit = np.einsum("ij,ij->i", g.normal, np.asarray(mono.position, float) - g.mid) > 0
    rho_po = np.where(lit, 2 * mono.normal_derivative(g.mid, g.normal, h=1e-4 / k), 0.0)
    E_po = mono.field(obs) + radiate_2d(g.mid, g.length, rho_po, obs, k)
    err = lambda v: float(np.max(np.abs(v - E_exact) / np.abs(E_exact)))          # noqa: E731
    return {"ka": ka, "n_segments": int(len(mid)), "per_wavelength": per_wavelength,
            "points": obs.tolist(),
            "exact_abs": np.abs(E_exact).tolist(), "mom_abs": np.abs(E_mom).tolist(), "po_abs": np.abs(E_po).tolist(),
            "max_rel_error_mom": err(E_mom), "max_rel_error_po": err(E_po),
            "mean_abs_db_mom": float(np.mean(np.abs(20 * np.log10(np.abs(E_mom) / np.abs(E_exact))))),
            "mean_abs_db_po": float(np.mean(np.abs(20 * np.log10(np.abs(E_po) / np.abs(E_exact)))))}


def _circle_exact(r, k: float, a: float, d: float) -> complex:
    """Exact total field at ``r`` (2-vector) for a monopole at ``(d, 0)`` and a PEC circle of radius ``a``."""
    r = np.asarray(r, float)
    rho = float(np.hypot(*r))
    ph = float(np.arctan2(r[1], r[0]))
    nmax = int(np.ceil(k * a + 8 * (k * a) ** (1 / 3) + 8))
    out = 0.0 + 0.0j
    for m in range(-nmax, nmax + 1):
        c_j = hankel2(m, k * d)
        coef = -c_j * jv(m, k * a) / hankel2(m, k * a)
        out += c_j * jv(m, k * rho) * np.exp(1j * m * ph) + coef * hankel2(m, k * rho) * np.exp(1j * m * ph)
    return complex(out)


# ---------------------------------------------------------------------------------------------- the space
@dataclass(frozen=True)
class Design:
    """One point of the 2-D design space: aperture size (as a fraction of the reference), feed ``q`` and an edge
    profile ``(roll factor, blend, aspect)``."""
    size: float
    q: float
    roll_factor: float
    blend: float
    aspect: float

    @property
    def tag(self) -> str:
        return f"a{self.size:g}_q{self.q:g}_r{self.roll_factor:g}_b{self.blend:g}_x{self.aspect:g}"

    def inputs(self) -> dict[str, float]:
        return {"size": self.size, "q": self.q, "roll_factor": self.roll_factor, "blend": self.blend,
                "aspect": self.aspect}

    def features(self) -> list[float]:
        return [self.size, self.q, self.roll_factor, self.blend, self.aspect]


def design_space(base: Range2D, sizes, qs, profiles=EDGE_FAMILY) -> list[tuple[Design, Range2D]]:
    """Every design of the study as ``(Design, Range2D)``: the aperture scaled about its centre (the quiet zone
    stays where it is), the feed ``q`` and the rolled-edge profile applied to the section."""
    out = []
    y0, y1 = base.section.y_low, base.section.y_high
    c = 0.5 * (y0 + y1)
    for s in sizes:
        half = 0.5 * (y1 - y0) * s
        for q in qs:
            for (rf, bl, asp) in profiles:
                sec = replace(base.section, y_low=c - half, y_high=c + half,
                              roll=rf * base.roll_reference, blend=bl, aspect=asp)
                d = Design(float(s), float(q), rf, bl, asp)
                out.append((d, replace(base, section=sec, q=float(q))))
    return out
