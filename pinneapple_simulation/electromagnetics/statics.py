"""Electrostatics and magnetostatics in closed form, and Biot-Savart integration of filamentary currents.

References: J. D. Jackson, *Classical Electrodynamics*, 3rd ed. (1999), ch. 1-5; D. J. Griffiths,
*Introduction to Electrodynamics*, 4th ed. (2017), ch. 2-5; the off-axis field of a circular loop in
complete elliptic integrals is Jackson's eq. 5.37 rewritten for ``B`` (Simpson et al., NASA TM 2001-210720
give the same expressions in Cartesian form).

All functions are SI and vectorised over field points.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .constants import EPS0, MU0

__all__ = [
    "parallel_plate_capacitance", "coax_capacitance", "coax_inductance", "spherical_capacitance",
    "point_charge_field", "line_charge_field", "infinite_wire_B", "biot_savart", "polyline_segments",
    "circle_segments", "helix_segments", "loop_field", "loop_on_axis_B", "finite_solenoid_on_axis_B",
    "helmholtz_coil_center_B", "mutual_inductance_coaxial_loops", "loop_self_inductance", "filament_inductance",
]


def parallel_plate_capacitance(area: float, gap: float, eps_r: float = 1.0) -> float:
    """``C = eps A / d`` (fringing neglected), F."""
    return eps_r * EPS0 * area / gap


def coax_capacitance(a: float, b: float, eps_r: float = 1.0) -> float:
    """Per-unit-length capacitance of a coaxial line, ``2 pi eps / ln(b/a)`` (F/m)."""
    return 2.0 * math.pi * eps_r * EPS0 / math.log(b / a)


def coax_inductance(a: float, b: float, mu_r: float = 1.0) -> float:
    """External per-unit-length inductance of a coaxial line, ``mu ln(b/a) / (2 pi)`` (H/m)."""
    return mu_r * MU0 * math.log(b / a) / (2.0 * math.pi)


def spherical_capacitance(a: float, b: float = math.inf, eps_r: float = 1.0) -> float:
    """Concentric spheres ``4 pi eps a b / (b - a)``; isolated sphere ``4 pi eps a`` for ``b = inf``."""
    if math.isinf(b):
        return 4.0 * math.pi * eps_r * EPS0 * a
    return 4.0 * math.pi * eps_r * EPS0 * a * b / (b - a)


def point_charge_field(points, q: float, at=(0.0, 0.0, 0.0), eps_r: float = 1.0):
    """(phi, E) of a point charge: ``phi = q / (4 pi eps r)``, ``E = q r_hat / (4 pi eps r^2)``."""
    d = np.asarray(points, float) - np.asarray(at, float)
    r = np.linalg.norm(d, axis=-1)
    k = q / (4.0 * math.pi * eps_r * EPS0)
    return k / r, k * d / r[..., None] ** 3


def line_charge_field(points, lam: float, eps_r: float = 1.0):
    """E of an infinite line charge on the z axis, ``lam / (2 pi eps s)`` radially (V/m)."""
    p = np.asarray(points, float)
    s2 = p[..., 0] ** 2 + p[..., 1] ** 2
    k = lam / (2.0 * math.pi * eps_r * EPS0)
    E = np.zeros_like(p)
    E[..., 0] = k * p[..., 0] / s2
    E[..., 1] = k * p[..., 1] / s2
    return E


def infinite_wire_B(points, current: float):
    """B of an infinite straight filament on the z axis carrying ``current`` towards +z: ``mu0 I / (2 pi s)``
    in the azimuthal direction (T)."""
    p = np.asarray(points, float)
    s2 = p[..., 0] ** 2 + p[..., 1] ** 2
    k = MU0 * current / (2.0 * math.pi)
    B = np.zeros_like(p)
    B[..., 0] = -k * p[..., 1] / s2
    B[..., 1] = k * p[..., 0] / s2
    return B


def polyline_segments(vertices) -> tuple[np.ndarray, np.ndarray]:
    """Segments (start, end) of an open polyline through ``vertices`` (n, 3)."""
    v = np.asarray(vertices, float)
    return v[:-1], v[1:]


def circle_segments(radius: float, n: int = 360, center=(0.0, 0.0, 0.0), z: float = 0.0):
    """A circle of ``n`` straight segments in a plane ``z = const`` (counter-clockwise seen from +z).

    The vertices lie on the circle, so the polygon is inscribed; its field converges to the circle's at
    second order in ``1/n``."""
    t = np.linspace(0.0, 2.0 * math.pi, n + 1)
    v = np.column_stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t),
                         np.full(n + 1, center[2] + z)])
    return polyline_segments(v)


def helix_segments(radius: float, length: float, turns: float, n_per_turn: int = 72, z0: float | None = None):
    """A helical coil along z from ``z0`` (default ``-length/2``) to ``z0 + length``."""
    z0 = -0.5 * length if z0 is None else z0
    n = max(int(round(turns * n_per_turn)), 2)
    t = np.linspace(0.0, 2.0 * math.pi * turns, n + 1)
    v = np.column_stack([radius * np.cos(t), radius * np.sin(t), z0 + length * t / t[-1]])
    return polyline_segments(v)


def biot_savart(points, starts, ends, current: float | Sequence[float] = 1.0, chunk: int = 4096):
    """B (T) at ``points`` (..., 3) from straight filament segments ``starts -> ends`` (m, 3) each carrying
    ``current`` (A, scalar or per segment). Exact field of each finite segment (Jackson problem 5.1 / Griffiths
    eq. 5.37): ``B = mu0 I / (4 pi) (a x b)(|a| + |b|) / (|a| |b| (|a| |b| + a.b))`` with ``a = start - p``,
    ``b = end - p``. Points on a segment's line get zero contribution from it."""
    P = np.asarray(points, float)
    shape = P.shape
    P = P.reshape(-1, 3)
    s = np.asarray(starts, float)
    e = np.asarray(ends, float)
    Iseg = np.broadcast_to(np.asarray(current, float), (len(s),))
    out = np.zeros_like(P)
    for i0 in range(0, len(P), chunk):
        p = P[i0:i0 + chunk, None, :]
        a = s[None] - p
        b = e[None] - p
        na = np.linalg.norm(a, axis=-1)
        nb = np.linalg.norm(b, axis=-1)
        cr = np.cross(a, b)
        ab = np.einsum("ijk,ijk->ij", a, b)
        cr2 = np.einsum("ijk,ijk->ij", cr, cr)
        on_line = np.sqrt(cr2) <= 1e-14 * np.maximum(na * nb, 1e-300)
        # beside a long segment |a||b| + a.b cancels; there use |a||b| + a.b = |a x b|^2 / (|a||b| - a.b)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(ab < 0, (na + nb) * (na * nb - ab) / (na * nb * np.where(on_line, 1.0, cr2)),
                         (na + nb) / (na * nb * (na * nb + ab)))
            f = np.where(on_line, 0.0, f)
        out[i0:i0 + chunk] = MU0 / (4.0 * math.pi) * np.einsum("ij,ijk->ik", f * Iseg[None], cr)
    return out.reshape(shape)


def loop_on_axis_B(z, radius: float, current: float):
    """Axial B of a circular loop in the plane z = 0: ``mu0 I R^2 / (2 (R^2 + z^2)^{3/2})``."""
    z = np.asarray(z, float)
    return MU0 * current * radius ** 2 / (2.0 * (radius ** 2 + z ** 2) ** 1.5)


def loop_field(points, radius: float, current: float):
    """B (T) of a circular loop of radius ``R`` in the plane z = 0, centred on the axis, everywhere off the
    wire, with complete elliptic integrals (``m = k^2 = 4 R s / ((R + s)^2 + z^2)``)::

        B_z = mu0 I / (2 pi) / sqrt((R + s)^2 + z^2) [K(m) + (R^2 - s^2 - z^2) / ((R - s)^2 + z^2) E(m)]
        B_s = mu0 I z / (2 pi s) / sqrt((R + s)^2 + z^2) [-K(m) + (R^2 + s^2 + z^2) / ((R - s)^2 + z^2) E(m)]
    """
    from scipy.special import ellipe, ellipk

    p = np.asarray(points, float)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    s = np.hypot(x, y)
    R = radius
    q = (R + s) ** 2 + z ** 2
    m = 4.0 * R * s / q
    K, E = ellipk(m), ellipe(m)
    d = (R - s) ** 2 + z ** 2
    c = MU0 * current / (2.0 * math.pi * np.sqrt(q))
    Bz = c * (K + (R * R - s * s - z * z) / d * E)
    with np.errstate(divide="ignore", invalid="ignore"):
        Bs = np.where(s > 1e-12 * R, c * z / np.where(s > 0, s, 1.0) * (-K + (R * R + s * s + z * z) / d * E), 0.0)
        cphi = np.where(s > 0, x / np.where(s > 0, s, 1.0), 1.0)
        sphi = np.where(s > 0, y / np.where(s > 0, s, 1.0), 0.0)
    return np.stack([Bs * cphi, Bs * sphi, Bz], axis=-1)


def finite_solenoid_on_axis_B(z, radius: float, length: float, n_turns: float, current: float):
    """Axial B of a tightly wound finite solenoid (current sheet ``K = N I / L``) centred at z = 0:
    ``(mu0 N I / (2 L)) [(z + L/2) / sqrt(R^2 + (z + L/2)^2) - (z - L/2) / sqrt(R^2 + (z - L/2)^2)]``."""
    z = np.asarray(z, float)
    n = n_turns / length
    zp, zm = z + 0.5 * length, z - 0.5 * length
    return 0.5 * MU0 * n * current * (zp / np.sqrt(radius ** 2 + zp ** 2) - zm / np.sqrt(radius ** 2 + zm ** 2))


def helmholtz_coil_center_B(radius: float, current: float, turns: int = 1) -> float:
    """Field at the centre of a Helmholtz pair (spacing = radius): ``(4/5)^{3/2} mu0 n I / R``."""
    return (0.8 ** 1.5) * MU0 * turns * current / radius


def mutual_inductance_coaxial_loops(r1: float, r2: float, d: float) -> float:
    """Maxwell's formula for two coaxial circular filaments at axial distance ``d``:
    ``M = mu0 sqrt(r1 r2) [(2/k - k) K(k^2) - (2/k) E(k^2)]``, ``k^2 = 4 r1 r2 / ((r1 + r2)^2 + d^2)``."""
    from scipy.special import ellipe, ellipk

    m = 4.0 * r1 * r2 / ((r1 + r2) ** 2 + d ** 2)
    k = math.sqrt(m)
    return MU0 * math.sqrt(r1 * r2) * ((2.0 / k - k) * ellipk(m) - 2.0 / k * ellipe(m))


def loop_self_inductance(R: float, a: float, internal: bool = True) -> float:
    """Thin round wire loop (loop radius ``R`` >> wire radius ``a``): ``mu0 R (ln(8R/a) - 2 + Y)`` with
    ``Y = 1/4`` for uniform current (low frequency, ``internal=True``) and 0 at high frequency (Rosa & Grover,
    *Formulas and tables for the calculation of mutual and self-inductance*, Bull. Bureau of Standards 8, 1912)."""
    return MU0 * R * (math.log(8.0 * R / a) - 2.0 + (0.25 if internal else 0.0))


def filament_inductance(paths, wire_radius, pieces_per_metre: float = 2000.0, gauss: int = 6,
                        mirror_z: float | None = None) -> np.ndarray:
    """Inductance matrix (H) of thin conductors along polylines ``paths`` (each an (n, 3) array, closed or open;
    open paths give the partial inductances of their current segments).

    Neumann's double integral with the thin-wire reduced kernel ``1 / sqrt(|x - x'|^2 + g^2)``, ``g`` the geometric
    mean distance of the round wire cross-section from itself (``g = a e^-1/4`` for ``wire_radius = a``): a circle of
    radius R then gives ``mu0 R (ln(8 R / a) - 7/4)``, Rosa's low-frequency loop inductance
    (:func:`loop_self_inductance`). The paths are cut into straight pieces; the diagonal blocks are integrated in
    closed form, the others with ``gauss``-point Gauss-Legendre rules. ``wire_radius``: one value or one per path.
    ``mirror_z``: a plane ``z = mirror_z`` of infinitely permeable iron (the image of each element carries the same
    tangential current), added to every mutual and self term."""
    paths = [np.asarray(p, float) for p in paths]
    radii = np.broadcast_to(np.asarray(wire_radius, float), (len(paths),))
    segs = []
    for k, P in enumerate(paths):
        for a, b in zip(P[:-1], P[1:], strict=True):
            L = float(np.linalg.norm(b - a))
            n = max(1, int(math.ceil(L * pieces_per_metre)))
            t = np.linspace(0.0, 1.0, n + 1)[:, None]
            pts = a + t * (b - a)
            for i in range(n):
                segs.append((k, pts[i], pts[i + 1]))
    owner = np.array([s_[0] for s_ in segs])
    A = np.array([s_[1] for s_ in segs])
    B = np.array([s_[2] for s_ in segs])
    D = B - A
    lens = np.linalg.norm(D, axis=1)
    x, w = np.polynomial.legendre.leggauss(gauss)
    x, w = 0.5 * (x + 1.0), 0.5 * w
    Q = A[:, None, :] + x[None, :, None] * D[:, None, :]  # (m, gauss, 3)
    g = (radii * math.exp(-0.25))[owner]
    n_p = len(paths)
    Lmat = np.zeros((n_p, n_p))
    images = [None] if mirror_z is None else [None, mirror_z]
    for img in images:
        Qi, Di = Q, D
        if img is not None:  # mirror image in z = img: positions reflected, tangential (x, y) components kept
            Qi = Q.copy()
            Qi[..., 2] = 2 * img - Q[..., 2]
            Di = D.copy()
            Di[:, 2] = -D[:, 2]
        dots = D @ Di.T  # (m, m)
        for i in range(len(segs)):
            r = Q[i][:, None, None, :] - Qi[None, :, :, :]  # (gauss, m, gauss, 3)
            gg = np.where(owner == owner[i], g[i], 0.0)  # the reduced kernel within one conductor only
            dist = np.sqrt((r ** 2).sum(-1) + (gg[None, :, None] ** 2 if img is None else 0.0))
            kern = (w[:, None, None] * w[None, None, :] / dist).sum((0, 2))
            val = dots[i] * kern
            if img is None:  # diagonal block in closed form
                ln, gi = lens[i], g[i]
                val[i] = 2.0 * (ln * math.asinh(ln / gi) - math.sqrt(ln * ln + gi * gi) + gi)
            np.add.at(Lmat[owner[i]], owner, val)
    return 1e-7 * Lmat
