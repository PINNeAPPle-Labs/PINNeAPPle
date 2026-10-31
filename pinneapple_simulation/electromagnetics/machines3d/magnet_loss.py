"""Eddy-current loss in a segmented permanent magnet, 3-D against 2-D and against the closed form.

A surface magnet sees a field ripple normal to its face (slot harmonics, armature MMF harmonics). The 2-D machine
model (``postprocess.magnet_eddy_loss_resistance_limited``) lets the eddy current flow along the axis with no net
current per piece: it is the loss of an infinitely long strip, and it cannot see axial segmentation. In 3-D the
current loops in the face of the magnet and closes across its width at the ends, so short pieces lose less.

The reference is the resistance-limited loss of a thin rectangular piece ``a x b`` (thickness ``h``) in a uniform
field ``B cos(wt)`` normal to it: the current stream function solves ``lap(psi) = j w sigma B`` with ``psi = 0`` on
the edge, whose double sine series gives

    P = (h sigma w^2 B^2 / 2) sum_{m, n odd} 64 a b / (pi^4 m^2 n^2 ((m pi / a)^2 + (n pi / b)^2)),

and for ``b -> inf`` the 2-D strip, ``P / b = h sigma w^2 B^2 a^3 / 24``. The 3-D model solves the full
time-harmonic problem (:func:`~pinneapple_simulation.electromagnetics.fem3d.nedelec.solve_eddy`) on a structured
mesh of the segmented block in a box whose boundary carries the uniform applied field, so it also shows where the
resistance-limited assumption fails (the skin depth reaching the piece size).
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence

import numpy as np

from ..fem3d.mesh import TetMesh, box_mesh
from ..fem3d.nedelec import NU0, EdgeSpace, solve_eddy

__all__ = ["plate_loss", "strip_loss", "loss_factor", "segmented_block", "block_loss_3d", "segmentation_study"]


def plate_loss(a: float, b: float, h: float, sigma: float, f: float, B: float, terms: int = 400) -> float:
    """Time-averaged resistance-limited loss (W) of a thin ``a x b`` piece, thickness ``h``, in a uniform normal
    field of peak ``B`` at ``f`` (the double sine series above, ``terms`` odd terms in each direction).

    Returns
    -------
    P : float [W]
        Time-averaged loss of the piece.
    """
    m = np.arange(1, 2 * terms, 2, dtype=float)
    M, N = np.meshgrid(m, m, indexing="ij")
    s = np.sum(64 * a * b / (math.pi ** 4 * M ** 2 * N ** 2 * ((M * math.pi / a) ** 2 + (N * math.pi / b) ** 2)))
    w = 2 * math.pi * f
    return 0.5 * h * sigma * w ** 2 * B ** 2 * float(s)


def strip_loss(a: float, b: float, h: float, sigma: float, f: float, B: float) -> float:
    """The 2-D value: an infinitely long strip of width ``a`` (axial current, no net current), over length ``b``.

    Returns
    -------
    P : float [W]
        Time-averaged loss of the strip over length ``b``.
    """
    w = 2 * math.pi * f
    return h * sigma * w ** 2 * B ** 2 * a ** 3 * b / 24


def loss_factor(a: float, b: float, terms: int = 400) -> float:
    """``plate_loss / strip_loss``: the share of the 2-D loss a piece of width ``a`` and length ``b`` keeps.

    Returns
    -------
    ratio : float [-]
        ``plate_loss / strip_loss``.
    """
    return plate_loss(a, b, 1.0, 1.0, 1.0, 1.0, terms) / strip_loss(a, b, 1.0, 1.0, 1.0, 1.0)


def _grid(lo: float, hi: float, pieces: int, gap: float, h: float) -> np.ndarray:
    """Nodes over ``[lo, hi]`` split into ``pieces`` conductors separated by insulating gaps of width ``gap``."""
    width = (hi - lo - (pieces - 1) * gap) / pieces
    pts = []
    x = lo
    for p in range(pieces):
        n = max(2, math.ceil(width / h))
        pts.append(np.linspace(x, x + width, n + 1))
        x += width
        if p < pieces - 1:
            pts.append(np.array([x + gap]))
            x += gap
    return np.unique(np.concatenate(pts))


def _air(lo: float, hi: float, margin: float, h0: float, growth: float = 1.4) -> tuple[np.ndarray, np.ndarray]:
    """Graded air cells from the block faces out to ``margin``."""
    steps, d, x = [], h0, 0.0
    while x < margin:
        x = min(margin, x + d)
        steps.append(x)
        d *= growth
    s = np.asarray(steps)
    return lo - s[::-1], hi + s


def segmented_block(width: float, length: float, thickness: float, n_width: int = 1, n_length: int = 1,
                    gap: float = 2e-4, h: float = 1.5e-3, h_thick: float | None = None,
                    margin: float = 0.03) -> TetMesh:
    """Structured mesh of a magnet ``width x length x thickness`` (x, y, z) cut into ``n_width x n_length``
    insulated pieces (``gap`` of air between them), centred at the origin in a box of air ``margin`` wide.
    Regions: ``magnet`` and ``air``."""
    xs = _grid(-width / 2, width / 2, n_width, gap, h)
    ys = _grid(-length / 2, length / 2, n_length, gap, h)
    hz = h_thick or thickness / 4
    zs = np.linspace(-thickness / 2, thickness / 2, max(2, math.ceil(thickness / hz)) + 1)
    ax = _air(xs[0], xs[-1], margin, h)
    ay = _air(ys[0], ys[-1], margin, h)
    az = _air(zs[0], zs[-1], margin, hz)
    X = np.concatenate([ax[0], xs, ax[1]])
    Y = np.concatenate([ay[0], ys, ay[1]])
    Z = np.concatenate([az[0], zs, az[1]])
    wp = (width - (n_width - 1) * gap) / n_width
    lp = (length - (n_length - 1) * gap) / n_length

    def inside(c, lo, size, count):
        u = c - lo
        k = np.floor(u / (size + gap))
        return (u >= 0) & (k < count) & (u - k * (size + gap) <= size)

    def region(c):
        mag = (inside(c[:, 0], -width / 2, wp, n_width) & inside(c[:, 1], -length / 2, lp, n_length)
               & (np.abs(c[:, 2]) < thickness / 2))
        return np.where(mag, 1, 0)

    return box_mesh(X, Y, Z, region=region, names={"air": 0, "magnet": 1})


def block_loss_3d(mesh: TetMesh, sigma: float, f: float, B: float, solver: str = "auto") -> dict:
    """Eddy loss (W) in the ``magnet`` region of ``mesh`` in a uniform field ``B`` (peak, along z) at ``f``: the
    box boundary carries ``n x A`` of ``A0 = B x r / 2``."""
    t0 = time.perf_counter()
    space = EdgeSpace(mesh)
    de = mesh.boundary_edges
    B0 = np.array([0.0, 0.0, B])
    vals = space.edge_values(lambda x: 0.5 * np.cross(B0, x), de).astype(complex)
    s = np.where(mesh.mask("magnet"), sigma, 0.0)
    sol = solve_eddy(mesh, NU0, s, 2 * math.pi * f, np.zeros((len(mesh.tets), 3)), dirichlet=(de, vals),
                     space=space, solver=solver, rtol=1e-10)
    return {"loss_W": float(sol.loss), "unknowns": sol.info["unknowns"], "tets": int(len(mesh.tets)),
            "residual": sol.info["residual"], "seconds": time.perf_counter() - t0}


def segmentation_study(width: float, length: float, thickness: float, sigma: float, B: float,
                       segments: Sequence[tuple[int, int]], freqs: Sequence[float], h: float = 1.5e-3,
                       gap: float = 2e-4, progress: Callable[[str], None] | None = None) -> dict:
    """For each ``(n_width, n_length)`` segmentation and frequency: the 3-D loss, the closed-form
    resistance-limited loss of the pieces and the 2-D (infinite strip) loss, all for the whole magnet."""
    rows = []
    for nw, nl in segments:
        mesh = segmented_block(width, length, thickness, nw, nl, gap=gap, h=h)
        wp = (width - (nw - 1) * gap) / nw
        lp = (length - (nl - 1) * gap) / nl
        for f in freqs:
            r = block_loss_3d(mesh, sigma, f, B)
            ref = nw * nl * plate_loss(wp, lp, thickness, sigma, f, B)
            two_d = nw * nl * strip_loss(wp, lp, thickness, sigma, f, B)
            row = {"n_width": nw, "n_length": nl, "f_Hz": float(f), "piece_width_m": wp, "piece_length_m": lp,
                   "loss_3d_W": r["loss_W"], "loss_closed_form_W": ref, "loss_2d_W": two_d,
                   "ratio_3d_closed_form": r["loss_W"] / ref, "ratio_3d_2d": r["loss_W"] / two_d,
                   "factor_closed_form": ref / two_d, "skin_depth_m": math.sqrt(2 / (2 * math.pi * f / NU0 * sigma)),
                   "unknowns": r["unknowns"], "tets": r["tets"], "seconds": r["seconds"]}
            rows.append(row)
            if progress:
                progress(f"{nw} x {nl} pieces, {f:g} Hz: 3-D {row['loss_3d_W']:.4g} W, closed form "
                         f"{ref:.4g} W, 2-D {two_d:.4g} W ({row['unknowns']} unknowns, {row['seconds']:.0f} s)")
    return {"width": width, "length": length, "thickness": thickness, "sigma": sigma, "B": B, "rows": rows}
