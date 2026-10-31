"""Studio views of a 3-D machine's flux density: the iron and the magnets coloured by ``|B|``, a cut view and a
turntable, on the generic studio scene of PINNeAPPle (:mod:`pinneapple_tools.visualization.studio`).

Works for any model of this package that has a tetrahedral ``mesh`` and a ``materials(theta, ...)`` returning the
reluctivity and the remanence per tetrahedron (:class:`.afpm.AFPM3D`, :class:`.spm.SPM3D`): iron is where the
reluctivity is below ``nu0 / 10``, magnet where the remanence is not zero (:func:`tet_labels`).

* :func:`region_surfaces` -- the boundary of the iron and of the magnets (optionally with a sector of the machine
  cut away, so the field inside shows on the cut faces) as studio ``Surface`` objects in millimetres. Every face
  carries the ``|B|`` of the tetrahedron it bounds, with the face's own vertices: no averaging between elements,
  nothing smoother than the mesh.
* :func:`machine_scene` -- a ``Scene`` of those surfaces, field ``"B"`` labelled ``|B| (T)``, with the colour range
  (minimum and maximum over the surfaces shown) returned with it.
* :func:`turntable` -- the scene turning once about the machine axis, as an ``Animation`` for
  :func:`~pinneapple_tools.visualization.studio.blender.render_animation` (camera fixed).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..fem3d.nedelec import NU0

__all__ = ["AIR", "IRON", "MAGNET", "tet_labels", "region_surfaces", "machine_scene", "turntable", "MATERIALS"]

AIR, IRON, MAGNET = 0, 1, 2
#: dark steel for the iron and a satin finish for the magnets (sRGB; used where a surface is not field-coloured)
MATERIALS = {
    "iron": {"color": [0.42, 0.44, 0.47], "metallic": 0.8, "roughness": 0.45},
    "magnet": {"color": [0.62, 0.62, 0.66], "metallic": 0.6, "roughness": 0.3},
}
# the four faces of a tetrahedron (a b c d), each listed so that it points away from the opposite vertex when
# (a b c d) is positively oriented; the orientation is checked per face anyway
_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))


def tet_labels(nu: np.ndarray, Br: np.ndarray, iron_below: float = 0.1) -> np.ndarray:
    """``AIR``, ``IRON`` or ``MAGNET`` per tetrahedron from the reluctivity ``nu`` and the remanence ``Br`` (M, 3):
    iron where ``nu < iron_below * nu0``, magnet where ``|Br| > 0``."""
    lab = np.full(len(nu), AIR, np.int64)
    lab[np.asarray(nu) < iron_below * NU0] = IRON
    lab[np.linalg.norm(np.asarray(Br), axis=1) > 0] = MAGNET
    return lab


def _boundary(nodes: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Faces that belong to one tetrahedron of ``tets`` only, oriented outwards, and the tetrahedron of each."""
    F = np.concatenate([tets[:, f] for f in _FACES])
    owner = np.tile(np.arange(len(tets)), len(_FACES))
    _, first, count = np.unique(np.sort(F, axis=1), axis=0, return_index=True, return_counts=True)
    once = first[count == 1]
    F, owner = F[once], owner[once]
    p = nodes[F]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    out = p.mean(axis=1) - nodes[tets[owner]].mean(axis=1)
    flip = np.einsum("ij,ij->i", n, out) < 0
    F[flip] = F[flip][:, ::-1]
    return F, owner


def region_surfaces(mesh, labels: np.ndarray, B: np.ndarray, *, cut: tuple[float, float] | None = None,
                    regions: Sequence[tuple[int, str]] = ((IRON, "iron"), (MAGNET, "magnet")),
                    scale: float = 1e3, group: str = "machine") -> list:
    """The boundary of each region as a studio ``Surface`` with field ``"B"`` = ``|B|`` (T) of the tetrahedron each
    face bounds (vertices duplicated per face: flat colours, no averaging between elements).

    cut: ``(start, end)`` in degrees round the ``z`` axis: the tetrahedra whose centroid lies in that sector are left
    out, so the field inside the iron and the magnets shows on the cut faces. scale: metres to scene units
    (1e3: millimetres)."""
    from pinneapple_tools.visualization.studio import Surface

    nodes = np.asarray(mesh.nodes, float)
    tets = np.asarray(mesh.tets)
    keep = np.ones(len(tets), bool)
    if cut is not None:
        c = nodes[tets].mean(axis=1)
        ang = np.degrees(np.arctan2(c[:, 1], c[:, 0])) % 360.0
        a, b = cut[0] % 360.0, cut[1] % 360.0
        keep = ~((ang >= a) & (ang <= b)) if a <= b else ~((ang >= a) | (ang <= b))
    Bmag = np.linalg.norm(np.asarray(B), axis=1)
    out = []
    for lab, name in regions:
        idx = np.flatnonzero((labels == lab) & keep)
        if not len(idx):
            continue
        F, owner = _boundary(nodes, tets[idx])
        V = nodes[F].reshape(-1, 3) * scale
        faces = np.arange(len(V)).reshape(-1, 3)
        out.append(Surface(name, V, faces, name, fields={"B": np.repeat(Bmag[idx][owner], 3)}, group=group))
    return out


def machine_scene(mesh, labels: np.ndarray, B: np.ndarray, *, cut: tuple[float, float] | None = None,
                  title: str = "", scale: float = 1e3, clip: float | None = None) -> tuple[Any, dict[str, Any]]:
    """A studio ``Scene`` of :func:`region_surfaces` (``z`` up, millimetres, field ``"B"``) and what it shows:
    ``B_min`` / ``B_max`` (T, over the faces shown), ``range`` (the colour range: ``B_min`` to ``B_max``, or to the
    ``clip`` percentile of the faces when given, so that a few singular elements, e.g. at tooth tips with linear
    iron, do not set the scale), ``above`` (the share of faces above the range), ``faces``, ``cut``. The colour-bar
    label states a clipped maximum: ``|B| (T), max 4.05 T off scale``."""
    from pinneapple_tools.visualization.studio import Scene

    surfs = region_surfaces(mesh, labels, B, cut=cut, scale=scale)
    vals = np.concatenate([s.fields["B"][::3] for s in surfs])          # one value per face
    lo, hi = float(vals.min()), float(vals.max())
    top = float(np.percentile(vals, clip)) if clip is not None else hi
    label = "|B| (T)" if top >= hi else f"|B| (T), max {hi:.2f} T off scale"
    sc = Scene(surfs, axes="z_up", title=title, labels={"B": label}, materials=dict(MATERIALS))
    return sc, {"B_min": lo, "B_max": hi, "range": (lo, top), "above": float(np.mean(vals > top)),
                "faces": int(len(vals)), "cut": cut}


def turntable(scene, seconds: float = 8.0, fps: float = 24.0, turns: float = 1.0, group: str = "machine"):
    """``scene`` with an ``Animation``: the surfaces of ``group`` turning ``turns`` times about ``z`` in ``seconds``
    (the last frame stops one step short of the first, so the loop is seamless). Returns the scene."""
    from pinneapple_tools.visualization.studio.animation import Animation

    n = max(2, int(round(seconds * fps)))
    t = np.arange(n) / fps
    a = 2 * math.pi * turns * np.arange(n) / n
    q = np.column_stack([np.cos(a / 2), np.zeros(n), np.zeros(n), np.sin(a / 2)])
    an = Animation(t, camera="fixed", time_label="t", time_unit="s")
    an.add_track(group, np.zeros((n, 3)), q)
    scene.animation = an
    return scene
