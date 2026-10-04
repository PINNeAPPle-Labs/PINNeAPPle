"""Interactive browser demo: 2D shallow-water dam break, viewed live in PINNeAPPle Twin3D.

Runs the validated MUSCL-HLL shallow-water solver (``pinneapple_simulation.numerical_solvers.
shallow_water_fv.ShallowWater2D`` -- see ``tests/test_shallow_water_fv.py`` for its exact-solution
validation against Ritter/Stoker dam breaks) on a gate-release scenario with a submerged obstacle,
snapshots the depth and speed fields over time, and exports them as a real PINNeAPPle-Twin3D scene
(glTF geometry + a binary field file + a JSON manifest -- the same format
``pinneapple_twin3d.openfoam.scene_from_case`` produces from a real OpenFOAM case) that the
project's own three.js viewer (``pinneapple_twin3d/viewer/``) can play back and step through time.

Geometry note: the water surface is rendered as a *flat, per-cell, non-shared-vertex* grid (4
vertices per cell, field value repeated across them -- the same "flat-shaded field on a static
mesh" convention the OpenFOAM bridge uses), not a deforming mesh: only the ``depth_m``/``speed_m_s``
*field* varies per time step, not vertex positions (Twin3D's field file is per-vertex scalars on a
FIXED glTF geometry -- see ``pinneapple_twin3d/scene.py::Scene.add_field``). The obstacle is a
second, static, field-less part in a different colour so it reads visually as a solid.

    python examples/numerical_solvers/13_shallow_water_twin3d_browser_demo.py
    # then open the printed http://127.0.0.1:8765/ URL in a browser
"""
from __future__ import annotations

import os

import numpy as np

from pinneapple_simulation.numerical_solvers.shallow_water_fv import ShallowWater2D
from pinneapple_twin3d.scene import Scene
from pinneapple_twin3d.serve import serve

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out", "shallow_water_twin3d_demo")


def _cell_quads(x: np.ndarray, y: np.ndarray, dx: float, dy: float, mask: np.ndarray):
    """(nx,ny) cell-centre grids -> a flat-shaded quad-per-cell mesh (no shared vertices): 4
    vertices and 2 triangles per included cell, plus the (i, j) each vertex belongs to (so a
    (nx,ny)-shaped field array can be turned into a matching per-vertex array with one indexing
    op: ``field[owner_i, owner_j]``)."""
    ii, jj = np.nonzero(mask)
    n = len(ii)
    x0, x1 = x[ii, jj] - dx / 2, x[ii, jj] + dx / 2
    y0, y1 = y[ii, jj] - dy / 2, y[ii, jj] + dy / 2
    z = np.zeros(n)
    verts = np.stack([np.stack([x0, y0, z], 1), np.stack([x1, y0, z], 1),
                       np.stack([x1, y1, z], 1), np.stack([x0, y1, z], 1)], 1).reshape(-1, 3)
    base = (4 * np.arange(n))[:, None]
    faces = np.concatenate([base + [0, 1, 2], base + [0, 2, 3]], axis=0)
    owner_i, owner_j = np.repeat(ii, 4), np.repeat(jj, 4)
    return verts, faces, owner_i, owner_j


def build_scene(n_snapshots: int = 30) -> Scene:
    nx, ny, L = 120, 30, 3.0
    sw = ShallowWater2D(nx, ny, L / nx)
    X, Y = sw.centers()
    sw.h[X < 1.0] = 0.5                                     # reservoir behind the gate
    gate = (X > 1.0) & (X < 1.0 + 2 * sw.dx)
    sw.add_gate("gate", gate, release_time=0.1)
    obstacle = (X > 2.2) & (X < 2.4) & (Y > 0.3) & (Y < 0.6)  # a submerged pillar downstream
    sw.add_wall(obstacle)

    fluid = ~sw.solid
    v, f, oi, oj = _cell_quads(X, Y, sw.dx, sw.dy, fluid)
    ov, of, ooi, ooj = _cell_quads(X, Y, sw.dx, sw.dy, obstacle)
    ov[:, 2] = 0.08  # raise the obstacle footprint above the water plane so it reads as a solid

    t_end = 1.2
    times = np.linspace(0.0, t_end, n_snapshots)
    depth = np.empty((n_snapshots, len(v)), dtype=np.float32)
    speed = np.empty((n_snapshots, len(v)), dtype=np.float32)
    for k, t in enumerate(times):
        if t > 0:
            sw.run(t)
        depth[k] = sw.h[oi, oj]
        sp = np.hypot(sw.hu, sw.hv) / np.maximum(sw.h, 1e-6)
        speed[k] = np.where(sw.h[oi, oj] > 1e-6, sp[oi, oj], 0.0)

    rep = sw.health_report()
    assert rep["ok"], f"solver health check failed: {rep}"  # never export a run that isn't clean

    sc = Scene("Shallow-water dam break (demo)", length_unit="m", times=times, time_unit="s",
               source="pinneapple_simulation.numerical_solvers.shallow_water_fv (MUSCL-HLL, "
                      "validated against Ritter/Stoker dam breaks -- see tests/test_shallow_water_fv.py)")
    sc.add_part("water", v, f, group="fluid", color=(0.15, 0.45, 0.85))
    sc.add_field("water", "depth_m", depth, unit="m")
    sc.add_field("water", "speed_m_s", speed, unit="m/s")
    sc.add_part("obstacle", ov, of, group="solid", color=(0.55, 0.4, 0.3))
    return sc


def main():
    sc = build_scene()
    sc.export(OUT_DIR, with_viewer=True)
    print(f"Exported {len(sc.parts)} parts, {len(sc.times)} time steps -> {OUT_DIR}")
    serve(OUT_DIR)  # blocks; prints the URL to open (Ctrl+C to stop)


if __name__ == "__main__":
    main()
