"""PyVista bridge: turn PINNeAPPle meshes and fields into publication-quality 3-D images.

Everything that builds a mesh (``calculix_grid``, ``scene_to_multiblock``, ``grid_from_arrays``) is
plain NumPy + PyVista and runs on any machine. Rendering (``render_png``, ``render_gif``) needs an
OpenGL context: a desktop session, or ``xvfb-run`` / OSMesa / EGL on a headless server. Because VTK
**segfaults** (kills the whole Python process) when no context exists, ``can_render`` probes in a
subprocess first and the render functions raise a ``RuntimeError`` with the fix instead.

Install: ``pip install "pinneapple[pyvista]"``. Headless Linux: ``xvfb-run -a python your_script.py``.
"""
from __future__ import annotations

import functools
import subprocess
import sys
from typing import Dict, Optional, Sequence

import numpy as np

from .theme import viewport as _viewport

# VTK cell types for the CalculiX/Abaqus element families (node order is identical, see tests)
_VTK_CELL = {"C3D4": 10, "C3D10": 24, "C3D8": 12, "C3D8R": 12, "C3D8I": 12, "C3D20": 25, "C3D20R": 25}

_INSTALL_HINT = 'PyVista is not installed: pip install "pinneapple[pyvista]"'

# The plot background is the house theme's, the same colour the studio and twin3d viewers draw on (#65).
BACKGROUND = _viewport(dark=True)


def _pv():
    try:
        import pyvista
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(_INSTALL_HINT) from exc
    return pyvista


def von_mises(stress6: np.ndarray) -> np.ndarray:
    """Von Mises equivalent stress from CalculiX components ``[SXX, SYY, SZZ, SXY, SYZ, SZX]``."""
    s = np.asarray(stress6, dtype=float)
    if s.ndim != 2 or s.shape[1] != 6:
        raise ValueError(f"stress must be (N, 6), got {s.shape}")
    sx, sy, sz, txy, tyz, tzx = s.T
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3.0 * (txy ** 2 + tyz ** 2 + tzx ** 2))


def grid_from_arrays(points: np.ndarray, cells: np.ndarray, vtk_cell_type: int,
                     point_data: Optional[Dict[str, np.ndarray]] = None):
    """Unstructured grid from ``points (N, 3)`` and ``cells (M, k)`` (0-based point indices)."""
    pv = _pv()
    points = np.asarray(points, dtype=float)
    cells = np.asarray(cells, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}")
    if cells.ndim != 2:
        raise ValueError(f"cells must be (M, k), got {cells.shape}")
    if cells.size and (cells.min() < 0 or cells.max() >= len(points)):
        raise ValueError("cell index out of range")
    flat = np.hstack([np.full((len(cells), 1), cells.shape[1], dtype=np.int64), cells]).ravel()
    grid = pv.UnstructuredGrid(flat, np.full(len(cells), vtk_cell_type, dtype=np.uint8), points)
    for name, values in (point_data or {}).items():
        values = np.asarray(values)
        if len(values) != len(points):
            raise ValueError(f"point field '{name}' has {len(values)} values for {len(points)} points")
        grid.point_data[name] = values
    return grid


def calculix_grid(model, frd=None, step: int = -1, warp: Optional[float] = None):
    """PyVista grid of a CalculiX model, with ``DISP``, ``|U|``, ``STRESS`` and ``VON_MISES`` if present.

    ``model`` is a ``CalculixModel`` (from ``read_inp``), ``frd`` an optional ``FrdResults``. ``warp`` is a
    displacement scale factor: the returned grid is deformed by ``warp * DISP`` (use e.g. 50 to make a
    millimetre deflection visible; say so in the figure caption, it is not to scale).
    """
    etype = model.element_type.upper()
    if etype not in _VTK_CELL:
        raise ValueError(f"unsupported element type {model.element_type!r}; supported: {sorted(_VTK_CELL)}")
    ids, xyz = model.node_array()
    index = {int(n): i for i, n in enumerate(ids)}
    cells = np.array([[index[int(n)] for n in model.elements[e]] for e in sorted(model.elements)], dtype=np.int64)
    data: Dict[str, np.ndarray] = {}
    if frd is not None:
        order = {int(n): i for i, n in enumerate(frd.node_ids)}
        missing = [n for n in ids if int(n) not in order]
        if missing:
            raise ValueError(f"{len(missing)} model nodes are not in the results (first: {missing[0]})")
        rows = np.array([order[int(n)] for n in ids])
        res = frd.steps[step]
        for name, arr in res.items():
            arr = np.asarray(arr)[rows]
            key = {"DISP": "DISP", "U": "DISP", "STRESS": "STRESS", "S": "STRESS"}.get(name.upper(), name)
            data[key] = arr
        if "DISP" in data:
            data["|U|"] = np.linalg.norm(data["DISP"], axis=1)
        if "STRESS" in data and data["STRESS"].shape[1] == 6:
            data["VON_MISES"] = von_mises(data["STRESS"])
    grid = grid_from_arrays(xyz, cells, _VTK_CELL[etype], data)
    if warp is not None:
        if "DISP" not in data:
            raise ValueError("warp needs displacements: pass frd results with a DISP block")
        grid.points = grid.points + float(warp) * data["DISP"]
    return grid


def scene_to_multiblock(scene, time_index: int = 0, field_name: Optional[str] = None):
    """``pinneapple_twin3d.Scene`` to a ``pyvista.MultiBlock`` (one surface per part).

    Fields of shape ``(T, V)`` are sliced at ``time_index``; ``(V,)`` fields are used as is.
    ``field_name`` keeps only that field on every part that has it.
    """
    pv = _pv()
    blocks = pv.MultiBlock()
    for part in scene.parts:
        faces = np.hstack([np.full((len(part.faces), 1), 3), part.faces.astype(np.int64)]).ravel()
        mesh = pv.PolyData(part.vertices.astype(float), faces)
        for name, values in part.fields.items():
            if field_name is not None and name != field_name:
                continue
            values = np.asarray(values)
            mesh.point_data[name] = values[time_index] if values.ndim == 2 else values
        blocks[part.name] = mesh
    return blocks


def export_vtk(mesh, path: str) -> str:
    """Write ``.vtu`` / ``.vtp`` / ``.vtk`` / ``.vtm`` (ParaView-ready); format chosen by extension."""
    mesh.save(path)
    return path


@functools.lru_cache(maxsize=1)
def can_render() -> bool:
    """True if an off-screen OpenGL context can be created here. Probed in a subprocess (VTK crashes the
    interpreter instead of raising when there is none)."""
    code = ("import pyvista as pv\n"
            "pl = pv.Plotter(off_screen=True); pl.add_mesh(pv.Sphere()); pl.screenshot(return_img=True)\n")
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=120)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return r.returncode == 0


def _require_rendering() -> None:
    if not can_render():
        raise RuntimeError(
            "No off-screen OpenGL context available. On a headless Linux server run the script under "
            "`xvfb-run -a python your_script.py` (apt install xvfb), or install libosmesa6 / libegl1.")


def _plotter(mesh, scalars, cmap, title, show_edges, window_size, background, scalar_bar_title, clim):
    pv = _pv()
    pl = pv.Plotter(off_screen=True, window_size=list(window_size))
    pl.set_background(background)
    kw = dict(show_edges=show_edges, smooth_shading=not show_edges)
    if scalars is not None:
        kw.update(scalars=scalars, cmap=cmap, clim=clim,
                  scalar_bar_args=dict(title=scalar_bar_title or scalars, n_labels=4, fmt="%.3g",
                                       color="white" if _is_dark(background) else "black"))
    pl.add_mesh(mesh, **kw)
    if title:
        pl.add_text(title, font_size=12, color="white" if _is_dark(background) else "black")
    pl.enable_anti_aliasing("ssaa")
    return pl


def _is_dark(background: str) -> bool:
    r, g, b = _pv().Color(background).float_rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5


def render_png(mesh, path: str, *, scalars: Optional[str] = None, cmap: str = "turbo", title: str = "",
               show_edges: bool = False, window_size: Sequence[int] = (1400, 900), background: str = BACKGROUND,
               scalar_bar_title: Optional[str] = None, clim: Optional[Sequence[float]] = None,
               camera_position: Optional[str] = "iso") -> str:
    """Render ``mesh`` (grid, PolyData or MultiBlock) to a PNG. ``scalars`` is a point/cell array name."""
    _require_rendering()
    pl = _plotter(mesh, scalars, cmap, title, show_edges, window_size, background, scalar_bar_title, clim)
    if camera_position:
        pl.camera_position = camera_position
    pl.screenshot(path)
    pl.close()
    return path


def render_gif(mesh, path: str, *, scalars: Optional[str] = None, cmap: str = "turbo", title: str = "",
               n_frames: int = 60, elevation: float = 25.0, window_size: Sequence[int] = (1000, 700),
               background: str = BACKGROUND, clim: Optional[Sequence[float]] = None, fps: int = 20) -> str:
    """Orbit camera around ``mesh`` and write an animated GIF (turntable)."""
    _require_rendering()
    pl = _plotter(mesh, scalars, cmap, title, False, window_size, background, None, clim)
    pl.camera_position = "iso"
    path_ = pl.generate_orbital_path(n_points=n_frames, factor=2.0, shift=0.0, viewup=[0, 0, 1])
    pl.open_gif(path, fps=fps)
    pl.orbit_on_path(path_, write_frames=True, viewup=[0, 0, 1], step=0.0)
    pl.close()
    return path
