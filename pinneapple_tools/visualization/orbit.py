"""Turntable animation of a field: the camera orbits while the field advances in time (#64).

    import numpy as np
    import pinneapple as pp

    t = np.linspace(0.0, 2.0, 40)                     # 40 instants
    x = y = np.linspace(0.0, 1.0, 60)
    X, Y = np.meshgrid(x, y, indexing="ij")
    field = np.sin(2 * np.pi * (X[None] - t[:, None, None])) * np.cos(2 * np.pi * Y[None])

    pp.viz.turntable(field, "case.gif", t=t, name="u", unit="m/s")    # 1080p GIF, one call
    pp.viz.turntable(field, "case.mp4", t=t, unit="m/s")              # the same as MP4

The frame carries a discreet mark and a caption with the unit and the time, so a clip lifted out of a report still
says what it is. Nothing here needs a display, a GPU or pyvista: the frames are drawn with matplotlib's Agg canvas,
and only the MP4 path needs the ``ffmpeg`` binary on ``PATH`` (the GIF path needs only Pillow).
"""
from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .style import DEFAULT_CMAP, get_cmap

DEFAULT_SIZE = (1920, 1080)
# Degrees the camera sweeps over the clip (one full turn by default) and the azimuth it starts at.
DEFAULT_ORBIT = 360.0
DEFAULT_AZIMUTH = 20.0


def _as_array(a) -> np.ndarray:
    """Array from anything array-like, including a torch tensor that needs detaching first."""
    try:
        return a.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(a)


def _trajectory(field) -> np.ndarray:
    """``field`` as ``(nt, nx, ny)`` floats, checked: a turntable needs a field that moves."""
    traj = _as_array(field).astype(float)
    if traj.ndim != 3:
        raise ValueError(f"field must be (nt, nx, ny) -- a stack of 2-D fields, one per instant; got shape "
                         f"{traj.shape}. For a single field, pass it as field[None].")
    nt, nx, ny = traj.shape
    if nt < 2:
        raise ValueError("field has one instant: a turntable needs at least two frames to move in time")
    if nx < 2 or ny < 2:
        raise ValueError(f"field is {nx}x{ny} in space; each frame needs at least a 2x2 grid")
    if not np.isfinite(traj).all():
        raise ValueError("field has non-finite values (nan/inf); the colour scale and the relief need numbers")
    return traj


def _resample(n: int, frames: int | None) -> np.ndarray:
    """The indices of the frames to draw: all of them, or ``frames`` of them spread evenly over the run.

    Asking for more frames than there are instants repeats the nearest instant, which is what makes a smooth
    orbit out of a handful of time samples: the camera keeps turning on the instants it has.
    """
    if frames is None:
        return np.arange(n)
    frames = int(frames)
    if frames < 2:
        raise ValueError(f"frames must be at least 2 (a turntable moves); got {frames}")
    return np.linspace(0, n - 1, frames).round().astype(int)


def _pixel_size(size: Sequence[int]) -> tuple[int, int]:
    w, h = (int(v) for v in size)
    if w < 64 or h < 64:
        raise ValueError(f"size must be at least 64x64 pixels; got {w}x{h}")
    return w, h


def _light(shape: tuple[int, int]):
    """Bake a fixed light into the face colours so the relief reads as a surface while the camera orbits."""
    from matplotlib.colors import LightSource

    return LightSource(azdeg=225.0, altdeg=45.0) if shape[0] > 2 and shape[1] > 2 else None


def turntable(field, path: str | Path | None = None, *, x=None, y=None, t=None, name: str = "u",
              unit: str = "", time_unit: str = "s", size: Sequence[int] = DEFAULT_SIZE, dpi: int = 100,
              frames: int | None = None, fps: float = 24.0, cmap: str = DEFAULT_CMAP, clim=None,
              deform: float = 0.25, elevation: float = 28.0, orbit: float = DEFAULT_ORBIT,
              azimuth: float = DEFAULT_AZIMUTH, zoom: float = 1.25, background: str = "#0b1220",
              mark: str = "PINNeAPPle", mark_alpha: float = 0.45, caption: bool = True,
              colorbar: bool = True, colors: int = 255):
    """Write a turntable animation of ``field`` to ``path`` (``.gif`` or ``.mp4``), or return its frames.

    The camera orbits by ``orbit`` degrees while the field advances from the first instant to the last, so both
    the time and the geometry are motion. Every frame is drawn at ``size`` (1080p by default) and carries the
    mark and a caption naming the field, its unit and the instant -- a clip that leaves the report still says
    what it shows.

    Parameters
    ----------
    field : (nt, nx, ny) array-like
        The field at each instant. A single field must be passed as ``field[None]``; a callable or a moving mesh
        is not a turntable input, it is a rendering of one.
    path : str or Path, optional
        ``.gif`` (Pillow) or ``.mp4`` (needs ``ffmpeg`` on ``PATH``). When omitted the frames are returned as a
        list of ``(h, w, 3)`` ``uint8`` arrays instead of being encoded, which is what a caller that wants to
        compose them somewhere else needs.
    x, y : array-like, optional
        1-D coordinates of the grid axes, ``nx`` and ``ny`` long. Default: the unit square.
    t : array-like, optional
        The physical time of each instant, ``nt`` long, shown in the caption. Default: the frame index.
    name, unit, time_unit : str
        What the caption says: ``"u [m/s] at t = 1.2 s"`` for ``name="u"``, ``unit="m/s"``.
    size : (int, int)
        Pixels per frame, ``(width, height)``. 1080p by default.
    frames : int, optional
        Draw this many frames, spread evenly over the run, instead of one per instant. Two frames at least:
        with one there is nothing to animate.
    fps : float
        Playback rate of the GIF or MP4.
    cmap : str
        A name from :data:`~pinneapple_tools.visualization.style.CMAPS`, or any matplotlib colour map.
    clim : (float, float), optional
        Colour limits, held fixed for the whole clip. Default: the range of the whole field, so a colour means
        the same thing in every frame.
    deform : float
        Relief height as a fraction of the width of the domain (``0`` leaves the plane flat). The field drives
        the height and the colour; the caption and the colour bar stay readable either way.
    elevation : float
        Camera elevation in degrees.
    orbit : float
        Degrees the camera sweeps over the clip, spread over the frames drawn. The last frame therefore lands
        one step short of closing the turn, which is what a looping GIF needs: end on the first frame again and
        the loop stutters. ``0`` holds the camera still and lets the time be the only motion.
    azimuth : float
        Azimuth of the first frame, in degrees.
    zoom : float
        How much of the frame the surface fills. ``1.0`` leaves matplotlib's generous default margin; much more
        than about 1.6 clips the surface at the edges as the camera turns.
    background : str
        Figure colour behind the surface.
    mark : str
        The discreet mark in the corner; ``""`` draws none.
    mark_alpha : float
        Its opacity: the point of a discreet mark is that it is not the subject.
    caption, colorbar : bool
        Turn the field/unit/time caption and the colour bar off for a clip that will be annotated elsewhere.
    colors : int
        Palette size of the GIF.

    Returns
    -------
    Path, or list of numpy arrays
        The written file, or the frames when ``path`` is omitted.

    Raises
    ------
    ValueError
        On a field that is not ``(nt, nx, ny)``, a single instant, a non-finite value, a size under 64 pixels,
        a frame count under 2, an unknown suffix, or ``clim`` that is reversed or empty.
    RuntimeError
        When an MP4 is asked for and ``ffmpeg`` is not on ``PATH``.
    """
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    traj = _trajectory(field)
    nt, nx, ny = traj.shape
    idx = _resample(nt, frames)
    width, height = _pixel_size(size)

    xs = np.linspace(0.0, 1.0, nx) if x is None else _as_array(x).astype(float)
    ys = np.linspace(0.0, 1.0, ny) if y is None else _as_array(y).astype(float)
    if xs.shape != (nx,) or ys.shape != (ny,):
        raise ValueError(f"x must be {nx} long and y {ny} long to match the field grid {nx}x{ny}; "
                         f"got {xs.shape} and {ys.shape}")
    times = None if t is None else _as_array(t).astype(float).ravel()
    if times is not None and times.shape != (nt,):
        raise ValueError(f"t must be {nt} long, one value per instant; got {times.shape}")

    if clim is None:
        lo, hi = float(traj.min()), float(traj.max())
    else:
        lo, hi = (float(v) for v in clim)
    if not hi > lo:
        raise ValueError(f"clim must be (low, high) with high > low; got ({lo}, {hi})")
    norm = mcolors.Normalize(lo, hi)
    cm = get_cmap(cmap)                                     # raises on a name that is not a colour map

    X, Y = np.meshgrid(xs, ys, indexing="ij")
    dx, dy = float(xs[-1] - xs[0]), float(ys[-1] - ys[0])
    dz = max(float(deform) * dx, 1e-3 * dx)
    scale = width / 1920.0                                  # keep the type the same size on any canvas
    light = _light((nx, ny))

    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=background)
    canvas = FigureCanvasAgg(fig)
    ax_rect = [0.0, 0.0, 0.88, 1.0] if colorbar else [0.0, 0.0, 1.0, 1.0]
    cbar_label = f"{name} [{unit}]" if unit else name

    out = []
    for j, k in enumerate(idx):
        fig.clf()
        ax = fig.add_axes(ax_rect, projection="3d")
        ax.set_axis_off()
        ax.set_facecolor(background)
        F = traj[k]
        colors_rgba = cm(norm(F))
        if light is not None:
            colors_rgba = light.shade_rgb(colors_rgba, F, blend_mode="soft")
        Z = dz * (norm(F) - 0.5)
        ax.plot_surface(X, Y, Z, facecolors=colors_rgba, rstride=1, cstride=1, linewidth=0,
                        antialiased=False, shade=False)
        ax.view_init(elev=float(elevation), azim=float(azimuth) + float(orbit) * (j / len(idx)))
        ax.set_zlim(-dz / 2.0, dz / 2.0)
        ax.set_box_aspect((dx, dy, dz), zoom=float(zoom))
        if colorbar:
            cax = fig.add_axes([0.905, 0.24, 0.011, 0.52])
            cax.set_facecolor(background)
            mappable = mcm.ScalarMappable(norm=norm, cmap=cm)
            mappable.set_array([])
            cbar = fig.colorbar(mappable, cax=cax)
            cbar.set_label(cbar_label, color="white", fontsize=9 * scale)
            cbar.outline.set_visible(False)
            cax.tick_params(colors="white", labelsize=8 * scale)
            for spine in cax.spines.values():
                spine.set_visible(False)

        if caption:
            label = cbar_label
            if times is not None:
                label += f"   t = {times[k]:.3g} {time_unit}"
            fig.text(0.014, 0.022, label, color="white", alpha=0.92, fontsize=10.5 * scale, va="bottom")
        if mark:
            fig.text(0.986, 0.022, mark, color="white", alpha=float(mark_alpha), fontsize=8.5 * scale,
                     va="bottom", ha="right")

        canvas.draw()
        out.append(np.asarray(canvas.buffer_rgba())[..., :3].copy())

    if path is None:
        return out

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".gif":
        from .dashboard.static import write_gif

        write_gif(str(path), out, fps=float(fps), colors=int(colors))
    elif suffix == ".mp4":
        _write_mp4(path, out, float(fps))
    else:
        raise ValueError(f"write a .gif or a .mp4; got {path.name!r}. A .png holds one instant, which "
                         f"pp.viz.render (studio) is for.")
    return path


def _write_mp4(path: Path, frames: list[np.ndarray], fps: float) -> None:
    """Encode the frames as H.264 by piping raw RGB into ``ffmpeg``; raises when it is not installed."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError(
            "writing an MP4 needs the ffmpeg binary on PATH (macOS: brew install ffmpeg; Debian/Ubuntu: "
            "apt install ffmpeg; Windows: choco install ffmpeg). Without it, write a .gif: that path needs "
            "only Pillow, which matplotlib already brings.")
    height, width = frames[0].shape[:2]
    command = [
        exe, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",         # H.264 4:2:0 needs even dimensions
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None and process.stderr is not None
    try:
        for frame in frames:
            process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        process.stdin.close()
    except (BrokenPipeError, OSError):                      # ffmpeg died early; its stderr says why
        pass
    message = process.stderr.read()                         # until EOF, so this cannot deadlock the pipe
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}: {message.decode('utf-8', 'replace').strip()}")


__all__ = ["turntable", "DEFAULT_SIZE", "DEFAULT_ORBIT", "DEFAULT_AZIMUTH"]
