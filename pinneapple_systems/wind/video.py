"""Frame-by-frame film of a wind-farm twin: the wake footprint sweeping with direction.

The scene is built once by :mod:`pinneapple_systems.wind.twin` and this module
rasterises it: the ground field it carries, the layout geometry the scene was
built from, and the per-turbine power the sensors recorded. Nothing is recomputed
here -- what you see is what the twin holds.

Rendering rather than screen capture is deliberate: the result is reproducible
from the committed code, it needs no display, and a reviewer can regenerate it
with one command.

Output is H.264 in an MP4 container, which is the only shape that plays in a
browser and in a slide deck alike. ``ffmpeg`` must be on the PATH.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

__all__ = ["render_farm_video", "render_frame"]


def _decode_rgb(hex_colour: str) -> np.ndarray:
    hex_colour = hex_colour.lstrip("#")
    return np.array([int(hex_colour[i:i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def _build_colormap():
    from matplotlib.colors import LinearSegmentedColormap

    # Ratio 1.0 = undisturbed (paper), 0.6 = heavily waked (warm). A perceptual
    # ramp would hide the structure at these low deficits; two tones do not.
    return LinearSegmentedColormap.from_list(
        "wake", ["#6d1200", "#b8481a", "#dd9a35", "#efdfb8", "#e6e2d8"]
    )


def render_frame(
    scene,
    layout,
    step: int,
    *,
    cmap=None,
    vmin: float = 0.6,
    vmax: float = 1.0,
    elev: float = 34.0,
    azim: float = -62.0,
    title: str = "",
    subtitle: str = "",
    wind_speed: float | None = None,
):
    """One frame: the ground field at ``step``, the turbines, and the farm summary."""
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.colors import Normalize

    cmap = cmap or _build_colormap()
    ground = next(p for p in scene.parts if p.name == "ground")
    field = ground.fields["velocity_ratio"]
    ratio = field[step]
    vertices = ground.vertices
    faces = ground.faces

    directions = np.asarray(scene.times, dtype=float)
    direction = float(directions[step])

    sensors = [s for s in scene.sensors if s.series is not None]
    total_kw = float(sum(s.series[step] for s in sensors)) if sensors else None
    series = np.array([s.series for s in sensors]) if sensors else None

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("#f7f6f3")
    grid = gridspec.GridSpec(1, 2, width_ratios=(2.35, 1), wspace=0.02,
                             left=0.02, right=0.985, top=0.90, bottom=0.06)
    ax = fig.add_subplot(grid[0, 0], projection="3d")
    ax.set_facecolor("#f7f6f3")

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    tris = vertices[faces]
    colours = cmap(Normalize(vmin=vmin, vmax=vmax)(ratio[faces].mean(axis=1)))
    poly = Poly3DCollection(tris, facecolors=colours, edgecolors="none", linewidths=0, shade=False)
    ax.add_collection3d(poly)

    lo = vertices[:, :2].min(axis=0)
    hi = vertices[:, :2].max(axis=0)
    span = float(np.max(hi - lo))
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(0.0, 140.0)
    ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], 0.22 * span), zoom=1.35)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    ux, uy = -np.sin(np.radians(direction)), -np.cos(np.radians(direction))
    anchor = np.array([lo[0] + 0.10 * (hi[0] - lo[0]), lo[1] + 0.10 * (hi[1] - lo[1]), 0.0])
    arrow = 0.18 * span
    ax.quiver(anchor[0], anchor[1], 118.0, ux * arrow, uy * arrow, 0.0,
              color="#004c6d", linewidth=3.0, arrow_length_ratio=0.3)
    ax.text(anchor[0] - ux * 0.05 * span, anchor[1] - uy * 0.05 * span, 122.0, "wind",
            color="#004c6d", fontsize=9, fontweight="bold")

    for turbine in layout.turbines:
        ax.plot([turbine.x, turbine.x], [turbine.y, turbine.y], [0.0, turbine.hub_height],
                color="#2b3136", linewidth=3.2, solid_capstyle="round")
        theta = np.linspace(0, 2 * np.pi, 41)
        radius = turbine.rotor_diameter / 2.0
        ax.plot(turbine.x + radius * np.cos(theta), turbine.y + radius * np.sin(theta),
                np.full(theta.size, turbine.hub_height), color="#00374f", linewidth=1.6, alpha=0.95)

    # -- right panel: the farm against direction ---------------------------
    ax2 = fig.add_subplot(grid[0, 1])
    ax2.set_facecolor("#f7f6f3")
    if series is not None:
        farm_kw = series.sum(axis=0)
        ax2.plot(directions, farm_kw / 1000.0, color="#004c6d", linewidth=2.0)
        ax2.plot([direction], [farm_kw[step] / 1000.0], "o", color="#a23a00", markersize=7, zorder=5)
        ax2.set_ylim(0, max(farm_kw / 1000.0) * 1.12)
    ax2.set_xlim(0, 360)
    ax2.set_xticks([0, 90, 180, 270, 360])
    ax2.set_xlabel("wind direction the wind comes from [deg]", fontsize=9)
    ax2.set_ylabel("farm power [MW]", fontsize=9)
    ax2.tick_params(labelsize=8)
    ax2.grid(alpha=0.25)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    header = title or scene.title
    fig.text(0.02, 0.955, header, fontsize=15, fontweight="bold", color="#16191c", family="DejaVu Sans")
    line = f"direction {direction:.0f}°"
    if wind_speed is not None:
        line += f"   ·   free stream {wind_speed:g} m/s"
    if total_kw is not None:
        line += f"   ·   farm {total_kw / 1000.0:.1f} MW"
    fig.text(0.02, 0.912, line, fontsize=10, color="#545c63")
    if subtitle:
        fig.text(0.02, 0.012, subtitle, fontsize=8.5, color="#7a8087")
    return fig


def render_farm_video(
    scene,
    layout,
    path,
    *,
    fps: int = 20,
    wind_speed: float | None = None,
    title: str = "",
    subtitle: str = "",
    hold_frames: int = 18,
    crf: int = 20,
) -> dict:
    """Render every scene step to a frame and encode the MP4.

    ``hold_frames`` freezes the first step so the opened video does not start
    mid-motion, and an equal number freezes the last so the sweep reads as a
    cycle rather than a cut.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not on the PATH; install it to encode the demo video")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_steps = len(scene.times)
    cmap = _build_colormap()

    with tempfile.TemporaryDirectory(prefix="pp-wind-frames-") as tmp:
        tmp = Path(tmp)
        index = 0
        for k in range(n_steps):
            fig = render_frame(scene, layout, k, cmap=cmap, title=title, subtitle=subtitle,
                               wind_speed=wind_speed)
            repeats = hold_frames if k == 0 else (hold_frames if k == n_steps - 1 else 1)
            for _ in range(repeats):
                fig.savefig(tmp / f"frame_{index:04d}.png", facecolor=fig.get_facecolor())
                index += 1
            plt.close(fig)
        out = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
             "-i", str(tmp / "frame_%04d.png"),
             "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            capture_output=True, text=True, check=False,
        )
        if out.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {out.stderr.strip()[:500]}")
        frames = index
    return {
        "path": str(path),
        "frames": frames,
        "fps": fps,
        "seconds": frames / fps,
        "size_mb": path.stat().st_size / 1e6,
    }
