"""Record the demos as short MP4s (matplotlib + ffmpeg), from the same twin data the Unity runtime reads.

    python examples/unity/make_movies.py                 # builds the twins if needed, writes media/*.mp4
    python examples/unity/make_movies.py --no-build      # only re-render what is already in _out/

This is demo material, not the library: it renders the exported twin (vertices, fields, colormap, legend, sensors)
with matplotlib and lets ffmpeg assemble the frames, so a machine without Unity and without a GPU can still show
what the integration produces. What it does *not* record is the Unity Play mode -- that needs a licensed editor and
the Unity Recorder package; the numbers and the colours it shows are the same ones, on the same geometry.

Files land in ``examples/unity/media/`` (tracked) and are the three videos the example README links.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from pinneapple_unity import scene_from_twin

HERE = os.path.dirname(os.path.abspath(__file__))
TWINS = os.path.join(HERE, "_out")
MEDIA = os.path.join(HERE, "media")
LIGHT = np.array([0.35, -0.45, 0.82])
LIGHT = LIGHT / np.linalg.norm(LIGHT)

#: What to record: (twin folder, field, movie, colormap, camera azimuth, zoom). The demo that writes each folder:
SOURCES = {
    "spool_unity": "01_engineering_twin_to_unity.py",
    "bracket_twin": "02_unity_to_pinneapple_to_openusd.py",
}
MOVIES = [
    ("spool_unity", "temperature", "01_spool_temperature.mp4", "turbo", -62, 1.6),
    ("bracket_twin", "temperature", "02_bracket_temperature.mp4", "turbo", -72, 1.45),
    ("bracket_twin", "bending_stress", "03_bracket_bending_stress.mp4", "viridis", -72, 1.45),
]


def _faces(vertices, faces, values, cmap, norm, grey=None):
    """A Poly3DCollection shaded flat: colour from the mean of the triangle's vertex values (or a grey part)."""
    tri = vertices[faces]
    if grey is not None:
        colors = np.tile(np.append(grey, 1.0), (len(tri), 1))
    else:
        colors = cmap(norm(values[faces].mean(axis=1)))
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    colors[:, :3] *= (0.55 + 0.45 * np.abs(normals @ LIGHT))[:, None]
    return Poly3DCollection(tri, facecolors=colors, edgecolors="none", linewidths=0)


def render(folder: str, field: str, movie: str, cmap_name: str = "turbo", azim: float = -62,
           zoom: float = 1.45, *, fps: int = 24, hold: int = 12) -> str:
    """One field of one twin, one frame per hold frame of each time step, into ``media/<movie>``."""
    scene = scene_from_twin(folder)
    times = list(scene.times) if scene.times is not None else [0.0]
    vals = [p.fields[field] for p in scene.parts if field in p.fields]
    lo, hi = float(min(np.nanmin(v) for v in vals)), float(max(np.nanmax(v) for v in vals))
    unit = scene.field_units.get(field, "")
    cmap, norm = plt.get_cmap(cmap_name), Normalize(lo, hi)
    allv = np.concatenate([p.vertices for p in scene.parts])
    mn, mx = allv.min(0), allv.max(0)
    frames = os.path.join(MEDIA, "_frames_" + os.path.splitext(movie)[0])
    os.makedirs(frames, exist_ok=True)
    k = 0
    for step in range(len(times)):
        for _ in range(hold):
            fig = plt.figure(figsize=(9, 6.2), dpi=110)
            ax = fig.add_subplot(111, projection="3d")
            for part in scene.parts:
                if field in part.fields:
                    a = part.fields[field]
                    coll = _faces(part.vertices, part.faces, a[step if a.shape[0] > 1 else 0], cmap, norm)
                else:
                    coll = _faces(part.vertices, part.faces, None, cmap, norm, grey=(0.62, 0.65, 0.70))
                ax.add_collection3d(coll)
            if scene.sensors:
                pos = np.array([s.position for s in scene.sensors])
                ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c="gold", s=45, depthshade=False,
                           edgecolors="black", linewidths=0.4)
                for sensor, p in zip(scene.sensors, pos, strict=False):
                    text = sensor.id
                    if sensor.series is not None and len(sensor.series) > step:
                        text += f" {sensor.series[step]:.0f} {sensor.unit}"
                    ax.text(p[0], p[1], p[2] + float((mx - mn).max()) * 0.06, text, fontsize=8)
            ax.set_xlim(mn[0], mx[0])
            ax.set_ylim(mn[1], mx[1])
            ax.set_zlim(mn[2], mx[2])
            ax.set_box_aspect(mx - mn, zoom=zoom)
            ax.view_init(elev=22, azim=azim)
            ax.set_axis_off()
            when = f"   t = {times[step]:g} {scene.time_unit}" if len(times) > 1 else ""
            ax.set_title(f"{scene.title}\n{field} [{unit}]   step {step + 1}/{len(times)}{when}", fontsize=10)
            cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6, pad=0.0)
            cb.set_label(f"{field} [{unit}]", fontsize=9)
            fig.subplots_adjust(left=0.0, right=0.88, top=0.88, bottom=0.0)
            fig.savefig(os.path.join(frames, f"f{k:04d}.png"))
            plt.close(fig)
            k += 1
    dest = os.path.join(MEDIA, movie)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", os.path.join(frames, "f%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", dest], check=True)
    for name in os.listdir(frames):
        os.remove(os.path.join(frames, name))
    os.rmdir(frames)
    print(f"{movie}: {k} frames, {k / fps:.1f} s, {os.path.getsize(dest) // 1024} kB "
          f"(range [{lo:.4g}, {hi:.4g}] {unit})")
    return dest


def build(folder_name: str) -> str:
    """The twin folder of a demo, running that demo when it is not there yet."""
    folder = os.path.join(TWINS, folder_name)
    if os.path.isdir(folder):
        return folder
    demo = SOURCES[folder_name]
    print(f"{os.path.basename(folder)} is not there yet: running {demo} (about a minute)")
    done = subprocess.run([sys.executable, os.path.join(HERE, demo)], capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"{demo} failed:\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}")
    return folder


def main() -> int:
    ap = argparse.ArgumentParser(description="record the Unity demos as MP4s (matplotlib + ffmpeg)")
    ap.add_argument("--no-build", action="store_true", help="do not run the demos, render only what exists")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--hold", type=int, default=12, help="frames per time step")
    a = ap.parse_args()
    if not MEDIA:
        return 1
    os.makedirs(MEDIA, exist_ok=True)
    written = []
    for folder_name, field, movie, cmap, azim, zoom in MOVIES:
        folder = os.path.join(TWINS, folder_name)
        if not os.path.isdir(folder):
            if a.no_build:
                print(f"skipping {movie}: {folder} is missing (run the demo, or drop --no-build)")
                continue
            folder = build(folder_name)
        written.append(render(folder, field, movie, cmap, azim, zoom, fps=a.fps, hold=a.hold))
    for path in written:
        print(path)
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
