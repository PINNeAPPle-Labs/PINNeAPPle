"""The same twin, at product quality: a Cycles turntable rendered by the library's own Blender path.

    python examples/unity/05_blender_turntable.py                 # needs bpy:  pip install bpy
    python examples/unity/05_blender_turntable.py --frames 72 --samples 16

Why this exists next to the matplotlib clips (make_movies.py, 04_surrogate_turntable.py): those show the data with
no dependencies and no GPU, and look like plots. This one uses the renderer the repository already ships for
Blender -- `pinneapple_tools.visualization.studio.blender.render` with ``background="dark"`` (key, rim and fill area
lights, AgX, a glossy floor, Cycles with denoising) -- and the camera orbits the part, so the twin looks like a
product shot instead of a chart. Both are the same field on the same geometry; only the renderer changes.

The field comes from the surrogate trained in ``04_surrogate_turntable.py`` when its weights are there, otherwise
from the twin's own time steps: either way each frame is one evaluation, not a baked step.

On a machine without a GPU Cycles runs on the CPU: a 1280x720 frame at 16 samples takes seconds, so the defaults
here are short on purpose (4 s). Raise ``--frames`` and ``--samples`` for a final clip.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(HERE, "media")
TWIN = os.path.join(HERE, "_out", "spool_unity")
WEIGHTS = os.path.join(HERE, "_out", "surrogate.pt")


def _surrogate():
    """The predictor of demo 04 (and the field range it was trained for), or None."""
    if not os.path.exists(WEIGHTS):
        return None, None
    spec = importlib.util.spec_from_file_location(
        "unity_demo_04", os.path.join(HERE, "04_surrogate_turntable.py"))
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    _model, predict = demo.load_surrogate(WEIGHTS)
    return predict, (demo.T_COLD, demo.T_HOT, demo.T_MIN, demo.T_END)


def turntable(frames: int, fps: int, samples: int, size: tuple[int, int], still: str | None) -> str:
    from pinneapple_tools.visualization.studio.blender import render
    from pinneapple_unity import scene_from_twin

    twin = scene_from_twin(TWIN)
    studio = twin.to_studio(0)                       # z-up; the surfaces carry their fields
    spool = next(s for s in studio.surfaces if s.name == "spool")
    predict, limits = _surrogate()
    if predict is None:
        print(f"no surrogate at {os.path.relpath(WEIGHTS, HERE)}: using the twin's own steps "
              f"(run 04_surrogate_turntable.py for the continuous field)")
        limits = (float(np.nanmin(spool.fields["temperature"])), float(np.nanmax(spool.fields["temperature"])),
                  float(twin.times[0]), float(twin.times[-1]))
    lo, hi, t_min, t_end = limits
    os.makedirs(MEDIA, exist_ok=True)
    out_dir = os.path.join(tempfile.gettempdir(), "pinneapple_cycles")
    os.makedirs(out_dir, exist_ok=True)
    x = spool.vertices[:, 0]

    for k in range(frames):
        u = k / max(1, frames - 1)
        t = t_min + (t_end - t_min) * u
        if predict is not None:
            spool.fields["temperature"] = predict(x, np.full(len(x), t))
        else:
            step = min(len(twin.times) - 1, int(round(u * (len(twin.times) - 1))))
            spool.fields["temperature"] = twin.parts[0].fields["temperature"][step if
                                               twin.parts[0].fields["temperature"].shape[0] > 1 else 0]
        angle = 2.0 * np.pi * u + 0.7          # start at three-quarters, not end-on
        view = [float(np.cos(angle)), float(np.sin(angle)), 0.5]
        path = os.path.join(out_dir, f"f{k:04d}.png")
        render(studio, path, field="temperature", field_range=(lo, hi), view=view,
               background="dark", samples=samples, size=size, floor=True, cmap="jet",
               distance=1.02, title=f"temperature [C]   t = {t / 60:.1f} min")
        if k % 12 == 0:
            print(f"  frame {k + 1}/{frames}", flush=True)
    dest = os.path.join(MEDIA, "05_blender_turntable.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", os.path.join(out_dir, "f%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", dest], check=True)
    print(f"{os.path.relpath(dest, HERE)}: {frames} frames, {frames / fps:.1f} s at {fps} fps, "
          f"{os.path.getsize(dest) // 1024} kB")
    if still:
        spool.fields["temperature"] = (predict(x, np.full(len(x), 0.55 * (t_end - t_min) + t_min))
                                       if predict is not None else spool.fields["temperature"])
        render(studio, os.path.join(MEDIA, still), field="temperature", field_range=(lo, hi),
               view=[0.75, 0.55, 0.5], distance=1.1, background="dark", samples=max(samples, 128),
               size=(1600, 900), floor=True, cmap="jet", title="temperature [C]")
        print(f"{os.path.join(MEDIA, still)} (still, {max(samples, 128)} samples)")
    for name in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, name))
    os.rmdir(out_dir)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description="Cycles turntable of the twin (needs bpy)")
    ap.add_argument("--frames", type=int, default=96, help="24 fps x 4 s")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--size", type=int, nargs=2, default=(1280, 720))
    ap.add_argument("--still", default="05_blender_still.png", help="also render one high-sample still")
    ap.add_argument("--no-still", action="store_true")
    a = ap.parse_args()
    try:
        import bpy  # noqa: F401
    except ImportError:
        print("bpy is not installed: pip install bpy   (about 220 MB; Cycles then runs on the CPU)")
        return 1
    if not os.path.isdir(TWIN):
        print(f"{TWIN} is missing: run 01_engineering_twin_to_unity.py first")
        return 1
    turntable(a.frames, a.fps, a.samples, tuple(a.size), None if a.no_still else a.still)
    return 0


if __name__ == "__main__":
    sys.exit(main())
