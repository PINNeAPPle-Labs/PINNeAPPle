"""Blender Cycles clips and stills of the flights, from the runs saved by 03_takeoff.py, 04_turbulence_encounters.py
and 07_same_gust.py (run those first):

* take-off of each aircraft: low chase camera on the runway, motion blur, depth of field, a thin trail coloured by
  the load factor n_z, the lattice loading on the wings, a caption with time, airspeed, height and n_z;
* severe-turbulence encounter of each aircraft: volumetric haze, the frozen gust field as translucent wisps coloured
  by the vertical gust (blue down, red up), the wing loading, an inset sparkline of n_z against time;
* the four aircraft through the same 1-cosine gust: a 2 x 2 clip and a still with one common colour bar.

    python examples/flight_dynamics/05_render_clips.py [--frames 72] [--size 720x405] [--samples 12] [--only a320]
                                                       [--skip takeoff,turbulence,gust] [--keep-in-out]

The committed clips are 720 x 405, H.264 at CRF 27 (each well under 3 MB), the stills 1920 x 1080, in docs/assets/flight_dynamics/clips/;
raise --size / --samples / --frames with --keep-in-out for full-resolution renders (they stay in _out/).
"""
import argparse
import json
import os
import pickle
import shutil

import numpy as np

from pinneapple_simulation.flight import catalog
from pinneapple_simulation.flight.figures import LABELS
from pinneapple_simulation.flight.visuals import flight_scene
from pinneapple_tools.visualization.studio.blender import render_animation, render_still
from pinneapple_tools.visualization.studio.compose import compose_grid, compose_grid_video

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_out")
CLIPS = os.path.join(HERE, "..", "..", "docs", "assets", "flight_dynamics", "clips")

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=72)
ap.add_argument("--size", default="720x405")
ap.add_argument("--samples", type=int, default=12)
ap.add_argument("--only", default=None)
ap.add_argument("--skip", default="")
ap.add_argument("--haze", type=float, default=0.8)
ap.add_argument("--still-samples", type=int, default=48)
ap.add_argument("--keep-in-out", action="store_true", help="do not copy the clips into docs/assets")
a = ap.parse_args()
size = tuple(int(v) for v in a.size.split("x"))
skip = set(a.skip.split(",")) - {""}
os.makedirs(CLIPS, exist_ok=True)


def load(case):
    src = os.path.join(OUT, case, "run.pkl")
    if not os.path.exists(src):
        print("missing", src, "- run the study script first")
        return None
    return pickle.load(open(src, "rb"))


def publish(r, name, sc, opts, still_frame=None):
    """Copy the 720p clip and render a 1080p still of the same scene (frame 2/3 of the way by default)."""
    k = int(sc.animation.n_frames * 2 // 3) if still_frame is None else still_frame
    png = render_still(sc, os.path.join(OUT, f"{name}_still.png"), k, size=(1920, 1080), samples=a.still_samples,
                       **opts)
    if a.keep_in_out:
        return
    if r.get("mp4"):
        shutil.copy(r["mp4"], os.path.join(CLIPS, f"{name}.mp4"))
    shutil.copy(png, os.path.join(CLIPS, f"{name}.png"))


names = [n for n in catalog.names() if not a.only or n == a.only]
for name in names:
    ac = catalog.build(name)
    run = load(f"{name}_takeoff") if "takeoff" not in skip else None
    if run:
        sc = flight_scene(ac, run["hist"], n_frames=a.frames, arrows=False, title=f"{ac.info['type']}: take-off")
        opts = dict(camera="chase_low", motion_blur=True, dof=5.6, trails={"alpha": 0.8}, hud_keys=["V", "h", "nz"])
        r = render_animation(sc, os.path.join(OUT, f"{name}_takeoff", "frames"), size=size, samples=a.samples,
                             fps=12, mp4=True, name=f"{name}_takeoff", **opts)
        publish(r, f"{name}_takeoff", sc, opts)
        print(name, "take-off", r["mp4"], flush=True)
    # the severe run of an aircraft that departs (outside the model) is replaced by its moderate run
    sev = "severe"
    if os.path.exists(os.path.join(OUT, f"{name}_turbulence_moderate", "run.pkl")):
        sev = "moderate"
    run = load(f"{name}_turbulence" + ("_moderate" if sev == "moderate" else "")) if "turbulence" not in skip else None
    if run:
        sc = flight_scene(ac, run["hist"], n_frames=a.frames, wind=run["wind"], runway=False, arrows=False,
                          wisps=True, title=f"{ac.info['type']}: {sev} turbulence")
        opts = dict(camera="chase", haze=a.haze, dof=8.0, trails={"alpha": 0.7}, hud_keys=["V", "nz", "phi"],
                    sparkline="nz")
        r = render_animation(sc, os.path.join(OUT, f"{name}_turbulence", "frames"), size=size, samples=a.samples,
                             fps=12, mp4=True, name=f"{name}_turbulence", **opts)
        publish(r, f"{name}_turbulence", sc, opts)
        print(name, "turbulence", r["mp4"], flush=True)

if "gust" not in skip and not a.only:
    mp4s, stills, labels = [], [], []
    for name in catalog.names():
        ac = catalog.build(name)
        run = load(f"{name}_gust_same")
        if not run:
            continue
        sc = flight_scene(ac, run["hist"], n_frames=120, wind=run["wind"], runway=False, arrows=False, wisps=True,
                          title=f"{ac.info['type']}: 1-cosine gust")
        r = render_animation(sc, os.path.join(OUT, f"{name}_gust_same", "frames"), size=size, samples=a.samples,
                             camera="chase", dof=8.0, trails={"alpha": 0.7}, hud_keys=["V", "nz", "theta"],
                             sparkline="nz", colorbars=False, fps=12, mp4=True, name=f"{name}_gust_same")
        mp4s.append(r["mp4"])
        # the still: the frame where this aircraft meets the gust peak (07_same_gust.py: lead-in + t_peak_s)
        sg = json.load(open(os.path.join(HERE, "results", "same_gust.json")))["aircraft"][name]
        k = int(np.argmin(np.abs(np.asarray(sc.animation.times) - (1.5 + sg["t_peak_s"]))))
        stills.append(r["frames"][min(k, len(r["frames"]) - 1)])
        labels.append(LABELS[name])
        print(name, "same gust", r["mp4"], flush=True)
    if mp4s:
        cb = ("Vertical gust w [m/s] (up +), wisps", -7.62, 7.62, "coolwarm")
        dst = OUT if a.keep_in_out else CLIPS
        compose_grid(stills, os.path.join(dst, "same_gust_grid.png"), labels=labels, colorbar=cb)
        g = compose_grid_video(mp4s, os.path.join(dst, "same_gust_grid.mp4"), labels=labels, colorbar=cb, crf=27)
        print("grid", g, flush=True)
