"""Render nodes for animated studio scenes: Blender Cycles clips and stills
(:func:`pinneapple_tools.visualization.studio.blender.render_animation` / ``render_still``) and a labelled grid of
clips with one colour bar (:mod:`pinneapple_tools.visualization.studio.compose`). They take the ``anim_scene`` of a
scene node (e.g. ``flight.scene``) and skip with the reason when bpy or ffmpeg is not installed."""
from __future__ import annotations

import os
import shutil
import time

import numpy as np

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node
from .examples import load_object


def _bpy_available():
    try:
        import bpy

        from pinneapple_tools.visualization.studio.blender import _check_bpy
        _check_bpy(bpy)
    except Exception as exc:  # noqa: BLE001 - bpy missing or too old
        return False, f"needs Blender's bpy >= 4.2 (pip install bpy): {type(exc).__name__}: {exc}"
    return True, ""


def _ffmpeg_available():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        return False, "needs ffmpeg and ffprobe on the PATH"
    return True, ""


PRESETS = {
    # the options of examples/flight_dynamics/05_render_clips.py per kind of flight
    "takeoff": dict(camera="chase_low", motion_blur=True, dof=5.6, trails={"alpha": 0.8}, hud_keys=["V", "h", "nz"]),
    "turbulence": dict(camera="chase", dof=8.0, trails={"alpha": 0.7}, hud_keys=["V", "nz", "phi"], sparkline="nz"),
    "gust": dict(camera="chase", dof=8.0, trails={"alpha": 0.7}, hud_keys=["V", "nz", "theta"], sparkline="nz",
                 colorbars=False),
}


def _preset(key: str, name: str, haze: float) -> dict:
    if name == "auto":
        name = "gust" if "gust_same" in key else "turbulence" if "turbulence" in key else "takeoff"
    p = dict(PRESETS[name])
    if name == "turbulence" and haze > 0:
        p["haze"] = haze
    return p


@node("render.blender_clip", "Blender clip (Cycles)", category="Visualization",
      inputs=[Port("scene", "anim_scene")],
      outputs=[Port("clip", "clip"), Port("figure", "figure", required=False), Port("metrics", "metrics")],
      params=[Param("preset", "enum", "auto", choices=["auto", "takeoff", "turbulence", "gust"],
                    help="camera, depth of field, trails and HUD of 05_render_clips.py for this kind of flight"),
              Param("frames", "int", 72, minimum=2, maximum=2000),
              Param("width", "int", 720, minimum=64, maximum=3840),
              Param("height", "int", 405, minimum=64, maximum=2160),
              Param("samples", "int", 12, minimum=1, maximum=4096, help="Cycles samples per pixel"),
              Param("fps", "int", 12, minimum=1, maximum=60),
              Param("haze", "float", 0.8, minimum=0.0, maximum=5.0, help="volumetric haze of turbulence clips"),
              Param("still", "bool", True, help="also a still (at the scene's marked time, else 2/3 of the way)"),
              Param("still_width", "int", 1920, minimum=64, maximum=3840),
              Param("still_height", "int", 1080, minimum=64, maximum=2160),
              Param("still_samples", "int", 48, minimum=1, maximum=4096),
              Param("only", "list", [], help="render only these scene keys; empty: all")],
      calls="pinneapple_tools.visualization.studio.blender.render_animation + render_still",
      available=_bpy_available)
def blender_clip(ctx, inputs, params):
    """Cycles (CPU, denoised) clip of each scene with the HUD burnt in, encoded to MP4, plus a still."""
    from pinneapple_tools.visualization.studio.blender import render_animation, render_still
    scenes = load_object(inputs["scene"])
    if params["only"]:
        scenes = [s for s in scenes if s["key"] in params["only"]]
    if not scenes:
        raise StepFailed("no scenes to render (check 'only')")
    size = (int(params["width"]), int(params["height"]))
    clips, t0 = [], time.perf_counter()
    for e in scenes:
        ctx.check_stop()
        key, sc0 = e["key"], e["scene"]
        opts = _preset(key, params["preset"], float(params["haze"]))
        ctx.log(f"render {key}: {opts}")
        out = ctx.path(f"frames_{key}")
        r = render_animation(sc0, str(out), frames=int(params["frames"]), size=size, samples=int(params["samples"]),
                             fps=int(params["fps"]), mp4=True, name=key, log=ctx.log, **opts)
        frames = list(r.get("frames") or [])
        # the scene was resampled to `frames` frames: the marked time (35 ft screen, gust peak) on that clock
        times = np.asarray(sc0.animation.times, float)
        tk = np.linspace(times[0], times[-1], len(frames)) if frames else times
        mark = e.get("t_mark")
        k = int(np.argmin(np.abs(tk - mark))) if mark is not None else int(len(tk) * 2 // 3)
        still = None
        if params["still"]:
            kk = int(np.argmin(np.abs(times - tk[min(k, len(tk) - 1)])))
            still = render_still(sc0, str(ctx.path(f"{key}_still.png")), kk,
                                 size=(int(params["still_width"]), int(params["still_height"])),
                                 samples=int(params["still_samples"]), **opts)
            ctx.report["figures"].append({"path": still, "name": f"{key}_still",
                                          "caption": f"Blender Cycles still: {e.get('title') or key}"})
        clips.append({"key": key, "mp4": r.get("mp4"), "still": still,
                      "mark_frame": frames[min(k, len(frames) - 1)] if frames else None, "title": e.get("title")})
    secs = time.perf_counter() - t0
    ctx.metric("render_seconds", round(secs, 1), f"Cycles on this machine's CPU ({os.cpu_count()} cores)")
    ctx.metric("clips", len(clips))
    first = next((c for c in clips if c["mp4"]), None)
    if first:
        ctx.view("video", {"path": str(first["mp4"]), "poster": str(first["still"] or first["mark_frame"] or ""),
                           "caption": f"{first['title'] or first['key']}: {params['frames']} frames, "
                                      f"{size[0]}x{size[1]}, {params['samples']} samples"})
    ctx.view("videos", [{"path": str(c["mp4"]), "poster": str(c["still"] or ""), "caption": c["title"] or c["key"]}
                        for c in clips if c["mp4"]])
    ctx.summary(f"{len(clips)} clip(s) in {secs:.0f} s")
    a = ctx.save_json("clips", clips)
    out = {"clip": a, "metrics": ctx.report["metrics"]}
    if clips and clips[0]["still"]:
        from ..runtime import artifact
        out["figure"] = artifact(clips[0]["still"], "png")
    return out


@node("render.clip_grid", "Clip grid with one colour bar", category="Visualization",
      inputs=[Port("clip", "clip", multiple=True)],
      outputs=[Port("figure", "figure"), Port("clip", "clip")],
      params=[Param("ncols", "int", 2, minimum=1, maximum=8),
              Param("colorbar_title", "str", "Vertical gust w [m/s] (up +), wisps"),
              Param("colorbar_min", "float", -7.62), Param("colorbar_max", "float", 7.62),
              Param("cmap", "enum", "coolwarm", choices=["coolwarm", "jet", "viridis", "turbo"]),
              Param("labels", "dict", {}, help="label per scene key (default: the scene title)"),
              Param("crf", "int", 27, minimum=10, maximum=40)],
      calls="pinneapple_tools.visualization.studio.compose.compose_grid + compose_grid_video",
      available=_ffmpeg_available)
def clip_grid(ctx, inputs, params):
    """The clips side by side (e.g. the four aircraft through the same gust) with labels and one colour bar; the
    still grid uses each clip's marked frame (the gust peak)."""
    from pinneapple_tools.visualization.studio.compose import compose_grid, compose_grid_video

    from ..runtime import load_artifact
    clips = [c for v in inputs["clip"] for c in load_artifact(v)]
    clips = [c for c in clips if c.get("mp4")]
    if not clips:
        raise StepFailed("no clips")
    labels = [params["labels"].get(c["key"], c.get("title") or c["key"]) for c in clips]
    cb = (params["colorbar_title"], float(params["colorbar_min"]), float(params["colorbar_max"]), params["cmap"])
    stills = [c.get("mark_frame") or c.get("still") for c in clips]
    png = compose_grid(stills, str(ctx.path("grid.png")), labels=labels, ncols=int(params["ncols"]), colorbar=cb)
    g = compose_grid_video([c["mp4"] for c in clips], str(ctx.path("grid.mp4")), labels=labels,
                           ncols=int(params["ncols"]), colorbar=cb, crf=int(params["crf"]))
    ctx.report["figures"].append({"path": png, "name": "grid", "caption": "the clips at their marked frame"})
    ctx.view("video", {"path": str(g["mp4"]), "poster": png, "caption": " | ".join(labels)})
    from ..runtime import artifact
    return {"figure": artifact(png, "png"),
            "clip": ctx.save_json("grid", [{"key": "grid", "mp4": g["mp4"], "still": png, "title": "grid"}])}
